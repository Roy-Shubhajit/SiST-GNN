import os
import urllib.request
import gzip
import tarfile
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict
from datetime import datetime

DATA_URLS = {
    "bitcoin-alpha": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
    "bitcoin-otc": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
    "uci-message": "https://snap.stanford.edu/data/CollegeMsg.txt.gz",
    "reddit-body": "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv",
    "reddit-title": "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv",
    "as-733": "https://snap.stanford.edu/data/as-733.tar.gz"
}

def download_and_extract(dataset_name: str, data_root: str = "datasets/roland") -> str:
    os.makedirs(data_root, exist_ok=True)
    if dataset_name in ["bsi-zk", "bsi-svt"]:
        # Proprietary datasets: expect them in data_root/dataset_name/
        extracted_path = os.path.join(data_root, dataset_name)
        if not os.path.exists(extracted_path):
            os.makedirs(extracted_path, exist_ok=True)
            print(f"Warning: {dataset_name} is a proprietary dataset.")
            print(f"Please ensure its .tsv files are placed directly in {extracted_path}.")
        return extracted_path

    if dataset_name not in DATA_URLS:
        raise ValueError(f"Dataset {dataset_name} not found in public URLs.")
    
    url = DATA_URLS[dataset_name]
    filename = os.path.basename(url)
    filepath = os.path.join(data_root, filename)
    extracted_path = os.path.join(data_root, dataset_name)
    os.makedirs(extracted_path, exist_ok=True)
    
    if not os.path.exists(filepath) and not any(os.scandir(extracted_path)):
        print(f"Downloading {dataset_name} from {url}...")
        urllib.request.urlretrieve(url, filepath)
        
        # Extract files if necessary
        if filename.endswith(".gz") and not filename.endswith(".tar.gz"):
            unzipped_file = os.path.join(extracted_path, filename[:-3])
            with gzip.open(filepath, 'rb') as f_in, open(unzipped_file, 'wb') as f_out:
                f_out.write(f_in.read())
        elif filename.endswith(".tar.gz"):
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=extracted_path)
        elif filename.endswith(".tsv"):
            import shutil
            shutil.copy(filepath, os.path.join(extracted_path, filename))
    
    return extracted_path

# ROLAND's own settings for the three fixed-split datasets, taken from
# run/replication_configs/table2/*.yaml and graphgym/contrib/loader/roland_*.py.
#   snapshot_freq  - split_by_seconds(edge_time // freq_sec), NOT calendar weeks
#   undirected     - roland_btc.py appends reversed edges when
#                    cfg.train.mode == 'live_update_fixed_split' (the mode those
#                    configs use). roland_ucimsg.py does not.
#   start_compute_mrr - MRR is only reported from this snapshot onwards.
ROLAND_SPEC = {
    "bitcoin-alpha": {"freq_sec": 1_200_000, "undirected": True,  "start_compute_mrr": 109},
    "bitcoin-otc":   {"freq_sec": 1_200_000, "undirected": True,  "start_compute_mrr": 110},
    "uci-message":   {"freq_sec":   190_080, "undirected": False, "start_compute_mrr": 72},
}


def process_snapshots_seconds(df: pd.DataFrame, src_col: str, dst_col: str,
                              time_col: str, freq_sec: int,
                              undirected: bool = False) -> List[Data]:
    """ROLAND's ``split_by_seconds``: bucket edges by ``edge_time // freq_sec``.

    Unlike calendar-frequency grouping this produces fixed-width buckets aligned
    to the Unix epoch, which is what ROLAND's configs use for these datasets.
    ``undirected`` appends the reversed edge set first, mirroring
    ``roland_btc.py``'s augmentation under ``live_update_fixed_split``.
    """
    df = df.sort_values(by=time_col).reset_index(drop=True)

    all_nodes = pd.concat([df[src_col], df[dst_col]]).unique()
    node_map = {nid: i for i, nid in enumerate(all_nodes)}
    num_nodes = len(node_map)

    src = df[src_col].map(node_map).to_numpy()
    dst = df[dst_col].map(node_map).to_numpy()
    t = df[time_col].to_numpy().astype(np.int64)

    if undirected:
        src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
        t = np.concatenate([t, t])

    lo, hi = t.min(), t.max()
    scaled = (t - lo) / (hi - lo) * 2.0 if hi > lo else np.zeros_like(t, dtype=float)

    x_dummy = torch.ones((num_nodes, 1), dtype=torch.float)
    bucket = t // freq_sec

    snapshots: List[Data] = []
    for b in np.unique(np.sort(bucket)):
        m = bucket == b
        edge_index = torch.tensor(np.stack([src[m], dst[m]]), dtype=torch.long)
        edge_attr = torch.tensor(scaled[m], dtype=torch.float).unsqueeze(1)
        snapshots.append(Data(x=x_dummy, edge_index=edge_index,
                              edge_attr=edge_attr, num_nodes=num_nodes))
    return snapshots


def process_snapshots(df: pd.DataFrame, src_col: str, dst_col: str, time_col: str, freq: str = "W", time_unit: str = 's') -> List[Data]:
    """
    Groups dataframe by a temporal frequency and returns a list of PyG Data objects (snapshots).
    Adds dummy node features (all 1s) to avoid node feature issues, as per Roland.
    freq: Pandas frequency string, e.g., 'W' for weekly, 'D' for daily.
    """
    df = df.sort_values(by=time_col).reset_index(drop=True)
    
    # Map node IDs to continuous 0-indexed integers across the entire dataset to track evolving graph
    all_nodes = pd.concat([df[src_col], df[dst_col]]).unique()
    node_map = {node_id: idx for idx, node_id in enumerate(all_nodes)}
    num_nodes = len(node_map)
    
    df['src_mapped'] = df[src_col].map(node_map)
    df['dst_mapped'] = df[dst_col].map(node_map)
    
    if pd.api.types.is_numeric_dtype(df[time_col]):
        # Assuming Unix timestamp
        df['datetime'] = pd.to_datetime(df[time_col], unit=time_unit)
    else:
        df['datetime'] = pd.to_datetime(df[time_col])

    # Time normalization for edge features
    time_nums = df['datetime'].astype('int64')
    min_time = time_nums.min()
    max_time = time_nums.max()
    if max_time > min_time:
        df['time_scaled'] = (time_nums - min_time) / (max_time - min_time) * 2.0
    else:
        df['time_scaled'] = 0.0
        
    df.set_index('datetime', inplace=True)
    
    snapshots = []
    # Pre-allocate dummy features for all unseen + seen nodes so dimensions safely match between runs
    x_dummy = torch.ones((num_nodes, 1), dtype=torch.float)
    
    for period, group in df.groupby(pd.Grouper(freq=freq)):
        if len(group) == 0:
            continue
            
        edge_index = torch.tensor(np.array([group['src_mapped'].values, group['dst_mapped'].values]), dtype=torch.long)
        edge_attr = torch.tensor(group['time_scaled'].values, dtype=torch.float).unsqueeze(1)
        
        snapshot_data = Data(x=x_dummy, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        snapshots.append(snapshot_data)
        
    return snapshots

def load_lp_dataset(dataset_name: str, data_root: str = "datasets/roland",
                    snapshot_mode: str = "paper") -> List[Data]:
    """
    Loads and processes a specific dataset into discrete temporal snapshots.

    ``snapshot_mode``:
      * ``"paper"``  - weekly calendar bins, directed (what the submission used).
      * ``"roland"`` - ROLAND's own construction for that dataset (see
        :data:`ROLAND_SPEC`): fixed-width second buckets, plus the reversed-edge
        augmentation for the bitcoin datasets. Falls back to weekly bins for
        datasets ROLAND's table-2 configs do not cover.
    """
    print(f"Loading {dataset_name}...")
    extracted_path = download_and_extract(dataset_name, data_root)

    spec = ROLAND_SPEC.get(dataset_name) if snapshot_mode == "roland" else None
    if snapshot_mode == "roland" and spec is None:
        print(f"  (no ROLAND table-2 spec for {dataset_name}; using weekly bins)")

    CSV = {
        "bitcoin-alpha": ("soc-sign-bitcoinalpha.csv", ",", ['src', 'dst', 'rating', 'time']),
        "bitcoin-otc":   ("soc-sign-bitcoinotc.csv",   ",", ['src', 'dst', 'rating', 'time']),
        "uci-message":   ("CollegeMsg.txt",            " ", ['src', 'dst', 'time']),
    }

    snapshots = []
    if dataset_name in CSV:
        fname, sep, names = CSV[dataset_name]
        df = pd.read_csv(os.path.join(extracted_path, fname), sep=sep, names=names)
        if spec is not None:
            print(f"  ROLAND construction: freq={spec['freq_sec']}s "
                  f"undirected={spec['undirected']}")
            snapshots = process_snapshots_seconds(
                df, 'src', 'dst', 'time',
                freq_sec=spec['freq_sec'], undirected=spec['undirected'])
        else:
            snapshots = process_snapshots(df, 'src', 'dst', 'time', freq="W")

    elif dataset_name == "reddit-body":
        df = pd.read_csv(os.path.join(extracted_path, "soc-redditHyperlinks-body.tsv"), sep='\t')
        snapshots = process_snapshots(df, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP', freq="W")

    elif dataset_name == "reddit-title":
        df = pd.read_csv(os.path.join(extracted_path, "soc-redditHyperlinks-title.tsv"), sep='\t')
        snapshots = process_snapshots(df, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP', freq="W")
        
    elif dataset_name == "as-733":
        # AS-733 consists of multiple txt files. We read them sequentially.
        as_dir = os.path.join(extracted_path, "as-733")
        if not os.path.exists(as_dir):
            # Fallback if tar extracts directly into extracted_path
            as_dir = extracted_path
            
        files = sorted([f for f in os.listdir(as_dir) if f.endswith('.txt')])
        
        all_dfs = []
        for time_idx, f in enumerate(files):
            df = pd.read_csv(os.path.join(as_dir, f), sep='\t', comment='#', names=['src', 'dst'])
            df['time'] = time_idx # Treat sequential files as distinct time steps
            all_dfs.append(df)
            
        combined_df = pd.concat(all_dfs)
        snapshots = process_snapshots(combined_df, 'src', 'dst', 'time', freq="D", time_unit="D") # 1 per day essentially

    elif dataset_name in ["bsi-zk", "bsi-svt"]:
        # Proprietary Bank datasets: expected to have Payer, Payee, Timestamp columns.
        import glob
        files = glob.glob(os.path.join(extracted_path, "*.tsv"))
        if not files:
            raise FileNotFoundError(f"No .tsv files found in {extracted_path}. Please place {dataset_name} files there.")
        
        all_dfs = []
        for f in files:
            df = pd.read_csv(f, sep='\t')
            all_dfs.append(df)
            
        combined_df = pd.concat(all_dfs)
        snapshots = process_snapshots(combined_df, 'Payer', 'Payee', 'Timestamp', freq="D", time_unit="s")

    else:
        raise NotImplementedError(f"Pre-processing for {dataset_name} is not implemented yet.")
        
    print(f"Processed {dataset_name} into {len(snapshots)} temporal snapshots.")
    return snapshots

if __name__ == "__main__":
    # Quick test
    snaps = load_lp_dataset("uci-message")
    print(snaps[0])
