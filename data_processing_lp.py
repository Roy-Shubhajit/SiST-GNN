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

def load_lp_dataset(dataset_name: str, data_root: str = "datasets/roland") -> List[Data]:
    """
    Loads and processes a specific dataset into discrete temporal snapshots.
    """
    print(f"Loading {dataset_name}...")
    extracted_path = download_and_extract(dataset_name, data_root)
    
    snapshots = []
    if dataset_name == "bitcoin-alpha":
        df = pd.read_csv(os.path.join(extracted_path, "soc-sign-bitcoinalpha.csv"), names=['src', 'dst', 'rating', 'time'])
        snapshots = process_snapshots(df, 'src', 'dst', 'time', freq="W")

    elif dataset_name == "bitcoin-otc":
        df = pd.read_csv(os.path.join(extracted_path, "soc-sign-bitcoinotc.csv"), names=['src', 'dst', 'rating', 'time'])
        snapshots = process_snapshots(df, 'src', 'dst', 'time', freq="W")

    elif dataset_name == "uci-message":
        df = pd.read_csv(os.path.join(extracted_path, "CollegeMsg.txt"), sep=' ', names=['src', 'dst', 'time'])
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
