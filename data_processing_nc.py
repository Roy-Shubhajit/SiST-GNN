"""
Dynamic Node Classification data pipeline (Wikipedia / Reddit via benchtemp).

Pipeline:
    1. Download raw JODIE CSV (wikipedia / reddit) into ``data_root``.
    2. Run benchtemp ``DataPreprocessor`` to produce
       ``ml_<name>.csv``, ``ml_<name>.npy`` and ``ml_<name>_node.npy``.
    3. Load via ``benchtemp.nc.DataLoader`` and bucket interactions into
       discrete temporal snapshots (compatible with SiST-GNN).

Returned snapshots are PyG ``Data`` objects with extra fields:
    - ``edge_label``       : per-edge binary label
    - ``edge_split``       : 0=train / 1=val / 2=test
    - ``edge_feat``        : per-edge raw features from JODIE
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

JODIE_URLS = {
    "wikipedia": "http://snap.stanford.edu/jodie/wikipedia.csv",
    "reddit": "http://snap.stanford.edu/jodie/reddit.csv",
    "mooc": "http://snap.stanford.edu/jodie/mooc.csv",
}


@dataclass
class NCDatasetBundle:
    snapshots: List[Data]
    num_nodes: int
    node_feat_dim: int
    edge_feat_dim: int
    node_features: torch.Tensor      # [num_nodes, node_feat_dim]
    train_end_snap: int              # last snapshot index that contains only train edges
    val_end_snap: int                # last snapshot index that contains only train+val edges


def _download_raw(dataset_name: str, data_root: str) -> str:
    """Download the JODIE raw CSV into ``data_root/<name>.csv`` (if missing)."""
    os.makedirs(data_root, exist_ok=True)
    raw_path = os.path.join(data_root, f"{dataset_name}.csv")
    if not os.path.exists(raw_path):
        url = JODIE_URLS[dataset_name]
        print(f"[NC] Downloading {dataset_name} from {url} ...")
        urllib.request.urlretrieve(url, raw_path)
    return raw_path


def _ensure_preprocessed(dataset_name: str, data_root: str) -> None:
    """Run benchtemp's preprocessing if the ml_* artifacts are missing."""
    needed = [
        os.path.join(data_root, f"ml_{dataset_name}.csv"),
        os.path.join(data_root, f"ml_{dataset_name}.npy"),
        os.path.join(data_root, f"ml_{dataset_name}_node.npy"),
    ]
    if all(os.path.exists(p) for p in needed):
        return

    _download_raw(dataset_name, data_root)

    from benchtemp.preprocess.preprocessing import DataPreprocessor

    # The benchtemp preprocessor expects a trailing slash and writes back to
    # the same folder. Wikipedia/Reddit are bipartite (user → item).
    print(f"[NC] Preprocessing {dataset_name} with benchtemp ...")
    cwd = os.getcwd()
    try:
        # benchtemp writes ml_<name>_node.npy to "./data/..." (hard-coded), so
        # chdir into data_root's parent so the relative path lands in our folder.
        os.makedirs("data", exist_ok=True)
        pre = DataPreprocessor(data_path=data_root.rstrip("/") + "/", data_name=dataset_name)
        pre.data_preprocess(bipartite=True)
    finally:
        os.chdir(cwd)

    # benchtemp hard-codes the node-feature save path to "./data/ml_<name>_node.npy".
    # Move it to data_root if it landed elsewhere.
    stray = os.path.join("data", f"ml_{dataset_name}_node.npy")
    target = os.path.join(data_root, f"ml_{dataset_name}_node.npy")
    if os.path.exists(stray) and not os.path.exists(target):
        os.replace(stray, target)


def _bucket_snapshots(
    sources: np.ndarray,
    destinations: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
    edge_idxs: np.ndarray,
    splits: np.ndarray,           # 0/1/2 per interaction
    edge_features: np.ndarray,    # [n_edges + 1, edge_feat_dim]
    bucket_seconds: float,        # temporal width of one snapshot, in seconds
    num_nodes: int,
) -> Tuple[List[Data], int, int]:
    """
    Bucket interactions into snapshots of width ``bucket_seconds``.

    JODIE timestamps are seconds-since-start, so a snapshot is a real time
    window (e.g. 6 hours). The total number of snapshots is determined by the
    dataset's time span, not a user-chosen count.

    Returns
    -------
    snapshots         : list of PyG Data objects (one per non-empty bucket)
    train_end_snap    : index after the last bucket containing only train edges
    val_end_snap      : index after the last bucket containing only train+val edges
    """
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be positive")

    t0 = float(timestamps.min())
    # Floor-divide each timestamp into its bucket; cast to int64 so we can
    # iterate buckets safely.
    bucket_id = ((timestamps - t0) // bucket_seconds).astype(np.int64)
    num_snapshots = int(bucket_id.max()) + 1

    # x is dummy (we use a learned node-embedding inside the wrapper), but we
    # still need an x tensor so PyG accepts the Data object.
    x_dummy = torch.ones((num_nodes, 1), dtype=torch.float)

    snapshots: List[Data] = []
    last_split_per_bucket: List[int] = []

    for b in range(num_snapshots):
        mask = bucket_id == b
        if not mask.any():
            continue

        src_b = torch.tensor(sources[mask], dtype=torch.long)
        dst_b = torch.tensor(destinations[mask], dtype=torch.long)
        edge_index = torch.stack([src_b, dst_b], dim=0)

        # Per-edge features lookup (edge_idxs are 1-indexed in benchtemp).
        feat_b = edge_features[edge_idxs[mask]]
        edge_feat = torch.tensor(feat_b, dtype=torch.float)

        edge_label = torch.tensor(labels[mask], dtype=torch.float)
        edge_split = torch.tensor(splits[mask], dtype=torch.long)

        data = Data(
            x=x_dummy,
            edge_index=edge_index,
            edge_attr=edge_feat,         # alias for compatibility with existing layer code
            num_nodes=num_nodes,
        )
        data.edge_label = edge_label
        data.edge_split = edge_split
        data.edge_feat = edge_feat
        snapshots.append(data)
        last_split_per_bucket.append(int(splits[mask].max()))

    # Determine snapshot-level cut points: train_end is one past the last
    # snapshot whose maximum split is 0; val_end one past the last with split <= 1.
    train_end = 0
    val_end = 0
    for i, s in enumerate(last_split_per_bucket):
        if s == 0:
            train_end = i + 1
        if s <= 1:
            val_end = i + 1

    return snapshots, train_end, val_end


def load_nc_dataset(
    dataset_name: str,
    data_root: str = "datasets/nc",
    bucket_hours: float = 6.0,
) -> NCDatasetBundle:
    """
    Load Wikipedia or Reddit for dynamic node classification.

    Parameters
    ----------
    dataset_name : "wikipedia" or "reddit".
    data_root    : Folder for raw + preprocessed files.
    bucket_hours : Snapshot width in hours. JODIE datasets are continuous-time
                   interaction streams with no inherent snapshots, so we
                   impose a time-based discretization; the total number of
                   snapshots is then determined by the dataset's temporal span.
    """
    if dataset_name not in JODIE_URLS:
        raise ValueError(
            f"Unknown NC dataset '{dataset_name}'. "
            f"Supported: {list(JODIE_URLS.keys())}"
        )

    _ensure_preprocessed(dataset_name, data_root)

    import benchtemp as bt

    loader = bt.nc.DataLoader(
        dataset_path=data_root.rstrip("/") + "/",
        dataset_name=dataset_name,
        use_validation=True,
    )
    full_data, node_features, edge_features, train_data, val_data, test_data = loader.load()

    # Build a global split vector aligned with full_data ordering.
    val_time = float(np.quantile(full_data.timestamps, 0.70))
    test_time = float(np.quantile(full_data.timestamps, 0.85))
    splits = np.zeros(len(full_data.timestamps), dtype=np.int64)
    splits[(full_data.timestamps > val_time) & (full_data.timestamps <= test_time)] = 1
    splits[full_data.timestamps > test_time] = 2

    num_nodes = int(max(full_data.sources.max(), full_data.destinations.max())) + 1
    node_feat_dim = int(node_features.shape[1])
    edge_feat_dim = int(edge_features.shape[1])

    bucket_seconds = float(bucket_hours) * 3600.0
    snapshots, train_end_snap, val_end_snap = _bucket_snapshots(
        sources=full_data.sources,
        destinations=full_data.destinations,
        timestamps=full_data.timestamps,
        labels=full_data.labels,
        edge_idxs=full_data.edge_idxs,
        splits=splits,
        edge_features=edge_features,
        bucket_seconds=bucket_seconds,
        num_nodes=num_nodes,
    )

    pos_rate = float(full_data.labels.sum() / max(len(full_data.labels), 1))
    t_span_days = float(full_data.timestamps.max() - full_data.timestamps.min()) / 86400.0
    print(
        f"[NC] {dataset_name}: bucket={bucket_hours}h → {len(snapshots)} non-empty snapshots "
        f"over {t_span_days:.1f} days, "
        f"{num_nodes} nodes, {len(full_data.timestamps)} interactions, "
        f"pos-rate={pos_rate:.4f}, "
        f"train_end_snap={train_end_snap}, val_end_snap={val_end_snap}"
    )

    return NCDatasetBundle(
        snapshots=snapshots,
        num_nodes=num_nodes,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        node_features=torch.tensor(node_features, dtype=torch.float),
        train_end_snap=train_end_snap,
        val_end_snap=val_end_snap,
    )


if __name__ == "__main__":
    bundle = load_nc_dataset("wikipedia", bucket_hours=6.0)
    print(bundle.snapshots[0])
