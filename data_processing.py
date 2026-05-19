from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.nodeproppred.dataset import NodePropPredDataset


LP_DATASETS = {"tgbl-wiki-v2", "tgbl-wiki", "tgbl-flight"}
NR_DATASETS = {"tgbn-trade", "tgbn-genre"}


@dataclass
class LPDataBundle:
    canonical_name: str
    dataset: LinkPropPredDataset
    full_data: Dict[str, np.ndarray]
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray


@dataclass
class NRDataBundle:
    canonical_name: str
    dataset: NodePropPredDataset
    full_data: Dict[str, np.ndarray]
    node_label_dict: Dict[int, Dict[int, np.ndarray]]
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray


def canonicalize_dataset_name(dataset_name: str) -> str:
    if dataset_name == "tgbl-wiki-v2":
        return "tgbl-wiki"
    return dataset_name


def _patch_tgbl_wiki_loader_readonly_bug() -> None:
    """
    py-tgb 2.2.0 may get read-only NumPy views from pandas for tgbl-wiki,
    causing `dst += ...` to fail in load_edgelist_wiki().
    This patch ensures writable arrays are used.
    """
    try:
        import tgb.linkproppred.dataset as lp_dataset_module
        import tgb.utils.pre_process as pp_module
    except Exception:
        return

    def _safe_load_edgelist_wiki(fname: str):
        df = pd.read_csv(fname, skiprows=1, header=None)
        src = df.iloc[:, 0].to_numpy(copy=True)
        dst = df.iloc[:, 1].to_numpy(copy=True)
        dst = dst + int(src.max()) + 1
        t = df.iloc[:, 2].to_numpy(copy=True)
        msg = df.iloc[:, 4:].to_numpy(copy=True)
        idx = np.arange(t.shape[0])
        w = np.ones(t.shape[0])
        return pd.DataFrame({"u": src, "i": dst, "ts": t, "idx": idx, "w": w}), msg, None

    pp_module.load_edgelist_wiki = _safe_load_edgelist_wiki
    lp_dataset_module.load_edgelist_wiki = _safe_load_edgelist_wiki


def load_lp_data(dataset_name: str, root: str = "datasets", download: bool = True) -> LPDataBundle:
    canonical = canonicalize_dataset_name(dataset_name)
    if canonical not in {"tgbl-wiki", "tgbl-flight"}:
        raise ValueError(f"Unsupported LP dataset: {dataset_name}")

    if canonical == "tgbl-wiki":
        _patch_tgbl_wiki_loader_readonly_bug()

    ds = LinkPropPredDataset(name=canonical, root=root, preprocess=True, download=download)

    # Use the dataset's own split logic so the masks match TGB's preprocessing.
    train_mask, val_mask, test_mask = ds.generate_splits(ds.full_data)
    
    return LPDataBundle(
        canonical_name=canonical,
        dataset=ds,
        full_data=ds.full_data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def load_nr_data(dataset_name: str, root: str = "datasets", download: bool = True) -> NRDataBundle:
    canonical = canonicalize_dataset_name(dataset_name)
    if canonical not in NR_DATASETS:
        raise ValueError(f"Unsupported NR dataset: {dataset_name}")

    ds = NodePropPredDataset(name=canonical, root=root, preprocess=True, download=download)

    # Use the dataset's own split logic so the masks match TGB's preprocessing.
    train_mask, val_mask, test_mask = ds.generate_splits(ds.full_data)
    
    return NRDataBundle(
        canonical_name=canonical,
        dataset=ds,
        full_data=ds.full_data,
        node_label_dict=ds.node_label_dict,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def split_time_boundaries(timestamps: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray) -> Tuple[int, int]:
    train_end = int(np.max(timestamps[train_mask]))
    val_end = int(np.max(timestamps[val_mask]))
    return train_end, val_end
