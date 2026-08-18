"""Shared utilities for the SiST-GNN node-classification ablation suite.

Mirrors the role of ``_common.py`` for the link-prediction ablations, but
wires up the NC pipeline:

    data_processing_nc.load_nc_dataset(...)  →  NCDatasetBundle
    NCStatefulWrapper(TemporalGNNModel(...)) →  binary node-classification head
    train_nc_fixed_split(...)                →  (val_auc, test_auc)

Every NC ablation script imports ``RunConfig``, ``run_one``, ``save_csv`` and
``parse_dataset_arg`` so each per-experiment file stays short and the training
loop is identical to ``main_nc.py`` modulo a single hyper-parameter override.
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

# Make the project root importable so we can reuse the existing codebase.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_processing_nc import load_nc_dataset                            # noqa: E402
from helper import seed_everything                                        # noqa: E402
from temporal_gnn import TemporalGNNModel, TemporalTransformerGNNModel    # noqa: E402
from train_nc import NCStatefulWrapper, train_nc_fixed_split              # noqa: E402

ABLATIONS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR   = os.path.join(ABLATIONS_DIR, "results")
FIGURES_DIR   = os.path.join(ABLATIONS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEFAULT_NC_DATASETS = ["wikipedia", "reddit", "mooc"]


# Tiny in-memory dataset cache so a sweep over (e.g.) 4 hidden_dim values does
# not preprocess + load the same dataset 4 times. Keyed by (dataset, bucket_hours).
_DATASET_CACHE: Dict[tuple, Any] = {}


def _load_bundle(dataset: str, bucket_hours: float, data_root: str):
    key = (dataset, float(bucket_hours), data_root)
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = load_nc_dataset(
            dataset, data_root=data_root, bucket_hours=bucket_hours
        )
    return _DATASET_CACHE[key]


@dataclass
class RunConfig:
    """One NC training configuration. Defaults match ``main_nc.py``."""
    dataset:           str
    hidden_dim:        int   = 128
    num_layers:        int   = 2
    gnn_type:          str   = "GCNConv"
    model:             str   = "lstm"          # "lstm" | "transformer"
    bucket_hours:      float = 6.0
    pos_weight:        str   = "none"          # "none" | "balanced" | "sqrt" | float-string
    lr:                float = 1e-3
    num_epochs:        int   = 100
    patience:          int   = 5
    use_node_emb:      bool  = True
    use_edge_emb:      bool  = True
    classifier_hidden: int   = 128             # main_nc.py uses hidden_dim here
    seed:              int   = 0
    data_root:         str   = "datasets/nc"
    device:            str   = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(cfg: RunConfig, bundle) -> NCStatefulWrapper:
    """Construct the SiST-GNN NC wrapper with the configuration's hyper-params."""
    seed_everything(cfg.seed)

    if cfg.model == "transformer":
        base = TemporalTransformerGNNModel(
            input_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            gnn_type=cfg.gnn_type,
        )
    else:
        base = TemporalGNNModel(
            input_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            gnn_type=cfg.gnn_type,
        )

    wrapper = NCStatefulWrapper(
        base_model=base,
        num_nodes=bundle.num_nodes,
        hidden_dim=cfg.hidden_dim,
        node_features=bundle.node_features,
        edge_feat_dim=bundle.edge_feat_dim,
        device=torch.device(cfg.device),
        use_node_emb=cfg.use_node_emb,
        use_edge_emb=cfg.use_edge_emb,
        classifier_hidden=cfg.classifier_hidden,
    ).to(cfg.device)
    return wrapper


def run_one(cfg: RunConfig) -> Dict[str, Any]:
    """Train + evaluate one NC configuration. Returns a CSV-flat dict."""
    bundle = _load_bundle(cfg.dataset, cfg.bucket_hours, cfg.data_root)
    model = build_model(cfg, bundle)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    metrics = train_nc_fixed_split(
        snapshots=bundle.snapshots,
        model=model,
        optimizer=optimiser,
        train_end=bundle.train_end_snap,
        val_end=bundle.val_end_snap,
        num_epochs=cfg.num_epochs,
        patience=cfg.patience,
        pos_weight=cfg.pos_weight,
        verbose=False,
    )

    # Free GPU memory between runs (sweeps can blow up otherwise).
    del model, optimiser
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset":      cfg.dataset,
        "hidden_dim":   cfg.hidden_dim,
        "num_layers":   cfg.num_layers,
        "gnn_type":     cfg.gnn_type,
        "model":        cfg.model,
        "bucket_hours": cfg.bucket_hours,
        "pos_weight":   cfg.pos_weight,
        "seed":         cfg.seed,
        "val_auc":      float(metrics["val_auc"]),
        "test_auc":     float(metrics["test_auc"]),
    }


def save_csv(rows: List[Dict[str, Any]], filename: str) -> str:
    if not rows:
        raise ValueError("save_csv: rows is empty")
    path = os.path.join(RESULTS_DIR, filename)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[nc-ablations] wrote {path} ({len(rows)} rows)")
    return path


def parse_dataset_arg(default: List[str]) -> List[str]:
    """`--datasets a,b,c` overrides; otherwise use `default`."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=",".join(default))
    args, _ = parser.parse_known_args()
    return [s for s in args.datasets.split(",") if s]
