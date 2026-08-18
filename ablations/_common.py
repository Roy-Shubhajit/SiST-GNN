"""Shared utilities for the SiST-GNN link-prediction ablation suite.

Every LP ablation script imports `build_model`, `run_one`, and `save_csv` from
this module so the per-experiment files stay short and the training loop is
identical to `main_lp.py` modulo a single hyper-parameter override.
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

from data_processing_lp import load_lp_dataset                            # noqa: E402
from helper import seed_everything                                         # noqa: E402
from temporal_gnn import TemporalGNNModel                                  # noqa: E402
from train_lp import (                                                     # noqa: E402
    StatefulGNNWrapper,
    train_fixed_split,
    train_live_update,
)

ABLATIONS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR   = os.path.join(ABLATIONS_DIR, "results")
FIGURES_DIR   = os.path.join(ABLATIONS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


@dataclass
class RunConfig:
    """One training configuration. Defaults match the paper's Implementation
    subsection."""
    dataset:        str
    hidden_dim:     int   = 128
    num_layers:     int   = 2
    gnn_type:       str   = "GCNConv"
    eval_method:    str   = "live-update"     # "live-update" | "fixed-split"
    num_epochs:     int   = 100
    lr:             float = 1e-3
    use_node_emb:   bool  = True
    seed:           int   = 0
    device:         str   = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(cfg: RunConfig, num_nodes: int) -> StatefulGNNWrapper:
    """Construct the SiST-GNN wrapper with the configuration's hyper-params."""
    seed_everything(cfg.seed)
    base = TemporalGNNModel(
        input_dim  = cfg.hidden_dim,
        hidden_dim = cfg.hidden_dim,
        output_dim = cfg.hidden_dim,
        num_layers = cfg.num_layers,
        gnn_type   = cfg.gnn_type,
    )
    wrapper = StatefulGNNWrapper(
        base, num_nodes, torch.device(cfg.device),
        use_node_emb=cfg.use_node_emb, node_emb_dim=cfg.hidden_dim,
    ).to(cfg.device)
    return wrapper


def run_one(cfg: RunConfig) -> Dict[str, Any]:
    """Train + evaluate for a single configuration. Returns a CSV-flat dict."""
    snapshots = load_lp_dataset(cfg.dataset)
    num_nodes = snapshots[0].num_nodes
    model = build_model(cfg, num_nodes)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if cfg.eval_method == "fixed-split":
        res = train_fixed_split(snapshots, model, optimiser, num_epochs=cfg.num_epochs)
    else:
        res = train_live_update(snapshots, model, optimiser,
                                epochs_per_snapshot=cfg.num_epochs)

    del model, optimiser
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset":     cfg.dataset,
        "hidden_dim":  cfg.hidden_dim,
        "num_layers":  cfg.num_layers,
        "gnn_type":    cfg.gnn_type,
        "eval_method": cfg.eval_method,
        "seed":        cfg.seed,
        "mrr":         float(res["mrr"]),
        "recall@1":    float(res["recall@1"]),
        "recall@10":   float(res["recall@10"]),
    }


def save_csv(rows: List[Dict[str, Any]], filename: str) -> str:
    """Write `rows` to `<RESULTS_DIR>/<filename>` and return the path."""
    if not rows:
        raise ValueError("save_csv: rows is empty")
    path = os.path.join(RESULTS_DIR, filename)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ablations] wrote {path} ({len(rows)} rows)")
    return path


def parse_dataset_arg(default: List[str]) -> List[str]:
    """If the user passes `--datasets a,b,c` we use that, otherwise `default`."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=",".join(default))
    args, _ = parser.parse_known_args()
    return [s for s in args.datasets.split(",") if s]
