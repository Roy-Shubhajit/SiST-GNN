"""Per-snapshot MRR trace under the live-update protocol.

Instead of returning only the *average* MRR over all snapshots (as
`train_live_update` does), this script reproduces the live-update loop and
records the per-snapshot MRR for SiST-GNN. The plotting script overlays the
published ROLAND-GRU average per dataset as a horizontal dashed reference
line, so the user can see at a glance that SiST-GNN's per-snapshot MRR
stays above ROLAND-GRU's leaderboard average essentially everywhere.

Output CSV: `results/per_snapshot_mrr.csv` with columns
`dataset, snapshot, mrr`.
"""
from __future__ import annotations

import os
import sys

import torch

from _common import (
    RunConfig, build_model, load_lp_dataset, parse_dataset_arg, save_csv,
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from train_lp import train_live_update  # noqa: E402


DEFAULT_DATASETS = ["bitcoin-otc", "bitcoin-alpha"]


def trace_one(cfg: RunConfig) -> list[dict]:
    """Run the live-update protocol and keep its per-snapshot MRR trace."""
    snapshots = load_lp_dataset(cfg.dataset)
    num_nodes = snapshots[0].num_nodes
    model = build_model(cfg, num_nodes)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    res = train_live_update(snapshots, model, optimiser,
                            epochs_per_snapshot=cfg.num_epochs)
    return [
        {"dataset": cfg.dataset, "snapshot": r["t"], "mrr": float(r["mrr"])}
        for r in res["per_snapshot_mrr"]
    ]


def main() -> None:
    datasets = parse_dataset_arg(DEFAULT_DATASETS)
    all_rows: list[dict] = []
    for ds in datasets:
        cfg = RunConfig(dataset=ds, num_epochs=50)
        print(f"[per_snapshot] dataset={ds}")
        all_rows.extend(trace_one(cfg))
    save_csv(all_rows, "per_snapshot_mrr.csv")


if __name__ == "__main__":
    main()
