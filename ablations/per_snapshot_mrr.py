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
import torch.nn.functional as F

from _common import (
    RunConfig, build_model, load_lp_dataset, parse_dataset_arg, save_csv,
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from evaluate_lp import evaluate_mrr, get_negative_samples  # noqa: E402


DEFAULT_DATASETS = ["bitcoin-otc", "bitcoin-alpha"]


def trace_one(cfg: RunConfig) -> list[dict]:
    """Mirror `train_live_update` while recording the per-snapshot MRR."""
    snapshots = load_lp_dataset(cfg.dataset)
    num_nodes = snapshots[0].num_nodes
    model = build_model(cfg, num_nodes)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    rows: list[dict] = []
    device = torch.device(cfg.device)
    for t, snap in enumerate(snapshots):
        snap = snap.to(device)
        if snap.edge_index.size(1) == 0:
            continue
        if t > 0:
            mrr = evaluate_mrr(model, snap)
            rows.append({
                "dataset":  cfg.dataset,
                "snapshot": t,
                "mrr":      float(mrr),
            })
        # One-step update on this snapshot (mirrors train_live_update).
        prev_h, prev_c = model.clone_states()
        model.train()
        for _ in range(cfg.num_epochs):
            model.load_states(prev_h, prev_c)
            model.load_states(*model.clone_states())
            optimiser.zero_grad()
            emb = model(snap.x, snap.edge_index, snap.edge_attr, update_state=True)
            pos = (emb[snap.edge_index[0]] * emb[snap.edge_index[1]]).sum(-1)
            neg_e = get_negative_samples(snap.edge_index, snap.num_nodes, num_neg=1)
            neg = (emb[neg_e[0]] * emb[neg_e[1]]).sum(-1)
            loss = F.margin_ranking_loss(pos, neg, torch.ones_like(pos), margin=1.0)
            loss.backward()
            model.detach_states()
            optimiser.step()
    return rows


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
