"""Trivial reference baselines for link prediction under ROLAND's protocol.

Two zero-parameter references, both scored on E_{t+1} with the same per-source
MRR and filtered negatives the models use:

* **persistence** - ``score(u,v) = 1 if (u,v) in E_t else 0``. Copies the
  current snapshot forward. On a graph whose edges largely repeat, this is
  most of what the task affords, so it bounds how much of a model's MRR
  reflects forecasting rather than reproducing G_t.
* **degree**      - ``score(u,v) = deg(v) in G_t``. Pure popularity.

Also reports the edge repeat rate |E_t & E_{t+1}| / |E_{t+1}|, which is what
drives the persistence number.

Writes ``results/lp_baselines.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from data_processing_lp import load_lp_dataset
from evaluate_lp import gen_negative_edges
from helper import save_json, seed_everything

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALL = ["as-733", "reddit-title", "reddit-body", "uci-message",
       "bitcoin-otc", "bitcoin-alpha"]


@torch.no_grad()
def _mrr_from_scores(pos_score, tgt, src_lst, neg_score, num_nodes):
    best = torch.full((num_nodes,), float("-inf"), device=pos_score.device)
    best.scatter_reduce_(0, tgt[0], pos_score, reduce="amax", include_self=True)
    ranks = (neg_score >= best[src_lst].view(-1, 1)).sum(1) + 1
    return float((1.0 / ranks.float()).mean())


@torch.no_grad()
def baselines_for_task(mp_snap, target_snap, num_neg: int = 1000) -> Dict[str, float]:
    n = mp_snap.num_nodes
    tgt = target_snap.edge_index
    mp_ids = mp_snap.edge_index[0].long() * n + mp_snap.edge_index[1].long()
    src_lst, neg_dst = gen_negative_edges(tgt, num_neg, n, forbidden=tgt)
    neg_ids = src_lst.view(-1, 1).long() * n + neg_dst.long()

    # persistence
    p_pos = torch.isin(tgt[0].long() * n + tgt[1].long(), mp_ids).float()
    p_neg = torch.isin(neg_ids, mp_ids).float()

    # degree
    deg = torch.zeros(n, device=tgt.device)
    deg.index_add_(0, mp_snap.edge_index.reshape(-1),
                   torch.ones(mp_snap.edge_index.numel(), device=tgt.device))

    return {
        "persistence": _mrr_from_scores(p_pos, tgt, src_lst, p_neg, n),
        "degree": _mrr_from_scores(deg[tgt[1]], tgt, src_lst, deg[neg_dst], n),
    }


def run(ds: str, num_neg: int = 1000) -> Dict:
    snaps = load_lp_dataset(ds)
    T = len(snaps)
    train_end = int(T * 0.9)
    n = snaps[0].num_nodes

    acc: Dict[str, Dict[str, List[float]]] = {
        "fixed-split": {"persistence": [], "degree": []},
        "live-update": {"persistence": [], "degree": []},
    }
    repeats: List[float] = []

    for t in range(T - 1):
        a, b = snaps[t].to(DEV), snaps[t + 1].to(DEV)
        if a.edge_index.size(1) == 0 or b.edge_index.size(1) == 0:
            continue
        res = baselines_for_task(a, b, num_neg)

        a_ids = set((a.edge_index[0].long() * n + a.edge_index[1].long()).tolist())
        b_ids = set((b.edge_index[0].long() * n + b.edge_index[1].long()).tolist())
        repeats.append(len(a_ids & b_ids) / len(b_ids))

        # live-update spans every task; fixed-split only the held-out tail.
        for k, v in res.items():
            acc["live-update"][k].append(v)
            if t >= train_end - 1:
                acc["fixed-split"][k].append(v)

    out = {"dataset": ds, "num_snapshots": T, "num_nodes": n,
           "edge_repeat_rate": float(np.mean(repeats))}
    for proto, d in acc.items():
        for k, vals in d.items():
            out[f"{proto}/{k}"] = float(np.mean(vals)) if vals else None
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=str, default=",".join(ALL))
    p.add_argument("--num-neg", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/lp_baselines.json")
    args = p.parse_args()

    seed_everything(args.seed)
    rows = [run(ds, args.num_neg) for ds in args.datasets.split(",") if ds]

    save_json(args.out, {"num_neg": args.num_neg, "seed": args.seed,
                         "baselines": rows})
    print(f"\n{'dataset':<15}{'repeat':>9}{'persist(fx)':>13}{'persist(lu)':>13}"
          f"{'degree(fx)':>12}{'degree(lu)':>12}")
    print("-" * 74)
    for r in rows:
        print(f"{r['dataset']:<15}{r['edge_repeat_rate']:>8.1%}"
              f"{r['fixed-split/persistence']:>13.4f}"
              f"{r['live-update/persistence']:>13.4f}"
              f"{r['fixed-split/degree']:>12.4f}"
              f"{r['live-update/degree']:>12.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
