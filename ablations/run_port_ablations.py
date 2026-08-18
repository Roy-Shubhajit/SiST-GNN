"""Architecture ablations on the ported ROLAND pipeline (fixed-split).

Three sweeps, each varying one factor from the configuration reported in the
main tables:

  hidden_dim  d_h in {32, 64, 128, 256}
  num_layers  L   in {1, 2, 3, 4}
  gnn_type        in {GCNConv, GATConv, SAGEConv}

Fixed elsewhere: paper_fixed_split, undirected, min_edges=10, inner-product
decoder, learned node features, 20 training negatives, 1000 evaluation
negatives, dropout 0.1, weight decay 1e-5, 100 epochs.

NOTE on the backbone sweep: ``_apply_gnn`` feeds the encoded edge attribute to
the convolution as an edge weight, which GCNConv consumes but SAGEConv ignores
entirely (verified: identical MRR with and without). To compare the three
operators on equal inputs the backbone sweep therefore runs with
``--edge-features none``; the other two sweeps keep the full configuration.

Writes results/ablation_{hidden_dim,num_layers,gnn_backbone}.csv in the schema
plot_results.py expects.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
RUNS = os.path.join(ROOT, "results", "ablation_port")
os.makedirs(RESULTS, exist_ok=True)

PYTHON = sys.executable
DATASETS = ["bitcoin-otc", "bitcoin-alpha", "uci-message"]
SEEDS = [0, 1, 2]

BASE = [
    "--mode", "paper_fixed_split", "--undirected", "yes", "--min-edges", "10",
    "--decoder", "dot", "--node-features", "learned",
    "--train-num-neg", "20", "--num-neg", "1000",
    "--dropout", "0.1", "--weight-decay", "1e-5",
    "--num-epochs", "100",
]

SWEEPS = {
    "hidden_dim":   ("--hidden-dim", ["32", "64", "128", "256"],
                     ["--edge-features", "learned"]),
    "num_layers":   ("--num-layers", ["1", "2", "3", "4"],
                     ["--edge-features", "learned"]),
    "gnn_backbone": ("--gnn-type", ["GCNConv", "GATConv", "SAGEConv"],
                     ["--edge-features", "none"]),
}

DEFAULTS = {"hidden_dim": "128", "num_layers": "2", "gnn_type": "GCNConv"}
FLAG2COL = {"--hidden-dim": "hidden_dim", "--num-layers": "num_layers",
            "--gnn-type": "gnn_type"}


def run(name: str, flag: str, value: str, extra: list, ds: str, seed: int) -> None:
    out = os.path.join(RUNS, name, f"{FLAG2COL[flag]}-{value}", ds, f"seed{seed}")
    if glob.glob(os.path.join(out, "exp_*_lp.json")):
        print(f"[skip] {name} {value} {ds} seed={seed}")
        return
    os.makedirs(out, exist_ok=True)
    cmd = [PYTHON, os.path.join(ROOT, "main_roland.py"),
           "--dataset", ds, *BASE, *extra, flag, value,
           "--seed", str(seed), "--out-dir", out]
    print(f"[run ] {name} {value} {ds} seed={seed}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print("       !! FAILED\n" + r.stdout[-500:] + r.stderr[-500:])


def collect(name: str, flag: str) -> None:
    rows = []
    for f in sorted(glob.glob(os.path.join(RUNS, name, "*", "*", "seed*",
                                           "exp_*_lp.json"))):
        payload = json.load(open(f))
        cfg = payload["config"]
        for ds, res in payload["results"]["lp"].items():
            if "mrr" not in res:
                continue
            rows.append({
                "dataset": ds,
                "hidden_dim": cfg["hidden_dim"],
                "num_layers": cfg["num_layers"],
                "gnn_type": cfg["gnn_type"],
                "edge_features": cfg.get("edge_features"),
                "eval_method": cfg["mode"],
                "seed": cfg["seed"],
                "mrr": f"{res['mrr']:.6f}",
                "recall@1": f"{res.get('recall@1', 0):.6f}",
            })
    if not rows:
        print(f"[collect] no rows for {name}")
        return
    path = os.path.join(RESULTS, f"ablation_{name}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[collect] wrote {path} ({len(rows)} rows)")


def main() -> None:
    only = sys.argv[1:] or list(SWEEPS)
    for name in only:
        flag, values, extra = SWEEPS[name]
        for ds in DATASETS:
            for value in values:
                for seed in SEEDS:
                    run(name, flag, value, extra, ds, seed)
        collect(name, flag)


if __name__ == "__main__":
    main()
