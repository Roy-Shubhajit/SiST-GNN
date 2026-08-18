"""Aggregate the ROLAND-protocol link-prediction runs.

Reads ``results/lp_roland{,_const}/<protocol>/<dataset>/seed<k>/exp_*_lp.json``
and reports, per protocol and dataset:

  SiST-GNN (learned node emb.)  - the architecture described in the paper
  SiST-GNN (constant features)  - matched to ROLAND's published setup
  persistence / degree          - zero-parameter references (baselines_lp.py)
  paper (leaky)                 - what the submission reported
  best published baseline       - strongest number from build_lp_table.py

Writes ``results/lp_summary_learned.csv`` and ``results/lp_summary_const.csv``
in the schema ``build_lp_table.py`` consumes.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {"learned": "results/lp_roland", "constant": "results/lp_roland_const"}
BASELINES = os.path.join(ROOT, "results", "lp_baselines.json")

PAPER = {
    "fixed-split": {"bitcoin-otc": 0.830, "bitcoin-alpha": 0.773, "uci-message": 0.480},
    "live-update": {"as-733": 0.599, "reddit-title": 0.720, "reddit-body": 0.694,
                    "uci-message": 0.328, "bitcoin-otc": 0.551, "bitcoin-alpha": 0.519},
}

BEST_BASELINE = {
    "fixed-split": {"bitcoin-otc": ("ROLAND GRU", 0.220),
                    "bitcoin-alpha": ("ROLAND GRU", 0.289),
                    "uci-message": ("ROLAND GRU", 0.229)},
    "live-update": {"as-733": ("GCRN-GRU", 0.344),
                    "reddit-title": ("ROLAND GRU-Upd.", 0.425),
                    "reddit-body": ("ROLAND GRU-Upd.", 0.362),
                    "uci-message": ("ROLAND GRU-Upd.", 0.112),
                    "bitcoin-otc": ("ROLAND GRU-Upd.", 0.194),
                    "bitcoin-alpha": ("TMetaNet", 0.176)},
}

ORDER = {
    "fixed-split": ["bitcoin-otc", "bitcoin-alpha", "uci-message"],
    "live-update": ["as-733", "reddit-title", "reddit-body",
                    "uci-message", "bitcoin-otc", "bitcoin-alpha"],
}


def mean_std(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


def collect(run_dir: str):
    acc = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(
            ROOT, run_dir, "*", "*", "*", "exp_*_lp.json"))):
        with open(path) as fh:
            payload = json.load(fh)
        proto = payload["eval_method"]
        for ds, res in payload.get("results", {}).get("lp", {}).items():
            if "mrr" not in res:
                print(f"[skip] {os.path.relpath(path, ROOT)}: "
                      f"{res.get('error', 'no mrr')}")
                continue
            acc[proto][ds].append(float(res["mrr"]))
    return acc


def load_baselines() -> Dict[str, Dict]:
    if not os.path.exists(BASELINES):
        return {}
    with open(BASELINES) as fh:
        return {r["dataset"]: r for r in json.load(fh)["baselines"]}


def write_csv(acc, path: str) -> None:
    rows = []
    for proto in ("fixed-split", "live-update"):
        for ds in ORDER[proto]:
            if ds not in acc.get(proto, {}):
                continue
            m, s = mean_std(acc[proto][ds])
            rows.append({"Dataset": ds, "Eval_Method": proto,
                         "Mean_MRR": f"{m:.6f}", "Std_MRR": f"{s:.6f}",
                         "Num_Seeds": len(acc[proto][ds])})
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(path, ROOT)} ({len(rows)} rows)")


def cell(acc, proto, ds) -> str:
    vals = acc.get(proto, {}).get(ds)
    if not vals:
        return "pending"
    m, s = mean_std(vals)
    return f"{m:.3f}±{s:.3f}({len(vals)})"


def report(learned, const, base) -> None:
    for proto in ("fixed-split", "live-update"):
        print(f"\n{'=' * 118}\n{proto.upper()}  |  ROLAND protocol: task t->t+1, "
              f"filtered per-source negatives, per-source MRR\n{'=' * 118}")
        print(f"{'dataset':<14}{'SiST learned':>20}{'SiST constant':>20}"
              f"{'persist':>10}{'degree':>9}{'paper(leaky)':>14}"
              f"{'best published':>22}{'repeat':>9}")
        print("-" * 118)
        for ds in ORDER[proto]:
            b = base.get(ds, {})
            p = PAPER[proto].get(ds)
            bname, bval = BEST_BASELINE[proto].get(ds, ("--", None))
            pers = b.get(f"{proto}/persistence")
            degr = b.get(f"{proto}/degree")
            rep = b.get("edge_repeat_rate")
            print(f"{ds:<14}{cell(learned, proto, ds):>20}{cell(const, proto, ds):>20}"
                  f"{'--' if pers is None else f'{pers:.3f}':>10}"
                  f"{'--' if degr is None else f'{degr:.3f}':>9}"
                  f"{'--' if p is None else f'{p:.3f}':>14}"
                  f"{f'{bval:.3f} {bname}':>22}"
                  f"{'--' if rep is None else f'{rep:.0%}':>9}")
        print("\n  cells are mean±std(n seeds).  'persist'/'degree' are "
              "zero-parameter references under the same protocol.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    learned = collect(VARIANTS["learned"])
    const = collect(VARIANTS["constant"])
    if not learned and not const:
        raise SystemExit("no results found under results/lp_roland*")

    report(learned, const, load_baselines())
    write_csv(learned, os.path.join(ROOT, "results", "lp_summary_learned.csv"))
    write_csv(const, os.path.join(ROOT, "results", "lp_summary_const.csv"))
