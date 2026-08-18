"""Summarise the improvement ablation (results/lp_ablation)."""

from __future__ import annotations

import glob
import json
import os
import statistics as st
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
VARIANTS = {"train on E_t": "lp_ablation_trcurrent",
            "train on E_{t+1}": "lp_ablation_trnext"}

LABEL = {
    "baseline":     "baseline",
    "selftemporal": "+ self-temporal per node",
    "negs20":       "+ train negatives = 20",
    "reg":          "+ dropout 0.1 / wd 1e-5",
    "mlpdec":       "+ MLP decoder",
    "modelsel":     "+ model selection",
    "all":          "+ ALL of the above",
    "best_nolreg":  "+ negs20 + MLP",
    "best":         "+ negs20 + MLP + reg",
    "const":        "constant node features",
    "best_const_noreg": "const + negs20 + MLP",
    "best_const":   "const + negs20 + MLP + reg",
    "fst100_learned":     "selfT + MLP + neg100, learned",
    "fst100_learned_reg": "selfT + MLP + neg100, learned + reg",
    "fst100_const":       "selfT + MLP + neg100, constant",
    "fst100_const_reg":   "selfT + MLP + neg100, constant + reg",
    "fst20_learned":      "selfT + MLP + neg20, learned",
    "fst20_learned_reg":  "selfT + negs20 + MLP + reg  <- best+selfT",
    "neg100_learned_reg": "negs100 + MLP + reg (no selfT)",
}
ORDER = ["baseline", "selftemporal", "negs20", "reg", "mlpdec", "modelsel",
         "all", "best_nolreg", "best",
         "const", "best_const_noreg", "best_const",
         "fst100_learned", "fst100_learned_reg",
         "fst100_const", "fst100_const_reg",
         "fst20_learned", "fst20_learned_reg", "neg100_learned_reg"]

# Reference points under the same protocol.
REFERENCE = [("persistence (0 params)", 0.244),
             ("ROLAND GRU (published)", 0.229)]


def collect(subdir: str):
    acc = defaultdict(list)
    for path in glob.glob(os.path.join(BASE, subdir, "*", "seed*", "exp_*_lp.json")):
        cfg = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path) as fh:
            payload = json.load(fh)
        for _, res in payload.get("results", {}).get("lp", {}).items():
            if "mrr" in res:
                acc[cfg].append(float(res["mrr"]))
    return acc


def cell(acc, cfg) -> str:
    v = acc.get(cfg)
    if not v:
        return "pending"
    sd = st.stdev(v) if len(v) > 1 else 0.0
    return f"{st.mean(v):.3f} ± {sd:.3f} ({len(v)})"


def main() -> None:
    data = {k: collect(v) for k, v in VARIANTS.items()}
    if not any(data.values()):
        raise SystemExit("no ablation results found")

    cols = list(VARIANTS)
    print(f"\nuci-message / fixed-split - evaluation is E_(t+1) in every cell\n")
    print(f"{'config':<28}" + "".join(f"{c:>24}" for c in cols))
    print("-" * (28 + 24 * len(cols)))
    for cfg in ORDER:
        print(f"{LABEL.get(cfg, cfg):<28}"
              + "".join(f"{cell(data[c], cfg):>24}" for c in cols))
    print("-" * (28 + 24 * len(cols)))
    for name, val in REFERENCE:
        print(f"{name:<28}{val:>24.3f}")

    best = None
    for col, acc in data.items():
        for cfg, v in acc.items():
            m = st.mean(v)
            if best is None or m > best[0]:
                best = (m, cfg, col, st.stdev(v) if len(v) > 1 else 0.0)
    if best:
        print(f"\nbest cell: {LABEL.get(best[1], best[1])} / {best[2]} "
              f"= {best[0]:.3f} ± {best[3]:.3f}")


if __name__ == "__main__":
    main()
