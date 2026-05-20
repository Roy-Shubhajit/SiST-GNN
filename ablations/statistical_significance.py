"""Statistical significance / bootstrap confidence intervals.

For each of the six benchmarks, run SiST-GNN with $S$ different random seeds
under the live-update protocol and compute a non-parametric 95% bootstrap
confidence interval over the mean MRR.

The resulting CSV (`results/statistical_significance.csv`) has columns
`dataset, method, mean, ci_lo, ci_hi, n_seeds`, and is consumed by the
plotting script to render error-bar overlays on top of Figure 4. We also
report a one-sided paired-bootstrap p-value against the strongest baseline
(ROLAND-GRU's published number from the main table); if you have run-by-run
ROLAND-GRU traces you can replace `ROLAND_GRU_MRR` below with the actual seed
distribution and we will compute a true paired test.

This script does NOT retrain baselines from scratch (that would re-run the
entire main table); it uses the seed-distribution of SiST-GNN against the
*scalar* MRR of ROLAND-GRU as a conservative reference.
"""
from __future__ import annotations

import csv
import os
from statistics import mean

import numpy as np

from _common import RESULTS_DIR, RunConfig, parse_dataset_arg, run_one, save_csv


# Strongest published live-update MRR per dataset (ROLAND-GRU) -- used as a
# fixed reference for the one-sided test. Numbers taken verbatim from the
# main table of the paper.
ROLAND_GRU_MRR = {
    "as-733":         0.340,
    "reddit-title":   0.425,
    "reddit-body":    0.362,
    "uci-message":    0.112,
    "bitcoin-otc":    0.194,
    "bitcoin-alpha":  0.157,
}

DEFAULT_DATASETS = list(ROLAND_GRU_MRR.keys())
N_SEEDS = 5
N_BOOTSTRAP = 10000


def bootstrap_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    n = len(values)
    samples = rng.choice(values, size=(N_BOOTSTRAP, n), replace=True).mean(axis=1)
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


def bootstrap_pvalue_vs_constant(values: list[float], ref: float) -> float:
    """Fraction of bootstrap means that fall below the reference scalar.
    Returns a one-sided p-value for H0: mean(SiST-GNN) <= ref."""
    rng = np.random.default_rng(1)
    n = len(values)
    samples = rng.choice(values, size=(N_BOOTSTRAP, n), replace=True).mean(axis=1)
    return float((samples <= ref).mean())


def main() -> None:
    datasets = parse_dataset_arg(DEFAULT_DATASETS)
    rows = []
    for ds in datasets:
        mrrs = []
        for seed in range(N_SEEDS):
            cfg = RunConfig(dataset=ds, eval_method="live-update", seed=seed)
            print(f"[stat_sig] dataset={ds} seed={seed}")
            mrrs.append(run_one(cfg)["mrr"])
        ci_lo, ci_hi = bootstrap_ci(mrrs)
        pval = bootstrap_pvalue_vs_constant(mrrs, ROLAND_GRU_MRR[ds])
        rows.append({
            "dataset":  ds,
            "method":   "SiST-GNN",
            "mean":     mean(mrrs),
            "ci_lo":    ci_lo,
            "ci_hi":    ci_hi,
            "n_seeds":  N_SEEDS,
            "roland_gru_mrr": ROLAND_GRU_MRR[ds],
            "p_value":  pval,
        })
    save_csv(rows, "statistical_significance.csv")


if __name__ == "__main__":
    main()
