"""Generate every LP figure consumed by the paper.

Reads the CSVs produced by the `ablation_*.py`, `per_snapshot_mrr.py` and
`statistical_significance.py` scripts in this folder and writes the
following PDFs into `figures/`:

  ablations.pdf                -- 3-panel ablation figure  (referenced as fig:ablations)
  per_snapshot_mrr.pdf         -- per-snapshot trajectory  (referenced as fig:per_snapshot)
  statistical_significance.pdf -- bootstrap CIs            (referenced as fig:stat_sig)

If a CSV is missing the corresponding panel is rendered with a "data
pending" placeholder so the LaTeX build still succeeds; the paper compiles
end-to-end before the experiments finish.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ABLATIONS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR   = os.path.join(ABLATIONS_DIR, "results")
FIGURES_DIR   = os.path.join(ABLATIONS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ROLAND-GRU published per-dataset MRR (used as a horizontal reference line
# in the per-snapshot trajectory figure).
ROLAND_GRU_MRR = {
    "as-733":         0.340,
    "reddit-title":   0.425,
    "reddit-body":    0.362,
    "uci-message":    0.112,
    "bitcoin-otc":    0.194,
    "bitcoin-alpha":  0.157,
}

# Match the paper's tighter aesthetic.
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":   10,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.6,
    "lines.linewidth":   1.2,
})


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_csv(path: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _placeholder(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=9, color="0.45",
            style="italic")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.7")


# ─────────────────────────────────────────────────────────────────────────────
# Ablations figure -- 3 panels: hidden_dim, num_layers, GNN backbone
# ─────────────────────────────────────────────────────────────────────────────
def _panel_hidden_dim(ax: plt.Axes) -> None:
    rows = _read_csv(os.path.join(RESULTS_DIR, "ablation_hidden_dim.csv"))
    ax.set_title("(a) Hidden dimension $d_h$", pad=20)
    ax.set_xlabel("$d_h$"); ax.set_ylabel("MRR")
    ax.set_xscale("log", base=2)
    if rows is None:
        _placeholder(ax, "run ablation_hidden_dim.py")
        return
    agg = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], int(r["hidden_dim"]))].append(float(r["mrr"]))
    datasets = sorted({k[0] for k in agg})
    dims = sorted({k[1] for k in agg})
    for ds in datasets:
        means = [np.mean(agg[(ds, d)]) for d in dims]
        stds  = [np.std(agg[(ds, d)])  for d in dims]
        ax.errorbar(dims, means, yerr=stds, marker="o", capsize=2, label=ds)
    ax.set_xticks(dims); ax.set_xticklabels(dims)
    ax.set_xlim(dims[0] / 1.12, dims[-1] * 1.12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=min(3, max(1, len(datasets))), frameon=False, fontsize=7,
              columnspacing=1.0, handletextpad=0.4, handlelength=1.4)


def _panel_num_layers(ax: plt.Axes) -> None:
    rows = _read_csv(os.path.join(RESULTS_DIR, "ablation_num_layers.csv"))
    ax.set_title("(b) Number of layers $L$", pad=20)
    ax.set_xlabel("$L$"); ax.set_ylabel("MRR")
    if rows is None:
        _placeholder(ax, "run ablation_num_layers.py")
        return
    agg = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], int(r["num_layers"]))].append(float(r["mrr"]))
    datasets = sorted({k[0] for k in agg})
    Ls = sorted({k[1] for k in agg})
    for ds in datasets:
        means = [np.mean(agg[(ds, L)]) for L in Ls]
        stds  = [np.std(agg[(ds, L)])  for L in Ls]
        ax.errorbar(Ls, means, yerr=stds, marker="s", capsize=2, label=ds)
    ax.set_xticks(Ls)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=min(3, max(1, len(datasets))), frameon=False, fontsize=7,
              columnspacing=1.0, handletextpad=0.4, handlelength=1.4)


def _panel_backbone(ax: plt.Axes) -> None:
    rows = _read_csv(os.path.join(RESULTS_DIR, "ablation_gnn_backbone.csv"))
    ax.set_title("(c) GNN backbone", pad=20)
    ax.set_ylabel("MRR")
    if rows is None:
        _placeholder(ax, "run ablation_gnn_backbone.py")
        return
    agg = defaultdict(list)
    for r in rows:
        bb = "GATConv" if r["gnn_type"] == "GATv2Conv" else r["gnn_type"]
        agg[(r["dataset"], bb)].append(float(r["mrr"]))
    datasets = sorted({k[0] for k in agg})
    backbones = ["GCNConv", "GATConv", "SAGEConv"]
    width = 0.26
    x = np.arange(len(datasets))
    palette = ["#3b6ea5", "#549673", "#a25f7a"]
    for i, bb in enumerate(backbones):
        means = [np.mean(agg[(d, bb)]) if (d, bb) in agg else np.nan for d in datasets]
        stds  = [np.std (agg[(d, bb)]) if (d, bb) in agg else 0.0    for d in datasets]
        ax.bar(x + (i - 1) * width, means, width,
               yerr=stds, capsize=2, label=bb, color=palette[i])
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=12, ha="right")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=7,
              columnspacing=1.0, handletextpad=0.4, handlelength=1.4)


def plot_ablations() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0),
                             constrained_layout=True)
    _panel_hidden_dim(axes[0])
    _panel_num_layers(axes[1])
    _panel_backbone (axes[2])
    path = os.path.join(FIGURES_DIR, "ablations.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-snapshot MRR -- SiST-GNN with ROLAND-GRU reference line
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_snapshot() -> None:
    path_in = os.path.join(RESULTS_DIR, "per_snapshot_mrr.csv")
    rows = _read_csv(path_in)
    datasets_default = ["bitcoin-otc", "bitcoin-alpha"]
    fig, axes = plt.subplots(1, len(datasets_default), figsize=(7.0, 2.6),
                             constrained_layout=True)
    if rows is None:
        for ax, ds in zip(axes, datasets_default):
            ax.set_title(ds, pad=20)
            _placeholder(ax, "run per_snapshot_mrr.py")
    else:
        by = defaultdict(list)              # by[ds] = [(t, mrr)]
        for r in rows:
            by[r["dataset"]].append((int(r["snapshot"]), float(r["mrr"])))
        datasets = sorted(by.keys()) or datasets_default
        for ax, ds in zip(axes, datasets):
            pts = sorted(by[ds])
            ts, mrrs = zip(*pts) if pts else ([], [])
            ax.plot(ts, mrrs, color="#3b6ea5", label="SiST-GNN")
            if ds in ROLAND_GRU_MRR:
                ax.axhline(ROLAND_GRU_MRR[ds], linestyle="--",
                           color="#d28a3c",
                           label=f"ROLAND-GRU avg ({ROLAND_GRU_MRR[ds]:.3f})")
            ax.set_title(ds, pad=20)
            ax.set_xlabel("snapshot $t$"); ax.set_ylabel("MRR")
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      ncol=2, frameon=False, fontsize=7,
                      columnspacing=1.0, handletextpad=0.4, handlelength=1.4)
    out = os.path.join(FIGURES_DIR, "per_snapshot_mrr.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance
# ─────────────────────────────────────────────────────────────────────────────
def plot_stat_sig() -> None:
    rows = _read_csv(os.path.join(RESULTS_DIR, "statistical_significance.csv"))
    fig, ax = plt.subplots(figsize=(7.0, 2.8), constrained_layout=True)
    if rows is None:
        _placeholder(ax, "run statistical_significance.py")
    else:
        labels  = [r["dataset"] for r in rows]
        means   = np.array([float(r["mean"])  for r in rows])
        ci_lo   = np.array([float(r["ci_lo"]) for r in rows])
        ci_hi   = np.array([float(r["ci_hi"]) for r in rows])
        ref     = np.array([float(r["roland_gru_mrr"]) for r in rows])
        x = np.arange(len(labels))
        ax.errorbar(x, means, yerr=[means - ci_lo, ci_hi - means],
                    fmt="o", color="#3b6ea5", capsize=3,
                    label="SiST-GNN (95\\% bootstrap CI)")
        ax.scatter(x, ref, marker="x", color="#d28a3c", label="ROLAND-GRU")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("MRR")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=2, frameon=False, fontsize=8,
                  columnspacing=1.2, handletextpad=0.4, handlelength=1.4)
        for xi, r in zip(x, rows):
            ax.annotate(f"p={float(r['p_value']):.3f}",
                        (xi, ci_hi[xi]), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=7)
    out = os.path.join(FIGURES_DIR, "statistical_significance.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main() -> None:
    plot_ablations()
    plot_per_snapshot()
    plot_stat_sig()


if __name__ == "__main__":
    main()
