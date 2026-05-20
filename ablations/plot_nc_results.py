"""Generate the NC ablation figure consumed by the paper.

Reads the four CSVs produced by ``ablation_nc_*.py`` from ``results/`` and
writes ``figures/ablations_nc.pdf`` — a 1×4 panel figure:

    (a) GNN backbone        — bar plot per dataset, grouped by backbone
    (b) Hidden dimension    — error-bar line plot vs. $d_h$
    (c) Bucket hours        — error-bar line plot vs. snapshot width
    (d) Positive-class wt.  — bar plot per dataset, grouped by strategy

Metric reported: test ROC AUC (mean ± std across seeds). If a CSV is
missing the corresponding panel is rendered with a "data pending"
placeholder so the LaTeX build still succeeds.
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

# Match the LP plot's aesthetic so paper figures are visually consistent.
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

# Consistent per-dataset colours across panels (b) and (c).
DATASET_COLOURS = {
    "wikipedia": "#3b6ea5",
    "reddit":    "#549673",
    "mooc":      "#a25f7a",
}
# Categorical palettes for the two bar-plot panels.
BACKBONE_PALETTE = ["#3b6ea5", "#549673", "#a25f7a"]
POSW_PALETTE     = ["#3b6ea5", "#d28a3c", "#a25f7a"]


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


def _datasets_in_order(found: List[str]) -> List[str]:
    """Return found datasets in a stable canonical order (wiki/reddit/mooc)."""
    canonical = ["wikipedia", "reddit", "mooc"]
    return [d for d in canonical if d in set(found)] + \
           [d for d in found if d not in canonical]


# ─────────────────────────────────────────────────────────────────────────────
# (a) GNN backbone
# ─────────────────────────────────────────────────────────────────────────────
def _panel_backbone(ax: plt.Axes) -> None:
    rows = _read_csv(os.path.join(RESULTS_DIR, "ablation_nc_gnn_backbone.csv"))
    ax.set_title("(a) GNN backbone", pad=20)
    ax.set_ylabel("Test ROC AUC")
    if rows is None:
        _placeholder(ax, "run ablation_nc_gnn_backbone.py")
        return
    agg = defaultdict(list)
    for r in rows:
        bb = "GATConv" if r["gnn_type"] == "GATv2Conv" else r["gnn_type"]
        agg[(r["dataset"], bb)].append(float(r["test_auc"]))
    datasets  = _datasets_in_order(list({k[0] for k in agg}))
    backbones = ["GCNConv", "GATConv", "SAGEConv"]
    width = 0.26
    x = np.arange(len(datasets))
    for i, bb in enumerate(backbones):
        means = [np.mean(agg[(d, bb)]) if (d, bb) in agg else np.nan for d in datasets]
        stds  = [np.std (agg[(d, bb)]) if (d, bb) in agg else 0.0    for d in datasets]
        ax.bar(x + (i - 1) * width, means, width,
               yerr=stds, capsize=2, label=bb, color=BACKBONE_PALETTE[i])
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=12, ha="right")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=7,
              columnspacing=1.0, handletextpad=0.4, handlelength=1.4)


# ─────────────────────────────────────────────────────────────────────────────
# (b) Hidden dimension
# ─────────────────────────────────────────────────────────────────────────────
def _panel_hidden_dim(ax: plt.Axes) -> None:
    rows = _read_csv(os.path.join(RESULTS_DIR, "ablation_nc_hidden_dim.csv"))
    ax.set_title("(b) Hidden dimension $d_h$", pad=20)
    ax.set_xlabel("$d_h$"); ax.set_ylabel("Test ROC AUC")
    ax.set_xscale("log", base=2)
    if rows is None:
        _placeholder(ax, "run ablation_nc_hidden_dim.py")
        return
    agg = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], int(r["hidden_dim"]))].append(float(r["test_auc"]))
    datasets = _datasets_in_order(list({k[0] for k in agg}))
    dims = sorted({k[1] for k in agg})
    for ds in datasets:
        means = [np.mean(agg[(ds, d)]) for d in dims]
        stds  = [np.std (agg[(ds, d)]) for d in dims]
        ax.errorbar(dims, means, yerr=stds, marker="o", capsize=2,
                    label=ds, color=DATASET_COLOURS.get(ds))
    ax.set_xticks(dims); ax.set_xticklabels(dims)
    ax.set_xlim(dims[0] / 1.12, dims[-1] * 1.12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=7,
              columnspacing=1.0, handletextpad=0.4, handlelength=1.4)


# ─────────────────────────────────────────────────────────────────────────────
# (c) Bucket hours
# ─────────────────────────────────────────────────────────────────────────────
def _panel_bucket_hours(ax: plt.Axes) -> None:
    rows = _read_csv(os.path.join(RESULTS_DIR, "ablation_nc_bucket_hours.csv"))
    ax.set_title("(c) Snapshot width", pad=20)
    ax.set_xlabel("bucket hours"); ax.set_ylabel("Test ROC AUC")
    ax.set_xscale("log", base=2)
    if rows is None:
        _placeholder(ax, "run ablation_nc_bucket_hours.py")
        return
    agg = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], float(r["bucket_hours"]))].append(float(r["test_auc"]))
    datasets = _datasets_in_order(list({k[0] for k in agg}))
    bhs = sorted({k[1] for k in agg})
    for ds in datasets:
        means = [np.mean(agg[(ds, b)]) for b in bhs]
        stds  = [np.std (agg[(ds, b)]) for b in bhs]
        ax.errorbar(bhs, means, yerr=stds, marker="s", capsize=2,
                    label=ds, color=DATASET_COLOURS.get(ds))
    ax.set_xticks(bhs); ax.set_xticklabels([f"{b:g}" for b in bhs])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=7,
              columnspacing=1.0, handletextpad=0.4, handlelength=1.4)


# ─────────────────────────────────────────────────────────────────────────────
# (d) Positive-class weight
# ─────────────────────────────────────────────────────────────────────────────
def _panel_pos_weight(ax: plt.Axes) -> None:
    rows = _read_csv(os.path.join(RESULTS_DIR, "ablation_nc_pos_weight.csv"))
    ax.set_title("(d) BCE positive-class weight", pad=20)
    ax.set_ylabel("Test ROC AUC")
    if rows is None:
        _placeholder(ax, "run ablation_nc_pos_weight.py")
        return
    agg = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r["pos_weight"])].append(float(r["test_auc"]))
    datasets = _datasets_in_order(list({k[0] for k in agg}))
    strategies = ["none", "sqrt", "balanced"]
    width = 0.26
    x = np.arange(len(datasets))
    for i, pw in enumerate(strategies):
        means = [np.mean(agg[(d, pw)]) if (d, pw) in agg else np.nan for d in datasets]
        stds  = [np.std (agg[(d, pw)]) if (d, pw) in agg else 0.0    for d in datasets]
        ax.bar(x + (i - 1) * width, means, width,
               yerr=stds, capsize=2, label=pw, color=POSW_PALETTE[i])
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=12, ha="right")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=7,
              columnspacing=1.0, handletextpad=0.4, handlelength=1.4)


# ─────────────────────────────────────────────────────────────────────────────
# Composite figure
# ─────────────────────────────────────────────────────────────────────────────
def plot_nc_ablations() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.1),
                             constrained_layout=True)
    _panel_backbone   (axes[0])
    _panel_hidden_dim (axes[1])
    _panel_bucket_hours(axes[2])
    _panel_pos_weight (axes[3])
    path = os.path.join(FIGURES_DIR, "ablations_nc.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[plot-nc] wrote {path}")


def main() -> None:
    plot_nc_ablations()


if __name__ == "__main__":
    main()
