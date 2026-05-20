"""
Build the LaTeX results tables for the SiST-GNN link-prediction experiments.

Reads results/lp_summary.csv (produced by ``run.sh --task lp``), formats
SiST-GNN's mean +/- std alongside the published baselines, and writes
two LaTeX tables (fixed-split + live-update) to
results/lp_results_tables.tex.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

# ─────────────────────────────────────────────────────────────────────────────
# Baseline tables (published numbers, transcribed from the paper).
#
# A cell value is one of:
#     (mean, std)   -- mean +/- std
#     (mean, None)  -- bare mean (no s.d. reported in source)
#     "OOM"         -- out-of-memory marker
#     None          -- not reported (\textsc{n/r})
# ─────────────────────────────────────────────────────────────────────────────

Entry = Union[Tuple[float, Optional[float]], str, None]
ResultRow = Tuple[str, str, Dict[str, Entry]]   # (method, cite_key, per_dataset)
Group = Tuple[Optional[str], List[ResultRow]]   # (caption_or_None, rows)


# ── Fixed-split (Table tab:fixed_split) ──────────────────────────────────────
FIXED_SPLIT_DATASETS = ["bitcoin-otc", "bitcoin-alpha", "uci-message"]
FIXED_SPLIT_HEADERS  = ["Bitcoin-OTC", "Bitcoin-Alpha", "UCI-Message"]

FIXED_SPLIT_GROUPS: List[Group] = [
    (None, [
        ("GCN",                "kipf2017semisupervised", {
            "bitcoin-otc":   (0.0025, None),
            "bitcoin-alpha": (0.0031, None),
            "uci-message":   (0.1141, None),
        }),
        ("DynGEM",             "goyal2018dyngem", {
            "bitcoin-otc":   (0.0921, None),
            "bitcoin-alpha": (0.1287, None),
            "uci-message":   (0.1055, None),
        }),
        ("dyngraph2vecAE",     "GOYAL2020104816", {
            "bitcoin-otc":   (0.0916, None),
            "bitcoin-alpha": (0.1478, None),
            "uci-message":   (0.0540, None),
        }),
        ("dyngraph2vecAERNN",  "GOYAL2020104816", {
            "bitcoin-otc":   (0.1268, None),
            "bitcoin-alpha": (0.1945, None),
            "uci-message":   (0.0713, None),
        }),
        ("EvolveGCN-H",        "egcn", {
            "bitcoin-otc":   (0.0690, None),
            "bitcoin-alpha": (0.1104, None),
            "uci-message":   (0.0899, None),
        }),
        ("EvolveGCN-O",        "egcn", {
            "bitcoin-otc":   (0.0968, None),
            "bitcoin-alpha": (0.1185, None),
            "uci-message":   (0.1379, None),
        }),
    ]),
    (None, [
        ("ROLAND Moving-Avg.", "10.1145/3534678.3539300", {
            "bitcoin-otc":   (0.047, 0.002),
            "bitcoin-alpha": (0.140, 0.011),
            "uci-message":   (0.065, 0.005),
        }),
        ("ROLAND MLP",         "10.1145/3534678.3539300", {
            "bitcoin-otc":   (0.078, 0.002),
            "bitcoin-alpha": (0.156, 0.011),
            "uci-message":   (0.088, 0.011),
        }),
        ("ROLAND GRU",         "10.1145/3534678.3539300", {
            "bitcoin-otc":   (0.220, 0.017),
            "bitcoin-alpha": (0.289, 0.012),
            "uci-message":   (0.229, 0.062),
        }),
    ]),
]


# ── Live-update (Table tab:live_update) ──────────────────────────────────────
LIVE_UPDATE_DATASETS = [
    "as-733", "reddit-title", "reddit-body",
    "uci-message", "bitcoin-otc", "bitcoin-alpha",
]
LIVE_UPDATE_HEADERS  = [
    "AS-733", "Reddit-Title", "Reddit-Body",
    "UCI-Message", "Bitcoin-OTC", "Bitcoin-Alpha",
]

LIVE_UPDATE_GROUPS: List[Group] = [
    ("Baselines with standard training (BPTT)", [
        ("EvolveGCN-H", "egcn", {
            "as-733": "OOM", "reddit-title": "OOM",
            "reddit-body": (0.148, 0.013), "uci-message": (0.061, 0.040),
            "bitcoin-otc": (0.067, 0.035), "bitcoin-alpha": (0.079, 0.032),
        }),
        ("EvolveGCN-O", "egcn", {
            "as-733": "OOM", "reddit-title": "OOM",
            "reddit-body": "OOM", "uci-message": (0.071, 0.009),
            "bitcoin-otc": (0.085, 0.022), "bitcoin-alpha": (0.071, 0.025),
        }),
        ("GCRN-GRU", "10.1007/978-3-030-04167-0_33", {
            "as-733": "OOM", "reddit-title": "OOM",
            "reddit-body": "OOM", "uci-message": (0.080, 0.012),
            "bitcoin-otc": "OOM", "bitcoin-alpha": "OOM",
        }),
        ("GCRN-LSTM", "10.1007/978-3-030-04167-0_33", {
            "as-733": "OOM", "reddit-title": "OOM",
            "reddit-body": "OOM", "uci-message": (0.083, 0.001),
            "bitcoin-otc": "OOM", "bitcoin-alpha": "OOM",
        }),
        ("GCRN-Baseline", "10.1007/978-3-030-04167-0_33", {
            "as-733": "OOM", "reddit-title": "OOM",
            "reddit-body": "OOM", "uci-message": (0.069, 0.004),
            "bitcoin-otc": (0.152, 0.011), "bitcoin-alpha": (0.141, 0.005),
        }),
        ("T-GCN", "8809901", {
            "as-733": "OOM", "reddit-title": "OOM",
            "reddit-body": "OOM", "uci-message": (0.054, 0.024),
            "bitcoin-otc": (0.128, 0.049), "bitcoin-alpha": (0.088, 0.038),
        }),
    ]),
    ("Baselines with ROLAND incremental training", [
        ("EvolveGCN-H", "egcn", {
            "as-733": (0.251, 0.079), "reddit-title": (0.165, 0.026),
            "reddit-body": (0.102, 0.010), "uci-message": (0.057, 0.012),
            "bitcoin-otc": (0.076, 0.022), "bitcoin-alpha": (0.054, 0.015),
        }),
        ("EvolveGCN-O", "egcn", {
            "as-733": (0.163, 0.002), "reddit-title": (0.047, 0.004),
            "reddit-body": (0.033, 0.001), "uci-message": (0.066, 0.012),
            "bitcoin-otc": (0.032, 0.004), "bitcoin-alpha": (0.034, 0.002),
        }),
        ("GCRN-GRU", "10.1007/978-3-030-04167-0_33", {
            "as-733": (0.344, 0.001), "reddit-title": (0.338, 0.006),
            "reddit-body": (0.217, 0.004), "uci-message": (0.089, 0.004),
            "bitcoin-otc": (0.173, 0.003), "bitcoin-alpha": (0.140, 0.004),
        }),
        ("GCRN-LSTM", "10.1007/978-3-030-04167-0_33", {
            "as-733": (0.341, 0.001), "reddit-title": (0.344, 0.005),
            "reddit-body": (0.216, 0.000), "uci-message": (0.091, 0.010),
            "bitcoin-otc": (0.174, 0.004), "bitcoin-alpha": (0.146, 0.005),
        }),
        ("GCRN-Baseline", "10.1007/978-3-030-04167-0_33", {
            "as-733": (0.336, 0.002), "reddit-title": (0.351, 0.001),
            "reddit-body": (0.218, 0.002), "uci-message": (0.095, 0.008),
            "bitcoin-otc": (0.183, 0.002), "bitcoin-alpha": (0.145, 0.003),
        }),
        ("T-GCN", "8809901", {
            "as-733": (0.343, 0.002), "reddit-title": (0.391, 0.004),
            "reddit-body": (0.251, 0.001), "uci-message": (0.080, 0.015),
            "bitcoin-otc": (0.083, 0.011), "bitcoin-alpha": (0.069, 0.013),
        }),
    ]),
    ("ROLAND variants, recent topology-augmented meta-learning, and SiST-GNN", [
        ("ROLAND Moving-Avg.", "10.1145/3534678.3539300", {
            "as-733": (0.309, 0.011), "reddit-title": (0.362, 0.007),
            "reddit-body": (0.289, 0.038), "uci-message": (0.075, 0.006),
            "bitcoin-otc": (0.120, 0.002), "bitcoin-alpha": (0.096, 0.010),
        }),
        ("ROLAND MLP-Update", "10.1145/3534678.3539300", {
            "as-733": (0.329, 0.021), "reddit-title": (0.395, 0.006),
            "reddit-body": (0.291, 0.008), "uci-message": (0.103, 0.010),
            "bitcoin-otc": (0.154, 0.010), "bitcoin-alpha": (0.148, 0.012),
        }),
        ("ROLAND GRU-Update", "10.1145/3534678.3539300", {
            "as-733": (0.340, 0.001), "reddit-title": (0.425, 0.015),
            "reddit-body": (0.362, 0.002), "uci-message": (0.112, 0.008),
            "bitcoin-otc": (0.194, 0.004), "bitcoin-alpha": (0.157, 0.007),
        }),
        ("TMetaNet", "li2025tmetanet", {
            "as-733": None,
            "reddit-title": (0.427, 0.010),
            "reddit-body": (0.349, 0.010), "uci-message": (0.109, 0.009),
            "bitcoin-otc": (0.180, 0.012), "bitcoin-alpha": (0.176, 0.005),
        }),
    ]),
]


# ─────────────────────────────────────────────────────────────────────────────
# CSV ingestion (SiST-GNN numbers come from results/lp_summary.csv)
# ─────────────────────────────────────────────────────────────────────────────
def load_sist_results(summary_path: str) -> Dict[str, Dict[str, Entry]]:
    """
    Returns ``{eval_method: {dataset: (mean, std)}}`` for SiST-GNN.

    Missing datasets are silently skipped; the table renderer falls back to
    ``--`` in that case.
    """
    out: Dict[str, Dict[str, Entry]] = {"fixed-split": {}, "live-update": {}}
    if not os.path.exists(summary_path):
        print(f"[lp-table] warning: {summary_path} missing -- SiST-GNN cells will be '--'")
        return out
    with open(summary_path) as fh:
        for row in csv.DictReader(fh):
            ds = row["Dataset"]
            method = row["Eval_Method"]
            try:
                m = float(row["Mean_MRR"])
                s = float(row["Std_MRR"])
            except (KeyError, ValueError):
                continue
            if method in out:
                out[method][ds] = (m, s)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _is_numeric(entry: Entry) -> bool:
    return isinstance(entry, tuple) and entry[0] is not None


def fmt_entry(entry: Entry, bold: bool = False, mean_only_decimals: int = 4) -> str:
    if entry is None:
        return r"\textsc{n/r}"
    if entry == "OOM":
        return "OOM"
    if not isinstance(entry, tuple):
        return "--"
    mean_val, std = entry
    if std is None:
        body = f"{mean_val:.{mean_only_decimals}f}"
    else:
        body = f"{mean_val:.3f} \\pm {std:.3f}"
    if bold:
        return f"$\\mathbf{{{body}}}$"
    return f"${body}$"


def _best_per_dataset(
    all_rows: List[ResultRow],
    sist_results: Dict[str, Entry],
    datasets: List[str],
) -> Dict[str, str]:
    """
    Return ``{dataset: winning_method_name}``. SiST-GNN participates in the
    comparison so its cells get bolded too when it wins. Method name "SiST-GNN"
    is reserved.
    """
    best: Dict[str, str] = {}
    for ds in datasets:
        candidates: List[Tuple[float, str]] = []
        for method, _cite, results in all_rows:
            entry = results.get(ds)
            if _is_numeric(entry):
                candidates.append((entry[0], method))
        sist = sist_results.get(ds)
        if _is_numeric(sist):
            candidates.append((sist[0], "SiST-GNN"))
        if candidates:
            best[ds] = max(candidates, key=lambda kv: kv[0])[1]
        else:
            best[ds] = ""
    return best


def _improvement_row(
    all_rows: List[ResultRow],
    sist_results: Dict[str, Entry],
    datasets: List[str],
) -> Dict[str, Optional[float]]:
    """
    Percentage gain of SiST-GNN over the strongest *baseline* (excluding
    SiST-GNN itself). ``None`` when no baseline or SiST-GNN value is available.
    """
    out: Dict[str, Optional[float]] = {}
    for ds in datasets:
        sist = sist_results.get(ds)
        if not _is_numeric(sist):
            out[ds] = None
            continue
        baseline_means = [
            results[ds][0]
            for _m, _c, results in all_rows
            if _is_numeric(results.get(ds))
        ]
        if not baseline_means:
            out[ds] = None
            continue
        best_base = max(baseline_means)
        if best_base <= 0:
            out[ds] = None
            continue
        out[ds] = (sist[0] - best_base) / best_base * 100.0
    return out


def _flatten_groups(groups: List[Group]) -> List[ResultRow]:
    return [row for _, rows in groups for row in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Table builders
# ─────────────────────────────────────────────────────────────────────────────
def build_fixed_split_table(sist_results: Dict[str, Entry]) -> str:
    n_cols   = len(FIXED_SPLIT_DATASETS)
    all_rows = _flatten_groups(FIXED_SPLIT_GROUPS)
    best     = _best_per_dataset(all_rows, sist_results, FIXED_SPLIT_DATASETS)
    impr     = _improvement_row(all_rows, sist_results, FIXED_SPLIT_DATASETS)

    col_spec = "l" + "r" * n_cols
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fixed-split MRR (mean $\pm$ s.e.\ over 3 seeds). "
        r"Following ROLAND~\cite{10.1145/3534678.3539300}, all methods use "
        r"identical snapshot frequencies, test snapshots, and MRR computation. "
        r"Best per column in \textbf{bold}.}",
        r"\label{tab:fixed_split}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"  \toprule",
        "  " + " & ".join([""] + FIXED_SPLIT_HEADERS) + r" \\",
        r"  \midrule",
    ]
    for gi, (_caption, rows) in enumerate(FIXED_SPLIT_GROUPS):
        if gi > 0:
            out.append(r"  \midrule")
        for method, cite, results in rows:
            cells = [f"{method}~\\cite{{{cite}}}"]
            for ds in FIXED_SPLIT_DATASETS:
                entry = results.get(ds)
                cells.append(fmt_entry(entry, bold=(best[ds] == method)))
            out.append("  " + " & ".join(cells) + r" \\")

    out.append(r"  \midrule")
    sist_cells = [r"\textbf{SiST-GNN (ours)}"]
    for ds in FIXED_SPLIT_DATASETS:
        entry = sist_results.get(ds)
        sist_cells.append(fmt_entry(entry, bold=(best[ds] == "SiST-GNN")))
    out.append("  " + " & ".join(sist_cells) + r" \\")

    out.append(r"  \midrule")
    impr_cells = ["Improvement"]
    for ds in FIXED_SPLIT_DATASETS:
        v = impr[ds]
        impr_cells.append(f"${v:+.1f}\\%$" if v is not None else "--")
    out.append("  " + " & ".join(impr_cells) + r" \\")

    out.extend([
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(out) + "\n"


def build_live_update_table(sist_results: Dict[str, Entry]) -> str:
    n_cols   = len(LIVE_UPDATE_DATASETS)
    all_rows = _flatten_groups(LIVE_UPDATE_GROUPS)
    best     = _best_per_dataset(all_rows, sist_results, LIVE_UPDATE_DATASETS)
    impr     = _improvement_row(all_rows, sist_results, LIVE_UPDATE_DATASETS)

    col_spec = "l" + "r" * n_cols
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Live-update MRR (mean $\pm$ std.\ over 3 seeds). "
        r"Top: baselines with standard BPTT training. "
        r"Middle: baselines with ROLAND incremental training. "
        r"Bottom: ROLAND variants, the recent topology-augmented meta-learner "
        r"TMetaNet, and SiST-GNN. OOM = out-of-memory after five retries; "
        r"\textsc{n/r} = not reported in the source publication. "
        r"Best per column in \textbf{bold}.}",
        r"\label{tab:live_update}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"  \toprule",
        "  " + " & ".join([""] + LIVE_UPDATE_HEADERS) + r" \\",
        r"  \midrule",
    ]
    for gi, (caption, rows) in enumerate(LIVE_UPDATE_GROUPS):
        if caption:
            if gi > 0:
                out.append(r"  \midrule")
            out.append(
                rf"  \multicolumn{{{n_cols + 1}}}{{c}}{{\textit{{{caption}}}}} \\"
            )
            out.append(r"  \midrule")
        for method, cite, results in rows:
            cells = [f"{method}~\\cite{{{cite}}}"]
            for ds in LIVE_UPDATE_DATASETS:
                entry = results.get(ds)
                cells.append(fmt_entry(entry, bold=(best[ds] == method)))
            out.append("  " + " & ".join(cells) + r" \\")

    out.append(r"  \midrule")
    sist_cells = [r"\textbf{SiST-GNN (ours)}"]
    for ds in LIVE_UPDATE_DATASETS:
        entry = sist_results.get(ds)
        sist_cells.append(fmt_entry(entry, bold=(best[ds] == "SiST-GNN")))
    out.append("  " + " & ".join(sist_cells) + r" \\")

    out.append(r"  \midrule")
    impr_cells = ["Improvement"]
    for ds in LIVE_UPDATE_DATASETS:
        v = impr[ds]
        impr_cells.append(f"${v:+.1f}\\%$" if v is not None else "--")
    out.append("  " + " & ".join(impr_cells) + r" \\")

    out.extend([
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(out) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def _print_sist_summary(sist: Dict[str, Dict[str, Entry]]) -> None:
    print("SiST-GNN LP results (mean +/- std), source: results/lp_summary.csv")
    for method, label in [("fixed-split", "Fixed-split"),
                          ("live-update", "Live-update")]:
        print(f"\n[{label}]")
        rows = sist.get(method, {})
        if not rows:
            print("  -- (no rows for this eval_method)")
            continue
        for ds in sorted(rows):
            entry = rows[ds]
            if _is_numeric(entry):
                m, s = entry
                print(f"  {ds:18s}  {m:.4f} +/- {s:.4f}")
            else:
                print(f"  {ds:18s}  --")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=str, default="results/lp_summary.csv",
                   help="CSV produced by run.sh --task lp")
    p.add_argument("--out", type=str, default="results/lp_results_tables.tex",
                   help="Where to write the combined LaTeX table file")
    args = p.parse_args()

    sist = load_sist_results(args.summary)
    _print_sist_summary(sist)

    fixed_tex = build_fixed_split_table(sist.get("fixed-split", {}))
    live_tex  = build_live_update_table(sist.get("live-update", {}))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("% SiST-GNN link-prediction results -- auto-generated by build_lp_table.py\n\n")
        fh.write("% Fixed-split (Bitcoin-OTC, Bitcoin-Alpha, UCI-Message)\n")
        fh.write(fixed_tex)
        fh.write("\n% Live-update (AS-733, Reddit-Title, Reddit-Body, UCI-Message, Bitcoin-OTC, Bitcoin-Alpha)\n")
        fh.write(live_tex)

    print(f"\nWrote LaTeX tables -> {args.out}")
    print("-" * 72)
    print(fixed_tex)
    print(live_tex)


if __name__ == "__main__":
    main()
