"""
Build the LaTeX results table for the SiST-GNN node classification experiments.

Reads results/nc_25seeds/summary.json, formats SiST-GNN's mean +/- std
alongside the published baselines, and writes the table to
results/nc_25seeds/results_table.tex.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean, stdev

# Published baselines copied verbatim from the paper figure shown by the user.
# Each value is (mean, std) in percent (AUC * 100).
BASELINES = {
    "GCN-ROLAND":  {"wikipedia": (58.86, 10.3),  "reddit": (48.25, 9.57), "mooc": (49.93, 6.74)},
    "GAT-ROLAND":  {"wikipedia": (62.81, 9.88),  "reddit": (47.95, 8.42), "mooc": (50.01, 6.34)},
    "TGAT":        {"wikipedia": (67.00, 5.35),  "reddit": (53.64, 5.50), "mooc": (59.27, 4.43)},
    "TGN":         {"wikipedia": (50.61, 13.6),  "reddit": (49.54, 6.23), "mooc": (50.33, 4.47)},
    "TREND":       {"wikipedia": (69.92, 9.27),  "reddit": (64.85, 4.71), "mooc": (66.79, 5.44)},
    "GraphMixer":  {"wikipedia": (65.43, 4.21),  "reddit": (60.21, 5.36), "mooc": (63.72, 4.98)},
}
DATASETS = ["wikipedia", "reddit", "mooc"]


def load_sist_results(summary_path: str) -> dict:
    with open(summary_path) as fh:
        summary = json.load(fh)
    out = {}
    for ds in DATASETS:
        block = summary["datasets"].get(ds)
        if not block:
            out[ds] = None
            continue
        seeds = [s for s in block["per_seed"] if "test_auc" in s]
        tests = [s["test_auc"] * 100.0 for s in seeds]
        if not tests:
            out[ds] = None
            continue
        out[ds] = (mean(tests), stdev(tests) if len(tests) > 1 else 0.0, len(tests))
    return out


def fmt(mean_val: float, std_val: float) -> str:
    """Format a 'mean +/- std' cell using subscripted std (matches the paper figure)."""
    return f"{mean_val:.2f}\\textsubscript{{$\\pm${std_val:.2f}}}"


def is_best(method: str, dataset: str, all_means: dict) -> bool:
    return all_means[dataset] is not None and method == all_means[dataset]


def best_per_dataset(rows: dict) -> dict:
    best = {}
    for ds in DATASETS:
        vals = [(m, rows[m][ds][0]) for m in rows if rows[m].get(ds) is not None]
        if not vals:
            best[ds] = None
            continue
        best[ds] = max(vals, key=lambda kv: kv[1])[0]
    return best


def build_table(sist: dict) -> str:
    rows = dict(BASELINES)
    sist_row = {}
    for ds in DATASETS:
        if sist.get(ds) is not None:
            m, s, _ = sist[ds]
            sist_row[ds] = (m, s)
        else:
            sist_row[ds] = None
    rows["SiST-GNN"] = sist_row

    best = best_per_dataset(rows)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Dynamic node classification ROC-AUC (\\%, mean$\\,\\pm\\,$std)")
    lines.append("over 25 random seeds. Baseline numbers are reproduced from the original")
    lines.append("papers. The best result on each dataset is shown in \\textbf{bold}.}")
    lines.append("\\label{tab:nc_results}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("\\multirow{2}{*}{Methods} & \\multicolumn{3}{c}{Node Classification} \\\\")
    lines.append("\\cmidrule(lr){2-4}")
    lines.append(" & Wikipedia & Reddit & MOOC \\\\")
    lines.append("\\midrule")

    order = ["GCN-ROLAND", "GAT-ROLAND", "TGAT", "TGN", "TREND", "GraphMixer", "SiST-GNN"]
    for method in order:
        cells = []
        if method == "SiST-GNN":
            cells.append("\\textbf{SiST-GNN (Ours)}")
        elif method == "GraphMixer":
            cells.append("\\textsc{GraphMixer}")
        else:
            cells.append(f"\\textsc{{{method}}}")
        for ds in DATASETS:
            entry = rows[method].get(ds)
            if entry is None:
                cells.append("--")
            else:
                m, s = entry
                cell = fmt(m, s)
                if best[ds] == method:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
        if method == "SiST-GNN":
            lines.append("\\midrule")
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=str, default="results/nc_25seeds/summary.json")
    p.add_argument("--out", type=str, default="results/nc_25seeds/results_table.tex")
    args = p.parse_args()

    sist = load_sist_results(args.summary)
    print("SiST-GNN aggregated test AUC (%, mean +/- std over seeds):")
    for ds in DATASETS:
        if sist[ds] is None:
            print(f"  {ds:10s}  -- (no completed runs)")
        else:
            m, s, n = sist[ds]
            print(f"  {ds:10s}  {m:6.2f} +/- {s:5.2f}  (n={n})")

    tex = build_table(sist)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(tex)
    print(f"\nWrote LaTeX table -> {args.out}")
    print("-" * 72)
    print(tex)


if __name__ == "__main__":
    main()
