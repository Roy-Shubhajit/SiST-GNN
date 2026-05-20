# SiST-GNN — ablation and analysis suite

> **Paper:** *SiST-GNN: Simultaneous Spatial-Temporal Message Passing for
> Dynamic Graph Representation Learning.*

This folder contains every experiment that produces a paper figure
beyond the two main results tables. None of these scripts are required to
reproduce the headline numbers (those come from `main_lp.py` and
`main_nc.py` at the repository root); they produce the *plots* in the
ablation / analysis sections.

## What is here

### Link-prediction ablations (use `_common.py` and the LP pipeline)

| File | What it does | Output CSV | Output figure |
|---|---|---|---|
| `ablation_hidden_dim.py`     | sweep `d_h ∈ {32, 64, 128, 256}`              | `results/ablation_hidden_dim.csv`        | `figures/ablations.pdf` (panel a) |
| `ablation_num_layers.py`     | sweep `L ∈ {1, 2, 3, 4}`                       | `results/ablation_num_layers.csv`        | `figures/ablations.pdf` (panel b) |
| `ablation_gnn_backbone.py`   | GCN vs. GAT vs. SAGE                           | `results/ablation_gnn_backbone.csv`      | `figures/ablations.pdf` (panel c) |
| `per_snapshot_mrr.py`        | per-`t` MRR under live-update                  | `results/per_snapshot_mrr.csv`           | `figures/per_snapshot_mrr.pdf`    |
| `statistical_significance.py`| 5-seed bootstrap 95 % CI per dataset           | `results/statistical_significance.csv`   | `figures/statistical_significance.pdf` |
| `plot_results.py`            | reads every LP CSV, regenerates LP figures     | —                                        | `figures/ablations.pdf`, `figures/per_snapshot_mrr.pdf`, `figures/statistical_significance.pdf` |

### Node-classification ablations (use `_nc_common.py` and the NC pipeline)

| File | What it does | Output CSV | Output figure |
|---|---|---|---|
| `ablation_nc_gnn_backbone.py`| GCN vs. SAGE vs. GATv2                         | `results/ablation_nc_gnn_backbone.csv`   | `figures/ablations_nc.pdf` (panel a) |
| `ablation_nc_hidden_dim.py`  | sweep `d_h ∈ {32, 64, 128, 256}`              | `results/ablation_nc_hidden_dim.csv`     | `figures/ablations_nc.pdf` (panel b) |
| `ablation_nc_bucket_hours.py`| sweep snapshot width ∈ {1, 3, 6, 12, 24} h     | `results/ablation_nc_bucket_hours.csv`   | `figures/ablations_nc.pdf` (panel c) |
| `ablation_nc_pos_weight.py`  | `none` / `sqrt` / `balanced` BCE pos_weight    | `results/ablation_nc_pos_weight.csv`     | `figures/ablations_nc.pdf` (panel d) |
| `plot_nc_results.py`         | reads every NC CSV, regenerates NC figure      | —                                        | `figures/ablations_nc.pdf` |

### Runner

| File | What it does |
|---|---|
| `run_ablations.sh` | Unified runner. Default: runs **both** LP and NC ablations, then regenerates every figure. |
| `_common.py`       | Shared LP utilities (`RunConfig`, `build_model`, `run_one`, `save_csv`). |
| `_nc_common.py`    | Shared NC utilities (mirror of `_common.py` for the JODIE pipeline). |

## Quick start

```bash
# Both tasks end-to-end:
bash run_ablations.sh

# Restrict to one task:
bash run_ablations.sh --task lp
bash run_ablations.sh --task nc

# Re-generate figures only, no training:
bash run_ablations.sh -p

# Skip the slow LP experiments (bootstrap CI, per-snapshot trace):
bash run_ablations.sh --task lp --skip-stat-sig --skip-traj
```

Other flags:

* `--lp-datasets ds1,ds2,...` — restrict the LP sensitivity ablations.
* `--nc-datasets ds1,ds2,...` — restrict the NC sensitivity ablations.

## What each experiment supports in the paper

### Link prediction
* **Hidden dimension** — `d_h = 128` is the Pareto knee.
* **Number of layers** — `L = 2` is the right depth (`L = 1` lacks
  two-hop reasoning, `L ≥ 3` over-smooths on the smaller benchmarks).
* **GNN backbone** — SiST-GNN's improvement is a property of the
  temporally augmented graph, not of any specific spatial operator.
* **Per-snapshot MRR** — visualises the live-update *trajectory* claim:
  SiST-GNN stays above ROLAND-GRU's leaderboard average essentially
  everywhere, not just on average.
* **Statistical significance** — non-parametric bootstrap 95 % CIs across
  5 seeds + one-sided p-values against ROLAND-GRU.

### Node classification
* **GNN backbone** — the SiST-GNN improvement carries across spatial
  operators on continuous-time JODIE streams.
* **Hidden dimension** — same `d_h = 128` Pareto behaviour as the LP
  ablation.
* **Bucket hours** — the snapshot-width / granularity trade-off.
* **BCE pos_weight** — `none` (plain BCE) follows the TGN/JODIE
  convention and gives the best ROC AUC on the imbalanced NC labels.

## Notes

* Default seeds: `{0, 1, 2}` for the sensitivity ablations, `{0, …, 4}`
  for the LP bootstrap CI.
* Every script writes to `results/` only. Figures are written to
  `figures/` only by `plot_results.py` and `plot_nc_results.py`.
* The shared `RunConfig` cache in `_nc_common.py` keeps the same dataset
  bundle in memory across a sweep so a 4-value hyper-param scan doesn't
  preprocess + reload the dataset 4 times.
