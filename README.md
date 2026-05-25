# SiST-GNN

**'Si'multaneous 'S'patial-'T'emporal Message Passing for Dynamic Graph Representation Learning**

A unified framework for dynamic graph representation learning that performs
*simultaneous* spatial and temporal message passing on a temporally
augmented graph. The same backbone is used for two downstream tasks:

| Task                         | Datasets                                                | Metric  |
|------------------------------|---------------------------------------------------------|---------|
| **Link Prediction (LP)**     | Bitcoin-Alpha, Bitcoin-OTC, UCI-Message, AS-733, Reddit-Title, Reddit-Body | MRR     |
| **Node Classification (NC)** | Wikipedia, Reddit, MOOC (JODIE)                          | ROC AUC |

---

## Repository layout

```
SiST-GNN/
├── temporal_gnn.py            # SiST-GNN backbone (LSTM + Transformer variants)
├── helper.py                  # seeding / JSON I/O / device utilities
│
├── main_lp.py                 # Link-prediction entrypoint
├── data_processing_lp.py      # Roland-style LP dataset loaders + snapshot bucketing
├── train_lp.py                # StatefulGNNWrapper + fixed-split / live-update loops
├── evaluate_lp.py             # MRR evaluation helpers
│
├── main_nc.py                 # Node-classification entrypoint
├── data_processing_nc.py      # JODIE NC loaders + benchtemp preprocessing + bucketing
├── train_nc.py                # NCStatefulWrapper + fixed-split training/eval
├── run_25seeds_nc.py          # 25-seed NC sweep used for the paper table
├── build_lp_table.py          # Build LP LaTeX tables (fixed-split + live-update)
├── build_nc_table.py          # Build NC LaTeX table from a 25-seed summary
│
├── run.sh                     # ★ Unified experiment runner (LP / NC / all)
│
├── datasets/
│   ├── roland/                # LP raw + extracted dataset files
│   └── nc/                    # NC raw + benchtemp-preprocessed (ml_*) files
│
├── results/                   # All experiment JSON logs + aggregated CSVs
│
└── ablations/                 # Sensitivity ablations + paper figures
    ├── run_ablations.sh       # ★ Unified ablation runner (LP / NC / all)
    ├── _common.py             # Shared LP-ablation utilities
    ├── _nc_common.py          # Shared NC-ablation utilities
    ├── ablation_*.py          # Individual sweeps (hidden dim, layers, backbone, ...)
    ├── per_snapshot_mrr.py    # Per-snapshot MRR trajectory under live-update
    ├── statistical_significance.py
    ├── plot_results.py        # LP figures
    ├── plot_nc_results.py     # NC figures
    ├── results/               # CSV outputs
    └── figures/               # PDF/PNG outputs
```

---

## Installation

```bash
# Python 3.10+ with CUDA-enabled torch is recommended.
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# benchtemp is needed for the JODIE node-classification preprocessing:
pip install benchtemp
```

`requirements.txt` lists `torch`, `torch_geometric`, `pandas`, `numpy`,
`scikit-learn`, `matplotlib`, and `tgb`. The NC pipeline additionally
needs `benchtemp`, which is installed separately.

---

## Reproducing the paper

Everything is driven by the single top-level **`run.sh`**:

```bash
# Run both LP and NC end-to-end across 3 seeds, 50 epochs:
bash run.sh

# Just one task:
bash run.sh --task lp
bash run.sh --task nc

# Quick smoke test: 2 epochs, fixed-split, 1 seed, 1 small dataset per task
bash run.sh --verify

# Hyper-parameter overrides:
bash run.sh --task lp  --seeds 5 --epochs 400 --lp-eval fixed-split
bash run.sh --task nc  --seeds 25 --epochs 50 --nc-datasets wikipedia,reddit
```

### Outputs

* Per-run JSON logs land in `results/exp_<timestamp>_lp.json` or
  `results/exp_<timestamp>_nc.json`.
* Aggregated mean/std across seeds is written to `results/lp_summary.csv`
  and `results/nc_summary.csv`.
* Paper-ready LaTeX tables:
    * **LP** — `python build_lp_table.py` reads `results/lp_summary.csv`
      and writes both fixed-split and live-update tables to
      `results/lp_results_tables.tex`.
    * **NC** — the 25-seed NC sweep (paper table) is launched with
      `python run_25seeds_nc.py`, and `python build_nc_table.py` then
      formats the LaTeX table to `results/nc_25seeds/results_table.tex`.

### Available flags

| Flag             | Default                                                       | Meaning |
|------------------|---------------------------------------------------------------|---------|
| `--task`         | `all`                                                         | `lp` / `nc` / `all` |
| `--seeds`        | `3`                                                           | Number of seeds (uses `0..N-1`) |
| `--epochs`       | `50`                                                          | Number of training epochs |
| `--verify`       | off                                                           | Smoke test: 2 epochs, 1 seed, small dataset |
| `--lp-eval`      | `both`                                                        | `fixed-split` / `live-update` / `both` |
| `--lp-datasets`  | `bitcoin-otc,bitcoin-alpha,uci-message,reddit-title,reddit-body,as-733` | Comma-separated subset |
| `--nc-datasets`  | `wikipedia,reddit,mooc`                                       | Comma-separated subset |
| `--out-dir`      | `results`                                                     | Where logs and summaries go |
| `--python`       | `python`                                                      | Interpreter to use |

---

## Ablations

Every sensitivity ablation is also unified:

```bash
cd ablations/

# Both tasks end-to-end, then regenerate all paper figures:
bash run_ablations.sh

# Restrict to one task:
bash run_ablations.sh --task lp
bash run_ablations.sh --task nc

# Skip slow experiments:
bash run_ablations.sh --task lp --skip-stat-sig --skip-traj

# Regenerate figures only (no retraining):
bash run_ablations.sh -p
```

The ablations cover:

* **LP** — hidden dimension, number of layers, GNN backbone,
  per-snapshot MRR trajectory, 5-seed bootstrap CI.
* **NC** — GNN backbone (GCN / SAGE / GATv2), hidden dimension,
  bucket-width (snapshot granularity), and BCE positive-class weight.

Outputs land in `ablations/results/*.csv` and `ablations/figures/*.pdf`.

---

## Direct script usage

The unified runners are thin wrappers around two Python entrypoints; both
expose the full hyper-parameter surface for fine-grained experimentation:

```bash
# LP (single config)
python main_lp.py \
    --dataset bitcoin-otc --eval-method fixed-split \
    --model lstm --gnn-type GCNConv --num-layers 2 \
    --hidden-dim 128 --num-epochs 200 --seed 0

# NC (single config)
python main_nc.py \
    --dataset wikipedia --model lstm --gnn-type GCNConv \
    --num-layers 2 --hidden-dim 128 --bucket-hours 6 \
    --num-epochs 50 --patience 5 --seed 0
```

---
