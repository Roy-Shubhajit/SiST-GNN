#!/usr/bin/env bash
# =============================================================================
#  SiST-GNN unified ablation runner.
#
#  Runs every sensitivity ablation for both tasks and regenerates the paper
#  figures.
#
#    LP ablations (data_processing_lp.py / train_lp.py):
#      1. Hidden-dimension sweep
#      2. Number-of-layers sweep
#      3. GNN-backbone comparison
#      4. Per-snapshot MRR trajectory
#      5. Statistical-significance test (5-seed bootstrap CI)
#
#    NC ablations (data_processing_nc.py / train_nc.py):
#      1. GNN backbone (GCN/SAGE/GATv2)
#      2. Hidden-dimension sweep
#      3. Bucket-hours sweep
#      4. BCE positive-class weight
#
#  Flags
#    --task lp|nc|all      Restrict to one task or run both. Default: all.
#    -p / --plots-only     Skip every ablation, just regenerate figures from
#                          whatever CSVs are already on disk.
#    -s / --skip-stat-sig  Skip the LP bootstrap significance test (slow).
#    -t / --skip-traj      Skip the LP per-snapshot MRR trajectory (slow).
#    -d / --lp-datasets ds   Comma-separated LP dataset subset
#                            (default: bitcoin-otc,bitcoin-alpha,uci-message).
#    -D / --nc-datasets ds   Comma-separated NC dataset subset
#                            (default: wikipedia,reddit,mooc).
#    -h / --help            Print this header and exit.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p results figures

TASK="all"
RUN_PLOTS=1
RUN_TRAINING=1
RUN_STATSIG=1
RUN_PERSNAP=1
LP_DATASETS="bitcoin-otc,bitcoin-alpha,uci-message"
NC_DATASETS="wikipedia,reddit,mooc"
PYTHON="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)             TASK="$2"; shift 2 ;;
    -p|--plots-only)    RUN_TRAINING=0; shift ;;
    -s|--skip-stat-sig) RUN_STATSIG=0; shift ;;
    -t|--skip-traj)     RUN_PERSNAP=0; shift ;;
    -d|--lp-datasets)   LP_DATASETS="$2"; shift 2 ;;
    -D|--nc-datasets)   NC_DATASETS="$2"; shift 2 ;;
    -h|--help)          sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# //'; exit 0 ;;
    *) echo "Unknown flag: $1  (try --help)" >&2; exit 1 ;;
  esac
done

step() {
  echo ""
  echo "=========================================================="
  echo "  $1"
  echo "=========================================================="
}

run_lp() {
  if [[ $RUN_TRAINING -eq 1 ]]; then
    step "LP ablation 1/3 -- Hidden dimension"
    $PYTHON ablation_hidden_dim.py    --datasets "$LP_DATASETS"
    step "LP ablation 2/3 -- Number of layers"
    $PYTHON ablation_num_layers.py    --datasets "$LP_DATASETS"
    step "LP ablation 3/3 -- GNN backbone"
    $PYTHON ablation_gnn_backbone.py  --datasets "$LP_DATASETS"

    if [[ $RUN_PERSNAP -eq 1 ]]; then
      step "LP per-snapshot MRR trajectory"
      $PYTHON per_snapshot_mrr.py --datasets "bitcoin-otc,bitcoin-alpha"
    fi

    if [[ $RUN_STATSIG -eq 1 ]]; then
      step "LP statistical-significance test"
      $PYTHON statistical_significance.py \
        --datasets "as-733,reddit-title,reddit-body,uci-message,bitcoin-otc,bitcoin-alpha"
    fi
  fi
  if [[ $RUN_PLOTS -eq 1 ]]; then
    step "Regenerating LP figure PDFs"
    $PYTHON plot_results.py
  fi
}

run_nc() {
  if [[ $RUN_TRAINING -eq 1 ]]; then
    step "NC ablation 1/4 -- GNN backbone (GCN/SAGE/GATv2)"
    $PYTHON ablation_nc_gnn_backbone.py --datasets "$NC_DATASETS"
    step "NC ablation 2/4 -- Hidden dimension"
    $PYTHON ablation_nc_hidden_dim.py   --datasets "$NC_DATASETS"
    step "NC ablation 3/4 -- Bucket hours"
    $PYTHON ablation_nc_bucket_hours.py --datasets "$NC_DATASETS"
    step "NC ablation 4/4 -- BCE positive-class weight"
    $PYTHON ablation_nc_pos_weight.py   --datasets "$NC_DATASETS"
  fi
  if [[ $RUN_PLOTS -eq 1 ]]; then
    step "Regenerating NC figure PDF"
    $PYTHON plot_nc_results.py
  fi
}

case "$TASK" in
  lp)  run_lp ;;
  nc)  run_nc ;;
  all) run_lp; run_nc ;;
  *)   echo "Unknown --task: $TASK (use lp|nc|all)" >&2; exit 1 ;;
esac

echo ""
echo "Done. Figures are under ${HERE}/figures/"
ls -la figures/ || true
