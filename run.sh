#!/usr/bin/env bash
# =============================================================================
#  SiST-GNN unified experiment runner.
#
#  Paper: "SiST-GNN: Simultaneous Spatial-Temporal Message Passing for
#          Dynamic Graph Representation Learning".
#
#  This single script reproduces every headline number in the paper:
#    --task lp   : dynamic Link Prediction        (Roland datasets)
#    --task nc   : dynamic Node Classification    (JODIE datasets)
#    --task all  : both                            (default)
#
#  Flags
#    --task lp|nc|all          Which experiment to run. Default: all.
#    --seeds N                 Run with seeds 0..N-1 (default: 3).
#    --epochs N                Number of training epochs (default: 50).
#    --verify                  Smoke-test mode: 2 epochs, one small dataset
#                              per task (uci-message LP + wikipedia NC).
#    --lp-eval METHOD          fixed-split | live-update | both. Default: both.
#    --lp-datasets ds          Comma-separated LP dataset subset.
#    --nc-datasets ds          Comma-separated NC dataset subset.
#    --out-dir DIR             Where to drop the JSON logs and summaries
#                              (default: results).
#    --python PATH             Python interpreter (default: python).
#    -h | --help               Print this header and exit.
#
#  Outputs
#    results/exp_<ts>_lp.json     One file per LP run (per dataset/eval)
#    results/exp_<ts>_nc.json     One file per NC run
#    results/lp_summary.csv       Aggregated mean +/- std across seeds
#    results/nc_summary.csv
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

TASK="all"
NUM_SEEDS=3
NUM_EPOCHS=50
VERIFY=0
LP_EVAL="both"
LP_DATASETS_DEFAULT="bitcoin-otc,bitcoin-alpha,uci-message,reddit-title,reddit-body,as-733"
NC_DATASETS_DEFAULT="wikipedia,reddit,mooc"
LP_DATASETS=""
NC_DATASETS=""
OUT_DIR="results"
PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)         TASK="$2"; shift 2 ;;
    --seeds)        NUM_SEEDS="$2"; shift 2 ;;
    --epochs)       NUM_EPOCHS="$2"; shift 2 ;;
    --verify)       VERIFY=1; shift ;;
    --lp-eval)      LP_EVAL="$2"; shift 2 ;;
    --lp-datasets)  LP_DATASETS="$2"; shift 2 ;;
    --nc-datasets)  NC_DATASETS="$2"; shift 2 ;;
    --out-dir)      OUT_DIR="$2"; shift 2 ;;
    --python)       PYTHON="$2"; shift 2 ;;
    -h|--help)      sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# //'; exit 0 ;;
    *) echo "Unknown flag: $1  (try --help)" >&2; exit 1 ;;
  esac
done

if [[ $VERIFY -eq 1 ]]; then
  NUM_EPOCHS=2
  NUM_SEEDS=1
  LP_EVAL="fixed-split"
  [[ -z "$LP_DATASETS" ]] && LP_DATASETS="uci-message"
  [[ -z "$NC_DATASETS" ]] && NC_DATASETS="wikipedia"
fi

[[ -z "$LP_DATASETS" ]] && LP_DATASETS="$LP_DATASETS_DEFAULT"
[[ -z "$NC_DATASETS" ]] && NC_DATASETS="$NC_DATASETS_DEFAULT"

mkdir -p "$OUT_DIR"

banner() {
  echo ""
  echo "=========================================================="
  echo "  $1"
  echo "=========================================================="
}

# -----------------------------------------------------------------------------
# Link Prediction
# -----------------------------------------------------------------------------
run_lp() {
  banner "SiST-GNN  |  Link Prediction"
  echo "  datasets: $LP_DATASETS"
  echo "  eval    : $LP_EVAL    seeds=$NUM_SEEDS    epochs=$NUM_EPOCHS"

  IFS=',' read -r -a lp_ds_arr <<< "$LP_DATASETS"
  for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    banner "LP  ::  seed $seed / $NUM_SEEDS"
    for ds in "${lp_ds_arr[@]}"; do
      if [[ "$LP_EVAL" == "fixed-split" || "$LP_EVAL" == "both" ]]; then
        echo ">> LP fixed-split  dataset=$ds  seed=$seed"
        $PYTHON main_lp.py \
          --eval-method fixed-split --dataset "$ds" \
          --num-epochs "$NUM_EPOCHS" --seed "$seed" \
          --out-dir "$OUT_DIR"
      fi
      if [[ "$LP_EVAL" == "live-update" || "$LP_EVAL" == "both" ]]; then
        echo ">> LP live-update  dataset=$ds  seed=$seed"
        $PYTHON main_lp.py \
          --eval-method live-update --dataset "$ds" \
          --num-epochs "$NUM_EPOCHS" --seed "$seed" \
          --out-dir "$OUT_DIR"
      fi
    done
  done

  banner "LP  ::  Aggregating across seeds → $OUT_DIR/lp_summary.csv"
  $PYTHON - <<PY
import csv, glob, json, os
from collections import defaultdict
from statistics import mean, stdev

agg = defaultdict(list)
for f in sorted(glob.glob(os.path.join("$OUT_DIR", "exp_*_lp.json"))):
    with open(f) as fh:
        data = json.load(fh)
    method = data.get("eval_method", "?")
    for ds, res in data.get("results", {}).get("lp", {}).items():
        if "mrr" in res:
            agg[(ds, method)].append(float(res["mrr"]))

out = os.path.join("$OUT_DIR", "lp_summary.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Dataset", "Eval_Method", "Mean_MRR", "Std_MRR", "N_Runs"])
    for (ds, method), vals in sorted(agg.items()):
        m = mean(vals)
        s = stdev(vals) if len(vals) > 1 else 0.0
        w.writerow([ds, method, f"{m:.6f}", f"{s:.6f}", len(vals)])
print(f"Wrote {out} ({len(agg)} rows)")
PY
}

# -----------------------------------------------------------------------------
# Node Classification
# -----------------------------------------------------------------------------
run_nc() {
  banner "SiST-GNN  |  Node Classification"
  echo "  datasets: $NC_DATASETS"
  echo "  seeds=$NUM_SEEDS    epochs=$NUM_EPOCHS"

  IFS=',' read -r -a nc_ds_arr <<< "$NC_DATASETS"
  for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    banner "NC  ::  seed $seed / $NUM_SEEDS"
    for ds in "${nc_ds_arr[@]}"; do
      echo ">> NC  dataset=$ds  seed=$seed"
      $PYTHON main_nc.py \
        --dataset "$ds" --model lstm --gnn-type GCNConv \
        --num-layers 2 --hidden-dim 128 \
        --num-epochs "$NUM_EPOCHS" --patience 5 \
        --bucket-hours 6 --seed "$seed" \
        --out-dir "$OUT_DIR"
    done
  done

  banner "NC  ::  Aggregating across seeds → $OUT_DIR/nc_summary.csv"
  $PYTHON - <<PY
import csv, glob, json, os
from collections import defaultdict
from statistics import mean, stdev

agg_test = defaultdict(list)
agg_val  = defaultdict(list)
for f in sorted(glob.glob(os.path.join("$OUT_DIR", "exp_*_nc.json"))):
    with open(f) as fh:
        data = json.load(fh)
    for ds, res in data.get("results", {}).items():
        if "test_auc" in res:
            agg_test[ds].append(float(res["test_auc"]))
            agg_val[ds].append(float(res.get("val_auc", 0.0)))

out = os.path.join("$OUT_DIR", "nc_summary.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Dataset", "Mean_Test_AUC", "Std_Test_AUC",
                "Mean_Val_AUC", "Std_Val_AUC", "N_Runs"])
    for ds in sorted(agg_test):
        ts, vs = agg_test[ds], agg_val[ds]
        w.writerow([ds,
                    f"{mean(ts):.6f}", f"{stdev(ts) if len(ts) > 1 else 0.0:.6f}",
                    f"{mean(vs):.6f}", f"{stdev(vs) if len(vs) > 1 else 0.0:.6f}",
                    len(ts)])
print(f"Wrote {out} ({len(agg_test)} rows)")
PY
}

case "$TASK" in
  lp)  run_lp ;;
  nc)  run_nc ;;
  all) run_lp; run_nc ;;
  *)   echo "Unknown --task: $TASK (use lp|nc|all)" >&2; exit 1 ;;
esac

banner "Done. Logs in $OUT_DIR/"
