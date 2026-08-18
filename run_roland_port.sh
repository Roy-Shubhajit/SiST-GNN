#!/usr/bin/env bash
# Sweep over the ported ROLAND pipeline (roland_pipeline.py + main_roland.py).
#
#   MODE=live_update_fixed_split DATASETS=uci-message ./run_roland_port.sh
#
# Varies decoder x edge-features. Sequential (JOBS=1) -- the GPU is shared.
# Results: results/roland_port/<mode>/<decoder>_<edgefeat>/<dataset>/seed<k>/

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-100}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MODE="${MODE:-live_update_fixed_split}"
DATASETS="${DATASETS:-uci-message}"
DECODERS="${DECODERS:-dot mlp}"
EDGE_FEATS="${EDGE_FEATS:-learned none}"
NODE_FEATS="${NODE_FEATS:-learned constant}"
# Pinned to the configuration established by the earlier ablation, so cells
# here are comparable with the results/lp_roland_faithful runs.
EXTRA="${EXTRA:---train-num-neg 20 --dropout 0.1 --weight-decay 1e-5}"
# Matched data setting across fixed-split and live-update.
UNDIRECTED="${UNDIRECTED:-auto}"
MIN_EDGES="${MIN_EDGES:--1}"
LOSS="${LOSS:-margin}"
ROOT="${ROOT:-results/roland_port}"

mkdir -p "$ROOT/logs"
for ds in $DATASETS; do
  for dec in $DECODERS; do
    for nf in $NODE_FEATS; do
      for ef in $EDGE_FEATS; do
        for seed in $SEEDS; do
          tag="${dec}_node-${nf}_edge-${ef}"
          out="$ROOT/$MODE${UNDIRECTED:+_und-$UNDIRECTED}${LOSS:+_loss-$LOSS}/$tag/$ds/seed$seed"
          log="$ROOT/logs/${MODE}_${tag}_${ds}_seed${seed}.log"
          if compgen -G "$out/exp_*_lp.json" > /dev/null; then
            echo "[skip] $MODE $tag $ds seed=$seed"; continue
          fi
          mkdir -p "$out"
          echo "[run ] $MODE $tag $ds seed=$seed  ($(date +%H:%M:%S))"
          $PYTHON main_roland.py \
            --dataset "$ds" --mode "$MODE" \
            --decoder "$dec" --edge-features "$ef" --node-features "$nf" \
            $EXTRA --undirected "$UNDIRECTED" --min-edges "$MIN_EDGES" --loss "$LOSS" \
            --num-epochs "$EPOCHS" --seed "$seed" \
            --out-dir "$out" > "$log" 2>&1 \
            && echo "       -> $(grep -oE 'MRR = [0-9.]+' "$log" | tail -1)" \
            || { echo "       !! FAILED"; tail -3 "$log"; }
        done
      done
    done
  done
done
echo "=== sweep complete ($(date +%H:%M:%S)) ==="
