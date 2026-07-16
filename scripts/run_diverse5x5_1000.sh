#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$(pwd)}"
cd "$ROOT"

CONFIG="${CONFIG:-configs/pipeline/qwen25_7b_diverse5x5_1000.yaml}"
MAPS="${MAPS:-data/generated/grid5x5_diverse_1000.jsonl}"
RUN="${RUN:-runs/qwen25_7b_diverse5x5_1000}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
GPUS="${GPUS:-0,1,2,3}"
LOG_DIR="${LOG_DIR:-$RUN/logs}"

mkdir -p "$LOG_DIR"

echo "=== Stage 1: diverse maps + 4-GPU trajectories + targets ==="
/usr/bin/time -p grid-world pipeline run \
  --config "$CONFIG" \
  --stages maps,generate,validate,targets \
  2>&1 | tee "$LOG_DIR/stage1_generation.log"

echo
echo "=== Map distribution ==="
grid-world maps summarize --maps "$MAPS" \
  2>&1 | tee "$LOG_DIR/map_distribution.log"

echo
echo "=== Stage 2: activations ==="
/usr/bin/time -p grid-world activations extract \
  --run "$RUN" \
  --model "$MODEL" \
  --layers auto \
  --positions default \
  --device cuda:0 \
  2>&1 | tee "$LOG_DIR/stage2_activations.log"

echo
echo "=== Stage 3: probes + report ==="
/usr/bin/time -p grid-world probes train \
  --run "$RUN" \
  --groups local,cells,planning \
  --positions auto \
  --layers all \
  --backend torch \
  --device cuda:0 \
  --splits 20 \
  --epochs 60 \
  2>&1 | tee "$LOG_DIR/stage3_probes.log"

grid-world probes report --run "$RUN" \
  2>&1 | tee "$LOG_DIR/stage3_report.log"

echo
echo "Completed."
echo "Map summary:  ${MAPS%.jsonl}.summary.json"
echo "Behavior:     $RUN/summary.json"
echo "Probe report: $RUN/probes/summary.md"
echo "Best tasks:   $RUN/probes/best_by_task.csv"
