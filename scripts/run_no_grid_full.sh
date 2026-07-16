#!/usr/bin/env bash
set -Eeuo pipefail

MAPS="${MAPS:-data/generated/grid5x5_diverse_1000.jsonl}"
MAP_CONFIG="${MAP_CONFIG:-configs/maps/grid5x5_diverse_1000.yaml}"
RUN="${RUN:-runs/qwen25_7b_no_grid_diverse5x5_1000}"
CONFIG="${CONFIG:-configs/experiments/qwen25_7b_no_grid_diverse5x5.yaml}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
GPUS="${GPUS:-0,1,2,3}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SPLITS="${SPLITS:-20}"
EPOCHS="${EPOCHS:-60}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"
SKIP_ACTIVATIONS="${SKIP_ACTIVATIONS:-0}"
SKIP_PROBES="${SKIP_PROBES:-0}"
EXPLICIT_ALL_LAYER_RUN="${EXPLICIT_ALL_LAYER_RUN:-runs/qwen25_7b_diverse5x5_1000_explicit_all_layers}"

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

if [[ ! -f "$MAPS" ]]; then
  echo "=== Generate the shared diverse map set ==="
  grid-world maps generate \
    --config "$MAP_CONFIG" \
    --output "$MAPS"
fi

if [[ "$SKIP_GENERATION" != "1" ]]; then
  echo "=== Generate no-grid trajectories on four GPUs ==="
  grid-world trajectories generate \
    --config "$CONFIG" \
    --maps "$MAPS" \
    --run "$RUN" \
    --gpus "$GPUS" \
    --parallel-mode data \
    --resume
fi

echo "=== Validate and build observable/true-map targets ==="
grid-world trajectories validate --run "$RUN"
grid-world targets build --run "$RUN"

if [[ "$SKIP_ACTIVATIONS" != "1" ]]; then
  echo "=== Extract all hidden-state indices for no-grid ==="
  rm -rf "$RUN/activations"
  grid-world activations extract \
    --run "$RUN" \
    --model "$MODEL" \
    --layers all \
    --positions prompt_last,pre_action_token,mean_history \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE"
fi

if [[ "$SKIP_PROBES" != "1" ]]; then
  echo "=== Train no-grid all-layer probes ==="
  rm -rf "$RUN/probes"
  grid-world probes train \
    --run "$RUN" \
    --groups cells,memory,true_cells,true_cells_unobserved,planning \
    --positions prompt_last,pre_action_token,mean_history \
    --layers all \
    --backend torch \
    --device "$DEVICE" \
    --splits "$SPLITS" \
    --epochs "$EPOCHS" \
    --min-class-count 5

  grid-world probes report --run "$RUN"
fi

echo "=== Plot no-grid layer curves ==="
rm -rf "$RUN/layer_curves"
python scripts/plot_layer_curves.py \
  --run "$RUN" \
  --condition-label "No explicit grid"

if [[ -f "$EXPLICIT_ALL_LAYER_RUN/probes/probe_results.csv" ]]; then
  echo "=== Compare explicit and no-grid conditions ==="
  rm -rf "$RUN/condition_comparison"
  python scripts/compare_layer_conditions.py \
    --explicit-run "$EXPLICIT_ALL_LAYER_RUN" \
    --no-grid-run "$RUN" \
    --positions prompt_last,pre_action_token \
    --groups cells,planning,true_cells,true_cells_unobserved \
    --output "$RUN/condition_comparison"
else
  echo "Explicit all-layer results not found; skipping condition comparison."
fi

echo
echo "No-grid experiment complete."
echo "Behavior summary: $RUN/summary.json"
echo "Layer report: $RUN/layer_curves/summary.md"
echo "Condition comparison: $RUN/condition_comparison/summary.md"
