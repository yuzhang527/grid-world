#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_RUN="${SOURCE_RUN:-runs/qwen25_7b_diverse5x5_1000}"
VIEW_RUN="${VIEW_RUN:-runs/qwen25_7b_diverse5x5_1000_explicit_all_layers}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SPLITS="${SPLITS:-20}"
EPOCHS="${EPOCHS:-60}"
RESET="${RESET:-0}"
SKIP_ACTIVATIONS="${SKIP_ACTIVATIONS:-0}"
SKIP_PROBES="${SKIP_PROBES:-0}"

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

PREPARE_ARGS=(
  --source-run "$SOURCE_RUN"
  --view-run "$VIEW_RUN"
)
if [[ "$RESET" == "1" ]]; then
  PREPARE_ARGS+=(--reset-derived)
fi
python scripts/prepare_analysis_view.py "${PREPARE_ARGS[@]}"

echo "=== Build extended targets inside the analysis view ==="
rm -rf "$VIEW_RUN/targets"
grid-world targets build --run "$VIEW_RUN"

if [[ "$SKIP_ACTIVATIONS" != "1" ]]; then
  echo "=== Extract all 29 hidden-state indices ==="
  rm -rf "$VIEW_RUN/activations"
  grid-world activations extract \
    --run "$VIEW_RUN" \
    --model "$MODEL" \
    --layers all \
    --positions prompt_last,pre_action_token,mean_current_belief_grid \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE"
fi

if [[ "$SKIP_PROBES" != "1" ]]; then
  echo "=== Train all-layer probes ==="
  rm -rf "$VIEW_RUN/probes"
  grid-world probes train \
    --run "$VIEW_RUN" \
    --groups cells,explicit_cells,true_cells,true_cells_unobserved,planning \
    --positions prompt_last,pre_action_token,mean_current_belief_grid \
    --layers all \
    --backend torch \
    --device "$DEVICE" \
    --splits "$SPLITS" \
    --epochs "$EPOCHS" \
    --min-class-count 5

  grid-world probes report --run "$VIEW_RUN"
fi

echo "=== Plot fixed-task macro-F1 layer curves ==="
rm -rf "$VIEW_RUN/layer_curves"
python scripts/plot_layer_curves.py \
  --run "$VIEW_RUN" \
  --condition-label "Explicit belief"

echo
echo "Explicit all-layer experiment complete."
echo "Report: $VIEW_RUN/layer_curves/summary.md"
