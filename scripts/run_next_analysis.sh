#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${RUN:-runs/qwen25_7b_diverse5x5_1000}"
DEVICE="${DEVICE:-cuda:0}"
SPLITS="${SPLITS:-20}"
EPOCHS="${EPOCHS:-60}"

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "=== 1. Behavior stratification ==="
python scripts/analyze_run_quality.py --run "$RUN"

echo "=== 2. Create legal and matched probe views ==="
python scripts/make_probe_views.py --run "$RUN"

VIEW_ROOT="$RUN/probe_views"

run_probe() {
  local name="$1"
  local view="$VIEW_ROOT/$name"
  echo "=== Probe view: $name ==="
  rm -rf "$view/probes"
  grid-world probes train \
    --run "$view" \
    --groups cells,planning \
    --positions pre_action_token \
    --layers 21 \
    --backend torch \
    --device "$DEVICE" \
    --splits "$SPLITS" \
    --epochs "$EPOCHS" \
    --min-class-count 5
  grid-world probes report --run "$view"
}

# One position × one layer: 20 fits per view instead of 400.
run_probe legal_all
run_probe matched_success_legal
run_probe matched_failure_legal

echo "=== 3. Compare fixed pre-action L21 results ==="
python scripts/compare_probe_views.py \
  --view all="$VIEW_ROOT/legal_all" \
  --view success="$VIEW_ROOT/matched_success_legal" \
  --view failure="$VIEW_ROOT/matched_failure_legal" \
  --position pre_action_token \
  --layer 21 \
  --output "$RUN/analysis/fixed_pre_action_L21"

echo
echo "Done."
echo "Behavior report:"
echo "  $RUN/analysis/behavior_quality/summary.md"
echo "Probe comparison:"
echo "  $RUN/analysis/fixed_pre_action_L21/comparison.md"
