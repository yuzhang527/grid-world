#!/usr/bin/env bash
set -euo pipefail
MAPS="${MAPS:-data/generated/grid5x5_100.jsonl}"
RUN="${RUN:-runs/qwen25_7b_100}"
CONFIG="${CONFIG:-configs/experiments/qwen25_7b_strategy_a.yaml}"
GPUS="${GPUS:-0,1,2,3}"
grid-world trajectories generate --config "$CONFIG" --maps "$MAPS" --run "$RUN"   --gpus "$GPUS" --parallel-mode data
grid-world trajectories validate --run "$RUN"
grid-world trajectories summarize --run "$RUN"
