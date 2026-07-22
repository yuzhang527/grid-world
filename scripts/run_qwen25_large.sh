#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODEL_SIZE=32b GPUS=0,1,2,3 MAPS=... RUN=... bash scripts/run_qwen25_large.sh
#   MODEL_SIZE=72b GPUS=0,1,2,3,4,5,6,7 MAPS=... RUN=... bash scripts/run_qwen25_large.sh
#
# Stages can be overridden:
#   STAGES=generate,validate,targets,activations,probes,report

MODEL_SIZE="${MODEL_SIZE:-32b}"
GPUS="${GPUS:-0,1,2,3}"
MAPS="${MAPS:-data/generated/grid5x5_diverse_1000.jsonl}"
RUN="${RUN:-runs/qwen25_${MODEL_SIZE}_diverse5x5_1000}"
STAGES="${STAGES:-generate,validate,targets,activations,probes,report}"
POSITIONS="${POSITIONS:-default}"
LAYERS="${LAYERS:-all}"
PROBE_GROUPS="${PROBE_GROUPS:-local,cells,planning}"
PROBE_SPLITS="${PROBE_SPLITS:-20}"
PROBE_EPOCHS="${PROBE_EPOCHS:-60}"
PROBE_OVERWRITE="${PROBE_OVERWRITE:-0}"
ACTIVATION_BATCH_SIZE="${ACTIVATION_BATCH_SIZE:-1}"
DEVICE_MAP="${DEVICE_MAP:-balanced}"
MAX_MEMORY="${MAX_MEMORY:-}"

case "$MODEL_SIZE" in
  32b)
    MODEL="Qwen/Qwen2.5-32B-Instruct"
    CONFIG="configs/experiments/qwen25_32b_strategy_a.yaml"
    ;;
  72b)
    MODEL="Qwen/Qwen2.5-72B-Instruct"
    CONFIG="configs/experiments/qwen25_72b_strategy_a.yaml"
    ;;
  *)
    echo "MODEL_SIZE must be 32b or 72b" >&2
    exit 2
    ;;
esac

contains_stage() {
  [[ ",${STAGES}," == *",$1,"* ]]
}

echo "MODEL=$MODEL"
echo "GPUS=$GPUS"
echo "MAPS=$MAPS"
echo "RUN=$RUN"
echo "STAGES=$STAGES"

if contains_stage generate; then
  # Tensor parallel keeps one logical model across all selected GPUs.
  grid-world trajectories generate \
    --config "$CONFIG" \
    --maps "$MAPS" \
    --run "$RUN" \
    --gpus "$GPUS" \
    --parallel-mode tensor
fi

if contains_stage validate; then
  grid-world trajectories validate --run "$RUN"
  grid-world trajectories summarize --run "$RUN"
fi

if contains_stage targets; then
  grid-world targets build --run "$RUN"
fi

if contains_stage activations; then
  EXTRA_MEMORY_ARGS=()
  if [[ -n "$MAX_MEMORY" ]]; then
    EXTRA_MEMORY_ARGS+=(--max-memory "$MAX_MEMORY")
  fi

  python scripts/extract_activations_model_parallel.py \
    --run "$RUN" \
    --model "$MODEL" \
    --gpus "$GPUS" \
    --device-map "$DEVICE_MAP" \
    --dtype bf16 \
    --layers "$LAYERS" \
    --positions "$POSITIONS" \
    --batch-size "$ACTIVATION_BATCH_SIZE" \
    --overwrite \
    "${EXTRA_MEMORY_ARGS[@]}"
fi

if contains_stage probes; then
  PROBE_EXTRA_ARGS=()
  if [[ "$PROBE_OVERWRITE" == "1" ]]; then
    PROBE_EXTRA_ARGS+=(--overwrite)
  fi

  python scripts/train_probes_multigpu.py \
    --run "$RUN" \
    --groups "$PROBE_GROUPS" \
    --positions auto \
    --layers all \
    --gpus "$GPUS" \
    --splits "$PROBE_SPLITS" \
    --epochs "$PROBE_EPOCHS" \
    --output-subdir probes_multigpu \
    "${PROBE_EXTRA_ARGS[@]}"
fi

if contains_stage report; then
  cat "$RUN/probes_multigpu/summary.md"
fi

