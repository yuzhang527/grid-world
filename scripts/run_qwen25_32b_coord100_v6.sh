#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
GPUS="${GPUS:-0,1,2,3}"
NUM_EPISODES="${NUM_EPISODES:-100}"
SAMPLE_SEED="${SAMPLE_SEED:-20260722}"
SOURCE_MAPS="${SOURCE_MAPS:-data/generated/grid5x5_diverse_1000.jsonl}"
MAPS="${MAPS:-data/generated/grid5x5_diverse_v6_100_seed${SAMPLE_SEED}.jsonl}"
RUN="${RUN:-runs/qwen25_32b_coordbelief_v6_100}"
STAGES="${STAGES:-sample,generate,targets,activations,probes,viewer}"
ACTIVATION_LAYERS="${ACTIVATION_LAYERS:-all}"
PROBE_SPLITS="${PROBE_SPLITS:-5}"
PROBE_EPOCHS="${PROBE_EPOCHS:-60}"
OVERWRITE_RUN="${OVERWRITE_RUN:-0}"
OVERWRITE_TARGETS="${OVERWRITE_TARGETS:-1}"
VIEWER_ALL="${VIEWER_ALL:-1}"

has_stage() {
  [[ ",${STAGES}," == *",$1,"* ]] || [[ ",${STAGES}," == *",all,"* ]]
}

if [[ "$NUM_EPISODES" != "100" ]]; then
  echo "[v6] This audited preset is fixed to NUM_EPISODES=100; got $NUM_EPISODES" >&2
  exit 2
fi
if [[ ! -f scripts/run_qwen25_32b_coord200_v4.sh ]]; then
  echo "[v6] Missing scripts/run_qwen25_32b_coord200_v4.sh. Deploy the existing Coordinate-Belief generator first." >&2
  exit 2
fi
if [[ ! -f scripts/extract_activations_model_parallel.py ]]; then
  echo "[v6] Missing scripts/extract_activations_model_parallel.py." >&2
  exit 2
fi
if [[ ! -f scripts/train_probes_multigpu.py ]]; then
  echo "[v6] Missing scripts/train_probes_multigpu.py." >&2
  exit 2
fi

printf '%s\n' \
  "============================================================" \
  "Coordinate-Belief v6 / strict 100" \
  "MODEL=$MODEL" \
  "GPUS=$GPUS" \
  "SOURCE_MAPS=$SOURCE_MAPS" \
  "MAPS=$MAPS" \
  "RUN=$RUN" \
  "STAGES=$STAGES" \
  "PROBES=cells only" \
  "POSITION=prompt_last only" \
  "============================================================"

if has_stage sample; then
  python scripts/sample_maps_stratified_v6.py \
    --source "$SOURCE_MAPS" \
    --output "$MAPS" \
    --count 100 \
    --seed "$SAMPLE_SEED"
fi

if has_stage generate; then
  if [[ -e "$RUN/steps.jsonl" && "$OVERWRITE_RUN" == "1" ]]; then
    case "$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$RUN")" in
      */runs/*v6*) rm -rf "$RUN" ;;
      *) echo "[v6] Refusing to delete suspicious RUN=$RUN" >&2; exit 2 ;;
    esac
  elif [[ -e "$RUN/steps.jsonl" ]]; then
    echo "[v6] $RUN/steps.jsonl already exists. Set OVERWRITE_RUN=1 to regenerate." >&2
    exit 2
  fi
  MODEL="$MODEL" \
  GPUS="$GPUS" \
  NUM_EPISODES=100 \
  MAPS="$MAPS" \
  RUN="$RUN" \
  STAGES=generate,validate \
  bash scripts/run_qwen25_32b_coord200_v4.sh
fi

if has_stage targets; then
  extra=()
  [[ "$OVERWRITE_TARGETS" == "1" ]] && extra+=(--overwrite)
  python scripts/build_coordbelief_targets_v6.py \
    --run "$RUN" \
    "${extra[@]}"
fi

if has_stage activations; then
  python scripts/extract_activations_model_parallel.py \
    --run "$RUN" \
    --model "$MODEL" \
    --gpus "$GPUS" \
    --device-map balanced \
    --dtype bf16 \
    --layers "$ACTIVATION_LAYERS" \
    --positions prompt_last \
    --batch-size 1 \
    --overwrite
fi

if has_stage probes; then
  python scripts/train_probes_multigpu.py \
    --run "$RUN" \
    --groups cells \
    --positions prompt_last \
    --layers all \
    --gpus "$GPUS" \
    --splits "$PROBE_SPLITS" \
    --epochs "$PROBE_EPOCHS" \
    --output-subdir probes_multigpu \
    --overwrite
  rm -rf "$RUN/probes"
  ln -s probes_multigpu "$RUN/probes"
fi

if has_stage viewer; then
  viewer_args=(--run "$RUN" --position prompt_last --layer auto --folds 5)
  [[ "$VIEWER_ALL" == "1" ]] && viewer_args+=(--all-episodes)
  python scripts/trajectory_probe_viewer_coordbelief_v6.py "${viewer_args[@]}"
fi

echo
printf '%s\n' \
  "Coordinate-Belief v6 finished." \
  "Run:            $RUN" \
  "Sample summary: ${MAPS%.jsonl}.sample_summary.json" \
  "Target audit:   $RUN/coordbelief_v6_target_audit.json" \
  "Probe report:   $RUN/probes_multigpu/summary.md" \
  "Viewer:         $RUN/trajectory_viewer_coordbelief_v6/index.html"
