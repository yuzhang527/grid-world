#!/usr/bin/env bash
set -euo pipefail

# Coordinate-Belief v4:
# Qwen2.5-32B-Instruct, 200 episodes, coordinate-set belief output.
#
# This is an additive pipeline.  It writes to a new run directory and never
# deletes or rewrites the older explicit-grid runs.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
MODEL_SIZE_LABEL="${MODEL_SIZE_LABEL:-32b}"
GPUS="${GPUS:-0,1,2,3}"
NUM_EPISODES="${NUM_EPISODES:-200}"
MAX_STEPS="${MAX_STEPS:-20}"

MAPS="${MAPS:-data/generated/grid5x5_coordbelief_v4_200.jsonl}"
MAP_CONFIG="${MAP_CONFIG:-configs/maps/grid5x5_coordbelief_v4_200.yaml}"
RUN="${RUN:-runs/qwen25_${MODEL_SIZE_LABEL}_coordbelief_v4_200}"

STAGES="${STAGES:-maps,generate,validate,targets,activations,probes,report,quality,viewer}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bf16}"
ACTIVATION_POSITIONS="${ACTIVATION_POSITIONS:-prompt_last,pre_action_token,mean_current_belief_grid}"
PROBE_GROUPS="${PROBE_GROUPS:-local,cells,planning}"
PROBE_SPLITS="${PROBE_SPLITS:-10}"
PROBE_EPOCHS="${PROBE_EPOCHS:-60}"
VIEWER_FOLDS="${VIEWER_FOLDS:-5}"

GEN_OVERWRITE="${GEN_OVERWRITE:-0}"
ACTIVATION_OVERWRITE="${ACTIVATION_OVERWRITE:-0}"
PROBE_OVERWRITE="${PROBE_OVERWRITE:-0}"

has_stage() {
  [[ ",${STAGES}," == *",$1,"* ]]
}

echo "============================================================"
echo "Coordinate-Belief v4"
echo "MODEL=$MODEL"
echo "GPUS=$GPUS"
echo "MAPS=$MAPS"
echo "RUN=$RUN"
echo "NUM_EPISODES=$NUM_EPISODES"
echo "STAGES=$STAGES"
echo "============================================================"

mkdir -p "$(dirname "$MAPS")" "$(dirname "$RUN")"

if has_stage maps; then
  if [[ -s "$MAPS" ]]; then
    echo "[maps] reuse $MAPS"
  else
    echo "[maps] generate $NUM_EPISODES maps"
    grid-world maps generate \
      --config "$MAP_CONFIG" \
      --output "$MAPS"
    grid-world maps validate --maps "$MAPS"
  fi
fi

if has_stage generate; then
  GENERATE_FLAGS=()
  if [[ "$GEN_OVERWRITE" == "1" ]]; then
    GENERATE_FLAGS+=(--overwrite)
  else
    GENERATE_FLAGS+=(--resume)
  fi

  python scripts/generate_coord_belief_v4.py \
    --maps "$MAPS" \
    --run "$RUN" \
    --model "$MODEL" \
    --backend vllm \
    --gpus "$GPUS" \
    --num-episodes "$NUM_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --dtype bfloat16 \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-tokens 320 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    "${GENERATE_FLAGS[@]}"
fi

if has_stage validate; then
  python scripts/validate_coord_belief_v4.py \
    --run "$RUN" \
    --strict-raw-format

  # Keep the repository's own validation/summarization as a second check.
  # coord-v4 feedback schema normalization (v4.1 hotfix)
  PYTHONPATH=. python scripts/normalize_coord_v4_feedback.py --run "$RUN"

  grid-world trajectories validate --run "$RUN"
  grid-world trajectories summarize --run "$RUN"
fi

if has_stage targets; then
  rm -rf "$RUN/targets"
  grid-world targets build --run "$RUN"
fi

if has_stage activations; then
  if [[ "$ACTIVATION_OVERWRITE" == "1" ]]; then
    rm -rf "$RUN/activations" "$RUN/activation_shards" "$RUN/activations_A_multi"
  fi

  if [[ -f scripts/extract_activations_model_parallel.py ]]; then
    EXTRA_FLAGS=()
    if [[ "$ACTIVATION_OVERWRITE" == "1" ]]; then
      EXTRA_FLAGS+=(--overwrite)
    fi
    python scripts/extract_activations_model_parallel.py \
      --run "$RUN" \
      --model "$MODEL" \
      --gpus "$GPUS" \
      --device-map balanced \
      --dtype "$ACTIVATION_DTYPE" \
      --layers all \
      --positions "$ACTIVATION_POSITIONS" \
      --batch-size 1 \
      "${EXTRA_FLAGS[@]}"
  else
    echo "[activations] model-parallel script not found; using the installed CLI"
    FIRST_GPU="${GPUS%%,*}"
    grid-world activations extract \
      --run "$RUN" \
      --model "$MODEL" \
      --layers all \
      --positions "$ACTIVATION_POSITIONS" \
      --device "cuda:${FIRST_GPU}" \
      --dtype auto \
      --batch-size 1
  fi
fi

if has_stage probes; then
  if [[ "$PROBE_OVERWRITE" == "1" ]]; then
    rm -rf "$RUN/probes_multigpu" "$RUN/probes"
  fi

  if [[ -f scripts/train_probes_multigpu.py ]]; then
    PROBE_FLAGS=()
    if [[ "$PROBE_OVERWRITE" == "1" ]]; then
      PROBE_FLAGS+=(--overwrite)
    fi
    python scripts/train_probes_multigpu.py \
      --run "$RUN" \
      --groups "$PROBE_GROUPS" \
      --positions "$ACTIVATION_POSITIONS" \
      --layers all \
      --gpus "$GPUS" \
      --splits "$PROBE_SPLITS" \
      --epochs "$PROBE_EPOCHS" \
      --output-subdir probes_multigpu \
      "${PROBE_FLAGS[@]}"

    # Standardize the path expected by the existing report/layer-curve tools.
    rm -rf "$RUN/probes"
    ln -s probes_multigpu "$RUN/probes"
  else
    FIRST_GPU="${GPUS%%,*}"
    grid-world probes train \
      --run "$RUN" \
      --groups "$PROBE_GROUPS" \
      --positions "$ACTIVATION_POSITIONS" \
      --layers all \
      --backend torch \
      --device "cuda:${FIRST_GPU}" \
      --splits "$PROBE_SPLITS" \
      --epochs "$PROBE_EPOCHS" \
      --min-class-count 5
  fi
fi

if has_stage report; then
  # The standard report command is kept for compatibility.
  grid-world probes report --run "$RUN" || true

  if [[ -f scripts/plot_layer_curves.py ]]; then
    python scripts/plot_layer_curves.py \
      --run "$RUN" \
      --condition-label "Qwen2.5-32B Coordinate Belief v4 (200 episodes)"
  fi
fi

if has_stage quality; then
  if [[ -f scripts/analyze_run_quality.py ]]; then
    python scripts/analyze_run_quality.py --run "$RUN"
  else
    echo "[quality] scripts/analyze_run_quality.py not found; summary.json is still available."
  fi
fi

if has_stage viewer; then
  if [[ -f scripts/generate_viewer_gallery.py ]]; then
    read -r POSITION LAYER < <(
      python - "$RUN" <<'PY'
import sys
from pathlib import Path
import pandas as pd

run = Path(sys.argv[1])
candidates = [
    run / "probes_multigpu" / "best_by_task.csv",
    run / "probes" / "best_by_task.csv",
]
path = next((p for p in candidates if p.exists()), None)
if path is None:
    print("prompt_last 0")
    raise SystemExit(0)

df = pd.read_csv(path)
preferred = df[
    (df.get("task_group", "") == "cells")
    & (df.get("position", "") == "prompt_last")
]
if preferred.empty:
    preferred = df[df.get("task_group", "") == "cells"]
if preferred.empty:
    preferred = df
score_col = "macro_f1_mean" if "macro_f1_mean" in preferred.columns else "mean_macro_f1"
row = preferred.sort_values(score_col, ascending=False).iloc[0]
print(str(row["position"]), int(row["layer"]))
PY
    )
    echo "[viewer] position=$POSITION layer=$LAYER"
    python scripts/generate_viewer_gallery.py \
      --run "$RUN" \
      --position "$POSITION" \
      --layer "$LAYER" \
      --folds "$VIEWER_FOLDS"
  else
    echo "[viewer] scripts/generate_viewer_gallery.py not found; skipping gallery."
  fi
fi

echo
echo "Coordinate-Belief v4 pipeline finished."
echo "Behavior summary:  $RUN/summary.json"
echo "Probe report:      $RUN/probes_multigpu/summary.md or $RUN/probes/summary.md"
echo "Layer curves:      $RUN/layer_curves/"
echo "Quality report:    $RUN/analysis/behavior_quality/summary.md"
echo "Viewer gallery:    $RUN/trajectory_viewer_gallery/index.html"

