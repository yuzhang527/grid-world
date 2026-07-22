#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
GPUS="${GPUS:-0,1,2,3}"
NUM_EPISODES="${NUM_EPISODES:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-20260722}"
SOURCE_MAPS="${SOURCE_MAPS:-data/generated/grid5x5_diverse_1000.jsonl}"
MAPS="${MAPS:-data/generated/grid5x5_diverse_stratified_${NUM_EPISODES}_seed${SAMPLE_SEED}.jsonl}"
RUN="${RUN:-runs/qwen25_32b_coordbelief_v5_stratified${NUM_EPISODES}_seed${SAMPLE_SEED}}"
STAGES="${STAGES:-sample,generate,validate,targets,activations,probes,report,quality,viewer}"
RESET_VIEWER_CACHE="${RESET_VIEWER_CACHE:-1}"

V4_PIPELINE="scripts/run_qwen25_32b_coord200_v4.sh"
SAMPLER="scripts/sample_maps_stratified.py"
VIEWER_PATCHER="scripts/patch_trajectory_viewer_single_class.py"

has_stage() {
  [[ ",${STAGES}," == *",$1,"* ]]
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

echo "============================================================"
echo "Coordinate-Belief v5: stratified sample from diverse 1000"
echo "ROOT=$ROOT"
echo "MODEL=$MODEL"
echo "GPUS=$GPUS"
echo "SOURCE_MAPS=$SOURCE_MAPS"
echo "MAPS=$MAPS"
echo "RUN=$RUN"
echo "NUM_EPISODES=$NUM_EPISODES"
echo "SAMPLE_SEED=$SAMPLE_SEED"
echo "STAGES=$STAGES"
echo "============================================================"

for required in "$SAMPLER" "$VIEWER_PATCHER" "$V4_PIPELINE"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing required file: $required" >&2
    exit 1
  fi
done

if has_stage sample; then
  if [[ ! -f "$SOURCE_MAPS" ]]; then
    cat >&2 <<EOF
ERROR: original diverse map pool is missing:
  $SOURCE_MAPS

Generate it first with:
  grid-world maps generate \
    --config configs/maps/grid5x5_diverse_1000.yaml \
    --output data/generated/grid5x5_diverse_1000.jsonl
EOF
    exit 1
  fi

  python "$SAMPLER" \
    --input "$SOURCE_MAPS" \
    --output "$MAPS" \
    --n "$NUM_EPISODES" \
    --seed "$SAMPLE_SEED" \
    --overwrite

  grid-world maps validate --maps "$MAPS"
  grid-world maps summarize --maps "$MAPS"
fi

if [[ ! -f "$MAPS" ]]; then
  echo "ERROR: sampled map file does not exist: $MAPS" >&2
  echo "Include 'sample' in STAGES or point MAPS to an existing sample." >&2
  exit 1
fi

core_stages=()
for stage in generate validate targets activations probes report quality; do
  if has_stage "$stage"; then
    core_stages+=("$stage")
  fi
done

if ((${#core_stages[@]})); then
  core_csv="$(join_by_comma "${core_stages[@]}")"
  echo "[v5] delegating core stages to coordinate-belief v4: $core_csv"

  MODEL="$MODEL" \
  GPUS="$GPUS" \
  NUM_EPISODES="$NUM_EPISODES" \
  MAPS="$MAPS" \
  RUN="$RUN" \
  STAGES="$core_csv" \
  bash "$V4_PIPELINE"
fi

if has_stage viewer; then
  echo "[v5] installing robust single-class viewer patch"
  python "$VIEWER_PATCHER" "$ROOT"

  if [[ "$RESET_VIEWER_CACHE" == "1" ]]; then
    echo "[v5] clearing v5 viewer caches"
    rm -rf \
      "$RUN/trajectory_viewer_cache_v3" \
      "$RUN/trajectory_viewer_gallery"
  fi

  MODEL="$MODEL" \
  GPUS="$GPUS" \
  NUM_EPISODES="$NUM_EPISODES" \
  MAPS="$MAPS" \
  RUN="$RUN" \
  STAGES="viewer" \
  bash "$V4_PIPELINE"
fi

echo
echo "Coordinate-Belief v5 stratified pipeline finished."
echo "Sample summary:    ${MAPS%.jsonl}.sample_summary.json"
echo "Behavior summary:  $RUN/summary.json"
echo "Probe report:      $RUN/probes_multigpu/summary.md or $RUN/probes/summary.md"
echo "Layer curves:      $RUN/layer_curves/"
echo "Quality report:    $RUN/analysis/behavior_quality/summary.md"
echo "Viewer gallery:    $RUN/trajectory_viewer_gallery/index.html"

