#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspace/luoyuzhang/grid-world}"
cd "$ROOT"

mkdir -p scripts

cat > scripts/sample_maps_stratified.py <<'PY'
#!/usr/bin/env python3
"""
Deterministically sample N maps from a larger JSONL pool while preserving the
difficulty distribution. Within each difficulty bucket, preserve direction
distribution when a direction field is available.

This script does not modify the selected map records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DIFFICULTY_PATHS = (
    "difficulty",
    "metadata.difficulty",
    "map_metadata.difficulty",
    "generation_metadata.difficulty",
)

DIRECTION_PATHS = (
    "direction_class",
    "direction",
    "metadata.direction_class",
    "metadata.direction",
    "map_metadata.direction_class",
    "map_metadata.direction",
    "generation_metadata.direction_class",
    "generation_metadata.direction",
)


def get_path(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def first_present_path(rows: list[dict[str, Any]], candidates: Iterable[str]) -> str | None:
    for path in candidates:
        if any(get_path(row, path) is not None for row in rows):
            return path
    return None


def largest_remainder_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate exactly total samples proportionally, respecting capacity."""
    if total < 0:
        raise ValueError("total must be non-negative")
    available = sum(counts.values())
    if total > available:
        raise ValueError(f"Requested {total} rows, but only {available} are available")
    if total == 0:
        return {key: 0 for key in counts}
    if available == 0:
        raise ValueError("Cannot allocate from empty buckets")

    exact = {key: total * count / available for key, count in counts.items()}
    quotas = {key: min(counts[key], math.floor(value)) for key, value in exact.items()}
    remaining = total - sum(quotas.values())

    order = sorted(
        counts,
        key=lambda key: (
            exact[key] - math.floor(exact[key]),
            counts[key] - quotas[key],
            key,
        ),
        reverse=True,
    )

    while remaining:
        progressed = False
        for key in order:
            if quotas[key] < counts[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Unable to allocate all requested samples")
    return quotas


def stable_row_key(row: dict[str, Any]) -> str:
    for key in ("map_id", "episode_id", "id", "base_map_id"):
        value = row.get(key)
        if value is not None:
            return f"{key}:{value}"
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def sample_bucket(
    rows: list[dict[str, Any]],
    quota: int,
    rng: random.Random,
    direction_path: str | None,
) -> list[dict[str, Any]]:
    if quota > len(rows):
        raise ValueError(f"Bucket quota {quota} exceeds bucket size {len(rows)}")
    if quota == len(rows):
        chosen = list(rows)
        rng.shuffle(chosen)
        return chosen
    if not direction_path:
        return rng.sample(rows, quota)

    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[dict[str, Any]] = []
    for row in rows:
        value = get_path(row, direction_path)
        if value is None:
            missing.append(row)
        else:
            by_direction[str(value)].append(row)

    if missing:
        by_direction["__MISSING__"].extend(missing)

    counts = {key: len(value) for key, value in by_direction.items()}
    direction_quotas = largest_remainder_quotas(counts, quota)

    chosen: list[dict[str, Any]] = []
    for key in sorted(by_direction):
        group = by_direction[key]
        q = direction_quotas[key]
        chosen.extend(rng.sample(group, q) if q < len(group) else list(group))
    rng.shuffle(chosen)
    return chosen


def summarize(
    rows: list[dict[str, Any]],
    difficulty_path: str,
    direction_path: str | None,
) -> dict[str, Any]:
    difficulty = Counter(str(get_path(row, difficulty_path)) for row in rows)
    direction = (
        Counter(str(get_path(row, direction_path)) for row in rows)
        if direction_path
        else Counter()
    )
    cross = Counter()
    if direction_path:
        for row in rows:
            cross[
                (
                    str(get_path(row, difficulty_path)),
                    str(get_path(row, direction_path)),
                )
            ] += 1

    return {
        "rows": len(rows),
        "difficulty_path": difficulty_path,
        "direction_path": direction_path,
        "difficulty": dict(sorted(difficulty.items())),
        "direction": dict(sorted(direction.items())),
        "difficulty_x_direction": {
            f"{difficulty_value}|{direction_value}": count
            for (difficulty_value, direction_value), count in sorted(cross.items())
        },
        "unique_row_keys": len({stable_row_key(row) for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--difficulty-path", default="auto")
    parser.add_argument("--direction-path", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.n <= 0:
        raise SystemExit("--n must be positive")
    if not args.input.is_file():
        raise SystemExit(f"Missing input map pool: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {args.output}\n"
            "Use --overwrite to regenerate it deterministically."
        )

    rows = load_jsonl(args.input)
    if args.n > len(rows):
        raise SystemExit(f"Requested {args.n} maps from a pool of only {len(rows)}")

    difficulty_path = (
        first_present_path(rows, DIFFICULTY_PATHS)
        if args.difficulty_path == "auto"
        else args.difficulty_path
    )
    if not difficulty_path:
        raise SystemExit(
            "Could not find a difficulty field. Tried: "
            + ", ".join(DIFFICULTY_PATHS)
        )

    if args.direction_path == "none":
        direction_path = None
    elif args.direction_path == "auto":
        direction_path = first_present_path(rows, DIRECTION_PATHS)
    else:
        direction_path = args.direction_path

    missing_difficulty = [
        index for index, row in enumerate(rows) if get_path(row, difficulty_path) is None
    ]
    if missing_difficulty:
        raise SystemExit(
            f"{len(missing_difficulty)} rows lack difficulty field {difficulty_path!r}; "
            f"first indices={missing_difficulty[:10]}"
        )

    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_difficulty[str(get_path(row, difficulty_path))].append(row)

    source_counts = {key: len(value) for key, value in by_difficulty.items()}
    quotas = largest_remainder_quotas(source_counts, args.n)

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for difficulty in sorted(by_difficulty):
        selected.extend(
            sample_bucket(
                by_difficulty[difficulty],
                quotas[difficulty],
                rng,
                direction_path,
            )
        )

    if len(selected) != args.n:
        raise RuntimeError(f"Internal error: selected {len(selected)} != requested {args.n}")
    keys = [stable_row_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Selected sample contains duplicate map records")

    rng.shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_summary = summarize(rows, difficulty_path, direction_path)
    sample_summary = summarize(selected, difficulty_path, direction_path)
    summary = {
        "schema_version": "stratified-map-sample-v1",
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "requested_n": args.n,
        "difficulty_quotas": dict(sorted(quotas.items())),
        "source": source_summary,
        "sample": sample_summary,
    }
    summary_path = args.output.with_suffix(".sample_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("[stratified-sample] source=", args.input)
    print("[stratified-sample] output=", args.output)
    print("[stratified-sample] seed=", args.seed)
    print("[stratified-sample] difficulty_path=", difficulty_path)
    print("[stratified-sample] direction_path=", direction_path)
    print("[stratified-sample] source_difficulty=", source_summary["difficulty"])
    print("[stratified-sample] quotas=", dict(sorted(quotas.items())))
    print("[stratified-sample] sample_difficulty=", sample_summary["difficulty"])
    if direction_path:
        print("[stratified-sample] sample_direction=", sample_summary["direction"])
    print("[stratified-sample] summary=", summary_path)


if __name__ == "__main__":
    main()

PY

cat > scripts/patch_trajectory_viewer_single_class.py <<'PY'
#!/usr/bin/env python3
"""
Patch trajectory_probe_viewer_v3.py so single-class training folds use a
constant decoder instead of crashing LogisticRegression.

This handles indented/local imports such as:
        from sklearn.linear_model import LogisticRegression
"""
from __future__ import annotations

import datetime as dt
import py_compile
import re
import shutil
import sys
from pathlib import Path


MARKER = "SINGLE_CLASS_SAFE_LOGREG_V2"


def indent_block(text: str, indent: str) -> str:
    return "\n".join(indent + line if line else "" for line in text.splitlines())


def main() -> None:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    target = repo / "scripts" / "trajectory_probe_viewer_v3.py"
    if not target.is_file():
        raise SystemExit(f"Missing viewer source: {target}")

    source = target.read_text(encoding="utf-8")
    original = source

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(target.name + f".bak_single_class_v2_{timestamp}")
    shutil.copy2(target, backup)
    print(f"[patch] backup={backup}")

    if MARKER not in source:
        import_pattern = re.compile(
            r"^(?P<indent>[ \t]*)from sklearn\.linear_model import LogisticRegression[ \t]*$",
            re.MULTILINE,
        )
        match = import_pattern.search(source)
        if not match:
            raise SystemExit(
                "[patch] could not locate any LogisticRegression import.\n"
                "Run: grep -n \"sklearn.linear_model\" "
                "scripts/trajectory_probe_viewer_v3.py"
            )

        indent = match.group("indent")
        wrapper = f"""from sklearn.linear_model import LogisticRegression as _SklearnLogisticRegression
import numpy as _single_class_np

# {MARKER}
class LogisticRegression:
    # Drop-in local wrapper supporting a one-class viewer fold.

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._model = None
        self._constant = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        y_array = _single_class_np.asarray(y)
        classes = _single_class_np.unique(y_array)
        if classes.size == 0:
            raise ValueError("Cannot fit a decoder with zero training labels")
        self.classes_ = classes
        if classes.size == 1:
            self._constant = classes[0]
            self._model = None
            return self

        self._constant = None
        self._model = _SklearnLogisticRegression(*self._args, **self._kwargs)
        if sample_weight is None:
            self._model.fit(X, y)
        else:
            self._model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X):
        if self._constant is not None:
            return _single_class_np.full(len(X), self._constant)
        return self._model.predict(X)

    def predict_proba(self, X):
        if self._constant is not None:
            return _single_class_np.ones((len(X), 1), dtype=float)
        return self._model.predict_proba(X)

    def decision_function(self, X):
        if self._constant is not None:
            return _single_class_np.zeros(len(X), dtype=float)
        return self._model.decision_function(X)

    def score(self, X, y, sample_weight=None):
        prediction = self.predict(X)
        correct = prediction == _single_class_np.asarray(y)
        if sample_weight is None:
            return float(correct.mean())
        weights = _single_class_np.asarray(sample_weight, dtype=float)
        return float((correct * weights).sum() / weights.sum())
"""
        source = source[: match.start()] + indent_block(wrapper, indent) + source[match.end() :]
        import_line = original[:match.start()].count("\n") + 1
        print(f"[patch] wrapped LogisticRegression import at original line {import_line}")
    else:
        print("[patch] safe LogisticRegression wrapper already present")

    raise_pattern = re.compile(
        r'^(?P<indent>[ \t]*)raise RuntimeError\('
        r'f"Task \{task\} has fewer than two train classes: \{classes\}"'
        r'\)[ \t]*$',
        re.MULTILINE,
    )

    replacements = 0

    def replace_raise(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        indent = match.group("indent")
        return (
            indent
            + 'print(f"[viewer/v3] task={task} has one train class {classes}; "'
            + '"using constant decoder (structural baseline, not a learned probe).")'
        )

    source = raise_pattern.sub(replace_raise, source)

    if replacements == 0:
        if "has fewer than two train classes" in source:
            raise SystemExit(
                "[patch] found the single-class error text but could not safely "
                "replace its raise statement. Inspect with:\n"
                "sed -n '260,310p' scripts/trajectory_probe_viewer_v3.py"
            )
        print("[patch] single-class raise already absent")
    else:
        print(f"[patch] replaced_single_class_raises={replacements}")

    target.write_text(source, encoding="utf-8")
    py_compile.compile(str(target), doraise=True)

    final = target.read_text(encoding="utf-8")
    if MARKER not in final:
        raise SystemExit("[patch] verification failed: wrapper marker missing")
    if re.search(
        r'raise RuntimeError\(f"Task \{task\} has fewer than two train classes',
        final,
    ):
        raise SystemExit("[patch] verification failed: crashing raise remains")

    print(f"[patch] installed={target}")
    print("[patch] py_compile=PASS")


if __name__ == "__main__":
    main()

PY

cat > scripts/run_qwen25_32b_coord200_stratified_v5.sh <<'SH'
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

SH

chmod +x   scripts/sample_maps_stratified.py   scripts/patch_trajectory_viewer_single_class.py   scripts/run_qwen25_32b_coord200_stratified_v5.sh

python -m py_compile   scripts/sample_maps_stratified.py   scripts/patch_trajectory_viewer_single_class.py

bash -n scripts/run_qwen25_32b_coord200_stratified_v5.sh

echo "[deploy] installed:"
echo "  $ROOT/scripts/sample_maps_stratified.py"
echo "  $ROOT/scripts/patch_trajectory_viewer_single_class.py"
echo "  $ROOT/scripts/run_qwen25_32b_coord200_stratified_v5.sh"
echo "[deploy] syntax checks: PASS"
