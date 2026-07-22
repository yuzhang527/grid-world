#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
RUN_TO_REPAIR="${2:-${RUN:-}}"

cd "$ROOT"

if [[ ! -f pyproject.toml ]] || [[ ! -d src/grid_world ]]; then
  echo "[coord-v4-feedback-patch] ERROR: not a grid-world repository root: $ROOT" >&2
  exit 2
fi

mkdir -p scripts

cat > scripts/normalize_coord_v4_feedback.py <<'PY'
#!/usr/bin/env python3
"""Restore the canonical top-level ``feedback`` field in Coordinate-Belief v4 steps.

The repository-wide trajectory schema requires every step to contain ``feedback``.
Early Coordinate-Belief v4 runs kept the same information inside prompt_text (or an
alias field) but did not materialize the canonical field. This utility repairs the
merged run atomically without rerunning model inference.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

ALIASES = (
    "last_feedback",
    "environment_feedback",
    "env_feedback",
    "adjacent_feedback",
    "observation",
)
PROMPT_KEYS = ("prompt_text", "prompt", "rendered_prompt")
REQUIRED_STEP_KEYS = (
    "episode_id",
    "step_id",
    "current_pos",
    "next_pos",
    "feedback",
    "action",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} line {line_no}")
            rows.append(row)
    return rows


def coerce_feedback(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def decode_json_object_after(text: str, marker_end: int) -> dict[str, Any] | None:
    start = text.find("{", marker_end)
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def feedback_from_prompt(text: str) -> dict[str, Any] | None:
    # Preferred structured prompt segment used by the existing pipeline.
    matches = list(
        re.finditer(
            r"<last_feedback\b[^>]*>(.*?)</last_feedback>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    for match in reversed(matches):
        body = match.group(1).strip()
        parsed = coerce_feedback(body)
        if parsed is not None:
            return parsed
        parsed = decode_json_object_after(body, 0)
        if parsed is not None:
            return parsed

    # Backward-compatible textual headings.
    heading_patterns = (
        r"\blast_feedback\b\s*[:=]",
        r"\blast\s+feedback\b\s*[:=]",
        r"\benvironment_feedback\b\s*[:=]",
        r"\benvironment\s+feedback\b\s*[:=]",
    )
    for pattern in heading_patterns:
        headings = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        for heading in reversed(headings):
            parsed = decode_json_object_after(text, heading.end())
            if parsed is not None:
                return parsed
    return None


def recover_feedback(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    existing = coerce_feedback(row.get("feedback"))
    if existing is not None:
        return existing, "feedback"

    for key in ALIASES:
        recovered = coerce_feedback(row.get(key))
        if recovered is not None:
            return recovered, key

    for key in PROMPT_KEYS:
        prompt = row.get(key)
        if isinstance(prompt, str) and prompt:
            recovered = feedback_from_prompt(prompt)
            if recovered is not None:
                return recovered, f"{key}:last_feedback"

    return None, None


def validate_rows(rows: Iterable[dict[str, Any]]) -> None:
    seen: set[tuple[str, int]] = set()
    for row_index, row in enumerate(rows):
        missing = [key for key in REQUIRED_STEP_KEYS if key not in row]
        if missing:
            episode_id = row.get("episode_id", "<missing>")
            step_id = row.get("step_id", "<missing>")
            raise ValueError(
                f"Step row {row_index} ({episode_id}, {step_id}) missing {missing}"
            )
        if not isinstance(row["feedback"], dict):
            raise ValueError(
                f"Step ({row['episode_id']}, {row['step_id']}) has non-object feedback"
            )
        key = (str(row["episode_id"]), int(row["step_id"]))
        if key in seen:
            raise ValueError(f"Duplicate step key after repair: {key}")
        seen.add(key)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create steps.jsonl.bak_coord_v4_feedback before the first rewrite.",
    )
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    steps_path = run / "steps.jsonl"
    if not steps_path.is_file():
        raise FileNotFoundError(f"Missing {steps_path}")

    rows = read_jsonl(steps_path)
    source_counts: dict[str, int] = {}
    repaired = 0
    unresolved: list[str] = []

    for row in rows:
        feedback, source = recover_feedback(row)
        if feedback is None:
            unresolved.append(f"{row.get('episode_id')}:{row.get('step_id')}")
            continue
        if "feedback" not in row or row.get("feedback") != feedback:
            row["feedback"] = feedback
            repaired += 1
        source_counts[source or "unknown"] = source_counts.get(source or "unknown", 0) + 1

    if unresolved:
        preview = ", ".join(unresolved[:10])
        suffix = " ..." if len(unresolved) > 10 else ""
        raise ValueError(
            f"Could not recover feedback for {len(unresolved)} step(s): {preview}{suffix}. "
            "No file was modified."
        )

    validate_rows(rows)

    if repaired:
        backup_path = run / "steps.jsonl.bak_coord_v4_feedback"
        if not args.no_backup and not backup_path.exists():
            shutil.copy2(steps_path, backup_path)
            print(f"[coord-v4-feedback] backup={backup_path}")
        atomic_write_jsonl(steps_path, rows)

    print(f"[coord-v4-feedback] run={run}")
    print(f"[coord-v4-feedback] rows={len(rows)} repaired={repaired}")
    print(f"[coord-v4-feedback] sources={json.dumps(source_counts, sort_keys=True)}")
    print("[coord-v4-feedback] PASS")


if __name__ == "__main__":
    main()
PY
chmod +x scripts/normalize_coord_v4_feedback.py

PIPELINE="scripts/run_qwen25_32b_coord200_v4.sh"
if [[ ! -f "$PIPELINE" ]]; then
  echo "[coord-v4-feedback-patch] ERROR: missing $ROOT/$PIPELINE" >&2
  exit 3
fi

python - "$PIPELINE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "# coord-v4 feedback schema normalization (v4.1 hotfix)"
command = 'PYTHONPATH=. python scripts/normalize_coord_v4_feedback.py --run "$RUN"'

if marker in text:
    print(f"[coord-v4-feedback-patch] pipeline already patched: {path}")
    raise SystemExit(0)

lines = text.splitlines()
insert_at = None
for index, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("#"):
        continue
    if "trajectories validate" in stripped:
        insert_at = index
        break

if insert_at is None:
    raise RuntimeError(
        f"Could not find the generic 'trajectories validate' command in {path}. "
        "The normalizer script was installed, but the pipeline was not edited."
    )

indent = lines[insert_at][: len(lines[insert_at]) - len(lines[insert_at].lstrip())]
block = [
    f"{indent}{marker}",
    f"{indent}{command}",
    "",
]
lines[insert_at:insert_at] = block

backup = path.with_suffix(path.suffix + ".bak_coord_v4_feedback")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[coord-v4-feedback-patch] patched pipeline: {path}")
print(f"[coord-v4-feedback-patch] backup: {backup}")
PY

python -m py_compile scripts/normalize_coord_v4_feedback.py
bash -n "$PIPELINE"

echo "[coord-v4-feedback-patch] installed normalizer and patched pipeline"

if [[ -n "$RUN_TO_REPAIR" ]]; then
  if [[ -f "$RUN_TO_REPAIR/steps.jsonl" ]]; then
    PYTHONPATH=. python scripts/normalize_coord_v4_feedback.py --run "$RUN_TO_REPAIR"
  else
    echo "[coord-v4-feedback-patch] WARNING: RUN has no steps.jsonl: $RUN_TO_REPAIR" >&2
  fi
fi

echo "[coord-v4-feedback-patch] DONE"
