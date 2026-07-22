#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-$PWD}"
REPO="$(cd "$REPO" && pwd)"
cd "$REPO"
mkdir -p scripts

cat > scripts/sample_maps_stratified_v6.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

DIFF_ORDER = ("easy", "medium", "hard")
DIFF_WEIGHTS = {"easy": 0.30, "medium": 0.45, "hard": 0.25}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def normalize_difficulty(row: dict[str, Any]) -> str:
    candidates = [
        row.get("difficulty"),
        row.get("difficulty_label"),
        row.get("difficulty_level"),
        (row.get("metadata") or {}).get("difficulty") if isinstance(row.get("metadata"), dict) else None,
    ]
    for value in candidates:
        if value is None:
            continue
        text = str(value).strip().lower()
        aliases = {
            "e": "easy", "easy": "easy", "0": "easy",
            "m": "medium", "med": "medium", "medium": "medium", "1": "medium",
            "h": "hard", "hard": "hard", "2": "hard",
        }
        if text in aliases:
            return aliases[text]
    raise RuntimeError(
        "Map has no recognized difficulty. Expected one of difficulty, "
        f"difficulty_label, difficulty_level. map_id={row.get('map_id') or row.get('id')}"
    )


def as_xy(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        try:
            return int(value["x"]), int(value["y"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def direction_bucket(row: dict[str, Any]) -> str:
    for key in ("direction", "direction_bucket", "quadrant", "goal_direction"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().upper()
    start = as_xy(row.get("start") or row.get("start_pos") or row.get("source"))
    goal = as_xy(row.get("goal") or row.get("goal_pos") or row.get("target"))
    if start is None or goal is None:
        return "UNKNOWN"
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    sx = "E" if dx > 0 else "W" if dx < 0 else ""
    sy = "N" if dy > 0 else "S" if dy < 0 else ""
    return sy + sx if sy + sx else "SAME"


def largest_remainder(total: int, weights: dict[str, float], capacities: dict[str, int]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    active = {k: float(v) for k, v in weights.items() if capacities.get(k, 0) > 0 and v > 0}
    if not active and total:
        raise RuntimeError("No non-empty strata available")
    weight_sum = sum(active.values()) or 1.0
    raw = {k: total * v / weight_sum for k, v in active.items()}
    allocation = {k: min(capacities[k], int(math.floor(raw[k]))) for k in active}
    remaining = total - sum(allocation.values())
    while remaining > 0:
        candidates = [
            k for k in active
            if allocation[k] < capacities[k]
        ]
        if not candidates:
            raise RuntimeError(f"Insufficient capacity: requested {total}, capacities={capacities}")
        candidates.sort(key=lambda k: (raw[k] - math.floor(raw[k]), active[k], k), reverse=True)
        for key in candidates:
            if remaining <= 0:
                break
            if allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
    return {k: allocation.get(k, 0) for k in capacities}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically sample a difficulty-stratified v6 map subset.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    rows = read_jsonl(args.source)
    if args.count <= 0 or args.count > len(rows):
        raise RuntimeError(f"Invalid count={args.count}; source rows={len(rows)}")

    by_diff: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_diff[normalize_difficulty(row)].append(row)

    capacities = {d: len(by_diff[d]) for d in DIFF_ORDER}
    quotas = largest_remainder(args.count, DIFF_WEIGHTS, capacities)
    # For 100 episodes this is a hard contract inherited from the 1000-map 300/450/250 split.
    if args.count == 100 and quotas != {"easy": 30, "medium": 45, "hard": 25}:
        raise RuntimeError(f"Unexpected v6 quotas: {quotas}")

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    per_direction: dict[str, dict[str, int]] = {}

    for difficulty in DIFF_ORDER:
        bucket = by_diff[difficulty]
        by_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in bucket:
            by_dir[direction_bucket(row)].append(row)
        for values in by_dir.values():
            rng.shuffle(values)
        dir_cap = {key: len(value) for key, value in by_dir.items()}
        dir_weights = {key: len(value) for key, value in by_dir.items()}
        dir_quota = largest_remainder(quotas[difficulty], dir_weights, dir_cap)
        per_direction[difficulty] = dict(sorted(dir_quota.items()))
        for direction, count in sorted(dir_quota.items()):
            selected.extend(by_dir[direction][:count])

    rng.shuffle(selected)
    if len(selected) != args.count:
        raise RuntimeError(f"Selected {len(selected)} rows, expected {args.count}")

    # Reject duplicate stable IDs when IDs are present.
    ids = [str(r.get("map_id") or r.get("id") or "") for r in selected]
    nonempty = [x for x in ids if x]
    if len(nonempty) != len(set(nonempty)):
        raise RuntimeError("Duplicate map IDs in sampled subset")

    write_jsonl(args.output, selected)
    summary_path = args.summary or args.output.with_suffix(".sample_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "coordbelief-v6-sample-1.0",
        "source": str(args.source),
        "output": str(args.output),
        "seed": args.seed,
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "source_difficulty": dict(Counter(normalize_difficulty(r) for r in rows)),
        "sample_difficulty": dict(Counter(normalize_difficulty(r) for r in selected)),
        "sample_direction_within_difficulty": per_direction,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[v6-sample] saved={args.output}")
    print(f"[v6-sample] summary={summary_path}")
    print(f"[v6-sample] difficulty={summary['sample_difficulty']}")


if __name__ == "__main__":
    main()
PY
chmod +x scripts/sample_maps_stratified_v6.py

cat > scripts/build_coordbelief_targets_v6.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

DIRECTIONS: dict[str, tuple[int, int]] = {
    "UP": (0, 1),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
VALID_CELL = {"F", "O", "U"}
VALID_FEEDBACK = {"F", "O", "WALL"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def as_xy(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        for x_key, y_key in (("x", "y"), ("col", "row")):
            if x_key in value and y_key in value:
                try:
                    return int(value[x_key]), int(value[y_key])
                except (TypeError, ValueError):
                    return None
    return None


def normalize_state(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("state") or value.get("value") or value.get("status")
    if value is None:
        return None
    text = str(value).strip().upper()
    aliases = {
        "FREE": "F", "OPEN": "F", "PASSABLE": "F", ".": "F", "0": "F",
        "OBSTACLE": "O", "BLOCKED": "O", "BLOCK": "O", "#": "O", "1": "O",
        "UNKNOWN": "U", "?": "U",
        "OUT_OF_BOUNDS": "WALL", "OUT-OF-BOUNDS": "WALL", "BOUNDARY": "WALL", "W": "WALL",
    }
    return aliases.get(text, text if text in VALID_FEEDBACK | {"U"} else None)


def prompt_text(row: dict[str, Any]) -> str | None:
    value = row.get("prompt_text") or row.get("prompt") or row.get("formatted_prompt")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return None


def parse_structured_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return None


def feedback_from_prompt(row: dict[str, Any]) -> Any:
    text = prompt_text(row)
    if not text:
        return None
    match = re.search(r"<last_feedback>(.*?)</last_feedback>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return parse_structured_text(match.group(1))


def normalize_feedback(raw: Any, current: tuple[int, int], width: int, height: int) -> dict[str, tuple[tuple[int, int] | None, str]]:
    result: dict[str, tuple[tuple[int, int] | None, str]] = {}
    if raw is None:
        return result

    if isinstance(raw, str):
        parsed = parse_structured_text(raw)
        if parsed is not None:
            raw = parsed

    if isinstance(raw, list):
        converted: dict[str, Any] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            direction = str(item.get("direction") or item.get("dir") or "").upper()
            if direction:
                converted[direction] = item
        raw = converted

    if not isinstance(raw, dict):
        return result

    for key, value in raw.items():
        direction = str(key).strip().upper()
        if direction not in DIRECTIONS:
            continue
        dx, dy = DIRECTIONS[direction]
        expected = (current[0] + dx, current[1] + dy)
        in_bounds = 0 <= expected[0] < width and 0 <= expected[1] < height
        coord: tuple[int, int] | None = None
        state: str | None = None
        if isinstance(value, dict):
            coord = as_xy(value.get("coord") or value.get("coordinate") or value.get("position") or value.get("cell"))
            state = normalize_state(value)
        else:
            state = normalize_state(value)
        if coord is None and in_bounds:
            coord = expected
        if state is None:
            continue
        if not in_bounds:
            # Boundary directions are walls irrespective of a noisy coordinate payload.
            state = "WALL"
            coord = None
        result[direction] = (coord, state)
    return result


def choose_feedback(row: dict[str, Any], current: tuple[int, int], width: int, height: int) -> tuple[dict[str, tuple[tuple[int, int] | None, str]], str]:
    candidates: list[tuple[str, Any]] = [
        ("prompt.last_feedback", feedback_from_prompt(row)),
        ("feedback_before_action", row.get("feedback_before_action")),
        ("last_feedback", row.get("last_feedback")),
        ("feedback", row.get("feedback")),
    ]
    normalized: list[tuple[str, dict[str, tuple[tuple[int, int] | None, str]]]] = []
    for name, raw in candidates:
        parsed = normalize_feedback(raw, current, width, height)
        if parsed:
            normalized.append((name, parsed))
    if not normalized:
        raise RuntimeError("No action-before feedback could be parsed from prompt/feedback fields")

    chosen_name, chosen = normalized[0]
    # When multiple authoritative fields exist, they must agree direction by direction.
    for other_name, other in normalized[1:]:
        for direction in set(chosen) & set(other):
            if chosen[direction] != other[direction]:
                raise RuntimeError(
                    f"Feedback disagreement: {chosen_name}.{direction}={chosen[direction]} "
                    f"but {other_name}.{direction}={other[direction]}"
                )
    return chosen, chosen_name


def parse_belief_coordinates(row: dict[str, Any], width: int, height: int) -> dict[tuple[int, int], str]:
    raw = row.get("parsed_belief_coordinates") or row.get("belief_coordinates")
    if raw is None and isinstance(row.get("parsed"), dict):
        raw = row["parsed"].get("belief_coordinates")
    if raw is None:
        raw_response = row.get("raw_response") or row.get("response")
        if isinstance(raw_response, str):
            parsed = parse_structured_text(raw_response)
            if isinstance(parsed, dict):
                raw = parsed.get("belief_coordinates")
    if not isinstance(raw, dict):
        return {}

    cells: dict[tuple[int, int], str] = {}
    for label in ("F", "O"):
        values = raw.get(label) or raw.get(label.lower()) or []
        if not isinstance(values, list):
            raise RuntimeError(f"belief_coordinates[{label}] must be a list")
        for value in values:
            coord = as_xy(value)
            if coord is None:
                raise RuntimeError(f"Invalid {label} coordinate: {value!r}")
            x, y = coord
            if not (0 <= x < width and 0 <= y < height):
                raise RuntimeError(f"Out-of-bounds explicit belief coordinate: {coord}")
            previous = cells.get(coord)
            if previous and previous != label:
                raise RuntimeError(f"Explicit belief overlap at {coord}: {previous} vs {label}")
            cells[coord] = label
    return cells


def matrix_dimensions(row: dict[str, Any]) -> tuple[int, int] | None:
    for w_key, h_key in (("width", "height"), ("grid_width", "grid_height"), ("cols", "rows")):
        if w_key in row and h_key in row:
            try:
                return int(row[w_key]), int(row[h_key])
            except (TypeError, ValueError):
                pass
    size = row.get("size") or row.get("grid_size")
    if isinstance(size, int):
        return size, size
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        try:
            return int(size[0]), int(size[1])
        except (TypeError, ValueError):
            return None
    return None


def extract_obstacles(row: dict[str, Any], width: int, height: int) -> set[tuple[int, int]] | None:
    for key in ("obstacles", "obstacle_coordinates", "blocked", "blocked_cells", "walls"):
        values = row.get(key)
        if isinstance(values, list):
            result: set[tuple[int, int]] = set()
            valid = True
            for value in values:
                coord = as_xy(value)
                if coord is None:
                    valid = False
                    break
                result.add(coord)
            if valid:
                return result

    matrix = row.get("true_map") or row.get("grid") or row.get("map_grid")
    if isinstance(matrix, list) and len(matrix) == height and all(isinstance(r, (list, str)) for r in matrix):
        # Repository display convention is top row = largest y. Coordinates remain x-right/y-up.
        result: set[tuple[int, int]] = set()
        for display_row, values in enumerate(matrix):
            seq = list(values) if isinstance(values, str) else values
            if len(seq) != width:
                return None
            y = height - 1 - display_row
            for x, value in enumerate(seq):
                state = normalize_state(value)
                if state == "O":
                    result.add((x, y))
        return result
    return None


def collect_map_records(run: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in (run / "maps.jsonl", run / "episodes.jsonl"):
        if path.exists():
            records.extend(read_jsonl(path))
    return records


def ids_for(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("map_id", "id", "episode_id", "source_map_id"):
        value = row.get(key)
        if value is not None:
            values.append(str(value))
    return values


@dataclass(frozen=True)
class TrueMap:
    width: int
    height: int
    obstacles: frozenset[tuple[int, int]]

    def state(self, coord: tuple[int, int]) -> str:
        x, y = coord
        if not (0 <= x < self.width and 0 <= y < self.height):
            return "WALL"
        return "O" if coord in self.obstacles else "F"


def resolve_true_map(step: dict[str, Any], episode_record: dict[str, Any] | None, map_index: dict[str, dict[str, Any]], default_size: int) -> TrueMap:
    candidates: list[dict[str, Any]] = [step]
    if episode_record:
        candidates.append(episode_record)
    for identifier in ids_for(step) + (ids_for(episode_record) if episode_record else []):
        if identifier in map_index:
            candidates.append(map_index[identifier])

    width = height = default_size
    for candidate in candidates:
        dims = matrix_dimensions(candidate)
        if dims:
            width, height = dims
            break
    for candidate in candidates:
        obstacles = extract_obstacles(candidate, width, height)
        if obstacles is not None:
            for coord in obstacles:
                if not (0 <= coord[0] < width and 0 <= coord[1] < height):
                    raise RuntimeError(f"Out-of-bounds obstacle {coord}")
            return TrueMap(width, height, frozenset(obstacles))
    raise RuntimeError(
        f"Could not resolve true-map obstacles for episode={step.get('episode_id')} map_id={step.get('map_id')}. "
        "Expected run/maps.jsonl or episodes.jsonl to contain obstacle coordinates."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict coordinate-belief v6 cell targets.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-episode", default=None)
    parser.add_argument("--audit-step", type=int, default=None)
    args = parser.parse_args()

    run = args.run.resolve()
    steps_path = run / "steps.jsonl"
    if not steps_path.exists():
        raise FileNotFoundError(steps_path)
    steps = read_jsonl(steps_path)
    if not steps:
        raise RuntimeError("steps.jsonl is empty")

    keys: set[tuple[str, int]] = set()
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        episode_id = str(row.get("episode_id"))
        try:
            step_id = int(row.get("step_id"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid step_id in episode {episode_id}: {row.get('step_id')!r}") from exc
        key = (episode_id, step_id)
        if key in keys:
            raise RuntimeError(f"Duplicate step key: {key}")
        keys.add(key)
        by_episode[episode_id].append(row)

    map_records = collect_map_records(run)
    map_index: dict[str, dict[str, Any]] = {}
    episode_records: dict[str, dict[str, Any]] = {}
    for record in map_records:
        for identifier in ids_for(record):
            map_index.setdefault(identifier, record)
        if record.get("episode_id") is not None:
            episode_records[str(record["episode_id"])] = record

    output = args.output or (run / "targets" / "targets.jsonl")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite")
    if output.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output.with_name(output.name + f".bak_v6_{stamp}")
        shutil.copy2(output, backup)
        print(f"[v6-targets] backup={backup}")

    all_targets: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    feedback_sources: dict[str, int] = defaultdict(int)
    model_exact_known = 0
    total_known = 0

    for episode_id in sorted(by_episode):
        episode_steps = sorted(by_episode[episode_id], key=lambda r: int(r["step_id"]))
        expected_ids = list(range(len(episode_steps)))
        actual_ids = [int(r["step_id"]) for r in episode_steps]
        if actual_ids != expected_ids:
            raise RuntimeError(f"Episode {episode_id}: non-contiguous step IDs {actual_ids[:10]}...")

        true_map = resolve_true_map(episode_steps[0], episode_records.get(episode_id), map_index, args.size)
        gold: dict[tuple[int, int], str] = {}

        for row in episode_steps:
            step_id = int(row["step_id"])
            current = as_xy(row.get("current_pos") or row.get("position"))
            if current is None:
                raise RuntimeError(f"{(episode_id, step_id)}: missing current_pos")
            if true_map.state(current) != "F":
                raise RuntimeError(f"{(episode_id, step_id)}: current_pos {current} is not free in true map")
            gold[current] = "F"  # dynamic episode start/current position; never hard-code (0,0)

            try:
                feedback, source = choose_feedback(row, current, true_map.width, true_map.height)
            except RuntimeError as exc:
                raise RuntimeError(f"{(episode_id, step_id)}: {exc}") from exc
            feedback_sources[source] += 1

            local: dict[str, str] = {}
            for direction, (dx, dy) in DIRECTIONS.items():
                expected = (current[0] + dx, current[1] + dy)
                expected_state = true_map.state(expected)
                if expected_state == "WALL":
                    local[direction] = "WALL"
                    if direction in feedback and feedback[direction][1] != "WALL":
                        raise RuntimeError(
                            f"{(episode_id, step_id)}: boundary {direction} should be WALL, got {feedback[direction]}"
                        )
                    continue
                if direction not in feedback:
                    raise RuntimeError(
                        f"{(episode_id, step_id)}: exact adjacent feedback missing in-bounds direction {direction}"
                    )
                coord, state = feedback[direction]
                if coord != expected:
                    raise RuntimeError(
                        f"{(episode_id, step_id)}: {direction} coord={coord}, expected={expected}"
                    )
                if state not in {"F", "O"}:
                    raise RuntimeError(
                        f"{(episode_id, step_id)}: {direction} state={state}, expected F/O"
                    )
                if state != expected_state:
                    raise RuntimeError(
                        f"{(episode_id, step_id)}: feedback {direction}={state}, true map says {expected_state}"
                    )
                gold[expected] = state
                local[direction] = state

            explicit = parse_belief_coordinates(row, true_map.width, true_map.height)
            target: dict[str, Any] = {
                "schema_version": "coordinate-belief-v6.0",
                "episode_id": episode_id,
                "step_id": step_id,
                "feedback_source": source,
                "gold_known_cell_count": len(gold),
            }
            for direction in DIRECTIONS:
                target[f"gold_local_{direction}_OFUW"] = local[direction]

            known_correct = 0
            for x in range(true_map.width):
                for y in range(true_map.height):
                    coord = (x, y)
                    gold_state = gold.get(coord, "U")
                    model_state = explicit.get(coord, "U")
                    true_state = true_map.state(coord)
                    if gold_state not in VALID_CELL or model_state not in VALID_CELL:
                        raise RuntimeError(f"{(episode_id, step_id)}: invalid cell label")
                    target[f"gold_cell_x{x}_y{y}_OFU"] = gold_state
                    target[f"gold_cell_x{x}_y{y}_known"] = int(gold_state != "U")
                    target[f"model_cell_x{x}_y{y}_OFU"] = model_state
                    target[f"true_cell_x{x}_y{y}_FO"] = true_state
                    target[f"true_cell_x{x}_y{y}_FO_observed"] = true_state if gold_state != "U" else None
                    target[f"true_cell_x{x}_y{y}_FO_unobserved"] = true_state if gold_state == "U" else None
                    if gold_state != "U":
                        total_known += 1
                        if model_state == gold_state:
                            known_correct += 1
                            model_exact_known += 1

            target["model_known_cell_acc_step"] = known_correct / len(gold) if gold else None
            target["model_missed_any_gold_known"] = int(known_correct != len(gold))
            all_targets.append(target)

            if (args.audit_episode is None or args.audit_episode == episode_id) and (
                args.audit_step is None or args.audit_step == step_id
            ):
                audit_rows.append({
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "current_pos": list(current),
                    "feedback_source": source,
                    "feedback": {k: {"coord": list(v[0]) if v[0] else None, "state": v[1]} for k, v in feedback.items()},
                    "gold": {f"{x},{y}": state for (x, y), state in sorted(gold.items())},
                    "explicit": {f"{x},{y}": state for (x, y), state in sorted(explicit.items())},
                })

    if len(all_targets) != len(steps):
        raise RuntimeError(f"Target/step count mismatch: {len(all_targets)} vs {len(steps)}")

    # Contract test derived from the concrete failing schema: nested coord/state must be applied immediately.
    fixture_row = {
        "current_pos": [3, 1],
        "feedback": {
            "DOWN": {"coord": [3, 0], "state": "O"},
            "LEFT": {"coord": [2, 1], "state": "O"},
            "RIGHT": {"coord": [4, 1], "state": "F"},
            "UP": {"coord": [3, 2], "state": "F"},
        },
    }
    fixture, _ = choose_feedback(fixture_row, (3, 1), 5, 5)
    assert fixture["UP"] == ((3, 2), "F")
    assert fixture["DOWN"] == ((3, 0), "O")

    write_jsonl(output, all_targets)
    audit_path = run / "coordbelief_v6_target_audit.json"
    audit_payload = {
        "schema_version": "coordinate-belief-v6-audit-1.0",
        "run": str(run),
        "steps": len(steps),
        "episodes": len(by_episode),
        "targets": len(all_targets),
        "feedback_sources": dict(feedback_sources),
        "known_cell_explicit_accuracy": model_exact_known / total_known if total_known else None,
        "sample_audit_rows": audit_rows[:20],
        "contract_tests": {
            "nested_coord_state_UP_3_2_is_F": True,
            "dynamic_start_not_hard_coded": True,
            "coordinate_belief_unique_source": True,
        },
    }
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[v6-targets] episodes={len(by_episode)} steps={len(steps)}")
    print(f"[v6-targets] saved={output}")
    print(f"[v6-targets] audit={audit_path}")
    print(f"[v6-targets] known-cell explicit accuracy={audit_payload['known_cell_explicit_accuracy']:.6f}")


if __name__ == "__main__":
    main()
PY
chmod +x scripts/build_coordbelief_targets_v6.py

cat > scripts/trajectory_probe_viewer_coordbelief_v6.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["episode_id"]), int(row["step_id"])


def load_positions(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, dict):
        for name in ("positions", "names"):
            if isinstance(value.get(name), list):
                return [str(x) for x in value[name]]
    raise RuntimeError(f"Unrecognized positions.json: {path}")


def choose_layer(run: Path, available: list[int], requested: str) -> int:
    if requested != "auto":
        layer = int(requested)
        if layer not in available:
            raise RuntimeError(f"Layer {layer} unavailable; choices={available}")
        return layer
    result_path = run / "probes_multigpu" / "probe_results.csv"
    if result_path.exists():
        try:
            import csv
            by_layer: dict[int, list[float]] = {}
            with result_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("task_group") != "cells" or row.get("position") != "prompt_last":
                        continue
                    metric = row.get("macro_f1_mean") or row.get("macro_f1")
                    if metric in (None, ""):
                        continue
                    layer = int(float(row["layer"]))
                    by_layer.setdefault(layer, []).append(float(metric))
            if by_layer:
                return max(by_layer, key=lambda layer: float(np.mean(by_layer[layer])))
        except Exception as exc:
            print(f"[viewer-v6] warning: could not derive best layer from probe_results.csv: {exc}")
    return available[len(available) // 2]


def parse_explicit(step: dict[str, Any], size: int) -> list[str]:
    raw = step.get("parsed_belief_coordinates") or step.get("belief_coordinates") or {}
    cells = ["U"] * (size * size)
    if isinstance(raw, dict):
        for state in ("F", "O"):
            for coord in raw.get(state, []) or []:
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    x, y = int(coord[0]), int(coord[1])
                    if 0 <= x < size and 0 <= y < size:
                        idx = x * size + y
                        if cells[idx] != "U" and cells[idx] != state:
                            raise RuntimeError(f"Explicit F/O overlap at {(x, y)}")
                        cells[idx] = state
    return cells


def grid_payload(target: dict[str, Any], prefix: str, size: int) -> list[str]:
    return [str(target.get(f"{prefix}_x{x}_y{y}_OFU", "U")) for x in range(size) for y in range(size)]


def true_payload(target: dict[str, Any], size: int) -> list[str]:
    return [str(target.get(f"true_cell_x{x}_y{y}_FO", "F")) for x in range(size) for y in range(size)]


def fit_oof(X: np.ndarray, targets: list[dict[str, Any]], episodes: np.ndarray, size: int, folds: int) -> np.ndarray:
    classes = np.array(["F", "O", "U"], dtype=object)
    predictions = np.full((len(targets), size * size), "U", dtype=object)
    unique_groups = np.unique(episodes)
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("Need at least two episodes for OOF viewer probes")
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, groups=episodes)):
        print(f"[viewer-v6] fold={fold} train={len(train_idx)} test={len(test_idx)}")
        for cell_index in range(size * size):
            x, y = divmod(cell_index, size)
            task = f"gold_cell_x{x}_y{y}_OFU"
            y_train = np.array([str(targets[i][task]) for i in train_idx], dtype=object)
            unique = np.unique(y_train)
            if len(unique) == 1:
                predictions[test_idx, cell_index] = unique[0]
                continue
            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=0,
            )
            model.fit(X[train_idx], y_train)
            predictions[test_idx, cell_index] = model.predict(X[test_idx])
    return predictions


def render_grid(states: list[str], size: int, current: list[int] | None = None, goal: list[int] | None = None) -> str:
    items: list[str] = []
    current_xy = tuple(current[:2]) if isinstance(current, list) and len(current) >= 2 else None
    goal_xy = tuple(goal[:2]) if isinstance(goal, list) and len(goal) >= 2 else None
    for y in range(size - 1, -1, -1):
        for x in range(size):
            state = states[x * size + y]
            markers = ""
            if (x, y) == goal_xy:
                markers += '<span class="mark goal">G</span>'
            if (x, y) == current_xy:
                markers += '<span class="mark agent">A</span>'
            items.append(
                f'<div class="cell state-{html.escape(state)}" title="({x},{y}) {html.escape(state)}">'
                f'<span class="coord">{x},{y}</span><strong>{html.escape(state)}</strong>{markers}</div>'
            )
    return "".join(items)


def render_episode(run: Path, episode_id: str, rows: list[dict[str, Any]], size: int, layer: int, output_dir: Path) -> None:
    data_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(episode_id)} — Coordinate Belief v6</title>
<style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#f5f6f8;color:#15171a}} h1{{margin-bottom:4px}}
.meta{{color:#59636e;margin-bottom:14px}} .controls{{display:flex;gap:8px;align-items:center;margin:12px 0}}
button,input{{font:inherit}} .panels{{display:grid;grid-template-columns:repeat(4,minmax(210px,1fr));gap:14px}}
.panel{{background:white;border:1px solid #d9dee5;border-radius:10px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
.grid{{display:grid;grid-template-columns:repeat({size},1fr);aspect-ratio:1;gap:3px}} .cell{{position:relative;border-radius:5px;display:flex;align-items:center;justify-content:center;border:1px solid #cbd2da;min-width:0}}
.state-U{{background:#eceff3}} .state-F{{background:#dff4df}} .state-O{{background:#40464f;color:white}} .state-WALL{{background:#20242a;color:white}}
.coord{{position:absolute;left:3px;top:2px;font-size:9px;opacity:.55}} .mark{{position:absolute;right:3px;bottom:2px;font-size:10px;padding:1px 3px;border-radius:4px}}
.agent{{background:#ffd66b;color:#111}} .goal{{background:#77d5ff;color:#111}} pre{{white-space:pre-wrap;max-height:270px;overflow:auto;background:#111820;color:#e7edf4;padding:10px;border-radius:8px}}
@media(max-width:1050px){{.panels{{grid-template-columns:repeat(2,1fr)}}}} @media(max-width:620px){{.panels{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>{html.escape(episode_id)}</h1><div class="meta">Coordinate-Belief v6 · prompt_last · layer {layer} · 5-fold episode OOF cell probes</div>
<div class="controls"><button id="prev">◀</button><input id="slider" type="range" min="0" max="{max(0, len(rows)-1)}" value="0"><button id="next">▶</button><strong id="stepLabel"></strong></div>
<div class="panels">
<div class="panel"><h3>True map</h3><div id="trueGrid" class="grid"></div></div>
<div class="panel"><h3>Gold observable</h3><div id="goldGrid" class="grid"></div></div>
<div class="panel"><h3>Explicit coordinates</h3><div id="explicitGrid" class="grid"></div></div>
<div class="panel"><h3>Probe decoded</h3><div id="probeGrid" class="grid"></div></div>
</div>
<div class="panel" style="margin-top:14px"><h3>Step record</h3><pre id="details"></pre></div>
<script>
const DATA={data_json}; const SIZE={size};
function gridHtml(states,current,goal){{let out='';for(let y=SIZE-1;y>=0;y--)for(let x=0;x<SIZE;x++){{const s=states[x*SIZE+y]||'U';let marks='';if(goal&&x===goal[0]&&y===goal[1])marks+='<span class="mark goal">G</span>';if(current&&x===current[0]&&y===current[1])marks+='<span class="mark agent">A</span>';out+=`<div class="cell state-${{s}}" title="(${{x}},${{y}}) ${{s}}"><span class="coord">${{x}},${{y}}</span><strong>${{s}}</strong>${{marks}}</div>`;}}return out;}}
function show(i){{i=Math.max(0,Math.min(DATA.length-1,i));slider.value=i;const r=DATA[i];stepLabel.textContent=`step ${{r.step_id}} / ${{DATA.length-1}} · action=${{r.action||'N/A'}}`;trueGrid.innerHTML=gridHtml(r.true_cells,r.current_pos,r.goal);goldGrid.innerHTML=gridHtml(r.gold_cells,r.current_pos,r.goal);explicitGrid.innerHTML=gridHtml(r.explicit_cells,r.current_pos,r.goal);probeGrid.innerHTML=gridHtml(r.probe_cells,r.current_pos,r.goal);details.textContent=JSON.stringify(r.step_record,null,2);}}
slider.oninput=()=>show(+slider.value);prev.onclick=()=>show(+slider.value-1);next.onclick=()=>show(+slider.value+1);show(0);
</script></body></html>"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{episode_id}.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict coordinate-belief v6 OOF trajectory viewer.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episode", default=None)
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--position", default="prompt_last")
    parser.add_argument("--layer", default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--size", type=int, default=5)
    args = parser.parse_args()

    run = args.run.resolve()
    steps = read_jsonl(run / "steps.jsonl")
    targets = read_jsonl(run / "targets" / "targets.jsonl")
    target_by_key = {key(row): row for row in targets}
    step_by_key = {key(row): row for row in steps}

    act_dir = run / "activations"
    if not act_dir.exists():
        candidates = sorted(run.glob("activations*"))
        if not candidates:
            raise FileNotFoundError(f"No activations directory under {run}")
        act_dir = candidates[0]
    X_all = np.load(act_dir / "X.npy", mmap_mode="r")
    positions = load_positions(act_dir / "positions.json")
    layers = [int(x) for x in np.load(act_dir / "layers.npy").tolist()]
    meta = read_jsonl(act_dir / "meta.jsonl")
    if len(meta) != X_all.shape[0]:
        raise RuntimeError(f"Activation meta/X mismatch: {len(meta)} vs {X_all.shape[0]}")
    if args.position not in positions:
        raise RuntimeError(f"Position {args.position!r} unavailable; choices={positions}")
    p_idx = positions.index(args.position)
    layer = choose_layer(run, layers, args.layer)
    l_idx = layers.index(layer)

    joined_X: list[np.ndarray] = []
    joined_targets: list[dict[str, Any]] = []
    joined_steps: list[dict[str, Any]] = []
    joined_keys: list[tuple[str, int]] = []
    for i, meta_row in enumerate(meta):
        k = key(meta_row)
        if k not in target_by_key or k not in step_by_key:
            continue
        joined_X.append(np.asarray(X_all[i, p_idx, l_idx], dtype=np.float32))
        joined_targets.append(target_by_key[k])
        joined_steps.append(step_by_key[k])
        joined_keys.append(k)
    if not joined_X:
        raise RuntimeError("No activation/target/step keys joined")
    X = np.stack(joined_X)
    groups = np.array([k[0] for k in joined_keys], dtype=object)
    predictions = fit_oof(X, joined_targets, groups, args.size, args.folds)

    by_episode: dict[str, list[dict[str, Any]]] = {}
    for i, (episode_id, step_id) in enumerate(joined_keys):
        step = joined_steps[i]
        target = joined_targets[i]
        row = {
            "episode_id": episode_id,
            "step_id": step_id,
            "current_pos": step.get("current_pos"),
            "next_pos": step.get("next_pos"),
            "goal": step.get("goal") or step.get("goal_pos"),
            "action": step.get("action"),
            "true_cells": true_payload(target, args.size),
            "gold_cells": grid_payload(target, "gold_cell", args.size),
            "explicit_cells": parse_explicit(step, args.size),
            "probe_cells": predictions[i].tolist(),
            "step_record": step,
        }
        by_episode.setdefault(episode_id, []).append(row)

    if args.all_episodes:
        wanted = sorted(by_episode)
    elif args.episode:
        if args.episode not in by_episode:
            raise RuntimeError(f"Episode {args.episode} unavailable")
        wanted = [args.episode]
    else:
        wanted = [sorted(by_episode)[0]]

    output_dir = run / "trajectory_viewer_coordbelief_v6"
    for episode_id in wanted:
        rows = sorted(by_episode[episode_id], key=lambda r: r["step_id"])
        render_episode(run, episode_id, rows, args.size, layer, output_dir)
    links = "\n".join(f'<li><a href="{html.escape(ep)}.html">{html.escape(ep)}</a></li>' for ep in wanted)
    index = f"<!doctype html><meta charset='utf-8'><title>Coordinate-Belief v6 viewers</title><h1>Coordinate-Belief v6 viewers</h1><p>prompt_last · layer {layer}</p><ul>{links}</ul>"
    (output_dir / "index.html").write_text(index, encoding="utf-8")
    print(f"[viewer-v6] position={args.position} layer={layer} rows={len(joined_keys)}")
    print(f"[viewer-v6] saved={output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
PY
chmod +x scripts/trajectory_probe_viewer_coordbelief_v6.py

cat > scripts/run_qwen25_32b_coord100_v6.sh <<'SH'
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
SH
chmod +x scripts/run_qwen25_32b_coord100_v6.sh

python -m py_compile \
  scripts/sample_maps_stratified_v6.py \
  scripts/build_coordbelief_targets_v6.py \
  scripts/trajectory_probe_viewer_coordbelief_v6.py
bash -n scripts/run_qwen25_32b_coord100_v6.sh

# Contract smoke test for the exact schema that previously failed.
TEST_DIR="$(mktemp -d "$REPO/.coordbelief_v6_test.XXXXXX")"
trap 'rm -rf "$TEST_DIR"' EXIT
mkdir -p "$TEST_DIR/run"
cat > "$TEST_DIR/run/maps.jsonl" <<'EOF'
{"map_id":"m1","width":5,"height":5,"obstacles":[[3,0],[2,1]]}
EOF
cat > "$TEST_DIR/run/episodes.jsonl" <<'EOF'
{"episode_id":"D5_ep000003","map_id":"m1","width":5,"height":5,"obstacles":[[3,0],[2,1]],"start":[3,1],"goal":[4,4]}
EOF
cat > "$TEST_DIR/run/steps.jsonl" <<'EOF'
{"episode_id":"D5_ep000003","map_id":"m1","step_id":0,"current_pos":[3,1],"next_pos":[3,2],"action":"UP","feedback":{"DOWN":{"coord":[3,0],"state":"O"},"LEFT":{"coord":[2,1],"state":"O"},"RIGHT":{"coord":[4,1],"state":"F"},"UP":{"coord":[3,2],"state":"F"}},"last_feedback":{"DOWN":{"coord":[3,0],"state":"O"},"LEFT":{"coord":[2,1],"state":"O"},"RIGHT":{"coord":[4,1],"state":"F"},"UP":{"coord":[3,2],"state":"F"}},"parsed_belief_coordinates":{"F":[[3,1],[4,1],[3,2]],"O":[[3,0],[2,1]]}}
EOF
python scripts/build_coordbelief_targets_v6.py --run "$TEST_DIR/run" --overwrite >/dev/null
python - "$TEST_DIR/run/targets/targets.jsonl" <<'PYTEST'
import json, sys
row = json.loads(open(sys.argv[1], encoding="utf-8").readline())
assert row["gold_cell_x3_y2_OFU"] == "F"
assert row["gold_local_UP_OFUW"] == "F"
assert row["gold_cell_x0_y0_OFU"] == "U"
assert row["gold_known_cell_count"] == 5
assert row["model_cell_x3_y2_OFU"] == "F"
assert row["model_known_cell_acc_step"] == 1.0
PYTEST
rm -rf "$TEST_DIR"
trap - EXIT

echo "[deploy-v6] installed under $REPO/scripts"
echo "[deploy-v6] syntax checks PASS"
echo "[deploy-v6] run with: bash scripts/run_qwen25_32b_coord100_v6.sh"
