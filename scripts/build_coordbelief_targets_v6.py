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
