#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
ROOT="$(cd "$ROOT" && pwd)"
mkdir -p "$ROOT/scripts"

cat > "$ROOT/scripts/audit_rebuild_coordbelief_v6.py" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

CELL_RE = re.compile(r"^(?:gold|true)_cell_x(\d+)_y(\d+)_(?:OFU|FO)$")
GOLD_CELL_RE = re.compile(r"^gold_cell_x(\d+)_y(\d+)_OFU$")
TRUE_CELL_RE = re.compile(r"^true_cell_x(\d+)_y(\d+)_FO$")
ACTIONS = {
    "UP": (0, 1),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            row["_source_line"] = line_no
            rows.append(row)
    return rows

def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((p for p in paths if p.is_file()), None)

def ep_of(row: dict[str, Any]) -> str:
    for key in ("episode_id", "episode", "trajectory_id", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""

def step_of(row: dict[str, Any]) -> int:
    for key in ("step_id", "step", "timestep", "turn_id"):
        value = row.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    raise KeyError(f"Missing integer step id in row keys={sorted(row)}")

def as_coord(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return x, y

def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None

def prompt_text(row: dict[str, Any]) -> str:
    for key in ("prompt_text", "prompt", "rendered_prompt", "model_prompt"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""

def output_text(row: dict[str, Any]) -> str:
    for key in ("raw_response_text", "raw_response", "response", "model_output"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""

def extract_tag_json(text: str, names: Iterable[str]) -> Any:
    for name in names:
        pattern = rf"<{re.escape(name)}>\s*(.*?)\s*</{re.escape(name)}>"
        match = re.search(pattern, text, flags=re.S | re.I)
        if match:
            parsed = parse_jsonish(match.group(1))
            if parsed is not None:
                return parsed
    return None

def extract_feedback_before_action(row: dict[str, Any]) -> dict[str, Any]:
    # The prompt is the authority for what the model had actually seen.
    prompt = prompt_text(row)
    parsed = extract_tag_json(
        prompt,
        ("last_feedback", "feedback", "current_feedback", "observation"),
    )
    if isinstance(parsed, dict):
        return parsed

    # Explicitly named pre-action fields are acceptable fallbacks.
    for key in (
        "feedback_before_action",
        "last_feedback_before_action",
        "prompt_feedback",
        "last_feedback",
    ):
        value = parse_jsonish(row.get(key))
        if isinstance(value, dict):
            return value

    raise RuntimeError(
        f"Cannot recover action-before feedback for {ep_of(row)}/{step_of(row)}. "
        "Do not substitute env_feedback/post-action feedback."
    )

def extract_explicit_coordinates(row: dict[str, Any]) -> dict[str, Any]:
    candidates: list[Any] = [
        row.get("parsed_belief_coordinates"),
        (row.get("parsed_response") or {}).get("belief_coordinates")
        if isinstance(row.get("parsed_response"), dict) else None,
        (row.get("parsed_response") or {}).get("belief")
        if isinstance(row.get("parsed_response"), dict) else None,
    ]
    raw = parse_jsonish(output_text(row))
    if isinstance(raw, dict):
        candidates.extend([raw.get("belief_coordinates"), raw.get("belief")])

    for candidate in candidates:
        if isinstance(candidate, dict) and ("F" in candidate or "O" in candidate):
            return candidate

    # Last resort: locate JSON object in a longer raw response.
    text = output_text(row)
    if text:
        match = re.search(r'"belief_coordinates"\s*:\s*(\{.*?\})\s*(?:,\s*"action"|})', text, re.S)
        if match:
            candidate = parse_jsonish(match.group(1))
            if isinstance(candidate, dict):
                return candidate

    raise RuntimeError(
        f"Missing belief_coordinates for {ep_of(row)}/{step_of(row)} "
        f"(parse_error={row.get('parse_error')}, repaired={row.get('repaired')})"
    )

def coord_sets_to_map(
    obj: dict[str, Any],
    width: int,
    height: int,
) -> dict[tuple[int, int], str]:
    result = {(x, y): "U" for y in range(height) for x in range(width)}
    seen: dict[tuple[int, int], str] = {}

    for label in ("F", "O"):
        values = obj.get(label, [])
        if values is None:
            values = []
        if not isinstance(values, list):
            raise ValueError(f"belief_coordinates[{label!r}] must be a list")
        for raw_coord in values:
            coord = as_coord(raw_coord)
            if coord is None:
                raise ValueError(f"Invalid {label} coordinate: {raw_coord!r}")
            x, y = coord
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Out-of-bounds {label} coordinate: {coord}")
            if coord in seen and seen[coord] != label:
                raise ValueError(f"F/O overlap at {coord}")
            seen[coord] = label
            result[coord] = label
    return result

def map_to_topdown_rows(
    values: dict[tuple[int, int], str],
    width: int,
    height: int,
) -> list[list[str]]:
    # Row 0 is y=height-1. This is the only matrix convention allowed.
    return [[values[(x, y)] for x in range(width)] for y in range(height - 1, -1, -1)]

def matrix_to_map(
    grid: Any,
    width: int,
    height: int,
    *,
    topdown: bool,
) -> dict[tuple[int, int], str] | None:
    if not isinstance(grid, list) or len(grid) != height:
        return None
    if any(not isinstance(row, list) or len(row) != width for row in grid):
        return None
    result: dict[tuple[int, int], str] = {}
    for row_idx, row in enumerate(grid):
        y = height - 1 - row_idx if topdown else row_idx
        for x, value in enumerate(row):
            result[(x, y)] = str(value).upper()
    return result

def infer_size(rows: list[dict[str, Any]]) -> tuple[int, int]:
    for row in rows:
        size = row.get("grid_size")
        if isinstance(size, int):
            return size, size
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return int(size[0]), int(size[1])
        for key in ("width", "grid_width"):
            if row.get(key) is not None:
                width = int(row[key])
                height = int(row.get("height", row.get("grid_height", width)))
                return width, height
    return 5, 5

def current_pos(row: dict[str, Any]) -> tuple[int, int]:
    for key in ("current_pos", "agent_pos_before", "position", "pos"):
        coord = as_coord(row.get(key))
        if coord is not None:
            return coord
    feedback = extract_feedback_before_action(row)
    coord = as_coord(feedback.get("position"))
    if coord is not None:
        return coord
    raise RuntimeError(f"Missing current position for {ep_of(row)}/{step_of(row)}")

def goal_pos(row: dict[str, Any]) -> tuple[int, int] | None:
    for key in ("goal", "goal_pos", "target_pos"):
        coord = as_coord(row.get(key))
        if coord is not None:
            return coord
    prompt = prompt_text(row)
    parsed = extract_tag_json(prompt, ("goal", "goal_pos", "target"))
    return as_coord(parsed)

def update_gold(
    gold: dict[tuple[int, int], str],
    feedback: dict[str, Any],
    pos: tuple[int, int],
    goal: tuple[int, int] | None,
    width: int,
    height: int,
) -> None:
    free_keys = ("free", "free_cells", "known_free")
    obstacle_keys = ("blocked", "obstacles", "occupied", "known_obstacles")

    def apply(keys: tuple[str, ...], label: str) -> None:
        for key in keys:
            values = feedback.get(key)
            if values is None:
                continue
            if not isinstance(values, list):
                raise ValueError(f"feedback[{key!r}] must be a list")
            for raw_coord in values:
                coord = as_coord(raw_coord)
                if coord is None:
                    raise ValueError(f"Invalid feedback coordinate {raw_coord!r}")
                x, y = coord
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError(f"In-grid feedback {key} contains out-of-bounds {coord}")
                previous = gold[coord]
                if previous in {"F", "O"} and previous != label:
                    raise ValueError(
                        f"Non-monotone gold conflict at {coord}: {previous} -> {label}"
                    )
                gold[coord] = label

    apply(free_keys, "F")
    apply(obstacle_keys, "O")
    if pos in gold:
        gold[pos] = "F"
    if goal is not None and goal in gold:
        gold[goal] = "F"

def extract_true_map_from_row(
    row: dict[str, Any],
    width: int,
    height: int,
) -> dict[tuple[int, int], str] | None:
    result: dict[tuple[int, int], str] = {}
    for key, value in row.items():
        match = TRUE_CELL_RE.match(key)
        if match:
            x, y = map(int, match.groups())
            result[(x, y)] = str(value).upper()
    if len(result) == width * height:
        if set(result.values()) <= {"F", "O"}:
            return result

    for key in ("true_map", "obstacle_map", "map", "grid"):
        grid = row.get(key)
        parsed = parse_jsonish(grid)
        if parsed is None:
            continue
        # F/O character grid.
        if isinstance(parsed, list) and len(parsed) == height:
            mapped = matrix_to_map(parsed, width, height, topdown=True)
            if mapped and set(mapped.values()) <= {"F", "O", ".", "#", "0", "1"}:
                normalized: dict[tuple[int, int], str] = {}
                for coord, value in mapped.items():
                    normalized[coord] = "O" if value in {"O", "#", "1"} else "F"
                return normalized
        # Coordinate obstacle list.
        if isinstance(parsed, dict):
            obstacles = parsed.get("obstacles") or parsed.get("blocked")
            if isinstance(obstacles, list):
                result = {(x, y): "F" for y in range(height) for x in range(width)}
                for raw_coord in obstacles:
                    coord = as_coord(raw_coord)
                    if coord in result:
                        result[coord] = "O"
                return result
    return None

def existing_target_path(run: Path) -> Path | None:
    return first_existing(
        (
            run / "targets" / "targets.jsonl",
            run / "probe_targets_A.jsonl",
            run / "targets.jsonl",
        )
    )

def print_grid(title: str, values: dict[tuple[int, int], str], width: int, height: int) -> None:
    print(f"\n{title}")
    for y in range(height - 1, -1, -1):
        print(f"y={y}  " + " ".join(values[(x, y)] for x in range(width)))
    print("     " + " ".join(f"x={x}" for x in range(width)))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episode")
    parser.add_argument("--step", type=int)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--output-targets",
        type=Path,
        default=None,
        help="Default: RUN/targets_coord_v6/targets.jsonl",
    )
    args = parser.parse_args()

    run = args.run.resolve()
    steps_path = run / "steps.jsonl"
    if not steps_path.is_file():
        raise FileNotFoundError(steps_path)

    steps = read_jsonl(steps_path)
    width, height = infer_size(steps)
    print(f"[v6] run={run}")
    print(f"[v6] steps={len(steps)} size={width}x{height}")

    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_keys: list[tuple[str, int]] = []
    seen_keys: set[tuple[str, int]] = set()
    for row in steps:
        key = (ep_of(row), step_of(row))
        if key in seen_keys:
            duplicate_keys.append(key)
        seen_keys.add(key)
        by_episode[key[0]].append(row)
    if duplicate_keys:
        raise RuntimeError(f"Duplicate episode/step keys: {duplicate_keys[:10]}")

    for rows in by_episode.values():
        rows.sort(key=step_of)

    targets_path = existing_target_path(run)
    old_targets: dict[tuple[str, int], dict[str, Any]] = {}
    if targets_path:
        for row in read_jsonl(targets_path):
            old_targets[(ep_of(row), step_of(row))] = row
        print(f"[v6] existing_targets={targets_path} rows={len(old_targets)}")
    else:
        print("[v6] existing_targets=NONE")

    corrected_targets: list[dict[str, Any]] = []
    normalized_steps: list[dict[str, Any]] = []
    stats = Counter()
    episode_true_maps: dict[str, dict[tuple[int, int], str]] = {}

    for episode_id, episode_rows in sorted(by_episode.items()):
        gold = {(x, y): "U" for y in range(height) for x in range(width)}
        true_map: dict[tuple[int, int], str] | None = None

        for row in episode_rows:
            key = (episode_id, step_of(row))
            feedback = extract_feedback_before_action(row)
            pos = current_pos(row)
            goal = goal_pos(row)

            # Gold observable at this response includes exactly what was in this prompt.
            update_gold(gold, feedback, pos, goal, width, height)

            explicit_obj = extract_explicit_coordinates(row)
            explicit_map = coord_sets_to_map(explicit_obj, width, height)
            explicit_topdown = map_to_topdown_rows(explicit_map, width, height)

            existing_grid = row.get("parsed_belief_grid")
            if existing_grid is not None:
                as_topdown = matrix_to_map(existing_grid, width, height, topdown=True)
                as_bottomup = matrix_to_map(existing_grid, width, height, topdown=False)
                top_match = as_topdown == explicit_map
                bottom_match = as_bottomup == explicit_map
                if top_match:
                    stats["existing_matrix_topdown_match"] += 1
                elif bottom_match:
                    stats["existing_matrix_bottomup_match_VERTICAL_FLIP"] += 1
                else:
                    stats["existing_matrix_neither_match"] += 1

            old = old_targets.get(key)
            if true_map is None:
                for candidate in (old, row):
                    if isinstance(candidate, dict):
                        true_map = extract_true_map_from_row(candidate, width, height)
                        if true_map is not None:
                            break
            if true_map is not None:
                episode_true_maps[episode_id] = true_map
                for coord, value in gold.items():
                    if value != "U" and true_map[coord] != value:
                        raise RuntimeError(
                            f"Gold/true contradiction {key} at {coord}: "
                            f"gold={value} true={true_map[coord]}"
                        )

            normalized = copy.deepcopy(row)
            normalized["feedback_before_action"] = feedback
            normalized["parsed_belief_coordinates"] = {
                "F": [[x, y] for (x, y), v in explicit_map.items() if v == "F"],
                "O": [[x, y] for (x, y), v in explicit_map.items() if v == "O"],
            }
            normalized["parsed_belief_grid"] = explicit_topdown
            normalized["belief_grid_row_order"] = "top_to_bottom_y_descending"
            normalized["coordinate_semantics_version"] = "coordbelief-v6-cartesian-authoritative"
            normalized_steps.append(normalized)

            target = copy.deepcopy(old) if old is not None else {
                "episode_id": episode_id,
                "step_id": step_of(row),
            }
            for target_key in list(target):
                if GOLD_CELL_RE.match(target_key):
                    del target[target_key]

            target["episode_id"] = episode_id
            target["step_id"] = step_of(row)
            target["current_pos"] = [pos[0], pos[1]]
            target["gold_observable_semantics"] = (
                "cumulative_feedback_visible_in_prompt_before_action"
            )
            for y in range(height):
                for x in range(width):
                    target[f"gold_cell_x{x}_y{y}_OFU"] = gold[(x, y)]
                    if true_map is not None:
                        target[f"true_cell_x{x}_y{y}_FO"] = true_map[(x, y)]

            # Strict local labels from the current prompt-visible belief.
            for action, (dx, dy) in ACTIONS.items():
                nx, ny = pos[0] + dx, pos[1] + dy
                field = f"gold_local_{action}_OFUW"
                if not (0 <= nx < width and 0 <= ny < height):
                    target[field] = "WALL"
                else:
                    value = gold[(nx, ny)]
                    if value == "U":
                        raise RuntimeError(
                            f"{key}: adjacent {action} cell {(nx, ny)} is U after "
                            "applying exact current feedback; feedback timing/schema is wrong."
                        )
                    target[field] = value

            corrected_targets.append(target)

            if args.episode == episode_id and (
                args.step is None or args.step == step_of(row)
            ):
                print(f"\n[v6] selected={episode_id}/{step_of(row)}")
                print("belief_coordinates:")
                print(json.dumps(explicit_obj, ensure_ascii=False, indent=2))
                print_grid("Explicit belief (authoritative coordinates)", explicit_map, width, height)
                print_grid("Gold observable (prompt-visible feedback only)", gold, width, height)
                if true_map is not None:
                    print_grid("True map", true_map, width, height)

                if old is not None:
                    old_gold = {
                        (x, y): str(old.get(f"gold_cell_x{x}_y{y}_OFU", "?")).upper()
                        for y in range(height) for x in range(width)
                    }
                    mismatches = {
                        f"{x},{y}": {"old": old_gold[(x, y)], "strict": gold[(x, y)]}
                        for y in range(height) for x in range(width)
                        if old_gold[(x, y)] != gold[(x, y)]
                    }
                    print(f"\nOld target vs strict gold mismatches: {len(mismatches)}")
                    if mismatches:
                        print(json.dumps(mismatches, ensure_ascii=False, indent=2))

    print("\n[v6] matrix compatibility audit:")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")

    if stats["existing_matrix_bottomup_match_VERTICAL_FLIP"] > 0:
        print(
            "[v6] ERROR CONFIRMED: some parsed_belief_grid rows use y-ascending "
            "while the old pipeline expects top-down y-descending."
        )

    # Global target comparison.
    target_mismatch_rows = 0
    target_mismatch_cells = 0
    if old_targets:
        for target in corrected_targets:
            key = (ep_of(target), step_of(target))
            old = old_targets.get(key)
            if old is None:
                continue
            n = 0
            for y in range(height):
                for x in range(width):
                    field = f"gold_cell_x{x}_y{y}_OFU"
                    if str(old.get(field, "?")).upper() != target[field]:
                        n += 1
            if n:
                target_mismatch_rows += 1
                target_mismatch_cells += n
        print(
            f"[v6] old_gold_target_mismatch_rows={target_mismatch_rows} "
            f"cells={target_mismatch_cells}"
        )

    if args.write:
        normalized_path = run / "steps_coord_v6.jsonl"
        target_path = args.output_targets or run / "targets_coord_v6" / "targets.jsonl"
        write_jsonl(normalized_path, normalized_steps)
        write_jsonl(target_path, corrected_targets)

        report = {
            "schema": "coordbelief-semantic-audit-v6",
            "run": str(run),
            "steps": len(steps),
            "episodes": len(by_episode),
            "width": width,
            "height": height,
            "existing_target_path": str(targets_path) if targets_path else None,
            "old_gold_target_mismatch_rows": target_mismatch_rows,
            "old_gold_target_mismatch_cells": target_mismatch_cells,
            "matrix_audit": dict(stats),
            "normalized_steps": str(normalized_path),
            "corrected_targets": str(target_path),
        }
        report_path = run / "coordbelief_v6_audit.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[v6] wrote={normalized_path}")
        print(f"[v6] wrote={target_path}")
        print(f"[v6] wrote={report_path}")
    else:
        print("[v6] dry-run only; add --write to create corrected files.")

if __name__ == "__main__":
    main()

PY

cat > "$ROOT/scripts/trajectory_probe_viewer_coordbelief_v6.py" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit("Install scikit-learn: pip install scikit-learn") from exc

GOLD_RE = re.compile(r"^gold_cell_x(\d+)_y(\d+)_OFU$")
TRUE_RE = re.compile(r"^true_cell_x(\d+)_y(\d+)_FO$")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows

def ep_of(row: dict[str, Any]) -> str:
    for key in ("episode_id", "episode", "trajectory_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    return ""

def step_of(row: dict[str, Any]) -> int:
    for key in ("step_id", "step", "timestep", "turn_id"):
        if row.get(key) is not None:
            return int(row[key])
    raise KeyError("Missing step id")

def find_activation_dir(run: Path) -> Path:
    candidates = [
        run / "activations",
        run / "activations_A_multi",
        run / "activations_coord",
    ]
    for path in candidates:
        if (path / "X.npy").is_file() and (path / "meta.jsonl").is_file():
            return path
    raise FileNotFoundError(
        "Cannot find activations directory with X.npy and meta.jsonl. Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )

def load_positions(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, dict):
        if "positions" in value:
            return [str(x) for x in value["positions"]]
        return [str(k) for k, _ in sorted(value.items(), key=lambda kv: int(kv[1]))]
    raise RuntimeError(f"Unsupported positions.json schema: {type(value)}")

def parse_layer_arg(value: str, layers: list[int]) -> tuple[int, int]:
    if value == "auto":
        # For 32B, use a middle-late layer unless caller specifies a validated layer.
        target = layers[len(layers) * 2 // 3]
    else:
        target = int(value)
    if target in layers:
        return layers.index(target), target
    if 0 <= target < len(layers):
        return target, layers[target]
    raise ValueError(f"Layer {target} not available; layers={layers}")

def coord_map_from_row(row: dict[str, Any], prefix: str, width: int, height: int) -> dict[str, str]:
    pattern = GOLD_RE if prefix == "gold" else TRUE_RE
    result: dict[str, str] = {}
    for key, value in row.items():
        match = pattern.match(key)
        if match:
            x, y = map(int, match.groups())
            result[f"{x},{y}"] = str(value).upper()
    expected = width * height
    if len(result) != expected:
        raise RuntimeError(
            f"{prefix} map coverage {len(result)}/{expected} for "
            f"{ep_of(row)}/{step_of(row)}"
        )
    return result

def explicit_map_from_step(row: dict[str, Any], width: int, height: int) -> dict[str, str]:
    obj = row.get("parsed_belief_coordinates")
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"steps_coord_v6 missing parsed_belief_coordinates for "
            f"{ep_of(row)}/{step_of(row)}"
        )
    result = {f"{x},{y}": "U" for y in range(height) for x in range(width)}
    seen: dict[str, str] = {}
    for label in ("F", "O"):
        values = obj.get(label, [])
        if not isinstance(values, list):
            raise RuntimeError(f"belief_coordinates[{label}] is not a list")
        for coord in values:
            if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                raise RuntimeError(f"Invalid coordinate {coord!r}")
            x, y = int(coord[0]), int(coord[1])
            key = f"{x},{y}"
            if key not in result:
                raise RuntimeError(f"Out-of-bounds coordinate {(x, y)}")
            if key in seen and seen[key] != label:
                raise RuntimeError(f"F/O overlap at {(x, y)}")
            seen[key] = label
            result[key] = label
    return result

def prompt_of(row: dict[str, Any]) -> str:
    for key in ("prompt_text", "prompt", "rendered_prompt"):
        if isinstance(row.get(key), str):
            return row[key]
    return ""

def output_of(row: dict[str, Any]) -> str:
    for key in ("raw_response_text", "raw_response", "response"):
        if isinstance(row.get(key), str):
            return row[key]
    return ""

def position_of(row: dict[str, Any]) -> list[int] | None:
    for key in ("current_pos", "agent_pos_before", "position"):
        value = row.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return [int(value[0]), int(value[1])]
    return None

def encode_classes(classes: list[str], labels: np.ndarray) -> np.ndarray:
    lookup = {label: i for i, label in enumerate(classes)}
    return np.asarray([lookup[str(x)] for x in labels], dtype=np.int64)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--position", default="prompt_last")
    parser.add_argument("--layer", default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()

    run = args.run.resolve()
    steps_path = run / "steps_coord_v6.jsonl"
    targets_path = run / "targets_coord_v6" / "targets.jsonl"
    if not steps_path.is_file() or not targets_path.is_file():
        raise SystemExit(
            "Run audit/rebuild first:\n"
            f"python scripts/audit_rebuild_coordbelief_v6.py --run {run} --write"
        )

    steps = read_jsonl(steps_path)
    targets = read_jsonl(targets_path)
    steps_by_key = {(ep_of(r), step_of(r)): r for r in steps}
    targets_by_key = {(ep_of(r), step_of(r)): r for r in targets}

    episode_keys = sorted(
        [key for key in steps_by_key if key[0] == args.episode],
        key=lambda key: key[1],
    )
    if not episode_keys:
        matches = sorted({ep for ep, _ in steps_by_key if args.episode in ep})
        if len(matches) == 1:
            args.episode = matches[0]
            episode_keys = sorted(
                [key for key in steps_by_key if key[0] == args.episode],
                key=lambda key: key[1],
            )
        else:
            raise SystemExit(
                f"Episode {args.episode!r} not found. Substring matches={matches[:20]}"
            )

    width = height = 5
    first_target = targets_by_key[episode_keys[0]]
    xs = [int(m.group(1)) for k in first_target for m in [GOLD_RE.match(k)] if m]
    ys = [int(m.group(2)) for k in first_target for m in [GOLD_RE.match(k)] if m]
    if xs and ys:
        width, height = max(xs) + 1, max(ys) + 1

    act_dir = find_activation_dir(run)
    X = np.load(act_dir / "X.npy", mmap_mode="r")
    meta = read_jsonl(act_dir / "meta.jsonl")
    positions = load_positions(act_dir / "positions.json")
    raw_layers = np.load(act_dir / "layers.npy").tolist()
    layers = [int(x) for x in raw_layers]

    if args.position not in positions:
        raise SystemExit(
            f"Position {args.position!r} not available. positions={positions}\n"
            "Do not silently alias mean_current_belief_grid to coordinate belief."
        )
    pos_index = positions.index(args.position)
    layer_index, layer_value = parse_layer_arg(args.layer, layers)

    if len(meta) != X.shape[0]:
        raise RuntimeError(f"meta rows={len(meta)} but X rows={X.shape[0]}")
    if X.ndim != 4:
        raise RuntimeError(f"Expected X[rows,positions,layers,hidden], got {X.shape}")

    activation_by_key: dict[tuple[str, int], int] = {}
    for i, row in enumerate(meta):
        key = (ep_of(row), step_of(row))
        if key in activation_by_key:
            raise RuntimeError(f"Duplicate activation key {key}")
        activation_by_key[key] = i

    common_keys = sorted(
        set(activation_by_key) & set(targets_by_key),
        key=lambda key: (key[0], key[1]),
    )
    if not common_keys:
        raise RuntimeError("No activation/target keys align by (episode_id, step_id)")

    selected_available = [key for key in episode_keys if key in activation_by_key]
    missing_selected = [key for key in episode_keys if key not in activation_by_key]
    if missing_selected:
        print(
            f"[viewer/v6] selected episode has {len(missing_selected)} steps without "
            "activations; they will show decoded=N/A."
        )

    row_indices = np.asarray([activation_by_key[key] for key in common_keys], dtype=np.int64)
    features = np.asarray(X[row_indices, pos_index, layer_index, :], dtype=np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError("Activation features contain NaN/Inf")

    groups = np.asarray([key[0] for key in common_keys])
    unique_groups = sorted(set(groups.tolist()))
    folds = min(args.folds, len(unique_groups))
    if folds < 2:
        raise RuntimeError("Need at least two episodes for held-out decoding")

    splitter = GroupKFold(n_splits=folds)
    selected_split: tuple[np.ndarray, np.ndarray] | None = None
    for train_idx, test_idx in splitter.split(features, groups=groups):
        test_groups = set(groups[test_idx].tolist())
        if args.episode in test_groups:
            selected_split = train_idx, test_idx
            break
    if selected_split is None:
        raise RuntimeError(f"Could not assign episode {args.episode} to a fold")

    train_idx, test_idx = selected_split
    test_key_to_local = {common_keys[i]: j for j, i in enumerate(test_idx)}
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_idx])
    X_test = scaler.transform(features[test_idx])

    predictions: dict[tuple[str, int], dict[str, dict[str, Any]]] = {
        common_keys[i]: {} for i in test_idx
    }

    print(
        f"[viewer/v6] activations={act_dir} X={X.shape}\n"
        f"[viewer/v6] position={args.position} layer={layer_value} "
        f"train_rows={len(train_idx)} test_rows={len(test_idx)}"
    )

    tasks = [
        (x, y, f"gold_cell_x{x}_y{y}_OFU")
        for y in range(height) for x in range(width)
    ]

    for task_no, (x, y, task) in enumerate(tasks, 1):
        labels = np.asarray([str(targets_by_key[key][task]) for key in common_keys])
        train_labels = labels[train_idx]
        classes = sorted(set(train_labels.tolist()))

        if len(classes) == 1:
            pred_labels = np.asarray([classes[0]] * len(test_idx), dtype=object)
            probs = np.ones((len(test_idx), 1), dtype=np.float64)
            model_classes = classes
            decoder = "constant"
        else:
            y_train = encode_classes(classes, train_labels)
            model = LogisticRegression(
                max_iter=args.max_iter,
                class_weight="balanced",
                solver="lbfgs",
            )
            model.fit(X_train, y_train)
            pred_ids = model.predict(X_test)
            probs = model.predict_proba(X_test)
            model_classes = [classes[int(i)] for i in model.classes_.tolist()]
            pred_labels = np.asarray([classes[int(i)] for i in pred_ids], dtype=object)
            decoder = "logistic_regression"

        test_targets = labels[test_idx]
        acc = float(accuracy_score(test_targets, pred_labels))
        f1 = float(f1_score(test_targets, pred_labels, average="macro", zero_division=0))
        print(
            f"[viewer/v6] probe {task_no:02d}/{len(tasks)} {task} "
            f"decoder={decoder} acc={acc:.3f} macro_f1={f1:.3f}"
        )

        for local_i, global_i in enumerate(test_idx):
            key = common_keys[global_i]
            probability_map = {
                model_classes[j]: float(probs[local_i, j])
                for j in range(len(model_classes))
            }
            pred = str(pred_labels[local_i])
            target = str(test_targets[local_i])
            predictions[key][f"{x},{y}"] = {
                "label": pred,
                "target": target,
                "confidence": float(max(probability_map.values())),
                "probabilities": probability_map,
                "correct": pred == target,
                "decoder": decoder,
            }

    payload_steps = []
    true_map_reference: dict[str, str] | None = None
    for key in episode_keys:
        step_row = steps_by_key[key]
        target_row = targets_by_key[key]
        gold = coord_map_from_row(target_row, "gold", width, height)
        true_map = coord_map_from_row(target_row, "true", width, height)
        explicit = explicit_map_from_step(step_row, width, height)
        if true_map_reference is None:
            true_map_reference = true_map
        elif true_map != true_map_reference:
            raise RuntimeError(f"True map changed within episode at {key}")

        decoded = predictions.get(key)
        payload_steps.append(
            {
                "step_id": key[1],
                "current_pos": position_of(step_row),
                "action": step_row.get("parsed_action", step_row.get("action")),
                "gold": gold,
                "explicit": explicit,
                "decoded": decoded,
                "prompt": prompt_of(step_row),
                "output": output_of(step_row),
                "feedback_before_action": step_row.get("feedback_before_action"),
                "repaired": bool(step_row.get("repaired")),
                "parse_error": bool(step_row.get("parse_error")),
            }
        )

    assert true_map_reference is not None
    payload = {
        "schema": "coordbelief-trajectory-viewer-v6",
        "episode": args.episode,
        "width": width,
        "height": height,
        "true_map": true_map_reference,
        "activation": {"position": args.position, "layer": layer_value},
        "steps": payload_steps,
    }

    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coordinate-Belief v6 · {html.escape(args.episode)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#f5f5f5;color:#111}}
header,.controls,.panel{{background:white;border:1px solid #ccc;border-radius:10px;padding:14px;margin-bottom:14px}}
.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.maps{{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:12px}}
.mapPanel{{background:white;border:1px solid #ccc;border-radius:10px;padding:12px}}
.grid{{display:grid;gap:3px;align-items:stretch}}
.cell{{min-height:52px;border:1px solid #999;border-radius:5px;padding:4px;font-weight:700;display:flex;flex-direction:column;justify-content:center;align-items:center;position:relative}}
.cell small{{font-weight:400;font-size:10px}}
.F{{background:#dff5df}} .O{{background:#ffdede}} .U{{background:#eee}} .NA{{background:#fff3cd}}
.agent{{outline:4px solid #1d4ed8}} .wrong{{box-shadow:inset 0 0 0 3px #dc2626}}
pre{{white-space:pre-wrap;max-height:360px;overflow:auto;background:#111;color:#eee;padding:12px;border-radius:8px}}
.axis{{font-size:11px;color:#555}}
@media(max-width:1100px){{.maps{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<header>
<h1>Coordinate-Belief trajectory viewer v6</h1>
<div id="meta"></div>
</header>
<div class="controls">
<button id="prev">Previous</button><button id="next">Next</button>
<input id="slider" type="range" min="0" max="{max(0,len(payload_steps)-1)}" value="0" style="min-width:360px">
<span id="badge"></span>
</div>
<div class="maps">
<div class="mapPanel"><h3>True map</h3><div id="trueMap"></div></div>
<div class="mapPanel"><h3>Gold observable</h3><div id="goldMap"></div></div>
<div class="mapPanel"><h3>Explicit coordinate belief</h3><div id="explicitMap"></div></div>
<div class="mapPanel"><h3>Probe-decoded gold belief</h3><div id="decodedMap"></div></div>
</div>
<div class="panel"><h3>Diagnostics</h3><pre id="diag"></pre></div>
<div class="panel"><h3>Feedback visible before action</h3><pre id="feedback"></pre></div>
<div class="panel"><h3>Prompt</h3><pre id="prompt"></pre></div>
<div class="panel"><h3>Model output</h3><pre id="output"></pre></div>
<script>
const DATA={data_json};
let index=0;
const slider=document.getElementById('slider');
function cellValue(map,key,decoded){{
  if(decoded){{
    const r=map ? map[key] : null;
    if(!r) return {{label:'N/A', extra:'no activation', wrong:false}};
    return {{label:r.label, extra:`${{Math.round(100*r.confidence)}}% · gold=${{r.target}}`, wrong:!r.correct}};
  }}
  return {{label:map[key], extra:key, wrong:false}};
}}
function renderMap(id,map,decoded=false){{
  const root=document.getElementById(id); root.innerHTML='';
  const grid=document.createElement('div'); grid.className='grid';
  grid.style.gridTemplateColumns=`repeat(${{DATA.width}},1fr)`;
  const step=DATA.steps[index];
  for(let y=DATA.height-1;y>=0;y--){{
    for(let x=0;x<DATA.width;x++){{
      const key=`${{x}},${{y}}`; const r=cellValue(map,key,decoded);
      const el=document.createElement('div');
      const cls=(r.label==='N/A'?'NA':r.label);
      el.className=`cell ${{cls}}${{r.wrong?' wrong':''}}`;
      if(step.current_pos && step.current_pos[0]===x && step.current_pos[1]===y) el.className+=' agent';
      el.innerHTML=`<div>${{r.label}}</div><small>(${{x}},${{y}}) · ${{r.extra}}</small>`;
      grid.appendChild(el);
    }}
  }}
  root.appendChild(grid);
}}
function render(){{
  const step=DATA.steps[index]; slider.value=index;
  document.getElementById('badge').textContent=`Step ${{index+1}}/${{DATA.steps.length}} · id=${{step.step_id}} · action=${{step.action}}`;
  document.getElementById('meta').textContent=`episode=${{DATA.episode}} · Cartesian origin bottom-left · x→right · y→up · activation=${{DATA.activation.position}}/L${{DATA.activation.layer}}`;
  renderMap('trueMap',DATA.true_map,false);
  renderMap('goldMap',step.gold,false);
  renderMap('explicitMap',step.explicit,false);
  renderMap('decodedMap',step.decoded,true);
  const wrong=[];
  if(step.decoded){{
    for(const [coord,r] of Object.entries(step.decoded)){{
      if(!r.correct) wrong.push(`${{coord}}: pred=${{r.label}} gold=${{r.target}} conf=${{r.confidence.toFixed(3)}}`);
    }}
  }} else wrong.push('Probe decoding unavailable: this step has no extracted activation.');
  document.getElementById('diag').textContent=wrong.length?wrong.join(String.fromCharCode(10)):'All decoded cells match gold.';
  document.getElementById('feedback').textContent=JSON.stringify(step.feedback_before_action,null,2);
  document.getElementById('prompt').textContent=step.prompt||'';
  document.getElementById('output').textContent=step.output||'';
}}
document.getElementById('prev').onclick=()=>{{index=Math.max(0,index-1);render();}};
document.getElementById('next').onclick=()=>{{index=Math.min(DATA.steps.length-1,index+1);render();}};
slider.oninput=()=>{{index=Number(slider.value);render();}};
render();
</script>
</body></html>"""

    out_dir = run / "trajectory_viewer_coord_v6"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.episode}.html"
    out_path.write_text(page, encoding="utf-8")
    print(f"[viewer/v6] output={out_path}")

if __name__ == "__main__":
    main()

PY

chmod +x   "$ROOT/scripts/audit_rebuild_coordbelief_v6.py"   "$ROOT/scripts/trajectory_probe_viewer_coordbelief_v6.py"

python -m py_compile   "$ROOT/scripts/audit_rebuild_coordbelief_v6.py"   "$ROOT/scripts/trajectory_probe_viewer_coordbelief_v6.py"

echo "[deploy/v6] installed:"
echo "  $ROOT/scripts/audit_rebuild_coordbelief_v6.py"
echo "  $ROOT/scripts/trajectory_probe_viewer_coordbelief_v6.py"
echo "[deploy/v6] py_compile=PASS"
