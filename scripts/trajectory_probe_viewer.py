#!/usr/bin/env python3
"""Generate a self-contained Cartesian trajectory viewer for grid-world runs.

The viewer intentionally keeps four map concepts separate:
  * true map: objective F/O map from true_cell_x*_y*_FO targets
  * gold belief: correct observable F/O/U belief from gold_cell_x*_y*_OFU
  * explicit belief: model-written belief from explicit_cell targets or response JSON
  * probe belief: leave-one-episode-out decoded gold-cell labels

Cartesian rendering convention:
  x increases to the right, y increases upward, and (0, 0) is bottom-left.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TRUE_CELL_RE = re.compile(r"^true_cell_x(\d+)_y(\d+)_FO$")
GOLD_CELL_RE = re.compile(r"^gold_cell_x(\d+)_y(\d+)_OFU$")
EXPLICIT_CELL_RES = (
    re.compile(r"^explicit_cell_x(\d+)_y(\d+)_OFU$"),
    re.compile(r"^model_cell_x(\d+)_y(\d+)_OFU$"),
    re.compile(r"^parsed_cell_x(\d+)_y(\d+)_OFU$"),
)
KNOWN_CELL_RE = re.compile(r"^gold_cell_x(\d+)_y(\d+)_known$")
LOCAL_RE = re.compile(r"^gold_local_(UP|DOWN|LEFT|RIGHT)_OFUW$")
TRUE_ACTION_RE = re.compile(r"^(?:true|requested)_action_(UP|DOWN|LEFT|RIGHT)_is_astar_best$")

ACTION_DELTA = {
    "UP": (0, 1),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

MISSING = object()


@dataclass(frozen=True)
class ActivationStore:
    X: np.ndarray
    positions: list[str]
    layers: list[int]
    meta: list[dict[str, Any]]
    position_mask: np.ndarray | None


class ViewerError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ViewerError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ViewerError(f"Expected object at {path}:{line_no}, got {type(value).__name__}")
            rows.append(value)
    return rows


def first_present(row: Mapping[str, Any] | None, keys: Sequence[str], default: Any = None) -> Any:
    if row is None:
        return default
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def episode_id_of(row: Mapping[str, Any]) -> str:
    value = first_present(row, ("episode_id", "episode", "id"), "")
    return str(value)


def step_id_of(row: Mapping[str, Any], fallback: int) -> Any:
    return first_present(row, ("step_id", "step", "t", "turn", "index"), fallback)


def sortable_step(value: Any) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def filtered_episode(rows: Sequence[dict[str, Any]], episode_id: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if episode_id_of(row) == episode_id]
    return sorted(selected, key=lambda row: sortable_step(step_id_of(row, 0)))


def parse_coord(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        x = first_present(value, ("x", "col", "column"), MISSING)
        y = first_present(value, ("y", "row"), MISSING)
        if x is not MISSING and y is not MISSING:
            try:
                return int(x), int(y)
            except (TypeError, ValueError):
                return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        match = re.search(r"(-?\d+)\s*[, ]\s*(-?\d+)", value)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def coord_from_row(row: Mapping[str, Any], kind: str) -> tuple[int, int] | None:
    if kind == "current":
        keys = ("current_pos", "current_position", "position", "pos")
    elif kind == "next":
        keys = ("next_pos", "next_position", "position_after", "result_pos")
    elif kind == "start":
        keys = ("start", "start_pos", "start_position")
    elif kind == "goal":
        keys = ("goal", "goal_pos", "goal_position", "target", "target_pos")
    else:
        raise ValueError(kind)
    return parse_coord(first_present(row, keys))


def normalize_action(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    aliases = {"N": "UP", "S": "DOWN", "W": "LEFT", "E": "RIGHT"}
    text = aliases.get(text, text)
    return text if text in ACTION_DELTA else text or None


def action_from_row(row: Mapping[str, Any], executed: bool = False) -> str | None:
    keys = (
        ("executed_action", "actual_action", "applied_action", "action_executed")
        if executed
        else ("requested_action", "chosen_action", "parsed_action", "action")
    )
    return normalize_action(first_present(row, keys))


def normalize_cell(value: Any, *, allow_unknown: bool) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    mapping = {
        "FREE": "F",
        "OPEN": "F",
        ".": "F",
        "0": "F",
        "OBSTACLE": "O",
        "BLOCKED": "O",
        "BLOCK": "O",
        "#": "O",
        "1": "O",
        "UNKNOWN": "U",
        "UNSEEN": "U",
        "?": "U",
        "WALL": "O",
    }
    text = mapping.get(text, text)
    allowed = {"F", "O", "U"} if allow_unknown else {"F", "O"}
    return text if text in allowed else None


def extract_cells(row: Mapping[str, Any], pattern: re.Pattern[str], *, allow_unknown: bool) -> dict[tuple[int, int], str]:
    cells: dict[tuple[int, int], str] = {}
    for key, raw in row.items():
        match = pattern.match(str(key))
        if not match:
            continue
        label = normalize_cell(raw, allow_unknown=allow_unknown)
        if label is None:
            raise ViewerError(f"Invalid cell label {raw!r} for {key}")
        x, y = int(match.group(1)), int(match.group(2))
        cells[(x, y)] = label
    return cells


def extract_explicit_target_cells(row: Mapping[str, Any]) -> dict[tuple[int, int], str]:
    for pattern in EXPLICIT_CELL_RES:
        cells = extract_cells(row, pattern, allow_unknown=True)
        if cells:
            return cells
    return {}


def true_map_from_targets(rows: Sequence[dict[str, Any]]) -> tuple[dict[tuple[int, int], str], int, int]:
    canonical: dict[tuple[int, int], str] | None = None
    for row in rows:
        cells = extract_cells(row, TRUE_CELL_RE, allow_unknown=False)
        if not cells:
            continue
        if canonical is None:
            canonical = cells
        elif cells != canonical:
            raise ViewerError("True-map labels change between steps in the same episode")
    if not canonical:
        available = sorted({key for row in rows for key in row if str(key).startswith("true_cell")})
        raise ViewerError(
            "No true_cell_x*_y*_FO fields were found in targets. "
            f"Available true-cell-like keys: {available[:20]}"
        )
    max_x = max(x for x, _ in canonical)
    max_y = max(y for _, y in canonical)
    width, height = max_x + 1, max_y + 1
    expected = {(x, y) for x in range(width) for y in range(height)}
    missing = sorted(expected - set(canonical))
    extra = sorted(set(canonical) - expected)
    if missing or extra:
        raise ViewerError(f"True map is not a complete rectangle. Missing={missing}, extra={extra}")
    invalid = {coord: value for coord, value in canonical.items() if value not in {"F", "O"}}
    if invalid:
        raise ViewerError(f"True map contains non-F/O labels: {invalid}")
    return canonical, width, height


def coord_dict_to_json(cells: Mapping[tuple[int, int], Any]) -> dict[str, Any]:
    return {f"{x},{y}": value for (x, y), value in sorted(cells.items())}


def parse_json_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(fenced)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def find_belief_grid(step: Mapping[str, Any]) -> Any:
    for key in ("parsed_belief_grid", "belief_grid", "model_belief_grid"):
        if key in step and step[key] is not None:
            return step[key]
    parsed = first_present(step, ("parsed_response", "parsed_output"))
    if isinstance(parsed, Mapping) and "belief_grid" in parsed:
        return parsed["belief_grid"]
    raw = first_present(step, ("raw_response", "response", "model_output", "output"))
    obj = parse_json_maybe(raw)
    if isinstance(obj, Mapping):
        return obj.get("belief_grid")
    return None


def matrix_to_cells(
    matrix: Any,
    width: int,
    height: int,
    gold: Mapping[tuple[int, int], str],
) -> tuple[dict[tuple[int, int], str], str]:
    if isinstance(matrix, Mapping):
        cells: dict[tuple[int, int], str] = {}
        for raw_key, raw_value in matrix.items():
            coord = parse_coord(raw_key)
            if coord is None and isinstance(raw_key, str):
                match = re.search(r"x\s*(\d+)\D+y\s*(\d+)", raw_key, flags=re.IGNORECASE)
                if match:
                    coord = int(match.group(1)), int(match.group(2))
            label = normalize_cell(raw_value, allow_unknown=True)
            if coord is not None and label is not None:
                cells[coord] = label
        return cells, "coordinate mapping"
    if not isinstance(matrix, list) or len(matrix) != height:
        return {}, "unavailable"
    if not all(isinstance(row, list) and len(row) == width for row in matrix):
        return {}, "unavailable"

    def build(y0_first: bool) -> dict[tuple[int, int], str]:
        result: dict[tuple[int, int], str] = {}
        for row_index, row in enumerate(matrix):
            y = row_index if y0_first else height - 1 - row_index
            for x, raw in enumerate(row):
                label = normalize_cell(raw, allow_unknown=True)
                if label is not None:
                    result[(x, y)] = label
        return result

    candidates = [(build(True), "matrix row 0 = y=0"), (build(False), "matrix row 0 = y=max")]

    def agreement(cells: Mapping[tuple[int, int], str]) -> tuple[int, int]:
        compared = 0
        correct = 0
        for coord, label in cells.items():
            gold_label = gold.get(coord)
            if label == "U" or gold_label in (None, "U"):
                continue
            compared += 1
            correct += int(label == gold_label)
        return correct, compared

    scored = [(agreement(cells), cells, note) for cells, note in candidates]
    scored.sort(key=lambda item: (item[0][0], item[0][1]), reverse=True)
    _, best_cells, best_note = scored[0]
    return best_cells, f"{best_note} (auto-selected by observed-cell agreement)"


def scalar_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return scalar_json(value)


def loop_annotations(step_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    positions = [coord_from_row(row, "current") for row in step_rows]
    actions = [action_from_row(row, executed=False) for row in step_rows]
    seen_pos: dict[tuple[int, int], int] = {}
    seen_pair: dict[tuple[tuple[int, int], str], int] = {}
    annotations: list[dict[str, Any]] = []
    for index, (position, action) in enumerate(zip(positions, actions)):
        repeated_position = position is not None and position in seen_pos
        pair = (position, action) if position is not None and action is not None else None
        repeated_pair = pair is not None and pair in seen_pair
        previous_index = seen_pos.get(position) if position is not None else None
        cycle_length = index - previous_index if previous_index is not None else None
        short_cycle = cycle_length is not None and 1 <= cycle_length <= 6
        severity = 2 if repeated_pair or short_cycle else (1 if repeated_position else 0)
        annotations.append(
            {
                "repeated_position": repeated_position,
                "repeated_position_action": repeated_pair,
                "short_cycle": short_cycle,
                "cycle_length": cycle_length,
                "loop_flag": severity > 0,
                "loop_severity": severity,
            }
        )
        if position is not None and position not in seen_pos:
            seen_pos[position] = index
        if pair is not None and pair not in seen_pair:
            seen_pair[pair] = index
    return annotations


def loop_score(step_rows: Sequence[dict[str, Any]]) -> float:
    annotations = loop_annotations(step_rows)
    score = 0.0
    for ann in annotations:
        score += 1.0 * int(ann["repeated_position"])
        score += 2.0 * int(ann["repeated_position_action"])
        score += 2.0 * int(ann["short_cycle"])
    return score + len(step_rows) / 1000.0


def episode_summary_lookup(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {episode_id_of(row): row for row in rows if episode_id_of(row)}


def choose_episode(
    requested: str,
    steps: Sequence[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
) -> str:
    ids = sorted({episode_id_of(row) for row in steps if episode_id_of(row)})
    if not ids:
        raise ViewerError("No episode_id values found in steps.jsonl")
    if requested == "auto-loop":
        return max(ids, key=lambda ep: loop_score(filtered_episode(steps, ep)))
    if requested in ids:
        return requested
    matches = [ep for ep in ids if requested in ep]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ViewerError(f"No episode matches {requested!r}")
    raise ViewerError(f"Episode selector {requested!r} is ambiguous: {matches[:20]}")


def print_episode_list(steps: Sequence[dict[str, Any]], episodes: Sequence[dict[str, Any]]) -> None:
    summaries = episode_summary_lookup(episodes)
    ids = sorted({episode_id_of(row) for row in steps if episode_id_of(row)})
    records = []
    for ep in ids:
        ep_steps = filtered_episode(steps, ep)
        summary = summaries.get(ep, {})
        success = first_present(summary, ("success", "reached_goal", "completed"), "")
        records.append((loop_score(ep_steps), ep, len(ep_steps), success))
    records.sort(reverse=True)
    print("episode_id\tsteps\tsuccess\tloop_score")
    for score, ep, count, success in records:
        print(f"{ep}\t{count}\t{success}\t{score:.3f}")


def load_positions(path: Path) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [str(value) for value in obj]
    if isinstance(obj, Mapping):
        for key in ("positions", "names", "position_names"):
            if key in obj and isinstance(obj[key], list):
                return [str(value) for value in obj[key]]
        if all(isinstance(value, int) for value in obj.values()):
            return [key for key, _ in sorted(obj.items(), key=lambda item: item[1])]
    raise ViewerError(f"Unsupported positions.json format: {path}")


def load_activations(run: Path) -> ActivationStore:
    root = run / "activations"
    X_path = root / "X.npy"
    positions_path = root / "positions.json"
    layers_path = root / "layers.npy"
    meta_path = root / "meta.jsonl"
    for path in (X_path, positions_path, layers_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    X = np.load(X_path, mmap_mode="r")
    if X.ndim != 4:
        raise ViewerError(f"Expected activations X with 4 dimensions [N,P,L,H], got {X.shape}")
    positions = load_positions(positions_path)
    layers = [int(value) for value in np.load(layers_path).tolist()]
    meta = read_jsonl(meta_path)
    if len(meta) != X.shape[0]:
        raise ViewerError(f"Activation/meta row mismatch: X={X.shape[0]}, meta={len(meta)}")
    if len(positions) != X.shape[1]:
        raise ViewerError(f"Activation/position mismatch: X={X.shape[1]}, positions={len(positions)}")
    if len(layers) != X.shape[2]:
        raise ViewerError(f"Activation/layer mismatch: X={X.shape[2]}, layers={len(layers)}")
    mask_path = root / "position_mask.npy"
    position_mask = np.load(mask_path, mmap_mode="r") if mask_path.is_file() else None
    return ActivationStore(X=X, positions=positions, layers=layers, meta=meta, position_mask=position_mask)


def make_row_keys(rows: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    counters: defaultdict[str, int] = defaultdict(int)
    keys: list[tuple[str, str]] = []
    for row in rows:
        ep = episode_id_of(row)
        fallback = counters[ep]
        counters[ep] += 1
        step = step_id_of(row, fallback)
        keys.append((ep, str(step)))
    return keys


def target_lookup(target_rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for key, row in zip(make_row_keys(target_rows), target_rows):
        if key in lookup:
            raise ViewerError(f"Duplicate target key {key}")
        lookup[key] = row
    return lookup


def discover_probe_tasks(target_rows: Sequence[dict[str, Any]], groups: set[str]) -> list[tuple[str, str]]:
    keys = sorted({str(key) for row in target_rows for key in row})
    tasks: list[tuple[str, str]] = []
    for task in keys:
        group: str | None = None
        if GOLD_CELL_RE.match(task) or KNOWN_CELL_RE.match(task):
            group = "cells"
        elif LOCAL_RE.match(task):
            group = "local"
        elif (
            task in {
                "chosen_action_is_astar_best",
                "chosen_action_reduces_true_distance",
                "requested_action_is_astar_best",
                "requested_action_reduces_true_distance",
                "loop_risk",
                "position_seen_before",
                "position_action_seen_before",
            }
            or TRUE_ACTION_RE.match(task)
        ):
            group = "planning"
        if group in groups:
            tasks.append((task, group))
    return tasks


def load_result_rows(run: Path) -> list[dict[str, str]]:
    candidates = (
        run / "layer_curves" / "task_layer_curves.csv",
        run / "probes" / "probe_results.csv",
        run / "probes" / "best_by_task.csv",
        run / "probe_hierarchy_A" / "probe_results.csv",
        run / "probe_hierarchy_A" / "best_by_task.csv",
    )
    rows: list[dict[str, str]] = []
    for path in candidates:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def desired_position(group: str, probe_site: str) -> str:
    if probe_site != "decision":
        return probe_site
    return {
        "cells": "prompt_last",
        "planning": "pre_action_token",
        "local": "mean_last_feedback",
    }[group]


def choose_position_layer(
    task: str,
    group: str,
    probe_site: str,
    store: ActivationStore,
    result_rows: Sequence[dict[str, str]],
) -> tuple[str, int, dict[str, Any]]:
    position = desired_position(group, probe_site)
    if position not in store.positions:
        fallback_order = {
            "cells": ("prompt_last", "mean_current_belief_grid", "pre_action_token"),
            "planning": ("pre_action_token", "prompt_last", "mean_last_feedback"),
            "local": ("mean_last_feedback", "prompt_last"),
        }[group]
        position = next((candidate for candidate in fallback_order if candidate in store.positions), store.positions[0])

    matches = []
    for row in result_rows:
        if row.get("task") != task or row.get("position") != position:
            continue
        layer = int_or_none(row.get("layer"))
        score = float_or_none(row.get("macro_f1_mean"))
        if layer is None or score is None or layer not in store.layers:
            continue
        matches.append((score, layer, row))
    if matches:
        score, layer, row = max(matches, key=lambda item: item[0])
        reliability = {
            "macro_f1_mean": score,
            "macro_f1_std": float_or_none(row.get("macro_f1_std")),
            "majority_macro_f1_mean": float_or_none(row.get("majority_macro_f1_mean")),
        }
        return position, layer, reliability

    target = store.layers[int(round((len(store.layers) - 1) * 0.68))]
    return position, target, {
        "macro_f1_mean": None,
        "macro_f1_std": None,
        "majority_macro_f1_mean": None,
    }


def canonical_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    if text in {"True", "true"}:
        return "1"
    if text in {"False", "false"}:
        return "0"
    return text


def cache_path_for(run: Path, episode: str, groups: Sequence[str], probe_site: str) -> Path:
    suffix = "-".join(sorted(groups))
    return run / "trajectory_viewer_cache" / f"{episode}__{suffix}__{probe_site}.json"


def load_probe_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(obj, Mapping) and obj.get("schema_version") == "2.0-cartesian":
        return dict(obj)
    return None


def fit_step_probes(
    run: Path,
    episode: str,
    target_rows: Sequence[dict[str, Any]],
    groups: Sequence[str],
    probe_site: str,
    force: bool,
) -> dict[str, Any]:
    cache_path = cache_path_for(run, episode, groups, probe_site)
    if not force:
        cached = load_probe_cache(cache_path)
        if cached is not None:
            log(f"[viewer] loaded step-probe cache: {cache_path}")
            return cached

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ViewerError("scikit-learn is required for --fit-step-probes") from exc

    store = load_activations(run)
    targets = target_lookup(target_rows)
    meta_keys = make_row_keys(store.meta)
    result_rows = load_result_rows(run)
    tasks = discover_probe_tasks(target_rows, set(groups))
    if not tasks:
        raise ViewerError(f"No probe tasks found for groups={groups}")

    position_index = {name: index for index, name in enumerate(store.positions)}
    layer_index = {layer: index for index, layer in enumerate(store.layers)}

    episode_meta_indices = [index for index, key in enumerate(meta_keys) if key[0] == episode]
    if not episode_meta_indices:
        raise ViewerError(f"No activations found for episode {episode}")

    records: dict[str, dict[str, Any]] = defaultdict(dict)
    for task_number, (task, group) in enumerate(tasks, 1):
        position, layer, reliability = choose_position_layer(
            task, group, probe_site, store, result_rows
        )
        p_idx = position_index[position]
        l_idx = layer_index[layer]
        log(f"[viewer] probe {task_number:>3}/{len(tasks)} {task} @ {position}/L{layer}")

        train_indices: list[int] = []
        train_labels: list[str] = []
        test_indices: list[int] = []
        test_labels: list[str | None] = []
        test_keys: list[tuple[str, str]] = []

        for index, key in enumerate(meta_keys):
            target = targets.get(key)
            if target is None:
                continue
            label = canonical_label(target.get(task))
            if store.position_mask is not None and not bool(store.position_mask[index, p_idx]):
                continue
            if key[0] == episode:
                test_indices.append(index)
                test_labels.append(label)
                test_keys.append(key)
            elif label is not None:
                train_indices.append(index)
                train_labels.append(label)

        classes = sorted(set(train_labels))
        if len(classes) < 2 or not train_indices:
            for key, target_label in zip(test_keys, test_labels):
                records[key[1]][task] = {
                    "task": task,
                    "task_group": group,
                    "target": target_label,
                    "prediction": None,
                    "confidence": None,
                    "probabilities": {},
                    "correct": None,
                    "position": position,
                    "layer": layer,
                    "status": "insufficient training classes",
                    "reliability": reliability,
                }
            continue

        X_train = np.asarray(store.X[train_indices, p_idx, l_idx, :], dtype=np.float32)
        X_test = np.asarray(store.X[test_indices, p_idx, l_idx, :], dtype=np.float32)
        class_to_int = {label: idx for idx, label in enumerate(classes)}
        y_train = np.asarray([class_to_int[label] for label in train_labels], dtype=np.int64)

        classifier = LogisticRegression(
            max_iter=600,
            class_weight="balanced",
            solver="lbfgs",
            random_state=0,
        )
        classifier.fit(X_train, y_train)
        predicted_int = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)

        for row_index, (key, target_label) in enumerate(zip(test_keys, test_labels)):
            pred_label = classes[int(predicted_int[row_index])]
            prob_map = {
                classes[int(class_id)]: float(probabilities[row_index, class_index])
                for class_index, class_id in enumerate(classifier.classes_)
            }
            confidence = max(prob_map.values()) if prob_map else None
            records[key[1]][task] = {
                "task": task,
                "task_group": group,
                "target": target_label,
                "prediction": pred_label,
                "confidence": confidence,
                "probabilities": prob_map,
                "correct": (pred_label == target_label) if target_label is not None else None,
                "position": position,
                "layer": layer,
                "status": "ok",
                "reliability": reliability,
            }

    payload = {
        "schema_version": "2.0-cartesian",
        "episode_id": episode,
        "groups": sorted(groups),
        "probe_site": probe_site,
        "records": records,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(json_safe(payload), ensure_ascii=False), encoding="utf-8")
    log(f"[viewer] saved step-probe cache: {cache_path}")
    return payload


def align_step_targets(
    step_rows: Sequence[dict[str, Any]],
    target_rows: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    target_by_step: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(target_rows):
        target_by_step[str(step_id_of(row, index))] = row
    aligned: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, step in enumerate(step_rows):
        step_id = str(step_id_of(step, index))
        target = target_by_step.get(step_id)
        if target is None and index < len(target_rows):
            target = target_rows[index]
        if target is None:
            raise ViewerError(f"Could not align a target row for step {step_id}")
        aligned.append((step, target))
    return aligned


def probe_records_by_step(cache: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if cache is None:
        return {}
    records = cache.get("records", {})
    return dict(records) if isinstance(records, Mapping) else {}


def infer_start_goal(
    episode_row: Mapping[str, Any] | None,
    step_rows: Sequence[dict[str, Any]],
    target_rows: Sequence[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    sources: list[Mapping[str, Any]] = []
    if episode_row is not None:
        sources.append(episode_row)
    if step_rows:
        sources.append(step_rows[0])
    if target_rows:
        sources.append(target_rows[0])
    start = next((coord_from_row(source, "start") for source in sources if coord_from_row(source, "start") is not None), None)
    goal = next((coord_from_row(source, "goal") for source in sources if coord_from_row(source, "goal") is not None), None)
    if start is None:
        start = coord_from_row(step_rows[0], "current") if step_rows else None
    return start or (0, 0), goal or (width - 1, height - 1)


def validate_motion(step_rows: Sequence[dict[str, Any]], width: int, height: int) -> list[str]:
    warnings: list[str] = []
    for index, row in enumerate(step_rows):
        current = coord_from_row(row, "current")
        next_pos = coord_from_row(row, "next")
        action = action_from_row(row, executed=True) or action_from_row(row, executed=False)
        if current is not None and not (0 <= current[0] < width and 0 <= current[1] < height):
            warnings.append(f"step {step_id_of(row, index)} current_pos out of bounds: {current}")
        if next_pos is not None and not (0 <= next_pos[0] < width and 0 <= next_pos[1] < height):
            warnings.append(f"step {step_id_of(row, index)} next_pos out of bounds: {next_pos}")
        if current is not None and next_pos is not None and action in ACTION_DELTA:
            dx, dy = ACTION_DELTA[action]
            expected = (current[0] + dx, current[1] + dy)
            if next_pos != expected and next_pos != current:
                warnings.append(
                    f"step {step_id_of(row, index)} action/position mismatch: "
                    f"{current} + {action} -> expected {expected}, logged {next_pos}"
                )
    return warnings


def build_payload(
    run: Path,
    episode: str,
    step_rows: Sequence[dict[str, Any]],
    target_rows: Sequence[dict[str, Any]],
    episode_row: Mapping[str, Any] | None,
    probe_cache: Mapping[str, Any] | None,
) -> dict[str, Any]:
    true_map, width, height = true_map_from_targets(target_rows)
    aligned = align_step_targets(step_rows, target_rows)
    annotations = loop_annotations(step_rows)
    start, goal = infer_start_goal(episode_row, step_rows, target_rows, width, height)
    warnings = validate_motion(step_rows, width, height)
    probe_by_step = probe_records_by_step(probe_cache)

    payload_steps: list[dict[str, Any]] = []
    explicit_notes: Counter[str] = Counter()
    for index, ((step, target), loop_info) in enumerate(zip(aligned, annotations)):
        step_id = str(step_id_of(step, index))
        gold_cells = extract_cells(target, GOLD_CELL_RE, allow_unknown=True)
        if len(gold_cells) != width * height:
            raise ViewerError(
                f"Step {step_id}: expected {width * height} gold cells, found {len(gold_cells)}"
            )
        for coord, value in gold_cells.items():
            if value != "U" and value != true_map[coord]:
                raise ViewerError(
                    f"Step {step_id}: gold belief {coord}={value} disagrees with true map {true_map[coord]}"
                )

        explicit_cells = extract_explicit_target_cells(target)
        explicit_note = "explicit_cell targets"
        if not explicit_cells:
            explicit_cells, explicit_note = matrix_to_cells(
                find_belief_grid(step), width, height, gold_cells
            )
        explicit_notes[explicit_note] += 1

        probes = probe_by_step.get(step_id, {})
        probe_cells: dict[tuple[int, int], dict[str, Any]] = {}
        if isinstance(probes, Mapping):
            for task, result in probes.items():
                match = GOLD_CELL_RE.match(str(task))
                if not match or not isinstance(result, Mapping):
                    continue
                coord = int(match.group(1)), int(match.group(2))
                probe_cells[coord] = {
                    "label": result.get("prediction"),
                    "confidence": result.get("confidence"),
                    "target": result.get("target"),
                    "correct": result.get("correct"),
                    "position": result.get("position"),
                    "layer": result.get("layer"),
                }

        current = coord_from_row(step, "current")
        next_pos = coord_from_row(step, "next")
        requested = action_from_row(step, executed=False)
        executed = action_from_row(step, executed=True) or requested
        prompt = first_present(step, ("prompt_text", "prompt"), "")
        raw_response = first_present(step, ("raw_response", "response", "model_output", "output"), "")

        payload_steps.append(
            {
                "index": index,
                "step_id": step_id,
                "current_pos": current,
                "next_pos": next_pos,
                "requested_action": requested,
                "executed_action": executed,
                "prompt": prompt,
                "raw_response": raw_response,
                "true_map": coord_dict_to_json(true_map),
                "gold_belief": coord_dict_to_json(gold_cells),
                "explicit_belief": coord_dict_to_json(explicit_cells),
                "explicit_belief_source": explicit_note,
                "probe_belief": coord_dict_to_json(probe_cells),
                "probes": json_safe(probes),
                "target": json_safe(target),
                "raw_step": json_safe(step),
                **loop_info,
            }
        )

    success = first_present(episode_row, ("success", "reached_goal", "completed"), None)
    return {
        "schema_version": "2.0-cartesian",
        "run": str(run),
        "episode_id": episode,
        "width": width,
        "height": height,
        "start": start,
        "goal": goal,
        "success": success,
        "coordinate_convention": "Cartesian: x increases right, y increases upward, origin at bottom-left",
        "true_map_label_counts": dict(Counter(true_map.values())),
        "explicit_belief_sources": dict(explicit_notes),
        "warnings": warnings,
        "steps": payload_steps,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>__TITLE__</title>
<style>
:root { color-scheme: dark; --bg:#0c111b; --panel:#151d2b; --panel2:#1b2638; --text:#edf3ff; --muted:#9ba8bc; --line:#34435b; --accent:#6ea8fe; }
* { box-sizing:border-box; }
body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif; background:var(--bg); color:var(--text); }
header { position:sticky; top:0; z-index:20; padding:14px 18px; background:rgba(12,17,27,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
h1 { margin:0 0 7px; font-size:20px; }
.meta { display:flex; flex-wrap:wrap; gap:8px; align-items:center; color:var(--muted); font-size:13px; }
.badge { border:1px solid var(--line); border-radius:999px; padding:4px 9px; background:var(--panel); }
.badge.cartesian { border-color:#608fd8; color:#cfe2ff; }
.controls { margin-top:10px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
button,select,input { font:inherit; }
button { cursor:pointer; color:var(--text); background:var(--panel2); border:1px solid var(--line); border-radius:8px; padding:7px 11px; }
button:hover { border-color:var(--accent); }
input[type=range] { width:min(520px,55vw); }
main { padding:16px; max-width:1700px; margin:0 auto; }
.warning { margin-bottom:12px; padding:10px 12px; border:1px solid #8a6c2c; background:#302713; border-radius:9px; color:#ffe4a3; }
.maps { display:grid; grid-template-columns:repeat(2,minmax(390px,1fr)); gap:14px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.panel h2 { font-size:15px; margin:0; padding:11px 13px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:8px; }
.panel .sub { color:var(--muted); font-size:11px; font-weight:400; }
.grid-wrap { overflow:auto; padding:13px; }
.coord-grid { display:grid; gap:4px; width:max-content; align-items:center; justify-items:center; }
.axis { color:var(--muted); font-size:11px; min-width:30px; text-align:center; }
.cell { width:54px; height:54px; position:relative; border:1px solid #53647c; border-radius:8px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:17px; }
.cell.F { background:#183e31; color:#aef3d2; }
.cell.O { background:#4a2630; color:#ffb4c2; }
.cell.U { background:#303848; color:#c6cfdd; }
.cell.missing { background:#232a36; color:#707b8d; border-style:dashed; }
.cell.path { box-shadow:inset 0 0 0 2px #d2b45d; }
.cell.current { outline:3px solid #fff; outline-offset:1px; }
.cell.next { outline:3px solid #65b5ff; outline-offset:1px; }
.marker { position:absolute; font-size:9px; border-radius:4px; padding:1px 3px; line-height:1.2; }
.marker.start { left:2px; top:2px; background:#235b39; }
.marker.goal { right:2px; top:2px; background:#664f17; }
.marker.current { left:2px; bottom:2px; background:#555; }
.marker.next { right:2px; bottom:2px; background:#164c76; }
.marker.path-index { right:2px; top:20px; background:#5e4f21; color:#fff3bd; }
.conf { position:absolute; bottom:2px; left:50%; transform:translateX(-50%); font-size:8px; font-weight:500; color:#dce7f7; }
.legend { padding:0 13px 12px; color:var(--muted); font-size:11px; }
.lower { margin-top:14px; display:grid; grid-template-columns:minmax(330px,.8fr) minmax(500px,1.5fr); gap:14px; }
.step-card { padding:13px; }
.kv { display:grid; grid-template-columns:145px 1fr; gap:6px 10px; font-size:13px; }
.kv .k { color:var(--muted); }
.probe-list { padding:12px; max-height:510px; overflow:auto; }
.probe { border:1px solid var(--line); border-radius:8px; padding:8px; margin-bottom:7px; }
.probe-head { display:flex; justify-content:space-between; gap:8px; font-size:12px; }
.bar { height:6px; background:#293347; border-radius:10px; margin-top:6px; overflow:hidden; }
.bar > i { display:block; height:100%; background:#75aaff; }
.tabs { display:flex; gap:5px; padding:9px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
.tab.active { border-color:var(--accent); background:#243c62; }
pre { margin:0; padding:13px; overflow:auto; max-height:620px; white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; }
.timeline { display:flex; gap:3px; overflow:auto; padding:10px 0 2px; }
.tick { min-width:22px; height:22px; border:1px solid #526078; border-radius:5px; padding:0; font-size:9px; background:#1e573e; }
.tick.repeat { background:#775b18; }
.tick.loop { background:#812e38; }
.tick.active { outline:2px solid #fff; }
@media (max-width:1050px) { .maps,.lower { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>Trajectory Probe Viewer — <span id="episodeTitle"></span></h1>
  <div class="meta">
    <span class="badge cartesian">Cartesian coordinates: x →, y ↑, origin (0,0) bottom-left</span>
    <span class="badge" id="stepBadge"></span>
    <span class="badge" id="actionBadge"></span>
    <span class="badge" id="loopBadge"></span>
  </div>
  <div class="controls">
    <button id="prev">◀ Previous</button><button id="play">▶ Play</button><button id="next">Next ▶</button>
    <button id="jumpLoop">Jump to loop</button>
    <label>Speed <select id="speed"><option value="1400">0.7×</option><option value="850" selected>1×</option><option value="450">2×</option><option value="220">4×</option></select></label>
    <input id="slider" type="range" min="0" max="0" value="0" />
  </div>
</header>
<main>
  <div id="warnings"></div>
  <section class="maps">
    <div class="panel"><h2>True map <span class="sub">objective source: true_cell_x*_y*_FO; only F/O permitted</span></h2><div class="grid-wrap"><div id="trueGrid" class="coord-grid"></div></div><div class="legend">F = free · O = obstacle. This panel never uses probe predictions and never silently fills U.</div></div>
    <div class="panel"><h2>Gold observable belief <span class="sub">gold_cell_x*_y*_OFU</span></h2><div class="grid-wrap"><div id="goldGrid" class="coord-grid"></div></div><div class="legend">U means not yet observed at this step. Every non-U cell is asserted to agree with the true map.</div></div>
    <div class="panel"><h2>Model explicit belief <span class="sub" id="explicitSource"></span></h2><div class="grid-wrap"><div id="explicitGrid" class="coord-grid"></div></div><div class="legend">Uses explicit_cell targets when available; response-matrix orientation is otherwise inferred, not assumed.</div></div>
    <div class="panel"><h2>Probe-decoded gold belief <span class="sub">leave-one-episode-out predictions</span></h2><div class="grid-wrap"><div id="probeGrid" class="coord-grid"></div></div><div class="legend">Each confidence is the probe probability for its predicted O/F/U label. Blank cells have no step-level prediction.</div></div>
  </section>
  <div class="timeline" id="timeline"></div>
  <section class="lower">
    <div>
      <div class="panel"><h2>Step state</h2><div class="step-card"><div class="kv" id="stepKV"></div></div></div>
      <div class="panel" style="margin-top:14px"><h2>Planning / loop probes</h2><div class="probe-list" id="probeList"></div></div>
    </div>
    <div class="panel">
      <div class="tabs"><button class="tab active" data-tab="prompt">Prompt</button><button class="tab" data-tab="output">Model output</button><button class="tab" data-tab="targets">Gold targets</button><button class="tab" data-tab="raw">Raw step</button><button class="tab" data-tab="reliability">Probe reliability</button></div>
      <pre id="detail"></pre>
    </div>
  </section>
</main>
<script>
const DATA = __DATA__;
let index = 0;
let timer = null;
let activeTab = 'prompt';
const $ = id => document.getElementById(id);
const coordKey = c => c ? `${c[0]},${c[1]}` : null;
const pretty = x => typeof x === 'string' ? x : JSON.stringify(x, null, 2);

function pathAt(i) {
  const result = [];
  for (let k=0; k<=i; k++) if (DATA.steps[k].current_pos) result.push(DATA.steps[k].current_pos);
  const next = DATA.steps[i].next_pos;
  if (next && (!result.length || coordKey(result[result.length-1]) !== coordKey(next))) result.push(next);
  return result;
}

function labelObject(raw, mode) {
  if (mode === 'probe') {
    if (!raw || typeof raw !== 'object') return {label:null};
    return raw;
  }
  return {label: raw};
}

function renderGrid(elementId, cells, step, mode) {
  const root = $(elementId);
  root.innerHTML = '';
  root.style.gridTemplateColumns = `34px repeat(${DATA.width}, 54px)`;
  const path = pathAt(index);
  const pathMap = new Map(path.map((coord, i) => [coordKey(coord), i]));
  for (let y=DATA.height-1; y>=0; y--) {
    const axis = document.createElement('div'); axis.className='axis'; axis.textContent=`y=${y}`; root.appendChild(axis);
    for (let x=0; x<DATA.width; x++) {
      const key = `${x},${y}`;
      const obj = labelObject(cells ? cells[key] : null, mode);
      const label = obj.label || null;
      const cell = document.createElement('div');
      cell.className = `cell ${label || 'missing'}`;
      cell.textContent = label || '—';
      const conf = obj.confidence == null ? '' : ` · p=${Number(obj.confidence).toFixed(2)}`;
      cell.title = `(${x}, ${y}) · ${label || 'no value'}${conf}`;
      if (pathMap.has(key)) { cell.classList.add('path'); const m=document.createElement('span'); m.className='marker path-index'; m.textContent=pathMap.get(key); cell.appendChild(m); }
      if (key === coordKey(DATA.start)) { const m=document.createElement('span'); m.className='marker start'; m.textContent='S'; cell.appendChild(m); }
      if (key === coordKey(DATA.goal)) { const m=document.createElement('span'); m.className='marker goal'; m.textContent='G'; cell.appendChild(m); }
      if (key === coordKey(step.current_pos)) { cell.classList.add('current'); const m=document.createElement('span'); m.className='marker current'; m.textContent='NOW'; cell.appendChild(m); }
      if (key === coordKey(step.next_pos)) { cell.classList.add('next'); const m=document.createElement('span'); m.className='marker next'; m.textContent='NEXT'; cell.appendChild(m); }
      if (mode === 'probe' && obj.confidence != null) { const m=document.createElement('span'); m.className='conf'; m.textContent=Number(obj.confidence).toFixed(2); cell.appendChild(m); }
      root.appendChild(cell);
    }
  }
  const corner=document.createElement('div'); corner.className='axis'; corner.textContent=''; root.appendChild(corner);
  for (let x=0; x<DATA.width; x++) { const axis=document.createElement('div'); axis.className='axis'; axis.textContent=`x=${x}`; root.appendChild(axis); }
}

function renderPlanning(step) {
  const root=$('probeList'); root.innerHTML='';
  const entries=Object.entries(step.probes || {}).filter(([task]) => !task.startsWith('gold_cell_'));
  if (!entries.length) { root.textContent='No step-level planning probes loaded. Generate with --fit-step-probes.'; return; }
  entries.sort(([a],[b]) => a.localeCompare(b));
  for (const [task,r] of entries) {
    const p1 = r.probabilities && r.probabilities['1'] != null ? Number(r.probabilities['1']) : Number(r.confidence || 0);
    const div=document.createElement('div'); div.className='probe';
    const correct = r.correct == null ? '' : (r.correct ? '✓' : '✗');
    div.innerHTML=`<div class="probe-head"><b>${task}</b><span>${correct} target=${r.target ?? '—'} pred=${r.prediction ?? '—'}</span></div><div class="probe-head"><span>${r.position ?? ''}/L${r.layer ?? ''}</span><span>confidence=${r.confidence == null ? '—' : Number(r.confidence).toFixed(3)}</span></div><div class="bar"><i style="width:${Math.max(0,Math.min(100,p1*100))}%"></i></div>`;
    root.appendChild(div);
  }
}

function reliability(step) {
  const rows=[];
  for (const [task,r] of Object.entries(step.probes || {})) rows.push({task, position:r.position, layer:r.layer, ...r.reliability});
  return rows;
}

function renderDetail(step) {
  const value = activeTab==='prompt' ? step.prompt : activeTab==='output' ? step.raw_response : activeTab==='targets' ? step.target : activeTab==='raw' ? step.raw_step : reliability(step);
  $('detail').textContent=pretty(value ?? '');
}

function render(i) {
  index=Math.max(0,Math.min(DATA.steps.length-1,i));
  const step=DATA.steps[index];
  $('episodeTitle').textContent=DATA.episode_id;
  $('stepBadge').textContent=`step ${index+1}/${DATA.steps.length} · id=${step.step_id}`;
  $('actionBadge').textContent=`requested=${step.requested_action || '—'} · executed=${step.executed_action || '—'}`;
  $('loopBadge').textContent=step.loop_flag ? `loop signal${step.cycle_length ? ` · cycle=${step.cycle_length}` : ''}` : 'no loop signal';
  $('loopBadge').style.borderColor=step.loop_flag ? '#d45b68' : '';
  $('slider').value=index;
  $('explicitSource').textContent=step.explicit_belief_source || 'unavailable';
  renderGrid('trueGrid',step.true_map,step,'plain');
  renderGrid('goldGrid',step.gold_belief,step,'plain');
  renderGrid('explicitGrid',step.explicit_belief,step,'plain');
  renderGrid('probeGrid',step.probe_belief,step,'probe');
  const items={current_position:step.current_pos,next_position:step.next_pos,requested_action:step.requested_action,executed_action:step.executed_action,repeated_position:step.repeated_position,repeated_position_action:step.repeated_position_action,short_cycle:step.short_cycle,cycle_length:step.cycle_length};
  $('stepKV').innerHTML=Object.entries(items).map(([k,v])=>`<div class="k">${k}</div><div>${pretty(v)}</div>`).join('');
  renderPlanning(step); renderDetail(step);
  document.querySelectorAll('.tick').forEach((el,k)=>el.classList.toggle('active',k===index));
}

function buildTimeline() {
  const root=$('timeline'); root.innerHTML='';
  DATA.steps.forEach((step,i)=>{ const b=document.createElement('button'); b.className=`tick ${step.loop_severity>=2?'loop':step.loop_severity===1?'repeat':''}`; b.textContent=i; b.title=`step ${step.step_id}`; b.onclick=()=>render(i); root.appendChild(b); });
}

function togglePlay() {
  if (timer) { clearInterval(timer); timer=null; $('play').textContent='▶ Play'; return; }
  $('play').textContent='⏸ Pause';
  timer=setInterval(()=>{ if (index>=DATA.steps.length-1) { clearInterval(timer); timer=null; $('play').textContent='▶ Play'; } else render(index+1); }, Number($('speed').value));
}

$('prev').onclick=()=>render(index-1); $('next').onclick=()=>render(index+1); $('play').onclick=togglePlay; $('slider').oninput=e=>render(Number(e.target.value));
$('jumpLoop').onclick=()=>{ const found=DATA.steps.findIndex(s=>s.loop_flag); if(found>=0) render(found); };
$('speed').onchange=()=>{ if(timer){ clearInterval(timer); timer=null; togglePlay(); } };
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{ document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active')); b.classList.add('active'); activeTab=b.dataset.tab; renderDetail(DATA.steps[index]); });
$('slider').max=Math.max(0,DATA.steps.length-1);
$('warnings').innerHTML=(DATA.warnings||[]).map(w=>`<div class="warning">${w}</div>`).join('');
buildTimeline(); render(0);
</script>
</body>
</html>
'''


def write_html(payload: Mapping[str, Any], output_path: Path) -> None:
    title = f"Trajectory viewer — {payload['episode_id']}"
    data_json = json.dumps(json_safe(payload), ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/")
    rendered = HTML_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def find_targets_path(run: Path) -> Path:
    candidates = (run / "targets" / "targets.jsonl", run / "probe_targets_A.jsonl")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("Could not find targets. Checked:\n" + "\n".join(str(p) for p in candidates))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episode", default="auto-loop", help="episode id, unique substring, or auto-loop")
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument("--fit-step-probes", action="store_true")
    parser.add_argument("--probe-groups", default="planning,cells")
    parser.add_argument("--probe-site", default="decision")
    parser.add_argument("--force-probes", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    steps_path = run / "steps.jsonl"
    episodes_path = run / "episodes.jsonl"
    targets_path = find_targets_path(run)

    steps = read_jsonl(steps_path)
    episodes = read_jsonl(episodes_path) if episodes_path.is_file() else []
    targets = read_jsonl(targets_path)

    if args.list_episodes:
        print_episode_list(steps, episodes)
        return

    episode = choose_episode(args.episode, steps, episodes)
    step_rows = filtered_episode(steps, episode)
    target_rows = filtered_episode(targets, episode)
    if not step_rows:
        raise ViewerError(f"No steps found for episode {episode}")
    if not target_rows:
        raise ViewerError(f"No targets found for episode {episode}")

    episode_row = episode_summary_lookup(episodes).get(episode)
    groups = [value.strip() for value in args.probe_groups.split(",") if value.strip()]
    cache: dict[str, Any] | None = None
    cache_path = cache_path_for(run, episode, groups, args.probe_site)
    if args.fit_step_probes:
        cache = fit_step_probes(
            run=run,
            episode=episode,
            target_rows=targets,
            groups=groups,
            probe_site=args.probe_site,
            force=args.force_probes,
        )
    else:
        cache = load_probe_cache(cache_path)
        if cache is not None:
            log(f"[viewer] loaded step-probe cache: {cache_path}")

    payload = build_payload(
        run=run,
        episode=episode,
        step_rows=step_rows,
        target_rows=target_rows,
        episode_row=episode_row,
        probe_cache=cache,
    )
    output = args.output or (run / "trajectory_viewer" / f"{episode}.html")
    write_html(payload, output)

    log(f"[trajectory_probe_viewer] episode={episode}")
    log(f"[trajectory_probe_viewer] steps={len(step_rows)}")
    log(
        "[trajectory_probe_viewer] true_map="
        f"{payload['width']}x{payload['height']} labels={payload['true_map_label_counts']}"
    )
    log("[trajectory_probe_viewer] coordinates=Cartesian(x-right,y-up,origin-bottom-left)")
    if payload["warnings"]:
        log(f"[trajectory_probe_viewer] semantic_warnings={len(payload['warnings'])}")
        for warning in payload["warnings"][:10]:
            log(f"  WARNING: {warning}")
    log(f"[trajectory_probe_viewer] output={output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, exiting.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
