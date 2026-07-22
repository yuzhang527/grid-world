#!/usr/bin/env bash
set -euo pipefail

mkdir -p scripts

cat > scripts/trajectory_probe_viewer.py <<'PYCODE'
#!/usr/bin/env python3
"""Build a self-contained HTML trajectory/probe viewer for grid-world runs.

The generated HTML supports previous/next controls, autoplay, keyboard navigation,
loop highlighting, prompt/output inspection, map/belief comparison, and optional
leave-one-episode-out linear-probe predictions for every step in one episode.

Designed to tolerate both the legacy grid-planner schema and the newer grid-world
schema used by this project.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing pandas. Install with: pip install pandas") from exc


DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
POSITION_ALIASES = {
    "decision": {
        "cells": "prompt_last",
        "memory": "prompt_last",
        "explicit_cells": "prompt_last",
        "true_cells": "prompt_last",
        "true_cells_unobserved": "prompt_last",
        "local": "mean_last_feedback",
        "planning": "pre_action_token",
        "faithfulness": "pre_action_token",
    }
}


@dataclass(frozen=True)
class Site:
    position: str
    layer: int
    reference_macro_f1: float | None = None
    reference_baseline: float | None = None


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def first_value(mapping: Mapping[str, Any] | None, keys: Sequence[str], default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if not math.isfinite(v) else v
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_file(run: Path, names: Sequence[str], recursive: bool = False) -> Path | None:
    for name in names:
        candidate = run / name
        if candidate.exists():
            return candidate
    if recursive:
        for name in names:
            matches = sorted(run.rglob(Path(name).name), key=lambda p: (len(p.parts), str(p)))
            if matches:
                return matches[0]
    return None


def parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def parse_position(value: Any) -> list[int] | None:
    value = parse_maybe_json(value)
    if isinstance(value, Mapping):
        x = first_value(value, ("x", "col", "column"))
        y = first_value(value, ("y", "row"))
        if x is not None and y is not None:
            return [int(x), int(y)]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+", value)
        if len(numbers) >= 2:
            return [int(numbers[0]), int(numbers[1])]
    return None


def normalize_cell(value: Any) -> str:
    if value is None:
        return "U"
    text = str(value).strip().upper()
    aliases = {
        "0": "F", "FREE": "F", ".": "F", "OPEN": "F", "EMPTY": "F",
        "1": "O", "OBSTACLE": "O", "BLOCKED": "O", "#": "O", "WALL": "WALL",
        "UNKNOWN": "U", "?": "U", "UNOBSERVED": "U",
        "START": "S", "GOAL": "G", "AGENT": "A",
    }
    return aliases.get(text, text if text in {"F", "O", "U", "WALL", "S", "G", "A"} else text[:8])


def parse_grid(value: Any) -> list[list[str]] | None:
    value = parse_maybe_json(value)
    if isinstance(value, Mapping):
        value = first_value(value, ("grid", "cells", "map", "belief_grid", "true_map"), value)
    if isinstance(value, list) and value:
        if all(isinstance(row, (list, tuple)) for row in value):
            widths = [len(row) for row in value]
            if widths and min(widths) > 0:
                width = min(widths)
                return [[normalize_cell(cell) for cell in row[:width]] for row in value]
        if all(isinstance(row, str) for row in value):
            parsed: list[list[str]] = []
            for row in value:
                tokens = row.strip().replace("|", " ").replace(",", " ").split()
                if len(tokens) == 1 and len(tokens[0]) > 1:
                    tokens = list(tokens[0])
                parsed.append([normalize_cell(token) for token in tokens])
            if parsed and min(map(len, parsed)) > 0:
                width = min(map(len, parsed))
                return [row[:width] for row in parsed]
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        parsed = []
        for line in lines:
            line = re.sub(r"^[\[\-\s]+|[\]\s]+$", "", line)
            tokens = [token for token in re.split(r"[\s,|]+", line) if token]
            if len(tokens) == 1 and len(tokens[0]) > 1:
                tokens = list(tokens[0])
            if tokens:
                parsed.append([normalize_cell(token.strip("'\"")) for token in tokens])
        if parsed and min(map(len, parsed)) > 0:
            width = min(map(len, parsed))
            return [row[:width] for row in parsed]
    return None


def episode_id_of(row: Mapping[str, Any]) -> str:
    value = first_value(row, ("episode_id", "episode", "id", "trajectory_id", "seed"), "unknown")
    return str(value)


def step_id_of(row: Mapping[str, Any], fallback: int = 0) -> int:
    value = first_value(row, ("step_id", "step", "t", "turn", "index"), fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def extract_response_object(row: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("parsed_response", "parsed_output", "model_output", "parsed", "response_json"):
        value = parse_maybe_json(row.get(key))
        if isinstance(value, Mapping):
            return dict(value)
    for key in ("response", "raw_response", "response_text", "output"):
        value = parse_maybe_json(row.get(key))
        if isinstance(value, Mapping):
            return dict(value)
    return None


def extract_raw_response(row: Mapping[str, Any]) -> str:
    value = first_value(row, ("raw_response", "response_text", "response", "output", "completion"), "")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def extract_prompt(row: Mapping[str, Any]) -> str:
    value = first_value(row, ("prompt_text", "prompt", "input", "messages"), "")
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, Mapping):
                chunks.append(f"[{item.get('role', 'message')}]\n{item.get('content', '')}")
            else:
                chunks.append(str(item))
        return "\n\n".join(chunks)
    return str(value)


def extract_model_belief(row: Mapping[str, Any]) -> list[list[str]] | None:
    direct = first_value(row, ("parsed_belief_grid", "model_belief_grid", "belief_grid"))
    parsed = parse_grid(direct)
    if parsed is not None:
        return parsed
    response = extract_response_object(row)
    if response:
        return parse_grid(first_value(response, ("belief_grid", "grid", "belief")))
    return None


def extract_requested_action(row: Mapping[str, Any]) -> str | None:
    value = first_value(row, ("requested_action", "model_action", "predicted_action", "parsed_action"))
    if value is None:
        response = extract_response_object(row)
        if response:
            value = first_value(response, ("action", "move", "next_action"))
    if value is None:
        value = first_value(row, ("action", "executed_action"))
    return str(value).strip().upper() if value is not None else None


def extract_executed_action(row: Mapping[str, Any]) -> str | None:
    value = first_value(row, ("executed_action", "action", "applied_action", "final_action"))
    return str(value).strip().upper() if value is not None else None


def extract_true_map(summary: Mapping[str, Any] | None, step: Mapping[str, Any]) -> list[list[str]] | None:
    for source in (summary, step):
        if not isinstance(source, Mapping):
            continue
        for key in ("true_map", "grid", "map", "world", "obstacle_map"):
            parsed = parse_grid(source.get(key))
            if parsed is not None:
                return parsed
        spec = source.get("grid_spec")
        parsed = parse_grid(spec)
        if parsed is not None:
            return parsed
    return None


def extract_success(summary: Mapping[str, Any] | None, steps: Sequence[Mapping[str, Any]]) -> bool | None:
    if isinstance(summary, Mapping):
        value = first_value(summary, ("success", "episode_success", "reached_goal", "solved"))
        if value is not None:
            return bool(value)
        status = first_value(summary, ("status", "termination_reason"))
        if isinstance(status, str):
            if status.lower() in {"success", "goal", "reached_goal", "solved"}:
                return True
            if status.lower() in {"failed", "failure", "timeout", "max_steps", "loop"}:
                return False
    if steps:
        value = first_value(steps[-1], ("success", "episode_success", "reached_goal"))
        if value is not None:
            return bool(value)
    return None


def extract_summary_rows(run: Path) -> dict[str, dict[str, Any]]:
    by_episode: dict[str, dict[str, Any]] = {}
    episodes_path = discover_file(run, ("episodes.jsonl", "episode_summaries.jsonl"))
    if episodes_path:
        for row in load_jsonl(episodes_path):
            by_episode[episode_id_of(row)] = row
    summary_path = discover_file(run, ("summary.json",))
    if summary_path:
        summary = load_json(summary_path)
        candidates: list[Any] = []
        if isinstance(summary, list):
            candidates = summary
        elif isinstance(summary, Mapping):
            for key in ("episodes", "results", "episode_summaries", "items"):
                if isinstance(summary.get(key), list):
                    candidates.extend(summary[key])
            if not candidates and any(k in summary for k in ("episode_id", "true_map", "trajectory")):
                candidates = [summary]
        for row in candidates:
            if isinstance(row, Mapping):
                by_episode.setdefault(episode_id_of(row), dict(row))
    return by_episode


def load_targets(run: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], Path | None]:
    path = discover_file(
        run,
        (
            "probe_targets.jsonl",
            "probe_targets_A.jsonl",
            "targets.jsonl",
            "targets/probe_targets.jsonl",
            "targets/targets.jsonl",
        ),
        recursive=True,
    )
    if path is None:
        return {}, None
    rows = load_jsonl(path)
    mapping = {(episode_id_of(row), step_id_of(row, i)): row for i, row in enumerate(rows)}
    return mapping, path


def group_steps(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        copied = dict(row)
        copied["__source_index"] = index
        grouped[episode_id_of(row)].append(copied)
    for episode, items in grouped.items():
        items.sort(key=lambda row: (step_id_of(row), row["__source_index"]))
    return dict(grouped)


def path_from_steps(steps: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    path: list[list[int]] = []
    for row in steps:
        pos = parse_position(first_value(row, ("current_pos", "position", "pos", "state_position")))
        if pos is not None:
            if not path or path[-1] != pos:
                path.append(pos)
    if steps:
        final_pos = parse_position(first_value(steps[-1], ("next_pos", "next_position", "new_pos")))
        if final_pos is not None and (not path or path[-1] != final_pos):
            path.append(final_pos)
    return path


def loop_annotations(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen_pos: dict[tuple[int, int], list[int]] = defaultdict(list)
    seen_pos_action: dict[tuple[tuple[int, int], str], list[int]] = defaultdict(list)
    annotations: list[dict[str, Any]] = []
    for i, row in enumerate(steps):
        pos_list = parse_position(first_value(row, ("current_pos", "position", "pos")))
        pos = tuple(pos_list) if pos_list is not None else None
        action = extract_requested_action(row) or extract_executed_action(row) or "?"
        prior_positions = seen_pos.get(pos, []) if pos is not None else []
        prior_pairs = seen_pos_action.get((pos, action), []) if pos is not None else []
        cycle_start = prior_positions[-1] if prior_positions else None
        cycle_length = i - cycle_start if cycle_start is not None else None
        short_cycle = bool(cycle_length is not None and 1 <= cycle_length <= 8)
        annotations.append(
            {
                "position_seen_before": bool(prior_positions),
                "position_action_seen_before": bool(prior_pairs),
                "visit_count_before": len(prior_positions),
                "cycle_start_step_index": cycle_start,
                "cycle_length": cycle_length,
                "short_cycle": short_cycle,
            }
        )
        if pos is not None:
            seen_pos[pos].append(i)
            seen_pos_action[(pos, action)].append(i)
    return annotations


def loop_score(steps: Sequence[Mapping[str, Any]], targets: Mapping[tuple[str, int], Mapping[str, Any]]) -> float:
    annotations = loop_annotations(steps)
    repeated = sum(int(a["position_seen_before"]) for a in annotations)
    repeated_pair = sum(int(a["position_action_seen_before"]) for a in annotations)
    short = sum(int(a["short_cycle"]) for a in annotations)
    target_loop = 0
    for i, step in enumerate(steps):
        target = targets.get((episode_id_of(step), step_id_of(step, i)), {})
        try:
            target_loop += int(target.get("loop_risk", 0) == 1)
        except Exception:
            pass
    return repeated * 3 + repeated_pair * 5 + short * 7 + target_loop * 2 + len(steps) * 0.01


def select_episode(
    requested: str,
    grouped: Mapping[str, list[dict[str, Any]]],
    summaries: Mapping[str, Mapping[str, Any]],
    targets: Mapping[tuple[str, int], Mapping[str, Any]],
) -> str:
    if requested in grouped:
        return requested
    lower = requested.lower()
    if lower not in {"auto", "auto-loop", "auto-failure", "longest"}:
        matches = [episode for episode in grouped if requested in episode]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"Episode {requested!r} not found. Available examples: {list(grouped)[:8]}")
    candidates = list(grouped)
    if lower == "auto-failure":
        failed = [ep for ep in candidates if extract_success(summaries.get(ep), grouped[ep]) is False]
        candidates = failed or candidates
        return max(candidates, key=lambda ep: (len(grouped[ep]), loop_score(grouped[ep], targets)))
    if lower == "longest":
        return max(candidates, key=lambda ep: len(grouped[ep]))
    return max(candidates, key=lambda ep: loop_score(grouped[ep], targets))


def grid_from_target(target: Mapping[str, Any], prefixes: Sequence[str]) -> list[list[str]] | None:
    found: dict[tuple[int, int], str] = {}
    for key, value in target.items():
        for prefix in prefixes:
            match = re.fullmatch(re.escape(prefix) + r"x(\d+)_y(\d+)(?:_OFU|_OFUW)?", key)
            if match:
                x, y = map(int, match.groups())
                found[(x, y)] = normalize_cell(value)
                break
    if not found:
        return None
    width = max(x for x, _ in found) + 1
    height = max(y for _, y in found) + 1
    grid = [["U" for _ in range(width)] for _ in range(height)]
    for (x, y), value in found.items():
        grid[y][x] = value
    return grid


def infer_task_group(task: str) -> str:
    if task.startswith("gold_local_"):
        return "local"
    if task.startswith("gold_cell_"):
        return "cells"
    if task.startswith("explicit_cell_") or task.startswith("model_cell_"):
        return "explicit_cells"
    if task.startswith("true_cell_") and "unobserved" in task:
        return "true_cells_unobserved"
    if task.startswith("true_cell_"):
        return "true_cells"
    if task.endswith("_known") or "known_mask" in task:
        return "memory"
    if task in {
        "chosen_action_is_astar_best",
        "requested_action_is_astar_best",
        "chosen_action_reduces_true_distance",
        "requested_action_reduces_true_distance",
        "loop_risk",
        "position_seen_before",
        "position_action_seen_before",
    } or task.startswith("true_action_") or task.startswith("requested_action_"):
        return "planning"
    if "belief" in task or "faithful" in task or "missed" in task:
        return "faithfulness"
    return "other"


def find_activation_dir(run: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if (path / "X.npy").exists() else None
    candidates = [run / "activations", run / "activations_A_multi"]
    candidates.extend(path.parent for path in run.rglob("X.npy") if "activation" in str(path.parent).lower())
    unique: list[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen and (resolved / "X.npy").exists():
            seen.add(resolved)
            unique.append(resolved)
    return unique[0] if unique else None


def load_positions(path: Path) -> list[str]:
    value = load_json(path)
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, Mapping):
        for key in ("positions", "names"):
            if isinstance(value.get(key), list):
                return [str(v) for v in value[key]]
        sortable = []
        for key, val in value.items():
            try:
                sortable.append((int(val), str(key)))
            except (TypeError, ValueError):
                continue
        if sortable:
            return [name for _, name in sorted(sortable)]
    raise ValueError(f"Unsupported positions.json format: {path}")


def meta_key(row: Mapping[str, Any], fallback: int) -> tuple[str, int]:
    return episode_id_of(row), step_id_of(row, fallback)


def find_probe_table(run: Path, explicit: str | None) -> pd.DataFrame | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return pd.read_csv(path)
    preferred = ("best_by_task.csv", "task_layer_curves.csv", "probe_results.csv")
    for name in preferred:
        matches = sorted(run.rglob(name), key=lambda p: (0 if "all_layer" in str(p).lower() else 1, len(p.parts), str(p)))
        for path in matches:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            required = {"task", "position", "layer"}
            if required.issubset(df.columns):
                return df
    return None


def best_sites(
    tasks: Sequence[str],
    groups: Mapping[str, str],
    positions: Sequence[str],
    layers: Sequence[int],
    table: pd.DataFrame | None,
    site_policy: str,
) -> dict[str, Site]:
    position_set = set(positions)
    layer_set = {int(v) for v in layers}
    sites: dict[str, Site] = {}
    score_col = None
    baseline_col = None
    if table is not None:
        for candidate in ("macro_f1_mean", "mean_macro_f1", "peak_mean_macro_f1"):
            if candidate in table.columns:
                score_col = candidate
                break
        for candidate in ("majority_macro_f1_mean", "mean_majority_macro_f1"):
            if candidate in table.columns:
                baseline_col = candidate
                break
    for task in tasks:
        group = groups[task]
        desired_position: str | None = None
        if site_policy == "decision":
            desired_position = POSITION_ALIASES["decision"].get(group)
        elif site_policy != "best":
            desired_position = site_policy
        subset = table[table["task"].astype(str) == task].copy() if table is not None else pd.DataFrame()
        if desired_position and desired_position in position_set:
            if not subset.empty:
                pos_subset = subset[subset["position"].astype(str) == desired_position]
                if not pos_subset.empty:
                    subset = pos_subset
            position = desired_position
        elif not subset.empty:
            position = str(subset.iloc[0]["position"])
        else:
            fallback_positions = {
                "planning": ("pre_action_token", "prompt_last"),
                "local": ("mean_last_feedback", "prompt_last"),
            }.get(group, ("prompt_last", "mean_current_belief_grid", "pre_action_token"))
            position = next((p for p in fallback_positions if p in position_set), positions[0])
        if position not in position_set:
            position = positions[0]
        if not subset.empty:
            subset = subset[subset["position"].astype(str) == position]
        if not subset.empty and score_col:
            subset = subset.sort_values(score_col, ascending=False)
        row = subset.iloc[0] if not subset.empty else None
        layer = int(row["layer"]) if row is not None else int(layers[len(layers) // 2])
        if layer not in layer_set:
            layer = min(layer_set, key=lambda value: abs(value - layer))
        score = float(row[score_col]) if row is not None and score_col and pd.notna(row[score_col]) else None
        baseline = float(row[baseline_col]) if row is not None and baseline_col and pd.notna(row[baseline_col]) else None
        sites[task] = Site(position=position, layer=layer, reference_macro_f1=score, reference_baseline=baseline)
    return sites


def relevant_tasks(target_rows: Iterable[Mapping[str, Any]], requested_groups: set[str]) -> list[str]:
    candidates = Counter()
    ignored = {"episode_id", "step_id", "step", "current_pos", "next_pos", "action", "requested_action"}
    for row in target_rows:
        for key, value in row.items():
            if key in ignored or value is None or isinstance(value, (dict, list)):
                continue
            group = infer_task_group(key)
            if group in requested_groups:
                candidates[key] += 1
    priority = {"planning": 0, "local": 1, "cells": 2, "memory": 3, "explicit_cells": 4, "faithfulness": 5, "true_cells": 6, "true_cells_unobserved": 7}
    return sorted(candidates, key=lambda task: (priority.get(infer_task_group(task), 99), task))


def fit_step_probes(
    *,
    run: Path,
    episode_id: str,
    target_map: Mapping[tuple[str, int], Mapping[str, Any]],
    groups_requested: set[str],
    activation_dir_arg: str | None,
    probe_table_arg: str | None,
    site_policy: str,
    max_tasks: int | None,
    force: bool,
    random_state: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    cache_dir = run / "trajectory_viewer_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    group_slug = "-".join(sorted(groups_requested))
    cache_path = cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', episode_id)}__{group_slug}__{site_policy}.json"
    if cache_path.exists() and not force:
        eprint(f"[viewer] loading cached step probes: {cache_path}")
        raw = load_json(cache_path)
        return {
            (str(row["episode_id"]), int(row["step_id"])): row.get("probes", {})
            for row in raw.get("rows", [])
        }

    activation_dir = find_activation_dir(run, activation_dir_arg)
    if activation_dir is None:
        raise FileNotFoundError("Could not find an activation directory containing X.npy")
    required = {
        "X": activation_dir / "X.npy",
        "positions": activation_dir / "positions.json",
        "layers": activation_dir / "layers.npy",
        "meta": activation_dir / "meta.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing activation files: " + ", ".join(missing))

    try:
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import f1_score
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing scikit-learn. Install with: pip install scikit-learn") from exc

    X = np.load(required["X"], mmap_mode="r")
    positions = load_positions(required["positions"])
    layers = [int(v) for v in np.load(required["layers"]).tolist()]
    meta_rows = load_jsonl(required["meta"])
    if X.shape[0] != len(meta_rows):
        raise ValueError(f"Activation/meta mismatch: X has {X.shape[0]} rows, meta has {len(meta_rows)}")
    mask_path = activation_dir / "position_mask.npy"
    position_mask = np.load(mask_path, mmap_mode="r") if mask_path.exists() else None
    keys = [meta_key(row, i) for i, row in enumerate(meta_rows)]
    meta_episode = np.asarray([key[0] for key in keys], dtype=object)

    target_rows = [target_map[key] for key in keys if key in target_map]
    tasks = relevant_tasks(target_rows, groups_requested)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError(f"No target tasks found for groups {sorted(groups_requested)}")
    task_groups = {task: infer_task_group(task) for task in tasks}
    probe_table = find_probe_table(run, probe_table_arg)
    sites = best_sites(tasks, task_groups, positions, layers, probe_table, site_policy)
    pos_to_idx = {name: idx for idx, name in enumerate(positions)}
    layer_to_idx = {value: idx for idx, value in enumerate(layers)}

    eprint(f"[viewer] activation_dir={activation_dir}")
    eprint(f"[viewer] fitting {len(tasks)} leave-one-episode-out probes for episode={episode_id}")
    predictions: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    test_episode_mask = meta_episode == episode_id
    if not np.any(test_episode_mask):
        raise KeyError(f"Episode {episode_id!r} has no activation rows")

    for task_no, task in enumerate(tasks, 1):
        site = sites[task]
        p_idx = pos_to_idx[site.position]
        l_idx = layer_to_idx[site.layer]
        labels = np.asarray([target_map.get(key, {}).get(task) for key in keys], dtype=object)
        valid = np.asarray([value is not None and str(value) not in {"", "nan", "None"} for value in labels])
        if position_mask is not None:
            if position_mask.ndim == 2:
                valid &= np.asarray(position_mask[:, p_idx], dtype=bool)
            elif position_mask.ndim == 3:
                valid &= np.asarray(position_mask[:, p_idx, l_idx], dtype=bool)
        train_idx = np.flatnonzero(valid & ~test_episode_mask)
        test_idx = np.flatnonzero(valid & test_episode_mask)
        if len(test_idx) == 0 or len(train_idx) < 20:
            continue
        train_labels = labels[train_idx].astype(str)
        classes = sorted(set(train_labels.tolist()))
        if len(classes) < 2:
            continue
        le = LabelEncoder().fit(classes)
        y_train = le.transform(train_labels)
        X_train = np.asarray(X[train_idx, p_idx, l_idx, :], dtype=np.float32)
        X_test = np.asarray(X[test_idx, p_idx, l_idx, :], dtype=np.float32)
        mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = X_train.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        class_counts = np.bincount(y_train, minlength=len(classes))
        use_early_stopping = bool(len(y_train) >= 50 and int(class_counts.min()) >= 5)
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-4,
            class_weight="balanced",
            early_stopping=use_early_stopping,
            validation_fraction=0.1,
            n_iter_no_change=12,
            random_state=random_state,
            average=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)
        pred_ids = clf.predict(X_test)
        probs = clf.predict_proba(X_test)
        pred_labels = le.inverse_transform(pred_ids)
        true_labels = labels[test_idx].astype(str)
        episode_f1 = None
        try:
            episode_f1 = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
        except Exception:
            pass
        class_names = [str(v) for v in le.classes_.tolist()]
        for local_i, activation_i in enumerate(test_idx):
            prob_map = {class_names[j]: float(probs[local_i, j]) for j in range(len(class_names))}
            pred = str(pred_labels[local_i])
            true = str(true_labels[local_i])
            predictions[keys[activation_i]][task] = {
                "task_group": task_groups[task],
                "position": site.position,
                "layer": site.layer,
                "target": true,
                "prediction": pred,
                "confidence": float(max(prob_map.values())),
                "probabilities": prob_map,
                "correct": pred == true,
                "episode_macro_f1": episode_f1,
                "reference_macro_f1": site.reference_macro_f1,
                "reference_baseline": site.reference_baseline,
                "train_rows": int(len(train_idx)),
            }
        eprint(f"[viewer] probe {task_no:>3}/{len(tasks)} {task} @ {site.position}/L{site.layer}")

    serialized_rows = [
        {"episode_id": key[0], "step_id": key[1], "probes": json_safe(value)}
        for key, value in sorted(predictions.items())
    ]
    cache_path.write_text(
        json.dumps({"episode_id": episode_id, "groups": sorted(groups_requested), "rows": serialized_rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    eprint(f"[viewer] saved step-probe cache: {cache_path}")
    return dict(predictions)


def summary_probe_rows(run: Path) -> list[dict[str, Any]]:
    table = find_probe_table(run, None)
    if table is None:
        return []
    score_col = next((c for c in ("macro_f1_mean", "mean_macro_f1", "peak_mean_macro_f1") if c in table.columns), None)
    baseline_col = next((c for c in ("majority_macro_f1_mean", "mean_majority_macro_f1") if c in table.columns), None)
    if score_col is None:
        return []
    rows = []
    for task, group in table.groupby("task", sort=False):
        row = group.sort_values(score_col, ascending=False).iloc[0]
        rows.append(
            {
                "task": str(task),
                "task_group": str(row.get("task_group", infer_task_group(str(task)))),
                "position": str(row["position"]),
                "layer": int(row["layer"]),
                "macro_f1": float(row[score_col]),
                "baseline": float(row[baseline_col]) if baseline_col and pd.notna(row[baseline_col]) else None,
            }
        )
    return rows


def extract_probe_grid(step_probes: Mapping[str, Any], prefixes: Sequence[str]) -> tuple[list[list[str]], list[list[float]]] | tuple[None, None]:
    found: dict[tuple[int, int], tuple[str, float]] = {}
    for task, result in step_probes.items():
        for prefix in prefixes:
            match = re.fullmatch(re.escape(prefix) + r"x(\d+)_y(\d+)(?:_OFU|_OFUW)?", task)
            if match:
                x, y = map(int, match.groups())
                found[(x, y)] = (normalize_cell(result.get("prediction")), float(result.get("confidence", 0.0)))
                break
    if not found:
        return None, None
    width = max(x for x, _ in found) + 1
    height = max(y for _, y in found) + 1
    grid = [["U" for _ in range(width)] for _ in range(height)]
    confidence = [[0.0 for _ in range(width)] for _ in range(height)]
    for (x, y), (value, conf) in found.items():
        grid[y][x] = value
        confidence[y][x] = conf
    return grid, confidence


def make_payload(
    *,
    run: Path,
    episode_id: str,
    steps: Sequence[dict[str, Any]],
    summary: Mapping[str, Any] | None,
    target_map: Mapping[tuple[str, int], Mapping[str, Any]],
    probe_predictions: Mapping[tuple[str, int], Mapping[str, Any]],
    aggregate_probes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    annotations = loop_annotations(steps)
    true_map = extract_true_map(summary, steps[0]) if steps else None
    start = parse_position(first_value(summary, ("start", "start_pos", "source"))) if summary else None
    goal = parse_position(first_value(summary, ("goal", "goal_pos", "target"))) if summary else None
    if start is None and steps:
        start = parse_position(first_value(steps[0], ("current_pos", "position", "pos")))
    path = path_from_steps(steps)
    if goal is None and true_map:
        for y, row in enumerate(true_map):
            for x, cell in enumerate(row):
                if cell == "G":
                    goal = [x, y]
    payload_steps = []
    for i, row in enumerate(steps):
        key = (episode_id, step_id_of(row, i))
        target = dict(target_map.get(key, {}))
        current = parse_position(first_value(row, ("current_pos", "position", "pos", "state_position")))
        next_pos = parse_position(first_value(row, ("next_pos", "next_position", "new_pos")))
        model_grid = extract_model_belief(row)
        gold_grid = grid_from_target(target, ("gold_cell_", "belief_cell_"))
        probes = dict(probe_predictions.get(key, {}))
        decoded_grid, decoded_conf = extract_probe_grid(probes, ("gold_cell_",))
        loop_target = target.get("loop_risk")
        payload_steps.append(
            {
                "step_id": key[1],
                "index": i,
                "prompt": extract_prompt(row),
                "raw_response": extract_raw_response(row),
                "response_object": extract_response_object(row),
                "current_pos": current,
                "next_pos": next_pos,
                "requested_action": extract_requested_action(row),
                "executed_action": extract_executed_action(row),
                "invalid_move": bool(first_value(row, ("invalid_move", "invalid_action", "fallback_used"), False)),
                "parse_error": bool(first_value(row, ("parse_error", "response_parse_error"), False)),
                "repaired": bool(first_value(row, ("repaired", "repair_used"), False)),
                "model_belief": model_grid,
                "gold_belief": gold_grid,
                "decoded_belief": decoded_grid,
                "decoded_confidence": decoded_conf,
                "target": target,
                "probes": probes,
                "loop": {**annotations[i], "target_loop_risk": loop_target},
                "raw_step": row,
            }
        )
    return {
        "title": f"Trajectory Probe Viewer — {episode_id}",
        "run": str(run),
        "episode_id": episode_id,
        "success": extract_success(summary, steps),
        "start": start,
        "goal": goal,
        "true_map": true_map,
        "path": path,
        "steps": payload_steps,
        "aggregate_probes": list(aggregate_probes),
        "summary": summary,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
:root { --bg:#0b1020; --panel:#121a2d; --panel2:#18233b; --text:#ecf2ff; --muted:#9fb0cc; --line:#2b3a5d; --good:#42d392; --warn:#f4b740; --bad:#ff6b6b; --accent:#6ea8fe; --unknown:#6c7890; }
*{box-sizing:border-box} body{margin:0;background:linear-gradient(135deg,#090d18,#111a2d);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
button,select,input{font:inherit}.app{max-width:1680px;margin:0 auto;padding:18px}.top{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap}.title h1{margin:0;font-size:25px}.subtitle{color:var(--muted);font-size:13px;margin-top:5px;max-width:1000px;overflow-wrap:anywhere}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.btn{border:1px solid var(--line);background:var(--panel2);color:var(--text);border-radius:9px;padding:8px 12px;cursor:pointer}.btn:hover{border-color:var(--accent)}.btn.primary{background:#245bba}.btn.loop{background:#7a3340}.control-label{color:var(--muted);font-size:12px}.slider{width:min(520px,75vw)}
.metrics{display:grid;grid-template-columns:repeat(7,minmax(110px,1fr));gap:9px;margin:15px 0}.metric{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px}.metric .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em}.metric .v{font-size:17px;font-weight:700;margin-top:3px}.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.timeline-wrap{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px;margin-bottom:12px;overflow-x:auto}.timeline{display:flex;align-items:center;min-width:max-content;gap:5px}.dot{width:18px;height:18px;border-radius:50%;background:#52617d;border:2px solid transparent;cursor:pointer;position:relative}.dot.current{border-color:white;transform:scale(1.22)}.dot.loop{background:var(--bad)}.dot.repeat{background:var(--warn)}.dot.good{background:var(--good)}.dot::after{content:attr(data-step);position:absolute;top:22px;left:50%;transform:translateX(-50%);font-size:9px;color:var(--muted);display:none}.dot:hover::after,.dot.current::after{display:block}.timeline{padding-bottom:14px}
.main-grid{display:grid;grid-template-columns:minmax(620px,1.15fr) minmax(520px,.85fr);gap:12px}.panel{background:rgba(18,26,45,.96);border:1px solid var(--line);border-radius:13px;padding:13px;min-width:0}.panel h2{font-size:15px;margin:0 0 10px}.panel h3{font-size:13px;color:var(--muted);margin:12px 0 7px}.maps{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px}.map-card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:9px;min-width:0}.map-title{display:flex;justify-content:space-between;gap:6px;align-items:center;font-size:12px;font-weight:700;margin-bottom:7px}.map-note{color:var(--muted);font-size:10px;font-weight:400}.grid-board{display:grid;gap:3px;aspect-ratio:1/1}.cell{position:relative;border:1px solid rgba(255,255,255,.12);border-radius:5px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;min-width:0}.cell .coord{position:absolute;left:3px;top:2px;font-size:7px;opacity:.55;font-weight:400}.cell .conf{position:absolute;right:3px;bottom:2px;font-size:7px;opacity:.75;font-weight:500}.cell.F{background:#d8e1ef;color:#182033}.cell.O,.cell.WALL{background:#202637;color:#fff}.cell.U{background:#68748b;color:#eef3ff}.cell.S{background:#3b8d69}.cell.G{background:#a9821f}.cell.path{box-shadow:inset 0 0 0 2px #6ea8fe}.cell.past{box-shadow:inset 0 0 0 2px #7788a8}.cell.current{outline:3px solid #fff;animation:pulse 1.2s infinite}.cell.goal{border:3px solid #f4b740}.cell.start{border:3px solid #42d392}@keyframes pulse{50%{outline-color:#6ea8fe}}
.probe-cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.probe-card{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:9px}.probe-head{display:flex;justify-content:space-between;gap:8px;font-size:11px}.bar{height:8px;background:#273550;border-radius:999px;overflow:hidden;margin:7px 0}.bar>span{display:block;height:100%;background:var(--accent)}.small{font-size:10px;color:var(--muted)}
.tabs{display:flex;gap:6px;margin-bottom:8px}.tab{padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:var(--panel2);color:var(--muted);cursor:pointer}.tab.active{color:white;border-color:var(--accent)}.tab-content{display:none}.tab-content.active{display:block}.code{white-space:pre-wrap;overflow:auto;max-height:530px;background:#090e1b;border:1px solid var(--line);border-radius:9px;padding:11px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;line-height:1.45;overflow-wrap:anywhere}.probe-table{width:100%;border-collapse:collapse;font-size:10px}.probe-table th,.probe-table td{padding:6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.probe-table th{color:var(--muted);position:sticky;top:0;background:var(--panel)}.table-wrap{max-height:430px;overflow:auto}.badge{display:inline-block;border-radius:999px;padding:2px 6px;font-size:9px;background:#273550}.badge.ok{background:#1f6048}.badge.no{background:#74343c}.legend{display:flex;gap:10px;flex-wrap:wrap;color:var(--muted);font-size:10px;margin-top:8px}.legend span::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;background:var(--legend)}.empty{color:var(--muted);font-size:12px;padding:12px;text-align:center}.footer{color:var(--muted);font-size:10px;margin-top:12px;text-align:center}
@media(max-width:1180px){.main-grid{grid-template-columns:1fr}.maps{grid-template-columns:repeat(2,1fr)}.metrics{grid-template-columns:repeat(4,1fr)}}@media(max-width:700px){.maps{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.probe-cards{grid-template-columns:1fr}}
</style>
</head>
<body><div class="app">
<div class="top"><div class="title"><h1 id="title"></h1><div class="subtitle" id="subtitle"></div></div>
<div class="controls"><button class="btn" id="prev">◀ Previous</button><button class="btn primary" id="play">▶ Play</button><button class="btn" id="next">Next ▶</button><button class="btn loop" id="jumpLoop">Jump to loop</button><span class="control-label">Speed</span><select id="speed" class="btn"><option value="2000">0.5×</option><option value="1000" selected>1×</option><option value="500">2×</option><option value="250">4×</option></select></div></div>
<div class="controls" style="margin-top:10px"><input id="slider" class="slider" type="range" min="0" max="0" value="0"><span id="stepLabel" class="control-label"></span></div>
<div id="metrics" class="metrics"></div><div class="timeline-wrap"><div id="timeline" class="timeline"></div></div>
<div class="main-grid">
<div>
<div class="panel"><h2>World and belief state</h2><div id="maps" class="maps"></div><div class="legend"><span style="--legend:#d8e1ef">Free</span><span style="--legend:#202637">Obstacle / wall</span><span style="--legend:#68748b">Unknown</span><span style="--legend:#6ea8fe">Trajectory</span><span style="--legend:#fff">Current position</span></div></div>
<div class="panel" style="margin-top:12px"><h2>Step-level probe decoding</h2><div id="planningProbes" class="probe-cards"></div><h3>All decoded tasks at this step</h3><div id="probeTable" class="table-wrap"></div></div>
</div>
<div class="panel"><div class="tabs"><button class="tab active" data-tab="prompt">Prompt</button><button class="tab" data-tab="output">Model output</button><button class="tab" data-tab="targets">Gold targets</button><button class="tab" data-tab="raw">Raw step</button><button class="tab" data-tab="aggregate">Probe reliability</button></div><div id="prompt" class="tab-content active"><div class="code"></div></div><div id="output" class="tab-content"><div class="code"></div></div><div id="targets" class="tab-content"><div class="code"></div></div><div id="raw" class="tab-content"><div class="code"></div></div><div id="aggregate" class="tab-content"><div class="table-wrap"></div></div></div>
</div><div class="footer">Arrow keys move one step. Space toggles autoplay. Red timeline markers indicate short-cycle/repeated-action evidence. Per-step probe predictions are leave-one-episode-out when enabled.</div>
</div>
<script>const DATA=__DATA__;let index=0;let timer=null;const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
function boolText(v){return v===true?'Yes':v===false?'No':'—'} function fmt(v,n=3){return (typeof v==='number'&&isFinite(v))?v.toFixed(n):(v??'—')} function probOne(r){if(!r||!r.probabilities)return null;for(const k of ['1','true','True','YES','yes'])if(k in r.probabilities)return r.probabilities[k];return null}
function gridDims(grid){return grid&&grid.length?[grid[0].length,grid.length]:[0,0]} function normalizeGrid(grid,w,h){if(!grid)return Array.from({length:h},()=>Array(w).fill('U'));return Array.from({length:h},(_,y)=>Array.from({length:w},(_,x)=>grid[y]?.[x]??'U'))}
function renderGrid(title,grid,step,confidence,note){const base=DATA.true_map||step.gold_belief||step.model_belief||step.decoded_belief;let [w,h]=gridDims(base);if(!w){w=5;h=5}grid=normalizeGrid(grid,w,h);const pastPath=DATA.path.slice(0,index+1);let cells='';for(let y=0;y<h;y++)for(let x=0;x<w;x++){let val=String(grid[y][x]??'U').toUpperCase();let classes=['cell',val];const isPast=pastPath.some(p=>p&&p[0]===x&&p[1]===y);const isCurrent=step.current_pos&&step.current_pos[0]===x&&step.current_pos[1]===y;const isStart=DATA.start&&DATA.start[0]===x&&DATA.start[1]===y;const isGoal=DATA.goal&&DATA.goal[0]===x&&DATA.goal[1]===y;if(isPast)classes.push('past');if(isCurrent)classes.push('current');if(isStart)classes.push('start');if(isGoal)classes.push('goal');let conf=confidence?.[y]?.[x];cells+=`<div class="${classes.join(' ')}" title="(${x}, ${y}) = ${esc(val)}${conf!=null?' / confidence '+fmt(conf):''}"><span class="coord">${x},${y}</span>${esc(val)}${conf!=null?`<span class="conf">${Math.round(conf*100)}%</span>`:''}</div>`}return `<div class="map-card"><div class="map-title"><span>${esc(title)}</span><span class="map-note">${esc(note||'')}</span></div><div class="grid-board" style="grid-template-columns:repeat(${w},1fr)">${cells}</div></div>`}
function metric(k,v,cls=''){return `<div class="metric"><div class="k">${esc(k)}</div><div class="v ${cls}">${esc(v)}</div></div>`}
function planningCards(probes){const cards=[];for(const dir of ['UP','DOWN','LEFT','RIGHT']){const names=[`true_action_${dir}_is_astar_best`,`requested_action_${dir}_is_astar_best`];const key=names.find(n=>probes[n]);if(key){const r=probes[key],p=probOne(r);cards.push(`<div class="probe-card"><div class="probe-head"><b>${dir} optimality</b><span>${p==null?esc(r.prediction):Math.round(p*100)+'%'}</span></div><div class="bar"><span style="width:${Math.round((p??r.confidence??0)*100)}%"></span></div><div class="small">${esc(r.position)} · L${r.layer} · target ${esc(r.target)} · ref F1 ${fmt(r.reference_macro_f1)}</div></div>`)}}for(const key of ['loop_risk','position_seen_before','position_action_seen_before','chosen_action_is_astar_best','requested_action_is_astar_best','chosen_action_reduces_true_distance','requested_action_reduces_true_distance']){const r=probes[key];if(!r)continue;const p=probOne(r);cards.push(`<div class="probe-card"><div class="probe-head"><b>${esc(key.replaceAll('_',' '))}</b><span>${p==null?esc(r.prediction):Math.round(p*100)+'%'}</span></div><div class="bar"><span style="width:${Math.round((p??r.confidence??0)*100)}%"></span></div><div class="small">${esc(r.position)} · L${r.layer} · target ${esc(r.target)} · ${r.correct?'correct':'incorrect'} · ref F1 ${fmt(r.reference_macro_f1)}</div></div>`)}return cards.length?cards.join(''):'<div class="empty">No step-level probe predictions were embedded. Rebuild with --fit-step-probes.</div>'}
function probeTable(probes){const rows=Object.entries(probes).sort((a,b)=>(a[1].task_group+a[0]).localeCompare(b[1].task_group+b[0]));if(!rows.length)return '<div class="empty">No step-level probe predictions.</div>';let out='<table class="probe-table"><thead><tr><th>Group</th><th>Task</th><th>Site</th><th>Target</th><th>Prediction</th><th>Confidence</th><th>Reference F1</th></tr></thead><tbody>';for(const [task,r] of rows){out+=`<tr><td>${esc(r.task_group)}</td><td>${esc(task)}</td><td>${esc(r.position)} / L${r.layer}</td><td>${esc(r.target)}</td><td><span class="badge ${r.correct?'ok':'no'}">${esc(r.prediction)}</span></td><td>${Math.round((r.confidence??0)*100)}%</td><td>${fmt(r.reference_macro_f1)}${r.reference_baseline!=null?' (base '+fmt(r.reference_baseline)+')':''}</td></tr>`}return out+'</tbody></table>'}
function aggregateTable(){const rows=DATA.aggregate_probes||[];if(!rows.length)return '<div class="empty">No aggregate probe table found in this run.</div>';let out='<table class="probe-table"><thead><tr><th>Group</th><th>Task</th><th>Best site</th><th>Macro-F1</th><th>Baseline</th></tr></thead><tbody>';for(const r of rows.sort((a,b)=>(a.task_group+a.task).localeCompare(b.task_group+b.task))){out+=`<tr><td>${esc(r.task_group)}</td><td>${esc(r.task)}</td><td>${esc(r.position)} / L${r.layer}</td><td>${fmt(r.macro_f1)}</td><td>${fmt(r.baseline)}</td></tr>`}return out+'</tbody></table>'}
function renderTimeline(){let out='';for(const s of DATA.steps){let c=['dot'];if(s.index===index)c.push('current');if(s.loop.short_cycle||s.loop.position_action_seen_before)c.push('loop');else if(s.loop.position_seen_before)c.push('repeat');else c.push('good');out+=`<div class="${c.join(' ')}" data-step="${s.step_id}" title="Step ${s.step_id}" onclick="go(${s.index})"></div>`}$('timeline').innerHTML=out}
function render(){const s=DATA.steps[index];$('title').textContent=DATA.title;$('subtitle').textContent=`Run: ${DATA.run}`;$('slider').max=Math.max(0,DATA.steps.length-1);$('slider').value=index;$('stepLabel').textContent=`Step ${s.step_id} (${index+1}/${DATA.steps.length})`;const loop=s.loop;const actionMismatch=s.requested_action&&s.executed_action&&s.requested_action!==s.executed_action;$('metrics').innerHTML=metric('Episode',DATA.episode_id)+metric('Outcome',DATA.success===true?'Success':DATA.success===false?'Failure':'Unknown',DATA.success===true?'good':DATA.success===false?'bad':'')+metric('Current position',s.current_pos?`(${s.current_pos.join(', ')})`:'—')+metric('Requested action',s.requested_action||'—',actionMismatch?'warn':'')+metric('Executed action',s.executed_action||'—',s.invalid_move?'bad':'')+metric('Repeated state',boolText(loop.position_seen_before),loop.position_seen_before?'warn':'')+metric('Cycle length',loop.cycle_length??'—',loop.short_cycle?'bad':'');
$('maps').innerHTML=renderGrid('True map',DATA.true_map,s,null,'oracle')+renderGrid('Gold observable belief',s.gold_belief,s,null,'from interaction history')+renderGrid('Model explicit belief',s.model_belief,s,null,s.model_belief?'model output':'not available')+renderGrid('Probe-decoded belief',s.decoded_belief,s,s.decoded_confidence,s.decoded_belief?'held-out episode':'not fitted');
$('planningProbes').innerHTML=planningCards(s.probes||{});$('probeTable').innerHTML=probeTable(s.probes||{});document.querySelector('#prompt .code').textContent=s.prompt||'';document.querySelector('#output .code').textContent=s.raw_response||JSON.stringify(s.response_object,null,2)||'';document.querySelector('#targets .code').textContent=JSON.stringify(s.target||{},null,2);document.querySelector('#raw .code').textContent=JSON.stringify(s.raw_step||{},null,2);document.querySelector('#aggregate .table-wrap').innerHTML=aggregateTable();renderTimeline()}
function go(i){index=Math.max(0,Math.min(DATA.steps.length-1,Number(i)||0));render()} function next(){if(index>=DATA.steps.length-1){stop();return}go(index+1)}function prev(){go(index-1)}function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='▶ Play'}function play(){if(timer){stop();return}$('play').textContent='⏸ Pause';timer=setInterval(next,Number($('speed').value))}function jumpLoop(){const i=DATA.steps.findIndex(s=>s.loop.short_cycle||s.loop.position_action_seen_before);if(i>=0)go(i)}
$('prev').onclick=prev;$('next').onclick=next;$('play').onclick=play;$('jumpLoop').onclick=jumpLoop;$('slider').oninput=e=>go(e.target.value);$('speed').onchange=()=>{if(timer){stop();play()}};document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.tab-content').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.tab).classList.add('active')});document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowRight')next();else if(e.key==='ArrowLeft')prev();else if(e.code==='Space'){e.preventDefault();play()}});render();
</script></body></html>'''


def write_html(payload: Mapping[str, Any], output: Path) -> None:
    data_json = json.dumps(json_safe(payload), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(str(payload.get("title", "Trajectory Probe Viewer")))
    content = HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory containing steps.jsonl")
    parser.add_argument("--episode", default="auto-loop", help="Episode id, substring, auto-loop, auto-failure, or longest")
    parser.add_argument("--output", help="Output HTML path; defaults to RUN/trajectory_viewer/<episode>.html")
    parser.add_argument("--fit-step-probes", action="store_true", help="Fit leave-one-episode-out linear probes and embed step predictions")
    parser.add_argument("--probe-groups", default="planning,cells", help="Comma-separated task groups for step probes")
    parser.add_argument("--probe-site", default="decision", help="decision, best, or an exact activation position")
    parser.add_argument("--activation-dir", help="Override activation directory")
    parser.add_argument("--probe-results", help="Override best_by_task/task_layer_curves/probe_results CSV")
    parser.add_argument("--max-probe-tasks", type=int, help="Optional task limit for a quick smoke test")
    parser.add_argument("--force-probes", action="store_true", help="Ignore cached step-probe predictions")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--list-episodes", action="store_true", help="Print ranked episode ids and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = Path(args.run).expanduser().resolve()
    steps_path = discover_file(run, ("steps.jsonl",))
    if steps_path is None:
        raise FileNotFoundError(f"Missing {run / 'steps.jsonl'}")
    step_rows = load_jsonl(steps_path)
    grouped = group_steps(step_rows)
    if not grouped:
        raise ValueError(f"No steps found in {steps_path}")
    summaries = extract_summary_rows(run)
    target_map, target_path = load_targets(run)
    if args.list_episodes:
        ranked = sorted(grouped, key=lambda ep: loop_score(grouped[ep], target_map), reverse=True)
        print("episode_id\tsteps\tsuccess\tloop_score")
        for episode in ranked[:200]:
            print(f"{episode}\t{len(grouped[episode])}\t{extract_success(summaries.get(episode), grouped[episode])}\t{loop_score(grouped[episode], target_map):.2f}")
        return
    episode_id = select_episode(args.episode, grouped, summaries, target_map)
    eprint(f"[viewer] selected episode={episode_id} steps={len(grouped[episode_id])}")
    if target_path:
        eprint(f"[viewer] targets={target_path}")
    else:
        eprint("[viewer] warning: no target JSONL found; gold belief and step probes may be unavailable")
    probe_predictions: dict[tuple[str, int], dict[str, Any]] = {}
    if args.fit_step_probes:
        if not target_map:
            raise FileNotFoundError("--fit-step-probes requires a target JSONL file")
        groups_requested = {part.strip() for part in args.probe_groups.split(",") if part.strip()}
        probe_predictions = fit_step_probes(
            run=run,
            episode_id=episode_id,
            target_map=target_map,
            groups_requested=groups_requested,
            activation_dir_arg=args.activation_dir,
            probe_table_arg=args.probe_results,
            site_policy=args.probe_site,
            max_tasks=args.max_probe_tasks,
            force=args.force_probes,
            random_state=args.random_state,
        )
    payload = make_payload(
        run=run,
        episode_id=episode_id,
        steps=grouped[episode_id],
        summary=summaries.get(episode_id),
        target_map=target_map,
        probe_predictions=probe_predictions,
        aggregate_probes=summary_probe_rows(run),
    )
    output = Path(args.output).expanduser().resolve() if args.output else run / "trajectory_viewer" / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', episode_id)}.html"
    write_html(payload, output)
    print(f"[trajectory_probe_viewer] episode={episode_id}")
    print(f"[trajectory_probe_viewer] steps={len(grouped[episode_id])}")
    print(f"[trajectory_probe_viewer] output={output}")
    if not args.fit_step_probes:
        print("[trajectory_probe_viewer] note=step-level probes not fitted; add --fit-step-probes to enable them")


if __name__ == "__main__":
    main()
PYCODE

chmod +x scripts/trajectory_probe_viewer.py
python -m py_compile scripts/trajectory_probe_viewer.py

cat <<'MSG'
Installed:
  scripts/trajectory_probe_viewer.py

Quick usage:
  python scripts/trajectory_probe_viewer.py --run "$RUN" --list-episodes

Build the highest-loop trajectory without per-step probe fitting:
  python scripts/trajectory_probe_viewer.py \
    --run "$RUN" \
    --episode auto-loop

Build a full viewer with held-out step-level probe predictions:
  python scripts/trajectory_probe_viewer.py \
    --run "$RUN" \
    --episode auto-loop \
    --fit-step-probes \
    --probe-groups planning,cells \
    --probe-site decision

Serve the generated files from a remote server:
  python -m http.server 8000 --directory "$RUN/trajectory_viewer"
MSG
