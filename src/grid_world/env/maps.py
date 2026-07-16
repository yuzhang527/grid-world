from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from grid_world.config import load_yaml
from grid_world.env.grid import GridSpec, as_coord, render_grid
from grid_world.env.planning import (
    count_shortest_paths,
    shortest_actions,
    shortest_path_length,
)
from grid_world.utils.io import read_jsonl, write_json, write_jsonl
from grid_world.utils.manifest import write_manifest


def _direction_class(start: tuple[int, int], goal: tuple[int, int]) -> str:
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    vertical = "N" if dy > 0 else "S" if dy < 0 else ""
    horizontal = "E" if dx > 0 else "W" if dx < 0 else ""
    return vertical + horizontal


def _manhattan(start: tuple[int, int], goal: tuple[int, int]) -> int:
    return abs(goal[0] - start[0]) + abs(goal[1] - start[1])


def _largest_remainder_quotas(
    names_and_weights: list[tuple[str, float]],
    total: int,
) -> dict[str, int]:
    if total < 1:
        raise ValueError("total must be positive")
    if not names_and_weights:
        raise ValueError("At least one weighted category is required")
    weight_sum = sum(weight for _, weight in names_and_weights)
    if weight_sum <= 0:
        raise ValueError("Category weights must sum to a positive value")

    exact = {
        name: total * weight / weight_sum
        for name, weight in names_and_weights
    }
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        exact,
        key=lambda name: (exact[name] - quotas[name], name),
        reverse=True,
    )
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def _weighted_schedule(
    names_and_weights: list[tuple[str, float]],
    total: int,
    rng: random.Random,
) -> list[str]:
    quotas = _largest_remainder_quotas(names_and_weights, total)
    schedule = [
        name
        for name, _ in names_and_weights
        for _ in range(quotas[name])
    ]
    rng.shuffle(schedule)
    return schedule


def _candidate_pairs(
    size: int,
    direction_classes: list[str],
    min_manhattan_distance: int,
    max_manhattan_distance: int | None,
) -> dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]:
    cells = [(x, y) for x in range(size) for y in range(size)]
    result = {name: [] for name in direction_classes}
    for start in cells:
        for goal in cells:
            if start == goal:
                continue
            distance = _manhattan(start, goal)
            if distance < min_manhattan_distance:
                continue
            if max_manhattan_distance is not None and distance > max_manhattan_distance:
                continue
            direction = _direction_class(start, goal)
            if direction in result:
                result[direction].append((start, goal))
    missing = [name for name, pairs in result.items() if not pairs]
    if missing:
        raise ValueError(
            "No valid start/goal pairs for direction classes "
            f"{missing}; relax the Manhattan-distance constraints."
        )
    return result


def _normalize_difficulty_buckets(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw = config.get("difficulty_buckets")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "diverse generation requires a non-empty difficulty_buckets list"
        )

    buckets = []
    seen_names = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each difficulty bucket must be a mapping")
        name = str(item.get("name", "")).strip()
        if not name or name in seen_names:
            raise ValueError(f"Invalid or duplicate difficulty name: {name!r}")
        seen_names.add(name)
        obstacle_counts = [int(value) for value in item.get("obstacle_counts", [])]
        if not obstacle_counts:
            raise ValueError(f"{name}: obstacle_counts must not be empty")
        buckets.append(
            {
                "name": name,
                "weight": float(item.get("weight", 1.0)),
                "obstacle_counts": obstacle_counts,
                "min_detour": int(item.get("min_detour", 0)),
                "max_detour": (
                    int(item["max_detour"])
                    if item.get("max_detour") is not None
                    else None
                ),
                "min_path_length": (
                    int(item["min_path_length"])
                    if item.get("min_path_length") is not None
                    else None
                ),
                "max_path_length": (
                    int(item["max_path_length"])
                    if item.get("max_path_length") is not None
                    else None
                ),
            }
        )
    return buckets


def _generate_fixed(config: dict[str, Any]) -> list[dict[str, Any]]:
    size = int(config.get("size", 5))
    count = int(config.get("num_episodes", 100))
    obstacle_count = int(config.get("num_obstacles", 4))
    seed_start = int(config.get("seed_start", 123))
    prefix = str(config.get("prefix", "A_seed"))
    start = as_coord(config.get("start", [0, 0]))
    goal = as_coord(config.get("goal", [size - 1, size - 1]))
    unique = bool(config.get("require_unique_shortest_path", False))
    max_attempts = int(config.get("max_attempts_per_episode", 10000))

    all_cells = [
        (x, y)
        for x in range(size)
        for y in range(size)
        if (x, y) not in {start, goal}
    ]
    rows = []
    for offset in range(count):
        seed = seed_start + offset
        rng = random.Random(seed)
        accepted = None
        for _ in range(max_attempts):
            obstacles = frozenset(rng.sample(all_cells, obstacle_count))
            spec = GridSpec(
                episode_id=f"{prefix}{seed}",
                seed=seed,
                size=size,
                start=start,
                goal=goal,
                obstacles=obstacles,
            )
            distance = shortest_path_length(spec)
            if distance is None:
                continue
            if unique and count_shortest_paths(spec) != 1:
                continue
            accepted = GridSpec(
                episode_id=spec.episode_id,
                seed=seed,
                size=size,
                start=start,
                goal=goal,
                obstacles=obstacles,
                shortest_path_length=distance,
            )
            break
        if accepted is None:
            raise RuntimeError(f"Could not generate a valid map for seed {seed}")
        row = accepted.to_dict()
        row.update(
            {
                "generation_mode": "fixed",
                "direction_class": _direction_class(start, goal),
                "difficulty": "fixed",
                "obstacle_count": obstacle_count,
                "manhattan_distance": _manhattan(start, goal),
                "detour": distance - _manhattan(start, goal),
            }
        )
        rows.append(row)
    return rows


def _generate_diverse(config: dict[str, Any]) -> list[dict[str, Any]]:
    size = int(config.get("size", 5))
    count = int(config.get("num_episodes", 1000))
    dataset_seed = int(config.get("dataset_seed", 20260715))
    seed_start = int(config.get("seed_start", 10000))
    prefix = str(config.get("prefix", "D5_ep"))
    max_attempts = int(config.get("max_attempts_per_episode", 50000))
    unique_shortest = bool(config.get("require_unique_shortest_path", False))
    min_manhattan = int(config.get("min_manhattan_distance", size - 1))
    max_manhattan = (
        int(config["max_manhattan_distance"])
        if config.get("max_manhattan_distance") is not None
        else None
    )
    direction_classes = [
        str(value).upper()
        for value in config.get("direction_classes", ["NE", "NW", "SE", "SW"])
    ]
    direction_weights_cfg = config.get("direction_weights", {})
    direction_weights = [
        (
            direction,
            float(direction_weights_cfg.get(direction, 1.0))
            if isinstance(direction_weights_cfg, dict)
            else 1.0,
        )
        for direction in direction_classes
    ]

    buckets = _normalize_difficulty_buckets(config)
    bucket_by_name = {bucket["name"]: bucket for bucket in buckets}
    candidate_pairs = _candidate_pairs(
        size,
        direction_classes,
        min_manhattan,
        max_manhattan,
    )

    schedule_rng = random.Random(dataset_seed)
    direction_schedule = _weighted_schedule(direction_weights, count, schedule_rng)
    difficulty_schedule = _weighted_schedule(
        [(bucket["name"], bucket["weight"]) for bucket in buckets],
        count,
        schedule_rng,
    )

    all_cells = [(x, y) for x in range(size) for y in range(size)]
    signatures = set()
    rows = []

    for offset, (target_direction, difficulty_name) in enumerate(
        zip(direction_schedule, difficulty_schedule)
    ):
        episode_seed = seed_start + offset
        episode_id = f"{prefix}{offset:06d}"
        rng = random.Random(dataset_seed * 1_000_003 + episode_seed)
        bucket = bucket_by_name[difficulty_name]
        accepted = None

        for attempt in range(1, max_attempts + 1):
            start, goal = rng.choice(candidate_pairs[target_direction])
            obstacle_count = rng.choice(bucket["obstacle_counts"])
            available_cells = [
                coord for coord in all_cells if coord not in {start, goal}
            ]
            if obstacle_count >= len(available_cells):
                raise ValueError(
                    f"{difficulty_name}: obstacle count {obstacle_count} is too large"
                )
            obstacles = frozenset(rng.sample(available_cells, obstacle_count))
            signature = (start, goal, tuple(sorted(obstacles)))
            if signature in signatures:
                continue

            spec = GridSpec(
                episode_id=episode_id,
                seed=episode_seed,
                size=size,
                start=start,
                goal=goal,
                obstacles=obstacles,
            )
            path_length = shortest_path_length(spec)
            if path_length is None:
                continue
            if unique_shortest and count_shortest_paths(spec) != 1:
                continue

            manhattan = _manhattan(start, goal)
            detour = path_length - manhattan
            if detour < bucket["min_detour"]:
                continue
            if (
                bucket["max_detour"] is not None
                and detour > bucket["max_detour"]
            ):
                continue
            if (
                bucket["min_path_length"] is not None
                and path_length < bucket["min_path_length"]
            ):
                continue
            if (
                bucket["max_path_length"] is not None
                and path_length > bucket["max_path_length"]
            ):
                continue

            accepted = {
                "schema_version": "1.0",
                "episode_id": episode_id,
                "seed": episode_seed,
                "size": size,
                "start": list(start),
                "goal": list(goal),
                "obstacles": [list(coord) for coord in sorted(obstacles)],
                "shortest_path_length": path_length,
                "generation_mode": "diverse",
                "dataset_seed": dataset_seed,
                "direction_class": target_direction,
                "difficulty": difficulty_name,
                "obstacle_count": obstacle_count,
                "manhattan_distance": manhattan,
                "detour": detour,
                "generation_attempts": attempt,
            }
            signatures.add(signature)
            break

        if accepted is None:
            raise RuntimeError(
                f"Could not generate episode {episode_id} for "
                f"direction={target_direction}, difficulty={difficulty_name} "
                f"after {max_attempts} attempts. Relax that bucket."
            )
        rows.append(accepted)

    return rows


def summarize_maps(path: str | Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("No maps found")

    direction_counts = Counter()
    difficulty_counts = Counter()
    obstacle_counts = Counter()
    path_lengths = Counter()
    manhattan_lengths = Counter()
    detours = Counter()
    unique_best_actions = Counter()
    best_action_membership = Counter()
    attempts = []

    for row in rows:
        spec = GridSpec.from_dict(row)
        direction = str(
            row.get("direction_class")
            or _direction_class(spec.start, spec.goal)
        )
        direction_counts[direction] += 1
        difficulty_counts[str(row.get("difficulty", "unknown"))] += 1
        obstacle_counts[len(spec.obstacles)] += 1
        path_length = shortest_path_length(spec)
        if path_length is None:
            raise ValueError(f"Disconnected map: {spec.episode_id}")
        manhattan = _manhattan(spec.start, spec.goal)
        path_lengths[path_length] += 1
        manhattan_lengths[manhattan] += 1
        detours[path_length - manhattan] += 1

        best_actions = shortest_actions(spec, spec.start)
        for action in best_actions:
            best_action_membership[action] += 1
        if len(best_actions) == 1:
            unique_best_actions[best_actions[0]] += 1
        else:
            unique_best_actions["TIE"] += 1

        if row.get("generation_attempts") is not None:
            attempts.append(int(row["generation_attempts"]))

    return {
        "schema_version": "1.0",
        "maps": len(rows),
        "generation_modes": dict(
            sorted(Counter(str(row.get("generation_mode", "unknown")) for row in rows).items())
        ),
        "direction_class": dict(sorted(direction_counts.items())),
        "difficulty": dict(sorted(difficulty_counts.items())),
        "obstacle_count": {
            str(key): value for key, value in sorted(obstacle_counts.items())
        },
        "shortest_path_length": {
            str(key): value for key, value in sorted(path_lengths.items())
        },
        "manhattan_distance": {
            str(key): value for key, value in sorted(manhattan_lengths.items())
        },
        "detour": {
            str(key): value for key, value in sorted(detours.items())
        },
        "optimal_start_action_membership": dict(
            sorted(best_action_membership.items())
        ),
        "unique_optimal_start_action": dict(sorted(unique_best_actions.items())),
        "generation_attempts": {
            "mean": sum(attempts) / len(attempts) if attempts else None,
            "max": max(attempts) if attempts else None,
        },
    }


def generate_maps(config_path: str | Path, output: str | Path) -> list[dict]:
    config = load_yaml(config_path)
    mode = str(config.get("generation_mode", "fixed")).lower()
    if mode == "fixed":
        rows = _generate_fixed(config)
    elif mode == "diverse":
        rows = _generate_diverse(config)
    else:
        raise ValueError("generation_mode must be 'fixed' or 'diverse'")

    write_jsonl(output, rows)
    summary = summarize_maps(output)
    summary_path = Path(output).with_suffix(".summary.json")
    write_json(summary_path, summary)
    write_manifest(
        Path(output).with_suffix(".manifest.json"),
        stage="maps",
        config=config,
        counts={"episodes": len(rows)},
        extra={"summary_path": str(summary_path)},
    )
    return rows


def validate_maps(path: str | Path) -> dict[str, int]:
    rows = read_jsonl(path)
    seen_ids = set()
    seen_signatures = set()
    for row in rows:
        spec = GridSpec.from_dict(row)
        if spec.episode_id in seen_ids:
            raise ValueError(f"Duplicate episode_id: {spec.episode_id}")
        seen_ids.add(spec.episode_id)

        signature = (spec.start, spec.goal, tuple(sorted(spec.obstacles)))
        if signature in seen_signatures:
            raise ValueError(f"Duplicate map layout: {spec.episode_id}")
        seen_signatures.add(signature)

        if spec.start in spec.obstacles or spec.goal in spec.obstacles:
            raise ValueError(f"Start/goal blocked in {spec.episode_id}")
        distance = shortest_path_length(spec)
        if distance is None:
            raise ValueError(f"No path in {spec.episode_id}")
        recorded = row.get("shortest_path_length")
        if recorded is not None and int(recorded) != distance:
            raise ValueError(
                f"Shortest-path mismatch in {spec.episode_id}: "
                f"recorded={recorded}, actual={distance}"
            )
        expected_direction = _direction_class(spec.start, spec.goal)
        if (
            row.get("direction_class") is not None
            and str(row["direction_class"]) != expected_direction
        ):
            raise ValueError(f"Direction metadata mismatch in {spec.episode_id}")
        if (
            row.get("obstacle_count") is not None
            and int(row["obstacle_count"]) != len(spec.obstacles)
        ):
            raise ValueError(f"Obstacle-count metadata mismatch in {spec.episode_id}")
        manhattan = _manhattan(spec.start, spec.goal)
        if (
            row.get("manhattan_distance") is not None
            and int(row["manhattan_distance"]) != manhattan
        ):
            raise ValueError(f"Manhattan metadata mismatch in {spec.episode_id}")
        if (
            row.get("detour") is not None
            and int(row["detour"]) != distance - manhattan
        ):
            raise ValueError(f"Detour metadata mismatch in {spec.episode_id}")

    return {
        "episodes": len(rows),
        "unique_layouts": len(seen_signatures),
    }


def show_map(
    path: str | Path,
    episode_id: str | None = None,
    index: int = 0,
) -> str:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("No maps found")
    if episode_id is not None:
        candidates = [
            row for row in rows if str(row["episode_id"]) == episode_id
        ]
        if not candidates:
            raise KeyError(f"Episode not found: {episode_id}")
        row = candidates[0]
    else:
        row = rows[index]
    spec = GridSpec.from_dict(row)
    metadata = {
        key: row.get(key)
        for key in [
            "difficulty",
            "direction_class",
            "obstacle_count",
            "manhattan_distance",
            "shortest_path_length",
            "detour",
        ]
        if row.get(key) is not None
    }
    return f"{spec.episode_id} {metadata}\n{render_grid(spec)}"
