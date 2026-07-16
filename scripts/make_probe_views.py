#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def direction_class(start: list[int], goal: list[int]) -> str:
    dx = int(goal[0]) - int(start[0])
    dy = int(goal[1]) - int(start[1])
    vertical = "N" if dy > 0 else "S" if dy < 0 else ""
    horizontal = "E" if dx > 0 else "W" if dx < 0 else ""
    return vertical + horizontal or "SAME"


def match_episode_ids(
    episodes: dict[str, dict[str, Any]],
    maps: dict[str, dict[str, Any]],
    eligible_episode_ids: set[str],
    seed: int,
) -> tuple[set[str], set[str], dict[str, Any]]:
    success_by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    failure_by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)

    for episode_id, episode in episodes.items():
        if episode_id not in eligible_episode_ids:
            continue
        map_row = maps.get(episode_id, {})
        start = map_row.get("start", episode.get("start", [0, 0]))
        goal = map_row.get("goal", episode.get("goal", [0, 0]))
        key = (
            str(map_row.get("difficulty", "unknown")),
            str(map_row.get("direction_class", direction_class(start, goal))),
        )
        target = success_by_stratum if episode.get("success") else failure_by_stratum
        target[key].append(episode_id)

    rng = random.Random(seed)
    selected_success: set[str] = set()
    selected_failure: set[str] = set()
    strata = []

    for key in sorted(set(success_by_stratum) | set(failure_by_stratum)):
        success_ids = sorted(success_by_stratum.get(key, []))
        failure_ids = sorted(failure_by_stratum.get(key, []))
        count = min(len(success_ids), len(failure_ids))
        if count == 0:
            continue
        selected_success.update(rng.sample(success_ids, count))
        selected_failure.update(rng.sample(failure_ids, count))
        strata.append(
            {
                "difficulty": key[0],
                "direction_class": key[1],
                "success_available": len(success_ids),
                "failure_available": len(failure_ids),
                "selected_each": count,
            }
        )

    if not selected_success or not selected_failure:
        raise ValueError(
            "Could not create matched success/failure sets. "
            "Check that each difficulty × direction stratum contains both outcomes."
        )

    return selected_success, selected_failure, {
        "matching_variables": ["difficulty", "direction_class"],
        "selected_success_episodes": len(selected_success),
        "selected_failure_episodes": len(selected_failure),
        "strata": strata,
    }


def create_view(
    *,
    run: Path,
    output_root: Path,
    name: str,
    allowed_episode_ids: set[str] | None,
    require_legal_action: bool,
    activation_meta: list[dict[str, Any]],
    steps_by_key: dict[tuple[str, int], dict[str, Any]],
    targets_by_key: dict[tuple[str, int], dict[str, Any]],
    episodes: dict[str, dict[str, Any]],
    maps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    view = output_root / name
    if view.exists():
        shutil.rmtree(view)
    (view / "activations").mkdir(parents=True)
    (view / "targets").mkdir(parents=True)

    selected_meta = []
    selected_keys: set[tuple[str, int]] = set()
    for meta in activation_meta:
        key = (str(meta["episode_id"]), int(meta["step_id"]))
        step = steps_by_key.get(key)
        if step is None or key not in targets_by_key:
            continue
        if allowed_episode_ids is not None and key[0] not in allowed_episode_ids:
            continue
        if require_legal_action and bool(
            step.get("illegal_action_before_fallback")
        ):
            continue
        selected_meta.append(meta)
        selected_keys.add(key)

    selected_episode_ids = {key[0] for key in selected_keys}
    selected_targets = [targets_by_key[key] for key in sorted(selected_keys)]
    selected_steps = [steps_by_key[key] for key in sorted(selected_keys)]
    selected_episodes = [
        episodes[episode_id]
        for episode_id in sorted(selected_episode_ids)
        if episode_id in episodes
    ]
    selected_maps = [
        maps[episode_id]
        for episode_id in sorted(selected_episode_ids)
        if episode_id in maps
    ]

    for filename in ["X.npy", "position_mask.npy", "positions.json", "layers.npy"]:
        source = run / "activations" / filename
        if not source.exists():
            raise FileNotFoundError(source)
        link_or_copy(source, view / "activations" / filename)

    write_jsonl(view / "activations" / "meta.jsonl", selected_meta)
    write_jsonl(view / "targets" / "targets.jsonl", selected_targets)
    link_or_copy(
        run / "targets" / "task_catalog.json",
        view / "targets" / "task_catalog.json",
    )
    write_jsonl(view / "steps.jsonl", selected_steps)
    write_jsonl(view / "episodes.jsonl", selected_episodes)
    write_jsonl(view / "maps.jsonl", selected_maps)

    if (run / "resolved_config.yaml").exists():
        link_or_copy(run / "resolved_config.yaml", view / "resolved_config.yaml")

    successes = sum(bool(row.get("success")) for row in selected_episodes)
    manifest = {
        "schema_version": "1.0",
        "source_run": str(run),
        "view": name,
        "require_legal_action": require_legal_action,
        "episodes": len(selected_episode_ids),
        "successes": successes,
        "success_rate": (
            successes / len(selected_episode_ids)
            if selected_episode_ids
            else None
        ),
        "activation_rows": len(selected_meta),
        "target_rows": len(selected_targets),
        "note": (
            "X.npy and position_mask.npy are linked to the source run. "
            "Filtered meta rows retain original row_index values."
        ),
    }
    (view / "view_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create legal-action and matched success/failure probe views "
            "without copying activation tensors."
        )
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    run = args.run.resolve()
    output_root = (args.output_root or run / "probe_views").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    required = [
        run / "maps.jsonl",
        run / "episodes.jsonl",
        run / "steps.jsonl",
        run / "targets" / "targets.jsonl",
        run / "targets" / "task_catalog.json",
        run / "activations" / "meta.jsonl",
        run / "activations" / "X.npy",
        run / "activations" / "position_mask.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing run artifacts:\n" + "\n".join(missing))

    maps = {str(row["episode_id"]): row for row in read_jsonl(run / "maps.jsonl")}
    episodes = {
        str(row["episode_id"]): row
        for row in read_jsonl(run / "episodes.jsonl")
    }
    steps_by_key = {
        (str(row["episode_id"]), int(row["step_id"])): row
        for row in read_jsonl(run / "steps.jsonl")
    }
    targets_by_key = {
        (str(row["episode_id"]), int(row["step_id"])): row
        for row in read_jsonl(run / "targets" / "targets.jsonl")
    }
    activation_meta = read_jsonl(run / "activations" / "meta.jsonl")

    legal_activation_episode_ids = {
        str(meta["episode_id"])
        for meta in activation_meta
        if not bool(
            steps_by_key.get(
                (str(meta["episode_id"]), int(meta["step_id"])), {}
            ).get("illegal_action_before_fallback")
        )
    }
    matched_success, matched_failure, matching = match_episode_ids(
        episodes,
        maps,
        legal_activation_episode_ids,
        args.seed,
    )
    (output_root / "matching_manifest.json").write_text(
        json.dumps(matching, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifests = [
        create_view(
            run=run,
            output_root=output_root,
            name="legal_all",
            allowed_episode_ids=None,
            require_legal_action=True,
            activation_meta=activation_meta,
            steps_by_key=steps_by_key,
            targets_by_key=targets_by_key,
            episodes=episodes,
            maps=maps,
        ),
        create_view(
            run=run,
            output_root=output_root,
            name="matched_success_legal",
            allowed_episode_ids=matched_success,
            require_legal_action=True,
            activation_meta=activation_meta,
            steps_by_key=steps_by_key,
            targets_by_key=targets_by_key,
            episodes=episodes,
            maps=maps,
        ),
        create_view(
            run=run,
            output_root=output_root,
            name="matched_failure_legal",
            allowed_episode_ids=matched_failure,
            require_legal_action=True,
            activation_meta=activation_meta,
            steps_by_key=steps_by_key,
            targets_by_key=targets_by_key,
            episodes=episodes,
            maps=maps,
        ),
    ]

    print(f"Created probe views under: {output_root}")
    for manifest in manifests:
        print(
            f"- {manifest['view']}: episodes={manifest['episodes']} "
            f"rows={manifest['activation_rows']} "
            f"success_rate={manifest['success_rate']}"
        )


if __name__ == "__main__":
    main()
