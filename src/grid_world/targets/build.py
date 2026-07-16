from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from grid_world.env.belief import (
    initial_belief,
    rows_to_belief,
    update_belief,
)
from grid_world.env.grid import ACTIONS, DELTAS, GridSpec
from grid_world.env.planning import distance_map
from grid_world.utils.io import (
    read_jsonl,
    write_json,
    write_jsonl,
)
from grid_world.utils.manifest import write_manifest


def _cell_key(prefix: str, x: int, y: int, suffix: str) -> str:
    return f"{prefix}_cell_x{x}_y{y}_{suffix}"


def _catalog(size: int):
    tasks = {}

    for direction in ACTIONS:
        tasks[f"gold_local_{direction}_OFUW"] = {
            "group": "local",
            "type": "categorical",
            "description": (
                "Observable state of the adjacent location after applying "
                "all feedback available at this step."
            ),
        }

    for x in range(size):
        for y in range(size):
            tasks[_cell_key("gold", x, y, "OFU")] = {
                "group": "cells",
                "type": "categorical",
                "map_target": "observable_belief",
                "description": (
                    "Correct observable map state: obstacle, free, or unknown."
                ),
            }
            tasks[_cell_key("gold", x, y, "known")] = {
                "group": "memory",
                "type": "binary",
                "map_target": "observed_mask",
                "description": (
                    "Whether this location has been observed by this step."
                ),
            }
            tasks[_cell_key("model", x, y, "OFU")] = {
                "group": "explicit_cells",
                "type": "categorical",
                "map_target": "explicit_model_output",
                "description": (
                    "State written by the model in its current explicit "
                    "belief-grid output. Missing in the no-grid condition."
                ),
            }
            tasks[_cell_key("true", x, y, "FO")] = {
                "group": "true_cells",
                "type": "categorical",
                "map_target": "complete_true_map",
                "description": (
                    "Complete true map state, regardless of observation."
                ),
            }
            tasks[_cell_key("true", x, y, "FO_observed")] = {
                "group": "true_cells_observed",
                "type": "categorical",
                "map_target": "true_map_observed_only",
                "description": (
                    "True free/obstacle state, included only after observation."
                ),
            }
            tasks[_cell_key("true", x, y, "FO_unobserved")] = {
                "group": "true_cells_unobserved",
                "type": "categorical",
                "map_target": "true_map_unobserved_only",
                "description": (
                    "True free/obstacle state, included only while unobserved."
                ),
            }

    for name in [
        "chosen_action_is_astar_best",
        "chosen_action_reduces_true_distance",
        "position_seen_before",
        "position_action_seen_before",
        "loop_risk",
    ]:
        tasks[name] = {
            "group": "planning",
            "type": "binary",
        }

    for direction in ACTIONS:
        tasks[f"true_action_{direction}_is_astar_best"] = {
            "group": "planning",
            "type": "binary",
        }

    tasks["gold_action_target_belief"] = {
        "group": "planning",
        "type": "categorical",
    }
    tasks["model_missed_any_gold_known"] = {
        "group": "faithfulness",
        "type": "binary",
    }
    tasks["action_optimal_but_belief_incomplete"] = {
        "group": "faithfulness",
        "type": "binary",
    }
    return tasks


def build_targets(run_dir: str | Path):
    run = Path(run_dir)
    maps = {
        str(row["episode_id"]): GridSpec.from_dict(row)
        for row in read_jsonl(run / "maps.jsonl")
    }

    grouped = defaultdict(list)
    for row in read_jsonl(run / "steps.jsonl"):
        grouped[str(row["episode_id"])].append(row)

    output = []
    max_size = 0

    for episode_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["step_id"]))
        spec = maps[episode_id]
        max_size = max(max_size, spec.size)
        belief = initial_belief(spec)
        distances = distance_map(spec)
        seen_positions = set()
        seen_pairs = set()
        visit_counts = defaultdict(int)

        for step in rows:
            position = tuple(int(value) for value in step["current_pos"])
            belief = update_belief(belief, step["feedback"])
            action = str(step["action"]).upper()

            action_distances = {}
            for direction in ACTIONS:
                dx, dy = DELTAS[direction]
                action_distances[direction] = distances.get(
                    (position[0] + dx, position[1] + dy)
                )

            valid_distances = [
                value
                for value in action_distances.values()
                if value is not None
            ]
            best_distance = (
                min(valid_distances) if valid_distances else None
            )
            best_actions = [
                direction
                for direction, value in action_distances.items()
                if value is not None and value == best_distance
            ]

            dx, dy = DELTAS[action]
            target = (position[0] + dx, position[1] + dy)
            if not (
                0 <= target[0] < spec.size
                and 0 <= target[1] < spec.size
            ):
                target_belief = "WALL"
            else:
                target_belief = belief[target]

            current_distance = distances.get(position)
            record = {
                "schema_version": "1.0",
                "episode_id": episode_id,
                "step_id": int(step["step_id"]),
                "gold_action_target_belief": target_belief,
                "a_star_best_actions": best_actions,
                "chosen_action_is_astar_best": int(
                    action in best_actions
                ),
                "chosen_action_reduces_true_distance": int(
                    current_distance is not None
                    and action_distances.get(action) is not None
                    and action_distances[action] < current_distance
                ),
                "gold_known_cell_count": sum(
                    value != "U" for value in belief.values()
                ),
                "position_seen_before": int(
                    position in seen_positions
                ),
                "position_action_seen_before": int(
                    (position, action) in seen_pairs
                ),
                "loop_risk": int(
                    visit_counts[position] >= 2
                    or (position, action) in seen_pairs
                ),
            }

            for direction in ACTIONS:
                record[
                    f"true_action_{direction}_is_astar_best"
                ] = int(direction in best_actions)
                adjacent_dx, adjacent_dy = DELTAS[direction]
                adjacent = (
                    position[0] + adjacent_dx,
                    position[1] + adjacent_dy,
                )
                if not (
                    0 <= adjacent[0] < spec.size
                    and 0 <= adjacent[1] < spec.size
                ):
                    local_value = "WALL"
                else:
                    local_value = belief[adjacent]
                record[
                    f"gold_local_{direction}_OFUW"
                ] = local_value

            model_grid = step.get("parsed_belief_grid")
            try:
                model_belief = (
                    rows_to_belief(model_grid, spec.size)
                    if isinstance(model_grid, list)
                    else None
                )
            except ValueError:
                model_belief = None

            known_correct = []
            missed = False

            for x in range(spec.size):
                for y in range(spec.size):
                    coord = (x, y)
                    observable_value = belief[coord]
                    true_value = (
                        "O" if coord in spec.obstacles else "F"
                    )
                    model_value = (
                        model_belief[coord]
                        if model_belief is not None
                        else None
                    )

                    record[
                        _cell_key("gold", x, y, "OFU")
                    ] = observable_value
                    record[
                        _cell_key("gold", x, y, "known")
                    ] = int(observable_value != "U")
                    record[
                        _cell_key("model", x, y, "OFU")
                    ] = model_value
                    record[
                        _cell_key("true", x, y, "FO")
                    ] = true_value
                    record[
                        _cell_key("true", x, y, "FO_observed")
                    ] = (
                        true_value
                        if observable_value != "U"
                        else None
                    )
                    record[
                        _cell_key("true", x, y, "FO_unobserved")
                    ] = (
                        true_value
                        if observable_value == "U"
                        else None
                    )

                    if (
                        model_belief is not None
                        and observable_value != "U"
                    ):
                        known_correct.append(
                            model_value == observable_value
                        )
                        missed = missed or model_value == "U"

            if model_belief is None:
                record["model_known_cell_acc_step"] = None
                record["model_missed_any_gold_known"] = None
                record[
                    "action_optimal_but_belief_incomplete"
                ] = None
            else:
                record["model_known_cell_acc_step"] = (
                    sum(known_correct) / len(known_correct)
                    if known_correct
                    else None
                )
                record["model_missed_any_gold_known"] = int(missed)
                record[
                    "action_optimal_but_belief_incomplete"
                ] = int(
                    record["chosen_action_is_astar_best"]
                    and record["model_missed_any_gold_known"]
                )

            output.append(record)
            seen_positions.add(position)
            seen_pairs.add((position, action))
            visit_counts[position] += 1

    target_dir = run / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(target_dir / "targets.jsonl", output)

    catalog = _catalog(max_size)
    write_json(target_dir / "task_catalog.json", catalog)
    write_manifest(
        target_dir / "manifest.json",
        stage="targets",
        config={
            "groups": sorted(
                {metadata["group"] for metadata in catalog.values()}
            )
        },
        counts={
            "rows": len(output),
            "tasks": len(catalog),
        },
        upstream={"run": str(run)},
    )
    return output
