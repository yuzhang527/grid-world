from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from grid_world.env.belief import initial_belief, rows_to_belief, update_belief
from grid_world.env.grid import ACTIONS, DELTAS, GridSpec
from grid_world.env.planning import distance_map
from grid_world.utils.io import read_jsonl, write_json, write_jsonl
from grid_world.utils.manifest import write_manifest

def _cell_key(x, y, suffix):
    return f"gold_cell_x{x}_y{y}_{suffix}"

def _catalog(size):
    tasks = {}
    for d in ACTIONS:
        tasks[f"gold_local_{d}_OFUW"] = {"group": "local", "type": "categorical"}
    for x in range(size):
        for y in range(size):
            tasks[_cell_key(x,y,"OFU")] = {"group": "cells", "type": "categorical"}
            tasks[_cell_key(x,y,"known")] = {"group": "memory", "type": "binary"}
    for name in ["chosen_action_is_astar_best","chosen_action_reduces_true_distance",
                 "position_seen_before","position_action_seen_before","loop_risk"]:
        tasks[name] = {"group": "planning", "type": "binary"}
    for d in ACTIONS:
        tasks[f"true_action_{d}_is_astar_best"] = {"group": "planning", "type": "binary"}
    tasks["gold_action_target_belief"] = {"group": "planning", "type": "categorical"}
    tasks["model_missed_any_gold_known"] = {"group": "faithfulness", "type": "binary"}
    tasks["action_optimal_but_belief_incomplete"] = {"group": "faithfulness", "type": "binary"}
    return tasks

def build_targets(run_dir: str | Path):
    run = Path(run_dir)
    maps = {str(x["episode_id"]): GridSpec.from_dict(x) for x in read_jsonl(run / "maps.jsonl")}
    grouped = defaultdict(list)
    for row in read_jsonl(run / "steps.jsonl"):
        grouped[str(row["episode_id"])].append(row)
    output, max_size = [], 0
    for episode_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda x: int(x["step_id"]))
        spec = maps[episode_id]; max_size = max(max_size, spec.size)
        belief, distances = initial_belief(spec), distance_map(spec)
        seen_positions, seen_pairs, visit_counts = set(), set(), defaultdict(int)
        for step in rows:
            position = tuple(int(x) for x in step["current_pos"])
            belief = update_belief(belief, step["feedback"])
            action = str(step["action"]).upper()
            action_distances = {}
            for d in ACTIONS:
                dx, dy = DELTAS[d]
                action_distances[d] = distances.get((position[0]+dx, position[1]+dy))
            valid = [x for x in action_distances.values() if x is not None]
            best_distance = min(valid) if valid else None
            best_actions = [d for d,v in action_distances.items() if v is not None and v == best_distance]
            dx, dy = DELTAS[action]
            target = (position[0]+dx, position[1]+dy)
            target_belief = "WALL" if not (0 <= target[0] < spec.size and 0 <= target[1] < spec.size) else belief[target]
            current_distance = distances.get(position)
            record = {
                "schema_version": "1.0", "episode_id": episode_id, "step_id": int(step["step_id"]),
                "gold_action_target_belief": target_belief, "a_star_best_actions": best_actions,
                "chosen_action_is_astar_best": int(action in best_actions),
                "chosen_action_reduces_true_distance": int(
                    current_distance is not None and action_distances.get(action) is not None
                    and action_distances[action] < current_distance),
                "gold_known_cell_count": sum(v != "U" for v in belief.values()),
                "position_seen_before": int(position in seen_positions),
                "position_action_seen_before": int((position, action) in seen_pairs),
                "loop_risk": int(visit_counts[position] >= 2 or (position, action) in seen_pairs),
            }
            for d in ACTIONS:
                record[f"true_action_{d}_is_astar_best"] = int(d in best_actions)
                ddx, ddy = DELTAS[d]; adjacent = (position[0]+ddx, position[1]+ddy)
                record[f"gold_local_{d}_OFUW"] = "WALL" if not (
                    0 <= adjacent[0] < spec.size and 0 <= adjacent[1] < spec.size) else belief[adjacent]
            for x in range(spec.size):
                for y in range(spec.size):
                    value = belief[(x,y)]
                    record[_cell_key(x,y,"OFU")] = value
                    record[_cell_key(x,y,"known")] = int(value != "U")
            model_grid, known_correct, missed = step.get("parsed_belief_grid"), [], False
            try:
                model_belief = rows_to_belief(model_grid, spec.size) if isinstance(model_grid, list) else None
            except ValueError:
                model_belief = None
            if model_belief is not None:
                for coord, gold in belief.items():
                    if gold != "U":
                        known_correct.append(model_belief[coord] == gold)
                        missed = missed or model_belief[coord] == "U"
            record["model_known_cell_acc_step"] = sum(known_correct)/len(known_correct) if known_correct else None
            record["model_missed_any_gold_known"] = int(missed or model_belief is None)
            record["action_optimal_but_belief_incomplete"] = int(
                record["chosen_action_is_astar_best"] and record["model_missed_any_gold_known"])
            output.append(record)
            seen_positions.add(position); seen_pairs.add((position,action)); visit_counts[position] += 1
    target_dir = run / "targets"; target_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(target_dir / "targets.jsonl", output)
    catalog = _catalog(max_size)
    write_json(target_dir / "task_catalog.json", catalog)
    write_manifest(target_dir / "manifest.json", stage="targets",
                   config={"groups": sorted({x["group"] for x in catalog.values()})},
                   counts={"rows": len(output), "tasks": len(catalog)},
                   upstream={"run": str(run)})
    return output
