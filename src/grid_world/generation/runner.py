from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from grid_world.env.belief import initial_belief, rows_to_belief, update_belief
from grid_world.env.grid import GridSpec, GridWorld
from grid_world.generation.backends import GenerationBackend, MockOracleBackend
from grid_world.prompting.parser import parse_response, repair_messages
from grid_world.prompting.strategy_a import build_messages
from grid_world.utils.io import write_json, write_jsonl
from grid_world.utils.manifest import write_manifest

@dataclass
class EpisodeState:
    spec: GridSpec
    env: GridWorld
    model_belief: dict
    gold_belief: dict
    history: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False

def summarize_episode_rows(episodes: list[dict], steps: list[dict]) -> dict:
    successes = [x for x in episodes if x.get("success")]
    gaps = [float(x["optimality_gap"]) for x in successes if x.get("optimality_gap") is not None]
    return {
        "schema_version": "1.0",
        "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes) if episodes else 0.0,
        "mean_steps_success_only": sum(int(x["steps"]) for x in successes) / len(successes) if successes else None,
        "mean_optimality_gap_success_only": sum(gaps) / len(gaps) if gaps else None,
        "total_parse_errors": sum(bool(x.get("parse_error")) for x in steps),
        "total_repaired": sum(bool(x.get("repaired")) for x in steps),
        "total_illegal_action_before_fallback": sum(bool(x.get("illegal_action_before_fallback")) for x in steps),
        "total_invalid_moves": sum(bool(x.get("invalid_move")) for x in steps),
    }

def run_episodes(*, map_rows: list[dict[str, Any]], output_dir: str | Path,
                 config: dict[str, Any], backend: GenerationBackend):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    max_steps = int(config.get("experiment", {}).get("max_steps", 20))
    repair_enabled = bool(config.get("generation", {}).get("repair_invalid_json", True))
    states = []
    for row in map_rows:
        spec = GridSpec.from_dict(row)
        states.append(EpisodeState(spec, GridWorld(spec), initial_belief(spec), initial_belief(spec)))

    step_rows = []
    for step_id in range(max_steps):
        active = [s for s in states if not s.done]
        if not active:
            break
        prompts, prepared, contexts = [], [], []
        for state in active:
            position = state.env.position
            feedback = state.env.feedback()
            state.gold_belief = update_belief(state.gold_belief, feedback)
            available = state.env.available_actions()
            messages = build_messages(
                spec=state.spec, position=position, feedback=feedback,
                model_belief=state.model_belief, history=state.history,
                available_actions=available,
            )
            prompt_text = backend.render(messages)
            prompts.append(prompt_text)
            prepared.append((state, position, feedback, available, messages, prompt_text))
            contexts.append({"spec": state.spec, "position": position,
                             "gold_belief": state.gold_belief,
                             "available_actions": available})
        if isinstance(backend, MockOracleBackend):
            backend.set_contexts(contexts)
        raw_outputs = backend.generate(prompts)
        parsed = [parse_response(raw, item[0].spec.size) for raw, item in zip(raw_outputs, prepared)]
        repaired_flags = [False] * len(parsed)
        invalid = [i for i, result in enumerate(parsed) if result.data is None]
        if invalid and repair_enabled and not isinstance(backend, MockOracleBackend):
            repair_prompts = [backend.render(repair_messages(raw_outputs[i], prepared[i][0].spec.size))
                              for i in invalid]
            repaired_outputs = backend.generate(repair_prompts)
            for idx, repaired_text in zip(invalid, repaired_outputs):
                candidate = parse_response(repaired_text, prepared[idx][0].spec.size)
                if candidate.data is not None:
                    raw_outputs[idx], parsed[idx], repaired_flags[idx] = repaired_text, candidate, True

        for i, ((state, position, feedback, available, messages, prompt_text), raw, result) in enumerate(
            zip(prepared, raw_outputs, parsed)
        ):
            data = result.data or {}
            requested = data.get("action")
            illegal = requested not in available
            action = requested if requested in available else available[0]
            next_position, valid = state.env.step(action)
            if result.data is not None:
                state.model_belief = rows_to_belief(result.data["belief_grid"], state.spec.size)
            item = {"step_id": step_id, "position": list(position),
                    "action": action, "next_position": list(next_position)}
            state.history.append(item)
            state.done = state.env.reached_goal or step_id + 1 >= max_steps
            step_rows.append({
                "schema_version": "1.0",
                "episode_id": state.spec.episode_id,
                "seed": state.spec.seed,
                "step_id": step_id,
                "size": state.spec.size,
                "start": list(state.spec.start),
                "goal": list(state.spec.goal),
                "obstacles": [list(x) for x in sorted(state.spec.obstacles)],
                "current_pos": list(position),
                "next_pos": list(next_position),
                "feedback": feedback,
                "available_actions": available,
                "messages": messages,
                "prompt_text": prompt_text,
                "raw_response": raw,
                "parsed_response": data or None,
                "parsed_belief_grid": data.get("belief_grid"),
                "requested_action": requested,
                "action": action,
                "parse_error": result.data is None,
                "parse_error_message": result.error,
                "repaired": repaired_flags[i],
                "illegal_action_before_fallback": illegal,
                "invalid_move": not valid,
                "reached_goal": state.env.reached_goal,
                "model_name": config.get("model", {}).get("name"),
                "backend": config.get("model", {}).get("backend"),
            })

    episodes = []
    for state in states:
        success, nsteps = state.env.reached_goal, len(state.history)
        shortest = state.spec.shortest_path_length
        episodes.append({
            "schema_version": "1.0",
            "episode_id": state.spec.episode_id,
            "seed": state.spec.seed,
            "size": state.spec.size,
            "start": list(state.spec.start),
            "goal": list(state.spec.goal),
            "obstacles": [list(x) for x in sorted(state.spec.obstacles)],
            "success": success,
            "steps": nsteps,
            "shortest_path_length": shortest,
            "optimality_gap": nsteps - shortest if success and shortest is not None else None,
            "final_position": list(state.env.position),
            "trajectory": state.history,
        })
    step_rows.sort(key=lambda x: (x["episode_id"], int(x["step_id"])))
    episodes.sort(key=lambda x: x["episode_id"])
    write_jsonl(out / "steps.jsonl", step_rows)
    write_jsonl(out / "episodes.jsonl", episodes)
    write_json(out / "summary.json", summarize_episode_rows(episodes, step_rows))
    write_manifest(out / "manifest.json", stage="trajectories", config=config,
                   counts={"episodes": len(episodes), "steps": len(step_rows)})
    return step_rows, episodes
