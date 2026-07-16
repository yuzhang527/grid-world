from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grid_world.env.belief import (
    initial_belief,
    rows_to_belief,
    update_belief,
)
from grid_world.env.grid import GridSpec, GridWorld
from grid_world.generation.backends import GenerationBackend, MockOracleBackend
from grid_world.prompting.parser import parse_response, repair_messages
from grid_world.prompting.strategy_a import (
    build_messages as build_explicit_belief_messages,
)
from grid_world.prompting.strategy_no_grid import (
    build_messages as build_no_grid_messages,
)
from grid_world.utils.io import write_json, write_jsonl
from grid_world.utils.manifest import write_manifest


EXPLICIT_BELIEF_MODE = "explicit_belief"
NO_GRID_MODE = "no_grid"


@dataclass
class EpisodeState:
    spec: GridSpec
    env: GridWorld
    model_belief: dict
    gold_belief: dict
    history: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    map_metadata: dict[str, Any] = field(default_factory=dict)


def _prompt_mode(config: dict[str, Any]) -> str:
    value = str(
        config.get("experiment", {}).get(
            "prompt_mode",
            EXPLICIT_BELIEF_MODE,
        )
    ).strip().lower()
    aliases = {
        "strategy_a": EXPLICIT_BELIEF_MODE,
        "explicit": EXPLICIT_BELIEF_MODE,
        "belief": EXPLICIT_BELIEF_MODE,
        "no-grid": NO_GRID_MODE,
        "nogrid": NO_GRID_MODE,
    }
    value = aliases.get(value, value)
    if value not in {EXPLICIT_BELIEF_MODE, NO_GRID_MODE}:
        raise ValueError(
            "experiment.prompt_mode must be explicit_belief or no_grid"
        )
    return value


def summarize_episode_rows(
    episodes: list[dict],
    steps: list[dict],
) -> dict:
    successes = [row for row in episodes if row.get("success")]
    gaps = [
        float(row["optimality_gap"])
        for row in successes
        if row.get("optimality_gap") is not None
    ]
    return {
        "schema_version": "1.0",
        "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": (
            len(successes) / len(episodes) if episodes else 0.0
        ),
        "mean_steps_success_only": (
            sum(int(row["steps"]) for row in successes) / len(successes)
            if successes
            else None
        ),
        "mean_optimality_gap_success_only": (
            sum(gaps) / len(gaps) if gaps else None
        ),
        "total_parse_errors": sum(
            bool(row.get("parse_error")) for row in steps
        ),
        "total_repaired": sum(bool(row.get("repaired")) for row in steps),
        "total_illegal_action_before_fallback": sum(
            bool(row.get("illegal_action_before_fallback"))
            for row in steps
        ),
        "total_invalid_moves": sum(
            bool(row.get("invalid_move")) for row in steps
        ),
    }


def run_episodes(
    *,
    map_rows: list[dict[str, Any]],
    output_dir: str | Path,
    config: dict[str, Any],
    backend: GenerationBackend,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    max_steps = int(config.get("experiment", {}).get("max_steps", 20))
    repair_enabled = bool(
        config.get("generation", {}).get("repair_invalid_json", True)
    )
    prompt_mode = _prompt_mode(config)
    require_belief_grid = prompt_mode == EXPLICIT_BELIEF_MODE
    build_messages = (
        build_explicit_belief_messages
        if require_belief_grid
        else build_no_grid_messages
    )

    states = []
    for row in map_rows:
        spec = GridSpec.from_dict(row)
        metadata_keys = [
            "generation_mode",
            "dataset_seed",
            "direction_class",
            "difficulty",
            "obstacle_count",
            "manhattan_distance",
            "detour",
            "generation_attempts",
        ]
        map_metadata = {
            key: row[key]
            for key in metadata_keys
            if row.get(key) is not None
        }
        states.append(
            EpisodeState(
                spec=spec,
                env=GridWorld(spec),
                model_belief=initial_belief(spec),
                gold_belief=initial_belief(spec),
                map_metadata=map_metadata,
            )
        )

    step_rows = []
    for step_id in range(max_steps):
        active = [state for state in states if not state.done]
        if not active:
            break

        prompts = []
        prepared = []
        contexts = []

        for state in active:
            position = state.env.position
            feedback = state.env.feedback()
            state.gold_belief = update_belief(
                state.gold_belief,
                feedback,
            )
            available = state.env.available_actions()
            messages = build_messages(
                spec=state.spec,
                position=position,
                feedback=feedback,
                model_belief=state.model_belief,
                history=state.history,
                available_actions=available,
            )
            prompt_text = backend.render(messages)
            prompts.append(prompt_text)
            prepared.append(
                (
                    state,
                    position,
                    feedback,
                    available,
                    messages,
                    prompt_text,
                )
            )
            contexts.append(
                {
                    "spec": state.spec,
                    "position": position,
                    "gold_belief": state.gold_belief,
                    "available_actions": available,
                }
            )

        if isinstance(backend, MockOracleBackend):
            backend.set_contexts(contexts)

        raw_outputs = backend.generate(prompts)
        parsed = [
            parse_response(
                raw,
                item[0].spec.size,
                require_belief_grid=require_belief_grid,
            )
            for raw, item in zip(raw_outputs, prepared)
        ]
        repaired_flags = [False] * len(parsed)
        invalid = [
            index
            for index, result in enumerate(parsed)
            if result.data is None
        ]

        if (
            invalid
            and repair_enabled
            and not isinstance(backend, MockOracleBackend)
        ):
            repair_prompts = [
                backend.render(
                    repair_messages(
                        raw_outputs[index],
                        prepared[index][0].spec.size,
                        require_belief_grid=require_belief_grid,
                    )
                )
                for index in invalid
            ]
            repaired_outputs = backend.generate(repair_prompts)
            for index, repaired_text in zip(invalid, repaired_outputs):
                candidate = parse_response(
                    repaired_text,
                    prepared[index][0].spec.size,
                    require_belief_grid=require_belief_grid,
                )
                if candidate.data is not None:
                    raw_outputs[index] = repaired_text
                    parsed[index] = candidate
                    repaired_flags[index] = True

        for index, (
            (
                state,
                position,
                feedback,
                available,
                messages,
                prompt_text,
            ),
            raw,
            result,
        ) in enumerate(zip(prepared, raw_outputs, parsed)):
            data = result.data or {}
            requested = data.get("action")
            illegal = requested not in available
            action = requested if requested in available else available[0]
            next_position, valid = state.env.step(action)

            if (
                require_belief_grid
                and result.data is not None
                and result.data.get("belief_grid") is not None
            ):
                state.model_belief = rows_to_belief(
                    result.data["belief_grid"],
                    state.spec.size,
                )

            history_item = {
                "step_id": step_id,
                "position": list(position),
                "feedback": feedback,
                "available_actions": available,
                "requested_action": requested,
                "action": action,
                "next_position": list(next_position),
            }
            state.history.append(history_item)
            state.done = (
                state.env.reached_goal or step_id + 1 >= max_steps
            )

            step_rows.append(
                {
                    "schema_version": "1.0",
                    "map_metadata": state.map_metadata,
                    **state.map_metadata,
                    "episode_id": state.spec.episode_id,
                    "seed": state.spec.seed,
                    "step_id": step_id,
                    "size": state.spec.size,
                    "start": list(state.spec.start),
                    "goal": list(state.spec.goal),
                    "obstacles": [
                        list(coord)
                        for coord in sorted(state.spec.obstacles)
                    ],
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
                    "prompt_mode": prompt_mode,
                    "parse_error": result.data is None,
                    "parse_error_message": result.error,
                    "repaired": repaired_flags[index],
                    "illegal_action_before_fallback": illegal,
                    "invalid_move": not valid,
                    "reached_goal": state.env.reached_goal,
                    "model_name": config.get("model", {}).get("name"),
                    "backend": config.get("model", {}).get("backend"),
                }
            )

    episodes = []
    for state in states:
        success = state.env.reached_goal
        number_of_steps = len(state.history)
        shortest = state.spec.shortest_path_length
        episodes.append(
            {
                "schema_version": "1.0",
                "map_metadata": state.map_metadata,
                **state.map_metadata,
                "episode_id": state.spec.episode_id,
                "seed": state.spec.seed,
                "size": state.spec.size,
                "start": list(state.spec.start),
                "goal": list(state.spec.goal),
                "obstacles": [
                    list(coord)
                    for coord in sorted(state.spec.obstacles)
                ],
                "prompt_mode": prompt_mode,
                "success": success,
                "steps": number_of_steps,
                "shortest_path_length": shortest,
                "optimality_gap": (
                    number_of_steps - shortest
                    if success and shortest is not None
                    else None
                ),
                "final_position": list(state.env.position),
                "trajectory": state.history,
            }
        )

    step_rows.sort(
        key=lambda row: (row["episode_id"], int(row["step_id"]))
    )
    episodes.sort(key=lambda row: row["episode_id"])
    write_jsonl(out / "steps.jsonl", step_rows)
    write_jsonl(out / "episodes.jsonl", episodes)
    write_json(
        out / "summary.json",
        summarize_episode_rows(episodes, step_rows),
    )
    write_manifest(
        out / "manifest.json",
        stage="trajectories",
        config=config,
        counts={
            "episodes": len(episodes),
            "steps": len(step_rows),
        },
        extra={"prompt_mode": prompt_mode},
    )
    return step_rows, episodes
