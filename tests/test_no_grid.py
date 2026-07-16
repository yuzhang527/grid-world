from pathlib import Path

from grid_world.env.grid import GridSpec
from grid_world.generation.backends import MockOracleBackend
from grid_world.generation.runner import run_episodes
from grid_world.prompting.strategy_no_grid import build_messages
from grid_world.targets.build import build_targets
from grid_world.utils.io import read_jsonl


def test_no_grid_prompt_contains_observations_but_no_explicit_map():
    spec = GridSpec(
        episode_id="example",
        seed=1,
        size=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles=frozenset({(2, 2)}),
        shortest_path_length=8,
    )
    history = [
        {
            "step_id": 0,
            "position": [0, 0],
            "feedback": {
                "free": [[0, 1], [1, 0]],
                "blocked": [],
                "wall": ["DOWN", "LEFT"],
            },
            "action": "UP",
            "next_position": [0, 1],
        }
    ]
    messages = build_messages(
        spec=spec,
        position=(0, 1),
        feedback={
            "free": [[0, 0], [0, 2], [1, 1]],
            "blocked": [],
            "wall": ["LEFT"],
        },
        model_belief={},
        history=history,
        available_actions=["UP", "DOWN", "RIGHT"],
    )
    text = "\n".join(message["content"] for message in messages)
    assert "<history>" in text
    assert '"feedback"' in text
    assert "current_belief_grid" not in text
    assert "required_belief_updates" not in text
    assert '"belief_grid"' not in text


def test_no_grid_mock_run_and_extended_targets(tmp_path: Path):
    map_row = {
        "schema_version": "1.0",
        "episode_id": "mock_no_grid",
        "seed": 7,
        "size": 5,
        "start": [0, 0],
        "goal": [4, 4],
        "obstacles": [[2, 2]],
        "shortest_path_length": 8,
        "difficulty": "easy",
        "direction_class": "NE",
        "obstacle_count": 1,
        "manhattan_distance": 8,
        "detour": 0,
    }
    config = {
        "experiment": {
            "prompt_mode": "no_grid",
            "max_steps": 12,
        },
        "model": {
            "backend": "mock",
            "name": "mock",
        },
        "generation": {
            "repair_invalid_json": False,
        },
    }
    from grid_world.utils.io import write_jsonl

    write_jsonl(tmp_path / "maps.jsonl", [map_row])
    run_episodes(
        map_rows=[map_row],
        output_dir=tmp_path,
        config=config,
        backend=MockOracleBackend(),
    )
    steps = read_jsonl(tmp_path / "steps.jsonl")
    assert steps
    assert all(step["prompt_mode"] == "no_grid" for step in steps)
    assert all(step["parsed_belief_grid"] is None for step in steps)
    assert all(
        "current_belief_grid" not in step["prompt_text"]
        for step in steps
    )

    build_targets(tmp_path)
    targets = read_jsonl(tmp_path / "targets" / "targets.jsonl")
    assert targets
    first = targets[0]
    assert first["model_cell_x0_y0_OFU"] is None
    assert first["true_cell_x2_y2_FO"] == "O"
    assert first["gold_cell_x4_y4_OFU"] in {"F", "U"}
    assert "true_cell_x4_y4_FO_unobserved" in first
