from collections import Counter

import yaml

from grid_world.env.maps import generate_maps, summarize_maps, validate_maps
from grid_world.utils.io import read_jsonl


def test_diverse_generation_balances_direction_and_difficulty(tmp_path):
    config = {
        "schema_version": "1.0",
        "generation_mode": "diverse",
        "prefix": "test_ep",
        "dataset_seed": 7,
        "seed_start": 100,
        "num_episodes": 40,
        "size": 5,
        "direction_classes": ["NE", "NW", "SE", "SW"],
        "min_manhattan_distance": 4,
        "difficulty_buckets": [
            {
                "name": "easy",
                "weight": 0.5,
                "obstacle_counts": [2, 3],
                "min_detour": 0,
                "max_detour": 0,
            },
            {
                "name": "medium",
                "weight": 0.5,
                "obstacle_counts": [4, 5, 6],
                "min_detour": 2,
                "max_detour": 2,
            },
        ],
        "max_attempts_per_episode": 10000,
    }
    config_path = tmp_path / "maps.yaml"
    output_path = tmp_path / "maps.jsonl"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    rows = generate_maps(config_path, output_path)
    assert len(rows) == 40
    assert validate_maps(output_path) == {
        "episodes": 40,
        "unique_layouts": 40,
    }

    direction_counts = Counter(row["direction_class"] for row in rows)
    difficulty_counts = Counter(row["difficulty"] for row in rows)
    assert direction_counts == {"NE": 10, "NW": 10, "SE": 10, "SW": 10}
    assert difficulty_counts == {"easy": 20, "medium": 20}
    assert all(row["manhattan_distance"] >= 4 for row in rows)
    assert all(
        row["detour"] == (0 if row["difficulty"] == "easy" else 2)
        for row in rows
    )

    summary = summarize_maps(output_path)
    assert summary["maps"] == 40
    assert summary["direction_class"] == {
        "NE": 10,
        "NW": 10,
        "SE": 10,
        "SW": 10,
    }
