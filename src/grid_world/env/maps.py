from __future__ import annotations
import random
from pathlib import Path
from grid_world.config import load_yaml
from grid_world.env.grid import GridSpec, as_coord, render_grid
from grid_world.env.planning import count_shortest_paths, shortest_path_length
from grid_world.utils.io import read_jsonl, write_jsonl
from grid_world.utils.manifest import write_manifest

def generate_maps(config_path: str | Path, output: str | Path) -> list[dict]:
    cfg = load_yaml(config_path)
    size = int(cfg.get("size", 5))
    count = int(cfg.get("num_episodes", 100))
    obstacle_count = int(cfg.get("num_obstacles", 4))
    seed_start = int(cfg.get("seed_start", 123))
    prefix = str(cfg.get("prefix", "A_seed"))
    start = as_coord(cfg.get("start", [0, 0]))
    goal = as_coord(cfg.get("goal", [size - 1, size - 1]))
    unique = bool(cfg.get("require_unique_shortest_path", False))
    max_attempts = int(cfg.get("max_attempts_per_episode", 10000))
    all_cells = [(x, y) for x in range(size) for y in range(size) if (x, y) not in {start, goal}]
    rows = []
    for offset in range(count):
        seed = seed_start + offset
        rng = random.Random(seed)
        accepted = None
        for _ in range(max_attempts):
            obstacles = frozenset(rng.sample(all_cells, obstacle_count))
            spec = GridSpec(f"{prefix}{seed}", seed, size, start, goal, obstacles)
            distance = shortest_path_length(spec)
            if distance is None or (unique and count_shortest_paths(spec) != 1):
                continue
            accepted = GridSpec(spec.episode_id, seed, size, start, goal, obstacles, distance)
            break
        if accepted is None:
            raise RuntimeError(f"Could not generate a valid map for seed {seed}")
        rows.append(accepted.to_dict())
    write_jsonl(output, rows)
    write_manifest(Path(output).with_suffix(".manifest.json"), stage="maps",
                   config=cfg, counts={"episodes": len(rows)})
    return rows

def validate_maps(path: str | Path) -> dict[str, int]:
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        spec = GridSpec.from_dict(row)
        if spec.episode_id in seen:
            raise ValueError(f"Duplicate episode_id: {spec.episode_id}")
        seen.add(spec.episode_id)
        if spec.start in spec.obstacles or spec.goal in spec.obstacles:
            raise ValueError(f"Start/goal blocked in {spec.episode_id}")
        distance = shortest_path_length(spec)
        if distance is None:
            raise ValueError(f"No path in {spec.episode_id}")
        if row.get("shortest_path_length") is not None and int(row["shortest_path_length"]) != distance:
            raise ValueError(f"Shortest-path mismatch in {spec.episode_id}")
    return {"episodes": len(rows)}

def show_map(path: str | Path, episode_id: str | None = None, index: int = 0) -> str:
    rows = read_jsonl(path)
    if episode_id is not None:
        rows = [row for row in rows if str(row["episode_id"]) == episode_id]
        if not rows:
            raise KeyError(f"Episode not found: {episode_id}")
    spec = GridSpec.from_dict(rows[index])
    return f"{spec.episode_id}\n{render_grid(spec)}"
