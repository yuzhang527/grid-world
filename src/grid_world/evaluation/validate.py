from __future__ import annotations
from collections import Counter
from pathlib import Path
from grid_world.env.grid import GridSpec
from grid_world.utils.io import read_jsonl

def validate_run(run_dir: str | Path) -> dict[str, int]:
    run = Path(run_dir)
    for name in ["maps.jsonl", "steps.jsonl", "episodes.jsonl"]:
        if not (run / name).exists():
            raise FileNotFoundError(run / name)
    maps, steps, episodes = read_jsonl(run / "maps.jsonl"), read_jsonl(run / "steps.jsonl"), read_jsonl(run / "episodes.jsonl")
    map_ids = {str(x["episode_id"]) for x in maps}
    episode_ids = [str(x["episode_id"]) for x in episodes]
    if any(v > 1 for v in Counter(episode_ids).values()):
        raise ValueError("Duplicate episode summaries")
    keys = [(str(x["episode_id"]), int(x["step_id"])) for x in steps]
    if any(v > 1 for v in Counter(keys).values()):
        raise ValueError("Duplicate steps")
    if set(episode_ids) - map_ids:
        raise ValueError("Episodes missing maps")
    for row in maps:
        GridSpec.from_dict(row)
    for row in steps:
        for key in ["episode_id","step_id","current_pos","next_pos","feedback","action"]:
            if key not in row:
                raise ValueError(f"Step missing {key}")
    return {"maps": len(maps), "episodes": len(episodes), "steps": len(steps),
            "parse_errors": sum(bool(x.get("parse_error")) for x in steps),
            "repaired": sum(bool(x.get("repaired")) for x in steps)}
