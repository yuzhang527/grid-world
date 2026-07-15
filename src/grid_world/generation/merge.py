from __future__ import annotations
from pathlib import Path
from grid_world.generation.runner import summarize_episode_rows
from grid_world.utils.io import read_jsonl, write_json, write_jsonl

def merge_run(run_dir: str | Path):
    run = Path(run_dir)
    step_sources, episode_sources = [], []
    if (run / "steps.jsonl").exists():
        step_sources.append(run / "steps.jsonl")
    if (run / "episodes.jsonl").exists():
        episode_sources.append(run / "episodes.jsonl")
    if (run / "shards").exists():
        for shard in sorted((run / "shards").glob("shard_*")):
            if (shard / "steps.jsonl").exists():
                step_sources.append(shard / "steps.jsonl")
            if (shard / "episodes.jsonl").exists():
                episode_sources.append(shard / "episodes.jsonl")
    step_by_key, episode_by_id = {}, {}
    for source in step_sources:
        for row in read_jsonl(source):
            step_by_key[(str(row["episode_id"]), int(row["step_id"]))] = row
    for source in episode_sources:
        for row in read_jsonl(source):
            episode_by_id[str(row["episode_id"])] = row
    steps = [step_by_key[k] for k in sorted(step_by_key)]
    episodes = [episode_by_id[k] for k in sorted(episode_by_id)]
    write_jsonl(run / "steps.jsonl", steps)
    write_jsonl(run / "episodes.jsonl", episodes)
    write_json(run / "summary.json", summarize_episode_rows(episodes, steps))
    return steps, episodes
