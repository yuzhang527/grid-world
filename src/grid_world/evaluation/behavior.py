from __future__ import annotations
from pathlib import Path
from grid_world.generation.runner import summarize_episode_rows
from grid_world.utils.io import read_jsonl, write_json

def summarize_run(run_dir: str | Path) -> dict:
    run = Path(run_dir)
    summary = summarize_episode_rows(read_jsonl(run / "episodes.jsonl"), read_jsonl(run / "steps.jsonl"))
    write_json(run / "summary.json", summary)
    return summary
