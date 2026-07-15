from __future__ import annotations
from pathlib import Path
from grid_world.utils.io import read_json

def load_catalog(run_dir):
    return read_json(Path(run_dir) / "targets" / "task_catalog.json")

def select_tasks(catalog, groups: str):
    wanted = {x.strip() for x in groups.split(",") if x.strip()}
    result = sorted(name for name, meta in catalog.items() if meta.get("group") in wanted)
    if not result:
        raise ValueError(f"No tasks for groups {wanted}")
    return result
