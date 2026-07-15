from __future__ import annotations
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from grid_world import __version__
from grid_world.config import config_hash
from grid_world.utils.io import write_json

def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

def write_manifest(path: str | Path, *, stage: str, config: dict[str, Any],
                   counts: dict[str, int] | None = None,
                   upstream: dict[str, str] | None = None,
                   extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "package_version": __version__,
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(config),
        "git_commit": git_commit(),
        "config": config,
        "counts": counts or {},
        "upstream": upstream or {},
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)
