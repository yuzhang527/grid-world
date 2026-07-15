from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import yaml

def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {p}")
    return data

def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

def config_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
