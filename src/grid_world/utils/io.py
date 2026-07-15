from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path: str | Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {p}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {p}:{line_no}")
            yield value

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))

def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
