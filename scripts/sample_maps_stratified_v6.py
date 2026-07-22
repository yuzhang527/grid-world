#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

DIFF_ORDER = ("easy", "medium", "hard")
DIFF_WEIGHTS = {"easy": 0.30, "medium": 0.45, "hard": 0.25}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def normalize_difficulty(row: dict[str, Any]) -> str:
    candidates = [
        row.get("difficulty"),
        row.get("difficulty_label"),
        row.get("difficulty_level"),
        (row.get("metadata") or {}).get("difficulty") if isinstance(row.get("metadata"), dict) else None,
    ]
    for value in candidates:
        if value is None:
            continue
        text = str(value).strip().lower()
        aliases = {
            "e": "easy", "easy": "easy", "0": "easy",
            "m": "medium", "med": "medium", "medium": "medium", "1": "medium",
            "h": "hard", "hard": "hard", "2": "hard",
        }
        if text in aliases:
            return aliases[text]
    raise RuntimeError(
        "Map has no recognized difficulty. Expected one of difficulty, "
        f"difficulty_label, difficulty_level. map_id={row.get('map_id') or row.get('id')}"
    )


def as_xy(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        try:
            return int(value["x"]), int(value["y"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def direction_bucket(row: dict[str, Any]) -> str:
    for key in ("direction", "direction_bucket", "quadrant", "goal_direction"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().upper()
    start = as_xy(row.get("start") or row.get("start_pos") or row.get("source"))
    goal = as_xy(row.get("goal") or row.get("goal_pos") or row.get("target"))
    if start is None or goal is None:
        return "UNKNOWN"
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    sx = "E" if dx > 0 else "W" if dx < 0 else ""
    sy = "N" if dy > 0 else "S" if dy < 0 else ""
    return sy + sx if sy + sx else "SAME"


def largest_remainder(total: int, weights: dict[str, float], capacities: dict[str, int]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    active = {k: float(v) for k, v in weights.items() if capacities.get(k, 0) > 0 and v > 0}
    if not active and total:
        raise RuntimeError("No non-empty strata available")
    weight_sum = sum(active.values()) or 1.0
    raw = {k: total * v / weight_sum for k, v in active.items()}
    allocation = {k: min(capacities[k], int(math.floor(raw[k]))) for k in active}
    remaining = total - sum(allocation.values())
    while remaining > 0:
        candidates = [
            k for k in active
            if allocation[k] < capacities[k]
        ]
        if not candidates:
            raise RuntimeError(f"Insufficient capacity: requested {total}, capacities={capacities}")
        candidates.sort(key=lambda k: (raw[k] - math.floor(raw[k]), active[k], k), reverse=True)
        for key in candidates:
            if remaining <= 0:
                break
            if allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
    return {k: allocation.get(k, 0) for k in capacities}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically sample a difficulty-stratified v6 map subset.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    rows = read_jsonl(args.source)
    if args.count <= 0 or args.count > len(rows):
        raise RuntimeError(f"Invalid count={args.count}; source rows={len(rows)}")

    by_diff: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_diff[normalize_difficulty(row)].append(row)

    capacities = {d: len(by_diff[d]) for d in DIFF_ORDER}
    quotas = largest_remainder(args.count, DIFF_WEIGHTS, capacities)
    # For 100 episodes this is a hard contract inherited from the 1000-map 300/450/250 split.
    if args.count == 100 and quotas != {"easy": 30, "medium": 45, "hard": 25}:
        raise RuntimeError(f"Unexpected v6 quotas: {quotas}")

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    per_direction: dict[str, dict[str, int]] = {}

    for difficulty in DIFF_ORDER:
        bucket = by_diff[difficulty]
        by_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in bucket:
            by_dir[direction_bucket(row)].append(row)
        for values in by_dir.values():
            rng.shuffle(values)
        dir_cap = {key: len(value) for key, value in by_dir.items()}
        dir_weights = {key: len(value) for key, value in by_dir.items()}
        dir_quota = largest_remainder(quotas[difficulty], dir_weights, dir_cap)
        per_direction[difficulty] = dict(sorted(dir_quota.items()))
        for direction, count in sorted(dir_quota.items()):
            selected.extend(by_dir[direction][:count])

    rng.shuffle(selected)
    if len(selected) != args.count:
        raise RuntimeError(f"Selected {len(selected)} rows, expected {args.count}")

    # Reject duplicate stable IDs when IDs are present.
    ids = [str(r.get("map_id") or r.get("id") or "") for r in selected]
    nonempty = [x for x in ids if x]
    if len(nonempty) != len(set(nonempty)):
        raise RuntimeError("Duplicate map IDs in sampled subset")

    write_jsonl(args.output, selected)
    summary_path = args.summary or args.output.with_suffix(".sample_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "coordbelief-v6-sample-1.0",
        "source": str(args.source),
        "output": str(args.output),
        "seed": args.seed,
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "source_difficulty": dict(Counter(normalize_difficulty(r) for r in rows)),
        "sample_difficulty": dict(Counter(normalize_difficulty(r) for r in selected)),
        "sample_direction_within_difficulty": per_direction,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[v6-sample] saved={args.output}")
    print(f"[v6-sample] summary={summary_path}")
    print(f"[v6-sample] difficulty={summary['sample_difficulty']}")


if __name__ == "__main__":
    main()
