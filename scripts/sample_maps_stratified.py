#!/usr/bin/env python3
"""
Deterministically sample N maps from a larger JSONL pool while preserving the
difficulty distribution. Within each difficulty bucket, preserve direction
distribution when a direction field is available.

This script does not modify the selected map records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DIFFICULTY_PATHS = (
    "difficulty",
    "metadata.difficulty",
    "map_metadata.difficulty",
    "generation_metadata.difficulty",
)

DIRECTION_PATHS = (
    "direction_class",
    "direction",
    "metadata.direction_class",
    "metadata.direction",
    "map_metadata.direction_class",
    "map_metadata.direction",
    "generation_metadata.direction_class",
    "generation_metadata.direction",
)


def get_path(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def first_present_path(rows: list[dict[str, Any]], candidates: Iterable[str]) -> str | None:
    for path in candidates:
        if any(get_path(row, path) is not None for row in rows):
            return path
    return None


def largest_remainder_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate exactly total samples proportionally, respecting capacity."""
    if total < 0:
        raise ValueError("total must be non-negative")
    available = sum(counts.values())
    if total > available:
        raise ValueError(f"Requested {total} rows, but only {available} are available")
    if total == 0:
        return {key: 0 for key in counts}
    if available == 0:
        raise ValueError("Cannot allocate from empty buckets")

    exact = {key: total * count / available for key, count in counts.items()}
    quotas = {key: min(counts[key], math.floor(value)) for key, value in exact.items()}
    remaining = total - sum(quotas.values())

    order = sorted(
        counts,
        key=lambda key: (
            exact[key] - math.floor(exact[key]),
            counts[key] - quotas[key],
            key,
        ),
        reverse=True,
    )

    while remaining:
        progressed = False
        for key in order:
            if quotas[key] < counts[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Unable to allocate all requested samples")
    return quotas


def stable_row_key(row: dict[str, Any]) -> str:
    for key in ("map_id", "episode_id", "id", "base_map_id"):
        value = row.get(key)
        if value is not None:
            return f"{key}:{value}"
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def sample_bucket(
    rows: list[dict[str, Any]],
    quota: int,
    rng: random.Random,
    direction_path: str | None,
) -> list[dict[str, Any]]:
    if quota > len(rows):
        raise ValueError(f"Bucket quota {quota} exceeds bucket size {len(rows)}")
    if quota == len(rows):
        chosen = list(rows)
        rng.shuffle(chosen)
        return chosen
    if not direction_path:
        return rng.sample(rows, quota)

    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[dict[str, Any]] = []
    for row in rows:
        value = get_path(row, direction_path)
        if value is None:
            missing.append(row)
        else:
            by_direction[str(value)].append(row)

    if missing:
        by_direction["__MISSING__"].extend(missing)

    counts = {key: len(value) for key, value in by_direction.items()}
    direction_quotas = largest_remainder_quotas(counts, quota)

    chosen: list[dict[str, Any]] = []
    for key in sorted(by_direction):
        group = by_direction[key]
        q = direction_quotas[key]
        chosen.extend(rng.sample(group, q) if q < len(group) else list(group))
    rng.shuffle(chosen)
    return chosen


def summarize(
    rows: list[dict[str, Any]],
    difficulty_path: str,
    direction_path: str | None,
) -> dict[str, Any]:
    difficulty = Counter(str(get_path(row, difficulty_path)) for row in rows)
    direction = (
        Counter(str(get_path(row, direction_path)) for row in rows)
        if direction_path
        else Counter()
    )
    cross = Counter()
    if direction_path:
        for row in rows:
            cross[
                (
                    str(get_path(row, difficulty_path)),
                    str(get_path(row, direction_path)),
                )
            ] += 1

    return {
        "rows": len(rows),
        "difficulty_path": difficulty_path,
        "direction_path": direction_path,
        "difficulty": dict(sorted(difficulty.items())),
        "direction": dict(sorted(direction.items())),
        "difficulty_x_direction": {
            f"{difficulty_value}|{direction_value}": count
            for (difficulty_value, direction_value), count in sorted(cross.items())
        },
        "unique_row_keys": len({stable_row_key(row) for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--difficulty-path", default="auto")
    parser.add_argument("--direction-path", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.n <= 0:
        raise SystemExit("--n must be positive")
    if not args.input.is_file():
        raise SystemExit(f"Missing input map pool: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {args.output}\n"
            "Use --overwrite to regenerate it deterministically."
        )

    rows = load_jsonl(args.input)
    if args.n > len(rows):
        raise SystemExit(f"Requested {args.n} maps from a pool of only {len(rows)}")

    difficulty_path = (
        first_present_path(rows, DIFFICULTY_PATHS)
        if args.difficulty_path == "auto"
        else args.difficulty_path
    )
    if not difficulty_path:
        raise SystemExit(
            "Could not find a difficulty field. Tried: "
            + ", ".join(DIFFICULTY_PATHS)
        )

    if args.direction_path == "none":
        direction_path = None
    elif args.direction_path == "auto":
        direction_path = first_present_path(rows, DIRECTION_PATHS)
    else:
        direction_path = args.direction_path

    missing_difficulty = [
        index for index, row in enumerate(rows) if get_path(row, difficulty_path) is None
    ]
    if missing_difficulty:
        raise SystemExit(
            f"{len(missing_difficulty)} rows lack difficulty field {difficulty_path!r}; "
            f"first indices={missing_difficulty[:10]}"
        )

    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_difficulty[str(get_path(row, difficulty_path))].append(row)

    source_counts = {key: len(value) for key, value in by_difficulty.items()}
    quotas = largest_remainder_quotas(source_counts, args.n)

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for difficulty in sorted(by_difficulty):
        selected.extend(
            sample_bucket(
                by_difficulty[difficulty],
                quotas[difficulty],
                rng,
                direction_path,
            )
        )

    if len(selected) != args.n:
        raise RuntimeError(f"Internal error: selected {len(selected)} != requested {args.n}")
    keys = [stable_row_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Selected sample contains duplicate map records")

    rng.shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_summary = summarize(rows, difficulty_path, direction_path)
    sample_summary = summarize(selected, difficulty_path, direction_path)
    summary = {
        "schema_version": "stratified-map-sample-v1",
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "requested_n": args.n,
        "difficulty_quotas": dict(sorted(quotas.items())),
        "source": source_summary,
        "sample": sample_summary,
    }
    summary_path = args.output.with_suffix(".sample_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("[stratified-sample] source=", args.input)
    print("[stratified-sample] output=", args.output)
    print("[stratified-sample] seed=", args.seed)
    print("[stratified-sample] difficulty_path=", difficulty_path)
    print("[stratified-sample] direction_path=", direction_path)
    print("[stratified-sample] source_difficulty=", source_summary["difficulty"])
    print("[stratified-sample] quotas=", dict(sorted(quotas.items())))
    print("[stratified-sample] sample_difficulty=", sample_summary["difficulty"])
    if direction_path:
        print("[stratified-sample] sample_direction=", sample_summary["direction"])
    print("[stratified-sample] summary=", summary_path)


if __name__ == "__main__":
    main()

