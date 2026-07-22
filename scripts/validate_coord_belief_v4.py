#!/usr/bin/env python3
"""Strict checker for Coordinate-Belief v4 runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--strict-raw-format", action="store_true")
    args = parser.parse_args()

    run = Path(args.run)
    steps = read_jsonl(run / "steps.jsonl")
    episodes = read_jsonl(run / "episodes.jsonl")
    errors: list[str] = []
    counts = Counter()

    for index, row in enumerate(steps, 1):
        ep = str(row.get("episode_id"))
        step = row.get("step_id")
        width = int(row.get("width", 5))
        height = int(row.get("height", 5))
        payload = row.get("parsed_belief_coordinates")
        grid = row.get("parsed_belief_grid")
        if not isinstance(payload, dict):
            errors.append(f"{ep}/{step}: missing parsed_belief_coordinates")
            continue
        if set(payload) - {"F", "O"}:
            errors.append(f"{ep}/{step}: coordinate payload has unexpected keys {set(payload)}")
        seen: dict[tuple[int, int], str] = {}
        for label in ("F", "O"):
            values = payload.get(label, [])
            if not isinstance(values, list):
                errors.append(f"{ep}/{step}: {label} is not a list")
                continue
            for item in values:
                if not isinstance(item, list) or len(item) != 2:
                    errors.append(f"{ep}/{step}: invalid coordinate {item!r}")
                    continue
                coord = (int(item[0]), int(item[1]))
                if not (0 <= coord[0] < width and 0 <= coord[1] < height):
                    errors.append(f"{ep}/{step}: out-of-bounds coordinate {coord}")
                if coord in seen and seen[coord] != label:
                    errors.append(f"{ep}/{step}: overlap at {coord}")
                seen[coord] = label
        if not isinstance(grid, list) or len(grid) != height or any(
            not isinstance(line, list) or len(line) != width for line in grid
        ):
            errors.append(f"{ep}/{step}: parsed_belief_grid has wrong shape")
        else:
            for y in range(height):
                for x in range(width):
                    expected = seen.get((x, y), "U")
                    if grid[y][x] != expected:
                        errors.append(
                            f"{ep}/{step}: grid mismatch at {(x,y)} "
                            f"grid={grid[y][x]} coordinates={expected}"
                        )
        raw = str(row.get("raw_response", ""))
        if args.strict_raw_format and '"belief_grid"' in raw:
            errors.append(f"{ep}/{step}: raw response used forbidden belief_grid matrix key")
        counts["parse_error"] += bool(row.get("parse_error"))
        counts["repaired"] += bool(row.get("repaired"))
        counts["invalid_move"] += bool(row.get("invalid_move"))

    episode_ids = [str(row.get("episode_id")) for row in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        errors.append("episodes.jsonl contains duplicate episode ids")

    print(f"[coord-v4-check] steps={len(steps)} episodes={len(episodes)}")
    print(
        "[coord-v4-check] "
        f"parse_error={counts['parse_error']} repaired={counts['repaired']} "
        f"invalid_move={counts['invalid_move']}"
    )
    if errors:
        print(f"[coord-v4-check] errors={len(errors)}")
        for error in errors[:50]:
            print("  -", error)
        raise SystemExit(1)
    print("[coord-v4-check] PASS")


if __name__ == "__main__":
    main()

