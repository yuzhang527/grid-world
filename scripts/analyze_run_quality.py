#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def direction_class(start: list[int], goal: list[int]) -> str:
    dx = int(goal[0]) - int(start[0])
    dy = int(goal[1]) - int(start[1])
    vertical = "N" if dy > 0 else "S" if dy < 0 else ""
    horizontal = "E" if dx > 0 else "W" if dx < 0 else ""
    return vertical + horizontal or "SAME"


def manhattan(start: list[int], goal: list[int]) -> int:
    return abs(int(goal[0]) - int(start[0])) + abs(int(goal[1]) - int(start[1]))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return centre - radius, centre + radius


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def aggregate(frame: pd.DataFrame, group_column: str | None) -> pd.DataFrame:
    groups = [("overall", frame)] if group_column is None else list(
        frame.groupby(group_column, dropna=False)
    )
    rows = []
    for group_name, group in groups:
        episodes = len(group)
        successes = int(group["success"].sum())
        lower, upper = wilson_interval(successes, episodes)
        successful = group[group["success"].astype(bool)]
        rows.append(
            {
                "group": str(group_name),
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / episodes if episodes else float("nan"),
                "success_ci95_low": lower,
                "success_ci95_high": upper,
                "mean_episode_steps": safe_mean(group["steps"]),
                "mean_steps_success_only": safe_mean(successful["steps"]),
                "mean_optimality_gap_success_only": safe_mean(successful["optimality_gap"]),
                "mean_parse_error_steps": safe_mean(group["parse_error_steps"]),
                "mean_repaired_steps": safe_mean(group["repaired_steps"]),
                "mean_illegal_action_steps": safe_mean(group["illegal_action_steps"]),
                "episodes_with_parse_error_rate": float((group["parse_error_steps"] > 0).mean()),
                "episodes_with_repair_rate": float((group["repaired_steps"] > 0).mean()),
                "episodes_with_illegal_action_rate": float(
                    (group["illegal_action_steps"] > 0).mean()
                ),
                "mean_activation_eligible_fraction": safe_mean(
                    group["activation_eligible_fraction"]
                ),
                "mean_legal_activation_fraction": safe_mean(
                    group["legal_activation_fraction"]
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    return frame[available].to_markdown(index=False, floatfmt=".3f")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratify behavior by difficulty, direction, and output quality."
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run = args.run.resolve()
    output = (args.output or run / "analysis" / "behavior_quality").resolve()
    output.mkdir(parents=True, exist_ok=True)

    required = [run / "maps.jsonl", run / "episodes.jsonl", run / "steps.jsonl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing run artifacts:\n" + "\n".join(missing))

    maps = {str(row["episode_id"]): row for row in read_jsonl(run / "maps.jsonl")}
    episodes = {str(row["episode_id"]): row for row in read_jsonl(run / "episodes.jsonl")}
    steps = read_jsonl(run / "steps.jsonl")

    step_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "step_rows": 0,
            "parse_error_steps": 0,
            "repaired_steps": 0,
            "illegal_action_steps": 0,
            "invalid_move_steps": 0,
            "activation_eligible_steps": 0,
            "legal_activation_steps": 0,
        }
    )

    for step in steps:
        episode_id = str(step["episode_id"])
        stats = step_stats[episode_id]
        stats["step_rows"] += 1
        parse_error = bool(step.get("parse_error"))
        repaired = bool(step.get("repaired"))
        illegal = bool(step.get("illegal_action_before_fallback"))
        stats["parse_error_steps"] += int(parse_error)
        stats["repaired_steps"] += int(repaired)
        stats["illegal_action_steps"] += int(illegal)
        stats["invalid_move_steps"] += int(bool(step.get("invalid_move")))
        # The default activation extractor excludes parse-error and repaired rows.
        activation_eligible = not parse_error and not repaired
        stats["activation_eligible_steps"] += int(activation_eligible)
        stats["legal_activation_steps"] += int(activation_eligible and not illegal)

    episode_rows = []
    for episode_id, episode in episodes.items():
        map_row = maps.get(episode_id, {})
        start = map_row.get("start", episode.get("start", [0, 0]))
        goal = map_row.get("goal", episode.get("goal", [0, 0]))
        shortest = episode.get(
            "shortest_path_length", map_row.get("shortest_path_length")
        )
        manhattan_distance = int(
            map_row.get("manhattan_distance", manhattan(start, goal))
        )
        detour = map_row.get("detour")
        if detour is None and shortest is not None:
            detour = int(shortest) - manhattan_distance
        stats = step_stats[episode_id]
        total_steps = max(1, stats["step_rows"])

        if stats["parse_error_steps"] > 0:
            quality_category = "unrepaired_parse_error"
        elif stats["illegal_action_steps"] > 0:
            quality_category = "illegal_action_fallback"
        elif stats["repaired_steps"] > 0:
            quality_category = "repaired_only"
        else:
            quality_category = "clean_episode"

        episode_rows.append(
            {
                "episode_id": episode_id,
                "success": int(bool(episode.get("success"))),
                "steps": int(episode.get("steps", stats["step_rows"])),
                "shortest_path_length": shortest,
                "optimality_gap": episode.get("optimality_gap"),
                "difficulty": map_row.get("difficulty", "unknown"),
                "direction_class": map_row.get(
                    "direction_class", direction_class(start, goal)
                ),
                "obstacle_count": int(
                    map_row.get(
                        "obstacle_count", len(map_row.get("obstacles", []))
                    )
                ),
                "manhattan_distance": manhattan_distance,
                "detour": detour,
                "quality_category": quality_category,
                **stats,
                "activation_eligible_fraction": (
                    stats["activation_eligible_steps"] / total_steps
                ),
                "legal_activation_fraction": (
                    stats["legal_activation_steps"] / total_steps
                ),
            }
        )

    frame = pd.DataFrame(episode_rows)
    frame.to_csv(output / "episode_quality.csv", index=False)

    aggregate_specs = {
        "overall": None,
        "by_difficulty": "difficulty",
        "by_direction": "direction_class",
        "by_obstacle_count": "obstacle_count",
        "by_detour": "detour",
        "by_shortest_path_length": "shortest_path_length",
        "by_output_quality": "quality_category",
    }
    aggregated: dict[str, pd.DataFrame] = {}
    for name, column in aggregate_specs.items():
        result = aggregate(frame, column)
        result.to_csv(output / f"{name}.csv", index=False)
        aggregated[name] = result

    overall = aggregated["overall"].iloc[0]
    clean = frame[frame["quality_category"] == "clean_episode"]
    non_clean = frame[frame["quality_category"] != "clean_episode"]

    def success_rate(group: pd.DataFrame) -> float:
        return float(group["success"].mean()) if len(group) else float("nan")

    report = [
        "# Behavior and output-quality analysis",
        "",
        f"- Run: `{run}`",
        f"- Episodes: **{len(frame)}**",
        f"- Overall success rate: **{overall['success_rate']:.3f}** "
        f"(95% CI {overall['success_ci95_low']:.3f}–"
        f"{overall['success_ci95_high']:.3f})",
        f"- Clean episodes: **{len(clean)}**; success rate "
        f"**{success_rate(clean):.3f}**",
        f"- Episodes with any repair, parse error, or illegal-action fallback: "
        f"**{len(non_clean)}**; success rate **{success_rate(non_clean):.3f}**",
        f"- Mean fraction of steps eligible for the existing activation extractor: "
        f"**{overall['mean_activation_eligible_fraction']:.3f}**",
        f"- Mean fraction of steps both activation-eligible and free of action fallback: "
        f"**{overall['mean_legal_activation_fraction']:.3f}**",
        "",
        "Important: clean/non-clean comparisons are observational. Difficult maps "
        "may cause both more output failures and lower navigation success.",
        "",
        "## By difficulty",
        "",
        markdown_table(
            aggregated["by_difficulty"],
            [
                "group",
                "episodes",
                "success_rate",
                "success_ci95_low",
                "success_ci95_high",
                "mean_steps_success_only",
                "mean_optimality_gap_success_only",
                "episodes_with_repair_rate",
                "episodes_with_illegal_action_rate",
            ],
        ),
        "",
        "## By goal direction",
        "",
        markdown_table(
            aggregated["by_direction"],
            [
                "group",
                "episodes",
                "success_rate",
                "success_ci95_low",
                "success_ci95_high",
                "mean_optimality_gap_success_only",
                "episodes_with_illegal_action_rate",
            ],
        ),
        "",
        "## By output-quality category",
        "",
        markdown_table(
            aggregated["by_output_quality"],
            [
                "group",
                "episodes",
                "success_rate",
                "success_ci95_low",
                "success_ci95_high",
                "mean_optimality_gap_success_only",
                "mean_parse_error_steps",
                "mean_repaired_steps",
                "mean_illegal_action_steps",
            ],
        ),
        "",
        "## Files",
        "",
        "- `episode_quality.csv`: one row per episode.",
        "- `by_difficulty.csv`, `by_direction.csv`, `by_detour.csv`: "
        "stratified summaries.",
        "- `by_output_quality.csv`: clean, repaired, illegal-fallback, "
        "and parse-error groups.",
    ]
    (output / "summary.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(f"Wrote behavior analysis to: {output}")


if __name__ == "__main__":
    main()
