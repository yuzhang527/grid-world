#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_best(run: Path) -> pd.DataFrame:
    path = run / "probes" / "best_by_task.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    wanted = [
        "task_group",
        "task",
        "position",
        "layer",
        "macro_f1_mean",
        "macro_f1_std",
        "majority_macro_f1_mean",
    ]
    return frame[wanted].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the exploratory best-by-task probe reports of two runs."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    baseline_run = Path(args.baseline)
    candidate_run = Path(args.candidate)
    output = (
        Path(args.output_dir)
        if args.output_dir
        else candidate_run / "probes" / "comparison"
    )
    output.mkdir(parents=True, exist_ok=True)

    baseline = load_best(baseline_run).add_prefix("baseline_")
    candidate = load_best(candidate_run).add_prefix("candidate_")
    merged = baseline.merge(
        candidate,
        left_on="baseline_task",
        right_on="candidate_task",
        how="outer",
    )
    merged["task"] = merged["candidate_task"].fillna(merged["baseline_task"])
    merged["task_group"] = merged["candidate_task_group"].fillna(
        merged["baseline_task_group"]
    )
    merged["macro_f1_delta"] = (
        merged["candidate_macro_f1_mean"]
        - merged["baseline_macro_f1_mean"]
    )
    merged["candidate_selectivity_over_majority"] = (
        merged["candidate_macro_f1_mean"]
        - merged["candidate_majority_macro_f1_mean"]
    )
    merged["baseline_selectivity_over_majority"] = (
        merged["baseline_macro_f1_mean"]
        - merged["baseline_majority_macro_f1_mean"]
    )

    columns = [
        "task_group",
        "task",
        "baseline_position",
        "baseline_layer",
        "baseline_macro_f1_mean",
        "baseline_macro_f1_std",
        "baseline_majority_macro_f1_mean",
        "candidate_position",
        "candidate_layer",
        "candidate_macro_f1_mean",
        "candidate_macro_f1_std",
        "candidate_majority_macro_f1_mean",
        "macro_f1_delta",
        "baseline_selectivity_over_majority",
        "candidate_selectivity_over_majority",
    ]
    merged = merged[columns].sort_values(["task_group", "task"])
    merged.to_csv(output / "best_by_task_comparison.csv", index=False)

    shared = merged.dropna(
        subset=["baseline_macro_f1_mean", "candidate_macro_f1_mean"]
    )
    group_summary = (
        shared.groupby("task_group")
        .agg(
            shared_tasks=("task", "count"),
            baseline_mean_best_macro_f1=("baseline_macro_f1_mean", "mean"),
            candidate_mean_best_macro_f1=("candidate_macro_f1_mean", "mean"),
            mean_macro_f1_delta=("macro_f1_delta", "mean"),
            baseline_mean_std=("baseline_macro_f1_std", "mean"),
            candidate_mean_std=("candidate_macro_f1_std", "mean"),
            baseline_mean_selectivity=(
                "baseline_selectivity_over_majority",
                "mean",
            ),
            candidate_mean_selectivity=(
                "candidate_selectivity_over_majority",
                "mean",
            ),
        )
        .reset_index()
    )
    group_summary.to_csv(output / "group_comparison.csv", index=False)

    lines = [
        "# Probe Run Comparison",
        "",
        f"- Baseline: `{baseline_run}`",
        f"- Candidate: `{candidate_run}`",
        "",
        "> These values compare the best layer/position selected per task and are "
        "exploratory; they include model-selection optimism.",
        "",
        "## Group comparison",
        "",
        group_summary.to_markdown(index=False),
        "",
        "## Largest candidate improvements",
        "",
        shared.nlargest(15, "macro_f1_delta")[
            [
                "task_group",
                "task",
                "baseline_macro_f1_mean",
                "candidate_macro_f1_mean",
                "macro_f1_delta",
            ]
        ].to_markdown(index=False),
        "",
        "## Largest candidate decreases",
        "",
        shared.nsmallest(15, "macro_f1_delta")[
            [
                "task_group",
                "task",
                "baseline_macro_f1_mean",
                "candidate_macro_f1_mean",
                "macro_f1_delta",
            ]
        ].to_markdown(index=False),
        "",
    ]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "comparison.md")


if __name__ == "__main__":
    main()
