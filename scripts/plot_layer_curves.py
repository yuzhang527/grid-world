#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def aggregate_layers(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["delta_over_majority"] = (
        work["macro_f1_mean"] - work["majority_macro_f1_mean"]
    )
    grouped = (
        work.groupby(
            ["task_group", "position", "layer"],
            dropna=False,
        )
        .agg(
            tasks=("task", "nunique"),
            mean_macro_f1=("macro_f1_mean", "mean"),
            median_macro_f1=("macro_f1_mean", "median"),
            task_std_macro_f1=("macro_f1_mean", "std"),
            mean_split_std=("macro_f1_std", "mean"),
            mean_majority_macro_f1=(
                "majority_macro_f1_mean",
                "mean",
            ),
            mean_delta_over_majority=(
                "delta_over_majority",
                "mean",
            ),
            mean_num_test=("num_test_mean", "mean"),
        )
        .reset_index()
        .sort_values(["task_group", "position", "layer"])
    )
    grouped["task_sem_macro_f1"] = grouped.apply(
        lambda row: (
            row["task_std_macro_f1"] / math.sqrt(row["tasks"])
            if row["tasks"] > 1
            else float("nan")
        ),
        axis=1,
    )
    return grouped


def plot_group_curves(
    summary: pd.DataFrame,
    output: Path,
    condition_label: str,
) -> list[Path]:
    paths = []
    for task_group, group_frame in summary.groupby("task_group"):
        figure, axis = plt.subplots(figsize=(8, 5))
        for position, position_frame in group_frame.groupby("position"):
            position_frame = position_frame.sort_values("layer")
            axis.plot(
                position_frame["layer"],
                position_frame["mean_macro_f1"],
                label=position,
            )
        axis.set_title(
            f"{condition_label}: {task_group} probe performance by layer"
        )
        axis.set_xlabel("Hidden-state index")
        axis.set_ylabel("Mean macro-F1 across tasks")
        axis.set_ylim(0.0, 1.0)
        axis.legend()
        figure.tight_layout()
        path = output / f"{safe_name(task_group)}_macro_f1_by_layer.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

        figure, axis = plt.subplots(figsize=(8, 5))
        for position, position_frame in group_frame.groupby("position"):
            position_frame = position_frame.sort_values("layer")
            axis.plot(
                position_frame["layer"],
                position_frame["mean_delta_over_majority"],
                label=position,
            )
        axis.set_title(
            f"{condition_label}: {task_group} improvement over majority baseline"
        )
        axis.set_xlabel("Hidden-state index")
        axis.set_ylabel("Mean macro-F1 minus majority macro-F1")
        axis.legend()
        figure.tight_layout()
        path = output / f"{safe_name(task_group)}_delta_by_layer.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

    return paths


def plot_planning_tasks(
    frame: pd.DataFrame,
    output: Path,
    condition_label: str,
    position: str,
) -> Path | None:
    selected = frame[
        (frame["task_group"] == "planning")
        & (frame["position"] == position)
    ].copy()
    if selected.empty:
        return None

    figure, axis = plt.subplots(figsize=(10, 6))
    for task, task_frame in selected.groupby("task"):
        task_frame = task_frame.sort_values("layer")
        axis.plot(
            task_frame["layer"],
            task_frame["macro_f1_mean"],
            label=task,
        )
    axis.set_title(
        f"{condition_label}: planning tasks at {position}"
    )
    axis.set_xlabel("Hidden-state index")
    axis.set_ylabel("Macro-F1")
    axis.set_ylim(0.0, 1.0)
    axis.legend(fontsize=7)
    figure.tight_layout()
    path = output / (
        f"planning_tasks_{safe_name(position)}_by_layer.png"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create fixed-task macro-F1 layer curves."
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--condition-label", default="Condition")
    parser.add_argument(
        "--planning-position",
        default="pre_action_token",
    )
    args = parser.parse_args()

    run = args.run.resolve()
    input_path = run / "probes" / "probe_results.csv"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output = (
        args.output or run / "layer_curves"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    frame["layer"] = pd.to_numeric(frame["layer"], errors="raise").astype(int)
    summary = aggregate_layers(frame)
    summary.to_csv(output / "layer_summary.csv", index=False)

    task_curves = frame[
        [
            "task_group",
            "task",
            "position",
            "layer",
            "macro_f1_mean",
            "macro_f1_std",
            "majority_macro_f1_mean",
            "num_test_mean",
        ]
    ].copy()
    task_curves["delta_over_majority"] = (
        task_curves["macro_f1_mean"]
        - task_curves["majority_macro_f1_mean"]
    )
    task_curves.to_csv(output / "task_layer_curves.csv", index=False)

    figures = plot_group_curves(
        summary,
        output,
        args.condition_label,
    )
    planning_path = plot_planning_tasks(
        frame,
        output,
        args.condition_label,
        args.planning_position,
    )
    if planning_path is not None:
        figures.append(planning_path)

    peak_rows = []
    for (task_group, position), group_frame in summary.groupby(
        ["task_group", "position"]
    ):
        best = group_frame.loc[group_frame["mean_macro_f1"].idxmax()]
        peak_rows.append(
            {
                "task_group": task_group,
                "position": position,
                "peak_layer": int(best["layer"]),
                "peak_mean_macro_f1": float(best["mean_macro_f1"]),
                "peak_delta_over_majority": float(
                    best["mean_delta_over_majority"]
                ),
                "tasks": int(best["tasks"]),
            }
        )
    peaks = pd.DataFrame(peak_rows)
    peaks.to_csv(output / "peak_layers.csv", index=False)

    lines = [
        "# Layer-wise probe report",
        "",
        f"- Condition: **{args.condition_label}**",
        f"- Run: `{run}`",
        "- Layer 0 is the embedding output.",
        "- The final hidden-state index is the final normalized representation.",
        "- Each group curve averages the same fixed set of tasks at every layer.",
        "- No best-layer selection is used to construct the curves.",
        "",
        "## Peak layer by task group and position",
        "",
        peaks.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Output files",
        "",
        "- `layer_summary.csv`: group-level layer curves.",
        "- `task_layer_curves.csv`: every task at every layer.",
        "- `peak_layers.csv`: exploratory peak locations.",
    ]
    for path in figures:
        lines.append(f"- `{path.name}`")
    (output / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote layer curves to: {output}")


if __name__ == "__main__":
    main()
