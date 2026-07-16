#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_condition(
    label: str,
    run: Path,
) -> pd.DataFrame:
    path = run / "probes" / "probe_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["condition"] = label
    frame["layer"] = pd.to_numeric(
        frame["layer"],
        errors="raise",
    ).astype(int)
    frame["delta_over_majority"] = (
        frame["macro_f1_mean"]
        - frame["majority_macro_f1_mean"]
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare explicit-belief and no-grid layer curves at common "
            "representation positions."
        )
    )
    parser.add_argument("--explicit-run", required=True, type=Path)
    parser.add_argument("--no-grid-run", required=True, type=Path)
    parser.add_argument(
        "--positions",
        default="prompt_last,pre_action_token",
    )
    parser.add_argument(
        "--groups",
        default="cells,planning,true_cells,true_cells_unobserved",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    positions = [
        value.strip()
        for value in args.positions.split(",")
        if value.strip()
    ]
    groups = [
        value.strip()
        for value in args.groups.split(",")
        if value.strip()
    ]

    frame = pd.concat(
        [
            load_condition(
                "Explicit belief",
                args.explicit_run.resolve(),
            ),
            load_condition(
                "No explicit grid",
                args.no_grid_run.resolve(),
            ),
        ],
        ignore_index=True,
    )
    frame = frame[
        frame["position"].isin(positions)
        & frame["task_group"].isin(groups)
    ].copy()
    if frame.empty:
        raise ValueError("No common rows for the requested groups and positions")

    summary = (
        frame.groupby(
            ["condition", "task_group", "position", "layer"],
            dropna=False,
        )
        .agg(
            tasks=("task", "nunique"),
            mean_macro_f1=("macro_f1_mean", "mean"),
            mean_majority_macro_f1=(
                "majority_macro_f1_mean",
                "mean",
            ),
            mean_delta_over_majority=(
                "delta_over_majority",
                "mean",
            ),
            mean_split_std=("macro_f1_std", "mean"),
            mean_num_test=("num_test_mean", "mean"),
        )
        .reset_index()
        .sort_values(
            ["task_group", "position", "condition", "layer"]
        )
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        output / "condition_layer_summary.csv",
        index=False,
    )

    figure_paths = []
    for (task_group, position), group_frame in summary.groupby(
        ["task_group", "position"]
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        for condition, condition_frame in group_frame.groupby(
            "condition"
        ):
            condition_frame = condition_frame.sort_values("layer")
            axis.plot(
                condition_frame["layer"],
                condition_frame["mean_macro_f1"],
                label=condition,
            )
        axis.set_title(
            f"{task_group} at {position}: explicit vs no-grid"
        )
        axis.set_xlabel("Hidden-state index")
        axis.set_ylabel("Mean macro-F1 across tasks")
        axis.set_ylim(0.0, 1.0)
        axis.legend()
        figure.tight_layout()
        filename = (
            f"{task_group}_{position}_condition_macro_f1.png"
            .replace("/", "_")
        )
        path = output / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        figure_paths.append(path)

        figure, axis = plt.subplots(figsize=(8, 5))
        for condition, condition_frame in group_frame.groupby(
            "condition"
        ):
            condition_frame = condition_frame.sort_values("layer")
            axis.plot(
                condition_frame["layer"],
                condition_frame["mean_delta_over_majority"],
                label=condition,
            )
        axis.set_title(
            f"{task_group} at {position}: improvement over baseline"
        )
        axis.set_xlabel("Hidden-state index")
        axis.set_ylabel("Mean macro-F1 minus majority macro-F1")
        axis.legend()
        figure.tight_layout()
        filename = (
            f"{task_group}_{position}_condition_delta.png"
            .replace("/", "_")
        )
        path = output / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        figure_paths.append(path)

    lines = [
        "# Explicit-belief versus no-grid layer comparison",
        "",
        f"- Explicit run: `{args.explicit_run.resolve()}`",
        f"- No-grid run: `{args.no_grid_run.resolve()}`",
        f"- Common positions: {', '.join(positions)}",
        f"- Task groups: {', '.join(groups)}",
        "",
        "The prompt-last comparison is the cleanest map-representation "
        "comparison because it occurs before either condition generates a response.",
        "",
        "The pre-action comparison is useful for action selection, but in the "
        "explicit condition the model may already have generated its belief grid "
        "before the action field.",
        "",
        "## Files",
        "",
        "- `condition_layer_summary.csv`",
    ]
    lines.extend(f"- `{path.name}`" for path in figure_paths)
    (output / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote condition comparison to: {output}")


if __name__ == "__main__":
    main()
