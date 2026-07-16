#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_fixed(
    run: Path, label: str, position: str, layer: int
) -> pd.DataFrame:
    path = run / "probes" / "probe_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    selected = frame[
        (frame["position"] == position)
        & (pd.to_numeric(frame["layer"], errors="coerce") == layer)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No results for position={position}, layer={layer} in {path}"
        )
    selected["delta_over_majority"] = (
        selected["macro_f1_mean"]
        - selected["majority_macro_f1_mean"]
    )
    keep = [
        "task_group",
        "task",
        "macro_f1_mean",
        "macro_f1_std",
        "majority_macro_f1_mean",
        "delta_over_majority",
        "num_test_mean",
    ]
    selected = selected[keep].rename(
        columns={
            column: f"{label}_{column}"
            for column in keep
            if column not in {"task_group", "task"}
        }
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed-position probe results across probe views."
    )
    parser.add_argument(
        "--view",
        action="append",
        required=True,
        help="LABEL=RUN_PATH; repeat for each view.",
    )
    parser.add_argument("--position", default="pre_action_token")
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parsed_views = []
    for value in args.view:
        if "=" not in value:
            raise ValueError("--view must use LABEL=RUN_PATH")
        label, path = value.split("=", 1)
        parsed_views.append((label.strip(), Path(path).resolve()))

    merged = None
    for label, run in parsed_views:
        frame = load_fixed(run, label, args.position, args.layer)
        merged = (
            frame
            if merged is None
            else merged.merge(frame, on=["task_group", "task"], how="outer")
        )

    assert merged is not None
    args.output.mkdir(parents=True, exist_ok=True)

    labels = [label for label, _ in parsed_views]
    if {"success", "failure"}.issubset(set(labels)):
        merged["success_minus_failure_macro_f1"] = (
            merged["success_macro_f1_mean"]
            - merged["failure_macro_f1_mean"]
        )
        merged["success_minus_failure_delta"] = (
            merged["success_delta_over_majority"]
            - merged["failure_delta_over_majority"]
        )

    merged.to_csv(args.output / "fixed_probe_comparison.csv", index=False)

    group_rows = []
    for group, group_frame in merged.groupby("task_group"):
        row = {"task_group": group, "tasks": len(group_frame)}
        for label in labels:
            for metric in [
                "macro_f1_mean",
                "macro_f1_std",
                "delta_over_majority",
                "num_test_mean",
            ]:
                column = f"{label}_{metric}"
                row[f"{label}_{metric}_mean"] = group_frame[column].mean()
        group_rows.append(row)
    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(args.output / "group_comparison.csv", index=False)

    report = [
        "# Fixed-position probe comparison",
        "",
        f"- Position: `{args.position}`",
        f"- Layer: `{args.layer}`",
        "",
        "Using one pre-specified position and layer avoids selecting a "
        "different best combination for each subset.",
        "",
        "## Group summary",
        "",
        group_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]

    if "success_minus_failure_macro_f1" in merged:
        planning = merged[
            merged["task_group"] == "planning"
        ].sort_values("success_minus_failure_macro_f1", ascending=False)
        cells = merged[merged["task_group"] == "cells"]
        report.extend(
            [
                "## Successful minus failed episodes",
                "",
                f"- Mean map-location-state difference: "
                f"**{cells['success_minus_failure_macro_f1'].mean():.3f}**",
                f"- Mean planning-task difference: "
                f"**{planning['success_minus_failure_macro_f1'].mean():.3f}**",
                "",
                "Positive values mean the information is more decodable in "
                "successful episodes.",
                "",
                "### Planning tasks",
                "",
                planning[
                    [
                        "task",
                        "success_macro_f1_mean",
                        "failure_macro_f1_mean",
                        "success_minus_failure_macro_f1",
                        "success_delta_over_majority",
                        "failure_delta_over_majority",
                    ]
                ].to_markdown(index=False, floatfmt=".3f"),
                "",
            ]
        )

    report.extend(
        [
            "## Interpretation caution",
            "",
            "A difference between successful and failed episodes is "
            "correlational. It can reflect stronger internal information, "
            "different trajectory lengths, or remaining distribution differences.",
        ]
    )
    (args.output / "comparison.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Wrote comparison to: {args.output}")


if __name__ == "__main__":
    main()
