#!/usr/bin/env python3
"""
Task-parallel multi-GPU linear probing for grid-world activations.

Why task parallel rather than DDP?
Each (position, layer, split) probe is statistically independent and the
linear heads are tiny. Synchronizing one head with DDP would add unnecessary
communication. This script gives each GPU different independent jobs and
merges the results deterministically.

It accepts the standard grid-world activation contract:
  RUN/activations/{X.npy,position_mask.npy,layers.npy,positions.json,meta.jsonl}
  RUN/targets/targets.jsonl

It writes:
  RUN/<output_subdir>/
    probe_results_splits.csv
    probe_results.csv
    best_by_task.csv
    group_summary.csv
    summary.md
    jobs/*.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


PLANNING_TASKS = {
    "chosen_action_is_astar_best",
    "chosen_action_reduces_true_distance",
    "loop_risk",
    "position_action_seen_before",
    "position_seen_before",
    "chosen_target_gold_belief",
}
FAITHFULNESS_TASKS = {
    "model_missed_any_gold_known",
    "action_optimal_but_belief_incomplete",
    "model_known_cell_acc_step",
    "belief_known_acc_step",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def first_present(row: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return default


def episode_id_of(row: dict[str, Any]) -> str:
    return str(first_present(row, ["episode_id", "episode", "episode_name", "id"], ""))


def step_id_of(row: dict[str, Any]) -> int:
    value = first_present(row, ["step_id", "step", "t", "step_index"], -1)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def canonical_key(row: dict[str, Any]) -> tuple[str, int]:
    return episode_id_of(row), step_id_of(row)


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return None
        rounded = round(float(value))
        if abs(float(value) - rounded) < 1e-8:
            return str(int(rounded))
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "nan", "null", "na"}:
            return None
        return stripped
    return None


def task_group_of(name: str) -> str | None:
    if re.fullmatch(r"gold_local_(UP|DOWN|LEFT|RIGHT)_OFUW", name):
        return "local"
    if re.fullmatch(r"gold_cell_x\d+_y\d+_OFU", name):
        return "cells"
    if re.fullmatch(r"gold_cell_x\d+_y\d+_known", name):
        return "memory"
    if re.fullmatch(r"(explicit|model)_cell_x\d+_y\d+_OFU", name):
        return "explicit_cells"
    if re.fullmatch(r"true_cell_x\d+_y\d+_unobserved_(OF|OFU)", name):
        return "true_cells_unobserved"
    if re.fullmatch(r"true_cell_x\d+_y\d+_(OF|OFU)", name):
        return "true_cells"
    if name in PLANNING_TASKS or re.fullmatch(
        r"true_action_(UP|DOWN|LEFT|RIGHT)_is_astar_best",
        name,
    ):
        return "planning"
    if name in FAITHFULNESS_TASKS:
        return "faithfulness"
    return None


def discover_tasks(
    targets: list[dict[str, Any]],
    requested_groups: set[str],
    explicit_tasks: set[str],
    task_regex: str | None,
    min_class_count: int,
) -> list[dict[str, Any]]:
    candidate_names: set[str] = set()
    for row in targets:
        candidate_names.update(row.keys())

    regex = re.compile(task_regex) if task_regex else None
    tasks: list[dict[str, Any]] = []

    for name in sorted(candidate_names):
        group = task_group_of(name)
        selected = (
            name in explicit_tasks
            or (group is not None and group in requested_groups)
            or (regex is not None and regex.search(name) is not None)
        )
        if not selected:
            continue
        if group is None:
            group = "custom"

        labels = [
            normalized
            for row in targets
            if (normalized := normalize_label(row.get(name))) is not None
        ]
        counts = Counter(labels)
        classes = sorted(
            [label for label, count in counts.items() if count >= min_class_count]
        )
        if len(classes) < 2:
            print(
                f"[probe/multigpu] skip task={name}: classes={dict(counts)}",
                flush=True,
            )
            continue
        tasks.append(
            {
                "name": name,
                "group": group,
                "classes": classes,
                "counts": dict(counts),
            }
        )

    if not tasks:
        raise RuntimeError("No probe tasks matched the requested groups/tasks.")
    return tasks


def resolve_positions(spec: str, available: list[str]) -> list[str]:
    lowered = spec.strip().lower()
    if lowered == "all":
        return list(available)
    if lowered == "auto":
        preferred = [
            "mean_last_feedback",
            "mean_current_belief_grid",
            "pre_action_token",
            "prompt_last",
        ]
        selected = [name for name in preferred if name in available]
        if not selected:
            raise RuntimeError("None of the automatic positions exist in activations.")
        return selected
    requested = [part.strip() for part in spec.split(",") if part.strip()]
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Unknown activation positions: {missing}")
    return requested


def resolve_layers(spec: str, available: list[int]) -> list[int]:
    lowered = spec.strip().lower()
    if lowered == "all":
        return list(available)
    if lowered == "auto":
        maximum = max(available)
        desired = {
            0,
            round(maximum * 0.25),
            round(maximum * 0.50),
            round(maximum * 0.75),
            maximum,
        }
        selected = [layer for layer in available if layer in desired]
        if not selected:
            selected = list(available)
        return selected
    requested = sorted(
        set(int(part.strip()) for part in spec.split(",") if part.strip())
    )
    missing = [layer for layer in requested if layer not in available]
    if missing:
        raise ValueError(f"Unknown activation layers: {missing}")
    return requested


def align_meta_and_targets(
    meta_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    target_map = {
        canonical_key(row): row
        for row in target_rows
        if canonical_key(row) != ("", -1)
    }

    activation_indices: list[int] = []
    aligned_targets: list[dict[str, Any]] = []
    groups: list[str] = []

    if target_map:
        for activation_index, meta in enumerate(meta_rows):
            key = canonical_key(meta)
            target = target_map.get(key)
            if target is None:
                continue
            activation_indices.append(activation_index)
            aligned_targets.append(target)
            groups.append(key[0])
    elif len(meta_rows) == len(target_rows):
        activation_indices = list(range(len(meta_rows)))
        aligned_targets = list(target_rows)
        groups = [episode_id_of(meta) for meta in meta_rows]
    else:
        raise RuntimeError(
            "Could not align activation metadata with targets by "
            "(episode_id, step_id), and row counts differ."
        )

    if not activation_indices:
        raise RuntimeError("No activation rows aligned with target rows.")

    return (
        np.asarray(activation_indices, dtype=np.int64),
        aligned_targets,
        np.asarray(groups, dtype=object),
    )


def build_splits(
    groups: np.ndarray,
    n_splits: int,
    test_size: float,
    random_seed: int,
) -> list[dict[str, list[str]]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise RuntimeError("At least three episodes are required for group splits.")

    dummy = np.zeros(len(groups), dtype=np.int8)
    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_seed,
    )
    splits = []
    for train_index, test_index in splitter.split(dummy, groups=groups):
        splits.append(
            {
                "train_groups": sorted(set(groups[train_index].tolist())),
                "test_groups": sorted(set(groups[test_index].tolist())),
            }
        )
    return splits


def encode_targets(
    aligned_targets: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> np.ndarray:
    labels = np.full(
        (len(aligned_targets), len(tasks)),
        fill_value=-100,
        dtype=np.int64,
    )
    for task_index, task in enumerate(tasks):
        class_to_index = {
            label: class_index
            for class_index, label in enumerate(task["classes"])
        }
        for row_index, row in enumerate(aligned_targets):
            label = normalize_label(row.get(task["name"]))
            if label in class_to_index:
                labels[row_index, task_index] = class_to_index[label]
    return labels


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def train_one_job(
    config: dict[str, Any],
    job: dict[str, Any],
    device_name: str,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(max(1, int(config["cpu_threads_per_worker"])))
    if device_name.startswith("cuda"):
        torch.cuda.set_device(0)
        torch.set_float32_matmul_precision("high")

    run = Path(config["run"])
    activations_dir = run / config["activations_subdir"]
    X_memmap = np.load(activations_dir / "X.npy", mmap_mode="r")
    position_mask = np.load(
        activations_dir / "position_mask.npy",
        mmap_mode="r",
    )

    activation_indices = np.asarray(config["activation_indices"], dtype=np.int64)
    labels = np.load(config["encoded_targets_path"], mmap_mode="r")
    groups = np.asarray(config["groups"], dtype=object)
    tasks = config["tasks"]
    split = config["splits"][job["split_index"]]

    pidx = int(job["position_index"])
    lidx = int(job["layer_index"])
    valid_position = np.asarray(
        position_mask[activation_indices, pidx],
        dtype=np.bool_,
    )
    train_group_set = set(split["train_groups"])
    test_group_set = set(split["test_groups"])
    train_rows = np.asarray(
        [
            index
            for index, group in enumerate(groups)
            if valid_position[index] and group in train_group_set
        ],
        dtype=np.int64,
    )
    test_rows = np.asarray(
        [
            index
            for index, group in enumerate(groups)
            if valid_position[index] and group in test_group_set
        ],
        dtype=np.int64,
    )
    if len(train_rows) == 0 or len(test_rows) == 0:
        return []

    X_train_np = np.asarray(
        X_memmap[activation_indices[train_rows], pidx, lidx, :],
        dtype=np.float32,
    )
    X_test_np = np.asarray(
        X_memmap[activation_indices[test_rows], pidx, lidx, :],
        dtype=np.float32,
    )
    y_train_np = np.asarray(labels[train_rows], dtype=np.int64)
    y_test_np = np.asarray(labels[test_rows], dtype=np.int64)

    if config["standardize"]:
        mean = X_train_np.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = X_train_np.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        X_train_np = (X_train_np - mean) / std
        X_test_np = (X_test_np - mean) / std

    active_task_indices: list[int] = []
    task_class_counts: list[int] = []
    task_class_weights: list[torch.Tensor] = []
    for task_index, task in enumerate(tasks):
        valid_train = y_train_np[:, task_index] >= 0
        values = y_train_np[valid_train, task_index]
        observed = sorted(set(values.tolist()))
        required = list(range(len(task["classes"])))
        if len(values) < config["min_samples_per_task"]:
            continue
        if len(observed) < 2:
            continue
        # A class absent from the training fold cannot be learned. Skip this
        # task for the current split instead of allowing an untrained logit to
        # produce "predicted class absent from y_true" warnings.
        if set(observed) != set(required):
            continue

        counts = np.bincount(values, minlength=len(required)).astype(np.float64)
        weights = np.zeros_like(counts, dtype=np.float32)
        nonzero = counts > 0
        weights[nonzero] = counts[nonzero].sum() / (
            nonzero.sum() * counts[nonzero]
        )
        active_task_indices.append(task_index)
        task_class_counts.append(len(required))
        task_class_weights.append(
            torch.tensor(weights, dtype=torch.float32, device=device_name)
        )

    if not active_task_indices:
        return []

    offsets = np.cumsum([0] + task_class_counts).tolist()

    class MultiHeadLinear(nn.Module):
        def __init__(self, hidden_size: int, total_classes: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(total_classes, hidden_size))
            self.bias = nn.Parameter(torch.zeros(total_classes))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, self.weight, self.bias)

    model = MultiHeadLinear(X_train_np.shape[1], offsets[-1]).to(device_name)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(config["random_seed"])
        + 100003 * int(job["split_index"])
        + 1009 * int(job["position_index"])
        + int(job["layer_value"])
    )
    batch_size = min(int(config["batch_size"]), len(X_train_np))
    epochs_ran = 0

    for epoch in range(int(config["epochs"])):
        permutation = torch.randperm(
            len(X_train_np),
            generator=generator,
        ).numpy()
        model.train()
        for start in range(0, len(permutation), batch_size):
            batch_indices = permutation[start : start + batch_size]
            xb = torch.from_numpy(X_train_np[batch_indices]).to(
                device_name,
                non_blocking=False,
            )
            yb = torch.from_numpy(y_train_np[batch_indices]).to(
                device_name,
                non_blocking=False,
            )
            logits = model(xb)
            losses = []
            for active_slot, task_index in enumerate(active_task_indices):
                valid = yb[:, task_index] >= 0
                if int(valid.sum()) == 0:
                    continue
                start_class = offsets[active_slot]
                end_class = offsets[active_slot + 1]
                losses.append(
                    F.cross_entropy(
                        logits[valid, start_class:end_class],
                        yb[valid, task_index],
                        weight=task_class_weights[active_slot],
                    )
                )
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        epochs_ran = epoch + 1

    model.eval()
    test_logits_parts = []
    eval_batch_size = max(batch_size, 1024)
    with torch.inference_mode():
        for start in range(0, len(X_test_np), eval_batch_size):
            xb = torch.from_numpy(X_test_np[start : start + eval_batch_size]).to(
                device_name
            )
            test_logits_parts.append(model(xb).cpu())
    test_logits = torch.cat(test_logits_parts, dim=0).numpy()

    rows: list[dict[str, Any]] = []
    for active_slot, task_index in enumerate(active_task_indices):
        task = tasks[task_index]
        valid_test = y_test_np[:, task_index] >= 0
        valid_train = y_train_np[:, task_index] >= 0
        if int(valid_test.sum()) == 0:
            continue

        y_true = y_test_np[valid_test, task_index]
        start_class = offsets[active_slot]
        end_class = offsets[active_slot + 1]
        y_pred = test_logits[valid_test, start_class:end_class].argmax(axis=1)
        metric_labels = list(range(len(task["classes"])))
        metrics = safe_metrics(y_true, y_pred, metric_labels)

        train_values = y_train_np[valid_train, task_index]
        majority_class = int(
            np.bincount(
                train_values,
                minlength=len(task["classes"]),
            ).argmax()
        )
        majority_pred = np.full_like(y_true, majority_class)
        majority_metrics = safe_metrics(y_true, majority_pred, metric_labels)

        rows.append(
            {
                "task": task["name"],
                "task_group": task["group"],
                "position": job["position"],
                "layer": int(job["layer_value"]),
                "split": int(job["split_index"]),
                "backend": "torch_multigpu_task_parallel",
                "classes": json.dumps(task["classes"], ensure_ascii=False),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "majority_accuracy": majority_metrics["accuracy"],
                "majority_macro_f1": majority_metrics["macro_f1"],
                "num_train": int(valid_train.sum()),
                "num_test": int(valid_test.sum()),
                "epochs": epochs_ran,
            }
        )

    del model, optimizer
    if device_name.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def worker_main(
    worker_index: int,
    gpu: str,
    config_path: str,
    jobs: list[dict[str, Any]],
) -> None:
    try:
        if gpu.lower() == "cpu":
            device_name = "cpu"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            device_name = "cuda:0"

        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        jobs_dir = Path(config["output_dir"]) / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[probe/worker {worker_index}] device={gpu} jobs={len(jobs)}",
            flush=True,
        )
        for local_index, job in enumerate(jobs, 1):
            job_path = jobs_dir / job["filename"]
            if job_path.exists() and config["resume"]:
                continue

            started = time.time()
            rows = train_one_job(config, job, device_name)
            payload = {
                "job": job,
                "rows": rows,
                "elapsed_seconds": time.time() - started,
                "worker_index": worker_index,
                "gpu": gpu,
            }
            temporary = job_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(job_path)
            print(
                f"[probe/worker {worker_index}] {local_index}/{len(jobs)} "
                f"{job['position']} L{job['layer_value']} "
                f"split={job['split_index']} tasks={len(rows)} "
                f"seconds={payload['elapsed_seconds']:.1f}",
                flush=True,
            )
    except Exception:
        traceback.print_exc()
        raise


def aggregate_results(output_dir: Path, jobs: list[dict[str, Any]]) -> None:
    split_rows: list[dict[str, Any]] = []
    missing = []
    for job in jobs:
        path = output_dir / "jobs" / job["filename"]
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        split_rows.extend(payload.get("rows", []))

    if missing:
        raise RuntimeError(
            f"{len(missing)} probe jobs are missing. First missing file: {missing[0]}"
        )
    if not split_rows:
        raise RuntimeError("Probe workers completed but produced no result rows.")

    split_df = pd.DataFrame(split_rows).sort_values(
        ["task_group", "task", "position", "layer", "split"]
    )
    split_df.to_csv(output_dir / "probe_results_splits.csv", index=False)

    group_columns = [
        "task",
        "task_group",
        "position",
        "layer",
        "backend",
        "classes",
    ]
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "majority_accuracy",
        "majority_macro_f1",
        "num_test",
    ]

    records = []
    for keys, frame in split_df.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        for metric in metric_columns:
            values = frame[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        records.append(row)

    result_df = pd.DataFrame(records).sort_values(
        ["task_group", "task", "position", "layer"]
    )
    result_df.to_csv(output_dir / "probe_results.csv", index=False)

    best_df = (
        result_df.sort_values(
            ["task", "macro_f1_mean", "macro_f1_std"],
            ascending=[True, False, True],
        )
        .groupby("task", as_index=False)
        .head(1)
        .sort_values(["task_group", "task"])
    )
    best_df.to_csv(output_dir / "best_by_task.csv", index=False)

    summary_records = []
    for task_group, frame in best_df.groupby("task_group"):
        summary_records.append(
            {
                "task_group": task_group,
                "tasks": int(len(frame)),
                "mean_best_macro_f1": float(frame["macro_f1_mean"].mean()),
                "median_best_macro_f1": float(frame["macro_f1_mean"].median()),
                "mean_split_std": float(frame["macro_f1_std"].mean()),
                "mean_delta_over_majority": float(
                    (
                        frame["macro_f1_mean"]
                        - frame["majority_macro_f1_mean"]
                    ).mean()
                ),
            }
        )
    group_summary = pd.DataFrame(summary_records).sort_values("task_group")
    group_summary.to_csv(output_dir / "group_summary.csv", index=False)

    lines = [
        "# Multi-GPU probe report",
        "",
        "## Best score by task group",
        "",
        group_summary.to_markdown(index=False),
        "",
        "## Best position/layer per task",
        "",
        best_df[
            [
                "task_group",
                "task",
                "position",
                "layer",
                "macro_f1_mean",
                "macro_f1_std",
                "majority_macro_f1_mean",
            ]
        ].to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- Work is parallelized across independent position/layer/split jobs.",
        "- Splits are grouped by episode.",
        "- Compare macro-F1 with the majority macro-F1 baseline.",
        "- Decodability does not by itself establish causal use.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--groups", default="local,cells,planning")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--task-regex", default=None)
    parser.add_argument("--positions", default="auto")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument("--min-samples-per-task", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=2)
    parser.add_argument("--activations-subdir", default="activations")
    parser.add_argument("--targets-subdir", default="targets")
    parser.add_argument("--output-subdir", default="probes_multigpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run = args.run.resolve()
    activations_dir = run / args.activations_subdir
    targets_path = run / args.targets_subdir / "targets.jsonl"
    required = [
        activations_dir / "X.npy",
        activations_dir / "position_mask.npy",
        activations_dir / "layers.npy",
        activations_dir / "positions.json",
        activations_dir / "meta.jsonl",
        targets_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    output_dir = run / args.output_subdir
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "jobs").mkdir(parents=True, exist_ok=True)

    available_positions = json.loads(
        (activations_dir / "positions.json").read_text(encoding="utf-8")
    )
    available_layers = [
        int(value)
        for value in np.load(activations_dir / "layers.npy").tolist()
    ]
    selected_positions = resolve_positions(args.positions, available_positions)
    selected_layers = resolve_layers(args.layers, available_layers)

    meta_rows = read_jsonl(activations_dir / "meta.jsonl")
    target_rows = read_jsonl(targets_path)
    activation_indices, aligned_targets, groups = align_meta_and_targets(
        meta_rows,
        target_rows,
    )

    requested_groups = {
        item.strip()
        for item in args.groups.split(",")
        if item.strip()
    }
    explicit_tasks = {
        item.strip()
        for item in args.tasks.split(",")
        if item.strip()
    }
    tasks = discover_tasks(
        aligned_targets,
        requested_groups,
        explicit_tasks,
        args.task_regex,
        args.min_class_count,
    )
    encoded_targets = encode_targets(aligned_targets, tasks)
    encoded_targets_path = output_dir / "encoded_targets.npy"
    np.save(encoded_targets_path, encoded_targets)

    splits = build_splits(
        groups,
        args.splits,
        args.test_size,
        args.random_seed,
    )
    (output_dir / "splits.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    position_to_index = {
        name: index for index, name in enumerate(available_positions)
    }
    layer_to_index = {
        value: index for index, value in enumerate(available_layers)
    }

    jobs: list[dict[str, Any]] = []
    for position in selected_positions:
        for layer in selected_layers:
            for split_index in range(args.splits):
                safe_position = re.sub(r"[^A-Za-z0-9_.-]+", "_", position)
                jobs.append(
                    {
                        "position": position,
                        "position_index": position_to_index[position],
                        "layer_value": layer,
                        "layer_index": layer_to_index[layer],
                        "split_index": split_index,
                        "filename": (
                            f"{safe_position}__L{layer}__S{split_index:03d}.json"
                        ),
                    }
                )

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain GPU ids or 'cpu'.")

    config = {
        "run": str(run),
        "output_dir": str(output_dir),
        "activations_subdir": args.activations_subdir,
        "activation_indices": activation_indices.tolist(),
        "groups": groups.tolist(),
        "tasks": tasks,
        "splits": splits,
        "encoded_targets_path": str(encoded_targets_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "min_samples_per_task": args.min_samples_per_task,
        "random_seed": args.random_seed,
        "standardize": args.standardize,
        "cpu_threads_per_worker": args.cpu_threads_per_worker,
        "resume": args.resume,
    }
    config_path = output_dir / "worker_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "jobs_manifest.json").write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pending_jobs = [
        job
        for job in jobs
        if not (
            args.resume
            and (output_dir / "jobs" / job["filename"]).exists()
        )
    ]

    print(
        f"[probe/multigpu] aligned_rows={len(aligned_targets)} "
        f"episodes={len(set(groups.tolist()))}",
        flush=True,
    )
    print(
        f"[probe/multigpu] tasks={len(tasks)} "
        f"positions={selected_positions} layers={selected_layers}",
        flush=True,
    )
    print(
        f"[probe/multigpu] jobs_total={len(jobs)} "
        f"jobs_pending={len(pending_jobs)} workers={gpus}",
        flush=True,
    )

    if pending_jobs:
        assignments = [pending_jobs[index::len(gpus)] for index in range(len(gpus))]
        context = mp.get_context("spawn")
        processes = []
        for worker_index, (gpu, assigned_jobs) in enumerate(
            zip(gpus, assignments)
        ):
            if not assigned_jobs:
                continue
            process = context.Process(
                target=worker_main,
                args=(
                    worker_index,
                    gpu,
                    str(config_path),
                    assigned_jobs,
                ),
            )
            process.start()
            processes.append(process)

        failures = []
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failures.append((process.pid, process.exitcode))
        if failures:
            raise RuntimeError(f"Probe workers failed: {failures}")

    aggregate_results(output_dir, jobs)
    print(f"[probe/multigpu] saved={output_dir}", flush=True)
    print(f"[probe/multigpu] report={output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()

