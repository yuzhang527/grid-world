#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    try:
        os.symlink(source.resolve(), destination, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an analysis run that reuses trajectories and targets "
            "without overwriting the source run's activations or probes."
        )
    )
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--view-run", required=True, type=Path)
    parser.add_argument(
        "--reset-derived",
        action="store_true",
        help="Delete activations, probes, and layer-curve outputs in the view.",
    )
    args = parser.parse_args()

    source = args.source_run.resolve()
    view = args.view_run.resolve()

    required = [
        source / "maps.jsonl",
        source / "steps.jsonl",
        source / "episodes.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source artifacts:\n" + "\n".join(missing))

    view.mkdir(parents=True, exist_ok=True)
    if args.reset_derived:
        for name in ["activations", "probes", "layer_curves"]:
            path = view / name
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)

    for name in [
        "maps.jsonl",
        "steps.jsonl",
        "episodes.jsonl",
        "summary.json",
        "resolved_config.yaml",
        "manifest.json",
    ]:
        source_path = source / name
        if source_path.exists():
            link_or_copy(source_path, view / name)

    # Targets are rebuilt inside the view so the source run remains unchanged.
    (view / "SOURCE_RUN.txt").write_text(
        str(source) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared analysis view: {view}")
    print(f"Source trajectories: {source}")


if __name__ == "__main__":
    main()
