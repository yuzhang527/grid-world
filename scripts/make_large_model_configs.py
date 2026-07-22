#!/usr/bin/env python3
"""
Create Qwen2.5-32B/72B experiment configs by cloning the existing 7B config.

The script preserves every prompt, decoding, environment, and logging setting,
and changes only the model identifier. This is preferable for a clean
model-scale comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MODELS = {
    "32b": "Qwen/Qwen2.5-32B-Instruct",
    "72b": "Qwen/Qwen2.5-72B-Instruct",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("configs/experiments/qwen25_7b_strategy_a.yaml"),
    )
    parser.add_argument(
        "--sizes",
        default="32b,72b",
        help="Comma-separated subset of 32b,72b.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing source config: {source}")

    text = source.read_text(encoding="utf-8")
    if "Qwen/Qwen2.5-7B-Instruct" not in text:
        raise RuntimeError(
            "The source config does not contain "
            "'Qwen/Qwen2.5-7B-Instruct'; refusing a blind replacement."
        )

    for size in [item.strip().lower() for item in args.sizes.split(",") if item.strip()]:
        if size not in MODELS:
            raise ValueError(f"Unknown size {size!r}; choose 32b or 72b.")
        destination = source.with_name(f"qwen25_{size}_strategy_a.yaml")
        replaced = text.replace("Qwen/Qwen2.5-7B-Instruct", MODELS[size])
        destination.write_text(replaced, encoding="utf-8")
        print(f"[large-config] {size}: {destination}")


if __name__ == "__main__":
    main()

