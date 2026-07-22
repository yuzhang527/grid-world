#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def episode_of(row: dict[str, Any]) -> str:
    for key in ("episode_id", "episode", "trajectory_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def parse_episode_catalog(text: str) -> dict[str, dict[str, Any]]:
    """
    Parse output from:

        trajectory_probe_viewer.py --list-episodes

    Expected logical fields:
        episode_id steps success loop_score
    """
    catalog: dict[str, dict[str, Any]] = {}

    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue

        bool_index = None
        for i, value in enumerate(parts):
            if value in {"True", "False"}:
                bool_index = i
                break

        if bool_index is None or bool_index < 2 or bool_index + 1 >= len(parts):
            continue

        try:
            episode_id = parts[0]
            steps = int(float(parts[bool_index - 1]))
            success = parts[bool_index] == "True"
            loop_score = float(parts[bool_index + 1])
        except (ValueError, IndexError):
            continue

        catalog[episode_id] = {
            "episode_id": episode_id,
            "steps": steps,
            "success": success,
            "loop_score": loop_score,
        }

    if not catalog:
        raise RuntimeError(
            "Could not parse any episodes from --list-episodes output.\n"
            "Run the command manually and inspect its column format."
        )

    return catalog


def load_step_flags(run: Path) -> dict[str, dict[str, int]]:
    steps_path = run / "steps.jsonl"
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing {steps_path}")

    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "step_rows": 0,
            "repaired_steps": 0,
            "parse_error_steps": 0,
            "invalid_move_steps": 0,
        }
    )

    with steps_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {steps_path}, line {line_number}: {exc}"
                ) from exc

            episode_id = episode_of(row)
            if not episode_id:
                continue

            item = stats[episode_id]
            item["step_rows"] += 1
            item["repaired_steps"] += int(bool(row.get("repaired", False)))
            item["parse_error_steps"] += int(bool(row.get("parse_error", False)))
            item["invalid_move_steps"] += int(bool(row.get("invalid_move", False)))

    return dict(stats)


def choose_unique(
    catalog: dict[str, dict[str, Any]],
    used: set[str],
    predicate,
    sort_key,
    reverse: bool = False,
) -> str | None:
    candidates = [
        row
        for row in catalog.values()
        if predicate(row) and row["episode_id"] not in used
    ]

    if not candidates:
        return None

    candidates.sort(key=sort_key, reverse=reverse)
    selected = str(candidates[0]["episode_id"])
    used.add(selected)
    return selected


def select_representative_cases(
    catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    used: set[str] = set()
    selected: list[dict[str, Any]] = []

    definitions = [
        (
            "success_clean_short",
            "Successful, short, and low-loop trajectory",
            lambda r: r["success"],
            lambda r: (r["loop_score"], r["steps"]),
            False,
        ),
        (
            "success_long",
            "Successful but unusually long trajectory",
            lambda r: r["success"],
            lambda r: (r["steps"], r["loop_score"]),
            True,
        ),
        (
            "success_with_loop",
            "Successful trajectory with the strongest looping behavior",
            lambda r: r["success"],
            lambda r: (r["loop_score"], r["steps"]),
            True,
        ),
        (
            "failure_with_loop",
            "Failed trajectory with the strongest looping behavior",
            lambda r: not r["success"],
            lambda r: (r["loop_score"], r["steps"]),
            True,
        ),
        (
            "failure_low_loop",
            "Failed trajectory without a strong loop score",
            lambda r: not r["success"],
            lambda r: (r["loop_score"], r["steps"]),
            False,
        ),
        (
            "repaired_heavy",
            "Trajectory containing many repaired model responses",
            lambda r: r.get("repaired_steps", 0) > 0,
            lambda r: (
                r.get("repaired_steps", 0),
                r.get("parse_error_steps", 0),
                r["steps"],
            ),
            True,
        ),
        (
            "parse_error_heavy",
            "Trajectory containing many response parse errors",
            lambda r: r.get("parse_error_steps", 0) > 0,
            lambda r: (
                r.get("parse_error_steps", 0),
                r.get("repaired_steps", 0),
                r["steps"],
            ),
            True,
        ),
        (
            "invalid_move_heavy",
            "Trajectory containing invalid moves",
            lambda r: r.get("invalid_move_steps", 0) > 0,
            lambda r: (r.get("invalid_move_steps", 0), r["steps"]),
            True,
        ),
    ]

    for label, description, predicate, sort_key, reverse in definitions:
        episode_id = choose_unique(
            catalog=catalog,
            used=used,
            predicate=predicate,
            sort_key=sort_key,
            reverse=reverse,
        )

        if episode_id is None:
            print(f"[gallery] skip category={label}: no matching episode")
            continue

        selected.append(
            {
                "label": label,
                "description": description,
                **catalog[episode_id],
            }
        )

    return selected


def validate_generated_html(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    # Detect the recently identified raw-newline JavaScript bug:
    broken_expression = "wrong.join('\n')"
    if broken_expression in text:
        raise RuntimeError(
            f"{path} contains a raw newline inside wrong.join(). "
            "Fix trajectory_probe_viewer_v3.py before continuing."
        )

    node = shutil.which("node")
    if node is None:
        print(f"[gallery] node unavailable; skipped JS syntax check for {path.name}")
        return

    scripts = re.findall(
        r"<script(?:\s[^>]*)?>(.*?)</script>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not scripts:
        raise RuntimeError(f"No inline JavaScript found in {path}")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".js",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(scripts[-1])
        js_path = Path(handle.name)

    try:
        result = subprocess.run(
            [node, "--check", str(js_path)],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"JavaScript syntax check failed for {path}:\n"
                f"{result.stdout}\n{result.stderr}"
            )
    finally:
        js_path.unlink(missing_ok=True)


def write_index(output_dir: Path, cases: list[dict[str, Any]]) -> None:
    cards = []

    for case in cases:
        success_text = "Success" if case["success"] else "Failure"
        success_class = "success" if case["success"] else "failure"

        cards.append(
            f"""
            <article class="card">
              <div class="category">{html.escape(case["label"])}</div>
              <h2>
                <a href="{html.escape(case["filename"])}">
                  {html.escape(case["episode_id"])}
                </a>
              </h2>
              <p>{html.escape(case["description"])}</p>
              <div class="metrics">
                <span class="{success_class}">{success_text}</span>
                <span>Steps: {case["steps"]}</span>
                <span>Loop score: {case["loop_score"]:.3f}</span>
                <span>Repaired: {case.get("repaired_steps", 0)}</span>
                <span>Parse errors: {case.get("parse_error_steps", 0)}</span>
                <span>Invalid moves: {case.get("invalid_move_steps", 0)}</span>
              </div>
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trajectory Viewer Gallery</title>
<style>
  body {{
    margin: 0;
    padding: 32px;
    background: #11151b;
    color: #eef2f7;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  main {{
    max-width: 1200px;
    margin: auto;
  }}
  h1 {{
    margin-bottom: 8px;
  }}
  .subtitle {{
    color: #aeb8c5;
    margin-bottom: 28px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
    gap: 18px;
  }}
  .card {{
    padding: 20px;
    border: 1px solid #303946;
    border-radius: 12px;
    background: #19202a;
  }}
  .category {{
    color: #9cb4cf;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
  }}
  a {{
    color: #8fc7ff;
    text-decoration: none;
  }}
  a:hover {{
    text-decoration: underline;
  }}
  .metrics {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .metrics span {{
    background: #252e3a;
    border-radius: 999px;
    padding: 5px 9px;
    font-size: 13px;
  }}
  .metrics .success {{
    background: #174b32;
  }}
  .metrics .failure {{
    background: #632a2a;
  }}
</style>
</head>
<body>
<main>
  <h1>Trajectory Viewer Gallery</h1>
  <p class="subtitle">
    Representative successful, failed, looping, repaired, and parse-error trajectories.
  </p>
  <section class="grid">
    {''.join(cards)}
  </section>
</main>
</body>
</html>
"""

    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--position", default="prompt_last")
    parser.add_argument("--layer", default="17")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Generate every episode instead of one representative per category.",
    )
    args = parser.parse_args()

    run = args.run.resolve()
    repo = Path(__file__).resolve().parents[1]

    catalog_viewer = repo / "scripts" / "trajectory_probe_viewer.py"
    viewer_v3 = repo / "scripts" / "trajectory_probe_viewer_v3.py"

    if not catalog_viewer.exists():
        raise FileNotFoundError(f"Missing {catalog_viewer}")
    if not viewer_v3.exists():
        raise FileNotFoundError(f"Missing {viewer_v3}")

    listing = subprocess.run(
        [
            sys.executable,
            str(catalog_viewer),
            "--run",
            str(run),
            "--list-episodes",
        ],
        text=True,
        capture_output=True,
    )

    if listing.returncode != 0:
        raise RuntimeError(
            f"Episode listing failed:\n{listing.stdout}\n{listing.stderr}"
        )

    catalog = parse_episode_catalog(listing.stdout)
    step_flags = load_step_flags(run)

    for episode_id, row in catalog.items():
        row.update(step_flags.get(episode_id, {}))

    if args.all_episodes:
        cases = []
        for row in sorted(catalog.values(), key=lambda r: r["episode_id"]):
            cases.append(
                {
                    "label": "all_episodes",
                    "description": "Complete episode export",
                    **row,
                }
            )
    else:
        cases = select_representative_cases(catalog)

    if not cases:
        raise RuntimeError("No trajectory cases were selected.")

    output_dir = run / "trajectory_viewer_gallery"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []

    for index, case in enumerate(cases, 1):
        episode_id = str(case["episode_id"])
        label = str(case["label"])

        print(
            f"\n[gallery] {index}/{len(cases)} "
            f"category={label} episode={episode_id}"
        )

        command = [
            sys.executable,
            str(viewer_v3),
            "--run",
            str(run),
            "--episode",
            episode_id,
            "--position",
            args.position,
            "--layer",
            str(args.layer),
            "--folds",
            str(args.folds),
            "--allow-missing",
        ]

        subprocess.run(command, check=True)

        source_html = run / "trajectory_viewer_v3" / f"{episode_id}.html"
        if not source_html.exists():
            raise FileNotFoundError(
                f"Viewer completed but expected HTML is missing: {source_html}"
            )

        validate_generated_html(source_html)

        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        safe_episode = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id)
        filename = f"{safe_label}__{safe_episode}.html"
        destination = output_dir / filename

        shutil.copy2(source_html, destination)

        case["filename"] = filename
        generated.append(case)

        print(f"[gallery] copied={destination}")

    write_index(output_dir, generated)

    cases_path = output_dir / "cases.json"
    cases_path.write_text(
        json.dumps(generated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    archive_path = run / "trajectory_viewer_gallery.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dir, arcname=output_dir.name)

    print("\n[gallery] complete")
    print(f"[gallery] index={output_dir / 'index.html'}")
    print(f"[gallery] metadata={cases_path}")
    print(f"[gallery] archive={archive_path}")


if __name__ == "__main__":
    main()
