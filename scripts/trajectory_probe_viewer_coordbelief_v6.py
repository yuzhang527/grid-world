#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["episode_id"]), int(row["step_id"])


def load_positions(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, dict):
        for name in ("positions", "names"):
            if isinstance(value.get(name), list):
                return [str(x) for x in value[name]]
    raise RuntimeError(f"Unrecognized positions.json: {path}")


def choose_layer(run: Path, available: list[int], requested: str) -> int:
    if requested != "auto":
        layer = int(requested)
        if layer not in available:
            raise RuntimeError(f"Layer {layer} unavailable; choices={available}")
        return layer
    result_path = run / "probes_multigpu" / "probe_results.csv"
    if result_path.exists():
        try:
            import csv
            by_layer: dict[int, list[float]] = {}
            with result_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("task_group") != "cells" or row.get("position") != "prompt_last":
                        continue
                    metric = row.get("macro_f1_mean") or row.get("macro_f1")
                    if metric in (None, ""):
                        continue
                    layer = int(float(row["layer"]))
                    by_layer.setdefault(layer, []).append(float(metric))
            if by_layer:
                return max(by_layer, key=lambda layer: float(np.mean(by_layer[layer])))
        except Exception as exc:
            print(f"[viewer-v6] warning: could not derive best layer from probe_results.csv: {exc}")
    return available[len(available) // 2]


def parse_explicit(step: dict[str, Any], size: int) -> list[str]:
    raw = step.get("parsed_belief_coordinates") or step.get("belief_coordinates") or {}
    cells = ["U"] * (size * size)
    if isinstance(raw, dict):
        for state in ("F", "O"):
            for coord in raw.get(state, []) or []:
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    x, y = int(coord[0]), int(coord[1])
                    if 0 <= x < size and 0 <= y < size:
                        idx = x * size + y
                        if cells[idx] != "U" and cells[idx] != state:
                            raise RuntimeError(f"Explicit F/O overlap at {(x, y)}")
                        cells[idx] = state
    return cells


def grid_payload(target: dict[str, Any], prefix: str, size: int) -> list[str]:
    return [str(target.get(f"{prefix}_x{x}_y{y}_OFU", "U")) for x in range(size) for y in range(size)]


def true_payload(target: dict[str, Any], size: int) -> list[str]:
    return [str(target.get(f"true_cell_x{x}_y{y}_FO", "F")) for x in range(size) for y in range(size)]


def fit_oof(X: np.ndarray, targets: list[dict[str, Any]], episodes: np.ndarray, size: int, folds: int) -> np.ndarray:
    classes = np.array(["F", "O", "U"], dtype=object)
    predictions = np.full((len(targets), size * size), "U", dtype=object)
    unique_groups = np.unique(episodes)
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("Need at least two episodes for OOF viewer probes")
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, groups=episodes)):
        print(f"[viewer-v6] fold={fold} train={len(train_idx)} test={len(test_idx)}")
        for cell_index in range(size * size):
            x, y = divmod(cell_index, size)
            task = f"gold_cell_x{x}_y{y}_OFU"
            y_train = np.array([str(targets[i][task]) for i in train_idx], dtype=object)
            unique = np.unique(y_train)
            if len(unique) == 1:
                predictions[test_idx, cell_index] = unique[0]
                continue
            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=0,
            )
            model.fit(X[train_idx], y_train)
            predictions[test_idx, cell_index] = model.predict(X[test_idx])
    return predictions


def render_grid(states: list[str], size: int, current: list[int] | None = None, goal: list[int] | None = None) -> str:
    items: list[str] = []
    current_xy = tuple(current[:2]) if isinstance(current, list) and len(current) >= 2 else None
    goal_xy = tuple(goal[:2]) if isinstance(goal, list) and len(goal) >= 2 else None
    for y in range(size - 1, -1, -1):
        for x in range(size):
            state = states[x * size + y]
            markers = ""
            if (x, y) == goal_xy:
                markers += '<span class="mark goal">G</span>'
            if (x, y) == current_xy:
                markers += '<span class="mark agent">A</span>'
            items.append(
                f'<div class="cell state-{html.escape(state)}" title="({x},{y}) {html.escape(state)}">'
                f'<span class="coord">{x},{y}</span><strong>{html.escape(state)}</strong>{markers}</div>'
            )
    return "".join(items)


def render_episode(run: Path, episode_id: str, rows: list[dict[str, Any]], size: int, layer: int, output_dir: Path) -> None:
    data_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(episode_id)} — Coordinate Belief v6</title>
<style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#f5f6f8;color:#15171a}} h1{{margin-bottom:4px}}
.meta{{color:#59636e;margin-bottom:14px}} .controls{{display:flex;gap:8px;align-items:center;margin:12px 0}}
button,input{{font:inherit}} .panels{{display:grid;grid-template-columns:repeat(4,minmax(210px,1fr));gap:14px}}
.panel{{background:white;border:1px solid #d9dee5;border-radius:10px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
.grid{{display:grid;grid-template-columns:repeat({size},1fr);aspect-ratio:1;gap:3px}} .cell{{position:relative;border-radius:5px;display:flex;align-items:center;justify-content:center;border:1px solid #cbd2da;min-width:0}}
.state-U{{background:#eceff3}} .state-F{{background:#dff4df}} .state-O{{background:#40464f;color:white}} .state-WALL{{background:#20242a;color:white}}
.coord{{position:absolute;left:3px;top:2px;font-size:9px;opacity:.55}} .mark{{position:absolute;right:3px;bottom:2px;font-size:10px;padding:1px 3px;border-radius:4px}}
.agent{{background:#ffd66b;color:#111}} .goal{{background:#77d5ff;color:#111}} pre{{white-space:pre-wrap;max-height:270px;overflow:auto;background:#111820;color:#e7edf4;padding:10px;border-radius:8px}}
@media(max-width:1050px){{.panels{{grid-template-columns:repeat(2,1fr)}}}} @media(max-width:620px){{.panels{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>{html.escape(episode_id)}</h1><div class="meta">Coordinate-Belief v6 · prompt_last · layer {layer} · 5-fold episode OOF cell probes</div>
<div class="controls"><button id="prev">◀</button><input id="slider" type="range" min="0" max="{max(0, len(rows)-1)}" value="0"><button id="next">▶</button><strong id="stepLabel"></strong></div>
<div class="panels">
<div class="panel"><h3>True map</h3><div id="trueGrid" class="grid"></div></div>
<div class="panel"><h3>Gold observable</h3><div id="goldGrid" class="grid"></div></div>
<div class="panel"><h3>Explicit coordinates</h3><div id="explicitGrid" class="grid"></div></div>
<div class="panel"><h3>Probe decoded</h3><div id="probeGrid" class="grid"></div></div>
</div>
<div class="panel" style="margin-top:14px"><h3>Step record</h3><pre id="details"></pre></div>
<script>
const DATA={data_json}; const SIZE={size};
function gridHtml(states,current,goal){{let out='';for(let y=SIZE-1;y>=0;y--)for(let x=0;x<SIZE;x++){{const s=states[x*SIZE+y]||'U';let marks='';if(goal&&x===goal[0]&&y===goal[1])marks+='<span class="mark goal">G</span>';if(current&&x===current[0]&&y===current[1])marks+='<span class="mark agent">A</span>';out+=`<div class="cell state-${{s}}" title="(${{x}},${{y}}) ${{s}}"><span class="coord">${{x}},${{y}}</span><strong>${{s}}</strong>${{marks}}</div>`;}}return out;}}
function show(i){{i=Math.max(0,Math.min(DATA.length-1,i));slider.value=i;const r=DATA[i];stepLabel.textContent=`step ${{r.step_id}} / ${{DATA.length-1}} · action=${{r.action||'N/A'}}`;trueGrid.innerHTML=gridHtml(r.true_cells,r.current_pos,r.goal);goldGrid.innerHTML=gridHtml(r.gold_cells,r.current_pos,r.goal);explicitGrid.innerHTML=gridHtml(r.explicit_cells,r.current_pos,r.goal);probeGrid.innerHTML=gridHtml(r.probe_cells,r.current_pos,r.goal);details.textContent=JSON.stringify(r.step_record,null,2);}}
slider.oninput=()=>show(+slider.value);prev.onclick=()=>show(+slider.value-1);next.onclick=()=>show(+slider.value+1);show(0);
</script></body></html>"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{episode_id}.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict coordinate-belief v6 OOF trajectory viewer.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episode", default=None)
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--position", default="prompt_last")
    parser.add_argument("--layer", default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--size", type=int, default=5)
    args = parser.parse_args()

    run = args.run.resolve()
    steps = read_jsonl(run / "steps.jsonl")
    targets = read_jsonl(run / "targets" / "targets.jsonl")
    target_by_key = {key(row): row for row in targets}
    step_by_key = {key(row): row for row in steps}

    act_dir = run / "activations"
    if not act_dir.exists():
        candidates = sorted(run.glob("activations*"))
        if not candidates:
            raise FileNotFoundError(f"No activations directory under {run}")
        act_dir = candidates[0]
    X_all = np.load(act_dir / "X.npy", mmap_mode="r")
    positions = load_positions(act_dir / "positions.json")
    layers = [int(x) for x in np.load(act_dir / "layers.npy").tolist()]
    meta = read_jsonl(act_dir / "meta.jsonl")
    if len(meta) != X_all.shape[0]:
        raise RuntimeError(f"Activation meta/X mismatch: {len(meta)} vs {X_all.shape[0]}")
    if args.position not in positions:
        raise RuntimeError(f"Position {args.position!r} unavailable; choices={positions}")
    p_idx = positions.index(args.position)
    layer = choose_layer(run, layers, args.layer)
    l_idx = layers.index(layer)

    joined_X: list[np.ndarray] = []
    joined_targets: list[dict[str, Any]] = []
    joined_steps: list[dict[str, Any]] = []
    joined_keys: list[tuple[str, int]] = []
    for i, meta_row in enumerate(meta):
        k = key(meta_row)
        if k not in target_by_key or k not in step_by_key:
            continue
        joined_X.append(np.asarray(X_all[i, p_idx, l_idx], dtype=np.float32))
        joined_targets.append(target_by_key[k])
        joined_steps.append(step_by_key[k])
        joined_keys.append(k)
    if not joined_X:
        raise RuntimeError("No activation/target/step keys joined")
    X = np.stack(joined_X)
    groups = np.array([k[0] for k in joined_keys], dtype=object)
    predictions = fit_oof(X, joined_targets, groups, args.size, args.folds)

    by_episode: dict[str, list[dict[str, Any]]] = {}
    for i, (episode_id, step_id) in enumerate(joined_keys):
        step = joined_steps[i]
        target = joined_targets[i]
        row = {
            "episode_id": episode_id,
            "step_id": step_id,
            "current_pos": step.get("current_pos"),
            "next_pos": step.get("next_pos"),
            "goal": step.get("goal") or step.get("goal_pos"),
            "action": step.get("action"),
            "true_cells": true_payload(target, args.size),
            "gold_cells": grid_payload(target, "gold_cell", args.size),
            "explicit_cells": parse_explicit(step, args.size),
            "probe_cells": predictions[i].tolist(),
            "step_record": step,
        }
        by_episode.setdefault(episode_id, []).append(row)

    if args.all_episodes:
        wanted = sorted(by_episode)
    elif args.episode:
        if args.episode not in by_episode:
            raise RuntimeError(f"Episode {args.episode} unavailable")
        wanted = [args.episode]
    else:
        wanted = [sorted(by_episode)[0]]

    output_dir = run / "trajectory_viewer_coordbelief_v6"
    for episode_id in wanted:
        rows = sorted(by_episode[episode_id], key=lambda r: r["step_id"])
        render_episode(run, episode_id, rows, args.size, layer, output_dir)
    links = "\n".join(f'<li><a href="{html.escape(ep)}.html">{html.escape(ep)}</a></li>' for ep in wanted)
    index = f"<!doctype html><meta charset='utf-8'><title>Coordinate-Belief v6 viewers</title><h1>Coordinate-Belief v6 viewers</h1><p>prompt_last · layer {layer}</p><ul>{links}</ul>"
    (output_dir / "index.html").write_text(index, encoding="utf-8")
    print(f"[viewer-v6] position={args.position} layer={layer} rows={len(joined_keys)}")
    print(f"[viewer-v6] saved={output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
