#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit("Install scikit-learn: pip install scikit-learn") from exc

GOLD_RE = re.compile(r"^gold_cell_x(\d+)_y(\d+)_OFU$")
TRUE_RE = re.compile(r"^true_cell_x(\d+)_y(\d+)_FO$")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows

def ep_of(row: dict[str, Any]) -> str:
    for key in ("episode_id", "episode", "trajectory_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    return ""

def step_of(row: dict[str, Any]) -> int:
    for key in ("step_id", "step", "timestep", "turn_id"):
        if row.get(key) is not None:
            return int(row[key])
    raise KeyError("Missing step id")

def find_activation_dir(run: Path) -> Path:
    candidates = [
        run / "activations",
        run / "activations_A_multi",
        run / "activations_coord",
    ]
    for path in candidates:
        if (path / "X.npy").is_file() and (path / "meta.jsonl").is_file():
            return path
    raise FileNotFoundError(
        "Cannot find activations directory with X.npy and meta.jsonl. Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )

def load_positions(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, dict):
        if "positions" in value:
            return [str(x) for x in value["positions"]]
        return [str(k) for k, _ in sorted(value.items(), key=lambda kv: int(kv[1]))]
    raise RuntimeError(f"Unsupported positions.json schema: {type(value)}")

def parse_layer_arg(value: str, layers: list[int]) -> tuple[int, int]:
    if value == "auto":
        # For 32B, use a middle-late layer unless caller specifies a validated layer.
        target = layers[len(layers) * 2 // 3]
    else:
        target = int(value)
    if target in layers:
        return layers.index(target), target
    if 0 <= target < len(layers):
        return target, layers[target]
    raise ValueError(f"Layer {target} not available; layers={layers}")

def coord_map_from_row(row: dict[str, Any], prefix: str, width: int, height: int) -> dict[str, str]:
    pattern = GOLD_RE if prefix == "gold" else TRUE_RE
    result: dict[str, str] = {}
    for key, value in row.items():
        match = pattern.match(key)
        if match:
            x, y = map(int, match.groups())
            result[f"{x},{y}"] = str(value).upper()
    expected = width * height
    if len(result) != expected:
        raise RuntimeError(
            f"{prefix} map coverage {len(result)}/{expected} for "
            f"{ep_of(row)}/{step_of(row)}"
        )
    return result

def explicit_map_from_step(row: dict[str, Any], width: int, height: int) -> dict[str, str]:
    obj = row.get("parsed_belief_coordinates")
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"steps_coord_v6 missing parsed_belief_coordinates for "
            f"{ep_of(row)}/{step_of(row)}"
        )
    result = {f"{x},{y}": "U" for y in range(height) for x in range(width)}
    seen: dict[str, str] = {}
    for label in ("F", "O"):
        values = obj.get(label, [])
        if not isinstance(values, list):
            raise RuntimeError(f"belief_coordinates[{label}] is not a list")
        for coord in values:
            if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                raise RuntimeError(f"Invalid coordinate {coord!r}")
            x, y = int(coord[0]), int(coord[1])
            key = f"{x},{y}"
            if key not in result:
                raise RuntimeError(f"Out-of-bounds coordinate {(x, y)}")
            if key in seen and seen[key] != label:
                raise RuntimeError(f"F/O overlap at {(x, y)}")
            seen[key] = label
            result[key] = label
    return result

def prompt_of(row: dict[str, Any]) -> str:
    for key in ("prompt_text", "prompt", "rendered_prompt"):
        if isinstance(row.get(key), str):
            return row[key]
    return ""

def output_of(row: dict[str, Any]) -> str:
    for key in ("raw_response_text", "raw_response", "response"):
        if isinstance(row.get(key), str):
            return row[key]
    return ""

def position_of(row: dict[str, Any]) -> list[int] | None:
    for key in ("current_pos", "agent_pos_before", "position"):
        value = row.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return [int(value[0]), int(value[1])]
    return None

def encode_classes(classes: list[str], labels: np.ndarray) -> np.ndarray:
    lookup = {label: i for i, label in enumerate(classes)}
    return np.asarray([lookup[str(x)] for x in labels], dtype=np.int64)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--position", default="prompt_last")
    parser.add_argument("--layer", default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()

    run = args.run.resolve()
    steps_path = run / "steps_coord_v6.jsonl"
    targets_path = run / "targets_coord_v6" / "targets.jsonl"
    if not steps_path.is_file() or not targets_path.is_file():
        raise SystemExit(
            "Run audit/rebuild first:\n"
            f"python scripts/audit_rebuild_coordbelief_v6.py --run {run} --write"
        )

    steps = read_jsonl(steps_path)
    targets = read_jsonl(targets_path)
    steps_by_key = {(ep_of(r), step_of(r)): r for r in steps}
    targets_by_key = {(ep_of(r), step_of(r)): r for r in targets}

    episode_keys = sorted(
        [key for key in steps_by_key if key[0] == args.episode],
        key=lambda key: key[1],
    )
    if not episode_keys:
        matches = sorted({ep for ep, _ in steps_by_key if args.episode in ep})
        if len(matches) == 1:
            args.episode = matches[0]
            episode_keys = sorted(
                [key for key in steps_by_key if key[0] == args.episode],
                key=lambda key: key[1],
            )
        else:
            raise SystemExit(
                f"Episode {args.episode!r} not found. Substring matches={matches[:20]}"
            )

    width = height = 5
    first_target = targets_by_key[episode_keys[0]]
    xs = [int(m.group(1)) for k in first_target for m in [GOLD_RE.match(k)] if m]
    ys = [int(m.group(2)) for k in first_target for m in [GOLD_RE.match(k)] if m]
    if xs and ys:
        width, height = max(xs) + 1, max(ys) + 1

    act_dir = find_activation_dir(run)
    X = np.load(act_dir / "X.npy", mmap_mode="r")
    meta = read_jsonl(act_dir / "meta.jsonl")
    positions = load_positions(act_dir / "positions.json")
    raw_layers = np.load(act_dir / "layers.npy").tolist()
    layers = [int(x) for x in raw_layers]

    if args.position not in positions:
        raise SystemExit(
            f"Position {args.position!r} not available. positions={positions}\n"
            "Do not silently alias mean_current_belief_grid to coordinate belief."
        )
    pos_index = positions.index(args.position)
    layer_index, layer_value = parse_layer_arg(args.layer, layers)

    if len(meta) != X.shape[0]:
        raise RuntimeError(f"meta rows={len(meta)} but X rows={X.shape[0]}")
    if X.ndim != 4:
        raise RuntimeError(f"Expected X[rows,positions,layers,hidden], got {X.shape}")

    activation_by_key: dict[tuple[str, int], int] = {}
    for i, row in enumerate(meta):
        key = (ep_of(row), step_of(row))
        if key in activation_by_key:
            raise RuntimeError(f"Duplicate activation key {key}")
        activation_by_key[key] = i

    common_keys = sorted(
        set(activation_by_key) & set(targets_by_key),
        key=lambda key: (key[0], key[1]),
    )
    if not common_keys:
        raise RuntimeError("No activation/target keys align by (episode_id, step_id)")

    selected_available = [key for key in episode_keys if key in activation_by_key]
    missing_selected = [key for key in episode_keys if key not in activation_by_key]
    if missing_selected:
        print(
            f"[viewer/v6] selected episode has {len(missing_selected)} steps without "
            "activations; they will show decoded=N/A."
        )

    row_indices = np.asarray([activation_by_key[key] for key in common_keys], dtype=np.int64)
    features = np.asarray(X[row_indices, pos_index, layer_index, :], dtype=np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError("Activation features contain NaN/Inf")

    groups = np.asarray([key[0] for key in common_keys])
    unique_groups = sorted(set(groups.tolist()))
    folds = min(args.folds, len(unique_groups))
    if folds < 2:
        raise RuntimeError("Need at least two episodes for held-out decoding")

    splitter = GroupKFold(n_splits=folds)
    selected_split: tuple[np.ndarray, np.ndarray] | None = None
    for train_idx, test_idx in splitter.split(features, groups=groups):
        test_groups = set(groups[test_idx].tolist())
        if args.episode in test_groups:
            selected_split = train_idx, test_idx
            break
    if selected_split is None:
        raise RuntimeError(f"Could not assign episode {args.episode} to a fold")

    train_idx, test_idx = selected_split
    test_key_to_local = {common_keys[i]: j for j, i in enumerate(test_idx)}
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_idx])
    X_test = scaler.transform(features[test_idx])

    predictions: dict[tuple[str, int], dict[str, dict[str, Any]]] = {
        common_keys[i]: {} for i in test_idx
    }

    print(
        f"[viewer/v6] activations={act_dir} X={X.shape}\n"
        f"[viewer/v6] position={args.position} layer={layer_value} "
        f"train_rows={len(train_idx)} test_rows={len(test_idx)}"
    )

    tasks = [
        (x, y, f"gold_cell_x{x}_y{y}_OFU")
        for y in range(height) for x in range(width)
    ]

    for task_no, (x, y, task) in enumerate(tasks, 1):
        labels = np.asarray([str(targets_by_key[key][task]) for key in common_keys])
        train_labels = labels[train_idx]
        classes = sorted(set(train_labels.tolist()))

        if len(classes) == 1:
            pred_labels = np.asarray([classes[0]] * len(test_idx), dtype=object)
            probs = np.ones((len(test_idx), 1), dtype=np.float64)
            model_classes = classes
            decoder = "constant"
        else:
            y_train = encode_classes(classes, train_labels)
            model = LogisticRegression(
                max_iter=args.max_iter,
                class_weight="balanced",
                solver="lbfgs",
            )
            model.fit(X_train, y_train)
            pred_ids = model.predict(X_test)
            probs = model.predict_proba(X_test)
            model_classes = [classes[int(i)] for i in model.classes_.tolist()]
            pred_labels = np.asarray([classes[int(i)] for i in pred_ids], dtype=object)
            decoder = "logistic_regression"

        test_targets = labels[test_idx]
        acc = float(accuracy_score(test_targets, pred_labels))
        f1 = float(f1_score(test_targets, pred_labels, average="macro", zero_division=0))
        print(
            f"[viewer/v6] probe {task_no:02d}/{len(tasks)} {task} "
            f"decoder={decoder} acc={acc:.3f} macro_f1={f1:.3f}"
        )

        for local_i, global_i in enumerate(test_idx):
            key = common_keys[global_i]
            probability_map = {
                model_classes[j]: float(probs[local_i, j])
                for j in range(len(model_classes))
            }
            pred = str(pred_labels[local_i])
            target = str(test_targets[local_i])
            predictions[key][f"{x},{y}"] = {
                "label": pred,
                "target": target,
                "confidence": float(max(probability_map.values())),
                "probabilities": probability_map,
                "correct": pred == target,
                "decoder": decoder,
            }

    payload_steps = []
    true_map_reference: dict[str, str] | None = None
    for key in episode_keys:
        step_row = steps_by_key[key]
        target_row = targets_by_key[key]
        gold = coord_map_from_row(target_row, "gold", width, height)
        true_map = coord_map_from_row(target_row, "true", width, height)
        explicit = explicit_map_from_step(step_row, width, height)
        if true_map_reference is None:
            true_map_reference = true_map
        elif true_map != true_map_reference:
            raise RuntimeError(f"True map changed within episode at {key}")

        decoded = predictions.get(key)
        payload_steps.append(
            {
                "step_id": key[1],
                "current_pos": position_of(step_row),
                "action": step_row.get("parsed_action", step_row.get("action")),
                "gold": gold,
                "explicit": explicit,
                "decoded": decoded,
                "prompt": prompt_of(step_row),
                "output": output_of(step_row),
                "feedback_before_action": step_row.get("feedback_before_action"),
                "repaired": bool(step_row.get("repaired")),
                "parse_error": bool(step_row.get("parse_error")),
            }
        )

    assert true_map_reference is not None
    payload = {
        "schema": "coordbelief-trajectory-viewer-v6",
        "episode": args.episode,
        "width": width,
        "height": height,
        "true_map": true_map_reference,
        "activation": {"position": args.position, "layer": layer_value},
        "steps": payload_steps,
    }

    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coordinate-Belief v6 · {html.escape(args.episode)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#f5f5f5;color:#111}}
header,.controls,.panel{{background:white;border:1px solid #ccc;border-radius:10px;padding:14px;margin-bottom:14px}}
.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.maps{{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:12px}}
.mapPanel{{background:white;border:1px solid #ccc;border-radius:10px;padding:12px}}
.grid{{display:grid;gap:3px;align-items:stretch}}
.cell{{min-height:52px;border:1px solid #999;border-radius:5px;padding:4px;font-weight:700;display:flex;flex-direction:column;justify-content:center;align-items:center;position:relative}}
.cell small{{font-weight:400;font-size:10px}}
.F{{background:#dff5df}} .O{{background:#ffdede}} .U{{background:#eee}} .NA{{background:#fff3cd}}
.agent{{outline:4px solid #1d4ed8}} .wrong{{box-shadow:inset 0 0 0 3px #dc2626}}
pre{{white-space:pre-wrap;max-height:360px;overflow:auto;background:#111;color:#eee;padding:12px;border-radius:8px}}
.axis{{font-size:11px;color:#555}}
@media(max-width:1100px){{.maps{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<header>
<h1>Coordinate-Belief trajectory viewer v6</h1>
<div id="meta"></div>
</header>
<div class="controls">
<button id="prev">Previous</button><button id="next">Next</button>
<input id="slider" type="range" min="0" max="{max(0,len(payload_steps)-1)}" value="0" style="min-width:360px">
<span id="badge"></span>
</div>
<div class="maps">
<div class="mapPanel"><h3>True map</h3><div id="trueMap"></div></div>
<div class="mapPanel"><h3>Gold observable</h3><div id="goldMap"></div></div>
<div class="mapPanel"><h3>Explicit coordinate belief</h3><div id="explicitMap"></div></div>
<div class="mapPanel"><h3>Probe-decoded gold belief</h3><div id="decodedMap"></div></div>
</div>
<div class="panel"><h3>Diagnostics</h3><pre id="diag"></pre></div>
<div class="panel"><h3>Feedback visible before action</h3><pre id="feedback"></pre></div>
<div class="panel"><h3>Prompt</h3><pre id="prompt"></pre></div>
<div class="panel"><h3>Model output</h3><pre id="output"></pre></div>
<script>
const DATA={data_json};
let index=0;
const slider=document.getElementById('slider');
function cellValue(map,key,decoded){{
  if(decoded){{
    const r=map ? map[key] : null;
    if(!r) return {{label:'N/A', extra:'no activation', wrong:false}};
    return {{label:r.label, extra:`${{Math.round(100*r.confidence)}}% · gold=${{r.target}}`, wrong:!r.correct}};
  }}
  return {{label:map[key], extra:key, wrong:false}};
}}
function renderMap(id,map,decoded=false){{
  const root=document.getElementById(id); root.innerHTML='';
  const grid=document.createElement('div'); grid.className='grid';
  grid.style.gridTemplateColumns=`repeat(${{DATA.width}},1fr)`;
  const step=DATA.steps[index];
  for(let y=DATA.height-1;y>=0;y--){{
    for(let x=0;x<DATA.width;x++){{
      const key=`${{x}},${{y}}`; const r=cellValue(map,key,decoded);
      const el=document.createElement('div');
      const cls=(r.label==='N/A'?'NA':r.label);
      el.className=`cell ${{cls}}${{r.wrong?' wrong':''}}`;
      if(step.current_pos && step.current_pos[0]===x && step.current_pos[1]===y) el.className+=' agent';
      el.innerHTML=`<div>${{r.label}}</div><small>(${{x}},${{y}}) · ${{r.extra}}</small>`;
      grid.appendChild(el);
    }}
  }}
  root.appendChild(grid);
}}
function render(){{
  const step=DATA.steps[index]; slider.value=index;
  document.getElementById('badge').textContent=`Step ${{index+1}}/${{DATA.steps.length}} · id=${{step.step_id}} · action=${{step.action}}`;
  document.getElementById('meta').textContent=`episode=${{DATA.episode}} · Cartesian origin bottom-left · x→right · y→up · activation=${{DATA.activation.position}}/L${{DATA.activation.layer}}`;
  renderMap('trueMap',DATA.true_map,false);
  renderMap('goldMap',step.gold,false);
  renderMap('explicitMap',step.explicit,false);
  renderMap('decodedMap',step.decoded,true);
  const wrong=[];
  if(step.decoded){{
    for(const [coord,r] of Object.entries(step.decoded)){{
      if(!r.correct) wrong.push(`${{coord}}: pred=${{r.label}} gold=${{r.target}} conf=${{r.confidence.toFixed(3)}}`);
    }}
  }} else wrong.push('Probe decoding unavailable: this step has no extracted activation.');
  document.getElementById('diag').textContent=wrong.length?wrong.join(String.fromCharCode(10)):'All decoded cells match gold.';
  document.getElementById('feedback').textContent=JSON.stringify(step.feedback_before_action,null,2);
  document.getElementById('prompt').textContent=step.prompt||'';
  document.getElementById('output').textContent=step.output||'';
}}
document.getElementById('prev').onclick=()=>{{index=Math.max(0,index-1);render();}};
document.getElementById('next').onclick=()=>{{index=Math.min(DATA.steps.length-1,index+1);render();}};
slider.oninput=()=>{{index=Number(slider.value);render();}};
render();
</script>
</body></html>"""

    out_dir = run / "trajectory_viewer_coord_v6"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.episode}.html"
    out_path.write_text(page, encoding="utf-8")
    print(f"[viewer/v6] output={out_path}")

if __name__ == "__main__":
    main()

