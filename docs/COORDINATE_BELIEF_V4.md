# Coordinate-Belief v4

This is an **additive** experiment for the existing `grid-world` repository.

It does not replace the old explicit-grid condition. It creates a new condition:

- Model: `Qwen/Qwen2.5-32B-Instruct`
- Episodes: `200`
- Belief output: coordinate sets, not a 5×5 matrix
- Coordinate system: Cartesian, `x` right, `y` up, `(0,0)` bottom-left
- Unknown state: implicit; every coordinate omitted from `F` and `O` is `U`
- Downstream pipeline: existing targets, activation extraction, probe training,
  layer plots, behavior-quality report, and trajectory viewer

## Model output schema

```json
{
  "belief_coordinates": {
    "F": [[0, 0], [1, 0]],
    "O": [[0, 1]]
  },
  "action": "RIGHT"
}
```

The model is explicitly forbidden to output a matrix. Internally, the generator
also writes `parsed_belief_grid[y][x]` solely as a compatibility field for the
old target/probe/viewer pipeline.

## Install

From the repository root:

```bash
bash /path/to/deploy_coord_belief_v4.sh \
  /workspace/luoyuzhang/grid-world
```

The installer only adds new files:

```text
scripts/generate_coord_belief_v4.py
scripts/validate_coord_belief_v4.py
scripts/run_qwen25_32b_coord200_v4.sh
configs/maps/grid5x5_coordbelief_v4_200.yaml
configs/experiments/qwen25_32b_coordbelief_v4.yaml
docs/COORDINATE_BELIEF_V4.md
```

## Smoke test without a model

```bash
cd /workspace/luoyuzhang/grid-world

grid-world maps generate \
  --config configs/maps/grid5x5_coordbelief_v4_200.yaml \
  --output /tmp/coord_v4_maps.jsonl

python scripts/generate_coord_belief_v4.py \
  --maps /tmp/coord_v4_maps.jsonl \
  --run /tmp/coord_v4_smoke \
  --backend mock \
  --num-episodes 2 \
  --max-steps 8 \
  --overwrite

python scripts/validate_coord_belief_v4.py \
  --run /tmp/coord_v4_smoke \
  --strict-raw-format
```

## Full 32B / 200-episode run

```bash
cd /workspace/luoyuzhang/grid-world

MODEL=Qwen/Qwen2.5-32B-Instruct \
GPUS=0,1,2,3 \
NUM_EPISODES=200 \
RUN=runs/qwen25_32b_coordbelief_v4_200 \
STAGES=maps,generate,validate,targets,activations,probes,report,quality,viewer \
bash scripts/run_qwen25_32b_coord200_v4.sh
```

For a locally downloaded model:

```bash
MODEL=/workspace/models/Qwen2.5-32B-Instruct \
GPUS=0,1,2,3 \
bash scripts/run_qwen25_32b_coord200_v4.sh
```

## Resume by stage

Generation supports completed-episode resume:

```bash
STAGES=generate,validate \
bash scripts/run_qwen25_32b_coord200_v4.sh
```

Continue after trajectories are complete:

```bash
STAGES=targets,activations,probes,report,quality,viewer \
bash scripts/run_qwen25_32b_coord200_v4.sh
```

Continue only probe/report/viewer:

```bash
STAGES=probes,report,quality,viewer \
bash scripts/run_qwen25_32b_coord200_v4.sh
```

Force regeneration of the new run only:

```bash
GEN_OVERWRITE=1 \
STAGES=generate,validate \
bash scripts/run_qwen25_32b_coord200_v4.sh
```

The older run directories are not touched.

## Recommended first real check

Before all 200 episodes, run ten:

```bash
NUM_EPISODES=10 \
MAPS=data/generated/grid5x5_coordbelief_v4_200.jsonl \
RUN=runs/qwen25_32b_coordbelief_v4_smoke10 \
STAGES=maps,generate,validate \
bash scripts/run_qwen25_32b_coord200_v4.sh
```

Inspect raw responses:

```bash
python - "$RUN" <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ["RUN"]) / "steps.jsonl"
for line in path.open():
    row = json.loads(line)
    print(row["raw_response"])
    break
PY
```

The response should contain `belief_coordinates` and should not contain a
2-D `belief_grid`.

## Main outputs

```text
RUN/summary.json
RUN/analysis/behavior_quality/summary.md
RUN/probes_multigpu/summary.md (also available through RUN/probes/)
RUN/layer_curves/summary.md
RUN/trajectory_viewer_gallery/index.html
```

The `mean_current_belief_grid` activation position is deliberately retained as
a legacy segment name. Its text content is now coordinate sets rather than a
matrix, which makes direct comparison with the older explicit-grid condition
possible without changing the extractor interface.

