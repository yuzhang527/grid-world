# grid-world

`grid-world` is a clean, reproducible research pipeline for studying planning and
world-state representations in language models under partial observation.

```text
map dataset
  -> vLLM trajectory generation (single GPU or multi-GPU)
  -> validation and behavioral summaries
  -> gold belief/planning targets
  -> Transformer activation extraction
  -> CPU or GPU linear probes
  -> reports and heatmaps
```

The default task is a Cartesian grid world with exact adjacent-cell feedback.
The model returns JSON containing an updated `belief_grid` and the next action.

中文文档：[`docs/README_zh.md`](docs/README_zh.md)

## Design

- One stable `grid-world` CLI.
- Versioned `maps.jsonl`, `steps.jsonl`, `episodes.jsonl`, `targets.jsonl`, and manifests.
- Automatic vLLM episode sharding across GPUs.
- Tensor parallel mode for models that do not fit on one GPU.
- Exact activation replay from the stored rendered `prompt_text`.
- Gold belief labels reconstructed from environment feedback.
- Episode-grouped probe splits.
- GPU PyTorch multi-task linear probes and CPU sklearn probes.
- Resume, merge, de-duplication, validation, and legacy-run migration.

## Layout

```text
grid-world/
├── configs/
├── docs/
├── scripts/
├── src/grid_world/
│   ├── activations/
│   ├── env/
│   ├── evaluation/
│   ├── generation/
│   ├── probes/
│   ├── prompting/
│   ├── targets/
│   └── utils/
├── tests/
└── pyproject.toml
```

A run uses a fixed artifact contract:

```text
runs/qwen100/
├── resolved_config.yaml
├── maps.jsonl
├── steps.jsonl
├── episodes.jsonl
├── summary.json
├── manifest.json
├── shards/
├── targets/
├── activations/
└── probes/
```

## Installation

```bash
cd grid-world
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Activation extraction and GPU probes:

```bash
pip install -e ".[activation]"
```

vLLM generation:

```bash
# Install a vLLM build compatible with the server CUDA environment.
pip install vllm
pip install -e ".[generation]"
```

Development:

```bash
pip install -e ".[dev]"
pytest
```

## Smoke test

```bash
bash scripts/smoke_test.sh
```

This uses a shortest-path mock oracle and does not load an LLM.

## 1. Generate maps

```bash
grid-world maps generate \
  --config configs/maps/grid5x5_100.yaml \
  --output data/generated/grid5x5_100.jsonl

grid-world maps validate \
  --maps data/generated/grid5x5_100.jsonl

grid-world maps show \
  --maps data/generated/grid5x5_100.jsonl \
  --episode A_seed123
```

Coordinates are `[x,y]`, with `[0,0]` at the bottom-left.

## 2. Generate trajectories

One GPU:

```bash
grid-world trajectories generate \
  --config configs/experiments/qwen25_7b_strategy_a.yaml \
  --maps data/generated/grid5x5_100.jsonl \
  --run runs/qwen25_7b_100 \
  --gpus 0
```

Four GPUs, episode data parallelism:

```bash
grid-world trajectories generate \
  --config configs/experiments/qwen25_7b_strategy_a.yaml \
  --maps data/generated/grid5x5_100.jsonl \
  --run runs/qwen25_7b_100 \
  --gpus 0,1,2,3 \
  --parallel-mode data
```

The launcher deterministically shards episodes, starts one vLLM worker per GPU,
merges shard outputs, de-duplicates `(episode_id, step_id)`, and validates the run.
Resume is enabled by default.

Tensor parallelism:

```bash
grid-world trajectories generate \
  --config configs/experiments/qwen25_7b_strategy_a.yaml \
  --maps data/generated/grid5x5_100.jsonl \
  --run runs/large_model_tp4 \
  --gpus 0,1,2,3 \
  --parallel-mode tensor
```

Validate and summarize:

```bash
grid-world trajectories validate --run runs/qwen25_7b_100
grid-world trajectories summarize --run runs/qwen25_7b_100
```

## 3. Build gold targets

```bash
grid-world targets build --run runs/qwen25_7b_100
```

Target groups:

- `local`: correct adjacent-cell state;
- `cells`: correct partial-observation belief for every cell;
- `memory`: whether every cell has been observed;
- `planning`: shortest-path action values, distance reduction, revisits, and loops;
- `faithfulness`: mismatch between internal/action behavior and explicit belief output.

The true map and gold belief differ. An unobserved free cell is `F` in the true
map but `U` in the gold belief.

## 4. Extract activations

Quick run:

```bash
grid-world activations extract \
  --run runs/qwen25_7b_100 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --layers auto \
  --positions default \
  --device cuda:0 \
  --max-rows 20
```

Full layers:

```bash
grid-world activations extract \
  --run runs/qwen25_7b_100 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --layers all \
  --positions default \
  --device cuda:0
```

Default positions:

```text
prompt_last
mean_all_prompt
mean_last_feedback
after_last_feedback
mean_required_belief_updates
mean_available_actions
mean_current_belief_grid
after_current_belief_grid
mean_history
pre_action_token
first_action_token
```

The output is memory-mapped:

```text
X shape = [rows, positions, selected_layers, hidden_size]
```

## 5. Train probes

GPU PyTorch backend:

```bash
grid-world probes train \
  --run runs/qwen25_7b_100 \
  --groups local,cells,planning \
  --positions auto \
  --layers all \
  --backend torch \
  --device cuda:0 \
  --splits 20 \
  --epochs 60
```

For a fixed position/layer/split, all selected tasks are trained as independent
output blocks in one linear layer. This reduces repeated GPU launch overhead.

CPU sklearn backend:

```bash
grid-world probes train \
  --run runs/qwen25_7b_100 \
  --groups local,cells,planning \
  --positions auto \
  --layers auto \
  --backend sklearn \
  --device cpu \
  --splits 20
```

Metrics include accuracy, balanced accuracy, macro-F1, majority baselines, and
split standard deviation. Prefer macro-F1 for imbalanced labels.

## 6. Report

```bash
grid-world probes report --run runs/qwen25_7b_100
```

This writes `best_by_task.csv`, `group_summary.csv`, `summary.md`, and
layer-position heatmaps.

## 7. Full pipeline

```bash
grid-world pipeline run \
  --config configs/pipeline/qwen25_7b_100.yaml \
  --stages maps,generate,validate,targets,activations,probes,report
```

Each stage remains independently runnable.

## 8. Migrate the existing legacy run

```bash
grid-world migrate legacy-run \
  --source /workspace/luoyuzhang/grid-planner/outputs/logs/strategy_A_qwen_100_vllm_4gpu_merged \
  --run runs/qwen25_7b_100_migrated
```

Then:

```bash
grid-world trajectories validate --run runs/qwen25_7b_100_migrated
grid-world targets build --run runs/qwen25_7b_100_migrated
```

Precise activation replay requires the old step log to contain a complete prompt
field; the migrator maps legacy `prompt` to the new `prompt_text` field.

## 9. Recommended server workflow

```bash
MAPS=data/generated/grid5x5_100.jsonl
RUN=runs/qwen25_7b_100
MODEL=Qwen/Qwen2.5-7B-Instruct

grid-world maps generate \
  --config configs/maps/grid5x5_100.yaml \
  --output "$MAPS"

grid-world trajectories generate \
  --config configs/experiments/qwen25_7b_strategy_a.yaml \
  --maps "$MAPS" \
  --run "$RUN" \
  --gpus 0,1,2,3 \
  --parallel-mode data

grid-world trajectories validate --run "$RUN"
grid-world trajectories summarize --run "$RUN"
grid-world targets build --run "$RUN"

grid-world activations extract \
  --run "$RUN" \
  --model "$MODEL" \
  --layers all \
  --positions default \
  --device cuda:0

grid-world probes train \
  --run "$RUN" \
  --groups local,cells,planning \
  --positions auto \
  --layers all \
  --backend torch \
  --device cuda:0 \
  --splits 20 \
  --epochs 60

grid-world probes report --run "$RUN"
```

## Reproducibility

- One explicit map seed per episode.
- Temperature-zero generation by default.
- Exact rendered prompts stored in every step.
- Fixed episode-grouped probe splits.
- Stage manifests include config hashes and upstream paths.
- Targets and activations join on `(episode_id, step_id)`.
- Probe decodability does not by itself prove causal use.


## Diverse 5x5 experiment

See [`docs/DIVERSE_5X5.md`](docs/DIVERSE_5X5.md) for the balanced 1000-episode 5x5 experiment.
