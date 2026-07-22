# Large-model and multi-GPU probe upgrade

This upgrade adds two independent capabilities:

1. **Model-parallel activation extraction** for Qwen2.5-32B/72B.
2. **Task-parallel multi-GPU probe training**.

## Why Qwen2.5-32B/72B

Use the same model family as the existing 7B experiment so that model scale is
the main changed variable. The generated 32B and 72B configs clone the current
7B experiment config and replace only the checkpoint name.

Create configs:

```bash
python scripts/make_large_model_configs.py
```

## Dependencies

```bash
python -m pip install -U "transformers>=4.45" accelerate pandas scikit-learn tabulate
```

For scientific comparability, BF16 is recommended. Avoid 4-bit activation
extraction unless quantization itself is an intended experimental variable.

## Recommended hardware modes

- 32B: commonly practical on 4 GPUs with tensor/model parallelism.
- 72B: usually needs 8 medium-memory GPUs or 4 high-memory GPUs.
- Exact feasibility depends on GPU memory, prompt length, KV cache settings,
  vLLM version, and whether CPU offload is allowed.

## Generate trajectories

32B:

```bash
MODEL_SIZE=32b \
GPUS=0,1,2,3 \
MAPS=data/generated/grid5x5_diverse_1000.jsonl \
RUN=runs/qwen25_32b_diverse5x5_1000 \
STAGES=generate,validate,targets \
bash scripts/run_qwen25_large.sh
```

72B:

```bash
MODEL_SIZE=72b \
GPUS=0,1,2,3,4,5,6,7 \
MAPS=data/generated/grid5x5_diverse_1000.jsonl \
RUN=runs/qwen25_72b_diverse5x5_1000 \
STAGES=generate,validate,targets \
bash scripts/run_qwen25_large.sh
```

The trajectory command uses `--parallel-mode tensor`, so all listed GPUs host
one logical model. This is different from data parallel generation, where each
GPU would need a complete model replica.

## Extract activations across GPUs

```bash
python scripts/extract_activations_model_parallel.py \
  --run "$RUN" \
  --model Qwen/Qwen2.5-32B-Instruct \
  --gpus 0,1,2,3 \
  --device-map balanced \
  --dtype bf16 \
  --layers all \
  --positions default \
  --batch-size 1 \
  --overwrite
```

For 72B:

```bash
python scripts/extract_activations_model_parallel.py \
  --run "$RUN" \
  --model Qwen/Qwen2.5-72B-Instruct \
  --gpus 0,1,2,3,4,5,6,7 \
  --device-map balanced \
  --dtype bf16 \
  --layers all \
  --positions default \
  --batch-size 1 \
  --overwrite
```

Optional per-device limits use **local visible indices** after
`CUDA_VISIBLE_DEVICES` remapping:

```bash
--max-memory '0=75GiB,1=75GiB,2=75GiB,3=75GiB,cpu=256GiB'
```

The extractor uses hooks and transfers only pooled vectors to CPU; it does not
retain every token from every layer.

## Train probes on multiple GPUs

```bash
python scripts/train_probes_multigpu.py \
  --run "$RUN" \
  --groups local,cells,planning \
  --positions auto \
  --layers all \
  --gpus 0,1,2,3 \
  --splits 20 \
  --epochs 60 \
  --output-subdir probes_multigpu \
  --overwrite
```

For eight GPUs, use:

```bash
--gpus 0,1,2,3,4,5,6,7
```

Each GPU receives different `(position, layer, split)` jobs. This is deliberate:
the probe heads are tiny and independent, so task parallelism avoids DDP
gradient synchronization overhead.

Outputs:

```text
RUN/probes_multigpu/
├── probe_results_splits.csv
├── probe_results.csv
├── best_by_task.csv
├── group_summary.csv
├── summary.md
├── splits.json
└── jobs/
```

Resume interrupted training by running the same command without `--overwrite`.
Completed job files are reused by default.

CPU smoke test:

```bash
python scripts/train_probes_multigpu.py \
  --run "$RUN" \
  --groups planning \
  --positions prompt_last \
  --layers 0 \
  --gpus cpu \
  --splits 2 \
  --epochs 2 \
  --output-subdir probes_smoke \
  --overwrite
```

