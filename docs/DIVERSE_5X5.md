# Diverse 5x5 experiment

This experiment replaces the fixed `[0,0] -> [4,4]` map distribution with a
balanced and stratified 5x5 dataset.

## Distribution

- 1000 unique layouts;
- randomized starts and goals;
- minimum Manhattan distance of 4;
- equal `NE`, `NW`, `SE`, and `SW` goal directions;
- easy, medium, and hard obstacle/detour strata;
- connectivity validation;
- duplicate-layout rejection.

Difficulty uses the shortest-path detour:

```text
detour = shortest_path_length - Manhattan(start, goal)
```

The default quotas are:

```text
easy   30%: 2-3 obstacles, detour 0
medium 45%: 4-6 obstacles, detour 2
hard   25%: 6-8 obstacles, detour >= 4
```

## Generate and inspect maps

```bash
grid-world maps generate \
  --config configs/maps/grid5x5_diverse_1000.yaml \
  --output data/generated/grid5x5_diverse_1000.jsonl

grid-world maps validate \
  --maps data/generated/grid5x5_diverse_1000.jsonl

grid-world maps summarize \
  --maps data/generated/grid5x5_diverse_1000.jsonl
```

A machine-readable summary is also saved to:

```text
data/generated/grid5x5_diverse_1000.summary.json
```

## Run generation through reports

```bash
grid-world pipeline run \
  --config configs/pipeline/qwen25_7b_diverse5x5_1000.yaml \
  --stages maps,generate,validate,targets,activations,probes,report
```

For safer server operation, run stages separately:

```bash
CONFIG=configs/pipeline/qwen25_7b_diverse5x5_1000.yaml
RUN=runs/qwen25_7b_diverse5x5_1000
MAPS=data/generated/grid5x5_diverse_1000.jsonl
MODEL=Qwen/Qwen2.5-7B-Instruct

grid-world pipeline run --config "$CONFIG" \
  --stages maps,generate,validate,targets

grid-world activations extract \
  --run "$RUN" \
  --model "$MODEL" \
  --layers auto \
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

The new run directory is independent of the original fixed-distribution run, so
the two reports can be compared directly.
