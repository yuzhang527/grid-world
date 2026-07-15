#!/usr/bin/env bash
set -euo pipefail
RUN="${RUN:?Set RUN to the run directory}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
grid-world targets build --run "$RUN"
grid-world activations extract --run "$RUN" --model "$MODEL"   --layers all --positions default --device "${ACTIVATION_GPU:-cuda:0}"
grid-world probes train --run "$RUN" --groups local,cells,planning   --positions auto --layers all --backend torch --device "${PROBE_GPU:-cuda:0}"   --splits 20 --epochs 60
grid-world probes report --run "$RUN"
