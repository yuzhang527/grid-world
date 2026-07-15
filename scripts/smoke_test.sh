#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
rm -rf data/generated/smoke_maps.jsonl data/generated/smoke_maps.manifest.json runs/mock_smoke
grid-world maps generate --config configs/maps/grid5x5_smoke.yaml --output data/generated/smoke_maps.jsonl
grid-world maps validate --maps data/generated/smoke_maps.jsonl
grid-world trajectories generate --config configs/experiments/mock_smoke.yaml   --maps data/generated/smoke_maps.jsonl --run runs/mock_smoke
grid-world trajectories validate --run runs/mock_smoke
grid-world trajectories summarize --run runs/mock_smoke
grid-world targets build --run runs/mock_smoke
echo "Smoke test completed successfully."
