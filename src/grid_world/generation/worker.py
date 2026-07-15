from __future__ import annotations
import argparse, json
from pathlib import Path
from grid_world.generation.backends import MockOracleBackend, VLLMBackend
from grid_world.generation.runner import run_episodes
from grid_world.utils.io import read_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--maps", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    name = config.get("model", {}).get("backend", "vllm")
    backend = MockOracleBackend() if name == "mock" else VLLMBackend(config)
    run_episodes(map_rows=read_jsonl(args.maps), output_dir=args.output,
                 config=config, backend=backend)
if __name__ == "__main__":
    main()
