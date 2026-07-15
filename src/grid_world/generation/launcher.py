from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path
from grid_world.config import load_yaml, save_yaml
from grid_world.evaluation.validate import validate_run
from grid_world.generation.backends import MockOracleBackend
from grid_world.generation.merge import merge_run
from grid_world.generation.runner import run_episodes
from grid_world.utils.io import read_jsonl, write_jsonl

def _completed_ids(run: Path) -> set[str]:
    return {str(x["episode_id"]) for x in read_jsonl(run / "episodes.jsonl")} if (run / "episodes.jsonl").exists() else set()

def _spawn(config_path: Path, maps_path: Path, output: Path, visible: str):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible
    output.mkdir(parents=True, exist_ok=True)
    log = (output / "worker.log").open("w", encoding="utf-8")
    process = subprocess.Popen([
        sys.executable, "-m", "grid_world.generation.worker",
        "--config-json", str(config_path), "--maps", str(maps_path), "--output", str(output)
    ], env=env, stdout=log, stderr=subprocess.STDOUT)
    return process, log

def generate_trajectories(*, config_path: str | Path, maps_path: str | Path,
                          run_dir: str | Path, gpus: str = "0",
                          parallel_mode: str = "data", resume: bool = True):
    config, run = load_yaml(config_path), Path(run_dir)
    if run.exists() and not resume:
        shutil.rmtree(run)
    run.mkdir(parents=True, exist_ok=True)
    save_yaml(run / "resolved_config.yaml", config)
    all_maps = read_jsonl(maps_path)
    write_jsonl(run / "maps.jsonl", all_maps)
    pending = [x for x in all_maps if str(x["episode_id"]) not in (_completed_ids(run) if resume else set())]
    if not pending:
        steps, episodes = merge_run(run)
        validate_run(run)
        return {"episodes": len(episodes), "steps": len(steps), "new_episodes": 0}
    backend_name = str(config.get("model", {}).get("backend", "vllm"))
    gpu_ids = [x.strip() for x in gpus.split(",") if x.strip()]
    if backend_name == "mock":
        run_episodes(map_rows=pending, output_dir=run / "shards" / "shard_0",
                     config=config, backend=MockOracleBackend())
    elif parallel_mode == "tensor":
        shard = run / "shards" / "shard_0"
        shard.mkdir(parents=True, exist_ok=True)
        write_jsonl(shard / "maps.jsonl", pending)
        worker_config = dict(config)
        worker_config["tensor_parallel_size"] = len(gpu_ids)
        cfg_path = shard / "worker_config.json"
        cfg_path.write_text(json.dumps(worker_config, indent=2), encoding="utf-8")
        process, log = _spawn(cfg_path, shard / "maps.jsonl", shard, ",".join(gpu_ids))
        code = process.wait(); log.close()
        if code:
            raise RuntimeError(f"vLLM worker failed; see {shard / 'worker.log'}")
    elif parallel_mode == "data":
        shards = [[] for _ in gpu_ids]
        for i, row in enumerate(pending):
            shards[i % len(shards)].append(row)
        processes = []
        for i, (gpu, rows) in enumerate(zip(gpu_ids, shards)):
            if not rows:
                continue
            shard = run / "shards" / f"shard_{i}"
            if shard.exists():
                shutil.rmtree(shard)
            shard.mkdir(parents=True)
            write_jsonl(shard / "maps.jsonl", rows)
            worker_config = dict(config)
            worker_config["tensor_parallel_size"] = 1
            cfg_path = shard / "worker_config.json"
            cfg_path.write_text(json.dumps(worker_config, indent=2), encoding="utf-8")
            process, log = _spawn(cfg_path, shard / "maps.jsonl", shard, gpu)
            processes.append((process, log, shard))
        failures = []
        for process, log, shard in processes:
            code = process.wait(); log.close()
            if code:
                failures.append(str(shard / "worker.log"))
        if failures:
            raise RuntimeError("Workers failed:\n" + "\n".join(failures))
    else:
        raise ValueError("parallel_mode must be data or tensor")
    steps, episodes = merge_run(run)
    validate_run(run)
    return {"episodes": len(episodes), "steps": len(steps), "new_episodes": len(pending)}
