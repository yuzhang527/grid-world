# grid-world 中文指南

这个新仓库把原来的地图、vLLM 多卡轨迹、activation 和 probe 脚本整理成统一 CLI。

## 安装

```bash
cd grid-world
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[activation]"
# vLLM 请按服务器 CUDA 环境安装，然后：
pip install -e ".[generation]"
```

## 完整命令

```bash
MAPS=data/generated/grid5x5_100.jsonl
RUN=runs/qwen25_7b_100
MODEL=Qwen/Qwen2.5-7B-Instruct

grid-world maps generate   --config configs/maps/grid5x5_100.yaml   --output "$MAPS"

grid-world trajectories generate   --config configs/experiments/qwen25_7b_strategy_a.yaml   --maps "$MAPS"   --run "$RUN"   --gpus 0,1,2,3   --parallel-mode data

grid-world trajectories validate --run "$RUN"
grid-world trajectories summarize --run "$RUN"
grid-world targets build --run "$RUN"

grid-world activations extract   --run "$RUN"   --model "$MODEL"   --layers all   --positions default   --device cuda:0

grid-world probes train   --run "$RUN"   --groups local,cells,planning   --positions auto   --layers all   --backend torch   --device cuda:0   --splits 20   --epochs 60

grid-world probes report --run "$RUN"
```

## 多卡逻辑

`--parallel-mode data` 会把独立 episode 分给多张卡，每张卡启动一个 vLLM
实例。Qwen2.5-7B 能放入单卡时优先使用这个模式。模型单卡放不下时使用
`--parallel-mode tensor`。

## Gold 标签

gold 是环境已知的监督答案。probe 的输入是 hidden state，训练目标是 gold
belief 或 planning label。它测试这些信息能否从 hidden state 中被线性读出。

## 迁移现有结果

```bash
grid-world migrate legacy-run   --source /workspace/luoyuzhang/grid-planner/outputs/logs/strategy_A_qwen_100_vllm_4gpu_merged   --run runs/qwen25_7b_100_migrated
```

迁移后可以直接构造 targets。activation 重放要求旧日志中有完整 `prompt`；
迁移工具会将它映射成 `prompt_text`。
