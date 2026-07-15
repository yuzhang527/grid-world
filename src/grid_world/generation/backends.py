from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import Any
from grid_world.env.belief import belief_to_rows
from grid_world.env.grid import GridSpec
from grid_world.env.planning import shortest_actions
from grid_world.prompting.strategy_a import render_plain

class GenerationBackend(ABC):
    @abstractmethod
    def render(self, messages: list[dict[str, str]]) -> str: ...
    @abstractmethod
    def generate(self, prompts: list[str]) -> list[str]: ...

class MockOracleBackend(GenerationBackend):
    def __init__(self):
        self.contexts: list[dict[str, Any]] = []
    def set_contexts(self, contexts: list[dict[str, Any]]) -> None:
        self.contexts = contexts
    def render(self, messages: list[dict[str, str]]) -> str:
        return render_plain(messages)
    def generate(self, prompts: list[str]) -> list[str]:
        if len(prompts) != len(self.contexts):
            raise RuntimeError("Mock contexts are not aligned with prompts")
        outputs = []
        for context in self.contexts:
            spec: GridSpec = context["spec"]
            actions = shortest_actions(spec, context["position"])
            action = actions[0] if actions else context["available_actions"][0]
            outputs.append(json.dumps({
                "thought": "Follow a shortest route.",
                "nl_obstacles": "Updated from exact adjacent feedback.",
                "belief_grid": belief_to_rows(context["gold_belief"], spec.size),
                "action": action,
            }))
        return outputs

class VLLMBackend(GenerationBackend):
    def __init__(self, config: dict[str, Any]):
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("Install generation dependencies with pip install -e '.[generation]'") from exc
        model_cfg, gen_cfg = config["model"], config["generation"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name"],
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        self.llm = LLM(
            model=model_cfg["name"],
            dtype=model_cfg.get("dtype", "auto"),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            tensor_parallel_size=int(config.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(gen_cfg.get("gpu_memory_utilization", 0.90)),
            max_model_len=int(gen_cfg.get("max_model_len", 4096)),
            max_num_seqs=int(gen_cfg.get("max_num_seqs", 32)),
            enforce_eager=bool(gen_cfg.get("enforce_eager", False)),
        )
        kwargs = {
            "temperature": float(gen_cfg.get("temperature", 0.0)),
            "top_p": float(gen_cfg.get("top_p", 1.0)),
            "max_tokens": int(gen_cfg.get("max_tokens", 256)),
        }
        if gen_cfg.get("seed") is not None:
            kwargs["seed"] = int(gen_cfg["seed"])
        self.sampling_params = SamplingParams(**kwargs)
        self.batch_size = int(gen_cfg.get("batch_size", 32))
    def render(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    def generate(self, prompts: list[str]) -> list[str]:
        texts = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            outputs = self.llm.generate(chunk, self.sampling_params, use_tqdm=False)
            texts.extend(item.outputs[0].text for item in outputs)
        return texts
