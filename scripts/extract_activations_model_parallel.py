#!/usr/bin/env python3
"""
Model-parallel activation extraction for the grid-world repository.

This extractor is intended for checkpoints such as Qwen2.5-32B/72B that may
not fit on one GPU. It uses Hugging Face Accelerate's device_map placement,
registers hooks only at requested hidden-state indices, pools requested token
positions on the owning device, and transfers only pooled vectors to CPU.

Output contract:
  RUN/<output_subdir>/
    X.npy                 [rows, positions, layers, hidden_size], float16
    position_mask.npy     [rows, positions], bool
    layers.npy            [layers], int64
    positions.json
    meta.jsonl
    manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_POSITIONS = [
    "mean_last_feedback",
    "mean_current_belief_grid",
    "pre_action_token",
    "prompt_last",
]

ALL_POSITIONS = [
    "prompt_last",
    "mean_all_prompt",
    "mean_last_feedback",
    "after_last_feedback",
    "mean_required_belief_updates",
    "after_required_belief_updates",
    "mean_available_actions",
    "after_available_actions",
    "mean_current_belief_grid",
    "after_current_belief_grid",
    "mean_history",
    "after_history",
    "pre_response",
    "pre_action_token",
    "first_action_token",
    "mean_output_action",
    "mean_output_belief_grid",
    "response_last",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_present(row: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return default


def episode_id_of(row: dict[str, Any]) -> str:
    value = first_present(row, ["episode_id", "episode", "episode_name", "id"], "")
    return str(value)


def step_id_of(row: dict[str, Any]) -> int:
    value = first_present(row, ["step_id", "step", "t", "step_index"], -1)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def prompt_of(row: dict[str, Any]) -> str:
    value = first_present(
        row,
        ["prompt_text", "prompt", "input_prompt", "rendered_prompt"],
        "",
    )
    return str(value or "")


def response_of(row: dict[str, Any]) -> str:
    value = first_present(
        row,
        ["raw_response", "response", "model_response", "output_text"],
        "",
    )
    return str(value or "")


def action_of(row: dict[str, Any]) -> str:
    direct = first_present(row, ["action", "chosen_action"], None)
    if direct:
        return str(direct).upper()

    parsed = row.get("parsed_response")
    if isinstance(parsed, dict) and parsed.get("action"):
        return str(parsed["action"]).upper()

    parsed_action = row.get("parsed_action")
    if parsed_action:
        return str(parsed_action).upper()

    response = response_of(row)
    match = re.search(
        r'["\']?action["\']?\s*:\s*["\']?(UP|DOWN|LEFT|RIGHT)["\']?',
        response,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else ""


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_positions(spec: str) -> list[str]:
    lowered = spec.strip().lower()
    if lowered == "default":
        return list(DEFAULT_POSITIONS)
    if lowered == "all":
        return list(ALL_POSITIONS)
    positions = parse_csv(spec)
    unknown = sorted(set(positions) - set(ALL_POSITIONS))
    if unknown:
        raise ValueError(f"Unknown positions: {unknown}")
    if not positions:
        raise ValueError("No positions selected.")
    return positions


def parse_max_memory(spec: str | None, visible_gpu_count: int) -> dict[Any, str] | None:
    if not spec:
        return None
    result: dict[Any, str] = {}
    for item in parse_csv(spec):
        if "=" not in item:
            raise ValueError(
                "--max-memory must look like '0=75GiB,1=75GiB,cpu=256GiB'"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key.lower() == "cpu":
            result["cpu"] = value
        else:
            local_index = int(key)
            if local_index < 0 or local_index >= visible_gpu_count:
                raise ValueError(
                    f"max-memory GPU {local_index} is outside visible local range "
                    f"0..{visible_gpu_count - 1}"
                )
            result[local_index] = value
    return result


def get_num_hidden_layers(config: Any) -> int:
    for name in ["num_hidden_layers", "n_layer", "num_layers"]:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise RuntimeError("Could not determine the number of transformer layers.")


def parse_layers(spec: str, num_hidden_layers: int) -> list[int]:
    # Hidden-state indexing matches Transformers:
    # 0 = embedding output; final index = final normalized representation.
    maximum = num_hidden_layers
    lowered = spec.strip().lower()
    if lowered == "all":
        return list(range(maximum + 1))
    if lowered == "auto":
        candidates = [
            0,
            round(maximum * 0.25),
            round(maximum * 0.50),
            round(maximum * 0.75),
            maximum,
        ]
        return sorted(set(int(x) for x in candidates))
    layers = sorted(set(int(x) for x in parse_csv(spec)))
    invalid = [x for x in layers if x < 0 or x > maximum]
    if invalid:
        raise ValueError(
            f"Invalid hidden-state indices {invalid}; valid range is 0..{maximum}"
        )
    if not layers:
        raise ValueError("No layers selected.")
    return layers


def find_tag_span(text: str, tag: str) -> tuple[int, int] | None:
    pattern = re.compile(
        rf"<{re.escape(tag)}(?:\s[^>]*)?>(.*?)</{re.escape(tag)}>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    return match.start(1), match.end(1)


def find_balanced_json_value_span(
    text: str,
    key: str,
    start_at: int = 0,
) -> tuple[int, int] | None:
    key_match = re.search(
        rf'["\']{re.escape(key)}["\']\s*:\s*',
        text[start_at:],
        flags=re.IGNORECASE,
    )
    if not key_match:
        return None
    value_start = start_at + key_match.end()
    while value_start < len(text) and text[value_start].isspace():
        value_start += 1
    if value_start >= len(text):
        return None

    opener = text[value_start]
    pairs = {"[": "]", "{": "}"}
    if opener not in pairs:
        end_match = re.search(r"[,}\n]", text[value_start:])
        end = len(text) if not end_match else value_start + end_match.start()
        return value_start, end

    closer = pairs[opener]
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(value_start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return value_start, index + 1
    return None


def find_action_span(text: str, action: str, start_at: int) -> tuple[int, int] | None:
    if not action:
        return None
    pattern = re.compile(
        rf'["\']?action["\']?\s*:\s*["\']?({re.escape(action)})["\']?',
        flags=re.IGNORECASE,
    )
    match = pattern.search(text, pos=start_at)
    if match:
        return match.start(1), match.end(1)

    match = re.search(rf"\b{re.escape(action)}\b", text[start_at:], re.IGNORECASE)
    if match:
        return start_at + match.start(), start_at + match.end()
    return None


def render_conversation(tokenizer: Any, prompt: str, response: str) -> tuple[str, str, int]:
    if getattr(tokenizer, "chat_template", None):
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_rendered = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if response:
            full_rendered = tokenizer.apply_chat_template(
                prompt_messages + [{"role": "assistant", "content": response}],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            full_rendered = prompt_rendered
    else:
        prompt_rendered = prompt
        full_rendered = prompt + ("\n" + response if response else "")

    if response:
        response_start = full_rendered.find(response)
        if response_start < 0:
            response_start = len(prompt_rendered)
    else:
        response_start = len(prompt_rendered)
    return prompt_rendered, full_rendered, response_start


def token_indices_for_char_span(
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int] | None:
    indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end > start and token_start < end
    ]
    if not indices:
        return None
    return min(indices), max(indices) + 1


def last_token_before(
    offsets: list[tuple[int, int]],
    char_position: int,
) -> tuple[int, int] | None:
    candidates = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and end <= char_position
    ]
    if not candidates:
        return None
    index = candidates[-1]
    return index, index + 1


def first_token_overlapping(
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int] | None:
    span = token_indices_for_char_span(offsets, start, end)
    if span is None:
        return None
    return span[0], span[0] + 1


def build_position_specs(
    tokenizer: Any,
    row: dict[str, Any],
    positions: list[str],
) -> tuple[dict[str, tuple[int, int] | None], dict[str, Any], str]:
    prompt = prompt_of(row)
    response = response_of(row)
    action = action_of(row)
    prompt_rendered, full_rendered, response_start = render_conversation(
        tokenizer,
        prompt,
        response,
    )

    encoded = tokenizer(
        full_rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]

    prompt_content_start = full_rendered.find(prompt)
    if prompt_content_start < 0:
        prompt_content_start = 0
    prompt_content_end = prompt_content_start + len(prompt)

    response_end = response_start + len(response) if response else len(full_rendered)
    action_span = find_action_span(full_rendered, action, response_start)

    char_spans: dict[str, tuple[int, int] | None] = {
        "mean_all_prompt": (prompt_content_start, prompt_content_end),
        "mean_last_feedback": find_tag_span(full_rendered, "last_feedback"),
        "mean_required_belief_updates": find_tag_span(
            full_rendered,
            "required_belief_updates",
        ),
        "mean_available_actions": find_tag_span(full_rendered, "available_actions"),
        "mean_current_belief_grid": find_tag_span(
            full_rendered,
            "current_belief_grid",
        ),
        "mean_history": find_tag_span(full_rendered, "history"),
        "mean_output_action": action_span,
        "mean_output_belief_grid": find_balanced_json_value_span(
            full_rendered,
            "belief_grid",
            response_start,
        ),
    }

    result: dict[str, tuple[int, int] | None] = {}
    prompt_last = last_token_before(offsets, response_start)
    response_tokens = token_indices_for_char_span(offsets, response_start, response_end)

    for name in positions:
        if name == "prompt_last" or name == "pre_response":
            result[name] = prompt_last
        elif name == "response_last":
            if response_tokens is None:
                result[name] = None
            else:
                result[name] = (response_tokens[1] - 1, response_tokens[1])
        elif name == "pre_action_token":
            if action_span is None:
                result[name] = None
            else:
                first = first_token_overlapping(offsets, *action_span)
                result[name] = None if first is None or first[0] == 0 else (first[0] - 1, first[0])
        elif name == "first_action_token":
            result[name] = (
                None
                if action_span is None
                else first_token_overlapping(offsets, *action_span)
            )
        elif name.startswith("after_"):
            base_name = "mean_" + name[len("after_") :]
            char_span = char_spans.get(base_name)
            if char_span is None:
                result[name] = None
            else:
                token_span = token_indices_for_char_span(offsets, *char_span)
                result[name] = (
                    None
                    if token_span is None
                    else (token_span[1] - 1, token_span[1])
                )
        else:
            char_span = char_spans.get(name)
            result[name] = (
                None
                if char_span is None
                else token_indices_for_char_span(offsets, *char_span)
            )

    metadata = {
        "episode_id": episode_id_of(row),
        "step_id": step_id_of(row),
        "action": action,
        "prompt_chars": len(prompt),
        "response_chars": len(response),
        "rendered_chars": len(full_rendered),
        "token_count": len(encoded["input_ids"]),
        "repaired": bool(row.get("repaired", False)),
        "parse_error": bool(row.get("parse_error", False)),
    }
    return result, metadata, full_rendered


def locate_backbone(model: Any) -> tuple[Any, Any, Any, Any]:
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "transformer", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        layers = getattr(candidate, "layers", None)
        embed = getattr(candidate, "embed_tokens", None)
        norm = getattr(candidate, "norm", None)
        if layers is not None and embed is not None and norm is not None:
            return candidate, layers, embed, norm

        decoder = getattr(candidate, "decoder", None)
        if decoder is not None:
            layers = getattr(decoder, "layers", None)
            embed = getattr(decoder, "embed_tokens", None)
            norm = getattr(decoder, "final_layer_norm", None)
            if layers is not None and embed is not None and norm is not None:
                return decoder, layers, embed, norm

        blocks = getattr(candidate, "h", None)
        embed = getattr(candidate, "wte", None)
        norm = getattr(candidate, "ln_f", None)
        if blocks is not None and embed is not None and norm is not None:
            return candidate, blocks, embed, norm

    raise RuntimeError(
        "Unsupported model architecture: could not locate decoder layers, "
        "token embedding, and final norm."
    )


def tensor_from_hook_output(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


@dataclass
class HookCollector:
    positions: list[str]
    selected_layers: list[int]
    current_specs: list[dict[str, tuple[int, int] | None]] | None = None
    values: dict[int, Any] | None = None

    def begin(self, specs: list[dict[str, tuple[int, int] | None]]) -> None:
        self.current_specs = specs
        self.values = {}

    def make_hook(self, hidden_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = tensor_from_hook_output(output)
            if not torch.is_tensor(hidden):
                raise RuntimeError(
                    f"Hook for hidden index {hidden_index} did not receive a tensor."
                )
            if self.current_specs is None or self.values is None:
                raise RuntimeError("Hook collector was not initialized for this batch.")

            batch_values = []
            for batch_index, spec in enumerate(self.current_specs):
                position_values = []
                for position in self.positions:
                    span = spec.get(position)
                    if span is None:
                        pooled = torch.zeros(
                            hidden.shape[-1],
                            dtype=torch.float32,
                            device=hidden.device,
                        )
                    else:
                        start, end = span
                        if start < 0 or end > hidden.shape[1] or start >= end:
                            pooled = torch.zeros(
                                hidden.shape[-1],
                                dtype=torch.float32,
                                device=hidden.device,
                            )
                        else:
                            pooled = hidden[batch_index, start:end].float().mean(dim=0)
                    position_values.append(pooled)
                batch_values.append(torch.stack(position_values, dim=0))

            pooled_batch = torch.stack(batch_values, dim=0)
            self.values[hidden_index] = pooled_batch.to(
                device="cpu",
                dtype=torch.float16,
                non_blocking=False,
            )

        return hook


def parse_dtype(name: str):
    import torch

    lowered = name.lower()
    if lowered in {"auto", "model"}:
        return "auto"
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "A fast tokenizer is required because position extraction uses "
            "offset mappings."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    num_hidden_layers = get_num_hidden_layers(config)
    selected_layers = parse_layers(args.layers, num_hidden_layers)

    visible_gpus = parse_csv(args.gpus)
    if not visible_gpus:
        raise ValueError("--gpus must contain at least one GPU id.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus)

    max_memory = parse_max_memory(args.max_memory, len(visible_gpus))
    dtype = parse_dtype(args.dtype)

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
        "device_map": args.device_map,
    }
    if dtype != "auto":
        load_kwargs["torch_dtype"] = dtype
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    if args.offload_folder:
        offload_folder = Path(args.offload_folder).resolve()
        offload_folder.mkdir(parents=True, exist_ok=True)
        load_kwargs["offload_folder"] = str(offload_folder)
        load_kwargs["offload_state_dict"] = True

    print(f"[large-activations] loading tokenizer={args.model}", flush=True)
    print(
        f"[large-activations] loading model={args.model} "
        f"device_map={args.device_map} visible_gpus={visible_gpus}",
        flush=True,
    )
    model = AutoModel.from_pretrained(args.model, **load_kwargs)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return tokenizer, model, num_hidden_layers, selected_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--device-map",
        default="balanced",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
    )
    parser.add_argument(
        "--max-memory",
        default=None,
        help="Local visible GPU limits, e.g. 0=75GiB,1=75GiB,cpu=256GiB.",
    )
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--positions", default="default")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--include-repaired", action="store_true")
    parser.add_argument("--include-parse-errors", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-subdir", default="activations")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    run = args.run.resolve()
    steps_path = run / "steps.jsonl"
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing {steps_path}")

    output_dir = run / args.output_subdir
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Use --overwrite or a different "
                "--output-subdir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = parse_positions(args.positions)
    all_rows = read_jsonl(steps_path)
    selected_rows: list[tuple[int, dict[str, Any]]] = []
    for source_index, row in enumerate(all_rows):
        if bool(row.get("parse_error", False)) and not args.include_parse_errors:
            continue
        if bool(row.get("repaired", False)) and not args.include_repaired:
            continue
        if not prompt_of(row):
            continue
        selected_rows.append((source_index, row))
        if args.max_rows is not None and len(selected_rows) >= args.max_rows:
            break

    if not selected_rows:
        raise RuntimeError("No eligible step rows remain after filtering.")

    tokenizer, model, num_hidden_layers, layers = load_model_and_tokenizer(args)
    backbone, decoder_layers, embedding, final_norm = locate_backbone(model)
    hidden_size = int(
        first_present(
            vars(model.config),
            ["hidden_size", "n_embd", "d_model"],
            0,
        )
    )
    if hidden_size <= 0:
        hidden_size = int(getattr(embedding, "embedding_dim"))

    print(
        f"[large-activations] rows={len(selected_rows)} positions={positions}",
        flush=True,
    )
    print(
        f"[large-activations] hidden_state_indices={layers} "
        f"num_decoder_layers={num_hidden_layers} hidden_size={hidden_size}",
        flush=True,
    )

    collector = HookCollector(positions=positions, selected_layers=layers)
    handles = []
    if 0 in layers:
        handles.append(embedding.register_forward_hook(collector.make_hook(0)))
    for hidden_index in layers:
        if 1 <= hidden_index < num_hidden_layers:
            handles.append(
                decoder_layers[hidden_index - 1].register_forward_hook(
                    collector.make_hook(hidden_index)
                )
            )
    if num_hidden_layers in layers:
        handles.append(
            final_norm.register_forward_hook(
                collector.make_hook(num_hidden_layers)
            )
        )

    x_path = output_dir / "X.npy"
    mask_path = output_dir / "position_mask.npy"
    X = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(selected_rows), len(positions), len(layers), hidden_size),
    )
    position_mask = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        dtype=np.bool_,
        shape=(len(selected_rows), len(positions)),
    )
    X[:] = 0
    position_mask[:] = False

    meta_rows: list[dict[str, Any]] = []
    started = time.time()

    try:
        for batch_start in range(0, len(selected_rows), args.batch_size):
            batch_items = selected_rows[batch_start : batch_start + args.batch_size]
            specs = []
            metadata_rows = []
            rendered_texts = []

            for source_index, row in batch_items:
                position_specs, metadata, rendered = build_position_specs(
                    tokenizer,
                    row,
                    positions,
                )
                metadata["source_row_index"] = source_index
                specs.append(position_specs)
                metadata_rows.append(metadata)
                rendered_texts.append(rendered)

            # Hooks use unpadded token indices. With right padding they remain valid.
            tokenizer.padding_side = "right"
            encoded = tokenizer(
                rendered_texts,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            input_device = embedding.weight.device
            model_inputs = {
                key: value.to(input_device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }

            collector.begin(specs)
            import torch

            with torch.inference_mode():
                _ = model(
                    **model_inputs,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

            assert collector.values is not None
            missing_layers = [layer for layer in layers if layer not in collector.values]
            if missing_layers:
                raise RuntimeError(
                    f"Hooks did not capture hidden-state indices {missing_layers}."
                )

            batch_size = len(batch_items)
            batch_array = np.zeros(
                (batch_size, len(positions), len(layers), hidden_size),
                dtype=np.float16,
            )
            for layer_slot, hidden_index in enumerate(layers):
                value = collector.values[hidden_index].numpy()
                batch_array[:, :, layer_slot, :] = value

            end = batch_start + batch_size
            X[batch_start:end] = batch_array
            for batch_index, spec in enumerate(specs):
                position_mask[batch_start + batch_index] = np.array(
                    [spec.get(position) is not None for position in positions],
                    dtype=np.bool_,
                )
                metadata_rows[batch_index]["activation_row"] = batch_start + batch_index
                metadata_rows[batch_index]["available_positions"] = [
                    position
                    for position in positions
                    if spec.get(position) is not None
                ]
                meta_rows.append(metadata_rows[batch_index])

            X.flush()
            position_mask.flush()

            processed = end
            if processed % 10 == 0 or processed == len(selected_rows):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[large-activations] processed={processed}/{len(selected_rows)} "
                    f"rows_per_second={processed / elapsed:.3f}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    np.save(output_dir / "layers.npy", np.asarray(layers, dtype=np.int64))
    (output_dir / "positions.json").write_text(
        json.dumps(positions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "meta.jsonl", meta_rows)

    manifest = {
        "schema_version": "large-model-activations-v1",
        "model": args.model,
        "source_steps": str(steps_path),
        "rows": len(selected_rows),
        "positions": positions,
        "hidden_state_indices": layers,
        "num_decoder_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "shape": [len(selected_rows), len(positions), len(layers), hidden_size],
        "dtype": "float16",
        "model_dtype": args.dtype,
        "gpus": parse_csv(args.gpus),
        "device_map": args.device_map,
        "max_memory": args.max_memory,
        "batch_size": args.batch_size,
        "include_repaired": args.include_repaired,
        "include_parse_errors": args.include_parse_errors,
        "created_unix": time.time(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[large-activations] saved={output_dir}", flush=True)
    print(f"[large-activations] X_shape={manifest['shape']}", flush=True)


if __name__ == "__main__":
    main()

