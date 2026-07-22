#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-$(pwd)}"
cd "$REPO"

mkdir -p scripts docs

cat > scripts/extract_activations_model_parallel.py <<'PYACT'
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

PYACT

cat > scripts/train_probes_multigpu.py <<'PYPROBE'
#!/usr/bin/env python3
"""
Task-parallel multi-GPU linear probing for grid-world activations.

Why task parallel rather than DDP?
Each (position, layer, split) probe is statistically independent and the
linear heads are tiny. Synchronizing one head with DDP would add unnecessary
communication. This script gives each GPU different independent jobs and
merges the results deterministically.

It accepts the standard grid-world activation contract:
  RUN/activations/{X.npy,position_mask.npy,layers.npy,positions.json,meta.jsonl}
  RUN/targets/targets.jsonl

It writes:
  RUN/<output_subdir>/
    probe_results_splits.csv
    probe_results.csv
    best_by_task.csv
    group_summary.csv
    summary.md
    jobs/*.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


PLANNING_TASKS = {
    "chosen_action_is_astar_best",
    "chosen_action_reduces_true_distance",
    "loop_risk",
    "position_action_seen_before",
    "position_seen_before",
    "chosen_target_gold_belief",
}
FAITHFULNESS_TASKS = {
    "model_missed_any_gold_known",
    "action_optimal_but_belief_incomplete",
    "model_known_cell_acc_step",
    "belief_known_acc_step",
}


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


def first_present(row: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return default


def episode_id_of(row: dict[str, Any]) -> str:
    return str(first_present(row, ["episode_id", "episode", "episode_name", "id"], ""))


def step_id_of(row: dict[str, Any]) -> int:
    value = first_present(row, ["step_id", "step", "t", "step_index"], -1)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def canonical_key(row: dict[str, Any]) -> tuple[str, int]:
    return episode_id_of(row), step_id_of(row)


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return None
        rounded = round(float(value))
        if abs(float(value) - rounded) < 1e-8:
            return str(int(rounded))
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "nan", "null", "na"}:
            return None
        return stripped
    return None


def task_group_of(name: str) -> str | None:
    if re.fullmatch(r"gold_local_(UP|DOWN|LEFT|RIGHT)_OFUW", name):
        return "local"
    if re.fullmatch(r"gold_cell_x\d+_y\d+_OFU", name):
        return "cells"
    if re.fullmatch(r"gold_cell_x\d+_y\d+_known", name):
        return "memory"
    if re.fullmatch(r"(explicit|model)_cell_x\d+_y\d+_OFU", name):
        return "explicit_cells"
    if re.fullmatch(r"true_cell_x\d+_y\d+_unobserved_(OF|OFU)", name):
        return "true_cells_unobserved"
    if re.fullmatch(r"true_cell_x\d+_y\d+_(OF|OFU)", name):
        return "true_cells"
    if name in PLANNING_TASKS or re.fullmatch(
        r"true_action_(UP|DOWN|LEFT|RIGHT)_is_astar_best",
        name,
    ):
        return "planning"
    if name in FAITHFULNESS_TASKS:
        return "faithfulness"
    return None


def discover_tasks(
    targets: list[dict[str, Any]],
    requested_groups: set[str],
    explicit_tasks: set[str],
    task_regex: str | None,
    min_class_count: int,
) -> list[dict[str, Any]]:
    candidate_names: set[str] = set()
    for row in targets:
        candidate_names.update(row.keys())

    regex = re.compile(task_regex) if task_regex else None
    tasks: list[dict[str, Any]] = []

    for name in sorted(candidate_names):
        group = task_group_of(name)
        selected = (
            name in explicit_tasks
            or (group is not None and group in requested_groups)
            or (regex is not None and regex.search(name) is not None)
        )
        if not selected:
            continue
        if group is None:
            group = "custom"

        labels = [
            normalized
            for row in targets
            if (normalized := normalize_label(row.get(name))) is not None
        ]
        counts = Counter(labels)
        classes = sorted(
            [label for label, count in counts.items() if count >= min_class_count]
        )
        if len(classes) < 2:
            print(
                f"[probe/multigpu] skip task={name}: classes={dict(counts)}",
                flush=True,
            )
            continue
        tasks.append(
            {
                "name": name,
                "group": group,
                "classes": classes,
                "counts": dict(counts),
            }
        )

    if not tasks:
        raise RuntimeError("No probe tasks matched the requested groups/tasks.")
    return tasks


def resolve_positions(spec: str, available: list[str]) -> list[str]:
    lowered = spec.strip().lower()
    if lowered == "all":
        return list(available)
    if lowered == "auto":
        preferred = [
            "mean_last_feedback",
            "mean_current_belief_grid",
            "pre_action_token",
            "prompt_last",
        ]
        selected = [name for name in preferred if name in available]
        if not selected:
            raise RuntimeError("None of the automatic positions exist in activations.")
        return selected
    requested = [part.strip() for part in spec.split(",") if part.strip()]
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Unknown activation positions: {missing}")
    return requested


def resolve_layers(spec: str, available: list[int]) -> list[int]:
    lowered = spec.strip().lower()
    if lowered == "all":
        return list(available)
    if lowered == "auto":
        maximum = max(available)
        desired = {
            0,
            round(maximum * 0.25),
            round(maximum * 0.50),
            round(maximum * 0.75),
            maximum,
        }
        selected = [layer for layer in available if layer in desired]
        if not selected:
            selected = list(available)
        return selected
    requested = sorted(
        set(int(part.strip()) for part in spec.split(",") if part.strip())
    )
    missing = [layer for layer in requested if layer not in available]
    if missing:
        raise ValueError(f"Unknown activation layers: {missing}")
    return requested


def align_meta_and_targets(
    meta_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    target_map = {
        canonical_key(row): row
        for row in target_rows
        if canonical_key(row) != ("", -1)
    }

    activation_indices: list[int] = []
    aligned_targets: list[dict[str, Any]] = []
    groups: list[str] = []

    if target_map:
        for activation_index, meta in enumerate(meta_rows):
            key = canonical_key(meta)
            target = target_map.get(key)
            if target is None:
                continue
            activation_indices.append(activation_index)
            aligned_targets.append(target)
            groups.append(key[0])
    elif len(meta_rows) == len(target_rows):
        activation_indices = list(range(len(meta_rows)))
        aligned_targets = list(target_rows)
        groups = [episode_id_of(meta) for meta in meta_rows]
    else:
        raise RuntimeError(
            "Could not align activation metadata with targets by "
            "(episode_id, step_id), and row counts differ."
        )

    if not activation_indices:
        raise RuntimeError("No activation rows aligned with target rows.")

    return (
        np.asarray(activation_indices, dtype=np.int64),
        aligned_targets,
        np.asarray(groups, dtype=object),
    )


def build_splits(
    groups: np.ndarray,
    n_splits: int,
    test_size: float,
    random_seed: int,
) -> list[dict[str, list[str]]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise RuntimeError("At least three episodes are required for group splits.")

    dummy = np.zeros(len(groups), dtype=np.int8)
    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_seed,
    )
    splits = []
    for train_index, test_index in splitter.split(dummy, groups=groups):
        splits.append(
            {
                "train_groups": sorted(set(groups[train_index].tolist())),
                "test_groups": sorted(set(groups[test_index].tolist())),
            }
        )
    return splits


def encode_targets(
    aligned_targets: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> np.ndarray:
    labels = np.full(
        (len(aligned_targets), len(tasks)),
        fill_value=-100,
        dtype=np.int64,
    )
    for task_index, task in enumerate(tasks):
        class_to_index = {
            label: class_index
            for class_index, label in enumerate(task["classes"])
        }
        for row_index, row in enumerate(aligned_targets):
            label = normalize_label(row.get(task["name"]))
            if label in class_to_index:
                labels[row_index, task_index] = class_to_index[label]
    return labels


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def train_one_job(
    config: dict[str, Any],
    job: dict[str, Any],
    device_name: str,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(max(1, int(config["cpu_threads_per_worker"])))
    if device_name.startswith("cuda"):
        torch.cuda.set_device(0)
        torch.set_float32_matmul_precision("high")

    run = Path(config["run"])
    activations_dir = run / config["activations_subdir"]
    X_memmap = np.load(activations_dir / "X.npy", mmap_mode="r")
    position_mask = np.load(
        activations_dir / "position_mask.npy",
        mmap_mode="r",
    )

    activation_indices = np.asarray(config["activation_indices"], dtype=np.int64)
    labels = np.load(config["encoded_targets_path"], mmap_mode="r")
    groups = np.asarray(config["groups"], dtype=object)
    tasks = config["tasks"]
    split = config["splits"][job["split_index"]]

    pidx = int(job["position_index"])
    lidx = int(job["layer_index"])
    valid_position = np.asarray(
        position_mask[activation_indices, pidx],
        dtype=np.bool_,
    )
    train_group_set = set(split["train_groups"])
    test_group_set = set(split["test_groups"])
    train_rows = np.asarray(
        [
            index
            for index, group in enumerate(groups)
            if valid_position[index] and group in train_group_set
        ],
        dtype=np.int64,
    )
    test_rows = np.asarray(
        [
            index
            for index, group in enumerate(groups)
            if valid_position[index] and group in test_group_set
        ],
        dtype=np.int64,
    )
    if len(train_rows) == 0 or len(test_rows) == 0:
        return []

    X_train_np = np.asarray(
        X_memmap[activation_indices[train_rows], pidx, lidx, :],
        dtype=np.float32,
    )
    X_test_np = np.asarray(
        X_memmap[activation_indices[test_rows], pidx, lidx, :],
        dtype=np.float32,
    )
    y_train_np = np.asarray(labels[train_rows], dtype=np.int64)
    y_test_np = np.asarray(labels[test_rows], dtype=np.int64)

    if config["standardize"]:
        mean = X_train_np.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = X_train_np.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        X_train_np = (X_train_np - mean) / std
        X_test_np = (X_test_np - mean) / std

    active_task_indices: list[int] = []
    task_class_counts: list[int] = []
    task_class_weights: list[torch.Tensor] = []
    for task_index, task in enumerate(tasks):
        valid_train = y_train_np[:, task_index] >= 0
        values = y_train_np[valid_train, task_index]
        observed = sorted(set(values.tolist()))
        required = list(range(len(task["classes"])))
        if len(values) < config["min_samples_per_task"]:
            continue
        if len(observed) < 2:
            continue
        # A class absent from the training fold cannot be learned. Skip this
        # task for the current split instead of allowing an untrained logit to
        # produce "predicted class absent from y_true" warnings.
        if set(observed) != set(required):
            continue

        counts = np.bincount(values, minlength=len(required)).astype(np.float64)
        weights = np.zeros_like(counts, dtype=np.float32)
        nonzero = counts > 0
        weights[nonzero] = counts[nonzero].sum() / (
            nonzero.sum() * counts[nonzero]
        )
        active_task_indices.append(task_index)
        task_class_counts.append(len(required))
        task_class_weights.append(
            torch.tensor(weights, dtype=torch.float32, device=device_name)
        )

    if not active_task_indices:
        return []

    offsets = np.cumsum([0] + task_class_counts).tolist()

    class MultiHeadLinear(nn.Module):
        def __init__(self, hidden_size: int, total_classes: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(total_classes, hidden_size))
            self.bias = nn.Parameter(torch.zeros(total_classes))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, self.weight, self.bias)

    model = MultiHeadLinear(X_train_np.shape[1], offsets[-1]).to(device_name)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(config["random_seed"])
        + 100003 * int(job["split_index"])
        + 1009 * int(job["position_index"])
        + int(job["layer_value"])
    )
    batch_size = min(int(config["batch_size"]), len(X_train_np))
    epochs_ran = 0

    for epoch in range(int(config["epochs"])):
        permutation = torch.randperm(
            len(X_train_np),
            generator=generator,
        ).numpy()
        model.train()
        for start in range(0, len(permutation), batch_size):
            batch_indices = permutation[start : start + batch_size]
            xb = torch.from_numpy(X_train_np[batch_indices]).to(
                device_name,
                non_blocking=False,
            )
            yb = torch.from_numpy(y_train_np[batch_indices]).to(
                device_name,
                non_blocking=False,
            )
            logits = model(xb)
            losses = []
            for active_slot, task_index in enumerate(active_task_indices):
                valid = yb[:, task_index] >= 0
                if int(valid.sum()) == 0:
                    continue
                start_class = offsets[active_slot]
                end_class = offsets[active_slot + 1]
                losses.append(
                    F.cross_entropy(
                        logits[valid, start_class:end_class],
                        yb[valid, task_index],
                        weight=task_class_weights[active_slot],
                    )
                )
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        epochs_ran = epoch + 1

    model.eval()
    test_logits_parts = []
    eval_batch_size = max(batch_size, 1024)
    with torch.inference_mode():
        for start in range(0, len(X_test_np), eval_batch_size):
            xb = torch.from_numpy(X_test_np[start : start + eval_batch_size]).to(
                device_name
            )
            test_logits_parts.append(model(xb).cpu())
    test_logits = torch.cat(test_logits_parts, dim=0).numpy()

    rows: list[dict[str, Any]] = []
    for active_slot, task_index in enumerate(active_task_indices):
        task = tasks[task_index]
        valid_test = y_test_np[:, task_index] >= 0
        valid_train = y_train_np[:, task_index] >= 0
        if int(valid_test.sum()) == 0:
            continue

        y_true = y_test_np[valid_test, task_index]
        start_class = offsets[active_slot]
        end_class = offsets[active_slot + 1]
        y_pred = test_logits[valid_test, start_class:end_class].argmax(axis=1)
        metric_labels = list(range(len(task["classes"])))
        metrics = safe_metrics(y_true, y_pred, metric_labels)

        train_values = y_train_np[valid_train, task_index]
        majority_class = int(
            np.bincount(
                train_values,
                minlength=len(task["classes"]),
            ).argmax()
        )
        majority_pred = np.full_like(y_true, majority_class)
        majority_metrics = safe_metrics(y_true, majority_pred, metric_labels)

        rows.append(
            {
                "task": task["name"],
                "task_group": task["group"],
                "position": job["position"],
                "layer": int(job["layer_value"]),
                "split": int(job["split_index"]),
                "backend": "torch_multigpu_task_parallel",
                "classes": json.dumps(task["classes"], ensure_ascii=False),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "majority_accuracy": majority_metrics["accuracy"],
                "majority_macro_f1": majority_metrics["macro_f1"],
                "num_train": int(valid_train.sum()),
                "num_test": int(valid_test.sum()),
                "epochs": epochs_ran,
            }
        )

    del model, optimizer
    if device_name.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def worker_main(
    worker_index: int,
    gpu: str,
    config_path: str,
    jobs: list[dict[str, Any]],
) -> None:
    try:
        if gpu.lower() == "cpu":
            device_name = "cpu"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            device_name = "cuda:0"

        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        jobs_dir = Path(config["output_dir"]) / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[probe/worker {worker_index}] device={gpu} jobs={len(jobs)}",
            flush=True,
        )
        for local_index, job in enumerate(jobs, 1):
            job_path = jobs_dir / job["filename"]
            if job_path.exists() and config["resume"]:
                continue

            started = time.time()
            rows = train_one_job(config, job, device_name)
            payload = {
                "job": job,
                "rows": rows,
                "elapsed_seconds": time.time() - started,
                "worker_index": worker_index,
                "gpu": gpu,
            }
            temporary = job_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(job_path)
            print(
                f"[probe/worker {worker_index}] {local_index}/{len(jobs)} "
                f"{job['position']} L{job['layer_value']} "
                f"split={job['split_index']} tasks={len(rows)} "
                f"seconds={payload['elapsed_seconds']:.1f}",
                flush=True,
            )
    except Exception:
        traceback.print_exc()
        raise


def aggregate_results(output_dir: Path, jobs: list[dict[str, Any]]) -> None:
    split_rows: list[dict[str, Any]] = []
    missing = []
    for job in jobs:
        path = output_dir / "jobs" / job["filename"]
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        split_rows.extend(payload.get("rows", []))

    if missing:
        raise RuntimeError(
            f"{len(missing)} probe jobs are missing. First missing file: {missing[0]}"
        )
    if not split_rows:
        raise RuntimeError("Probe workers completed but produced no result rows.")

    split_df = pd.DataFrame(split_rows).sort_values(
        ["task_group", "task", "position", "layer", "split"]
    )
    split_df.to_csv(output_dir / "probe_results_splits.csv", index=False)

    group_columns = [
        "task",
        "task_group",
        "position",
        "layer",
        "backend",
        "classes",
    ]
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "majority_accuracy",
        "majority_macro_f1",
        "num_test",
    ]

    records = []
    for keys, frame in split_df.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        for metric in metric_columns:
            values = frame[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        records.append(row)

    result_df = pd.DataFrame(records).sort_values(
        ["task_group", "task", "position", "layer"]
    )
    result_df.to_csv(output_dir / "probe_results.csv", index=False)

    best_df = (
        result_df.sort_values(
            ["task", "macro_f1_mean", "macro_f1_std"],
            ascending=[True, False, True],
        )
        .groupby("task", as_index=False)
        .head(1)
        .sort_values(["task_group", "task"])
    )
    best_df.to_csv(output_dir / "best_by_task.csv", index=False)

    summary_records = []
    for task_group, frame in best_df.groupby("task_group"):
        summary_records.append(
            {
                "task_group": task_group,
                "tasks": int(len(frame)),
                "mean_best_macro_f1": float(frame["macro_f1_mean"].mean()),
                "median_best_macro_f1": float(frame["macro_f1_mean"].median()),
                "mean_split_std": float(frame["macro_f1_std"].mean()),
                "mean_delta_over_majority": float(
                    (
                        frame["macro_f1_mean"]
                        - frame["majority_macro_f1_mean"]
                    ).mean()
                ),
            }
        )
    group_summary = pd.DataFrame(summary_records).sort_values("task_group")
    group_summary.to_csv(output_dir / "group_summary.csv", index=False)

    lines = [
        "# Multi-GPU probe report",
        "",
        "## Best score by task group",
        "",
        group_summary.to_markdown(index=False),
        "",
        "## Best position/layer per task",
        "",
        best_df[
            [
                "task_group",
                "task",
                "position",
                "layer",
                "macro_f1_mean",
                "macro_f1_std",
                "majority_macro_f1_mean",
            ]
        ].to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- Work is parallelized across independent position/layer/split jobs.",
        "- Splits are grouped by episode.",
        "- Compare macro-F1 with the majority macro-F1 baseline.",
        "- Decodability does not by itself establish causal use.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--groups", default="local,cells,planning")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--task-regex", default=None)
    parser.add_argument("--positions", default="auto")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument("--min-samples-per-task", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=2)
    parser.add_argument("--activations-subdir", default="activations")
    parser.add_argument("--targets-subdir", default="targets")
    parser.add_argument("--output-subdir", default="probes_multigpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run = args.run.resolve()
    activations_dir = run / args.activations_subdir
    targets_path = run / args.targets_subdir / "targets.jsonl"
    required = [
        activations_dir / "X.npy",
        activations_dir / "position_mask.npy",
        activations_dir / "layers.npy",
        activations_dir / "positions.json",
        activations_dir / "meta.jsonl",
        targets_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    output_dir = run / args.output_subdir
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "jobs").mkdir(parents=True, exist_ok=True)

    available_positions = json.loads(
        (activations_dir / "positions.json").read_text(encoding="utf-8")
    )
    available_layers = [
        int(value)
        for value in np.load(activations_dir / "layers.npy").tolist()
    ]
    selected_positions = resolve_positions(args.positions, available_positions)
    selected_layers = resolve_layers(args.layers, available_layers)

    meta_rows = read_jsonl(activations_dir / "meta.jsonl")
    target_rows = read_jsonl(targets_path)
    activation_indices, aligned_targets, groups = align_meta_and_targets(
        meta_rows,
        target_rows,
    )

    requested_groups = {
        item.strip()
        for item in args.groups.split(",")
        if item.strip()
    }
    explicit_tasks = {
        item.strip()
        for item in args.tasks.split(",")
        if item.strip()
    }
    tasks = discover_tasks(
        aligned_targets,
        requested_groups,
        explicit_tasks,
        args.task_regex,
        args.min_class_count,
    )
    encoded_targets = encode_targets(aligned_targets, tasks)
    encoded_targets_path = output_dir / "encoded_targets.npy"
    np.save(encoded_targets_path, encoded_targets)

    splits = build_splits(
        groups,
        args.splits,
        args.test_size,
        args.random_seed,
    )
    (output_dir / "splits.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    position_to_index = {
        name: index for index, name in enumerate(available_positions)
    }
    layer_to_index = {
        value: index for index, value in enumerate(available_layers)
    }

    jobs: list[dict[str, Any]] = []
    for position in selected_positions:
        for layer in selected_layers:
            for split_index in range(args.splits):
                safe_position = re.sub(r"[^A-Za-z0-9_.-]+", "_", position)
                jobs.append(
                    {
                        "position": position,
                        "position_index": position_to_index[position],
                        "layer_value": layer,
                        "layer_index": layer_to_index[layer],
                        "split_index": split_index,
                        "filename": (
                            f"{safe_position}__L{layer}__S{split_index:03d}.json"
                        ),
                    }
                )

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain GPU ids or 'cpu'.")

    config = {
        "run": str(run),
        "output_dir": str(output_dir),
        "activations_subdir": args.activations_subdir,
        "activation_indices": activation_indices.tolist(),
        "groups": groups.tolist(),
        "tasks": tasks,
        "splits": splits,
        "encoded_targets_path": str(encoded_targets_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "min_samples_per_task": args.min_samples_per_task,
        "random_seed": args.random_seed,
        "standardize": args.standardize,
        "cpu_threads_per_worker": args.cpu_threads_per_worker,
        "resume": args.resume,
    }
    config_path = output_dir / "worker_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "jobs_manifest.json").write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pending_jobs = [
        job
        for job in jobs
        if not (
            args.resume
            and (output_dir / "jobs" / job["filename"]).exists()
        )
    ]

    print(
        f"[probe/multigpu] aligned_rows={len(aligned_targets)} "
        f"episodes={len(set(groups.tolist()))}",
        flush=True,
    )
    print(
        f"[probe/multigpu] tasks={len(tasks)} "
        f"positions={selected_positions} layers={selected_layers}",
        flush=True,
    )
    print(
        f"[probe/multigpu] jobs_total={len(jobs)} "
        f"jobs_pending={len(pending_jobs)} workers={gpus}",
        flush=True,
    )

    if pending_jobs:
        assignments = [pending_jobs[index::len(gpus)] for index in range(len(gpus))]
        context = mp.get_context("spawn")
        processes = []
        for worker_index, (gpu, assigned_jobs) in enumerate(
            zip(gpus, assignments)
        ):
            if not assigned_jobs:
                continue
            process = context.Process(
                target=worker_main,
                args=(
                    worker_index,
                    gpu,
                    str(config_path),
                    assigned_jobs,
                ),
            )
            process.start()
            processes.append(process)

        failures = []
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failures.append((process.pid, process.exitcode))
        if failures:
            raise RuntimeError(f"Probe workers failed: {failures}")

    aggregate_results(output_dir, jobs)
    print(f"[probe/multigpu] saved={output_dir}", flush=True)
    print(f"[probe/multigpu] report={output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()

PYPROBE

cat > scripts/make_large_model_configs.py <<'PYCONFIG'
#!/usr/bin/env python3
"""
Create Qwen2.5-32B/72B experiment configs by cloning the existing 7B config.

The script preserves every prompt, decoding, environment, and logging setting,
and changes only the model identifier. This is preferable for a clean
model-scale comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MODELS = {
    "32b": "Qwen/Qwen2.5-32B-Instruct",
    "72b": "Qwen/Qwen2.5-72B-Instruct",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("configs/experiments/qwen25_7b_strategy_a.yaml"),
    )
    parser.add_argument(
        "--sizes",
        default="32b,72b",
        help="Comma-separated subset of 32b,72b.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing source config: {source}")

    text = source.read_text(encoding="utf-8")
    if "Qwen/Qwen2.5-7B-Instruct" not in text:
        raise RuntimeError(
            "The source config does not contain "
            "'Qwen/Qwen2.5-7B-Instruct'; refusing a blind replacement."
        )

    for size in [item.strip().lower() for item in args.sizes.split(",") if item.strip()]:
        if size not in MODELS:
            raise ValueError(f"Unknown size {size!r}; choose 32b or 72b.")
        destination = source.with_name(f"qwen25_{size}_strategy_a.yaml")
        replaced = text.replace("Qwen/Qwen2.5-7B-Instruct", MODELS[size])
        destination.write_text(replaced, encoding="utf-8")
        print(f"[large-config] {size}: {destination}")


if __name__ == "__main__":
    main()

PYCONFIG

cat > scripts/run_qwen25_large.sh <<'SHRUN'
#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODEL_SIZE=32b GPUS=0,1,2,3 MAPS=... RUN=... bash scripts/run_qwen25_large.sh
#   MODEL_SIZE=72b GPUS=0,1,2,3,4,5,6,7 MAPS=... RUN=... bash scripts/run_qwen25_large.sh
#
# Stages can be overridden:
#   STAGES=generate,validate,targets,activations,probes,report

MODEL_SIZE="${MODEL_SIZE:-32b}"
GPUS="${GPUS:-0,1,2,3}"
MAPS="${MAPS:-data/generated/grid5x5_diverse_1000.jsonl}"
RUN="${RUN:-runs/qwen25_${MODEL_SIZE}_diverse5x5_1000}"
STAGES="${STAGES:-generate,validate,targets,activations,probes,report}"
POSITIONS="${POSITIONS:-default}"
LAYERS="${LAYERS:-all}"
PROBE_GROUPS="${PROBE_GROUPS:-local,cells,planning}"
PROBE_SPLITS="${PROBE_SPLITS:-20}"
PROBE_EPOCHS="${PROBE_EPOCHS:-60}"
PROBE_OVERWRITE="${PROBE_OVERWRITE:-0}"
ACTIVATION_BATCH_SIZE="${ACTIVATION_BATCH_SIZE:-1}"
DEVICE_MAP="${DEVICE_MAP:-balanced}"
MAX_MEMORY="${MAX_MEMORY:-}"

case "$MODEL_SIZE" in
  32b)
    MODEL="Qwen/Qwen2.5-32B-Instruct"
    CONFIG="configs/experiments/qwen25_32b_strategy_a.yaml"
    ;;
  72b)
    MODEL="Qwen/Qwen2.5-72B-Instruct"
    CONFIG="configs/experiments/qwen25_72b_strategy_a.yaml"
    ;;
  *)
    echo "MODEL_SIZE must be 32b or 72b" >&2
    exit 2
    ;;
esac

contains_stage() {
  [[ ",${STAGES}," == *",$1,"* ]]
}

echo "MODEL=$MODEL"
echo "GPUS=$GPUS"
echo "MAPS=$MAPS"
echo "RUN=$RUN"
echo "STAGES=$STAGES"

if contains_stage generate; then
  # Tensor parallel keeps one logical model across all selected GPUs.
  grid-world trajectories generate \
    --config "$CONFIG" \
    --maps "$MAPS" \
    --run "$RUN" \
    --gpus "$GPUS" \
    --parallel-mode tensor
fi

if contains_stage validate; then
  grid-world trajectories validate --run "$RUN"
  grid-world trajectories summarize --run "$RUN"
fi

if contains_stage targets; then
  grid-world targets build --run "$RUN"
fi

if contains_stage activations; then
  EXTRA_MEMORY_ARGS=()
  if [[ -n "$MAX_MEMORY" ]]; then
    EXTRA_MEMORY_ARGS+=(--max-memory "$MAX_MEMORY")
  fi

  python scripts/extract_activations_model_parallel.py \
    --run "$RUN" \
    --model "$MODEL" \
    --gpus "$GPUS" \
    --device-map "$DEVICE_MAP" \
    --dtype bf16 \
    --layers "$LAYERS" \
    --positions "$POSITIONS" \
    --batch-size "$ACTIVATION_BATCH_SIZE" \
    --overwrite \
    "${EXTRA_MEMORY_ARGS[@]}"
fi

if contains_stage probes; then
  PROBE_EXTRA_ARGS=()
  if [[ "$PROBE_OVERWRITE" == "1" ]]; then
    PROBE_EXTRA_ARGS+=(--overwrite)
  fi

  python scripts/train_probes_multigpu.py \
    --run "$RUN" \
    --groups "$PROBE_GROUPS" \
    --positions auto \
    --layers all \
    --gpus "$GPUS" \
    --splits "$PROBE_SPLITS" \
    --epochs "$PROBE_EPOCHS" \
    --output-subdir probes_multigpu \
    "${PROBE_EXTRA_ARGS[@]}"
fi

if contains_stage report; then
  cat "$RUN/probes_multigpu/summary.md"
fi

SHRUN

cat > docs/LARGE_MODELS_AND_MULTIGPU_PROBES.md <<'MDREADME'
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

MDREADME

chmod +x   scripts/extract_activations_model_parallel.py   scripts/train_probes_multigpu.py   scripts/make_large_model_configs.py   scripts/run_qwen25_large.sh

python -m py_compile   scripts/extract_activations_model_parallel.py   scripts/train_probes_multigpu.py   scripts/make_large_model_configs.py

python scripts/make_large_model_configs.py

echo
echo "Upgrade installed in: $REPO"
echo "Documentation: docs/LARGE_MODELS_AND_MULTIGPU_PROBES.md"
echo
echo "Install/verify dependencies:"
echo '  python -m pip install -U "transformers>=4.45" accelerate pandas scikit-learn tabulate'
echo
echo "Example:"
echo '  MODEL_SIZE=32b GPUS=0,1,2,3 MAPS=... RUN=... bash scripts/run_qwen25_large.sh'
