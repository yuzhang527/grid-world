from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from grid_world.activations.positions import (
    action_value_span,
    belief_output_span,
    parse_positions,
    tag_span,
)
from grid_world.utils.io import read_jsonl, write_json, write_jsonl
from grid_world.utils.manifest import write_manifest


def _torch_dtype(name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(name, "auto")


def _resolve_layers(value: str, total: int) -> list[int]:
    if value == "all":
        return list(range(total))
    if value == "auto":
        return sorted(set([0, total // 4, total // 2, 3 * total // 4, total - 1]))
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or min(result) < 0 or max(result) >= total:
        raise ValueError(f"Layer indices must be in [0, {total - 1}]")
    return result


def _token_span(tokenizer, text: str, span) -> tuple[int, int] | None:
    if span is None or span.end <= span.start:
        return None
    start = len(tokenizer(text[: span.start], add_special_tokens=False).input_ids)
    end = len(tokenizer(text[: span.end], add_special_tokens=False).input_ids)
    return (start, end) if end > start else None


def _position_specs(
    tokenizer,
    prompt: str,
    response: str,
    names: list[str],
) -> tuple[dict[str, tuple[str, tuple[int, int] | int] | None], int]:
    full = prompt + response
    prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    full_len = len(tokenizer(full, add_special_tokens=False).input_ids)

    tags = {
        "mean_last_feedback": "last_feedback",
        "after_last_feedback": "last_feedback",
        "mean_required_belief_updates": "required_belief_updates",
        "after_required_belief_updates": "required_belief_updates",
        "mean_available_actions": "available_actions",
        "after_available_actions": "available_actions",
        "mean_current_belief_grid": "current_belief_grid",
        "after_current_belief_grid": "current_belief_grid",
        "mean_history": "history",
        "after_history": "history",
    }

    action_tokens = _token_span(
        tokenizer,
        full,
        action_value_span(response, len(prompt)),
    )
    belief_tokens = _token_span(
        tokenizer,
        full,
        belief_output_span(response, len(prompt)),
    )

    result: dict[str, tuple[str, tuple[int, int] | int] | None] = {}
    for name in names:
        if name in {"prompt_last", "pre_response"}:
            result[name] = ("index", max(0, prompt_len - 1))
        elif name == "mean_all_prompt":
            result[name] = ("mean", (0, prompt_len))
        elif name == "response_last":
            result[name] = ("index", max(0, full_len - 1))
        elif name in tags:
            tokens = _token_span(tokenizer, full, tag_span(prompt, tags[name]))
            if tokens is None:
                result[name] = None
            elif name.startswith("after_"):
                result[name] = ("index", tokens[1] - 1)
            else:
                result[name] = ("mean", tokens)
        elif name == "first_action_token":
            result[name] = ("index", action_tokens[0]) if action_tokens else None
        elif name == "pre_action_token":
            result[name] = (
                ("index", max(0, action_tokens[0] - 1)) if action_tokens else None
            )
        elif name == "mean_output_action":
            result[name] = ("mean", action_tokens) if action_tokens else None
        elif name == "mean_output_belief_grid":
            result[name] = ("mean", belief_tokens) if belief_tokens else None
        else:
            result[name] = None
    return result, full_len


def _locate_transformer_modules(model):
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(model, "h", None)
    if layers is None and hasattr(model, "decoder"):
        layers = getattr(model.decoder, "layers", None)
    if layers is None:
        raise RuntimeError(
            "Could not locate transformer layers. Qwen2/Qwen2.5 is supported through "
            "model.layers."
        )

    final_norm = getattr(model, "norm", None)
    if final_norm is None and hasattr(model, "decoder"):
        final_norm = getattr(model.decoder, "final_layer_norm", None)
    if final_norm is None:
        raise RuntimeError("Could not locate the model's final normalization layer.")

    embedding = model.get_input_embeddings()
    return embedding, layers, final_norm


def _tensor_from_hook_output(output):
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def _pool_hidden_tensor(
    hidden,
    batch_specs: list[dict[str, tuple[str, tuple[int, int] | int] | None]],
    names: list[str],
    sequence_lengths: list[int],
):
    """Pool all requested positions before transferring one compact tensor to CPU."""
    import torch

    batch_size, _, hidden_size = hidden.shape
    pooled = torch.zeros(
        (batch_size, len(names), hidden_size),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    valid = torch.zeros(
        (batch_size, len(names)),
        dtype=torch.bool,
        device=hidden.device,
    )

    for sample_index, (specs, sequence_length) in enumerate(
        zip(batch_specs, sequence_lengths)
    ):
        for position_index, name in enumerate(names):
            spec = specs.get(name)
            if spec is None:
                continue
            mode, payload = spec
            if mode == "index":
                token_index = int(payload)
                if 0 <= token_index < sequence_length:
                    pooled[sample_index, position_index] = hidden[
                        sample_index, token_index
                    ]
                    valid[sample_index, position_index] = True
            else:
                start, end = int(payload[0]), int(payload[1])
                start = max(0, start)
                end = min(sequence_length, end)
                if end > start:
                    pooled[sample_index, position_index] = hidden[
                        sample_index, start:end
                    ].mean(dim=0)
                    valid[sample_index, position_index] = True
    return pooled, valid


def _register_selected_hidden_state_hooks(
    *,
    model,
    selected_hidden_state_ids: list[int],
    names: list[str],
    state: dict[str, Any],
):
    """Capture HF-style hidden-state indices without output_hidden_states=True.

    For Qwen2/Qwen2.5 with N blocks:
      0       = embedding output
      1..N-1  = output of blocks 0..N-2
      N       = final normalized hidden state
    """
    embedding, layers, final_norm = _locate_transformer_modules(model)
    num_layers = len(layers)
    handles = []

    def make_hook(hidden_state_id: int):
        def hook(_module, _inputs, output):
            tensor = _tensor_from_hook_output(output)
            pooled, valid = _pool_hidden_tensor(
                tensor,
                state["batch_specs"],
                names,
                state["sequence_lengths"],
            )
            state["captured"][hidden_state_id] = (
                pooled.detach().float().cpu().numpy(),
                valid.detach().cpu().numpy(),
            )

        return hook

    for hidden_state_id in selected_hidden_state_ids:
        if hidden_state_id == 0:
            module = embedding
        elif 1 <= hidden_state_id < num_layers:
            module = layers[hidden_state_id - 1]
        elif hidden_state_id == num_layers:
            module = final_norm
        else:
            raise ValueError(
                f"Hidden-state index {hidden_state_id} is invalid for {num_layers} layers"
            )
        handles.append(module.register_forward_hook(make_hook(hidden_state_id)))
    return handles


def extract_activations(
    *,
    run_dir: str | Path,
    model_name: str,
    layers: str = "auto",
    positions: str = "default",
    device: str = "cuda:0",
    dtype: str = "auto",
    include_repaired: bool = False,
    include_parse_errors: bool = False,
    max_rows: int | None = None,
    trust_remote_code: bool = False,
    batch_size: int = 1,
    steps_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install activation dependencies with pip install -e '.[activation]'"
        ) from exc

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    run = Path(run_dir)
    source_steps = Path(steps_path) if steps_path else run / "steps.jsonl"
    rows = []
    for row in read_jsonl(source_steps):
        if row.get("parse_error") and not include_parse_errors:
            continue
        if row.get("repaired") and not include_repaired:
            continue
        if row.get("prompt_text"):
            rows.append(row)
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        raise ValueError("No eligible rows")

    names = parse_positions(positions)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    prepared = []
    for original_index, row in enumerate(rows):
        prompt = str(row["prompt_text"])
        response = str(row.get("raw_response") or "")
        specs, sequence_length = _position_specs(
            tokenizer,
            prompt,
            response,
            names,
        )
        prepared.append(
            {
                "original_index": original_index,
                "row": row,
                "full_text": prompt + response,
                "specs": specs,
                "sequence_length": sequence_length,
            }
        )

    # Length bucketing reduces padding while writes still use original row order.
    prepared.sort(key=lambda item: item["sequence_length"])

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    torch_dtype = _torch_dtype(dtype)
    model_kwargs["torch_dtype"] = "auto" if torch_dtype == "auto" else torch_dtype
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    total_hidden_states = num_layers + 1
    layer_ids = _resolve_layers(layers, total_hidden_states)
    hidden_size = int(model.config.hidden_size)

    out = Path(output_dir) if output_dir else run / "activations"
    out.mkdir(parents=True, exist_ok=True)
    storage_dtype = (
        np.float16 if dtype in {"auto", "float16", "bfloat16"} else np.float32
    )
    X = np.lib.format.open_memmap(
        out / "X.npy",
        mode="w+",
        dtype=storage_dtype,
        shape=(len(rows), len(names), len(layer_ids), hidden_size),
    )
    mask = np.lib.format.open_memmap(
        out / "position_mask.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(len(rows), len(names)),
    )
    X[:] = 0
    mask[:] = False

    state: dict[str, Any] = {
        "batch_specs": [],
        "sequence_lengths": [],
        "captured": {},
    }
    handles = _register_selected_hidden_state_hooks(
        model=model,
        selected_hidden_state_ids=layer_ids,
        names=names,
        state=state,
    )

    try:
        with torch.inference_mode():
            for batch_start in range(0, len(prepared), batch_size):
                batch = prepared[batch_start : batch_start + batch_size]
                texts = [item["full_text"] for item in batch]
                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                state["batch_specs"] = [item["specs"] for item in batch]
                state["sequence_lengths"] = [
                    item["sequence_length"] for item in batch
                ]
                state["captured"] = {}

                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )

                missing_layers = sorted(set(layer_ids) - set(state["captured"]))
                if missing_layers:
                    raise RuntimeError(
                        f"Hooks did not capture hidden-state indices: {missing_layers}"
                    )

                for local_index, item in enumerate(batch):
                    destination = item["original_index"]
                    for selected_layer_index, hidden_state_id in enumerate(layer_ids):
                        pooled, valid = state["captured"][hidden_state_id]
                        X[destination, :, selected_layer_index] = pooled[
                            local_index
                        ].astype(storage_dtype)
                        mask[destination] |= valid[local_index]

                processed = min(batch_start + len(batch), len(prepared))
                if processed % 10 == 0 or processed == len(prepared):
                    print(
                        f"[activations] processed {processed}/{len(prepared)} "
                        f"batch_size={batch_size}",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()

    X.flush()
    mask.flush()
    np.save(out / "layers.npy", np.asarray(layer_ids, dtype=np.int64))
    write_json(out / "positions.json", names)

    meta = []
    for row_index, row in enumerate(rows):
        meta.append(
            {
                "row_index": row_index,
                "episode_id": row["episode_id"],
                "step_id": int(row["step_id"]),
                "parse_error": bool(row.get("parse_error")),
                "repaired": bool(row.get("repaired")),
            }
        )
    write_jsonl(out / "meta.jsonl", meta)
    write_manifest(
        out / "manifest.json",
        stage="activations",
        config={
            "model": model_name,
            "layers": layers,
            "positions": positions,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "capture_mode": "forward_hooks",
            "steps_path": str(source_steps),
        },
        counts={
            "rows": len(rows),
            "positions": len(names),
            "layers": len(layer_ids),
            "hidden_size": hidden_size,
        },
        upstream={"steps": str(source_steps)},
        extra={"shape": list(X.shape)},
    )
    return {
        "shape": tuple(X.shape),
        "positions": names,
        "layers": layer_ids,
        "output_dir": str(out),
    }
