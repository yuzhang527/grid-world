from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from grid_world.activations.positions import (
    action_value_span, belief_output_span, parse_positions, tag_span
)
from grid_world.utils.io import read_jsonl, write_json, write_jsonl
from grid_world.utils.manifest import write_manifest

def _torch_dtype(name):
    import torch
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}.get(name, "auto")

def _layers(value: str, total: int):
    if value == "all":
        return list(range(total))
    if value == "auto":
        return sorted(set([0, total//4, total//2, 3*total//4, total-1]))
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result or min(result) < 0 or max(result) >= total:
        raise ValueError(f"Layer indices must be in [0,{total-1}]")
    return result

def _token_span(tokenizer, text, span):
    if span is None or span.end <= span.start:
        return None
    start = len(tokenizer(text[:span.start], add_special_tokens=False).input_ids)
    end = len(tokenizer(text[:span.end], add_special_tokens=False).input_ids)
    return (start, end) if end > start else None

def _specs(tokenizer, prompt, response, names):
    full = prompt + response
    prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    full_len = len(tokenizer(full, add_special_tokens=False).input_ids)
    tags = {
        "mean_last_feedback":"last_feedback","after_last_feedback":"last_feedback",
        "mean_required_belief_updates":"required_belief_updates",
        "after_required_belief_updates":"required_belief_updates",
        "mean_available_actions":"available_actions",
        "after_available_actions":"available_actions",
        "mean_current_belief_grid":"current_belief_grid",
        "after_current_belief_grid":"current_belief_grid",
        "mean_history":"history","after_history":"history",
    }
    action_tokens = _token_span(tokenizer, full, action_value_span(response, len(prompt)))
    belief_tokens = _token_span(tokenizer, full, belief_output_span(response, len(prompt)))
    result = {}
    for name in names:
        if name in {"prompt_last","pre_response"}:
            result[name] = ("index", max(0, prompt_len-1))
        elif name == "mean_all_prompt":
            result[name] = ("mean", (0, prompt_len))
        elif name == "response_last":
            result[name] = ("index", max(0, full_len-1))
        elif name in tags:
            tokens = _token_span(tokenizer, full, tag_span(prompt, tags[name]))
            if tokens is None:
                result[name] = None
            elif name.startswith("after_"):
                result[name] = ("index", tokens[1]-1)
            else:
                result[name] = ("mean", tokens)
        elif name == "first_action_token":
            result[name] = ("index", action_tokens[0]) if action_tokens else None
        elif name == "pre_action_token":
            result[name] = ("index", max(0, action_tokens[0]-1)) if action_tokens else None
        elif name == "mean_output_action":
            result[name] = ("mean", action_tokens) if action_tokens else None
        elif name == "mean_output_belief_grid":
            result[name] = ("mean", belief_tokens) if belief_tokens else None
        else:
            result[name] = None
    return result

def extract_activations(*, run_dir: str | Path, model_name: str, layers="auto",
                        positions="default", device="cuda:0", dtype="auto",
                        include_repaired=False, include_parse_errors=False,
                        max_rows: int | None = None, trust_remote_code=False):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install activation dependencies with pip install -e '.[activation]'") from exc
    run = Path(run_dir)
    rows = []
    for row in read_jsonl(run / "steps.jsonl"):
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code,
                                              use_fast=True)
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "low_cpu_mem_usage": True}
    tdtype = _torch_dtype(dtype)
    kwargs["torch_dtype"] = "auto" if tdtype == "auto" else tdtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(device); model.eval()
    total_states = int(model.config.num_hidden_layers) + 1
    layer_ids = _layers(layers, total_states)
    hidden_size = int(model.config.hidden_size)
    out = run / "activations"; out.mkdir(parents=True, exist_ok=True)
    storage = np.float16 if dtype in {"auto","float16","bfloat16"} else np.float32
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=storage,
                                  shape=(len(rows),len(names),len(layer_ids),hidden_size))
    mask = np.lib.format.open_memmap(out / "position_mask.npy", mode="w+", dtype=np.bool_,
                                     shape=(len(rows),len(names)))
    X[:] = 0; mask[:] = False
    meta = []
    with torch.inference_mode():
        for row_index, row in enumerate(rows):
            prompt, response = str(row["prompt_text"]), str(row.get("raw_response") or "")
            full = prompt + response
            encoded = tokenizer(full, add_special_tokens=False, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention = encoded.get("attention_mask")
            if attention is not None:
                attention = attention.to(device)
            output = model(input_ids=input_ids, attention_mask=attention,
                           output_hidden_states=True, use_cache=False, return_dict=True)
            hidden = output.hidden_states
            specs = _specs(tokenizer, prompt, response, names)
            seq_len = input_ids.shape[1]
            for pidx, name in enumerate(names):
                spec = specs.get(name)
                if spec is None:
                    continue
                mode, payload = spec
                if mode == "index":
                    token = int(payload)
                    if not 0 <= token < seq_len:
                        continue
                    vectors = [hidden[layer][0,token].detach().float().cpu().numpy()
                               for layer in layer_ids]
                else:
                    start, end = max(0,int(payload[0])), min(seq_len,int(payload[1]))
                    if end <= start:
                        continue
                    vectors = [hidden[layer][0,start:end].mean(0).detach().float().cpu().numpy()
                               for layer in layer_ids]
                X[row_index,pidx] = np.stack(vectors).astype(storage)
                mask[row_index,pidx] = True
            meta.append({"row_index":row_index,"episode_id":row["episode_id"],
                         "step_id":int(row["step_id"]),
                         "parse_error":bool(row.get("parse_error")),
                         "repaired":bool(row.get("repaired"))})
            if (row_index+1) % 10 == 0 or row_index+1 == len(rows):
                print(f"[activations] processed {row_index+1}/{len(rows)}", flush=True)
    X.flush(); mask.flush()
    np.save(out / "layers.npy", np.asarray(layer_ids,dtype=np.int64))
    write_json(out / "positions.json", names)
    write_jsonl(out / "meta.jsonl", meta)
    write_manifest(out / "manifest.json", stage="activations",
                   config={"model":model_name,"layers":layers,"positions":positions,
                           "device":device,"dtype":dtype},
                   counts={"rows":len(rows),"positions":len(names),
                           "layers":len(layer_ids),"hidden_size":hidden_size},
                   upstream={"steps":str(run / "steps.jsonl")},
                   extra={"shape":list(X.shape)})
    return {"shape":tuple(X.shape),"positions":names,"layers":layer_ids}
