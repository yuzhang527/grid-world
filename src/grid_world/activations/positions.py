from __future__ import annotations
import re
from dataclasses import dataclass

DEFAULT_POSITIONS = [
    "prompt_last","mean_all_prompt","mean_last_feedback","after_last_feedback",
    "mean_required_belief_updates","mean_available_actions",
    "mean_current_belief_grid","after_current_belief_grid","mean_history",
    "pre_action_token","first_action_token",
]
ALL_POSITIONS = DEFAULT_POSITIONS + [
    "pre_response","after_required_belief_updates","after_available_actions",
    "after_history","mean_output_action","mean_output_belief_grid","response_last",
]

@dataclass(frozen=True)
class Span:
    start: int
    end: int

def tag_span(text: str, tag: str) -> Span | None:
    match = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, re.DOTALL)
    return Span(match.start(1), match.end(1)) if match else None

def action_value_span(response: str, offset: int = 0) -> Span | None:
    match = re.search(r'"action"\s*:\s*"([^"]+)"', response, re.IGNORECASE)
    return Span(offset + match.start(1), offset + match.end(1)) if match else None

def belief_output_span(response: str, offset: int = 0) -> Span | None:
    match = re.search(r'"belief_grid"\s*:\s*(\[\s*\[.*?\]\s*\])',
                      response, re.DOTALL | re.IGNORECASE)
    return Span(offset + match.start(1), offset + match.end(1)) if match else None

def parse_positions(value: str) -> list[str]:
    if value == "default":
        return list(DEFAULT_POSITIONS)
    if value == "all":
        return list(ALL_POSITIONS)
    items = [x.strip() for x in value.split(",") if x.strip()]
    unknown = sorted(set(items) - set(ALL_POSITIONS))
    if unknown:
        raise ValueError(f"Unknown positions: {unknown}")
    return items
