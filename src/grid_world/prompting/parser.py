from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParseResult:
    data: dict[str, Any] | None
    error: str | None


def _json_candidates(text: str):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
            if isinstance(value, dict):
                yield value
        except json.JSONDecodeError:
            continue


def parse_response(
    text: str,
    size: int,
    *,
    require_belief_grid: bool = True,
) -> ParseResult:
    data = next(_json_candidates(text), None)
    if data is None:
        try:
            value = ast.literal_eval(text.strip())
            data = value if isinstance(value, dict) else None
        except Exception:
            data = None
    if data is None:
        return ParseResult(None, "No JSON object found")

    action = str(data.get("action", "")).upper().strip()
    if action not in {"UP", "DOWN", "LEFT", "RIGHT"}:
        return ParseResult(None, f"Invalid action: {action!r}")

    if not require_belief_grid:
        # Deliberately discard any extra map or reasoning fields. The no-grid
        # condition is scored and replayed as an action-only condition.
        return ParseResult({"action": action}, None)

    grid = data.get("belief_grid")
    if not isinstance(grid, list) or len(grid) != size:
        return ParseResult(None, f"belief_grid must have {size} rows")

    normalized = []
    for row in grid:
        if not isinstance(row, list) or len(row) != size:
            return ParseResult(
                None,
                f"Each belief_grid row must have {size} entries",
            )
        values = [str(value).upper().strip() for value in row]
        if any(value not in {"O", "F", "U"} for value in values):
            return ParseResult(
                None,
                "belief_grid labels must be O, F, or U",
            )
        normalized.append(values)

    clean = dict(data)
    clean["action"] = action
    clean["belief_grid"] = normalized
    return ParseResult(clean, None)


def repair_messages(
    raw_response: str,
    size: int,
    *,
    require_belief_grid: bool = True,
) -> list[dict[str, str]]:
    if require_belief_grid:
        instruction = (
            f"Return a {size}x{size} belief_grid using O/F/U and an action in "
            "UP/DOWN/LEFT/RIGHT. Malformed response:\n"
        )
    else:
        instruction = (
            'Return exactly one JSON object such as {"action":"UP"}. '
            "The action must be UP, DOWN, LEFT, or RIGHT. "
            "Do not add reasoning, a map, or commentary. Malformed response:\n"
        )
    return [
        {
            "role": "system",
            "content": (
                "Repair the response into valid JSON only. "
                "Do not add commentary."
            ),
        },
        {
            "role": "user",
            "content": instruction + raw_response,
        },
    ]
