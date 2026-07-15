from __future__ import annotations
import ast, json, re
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
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                yield value
        except json.JSONDecodeError:
            continue

def parse_response(text: str, size: int) -> ParseResult:
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
    grid = data.get("belief_grid")
    if not isinstance(grid, list) or len(grid) != size:
        return ParseResult(None, f"belief_grid must have {size} rows")
    normalized = []
    for row in grid:
        if not isinstance(row, list) or len(row) != size:
            return ParseResult(None, f"Each belief_grid row must have {size} entries")
        values = [str(x).upper().strip() for x in row]
        if any(x not in {"O", "F", "U"} for x in values):
            return ParseResult(None, "belief_grid labels must be O, F, or U")
        normalized.append(values)
    clean = dict(data)
    clean["action"] = action
    clean["belief_grid"] = normalized
    return ParseResult(clean, None)

def repair_messages(raw_response: str, size: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Repair the response into valid JSON only. Do not add commentary."},
        {"role": "user", "content": (
            f"Return a {size}x{size} belief_grid using O/F/U and an action in "
            "UP/DOWN/LEFT/RIGHT. Malformed response:\\n" + raw_response
        )},
    ]
