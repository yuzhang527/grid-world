from __future__ import annotations

import json
from typing import Any

from grid_world.env.grid import GridSpec


SYSTEM_PROMPT = """You are navigating a partially observed grid world.
Use Cartesian coordinates [x,y] with [0,0] at the bottom-left.
Use the complete observation history to choose the next move.
Choose exactly one action from available_actions.
Do not output a map, table, belief state, reasoning, or commentary.
Return JSON only in the form {"action":"UP"}."""


def build_messages(
    *,
    spec: GridSpec,
    position: tuple[int, int],
    feedback: dict[str, Any],
    model_belief: dict,
    history: list[dict[str, Any]],
    available_actions: list[str],
) -> list[dict[str, str]]:
    del model_belief  # The no-grid prompt never exposes this state.

    observation_history = [
        {
            "step_id": item["step_id"],
            "position": item["position"],
            "feedback": item.get("feedback"),
            "action": item["action"],
            "next_position": item["next_position"],
        }
        for item in history
    ]

    user = f"""<task>
Navigate to the goal using the observations received so far.
Do not output a map or any intermediate reasoning.
</task>
<environment>
size={spec.size}
start={json.dumps(list(spec.start))}
goal={json.dumps(list(spec.goal))}
coordinate_system=cartesian_bottom_left
</environment>
<current_position>
{json.dumps(list(position))}
</current_position>
<last_feedback>
{json.dumps(feedback, ensure_ascii=False)}
</last_feedback>
<available_actions>
{json.dumps(available_actions)}
</available_actions>
<history>
{json.dumps(observation_history, ensure_ascii=False)}
</history>
Return JSON only in the form {{"action":"UP"}}."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
