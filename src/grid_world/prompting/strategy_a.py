from __future__ import annotations
import json
from typing import Any
from grid_world.env.belief import belief_to_rows
from grid_world.env.grid import GridSpec

SYSTEM_PROMPT = """You are navigating a partially observed grid world.
Use Cartesian coordinates [x,y] with [0,0] at the bottom-left.
Update the persistent belief grid from all information available.
Rows are top-to-bottom. Use O for obstacle, F for free, and U for unknown.
Choose exactly one action from available_actions.
Return JSON only with thought, nl_obstacles, belief_grid, and action."""

def build_messages(*, spec: GridSpec, position: tuple[int, int], feedback: dict[str, Any],
                   model_belief: dict, history: list[dict[str, Any]],
                   available_actions: list[str]) -> list[dict[str, str]]:
    required_updates = {
        "mark_free": feedback.get("free", []),
        "mark_obstacle": feedback.get("blocked", []),
        "ignore_out_of_bounds": feedback.get("wall", []),
    }
    compact_history = [{
        "step_id": x["step_id"], "position": x["position"],
        "action": x["action"], "next_position": x["next_position"]
    } for x in history[-12:]]
    user = f"""<task>
Navigate to the goal while maintaining a correct persistent belief grid.
</task>
<grid>
size={spec.size}
start={json.dumps(list(spec.start))}
goal={json.dumps(list(spec.goal))}
coordinate_system=cartesian_bottom_left
belief_row_order=topdown_y_desc
</grid>
<current_position>
{json.dumps(list(position))}
</current_position>
<last_feedback>
{json.dumps(feedback, ensure_ascii=False)}
</last_feedback>
<required_belief_updates>
{json.dumps(required_updates, ensure_ascii=False)}
</required_belief_updates>
<available_actions>
{json.dumps(available_actions)}
</available_actions>
<current_belief_grid>
{json.dumps(belief_to_rows(model_belief, spec.size))}
</current_belief_grid>
<history>
{json.dumps(compact_history, ensure_ascii=False)}
</history>
Return JSON only. The belief_grid must contain exactly {spec.size} rows of {spec.size} O/F/U labels."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

def render_plain(messages: list[dict[str, str]]) -> str:
    chunks = [f"<|{m['role']}|>\n{m['content']}\n" for m in messages]
    chunks.append("<|assistant|>\n")
    return "".join(chunks)
