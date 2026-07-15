from __future__ import annotations
from grid_world.env.grid import Coord, GridSpec
Belief = dict[Coord, str]

def initial_belief(spec: GridSpec) -> Belief:
    belief = {(x, y): "U" for x in range(spec.size) for y in range(spec.size)}
    belief[spec.start] = "F"
    belief[spec.goal] = "F"
    return belief

def update_belief(belief: Belief, feedback: dict) -> Belief:
    result = dict(belief)
    position = tuple(feedback.get("position", []))
    if len(position) == 2:
        result[(int(position[0]), int(position[1]))] = "F"
    for coord in feedback.get("free", []):
        result[(int(coord[0]), int(coord[1]))] = "F"
    for coord in feedback.get("blocked", []):
        result[(int(coord[0]), int(coord[1]))] = "O"
    return result

def belief_to_rows(belief: Belief, size: int) -> list[list[str]]:
    return [[belief[(x, y)] for x in range(size)] for y in range(size - 1, -1, -1)]

def rows_to_belief(rows: list[list[str]], size: int) -> Belief:
    if len(rows) != size or any(len(row) != size for row in rows):
        raise ValueError(f"Expected a {size}x{size} belief grid")
    belief = {}
    for row_idx, row in enumerate(rows):
        y = size - 1 - row_idx
        for x, value in enumerate(row):
            value = str(value).upper()
            if value not in {"O", "F", "U"}:
                raise ValueError(f"Invalid belief label {value!r}")
            belief[(x, y)] = value
    return belief
