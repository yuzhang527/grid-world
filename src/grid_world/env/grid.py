from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
DELTAS = {"UP": (0, 1), "DOWN": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
Coord = tuple[int, int]

def as_coord(value: Iterable[int]) -> Coord:
    x, y = value
    return int(x), int(y)

@dataclass(frozen=True)
class GridSpec:
    episode_id: str
    seed: int
    size: int
    start: Coord
    goal: Coord
    obstacles: frozenset[Coord]
    shortest_path_length: int | None = None

    @classmethod
    def from_dict(cls, row: dict) -> "GridSpec":
        return cls(
            episode_id=str(row["episode_id"]),
            seed=int(row.get("seed", 0)),
            size=int(row["size"]),
            start=as_coord(row["start"]),
            goal=as_coord(row["goal"]),
            obstacles=frozenset(as_coord(x) for x in row.get("obstacles", [])),
            shortest_path_length=int(row["shortest_path_length"])
            if row.get("shortest_path_length") is not None else None,
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": "1.0",
            "episode_id": self.episode_id,
            "seed": self.seed,
            "size": self.size,
            "start": list(self.start),
            "goal": list(self.goal),
            "obstacles": [list(x) for x in sorted(self.obstacles)],
            "shortest_path_length": self.shortest_path_length,
        }

class GridWorld:
    def __init__(self, spec: GridSpec):
        self.spec = spec
        self.position = spec.start
        self.steps = 0

    def in_bounds(self, coord: Coord) -> bool:
        x, y = coord
        return 0 <= x < self.spec.size and 0 <= y < self.spec.size

    def is_free(self, coord: Coord) -> bool:
        return self.in_bounds(coord) and coord not in self.spec.obstacles

    def target(self, action: str, position: Coord | None = None) -> Coord:
        if action not in DELTAS:
            raise ValueError(f"Unknown action: {action}")
        x, y = self.position if position is None else position
        dx, dy = DELTAS[action]
        return x + dx, y + dy

    def available_actions(self, position: Coord | None = None) -> list[str]:
        return [a for a in ACTIONS if self.is_free(self.target(a, position))]

    def feedback(self, position: Coord | None = None) -> dict:
        pos = self.position if position is None else position
        blocked, free, wall = [], [], []
        for action in ACTIONS:
            target = self.target(action, pos)
            item = list(target)
            if not self.in_bounds(target):
                wall.append(item)
            elif target in self.spec.obstacles:
                blocked.append(item)
            else:
                free.append(item)
        return {
            "type": "adjacent_exact",
            "coordinate_system": "cartesian_bottom_left",
            "position": list(pos),
            "blocked": blocked,
            "free": free,
            "wall": wall,
        }

    def step(self, action: str) -> tuple[Coord, bool]:
        self.steps += 1
        target = self.target(action)
        valid = self.is_free(target)
        if valid:
            self.position = target
        return self.position, valid

    @property
    def reached_goal(self) -> bool:
        return self.position == self.spec.goal

def render_grid(spec: GridSpec, position: Coord | None = None) -> str:
    lines = []
    for y in range(spec.size - 1, -1, -1):
        cells = []
        for x in range(spec.size):
            coord = (x, y)
            if position == coord:
                cell = "A"
            elif coord == spec.start:
                cell = "S"
            elif coord == spec.goal:
                cell = "G"
            elif coord in spec.obstacles:
                cell = "#"
            else:
                cell = "."
            cells.append(cell)
        lines.append(f"y={y:<2} " + " ".join(cells))
    lines.append("     " + " ".join(str(x) for x in range(spec.size)))
    return "\n".join(lines)
