#!/usr/bin/env python3
"""
Coordinate-Belief v4 trajectory generator.

This is an additive generator for the grid-world repository.  It changes only
the explicit belief protocol seen by the model:

    {
      "belief_coordinates": {
        "F": [[x, y], ...],
        "O": [[x, y], ...]
      },
      "action": "UP"
    }

Unknown cells are omitted and therefore implicitly U.  The raw model response
never needs to contain a matrix.  For compatibility with the existing target,
activation, probe, and trajectory-viewer pipeline, each step additionally
stores a normalized ``parsed_belief_grid`` using grid[y][x] with y=0 as the
Cartesian bottom row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ACTIONS: dict[str, tuple[int, int]] = {
    "UP": (0, 1),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
ACTION_ORDER = ("UP", "RIGHT", "DOWN", "LEFT")
VALID_STATES = {"F", "O", "U"}
SCHEMA_VERSION = "coordinate-belief-v4.1"

SYSTEM_PROMPT = """You are navigating a partially observed Cartesian grid world.

Coordinate system:
- Coordinates are (x,y).
- x increases to the RIGHT.
- y increases UPWARD.
- (0,0) is the bottom-left cell.
- UP=(0,+1), DOWN=(0,-1), LEFT=(-1,0), RIGHT=(+1,0).

Maintain an explicit belief using COORDINATES, never a row/column matrix.
Return exactly one JSON object with this schema:
{
  "belief_coordinates": {
    "F": [[x,y], ...],
    "O": [[x,y], ...]
  },
  "action": "UP"
}

Belief rules:
- F contains every coordinate you currently believe is known free.
- O contains every coordinate you currently believe is a known obstacle.
- Unknown coordinates are omitted; omission means U.
- A coordinate must not appear in both F and O.
- Use integer coordinates inside the stated map bounds.
- Preserve previously known facts unless newer exact feedback contradicts them.
- The current position is free.
- Do not output a 2-D array, ASCII map, explanation, markdown, or extra keys.

Action rules:
- action must be one of UP, DOWN, LEFT, RIGHT.
- choose only from <available_actions>.
- try to reach the goal efficiently while avoiding repeated loops.
"""


def json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json_dumps(dict(row)) + "\n")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json_dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def as_coord(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            value = [value["x"], value["y"]]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be [x,y], got {value!r}")
    try:
        x, y = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integers, got {value!r}") from exc
    return x, y


def infer_episode_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("episode_id", "id", "map_id", "name"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    seed = row.get("seed")
    if seed is not None:
        return f"coord_v4_seed{seed}"
    return f"coord_v4_ep{index:06d}"


def parse_matrix_obstacles(
    matrix: Sequence[Sequence[Any]],
    width: int,
    height: int,
) -> set[tuple[int, int]]:
    if len(matrix) != height:
        raise ValueError(f"Matrix height {len(matrix)} != expected {height}")
    obstacles: set[tuple[int, int]] = set()
    obstacle_tokens = {"O", "X", "#", "BLOCKED", "OBSTACLE", 1, True}
    for y, row in enumerate(matrix):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != width:
            raise ValueError(f"Matrix row {y} does not have width={width}")
        for x, value in enumerate(row):
            normalized = value.upper() if isinstance(value, str) else value
            if normalized in obstacle_tokens:
                obstacles.add((x, y))
    return obstacles


@dataclass(frozen=True)
class MapSpec:
    episode_id: str
    width: int
    height: int
    start: tuple[int, int]
    goal: tuple[int, int]
    obstacles: frozenset[tuple[int, int]]
    source: dict[str, Any]

    def in_bounds(self, coord: tuple[int, int]) -> bool:
        x, y = coord
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, coord: tuple[int, int]) -> bool:
        return self.in_bounds(coord) and coord not in self.obstacles

    def true_grid(self) -> list[list[str]]:
        return [
            ["O" if (x, y) in self.obstacles else "F" for x in range(self.width)]
            for y in range(self.height)
        ]


def parse_map(row: Mapping[str, Any], index: int) -> MapSpec:
    size = row.get("size", row.get("grid_size"))
    width = int(row.get("width", size if size is not None else 5))
    height = int(row.get("height", size if size is not None else width))
    start = as_coord(
        row.get("start", row.get("start_pos", row.get("source", [0, 0]))),
        "start",
    )
    goal = as_coord(
        row.get("goal", row.get("goal_pos", row.get("target", [width - 1, height - 1]))),
        "goal",
    )

    obstacle_value = None
    for key in ("obstacles", "blocked", "blocked_cells", "obstacle_coords"):
        if key in row:
            obstacle_value = row[key]
            break

    obstacles: set[tuple[int, int]]
    if obstacle_value is not None:
        if not isinstance(obstacle_value, Sequence) or isinstance(obstacle_value, (str, bytes)):
            raise ValueError("obstacles must be a coordinate list")
        obstacles = {as_coord(item, "obstacle") for item in obstacle_value}
    else:
        matrix = row.get("true_map", row.get("grid", row.get("map")))
        if matrix is None:
            raise ValueError(
                "Map row needs obstacles/blocked coordinates or a true_map/grid matrix."
            )
        if isinstance(matrix, Mapping):
            obstacles = {
                as_coord(key, "map coordinate")
                for key, value in matrix.items()
                if str(value).upper() in {"O", "X", "#", "BLOCKED", "OBSTACLE"}
            }
        else:
            obstacles = parse_matrix_obstacles(matrix, width, height)

    episode_id = infer_episode_id(row, index)
    spec = MapSpec(
        episode_id=episode_id,
        width=width,
        height=height,
        start=start,
        goal=goal,
        obstacles=frozenset(obstacles),
        source=dict(row),
    )
    if not spec.in_bounds(start) or not spec.in_bounds(goal):
        raise ValueError(f"{episode_id}: start/goal out of bounds")
    if start in obstacles or goal in obstacles:
        raise ValueError(f"{episode_id}: start or goal is an obstacle")
    for obstacle in obstacles:
        if not spec.in_bounds(obstacle):
            raise ValueError(f"{episode_id}: obstacle {obstacle} out of bounds")
    return spec


def shortest_path_length(spec: MapSpec) -> int | None:
    queue: deque[tuple[tuple[int, int], int]] = deque([(spec.start, 0)])
    seen = {spec.start}
    while queue:
        coord, distance = queue.popleft()
        if coord == spec.goal:
            return distance
        for dx, dy in ACTIONS.values():
            nxt = (coord[0] + dx, coord[1] + dy)
            if nxt not in seen and spec.is_free(nxt):
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def normalize_action(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    action = value.strip().upper()
    aliases = {
        "U": "UP",
        "D": "DOWN",
        "L": "LEFT",
        "R": "RIGHT",
        "NORTH": "UP",
        "SOUTH": "DOWN",
        "WEST": "LEFT",
        "EAST": "RIGHT",
    }
    action = aliases.get(action, action)
    return action if action in ACTIONS else None


def state_of_neighbor(spec: MapSpec, coord: tuple[int, int], action: str) -> dict[str, Any]:
    dx, dy = ACTIONS[action]
    nxt = (coord[0] + dx, coord[1] + dy)
    if not spec.in_bounds(nxt):
        state = "WALL"
    elif nxt in spec.obstacles:
        state = "O"
    else:
        state = "F"
    return {"coord": [nxt[0], nxt[1]], "state": state}


def local_feedback(spec: MapSpec, coord: tuple[int, int]) -> dict[str, dict[str, Any]]:
    return {action: state_of_neighbor(spec, coord, action) for action in ACTION_ORDER}


def available_actions_from_feedback(feedback: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [action for action in ACTION_ORDER if feedback[action]["state"] == "F"]


def coord_sort_key(coord: tuple[int, int]) -> tuple[int, int]:
    return coord[1], coord[0]


def belief_to_payload(belief: Mapping[tuple[int, int], str]) -> dict[str, list[list[int]]]:
    return {
        "F": [[x, y] for x, y in sorted(
            (coord for coord, state in belief.items() if state == "F"),
            key=coord_sort_key,
        )],
        "O": [[x, y] for x, y in sorted(
            (coord for coord, state in belief.items() if state == "O"),
            key=coord_sort_key,
        )],
    }


def belief_to_grid(
    belief: Mapping[tuple[int, int], str],
    width: int,
    height: int,
) -> list[list[str]]:
    grid = [["U" for _ in range(width)] for _ in range(height)]
    for (x, y), state in belief.items():
        if 0 <= x < width and 0 <= y < height and state in {"F", "O"}:
            grid[y][x] = state
    return grid


def required_updates(
    current: tuple[int, int],
    feedback: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    updates = [{"coord": [current[0], current[1]], "state": "F", "source": "current_position"}]
    for action in ACTION_ORDER:
        item = feedback[action]
        if item["state"] in {"F", "O"}:
            updates.append(
                {
                    "coord": list(item["coord"]),
                    "state": item["state"],
                    "source": f"last_feedback.{action}",
                }
            )
    return updates


def build_prompt(state: "EpisodeState") -> str:
    feedback = local_feedback(state.spec, state.current)
    available = available_actions_from_feedback(feedback)
    current_belief_payload = {
        "format": "coordinate_sets",
        "F": belief_to_payload(state.model_belief)["F"],
        "O": belief_to_payload(state.model_belief)["O"],
        "implicit_U": "Every in-bounds coordinate omitted from F and O",
    }
    history_payload = state.history[-20:]
    user_prompt = f"""<task>
Navigate to the goal while maintaining the coordinate-based belief.
</task>

<map>
width={state.spec.width}
height={state.spec.height}
start={list(state.spec.start)}
goal={list(state.spec.goal)}
coordinate_system=Cartesian(x-right,y-up,origin-bottom-left)
</map>

<step>
episode_id={state.spec.episode_id}
step_id={state.step_id}
current_position={list(state.current)}
</step>

<last_feedback>
{json_dumps(feedback, indent=2)}
</last_feedback>

<required_belief_updates>
{json_dumps(required_updates(state.current, feedback), indent=2)}
</required_belief_updates>

<available_actions>
{json_dumps(available)}
</available_actions>

<current_belief_grid>
{json_dumps(current_belief_payload, indent=2)}
</current_belief_grid>

<history>
{json_dumps(history_payload, indent=2)}
</history>

Return only the required JSON object.  Do not output a matrix."""
    return user_prompt


def extract_first_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    decoder = json.JSONDecoder()
    starts = [match.start() for match in re.finditer(r"\{", candidate)]
    errors: list[str] = []
    for start in starts:
        try:
            value, _ = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No JSON object found" + (f": {errors[-1]}" if errors else ""))


def parse_coord_list(
    value: Any,
    *,
    label: str,
    width: int,
    height: int,
) -> set[tuple[int, int]]:
    if value is None:
        return set()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list of [x,y] coordinates")
    coords: set[tuple[int, int]] = set()
    for index, item in enumerate(value):
        coord = as_coord(item, f"{label}[{index}]")
        x, y = coord
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{label}[{index}]={coord} is out of bounds")
        coords.add(coord)
    return coords


def parse_belief_object(
    obj: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[dict[tuple[int, int], str], dict[str, list[list[int]]]]:
    container: Any = None
    for key in ("belief_coordinates", "belief", "coordinate_belief", "belief_by_coordinate"):
        if key in obj:
            container = obj[key]
            break
    if container is None:
        raise ValueError("Missing belief_coordinates")

    belief: dict[tuple[int, int], str] = {}
    if isinstance(container, Mapping) and any(key in container for key in ("F", "O", "free", "obstacles")):
        free = parse_coord_list(
            container.get("F", container.get("free", [])),
            label="belief_coordinates.F",
            width=width,
            height=height,
        )
        obstacles = parse_coord_list(
            container.get("O", container.get("obstacles", [])),
            label="belief_coordinates.O",
            width=width,
            height=height,
        )
        overlap = free & obstacles
        if overlap:
            raise ValueError(f"Coordinates appear in both F and O: {sorted(overlap)}")
        belief.update({coord: "F" for coord in free})
        belief.update({coord: "O" for coord in obstacles})
    elif isinstance(container, Mapping):
        for key, value in container.items():
            match = re.fullmatch(r"\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?\s*", str(key))
            if not match:
                raise ValueError(f"Invalid coordinate key {key!r}")
            coord = (int(match.group(1)), int(match.group(2)))
            if not (0 <= coord[0] < width and 0 <= coord[1] < height):
                raise ValueError(f"Coordinate {coord} is out of bounds")
            state = str(value).upper()
            if state == "U":
                continue
            if state not in {"F", "O"}:
                raise ValueError(f"Invalid state {value!r} at {coord}")
            belief[coord] = state
    elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        for index, item in enumerate(container):
            if not isinstance(item, Mapping):
                raise ValueError(f"belief[{index}] must be an object")
            coord = as_coord(
                item.get("coord", [item.get("x"), item.get("y")]),
                f"belief[{index}].coord",
            )
            if not (0 <= coord[0] < width and 0 <= coord[1] < height):
                raise ValueError(f"Coordinate {coord} is out of bounds")
            state = str(item.get("state", "")).upper()
            if state == "U":
                continue
            if state not in {"F", "O"}:
                raise ValueError(f"Invalid state {state!r} at {coord}")
            belief[coord] = state
    else:
        raise ValueError("Unsupported belief_coordinates representation")

    payload = belief_to_payload(belief)
    return belief, payload


@dataclass
class ParsedOutput:
    belief: dict[tuple[int, int], str]
    belief_payload: dict[str, list[list[int]]]
    action: str
    obj: dict[str, Any]


def parse_model_output(text: str, spec: MapSpec) -> ParsedOutput:
    obj = extract_first_json_object(text)
    action = normalize_action(obj.get("action", obj.get("move", obj.get("next_action"))))
    if action is None:
        raise ValueError("Missing or invalid action")
    belief, payload = parse_belief_object(obj, width=spec.width, height=spec.height)
    return ParsedOutput(belief=belief, belief_payload=payload, action=action, obj=dict(obj))


def build_repair_prompt(original_prompt: str, raw_response: str, error: str) -> str:
    return f"""{original_prompt}

<repair_request>
Your previous response could not be parsed.
error={error}
previous_response={json_dumps(raw_response)}

Return only:
{{"belief_coordinates":{{"F":[[x,y],...],"O":[[x,y],...]}},"action":"UP"}}
Unknown cells must be omitted.  Do not output a matrix or explanation.
</repair_request>"""


def apply_action(coord: tuple[int, int], action: str) -> tuple[int, int]:
    dx, dy = ACTIONS[action]
    return coord[0] + dx, coord[1] + dy


def fallback_action(spec: MapSpec, current: tuple[int, int], legal: Sequence[str]) -> str:
    if not legal:
        # This should not happen on a connected map, but keeps the record valid.
        return "UP"
    def key(action: str) -> tuple[int, int]:
        nxt = apply_action(current, action)
        manhattan = abs(nxt[0] - spec.goal[0]) + abs(nxt[1] - spec.goal[1])
        return manhattan, ACTION_ORDER.index(action)
    return min(legal, key=key)


@dataclass
class EpisodeState:
    spec: MapSpec
    current: tuple[int, int]
    step_id: int = 0
    model_belief: dict[tuple[int, int], str] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    parse_error_steps: int = 0
    repaired_steps: int = 0
    invalid_move_steps: int = 0
    done: bool = False
    success: bool = False
    termination_reason: str | None = None

    @classmethod
    def create(cls, spec: MapSpec) -> "EpisodeState":
        return cls(
            spec=spec,
            current=spec.start,
            model_belief={spec.start: "F"},
            trajectory=[spec.start],
        )


class GenerationBackend:
    def render_prompts(
        self,
        user_prompts: Sequence[str],
        *,
        repair: bool = False,
    ) -> list[str]:
        """Return the exact text that will be tokenized by the backend."""
        return [SYSTEM_PROMPT + "\n\n" + prompt for prompt in user_prompts]

    def generate(
        self,
        prompts: Sequence[str],
        contexts: Sequence[EpisodeState],
        *,
        repair: bool = False,
    ) -> list[str]:
        raise NotImplementedError


class MockBackend(GenerationBackend):
    def generate(
        self,
        prompts: Sequence[str],
        contexts: Sequence[EpisodeState],
        *,
        repair: bool = False,
    ) -> list[str]:
        outputs: list[str] = []
        for state in contexts:
            feedback = local_feedback(state.spec, state.current)
            belief = dict(state.model_belief)
            belief[state.current] = "F"
            for item in feedback.values():
                coord = tuple(item["coord"])
                if item["state"] in {"F", "O"} and state.spec.in_bounds(coord):
                    belief[coord] = item["state"]
            legal = available_actions_from_feedback(feedback)
            action = fallback_action(state.spec, state.current, legal)
            outputs.append(
                json_dumps(
                    {
                        "belief_coordinates": belief_to_payload(belief),
                        "action": action,
                    }
                )
            )
        return outputs


class VLLMBackend(GenerationBackend):
    def __init__(self, args: argparse.Namespace) -> None:
        gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
        if not gpu_ids:
            raise ValueError("--gpus must contain at least one GPU id")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Install it in the server environment or use --backend mock."
            ) from exc

        llm_kwargs: dict[str, Any] = {
            "model": args.model,
            "tensor_parallel_size": len(gpu_ids),
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
        }
        if args.download_dir:
            llm_kwargs["download_dir"] = args.download_dir
        self.llm = LLM(**llm_kwargs)
        self.SamplingParams = SamplingParams
        try:
            self.tokenizer = self.llm.get_tokenizer()
        except Exception:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                trust_remote_code=args.trust_remote_code,
            )
        self.normal_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        self.repair_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_tokens,
            seed=args.seed + 1,
        )

    def render_prompts(
        self,
        user_prompts: Sequence[str],
        *,
        repair: bool = False,
    ) -> list[str]:
        rendered: list[str] = []
        for user_prompt in user_prompts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            rendered.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return rendered

    def generate(
        self,
        prompts: Sequence[str],
        contexts: Sequence[EpisodeState],
        *,
        repair: bool = False,
    ) -> list[str]:
        params = self.repair_params if repair else self.normal_params
        results = self.llm.generate(list(prompts), params, use_tqdm=True)
        texts: list[str] = []
        for result in results:
            if not result.outputs:
                texts.append("")
            else:
                texts.append(result.outputs[0].text)
        return texts


def initialize_backend(args: argparse.Namespace) -> GenerationBackend:
    if args.backend == "mock":
        return MockBackend()
    if args.backend == "vllm":
        return VLLMBackend(args)
    raise ValueError(f"Unsupported backend: {args.backend}")


def normalize_map_row(spec: MapSpec) -> dict[str, Any]:
    row = dict(spec.source)
    row.update(
        {
            "episode_id": spec.episode_id,
            "width": spec.width,
            "height": spec.height,
            "size": spec.width if spec.width == spec.height else None,
            "start": list(spec.start),
            "goal": list(spec.goal),
            "obstacles": [list(coord) for coord in sorted(spec.obstacles, key=coord_sort_key)],
            "true_map": spec.true_grid(),
            "coordinate_system": "Cartesian(x-right,y-up,origin-bottom-left)",
        }
    )
    if row.get("size") is None:
        row.pop("size", None)
    return row


def prepare_run(
    run: Path,
    maps_path: Path,
    specs: Sequence[MapSpec],
    args: argparse.Namespace,
) -> set[str]:
    run.mkdir(parents=True, exist_ok=True)
    steps_path = run / "steps.jsonl"
    episodes_path = run / "episodes.jsonl"

    if args.overwrite:
        for path in (
            steps_path,
            episodes_path,
            run / "summary.json",
            run / "manifest.json",
            run / "resolved_config.json",
            run / "maps.jsonl",
        ):
            if path.exists():
                path.unlink()
    elif not args.resume and (steps_path.exists() or episodes_path.exists()):
        raise RuntimeError(
            f"{run} already contains trajectory output. Use --resume or --overwrite."
        )

    completed: set[str] = set()
    if args.resume and episodes_path.exists():
        for row in read_jsonl(episodes_path):
            completed.add(str(row.get("episode_id")))

    maps_out = run / "maps.jsonl"
    maps_out.write_text(
        "".join(json_dumps(normalize_map_row(spec)) + "\n" for spec in specs),
        encoding="utf-8",
    )

    resolved = {
        "schema_version": SCHEMA_VERSION,
        "condition": "coordinate_belief_v4",
        "maps": str(maps_path.resolve()),
        "run": str(run.resolve()),
        "model": args.model,
        "backend": args.backend,
        "gpus": args.gpus,
        "num_episodes": len(specs),
        "max_steps": args.max_steps,
        "output_schema": {
            "belief_coordinates": {"F": "[[x,y], ...]", "O": "[[x,y], ...]"},
            "implicit_unknown": True,
            "action": "UP|DOWN|LEFT|RIGHT",
        },
        "compatibility": {
            "parsed_belief_grid": "grid[y][x], y=0 is Cartesian bottom row",
            "legacy_pipeline_supported": True,
        },
    }
    atomic_write_json(run / "resolved_config.json", resolved)
    return completed


def step_row(
    state: EpisodeState,
    *,
    prompt: str,
    raw_response: str,
    repair_response: str | None,
    parsed: ParsedOutput | None,
    parse_error: bool,
    parse_error_message: str | None,
    repaired: bool,
    requested_action: str,
    executed_action: str,
    invalid_move: bool,
    before: tuple[int, int],
    after: tuple[int, int],
    feedback: Mapping[str, Any],
    legal: Sequence[str],
) -> dict[str, Any]:
    belief = parsed.belief if parsed is not None else dict(state.model_belief)
    payload = parsed.belief_payload if parsed is not None else belief_to_payload(belief)
    grid = belief_to_grid(belief, state.spec.width, state.spec.height)
    parsed_response = {
        "belief_coordinates": payload,
        # Internal compatibility field only.  It was not required in raw model output.
        "belief_grid": grid,
        "action": requested_action,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "condition": "coordinate_belief_v4",
        "belief_output_format": "known_coordinate_sets_implicit_U",
        "episode_id": state.spec.episode_id,
        "step_id": state.step_id,
        "width": state.spec.width,
        "height": state.spec.height,
        "start": list(state.spec.start),
        "goal": list(state.spec.goal),
        "current_pos": list(before),
        "position_before": list(before),
        "next_pos": list(after),
        "position_after": list(after),
        "last_feedback": feedback,
        "available_actions": list(legal),
        "prompt": prompt,
        "raw_response": raw_response,
        "repair_response": repair_response,
        "parsed_response": parsed_response,
        "parsed_belief_coordinates": payload,
        "parsed_belief_grid": grid,
        "requested_action": requested_action,
        "executed_action": executed_action,
        # Legacy environment action means the action actually executed.
        "action": executed_action,
        "parse_error": bool(parse_error),
        "parse_error_message": parse_error_message,
        "repaired": bool(repaired),
        "invalid_move": bool(invalid_move),
        "illegal_action_before_fallback": bool(invalid_move),
        "success": after == state.spec.goal,
        "done": after == state.spec.goal or state.step_id + 1 >= 1_000_000,
        "coordinate_system": "Cartesian(x-right,y-up,origin-bottom-left)",
    }


def finalize_episode(state: EpisodeState) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "condition": "coordinate_belief_v4",
        "episode_id": state.spec.episode_id,
        "success": state.success,
        "steps": state.step_id,
        "num_steps": state.step_id,
        "start": list(state.spec.start),
        "goal": list(state.spec.goal),
        "final_pos": list(state.current),
        "termination_reason": state.termination_reason,
        "trajectory": [list(coord) for coord in state.trajectory],
        "parse_error_steps": state.parse_error_steps,
        "repaired_steps": state.repaired_steps,
        "invalid_move_steps": state.invalid_move_steps,
        "shortest_path_length": shortest_path_length(state.spec),
        "width": state.spec.width,
        "height": state.spec.height,
        "obstacles": [list(coord) for coord in sorted(state.spec.obstacles, key=coord_sort_key)],
        "true_map": state.spec.true_grid(),
        "coordinate_system": "Cartesian(x-right,y-up,origin-bottom-left)",
    }


def summarize(run: Path) -> dict[str, Any]:
    episodes_path = run / "episodes.jsonl"
    episodes = read_jsonl(episodes_path) if episodes_path.exists() else []
    steps = read_jsonl(run / "steps.jsonl") if (run / "steps.jsonl").exists() else []
    successes = sum(bool(row.get("success")) for row in episodes)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "condition": "coordinate_belief_v4",
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "mean_steps": (
            sum(int(row.get("num_steps", row.get("steps", 0))) for row in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
        "step_rows": len(steps),
        "parse_error_steps": sum(bool(row.get("parse_error")) for row in steps),
        "repaired_steps": sum(bool(row.get("repaired")) for row in steps),
        "invalid_move_steps": sum(bool(row.get("invalid_move")) for row in steps),
        "termination_reasons": dict(Counter(str(row.get("termination_reason")) for row in episodes)),
        "belief_output_format": "known_coordinate_sets_implicit_U",
        "coordinate_system": "Cartesian(x-right,y-up,origin-bottom-left)",
    }
    for numerator_key, rate_key in (
        ("parse_error_steps", "parse_error_rate"),
        ("repaired_steps", "repaired_step_rate"),
        ("invalid_move_steps", "invalid_move_rate"),
    ):
        summary[rate_key] = summary[numerator_key] / len(steps) if steps else 0.0
    atomic_write_json(run / "summary.json", summary)
    return summary


def run_generation(args: argparse.Namespace) -> None:
    maps_path = Path(args.maps).resolve()
    run = Path(args.run).resolve()
    map_rows = read_jsonl(maps_path)
    if args.num_episodes is not None:
        map_rows = map_rows[: args.num_episodes]
    specs = [parse_map(row, index) for index, row in enumerate(map_rows)]
    if not specs:
        raise RuntimeError("No maps were loaded")

    ids = [spec.episode_id for spec in specs]
    duplicate_ids = [key for key, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise RuntimeError(f"Duplicate episode ids: {duplicate_ids[:10]}")

    disconnected = [spec.episode_id for spec in specs if shortest_path_length(spec) is None]
    if disconnected:
        raise RuntimeError(f"Disconnected maps: {disconnected[:10]}")

    completed = prepare_run(run, maps_path, specs, args)
    pending_specs = [spec for spec in specs if spec.episode_id not in completed]
    print(f"[coord-v4] maps={len(specs)} completed={len(completed)} pending={len(pending_specs)}")
    print("[coord-v4] belief protocol=coordinate sets F/O; omitted coordinates are U")
    print("[coord-v4] coordinates=Cartesian(x-right,y-up,origin-bottom-left)")

    if not pending_specs:
        summary = summarize(run)
        print(f"[coord-v4] nothing to do; success_rate={summary['success_rate']:.4f}")
        return

    backend = initialize_backend(args)
    states = [EpisodeState.create(spec) for spec in pending_specs]
    steps_path = run / "steps.jsonl"
    episodes_path = run / "episodes.jsonl"

    for global_step in range(args.max_steps):
        active = [state for state in states if not state.done]
        if not active:
            break
        user_prompts = [build_prompt(state) for state in active]
        prompts = backend.render_prompts(user_prompts, repair=False)
        raw_outputs = backend.generate(prompts, active, repair=False)
        if len(raw_outputs) != len(active):
            raise RuntimeError("Backend returned a different number of outputs than prompts")

        parsed_outputs: list[ParsedOutput | None] = [None] * len(active)
        parse_errors: list[str | None] = [None] * len(active)
        repair_indices: list[int] = []
        for index, (state, text) in enumerate(zip(active, raw_outputs)):
            try:
                parsed_outputs[index] = parse_model_output(text, state.spec)
            except Exception as exc:  # noqa: BLE001 - error is recorded in data.
                parse_errors[index] = f"{type(exc).__name__}: {exc}"
                if args.repair_invalid_json:
                    repair_indices.append(index)

        repair_outputs: dict[int, str] = {}
        if repair_indices:
            repair_user_prompts = [
                build_repair_prompt(
                    user_prompts[index],
                    raw_outputs[index],
                    parse_errors[index] or "parse error",
                )
                for index in repair_indices
            ]
            repair_prompts = backend.render_prompts(repair_user_prompts, repair=True)
            repair_contexts = [active[index] for index in repair_indices]
            repaired_texts = backend.generate(repair_prompts, repair_contexts, repair=True)
            for index, repaired_text in zip(repair_indices, repaired_texts):
                repair_outputs[index] = repaired_text
                try:
                    parsed_outputs[index] = parse_model_output(repaired_text, active[index].spec)
                except Exception as exc:  # noqa: BLE001
                    parse_errors[index] = (
                        f"{parse_errors[index]}; repair_failed={type(exc).__name__}: {exc}"
                    )

        for index, state in enumerate(active):
            prompt = prompts[index]
            raw_response = raw_outputs[index]
            parsed = parsed_outputs[index]
            repaired = index in repair_outputs and parsed is not None
            parse_error = parsed is None

            feedback = local_feedback(state.spec, state.current)
            legal = available_actions_from_feedback(feedback)
            before = state.current

            if parsed is None:
                requested_action = fallback_action(state.spec, state.current, legal)
                executed_action = requested_action
                belief_for_next = dict(state.model_belief)
                state.parse_error_steps += 1
            else:
                requested_action = parsed.action
                belief_for_next = dict(parsed.belief)
                if repaired:
                    state.repaired_steps += 1
                if requested_action in legal:
                    executed_action = requested_action
                else:
                    executed_action = fallback_action(state.spec, state.current, legal)

            invalid_move = requested_action not in legal
            if invalid_move:
                state.invalid_move_steps += 1

            after = apply_action(before, executed_action)
            if not state.spec.is_free(after):
                # Defensive guard if a malformed map/feedback slips through.
                after = before

            row = step_row(
                state,
                prompt=prompt,
                raw_response=raw_response,
                repair_response=repair_outputs.get(index),
                parsed=parsed,
                parse_error=parse_error,
                parse_error_message=parse_errors[index],
                repaired=repaired,
                requested_action=requested_action,
                executed_action=executed_action,
                invalid_move=invalid_move,
                before=before,
                after=after,
                feedback=feedback,
                legal=legal,
            )
            row["done"] = after == state.spec.goal or state.step_id + 1 >= args.max_steps
            append_jsonl(steps_path, row)

            state.history.append(
                {
                    "step_id": state.step_id,
                    "position": list(before),
                    "feedback": feedback,
                    "requested_action": requested_action,
                    "executed_action": executed_action,
                    "next_position": list(after),
                }
            )
            state.model_belief = belief_for_next
            state.current = after
            state.trajectory.append(after)
            state.step_id += 1

            if after == state.spec.goal:
                state.done = True
                state.success = True
                state.termination_reason = "goal_reached"
            elif state.step_id >= args.max_steps:
                state.done = True
                state.success = False
                state.termination_reason = "max_steps"

            if state.done:
                append_jsonl(episodes_path, finalize_episode(state))

        done_count = sum(state.done for state in states)
        print(f"[coord-v4] rollout_step={global_step + 1}/{args.max_steps} completed={done_count}/{len(states)}")

    for state in states:
        if not state.done:
            state.done = True
            state.success = state.current == state.spec.goal
            state.termination_reason = "generator_stopped"
            append_jsonl(episodes_path, finalize_episode(state))

    summary = summarize(run)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "trajectory_run",
        "condition": "coordinate_belief_v4",
        "created_at_unix": time.time(),
        "model": args.model,
        "backend": args.backend,
        "maps_sha256": hashlib.sha256(maps_path.read_bytes()).hexdigest(),
        "num_requested_episodes": len(specs),
        "summary": summary,
    }
    atomic_write_json(run / "manifest.json", manifest)
    print(f"[coord-v4] saved={run}")
    print(json_dumps(summary, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate coordinate-belief trajectories with Qwen/vLLM."
    )
    parser.add_argument("--maps", required=True, help="Input maps JSONL.")
    parser.add_argument("--run", required=True, help="New output run directory.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--backend", choices=("vllm", "mock"), default="vllm")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--download-dir")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--repair-invalid-json",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_episodes is not None and args.num_episodes <= 0:
        raise SystemExit("--num-episodes must be positive")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    run_generation(args)


if __name__ == "__main__":
    main()

