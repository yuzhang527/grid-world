from __future__ import annotations
from collections import deque
from grid_world.env.grid import ACTIONS, DELTAS, Coord, GridSpec

def neighbors(spec: GridSpec, coord: Coord) -> list[Coord]:
    x, y = coord
    out = []
    for action in ACTIONS:
        dx, dy = DELTAS[action]
        nxt = (x + dx, y + dy)
        if 0 <= nxt[0] < spec.size and 0 <= nxt[1] < spec.size and nxt not in spec.obstacles:
            out.append(nxt)
    return out

def distance_map(spec: GridSpec, goal: Coord | None = None) -> dict[Coord, int]:
    target = spec.goal if goal is None else goal
    dist = {target: 0}
    queue = deque([target])
    while queue:
        node = queue.popleft()
        for nxt in neighbors(spec, node):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist

def shortest_path_length(spec: GridSpec) -> int | None:
    return distance_map(spec).get(spec.start)

def shortest_actions(spec: GridSpec, position: Coord) -> list[str]:
    dist = distance_map(spec)
    candidates = []
    x, y = position
    for action in ACTIONS:
        dx, dy = DELTAS[action]
        nxt = (x + dx, y + dy)
        if nxt in dist and nxt not in spec.obstacles:
            candidates.append((dist[nxt], action))
    if not candidates:
        return []
    best = min(d for d, _ in candidates)
    return [a for d, a in candidates if d == best]

def count_shortest_paths(spec: GridSpec, cap: int = 2) -> int:
    dist = {spec.start: 0}
    ways = {spec.start: 1}
    queue = deque([spec.start])
    while queue:
        node = queue.popleft()
        for nxt in neighbors(spec, node):
            nd = dist[node] + 1
            if nxt not in dist:
                dist[nxt] = nd
                ways[nxt] = ways[node]
                queue.append(nxt)
            elif dist[nxt] == nd:
                ways[nxt] = min(cap, ways[nxt] + ways[node])
    return ways.get(spec.goal, 0)
