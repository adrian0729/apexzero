"""Exact optimal return for JaxUED mazes (the E5 external check's ground truth).

JaxUED maze semantics (jaxued/environments/maze/env.py at the pinned commit):
- state (x, y, dir), dir in {0: right, 1: down, 2: left, 3: up};
- actions: left (dir - 1), right (dir + 1), forward (move one cell unless the cell is a
  wall or the goal, or off the grid), and four no-ops;
- moving forward *into* the goal ends the episode with reward 1 - 0.9 * k / max_steps,
  where k is the number of actions taken including that one (penalize_time = True);
- the episode also ends after max_steps actions.

The optimum is a breadth-first search over (x, y, dir) for the fewest actions k that end
with a forward move into the goal, so J* = gamma^(k-1) * (1 - 0.9 k / max_steps), or 0 if
the goal is unreachable within max_steps.
"""

from __future__ import annotations

from collections import deque

import numpy as np

DIR_TO_VEC = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])


def min_actions_to_goal(wall_map: np.ndarray, agent_pos, agent_dir: int, goal_pos,
                        max_steps: int = 250) -> int | None:
    """Fewest actions to step into the goal (None if impossible within max_steps)."""
    h, w = wall_map.shape
    gx, gy = int(goal_pos[0]), int(goal_pos[1])
    start = (int(agent_pos[0]), int(agent_pos[1]), int(agent_dir) % 4)
    seen = {start: 0}
    q = deque([start])
    while q:
        x, y, d = q.popleft()
        k = seen[(x, y, d)]
        if k >= max_steps:
            continue
        dx, dy = DIR_TO_VEC[d]
        fx, fy = min(max(x + dx, 0), w - 1), min(max(y + dy, 0), h - 1)
        if (fx, fy) == (gx, gy):
            return k + 1
        nxt = [(x, y, (d - 1) % 4), (x, y, (d + 1) % 4)]
        if not wall_map[fy, fx]:
            nxt.append((fx, fy, d))
        for s in nxt:
            if s not in seen:
                seen[s] = k + 1
                q.append(s)
    return None


def optimal_return(wall_map, agent_pos, agent_dir, goal_pos, gamma: float = 0.995,
                   max_steps: int = 250) -> float:
    k = min_actions_to_goal(np.asarray(wall_map, bool), agent_pos, agent_dir, goal_pos, max_steps)
    if k is None:
        return 0.0
    return float(gamma ** (k - 1) * (1.0 - 0.9 * k / max_steps))


def optimal_plan(wall_map: np.ndarray, agent_pos, agent_dir: int, goal_pos,
                 max_steps: int = 250) -> list[int] | None:
    """One shortest action sequence (0 = left, 1 = right, 2 = forward) that ends by stepping
    into the goal, or None. Same search as ``min_actions_to_goal`` with parent pointers, so
    ``len(plan) == min_actions_to_goal(...)``; executed open-loop it attains J* exactly."""
    wall_map = np.asarray(wall_map, bool)
    h, w = wall_map.shape
    gx, gy = int(goal_pos[0]), int(goal_pos[1])
    start = (int(agent_pos[0]), int(agent_pos[1]), int(agent_dir) % 4)
    parent: dict = {start: None}
    depth = {start: 0}
    q = deque([start])
    while q:
        s = q.popleft()
        x, y, d = s
        if depth[s] >= max_steps:
            continue
        dx, dy = DIR_TO_VEC[d]
        fx, fy = min(max(x + dx, 0), w - 1), min(max(y + dy, 0), h - 1)
        if (fx, fy) == (gx, gy):
            plan, cur = [2], s
            while parent[cur] is not None:
                cur, a = parent[cur]
                plan.append(a)
            return plan[::-1]
        nxt = [(0, (x, y, (d - 1) % 4)), (1, (x, y, (d + 1) % 4))]
        if not wall_map[fy, fx]:
            nxt.append((2, (fx, fy, d)))
        for a, n in nxt:
            if n not in parent:
                parent[n] = (s, a)
                depth[n] = depth[s] + 1
                q.append(n)
    return None
