"""BFS optimum vs JaxUED's own maze environment (exact match on executed optimal plans)."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxued.environments import Maze
from jaxued.environments.maze import make_level_generator

import apexzero.compat  # noqa: F401  (must precede jaxued)
from apexzero.maze.oracle import DIR_TO_VEC, min_actions_to_goal, optimal_return


def test_corridor_by_hand():
    walls = np.zeros((3, 5), bool)
    # Agent at (0, 1) facing right, goal at (4, 1): 3 forward moves, then forward into goal.
    assert min_actions_to_goal(walls, (0, 1), 0, (4, 1)) == 4
    # Facing left: two turns first.
    assert min_actions_to_goal(walls, (0, 1), 2, (4, 1)) == 6
    assert np.isclose(optimal_return(walls, (0, 1), 0, (4, 1)), 0.995**3 * (1 - 0.9 * 4 / 250))


def test_unreachable_goal_has_zero_optimum():
    walls = np.zeros((3, 5), bool)
    walls[:, 2] = True
    assert min_actions_to_goal(walls, (0, 1), 0, (4, 1)) is None
    assert optimal_return(walls, (0, 1), 0, (4, 1)) == 0.0


def _bfs_plan(walls, pos, d, goal):
    """Recover an optimal action sequence (for executing it in the real env)."""
    from collections import deque
    h, w = walls.shape
    start = (pos[0], pos[1], d)
    prev = {start: None}
    q = deque([start])
    while q:
        s = q.popleft()
        x, y, dd = s
        dx, dy = DIR_TO_VEC[dd]
        fx, fy = min(max(x + dx, 0), w - 1), min(max(y + dy, 0), h - 1)
        if (fx, fy) == tuple(goal):
            acts, cur = [2], s
            while prev[cur] is not None:
                cur, a = prev[cur]
                acts.append(a)
            return acts[::-1]
        for a, n in ((0, (x, y, (dd - 1) % 4)), (1, (x, y, (dd + 1) % 4)),
                     (2, (fx, fy, dd) if not walls[fy, fx] else None)):
            if n is not None and n not in prev:
                prev[n] = (s, a)
                q.append(n)
    return None


def test_bfs_optimum_is_achieved_exactly_in_the_jaxued_env():
    env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    params = env.default_params
    gen = make_level_generator(13, 13, 25)
    checked = 0
    for i in range(30):
        lvl = gen(jax.random.PRNGKey(i))
        walls = np.asarray(lvl.wall_map, bool)
        pos, d, goal = np.asarray(lvl.agent_pos), int(lvl.agent_dir), np.asarray(lvl.goal_pos)
        plan = _bfs_plan(walls, tuple(int(v) for v in pos), d, goal)
        if plan is None or len(plan) > params.max_steps_in_episode:
            continue
        _, st = env.reset_to_level(jax.random.PRNGKey(0), lvl, params)
        ret, disc, done = 0.0, 1.0, False
        for a in plan:
            out = env.step_env(jax.random.PRNGKey(1), st, jnp.asarray(a), params)
            st, r, done = out[1], float(out[2]), bool(out[3])
            ret += disc * r
            disc *= 0.995
        assert done and len(plan) == min_actions_to_goal(walls, pos, d, goal)
        assert np.isclose(ret, optimal_return(walls, pos, d, goal), atol=1e-6)
        checked += 1
    assert checked >= 15
