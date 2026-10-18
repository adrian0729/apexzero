"""Timed laps with a deterministic policy.

Protocol (plan section 4): rolling start at s = 0, deterministic mean action, run two
laps, time lap 2. Repeated from jittered starts (small lateral and speed offsets) to
measure completion and consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.env import racing
from apexzero.tracks.track import Track
from apexzero.vehicle.sim import track_arrays


@dataclass
class LapResult:
    completed: np.ndarray       # [R] bool, lap 2 finished without termination
    lap_time: np.ndarray        # [R] seconds (nan if not completed)
    s_reached: np.ndarray       # [R] distance covered
    trace: dict                 # first run: s, n, vx, actions over time

    @property
    def completion_rate(self) -> float:
        return float(self.completed.mean())

    @property
    def median_lap(self) -> float:
        lt = self.lap_time[self.completed]
        return float(np.median(lt)) if len(lt) else float("nan")


def timed_laps(policy_fn, track: Track, v_start: float, runs: int = 20, laps: int = 2,
               params: racing.EnvParams = racing.EnvParams(), ctx=None, seed: int = 0) -> LapResult:
    """``policy_fn(obs [B, D]) -> action [B, 2]`` (deterministic)."""
    tr = track_arrays(track)
    ctx = racing.default_context() if ctx is None else ctx
    L = float(track.length)
    max_steps = int(laps * 3.0 * L / max(v_start, 20.0) / params.dt) + 200
    eval_params = params._replace(window_steps=10**9)

    rng = np.random.default_rng(seed)
    n0 = np.where(np.arange(runs) == 0, 0.0, rng.uniform(-1.0, 1.0, runs))
    v0 = v_start * np.where(np.arange(runs) == 0, 1.0, rng.uniform(0.95, 1.05, runs))
    x0 = np.zeros((runs, 8), np.float32)
    x0[:, 1], x0[:, 3] = n0, v0

    def init_state(x):
        return racing.EnvState(x=x, prev_action=jnp.zeros(2), accel=jnp.zeros(2), ctx=ctx,
                               t=jnp.zeros((), jnp.int32), slow=jnp.zeros(()), s0=x[0])

    states = jax.vmap(init_state)(jnp.asarray(x0))
    p = racing.params_from_context(ctx)
    obs = jax.vmap(lambda st: racing.observe(st, tr, p))(states)

    @jax.jit
    def run(states, obs):
        def body(c, _):
            st, ob, alive = c
            a = policy_fn(ob)
            ob2, st2, _, term, _, _ = jax.vmap(
                lambda s, aa: racing.step_no_reset(s, aa, tr, eval_params))(st, a)
            alive2 = alive & ~term
            st2 = jax.tree.map(lambda new, old: jnp.where(
                alive.reshape((-1,) + (1,) * (new.ndim - 1)), new, old), st2, st)
            return (st2, ob2, alive2), (st2.x[:, 0], st2.x[:, 1], st2.x[:, 3], a, alive2)
        _, out = jax.lax.scan(body, (states, obs, jnp.ones(runs, bool)), None, max_steps)
        return out

    s, n, vx, acts, alive = (np.asarray(a) for a in run(states, obs))
    t = (np.arange(max_steps) + 1) * params.dt
    completed = np.zeros(runs, bool)
    lap_time = np.full(runs, np.nan)
    for r in range(runs):
        def cross(target, r=r):
            idx = np.nonzero(s[:, r] >= target)[0]
            if not len(idx):
                return None
            k = idx[0]
            s_prev = s[k - 1, r] if k else 0.0
            return t[k] - params.dt + (target - s_prev) / max(s[k, r] - s_prev, 1e-6) * params.dt
        t1, t2 = cross((laps - 1) * L), cross(laps * L)
        if t1 is not None and t2 is not None and alive[np.nonzero(s[:, r] >= laps * L)[0][0], r]:
            completed[r] = True
            lap_time[r] = t2 - t1
    first_end = int(np.argmax(s[:, 0] >= laps * L)) or max_steps
    trace = {"t": t[:first_end], "s": s[:first_end, 0], "n": n[:first_end, 0],
             "vx": vx[:first_end, 0], "action": acts[:first_end, 0]}
    return LapResult(completed, lap_time, s.max(axis=0), trace)
