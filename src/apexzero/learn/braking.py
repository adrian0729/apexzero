"""The 1-D braking problem, solved by Pontryagin and by value iteration.

A point mass approaches a corner at s_c with speed limit v_c along a straight, on the same
speed-dependent g-g envelope as the QSS oracle. Arriving faster than v_c is a crash.

- **Pontryagin / minimum time.** The Hamiltonian is linear in the acceleration, so the
  optimum is bang-bang: full throttle, then full braking from the switch point where the
  acceleration curve meets the braking curve that ends at (s_c, v_c).
- **Discounted RL (Bellman).** The agent's objective: reward = progress / 10 per 50 ms
  step, a crash costs -(5 + 5 (v/50)^2) (the env's penalty) and ends the episode, and
  passing the corner safely continues with full-throttle progress afterwards. Value
  iteration on an (s, v) grid gives the optimal discounted policy for each gamma.

Short horizons make the agent brake late: a crash 2 s away is discounted by gamma^40,
so for small gamma the progress gained by braking later outweighs it. This is the
gamma lesson in the plan, measured exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from apexzero.oracle.qss import GG
from apexzero.vehicle.params import F1_2025


@dataclass(frozen=True)
class BrakingProblem:
    s_c: float = 400.0       # corner position (m)
    v_c: float = 25.0        # corner speed limit (m/s), 90 km/h
    v0: float = 80.0         # entry speed at s = 0
    dt: float = 0.05
    n_actions: int = 21


def accel_bounds(gg: GG, v):
    return -gg.a_brake_max(v), np.maximum(gg.a_acc_max(v), 0.0)


def pontryagin_switch(prob: BrakingProblem, gg: GG, ds: float = 0.01) -> float:
    """Switch point of the bang-bang minimum-time solution."""
    s = np.arange(0.0, prob.s_c + ds, ds)
    # Full throttle from (0, v0), forward in s.
    v_acc = np.empty_like(s)
    v_acc[0] = prob.v0
    for i in range(1, len(s)):
        v_acc[i] = np.sqrt(v_acc[i - 1] ** 2 + 2 * ds * max(gg.a_acc_max(v_acc[i - 1]), 0.0))
    # Full braking ending at (s_c, v_c), backward in s.
    v_brk = np.empty_like(s)
    v_brk[-1] = prob.v_c
    for i in range(len(s) - 2, -1, -1):
        v_brk[i] = np.sqrt(v_brk[i + 1] ** 2 + 2 * ds * gg.a_brake_max(v_brk[i + 1]))
    k = np.argmax(v_acc >= v_brk)
    return float(s[k])


def exit_value(v, gamma, gg: GG, dt: float, horizon_s: float = 30.0):
    """Discounted progress after passing the corner at speed v under full throttle."""
    v = np.atleast_1d(np.asarray(v, float)).copy()
    total = np.zeros_like(v)
    disc = 1.0
    for _ in range(int(horizon_s / dt)):
        a = np.maximum(gg.a_acc_max(v), 0.0)
        v_new = v + a * dt
        total += disc * 0.5 * (v + v_new) * dt / 10.0
        v, disc = v_new, disc * gamma
    return total


def _interp_matrix(s_grid, v_grid, s_q, v_q):
    """Sparse bilinear-interpolation operator from grid values to query points."""
    from scipy import sparse

    ns, nv = len(s_grid), len(v_grid)
    fs = np.clip((s_q - s_grid[0]) / (s_grid[1] - s_grid[0]), 0, ns - 1 - 1e-9)
    fv = np.clip((v_q - v_grid[0]) / (v_grid[1] - v_grid[0]), 0, nv - 1 - 1e-9)
    i, j = np.floor(fs).astype(int), np.floor(fv).astype(int)
    ws, wv = fs - i, fv - j
    rows = np.repeat(np.arange(len(s_q)), 4)
    cols = np.stack([i * nv + j, (i + 1) * nv + j, i * nv + j + 1, (i + 1) * nv + j + 1], 1).ravel()
    vals = np.stack([(1 - ws) * (1 - wv), ws * (1 - wv), (1 - ws) * wv, ws * wv], 1).ravel()
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(s_q), ns * nv))


def value_iteration(prob: BrakingProblem, gg: GG, gamma: float, ns: int = 401, nv: int = 181,
                    iters: int = 5000, tol: float = 1e-7):
    """Discounted value iteration. Transitions are deterministic and fixed, so the
    interpolation of the next-state value is a fixed sparse operator built once."""
    s_grid = np.linspace(0.0, prob.s_c, ns)
    v_grid = np.linspace(0.0, 90.0, nv)
    S, V = np.meshgrid(s_grid, v_grid, indexing="ij")
    lo, hi = accel_bounds(gg, V)
    frac = np.linspace(0.0, 1.0, prob.n_actions)
    A = lo[..., None] + frac[None, None, :] * (hi - lo)[..., None]       # [ns, nv, na]
    v_next = np.clip(V[..., None] + A * prob.dt, 0.0, 90.0)
    s_next = S[..., None] + 0.5 * (V[..., None] + v_next) * prob.dt
    reward = (s_next - S[..., None]) / 10.0
    crossed = s_next >= prob.s_c
    crash = crossed & (v_next > prob.v_c)
    term = np.where(crash, -(5.0 + 5.0 * (v_next / 50.0) ** 2),
                    exit_value(v_next.ravel(), gamma, gg, prob.dt).reshape(v_next.shape))
    W = _interp_matrix(s_grid, v_grid, np.minimum(s_next, prob.s_c).ravel(), v_next.ravel())
    fixed = (reward + gamma * np.where(crossed, term, 0.0)).ravel()
    live = (~crossed).ravel().astype(float)
    val = np.zeros(ns * nv)
    for _ in range(iters):
        q = fixed + gamma * live * (W @ val)
        new = q.reshape(ns, nv, -1).max(-1).ravel()
        if np.max(np.abs(new - val)) < tol:
            val = new
            break
        val = new
    q = q.reshape(ns, nv, -1)
    policy = A[np.arange(ns)[:, None], np.arange(nv)[None, :], q.argmax(-1)]
    return s_grid, v_grid, val.reshape(ns, nv), policy


def rollout(prob: BrakingProblem, s_grid, v_grid, policy, max_steps: int = 2000):
    """Greedy rollout of a grid policy from (0, v0): returns trajectory, braking point, crash."""
    pol = RegularGridInterpolator((s_grid, v_grid), policy, bounds_error=False, fill_value=None)
    s, v = 0.0, prob.v0
    traj, brake_at = [(s, v)], None
    for _ in range(max_steps):
        a = float(pol([[min(s, prob.s_c), v]])[0])
        if a < 0 and brake_at is None:
            brake_at = s
        v_new = min(max(v + a * prob.dt, 0.0), 90.0)
        s += 0.5 * (v + v_new) * prob.dt
        v = v_new
        traj.append((s, v))
        if s >= prob.s_c:
            return np.array(traj), brake_at, v > prob.v_c
    return np.array(traj), brake_at, False


def gamma_study(gammas=(0.9, 0.95, 0.98, 0.99, 0.995, 0.999), prob: BrakingProblem = BrakingProblem()):
    gg = GG.from_params(F1_2025)
    out = {"switch_pontryagin": pontryagin_switch(prob, gg), "runs": {}}
    for g in gammas:
        sg, vg, _, pol = value_iteration(prob, gg, g)
        traj, brake_at, crashed = rollout(prob, sg, vg, pol)
        out["runs"][g] = {"brake_at": brake_at, "crashed": bool(crashed),
                          "v_at_corner": float(traj[-1, 1]), "traj": traj}
    return out
