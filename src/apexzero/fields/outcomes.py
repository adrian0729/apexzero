"""Corner outcomes on a trajectory (design Section 3), for the optimum's plan and the network's
closed loop alike.

Each outcome is computed in two parts:

1. A **plan** on the base trajectory (NumPy, float64): the window, the validity rule, and the
   *pick*, i.e. where the outcome is read (the argmin state, an interpolated window end, or
   the brake-onset crossing), plus the near-tie flags. Step 1's definitions are reused from
   the frozen ``interp/measures.py``: ``_View`` (measures.py:829, the measured part of a run:
   cut at W, at the terminal state and at the first non-increasing s), the window segment
   with interpolated ends (``_View.seg``, measures.py:902; copied here so that each element
   carries its pick) and ``brake_onset`` (measures.py:1444).
2. A **JAX function** of the states x [K, 8] (and, for c_min, the TrackArrays) that reads the
   outcome at the fixed pick. With the pick fixed from the base run this is the Danskin
   derivative of a min (argmin held), and of the onset with its run held (design Section 3).
   Picks are NamedTuples of scalars (or arrays), so the functions jit and vmap over keys.

Outcomes, window [a, b] = [s_c0 - 50, s_c1 + 50] (``V_MIN_MARGIN``, constants.py:205) with
interpolated ends; a start inside the window clips a to s_0, as ``_View.seg`` does:

- **c_min**: min over the window of the inside clearance c(s) = w_in(s) - 1 - sigma n(s),
  sigma the turn direction (+1 left). Conventions: n > 0 is left and kappa > 0 turns left
  (track.py:1-6); the env terminates when n > w_left - 1 or n < -(w_right - 1)
  (racing.py:160, edge_margin 1 m), so w_in is w_left on a left corner and w_right on a
  right corner, and c < 0 is exactly the inside half of the termination set. Clearance is
  computed at the states and interpolated linearly in s at the window ends (the treatment
  step 1 gives every window measure). Widths are the trajectory's own track arrays
  (``sim.widths_at``, the env's spline), so c_min is differentiable with respect to the width
  samples through ``strip_arrays`` / ``perturb.perturbed_arrays``.
- **v_apex**: step 1's v_min, min vx over the window (measures.py:1104-1109).
- **s_b**: step 1's brake onset, the interpolated 0.3 crossing of beta_f =
  ``front_brake_demand`` (measures.py:128) at the start of the last run of >= 3 states with
  beta_f > 0.3 that begins before s_apex (measures.py:1444-1473). The run is picked by the
  frozen ``brake_onset`` on the base run; the JAX version interpolates the crossing between
  states k_b - 1 and k_b.

**Validity** (design Section 3): an outcome is defined only when the run is alive (a state
before the terminal one, ``_View.kt``) at some state with s >= s_c1 + 50 m; s_b also needs an
onset. **Near-ties**: an outcome is flagged when the windowed series (ends included) has two
strict local minima within 0.05 m (c_min) or 0.5 m/s (v_apex) of each other; plateaus count
once and the window ends count as one-sided minima.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from apexzero.env import racing
from apexzero.interp import measures as M
from apexzero.interp.constants import BRAKE_ON, DT, V_MIN_MARGIN
from apexzero.interp.rollout import default_ctx
from apexzero.interp.stimuli import Landmarks
from apexzero.tracks.spline import spline_np
from apexzero.vehicle.ops import JAX
from apexzero.vehicle.params import F1_2025, VehicleParams
from apexzero.vehicle.sim import TrackArrays, widths_at

WINDOW_MARGIN = V_MIN_MARGIN                 # m: window [s_c0 - 50, s_c1 + 50]
EDGE_MARGIN = racing.EnvParams().edge_margin  # m: the "- 1" of the clearance (termination)
C_MIN_TIE = 0.05                             # m, near-tie band of c_min
V_APEX_TIE = 0.5                             # m/s, near-tie band of v_apex
OUTCOMES = ("c_min", "v_apex", "s_b")

# ------------------------------------------------------------------------- picks and plans


class Pick(NamedTuple):
    """Where a windowed outcome is read: state i0 (``interp`` False), or y interpolated
    linearly in s at the fixed position ``s_star`` between states i0 and i1 = i0 + 1 (a window
    end)."""

    i0: Any
    i1: Any
    s_star: Any
    interp: Any


class OnsetPick(NamedTuple):
    """The brake onset's run start k (state index): s_b is the ``thr`` crossing of beta_f
    between states k - 1 and k (s_0 when k = 0)."""

    k: Any
    thr: Any = BRAKE_ON


@dataclass(frozen=True)
class Plan:
    """One outcome on a base trajectory."""

    outcome: str              # 'c_min' | 'v_apex' | 's_b'
    valid: bool
    reason: str               # why not valid ('' when valid)
    value: float              # the base value (NaN when not valid)
    pick: Pick | OnsetPick | None
    s_at: float = np.nan      # where it is read (m)
    near_tie: bool = False
    minima: tuple[tuple[float, float], ...] = ()   # (s, y) of the strict local minima


def window(lm: Landmarks) -> tuple[float, float]:
    """[s_c0 - 50, s_c1 + 50] (m)."""
    return lm.s_c0 - WINDOW_MARGIN, lm.s_c1 + WINDOW_MARGIN


def drive_params(ctx: np.ndarray | None = None, hidden_mu: float = 1.0) -> VehicleParams:
    """The car both sides drive (``measures.car_params`` in float64): ``default_ctx()``'s
    float32 values (rollout.py:179; the keys' ``key_ctx`` is the same), hidden_mu = 1."""
    c = np.asarray(default_ctx() if ctx is None else ctx, np.float32).astype(np.float64)
    return M.car_params(F1_2025, c, hidden_mu)


def trajectory(x: np.ndarray, valid: np.ndarray | None = None, term_step: int = -1,
               source: str = "fields") -> M.Trajectory:
    """A ``measures.Trajectory`` of states [K, 8] at 20 Hz (``valid`` all True by default;
    ``term_step`` as the harness records it, rollout.py:29-31)."""
    x = np.asarray(x, np.float64)
    K = len(x)
    v = np.ones(K, bool) if valid is None else np.asarray(valid, bool)
    return M.Trajectory(source=source, t=np.arange(K) * DT, x=x, valid=v, term_step=int(term_step),
                        dt=DT)


def view(traj: M.Trajectory, W: int | None = None, p: VehicleParams | None = None) -> M._View:
    """Step 1's measured part of a run (measures.py:829)."""
    return M._View(traj, W, drive_params() if p is None else p)


def alive_through(v: M._View, s_end: float) -> bool:
    """A state before the terminal one (all states of a run that did not crash), on the
    increasing prefix, has s >= s_end."""
    n = min(v.km, v.kt) if v.crashed else v.km
    return bool(n > 0 and np.any(v.s[:n] >= s_end))


def validity(traj: M.Trajectory, lm: Landmarks, W: int | None = None,
             p: VehicleParams | None = None) -> tuple[bool, str]:
    """(valid, reason): alive at the window's end s_c1 + 50 m (design Section 3)."""
    v = view(traj, W, p)
    if alive_through(v, window(lm)[1]):
        return True, ""
    return False, "crashed before the window end" if v.crashed else "window end not reached"


def _end_pick(s: np.ndarray, s_star: float) -> Pick:
    """The pick np.interp(s_star, s, y) evaluates: the bracket [s_j, s_{j+1}) holding s_star."""
    if len(s) < 2 or s_star <= s[0]:
        return Pick(0, 0, 0.0, False)
    j = int(np.clip(np.searchsorted(s, s_star, side="right") - 1, 0, len(s) - 2))
    return Pick(j, j + 1, float(s_star), True)


def segment(v: M._View, a: float, b: float, y: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, list[Pick]] | None:
    """``_View.seg`` (measures.py:902-916) with the pick of every element: (s, y) over [a, b]
    (the states inside plus the interpolated ends), or None unless the run reached b; a start
    after a clips the window to the start (whose end is then state 0 itself)."""
    if not v.km or not v.reached(b) or np.isnan(a):
        return None
    s = v.s[: v.km]
    yy = np.asarray(y[: v.km], float)
    a = max(a, float(s[0]))
    if a > b:
        return None
    inside = (s > a) & (s < b)
    ends = np.interp([a, b], s, yy)
    idx = np.flatnonzero(inside)
    picks = [_end_pick(s, a)] + [Pick(int(k), int(k), 0.0, False) for k in idx] + [_end_pick(s, b)]
    return np.r_[a, s[inside], b], np.r_[ends[0], yy[inside], ends[1]], picks


def strict_minima(y: np.ndarray) -> list[int]:
    """Indices of the strict local minima of y, a plateau counted once (its first index) and
    the ends as one-sided minima."""
    y = np.asarray(y, float)
    if len(y) == 0:
        return []
    keep = np.r_[True, y[1:] != y[:-1]]          # first index of each plateau
    first = np.flatnonzero(keep)
    z = y[first]
    if len(z) == 1:
        return [int(first[0])]
    left = np.r_[np.inf, z[:-1]]
    right = np.r_[z[1:], np.inf]
    return [int(first[i]) for i in np.flatnonzero((z < left) & (z < right))]


def _tie(s_seg: np.ndarray, y_seg: np.ndarray, tol: float) -> tuple[bool, tuple]:
    mins = strict_minima(y_seg)
    lo = float(np.nanmin(y_seg))
    close = [i for i in mins if y_seg[i] <= lo + tol]
    return len(close) >= 2, tuple((float(s_seg[i]), float(y_seg[i])) for i in mins)


def _min_plan(name: str, traj: M.Trajectory, lm: Landmarks, y: np.ndarray, tol: float,
              W: int | None, p: VehicleParams | None) -> Plan:
    v = view(traj, W, p)
    a, b = window(lm)
    if not alive_through(v, b):
        why = "crashed before the window end" if v.crashed else "window end not reached"
        return Plan(name, False, why, np.nan, None)
    sg = segment(v, a, b, y)
    if sg is None or not np.isfinite(sg[1]).any():
        return Plan(name, False, "no finite values in the window", np.nan, None)
    s_seg, y_seg, picks = sg
    j = int(np.nanargmin(y_seg))
    tie, mins = _tie(s_seg, y_seg, tol)
    return Plan(name, True, "", float(y_seg[j]), picks[j], float(s_seg[j]), tie, mins)


def clearance_np(x: np.ndarray, tr: TrackArrays, sigma: int) -> np.ndarray:
    """c_k = w_in(s_k) - 1 - sigma n_k at states x [K, 8] (NumPy float64, ``spline_np`` of the
    arrays' width coefficients: identical to ``eval_jax``, spline.py:96)."""
    x = np.asarray(x, float)
    ds, n, closed = float(np.asarray(tr.ds)), int(np.asarray(tr.n_points)), bool(np.asarray(tr.closed))
    coef = np.asarray(tr.w_left_c if int(sigma) > 0 else tr.w_right_c, float)
    return spline_np(coef, x[:, 0], ds, n, closed) - EDGE_MARGIN - _sigma(sigma) * x[:, 1]


def _sigma(sigma: int) -> int:
    if int(sigma) not in (1, -1):
        raise ValueError(f"turn direction {sigma} is not +1 (left) or -1 (right)")
    return int(sigma)


def c_min_plan(traj: M.Trajectory, lm: Landmarks, sigma: int, tr: TrackArrays,
               W: int | None = None, p: VehicleParams | None = None) -> Plan:
    """c_min on the base run: the argmin's pick and the near-tie flag (0.05 m)."""
    return _min_plan("c_min", traj, lm, clearance_np(traj.x, tr, sigma), C_MIN_TIE, W, p)


def v_apex_plan(traj: M.Trajectory, lm: Landmarks, W: int | None = None,
                p: VehicleParams | None = None) -> Plan:
    """v_apex (step 1's v_min) on the base run: the argmin's pick and the near-tie flag
    (0.5 m/s)."""
    return _min_plan("v_apex", traj, lm, np.asarray(traj.x, float)[:, 3], V_APEX_TIE, W, p)


def s_b_plan(traj: M.Trajectory, lm: Landmarks, W: int | None = None,
             p: VehicleParams | None = None) -> Plan:
    """s_b on the base run: step 1's ``brake_onset`` (on beta_f of the states, as
    ``response_metrics`` calls it, measures.py:1067) picks the run; its start k_b is fixed."""
    p = drive_params() if p is None else p
    v = view(traj, W, p)
    if not alive_through(v, window(lm)[1]):
        why = "crashed before the window end" if v.crashed else "window end not reached"
        return Plan("s_b", False, why, np.nan, None)
    beta = np.asarray(M.front_brake_demand(v.x, p, M.NP)[0], float)
    bo = M.brake_onset(v.s, v.t, beta, np.ones(v.n, bool), lm.s_apex, v.vx)
    if not np.isfinite(bo.s_b):
        return Plan("s_b", False, "no sustained brake before s_apex", np.nan, None)
    return Plan("s_b", True, "", float(bo.s_b), OnsetPick(int(bo.k_b), BRAKE_ON), float(bo.s_b))


def plans(traj: M.Trajectory, lm: Landmarks, sigma: int, tr: TrackArrays, W: int | None = None,
          p: VehicleParams | None = None) -> dict[str, Plan]:
    """The three outcome plans of one base run."""
    return {"c_min": c_min_plan(traj, lm, sigma, tr, W, p),
            "v_apex": v_apex_plan(traj, lm, W, p), "s_b": s_b_plan(traj, lm, W, p)}


# ------------------------------------------------------------------------- JAX readers


def read(y, s, pick: Pick):
    """y at the pick: y[i0], or y interpolated in s at s_star between i0 and i1 (np.interp's
    arithmetic). The unused branch is made finite before the ``where`` (safe double-where), so
    gradients stay finite."""
    y0, y1 = y[pick.i0], y[pick.i1]
    s0, s1 = s[pick.i0], s[pick.i1]
    interp = jnp.asarray(pick.interp, bool)
    den = jnp.where(interp, s1 - s0, 1.0)
    slope = jnp.where(interp, (y1 - y0) / den, 0.0)
    s_star = jnp.where(interp, pick.s_star, s0)
    return jnp.where(interp, slope * (s_star - s0) + y0, y0)


def clearance(x, tr: TrackArrays, sigma):
    """c_k = w_in(s_k) - 1 - sigma n_k [K] (JAX; widths from the track arrays' splines)."""
    wl, wr = widths_at(x[:, 0], tr)
    w_in = jnp.where(jnp.asarray(sigma) > 0, wl, wr)
    return w_in - EDGE_MARGIN - sigma * x[:, 1]


def c_min(x, tr: TrackArrays, sigma, pick: Pick):
    """c_min at the base run's argmin (Danskin)."""
    return read(clearance(x, tr, sigma), x[:, 0], pick)


def v_apex(x, pick: Pick):
    """v_apex (step 1's v_min) at the base run's argmin (Danskin)."""
    return read(x[:, 3], x[:, 0], pick)


def brake_demand(x, p: VehicleParams):
    """beta_f [K] = ``measures.front_brake_demand`` with the JAX ops."""
    return M.front_brake_demand(x, p, JAX)[0]


def s_b(x, pick: OnsetPick, p: VehicleParams):
    """The interpolated ``thr`` crossing of beta_f between states k - 1 and k (s_0 for k = 0),
    with k fixed from the base run; ``measures._frac`` / ``_lerp`` arithmetic
    (measures.py:1412-1421)."""
    beta = brake_demand(x, p)
    k = jnp.asarray(pick.k)
    km = jnp.maximum(k - 1, 0)
    b0, b1 = beta[km], beta[k]
    s0, s1 = x[km, 0], x[k, 0]
    ok = (k > 0) & (b1 != b0)
    den = jnp.where(ok, b1 - b0, 1.0)
    w = jnp.where(ok, jnp.clip((pick.thr - b0) / den, 0.0, 1.0), 1.0)
    return jnp.where(k > 0, s0 + w * (s1 - s0), x[0, 0])


def outcome(name: str, x, plan: Plan, tr: TrackArrays | None = None, sigma: int = 1,
            p: VehicleParams | None = None):
    """The named outcome at a valid plan's pick."""
    if not plan.valid:
        raise ValueError(f"{name} is not defined on this run: {plan.reason}")
    if name == "c_min":
        return c_min(x, tr, sigma, plan.pick)
    if name == "v_apex":
        return v_apex(x, plan.pick)
    if name == "s_b":
        return s_b(x, plan.pick, drive_params() if p is None else p)
    raise ValueError(f"unknown outcome {name!r}; expected one of {OUTCOMES}")
