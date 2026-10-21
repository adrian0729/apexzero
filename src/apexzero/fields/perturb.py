"""Sample-space perturbations of a stimulus and the track arrays they act on (design Section 4).

**What is perturbed.** The stimulus bank's knot samples kappa_j, w_left_j and w_right_j of one
row (``StimulusBank``, stimuli.py:435; j = 0..N-1 at s_j = j ds, ds = DS = 2 m,
constants.py:27). Each feature separately; the generator's inside cap
(``apply_inside_cap``, stimuli.py:616) is not re-applied.

**Network side** (a float64 process). The agents drove float32 ``TrackArrays`` built by
``strip_arrays`` (generator.py:258) from the float32-cast samples (``StimulusBank.arrays``,
stimuli.py:459, gathered by ``rollout.bank_arrays``, rollout.py:656). The base arrays here are
their exact float64 upcasts (``rollout.float32_bank_arrays`` and ``rollout._upcast``,
rollout.py:798-826: T64's recipe, ruling R20). A perturbation adds the float64 tangent of
``strip_arrays`` at the upcast samples, J(samples) delta (``strip_tangent``), so the spline
coefficients (kappa_c, w_left_c, w_right_c: linear in the samples) and the pose (theta: linear
in kappa; x, y through cos and sin of theta) move together. The tangent is linear in delta, so
``jax.vjp`` through ``perturbed_arrays`` gives per-sample impulse responses. ``mode='secant'``
adds strip_arrays(samples + delta) - strip_arrays(samples) instead (finite amplitudes); the two
agree to first order. Both need x64 on (``_require_x64``), never in a rollout process.

**Optimum side.** The TO-OCP reads a NumPy Track (``StimulusBank.track``, stimuli.py:476:
``to_track`` of the float32 arrays, whose node values reproduce the float32 coefficients) and
turns its node values into coefficient windows with ``packed_coeffs`` (window.py:187-201,
spline.py:70). packed_coeffs is linear, so adding delta to the Track's node values
(``perturbed_track``) changes the coefficients by exactly the natural-spline coefficients of
delta, the same change ``strip_tangent`` gives the network (``coeffs_open_jax`` and
``coeffs_open_np`` solve the same tridiagonal system, spline.py:33-67).

**Profiles.**
- ``trapezoid`` h(s): 0 before s_c0 - 40, linear to 1 at s_c0 - 20, 1 through s_c1 + 20, linear
  back to 0 at s_c1 + 40.
- Inside-edge shift ``edge_shift(..., 'inside')``: Delta w_in(s) = Delta h(s), where w_in is the
  left width on a left corner (sigma = +1) and the right width on a right corner (sigma = -1).
  Conventions: kappa > 0 turns left and n > 0 is left of the centreline (track.py:1-6); the env
  terminates at n > w_left - 1 or n < -(w_right - 1) (racing.py:160). Positive Delta widens the
  inside, i.e. moves the inside edge further into the turn.
- The gates' directions (``gate_directions``, one definition for Gates 1, 2 and 5): bumps near
  the car (s_1 + d) and at the corner's landmarks, scored as separate field vectors
  (``direction_group``), plus both trapezoids.
- Gaussian bumps G_d(s) = exp(-((s - s_1 - d) / sigma_b)^2 / 2), unit peak at s_1 + d
  (``bump_profiles``). A bump field is the per-sample impulse responses contracted with the
  bump profiles (``combine``), so no bump is truncated.

**Turn recoding** (``recode``): curvature is signed toward the turn (kappa_turn = sigma kappa, so
a field with respect to kappa_turn is sigma times the field with respect to kappa), steer is
multiplied by sigma (``turn_steer``), widths become inside / outside.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.interp.constants import DS
from apexzero.interp.rollout import _upcast, float32_bank_arrays
from apexzero.interp.stimuli import Landmarks, StimulusBank
from apexzero.tracks.generator import strip_arrays
from apexzero.tracks.track import Track
from apexzero.vehicle.sim import TrackArrays

TRAPEZOID_OUTER = 40.0     # m, h = 0 beyond s_c0 - 40 and s_c1 + 40 (design Section 4)
TRAPEZOID_INNER = 20.0     # m, h = 1 within [s_c0 - 20, s_c1 + 20]
D_START = -20.0            # m, first bump offset d from s_1
D_STEP = 2.0               # m, the d grid's spacing
SIGMA_B = (8.0, 4.0, 16.0)  # m, bump widths: 8 m primary, 4 and 16 m secondary
FEATURES = ("kappa", "w_left", "w_right")
POSE_FIELDS = ("x", "y", "theta")
FLOAT_FIELDS = ("kappa_c", "w_left_c", "w_right_c", *POSE_FIELDS)


class Samples(NamedTuple):
    """Knot samples (or their perturbation) of one strip, one [N] array per feature. A JAX
    pytree, so it can be the argument of ``jax.vjp``."""

    kappa: Any
    w_left: Any
    w_right: Any


def zeros(n: int) -> Samples:
    return Samples(np.zeros(n), np.zeros(n), np.zeros(n))


def sample_s(n: int, ds: float = DS) -> np.ndarray:
    """Arc lengths s_j = j ds of the knot samples (m)."""
    return np.arange(n) * float(ds)


# ------------------------------------------------------------------------- profiles


def trapezoid(s, s_c0: float, s_c1: float, outer: float = TRAPEZOID_OUTER,
              inner: float = TRAPEZOID_INNER) -> np.ndarray:
    """h(s): 0 before s_c0 - outer, linear to 1 at s_c0 - inner, 1 through s_c1 + inner, linear
    back to 0 at s_c1 + outer (design Section 4)."""
    s = np.asarray(s, float)
    ramp = outer - inner
    up = np.clip((s - (s_c0 - outer)) / ramp, 0.0, 1.0)
    down = np.clip(((s_c1 + outer) - s) / ramp, 0.0, 1.0)
    return np.minimum(up, down)


def side_feature(sigma: int, side: str) -> str:
    """'w_left' or 'w_right': the inside width is the left one on a left corner (sigma = +1)."""
    if int(sigma) not in (1, -1):
        raise ValueError(f"turn direction {sigma} is not +1 (left) or -1 (right)")
    if side not in ("inside", "outside"):
        raise ValueError(f"side {side!r} is not 'inside' or 'outside'")
    left = (int(sigma) > 0) == (side == "inside")
    return "w_left" if left else "w_right"


def edge_shift(n: int, landmarks: Landmarks, sigma: int, side: str = "inside",
               delta: float = 1.0, ds: float = DS) -> Samples:
    """Delta h(s_j) on the inside (or outside) width samples, zero elsewhere: the primary
    perturbation (design Section 4). Positive ``delta`` widens that side."""
    h = float(delta) * trapezoid(sample_s(n, ds), landmarks.s_c0, landmarks.s_c1)
    out = zeros(n)._asdict()
    out[side_feature(sigma, side)] = h
    return Samples(**out)


def gaussian_bump(s, centre, sigma_b: float) -> np.ndarray:
    """exp(-((s - centre) / sigma_b)^2 / 2): unit peak at ``centre`` (broadcasts)."""
    z = (np.asarray(s, float) - np.asarray(centre, float)) / float(sigma_b)
    return np.exp(-0.5 * z * z)


def d_grid(d_end: float, d_start: float = D_START, step: float = D_STEP) -> np.ndarray:
    """Bump offsets d = d_start, d_start + step, ... <= d_end (m, relative to s_1)."""
    n = int(np.floor((float(d_end) - float(d_start)) / float(step) + 1e-9)) + 1
    return float(d_start) + float(step) * np.arange(max(n, 0))


def bump_profiles(s, s1: float, d: np.ndarray, sigma_b: float) -> np.ndarray:
    """[M, N]: row m is G_{d_m}(s_j), the bump centred at s_1 + d_m (design Section 4)."""
    centres = float(s1) + np.asarray(d, float).reshape(-1, 1)
    return gaussian_bump(np.asarray(s, float)[None, :], centres, sigma_b)


# ------------------------------------------------------------------------- gate directions
#
# Gates 1, 2 and 5 probe the fields along the same directions (one definition, imported by
# kkt.gate1_directions and network.gate2_directions): unit-peak Gaussian bumps of each
# feature in two regions, plus the inside and outside trapezoid shifts.
# - near the car, at s_1 + d for d in GATE_NEAR_D: option A's fixed a_0 and the decision
#   step's steer (an anchored key's s_1 lies 108-171 m before s_c0, so these never reach the
#   corner);
# - at the corner's landmarks (GATE_CORNER: s_c0 - 20, s_c0, s_apex, s_c1, s_c1 + 20), where
#   the in-corner fields that H1's v_apex and s_b comparisons rest on live, on every key.
# The two regions are scored as separate field vectors (``direction_group``): in-corner fields
# are 40-300 times the near-car ones and would swamp their errors in one vector.

GATE_NEAR_D = (-10.0, 0.0, 10.0)            # m, near-car bump offsets from s_1
GATE_CORNER = (("c0-20", "s_c0", -20.0), ("c0", "s_c0", 0.0), ("apex", "s_apex", 0.0),
               ("c1", "s_c1", 0.0), ("c1+20", "s_c1", 20.0))   # (name, landmark, offset m)
GATE_SIGMA_B = 8.0                          # m, the gate bumps' width
REGIONS = ("near", "corner")


def gate_points(s1: float, landmarks: Landmarks, near=GATE_NEAR_D, corner=GATE_CORNER
                ) -> dict[str, float]:
    """Bump centres of the gate directions by name: 'near<d>' at s_1 + d, 'corner:<name>' at
    the landmark plus its offset."""
    out = {f"near{float(d):+g}": float(s1) + float(d) for d in near}
    for name, mark, off in corner:
        out[f"corner:{name}"] = float(getattr(landmarks, mark)) + float(off)
    return out


def gate_directions(n: int, s1: float, landmarks: Landmarks, sigma: int, near=GATE_NEAR_D,
                    corner=GATE_CORNER, sigma_b: float = GATE_SIGMA_B, ds: float = DS
                    ) -> dict[str, Samples]:
    """The gates' directions (module comment above): '<feature>@near<d>' and
    '<feature>@corner:<name>' unit-peak bumps, then 'trapezoid_inside' and
    'trapezoid_outside' (``edge_shift``)."""
    s = sample_s(n, ds)
    out = {}
    pts = gate_points(s1, landmarks, near, corner)
    for f in FEATURES:
        for name, c in pts.items():
            z = zeros(n)._asdict()
            z[f] = gaussian_bump(s, c, sigma_b)
            out[f"{f}@{name}"] = Samples(**z)
    for side in ("inside", "outside"):
        out[f"trapezoid_{side}"] = edge_shift(n, landmarks, sigma, side, 1.0, ds)
    return out


def direction_region(name: str) -> str:
    """'near', 'corner', or '' for a direction outside the two regions (a trapezoid, or a
    bump named otherwise)."""
    if "@" not in name:
        return ""
    where = name.split("@", 1)[1]
    return next((r for r in REGIONS if where.startswith(r)), "")


def direction_group(name: str) -> str:
    """The field vector a direction belongs to: '<feature>@<region>' for the gate bumps
    ('kappa@near+10' -> 'kappa@near', 'w_left@corner:apex' -> 'w_left@corner'), the feature
    alone for other bumps ('kappa@brake' -> 'kappa'), any other direction its own group."""
    if "@" not in name:
        return name
    f = name.split("@", 1)[0]
    r = direction_region(name)
    return f"{f}@{r}" if r else f


def combine(impulse, profiles) -> np.ndarray:
    """Linear combination of impulse responses: impulse [..., N] (d outcome / d sample j) and
    profiles [M, N] (perturbation shapes in sample space) -> [..., M], the outcome's
    derivative along each profile."""
    return np.asarray(impulse, float) @ np.asarray(profiles, float).T


def recode(fields: Samples, sigma: int) -> dict[str, np.ndarray]:
    """Fields with respect to (kappa, w_left, w_right) -> with respect to (kappa_turn, w_in,
    w_out): kappa_turn = sigma kappa (so its field is sigma times kappa's), w_in the inside
    width (left on a left corner)."""
    sig = int(sigma)
    side_feature(sig, "inside")                          # validates sigma
    f = fields._asdict()
    return {"kappa_turn": sig * np.asarray(f["kappa"], float),
            "w_in": np.asarray(f[side_feature(sig, "inside")], float),
            "w_out": np.asarray(f[side_feature(sig, "outside")], float)}


def turn_steer(steer, sigma: int):
    """Steer signed toward the turn (flipped on right-hand corners)."""
    side_feature(int(sigma), "inside")
    return int(sigma) * steer


# ------------------------------------------------------------------------- network-side arrays

_F32: dict[int, tuple[StimulusBank, TrackArrays]] = {}


def float32_arrays(bank: StimulusBank) -> TrackArrays:
    """The whole bank's float32 TrackArrays [B, ...] as the agents' path gathers them
    (``rollout.float32_bank_arrays``: built with x64 off whatever this process runs), NumPy,
    cached per bank object."""
    hit = _F32.get(id(bank))
    if hit is None or hit[0] is not bank:
        hit = (bank, float32_bank_arrays(bank))
        _F32[id(bank)] = hit
    return hit[1]


def float32_row(bank: StimulusBank, stim_id: int) -> TrackArrays:
    """One row of ``float32_arrays`` (NumPy float32; the arrays the agents drove)."""
    return jax.tree.map(lambda a: np.asarray(a)[int(stim_id)], float32_arrays(bank))


def upcast_row(bank: StimulusBank, stim_id: int) -> TrackArrays:
    """Exact float64 upcast of ``float32_row`` (``rollout._upcast``): every float32 value is a
    float64 value, ints and bools unchanged. NumPy."""
    return _upcast(float32_row(bank, stim_id))


def samples32(bank: StimulusBank, stim_id: int) -> Samples:
    """The row's samples as the agents' arrays were built from them (cast to float32), upcast
    exactly to float64: the point at which ``strip_tangent`` linearises."""
    return Samples(*(np.asarray(getattr(bank, f)[int(stim_id)], np.float32).astype(np.float64)
                     for f in FEATURES))


def _require_x64() -> None:
    if jax.dtypes.canonicalize_dtype(jnp.float64) != jnp.float64:
        raise RuntimeError("float64 track arrays need x64 on: run in a float64 process (or "
                           "inside `with jax.enable_x64(True)`), never in a rollout process")


def _strip(ds: float):
    return lambda k, wl, wr: strip_arrays(k, wl, wr, float(ds))


def strip_tangent(samples: Samples, delta: Samples, ds: float = DS) -> TrackArrays:
    """float64 tangent of ``strip_arrays`` at ``samples`` along ``delta`` (``jax.jvp``). Only the
    FLOAT_FIELDS of the result are meaningful (ds, n_points, closed have zero / float0
    tangents). Traceable and linear in ``delta``."""
    _require_x64()
    prim = tuple(jnp.asarray(a, jnp.float64) for a in samples)
    tan = tuple(jnp.asarray(a, jnp.float64) for a in delta)
    return jax.jvp(_strip(ds), prim, tan)[1]


def strip_secant(samples: Samples, delta: Samples, ds: float = DS) -> TrackArrays:
    """strip_arrays(samples + delta) - strip_arrays(samples) in float64 (FLOAT_FIELDS only
    meaningful); the finite-amplitude counterpart of ``strip_tangent``."""
    _require_x64()
    f = _strip(ds)
    s0 = tuple(jnp.asarray(a, jnp.float64) for a in samples)
    s1 = tuple(a + jnp.asarray(d, jnp.float64) for a, d in zip(s0, delta, strict=True))
    a0, a1 = f(*s0), f(*s1)
    return a0._replace(**{k: getattr(a1, k) - getattr(a0, k) for k in FLOAT_FIELDS})


def perturbed_arrays(base: TrackArrays, samples: Samples, delta: Samples, ds: float = DS,
                     mode: str = "tangent") -> TrackArrays:
    """``base`` (float64, the exact upcast of the agents' arrays, ``upcast_row``) plus the
    tangent (or secant) of ``strip_arrays`` at ``samples`` (``samples32``) along ``delta``, on
    the coefficient and pose fields. At delta = 0 it is ``base`` exactly. JAX arrays."""
    if mode == "tangent":
        d = strip_tangent(samples, delta, ds)
    elif mode == "secant":
        d = strip_secant(samples, delta, ds)
    else:
        raise ValueError(f"mode {mode!r} is not 'tangent' or 'secant'")
    out = {k: jnp.asarray(getattr(base, k), jnp.float64) + getattr(d, k) for k in FLOAT_FIELDS}
    return TrackArrays(**out, ds=jnp.asarray(base.ds, jnp.float64),
                       n_points=jnp.asarray(base.n_points, jnp.int32),
                       closed=jnp.asarray(base.closed))


# ------------------------------------------------------------------------- optimum-side track


def _pose(kappa: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(theta, x, y) integrated as ``strip_arrays`` does (trapezoid rule, heading 0), float64."""
    k = np.asarray(kappa, float)
    theta = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * ds)])
    c, s = np.cos(theta), np.sin(theta)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (c[1:] + c[:-1]) * ds)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * ds)])
    return theta, x, y


def perturbed_track(track: Track, delta: Samples | None) -> Track:
    """The optimum's Track with ``delta`` added to its node values of kappa and the widths (the
    TO-OCP's coefficient windows then move by the coefficients of delta, ``packed_coeffs``
    being linear). The pose (not read by the TO-OCP) moves by the float64 secant of the
    strip integration. Name and meta are kept (the lockbox guard reads the name)."""
    if delta is None:
        return track
    if track.closed:
        raise ValueError("perturbations are defined on open stimulus strips")
    dk, dwl, dwr = (np.asarray(a, float) for a in delta)
    k1 = np.asarray(track.kappa, float) + dk
    th0, x0, y0 = _pose(track.kappa, track.ds)
    th1, x1, y1 = _pose(k1, track.ds)
    return dataclasses.replace(
        track, kappa=k1, w_left=np.asarray(track.w_left, float) + dwl,
        w_right=np.asarray(track.w_right, float) + dwr, theta=track.theta + (th1 - th0),
        x=track.x + (x1 - x0), y=track.y + (y1 - y0), meta=dict(track.meta, perturbed=True))
