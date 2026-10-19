"""Measured values behind the physics gates of the paper's Table 3 (tab:gates).

The unit tests only assert thresholds. This script repeats four of their computations
exactly and prints the measured quantity, so the table can report a measurement rather
than the criterion:

- JAX-CasADi parity   tests/test_dynamics.py::test_jax_casadi_parity
- braking             tests/test_dynamics.py::test_braking_deceleration_is_f1_like
- track lookup        tests/test_tracks.py::test_spline_lookup_is_identical_in_jax_numpy_and_casadi
- TO-OCP step         tests/test_window_ocp.py::test_windowed_spline_and_step_match_the_simulator
                      (on Hockenheim, a development circuit; corners.R_DEV)

Run from the repository root (needs data/tracks/tumftm):

    uv run python paper/tools/gate_values.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
jax.config.update("jax_enable_x64", True)

from test_dynamics import random_states

from apexzero.ocp.window import WindowSpec, step_function, window_eval
from apexzero.tracks.corners import R_DEV
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.spline import eval_casadi, eval_jax, packed_coeffs, spline_np
from apexzero.tracks.track import load_tumftm
from apexzero.vehicle import dynamics
from apexzero.vehicle.dynamics import body_derivatives
from apexzero.vehicle.ops import CASADI, JAX
from apexzero.vehicle.params import F1_2025 as P
from apexzero.vehicle.sim import rk2_step, track_arrays


def parity() -> tuple[float, float]:
    rng = np.random.default_rng(0)
    n = 10_000
    xs = random_states(n, rng)
    us = rng.uniform(-1, 1, (n, 2))
    ks = rng.uniform(-0.08, 0.08, n)
    fj = np.asarray(dynamics.f(jnp.asarray(xs), jnp.asarray(us), jnp.asarray(ks), P, JAX))
    x, u, k = ca.MX.sym("x", 8), ca.MX.sym("u", 2), ca.MX.sym("k")
    fun = ca.Function("f", [x, u, k], [dynamics.f(x, u, k, P, CASADI)])
    fc = np.asarray(fun.map(n)(xs.T, us.T, ks[None, :])).T
    diff = np.abs(fj - fc)
    return float(np.max(diff / np.maximum(np.abs(fc), 1.0))), float(diff.max())


def braking() -> dict[float, float]:
    out = {}
    for v in (83.0, 30.0):
        x = jnp.array([0, 0, 0, v, 0, 0, 0, -1.0])
        out[v] = -float(body_derivatives(x, jnp.array([0.0, -1.0]), P, JAX)[0]) / P.g
    return out


def lookup() -> float:
    rng = np.random.default_rng(0)
    worst = 0.0
    for closed in (True, False):
        n, ds = 400, 2.0
        sg = np.arange(n) * ds
        f = np.sin(sg / 40.0) + 0.2 * np.cos(sg / 7.0)
        coef = packed_coeffs(f, closed, pad_to=n + 10)
        s = rng.uniform(0, (n - 1) * ds, 2000)
        vj = np.asarray(eval_jax(jnp.asarray(coef), jnp.asarray(s), ds, jnp.asarray(n),
                                 jnp.asarray(closed)))
        vn = spline_np(coef, s, ds, n, closed)
        vc = np.asarray(eval_casadi(coef, ds, n, closed).map(len(s))(s[None, :])).ravel()
        worst = max(worst, float(np.max(np.abs(vj - vc))), float(np.max(np.abs(vn - vc))))
    return worst


def to_ocp_step() -> tuple[float, float]:
    name = "Hockenheim"
    assert name in R_DEV
    tr = refine_reference(load_tumftm(ROOT / "data/tracks/tumftm" / f"{name}.csv"))
    spec = WindowSpec()
    K, n = spec.n_coef, tr.n_points
    coef = packed_coeffs(tr.kappa, True, n)
    s, C, j0 = ca.SX.sym("s"), ca.SX.sym("C", K), ca.SX.sym("j0")
    F = ca.Function("w", [s, C, j0], [window_eval(s, C, j0, 1.0 / tr.ds, K)])
    phi = step_function(spec)
    cla = P.cla_f + P.cla_r
    ctx = np.array([P.m, P.p_max, cla, P.cda, P.mu_scale, P.cla_f / cla])
    ta = track_arrays(tr, dtype=jnp.float64)
    look, state = 0.0, 0.0
    for sigma in (30.0, 2500.0, tr.length - 3.0):
        jj = int(np.floor(sigma / tr.ds)) - spec.k_behind
        idx = np.mod(jj + np.arange(K), n) + 1
        for q in np.linspace(sigma - spec.radius, sigma + spec.radius + 6.0, 25):
            ref = float(eval_jax(jnp.asarray(coef), jnp.asarray(q), tr.ds, jnp.asarray(n),
                                 jnp.asarray(True)))
            look = max(look, abs(float(F(q, coef[idx], jj)) - ref))
        x = np.array([sigma, 0.7, 0.02, 55.0, 0.4, 0.08, 0.03, 0.2])
        a = np.array([0.1, 0.3])
        xc = np.asarray(phi(x, a, coef[idx], jj, 1.0 / tr.ds, ctx)).ravel()
        xj = jnp.asarray(x)
        for _ in range(spec.substeps):
            xj = rk2_step(xj, jnp.asarray(a), ta, P, spec.dt / spec.substeps)
        state = max(state, float(np.max(np.abs(xc - np.asarray(xj)))))
    return state, look


if __name__ == "__main__":
    rel, absd = parity()
    print(f"parity: max |f_jax - f_casadi| / max(|f_casadi|, 1) = {rel:.2e} (max abs {absd:.2e})")
    for v, g in braking().items():
        print(f"braking: full pedal at {v:.0f} m/s decelerates at {g:.2f} g")
    print(f"track lookup: max |JAX or NumPy - CasADi| = {lookup():.2e}")
    st, lk = to_ocp_step()
    print(f"TO-OCP step (Hockenheim): max state diff {st:.2e}, max windowed lookup diff {lk:.2e}")
