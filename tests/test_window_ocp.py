"""TO-OCP: the CasADi step and lookup are the simulator's, to machine precision."""

from pathlib import Path

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.ocp.window import WindowSpec, step_function, window_eval
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.spline import eval_jax, packed_coeffs
from apexzero.tracks.track import load_tumftm
from apexzero.vehicle.params import F1_2025 as P
from apexzero.vehicle.sim import rk2_step, track_arrays

TUMFTM = Path("data/tracks/tumftm")
pytestmark = pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")


def test_windowed_spline_and_step_match_the_simulator():
    jax.config.update("jax_enable_x64", True)
    tr = refine_reference(load_tumftm(TUMFTM / "Hockenheim.csv"))
    spec = WindowSpec()
    K, n = spec.n_coef, tr.n_points
    coef = packed_coeffs(tr.kappa, True, n)
    s, C, j0 = ca.SX.sym("s"), ca.SX.sym("C", K), ca.SX.sym("j0")
    F = ca.Function("w", [s, C, j0], [window_eval(s, C, j0, 1.0 / tr.ds, K)])
    phi = step_function(spec)
    cla = P.cla_f + P.cla_r
    ctx = np.array([P.m, P.p_max, cla, P.cda, P.mu_scale, P.cla_f / cla])
    ta = track_arrays(tr, dtype=jnp.float64)
    for sigma in (30.0, 2500.0, tr.length - 3.0):   # includes the start/finish wrap
        jj = int(np.floor(sigma / tr.ds)) - spec.k_behind
        idx = np.mod(jj + np.arange(K), n) + 1
        for q in np.linspace(sigma - spec.radius, sigma + spec.radius + 6.0, 25):
            ref = float(eval_jax(jnp.asarray(coef), jnp.asarray(q), tr.ds, jnp.asarray(n), jnp.asarray(True)))
            assert abs(float(F(q, coef[idx], jj)) - ref) < 1e-12
        x = np.array([sigma, 0.7, 0.02, 55.0, 0.4, 0.08, 0.03, 0.2])
        a = np.array([0.1, 0.3])
        xc = np.asarray(phi(x, a, coef[idx], jj, 1.0 / tr.ds, ctx)).ravel()
        xj = jnp.asarray(x)
        for _ in range(spec.substeps):
            xj = rk2_step(xj, jnp.asarray(a), ta, P, spec.dt / spec.substeps)
        assert np.max(np.abs(xc - np.asarray(xj))) < 1e-10
