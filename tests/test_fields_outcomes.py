"""Step 2's outcomes (fields.outcomes): c_min, v_apex and s_b.

- On stored answer keys, v_apex and s_b are step 1's stored v_min and s_b, and the JAX readers
  return the plan's values.
- On synthetic float64 trajectories, the JAX gradients (pick held: Danskin) agree with central
  differences of the full outcome (window, argmin and onset recomputed), with respect to the
  states and to the width samples.
- Sign conventions: c < 0 is exactly the inside half of the env's termination set, on left and
  right corners, and c_min is mirror-invariant.

float64 runs inside ``jax.enable_x64(True)`` blocks; the module restores the global setting.
No agent network is evaluated."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.fields import keys as K
from apexzero.fields import outcomes as O
from apexzero.fields import perturb as PT
from apexzero.interp import io
from apexzero.interp.stimuli import Landmarks
from apexzero.tracks.generator import strip_arrays
from apexzero.vehicle.sim import widths_at

N = 700
DS = 2.0
LM = Landmarks(600.0, 612.0, 640.0, 668.0, 680.0)
HAVE_KEYS = (io.KEYS_DIR / "anchored").exists()


@pytest.fixture(autouse=True)
def _restore_x64():
    prev = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", prev)


def corner_samples(sigma: int = 1) -> PT.Samples:
    """A trapezoid corner at s_c0 = 600 m (radius 40 m) with smoothly varying widths."""
    s = np.arange(N) * DS
    u = s - LM.s_c0
    prof = np.clip(np.minimum(u / 12.0, (LM.L_c - u) / 12.0), 0.0, 1.0)
    k = sigma * prof / 40.0
    wl = 6.0 + 0.4 * np.sin(s / 37.0)
    wr = 6.0 + 0.3 * np.cos(s / 29.0)
    return PT.Samples(k, wl, wr)


def synthetic_run(sigma: int = 1, K_: int = 120, s0: float = 480.0, c_floor: float = 0.4,
                  v_end_dip: bool = False) -> np.ndarray:
    """States [K, 8]: s advancing ~4 m/step with a wobble, a line whose inside clearance has a
    single dip near the apex, a speed dip, and a brake run (pedal to -0.8) before the corner."""
    k = np.arange(K_, dtype=float)
    s = s0 + 4.0 * k + 0.3 * np.sin(k / 3.0)
    x = np.zeros((K_, 8))
    x[:, 0] = s
    w_in = (6.0 + 0.4 * np.sin(s / 37.0)) if sigma > 0 else (6.0 + 0.3 * np.cos(s / 29.0))
    c = c_floor + 0.8 * ((s - 645.0) / 30.0) ** 2 / (1.0 + ((s - 645.0) / 30.0) ** 2)
    x[:, 1] = sigma * (w_in - 1.0 - c)
    x[:, 2] = 0.01 * np.sin(k / 5.0)
    if v_end_dip:
        x[:, 3] = 70.0 - 0.1 * (s - s0)                 # still slowing through the window
    else:
        x[:, 3] = 70.0 - 35.0 * np.exp(-(((s - 640.0) / 45.0) ** 2))
    x[:, 4] = 0.2 * np.sin(k / 7.0)
    x[:, 5] = 0.1 * np.cos(k / 6.0)

    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    x[:, 7] = 1.0 - 1.8 * sig((k - 20.3) / 1.3) * (1.0 - sig((k - 34.7) / 1.3))
    return x


def tr64(samples: PT.Samples):
    return strip_arrays(*(jnp.asarray(a, jnp.float64) for a in samples), DS)


def np_tr(tr):
    return jax.tree.map(np.asarray, tr)


# ------------------------------------------------------------------------- stored keys


@pytest.mark.skipif(not HAVE_KEYS, reason="step 1's answer keys are not on this machine")
def test_key_outcomes_equal_step1_measures():
    a2 = K.a2_refs()
    fin = K.anchored_finals()
    refs = ([r for r in a2 if r.kind == "nominal"][::40] + [r for r in a2 if r.kind == "phase"][::90]
            + [fin[a][i] for a in ("G1", "G3s1", "S1") for i in (0, 333)])
    bank, _ = K.design()
    p = O.drive_params()
    n_brake = 0
    for ref in refs:
        rec = K.load_key(ref)
        stim = bank.stimuli[rec.stim_id]
        lm, sigma = stim.landmarks, int(stim.direction)
        tr = K.key_arrays(bank.track(rec.stim_id))
        traj = O.trajectory(rec.x)
        pl = O.plans(traj, lm, sigma, tr)
        m = rec.measures
        assert pl["v_apex"].valid and abs(pl["v_apex"].value - m["v_min"]) <= 1e-12, ref.key_id
        if np.isfinite(m["s_b"]):
            n_brake += 1
            assert pl["s_b"].valid and abs(pl["s_b"].value - m["s_b"]) <= 1e-12, ref.key_id
        else:
            assert not pl["s_b"].valid
        # the optimum's corridor carries a 3 cm safety margin: c_min >= 0.03 up to the
        # rebuilt constraint tolerance (rented-machine keys: ~1e-7)
        assert pl["c_min"].valid and pl["c_min"].value >= 0.03 - 1e-6
        with jax.enable_x64(True):
            x = jnp.asarray(rec.x)
            trj = jax.tree.map(jnp.asarray, tr)
            assert abs(float(O.c_min(x, trj, sigma, pl["c_min"].pick)) - pl["c_min"].value) <= 1e-11
            assert abs(float(O.v_apex(x, pl["v_apex"].pick)) - pl["v_apex"].value) <= 1e-11
            if pl["s_b"].valid:
                assert abs(float(O.s_b(x, pl["s_b"].pick, p)) - pl["s_b"].value) <= 1e-9
    assert n_brake >= 6


# ------------------------------------------------------------------------- synthetic gradients


def _full(name, x, tr, sigma):
    """The outcome with everything recomputed from (x, the track arrays): window, argmin,
    onset."""
    traj = O.trajectory(x)
    if name == "c_min":
        return O.c_min_plan(traj, LM, sigma, tr).value
    if name == "v_apex":
        return O.v_apex_plan(traj, LM).value
    return O.s_b_plan(traj, LM).value


def _fd_states(name, x, samples, sigma, cols, h=1e-6):
    tr = np_tr(tr64(samples))
    g = np.zeros_like(x)
    for i in range(len(x)):
        for c in cols:
            xp, xm = x.copy(), x.copy()
            xp[i, c] += h
            xm[i, c] -= h
            g[i, c] = (_full(name, xp, tr, sigma) - _full(name, xm, tr, sigma)) / (2 * h)
    return g


@pytest.mark.parametrize("sigma", [1, -1])
def test_gradients_match_finite_differences(sigma):
    with jax.enable_x64(True):
        samples = corner_samples(sigma)
        x = synthetic_run(sigma)
        tr = tr64(samples)
        traj = O.trajectory(x)
        pl = O.plans(traj, LM, sigma, np_tr(tr))
        assert all(v.valid for v in pl.values()), {k: v.reason for k, v in pl.items()}
        assert not pl["c_min"].near_tie and not pl["v_apex"].near_tie
        assert pl["s_b"].pick.k > 0
        p = O.drive_params()
        fns = {"c_min": lambda xx: O.c_min(xx, tr, sigma, pl["c_min"].pick),
               "v_apex": lambda xx: O.v_apex(xx, pl["v_apex"].pick),
               "s_b": lambda xx: O.s_b(xx, pl["s_b"].pick, p)}
        cols = {"c_min": (0, 1), "v_apex": (0, 3), "s_b": (0, 3, 7)}
        for name, fn in fns.items():
            assert abs(float(fn(jnp.asarray(x))) - pl[name].value) <= 1e-12
            g = np.asarray(jax.grad(fn)(jnp.asarray(x)))
            fd = _fd_states(name, x, samples, sigma, cols[name])
            assert np.abs(g).max() > 0.0
            others = [c for c in range(8) if c not in cols[name]]
            assert np.all(g[:, others] == 0.0), name
            np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7, err_msg=name)


@pytest.mark.parametrize("sigma", [1, -1])
def test_c_min_width_gradient(sigma):
    """d c_min / d width samples through strip_arrays: only the inside width moves it, its
    knots around the argmin dominate (the natural spline's cardinal functions decay fast), a
    uniform shift moves it one for one, and the derivative matches central differences."""
    with jax.enable_x64(True):
        samples = corner_samples(sigma)
        x = synthetic_run(sigma)
        pl = O.c_min_plan(O.trajectory(x), LM, sigma, np_tr(tr64(samples)))
        assert not pl.pick.interp

        def f(wl, wr):
            return O.c_min(jnp.asarray(x), tr64(PT.Samples(samples.kappa, wl, wr)), sigma, pl.pick)

        gl, gr = (np.asarray(v) for v in jax.grad(f, argnums=(0, 1))(
            jnp.asarray(samples.w_left), jnp.asarray(samples.w_right)))
        g_in, g_out = (gl, gr) if sigma > 0 else (gr, gl)
        assert np.all(g_out == 0.0)
        j = int(x[pl.pick.i0, 0] // DS)
        # natural-spline coefficients are global but decay fast: the knots near s* dominate
        assert np.abs(g_in[j - 1: j + 3]).sum() > 0.9 * np.abs(g_in).sum()
        np.testing.assert_allclose(g_in.sum(), 1.0, atol=1e-9)     # a uniform shift moves c 1:1
        h = 1e-6
        feat = "w_left" if sigma > 0 else "w_right"
        for jj in range(j - 3, j + 5):
            d = np.zeros(N)
            d[jj] = h
            plus = samples._replace(**{feat: getattr(samples, feat) + d})
            minus = samples._replace(**{feat: getattr(samples, feat) - d})
            fd = (_full("c_min", x, np_tr(tr64(plus)), sigma)
                  - _full("c_min", x, np_tr(tr64(minus)), sigma)) / (2 * h)
            np.testing.assert_allclose(g_in[jj], fd, rtol=1e-5, atol=1e-8)


def test_window_end_pick():
    """A speed still falling at s_c1 + 50: v_apex is read at the interpolated window end, as
    np.interp reads it, and its gradient (vx and s of the bracketing states) matches FD."""
    with jax.enable_x64(True):
        x = synthetic_run(1, v_end_dip=True)
        pl = O.v_apex_plan(O.trajectory(x), LM)
        assert pl.valid and pl.pick.interp and pl.s_at == O.window(LM)[1]
        s, vx = x[:, 0], x[:, 3]
        assert pl.value == pytest.approx(float(np.interp(pl.s_at, s, vx)), abs=1e-12)
        g = np.asarray(jax.grad(lambda xx: O.v_apex(xx, pl.pick))(jnp.asarray(x)))
        assert set(np.flatnonzero(np.abs(g).sum(1))) == {pl.pick.i0, pl.pick.i1}
        fd = _fd_states("v_apex", x, corner_samples(1), 1, (0, 3))
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


def test_window_start_clipped_to_the_run_start():
    """A run starting inside the window (A2 cells with D < 50 m): the window's start is the
    initial state itself, as in step 1's ``_View.seg``."""
    x = synthetic_run(1, s0=560.0)
    x[:, 3] = 40.0 + 0.05 * (x[:, 0] - 560.0)        # slowest at the start
    pl = O.v_apex_plan(O.trajectory(x), LM)
    assert pl.pick == O.Pick(0, 0, 0.0, False) and pl.value == 40.0


# ------------------------------------------------------------------------- conventions


@pytest.mark.parametrize("sigma", [1, -1])
def test_clearance_sign_and_termination_boundary(sigma):
    """c < 0 exactly where the env terminates on the inside edge (racing.py: n > w_left - 1 or
    n < -(w_right - 1)); the outside edge never makes c negative."""
    with jax.enable_x64(True):
        samples = corner_samples(sigma)
        tr = tr64(samples)
        s = np.full(41, 641.3)
        wl, wr = (np.asarray(v) for v in widths_at(jnp.asarray(s), tr))
        edge_in = wl[0] - 1.0 if sigma > 0 else -(wr[0] - 1.0)
        n = edge_in + sigma * np.linspace(-0.2, 0.2, 41)      # through the inside boundary
        x = np.zeros((41, 8))
        x[:, 0], x[:, 1] = s, n
        c = np.asarray(O.clearance(jnp.asarray(x), tr, sigma))
        off_inside = (n > wl - 1.0) if sigma > 0 else (n < -(wr - 1.0))
        np.testing.assert_array_equal(c < 0.0, off_inside)
        assert abs(c[20]) <= 1e-12                              # on the boundary
        np.testing.assert_allclose(c, O.clearance_np(x, np_tr(tr), sigma), atol=1e-12)
        assert PT.side_feature(sigma, "inside") == ("w_left" if sigma > 0 else "w_right")
        # deep on the outside: large clearance, never negative
        x[:, 1] = -sigma * 4.9
        assert np.all(np.asarray(O.clearance(jnp.asarray(x), tr, sigma)) > 0.0)


def test_c_min_is_mirror_invariant():
    with jax.enable_x64(True):
        left = corner_samples(1)
        right = PT.Samples(-left.kappa, left.w_right, left.w_left)    # the mirror strip
        x = synthetic_run(1)
        xm = x * np.array([1, -1, -1, 1, -1, -1, -1, 1.0])
        a = O.c_min_plan(O.trajectory(x), LM, 1, np_tr(tr64(left)))
        b = O.c_min_plan(O.trajectory(xm), LM, -1, np_tr(tr64(right)))
        assert a.value == pytest.approx(b.value, abs=1e-12) and a.pick == b.pick


# ------------------------------------------------------------------------- validity, ties


def test_validity():
    x = synthetic_run(1)
    lm_b = O.window(LM)[1]
    assert O.validity(O.trajectory(x), LM) == (True, "")
    # crashed before the window end (states frozen after termination, as the harness does)
    kt = int(np.flatnonzero(x[:, 0] > 650.0)[0])
    xc = x.copy()
    xc[kt + 1:] = xc[kt]
    valid = np.arange(len(x)) <= kt
    tc = O.trajectory(xc, valid, term_step=kt - 1)
    ok, why = O.validity(tc, LM)
    assert not ok and "crashed" in why
    with jax.enable_x64(True):
        tr = np_tr(tr64(corner_samples(1)))
    assert all(not v.valid for v in O.plans(tc, LM, 1, tr).values())
    # alive but too short to reach s_c1 + 50
    short = x[x[:, 0] < lm_b - 1.0]
    ok, why = O.validity(O.trajectory(short), LM)
    assert not ok and "not reached" in why
    # a crash after the window end leaves the outcome defined
    kt2 = int(np.flatnonzero(x[:, 0] > lm_b + 20.0)[0])
    assert O.validity(O.trajectory(x[: kt2 + 1], term_step=kt2 - 1), LM) == (True, "")
    with pytest.raises(ValueError):
        O.outcome("v_apex", x, O.v_apex_plan(tc, LM))


def test_near_tie_rule():
    assert O.strict_minima(np.array([3, 1, 2, 1.02, 3.0])) == [1, 3]
    assert O.strict_minima(np.array([3, 1, 1, 1, 3.0])) == [1]          # a plateau once
    assert O.strict_minima(np.array([1, 2, 3.0])) == [0]                # one-sided end
    assert O.strict_minima(np.array([3, 2, 1.0])) == [2]
    s = np.arange(5.0)
    assert O._tie(s, np.array([3, 1, 2, 1.02, 3.0]), O.C_MIN_TIE)[0]
    assert not O._tie(s, np.array([3, 1, 2, 1.2, 3.0]), O.C_MIN_TIE)[0]
    assert O._tie(s, np.array([30, 10, 20, 10.4, 30.0]), O.V_APEX_TIE)[0]
    # a double dip in a synthetic run is flagged on the run itself
    x = synthetic_run(1)
    x[:, 3] = 60.0 - 20.0 * np.exp(-(((x[:, 0] - 610.0) / 10.0) ** 2)) \
        - 19.8 * np.exp(-(((x[:, 0] - 670.0) / 10.0) ** 2))
    pl = O.v_apex_plan(O.trajectory(x), LM)
    assert pl.near_tie and len(pl.minima) >= 2
