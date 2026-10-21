"""Step 2's sample-space perturbations (fields.perturb): profiles, recoding, and the float64
track arrays both pipelines perturb.

- The base arrays are exact upcasts of the agents' float32 arrays; a zero perturbation leaves
  them bit for bit.
- The network's tangent of strip_arrays and the optimum's perturbed Track move the spline
  coefficients identically (both natural-spline solves of the same delta).
- A VJP through the perturbed arrays gives per-sample impulse responses whose combination
  with the trapezoid is the inside-shift derivative: 1 for a line held fixed (a width-blind
  line has G = 0).

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
from apexzero.tracks.spline import packed_coeffs

HAVE_DESIGN = (io.STIMULI_DIR / "cells.json").exists()
HAVE_KEYS = (io.KEYS_DIR / "nominal").exists()
LM = Landmarks(1000.0, 1010.0, 1030.0, 1050.0, 1060.0)


@pytest.fixture(autouse=True)
def _restore_x64():
    prev = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", prev)


# ------------------------------------------------------------------------- profiles


def test_trapezoid():
    s = np.array([0.0, 959.9, 960.0, 970.0, 980.0, 1030.0, 1080.0, 1090.0, 1100.0, 1200.0])
    h = PT.trapezoid(s, LM.s_c0, LM.s_c1)
    np.testing.assert_allclose(h, [0, 0, 0, 0.5, 1, 1, 1, 0.5, 0, 0], atol=1e-12)
    # a short corner keeps its plateau from s_c0 - 20 to s_c1 + 20
    assert PT.trapezoid(np.array([990.0]), 1000.0, 1000.0)[0] == 1.0


@pytest.mark.parametrize("sigma,inside,outside", [(1, "w_left", "w_right"),
                                                  (-1, "w_right", "w_left")])
def test_edge_shift_and_recoding(sigma, inside, outside):
    n = 600
    d = PT.edge_shift(n, LM, sigma, "inside", delta=0.5)
    assert np.all(d.kappa == 0) and np.all(getattr(d, outside) == 0)
    h = PT.trapezoid(PT.sample_s(n), LM.s_c0, LM.s_c1)
    np.testing.assert_array_equal(getattr(d, inside), 0.5 * h)
    o = PT.edge_shift(n, LM, sigma, "outside")
    np.testing.assert_array_equal(getattr(o, outside), h)
    f = PT.Samples(np.full(3, 2.0), np.full(3, 3.0), np.full(3, 5.0))
    r = PT.recode(f, sigma)
    assert r["kappa_turn"][0] == 2.0 * sigma
    assert r["w_in"][0] == getattr(f, inside)[0] and r["w_out"][0] == getattr(f, outside)[0]
    assert PT.turn_steer(0.3, sigma) == 0.3 * sigma
    with pytest.raises(ValueError):
        PT.recode(f, 0)


def test_gate_directions_reach_the_corner_on_every_key():
    """One definition of the gates' directions: near-car bumps at s_1 + d, corner bumps at
    the landmarks whatever s_1 is (an anchored key's s_1 lies 108-171 m before s_c0), and
    both trapezoids; each feature's near and corner bumps are separate field vectors."""
    n = 800
    s = PT.sample_s(n)
    for s1 in (LM.s_c0 - 171.0, LM.s_c0 - 50.0):
        dirs = PT.gate_directions(n, s1, LM, 1)
        assert len(dirs) == 3 * (len(PT.GATE_NEAR_D) + len(PT.GATE_CORNER)) + 2
        for f in PT.FEATURES:
            c = dirs[f"{f}@corner:apex"]
            assert s[int(np.argmax(getattr(c, f)))] == pytest.approx(LM.s_apex, abs=1.0)
            nr = dirs[f"{f}@near+10"]
            assert s[int(np.argmax(getattr(nr, f)))] == pytest.approx(s1 + 10.0, abs=1.0)
            assert all(np.all(getattr(c, g) == 0) for g in PT.FEATURES if g != f)
        # the corner bumps sit inside the c_min window [s_c0 - 50, s_c1 + 50]
        pts = PT.gate_points(s1, LM)
        assert all(LM.s_c0 - 50 <= v <= LM.s_c1 + 50 for k, v in pts.items() if "corner" in k)
    assert PT.direction_group("kappa@near-10") == "kappa@near"
    assert PT.direction_group("w_left@corner:c1+20") == "w_left@corner"
    assert PT.direction_group("trapezoid_inside") == "trapezoid_inside"
    assert PT.direction_group("kappa@brake") == "kappa"                 # other bumps
    assert PT.direction_region("w_right@corner:c0") == "corner"


def test_bumps_and_combination():
    s = PT.sample_s(800)
    d = PT.d_grid(100.0)
    assert d[0] == -20.0 and d[-1] == 100.0 and np.allclose(np.diff(d), 2.0)
    G = PT.bump_profiles(s, 1000.0, d, 8.0)
    assert G.shape == (len(d), len(s))
    j = int(np.argmax(G[10]))
    assert s[j] == 1000.0 + d[10] and G[10, j] == 1.0
    np.testing.assert_allclose(G[10, j + 4], np.exp(-0.5), rtol=1e-12)    # 8 m = one sigma_b
    rng = np.random.default_rng(1)
    imp = rng.standard_normal((3, len(s)))
    out = PT.combine(imp, G)
    assert out.shape == (3, len(d))
    np.testing.assert_allclose(out[:, 10], imp @ G[10], rtol=1e-12)
    np.testing.assert_allclose(PT.combine(np.eye(len(s))[[j]], G)[0], G[:, j], rtol=0)


# ------------------------------------------------------------------------- track arrays


@pytest.mark.skipif(not HAVE_DESIGN, reason="step 1's stimulus design is not on this machine")
def test_upcast_and_zero_perturbation():
    bank, _ = K.design()
    sid = 40
    f32 = PT.float32_row(bank, sid)
    up = PT.upcast_row(bank, sid)
    assert f32.kappa_c.dtype == np.float32 and up.kappa_c.dtype == np.float64
    for k in PT.FLOAT_FIELDS:
        np.testing.assert_array_equal(getattr(up, k), getattr(f32, k).astype(np.float64))
    assert int(up.n_points) == bank.kappa.shape[1] and not bool(up.closed)
    # the optimum's splines (packed_coeffs of bank.track's node values) are the agents' to
    # rounding: both pipelines start from the same road
    ka = K.key_arrays(bank.track(sid))
    for k in ("kappa_c", "w_left_c", "w_right_c"):
        np.testing.assert_allclose(getattr(ka, k), getattr(up, k), rtol=0, atol=1e-12)
    samples = PT.samples32(bank, sid)
    with jax.enable_x64(True):
        same = PT.perturbed_arrays(up, samples, PT.zeros(len(samples.kappa)))
        for k in PT.FLOAT_FIELDS:
            np.testing.assert_array_equal(np.asarray(getattr(same, k)), getattr(up, k))
    with jax.enable_x64(False), pytest.raises(RuntimeError):
        PT.strip_tangent(samples, PT.zeros(len(samples.kappa)))


@pytest.mark.skipif(not HAVE_DESIGN, reason="step 1's stimulus design is not on this machine")
def test_tangent_secant_and_optimum_agree():
    """Widths are linear: tangent = secant to rounding. Curvature: the pose is not, so the two
    differ at second order. The optimum's perturbed Track moves its packed coefficients by
    exactly the tangent's coefficient change."""
    bank, _ = K.design()
    sid = 40
    samples = PT.samples32(bank, sid)
    n = len(samples.kappa)
    rng = np.random.default_rng(2)
    bump = PT.gaussian_bump(PT.sample_s(n), 1000.0, 8.0)
    dw = PT.Samples(np.zeros(n), 0.3 * bump, -0.2 * bump * rng.uniform(0.5, 1.5, n))
    with jax.enable_x64(True):
        t = PT.strip_tangent(samples, dw)
        c = PT.strip_secant(samples, dw)
        for k in ("w_left_c", "w_right_c"):
            np.testing.assert_allclose(np.asarray(getattr(t, k)), np.asarray(getattr(c, k)),
                                       rtol=0, atol=1e-13)
        for k in ("kappa_c", "x", "y", "theta"):
            assert np.max(np.abs(np.asarray(getattr(t, k)))) == 0.0
        errs = []
        for eps in (1e-3, 5e-4):
            dk = PT.Samples(eps * bump, np.zeros(n), np.zeros(n))
            t, c = PT.strip_tangent(samples, dk), PT.strip_secant(samples, dk)
            np.testing.assert_allclose(np.asarray(t.kappa_c), np.asarray(c.kappa_c), rtol=0,
                                       atol=1e-15)
            np.testing.assert_allclose(np.asarray(t.theta), np.asarray(c.theta), rtol=0, atol=1e-15)
            errs.append(np.max(np.hypot(np.asarray(t.x) - np.asarray(c.x),
                                        np.asarray(t.y) - np.asarray(c.y))))
        assert 3.0 < errs[0] / errs[1] < 5.0                      # second order
        # the optimum's side: packed_coeffs of the Track's node values plus delta
        track = bank.track(sid)
        t = PT.strip_tangent(samples, dw)
        moved = PT.perturbed_track(track, dw)
        for f, k in (("w_left", "w_left_c"), ("w_right", "w_right_c")):
            dc = (packed_coeffs(getattr(moved, f), False, n)
                  - packed_coeffs(getattr(track, f), False, n))
            np.testing.assert_allclose(dc, np.asarray(getattr(t, k)), rtol=0, atol=1e-12)
        assert moved.name == track.name and PT.perturbed_track(track, None) is track


@pytest.mark.skipif(not (HAVE_DESIGN and HAVE_KEYS), reason="step 1's keys are not here")
def test_impulse_responses_by_vjp():
    """c_min of a stored key's line, held fixed, on the network-side arrays: the VJP's
    per-sample impulse responses, combined with the trapezoid, give d c_min / d Delta = 1 (a
    fixed line does not follow the edge: G = 0), and match finite differences (secant)."""
    bank, _ = K.design()
    for ref in K.a2_refs()[::7]:                   # a key whose c_min has no near-tie
        rec = K.load_key(ref)
        stim = bank.stimuli[rec.stim_id]
        sigma, lm = int(stim.direction), stim.landmarks
        base = PT.upcast_row(bank, rec.stim_id)
        plan = O.c_min_plan(O.trajectory(rec.x), lm, sigma, base)
        if plan.valid and not plan.near_tie:
            break
    assert plan.valid and not plan.near_tie
    samples = PT.samples32(bank, rec.stim_id)
    n = len(samples.kappa)
    with jax.enable_x64(True):
        x = jnp.asarray(rec.x)

        def c_of(delta):
            return O.c_min(x, PT.perturbed_arrays(base, samples, delta), sigma, plan.pick)

        val, vjp = jax.vjp(c_of, PT.zeros(n))
        assert float(val) == pytest.approx(plan.value, abs=1e-12)
        imp = vjp(jnp.ones(()))[0]
        fields = PT.recode(PT.Samples(*(np.asarray(a) for a in imp)), sigma)
        h = PT.edge_shift(n, lm, sigma, "inside")
        side = PT.side_feature(sigma, "inside")
        assert PT.combine(fields["w_in"], getattr(h, side)[None, :])[0] == pytest.approx(1.0, abs=1e-5)
        assert np.all(fields["kappa_turn"] == 0.0) and np.all(fields["w_out"] == 0.0)
        j = int(np.argmax(np.abs(fields["w_in"])))
        for jj in (j - 1, j, j + 2):
            d = PT.zeros(n)._asdict()
            d[side] = np.zeros(n)
            d[side][jj] = 1e-4
            plus = PT.perturbed_arrays(base, samples, PT.Samples(**d), mode="secant")
            d[side][jj] = -1e-4
            minus = PT.perturbed_arrays(base, samples, PT.Samples(**d), mode="secant")
            fd = (O.c_min_plan(O.trajectory(rec.x), lm, sigma, jax.tree.map(np.asarray, plus)).value
                  - O.c_min_plan(O.trajectory(rec.x), lm, sigma,
                                 jax.tree.map(np.asarray, minus)).value) / 2e-4
            assert fields["w_in"][jj] == pytest.approx(fd, rel=1e-6, abs=1e-9)
