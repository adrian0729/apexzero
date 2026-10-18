import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.env.racing import (
    OBS_DIM,
    EnvParams,
    EnvState,
    default_context,
    observe,
    params_from_context,
    reset,
    step,
    step_no_reset,
)
from apexzero.tracks.track import from_centreline
from apexzero.vehicle.sim import track_arrays

jax.config.update("jax_enable_x64", True)


def wavy_track(n=1500, mirror=False):
    """A closed track with left and right corners and asymmetric widths."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 600 + 80 * np.sin(3 * t) + 30 * np.cos(5 * t)
    xy = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    wl = 6.0 + 1.0 * np.sin(2 * t)
    wr = 5.0 + 0.5 * np.cos(4 * t)
    tr = from_centreline("wavy", xy, wl, wr, ds=2.0)
    return tr.mirrored() if mirror else tr


def make_state(x, ctx=None):
    x = jnp.asarray(x, dtype=jnp.float64)
    return EnvState(x=x, prev_action=jnp.array([0.1, 0.2]), accel=jnp.array([1.0, -2.0]),
                    ctx=default_context() if ctx is None else ctx, t=jnp.array(0),
                    slow=jnp.array(0.0), s0=x[0])


def test_observation_shape_and_finiteness():
    ta = track_arrays(wavy_track(), dtype=jnp.float64)
    v_ref = jnp.full(ta.x.shape, 40.0)
    obs, st = reset(jax.random.PRNGKey(0), ta, v_ref, default_context())
    assert obs.shape == (OBS_DIM,)
    assert bool(jnp.all(jnp.isfinite(obs)))


def test_mirror_symmetry_of_observations():
    """Mirroring the track and the car's lateral state must mirror every observation."""
    ta = track_arrays(wavy_track(), dtype=jnp.float64)
    tm = track_arrays(wavy_track(mirror=True), dtype=jnp.float64)
    x = np.array([1234.0, 1.3, 0.05, 55.0, 0.8, 0.12, 0.03, 0.4])
    xm = x * np.array([1, -1, -1, 1, -1, -1, -1, 1])
    st, stm = make_state(x), make_state(xm)
    stm = stm._replace(prev_action=jnp.array([-0.1, 0.2]), accel=jnp.array([1.0, 2.0]))
    p = params_from_context(st.ctx)
    o, om = np.asarray(observe(st, ta, p)), np.asarray(observe(stm, tm, p))

    pre, prem = o[:200].reshape(40, 5), om[:200].reshape(40, 5)
    # Left edge of the mirror = mirrored right edge of the original, and vice versa.
    assert np.allclose(prem[:, 0], pre[:, 2], atol=1e-6)
    assert np.allclose(prem[:, 1], -pre[:, 3], atol=1e-6)
    assert np.allclose(prem[:, 2], pre[:, 0], atol=1e-6)
    assert np.allclose(prem[:, 3], -pre[:, 1], atol=1e-6)
    assert np.allclose(prem[:, 4], -pre[:, 4], atol=1e-6)
    ego, egom = o[200:216], om[200:216]
    flip = np.array([1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1])
    assert np.allclose(egom, flip * ego, atol=1e-6)
    assert np.allclose(o[216:], om[216:])


def test_progress_reward_on_a_straight():
    n = 2001
    xy = np.stack([np.linspace(0, 4000, n), np.zeros(n)], axis=1)
    w = np.full(n, 6.0)
    tr = from_centreline("straight", xy, w, w, ds=2.0, smoothing=0.0, closed=False)
    ta = track_arrays(tr, dtype=jnp.float64)
    st = make_state([100.0, 0, 0, 50.0, 0, 0, 0, 0.0])._replace(prev_action=jnp.zeros(2))
    _, new, r, term, trunc, info = step_no_reset(st, jnp.array([0.0, 0.0]), ta)
    assert not bool(term) and not bool(trunc)
    assert float(info["progress"]) == pytest.approx(50.0 * 0.05, rel=0.01)
    assert float(r) == pytest.approx(float(info["progress"]) / 10.0, rel=1e-6)


def test_leaving_the_corridor_terminates_with_speed_scaled_penalty():
    ta = track_arrays(wavy_track(), dtype=jnp.float64)
    st = make_state([500.0, 5.8, 0.6, 60.0, 0, 0, 0, 0.0])
    _, _, r, term, _, info = step_no_reset(st, jnp.array([0.0, 0.0]), ta)
    assert bool(term) and bool(info["off_track"])
    assert float(r) < -5.0 - 5.0 * (55.0 / 50.0) ** 2 + 1.0


def test_auto_reset_and_truncation():
    ta = track_arrays(wavy_track(), dtype=jnp.float64)
    v_ref = jnp.full(ta.x.shape, 40.0)
    params = EnvParams(window_steps=3)
    obs, st = reset(jax.random.PRNGKey(0), ta, v_ref, default_context(), params)
    flags = []
    for i in range(3):
        obs, st, r, term, trunc, info = step(jax.random.PRNGKey(i + 1), st, jnp.array([0.0, 0.1]),
                                             ta, v_ref, params)
        flags.append(bool(trunc))
    assert flags == [False, False, True]
    assert int(st.t) == 0                      # reset happened
    assert info["final_obs"].shape == (OBS_DIM,)


def test_closed_track_wraps_past_the_finish_line():
    tr = wavy_track()
    ta = track_arrays(tr, dtype=jnp.float64)
    st = make_state([tr.length - 1.0, 0, 0, 50.0, 0, 0, 0, 0.0])
    p = params_from_context(st.ctx)
    obs = observe(st, ta, p)
    _, new, _, term, _, _ = step_no_reset(st, jnp.array([0.0, 0.0]), ta)
    assert bool(jnp.all(jnp.isfinite(obs)))
    assert float(new.x[0]) > tr.length
