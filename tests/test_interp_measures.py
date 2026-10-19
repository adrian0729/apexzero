"""Protocol 5 measures on synthetic trajectories with known answers (step 3), the brake-demand
re-implementation, the model tables and the trajectory adapters. No agent is rolled out."""

import jax
import numpy as np
import pytest

from apexzero.interp import measures as M
from apexzero.interp.constants import FLAG_BITS, KPH, N_KNOTS
from apexzero.interp.stimuli import Landmarks
from apexzero.vehicle import dynamics
from apexzero.vehicle.params import F1_2025 as P


@pytest.fixture(scope="module", autouse=True)
def _float32():
    """Rollouts, controls and stimuli arrays run in float32. tests/test_dynamics.py and
    tests/test_env.py switch x64 on globally when they are imported, so in a full-suite run
    this module switches it off for its own tests and restores it afterwards."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


DT = 0.05
LM = Landmarks(s_c0=760.0, s_a0=790.0, s_apex=820.0, s_a1=850.0, s_c1=880.0)
DRAG = 0.5 * P.rho * P.cda / P.m          # drag / m = DRAG * vx^2


def bit(*names):
    return sum(1 << FLAG_BITS[n] for n in names)


def marks(direction=1, s0=300.0, W=400, wl=6.0, wr=6.0, lm=LM, triangle=False):
    return M.Marks(lm, direction, s0, W, np.full(N_KNOTS, wl), np.full(N_KNOTS, wr), triangle)


def traj(s, vx, *, n=None, vy=None, r=None, pedal=None, beta=None, util_f=None, util_r=None,
         a=None, a_mean=None, reward=None, term_step=-1, flags=None, prev0=None,
         clip_frac=None, clip_max=None, valid=None, dt=DT):
    s, vx = np.asarray(s, float), np.asarray(vx, float)
    K = len(s)
    x = np.zeros((K, 8))
    x[:, 0], x[:, 3] = s, vx
    for col, arr in ((1, n), (4, vy), (5, r), (7, pedal)):
        if arr is not None:
            x[:, col] = arr
    half = np.full(K, 0.5)
    return M.Trajectory(
        source="agent", t=np.arange(K) * dt, x=x,
        valid=np.ones(K, bool) if valid is None else valid, a=a, a_mean=a_mean, reward=reward,
        util_f=half if util_f is None else np.asarray(util_f, float),
        util_r=half if util_r is None else np.asarray(util_r, float),
        beta_f=np.zeros(K) if beta is None else np.asarray(beta, float),
        flags=np.zeros(K - 1, np.uint8) if flags is None else flags, term_step=term_step,
        clip_frac=clip_frac, clip_max=clip_max, dt=dt,
        prev_action0=np.zeros(2) if prev0 is None else np.asarray(prev0, float))


def integrate(v, s0, dt=DT):
    """Exact positions of a speed profile that is linear between states."""
    return s0 + np.r_[0.0, np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)]


# ------------------------------------------------------------------------- brake demand


def random_states(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 8))
    x[:, 3] = rng.uniform(-0.5, 90.0, n)
    x[:, 4] = rng.uniform(-3.0, 3.0, n)
    x[:, 5] = rng.uniform(-1.5, 1.5, n)
    x[:, 6] = rng.uniform(-0.45, 0.45, n)
    x[:, 7] = rng.uniform(-1.0, 1.0, n)
    x[:8, 7] = [-1.0, -0.02, -0.01, 0.0, 0.01, 0.02, 1.0, -0.3]   # relu kinks
    return x


def test_front_brake_demand_reproduces_tyre_forces():
    x = random_states()
    for p in (P, M.car_params(P, None, 1.03), P.replace(m=760.0, cla_f=2.6, mu_scale=0.9)):
        beta, fx_dem, fx_max = M.front_brake_demand(x, p, M.NP)
        fo = dynamics.tyre_forces(x, p, M.NP)
        scale = 1.0 / M.NP.smax(fo.util_f, 1.0, 5e-3)
        np.testing.assert_allclose(fx_dem * scale, fo.fx_f, rtol=0, atol=1e-10)
        assert np.all(beta >= 0) and np.all(beta[fx_dem >= 0] == 0)
    # Straight running: no lateral force, so util_f^2 = beta_f^2 + 1e-8 exactly.
    xs = x.copy()
    xs[:, 4:7] = 0.0
    xs[:, 3] = np.abs(xs[:, 3]) + 0.5
    beta, fx_dem, fx_max = M.front_brake_demand(xs, P, M.NP)
    fo = dynamics.tyre_forces(xs, P, M.NP)
    np.testing.assert_allclose(beta, np.sqrt(fo.util_f**2 - 1e-8), atol=1e-10)


def test_front_brake_demand_jax_matches_numpy():
    import jax
    import jax.numpy as jnp

    from apexzero.vehicle.ops import JAX

    x = random_states(500, seed=1)
    bj = jax.jit(jax.vmap(lambda xi: M.front_brake_demand(xi, P, JAX)[0]))(jnp.asarray(x, jnp.float32))
    bn = M.front_brake_demand(x, P, M.NP)[0]
    np.testing.assert_allclose(np.asarray(bj), bn, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("kph,pedal", [(310, 0.70), (200, 0.46), (100, 0.30), (60, 0.26)])
def test_front_axle_saturates_at_protocol_pedals(kph, pedal):
    """Protocol 1.2: the pedal that saturates the front axle in straight braking."""
    ped = np.linspace(-1, 0, 2001)
    x = np.zeros((len(ped), 8))
    x[:, 3], x[:, 7] = kph * KPH, ped
    beta = M.front_brake_demand(x, P, M.NP)[0]
    sat = -ped[np.argmin(np.abs(beta - 1.0))]
    assert sat == pytest.approx(pedal, abs=0.015)          # protocol values are rounded


# ------------------------------------------------------------------------- model tables


@pytest.fixture(scope="module")
def tables():
    return M.build_model_tables(speeds_kph=np.array([40.0, 150.0, 280.0]))


def test_a_dec_table(tables):
    assert len(tables.v_dec) == 316 and tables.v_dec[-1] == pytest.approx(315 * KPH)
    assert tables.a_dec[0] == pytest.approx(0.0, abs=1e-12)
    assert 65.0 <= tables.a_dec_at(310 * KPH) <= 66.5          # protocol 1.2: 65.4 m/s^2
    assert np.all(np.diff(tables.a_dec[60:]) > 0)              # downforce: grows with speed
    # The table's value is the best pedal of the sweep at that speed.
    v = 200 * KPH
    ped = np.linspace(-1, 0, 201)
    dvx = M._derivs(v, 0.0, 0.0, 0.0, ped, P)[0]
    assert tables.a_dec_at(v) == pytest.approx(-dvx.min(), rel=1e-12)


def test_a_y_max_is_a_steady_state(tables):
    from apexzero.oracle.qss import GG

    assert list(tables.limit_y) == ["lock", "grip", "power"]
    for i, V in enumerate(tables.v_lat):
        b, r, d, ped = tables.beta_y[i], tables.r_y[i], tables.delta_y[i], tables.pedal_y[i]
        dvx, dvy, dr = M._derivs(V * np.cos(b), V * np.sin(b), r, d, ped, P)
        assert max(abs(dvx), abs(dvy), abs(dr)) < 1e-6
        assert tables.a_y_max[i] == pytest.approx(V * r)
        assert max(abs(tables.alpha_f_y[i]), abs(tables.alpha_r_y[i])) <= P.alpha_max
        assert 0.0 <= d <= P.delta_max and -1.0 <= ped <= 1.0
    assert tables.delta_y[0] == pytest.approx(P.delta_max)
    ratio = tables.a_y_max[1] / GG.from_params(P).a_lat_max(tables.v_lat[1])
    assert 0.88 <= ratio <= 0.96
    # No coarse-grid trim at 150 km/h beats the polished maximum.
    V = tables.v_lat[1]
    D, B, r, ped, dvy, ok = M._trim_grid(V, np.linspace(0.05, 0.2, 31), np.linspace(-0.1, 0.05, 61), P)
    e = np.where(ok, dvy, np.nan)
    cross = np.isfinite(e[:, :-1]) & np.isfinite(e[:, 1:]) & (np.sign(e[:, :-1]) != np.sign(e[:, 1:]))
    w = np.where(cross, e[:, :-1] / np.where(cross, e[:, :-1] - e[:, 1:], 1), 0)
    rc = r[:, :-1] + w * (r[:, 1:] - r[:, :-1])
    assert np.max(np.where(cross, V * rc, -np.inf)) <= tables.a_y_max[1] * (1 + 2e-4)


def test_table_functions_known_answers(tmp_path, tables):
    v = np.arange(0, 316) * KPH
    lat = np.arange(20, 316, 5) * KPH
    const = M.ModelTables(v_dec=v, a_dec=np.full_like(v, 40.0), pedal_dec=-np.ones_like(v),
                          v_lat=lat, a_y_max=np.full_like(lat, 20.0), alpha_max=0.175)
    assert const.d_b_phys(80.0, 30.0) == pytest.approx((80**2 - 30**2) / 80.0, rel=1e-6)
    assert const.d_b_phys(30.0, 80.0) == 0.0
    assert np.isnan(const.d_b_phys(np.nan, 30.0))
    np.testing.assert_allclose(const.d_b_phys(np.array([50.0, 60.0]), 20.0),
                               (np.array([50.0, 60.0]) ** 2 - 400) / 80.0, rtol=1e-6)
    rho = np.array([30.0, 100.0, 200.0])
    np.testing.assert_allclose(const.v_ss(rho), np.sqrt(20.0 * rho), rtol=1e-12)
    assert np.isnan(const.v_ss(-1.0)) and const.v_ss(np.inf) == pytest.approx(lat[-1])
    assert const.v_ss(1e6) == pytest.approx(lat[-1])            # never short of grip
    lin = M.ModelTables(v_dec=v, a_dec=np.full_like(v, 40.0), pedal_dec=-np.ones_like(v),
                        v_lat=lat, a_y_max=0.4 * lat, alpha_max=0.175)
    assert lin.v_ss(50.0) == pytest.approx(0.4 * 50.0)           # v^2 / rho = 0.4 v
    # Real tables: v_ss inverts a_y,max inside the grid.
    rho_star = 40.0
    vs = tables.v_ss(rho_star)
    assert vs**2 / rho_star == pytest.approx(tables.a_y_at(vs), rel=1e-9)
    tables.save(tmp_path / "t.npz")
    back = M.ModelTables.load(tmp_path / "t.npz")
    np.testing.assert_array_equal(back.a_y_max, tables.a_y_max)
    assert list(back.limit_y) == list(tables.limit_y) and back.params == tables.params
    assert back.d_b_phys(80.0, 20.0) == tables.d_b_phys(80.0, 20.0)


# ------------------------------------------------------------------------- onsets


def test_brake_onset_blips_release_and_apex_rule():
    K = 80
    s = 500.0 + 4.0 * np.arange(K)
    t = np.arange(K) * DT
    vx = 80.0 - 0.1 * np.arange(K)
    beta = np.zeros(K)
    beta[5:7] = 0.9            # 2 states: not sustained, ignored
    beta[10:13] = 0.6          # 3 states: a blip
    beta[20:30] = 1.0          # the onset run
    beta[30] = 0.0
    beta[60:64] = 1.0          # begins after s_apex: ignored
    s_apex = s[50]
    bo = M.brake_onset(s, t, beta, np.ones(K, bool), s_apex, vx)
    w_blip = 0.3 / 0.6
    assert bo.s_b == pytest.approx(s[19] + 0.3 * 4.0)
    assert bo.k_b == 20 and not bo.forced and bo.k_r == 30
    assert bo.v_b == pytest.approx(vx[19] + 0.3 * (vx[20] - vx[19]))
    assert bo.t_b == pytest.approx(t[19] + 0.3 * DT)
    assert len(bo.blips) == 1
    assert bo.blips[0][0] == pytest.approx(s[9] + w_blip * 4.0)
    assert bo.blips[0][1] == pytest.approx((vx[9] + w_blip * (vx[10] - vx[9])) - vx[13])
    assert bo.s_r == pytest.approx(s[29] + 0.9 * 4.0)           # 0.1 crossing of 1 -> 0
    # Onset within 0.5 s of the start is forced; invalid states are ignored.
    beta2 = np.zeros(K)
    beta2[3:10] = 1.0
    assert M.brake_onset(s, t, beta2, np.ones(K, bool), s_apex, vx).forced
    valid = np.ones(K, bool)
    valid[21:] = False
    assert M.brake_onset(s, t, beta, valid, s_apex, vx).s_b == pytest.approx(bo.blips[0][0])


def test_response_onset_alignment_and_spikes():
    K = 100
    s = 300.0 + 4.0 * np.arange(K)
    pedal = np.ones(K)
    pedal[20] = 0.7                       # single-state spike: ignored
    pedal[30:34] = 0.6                    # the null does the same at the same distance
    pedal[60] = 0.95
    pedal[61:] = 0.3
    s_null = 1400.0 + 4.0 * np.arange(K)  # different start: aligned by distance travelled
    pn = np.ones(K)
    pn[30:34] = 0.6
    got = M.response_onset(s, pedal, 300.0, s_null, pn, 1400.0)
    assert got == pytest.approx(s[60] + (0.1 - 0.05) / (0.7 - 0.05) * 4.0)
    # Not covered by the null beyond its end: NaN differences cannot form a response.
    assert np.isnan(M.response_onset(s, pedal, 300.0, s_null[:40], pn[:40], 1400.0))


# ------------------------------------------------------------------------- response_metrics


def braking_run():
    """80 m/s, constant 40 m/s^2 braking from t = 5 s (s = 700) to 30 m/s, s_c0 at a state
    during the braking (v_e = 40 m/s, d_b = 60 m, so abar = 40 exactly)."""
    K, kb, ke = 401, 100, 125
    k = np.arange(K)
    vx = np.where(k <= kb, 80.0, np.maximum(80.0 - 40.0 * (k - kb) * DT, 30.0))
    s = integrate(vx, 300.0)
    assert s[kb] == pytest.approx(700.0) and s[kb + 20] == pytest.approx(760.0)
    beta = np.zeros(K)
    beta[kb] = 0.3
    beta[kb + 1: ke] = 1.0
    pedal = np.where(k < kb, 1.0, np.where(k < ke, -0.6, 0.3))
    pedal[kb - 2], pedal[kb - 1], pedal[kb] = 0.95, 0.5, -0.2
    pedal[50] = 0.7
    pedal[30:34] = 0.6
    a = np.zeros((K - 1, 2))
    a[:, 1] = np.where(k[:-1] < kb - 1, 1.0, np.where(k[:-1] < ke, -0.6, 0.3))
    a_mean = a.copy()
    a_mean[:100, 0] = 0.01 * (-1.0) ** np.arange(100)
    prev0 = np.array([0.0, 0.5])
    prev = np.vstack([prev0, a[:-1]])
    reward = np.diff(s) / 10 - 0.02 * np.sum((a - prev) ** 2, 1)
    cf = np.zeros((K, 4))
    cm = np.full((K, 4), 2.0)
    cf[150, 0], cm[150, 0] = 0.05, 10.0        # a kappa channel clipped before s_c1
    cf[350, 2], cm[350, 2] = 0.1, 10.0         # ego clipped after the corner
    return traj(s, vx, pedal=pedal, beta=beta, a=a, a_mean=a_mean, reward=reward,
                prev0=prev0, clip_frac=cf, clip_max=cm), (kb, ke)


def null_run():
    K = 401
    vx = np.full(K, 80.0)
    pedal = np.ones(K)
    pedal[30:34] = 0.6
    return traj(integrate(vx, 1400.0), vx, pedal=pedal)


def test_constant_deceleration_known_answers():
    tr, (kb, ke) = braking_run()
    s = tr.s
    mk = marks()
    m = M.response_metrics(tr, mk, null=null_run())
    assert m.has_brake and not m.forced_onset and m.n_blips == 0
    assert m.s_b == pytest.approx(700.0) and m.d_b == pytest.approx(60.0)
    assert m.v_b == pytest.approx(80.0) and m.t_b == pytest.approx(5.0)
    assert m.v_e == pytest.approx(40.0) and m.abar == pytest.approx(40.0, rel=1e-12)
    assert m.s_r == pytest.approx(s[ke - 1] + 0.9 * (s[ke] - s[ke - 1]))
    assert m.trail == pytest.approx(m.s_r - 760.0)
    assert m.decel_peak == pytest.approx(40.0 - DRAG * 32.0**2)
    v_r = 80.0 - 40.0 * (1.2 + 0.9 * DT)
    exact_drag = DRAG * (80.0**2 + 80.0 * v_r + v_r**2) / 3.0
    assert m.decel_mean == pytest.approx(40.0 - exact_drag, abs=5e-3)
    assert m.s_legacy == pytest.approx(700.0) and m.d_legacy == pytest.approx(60.0)
    assert m.s_dec == pytest.approx(s[kb - 2] + (1 / 1.6) * (s[kb - 1] - s[kb - 2]))
    assert m.s_resp == pytest.approx(s[kb - 2] + (0.05 / 0.45) * (s[kb - 1] - s[kb - 2]))
    assert m.d_resp == pytest.approx(760.0 - m.s_resp)
    assert m.v_peak == 80.0 and m.v_min == 30.0
    assert m.s_vmin_rel == pytest.approx(s[ke] - LM.s_apex)
    assert m.v_arc == 30.0 and m.v_exit100 == 30.0
    assert m.t_c1 == pytest.approx(tr.t[ke] + (880.0 - s[ke]) / 30.0)
    assert m.completed and not m.crashed and not m.time_censored
    assert m.sight_at_start and m.k_sight == 0 and not m.unsettled
    gamma = 0.995 ** np.arange(400)
    r0 = tr.reward.copy()
    r0[0] = (s[1] - s[0]) / 10 - 0.02 * np.sum(tr.a[0] ** 2)    # a_{-1} = 0 for regret
    assert m.J == pytest.approx(np.sum(gamma * r0), rel=1e-12)
    assert m.kappa_clipped and m.clip_max_kappa == 10.0 and m.clip_max_ego == 10.0
    assert m.clip_frac_ego == pytest.approx(0.1 / 401)
    assert m.osc_index == pytest.approx(0.02)
    assert m.util_f_peak == 0.5 and m.eta_apex == 0.0


def test_flat_out_run_has_no_brake_and_unsettled_sight():
    K = 401
    vx = np.full(K, 80.0)
    s = integrate(vx, 100.0)
    pedal = np.ones(K)
    pedal[45] = 0.5
    m = M.response_metrics(traj(s, vx, pedal=pedal), marks(s0=100.0))
    assert not m.has_brake and np.isnan(m.s_b) and np.isnan(m.d_b) and np.isnan(m.s_dec)
    assert m.v_peak == 80.0 and m.v_e == 80.0 and np.isnan(m.s_resp)
    # Corner start enters the 480 m preview when s >= 280: state 45, at pedal 0.5.
    assert m.k_sight == 45 and m.d_sight == pytest.approx(180.0) and m.unsettled
    assert M.first_sight(s, vx, 760.0) == 45
    assert M.first_sight(np.array([0.0]), np.array([10.0]), 760.0) == -1


def test_crash_censors_window_measures():
    K = 401
    vx = np.full(K, 60.0)
    s = integrate(vx, 300.0)
    kt = int(np.argmax(s >= 800.0))                  # first state in the corner entry
    valid = np.arange(K) <= kt
    x_garbage = s.copy()
    x_garbage[kt + 1:] = 99999.0
    vxg = vx.copy()
    vxg[kt + 1:] = 999.0
    n = np.zeros(K)
    n[kt] = 5.2
    flags = np.zeros(K - 1, np.uint8)
    flags[kt - 1] = bit("off_track", "spun")
    reward = np.full(K - 1, 0.3)
    reward[kt - 1] = -20.0
    reward[kt:] = 123.0                              # beyond the crash: never summed
    tr = traj(x_garbage, vxg, n=n, flags=flags, term_step=kt - 1, valid=valid, reward=reward)
    m = M.response_metrics(tr, marks())
    assert m.crashed and m.cause == "corridor" and m.flags_fired == bit("off_track", "spun")
    assert m.crash_side == 1 and m.crash_phase == "entry"
    assert m.s_crash == pytest.approx(s[kt]) and m.v_crash == 60.0 and m.t_crash == kt * DT
    assert m.s_end == pytest.approx(s[kt]) and not m.completed and not m.time_censored
    assert m.v_e == 60.0
    assert m.v_arc == 60.0                           # s_a0 = 790 was reached before the crash
    for name in ("v_min", "util_f_peak", "util_r_peak", "rho_min", "clearance", "v_exit100",
                 "t_c1", "eta_post", "eta_apex"):
        assert np.isnan(getattr(m, name)), name
    assert m.J == pytest.approx(np.sum(0.995 ** np.arange(kt) * reward[:kt]))
    # Right-hand corner: the same positive n is the outside edge.
    assert M.response_metrics(tr, marks(direction=-1)).crash_side == -1
    # A run that is alive at W short of s_c1 + 250 m is time-censored.
    slow = np.full(K, 5.0)
    m2 = M.response_metrics(traj(integrate(slow, 300.0), slow), marks())
    assert m2.time_censored and not m2.completed and not m2.crashed


def test_line_eta_both_sides_and_line_apex():
    s = np.full(4, 820.0)
    left, right = marks(direction=1, wl=5.0, wr=7.0), marks(direction=-1, wl=5.0, wr=7.0)
    n = np.array([2.0, -3.0, 4.0, -6.0])
    np.testing.assert_allclose(M.line_eta(s, n, left), [0.5, -0.5, 1.0, -1.0])
    np.testing.assert_allclose(M.line_eta(s, -n, right), [2 / 6, -3 / 4, 4 / 6, -6 / 4])
    K = 401
    vx = np.full(K, 20.0)
    ss = integrate(vx, 500.0)
    nn = 3.0 - ((ss - (LM.s_apex + 5.0)) / 40.0) ** 2
    m = M.response_metrics(traj(ss, vx, n=nn), left)
    assert m.line_apex_rel == pytest.approx(5.0, abs=1.0)
    assert m.clearance == pytest.approx(0.25, abs=1e-3)
    assert m.eta_apex == pytest.approx((3.0 - (5.0 / 40.0) ** 2) / 4.0)
    assert m.eta_pre == pytest.approx((3.0 - (95.0 / 40.0) ** 2) / 6.0)   # outside: w_out - 1
    # Mirror image on a right-hander gives the same eta.
    mr = M.response_metrics(traj(ss, vx, n=-nn), marks(direction=-1, wl=7.0, wr=5.0))
    assert mr.eta_apex == pytest.approx(m.eta_apex) and mr.clearance == pytest.approx(m.clearance)


def test_path_curvature_and_rho_min():
    K = 401
    V, rho, b0 = 20.0, 50.0, 0.05
    t = np.arange(K) * DT
    x = np.zeros((K, 8))
    x[:, 3], x[:, 4], x[:, 5] = V * np.cos(b0), V * np.sin(b0), V / rho
    np.testing.assert_allclose(M.path_curvature(x, t), 1 / rho, rtol=1e-12)
    x2 = x.copy()
    beta = b0 + 0.01 * t
    x2[:, 3], x2[:, 4] = V * np.cos(beta), V * np.sin(beta)
    np.testing.assert_allclose(M.path_curvature(x2, t), (V / rho + 0.01) / V, rtol=1e-9)
    s = integrate(np.full(K, V), 600.0)
    tr = traj(s, x[:, 3], vy=x[:, 4], r=x[:, 5])
    for direction, expect in ((1, rho), (-1, np.inf)):
        m = M.response_metrics(tr, marks(direction=direction))
        assert m.rho_min == pytest.approx(expect)
    tab = M.ModelTables(v_dec=np.arange(316) * KPH, a_dec=np.full(316, 40.0),
                        pedal_dec=-np.ones(316), v_lat=np.arange(20, 316, 5) * KPH,
                        a_y_max=np.full(60, 20.0), alpha_max=0.175)
    m = M.response_metrics(tr, marks(), tables=tab)
    assert m.e_vmin_line == pytest.approx(V * np.cos(b0) / np.sqrt(20.0 * rho) - 1.0)


# ------------------------------------------------------------------------- crashes


def crash_run(v=60.0, util_f=None, util_r=None, beta=None):
    K = 401
    vx = np.full(K, v)
    s = integrate(vx, 300.0)
    kt = int(np.argmax(s >= 830.0))
    flags = np.zeros(K - 1, np.uint8)
    flags[kt - 1] = bit("slipped", "spun")
    valid = np.arange(K) <= kt
    uf = np.full(K, 0.5) if util_f is None else util_f
    return traj(s, vx, util_f=uf, util_r=util_r, beta=beta, flags=flags, term_step=kt - 1,
                valid=valid), kt


def test_crash_type_rules():
    K = 401
    key_v = np.full(K, 50.0)
    key = traj(integrate(key_v, 300.0), key_v)
    mk = marks()
    tr, kt = crash_run()
    assert M.response_metrics(tr, mk).cause == "slip"
    ur = np.full(K, 0.5)
    ur[kt - 10: kt - 6] = 0.99                         # 4 saturated steps, beta_f < 1
    sat, _ = crash_run(util_r=ur)
    assert M.crash_type(sat, mk, key, None) == "overspeed"
    unresolved = M.NoReturn(s=None, certified=False, dv=np.nan)
    assert M.crash_type(sat, mk, key, unresolved) == "overspeed"
    slower = traj(sat.s, np.full(K, 70.0))            # key faster: dv < 0
    assert M.crash_type(sat, mk, slower, None) == "other"
    # A resolved point of no return decides; the ellipse rule is then not used.
    after = M.NoReturn(s=None, certified=True, dv=np.nan, s_last_viable=LM.s_a0)
    assert M.crash_type(sat, mk, key, after) == "other"
    late = M.NoReturn(s=LM.s_a0 + 10.0, certified=True, dv=3.0)
    assert M.crash_type(sat, mk, key, late) == "other"
    early = M.NoReturn(s=LM.s_c0, certified=True, dv=3.0)
    plain, _ = crash_run()
    assert M.crash_type(plain, mk, key, early) == "overspeed"
    assert M.crash_type(plain, mk, key, M.NoReturn(LM.s_c0, True, -1.0)) == "line_steer"
    # Over-brake: front locked under brake demand in the last second, slower than the key.
    kt45 = crash_run(v=45.0)[1]
    uf = np.full(K, 0.5)
    bf = np.zeros(K)
    uf[kt45 - 5: kt45 - 1] = 0.99
    bf[kt45 - 5: kt45 - 1] = 1.2
    ob, _ = crash_run(v=45.0, util_f=uf, beta=bf)
    assert M.crash_type(ob, mk, key, None) == "overbrake"
    assert M.crash_type(ob, mk, None, None) == "other"           # no key: dv unknown
    near = np.full(K, 0.95)
    assert M.crash_type(crash_run(util_f=near, util_r=near)[0], mk, key, None) == "other"
    assert M.crash_type(plain, mk, key, None) == "line_steer"
    alive = traj(key.s, key_v)
    assert M.crash_type(alive, mk, key, None) == ""


def test_crash_cause_precedence():
    assert M.crash_cause(bit("off_track", "slipped", "spun")) == "corridor"
    assert M.crash_cause(bit("slipped", "spun")) == "slip"
    assert M.crash_cause(bit("spun")) == "spin"
    assert M.crash_cause(bit("stalled")) == "stall"
    assert M.crash_cause(bit("nonfinite")) == "nonfinite"
    assert M.crash_cause(bit("truncated")) == "" and M.crash_cause(0) == ""


def test_oscillation_index():
    assert M.oscillation_index(np.zeros(200)) == 0.0
    sm = np.r_[0.1 * (-1.0) ** np.arange(100), np.full(100, 5.0)]
    assert M.oscillation_index(sm) == pytest.approx(0.2)
    assert np.isnan(M.oscillation_index(np.array([0.3])))


# ------------------------------------------------------------------------- errors vs keys


def piecewise(v1, v2, t_switch, s0, K=401):
    t = np.arange(K) * DT
    vx = np.where(t <= t_switch + 1e-9, v1, v2)
    s = s0 + np.where(t <= t_switch + 1e-9, v1 * t, v1 * t_switch + v2 * (t - t_switch))
    return s, vx


def test_key_errors_corner_time_and_regret():
    W = 400
    s_k, v_k = piecewise(60.0, 86.0, 10.0, 300.0)
    s_r, v_r = piecewise(55.0, 86.0, 11.0, 302.0)
    key = traj(s_k, v_k, reward=np.full(W, 0.2))
    run = traj(s_r, v_r, n=np.full(W + 1, 0.5), reward=np.full(W, 0.1))
    vn = np.full(W + 1, 86.0)
    null = traj(integrate(vn, 400.0), vn, reward=np.full(W, 0.3))
    null_key = traj(integrate(vn, 400.0), vn, reward=np.full(W, 0.35))
    mk = marks()
    m, mkey = M.response_metrics(run, mk), M.response_metrics(key, mk)
    e = M.key_errors(m, mkey, run, key, mk, null, null_key, socp=key, null_socp=null_key)
    assert e.e_ve == pytest.approx(55 / 60 - 1) and e.e_vmin == pytest.approx(55 / 60 - 1)
    assert e.rms_dn == pytest.approx(0.5)
    geo = np.sum(0.995 ** np.arange(W))
    assert e.regret == pytest.approx(0.1 * geo) and e.corner_regret == pytest.approx(0.05 * geo)
    # s_ff = 912 m: run at 86 m/s from s = 907 (t = 11 s), key from s = 900 (t = 10 s);
    # the nulls add back the run's 2 m head start.
    expect = (11 + 5 / 86) - (10 + 12 / 86) + 2 / 86
    assert e.dt_corner == pytest.approx(expect) and not e.dt_flag
    assert e.dt_corner_socp == pytest.approx(expect)
    assert np.isnan(e.dv_crash) and np.isnan(e.dd_b)
    # Not at the key's phase: no regret. No nulls: no corner time.
    e2 = M.key_errors(m, mkey, run, key, mk, key_at_phase=False)
    assert np.isnan(e2.regret) and np.isnan(e2.dt_corner)
    # Neither reaches 97% of top speed within W: last common s, flagged.
    slow_r, slow_k = traj(*piecewise(55.0, 70.0, 11.0, 302.0)), traj(*piecewise(60.0, 70.0, 10.0, 300.0))
    dt, flag = M.corner_time_lost(slow_r, slow_k, mk, null, null_key)
    assert flag and np.isfinite(dt)
    grid, dv = M.delta_v(run, key, mk)
    assert grid[0] == 460.0 and grid[-1] == 980.0 and len(grid) == 521
    assert np.all(dv[grid < 899] == pytest.approx(-5.0))
    # Crashed run: dv at the crash point, NaN beyond it.
    tr, kt = crash_run(v=60.0)
    mc = M.response_metrics(tr, mk)
    ec = M.key_errors(mc, mkey, tr, key, mk, null, null_key)
    assert ec.dv_crash == pytest.approx(0.0) and np.isnan(ec.dt_corner)
    grid, dv = M.delta_v(tr, key, mk)
    assert np.all(np.isnan(dv[grid > mc.s_crash])) and np.isfinite(dv[grid < 800]).all()


# ------------------------------------------------------------------------- adapters


def fake_rollout(T=6):
    R = 2
    rng = np.random.default_rng(3)
    d = {
        "det_W": np.array([6, 4], np.int16), "det_cell": np.array([7, 8], np.int32),
        "det_stim": np.array([1, 2], np.int32), "det_pair": np.array([3, 3], np.int32),
        "det_phase": np.array([0, 1], np.int16), "det_sample": np.array([-1, -1], np.int32),
        "det_steer_sign": np.array([1, -1], np.int8), "det_s0": np.array([400.0, 401.0]),
        "det_term_step": np.array([-1, 2], np.int32), "det_trunc_step": np.array([-1, -1], np.int32),
        "det_valid": np.ones((R, T + 1), bool),
        "det_flags": np.zeros((R, T), np.uint8),
        "det_x": rng.normal(size=(R, T + 1, 8)).astype(np.float32),
        "det_prev_action": np.zeros((R, T + 1, 2), np.float32),
        "det_accel": np.zeros((R, T + 1, 2), np.float32),
        "det_a_mean": rng.normal(size=(R, T, 2)).astype(np.float32),
        "det_a": 3 * rng.normal(size=(R, T, 2)).astype(np.float32),
        "det_reward": rng.normal(size=(R, T)).astype(np.float32),
        "det_clip_frac": np.zeros((R, T + 1, 4), np.float32),
        "det_clip_max": np.zeros((R, T + 1, 4), np.float32),
        "log_std": np.array([-3.0, -2.5], np.float32),
    }
    for k in ("util_f", "util_r", "beta_f"):
        d[f"det_{k}"] = rng.uniform(size=(R, T + 1)).astype(np.float32)
    d["det_valid"][1, 4:] = False
    d["det_flags"][1, 2] = bit("off_track")
    d["det_prev_action"][1, 0] = [0.1, -0.2]
    for k in ("W", "cell", "stim", "pair", "phase", "sample", "steer_sign", "s0",
              "term_step", "trunc_step", "valid", "flags", "util_f", "util_r", "beta_f"):
        d[f"nat_{k}"] = d[f"det_{k}"]
    d["nat_s"] = (400.0 + 4.0 * np.arange(T + 1))[None].repeat(R, 0).astype(np.float32)
    d["nat_vx"] = np.full((R, T + 1), 80.0, np.float32)
    d["nat_n"] = np.zeros((R, T + 1), np.float16)
    d["nat_pedal"] = np.ones((R, T + 1), np.float16)
    d["nat_xi"] = np.full((R, T + 1), 0.01, np.float16)
    return d


def test_trajectory_from_rollout_det_and_stochastic():
    d = fake_rollout()
    t0 = M.trajectory_from_rollout(d, 0, "det")
    assert t0.x.shape == (7, 8) and t0.a.shape == (6, 2) and t0.x.dtype == np.float64
    assert np.all(np.abs(t0.a) <= 1.0) and np.abs(d["det_a"][0]).max() > 1.0
    np.testing.assert_array_equal(t0.a_mean, d["det_a_mean"][0].astype(np.float64))
    assert t0.term_step == -1 and t0.meta["cell"] == 7 and t0.log_std.tolist() == [-3.0, -2.5]
    t1 = M.trajectory_from_rollout(d, 1, "det")
    assert t1.x.shape == (5, 8) and t1.term_step == 2 and t1.flags[2] == 1
    assert t1.valid.tolist() == [True] * 4 + [False]
    np.testing.assert_allclose(t1.prev_action0, [0.1, -0.2], rtol=1e-6)
    grp = {k[4:]: v for k, v in d.items() if k.startswith("det_")}   # load_rollouts layout
    assert np.array_equal(M.trajectory_from_rollout(grp, 1).x, t1.x)
    st = M.trajectory_from_rollout(d, 0, "nat")
    assert np.isnan(st.x[:, [4, 5, 6]]).all() and st.x[3, 0] == pytest.approx(412.0)
    assert st.x[0, 2] == pytest.approx(0.01, rel=1e-3)
    assert st.a is None and st.reward is None and st.beta_f is not None
    m = M.response_metrics(st, marks(s0=400.0, W=6))
    assert np.isnan(m.v_e_abs) and np.isnan(m.J) and m.n_sight == 0.0


def straight_track(n=1200, w=6.0):
    from apexzero.tracks.track import Track

    s = 2.0 * np.arange(n)
    z = np.zeros(n)
    return Track(name="straight", s=s, x=s.copy(), y=z.copy(), theta=z.copy(), kappa=z.copy(),
                 w_left=np.full(n, w), w_right=np.full(n, w), closed=False)


def test_replay_and_window_adapters_agree():
    import jax

    from apexzero.ocp.window import WindowResult, WindowSpec, window_return

    track = straight_track()
    T = 160
    k = np.arange(T)
    acts = np.zeros((T, 2))
    acts[:, 0] = 0.004 * np.sin(k / 3.0)
    acts[:, 1] = np.where(k < 40, 1.0, np.where(k < 70, -0.6, 0.4))
    x0 = np.array([300.0, 0.0, 0.0, 50.0, 0, 0, 0, 0])
    before = jax.config.jax_enable_x64
    rep = M.trajectory_from_replay(track, x0, acts)
    assert jax.config.jax_enable_x64 == before                    # the context restored it
    assert rep.term_step == -1 and rep.valid.all() and rep.x.dtype == np.float64
    spec = WindowSpec(steps=T)
    res = WindowResult(x=rep.x, a=acts, value=window_return(rep.x, acts, spec), success=True,
                       status="synthetic", iterations=0, solve_seconds=0.0, rounds=0,
                       meta={"spec": spec.__dict__})
    lm = Landmarks(800.0, 820.0, 840.0, 860.0, 880.0)
    mk = M.Marks(lm, 1, 300.0, T, track.w_left, track.w_right)
    out = M.check_window_replay(res, track, mk)
    assert out["pass"] and out["max_diff"] <= 1e-6 and out["survives"]
    assert out["J_replay"] == pytest.approx(res.value, rel=1e-12)
    mw = M.response_metrics(M.trajectory_from_window(res, track), mk)
    assert mw.has_brake and mw.J == pytest.approx(res.value, rel=1e-12)
    # A full-lock replay terminates: flags recorded, later states frozen and invalid.
    hard = acts.copy()
    hard[:, 0] = 1.0
    crash = M.trajectory_from_replay(track, x0, hard)
    kt = crash.term_step + 1
    assert crash.term_step >= 0 and crash.valid[:kt + 1].all() and not crash.valid[kt + 1:].any()
    assert np.all(crash.x[kt + 1:] == crash.x[kt]) and crash.flags[crash.term_step] != 0
    mc = M.response_metrics(crash, mk)
    assert mc.crashed and mc.cause in ("corridor", "slip", "spin")


def corner_strip(n=700, R=35.0, lr=20.0, sc=260.0):
    from apexzero.tracks.track import Track

    s = 2.0 * np.arange(n)
    Lp = (np.pi / 2) * R - lr
    u = s - sc
    prof = np.where(u < 0, 0.0, np.clip(np.minimum(u / lr, (2 * lr + Lp - u) / lr), 0.0, 1.0))
    kappa = prof / R
    theta = np.r_[0.0, np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * 2.0)]
    track = Track("corner", s, np.cumsum(np.cos(theta)) * 2.0, np.cumsum(np.sin(theta)) * 2.0,
                  theta, kappa, np.full(n, 6.0), np.full(n, 6.0), closed=False)
    lm = Landmarks(sc, sc + lr, sc + lr + Lp / 2, sc + lr + Lp, sc + 2 * lr + Lp)
    return track, lm


def test_genuine_window_solution_replays_to_the_same_measures():
    """Step 3 on a real (short, 24-step) TO-OCP solution: identical measures to 1e-6."""
    import jax
    import jax.numpy as jnp

    from apexzero.ocp.window import WindowSolver, WindowSpec
    from apexzero.vehicle.sim import control_step, track_arrays

    track, lm = corner_strip()
    T = 24
    x0 = np.array([225.0, 0, 0, 36.0, 0, 0, 0, 0])
    with jax.enable_x64(True):
        ta = track_arrays(track, dtype=jnp.float64)
        step = jax.jit(lambda x, a: control_step(x, a, ta, P))
        xs, acts = [x0], []
        for _ in range(T):
            x = xs[-1]
            kap = np.interp(x[0] + 8.0, track.s, track.kappa)
            acts.append(np.array([np.clip(kap * P.wheelbase / P.delta_max - 0.2 * x[1] - x[2], -1, 1), 0.15]))
            xs.append(np.asarray(step(jnp.asarray(x), jnp.asarray(acts[-1]))))
    res = WindowSolver(WindowSpec(steps=T, radius=30.0), compile=False).solve(
        track, x0, init=(np.array(xs), np.array(acts)))
    assert res.success
    out = M.check_window_replay(res, track, M.Marks(lm, 1, 225.0, T, track.w_left, track.w_right))
    assert out["pass"] and out["survives"] and out["max_diff"] < 1e-9
    assert out["J_window"] == pytest.approx(res.value, rel=1e-12)


def test_socp_adapter():
    from apexzero.ocp.min_time import OCPResult

    s = 300.0 + 2.0 * np.arange(400)
    vx = np.where(s < 700, 80.0, np.maximum(np.sqrt(np.maximum(80.0**2 - 2 * 40.0 * (s - 700), 0)), 30.0))
    t = np.r_[0.0, np.cumsum(2.0 / (0.5 * (vx[1:] + vx[:-1])))]
    z = np.zeros((400, 7))
    z[:, 2] = vx
    z[:, 6] = np.where((s >= 700) & (vx > 30.0), -0.7, 0.4)
    u = np.zeros((400, 2))
    u[:, 1] = z[:, 6]
    res = OCPResult(s=s, z=z, u=u, t=t, lap_time=float(t[-1]), success=True, status="ok",
                    iterations=0, solve_seconds=0.0, kappa=np.zeros(400), w_left=np.full(400, 6.0),
                    w_right=np.full(400, 6.0), dual_left=np.zeros(400), dual_right=np.zeros(400))
    tr = M.trajectory_from_socp(res)
    assert tr.dt is None and tr.a.shape == (399, 2) and np.array_equal(tr.t, t)
    m = M.response_metrics(tr, marks())
    assert m.has_brake and m.s_b == pytest.approx(700.0, abs=2.0)
    assert m.v_e == pytest.approx(40.0, abs=0.5) and np.isnan(m.J) and np.isnan(m.osc_index)


def test_metrics_schema_and_diff():
    assert len(set(M.METRIC_COLUMNS)) == len(M.METRIC_COLUMNS)
    assert {"sight_at_start", "blip_s_last"} <= set(M.METRIC_COLUMNS)
    assert {"corner_regret", "dt_corner_socp"} <= set(M.KEY_ERROR_COLUMNS)
    a, b = M.ResponseMetrics(v_e=40.0, cause="slip"), M.ResponseMetrics(v_e=40.0 + 1e-9, cause="slip")
    d, f = M.metrics_max_diff(a, b)
    assert d == pytest.approx(1e-9) and f == "v_e"
    assert M.metrics_max_diff(a, M.ResponseMetrics(v_e=40.0, cause="spin"))[1] == "cause"
    assert M.metrics_max_diff(a, M.ResponseMetrics())[0] == np.inf
