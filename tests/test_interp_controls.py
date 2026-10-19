"""Scripted positive controls (protocol 7) and the P7 probes, on synthetic corners only.

No trained agent is touched (freeze rule): the controls are scripted and the one network
is randomly initialised. Stimuli come from a local trapezoid builder (the protocol 3.1
formula) standing in for ``stimuli.corner_profile`` / ``shift_corner``; model tables come
from ``measures.build_model_tables`` when it exists, else from a local fallback; runs use a
plain ``step_no_reset`` loop standing in for the harness.
"""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.env import racing
from apexzero.interp import controls as C
from apexzero.interp import probe as PR
from apexzero.interp.measures import ModelTables, Trajectory
from apexzero.interp.stimuli import CornerSpec, Landmarks


@pytest.fixture(scope="module", autouse=True)
def _float32():
    """Rollouts, controls and stimuli arrays run in float32. tests/test_dynamics.py and
    tests/test_env.py switch x64 on globally when they are imported, so in a full-suite run
    this module switches it off for its own tests and restores it afterwards."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


from apexzero.oracle.qss import GG
from apexzero.tracks.generator import strip_arrays
from apexzero.vehicle.dynamics import tyre_forces
from apexzero.vehicle.params import F1_2025

CV = 0.9
A1 = 310 / 3.6
S_C = 1000.0
_EP = racing.EnvParams(window_steps=10**9)


# ------------------------------------------------------------------------- doubles


def _fallback_tables() -> ModelTables:
    """a_dec from a pedal sweep of the model; a_y,max from the point-mass envelope scaled by
    protocol 1.2's ratio to the model (0.99 at 40 km/h to 0.87 at 300 km/h)."""
    v_dec = np.arange(0.0, 315.0 + 1e-9, 1.0) / 3.6
    pedals = np.arange(-1.0, 1e-9, 0.005)
    acc = C.straight_accel(v_dec[:, None], pedals[None, :])
    v_lat = np.arange(20.0, 315.0 + 1e-9, 5.0) / 3.6
    ratio = np.interp(v_lat, [40 / 3.6, 300 / 3.6], [0.99, 0.87])
    return ModelTables(v_dec=v_dec, a_dec=-acc.min(axis=1), pedal_dec=pedals[acc.argmin(axis=1)],
                       v_lat=v_lat, a_y_max=GG.from_params(F1_2025).a_lat_max(v_lat) * ratio,
                       alpha_max=F1_2025.alpha_max)


@pytest.fixture(scope="module")
def car():
    try:
        from apexzero.interp.measures import build_model_tables

        tables = build_model_tables()
    except NotImplementedError:
        tables = _fallback_tables()
    return C.car_tables(tables)


def l_typ(R: float) -> float:
    return float(np.interp(np.log(R), np.log([9.4, 14.6, 22.9, 40.0, 75.0]),
                           [26.0, 33.0, 40.0, 44.0, 44.0]))


def ramp(R: float, theta_deg: float) -> float:
    return max(10.0, min(l_typ(R), math.radians(theta_deg) * R))


def trapezoid(R, theta_deg, direction, l_r, width=12.0, s_c=S_C, cap=True, n=2560, ds=2.0):
    """Protocol 3.1 trapezoid strip with the inside cap (test double of corner_profile)."""
    th = math.radians(theta_deg)
    kp = min(1.0 / R, th / l_r)
    Lp = max(th / kp - l_r, 0.0)
    u = np.arange(n) * ds - s_c
    prof = np.clip(np.where(u < l_r, u / l_r, np.where(u < l_r + Lp, 1.0,
                                                       (2 * l_r + Lp - u) / l_r)), 0.0, 1.0)
    kappa = direction * kp * prof
    wl, wr = np.full(n, width / 2), np.full(n, width / 2)
    if cap:
        c = 1.0 + 0.6 / np.maximum(np.abs(kappa), 1e-6)
        wl = np.where(kappa > 0, np.minimum(wl, c), wl)
        wr = np.where(kappa < 0, np.minimum(wr, c), wr)
    L_c = Lp + 2 * l_r
    lm = Landmarks(s_c, s_c + l_r, s_c + L_c / 2, s_c + L_c - l_r, s_c + L_c)
    return SimpleNamespace(kappa=kappa, w_left=wl, w_right=wr, landmarks=lm)


def profile_double(spec: CornerSpec, delta: float):
    """Test double of stimuli.shift_corner / corner_profile."""
    return trapezoid(spec.R, spec.theta_deg, spec.direction, spec.l_r, spec.width,
                     spec.s_c + delta, cap=not spec.no_cap)


def init_state(x0, prev=(0.0, 0.0), accel=(0.0, 0.0)):
    x0 = jnp.asarray(x0, jnp.float32)
    return racing.EnvState(
        x=x0, prev_action=jnp.asarray(prev, jnp.float32), accel=jnp.asarray(accel, jnp.float32),
        ctx=jnp.asarray(np.asarray(racing.default_context(), np.float32)),
        hidden_mu=jnp.float32(1.0), t=jnp.zeros((), jnp.int32), slow=jnp.float32(0.0),
        s0=x0[0])


_LOOPS: dict = {}


def rollout(policy, params, profiles, x0s, aux, steps):
    """Plain step loop (test double of the harness): deterministic mean actions, state
    frozen after termination. Returns x [B, T+1, 8], a [B, T, 2] (clipped), accel and
    prev_action [B, T+1, 2], valid [B, T+1], term_step [B]."""
    key = (id(policy), steps)
    if key not in _LOOPS:
        def one(prm, tr, st, ax):
            p = racing.params_from_context(st.ctx, hidden_mu=st.hidden_mu)
            o = racing.observe(st, tr, p)

            def body(c, _):
                st, o, alive = c
                mean, _ = policy(prm, o, st, tr, ax)
                o2, st2, _, term, trunc, _ = racing.step_no_reset(st, mean, tr, _EP)
                st_n = jax.tree.map(lambda a, b: jnp.where(alive, a, b), st2, st)
                out = (st_n.x, st_n.accel, st_n.prev_action, jnp.clip(mean, -1, 1), alive,
                       alive & term)
                return (st_n, jnp.where(alive, o2, o), alive & ~term & ~trunc), out

            _, out = jax.lax.scan(body, (st, o, jnp.asarray(True)), None, steps)
            return out

        _LOOPS[key] = (jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0))), policy)
    tr = jax.vmap(strip_arrays)(*(jnp.asarray(np.stack([getattr(p, f) for p in profiles]),
                                              jnp.float32)
                                  for f in ("kappa", "w_left", "w_right")))
    st = jax.tree.map(lambda *a: jnp.stack(a), *[init_state(x) for x in x0s])
    xs, acc, prev, a, alive, termd = (np.asarray(z) for z in _LOOPS[key][0](
        params, tr, st, jnp.asarray(aux, jnp.int32)))
    x0 = np.asarray(st.x)[:, None]
    term = np.array([int(np.argmax(t)) if t.any() else -1 for t in termd])
    return {"x": np.concatenate([x0, xs], 1), "a": a,
            "accel": np.concatenate([np.asarray(st.accel)[:, None], acc], 1),
            "prev_action": np.concatenate([np.asarray(st.prev_action)[:, None], prev], 1),
            "valid": np.concatenate([np.ones((len(x0s), 1), bool), alive], 1),
            "term_step": term}


def beta_f(x: np.ndarray) -> np.ndarray:
    """Front brake demand (protocol 5), test double of measures.front_brake_demand."""
    p = F1_2025
    fo = tyre_forces(x.astype(float), p, C._NP)
    brake = np.maximum(-x[..., 7], 0.0) * np.tanh(x[..., 3] / 0.5)
    mux = np.maximum(p.mu_scale * (p.mux_f + p.kmu * (0.5 * fo.fz_f - p.fz0)), 0.3)
    return brake * p.f_brake_max * p.brake_bias / (mux * fo.fz_f)


def onset(x: np.ndarray, s_apex: float) -> tuple[float, float, int]:
    """(s_b, v_b, k_b): start of the last run of >= 3 states with beta_f > 0.3 beginning
    before s_apex, interpolated at the 0.3 crossing (test double of measures.brake_onset)."""
    b = beta_f(x)
    on = b > 0.3
    starts = [k for k in range(1, len(on) - 2)
              if on[k] and not on[k - 1] and on[k:k + 3].all() and x[k, 0] < s_apex]
    k = starts[-1]
    w = (0.3 - b[k - 1]) / (b[k] - b[k - 1])
    s_b = x[k - 1, 0] + w * (x[k, 0] - x[k - 1, 0])
    return float(s_b), float(np.interp(s_b, x[:, 0], x[:, 3])), k


def completed(run: dict, i: int, s_c1: float) -> bool:
    t = run["term_step"][i]
    return t < 0 or run["x"][i, t + 1, 0] >= s_c1 + 250.0


# ------------------------------------------------------------------------- car tables


def test_pedal_map_inverts_the_straight_line_model(car):
    for v in (15.0, 40.0, 70.0, 85.0):
        for a in (-20.0, -5.0, 0.0, 3.0):
            if a > np.interp(v, car.v, car.a_acc) - 0.5:
                continue                                   # beyond full throttle
            ped = float(C._pedal_for(jnp.float32(v), jnp.float32(a), C._as_jax(car)))
            assert C.straight_accel(v, ped) == pytest.approx(a, abs=0.3)
        # Beyond the tyres' limit the map returns the saturating pedal, not -1.
        p_sat = float(C._pedal_for(jnp.float32(v), jnp.float32(-200.0), C._as_jax(car)))
        assert -1.0 < p_sat < 0.0
        assert C.straight_accel(v, p_sat) == pytest.approx(-np.interp(v, car.v, car.a_dec),
                                                           rel=0.01)
    assert 300.0 < car.v_top * 3.6 < 320.0
    assert math.degrees(car.alpha_peak) == pytest.approx(7.15, abs=0.05)
    # Understeer: the model needs more steering than the kinematic angle, more near the limit.
    i = int(np.argmin(np.abs(car.v - 60.0)))
    assert np.all(np.diff(car.us_dd[i]) >= -1e-9) and 0.005 < car.us_dd[i, -1] < 0.05


def test_grip_profile_is_the_cornering_limit_on_an_arc_and_brakes_within_a_dec(car):
    p = trapezoid(40.0, 180.0, 1, 44.0)
    T = C.q_target_profile(p.kappa, car, 1.0)
    lm = p.landmarks
    s = np.arange(len(T)) * 2.0
    arc = (s > lm.s_a0 + 2) & (s < lm.s_a1 - 2)
    v_ss = float(np.interp(1 / 40.0, car.kap_ss, car.v_ss))
    assert np.allclose(T[arc], v_ss, rtol=1e-5)
    # Before the corner the target falls no faster than straight-line braking.
    before = (s < lm.s_c0) & (T < car.v_top - 1e-3)
    i = np.nonzero(before)[0]
    decel = (T[i] ** 2 - T[i + 1] ** 2) / (2 * 2.0)
    assert np.all(decel <= np.interp(T[i + 1], car.v, car.a_dec) * 1.001)
    # c_v scales the grip-limited part only: straights stay at the car's top speed.
    T9 = C.q_target_profile(p.kappa, car, 0.9)
    assert T9[0] == pytest.approx(car.v_top, rel=1e-6)
    assert np.allclose(T9[arc], 0.9 * v_ss, rtol=1e-5)


def test_c2_reads_a_9m_corner_as_12m(car):
    p = trapezoid(9.0, 90.0, 1, ramp(9.0, 90.0))
    q = C.q_target_profile(p.kappa, car, CV)
    c2 = C.q_target_profile(p.kappa, car, CV, kappa_cap=C.C2_KAPPA_CAP)
    v9 = float(np.interp(1 / 9.0, car.kap_ss, car.v_ss))
    v12 = float(np.interp(1 / 12.0, car.kap_ss, car.v_ss))
    assert c2.min() == pytest.approx(CV * v12, rel=1e-5)
    assert q.min() == pytest.approx(CV * v9, rel=0.01) and q.min() < c2.min()
    # At R >= 12 m the cap never binds: c2 is Q.
    p20 = trapezoid(20.0, 90.0, 1, ramp(20.0, 90.0))
    assert np.array_equal(C.q_target_profile(p20.kappa, car, CV),
                          C.q_target_profile(p20.kappa, car, CV, kappa_cap=C.C2_KAPPA_CAP))
    # Closed loop: c2's minimum speed in the R9 corner is the R12 cornering speed.
    prm = C.q_params_for_kappa(p.kappa[None], [S_C], car, CV, "c2")
    run = rollout(C.make_q_policy("c2"), prm, [p], [[400.0, 0, 0, A1, 0, 0, 0, 0]], [0], 400)
    x = run["x"][0]
    win = (x[:, 0] > S_C - 50) & (x[:, 0] < p.landmarks.s_c1 + 50)
    assert completed(run, 0, p.landmarks.s_c1)
    assert x[win, 3].min() == pytest.approx(CV * v12, rel=0.03)


def test_c1_moves_the_braking_branch_15m_towards_the_corner(car):
    p = trapezoid(22.9, 90.0, 1, ramp(22.9, 90.0))
    q = C.q_target_profile(p.kappa, car, CV)
    c1 = C.q_target_profile(p.kappa, car, CV, brake_shift=C.C1_SHIFT)

    def branch_start(T):
        falling = np.r_[T[:-1] > T[1:] + 1e-6, False]
        k = int(np.nonzero(falling)[0][0])
        return 2.0 * k

    assert branch_start(c1) - branch_start(q) == pytest.approx(15.0, abs=2.0)
    assert np.all(c1 >= q - 1e-9)
    assert c1.min() == pytest.approx(q.min(), rel=1e-9)      # the cornering limit is kept
    assert c1[int(S_C / 2)] > 1.2 * q[int(S_C / 2)]          # late braking: fast at s_c0


# ------------------------------------------------------------------------- Q closed loop


def test_q_completes_gentle_and_moderate_corners_deterministically(car):
    specs = [(100.0, 90.0), (30.0, 90.0)]          # gentle, moderate
    profs = [trapezoid(R, th, 1, ramp(R, th)) for R, th in specs]
    prm = C.q_params_for_kappa(np.stack([p.kappa for p in profs]), [S_C, S_C], car, CV)
    x0 = [[400.0, 0, 0, A1, 0, 0, 0, 0], [750.0, 0, 0, 200 / 3.6, 0, 0, 0, 0]]
    pol = C.make_q_policy("Q")
    run = rollout(pol, prm, profs, x0, [0, 1], 400)
    again = rollout(pol, prm, profs, x0, [0, 1], 400)
    assert np.array_equal(run["x"], again["x"])
    s_grid = np.arange(2560) * 2.0
    for i, p in enumerate(profs):
        lm = p.landmarks
        assert completed(run, i, lm.s_c1)
        x = run["x"][i]
        T = np.asarray(prm.v_target[i])
        v_e = np.interp(lm.s_c0, x[:, 0], x[:, 3])
        assert v_e == pytest.approx(np.interp(lm.s_c0, s_grid, T), rel=0.02)
        win = (x[:, 0] > lm.s_c0 - 50) & (x[:, 0] < lm.s_c1 + 50)
        tw = (s_grid > lm.s_c0 - 50) & (s_grid < lm.s_c1 + 50)
        assert x[win, 3].min() == pytest.approx(T[tw].min(), rel=0.03)
        assert np.abs(x[win, 1]).max() < 5.0 - 0.5


def test_q_is_mirror_symmetric(car):
    profs = [trapezoid(14.6, 180.0, d, ramp(14.6, 180.0)) for d in (1, -1)]
    prm = C.q_params_for_kappa(np.stack([p.kappa for p in profs]), [S_C, S_C], car, CV)
    x0 = [[400.0, 0, 0, A1, 0, 0, 0, 0]] * 2
    run = rollout(C.make_q_policy("Q"), prm, profs, x0, [0, 1], 400)
    xl, xr = run["x"]
    sign = np.array([1, -1, -1, 1, -1, -1, -1, 1])
    assert np.abs(xl - xr * sign).max() < 1e-3
    assert completed(run, 0, profs[0].landmarks.s_c1)


def test_offline_prediction_matches_the_closed_loop(car):
    cases = [(22.9, 90.0, 400.0, A1), (13.1, 180.0, 400.0, A1), (22.9, 90.0, 750.0, 220 / 3.6)]
    profs = [trapezoid(R, th, 1, ramp(R, th)) for R, th, _, _ in cases]
    prm = C.q_params_for_kappa(np.stack([p.kappa for p in profs]), [S_C] * 3, car, CV)
    x0 = [[s0, 0, 0, v0, 0, 0, 0, 0] for _, _, s0, v0 in cases]
    run = rollout(C.make_q_policy("Q"), prm, profs, x0, [0, 1, 2], 400)
    for i, (p, (_, _, s0, v0)) in enumerate(zip(profs, cases)):
        lm = p.landmarks
        x = run["x"][i]
        s_b, v_b, _ = onset(x, lm.s_apex)
        win = (x[:, 0] > lm.s_c0 - 50) & (x[:, 0] < lm.s_c1 + 50)
        meas = {"v_e": float(np.interp(lm.s_c0, x[:, 0], x[:, 3])),
                "v_min": float(x[win, 3].min()), "d_b": lm.s_c0 - s_b, "v_b": v_b}
        # With the closed loop itself as the key, the harness errors are 0 and the predicted
        # errors are the prediction-minus-harness differences that protocol 7 bounds.
        pred = C.offline_prediction(np.asarray(prm.v_target[i]), p.kappa, lm, s0, v0, meas, car)
        res = {"e_ve": 0.0, "e_vmin": 0.0, "dd_b": 0.0, "v_b": v_b}
        assert C.offline_match(pred, res) == {"e_ve": True, "e_vmin": True, "dd_b": True}


def test_qp_equals_q_when_the_preview_resolves_the_corner(car):
    p = trapezoid(100.0, 90.0, 1, 44.0)           # 44 m ramps, 113 m arc: >> preview spacing
    prm = C.q_params_for_kappa(p.kappa[None], [S_C], car, CV)
    x0 = [[400.0, 0, 0, A1, 0, 0, 0, 0]]
    run_q = rollout(C.make_q_policy("Q"), prm, [p], x0, [0], 300)
    run_p = rollout(C.make_q_policy("Qp"), prm, [p], x0, [0], 300)
    xq, xp = run_q["x"][0], run_p["x"][0]
    assert completed(run_p, 0, p.landmarks.s_c1)
    assert np.abs(xq[:, 3] - xp[:, 3]).max() / A1 < 0.01
    # The per-step targets agree along Q's own run while the corner is in view.
    tr = strip_arrays(*(jnp.asarray(getattr(p, f), jnp.float32)
                        for f in ("kappa", "w_left", "w_right")))
    sel = (xq[:, 0] > S_C - 450) & (xq[:, 0] < p.landmarks.s_a1 - 20)   # exit: T >> v
    states = [init_state(x) for x in xq[sel]]
    st = jax.tree.map(lambda *a: jnp.stack(a), *states)
    pr = racing.params_from_context(st.ctx[0])
    obs = jax.vmap(lambda s: racing.observe(s, tr, pr))(st)
    vq = jnp.interp(st.x[:, 0], jnp.arange(2560) * 2.0, prm.v_target[0])
    vp = jax.vmap(lambda o, s: C.preview_profile(o, jnp.maximum(s.x[3], 0.0), prm)[0])(obs, st)
    assert float(jnp.max(jnp.abs(vp - vq) / vq)) < 0.01


def test_c3_brakes_at_a_time_to_corner_with_a_lawful_entry_speed(car):
    p = trapezoid(22.9, 90.0, 1, ramp(22.9, 90.0))
    pol = C.make_q_policy("c3")
    x0 = [[400.0, 0, 0, A1, 0, 0, 0, 0]]
    d_b = {}
    for tau in (1.2, 1.6):                                # braking passes beta_f 0.3 at once
        prm = C.q_params_for_kappa(p.kappa[None], [S_C], car, CV, "c3", tau_c=tau)
        T = np.asarray(prm.v_target[0])
        run = rollout(pol, prm, [p], x0, [0], 400)
        x, a = run["x"][0], run["a"][0]
        assert completed(run, 0, p.landmarks.s_c1)
        k = int(np.nonzero(a[:, 1] < 0)[0][0])            # first braking command
        assert S_C - x[k, 0] == pytest.approx(tau * x[k, 3], abs=x[k, 3] * 0.05)
        v_e = np.interp(S_C, x[:, 0], x[:, 3])
        assert v_e == pytest.approx(np.interp(S_C, np.arange(2560) * 2.0, T), rel=0.03)
        d_b[tau] = S_C - onset(x, p.landmarks.s_apex)[0]
        assert tau * x[k, 3] - 0.2 * x[k, 3] < d_b[tau] <= tau * x[k, 3]   # pedal lag
    assert d_b[1.6] > d_b[1.2] + 25.0


def test_tune_cv_and_fit_tau_c_logic(car):
    bank = SimpleNamespace(stimuli=[], kappa=None)
    calls = []

    def completion_fn(c_v):
        calls.append(c_v)
        return np.array([True, c_v <= 0.85, True])

    c_v, rates = C.tune_cv(bank, [], car, completion_fn)
    assert c_v == 0.85 and calls == [0.95, 0.90, 0.85]
    assert rates[0.95] == pytest.approx(2 / 3)
    assert C.tune_cv(bank, [], car, lambda c: np.array([False]))[0] is None
    tau = C.fit_tau_c(bank, [None], car, d_b_key=200.0, c_v=CV,
                      d_b_fn=lambda t: 80.0 * t, tol=0.01)
    assert tau == pytest.approx(2.5, abs=1e-3)
    # Gentle braking at long tau_c passes the beta_f threshold late: the first crossing wins.
    tau = C.fit_tau_c(bank, [None], car, 150.0, CV, tol=0.01,
                      d_b_fn=lambda t: 86.0 * t if t < 2.5 else 215.0 - 60.0 * (t - 2.5))
    assert tau == pytest.approx(150.0 / 86.0, abs=1e-3)
    with pytest.raises(ValueError):
        C.fit_tau_c(bank, [None], car, 1e4, CV, d_b_fn=lambda t: 80.0 * t)


def test_tuning_through_the_harness_on_the_pilot_minigrid(car):
    """tune_cv, completion and fit_tau_c on the real harness (rollout, measures) with the
    pilot mini-grid bank (scripted control only)."""
    from apexzero.interp import stimuli

    bank, cells = stimuli.pilot_minigrid()
    c_v, rates = C.tune_cv(bank, cells, car)
    assert c_v in C.CV_CHOICES and rates[c_v] == 1.0
    cell = next(c for c in cells if bank.stimuli[c.stim_id].name == "pilot/R30/T90/L")
    d_b = C._harness_d_b(bank, cell, car, c_v)
    target = d_b(1.5)
    assert d_b(1.0) < target < d_b(2.0)
    assert C.fit_tau_c(bank, [cell], car, target, c_v, tol=0.5) == pytest.approx(1.5, abs=0.05)


# ------------------------------------------------------------------------- T and K


def test_tape_replays_the_key_commands():
    tp = C.tape_params([np.array([[0.1, 1.0], [0.2, 0.5]]), np.array([[-0.3, -0.4]])])
    pol = C.make_tape_policy()
    o = jnp.zeros(racing.OBS_DIM, jnp.float32)
    for aux, t, want in ((0, 0, (0.1, 1.0)), (0, 1, (0.2, 0.5)), (0, 7, (0.2, 0.5)),
                         (1, 0, (-0.3, -0.4)), (1, 1, (-0.3, -0.4))):
        st = init_state(np.zeros(8))._replace(t=jnp.asarray(t, jnp.int32))
        mean, log_std = pol(tp, o, st, None, jnp.asarray(aux))
        assert np.allclose(mean, want) and np.all(np.isneginf(log_std))


def test_knn_returns_the_neighbours_action_and_the_weighted_fallback():
    rng = np.random.default_rng(0)
    M = 300
    obs = rng.normal(size=(M, racing.OBS_DIM)).astype(np.float32)
    act = rng.uniform(-1, 1, size=(M, 2)).astype(np.float32)
    lap = np.repeat(np.arange(6), M // 6)
    buf = C.KnnBuffer(obs=obs, act=act, lap=lap, lolo_median_nn=1.0)
    mean, var = rng.normal(size=racing.OBS_DIM), rng.uniform(0.5, 2.0, racing.OBS_DIM)
    prm = C.knn_params(buf, mean, var)
    raw = obs[17] * np.sqrt(var + 1e-8) + mean           # un-normalised buffer row 17
    st = init_state(np.zeros(8))
    got, log_std = C.make_knn_policy(1)(prm, jnp.asarray(raw, jnp.float32), st, None, 0)
    assert np.allclose(got, act[17]) and np.all(np.isneginf(log_std))
    q = obs[3] + 0.3 * rng.normal(size=racing.OBS_DIM).astype(np.float32)
    raw = q * np.sqrt(var + 1e-8) + mean
    got5, _ = C.make_knn_policy(C.KNN_K_FALLBACK)(prm, jnp.asarray(raw, jnp.float32), st,
                                                  None, 0)
    d = np.linalg.norm(obs - q, axis=1)
    idx = np.argsort(d)[:5]
    want = (act[idx] / d[idx, None]).sum(0) / (1 / d[idx]).sum()
    assert np.allclose(got5, want, atol=1e-4)


def test_buffer_lolo_distance_and_d_ood():
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(500, 12))
    lap = rng.integers(0, 5, 500)
    buf = C.assemble_buffer(obs, rng.normal(size=(500, 2)), lap, n_obs=200,
                            rng=np.random.default_rng(2))
    assert len(buf.obs) == 200
    d = np.linalg.norm(buf.obs[:, None] - buf.obs[None], axis=-1)
    d[buf.lap[:, None] == buf.lap[None, :]] = np.inf
    assert buf.lolo_median_nn == pytest.approx(np.median(d.min(1)), rel=1e-4)
    q = rng.normal(size=(7, 12))
    want = np.linalg.norm(q[:, None] - buf.obs[None], axis=-1).min(1) / buf.lolo_median_nn
    assert np.allclose(C.d_ood(q, buf), want, rtol=1e-4)


def test_collect_laps_with_a_scripted_actor():
    n = 2560
    z = jnp.zeros(n, jnp.float32)
    tr = strip_arrays(z, z + 6.0, z + 6.0)
    v_ref = np.full(n, 60.0)

    def actor_fn(o):
        return jnp.asarray([0.0, 0.3]), jnp.asarray([-3.0, -3.0])

    out = C.collect_laps(actor_fn, tr, v_ref, 150.0, 3, jax.random.PRNGKey(0),
                         np.zeros(racing.OBS_DIM), np.ones(racing.OBS_DIM), max_steps=120)
    assert out["obs"].shape[1] == racing.OBS_DIM and len(out["obs"]) == len(out["lap"])
    assert set(np.unique(out["lap"])) == {0, 1, 2}
    assert np.abs(out["obs"]).max() <= 10.0
    assert np.allclose(out["act"], [0.0, 0.3])
    assert not np.allclose(out["act_sampled"], out["act"])


# ------------------------------------------------------------------------- probes


def test_preview_geometry_matches_the_observation():
    for v in (40.0, 70.0, 86.0):
        d_k = np.asarray(racing.preview_offsets(jnp.float32(v)), float)
        assert np.allclose(PR.preview_index(d_k, v), np.arange(1, 41), atol=1e-4)
        mid = 0.5 * (d_k[10:] + d_k[9:-1])
        assert np.allclose(PR.preview_spacing(mid, v), np.diff(d_k)[9:], rtol=0.01)


def test_detrend_has_six_degrees_of_freedom():
    d = np.linspace(100.0, 400.0, 160)
    assert PR.hat_trace(d) == pytest.approx(6.0, abs=1e-3)
    lin = 0.3 + 0.002 * d
    assert np.abs(PR.detrend(d, lin)).max() < 1e-8
    k = PR.preview_index(d, 80.0)
    ripple = 0.05 * np.sin(2 * np.pi * k)
    res = PR.detrend(d, 0.8 - 1e-5 * (d - 250) ** 2 + ripple)
    assert np.sqrt(np.mean((res - ripple) ** 2)) < 0.01


def test_ripple_ratio_and_amplitude():
    rng = np.random.default_rng(3)
    d = np.arange(150.0, 480.0, 0.25)
    k = PR.preview_index(d, 86.0)
    Pi, A = PR.ripple(0.05 * np.sin(2 * np.pi * k) + 0.002 * rng.normal(size=len(d)), k, d,
                      500.0)
    assert Pi > 100.0 and A == pytest.approx(0.05 / np.sqrt(2), rel=0.05)
    Pi0, _ = PR.ripple(rng.normal(size=len(d)), k, d, 500.0)
    assert Pi0 < 20.0
    assert np.isnan(PR.ripple(np.zeros(5), k[:5], np.full(5, 450.0), 500.0)[1])


def test_scan_state_is_steady_full_throttle():
    st = PR.scan_state(70.0, 200.0, S_C)
    assert np.allclose(np.asarray(st.x), [S_C - 200.0, 0, 0, 70.0, 0, 0, 0, 1.0])
    assert np.allclose(np.asarray(st.prev_action), [0.0, 1.0])
    ax = PR.full_throttle_accel(70.0)
    assert float(st.accel[0]) == pytest.approx(ax) and float(st.accel[1]) == 0.0
    assert 0.0 < ax < C.straight_accel(70.0, 1.0) * 1.05


def _random_actor():
    from apexzero.agents.ppo import Actor, PPOConfig, RunningMeanStd

    cfg = PPOConfig(network="mlp", actor_width=32, mlp_layers=2, activation="tanh")
    actor = Actor(cfg, 2, False)
    params = actor.init(jax.random.PRNGKey(0), jnp.zeros(racing.OBS_DIM))

    def policy(prm, o, st, tr, aux):
        return actor.apply(prm, o)

    rms = RunningMeanStd(jnp.zeros(racing.OBS_DIM), jnp.ones(racing.OBS_DIM), jnp.asarray(1.0))
    return policy, params, rms


def test_open_loop_scan_runs_on_a_network_and_on_q(car):
    spec = CornerSpec(R=11.7, theta_deg=90.0, direction=1, l_r=ramp(11.7, 90.0))
    policy, params, rms = _random_actor()
    res = PR.open_loop_scan(policy, params, rms, spec, 250 / 3.6, d_min=200.0,
                            profile_fn=profile_double)
    d_max = float(PR.d_max_of(250 / 3.6))
    assert res.d[0] == 200.0 and d_max - 0.25 < res.d[-1] <= d_max
    assert np.all(np.isfinite(res.pedal)) and np.isfinite(res.A) and np.isfinite(res.Pi)
    assert np.allclose(res.k_index, PR.preview_index(res.d, 250 / 3.6))

    def q_fn(kappa):
        return C.q_params_for_kappa(kappa, [S_C] * len(kappa), car, CV), np.arange(len(kappa))

    rq = PR.open_loop_scan(C.make_q_policy("Q"), None, None, spec, 250 / 3.6, d_min=200.0,
                           profile_fn=profile_double, params_fn=q_fn)
    assert np.allclose(rq.pedal, rq.pedal[0], atol=1e-6)     # far from braking: flat out


def test_aliasing_fit_is_zero_for_a_linear_reader_and_positive_for_a_sampled_one():
    """Mechanics of A_alias with two synthetic readers of the corner's position: one whose
    pedal is linear in the distance to the corner start (A_alias = 0) and one that reads it
    through a ripple at the preview spacing (A_alias > 0)."""
    spec = CornerSpec(R=11.7, theta_deg=90.0, direction=1, l_r=ramp(11.7, 90.0))
    grid = jnp.arange(0.0, 600.0, 0.05)

    def reader(ripple):
        def policy(prm, o, st, tr, aux):
            from apexzero.vehicle.sim import kappa_at

            hit = jnp.abs(kappa_at(st.x[0] + grid, tr)) > 0.02    # inside the ramp
            d = grid[jnp.argmax(hit)]
            ped = 0.002 * d - 0.5 + ripple * jnp.sin(2 * jnp.pi * d / 12.0)
            return jnp.stack([0.0, ped]), jnp.full(2, -jnp.inf)
        return policy

    x = np.zeros((41, 8))
    x[:, 0] = np.linspace(600.0, 800.0, 41)
    x[:, 3] = 80.0
    traj = Trajectory(source="control", t=np.arange(41) * 0.05, x=x, valid=np.ones(41, bool),
                      a=np.zeros((40, 2)))
    lin = PR.aliasing_detail(reader(0.0), None, None, spec, traj, 2.0, profile_fn=profile_double,
                             clip=False)
    rip = PR.aliasing_detail(reader(0.1), None, None, spec, traj, 2.0, profile_fn=profile_double,
                             clip=False)
    assert len(lin.t) == 21 and lin.A_alias < 1e-3
    assert np.allclose(lin.beta, 0.002 * lin.spacing, rtol=0.02, atol=1e-4)
    assert rip.A_alias > 0.02


def test_aliasing_amplitude_of_q_and_qp(car):
    p = trapezoid(9.4, 90.0, 1, ramp(9.4, 90.0))
    spec = CornerSpec(R=9.4, theta_deg=90.0, direction=1, l_r=ramp(9.4, 90.0))
    prm = C.q_params_for_kappa(p.kappa[None], [S_C], car, CV)
    run = rollout(C.make_q_policy("Qp"), prm, [p], [[400.0, 0, 0, A1, 0, 0, 0, 0]], [0], 300)
    x = run["x"][0]
    t = np.arange(len(x)) * 0.05
    traj = Trajectory(source="control", t=t, x=x.astype(float), valid=run["valid"][0],
                      a=run["a"][0].astype(float))
    # prev_action and accel rebuilt from the trajectory equal the env's own.
    pa, ac = PR._run_aux(traj, None, None, 0.05)
    assert np.abs(pa - run["prev_action"][0]).max() < 1e-6
    assert np.abs(ac - run["accel"][0]).max() < 2e-2
    t_b = t[onset(x, p.landmarks.s_apex)[2]]

    def q_fn(kappa):
        return C.q_params_for_kappa(kappa, [S_C] * len(kappa), car, CV), np.arange(len(kappa))

    det_q = PR.aliasing_detail(C.make_q_policy("Q"), None, None, spec, traj, t_b,
                               profile_fn=profile_double, params_fn=q_fn,
                               prev_action=run["prev_action"][0], accel=run["accel"][0])
    a_qp = PR.aliasing_amplitude(C.make_q_policy("Qp"), prm, None, spec, traj, t_b,
                                 profile_fn=profile_double,
                                 prev_action=run["prev_action"][0], accel=run["accel"][0])
    assert len(det_q.t) == 21 and np.all(det_q.d_c > 0)
    # Q reads the full-resolution road: no residual away from the onset. Within a shift range
    # of the onset the throttle-to-brake switch crosses the state (a step in delta), which
    # the per-state line cannot fit; both controls carry that term.
    early = det_q.t < t_b - 0.2
    assert early.sum() >= 15 and np.abs(det_q.residual[early]).max() < 1e-5
    assert np.isfinite(det_q.A_alias) and np.isfinite(a_qp)
    assert np.isnan(PR.aliasing_amplitude(C.make_q_policy("Qp"), prm, None, spec, traj,
                                          float("nan"), profile_fn=profile_double))
