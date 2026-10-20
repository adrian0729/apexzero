"""Dry-run rulings R20-R26 on the controls (protocol section 12, 2026-10-20).

R22 Q's pedal law (A), offline profile (B) and offline onset (C); R23 c1's per-cell shift;
R24 c2's planted-law check; R26 c3's tau_c; R20 T64 (the float64 tape path) and the x64
assertion of the agents' path; K's C(a) at both k.

Freeze rule: no trained agent is touched. Everything runs on synthetic corners, the pilot
mini-grid bank and scripted controls; keys are simulated where a test needs one (Q's own
offline prediction shifted by a stated lead), never solved, and nothing is written under
outputs/. Runs use a plain ``step_no_reset`` loop standing in for the harness (as
tests/test_interp_controls.py) except where the harness itself is the subject.
"""

import dataclasses
import importlib
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.env import racing
from apexzero.interp import controls as C
from apexzero.interp import measures as M
from apexzero.interp import rollout as R
from apexzero.interp.constants import DT, N_KNOTS, S_C
from apexzero.interp.stimuli import Cell, Landmarks
from apexzero.tracks.generator import strip_arrays
from apexzero.vehicle.params import F1_2025


@pytest.fixture(scope="module", autouse=True)
def _float32():
    """Controls and the harness run in float32 (other test modules switch x64 on globally
    when imported); T64's tests switch it on locally with ``jax.enable_x64(True)``."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    R._compiled.cache_clear()
    yield
    R._compiled.cache_clear()
    jax.config.update("jax_enable_x64", prev)


CV = 0.9
A1 = 310 / 3.6
_EP = racing.EnvParams(window_steps=10**9)


@pytest.fixture(scope="module")
def tables():
    """``M.build_model_tables()``, built once per test session (shared with
    tests/test_interp_controls.py, which caches it)."""
    return importlib.import_module("test_interp_controls").model_tables_once()


@pytest.fixture(scope="module")
def car(tables):
    return C.car_tables(tables)


# ------------------------------------------------------------------------- doubles


def l_typ(R_: float) -> float:
    return float(np.interp(np.log(R_), np.log([9.4, 14.6, 22.9, 40.0, 75.0]),
                           [26.0, 33.0, 40.0, 44.0, 44.0]))


def ramp(R_: float, theta_deg: float) -> float:
    return max(10.0, min(l_typ(R_), math.radians(theta_deg) * R_))


def trapezoid(R_, theta_deg, direction=1, width=12.0, n=N_KNOTS, ds=2.0):
    """Protocol 3.1 trapezoid at S_C with typical ramps and the inside cap (test double of
    stimuli.corner_profile)."""
    l_r = ramp(R_, theta_deg)
    th = math.radians(theta_deg)
    kp = min(1.0 / R_, th / l_r)
    Lp = max(th / kp - l_r, 0.0)
    u = np.arange(n) * ds - S_C
    prof = np.clip(np.where(u < l_r, u / l_r, np.where(u < l_r + Lp, 1.0,
                                                       (2 * l_r + Lp - u) / l_r)), 0.0, 1.0)
    kappa = direction * kp * prof
    wl, wr = np.full(n, width / 2), np.full(n, width / 2)
    c = 1.0 + 0.6 / np.maximum(np.abs(kappa), 1e-6)
    wl = np.where(kappa > 0, np.minimum(wl, c), wl)
    wr = np.where(kappa < 0, np.minimum(wr, c), wr)
    L_c = Lp + 2 * l_r
    lm = Landmarks(S_C, S_C + l_r, S_C + L_c / 2, S_C + L_c - l_r, S_C + L_c)
    return SimpleNamespace(kappa=kappa, w_left=wl, w_right=wr, landmarks=lm)


class _Bank:
    """The fields of a StimulusBank the control functions read (kappa, widths, stimuli
    records with landmarks, ds) and the harness's ``arrays()``."""

    def __init__(self, profiles):
        self.kappa = np.stack([p.kappa for p in profiles])
        self.w_left = np.stack([p.w_left for p in profiles])
        self.w_right = np.stack([p.w_right for p in profiles])
        self.ds = 2.0
        self.stimuli = [SimpleNamespace(stim_id=i, landmarks=p.landmarks, spec=None,
                                        direction=1, mirror_id=-1, name=f"syn{i}")
                        for i, p in enumerate(profiles)]
        self.profiles = profiles

    def arrays(self):
        fn = jax.jit(jax.vmap(lambda k, wl, wr: strip_arrays(k, wl, wr, 2.0)))
        return fn(*(jnp.asarray(getattr(self, a), jnp.float32)
                    for a in ("kappa", "w_left", "w_right")))


def cell(cid, stim, v0, D, n_phases=1, W=400):
    return Cell(cell_id=cid, stim_id=stim, pair_id=cid, tiers=("A",), approach="A1", v0=v0,
                D=D, W=W, P=v0 * DT, n_phases=n_phases)


def init_state(x0):
    x0 = jnp.asarray(x0, jnp.float32)
    return racing.EnvState(
        x=x0, prev_action=jnp.zeros(2, jnp.float32), accel=jnp.zeros(2, jnp.float32),
        ctx=jnp.asarray(np.asarray(racing.default_context(), np.float32)),
        hidden_mu=jnp.float32(1.0), t=jnp.zeros((), jnp.int32), slow=jnp.float32(0.0),
        s0=x0[0])


_LOOPS: dict = {}


def loop(policy, params, profiles, x0s, aux, steps=400):
    """Plain step loop (test double of the harness), deterministic: x [B, T+1, 8] float64,
    a [B, T, 2] (clipped), term_step [B]."""
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
                return ((st_n, jnp.where(alive, o2, o), alive & ~term & ~trunc),
                        (st_n.x, jnp.clip(mean, -1, 1), alive & term))

            _, out = jax.lax.scan(body, (st, o, jnp.asarray(True)), None, steps)
            return out

        _LOOPS[key] = (jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0))), policy)
    tr = jax.vmap(strip_arrays)(*(jnp.asarray(np.stack([getattr(p, f) for p in profiles]),
                                              jnp.float32)
                                  for f in ("kappa", "w_left", "w_right")))
    st = jax.tree.map(lambda *a: jnp.stack(a), *[init_state(x) for x in x0s])
    xs, a, termd = (np.asarray(z) for z in _LOOPS[key][0](params, tr, st,
                                                          jnp.asarray(aux, jnp.int32)))
    x = np.concatenate([np.asarray(st.x)[:, None], xs], 1).astype(float)
    term = np.array([int(np.argmax(t)) if t.any() else -1 for t in termd])
    return {"x": x, "a": a.astype(float), "term_step": term}


def measured(x, lm):
    """The harness's measures of one deterministic run (measures' own brake demand and
    onset rule): v_e, v_min, d_b, v_b, and the onset record."""
    beta = np.asarray(M.front_brake_demand(x, F1_2025, M.NP)[0], float)
    bo = M.brake_onset(x[:, 0], np.arange(len(x)) * DT, beta, np.ones(len(x), bool),
                       lm.s_apex, x[:, 3])
    win = (x[:, 0] > lm.s_c0 - 50) & (x[:, 0] < lm.s_c1 + 50)
    return dict(v_e=float(np.interp(lm.s_c0, x[:, 0], x[:, 3])), v_min=float(x[win, 3].min()),
                d_b=lm.s_c0 - bo.s_b, v_b=bo.v_b, onset=bo)


# ------------------------------------------------------------------------- R22 A: the pedal law


def _jacobian(car, v_op, a_brk, law):
    """One-step Jacobian of the closed braking loop on the straight-line model at speed
    v_op tracking a target that falls at a_brk (m/s^2), pedal at its steady value. State
    (v, pedal) for law A; (v, pedal, measured acceleration) for the as-built law."""
    s = np.arange(N_KNOTS) * 2.0
    s_end = 3000.0
    T = np.minimum(np.sqrt(1.0 + 2 * a_brk * np.maximum(s_end - s, 0.0)), car.v_top)
    s_op = s_end - (v_op**2 - 1.0) / (2 * a_brk) - C.LEAD_TIME * v_op
    p_op = float(C._pedal_for(np.float64(v_op), np.float64(-a_brk), car, np))

    def step(z):
        v, p, acc = z
        v_t, a_t = C._at(T, np.float64(s_op + C.LEAD_TIME * v), 2.0, np)
        p_hi = C._pedal_for(v, C._uinterp(v, car.v, car.a_acc, np), car, np)
        p_lo = C._pedal_for(v, -C._uinterp(v, car.v, car.a_dec, np), car, np)
        if law == "A":
            c = C.law_a(np.float64(v), np.float64(p), v_t, a_t, p_lo, p_hi, car, C.K_V,
                        C.LEAD_TIME, C.PHI, np)
        else:                                              # as built: v + L * measured accel
            c = np.clip(C.K_V * (v_t - (v + C.LEAD_TIME * acc)) + C._pedal_for(v, a_t, car, np),
                        p_lo, p_hi)
        _, v2, p2 = C.straight_step(np.zeros(1), np.array([v]), np.array([p]),
                                    np.array([float(np.clip(c, -1.0, 1.0))]))
        return np.array([v2[0], p2[0], (v2[0] - v) / DT])

    n = 2 if law == "A" else 3
    base, h = np.array([v_op, p_op, -a_brk]), np.array([1e-4, 1e-5, 1e-4])
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(3)
        e[j] = h[j]
        J[:, j] = (step(base + e) - step(base - e))[:n] / (2 * h[j])
    return J


def test_law_a_has_no_free_parameter_and_solves_its_fixed_point(car):
    tau = F1_2025.tau_pedal
    assert C.LEAD_TIME == pytest.approx(0.5 * DT + tau)
    assert C.PHI == pytest.approx(1 - (tau / C.LEAD_TIME) * (1 - math.exp(-C.LEAD_TIME / tau)))
    assert C.PHI == pytest.approx(0.482, abs=5e-4)
    rng = np.random.default_rng(0)
    v = rng.uniform(10.0, 85.0, 200)
    ped = rng.uniform(-0.8, 0.8, 200)
    v_t = v + rng.uniform(-8.0, 8.0, 200)
    a_t = rng.uniform(-40.0, 8.0, 200)
    p_hi = C._pedal_for(v, C._uinterp(v, car.v, car.a_acc, np), car, np)
    p_lo = C._pedal_for(v, -C._uinterp(v, car.v, car.a_dec, np), car, np)
    c = C.law_a(v, ped, v_t, a_t, p_lo, p_hi, car, C.K_V, C.LEAD_TIME, C.PHI, np)
    v_pred = v + C.LEAD_TIME * ((1 - C.PHI) * C._a_ss(v, ped, car, np)
                                + C.PHI * C._a_ss(v, c, car, np))
    rhs = np.clip(C.K_V * (v_t - v_pred) + C._pedal_for(v, a_t, car, np), p_lo, p_hi)
    assert np.abs(c - rhs).max() < 1e-6
    assert (c >= p_lo - 1e-12).all() and (c <= p_hi + 1e-12).all()
    # The policy's float32 JAX evaluation is the same function.
    cj = C.law_a(*(jnp.asarray(z, jnp.float32) for z in (v, ped, v_t, a_t, p_lo, p_hi)),
                 C._as_jax(car), jnp.float32(C.K_V), jnp.float32(C.LEAD_TIME),
                 jnp.float32(C.PHI))
    assert np.abs(np.asarray(cj, float) - c).max() < 2e-4


def test_q_linearised_braking_loop_is_stable(car):
    """R22 A: the as-built predictor (measured acceleration) has |lambda| > 1 in the
    low-speed braking trail (dry run: 1.09, a ~3-step pedal oscillation); law A has
    |lambda| < 1 everywhere on the braking grid (dry run: 0.43)."""
    lam_a, lam_old = [], []
    for v in (10.0, 12.0, 15.0, 20.0, 40.0, 60.0, 80.0):
        a_dec = float(np.interp(v, car.v, car.a_dec))
        for frac in (0.5, 0.8):
            lam_a.append(np.abs(np.linalg.eigvals(_jacobian(car, v, frac * a_dec, "A"))).max())
            lam_old.append(np.abs(np.linalg.eigvals(_jacobian(car, v, frac * a_dec, "old"))).max())
    assert max(lam_a) < 0.6
    assert max(lam_old) > 1.0


def _old_q_policy(params, o, state, tr, aux):
    """Q with the as-built predictor v + L * measured acceleration (for contrast only)."""
    x, car = state.x, params.car
    v = jnp.maximum(x[3], 0.0)
    steer = C.pursuit_steer(state, tr, params.lookahead_min, params.lookahead_time, car)
    v_t, a_t = C._at(params.v_target[aux], x[0] + params.lead_time * v, tr.ds)
    room_f, room_r = C.axle_room(x, car)
    p_hi = C._pedal_for(v, C._uinterp(v, car.v, car.a_acc) * room_r, car)
    p_lo = C._pedal_for(v, -C._uinterp(v, car.v, car.a_dec) * jnp.minimum(room_f, room_r), car)
    ped = jnp.clip(params.k_v * (v_t - v - params.lead_time * state.accel[0])
                   + C._pedal_for(v, a_t, car), p_lo, p_hi)
    return jnp.stack([steer, ped]).astype(o.dtype), jnp.full((2,), -jnp.inf, o.dtype)


TIGHT = [(9.4, 90.0, 400.0, A1), (13.1, 90.0, 400.0, A1), (14.6, 90.0, 400.0, A1),
         (13.1, 180.0, 400.0, A1), (7.5, 90.0, 400.0, A1), (18.3, 90.0, 400.0, A1),
         (22.9, 90.0, 750.0, 220 / 3.6), (14.6, 90.0, 840.0, 160 / 3.6)]


@pytest.fixture(scope="module")
def tight_runs(car):
    profs = [trapezoid(R_, th) for R_, th, _, _ in TIGHT]
    prm = C.q_params_for_kappa(np.stack([p.kappa for p in profs]), [S_C] * len(TIGHT), car, CV)
    x0 = [[s0, 0, 0, v0, 0, 0, 0, 0] for _, _, s0, v0 in TIGHT]
    aux = list(range(len(TIGHT)))
    return profs, prm, loop(C.make_q_policy("Q"), prm, profs, x0, aux, 500), \
        loop(_old_q_policy, prm, profs, x0, aux, 500)


def test_q_pedal_is_smooth_and_brakes_once_on_tight_corners(tight_runs):
    """R22 A on synthetic corners (the dry run's failures were R7.5-R18.3): the pedal
    command on the approach and corner entry has no 3-step oscillation (mean |second
    difference| ~0.03 against ~0.15-0.19 as built) and one sustained brake before the apex
    (no re-application)."""
    profs, _, run, old = tight_runs
    for i, p in enumerate(profs):
        lm = p.landmarks
        chatter = {}
        for name, r in (("A", run), ("old", old)):
            x, a = r["x"][i], r["a"][i]
            assert r["term_step"][i] == -1
            sel = (x[:-1, 0] > lm.s_c0 - 150.0) & (x[:-1, 0] < lm.s_c1)
            chatter[name] = float(np.mean(np.abs(np.diff(a[sel, 1], 2))))
        assert chatter["A"] < 0.06 and chatter["old"] > 0.1, (TIGHT[i], chatter)
        assert measured(run["x"][i], lm)["onset"].blips == ()


# ------------------------------------------------------------------------- R22 B, C: offline


def _clamped_forward(T, kappa, s0, v0, car, ds=2.0):
    """The as-built forward pass (acceleration clamped at >= 0), for contrast."""
    s = np.arange(len(T)) * ds
    out = np.full(len(T), np.nan)
    v, sp = min(v0, np.interp(s0, s, T)), s0
    for i in range(int(np.ceil(s0 / ds - 1e-9)), len(T)):
        ay = max(np.interp(v, car.v, car.a_y), 1e-3)
        lat = min(v * v * np.interp(sp, s, np.abs(kappa)) / ay, 1.0)
        a = np.interp(v, car.v, car.a_acc) * np.sqrt(1.0 - lat**2)
        v = min(T[i], np.sqrt(max(v * v + 2.0 * max(a, 0.0) * (s[i] - sp), 0.0)))
        out[i], sp = v, s[i]
    return out


def test_offline_profile_predicts_a_power_limited_speed_loss(car):
    """R22 B: at R171 / 90 deg from 310 km/h Q is flat out at the throttle limit and loses
    speed to cornering drag (dry run: 83.76 m/s measured against 87.07 predicted by the
    clamped pass); the signed steady-turn pass predicts the loss."""
    p = trapezoid(171.0, 90.0)
    lm = p.landmarks
    prm = C.q_params_for_kappa(p.kappa[None], [S_C], car, CV)
    T = np.asarray(prm.v_target[0], float)
    run = loop(C.make_q_policy("Q"), prm, [p], [[400.0, 0, 0, A1, 0, 0, 0, 0]], [0], 400)
    meas = measured(run["x"][0], lm)
    assert meas["v_min"] < meas["v_e"] - 1.5                    # the harness loses speed
    fwd = C.forward_profile(T, p.kappa, 400.0, A1, car)
    s = np.arange(len(T)) * 2.0
    win = (s >= lm.s_c0 - 50) & (s <= lm.s_c1 + 50)
    assert np.nanmin(fwd[win]) < np.interp(lm.s_c0, s, fwd) - 1.5
    assert np.nanmin(fwd[win]) / meas["v_min"] - 1 == pytest.approx(0.0, abs=0.01)
    old = _clamped_forward(T, p.kappa, 400.0, A1, car)
    assert np.nanmin(old[win]) / meas["v_min"] - 1 > 0.02         # the as-built miss
    # Off the table (slow) and on straights the pass is the as-built one.
    straight = np.zeros(N_KNOTS)
    Ts = np.full(N_KNOTS, 25.0)
    assert np.allclose(C.forward_profile(Ts, straight, 400.0, 10.0, car)[200:],
                       _clamped_forward(Ts, straight, 400.0, 10.0, car)[200:], rtol=1e-9,
                       equal_nan=True)


def test_offline_onset_rule_is_independent_and_agrees_with_measures(car):
    """R22 C: Q's offline onset uses its own brake demand and 3-step rule; on straight-line
    states they agree with measures.front_brake_demand / brake_onset (two implementations
    of protocol 5, cross-checked here, never shared)."""
    rng = np.random.default_rng(1)
    v = rng.uniform(0.0, 90.0, 400)
    ped = rng.uniform(-1.0, 1.0, 400)
    x = np.zeros((400, 8))
    x[:, 3], x[:, 7] = v, ped
    want = np.asarray(M.front_brake_demand(x, F1_2025, M.NP)[0], float)
    assert np.abs(C.brake_demand_1d(v, ped) - want).max() < 1e-9
    for _ in range(200):
        K = 60
        beta = np.clip(rng.normal(0.25, 0.25, K), 0.0, 1.2)
        beta[rng.integers(0, K, 6)] = 0.31
        s = np.cumsum(rng.uniform(1.0, 4.0, K))
        vv = rng.uniform(10.0, 80.0, K)
        apex = float(s[rng.integers(5, K)])
        bo = M.brake_onset(s, np.arange(K) * DT, beta, np.ones(K, bool), apex, vv)
        s_b, v_b, k_b = C.onset_1d(s[None], vv[None], beta[None], apex)
        assert k_b[0] == bo.k_b
        if bo.k_b >= 0:
            assert s_b[0] == pytest.approx(bo.s_b, abs=1e-9)
            assert v_b[0] == pytest.approx(bo.v_b, abs=1e-9)


def test_offline_prediction_matches_the_closed_loop_on_tight_corners(car, tight_runs):
    """R22 B + C: on the corners where the dry run's offline match failed (R7.5-R18.3), the
    offline v_e, v_min and onset match the closed loop's own (as the key) within protocol
    7's 2 points and one step of the predicted v_b."""
    profs, prm, run, _ = tight_runs
    for i, (p, (_, _, s0, v0)) in enumerate(zip(profs, TIGHT, strict=True)):
        meas = measured(run["x"][i], p.landmarks)
        pred = C.offline_prediction(np.asarray(prm.v_target[i]), p.kappa, p.landmarks, s0, v0,
                                    meas, car)
        res = {"e_ve": 0.0, "e_vmin": 0.0, "dd_b": 0.0, "v_b": meas["v_b"]}
        ok = C.offline_match(pred, res)
        assert all(ok.values()), (TIGHT[i], pred)
        assert abs(pred["dd_b"]) < 0.25 * abs(pred["v_b"]) * DT


def test_offline_match_tolerance_uses_the_predicted_v_b():
    pred = {"e_ve": 0.0, "e_vmin": 0.0, "dd_b": 0.0, "v_b": 80.0}       # one step: 4 m
    assert C.offline_match(pred, {"e_ve": 0.0, "e_vmin": 0.0, "dd_b": 3.0, "v_b": 20.0})["dd_b"]
    assert not C.offline_match(pred, {"e_ve": 0, "e_vmin": 0, "dd_b": 4.5, "v_b": 90.0})["dd_b"]
    nan_pred = {**pred, "dd_b": np.nan, "v_b": np.nan}
    assert not C.offline_match(nan_pred, {"e_ve": 0, "e_vmin": 0, "dd_b": 1.0, "v_b": 60.0})["dd_b"]


def test_offline_match_through_the_harness_on_the_pilot_minigrid(car):
    """R22 end to end: Q through the real harness (rollout, measures) on the pilot
    mini-grid, its harness measures against its offline prediction for every cell, in the
    batched path ``offline_match_runs`` uses (a cell's own run stands in for its key)."""
    from apexzero.interp import stimuli

    bank, cells = stimuli.pilot_minigrid()
    prm, aux = C.q_params(bank, cells, car, CV)
    arr = C.run_control(C.make_q_policy("Q"), prm, aux, bank, cells)
    rows = {"cell": [], "mode": [], "phase": [], "crashed": [], "s0": [], "v_e": [],
            "v_min": [], "d_b": [], "v_b": []}
    keys = {"cell": [], "key_status": [], "v_e_star": [], "v_min_star": [], "d_b_star": []}
    for i in range(len(arr["cell"])):
        c = cells[int(arr["cell"][i])]
        st = bank.stimuli[c.stim_id]
        traj = M.trajectory_from_rollout(arr, i, "det", source="control")
        mk = M.make_marks(st, bank.w_left[c.stim_id], bank.w_right[c.stim_id],
                          float(arr["s0"][i]), int(arr["W"][i]))
        m = M.response_metrics(traj, mk)
        for k, val in (("cell", c.cell_id), ("mode", "det"), ("phase", 0),
                       ("crashed", int(arr["term_step"][i]) >= 0), ("s0", float(arr["s0"][i])),
                       ("v_e", m.v_e), ("v_min", m.v_min), ("d_b", m.d_b), ("v_b", m.v_b)):
            rows[k].append(val)
        for k, val in (("cell", c.cell_id), ("key_status", "verified"), ("v_e_star", m.v_e),
                       ("v_min_star", m.v_min), ("d_b_star", m.d_b)):
            keys[k].append(val)
    pilot = [dataclasses.replace(c, tiers=("A",)) for c in cells]
    om = C.offline_match_runs({k: np.asarray(v) for k, v in rows.items()},
                              {k: np.asarray(v) for k, v in keys.items()}, bank, pilot, car, CV)
    assert om["n"] == len(cells) and om["passed"], om["fails"]
    assert max(abs(r["dd_b"]) / r["step"] for r in om["residuals"]) < 0.5


# ------------------------------------------------------------------------- R23: c1


@pytest.fixture(scope="module")
def c1_setup(car):
    """Two A1 braking corners and a simulated key table: each key brakes 20 m after Q's
    offline-predicted onset (the dry run's Q led the key by 12-26 m), one cell without a
    verified key and one whose key does not brake."""
    profs = [trapezoid(22.9, 90.0), trapezoid(13.1, 180.0)]
    bank = _Bank(profs)
    cells = [cell(0, 0, A1, 600.0), cell(1, 1, A1, 600.0), cell(2, 0, A1, 450.0),
             cell(3, 1, A1, 450.0)]
    q_rows = C.target_profiles(bank.kappa, car, CV)
    d_b_q = []
    for c in cells:
        x0 = c.initial_state(0)
        pred = C.offline_prediction(q_rows[c.stim_id], bank.kappa[c.stim_id],
                                    profs[c.stim_id].landmarks, float(x0[0]), float(x0[3]),
                                    {"v_e": 1.0, "v_min": 1.0, "d_b": 0.0}, car)
        d_b_q.append(pred["d_b"])
    keys = {"cell": np.arange(4), "key_status": np.array(["verified", "verified", "unsolved",
                                                           "verified"]),
            "d_b_star": np.array([d_b_q[0] - 20.0, d_b_q[1] - 20.0, d_b_q[2] - 20.0, np.nan])}
    return bank, cells, keys, d_b_q


def test_c1_shift_is_15m_plus_q_offline_lead_over_the_key(car, c1_setup):
    bank, cells, keys, d_b_q = c1_setup
    sh = C.c1_shifts(bank, cells, keys, car, CV)
    for cid in (0, 1):
        assert sh[cid]["status"] == "ok"
        assert sh[cid]["lead"] == pytest.approx(20.0, abs=1e-9)
        assert sh[cid]["shift"] == pytest.approx(35.0, abs=1e-9)
        assert sh[cid]["d_b_q"] == pytest.approx(d_b_q[cid], abs=1e-9)
    assert sh[2]["status"] == "no_verified_key" and sh[2]["shift"] == C.C1_SHIFT
    assert sh[3]["status"] == "key_flat" and sh[3]["shift"] == C.C1_SHIFT
    with pytest.raises(ValueError, match="R23"):
        C.q_params(bank, cells, car, CV, "c1")
    with pytest.raises(ValueError, match="R23"):
        C.q_params_for_kappa(bank.kappa, [S_C, S_C], car, CV, "c1")
    # One target row per (stimulus, shift); Q's rows are untouched.
    prm, aux = C.q_params(bank, cells, car, CV, "c1", shifts=sh)
    assert len(set(aux.tolist())) == 4 and prm.v_target.shape[0] == 4
    prm_q, aux_q = C.q_params(bank, cells, car, CV)
    assert prm_q.v_target.shape[0] == 2 and aux_q.tolist() == [0, 1, 0, 1]


def test_c1_is_15m_late_against_the_key_in_closed_loop(car, c1_setup):
    """R23: with the per-cell shift, c1's harness onset is 15 m (+- one step) after the
    key's; relative to Q it is about 35 m late here (15 m + Q's 20 m lead)."""
    bank, cells, keys, _ = c1_setup
    sh = C.c1_shifts(bank, cells[:2], keys, car, CV)
    prm, aux = C.q_params(bank, cells[:2], car, CV, "c1", shifts=sh)
    x0 = [c.initial_state(0) for c in cells[:2]]
    run = loop(C.make_q_policy("c1"), prm, bank.profiles, x0, aux, 400)
    prm_q, aux_q = C.q_params(bank, cells[:2], car, CV)
    run_q = loop(C.make_q_policy("Q"), prm_q, bank.profiles, x0, aux_q, 400)
    for i, c in enumerate(cells[:2]):
        lm = bank.profiles[c.stim_id].landmarks
        m, mq = measured(run["x"][i], lm), measured(run_q["x"][i], lm)
        step = m["v_b"] * DT
        assert m["d_b"] - keys["d_b_star"][i] == pytest.approx(-15.0, abs=step)
        assert mq["d_b"] - m["d_b"] == pytest.approx(35.0, abs=step)


# ------------------------------------------------------------------------- R24: c2


def test_c2_planted_law_check_logic(car, tables):
    """R24's check on simulated run tables: a c2 whose v_min ratio to Q is the offline
    prediction on capped cells and 1 elsewhere passes; 3 points off fails; a lock-limited
    cell (raw R <= lock radius) is listed, not judged, even when c2 crashes there."""
    profs = [trapezoid(9.4, 180.0), trapezoid(30.0, 90.0), trapezoid(7.5, 180.0)]
    bank = _Bank(profs)
    cells = [cell(i, i, A1, 600.0) for i in range(3)]
    r_lock = C.lock_radius(tables)
    assert 7.6 < r_lock < 8.4                      # 8.16 m: the model table's 'lock' rows
    q_rows = C.target_profiles(bank.kappa, car, CV)
    c2_rows = C.target_profiles(bank.kappa, car, CV, kappa_cap=C.C2_KAPPA_CAP)
    s = np.arange(N_KNOTS) * 2.0
    vmin_q, vmin_c2 = [], []
    for i, p in enumerate(profs):
        lm = p.landmarks
        win = (s >= lm.s_c0 - 50) & (s <= lm.s_c1 + 50)
        fq = C.forward_profile(q_rows[i], p.kappa, 400.0, A1, car)
        fc = C.forward_profile(c2_rows[i], p.kappa, 400.0, A1, car)
        vmin_q.append(np.nanmin(fq[win]))
        vmin_c2.append(np.nanmin(fc[win]))
    assert vmin_c2[0] / vmin_q[0] > 1.1 and vmin_c2[1] == pytest.approx(vmin_q[1])

    def table(vmin, crashed):
        return {"cell": np.arange(3), "phase": np.zeros(3, int), "mode": np.array(["det"] * 3),
                "v_min": np.asarray(vmin, float), "crashed": np.asarray(crashed, bool),
                "s0": np.full(3, 400.0)}

    q = table(vmin_q, [False] * 3)
    res = C.c2_planted_law(table(vmin_c2, [False, False, True]), q, bank, cells, car, CV, r_lock)
    assert res["passed"] and res["n_capped"] == 1 and res["n_uncapped"] == 1
    assert [r["cell"] for r in res["lock_limited"]] == [2]
    off = [vmin_c2[0] * 1.03, vmin_c2[1], vmin_c2[2]]
    assert not C.c2_planted_law(table(off, [False] * 3), q, bank, cells, car, CV, r_lock)["passed"]
    off = [vmin_c2[0], vmin_c2[1] * 1.03, vmin_c2[2]]
    assert not C.c2_planted_law(table(off, [False] * 3), q, bank, cells, car, CV, r_lock)["passed"]
    crash = C.c2_planted_law(table(vmin_c2, [True, False, False]), q, bank, cells, car, CV, r_lock)
    assert not crash["passed"] and crash["fails"][0]["crashed"]


def test_c2_planted_law_holds_in_closed_loop(car, tables):
    """R24 on synthetic corners: c2 and Q run, their v_min ratio matches the offline
    prediction within 2 points (capped R9.4) and is 1 within 2 points (uncapped R30)."""
    profs = [trapezoid(9.4, 90.0), trapezoid(30.0, 90.0)]
    bank = _Bank(profs)
    cells = [cell(i, i, A1, 600.0) for i in range(2)]
    x0 = [c.initial_state(0) for c in cells]
    tabs = {}
    for name in ("Q", "c2"):
        prm, aux = C.q_params(bank, cells, car, CV, name)
        run = loop(C.make_q_policy(name), prm, profs, x0, aux, 400)
        ms = [measured(run["x"][i], profs[i].landmarks) for i in range(2)]
        tabs[name] = {"cell": np.arange(2), "phase": np.zeros(2, int),
                      "mode": np.array(["det"] * 2), "v_min": np.array([m["v_min"] for m in ms]),
                      "crashed": run["term_step"] >= 0, "s0": np.full(2, float(x0[0][0]))}
    res = C.c2_planted_law(tabs["c2"], tabs["Q"], bank, cells, car, CV, C.lock_radius(tables))
    assert res["passed"], res["rows"]
    assert res["rows"][0]["measured"] > 1.1


# ------------------------------------------------------------------------- R26: c3


@pytest.fixture(scope="module")
def c3_setup(car):
    profs = [trapezoid(22.9, 90.0), trapezoid(13.1, 180.0), trapezoid(40.0, 90.0)]
    bank = _Bank(profs)
    cells = [cell(0, 0, A1, 600.0, n_phases=2), cell(1, 1, A1, 600.0, n_phases=2),
             cell(2, 2, 200 / 3.6, 250.0, n_phases=2),
             cell(3, 1, A1, 60.0)]                       # starts inside its braking distance
    return bank, cells


@pytest.fixture(scope="module")
def c3_tau(car, c3_setup):
    """tau_c_k1 on c3_setup (once: each call bisects the 1-D replay 24 times). The
    infeasible cell 3 is listed, not used, so cells[:3] give the same tau_c."""
    bank, cells = c3_setup
    return C.tau_c_k1(bank, cells, car, CV)


def test_c3_tau_c_is_the_smallest_with_k_le_1(car, c3_setup, c3_tau):
    bank, cells = c3_setup
    tau, det = c3_tau
    rows = {(r["cell"], r["phase"]): r for r in det["rows"]}
    assert det["verified"] and det["n_ok"] == 6 and det["n_infeasible"] == 1
    assert rows[(3, 0)]["status"] == "infeasible"
    ok = [r for r in det["rows"] if r["status"] == "ok"]
    assert det["threshold"] == pytest.approx(max(r["tau_needed"] for r in ok))
    assert tau == det["threshold"] + C.TAU_C_GUARD and det["guard"] == C.TAU_C_GUARD
    assert all(r["k_max"] <= 1.0 for r in ok)
    assert 0.6 < tau < 1.6
    # Smallest (at control-step resolution): just below the threshold some run's commanded
    # fraction exceeds 1.
    q_rows = C.target_profiles(bank.kappa, car, CV)
    runs = [(c, k) for c in cells[:3] for k in range(c.n_phases)]
    x0 = np.array([c.initial_state(k) for c, k in runs], float)
    args = (np.stack([q_rows[c.stim_id] for c, _ in runs]), x0[:, 0], x0[:, 3], x0[:, 7],
            np.full(len(runs), S_C), 400)
    assert np.nanmax(C._c3_kmax(*args, np.full(len(runs), tau), car, 2.0)) <= 1.0
    assert np.nanmax(C._c3_kmax(*args, np.full(len(runs), det["threshold"]), car, 2.0)) <= 1.0
    assert np.nanmax(C._c3_kmax(*args, np.full(len(runs), tau - 0.01), car, 2.0)) > 1.0


def test_c3_tau_c_guard_keeps_the_binding_trigger_clear_of_float32(car, c3_setup, c3_tau):
    """Review of R26 (controls finding 1): at the bisected threshold itself the binding run
    triggers within the bisection's resolution of the condition d_c <= tau v (~1e-5 m, below
    float32 resolution of s near the corner, 6.1e-5 m at 900 m), so the float32 harness can
    trigger it one step later, where k jumps far above 1. tau_c carries TAU_C_GUARD: the
    binding trigger is then ~guard v inside the condition, hundreds of float32 ulps."""
    bank, cells = c3_setup
    tau, det = c3_tau
    q_rows = C.target_profiles(bank.kappa, car, CV)
    need = {(r["cell"], r["phase"]): r["tau_needed"] for r in det["rows"] if r["status"] == "ok"}
    bind = [(c, k) for c in cells for k in range(c.n_phases)
            if need.get((c.cell_id, k), -1.0) >= det["threshold"] - 1e-9]
    x0 = np.array([c.initial_state(k) for c, k in bind], float)
    args = (np.stack([q_rows[c.stim_id] for c, _ in bind]), x0[:, 0], x0[:, 3], x0[:, 7],
            np.full(len(bind), S_C), 400)
    at_thr = C._c3_trigger_margin(*args, np.full(len(bind), det["threshold"]), car, 2.0)
    at_tau = C._c3_trigger_margin(*args, np.full(len(bind), tau), car, 2.0)
    ulp = float(np.spacing(np.float32(S_C)))
    assert np.nanmin(at_thr) < ulp                       # the knife edge the guard removes
    assert det["trigger_margin_min_m"] == pytest.approx(np.nanmin(at_tau))
    assert np.nanmin(at_tau) > 100 * ulp and np.nanmin(at_tau) > 0.5 * C.TAU_C_GUARD * 40.0
    # just below the threshold the binding run triggers a step later: k far above 1
    late = C._c3_kmax(*args, np.full(len(bind), det["threshold"] - 2e-6), car, 2.0)
    assert np.nanmax(late) > 1.0


def test_c3_is_a_pure_trigger_with_k_le_1_and_lawful_entry_in_closed_loop(car, c3_setup,
                                                                          c3_tau):
    """R26's pre-P3 checks on c3 at its tau_c: braking starts when d_c <= tau_c v (no
    "earlier of" rule, so it is later than Q's braking where Q brakes earlier), k <= 1, and
    the entry speed is Q's target within one step of braking."""
    bank, cells = c3_setup
    tau, _ = c3_tau                      # tau_c_k1(bank, cells[:3]) is the same value
    prm, aux = C.q_params(bank, cells[:3], car, CV, "c3", tau)
    x0 = [c.initial_state(0) for c in cells[:3]]
    run = loop(C.make_q_policy("c3"), prm, bank.profiles, x0, aux, 400)
    for i, c in enumerate(cells[:3]):
        lm = bank.profiles[c.stim_id].landmarks
        x, a = run["x"][i], run["a"][i]
        row = np.asarray(prm.v_target[aux[i]], float)
        assert run["term_step"][i] == -1
        k = C.c3_fraction(x[:-1, 0], x[:-1, 3], row, lm.s_c0, tau, car)
        assert np.nanmax(k) <= 1.0 + 2e-3                      # float32 closed loop
        first = int(np.nonzero(np.isfinite(k))[0][0])
        assert lm.s_c0 - x[first, 0] <= tau * x[first, 3] + 1e-6
        assert lm.s_c0 - x[first - 1, 0] > tau * x[first - 1, 3]
        assert a[first, 1] < 0.0 and (a[:first, 1] > 0.0).all()   # flat out until then
        v_e = np.interp(lm.s_c0, x[:, 0], x[:, 3])
        T_e = np.interp(lm.s_c0, np.arange(N_KNOTS) * 2.0, row)
        assert abs(v_e - T_e) <= np.interp(v_e, car.v, car.a_dec) * DT


def test_c3_tau_c_through_the_harness_on_the_pilot_minigrid(car):
    """R26 end to end: tau_c from the 1-D replay, c3 through the real harness at it, its
    commanded fraction read back from the harness traces (``c3_fraction``) stays <= 1."""
    from apexzero.interp import stimuli

    bank, cells = stimuli.pilot_minigrid()
    sel = [c for c in cells if bank.stimuli[c.stim_id].name.endswith("/L")]
    tau, det = C.tau_c_k1(bank, sel, car, CV)
    assert det["verified"] and det["n_ok"] == len(sel)
    prm, aux = C.q_params(bank, sel, car, CV, "c3", tau)
    arr = C.run_control(C.make_q_policy("c3"), prm, aux, bank, sel)
    for i in range(len(sel)):
        c = sel[i]
        W = int(arr["W"][i])
        x = arr["x"][i, :W + 1].astype(float)
        k = C.c3_fraction(x[:-1, 0], x[:-1, 3], np.asarray(prm.v_target[aux[i]], float),
                          bank.stimuli[c.stim_id].landmarks.s_c0, tau, car, bank.ds)
        assert np.isfinite(k).any() and np.nanmax(k) <= 1.0          # strict: the guard
    assert det["trigger_margin_min_m"] > 1e-2                          # cm, not float32 ulps
    # R26's pre-P3 checks as the dry run applies them (``c3_checks``), on these harness runs:
    # run table columns from the traces, every cell braking with a verified tier A key.
    n = len(arr["cell"])
    v_e = np.array([np.interp(bank.stimuli[int(arr["stim"][i])].landmarks.s_c0,
                              arr["x"][i, :int(arr["W"][i]) + 1, 0].astype(float),
                              arr["x"][i, :int(arr["W"][i]) + 1, 3].astype(float))
                    for i in range(n)])
    rt = {"cell": np.asarray(arr["cell"]), "phase": np.asarray(arr["phase"]),
          "mode": np.array(["det"] * n), "crashed": np.asarray(arr["term_step"]) >= 0,
          "crash_type": np.array([""] * n), "forced_onset": np.zeros(n, bool), "v_e": v_e}
    kt = {"cell": np.array([c.cell_id for c in sel]), "tiers": np.array(["A"] * len(sel)),
          "braking": np.ones(len(sel), bool), "key_status": np.array(["verified"] * len(sel))}
    ch = C.c3_checks(arr, rt, kt, bank, sel, car, CV, tau)
    assert (ch["k_le_1"], ch["entry"], ch["overspeed_crashes"]) == ("pass", "pass", "none")
    assert ch["n_judged"] == n and ch["n_triggered"] == n and ch["k_max"] <= 1.0
    assert ch["entry_max_steps"] <= 1.0
    # an entry 6 m/s fast (over one step: a_dec DT <= 3 m/s) fails entry; an overspeed crash
    # is counted; a forced onset or a flat (non-braking) key leaves the run out of the k and
    # entry checks
    fast = dict(rt, v_e=v_e + 6.0)
    assert C.c3_checks(arr, fast, kt, bank, sel, car, CV, tau)["entry"] == "fail"
    crash = dict(rt, crashed=np.r_[True, np.zeros(n - 1, bool)],
                 crash_type=np.array(["overspeed"] + [""] * (n - 1)))
    assert C.c3_checks(arr, crash, kt, bank, sel, car, CV, tau)["overspeed_crashes"] == "1 crashes"
    forced = dict(fast, forced_onset=np.ones(n, bool))
    out = C.c3_checks(arr, forced, kt, bank, sel, car, CV, tau)
    assert out["n_judged"] == 0 and out["n_forced_excluded"] == n
    assert out["entry"] == out["k_le_1"] == "no judged run"           # vacuous: not a pass
    flat = dict(kt, braking=np.zeros(len(sel), bool))
    assert C.c3_checks(arr, fast, flat, bank, sel, car, CV, tau)["n_judged"] == 0


# ------------------------------------------------------------------------- K, specs


def test_knn_fallback_logs_c_a_at_both_k():
    calls = []

    def run_k(c_a):
        def f(k):
            calls.append(k)
            return f"arrays k={k}", c_a[k]
        return f

    arr, rec = C.knn_fallback(run_k({1: 0.95, 5: 0.5}))
    # k = 1 passes: its arrays are used, and k = 5's C(a) is still logged (plan step 1;
    # review of the controls wiring, finding 4: it was None on this path)
    assert arr == "arrays k=1" and rec["k"] == 1 and not rec["fallback"]
    assert rec["C_a_k5"] == 0.5 and calls == [1, 5]
    calls.clear()
    arr, rec = C.knn_fallback(run_k({1: 0.95, 5: 0.5}), c_a_at=lambda k: calls.append(
        ("C(a) only", k)) or 0.75)
    assert arr == "arrays k=1" and rec["C_a_k5"] == 0.75 and calls == [1, ("C(a) only", 5)]
    arr, rec = C.knn_fallback(run_k({1: 8 / 9, 5: 8 / 9}))
    assert arr == "arrays k=5" and rec["fallback"]
    assert rec["C_a_k1"] == pytest.approx(8 / 9) and rec["C_a_k5"] == pytest.approx(8 / 9)
    arr, rec = C.knn_fallback(run_k({1: float("nan"), 5: 1.0}))
    assert rec["k"] == 5 and rec["C_a_k5"] == 1.0


def test_control_specs_follow_the_dry_run_rulings():
    spec = C.CONTROLS
    assert spec["T"].required == {"errors": ("within_tau",)}               # R20 T32
    assert set(spec["T"].reported) == {"termination", "regret", "crash_types"}   # R30 count
    assert C.T64.required == {"regret": ("zero",), "errors": ("within_tau",),
                              "termination": ("none",)} and C.T64_REGRET_TOL == 1e-6
    assert "T64" not in spec                                                # own process
    assert spec["c2"].required == {"P1a": (C.SAME_AS_Q,), "planted_law": ("pass",)}  # R24
    assert "P1b" in spec["c2"].reported and "P1b" not in spec["c2"].required
    assert "lock_crashes" in spec["c2"].reported                     # R30 count, R29(1)
    assert "P3" not in spec["c3"].required and spec["c3"].expected == {"P3": ("refute:M_idx",)}
    assert set(spec["c3"].required) == {"k_le_1", "entry", "overspeed_crashes"}  # R26
    assert spec["c1"].required == {"P2": ("confirm",), "P2b": ("onset_share>0.5",),
                                   "viability_typing": ("pass",)}               # R23, R28
    from apexzero.interp import analysis as A
    assert C.T64_REGRET_TOL == A.TAPE_REGRET_RTOL                           # one bound
    assert spec["K"].required == {"P4": ("class 2", "class 3")}            # R27 in the script


# ------------------------------------------------------------------------- R20: T64


@pytest.fixture(scope="module")
def t64_setup():
    """A corner strip and a straight; a float64 'key-like' tape: the lane keeper's
    closed-loop commands from a plain float64 step loop (the reference trajectory), on the
    exact float64 upcast of the float32 strip arrays."""
    profs = [trapezoid(30.0, 90.0), trapezoid(1e6, 1.0)]
    bank = _Bank(profs)
    cells = [cell(0, 0, 60.0, 200.0, W=160), cell(1, 1, 50.0, 300.0, W=80)]
    keeper = (60.0, np.array([-np.inf, -np.inf]))
    tapes, refs = [], []
    with jax.enable_x64(True):
        arrays = R._upcast(R.float32_bank_arrays(bank))
        step = jax.jit(lambda st, a, tr: racing.step_no_reset(st, a, tr, _EP))
        act = jax.jit(lambda st, tr: R._lane_keeper(
            (jnp.float64(keeper[0]), jnp.asarray(keeper[1])), None, st, tr, 0)[0])
        for c in cells:
            tr = jax.tree.map(lambda a, i=c.stim_id: jnp.asarray(a)[i], arrays)
            x0 = np.asarray(c.initial_state(0), np.float32).astype(np.float64)
            st = racing.EnvState(x=jnp.asarray(x0), prev_action=jnp.zeros(2), accel=jnp.zeros(2),
                                 ctx=jnp.asarray(R.default_ctx().astype(np.float64)),
                                 hidden_mu=jnp.float64(1.0), t=jnp.zeros((), jnp.int32),
                                 slow=jnp.float64(0.0), s0=jnp.float64(x0[0]))
            xs, acts, rew = [np.asarray(st.x)], [], []
            for _ in range(c.W):
                a = act(st, tr) + jnp.asarray([0.01 * np.sin(len(acts) / 7.0), 0.0])
                _, st, r, term, trunc, _ = step(st, a, tr)
                assert not bool(term)
                xs.append(np.asarray(st.x))
                acts.append(np.asarray(a))
                rew.append(float(r))
            tapes.append(np.stack(acts))
            refs.append((np.stack(xs), np.array(rew)))
    return bank, cells, tapes, refs


def test_t64_replays_a_float64_key_like_tape_exactly(t64_setup):
    """R20: the same harness code in float64 replays a float64 tape to its own trajectory
    (regret / |J*| and states at float64 rounding), where T32 drifts at float32's."""
    bank, cells, tapes, refs = t64_setup
    assert all(np.any(t != t.astype(np.float32)) for t in tapes)   # not float32 values
    runs = R.plan_runs(cells, bank, "det", 0, aux={0: 0, 1: 1})
    with jax.enable_x64(True):
        out = R.run_tape_rollouts_x64(tapes, bank, cells, runs)
    assert out["x"].dtype == np.float64 and out["reward"].dtype == np.float64
    gamma = 0.995
    for i, c in enumerate(cells):
        x_ref, r_ref = refs[i]
        assert out["term_step"][i] == -1
        np.testing.assert_allclose(out["x"][i, :c.W + 1], x_ref, rtol=1e-12, atol=1e-10)
        J_star = np.sum(gamma ** np.arange(c.W) * r_ref)
        J = np.sum(gamma ** np.arange(c.W) * out["reward"][i, :c.W])
        assert abs(J - J_star) / abs(J_star) <= 1e-12
    # T32: the same tape through the agents' float32 path (unchanged) is float32-exact only.
    t32 = R.run_rollouts(C.make_tape_policy(), C.tape_params(tapes), None, bank, cells, runs,
                         "full")
    assert t32["x"].dtype == np.float32
    d32 = max(np.abs(t32["x"][i, :c.W + 1] - refs[i][0]).max() for i, c in enumerate(cells))
    assert 1e-7 < d32 < 0.1                   # cm-level open-loop drift, as the dry run found


def test_t64_is_the_only_exemption_from_the_x64_assertion(t64_setup):
    """R20: the float64 path takes the tape control only, in a process with x64 on; the
    agents' path (any policy through run_rollouts or the compiled harness) still refuses
    float64."""
    from apexzero.agents.ppo import Actor, PPOConfig, RunningMeanStd

    bank, cells, tapes, _ = t64_setup
    runs = R.plan_runs(cells, bank, "det", 0, aux={0: 0, 1: 1})
    with pytest.raises(RuntimeError, match="float64 process"):
        R.run_tape_rollouts_x64(tapes, bank, cells, runs)          # x64 off here
    with pytest.raises(RuntimeError, match="x64"):
        C.tape_params(tapes, np.float64)
    cfg = PPOConfig(network="mlp", actor_width=32, mlp_layers=2, activation="tanh")
    params = Actor(cfg, 2, False).init(jax.random.PRNGKey(0), jnp.zeros(racing.OBS_DIM))
    rms = RunningMeanStd(jnp.zeros(racing.OBS_DIM), jnp.ones(racing.OBS_DIM), jnp.asarray(1.0))
    agent = R.agent_policy(cfg)
    batch, _ = R.make_batch(runs[:1], {c.cell_id: c for c in cells})
    arrays32 = R.bank_arrays(bank)                      # cached on the bank with x64 off
    with jax.enable_x64(True):
        with pytest.raises(RuntimeError, match="x64"):
            R.run_rollouts(agent, params, rms, bank, cells, runs, "full")
        with pytest.raises(RuntimeError, match="x64"):
            R.make_stimulus_rollout(agent, 10)(params, rms, arrays32, batch)
        with pytest.raises(RuntimeError, match="x64"):                  # T32's path too
            R.run_rollouts(C.make_tape_policy(), C.tape_params(tapes), None, bank, cells,
                           runs, "full")
        arrays = R._upcast(R.float32_bank_arrays(bank))
        with pytest.raises(ValueError, match="T64"):
            R._run_chunks(agent, params, None, arrays, cells, runs, "full", x64=True)
        with pytest.raises(ValueError, match="T64"):
            R._run_chunks(C.make_tape_policy(), C.tape_params(tapes, np.float64), rms, arrays,
                          cells, runs, "full", x64=True)


def test_t64_track_arrays_are_the_agents_float32_arrays(t64_setup):
    from apexzero.interp import stimuli

    bank, _, _, _ = t64_setup
    pilot, _ = stimuli.pilot_minigrid()
    for b in (bank, pilot):
        want = jax.tree.map(np.asarray, R.bank_arrays(b))
        got = R.float32_bank_arrays(b)
        with jax.enable_x64(True):
            got64 = R.float32_bank_arrays(b)
        for w, g, g64 in zip(jax.tree.leaves(want), jax.tree.leaves(got),
                             jax.tree.leaves(got64), strict=True):
            assert g.dtype == w.dtype == g64.dtype
            np.testing.assert_array_equal(g, w)
            np.testing.assert_array_equal(g64, w)
