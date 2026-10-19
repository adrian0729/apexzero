"""Answer keys (protocol 6): solver copies, pins, the recipe on one small validation cell, the
tolerance rules, active sets, the infeasibility bound, the cache and V1-V11's checks.

No agent is rolled out. Measures are a local test double (``measures`` is built in
parallel); the TO-OCP solves are validation-sized (a 200-step window on R40/90)."""

import dataclasses
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from apexzero.interp import answer_key as ak
from apexzero.interp.constants import BACKOFF_ALPHA_MAX, DT, kph
from apexzero.interp.measures import ModelTables, Trajectory
from apexzero.interp.stimuli import Cell, CornerSpec, Landmarks
from apexzero.ocp.min_time import OCPResult
from apexzero.ocp.window import WindowSolver, WindowSpec
from apexzero.vehicle.params import F1_2025

S_C = 600.0           # corner start of the test strips (a short layout: 700 knots)
N = 700


@pytest.fixture(autouse=True)
def _restore_x64():
    """Keys switch x64 on (as ``replay`` does); give other test modules their setting back."""
    prev = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", prev)


# ------------------------------------------------------------------------- doubles


@dataclass
class TCell(Cell):
    """Cell with the synthetic initial state (stimuli.Cell.initial_state is built in parallel)."""

    s_c: float = S_C

    def initial_state(self, k: int) -> np.ndarray:
        u = 0.0 if self.P is None else k * self.P / self.n_phases
        return np.array([self.s_c - self.D - u, 0, 0, self.v0, 0, 0, 0, 0], np.float32)


def strip(R: float, theta_deg: float, direction: int = 1, l_r: float = 20.0, width: float = 12.0):
    """Knot samples of a trapezoid corner at S_C (protocol 3.1 profile; no cap at R >= 13)."""
    spec = CornerSpec(R=R, theta_deg=theta_deg, direction=direction, width=width, l_r=l_r, s_c=S_C)
    u = np.arange(N) * 2.0 - S_C
    lr, lp = spec.l_r, spec.L_p
    prof = np.clip(np.where(u < lr, u / lr, np.where(u < lr + lp, 1.0, (2 * lr + lp - u) / lr)), 0, 1)
    k = direction * spec.kappa_peak * prof
    w = np.full(N, width / 2)
    L_c = spec.L_c
    lm = Landmarks(S_C, S_C + lr, S_C + L_c / 2, S_C + L_c - lr, S_C + L_c)
    return spec, k, w.copy(), w.copy(), lm


class FakeBank:
    """The two members of one mirror pair (stimuli.StimulusBank is built in parallel)."""

    ds = 2.0

    def __init__(self, R: float = 40.0, theta_deg: float = 90.0):
        rows = [strip(R, theta_deg, +1), strip(R, theta_deg, -1)]
        self.kappa = np.stack([r[1] for r in rows])
        self.w_left = np.stack([r[2] for r in rows])
        self.w_right = np.stack([r[3] for r in rows])
        self.stimuli = [SimpleNamespace(stim_id=i, name=f"syn/R{R:g}/T{theta_deg:g}/W12/test/{'LR'[i]}",
                                        landmarks=r[4], direction=r[0].direction, mirror_id=1 - i,
                                        spec=r[0], source={}, flags=(), geometry=None)
                        for i, r in enumerate(rows)]

    def track(self, i):
        import jax.numpy as jnp

        from apexzero.tracks.generator import strip_arrays, to_track

        ak._x64()
        tr = strip_arrays(*(jnp.asarray(a[i], jnp.float64) for a in (self.kappa, self.w_left, self.w_right)))
        return to_track(tr, name=self.stimuli[i].name)


def fake_metrics(s, vx, n, pedal, marks) -> dict:
    """Test double of measures.response_metrics for the fields the keys use (legacy pedal
    onset instead of the brake demand)."""
    lm = marks.landmarks
    s = np.maximum.accumulate(np.asarray(s, float))
    v_e = float(np.interp(lm.s_c0, s, vx))
    win = (s >= lm.s_c0 - 50) & (s <= lm.s_c1 + 50)
    k = np.flatnonzero(win)[np.argmin(vx[win])]
    brake = np.flatnonzero((pedal < -0.2) & (s < lm.s_apex))
    s_b = float(s[brake[0]]) if brake.size else np.nan
    sig = marks.direction or 1
    wl = np.interp(s, np.arange(len(marks.w_left)) * 2.0, marks.w_left)
    eta = sig * n / (wl - 1.0)
    corner = (s >= lm.s_c0) & (s <= lm.s_c1)
    apex = float(s[np.flatnonzero(corner)[np.argmax(eta[corner])]])
    return {"s_b": s_b, "d_b": lm.s_c0 - s_b, "v_b": float(np.interp(s_b, s, vx)) if brake.size else np.nan,
            "has_brake": bool(brake.size), "v_e": v_e, "v_min": float(vx[k]),
            "s_vmin_rel": float(s[k] - lm.s_apex), "line_apex_rel": apex - lm.s_apex}


@pytest.fixture
def doubles(monkeypatch):
    monkeypatch.setattr(ak, "key_metrics", lambda res, track, marks, *a, **k: fake_metrics(
        res.x[:, 0], res.x[:, 3], res.x[:, 1], res.x[:, 7], marks))
    monkeypatch.setattr(ak, "socp_metrics", lambda soc, marks, *a, **k: fake_metrics(
        soc.result.s, soc.result.z[:, 2], soc.result.z[:, 0], soc.result.z[:, 6], marks))


def tables() -> ModelTables:
    """Plain point-mass tables (the real ones come from measures.build_model_tables)."""
    v = np.arange(0.0, 88.0, 1 / 3.6)
    vl = np.arange(20.0, 316.0, 5.0) / 3.6
    return ModelTables(v_dec=v, a_dec=20.0 + 0.008 * v**2, pedal_dec=np.full_like(v, 0.5), v_lat=vl,
                       a_y_max=15.0 + 0.006 * vl**2, alpha_max=F1_2025.alpha_max)


def start(name, J, verified=True, **m):
    meas = {"d_b": 100.0, "v_e": 50.0, "v_min": 30.0, "s_vmin": 0.0, "line_apex": 0.0} | m
    return ak.StartRecord(name=name, J=J, success=verified, verified=verified, capped=False,
                          status="Solve_Succeeded", iterations=10, cpu_seconds=1.0, measures=meas)


def record(cell_id=0, kind="nominal", phase=0, J=50.0, R=13.0, theta=180.0, approach="A1", **kw):
    base = dict(key_id=f"k{cell_id}{kind}{phase}", kind=kind, status="verified", stim_id=0,
                stim_name="syn/test", cell_id=cell_id, phase=phase, direction=1, x0=np.zeros(8),
                W=400, spec={}, alpha_max=F1_2025.alpha_max, shrink=0.0, J=J,
                replay={"verified": True, "survives": True, "rel": 1e-12},
                labels={"R": R, "theta_deg": theta, "width": 12.0, "approach": approach})
    base.update(kw)
    return ak.KeyRecord(**base)


# ------------------------------------------------------------------------- solver copies, pins


def test_v10_solver_copies_are_exact():
    t0 = time.process_time()
    res = ak.v10_solver_copies()
    cpu = time.process_time() - t0
    assert res.passed, res.details
    assert max(res.details["min_time_copy"].values()) <= 1e-8
    assert res.details["window_copy"]["x"] <= 1e-10 and res.details["window_copy"]["J"] <= 1e-10
    assert res.details["a0_fixed"]["da0"] == 0.0 and res.details["a0_fixed"]["verified"]
    assert res.details["pin"]["z_pin"] <= 1e-9 and res.details["pin"]["t0"] == 0.0
    print(f"\nV10: {cpu:.1f} CPU-s")


def test_pinned_strip_lands_a_node_on_any_s0():
    spec, k, wl, wr, lm = strip(40.0, 90.0)
    track = FakeBank().track(0)
    for s0, back in ((401.37, 20.0), (7.3, 6.0), (0.4, 0.0)):
        cut = ak.pinned_strip(track, s0, lm.s_c1 + 100.0)
        i = cut.meta["i_pin"]
        assert cut.meta["s_offset"] + cut.s[i] == pytest.approx(s0, abs=1e-12)
        assert cut.s[i] == pytest.approx(back)
        s_abs = cut.meta["s_offset"] + cut.s
        assert np.max(np.abs(cut.kappa - ak._spline_at(track.kappa, track, s_abs))) == 0.0
        assert cut.ds == 2.0 and not cut.closed


def test_pinned_socp_holds_its_pin_and_vx_bound():
    track = FakeBank().track(0)
    lm = FakeBank().stimuli[0].landmarks
    x0 = np.array([451.13, -0.7, 0.005, 46.0, 0.1, 0.0, 0.01, 0.8])
    cap = 40.0
    soc = ak.socp_from_state(track, x0, lm.s_c1 + 120.0, vx_upper={lm.s_c0: cap})
    r = soc.result
    assert r.success, r.status
    assert r.s[0] == pytest.approx(x0[0], abs=1e-12) and r.t[0] == 0.0
    np.testing.assert_allclose(r.z[0], x0[1:], atol=1e-9)
    assert max(soc.pin_check.values()) <= 1e-9
    # No free lead-in in the NLP: the pin is node 0 (numerics m2).
    assert r.meta["i_pin"] == 0 and r.meta["s_offset"] == pytest.approx(x0[0])
    # s_c0 lies between nodes (600 - 451.13 is not a multiple of 2 m): the bound holds on the
    # speed interpolated at s_c0, not at the nearest node (numerics M2, V8).
    assert r.meta["vx_lin"] and not r.meta["vx_upper"]
    assert float(np.interp(lm.s_c0, r.s, r.z[:, 2])) <= cap + 1e-6
    assert r.lap_time == pytest.approx(r.t[-1])
    # The guess starts at the pin and runs past the s-OCP's end on the straight tail.
    xs, acts = ak.socp_guess(soc, track, x0, 300)
    assert xs.shape == (301, 8) and acts.shape == (300, 2)
    np.testing.assert_array_equal(xs[0], x0)
    assert xs[-1, 0] > r.s[-1] and np.all(np.diff(xs[:, 0]) > 0) and np.all(acts[-50:, 1] == 1.0)


def test_backoff_is_a_runtime_parameter_and_data():
    track = FakeBank().track(0)
    t_bo, p_bo = ak.backoff_problem(track, F1_2025)
    assert p_bo.alpha_max == pytest.approx(math.radians(8.5)) and p_bo.delta_max == F1_2025.delta_max
    np.testing.assert_allclose(t_bo.w_left, track.w_left - 0.35)
    np.testing.assert_allclose(t_bo.w_right, track.w_right - 0.35)
    np.testing.assert_array_equal(t_bo.kappa, track.kappa)
    solver = WindowSolver(WindowSpec(steps=4, radius=20.0), compile=False)
    centres = np.full(4, 300.0)
    pn = solver.parameters(track, centres, ak.key_ctx(), F1_2025)
    pb = solver.parameters(t_bo, centres, ak.key_ctx(), p_bo)
    assert pn.shape == pb.shape == (solver.n_param,)          # same compiled problem
    assert pb[-1] == pytest.approx(BACKOFF_ALPHA_MAX) and pn[-1] == F1_2025.alpha_max


# ------------------------------------------------------------------------- the recipe


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """R40/90 left at an A2-like approach (160 km/h, D = 200 m), W = 200: the nominal key
    (s-OCP + starts (1) and (3)) and the backed-off key warm from it. Two small solves."""
    mp = pytest.MonkeyPatch()
    mp.setattr(ak, "key_metrics", lambda res, track, marks, *a, **k: fake_metrics(
        res.x[:, 0], res.x[:, 3], res.x[:, 1], res.x[:, 7], marks))
    mp.setattr(ak, "socp_metrics", lambda soc, marks, *a, **k: fake_metrics(
        soc.result.s, soc.result.z[:, 2], soc.result.z[:, 0], soc.result.z[:, 6], marks))
    prev = jax.config.jax_enable_x64
    bank = FakeBank()
    cache = ak.KeyCache(tmp_path_factory.mktemp("keys"))
    cells = [TCell(cell_id=i, stim_id=i, pair_id=7, tiers=("V",), approach="A2", v0=kph(160.0),
                   D=200.0, W=200) for i in (0, 1)]

    def q_double(bank_, cell, x0, steps):           # an agent-free scripted start: the
        soc = ak.key_for(cell, "socp", cache=cache)  # s-OCP guess with a 0.3 s later brake
        xs, acts = ak.socp_guess(soc, bank_.track(cell.stim_id), x0, steps)
        return xs, np.vstack([acts[:1].repeat(6, 0), acts[:-6]])

    t0 = time.process_time()
    nom = ak.full_recipe(bank, cells[0], cache=cache, q_guess=q_double)
    t1 = time.process_time()
    bo = ak.warm_key(nom, bank, cells[0], "backoff", cache=cache)
    t2 = time.process_time()
    yield SimpleNamespace(bank=bank, cells=cells, cache=cache, nom=nom, bo=bo,
                          cpu=(t1 - t0, t2 - t1))
    mp.undo()
    jax.config.update("jax_enable_x64", prev)


def test_nominal_key_is_verified_and_complete(solved):
    nom = solved.nom
    assert nom.status == "verified", [(s.name, s.status) for s in nom.starts]
    assert [s.name for s in nom.starts][:2] == ["socp", "Q"]     # no neighbour: Q is added
    assert nom.x.shape == (201, 8) and nom.a.shape == (200, 2)
    assert nom.replay["verified"] and abs(nom.replay["rel"]) <= 1e-6
    assert nom.J == max(s.J for s in nom.starts if s.verified)
    assert nom.near_equal and set(nom.near_equal) <= {s.name for s in nom.starts if s.verified}
    assert set(nom.tau) == set(ak.START_MEASURES)
    assert all(nom.tau[m] >= f for m, f in ak.MULTI_BASIN_FLOORS.items())
    assert nom.measures["has_brake"] and nom.measures["v_min"] < nom.measures["v_e"]
    assert nom.active_steps.shape == (201, 5) and nom.lam_g is not None and nom.lam_x is not None
    assert nom.labels["R"] == 40.0 and nom.labels["approach"] == "A2"
    assert nom.socp_id and nom.cpu_seconds > 0
    print(f"\nnominal R40/90 W200: {solved.cpu[0]:.1f} CPU-s in the test process; "
          f"starts {[(s.name, round(s.cpu_seconds, 1), s.iterations) for s in nom.starts]}")


def test_backoff_key_is_slower_and_off_the_slip_limit(solved):
    nom, bo = solved.nom, solved.bo
    assert bo.status == "verified" and bo.starts[0].name == f"warm:{nom.key_id}"
    assert bo.alpha_max == pytest.approx(BACKOFF_ALPHA_MAX) and bo.shrink == 0.35
    assert ak._max_slip(bo.x, F1_2025) <= BACKOFF_ALPHA_MAX + 1e-6
    v9 = ak.v9_backoff([nom], [bo])
    assert v9.passed, v9.details
    assert bo.J < nom.J
    print(f"\nbacked-off R40/90 W200 (warm): {solved.cpu[1]:.1f} CPU-s")


def test_cached_keys_are_found_and_mirrored(solved):
    cache, bank, (left, right) = solved.cache, solved.bank, solved.cells
    again = ak.key_for(left, "nominal", cache=cache, bank=bank)
    np.testing.assert_array_equal(again.x, solved.nom.x)
    assert again.J == solved.nom.J and again.starts[0].name == "socp"
    assert ak.key_for(right, "nominal", cache=cache, bank=bank) is None     # V5 not passed
    cache.set_flag("mirror_ok", True)
    m = ak.key_for(right, "nominal", cache=cache, bank=bank)
    assert "mirrored" in m.flags and m.cell_id == 1 and m.direction == -1
    np.testing.assert_array_equal(m.x[:, 1], -solved.nom.x[:, 1])
    np.testing.assert_array_equal(m.a[:, 0], -solved.nom.a[:, 0])
    np.testing.assert_array_equal(m.x[:, 3], solved.nom.x[:, 3])
    cache.set_flag("mirror_ok", False)
    # A second recipe call is a cache hit (no solve).
    t0 = time.process_time()
    hit = ak.full_recipe(bank, left, cache=cache)
    assert time.process_time() - t0 < 5.0 and hit.key_id == solved.nom.key_id
    v1 = ak.v1_replay([solved.nom, solved.bo])
    assert v1.passed and v1.details["n_used"] == 2


def test_key_for_looks_up_the_cells_current_problem(solved):
    """Agent-free keys are found by the id of the cell's current problem, not by its alias:
    a key solved for the same cell at another W (V2's second horizon) must not be returned."""
    cache, bank, (left, right) = solved.cache, solved.bank, solved.cells
    assert ak.expected_key_id(bank, left, "nominal") == solved.nom.key_id
    assert ak.expected_key_id(bank, left, "backoff") == solved.bo.key_id
    assert ak.expected_key_id(bank, left, "socp") == solved.nom.socp_id
    assert ak.expected_key_id(bank, dataclasses.replace(left, W=None), "nominal") is None
    # The cell alias now names another window's key: key_for still returns the W = 200 key.
    other = dataclasses.replace(solved.nom, key_id="not-this-one", W=400)
    cache.put(other)
    assert cache.alias("nominal", f"cell{left.cell_id}_ph0_-") == "not-this-one"
    assert ak.key_for(left, "nominal", cache=cache, bank=bank).key_id == solved.nom.key_id
    assert ak.key_for(dataclasses.replace(left, W=400), "nominal", cache=cache, bank=bank) is None
    cache.set_flag("mirror_ok", True)
    m = ak.key_for(right, "nominal", cache=cache, bank=bank)
    assert m is not None and m.key_id == solved.nom.key_id + "m"
    cache.set_flag("mirror_ok", False)


# ------------------------------------------------------------------------- guesses


def test_retime_aligns_in_s():
    v = 50.0
    xs = np.zeros((101, 8))
    xs[:, 0] = 100.0 + v * DT * np.arange(101)
    xs[:, 3] = v
    src = SimpleNamespace(x=xs, a=np.tile([0.0, 0.5], (100, 1)))
    later = np.array([137.5, 0, 0, v, 0, 0, 0, 0])
    g, a = ak.retime(src, later, 20)
    np.testing.assert_allclose(g[:, 0], 137.5 + v * DT * np.arange(21), atol=1e-6)
    earlier = np.array([95.0, 0, 0, v, 0, 0, 0, 0])          # a phase u = 5 m before the source
    g, a = ak.retime(src, earlier, 10)
    np.testing.assert_allclose(g[:, 0], 95.0 + v * DT * np.arange(11), atol=1e-6)
    g, a = ak.retime(src, later, 150)                         # past the source: speed held
    assert g.shape == (151, 8) and a.shape == (150, 2)
    assert np.all(np.diff(g[:, 0]) > 0) and g[-1, 0] == pytest.approx(137.5 + v * DT * 150, abs=1e-6)


# ------------------------------------------------------------------------- tolerance (6.3)


def test_near_equal_set_and_logged_starts():
    starts = [start("a", 100.0), start("b", 100.0 - 0.5e-2), start("c", 100.0 - 0.05),
              start("d", 99.0), start("e", 101.0, verified=False)]
    members, logged = ak.near_equal(starts)
    assert members == ["a", "b"] and logged == ["c"]


def test_multi_basin_flags_and_tau():
    a = start("a", 1.0, v_e=50.0, d_b=100.0)
    b = start("b", 1.0, v_e=50.0 + kph(0.4), d_b=101.5)
    flag, spread, tau = ak.multi_basin([a, b], ["a", "b"])
    assert not flag and spread["d_b"] == pytest.approx(1.5) and tau["d_b"] == pytest.approx(1.5)
    assert tau["v_min"] == ak.MULTI_BASIN_FLOORS["v_min"]
    c = start("c", 1.0, d_b=103.0)
    assert ak.multi_basin([a, c], ["a", "c"])[0]
    flat = start("f", 1.0, d_b=np.nan)                        # one brakes, the other does not
    flag, spread, _ = ak.multi_basin([a, flat], ["a", "f"])
    assert flag and spread["d_b"] == np.inf


def test_onset_floor_and_agent_beats_key():
    assert ak.onset_floor(kph(310.0)) == pytest.approx(0.5 * 310 / 3.6 * 0.05)
    assert ak.onset_floor(10.0) == 1.0
    key = record(J=100.0)
    assert ak.agent_beats_key(key, {"G1": 100.0 + 2e-4, "G2": 100.0 + 5e-5, "G3": np.nan}) == ["G1"]
    assert ak.agent_beats_key(record(status="unsolved", J=np.nan), {"G1": 1e9}) == []


# ------------------------------------------------------------------------- active set, bound


def test_active_set_flags_around_v_min():
    _, k, wl, wr, lm = strip(13.0, 180.0)
    track = FakeBank(13.0, 180.0).track(0)
    t = np.arange(101) * DT
    x = np.zeros((101, 8))
    x[:, 0] = lm.s_c0 - 20 + 2.0 * np.arange(101)
    x[:, 3] = 40.0 - 20.0 * np.exp(-((t - 2.5) / 0.5) ** 2)    # v_min at t = 2.5 s
    x[:, 6] = 0.1
    k_v = int(np.argmin(x[:, 3]))
    x[k_v, 6] = F1_2025.delta_max - math.radians(1.0)          # lock-active at v_min
    x[10, 1] = 6.0 - 1.03 - 0.02                               # corridor-active, > 1 s before
    traj = Trajectory(source="to_ocp", t=t, x=x, valid=np.ones(101, bool))
    act = ak.active_set(traj, track, landmarks=lm)
    assert act.steps[k_v, 2] and act.lock_limited
    assert act.steps[10, 3] and not act.corridor_limited
    assert act.steps.shape == (101, 5)


def test_infeasibility_bound_certifies_only_impossible_approaches():
    track = FakeBank(13.0, 180.0).track(0)
    lm = FakeBank(13.0, 180.0).stimuli[0].landmarks
    tb = tables()
    hot = ak.infeasibility_bound(track, np.array([S_C - 30.0, 0, 0, 86.0, 0, 0, 0, 1]), lm, tb)
    assert hot.infeasible and hot.v_reach_apex > hot.v_cap
    calm = ak.infeasibility_bound(track, np.array([S_C - 600.0, 0, 0, 86.0, 0, 0, 0, 1]), lm, tb)
    assert not calm.infeasible and calm.v_reach_apex == 0.0
    # The repo's min-curvature line wraps the hairpin round its inner edge (R_in = 8 m); the
    # bound keeps the widest of the candidate lines, so it stays relaxed.
    assert hot.rho_lines["audit_min_curvature"] < 9.0
    assert hot.rho_ub == pytest.approx(1.1 * max(hot.rho_lines.values())) and hot.rho_ub > 1.1 * 13.0


# ------------------------------------------------------------------------- layout


def test_layout_a2_sets_D_from_the_onset_and_steps_it_into_band(monkeypatch):
    bank = FakeBank()
    cells = [TCell(cell_id=0, stim_id=0, pair_id=1, tiers=("A",), approach="A2", v0=40.0, D=None),
             TCell(cell_id=1, stim_id=0, pair_id=2, tiers=("A",), approach="A1", v0=86.0, D=600.0)]
    d_b0 = 120.0

    def fake_socp(prob, cache, s_end=None, vx_upper=None):
        D = prob.landmarks.s_c0 - prob.x0[0]
        s = prob.x0[0] + np.arange(400) * 2.0
        res = OCPResult(s=s, z=np.zeros((400, 7)), u=np.zeros((400, 2)), t=np.arange(400) * 0.04,
                        lap_time=16.0, success=True, status="ok", iterations=1, solve_seconds=0.0,
                        kappa=np.zeros(400), w_left=np.full(400, 6.0), w_right=np.full(400, 6.0),
                        dual_left=np.zeros(400), dual_right=np.zeros(400))
        res.z[:, 2] = 50.0
        # v_b grows with the run-up beyond the onset: 40 m/s at D = d_b0, +2% per 10 m.
        m = {"has_brake": True, "d_b": d_b0, "v_min": 20.0, "v_b": prob.x0[3] * (1 + 0.002 * (D - d_b0))}
        return ak.SocpKey("k", 0, prob.cell.cell_id, 0, prob.x0, 0.0, res, measures=m)

    monkeypatch.setattr(ak, "_socp_for", fake_socp)
    from apexzero.interp import rollout

    monkeypatch.setattr(rollout, "sight_speed", lambda s, vx, s_c0: float(vx[0]), raising=False)
    out = ak.layout_a2(bank, cells, tables())
    a2 = out[0]
    # D = 120 + 25 gives v_b = +5% (in band); the A1 cell keeps its D.
    assert a2.D == pytest.approx(d_b0 + 25.0) and "braking" in a2.flags
    assert out[1].D == 600.0 and out[1].W in (400, 800) and out[0].W in (400, 800)
    assert a2.P == pytest.approx(50.0 * DT)
    # A margin that puts v_b outside +-10% is stepped back by 10 m until in band.
    monkeypatch.setattr(ak, "A2_D_MARGIN", 85.0)
    out = ak.layout_a2(bank, cells[:1], tables())
    assert out[0].D == pytest.approx(d_b0 + 85.0 - 40.0)      # v_b +17% -> +9% after 4 steps


def test_windows_and_stages_follow_the_socp_time_line():
    lm = Landmarks(1000.0, 1020.0, 1050.0, 1080.0, 1100.0)
    s = np.arange(0.0, 1800.0, 2.0) + 400.0
    v = np.full(len(s), 80.0)

    def soc(speed):
        z = np.zeros((len(s), 7))
        z[:, 2] = speed
        t = (s - s[0]) / speed
        res = OCPResult(s=s, z=z, u=np.zeros((len(s), 2)), t=t, lap_time=t[-1], success=True,
                        status="", iterations=0, solve_seconds=0, kappa=0 * s, w_left=0 * s + 6,
                        w_right=0 * s + 6, dual_left=0 * s, dual_right=0 * s)
        return ak.SocpKey("k", 0, 0, 0, np.zeros(8), s[-1], res)

    fast, slow = soc(v), soc(np.full(len(s), 40.0))
    assert ak.window_length(fast, lm) == 800          # 700 m at 80 m/s = 8.75 s > 5 s
    lm_near = Landmarks(500.0, 510.0, 520.0, 530.0, 540.0)
    assert ak.window_length(fast, lm_near) == 400     # 140 m in 1.75 s
    # Stages ending before s_c1 + 300 = 1400 m are skipped: 200 steps reach 1200 m at 80 m/s.
    assert ak.stages_for(800, fast, lm) == (400, 800)
    assert ak.stages_for(800, slow, lm) == (800,)                 # 400 steps reach only 1200 m
    assert ak.stages_for(400, fast, lm_near) == (200, 400)
    assert ak.stages_for(600, fast, lm) == (400, 600)


# ------------------------------------------------------------------------- cache


def test_cache_roundtrip_hash_and_aliases(tmp_path):
    cache = ak.KeyCache(tmp_path)
    rec = record(x=np.arange(16.0).reshape(2, 8), a=np.ones((1, 2)), lam_g=np.arange(3.0),
                 lam_x=np.arange(4.0), active_steps=np.zeros((2, 5), bool), pair_id=3,
                 starts=[start("socp", 1.0), start("Q", np.nan, verified=False)],
                 measures={"v_e": 1.0, "d_b": np.nan}, tau={"d_b": 1.0}, spread={"d_b": 0.0})
    cache.put(rec)
    back = cache.get("nominal", rec.key_id)
    np.testing.assert_array_equal(back.x, rec.x)
    assert np.isnan(back.measures["d_b"]) and np.isnan(back.starts[1].J)
    assert back.starts[0] == rec.starts[0] and back.labels == rec.labels
    h = cache.content_hash()
    cache.put(back)                                   # rewritten files, same content
    assert cache.content_hash() == h
    assert cache.alias("nominal", "cell0_ph0_-") == rec.key_id
    assert cache.alias("nominal", "pair3_dir+1_ph0_-") == rec.key_id
    rec.J = 51.0
    cache.put(rec)
    assert cache.content_hash() != h
    with pytest.raises(ValueError):
        cache.get("nonsense", "x")


def test_key_id_is_deterministic_and_sensitive():
    p, spec = F1_2025, WindowSpec(steps=400)
    x0 = np.array([400.0, 0, 0, 44.4, 0, 0, 0, 0])
    a = ak.key_id("nominal", "row", x0, 400, p, spec, {"shrink": 0.0})
    assert a == ak.key_id("nominal", "row", x0.copy(), 400, p, spec, {"shrink": 0.0}) and len(a) == 24
    x1 = x0.copy()
    x1[0] = np.nextafter(x0[0], 1e9)
    assert ak.key_id("nominal", "row", x1, 400, p, spec, {"shrink": 0.0}) != a
    assert ak.key_id("backoff", "row", x0, 400, p._replace(alpha_max=BACKOFF_ALPHA_MAX), spec,
                     {"shrink": 0.35}) != a
    assert ak.key_id("nominal", "row", x0, 800, p, dataclasses.replace(spec, steps=800),
                     {"shrink": 0.0}) != a


def test_mirror_key_mirrors_states_and_corridor_flags():
    steps = np.zeros((2, 5), bool)
    steps[0, 3] = True
    rec = record(x=np.array([[1, 2, 3, 4, 5, 6, 7, 8.0]] * 2), a=np.array([[0.3, -0.2]]),
                 active_steps=steps, lam_g=np.ones(2))
    m = ak.mirror_key(rec)
    np.testing.assert_array_equal(m.x[0], [1, -2, -3, 4, -5, -6, -7, 8])
    np.testing.assert_array_equal(m.a[0], [-0.3, -0.2])
    assert m.active_steps[0, 4] and not m.active_steps[0, 3] and m.lam_g is None
    assert m.direction == -1 and "mirrored" in m.flags and rec.direction == 1


# ------------------------------------------------------------------------- validation rules


def test_band_from_costs():
    # 0.10 s is reached at e = 0.075 + 0.025 * 0.01 / 0.04 = 8.125%, rounded to 8.0%.
    assert ak.band_from_costs({0.05: 0.05, 0.075: 0.09, 0.10: 0.13}) == (pytest.approx(0.08), True)
    assert ak.band_from_costs({0.05: 0.2, 0.075: 0.3, 0.10: 0.4})[0] == pytest.approx(0.05)
    assert ak.band_from_costs({0.05: 0.01, 0.075: 0.02, 0.10: 0.03}) == (pytest.approx(0.10), False)


def test_v1_v4_v6_v7_v11_on_records():
    good = record(cell_id=1, R=13.0, active={"slip_limited": True, "lock_limited": False},
                  measures={"v_e": 30.0, "v_min": 14.0, "d_b": 150.0})
    bad = record(cell_id=2, replay={"verified": False, "survives": True, "rel": 1e-3})
    unsolved = record(cell_id=3, status="unsolved", J=np.nan)
    v1 = ak.v1_replay([good, bad, unsolved])
    assert not v1.passed and v1.details["failed"] == [bad.key_id] and unsolved.key_id in v1.details["not_used"]
    assert ak.v11_active_set([good]).passed
    lock = dataclasses.replace(good, active={"slip_limited": True, "lock_limited": True})
    assert not ak.v11_active_set([lock]).passed and not ak.v11_active_set([]).passed
    mb = dataclasses.replace(good, multi_basin=True, spread={"d_b": 3.0})
    v4 = ak.v4_spread([mb, good])
    assert v4.passed and v4.details["multi_basin_cells"] == [1]
    tau = dict(ak.MULTI_BASIN_FLOORS)
    small = record(cell_id=4, R=13.0, theta=90.0, measures={"v_e": 31.0, "v_min": 15.0, "d_b": 150.0},
                   tau=tau)
    large = record(cell_id=5, R=16.4, theta=90.0, measures={"v_e": 30.0, "v_min": 17.0, "d_b": 140.0},
                   tau=tau)
    v6, _ = ak.v6_monotone([small, large])
    assert not v6.passed and [v["measure"] for v in v6.details["violations"]] == ["v_e"]
    ok = dataclasses.replace(large, measures={"v_e": 31.0 - 0.1, "v_min": 17.0, "d_b": 140.0})
    assert ak.v6_monotone([small, ok])[0].passed                 # within tau (0.14 m/s)
    ph = [record(cell_id=9, kind="nominal", phase=0, measures={"v_e": 30.0, "v_min": 14.0, "d_b": 150.0}),
          record(cell_id=9, kind="phase", phase=1, measures={"v_e": 30.1, "v_min": 14.0, "d_b": 151.0}),
          record(cell_id=9, kind="phase", phase=2, measures={"v_e": 30.0, "v_min": 14.1, "d_b": 149.0})]
    assert ak.v7_phase(ph).passed
    ph[1].measures["v_e"] = 30.0 + kph(0.6)
    assert not ak.v7_phase(ph).passed


def test_no_return_from_viability_records():
    def viab(s, status):
        return record(kind="viability", status=status, anchor={"s_anchor": s - 4.0, "s_state": s})

    x = np.zeros((50, 8))
    x[:, 0] = np.arange(50) * 4.0
    x[:, 3] = 80.0
    run = Trajectory(source="agent", t=np.arange(50) * DT, x=x, valid=np.ones(50, bool))
    key = Trajectory(source="to_ocp", t=np.arange(50) * DT, x=x * [1, 1, 1, 0.9, 1, 1, 1, 1],
                     valid=np.ones(50, bool))
    nr = ak.no_return([viab(120.0, "verified"), viab(40.0, "verified"), viab(160.0, "unsolved")], run, key)
    assert nr.s == 160.0 and nr.certified and nr.dv == pytest.approx(8.0) and nr.s_last_viable == 120.0
    nr = ak.no_return([viab(40.0, "unsolved")], run, key)
    assert nr.s == 40.0 and not nr.certified and np.isnan(nr.s_last_viable)
    nr = ak.no_return([viab(40.0, "verified"), viab(90.0, "verified")], run, key)
    assert nr.s is None and nr.s_last_viable == 90.0


def test_project_budget_applies_the_cut_rules():
    timings = {"socp": 30.0, "nominal": 350.0, "warm": 80.0, "null": 40.0, "anchored": 120.0,
               "viability": 60.0, "rollout_sps": 20_000}
    counts = {"nominal_A": 74, "nominal_E1": 30, "null": 28, "backoff": 66, "phase": 60,
              "anchored_G": 96, "viability": 200, "layout_socp": 33, "v8_socp": 36,
              "E1_twins": 15, "E2": 39, "E3": 28, "tierD_socp": 140, "tierC": 23, "recovery": 12}
    out = ak.project_budget(timings, counts)
    assert out["before"]["P0"] == pytest.approx(sum(
        counts[k] * timings[ak.BUDGET_ITEMS[k][0]] for k in counts if ak.BUDGET_ITEMS[k][1] == "P0") / 3600)
    assert out["cuts"] == []
    slow = ak.project_budget(dict(timings, nominal=2000.0, rollout_sps=9_000), counts)
    assert len(slow["cuts"]) == 3 and slow["after"]["P2"] == 0.0
    assert slow["items"]["phase"]["n"] == 60 - ak.DEFAULT_CUTS["phase"]
    assert slow["items"]["nominal_A"]["n"] == 74 - 6


# ------------------------------------------------------------------------- plumbing (no IPOPT)


def fake_socp(prob, cache, s_end=None, vx_upper=None):
    """A braking profile from the problem's x0 to the strip end (no solve)."""
    x0, lm = prob.x0, prob.landmarks
    s = np.arange(x0[0], 2.0 * (N - 1), 2.0)
    v_min, s_b = 30.0, lm.s_c0 - 80.0
    vx = np.where(s < s_b, x0[3], np.where(s < lm.s_apex, x0[3] + (v_min - x0[3]) * (s - s_b)
                                           / (lm.s_apex - s_b), np.minimum(v_min + 0.2 * (s - lm.s_apex), 80.0)))
    z = np.zeros((len(s), 7))
    z[:, 2] = vx
    z[:, 6] = np.where((s >= s_b) & (s < lm.s_apex), -0.6, 1.0)
    t = np.r_[0.0, np.cumsum(2.0 * np.diff(s) / (vx[1:] + vx[:-1]))]
    res = OCPResult(s=s, z=z, u=np.zeros((len(s), 2)), t=t, lap_time=t[-1], success=True,
                    status="Solve_Succeeded", iterations=1, solve_seconds=0.0, kappa=0 * s,
                    w_left=0 * s + 6, w_right=0 * s + 6, dual_left=0 * s, dual_right=0 * s)
    soc = ak.SocpKey(f"socp{prob.cell.cell_id}", prob.cell.stim_id, prob.cell.cell_id, prob.phase,
                     x0, s[-1], res)
    soc.measures = fake_metrics(s, vx, z[:, 0], z[:, 6], SimpleNamespace(
        landmarks=lm, direction=1, w_left=np.full(N, 6.0)))
    return soc


@pytest.fixture
def fake_solves(monkeypatch, doubles):
    from apexzero.ocp.window import WindowResult

    calls, outcome = [], {}

    def run_start(prob, name, first_guess, extend, stages, max_iter):
        calls.append(SimpleNamespace(name=name, stages=stages, max_iter=max_iter, kind=prob.kind,
                                     a0=prob.a0_fixed, W=prob.W, shrink=prob.shrink,
                                     alpha_max=prob.p.alpha_max, spec=prob.spec))
        J, ver = next((v for k, v in outcome.items() if name.startswith(k)), (10.0, True))
        xs, acts = first_guess
        res = WindowResult(x=np.asarray(xs[: prob.W + 1]), a=np.asarray(acts[: prob.W]), value=J,
                           success=ver, status="Solve_Succeeded" if ver else "Infeasible_Problem_Detected",
                           iterations=3, solve_seconds=0.0, rounds=1,
                           meta={"lam": (np.zeros(3), np.zeros(4)), "rounds": [], "cpu_seconds": 0.01,
                                 "replay": {"verified": ver, "survives": True, "rel": 0.0}})
        meas = ak._start_measures(prob.metrics(res)) if ver else {}
        return ak.StartRecord(name=name, J=J if ver else np.nan, success=ver, verified=ver,
                              capped=False, status=res.status, iterations=3, cpu_seconds=0.01,
                              measures=meas), res

    monkeypatch.setattr(ak, "_run_start", run_start)
    monkeypatch.setattr(ak, "_socp_for", fake_socp)
    monkeypatch.setattr(ak, "run_metrics", lambda run, marks, *a, **k: fake_metrics(
        run.x[:, 0], run.x[:, 3], run.x[:, 1], run.x[:, 7], marks))
    return SimpleNamespace(calls=calls, outcome=outcome)


def test_recipe_plumbing_without_ipopt(fake_solves, tmp_path):
    f = fake_solves
    bank, cache = FakeBank(), ak.KeyCache(tmp_path)
    cell = TCell(cell_id=0, stim_id=0, pair_id=3, tiers=("A",), approach="A2", v0=40.0, D=200.0, W=400)
    q = []

    def q_guess(bank_, c, x0, steps):
        q.append(steps)
        return ak.socp_guess(fake_socp(SimpleNamespace(x0=x0, landmarks=bank_.stimuli[0].landmarks,
                                                       cell=c, phase=0), None), bank_.track(0), x0, steps)

    # No neighbour: the largest R of its series gets the Q start.
    nb = ak.full_recipe(bank, cell, q_guess=q_guess, force=True)
    assert [s.name for s in nb.starts] == ["socp", "Q"] and q == [400] and nb.status == "verified"
    assert f.calls[0].stages == (400,) or f.calls[0].stages[-1] == 400
    # A neighbour that agrees: no Q; one that disagrees by > 1e-4 in J: Q.
    f.calls.clear()
    rec = ak.full_recipe(bank, cell, neighbour=nb, q_guess=q_guess, force=True)
    assert [s.name for s in rec.starts] == ["socp", f"neighbour:{nb.key_id}"]
    f.outcome["neighbour"] = (10.01, True)
    rec = ak.full_recipe(bank, cell, neighbour=nb, q_guess=q_guess, force=True)
    assert [s.name for s in rec.starts][-1] == "Q" and rec.J == 10.01
    assert rec.near_equal == [f"neighbour:{nb.key_id}"] and rec.logged == ["socp", "Q"]   # 1e-3 off
    # Every start fails: one retry at max_iter 3000, then 'unsolved' (the bound does not certify).
    f.outcome.clear()
    f.outcome.update({"socp": (0, False), "Q": (0, False), "retry": (0, False)})
    f.calls.clear()
    rec = ak.full_recipe(bank, cell, q_guess=q_guess, tables=tables(), force=True)
    assert [c.name for c in f.calls] == ["socp", "Q", "retry"] and f.calls[-1].max_iter == 3000
    assert rec.status == "unsolved" and rec.bound and not rec.bound["infeasible"]
    # An imposed cell the bound certifies is never solved.
    f.outcome.clear()
    f.calls.clear()
    hb = FakeBank(13.0, 180.0)
    imposed = TCell(cell_id=5, stim_id=0, pair_id=9, tiers=("V",), approach="imposed", v0=86.0, D=30.0)
    rec = ak.full_recipe(hb, imposed, tables=tables(), cache=cache)
    assert rec.status == "infeasible_bound" and not f.calls and rec.bound["infeasible"]
    # Backed-off key: a failed warm start falls back to the full recipe on the backed-off problem.
    f.outcome["warm"] = (0, False)
    bo = ak.warm_key(nb, bank, cell, "backoff", q_guess=q_guess)
    assert [s.name for s in bo.starts][:2] == [f"warm:{nb.key_id}", "socp"] and bo.status == "verified"
    assert "warm start failed: full recipe" in bo.flags
    assert {(c.shrink, c.alpha_max) for c in f.calls[-3:]} == {(0.35, BACKOFF_ALPHA_MAX)}
    f.outcome.clear()
    # Anchored key: solved from x_{L-1} with a_0 fixed to the run's a_{L-1}.
    run = Trajectory(source="agent", t=np.arange(401) * DT, x=nb.x.copy(), valid=np.ones(401, bool),
                     a=nb.a + 0.01)
    an = ak.anchored_key(bank, cell, nb, "G9", run, W=200, snapshot="final")
    L = an.anchor["step"]
    m = nb.measures
    s_L = bank.stimuli[0].landmarks.s_c0 - (m["d_b"] + m["v_b"] * 1.0)
    assert an.kind == "anchored" and nb.x[L - 1, 0] < s_L <= nb.x[L, 0]
    np.testing.assert_array_equal(f.calls[-1].a0, run.a[L - 1])
    np.testing.assert_array_equal(an.x0, nb.x[L - 1])
    no_brake = dataclasses.replace(nb, measures=dict(nb.measures, has_brake=False))
    assert ak.anchored_key(bank, cell, no_brake, "G9", run) is no_brake
    # Viability keys at the targets before the crash.
    crashed = dataclasses.replace(run, term_step=int(np.searchsorted(run.x[:, 0], 640.0)))
    vks = ak.viability_keys(bank, cell, crashed, nb, bo, agent="G9")
    assert [v.anchor["target"] for v in vks] == ["s_b", "s_c0-50", "s_c0", "s_a0"]   # all < 640 m
    late = dataclasses.replace(run, term_step=int(np.searchsorted(run.x[:, 0], 610.0)))
    assert [v.anchor["target"] for v in ak.viability_keys(bank, cell, late, nb, bo)] == \
        ["s_b", "s_c0-50", "s_c0"]
    assert all(v.W == 200 and v.kind == "viability" for v in vks)
    assert [c.name.split(":")[0] for c in f.calls[-2:]] == ["nominal", "backoff"]
    # Null and actuation keys.
    null = ak.null_key(bank, cell)
    assert null.kind == "null" and [s.name for s in null.starts] == ["socp"]
    act = ak.actuation_key(bank, cell, nominal=nb)
    assert act.spec["gamma"] == 1.0 and act.spec["action_cost"] == 1e-4
    assert [s.name for s in act.starts] == ["socp", f"nominal:{nb.key_id}"]
    # V5 on the pair sets the mirror flag; key_for then mirrors.
    right = TCell(cell_id=1, stim_id=1, pair_id=3, tiers=("A",), approach="A2", v0=40.0, D=200.0, W=400)
    v5 = ak.v5_mirror(bank, [cell, right], cache)
    assert v5.passed and cache.get_flag("mirror_ok") is True
    # Phase keys through key_for(solve=True): warm from the nominal.
    cell_p = dataclasses.replace(cell, P=4.0)
    ph = ak.key_for(cell_p, "phase", 2, cache=cache, bank=bank, solve=True)
    assert ph.kind == "phase" and ph.phase == 2 and ph.x0[0] == pytest.approx(400.0 - 2.0)
    assert ak.key_for(cell_p, "phase", 2, cache=cache, bank=bank).key_id == ph.key_id
    # V2, V3 and V8 orchestration (identical fake solutions: no gaps, no time cost).
    v2 = ak.v2_horizon(bank, [cell])
    assert v2.passed and v2.details["cells"][0]["windows"] == (400, 800)
    v3 = ak.v3_objective(bank, [cell])
    assert v3.passed and v3.details["cells"][0]["objective_e_ve"] == 0.0
    v8, b = ak.v8_band(bank, [cell])
    assert v8.passed and b == 0.10 and not v8.details["reached"]
    # The 6.3 re-solve keeps the old key unless an agent-seeded start beats it.
    same, dev = ak.resolve_beaten_key(bank, cell, nb, {"G9": run})
    assert same is nb and not dev["replaced"]
    f.outcome["agent"] = (10.5, True)
    new, dev = ak.resolve_beaten_key(bank, cell, nb, {"G9": run}, cache)
    assert dev["replaced"] and new.J == 10.5 and new.key_id == nb.key_id
    assert [s.name for s in new.starts][-1] == "agent:G9"


def test_series_are_solved_in_descending_R():
    stim = [SimpleNamespace(spec=CornerSpec(R=r, theta_deg=90.0, direction=1), name=f"R{r}")
            for r in (13.0, 40.0, 22.9)]
    stim.append(SimpleNamespace(spec=CornerSpec(R=40.0, theta_deg=180.0, direction=1), name="h"))
    bank = SimpleNamespace(stimuli=stim)
    cells = [TCell(cell_id=i, stim_id=i, pair_id=i, tiers=("A",), approach="A1", v0=86.0, D=600.0)
             for i in range(4)]
    series = ak.series_of(bank, cells)
    assert sorted([[c.stim_id for c in s] for s in series]) == [[1, 2, 0], [3]]


def test_a_non_finite_guess_is_not_solved():
    bank = FakeBank()
    cell = TCell(cell_id=0, stim_id=0, pair_id=3, tiers=("A",), approach="A2", v0=40.0, D=200.0, W=200)
    prob = ak._problem(bank, cell, "nominal")
    prob.W = 200
    xs = np.zeros((201, 8))
    acts = np.full((200, 2), np.nan)
    sr, res = ak._run_start(prob, "Q", (xs, acts), None, (200,), 800)
    assert res is None and not sr.verified and sr.status.startswith("non-finite")


# ------------------------------------------------------------------------- review regressions


def test_v6_compares_keys_at_the_same_approach_state():
    """Numerics M3: d_b* is compared only at a fixed (v0, D); at A2 (v0 = v_typ per R bin,
    D from the layout) v_e* and v_min* are still compared; equal R and opposite directions
    are never ordered against each other."""
    tau = dict(ak.MULTI_BASIN_FLOORS)

    def rec(cid, R, v0, D, d_b, v_e=30.0, direction=1, approach="A2"):
        return record(cell_id=cid, R=R, theta=90.0, approach=approach, direction=direction,
                      tau=tau, measures={"v_e": v_e + R / 100, "v_min": 20.0 + R / 10, "d_b": d_b},
                      labels={"R": R, "theta_deg": 90.0, "width": 12.0, "approach": approach,
                              "v0": v0, "D": D})

    # A2: d_b* grows from R 18.3 (v0 160 km/h) to R 22.9 (v0 180 km/h): not a violation.
    a2 = [rec(1, 18.3, kph(160), 120.0, 100.0), rec(2, 22.9, kph(180), 140.0, 120.0)]
    assert ak.v6_monotone(a2)[0].passed
    # The same pair at one (v0, D) is a violation.
    a1 = [rec(3, 18.3, kph(310), 600.0, 100.0, approach="A1"),
          rec(4, 22.9, kph(310), 600.0, 120.0, approach="A1")]
    v6 = ak.v6_monotone(a1)[0]
    assert not v6.passed and [v["measure"] for v in v6.details["violations"]] == ["d_b"]
    # A2 v_e* is still checked across the bins' v0.
    bad = [rec(5, 18.3, kph(160), 120.0, 100.0, v_e=35.0), rec(6, 22.9, kph(180), 140.0, 90.0)]
    v6 = ak.v6_monotone(bad)[0]
    assert [v["measure"] for v in v6.details["violations"]] == ["v_e"]
    # Equal R (V5's two directions solved separately) are not ordered.
    lr = [rec(7, 18.3, kph(310), 600.0, 100.0, approach="A1"),
          rec(8, 18.3, kph(310), 600.0, 140.0, approach="A1", direction=-1)]
    assert ak.v6_monotone(lr)[0].passed


def test_v9_slip_criterion_can_fail():
    """Numerics m4: both keys' slip-active steps on one scale (>= 8.0 deg), so a backed-off
    key that spends longer near its limit than the nominal one does above 8 deg fails."""
    nom = record(cell_id=1, measures={"v_e": 30.0, "v_min": 20.0, "has_brake": True},
                 active={"n_slip_backoff_limit": 10, "n_slip_nominal_limit": 6})
    bo_ok = record(cell_id=1, kind="backoff", measures={"v_e": 29.0, "v_min": 19.5},
                   active={"n_slip_backoff_limit": 4, "n_slip_f": 4, "n_slip_r": 0})
    bo_bad = dataclasses.replace(bo_ok, active={"n_slip_backoff_limit": 14, "n_slip_f": 14,
                                                "n_slip_r": 0})
    assert ak.v9_backoff([nom], [bo_ok]).passed
    v9 = ak.v9_backoff([nom], [bo_bad])
    assert not v9.passed and v9.details["cells"][1]["slip_steps_8deg"] == (10, 14)


def test_a_capped_refine_round_does_not_fail_a_converged_start():
    """Numerics m3: only solve rounds count for the cap; the refine round's cap keeps the
    unrefined optimum (WindowSolver.solve)."""
    res = SimpleNamespace(meta={"rounds": [{"round": 0, "status": "Solve_Succeeded"},
                                           {"round": "refine", "status": "Maximum_Iterations_Exceeded"}]})
    assert not ak._was_capped(res)
    res.meta["rounds"][0]["status"] = "Maximum_CpuTime_Exceeded"
    assert ak._was_capped(res)


def test_extend_straight_follows_a_curved_road():
    """Numerics m5: the guess tail keeps to a curved road (curvature feed-forward) instead of
    settling at the heading error -kappa v and leaving the corridor."""
    import jax.numpy as jnp

    from apexzero.tracks.generator import strip_arrays, to_track

    ak._x64()
    n = 2500
    k = np.full(n, 1.0 / 500.0)                        # full throttle: a_y reaches 15 m/s^2
    tr = to_track(strip_arrays(jnp.asarray(k), jnp.asarray(np.full(n, 6.0)),
                               jnp.asarray(np.full(n, 6.0)), ds=2.0), name="test/R500")
    x0 = np.array([100.0, 0.5, 0.02, 55.0, 0.0, 55.0 / 500.0, 0.0, 0.3])
    xs, acts = ak.extend_straight(x0, 300, tr)
    assert np.all(np.isfinite(xs)) and xs[-1, 0] > 1000.0
    # Inside the 5 m usable half-width (the old law reached 15 m here, 67 m from n = 0 on a
    # longer run) and back near the centreline at the end.
    assert np.max(np.abs(xs[:, 1])) < 4.0 and abs(xs[-1, 1]) < 0.5


def test_refresh_measures_recomputes_stale_cached_measures(solved):
    """Numerics M4: a record whose measures were computed by other measures code is
    recomputed from its stored solution; a change in a multi-basin measure is reported."""
    old = dataclasses.replace(solved.nom, measures={"d_b": -1.0, "v_e": 0.0},
                              measures_hash="stale")
    rec, moved = ak.refresh_measures(old, solved.bank)
    assert moved and rec.measures_hash == ak.measures_hash()
    assert np.isfinite(rec.measures["d_b"]) and rec.measures["d_b"] > 0.0
    again, moved2 = ak.refresh_measures(rec, solved.bank)
    assert again is rec and not moved2                # current hash: nothing to do
    assert solved.nom.measures_hash == ak.measures_hash()      # new records carry the hash


def test_v8_lists_failed_solves_and_takes_medians_over_the_same_cells(fake_solves, monkeypatch):
    """Numerics m8: a failed constrained s-OCP is listed and its cell leaves every median."""
    bank = FakeBank()
    cells = [TCell(cell_id=i, stim_id=0, pair_id=i, tiers=("V",), approach="A2", v0=40.0,
                   D=200.0 + 10.0 * i, W=400) for i in range(2)]
    base = ak._socp_for

    def socp_for(prob, cache, s_end=None, vx_upper=None):
        soc = base(prob, cache, s_end, vx_upper)
        if vx_upper and prob.cell.cell_id == 1:
            soc = dataclasses.replace(soc, result=dataclasses.replace(
                soc.result, success=False, status="Infeasible_Problem_Detected"))
        return soc

    monkeypatch.setattr(ak, "_socp_for", socp_for)
    res, b = ak.v8_band(bank, cells)
    assert res.passed and list(res.details["failed"]) == [1] and res.details["n_cells"] == 1
    # A failed s-OCP never sets the window from its time line.
    soc = base(ak._problem(bank, cells[0], "nominal"), None)
    failed = dataclasses.replace(soc, result=dataclasses.replace(soc.result, success=False))
    assert ak.window_length(failed, bank.stimuli[0].landmarks) == ak.W_LONG


def test_a_beaten_mirrored_key_is_re_solved_as_the_cells_own(fake_solves, tmp_path):
    """Fidelity M2 / numerics M1: the 6.3 re-solve of a right-hand key mirrored on the fly
    is stored under the right cell's own id, which key_for looks up first."""
    f = fake_solves
    bank, cache = FakeBank(), ak.KeyCache(tmp_path)
    left = TCell(cell_id=0, stim_id=0, pair_id=3, tiers=("A",), approach="A2", v0=40.0, D=200.0, W=400)
    right = TCell(cell_id=1, stim_id=1, pair_id=3, tiers=("A",), approach="A2", v0=40.0, D=200.0, W=400)
    nb = ak.full_recipe(bank, left, cache=cache, add_q_start=False)
    mk = ak.mirror_key(nb, right)
    run = Trajectory(source="agent", t=np.arange(401) * DT, x=mk.x.copy(), valid=np.ones(401, bool),
                     a=mk.a.copy())
    f.outcome["agent"] = (10.5, True)
    new, dev = ak.resolve_beaten_key(bank, right, mk, {"G9": run}, cache)
    want = ak.expected_key_id(bank, right, "nominal")
    assert dev["replaced"] and new.key_id == want != mk.key_id
    assert ak.key_for(right, "nominal", cache=cache, bank=bank).key_id == want
