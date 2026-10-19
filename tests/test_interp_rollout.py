"""The stimulus harness (protocol 4, step-2 checks) and agent loading / manifest (protocol 8).

Freeze rule: nothing here touches a trained snapshot. Runs use a randomly initialised
Actor with the agents' architecture (MLP 3 x 256 tanh) and scripted controllers, on small
hand-built strips; snapshots for the manifest tests are written to tmp_path from random
parameters. ``measures.front_brake_demand`` and ``Cell.initial_state`` belong to other
modules; when they are still stubs, local test doubles stand in (restored afterwards).
"""

from __future__ import annotations

import dataclasses
import math
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.agents.ppo import Actor, PPOConfig, RunningMeanStd
from apexzero.env import racing
from apexzero.interp import agents as A
from apexzero.interp import measures
from apexzero.interp import rollout as R
from apexzero.interp.constants import (
    DT,
    FLAG_BITS,
    MATCHED_NOISE_STD,
    N_KNOTS,
    REPLICATION,
    ROOT_SEED,
    S_C,
    S_TRUNC,
)
from apexzero.interp.stimuli import Cell, CornerSpec, RealState
from apexzero.tracks.generator import strip_arrays
from apexzero.vehicle.dynamics import _mu

# ------------------------------------------------------------------------- test doubles


def _brake_demand_double(x, p, ops):
    """The demand lines of dynamics.tyre_forces (measures.front_brake_demand's contract)."""
    vx, pedal = x[..., 3], x[..., 7]
    L = p.lf + p.lr
    v2 = vx * vx
    fz_f0 = p.m * p.g * p.lr / L + 0.5 * p.rho * p.cla_f * v2
    fz_r0 = p.m * p.g * p.lf / L + 0.5 * p.rho * p.cla_r * v2
    f_drag = 0.5 * p.rho * p.cda * v2
    f_roll = p.f_roll * (fz_f0 + fz_r0) * ops.tanh(vx / 0.5)
    brake = ops.relu(-pedal) * ops.tanh(vx / 0.5)
    f_drive = ops.relu(pedal) * p.p_max / ops.smax(vx, p.v_drive_floor, 0.5)
    fx_f_dem = -brake * p.f_brake_max * p.brake_bias
    fx_r_dem = f_drive - brake * p.f_brake_max * (1.0 - p.brake_bias)
    fx_f1 = ops.softsat(fx_f_dem, _mu(p.mux_f, 0.5 * fz_f0, p, ops) * fz_f0)
    fx_r1 = ops.softsat(fx_r_dem, _mu(p.mux_r, 0.5 * fz_r0, p, ops) * fz_r0)
    ax_est = (fx_f1 + fx_r1 - f_drag - f_roll) / p.m
    fz_f = ops.smax(fz_f0 - p.m * ax_est * p.h_cog / L, 200.0, 10.0)
    fx_max = _mu(p.mux_f, 0.5 * fz_f, p, ops) * fz_f
    return jnp.maximum(0.0, -fx_f_dem) / fx_max, fx_f_dem, fx_max


@pytest.fixture(scope="module", autouse=True)
def _float32():
    """Rollouts, controls and stimuli arrays run in float32. tests/test_dynamics.py and
    tests/test_env.py switch x64 on globally when they are imported, so in a full-suite run
    this module switches it off for its own tests and restores it afterwards."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module", autouse=True)
def brake_demand():
    """Use measures.front_brake_demand if implemented, else the double; clear the harness's
    compile cache afterwards so nothing traced with the double outlives this module."""
    mp = pytest.MonkeyPatch()
    try:
        measures.front_brake_demand(jnp.zeros(8), racing.params_from_context(
            jnp.asarray(R.default_ctx())), measures.JAX)
    except NotImplementedError:
        mp.setattr(measures, "front_brake_demand", _brake_demand_double)
    R._compiled.cache_clear()
    yield
    mp.undo()
    R._compiled.cache_clear()


class _Cell(Cell):
    """stimuli.Cell, with protocol 4's initial state while Cell.initial_state is a stub."""

    def initial_state(self, k: int = 0) -> np.ndarray:
        try:
            return super().initial_state(k)
        except NotImplementedError:
            u = self.phase_offsets()[k]
            if self.x0 is not None:
                x = np.asarray(self.x0, np.float64).copy()
                x[0] -= u
                return x.astype(np.float32)
            return R.approach_state(S_C - self.D - u, self.v0)


class _Bank:
    """Stimulus bank double: Stimulus records and stacked float32 TrackArrays."""

    def __init__(self, rows, stimuli):
        self.stimuli = stimuli
        arrs = [strip_arrays(*(jnp.asarray(a, jnp.float32) for a in r), 2.0) for r in rows]
        self._arrays = jax.tree.map(lambda *xs: jnp.stack(xs), *arrs)

    def arrays(self):
        return self._arrays


def _trapezoid(R_: float, theta_deg: float, l_r: float, direction: int) -> np.ndarray:
    s = np.arange(N_KNOTS) * 2.0
    th = math.radians(theta_deg)
    kp = min(1 / R_, th / l_r)
    Lp = max(th / kp - l_r, 0.0)
    u = s - S_C
    prof = np.clip(np.minimum(np.minimum(u / l_r, 1.0), (2 * l_r + Lp - u) / l_r), 0.0, None)
    return direction * kp * prof


def _stim(i, name, kind, direction, mirror_id, spec=None):
    """The fields of a stimuli.Stimulus record the harness reads."""
    return SimpleNamespace(stim_id=i, name=name, kind=kind, direction=direction,
                           mirror_id=mirror_id, spec=spec)


@pytest.fixture(scope="module")
def bank():
    k = _trapezoid(30.0, 90.0, 40.0, 1)
    w = np.full(N_KNOTS, 6.0)
    wr = w.copy()
    wr[400:700] = 5.0                     # asymmetric widths: the mirror swaps them
    spec = CornerSpec(30.0, 90.0, 1, l_r=40.0)
    rows = [(k, w, wr), (-k, wr, w), (np.zeros(N_KNOTS), w, w), (k, w, wr), (-k, wr, w)]
    stims = [_stim(0, "syn/R30/T90/W12/typical/L", "synthetic", 1, 1, spec),
             _stim(1, "syn/R30/T90/W12/typical/R", "synthetic", -1, 0,
                   dataclasses.replace(spec, direction=-1)),
             _stim(2, "syn/straight/W12", "straight", 0, -1),
             _stim(3, "E1/X/real", "transplant", 1, 4),
             _stim(4, "E1/X/mirror", "transplant", -1, 3)]
    return _Bank(rows, stims)


def _pair(v0=60.0, D=300.0, W=400, tiers=("A",), pair=0, first_id=0):
    kw = dict(tiers=tiers, approach="A1", v0=v0, D=D, W=W, P=v0 * DT)
    return (_Cell(first_id, 0, pair, **kw), _Cell(first_id + 1, 1, pair, mirrored=True, **kw))


@pytest.fixture(scope="module")
def actor():
    """Randomly initialised actor with the agents' architecture, and an obs_rms that
    leaves most channels near unit scale."""
    cfg = PPOConfig(network="mlp", actor_width=256, mlp_layers=3, activation="tanh")
    params = Actor(cfg, 2, False).init(jax.random.PRNGKey(0), jnp.zeros(racing.OBS_DIM))
    rms = RunningMeanStd(jnp.zeros(racing.OBS_DIM), jnp.full(racing.OBS_DIM, 0.5),
                         jnp.asarray(1e6))
    return R.agent_policy(cfg), params, rms


KEEPER = (jnp.float32(60.0), jnp.full(2, -jnp.inf, jnp.float32))


# ------------------------------------------------------------------------- states, noise


def test_initial_state_helpers():
    x = R.approach_state(400.0, 86.1)
    assert x.dtype == np.float32 and x.tolist() == pytest.approx([400, 0, 0, 86.1, 0, 0, 0, 0])
    ctx = R.default_ctx()
    assert ctx.dtype == np.float32
    np.testing.assert_array_equal(ctx, np.asarray(racing.default_context(), np.float32))
    st = R.init_env_state(x)
    assert st.x.dtype == st.ctx.dtype == st.slow.dtype == jnp.float32 and st.t.dtype == jnp.int32
    assert float(st.s0) == 400.0 and float(st.hidden_mu) == 1.0
    np.testing.assert_allclose(R.phases(80.0), [0.0, 1.0, 2.0, 3.0])
    assert len(R.phases(80.0, 16)) == 16
    # Sighting: d_max(vx) = clip(6 vx, 150, 500); at 50 m/s the corner is seen 300 m out.
    s = np.arange(0.0, 1000.0, 2.5)
    assert R.sight_speed(s, np.full_like(s, 50.0), 1000.0) == 50.0
    vx = np.linspace(80.0, 40.0, len(s))
    i = np.nonzero(1000.0 - s <= np.clip(6 * vx, 150, 500))[0][0]
    assert R.sight_speed(s, vx, 1000.0) == vx[i]
    with pytest.raises(ValueError):
        R.sight_speed(s[:10], vx[:10], 5000.0)


def test_noise_keys_are_common_random_numbers():
    k = R.noise_key(7, 3)
    want = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(ROOT_SEED), 7), 3)
    np.testing.assert_array_equal(k, np.asarray(want))
    assert k.dtype == np.uint32 and not np.array_equal(k, R.noise_key(7, 4))
    e = R.noise_eps(k, 5)
    np.testing.assert_array_equal(R.noise_eps(k, 5, -1.0), e * [-1, 1])
    np.testing.assert_array_equal(e[3], np.asarray(jax.random.normal(jax.random.fold_in(want, 3), (2,))))
    with pytest.raises(ValueError):
        R.noise_key(10**6, 0)              # would collide with the seed streams
    with pytest.raises(ValueError):
        R.noise_key(1, -1)


def test_plan_runs_and_batch(bank):
    a = _pair(tiers=("A",))
    e = (_Cell(2, 3, 1, ("E1",), "real", 50.0, None, W=400, P=2.5,
               x0=(200.0, 0.5, 0.01, 50.0, 0.1, 0.02, 0.01, 0.3), prev_action0=(0.1, 0.4),
               accel0=(1.0, -2.0)),
         _Cell(3, 4, 1, ("E1",), "real", 50.0, None, W=400, P=2.5,
               x0=(200.0, -0.5, -0.01, 50.0, -0.1, -0.02, -0.01, 0.3),
               prev_action0=(-0.1, 0.4), accel0=(1.0, 2.0), mirrored=True))
    b = _Cell(4, 2, 2, ("B",), "B", 70.0, 250.0, W=800, P=3.5)
    cells = [*a, *e, b]
    det = R.plan_runs(cells, bank, "det", 0)
    assert len(det) == 5 * 4 and {r.sample for r in det} == {-1}
    assert [r.steer_sign for r in det[::4]] == [1, -1, 1, -1, 1]
    nat = R.plan_runs(cells, bank, "nat", REPLICATION["final"]["native"])
    counts = {c.cell_id: sum(r.cell_id == c.cell_id for r in nat) for c in cells}
    assert counts == {0: 32, 1: 32, 2: 16, 3: 16, 4: 16}
    assert all(r.phase == r.sample % 4 for r in nat)
    mat = R.plan_runs(cells, bank, "mat", REPLICATION["final"]["matched"])
    assert {r.cell_id for r in mat} == {0, 1, 2, 3}          # 'E' matches E1; B gets none
    # phases_det picks that many of a cell's own phases, spread over the cycle; it never
    # invents phases a cell does not have (4 here).
    assert len(R.plan_runs(cells, bank, "det", 0, phases_det=16)) == 5 * 4
    assert [r.phase for r in R.plan_runs(cells, bank, "det", 0, phases_det=2)][:2] == [0, 2]

    runs = [det[1], nat[5], mat[-1]]
    batch, table = R.make_batch(runs, {c.cell_id: c for c in cells})
    np.testing.assert_allclose(batch.x0[0], R.approach_state(S_C - 300.0 - 60 * DT / 4, 60.0))
    np.testing.assert_allclose(batch.x0[2][1:], e[1].x0[1:])
    assert batch.x0[2][0] == np.float32(200.0 - mat[-1].phase * 2.5 / 4)
    np.testing.assert_array_equal(batch.prev_action[2], np.float32([-0.1, 0.4]))
    np.testing.assert_array_equal(batch.noise_key[0], 0)
    np.testing.assert_array_equal(batch.noise_key[1], R.noise_key(0, nat[5].sample))
    assert batch.native.tolist() == [False, True, False]
    np.testing.assert_allclose(batch.noise_std, [[0, 0], [0, 0], MATCHED_NOISE_STD])
    assert table["phase"].dtype == np.int16 and table["s0"].dtype == np.float64
    assert table["steer_sign"].tolist() == [1, 1, -1]
    # A pair whose members share a sign would not see mirrored noise.
    with pytest.raises(ValueError, match="same steer sign"):
        R.plan_runs([a[0], dataclasses.replace(a[1], mirrored=False)], bank, "det", 0)
    # Cells listing their runners are planned for those agents only.
    only_g2 = dataclasses.replace(b, runners=("G2",))
    assert R.plan_runs([only_g2], bank, "det", 0, agent="G1") == []
    assert len(R.plan_runs([only_g2], bank, "det", 0, agent="G2")) == 4
    # Without the Cell field, the bank record decides (left +1, right -1; mirror rows -1).
    assert [R.steer_sign(st) for st in bank.stimuli] == [1, -1, 1, 1, -1]


# ------------------------------------------------------------------------- step-2 checks


def test_harness_matches_step_no_reset_loop(bank, actor):
    pol, params, rms = actor
    cell = _pair()[0]
    # Random network (obs normalised and clipped): it leaves the road after ~60 steps.
    assert R.check_step_loop(pol, params, rms, bank, cell, steps=400, phase=1) <= 1e-6
    # Scripted lane keeper on raw observations: alive for all 400 steps through the corner.
    assert R.check_step_loop(R._lane_keeper, KEEPER, None, bank, cell, steps=400) <= 1e-6
    tr = R.trace_arrays(R.make_stimulus_rollout(R._lane_keeper, 400)(
        KEEPER, None, R.bank_arrays(bank), R.make_batch(
            [R.RunSpec(0, 0, 0, 0, -1, "det", 1, 400)], {0: cell})[0]), 1, "full")
    assert tr["term_step"][0] == -1 and tr["valid"][0].all()
    assert tr["x"][0, -1, 0] > S_C + 100                     # through the corner


def test_batch_equals_single(bank, actor):
    pol, params, rms = actor
    cells = list(_pair())
    # Exact: each run redone among decoys in a batch of the same shape.
    assert R.check_batch_equals_single(pol, params, rms, bank, cells, steps=60) == 0.0
    assert R.check_batch_equals_single(pol, params, rms, bank, cells, steps=60, mode="nat",
                                       samples=3) == 0.0
    assert R.check_batch_equals_single(R._lane_keeper, KEEPER, None, bank, cells,
                                       steps=60) == 0.0
    # Batch of one: equal to float32 rounding (XLA vectorises another shape differently).
    assert R.check_batch_equals_single(pol, params, rms, bank, cells, steps=60,
                                       isolated=False) <= 1e-5


def test_mirror_gate(bank):
    pair = _pair()
    assert R.mirror_gate(bank, pair, steps=400) <= 1e-5
    # Negative control: the same tape (steer not negated) on the mirror does not mirror.
    arrays = R.bank_arrays(bank)
    acts = np.zeros((2, 400, 2), np.float32)
    acts[:, :, 0], acts[:, :, 1] = 0.02, 0.3
    x0 = np.stack([R._cell_x0(c) for c in pair])
    tr = R.trace_arrays(R.make_stimulus_rollout(R._tape_policy, 400)(
        jnp.asarray(acts), None, arrays, R.batch_from_states([0, 1], x0, aux=[0, 1])), 2, "full")
    v = tr["valid"][0] & tr["valid"][1]
    assert np.abs(tr["x"][0][v] - tr["x"][1][v] * R._MIRROR).max() > 1e-2


def test_common_random_numbers(bank):
    a, b = _pair()
    e = (_Cell(2, 3, 1, ("E1",), "real", 60.0, 300.0, W=400, P=3.0),
         _Cell(3, 4, 1, ("E1",), "real", 60.0, 300.0, W=400, P=3.0, mirrored=True))
    assert R.check_common_noise(bank, [a, b, *e], steps=40, samples=2)
    # Without opposite steer signs the pair would not see mirrored noise.
    assert not R.check_common_noise(bank, [a, b], steps=40, samples=2, signs={0: 1, 1: 1})


def test_freeze_after_termination_and_flags(bank):
    cell = _pair()[0]
    fn = R.make_stimulus_rollout(R._const_policy, 40)
    params = (np.array([1.0, 1.0], np.float32), np.full(2, -np.inf, np.float32))
    tr = R.trace_arrays(fn(params, None, R.bank_arrays(bank),
                           R.make_batch([R.RunSpec(0, 0, 0, 0, -1, "det", 1, 40)], {0: cell})[0]),
                        1, "full")
    k = int(tr["term_step"][0])
    assert 0 <= k < 39 and tr["trunc_step"][0] == -1
    v = tr["valid"][0]
    assert v[: k + 2].all() and not v[k + 2:].any()          # terminal state is valid
    assert (tr["x"][0][k + 2:] == tr["x"][0][k + 1]).all()   # frozen copies
    assert R.cause_of(tr["flags"][0][k]) in ("corridor", "slip", "spin")
    assert (tr["flags"][0][k + 1:] == 0).all() and (tr["flags"][0][:k] == 0).all()
    assert np.isnan(tr["a"][0][k + 1:]).all() and (tr["reward"][0][k + 1:] == 0).all()
    assert tr["reward"][0][k] < -5                           # the crash penalty
    assert np.isnan(tr["clip_frac"][0]).all()                # controls: no normaliser


def test_truncation_at_strip_end(bank):
    cell = _Cell(9, 2, 5, ("G",), "G", 60.0, None, W=40, P=3.0,
                 x0=(S_TRUNC - 30.0, 0, 0, 60.0, 0, 0, 0, 0))
    fn = R.make_stimulus_rollout(R._lane_keeper, 40)
    tr = R.trace_arrays(fn(KEEPER, None, R.bank_arrays(bank), R.make_batch(
        [R.RunSpec(9, 2, 5, 0, -1, "det", 1, 40)], {9: cell})[0]), 1, "full")
    k = int(tr["trunc_step"][0])
    assert tr["term_step"][0] == -1 and 0 <= k < 15
    assert tr["flags"][0][k] == 1 << FLAG_BITS["truncated"]
    assert tr["x"][0][k + 1, 0] > S_TRUNC and not tr["valid"][0][k + 2:].any()


def test_x64_is_asserted_off(bank):
    cell = _pair()[0]
    runs = [R.RunSpec(0, 0, 0, 0, -1, "det", 1, 40)]
    with jax.enable_x64(True), pytest.raises(RuntimeError, match="x64"):
        R.run_rollouts(R._lane_keeper, KEEPER, None, bank, [cell], runs, "full")


def test_clip_statistics_and_recomputed_observations(bank):
    cell = _pair()[0]
    ctx_raw = np.asarray(2 * (R.default_ctx() - racing._CTX_LO) / (racing._CTX_HI - racing._CTX_LO) - 1)
    mean = np.zeros(racing.OBS_DIM, np.float32)
    var = np.ones(racing.OBS_DIM, np.float32)
    mean[-6:], var[-6:] = ctx_raw + 2e-3, 5e-13               # context channels saturate
    rms = RunningMeanStd(jnp.asarray(mean), jnp.asarray(var), jnp.asarray(1.0))
    pol = lambda p_, o, st, tr, aux: R._lane_keeper(p_, o, st, tr, aux)
    fn = R.make_stimulus_rollout(pol, 40)
    tr = R.trace_arrays(fn(KEEPER, rms, R.bank_arrays(bank), R.make_batch(
        [R.RunSpec(0, 0, 0, 0, -1, "det", 1, 40)], {0: cell})[0]), 1, "full")
    assert (tr["clip_frac"][0][:, 3] == 1.0).all() and (tr["clip_max"][0][:, 3] > 15.0).all()
    assert (tr["clip_frac"][0][:, :3] == 0.0).all()
    one = jax.tree.map(lambda a: a[0], R.bank_arrays(bank))
    obs = R.recompute_observations(tr["x"][0], tr["prev_action"][0], tr["accel"][0], one, rms)
    assert obs.shape == (41, racing.OBS_DIM) and (np.abs(obs[:, -6:]) == 10.0).all()
    raw = R.recompute_observations(tr["x"][0][:1], tr["prev_action"][0][:1],
                                   tr["accel"][0][:1], one)
    st = R.init_env_state(tr["x"][0][0])
    want = racing.observe(st, one, racing.params_from_context(st.ctx))
    np.testing.assert_allclose(raw[0], np.asarray(want), rtol=1e-6, atol=1e-6)


def test_context_checks():
    ctx_raw = np.asarray(2 * (R.default_ctx() - racing._CTX_LO) / (racing._CTX_HI - racing._CTX_LO) - 1)
    mean = np.zeros(racing.OBS_DIM, np.float32)
    var = np.ones(racing.OBS_DIM, np.float32)
    mean[-6:], var[-6:] = ctx_raw, 5e-13
    s1 = RunningMeanStd(mean, var, np.float32(1.0))
    g = RunningMeanStd(np.zeros(racing.OBS_DIM), np.ones(racing.OBS_DIM), np.float32(1.0))
    out = R.check_context({"S1": s1, "G1": g})
    assert out["S1"] <= 1e-3 and 0.0 < out["G1"] <= 1.0
    # The float64-rounding pitfall: a 1e-6 offset in the mean puts S1's context near the clip.
    bad = RunningMeanStd(mean + np.r_[np.zeros(216), np.full(6, 1e-6)].astype(np.float32),
                         var, np.float32(1.0))
    with pytest.raises(ValueError):
        R.check_context({"S1": bad})


def test_run_rollouts_chunks_pads_and_files(bank, tmp_path):
    a, b = _pair(W=30)
    c = _Cell(2, 2, 3, ("G",), "G", 60.0, 300.0, W=40, P=3.0)
    cells = [a, b, c]
    runs = R.plan_runs(cells, bank, "det", 0)
    runs = runs[::-1]                                        # mixed W, unsorted
    one = R.run_rollouts(R._lane_keeper, KEEPER, None, bank, cells, runs, "full")
    small = R.run_rollouts(R._lane_keeper, KEEPER, None, bank, cells, runs, "full", chunk=3)
    assert one["x"].shape == (12, 41, 8) and one["flags"].shape == (12, 40)
    assert one["cell"].tolist() == [r.cell_id for r in runs]
    for k in ("x", "util_f", "beta_f"):
        np.testing.assert_allclose(small[k], one[k], rtol=1e-6, atol=1e-5)
    w30 = one["W"] == 30
    assert not one["valid"][w30][:, 31:].any() and one["valid"][~w30].all()
    assert (one["x"][w30][:, 31:] == one["x"][w30][:, 30:31]).all()
    assert np.isnan(one["a"][w30][:, 30:]).all()
    assert one["beta_f"].dtype == np.float32 and one["s0"].dtype == np.float64

    nat_runs = R.plan_runs(cells, bank, "nat", 2)
    nat = R.run_rollouts(R._lane_keeper, KEEPER, None, bank, cells, nat_runs, "stoch")
    assert set(nat) >= {"s", "vx", "n", "pedal", "xi", "util_f", "valid", "flags", "term_step"}
    assert "x" not in nat and nat["n"].dtype == np.float16 and nat["s"].dtype == np.float32

    path = R.save_rollouts(tmp_path / "S9" / "000000001000.npz", {"det": one, "nat": nat},
                           {"log_std": [-3.0, -2.5], "agent": "S9", "timesteps": 1000,
                            "code_hash": "abc", "note": "unit"})
    groups, meta = R.load_rollouts(path)
    assert meta["agent"] == "S9" and meta["timesteps"] == 1000 and meta["note"] == "unit"
    np.testing.assert_array_equal(meta["log_std"], np.float32([-3.0, -2.5]))
    for g, src in (("det", one), ("nat", nat)):
        assert set(groups[g]) == set(src)
        for k, v in src.items():
            np.testing.assert_array_equal(groups[g][k], v)
            assert groups[g][k].dtype == v.dtype


def test_benchmark_runs(bank, actor):
    pol, params, rms = actor
    assert R.benchmark(pol, params, rms, bank, batch_size=16, steps=20) > 0


def test_pilot_verdict():
    keys = [(r, t, d) for r in (9.0, 11.0, 13.0, 16.0, 20.0, 30.0) for t in (90.0, 180.0)
            for d in (1, -1)]
    none = dict.fromkeys(keys, False)
    assert R.pilot_verdict("G1", none) and not R.pilot_verdict("G2", none)
    g2 = {**none, (9.0, 180.0, -1): True}
    assert R.pilot_verdict("G2", g2) and not R.pilot_verdict("G1", g2)
    assert not R.pilot_verdict("G2", {**g2, (9.0, 90.0, -1): True})
    stim = _stim(0, "pilot/R13/T180/W11/pilot/R", "pilot", -1, 1)
    assert R._pilot_key(stim) == (13.0, 180.0, -1)


# ------------------------------------------------------------------------- agents


def _write_snapshot(path, cfg, timesteps=None, seed=0):
    params = jax.device_get(Actor(cfg, 2, False).init(jax.random.PRNGKey(seed),
                                                      jnp.zeros(racing.OBS_DIM)))
    rms = {"mean": np.zeros(racing.OBS_DIM, np.float32), "var": np.ones(racing.OBS_DIM, np.float32),
           "count": np.float32(10.0)}
    d = {"actor": params, "critic": {}, "obs_rms": rms, "ret_rms": {"mean": np.float32(0),
         "var": np.float32(1), "count": np.float32(1)},
         "config": {**dataclasses.asdict(cfg), "retired_option": 1}}
    if timesteps is not None:
        d["timesteps"] = timesteps
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(d, fh)
    return path


def test_load_agent_and_manifest(tmp_path):
    cfg = PPOConfig(network="mlp", actor_width=32, mlp_layers=2, num_envs=64, num_steps=16,
                    total_timesteps=100_000)
    spec_s = A.AgentSpec("S9", str(tmp_path / "spec"), "specialist", True, float("inf"),
                         "agent.pkl", "agent.pkl", None)
    spec_g = A.AgentSpec("G9", str(tmp_path / "gen"), "generalist", False, 15_000, "ckpt",
                         "eval_rdev.json", None)
    spec_u = A.AgentSpec("G8", str(tmp_path / "unfinished"), "generalist", False, 0.0, "ckpt",
                         "eval_rdev.json", None)
    _write_snapshot(tmp_path / "spec" / "agent.pkl", cfg)
    for ts in (10_000, 20_000):
        _write_snapshot(tmp_path / "gen" / "ckpt" / f"{ts:012d}.pkl", cfg, ts, seed=ts)
    (tmp_path / "gen" / "eval_rdev.json").write_text("{}")
    _write_snapshot(tmp_path / "unfinished" / "ckpt" / "000000010000.pkl", cfg, 10_000)

    ag = A.load_agent(tmp_path / "spec" / "agent.pkl", "S9")
    assert ag.timesteps == cfg.num_updates * cfg.batch_size == 99_328
    assert ag.cfg == cfg and ag.log_std.shape == (2,) and A.forward_pass_ok(ag)
    np.testing.assert_allclose(ag.log_std, cfg.log_std_init)
    assert isinstance(ag.obs_rms, RunningMeanStd)

    bad = tmp_path / "bad.pkl"
    bad.write_bytes((tmp_path / "spec" / "agent.pkl").read_bytes()[:100])
    with pytest.raises(A.SnapshotError):
        A.load_agent(bad)

    assert A.list_snapshots(spec_u) == []
    assert [p.name for p in A.last_three(spec_g)] == ["000000010000.pkl", "000000020000.pkl"]
    specs = {"S9": spec_s, "G9": spec_g, "G8": spec_u}
    out = tmp_path / "manifest.json"
    m = A.freeze_manifest(out, specs)
    assert len(m.snapshots) == 3 and m.code_hash and not m.agents["G8"]["finished"]
    by_ts = {(e.agent, e.timesteps): e for e in m.snapshots}
    e1, e2 = by_ts[("G9", 10_000)], by_ts[("G9", 20_000)]
    assert not e1.prospective and e2.prospective and e2.is_final and not e1.is_final
    assert all(e.forward_pass_ok and not e.evaluated_on_stimuli and not e.error
               for e in m.snapshots)
    assert e1.std == pytest.approx(tuple(math.exp(v) for v in e1.log_std))
    assert e1.sha256 != e2.sha256 and e1.size_bytes > 0

    # A new checkpoint and a corrupt one appear; existing entries keep their values (only
    # the derived is_final moves to the new last snapshot).
    A.mark_evaluated(m, e1.path, "rollouts-det")
    A.save_manifest(m, out)
    before1, before2 = dataclasses.replace(e1), dataclasses.replace(e2)
    _write_snapshot(tmp_path / "gen" / "ckpt" / "000000030000.pkl", cfg, 30_000)
    (tmp_path / "gen" / "ckpt" / "000000040000.pkl").write_bytes(b"\x80\x04garbage")
    m2 = A.freeze_manifest(out, specs)                        # reads the saved manifest
    assert m2.entry(e1.path) == before1 and m2.entry(e1.path).evaluated_on_stimuli
    assert m2.entry(e2.path) == dataclasses.replace(before2, is_final=False)
    broken = m2.entry(A._rel(tmp_path / "gen" / "ckpt" / "000000040000.pkl"))
    assert broken.error and broken.timesteps == 40_000 and broken.is_final
    back = A.load_manifest(out)
    assert len(back.snapshots) == 5
    assert [e for e in back.snapshots if not e.error] == [e for e in m2.snapshots if not e.error]
    assert back.entry(e1.path).first_evaluated.endswith("rollouts-det")


def test_real_lap_refuses_lockbox():
    with pytest.raises(PermissionError):
        A.real_lap(None, "Monza")


def test_real_state():
    L = 1000.0
    s = np.linspace(0.0, 2.5 * L, 1001)
    x = np.zeros((1001, 8), np.float32)
    x[:, 0], x[:, 3] = s, 50.0
    valid = np.ones(1001, bool)
    lap = A.RealLap("G9", "Hockenheim", 50.0, x, np.zeros((1001, 2)), np.ones((1001, 2)),
                    valid, -1, "", L)
    st = A.real_state(lap, 300.0, 2)
    assert isinstance(st, RealState) and st.lap == 2
    assert st.s_real == pytest.approx(300.0, abs=2.5) and st.x[0] == st.s_real
    assert A.real_state(lap, -500.0, 1).x[0] == 0.0          # precedes the start
    valid[601:] = False                                       # crash: terminal state at 600
    crashed = dataclasses.replace(lap, valid=valid, term_step=599, cause="corridor")
    assert A.real_state(crashed, 400.0, 2).s_real < 500.0
    with pytest.raises(ValueError, match="corridor"):
        A.real_state(crashed, 499.0, 2)                      # only the terminal state is past it


def test_real_lap_with_a_random_network(tmp_path):
    """The real-lap convention end to end on an R_DEV circuit with a random (untrained)
    network, which leaves the road within seconds; no trained snapshot is involved."""
    cfg = PPOConfig(network="mlp", actor_width=32, mlp_layers=2)
    ag = A.load_agent(_write_snapshot(tmp_path / "rnd.pkl", cfg, 1000), "")
    lap = A.real_lap(ag, "Hockenheim", laps=1)
    assert lap.x[0, 0] == 0.0 and lap.x[0, 3] == pytest.approx(lap.v_start)
    assert lap.length == pytest.approx(4569.54, abs=0.1) and lap.x.shape[1] == 8
    k = lap.term_step
    assert k >= 0 and lap.cause and lap.valid[: k + 2].all() and not lap.valid[k + 2:].any()
    st = A.real_state(lap, -800.0, 1)                         # before the start: x0
    assert st.x[3] == pytest.approx(lap.v_start) and st.prev_action == (0.0, 0.0)
