"""Ruling R21 (dry-run rulings, protocol 12): phase k of a real-lap state starts on the
recorded run, (k / n_phases) DT earlier, interpolated in time; designed states keep the
s-shift. The recorded runs here are agent-free: the env stepped under a constant action on
the real R_DEV circuit, recorded as the harness records a run (x, prev_action, accel per
step), and the state taken from it as ``agents.real_state`` takes one."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.env import racing
from apexzero.interp import io, rollout
from apexzero.interp import stimuli as T
from apexzero.interp import support as S
from apexzero.interp.constants import DS, DT, S_C
from apexzero.vehicle.params import F1_2025
from apexzero.vehicle.sim import track_arrays


@pytest.fixture(scope="module", autouse=True)
def _float32():
    """Stimuli, rollouts and the env run in float32 (see tests/test_interp_stimuli.py)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(autouse=True)
def _no_outputs(monkeypatch, tmp_path):
    """The crash pairs are the named ones only: no file under outputs/ is read."""
    monkeypatch.setattr(T.io, "STIMULI_DIR", tmp_path / "stimuli")


LT = S.LTypTable(l_nodes=(26.0, 33.0, 40.0, 44.0, 46.0), counts=(1, 1, 1, 1, 1))
VT = S.VTypTable(choice_kph={f"{b}|{c}": v for b, v in enumerate((160.0, 160.0, 180.0, 220.0, 250.0))
                             for c in ("lt120", "ge120")})
STATE_FIELDS = ("s", "n", "xi", "vx", "vy", "r", "delta", "pedal")
# The observation's own scales of each state variable (racing.observe: vx / 50, vy / 5, r,
# xi, delta / delta_max, pedal); s and n in metres.
OBS_SCALE = np.array([1.0, 1.0, 1.0, 50.0, 5.0, 1.0, F1_2025.delta_max, 1.0])
I0 = 3                                  # trace index of the recorded state


@pytest.fixture(scope="module")
def design():
    return T.build_design(S.SupportTables(LT, VT, {}, (1.0,) * 5, {}, ""))


def _recorded_run(track_name: str, s_start: float, steps: int = I0 + 1) -> dict:
    """An agent-free recorded run: the env (float32, default context) from a transient
    state at ``s_start`` under a constant action, recorded as the harness records a run
    (x [K, 8] with s unwrapped, prev_action and accel; step 0 the initial values)."""
    track = T.real_track(track_name)
    tr = track_arrays(track)
    env = racing.EnvParams(window_steps=10**9)
    st = rollout.init_env_state(np.array([s_start, 0.4, 0.02, 68.0, 0.3, 0.05, 0.015, 0.2],
                                         np.float32), (0.05, 0.4), (1.5, 2.0))
    a = jnp.asarray([-0.03, -0.6], jnp.float32)        # braking and unwinding: a transient
    step = jax.jit(lambda s: racing.step_no_reset(s, a, tr, env)[1])
    xs, pas, acs = [st.x], [st.prev_action], [st.accel]
    for _ in range(steps):
        st = step(st)
        xs.append(st.x)
        pas.append(st.prev_action)
        acs.append(st.accel)
    x = np.asarray(jnp.stack(xs), np.float32)
    return dict(x=x, prev_action=np.asarray(jnp.stack(pas), np.float32),
                accel=np.asarray(jnp.stack(acs), np.float32), valid=np.ones(len(x), bool),
                length=float(track.length), ds=float(track.ds))


def _state(run: dict, agent: str, track: str, lap: int, i0: int = I0) -> T.RealState:
    """``agents.real_state``'s record of trace step i0 (s modulo the circuit length)."""
    x = run["x"][i0].astype(np.float64).copy()
    x[0] = x[0] % run["length"]
    return T.RealState(agent=agent, track=track, lap=lap, s_real=float(x[0]),
                       x=tuple(float(v) for v in x),
                       prev_action=tuple(float(v) for v in run["prev_action"][i0]),
                       accel=tuple(float(v) for v in run["accel"][i0]))


def _attached(run: dict, st: T.RealState, i0: int = I0) -> T.RealState:
    return T.attach_lap_step(st, run["x"], run["prev_action"], run["accel"], i0, run["length"],
                             run["valid"])


@pytest.fixture(scope="module")
def real(design):
    """The stand-in test's two real states (G2's E1 Norisring 874 state and S1's tier D
    corner 14 state), taken from agent-free recorded runs, built with and without their
    recorded steps attached."""
    bank, cells = design
    reqs = T.real_state_requests()
    e1 = next(r for r in reqs if r["key"].startswith("E1/Norisring/874") and r["agent"] == "G2")
    d14 = next(r for r in reqs if r["key"] == "D/14" and r["agent"] == "S1")
    runs, plain, attached = {}, {"G2": {}, "S1": {}}, {"G2": {}, "S1": {}}
    for r in (e1, d14):
        L = T.real_track(r["track"]).length
        run = _recorded_run(r["track"], r["s_target"] % L - I0 * 68.0 * DT)
        st = _state(run, r["agent"], r["track"], r["lap"])
        runs[r["key"]] = run
        plain[r["agent"]][r["key"]] = st
        attached[r["agent"]][r["key"]] = _attached(run, st)
    return dict(design=design, runs=runs, keys=(e1["key"], d14["key"]),
                plain=T.build_real_design(bank, cells, plain),
                lap=T.build_real_design(bank, cells, attached))


def _new_cells(real, which="lap"):
    bank, cells = real["design"]
    b2, c2 = real[which]
    return b2, c2[len(cells):]


def _key(c: T.Cell) -> str:
    return c.state_source.split("@", 1)[1]


# ------------------------------------------------------------------------- attachment


def test_attach_lap_step_maps_the_previous_step_and_checks_the_state():
    run = _recorded_run("Norisring", 50.0)
    st = _state(run, "G2", "Norisring", 1)
    at = _attached(run, st)
    assert at.trace_index == I0 and at.x == st.x and at.prev_action == st.prev_action
    assert at.x_before[1:] == tuple(float(v) for v in run["x"][I0 - 1, 1:])
    assert at.x_before[0] == pytest.approx(st.s_real - float(run["x"][I0, 0] - run["x"][I0 - 1, 0]),
                                           abs=1e-12)
    assert at.prev_action_before == tuple(float(v) for v in run["prev_action"][I0 - 1])
    assert at.accel_before == tuple(float(v) for v in run["accel"][I0 - 1])
    # The run's own start: no recorded step before it.
    st0 = _state(run, "G2", "Norisring", 1, 0)
    at0 = _attached(run, st0, 0)
    assert at0.trace_index == 0 and at0.x_before is None
    with pytest.raises(ValueError):                       # not the state at that index
        _attached(run, st, I0 - 1)
    with pytest.raises(ValueError):                       # the step before is not live
        T.attach_lap_step(st, run["x"], run["prev_action"], run["accel"], I0, run["length"],
                          np.array([True, True, False, True, True]))


def test_load_real_states_reads_the_real_laps_files(tmp_path):
    """The format ``rollouts --real-laps`` writes: real_states.json, the index and the
    per-agent lap npz (scripts/psychophysics.py ``_real_laps``)."""
    run = _recorded_run("Norisring", 50.0)
    st = _state(run, "G2", "Norisring", 1)
    st0 = _state(run, "G2", "Norisring", 1, 0)
    d = tmp_path / "stimuli"
    io.write_json(d / T.REAL_STATES_FILE, {"G2": {"E1/Norisring/874": dataclasses.asdict(st),
                                                  "E1/Norisring/446": dataclasses.asdict(st0)}})
    io.write_json(d / T.REAL_STATES_INDEX_FILE, {"G2": {"E1/Norisring/874": I0,
                                                        "E1/Norisring/446": 0}})
    (d / T.REAL_LAPS_DIR).mkdir(parents=True)
    np.savez(d / T.REAL_LAPS_DIR / "G2_Norisring.npz", x=run["x"], prev_action=run["prev_action"],
             accel=run["accel"], valid=run["valid"], term_step=-1, cause="", length=run["length"],
             v_start=68.0, ds=run["ds"])
    out = T.load_real_states(d)
    assert out["G2"]["E1/Norisring/874"] == _attached(run, st)
    assert out["G2"]["E1/Norisring/446"].trace_index == 0
    io.write_json(d / T.REAL_STATES_INDEX_FILE, {"G2": {"E1/Norisring/874": I0 - 1,
                                                        "E1/Norisring/446": 0}})
    with pytest.raises(ValueError):
        T.load_real_states(d)


# ------------------------------------------------------------------------- R21 acceptance


def test_phase0_and_strips_unchanged(real):
    """Phase 0 (state, prev_action, accel) and every strip are bit-identical with and
    without the recorded steps; only the R21 fields differ."""
    b_plain, plain = _new_cells(real, "plain")
    b_lap, lap = _new_cells(real, "lap")
    assert b_plain.content_hash() == b_lap.content_hash()
    assert len(plain) == len(lap) == 7 + 22
    for p, q in zip(plain, lap, strict=True):
        assert p.x0 == q.x0 and p.prev_action0 == q.prev_action0 and p.accel0 == q.accel0
        assert np.array_equal(p.initial_state(0), q.initial_state(0))
        for a, b in zip(p.initial_aux(0), q.initial_aux(0), strict=True):
            assert np.array_equal(a, b)
        assert dataclasses.replace(q, x0_before=None, prev_action_before=None,
                                   accel_before=None) == p
        assert q.phase_rule() == "lap" and p.phase_rule() == "missing"
        np.testing.assert_array_equal(q.initial_state(0),
                                      np.asarray(q.x0, np.float64).astype(np.float32))


def test_phase_k_starts_on_the_recorded_run(real):
    """Phase k equals the run interpolated at t0 - (k/4) DT (state, prev_action, accel),
    with s mapped into the strip by the excision's own map; normalised by the observation's
    scales the difference is far inside 0.005 (float32 rounding only)."""
    b2, cells = _new_cells(real)
    worst = 0.0
    for c in cells:
        run = real["runs"][_key(c)]
        st = b2.stimuli[c.stim_id]
        sign = -1.0 if c.mirrored else 1.0
        flip = np.array([1, sign, sign, 1, sign, sign, sign, 1.0])
        L, x = run["length"], run["x"].astype(np.float64)
        s_real = x[I0, 0] % L
        to_strip = c.x0[0] - s_real / st.source["ds_source"] * DS      # x0 = to_strip(s_real)
        for k in range(1, c.n_phases):
            w = k / c.n_phases
            want = (1.0 - w) * x[I0] + w * x[I0 - 1]
            want[0] = to_strip + (want[0] - x[I0, 0] + s_real) / st.source["ds_source"] * DS
            want = want * flip
            got = c.initial_state(k).astype(np.float64)
            assert c.initial_state(k).dtype == np.float32
            worst = max(worst, float(np.max(np.abs(got - want) / OBS_SCALE)))
            np.testing.assert_allclose(got[1:], want[1:], rtol=1e-6, atol=1e-7)
            assert abs(got[0] - want[0]) <= 2e-4                       # float32 ulp at ~800 m
            pa_r, ac_r = run["prev_action"].astype(np.float64), run["accel"].astype(np.float64)
            pa_w = pa_r[I0] * [sign, 1]                         # the command held over the step
            ac_w = ((1 - w) * ac_r[I0] + w * ac_r[I0 - 1]) * [1, sign]
            pa, ac = c.initial_aux(k)
            assert pa.dtype == ac.dtype == np.float32
            np.testing.assert_allclose(pa, pa_w, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(ac, ac_w, rtol=1e-6, atol=1e-6)
    assert worst <= 0.005


def test_phase_k_keeps_the_held_command_and_interpolates_accel():
    """Review of R21 (agent-facing finding 2): over the control step that ends at x0 the env
    applies, and records as prev_action, the command prev_action0; a start inside that step
    sees prev_action0 at every phase, not a blend toward the step before's command (on the
    real states the pedal command changes by up to 0.31 between the two steps, so the blend
    was off by up to 0.23 at k = 3). accel, the trailing-step acceleration, is interpolated
    with the state. The mirror member's aux is the exact mirror at every phase."""
    x0 = (1000.0, 0.1, 0.01, 60.0, 0.2, 0.03, 0.01, 0.5)
    xb = (997.0, 0.12, 0.012, 60.5, 0.21, 0.031, 0.011, 0.2)
    c = T.Cell(0, 0, 0, ("D",), "real", 60.0, 300.0, W=400, P=3.0, x0=x0,
               prev_action0=(0.02, 0.9), accel0=(-3.0, 4.0), state_source="S1:lap1@D/1",
               x0_before=xb, prev_action_before=(0.01, 0.6), accel_before=(-1.0, 2.0))
    assert c.phase_rule() == "lap"
    m = c.mirrored_start()
    for k in range(c.n_phases):
        w = k / c.n_phases
        pa, ac = c.initial_aux(k)
        np.testing.assert_array_equal(pa, np.asarray(c.prev_action0, np.float32))
        np.testing.assert_allclose(ac, (1 - w) * np.array([-3.0, 4.0]) + w * np.array([-1.0, 2.0]),
                                   rtol=1e-6)
        want = T.mirror_aux(pa, ac)
        assert all(np.array_equal(u, v) for u, v in zip(m.initial_aux(k), want, strict=True))
    assert not np.allclose(c.initial_aux(3)[0], 0.25 * np.array([0.02, 0.9])
                           + 0.75 * np.array([0.01, 0.6]))          # not the blend


def test_phase_k_moves_s_by_about_u_k(real):
    """Delta s_k is k/4 of the recorded step's distance (rescaled to the strip), which is
    u_k = (k/4) P when P is the recorded speed times DT: x0's vx is the speed at the end of
    the step, so the two differ by about half the step's own speed change (1.5% under the
    braking here; bound 3%). The other state variables move along the run, not frozen. (On
    the saved design, where P is the sighting speed times DT, the ratio spans 0.41-1.80.)"""
    b2, cells = _new_cells(real)
    for c in cells:
        run = real["runs"][_key(c)]
        step = float(run["x"][I0, 0] - run["x"][I0 - 1, 0]) / b2.stimuli[c.stim_id].source["ds_source"] * DS
        cp = dataclasses.replace(c, P=float(c.x0[3]) * DT)
        for k in range(1, c.n_phases):
            ds_k = float(np.float32(c.x0[0])) - float(cp.initial_state(k)[0])
            assert ds_k == pytest.approx(k / 4 * step, abs=3e-4)
            assert ds_k == pytest.approx(cp.phase_offsets()[k], rel=0.03)
        d = np.abs(c.initial_state(3).astype(float)[1:] - np.asarray(c.x0[1:]))
        assert d.max() > 1e-4                                           # not the frozen state


def test_mirror_members_start_from_exact_mirrors(real):
    """F4/mirror and the E1 mirror rung start every phase from the exact mirror of their
    partner's start (state, prev_action, accel), and ``Cell.mirrored_start`` reproduces it."""
    b2, cells = _new_cells(real)
    pairs = {}
    for c in cells:
        pairs.setdefault(c.pair_id, []).append(c)
    pairs = [sorted(v, key=lambda c: c.mirrored) for v in pairs.values() if len(v) == 2]
    names = sorted(b2.stimuli[m.stim_id].name for _, m in pairs)
    assert names == ["D/14/F4/mirror", "E1/Norisring/874/mirror"]
    for a, m in pairs:
        assert not a.mirrored and m.mirrored
        mm = a.mirrored_start()
        for k in range(a.n_phases):
            assert np.array_equal(T.mirror_state(a.initial_state(k)), m.initial_state(k))
            assert np.array_equal(mm.initial_state(k), m.initial_state(k))
            pa, ac = a.initial_aux(k)
            pm, am = T.mirror_aux(pa, ac)
            pb, ab = m.initial_aux(k)
            assert np.array_equal(pm, pb) and np.array_equal(am, ab)
            assert all(np.array_equal(u, v) for u, v in zip(mm.initial_aux(k), (pb, ab), strict=True))
    chk = T.check_design_details(*real["lap"])["real_phase_starts"]
    assert chk["ok"] and chk["mirror_max_abs"] == 0.0 and chk["lap"] == 29 and chk["n_missing"] == 0


def test_real_phase_check_after_the_layout_lists_per_member_p():
    """Review of R21 (agent-facing finding 3): a mirror pair that starts by s-shift (a run's
    own start) is checkable only once both members have P; each member's P comes from its
    own direction's key, so after the layout the pair starts (k / 4) |Delta P| apart at phase
    k (0.8 mm on the saved E1 pair 504/510). That is listed (p_mismatch), not a failure;
    phase 0 and every 'lap' phase must still be exact mirrors."""
    x0 = (1000.0, 0.1, 0.01, 60.0, 0.2, 0.03, 0.01, 0.5)
    a = T.Cell(0, 0, 7, ("E1",), "real", 60.0, 300.0, W=400, P=3.70774, x0=x0,
               state_source="G1:lap1@E1/x", flags=(T.RUN_START_FLAG,))
    b = dataclasses.replace(a.mirrored_start(), cell_id=1, stim_id=1, mirrored=True, P=3.70883)
    pre = T.real_phase_check([dataclasses.replace(a, P=None), dataclasses.replace(b, P=None)])
    assert pre["ok"] and pre["run_start"] == 2 and pre["p_mismatch"] == []
    post = T.real_phase_check([a, b])
    assert post["ok"] and post["mirror_max_abs"] == 0.0 and len(post["p_mismatch"]) == 3
    assert post["p_mismatch_max"] == pytest.approx(0.75 * (b.P - a.P), abs=1.3e-4)
    same = T.real_phase_check([a, dataclasses.replace(b, P=a.P)])
    assert same["ok"] and same["p_mismatch"] == [] and same["p_mismatch_max"] == 0.0
    lap = dataclasses.replace(a, x0_before=(997.0, 0.12, 0.012, 60.5, 0.21, 0.031, 0.011, 0.2),
                              prev_action_before=(0.0, 0.0), accel_before=(0.0, 0.0), flags=())
    mate = lap.mirrored_start()
    bent = dataclasses.replace(mate, cell_id=1, stim_id=1, mirrored=True, P=3.70883,
                               x0_before=(997.0,) + mate.x0_before[1:7] + (0.21,))
    bad = T.real_phase_check([lap, bent])
    assert not bad["ok"] and bad["mismatches"] and bad["p_mismatch"] == []


def test_designed_cells_unchanged(real, tmp_path):
    """Designed cells keep the s-shift and prev_action0 / accel0 at every phase; their
    records, cells.json and the design hash are exactly the pre-R21 ones."""
    bank, cells = real["design"]
    _, c2 = real["lap"]
    assert c2[:len(cells)] == cells
    pre = {f.name for f in dataclasses.fields(T.Cell)} - {"x0_before", "prev_action_before",
                                                           "accel_before"}
    for c in cells:
        assert c.phase_rule() == "s_shift" and c.x0_before is None
        assert T.cell_record(c) == {k: getattr(c, k) for k in pre}
    laid = [dataclasses.replace(c, P=float(c.v0) * DT, D=c.D if c.D is not None else 300.0)
            for c in cells]
    for c in laid:
        for k in range(c.n_phases):
            u = k / c.n_phases * c.P
            want = np.array([S_C - c.D - u, 0, 0, c.v0, 0, 0, 0, 0], np.float32)
            np.testing.assert_array_equal(c.initial_state(k), want)
            pa, ac = c.initial_aux(k)
            np.testing.assert_array_equal(pa, np.asarray(c.prev_action0, np.float32))
            np.testing.assert_array_equal(ac, np.asarray(c.accel0, np.float32))
    pre_hash = io.sha256_json({"bank": bank.content_hash(),
                               "cells": [{k: getattr(c, k) for k in pre} for c in cells]})
    assert T.design_hash(bank, cells) == pre_hash
    T.save_cells(tmp_path / "cells.json", cells)
    text = (tmp_path / "cells.json").read_text()
    assert "x0_before" not in text and T.load_cells(tmp_path / "cells.json") == cells
    pilot = T.pilot_minigrid()[1][0]                 # an explicit designed x0: still s-shifted
    assert pilot.phase_rule() == "s_shift"


def test_real_cells_round_trip_and_hash(real, tmp_path):
    b2, c2 = real["lap"]
    T.save_cells(tmp_path / "cells.json", c2)
    back = T.load_cells(tmp_path / "cells.json")
    assert back == c2 and T.design_hash(b2, back) == T.design_hash(b2, c2)
    for c, d in zip(c2[-29:], back[-29:], strict=True):
        for k in range(c.n_phases):
            assert np.array_equal(c.initial_state(k), d.initial_state(k))
    assert T.design_hash(*real["lap"]) != T.design_hash(*real["plain"])


def test_missing_lap_step_raises_and_run_start_keeps_the_s_shift(real):
    """A real-lap cell without its recorded step (stand-in states, a pre-R21 design) raises
    on phases > 0 with an error no 'no D / no P yet' handler swallows; a state at its run's
    own start is flagged 'run_start' and keeps the designed s-shift."""
    _, plain = _new_cells(real, "plain")
    c = plain[0]
    assert c.initial_state(0).shape == (8,)
    with pytest.raises(T.LapStepMissingError):
        c.initial_state(1)
    with pytest.raises(T.LapStepMissingError):
        c.initial_aux(2)
    assert not issubclass(T.LapStepMissingError, ValueError)
    chk = T.real_phase_check(real["plain"][1])
    assert not chk["ok"] and chk["n_missing"] == 29
    # The run's own start (trace index 0): the evaluation's designed rolling start.
    bank, cells = real["design"]
    e1 = next(r for r in T.real_state_requests() if r["key"].startswith("E1/Norisring/874")
              and r["agent"] == "G2")
    run = _recorded_run("Norisring", e1["s_target"])
    st0 = _attached(run, _state(run, "G2", "Norisring", 1, 0), 0)
    _, c3 = T.build_real_design(bank, cells, {"G2": {e1["key"]: st0}})
    new = c3[len(cells):]
    assert all(T.RUN_START_FLAG in c.flags and c.phase_rule() == "s_shift" for c in new)
    c = dataclasses.replace(new[0], P=4.0)
    x = c.initial_state(2)
    assert x[0] == np.float32(c.x0[0] - 2.0) and np.array_equal(x[1:], np.float32(c.x0[1:]))
    assert T.real_phase_check(c3)["ok"]
