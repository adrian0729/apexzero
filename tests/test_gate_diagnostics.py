"""scripts/gate_diagnostics.py (GW-1, exploratory): the rollout's equality with timed_laps and
with scripts/eval_rdev.py's JSON, clearance and side on planted traces, the former-crash-corner
and gain-circuit rules, the bootstraps and labels, and the lockbox refusal.

The checkpoint tests need runs/ (local only) and are skipped without it; the rest are fast."""

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TIMED_RUN, TIMED_STEP = "h4r-v1-base-s10", 10_000_000
TIMED_CKPT = ROOT / "runs" / TIMED_RUN / "ckpt" / f"{TIMED_STEP:012d}.pkl"
HAVE_RUNS = TIMED_CKPT.exists()
HAVE_TRACKS = (ROOT / "data" / "tracks" / "tumftm" / "Norisring.csv").exists()
needs_runs = pytest.mark.skipif(not HAVE_RUNS, reason="the H4R checkpoints are not here")
needs_tracks = pytest.mark.skipif(not HAVE_TRACKS, reason="the R-dev tracks are not here")
XLA_2 = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"


@pytest.fixture(scope="module")
def gd():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("gate_diagnostics")
    finally:
        sys.path.remove(str(SCRIPTS))


# ------------------------------------------------------------------------- lockbox


@pytest.mark.parametrize("name", ["Monza", "monza", "Spa_reversed", "Silverstone-mirrored",
                                  "Mexico City", "data/tracks/tumftm/Suzuka.csv"])
def test_lockbox_names_are_refused_before_any_rollout(gd, name, tmp_path):
    def policy(obs):
        raise AssertionError("a policy ran on a lockbox circuit")

    with pytest.raises(PermissionError):
        gd.check_circuit(name)
    with pytest.raises(PermissionError):
        gd.circuit(name)
    with pytest.raises(PermissionError):
        gd.ocp_lap(name)
    with pytest.raises(PermissionError):           # the guard runs on every name first
        gd.evaluate(policy, circuits=("Norisring", name))
    with pytest.raises(PermissionError):
        gd.circuit_inputs([name])


def test_non_rdev_names_are_refused(gd):
    with pytest.raises(ValueError):
        gd.check_circuit("IMS")
    with pytest.raises(ValueError):
        gd.check_circuit("Norisring-mirrored")


def test_planted_lockbox_trace_is_refused(gd, tmp_path, monkeypatch):
    monkeypatch.setattr(gd, "OUT", tmp_path)
    p = gd.trace_path("planted")
    p.parent.mkdir(parents=True)
    np.savez(p, Monza__s=np.zeros((2, 1)))
    with pytest.raises(PermissionError):
        gd.load_traces("planted")


def test_grid_and_networks(gd):
    assert len(gd.NETWORKS) == 28 and gd.N_EVAL == 168
    assert {n.arm for n in gd.NETWORKS} == set(gd.ARMS)
    assert [sum(n.arm == a for n in gd.NETWORKS) for a in gd.ARMS] == [5, 5, 5, 5, 8]
    assert gd.GRID_STEPS == (10_000_000, 18_205_642, 33_144_540, 60_341_763, 109_856_054,
                             199_884_800)
    assert gd.LATE_FROM == 109_856_054 and gd.MID_FROM == 60_341_763
    with pytest.raises(ValueError):
        gd.checkpoint_path(TIMED_RUN, 12_210_553)        # not on the grid


# ------------------------------------------------------------------------- planted traces


def test_window_contains_and_wraps(gd):
    w = gd.Window("X", 0, +1, s_start=200.0, span=60.0, length=1000.0)
    assert w.a == 150.0 and w.width == 160.0
    s = np.array([149.0, 150.0, 230.0, 310.0, 311.0, 1230.0])
    assert w.contains(s).tolist() == [False, True, True, True, False, True]
    assert w.core_distance(np.array([230.0, 190.0, 290.0])).tolist() == [0.0, 10.0, 30.0]
    wrap = gd.Window("X", 1, -1, s_start=980.0, span=40.0, length=1000.0)   # crosses s = 0
    assert wrap.contains(np.array([930.0, 999.0, 0.0, 20.0, 70.0, 71.0, 929.0])).tolist() == [
        True, True, True, True, True, False, False]
    assert wrap.core_distance(np.array([990.0, 10.0, 950.0, 40.0])).tolist() == [
        0.0, 0.0, 30.0, 20.0]


def test_assign_corner_nearest_core_on_overlap(gd):
    wins = [gd.Window("X", 0, +1, 100.0, 50.0, 1000.0),     # window [50, 200]
            gd.Window("X", 1, -1, 220.0, 30.0, 1000.0)]     # window [170, 300]
    assert gd.assign_corner(120.0, wins) == 0
    assert gd.assign_corner(180.0, wins) == 0               # 30 m after corner 0, 40 m before 1
    assert gd.assign_corner(195.0, wins) == 1               # 45 m after 0, 25 m before 1
    assert gd.assign_corner(185.0, wins) == 0               # tie (35 m): the earlier corner
    assert gd.assign_corner(400.0, wins) is None


def test_clearance_and_side(gd):
    wl, wr, n = np.array([6.0, 6.0]), np.array([5.0, 5.0]), np.array([2.0, -3.0])
    assert gd.clearance(wl, wr, n, +1).tolist() == [3.0, 8.0]     # 6 - 1 - n
    assert gd.clearance(wl, wr, n, -1).tolist() == [6.0, 1.0]     # 5 - 1 + n
    assert gd.side_of(2.0, +1) == "inside" and gd.side_of(-2.0, +1) == "outside"
    assert gd.side_of(-2.0, -1) == "inside" and gd.side_of(2.0, -1) == "outside"
    assert gd.side_of(2.0, None) == "straight"
    # An inside exit (n beyond w_left - 1 on a left turn) has c <= 0.
    assert gd.clearance(np.array([6.0]), np.array([5.0]), np.array([5.2]), +1)[0] < 0


def _planted_run(L=1000.0, laps=2, crash_at=None, ds=10.0):
    """A run at constant speed: s every ds metres over two laps; n dips toward the left (inside
    of a left turn at 200-260 m) to 4.0 m at the apex (c = 6 - 1 - 4 = 1)."""
    s = np.arange(0.0, laps * L + 3 * ds, ds)
    u = np.mod(s, L)
    n = np.where((u >= 200) & (u <= 260), 4.0 - np.abs(u - 230.0) / 30.0, 0.0)
    wl, wr = np.full_like(s, 6.0), np.full_like(s, 5.0)
    if crash_at is None:
        end = int(np.nonzero(s >= laps * L)[0][0]) - 1
        return s, n, wl, wr, end, False
    end = int(np.nonzero(s >= crash_at)[0][0])
    n[end] = 5.5                                  # beyond w_left - 1: c = -0.5
    return s, n, wl, wr, end, True


def test_run_passages_two_laps_and_termination(gd):
    w = gd.Window("X", 0, +1, 200.0, 60.0, 1000.0)
    s, n, wl, wr, end, crash = _planted_run()
    ps = gd.run_passages(s, n, wl, wr, end, crash, w)
    assert [p["lap"] for p in ps] == [1, 2]
    assert [p["c"] for p in ps] == pytest.approx([1.0, 1.0])
    assert not any(p["terminated"] for p in ps)
    assert s[ps[0]["i0"]] == 150.0 and s[ps[0]["i1"]] == 310.0
    # Crash on lap 2 inside the window: lap 1 clears, lap 2 terminates with c < 0.
    s, n, wl, wr, end, crash = _planted_run(crash_at=1240.0)
    ps = gd.run_passages(s, n, wl, wr, end, crash, w)
    assert [p["lap"] for p in ps] == [1, 2]
    assert ps[0]["c"] == pytest.approx(1.0) and not ps[0]["terminated"]
    assert ps[1]["c"] == pytest.approx(-0.5) and ps[1]["terminated"]
    assert ps[1]["i1"] == end
    # Crash on lap 1 before the window: no passage at all.
    s, n, wl, wr, end, crash = _planted_run(crash_at=100.0)
    assert gd.run_passages(s, n, wl, wr, end, crash, w) == []


def test_run_passages_ignore_samples_after_lap_two(gd):
    w = gd.Window("X", 0, -1, 980.0, 40.0, 1000.0)     # window [930, 1070] across the line
    s = np.arange(0.0, 2100.0, 10.0)
    n = np.zeros_like(s)
    end = int(np.nonzero(s >= 2000.0)[0][0]) - 1
    ps = gd.run_passages(s, n, np.full_like(s, 6.0), np.full_like(s, 5.0), end, False, w)
    # The start [0, 70], the line crossing [930, 1070] and the last [1930, 2000).
    assert [(s[p["i0"]], s[p["i1"]]) for p in ps] == [(0.0, 70.0), (930.0, 1070.0),
                                                      (1930.0, 1990.0)]
    assert [p["lap"] for p in ps] == [1, 1, 2]
    assert all(p["c"] == 4.0 for p in ps)              # w_right - 1, n = 0


def test_outcomes_on_a_planted_rollout(gd):
    L, T = 100.0, 30
    s = np.zeros((T, 3), np.float32)
    s[:, 0] = np.arange(1, T + 1) * 10.0               # completes 2 laps at step 19
    s[:, 1] = np.minimum(np.arange(1, T + 1), 13) * 10.0   # stops at 130 m (crash at step 12)
    s[:, 2] = np.arange(1, T + 1) * 1.0                # never finishes, never terminates
    alive = np.ones((T, 3), bool)
    alive[12:, 1] = False
    term = np.zeros((T, 3), bool)
    term[12:, 1] = True                                 # frozen state re-terminates every step
    off = term.copy()
    n = np.zeros((T, 3), np.float32)
    n[12, 1] = 3.0
    z = np.zeros((T, 3), bool)
    ro = gd.LapRollout(np.array([True, False, False]), np.array([10.0, np.nan, np.nan]),
                       s.max(0), {}, np.arange(T) * 0.05, s, n, s * 0, alive, term, off, z, z, z,
                       L, 2)
    oc = gd.outcomes(ro)
    assert oc["crash"].tolist() == [False, True, False]
    assert oc["timeout"].tolist() == [False, False, True]
    assert oc["end"].tolist() == [18, 12, T - 1]
    assert oc["term_step"][1] == 12 and oc["cause"][1] == "corridor"
    assert oc["term_s"][1] == pytest.approx(30.0) and oc["term_lap"][1] == 2
    assert oc["term_n"][1] == 3.0


# ------------------------------------------------------------------------- rules


def test_former_crash_corners_rule(gd):
    counts = {("A", 0): 3, ("A", 1): 2, ("B", 4): 7, ("B", 5): 0}
    assert gd.former_crash_corners(counts) == [("A", 0), ("B", 4)]
    assert gd.FORMER_MIN == 3


def test_gain_circuits_rule(gd):
    base = {c: [False] * 5 for c in gd.R_DEV}
    realw = {c: [False] * 5 for c in gd.R_DEV}
    base["Sepang"], realw["Sepang"] = [True] * 5, [True] * 5            # equal: no gain
    realw["Sochi"] = [True, True, False, False, False]                  # 2 > 0: gain
    base["Melbourne"], realw["Melbourne"] = [True] * 3 + [False] * 2, [True] * 2 + [False] * 3
    assert gd.gain_circuits(base, realw) == ["Sochi"]


def test_boot_mean_and_late_labels(gd):
    r = gd.boot_mean([2.0] * 28)
    assert (r["mean"], r["lo"], r["hi"], r["n"]) == (2.0, 2.0, 2.0, 28)
    x = np.arange(28.0) % 5 - 1.5
    a, b = gd.boot_mean(x), gd.boot_mean(x)
    assert a == b and a["lo"] < a["mean"] < a["hi"]
    rng = np.random.default_rng(0)
    m = x[rng.integers(0, 28, (20_000, 28))].mean(1)
    assert a["lo"] == pytest.approx(np.percentile(m, 2.5))
    assert gd.late_label(0.1, 0.4) == {"label": "rising", "holds": ["rising", "flat"]}
    assert gd.late_label(-0.3, 0.3) == {"label": "flat", "holds": ["flat"]}
    assert gd.late_label(-0.4, -0.1) == {"label": "falling", "holds": ["flat", "falling"]}
    assert gd.late_label(-0.8, -0.1)["label"] == "falling"
    assert gd.late_label(-0.8, 0.9) == {"label": "unresolved", "holds": ["unresolved"]}
    assert gd.late_label(0.0, 0.7)["label"] == "unresolved"     # touches 0: not above it


def test_change_summary_and_best_checkpoint(gd):
    c = gd.change_summary([1, 0, 0, -2, 3])
    assert (c["rise"], c["level"], c["fall"], c["median"]) == (2, 2, 1, 0.0)
    steps = gd.GRID_STEPS
    b = gd.best_checkpoint(dict(zip(steps, [2, 5, 7, 7, 6, 6], strict=True)))
    assert b["best_before_final"] and b["best_minus_final"] == 1 and b["best_step"] == steps[2]
    b = gd.best_checkpoint(dict(zip(steps, [2, 5, 7, 7, 6, 7], strict=True)))   # tie: final
    assert not b["best_before_final"] and b["best_step"] == steps[-1]


def test_boot_corner_diff(gd):
    base = np.array([[1.0, 2.0, 0.5]] * 5)
    realw = base + np.array([0.2, 0.4, 0.3])
    r = gd.boot_corner_diff(base, realw, n_boot=500)
    assert r["dc"] == pytest.approx(0.3) and r["lo"] == pytest.approx(0.3)
    assert r["hi"] == pytest.approx(0.3) and r["n_corners"] == 3
    # A corner without passages in one network is averaged over the others; a corner without
    # any in a draw is left out of that draw.
    b2 = base.copy()
    b2[0, 2] = np.nan
    r2 = gd.boot_corner_diff(b2, realw, n_boot=2000)
    assert r2["dc"] == pytest.approx(0.3)
    b3 = base.copy()
    b3[1:, 2] = np.nan                     # only network 0 has corner 2
    b3[0, 2] = 0.5
    r3 = gd.boot_corner_diff(b3, realw, n_boot=2000)
    assert r3["draws_with_dropped"] > 0 and r3["dc"] == pytest.approx(0.3)
    # Resampling is within arms: spread across networks widens the CI.
    rb = base + np.arange(5.0)[:, None] * 0.1
    r4 = gd.boot_corner_diff(rb, rb + 0.3)
    assert r4["lo"] < 0.3 < r4["hi"]
    assert gd.boot_corner_diff(np.zeros((5, 0)), np.zeros((5, 0)))["dc"] is None


def test_section8_d3_rows(gd):
    q = {"dc": 0.5, "band": [0.22, 0.88]}
    assert gd.section8_d3(0.8, q, 0.2, q) == {
        "v1": "margin-consistent: s_in >= 2/3 and dc in the band", "v3": "s_in < 1/3"}
    assert gd.section8_d3(0.7, {**q, "dc": 0.1}, None, q)["v1"].endswith("below the band")
    assert gd.section8_d3(0.7, {**q, "dc": 1.0}, 0.5, q)["v1"].endswith("above the band")
    assert gd.section8_d3(0.7, q, 0.5, q)["v3"] == "1/3 <= s_in < 2/3 (no row)"


def test_platform_check_and_wilson(gd):
    full = {c: {"completion": 1.0} for c in gd.R_DEV}
    stored = {"a": dict(full), "b": dict(full)}
    mac = {"a": dict(full), "b": {**full, "Sochi": {"completion": 0.5}}}   # 0.5 is not > 0.5
    r = gd.platform_check(stored, mac)
    assert (r["calls_agree"], r["calls_total"]) == (23, 24)
    assert r["disagreements"] == [{"run": "b", "circuit": "Sochi", "stored": 1.0, "mac": 0.5}]
    assert r["per_network"]["b"]["dC"] == -1 and r["networks_C_within_1"] == 2
    lo, hi = gd.wilson(900, 1000)
    assert lo == pytest.approx(0.8798, abs=1e-4) and hi == pytest.approx(0.9171, abs=1e-4)


def test_adr_keys_are_the_pilot_audits(gd):
    import jax

    from apexzero.tracks.levels import make_levels
    a = make_levels(jax.random.PRNGKey(2026), "a_dr")
    b = make_levels(jax.random.PRNGKey(2026), "a_dr", n=150)     # pilot_audit's first 150
    assert a.keys.shape[0] == 1000 and a.generator == "train"
    assert np.array_equal(np.asarray(a.keys[:150]), np.asarray(b.keys))
    assert np.array_equal(np.asarray(a.ctx[:150]), np.asarray(b.ctx))


# ------------------------------------------------------------------------- equality checks


@needs_tracks
def test_rollout_matches_timed_laps_on_a_fixed_policy(gd):
    """Without a checkpoint: a fixed linear policy on Norisring, 4 runs."""
    import jax.numpy as jnp

    from apexzero.env.adapters import reference_speed
    from apexzero.eval.laps import timed_laps

    def policy(obs):
        return jnp.stack([-0.8 * obs[:, 204], 0.3 + 0.0 * obs[:, 200]], axis=-1)

    tr = gd.circuit("Norisring")
    v0 = float(reference_speed(tr)[0]) * 0.9
    a = timed_laps(policy, tr, v0, runs=4)
    b = gd.rollout_laps(policy, tr, v0, runs=4)
    _assert_same(a, b)


def _assert_same(a, b):
    assert np.array_equal(a.completed, b.completed)
    assert np.array_equal(a.lap_time, b.lap_time, equal_nan=True)
    assert np.array_equal(a.s_reached, b.s_reached)
    for k in ("t", "s", "n", "vx", "action"):
        assert np.array_equal(a.trace[k], b.trace[k]), k
    assert a.completion_rate == b.completion_rate
    assert np.array_equal(a.median_lap, b.median_lap, equal_nan=True)


@needs_runs
@needs_tracks
def test_rollout_matches_timed_laps_on_a_checkpoint(gd):
    """D1's timed checkpoint on two circuits, 10 runs: every LapResult field is equal."""
    from apexzero.env.adapters import reference_speed
    from apexzero.eval.laps import timed_laps

    d, _ = gd.load_checkpoint(TIMED_CKPT)
    policy, _, _ = gd.make_policy(d)
    for name in ("Norisring", "Zandvoort"):
        tr = gd.circuit(name)
        v0 = float(reference_speed(tr)[0]) * 0.9
        a = timed_laps(policy, tr, v0, runs=10)
        b = gd.rollout_laps(policy, tr, v0, runs=10)
        _assert_same(a, b)
        oc = gd.outcomes(b)
        assert np.array_equal(oc["crash"] | oc["timeout"], ~b.completed)
        k = oc["term_step"][oc["crash"]]
        assert np.all(~b.alive[k, np.nonzero(oc["crash"])[0]])


@needs_runs
@needs_tracks
def test_eval_one_matches_eval_rdev_json(gd, tmp_path):
    """scripts/eval_rdev.py on a copy of the timed checkpoint and the new script's eval-one,
    each in its own process with the same XLA threads: every circuit's completion, median lap
    and gap are equal."""
    env = {**os.environ, "XLA_FLAGS": XLA_2}
    ck = tmp_path / TIMED_CKPT.name
    shutil.copy(TIMED_CKPT, ck)
    subprocess.run([sys.executable, str(SCRIPTS / "eval_rdev.py"), str(ck)], cwd=ROOT, env=env,
                   check=True, capture_output=True)
    ref = json.loads(ck.with_suffix(".eval_rdev.json").read_text())
    out = tmp_path / "eval.json"
    subprocess.run([sys.executable, str(SCRIPTS / "gate_diagnostics.py"), "eval-one", "--run",
                    TIMED_RUN, "--step", str(TIMED_STEP), "--out", str(out), "--threads", "2"],
                   cwd=ROOT, env=env, check=True, capture_output=True)
    rec = json.loads(out.read_text())
    assert ref["timesteps"] == rec["timesteps"] == TIMED_STEP
    for name in gd.R_DEV:
        r, m = ref[name], rec["circuits"][name]
        assert r["completion"] == m["completion"], name
        for k in ("median_lap_s", "gap"):
            assert (r[k] is None or not np.isfinite(r[k])) == (m[k] is None), (name, k)
            if m[k] is not None:
                assert r[k] == m[k], (name, k)
        assert r["ocp_lap_s"] == m["ocp_lap_s"]
    done = [n for n in gd.R_DEV if ref[n]["completion"] > 0.5]
    assert rec["C"] == len(done)


@needs_runs
def test_bindings_of_the_28_finals(gd):
    bound = gd.bound_finals()
    assert len(bound) == 28
    for net in gd.NETWORKS:
        gd.checkpoint_paths(net.name)
        path = gd.checkpoint_path(net.name, gd.FINAL)
        d, sha = gd.load_checkpoint(path, bound[net.name]["sha256"])
        assert sha == bound[net.name]["sha256"] and int(d["timesteps"]) == gd.FINAL
    with pytest.raises(gd.CheckFailed):
        gd.load_checkpoint(gd.checkpoint_path(TIMED_RUN, gd.FINAL), "0" * 64)
