"""Drivers-lite, analysis side: circuits, segments, lap filter, deltas, injection-recovery."""

from types import SimpleNamespace

import numpy as np
import pytest

from apexzero.drivers import deltas as dl
from apexzero.drivers import laps as lapf
from apexzero.drivers import recovery, segments
from apexzero.drivers import session as sx
from apexzero.drivers.circuits import CIRCUITS, NOT_ON_CALENDAR, circuit_for
from apexzero.drivers.geometry import Centreline, wrap
from apexzero.drivers.registration import Similarity
from apexzero.drivers.synth import CircuitModel, SynthConfig, phase_bumps, qss_model
from apexzero.oracle.qss import GG, speed_profile
from apexzero.tracks.track import from_centreline
from apexzero.vehicle.params import F1_2025


def track_from_pieces(pieces, ds=0.25, name="test"):
    kappa = np.concatenate([np.full(round(length / ds), k) for length, k in pieces])
    theta = np.cumsum(kappa * ds)
    xy = np.stack([np.cumsum(np.cos(theta) * ds), np.cumsum(np.sin(theta) * ds)], axis=1)[::8]
    w = np.full(len(xy), 6.0)
    return from_centreline(name, xy, w, w, ds=2.0, smoothing=0.01)


def two_part_corners(gap, a=700.0, b=400.0, r=35.0):
    """A rectangle whose four 90-degree corners are each two 45-degree arcs ``gap`` apart."""
    half = (0.25 * np.pi * r, 1.0 / r)
    corner = [half, (gap, 0.0), half]
    return track_from_pieces([(a, 0.0), *corner, (b, 0.0), *corner] * 2, name=f"gap{gap:g}")


S_LINE = 350.0   # timing line in the middle of the first straight, as on a real circuit


def reference(line, s_line=S_LINE):
    """QSS speed on a synthetic test track, as a function of lap distance from s_line."""
    tr = line.track
    v = speed_profile(tr.kappa, tr.ds, GG.from_params(F1_2025), closed=True)
    x = (tr.s * (line.length / tr.length) - s_line) % line.length
    order = np.argsort(x)
    return x[order], v[order]


# ----------------------------------------------------------------------------- circuits


def test_circuit_mapping_and_layout_flags():
    assert circuit_for("Hungaroring").tumftm == "Budapest"
    assert circuit_for("Interlagos").tumftm == "SaoPaulo"
    assert circuit_for("Monza").lockbox
    for old in ("Melbourne", "Catalunya", "Yas Marina Circuit", "Zandvoort"):
        with pytest.raises(ValueError, match="predates"):
            circuit_for(old)
        assert circuit_for(old, allow_old_layout=True).tumftm == CIRCUITS[old].tumftm
    with pytest.raises(ValueError, match="not in the TUMFTM"):
        circuit_for("Jeddah")
    with pytest.raises(KeyError):
        circuit_for("Nowhere")
    assert "IMS" in NOT_ON_CALENDAR and "Hockenheim" in NOT_ON_CALENDAR
    assert "Monza" not in NOT_ON_CALENDAR


def test_qss_synthesis_refuses_the_lockbox():
    with pytest.raises(ValueError, match="lockbox"):
        qss_model("Suzuka", "data/tracks/tumftm", np.array([0.25]), np.array([0.25]))


# ----------------------------------------------------------------------------- segments


@pytest.fixture(scope="module")
def chicane_line():
    return Centreline(two_part_corners(gap=40.0))


def test_phases_tile_the_lap_and_chicanes_merge(chicane_line):
    x, v = reference(chicane_line)
    seg = segments.build(chicane_line, S_LINE, x, v)
    cuts = seg.cuts
    assert cuts[0] == 0.0 and cuts[-1] == pytest.approx(chicane_line.length)
    assert np.all(np.diff(cuts) >= 29.9)                 # every phase >= min_phase
    assert len(seg.labels()) == len(cuts) - 1 == 3 * len(seg.complexes)
    # Two 45-degree arcs 40 m apart cannot be timed apart at 3.7 Hz: one complex each.
    assert len(seg.complexes) == 4
    for c in seg.complexes:
        assert len(c.corners) >= 2          # both arcs (curvature may split one in two)
        assert c.x_entry_end <= c.x_min <= c.x_apex_end
        assert c.kind == "slow"


def test_corners_far_apart_stay_separate():
    line = Centreline(two_part_corners(gap=250.0))
    x, v = reference(line)
    assert len(segments.build(line, S_LINE, x, v).complexes) == 8


def test_corner_types():
    assert segments.corner_type(100 / 3.6) == "slow"
    assert segments.corner_type(150 / 3.6) == "medium"
    assert segments.corner_type(250 / 3.6) == "fast"


def test_phase_bumps_integrate_to_one_inside_their_phase(chicane_line):
    x, v = reference(chicane_line)
    seg = segments.build(chicane_line, S_LINE, x, v)
    grid = np.arange(0.25, chicane_line.length, 0.5)
    b = phase_bumps(seg, grid)
    assert np.allclose(b.sum(axis=1) * 0.5, 1.0)
    cuts = seg.cuts
    for p in range(len(cuts) - 1):
        outside = (grid < cuts[p]) | (grid > cuts[p + 1])
        assert np.all(b[p, outside] == 0.0)


# ----------------------------------------------------------------------------- lap filter


def _lap(driver, number, start, duration, pit_out=False):
    third = round(duration / 3, 3)
    return sx.LapRecord(driver, number, start, duration,
                        (third, third, round(duration - 2 * third, 3)), pit_out)


def test_classify_applies_every_rule():
    laps = [
        _lap(1, 1, 100.0, 120.0, pit_out=True),     # out-lap
        _lap(1, 2, 220.0, 80.0),                    # good
        _lap(1, 3, 300.0, 80.5),                    # good (within 102%)
        _lap(1, 4, 380.5, 84.0),                    # not push (> 102% of 80.0)
        _lap(1, 5, 465.0, 81.0),                    # in-lap: lap 6 is an out-lap
        _lap(1, 6, 600.0, 130.0, pit_out=True),
        _lap(1, 7, 730.0, 80.2),                    # deleted
        _lap(1, 8, 811.0, 80.3),                    # yellow during the lap
        _lap(2, 2, 220.0, 80.1),                    # medium tyres
    ]
    rc = [
        {"t": 0.0, "category": "SessionStatus", "qualifying_phase": 1,
         "message": "SESSION STARTED"},
        {"t": 812.0, "category": "Other", "message": "CAR 1 (AAA) TIME 1:20.200 DELETED - TRACK "
                                                    "LIMITS AT TURN 4 LAP 8 14:00:00"},
        {"t": 850.0, "category": "Flag", "flag": "YELLOW", "scope": "Sector", "sector": 3},
        {"t": 870.0, "category": "Flag", "flag": "CLEAR", "scope": "Sector", "sector": 3},
    ]
    stints = [{"driver_number": 1, "lap_start": 1, "lap_end": 10, "compound": "SOFT",
               "tyre_age_at_start": 0},
              {"driver_number": 2, "lap_start": 1, "lap_end": 10, "compound": "MEDIUM",
               "tyre_age_at_start": 0}]
    sess = sx.SessionData(1, 2025, "Qualifying", "X", {}, laps, stints,
                          np.arange(0, 1000, 60.0), np.zeros(17), rc)
    st = lapf.classify(sess)
    assert st[(1, 2)].ok and st[(1, 3)].ok
    assert st[(1, 1)].reasons == ["out-lap"]
    assert st[(1, 4)].reasons == ["not push"]
    assert "in-lap" in st[(1, 5)].reasons
    assert st[(1, 7)].reasons == ["deleted"]
    assert st[(1, 8)].reasons == ["flag"]
    assert st[(2, 2)].reasons == ["tyre MEDIUM"]


def test_gap_ahead_measures_time_to_the_car_in_front():
    length = 2000.0
    v = 50.0
    ego = SimpleNamespace(driver=1, t_start=100.0, duration=length / v,
                          elapsed=lambda x: np.asarray(x) / v,
                          speed=lambda x: np.full(np.shape(x), v))
    t = np.arange(90.0, 160.0, 0.25)
    # Car 2 drives the same pace 30 m ahead (0.6 s); car 3 is 400 m ahead (8 s).
    others = {2: lapf.CarTrack(t, (30.0 + v * (t - 100.0)) % length),
              3: lapf.CarTrack(t, (400.0 + v * (t - 100.0)) % length)}
    x = np.array([100.0, 500.0, 1500.0])
    gap = lapf.gap_ahead(ego, 0.0, others, length, x)
    assert np.allclose(gap, 0.6, atol=1e-6)
    gap3 = lapf.gap_ahead(ego, 0.0, {3: others[3]}, length, x)
    assert np.allclose(gap3, 8.0, atol=1e-6)


def test_gap_ahead_ignores_a_branch_jump_at_a_crossover():
    t_ = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    xy8 = np.stack([300 * np.sin(t_), 300 * np.sin(t_) * np.cos(t_)], axis=1)
    line = Centreline(from_centreline("eight", xy8, np.full(800, 6.0), np.full(800, 6.0), ds=2.0,
                                      smoothing=0.01))
    length = line.length
    grid = np.arange(0.0, length, 0.25)
    r = np.linalg.norm(line.point(grid), axis=1)
    s_a = grid[np.argmin(np.where(grid < 0.25 * length, r, np.inf))]     # the crossing ...
    s_b = grid[np.argmin(np.where(abs(grid - 0.5 * length) < 0.25 * length, r, np.inf))]
    v, s_line = 50.0, 0.25 * length
    ego = SimpleNamespace(driver=1, t_start=100.0, duration=length / v,
                          elapsed=lambda x: np.asarray(x) / v,
                          speed=lambda x: np.full(np.shape(x), v))
    # Car 2 runs 30 m (0.6 s) ahead. At the crossing the feed snaps one of its samples
    # onto the other branch, half a lap away along the centreline.
    t = np.arange(90.0, 100.0 + length / v + 10.0, 0.27)
    s2 = s_line + 30.0 + v * (t - 100.0)
    xy = line.point(s2)
    j = int(np.argmin(np.abs(wrap(s2 - s_b, length))))
    xy[j] = line.point(s_a + 0.5)
    sess = sx.SessionData(1, 2025, "Qualifying", "X", {}, [], [], np.zeros(0), np.zeros(0), [],
                          location={2: (t, xy)})
    tracks = lapf.car_tracks(sess, Similarity(1.0, np.eye(2), np.zeros(2)), line)
    assert len(tracks[2].t) == len(t) - 1                  # only the jump is dropped
    x = np.arange(0.0, length, 2.0)
    gap = lapf.gap_ahead(ego, s_line, tracks, length, x)
    assert np.allclose(gap, 0.6, atol=0.01)


# ----------------------------------------------------------------------------- deltas


def _table(effect, n_laps=4, n_phase=6, noise=0.01, seed=0, teams=3):
    rng = np.random.default_rng(seed)
    keys, drv, team, q, pt = [], [], [], [], []
    base = rng.uniform(3, 10, n_phase)
    for k in range(teams):
        for j in range(2):
            d = 2 * k + j + 1
            for lap in range(n_laps):
                keys.append((d, lap + 1))
                drv.append(d)
                team.append(f"T{k}")
                q.append(1 + lap % 2)
                pt.append(base + (effect if j == 1 else 0.0) + noise * rng.standard_normal(n_phase))
    pt = np.array(pt)
    labels = [(f"C{i // 3:02d}", segments.PHASES[i % 3], "slow") for i in range(n_phase)]
    return dl.LapTable("test", keys, np.array(drv), team, np.array(q), pt.sum(axis=1), pt,
                       labels, {})


def test_pair_delta_is_additive_and_uses_the_fastest_laps():
    effect = np.array([0.05, 0.0, -0.02, 0.0, 0.01, 0.0])
    tab = _table(effect, noise=0.0)
    d = dl.pair_delta(tab, "T0", top_n=2)
    assert (d.a, d.b) == (1, 2)
    assert np.allclose(d.delta, effect)
    assert d.lap_delta == pytest.approx(effect.sum())
    # With noise, top-2 really are each driver's two fastest laps.
    tab = _table(effect, noise=0.05, seed=1)
    d = dl.pair_delta(tab, "T1", top_n=2)
    m = (tab.driver == 4)
    fastest = np.sort(tab.lap_time[m])[:2].mean()
    m_a = (tab.driver == 3)
    assert d.lap_delta == pytest.approx(fastest - np.sort(tab.lap_time[m_a])[:2].mean())


def test_pair_delta_needs_a_shared_q_segment():
    tab = _table(np.zeros(6))
    only_q1 = (tab.driver != 2) | (tab.q_segment == 2)
    sub = tab.rows(only_q1 & ~((tab.driver == 1) & (tab.q_segment == 2)))
    # Driver 1 only in Q1, driver 2 only in Q2: nothing comparable.
    assert dl.pair_delta(sub, "T0") is None


def test_r1_envelope_is_the_best_phase_time_per_car():
    tab = _table(np.zeros(6), noise=0.05)
    r1 = dl.r1_envelope(tab)
    m = np.array([t == "T2" for t in tab.team])
    assert np.allclose(r1["T2"], tab.phase_t[m].min(axis=0))


def test_placebo_collapses_effects_and_split_half_finds_real_ones():
    effect = np.array([0.08, -0.06, 0.05, -0.07, 0.06, -0.05])
    tabs = [_table(effect, n_laps=6, noise=0.02, seed=s) for s in range(3)]
    pb = dl.placebo(tabs, top_n=None, n_shuffles=100)
    assert abs(pb.placebo_mean) < 3 * pb.placebo_mean_se + 1e-3
    assert pb.real_abs_mean > 2 * pb.placebo_abs_mean
    assert pb.ss_p_value < 0.02
    sh = dl.split_half_within(tabs)
    assert sh.rho > 0.8
    null_tabs = [_table(np.zeros(6), n_laps=6, noise=0.02, seed=s) for s in range(3)]
    pb0 = dl.placebo(null_tabs, top_n=None, n_shuffles=100)
    assert pb0.ss_p_value > 0.05
    assert abs(dl.split_half_within(null_tabs).rho) < 0.5


# ----------------------------------------------------------------------------- injection


@pytest.fixture(scope="module")
def synthetic_model(chicane_line):
    x, v = reference(chicane_line)
    seg = segments.build(chicane_line, S_LINE, x, v)
    rng = np.random.default_rng(0)
    dts = rng.uniform(0.14, 0.42, 500)
    length = chicane_line.length
    return CircuitModel("synthetic", chicane_line, Similarity(1.0, np.eye(2), np.zeros(2)),
                        S_LINE + np.array([0.0, length / 3, 2 * length / 3]), x, v, x,
                        np.full(len(x), 1 / 1.006), seg, dts, dts)


def test_injection_recovery_end_to_end(synthetic_model):
    rec = recovery.run(synthetic_model, {("slow", "entry"): 0.040}, pairs=3, laps_per_driver=3,
                       reps=1, top_n=None)
    # Lap times are anchored, so per-lap phase errors cancel over the lap.
    assert np.max(np.abs(rec.lap_total_err())) < 1e-3
    inj = rec.injected > 0
    assert inj.sum() == 4
    rec_eff = rec.pair_est[:, inj].mean()
    assert rec_eff == pytest.approx(0.040, abs=0.015)
    assert np.abs(rec.pair_est[:, ~inj].mean(axis=0)).max() < 0.03
    rows = recovery.budget([rec])
    assert {(r.kind, r.phase) for r in rows} == {("slow", p) for p in segments.PHASES}
    agg = recovery.aggregate_budget([rec])
    assert agg[("slow", "entry")]["injected_ms"] == pytest.approx(160.0)


def test_within_driver_sd_pools_laps_of_one_driver_in_one_q_segment():
    tab = _table(np.zeros(6), n_laps=6, noise=0.02, seed=3)
    # Shifting one Q segment wholesale (track evolution) must not change the estimate.
    shifted = dl.LapTable(tab.session, tab.keys, tab.driver, tab.team, tab.q_segment,
                          tab.lap_time, tab.phase_t + 0.5 * (tab.q_segment == 2)[:, None],
                          tab.labels, {})
    sd = recovery.within_driver_sd(tab)
    assert np.allclose(sd, recovery.within_driver_sd(shifted))
    assert np.all((sd > 0.012) & (sd < 0.03))


def test_calibration_picks_the_wobble_that_matches_the_real_held_out_error(synthetic_model):
    # Pretend the "real" session had the held-out error of a noisy synthetic feed.
    noisy = recovery.run(synthetic_model, {}, pairs=2, laps_per_driver=2, reps=1,
                         synth_cfg=SynthConfig(wobble_sd=0.03), seed=9,
                         held_out=True)
    n_phase = len(synthetic_model.seg.labels())
    tab = _table(np.zeros(n_phase), n_laps=4, n_phase=n_phase, noise=0.03, seed=4)
    tab.labels = synthetic_model.seg.labels()
    cal = recovery.calibrate(synthetic_model, noisy.held_out_s2, tab, wobbles=(0.003, 0.03),
                             pairs=2, laps_per_driver=2, seed=9)
    assert cal.wobble_sd == 0.03
    assert cal.tried[0.003] < cal.tried[0.03]
    assert cal.natural.shape == (len(synthetic_model.seg.labels()),)
    for (real, meas) in cal.by_cell.values():
        assert real > 0 and meas > 0
    assert np.all(cal.natural >= 0.0)
