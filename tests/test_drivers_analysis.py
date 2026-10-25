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


def test_retype_changes_types_only(chicane_line):
    x, v = reference(chicane_line)
    seg = segments.build(chicane_line, S_LINE, x, v)
    assert {c.kind for c in seg.complexes} == {"slow"}
    # An envelope 100 km/h above the reference puts every apex between 120 and 200 km/h.
    faster = segments.retype(seg, x, v + 100 / 3.6)
    assert np.array_equal(faster.cuts, seg.cuts)
    assert [c.name for c in faster.complexes] == [c.name for c in seg.complexes]
    assert {c.kind for c in faster.complexes} == {"medium"}
    for c0, c1 in zip(seg.complexes, faster.complexes, strict=True):
        assert c1.v_type == pytest.approx(c0.v_min + 100 / 3.6, abs=0.5)


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


def _with_phase_times(tab, pt):
    return dl.LapTable(tab.session, tab.keys, tab.driver, tab.team, tab.q_segment,
                       pt.sum(axis=1), pt, tab.labels, tab.acronym)


def test_pair_delta_is_additive_and_uses_the_fastest_laps_of_each_segment():
    effect = np.array([0.05, 0.0, -0.02, 0.0, 0.01, 0.0])
    tab = _table(effect, noise=0.0)
    d = dl.pair_delta(tab, "T0", top_n=2)
    assert (d.a, d.b) == (1, 2)
    assert np.allclose(d.delta, effect)
    assert d.lap_delta == pytest.approx(effect.sum())
    # With noise and 3 laps per driver per segment, top-2 are each driver's two fastest
    # laps *within* each Q segment; the segments get equal weight (2 + 2 laps each).
    tab = _table(effect, n_laps=6, noise=0.05, seed=1)
    d = dl.pair_delta(tab, "T1", top_n=2)
    assert d.segments == ((1, 2, 2), (2, 2, 2)) and d.var_factor == pytest.approx(0.5)

    def top2(drv, q):
        return np.sort(tab.lap_time[(tab.driver == drv) & (tab.q_segment == q)])[:2].mean()

    expect = np.mean([top2(4, q) - top2(3, q) for q in (1, 2)])
    assert d.lap_delta == pytest.approx(expect)


def test_pair_delta_never_compares_laps_across_q_segments():
    effect = np.array([0.05, 0.0, -0.02, 0.0, 0.01, 0.0])
    tab = _table(effect, n_laps=6, noise=0.03, seed=5)
    # Track evolution: every lap in Q2 is faster, by a different amount in each phase,
    # enough to reorder the drivers' laps across segments. The deltas must not move.
    evo = -np.linspace(0.02, 0.12, 6)
    shifted = _with_phase_times(tab, tab.phase_t + (tab.q_segment == 2)[:, None] * evo)
    for team in ("T0", "T1", "T2"):
        for top_n in (1, 2, None):
            d0, d1 = dl.pair_delta(tab, team, top_n), dl.pair_delta(shifted, team, top_n)
            assert np.allclose(d0.delta, d1.delta, atol=1e-12)
            assert d0.lap_delta == pytest.approx(d1.lap_delta, abs=1e-12)
    # Driver 2 only ran in Q2: only Q2 is compared, whatever driver 1 did in Q1.
    sub = shifted.rows(~((shifted.driver == 2) & (shifted.q_segment == 1)))
    d = dl.pair_delta(sub, "T0", top_n=2)
    q2 = dl.pair_delta(shifted.rows(shifted.q_segment == 2), "T0", top_n=2)
    assert d.q_segments == (2,) and np.allclose(d.delta, q2.delta)
    # Unequal lap counts: inverse-variance weights 1 / (1/n_A + 1/n_B).
    sub = tab.rows(~((tab.driver <= 2) & (tab.q_segment == 1) & (np.array(
        [k[1] for k in tab.keys]) > 2)))
    d = dl.pair_delta(sub, "T0", top_n=None)
    assert d.segments == ((1, 1, 1), (2, 3, 3))
    w = np.array([0.5, 1.5]) / 2.0
    parts = [dl.pair_delta(sub.rows(sub.q_segment == q), "T0", None).delta for q in (1, 2)]
    assert np.allclose(d.delta, w[0] * parts[0] + w[1] * parts[1])
    assert d.var_factor == pytest.approx(0.5)


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
    y = dl.r1_residuals(tab)
    assert np.all(y >= 0) and np.allclose(y[m].min(axis=0), 0.0)


def test_placebo_and_split_half_separate_real_effects_from_none():
    effect = np.array([0.08, -0.06, 0.05, -0.07, 0.06, -0.05])
    tabs = [_table(effect, n_laps=6, noise=0.02, seed=s) for s in range(3)]
    pb = dl.placebo(tabs, top_n=None, n_shuffles=100)
    assert pb.n_pairs == 9 and pb.n_cells == 54
    assert pb.real_abs_mean > 2 * pb.placebo_abs_mean
    assert all(r > 2 for r, _ in pb.abs_ratio_by_cell().values())
    assert pb.ss_p_value < 0.02
    sh = dl.split_half_within(tabs)
    assert sh.rho > 0.8 and sh.p_value < 0.02 and sh.n_pairs == 9
    null_tabs = [_table(np.zeros(6), n_laps=6, noise=0.02, seed=s) for s in range(3)]
    pb0 = dl.placebo(null_tabs, top_n=None, n_shuffles=100)
    assert pb0.ss_p_value > 0.05
    assert all(0.5 < r < 2 for r, _ in pb0.abs_ratio_by_cell().values())
    # A correct 2-sigma limit (2 x 0.02 s at 2 + 2 laps, scaled to 3 + 3 laps in each of
    # two segments) is exceeded by about 5% of shuffled cells; one half as wide is not.
    lim = {("slow", ph): 0.04 for ph in segments.PHASES}
    share, _, n = pb0.exceed_share(lim)
    assert n == 54 and 0.025 < share < 0.08
    narrow, _, _ = pb0.exceed_share({k: 0.5 * v for k, v in lim.items()})
    assert narrow > 0.2
    sh0 = dl.split_half_within(null_tabs)
    assert abs(sh0.rho) < 0.5 and sh0.p_value > 0.05


def test_profile_removes_a_uniform_pace_gap_and_keeps_a_local_one():
    tab = _table(np.zeros(6), noise=0.0)
    slower = tab.phase_t.copy()
    slower[tab.driver == 2] *= 1.01        # every phase 1% slower
    tab = _with_phase_times(tab, slower)
    shares = dl.time_shares(tab)
    agg = dl.aggregate(dl.pair_delta(tab, "T0").delta, tab.labels, "type")
    assert min(abs(v) for v in agg.values()) > 1e-3
    assert all(abs(v) < 1e-12 for v in dl.profile_deltas(agg, shares).values())
    # A local effect survives (diluted by its share of the lap), and profiles sum to zero.
    local = slower.copy()
    local[tab.driver == 2, 0] += 0.05
    tab = _with_phase_times(tab, local)
    prof = dl.profile_deltas(dl.aggregate(dl.pair_delta(tab, "T0").delta, tab.labels, "type"),
                             dl.time_shares(tab))
    assert prof[("slow", "entry")] > 0.03
    assert abs(sum(prof.values())) < 1e-12


def test_split_half_profile_ignores_a_reliable_overall_gap():
    """Pairs differ only by a uniform pace gap that repeats in every session: the raw
    statistic calls that reliable, the profile does not."""
    gaps = np.linspace(-0.012, 0.012, 8)
    tabs = []
    for s in range(4):
        tab = _table(np.zeros(6), noise=0.002, seed=s, teams=len(gaps))
        pt = tab.phase_t.copy()
        for k, g in enumerate(gaps):
            pt[(np.array(tab.team) == f"T{k}") & (tab.driver % 2 == 0)] *= 1.0 + g
        tabs.append(_with_phase_times(tab, pt))
    raw = dl.split_half_across(tabs, n_perm=200)
    prof = dl.split_half_across(tabs, n_perm=200, profile=True)
    assert raw.rho > 0.8
    assert abs(prof.rho) < 0.5 and prof.p_value > 0.05


def test_split_half_halves_stay_inside_q_segments():
    tab = _table(np.zeros(6), n_laps=6, noise=0.02, seed=7)
    h1, h2 = dl._halves(tab, "T0")
    for q in (1, 2):
        for drv in (1, 2):
            m = (tab.driver == drv) & (tab.q_segment == q)
            assert np.isin(np.flatnonzero(m), h1).sum() == 2
            assert np.isin(np.flatnonzero(m), h2).sum() == 1
    # A segment where one driver has a single lap cannot be split; it is left out.
    sub = tab.rows(~((tab.driver == 1) & (tab.q_segment == 1) & (np.array(
        [k[1] for k in tab.keys]) > 1)))
    h1, h2 = dl._halves(sub, "T0")
    assert set(sub.q_segment[np.r_[h1, h2]]) == {2}


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


def test_held_out_rates_and_persistent_errors(synthetic_model):
    nat = np.full(len(synthetic_model.seg.labels()), 0.02)
    base = recovery.run(synthetic_model, {("slow", "entry"): 0.040}, pairs=3,
                        laps_per_driver=3, reps=2, seed=31, natural=nat)
    assert list(base.rep) == [0, 0, 0, 1, 1, 1] and set(base.lap_rep) == {0, 1}
    assert base.subset([1]).pair_est.shape == (3, base.pair_est.shape[1])
    rates = recovery.held_out_rates([base])
    fpr, n = rates["fpr"]
    assert n == 6 * int((base.injected == 0).sum()) and 0.0 <= fpr <= 0.3
    assert set(rates["detection"]) == {("slow", "entry")}
    # Persistent per-car errors come from their own random stream: the truth is
    # unchanged and only the estimates move, by an amount that grows with the error.
    small = SynthConfig(driver_wobble_sd=0.002, car_offset_sd=0.2)
    large = SynthConfig(driver_wobble_sd=0.02, car_offset_sd=2.0)
    runs = [recovery.run(synthetic_model, {("slow", "entry"): 0.040}, pairs=3,
                         laps_per_driver=3, reps=2, seed=31, natural=nat, synth_cfg=c)
            for c in (small, large)]
    for r in runs:
        assert np.array_equal(r.pair_true, base.pair_true)
    sys_small = recovery.systematic_error([base], [runs[0]])
    sys_large = recovery.systematic_error([base], [runs[1]])
    for k in sys_small:
        assert 0.0 < sys_small[k][0] < sys_large[k][0]


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
    # Pretend the "real" session had the held-out error of a synthetic feed whose wobble
    # (0.025) is not on the calibration grid, drawn with a seed the calibration never
    # uses: the nearest grid wobble must win, not an exact replay.
    real = recovery.run(synthetic_model, {}, pairs=3, laps_per_driver=3, reps=1,
                        synth_cfg=SynthConfig(wobble_sd=0.025), seed=21, held_out=True)
    n_phase = len(synthetic_model.seg.labels())
    tab = _table(np.zeros(n_phase), n_laps=4, n_phase=n_phase, noise=0.03, seed=4)
    tab.labels = synthetic_model.seg.labels()
    cal = recovery.calibrate(synthetic_model, real.held_out_s2, tab,
                             wobbles=(0.003, 0.01, 0.03), pairs=3, laps_per_driver=3, seed=5)
    assert cal.wobble_sd == 0.03
    assert cal.tried[0.003] < cal.tried[0.01] < cal.tried[0.03]
    assert cal.synth_held_sd != pytest.approx(cal.real_held_sd, rel=1e-6)
    assert cal.natural.shape == (len(synthetic_model.seg.labels()),)
    for (real, meas) in cal.by_cell.values():
        assert real > 0 and meas > 0
    assert np.all(cal.natural >= 0.0)
