"""Drivers-lite, timing side: geometry, registration, session parsing, smoother, s(t)."""

import numpy as np
import pytest

from apexzero.drivers import session as sx
from apexzero.drivers.geometry import Centreline, wrap
from apexzero.drivers.progress import LapInput, fit_session, held_out_sector, track_samples
from apexzero.drivers.registration import (
    Similarity,
    corridor_excursion,
    inside_offset,
    register,
    umeyama,
)
from apexzero.drivers.smoother import ANCHOR, POS, SPEED, SmootherConfig, hermite, smooth
from apexzero.tracks.track import from_centreline


def track_from_pieces(pieces, ds=0.25, name="test"):
    """Closed track from (length, curvature) pieces whose turning angles sum to 2 pi."""
    kappa = np.concatenate([np.full(round(length / ds), k) for length, k in pieces])
    theta = np.cumsum(kappa * ds)
    xy = np.stack([np.cumsum(np.cos(theta) * ds), np.cumsum(np.sin(theta) * ds)], axis=1)
    xy = xy[::8]
    w = np.full(len(xy), 6.0)
    return from_centreline(name, xy, w, w, ds=2.0, smoothing=0.01)


def rounded_rectangle(a=600.0, b=300.0, r=40.0):
    """Straights of length a and b joined by quarter circles of radius r."""
    arc = (0.5 * np.pi * r, 1.0 / r)
    return track_from_pieces([(a, 0.0), arc, (b, 0.0), arc] * 2, name="rect")


def lopsided(n=800):
    """A closed curve with no rotational or mirror symmetry (registration is unique)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.stack([300 * np.cos(t) + 60 * np.cos(2 * t), 200 * np.sin(t) + 40 * np.sin(3 * t)
                   + 30 * np.cos(3 * t)], axis=1)
    w = np.full(n, 6.0)
    return from_centreline("lopsided", xy, w, w, ds=2.0, smoothing=0.01)


def figure_eight(scale=300.0, n=800):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.stack([scale * np.sin(t), scale * np.sin(t) * np.cos(t)], axis=1)
    w = np.full(n, 6.0)
    return from_centreline("eight", xy, w, w, ds=2.0, smoothing=0.01)


@pytest.fixture(scope="module")
def rect_line():
    return Centreline(rounded_rectangle())


# ----------------------------------------------------------------------------- geometry


def test_projection_is_exact_between_vertices(rect_line):
    s = np.array([10.3, 250.71, 1000.05, rect_line.length - 0.2])
    xy = rect_line.point(s)
    s_hat, d = rect_line.nearest(xy)
    assert np.allclose(wrap(s_hat - s, rect_line.length), 0.0, atol=1e-6)
    assert np.all(d < 1e-6)
    # A point 3 m off the line projects to the same s on a straight.
    tan = rect_line.tangent(s[:1])
    off = xy[:1] + 3.0 * np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    s_off, d_off = rect_line.nearest(off)
    assert s_off[0] == pytest.approx(s[0], abs=1e-6)
    assert d_off[0] == pytest.approx(3.0, abs=1e-6)


def test_windowed_tracking_does_not_jump_at_a_crossing():
    line = Centreline(figure_eight())
    # Drive one lap and a bit at 60 m/s, sampled at 3.7 Hz, through the crossing twice.
    t = np.arange(0, (line.length + 200) / 60.0, 0.27)
    s_true = 5.0 + 60.0 * t
    s, d = track_samples(line, t, line.point(s_true) + 0.3, v0=60.0)
    assert np.allclose(s - s[0], s_true - s_true[0], atol=1.0)
    # The global projection, by contrast, may pick the other branch near the crossing.
    assert np.all(np.diff(s) > 0)


# ----------------------------------------------------------------------------- registration


def test_umeyama_recovers_a_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 2)) * 100
    rot = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
    tf = Similarity(1.01, rot, np.array([30.0, -12.0]))
    est = umeyama(src, tf.apply(src))
    assert est.scale == pytest.approx(1.01, rel=1e-9)
    assert np.allclose(est.rotation, rot)
    assert np.allclose(est.inverse(tf.apply(src)), src)


@pytest.mark.parametrize("reflect", [False, True])
def test_register_recovers_rotation_scale_and_handedness(reflect):
    rect_line = Centreline(lopsided())
    rng = np.random.default_rng(1)
    s = rng.uniform(0, rect_line.length, 3000)
    on_line = rect_line.point(s) + rng.normal(scale=0.3, size=(len(s), 2))
    ang = np.radians(137.0)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    if reflect:
        rot = rot @ np.diag([1.0, -1.0])
    truth = Similarity(1.0 / 0.995, rot, np.array([500.0, -250.0]))
    feed = truth.inverse(on_line)          # what the feed would report
    reg = register(feed, rect_line, length_openf1=rect_line.length * 0.995)
    assert reg.transform.reflected == reflect
    assert np.allclose(reg.transform.rotation, rot, atol=2e-3)
    assert reg.rms_m < 0.5
    assert reg.transform.scale == pytest.approx(1 / 0.995, rel=2e-3)
    assert abs(reg.length_mismatch) < 2e-3
    assert reg.accepted


def test_registration_gate_rejects_a_length_mismatch(rect_line):
    rng = np.random.default_rng(2)
    feed = rect_line.point(rng.uniform(0, rect_line.length, 2000))
    reg = register(feed, rect_line, length_openf1=rect_line.length * 1.03)
    assert reg.rms_m < 0.1
    assert not reg.accepted
    assert "length mismatch" in reg.reason


def test_corridor_and_inside_offset_diagnostics(rect_line):
    """A line cutting 2 m to the inside of every corner stays in the 6 m corridor; a
    section displaced by 25 m (a different layout) leaves it."""
    eye = Similarity(1.0, np.eye(2), np.zeros(2))
    s = np.linspace(0, rect_line.length, 4000, endpoint=False)
    tan = rect_line.tangent(s)
    left = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    k = rect_line.curvature(s)
    inside = 2.0 * np.sign(k) * (np.abs(k) > 1 / 150)
    pts = rect_line.point(s) + left * inside[:, None]
    # The diagnostic averages over a smoothed-curvature window a little wider than the
    # arcs, where this test line is back on the centreline, so it reads 1-2 m.
    assert 1.0 < inside_offset(pts, rect_line, eye) <= 2.0
    assert abs(inside_offset(rect_line.point(s), rect_line, eye)) < 0.01
    frac, p99 = corridor_excursion(pts, rect_line, eye)
    assert frac == 0.0 and p99 < -3.5
    moved = pts.copy()
    moved[:200] += left[:200] * 25.0
    frac, p99 = corridor_excursion(moved, rect_line, eye)
    assert frac == pytest.approx(0.05, abs=0.002) and p99 > 15.0


# ----------------------------------------------------------------------------- session parsing


def test_parse_times_matches_fromisoformat():
    stamps = ["2025-09-06T14:00:10.426000+00:00", "2025-09-06T14:00:00+00:00"]
    from datetime import datetime

    ref = [datetime.fromisoformat(s).timestamp() for s in stamps]
    assert np.allclose(sx.parse_times(stamps), ref, atol=1e-6)


def _sess(race_control, laps=(), stints=()):
    rc = [dict(m) for m in race_control]
    return sx.SessionData(1, 2025, "Qualifying", "X", {}, list(laps), list(stints),
                          np.array([0.0, 60.0]), np.array([0.0, 0.0]), rc)


def test_race_control_parsing():
    sess = _sess([
        {"t": 0.0, "category": "SessionStatus", "qualifying_phase": 1,
         "message": "SESSION STARTED"},
        {"t": 100.0, "category": "Other", "message":
         "CAR 55 (SAI) TIME 1:20.470 DELETED - TRACK LIMITS AT TURN 7 LAP 3 16:03:35"},
        {"t": 110.0, "category": "Other", "message":
         "CAR 30 (LAW) LAP DELETED - TRACK LIMITS AT TURN 6 LAP 6 16:16:28 (PIT)"},
        {"t": 200.0, "category": "Flag", "flag": "YELLOW", "scope": "Sector", "sector": 7},
        {"t": 230.0, "category": "Flag", "flag": "CLEAR", "scope": "Sector", "sector": 7},
        {"t": 900.0, "category": "SessionStatus", "qualifying_phase": 2,
         "message": "SESSION STARTED"},
    ])
    assert sx.deleted_laps(sess) == {(55, 3), (30, 6)}
    assert sx.q_segment(sess, 50.0) == 1 and sx.q_segment(sess, 950.0) == 2
    assert sx.flag_intervals(sess) == [(200.0, 230.0, "YELLOW")]


# ----------------------------------------------------------------------------- smoother


def test_hermite_reproduces_a_cubic():
    t = np.array([0.0, 1.0, 2.5])
    f = lambda x: 2 + 3 * x - x**2 + 0.5 * x**3
    df = lambda x: 3 - 2 * x + 1.5 * x**2
    tq = np.linspace(0, 2.5, 11)
    assert np.allclose(hermite(t, f(t), df(t), tq), f(tq))


def test_smoother_fuses_positions_speed_and_anchors():
    cfg = SmootherConfig(jerk_psd=400.0, sigma_pos=2.2, sigma_speed=0.35)
    errs, z = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        # A car slowing from 80 to 30 m/s and accelerating again, sampled like the feed.
        tt = np.linspace(0, 20, 20001)
        v = 55 + 25 * np.cos(2 * np.pi * tt / 20)
        s = np.r_[0.0, np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(tt))]
        tp = np.sort(rng.uniform(0, 20, 75))
        tv = np.sort(rng.uniform(0, 20, 75))
        zp = np.interp(tp, tt, s) + rng.normal(scale=2.0, size=len(tp))
        zv = np.round(np.interp(tv, tt, v) * 3.6) / 3.6
        ta = np.array([0.0, 7.3, 20.0])
        za = np.interp(ta, tt, s)
        t = np.r_[tp, tv, ta]
        kind = np.r_[np.full(len(tp), POS), np.full(len(tv), SPEED), np.full(3, ANCHOR)]
        fit = smooth(t, kind, np.r_[zp, zv, za], cfg=cfg)
        assert np.allclose(fit.s_at(ta), za, atol=5e-3)      # anchors honoured to ~mm
        grid = np.linspace(0.5, 19.5, 200)
        e = fit.s_at(grid) - np.interp(grid, tt, s)
        errs.append(e)
        z.append(e / fit.sigma_s_at(grid))
        sq = np.linspace(s[100], s[-100], 50)
        assert np.allclose(fit.s_at(fit.t_at(sq)), sq, atol=0.05)   # consistent inverse
    errs, z = np.concatenate(errs), np.concatenate(z)
    # 2 m position noise, but s is good to ~0.3 m RMS between anchors ...
    assert np.sqrt(np.mean(errs**2)) < 0.5
    assert np.max(np.abs(errs)) < 1.5
    # ... and the smoother's own uncertainty is honest (not overconfident).
    assert np.sqrt(np.mean(z**2)) < 1.2


def test_gate_rejects_a_position_glitch():
    t = np.arange(0, 10, 0.25)
    z = 50.0 * t
    z[20] += 80.0
    fit = smooth(t, np.full(len(t), POS), z)
    assert not fit.used[20] and fit.used.sum() == len(t) - 1


def test_gate_cannot_lock_out_after_a_persistent_shift():
    """If positions really are elsewhere (here a persistent +80 m), the filter re-acquires
    after a few rejections instead of rejecting every later sample."""
    cfg = SmootherConfig(jerk_psd=100.0, max_rejects=3)
    t = np.arange(0, 20, 0.25)
    z = 50.0 * t + np.where(t >= 5.0, 80.0, 0.0)
    fit = smooth(t, np.full(len(t), POS), z, cfg=cfg)
    late = t >= 8.0
    assert fit.used[late].all()
    assert np.allclose(fit.x[late, 0], z[late], atol=1.0)


# ----------------------------------------------------------------------------- s(t) per lap


def _lap_inputs(line, n_laps=6, seed=0, jitter=0.1, s_line=37.0):
    """Constant-pace laps on a track: v(x) = 60 + 15 sin(2 pi x / L)."""
    rng = np.random.default_rng(seed)
    length = line.length
    x = np.linspace(-400, length + 400, 40001)
    v = 60 + 15 * np.sin(2 * np.pi * x / length)
    t_of_x = np.r_[0.0, np.cumsum(np.diff(x) / (0.5 * (v[1:] + v[:-1])))]
    t_of_x -= np.interp(0.0, x, t_of_x)
    out, truth = [], []
    for k in range(n_laps):
        t0 = 1000.0 + 300.0 * k
        lap_t = np.interp([0.0, length / 3, 2 * length / 3, length], x, t_of_x)
        tp = np.sort(rng.uniform(t0 - 4, t0 + lap_t[-1] + 4, int(3.7 * (lap_t[-1] + 8))))
        tv = np.sort(rng.uniform(t0 - 4, t0 + lap_t[-1] + 4, int(3.7 * (lap_t[-1] + 8))))
        xp = np.interp(tp - t0, t_of_x, x) + rng.normal(scale=1.5, size=len(tp))
        sp, _ = track_samples(line, tp, line.point(s_line + xp))
        vv = np.round(np.interp(tv - t0, t_of_x, v) * 3.6) / 3.6
        dur = round(lap_t[-1], 3)
        s1 = round(lap_t[1], 3)
        s2 = round(lap_t[2] - lap_t[1], 3)
        out.append(LapInput(1 + k % 2, k + 1, t0 + rng.uniform(-jitter, jitter), dur,
                            (s1, s2, round(dur - s1 - s2, 3)), tp, sp, np.zeros(len(tp)), tv, vv))
        truth.append((t0, lambda q, t0=t0: np.interp(q, x, t_of_x)))
    return out, truth


def test_fit_session_recovers_lines_start_times_and_timing(rect_line):
    inputs, truth = _lap_inputs(rect_line)
    timing = fit_session(inputs, rect_line)
    length = rect_line.length
    expect = (37.0 + np.array([0.0, length / 3, 2 * length / 3])) % length
    # Pinning the median start correction to zero leaves one common time offset (the
    # median of six +/-0.1 s jitters), so the lines move together by a metre or two.
    line_err = wrap(timing.s_lines - expect, length)
    assert np.all(np.abs(line_err) < 3.0)
    t_err = np.array([timing.laps[li.key].t_start - t0 for li, (t0, _) in zip(inputs, truth,
                                                                               strict=True)])
    assert np.ptp(t_err) < 0.02 and abs(t_err.mean()) < 0.06
    # In the estimated frame, elapsed time to every lap distance matches the truth to a
    # couple of hundredths once that common line shift is accounted for.
    shift = line_err[0]
    xq = np.linspace(20, length - 20, 40)
    for li, (_, t_of) in zip(inputs, truth, strict=True):
        lf = timing.laps[li.key]
        assert lf.elapsed(np.array([length]))[0] == pytest.approx(li.duration, abs=1e-5)
        true_elapsed = t_of(xq + shift) - t_of(np.array(shift))
        assert np.max(np.abs(lf.elapsed(xq) - true_elapsed)) < 0.025


def test_held_out_sector_error_is_small_on_clean_data(rect_line):
    inputs, _ = _lap_inputs(rect_line, seed=4)
    timing = fit_session(inputs, rect_line)
    err = held_out_sector(inputs, rect_line, timing)
    assert np.max(np.abs(err)) < 0.03
