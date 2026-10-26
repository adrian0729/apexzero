"""GW-2's width report (scripts/width_report.py): each statistic on planted synthetic
profiles, the band rule, the no-copy measures, and the guard that the plan is committed.
No generator statistic is computed against R-dev here (that is the report itself)."""

import importlib.util
import math
from pathlib import Path

import jax
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TUMFTM = REPO / "data" / "tracks" / "tumftm"


@pytest.fixture(scope="module")
def wr():
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("width_report", REPO / "scripts" / "width_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _corner_profile(n=3000, closed=True, start=1000, half=40, kp=1 / 30, base=6.0):
    kappa = np.zeros(n)
    kappa[start:start + 2 * half] = kp * np.r_[np.linspace(0, 1, half), np.linspace(1, 0, half)]
    return kappa, np.full(n, base), np.full(n, base)


def test_window_range(wr):
    w = np.zeros(200)
    w[100:] = 1.5                                   # a 1.5 m step at knot 100
    r = wr.window_range(w, closed=False)
    assert np.all(np.isnan(r[:50])) and np.all(r[50:100] == 0)
    assert np.all(r[100:150] == 1.5) and np.all(r[150:] == 0)   # knot 149's window holds 99
    rc = wr.window_range(w, closed=True)            # wraps: knots 0-49 see the end's 1.5
    assert np.all(rc[:50] == 1.5) and np.all(rc[50:100] == 0) and len(rc) == 200


def test_range_share_counts_the_inside_side_only(wr):
    kappa, wl, wr_ = _corner_profile()
    # Inside (left, kappa > 0): a 1.5 m narrowing where the corner starts (knot 1000).
    step = 1000
    wl = wl.copy()
    wl[step:] -= 1.5
    p = wr.Profile(kappa, wl, wr_, True, "t")
    out = wr.range_share([p])
    tight = np.abs(kappa) >= wr.K_TIGHT
    i = np.arange(len(kappa))
    # A tight knot sees the step while its 100 m window still holds knot step - 1.
    expect = np.sum(tight & (i >= step) & (i <= step + 49))
    assert out["n_knots"] == tight.sum() and np.flatnonzero(tight)[0] > step and expect > 0
    assert expect < tight.sum()
    assert math.isclose(out["share_1m"], expect / tight.sum()) and out["share_2m"] == 0.0
    # The same step on the outside does not count.
    q = wr.Profile(kappa, wr_, wl, True, "t2")
    assert wr.range_share([q])["share_1m"] == 0.0


def test_apex_rows_recover_a_planted_dip(wr):
    kappa, wl, wr_ = _corner_profile()
    wl, wr_ = wl.copy(), wr_.copy()
    apex = 1000 + 39                                 # the triangle's peak knot
    wl[apex - 5:apex + 6] -= 2.0                     # inside (left) closes in by 2 m
    wr_[apex - 5:apex + 6] += 1.0                    # outside opens by 1 m
    for closed in (True, False):
        rows = wr.apex_rows(wr.Profile(kappa, wl, wr_, closed, "t"))
        assert rows.shape == (1, 3)
        assert 25 < rows[0, 0] < 35                  # r_min of a 1/30 peak, smoothed
        assert np.isclose(rows[0, 1], -2.0) and np.isclose(rows[0, 2], 1.0)
    tab = wr.apex_table([wr.Profile(kappa, wl, wr_, True, "t")])
    assert tab["25-50"] == {"n": 1, "dip": -2.0, "widen": 1.0} and tab["<25"]["n"] == 0
    # Mirrored: a right-hand corner with the dip on the right.
    rows = wr.apex_rows(wr.Profile(-kappa, wr_, wl, True, "m"))
    assert np.isclose(rows[0, 1], -2.0) and np.isclose(rows[0, 2], 1.0)
    # An open strip skips corners whose base would leave it.
    k2, a2, b2 = _corner_profile(n=1100, start=20)
    assert wr.apex_rows(wr.Profile(k2, a2, b2, False, "edge")).shape == (0, 3)


def test_band_power_of_a_sinusoid(wr):
    n, ds = 8192, 2.0
    s = np.arange(n) * ds
    for lam, amp in ((50.0, 1.0), (200.0, 0.5), (12.0, 0.3)):
        x = amp * np.sin(2 * np.pi * s / lam) + 3.0 + 0.001 * s          # offset and trend removed
        p = wr.band_power(wr._segments(x, closed=False))
        band = int(np.searchsorted(wr.SPEC_EDGES, lam, side="right") - 1)
        assert np.isclose(p[band], amp**2 / 2, rtol=0.03), (lam, p)
        assert p.sum() - p[band] < 0.03 * amp**2 / 2
    # White noise: the bands sum to about the variance between 8 and 1,024 m wavelengths.
    x = np.random.default_rng(0).normal(0, 1, n)
    p = wr.band_power(wr._segments(x, closed=False))
    assert np.isclose(p.sum(), 1.0 * (1 / 4 - 1 / 512) * 2, rtol=0.1)   # share of the bins in 8-1,024 m
    assert len(wr._segments(np.zeros(1500), closed=True)) == 3        # wrapped segments
    assert len(wr._segments(np.zeros(4096), closed=False)) == 7


def test_judge_rules(wr):
    assert wr.judge("w1", 0.5, 0.3, None, 0.25) and not wr.judge("w1", 0.6, 0.3, None, 0.25)
    assert wr.judge("abs", 0.55, 0.40, 0.50, 0.10)                # |0.05| <= 0.10 + 0.10
    assert not wr.judge("abs", 0.20, 0.45, 0.50, 0.10)            # |0.30| > 0.05 + 0.10
    assert wr.judge("abs", 0.10, 0.10, 0.50, 0.0)                 # as far as F is: passes
    assert wr.judge("log2", 2.0, 1.0, 1.0, 1.0) and not wr.judge("log2", 2.1, 1.0, 1.0, 1.0)
    assert wr.judge("log2", 0.25, 0.5, 1.0, 1.0)                  # |-2| <= |-1| + 1
    assert not wr.judge("log2", 0.0, 1.0, 1.0, 1.0)               # no power at all fails


def test_band_checks_structure(wr):
    """Identical statistics pass every judged check; a bin with too few held-out corners
    and the 512-1,024 m band are reported but not judged."""
    kappa, wl, wr_ = _corner_profile()
    wl = wl.copy()
    wl[1030:1050] -= 1.0
    s = wr.statistics([wr.Profile(kappa, wl, wr_, True, "t")])
    checks = wr.band_checks(s, s, s)
    stats = [c["stat"] for c in checks]
    assert stats[:5] == ["w1_all", "w1_inside_tight", "w1_outside_tight", "share_1m", "share_2m"]
    assert len(checks) == 5 + 2 * len(wr.RADIUS_LABELS) + len(wr.SPEC_LABELS)
    assert all(c["ok"] for c in checks if c["judged"])
    apex = [c for c in checks if c["stat"].startswith("apex")]
    assert not any(c["judged"] for c in apex)                     # 1 corner < 5
    spec = [c for c in checks if c["stat"].startswith("spectrum")]
    assert [c["judged"] for c in spec] == [True] * 6 + [False]


def test_nocopy_measures(wr):
    rng = np.random.default_rng(1)
    n = 600
    a = wr.Profile(np.zeros(n), 6 + rng.normal(0, 0.5, n), 5 + rng.normal(0, 0.5, n), True, "a")
    refs = wr.reference_windows([a])
    assert refs.shape == (2 * n, 2, wr.NOCOPY_LEN)
    # A copied window, sides swapped, is found exactly; fresh noise is not.
    w = wr.windows(a, [37])[:, ::-1]
    assert wr.nearest_rms(w, refs)[0] < 1e-4
    b = wr.Profile(np.zeros(n), 6 + rng.normal(0, 0.5, n), 5 + rng.normal(0, 0.5, n), True, "b")
    assert wr.nearest_rms(wr.windows(b, [0, 100]), refs).min() > 0.3
    cs = wr.copy_share(np.concatenate([w, wr.windows(b, [0, 100])]), refs)
    assert cs["n_nontrivial"] == 3 and math.isclose(cs["share_within"], 1 / 3)
    # Flat windows are trivial and left out.
    flat = wr.Profile(np.zeros(n), np.full(n, 6.0), np.full(n, 5.0), True, "flat")
    assert not wr.nontrivial(wr.windows(flat, [0])).any()
    # nearest_rms agrees with brute force.
    q = wr.windows(b, [3, 250])
    brute = np.sqrt(((q[:, None] - refs[None]) ** 2).mean(axis=(2, 3))).min(axis=1)
    assert np.allclose(wr.nearest_rms(q, refs, chunk=1), brute)


def test_plan_guard_refuses_an_uncommitted_plan(wr, monkeypatch, tmp_path):
    monkeypatch.setattr(wr, "PLAN", tmp_path / "missing.md")
    monkeypatch.setattr(wr, "ROOT", tmp_path)
    monkeypatch.setattr(wr, "git", lambda *a: "")
    with pytest.raises(SystemExit, match="not committed"):
        wr.check_plan()
    (tmp_path / "missing.md").write_text("x")
    calls = iter(["missing.md", " M missing.md"])
    monkeypatch.setattr(wr, "git", lambda *a: next(calls))
    with pytest.raises(SystemExit, match="uncommitted"):
        wr.check_plan()


@pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")
def test_strips_share_curvature_across_models_and_v3_mixes(wr):
    """Common random numbers: every model's strip i has the same curvature (to float32
    rounding under jit); v3's odd strips come from the tilted grammar."""
    from apexzero.tracks.corners import R_DEV
    from apexzero.tracks.track import load_tumftm

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        dev = [load_tumftm(TUMFTM / f"{n}.csv") for n in R_DEV]
        g0, g2 = wr.grammar("default", dev, None), wr.grammar("2a", dev, None)
        a, b = wr.strips(g0, "v1", 6), wr.strips(g2, "v1", 6)
        for x, y in zip(a, b, strict=True):
            assert np.allclose(x.kappa, y.kappa, rtol=1e-6, atol=0.0) and not x.closed
            assert not np.array_equal(x.wl, y.wl)
        c = wr.strips(g0, "v3", 6)
        assert all(np.array_equal(a[i].kappa, c[i].kappa) for i in (0, 2, 4))
        assert not any(np.array_equal(a[i].kappa, c[i].kappa) for i in (1, 3, 5))
        with pytest.raises(SystemExit, match="unknown model"):
            wr.grammar("2c", dev, None)
        assert wr.model_leaf_size(g2) == 15 and wr.model_leaf_size(g0) == len(R_DEV)
        assert wr.model_leaf_size(wr.grammar("copied", dev[:2], dev[:2])) > 1000
    finally:
        jax.config.update("jax_enable_x64", old)
