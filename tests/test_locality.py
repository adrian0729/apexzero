"""scripts/locality_exploratory.py's profiles, s* recovery and G_loc on synthetic arrays (fast;
no outputs are read)."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from apexzero.fields import measures as MS
from apexzero.fields import perturb as PT
from apexzero.fields.perturb import Samples
from apexzero.interp.stimuli import Landmarks
from apexzero.tracks.spline import packed_coeffs, spline_np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def loc():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("locality_exploratory")
    finally:
        sys.path.remove(str(SCRIPTS))


def _basis(n: int, s_star: float, ds: float = 2.0) -> np.ndarray:
    """The arrays' spline basis at s_star: b_j = spline of the unit sample j, evaluated there."""
    out = np.empty(n)
    for j in range(n):
        f = np.zeros(n)
        f[j] = 1.0
        out[j] = spline_np(packed_coeffs(f, False, n), np.array([s_star]), ds, n, False)[0]
    return out


def test_profiles_heights_and_shapes(loc):
    s_star, s_c0, s_c1 = 151.0, 120.0, 190.0
    P, H = loc.profiles(np.array([s_star]), s_star, s_c0, s_c1)
    for m, (fam, lab, p) in enumerate(loc.MEMBERS):
        if fam in ("bump", "ramp") or (fam == "plateau" and p is not None):
            assert P[m, 0] == pytest.approx(1.0) and H[m] == 1.0, lab
        if fam == "offset":
            assert H[m] == pytest.approx(np.exp(-0.5 * (p / 8.0) ** 2)), lab
            assert P[m, 0] == pytest.approx(H[m]), lab
    s = np.arange(0.0, 400.0, 0.5)
    r = loc.ramp_profile(s, s_star, 10.0, s_c1)
    assert np.all(r[s <= s_star - 10.0] == 0.0)
    assert r[s == s_star - 5.0][0] == pytest.approx(0.5)
    assert np.all(r[(s >= s_star) & (s <= s_c1 + 20.0)] == 1.0)
    assert np.all(r[s >= s_c1 + 40.0] == 0.0)
    late = loc.ramp_profile(s, 215.0, 10.0, s_c1)      # s* beyond s_c1 + 20: fall starts at s*
    assert late[s == 215.0][0] == 1.0 and late[s == 225.0][0] == pytest.approx(0.5)
    q = loc.plateau_profile(s, s_star, 20.0)
    assert np.all(q[np.abs(s - s_star) <= 10.0] == 1.0)
    assert np.all(q[np.abs(s - s_star) >= 30.0] == 0.0)
    assert q[s == s_star + 20.0][0] == pytest.approx(0.5)


@pytest.mark.parametrize("s_star", [37.0, 38.3, 61.9])
def test_s_star_from_spline_basis(loc, s_star):
    n = 48
    b = _basis(n, s_star)
    got, dsum = loc.s_star_from_basis(b, PT.sample_s(n))
    assert abs(dsum) < 1e-12
    assert got == pytest.approx(s_star, abs=1e-10)


def test_trapezoid_is_step2_G_and_line_forms(loc):
    n, s_star, sigma = 160, 131.0, -1
    lm = Landmarks(100.0, 100.0, 140.0, 180.0, 180.0)
    rng = np.random.default_rng(0)
    imp = Samples(rng.normal(size=n), rng.normal(size=n), rng.normal(size=n))
    fi = loc.inside_index(sigma)
    s = PT.sample_s(n)
    b = _basis(n, s_star)
    v = loc.key_values(imp[fi], b, s, s_star, lm.s_c0, lm.s_c1)
    assert v["G_loc"][loc.TRAP] == pytest.approx(MS.edge_gain(imp, lm, sigma), abs=1e-12)
    blind = loc.key_values(b, b, s, s_star, lm.s_c0, lm.s_c1)   # c_min moves with the edge only
    assert np.allclose(blind["L"], 0.0, atol=1e-12)
    assert np.allclose(blind["L_per_e"][loc.RATIO], 0.0, atol=1e-12)
    keeper = loc.key_values(np.zeros(n), b, s, s_star, lm.s_c0, lm.s_c1)   # the line follows
    assert np.allclose(keeper["G_loc"], keeper["H"])
    assert np.allclose(keeper["L"], keeper["e"])
    assert np.allclose(keeper["L_per_e"][loc.RATIO], 1.0)
    assert np.all(np.isnan(keeper["L_per_e"][~loc.RATIO]))
