"""Psychophysics analysis (protocol 2.2-2.4, 7, 9): pair values, every decision rule on
synthetic data whose verdict is known (confirm, refute, inconclusive), the intervals, and the
dry run through a synthetic key design that mimics the real one (tiers A and B, mirror
pairs, 4 phases, lawful keys from a closed-form braking model)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import optimize, stats
from scipy.special import expit, log_expit

from apexzero.interp import analysis as A
from apexzero.interp import measures, registry
from apexzero.interp.constants import R_BRAKING, TIER_B_D, TIER_B_R, TIER_B_V0_KPH, kph
from apexzero.interp.io import load_table, select
from apexzero.interp.stimuli import (
    COND_A,
    D_CONDITIONS,
    Cell,
    CornerSpec,
    Geometry,
    Landmarks,
    Stimulus,
)

B = 0.075
REG = A.Regime(12.0, 20.0)


# ------------------------------------------------------------------------- fixtures


class FakeTables:
    """a_dec(v) = A0 + C v^2 (drag and downforce), so d_b^phys has a closed form."""

    A0, C = 12.0, 0.0055

    def d_b_phys(self, v_b, v_e):
        if v_e >= v_b:
            return 0.0
        return math.log((self.A0 + self.C * v_b**2) / (self.A0 + self.C * v_e**2)) / (2 * self.C)


TABLES = FakeTables()


def _v_typ(R: float) -> float:
    return kph(160.0) if R < 18.3 else (kph(200.0) if R < 44.7 else kph(250.0))


def make_keys(r_values=R_BRAKING, tier_b: bool = True) -> dict:
    """A KEY_COLUMNS table shaped like the real design: tier A (R ladder x 90 deg, 180 deg up
    to 55.9 m; A1 and A2) and tier B (R x theta x v0 x D, sharing (310, 600) with A1). Keys
    are lawful: v_e* = 1.08 sqrt(18 R) (0.95x at 180 deg), d_b* = d_b^phys(v_b* = v0, v_e*)."""
    conds: dict[tuple, set] = {}
    for R in r_values:
        for th in (90.0, 180.0):
            if th == 180.0 and R > 55.9:
                continue
            conds.setdefault((R, th, kph(310.0), 600.0), set()).add("A1")
            conds.setdefault((R, th, _v_typ(R), 300.0), set()).add("A2")
    if tier_b:
        for R in TIER_B_R:
            for th in (90.0, 180.0):
                for v0 in TIER_B_V0_KPH:
                    for D in TIER_B_D:
                        conds.setdefault((R, th, kph(v0), D), set()).add("B")
    rows, cell = [], 0
    for pair, ((R, th, v0, D), tiers) in enumerate(sorted(conds.items())):
        approach = "A1" if "A1" in tiers else ("A2" if "A2" in tiers else "B")
        tiers_s = ",".join(t for t in ("A", "B") if (t == "A" and tiers & {"A1", "A2"})
                           or (t == "B" and "B" in tiers))
        v_min = math.sqrt(18.0 * R) * (0.95 if th == 180.0 else 1.0)
        v_e = 1.08 * v_min
        d_b = TABLES.d_b_phys(v0, v_e)
        L_c = math.radians(th) * R + 20.0
        geo = f"syn/R{R:g}/T{th:g}/W12/typical"
        for direction in (1, -1):
            rows.append(dict(
                cell=cell, stim=cell, pair=pair, name=f"{geo}/{'L' if direction > 0 else 'R'}",
                geometry=geo, tiers=tiers_s, approach=approach, v0=v0, D=D, W=400, n_phases=4,
                direction=direction, s_c0=1000.0, s_a0=1010.0, s_apex=1000.0 + L_c / 2,
                s_c1=1000.0 + L_c, R_nom=R, theta_deg=th, width=12.0, ramp_kind="typical",
                no_cap=False, R_seg=R, R_eff=R, corner=-1, condition="", swap_j=-1,
                key_status="verified", J_star=100.0, v_e_star=v_e, v_min_star=v_min,
                d_b_star=d_b, v_b_star=v0, has_brake_star=True, tau_vmin=kph(0.5),
                multi_basin=False, lock_limited=False, slip_limited=False,
                braking=bool(v_min < v0 - kph(0.5)), v_e_star_bo=0.97 * v_e,
                v_min_star_bo=0.97 * v_min, v_e_socp=v_e, v_min_socp=v_min, d_b_socp=d_b,
                v_b_socp=v0, braking_socp=True, v_min_qss=0.9 * v_min))
            cell += 1
    return A._to_table(rows, A.KEY_COLUMNS)


KEYS = make_keys()


@pytest.fixture(scope="module")
def sims():
    out = {}
    for i, kind in enumerate(("lawful", "overspeed_below_12", "late_onset", "time_to_corner")):
        runs = A.simulate_runs(kind, KEYS, A._dry_rng(7, i))
        out[kind] = (runs, A.pair_values(runs, KEYS, {kind: REG}, B))
    return out


def _set(t: dict, col: str, mask, value) -> dict:
    t = {k: np.array(v, copy=True) for k, v in t.items()}
    t[col][mask] = value
    return t


# ------------------------------------------------------------------------- band, slope


def test_band_side():
    assert A.band_side(0.0, 0.0, B) == "in"
    assert A.band_side(0.08, 0.2, B) == "over"
    assert A.band_side(np.inf, np.nan, B) == "over"
    assert A.band_side(0.0, -0.08, B) == "under"
    assert A.band_side(0.09, -0.2, B) == "over"           # overspeed takes precedence
    assert A.band_side(np.nan, 0.0, B) == "undefined"
    assert A.band_side(0.0, np.nan, B) == "undefined"
    assert A.band_side(B, -B, B) == "in"                  # boundaries are inside


def _slope_pairs(S=0.9, Rs=(22.9, 28.6, 35.8, 44.7, 55.9), noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for th, a in ((90.0, 0.1), (180.0, -0.05)):
        for R in Rs:
            x = math.log(math.sqrt(18 * R))
            rows.append(dict(R_nom=R, theta_deg=th, v_e_star=math.exp(x),
                             v_e=math.exp(a + S * x + noise * rng.standard_normal())))
    return A._to_table(rows, ("R_nom", "theta_deg", "v_e_star", "v_e"))


def test_sensitivity_recovers_slope_and_loo():
    fit = A.sensitivity(_slope_pairs(0.9))
    assert fit.S == pytest.approx(0.9, abs=1e-12)
    assert fit.intercepts[90.0] == pytest.approx(0.1) and fit.intercepts[180.0] == pytest.approx(-0.05)
    assert set(fit.loo) == {22.9, 28.6, 35.8, 44.7, 55.9}
    assert all(v == pytest.approx(0.9) for v in fit.loo.values())
    # log range of the within-theta-centred log v_e*: 0.5 log(55.9 / 22.9)
    assert fit.log_range == pytest.approx(0.5 * math.log(55.9 / 22.9))
    assert fit.estimable and fit.n == 10


def test_sensitivity_estimability_and_overspeed_excluded():
    narrow = _slope_pairs(1.0, Rs=(22.9, 28.6))            # range 0.5 log(1.25) = 0.11
    assert not A.sensitivity(narrow).estimable
    p = _slope_pairs(1.0)
    p["v_e"][0] = np.inf                                    # overspeed crash pair
    fit = A.sensitivity(p)
    assert fit.n == 9 and fit.S == pytest.approx(1.0)


def test_extrapolation_residual():
    p = _slope_pairs(1.0)
    fit = A.sensitivity(p)
    r = A.extrapolation_residual(p, fit)
    np.testing.assert_allclose(r, 0.0, atol=1e-12)
    q = dict(p)
    q["v_e"] = p["v_e"] * 1.1
    q["v_e"][0] = np.inf
    r = A.extrapolation_residual(q, fit)
    assert np.isinf(r[0]) and np.allclose(r[1:], math.log(1.1))


# ------------------------------------------------------------------------- pair values


def _pair_keys(approach="A2", D=300.0, regime_R=30.0):
    rows = []
    for c, d in ((0, 1), (1, -1)):
        rows.append(dict(cell=c, stim=c, pair=7, name=f"syn/R{regime_R}/T90/W12/typical/{'LR'[c]}",
                         geometry=f"syn/R{regime_R}/T90/W12/typical", tiers="A",
                         approach=approach, v0=60.0, D=D, W=400, n_phases=4, direction=d,
                         s_c0=1000.0, s_a0=1010.0, s_apex=1030.0, s_c1=1060.0, R_nom=regime_R,
                         theta_deg=90.0, width=12.0, ramp_kind="typical", no_cap=False,
                         R_seg=regime_R, R_eff=regime_R, corner=-1, condition="", swap_j=-1,
                         key_status="verified", J_star=1.0, v_e_star=30.0, v_min_star=25.0,
                         d_b_star=100.0, v_b_star=60.0, has_brake_star=True, tau_vmin=0.1,
                         multi_basin=False, lock_limited=False, slip_limited=False,
                         braking=True, v_e_star_bo=28.0, v_min_star_bo=24.0, v_e_socp=30.0,
                         v_min_socp=25.0, d_b_socp=100.0, v_b_socp=60.0, braking_socp=True,
                         v_min_qss=20.0))
    return A._to_table(rows, A.KEY_COLUMNS)


def _pair_runs(v_e, crashed=None, ctype=None, s_crash=None, has_brake=None, forced=None,
               d_b=None, e_ve_ra=None, v_ra=None, nat=(), mat=(), mat_type=()):
    n = len(v_e)
    crashed = crashed or [False] * n
    rows = []
    for i in range(n):
        ve = v_e[i]
        rows.append(dict(agent="G1", timesteps=5, cell=i // 4, stim=i // 4, pair=7, phase=i % 4,
                         sample=-1, mode="det", steer_sign=1 if i < 4 else -1,
                         crashed=crashed[i], completed=not crashed[i],
                         crash_type=(ctype or [""] * n)[i],
                         s_crash=(s_crash or [np.nan] * n)[i], v_e=ve, e_ve=ve / 30.0 - 1,
                         e_ve_bo=ve / 28.0 - 1,
                         e_ve_ra=(e_ve_ra or [ve / 30.0 - 1] * n)[i],
                         ra_key=e_ve_ra is not None,     # e_ve_ra given: a re-anchored key
                         v_e_star_ra=(v_ra or [30.0] * n)[i], v_min=ve - 5.0,
                         e_vmin=(ve - 5.0) / 25.0 - 1,
                         has_brake=(has_brake or [True] * n)[i],
                         forced_onset=(forced or [False] * n)[i],
                         d_b=(d_b or [100.0 + i for i in range(n)])[i], dd_b=0.0, v_b=60.0))
    for mode, flags, types in (("nat", nat, ()), ("mat", mat, mat_type)):
        for j, f in enumerate(flags):
            rows.append(dict(agent="G1", timesteps=5, cell=j % 2, stim=j % 2, pair=7,
                             phase=j % 4, sample=j, mode=mode, steer_sign=1, crashed=f,
                             completed=not f, crash_type=(types[j] if f and types else ""),
                             s_crash=1030.0 if f else np.nan))
    cols = sorted({k for r in rows for k in r})
    return A._to_table(rows, cols)


def test_pair_values_overspeed_crash_is_plus_inf_in_the_median():
    v = [30, 31, 32, 33, 30, 31, 32, 33.0]
    runs = _pair_runs(v, crashed=[True] + [False] * 7, ctype=["overspeed"] + [""] * 7,
                      s_crash=[1030.0] + [np.nan] * 7)
    p = A.pair_values(runs, _pair_keys(), {"G1": REG}, B)
    # values: inf, 31, 32, 33, 30, 31, 32, 33 -> median (31 + 32) / 2... sorted
    # [30, 31, 31, 32, 32, 33, 33, inf] -> (32 + 32) / 2
    assert p["v_e"][0] == pytest.approx(32.0)
    assert p["det_overspeed"][0] == 1 and p["det_crashes"][0] == 1 and p["defined"][0]
    assert p["v_min"][0] == pytest.approx(27.0)            # inf counts for v_min too
    assert p["v_e_obs"][0] == pytest.approx(31.5)          # observed, no +inf (median of 8)
    assert p["regime"][0] == "bulk" and p["side"][0] == "in"


def test_pair_values_missing_rules():
    v = [30.0] * 8
    pre = [True] * 4 + [False] * 4
    runs = _pair_runs(v, crashed=pre, ctype=["other"] * 4 + [""] * 4,
                      s_crash=[900.0] * 4 + [np.nan] * 4)
    p = A.pair_values(runs, _pair_keys(), {"G1": REG}, B)
    assert p["n_missing"][0] == 4 and not p["defined"][0] and np.isnan(p["v_e"][0])
    assert p["det_crashes"][0] == 4 and p["side"][0] == "undefined"
    # three crashes before s_c0: still defined
    runs = _pair_runs(v, crashed=[True] * 3 + [False] * 5, ctype=["other"] * 3 + [""] * 5,
                      s_crash=[900.0] * 3 + [np.nan] * 5)
    p = A.pair_values(runs, _pair_keys(), {"G1": REG}, B)
    assert p["defined"][0] and p["n_missing"][0] == 3 and p["v_e"][0] == 30.0
    # non-overspeed crashes after s_c0 keep v_e but lose v_min
    runs = _pair_runs([30, 30, 30, 30, 34, 34, 34, 34.0], crashed=[False] * 4 + [True] * 4,
                      ctype=[""] * 4 + ["line_steer"] * 4, s_crash=[np.nan] * 4 + [1040.0] * 4)
    p = A.pair_values(runs, _pair_keys(), {"G1": REG}, B)
    assert p["v_e"][0] == pytest.approx(32.0)
    assert np.isnan(p["v_min"][0])                         # 4 of 8 v_min missing


def test_pair_values_onsets_need_five_unforced_brakes():
    v = [30.0] * 8
    hb = [True] * 4 + [False] * 4
    p = A.pair_values(_pair_runs(v, has_brake=hb), _pair_keys(), {"G1": REG}, B)
    assert np.isnan(p["d_b"][0]) and p["n_brake"][0] == 4
    hb = [True] * 6 + [False] * 2
    forced = [True] + [False] * 7
    p = A.pair_values(_pair_runs(v, has_brake=hb, forced=forced), _pair_keys(), {"G1": REG}, B)
    assert p["d_b"][0] == pytest.approx(np.median([101, 102, 103, 104, 105]))
    assert p["forced_frac"][0] == pytest.approx(1 / 6) and p["n_brake"][0] == 6
    forced = [True, True] + [False] * 6                     # leaves 4 unforced
    p = A.pair_values(_pair_runs(v, has_brake=hb, forced=forced), _pair_keys(), {"G1": REG}, B)
    assert np.isnan(p["d_b"][0])


def test_pair_values_stochastic_counts_and_crash_types():
    runs = _pair_runs([30.0] * 8, crashed=[True] + [False] * 7, ctype=["overbrake"] + [""] * 7,
                      s_crash=[1040.0] + [np.nan] * 7, nat=[True, False, True, False],
                      mat=[True, True, False], mat_type=["overspeed", "overspeed", ""])
    p = A.pair_values(runs, _pair_keys(), {"G1": REG}, B)
    assert (p["nat_k"][0], p["nat_n"][0], p["mat_k"][0], p["mat_n"][0]) == (2, 4, 2, 3)
    assert p["typed_crashes"][0] == 3 and p["typed_overspeed"][0] == 2
    assert p["typed_overbrake"][0] == 1 and p["overspeed_frac"][0] == pytest.approx(2 / 3)


def test_pair_values_reanchored_band_and_exclusions():
    # A1: the band uses e_ve^RA and the backed-off error scaled by v_e*/v_e*bo
    v = [33.0] * 8
    runs = _pair_runs(v, e_ve_ra=[33 / 34 - 1] * 8, v_ra=[34.0] * 8)
    p = A.pair_values(runs, _pair_keys("A1", 600.0), {"G1": REG}, B)
    assert p["uses_ra"][0] and p["v_e_star_band"][0] == 34.0
    assert p["e_ve_bo_ra"][0] == pytest.approx(33 / 34 * 30 / 28 - 1)
    assert p["side"][0] == "in"                           # e_ve = +10% vs nominal, but RA -3%
    p = A.pair_values(runs, _pair_keys("A2"), {"G1": REG}, B)
    assert not p["uses_ra"][0] and p["side"][0] == "over"
    keys = _pair_keys()
    keys["key_status"][:] = "unsolved"
    assert A.pair_values(runs, keys, {"G1": REG}, B)["excluded"][0] == "unsolved"
    keys = _pair_keys()
    keys["multi_basin"][:] = True
    assert A.pair_values(runs, keys, {"G1": REG}, B)["excluded"][0] == "multi_basin"


def test_a1_runs_without_a_reanchored_key_fall_back_to_their_own_e_ve_bo():
    """Review of the V7 remedy: an A1 run with no re-anchored key falls back to e_ve for
    e_ve_ra and to its own e_ve_bo for e_ve_bo_ra (AD5, AD23). Scaling the fallback by
    v_e*/v_e*bo moved the backed-off edge by the phase key's ratio once e_ve was judged
    against a phase-matched key (v_e*_k 33 here against the u = 0 key's 30)."""
    v = [33.0] * 8
    runs = _pair_runs(v)                       # no re-anchored key: ra_key False
    runs["e_ve"][:] = runs["e_ve_ra"][:] = 0.0  # against the phase key, v_e*_k = 33
    keys = _pair_keys("A1", 600.0)
    p = A.pair_values(runs, keys, {"G1": REG}, B)
    assert p["uses_ra"][0] and p["ra_fallback"][0] == 8
    assert p["e_ve_bo_ra"][0] == pytest.approx(33 / 28 - 1) == p["e_ve_bo"][0]
    # with a re-anchored key the backed-off error stays scaled to it
    runs["ra_key"][:] = True
    p = A.pair_values(runs, keys, {"G1": REG}, B)
    assert p["ra_fallback"][0] == 0 and p["e_ve_bo_ra"][0] == pytest.approx(30 / 28 - 1)
    # regimes: none without a regime table
    assert A.pair_values(runs, _pair_keys(), {}, B)["regime"][0] == ""
    for R, reg in ((6.0, "clip"), (9.0, "below"), (14.0, "tail"), (30.0, "bulk")):
        p = A.pair_values(runs, _pair_keys(regime_R=R), {"G1": A.Regime(12, 20, 7.6)}, B)
        assert p["regime"][0] == reg


def test_pair_values_on_the_design(sims):
    _, pairs = sims["lawful"]
    assert A._n(pairs) == np.unique(KEYS["pair"]).size
    assert set(np.unique(pairs["n_runs"])) == {8}
    assert np.all(pairs["side"][pairs["braking"]] == "in")
    np.testing.assert_allclose(pairs["e_ve"], -0.01, atol=0.012)
    bulk = A._mask(pairs, approach="A2", regimes=("bulk",))
    np.testing.assert_allclose(pairs["r_extrap"][bulk], 0.0, atol=0.01)


# ------------------------------------------------------------------------- inference


def test_clopper_pearson():
    lo, hi = A.clopper_pearson(0, 100, 0.99)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.005 ** (1 / 100))
    lo, hi = A.clopper_pearson(5, 20, 0.95)
    assert lo == pytest.approx(0.0866, abs=1e-4) and hi == pytest.approx(0.4910, abs=1e-4)


def test_beta_binomial_likelihood_is_exact():
    k, n = np.array([0, 3, 10, 64, 7]), np.array([64, 64, 64, 64, 32])
    tabs = A._bb_tabs(k, n)
    for mu, rho in ((0.2, 0.0), (0.2, 0.3), (0.05, 0.01), (0.6, 0.9)):
        if rho == 0:
            ref = stats.binom.logpmf(k, n, mu).sum()
        else:
            a, b = mu * (1 - rho) / rho, (1 - mu) * (1 - rho) / rho
            ref = stats.betabinom.logpmf(k, n, a, b).sum()
        ref -= sum(math.log(math.comb(int(nn), int(kk))) for kk, nn in zip(k, n, strict=True))
        assert A._bb_ll(mu, rho, tabs)[0] == pytest.approx(ref, rel=1e-10, abs=1e-9)


def test_beta_binomial_zero_and_equidispersed_use_clopper_pearson():
    r = A.beta_binomial(np.zeros(9, int), np.full(9, 64), 0.99)
    assert r.method == "clopper-pearson" and r.rho == 0.0 and r.p_hat == 0.0
    assert r.hi == pytest.approx(1 - 0.005 ** (1 / 576)) and r.hi <= 0.05
    r = A.beta_binomial(np.full(6, 3), np.full(6, 64), 0.95)
    assert r.method == "clopper-pearson"
    assert (r.lo, r.hi) == pytest.approx(A.clopper_pearson(18, 384, 0.95))


def test_beta_binomial_overdispersed_profile_interval():
    k = np.array([0, 0, 0, 20, 0, 0, 1, 25])
    n = np.full(8, 64)
    r = A.beta_binomial(k, n, 0.95)
    assert r.method == "beta-binomial profile" and r.rho > 0.05
    assert r.lo < r.p_hat < r.hi
    cp = A.clopper_pearson(int(k.sum()), int(n.sum()), 0.95)
    assert r.hi - r.lo > cp[1] - cp[0]                    # clustering widens the interval
    tabs = A._bb_tabs(k, n)
    ll_hat = A._bb_profile(r.p_hat, tabs)[0]
    crit = stats.chi2.ppf(0.95, 1)
    for bound in (r.lo, r.hi):
        assert 2 * (ll_hat - A._bb_profile(bound, tabs)[0]) == pytest.approx(crit, abs=1e-4)
    # brute-force check of the ML on a grid
    grid = [(m, rho) for m in np.linspace(0.01, 0.4, 80) for rho in np.linspace(0, 0.9, 91)]
    best = max(grid, key=lambda mr: A._bb_ll(mr[0], mr[1], tabs)[0])
    assert A._bb_ll(*best, tabs)[0] <= ll_hat + 1e-9
    assert r.p_hat == pytest.approx(best[0], abs=0.01)
    assert A.beta_binomial(k, n, 0.99).hi > r.hi


def _pll(beta, x, y, n):
    X = np.column_stack([np.ones_like(x), x])
    eta = X @ beta
    p = expit(eta)
    info = X.T @ ((n * p * (1 - p))[:, None] * X)
    return (y * log_expit(eta) + (n - y) * log_expit(-eta)).sum() + 0.5 * np.linalg.slogdet(info)[1]


def test_firth_logistic_maximises_the_penalised_likelihood():
    x, y = np.array([-2.0, -1, 1, 2]), np.array([0.0, 0, 1, 1])     # complete separation
    coef, se = A.firth_logistic(x, y)
    assert np.all(np.isfinite(coef)) and np.all(np.isfinite(se)) and coef[1] > 0
    ref = optimize.minimize(lambda b: -_pll(b, x, y, np.ones(4)), np.zeros(2),
                            method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12})
    np.testing.assert_allclose(coef, ref.x, atol=1e-4)
    # grouped counts give the same fit as the expanded Bernoulli data
    xs, ys, ns = np.log([9.4, 11.7, 14.6, 18.3]), np.array([7.0, 5, 2, 0]), np.full(4, 8.0)
    xe = np.repeat(xs, 8)
    ye = np.concatenate([[1.0] * int(k) + [0.0] * (8 - int(k)) for k in ys])
    np.testing.assert_allclose(A.firth_logistic(xs, ys, ns)[0], A.firth_logistic(xe, ye)[0],
                               atol=1e-8)


def test_r50_bracket_and_firth():
    R = np.array([9.4, 11.7, 14.6, 18.3, 22.9])
    out = A.r50(R, np.array([1.0, 0.75, 0.25, 0.0, 0.0]), np.full(5, 8))
    assert out["bracket"] == (11.7, 14.6) and out["position"] == "inside" and out["monotone"]
    assert 11.7 < out["firth_R50"] < 14.6
    assert A.r50(R, np.zeros(5))["position"] == "below_grid"
    assert np.isnan(A.r50(R, np.zeros(5), np.full(5, 8))["firth_R50"])
    assert A.r50(R, np.zeros(5))["bracket"][1] == 9.4
    assert A.r50(R, np.ones(5))["bracket"] == (22.9, np.nan) or \
        A.r50(R, np.ones(5))["position"] == "above_grid"
    nm = A.r50(R, np.array([1.0, 0.0, 0.75, 0.0, 0.0]))
    assert nm["bracket"] == (14.6, 18.3) and not nm["monotone"]


def test_kendall_and_spearman():
    rng = np.random.default_rng(1)
    x, y = rng.integers(0, 5, 30).astype(float), rng.integers(0, 4, 30).astype(float)
    assert A.kendall_tau(x, y) == pytest.approx(stats.kendalltau(x, y).statistic)
    assert A.kendall_tau([1, 2, np.inf, np.inf], [4, 3, 2, 1]) == pytest.approx(
        stats.kendalltau([1, 2, 3, 3], [4, 3, 2, 1]).statistic)
    assert A.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic)
    rho, p = A.permutation_spearman(np.arange(8.0), -np.arange(8.0), 5000,
                                    np.random.default_rng(0))
    assert rho == pytest.approx(-1.0) and p < 0.002
    rho, p = A.permutation_spearman(np.arange(8.0), np.arange(8.0), 2000, np.random.default_rng(0))
    assert rho == pytest.approx(1.0) and p > 0.99


def test_bootstrap_band():
    rng = np.random.default_rng(3)
    vals = np.concatenate([rng.normal(0, 1, 12), rng.normal(5, 1, 8)])
    groups = np.array(["bulk"] * 12 + ["tail"] * 8)
    lo, hi = A.bootstrap_band(vals, groups, 4000, np.random.default_rng(0))
    assert lo.shape == (2,) and np.all(lo < hi)
    assert lo[0] < np.median(vals[:12]) < hi[0] and lo[1] < np.median(vals[12:]) < hi[1]
    lo2, hi2 = A.bootstrap_band(vals, groups, 4000, np.random.default_rng(0))
    np.testing.assert_array_equal(lo, lo2)                 # reproducible with the same stream


# ------------------------------------------------------------------------- P1


def test_p1a_verdicts(sims):
    _, lawful = sims["lawful"]
    d = A.p1a(lawful, B)
    assert d.verdict == "confirm" and d.family == "secondary"
    assert d.details["slope"]["S"] == pytest.approx(1.0, abs=0.02)
    bulk = A._mask(lawful, approach="A2", regimes=("bulk",))
    idx = np.flatnonzero(bulk)
    # >= 50% overspeed: refute
    assert A.p1a(_set(lawful, "e_ve", idx[: len(idx) // 2 + 1], 0.1), B).verdict == "refute"
    # >= 50% under with a lawful slope: cautious but lawful
    under = _set(lawful, "e_ve_bo", idx[: len(idx) // 2 + 1], -0.1)
    assert A.p1a(under, B).verdict == "cautious but lawful"
    # ... but under plus two crash pairs is a plain refute
    under2 = _set(under, "det_crashes", idx[:2], 1)
    assert A.p1a(under2, B).verdict == "refute"
    # two deterministic crash pairs: refute; one: still confirm
    assert A.p1a(_set(lawful, "det_crashes", idx[:2], 1), B).verdict == "refute"
    assert A.p1a(_set(lawful, "det_crashes", idx[:1], 1), B).verdict == "confirm"
    # 25% overspeed: criterion 1 fails, nothing refutes: inconclusive
    q = len(idx) // 4
    assert A.p1a(_set(lawful, "e_ve", idx[:q], 0.1), B).verdict == "inconclusive"
    # slope 0.5 (band still fine): refuted on its slope
    slow = _set(lawful, "v_e", bulk, lawful["v_e_star"][bulk] ** 0.5 * 6.0)
    d = A.p1a(slow, B)
    assert d.verdict == "refute" and d.details["refute_slope"]
    # native-noise crashes push the 99% upper bound above 5%
    nat = _set(lawful, "nat_k", idx, 3)
    d = A.p1a(nat, B)
    assert d.verdict == "inconclusive" and not d.details["tracking"]["c4"]
    # no bulk pairs: not testable
    none = _set(lawful, "regime", bulk, "tail")
    assert A.p1a(none, B).verdict == "not testable"


def test_p1_tail_verdicts(sims):
    assert A.p1_tail(sims["late_onset"][1], B).verdict == "confirm"
    assert A.p1_tail(sims["lawful"][1], B).verdict == "refute"
    _, lawful = sims["lawful"]
    m1 = np.flatnonzero(A._mask(lawful, approach="A1", regimes=("tail",)))
    half = _set(lawful, "e_ve_ra", m1[: len(m1) // 2], 0.2)  # 50% over: neither part
    d = A.p1_tail(half, B)
    assert d.verdict == "inconclusive" and d.details["a2_part"] == "holds"


def test_p1b_verdicts(sims):
    assert A.p1b(sims["overspeed_below_12"][1], B).verdict == "confirm"
    d = A.p1b(sims["lawful"][1], B)
    assert d.verdict == "refute" and d.details["n_pairs"] >= 4
    runs, _ = sims["lawful"]
    p = A.pair_values(runs, KEYS, {"lawful": A.Regime(5.0, 20.0)}, B)
    assert A.p1b(p, B).verdict == "not testable"
    # κ-clipped below-support pairs leave the set
    p = A.pair_values(runs, KEYS, {"lawful": A.Regime(12.0, 20.0, 11.0)}, B)
    clip = p["regime"] == "clip"
    p = _set(p, "kappa_clipped", clip, True)
    assert A.p1b(p, B).details["n_pairs"] == int(((p["regime"] == "below")
                                                  & A._mask(p, approach="A1")).sum()
                                                 + ((p["regime"] == "below")
                                                    & A._mask(p, approach="A2")).sum())
    # departure in half the below-support pairs only (no crashes): inconclusive
    _, ob = sims["lawful"]
    below = np.flatnonzero(A._below_mask(ob, "A1") | A._below_mask(ob, "A2"))[::2]
    half = _set(ob, "v_e", below, ob["v_e_star_band"][below] * 0.99 * 1.2)
    d = A.p1b(half, B)
    assert d.details["frac_departure"] == pytest.approx(0.5, abs=0.05)
    assert d.verdict == "inconclusive"
    # slope-refuted P1a: r becomes log(1 + e_ve)
    d = A.p1b(sims["overspeed_below_12"][1], B, slope_refuted=True)
    assert d.details["slope_refuted"] and d.verdict == "confirm"


def test_R12_p1b_reads_e_ve_ra_throughout_a1(sims):
    """Ruling R12: P0's re-anchored keys cover the non-clipped below-support A1 cells, so
    P1b's A1 set is on e_ve^RA throughout; a pair whose runs fell back to the nominal key is
    counted (0 when every A1 run has its re-anchored key; A2 pairs never use one)."""
    _, lawful = sims["lawful"]
    assert int(lawful["ra_fallback"].max()) == 0          # simulated runs carry their RA key
    assert A.p1b(lawful, B).details["ra_fallback_pairs"] == 0
    a1 = np.flatnonzero(A._below_mask(lawful, "A1"))
    assert A.p1b(_set(lawful, "ra_fallback", a1[:2], 8), B).details["ra_fallback_pairs"] == 2
    a2 = np.flatnonzero(A._below_mask(lawful, "A2"))
    assert A.p1b(_set(lawful, "ra_fallback", a2[:2], 8), B).details["ra_fallback_pairs"] == 0
    e, _ = A.band_errors(lawful)
    m = A._below_mask(lawful, "A1")
    np.testing.assert_array_equal(e[m], lawful["e_ve_ra"][m])


def test_r_dep_and_p1c(sims):
    _, lawful = sims["lawful"]
    assert A.r_dep(lawful, "A2", B) == min(R_BRAKING)
    _, ob = sims["overspeed_below_12"]
    assert A.r_dep(ob, "A2", B) == 13.1
    worst = _set(lawful, "det_crashes", (lawful["R_nom"] == 87.3), 1)
    assert A.r_dep(worst, "A2", B) == np.inf
    ds = A.p1c({"G1": {"A1": 18.3, "A2": 11.7}, "G2": {"A1": 14.6, "A2": 14.6},
                "G3": {"A1": 14.6, "A2": 16.4}})
    main = ds[0]
    assert main.prediction == "P1c" and main.verdict == "confirm"
    # realised values (18.3, 14.6, 14.6): 2 of the 6 labelings satisfy the ordering
    assert main.details["p_null"] == pytest.approx(2 / 6)
    assert [d.verdict for d in ds[1:]] == ["confirm", "confirm", "refute"]
    assert A.p1c({a: {"A1": 14.6, "A2": 14.6} for a in A.GENERALISTS})[0].verdict == "inconclusive"
    assert A.p1c({"G1": {"A1": 11.7, "A2": 9.4}, "G2": {"A1": 14.6, "A2": 9.4},
                  "G3": {"A1": 9.4, "A2": 9.4}})[0].verdict == "refute"
    assert A.p1c({"G1": {"A1": 11.7, "A2": 9.4}})[0].verdict == "not testable"


# ------------------------------------------------------------------------- P2


def test_p2_verdicts(sims):
    runs, late = sims["late_onset"]
    d = A.p2(late, runs, B)
    assert d.verdict == "confirm" and d.details["median_overspeed_share"] == 1.0
    assert A.p2(sims["lawful"][1], None, B).verdict == "not applicable"
    O = A._o_set(late, B)
    # (a′): the agent finds the corner in time
    assert A.p2(_set(late, "dd_b", O, 5.0), None, B).verdict == "refute"
    # (b′): crashes typed over-brake
    cells = A._p2_scope(late, A.EXCL_ALL) & (late["typed_crashes"] > 0)
    ob = _set(late, "overspeed_frac", cells, 0.0)
    ob["typed_overbrake"] = np.where(cells, ob["typed_crashes"], 0)
    ob["typed_overspeed"] = np.zeros_like(ob["typed_overspeed"])
    d = A.p2(ob, None, B)
    assert d.verdict == "refute" and d.details["b_refute"]
    # mixed: onsets late, response onsets not, crash shares 0.6: inconclusive
    mix = _set(late, "dd_resp", O, 3.0)
    mix = _set(mix, "overspeed_frac", cells, 0.6)
    assert A.p2(mix, None, B).verdict == "inconclusive"
    # no-brake entries count as onset failures
    nb = _set(late, "dd_b", O, np.nan)
    nb = _set(nb, "n_brake", O, 0)
    d = A.p2(nb, None, B)
    assert d.details["no_brake_entries"] == int(O.sum()) and d.verdict == "confirm"


def test_p2b_decomposition_is_exact_and_onset_share():
    rng = np.random.default_rng(5)
    v_b, v_bs = rng.uniform(60, 90, 50), rng.uniform(60, 90, 50)
    v_e, v_es = rng.uniform(15, 40, 50), rng.uniform(15, 40, 50)
    d_b, d_bs = rng.uniform(50, 200, 50), rng.uniform(50, 200, 50)
    ap, on, de = A.p2b_decompose(v_b, v_e, d_b, v_bs, v_es, d_bs)
    np.testing.assert_allclose(ap + on + de, v_e**2 - v_es**2, rtol=1e-10, atol=1e-8)
    assert A.onset_share(10.0, 30.0, -5.0) == pytest.approx(0.75)
    assert np.isnan(A.onset_share(-1.0, -1.0, -1.0))


def test_p2b_verdicts(sims):
    d = A.p2b(sims["late_onset"][1], B)
    assert d.verdict == "confirm" and d.details["median_onset_share"] > 0.9
    _, late = sims["late_onset"]
    O = A._o_set(late, B)
    # same onset as the key, overspeed from a faster approach: approach term dominates
    fast = _set(late, "d_b", O, late["d_b_star"][O])
    fast = _set(fast, "v_b", O, late["v_b_star"][O] * 1.2)
    assert A.p2b(fast, B).verdict == "refute"
    assert A.p2b(sims["lawful"][1], B).verdict == "not applicable"


def test_p2d_descriptive(sims):
    d = A.p2d(sims["lawful"][1], B)
    assert d.verdict == "descriptive" and d.family == "exploratory"
    assert d.details["frac_nearer_qss"] == 0.0             # lawful v_min sits on the key


# ------------------------------------------------------------------------- P3


def test_p3_lawful_and_time_to_corner(sims):
    runs, _ = sims["lawful"]
    res = A.p3_fit(runs, KEYS, TABLES)
    assert res.b == pytest.approx(1.0, abs=0.05) and res.winner == "M_phys"
    assert res.n_geometries > 10 and res.n_predicted > 0
    assert A.p3(res, A.brake_share(runs, KEYS)).verdict == "confirm"
    runs, _ = sims["time_to_corner"]
    res = A.p3_fit(runs, KEYS, TABLES)
    d = A.p3(res, A.brake_share(runs, KEYS))
    assert d.verdict == "refute" and d.details["winner"] == "M_idx"
    assert res.rmse["M_idx"] < 0.9 * res.rmse["M_phys"]


def test_p3_rule_table():
    ok = A.P3Result(1.02, {"g1": 0.95, "g2": 1.1}, {"M_phys": 0.1, "M_idx": 0.2, "M_fd": 0.3},
                    "M_phys", 100, agent="G3")
    assert A.p3(ok, 0.9).verdict == "confirm"
    assert A.p3(ok, 0.3).verdict == "not applicable"
    loo_bad = A.P3Result(1.02, {"g1": 0.8}, ok.rmse, "M_phys", 100, agent="G3")
    assert A.p3(loo_bad, 0.9).verdict == "inconclusive"
    fd = A.P3Result(0.3, {}, {"M_phys": 0.3, "M_idx": 0.29, "M_fd": 0.1}, "M_fd", 100)
    d = A.p3(fd, 0.9)
    assert d.verdict == "refute" and d.details["winner"] == "M_fd"
    near = A.P3Result(0.5, {}, {"M_phys": 0.3, "M_idx": 0.28, "M_fd": 0.4}, "M_idx", 100)
    assert A.p3(near, 0.9).verdict == "inconclusive"       # beats by < 10%
    in_band = A.P3Result(0.9, {}, {"M_phys": 0.3, "M_idx": 0.1, "M_fd": 0.4}, "M_idx", 100)
    assert A.p3(in_band, 0.9).verdict == "inconclusive"    # b inside [0.85, 1.15]


def test_p3_separation_and_beta():
    sep = A.p3_separation(KEYS, TABLES)
    assert sep["passed"] and not sep["demote_p3"]
    assert abs(sep["b_idx"] - 1) >= 0.25 and sep["b_fd"] == pytest.approx(0.0)
    runs = A.simulate_runs("lawful", KEYS, A._dry_rng(1))
    betas = A.p3_beta(runs, KEYS)
    assert [d.prediction for d in betas] == ["P3-beta-lt120", "P3-beta-ge120"]
    assert all(d.verdict == "confirm" for d in betas)


# ------------------------------------------------------------------------- P4


def make_design(n_corners: int = 10, n_braking: int = 8) -> dict:
    rows = []
    for i in range(n_corners):
        v_i = 20.0 + 4.0 * i
        for c in D_CONDITIONS:
            lam = float(c.split("lambda")[1]) if "lambda" in c else 1.0
            rows.append(dict(corner=i, condition=c, braking=i < n_braking,
                             v_min_star=v_i * math.sqrt(lam),
                             swap_j=(i + 3) % n_braking if c == "F5/swapA" else -1))
    return A._to_table(rows, ("corner", "condition", "braking", "v_min_star", "swap_j"))


DESIGN = make_design()


def _p4(kind, **kw):
    return A.p4_classify(A.simulate_conds(kind, DESIGN, np.random.default_rng(0), agent="S9"),
                         A.simulate_conds("reader", DESIGN, np.random.default_rng(1), "G1"), **kw)


def test_p4_reference_classes():
    r = _p4("context_keyed")
    assert r.cls == 2 and r.key_factors == ("before",) and not r.details["vacuous"]
    assert r.MI == pytest.approx(1.0)                      # iso fails for it, not for G1
    r = _p4("reader")
    assert r.cls == 5 and r.S_spec == pytest.approx(1.0, abs=0.02)
    assert abs(r.phi) < 0.05 and abs(r.m) < 0.05
    assert _p4("corner_keyed").cls == 3


def test_p4_memorisation_index():
    spec = A.simulate_conds("context_keyed", DESIGN, np.random.default_rng(0), agent="S9")
    gen = A.simulate_conds("reader", DESIGN, np.random.default_rng(1), agent="G1")
    # specialist: C(a) = 1, C(iso) = 0 (straight before); generalist: 1 and 1
    assert A.memorisation_index(spec, gen) == pytest.approx(1.0)
    assert np.isnan(A.memorisation_index(spec, None))


def _with(cond, mask, **cols):
    out = {k: np.array(v, copy=True) for k, v in cond.items()}
    for k, v in cols.items():
        out[k][mask] = v
    return out


def test_R10_vacuous_class_2_is_flagged_in_the_p4_decision():
    """Ruling R10 (AD16): P4 reads only the single-factor F1 conditions, and a class 2 met
    vacuously (no F6 perturbation as far OOD as the failing condition) is flagged in the
    decision's details, which the decision table reports."""
    base = A.simulate_conds("reader", DESIGN, np.random.default_rng(0), agent="S9")
    gen = A.simulate_conds("reader", DESIGN, np.random.default_rng(1), agent="G1")
    cn = base["condition"]
    wk = _with(base, np.isin(cn, [c for c in A.F1_CONDS if c.endswith("/const")]),
               complete=False, d_ood=1.5)
    vac = _with(wk, np.isin(cn, A.F6_CONDS), d_ood=0.5)
    d = A.p4(A.p4_classify(vac, gen))
    assert d.verdict == "class 2" and d.details["vacuous"] is True
    assert A.p4(A.p4_classify(wk, gen)).details["vacuous"] is False


def test_p4_other_classes():
    base = A.simulate_conds("reader", DESIGN, np.random.default_rng(0), agent="S9")
    gen = A.simulate_conds("reader", DESIGN, np.random.default_rng(1), agent="G1")
    cn = base["condition"]
    # 0: not competent in (a)
    assert A.p4_classify(_with(base, cn == COND_A, complete=False), gen).cls == 0
    # 1: a single-factor F1 condition fails and so does a nearer F6 perturbation
    frag = _with(base, cn == "F1/real/real/const", complete=False, d_ood=2.0)
    frag = _with(frag, cn == "F6/wobble0.3", complete=False, d_ood=1.0)
    r = A.p4_classify(frag, gen)
    assert r.cls == 1 and r.details["failing_condition"] == "F1/real/real/const"
    # 2, width-keyed
    wk = _with(base, np.isin(cn, [c for c in A.F1_CONDS if c.endswith("/const")]),
               complete=False, d_ood=1.5)
    r = A.p4_classify(wk, gen)
    assert r.cls == 2 and r.key_factors == ("width",) and "width-keyed" in r.label
    # 2, vacuous (no F6 perturbation as far OOD as the failing condition)
    vac = _with(wk, np.isin(cn, A.F6_CONDS), d_ood=0.5)
    assert A.p4_classify(vac, gen).details["vacuous"]
    # 3 via the mirror
    assert A.p4_classify(_with(base, cn == "F4/mirror", complete=False), gen).cls == 3
    # 4: class 5 plus tier A completion and no tier G crash
    tier_a = A._to_table([dict(tiers="A", approach="A2", R_seg=20.0, n_runs=8, det_complete=8,
                               excluded="", braking=True, no_cap=False)],
                         ("tiers", "approach", "R_seg", "n_runs", "det_complete", "excluded",
                          "braking", "no_cap"))
    tier_g = A._to_table([dict(tiers="G", det_crashes=0)], ("tiers", "det_crashes"))
    assert A.p4_classify(base, gen, tier_a, tier_g).cls == 4
    crash_g = A._to_table([dict(tiers="G", det_crashes=1)], ("tiers", "det_crashes"))
    assert A.p4_classify(base, gen, tier_a, crash_g).cls == 5
    # 6: everything completes, but a partial phantom slowing (phi = 0.3) fits no class
    part = {k: np.array(v, copy=True) for k, v in base.items()}
    v_a = {int(c): v for c, n, v in zip(base["corner"], cn, base["v_min"], strict=True)
           if n == COND_A}
    for i in np.flatnonzero(cn == "F5/swapB"):
        part["v_min_region"][i] = 80.0 - 0.3 * (80.0 - v_a[int(base["corner"][i])])
    r = A.p4_classify(part, gen)
    assert r.phi == pytest.approx(0.3) and r.cls == 6 and r.label == "mixed"


def test_p4_indices_and_recipe():
    assert A.phantom_index(80.0, 80.0, 30.0) == 0.0
    assert A.phantom_index(80.0, 30.0, 30.0) == 1.0
    assert A.swap_index(25.0, 40.0, 25.0) == pytest.approx(0.0)
    assert A.swap_index(40.0, 40.0, 25.0) == pytest.approx(1.0)
    assert np.isnan(A.swap_index(30.0, 30.0, 30.0))
    rs = [A.P4Result(a, c, "", {}, 1.0, 0.0, 0.0, 0.0) for a, c in (("S1", 2), ("S2", 2),
                                                                   ("S3", 5))]
    d = A.p4_recipe(rs)
    assert d.verdict == "class 2" and d.family == "primary" and not d.details[
        "predicted_class_2_all"]
    rs[1].cls = 3
    assert A.p4_recipe(rs).verdict == "inconclusive"
    assert A.p4(rs[0]).verdict == "class 2" and A.p4(rs[0]).family == "secondary"


# ------------------------------------------------------------------------- P5


def _sym_runs(shift_right=0.0, crash_right=False):
    rows = []
    for p in range(6):
        for c, d in ((2 * p, 1), (2 * p + 1, -1)):
            for k in range(4):
                e = -0.01 + (shift_right if d < 0 else 0.0) + 0.001 * k
                rows.append(dict(agent="G1", cell=c, pair=p, phase=k, mode="det", steer_sign=d,
                                 e_ve=e, d_b=100.0 + k, has_brake=True, forced_onset=False,
                                 crashed=bool(crash_right and d < 0 and k == 0),
                                 crash_type="", s_crash=np.nan))
    keys = A._to_table([dict(cell=c, direction=1 if c % 2 == 0 else -1, R_nom=14.6, s_c0=1000.0,
                             key_status="verified") for c in range(12)],
                       ("cell", "direction", "R_nom", "s_c0", "key_status"))
    return A._to_table(rows, sorted(rows[0])), keys


def test_p5_symmetry():
    runs, keys = _sym_runs()
    d = A.p5(runs, keys)
    assert d.verdict == "confirm" and d.details["concordance"] == 1.0 and d.family == "control"
    runs, keys = _sym_runs(shift_right=0.05)
    d = A.p5(runs, keys)
    assert d.verdict == "refute" and d.details["right_minus_left_e_ve_tight"] == pytest.approx(0.05)
    runs, keys = _sym_runs(crash_right=True)
    d = A.p5(runs, keys)
    assert d.verdict == "refute" and d.details["concordance"] == pytest.approx(0.75)


# ------------------------------------------------------------------------- P6


def test_t_r():
    comp = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], bool)
    t = A.t_r(comp, np.array([10, 20, 30, 40]))
    assert t.tolist() == [30.0, 40.0, 10.0]
    t = A.t_r(np.array([[1, 0], [1, 0]], bool), np.array([1, 2]))
    assert t[0] == 1.0 and np.isinf(t[1])
    np.testing.assert_allclose(A.t_r_ranks([5.0, np.inf, np.inf, 1.0]), [2.0, 3.5, 3.5, 1.0])


def _ckpt_pairs(levels, competent_from, tail_e=None, ckpt=0):
    rows = []
    for R in levels:
        ok = ckpt >= competent_from[R]
        rows.append(dict(agent="G1", tiers="A", approach="A2", braking=True, no_cap=False,
                         excluded="", R_nom=R, n_runs=8, det_complete=8 if ok else 6,
                         e_ve=-0.01, e_ve_bo=0.01, uses_ra=False,
                         regime="bulk" if R >= 20 else "tail"))
        if R < 20:
            rows.append(dict(agent="G1", tiers="A", approach="A1", braking=True, no_cap=False,
                             excluded="", R_nom=R, n_runs=8, det_complete=8,
                             e_ve=(tail_e or {}).get(ckpt, 0.0), e_ve_bo=0.0,
                             e_ve_ra=(tail_e or {}).get(ckpt, 0.0), e_ve_bo_ra=0.0,
                             uses_ra=True, regime="tail"))
    return A._to_table(rows, sorted({k for r in rows for k in r}))


def test_R11_p6a_confirm_needs_the_permutation_p(monkeypatch):
    """Ruling R11 (AD18): rho <= -0.5 confirms P6a only with permutation p <= 0.05."""
    levels = (11.7, 14.6, 22.9, 44.7, 87.3, 109.0)
    order = {109.0: 1, 87.3: 2, 44.7: 3, 22.9: 4, 14.6: 5, 11.7: 6}
    by = {c: _ckpt_pairs(levels, order, None, c) for c in range(1, 8)}
    gate = {c: True for c in by}
    monkeypatch.setattr(A, "permutation_spearman", lambda *a, **k: (-0.9, 0.2))
    d = A.p6(by, B, gate)[0]
    assert d.verdict == "inconclusive" and d.details["p"] == 0.2
    monkeypatch.setattr(A, "permutation_spearman", lambda *a, **k: (-0.9, 0.05))
    assert A.p6(by, B, gate)[0].verdict == "confirm"


def test_p6_order_and_overspeed_timing():
    levels = (11.7, 14.6, 22.9, 44.7, 87.3, 109.0)
    order = {109.0: 1, 87.3: 2, 44.7: 3, 22.9: 4, 14.6: 5, 11.7: 6}
    tail_e = {5: 0.0, 6: 0.2, 7: 0.2}
    by = {c: _ckpt_pairs(levels, order, tail_e, c) for c in range(1, 8)}
    gate = {c: True for c in by}
    d6a, d6b, d6c, d6d = A.p6(by, B, gate, log_std={c: [-3.0, -2.5] for c in by})
    assert d6a.verdict == "confirm" and d6a.details["rho"] == pytest.approx(-1.0)
    assert d6a.details["p"] < 0.01
    assert d6b.verdict == "refute" and d6b.family == "exploratory"
    assert d6c.verdict == "confirm" and d6c.details["first_overspeed"] == 6
    assert d6c.details["t_bulk"] == 4.0
    assert d6d.details["large_first"]
    # reversed order: small R first -> refuted
    rev = {R: 7 - v for R, v in order.items()}
    by = {c: _ckpt_pairs(levels, rev, None, c) for c in range(1, 8)}
    assert A.p6(by, B, gate)[0].verdict == "refute"
    # overspeed appears before the bulk is competent -> P6c refuted
    by = {c: _ckpt_pairs(levels, order, {c: 0.2 for c in range(2, 8)}, c) for c in range(1, 8)}
    assert A.p6(by, B, gate)[2].verdict == "refute"
    # gate failures count as not competent
    gate_f = {c: c != 7 for c in range(1, 8)}
    by = {c: _ckpt_pairs(levels, order, None, c) for c in range(1, 8)}
    assert A.p6(by, B, gate_f)[0].verdict == "not testable"  # every t_R = inf


def _gate_pairs(gentle_complete=8, null_complete=8, osc=0.01, null=True):
    rows = [dict(tiers="A", approach=ap, R_nom=R, width=12.0, excluded="", n_runs=8,
                 det_complete=gentle_complete, osc_index=osc)
            for ap in ("A1", "A2") for R in (87.3, 109.0, 171.0, 300.0)]
    if null:
        rows += [dict(tiers="G", approach="G", R_nom=np.nan, width=12.0, excluded="", n_runs=4,
                      det_complete=null_complete, osc_index=osc) for _ in range(8)]
    return A._to_table(rows, ("tiers", "approach", "R_nom", "width", "excluded", "n_runs",
                              "det_complete", "osc_index"))


def test_checkpoint_gate():
    keys = make_keys(r_values=R_BRAKING + (109.0, 171.0, 300.0), tier_b=False)
    runs = A.simulate_runs("lawful", keys, A._dry_rng(2))
    p = A.pair_values(runs, keys, {"lawful": REG}, B)
    g = A.checkpoint_gate(p)
    # Gentle cells complete, but there are no tier G straights: not evaluable, not passed.
    assert g["completion_gentle"] == 1.0 and not g["evaluable"] and not g["passed"]
    assert A.checkpoint_gate(_gate_pairs())["passed"]
    # Each set needs >= 90% on its own: 64/64 gentle runs and 24/32 null runs pool to
    # 88/96 = 92% (the old pooled rule passed), but the null straights alone are at 75%.
    g = A.checkpoint_gate(_gate_pairs(null_complete=3))
    assert g["completion"] >= 0.9 and g["completion_null"] == 0.75 and not g["passed"]
    assert not A.checkpoint_gate(_gate_pairs(osc=0.2))["passed"]
    g = A.checkpoint_gate(_gate_pairs(osc=np.nan))         # undefined oscillation: not passed
    assert not g["evaluable"] and not g["passed"]


# ------------------------------------------------------------------------- P7


def _p7_tables(short_excess, long_excess, a_agent, a_qp):
    sd, al = [], []
    for R in (9.4, 11.7, 14.6, 28.6, 55.9):
        for th in (90.0, 180.0):
            short = th == 90.0 and R <= 14.6
            ex = short_excess if short else (long_excess if R <= 14.6 else 0.2)
            sd.append(dict(agent="G1", geometry=f"R{R}/T{th}", R_nom=R, theta_deg=th,
                           sd_agent=1.0 + ex, sd_key=np.nan, sd_qp=1.0, v_b=86.0, sd_null=1.0,
                           excess=ex))
            al.append(dict(agent="G1", geometry=f"R{R}/T{th}", R_nom=R, theta_deg=th,
                           short=short, a_agent=a_agent, a_q=0.0, a_qp=a_qp))
    return A._to_table(sd, A.P7_SD_COLUMNS), A._to_table(al, A.P7_ALIAS_COLUMNS)


def test_p7_verdicts_and_null():
    assert A.p7_null(None, 0.5, 310 / 3.6) == pytest.approx(310 / 3.6 * 0.05 / math.sqrt(12))
    assert A.p7_null(2.0, 0.5, 80.0) == 2.0
    assert A.p7_null(np.nan, 3.0, 80.0) == 3.0
    assert A.p7(*_p7_tables(3.0, 0.5, 0.04, 0.01)).verdict == "confirm"
    assert A.p7(*_p7_tables(0.3, 0.3, 0.01, 0.02)).verdict == "refute"
    assert A.p7(*_p7_tables(0.3, 0.3, 0.05, 0.02)).verdict == "mixed"
    assert A.p7(*_p7_tables(3.0, 0.5, 0.015, 0.01)).verdict == "inconclusive"


def test_phase_sd_table():
    keys = make_keys(r_values=(9.4, 28.6), tier_b=False)
    keys["n_phases"] = np.where(keys["approach"] == "A1", 16, 4)
    runs = A.simulate_runs("lawful", keys, A._dry_rng(3), n_native=0, n_matched=0)
    sd = A.phase_sd_table(runs, keys, qp_runs=runs, key_sd={"syn/R9.4/T90/W12/typical": 9.0})
    assert A._n(sd) == 4                                   # 2 R x 2 theta at A1
    row = sd["geometry"] == "syn/R9.4/T90/W12/typical"
    assert sd["sd_null"][row][0] == 9.0 and sd["sd_key"][row][0] == 9.0
    assert np.all(sd["sd_agent"] > 0) and np.allclose(sd["sd_agent"], sd["sd_qp"])


# ------------------------------------------------------------------------- P8


def _ladder(tail_single=3, tail_repro=3, bulk_context=True):
    rows = []

    def add(agent, unit, set_, version, det, fid=True, e_ve=0.0, ov=0):
        rows.append(dict(agent=agent, unit=unit, set=set_, version=version, fidelity_ok=fid,
                         det_crashes=det, det_n=4, nat_k=0, nat_n=16, e_ve=e_ve, det_overspeed=ov))

    for i in range(5):
        u = f"G2/T{i}"
        repro = i < tail_repro
        single = i < tail_single
        add("G2", u, "tail", "real", 4 if repro else 0)
        add("G2", u, "tail", "isolated", 3 if single else 0)
        for f in ("before_straight", "after_straight", "const_width"):
            add("G2", u, "tail", f, 0 if f == "before_straight" else 3)
        add("G2", u, "tail", "twin", 2 if single else 0, e_ve=0.2)
    for i, agent in enumerate(("G1", "G1", "G2")):
        u = f"{agent}/B{i}"
        add(agent, u, "bulk", "real", 4)
        add(agent, u, "bulk", "isolated", 0 if bulk_context else 4)
        for f in ("before_straight", "after_straight", "const_width"):
            add(agent, u, "bulk", f, 2)
        add(agent, u, "bulk", "twin", 0)
    return A._to_table(rows, A.LADDER_COLUMNS)


def test_p8_verdicts():
    ds = {d.prediction + d.agent: d for d in A.p8(_ladder())}
    assert ds["P8-tailG2"].verdict == "confirm"
    labels = ds["P8-tailG2"].details["labels"]
    assert labels["G2/T0"] == "single-corner" and labels["G2/T4"] == "not reproduced"
    assert ds["P8-bulkG1,G2"].verdict == "confirm"
    assert ds["P8-FG2"].details["F"] == pytest.approx(3 / 4)   # 3 of 4 reproduced G2 units
    ds = {d.prediction + d.agent: d for d in A.p8(_ladder(tail_single=1))}
    assert ds["P8-tailG2"].verdict == "refute"
    units = A.p8_units(_ladder(tail_single=1))
    t1 = units["label"][units["unit"] == "G2/T1"][0]
    assert t1 == "context: before_straight"
    ds = {d.prediction + d.agent: d for d in A.p8(_ladder(tail_single=2))}
    assert ds["P8-tailG2"].verdict == "inconclusive"
    ds = {d.prediction + d.agent: d for d in A.p8(_ladder(bulk_context=False))}
    assert ds["P8-bulkG1,G2"].verdict == "refute"


def test_p8_twin_transfer_and_curve(sims):
    rows = [dict(agent="G2", corner=f"c{i}", source="E1", e_ve_real=0.01 * i,
                 e_ve_isolated=0.01 * i, e_ve_twin=0.01 * i + 0.01, e_ve_curve=0.02 * i)
            for i in range(10)]
    twins = A._to_table(rows, A.TWIN_COLUMNS)
    d = next(x for x in A.p8(_ladder(), twins) if x.prediction == "P8-twin")
    assert d.verdict == "confirm" and d.details["spearman"] == pytest.approx(1.0)
    bad = {k: np.array(v, copy=True) for k, v in twins.items()}
    bad["e_ve_twin"] = bad["e_ve_twin"] + 0.2
    bad["e_ve_curve"] = -bad["e_ve_curve"]
    d = next(x for x in A.p8(_ladder(), bad) if x.prediction == "P8-twin")
    assert d.verdict == "refute"
    _, lawful = sims["lawful"]
    assert A.tier_a_curve_at(lawful, 13.0, 90.0, "A2") == pytest.approx(-0.01, abs=0.01)


# ------------------------------------------------------------------------- plumbing


def _stim(i, name, direction, R=20.0):
    spec = CornerSpec(R, 90.0, direction, l_r=20.0)
    lm = Landmarks(1000.0, 1020.0, 1000.0 + spec.L_c / 2, 1000.0 + spec.L_c - 20.0,
                   1000.0 + spec.L_c)
    geo = Geometry(R, 90.0, 1 / R / 20, R, 90.0, spec.L_c, 0.0, 1030.0, 1, False, 6.0)
    if direction == 0:                                       # a straight has no CornerSpec
        return Stimulus(i, name, "straight", ("G",), 0, -1, lm, geo, None)
    return Stimulus(i, name, "synthetic", ("A",), direction, 1 - i, lm, geo, spec)


class FakeKeys:
    def __init__(self, status="verified"):
        self.status = status

    def nominal(self, cell, phase):
        if "G" in cell.tiers:
            return None
        m = measures.ResponseMetrics(v_e=30.0, v_min=25.0, d_b=100.0, v_b=60.0, has_brake=True)
        return A.KeyView(f"key{cell.cell_id}", m, phase in (0, 2), self.status, False,
                         {"v_min": 0.2}, {"lock_limited": cell.cell_id == 1})

    def backoff(self, cell):
        return A.KeyView(None, measures.ResponseMetrics(v_e=29.0, v_min=24.0))

    def anchored(self, cell, agent, timesteps, phase=0):
        if agent != "G1":
            return None
        return A.KeyView(None, measures.ResponseMetrics(v_e=31.0))

    def null(self, cell):
        return None

    def socp(self, cell):
        return A.KeyView(None, measures.ResponseMetrics(v_e=30.5, v_min=24.5, d_b=95.0,
                                                        v_b=60.0, has_brake=True))

    def no_return(self, cell, agent, timesteps, phase, sample, traj, key_traj):
        return "NR"


def _bank_cells():
    stims = [_stim(0, "syn/R20/T90/W12/typical/L", 1), _stim(1, "syn/R20/T90/W12/typical/R", -1),
             _stim(2, "straight/W12", 0)]
    bank = SimpleNamespace(stimuli=stims, w_left=np.full((3, 600), 6.0),
                           w_right=np.full((3, 600), 6.0), ds=2.0)
    cells = [Cell(0, 0, 4, ("A",), "A2", 50.0, 300.0, 400, null_cell=2),
             Cell(1, 1, 4, ("A",), "A2", 50.0, 300.0, 400, null_cell=2),
             Cell(2, 2, 9, ("G",), "G", 50.0, 600.0, 400)]
    return bank, cells


def test_key_table_from_a_key_source():
    bank, cells = _bank_cells()
    kt = A.key_table(cells, bank, FakeKeys(), extra={0: {"v_min_qss": 22.0}})
    assert kt["geometry"].tolist() == ["syn/R20/T90/W12/typical"] * 2 + ["straight/W12"]
    assert kt["braking"].tolist() == [True, True, False]          # 25 < 50 - 0.2; G: no key
    assert kt["key_status"].tolist() == ["verified", "verified", "missing"]
    assert kt["lock_limited"].tolist() == [False, True, False]
    assert kt["v_e_star_bo"][0] == 29.0 and kt["v_min_qss"][0] == 22.0
    assert kt["braking_socp"][0] and kt["d_b_socp"][0] == 95.0
    assert kt["s_c0"][0] == 1000.0 and kt["R_nom"][0] == 20.0
    assert kt["width"].tolist() == [12.0, 12.0, 12.0]       # straights: from the samples
    unsolved = A.key_table(cells, bank, FakeKeys("unsolved"))
    assert not unsolved["braking"].any()


def test_run_table_plumbing(monkeypatch):
    bank, cells = _bank_cells()
    v_e = np.array([33.0, 30.0, 27.0, 30.0, 40.0])
    det = dict(cell=np.array([0, 0, 1, 2, 2]), phase=np.array([0, 1, 2, 0, 1]),
               sample=np.full(5, -1), steer_sign=np.array([1, 1, -1, 1, 1]),
               W=np.full(5, 400), s0=np.full(5, 700.0), v_e_test=v_e)
    nat = dict(cell=np.array([1]), phase=np.array([3]), sample=np.array([7]),
               steer_sign=np.array([-1]), W=np.array([400]), s0=np.array([700.0]),
               v_e_test=np.array([36.0]))
    calls = {}

    def fake_traj(arr, i, g):
        return SimpleNamespace(v_e=float(arr["v_e_test"][i]), g=g, i=i)

    def fake_metrics(traj, marks, null, tables):
        calls.setdefault("nulls", []).append(null)
        assert marks.s0 == 700.0 and marks.W == 400
        return measures.ResponseMetrics(v_e=traj.v_e, crashed=traj.v_e > 35, v_min=traj.v_e - 5)

    def fake_errors(m, mk, traj, key, marks, null, null_key, at_phase, **kw):
        return measures.KeyErrors(e_ve=m.v_e / mk.v_e - 1, regret=0.0 if at_phase else np.nan)

    monkeypatch.setattr(measures, "trajectory_from_rollout", fake_traj)
    monkeypatch.setattr(measures, "response_metrics", fake_metrics)
    monkeypatch.setattr(measures, "key_errors", fake_errors)
    monkeypatch.setattr(measures, "crash_type", lambda traj, marks, key, nr: f"overspeed:{nr}")
    rt = A.run_table("G1", 123, {"det": det, "nat": nat}, cells, bank, FakeKeys(), None,
                     check_design=False)
    assert set(rt) == set(A.RUN_COLUMNS) and A._n(rt) == 6
    assert rt["mode"].tolist() == ["det"] * 5 + ["nat"]
    np.testing.assert_allclose(rt["e_ve"][:3], v_e[:3] / 30 - 1)
    np.testing.assert_allclose(rt["e_ve_bo"][:3], v_e[:3] / 29 - 1)
    np.testing.assert_allclose(rt["e_ve_ra"][:3], v_e[:3] / 31 - 1)
    assert rt["v_e_star_ra"][0] == 31.0
    assert rt["key_at_phase"].tolist()[:3] == [True, False, True]
    assert np.isnan(rt["e_ve"][3])                          # tier G: no nominal key
    assert rt["crash_type"].tolist() == ["", "", "", "", "overspeed:NR", "overspeed:NR"]
    # the null run is the driver's own phase 0 run on the null cell
    assert calls["nulls"][0] is not None and calls["nulls"][0].i == 3
    # an agent without anchored keys falls back to the nominal error
    rt2 = A.run_table("G2", 123, ({"det": det}, {}), cells, bank, FakeKeys(), None,
                      check_design=False)
    np.testing.assert_allclose(rt2["e_ve_ra"][:3], rt2["e_ve"][:3])
    assert rt2["v_e_star_ra"][0] == 30.0
    kt = A.key_table(cells, bank, FakeKeys())
    pv = A.pair_values(rt, kt, {"G1": REG}, B)
    assert pv["pair"].tolist() == [4, 9] and pv["n_runs"].tolist() == [3, 2]
    assert pv["nat_n"].tolist() == [1, 0] and pv["nat_k"][0] == 1


def test_cond_table():
    keys = A._to_table([dict(cell=i, corner=i // 2, condition=COND_A if i % 2 == 0 else
                             "F5/swapB", braking_socp=True, v_min_socp=20.0 + i, swap_j=-1)
                        for i in range(4)],
                       ("cell", "corner", "condition", "braking_socp", "v_min_socp", "swap_j"))
    rows = [dict(agent="S1", timesteps=0, cell=c, mode="det", completed=k < 3, v_min=20.0 + k,
                 v_peak=80.0) for c in range(4) for k in range(4)]
    runs = A._to_table(rows, sorted(rows[0]))
    ct = A.cond_table(runs, keys, {("S1", 0): 1.5})
    assert A._n(ct) == 4 and ct["completes"].tolist() == [3, 3, 3, 3]
    assert ct["complete"].all() and ct["braking"].all()
    assert ct["v_min_region"][1] == ct["v_min"][1] == 21.5
    assert ct["d_ood"][0] == 1.5 and np.isnan(ct["d_ood"][1])


# ------------------------------------------------------------------------- dry run, power


def test_dry_run_returns_every_planted_verdict():
    out = A.dry_run(KEYS, TABLES, B, design=DESIGN)
    failed = [(k, p, v, e) for k, p, v, e, ok in zip(out["kind"], out["prediction"],
                                                     out["verdict"], out["expected"],
                                                     out["passed"], strict=True) if not ok]
    assert not failed, failed
    assert set(out["kind"]) == set(A.DRY_RUN_AGENTS)


def test_power_table():
    pt = A.power_table(KEYS, {"G1": A.Regime(11.4, 20.2), "G2": A.Regime(10.4, 16.4, 7.6)}, B,
                       TABLES, steps=A.POWER_STEPS[:6])
    assert A._n(pt) == 10 and set(pt["test"]) == {"P1a", "P1-tail", "P1b", "P2", "P3"}
    row = (pt["test"] == "P1a") & (pt["agent"] == "G1")
    assert pt["pairs_A2_bulk"][row][0] == 12 and pt["pairs_A2_clip"][row][0] == 0
    g2 = pt["agent"] == "G2"
    assert pt["pairs_A1_clip"][g2][0] == 4                # R 6.0 and 7.5 x 2 theta
    # the band is 7.5%; the lawful agent sits at -1%, so detection needs > 8.6% planted
    assert pt["min_overspeed"][row][0] == pytest.approx(0.1)
    assert pt["lawful_verdict"][row][0] == "confirm"
    for test in ("P1-tail", "P1b", "P2"):
        m = pt["test"] == test
        assert np.all(np.isfinite(pt["min_overspeed"][m])) and np.all(pt["min_overspeed"][m] <= 0.15)
    p3 = pt["test"] == "P3"
    assert pt["lawful_verdict"][p3].tolist() == ["confirm"] * 2
    assert pt["planted_verdict"][p3].tolist() == ["refute:M_idx"] * 2


def test_R18_power_table_keeps_the_other_agents_rows():
    """A replicate's regime adds its rows after the others' and changes none of them (each
    agent's simulation streams follow its place in the order)."""
    reg = {"G1": A.Regime(11.4, 20.2), "G2": A.Regime(10.4, 16.4, 7.6)}
    plain = A.power_table(KEYS, reg, B, None, steps=A.POWER_STEPS[:1])
    more = A.power_table(KEYS, {**reg, "G1s1": A.Regime(11.4, 20.2, 9.0)}, B, None,
                         steps=A.POWER_STEPS[:1])
    n = A._n(plain)
    assert A._n(more) == n + 5 and set(more["agent"][n:]) == {"G1s1"}
    for c in A.POWER_COLUMNS:
        np.testing.assert_array_equal(more[c][:n], plain[c][:n], err_msg=c)


# ------------------------------------------------------------------------- decision table


def _inputs(sims):
    runs = {"G1": sims["lawful"][0], "G2": sims["late_onset"][0], "G3": sims["lawful"][0]}
    runs = {a: {**r, "agent": np.full(A._n(r), a)} for a, r in runs.items()}
    hist = {"G1": [runs["G1"], runs["G1"]],
            "G2": [{**sims["lawful"][0], "agent": np.full(A._n(sims["lawful"][0]), "G2")}] * 2}
    conds = {s: A.simulate_conds("context_keyed", DESIGN, np.random.default_rng(i), agent=s)
             for i, s in enumerate(A.SPECIALISTS)}
    conds["G1"] = A.simulate_conds("reader", DESIGN, np.random.default_rng(9), agent="G1")
    regimes = {a: REG for a in A.GENERALISTS}
    # The simulated snapshots (timesteps 0) pass protocol 8's gate (the stage computes it
    # from their gentle and tier G runs; the simulations have no tier G).
    gates = {a: {0: True} for a in A.GENERALISTS}
    return A.AnalysisInputs(b=B, keys=KEYS, regimes=regimes, runs=runs, history=hist,
                            tables=TABLES, conds=conds, ladder=_ladder(), gates=gates)


def test_decision_table_confirmatory(sims, tmp_path):
    inp = _inputs(sims)
    t = A.decision_table(inp, True, tmp_path)
    assert set(t) == set(A.DECISION_COLUMNS)
    got = {(p, a): (v, s, f) for p, a, v, s, f in zip(t["prediction"], t["agent"], t["verdict"],
                                                       t["stable"], t["family"], strict=True)}
    assert got[("P1a", "G1")] == ("confirm", "yes", "secondary")
    assert got[("P1a", "G3")] == ("confirm", "n/a", "primary")
    assert got[("P3", "G1")][0] == "confirm"
    # G2's late onset confirms P1-tail at the final, but its history was lawful: unstable
    assert got[("P1-tail", "G2")][:2] == ("unstable", "no")
    d = json.loads(t["details"][(t["prediction"] == "P1-tail") & (t["agent"] == "G2")][0])
    assert d["final_verdict"] == "confirm" and d["history_verdicts"] == ["refute", "refute"]
    assert got[("P4", "S2")] == ("class 2", "n/a", "primary")
    assert got[("P4-recipe", "S1,S2,S3")][0] == "class 2"
    assert got[("P8-tail", "G2")][0] == "confirm"
    assert ("P5", "G1") not in got                          # confirmatory run only
    assert (tmp_path / "decisions.npz").exists() and (tmp_path / "decisions.json").exists()
    back = load_table(tmp_path / "decisions.npz")
    assert back["verdict"].tolist() == t["verdict"].tolist()
    with pytest.raises(FileExistsError):
        A.decision_table(inp, True, tmp_path)


def _with_replicates(sims, inp):
    """Ruling R18's six replicates on top of ``_inputs``: v1 G1s1 lawful, G1s2 a
    time-to-corner rule; v2 G2s1 late onset, G2s2 overspeeding below 12 m; v3 G3s1
    overspeeding below 12 m, G3s2 late onset. G1s1 has a lawful history; the others none."""
    kinds = {"G1s1": "lawful", "G1s2": "time_to_corner", "G2s1": "late_onset",
             "G2s2": "overspeed_below_12", "G3s1": "overspeed_below_12", "G3s2": "late_onset"}
    for a, k in kinds.items():
        r = sims[k][0]
        inp.runs[a] = {**r, "agent": np.full(A._n(r), a)}
        inp.regimes[a] = REG
        inp.gates[a] = {0: True}
    inp.history["G1s1"] = [inp.runs["G1s1"]] * 2
    return inp


def test_R18_decision_table_writes_the_replication_family(sims, tmp_path):
    """Ruling R18: with the replicates' finals in the inputs the confirmatory table's rows
    are the pre-registered ones, unchanged; the replicates' P1a-P3 (family 'replication'),
    the variant-level 2-of-3 table and the seed spread go to replication_*, written once."""
    plain = A.decision_table(_inputs(sims), True, tmp_path / "plain")
    inp = _with_replicates(sims, _inputs(sims))
    t = A.decision_table(inp, True, tmp_path)
    for c in A.DECISION_COLUMNS:
        assert t[c].tolist() == plain[c].tolist(), c
    assert not (tmp_path / "plain" / "replication.json").exists()
    rd = load_table(tmp_path / "replication_decisions.npz")
    assert set(rd["agent"]) == set(registry.replicates())
    assert set(rd["family"]) == {"replication"} and A._n(rd) == 6 * len(A.CONFIRMATORY_G)
    got = {(p, a): (v, s) for p, a, v, s in zip(rd["prediction"], rd["agent"], rd["verdict"],
                                                rd["stable"], strict=True)}
    assert got[("P1a", "G1s1")] == ("confirm", "yes")          # lawful history: stable
    assert got[("P1-tail", "G2s1")] == ("confirm", "n/a")      # no history given
    vt = load_table(tmp_path / "replication_variants.npz")
    row = {(v, p): i for i, (v, p) in enumerate(zip(vt["variant"], vt["prediction"],
                                                    strict=True))}
    assert A._n(vt) == 15
    i = row[("v1", "P1a")]                                     # confirmed by all three seeds
    assert (vt["verdict"][i], bool(vt["claim"][i]), int(vt["n_agree"][i])) == ("confirm", True, 3)
    i = row[("v1", "P3")]                                      # G1s2's time-to-corner rule
    assert (vt["verdict"][i], int(vt["n_agree"][i])) == ("confirm", 2)
    assert json.loads(vt["seed_verdicts"][i])["G1s2"] == "refute"
    i = row[("v2", "P1-tail")]                                 # unstable, confirm, refute
    assert vt["verdict"][i] == A.SEED_SPECIFIC and not vt["claim"][i]
    assert json.loads(vt["seed_verdicts"][i]) == {"G2": "unstable", "G2s1": "confirm",
                                                  "G2s2": "refute"}
    i = row[("v1", "P2")]                                      # a shared non-claim
    assert (vt["verdict"][i], bool(vt["claim"][i])) == ("not applicable", False)
    sp = load_table(tmp_path / "replication_spread.npz")
    assert A._n(sp) == 9 and sp["agent"].tolist()[:3] == ["G1", "G1s1", "G1s2"]
    k = sp["agent"].tolist().index("G2s1")
    assert np.isinf(sp["r_dep_A1"][k]) and sp["P2"][k] == "confirm"
    assert "frac_onset_late" in json.loads(sp["numbers"][k])["P2"]
    rep = json.loads((tmp_path / "replication.json").read_text())
    assert rep["confirmatory"] and len(rep["decisions"]) == 30
    # one run: the replication family is spent with the confirmatory table
    (tmp_path / "decisions.npz").unlink()
    with pytest.raises(FileExistsError, match="replication.json"):
        A.decision_table(inp, True, tmp_path)
    # the exploratory set: replicate rows (replication / control families) and P1c on each
    # variant's median R_dep over its seeds (v1 6, v2 median(inf, inf, 13.1), v3 13.1)
    ds = A.decisions(inp, False)
    fam = {(d.prediction, d.agent): d.family for d in ds}
    assert fam[("P1a", "G3s2")] == "replication" and fam[("P5", "G2s1")] == "control"
    assert fam[("P2b", "G1s1")] == "replication" and fam[("P1a", "G3")] == "primary"
    p1c = next(d for d in ds if d.prediction == "P1c")
    assert p1c.agent == "G1,G2,G3" and p1c.verdict == "refute"
    assert p1c.details["r_dep_A1"] == (6.0, np.inf, 13.1)
    assert p1c.details["r_dep_A1_seed0"] == (6.0, np.inf, 6.0)
    assert not [d for d in ds if d.prediction.startswith("P6") or (
        d.prediction.startswith("P8") and registry.is_replicate(d.agent))]


def test_R18_review_history_snapshots_without_tier_g_cannot_pass_the_gate(sims):
    """R18 review finding 1. Protocol 8's gate needs the tier G width-12 straights, so a
    history snapshot without them (protocol 4's last-3 rows before ruling R19a ran tiers A
    and B only; since R19a such inputs are incomplete) is 'not testable' (AD24) and a claim
    at the final becomes 'unstable' (AD21). The decisions record ``history_gate_judgeable``,
    and ``history_gate_problems`` names every generalist whose stability rule cannot be
    applied (the confirmatory stage refuses on it). R19a's complete inputs:
    ``test_R19a_history_snapshots_with_tier_g_are_judged_and_claims_stay_stable``."""
    inp = _inputs(sims)
    n = A._n(inp.runs["G1"])
    # history snapshots at their own timesteps, with no gate given: computed from the runs
    inp.history["G1"] = [{**inp.runs["G1"], "timesteps": np.full(n, t)} for t in (5, 6)]
    ds = {(d.prediction, d.agent): d for d in A.decisions(inp, True)}
    p1a = ds[("P1a", "G1")]
    assert p1a.verdict == "unstable" and p1a.details["final_verdict"] == "confirm"
    assert p1a.details["history_verdicts"] == ["not testable", "not testable"]
    assert p1a.details["history_gate_judgeable"] == [False, False]
    assert "history_gate_judgeable" not in ds[("P1a", "G3")].details    # no history: n/a
    probs = A.history_gate_problems(inp)
    assert [p.split(":")[0] for p in probs] == ["G1", "G2", "G3"]
    assert "gate cannot judge" in probs[0] and "[5, 6]" in probs[0] and "R19a" in probs[0]
    assert probs[2].startswith("G3: 0 of the 2 snapshots before the final")
    # a history snapshot with gentle and tier G width-12 runs can be judged
    pairs = dict(R_nom=np.array([87.3, 87.3, np.nan]), width=np.array([12.0] * 3),
                 tiers=np.array(["A", "A", "G"]), approach=np.array(["A1", "A2", "G"]),
                 excluded=np.array([""] * 3), n_runs=np.array([8] * 3),
                 det_complete=np.array([8] * 3), osc_index=np.array([0.01] * 3))
    assert A.checkpoint_gate(pairs)["evaluable"]
    assert not A.checkpoint_gate({k: v[:2] for k, v in pairs.items()})["evaluable"]


def _tier_g_keys(keys: dict, n: int = 3) -> tuple[dict, list[dict]]:
    """``keys`` plus ``n`` tier G null straights (width 12, approach 'G', no corner), each its
    own unmirrored pair as in the real design, and their key rows."""
    i0 = int(np.max(keys["cell"])) + 1
    p0 = int(np.max(keys["pair"])) + 1
    base = {c: np.asarray(v)[0].item() for c, v in keys.items()}
    rows = [dict(base, cell=i0 + j, stim=i0 + j, pair=p0 + j, name="straight/W12",
                 geometry="straight/W12", tiers="G", approach="G", v0=kph(140.0 + 40 * j),
                 D=600.0, W=400, n_phases=4, direction=0, R_nom=np.nan, theta_deg=0.0,
                 width=12.0, R_seg=np.inf, R_eff=np.inf, braking=False, has_brake_star=False)
            for j in range(n)]
    extra = A._to_table(rows, A.KEY_COLUMNS)
    return {c: np.concatenate([np.asarray(keys[c]), extra[c]]) for c in A.KEY_COLUMNS}, rows


def _with_tier_g(runs: dict, g_rows: list[dict], agent: str, ts: int,
                 completed: bool = True) -> dict:
    """A history snapshot's run table at ``ts``: ``runs`` plus the deterministic tier G runs
    that ruling R19a adds to protocol 4's last-3 rows (every tier G cell, 4 phases)."""
    cols = list(runs)
    extra = [A._sim_run(agent, r, k, -1, "det", 1, not completed, 0.0, False, {})
             for r in g_rows for k in range(4)]
    g = A._to_table(extra, cols)
    out = {c: np.concatenate([np.asarray(runs[c]), g[c]]) for c in cols}
    out["agent"] = np.full(A._n(out), agent)
    out["timesteps"] = np.full(A._n(out), ts)
    return out


def test_R19a_history_snapshots_with_tier_g_are_judged_and_claims_stay_stable(sims):
    """Ruling R19a: the two snapshots before a final run tier G too, so protocol 8's gate
    judges each history snapshot from its own runs (gentle tier A cells and the tier G
    width-12 straights; no gate is given for them) and a claim that recurs is reported
    stable, not 'unstable'. ``history_gate_problems`` is then empty for that agent. The tier
    G runs change only the gate: every history verdict equals the one with the gate given."""
    keys, g_rows = _tier_g_keys(KEYS)
    inp = _inputs(sims)
    inp.keys = keys
    lawful = sims["lawful"][0]
    inp.history["G1"] = [_with_tier_g(lawful, g_rows, "G1", t) for t in (5, 6)]
    assert 5 not in inp.gates["G1"] and 6 not in inp.gates["G1"]
    for h in inp.history["G1"]:
        gate = A.checkpoint_gate(A.pair_values(h, keys, inp.regimes, B))
        assert gate["evaluable"] and gate["passed"]
        assert gate["completion_null"] == 1.0 and gate["completion_gentle"] == 1.0
    assert A.history_judgeable(inp, "G1") == [True, True]
    assert [p.split(":")[0] for p in A.history_gate_problems(inp)] == ["G2", "G3"]
    # complete inputs (every generalist final with both history snapshots and their tier G
    # runs): nothing for the confirmatory stage to refuse on
    full = _inputs(sims)
    full.keys = keys
    for a in ("G1", "G2", "G3"):
        full.history[a] = [_with_tier_g(lawful, g_rows, a, t) for t in (5, 6)]
    assert A.history_gate_problems(full) == []
    ds = {(d.prediction, d.agent): d for d in A.decisions(inp, True)}
    p1a = ds[("P1a", "G1")]
    assert (p1a.verdict, p1a.stable) == ("confirm", True)
    assert p1a.details["history_verdicts"] == ["confirm", "confirm"]
    assert p1a.details["history_gate_judgeable"] == [True, True]
    assert "final_verdict" not in p1a.details
    t = A._decision_rows(list(ds.values()))
    i = [k for k in range(A._n(t)) if t["prediction"][k] == "P1a" and t["agent"][k] == "G1"]
    assert (t["verdict"][i[0]], t["stable"][i[0]]) == ("confirm", "yes")
    # the same history with its gate given (no tier G runs): identical verdicts
    ref = _inputs(sims)
    ref.history["G1"] = [{**lawful, "agent": np.full(A._n(lawful), "G1"),
                          "timesteps": np.full(A._n(lawful), t)} for t in (5, 6)]
    ref.gates["G1"].update({5: True, 6: True})
    rd = {(d.prediction, d.agent): d for d in A.decisions(ref, True)}
    for p in A.CONFIRMATORY_G:
        assert ds[(p, "G1")].verdict == rd[(p, "G1")].verdict, p
        assert ds[(p, "G1")].details.get("history_verdicts") == \
            rd[(p, "G1")].details.get("history_verdicts"), p
    # the gate still bites: tier G straights that do not complete fail it (AD24)
    inp2 = _inputs(sims)
    inp2.keys = keys
    inp2.history["G1"] = [_with_tier_g(lawful, g_rows, "G1", t, completed=t == 6)
                          for t in (5, 6)]
    g5 = A.checkpoint_gate(A.pair_values(inp2.history["G1"][0], keys, inp2.regimes, B))
    assert g5["evaluable"] and not g5["passed"] and g5["completion_null"] == 0.0
    d2 = {(d.prediction, d.agent): d for d in A.decisions(inp2, True)}[("P1a", "G1")]
    assert d2.details["history_verdicts"] == ["not testable", "confirm"]
    assert (d2.verdict, d2.stable) == ("confirm", True)


def test_R19_review_p2_claim_recurs_at_history_snapshots_judged_from_their_tier_g_runs(sims):
    """R19 review finding 1 (gate side): P2 is a claim at G2's final (late onset), and the
    two history snapshots, judged by protocol 8's gate from their own tier G runs (no gate
    given), reach it too, so it is reported stable. The same history without its tier G
    runs (the last-3 rows before R19a) is 'not testable' and P2 'unstable'; with the gate
    given instead, every G2 verdict equals the one judged from the tier G runs. The
    run-level side (the history snapshots' own null runs, so d_resp and Δd_resp are defined
    there) is in test_interp_r19.py."""
    keys, g_rows = _tier_g_keys(KEYS)
    late = sims["late_onset"][0]
    # The simulated late-onset driver also crashes the gentle A1 cells (R 87.3 from 310 km/h)
    # and so fails the gate; this history completes them. P2 reads the A1 tail and
    # below-support pairs only, so its verdict does not depend on them.
    gentle = np.isin(late["pair"], keys["pair"][np.isclose(A._fl(keys, "R_nom"), A.GENTLE_R)])
    assert gentle.any() and not late["completed"][gentle].all()
    hist = _set(_set(_set(late, "crashed", gentle, False), "completed", gentle, True),
                "crash_type", gentle, "")
    inp = _inputs(sims)
    inp.keys = keys
    inp.history["G2"] = [_with_tier_g(hist, g_rows, "G2", t) for t in (5, 6)]
    assert 5 not in inp.gates["G2"] and 6 not in inp.gates["G2"]
    assert A.history_judgeable(inp, "G2") == [True, True]
    ds = {(d.prediction, d.agent): d for d in A.decisions(inp, True)}
    p2 = ds[("P2", "G2")]
    assert (p2.verdict, p2.stable) == ("confirm", True)
    assert p2.details["history_verdicts"] == ["confirm", "confirm"]
    assert p2.details["history_gate_judgeable"] == [True, True]

    def plain(t):
        return {**hist, "agent": np.full(A._n(hist), "G2"), "timesteps": np.full(A._n(hist), t)}

    # before R19a: the history has no tier G runs and the gate cannot judge it
    pre = _inputs(sims)
    pre.history["G2"] = [plain(t) for t in (5, 6)]
    d0 = {(d.prediction, d.agent): d for d in A.decisions(pre, True)}[("P2", "G2")]
    assert (d0.verdict, d0.details["final_verdict"]) == ("unstable", "confirm")
    assert d0.details["history_verdicts"] == ["not testable", "not testable"]
    # the gate given for the same history: the verdicts judged from the tier G runs
    ref = _inputs(sims)
    ref.history["G2"] = [plain(t) for t in (5, 6)]
    ref.gates["G2"].update({5: True, 6: True})
    rd = {(d.prediction, d.agent): d for d in A.decisions(ref, True)}
    for p in A.CONFIRMATORY_G:
        assert ds[(p, "G2")].verdict == rd[(p, "G2")].verdict, p
        assert ds[(p, "G2")].details.get("history_verdicts") == \
            rd[(p, "G2")].details.get("history_verdicts"), p


def test_R19c_stability_compares_decision_classes(sims):
    """Ruling R19c: AD21's 2-of-last-3 rule compares decision classes exactly as R18's
    variant rule does ('cautious but lawful' is a P1a refutation), and the final's verdict
    label is the one reported. Through ``_generalist_decisions`` with a history that the
    gate judges from its own tier G runs (R19a)."""
    keys, g_rows = _tier_g_keys(KEYS)
    inp = _inputs(sims)
    inp.keys = keys
    lawful = sims["lawful"][0]
    inp.history["G1"] = [_with_tier_g(lawful, g_rows, "G1", t) for t in (5, 6)]

    def rule_of(by_ts):
        def rule(pairs, runs):
            return A._d("P1a", "G1", by_ts[int(A._it(runs, "timesteps", 0)[0])])
        return {"P1a": rule}

    # the final's 'cautious but lawful' recurs as the history's 'refute': stable, label kept
    d = A._generalist_decisions(inp, "G1", rule_of(
        {0: "cautious but lawful", 5: "refute", 6: "not testable"}))[0]
    assert (d.verdict, d.stable) == ("cautious but lawful", True)
    assert d.details["history_verdicts"] == ["refute", "not testable"]
    assert "final_verdict" not in d.details
    # and the other way round
    d = A._generalist_decisions(inp, "G1", rule_of(
        {0: "refute", 5: "inconclusive", 6: "cautious but lawful"}))[0]
    assert (d.verdict, d.stable) == ("refute", True)
    # a different decision class does not recur: 'unstable', the final verdict in details
    d = A._generalist_decisions(inp, "G1", rule_of(
        {0: "cautious but lawful", 5: "confirm", 6: "not testable"}))[0]
    assert (d.verdict, d.stable) == ("unstable", False)
    assert d.details["final_verdict"] == "cautious but lawful"
    # the rule itself: classes on both sides, exact labels otherwise
    for final, hist, stable in (("cautious but lawful", ["refute", "confirm"], True),
                                ("refute", ["cautious but lawful", "not testable"], True),
                                ("cautious but lawful", ["cautious but lawful"], True),
                                ("confirm", ["cautious but lawful", "refute"], False),
                                ("inconclusive", ["refute", "confirm"], False)):
        s = A._stabilise(A._d("P1a", "G1", final), hist)
        assert s.stable is stable, (final, hist)
        assert s.verdict == (final if stable or final not in A.CLAIMS else "unstable")
    assert A._stabilise(A._d("P1a", "G1", "refute"), []).details["stability"] == "n/a"


def test_R18_review_decision_table_serialises_before_writing(sims, tmp_path, monkeypatch):
    """R18 review finding 5: every output of the confirmatory run is serialised before the
    first is written, and the one-run markers (replication.json, then decisions.npz) are
    written last; a failure while building the replication family leaves no file, so the
    one-shot run can be repeated."""
    inp = _with_replicates(sims, _inputs(sims))
    real = A.seed_spread
    monkeypatch.setattr(A, "seed_spread", lambda *a, **k: {"agent": np.array([object()])})
    with pytest.raises(TypeError, match="dtype object"):
        A.decision_table(inp, True, tmp_path)
    assert list(tmp_path.iterdir()) == []                          # nothing written
    monkeypatch.setattr(A, "seed_spread", real)
    order: list[str] = []
    real_write = A.write_bytes
    monkeypatch.setattr(A, "write_bytes", lambda p, d: (order.append(p.name), real_write(p, d)))
    A.decision_table(inp, True, tmp_path)                         # not refused: no marker
    assert order[-2:] == ["replication.json", "decisions.npz"]
    assert set(order) == {"decisions.json", "replication_decisions.npz",
                          "replication_variants.npz", "replication_spread.npz",
                          "replication.json", "decisions.npz"}
    with pytest.raises(FileExistsError):
        A.decision_table(inp, True, tmp_path)


def test_R12_p1b_reports_the_history_snapshots_ra_fallback(sims):
    """R12 re-anchors the below-support A1 cells at P0, the finals, only. P1b's stability
    check re-reads P1b on the two snapshots before the final; a history snapshot whose A1
    runs fell back to the nominal key (no P0 keys for it) is reported per snapshot."""
    inp = _inputs(sims)
    a1 = np.asarray(KEYS["cell"])[np.asarray(KEYS["approach"]).astype(str) == "A1"]
    h0, h1 = inp.history["G1"]
    h0 = {**h0, "ra_key": ~np.isin(h0["cell"], a1)}
    inp.history["G1"] = [h0, dict(h1)]
    d = next(x for x in A.decisions(inp) if x.prediction == "P1b" and x.agent == "G1")
    assert d.details["ra_fallback_pairs"] == 0
    fb = d.details["history_ra_fallback_pairs"]
    assert len(fb) == 2 and fb[0] > 0 and fb[1] == 0
    d3 = next(x for x in A.decisions(inp) if x.prediction == "P1b" and x.agent == "G3")
    assert "history_ra_fallback_pairs" not in d3.details          # G3 has no history here


def test_decision_table_all_and_clip_sensitivity(sims, tmp_path):
    inp = _inputs(sims)
    # ego-group clipping on half the deterministic bulk A2 runs of G1, which overspeed: with
    # them P1a is refuted, without them it is confirmed, so that version is primary (9.4)
    r = {k: np.array(v, copy=True) for k, v in inp.runs["G1"].items()}
    pk = A._join_keys(r, KEYS, ("approach", "R_nom", "braking"))
    hit = (pk["approach"].astype(str) == "A2") & (pk["R_nom"].astype(float) >= 20.0) & \
        (r["mode"] == "det") & (r["phase"] < 2)
    r["clip_frac_ego"] = np.where(hit, 0.1, 0.0)
    r["e_ve"] = np.where(hit, 0.2, r["e_ve"])
    inp.runs["G1"] = r
    t = A.decision_table(inp, False, tmp_path)
    m = (t["prediction"] == "P1a") & (t["agent"] == "G1")
    assert m.sum() == 1
    d = json.loads(t["details"][m][0])
    assert d["with_clipped_runs"] == "refute" and t["verdict"][m][0] == "confirm"
    assert "9.4" in d["primary_version"]
    preds = set(t["prediction"])
    assert {"P1c", "P1c-approach", "P2b", "P2d", "P3-beta-lt120", "P5", "P8-bulk",
            "P8-F"} <= preds
    p5 = (t["prediction"] == "P5") & (t["agent"] == "G3")
    assert t["verdict"][p5][0] == "confirm"
    assert (tmp_path / "decisions_all.npz").exists()
    assert (tmp_path / "G1_r50_nat.npz").exists() and (tmp_path / "G2_pairs.npz").exists()
    assert "P3-ratio" in preds
    A.decision_table(inp, False, tmp_path)                  # the full table may be rewritten


def test_family_of():
    assert A.family_of("P1a", "G3") == "primary" and A.family_of("P1a", "G1") == "secondary"
    assert A.family_of("P4", "S1") == "secondary" and A.family_of("P4", "S3") == "primary"
    assert A.family_of("P6b", "G1") == "exploratory" and A.family_of("P5", "G2") == "control"
    assert A.family_of("P8-tail", "G2") == "secondary"
    assert A.family_of("P6a", "G1") == "secondary" and A.family_of("P6a", "S2") == "exploratory"


def test_select_keeps_table_shape(sims):
    _, p = sims["lawful"]
    s = select(p, p["braking"])
    assert set(s) == set(A.PAIR_COLUMNS)


# ------------------------------------------------------------------------- descriptive (9.1, 9.3)


def test_viability_targets_and_budget():
    keys = _pair_keys()
    rows = []
    for c in (0, 1):
        for k in range(4):
            crash = (c == 0 and k in (1, 2)) or (c == 1 and k in (0, 3))
            rows.append(dict(agent="G1", timesteps=5, cell=c, pair=7, phase=k, sample=-1,
                             mode="det", crashed=crash, s_crash=1035.0 if crash else np.nan,
                             s_b=900.0))
    runs = A._to_table(rows, sorted(rows[0]))
    vt = A.viability_targets(runs, keys)
    # cell 0: first crashed phase 1; cell 1: its u = 0 run crashed
    assert vt["cell"].tolist() == [0, 1] and vt["phase"].tolist() == [1, 0]
    # s_b 900, s_c0 - 50, s_c0 1000 and s_a0 1010 all precede the crash at 1035
    assert vt["n_solves"].tolist() == [4, 4]
    many = A.concat_tables([{**runs, "pair": runs["pair"] + 100 * j, "cell": runs["cell"] + 2 * j}
                            for j in range(40)])
    kk = A.concat_tables([{**keys, "cell": keys["cell"] + 2 * j, "pair": keys["pair"] + 100 * j}
                          for j in range(40)])
    vt = A.viability_targets(many, kk, cap=200, subset=10, rng=np.random.default_rng(0))
    assert np.unique(vt["pair"]).size == 10 and vt["n_solves"].sum() == 80


def test_cell_rates_r50_table_and_slack(sims):
    _, lawful = sims["lawful"]
    p = {k: np.array(v, copy=True) for k, v in lawful.items()}
    tight = p["R_seg"] < 12.0
    p["nat_k"] = np.where(tight, p["nat_n"], 0)
    cr = A.cell_rates(p, "nat")
    assert A._n(cr) == A._n(p)
    i = int(np.flatnonzero(tight)[0])
    assert cr["p"][i] == 1.0 and cr["hi"][i] == 1.0 and cr["lo"][i] == pytest.approx(
        (0.025) ** (1 / p["nat_n"][i]))
    rt = A.r50_table(p, "nat")
    a1 = (rt["v0"] == round(kph(310.0), 3)) & (rt["D"] == 600.0) & (rt["theta_deg"] == 90.0)
    assert rt["position"][a1][0] == "inside" and rt["monotone"][a1][0]
    assert (rt["bracket_lo"][a1][0], rt["bracket_hi"][a1][0]) == (11.7, 13.1)
    assert 11.7 < rt["firth_R50"][a1][0] < 13.1
    sl = A.slack(lawful, KEYS, TABLES)
    m = (lawful["approach"] == "A1") & (lawful["R_nom"] == 22.9) & (lawful["theta_deg"] == 90.0)
    ve = KEYS["v_e_star"][(KEYS["approach"] == "A1") & (KEYS["R_nom"] == 22.9)
                          & (KEYS["theta_deg"] == 90.0)][0]
    assert sl[m][0] == pytest.approx((600 - TABLES.d_b_phys(kph(310), ve)) / kph(310))


def test_support_correlations_and_covariates(sims):
    _, lawful = sims["lawful"]
    p = {k: np.array(v, copy=True) for k, v in lawful.items()}
    p["log_support"] = np.log(p["R_seg"])
    p["e_ve"] = np.where(p["uses_ra"], p["e_ve"], -0.2 / p["R_seg"])
    p["e_ve_ra"] = np.where(p["uses_ra"], -0.2 / p["R_seg"], p["e_ve_ra"])
    sc = A.support_correlations(p)
    row = (sc["covariate"] == "log_support") & (sc["measure"] == "abs_e_ve")
    assert sc["rho"][row][0] < -0.95                         # larger error, lower support
    assert np.isnan(sc["rho"][(sc["covariate"] == "log_support_tightfast")][0])
    keys = make_keys(r_values=(22.9,), tier_b=False)
    keys = A.concat_tables([keys, {**{k: v[:1] for k, v in keys.items()}, "cell": np.array([99]),
                                   "tiers": np.array(["G"]), "width": np.array([12.0])}])
    runs = A._to_table([dict(cell=99, mode="det", v_peak=70.0, v_max=85.0),
                        dict(cell=99, mode="det", v_peak=70.0, v_max=86.5),
                        dict(cell=0, mode="det", v_peak=90.0, v_max=90.0)],
                       ("cell", "mode", "v_peak", "v_max"))
    assert A.v_top(runs, keys) == 86.5        # the top speed over the whole null run
    fin = {k: np.array(v, copy=True) for k, v in lawful.items()}
    assert A.reanchor_immaterial([fin])["immaterial"]       # RA = nominal in the simulation
    a1 = fin["approach"] == "A1"
    fin["v_e_star_ra"] = np.where(a1, fin["v_e_star"] + kph(3.0), fin["v_e_star_ra"])
    assert not A.reanchor_immaterial([fin])["immaterial"]


def test_p3_ratio(sims):
    d = A.p3_ratio(sims["lawful"][0], KEYS, TABLES)
    assert d.verdict == "descriptive" and abs(d.details["ratio_slope_vs_log_vb"]) < 0.05
    assert d.details["b_without_310"] == pytest.approx(1.0, abs=0.05)
    d = A.p3_ratio(sims["time_to_corner"][0], KEYS, TABLES)
    assert d.details["ratio_slope_vs_log_vb"] < -0.3       # time-to-corner: ratio falls with v_b


# ------------------------------------------------------------------------- input-table builders


def _ladder_runs():
    """One unit: real rung cells 0, 1 (4 det phases each), isolated cell 2, twin cell 3; the
    real rung crashes by overspeed in 3 of 8 det runs and 5 of 16 native runs."""
    rows, keys = [], []
    for c, ver in ((0, "real"), (1, "real"), (2, "isolated"), (3, "twin")):
        keys.append(dict(cell=c, s_c0=1000.0))
        for k in range(4):
            crash = ver == "real" and (c, k) in ((0, 0), (0, 1), (1, 3))
            rows.append(dict(agent="G2", timesteps=5, cell=c, mode="det", phase=k, crashed=crash,
                             crash_type="overspeed" if crash else "", s_crash=1020.0 if crash else np.nan,
                             e_ve=0.05 * (k + 1), v_b=80.0 if ver == "real" else 60.0))
        for j in range(4 if ver == "real" else 0):
            for s in range(2):
                crash = ver == "real" and c == 0 and j < 2 or (c == 1 and j == 0 and s == 0)
                rows.append(dict(agent="G2", timesteps=5, cell=c, mode="nat", phase=j,
                                 crashed=crash, crash_type="", s_crash=np.nan, e_ve=np.nan,
                                 v_b=np.nan))
    cols = sorted({k for r in rows for k in r})
    return A._to_table(rows, cols), A._to_table(keys, ("cell", "s_c0"))


def test_ladder_table_counts_and_overspeed_median():
    runs, keys = _ladder_runs()
    units = [dict(agent="G2", unit="G2/Norisring/910", set="tail",
                  cells={"real": [0, 1], "isolated": [2], "twin": [3]})]
    t = A.ladder_table(runs, keys, units, {"G2/Norisring/910": True})
    real = t["version"] == "real"
    assert t["det_crashes"][real][0] == 3 and t["det_n"][real][0] == 8
    assert t["det_overspeed"][real][0] == 3
    assert t["nat_n"][real][0] == 16 and t["nat_k"][real][0] == 5
    # e_ve of the 8 real runs: [0.05, 0.1, 0.15, 0.15, 0.2, inf, inf, inf] (overspeed = +inf)
    assert t["e_ve"][real][0] == pytest.approx(0.175)
    assert bool(t["fidelity_ok"][real][0]) and t["set"][0] == "tail"
    iso = t["version"] == "isolated"
    assert t["det_crashes"][iso][0] == 0 and t["e_ve"][iso][0] == pytest.approx(0.125)
    u = A.p8_units(t)
    assert u["reproduced"][0] and not u["single_corner"][0]


def test_twin_table_uses_the_real_rung_v_b_for_the_approach(sims):
    runs, keys = _ladder_runs()
    _, lawful = sims["lawful"]
    corners = [dict(corner="E1/Norisring/850", source="E1", R_seg=13.0, theta_deg=90.0,
                    cells={"real": [0, 1], "isolated": [2], "twin": [3]}, length_shortfall=26.0)]
    t = A.twin_table(runs, keys, lawful, corners)
    assert A._n(t) == 1 and t["source"][0] == "E1"
    assert t["e_ve_twin"][0] == pytest.approx(0.125)
    assert t["length_shortfall"][0] == 26.0                    # R4: carried as a covariate
    # real v_b 80 m/s >= 250 km/h -> the A1 curve
    assert t["e_ve_curve"][0] == pytest.approx(A.tier_a_curve_at(lawful, 13.0, 90.0, "A1"))


def test_density_table_means_over_the_pair():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(400, 5)) + np.array([3.0, 90.0, 30.0, 60.0, 4.0])
    cs = {"cell": np.arange(4), "R_seg": np.array([20.0, 20.0, 9.0, 9.0]),
          "theta_deg": np.array([90.0, -90.0, 90.0, -90.0]), "l_r": np.full(4, 30.0),
          "L_c": np.full(4, 60.0), "v_peak": np.full(4, 55.0)}
    cells = [SimpleNamespace(cell_id=i, pair_id=i // 2) for i in range(4)]
    t = A.density_table(cs, cells, {"G1": "base"}, {"base": train}, {"base": train[:50]},
                        np.ones(5))
    assert A._n(t) == 2 and list(t["pair"]) == [0, 1]
    assert np.all(np.isfinite(t["log_support"])) and t["log_support"][0] > t["log_support"][1]
    assert np.all(np.isfinite(t["log_support_tightfast"]))


def test_alias_table_medians_per_geometry():
    keys = {"cell": np.arange(4), "geometry": np.array(["g9", "g9", "g28", "g28"]),
            "R_nom": np.array([9.4, 9.4, 28.6, 28.6]), "theta_deg": np.full(4, 90.0)}
    rows = [dict(agent=a, cell=c, A_alias=v) for a, c, v in (
        ("G1", 0, 0.2), ("G1", 1, 0.4), ("G1", 2, 0.1), ("Q", 0, 0.1), ("Qp", 0, 0.05),
        ("Qp", 1, 0.07))]
    t = A.alias_table(A._to_table(rows, ("agent", "cell", "A_alias")), keys)
    g9 = t["geometry"] == "g9"
    assert t["a_agent"][g9][0] == pytest.approx(0.3) and bool(t["short"][g9][0])
    assert t["a_q"][g9][0] == pytest.approx(0.1) and t["a_qp"][g9][0] == pytest.approx(0.06)
    assert not t["short"][t["geometry"] == "g28"][0]
