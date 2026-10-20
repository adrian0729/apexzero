"""Dry-run rulings R20-R28 (protocol 12, 2026-10-20) in the analysis, on simulated data only.

R25: the key table carries the key's onset time and forced flag, the dry-run simulator flags
forced onsets as real runs are flagged, P3 is calibrated to b* (the key's slope on the agent's
included cells, key-forced cells left out), AD15's folds are mirror-pair conditions, and the
separation check is against b*. R24 / R26: the power table's saturation and hybrid rows. R20:
T's regret over |J*| and the T64 / T32 verdicts. R28: c1's crashes typed by the viability
solves (on the cells with a verified u = 0 key). R24's c2 planted-law check is
``controls.c2_planted_law`` (tests/test_interp_dryrun_controls.py).

Two synthetic key designs are used. ``KEYS`` (test_interp_analysis) has d_b* = d_b^phys, so
b* = 1 and no key onset is forced. ``SLOW`` mimics what the dry run found on the real keys:
measured d_b* grows more slowly than d_b^phys (b* = 0.85 by construction; the real keys give
0.854), the A2 cells of tight corners start within 0.5 s of the key's onset (forced, with d_b*
inflated as the real forced A2 cells are, braking partly while crossing the road), and the
mirror members of an A2 pair have D 0.1 m apart (the real layout's 61.9 / 62.0 m)."""

from __future__ import annotations

import importlib
import json
import math

import numpy as np
import pytest

from apexzero.interp import analysis as A
from apexzero.interp import measures
from apexzero.interp.constants import R_BRAKING, TIER_B_D, TIER_B_R, TIER_B_V0_KPH, kph

T = importlib.import_module("test_interp_analysis")       # the synthetic design and tables
TABLES, B, KEYS, DESIGN = T.TABLES, T.B, T.KEYS, T.DESIGN
B_KEY = 0.85            # SLOW's key slope b*
FORCED_R = 14.6         # SLOW: A2 cells of corners up to this radius start at the key's onset
INFLATE = 1.6           # ... and their key brakes over 1.6 x the distance
D_OFFSET = 0.1          # m: the right-hand member's D at A2


def slow_keys(b_key: float = B_KEY) -> dict:
    """KEY_COLUMNS shaped like test_interp_analysis.make_keys (tier A at A1 and A2, tier B),
    with d_b* = 100 (d_b^phys / 100)^b_key, forced key onsets on the tight A2 cells and A2
    mirror members whose D differ by 0.1 m. t_b* is the time to the onset cruising at
    v_b* = v0."""
    conds: dict[tuple, set] = {}
    for R in R_BRAKING:
        for th in (90.0, 180.0):
            if th == 180.0 and R > 55.9:
                continue
            conds.setdefault((R, th, kph(310.0), 600.0), set()).add("A1")
            conds.setdefault((R, th, T._v_typ(R), -1.0), set()).add("A2")
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
        d_b = 100.0 * (TABLES.d_b_phys(v0, v_e) / 100.0) ** b_key
        L_c = math.radians(th) * R + 20.0
        geo = f"syn/R{R:g}/T{th:g}/W12/typical"
        forced_cell = approach == "A2" and R <= FORCED_R
        for direction in (1, -1):
            db, Dc = d_b, D
            if approach == "A2":
                db = d_b * INFLATE if forced_cell else d_b
                Dc = db + (0.3 if forced_cell else 2.0) * v0 + (0.0 if direction > 0
                                                                else D_OFFSET)
            t_b = max(Dc - db, 0.0) / v0
            rows.append(dict(
                cell=cell, stim=cell, pair=pair, name=f"{geo}/{'L' if direction > 0 else 'R'}",
                geometry=geo, tiers=tiers_s, approach=approach, v0=v0, D=Dc, W=400, n_phases=4,
                direction=direction, s_c0=1000.0, s_a0=1010.0, s_apex=1000.0 + L_c / 2,
                s_c1=1000.0 + L_c, R_nom=R, theta_deg=th, width=12.0, ramp_kind="typical",
                no_cap=False, R_seg=R, R_eff=R, corner=-1, condition="", swap_j=-1,
                key_status="verified", J_star=100.0, v_e_star=v_e, v_min_star=v_min,
                d_b_star=db, v_b_star=v0, has_brake_star=True, tau_vmin=kph(0.5),
                t_b_star=t_b, forced_star=bool(t_b <= 0.5),
                multi_basin=False, lock_limited=False, slip_limited=False,
                braking=bool(v_min < v0 - kph(0.5)), v_e_star_bo=0.97 * v_e,
                v_min_star_bo=0.97 * v_min, v_e_socp=v_e, v_min_socp=v_min, d_b_socp=db,
                v_b_socp=v0, braking_socp=True, v_min_qss=0.9 * v_min))
            cell += 1
    return A._to_table(rows, A.KEY_COLUMNS)


SLOW = slow_keys()


def _fit(kind: str, keys: dict, seed: int = 0, phase_noise: float = 1.0, **kw):
    runs = A.simulate_runs(kind, keys, A._dry_rng(70, seed), phase_noise, n_native=0,
                           n_matched=0, **kw)
    res = A.p3_fit(runs, keys, TABLES)
    return runs, res, A.p3(res, A.brake_share(runs, keys))


def _key_slope(keys: dict, mask: np.ndarray) -> float:
    x = np.log(A._dphys(TABLES, A._fl(keys, "v_b_star")[mask], A._fl(keys, "v_e_star")[mask]))
    return A._fe_slope(x, np.log(A._fl(keys, "d_b_star")[mask]), A._st(keys, "geometry")[mask])[0]


# ------------------------------------------------------------------------- R25: key table, simulator


class _OnsetKeys(T.FakeKeys):
    """FakeKeys whose nominal key at cell 0 brakes 0.3 s after its start (forced) and at
    cell 1 after 2.0 s."""

    def nominal(self, cell, phase):
        v = super().nominal(cell, phase)
        if v is None:
            return None
        forced = cell.cell_id == 0
        m = measures.ResponseMetrics(v_e=30.0, v_min=25.0, d_b=100.0, v_b=60.0, has_brake=True,
                                     t_b=0.3 if forced else 2.0, forced_onset=forced)
        return v._replace(metrics=m)


def test_R25_key_table_records_the_key_onset_time_and_forced_flag():
    bank, cells = T._bank_cells()
    kt = A.key_table(cells, bank, _OnsetKeys())
    assert {"t_b_star", "forced_star"} <= set(kt) <= set(A.KEY_COLUMNS)
    assert kt["t_b_star"][:2].tolist() == [0.3, 2.0]
    assert kt["forced_star"].tolist() == [True, False, False]   # tier G: no key, not forced
    assert np.isnan(kt["t_b_star"][2])


def test_R25_the_key_onsets_are_required():
    """Without t_b_star / forced_star the simulator would silently never force an onset (the
    dry-run bug) and b* could not leave the key's forced onsets out: both refuse."""
    old = {k: v for k, v in SLOW.items() if k not in ("t_b_star", "forced_star")}
    with pytest.raises(ValueError, match="R25"):
        A.simulate_runs("lawful", old, A._dry_rng(1))
    runs = A.simulate_runs("lawful", SLOW, A._dry_rng(1), n_native=0, n_matched=0)
    with pytest.raises(ValueError, match="R25"):
        A.p3_fit(runs, old, TABLES)
    with pytest.raises(ValueError, match="R25"):
        A.p3_separation(old, TABLES)


def test_R25_sim_forced_uses_the_key_flag_or_the_agents_own_onset():
    r = dict(t_b_star=0.6, v_b_star=50.0, d_b_star=100.0, forced_star=False)
    assert A._sim_forced(r, 100.0) is False                  # at the key's onset: its flag
    assert A._sim_forced({**r, "forced_star": True}, 100.0) is True
    assert A._sim_forced(r, 85.0) is False                   # 15 m later: t_b = 0.9 s
    assert A._sim_forced(r, 104.0) is False                  # 4 m earlier: t_b = 0.52 s
    assert A._sim_forced(r, 110.0) is True                   # 10 m earlier: t_b = 0.4 s
    assert A._sim_forced(r, 200.0) is True                   # before the start: forced
    assert A._sim_forced({**r, "t_b_star": np.nan}, 85.0) is False   # no time: the key's flag


def test_R25_simulated_runs_carry_forced_onsets():
    """Lawful runs are forced exactly where the key's onset is; the late-onset and
    time-to-corner agents use their own onset time; stochastic runs carry no onset."""
    kidx = {int(c): i for i, c in enumerate(SLOW["cell"])}
    assert SLOW["forced_star"].sum() == 32                   # the tight A2 cells, both members

    def key_col(runs, col):
        return np.asarray(SLOW[col])[[kidx[int(c)] for c in runs["cell"]]]

    lawful = A.simulate_runs("lawful", SLOW, A._dry_rng(2))
    det = lawful["mode"] == "det"
    assert lawful["forced_onset"][det].tolist() == key_col(lawful, "forced_star")[det].tolist()
    assert lawful["forced_onset"][det].sum() == 32 * 4
    assert not lawful["forced_onset"][~det].any()
    # late onset: 15 m late at A1 (never forced there, D = 600 m), the key's flag elsewhere
    late = A.simulate_runs("late_onset", SLOW, A._dry_rng(3), n_native=0, n_matched=0)
    a1 = key_col(late, "approach") == "A1"
    t_own = key_col(late, "t_b_star") + 15.0 / key_col(late, "v_b_star")
    assert not late["forced_onset"][a1].any() and (t_own[a1] > 0.5).all()
    assert late["forced_onset"][~a1].tolist() == key_col(late, "forced_star")[~a1].tolist()
    # time to corner: its own onset, d_b = c clip(6 v_b, 150, 500), not the key's flag
    ttc = A.simulate_runs("time_to_corner", SLOW, A._dry_rng(4), 0.0, n_native=0, n_matched=0)
    c = A._ttc_constant(A._key_rows(SLOW))
    vb, db, tb = (key_col(ttc, k) for k in ("v_b_star", "d_b_star", "t_b_star"))
    d_ttc = c * np.clip(6.0 * vb, 150.0, 500.0)
    np.testing.assert_allclose(ttc["d_b"], d_ttc)
    assert ttc["forced_onset"].tolist() == (tb + (db - d_ttc) / vb <= 0.5 + 1e-12).tolist()
    assert ttc["forced_onset"].tolist() != key_col(ttc, "forced_star").tolist()


# ------------------------------------------------------------------------- R25: P3 against b*


def test_R25_lawful_confirms_against_b_star_where_the_old_band_missed():
    _, res, d = _fit("lawful", SLOW)
    assert res.b_star == pytest.approx(B_KEY, abs=1e-9)
    assert res.b == pytest.approx(B_KEY, abs=0.02) and res.winner == "M_phys"
    assert d.verdict == "confirm"
    assert d.details["b_star"] == res.b_star
    assert d.details["confirm_band"] == pytest.approx([B_KEY - 0.1, 1.1])
    assert d.details["loo_band"] == pytest.approx([B_KEY - 0.15, 1.15])
    inp = d.details["b_star_inputs"]
    assert inp["key"] == "u = 0 nominal" and inp["n_used"] == inp["n_cells"] > 100
    assert inp["n_forced_excluded"] == 0          # its own onsets are forced there: no cell
    # The pre-R25 band [0.9, 1.1] (b* = 1) excluded the key's own slope.
    old = A.P3Result(res.b, res.loo_b, res.rmse, res.winner, res.n_runs, agent="lawful",
                     b_star=1.0)
    assert A.p3(old, d.details["brake_share"]).verdict == "inconclusive"
    # On KEYS (d_b* = d_b^phys) b* = 1 and nothing changes.
    _, res1, d1 = _fit("lawful", KEYS)
    assert res1.b_star == pytest.approx(1.0) and d1.verdict == "confirm"


def test_R25_b_star_leaves_out_cells_where_the_key_onset_is_forced():
    """The time-to-corner agent brakes off the key's onset, so its own onsets at the key's
    forced cells are not forced and those cells are included; b* still leaves them out."""
    _, res, _ = _fit("time_to_corner", SLOW)
    inp = res.b_star_inputs
    assert inp["n_forced_excluded"] == 32 and inp["n_used"] == inp["n_cells"] - 32
    assert res.b_star == pytest.approx(B_KEY, abs=1e-9)
    every = np.ones(A._n(SLOW), bool)
    with_forced = _key_slope(SLOW, every)
    assert with_forced < B_KEY - 0.1                 # the forced cells would bias b* down
    cleared = {**SLOW, "forced_star": np.zeros(A._n(SLOW), bool)}
    assert A.p3_b_star(cleared, SLOW["cell"], TABLES)[0] == pytest.approx(with_forced)
    # b* is the key's slope on the agent's included cells only
    some = np.asarray(SLOW["approach"]) != "A2"
    b_some, inp_some = A.p3_b_star(SLOW, np.asarray(SLOW["cell"])[some], TABLES)
    assert inp_some["n_cells"] == int(some.sum()) and inp_some["n_forced_excluded"] == 0
    assert b_some == pytest.approx(_key_slope(SLOW, some))


def test_R25_b_star_counts_the_included_cells_without_a_key():
    """Review of R25 (agent-facing finding 1, a point for a ruling note): P3 keeps runs on
    cells whose key is 'missing' (only EXCL_ALL is dropped), while b* can only use verified
    keys. On the real design no keys-stage item solves tier B D = 250 (88 of its 96 cells
    'missing', 26% of tier A/B cells), so the agent's b and b* are on different cells. The
    counts are recorded with b* (n_no_key cells, n_runs_no_key of n_runs) so the report
    shows the mismatch; the cell sets themselves are unchanged pending the ruling."""
    runs, res, _ = _fit("lawful", SLOW)
    assert res.b_star_inputs["n_no_key"] == res.b_star_inputs["n_runs_no_key"] == 0
    assert res.b_star_inputs["n_runs"] == res.n_runs
    d250 = (np.char.find(np.asarray(SLOW["tiers"]).astype(str), "B") >= 0) & (
        np.asarray(SLOW["D"], float) == 250.0)
    missing = {**SLOW, "key_status": np.where(d250, "missing", SLOW["key_status"])}
    res_m = A.p3_fit(runs, missing, TABLES)
    inp = res_m.b_star_inputs
    on = np.isin(A._it(runs, "cell"), np.asarray(SLOW["cell"])[d250])
    cells_in = set(A._p3_data(runs, missing, TABLES)["cell"].tolist())
    want_cells = len(cells_in & set(np.asarray(SLOW["cell"])[d250].tolist()))
    assert inp["n_no_key"] == want_cells > 0
    assert 0 < inp["n_runs_no_key"] <= int(on.sum()) and inp["n_runs"] == res_m.n_runs
    assert res_m.n_runs == res.n_runs and res_m.b == pytest.approx(res.b)   # b keeps them
    assert inp["n_used"] == res.b_star_inputs["n_used"] - want_cells       # b* cannot


@pytest.mark.parametrize("keys", ["KEYS", "SLOW"])
def test_R25_time_to_corner_refutes_naming_M_idx(keys):
    kt = {"KEYS": KEYS, "SLOW": SLOW}[keys]
    _, res, d = _fit("time_to_corner", kt)
    assert d.verdict == "refute" and d.details["winner"] == "M_idx"
    assert res.rmse["M_idx"] <= 0.9 * res.rmse["M_phys"]
    assert not d.details["loo_band"][0] <= res.b <= d.details["loo_band"][1]


@pytest.mark.parametrize("phase_noise", [1.0, 3.0])
def test_R25_lawful_never_refutes_over_many_rng_streams(phase_noise):
    verdicts = [_fit("lawful", SLOW, seed, phase_noise)[2].verdict for seed in range(24)]
    assert "refute" not in verdicts
    assert verdicts.count("confirm") == len(verdicts)


def test_R25_rule_table_around_b_star():
    rmse_phys = {"M_phys": 0.03, "M_idx": 0.08, "M_fd": 0.14}
    rmse_idx = {"M_phys": 0.10, "M_idx": 0.08, "M_fd": 0.14}       # M_idx better by 20%

    def v(b, loo=None, rmse=rmse_phys, b_star=B_KEY):
        winner = min(rmse, key=rmse.get)
        r = A.P3Result(b, loo or {"g1": b, "g2": b}, rmse, winner, 100, agent="X",
                       b_star=b_star)
        d = A.p3(r, 0.9)
        return f"refute:{d.details['winner']}" if d.verdict == "refute" else d.verdict

    assert v(0.80) == "confirm"                       # inside [0.75, 1.1] (was inconclusive)
    assert v(0.76, {"g1": 0.71, "g2": 0.80}) == "confirm"
    assert v(0.80, {"g1": 0.69, "g2": 0.80}) == "inconclusive"     # a LOGO b below b* - 0.15
    assert v(0.74) == "inconclusive"                  # below b* - 0.1
    assert v(1.12) == "inconclusive"                  # the upper edges are unchanged
    assert v(0.78, rmse=rmse_idx) == "inconclusive"   # M_idx wins, b inside the refute band
    assert v(0.72, rmse=rmse_idx) == "inconclusive"   # flagged in R25: refuted under 0.85
    assert v(0.69, rmse=rmse_idx) == "refute:M_idx"
    assert v(1.2, rmse={"M_phys": 0.1, "M_idx": 0.2, "M_fd": 0.05}) == "refute:M_fd"
    assert v(0.69, rmse={"M_phys": 0.1, "M_idx": 0.095, "M_fd": 0.2}) == "inconclusive"
    assert v(0.80, b_star=float("nan")) == "not testable"


def test_R25_separation_check_is_against_b_star():
    sep = A.p3_separation(KEYS, TABLES)                         # b* = 1
    assert sep["passed"] and sep["b_star"] == pytest.approx(1.0)
    assert sep["separation_idx"] == pytest.approx(abs(sep["b_idx"] - sep["b_star"]))
    assert sep["n_forced_excluded"] == 0
    # SLOW: M_idx's slope is 0.65, 0.35 from 1 but only 0.20 from b* = 0.85, so P3 is demoted.
    sep = A.p3_separation(SLOW, TABLES)
    assert sep["b_star"] == pytest.approx(B_KEY, abs=1e-9)
    assert abs(sep["b_idx"] - 1) >= A.P3_SEPARATION > sep["separation_idx"]
    assert not sep["passed"] and sep["demote_p3"]
    assert sep["n_forced_excluded"] == 32 and sep["reference"] == "b* (R25)"
    # A key whose d_b follows the physics more closely separates again.
    sep = A.p3_separation(slow_keys(0.97), TABLES)
    assert sep["passed"] and sep["separation_idx"] >= A.P3_SEPARATION


def test_R25_dry_run_returns_every_planted_verdict_on_the_slow_key_design():
    out = A.dry_run(SLOW, TABLES, B, design=DESIGN)
    failed = [(k, p, v) for k, p, v, ok in zip(out["kind"], out["prediction"], out["verdict"],
                                               out["passed"], strict=True) if not ok]
    assert not failed, failed
    i = [k == "lawful" and p == "P3" for k, p in zip(out["kind"], out["prediction"],
                                                     strict=True)].index(True)
    det = json.loads(out["details"][i])
    assert det["b_star"] == pytest.approx(B_KEY, abs=1e-9) and det["b"] < 0.9


# ------------------------------------------------------------------------- AD15 folds


def test_AD15_mirror_members_share_a_fold():
    runs, _, _ = _fit("lawful", SLOW)
    folds = A.p3_folds(runs, SLOW)
    by_pair: dict[int, set] = {}
    for p, f in zip(runs["pair"], folds, strict=True):
        by_pair.setdefault(int(p), set()).add(str(f))
    assert all(len(f) == 1 for f in by_pair.values())
    # the cells' own (v0, D) split every A2 pair (D 0.1 m apart)
    a2 = np.asarray(SLOW["approach"]) == "A2"
    by_cell: dict[int, set] = {}
    for p, a, c in zip(SLOW["pair"][a2], SLOW["v0"][a2], SLOW["D"][a2], strict=True):
        by_cell.setdefault(int(p), set()).add(f"{a:.3f}/{c:.1f}")
    assert by_cell and all(len(v) == 2 for v in by_cell.values())
    assert all(len(by_pair[p]) == 1 for p in by_cell)
    # conditions shared across geometries (A1 and tier B) stay one fold
    a1 = np.isin(runs["pair"], np.asarray(SLOW["pair"])[np.asarray(SLOW["approach"]) == "A1"])
    assert set(folds[a1]) == {f"{kph(310.0):.3f}/600.0"}
    # a run whose pair has no key row keeps its own cell's (v0, D)
    odd = {**runs, "pair": np.where(np.arange(A._n(runs)) == 0, 10**6, runs["pair"])}
    i0 = int(np.flatnonzero(np.asarray(SLOW["cell"]) == runs["cell"][0])[0])
    assert A.p3_folds(odd, SLOW)[0] == f"{SLOW['v0'][i0]:.3f}/{SLOW['D'][i0]:.1f}"


def test_AD15_pair_folds_remove_the_mirror_leak():
    """One geometry, two conditions: the mirror pair X (L and R, D 61.9 / 62.0) sits 1 above
    the physics law, condition Y on it. Held out by cell, each X run is predicted from its
    mirror partner; held out by pair, from Y only."""
    keys = A._to_table([dict(cell=0, pair=0, v0=50.0, D=61.9), dict(cell=1, pair=0, v0=50.0,
                                                                    D=62.0),
                        dict(cell=2, pair=1, v0=80.0, D=300.0)], ("cell", "pair", "v0", "D"))
    runs = A._to_table([dict(cell=c, pair=p) for c, p in ((0, 0), (1, 0), (2, 1))],
                       ("cell", "pair"))
    y, f, g = np.array([1.0, 1.0, 0.0]), np.zeros(3), np.array(["geo"] * 3)
    by_pair = A.p3_folds(runs, keys)
    assert by_pair[0] == by_pair[1] != by_pair[2]
    by_cell = np.array(["50.000/61.9", "50.000/62.0", "80.000/300.0"])
    leak, _ = A._heldout_rmse(y, f, g, by_cell)
    clean, _ = A._heldout_rmse(y, f, g, by_pair)
    assert clean == pytest.approx(1.0) and leak < clean


# ------------------------------------------------------------------------- R24 / R26: power table


def test_R24_saturation_plant():
    plant = A._saturation_plant(12.0)
    row = dict(approach="A1", R_seg=12.0 / 1.21)
    assert plant(row)[0] == pytest.approx(0.1)        # above log(1 + b) only below q1 / 1.21
    assert plant({**row, "R_seg": 12.0}) == (0.0, 0.0)
    assert plant({**row, "R_seg": 3.0})[0] == pytest.approx(1.0)
    assert plant({**row, "approach": "B"}) == (0.0, 0.0)


def test_R24_R26_power_table_rows():
    rt = A.Regime(11.4, 20.2)
    pt = A.power_table(SLOW, {"G1": rt}, B, TABLES, steps=A.POWER_STEPS[:2])
    assert pt["test"].tolist() == ["P1a", "P1-tail", "P1b", "P2", "P1b-sat", "P3", "P3-hybrid"]
    assert set(pt) == set(A.POWER_COLUMNS)
    row = {t: i for i, t in enumerate(pt["test"])}
    i = row["P1b-sat"]
    det = json.loads(pt["details"][i])
    scanned = det["verdict_by_q1"]
    qs = [q for q, _ in scanned]
    assert qs == sorted(qs) and all(q < rt.q10 for q in qs) and qs[0] == A.POWER_Q1_GRID[0]
    assert np.isnan(pt["min_overspeed"][i])
    if np.isfinite(pt["min_q1"][i]):
        assert scanned[-1] == [pt["min_q1"][i], "confirm"]
        assert all(v != "confirm" for _, v in scanned[:-1])
    else:
        assert all(v != "confirm" for _, v in scanned)
    # the planted verdict is P1b's on a reader saturating at the agent's own q1
    runs = A.simulate_runs("planted", SLOW, A._dry_rng(100, 6), plant=A._saturation_plant(11.4),
                           agent="G1")
    own = A.p1b(A.pair_values(runs, SLOW, {"G1": rt}, B), B)
    assert pt["planted_verdict"][i] == own.verdict and det["q1_own"] == 11.4
    assert pt["lawful_verdict"][i] == pt["lawful_verdict"][row["P1b"]]
    for test in ("P1a", "P1-tail", "P1b", "P2", "P3", "P3-hybrid"):
        assert np.isnan(pt["min_q1"][row[test]])
    # P3 rows: b, b* and the lawful agent's numbers are recorded
    p3, hy = (json.loads(pt["details"][row[t]]) for t in ("P3", "P3-hybrid"))
    assert pt["planted_verdict"][row["P3"]] == "refute:M_idx"
    assert p3["planted"]["b_star"] == pytest.approx(B_KEY, abs=1e-9)
    assert p3["lawful"]["b"] == pytest.approx(B_KEY, abs=0.02)
    # the earlier-of driver lies between the pure time-to-corner rule and the lawful key
    assert p3["planted"]["b"] < hy["planted"]["b"] < p3["lawful"]["b"]


def test_R26_hybrid_brakes_at_the_earlier_of_the_key_and_the_time_to_corner_trigger():
    kw = dict(n_native=0, n_matched=0)
    hy = A.simulate_runs("time_to_corner_hybrid", SLOW, A._dry_rng(5), 0.0, **kw)
    law = A.simulate_runs("lawful", SLOW, A._dry_rng(5), 0.0, **kw)
    ttc = A.simulate_runs("time_to_corner", SLOW, A._dry_rng(5), 0.0, **kw)
    np.testing.assert_allclose(hy["d_b"], np.maximum(law["d_b"], ttc["d_b"]))
    np.testing.assert_allclose(hy["v_e"], law["v_e"])            # lawful entry speed
    earlier = ttc["d_b"] > law["d_b"]
    assert 0 < earlier.sum() < earlier.size
    assert hy["forced_onset"][~earlier].tolist() == law["forced_onset"][~earlier].tolist()
    assert hy["forced_onset"][earlier].tolist() == ttc["forced_onset"][earlier].tolist()


# ------------------------------------------------------------------------- R20: control T


def _tape_tables(crash: bool = True):
    keys = A._to_table([dict(cell=c, J_star=J, v_e_star=30.0, v_min_star=20.0, d_b_star=100.0,
                             tau_ve=0.1, tau_vmin=0.1, tau_db=0.5, key_status="verified")
                        for c, J in ((0, 100.0), (1, -50.0), (2, 100.0))],
                       ("cell", "J_star", "v_e_star", "v_min_star", "d_b_star", "tau_ve",
                        "tau_vmin", "tau_db", "key_status"))
    rows = [dict(cell=0, regret=1e-9, J=100.0 - 1e-9, crashed=False, completed=True),
            dict(cell=1, regret=1e-5, J=-50.0 - 1e-5, crashed=False, completed=True)]
    if crash:      # cell 130's pattern: J = -0.376 after a slip crash, regret 100.376
        rows.append(dict(cell=2, regret=100.376, J=-0.376, crashed=True, completed=False))
    for r in rows:
        crashed = r["crashed"]
        r.update(mode="det", phase=0, sample=-1, v_e=np.nan if crashed else 30.0,
                 v_min=np.nan if crashed else 20.0, d_b=np.nan if crashed else 100.0)
    return A._to_table(rows, ("cell", "mode", "phase", "sample", "regret", "J", "crashed",
                              "completed", "v_e", "v_min", "d_b")), keys


def test_R20_tape_regret_is_divided_by_J_star():
    runs, keys = _tape_tables()
    reg = A.tape_regret(runs, keys)
    assert reg["max_rel"] == pytest.approx(100.376 / 100.0)     # not 100.376 / 0.376 = 267
    assert reg["n_beyond"] == 1 and not reg["passed"] and reg["worst_cells"][0] == 2
    clean = A.tape_regret(*_tape_tables(crash=False))
    assert clean["passed"] and clean["max_rel"] == pytest.approx(2e-7)   # |J*| = 50


def test_R20_t64_blocks_on_everything_and_t32_on_the_measures_of_runs_that_reach_the_exit():
    runs, keys = _tape_tables()
    t64 = A.tape_check(runs, keys, "f64")
    assert not t64["passed"] and t64["blocking"] == ["termination", "regret", "errors"]
    assert t64["got"] == {"termination": "1 crashes", "regret": "max rel 1.00e+00",
                          "errors": "1 beyond tau"}
    t32 = A.tape_check(runs, keys, "f32")
    assert t32["passed"] and t32["blocking"] == ["errors"]
    assert t32["got"]["errors"] == "within_tau" and t32["got"]["termination"] == "1 crashes"
    assert t32["details"]["crash_cells"] == [2] and t32["details"]["errors"]["n"] == 2
    ok = A.tape_check(*_tape_tables(crash=False), "f64")
    assert ok["passed"] and ok["got"] == {"termination": "none", "regret": "zero",
                                          "errors": "within_tau"}
    with pytest.raises(ValueError):
        A.tape_check(runs, keys, "f16")


# ------------------------------------------------------------------------- R28 / R30: viability typing


def test_R30_no_return_resolved_mirrors_crash_typing():
    """Ruling R30's resolution rule: some tested state non-viable (s set, certified or not),
    or every tested state through s_a0 viable (the s_a0 target tested); else unresolved."""
    nr = measures.NoReturn
    ok, s = A.no_return_resolved(None, 1020.0)
    assert not ok and np.isnan(s)
    assert A.no_return_resolved(nr(1005.0, True, 2.0), 1020.0) == (True, 1005.0)
    assert A.no_return_resolved(nr(1005.0, False, 2.0)) == (True, 1005.0)   # R30: certified
    ok, s = A.no_return_resolved(nr(None, False, np.nan, 1030.0,                # decides nothing
                                    tested=("s_b", "s_c0", "s_a0")), 1020.0)
    assert ok and np.isnan(s)                      # viable through s_a0: resolved
    assert not A.no_return_resolved(nr(None, False, np.nan, 1000.0, tested=("s_b", "s_c0")))[0]
    assert not A.no_return_resolved(nr(None, False, np.nan, 1030.0))[0]       # no target named
    assert not A.no_return_resolved("NR", 1020.0)[0]
    cols = A.no_return_columns(nr(1005.0, False, 2.0, 990.0, target="s_a0", s_target=1000.0,
                                  cf_verified_at="s_a0", overruled=("s_b",),
                                  tested=("s_b", "s_c0", "s_a0")))
    nan_dv = cols.pop("nr_cf_dv")
    assert np.isnan(nan_dv)
    assert cols == dict(nr_resolved=True, nr_s=1005.0, nr_target="s_a0", nr_s_target=1000.0,
                        nr_dv=2.0, nr_cf="s_a0", nr_overruled="s_b", nr_tested="s_b,s_c0,s_a0",
                        nr_missing="", nr_stale=0)
    # R30 review: a verdict that still needs a solve (a tested state of the run with no
    # record, or a qualifying state's counterfactual) is unresolved, and listed
    gap = nr(1005.0, False, 2.0, 990.0, target="s_c0", tested=("s_b", "s_c0"),
             missing=("s_a0",), n_stale=2)
    assert not measures.no_return_resolved(gap) and not A.no_return_resolved(gap)[0]
    cols = A.no_return_columns(gap)
    assert (cols["nr_resolved"], cols["nr_missing"], cols["nr_stale"]) == (False, "s_a0", 2)


def test_R28_run_table_records_the_typing_path(monkeypatch):
    bank, cells = T._bank_cells()
    det = dict(cell=np.array([0, 1, 1]), phase=np.zeros(3, int), sample=np.full(3, -1),
               steer_sign=np.array([1, -1, -1]), W=np.full(3, 400), s0=np.full(3, 700.0),
               v_e_test=np.array([40.0, 41.0, 30.0]))

    class Keys(T.FakeKeys):
        def no_return(self, cell, agent, timesteps, phase, sample, traj, key_traj):
            if cell.cell_id == 0:
                return measures.NoReturn(1005.0, True, 2.0)
            return measures.NoReturn(None, False, np.nan, 1000.0)   # unresolved

    monkeypatch.setattr(measures, "trajectory_from_rollout",
                        lambda arr, i, g: type("Tr", (), {"v_e": float(arr["v_e_test"][i])})())
    monkeypatch.setattr(measures, "response_metrics", lambda traj, marks, null, tables:
                        measures.ResponseMetrics(v_e=traj.v_e, crashed=traj.v_e > 35,
                                                 v_min=traj.v_e - 5))
    monkeypatch.setattr(measures, "key_errors", lambda *a, **k: measures.KeyErrors())
    monkeypatch.setattr(measures, "crash_type", lambda traj, marks, key, nr: "overspeed")
    rt = A.run_table("c1", 0, {"det": det}, cells, bank, Keys(), None, check_design=False)
    assert set(rt) == set(A.RUN_COLUMNS)
    assert rt["nr_resolved"].tolist() == [True, False, False]
    assert rt["nr_s"][0] == 1005.0 and np.isnan(rt["nr_s"][1:]).all()
    v = A.viability_typing(rt)
    assert not v["passed"] and v["unresolved_cells"] == [1]
    assert (v["n_crashed"], v["n_resolved"], v["n_overspeed"]) == (2, 1, 2)


def test_R30_viability_typing():
    """Ruling R30's check of c1 (it replaces R28's): every crashed u = 0 run resolved by the
    viability path, none over-brake or line/steer, every one unrecoverable at or before s_c0
    overspeed; the others (an s_a0 decision, an s_b after s_c0) listed with the solve."""
    def runs(nr, ct, tgt=None, s_t=None, crashed=None, phase=None):
        n = len(nr)
        return A._to_table([dict(cell=i, mode="det", phase=0 if phase is None else phase[i],
                                 sample=-1, crashed=True if crashed is None else crashed[i],
                                 nr_resolved=nr[i], crash_type=ct[i],
                                 nr_target=(tgt or ["s_c0"] * n)[i],
                                 nr_s_target=(s_t or [1000.0] * n)[i], nr_dv=3.0,
                                 nr_cf="s_c0" if ct[i] == "overspeed" else "", nr_overruled="")
                            for i in range(n)],
                           ("cell", "mode", "phase", "sample", "crashed", "nr_resolved",
                            "crash_type", "nr_target", "nr_s_target", "nr_dv", "nr_cf",
                            "nr_overruled"))

    ok = A.viability_typing(runs([True, True], ["overspeed", "overspeed"]))
    assert ok["passed"] and ok["n_early_overspeed"] == ok["n_early"] == 2
    assert ok["types"] == {"overspeed": 2} and ok["others"] == []
    # an early state not typed overspeed fails; one decided at s_a0 is listed, not failed
    other = A.viability_typing(runs([True, True], ["overspeed", "other"]))
    assert not other["passed"] and other["early_not_overspeed_cells"] == [1]
    a0 = A.viability_typing(runs([True, True], ["overspeed", "other"], tgt=["s_c0", "s_a0"],
                                 s_t=[1000.0, 1036.0]))
    assert a0["passed"] and [o["cell"] for o in a0["others"]] == [1]
    assert a0["others"][0]["target"] == "s_a0" and a0["others"][0]["cf"] == ""
    # over-brake or line/steer anywhere fails; so does a run the fallback typed
    assert not A.viability_typing(runs([True, True], ["overspeed", "line_steer"],
                                       tgt=["s_c0", "s_a0"]))["passed"]
    unres = A.viability_typing(runs([True, False], ["overspeed", "overspeed"]))
    assert not unres["passed"] and unres["unresolved_cells"] == [1]
    # crashes at u_k > 0 and survivors do not count; none at u = 0 exercises nothing
    only_k = runs([False, True], ["other", "overspeed"], phase=[1, 0])
    assert A.viability_typing(only_k)["passed"]
    none = A.viability_typing(runs([False], [""], crashed=[False]))
    assert not none["passed"] and none["n_crashed"] == 0
    # with the key table: a crash on a cell without a verified u = 0 key (no viability
    # solve can start there) is listed, not judged; s_b counts as early only before s_c0
    keys = A._to_table([dict(cell=0, key_status="verified", s_c0=1000.0, s_a0=1036.0),
                        dict(cell=1, key_status="missing", s_c0=1000.0, s_a0=1036.0)],
                       ("cell", "key_status", "s_c0", "s_a0"))
    two = runs([True, False], ["overspeed", "other"])
    assert not A.viability_typing(two)["passed"]
    scoped = A.viability_typing(two, keys)
    assert scoped["passed"] and scoped["n_crashed"] == 1
    assert scoped["n_no_key"] == 1 and scoped["no_key_cells"] == [1]
    assert not A.viability_typing(runs([False, True], ["other", "overspeed"]), keys)["passed"]
    keys2 = A._to_table([dict(cell=i, key_status="verified", s_c0=1000.0, s_a0=1036.0)
                         for i in range(2)], ("cell", "key_status", "s_c0", "s_a0"))
    sb = runs([True, True], ["other", "overspeed"], tgt=["s_b", "s_b"], s_t=[1010.0, 980.0])
    v = A.viability_typing(sb, keys2)                      # 1010 > s_c0: listed (138, 139)
    assert v["passed"] and v["n_early"] == 1 and v["others"][0]["s_target_rel_c0"] == 10.0
    assert not A.viability_typing(runs([True], ["other"], tgt=["s_b"], s_t=[980.0]),
                                  keys2)["passed"]