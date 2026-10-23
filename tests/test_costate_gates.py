"""Idea 2's gates (costate.gates, costate.own; design v2.1, Section 7: K0-K6).

- K0 on stored keys (a W = 400 key, a W = 800 key and an anchored final): every identity
  within its tolerance and the record 'ok'.
- K1: the sample is step 2's Gate 1 stratification; the pass rule (kinks and unresolved
  comparisons left out, the re-solves' noise floor, the descriptive share, planted bugs that
  must fail) on rows with known answers; one stored key's states against real re-solves,
  with every planted bug failing.
- K2: the twin pairs, the retention rule t_H, the twins' discrepancies on T; K2b's second
  corner (its total turn) and the exclusion rule.
- K3's judge fails on every planted defect of a passing report.
- K3b in a float64 subprocess: the own-costate loop reproduces step 2's closed loop, its AD
  matches finite differences of the env's own return on step 2's linear driver, restarts
  match per-state VJPs, and the planted bugs fail.
- K4's plants that need no network (PR-5 family through C1, PR-9's heads) and PR-1a/PR-7 on a
  synthetic layer; K6's counts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from apexzero.costate import critic as CR
from apexzero.costate import data_dirs
from apexzero.costate import gates as GT
from apexzero.costate import probes as PB
from apexzero.costate import states as ST
from apexzero.costate import targets as TG
from apexzero.fields import gates as FG
from apexzero.fields import keys as K
from apexzero.fields import kkt as KK

REPO = Path(__file__).resolve().parents[1]
KEYS, STIM = data_dirs()
HAVE_KEYS = (KEYS / "nominal").exists() and (STIM / "cells.json").exists()
needs_keys = pytest.mark.skipif(not HAVE_KEYS, reason="step 1's keys are not on this machine")
EXAMPLE = "03b6076863e3adfe335e6757"


@pytest.fixture(scope="module")
def bank():
    return K.design(STIM)[0]


@needs_keys
def test_k0_on_stored_keys(bank):
    refs = ST.a2_refs()
    pick = [r for r in refs if r.key_id == EXAMPLE]
    pick.append(next(r for r in refs if r.W == 800))
    pick.append(next(iter(ST.n_refs().values()))[0])
    res = GT.k0(pick, bank)
    s = res["summary"]
    assert s["pass"] and s["n"] == 3 and not s["failed"]
    assert s["max"]["x0_identity"] <= TG.X0_TOL and s["max"]["route_gap"] <= TG.ROUTE_TOL
    assert s["max"]["command_sign_violation"] <= TG.SIGN_TOL
    assert s["max"]["straight_identity"] < 1e-12 and s["plants_fail"]
    assert s["max"]["action_identity"] <= TG.ACTION_TOL and s["max"]["end_gap"] <= TG.ROUTE_TOL
    assert all(r["ok"] for r in res["rows"])


@needs_keys
def test_k1_sample_is_gate1s_stratification():
    a = GT.k1_sample()
    b = GT.k1_sample()
    assert [r.key_id for r in a] == [r.key_id for r in b] and len(a) == 50
    fam = {}
    for r in a:
        fam[FG.family(r)] = fam.get(FG.family(r), 0) + 1
    assert fam == FG.GATE1_STRATA


def _all_plants(**kw):
    out = {p: 0.5 for p in TG.PLANTS}
    out.update(pre_jump=0.0, unfolded=0.0)
    out.update(kw)
    return out


def _row(c, err, kink=False, unres=False, ok=True, contact=False, plants=None, **kw):
    return {"component": c, "err": err, "kink": kink, "unresolved": unres, "ok": ok,
            "contact": contact, "plants": plants or _all_plants(), **kw}


def test_k1_rules_with_known_answers():
    rows = [_row("s", 1e-5) for _ in range(40)] + [_row("n", 1e-5) for _ in range(40)]
    d = GT.k1_decision(rows)
    assert d["pass"] and d["components"]["s"]["median"] == pytest.approx(1e-5)
    # the pre-jump plant is scored on contact states only and 'unfolded' on the pedal rows
    # whose fold is active: here neither is scored, so the plants do not all fail
    # (a plant cannot pass vacuously)
    assert d["plants"]["no_es"]["fails"] and not d["plants"]["pre_jump"]["fails"]
    assert not d["plants_fail"] and set(d["plants_missing"]) == {"pre_jump", "unfolded"}
    rows2 = rows + [_row("n", 0.0, contact=True, plants=_all_plants(pre_jump=0.3))] * 5 \
        + [_row("pedal", 1e-5, saturated=True, fold_active=True,
                plants=_all_plants(unfolded=0.3)) for _ in range(5)]
    d2 = GT.k1_decision(rows2)
    assert d2["plants_fail"] and d2["pass_with_plants"], d2["plants_missing"]
    assert d2["components"]["pedal"]["saturated_counted_share"] == 1.0
    # a plant missing from the rows fails the plants
    drop = [{**r, "plants": {k: v for k, v in r["plants"].items() if k != "off_by_one"}}
            for r in rows2]
    assert not GT.k1_decision(drop)["plants_fail"]
    # an anchored decision needs j_w, and pi rows; without them K1 does not pass
    anch = rows2 + [_row("s", 1e-5, kind="anchored", state="decision")]
    da = GT.k1_decision(anch)
    assert "j_w" in da["plants_missing"] and not da["pass"] and da["pi"]["n"] == 0
    # a p95 above 1e-2 fails; kinks and unresolved comparisons do not count
    bad = rows + [_row("s", 0.5) for _ in range(5)]
    assert not GT.k1_decision(bad)["pass"]
    hidden = rows + [_row("s", 0.5, kink=True) for _ in range(5)] \
        + [_row("s", 0.5, unres=True) for _ in range(5)]
    dh = GT.k1_decision(hidden)
    assert dh["pass"] and dh["components"]["s"]["unresolved_share"] == pytest.approx(5 / 50)
    assert not dh["components"]["s"]["design_rule"]["pass"]      # the literal rule counts them
    # more than 30% kinks or unresolved: descriptive, and out of the pass
    many = rows + [_row("s", 0.5, kink=True) for _ in range(30)]
    dm = GT.k1_decision(many)
    assert dm["descriptive"] == ["s"] and dm["pass"]
    # pi rows with their plants
    pi = [{"err": 1e-7, "ok": True, "plants": {p: 0.6 for p in TG.PI_PLANTS}}] * 4
    dp = GT.k1_decision(rows2, pi)
    assert dp["pi"]["pass"] and all(v["fails"] for v in dp["pi"]["plants"].values())
    assert dp["pass_with_plants"]
    # a saturated pedal row left unresolved counts toward pedal's share, not its pass
    sat = rows2 + [_row("pedal", 0.9, unres=True, saturated=True, fold_active=True,
                        plants=_all_plants(unfolded=0.3))] * 5
    ds = GT.k1_decision(sat)
    assert ds["components"]["pedal"]["saturated_counted_share"] == pytest.approx(0.5)
    assert ds["components"]["pedal"]["unresolved_share"] == pytest.approx(0.5)
    # the noise floor: two values under the re-solves' noise are a zero field
    fd = TG.FD(value=-1.16e-6, fd_h=0, fd_h2=0, discrepancy=5e-9, side="central", h=1e-6,
               ok=True, iterations=1, forward=0, backward=0, noise=8.6e-5)
    assert GT.k1_error(-9.7e-7, fd) == 0.0 and TG.fd_error(-9.7e-7, fd) > 0.1
    fd2 = TG.FD(value=1.94, fd_h=0, fd_h2=0, discrepancy=0.96, side="below", h=1e-7, ok=True,
                iterations=1, noise=6.5e-4)
    assert GT.unresolved(fd2) and not GT.unresolved(fd)


@needs_keys
def test_k1_on_a_stored_key_with_planted_bugs(bank):
    ref = next(r for r in ST.a2_refs() if r.key_id == EXAMPLE)
    rows = GT.k1_key(ref, bank, components=(TG.S, TG.N, TG.VX),
                     states={"decision": 0, "braking": 20, "contact": 55})
    assert len(rows) == 9 and all(r["ok"] for r in rows)
    counted = [r for r in rows if not r["kink"] and not r["unresolved"]]
    assert counted and max(r["err"] for r in counted) < 1e-3
    contact = [r for r in rows if r["state"] == "contact"]
    assert contact and all(r["contact"] for r in contact)
    for p in ("no_es", "no_discount", "flip_sign", "off_by_one"):
        assert max(r["plants"][p] for r in counted) > 1e-2, p
    assert max(r["plants"]["pre_jump"] for r in contact) > 1e-2


@needs_keys
def test_k1_pedal_ladder_on_a_saturated_anchored_state(bank):
    """Review finding: at a pedal-saturated state the design's c = 1e-5 one-sided difference
    is noise-limited (the warm re-solves' J scatters by ~1.5e-8), and shrinking c made it
    worse (FD 1.94 against 0.0307 on this key at t = 15). The ladder from c = 1e-3 resolves
    the folded target, and the raw (unfolded) multipliers fail it."""
    ref = next(r for v in ST.n_refs().values() for r in v if r.key_id.startswith("9b1695f1"))
    kt = TG.key_targets(ref, bank=bank)
    cs = kt.jac
    assert cs.pedal_saturated[15] and abs(cs.fold.c[15]) > TG.ACTIVE_TOL
    fd = TG.fd_costate_k1(kt.problem, 15, TG.PEDAL, "below", solver=KK.fd_solver(kt.problem))
    assert fd.ok and fd.h >= TG.PEDAL_MIN_H and fd.ladder and not GT.unresolved(fd)
    assert GT.k1_error(float(cs.lam[15, TG.PEDAL]), fd) < 1e-3
    unf = TG.costate(kt.problem, "jac", plant="unfolded")
    assert GT.k1_error(float(unf.lam[15, TG.PEDAL]), fd) > 1e-2
    assert np.array_equal(np.delete(unf.lam, TG.PEDAL, 1), np.delete(cs.lam, TG.PEDAL, 1))


@needs_keys
def test_k2_twins_and_the_retention_rule(bank):
    pairs = GT.twin_pairs()
    assert len(pairs) == 6 and all(a.W == 400 and b.W == 800 for a, b in pairs)
    assert all(a.cell_id == b.cell_id for a, b in pairs)
    d = np.zeros((4, 300, 4))
    d[:, 250:, 3] = 1.0
    h = GT.k2_horizon(d)
    assert h["t_H"] == 249 and h["first_divergence"] == 250
    assert GT.k2_horizon(np.zeros((3, 50, 4)))["t_H"] == 49
    sd = np.ones(4)
    res = GT.k2_twins(pairs[:2], sd, bank)
    assert res["T_max_t"] <= 130 and res["horizon"]["t_H"] > res["T_max_t"]
    assert res["pass"] and all(max(p["T_max"]) < 0.05 for p in res["pairs"])


def test_k2_retention_reads_ts_uncut_t_end():
    """Review finding: drops_T compared t_H with the twins' T range only. From T's uncut
    t_end, a W = 400 plan that has not reached s_c1 + 50 m by t_H is the design's Fallback."""
    tab = {"key_id": np.array(["a", "a", "b", "c"]), "W": np.array([400, 400, 400, 800]),
           "t_end": np.array([110, 110, 129, 595])}
    r = GT.k2_retention(120, tab)
    assert r["drops_T"] and r["fallback_keys"] == ["b"] and not r["pass"]
    assert r["T_max_t_by_W"] == {"400": 129, "800": 195}
    ok = GT.k2_retention(267, tab)
    assert ok["pass"] and not ok["drops_T"] and not ok["fallback_keys"]
    assert GT.k2_retention(190, tab)["cut_w800"] == ["c"]


def test_k0_summary_requires_the_planted_action_bugs_and_skips_none():
    good = {"key_id": "a", "ok": True, "plants_fail": True, "straight_identity": 1e-15,
            "x0_identity": 0.0, "action_plants": {p: 1e-3 for p in TG.ACTION_PLANTS},
            "repolish_change": {"lam": 0.0, "pi": 0.0, "switch": 0.0}}
    other = {**good, "key_id": "b", "straight_identity": None, "a0_identity": 1e-18}
    s = GT.k0_summary([good, other])
    assert s["pass"] and s["max"]["straight_identity"] == 1e-15
    assert s["max"]["a0_identity"] == 1e-18
    bad = {**other, "plants_fail": False}
    assert not GT.k0_summary([good, bad])["pass"]


def test_k2b_second_corner_and_rule():
    sc = GT.second_corner(1000, 1200.0, 30.0, 90.0, -1)
    turn = np.trapezoid(sc.kappa, dx=2.0)
    assert turn == pytest.approx(-np.pi / 2, rel=0.02)
    assert np.all(sc.kappa[np.arange(1000) * 2.0 < 1200.0] == 0.0)
    assert not np.any(sc.w_left) and not np.any(sc.w_right)
    keys = [{"severities": {"fast": {"ok": True, "by_phase": {"exit": [0.0, 0.0, 0.0, 0.3],
                                                               "corner": [0.0] * 4}}}}]
    r = GT.k2b_rule(keys)
    assert r["excluded"]["exit"] == ["vx"] and r["excluded"]["corner"] == []
    # the design's statistic: the p90 pooled over every key's rows in the phase, not the
    # maximum of per-key p90s (a key with few rows above 0.1 does not exclude alone)
    k1 = {"severities": {"slow": {"ok": True, "rows": [[0.0, 0.0, 0.0, 0.5]] * 2
                                  + [[0.0] * 4] * 8, "phase": ["exit"] * 10}}}
    k2 = {"severities": {"slow": {"ok": True, "rows": [[0.0] * 4] * 30, "phase": ["exit"] * 30}}}
    pooled = GT.k2b_rule([k1, k2])
    assert pooled["excluded"]["exit"] == [] and pooled["pooled_rows"]
    per_key_max = np.quantile(np.asarray(k1["severities"]["slow"]["rows"]), 0.9, axis=0)[3]
    assert per_key_max > GT.K2B_TOL                  # the old rule would have excluded vx
    assert GT.k2b_rule([k1])["excluded"]["exit"] == ["vx"]


def _good_reports():
    lin = {c: {"g": 0.0, "g_held": 0.0, "pi": 0.0, "accel": 0.0}
           for c in ("vx", "lateral", "xi", "pedal")}
    lin["accel_prev"] = {k: 0.0 for k in ("g_held", "accel_sens", "pi_held", "g", "pi",
                                          "g_reset")}
    lin["heads"] = {"n_heads": 2, "head0_xi": 0.0, "head1": 0.0, "main": 0.0}
    lin["composed"] = {"g_held": 1e-16, "pi_held": 0.0, "g_xprev": 1e-16}
    lin["fd"] = {v: {"max_err": 1e-9, "share": 1.0} for v in ("manifold", "held", "xprev")}
    plants = {"env_accel_err": 0.0, "env_step_err": 0.0, "frenet_leak": 0.0,
              "maps": {"max_err": 5e-9, "newton": 1e-14}, "J_acc_err": 0.0,
              "J_acc_other": 0.0, "J_prev_err": 0.0, "J_prev_other": 0.0,
              "J_accel_rows": 0.0, "linear": lin,
              "plants": {"state": {"g": 1e-16, "pi": 1e-16},
                         "c=0": {"g": 1e-16, "pi": 1e-16, "g_held": 1e-16, "accel_sens": 0.0},
                         "c=20": {"g": 2e-11, "pi": 1e-11, "g_held": 2e4, "pi_held": 1.5e4,
                                  "g_xprev": 2e4, "accel_sens": 14.0,
                                  **{f"fd_{v}": {"max_err": 1e-7, "share": 1.0}
                                     for v in CR.VARIANTS}}}}
    plants["wrong"] = {"linear": {k: {"max": 0.03, "row_min": 0.018} for k in CR.WRONG_PAIRS},
                       "c=20": {**{k: {"max": 1.0, "row_min": 0.02} for k in CR.WRONG_PAIRS},
                                **{f"closed:{f}": {"max": 2e4, "row_min": None}
                                   for f in CR.WRONG_CLOSED_FORMS}}}
    random = {"defect_body": 5e-12, "stop_gradient": {"flax": 0.0, "extractor": 0.1},
              "apply_gap": [{"v": 0.0, "heads": 0.0, "actor": 0.0, "n_heads": 5,
                             "flax_heads": 5}],
              "jz": {"max_err": 3e-10, "n_states": 205, "n": 1640},
              "fd": {v: {"main": 8e-9, "heads": 2e-8, "n_states": 205} for v in CR.VARIANTS},
              "composed": [{"g_held": 1e-16, "g_xprev": 2e-16}],
              "observability": {"n_exit": 4, "exit_flagged": 4, "before_flagged": 0,
                                "apex_flagged": 0, "f32_same_flags": True}}
    return plants, random, [{"pass": True}]


def test_k3_judge_fails_on_planted_defects():
    import copy

    plants, random, ext = _good_reports()
    assert GT.k3_judge(plants, random, ext)["pass"]
    for path, value in ((("plants", "c=20", "g_held"), 1e-12),      # held passes: wrong
                        (("plants", "c=20", "g"), 1e-6),             # primary off
                        (("linear", "vx", "g"), 1e-9),
                        (("maps", "max_err"), 1e-3)):
        bad = copy.deepcopy(plants)
        d = bad
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
        assert not GT.k3_judge(bad, random, ext)["pass"], path
    bad = copy.deepcopy(random)
    bad["stop_gradient"]["flax"] = 0.3
    assert not GT.k3_judge(plants, bad, ext)["pass"]
    bad = copy.deepcopy(random)
    bad["observability"]["before_flagged"] = 1
    assert not GT.k3_judge(plants, bad, ext)["pass"]
    assert not GT.k3_judge(plants, random, [{"pass": False}])["pass"]
    # review finding: entries the old whitelist ignored are judged now
    for path in (("linear", "vx", "pi"), ("linear", "lateral", "accel"),
                 ("linear", "composed", "g_xprev")):
        bad = copy.deepcopy(plants)
        bad[path[0]][path[1]][path[2]] = 1e-9
        assert not GT.k3_judge(bad, random, ext)["pass"], path
    for key, value in ((("composed",), [{"g_held": 1e-9, "g_xprev": 0.0}]),
                       (("composed",), []),
                       (("jz", "n_states"), 69),
                       (("fd", "manifold", "n_states"), 69),
                       (("observability", "f32_same_flags"), False),
                       (("observability", "apex_flagged"), 1)):
        bad = copy.deepcopy(random)
        d = bad
        for k in key[:-1]:
            d = d[k]
        d[key[-1]] = value
        assert not GT.k3_judge(plants, bad, ext)["pass"], key
    # K3's pass rule (design v2.2 Section 13): held and x_{t-1}-held forms below 1e-6, the
    # on-manifold form below 1e-5, every form present
    for form, value, ok in (("manifold", 5e-6, True), ("manifold", 2e-5, False),
                            ("held", 5e-6, False), ("xprev", 2e-6, False)):
        for out in ("main", "heads"):
            bad = copy.deepcopy(random)
            bad["fd"][form][out] = value
            assert GT.k3_judge(plants, bad, ext)["pass"] is ok, (form, out, value)
    bad = copy.deepcopy(random)
    del bad["fd"]["xprev"]
    assert not GT.k3_judge(plants, bad, ext)["pass"]
    # every planted wrong pullback must exceed 1e-4 (ten times the on-manifold tolerance),
    # none may be missing, and the plant's own forms must match their differences
    res = GT.k3_judge(plants, random, ext)
    assert res["planted_wrong"]["min"] == 0.03 and res["planted_wrong"]["argmin"].startswith(
        "linear:") and not res["planted_wrong"]["missing"]
    assert res["tolerances"]["fd"] == {"manifold": 1e-5, "held": 1e-6, "xprev": 1e-6}
    assert res["tolerances"]["wrong_min"] == 1e-4 == CR.WRONG_MIN
    assert res["planted_wrong"]["pass_rule"] == "each > 0.0001"
    # the floor pinned from both sides (between 9e-5 and 2e-4), on the cross-form errors and
    # the wrong closed forms alike; the planted critics' own forms under the per-form rule
    # (held and x_{t-1}-held 1e-6, on-manifold 1e-5), the linear critic's too
    for path, value, ok in ((("wrong", "c=20", "held@manifold", "max"), 2e-4, True),
                            (("wrong", "c=20", "held@manifold", "max"), 5e-5, False),
                            (("wrong", "linear", "xprev@held", "max"), 2e-4, True),
                            (("wrong", "linear", "xprev@held", "max"), 9e-5, False),
                            (("wrong", "c=20", "closed:g_held", "max"), None, False),
                            (("plants", "c=20", "pi_held"), 2e-4, True),
                            (("plants", "c=20", "pi_held"), 5e-5, False),
                            (("plants", "c=20", "g_xprev"), 1e-12, False),
                            (("plants", "c=20", "fd_manifold", "max_err"), 2e-6, True),
                            (("plants", "c=20", "fd_manifold", "max_err"), 2e-5, False),
                            (("plants", "c=20", "fd_held", "max_err"), 2e-6, False),
                            (("plants", "c=20", "fd_xprev", "max_err"), 2e-6, False),
                            (("linear", "fd", "manifold", "max_err"), 5e-6, True),
                            (("linear", "fd", "manifold", "max_err"), 2e-5, False),
                            (("linear", "fd", "held", "max_err"), 2e-6, False),
                            (("linear", "fd", "xprev", "max_err"), 2e-6, False)):
        bad = copy.deepcopy(plants)
        d = bad
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
        assert GT.k3_judge(bad, random, ext)["pass"] is ok, (path, value)
    bad = copy.deepcopy(plants)
    del bad["linear"]["fd"]["xprev"]              # every form present
    assert not GT.k3_judge(bad, random, ext)["pass"]
    for drop in (("wrong", "linear", "manifold@xprev"), ("wrong", "c=20"), ("wrong",),
                 ("plants", "c=20", "fd_xprev")):
        bad = copy.deepcopy(plants)
        d = bad
        for k in drop[:-1]:
            d = d[k]
        del d[drop[-1]]
        r = GT.k3_judge(bad, random, ext)
        assert not r["pass"], drop
    bad = copy.deepcopy(plants)
    del bad["plants"]["c=20"]                    # no accel-reading plant: vacuous, fails
    assert not GT.k3_judge(bad, random, ext)["pass"]
    # a partial K3 (fewer keys) is judged without the sample size
    small = copy.deepcopy(random)
    small["jz"]["n_states"] = small["fd"]["manifold"]["n_states"] = 69
    assert GT.k3_judge(plants, small, ext, min_states=0)["pass"]


def test_k3_trained_judge_names_each_failure():
    """K3 after the tag (Section 7, ``gates.k3_trained_judge``): a passing set of
    measurements passes; each planted defect fails its own check with the reason 'K3
    trained check failed: <check> (...)'; a measurement that raised fails (contained)."""
    import copy

    gap = {"v": 0.0, "heads": 0.0, "n_heads": 5, "flax_heads": 5, "dtype": "float64",
           "actor": 1e-16, "actor_dtype": "float64"}
    ext = {"dtype": "float64", "actor_mean": 1e-17, "heads": 5, "critic_heads": 2e-15,
           "critic_v": 1e-15}
    f32 = {"max_abs": 7e-7, "max_rel": 3e-5, "v_absmax": 1.1, "dtype32": "float32", "n": 477,
           "tol": CR.F32_TOL}
    ok = GT.k3_trained_judge(gap, ext, f32, 5, 5)
    assert ok["decision"] == {"pass": True, "failed": [], "reason": None}
    assert ok["checks"] == dict.fromkeys(GT.K3_TRAINED_CHECKS, True)
    assert ok["tolerances"] == {"extractor": PB.EXTRACTOR_TOL, "float32": CR.F32_TOL} == {
        "extractor": 1e-10, "float32": 1e-4}
    assert GT.expected_heads("runs/dr-pilot-200M") == 5 and GT.expected_heads("S2") == 0
    assert GT.expected_heads("runs/elsewhere") is None

    def failed(g=gap, e=ext, f=f32, heads=5, cfg_heads=5):
        d = GT.k3_trained_judge(g, e, f, heads, cfg_heads)["decision"]
        assert not d["pass"] and d["reason"].startswith("K3 trained check failed: ")
        return d

    for src, key, val in ((gap, "v", 2e-10), (gap, "heads", 1e-9), (gap, "actor", 1e-8),
                          (ext, "actor_mean", 3e-10), (ext, "critic_v", float("nan")),
                          (ext, "critic_heads", 1e-6), (gap, "dtype", "float32"),
                          (ext, "dtype", "float32")):
        bad = copy.deepcopy(src)
        bad[key] = val
        d = failed(bad if src is gap else gap, bad if src is ext else ext)
        assert d["failed"] == ["extractor"] and key in d["reason"], (key, d)
    # the head rule: 5 per generalist, none for S1-S3, every count equal
    assert failed(heads=0)["failed"] == ["heads"]
    assert failed(heads=None)["failed"] == ["heads"]
    d = failed(g={**gap, "flax_heads": 4})
    assert d["failed"] == ["heads"] and "Flax 4" in d["reason"]
    assert failed(e={**ext, "heads": 3})["failed"] == ["heads"]
    # the float32 check: 1e-4 in the critic's units, a run really in float32
    d = failed(f={**f32, "max_abs": 3e-4})
    assert d["failed"] == ["float32"] and "0.0003 > 0.0001" in d["reason"]
    assert failed(f={**f32, "dtype32": "float64"})["failed"] == ["float32"]
    assert failed(f={**f32, "max_abs": float("nan")})["failed"] == ["float32"]
    # a measurement that raised fails every check it feeds
    d = failed(g={"error": "ValueError: boom"})
    assert d["failed"] == ["extractor", "heads"] and "critic: ValueError: boom" in d["reason"]
    d = failed(f={"error": "ValueError: boom"})
    assert d["failed"] == ["float32"] and "ValueError: boom" in d["reason"]
    d = failed(g={**gap, "v": 1.0}, f={**f32, "max_abs": 1.0})
    assert d["failed"] == ["extractor", "float32"]
    assert d["reason"].index("extractor (") < d["reason"].index("; float32 (")


_K3B = r"""
import json, sys
import jax
jax.config.update("jax_enable_x64", True)
from apexzero.costate import own as OW, inputs as IN
from apexzero.fields import network as NW
bank, cells = IN.design()
st = NW.cell_start(cells[344], 0, bank)
rd = NW.road(bank, st.stim_id)
net = NW.linear_net(*NW.linear_driver(st.sigma))
s = OW.k3b_start(net, rd.base, NW.loop_inputs(st, 60), 60, restarts=(0.5,), name="linear@344")
acc = OW.k3b_start(NW.linear_net(*OW.accel_driver(st.sigma)), rd.base, NW.loop_inputs(st, 60),
                   60, restarts=(), name="accel@344")
d = OW.k3b_decision([s, acc])
d1 = OW.k3b_decision([s])
print(json.dumps({"rec": s.record(), "acc": acc.record(), "dec": d, "dec1": d1}))
"""


@pytest.mark.skipif(not HAVE_KEYS, reason="step 1's stimuli are not on this machine")
def test_k3b_own_costate_loop_on_the_linear_driver():
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([sys.executable, "-c", _K3B], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=600, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    rec, dec = out["rec"], out["dec"]
    assert rec["valid"] and rec["count"] >= 40
    assert rec["trace_err"] <= 1e-9 and rec["j_gap"] <= 1e-10
    assert rec["restart_err"] <= 1e-9
    assert sum(rec["counted"]) >= 7
    errs = [e for e, c in zip(rec["err"], rec["counted"], strict=True) if c]
    assert max(errs) < 1e-3
    assert all(v["fails"] for v in dec["plants"].values()), dec["plants"]
    assert dec["pass"] and dec["manifold"]["pass"]
    # the on-manifold check (review finding: C1-own compared the critic's on-manifold g with
    # an accel-held lambda^own): on the linear driver (dJ/daccel = 0) held = manifold; on a
    # control that reads the accel channels the on-manifold lambda^own matches differences
    # with x_{k-1} recovered by Newton and accel recomputed, and the held one fails them
    m, ma = rec["manifold"], out["acc"]["manifold"]
    assert m["accel_sens"] == 0.0 and m["ad_manifold"] == m["ad_held"]
    assert ma["accel_sens"] > 0.0 and ma["newton"] < 1e-9 and sum(ma["counted"]) >= 7
    em = [e for e, c in zip(ma["err_manifold"], ma["counted"], strict=True) if c]
    eh = [e for e, c in zip(ma["err_held"], ma["counted"], strict=True) if c]
    assert max(em) < 1e-3 and max(eh) > 1e-2
    assert dec["manifold"]["held_on_accel_plant"]["fails"]
    assert "held_on_accel_plant" not in out["dec1"]["manifold"]


def _synthetic_world(n_geo=24, keys=4, steps=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_geo):
        for k in range(keys):
            d = 1 if k % 2 == 0 else -1
            for t in range(steps):
                rows.append((g, 10.0 + 5.0 * g, (90.0, 180.0)[g % 2], f"g{g}k{k}", d, t))
    tab = {"R": np.array([r[1] for r in rows]), "theta": np.array([r[2] for r in rows]),
           "key_id": np.array([r[3] for r in rows]), "direction": np.array([r[4] for r in rows]),
           "pair_id": np.array([r[0] for r in rows]), "t": np.array([r[5] for r in rows]),
           "ds_apex": rng.normal(size=len(rows)) * 20, "vx": 40 + 10 * rng.normal(size=len(rows)),
           "width": np.full(len(rows), 12.0)}
    n = len(rows)
    z = rng.normal(size=(n, 30))
    lam = np.stack([np.sin(z[:, 0]) + 0.1 * tab["t"], tab["direction"] * np.cos(z[:, 1]),
                    tab["direction"] * z[:, 2], z[:, 3] ** 2], 1)
    w = ST.weights(tab, "key")
    fo = PB.folds(tab)
    res = ST.cross_fit(ST.decoy_d1(tab), lam, fo.outer, w)
    h3 = [np.tanh(z @ rng.normal(size=(30, 24)) / 5) for _ in range(3)]
    m = [h @ rng.normal(size=(24, 2)) for h in h3]
    return GT.ProbeWorld(tab, z, fo, w, tab["direction"], lam, res.by_fold, h3, m)


@pytest.fixture(scope="module")
def big_world():
    """4,608 rows over 48 geometries: large enough for K4's own 0.03 band."""
    pw = _synthetic_world(n_geo=48, keys=8, steps=12)
    G = int(pw.fo.cluster.max()) + 1
    return pw, PB.cluster_counts(G, 300, np.random.default_rng(2))


def test_k4_plants_without_a_network(big_world):
    pw, counts = big_world
    r9 = GT.pr9(pw, counts)
    assert r9["pass"] and r9["D_V"] <= 1e-20
    r5 = GT.pr5(pw, counts)
    cases = {r["case"]: r for r in r5["rows"]}
    assert cases["snr=4"]["decision"] in ("confirm", "inconclusive")
    # every SNR within K4's own band of SNR / (1 + SNR) on every component, the heavy-tailed
    # z^2 component included (C1's least-squares slope; the Huber slope fell 0.034-0.044
    # short at SNR 1), the odd components' expectation no longer pooled
    for snr in GT.K4_C1_SNR:
        r = cases[f"snr={snr:g}"]
        assert np.allclose(r["expected"], snr / (1 + snr))
        assert np.max(np.abs(np.subtract(r["r2c"], r["expected"]))) <= GT.K4_TOL, r
    assert cases["negated"]["decision"] != "confirm" and not cases["negated"]["label"]
    assert cases["one_flipped"]["decision"] != "confirm"
    assert cases["times10"]["decision"] == "confirm" and not cases["times10"]["label"]
    assert r5["PR-5e"]["pass"], r5["PR-5e"]      # the floor: uninformative gradients ~ 0
    assert r5["pass"]
    out = GT.k4(pw, counts, which=("PR-1a", "PR-7"))
    assert out["PR-7"]["pass"], out["PR-7"]
    assert out["PR-1a"]["pass"], out["PR-1a"]     # K4_TOL (0.03), partial and Delta2


def test_k6_counts():
    tab = {"key_id": np.array(["a"] * 4 + ["b"] * 4), "R": np.array([10.0] * 4 + [20.0] * 4),
           "theta": np.full(8, 90.0), "pair_id": np.arange(8) // 4,
           "ds_c1": np.array([-5, 5, 15, 30, -5, 5, 15, 30.0]),
           "multi_basin": np.zeros(8, bool), "contact": np.array([0, 1, 0, 0, 0, 0, 1, 0], bool),
           "pedal_saturated": np.array([1, 0, 0, 1, 1, 0, 0, 1], bool),
           "interior": np.array([1, 0, 1, 1, 1, 1, 0, 0], bool),
           "phase": np.array(["approach", "braking", "corner", "exit"] * 2)}
    obs = np.array([1, 1, 1, 0, 1, 1, 1, 0], bool)
    out = GT.k6(tab, obs, np.zeros(8, bool), {"exit": ["vx"], "approach": []},
                [{"key_id": "m1", "rows": 20}, {"key_id": "m2", "rows": 21}])
    assert out["n"] == 8 and out["keys"] == 2 and out["geometries"] == 2
    assert out["multi_basin_excluded"] == {"keys": 2, "rows": 41}
    assert out["s_unobservable"] == {"before": 0, "0-10": 0, "10-25": 0, "25+": 2}
    assert out["contact"] == 2 and out["interior"] == pytest.approx(5 / 8)
    assert out["beyond_preview"]["vx"] == 2 and out["beyond_preview"]["s"] == 0


def test_horizon_corrected_value_is_the_infinite_horizon_value():
    """Review finding (the value fits' target depended on W): at constant speed the
    horizon-corrected V*inf_t is the same at every t and for every W, while V*_t falls as the
    window's end nears."""
    from apexzero.costate import nulls as NL

    for W in (400, 800):
        vx, dt = 80.0, 0.05
        x = np.zeros((W + 1, 8))
        x[:, 0] = vx * dt * np.arange(W + 1)
        x[:, 3] = vx
        a = np.zeros((W, 2))
        v = NL.horizon_values(x, a)
        want = NL.PROGRESS * vx * dt / (1.0 - NL.GAMMA)
        np.testing.assert_allclose(v, want, rtol=1e-12)
        tail = NL.tail_values(x, a)
        assert tail[0] < want and tail[-1] == 0.0
    assert min(NL.VALUE_GRID) >= NL.VALUE_GRID_MIN and "value_ridge_window" in NL.NULLS
