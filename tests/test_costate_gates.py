"""Idea 2's gates (costate.gates, costate.own; design v2.1, Section 7: K0-K6).

- K0 on stored keys (a W = 400 key, a W = 800 key and an anchored final): every identity
  within its tolerance and the record 'ok'.
- K1: the sample is step 2's Gate 1 stratification; the pass rule (kinks and unresolved
  comparisons left out, the difference's floor, zero comparisons counted and never pooled,
  the descriptive share and no vacuous pass, planted bugs that must fail) on rows with known
  answers; pi under the same zero rule at its own floor (its zeros and its plants' counted,
  never pooled, no vacuous pass or fail); the literal kink reading beside it; the rule in
  the frozen constants; one stored key's states against real re-solves, with every planted
  bug failing; pi on a stored anchored key, its pedal comparison a zero one.
- K2: the twin pairs, the retention rule t_H, the twins' discrepancies on T; K2b's second
  corner (its total turn) and the exclusion rule.
- K3's judge fails on every planted defect of a passing report.
- K3b in a float64 subprocess: the own-costate loop reproduces step 2's closed loop, its AD
  matches finite differences of the env's own return on step 2's linear driver, restarts
  match per-state VJPs, and the planted bugs fail.
- K4's plants that need no network (PR-5 family through C1, PR-9's heads) and PR-1a/PR-7 on a
  synthetic layer; on a world with T-like decoys: PR-1b on one target array with its
  by-fold record, PR-6's one-sided rule and gain grid, PR-6b's known truth passing and its
  planted faults failing (the residualisation off or skipped, an over-smoothed kNN, D1
  without s - s_apex), PR-6c's record and rule, C2's primary statistic on C2's decision
  rows, K4 needing PR-6b and PR-6c, and the rules in the frozen constants; the no-gain
  plants (PR-2, PR-3's grouped split, PR-6, PR-6b) as paired increments on a host with a
  chance offset against its controls: the offset alone fails none of them, the unpaired
  value and the host's offset are recorded, and a planted gain still fails; f_D's k (the
  largest candidate whose every validation seed passes, the smallest as a blocker; the
  validation seeds on their own stream; at_decoy_k's residuals and floor; states.KNN_K the
  rule's choice on the recorded leaks); K5's wiring record and its rule in the frozen
  constants; K6's counts."""

import dataclasses
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
    # the difference's floor: two values under it are a zero comparison (None, not 0)
    fd = TG.FD(value=-1.16e-6, fd_h=0, fd_h2=0, discrepancy=5e-9, side="central", h=1e-6,
               ok=True, iterations=1, forward=0, backward=0, noise=8.6e-5)
    assert GT.k1_error(-9.7e-7, fd) is None and TG.fd_error(-9.7e-7, fd) > 0.1
    assert GT.k1_design_error(-9.7e-7, fd) > 0.1
    fd2 = TG.FD(value=1.94, fd_h=0, fd_h2=0, discrepancy=0.96, side="below", h=1e-7, ok=True,
                iterations=1, noise=6.5e-4)
    assert GT.unresolved(fd2) and not GT.unresolved(fd)


def test_k1_zero_comparisons_are_counted_not_pooled():
    """Gate 1's zero-field rule taken literally (the smoke run's K1 review): a zero
    comparison (None) is counted per component, per plant and per reading and never pooled
    as 0, and a component with no non-zero comparison that counts is descriptive, not a
    vacuous pass."""
    # 30 comparisons at 5e-3 fail the median; 40 zeros pooled as 0 would have passed it
    rows = [_row("s", 5e-3, err_design=5e-3) for _ in range(30)] \
        + [_row("s", None, err_design=None, plants=_all_plants(no_es=None))
           for _ in range(40)] + [_row("n", 1e-5) for _ in range(40)]
    d = GT.k1_decision(rows)
    s = d["components"]["s"]
    assert (s["n"], s["n_zero"]) == (30, 40) and s["median"] == pytest.approx(5e-3)
    assert not s["descriptive"] and not s["pass"] and not d["pass"]
    assert (s["design_rule"]["n"], s["design_rule"]["n_zero"]) == (30, 40)
    assert not s["design_rule"]["pass"]
    ps = d["plants"]["no_es"]["components"]["s"]
    assert (ps["n"], ps["n_zero"]) == (30, 40) and d["plants"]["no_es"]["n_zero"] == 40
    # every comparison of s a zero one: s is descriptive and n decides; nothing decided, no pass
    allzero = [_row("s", None, err_design=None) for _ in range(10)] \
        + [_row("n", 1e-5) for _ in range(40)]
    dz = GT.k1_decision(allzero)
    sz = dz["components"]["s"]
    assert (sz["n"], sz["n_zero"]) == (0, 10) and sz["descriptive"] and not sz["pass"]
    assert dz["descriptive"] == ["s"] and dz["pass"]
    assert dz["design_rule"] == {"decided": ["n"], "descriptive": ["s"], "pass": True}
    assert not GT.k1_decision(allzero[:10])["pass"]
    # a plant scored only on zero comparisons is not scored, so K1's plants do not fail
    good = [_row("s", 1e-5) for _ in range(40)] \
        + [_row("n", 1e-5, contact=True, plants=_all_plants(pre_jump=0.3)) for _ in range(5)] \
        + [_row("pedal", 1e-5, saturated=True, fold_active=True,
                plants=_all_plants(unfolded=0.3)) for _ in range(5)]
    assert GT.k1_decision(good)["plants_fail"]
    zp = GT.k1_decision([{**r, "plants": {**r["plants"], "pre_jump": None}} for r in good])
    assert zp["plants_missing"] == ["pre_jump"] and not zp["plants_fail"]
    assert zp["plants"]["pre_jump"]["n_zero"] == 5 and not zp["pass_with_plants"]


def _pi(err, plant=0.6, **kw):
    return {"err": err, "ok": True, "plants": {p: plant for p in TG.PI_PLANTS}, **kw}


def _k1_passing():
    """Lambda rows on which K1 passes with its plants (every required plant scored)."""
    return [_row("s", 1e-5) for _ in range(40)] \
        + [_row("n", 1e-5, contact=True, plants=_all_plants(pre_jump=0.3)) for _ in range(5)] \
        + [_row("pedal", 1e-5, saturated=True, fold_active=True,
                plants=_all_plants(unfolded=0.3)) for _ in range(5)]


def test_k1_pi_follows_the_zero_rule():
    """pi follows the lambda components' zero rule (the smoke run's K1 re-score: 8 of 20 pi
    comparisons, j = 1 on 8 anchored keys with |pi| ~ 1e-11 and |FD| <= 1.1e-9, were zero
    ones pooled as 0): at fd_pi's floor max(1e-6, 6 KINK_J_NOISE / (PI_H gamma)), a zero
    comparison of pi or of a plant is counted and never pooled, pi needs a non-zero one, and
    a pi plant fails only on non-zero comparisons."""
    lam = _k1_passing()
    assert GT.k1_decision(lam, [_pi(1e-6)])["pass_with_plants"]
    # the floor: fd_pi's step and gamma, above Gate 1's 1e-6
    noise = 6.0 * TG.KINK_J_NOISE / (TG.PI_H * 0.995)
    fd = TG.FD(value=1.184e-10, fd_h=0, fd_h2=0, discrepancy=3.55e-10, side="central",
               h=TG.PI_H, ok=True, iterations=4, noise=noise)
    assert noise == pytest.approx(6.03e-6, rel=1e-3) and noise > TG.ZERO_ATOL
    assert GT.k1_error(-8.312e-12, fd) is None and GT.k1_design_error(-8.312e-12, fd) is None
    assert TG.fd_error(-8.312e-12, fd) == 0.0                # the old rule's pooled zero
    mid = dataclasses.replace(fd, value=3e-6)                # between 1e-6 and the floor
    assert GT.k1_error(3e-6 + 1e-9, mid) is None and GT.k1_design_error(3e-6 + 1e-9, mid) > 0
    # zeros are counted, never pooled: pooled as 0, 3 errors of 5e-3 among 7 would pass
    assert GT._pass_rule([5e-3] * 3 + [0.0] * 4)["pass"]
    d = GT.k1_decision(lam, [_pi(5e-3) for _ in range(3)] + [_pi(None) for _ in range(4)])
    assert (d["pi"]["n"], d["pi"]["n_zero"]) == (3, 4) and not d["pi"]["pass"]
    assert d["pi"]["median"] == pytest.approx(5e-3) and not d["pass"]
    # the smoke run's shape: 12 non-zero comparisons and 8 zero ones
    pi = [_pi(1e-6, err_design=1e-6) for _ in range(12)] \
        + [_pi(None, plant=1.0, err_design=None) for _ in range(8)]
    d = GT.k1_decision(lam, pi)
    assert (d["pi"]["n"], d["pi"]["n_zero"]) == (12, 8) and d["pi"]["pass"]
    assert (d["pi"]["design_rule"]["n"], d["pi"]["design_rule"]["n_zero"]) == (12, 8)
    assert all((v["n"], v["n_zero"]) == (20, 0) and v["fails"]
               for v in d["pi"]["plants"].values())
    assert d["pass_with_plants"]
    # every pi comparison a zero one: pi is not decided and K1 does not pass
    dz = GT.k1_decision(lam, [_pi(None) for _ in range(4)])
    assert dz["pi"]["n"] == 0 and dz["pi"]["reason"] == "no non-zero pi comparison"
    assert not dz["pi"]["pass"] and not dz["pass"] and not dz["pass_with_plants"]
    # a pi plant cannot fail vacuously: zero comparisons only, it is not scored
    vac = [{**r, "plants": {**r["plants"], "j_w": None}} for r in pi]
    dv = GT.k1_decision(lam, vac)
    jw = dv["pi"]["plants"]["j_w"]
    assert (jw["n"], jw["n_zero"]) == (0, 20) and not jw["fails"]
    assert dv["pass"] and not dv["plants_fail"] and not dv["pass_with_plants"]
    # its zero comparisons left out, a plant still fails on its non-zero ones
    part = [{**r, "plants": {**r["plants"], "j_w": None if r["err"] is None else 0.6}}
            for r in pi]
    dp = GT.k1_decision(lam, part)
    assert (dp["pi"]["plants"]["j_w"]["n"], dp["pi"]["plants"]["j_w"]["n_zero"]) == (12, 8)
    assert dp["pi"]["plants"]["j_w"]["fails"] and dp["pass_with_plants"]
    # a pi plant that matches the re-solves does not fail
    ok = [{**r, "plants": {**r["plants"], "plus_shift": 1e-7}} for r in pi]
    assert not GT.k1_decision(lam, ok)["plants_fail"]


def _central(c, err, fwd, bwd, disc=1e-9, **kw):
    return _row(c, err, side="central", forward=fwd, backward=bwd, discrepancy=disc,
                err_design=err, **kw)


def test_k1_literal_kink_reading():
    """Section 7's kink rule to the letter (|forward - backward| > 10 discrepancy) is
    reported beside the coded one, from the one-sided estimates each row stores; it binds
    nothing. Here 40% of s's comparisons are kinks by it and none by the coded rule, so s is
    descriptive under that reading alone."""
    smooth = [_central("s", 1e-5, 1.0, 1.0 + 5e-9) for _ in range(60)]      # 5 x disc
    split = [_central("s", 1e-5, 1.0, 1.0 + 5e-8) for _ in range(40)]       # 50 x disc
    d = GT.k1_decision(smooth + split)
    s = d["components"]["s"]
    assert s["pass"] and not s["descriptive"] and s["kink_share"] == 0.0
    lk = s["design_rule_literal_kink"]
    assert lk["kink_share"] == pytest.approx(0.4) and lk["descriptive"] and lk["n"] == 60
    assert d["design_rule_literal_kink"] == {"decided": [], "descriptive": ["s"], "pass": False}
    assert d["design_rule"] == {"decided": ["s"], "descriptive": [], "pass": True}
    assert d["pass"]
    # the coded rule's noise term only removes flags: a J(0) offset within KINK_J_NOISE
    fd = TG.FD(value=1.0, fd_h=1.0, fd_h2=1.0, discrepancy=1e-9, side="central", h=1e-6,
               ok=True, iterations=1, forward=1.0, backward=1.0 + 5e-8, noise=6e-5)
    row = {"side": fd.side, "forward": fd.forward, "backward": fd.backward,
           "discrepancy": fd.discrepancy}
    assert not fd.kink and GT.literal_kink(row)
    big = dataclasses.replace(fd, backward=1.0 + 1e-3)
    assert big.kink and GT.literal_kink({**row, "backward": big.backward})
    # a one-sided (pedal-saturated) row has no kink test; a row without its one-sided
    # estimates (written before they were stored) leaves the literal reading unread
    assert GT.literal_kink({"side": "below", "forward": None, "backward": None,
                            "discrepancy": 0.5}) is False
    assert GT.literal_kink(_row("s", 1e-5, discrepancy=1e-9)) is None
    old = GT.k1_decision(smooth + [_row("s", 1e-5, discrepancy=1e-9)])
    assert old["design_rule_literal_kink"] is None and old["design_rule"]["pass"]
    assert old["components"]["s"]["design_rule_literal_kink"] is None and old["pass"]


def test_k1_rule_in_the_frozen_constants():
    """The record states K1's rule as coded: the floor's KINK_J_NOISE, the smaller c's
    SHRINK_RTOL and MAX_SHRINK, and the floor's formula, so zero_atol alone does not read as
    K1's zero threshold."""
    from apexzero.costate import prereg as PR

    t = PR.constants()["targets"]
    assert (t["kink_j_noise"], t["shrink_rtol"], t["max_shrink"]) \
        == (TG.KINK_J_NOISE, TG.SHRINK_RTOL, TG.MAX_SHRINK) == (1e-11, 1e-3, 2)
    assert (t["zero_atol"], t["kink_factor"]) == (TG.ZERO_ATOL, TG.KINK_FACTOR) == (1e-6, 10.0)
    assert "eps_t = max(zero_atol, 6 kink_j_noise / (h gamma^t))" in t["k1_rule"]
    assert "never pooled" in t["k1_rule"]
    assert (t["pi_h"], t["pi_plants"]) == (TG.PI_H, list(TG.PI_PLANTS)) and TG.PI_H == 1e-5
    assert "max(zero_atol, 6 kink_j_noise / (pi_h gamma))" in t["k1_rule"]


@needs_keys
def test_k1_on_a_stored_key_with_planted_bugs(bank):
    ref = next(r for r in ST.a2_refs() if r.key_id == EXAMPLE)
    rows = GT.k1_key(ref, bank, components=(TG.S, TG.N, TG.VX),
                     states={"decision": 0, "braking": 20, "contact": 55})
    assert len(rows) == 9 and all(r["ok"] for r in rows)
    counted = [r for r in rows if not r["kink"] and not r["unresolved"]]
    scored = [r for r in counted if r["err"] is not None]
    assert scored and max(r["err"] for r in scored) < 1e-3
    contact = [r for r in rows if r["state"] == "contact"]
    assert contact and all(r["contact"] for r in contact)
    for p in ("no_es", "no_discount", "flip_sign", "off_by_one"):
        assert max(r["plants"][p] for r in counted if r["plants"][p] is not None) > 1e-2, p
    assert max(r["plants"]["pre_jump"] for r in contact if r["plants"]["pre_jump"] is not None) \
        > 1e-2
    # the one-sided estimates are stored (Section 7's literal kink rule), and a coded kink is
    # a literal one
    assert all(np.isfinite(r["forward"]) and np.isfinite(r["backward"]) for r in rows)
    assert all(GT.literal_kink(r) is not None for r in rows)
    assert all(GT.literal_kink(r) for r in rows if r["kink"])


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
def test_k1_pi_on_a_stored_anchored_key(bank):
    """pi on one of the smoke run's anchored keys: pi_pedal (j = 1) is a zero comparison
    there (|pi| 7.1e-12, FD -1.07e-9), counted and not pooled; pi_steer matches its
    re-solves; every planted pi bug is scored on both (its FD far above the floor) and
    fails."""
    ref = next(r for v in ST.n_refs().values() for r in v if r.key_id.startswith("9b1695f1"))
    rows = GT.k1_pi(ref, bank)
    noise = 6.0 * TG.KINK_J_NOISE / (TG.PI_H * 0.995)
    assert [r["j"] for r in rows] == [0, 1] and all(r["ok"] for r in rows)
    assert all(r["noise"] == pytest.approx(noise) and r["h"] == TG.PI_H for r in rows)
    steer, pedal = rows
    assert steer["err"] < 1e-3 and steer["err_design"] == steer["err"]
    assert abs(pedal["target"]) < 1e-10 and abs(pedal["fd"]) < 1e-8
    assert pedal["err"] is None and pedal["err_design"] is None
    for r in rows:
        assert all(abs(r["plant_fd"][p]) > noise for p in TG.PI_PLANTS), r["plant_fd"]
        assert all(r["plants"][p] > 1e-2 for p in TG.PI_PLANTS), r["plants"]
    d = GT.k1_decision(_k1_passing(), rows)
    assert (d["pi"]["n"], d["pi"]["n_zero"]) == (1, 1) and d["pi"]["pass"]
    assert all(v["fails"] and v["n"] == 2 for v in d["pi"]["plants"].values())
    assert d["pass_with_plants"]


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


def _decoy_world(n_geo=48, keys=8, steps=16, seed=0):
    """A world whose decoys are structured as T's (s - s_apex along each key, vx a function
    of it and R, mirror pairs) and whose z reads them noisily, with 6 small random 'actors'
    (the host and C2's 5 controls): the decoy kNN works here, so PR-6b's known truth has a
    residualisation to test."""
    rng = np.random.default_rng(seed)
    R = np.round(np.exp(np.linspace(np.log(6.0), np.log(300.0), n_geo)), 2)
    cols = {c: [] for c in ("R", "theta", "key_id", "direction", "t", "ds_apex", "vx",
                            "pair_id")}
    for g in range(n_geo):
        vmin = min(np.sqrt(9.8 * 2.0 * R[g]), 80.0)
        for k in range(keys):
            ds = np.linspace(-150.0, 100.0, steps) + 15.0 * (k // 2)
            for c, v in (("R", R[g]), ("theta", (90.0, 180.0)[g % 2]), ("key_id", f"g{g}k{k}"),
                         ("direction", 1 if k % 2 == 0 else -1), ("pair_id", f"g{g}p{k // 2}")):
                cols[c].append(np.full(steps, v))
            cols["t"].append(5 * np.arange(steps))
            cols["ds_apex"].append(ds)
            cols["vx"].append(vmin + (85.0 - vmin) * (1.0 - np.exp(-(np.abs(ds) / 60.0) ** 2)))
    tab = {c: np.concatenate(v) for c, v in cols.items()}
    tab["width"] = np.full(len(tab["R"]), 12.0)
    n = len(tab["R"])
    ds, vx, sig, lR = tab["ds_apex"], tab["vx"], tab["direction"], np.log(tab["R"])
    phi = np.stack([ds / 100, vx / 50, sig, np.tanh(ds / 60), sig * np.tanh(ds / 60), lR / 3,
                    np.exp(-(ds / 50) ** 2)], 1)
    z = phi @ rng.normal(0.0, 0.5, (phi.shape[1], 30)) + 0.3 * rng.normal(size=(n, 30))
    lam = np.stack([np.exp(-((ds + 20) / 40) ** 2) * (1 + 0.3 * lR),
                    sig * np.tanh(ds / 30) * np.exp(-(ds / 120) ** 2),
                    sig * np.exp(-(ds / 25) ** 2) * (vx / 50),
                    np.tanh((ds + 60) / 20) * (vx / 80)], 1) + 0.3 * rng.normal(size=(n, 4))
    w = ST.weights(tab, "key")
    fo = PB.folds(tab)
    res = ST.cross_fit(ST.decoy_d1(tab), lam, fo.outer, w)
    h3 = [np.tanh(z @ rng.normal(size=(30, 24)) / 5) for _ in range(6)]
    m = [h @ rng.normal(size=(24, 2)) for h in h3]
    return GT.ProbeWorld(tab, z, fo, w, tab["direction"], lam, res.by_fold, h3, m)


@pytest.fixture(scope="module")
def decoy_world():
    pw = _decoy_world()
    G = int(pw.fo.cluster.max()) + 1
    return pw, PB.cluster_counts(G, 300, np.random.default_rng(3))


def test_k4_pr1b_one_target_array_and_its_by_fold_record(decoy_world):
    """PR-1b on one target array (each row's own-fold lambda_res) for its baseline, probe and
    controls, all four components at once, recovered within K4's band; the fold-consistent
    plant on C2's by-fold targets is recorded beside it and binds nothing."""
    pw, counts = decoy_world
    rec = GT.k4(pw, counts, which=("PR-1b",))["PR-1b"]
    assert rec["target"] == "each row's own-fold lambda_res"
    assert rec["pass"], rec
    bf = rec["by_fold"]
    assert bf["descriptive"] and len(bf["partial_vs_own"]) == len(bf["delta2"]) == 4
    assert np.allclose(bf["expected_delta2"], (1 - np.asarray(bf["r2_b"])) * 0.8)


def test_k4_pr6_one_sided_rule_and_gain_grid(decoy_world):
    """PR-6 on real lambda: biased decoy units, the grid's gain whose probe-alone R2 lies
    closest to the kNN's (every gain's R2 recorded), Delta2 <= 0.01 one-sided (its negative
    part the over-fitting cost) and PR-1b passing."""
    pw, counts = decoy_world
    r = GT.pr6(pw, counts, {"pass": True})
    assert r["gains"] == list(GT.K4_PR6_GAINS) and len(r["r2_by_gain"]) == 6
    i = int(np.argmin(np.abs(np.subtract(r["r2_by_gain"], r["knn_r2"]))))
    assert r["gain"] == GT.K4_PR6_GAINS[i] and r["decoy_r2"] == r["r2_by_gain"][i]
    assert r["calibrated"] == (abs(r["decoy_r2"] - r["knn_r2"]) <= GT.K4_CALIBRATION_TOL)
    assert r["overfit_cost"] == max(0.0, -r["delta2"])
    assert r["pass"] == (r["delta2"] <= GT.K4_ZERO_TOL), r
    # the paired increment: the unpaired Delta2 minus the host's own offset
    assert r["statistic"] == GT.K4_PAIRED
    assert r["delta2"] == pytest.approx(r["unpaired"]["delta2"] - r["host_offset"]["delta2"],
                                        abs=1e-12)
    assert r["pass"], r                       # one-sided: a negative Delta2 passes
    assert not GT.pr6(pw, counts, {"pass": False})["pass"]
    assert not GT.pr6(pw, counts, None)["pass"]


def test_k4_pr6b_known_truth_passes_and_its_planted_faults_fail(decoy_world, monkeypatch):
    """PR-6b: units holding a smooth decoy function F of y = std(F) + 0.5 AR(1) eps gain at
    most 0.01 on C2's statistic once y is residualised, beat the kNN alone, and gain more
    than 0.01 when the residualisation is off, the kNN over-smoothed or s - s_apex left out of
    D1; skipping the residualisation fails the plant."""
    pw, _ = decoy_world
    r = GT.pr6_world(pw, "smooth", seeds=(0,), t_wide=False)
    s = r["seeds"][0]
    assert r["pass"] and r["residualised_ok"] and r["off_detected"] and r["alone_ok"], s
    assert s["delta2"] <= GT.K4_ZERO_TOL < s["off"]["delta2"]
    assert s["alone_mean"] >= s["knn_mean"] and s["k"] == r["k"] == ST.KNN_K
    assert all(f["detected"] for f in r["faults"].values()), r["faults"]
    assert r["faults"]["knn_k300"]["k"] == GT.K4_OVERSMOOTH_K      # the cap does not bind
    assert r["faults"]["d1_without_ds_apex"]["k"] == ST.KNN_K
    assert r["leak_bound"] == s["delta2"] and len(r["gain_max"]) == 4
    assert r["n_controls"] == PB.N_RANDOM_ACTORS

    def skipped(X, Y, fold, w=None, k=ST.KNN_K):
        Y = np.asarray(Y, float)
        F = int(np.max(fold)) + 1
        return ST.Residuals(np.array(np.broadcast_to(Y, (F, *Y.shape))), np.asarray(fold), k)

    monkeypatch.setattr(ST, "cross_fit", skipped)
    bad = GT.pr6_world(pw, "smooth", seeds=(0,), t_wide=False, faults=False)
    assert not bad["pass"] and not bad["residualised_ok"], bad["seeds"][0]


def test_k4_pr6_world_faults_on_a_t_smaller_than_the_oversmoothed_k():
    """The over-smoothed fault's k is capped at the cross-fit's smallest training set (the
    rows outside two outer folds): on a T that small (the rehearsal's reduced T had 283
    such rows, and k = 300 crashed knn_fit) PR-6b still scores both faults."""
    fold = np.repeat(np.arange(6), [40, 50, 60, 45, 55, 50])
    assert GT.oversmooth_k(fold) == 300 - 60 - 55          # all rows less the two largest folds
    assert GT.oversmooth_k(np.repeat(np.arange(6), 1000)) == GT.K4_OVERSMOOTH_K
    pw = _decoy_world(n_geo=12, keys=4, steps=8)
    cap = GT.oversmooth_k(pw.fo.outer)
    assert cap < GT.K4_OVERSMOOTH_K
    with pytest.raises(ValueError, match="training states < k"):
        ST.cross_fit(ST.decoy_d1(pw.tab), pw.lam, pw.fo.outer, pw.w, k=GT.K4_OVERSMOOTH_K)
    ST.cross_fit(ST.decoy_d1(pw.tab), pw.lam, pw.fo.outer, pw.w, k=cap)
    r = GT.pr6_world(pw, "smooth", seeds=(0,), t_wide=False)
    assert r["faults"]["knn_k300"]["k"] == cap and r["faults"]["knn_k300"]["descriptive"]
    assert r["faults"]["d1_without_ds_apex"]["k"] == ST.KNN_K


def test_k4_pr6c_record_and_rule(decoy_world):
    """PR-6c: F the leave-one-geometry-out k = 15 kNN of lambda on D1; its verdict is every
    seed's residualised Delta2 <= 0.01 and every unresidualised one above it (the probe-alone
    comparison is recorded, not required); the leak bound is the largest Delta2."""
    pw, _ = decoy_world
    F = GT.pr6c_f(pw)
    geo = ST.geometry_index(pw.tab)
    s0 = geo == geo[0]
    D1 = ST.decoy_d1(pw.tab)
    np.testing.assert_allclose(F[s0], ST.knn_fit(D1[~s0], pw.lam[~s0], pw.w[~s0], 15)
                               .predict(D1[s0]))
    r = GT.pr6_world(pw, "knn", seeds=(0, 1), t_wide=True, faults=False)
    assert [s["seed"] for s in r["seeds"]] == [0, 1] and "faults" not in r
    assert r["pass"] == (r["residualised_ok"] and r["off_detected"])
    assert r["leak_bound"] == max(s["delta2"] for s in r["seeds"])
    for s in r["seeds"]:
        assert s["T"]["delta2"] is not None and len(s["T"]["gain"]) == 4
        assert s["off"]["delta2"] > GT.K4_ZERO_TOL
    notes = GT.band_notes({"PR-6": {"overfit_cost": 0.01, "delta2": -0.01, "ci": [-0.02, 0.0]},
                           "PR-6c": r})
    assert notes["pr6c_leak"]["bound"] == r["leak_bound"] and notes["pr6_cost"]["cost"] == 0.01


def test_c2_primary_scores_c2s_decision_rows(decoy_world):
    """PR-6b and PR-6c's statistic is C2's primary one: fitted on C2's decision stratum, each
    component scored on its decision rows, averaged over the power floor's components."""
    pw, _ = decoy_world
    n = len(pw.w)
    rows = np.asarray(pw.tab["ds_apex"] < 40.0)
    mask = np.ones((n, 4), bool)
    mask[:, 1] = False                         # n: no decision row
    mask[pw.tab["ds_apex"] > 0, 0] = False     # s only before the apex
    floor = {"s": True, "n": True, "xi": True, "vx": False}
    pwc = dataclasses.replace(pw, rows_c2=rows, mask=mask, power_floor=floor)
    rec = GT.c2_primary(pwc, pw.h3[0], pw.lam_res)
    assert rec["n_stratum"] == rows.sum()
    assert rec["scored_rows"] == (mask & rows[:, None]).sum(0).tolist()
    assert rec["passing"] == [True, False, True, False]   # n has no row, vx is below the floor
    assert rec["stat"] == pytest.approx(np.mean(np.asarray(rec["gain"])[[0, 2]]))


def _lucky(pw, Y, snr=0.3, seed=21):
    """``pw`` with its host (h3[0]) given units of ``Y`` [n, m] at a low SNR: a random network
    with a chance offset against its controls on those targets."""
    host = PB.plant_units(pw.h3[0], Y, snr, pw.w, np.random.default_rng(seed))
    return dataclasses.replace(pw, h3=[host, *pw.h3[1:]])


def test_k4_no_gain_plants_are_paired_increments(decoy_world):
    """PR-2 (both variants), PR-3's grouped split and PR-6 are scored by the paired increment
    over the host alone: on a host whose own offset against the controls exceeds 0.01 (so
    every unpaired value does too), plants that add nothing still pass; each record keeps the
    unpaired value and the host's offset, and K4's 'host_offset' gathers them."""
    pw, counts = decoy_world
    n = len(pw.w)
    rng = np.random.default_rng(5)
    tab = {**pw.tab, "a_steer": np.tanh(pw.z[:, 0] + 0.1 * rng.standard_normal(n)),
           "a_pedal": np.tanh(pw.z[:, 1] + 0.1 * rng.standard_normal(n))}
    own = np.asarray(pw.lam_res)[pw.fo.outer, np.arange(n)]
    pl = dataclasses.replace(_lucky(pw, own), tab=tab)
    out = GT.k4(pl, counts, which=("PR-2", "PR-3"))
    p2, p3 = out["PR-2"], out["PR-3"]
    for v in ("m", "astar"):
        r = p2[v]
        assert r["host_offset"]["delta2"] > GT.K4_ZERO_TOL < r["unpaired"]["delta2"], r
        assert r["delta2"] == pytest.approx(r["unpaired"]["delta2"]
                                            - r["host_offset"]["delta2"], abs=1e-12)
        assert r["delta2"] == p2[f"delta2_{v}"] and abs(r["delta2"]) <= GT.K4_ZERO_TOL
        assert len(r["host_offset"]["networks"]) == 1 + GT.K4_N_CONTROLS
    assert p2["pass"] and p2["statistic"] == GT.K4_PAIRED
    g = p3["grouped_record"]
    assert g["unpaired"]["delta2"] > GT.K4_ZERO_TOL >= p3["grouped"] == g["delta2"], g
    assert p3["by_key"] == p3["by_key_record"]["unpaired"]["delta2"]      # its rule: unpaired
    assert p3["by_key_paired"] == p3["by_key_record"]["delta2"]
    assert p3["pass"] == (max(p3["by_key"], p3["base_rise_by_key"]) > GT.K4_ZERO_TOL), p3
    ho = out["host_offset"]
    assert ho["PR-2"] == {v: p2[v]["host_offset"]["delta2"] for v in ("m", "astar")}
    assert ho["PR-3"]["grouped"] == g["host_offset"]["delta2"] and ho["descriptive"]
    r6 = GT.pr6(pl, counts, {"pass": True})
    assert r6["host_offset"]["delta2"] > GT.K4_ZERO_TOL and r6["pass"], r6
    assert r6["unpaired"]["delta2"] > r6["delta2"]


def test_k4_pr6b_host_offset_cannot_fail_it_and_a_planted_gain_still_does(decoy_world):
    """PR-6b on a host that holds part of the residualised y (its offset against C2's
    controls above 0.01, so the unpaired statistic fails): the units holding F still pass
    on the paired increment; the residualisation turned off (a planted gain) is still
    detected."""
    pw, _ = decoy_world
    F = GT.pr6b_f(pw)
    y = GT._std(F, pw.w) + GT.K4_PR6_NOISE * GT._ar1(pw.tab, pw.w, 4, GT.k4_rng(14, 0))
    res = ST.cross_fit(ST.decoy_d1(pw.tab), y, pw.fo.outer, pw.w)
    r = GT.pr6_world(_lucky(pw, res.oof), "smooth", seeds=(0,), t_wide=False, faults=False)
    s = r["seeds"][0]
    assert s["host_offset"]["delta2"] > GT.K4_ZERO_TOL < s["unpaired"]["delta2"], s
    assert s["delta2"] <= GT.K4_ZERO_TOL and r["residualised_ok"], s
    assert s["delta2"] == pytest.approx(s["unpaired"]["delta2"] - s["host_offset"]["delta2"],
                                        abs=1e-12)
    assert r["leak_bound"] == s["delta2"] and r["leak_bound_unpaired"] > GT.K4_ZERO_TOL
    assert r["host_offset"] == [s["host_offset"]["delta2"]]
    assert s["off"]["delta2"] > GT.K4_ZERO_TOL and r["off_detected"], s["off"]
    assert "unpaired" in s["off"] and r["pass"] and r["statistic"] == GT.K4_PAIRED


def test_k4_needs_pr6b_and_pr6c(decoy_world, monkeypatch):
    """K4 runs PR-6b and PR-6c with PR-6 and passes only when both do."""
    pw, counts = decoy_world
    for name in ("pr6", "pr1b_by_fold"):
        monkeypatch.setattr(GT, name, lambda *a, **k: {"pass": True})
    seen = []

    def world(pw_, kind, **kw):
        seen.append(kind)
        return {"pass": kind == "smooth"}
    monkeypatch.setattr(GT, "pr6_world", world)
    out = GT.k4(pw, counts, which=("PR-6",))
    assert seen == ["smooth", "knn"] and {"PR-1b", "PR-6", "PR-6b", "PR-6c"} <= set(out)
    assert not out["pass"]


def test_k4_pr6_constants_in_the_record():
    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR

    g = PR.constants()["gates"]
    assert g["k4_pr6"]["gains"] == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    assert g["k4_pr6"]["seeds"] == [0, 1, 2] and g["k4_controls"] == 3
    assert "one-sided" in g["k4_pr6_rule"] and "own-fold" in g["k4_pr1b_rule"]
    assert g["band_note_margin"] == AN.BAND_NOTE_MARGIN == 0.02
    assert GT.K4_PAIRED in g["k4_no_gain_rule"] and "paired" in g["k4_pr6_rule"]
    assert "one-sided" in g["k4_pr4_rule"]


def test_decoy_k_rule():
    """f_D's k (module docstring 'f_D's k'): the largest candidate whose every validation
    seed of PR-6b and PR-6c gives C2's primary Delta2 <= K4_ZERO_TOL (a missing, None or NaN
    Delta2 fails); when none does, the smallest, flagged as a blocker."""
    ok, bad = [0.004, -0.02, 0.01], [0.004, 0.0101, -0.02]
    t = {3: {"smooth": ok, "knn": ok}, 5: {"smooth": ok, "knn": ok},
         8: {"smooth": bad, "knn": ok}, 15: {"smooth": ok, "knn": [0.0, None]}}
    c = GT.decoy_k_choice(t)
    assert (c["k"], c["qualifying"], c["blocker"]) == (5, [3, 5], False)
    assert c["candidates"] == list(GT.DECOY_K_CANDIDATES) == [3, 5, 8, 15]
    t[8]["smooth"] = ok
    assert GT.decoy_k_choice(t)["k"] == 8                     # the largest, not the smallest
    assert GT.decoy_k_choice({**t, 15: {"smooth": ok, "knn": ok}})["k"] == 15
    assert GT.decoy_k_choice({**t, 8: {"smooth": [np.nan], "knn": ok}})["k"] == 5
    assert GT.decoy_k_choice({**t, 8: {"smooth": [], "knn": ok}})["k"] == 5
    assert GT.decoy_k_choice({str(k): v for k, v in t.items()})["k"] == 8   # JSON keys
    none = GT.decoy_k_choice({k: {"smooth": ok, "knn": bad} for k in GT.DECOY_K_CANDIDATES})
    assert (none["k"], none["qualifying"], none["blocker"]) == (3, [], True)


def test_decoy_k_is_the_rules_choice_on_its_own_validation_seeds(decoy_world, monkeypatch):
    """states.KNN_K is the rule's choice on the recorded validation leaks; the validation
    seeds draw on their own stream ('costate_decoy_k'), never K4's gate seeds; at_decoy_k
    re-cross-fits lambda at k and recomputes C2's power floor there (frozen_values' rule);
    decoy_k_leaks runs PR-6b's and PR-6c's construction on every validation seed at k."""
    from apexzero.costate import STREAMS, stream_rng
    from apexzero.costate import prereg as PR
    from apexzero.costate.targets import K_P

    assert ST.KNN_K == GT.decoy_k_choice(GT.DECOY_K_VALIDATION)["k"]
    assert ST.KNN_K in GT.DECOY_K_CANDIDATES and GT.DECOY_K_SEEDS == tuple(range(6))
    assert all(len(t[k][kind]) == len(GT.DECOY_K_SEEDS)
               for t in (GT.DECOY_K_VALIDATION, GT.DECOY_K_VALIDATION_UNPAIRED)
               for k in GT.DECOY_K_CANDIDATES for kind in ("smooth", "knn"))
    c = PR.constants()["states"]
    assert c["knn_k"] == ST.KNN_K and c["knn_k_candidates"] == list(GT.DECOY_K_CANDIDATES)
    assert c["knn_k_validation"]["seeds"] == list(GT.DECOY_K_SEEDS)
    assert c["knn_k_validation"]["statistic"] == GT.K4_PAIRED and "paired" in c["knn_k_rule"]
    assert c["knn_k_validation"]["delta2_unpaired"]["8"]["knn"] == list(
        GT.DECOY_K_VALIDATION_UNPAIRED[8]["knn"])
    assert c["knn_k_validation"]["stream"] == "costate_decoy_k" and "largest" in c["knn_k_rule"]
    # the stream is its own, and no validation draw is a gate draw
    assert list(STREAMS.values()).count(STREAMS["costate_decoy_k"]) == 1
    gate = [GT.k4_rng(14, s).standard_normal(16) for s in GT.K4_PR6_SEEDS]
    for v in GT.DECOY_K_SEEDS:
        a = GT.decoy_k_rng(v).standard_normal(16)
        np.testing.assert_array_equal(a, stream_rng("costate_decoy_k", v).standard_normal(16))
        assert not any(np.allclose(a, g) for g in gate)
    # at_decoy_k: lambda_res and the floor at k (on a world with an interior stratum)
    pw, _ = decoy_world
    pw = dataclasses.replace(pw, tab={**pw.tab, "interior": pw.tab["ds_apex"] < 40.0})
    pk, info = GT.at_decoy_k(pw, 5)
    res = ST.cross_fit(ST.decoy_d1(pw.tab), pw.lam, pw.fo.outer, pw.w, k=5)
    np.testing.assert_array_equal(pk.lam_res, res.by_fold)
    share, share_int = ST.decoy_shares(pw.tab, pw.lam, res.oof, pw.w)
    assert info["residual_share_interior"] == dict(zip(K_P, share_int.tolist(), strict=True))
    assert pk.power_floor == info["power_floor"] == ST.power_floor(share_int)
    assert info["power_floor_T"] == ST.power_floor(share) and info["k"] == 5
    # decoy_k_leaks: pr6_world on each validation seed, at k, on the validation stream
    calls = []

    def world(pw_, kind, seeds, t_wide=True, faults=True, k=None, noise=None):
        calls.append((kind, tuple(seeds), t_wide, faults, k, noise, pw_.power_floor))
        v = seeds[0]
        return {"seeds": [{"seed": v, "delta2": 0.001 * v, "unpaired": {"delta2": 0.002 * v},
                           "host_offset": {"delta2": 0.001 * v}}], "faults": {}}
    monkeypatch.setattr(GT, "pr6_world", world)
    out = GT.decoy_k_leaks(pw, 8, faults=True)
    assert [c_[:5] for c_ in calls] == [(kind, (v,), False, True, 8)
                                        for kind in ("smooth", "knn") for v in GT.DECOY_K_SEEDS]
    assert all(c_[5] is GT.decoy_k_rng for c_ in calls)
    assert all(c_[6] == out["real"]["power_floor"] for c_ in calls)
    # the rule reads the paired increment; the unpaired value and the host offset ride along
    assert out["delta2"] == {kind: [0.001 * v for v in GT.DECOY_K_SEEDS]
                             for kind in ("smooth", "knn")}
    assert out["delta2_unpaired"]["knn"] == [0.002 * v for v in GT.DECOY_K_SEEDS]
    assert out["host_offset"]["smooth"] == [0.001 * v for v in GT.DECOY_K_SEEDS]
    assert out["k"] == 8 and len(out["smooth"]["faults"]) == len(GT.DECOY_K_SEEDS)


def test_k5_wiring_record_and_its_rule_in_the_frozen_constants(big_world):
    """K5's verdict is the wiring check's: 10 draws averaged, each draw's point recorded, a
    component passing with its CI lower end <= 0 and point <= K4_ZERO_TOL; the record states
    the draws, the tolerance and the scheme."""
    from apexzero.costate import prereg as PR

    pw, counts = big_world
    out = GT.k5(pw, counts, ceiling="ridge")
    wn = out["wiring"]
    assert (wn["n_draws"], wn["tol"]) == (PB.WIRING_DRAWS, PB.WIRING_TOL) == (10, 0.01)
    assert PB.WIRING_TOL == GT.K4_ZERO_TOL
    assert np.shape(wn["per_draw"]) == (10, 4) and len(wn["pass_by_component"]) == 4
    ok = (np.array(wn["lo"]) <= 0) & (np.array(wn["r2"]) <= wn["tol"])
    assert wn["pass_by_component"] == ok.tolist() and out["pass"] == wn["pass"] == ok.all()
    assert out["pass"], wn
    c = PR.constants()["probes"]
    assert c["wiring"] == [10, 0.01]
    assert "within turn direction" in c["wiring_rule"] and "lower end <= 0" in c["wiring_rule"]


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
