"""scripts/p4_exploratory.py (PS-P4, option A): the extended-set classifier against the frozen
``p4_classify``, the K-str and W+ builders, the certification rules and the lockbox refusal.
Fast and synthetic; one test builds Hockenheim's corner 0 and skips without the track data."""

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from apexzero.interp import analysis as A
from apexzero.interp import stimuli as ST
from apexzero.interp.constants import DS, STREAM_OFFSET, STREAMS
from apexzero.interp.stimuli import COND_A, D_CONDITIONS
from apexzero.tracks.corners import R_DEV, R_TEST

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


@pytest.fixture(scope="module")
def p4():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("p4_exploratory")
    finally:
        sys.path.remove(str(SCRIPTS))


# ------------------------------------------------------------------------- planted tables


def make_design(n_corners: int = 10, n_braking: int = 8) -> dict:
    """tests/test_interp_analysis.py's planted tier D design."""
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


def _with(cond, mask, **cols):
    out = {k: np.array(v, copy=True) for k, v in cond.items()}
    for k, v in cols.items():
        out[k][mask] = v
    return out


def _eq(a, b) -> bool:
    """Equality with NaN equal to NaN, through dicts, lists and tuples."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float | np.floating) and isinstance(b, float | np.floating):
        return bool((np.isnan(a) and np.isnan(b)) or a == b)
    return a == b


def _same(a: A.P4Result, b: A.P4Result) -> None:
    assert (a.cls, a.label, a.key_factors, a.agent) == (b.cls, b.label, b.key_factors, b.agent)
    assert _eq(a.C, b.C)
    for k in ("S_spec", "phi", "m", "MI"):
        assert _eq(float(getattr(a, k)), float(getattr(b, k))), k
    mine = {k: v for k, v in a.details.items() if not k.startswith("p4a_")}
    assert _eq(mine, b.details)


def _planted() -> list[tuple[str, dict, dict, dict]]:
    """(label, specialist table, kwargs) cases: the dry run's simulated context-keyed agent
    and reader (SIM_D_OOD), the corner-keyed agent, and test_p4_other_classes' variants."""
    gen = A.simulate_conds("reader", DESIGN, np.random.default_rng(1), agent="G1")
    out = [(k, A.simulate_conds(k, DESIGN, np.random.default_rng(0), agent="S9"), gen, {})
           for k in ("context_keyed", "reader", "corner_keyed")]
    base = A.simulate_conds("reader", DESIGN, np.random.default_rng(0), agent="S9")
    cn = base["condition"]
    frag = _with(base, cn == "F1/real/real/const", complete=False, d_ood=2.0)
    frag = _with(frag, cn == "F6/wobble0.3", complete=False, d_ood=1.0)
    wk = _with(base, np.isin(cn, [c for c in A.F1_CONDS if c.endswith("/const")]),
               complete=False, d_ood=1.5)
    vac = _with(wk, np.isin(cn, A.F6_CONDS), d_ood=0.5)
    blocked = _with(wk, cn == "F6/knoise1e-3", complete=False, d_ood=3.0)
    missing = _with(wk, cn == "F6/wobble0.6", d_ood=np.nan)
    out += [("not competent", _with(base, cn == COND_A, complete=False), gen, {}),
            ("fragile", frag, gen, {}), ("width-keyed", wk, gen, {}), ("vacuous", vac, gen, {}),
            ("blocked", blocked, gen, {}), ("missing d_OOD", missing, gen, {}),
            ("mirror", _with(base, cn == "F4/mirror", complete=False), gen, {}),
            ("no generalists", wk, None, {})]
    tier_a = A._to_table([dict(tiers="A", approach="A2", R_seg=20.0, n_runs=8, det_complete=8,
                               excluded="", braking=True, no_cap=False)],
                         ("tiers", "approach", "R_seg", "n_runs", "det_complete", "excluded",
                          "braking", "no_cap"))
    tier_g = A._to_table([dict(tiers="G", det_crashes=0)], ("tiers", "det_crashes"))
    crash_g = A._to_table([dict(tiers="G", det_crashes=1)], ("tiers", "det_crashes"))
    out += [("class 4", base, gen, dict(tier_a=tier_a, tier_g=tier_g)),
            ("class 5", base, gen, dict(tier_a=tier_a, tier_g=crash_g))]
    part = {k: np.array(v, copy=True) for k, v in base.items()}
    v_a = {int(c): v for c, n, v in zip(base["corner"], cn, base["v_min"], strict=True)
           if n == COND_A}
    for i in np.flatnonzero(cn == "F5/swapB"):
        part["v_min_region"][i] = 80.0 - 0.3 * (80.0 - v_a[int(base["corner"][i])])
    out.append(("mixed", part, gen, {}))
    return out


@pytest.mark.parametrize("case", range(len(_planted())))
def test_classify_with_the_registered_set_is_the_frozen_p4_classify(p4, case):
    label, spec, gen, kw = _planted()[case]
    _same(p4.classify(spec, gen, A.F6_CONDS, **kw), A.p4_classify(spec, gen, **kw))


def test_planted_cases_cover_every_class(p4):
    got = {A.p4_classify(s, g, **kw).cls for _, s, g, kw in _planted()}
    assert got == {-1, 0, 1, 2, 3, 4, 5, 6}


def _add_condition(cond: dict, name: str, complete: bool, d_ood: float) -> dict:
    """A new F6-set condition on every corner, copied from condition (a)'s rows."""
    m = A._st(cond, "condition") == COND_A
    new = {k: np.array(v, copy=True)[m] for k, v in cond.items()}
    new["condition"] = np.full(m.sum(), name)
    new["complete"] = np.full(m.sum(), complete)
    new["d_ood"] = np.full(m.sum(), d_ood)
    return {k: np.concatenate([np.asarray(cond[k]), new[k]]) for k in cond}


def test_extended_set_decides_class_1_and_class_2(p4):
    gen = A.simulate_conds("reader", DESIGN, np.random.default_rng(1), agent="G1")
    base = A.simulate_conds("reader", DESIGN, np.random.default_rng(0), agent="S9")
    cn = base["condition"]
    wk = _with(base, np.isin(cn, [c for c in A.F1_CONDS if c.endswith("/const")]),
               complete=False, d_ood=1.5)
    vac = _with(wk, np.isin(cn, A.F6_CONDS), d_ood=0.5)
    assert A.p4_classify(vac, gen).details["vacuous"]
    ext = (*A.F6_CONDS, "P4A/x")
    # A far, completed perturbation makes class 2 non-vacuous.
    far_ok = _add_condition(vac, "P4A/x", True, 2.0)
    r = p4.classify(far_ok, gen, ext)
    assert r.cls == 2 and not r.details["vacuous"] and r.details["n_f6_at_or_above"] == 1
    # The same perturbation left out of the set: still the vacuous class 2.
    assert p4.classify(far_ok, gen, A.F6_CONDS).details["vacuous"]
    # A far perturbation that fails blocks class 2 (the specialist falls through to 6).
    far_bad = _add_condition(vac, "P4A/x", False, 2.0)
    r = p4.classify(far_bad, gen, ext)
    assert r.cls == 6 and r.details["p4a_class2_blockers"]["F1/real/real/const"] == ["P4A/x"]
    # A near perturbation that fails gives class 1, with it as the witness.
    near_bad = _add_condition(vac, "P4A/x", False, 1.2)
    r = p4.classify(near_bad, gen, ext)
    assert r.cls == 1 and r.details["p4a_class1_witnesses"] == ["P4A/x"]
    # A new condition without d_OOD makes P4 not testable, as a registered one does.
    assert p4.classify(_add_condition(vac, "P4A/x", True, np.nan), gen, ext).cls == -1


# ------------------------------------------------------------------------- builders


def test_straight_runs_footprint(p4):
    n = 1000
    spans = [(600.0, 700.0), (1100.0, 1150.0)]
    runs = p4.straight_runs(n, spans, lead1=100, i_end=900)
    i = np.arange(n)
    s = i * DS
    inside = np.zeros(n, bool)
    for a, b in runs:
        assert (b - a) * DS > p4.STRAIGHT_MIN
        inside[a:b + 1] = True
    assert not inside[i <= 100].any() and not inside[i > 900].any()
    for a, b in spans:
        assert not inside[(s >= a - 50.0) & (s <= b + 50.0)].any()
    # (100, 274] is 346 m: kept; [376, 524] (s 752-1048) is 296 m: kept; after 1200 m: kept.
    assert runs == [(101, 274), (376, 524), (601, 900)]
    assert p4.straight_runs(n, spans, lead1=100, i_end=220) == [(101, 220)]   # 238 m
    assert p4.straight_runs(n, [], lead1=0, i_end=101) == []           # 200 m: not > 200


def test_kstr_noise_zero_net_heading_peak_and_footprint(p4):
    n = 1000
    runs = [(101, 274), (376, 524)]
    rng = np.random.default_rng(0)
    on = np.zeros(n, bool)
    for a, b in runs:
        on[a + 1:b] = True                                  # the runs' interiors
    for level in p4.KSTR_LEVELS:
        noise = p4.kstr_noise(rng, n, runs, level)
        assert np.max(np.abs(noise)) == pytest.approx(level, rel=1e-12)
        for a, b in runs:
            seg = noise[a:b + 1]
            assert abs(seg.sum()) * DS < 1e-12              # zero net heading per run
            assert seg[0] == 0.0 and seg[-1] == 0.0
            assert np.max(np.abs(seg)) == pytest.approx(level, rel=1e-12)
        assert np.all(noise[~on] == 0.0)
    w = p4.run_window(n, runs)
    assert np.all(w[~on] == 0.0) and np.all(w[on] > 0.0)
    assert w.max() == 1.0 and w[101] == 0.0 and w[111] == 1.0 and w[274] == 0.0


def test_widening_only(p4):
    n = 400
    foot = np.r_[np.zeros(60), np.linspace(0, 1, 11), np.ones(n - 71)]
    for a in p4.WPLUS_AMPL:
        add = p4.widening(n, a, 1.3, foot)
        assert np.all(add >= 0.0) and add.max() <= a + 1e-12
        assert np.all(add[:60] == 0.0)                      # nothing in the lead-in
        assert add.max() == pytest.approx(a, rel=1e-2)       # sin sampled every 12 deg
    kappa = np.zeros(n)
    wl, wr = np.full(n, 6.0), np.full(n, 5.0)
    add = p4.widening(n, 2.0, 0.4, foot)
    wl2, wr2, cap = p4.widen(kappa, wl, wr, add)
    assert np.array_equal(wl2, wl + add) and np.array_equal(wr2, wr + add) and not cap.any()
    # The cap binds on a tight left turn: it may trim the widening, never the road.
    kappa[200:220] = 0.3                                   # cap 1 + 0.6 / 0.3 = 3 m
    wl = np.where(kappa > 0, 2.5, 6.0)
    wl2, wr2, cap = p4.widen(kappa, wl, wr, add)
    assert np.all(wl2 >= wl) and np.all(wr2 >= wr) and np.all(wl2[200:220] <= 3.0 + 1e-12)
    wl_over = np.where(kappa > 0, 4.0, 6.0)                # a road already over the cap
    if (add[200:220] > 0).any():
        with pytest.raises(ValueError, match="narrowed"):
            p4.widen(kappa, wl_over, wr, add)
    with pytest.raises(ValueError, match="negative"):
        p4.widen(kappa, wl, wr, -add)


def test_new_stream_and_catalogue(p4):
    frozen = {STREAM_OFFSET + v for v in STREAMS.values()}
    assert STREAM_OFFSET + p4.P4_STREAM not in frozen
    assert p4.P4_STREAM not in range(100, 400)              # fields' 101-105, costate's 3xx
    a = p4.p4_rng(0, "Kstr", 0).normal(size=4)
    assert not np.allclose(a, ST._f6_rng(0, "F6/knoise1e-3").normal(size=4))
    assert not np.allclose(a, p4.p4_rng(0, "Wplus", 0).normal(size=4))
    cat = p4.CATALOGUE
    assert len(cat) == 38 and len({c["name"] for c in cat}) == 38
    assert sum(not c["contingency"] for c in cat) == 33
    f6s = [c for c in cat if c["family"] == "F6s"]
    wob = sorted({c["amplitude"] for c in f6s if c["kind"] == "wobble"})
    kn = sorted({c["amplitude"] for c in f6s if c["kind"] == "knoise"})
    assert wob[0] == pytest.approx(0.3) and wob[-1] == pytest.approx(4.8)
    assert kn[0] == pytest.approx(0.5e-3) and kn[-1] == pytest.approx(8e-3)
    assert len(p4.active_conditions(None, None)) == 33
    cut = p4.active_conditions({"cut": True}, {"triggered": True})
    assert sum(p4.BY_NAME[c]["family"] == "F6s" for c in cut) == 16
    assert sum(p4.BY_NAME[c]["contingency"] for c in cut) == 5


# ------------------------------------------------------------------------- certification


def test_socp_certification_rule(p4):
    assert p4.socp_certified(30.29, 30.0, True, True)         # within 1%
    assert p4.socp_certified(29.71, 30.0, True, True)
    assert not p4.socp_certified(29.69, 30.0, True, True)
    assert not p4.socp_certified(30.31, 30.0, True, True)
    assert not p4.socp_certified(30.0, 30.0, False, True)     # not converged
    assert not p4.socp_certified(30.0, 30.0, True, False)     # failed its checks
    assert not p4.socp_certified(float("nan"), 30.0, True, True)


def test_generalist_rule_fallback_corner_count_and_contingency(p4):
    c = p4.generalist_criterion(1.0)
    assert not c["fallback"] and c["threshold"] == 0.9
    c = p4.generalist_criterion(8 / 9)
    assert c["fallback"] and c["threshold"] == pytest.approx(8 / 9 - 0.1)
    assert p4.enters_p4(True, list(range(7))) and not p4.enters_p4(True, list(range(6)))
    assert not p4.enters_p4(False, list(range(9)))
    ok = {"socp_corners": list(range(9)), "passes_generalists": True}
    few = {"socp_corners": list(range(6)), "passes_generalists": True}
    gen_fail = {"socp_corners": list(range(9)), "passes_generalists": False}
    name = "P4A/Wplus/{:g}m".format
    st = {name(a): few for a in (1, 2, 3, 4)} | {name(0.5): ok}
    assert p4.contingency_rule(st)["triggered"]               # A = 0.5 does not count
    st[name(3)] = gen_fail
    assert p4.contingency_rule(st)["triggered"]
    st[name(2)] = ok
    assert not p4.contingency_rule(st)["triggered"]


def test_lockbox_refusal(p4):
    for name in (R_TEST[0], f"{R_TEST[4]}_mirrored", f"{R_TEST[-1]}-reversed"):
        with pytest.raises(PermissionError):
            p4.hockenheim_track(name)
    other = next(n for n in R_DEV if n != "Hockenheim")
    with pytest.raises(ValueError, match="Hockenheim only"):
        p4.hockenheim_track(other)


# ------------------------------------------------------------------------- Hockenheim


@pytest.mark.skipif(not (ROOT / "data" / "tracks" / "tumftm" / "Hockenheim.csv").exists(),
                    reason="needs the TUMFTM track data (not in the repo)")
def test_hockenheim_corner_conditions(p4):
    track = p4.hockenheim_track()
    dc = ST.hockenheim_corners(track)[0]
    prof, info = p4.new_conditions(dc, track, None)
    assert set(prof) == set(p4.BY_NAME)
    rr = ST._RealRoad(ST.D_TRACK, dc.corner, dc.before, dc.after, ST.N_KNOTS, track)
    k, wl, wr = rr.real()
    conds2, _ = ST.hockenheim_conditions(0, track, None, 2.0)
    p = prof["P4A/F6s/wobble0.6/x2"]
    assert np.array_equal(p.w_left, conds2["F6/wobble0.6"].w_left)
    i = np.arange(ST.N_KNOTS)
    lm = rr.landmarks(k)
    region = (i * DS >= lm.s_c0 - 50) & (i * DS <= lm.s_c1 + 50)
    for name in ("P4A/Kstr/1e-3", "P4A/Kstr/8e-3"):
        q = prof[name]
        d = q.kappa - k
        assert np.all(d[region] == 0.0) and np.all(d[i <= info["lead1"]] == 0.0)
        assert np.all(d[i > info["i_b"]] == 0.0) and np.abs(d).max() > 0
        assert np.array_equal(q.w_left, wl) or q.capped.any()
    for fam in ("Wplus", "WplusStr"):
        for a in p4.WPLUS_AMPL:
            q = prof[f"P4A/{fam}/{a:g}m"]
            assert np.array_equal(q.kappa, k)
            assert np.all(q.w_left >= wl) and np.all(q.w_right >= wr)
            assert np.all(q.w_left[i <= info["lead1"]] == wl[i <= info["lead1"]])
    wstr = prof["P4A/WplusStr/4m"]
    assert np.all(wstr.w_left[region] == wl[region])
    # Amendment 2's builder: the same footprint, and the planned rungs rebuilt bit for bit.
    eprof, einfo = p4.extension_conditions(dc, track)
    assert einfo["straight_runs"] == info["straight_runs"]
    runs = [tuple(r) for r in info["straight_runs"]]
    mod = rr.w_lead() < 1.0
    for li, (lv, tag) in enumerate(zip(p4.KSTR_LEVELS, p4.KSTR_TAGS, strict=True)):
        q = p4.kstr_profile(0, k, wl, wr, lm, mod, runs, lv, li)
        assert np.array_equal(q.kappa, prof[f"P4A/Kstr/{tag}"].kappa)
    for c in p4.extension_catalogue():
        d = eprof[c["name"]].kappa - k
        assert np.all(d[region] == 0.0) and np.abs(d).max() == pytest.approx(c["amplitude"])


def test_extension_catalogue_and_configure(p4):
    ext = p4.extension_catalogue()
    assert [c["amplitude"] for c in ext] == [16e-3, 32e-3, 64e-3]
    assert [c["level_index"] for c in ext] == [4, 5, 6]          # continues K-str's ladder
    assert all(c["family"] == "Kstr" and c["ladder"] == "Kstr" for c in ext)
    assert not {c["name"] for c in ext} & set(p4.BY_NAME)
    try:
        p4.configure(True)
        assert p4.BUILD == p4.OUT / "extension" / "build" and p4.EXTENSION
        assert set(p4.BY_NAME) == {c["name"] for c in ext}
        assert p4.active_conditions(None, None) == [c["name"] for c in ext]
    finally:
        p4.configure(False)
    assert p4.BUILD == p4.OUT / "build" and len(p4.CATALOGUE) == 38 and not p4.EXTENSION
