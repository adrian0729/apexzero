"""scripts/interp_leftovers.py (IN-1 to IN-4) on synthetic rows and values: fast, and no output
file is read."""

import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from apexzero.fields import measures as MS
from apexzero.fields import stages as S

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def il():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("interp_leftovers")
    finally:
        sys.path.remove(str(SCRIPTS))


def _row(key="k0", pair=1, approach="A1", direction=1, phase=0, G=0.8, w_is_cell=False,
         branch_ok=True, c_min_branch_ok=True, on_plateau=True, valid=True):
    return {"key_id": key, "pair_id": pair, "approach": approach, "direction": direction,
            "phase": phase, "n_phases": 4, "R": 22.9, "theta_deg": 90.0, "w_is_cell": w_is_cell,
            "cap_binds": False, "G": G, "G_out": -0.1, "B": 0.5,
            "valid": {"c_min": valid}, "reasons": {"c_min": ""},
            "gate3": {"ok": True}, "kappa_clip": {"c_min": False}, "on_plateau": on_plateau,
            "branch": {"ok": branch_ok, "ok_by": {"c_min": c_min_branch_ok, "s_b": branch_ok}}}


def _opt(key="k0"):
    return {"key_id": key, "failed": False, "adjoint_ok": True, "valid": {"c_min": True},
            "inertia_failed": False, "in_band": True}


def _km(key, value, reason="", pair=1, direction=1, phase=0):
    v = float("nan") if reason else value
    return MS.KeyMeasure(key, pair, "A2", direction, phase, 4, 22.9, 90.0, v, reason)


# ------------------------------------------------------------------------- IN-1


def test_copy_rows_sets_w_is_cell_except_twins(il):
    rows = [_row("a"), _row("t")]
    out = il.copy_rows(rows, {"t"})
    assert out[0]["w_is_cell"] is True and rows[0]["w_is_cell"] is False
    assert out[1] is rows[1] and out[1]["w_is_cell"] is False


def test_copy_lets_the_frozen_edge_key_measures_keep_anchored_rows(il):
    rows, opt = [_row("a", G=0.83)], {"a": _opt("a")}
    assert S.edge_key_measures(rows, opt) == []          # the frozen filter drops it
    km = S.edge_key_measures(il.copy_rows(rows, set()), opt)
    assert len(km) == 1 and km[0].value == pytest.approx(0.83) and km[0].reason == ""


def test_hand_rule_drops_keys_whose_overall_branch_verdict_fails(il):
    rows = [_row("a", G=0.7), _row("b", G=0.9, phase=1, branch_ok=False, c_min_branch_ok=True),
            _row("c", G=0.1, phase=2, branch_ok=False, c_min_branch_ok=False)]
    opt = {r["key_id"]: _opt(r["key_id"]) for r in rows}
    km = S.edge_key_measures(il.copy_rows(rows, set()), opt)
    assert [k.reason for k in km] == ["", "", MS.PIPELINE_BRANCH]
    pk = il.per_key(km, rows)
    assert pk["frozen"] == {"n": 2, "median": pytest.approx(0.8)}
    assert pk["hand"] == {"n": 1, "median": pytest.approx(0.7)}
    assert pk["differ"] == ["b"]


def test_check_tables_passes_on_the_tables_and_fails_on_a_mismatch(il):
    def entry(n, g, b=None, t=None):
        out = {"G": {"hand": {"n": n, "median": g}, "frozen": {"n": n, "median": g}}}
        for m, v in (("B", b), ("T", t)):
            if v is not None:
                out[m] = {"hand": {"n": n, "median": v}, "frozen": {"n": n, "median": v}}
        return out

    pk = {"7.9": {a: entry(*v) for a, v in il.TABLE_7_9.items()},
          "7.8": {a: entry(*v) for a, v in il.TABLE_7_8.items()}}
    assert il.check_tables(pk)["pass"]
    pk["7.8"]["G3@134140317"]["G"]["hand"] = {"n": 191, "median": 0.8916}
    with pytest.raises(il.CheckFailed):
        il.check_tables(pk)


def test_overlap_and_wholly_above(il):
    a, b = {"lo": 0.7, "hi": 0.8}, {"lo": 0.8, "hi": 0.9}
    assert il.overlap(a, b) and il.overlap(b, a)
    assert not il.overlap(a, {"lo": 0.81, "hi": 0.9})
    assert il.wholly_above({"lo": 0.81, "hi": 0.9}, a) and not il.wholly_above(b, a)


def test_snapshot_order_sorts_by_timesteps(il):
    got = il.snapshot_order(["G1@109856054", "G2@134140317", "G1@73680629"])
    assert got == {"G1": ["G1@73680629", "G1@109856054"], "G2": ["G2@134140317"]}


def _summary(med, lo, hi):
    return {"pooled": {"median": med, "lo": lo, "hi": hi}}


def test_e1_readings(il):
    res = {"G1": {"G": _summary(0.79, 0.74, 0.83)},
           "S1": {"G": _summary(0.1, 0.0, 0.3), "coverage": {"share_excluded": 0.79,
                                                            "descriptive": True}},
           "G1@1": {"G": _summary(0.80, 0.75, 0.84)}, "G1@2": {"G": _summary(0.86, 0.85, 0.88)}}
    a1 = {"G1": _summary(0.80, 0.78, 0.82)}
    pk = {"G1": {"hand": {"median": 0.787}, "frozen": {"median": 0.787}}}
    dec = {"H2": {"G1": _summary(0.79, 0.742, 0.836)}}
    r = il.e1_readings(res, a1, pk, dec, ("G1",), ("S1",), {"G1": ["G1@1", "G1@2"]})
    assert r["E1.1"]["holds"] and r["E1.1"]["per_agent"]["G1"]["overlaps_a2"]
    assert r["E1.2"]["per_agent"]["G1"]["later_ci_wholly_above_earliest"] == ["G1@2"]
    assert not r["E1.2"]["holds"]
    assert r["E1.3"]["holds"]


# ------------------------------------------------------------------------- IN-2


def test_common_keys_and_masks(il):
    km = {"n1": [_km("a", 0.8), _km("b", 0.7), _km("c", 0.0, "argmin off plateau")],
          "n2": [_km("a", 0.9), _km("b", 0.0, "invalid outcome"), _km("c", 0.6)]}
    K = il.common_keys(km)
    assert K == {"a"}
    m = il.mask_outside(km["n1"], K)
    assert [k.reason for k in m] == ["", il.NOT_COMMON, "argmin off plateau"]
    assert np.isnan(m[1].value) and m[0].value == 0.8
    assert [k.key_id for k in il.drop_outside(km["n1"], K)] == ["a"]


def test_masking_keeps_the_pair_denominator_and_dropping_shrinks_it(il):
    keys = [_km(f"k{d}{p}", 0.8 + 0.01 * p, direction=d, phase=p)
            for d in (1, -1) for p in range(4)]
    K = {k.key_id for k in keys if k.phase < 2}                 # 4 of the pair's 8 keys
    masked = MS.pair_value(il.mask_outside(keys, K))
    dropped = MS.pair_value(il.drop_outside(keys, K))
    assert not masked.defined and masked.n_keys == 8          # 4 > 3 of 8 excluded
    assert dropped.defined and dropped.n_keys == 4 and dropped.n_excluded == 0


def test_e2_readings(il):
    def net(m, lo, hi, band="inconclusive"):
        return {"pooled": {"median": m, "lo": lo, "hi": hi}, "band_decision": band,
                "decision": band}

    nets = {f"N{i}": net(0.80 + 0.005 * i, 0.77, 0.85) for i in range(8)}
    nets["G3"] = net(0.81, 0.79, 0.84)
    dec = {"H2": {n: {"pooled": {"median": 0.79}} for n in nets}}
    r = il.e2_readings({"networks": nets}, dec)
    assert r["E2.1"]["G3_within_others_range"] and not r["E2.1"]["G3_ci_clears_0.8"]
    assert r["E2.1"]["holds"] and r["E2.2"]["n_higher"] == 9 and r["E2.3"]["holds"]


# ------------------------------------------------------------------------- IN-3


def test_trend_reading(il):
    up = il.trend([1, 2, 3, 4], [1.0, 1.5, 1.4, 2.0])
    assert up["rising"] and not up["falling"] and up["rho"] > 0
    down = il.trend([1, 2, 3, 4], [0.4, 0.3, 0.2, 0.1])
    assert down["falling"] and down["rho"] == pytest.approx(-1.0)
    flat_end = il.trend([1, 2, 3], [1.0, 2.0, 1.0])
    assert not flat_end["rising"] and not flat_end["falling"]


def _dec(pred, agent, details, verdict="descriptive", family="exploratory"):
    return {"prediction": pred, "agent": agent, "details": details, "verdict": verdict,
            "family": family}


def test_in3_tables_readings(il):
    gens = ("G1", "G2")
    ds = [_dec("P2d", "G1", {"n": 26, "frac_nearer_qss": 0.0, "median_line_excess": -0.03}),
          _dec("P2d", "G2", {"n": 20, "frac_nearer_qss": 0.6, "median_line_excess": 0.2}),
          *(_dec("P3-ratio", a, {"ratio_slope_vs_log_vb": s, "median_ratio": 1.4,
                                 "median_ratio_310": 1.4, "b_without_310": bw, "n": 10},
                 family="secondary") for a, s, bw in (("G1", 0.01, 1.06), ("G2", -0.3, 1.14))),
          *(_dec("P3", a, {"b": b}, "confirm", "secondary") for a, b in (("G1", 1.0),
                                                                          ("G2", 1.15))),
          _dec("P6d", "G1", {"large_first": False, "median_t_large": "inf",
                             "median_t_mid": 1.6e8}),
          _dec("P8-F", "G1", {"F": 0.125, "reproduced": 8, "units": 8})]
    t = il.in3_tables(il.by_prediction(ds), 0.1, gens)
    r = t["readings"]
    assert r["P2d"]["n_frac_nearer_qss_below_0.5"] == 1 and r["P2d"]["outside_b"] == ["G2"]
    assert r["P3-ratio"]["n_abs_slope_below_0.2"] == 1
    assert r["P3-ratio"]["b_shift_outside_0.05"] == ["G1"]
    assert t["P3-ratio"]["G2"]["b_shift"] == pytest.approx(-0.01)
    assert t["P6d"]["G1"]["median_t_large"] == float("inf") and r["P6d"]["holds"]
    assert r["P8-F"]["holds"]


def test_end_label_groups_merge_close_line_ends(il):
    ends = [(0.05, 200.0, "G1"), (0.049, 200.0, "G2"), (0.2, 200.0, "G3")]
    groups = il.end_label_groups(ends, 0.01)
    assert [[a for _, _, a in g] for g in groups] == [["G3"], ["G1", "G2"]]


# ------------------------------------------------------------------------- IN-4


def test_completion_min_is_h4r_rule(il):
    text = (SCRIPTS / "h4r.py").read_text()
    m = re.search(r"^COMPLETION_MIN = ([0-9.]+)", text, re.MULTILINE)
    assert m and float(m.group(1)) == il.COMPLETION_MIN


def test_completed_gaps_use_strictly_more_than_half(il):
    ev = {"A": {"completion": 0.5, "gap": 0.07}, "B": {"completion": 0.6, "gap": 0.065},
          "C": {"completion": 0.0, "gap": float("nan")}}
    assert il.completed_gaps(ev, ["A", "B", "C"]) == {"B": pytest.approx(6.5)}


def test_primary_delta(il):
    base = [{"A": 7.0, "B": 8.0}, {"A": 9.0}]
    real = [{"A": 8.5}, {"A": 9.5, "C": 5.0}]
    r = il.primary_delta(base, real)
    assert r["circuits"] == ["A"] and r["per_circuit"]["A"]["base"] == 8.0
    assert r["per_circuit"]["A"]["real"] == 9.0 and r["median_delta"] == 1.0
    assert r["n_positive"] == 1 and r["per_circuit"]["A"]["n_base"] == 2


def test_sign_flip_p(il):
    p, n = il.sign_flip_p(np.ones(10))
    assert n == 1024 and p == pytest.approx(2 / 1024)
    assert il.sign_flip_p([1.0, -1.0])[0] == 1.0


def test_boot_mean_ci_is_seeded(il):
    x = np.arange(10.0)
    assert il.boot_mean_ci(x) == il.boot_mean_ci(x)
    lo, hi = il.boot_mean_ci(np.full(10, 0.4))
    assert lo == pytest.approx(0.4) and hi == pytest.approx(0.4)


def test_paired_skips_pairs_without_a_common_circuit(il):
    pairs = {"v1/s10": ({"A": 7.0, "B": 8.0}, {"A": 7.5, "B": 9.0}),
             "v1/s11": ({"A": 7.0}, {"C": 7.0})}
    r = il.paired(pairs)
    assert r["per_pair"]["v1/s10"]["mean_diff"] == pytest.approx(0.75)
    assert r["left_out"] == ["v1/s11"] and r["n_pairs"] == 1 and r["n_sign_flips"] == 2


def test_arm_stats_and_network_medians(il):
    s = il.arm_stats([7.0, 8.0, 9.0], [8.0, 9.0, 10.0])
    assert s["diff"] == pytest.approx(1.0) and s["base_sd"] == pytest.approx(1.0)
    assert s["se_diff"] == pytest.approx(np.sqrt(2 / 3))
    g = {"A": 7.0, "B": 9.0, "C": 8.0}
    assert il.per_network_median(g) == 8.0 and il.per_network_median(g, {"A", "B"}) == 8.0


# ------------------------------------------------------------------------- common


def test_clean_and_frozen_paths(il):
    assert il.clean({"a": float("nan"), "b": [float("inf"), np.float64(1.5)]}) == \
        {"a": None, "b": ["inf", 1.5]}
    assert set(il.FROZEN) == {"src", "scripts/fields.py", "scripts/psychophysics.py"}
