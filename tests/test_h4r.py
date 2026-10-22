"""H4R's script (scripts/h4r.py; docs/interp/h4r-design.md), tested before the tag on synthetic
rows, stand-in run directories and temporary git repositories: no trained network is loaded,
run or differentiated here.

- The decisions: the exact within-variant permutation test against brute-force enumeration
  (63,504 relabellings for 5 vs 5, every relabelling of the remaining sets when networks
  drop), its direction, 'not testable' below 3 per arm; Holm's step-down over the two primary
  tests, with m = 2 when one is not testable (values written out by hand); the paired
  sign-flip test (1,024 flips) and Spearman's rho with its bootstrap (stream 201).
- Each network's G, B and T equal to planted values (G_out differing per network) with T's
  labels; S2 for G and B in their directions; S3's rho against scipy on the kept networks.
- H4R's gate2-trained rule (0, 1, 2 and 3 failing keys): failing keys excluded as pipeline
  exclusions counted in coverage, a network descriptive only above 2; missing and descriptive
  networks dropped and listed with their reasons, through the whole decision function.
- The manifest: config.json against design Section 2, the final checkpoint (its timesteps and
  name) and step 2's loader resolving it, the evaluation, the queue log, every missing run
  listed with its reason; --accept-missing naming each run (RUN=REASON), a run finished on the
  box refused until named, no missing run without the queue log.
- The pre-registration and the freeze in temporary git repositories: the prereg stage's
  refusals and its record; the clean-tree rule, the record built from step 2's, the annotated,
  pushed tag, the bound files at HEAD and on disk, the record's networks, missing and accepted
  runs and design block against the tagged manifest, the record rebuilt, the base commit, the
  record version, the write-once tag pin (named by the tag), the run-once rule; the post-tag
  gate refusing before ``unlock_trained``; re-hashing the recorded checkpoints and evaluations.
- The analysis stage end to end on stamped synthetic inputs (the freeze check replaced by a
  record): refusing incomplete, partial, unstamped or foreign inputs and writing its decisions
  once.
- harness-f32, gate2-trained and fields end to end on a RANDOM actor's checkpoint (one A2 key,
  one gate2-trained key; in a subprocess, since x64 and the trained-network lock are
  process-wide): stamped outputs the analysis accepts, and a resumed second fields run.
"""

import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from apexzero.fields import analysis as A
from apexzero.fields import measures as MS
from apexzero.fields import prereg
from apexzero.fields import stages as S

ROOT = Path(__file__).resolve().parents[1]


def _load():
    """scripts/h4r.py as a module (nothing runs on import)."""
    path = ROOT / "scripts" / "h4r.py"
    spec = importlib.util.spec_from_file_location("h4r_script", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


H = _load()
F = H.F
TAG_ID = {"tag": H.TAG, "object": "a" * 40, "type": "tag", "commit": "b" * 40}


# ------------------------------------------------------------------------- the tests


def _brute(values, kept, direction):
    """The within-variant permutation test written out: every joint relabelling."""
    per, obs = [], 0.0
    for v in H.VARIANTS:
        real = [values[x] for x in kept[v][H.REAL]]
        base = [values[x] for x in kept[v][H.BASE]]
        obs += np.mean(real) - np.mean(base)
        pooled, k = real + base, len(real)
        per.append([np.mean([pooled[i] for i in idx])
                    - np.mean([pooled[i] for i in range(len(pooled)) if i not in idx])
                    for idx in itertools.combinations(range(len(pooled)), k)])
    up = down = count = 0
    for combo in itertools.product(*per):
        t = sum(combo)
        count += 1
        up += t >= obs - 1e-9
        down += t <= obs + 1e-9
    p, q = (up / count, down / count) if direction > 0 else (down / count, up / count)
    return count, p, q


def _kept(drop=()):
    kept = {v: {a: [H.run_name(v, a, s) for s in H.SEEDS if H.run_name(v, a, s) not in drop]
                for a in (H.REAL, H.BASE)} for v in H.VARIANTS}
    return kept


def test_names_and_fixed_numbers():
    assert len(H.NETWORKS) == 20 and len(set(H.NETWORKS)) == 20
    assert H.parse_name("runs/h4r-v3-realw-s12") == ("v3", "realw", 12)
    with pytest.raises(ValueError):
        H.parse_name("runs/dr-realw-v1-s0")
    assert H.FINAL_TIMESTEPS == 199_884_800
    d = H.design_block()
    assert d["permutation"]["relabellings_5v5"] == 63_504
    assert d["holm"] == {"alpha": 0.05, "family": ["H4R-G", "H4R-B"], "m": 2,
                         "not_testable_p": 1.0, "confirm": d["holm"]["confirm"]}
    # the statistic in step 2's words (stages.median_statistic), in the record and the script
    words = "median over geometries of the per-geometry medians of its mirror-pair values"
    assert words in d["statistic"] and "per-geometry medians" in H.__doc__


@pytest.mark.parametrize("drop", [(), ("runs/h4r-v1-realw-s10",),
                                  ("runs/h4r-v1-base-s11", "runs/h4r-v3-realw-s13",
                                   "runs/h4r-v3-realw-s14")])
def test_permutation_test_against_brute_force(drop):
    rng = np.random.default_rng(len(drop))
    values = {n: float(rng.normal()) for n in H.NETWORKS}
    kept = _kept(drop)
    for direction in (1, -1):
        res = H.within_variant_test(values, kept, direction)
        count, p, q = _brute(values, kept, direction)
        assert res["testable"] and res["n_relabellings"] == count
        assert res["p"] == pytest.approx(p) and res["p_opposite"] == pytest.approx(q)
    sizes = {v: (len(kept[v][H.REAL]), len(kept[v][H.BASE])) for v in H.VARIANTS}
    from math import comb

    assert res["n_relabellings"] == np.prod([comb(r + b, r) for r, b in sizes.values()])
    if not drop:
        assert res["n_relabellings"] == 63_504
        assert all(s["n_relabellings"] == 252 for s in res["strata"].values())


def test_permutation_direction_ties_and_minimum_p():
    # real below baseline in both variants: H4R-G's p is the smallest possible
    values = {n: (0.2 if H.parse_name(n)[1] == H.REAL else 0.4) + 0.001 * H.parse_name(n)[2]
              for n in H.NETWORKS}
    g = H.within_variant_test(values, _kept(), -1)
    b = H.within_variant_test(values, _kept(), 1)
    assert g["p"] == pytest.approx(1 / 63_504) and g["p_opposite"] == 1.0
    assert b["p"] == 1.0 and b["p_opposite"] == pytest.approx(1 / 63_504)
    assert g["strata"]["v1"]["p"] == pytest.approx(1 / 252)
    # integer values (circuits completed) with ties: as the brute force counts them
    rng = np.random.default_rng(3)
    ints = {n: float(rng.integers(5, 9)) for n in H.NETWORKS}
    res = H.within_variant_test(ints, _kept(), 1)
    count, p, q = _brute(ints, _kept(), 1)
    assert (res["n_relabellings"], res["p"], res["p_opposite"]) == (count, pytest.approx(p),
                                                                    pytest.approx(q))


def test_not_testable_below_three_per_arm():
    values = {n: 0.1 for n in H.NETWORKS}
    drop = {H.run_name("v3", H.REAL, s) for s in (10, 11, 12)}
    res = H.within_variant_test(values, _kept(drop), -1)
    assert not res["testable"] and "v3/realw: 2" in res["why"]
    assert res["n"]["v3"][H.REAL] == 2 and res["n"]["v1"][H.REAL] == 5
    assert H.primary_decision(res, {"reject": False}).startswith("not testable (fewer than 3")
    # three remain: testable, every relabelling of the remaining sets (C(8,3) x C(10,5))
    drop2 = {H.run_name("v3", H.REAL, s) for s in (10, 11)}
    res = H.within_variant_test({n: float(i) for i, n in enumerate(H.NETWORKS)}, _kept(drop2), -1)
    assert res["testable"] and res["n_relabellings"] == 56 * 252


def test_kept_sets_list_every_dropped_network():
    values = {n: 1.0 for n in H.NETWORKS}
    values[H.run_name("v1", H.BASE, 12)] = float("nan")
    del values[H.run_name("v3", H.BASE, 13)]
    status = {H.run_name("v1", H.REAL, 10): "missing: no run directory",
              H.run_name("v3", H.REAL, 11): "descriptive (gate2-trained)"}
    kept, dropped = H.kept_sets(values, status)
    assert dropped == {H.run_name("v1", H.REAL, 10): "missing: no run directory",
                       H.run_name("v3", H.REAL, 11): "descriptive (gate2-trained)",
                       H.run_name("v1", H.BASE, 12): "statistic undefined",
                       H.run_name("v3", H.BASE, 13): "statistic undefined"}
    assert sum(len(x) for d in kept.values() for x in d.values()) == 16


def _holm_brute(ps, alpha=0.05):
    """Holm by its definition: reject H_(1..k) for the largest k with p_(j) <= alpha/(m-j+1)
    for every j <= k."""
    m = len(ps)
    order = sorted(ps, key=lambda k: (1.0 if ps[k] is None else ps[k], k))
    out, ok = {}, True
    for j, k in enumerate(order):
        p = 1.0 if ps[k] is None else ps[k]
        ok = ok and p <= alpha / (m - j)
        out[k] = ok
    return out


@pytest.mark.parametrize("ps", [{"G": 0.01, "B": 0.04}, {"G": 0.03, "B": 0.04},
                                {"G": 0.02, "B": 0.3}, {"G": 0.04, "B": 0.01},
                                {"G": 0.01, "B": None}, {"G": None, "B": None},
                                {"G": 0.03, "B": 0.03}, {"G": 0.025, "B": 0.05},
                                {"G": 0.026, "B": 0.001}])
def test_holm(ps):
    h = H.holm(ps)
    want = _holm_brute(ps)
    assert {k: v["reject"] for k, v in h.items()} == want
    assert all(v["reject"] == (v["p_adjusted"] <= 0.05) for v in h.values())
    assert all(v["testable"] == (ps[k] is not None) for k, v in h.items())
    if ps == {"G": 0.01, "B": 0.04}:
        assert h["G"]["p_adjusted"] == pytest.approx(0.02) and h["B"]["level"] == 0.05
    if ps == {"G": 0.03, "B": 0.04}:          # the first fails at 0.025: neither is rejected
        assert h["G"]["p_adjusted"] == pytest.approx(0.06) == h["B"]["p_adjusted"]


def test_holm_counts_an_untestable_test_in_m():
    """The design's family is both primary tests (m = 2) even when one is not testable: it
    enters as p = 1, so the other is held to 0.025, not 0.05 (values written out by hand)."""
    d = H.design_block()["holm"]
    for g, b in (("H4R-G", "H4R-B"), ("H4R-B", "H4R-G")):
        h = H.holm({g: 0.03, b: None})
        assert h[g]["level"] == pytest.approx(0.025) == d["alpha"] / d["m"]
        assert h[g]["p_adjusted"] == pytest.approx(0.06)
        assert h[g]["reject"] is False and h[g]["testable"] is True and h[g]["rank"] == 1
        assert h[b]["p"] == 1.0 == d["not_testable_p"]
        assert h[b]["testable"] is False and h[b]["reject"] is False
    assert H.holm({"G": 0.025, "B": None})["G"]["reject"] is True
    assert H.holm({"G": 0.0251, "B": None})["G"]["reject"] is False
    # the decision this changes: p 0.03 with the other test out is inconclusive, not confirmed
    t = {"testable": True, "p": 0.03, "p_opposite": 0.97}
    assert H.primary_decision(t, H.holm({"H4R-G": 0.03, "H4R-B": None})["H4R-G"]) == \
        "inconclusive"


def test_analyse_h4r_holds_one_test_to_m_2_when_the_other_drops(monkeypatch):
    """End to end through analyse_h4r: H4R-B not testable, H4R-G's p 0.03 -> inconclusive."""
    real = H.within_variant_test

    def fake(values, kept, direction):
        res = real(values, kept, direction)
        if direction > 0:
            return {**res, "testable": False, "why": "planted"}
        return {**res, "p": 0.03, "p_opposite": 0.97}

    monkeypatch.setattr(H, "within_variant_test", fake)
    rows = _all_rows()
    opt = _opt_rows(rows[H.NETWORKS[0]])
    res = H.analyse_h4r(rows, opt, {}, dict.fromkeys(rows, 6), {}, None, None, n=100)
    assert res["H4R_B"]["decision"] == "not testable (planted)"
    assert res["holm"]["tests"]["H4R-G"]["level"] == pytest.approx(0.025)
    assert res["H4R_G"]["decision"] == "inconclusive"


def test_primary_decision_bands():
    t = {"testable": True, "p": 0.001, "p_opposite": 0.999}
    assert H.primary_decision(t, {"reject": True}) == "confirm"
    assert H.primary_decision({**t, "p": 0.04, "p_opposite": 0.97}, {"reject": False}) \
        == "inconclusive"
    assert H.primary_decision({**t, "p": 0.97, "p_opposite": 0.04}, {"reject": False}) \
        == "refute"
    assert H.primary_decision({**t, "p": 0.95, "p_opposite": 0.05}, {"reject": False}) \
        == "refute"                              # opposite p <= 0.05, unadjusted
    assert H.primary_decision({"testable": False, "why": "x"}, {"reject": False}) \
        == "not testable (x)"


def test_sign_flip_test_against_brute_force():
    rng = np.random.default_rng(5)
    d = rng.normal(size=10)
    res = H.sign_flip_test(d, -1)
    obs = d.mean()
    flips = [np.mean(np.array(s) * d) for s in itertools.product((1, -1), repeat=10)]
    assert res["n_flips"] == 1024
    assert res["p"] == pytest.approx(np.mean([f <= obs + 1e-12 for f in flips]))
    assert res["p_opposite"] == pytest.approx(np.mean([f >= obs - 1e-12 for f in flips]))
    neg = H.sign_flip_test(-np.abs(d), -1)
    assert neg["p"] == pytest.approx(1 / 1024) and neg["p_opposite"] == 1.0


def test_paired_test_uses_complete_pairs():
    values = {n: (0.2 if H.parse_name(n)[1] == H.REAL else 0.3) for n in H.NETWORKS}
    kept = _kept({H.run_name("v1", H.BASE, 12), H.run_name("v3", H.REAL, 14)})
    res = H.paired_test(values, kept, -1)
    assert res["pairs"] == {"v1": [10, 11, 13, 14], "v3": [10, 11, 12, 13]}
    assert res["n_pairs"] == 8 and res["n_flips"] == 256
    assert res["p"] == pytest.approx(1 / 256) and res["decision"] == "confirm"
    kept = _kept({H.run_name("v1", H.BASE, s) for s in (10, 11, 12)})
    assert not H.paired_test(values, kept, -1)["testable"]


def test_spearman_and_its_bootstrap():
    from scipy import stats

    rng = np.random.default_rng(0)
    x, y = rng.normal(size=10), rng.integers(0, 6, size=10).astype(float)
    assert H.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic)
    assert np.isnan(H.spearman(x, np.ones(10))) and np.isnan(H.spearman(x[:2], y[:2]))
    ci = H.spearman_ci(np.arange(10.0), np.arange(10.0) ** 2, 2000, np.random.default_rng(1))
    assert ci["rho"] == pytest.approx(1.0) and ci["hi"] == pytest.approx(1.0)
    assert ci["lo"] <= 1.0 and ci["n"] == 10
    a = H.spearman_ci(x, y, 500, H._s3_rng())
    b = H.spearman_ci(x, y, 500, H._s3_rng())
    assert a == b and a["lo"] <= a["rho"] <= a["hi"]


# ------------------------------------------------------------------------- gate rule


def _net_rows(G, B, n_geo=6, n_phase=2, invalid=(), loop=(), seed=0, G_out=0.5):
    """Synthetic fields rows of one network: n_geo geometries x 2 directions x n_phase
    phases (key ids k<geo><dir><phase>); keys in ``invalid`` have an invalid c_min, keys in
    ``loop`` are pipeline failures (the loop exhausted)."""
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_geo):
        for sig in (1, -1):
            for ph in range(n_phase):
                k = f"k{g}{sig}{ph}"
                meta = {"key_id": k, "pair_id": g, "approach": "A2", "direction": sig,
                        "phase": ph, "n_phases": n_phase, "R": 10.0 * 1.6 ** g,
                        "theta_deg": 90.0, "cell": 100 + g, "w_is_cell": True,
                        "cap_binds": False}
                if k in loop:
                    out.append({**meta, "pipeline": S.PIPELINE_LOOP, "error": "x"})
                    continue
                ok = k not in invalid
                gv = float(G + 0.002 * rng.standard_normal())
                out.append({**meta, "valid": {"c_min": ok, "v_apex": ok, "s_b": ok,
                                              "steer": True},
                            "reasons": {"c_min": "" if ok else "crash", "v_apex": "",
                                        "s_b": "", "steer": ""},
                            "gate3": {"ok": True},
                            "branch": {"ok": True, "ok_by": dict.fromkeys(
                                ("c_min", "v_apex", "s_b", "steer"), True)},
                            "kappa_clip": dict.fromkeys(("c_min", "v_apex", "s_b", "steer"),
                                                        False),
                            "on_plateau": True, "G": gv if ok else None,
                            "G_out": float(G_out + 0.002 * rng.standard_normal()) if ok else None,
                            "B": float(B + 0.002 * rng.standard_normal()) if ok else None})
    return out


def _opt_rows(rows):
    return {r["key_id"]: {"key_id": r["key_id"], "kind": "nominal", "failed": False,
                          "adjoint_ok": True, "inertia_failed": False, "in_band": True,
                          "valid": {"c_min": True}, "w_is_cell": True, "cap_binds": False,
                          **{k: r[k] for k in ("pair_id", "approach", "direction", "phase",
                                               "n_phases", "R", "theta_deg")},
                          "G_star": 1.0, "T_star": 1.0}
            for r in rows}


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_gate_rule_and_its_exclusions(k):
    rows = _net_rows(0.3, 0.5, invalid={"k110"}, loop={"k210"})
    opt = _opt_rows(rows)
    keys = ["k010", "k110", "k210", "k31-11"][:k]
    result = {"rows": [{"key_id": r["key_id"], "pass": r["key_id"] not in keys}
                       for r in rows[:12]] + [{"key_id": "k5-11", "pass": True}]}
    v = H.gate_verdict(result)
    assert v["failing_keys"] == sorted(keys) and v["n_failing"] == k
    assert v["descriptive"] is (k == 3)
    km = H.h4r_key_measures(rows, opt, "G", keys)
    by = {m.key_id: m for m in km}
    for key in keys:
        # step 2's pipeline failure keeps its reason; otherwise the gate's comes first
        want = MS.PIPELINE_LOOP if key == "k210" else H.GATE2_REASON
        assert by[key].reason == want and np.isnan(by[key].value)
    assert by["k110"].reason == (H.GATE2_REASON if "k110" in keys else "invalid outcome")
    cov = MS.coverage(A.coverage_records(km, "n", "final", "G"))["n@final"]["G"]
    n_excl = len(set(keys) | {"k110", "k210"})
    assert cov["n"] == 24 and cov["share_excluded"] == pytest.approx(n_excl / 24)
    assert cov["excluded"].get(H.GATE2_REASON, 0) == len(set(keys) - {"k210"})
    # an exhausted gate2-trained loop is a failing key too
    ex = H.gate_verdict({"rows": [{"key_id": "a", "pass": False, "exhausted": True},
                                  {"key_id": "b", "pass": True}]})
    assert ex["failing_keys"] == ["a"] and not ex["descriptive"]


def _all_rows(drop=(), seed=0):
    out = {}
    for i, name in enumerate(H.NETWORKS):
        if name in drop:
            continue
        v, arm, s = H.parse_name(name)
        G = (0.2 if arm == H.REAL else 0.35) + 0.01 * (s - 10)
        B = (0.6 if arm == H.REAL else 0.5) + 0.01 * (s - 10)
        out[name] = _net_rows(G, B, seed=seed + i)
    return out


def test_analyse_h4r_decisions_with_dropped_networks():
    missing = {H.run_name("v1", H.BASE, 12): "no eval_rdev.json (training did not finish)"}
    rows = _all_rows(drop=missing)
    cov_net = H.run_name("v1", H.REAL, 13)
    rows[cov_net] = _net_rows(0.2, 0.6, invalid={f"k{g}{d}0" for g in range(4)
                                                  for d in (1, -1)})       # 8 of 24 > 30%
    gate_net, two_net = H.run_name("v3", H.REAL, 14), H.run_name("v3", H.BASE, 10)
    failing = {gate_net: ["k010", "k110", "k210"], two_net: ["k010", "k110"]}
    opt = _opt_rows(rows[H.NETWORKS[0]])
    completion = {n: (8 if H.parse_name(n)[1] == H.REAL else 3) + (H.parse_name(n)[2] - 10)
                  for n in rows}
    cells = {v: {"bulk": [100, 101, 102], "tight": [103, 104, 105]} for v in H.VARIANTS}
    s2 = {"statistics": {n: 0.7 for n in A.h4_networks()},
          "edge_descriptive": {n: {"B": {"pooled": {"median": 1.5}},
                                   "T": {"pooled": {"median": 0.8}}} for n in A.h4_networks()},
          "descriptive": {"runs/dr-realw-v1-s1": "gate2-trained"}}
    res = H.analyse_h4r(rows, opt, failing, completion, missing, cells, s2, n=200)
    assert res["status"][cov_net] == "descriptive (coverage)"
    assert res["status"][gate_net] == "descriptive (gate2-trained)"
    assert res["status"][two_net] == "kept"                  # 2 of 12: excluded keys only
    assert res["networks"][two_net]["coverage"]["gate2_keys"] == ["k010", "k110"]
    assert res["networks"][two_net]["coverage"]["share_excluded"] == pytest.approx(2 / 24)
    assert res["status"][next(iter(missing))].startswith("missing: no eval_rdev.json")
    g, b = res["H4R_G"], res["H4R_B"]
    assert set(g["dropped"]) == {cov_net, gate_net, *missing}
    assert g["n"] == {"v1": {H.REAL: 4, H.BASE: 4}, "v3": {H.REAL: 4, H.BASE: 5}}
    assert g["n_relabellings"] == 70 * 126 == b["n_relabellings"]
    assert g["p"] == pytest.approx(1 / 8820) and b["p"] == pytest.approx(1 / 8820)
    assert g["decision"] == "confirm" == b["decision"]
    assert res["holm"]["tests"]["H4R-G"]["reject"] and g["direction"] == "real < baseline"
    s1 = res["secondary"]["S1"]                              # every present network
    assert s1["n"] == {"v1": {H.REAL: 5, H.BASE: 4}, "v3": {H.REAL: 5, H.BASE: 5}}
    assert s1["testable"] and s1["decision"] == "confirm"
    s2g = res["secondary"]["S2"]["G"]
    assert s2g["pairs"] == {"v1": [10, 11, 14], "v3": [10, 11, 12, 13]}
    assert s2g["n_flips"] == 128 and s2g["decision"] == "confirm"
    assert s2g["p"] == pytest.approx(1 / 128) and s2g["mean_diff"] < 0
    s2b = res["secondary"]["S2"]["B"]                        # B's pairs, in B's direction
    assert s2b["pairs"] == {"v1": [10, 11, 14], "v3": [10, 11, 12, 13]}
    assert s2b["n_pairs"] == 7 and s2b["n_flips"] == 128 and s2b["mean_diff"] > 0
    assert s2b["p"] == pytest.approx(1 / 128) and s2b["decision"] == "confirm"
    s3 = res["secondary"]["S3"]
    assert len(s3[H.REAL]["pooled"]["networks"]) == 8 and "v1" in s3[H.BASE]
    # S3's rho against scipy on the kept networks' (B, completed), every arm and label
    from scipy import stats

    for arm in (H.REAL, H.BASE):
        kept_b = res["H4R_B"]["networks"]
        assert s3[arm]["pooled"]["networks"] == kept_b["v1"][arm] + kept_b["v3"][arm]
        for lab, d in s3[arm].items():
            if lab != "pooled":
                assert d["networks"] == kept_b[lab][arm]
            b = [res["statistics"]["B"][x] for x in d["networks"]]
            c = [completion[x] for x in d["networks"]]
            assert d["rho"] == pytest.approx(stats.spearmanr(b, c).statistic, nan_ok=True)
    assert s3[H.REAL]["pooled"]["rho"] > 0.5 and s3[H.BASE]["pooled"]["rho"] > 0.5
    # its bootstrap is stream 201's: REAL/pooled is the generator's first draw
    from apexzero.interp.constants import ROOT_SEED, STREAM_OFFSET

    assert H.design_block()["secondary"]["S3"]["stream"] == 201 == H.S3_STREAM
    nets = s3[H.REAL]["pooled"]["networks"]
    ref = H.spearman_ci([res["statistics"]["B"][x] for x in nets], [completion[x] for x in nets],
                        200, np.random.default_rng([ROOT_SEED, STREAM_OFFSET + 201]))
    assert (ref["lo"], ref["hi"]) == (s3[H.REAL]["pooled"]["lo"], s3[H.REAL]["pooled"]["hi"])
    reg = res["networks"][two_net]["regimes"]
    assert set(reg) == {"bulk", "tight"} and reg["bulk"]["n_keys"] == 12
    pooled = res["descriptive"]["pooled_with_step2"]
    assert len(pooled["networks"]) == 18 + 19
    assert pooled["groups"]["step2/v1/realw"]["G"] == pytest.approx(0.7)
    assert pooled["groups"]["h4r/v3/base"]["n"] == 5
    assert res["networks"][two_net]["T"]["label"] in ("width-blind", "mixed",
                                                      "corridor-anchored")


def test_analyse_h4r_statistics_are_the_planted_values():
    """Each network's G, B and T (= G + G_out) are its planted values, G_out differing per
    network so that a measure swap shows; T's labels have known answers (corridor-anchored
    for the real-width arm, width-blind for the baselines), and T's arm order is the reverse
    of G's, so G built from T would also flip H4R-G."""
    rows, want = {}, {}
    for i, name in enumerate(H.NETWORKS):
        v, arm, s = H.parse_name(name)
        G = (0.2 if arm == H.REAL else 0.35) + 0.01 * (s - 10)
        B = (0.6 if arm == H.REAL else 0.5) + 0.01 * (s - 10)
        go = (0.8 if arm == H.REAL else 0.0) + 0.002 * i
        rows[name] = _net_rows(G, B, seed=i, G_out=go)
        want[name] = {"G": G, "B": B, "T": G + go}
    opt = _opt_rows(rows[H.NETWORKS[0]])
    res = H.analyse_h4r(rows, opt, {}, dict.fromkeys(rows, 6), {}, None, None, n=100)
    for name, w in want.items():
        for m in ("G", "B", "T"):
            assert res["statistics"][m][name] == pytest.approx(w[m], abs=0.005), (name, m)
            assert res["networks"][name]["statistics"][m] == res["statistics"][m][name]
        label = res["networks"][name]["T"]["label"]
        assert label == ("corridor-anchored" if H.parse_name(name)[1] == H.REAL
                         else "width-blind"), (name, label)
    assert res["H4R_G"]["decision"] == "confirm" and res["H4R_B"]["decision"] == "confirm"


def test_analyse_h4r_refute_and_not_testable():
    rows = _all_rows()
    # G higher with real widths (step 2's original H4 direction): H4R-G is refuted
    for name in rows:
        if H.parse_name(name)[1] == H.REAL:
            rows[name] = _net_rows(0.5, 0.5, seed=1)
    opt = _opt_rows(rows[H.NETWORKS[0]])
    res = H.analyse_h4r(rows, opt, {}, dict.fromkeys(rows, 6), {}, None, None, n=100)
    assert res["H4R_G"]["decision"] == "refute"
    assert res["secondary"]["S1"]["decision"] == "inconclusive"
    # three real-width v1 networks gone: neither primary test is testable; Holm sees p = 1
    missing = {H.run_name("v1", H.REAL, s): "no run directory" for s in (10, 11, 12)}
    rows = {k: v for k, v in rows.items() if k not in missing}
    res = H.analyse_h4r(rows, opt, {}, dict.fromkeys(rows, 6), missing, None, None, n=100)
    assert res["H4R_G"]["decision"].startswith("not testable (fewer than 3 networks remain in "
                                               "v1/realw: 2")
    assert not res["holm"]["tests"]["H4R-B"]["testable"]
    assert res["holm"]["tests"]["H4R-B"]["p"] == 1.0


# ------------------------------------------------------------------------- the manifest


def _eval(k: int) -> dict:
    from apexzero.tracks.corners import R_DEV

    return {c: {"completion": 1.0 if i < k else (0.5 if i == k else 0.1), "median_lap_s": 80.0}
            for i, c in enumerate(R_DEV)}


@pytest.fixture(scope="module")
def ppo_env():
    return H.expected_ppo(), H.expected_env()


def _make_run(root: Path, name: str, ppo: dict, env: dict, final=None, eval_k=8, args=None):
    v, arm, s = H.parse_name(name)
    run = root / name
    (run / "ckpt").mkdir(parents=True)
    a = H.expected_args(v, arm, s) if args is None else args
    (run / "config.json").write_text(json.dumps({"ppo": ppo, "args": a, "env": env}))
    for ts in (10_000_000, 163_792_745, H.FINAL_TIMESTEPS if final is None else final):
        (run / "ckpt" / f"{ts:012d}.pkl").write_bytes(f"{name}{ts}".encode())
    if eval_k is not None:
        (run / "eval_rdev.json").write_text(json.dumps(_eval(eval_k)))
    return run


def _ts(path: Path, name: str) -> int:
    return int(path.stem)


def test_manifest_records_every_run_and_lists_the_missing(tmp_path, ppo_env):
    ppo, env = ppo_env
    for name in H.NETWORKS:
        _make_run(tmp_path, name, ppo, env)
    man = H.build_manifest(tmp_path, _ts, ppo, env)
    assert man["present"] == list(H.NETWORKS) and not man["missing"]
    assert not man["config_problems"] and not man["queue_log"]["present"]
    e = man["networks"]["runs/h4r-v3-realw-s12"]
    assert e["timesteps"] == H.FINAL_TIMESTEPS and e["completed"] == 8
    assert e["path"] == "runs/h4r-v3-realw-s12/ckpt/000199884800.pkl"
    assert e["sha256"] == prereg.sha256_bytes(b"runs/h4r-v3-realw-s12199884800")
    assert e["args"]["tilt_fraction"] == 0.5 and e["args"]["real_widths"] is True

    # the failures, each listed with its reason
    import shutil

    shutil.rmtree(tmp_path / "runs/h4r-v1-base-s10")
    (tmp_path / "runs/h4r-v1-base-s11/eval_rdev.json").unlink()
    (tmp_path / "runs/h4r-v1-realw-s10/ckpt/000199884800.pkl").unlink()
    ev = _eval(8)
    ev.pop("Sepang")
    (tmp_path / "runs/h4r-v3-base-s14/eval_rdev.json").write_text(json.dumps(ev))
    (tmp_path / "runs").joinpath("h4r.log").write_text(
        "[2026-10-22 12:00:00] start h4r-v3-base-s11\n"
        "[2026-10-22 12:20:00] retry h4r-v3-base-s11 (first attempt failed; see x)\n"
        "[2026-10-22 12:40:00] done  h4r-v3-base-s11\n"
        "[2026-10-22 12:00:00] start h4r-v3-realw-s13\n"
        "[2026-10-22 12:10:00] retry h4r-v3-realw-s13 (first attempt failed; see x)\n"
        "[2026-10-22 12:30:00] FAILED h4r-v3-realw-s13 (twice; see x)\n"
        "[2026-10-22 13:00:00] start h4r-v9-x-s1\n")
    (tmp_path / "runs/h4r-v3-base-s11.log.1").write_text("boom")
    man = H.build_manifest(tmp_path, _ts, ppo, env)
    m = man["missing"]
    assert m["runs/h4r-v1-base-s10"] == "no run directory"
    assert m["runs/h4r-v1-base-s11"].startswith("no eval_rdev.json")
    assert "163792745 timesteps, not 199884800" in m["runs/h4r-v1-realw-s10"]
    assert "lacks R-dev circuits ['Sepang']" in m["runs/h4r-v3-base-s14"]
    assert m["runs/h4r-v3-realw-s13"].startswith("failed twice in the queue")
    assert len(man["present"]) == 15 and set(man["present"]).isdisjoint(m)
    e = man["networks"]["runs/h4r-v3-base-s11"]
    assert e["retried"] and not e["failed_twice"] and e["first_attempt_log"]
    assert man["queue_log"]["unexpected_runs"] == ["h4r-v9-x-s1"]
    assert man["networks"]["runs/h4r-v1-base-s12"]["queue"] == []


def test_manifest_problems_gate_the_record(tmp_path, ppo_env):
    ppo, env = ppo_env
    for name in H.NETWORKS:
        _make_run(tmp_path, name, ppo, env)
    v1 = H.expected_args("v1", H.REAL, 11)
    _make_run(tmp_path / "x", "runs/h4r-v1-realw-s11", ppo, env, args={**v1, "seed": 0})
    import shutil

    shutil.rmtree(tmp_path / "runs/h4r-v1-realw-s11")
    shutil.copytree(tmp_path / "x/runs/h4r-v1-realw-s11", tmp_path / "runs/h4r-v1-realw-s11")
    man = H.build_manifest(tmp_path, _ts, ppo, env)
    assert man["unexplained_missing"] == [] and "runs/h4r-v1-realw-s11" in man["present"]
    assert H.manifest_problems(man) == ["runs/h4r-v1-realw-s11: args.seed = 0, want 11"]
    # a missing run: blocks unless the queue log shows it failing twice, or it is accepted
    shutil.rmtree(tmp_path / "runs/h4r-v1-realw-s11")
    shutil.rmtree(tmp_path / "runs/h4r-v3-base-s10/ckpt")
    man = H.build_manifest(tmp_path, _ts, ppo, env)
    un = ["runs/h4r-v1-realw-s11", "runs/h4r-v3-base-s10"]
    assert man["unexplained_missing"] == un
    # without the queue log, Section 2's only evidence, even a named run is refused
    probs = H.manifest_problems(man, dict.fromkeys(un, "the box died in round 3"))
    assert probs == [(f"runs are missing and the queue log {H.QUEUE_LOG} is not here: pull it "
                      "back and re-run the manifest stage (design Section 2's only evidence for "
                      "a missing run)")]
    (tmp_path / "runs/h4r.log").write_text("[t] start h4r-v1-realw-s11\n"
                                           "[t] start h4r-v3-base-s10\n")
    man = H.build_manifest(tmp_path, _ts, ppo, env)
    assert man["unexplained_missing"] == un and man["finished_missing"] == []
    assert any("does not show failing twice" in p for p in H.manifest_problems(man))
    assert H.manifest_problems(man, dict.fromkeys(un, "the box died in round 3")) == []
    one = H.manifest_problems(man, {un[0]: "the box died in round 3"})
    assert len(one) == 1 and un[1] in one[0] and un[0] not in one[0]
    (tmp_path / "runs/h4r.log").write_text(
        "[t] retry h4r-v1-realw-s11 (x)\n[t] FAILED h4r-v1-realw-s11 (twice)\n"
        "[t] retry h4r-v3-base-s10 (x)\n[t] FAILED h4r-v3-base-s10 (twice)\n")
    man = H.build_manifest(tmp_path, _ts, ppo, env)
    assert man["unexplained_missing"] == [] and H.manifest_problems(man) == []
    # a missing run's config problems are reported, never blocking
    (tmp_path / "runs/h4r.log").write_text("[t] FAILED h4r-v1-base-s10 (twice)\n")
    _make_run(tmp_path / "y", "runs/h4r-v1-base-s10", ppo, env, eval_k=None,
              args={**H.expected_args("v1", H.BASE, 10), "bank": 1})
    shutil.rmtree(tmp_path / "runs/h4r-v1-base-s10")
    shutil.copytree(tmp_path / "y/runs/h4r-v1-base-s10", tmp_path / "runs/h4r-v1-base-s10")
    man = H.build_manifest(tmp_path, _ts, ppo, env)
    assert "runs/h4r-v1-base-s10" in man["config_problems"]
    assert not any("h4r-v1-base-s10:" in p for p in H.manifest_problems(
        man, dict.fromkeys(man["unexplained_missing"], "x")))
    assert H.manifest_problems({"present": [], "config_problems": {}}) == ["no run is present"]


def test_accept_missing_names_each_run(tmp_path, ppo_env):
    """--accept-missing binds one named run at a time (RUN=REASON); a run the queue log shows
    done on the box is refused with its own message until named; a name that is not an
    unexplained missing run is refused; the record keeps each reason with the queue events."""
    import shutil

    ppo, env = ppo_env
    for name in H.NETWORKS:
        _make_run(tmp_path, name, ppo, env)
    done, failed, killed = "runs/h4r-v1-realw-s11", "runs/h4r-v3-base-s12", "runs/h4r-v1-base-s13"
    (tmp_path / "runs/h4r.log").write_text(
        "[2026-10-22 12:00:00] start h4r-v1-realw-s11\n"
        "[2026-10-22 12:26:00] done  h4r-v1-realw-s11\n"
        "[t] start h4r-v3-base-s12\n[t] retry h4r-v3-base-s12 (x)\n"
        "[t] FAILED h4r-v3-base-s12 (twice)\n[t] start h4r-v1-base-s13\n")
    for n in (done, failed, killed):
        shutil.rmtree(tmp_path / n)
    man = H.build_manifest(tmp_path, _ts, ppo, env)
    assert man["unexplained_missing"] == [killed, done] and man["finished_missing"] == [done]
    probs = H.manifest_problems(man)
    assert any("finished on the box" in p and done in p and killed not in p for p in probs)
    assert any("does not show failing twice" in p and killed in p and done not in p
               for p in probs)
    # each run is named: a blanket reason, an empty one, an unknown or repeated run refused
    for bad, msg in ((["the box died"], "RUN=REASON"), ([f"{done}= "], "RUN=REASON"),
                     (["runs/dr-realw-v1-s0=x"], "not an H4R run"),
                     ([f"{done}=a", "h4r-v1-realw-s11=b"], "twice")):
        with pytest.raises(SystemExit, match=msg):
            H.parse_accept(bad)
    assert H.build_parser().parse_args(["prereg", "--accept-missing", "a=b", "--accept-missing",
                                        "c=d"]).accept_missing == ["a=b", "c=d"]
    acc = H.parse_accept(["h4r-v1-base-s13=killed in round 4"])
    assert acc == {killed: "killed in round 4"}
    probs = H.manifest_problems(man, acc)
    assert len(probs) == 1 and "finished on the box" in probs[0] and done in probs[0]
    acc = H.parse_accept(["h4r-v1-base-s13=killed in round 4",
                          f"{done}=the box was deleted before the pull"])
    assert H.manifest_problems(man, acc) == []
    # a name that is not unexplained missing (present, or failed twice): refused
    for extra in ("runs/h4r-v1-base-s10", failed):
        assert any("not unexplained missing" in p and extra in p
                   for p in H.manifest_problems(man, {**acc, extra: "x"}))
    # the record keeps the reason, the queue events and whether it finished on the box
    blk = H.accepted_block(man, acc)
    assert blk[done] == {"reason": "the box was deleted before the pull",
                         "queue": [{"time": "2026-10-22 12:00:00", "event": "start"},
                                   {"time": "2026-10-22 12:26:00", "event": "done"}],
                         "finished_on_box": True}
    assert blk[killed]["finished_on_box"] is False and set(blk) == {done, killed}
    # no queue log: refused, every missing run named or not
    (tmp_path / "runs/h4r.log").unlink()
    man = H.build_manifest(tmp_path, _ts, ppo, env)
    assert man["finished_missing"] == []
    assert any("is not here" in p for p in H.manifest_problems(
        man, dict.fromkeys(man["unexplained_missing"], "x")))


@pytest.mark.parametrize("change,want", [
    ({"seed": 3}, "args.seed = 3, want 12"),
    ({"tilt_fraction": 0.0}, "args.tilt_fraction = 0.0, want 0.5"),
    ({"real_widths": False}, "args.real_widths = False, want True"),
    ({"real_widths": 1}, "args.real_widths = 1, want True"),
    ({"real_widths_from": "Sepang"}, "args.real_widths_from = 'Sepang', want ''"),
    ({"steps": 100e6}, "args.steps = 100000000.0, want 200000000.0"),
    ({"bank": 2048}, "args.bank = 2048, want 4096"),
    ({"extra": 1}, "args.extra = 1 is not in design Section 2"),
    ({"name": "h4r-v3-realw-s11"}, "args.name"),
])
def test_config_against_design_section_2(change, want, ppo_env):
    ppo, env = ppo_env
    args = {**H.expected_args("v3", H.REAL, 12), **change}
    probs = H.config_problems({"args": args, "ppo": ppo, "env": env}, "v3", H.REAL, 12, ppo, env)
    assert any(p.startswith(want) for p in probs), probs
    assert H.config_problems({"args": H.expected_args("v3", H.REAL, 12), "ppo": ppo, "env": env},
                             "v3", H.REAL, 12, ppo, env) == []
    a = H.expected_args("v1", H.BASE, 10)
    assert a["tilt_fraction"] == 0.0 and a["tilt_alpha"] == 1.0 and a["real_widths"] is False
    missing = dict(H.expected_args("v3", H.REAL, 12))
    missing.pop("refresh")
    assert "args.refresh missing (want 256)" in H.config_problems(
        {"args": missing, "ppo": ppo, "env": env}, "v3", H.REAL, 12, ppo, env)


def test_config_ppo_and_env_blocks(ppo_env):
    ppo, env = ppo_env
    assert ppo["total_timesteps"] == 200_000_000 and ppo["num_envs"] == 2048
    args = H.expected_args("v1", H.REAL, 11)
    bad = H.config_problems({"args": args, "ppo": {**ppo, "total_timesteps": 100_000_000},
                             "env": env}, "v1", H.REAL, 11, ppo, env)
    assert any("ppo.total_timesteps" in p for p in bad)
    assert any("ppo ends at 99876864 timesteps" in p for p in bad)
    bad = H.config_problems({"args": args, "ppo": ppo, "env": {**env, "dt": 0.1}}, "v1", H.REAL,
                            11, ppo, env)
    assert bad == ["env.dt = 0.1, want 0.05"]


def test_step2_runs_match_the_expected_blocks(ppo_env):
    """The expected PPO and env blocks are what train_dr.py wrote for step 2's real-width runs
    (their config.json only; skipped where the runs are not on this machine)."""
    ppo, env = ppo_env
    p = ROOT / "runs" / "dr-realw-v3-s2" / "config.json"
    if not p.exists():
        pytest.skip("step 2's runs are not on this machine")
    cfg = json.loads(p.read_text())
    args = {**cfg["args"], "real_widths_from": ""}     # added to train_dr.py after these runs
    assert H.config_problems({**cfg, "args": args}, "v3", H.REAL, 2, ppo, env) == \
        ["args.name = 'dr-realw-v3-s2', want 'h4r-v3-realw-s2'"]


def test_queue_events():
    q = H.queue_events("[2026-10-22 12:00:00] start h4r-v1-base-s10\nnoise\n"
                       "[2026-10-22 12:26:00] done  h4r-v1-base-s10\n"
                       "[2026-10-22 14:00:00] H4R runs finished\n")
    assert q == {"h4r-v1-base-s10": [{"time": "2026-10-22 12:00:00", "event": "start"},
                                     {"time": "2026-10-22 12:26:00", "event": "done"}]}


# ------------------------------------------------------------------------- git: prereg, freeze


def _g(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                        "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _put(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _manifest(present=(H.NETWORKS[0],), missing=None, unexplained=(), finished=(),
              queue=True) -> dict:
    """A manifest in build_manifest's shape: every run's entry, the NET_FIELDS of each present
    one, the missing and unexplained runs ('finished' ones with a 'done' queue event)."""
    nets = {}
    for n in H.NETWORKS:
        v, arm, s = H.parse_name(n)
        nets[n] = {"variant": v, "arm": arm, "seed": s,
                   "queue": [{"time": "t", "event": "done"}] if n in finished else []}
        if n in present:
            nets[n].update(path=f"{n}/ckpt/{H.FINAL_TIMESTEPS:012d}.pkl", sha256=f"s-{n}",
                           timesteps=H.FINAL_TIMESTEPS, eval_path=f"{n}/eval_rdev.json",
                           eval_sha256=f"e-{n}", completed=8, config_sha256=f"c-{n}")
    return {"design_doc": H.DESIGN_DOC, "final_timesteps": H.FINAL_TIMESTEPS, "networks": nets,
            "present": list(present), "missing": dict(missing or {}),
            "unexplained_missing": list(unexplained), "finished_missing": list(finished),
            "config_problems": {}, "queue_log": {"path": H.QUEUE_LOG, "present": queue}}


def _repo(tmp_path: Path, man: dict | None = None) -> Path:
    """A repository with H4R's bound files committed (the manifest ignored, added with -f)."""
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    _g(root, "init", "-q", "-b", "main")
    _put(root, ".gitignore", "outputs/\nruns/\n")
    _put(root, H.SCRIPT_REL, "# h4r\n")
    _put(root, H.DESIGN_DOC, "# H4R design\n")
    _put(root, H.MANIFEST_REL, json.dumps(_manifest() if man is None else man))
    _g(root, "add", ".gitignore", H.SCRIPT_REL, H.DESIGN_DOC)
    _g(root, "add", "-f", H.MANIFEST_REL)
    _g(root, "commit", "-q", "-m", "base")
    return root


def _record(root: Path, **over) -> dict:
    """A record of the committed bound files whose networks, missing runs, acceptances and
    design block are the manifest's and this script's (``over`` replaces items)."""
    sha = prereg.sha256_file
    man = json.loads((root / H.MANIFEST_REL).read_text())
    rec = {"version": H.RECORD_VERSION, "tag": H.TAG, "prereg_base_commit": _g(root, "rev-parse",
                                                                                "HEAD"),
           "script_sha256": sha(root / H.SCRIPT_REL),
           "design_doc_sha256": sha(root / H.DESIGN_DOC),
           "manifest": {"path": H.MANIFEST_REL, "sha256": sha(root / H.MANIFEST_REL)},
           "networks": {n: {k: man["networks"][n][k] for k in H.NET_FIELDS}
                        for n in man["present"]},
           "missing": man["missing"], "design": json.loads(json.dumps(H.design_block())),
           "environment": F.environment(root)}
    if man["unexplained_missing"]:
        rec["missing_accepted"] = H.accepted_block(man, dict.fromkeys(man["unexplained_missing"],
                                                                      "lost"))
    rec.update(over)
    _put(root, H.PREREG_REL, json.dumps(rec))
    _g(root, "add", H.PREREG_REL)
    _g(root, "commit", "-q", "-m", "record")
    return rec


def test_tree_problems_need_the_bound_files_committed(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _g(root, "init", "-q", "-b", "main")
    _put(root, ".gitignore", "outputs/\n")
    _put(root, H.SCRIPT_REL, "# h4r\n")
    _put(root, H.DESIGN_DOC, "# design\n")
    _put(root, H.MANIFEST_REL, "{}")
    _g(root, "add", ".gitignore", H.SCRIPT_REL)
    _g(root, "commit", "-q", "-m", "x")
    probs = H.tree_problems(root)
    assert any(f"{H.DESIGN_DOC} differs from HEAD" in p for p in probs)
    assert f"{H.MANIFEST_REL} is not committed at HEAD (git add -f {H.MANIFEST_REL})" in probs
    assert f"{H.DESIGN_DOC} is not committed at HEAD" in probs          # not ignored: no -f
    _g(root, "add", H.DESIGN_DOC)
    _g(root, "add", "-f", H.MANIFEST_REL)
    _g(root, "commit", "-q", "-m", "y")
    assert H.tree_problems(root) == []
    _put(root, H.SCRIPT_REL, "# h4r, edited\n")
    assert any("scripts/h4r.py on disk is not HEAD's" in p for p in H.tree_problems(root))
    _g(root, "update-index", "--assume-unchanged", H.SCRIPT_REL)    # hidden from git status
    assert H.tree_problems(root) == [f"{H.SCRIPT_REL} on disk is not HEAD's"]


def test_freeze_problems_in_a_repository(tmp_path):
    root = _repo(tmp_path)
    assert H.freeze_problems(root=root, remote=None)[0].startswith(
        f"the git tag {H.TAG} does not exist yet")
    _record(root)
    _g(root, "tag", H.TAG)                                         # lightweight
    assert any("lightweight" in p for p in H.freeze_problems(root=root, remote=None))
    _g(root, "tag", "-d", H.TAG)
    _g(root, "tag", "-a", H.TAG, "-m", "H4R")
    assert H.freeze_problems(root=root, remote=None) == []
    # on origin as the same object
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    _g(root, "remote", "add", "origin", str(bare))
    assert any("is not on origin" in p for p in H.freeze_problems(root=root, remote="origin"))
    _g(root, "push", "-q", "origin", "main", H.TAG)
    assert H.freeze_problems(root=root, remote="origin") == []
    # the write-once pin, named by the tag: committed before the next stage
    pin = H.pin_tag(root=root)
    assert pin == root / "outputs/h4r/tag-h4r-prereg-v1.json" == root / H.pin_rel()
    assert json.loads(pin.read_text())["type"] == "tag"
    assert any(f"{H.pin_rel()} is not committed" in p
               for p in H.freeze_problems(root=root, remote=None))
    _g(root, "add", "-f", str(pin.relative_to(root)))
    _g(root, "commit", "-q", "-m", "pin")
    assert H.freeze_problems(root=root, remote=None) == []
    before = pin.read_text()
    H.pin_tag(root=root)
    assert pin.read_text() == before                              # never rewritten
    # a re-registration under a new tag never trips over this tag's committed pin
    v2 = "h4r-prereg-v2"
    _g(root, "tag", "-a", v2, "-m", "v2")
    assert H.pin_rel(v2) != H.pin_rel()
    assert F._pin_committed_problems(root / H.pin_rel(v2), root, _g(root, "rev-parse", v2),
                                     _g(root, "rev-parse", f"{v2}^{{commit}}")) == []
    _g(root, "tag", "-d", v2)
    # a docs-only commit after the tag changes nothing
    _put(root, "docs/findings/h4r.md", "x")
    _g(root, "add", "docs/findings/h4r.md")
    _g(root, "commit", "-q", "-m", "docs")
    assert H.freeze_problems(root=root, remote=None) == []
    # the script edited on disk, then committed
    _put(root, H.SCRIPT_REL, "# h4r, fixed after the tag\n")
    assert f"{H.SCRIPT_REL} on disk is not the tagged one" in H.freeze_problems(root=root,
                                                                               remote=None)
    _g(root, "commit", "-q", "-am", "edit")
    probs = H.freeze_problems(root=root, remote=None)
    assert f"HEAD's {H.SCRIPT_REL} is not the tagged one" in probs
    _g(root, "reset", "-q", "--hard", "HEAD~1")
    # the record or the manifest on disk
    _put(root, H.MANIFEST_REL, json.dumps({"present": ["x"]}))
    assert f"{H.MANIFEST_REL} on disk is not the tagged one" in H.freeze_problems(root=root,
                                                                                 remote=None)
    _g(root, "checkout", "-q", "--", H.MANIFEST_REL)
    assert H.freeze_problems(root=root, remote=None) == []
    # verify's problems are reported
    assert H.freeze_problems(root=root, remote=None, verify=lambda r, rt: ["ckpt changed"]) == \
        ["ckpt changed"]
    # HEAD not after the tag
    _g(root, "checkout", "-q", "-b", "old", "HEAD~3")
    assert any("is not an ancestor of HEAD" in p for p in H.freeze_problems(root=root,
                                                                            remote=None))
    _g(root, "checkout", "-q", "main")
    # the tag moved after its pin was committed
    _g(root, "tag", "-f", "-a", H.TAG, "-m", "moved", "HEAD")
    probs = H.freeze_problems(root=root, remote=None)
    assert any("the tag moved" in p for p in probs)


def test_freeze_problems_bind_the_record_to_the_tag(tmp_path):
    root = _repo(tmp_path)
    _record(root, script_sha256="0" * 64)
    _g(root, "tag", "-a", H.TAG, "-m", "H4R")
    assert H.freeze_problems(root=root, remote=None) == \
        [f"the record does not register the tagged {H.SCRIPT_REL}"]
    # the tag differs from the record's base commit by more than the record
    root = _repo(tmp_path / "b")
    base = _g(root, "rev-parse", "HEAD")
    _put(root, H.DESIGN_DOC, "# H4R design, changed after the base\n")
    _g(root, "commit", "-q", "-am", "doc")
    _record(root, prereg_base_commit=base)
    _g(root, "tag", "-a", H.TAG, "-m", "H4R")
    probs = H.freeze_problems(root=root, remote=None)
    assert any("by more than the record" in p and H.DESIGN_DOC in p for p in probs)
    # another environment's package versions
    root = _repo(tmp_path / "c")
    env = F.environment(root)
    _record(root, environment={**env, "packages": {**env["packages"], "jax": "0.0.1"}})
    _g(root, "tag", "-a", H.TAG, "-m", "H4R")
    assert any("package versions differ" in p for p in H.freeze_problems(root=root, remote=None))
    # another record version or tag name
    root = _repo(tmp_path / "d")
    _record(root, version=2)
    _g(root, "tag", "-a", H.TAG, "-m", "H4R")
    assert H.freeze_problems(root=root, remote=None) == \
        [f"the record names tag {H.TAG!r}, version 2"]
    # a base commit on a side branch (touching no bound file): not an ancestor of the tag
    root = _repo(tmp_path / "e")
    _g(root, "checkout", "-q", "-b", "side")
    _put(root, "notes.txt", "x")
    _g(root, "add", "notes.txt")
    _g(root, "commit", "-q", "-m", "side")
    side = _g(root, "rev-parse", "HEAD")
    _g(root, "checkout", "-q", "main")
    _record(root, prereg_base_commit=side)
    _g(root, "tag", "-a", H.TAG, "-m", "H4R")
    probs = H.freeze_problems(root=root, remote=None)
    assert probs == [f"the record's base commit {side[:12]} is not an ancestor of the tag {H.TAG}"]


def test_freeze_compares_the_record_with_the_tagged_manifest(tmp_path):
    """Which networks enter the tests, and why one is dropped, comes from the tagged manifest:
    a record edited by hand (a present run moved to 'missing', a field changed, an acceptance
    added, dropped or stripped, the design block changed) is refused with git content alone
    (no verify)."""
    two, lost = (H.NETWORKS[0], H.NETWORKS[5]), H.NETWORKS[7]
    man = _manifest(present=two, missing={lost: "no run directory"}, unexplained=(lost,),
                    finished=(lost,))

    def tagged(sub, **over):
        root = _repo(tmp_path / sub, man)
        _record(root, **over)
        _g(root, "tag", "-a", H.TAG, "-m", "H4R")
        return H.freeze_problems(root=root, remote=None)

    assert tagged("honest") == []
    nets = {n: {k: man["networks"][n][k] for k in H.NET_FIELDS} for n in two}
    probs = tagged("moved", networks={two[0]: nets[two[0]]},
                   missing={**man["missing"], two[1]: "hand-edited: dropped after seeing eval"})
    assert any("networks are not the tagged manifest's present runs" in p and two[1] in p
               for p in probs)
    assert "the record's missing runs are not the tagged manifest's" in probs
    probs = tagged("field", networks={**nets, two[0]: {**nets[two[0]], "completed": 12}})
    assert any("networks are not the tagged manifest's present runs" in p and two[0] in p
               for p in probs)
    acc = H.accepted_block(man, {lost: "the box was deleted before the pull"})
    assert any(f"accepted missing runs [] are not the tagged manifest's unexplained missing "
               f"runs [{lost!r}]" in p for p in tagged("none", missing_accepted={}))
    assert any("are not the tagged manifest's unexplained" in p
               for p in tagged("extra", missing_accepted={**acc, two[0]: acc[lost]}))
    probs = tagged("noreason", missing_accepted={lost: {**acc[lost], "reason": " "}})
    assert any("do not carry a reason" in p for p in probs)
    assert any(p.startswith("the tagged manifest: runs the queue log shows finished") for p in probs)
    assert any("do not carry a reason" in p
               for p in tagged("queue", missing_accepted={lost: {**acc[lost], "queue": []}}))
    assert any("do not carry a reason" in p for p in tagged(
        "fin", missing_accepted={lost: {**acc[lost], "finished_on_box": False}}))
    d = json.loads(json.dumps(H.design_block()))
    d["holm"]["m"] = 1
    assert tagged("design", design=d) == ["the record's design block is not this script's"]
    # the manifest's own rule still holds at the tag (a present run's config problem)
    bad = {**man, "config_problems": {two[0]: ["args.seed = 0, want 10"]}}
    root = _repo(tmp_path / "cfg", bad)
    _record(root)
    _g(root, "tag", "-a", H.TAG, "-m", "H4R")
    assert H.freeze_problems(root=root, remote=None) == \
        [f"the tagged manifest: {two[0]}: args.seed = 0, want 10"]


def test_pre_tag_stages_refuse_after_the_tag(tmp_path):
    root = _repo(tmp_path)
    H.refuse_after_tag("manifest", root)
    _g(root, "tag", "-a", H.TAG, "-m", "H4R")
    with pytest.raises(SystemExit, match="after the tag h4r-prereg-v1"):
        H.refuse_after_tag("prereg", root)


def _step2_record():
    a2 = [["nominal", f"k{i}"] for i in range(6)]
    return {"key_sets": {"a2": a2, "a2_twins": [["phase", "t1"]]},
            "content": {"a2": "c-a2", "a2_twins": "c-tw"},
            "gate2_trained": a2[:3], "code_hash": "step2-code",
            "calibration": {"tolerance": {"c_min": 0.003, "s_b": 0.07, "v_apex": 0.001},
                            "path": "calibration/branch_tol.json", "sha256": "s", "factor": 5.0,
                            "trained_mean_err": 5e-5},
            "outputs": {"opt_fields": {"rows_sha256": "r", "arrays_hash": "h"}},
            "regimes": {"variant_regime_cells": {"v1": {"bulk": [1]}, "v3": {"bulk": [2]},
                                                 "transfer": {"bulk": [3]}},
                        "h4_typing": {"v1": "G1", "v3": "G3", "transfer": "G3"}}}


def test_record_copies_step2_and_the_manifest(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    rec2 = _step2_record()
    _put(root, H.STEP2_RECORD_REL, json.dumps(rec2))
    _g(root, "add", H.STEP2_RECORD_REL)
    _g(root, "commit", "-q", "-m", "step 2 record")
    _g(root, "tag", "-a", H.STEP2_TAG, "-m", "step 2")
    s2, probs = H.step2_identity(root)
    assert probs == [] and s2["tag_object"] == _g(root, "rev-parse", H.STEP2_TAG)
    assert s2["decisions"]["sha256"] == H.STEP2_DECISIONS_SHA256
    man = {"present": ["runs/h4r-v1-base-s10"],
           "missing": {"runs/h4r-v1-base-s11": "no run directory"},
           "networks": {"runs/h4r-v1-base-s10": {
               "variant": "v1", "arm": "base", "seed": 10, "path": "p", "sha256": "s",
               "timesteps": H.FINAL_TIMESTEPS, "eval_path": "e", "eval_sha256": "es",
               "completed": 8, "config_sha256": "cs", "queue": []}}}
    rec = H.build_record(root, man, "man-sha", s2, rec2, "base-commit",
                         keys_root=tmp_path / "keys")
    assert rec["networks"] == {"runs/h4r-v1-base-s10": {
        "variant": "v1", "arm": "base", "seed": 10, "path": "p", "sha256": "s",
        "timesteps": H.FINAL_TIMESTEPS, "eval_path": "e", "eval_sha256": "es", "completed": 8,
        "config_sha256": "cs"}}
    assert rec["missing"] == man["missing"] and rec["manifest"]["sha256"] == "man-sha"
    assert rec["keys"]["a2"]["ids"] == rec2["key_sets"]["a2"]
    assert rec["keys"]["a2"]["content"] == "c-a2" and rec["keys"]["gate2_trained"]["n"] == 3
    assert rec["branch_tolerance"]["tolerance"] == rec2["calibration"]["tolerance"]
    assert rec["regime_cells"] == {"v1": {"bulk": [1]}, "v3": {"bulk": [2]}}
    assert rec["step2"]["record_sha256"] == prereg.sha256_file(root / H.STEP2_RECORD_REL)
    assert rec["design"]["gate2_trained"]["descriptive_if_failing_more_than"] == 2
    assert H.step2_record_problems(rec, root) == []
    assert H.step2_record_problems({**rec, "keys": {**rec["keys"], "gate2_trained": {
        "ids": rec2["key_sets"]["a2"][:2]}}}, root) == \
        ["the record's gate2-trained keys are not step 2's"]
    # each other copied item altered gives exactly its own message
    tol = rec["branch_tolerance"]
    for alt, msg in (
            ({"keys": {**rec["keys"], "a2": {**rec["keys"]["a2"],
                                             "ids": rec["keys"]["a2"]["ids"][1:]}}},
             "the record's A2 keys are not step 2's"),
            ({"branch_tolerance": {**tol, "tolerance": {**tol["tolerance"], "c_min": 1.0}}},
             "the record's branch tolerance is not step 2's"),
            ({"step2": {**rec["step2"], "record_sha256": "x" * 64}},
             "step 2's record_sha256 is"),
            ({"step2": {**rec["step2"], "tag_object": "y" * 40}}, "step 2's tag_object is"),
            ({"step2": {**rec["step2"], "decisions": {"path": H.STEP2_DECISIONS_REL,
                                                      "sha256": "z" * 64}}},
             (f"step 2's decisions are pinned at {H.STEP2_DECISIONS_SHA256[:12]}, the "
              "record's zzzzzzzzzzzz"))):
        probs = H.step2_record_problems({**rec, **alt}, root)
        assert len(probs) == 1 and probs[0].startswith(msg), probs
    assert "missing_accepted" not in rec
    # an accepted missing run: its reason, queue events and whether it finished on the box
    lost = "runs/h4r-v1-base-s11"
    man2 = {**man, "unexplained_missing": [lost], "finished_missing": [],
            "networks": {**man["networks"], lost: {"queue": [{"time": "t", "event": "start"}]}}}
    rec_a = H.build_record(root, man2, "man-sha", s2, rec2, "base-commit",
                           keys_root=tmp_path / "keys", accept={lost: "killed in round 4"})
    assert rec_a["missing_accepted"] == {lost: {"reason": "killed in round 4", "queue": [
        {"time": "t", "event": "start"}], "finished_on_box": False}}
    # step 2's tag must be its pinned object in this repository
    monkeypatch.setattr(H, "ROOT", root)
    assert any("not its pinned object" in p for p in H.step2_identity(root)[1])
    monkeypatch.setattr(H, "STEP2_TAG_OBJECT", _g(root, "rev-parse", H.STEP2_TAG))
    assert H.step2_identity(root)[1] == []
    _put(root, H.STEP2_RECORD_REL, json.dumps({**rec2, "code_hash": "edited"}))
    assert any("on disk is not the one at" in p for p in H.step2_record_problems(rec, root))


def test_network_problems_rehash_the_recorded_files(tmp_path):
    name = H.NETWORKS[0]
    ck = _put(tmp_path, f"{name}/ckpt/0000.pkl", "ckpt")
    ev = _put(tmp_path, f"{name}/eval_rdev.json", "{}")
    rec = {"networks": {name: {"path": f"{name}/ckpt/0000.pkl", "sha256": prereg.sha256_file(ck),
                               "eval_path": f"{name}/eval_rdev.json",
                               "eval_sha256": prereg.sha256_file(ev)}}}
    assert H.network_problems(rec, tmp_path) == []
    ck.write_text("ckpt, changed")
    assert H.network_problems(rec, tmp_path) == \
        [f"{name}: the checkpoint {name}/ckpt/0000.pkl is not the recorded one"]
    ck.write_text("ckpt")
    later = _put(tmp_path, f"{name}/ckpt/0001.pkl", "newer")      # sorts after the pinned one
    probs = H.network_problems(rec, tmp_path)
    assert len(probs) == 1 and probs[0].startswith(f"{name}: its last checkpoint is now")
    assert "0001.pkl" in probs[0]
    later.unlink()
    ev.write_text('{"Sepang": {}}')
    assert H.network_problems(rec, tmp_path) == \
        [f"{name}: {name}/eval_rdev.json is not the recorded evaluation"]
    ev.write_text("{}")
    ck.unlink()
    assert H.network_problems(rec, tmp_path) == \
        [f"{name}: the checkpoint {name}/ckpt/0000.pkl is not the recorded one"]


def test_run_once_and_imports(tmp_path):
    root = _repo(tmp_path)
    path = root / H.DECISIONS_REL
    assert F.run_once_problems(path, root) == []
    _put(root, H.DECISIONS_REL, "{}")
    assert F.run_once_problems(path, root) == [f"{path} exists"]
    _g(root, "add", "-f", H.DECISIONS_REL)
    _g(root, "commit", "-q", "-m", "decisions")
    path.unlink()
    assert F.run_once_problems(path, root) == [f"{H.DECISIONS_REL} was committed before"]
    # this script and step 2's frozen code may be imported; nothing else from the repository
    import apexzero

    mods = {"h4r_script": H, "fields": F, "apexzero": apexzero}
    assert H.import_problems(ROOT, mods) == []
    probs = H.import_problems(ROOT, {**mods, "test_h4r": sys.modules[__name__]})
    assert probs and "outside the frozen code" in probs[0]


def test_partial_and_exploratory_options(tmp_path):
    p = H.build_parser()
    a = p.parse_args(["fields"])
    assert H._partial(a) == [] and a.out == str(H.OUT)
    assert H._partial(p.parse_args(["fields", "--out", str(tmp_path), "--limit", "3"])) == \
        ["--out", "--limit"]
    rec = {"networks": {n: {} for n in H.NETWORKS[:3]}}
    assert H._networks(a, rec) == list(H.NETWORKS[:3])
    with pytest.raises(SystemExit, match="exploratory run"):
        H._networks(p.parse_args(["fields", "--network", H.NETWORKS[0]]), rec)
    with pytest.raises(SystemExit, match="not recorded"):
        H._networks(p.parse_args(["fields", "--out", str(tmp_path), "--network",
                                  H.NETWORKS[5]]), rec)


def test_post_tag_refuses_before_unlocking(tmp_path, monkeypatch):
    """The only code gate between 'before the tag' and trained networks' fields: a partial
    run into outputs/h4r, a module from outside the frozen code or a freeze problem stops a
    post-tag stage before the pin, the record, x64 and ``unlock_trained``."""
    from apexzero.fields import network as N

    calls, fz = [], []
    for name in ("pin_tag", "tagged_record"):
        monkeypatch.setattr(H, name, lambda *a, _n=name, **k: calls.append(_n))
    monkeypatch.setattr(H.F, "_x64", lambda: calls.append("x64"))
    monkeypatch.setattr(H.F, "tag_identity", lambda *a, **k: None)
    monkeypatch.setattr(N, "unlock_trained", lambda: calls.append("unlock"))
    p = H.build_parser()
    H._PROV[:] = [H.Provenance(p.parse_args(["fields"]), "fields")]
    monkeypatch.setattr(H, "freeze_problems", lambda **k: fz.append(k) or ["tag missing"])
    monkeypatch.setattr(H, "import_problems", lambda *a, **k: ["mod outside"])
    with pytest.raises(SystemExit, match="--limit writes a partial run"):
        H._post_tag(p.parse_args(["fields", "--limit", "3"]), "w")
    with pytest.raises(SystemExit, match="refusing to run w: mod outside"):
        H._post_tag(p.parse_args(["fields"]), "w")
    assert fz == [] and calls == []
    monkeypatch.setattr(H, "import_problems", lambda *a, **k: [])
    with pytest.raises(SystemExit, match="refusing to run w: tag missing"):
        H._post_tag(p.parse_args(["fields"]), "w")
    assert fz == [{"verify": H._verify_on_disk}] and calls == []
    monkeypatch.setattr(H, "freeze_problems", lambda **k: [])
    H._post_tag(p.parse_args(["fields", "--out", str(tmp_path), "--limit", "3"]), "w")
    assert calls == ["pin_tag", "tagged_record", "x64", "unlock"]
    calls.clear()
    H._post_tag(p.parse_args(["harness-f32"]), "w", x64=False)
    assert calls == ["pin_tag", "tagged_record"]


def test_manifest_loader_resolution_and_final_name(tmp_path, ppo_env):
    ppo, env = ppo_env
    for name in H.NETWORKS:
        _make_run(tmp_path, name, ppo, env)
    man = H.build_manifest(tmp_path, _ts, ppo, env,
                           resolve=lambda n: tmp_path / n / "ckpt" / "000010000000.pkl")
    assert all(any("step 2's loader resolves" in p for p in man["config_problems"][n])
               for n in H.NETWORKS)
    final = f"{H.FINAL_TIMESTEPS:012d}.pkl"
    man = H.build_manifest(tmp_path, _ts, ppo, env, resolve=lambda n: tmp_path / n / "ckpt" / final)
    assert not man["config_problems"]
    # the last checkpoint holds the final timesteps but is not named by them: not finished
    ck = tmp_path / H.NETWORKS[0] / "ckpt"
    (ck / final).rename(ck / "zzz.pkl")
    man = H.build_manifest(tmp_path, lambda p, n: H.FINAL_TIMESTEPS, ppo, env)
    assert "zzz.pkl holds 199884800 timesteps" in man["missing"][H.NETWORKS[0]]


def test_stamp_refuses_a_code_change_and_names_it(tmp_path):
    root = _repo(tmp_path)
    p = H.Provenance(None, "x", root=root)
    p.stamp()
    _put(root, H.SCRIPT_REL, "# h4r, edited mid-run\n")
    with pytest.raises(SystemExit, match=f"code changed while the stage ran .*{H.SCRIPT_REL}"):
        p.stamp()
    _g(root, "checkout", "-q", "--", H.SCRIPT_REL)
    p.stamp()
    _put(root, "src/apexzero/scratch.py", "x = 1\n")              # a new file under src/
    with pytest.raises(SystemExit, match=r"src/apexzero/scratch.py \(untracked\)"):
        p.stamp()


def test_fields_refuses_another_checkpoints_harness(tmp_path, monkeypatch):
    import types

    from apexzero.fields import network as N

    name = H.NETWORKS[0]
    rec = {"networks": {name: {"path": f"{name}/ckpt/x.pkl", "sha256": "mine", "timesteps": 7}},
           "branch_tolerance": {"tolerance": {"c_min": 1.0}}}
    monkeypatch.setattr(H, "_post_tag", lambda args, what, x64=True: rec)
    monkeypatch.setattr(H, "frozen_refs", lambda r, w: [types.SimpleNamespace(key_id="k0")])

    def boom(*a, **k):
        raise AssertionError("loaded")

    monkeypatch.setattr(S, "load_network", boom)
    args = H.build_parser().parse_args(["fields", "--out", str(tmp_path / "o"), "--log-dir",
                                        str(tmp_path)])
    H._PROV[:] = [H.Provenance(args, "fields")]
    with pytest.raises(SystemExit, match="no float32 harness runs"):
        H.stage_fields(args)
    N.save_det(F.harness_path(tmp_path / "o", name), {"x": np.zeros(2)}, {"ckpt_sha256": "other"})
    with pytest.raises(SystemExit, match="not this checkpoint's harness reference"):
        H.stage_fields(args)


def test_rows_done_needs_complete_rows_of_this_script(tmp_path):
    name = H.NETWORKS[0]
    pin = {"path": f"{name}/ckpt/x.pkl", "sha256": "sha0", "timesteps": 7}
    rows = _net_rows(0.3, 0.5)
    want = {r["key_id"] for r in rows}
    H._PROV[:] = [H.Provenance(H.build_parser().parse_args(["fields"]), "fields")]
    H._prov().tag = TAG_ID
    net = S.expected_net(name, pin)
    arr = tmp_path / "arrays"
    for r in rows:
        f = S.save_key_arrays(S.arrays_dir(arr, net), r["key_id"],
                              {"impulse": np.zeros((4, 3, 2))})
        r.update(net=net, ckpt_sha256=pin["sha256"], arrays_sha256=prereg.sha256_file(f))
    path = tmp_path / "rows.jsonl"
    assert not H._rows_done(path, want, name, pin, arr)               # nothing yet
    H._write_rows(path, rows)
    assert H._rows_done(path, want, name, pin, arr)
    H._write_rows(path, rows[1:])                                    # incomplete
    assert not H._rows_done(path, want, name, pin, arr)
    H._write_rows(path, rows)
    sc = prereg.stamp_path(path)
    st = json.loads(sc.read_text())
    for bad in ({"partial": ["--limit"]}, {"h4r_sha256": "0" * 64}, {"code_hash": "other"},
                {"tag": {**TAG_ID, "object": "c" * 40}}):
        sc.write_text(json.dumps({**st, **bad}))
        assert not H._rows_done(path, want, name, pin, arr), bad
    sc.write_text(json.dumps(st))
    assert H._rows_done(path, want, name, pin, arr)
    assert not H._rows_done(path, want, name, {**pin, "sha256": "other"}, arr)


def _prereg_repo(tmp_path: Path, monkeypatch, tol=None, lost=True):
    """A repository ready for the prereg stage: H4R's bound files and a manifest (one present
    run and, with ``lost``, one unexplained missing run) committed, step 2's record committed
    and tagged as its pinned object, the A2 key files on disk; step 2's full freeze check and
    the fresh manifest stubbed."""
    missing = {H.NETWORKS[1]: "no run directory"} if lost else {}
    man = _manifest(present=(H.NETWORKS[0],), missing=missing, unexplained=tuple(missing))
    ent = man["networks"][H.NETWORKS[0]]                # the present run's files, untracked
    root = tmp_path / "repo"
    for k, rel in (("sha256", ent["path"]), ("eval_sha256", ent["eval_path"])):
        ent[k] = prereg.sha256_file(_put(root, rel, f"{k} of {H.NETWORKS[0]}"))
    root = _repo(tmp_path, man)
    keys = tmp_path / "keys"
    rec2 = _step2_record()
    for kind, kid in rec2["key_sets"]["a2"]:
        _put(keys, f"{kind}/{kid}.json", json.dumps({"key_id": kid}))
        (keys / kind / f"{kid}.npz").write_bytes(kid.encode())
    rec2["content"]["a2"] = prereg.content_hash([tuple(t) for t in rec2["key_sets"]["a2"]], keys)
    if tol is not None:
        rec2["calibration"]["tolerance"] = tol
    _put(root, H.STEP2_RECORD_REL, json.dumps(rec2))
    _g(root, "add", H.STEP2_RECORD_REL)
    _g(root, "commit", "-q", "-m", "step 2 record")
    _g(root, "tag", "-a", H.STEP2_TAG, "-m", "step 2")
    monkeypatch.setattr(H, "ROOT", root)
    monkeypatch.setattr(H, "KEYS_DIR", keys)
    monkeypatch.setattr(H, "STEP2_TAG_OBJECT", _g(root, "rev-parse", H.STEP2_TAG))
    monkeypatch.setattr(H, "step2_freeze_problems", list)
    monkeypatch.setattr(H, "build_manifest", lambda *a, **k: json.loads(json.dumps(man)))
    return root, keys, man, rec2


def test_prereg_stage_refuses_then_writes_the_record(tmp_path, monkeypatch):
    """The stage that writes the frozen record, in a temporary repository: each refusal
    writes nothing; the record keeps the named missing run's reason; once committed and
    tagged, the freeze accepts it and rebuilds it item by item."""
    root, keys, man, rec2 = _prereg_repo(tmp_path, monkeypatch)
    lost = man["unexplained_missing"][0]
    p = H.build_parser()

    def refused(root, *extra, match):
        with pytest.raises(SystemExit, match=match):
            H.stage_prereg(p.parse_args(["prereg", *extra]))
        assert not (root / H.PREREG_REL).exists()

    ok = ["--accept-missing", f"{lost}=killed in round 4"]
    refused(root, match="does not show failing twice")
    refused(root, *ok, "--accept-missing", f"{H.NETWORKS[0]}=x", match="not unexplained missing")
    _put(root, H.SCRIPT_REL, "# h4r, edited\n")
    refused(root, *ok, match="dirty tree")
    _g(root, "checkout", "-q", "--", H.SCRIPT_REL)
    monkeypatch.setattr(H, "build_manifest", lambda *a, **k: {**man, "present": []})
    refused(root, *ok, match="no longer matches")
    monkeypatch.setattr(H, "build_manifest", lambda *a, **k: json.loads(json.dumps(man)))
    kf = keys / "nominal" / "k0.npz"
    kb = kf.read_bytes()
    kf.write_bytes(b"changed")
    refused(root, *ok, match="A2 key files differ")
    kf.write_bytes(kb)
    monkeypatch.setattr(H, "step2_freeze_problems", lambda: ["step 2: x"])
    refused(root, *ok, match="step 2's freeze does not hold: step 2: x")
    monkeypatch.setattr(H, "step2_freeze_problems", list)
    tag2 = _g(root, "rev-parse", H.STEP2_TAG)
    monkeypatch.setattr(H, "STEP2_TAG_OBJECT", "0" * 40)
    refused(root, *ok, match="not its pinned object")
    monkeypatch.setattr(H, "STEP2_TAG_OBJECT", tag2)
    H.stage_prereg(p.parse_args(["prereg", *ok]))
    rec = json.loads((root / H.PREREG_REL).read_text())
    assert rec["missing_accepted"] == {lost: {"reason": "killed in round 4", "queue": [],
                                              "finished_on_box": False}}
    assert rec["manifest"] == {"path": H.MANIFEST_REL,
                               "sha256": prereg.sha256_file(root / H.MANIFEST_REL)}
    assert rec["prereg_base_commit"] == _g(root, "rev-parse", "HEAD")
    assert rec["networks"] == {H.NETWORKS[0]: {k: man["networks"][H.NETWORKS[0]][k]
                                               for k in H.NET_FIELDS}}
    assert rec["keys"]["a2"]["content_now"] == rec["keys"]["a2"]["content"] == \
        rec2["content"]["a2"]
    assert rec["step2"]["tag_object"] == tag2 and rec["missing"] == man["missing"]
    # committed and tagged: the freeze accepts it, and the record rebuilds item by item
    _g(root, "add", H.PREREG_REL)
    _g(root, "commit", "-q", "-m", "record")
    _g(root, "tag", "-a", H.TAG, "-m", "H4R")
    assert H.freeze_problems(root=root, remote=None) == []
    assert H.step2_record_problems(rec, root) == []
    assert H.record_rebuild_problems(rec, root) == []
    for k, v in (("regime_typing", {"v1": "G3", "v3": "G3"}), ("opt_fields", None),
                 ("keys", {**rec["keys"], "a2_twins": {"ids": [], "content": None}}),
                 ("branch_tolerance", {**rec["branch_tolerance"], "factor": 1.0})):
        assert H.record_rebuild_problems({**rec, k: v}, root) == [
            ("the record is not what build_record gives from the tagged manifest and step 2's "
             f"record: [{k!r}]")]
    assert H.record_rebuild_problems({**rec, "tests": {}}, root) == []    # informative only
    # the post-tag verification wires it in (with the re-hash and step 2's copies)
    assert H._verify_on_disk(rec, root) == []
    assert H._verify_on_disk({**rec, "regime_typing": None}, root) == [
        ("the record is not what build_record gives from the tagged manifest and step 2's "
         "record: ['regime_typing']")]
    monkeypatch.setattr(H, "step2_freeze_problems", lambda: ["step 2: x"])
    assert H._verify_on_disk(rec, root) == ["step 2: x"]
    monkeypatch.setattr(H, "step2_freeze_problems", list)
    (root / rec["networks"][H.NETWORKS[0]]["eval_path"]).write_text("changed")
    assert H.freeze_problems(root=root, remote=None, verify=H._verify_on_disk) == [
        f"{H.NETWORKS[0]}: {H.NETWORKS[0]}/eval_rdev.json is not the recorded evaluation"]
    # step 2's record without a calibrated tolerance; no missing run (no acceptance block)
    root2, *_ = _prereg_repo(tmp_path / "t", monkeypatch, tol={}, lost=False)
    refused(root2, match="no calibrated branch tolerance")
    root3, *_ = _prereg_repo(tmp_path / "n", monkeypatch, lost=False)
    H.stage_prereg(p.parse_args(["prereg"]))
    assert "missing_accepted" not in json.loads((root3 / H.PREREG_REL).read_text())


# ------------------------------------------------------------------------- the analysis stage


def test_analyse_stage_runs_once_on_complete_stamped_inputs(tmp_path, monkeypatch):
    """The analysis stage on synthetic, stamped inputs (the freeze check replaced by a
    record): refuses an incomplete, partial or other-script input, another checkpoint's
    harness, an incomplete or foreign gate2-trained file, other optimum rows, step 2's
    decisions changed, missing or changed arrays, and writes nothing; then writes its
    decisions once to the fixed path and refuses a second run."""
    from apexzero.fields import network as N

    out, s2root = tmp_path / "out", tmp_path / "s2"
    monkeypatch.setattr(H, "OUT", out)
    monkeypatch.setattr(H, "STEP2_OUT", s2root)
    monkeypatch.setattr(H, "ROOT", tmp_path)
    monkeypatch.setattr(H, "BOOTSTRAP_N", 100)
    missing = {H.run_name("v3", H.BASE, 13): "no run directory"}
    rows = _all_rows(drop=missing)
    opt = _opt_rows(next(iter(rows.values())))
    a2 = [["nominal", k] for k in opt]
    g2keys = sorted(opt)[:12]
    pins, failing = {}, {H.run_name("v1", H.REAL, 11): g2keys[:1]}
    for i, name in enumerate(rows):
        ev = tmp_path / name / "eval_rdev.json"
        ev.parent.mkdir(parents=True)
        ev.write_text(json.dumps(_eval(9 if H.parse_name(name)[1] == H.REAL else 7)))
        pins[name] = {"path": f"{name}/ckpt/x.pkl", "sha256": f"sha{i}", "timesteps": 7,
                      "eval_path": f"{name}/eval_rdev.json",
                      "eval_sha256": prereg.sha256_file(ev)}
    S.write_rows(s2root / "opt-fields" / "rows.jsonl", list(opt.values()))
    s2d = s2root / "analyse" / "decisions.json"
    s2d.parent.mkdir(parents=True)
    s2d.write_text(json.dumps({"statistics": {n: 0.7 for n in A.h4_networks()}}))
    rec = {"networks": pins, "missing": missing,
           "keys": {"a2": {"ids": a2}, "gate2_trained": {"ids": [["nominal", k]
                                                                for k in g2keys]}},
           "opt_fields": {"rows_sha256": prereg.sha256_file(s2root / "opt-fields/rows.jsonl")},
           "step2": {"decisions": {"sha256": prereg.sha256_file(s2d)}},
           "environment": F.environment(), "regime_cells": None}

    def post_tag(args, what, x64=True):
        H._prov().tag = TAG_ID
        return rec

    monkeypatch.setattr(H, "_post_tag", post_tag)
    H._PROV[:] = [H.Provenance(H.build_parser().parse_args(["fields"]), "fields")]
    H._prov().tag = TAG_ID
    for name, rs in rows.items():
        net = S.expected_net(name, pins[name])
        d = S.arrays_dir(out / "fields" / "arrays", net)
        for r in rs:
            f = S.save_key_arrays(d, r["key_id"], {"impulse": np.zeros((4, 3, 2))})
            r.update(net=net, ckpt_sha256=pins[name]["sha256"],
                     arrays_sha256=prereg.sha256_file(f))
        H._write_rows(out / "fields" / f"{H._file(name)}.jsonl", rs)
        H._stamped_write(F.harness_path(out, name), lambda q, s=pins[name]["sha256"]: N.save_det(
            q, {"x": np.zeros(2)}, {"ckpt_sha256": s}))
    H._write(out / "gate2-trained" / "gate2_trained.json",
             {"keys": g2keys, "networks": {n: {
                 "net": S.expected_net(n, pins[n]), "ckpt_sha256": pins[n]["sha256"],
                 "pass": n not in failing,
                 "rows": [{"key_id": k, "pass": k not in failing.get(n, [])} for k in g2keys]}
                 for n in rows}})
    decisions = out / "analyse" / "decisions.json"

    def run(*argv):
        full = ["analyse", "--log-dir", str(tmp_path), *argv]
        monkeypatch.setattr(sys, "argv", ["scripts/h4r.py", *full])
        args = H.build_parser().parse_args(full)
        H._PROV[:] = [H.Provenance(args, "analyse")]
        H.stage_analyse(args)

    # an incomplete input (one key's row lost): refused, nothing written
    f = out / "fields" / f"{H._file(H.NETWORKS[0])}.jsonl"
    full = f.read_text()
    f.write_text("".join(full.splitlines(keepends=True)[1:]))
    with pytest.raises(SystemExit, match="complete recorded inputs") as e:
        run()
    assert "frozen keys without a row" in str(e.value)
    assert "changed after it was stamped" in str(e.value)
    f.write_text(full)
    # a partial run's stamp, another script's stamp, a missing harness file
    sc = prereg.stamp_path(f)
    st = json.loads(sc.read_text())
    sc.write_text(json.dumps({**st, "partial": ["--limit"]}))
    with pytest.raises(SystemExit, match="a partial run"):
        run()
    sc.write_text(json.dumps({**st, "h4r_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="written by another scripts/h4r.py"):
        run()
    sc.write_text(json.dumps(st))
    hp = F.harness_path(out, H.NETWORKS[1])
    hb = hp.read_bytes()
    hp.unlink()
    with pytest.raises(SystemExit, match="no harness-f32/"):
        run()
    hp.write_bytes(hb)
    # an evaluation that is not the recorded one
    ev = tmp_path / H.NETWORKS[2] / "eval_rdev.json"
    evt = ev.read_text()
    ev.write_text(json.dumps(_eval(12)))
    with pytest.raises(SystemExit, match="not the recorded evaluation"):
        run()
    ev.write_text(evt)
    assert not decisions.exists()

    def refuse(msg):
        with pytest.raises(SystemExit, match="complete recorded inputs") as e:
            run()
        assert msg in str(e.value), str(e.value)
        assert not decisions.exists()

    # another checkpoint's harness reference, stamped by this code
    n3 = H.NETWORKS[3]
    hp = F.harness_path(out, n3)
    H._stamped_write(hp, lambda q: N.save_det(q, {"x": np.zeros(2)}, {"ckpt_sha256": "other"}))
    refuse(f"{n3}: {hp.name} is not this checkpoint's harness reference")
    H._stamped_write(hp, lambda q: N.save_det(q, {"x": np.zeros(2)},
                                              {"ckpt_sha256": pins[n3]["sha256"]}))
    # gate2-trained: each edit re-stamped, so that only the targeted check can fire
    g2p = out / "gate2-trained" / "gate2_trained.json"
    g2b = g2p.read_bytes()
    g2d = {k: v for k, v in json.loads(g2b).items() if k != "provenance"}
    n0, n4 = H.NETWORKS[0], H.NETWORKS[4]
    e0 = g2d["networks"][n0]
    H._write(g2p, {**g2d, "networks": {k: v for k, v in g2d["networks"].items() if k != n4}})
    refuse(f"gate2-trained lacks {n4}")
    H._write(g2p, {**g2d, "keys": g2d["keys"][1:]})
    refuse("gate2-trained: its keys are not the record's")
    foreign = f"gate2-trained: {n0} is not from its recorded checkpoint"
    for entry in ({**e0, "ckpt_sha256": "other"}, {**e0, "net": "other@7"},
                  {**e0, "rows": e0["rows"][1:]},
                  {**e0, "rows": [{**r, "pass": "yes"} for r in e0["rows"]]}):
        H._write(g2p, {**g2d, "networks": {**g2d["networks"], n0: entry}})
        refuse(foreign)
    g2p.unlink()
    refuse("no gate2-trained results")
    g2p.write_bytes(g2b)                       # the original bytes carry the original stamp
    # step 2's optimum rows: other bytes; then one A2 key without its row (sha re-recorded)
    optp = s2root / "opt-fields" / "rows.jsonl"
    ob = optp.read_bytes()
    optp.write_bytes(ob + b"\n")
    refuse("step 2's opt-fields rows are not the recorded ones")
    S.write_rows(optp, list(opt.values())[1:])
    rec["opt_fields"]["rows_sha256"] = prereg.sha256_file(optp)
    refuse("1 frozen A2 keys without an optimum row")
    optp.write_bytes(ob)
    rec["opt_fields"]["rows_sha256"] = prereg.sha256_file(optp)
    # step 2's decisions changed
    s2b = s2d.read_bytes()
    s2d.write_bytes(s2b + b" ")
    refuse("step 2's decisions.json is not its committed confirmatory file")
    s2d.write_bytes(s2b)
    # one key's arrays changed, then missing
    npz = min((out / "fields" / "arrays").rglob("*.npz"))
    nb = npz.read_bytes()
    np.savez(npz, impulse=np.ones((4, 3, 2)))
    refuse("arrays differ from their rows' sha256")
    npz.unlink()
    refuse("rows without their arrays")
    npz.write_bytes(nb)
    run("--check")
    assert not decisions.exists()
    run()
    d = json.loads(decisions.read_text())
    assert d["confirmatory"] and d["H4R_G"]["decision"] == "confirm"
    assert d["H4R_B"]["decision"] == "confirm"
    assert d["H4R_G"]["n_relabellings"] == 252 * 126
    assert d["missing"] == missing and d["status"][next(iter(missing))].startswith("missing")
    assert d["networks"][H.run_name("v1", H.REAL, 11)]["coverage"]["gate2_keys"] == g2keys[:1]
    assert d["secondary"]["S1"]["decision"] == "confirm"
    assert d["provenance"]["stage"] == "analyse" and d["provenance"]["tag"] == TAG_ID
    assert d["provenance"]["h4r_sha256"] == H._script_sha(ROOT)
    assert set(d["inputs"]) >= {"gate2-trained/gate2_trained.json",
                                "step2:opt-fields/rows.jsonl"}
    sha = prereg.sha256_file(decisions)
    with pytest.raises(SystemExit, match="runs once"):
        run()
    assert prereg.sha256_file(decisions) == sha


# ------------------------------------------------------------------------- post-tag stages, end to end


def _stages_child(work: str) -> None:
    """Run by test_post_tag_stages_end_to_end in its own process (x64 and the trained-network
    lock are process-wide): harness-f32 (x64 off), gate2-trained, then fields twice on a
    RANDOM actor's checkpoint (never a trained network), one frozen A2 key and one
    gate2-trained key of step 2's record, the freeze check replaced by a stub; then the
    analysis stage's input checks on what they wrote. Prints a JSON report."""
    import contextlib
    import dataclasses
    import io
    import pickle

    import jax
    import jax.numpy as jnp

    from apexzero.agents.ppo import Actor, PPOConfig
    from apexzero.env import racing
    from apexzero.fields import network as N

    work = Path(work)
    name, ts = H.NETWORKS[0], 204_800
    ck = work / name / "ckpt" / f"{ts:012d}.pkl"
    cfg = PPOConfig(network="mlp")
    params = jax.device_get(Actor(cfg, 2, False).init(jax.random.PRNGKey(0),
                                                      jnp.zeros(racing.OBS_DIM)))
    snap = {"actor": params, "critic": {},
            "obs_rms": {"mean": np.zeros(racing.OBS_DIM, np.float32),
                        "var": np.ones(racing.OBS_DIM, np.float32), "count": np.float32(10.0)},
            "ret_rms": {"mean": np.float32(0), "var": np.float32(1), "count": np.float32(1)},
            "config": dataclasses.asdict(cfg), "timesteps": ts}
    ck.parent.mkdir(parents=True)
    with open(ck, "wb") as fh:
        pickle.dump(snap, fh)
    r2 = json.loads((ROOT / H.STEP2_RECORD_REL).read_text())
    rec = {"networks": {name: {"path": f"{name}/ckpt/{ck.name}",
                               "sha256": prereg.sha256_file(ck), "timesteps": ts}},
           "keys": {"a2": {"ids": r2["key_sets"]["a2"][:1]},
                    "gate2_trained": {"ids": r2["gate2_trained"][:1]}},
           "branch_tolerance": {"tolerance": r2["calibration"]["tolerance"]}}
    x64 = []

    def post_tag(args, what, x64_on=True):
        H._prov().tag = TAG_ID
        x64.append(bool(jax.config.jax_enable_x64))
        if x64_on:
            F._x64()
            N.unlock_trained()
        return rec

    H._post_tag = lambda args, what, x64=True: post_tag(args, what, x64)
    H.ROOT, H.OUT = work, work / "out"

    def run(stage):
        args = H.build_parser().parse_args([stage, "--log-dir", str(work)])
        H._PROV[:] = [H.Provenance(args, stage)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            getattr(H, "stage_" + stage.replace("-", "_"))(args)
        return buf.getvalue()

    run("harness-f32")
    run("gate2-trained")
    run("fields")
    rows = H.OUT / "fields" / f"{H._file(name)}.jsonl"
    sha = prereg.sha256_file(rows)
    second = run("fields")
    args = H.build_parser().parse_args(["analyse", "--log-dir", str(work)])
    H._PROV[:] = [H.Provenance(args, "analyse")]
    H._prov().tag = TAG_ID
    paths, probs = H._inputs(rec, H.OUT)
    probs += F.input_stamp_problems(paths, H._prov().code_hash, None, TAG_ID)
    probs += H._stamp_extra_problems(paths)
    g2 = json.loads((H.OUT / "gate2-trained" / "gate2_trained.json").read_text())
    print(json.dumps({"x64_at_entry": x64, "second_fields": second,
                      "rows_unchanged": prereg.sha256_file(rows) == sha,
                      "inputs": sorted(p.name for p in paths), "problems": probs,
                      "gate2": {k: g2["networks"][name]["h4r"][k]
                                for k in ("n_keys", "n_failing", "descriptive")},
                      "rows": S.read_rows(rows)}, default=str))


def test_post_tag_stages_end_to_end(tmp_path):
    """harness-f32, gate2-trained and fields as they will run after the tag (only the freeze
    check stubbed), on a random actor: they write stamped outputs the analysis accepts, the
    second fields run resumes without recomputing, and the analysis inputs are complete."""
    from apexzero.interp import io

    r2 = json.loads((ROOT / H.STEP2_RECORD_REL).read_text())
    need = [r2["key_sets"]["a2"][0], r2["gate2_trained"][0]]
    if not all((io.KEYS_DIR / kind / f"{kid}.json").exists() for kind, kid in need):
        pytest.skip("step 1's key store (outputs/psychophysics/keys) is not on this machine")
    code = ("import importlib.util; "
            f"spec = importlib.util.spec_from_file_location('t', {str(Path(__file__))!r}); "
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            f"m._stages_child({str(tmp_path)!r})")
    import os

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])])}
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=600, check=False)
    assert r.returncode == 0, r.stderr[-4000:]
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep["x64_at_entry"][0] is False                 # harness-f32 ran with x64 off
    assert "complete stamped rows" in rep["second_fields"] and rep["rows_unchanged"]
    assert rep["problems"] == []
    name = H.NETWORKS[0]
    assert rep["inputs"] == sorted([f"{H._file(name)}.jsonl", f"{H._file(name)}.npz",
                                    "gate2_trained.json"])
    assert rep["gate2"]["n_keys"] == 1
    (row,) = rep["rows"]
    assert row["key_id"] == r2["key_sets"]["a2"][0][1]
    assert row["net"] == f"{Path(name).name}@204800"
    # a full row, not a pipeline failure: Gate 3 and branch consistency ran against the harness
    # reference with step 2's tolerance, and the row binds its arrays
    assert "pipeline" not in row and row["gate3"]["ok"] is True and row["branch"]["ok"] is True
    assert row["branch"]["tol"] == pytest.approx(r2["calibration"]["tolerance"])
    assert row["arrays_sha256"] and row["ckpt_sha256"]
