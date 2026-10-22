"""Step 2's batch-stage bodies (fields.stages) and decision rules (fields.analysis), built and
tested before the tag (design Section 9 freezes the code that computes and decides).

- The decision rules on synthetic data with known answers: H4's exact within-variant
  permutation test against brute-force enumeration (the decision without the seed-0
  baselines, 4,900 relabellings; the sensitivity with them, 15,876; H4-transfer 495 and
  1,287), the 'not robust' label and 'not testable' on coverage, R18's 2-of-3 truth table
  with descriptive seeds, H2-speed's bands, T's labels, the descriptive H1-typical and
  H1-fast, exclusions in their order (pipeline failures first), lambda* and lambda*_lin, the
  regime cell sets against step 1's ``regime_of``.
- The analysis assembly on synthetic rows: the 30% coverage rule (H2' descriptive, H4 not
  testable), gate2-trained failures, H4 on every regime with the descriptive per-regime
  breakdown, the capped stratum (H2', H4, H1-typical, H1-fast and H2-speed), pipeline
  exclusions (no optimum row, branch, loop), the plumbing-only run, the run-once input
  checks, the descriptive items (other bump widths, steer_0, anchored-key H2', near-ties),
  and scripts/fields.py's analyse stage end to end on stamped synthetic inputs (refusing
  incomplete or partial inputs, writing its decisions once).
- The optimum's per-key fields stage on stored keys (A2 and anchored; optimum side only), the
  phase ceiling of one cell, the dose and gate2-trained subsamples, H3's twins and the H3 key
  driver's plumbing (step 1's recipes faked: no solve here), into step 2's own key cache.
- gate2-trained's pass rule (every key checked and passing; an exhausted key fails the
  network), the s_1-anchored grid of H1-fast's gain and steer_0, the arrays the stamped rows
  bind, the confirmatory unlock and the input-stamp refusals. The network stage bodies end
  to end in a float64 process are in tests/test_fields_dryrun.py.
This process keeps x64 off (the optimum's float64 JAX runs inside ``jax.enable_x64`` blocks)."""

import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from apexzero.fields import analysis as A
from apexzero.fields import keys as K
from apexzero.fields import measures as MS
from apexzero.fields import stages as S
from apexzero.interp import agents, io
from apexzero.interp import analysis as step1

SRC = Path(S.__file__).resolve().parents[2]
HAVE_KEYS = (io.KEYS_DIR / "anchored").exists() and (io.STIMULI_DIR / "cells.json").exists()
needs_keys = pytest.mark.skipif(not HAVE_KEYS, reason="step 1's keys are not on this machine")


def _have_all() -> bool:
    try:
        return HAVE_KEYS and agents.final_snapshot(agents.AGENTS["G1"]) is not None \
            and (io.ROLLOUTS_DIR / "G2").exists()
    except (KeyError, OSError):
        return False


@pytest.fixture(autouse=True)
def _restore_x64():
    prev = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", prev)


# ------------------------------------------------------------------------- decision rules


def _brute(groups):
    """Every joint relabelling by brute force (no shared code with analysis._diffs)."""
    per = []
    for real, base in groups.values():
        v = list(real) + list(base)
        k = len(real)
        per.append([np.mean([v[i] for i in c]) - np.mean([v[i] for i in range(len(v)) if i not in c])
                    for c in itertools.combinations(range(len(v)), k)])
    return np.array([sum(t) for t in itertools.product(*per)])


def _h4_stats(rng, real=0.6, base=0.3):
    stats = {}
    for d in A.H4_VARIANTS.values():
        for r in d["real"]:
            stats[r] = real + 0.05 * rng.standard_normal()
        for r in d["baseline"]:
            stats[r] = base + 0.05 * rng.standard_normal()
    for r in A.H4_TRANSFER["real"]:
        stats[r] = real + 0.05 * rng.standard_normal()
    return stats


def test_h4_permutation_test_against_brute_force():
    """Design Section 7 (v3.2): the decision test leaves out the seed-0 baselines (4 vs 4 per
    variant, 4,900 relabellings), the sensitivity includes them (15,876); both enumerated
    exactly (checked by brute force); H4-transfer decides on 495 and reports 1,287."""
    rng = np.random.default_rng(4)
    stats = _h4_stats(rng)
    res = A.h4_test(stats)
    main, sens = res["main"], res["sensitivity"]
    assert main["n_relabellings"] == 70 ** 2 == 4900 and sens["n_relabellings"] == 126 ** 2
    assert len(A.h4_networks()) == 18 and len(A.transfer_networks()) == 13
    for test, with0 in ((main, False), (sens, True)):
        groups = {v: ([stats[r] for r in d["real"]],
                      [stats[r] for r in d["baseline"] if with0 or not A.is_baseline_seed0(r)])
                  for v, d in A.H4_VARIANTS.items()}
        dist = _brute(groups)
        assert len(dist) == test["n_relabellings"]
        t = sum(np.mean(r) - np.mean(b) for r, b in groups.values())
        assert test["T"] == pytest.approx(t)
        assert test["p"] == pytest.approx(np.mean(dist >= t - 1e-12))
        assert test["p_opposite"] == pytest.approx(np.mean(dist <= t + 1e-12))
    # every real network above every baseline: the observed labelling is the most extreme
    assert main["p"] == pytest.approx(1 / 4900) and res["decision"] == "confirm"
    assert sens["p"] == pytest.approx(1 / 15876) and res["robust_to_seed0"] and not res["label"]
    assert all(s_["n_relabellings"] == 70 for s_ in main["strata"].values())
    # the reverse: refuted; equal statistics: inconclusive
    assert A.h4_test({r: -v for r, v in stats.items()})["decision"] == "refute"
    assert A.h4_test(dict.fromkeys(stats, 0.5))["decision"] == "inconclusive"
    assert [A.is_baseline_seed0(r) for r in A.H4_VARIANTS["v1"]["baseline"]] == \
        [True, False, False, False, False]
    tr = A.h4_transfer(stats)
    assert tr["main"]["n_relabellings"] == 495 and tr["sensitivity"]["n_relabellings"] == 1287
    assert tr["decision"] == "confirm" and tr["main"]["p"] == pytest.approx(1 / 495)
    groups = {"t": ([stats[r] for r in A.H4_TRANSFER["real"]],
                    [stats[r] for r in A.H4_TRANSFER["baseline"] if not A.is_baseline_seed0(r)])}
    assert len(_brute(groups)) == 495


def test_h4_decides_confirm_at_p_exactly_005():
    """Design Section 7: confirm if p <= 0.05, and p = 245/4900 = 0.05 can occur. v1's eight
    non-seed-0 networks get distinct powers of two, the real seeds the 7th-largest 4-subset
    by sum (7 relabellings at least as extreme); v3's diffs are +-250, 35 each: p = 7 x 35 /
    4900 exactly."""
    v1, v3 = A.H4_VARIANTS["v1"], A.H4_VARIANTS["v3"]
    nets1 = list(v1["real"]) + [r for r in v1["baseline"] if not A.is_baseline_seed0(r)]
    vals = [2.0 ** k for k in range(8)]
    subsets = sorted(itertools.combinations(range(8), 4), key=lambda c: -sum(vals[i] for i in c))
    real = subsets[6]
    order = list(real) + [i for i in range(8) if i not in real]
    stats = {n: vals[i] for n, i in zip(nets1, order, strict=True)}
    stats[next(r for r in v1["baseline"] if A.is_baseline_seed0(r))] = 0.0
    stats.update({r: 1000.0 if k == 0 else 0.0 for k, r in enumerate(v3["real"])})
    stats.update(dict.fromkeys(v3["baseline"], 0.0))
    res = A.h4_test(stats)
    assert res["main"]["p"] == 0.05 and res["decision"] == "confirm"


def test_h4_sensitivity_label_and_coverage():
    """The seed-0 baselines can flip the decision only in the labelled sensitivity; a network
    that is descriptive (coverage or gate2-trained) or has an undefined statistic makes H4
    and H4-transfer 'not testable (...)', and no test is run."""
    rng = np.random.default_rng(7)
    stats = _h4_stats(rng, 0.5, 0.5)
    # baseline seeds 1-4 low, seed 0 very high: the decision confirms, the sensitivity not
    for d in A.H4_VARIANTS.values():
        for r in d["baseline"]:
            stats[r] = 2.0 if A.is_baseline_seed0(r) else 0.0
        for r in d["real"]:
            stats[r] = 1.0 + 0.01 * rng.standard_normal()
    res = A.h4_test(stats)
    assert res["decision"] == "confirm" and res["sensitivity"]["decision"] != "confirm"
    assert not res["robust_to_seed0"] and res["label"] == A.H4_NOT_ROBUST
    assert res["reported"] == f"confirm ({A.H4_NOT_ROBUST})"
    v3_seed0 = A.H4_VARIANTS["v3"]["baseline"][0]
    tr = A.h4_transfer({**stats, v3_seed0: 6.0,
                        **{r: 1.0 + 0.01 * k for k, r in enumerate(A.H4_TRANSFER["real"])}})
    assert tr["decision"] == "confirm" and tr["main"]["p"] == pytest.approx(1 / 495)
    assert tr["sensitivity"]["decision"] != "confirm" and tr["label"] == A.H4_NOT_ROBUST
    real0 = A.H4_VARIANTS["v1"]["real"][0]
    base0 = A.H4_VARIANTS["v3"]["baseline"][0]
    nt = A.h4_test(stats, descriptive={real0: "coverage", base0: "gate2-trained"})
    assert nt["decision"] == f"not testable (coverage: {real0}; gate2-trained: {base0})"
    assert "main" not in nt
    nan = A.h4_test({**stats, real0: float("nan")})
    assert nan["decision"] == f"not testable (statistic undefined: {real0})"
    assert A.h4_transfer(stats, descriptive={base0: "coverage"})["decision"].startswith(
        "not testable (coverage")
    # a descriptive network outside H4 does not stop it
    assert A.h4_test(stats, descriptive={"G2": "coverage"})["decision"] == "confirm"


def test_descriptive_labels_and_signs():
    assert A.sign_label(-3.0, -0.5) == "below 0" and A.sign_label(0.5, 3.0) == "above 0"
    assert A.sign_label(-1.0, 1.0) == "contains 0" and A.sign_label(np.nan, 1.0) == "undefined"
    assert A.step1_direction("below 0", -1) == "consistent"
    assert A.step1_direction("above 0", -1) == "inconsistent"
    assert A.step1_direction("contains 0", -1) == "undetermined"
    assert A.step1_direction("contains 0", 0) == "consistent"
    assert A.step1_direction("below 0", 0) == "inconsistent"
    assert A.step1_direction("below 0", None) == ""
    assert A.h2_speed(0.1, 0.4) == "under" and A.h2_speed(0.6, 1.8) == "lawful"
    assert A.h2_speed(2.5, 3.0) == "over" and A.h2_speed(0.3, 1.0) == "inconclusive"
    assert A.t_label(-0.2, 0.4) == "width-blind" and A.t_label(0.85, 1.1) == "corridor-anchored"
    assert A.t_label(0.3, 0.9) == "mixed" and A.t_label(np.nan, 1.0) == "undefined"
    assert not hasattr(A, "equivalence") and not hasattr(A, "H1_EQUIV")
    # the boundaries (design Sections 7 and 15): p <= 0.05 confirms, the opposite p <= 0.05
    # refutes; T's upper end <= 0.5 is width-blind, [0.8, 1.2] inclusive corridor-anchored
    assert A.decide(0.05, 1.0) == "confirm" and A.decide(0.5, 0.05) == "refute"
    assert A.decide(0.0501, 0.0501) == "inconclusive"
    assert A.t_label(0.1, 0.5) == "width-blind" and A.t_label(0.8, 1.2) == "corridor-anchored"


def test_r18_variant_rule_truth_table():
    v = "v1"
    s0, s1, s2 = A.registry.variant_seeds(v)
    assert A.variant_rule({s0: "confirm", s1: "confirm", s2: "lawful"}, v) == "confirm"
    assert A.variant_rule({s0: "lawful", s1: "confirm", s2: "confirm"}, v) == "confirm"
    assert A.variant_rule({s0: "confirm", s1: "lawful", s2: "inconclusive"}, v) == "seed-specific"
    assert A.variant_rule({s0: "confirm", s1: "confirm"}, v) == "confirm"
    assert A.variant_rule({s0: "confirm"}, v) == "seed-specific"
    assert A.variant_rule({}, v) == "seed-specific"
    # H2': a descriptive seed has no verdict; two descriptive seeds make the variant descriptive
    ok = {"decision": "confirm", "descriptive": ""}
    cov = {"decision": A.DESCRIPTIVE_COVERAGE, "descriptive": "coverage"}
    assert A.h2_variant({s0: ok, s1: ok, s2: cov}, v) == "confirm"
    assert A.h2_variant({s0: ok, s1: cov, s2: {**ok, "decision": "lawful"}}, v) == "seed-specific"
    assert A.h2_variant({s0: ok, s1: cov, s2: cov}, v) == A.DESCRIPTIVE_COVERAGE
    g2 = {"decision": A.DESCRIPTIVE_GATE2, "descriptive": "gate2-trained"}
    assert A.h2_variant({s0: g2, s1: g2, s2: ok}, v) == A.DESCRIPTIVE_GATE2
    summ = {"pooled": {"lo": 0.1, "hi": 0.4}, "decision": "confirm"}
    r = A.h2_network(summ, {"share_excluded": 0.31, "descriptive": True})
    assert r["decision"] == A.DESCRIPTIVE_COVERAGE and r["band_decision"] == "confirm"
    assert A.h2_network(summ, {"share_excluded": 0.29, "descriptive": False})["decision"] == "confirm"
    r = A.h2_network(summ, {"share_excluded": 0.1, "descriptive": False}, gate2_ok=False)
    assert r["decision"] == A.DESCRIPTIVE_GATE2 and r["band_decision"] == "confirm"


def _rows(fn, n_geo=6, n_phase=4, seed=0):
    """Synthetic key measures: n_geo geometries x 2 directions x n_phase phases."""
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_geo):
        R = 10.0 * 1.6 ** g
        for sig in (1, -1):
            for ph in range(n_phase):
                out.append(MS.KeyMeasure(f"k{g}{sig}{ph}", g, "A2", sig, ph, n_phase, R, 90.0,
                                         float(fn(g, rng))))
    return out


def test_h1_typical_and_fast_are_descriptive():
    """H1-typical and H1-fast report numbers with CIs only (design Section 7, v3.1 / v3.2):
    no pass/fail, no band; H1-fast's signs are labelled against step 1's direction."""
    gain = _rows(lambda g, r: 1.0 + 0.02 * r.standard_normal())
    loc = _rows(lambda g, r: 2.0 * r.standard_normal())
    shape = _rows(lambda g, r: 0.3 + 0.02 * r.standard_normal())
    ctrl = {"gain": _rows(lambda g, r: 0.4 + 0.02 * r.standard_normal()),
            "location": _rows(lambda g, r: 30.0 + r.standard_normal()),
            "shape": _rows(lambda g, r: 0.02 * r.standard_normal())}
    res = A.h1_typical(gain, loc, shape, ctrl, n=1000)
    assert res["descriptive"] and "pass" not in json.dumps(res)
    g = res["criteria"]["gain"]
    assert set(g) >= {"left", "right", "pooled", "vs_control"}
    assert g["pooled"]["lo"] <= 1.0 <= g["pooled"]["hi"]
    # the paired difference against the control: -|g - 1| (net) minus -|0.4 - 1| (control)
    assert g["vs_control"]["pooled"]["median"] == pytest.approx(0.6, abs=0.05)
    assert res["criteria"]["location"]["vs_control"]["pooled"]["median"] > 20
    early = _rows(lambda g, r: -8.0 + r.standard_normal())
    fast = A.h1_fast(early, early, gain, "G1s1", n=1000)
    assert fast["predicted"] == -1 and fast["descriptive"]
    assert fast["s_b"]["sign"] == "below 0" and fast["s_b"]["step1_direction"] == "consistent"
    assert "decision" not in json.dumps(fast) and "pass" not in json.dumps(fast)
    assert fast["gain"]["pooled"]["lo"] <= 1.0 <= fast["gain"]["pooled"]["hi"]
    assert A.h1_fast(early, early, gain, "G2", n=500)["s_b"]["step1_direction"] == "inconsistent"
    s1 = A.h1_fast(early, early, gain, "S1", n=500)
    assert s1["predicted"] is None and s1["s_b"]["step1_direction"] == ""
    # per approach when the rows hold more than one
    mixed = [MS.KeyMeasure(r.key_id + "b", r.pair_id + 100, "B", r.direction, r.phase,
                           r.n_phases, r.R, r.theta_deg, r.value + 20) for r in early]
    both = A.h1_fast(early + mixed, early + mixed, gain, "G3", n=500)
    assert set(both["s_b"]["by_approach"]) == {"A2", "B"}
    assert both["s_b"]["by_approach"]["B"]["pooled"]["median"] == pytest.approx(
        both["s_b"]["by_approach"]["A2"]["pooled"]["median"] + 20, abs=1.5)


def test_key_measures_and_their_exclusions():
    meta = {"key_id": "k", "pair_id": 1, "approach": "A2", "direction": 1, "phase": 0,
            "n_phases": 4, "R": 10.0, "theta_deg": 90.0}
    assert A.key_measure(meta, 0.4, {}).value == 0.4
    km = A.key_measure(meta, 0.4, {"invalid": True, "kappa_clipped": True})
    assert np.isnan(km.value) and km.reason == "invalid outcome"
    assert A.key_measure(meta, 0.4, {"kappa_clipped": True, "off_plateau": True}).reason == \
        "kappa clipping"
    # design Section 15: pipeline failures are exclusions of the key, counted first
    assert A.key_measure(meta, 0.4, {"invalid": True, "why_invalid": MS.LOOP_EXHAUSTED}
                         ).reason == MS.PIPELINE_LOOP
    assert A.key_measure(meta, 0.4, {"no_optimum": True}).reason == MS.PIPELINE_NO_OPT
    assert A.key_measure(meta, 0.4, {"branch_failed": True, "invalid": True}).reason == \
        MS.PIPELINE_BRANCH
    assert A.key_measure(meta, 0.4, {"gate4_failed": True}).reason == "gate 4"
    recs = A.coverage_records([km], "G1", "1", "G")
    assert MS.coverage(recs)["G1@1"]["G"]["excluded"] == {"invalid outcome": 1}


def test_h3_lambda_rules():
    assert A.lambda_star(A.H3_LAMBDAS, [lam >= 0.35 for lam in A.H3_LAMBDAS]) == pytest.approx(0.35)
    assert A.lambda_star([0.0, 0.5], [False, False]) == float("inf")
    assert len(A.H3_LAMBDAS) == 21 and A.H3_LAMBDAS[1] == pytest.approx(0.05)
    n = 50
    imp = MS.Samples(np.zeros(n), np.r_[np.zeros(20), np.full(5, 0.2), np.zeros(25)],
                     np.r_[np.zeros(20), np.full(5, 0.1), np.zeros(25)])
    ex = MS.Samples(np.zeros(n), np.r_[np.zeros(20), np.full(5, 0.15), np.zeros(25)], np.zeros(n))
    dw = MS.Samples(np.zeros(n), -np.ones(n), -np.ones(n))
    lin = A.lambda_lin(0.6, imp, ex, dw)
    assert lin["dc_dlambda"] == pytest.approx(-1.5)
    assert lin["lambda_lin"] == pytest.approx(0.4)
    assert lin["lambda_lin_left"] == pytest.approx(0.6) and lin["lambda_lin_right"] == pytest.approx(1.2)
    assert lin["edge_move"] == pytest.approx(-0.75) and lin["line_move"] == pytest.approx(-0.75)
    widen = A.lambda_lin(0.6, imp, None, MS.Samples(np.zeros(n), np.ones(n), np.ones(n)))
    assert widen["lambda_lin"] == float("inf")


@needs_keys
def test_regime_cell_sets_follow_step1_typing():
    from apexzero.interp import support

    bank, cells = K.design()
    regimes = support.load_support(io.SUPPORT_DIR).regimes
    a2 = {r.cell_id for r in K.a2_primary_refs()}
    for agent in ("G1", "G3"):
        sets = A.a2_cells_by_regime(agent, bank, cells, regimes, a2)
        assert sum(len(v) for v in sets.values()) == len(a2)
        rt = regimes[agent]
        for reg, cids in sets.items():
            for c in cids[:: max(1, len(cids) // 5)]:
                R = bank.stimuli[cells[c].stim_id].geometry.R_seg
                assert step1.regime_of(R, rt.q1, rt.q10, rt.r_clip_max) == reg
        assert "bulk" in sets and len(sets["bulk"]) > 30
    # G3's grammar supports tighter corners: no fewer bulk cells than G1's
    s1 = A.a2_cells_by_regime("G1", bank, cells, regimes, a2)
    s3 = A.a2_cells_by_regime("G3", bank, cells, regimes, a2)
    assert len(s3["bulk"]) >= len(s1["bulk"])
    assert A.H4_VARIANTS["v1"]["regime_of"] == "G1" and A.H4_VARIANTS["v3"]["regime_of"] == "G3"


# ------------------------------------------------------------------------- optimum stages


@pytest.fixture(scope="module")
def opt(tmp_path_factory):
    """The optimum's fields stage on stored keys (optimum side only): phases 0 and 1 of four
    A2 cells spread over R (W = 400, both phases keyed), plus one anchored final."""
    if not HAVE_KEYS:
        pytest.skip("step 1's keys are not on this machine")
    bank, cells = K.design()
    a2 = [r for r in K.a2_primary_refs() if r.W == 400]
    by = {}
    for r in a2:
        by.setdefault(r.cell_id, {})[r.phase] = r
    both = sorted((c for c, ph in by.items() if {0, 1} <= set(ph)),
                  key=lambda c: K.strata(by[c][0]).R)
    pick = [both[int(i)] for i in np.linspace(0, len(both) - 1, 4)]
    refs = [by[c][k] for c in pick for k in (0, 1)] + [K.anchored_finals()["G2"][50]]
    out = tmp_path_factory.mktemp("opt")
    rows = S.opt_fields(refs, out, bank)
    return SimpleNamespace(rows=rows, dir=out, cells=pick, refs=refs, bank=bank)


def test_optimum_fields_stage_and_phase_ceiling(opt):
    """Per-key optimum fields with their flags (A2 and anchored keys), persisted and read
    back; the phase ceiling of each cell from the stored fields."""
    for r in opt.rows:
        assert not r["failed"] and r["repolish_ok"] and r["adjoint_ok"] and r["inertia_ok"]
        assert r["max_adjoint_residual"] <= 1e-10 and r["inertia_determined"]
        arr = S.load_key_arrays(opt.dir, r["key_id"])
        assert arr["impulse"].shape == (4, 3, r["n_samples"]) == arr["explicit"].shape
        if r["kind"] != "anchored":
            assert r["in_band"] and abs(r["G_star"] - 1) < 1e-6 and r["on_plateau"]
    anch = opt.rows[-1]
    assert anch["decision"] == 1 and anch["kind"] == "anchored"
    pc = S.phase_ceiling_rows(opt.rows, opt.dir)
    assert [c["cell"] for c in pc] == sorted(opt.cells)
    assert all(c["v_apex"]["n_pairs"] == 1 and c["v_apex"]["rho_median"] > 0.5 for c in pc), pc


def test_shape_specificity_through_the_optimum_pipeline(opt):
    """H1-typical's shape criterion validated on the optimum's own fields (design Section 8):
    a cell's other phase used as the 'network' field passes (delta_rho > 0), a far
    geometry's field substituted for it fails."""
    rows = [r for r in opt.rows if r["kind"] != "anchored"]
    j = S.OPT_OUTCOMES.index("v_apex")

    def fmap(phase):
        return {r["cell"]: ((r["R"], r["theta_deg"]), r["s_c0"], r["direction"],
                            S.samples_of(S.load_key_arrays(opt.dir, r["key_id"])["impulse"], j))
                for r in rows if r["phase"] == phase and r["valid"]["v_apex"]}

    ref, other = fmap(0), fmap(1)
    pos = MS.shape_specificity(other, ref, "v_apex")
    assert len(pos) == 4 and all(p["n_far"] >= 1 for p in pos)
    assert all(p["delta_rho"] > 0 for p in pos), pos
    far = {}
    keys = sorted(other, key=lambda c: other[c][0][0])
    for a, b in zip(keys, keys[::-1], strict=True):        # smallest R <-> largest R
        far[a] = (other[a][0], other[a][1], other[a][2], other[b][3])
    neg = {p["label"]: p for p in MS.shape_specificity(far, ref, "v_apex")}
    assert neg[keys[0]]["delta_rho"] < 0 and neg[keys[-1]]["delta_rho"] < 0, neg


@needs_keys
def test_a_failed_repolish_is_a_numerical_row(monkeypatch):
    """A key whose re-polish fails is recorded as such, never computed at the stored point."""
    orig = K.repolish

    def failing(prob):
        rp = orig(prob)
        rp.status = "Maximum_Iterations_Exceeded"
        return rp

    monkeypatch.setattr(K, "repolish", failing)
    ref = [r for r in K.a2_primary_refs() if r.W == 400][7]
    row, arrays = S.opt_fields_key(ref)
    assert row["failed"] and row["reason"] == S.NUMERICAL and not arrays
    assert "RepolishFailed" in row["error"]
    # any other exception is a crash, never a silent 'numerical' row: it stops the stage

    def broken(prob):
        raise KeyError("a corrupt key file")

    monkeypatch.setattr(K, "repolish", broken)
    with pytest.raises(KeyError, match="corrupt"):
        S.opt_fields_key(ref)


@needs_keys
def test_dose_sample_and_h3_twins(tmp_path):
    a, b = S.dose_sample(), S.dose_sample()
    assert [r.key_id for r in a] == [r.key_id for r in b] and len(a) == S.DOSE_N
    assert {A.registry.AGENTS.get(r.agent) is None for r in a if r.kind != "anchored"} == {True}
    bank, cells = K.design()
    pairs = A.h3_pairs(bank, cells)
    assert set(pairs) == set(A.H3_CELLS)
    for c, sid in pairs.items():
        assert bank.stimuli[sid].name.endswith("/real")
        dw = A.width_change(bank, cells[c].stim_id, sid)
        assert np.all(dw.kappa == 0) and np.abs(np.r_[dw.w_left, dw.w_right]).max() > 0.5
    with pytest.raises(ValueError, match="step 1's key store"):
        S.h3_cache(io.KEYS_DIR)


@needs_keys
def test_h3_key_driver_plumbing(tmp_path, monkeypatch):
    """The H3 driver calls step 1's recipes (faked here: no solve) per cell and phase, into
    step 2's own cache, through step 1's lockbox guard."""
    from apexzero.interp import answer_key as ak

    calls = []
    fake = SimpleNamespace(key_id="k", status="verified")
    monkeypatch.setattr(ak, "full_recipe", lambda bank, cell, **kw: calls.append(
        ("nominal", cell.cell_id, kw["cache"])) or fake)
    monkeypatch.setattr(ak, "phase_cycle", lambda bank, cell, nominal=None: 10.0)
    monkeypatch.setattr(ak, "warm_key", lambda nom, bank, c, kind, phase=0, **kw: calls.append(
        (kind, c.cell_id, phase)) or fake)
    guarded = []
    monkeypatch.setattr(ak, "_guard", lambda stim: guarded.append(stim.name))
    cache = S.h3_cache(tmp_path / "keys")
    out = S.h3_base_keys(cache, tables=object())
    _, cells = K.design()
    n = sum(cells[c].n_phases for c in A.H3_CELLS)
    assert len(out) == n and len(guarded) == len(A.H3_CELLS)
    assert [c for c in calls if c[0] == "nominal"] == [("nominal", c, cache) for c in A.H3_CELLS]
    assert {c[2] for c in calls if c[0] == "phase"} == {1, 2, 3}


# ------------------------------------------------------------------------- network stages


def _script():
    """scripts/fields.py as a module (its stage helpers; nothing runs on import)."""
    import importlib.util

    path = SRC.parent / "scripts" / "fields.py"
    spec = importlib.util.spec_from_file_location("fields_script", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


def test_confirmatory_guard(tmp_path):
    path = S.confirmatory_guard(tmp_path)
    path.write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        S.confirmatory_guard(tmp_path)


# ------------------------------------------------------------------------- analysis assembly


def _net_rows(G, n_geo=6, n_phase=4, excluded=0, cells=None, capped_geo=None, G_out=None,
              seed=0):
    """Synthetic fields-a2 rows: n_geo geometries x 2 directions x n_phase phases, G from
    ``G(geo, rng)``; the first ``excluded`` keys have an invalid c_min; ``cells(geo)`` gives
    a row's cell id; geometry ``capped_geo`` is a capped stimulus."""
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_geo):
        for sig in (1, -1):
            for ph in range(n_phase):
                i = len(out)
                ok = i >= excluded
                gv = float(G(g, rng))
                go = 1.0 - gv if G_out is None else float(G_out)
                out.append({"key_id": f"k{g}{sig}{ph}", "pair_id": g, "approach": "A2",
                            "direction": sig, "phase": ph, "n_phases": n_phase,
                            "R": 10.0 * 1.6 ** g, "theta_deg": 90.0,
                            "cell": cells(g) if cells else 100 + g, "w_is_cell": True,
                            "cap_binds": g == capped_geo,
                            "valid": {"c_min": ok, "v_apex": ok, "s_b": ok, "steer": True},
                            "reasons": {"c_min": "" if ok else "crash", "v_apex": "",
                                        "s_b": "", "steer": ""},
                            "gate3": {"ok": True}, "branch": None,
                            "kappa_clip": {"c_min": False, "v_apex": False, "s_b": False,
                                           "steer": False},
                            "on_plateau": True, "G": gv if ok else None,
                            "G_out": go if ok else None, "B": 0.5 if ok else None})
    return out


def _opt_rows(rows):
    return {r["key_id"]: {"key_id": r["key_id"], "kind": "nominal", "failed": False,
                          "adjoint_ok": True, "inertia_failed": False, "in_band": True,
                          "valid": {"c_min": True}, "w_is_cell": True, "cap_binds": False,
                          **{k: r[k] for k in ("pair_id", "approach", "direction", "phase",
                                               "n_phases", "R", "theta_deg")},
                          "G_star": 1.0, "T_star": 1.0}
            for r in rows}


def _all_networks(G=lambda g, r: 0.3 + 0.01 * r.standard_normal(), **kw):
    return {name: _net_rows(G, seed=i, **kw) for i, name in enumerate(A.A2_NETWORKS)}


def test_analyse_edges_coverage_rule_and_h4_data():
    """Design Section 15 in analyse_edges: a network with more than 30% of its keys excluded
    (every reason, pipeline failures included) is 'descriptive (coverage)' for H2' and makes
    H4 'not testable'; a failed gate2-trained network likewise; H4's data are every primary,
    non-capped A2 key (a tail-cell row moves its statistic; regime typing is descriptive)."""
    nets = _all_networks()
    opt = _opt_rows(next(iter(nets.values())))
    out = S.analyse_edges(nets, opt, n=200)
    assert out["decided"] and set(out["H2"]) == set(A.H2_NETWORKS)
    assert all(v["decision"] == "confirm" for v in out["H2"].values())
    assert all(v == "confirm" for v in out["H2_variants"].values())
    assert out["H4"]["decision"] == "inconclusive" and out["H4"]["main"]["n_relabellings"] == 4900
    assert out["H4_transfer"]["main"]["n_relabellings"] == 495
    t = out["edge_descriptive"]["G1"]["T"]
    assert t["pooled"]["median"] == pytest.approx(1.0) and t["label"] == "corridor-anchored"
    assert out["edge_descriptive"]["G1"]["asymmetry_G"]["n_pairs"] == 6
    for m in ("G", "T", "G_out", "B"):           # every reported measure has its index
        a = out["edge_descriptive"]["G1"][m]["asymmetry"]
        assert a["n_pairs"] == 6 and a["n_phases"] == 24 and np.isfinite(a["est"]), m
    assert out["H2"]["G1"]["asymmetry"]["n_pairs"] == 6
    assert out["optimum"]["T_star"]["pooled"]["median"] == pytest.approx(1.0)
    # 15 of 48 keys excluded (31%): descriptive for H2' (band kept) and H4 not testable
    g1s1 = "G1s1"
    nets2 = {**nets, g1s1: _net_rows(lambda g, r: 0.3, excluded=15)}
    out2 = S.analyse_edges(nets2, opt, n=200)
    assert out2["coverage"]["G1s1@final"]["G"]["share_excluded"] == pytest.approx(15 / 48)
    assert out2["H2"][g1s1]["decision"] == A.DESCRIPTIVE_COVERAGE
    assert out2["H2"][g1s1]["band_decision"] == "confirm"
    assert out2["H2_variants"]["v1"] == "confirm"           # from the other two seeds
    base = A.H4_VARIANTS["v1"]["baseline"][1]                # runs/dr-pilot-200M-s1
    nets3 = {**nets, base: _net_rows(lambda g, r: 0.3, excluded=15)}
    out3 = S.analyse_edges(nets3, opt, n=200)
    assert out3["H4"]["decision"] == f"not testable (coverage: {base})"
    out4 = S.analyse_edges(nets, opt, n=200, gate2_ok={"G2": False, base: False})
    assert out4["H2"]["G2"]["decision"] == A.DESCRIPTIVE_GATE2
    assert out4["H4"]["decision"] == f"not testable (gate2-trained: {base})"
    # H4 uses every regime: raising G on the rows of G1's tail cells moves the statistic
    tail = {name: [dict(r, G=1.5) if r["cell"] in (100, 101, 102, 103) else r for r in rows]
            for name, rows in nets.items()}
    out5 = S.analyse_edges(tail, opt, n=200, regime_cells={
        "G1": {"bulk": [104, 105], "tail": [100, 101, 102, 103]},
        "G3": {"bulk": [100, 101, 102, 103, 104, 105]}})
    r0 = A.H4_VARIANTS["v1"]["real"][0]
    assert out5["statistics"][r0] != out["statistics"][r0]
    reg = out5["H4_regimes"][r0]
    assert reg["typed_by"] == "G1" and set(reg["regimes"]) == {"bulk", "tail"}
    assert reg["regimes"]["tail"]["median"] == pytest.approx(1.5)
    assert out5["H4_regimes"][A.H4_TRANSFER["real"][0]]["typed_by"] == "G3"
    assert "G1" not in out5["H4_regimes"]                    # H2' networks are not typed
    # before the confirmatory run: coverage and plumbing only
    plumb = S.analyse_edges(nets, opt, n=200, decide=False)
    assert not plumb["decided"] and "coverage" in plumb
    assert not {"H2", "H4", "H4_transfer", "statistics"} & set(plumb)
    # a missing network: H4 is 'not computed' and listed
    some = {k: v for k, v in nets.items() if k != base}
    out6 = S.analyse_edges(some, opt, n=200)
    assert out6["H4"] == {"decision": "not computed", "missing": [base]}
    assert out6["missing"]["H4"] == [base]


def test_anchored_key_h4_is_descriptive():
    """The anchored-key version of H4 (design Section 7, descriptive): every H4 network's
    median G on the pooled anchored rows (off-policy rows counted, never excluded for it), and
    H4's test on those statistics; a network above 30% excluded makes it 'not testable'."""
    real = set(A.H4_VARIANTS["v1"]["real"] + A.H4_VARIANTS["v3"]["real"])
    nets = {}
    for i, n in enumerate(A.h4_networks()):
        level = 0.6 if n in real else 0.3
        nets[n] = [dict(r, off_policy=i > 0) for r in _net_rows(
            lambda g, r, level=level: level + 0.01 * r.standard_normal(), seed=i)]
    opt = _opt_rows(next(iter(nets.values())))
    out = S.anchored_h4(nets, opt, n=200)
    assert out["descriptive"] and out["test"]["decision"] == "confirm"
    assert out["test"]["main"]["n_relabellings"] == 4900
    first = A.h4_networks()[0]
    assert out["n_off_policy"][first] == 0 and out["n_off_policy"][A.h4_networks()[1]] == 48
    bad = {**nets, first: _net_rows(lambda g, r: 0.3, excluded=20)}
    assert S.anchored_h4(bad, opt, n=200)["test"]["decision"] == f"not testable (coverage: {first})"
    less = {k: v for k, v in nets.items() if k != first}
    assert S.anchored_h4(less, opt, n=200)["test"] == {"decision": "not computed",
                                                         "missing": [first]}


def test_edge_measures_exclusions_capped_and_pipeline_failures():
    """Capped stimuli leave H2' and H4 for a descriptive stratum; a missing optimum row, a
    branch-consistency failure of c_min and a loop exhausted at the key are pipeline
    exclusions of that key (design Section 15); Gate 4 is its own reason."""
    rows = _net_rows(lambda g, r: 0.3, capped_geo=5)
    opt = _opt_rows(rows)
    km = S.edge_key_measures(rows, opt)
    assert len(km) == 40 and all(np.isfinite(k.value) for k in km)
    capped = S.edge_key_measures(rows, opt, capped_only=True)
    assert len(capped) == 8 and {k.pair_id for k in capped} == {5}
    out = S.analyse_edges({"G1": rows}, opt, n=200)
    assert out["edge_descriptive"]["G1"]["capped_G"]["pooled"]["median"] == pytest.approx(0.3)
    rows2 = [dict(r) for r in rows]
    del opt[rows2[0]["key_id"]]
    rows2[1]["branch"] = {"ok": False, "ok_by": {"c_min": False, "v_apex": True, "s_b": True,
                                                 "steer": True}}
    rows2[2]["branch"] = {"ok": False, "ok_by": {"c_min": True, "v_apex": False, "s_b": True,
                                                 "steer": True}}
    rows2[3] = {k: rows2[3][k] for k in ("key_id", "pair_id", "approach", "direction", "phase",
                                         "n_phases", "R", "theta_deg", "cell", "w_is_cell",
                                         "cap_binds")} | {"pipeline": S.PIPELINE_LOOP}
    opt[rows2[4]["key_id"]] = dict(opt[rows2[4]["key_id"]], in_band=False)
    reasons = [k.reason for k in S.edge_key_measures(rows2, opt)[:5]]
    assert reasons == [MS.PIPELINE_NO_OPT, MS.PIPELINE_BRANCH, "", MS.PIPELINE_LOOP, "gate 4"]
    cov = S.analyse_edges({"G1": rows2}, opt, n=200, decide=False)["coverage"]["G1@final"]["G"]
    assert set(cov["pipeline_keys"]) == {rows2[i]["key_id"] for i in (0, 1, 3)}
    assert cov["excluded"] == {MS.PIPELINE_NO_OPT: 1, MS.PIPELINE_BRANCH: 1, MS.PIPELINE_LOOP: 1,
                               "gate 4": 1}
    # branch per outcome: the v_apex failure excludes v_apex, not c_min
    assert S.branch_failed(rows2[2], "v_apex") and not S.branch_failed(rows2[2], "c_min")
    assert not S.branch_failed(rows[0], "c_min")              # no det row: not checked
    # T = G + G_out
    t = S.edge_key_measures(rows, _opt_rows(rows), "T")
    assert all(k.value == pytest.approx(1.0) for k in t)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _arrays(d: Path, k: str) -> str:
    """A stand-in arrays file for key k under d; its sha256 (what a row's 'arrays_sha256'
    binds)."""
    from apexzero.fields import prereg

    return prereg.sha256_file(S.save_key_arrays(d, k, {"impulse": np.zeros((4, 3, 2))}))


def test_confirmatory_input_problems(tmp_path):
    """The run-once analysis refuses incomplete or unrecorded inputs (design Section 15)."""
    rec = {"key_sets": {"a2": [["nominal", "k1"], ["nominal", "k2"]],
                        "finals:G1": [["anchored", "a1"]]},
           "networks": {"G1": {"path": "p", "sha256": "s1", "timesteps": 7},
                        "runs/dr-x": {"path": "q", "sha256": "s2", "timesteps": 9}},
           "regimes": {"a2_regime_cells": {"G1": {"bulk": [1, 2]}}},
           "gate2_trained": [["nominal", "k2"]]}
    nets = ("G1", "runs/dr-x")
    arr = {"G1": tmp_path / "fields-a2" / "arrays" / "G1_7",
           "runs/dr-x": tmp_path / "fields-a2" / "arrays" / "dr-x_9"}

    def row(k, net, sha, d=None, **kw):
        out = {"key_id": k, "net": net, "ckpt_sha256": sha, "branch": {"ok": True}, **kw}
        if d is not None:
            out["arrays_sha256"] = _arrays(d, k)
        return out

    ok = [row("k1", "G1@7", "s1", arr["G1"]), row("k2", "G1@7", "s1", arr["G1"])]
    a2 = tmp_path / "fields-a2" / "G1.jsonl"
    _write_jsonl(a2, ok)
    _write_jsonl(tmp_path / "fields-a2" / "runs_dr-x.jsonl",
                 [row("k1", "dr-x@9", "s2", arr["runs/dr-x"]),
                  row("k2", "dr-x@9", "s2", arr["runs/dr-x"])])
    _write_jsonl(tmp_path / "fields-anchored" / "G1.jsonl",
                 [row("a1", "G1@7", "s1", tmp_path / "fields-anchored" / "arrays" / "G1_7")])
    _write_jsonl(tmp_path / "opt-fields" / "rows.jsonl", [{"key_id": k} for k in ("k1", "k2",
                                                                                  "a1")])
    g2 = tmp_path / "gate2-trained" / "gate2_trained.json"
    g2.parent.mkdir(parents=True)
    g2r = {n: {"pass": True, "net": S.expected_net(n, rec["networks"][n]),
               "ckpt_sha256": rec["networks"][n]["sha256"], "rows": [{"key_id": "k2"}]}
           for n in nets}
    g2.write_text(json.dumps({"keys": ["k2"], "networks": g2r}))

    def probs():
        return S.confirmatory_input_problems(rec, tmp_path, nets)

    assert probs() == []
    # an H2' network without its bulk A2 cells in the record (H1-typical's data)
    no_reg = {k: v for k, v in rec.items() if k != "regimes"}
    assert any("no bulk A2 cells" in p for p in S.confirmatory_input_problems(no_reg, tmp_path,
                                                                             nets))
    # a --limit'ed file (one key), a duplicate, a key outside the frozen list
    _write_jsonl(a2, ok[:1])
    assert any("frozen keys without a row" in p for p in probs())
    _write_jsonl(a2, ok[:1] * 2 + ok[1:])
    assert any("duplicate" in p for p in probs())
    _write_jsonl(a2, ok + [row("k9", "G1@7", "s1", arr["G1"])])
    assert probs() == [("G1: 0 frozen keys without a row (e.g. []), 1 rows of unlisted keys "
                        "(e.g. ['k9'])")]
    # the checkpoint's net id and sha256 are each checked on their own
    for bad in (row("k1", "G1@7", "other", arr["G1"]), row("k1", "G1@6", "s1", arr["G1"])):
        _write_jsonl(a2, [bad, ok[1]])
        assert probs() == ["G1: 1 rows not from the recorded checkpoint G1@7 (e.g. ['k1'])"]
    _write_jsonl(a2, [ok[0], dict(ok[1], branch=None)])
    assert any("without the wiring and branch checks" in p for p in probs())
    # the arrays the rows bind: a replaced or a missing npz is refused, a stale one ignored
    _write_jsonl(a2, ok)
    good = (arr["G1"] / "k1.npz").read_bytes()
    S.save_key_arrays(arr["G1"], "k1", {"impulse": np.ones((4, 3, 2))})
    assert probs() == ["G1: 1 arrays differ from their rows' sha256 (e.g. ['k1'])"]
    (arr["G1"] / "k1.npz").unlink()
    assert probs() == ["G1: 1 rows without their arrays (e.g. ['k1'])"]
    (arr["G1"] / "k1.npz").write_bytes(good)
    _arrays(arr["G1"], "stale")
    assert probs() == []
    # a duplicated optimum row (the analysis would keep only the last one), a missing one
    _write_jsonl(tmp_path / "opt-fields" / "rows.jsonl",
                 [{"key_id": k} for k in ("k1", "k1", "k2", "a1")])
    assert probs() == ["opt-fields: duplicate key rows"]
    _write_jsonl(tmp_path / "opt-fields" / "rows.jsonl", [{"key_id": "k1"}, {"key_id": "a1"}])
    assert any("without an optimum row" in p for p in probs())
    _write_jsonl(tmp_path / "opt-fields" / "rows.jsonl", [{"key_id": k} for k in ("k1", "k2",
                                                                                  "a1")])
    # gate2-trained against the record: its keys, each network's checkpoint, rows and verdict
    for bad in ({"keys": ["k1"], "networks": g2r},
                {"keys": ["k2"], "networks": {**g2r, "G1": {**g2r["G1"], "ckpt_sha256": "x"}}},
                {"keys": ["k2"], "networks": {**g2r, "G1": {**g2r["G1"], "pass": None}}},
                {"keys": ["k2"], "networks": {**g2r, "G1": {**g2r["G1"], "rows": []}}}):
        g2.write_text(json.dumps(bad))
        assert len(probs()) == 1 and probs()[0].startswith("gate2-trained:"), bad
    (tmp_path / "fields-a2" / "runs_dr-x.jsonl").unlink()
    g2.write_text(json.dumps({"keys": ["k2"], "networks": {"G1": g2r["G1"]}}))
    p = probs()
    assert any("runs/dr-x: no fields-a2 rows" in x for x in p)
    assert any("gate2-trained lacks" in x for x in p)


def test_gate2_trained_pass_rule(monkeypatch):
    """Design Section 15: gate2-trained applies Gate 2's rule (every row passes) on all of the
    record's keys; a key whose loop is exhausted cannot be checked, so it fails the network,
    and the stage goes on (no network, no loop: stubs)."""
    from apexzero.fields import gates as G
    from apexzero.fields import network as N

    verdict = {}

    def row(net, rd, st, steps, dirs, label=""):
        k = label.split("@", 1)[1]
        if verdict[k] is None:
            raise N.LoopExhausted("alive after 256 steps")
        return {"label": label, "pass": verdict[k], "share_counted": [1.0] * 4}

    monkeypatch.setattr(G, "gate2_row", row)
    monkeypatch.setattr(K, "load_key", lambda ref: None)
    monkeypatch.setattr(N, "a2_start", lambda rec, bank, cells: SimpleNamespace(stim_id=0))
    monkeypatch.setattr(N, "road", lambda bank, sid: None)
    monkeypatch.setattr(S, "gate2_trained_directions", lambda rd, st: {})
    net = SimpleNamespace(name="n", checkpoint="sha")
    refs = [SimpleNamespace(key_id="k1"), SimpleNamespace(key_id="k2")]

    def run(k1, k2):
        verdict.update(k1=k1, k2=k2)
        return S.gate2_trained_network(net, refs, bank=0, cells=0)

    r = run(True, True)
    assert r["pass"] and r["n_checked"] == 2 and r["n_exhausted"] == 0
    r = run(True, False)                                     # Gate 2's rule: every row ('all')
    assert not r["pass"] and r["n_checked"] == 2
    r = run(None, None)                                      # never checked: no vacuous pass
    assert not r["pass"] and r["n_checked"] == 0 and r["share_counted_mean"] == []
    r = run(None, True)                                      # one key exhausted: the network fails
    assert not r["pass"] and r["n_checked"] == 1 and r["n_exhausted"] == 1 and len(r["rows"]) == 2
    assert r["rows"][0]["exhausted"] and r["rows"][0]["pass"] is False
    assert r["share_counted_mean"] == [1.0] * 4              # over the checked keys


def test_the_s1_grid_keeps_the_near_field(tmp_path):
    """Design Sections 4 and 6 (H1-fast's gain, steer_0): bumps at s_1 + d, d from -20 m to the
    support end. The optimum's field has a near part A (30 m past s_1) and a corner part B
    (s_c0 + 100 m); the network's is 0.9 A + 0.5 B, so g is 0.7 over both parts and 0.5 over B
    alone. With the corner 250 m past s_1, the s_c0-aligned grid (d >= -150 m from s_c0)
    loses A and the s_1-anchored grid keeps it; with the corner 130 m past s_1 they agree."""
    rows = _net_rows(lambda g, r: 0.3, n_geo=2, n_phase=1)
    s = np.arange(N_SYN) * 2.0
    for r in rows:
        r.update(s1=20.0, support_end=dict.fromkeys(("c_min", "v_apex", "s_b", "steer"), 600.0))
    for s_c0, want_aligned in ((270.0, 0.5), (150.0, 0.7)):
        near = np.exp(-0.5 * ((s - 50.0) / 10.0) ** 2)
        far = np.exp(-0.5 * ((s - (s_c0 + 100.0)) / 10.0) ** 2)
        imp_o, imp_n = np.zeros((4, 3, N_SYN)), np.zeros((4, 3, N_SYN))
        imp_o[:, 0], imp_n[:, 0] = near + far, 0.9 * near + 0.5 * far
        opt = _synthetic_fields(rows, None, None, s_c0=s_c0, s_c1=s_c0 + 60.0)
        for r in rows:
            opt[r["key_id"]]["s1"] = 20.0
            S.save_key_arrays(tmp_path / "opt", r["key_id"], {"impulse": imp_o})
            S.save_key_arrays(tmp_path / "net", r["key_id"], {"impulse": imp_n,
                                                               "blind_impulse": 0.5 * imp_n})
        for o in ("s_b", "steer"):
            s1g = S.field_comparisons(rows, opt, tmp_path / "net", tmp_path / "opt", o,
                                      grid="s1")
            assert all(c["g"] == pytest.approx(0.7, abs=1e-6) for c in s1g), (s_c0, o)
            al = S.field_comparisons(rows, opt, tmp_path / "net", tmp_path / "opt", o)
            assert all(c["g"] == pytest.approx(want_aligned, abs=1e-6) for c in al), (s_c0, o)
    with pytest.raises(ValueError, match="grid"):
        S.field_comparisons(rows, opt, tmp_path / "net", tmp_path / "opt", "s_b", grid="x")
    r0, o0 = rows[0], opt[rows[0]["key_id"]]
    s1, d = S.s1_grid(r0, o0, "s_b")
    assert s1 == 20.0 and d[0] == -20.0 and d[-1] == pytest.approx(580.0)
    s1, d = S.s1_grid({k: v for k, v in r0.items() if k != "support_end"}, o0, "s_b")
    assert d[-1] == pytest.approx(o0["s_c0"] + MS.NULL_D[1] - 20.0)


def test_fixed_network_lists():
    assert len(A.A2_NETWORKS) == 35 and len(set(A.A2_NETWORKS)) == 35
    assert A.A2_NETWORKS[:9] == A.H2_NETWORKS
    assert len(A.DOSE_NETWORKS) == 27 and set(A.h4_networks()) <= set(A.DOSE_NETWORKS)
    assert set(A.H2_NETWORKS) <= set(A.DOSE_NETWORKS)
    assert not set(A.H4_TRANSFER["real"]) & set(A.DOSE_NETWORKS)
    assert A.h4_regime_agent("runs/dr-realw-v1-s0") == "G1"
    assert A.h4_regime_agent("runs/dr-pilot-v3-tilt50a2-s4") == "G3"
    assert A.h4_regime_agent("runs/dr-realw-loo-f2-s3") == "G3"
    assert A.h4_regime_agent("G1") is None
    assert S.registry_of_run("runs/dr-pilot-200M") == "G1"
    assert S.registry_of_run("runs/dr-pilot-200M-s3") is None and S.registry_of_run("G1") is None
    assert S.network_agent_id("runs/dr-realw-v1-s0") == "dr-realw-v1-s0"
    assert S.expected_net("runs/dr-realw-v1-s0", {"timesteps": 5}) == "dr-realw-v1-s0@5"


@needs_keys
def test_gate2_trained_sample_and_directions():
    a, b = S.gate2_trained_sample(), S.gate2_trained_sample()
    assert [r.key_id for r in a] == [r.key_id for r in b] and len(a) == S.GATE2_TRAINED_N
    prim = {r.key_id for r in K.a2_primary_refs()}
    assert all(r.key_id in prim and not K.strata(r).cap_binds for r in a)
    assert [r.key_id for r in a] == sorted(r.key_id for r in a)
    assert MS.FIELDS_STREAMS["gate2_trained"] not in {v for k, v in MS.FIELDS_STREAMS.items()
                                                      if k != "gate2_trained"}


def test_capped_stimuli_leave_every_decided_set():
    """Design Section 15: capped stimuli leave H2-speed, H1-typical and H1-fast as well as H2'
    and H4 (a descriptive stratum)."""
    rows = _net_rows(lambda g, r: 0.3, capped_geo=2)
    assert {r["pair_id"] for r in S._non_capped(rows)} == {0, 1, 3, 4, 5}
    assert {r["pair_id"] for r in S._capped(rows)} == {2}
    cmp = [{"row": r, "g": 1.0, "location": 0.0, "delta_rho": 0.2, "flags": {}} for r in rows]
    a, _ = S.h1_typical_rows(cmp, [], "s_b")
    assert len(a["gain"]) == 40 and {k.pair_id for k in a["gain"]} == {0, 1, 3, 4, 5}


N_SYN = 360                                   # samples (720 m) of the synthetic fields


def _synthetic_fields(rows, net_dir: Path | None, opt_dir: Path | None, gain: float = 0.9,
                      s_c0: float = 200.0, s_c1: float = 260.0):
    """Give synthetic fields-a2 / fields-anchored rows their outcome values, inside-shift speed
    response and arrays (a curvature bump and a flat width response; the network's fields are
    ``gain`` x the optimum's, its blind-pedal control's half of them; a None directory writes
    no arrays), and return the matching optimum rows (s_b,net - s_b* = -10 m,
    v_apex,net - v_apex* = -1 m/s)."""
    s = np.arange(N_SYN) * 2.0
    imp = np.zeros((4, 3, N_SYN))
    imp[:, 0] = np.exp(-0.5 * ((s - (s_c0 + 40.0)) / 20.0) ** 2)
    imp[:, 1:] = 0.01
    opt = {}
    for r in rows:
        r.update(net=r.get("net", "syn@1"), ckpt_sha256=r.get("ckpt_sha256", ""), dv_in=0.005,
                 values={"c_min": 0.5, "v_apex": 20.0, "s_b": 250.0, "steer": 0.0})
        r.setdefault("s1", s_c0 - 150.0)
        if r["branch"] is None:
            r["branch"] = {"ok": True, "ok_by": dict.fromkeys(("c_min", "v_apex", "s_b",
                                                                "steer"), True)}
        opt[r["key_id"]] = {**_opt_rows([r])[r["key_id"]], "s_c0": s_c0, "s_c1": s_c1,
                            "s1": r["s1"],
                            "valid": dict.fromkeys(("c_min", "v_apex", "s_b", "steer"), True),
                            "values": {"c_min": 0.03, "v_apex": 21.0, "s_b": 260.0,
                                       "steer": 0.0}}
        if opt_dir is not None:
            S.save_key_arrays(opt_dir, r["key_id"], {"impulse": imp})
        if net_dir is not None:
            f = S.save_key_arrays(net_dir, r["key_id"], {"impulse": gain * imp,
                                                         "blind_impulse": 0.5 * imp})
            r["arrays_sha256"] = prereg_sha(f)
    return opt


def prereg_sha(path: Path) -> str:
    from apexzero.fields import prereg

    return prereg.sha256_file(path)


def test_h1_fast_and_h2_speed_leave_out_capped_stimuli(tmp_path):
    """Design Section 15: capped stimuli leave H1-fast and H2-speed (as H2', H4 and
    H1-typical) for a descriptive stratum reported apart; every non-capped key counts in the
    coverage."""
    rows = _net_rows(lambda g, r: 0.3, capped_geo=2)
    opt = _synthetic_fields(rows, tmp_path / "net", tmp_path / "opt")
    fast = S.h1_fast_rows(rows, opt, tmp_path / "net", tmp_path / "opt", "G1", n=200)
    assert fast["s_b"]["pooled"]["n_pairs"] == 5 and fast["v_apex"]["pooled"]["n_pairs"] == 5
    assert fast["s_b"]["pooled"]["median"] == pytest.approx(-10.0)
    assert fast["v_apex"]["pooled"]["median"] == pytest.approx(-1.0)
    assert fast["gain"]["pooled"]["median"] == pytest.approx(0.9)
    assert fast["s_b"]["step1_direction"] == "consistent"
    assert fast["coverage"]["s_b"]["n"] == 40 and not fast["coverage"]["s_b"]["excluded"]
    assert {p for p in ("s_b", "v_apex", "gain")} == set(fast["capped"])
    assert fast["capped"]["s_b"]["pooled"]["n_pairs"] == 1
    assert S.h1_fast_rows(rows, opt, tmp_path / "net", tmp_path / "opt", "G1", n=200,
                          decide=False).keys() == {"coverage"}
    af = S.analyse_fields(rows, opt, tmp_path / "net", tmp_path / "opt", n=200)
    assert af["H2_speed"]["pooled"]["n_pairs"] == 5
    assert af["H2_speed_capped"]["pooled"]["n_pairs"] == 1
    assert af["coverage"]["H2_speed/v_apex"]["n"] == 40
    g = af["H1_typical"]["s_b"]["criteria"]["gain"]
    assert g["pooled"]["n_pairs"] == 5 and g["pooled"]["median"] == pytest.approx(0.9)
    # paired against the blind-pedal control: -|0.9 - 1| - (-|0.5 - 1|)
    assert g["vs_control"]["pooled"]["median"] == pytest.approx(0.4)
    assert S.analyse_fields(rows, opt, tmp_path / "net", tmp_path / "opt", n=200,
                            decide=False).keys() == {"coverage"}
    # descriptive: the other bump widths (bulk cells) and steer_0 (every non-capped key)
    de = af["descriptive"]
    assert set(de) == {"sigma_b=4/s_b", "sigma_b=4/v_apex", "sigma_b=16/s_b",
                       "sigma_b=16/v_apex", "steer_0"}
    for v in de.values():
        assert v["gain"]["pooled"]["median"] == pytest.approx(0.9)
        assert v["gain"]["pooled"]["n_pairs"] == 5
    # the anchored-key version of H2' (descriptive): G, T = G + G_out and B, capped left out
    ae = S.anchored_edges(rows, opt, n=200)
    assert ae["G"]["pooled"]["median"] == pytest.approx(0.3) and ae["coverage"]["n"] == 40
    assert ae["T"]["label"] == "corridor-anchored" and ae["B"]["pooled"]["median"] == 0.5
    # near-ties are counted (reported, kept)
    tied = [dict(r, near_tie={"c_min": i < 3, "v_apex": False, "s_b": i == 0, "steer": False})
            for i, r in enumerate(rows)]
    assert S.near_tie_counts(tied) == {"c_min": 3, "v_apex": 0, "s_b": 1, "steer": 0}


TAG_ID = {"tag": "receptive-fields-prereg-v1", "object": "a" * 40, "type": "tag",
          "commit": "b" * 40}


def test_confirmatory_analysis_runs_once_on_complete_inputs(tmp_path, monkeypatch):
    """Design Section 15, through scripts/fields.py's analyse stage on synthetic, stamped
    inputs (the freeze check replaced by the record): before the confirmatory run, coverage
    and plumbing only (a touched or hand-made decisions.json unlocks nothing); --confirmatory
    refuses an incomplete or partial input and writes nothing, then writes its decisions once
    to the fixed path (gate2-trained failures reaching H2' and the H4 family, the bulk cells
    reaching H1-typical, the regime cells H4's breakdown) and refuses a second run; a dry run
    after it decides, naming the decisions' sha256."""
    from apexzero.fields import prereg

    fs = _script()
    root = tmp_path / "out"
    monkeypatch.setattr(fs, "OUT", root)
    base = _net_rows(lambda g, r: 0.3, n_phase=2, seed=0)
    pins = {n: {"path": f"runs/{i}", "sha256": f"sha{i}", "timesteps": 7}
            for i, n in enumerate(A.A2_NETWORKS)}
    anch = [dict(r, key_id="a" + r["key_id"], approach="A1") for r in _net_rows(
        lambda g, r: 0.6, n_geo=4, n_phase=2, seed=99)]
    g2keys = [r["key_id"] for r in base[:2]]
    rec = {"key_sets": {"a2": [["nominal", r["key_id"]] for r in base],
                        "finals:G1": [["anchored", r["key_id"]] for r in anch]},
           "networks": pins, "gate2_trained": [["nominal", k] for k in g2keys],
           "environment": fs.environment(),
           "regimes": {"a2_regime_cells": {a: {"bulk": [100, 101, 102]}
                                           for a in A.H2_NETWORKS}}}

    def post_tag(args, what, x64=True):
        fs._prov().tag = TAG_ID
        return rec

    monkeypatch.setattr(fs, "_post_tag", post_tag)

    def run(*argv):
        full = ["analyse", "--log-dir", str(tmp_path), *argv]
        monkeypatch.setattr(sys, "argv", ["scripts/fields.py", *full])     # what the stamp records
        args = fs.build_parser().parse_args(full)
        fs._PROV[:] = [fs.Provenance(args, "analyse")]
        fs.stage_analyse(args)
        return json.loads((root / "analyse" / "decisions_dry.json").read_text()) \
            if "--confirmatory" not in argv else None

    fs._PROV[:] = [fs.Provenance(fs.build_parser().parse_args(["fields-a2"]), "fields-a2")]
    fs._PROV[0].tag = TAG_ID
    opt = {}
    for i, name in enumerate(A.A2_NETWORKS):
        rows = _net_rows(lambda g, r: 0.3 + 0.01 * r.standard_normal(), n_phase=2, seed=i)
        net = S.expected_net(name, pins[name])
        for r in rows:
            r.update(net=net, ckpt_sha256=pins[name]["sha256"], branch=None)
        opt = _synthetic_fields(rows, S.arrays_dir(root / "fields-a2" / "arrays", net),
                                None if i else root / "opt-fields" / "arrays")
        fs._write_rows(root / "fields-a2" / f"{name.replace('/', '_')}.jsonl", rows)
    for r in anch:
        r.update(net=S.expected_net("G1", pins["G1"]), ckpt_sha256=pins["G1"]["sha256"])
    opt.update(_synthetic_fields(anch, root / "fields-anchored" / "arrays" / "G1_7",
                                 root / "opt-fields" / "arrays"))
    fs._write_rows(root / "fields-anchored" / "G1.jsonl", anch)
    S.write_rows(root / "opt-fields" / "rows.jsonl", list(opt.values()))
    # gate2-trained fails G2 (an H2' network) and a leave-out run (H4-transfer only)
    fails = {"G2", "runs/dr-realw-loo-f1-s0"}
    fs._write(root / "gate2-trained" / "gate2_trained.json",
              {"keys": g2keys, "networks": {n: {"pass": n not in fails, "n_keys": 2,
                                                "n_checked": 2, "n_exhausted": 0,
                                                "net": S.expected_net(n, pins[n]),
                                                "ckpt_sha256": pins[n]["sha256"],
                                                "rows": [{"key_id": k} for k in g2keys]}
                                            for n in A.A2_NETWORKS}})
    decisions = root / "analyse" / "decisions.json"
    dry_ = run()
    assert not dry_["decided"] and not {"H2", "H4", "H4_transfer", "statistics"} & set(dry_)
    assert dry_["coverage"]["G1@final"]["G"]["n"] == 24 and not decisions.exists()
    assert dry_["H1_fast"]["G1"].keys() == {"coverage"} and not dry_["anchored_edges"]
    assert set(dry_["fields"]) == set(A.H2_NETWORKS)
    assert all(v.keys() == {"coverage"} for v in dry_["fields"].values())     # no early decision
    assert dry_["unlocked_by"] is None and dry_["gate2_trained"]["G2"]["pass"] is False
    # a touched or hand-made decisions.json unlocks nothing
    for junk in ("", json.dumps({"confirmatory": True, "decided": True, "H2": {}, "H4": {},
                                 "H4_transfer": {}, "inputs": {"x": "y"}})):
        decisions.parent.mkdir(parents=True, exist_ok=True)
        decisions.write_text(junk)
        d = run()
        assert not d["decided"] and not {"H2", "H4", "H4_transfer", "statistics"} & set(d)
        decisions.unlink()
    # an incomplete input (one key's row lost) and a partial run's stamp: refused, nothing written
    f = root / "fields-a2" / "runs_dr-realw-v1-s0.jsonl"
    full = f.read_text()
    f.write_text("".join(full.splitlines(keepends=True)[1:]))
    with pytest.raises(SystemExit, match="complete inputs") as e:
        run("--confirmatory")
    assert "frozen keys without a row" in str(e.value)
    assert "changed after it was stamped" in str(e.value)
    assert not decisions.exists()
    f.write_text(full)
    sc = prereg.stamp_path(f)
    st = json.loads(sc.read_text())
    sc.write_text(json.dumps({**st, "partial": ["--limit"]}))
    with pytest.raises(SystemExit, match="a partial run"):
        run("--confirmatory")
    sc.write_text(json.dumps(st))
    # a hand-flipped verdict in the stamped gate2-trained output is refused
    g2f = root / "gate2-trained" / "gate2_trained.json"
    g2_txt = g2f.read_text()
    g2f.write_text(g2_txt.replace('"pass": false', '"pass": true', 1))
    with pytest.raises(SystemExit, match="gate2_trained.json: changed after it was stamped"):
        run("--confirmatory")
    g2f.write_text(g2_txt)
    # an input written under another tag is refused
    st_tag = json.loads(sc.read_text())
    sc.write_text(json.dumps({**st_tag, "tag": {**TAG_ID, "object": "c" * 40}}))
    with pytest.raises(SystemExit, match="written under another tag"):
        run("--confirmatory")
    sc.write_text(json.dumps(st_tag))
    run("--confirmatory")
    res = json.loads(decisions.read_text())
    assert res["confirmatory"] and res["decided"]
    assert set(res["H2"]) == set(A.H2_NETWORKS)
    assert res["H2"]["G2"]["decision"] == A.DESCRIPTIVE_GATE2     # gate2-trained reaches H2'
    assert res["H4"]["main"]["n_relabellings"] == 4900
    assert res["H4"]["sensitivity"]["n_relabellings"] == 15876
    assert res["H4_transfer"]["decision"] == \
        "not testable (gate2-trained: runs/dr-realw-loo-f1-s0)"   # ...and the H4 family
    assert len([k for k in res["inputs"] if k.endswith(".jsonl") and "fields-" in k]) == \
        len(A.A2_NETWORKS) + 1
    assert {"opt-fields/rows.jsonl", "opt-fields/arrays",
            "gate2-trained/gate2_trained.json"} <= set(res["inputs"])
    assert res["provenance"]["stage"] == "analyse" and not res["provenance"]["partial"]
    assert res["deviations"] == [] and res["tag_pin"] is None
    assert set(res["fields"]) == set(A.H2_NETWORKS)
    # the bulk cells reach H1-typical (pairs 0-2 of 6), the regime cells H4's breakdown
    g = res["fields"]["G1"]["H1_typical"]["s_b"]["criteria"]["gain"]
    assert g["pooled"]["n_pairs"] == 3
    reg = res["H4_regimes"]["runs/dr-realw-v1-s0"]
    assert reg["typed_by"] == "G1" and reg["regimes"]["bulk"]["n_keys"] == 12
    assert "G1" not in res["H4_regimes"]
    assert res["H1_fast"]["G1"]["s_b"]["pooled"]["median"] == pytest.approx(-10.0)
    assert res["anchored_edges"]["G1"]["G"]["pooled"]["median"] == pytest.approx(0.6)
    # every reported measure carries its left-right asymmetry index
    assert res["edge_descriptive"]["G1"]["T"]["asymmetry"]["n_pairs"] == 6
    assert res["edge_descriptive"]["G1"]["asymmetry_G"]["n_pairs"] == 6
    assert "asymmetry" in res["H1_fast"]["G1"]["s_b"]
    with pytest.raises(SystemExit, match="runs once"):
        run("--confirmatory")
    # after it, a dry run decides and names the decisions it was unlocked by
    d = run()
    assert d["decided"] and d["unlocked_by"] == prereg.sha256_file(decisions)
    assert S.confirmatory_done(decisions, TAG_ID) == d["unlocked_by"]
    for bad in ({"partial": ["--limit"]}, {"tag": {**TAG_ID, "commit": "c" * 40}},
                {"argv": ["fields.py", "analyse"]}, {"stage": "fields-a2"}):
        cp = tmp_path / "copy.json"
        cp.write_text(json.dumps({**res, "provenance": {**res["provenance"], **bad}}))
        assert S.confirmatory_done(cp, TAG_ID) is None, bad
    assert S.confirmatory_done(decisions, {**TAG_ID, "object": "c" * 40}) is None


def test_input_stamp_problems_refusals(tmp_path):
    """Each refusal of the confirmatory analysis's input check (scripts/fields.py
    ``input_stamp_problems``), alone: no stamp, other code, an undeclared deviation (and a
    declared one), another tag, other package versions, an embedded stamp without a content
    hash."""
    from apexzero.fields import prereg

    fs = _script()
    code = "c" * 64
    p = tmp_path / "x.jsonl"
    p.write_text('{"key_id": "k"}\n')
    env = fs.environment()
    clean = {"code_hash": code, "files": {p.name: prereg.sha256_file(p)}, "partial": [],
             "deviation": None, "tag": TAG_ID, "environment": env}
    sc = prereg.stamp_path(p)

    def probs(stamp, deviation=None, tag=TAG_ID, env_=None):
        if stamp is None:
            sc.unlink(missing_ok=True)
        else:
            sc.write_text(json.dumps(stamp))
        return fs.input_stamp_problems([p], code, deviation, tag, env_)

    assert probs(clean) == []
    assert probs(None) == ["x.jsonl: no provenance stamp"]
    (only,) = probs({**clean, "code_hash": "x" * 64})
    assert only.startswith("x.jsonl: written by other code (xxxxxxxxxxxx)")
    assert probs({**clean, "deviation": {"reason": "r"}}) == [
        "x.jsonl: written under a deviation this run does not declare"]
    assert probs({**clean, "deviation": {"reason": "r"}}, deviation={"reason": "r"}) == []
    assert probs(clean, tag={**TAG_ID, "commit": "d" * 40}) == [
        "x.jsonl: written under another tag (aaaaaaaaaaaa)"]
    other = {**env, "packages": {**env["packages"], "jax": "0.0.1"}}
    (only,) = probs({**clean, "environment": other}, env_=env)
    assert only.startswith("x.jsonl: computed with other package versions")
    assert probs({**clean, "environment": other}, deviation={"r": 1}, env_=env) == []
    assert probs({**clean, "files": {}}) == ["x.jsonl: its stamp binds no content hash"]
    # an embedded stamp binds the object's content without its stamp
    j = tmp_path / "g.json"
    body = {"networks": {"G1": {"pass": False}}}
    j.write_text(json.dumps({**body, "provenance": {**clean, "files": {},
                                                    "content_sha256": fs.content_sha256(body)}}))
    assert fs.input_stamp_problems([j], code, None, TAG_ID) == []
    j.write_text(j.read_text().replace('"pass": false', '"pass": true'))
    assert fs.input_stamp_problems([j], code, None, TAG_ID) == [
        "g.json: changed after it was stamped"]


def test_a_trained_network_needs_the_recorded_branch_tolerance():
    """Design Section 15: after the tag the branch tolerance is the record's calibrated one,
    never a code constant; ``network_key`` refuses a trained network without it (before any
    key is read or any run made)."""
    from apexzero.fields import network as N

    net = N.NetPolicy(None, None, None, "G1@1", trained=True)
    with pytest.raises(ValueError, match="record's calibrated one"):
        S.network_key(net, None, None, None)
    # with a tolerance, still refused before the tag: before any key is read (ref None)
    assert not N._TRAINED_OK
    with pytest.raises(PermissionError, match="waits for the tag"):
        S.network_key(net, None, None, None, branch_tol={"c_min": 1e-3})
