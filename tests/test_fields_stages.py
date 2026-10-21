"""Step 2's batch-stage bodies (fields.stages) and decision rules (fields.analysis), built and
tested before the tag (design Section 9 freezes the code that computes and decides).

- The decision rules on synthetic data with known answers: H4's exact within-variant
  permutation test against brute-force enumeration (15,876 and 4,900 relabellings), the
  equivalence and sign tests on known intervals, R18's 2-of-3 truth table, H2-speed's bands,
  H1-typical's criteria with the paired blind-pedal control, exclusions in their order,
  lambda* and lambda*_lin, the regime cell sets against step 1's ``regime_of``.
- The optimum's per-key fields stage on stored keys (A2 and anchored; optimum side only), the
  phase ceiling of one cell, the dose subsample, H3's twins and the H3 key driver's plumbing
  (step 1's recipes faked: no solve here), into step 2's own key cache.
- The network stage bodies end to end in a float64 process (``python -m apexzero.fields.stages
  dry-run``) on policies that are not trained networks: the planted linear driver and a
  random-init actor (G1's architecture and obs_rms, read only) at an A2 key, the dose
  runs, H3's lambda grid on a scripted keeper, and the analysis assembly.
This process keeps x64 off (the optimum's float64 JAX runs inside ``jax.enable_x64`` blocks)."""

import itertools
import json
import os
import subprocess
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


def test_h4_permutation_test_against_brute_force():
    rng = np.random.default_rng(4)
    stats = {}
    for d in A.H4_VARIANTS.values():
        for r in d["real"]:
            stats[r] = 0.6 + 0.05 * rng.standard_normal()
        for r in d["baseline"]:
            stats[r] = 0.3 + 0.05 * rng.standard_normal()
    res = A.h4_test(stats)
    main, sens = res["main"], res["sensitivity"]
    assert main["n_relabellings"] == 126 ** 2 == 15876 and sens["n_relabellings"] == 70 ** 2
    groups = {v: ([stats[r] for r in d["real"]], [stats[r] for r in d["baseline"]])
              for v, d in A.H4_VARIANTS.items()}
    dist = _brute(groups)
    t = sum(np.mean(r) - np.mean(b) for r, b in groups.values())
    assert main["T"] == pytest.approx(t)
    assert main["p"] == pytest.approx(np.mean(dist >= t - 1e-12))
    assert main["p_opposite"] == pytest.approx(np.mean(dist <= t + 1e-12))
    # every real network above every baseline: the observed labelling is the most extreme
    assert main["p"] == pytest.approx(1 / 15876) and main["decision"] == "confirm"
    assert sens["decision"] == "confirm" and sens["p"] == pytest.approx(1 / 4900)
    assert all(s["n_relabellings"] == 126 for s in main["strata"].values())
    # the reverse: refuted; equal statistics: inconclusive
    flip = {r: -v for r, v in stats.items()}
    assert A.h4_test(flip)["main"]["decision"] == "refute"
    assert A.h4_test(dict.fromkeys(stats, 0.5))["main"]["decision"] == "inconclusive"
    with pytest.raises(ValueError, match="NaN"):
        A.h4_test({**stats, A.H4_VARIANTS["v1"]["real"][0]: float("nan")})
    assert [A.is_baseline_seed0(r) for r in A.H4_VARIANTS["v1"]["baseline"]] == \
        [True, False, False, False, False]
    tr = A.h4_transfer(stats | {r: 0.7 for r in A.H4_TRANSFER["real"]})
    assert tr["n_relabellings"] == 1287 and tr["decision"] == "confirm"


def test_bands_equivalence_and_signs():
    assert A.equivalence(0.85, 1.2) == "equivalent"
    assert A.equivalence(0.5, 0.75) == "not equivalent" and A.equivalence(1.3, 2.0) == "not equivalent"
    assert A.equivalence(0.7, 1.1) == "inconclusive" and A.equivalence(np.nan, 1.0) == "inconclusive"
    assert A.sign_decision(-3.0, -0.5, -1) == "confirm"
    assert A.sign_decision(0.5, 3.0, -1) == "refute"
    assert A.sign_decision(-1.0, 1.0, -1) == "inconclusive"
    assert A.sign_decision(0.5, 1.0, 1) == "confirm" and A.sign_decision(-1, -0.5, 1) == "refute"
    assert A.sign_decision(-1.0, 1.0, 0) == "consistent"
    assert A.sign_decision(0.2, 1.0, 0) == "inconsistent"
    assert A.h2_speed(0.1, 0.4) == "under" and A.h2_speed(0.6, 1.8) == "lawful"
    assert A.h2_speed(2.5, 3.0) == "over" and A.h2_speed(0.3, 1.0) == "inconclusive"


def test_r18_variant_rule_truth_table():
    v = "v1"
    s0, s1, s2 = A.registry.variant_seeds(v)
    assert A.variant_rule({s0: "confirm", s1: "confirm", s2: "lawful"}, v) == "confirm"
    assert A.variant_rule({s0: "lawful", s1: "confirm", s2: "confirm"}, v) == "confirm"
    assert A.variant_rule({s0: "confirm", s1: "lawful", s2: "inconclusive"}, v) == "seed-specific"
    assert A.variant_rule({s0: "confirm", s1: "confirm"}, v) == "confirm"
    assert A.variant_rule({s0: "confirm"}, v) == "seed-specific"
    assert A.variant_rule({}, v) == "seed-specific"


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


def test_h1_typical_and_the_paired_blind_pedal_control():
    gain = _rows(lambda g, r: 1.0 + 0.02 * r.standard_normal())
    loc = _rows(lambda g, r: 2.0 * r.standard_normal())
    shape = _rows(lambda g, r: 0.3 + 0.02 * r.standard_normal())
    ctrl = {"gain": _rows(lambda g, r: 0.4 + 0.02 * r.standard_normal()),
            "location": _rows(lambda g, r: 30.0 + r.standard_normal()),
            "shape": _rows(lambda g, r: 0.02 * r.standard_normal())}
    res = A.h1_typical(gain, loc, shape, ctrl, n=1000)
    assert res["pass"], res
    # a control as good as the network: the paired criterion fails although the absolute holds
    same = A.h1_typical(gain, loc, shape, {"gain": gain, "location": loc, "shape": shape}, n=1000)
    assert not same["pass"] and same["criteria"]["gain"]["pass"]
    assert not same["criteria"]["gain"]["vs_control"]["pass"]
    # a field shape no more specific than far geometries fails the shape criterion
    flat = _rows(lambda g, r: 0.02 * r.standard_normal())
    assert not A.h1_typical(gain, loc, flat, ctrl, n=1000)["criteria"]["shape"]["pass"]
    # H1-fast: G1 predicted earlier braking (s_b difference < 0)
    early = _rows(lambda g, r: -8.0 + r.standard_normal())
    fast = A.h1_fast(early, early, gain, "G1s1", n=1000)
    assert fast["predicted"] == -1 and fast["s_b"]["decision"] == "confirm"
    assert fast["gain"]["decision"] == "equivalent"
    assert A.h1_fast(early, early, gain, "G2", n=500)["s_b"]["decision"] == "inconsistent"


def test_key_measures_and_their_exclusions():
    meta = {"key_id": "k", "pair_id": 1, "approach": "A2", "direction": 1, "phase": 0,
            "n_phases": 4, "R": 10.0, "theta_deg": 90.0}
    assert A.key_measure(meta, 0.4, {}).value == 0.4
    km = A.key_measure(meta, 0.4, {"invalid": True, "kappa_clipped": True})
    assert np.isnan(km.value) and km.reason == "invalid outcome"
    assert A.key_measure(meta, 0.4, {"kappa_clipped": True, "off_plateau": True}).reason == \
        "kappa clipping"
    with pytest.raises(ValueError, match="pipeline failure"):
        A.key_measure(meta, 0.4, {"invalid": True, "why_invalid": MS.LOOP_EXHAUSTED})
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


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    if not _have_all():
        pytest.skip("step 1's design, keys, G1's final or G2's rollouts are not here")
    out = tmp_path_factory.mktemp("stages") / "dry.json"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(SRC), *filter(None, [os.environ.get("PYTHONPATH")])]))
    p = subprocess.run([sys.executable, "-m", "apexzero.fields.stages", "dry-run", "--out",
                        str(out)], env=env, capture_output=True, text=True, timeout=600,
                       check=False)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return json.loads(out.read_text())


def test_network_key_stage_on_planted_and_random_policies(dry):
    lin, rnd = dry["linear"], dry["random"]
    assert all(lin["valid"].values()) and lin["on_plateau"] and lin["e_star"] >= 0.99
    assert 0.2 < lin["G"] < 0.8 and lin["B"] > 0 and lin["paths_sum_rel_err"] <= 1e-9
    assert lin["blind"]["reproduces"] and not any(lin["kappa_clip"].values())
    assert lin["gate3"] is None                         # no float32 rollouts for this policy
    shapes = dry["arrays"]["linear"]
    assert shapes["impulse"][:2] == [4, 3] and shapes["paths_kappa"][:2] == [4, 4]
    # the random actor crashes: its plan outcomes are exclusions with NaN rows (G NaN)
    assert rnd["valid"] == {"c_min": False, "v_apex": False, "s_b": False, "steer": True}
    assert rnd["G"] is None and dry["arrays"]["random_impulse_nan_rows"] == [True, True, True,
                                                                               False]


def test_dose_and_h3_stage_bodies(dry):
    d = dry["dose"]
    assert [r["amplitude"] for r in d["rows"]] == [0.1, 0.3, 1.0, -0.1, -0.3, -1.0, -2.0]
    c = {r["amplitude"]: r["values"][0] for r in d["rows"]}
    assert c[-0.1] < c[0.1] and c[-2.0] < c[-1.0]       # narrowing the inside cuts c_min
    assert d["crash_narrowing_predicted"] == pytest.approx(d["B"] / (1 - d["G"]))
    h = dry["h3"]
    assert [r["lambda"] for r in h["runs"]] == [0.0, 0.5, 1.0]
    assert h["lambda_star"] == min([r["lambda"] for r in h["runs"] if r["crashed"]],
                                   default=float("inf"))
    au = dry["h3_loop_audit"]
    assert au["ok"] and au["n"] >= 24 and au["max"] > 300 and au["max"] <= S.H3_STEPS - 40


def test_analysis_assembly(dry):
    a = dry["analysis"]
    cov = a["coverage"]
    assert cov["random@final"]["G"]["excluded"] == {"invalid outcome": 1}
    assert cov["linear_driver@final"]["G"]["excluded"] == {}
    assert set(a["H2_variants"]) == {"v1", "v2", "v3"}
    # the field comparisons against the key's real optimum fields (one key: no far geometry,
    # so delta_rho is undefined and every interval is too)
    fa = dry["fields_analysis"]
    assert set(fa["H1_typical"]) == {"s_b", "v_apex"}
    assert set(fa["H1_typical"]["s_b"]["criteria"]) == {"gain", "location", "shape"}
    assert not fa["H1_typical"]["s_b"]["pass"]
    assert "decision" in fa["H2_speed"]
    c = dry["comparison"]["v_apex"]
    assert np.isfinite(c["g"]) and -1.0 <= c["rho"] <= 1.0 and np.isfinite(c["location"])
    # the optimum does not brake at this gentle corner: its s_b is invalid, and so is the
    # comparison (NaN, an exclusion), not a zero
    assert dry["comparison"]["s_b"]["g"] is None


def test_confirmatory_guard(tmp_path):
    path = S.confirmatory_guard(tmp_path)
    path.write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        S.confirmatory_guard(tmp_path)
