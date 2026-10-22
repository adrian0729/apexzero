"""Step 2's network stage bodies end to end (fields.stages), in a float64 process (``python -m
apexzero.fields.stages dry-run``), on policies that are not trained networks: the planted linear
driver and a random-init actor (G1's architecture and obs_rms, read only) at an A2 key, a loop
exhausted at one key, the dose runs, gate2-trained's body, Qp's reader fields, H3's lambda grid
on a scripted keeper, and the analysis assembly; and the wiring of a network without step 1
rollouts: the hybrid control's float32 harness run, made here (x64 off) as 'harness-f32' makes
one, against its float64 loop (a branch failure is a pipeline exclusion). Apart from
tests/test_fields_stages.py so that each file stays well under two minutes on its own.
This process keeps x64 off."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from apexzero.fields import keys as K
from apexzero.fields import measures as MS
from apexzero.fields import stages as S
from apexzero.interp import agents, io

SRC = Path(S.__file__).resolve().parents[2]
HAVE_KEYS = (io.KEYS_DIR / "anchored").exists() and (io.STIMULI_DIR / "cells.json").exists()


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


def _script():
    """scripts/fields.py as a module (its stage helpers; nothing runs on import)."""
    path = SRC.parent / "scripts" / "fields.py"
    spec = importlib.util.spec_from_file_location("fields_script", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


def _hybrid_harness(root: Path, cell: int = 344, phase: int = 0) -> Path:
    """The hybrid control's float32 harness file from (cell, phase), made as the post-tag
    'harness-f32' stage makes a network's (``harness_det`` in this x64-off process, the loop's
    length, ``harness_path``)."""
    from apexzero.fields import network as N

    fs = _script()
    bank, cells = K.design()
    g1 = N._final_agent("G1")
    sigma = int(bank.stimuli[cells[cell].stim_id].direction)
    prm = N.hybrid_reference_params([sigma], g1.cfg, g1.obs_rms)
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        det = fs.harness_det(N.hybrid_policy(g1.cfg), prm, g1.obs_rms, bank, cells,
                             [(cell, phase)], N.STEPS)
    finally:
        jax.config.update("jax_enable_x64", prev)
    return N.save_det(fs.harness_path(root, "runs/hybrid-control"), det,
                      {"net": "hybrid-0", "loop_steps": N.STEPS, "n_runs": 1})


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    if not _have_all():
        pytest.skip("step 1's design, keys, G1's final or G2's rollouts are not here")
    tmp = tmp_path_factory.mktemp("stages")
    out = tmp / "dry.json"
    harness = _hybrid_harness(tmp)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(SRC), *filter(None, [os.environ.get("PYTHONPATH")])]))
    p = subprocess.run([sys.executable, "-m", "apexzero.fields.stages", "dry-run", "--out",
                        str(out), "--harness", str(harness)], env=env, capture_output=True,
                       text=True, timeout=600, check=False)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return json.loads(out.read_text())


def test_network_key_stage_on_planted_and_random_policies(dry):
    lin, rnd = dry["linear"], dry["random"]
    assert all(lin["valid"].values()) and lin["on_plateau"] and lin["e_star"] >= 0.99
    assert 0.2 < lin["G"] < 0.8 and lin["B"] > 0 and lin["paths_sum_rel_err"] <= 1e-9
    # the pure-pursuit driver steers to the corridor's centre: low G, but T = G + G_out ~ 1
    assert lin["T"] == pytest.approx(lin["G"] + lin["G_out"]) and 0.85 < lin["T"] < 1.1
    assert lin["ckpt_sha256"] == "" and lin["loop_steps"] == 256
    # per outcome, the support end the s_1-anchored grid runs to (design Sections 4 and 6):
    # the decision steer reads the preview from the start (150-500 m ahead plus the spline's
    # reach), the windowed outcomes read further along the run
    se = lin["support_end"]
    assert set(se) == {"c_min", "v_apex", "s_b", "steer"}
    assert lin["s1"] + 150 <= se["steer"] <= lin["s1"] + 520 + MS.SPLINE_REACH
    assert se["c_min"] >= se["steer"] and se["v_apex"] >= se["steer"]
    assert lin["blind"]["reproduces"] and not any(lin["kappa_clip"].values())
    assert lin["gate3"] is None                         # no float32 rollouts for this policy
    shapes = dry["arrays"]["linear"]
    assert shapes["impulse"][:2] == [4, 3] and shapes["paths_kappa"][:2] == [4, 4]
    # the random actor crashes: its plan outcomes are exclusions with NaN rows (G NaN)
    assert rnd["valid"] == {"c_min": False, "v_apex": False, "s_b": False, "steer": True}
    assert rnd["G"] is None and dry["arrays"]["random_impulse_nan_rows"] == [True, True, True,
                                                                               False]
    # a loop exhausted at one key is that key's failure row; the stage goes on
    (short,) = dry["short_loop"]
    assert short["pipeline"] == S.PIPELINE_LOOP and short["key_id"] == lin["key_id"]
    assert "alive after 20 steps" in short["error"]


def test_wiring_of_a_network_without_step1_rollouts(dry):
    """Design Section 15: a network without step 1 rollouts gets its Gate 3 and branch
    reference from its own float32 harness run (x64 off, 'harness-f32'); fields-a2's
    ``network_key`` compares its float64 loop with it per outcome, and a key whose branch
    check fails is that key's pipeline exclusion, listed in the coverage table."""
    from apexzero.fields import network as N

    h = dry["harness"]
    assert h["meta"]["loop_steps"] == N.STEPS
    ok, strict = h["hybrid"], h["hybrid_strict"]
    assert ok["gate3"]["ok"] and ok["gate3"]["mean_err"] <= N.MEAN_TOL
    assert ok["branch"]["ok"] and all(ok["branch"]["ok_by"].values())
    assert ok["branch"]["tol"] == N.BRANCH_TOL                 # an untrained control's default
    assert all(ok["valid"].values())
    # the same run against a zero tolerance: the same branch (end, saturation, events), but
    # every finite outcome's value differs by the float32 / float64 spread
    b = strict["branch"]
    assert b["end_ok"] and b["saturation_ok"] and b["events_ok"] and not b["values_ok"]
    assert b["ok_by"] == {"c_min": False, "v_apex": False, "s_b": False, "steer": True}
    reasons = dry["harness_reasons"]
    assert not any(r.startswith("pipeline") for r in reasons["hybrid"].values())
    assert reasons["hybrid_strict"] == dict.fromkeys(("c_min", "v_apex", "s_b"),
                                                     MS.PIPELINE_BRANCH)
    cov = dry["harness_coverage"]
    assert cov["hybrid_strict@final"]["G"]["excluded"] == {MS.PIPELINE_BRANCH: 1}
    assert cov["hybrid_strict@final"]["G"]["pipeline_keys"] == [strict["key_id"]]
    assert MS.PIPELINE_BRANCH not in cov["hybrid@final"]["G"]["excluded"]
    # off-policy (the anchored-key H4 on another network's states): the start from the det
    # row, but no Gate 3 or branch check, so the zero tolerance excludes nothing
    off = dry["off_policy"]
    assert off["off_policy"] and off["gate3"] is None and off["branch"] is None
    assert off["G"] == pytest.approx(ok["G"])


def test_gate2_trained_body_on_the_linear_driver(dry):
    """gate2-trained's stage body (post-tag on trained networks) run on the planted linear
    driver: both trapezoids and the corner bumps of kappa and the inside width, Gate 2's rule."""
    g = dry["gate2_trained"]
    assert g["pass"] and g["n_checked"] == 1 and g["n_keys"] == 1
    names = dry["gate2_trained_directions"]
    assert len(names) == 12 and {"trapezoid_inside", "trapezoid_outside"} <= set(names)
    assert sum(n.startswith("kappa@corner") for n in names) == 5
    assert sum(n.startswith("w_left@corner") for n in names) == 5       # cell 344 turns left
    (row,) = g["rows"]
    assert row["pass"] and all(e < 1e-3 for e in row["rel_err"]) and min(row["share_counted"]) > 0.5


def test_dose_and_h3_stage_bodies(dry):
    d = dry["dose"]
    assert [r["amplitude"] for r in d["rows"]] == [0.1, 0.3, 1.0, -0.1, -0.3, -1.0, -2.0]
    c = {r["amplitude"]: r["values"][0] for r in d["rows"]}
    assert c[-0.1] < c[0.1] and c[-2.0] < c[-1.0]       # narrowing the inside cuts c_min
    assert d["crash_narrowing_predicted"] == pytest.approx(d["B"] / (1 - d["G"]))
    # descriptive ratios (design Section 4): the linearity ratio per amplitude and the
    # narrowing/widening asymmetry; the linear driver's loop is near-linear up to 2 m
    assert d["slope"]["c_min"] == pytest.approx(1 - d["G"])
    for r in d["rows"]:
        for j, o in enumerate(("c_min", "v_apex")):
            want = (r["values"][j] - d["base_values"][j]) / (r["amplitude"] * d["slope"][o])
            assert r["linearity"][o] == pytest.approx(want)
        assert abs(r["linearity"]["c_min"] - 1) < 0.01
    assert set(d["asymmetry"]) == {"0.1", "0.3", "1"}
    assert all(abs(v["c_min"] - 1) < 0.01 for v in d["asymmetry"].values())
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
    assert a["H4"]["decision"] == "not computed" and len(a["H4"]["missing"]) == 18
    t = a["edge_descriptive"]["linear_driver"]["T"]
    assert t["label"] == "corridor-anchored"
    # the optimum: G* = 1, outside field 0, T* = 1
    o = dry["optimum_row"]
    assert o["G_star"] == pytest.approx(1.0) and o["T_star"] == pytest.approx(1.0)
    assert a["optimum"]["T_star"]["pooled"]["median"] == pytest.approx(1.0)
    # no decisions before the confirmatory run; a missing optimum row is a pipeline exclusion
    assert not dry["plumbing"]["decided"] and "H2" not in dry["plumbing"]
    no = dry["analysis_no_opt"]["linear_driver@final"]["G"]
    assert no["excluded"] == {MS.PIPELINE_NO_OPT: 1}
    assert no["pipeline_keys"] == [dry["linear"]["key_id"]]
    # the field comparisons against the key's real optimum fields (one key: no far geometry,
    # so delta_rho is undefined and every interval is too); H1-typical is descriptive
    fa = dry["fields_analysis"]
    assert set(fa["H1_typical"]) == {"s_b", "v_apex"}
    assert set(fa["H1_typical"]["s_b"]["criteria"]) == {"gain", "location", "shape"}
    assert fa["H1_typical"]["s_b"]["descriptive"] and "pass" not in json.dumps(fa["H1_typical"])
    assert fa["H2_speed"]["decision"] == fa["H2_speed"]["band_decision"]
    assert set(fa["coverage"]) == {"H1_typical/s_b", "H1_typical/v_apex", "H2_speed/v_apex"}
    # against Qp (H1-typical's lawful-reader reference), and Qp against the optimum
    q = dry["qp_row"]
    assert q["net"] == "reader-Qp" and q["valid"]["v_apex"] and q["gate3"] is None
    fq = dry["fields_analysis_qp"]
    assert set(fq["H1_typical_vs_Qp"]) == {"s_b", "v_apex"} and set(fq["Qp_vs_optimum"]) == \
        {"s_b", "v_apex"}
    assert fq["H1_typical_vs_Qp"]["v_apex"]["descriptive"]
    c = dry["comparison"]["v_apex"]
    assert np.isfinite(c["g"]) and -1.0 <= c["rho"] <= 1.0 and np.isfinite(c["location"])
    # the optimum does not brake at this gentle corner: its s_b is invalid, and so is the
    # comparison (NaN, an exclusion), not a zero
    assert dry["comparison"]["s_b"]["g"] is None and dry["comparison"]["s_b"]["flags"]["invalid"]
