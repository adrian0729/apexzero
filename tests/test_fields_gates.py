"""Step 2's gates (fields.gates) and the stage script (scripts/fields.py).

- Gate 0 on a spread of A2 and anchored keys; a failing row fails the summary.
- Gate 1's stratified sample (50 keys, reproducible, no W twins), and Gate 1 on one A2 key with
  the design's 17 directions: the clean fields pass, each planted bug fails the pooled rule.
- Gate 4 on A2 keys of both directions: G* = 1 within 1e-6 and the inertia right.
- Gate 2 (random actor, edge keeper), Gate 3 (G2's stored rows, teacher-forced) and Gate 5
  (the KKT replay through the network-side loop on an A2 key at W = 400 and 800 and an
  anchored key) run through ``scripts/fields.py`` in float64 processes, three at once, with
  outputs and logs under a temporary directory.
- The freeze check of the post-tag stages, on a scratch git repository: no tag, HEAD moved,
  a changed or untracked file under src/apexzero/fields or scripts/fields.py each refuse; the
  real repository has no tag yet, so every post-tag stage refuses.
This process keeps x64 off (the optimum's float64 JAX runs inside ``jax.enable_x64`` blocks)."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from apexzero.fields import gates as G
from apexzero.fields import keys as K
from apexzero.fields import kkt as KK
from apexzero.interp import agents, io

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "fields.py"
HAVE_KEYS = (io.KEYS_DIR / "anchored").exists() and (io.STIMULI_DIR / "cells.json").exists()


def _have_agents() -> bool:
    try:
        return all(agents.final_snapshot(agents.AGENTS[a]) is not None for a in ("G1", "G2"))
    except (KeyError, OSError):
        return False


HAVE_ALL = HAVE_KEYS and _have_agents() and (io.ROLLOUTS_DIR / "G2").exists()
needs_keys = pytest.mark.skipif(not HAVE_KEYS, reason="step 1's keys are not on this machine")


def _script():
    spec = importlib.util.spec_from_file_location("fields_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(SCRIPT.parent))
    return mod


@pytest.fixture(autouse=True)
def _restore_x64():
    prev = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", prev)


# ------------------------------------------------------------------------- float64 stages


@pytest.fixture(scope="module")
def stages(tmp_path_factory):
    if not HAVE_ALL:
        pytest.skip("step 1's keys, G1 / G2's finals or G2's rollouts are not here")
    out = tmp_path_factory.mktemp("fields")
    common = ["--out", str(out / "out"), "--log-dir", str(out / "logs")]
    cmds = {"gate5": ["gate5", "--n", "3"],
            "gate2": ["gate2", "--random", "62", "--keeper", "344", "--linear"],
            "gate3": ["gate3", "--agent", "G2", "--n-anchored", "1", "--n-a2", "1", "--all",
                      "--limit", "12"],
            "opt": ["opt-fields", "--kinds", "a2", "--limit", "3"],
            "repolish": ["repolish", "--kinds", "a2", "--limit", "3"],
            "keys": ["keys", "--kinds", "a2", "--limit", "3", "--no-audit"]}
    outputs = {"opt": ("opt-fields", "opt-fields/summary.json"),
               "repolish": ("repolish", "repolish/summary.json"),
               "keys": ("keys", "keys/gate0.json")}
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(REPO / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]))
    procs = {k: subprocess.Popen([sys.executable, str(SCRIPT), *c, *common], cwd=REPO, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for k, c in cmds.items()}
    res = {}
    for k, p in procs.items():
        log, _ = p.communicate(timeout=600)
        assert p.returncode == 0, f"{k} failed:\n{log[-4000:]}"
        name, rel = outputs.get(k, (k, f"gates/{k}.json"))
        res[k] = json.loads((out / "out" / rel).read_text())
        assert (out / "logs" / f"fields-{name}.log").read_text().splitlines()[-1].split()[2] == \
            "done"
    res["root"] = out
    return res


def test_keys_and_repolish_stages(stages):
    """The keys stage (lists and Gate 0) and the re-polish stage, whose summary passes only
    when every key re-polished."""
    g0 = stages["keys"]["summary"]
    assert g0["pass"] and g0["n"] == 3
    rp = stages["repolish"]
    assert rp["n"] == 3 and rp["pass"] and rp["n_ok"] == 3 and rp["not_ok"] == []
    lists = json.loads((stages["root"] / "out" / "keys" / "lists.json").read_text())
    assert lists["a2"]["n"] == 481 and len(lists["a2_primary"]) == 475
    assert len(lists["a2"]["w_twins"]) == 6


def test_opt_fields_and_phase_ceiling_stages(stages):
    """The optimum-fields stage through the CLI (rows, arrays, summary), then the phase
    ceiling stage on its output."""
    s = stages["opt"]
    assert s["n"] == 3 and s["n_failed"] == 0 and s["n_outside_gate4"] == 0
    root = stages["root"] / "out"
    rows = [json.loads(ln) for ln in (root / "opt-fields" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all((root / "opt-fields" / "arrays" / f"{r['key_id']}.npz").exists()
                                  for r in rows)
    fs = _script()
    fs.main(["phase-ceiling", "--out", str(root), "--log-dir", str(stages["root"] / "logs")])
    assert isinstance(json.loads((root / "phase-ceiling" / "phase_ceiling.json").read_text()),
                      list)


def test_gate5_kkt_replay(stages):
    """The KKT fields along Gate 1's directions equal the network-side replay of a* + eps da*
    on the perturbed road (relative L2 << 1e-3), and the state tangents agree to 1e-8."""
    r = stages["gate5"]
    assert r["pass"]
    kinds = {(row["kind"] == "anchored", row["W"]) for row in r["rows"]}
    assert (False, 400) in kinds and (False, 800) in kinds and any(k for k, _ in kinds)
    for row in r["rows"]:
        assert row["n_directions"] == 3 * 8 + 2           # near and corner bumps, trapezoids
        assert set(row["err"]) == set(G.GATE5_OUTCOMES)    # steer is not cross-pipeline
        for q in ("v_apex", "s_b", "c_line"):
            assert row["err"][q] < 1e-8, (row["key_id"], q, row["err"])
        assert all(row["outcome_ok"].values())
        assert row["max_field"]["v_apex"] > 1.0 and row["max_field"]["c_line"] > 0.5
        assert row["err"]["c_min"] == 0.0 and row["max_field"]["c_min"] < 1e-6   # G* = 1
        assert row["state_tangent_err"] < 1e-10 and row["replay_dx"] < 1e-8
        assert row["bound_da"] < G.GATE5_BOUND_TOL
        assert row["picks_ok"] and row["repolish_ok"] and row["adjoint_ok"]
        # the decision check: KKT symmetry plus the decision index both sides read (an
        # identity, reported apart from the cross-pipeline errors)
        dc = row["decision_check"]
        assert dc["ok"] and dc["decision_net"] == dc["decision_kkt"]
        assert dc["steer_err"] < 1e-8
        # the planted road misalignment (delta rolled 10 m, tape unchanged) fails on the
        # road-reading outcomes and leaves the tape's steer untouched
        pr = row["planted_roll"]
        assert pr["fails"] and pr["road_fails"] and pr["steer_unchanged"]
        assert max(pr["err"][q] for q in ("s_b", "c_line") if row["valid"][q]) > G.GATE5_TOL


def test_gate2_random_actor_and_keeper(stages):
    r = stages["gate2"]
    assert r["pass"]
    rows = {row["label"]: row for row in r["rows"]}
    rnd, keep = rows["random@62"], rows["edge-keeper@344"]
    assert rnd["valid"] == [False, False, False, True] and rnd["rel_err"][3] < 1e-6
    assert all(keep["valid"]) and all(keep["outcome_ok"])
    assert all(e < G.GATE2_TOL for e in keep["rel_err"])
    assert min(keep["share_counted"]) >= 0.5
    assert len(keep["names"]) == 3 * 8 + 2                # Gate 1's shared directions


def test_gate3_teacher_forced_wiring(stages):
    r = stages["gate3"]
    g2 = r["agents"]["G2"]
    assert r["pass"] and g2["wrong_obs_rms"] == "G1" and g2["planted_pass"]
    rows = [row for row in g2["rows"] if "skipped" not in row]
    assert {row["kind"] for row in rows} >= {"anchored"}
    for row in rows:
        assert row["ok"] and row["mean_err"] <= 1e-4 and row["planted_fail"]
        assert all(p["mean_err"] > 10 * 1e-4 for p in row["planted"].values())
    # the per-key exclusion list: the clean check on keys spread over every key with a row
    ev = g2["every_key"]
    assert ev["n"] == 12 and ev["n_checked"] >= 10 and not ev["failures"]
    assert ev["max_mean_err"] <= 1e-4 and 50 < ev["max_horizon"] <= 256


# ------------------------------------------------------------------------- in-process gates


@needs_keys
def test_gate0_on_a_spread_of_keys():
    refs = list(K.a2_refs())[::60] + [r for refs in K.anchored_finals().values()
                                      for r in refs[::400]]
    res = G.gate0(refs)
    s = res["summary"]
    assert s["pass"] and s["n"] == len(refs) and s["max_r_free"] <= K.GATE0_TOL
    assert any(k.startswith("anchored/") for k in s["by_group"])
    bad = dict(res["rows"][0], ok=False, r_free=1.0)
    assert not G.gate0_summary(res["rows"] + [bad])["pass"]


@needs_keys
def test_gate1_sample_is_stratified_and_reproducible():
    a, b = G.gate1_sample(), G.gate1_sample()
    assert [r.key_id for r in a] == [r.key_id for r in b] and len(a) == 50
    counts = {s: sum(G.family(r) == s for r in a) for s in G.GATE1_STRATA}
    assert counts == G.GATE1_STRATA
    assert all(K.strata(r).w_is_cell for r in a if r.kind != "anchored")
    assert len({r.key_id for r in a}) == 50


@needs_keys
def test_gate1_on_one_key_with_planted_bugs():
    ref = [r for r in K.a2_refs() if r.W == 400 and K.strata(r).w_is_cell][120]
    row = G.gate1_key(ref)
    assert row["ok_solves"] and row["inertia_ok"] and row["n_solves"] == 4 * (3 * 8 + 2)
    assert row["repolish_ok"] and row["adjoint_ok"] and row["inertia_determined"]
    groups = {k.split("|")[1] for k in row["err"]}
    assert groups == {f"{f}@{r}" for f in ("kappa", "w_left", "w_right")
                      for r in ("near", "corner")} | {"trapezoid_inside", "trapezoid_outside"}
    assert all(row["corner_coverage"].values()), row["corner_coverage"]
    dec = G.gate1_decision([row])
    assert dec["pass"] and dec["clean"]["median"] < KK.GATE1_MEDIAN
    assert dec["quantities_pass"] and dec["coverage_pass"]
    # zero fields are counted, never pooled
    assert dec["clean"]["n"] + dec["clean"]["n_zero"] == len(row["err"])
    assert dec["clean"]["n_zero"] == len(row["zero"])
    assert set(dec["plants"]) == set(KK.PLANTS)
    assert all(p["fails"] for p in dec["plants"].values()), dec["plants"]


def _g1_row(err, zero=(), stratum="A2/400", cover=True, **kw):
    cov = {f"{q}|{f}": cover for q, f in G.GATE1_COVERAGE}
    return {"key_id": "k", "stratum": stratum, "err": err, "zero": list(zero), "planted": {},
            "corner_coverage": cov, "ok_solves": True, "inertia_ok": True, "repolish_ok": True,
            "adjoint_ok": True, **kw}


def test_gate1_decision_rules():
    """The pooled rule runs on non-zero vectors only, again within each quantity; strata
    need corner coverage; a failed key (e.g. a re-polish failure) fails the gate."""
    good = {"v_apex|kappa@corner": 1e-6, "s_b|kappa@corner": 2e-6, "c_line|kappa@near": 1e-7}
    ok = G.gate1_decision([_g1_row(good)])
    assert ok["pass"] and ok["clean"]["n"] == 3
    # 30 zero fields would make any median pass: they are left out of the pool
    zeros = {f"c_min|z{i}": 0.0 for i in range(30)}
    bad = {"steer|kappa@near": 5e-2, "steer|w_left@near": 4e-2}
    d = G.gate1_decision([_g1_row({**good, **zeros, **bad}, zero=list(zeros))])
    assert d["clean"]["n_zero"] == 30 and d["clean"]["n"] == 5
    assert not d["by_quantity"]["steer"]["pass"] and not d["quantities_pass"] and not d["pass"]
    assert not G.gate1_decision([_g1_row(good, cover=False)])["pass"]
    failed = {"key_id": "x", "failed": True, "error": "RepolishFailed: ...", "stratum": "v1"}
    assert not G.gate1_decision([_g1_row(good), failed])["pass"]


def test_gate5_outcome_rule_fails_non_finite_errors():
    valid = dict.fromkeys(G.GATE5_OUTCOMES, True)
    fine = dict.fromkeys(G.GATE5_OUTCOMES, 1e-9)
    assert all(G.gate5_outcomes_ok(fine, valid).values())
    for bad in (float("inf"), float("nan"), 2e-3):
        res = G.gate5_outcomes_ok({**fine, "s_b": bad}, valid)
        assert res["s_b"] is False and res["v_apex"] is True
    assert G.gate5_outcomes_ok({**fine, "s_b": float("nan")}, {**valid, "s_b": False})["s_b"]
    assert np.isnan(G._rel_l2(np.array([1.0, np.nan]), np.array([1.0, 1.0])))
    assert np.isinf(G._rel_l2(np.array([1.0]), np.array([0.0])))
    assert G._rel_l2(np.array([1e-9]), np.array([0.0])) == 0.0


@needs_keys
def test_gate4_on_a2_keys():
    refs = [r for r in K.a2_refs() if K.strata(r).w_is_cell][::150] + list(K.a2_twins()[:1])
    res = G.gate4(refs)
    assert res["pass"] and res["n"] == len(refs) and res["max_abs_dev"] < 1e-6
    assert {row["direction"] for row in res["rows"]} == {1, -1}
    assert any(not row["w_is_cell"] for row in res["rows"])      # a twin is gated too
    assert all(row["ok"] and row["adjoint_ok"] and row["inertia_determined"]
               for row in res["rows"])


@needs_keys
def test_a_failed_repolish_is_a_failed_row(monkeypatch):
    """Fields are never computed at a point that did not re-polish: the gate lists the key."""
    orig = K.repolish

    def failing(prob):
        rp = orig(prob)
        rp.status = "Restoration_Failed"
        return rp

    monkeypatch.setattr(K, "repolish", failing)
    ref = [r for r in K.a2_refs() if r.W == 400][3]
    with pytest.raises(G.RepolishFailed):
        G.problem(ref)
    res = G.gate4([ref])
    assert not res["pass"] and "RepolishFailed" in res["outside"][0]["error"]


def test_zero_rule_of_gate2():
    g = type("G2", (), {})()
    g.ad = np.array([[1e-9, 1.0], [2e-9, 2.0]])
    g.fd = np.array([[0.0, 1.0], [0.0, 2.0 + 1e-6]])
    g.counted = np.array([[True, True], [True, False]])
    err = G.zero_rule_error(g)
    assert err[0] == 0.0 and err[1] == 0.0
    g.fd = np.array([[0.0, 1.1], [0.0, 2.0]])
    assert G.zero_rule_error(g)[1] == pytest.approx(0.1 / 1.1)
    g.fd = np.array([[0.0, 0.0], [0.0, 2.0]])           # FD exactly 0 where AD is not: inf
    assert np.isinf(G.zero_rule_error(g)[1])


# ------------------------------------------------------------------------- the freeze check


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                   env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                            GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))


def _scratch_repo(tmp_path, fs):
    """A scratch repository with step 2's and step 1's frozen paths, a design document, fake
    key files and a pre-registration record (``fields.prereg``) tagged with both tags."""
    from apexzero.fields import prereg

    root = tmp_path / "repo"
    for d in ("src/apexzero/fields", "src/apexzero/interp", "src/apexzero/ocp", "scripts",
              "docs/interp"):
        (root / d).mkdir(parents=True)
    (root / "src/apexzero/fields/a.py").write_text("x = 1\n")
    (root / "src/apexzero/interp/m.py").write_text("y = 1\n")
    (root / "src/apexzero/ocp/w.py").write_text("z = 1\n")
    (root / "scripts/fields.py").write_text("y = 1\n")
    (root / "scripts/psychophysics.py").write_text("p = 1\n")
    (root / prereg.DESIGN_DOC).write_text("design v3\n")
    (root / "notes.txt").write_text("n\n")
    keys = tmp_path / "keys"
    (keys / "nominal").mkdir(parents=True)
    for k in ("k1", "k2"):
        (keys / "nominal" / f"{k}.json").write_text(f'{{"key_id": "{k}"}}')
        np.savez(keys / "nominal" / f"{k}.npz", x=np.arange(3.0))
    sets = {"a2": [("nominal", "k1"), ("nominal", "k2")]}
    rows = [{"key_id": "k1", "max_dx": 1e-9, "dJ_rel": 0.0, "flagged": False, "ok": True},
            {"key_id": "k2", "max_dx": 2e-3, "dJ_rel": 1e-9, "flagged": True, "ok": True}]
    rec = prereg.build_record(root, sets, keys, rows, "design-hash")
    prereg.save(rec, root / fs.PREREG_REL)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c0")
    _git(root, "tag", fs.STEP1_TAG)
    _git(root, "tag", fs.TAG)

    def verify(rec_, root_, current=lambda: sets, rows_=rows, design="design-hash"):
        return prereg.verify_record(rec_, root_, keys, current, rows_, design)

    return root, keys, sets, rows, verify


def test_freeze_check_on_a_scratch_repository(tmp_path):
    fs = _script()
    from apexzero.fields import prereg

    root, keys, sets, rows, verify = _scratch_repo(tmp_path, fs)
    assert fs.freeze_problems(root=root, verify=verify) == []
    # later commits that touch only docs or notes are allowed (HEAD may move)
    (root / "notes.txt").write_text("findings\n")
    _git(root, "commit", "-q", "-am", "findings")
    assert fs.freeze_problems(root=root, verify=verify) == []
    # any code under src/apexzero, committed or not, blocks; so do untracked .py files
    (root / "src/apexzero/ocp/w.py").write_text("z = 2\n")
    assert any("code differs" in p for p in fs.freeze_problems(root=root, verify=verify))
    _git(root, "checkout", "-q", "--", "src")
    (root / "src/apexzero/fields/b.py").write_text("")
    assert any("untracked" in p for p in fs.freeze_problems(root=root, verify=verify))
    (root / "src/apexzero/fields/b.py").unlink()
    (root / "scripts/new.py").write_text("")
    assert fs.freeze_problems(root=root, verify=verify) == []       # not a frozen path
    # step 1's frozen code is checked against its own tag, and its tag must exist
    (root / "src/apexzero/interp/m.py").write_text("y = 2\n")
    probs = fs.freeze_problems(root=root, verify=verify)
    assert any("step 1's frozen code" in p for p in probs)
    _git(root, "checkout", "-q", "--", "src")
    _git(root, "tag", "-d", fs.STEP1_TAG)
    assert any("is missing" in p for p in fs.freeze_problems(root=root, verify=verify))
    _git(root, "tag", fs.STEP1_TAG, fs._tag_commit(fs.TAG, root))
    # the record: edited key files, a changed index, other re-polish movement, a changed
    # design or design document each fail
    rec = fs.tagged_record(root=root)
    assert rec is not None and rec["key_counts"] == {"a2": 2}
    np.savez(keys / "nominal" / "k2.npz", x=np.arange(4.0))
    assert any("key files differ" in p for p in fs.freeze_problems(root=root, verify=verify))
    np.savez(keys / "nominal" / "k2.npz", x=np.arange(3.0))
    assert fs.freeze_problems(root=root, verify=verify) == []
    more = {"a2": sets["a2"] + [("nominal", "k3")]}
    assert any("no longer gives" in p for p in fs.freeze_problems(
        root=root, verify=lambda r, rt: verify(r, rt, current=lambda: more)))
    moved = [dict(rows[1], max_dx=5e-3), rows[0]]
    assert any("re-polish differs" in p for p in fs.freeze_problems(
        root=root, verify=lambda r, rt: verify(r, rt, rows_=moved)))
    near = [dict(rows[1], max_dx=2e-3 + 1e-8), rows[0]]           # within the recheck tolerance
    assert fs.freeze_problems(root=root, verify=lambda r, rt: verify(r, rt, rows_=near)) == []
    assert any("design differs" in p for p in fs.freeze_problems(
        root=root, verify=lambda r, rt: verify(r, rt, design="other")))
    (root / prereg.DESIGN_DOC).write_text("design v4\n")
    assert any("receptive-fields-design.md differs" in p
               for p in fs.freeze_problems(root=root, verify=verify))
    (root / prereg.DESIGN_DOC).write_text("design v3\n")
    # an edited record in the working tree is a code change (it lies in the frozen paths)
    (root / fs.PREREG_REL).write_text("{}")
    assert any("code differs" in p for p in fs.freeze_problems(root=root, verify=verify))
    _git(root, "checkout", "-q", "--", "src")
    assert fs.freeze_problems(root=root, verify=verify) == []


def test_freeze_check_without_a_record_or_after_a_late_code_change(tmp_path):
    fs = _script()
    from apexzero.fields import prereg

    root = tmp_path / "r"
    (root / "src/apexzero/fields").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "src/apexzero/fields/a.py").write_text("x = 1\n")
    (root / "scripts/fields.py").write_text("y = 1\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c0")
    assert "does not exist" in fs.freeze_problems(root=root, verify=None)[0]
    _git(root, "tag", fs.TAG)
    _git(root, "tag", fs.STEP1_TAG)
    assert any("no pre-registration record" in p
               for p in fs.freeze_problems(root=root, verify=None))
    # a record written, then code edited before the tag: the code hash no longer matches
    keys = tmp_path / "k"
    keys.mkdir()
    rec = prereg.build_record(root, {}, keys, [], "")
    prereg.save(rec, root / fs.PREREG_REL)
    (root / "src/apexzero/fields/a.py").write_text("x = 2\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c1")
    _git(root, "tag", "-f", fs.TAG)
    probs = fs.freeze_problems(root=root, verify=lambda r, rt: prereg.verify_record(r, rt, keys))
    assert any("code changed after the pre-registration record" in p for p in probs)


def test_post_tag_stages_are_built_and_unlock_trained_networks(monkeypatch, tmp_path):
    """No post-tag stage is a stub (each body calls fields.stages); with the freeze check
    passing, _post_tag switches x64 on and unlocks trained networks' fields."""
    import inspect

    fs = _script()
    src = inspect.getsource(fs)
    assert "NotImplementedError" not in src
    for stage in fs.POST_TAG:
        body = inspect.getsource(getattr(fs, "stage_" + stage.replace("-", "_")))
        assert "_post_tag(" in body and "S." in body, stage
    from apexzero.fields import network as N

    monkeypatch.setattr(fs, "freeze_problems", lambda *a, **k: [])
    monkeypatch.setattr(N, "_TRAINED_OK", False)
    args = fs.build_parser().parse_args(["analyse", "--out", str(tmp_path)])
    fs._post_tag(args, "a test")
    assert N._TRAINED_OK and jax.config.jax_enable_x64

def test_post_tag_stages_refuse_without_the_tag(tmp_path):
    fs = _script()
    if fs._tag_commit(fs.TAG) is not None:
        pytest.skip("the tag exists: the refusal is tested on the scratch repository")
    for stage in fs.POST_TAG:
        argv = [stage, "--out", str(tmp_path), "--log-dir", str(tmp_path)]
        with pytest.raises(SystemExit, match="refusing"):
            fs.main(argv)
    ap = fs.build_parser()
    for stage in fs.STAGES:
        assert ap.parse_args([stage]).stage == stage
