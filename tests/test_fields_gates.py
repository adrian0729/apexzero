"""Step 2's gates (fields.gates) and the stage script (scripts/fields.py).

- Gate 0 on a spread of A2 and anchored keys; a failing row fails the summary.
- Gate 1's stratified sample (50 keys, reproducible, no W twins), and Gate 1 on one A2 key with
  the design's 17 directions: the clean fields pass, each planted bug fails the pooled rule.
- Gate 4 on A2 keys of both directions: G* = 1 within 1e-6 and the inertia right.
- Gate 2 (random actor, edge keeper), Gate 3 (G2's stored rows, teacher-forced) and Gate 5
  (the KKT replay through the network-side loop on an A2 key at W = 400 and 800 and an
  anchored key) run through ``scripts/fields.py`` in float64 processes, three at once, with
  outputs and logs under a temporary directory.
- The freeze check of the post-tag stages, on scratch git repositories with a bare origin: no
  tag, a lightweight, unpushed or moved tag, an uncommitted, missing or moved tag pin, a
  changed or untracked file under the code paths (in git or byte by byte: assume-unchanged,
  an ignored directory), a record that does not register the tagged design document or does
  not hash the tagged code, changed keys, outputs, calibration or checkpoints each refuse; a
  deviation waives only changed code or a changed design document and records the diff's
  hash; pre-tag stages refuse after the tag, and never write into outputs/fields under a
  deviation; the provenance stamp refuses a mid-run code change before writing; the prereg
  stage's bindings (Gate 2's fixed rows, Gate 3 on every agent, the exclusion lists across
  files), its refusals and the record's layout; the calibration's refusals; the run-once
  guard anchored in git; a post-tag stage refuses code imported from anywhere but the frozen
  paths. The real repository has no tag yet, so every post-tag stage refuses.
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


def _post_tag_args() -> list[str]:
    """After step 2's tag the pre-tag stages refuse to rewrite pre-registered outputs; a test run
    into a tmp --out is a logged deviation there and touches nothing under outputs/fields."""
    tagged = subprocess.run(["git", "tag", "-l", "receptive-fields-prereg-v1"], cwd=REPO,
                            capture_output=True, text=True, check=False).stdout.strip()
    return ["--post-tag-deviation", "test run after the tag (tmp --out)"] if tagged else []


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
    # After step 2's tag the pre-tag stages refuse to rewrite pre-registered outputs; a run into
    # a tmp --out is a logged test deviation there and touches nothing under outputs/fields.
    common += _post_tag_args()
    cmds = {"gate5": ["gate5", "--n", "3"],
            "gate2": ["gate2", "--random", "62", "--keeper", "344", "--linear", "--hybrid",
                      "344"],
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
    from apexzero.fields import prereg

    assert s["provenance"]["partial"] == ["--out", "--limit", "--kinds"]
    st = prereg.read_stamp(root / "opt-fields" / "rows.jsonl")
    assert st["files"]["rows.jsonl"] == prereg.sha256_file(root / "opt-fields" / "rows.jsonl")
    assert s["rows_sha256"] == st["files"]["rows.jsonl"] and s["n_arrays"] == 3
    fs = _script()
    fs.main(["phase-ceiling", "--out", str(root), "--log-dir", str(stages["root"] / "logs"),
             *_post_tag_args()])
    pc = root / "phase-ceiling" / "phase_ceiling.json"
    assert isinstance(json.loads(pc.read_text()), list)
    assert prereg.read_stamp(pc)["stage"] == "phase-ceiling"


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
    # the hybrid control: the trained architecture (a random-init MLP through G1's obs_rms)
    # inside the loop, on plan-level outcomes
    hyb = rows["hybrid@344"]
    assert all(hyb["valid"]) and all(hyb["outcome_ok"]) and hyb["pass"]
    assert all(e < G.GATE2_TOL for e in hyb["rel_err"]) and min(hyb["share_counted"]) >= 0.5
    # every output carries its provenance stamp
    st = r["provenance"]
    assert st["stage"] == "gate2" and len(st["code_hash"]) == 64
    assert st["partial"] == ["--out", "--random", "--linear", "--hybrid"]   # not Gate 2's rows
    assert r["cells"] == {"random": [62], "keeper": [344], "linear": [], "hybrid": [344]}
    assert st["environment"]["packages"]["jax"]


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
    # design Sections 5 and 8: a failed re-polish is a listed exclusion, and Gate 4 passes
    res = G.gate4([ref])
    assert res["pass"] and res["crashed"] == [] and res["n_outside"] == 1
    assert res["outside"][0]["repolish_failed"] and "RepolishFailed" in res["outside"][0]["error"]

    def broken(prob):
        raise KeyError("a corrupt key file")

    monkeypatch.setattr(K, "repolish", broken)          # any other exception: a crash
    res = G.gate4([ref])
    assert not res["pass"] and res["crashed"] == [ref.key_id]


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


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True,
                       env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                                GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))
    return r.stdout


def _tag(root: Path, tag: str, *args: str) -> None:
    _git(root, "tag", "-a", tag, "-m", tag, *args)


def _scratch_repo(tmp_path, fs, push: bool = True):
    """A scratch repository with step 2's and step 1's frozen paths, a design document, fake
    key files and a pre-registration record (``fields.prereg``), built as the real one is:
    the code and the design document committed (step 1's tag there), the record built on
    that commit (its ``prereg_base_commit``) and committed alone, then tagged; both annotated
    tags pushed to a bare origin. outputs/ and build/ are ignored, as in this repository."""
    from apexzero.fields import prereg

    root = tmp_path / "repo"
    for d in ("src/apexzero/fields", "src/apexzero/interp", "src/apexzero/ocp", "scripts",
              "docs/interp"):
        (root / d).mkdir(parents=True)
    (root / "src/apexzero/fields/a.py").write_text("x = 1\n")
    (root / "src/apexzero/interp/m.py").write_text("y = 1\n")
    (root / "src/apexzero/ocp/w.py").write_text("z = 1\n")
    (root / "scripts/fields.py").write_text("y = 1\n")
    (root / "scripts/_bootstrap.py").write_text("b = 1\n")
    (root / "scripts/psychophysics.py").write_text("p = 1\n")
    (root / prereg.DESIGN_DOC).write_text("design v3 — σ\n")
    (root / "notes.txt").write_text("n\n")
    (root / "uv.lock").write_text("lock 1\n")
    (root / ".gitignore").write_text("outputs/\nbuild/\n__pycache__/\n")
    keys = tmp_path / "keys"
    (keys / "nominal").mkdir(parents=True)
    for k in ("k1", "k2"):
        (keys / "nominal" / f"{k}.json").write_text(f'{{"key_id": "{k}"}}')
        np.savez(keys / "nominal" / f"{k}.npz", x=np.arange(3.0))
    sets = {"a2": [("nominal", "k1"), ("nominal", "k2")]}
    rows = [{"key_id": "k1", "max_dx": 1e-9, "dJ_rel": 0.0, "flagged": False, "ok": True},
            {"key_id": "k2", "max_dx": 2e-3, "dJ_rel": 1e-9, "flagged": True, "ok": True}]
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c0")
    _tag(root, fs.STEP1_TAG)
    base = _git(root, "rev-parse", "HEAD").strip()
    rec = prereg.build_record(root, sets, keys, rows, "design-hash", extra={
        "prereg_base_commit": base})
    prereg.save(rec, root / fs.PREREG_REL)
    _git(root, "add", fs.PREREG_REL)
    _git(root, "commit", "-q", "-m", "record")
    _tag(root, fs.TAG)
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(root, "remote", "add", "origin", f"file://{origin}")
    if push:
        _git(root, "push", "-q", "origin", fs.TAG, fs.STEP1_TAG)

    def verify(rec_, root_, current=lambda: sets, rows_=rows, design="design-hash"):
        return prereg.verify_record(rec_, root_, keys, current, rows_, design)

    return root, keys, sets, rows, verify


def test_freeze_check_on_a_scratch_repository(tmp_path):
    fs = _script()
    from apexzero.fields import prereg

    root, keys, sets, rows, verify = _scratch_repo(tmp_path, fs)
    pin = tmp_path / "tag.json"

    def probs(**kw):
        return fs.freeze_problems(root=root, verify=kw.pop("verify", verify), pin_path=pin, **kw)

    assert probs() == []
    # later commits that touch only docs or notes are allowed (HEAD may move)
    (root / "notes.txt").write_text("findings\n")
    _git(root, "commit", "-q", "-am", "findings")
    assert probs() == []
    # any code under src/apexzero, committed or not, blocks; so do untracked .py files and
    # scripts/_bootstrap.py
    (root / "src/apexzero/ocp/w.py").write_text("z = 2\n")
    assert any("code differs" in p and p.code == "code_changed" for p in probs())
    _git(root, "checkout", "-q", "--", "src")
    (root / "src/apexzero/fields/b.py").write_text("")
    assert any("untracked" in p for p in probs())
    (root / "src/apexzero/fields/b.py").unlink()
    (root / "scripts/_bootstrap.py").write_text("b = 2\n")
    assert any("code differs" in p for p in probs())
    _git(root, "checkout", "-q", "--", "scripts")
    (root / "scripts/new.py").write_text("")
    assert probs() == []                                            # not a frozen path
    (root / "scripts/new.py").unlink()
    # step 1's frozen code is checked against its own tag, and its tag must exist
    (root / "src/apexzero/interp/m.py").write_text("y = 2\n")
    assert any("step 1's frozen code" in p and p.code == "step1" for p in probs())
    _git(root, "checkout", "-q", "--", "src")
    _git(root, "tag", "-d", fs.STEP1_TAG)
    assert any("is missing" in p for p in probs())
    _tag(root, fs.STEP1_TAG, fs._tag_commit(fs.TAG, root))
    # the record: edited key files, a changed index, other re-polish movement, a changed
    # design or design document each fail
    rec = fs.tagged_record(root=root)
    assert rec is not None and rec["key_counts"] == {"a2": 2} and rec["version"] == 2
    np.savez(keys / "nominal" / "k2.npz", x=np.arange(4.0))
    assert any("key files differ" in p and p.code == "keys" for p in probs())
    np.savez(keys / "nominal" / "k2.npz", x=np.arange(3.0))
    assert probs() == []
    more = {"a2": sets["a2"] + [("nominal", "k3")]}
    assert any("no longer gives" in p for p in probs(
        verify=lambda r, rt: verify(r, rt, current=lambda: more)))
    moved = [dict(rows[1], max_dx=5e-3), rows[0]]
    assert any("re-polish differs" in p for p in probs(
        verify=lambda r, rt: verify(r, rt, rows_=moved)))
    near = [dict(rows[1], max_dx=2e-3 + 1e-8), rows[0]]           # within the recheck tolerance
    assert probs(verify=lambda r, rt: verify(r, rt, rows_=near)) == []
    assert any("design differs" in p for p in probs(
        verify=lambda r, rt: verify(r, rt, design="other")))
    (root / prereg.DESIGN_DOC).write_text("design v4\n")
    assert any("receptive-fields-design.md differs" in p and p.code == "design_doc_changed"
               for p in probs())
    (root / prereg.DESIGN_DOC).write_text("design v3 — σ\n")
    # an edited record in the working tree is a code change (it lies in the frozen paths),
    # though the record never enters the code hash
    h = prereg.code_hash(root)
    (root / fs.PREREG_REL).write_text("{}")
    assert prereg.code_hash(root) == h
    assert any("code differs" in p for p in probs())
    _git(root, "checkout", "-q", "--", "src")
    assert probs() == []


def test_the_tag_is_annotated_pushed_and_pinned(tmp_path):
    """Design Section 15: the tag must be annotated, on origin as the same object, and match
    the write-once tag.json of the first post-tag run."""
    fs = _script()
    root, keys, sets, rows, verify = _scratch_repo(tmp_path, fs, push=False)
    pin = tmp_path / "out" / "tag.json"

    def probs():
        return fs.freeze_problems(root=root, verify=verify, pin_path=pin)

    assert any("is not on origin" in p and p.code == "tag" for p in probs())
    _git(root, "push", "-q", "origin", fs.TAG, fs.STEP1_TAG)
    assert probs() == []
    assert fs.require_tag("a test", root=root, pin_path=pin, problems=probs()) is None
    ident = json.loads(pin.read_text())
    assert ident["type"] == "tag" and ident["commit"] == fs._tag_commit(fs.TAG, root)
    first = pin.read_text()
    fs.pin_tag(root=root, pin_path=pin)                    # write-once
    assert pin.read_text() == first
    # a lightweight tag is refused
    _git(root, "tag", "-d", fs.TAG)
    _git(root, "tag", fs.TAG)
    assert any("lightweight" in p for p in probs())
    # a moved (re-made) annotated tag differs from origin's and from the pin
    _git(root, "tag", "-d", fs.TAG)
    (root / "notes.txt").write_text("later\n")
    _git(root, "commit", "-q", "-am", "later")
    _tag(root, fs.TAG)
    p = probs()
    assert any("is not on origin" in x for x in p) and any("tag moved" in x for x in p)
    _git(root, "push", "-q", "-f", "origin", fs.TAG)
    assert any("tag moved" in x for x in probs())          # the pin still catches it
    pin.unlink()
    assert probs() == []
    # an unreachable remote is a problem, never a pass
    _git(root, "remote", "set-url", "origin", f"file://{tmp_path / 'gone.git'}")
    assert any("cannot confirm" in x and x.code == "tag" for x in probs())


def test_the_record_registers_the_tagged_design_document(tmp_path):
    """The record's design-document hash is checked against the document's blob at the tag
    (bytes): a record built while the document was edited and uncommitted fails."""
    fs = _script()
    from apexzero.fields import prereg

    root, keys, sets, rows, verify = _scratch_repo(tmp_path, fs)
    pin = tmp_path / "tag.json"
    assert fs.tagged_doc_sha256(root=root) == prereg.sha256_file(root / prereg.DESIGN_DOC)
    (root / prereg.DESIGN_DOC).write_text("design v3, still being edited\n")
    rec = prereg.build_record(root, sets, keys, rows, "design-hash", extra={
        "prereg_base_commit": _git(root, "rev-parse", "HEAD").strip()})
    prereg.save(rec, root / fs.PREREG_REL)
    _git(root, "add", fs.PREREG_REL)
    _git(root, "commit", "-q", "-m", "record from a dirty doc")
    _git(root, "tag", "-d", fs.TAG)
    _tag(root, fs.TAG)
    _git(root, "push", "-q", "-f", "origin", fs.TAG)
    p = fs.freeze_problems(root=root, verify=verify, pin_path=pin)
    assert any("not the one the record registers" in x and x.code == "record" for x in p)
    # the prereg stage refuses such a tree in the first place
    assert prereg.DESIGN_DOC in fs.dirty_paths(root)
    (root / fs.PREREG_REL).write_text("{}\n")
    assert fs.PREREG_REL not in fs.dirty_paths(root)       # the record itself may differ


def test_deviations_waive_only_changed_code_or_design_document(tmp_path):
    """--post-tag-deviation waives only 'code_changed' and 'design_doc_changed' problems,
    recording the diff's sha256, HEAD and the tag's commit; it never waives a missing tag or
    record, changed keys or checkpoints, step 1's code or a git error."""
    fs = _script()
    from apexzero.fields import prereg

    root, keys, sets, rows, verify = _scratch_repo(tmp_path, fs)
    pin, log = tmp_path / "tag.json", tmp_path / "log"
    (root / "src/apexzero/fields/a.py").write_text("x = 2\n")
    probs = fs.freeze_problems(root=root, verify=verify, pin_path=pin)
    assert probs and prereg.waivable(probs)
    with pytest.raises(SystemExit, match="refusing"):
        fs.require_tag("a test", root=root, problems=probs, pin_path=pin, log_root=log)
    dev = fs.require_tag("a test", "a fix", root=root, problems=probs, pin_path=pin,
                         log_root=log)
    entry = json.loads((log / "deviations.jsonl").read_text().splitlines()[-1])
    assert dev == entry and entry["reason"] == "a fix" and entry["diff_bytes"] > 0
    assert len(entry["diff_sha256"]) == 64 and entry["head"] == _git(root, "rev-parse",
                                                                      "HEAD").strip()
    assert entry["tag_commit"] == fs._tag_commit(fs.TAG, root)
    assert {p["code"] for p in entry["problems"]} == {"code_changed"}
    for code in ("keys", "checkpoints", "record", "tag", "step1", "git", "outputs", ""):
        bad = probs + [prereg.Problem(code, f"a {code} problem")]
        with pytest.raises(SystemExit, match="cannot waive"):
            fs.require_tag("a test", "a fix", root=root, problems=bad, pin_path=pin,
                           log_root=log)
    # a plain string (no category) is never waivable
    assert not prereg.waivable(["something"])


def test_pre_tag_stages_refuse_after_the_tag(tmp_path):
    fs = _script()
    root, *_ = _scratch_repo(tmp_path, fs)
    with pytest.raises(SystemExit, match="after the tag"):
        fs.refuse_after_tag("opt-fields", root=root, log_root=tmp_path)
    dev = fs.refuse_after_tag("opt-fields", "re-run on another machine", root=root,
                              log_root=tmp_path)
    assert dev["category"] == "pre-tag stage after the tag" and dev["stage"] == "opt-fields"
    with pytest.raises(SystemExit, match="after the tag"):
        fs.refuse_after_tag("prereg", "any reason", always=True, root=root, log_root=tmp_path)
    # without the tag nothing is refused
    _git(root, "tag", "-d", fs.TAG)
    assert fs.refuse_after_tag("opt-fields", root=root) is None


def test_provenance_stamps_refuse_a_mid_run_code_change(tmp_path):
    fs = _script()
    root, *_ = _scratch_repo(tmp_path, fs)
    args = fs.build_parser().parse_args(["opt-fields", "--limit", "3"])
    pv = fs.Provenance(args, "opt-fields", root)
    st = pv.stamp({"rows.jsonl": "abc"})
    assert st["code_hash"] == pv.code_hash and st["partial"] == ["--limit"]
    elsewhere = fs.build_parser().parse_args(["gate2", "--out", str(tmp_path)])
    assert fs._partial(elsewhere) == ["--out"]
    assert st["head"] == _git(root, "rev-parse", "HEAD").strip() and st["changed_vs_head"] == []
    (root / "src/apexzero/fields/a.py").write_text("x = 3\n")
    with pytest.raises(SystemExit, match="code changed while the stage ran"):
        pv.stamp()
    full = fs.Provenance(fs.build_parser().parse_args(["opt-fields"]), "opt-fields", root)
    assert full.partial == [] and "src/apexzero/fields/a.py" in full.changed
    g3 = fs.build_parser().parse_args(["gate3", "--all", "--agent", "G2"])
    assert fs._partial(g3) == ["gate3 without --all on every agent"]


def test_calibration_refuses_incomplete_gate3():
    fs = _script()
    good = {"rows": [{"ok": True, "mean_err": 2e-5}],
            "every_key": {"n_checked": 2, "failures": [],
                          "rows": [{"ok": True, "mean_err": 3e-5}, {"skipped": "no row"},
                                   {"ok": False, "mean_err": 9.0}]}}
    per, probs = fs.calibration_errors({"agents": {"A": good}}, ["A"])
    assert probs == [] and per == {"A": 3e-5}           # a failing key never inflates it
    for g3, want in (({"agents": {}}, "missing"),
                     ({"agents": {"A": {"rows": []}}}, "every-key"),
                     ({"agents": {"A": {**good, "every_key": {"n_checked": 0, "rows": []}}}},
                      "no key checked"),
                     ({"agents": {"A": {**good, "every_key": {
                         "n_checked": 1, "rows": [{"ok": True, "mean_err": float("nan")}]}}}},
                      "non-finite")):
        assert any(want in p for p in fs.calibration_errors(g3, ["A"])[1]), want


def _stamped(path: Path, obj, code: str, partial=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**obj, "provenance": {"code_hash": code,
                                                      "partial": list(partial)}}))


def _gate2_rows(labels=None) -> dict:
    labels = sorted(G.gate2_labels()) if labels is None else labels
    return {"rows": [{"label": x, "pass": True} for x in labels], "pass": True}


def _gate3_all() -> dict:
    row = {"ok": True, "mean_err": 1e-5}
    return {"agents": {a: {"rows": [row] * (G.GATE3_N_ANCHORED + G.GATE3_N_A2), "pass": True,
                           "every_key": {"n_checked": 1, "rows": [row], "failures": []}}
                       for a in agents.AGENTS}, "pass": True}


def test_bound_outputs_and_their_verification(tmp_path):
    """bind_outputs: every pre-tag output stamped with this code, passing (complete: Gate 2's
    fixed rows, Gate 3 on every agent; listed exclusions are not failures), and covering
    exactly the listed keys; Gate 4's outside list equal to opt-fields' A2 exclusions and
    re-polish's not_ok list to its failed rows; the record then re-hashes them, the
    opt-fields arrays and the calibration."""
    from apexzero.fields import prereg

    code = "c" * 64
    sets = {"a2": [("nominal", "k1"), ("nominal", "k2")], "finals:G1": [("anchored", "a1")]}
    every = [("nominal", "k1"), ("nominal", "k2"), ("anchored", "a1")]
    r = tmp_path
    _stamped(r / "keys/gate0.json", {"summary": {"pass": True},
                                     "rows": [{"kind": k, "key_id": i} for k, i in every]}, code)
    _stamped(r / "keys/loop_audit.json", {"ok": True, "h3": {"ok": True}}, code)
    _stamped(r / "repolish/summary.json", {"pass": True, "not_ok": []}, code)
    _stamped(r / "gates/gate1.json", {"decision": {"pass": True}}, code)
    _stamped(r / "gates/gate2.json", _gate2_rows(), code)
    _stamped(r / "gates/gate3.json", _gate3_all(), code)
    _stamped(r / "gates/gate5.json", {"pass": True}, code)
    _stamped(r / "gates/gate4.json", {"pass": True, "outside": [],
                                      "rows": [{"key_id": "k1"}, {"key_id": "k2"}]}, code)
    _stamped(r / "controls/controls.json", {"pass": {"RF-C1": True}}, code)
    _stamped(r / "calibration/branch_tol.json", {"controls_branch_ok": True,
                                                 "agents_missing_from_gate3": [],
                                                 "tolerance": {"c_min": 7e-4},
                                                 "gate3_sha256": "g3"}, code)
    opt = [{"key_id": i, "kind": k, "failed": False, "adjoint_ok": True,
            "inertia_failed": False, "in_band": True, "on_plateau": True,
            "valid": {"c_min": True}} for k, i in every]
    (r / "opt-fields/arrays").mkdir(parents=True)
    (r / "opt-fields/rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in opt))
    for _, i in every:
        np.savez(r / "opt-fields/arrays" / f"{i}.npz", x=np.arange(2.0))
    _stamped(r / "opt-fields/summary.json", {"n": 3, "failed": []}, code)
    qp = [{"key_id": "k1"}, {"key_id": "k2"}]
    (r / "qp-fields/arrays/reader-Qp").mkdir(parents=True)
    (r / "qp-fields/rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in qp))
    for x in qp:
        np.savez(r / "qp-fields/arrays/reader-Qp" / f"{x['key_id']}.npz", x=np.arange(2.0))
    _stamped(r / "qp-fields/summary.json", {"n": 2}, code)
    out, probs = prereg.bind_outputs(r, code, sets, {"k1", "k2"})
    assert probs == [], probs
    assert set(out["files"]) >= set(prereg.BOUND_OUTPUTS) and out["opt_fields"]["n_arrays"] == 3
    assert out["exclusions"]["outside_gate4"] == []
    cal, cp = prereg.calibration_binding(r, "g3")
    assert cp == [] and cal["tolerance"] == {"c_min": 7e-4}
    assert prereg.calibration_binding(r, "other")[1]
    # an empty or reduced Gate 2, an incomplete Gate 3: never bound as passing
    for bad in ({"rows": [], "pass": True},
                _gate2_rows(sorted(G.gate2_labels() - {"hybrid@66"})),
                _gate2_rows(sorted(G.gate2_labels() | {"linear@62"})),
                {**_gate2_rows(), "rows": [{"label": x, "pass": x != "random@62"}
                                           for x in sorted(G.gate2_labels())]}):
        _stamped(r / "gates/gate2.json", bad, code)
        assert "gates/gate2.json does not pass" in prereg.bind_outputs(r, code, sets,
                                                                        {"k1", "k2"})[1]
    _stamped(r / "gates/gate2.json", _gate2_rows(), code)
    g3 = _gate3_all()
    for drop in ("agent", "every_key", "planted"):
        bad = json.loads(json.dumps(g3))
        if drop == "agent":
            bad["agents"].pop("G2")
        elif drop == "every_key":
            bad["agents"]["G2"].pop("every_key")
        else:
            bad["agents"]["G2"]["rows"] = bad["agents"]["G2"]["rows"][:1]
        _stamped(r / "gates/gate3.json", bad, code)
        assert "gates/gate3.json does not pass" in prereg.bind_outputs(r, code, sets,
                                                                        {"k1", "k2"})[1], drop
    _stamped(r / "gates/gate3.json", g3, code)
    # design Sections 5 and 8: an A2 key outside Gate 4's band (or failing to re-polish),
    # listed consistently in gate4.json, opt-fields and re-polish, is an exclusion, not a
    # failure of the record
    _stamped(r / "gates/gate4.json", {"pass": True, "outside": [{"key_id": "k2"}],
                                      "rows": [{"key_id": "k1"}, {"key_id": "k2"}]}, code)
    (r / "opt-fields/rows.jsonl").write_text("".join(
        json.dumps(dict(x, in_band=x["key_id"] != "k2")) + "\n" for x in opt))
    out2, probs = prereg.bind_outputs(r, code, sets, {"k1", "k2"})
    assert probs == [] and out2["exclusions"]["outside_gate4"] == ["k2"]
    failed = [dict(x, failed=x["key_id"] == "k2") for x in opt]
    (r / "opt-fields/rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in failed))
    (r / "opt-fields/arrays/k2.npz").unlink()
    _stamped(r / "repolish/summary.json", {"pass": True, "not_ok": ["k2"]}, code)
    _stamped(r / "opt-fields/summary.json", {"n": 3, "failed": ["k2"], "crashed": []}, code)
    out2, probs = prereg.bind_outputs(r, code, sets, {"k1", "k2"})
    assert probs == [] and out2["exclusions"]["numerical_failed"] == ["k2"]
    _stamped(r / "repolish/summary.json", {"pass": True, "not_ok": []}, code)  # disagreement
    assert any("re-polish's not_ok list" in p for p in prereg.bind_outputs(r, code, sets,
                                                                          {"k1", "k2"})[1])
    _stamped(r / "opt-fields/summary.json", {"n": 3, "failed": [], "crashed": ["k1"]}, code)
    assert "opt-fields/summary.json does not pass" in prereg.bind_outputs(r, code, sets)[1]
    _stamped(r / "opt-fields/summary.json", {"n": 3, "failed": []}, code)
    _stamped(r / "repolish/summary.json", {"pass": True, "not_ok": []}, code)
    np.savez(r / "opt-fields/arrays/k2.npz", x=np.arange(2.0))
    (r / "opt-fields/rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in opt))
    _stamped(r / "gates/gate4.json", {"pass": True, "outside": [],
                                      "rows": [{"key_id": "k1"}, {"key_id": "k2"}]}, code)
    # other code, a partial run, a failing gate, an extra array, a Gate 4 disagreement
    _stamped(r / "gates/gate2.json", _gate2_rows(), "d" * 64)
    _stamped(r / "gates/gate5.json", {"pass": False}, code)
    _stamped(r / "repolish/summary.json", {"pass": True, "not_ok": []}, code, ["--limit"])
    np.savez(r / "opt-fields/arrays/extra.npz", x=np.arange(1.0))
    (r / "opt-fields/rows.jsonl").write_text("".join(
        json.dumps(dict(x, in_band=x["key_id"] != "k2")) + "\n" for x in opt))
    _, probs = prereg.bind_outputs(r, code, sets, {"k1", "k2"})
    txt = " | ".join(probs)
    for want in ("gates/gate2.json was written by other code", "gates/gate5.json does not pass",
                 "repolish/summary.json is a partial run", "array files of no row",
                 "Gate 4's outside list"):
        assert want in txt, want
    (r / "opt-fields/arrays/extra.npz").unlink()
    (r / "opt-fields/rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in opt[:2]))
    assert any("not exactly the record's" in p for p in prereg.bind_outputs(r, code, sets)[1])
    # verify_record against a record holding the binding
    (r / "opt-fields/rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in opt))
    _stamped(r / "gates/gate2.json", _gate2_rows(), code)
    _stamped(r / "gates/gate5.json", {"pass": True}, code)
    _stamped(r / "repolish/summary.json", {"pass": True, "not_ok": []}, code)
    out, probs = prereg.bind_outputs(r, code, sets, {"k1", "k2"})
    assert probs == []
    rec = {"version": prereg.VERSION, "key_sets": {}, "content": {},
           "design_doc_sha256": "x", "outputs": out, "calibration": cal,
           "networks": {"G1": {"path": "p", "sha256": "s", "timesteps": 1}},
           "regimes": {"support_sha256": "sup"}}

    def codes(**kw):
        return [p.code for p in prereg.verify_record(rec, tmp_path, tmp_path, out_root=r, **kw)]

    base = codes()
    assert "outputs" not in base and "calibration" not in base
    np.savez(r / "opt-fields/arrays/k1.npz", x=np.arange(5.0))
    assert "outputs" in codes()
    (r / "calibration/branch_tol.json").write_text("{}")
    assert "calibration" in codes()
    assert "checkpoints" in codes(networks=lambda pins: ["G1: changed"])
    assert "support" in codes(support_sha256="other")
    assert "support" not in codes(support_sha256="sup")
    assert prereg.record_tolerance(rec) == {"c_min": 7e-4}
    assert prereg.record_tolerance({}) is None


def test_freeze_check_without_a_record_or_after_a_late_code_change(tmp_path):
    fs = _script()
    from apexzero.fields import prereg

    root = tmp_path / "r"
    (root / "src/apexzero/fields").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "docs/interp").mkdir(parents=True)
    (root / "src/apexzero/fields/a.py").write_text("x = 1\n")
    (root / "scripts/fields.py").write_text("y = 1\n")
    (root / prereg.DESIGN_DOC).write_text("d\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c0")
    pin = tmp_path / "tag.json"
    assert "does not exist" in fs.freeze_problems(root=root, verify=None, remote=None,
                                                  pin_path=pin)[0]
    _tag(root, fs.TAG)
    _tag(root, fs.STEP1_TAG)
    assert any("no pre-registration record" in p and p.code == "record"
               for p in fs.freeze_problems(root=root, verify=None, remote=None, pin_path=pin))
    # a record written, then code edited before the tag: the code hash no longer matches
    keys = tmp_path / "k"
    keys.mkdir()
    rec = prereg.build_record(root, {}, keys, [], "")
    prereg.save(rec, root / fs.PREREG_REL)
    (root / "src/apexzero/fields/a.py").write_text("x = 2\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c1")
    _git(root, "tag", "-d", fs.TAG)
    _tag(root, fs.TAG)
    probs = fs.freeze_problems(root=root, verify=lambda r, rt: prereg.verify_record(r, rt, keys),
                               remote=None, pin_path=pin)
    assert any("code changed after the pre-registration record" in p for p in probs)
    # the tagged code is not the code the record hashes: never waivable (a 'record' problem)
    assert any("the code at the tag is not the code the record was built from" in p
               and p.code == "record" for p in probs)
    assert any("base commit" in p and p.code == "record" for p in probs)
    assert not prereg.waivable(probs)
    with pytest.raises(SystemExit, match="cannot waive"):
        fs.require_tag("a test", "a fix", root=root, problems=probs, pin_path=pin,
                       log_root=tmp_path / "log")
    with pytest.raises(FileNotFoundError):
        prereg.build_record(tmp_path / "nowhere", {}, keys, [], "")


def test_the_record_must_hash_the_tagged_code(tmp_path):
    """Design Section 15 ('hashes against the tagged copy'): the record's code hash is compared
    with the code the tag holds (``prereg.tree_code_hash``, the same rule as ``code_hash``),
    and its base commit with the tag. Code committed between prereg and the tag (with the
    record) is a non-waivable 'record' problem; a post-tag working-tree edit stays a waivable
    'code_changed' one."""
    fs = _script()
    from apexzero.fields import prereg

    root, keys, sets, rows, verify = _scratch_repo(tmp_path, fs)
    pin = tmp_path / "tag.json"

    def probs():
        return fs.freeze_problems(root=root, verify=verify, pin_path=pin)

    assert probs() == []
    assert prereg.tree_code_hash(root, "HEAD") == prereg.code_hash(root)      # one rule, one order
    assert prereg.tree_code_hash(root, fs.TAG) == fs.tagged_record(root=root)["code_hash"]
    # a post-tag working-tree edit: waivable
    (root / "src/apexzero/fields/a.py").write_text("x = 3\n")
    p = probs()
    assert p and prereg.waivable(p) and {x.code for x in p} == {"code_changed"}
    ev = fs.diff_evidence(root=root)
    assert ev["byte_changes"] and ev["prereg_base_commit"] == fs.tagged_record(
        root=root)["prereg_base_commit"]
    # code edited after prereg and committed with the record, then tagged: not waivable
    _git(root, "commit", "-q", "-am", "a fix committed with the record")
    _git(root, "tag", "-d", fs.TAG)
    _tag(root, fs.TAG)
    _git(root, "push", "-q", "-f", "origin", fs.TAG)
    p = probs()
    assert any("the code at the tag is not the code the record was built from" in x
               and x.code == "record" for x in p)
    assert any("by more than the record" in x and x.code == "record" for x in p)
    assert not prereg.waivable(p)
    with pytest.raises(SystemExit, match="cannot waive"):
        fs.require_tag("a test", "a fix", root=root, problems=p, pin_path=pin,
                       log_root=tmp_path / "log")
    # a base commit that is not an ancestor of the tag
    rec = {"code_hash": prereg.tree_code_hash(root, fs.TAG), "prereg_base_commit": "f" * 40}
    assert any("missing or unknown" in x for x in fs._record_base_problems(rec, fs.TAG, root))


def test_hidden_edits_are_caught_byte_by_byte(tmp_path):
    """git's diff misses an assume-unchanged or skip-worktree edit and a .py file under an
    ignored directory; the freeze check compares the bytes on disk with the tag's blobs (step
    1's paths against step 1's tag, never waivable), and the prereg stage refuses bytes that
    HEAD does not hold. All of src/ is frozen; uv.lock is not (its versions are compared)."""
    fs = _script()

    root, keys, sets, rows, verify = _scratch_repo(tmp_path, fs)
    pin = tmp_path / "tag.json"

    def probs():
        return fs.freeze_problems(root=root, verify=verify, pin_path=pin)

    assert probs() == [] and fs.head_problems(root) == []
    a = root / "src/apexzero/fields/a.py"
    _git(root, "update-index", "--assume-unchanged", "src/apexzero/fields/a.py")
    a.write_text("x = 99\n")
    assert fs.dirty_paths(root) == []                       # git sees nothing
    p = probs()
    assert any("byte by byte" in x and "a.py" in x and x.code == "code_changed" for x in p)
    assert fs.head_problems(root) and "not HEAD's" in fs.head_problems(root)[0]
    assert "src/apexzero/fields/a.py" in fs.diff_evidence(root=root)["byte_changes"]
    a.write_text("x = 1\n")
    _git(root, "update-index", "--no-assume-unchanged", "src/apexzero/fields/a.py")
    assert probs() == []
    # step 1's frozen code hidden the same way: 'step1', never waivable
    m = root / "src/apexzero/interp/m.py"
    _git(root, "update-index", "--skip-worktree", "src/apexzero/interp/m.py")
    m.write_text("y = 7\n")
    assert any(x.code == "step1" and "byte by byte" in x for x in probs())
    assert fs.step1_problems(root)
    m.write_text("y = 1\n")
    _git(root, "update-index", "--no-skip-worktree", "src/apexzero/interp/m.py")
    # a .py file in an ignored directory under src/, and an untracked module beside apexzero
    (root / "src/apexzero/build").mkdir()
    (root / "src/apexzero/build/x.py").write_text("")
    assert any("byte by byte" in x and "build/x.py" in x for x in probs())
    (root / "src/apexzero/build/x.py").unlink()
    (root / "src/sitecustomize.py").write_text("import os\n")
    assert any("code differs" in x and x.code == "code_changed" for x in probs())
    (root / "src/sitecustomize.py").unlink()
    assert probs() == []
    # a committed uv.lock change alone is not a code change
    (root / "uv.lock").write_text("lock 2\n")
    _git(root, "commit", "-q", "-am", "lock")
    assert probs() == []


def test_package_versions_are_compared_with_the_record():
    """The record's Python and package versions against the running ones ('code_changed':
    waivable by a logged deviation); the machine is never compared."""
    fs = _script()
    env = fs.environment()
    assert set(env) >= {"python", "packages", "machine", "uv_lock_sha256"}
    assert fs.env_problems(env, {**env, "machine": "other"}) == []
    other = {**env, "packages": {**env["packages"], "numpy": "0.1"}}
    assert fs.env_problems(other, env) == [f"numpy 0.1 -> {env['packages']['numpy']}"]
    assert fs.env_problems(None, env) == []


def test_the_tag_pin_is_committed_and_compared(tmp_path):
    """The write-once tag.json (outputs/fields, ignored by git) must be committed with
    ``git add -f`` before the next post-tag stage; from then on its first committed version
    is the reference: a deleted pin or a moved, force-pushed tag is refused even when the
    pin on disk is gone."""
    fs = _script()
    root, keys, sets, rows, verify = _scratch_repo(tmp_path, fs)
    pin = root / "outputs" / "fields" / "tag.json"

    def probs():
        return fs.freeze_problems(root=root, verify=verify, pin_path=pin)

    assert probs() == []
    assert fs.require_tag("a test", root=root, pin_path=pin, problems=probs()) is None
    assert pin.exists()
    (p,) = probs()
    assert p.code == "tag" and "is not committed" in p and "git add -f" in p
    _git(root, "add", "-f", "outputs/fields/tag.json")
    _git(root, "commit", "-q", "-m", "tag pin")
    assert probs() == []
    pin.unlink()
    assert any("missing on disk" in x and x.code == "tag" for x in probs())
    _git(root, "checkout", "-q", "--", "outputs/fields/tag.json")
    assert probs() == []
    # the tag moved and force-pushed, the disk pin deleted: the committed pin still refuses
    _git(root, "tag", "-d", fs.TAG)
    (root / "notes.txt").write_text("later\n")
    _git(root, "commit", "-q", "-am", "later")
    _tag(root, fs.TAG)
    _git(root, "push", "-q", "-f", "origin", fs.TAG)
    pin.unlink()
    assert any("moved since its pin was committed" in x and x.code == "tag" for x in probs())


def test_the_confirmatory_run_once_is_anchored_in_git(tmp_path):
    """Run-once (design Section 15): decisions.json is committed (``git add -f``); once it was
    ever committed the confirmatory analysis refuses even after the file is deleted, and a
    non-confirmatory analysis is unlocked only by a stamped confirmatory file whose bytes are
    committed."""
    fs = _script()
    root, *_ = _scratch_repo(tmp_path, fs)
    path = root / fs.DECISIONS_REL
    assert fs.run_once_problems(path, root) == []
    tag = fs.tag_identity(root=root)
    body = {"confirmatory": True, "decided": True, "H2": {}, "H4": {}, "H4_transfer": {},
            "inputs": {"a": "b"}}
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**body, "provenance": {
        "stage": "analyse", "partial": [], "argv": ["scripts/fields.py", "analyse",
                                                    "--confirmatory"], "tag": tag}}))
    assert fs.run_once_problems(path, root) == [f"{path} exists"]
    sha, why = fs.unlocked_by(path, tag, root)
    assert sha is None and "is not committed" in why          # stamped, not yet committed
    _git(root, "add", "-f", fs.DECISIONS_REL)
    _git(root, "commit", "-q", "-m", "decisions")
    from apexzero.fields import prereg

    assert fs.unlocked_by(path, tag, root) == (prereg.sha256_file(path), "")
    path.unlink()
    assert fs.run_once_problems(path, root) == [f"{fs.DECISIONS_REL} was committed before"]
    assert fs.unlocked_by(path, tag, root) == (None, "no confirmatory run yet")


def test_a_deviated_pre_tag_stage_never_writes_into_outputs_fields(tmp_path, monkeypatch):
    """After the tag a pre-tag stage runs only under a logged deviation, and only into a root
    outside outputs/fields: one mistaken command must not overwrite the outputs the record
    binds (nothing is logged, unlinked or written before the refusal)."""
    fs = _script()
    root, *_ = _scratch_repo(tmp_path, fs)
    canon = root / "outputs" / "fields"
    for out in (canon, canon / "gates", root / "outputs"):
        with pytest.raises(SystemExit, match="overwrite pre-registered outputs"):
            fs.refuse_after_tag("gate2", "x", root=root, log_root=tmp_path / "log",
                                out_root=out, canon=canon)
    assert not (tmp_path / "log").exists()
    dev = fs.refuse_after_tag("gate2", "x", root=root, log_root=tmp_path / "log",
                              out_root=tmp_path / "elsewhere", canon=canon)
    assert dev["category"] == "pre-tag stage after the tag"
    # through main: refused before the stage body (a repolish --force would unlink its rows)
    logged, ran = [], []
    monkeypatch.setattr(fs, "_tag_object", lambda tag, root=fs.ROOT: ("o" * 40, "tag"))
    monkeypatch.setattr(fs, "log_deviation", lambda *a, **k: logged.append(a) or {"x": 1})
    monkeypatch.setattr(fs, "stage_gate2", lambda args: ran.append("gate2"))
    monkeypatch.setattr(fs, "stage_repolish", lambda args: ran.append("repolish"))
    for argv in (["gate2", "--post-tag-deviation", "x"],
                 ["repolish", "--force", "--post-tag-deviation", "x"]):
        with pytest.raises(SystemExit, match="overwrite pre-registered outputs"):
            fs.main(argv + ["--log-dir", str(tmp_path)])
    assert not logged and not ran
    fs.main(["gate2", "--post-tag-deviation", "x", "--out", str(tmp_path / "o"), "--log-dir",
             str(tmp_path)])
    assert ran == ["gate2"] and len(logged) == 1


def test_gate2_and_gate3_partial_options():
    """Gate 2's rows are fixed (gates.GATE2_CELLS): any other cell list is partial, so the
    record never binds it; Gate 3's planted sample below its default is partial too."""
    fs = _script()
    ap = fs.build_parser()
    assert fs._partial(ap.parse_args(["gate2"])) == []
    assert fs._partial(ap.parse_args(["gate2", "--hybrid"])) == ["--hybrid"]
    assert fs._partial(ap.parse_args(["gate2", "--random", "--keeper", "--linear",
                                      "--hybrid"])) == ["--random", "--keeper", "--linear",
                                                        "--hybrid"]
    assert fs._partial(ap.parse_args(["gate2", "--linear", "66", "345", "344"])) == []
    every = [x for a in agents.AGENTS for x in ("--agent", a)]
    assert fs._partial(ap.parse_args(["gate3", "--all", *every])) == []
    assert fs._partial(ap.parse_args(["gate3", "--all", *every, "--n-a2", "1"])) == ["--n-a2"]
    assert fs._partial(ap.parse_args(["gate3", "--all", *every, "--n-anchored", "0"])) == \
        ["--n-anchored"]
    assert fs._gate2_cells(ap.parse_args(["gate2"])) == G.GATE2_CELLS
    assert G.gate2_labels() == {"random@62", "random@345", "edge-keeper@344", "linear@344",
                                "linear@345", "linear@66", "hybrid@344", "hybrid@66"}


def test_stamped_writes_refuse_before_touching_the_output(tmp_path):
    """Design Section 15: the code hash is taken before writing. A forced mismatch refuses the
    write and leaves the previous output and its stamp as they were, with no temporary file;
    a JSON object's embedded stamp binds its content."""
    fs = _script()
    from apexzero.fields import prereg

    root, *_ = _scratch_repo(tmp_path, fs)
    fs._PROV[:] = [fs.Provenance(fs.build_parser().parse_args(["opt-fields"]), "opt-fields",
                                 root)]
    out = tmp_path / "o"
    rows = out / "rows.jsonl"
    fs._write_rows(rows, [{"key_id": "k1"}])
    before, stamp = rows.read_bytes(), prereg.stamp_path(rows).read_bytes()
    assert prereg.read_stamp(rows)["files"]["rows.jsonl"] == prereg.sha256_file(rows)
    d = fs._write(out / "summary.json", {"n": 1})
    st = prereg.read_stamp(d)
    assert st["content_sha256"] == fs.content_sha256(json.loads(d.read_text()))
    (root / "src/apexzero/fields/a.py").write_text("x = 5\n")       # the code changes mid-run
    with pytest.raises(SystemExit, match="code changed while the stage ran"):
        fs._write_rows(rows, [{"key_id": "k2"}])
    with pytest.raises(SystemExit, match="code changed while the stage ran"):
        fs._write(out / "list.json", [1, 2])
    assert rows.read_bytes() == before and prereg.stamp_path(rows).read_bytes() == stamp
    assert not list(out.glob(".tmp-*")) and not (out / "list.json").exists()
    fs._PROV[:] = []


def test_prereg_refuses_step1_code_that_is_not_its_tag(tmp_path, monkeypatch):
    """The prereg stage checks step 1's freeze too (``step1_problems``): a commit under
    src/apexzero/interp after step 1's tag is refused before any record is built, although
    the tree is clean."""
    fs = _script()
    root, *_ = _scratch_repo(tmp_path, fs)
    assert fs.step1_problems(root) == []
    (root / "src/apexzero/interp/m.py").write_text("y = 3\n")
    _git(root, "commit", "-q", "-am", "an interp change")
    assert fs.dirty_paths(root) == []
    (p,) = fs.step1_problems(root)
    assert p.code == "step1" and "differs from" in p
    monkeypatch.setattr(fs, "dirty_paths", lambda root=fs.ROOT: [])
    monkeypatch.setattr(fs, "step1_problems", lambda root=fs.ROOT: [p])
    monkeypatch.setattr(fs, "_tag_object", lambda tag, root=fs.ROOT: None)
    with pytest.raises(SystemExit, match="step 1's frozen code is not its tag's"):
        fs.stage_prereg(fs.build_parser().parse_args(["prereg", "--out", str(tmp_path)]))
    monkeypatch.setattr(fs, "dirty_paths", lambda root=fs.ROOT: ["x"])
    with pytest.raises(SystemExit, match="prereg refuses a dirty tree"):
        fs.stage_prereg(fs.build_parser().parse_args(["prereg", "--out", str(tmp_path)]))


def test_calibration_control_rows_and_refusals(tmp_path):
    """calibrate's control rows: the linear driver's, the hybrid control's on both of its
    cells and the lane keeper's; a control whose own branch check fails is refused; the
    stage refuses without a complete Gate 3 (x64 off in this process, as calibrate needs)."""
    from apexzero.fields import network as N

    fs = _script()

    def row(src, ok=True):
        return {"source": src, "branch": {"values": {"c_min": [0.1, 0.1]}, "ok": ok},
                "wiring": {"mean_err": 1e-6}}

    reps = {"linear": {"rows": [row("linear@344")]},
            "hybrid": {"rows": [row(f"hybrid@{c}") for c in N.HYBRID_CELLS]},
            "random": {"keeper": [row("keeper@380")]}}
    got = fs.control_rows(reps)
    assert [r["source"] for r in got] == ["linear@344", "hybrid@344", "hybrid@66", "keeper@380"]
    assert all(r["branch_ok"] and r["mean_err"] == 1e-6 for r in got)
    reps["hybrid"]["rows"][1] = row("hybrid@66", ok=False)
    with pytest.raises(SystemExit, match=r"own branch check failed.*hybrid@66"):
        fs.control_rows(reps)
    jax.config.update("jax_enable_x64", False)
    args = fs.build_parser().parse_args(["calibrate", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="gate3 --all"):
        fs.stage_calibrate(args)
    (tmp_path / "o" / "gates").mkdir(parents=True, exist_ok=True)
    (tmp_path / "o" / "gates" / "gate3.json").write_text(json.dumps({"agents": {}}))
    with pytest.raises(SystemExit, match="calibrate refuses"):
        fs.stage_calibrate(args)


@needs_keys
def test_the_record_layout_is_what_post_tag_readers_expect(tmp_path):
    """The record as the prereg stage assembles it (``record_extra``), saved and read back as
    JSON (tuples become lists), is what the post-tag readers take: the calibrated tolerance,
    the gate2-trained and dose key lists, the bulk A2 cells, the support file's hash."""
    from apexzero.fields import prereg
    from apexzero.fields import stages as S

    fs = _script()
    g2 = [(r.kind, r.key_id) for r in S.gate2_trained_sample()]
    dose = [(r.kind, r.key_id) for r in S.dose_sample()]
    cal = {"path": prereg.CALIBRATION, "sha256": "c", "tolerance": {"c_min": 1e-3, "v_apex": 1e-3,
                                                                    "s_b": 1e-2}}
    regimes = {"support_sha256": "sup", "a2_regime_cells": {"G1": {"bulk": [1, 2]}}}
    extra = fs.record_extra("h" * 40, {"G1": {"path": "p", "sha256": "s", "timesteps": 7}},
                            regimes, {"G1@7": {"path": "r", "det_sha256": "d"}}, {"files": {}},
                            cal, g2, dose)
    assert extra["step1_tag"] == {"tag": fs.STEP1_TAG, "object": fs.STEP1_TAG_OBJECT}
    rec = prereg.build_record(REPO, {}, tmp_path, [], "", {}, prereg.tolerances(
        cal["tolerance"]), extra)
    path = prereg.save(rec, tmp_path / "prereg.json")
    rec = json.loads(path.read_text())
    assert prereg.record_tolerance(rec) == cal["tolerance"]
    assert [(r.kind, r.key_id) for r in fs._frozen_refs((), rec, rec["gate2_trained"])] == \
        sorted(g2)
    assert {(r.kind, r.key_id) for r in fs._frozen_refs((), rec, rec["dose_sample"])} == set(dose)
    rec["key_sets"] = {"a2": [list(t) for t in g2]}
    assert not any("no bulk A2 cells" in p for p in
                   S.confirmatory_input_problems(rec, tmp_path, ("G1",)))
    codes = {p.code for p in prereg.verify_record(rec, REPO, tmp_path, support_sha256="sup",
                                                  rollouts=lambda rr: [])}
    assert "support" not in codes and rec["prereg_base_commit"] == "h" * 40


def test_post_tag_stages_are_built_and_unlock_trained_networks(monkeypatch, tmp_path):
    """No post-tag stage is a stub (each body calls fields.stages); with the freeze check
    passing, _post_tag switches x64 on and unlocks trained networks' fields, and refuses a
    record without a calibrated tolerance or checkpoint pins."""
    import inspect

    fs = _script()
    src = inspect.getsource(fs)
    assert "NotImplementedError" not in src
    for stage in fs.POST_TAG:
        body = inspect.getsource(getattr(fs, "stage_" + stage.replace("-", "_")))
        assert "_post_tag(" in body and "S." in body, stage
    from apexzero.fields import network as N

    rec = {"calibration": {"tolerance": {"c_min": 1e-3, "v_apex": 1e-3, "s_b": 1e-2}},
           "networks": {"G1": {"path": "p", "sha256": "s", "timesteps": 1}}}
    monkeypatch.setattr(fs, "freeze_problems", lambda *a, **k: [])
    monkeypatch.setattr(fs, "tagged_record", lambda *a, **k: rec)
    monkeypatch.setattr(fs, "import_problems", lambda *a, **k: [])   # pytest's own modules
    monkeypatch.setattr(N, "_TRAINED_OK", False)
    args = fs.build_parser().parse_args(["analyse", "--out", str(tmp_path)])
    assert fs._post_tag(args, "a test") is rec
    assert N._TRAINED_OK and jax.config.jax_enable_x64
    monkeypatch.setattr(fs, "tagged_record", lambda *a, **k: {**rec, "calibration": {}})
    with pytest.raises(SystemExit, match="calibrated branch tolerance"):
        fs._post_tag(args, "a test")
    monkeypatch.setattr(fs, "tagged_record", lambda *a, **k: {**rec, "networks": {}})
    with pytest.raises(SystemExit, match="pins no network"):
        fs._post_tag(args, "a test")
    # the confirmatory analysis refuses --agent and any root but outputs/fields
    monkeypatch.setattr(fs, "tagged_record", lambda *a, **k: rec)
    for argv, want in ((["analyse", "--confirmatory", "--agent", "G1"], "--agent"),
                       (["analyse", "--confirmatory", "--out", str(tmp_path)], "drop --out")):
        with pytest.raises(SystemExit, match=want):
            fs.stage_analyse(fs.build_parser().parse_args(argv))
    # an exploratory --network never lands in outputs/fields
    a = fs.build_parser().parse_args(["fields-a2", "--network", "G1"])
    with pytest.raises(SystemExit, match="exploratory"):
        fs._exploratory(a, ["G2"])
    assert fs._exploratory(fs.build_parser().parse_args(
        ["fields-a2", "--network", "G1", "--out", str(tmp_path)]), ["G2"]) == ["G1"]


def test_post_tag_stages_use_the_checked_out_package(tmp_path):
    """The freeze check hashes <root>/src, so a post-tag stage refuses to run on an apexzero
    imported from anywhere else, or on any module loaded from inside the repository but
    outside the frozen code (a scripts/numpy.py would shadow numpy); the environment's own
    packages (.venv) and the frozen files are fine."""
    from types import SimpleNamespace

    fs = _script()
    assert fs.import_problems(modules={}) == []
    (bad,) = fs.import_problems(tmp_path, modules={})
    assert "apexzero is imported from" in bad and str(tmp_path) in bad
    ok = {"apexzero.fields.stages": SimpleNamespace(__file__=str(REPO / "src/apexzero/x.py")),
          "fields_script": SimpleNamespace(__file__=str(SCRIPT)),
          "numpy": SimpleNamespace(__file__=np.__file__), "builtin": SimpleNamespace()}
    assert fs.import_problems(modules=ok) == []
    shadow = {"numpy": SimpleNamespace(__file__=str(REPO / "scripts" / "numpy.py")),
              "sitecustomize": SimpleNamespace(__file__=str(REPO / "src" / "sitecustomize.py"))}
    p = fs.import_problems(modules=shadow)
    assert len(p) == 2 and all("outside the frozen code" in x for x in p)


def test_post_tag_stages_refuse_without_the_tag(tmp_path, monkeypatch):
    fs = _script()
    if fs._tag_object(fs.TAG) is not None:
        pytest.skip("the tag exists: the refusal is tested on the scratch repository")
    monkeypatch.setattr(fs, "import_problems", lambda *a, **k: [])   # pytest's own modules
    for stage in fs.POST_TAG:
        argv = [stage, "--out", str(tmp_path), "--log-dir", str(tmp_path)]
        with pytest.raises(SystemExit, match="refusing.*does not exist yet"):
            fs.main(argv)
    ap = fs.build_parser()
    for stage in fs.STAGES:
        assert ap.parse_args([stage]).stage == stage
    assert not (fs.OUT / "tag.json").exists()              # nothing pinned without the tag


@pytest.mark.skipif(not HAVE_ALL, reason="step 1's keys, G1 / G2's finals or G2's rollouts are not here")
def test_harness_runs_in_chunks_and_the_det_routing(tmp_path):
    """harness-f32's body (post-tag on trained networks) on a random-init actor: the chunked
    float32 harness gives one det group, one run per (cell, phase); fields-a2 reads it for a
    network without step 1 rollouts, and step 1's rollouts for an H4 baseline whose recorded
    checkpoint is a registry agent's."""
    from apexzero.fields import analysis as A
    from apexzero.fields import network as N
    from apexzero.interp import rollout

    fs = _script()
    bank, cells = K.design()
    g1 = N._final_agent("G1")
    starts = [(62, 0), (345, 1), (102, 0)]
    fn, prm = rollout.agent_policy(g1.cfg), N.random_actor_params(g1.cfg, 0)
    one = fs.harness_det(fn, prm, g1.obs_rms, bank, cells, starts, 40, chunk=8)
    two = fs.harness_det(fn, prm, g1.obs_rms, bank, cells, starts, 40, chunk=2)
    assert set(one) == set(two)
    for k in one:                       # the batch's size moves float32 rounding (fixed chunk)
        if np.issubdtype(np.asarray(one[k]).dtype, np.floating):
            np.testing.assert_allclose(one[k], two[k], rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_array_equal(one[k], two[k])
    assert [N.det_row(two, c, k) for c, k in starts] == [0, 1, 2]
    assert len(fs.without_rollouts(A.A2_NETWORKS)) == 20
    assert "runs/dr-pilot-200M" not in fs.without_rollouts(A.A2_NETWORKS)
    real = "runs/dr-realw-v1-s0"
    with pytest.raises(SystemExit, match="harness-f32"):
        fs._det_of(real, own=True, rec={"networks": {}}, harness_root=tmp_path, require=True)
    N.save_det(fs.harness_path(tmp_path, real), two, {"net": "x"})
    get = fs._det_of(real, own=True, rec={"networks": {}}, harness_root=tmp_path, require=True)
    ref = next(r for r in K.a2_primary_refs() if (r.cell_id, r.phase) == (62, 0))
    assert N.det_row(get(ref), 62, 0) == 0
    pins = {"G1": {"sha256": "same"}, "runs/dr-pilot-200M": {"sha256": "same"}}
    via = fs._det_of("runs/dr-pilot-200M", own=True, rec={"networks": pins},
                     harness_root=tmp_path, require=True)(ref)
    assert len(via["cell"]) > 100                          # G1's step 1 det group
    other = {**pins, "runs/dr-pilot-200M": {"sha256": "other"}}
    with pytest.raises(SystemExit, match="harness-f32"):
        fs._det_of("runs/dr-pilot-200M", own=True, rec={"networks": other},
                   harness_root=tmp_path, require=True)


@pytest.mark.skipif(not HAVE_ALL, reason="step 1's keys, G1 / G2's finals or G2's rollouts are not here")
def test_record_pieces_from_this_repository():
    """The prereg stage's pieces on this repository (file reads only): the recorded
    checkpoints (a registry final is step 1's, an H4 baseline's run path the same file), the
    per-regime A2 cell sets with support.json's hash (step 1's manifest's), the stored
    rollouts' det-group hash, and the record's view of the code (it never hashes itself)."""
    from apexzero.fields import prereg
    from apexzero.fields import stages as S

    fs = _script()
    pins, probs = fs.network_pins(["G1", "runs/dr-pilot-200M", "runs/dr-realw-v1-s0"])
    assert probs == []
    assert pins["G1"]["sha256"] == pins["runs/dr-pilot-200M"]["sha256"]
    assert pins["runs/dr-realw-v1-s0"]["path"].startswith("runs/dr-realw-v1-s0/ckpt/")
    assert S.checkpoint_path("G1").name == Path(pins["G1"]["path"]).name
    with pytest.raises(ValueError, match="not the recorded checkpoint"):
        S.load_network("G1", {**pins["G1"], "sha256": "0" * 64}, REPO)
    # the earlier snapshots (fields-earlier, descriptive) are pinned from step 1's manifest
    early = fs.earlier_networks()
    pe, probs = fs.network_pins(early)
    assert probs == [] and len(pe) == 8 and set(early) <= set(fs.pinned_networks())
    for n, pin in pe.items():
        agent, ts = S.split_snapshot(n)
        assert pin["timesteps"] == ts and pin["sha256"] == S.step1_snapshot(agent, ts)["sha256"]
        assert S.expected_net(n, pin) == n and S.network_agent_id(n) == agent
    assert S.split_snapshot("runs/dr-pilot-200M") is None and S.split_snapshot("G1") is None
    with pytest.raises(SystemExit, match="not a recorded network"):
        fs._pin({"networks": {}}, early[0])
    sets = prereg.key_sets()
    reg, probs = fs.regime_record(sets)
    assert probs == [] and len(reg["support_sha256"]) == 64
    assert set(reg["a2_regime_cells"]) >= {"G1", "G3", "G2s1"}
    g1 = reg["a2_regime_cells"]["G1"]
    assert sum(len(v) for v in g1.values()) == len(reg["primary_cells"]) and len(g1["bulk"]) > 30
    assert reg["h4_typing"] == {"v1": "G1", "v3": "G3", "transfer": "G3"}
    assert reg["variant_regime_cells"]["v3"] == reg["a2_regime_cells"]["G3"]
    rr = fs.rollout_record({"G2": K.final_timesteps("G2")})
    (who,) = rr
    assert who.startswith("G2@") and fs._rollout_problems(rr, REPO) == []
    assert fs._rollout_problems({who: {**rr[who], "det_sha256": "x"}}, REPO)
    assert "src/apexzero/fields/prereg.json" not in fs.dirty_paths()
    assert set(fs.environment()["packages"]) >= {"jax", "numpy", "casadi"}
