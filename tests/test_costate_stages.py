"""scripts/costate.py: the freeze, the stamps, the run-once rule and the post-tag stage
bodies (design v2.1, Section 7 'Freeze'), before the tag.

- The freeze check on a scratch repository built as the real one will be (the code, the
  design document and step 1's and step 2's tags; the record built on that commit and
  committed alone; the annotated tag pushed to a bare origin): it passes, then fails for a
  code change (committed or not, untracked, byte by byte), step 1's or step 2's frozen code
  changed, a lightweight or unpushed tag, a changed key file, design document or record, a
  step 1 or step 2 tag that is not the record's.
- Pre-tag stages refuse after the tag (a deviation only into a root outside
  outputs/costate); post-tag stages refuse without it; stamps refuse a mid-run code change;
  the confirmatory analysis runs once; partial options are stamped; k4 needs C2's decision
  setup (inputs, k2b) and prereg freezes K4's band-edge notes.
- The post-tag stage bodies (activations, critic gradients, C1 with its nulls, C2 and C2-sw)
  end to end in a float64 subprocess on a few stored keys, with random-init stand-ins for a
  trained snapshot (G1's config and obs_rms; no trained parameter is evaluated).
- C2's control pool: the primary C2 and C2-sw against the 20 random-init actors with the
  two-level CI, every other record (variants, robustness, strata rows but the primary) against
  the 5 controls with the geometry-only CI, each labelled; the probes stage refuses
  activations without the pool."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apexzero.costate import data_dirs
from apexzero.costate import prereg as PR
from apexzero.costate.probes import POOL_SEEDS as POOL
from apexzero.interp import agents

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "costate.py"
KEYS, STIM = data_dirs()
HAVE_KEYS = (KEYS / "nominal").exists() and (STIM / "cells.json").exists()


def _have_g1() -> bool:
    try:
        return agents.final_snapshot(agents.AGENTS["G1"]) is not None
    except (KeyError, OSError):
        return False


def _script():
    spec = importlib.util.spec_from_file_location("costate_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(SCRIPT.parent))
    return mod


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True,
                       env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                                GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))
    return r.stdout


def _scratch(tmp_path, cs, push: bool = True):
    """A scratch repository (module docstring) with fake key files."""
    root = tmp_path / "repo"
    for d in ("src/apexzero/costate", "src/apexzero/fields", "src/apexzero/interp", "scripts",
              "docs/interp"):
        (root / d).mkdir(parents=True)
    (root / "src/apexzero/costate/a.py").write_text("x = 1\n")
    (root / "src/apexzero/fields/f.py").write_text("f = 1\n")
    (root / "src/apexzero/interp/m.py").write_text("m = 1\n")
    for n in ("costate.py", "fields.py", "_bootstrap.py", "psychophysics.py"):
        (root / "scripts" / n).write_text(f"# {n}\n")
    (root / PR.DESIGN_DOC).write_text("design v2.1 — λ\n")
    (root / "notes.txt").write_text("n\n")
    (root / ".gitignore").write_text("outputs/\n__pycache__/\n")
    keys = tmp_path / "keys"
    (keys / "nominal").mkdir(parents=True)
    for k in ("k1", "k2"):
        (keys / "nominal" / f"{k}.json").write_text(f'{{"key_id": "{k}"}}')
        np.savez(keys / "nominal" / f"{k}.npz", x=np.arange(3.0))
    sets = {"a2": [("nominal", "k1"), ("nominal", "k2")]}
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c0")
    _git(root, "tag", "-a", cs.STEP1_TAG, "-m", "s1")
    _git(root, "tag", "-a", cs.STEP2_TAG, "-m", "s2")
    base = _git(root, "rev-parse", "HEAD").strip()
    steps = {k: {"tag": t, "object": _git(root, "rev-parse", f"refs/tags/{t}").strip()}
             for k, t in (("step1", cs.STEP1_TAG), ("step2", cs.STEP2_TAG))}
    rec = PR.build_record(root, sets, keys, [("k1", 0), ("k1", 5), ("k2", 0)], {"t_H": 270},
                          {"prereg_base_commit": base, "networks": {}, "steps": steps})
    PR.save(rec, root / PR.PREREG_REL)
    _git(root, "add", PR.PREREG_REL)
    _git(root, "commit", "-q", "-m", "record")
    _git(root, "tag", "-a", cs.TAG, "-m", "tag")
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(root, "remote", "add", "origin", f"file://{origin}")
    if push:
        _git(root, "push", "-q", "origin", cs.TAG)

    def verify(rec_, root_, current=lambda: sets):
        return PR.verify_record(rec_, root_, keys, current)

    return root, keys, sets, verify


def test_freeze_check_on_a_scratch_repository(tmp_path):
    cs = _script()
    root, keys, sets, verify = _scratch(tmp_path, cs)
    pin = tmp_path / "tag.json"

    def probs(**kw):
        return cs.freeze_problems(root=root, verify=kw.pop("verify", verify), pin_path=pin, **kw)

    assert probs() == []
    rec = cs.tagged_record(root=root)
    assert rec["key_counts"] == {"a2": 2} and rec["version"] == PR.VERSION
    assert rec["tag"] == cs.TAG and len(rec["t_pairs"]) == 3 and "constants" in rec
    # docs-only commits after the tag are fine
    (root / "notes.txt").write_text("findings\n")
    _git(root, "commit", "-q", "-am", "findings")
    assert probs() == []
    # code under src (uncommitted, committed, untracked) or the script blocks
    (root / "src/apexzero/costate/a.py").write_text("x = 2\n")
    assert any(p.code == "code_changed" for p in probs())
    _git(root, "checkout", "-q", "--", "src")
    (root / "src/apexzero/costate/b.py").write_text("")
    assert any("untracked" in p for p in probs())
    (root / "src/apexzero/costate/b.py").unlink()
    (root / "scripts/costate.py").write_text("# changed\n")
    assert any("code differs" in p for p in probs())
    _git(root, "checkout", "-q", "--", "scripts")
    assert probs() == []
    # an edit git does not see (assume-unchanged) is caught byte by byte
    _git(root, "update-index", "--assume-unchanged", "src/apexzero/costate/a.py")
    (root / "src/apexzero/costate/a.py").write_text("x = 3\n")
    assert any("byte by byte" in p for p in probs())
    (root / "src/apexzero/costate/a.py").write_text("x = 1\n")
    _git(root, "update-index", "--no-assume-unchanged", "src/apexzero/costate/a.py")
    assert probs() == []
    # step 2's frozen code against its own tag (it is under src too)
    (root / "src/apexzero/fields/f.py").write_text("f = 2\n")
    assert any(p.code == "frozen" and "step 2" in p for p in probs())
    _git(root, "checkout", "-q", "--", "src")
    # the record against the disk: key files, design document
    np.savez(keys / "nominal" / "k2.npz", x=np.arange(4.0))
    assert any(p.code == "keys" for p in probs())
    np.savez(keys / "nominal" / "k2.npz", x=np.arange(3.0))
    assert any("no longer gives" in p for p in probs(
        verify=lambda r, rt: verify(r, rt, current=lambda: {"a2": sets["a2"][:1]})))
    (root / PR.DESIGN_DOC).write_text("design v3\n")
    assert any(p.code == "design_doc_changed" for p in probs())
    (root / PR.DESIGN_DOC).write_text("design v2.1 — λ\n")
    assert probs() == []
    # the write-once pin: written by the first post-tag run, then it must be committed
    cs.pin_tag(root, pin_path=root / cs.OUT_REL / "tag.json")
    assert any("not committed" in p for p in cs.freeze_problems(
        root=root, verify=verify, pin_path=root / cs.OUT_REL / "tag.json"))


def test_the_tag_must_be_annotated_pushed_and_hold_the_record(tmp_path):
    cs = _script()
    root, _, _, verify = _scratch(tmp_path, cs, push=False)
    probs = cs.freeze_problems(root=root, verify=verify, pin_path=tmp_path / "p.json")
    assert any("is not on origin" in p for p in probs)
    _git(root, "tag", "-d", cs.TAG)
    _git(root, "tag", cs.TAG)                        # lightweight
    probs = cs.freeze_problems(root=root, verify=verify, remote=None,
                               pin_path=tmp_path / "p.json")
    assert any("lightweight" in p for p in probs)
    _git(root, "tag", "-d", cs.TAG)
    assert any("does not exist" in p for p in cs.freeze_problems(root=root, verify=verify))
    # a record built on other code than the tag's
    _git(root, "tag", "-a", cs.TAG, "-m", "t", "HEAD~1")
    probs = cs.freeze_problems(root=root, verify=None, remote=None, pin_path=tmp_path / "p.json")
    assert any("no pre-registration record" in p for p in probs)


def test_pre_tag_stages_refuse_after_the_tag_and_post_tag_ones_before(tmp_path, monkeypatch):
    cs = _script()
    root, _, _, _ = _scratch(tmp_path, cs)
    with pytest.raises(SystemExit, match="refusing to run k1 after the tag"):
        cs.refuse_after_tag("k1", root=root)
    with pytest.raises(SystemExit, match="after the tag"):
        cs.refuse_after_tag("prereg", "reason", always=True, root=root)
    with pytest.raises(SystemExit, match="give --out outside"):
        cs.refuse_after_tag("k1", "reason", root=root, out_root=root / cs.OUT_REL / "x")
    dev = cs.refuse_after_tag("k1", "a rerun", root=root, out_root=tmp_path / "elsewhere")
    assert dev["reason"] == "a rerun"
    assert (root / cs.OUT_REL / "deviations.jsonl").exists()
    # this repository has no costate tag: every post-tag stage refuses
    monkeypatch.setattr(cs, "import_problems", lambda *a, **k: [])
    cs._use_provenance(cs.Provenance(None, "analyse"))
    from apexzero.costate import critic as CR
    from apexzero.costate import probes as PB
    from apexzero.fields import network as NW

    for stage in cs.POST_TAG:
        with pytest.raises(SystemExit, match="does not exist yet"):
            cs.main([stage, "--out", str(tmp_path / "o")])
        # no trained network was unlocked on the way (review finding: an unlock moved
        # before the freeze check went unnoticed)
        assert not CR._TRAINED_OK and not PB._TRAINED_OK and not NW._TRAINED_OK, stage
    # a waivable problem with a deviation proceeds; a hard one never
    from apexzero.fields.prereg import Problem

    soft = [Problem("code_changed", "x")]
    assert cs.require_tag("s", "why", root=root, problems=soft,
                          pin_path=tmp_path / "pin.json")["category"] == "post-tag deviation"
    with pytest.raises(SystemExit):
        cs.require_tag("s", "why", root=root, problems=[Problem("tag", "y")])


def test_stamps_run_once_and_partial_options(tmp_path, monkeypatch):
    cs = _script()
    args = cs.build_parser().parse_args(["k0", "--limit", "3", "--out", str(tmp_path)])
    assert cs._partial(args) == ["--out", "--limit"]
    p = cs.Provenance(args, "k0")
    cs._use_provenance(p)
    out = cs._write(tmp_path / "a.json", {"x": 1})
    d = json.loads(out.read_text())
    assert d["provenance"]["code_hash"] == PR.code_hash(REPO)
    assert d["provenance"]["partial"] == ["--out", "--limit"]
    cs._write_npz(tmp_path / "b.npz", {"y": np.arange(3)})
    assert PR.read_stamp(tmp_path / "b.npz")["files"]["b.npz"]
    monkeypatch.setattr(PR, "code_hash", lambda repo: "other")
    with pytest.raises(SystemExit, match="code changed"):
        cs._write(tmp_path / "c.json", {"x": 2})
    assert not (tmp_path / "c.json").exists()
    monkeypatch.undo()
    assert cs.F.run_once_problems(tmp_path / "a.json")              # exists: refused
    assert cs.F.run_once_problems(tmp_path / "none.json") == []
    # the bound outputs: stamped by this code, passing, not partial
    gates = tmp_path / "root" / "gates"
    gates.mkdir(parents=True)
    cs._use_provenance(cs.Provenance(None, "k0"))
    cs._write(gates / "k0.json", {"summary": {"pass": True}})
    b, probs = PR.bind_outputs(tmp_path / "root", PR.code_hash(REPO))
    assert "gates/k0.json" in b and not any("k0.json" in q for q in probs)
    assert any("k1.json is missing" in q for q in probs)
    cs._write(gates / "k1.json", {"decision": {"pass": False}})
    _, probs = PR.bind_outputs(tmp_path / "root", PR.code_hash(REPO))
    assert any("k1.json does not pass" in q for q in probs)


def test_import_rule_and_record_pieces():
    cs = _script()
    mods = {"ok": SimpleNamespace(__file__=str(REPO / "src/apexzero/costate/gates.py")),
            "script": SimpleNamespace(__file__=str(REPO / "scripts/fields.py")),
            "bad": SimpleNamespace(__file__=str(REPO / "scripts/other.py"))}
    probs = cs.import_problems(REPO, mods)
    assert len(probs) == 1 and "bad" in probs[0]
    c = PR.constants()
    assert c["states"]["interior_k"] == 10 and c["analysis"]["huber_c"] == 1.345
    assert c["analysis"]["c3"][0] == 100_000 and len(c["analysis"]["c3_strata"]) == 31
    assert PR.t_hash([("a", 1), ("b", 0)]) == PR.t_hash([("b", 0), ("a", 1)])
    assert set(cs.POST_TAG).isdisjoint(cs.PRE_TAG)
    assert len(cs.all_networks()) == 34
    # value accuracy's frozen record (Section 3.3): the own policy's return on the own
    # stage's rows, up to 1,000 steps; the strata lists as the analysis reads them
    from apexzero.costate import analysis as AN
    from apexzero.costate import own as OW
    assert cs.OWN_EVERY == OW.RESTART_EVERY == 20 and OW.V_STEPS == 1000
    va = c["analysis"]["value_accuracy"]
    assert (va["r2_min"], va["target"], va["rows"], va["steps"]) == (0.8, "v_own_det", "own", 1000)
    assert va["descriptive"] == {"target": "v_star_inf", "rows": "interior"}
    assert c["analysis"]["strata"] == {h: list(v) for h, v in AN.STRATA.items()}
    assert c["analysis"]["strata_per_network_columns"] == ["kappa_clipped"]
    assert c["own"]["v_steps"] == 1000 and c["own"]["restart_every"] == 20
    assert c["own"]["k3b_steps"] == OW.K3B_STEPS == 160
    # a network that failed K3 after the tag: its analysis is contained (stage_analyse)
    assert "failed K3 after the tag" in c["analysis"]["containment"]
    assert "contained" in c["critic"]["k3_trained"]["fail"]


_SUBPROCESS = r"""
import json, sys
import importlib.util
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from types import SimpleNamespace
from apexzero.agents.ppo import RunningMeanStd
from apexzero.costate import critic as CR, probes as PB, inputs as IN, states as ST
from apexzero.costate import nulls as NL, analysis as AN, prereg as PR
from apexzero.fields import network as NW
spec = importlib.util.spec_from_file_location("costate_script", sys.argv[1])
cs = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(spec.origin.rsplit("/", 1)[0]))
spec.loader.exec_module(cs)
bank, _ = IN.design()
refs = sorted(ST.a2_refs(), key=lambda r: r.key_id)
by = {}
for r in refs:
    st = ST.strata(r)
    if st.w_is_cell:
        by.setdefault((st.R, st.theta_deg), r)
pick = [by[g] for g in sorted(by)][::4][:12]
world = IN.build_world([IN.key_data(r, bank) for r in pick])
rms, cfg = IN.obs_rms_of("G1")
# a random-init stand-in for a trained snapshot: G1's config and obs_rms, random params
snap = SimpleNamespace(actor=NW.random_actor_params(cfg, 3), critic=CR.random_critic_params(cfg, 3),
                       ret_rms=RunningMeanStd(np.zeros(()), np.float32(4.0), np.float32(1.0)))
agent = SimpleNamespace(agent_id="stand-in", timesteps=0, cfg=cfg, obs_rms=rms, snapshot=snap)
PB.unlock_trained(); CR.unlock_trained()
acts = cs.activations_of(agent, world, n_random=PB.N_POOL)
crit = cs.critic_of(agent, world, seeds=(0, 1, 2), bank=bank)
# the width derivative against a central difference of V_c on the tangent-perturbed road
from apexzero.fields import perturb as PT
net = CR.trained_critic(agent)
k0 = world.keys[0]
i0 = int(k0.rows[0])
rd = NW.road(bank, int(world.tab["stim_id"][i0]))
sig = int(world.tab["direction"][i0])
dl = IN.inside_widening(rd.n, sig)
h = 1e-4
def v_at(e):
    tr = PT.perturbed_arrays(rd.base, rd.samples, PT.Samples(*(e * a for a in dl)))
    return CR.values(net, tr, k0.inp.x, k0.inp.prev_action, k0.inp.accel)[:, 0]
fd = net.sigma * (v_at(h) - v_at(-h)) / (2 * h)
ad = crit["dV_dwin"][k0.rows]
width_err = float(np.max(np.abs(ad - fd)) / max(np.max(np.abs(fd)), 1e-12))
fo = PB.folds(world.tab)
w = ST.weights(world.tab, "key")
nl = NL.null_gradients(world.keys, world.tab, world.v_star, fo, w, rms, world.lam,
                       which=("value_ridge", "qss", "barrier", "far_geometry"))
frozen = PR.frozen_values(world.tab, None, partitions=1)
fl = IN.flags(world, rms)
counts = PB.cluster_counts(AN.geometry_clusters(world.tab)[1], 200)
frozen["power_floor"] = {c: True for c in ("s", "n", "xi", "vx")}
# the own stage's value records on its rows (T's interior restarts every 20th step, and D),
# planted: the stand-in policy's own return replaced by V*inf (a known column of real T)
tt = np.asarray(world.tab["t"])
inner = np.asarray(world.tab["interior"], bool)
asked = (inner & (tt % 20 == 0) & (tt > 0)) | (tt == 0)
vt = np.asarray(world.tab["v_star_inf"], float)
lo = world.lam + 0.1 * np.random.default_rng(0).normal(size=world.lam.shape)
own = {"lam_own": lo, "lam_own_held": lo, "valid": np.ones(world.n, bool), "v_rows": asked,
       "v_own_det": np.where(asked, vt, np.nan),
       "v_own_valid": asked, "v_own_terminated": np.zeros(world.n, bool),
       "v_own_strip_end": np.zeros(world.n, bool), "v_own_H": np.where(asked, 1000, 0),
       "v_own_tail": np.where(asked, 0.995 ** 1000, np.nan), "v_steps": np.int64(1000)}
an = cs.analyse_network("stand-in", world, frozen, crit, {k: v for k, v in nl.items()}, fl, counts,
                        own, True, variants=["braking_keys", "pullback:held"],
                        strata=("interior", "interior20", "contact", "interior&braking_key",
                                "interior&kappa_clipped"))
sw = np.asarray(world.tab[ST.SW_STRATUM], bool)
pr = cs.probes_of(acts, world, frozen, np.ones(world.n, bool), sw, variants=["primary"],
                  strata={"C2": ("interior", "contact"), "C2-sw": ("pedal_switching",
                                                                   "pedal_saturated")},
                  kappa_clipped=fl["kappa_clipped"])
# value accuracy on the real columns: the random-init critic above fails; a planted critic
# whose value is the own return plus small noise passes on the same rows and folds
cl, G = AN.geometry_clusters(world.tab)
planted = vt + 0.05 * vt[asked].std() * np.random.default_rng(1).normal(size=world.n)
va_planted = AN.value_accuracy(planted, own["v_own_det"], fo.outer,
                               ST.stratum_weights(world.tab, asked), asked, cl, G, counts)
# every frozen stratum resolves on real T's columns (and this network's kappa flags); C2-sw's
# phase parts split its stratum
tab_s = AN.strata_tab(world.tab, fl)
resolved = {h: AN.strata_counts(tab_s, names) for h, names in AN.STRATA.items()}
sw_parts = sum(resolved["C2-sw"][f"pedal_switching&phase:{p}"] for p in ST.PHASES)
print(json.dumps({"n": world.n, "keys": len(pick),
  "acts": {k: list(v.shape) for k, v in acts.items()},
  "crit_g": list(crit["g"].shape), "random_g": list(crit["random_g"].shape),
  "finite": bool(np.isfinite(crit["g"]).all() and np.isfinite(acts["actor_h3"]).all()),
  "nulls": {k: bool(np.isfinite(v).all()) for k, v in nl.items() if k != "fits"},
  "c1": an["C1"]["decision"], "c1_nulls": sorted(an["C1"]["null_diff"]),
  "c1_limit": an["C1-limit"].get("decision"), "c2": pr["C2"], "c2sw": pr["C2-sw"],
  "sw_rows": int(sw.sum()), "sigma": float(crit["sigma"]),
  "c1_braking": an["C1-braking"]["decision"], "sens": sorted(an["sensitivity"]),
  "width_err": width_err, "width_finite": bool(np.isfinite(crit["dV_dwin"]).all()),
  "pattern": an["C1-pattern"]["decision"], "heads": sorted(an["C1-heads"]),
  "w1": an["W1"]["r2c"], "va": an["value_accuracy"], "va_planted": va_planted,
  "own_status": an["C1-own"]["status"], "strata_c1": {k: [v["n"], v["band"], v["primary"]]
                                                      for k, v in an["by_stratum"]["C1"].items()},
  "strata_c2": sorted(pr["by_stratum"]["C2"]), "strata_sw": sorted(pr["by_stratum"]["C2-sw"]),
  "n_inner": int(inner.sum()), "n_asked": int(asked.sum()), "resolved": resolved,
  "sw_parts": sw_parts, "n_kappa": int(np.asarray(fl["kappa_clipped"]).sum()),
  "va_vstar": an["value_accuracy_vstar"], "held": an["sensitivity"]["pullback:held"],
  "own_held": an["C1-own_held"], "c1_stat": an["C1"]["stat"],
  "decoy_k": frozen["decoy_k"], "pr_decoy_k": pr["decoy_k"]}, default=float))
"""


@pytest.mark.skipif(not (HAVE_KEYS and _have_g1()), reason="step 1's keys or G1 are missing")
def test_post_tag_bodies_end_to_end_on_random_stand_ins(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([sys.executable, "-c", _SUBPROCESS, str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=600, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["keys"] == 12 and out["finite"]
    n = out["n"]
    assert out["acts"]["actor_h3"] == [n, 256] and out["acts"]["heads"] == [n, 5]
    assert out["acts"]["head_units"] == [5, n, 64] and out["acts"]["random4_h3"] == [n, 256]
    assert out["acts"]["random19_h3"] == [n, 256] and "random20_h3" not in out["acts"]
    assert out["crit_g"] == [n, 6, 8] and out["random_g"] == [3, n, 8]
    assert out["sigma"] == pytest.approx(2.0, rel=1e-6)
    assert all(out["nulls"][k] for k in ("value_ridge", "qss", "barrier"))
    assert out["c1_nulls"] == ["barrier", "far_geometry", "qss", "random_critics", "value_ridge"]
    assert out["c1"] in ("refute", "inconclusive")       # a random critic never confirms
    # the stand-in actor is seed 3, which is also random control 3: its full probe is that
    # control's exactly, the gain is raw minus the controls' median, and it never confirms
    for key in ("c2", "c2sw"):
        g = out[key]
        assert g["decision"] not in ("confirm", "not testable"), (key, g["decision"])
        assert all(g["passing"]), key
        assert np.array_equal(g["r2_full"], np.asarray(g["r2_random"])[3]), key
        want = np.subtract(g["r2_full"], g["r2_base"]) - np.median(
            np.subtract(g["r2_random"], g["r2_base"]), axis=0)
        assert np.allclose(g["gain"], want, rtol=0, atol=1e-12), key
        # the primary against the control pool (seeds 0-19) with the two-level CI
        assert len(g["r2_random"]) == 20 and g["controls"]["seeds"] == list(range(20)), key
        assert g["ci_kind"] == "two-level" and g["pool"]["n_tied"] >= 1, key
    assert out["sw_rows"] > 0 and out["sens"] == ["braking_keys", "pullback:held"]
    # the accel-held pullback on the critic stage's own g_held (Section 3.2): scored as C1,
    # every null left out; C1-own's held form beside C1-own (Section 13)
    held = out["held"]
    assert "error" not in held and held["null_diff"] == {} and held["pullback"] == "held"
    assert held["nulls_omitted"] == out["c1_nulls"] and np.isfinite(held["stat"])
    assert "error" not in out["own_held"] and out["own_held"]["descriptive"]
    assert out["c1_braking"] in ("refute", "inconclusive", "not testable")
    assert out["width_finite"] and out["width_err"] < 1e-6
    assert out["heads"] == ["D", "grad_KP_minus_V", "grad_secondary_minus_KP"]
    # value accuracy (Section 13): the random-init stand-in fails, so C1-own is descriptive
    # even with K3b passed; the planted value passes on the same rows and folds
    va, vp = out["va"], out["va_planted"]
    assert not va["pass"] and va["r2"] < 0.8 and va["n"] == out["n_asked"], va
    assert va["target"] == "v_own_det" and va["n_d"] == out["keys"]           # D: t = 0
    assert len(va["heads"]) == 5 and max(va["heads"]) < 0.8
    assert vp["pass"] and vp["r2"] > 0.95 and vp["n_scored"] == out["n_asked"], vp
    assert out["own_status"] == "descriptive (value accuracy)"
    vs = out["va_vstar"]
    assert vs["descriptive"] and not vs["meets_threshold"] and vs["n"] == out["n_inner"]
    # the strata tables on the real columns; every frozen stratum resolves there
    sc = out["strata_c1"]
    assert list(sc) == ["interior", "interior20", "contact", "interior&braking_key",
                        "interior&kappa_clipped"]
    assert sc["interior&kappa_clipped"][0] <= out["n_kappa"]
    from apexzero.costate import analysis as AN
    assert {h: list(v) for h, v in out["resolved"].items()} == {
        h: list(v) for h, v in AN.STRATA.items()}
    assert all(isinstance(c, int) for d in out["resolved"].values() for c in d.values())
    assert out["resolved"]["C1"]["interior"] == out["n_inner"]
    assert out["sw_parts"] == out["resolved"]["C2-sw"]["pedal_switching"] == out["sw_rows"]
    assert sc["interior"][0] == out["n_inner"]
    assert sc["interior"][2] and not any(v[2] for k, v in sc.items() if k != "interior")
    assert all(v[1] in ("confirm", "refute", "inconclusive", "not testable") for v in sc.values())
    assert sc["interior"][1] != "confirm"
    assert out["strata_c2"] == ["contact", "interior"]
    assert out["strata_sw"] == ["pedal_saturated", "pedal_switching"]
    # f_D's one k (states.KNN_K): frozen in the record and read by the post-tag C2
    # (probes_of: 'frozen')
    from apexzero.costate import states as ST
    assert out["decoy_k"]["k"] == ST.KNN_K and "every partition" in out["decoy_k"]["scope"]
    assert out["pr_decoy_k"] == {"k": ST.KNN_K, "source": "frozen"}


_OWN = r"""
import json, sys
import importlib.util
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from apexzero.costate import own as OW, inputs as IN, states as ST
from apexzero.fields import network as NW
from apexzero.interp.constants import S_TRUNC
spec = importlib.util.spec_from_file_location("costate_script", sys.argv[1])
cs = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(spec.origin.rsplit("/", 1)[0]))
spec.loader.exec_module(cs)
bank, cells = IN.design()
# (1) own_return on step 2's linear driver (planted, untrained) from cell 344, which it drives
# through the corner; from 150 m before the strip's end (truncation); off the road (a crash)
st = NW.cell_start(cells[344], 0, bank)
rd = NW.road(bank, st.stim_id)
net = NW.linear_net(*NW.linear_driver(st.sigma))
steps = 300
li = NW.loop_inputs(st, steps)
x0 = np.asarray(st.x0, float)
late = x0.copy(); late[0] = S_TRUNC - 150.0; late[1:3] = 0.0; late[4:7] = 0.0
off = x0.copy(); off[1] = 30.0
x0s = np.stack([x0, late, off]); pas = np.stack([st.prev_action] * 3)
accs = np.stack([st.accel] * 3)
r = OW.own_return(net, rd.base, li, x0s, pas, accs, steps)
# the reference: closed_loop's Trace, J rebuilt from Trace.x, clip(Trace.a) and the previous
# command (Section 3.3, racing.py:174-175)
cl = jax.jit(NW.closed_loop, static_argnums=(0, 5))
ref = []
for i in range(3):
    lii = li._replace(x0=x0s[i], prev_action=pas[i], accel=accs[i], a_forced=np.zeros((steps, 2)),
                      n_forced=np.int32(0), s_stop=np.float64(np.inf))
    T = cl(net.fn, net.params, net.obs_rms, jax.tree.map(jax.numpy.asarray, rd.base),
           jax.tree.map(jax.numpy.asarray, lii), steps)
    x, a, v = np.asarray(T.x), np.asarray(T.a), np.asarray(T.valid)
    H = int(v[1:].sum())
    prev = np.vstack([pas[i][None], np.clip(a[:-1], -1, 1)])
    rr = (x[1:, 0] - x[:-1, 0]) / 10.0 - 0.02 * np.sum((np.clip(a, -1, 1) - prev) ** 2, 1)
    ref.append({"J": float(np.sum(OW.GAMMA ** np.arange(H) * rr[:H])), "H": H,
                "term": bool(np.asarray(T.term).any()), "trunc": bool(np.asarray(T.trunc).any())})
# (2) own_of on a two-key world with the same driver: its records are own_return's on the rows
# it restarts from (T's interior every 20th step, and D: every key's t = 0)
refs = sorted(ST.a2_refs(), key=lambda q: q.key_id)
by = {}
for q in refs:
    s_ = ST.strata(q)
    if s_.w_is_cell:
        by.setdefault((s_.R, s_.theta_deg), q)
world = IN.build_world([IN.key_data(q, bank) for q in [by[g] for g in sorted(by)][::4][:2]])
k0 = world.keys[0]
sig = int(world.tab["direction"][int(k0.rows[0])])
drv = NW.linear_net(*NW.linear_driver(sig))
own = cs.own_of(None, world, net=drv, keys=[k0.key_id], v_steps=200, bank=bank)
tt, inner = np.asarray(world.tab["t"]), np.asarray(world.tab["interior"], bool)
mine = np.isin(np.asarray(world.tab["key_id"]).astype(str), [k0.key_id])
want = ((inner & (tt % 20 == 0) & (tt > 0)) | (tt == 0)) & mine
js = np.flatnonzero(want[k0.rows])
rr2 = OW.own_return(drv, NW.road(bank, int(world.tab["stim_id"][int(k0.rows[0])])).base,
                    li._replace(a_forced=np.zeros((200, 2))), k0.inp.take(js).x,
                    k0.inp.take(js).prev_action, k0.inp.take(js).accel, 200)
idx = np.asarray(k0.rows)[js]
print(json.dumps({"r": {k: np.asarray(v).tolist() for k, v in r.__dict__.items()}, "ref": ref,
  "v_rows_ok": bool(np.array_equal(own["v_rows"], want)),
  "lam_in_v": bool(np.all(want[own["rows"]]) and np.all(tt[own["rows"]] > 0)),
  "same": bool(np.array_equal(own["v_own_det"][idx], rr2.J)
               and np.array_equal(own["v_own_H"][idx], rr2.H)
               and np.array_equal(own["v_own_valid"][idx], rr2.valid)
               and np.array_equal(own["v_own_tail"][idx], rr2.tail)),
  "outside_nan": bool(np.all(np.isnan(own["v_own_det"][~want])) and not own["v_own_valid"][~want].any()),
  "n_d": int((want & (tt == 0)).sum()), "v_steps": int(own["v_steps"]),
  "keys": sorted(own) }))
"""


@pytest.mark.skipif(not HAVE_KEYS, reason="step 1's keys or stimuli are missing")
def test_own_returns_on_a_planted_driver():
    """Section 3.3 'Value accuracy': value accuracy's target is the deterministic policy's own
    discounted return, J over up to 1,000 steps or to the strip's end, gamma^H reported, a
    crashed run recorded and dropped (``own.own_return``). On step 2's linear driver
    (untrained): J, H and the end (strip end, crash) equal closed_loop's Trace with J rebuilt
    in NumPy; own_of's records are own_return's on exactly the rows it restarts from (T's
    interior every 20th step and D)."""
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([sys.executable, "-c", _OWN, str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=600, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    got, ref = out["r"], out["ref"]
    for i in range(3):
        assert got["J"][i] == pytest.approx(ref[i]["J"], rel=1e-12, abs=1e-12), i
        assert got["H"][i] == ref[i]["H"] and got["terminated"][i] == ref[i]["term"], i
        assert got["strip_end"][i] == ref[i]["trunc"], i
        assert got["tail"][i] == pytest.approx(0.995 ** ref[i]["H"], rel=1e-12)
    assert got["valid"] == [True, True, False]                 # the crash is dropped
    assert got["H"][0] == 300 and not got["strip_end"][0]      # the whole loop
    assert got["strip_end"][1] and got["H"][1] < 300           # ... or to the strip's end
    assert got["terminated"][2] and got["H"][2] == 1
    assert got["J"][1] == pytest.approx(got["J_env"][1], rel=1e-12)    # no penalty on a valid run
    assert got["J_env"][2] < got["J"][2]                       # the env's crash penalty
    assert out["v_rows_ok"] and out["lam_in_v"] and out["same"] and out["outside_nan"]
    assert out["n_d"] == 1 and out["v_steps"] == 200
    assert {"v_own_det", "v_own_valid", "v_own_tail", "v_own_H", "v_own_terminated",
            "v_own_strip_end", "v_own_inf", "v_rows", "lam_own", "valid"} <= set(out["keys"])


# ------------------------------------------------------------------------- integrity (review)


def _retag(root: Path, cs, change=None, files: dict | None = None) -> None:
    """Rewrite the scratch repository's record (``change(rec)``), commit it with ``files``,
    and move the annotated tag onto that commit."""
    rec = json.loads((root / PR.PREREG_REL).read_text())
    rec["retagged"] = int(rec.get("retagged", 0)) + 1      # always a new record commit
    if change is not None:
        change(rec)
    PR.save(rec, root / PR.PREREG_REL)
    for rel, text in (files or {}).items():
        (root / rel).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "record again")
    _git(root, "tag", "-d", cs.TAG)
    _git(root, "tag", "-a", cs.TAG, "-m", "tag")


def test_each_freeze_rule_names_its_problem(tmp_path):
    """Review finding: five freeze rules could be deleted with the tests still passing."""
    cs = _script()

    def probs(root, verify):
        return [str(p) for p in cs.freeze_problems(root=root, verify=verify, remote=None,
                                                   pin_path=root.parent / "pin.json")]

    root, _, _, verify = _scratch(tmp_path / "a", cs)
    base = _git(root, "rev-parse", "HEAD~1").strip()
    _git(root, "checkout", "-q", "--detach", base)
    (root / "notes.txt").write_text("elsewhere\n")
    _git(root, "commit", "-q", "-am", "off the tag")
    assert any("not an ancestor of HEAD" in p for p in probs(root, verify))

    root, _, _, verify = _scratch(tmp_path / "b", cs)
    _retag(root, cs, lambda r: r.update(code_hash="0" * 64))
    assert any("not the code the record was built from" in p for p in probs(root, verify))

    root, _, _, verify = _scratch(tmp_path / "c", cs)
    (root / "src/apexzero/costate/a.py").write_text("x = 5\n")
    _git(root, "commit", "-q", "-am", "code after the base")
    _retag(root, cs)
    assert any("differs from the record's base commit" in p for p in probs(root, verify))

    root, _, _, verify = _scratch(tmp_path / "d", cs)
    _retag(root, cs, files={PR.DESIGN_DOC: "design v2.2\n"})
    assert any("design document at the tag" in p for p in probs(root, verify))

    root, _, _, verify = _scratch(tmp_path / "e", cs)
    _retag(root, cs, lambda r: r.update(environment={"python": "0", "packages": {}}))
    assert any("package versions differ" in p for p in probs(root, verify))


def test_step_tags_must_be_the_records(tmp_path, monkeypatch):
    """Review finding: scripts/fields.py runs in every stage but is outside the code hash,
    frozen only by a diff against the local step 2 tag, and the step tag objects the record
    registers were never compared: a moved local tag let changed frozen code through."""
    from apexzero.fields.prereg import waivable

    cs = _script()

    def probs(root):
        return cs.freeze_problems(root=root, verify=None, remote=None,
                                  pin_path=root.parent / "pin.json")

    # step 2's tag moved onto a commit that changes scripts/fields.py (not a code path)
    root, _, _, _ = _scratch(tmp_path / "a", cs)
    assert probs(root) == []
    (root / "scripts/fields.py").write_text("# changed\n")
    _git(root, "commit", "-q", "-am", "fields.py changed")
    _git(root, "tag", "-f", "-a", cs.STEP2_TAG, "-m", "moved")
    p = probs(root)
    assert [q.code for q in p] == ["frozen"] and "step2" in p[0] and not waivable(p)
    # step 1's tag re-made (a new tag object on unchanged frozen code)
    root, _, _, _ = _scratch(tmp_path / "b", cs)
    _git(root, "tag", "-f", "-a", cs.STEP1_TAG, "-m", "moved")
    p = probs(root)
    assert [q.code for q in p] == ["frozen"] and "step1" in p[0]
    # a record without the step tags
    root, _, _, _ = _scratch(tmp_path / "c", cs)
    _retag(root, cs, lambda r: r.pop("steps"))
    assert sum("registers no object" in q for q in probs(root)) == 2
    # in this repository step 1's tag must also be its pinned object
    root, _, _, _ = _scratch(tmp_path / "d", cs)
    monkeypatch.setattr(cs, "ROOT", root)
    assert any(q.code == "frozen" and "pinned object" in q for q in probs(root))
    monkeypatch.setattr(cs.F, "STEP1_TAG_OBJECT",
                        _git(root, "rev-parse", f"refs/tags/{cs.STEP1_TAG}").strip())
    assert probs(root) == []


def test_pre_tag_stages_refuse_through_main_once_the_tag_exists(tmp_path, monkeypatch):
    """Review finding: deleting the pre-tag refusal from main went unnoticed."""
    cs = _script()
    monkeypatch.setattr(cs.F, "_tag_object", lambda *a, **k: ("0" * 40, "tag"))
    for stage in cs.PRE_TAG:
        monkeypatch.setattr(cs, f"stage_{stage.replace('-', '_')}", lambda args: (
            _ for _ in ()).throw(AssertionError("the stage ran")))
        with pytest.raises(SystemExit, match="after the tag"):
            cs.main([stage, "--out", str(tmp_path / "o")])


def test_confirmatory_run_once_is_checked_before_any_work(monkeypatch):
    """Review finding: the run-once check in stage_analyse could be deleted unnoticed. The
    stubs keep this process locked and write nothing under outputs/costate."""
    cs = _script()

    def boom(*a, **k):
        raise AssertionError("the analysis started")

    monkeypatch.setattr(cs, "_post_tag", lambda a, w: {"frozen": {}})
    monkeypatch.setattr(cs.F, "run_once_problems", lambda p, *a, **k: [f"{p} exists"])
    monkeypatch.setattr(cs, "_load_world", boom)
    monkeypatch.setattr(cs, "_load_world_frozen", boom)
    with pytest.raises(SystemExit, match="confirmatory analysis again"):
        cs.main(["analyse", "--confirmatory"])
    # incomplete inputs: refused before any computation or write
    monkeypatch.setattr(cs.F, "run_once_problems", lambda p, *a, **k: [])
    monkeypatch.setattr(cs, "confirmatory_input_problems",
                        lambda rec, root, prov=None: ["post/critic/x.npz is missing"])
    with pytest.raises(SystemExit, match="on complete inputs"):
        cs.main(["analyse", "--confirmatory"])


def test_nulls_are_named_as_the_analysis_reads_them():
    """Review finding: stage_nulls wrote nulls/G1.npz, the analysis read
    nulls/runs_dr-pilot-200M.npz, and a full --network run was always partial."""
    cs = _script()
    from apexzero.costate import analysis as AN

    assert PR.net_file("G1") == PR.net_file("runs/dr-pilot-200M") == "runs_dr-pilot-200M"
    stems = [p.stem for p in cs.analysis_paths(Path("/x"))["nulls"]]
    assert stems == [PR.net_file(n) for n in PR.null_networks()] and len(stems) == 9
    full = cs.build_parser().parse_args(
        ["nulls", *[a for n in AN.DECISION_NETWORKS for a in ("--network", n)]])
    assert cs._partial(full) == []
    assert cs._partial(cs.build_parser().parse_args(["nulls", "--network", "G1"])) == \
        ["--network"]
    assert cs._partial(cs.build_parser().parse_args(["inputs", "--network", "G1"])) == \
        ["--network"]
    from apexzero.costate import states as ST

    # every C1 null in its layout: the fold-dependent ones per held-out fold (review finding
    # on the fold-dependent nulls), QSS and the barrier [n, 8]
    shapes = {k: ([ST.N_FOLDS, 50, 8] if k in AN.FOLD_NULLS else [50, 8])
              for k in AN.NULL_NAMES if k != "random_critics"}
    good = {"networks": {n: {"finite": {"qss": True}, "sha256": "x", "shapes": shapes}
                         for n in PR.null_networks()}, "failed": []}
    assert PR._passes("nulls/summary.json", good)
    assert not PR._passes("nulls/summary.json", {"networks": {"G1": good["networks"][
        "runs/dr-pilot-200M"]}})
    for bad in ({"sha256": None}, {"finite": {"qss": False}}, {"shapes": None},
                {"shapes": {**shapes, "far_geometry": [50, 8]}},
                {"shapes": {**shapes, "value_mlp": [3, 50, 8]}},
                {"shapes": {**shapes, "qss": [ST.N_FOLDS, 50, 8]}},
                {"shapes": {**shapes, "barrier": [49, 8]}},
                {"shapes": {k: v for k, v in shapes.items() if k != "value_ridge_window"}}):
        d = json.loads(json.dumps(good))
        d["networks"]["runs/dr-pilot-200M"].update(bad)
        assert not PR._passes("nulls/summary.json", d), bad
    # K1 binds on its planted bugs too, inputs only with K2's t_H
    assert not PR._passes("gates/k1.json", {"decision": {"pass": True, "plants_fail": False,
                                                         "pass_with_plants": False}})
    assert PR._passes("gates/k1.json", {"decision": {"pass": True, "pass_with_plants": True}})
    assert not PR._passes("inputs/summary.json", {"n": 3, "failed": [], "t_H": None})


def test_stage_nulls_writes_the_per_fold_layout_prereg_binds(tmp_path, monkeypatch):
    """Review finding (fold-dependent C1 nulls): stage_nulls writes the value fits and the
    far-geometry lambda per held-out fold [F, n, 8] and QSS and the barrier [n, 8], and its
    summary (shapes, finiteness, the far null's gaps recorded) passes prereg's check (the
    null computation stubbed: no obs_rms is read)."""
    cs = _script()
    from apexzero.costate import analysis as AN
    from apexzero.costate import inputs as IN
    from apexzero.costate import nulls as NL
    from apexzero.costate import states as ST

    world, _ = _toy_world()
    n = world.n
    w = SimpleNamespace(tab=world.tab, n=n, keys=[], v_star=np.zeros(n), v_star_inf=np.zeros(n),
                        lam=np.zeros((n, 8)))
    monkeypatch.setattr(cs, "_load_world", lambda args: w)
    monkeypatch.setattr(IN, "obs_rms_of", lambda name: (None, None))

    def nulls(keys, tab, v_star, fo, w_, rms, lam, **kw):
        out = {"fits": {}}
        for k in NL.NULLS:
            out[k] = np.zeros((fo.n_folds, n, 8) if k in AN.FOLD_NULLS else (n, 8))
        out["far_geometry"][fo.n_folds - 1 - fo.outer[0], 0] = np.nan   # a training row's gap
        return out

    monkeypatch.setattr(NL, "null_gradients", nulls)
    args = cs.build_parser().parse_args(["nulls", "--out", str(tmp_path)])
    cs._use_provenance(cs.Provenance(args, "nulls"))
    cs.stage_nulls(args)
    summ = json.loads((tmp_path / "nulls" / "summary.json").read_text())
    v = summ["networks"]["runs/dr-pilot-200M"]
    assert v["shapes"]["far_geometry"] == [ST.N_FOLDS, n, 8] and v["shapes"]["qss"] == [n, 8]
    assert v["far_geometry_nonfinite"] == {"slices": 1, "out_of_fold": 0}
    assert PR._passes("nulls/summary.json", summ) and not PR._cross_problems(tmp_path)
    with np.load(tmp_path / "nulls" / "runs_dr-pilot-200M.npz") as d:
        assert d["value_mlp"].shape == (ST.N_FOLDS, n, 8) and d["barrier"].shape == (n, 8)


def test_bound_arrays_and_cross_checks(tmp_path):
    """Review finding: world.npz and the nulls arrays were not bound; t_H was not tied to
    K2's."""
    cs = _script()
    root = tmp_path / "root"
    cs._use_provenance(cs.Provenance(None, "inputs"))
    cs._write_npz(root / "inputs" / "world.npz", {"a": np.arange(3)})
    for n in PR.null_networks():
        cs._write_npz(root / "nulls" / f"{PR.net_file(n)}.npz", {"qss": np.ones((2, 8))})
    code = PR.code_hash(REPO)
    arrays, probs = PR.bind_arrays(root, code)
    assert not probs and len(arrays) == 10
    np.savez(root / "nulls" / "runs_dr-pilot-200M.npz", qss=np.zeros((2, 8)))
    _, probs = PR.bind_arrays(root, code)
    assert any("changed after it was stamped" in p for p in probs)
    cs._write(root / "inputs" / "summary.json", {"n": 3, "failed": [], "t_H": 267})
    cs._write(root / "gates" / "k2.json", {"horizon": {"t_H": 260}, "pass": True})
    cs._write(root / "nulls" / "summary.json",
              {"networks": {n: {"sha256": "0" * 64, "finite": {"qss": True}}
                            for n in PR.null_networks()}, "failed": []})
    _, probs = PR.bind_outputs(root, code)
    assert any("is not K2's" in p for p in probs)
    assert any("is not the array its summary hashed" in p for p in probs)


def test_options_that_change_a_bound_output_are_partial():
    """Review finding: k0 without N's finals (its old default), k4 --which, k5 --ceiling
    ridge, k3b --steps, the nulls' MLP options and inputs --partitions wrote stamped,
    non-partial outputs that prereg would bind; and a --n, --limit or --n-extend of 0 (a
    vacuous K2b passes on no key) read as not given."""
    from apexzero.costate import own as OW
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST

    cs = _script()
    ap = cs.build_parser()

    def part(*argv):
        return cs._partial(ap.parse_args(list(argv)))

    for stage in ("k0", "k3b", "k4", "k5", "nulls", "inputs"):
        assert part(stage) == [], stage
    assert ap.parse_args(["k0"]).kinds == ["a2", "n"]
    assert part("k0", "--kinds", "a2") == ["--kinds"]
    assert part("k0", "--kinds", "n", "a2") == []
    assert part("k4", "--which", "PR-2") == ["--which"]
    assert part("k4", "--which") == []                    # no plant named: every plant runs
    assert part("k5", "--ceiling", "ridge") == ["--ceiling"]
    assert part("nulls", "--mlp-alphas", "1e-3") == ["--mlp-alphas"]
    assert part("nulls", "--mlp-alphas", *map(str, PB.MLP_ALPHAS)) == []
    assert part("nulls", "--mlp-maxiter", "50") == ["--mlp-maxiter"]
    assert part("inputs", "--partitions", "2") == ["--partitions"]
    assert part("inputs", "--partitions", str(ST.N_PARTITIONS)) == []
    assert part("k0", "--kinds", "a2", "--limit", "3") == ["--limit", "--kinds"]
    assert OW.K3B_STEPS == 160 and ap.parse_args(["k3b"]).steps is None
    assert part("k3b", "--steps", "20") == ["--steps"]
    assert part("k3b", "--steps", str(OW.K3B_STEPS)) == []
    for argv, opt in ((("k2b", "--n", "0"), "--n"), (("k3", "--n", "0"), "--n"),
                      (("k2", "--n-extend", "0"), "--n-extend"),
                      (("k0", "--limit", "0"), "--limit"),
                      (("inputs", "--limit", "0"), "--limit")):
        assert part(*argv) == [opt], argv


def test_k4_reads_c2s_setup_and_prereg_freezes_its_band_notes(tmp_path):
    """K4's PR-6b and PR-6c score C2's primary statistic, so k4 (every plant, or PR-6)
    refuses before inputs and k2b have run; prereg copies K4's band-edge notes from the bound
    gates/k4.json into the frozen values (the analysis reads them there)."""
    cs = _script()
    args = SimpleNamespace(out=str(tmp_path), which=None)
    with pytest.raises(SystemExit, match="run inputs and k2b first"):
        cs._probe_world(args, c2_setup=True)
    with pytest.raises(SystemExit, match="no .*world.npz"):           # PR-2 alone: no setup
        cs._probe_world(args, c2_setup=False)
    (tmp_path / "gates").mkdir()
    (tmp_path / "gates" / "k2b.json").write_text(json.dumps({"rule": {"excluded": {}}}))
    (tmp_path / "gates" / "k2.json").write_text(json.dumps({"horizon": {"t_H": 100}}))
    (tmp_path / "gates" / "k4.json").write_text(json.dumps({
        "PR-6": {"overfit_cost": 0.011, "delta2": -0.011, "ci": [-0.03, 0.0]},
        "PR-6c": {"leak_bound": 0.008, "gain_max": [0.03, None, 0.001, None],
                  "seeds": [{}, {}, {}]}}))
    with pytest.raises(FileNotFoundError, match="c2_null.json"):   # the null is bound too
        cs.prereg_frozen(tmp_path, {"power_floor": {"s": True}})
    (tmp_path / "gates" / "c2_null.json").write_text(json.dumps({
        "seeds": [5, 6], "controls": [0, 1, 2, 3, 4], "rows": 10, "obs_rms": "G1",
        "passing": {"s": True}, "decoy_k": 8, "summary": {"n": 2, "mean": -0.01},
        "networks": [{"seed": 5, "stat": -0.02, "ci": [-0.1, 0.0], "decision": "refute"},
                     {"seed": 6, "stat": 0.0, "ci": [-0.05, 0.05],
                      "decision": "inconclusive"}]}))
    fr = cs.prereg_frozen(tmp_path, {"power_floor": {"s": True}})
    assert fr["power_floor"] == {"s": True} and fr["k2b_excluded"] == {}
    assert fr["k4_band_notes"]["pr6_cost"]["cost"] == 0.011
    assert fr["k4_band_notes"]["pr6c_leak"]["bound"] == 0.008
    assert fr["k4_band_notes"]["pr6c_leak"]["seeds"] == 3
    assert fr["c2_null"]["stats"] == [-0.02, 0.0] and fr["c2_null"]["summary"]["n"] == 2
    assert fr["c2_null"]["decisions"] == ["refute", "inconclusive"]
    # decoy-k and c2-null score C2's primary statistic too: they need C2's setup
    for stage in ("decoy-k", "c2-null"):
        with pytest.raises(SystemExit, match=f"{stage} scores C2's primary statistic.*run "
                                             "inputs and k2b first"):
            cs._probe_world(SimpleNamespace(out=str(tmp_path / "none"), stage=stage),
                            c2_setup=True)
    assert cs.PRE_TAG.index("k4") < cs.PRE_TAG.index("decoy-k") < cs.PRE_TAG.index("prereg")
    assert cs.PRE_TAG.index("c2-null") < cs.PRE_TAG.index("prereg")
    assert {"gates/decoy_k.json", "gates/c2_null.json"} <= set(PR.BOUND_OUTPUTS)


def test_probe_world_outputs_record_and_bind_their_inputs(tmp_path, monkeypatch):
    """Review finding (stale inputs): K4, decoy-k and c2-null read C2's decision mask from
    gates/k2b.json and the power floor from inputs/summary.json, and K5 reads the world, so
    each records what it read ('inputs': the world's sha256 and, for those that score C2's
    primary statistic, the floor, f_D's k and K2b's exclusions' hash), and prereg's cross
    check refuses an output computed on other inputs than the bound ones (a K4 on a partial
    K2b's mask, a world rebuilt after K5) or one that records none."""
    cs = _script()
    from apexzero.costate import gates as GT
    from apexzero.costate import inputs as IN
    from apexzero.costate import states as ST

    root = tmp_path
    (root / "inputs").mkdir()
    (root / "gates").mkdir()
    (root / "inputs" / "world.npz").write_bytes(b"world-1")
    floor = {"s": True, "n": False, "xi": True, "vx": False}
    (root / "inputs" / "summary.json").write_text(json.dumps(
        {"frozen": {"power_floor": floor, "decoy_k": {"k": 8}}}))
    exc = {"approach": ["n", "vx"], "braking": ["s"]}
    (root / "gates" / "k2b.json").write_text(json.dumps({"rule": {"excluded": exc}}))
    seen = {}
    monkeypatch.setattr(cs, "_load_world", lambda args: SimpleNamespace(tab={"t": [0]}))
    monkeypatch.setattr(IN, "obs_rms_of", lambda name: (None, "cfg"))
    monkeypatch.setattr(IN, "z_and_jz", lambda world, rms: (np.zeros((1, 2)), None, None))
    monkeypatch.setattr(IN, "flags", lambda world, rms: {"observability": SimpleNamespace(
        s_unobservable=np.zeros(1, bool))})
    monkeypatch.setattr(ST, "decision_mask", lambda tab, obs, e: seen.setdefault("exc", e))
    monkeypatch.setattr(GT, "probe_world", lambda *a, **k: ("pw", k.get("power_floor")))
    args = SimpleNamespace(out=str(root), stage="k4")
    pw, cfg, ins = cs._probe_world(args, c2_setup=True)
    assert pw == ("pw", floor) and cfg == "cfg" and seen["exc"] == exc
    assert ins == PR.probe_world_inputs(root)
    assert ins["decoy_k"] == 8 and ins["power_floor"] == floor
    _, _, ins5 = cs._probe_world(SimpleNamespace(out=str(root), stage="k5"))
    assert ins5 == {"world_sha256": ins["world_sha256"]}
    for rel in ("gates/k4.json", "gates/decoy_k.json", "gates/c2_null.json"):
        (root / rel).write_text(json.dumps({"pass": True, "inputs": ins}))
    (root / "gates" / "k5.json").write_text(json.dumps({"pass": True, "inputs": ins5}))
    assert PR.probe_world_problems(root) == [] and PR._cross_problems(root) == []
    # K2b re-run in full after K4 ran on a partial K2b's mask: K4, decoy-k, c2-null are stale
    (root / "gates" / "k2b.json").write_text(json.dumps(
        {"rule": {"excluded": {**exc, "exit": ["n"]}}}))
    probs = PR.probe_world_problems(root)
    assert len(probs) == 3 and all("k2b_excluded_sha256" in p for p in probs)
    assert any(p.endswith("rerun k4") for p in probs)
    assert any(p.endswith("rerun decoy-k") for p in probs)
    (root / "gates" / "k2b.json").write_text(json.dumps({"rule": {"excluded": exc}}))
    # the world rebuilt after the gates: every probe-world output, K5 included
    (root / "inputs" / "world.npz").write_bytes(b"world-2")
    probs = PR.probe_world_problems(root)
    assert len(probs) == 4 and any(p.endswith("rerun k5") for p in probs)
    (root / "inputs" / "world.npz").write_bytes(b"world-1")
    # another floor: the C2-setup outputs only; an output that records no inputs
    (root / "inputs" / "summary.json").write_text(json.dumps(
        {"frozen": {"power_floor": {**floor, "n": True}, "decoy_k": {"k": 8}}}))
    assert len(PR.probe_world_problems(root)) == 3
    (root / "inputs" / "summary.json").write_text(json.dumps(
        {"frozen": {"power_floor": floor, "decoy_k": {"k": 8}}}))
    (root / "gates" / "k4.json").write_text(json.dumps({"pass": True}))
    assert PR.probe_world_problems(root) == ["gates/k4.json records no inputs: rerun k4"]


def test_confirmatory_nulls_against_the_record_and_prereg_against_this_environment(
        tmp_path, monkeypatch):
    """Review finding: the confirmatory check held the pre-tag nulls to this run's code hash
    (a dead end under a logged code deviation, since pre-tag stages cannot rewrite them after
    the tag), and a package change between a pre-tag stage and prereg was refused only at
    the last post-tag stage."""
    cs = _script()
    root = tmp_path / "root"
    cs._use_provenance(cs.Provenance(None, "nulls"))
    for n in PR.null_networks():
        cs._write_npz(root / "nulls" / f"{PR.net_file(n)}.npz", {"qss": np.ones((2, 8))})
    code = PR.code_hash(REPO)
    rec = {"code_hash": code, "environment": cs.F.environment(),
           "outputs": {f"nulls/{p.name}": {"sha256": PR.sha256_file(p)}
                       for p in (root / "nulls").glob("*.npz")}}

    def stamp_probs(r, prov):
        return [p for p in cs.confirmatory_input_problems(r, root, prov)
                if "nulls/" in p or "other code" in p or "package versions" in p]

    dev = SimpleNamespace(code_hash="0" * 64, deviation={"reason": "a logged fix"}, tag=None)
    assert stamp_probs(rec, dev) == []
    assert stamp_probs(rec, SimpleNamespace(code_hash=code, deviation=None, tag=None)) == []
    assert any("written by other code" in p for p in stamp_probs({**rec, "code_hash": "1" * 64},
                                                                 dev))
    # prereg: a bound output stamped under other package versions
    p = cs.Provenance(None, "k0")
    p.environment = {**p.environment, "packages": {**p.environment["packages"], "jax": "0.0.1"}}
    cs._use_provenance(p)
    cs._write(root / "gates" / "k0.json", {"summary": {"pass": True}})
    rels = ["gates/k0.json", *(f"nulls/{q.name}" for q in (root / "nulls").glob("*.npz"))]
    probs = cs.bound_env_problems(root, rels, cs.F.environment())
    assert len(probs) == 1 and "gates/k0.json" in probs[0] and "jax" in probs[0]
    # ... is refused by stage_prereg itself (every other check stubbed; nothing is written)

    class Reached(Exception):
        pass

    def never(*a, **k):
        raise Reached
    for name, value in (("dirty_paths", lambda *a, **k: []),
                        ("frozen_steps_problems", lambda *a, **k: []),
                        ("_load_world", lambda a: SimpleNamespace(
                            tab={"key_id": np.array([]), "t": np.array([])})),
                        ("prereg_frozen", lambda o, f: {})):
        monkeypatch.setattr(cs, name, value)
    monkeypatch.setattr(PR, "tree_code_hash", lambda repo, rev: code)
    monkeypatch.setattr(PR, "bind_outputs", lambda o, c: ({"gates/k0.json": {}}, []))
    monkeypatch.setattr(PR, "bind_arrays", lambda o, c: ({}, []))
    for name in ("network_pins", "snapshot_pins", "c3w_inputs"):
        monkeypatch.setattr(PR, name, lambda *a, **k: ({}, []))
    monkeypatch.setattr(PR, "stimulus_design_hash", lambda: "d")
    monkeypatch.setattr(PR, "key_sets", lambda *a: {})
    monkeypatch.setattr(PR, "build_record", never)
    monkeypatch.setattr(PR, "save", never)
    cs._write(root / "inputs" / "summary.json", {"frozen": {}})
    args = SimpleNamespace(out=str(root))
    with pytest.raises(SystemExit, match="other package versions"):
        cs.stage_prereg(args)
    cs._use_provenance(cs.Provenance(None, "k0"))
    cs._write(root / "gates" / "k0.json", {"summary": {"pass": True}})
    with pytest.raises(Reached):                       # this environment's stamp passes
        cs.stage_prereg(args)


def test_k3_judges_only_this_runs_reports(tmp_path, monkeypatch):
    """Review finding: K3 accepted exit 1 from a crashed dry run (Python's status for an
    uncaught exception too) and judged the reports a previous run left on disk."""
    from apexzero.costate import gates as GT

    cs = _script()
    out = tmp_path / "o"
    d = out / "gates" / "k3"
    args = cs.build_parser().parse_args(["k3", "--out", str(out)])
    cs._use_provenance(cs.Provenance(args, "k3"))
    monkeypatch.setattr(GT, "k3_judge", lambda *a, **k: {"pass": True})
    real = subprocess.run

    def run_with(**beh):
        """subprocess.run with each K3 subprocess's (exit status, report or None)."""
        def run(cmd, *a, **k):
            if cmd[0] == "git":
                return real(cmd, *a, **k)
            argv = cmd[3:]
            dry = argv[0] == "dry-run"
            name = argv[1] if dry else f"extractor_{argv[argv.index('--heads') + 1]}"
            rc, rep = beh.get(name, (0, {"fresh": name} if dry else {"pass": True,
                                                                      "fresh": name}))
            target = Path(argv[argv.index("--out") + 1])
            if rep is not None:
                (target.with_suffix(".json") if dry else target).write_text(json.dumps(rep))
            return SimpleNamespace(returncode=rc, stdout="", stderr="Traceback: boom")
        return run

    def stale():
        d.mkdir(parents=True, exist_ok=True)
        for n in ("plants", "random", "extractor_5", "extractor_0"):
            (d / f"{n}.json").write_text(json.dumps({"pass": True, "stale": n}))

    for beh, match in (({"random": (1, None)}, "exited 1"),
                       ({"plants": (1, None)}, "exited 1"),
                       ({"random": (1, {"fresh": "random"})}, "exited 1"),
                       ({"plants": (0, None)}, "without writing plants.json"),
                       ({"extractor_0": (1, None)}, "without writing"),
                       ({"extractor_5": (1, {"pass": True})}, "a crash"),
                       ({"extractor_5": (0, {"pass": False})}, "a crash")):
        stale()
        monkeypatch.setattr(subprocess, "run", run_with(**beh))
        with pytest.raises(SystemExit, match=match):
            cs.stage_k3(args)
        assert not (out / "gates" / "k3.json").exists(), beh
        crashed = next(iter(beh))
        if beh[crashed][1] is None:                   # its stale report was removed first
            assert not (d / f"{crashed}.json").exists(), beh
    # an extractor's fail (exit 1 with its rewritten failing report) is a verdict, not a crash
    stale()
    monkeypatch.setattr(subprocess, "run", run_with(extractor_5=(1, {"pass": False,
                                                                    "fresh": "e5"})))
    cs.stage_k3(args)
    res = json.loads((out / "gates" / "k3.json").read_text())
    assert res["returncodes"] == {"plants": 0, "random": 0, "extractor_5": 1, "extractor_0": 0}
    assert res["reports"]["plants"] == {"fresh": "plants"}
    assert res["reports"]["random"] == {"fresh": "random"}
    assert [e["fresh"] for e in res["reports"]["extractor"]] == ["e5", "extractor_0"]


@pytest.mark.skipif(not HAVE_KEYS, reason="step 1's stimuli are not on disk")
def test_stimulus_design_hash_on_step_1s_stimuli():
    """Review finding (blocker): stimulus_design_hash handed keys.design's dict of cells to
    stimuli.design_hash, a TypeError in prereg and in every post-tag freeze check."""
    import re

    from apexzero.fields import prereg as FP

    h = PR.stimulus_design_hash()
    assert isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)
    step2 = json.loads(FP.PREREG_PATH.read_text()) if FP.PREREG_PATH.exists() else {}
    if step2.get("design_hash"):                     # step 2 hashed the same design
        assert h == step2["design_hash"]


def test_the_record_lists_the_deferred_analyses(tmp_path):
    """Review finding: the record kept C3-int's pairs and C1-dev's snapshots unlabelled, and
    the analysis module called the deferred analyses pre-registered (Section 13 'Deferred')."""
    from apexzero.costate import analysis as AN

    cs = _script()
    names = [n for n, _, _ in PR.DEFERRED]
    assert {"C3-int", "C1-dev", "DeltaC1", "C2-pairs"} <= set(names)
    assert any(n.startswith("C1-own") for n in names)
    # the named secondaries the analysis does not compute (review finding: absent and not
    # deferred): N's replication, W1's Pi probes and own-policy width price, C2 and C2-sw
    # beyond the decision networks
    assert {"N's replication", "W1 on the Pi targets", "W1's own-policy width price",
            "C2 and C2-sw beyond the decision networks"} <= set(names)
    c3r = PR.constants()["analysis"]["c3_r2c"]
    assert c3r["stream"] == ["costate_c3", AN.C3_R2C_SUBSTREAM] != ["costate_c3", 1]
    c = PR.constants()
    for item in (i for d in PR.deferred() for i in d["kept_not_preregistered"]):
        head, *path = item.split(".")
        if head == "constants":                      # each kept constant is in the record
            v = c
            for k in path:
                v = v[k]
    root, _, _, _ = _scratch(tmp_path, cs)
    assert cs.tagged_record(root=root)["deferred"] == PR.deferred()
    pre = AN.__doc__.split("**C1**")[0].split("are pre-registered secondaries")[0]
    assert not any(n in pre for n in ("C3-int", "C1-dev", "Delta C1", "C2-pairs"))


def test_verify_record_binds_outputs_checkpoints_and_c3w(tmp_path):
    cs = _script()
    root, keys, sets, _ = _scratch(tmp_path, cs)
    rec = cs.tagged_record(root=root)
    out = tmp_path / "out"
    (out / "gates").mkdir(parents=True)
    (out / "gates" / "k0.json").write_text("{}")
    ck = root / "ck.pkl"
    ck.write_bytes(b"weights")
    rec["outputs"] = {"gates/k0.json": {"sha256": PR.sha256_file(out / "gates" / "k0.json")}}
    rec["networks"] = {"n": {"path": "ck.pkl", "sha256": PR.sha256_file(ck), "timesteps": 1}}
    (root / "runs" / "x").mkdir(parents=True)
    for f in ("eval_rdev.json", "metrics.csv"):
        (root / "runs" / "x" / f).write_text("0\n")
    c3w, p = PR.c3w_inputs(["runs/x"], root, lambda n: 0.5, lambda n: 1.0)
    assert not p and c3w["runs/x"]["completion"] == 0.5
    rec["c3w_inputs"] = c3w
    rec["design_hash"] = "d"

    def v(r):
        return PR.verify_record(r, root, keys, None, out, lambda pins: PR.pins_problems(pins, root),
                                design_hash="d")

    assert v(rec) == []
    (out / "gates" / "k0.json").write_text('{"x": 1}')
    assert any(q.code == "outputs" for q in v(rec))
    assert any(q.code == "record" for q in v({**rec, "outputs": {}}))
    (out / "gates" / "k0.json").write_text("{}")
    ck.write_bytes(b"other")
    assert any(q.code == "checkpoints" for q in v(rec))
    assert any(q.code == "record" for q in v({**rec, "networks": {}}))
    ck.write_bytes(b"weights")
    (root / "runs" / "x" / "metrics.csv").write_text("1\n")
    assert any(q.code == "c3w" for q in v(rec))
    (root / "runs" / "x" / "metrics.csv").write_text("0\n")
    assert any(q.code == "design" for q in PR.verify_record(rec, root, keys, None, out, None,
                                                           design_hash="other"))


def test_network_pins_survive_a_symlinked_runs_directory(tmp_path, monkeypatch):
    """Review finding: in a worktree whose runs/ links to the main checkout, step 2's
    network_pin resolved the link and crashed with ValueError."""
    from apexzero.fields import stages as S

    repo, real = tmp_path / "repo", tmp_path / "real_runs"
    (real / "x" / "ckpt").mkdir(parents=True)
    (real / "x" / "ckpt" / "000000000001.pkl").write_bytes(b"ckpt")
    repo.mkdir()
    (repo / "runs").symlink_to(real)
    monkeypatch.setattr(S, "checkpoint_path",
                        lambda n: repo / "runs" / "x" / "ckpt" / "000000000001.pkl")
    monkeypatch.setattr(agents, "load_agent", lambda path, aid: SimpleNamespace(timesteps=7))
    pins, probs = PR.network_pins(["runs/x"], repo)
    assert not probs and pins["runs/x"]["path"] == "runs/x/ckpt/000000000001.pkl"
    assert pins["runs/x"]["timesteps"] == 7
    monkeypatch.setattr(S, "checkpoint_path", lambda n: tmp_path / "elsewhere.pkl")
    pins, probs = PR.network_pins(["runs/x"], repo)
    assert not pins and probs and "not under the repository" in probs[0]
    snaps, probs = PR.snapshot_pins("runs/x", repo)
    assert list(snaps) == ["runs/x@7"] and snaps["runs/x@7"]["path"].startswith("runs/x/")


def test_the_overlap_file_is_repo_relative_through_a_linked_outputs_directory(tmp_path,
                                                                              monkeypatch):
    """Post-tag rehearsal finding: through a checkout whose step 2 outputs are reached by the
    outputs/outputs link, ``overlap`` resolved the link and wrote the rows file's absolute
    local path into the record (C3-w's 'overlap_file'); it is repo-relative, as the pins."""
    from apexzero.fields import stages as S

    cs = _script()
    real = tmp_path / "main_outputs" / "fields"
    (real / "fields-a2").mkdir(parents=True)
    (real / "opt-fields").mkdir()
    (real / "fields-a2" / "runs_x.jsonl").write_text("{}\n")
    (real / "opt-fields" / "rows.jsonl").write_text("{}\n")
    repo = tmp_path / "repo"
    (repo / "outputs").mkdir(parents=True)
    (repo / "outputs" / "outputs").symlink_to(tmp_path / "main_outputs")
    monkeypatch.setattr(S, "read_rows", lambda p: [])
    monkeypatch.setattr(S, "edge_key_measures", lambda rows, opt, m: {})
    monkeypatch.setattr(S, "median_statistic", lambda d, n=1: 0.5)
    b, rel, sha = cs.overlap("runs/x", repo)
    assert (b, rel) == (0.5, "outputs/outputs/fields/fields-a2/runs_x.jsonl") and len(sha) == 64


def test_load_agent_checks_the_pin(tmp_path, monkeypatch):
    cs = _script()
    (tmp_path / "ck.pkl").write_bytes(b"w")
    pin = {"path": "ck.pkl", "sha256": PR.sha256_file(tmp_path / "ck.pkl"), "timesteps": 6}
    with pytest.raises(SystemExit, match="not the recorded checkpoint"):
        cs.load_agent("G1", {**pin, "sha256": "0" * 64}, root=tmp_path)
    monkeypatch.setattr(agents, "load_agent", lambda path, aid: SimpleNamespace(timesteps=5))
    with pytest.raises(SystemExit, match="timesteps 5"):
        cs.load_agent("G1", pin, root=tmp_path)


def test_trained_networks_are_refused_everywhere_before_the_tag():
    """Review finding: the trained marks of agent_nets, the own-costate loop, K3b and the
    width derivative were untested."""
    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN
    from apexzero.costate import own as OW
    from apexzero.costate import probes as PB
    from apexzero.fields import network as NW

    cfg = PB.g1_config()
    snap = SimpleNamespace(actor=NW.random_actor_params(cfg, 0),
                           critic=CR.random_critic_params(cfg, 0))
    actor, critic = PB.agent_nets(SimpleNamespace(agent_id="x", timesteps=1, cfg=cfg,
                                                  snapshot=snap))
    assert actor.trained and critic.trained
    with pytest.raises(PermissionError):
        PB.hidden(actor, np.zeros((2, 222)))
    with pytest.raises(PermissionError):           # before the Critic.apply gradient
        PB._grad_record(critic, None)
    pol = NW.NetPolicy(lambda *a: None, None, None, "trained", trained=True)
    with pytest.raises(PermissionError):
        OW.own_costate(pol, None, None, 1, 1)
    with pytest.raises(PermissionError):
        OW.k3b_start(pol, None, None, 1)
    with pytest.raises(PermissionError):           # value accuracy's own returns
        OW.own_return(pol, None, None, np.zeros((1, 8)), np.zeros((1, 2)), np.zeros((1, 2)), 1)
    net = CR.CriticNet(lambda p, o: o, None, None, trained=True, name="trained")
    with pytest.raises(PermissionError):
        IN.width_values(net, None, None, 1)
    with pytest.raises(PermissionError):           # the composition's own autodiff
        CR._composed(net, None, None)
    # K3 after the tag (the critic stage's ``gates.k3_trained``): every measurement
    from apexzero.agents.ppo import RunningMeanStd
    from apexzero.costate import gates as GT

    with pytest.raises(PermissionError):
        PB.extractor_gap(actor, critic, np.zeros((2, 222)))
    for fn in (CR.net_apply_gap, CR.float32_values, CR.float32_gap):
        with pytest.raises(PermissionError):
            fn(net, cfg, np.zeros((2, 222)))
    snap.ret_rms = RunningMeanStd(np.zeros(()), np.float32(1.0), np.float32(1.0))
    agent = SimpleNamespace(agent_id="x", timesteps=1, cfg=cfg, snapshot=snap,
                            obs_rms=RunningMeanStd(np.zeros(222), np.ones(222), np.float32(1.0)))
    with pytest.raises(PermissionError):           # refused, never contained
        GT.k3_trained(agent, np.zeros((2, 222)), 5)


def _toy_world(n_geo: int = 12, keys: int = 4, steps: int = 12, seed: int = 0):
    """A toy T for the analysis plumbing: every column the analysis reads."""
    from apexzero.costate import targets as TG

    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_geo):
        R, th = 8.0 + 6.0 * g, (90.0, 180.0, 45.0)[g % 3]
        for k in range(keys):
            for t in range(steps):
                rows.append((g, R, th, f"g{g}k{k}", 1 if k % 2 == 0 else -1, 5 * t, k < 3))
    n = len(rows)
    tab = {"R": np.array([r[1] for r in rows]), "theta": np.array([r[2] for r in rows]),
           "key_id": np.array([r[3] for r in rows]), "direction": np.array([r[4] for r in rows]),
           "pair_id": np.array([r[0] for r in rows]), "t": np.array([r[5] for r in rows]),
           "braking_key": np.array([r[6] for r in rows]),
           "interior": rng.random(n) < 0.7, "delta_bound": rng.random(n) < 0.1,
           "pedal_saturated": rng.random(n) < 0.2, "wall_side": rng.choice([-1, 0, 0, 1], n),
           "phase": rng.choice(["approach", "braking", "corner", "exit"], n),
           "multi_basin": np.zeros(n, bool), "ds_apex": rng.normal(size=n) * 30,
           "vx": 40 + 5 * rng.normal(size=n), "width": np.full(n, 12.0),
           "Pi_in": np.abs(rng.normal(size=n)), "W": np.full(n, 400)}
    tab["interior20"] = tab["interior"] & (rng.random(n) < 0.6)
    tab["near_limit"] = ~tab["interior"]
    tab["contact"] = tab["near_limit"] & (rng.random(n) < 0.4)
    tab["corner_band"] = rng.random(n) < 0.4
    for c in TG.STATE_NAMES:
        tab[f"lam_{c}"] = rng.normal(size=n)
    tab["sigma_pedal"] = np.abs(rng.normal(size=n))
    tab["pedal_switching"] = rng.random(n) < 0.3
    tab["v_star_inf"] = (60.0 + 0.5 * tab["vx"] + 4.0 * tab["lam_vx"] + 0.1 * tab["t"]
                         + 0.2 * rng.normal(size=n))
    lams = {c: tab[c] for c in tab if c.startswith("lam_")}
    frozen = {"winsor_bounds": {**{c: [float(v.min()), float(v.max())] for c, v in lams.items()},
                                "Pi_in": [0.0, 10.0], "sigma_pedal": [0.0, 10.0]},
              "power_floor": dict.fromkeys(("s", "n", "xi", "vx"), True),
              "kappa_set": [[8.0, 90.0]], "k2b_excluded": {"exit": ["vx"]}}
    return SimpleNamespace(tab=tab, n=n), frozen


def test_every_pre_registered_variant_is_reported(tmp_path):
    """Review finding: only partition 0 under key weights was computed. analyse_network and
    probes_of report partitions 1-10, geometry and state weights, the small geometries and
    the kappa set dropped, and the braking / non-braking sub-scores; C3-w carries both
    covariates and leaves out, with its reason, a network whose x keeps no scored row."""
    cs = _script()
    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB

    world, frozen = _toy_world()
    n = world.n
    rng = np.random.default_rng(3)
    lam8 = np.stack([world.tab[f"lam_{c}"] for c in ("s", "n", "xi", "vx", "vy", "r", "delta",
                                                     "pedal")], 1)
    g = np.repeat((lam8 + rng.normal(size=lam8.shape))[:, None], 6, 1)
    crit = {"g": g, "g_held": g.copy(), "value": rng.normal(size=(n, 6)),
            "pi": rng.normal(size=(n, 6, 2)), "random_g": rng.normal(size=(2, n, 8)),
            "dV_dwin": rng.normal(size=n)}
    # a value fit's null per held-out fold [F, n, 8] (``analysis.FOLD_NULLS``), QSS [n, 8]
    nulls = {"qss": rng.normal(size=(n, 8)), "value_ridge": rng.normal(size=(6, n, 8))}
    flags = {"observability": SimpleNamespace(s_unobservable=np.zeros(n, bool))}
    G = AN.geometry_clusters(world.tab)[1]
    out = cs.analyse_network("toy", world, frozen, crit, nulls, flags,
                             PB.cluster_counts(G, 50))
    assert set(out["C1"]["null_diff"]) == {"qss", "value_ridge", "random_critics"}
    want = {f"partition{p}" for p in range(1, 11)} | {
        "weights:geometry", "weights:state", "drop_small", "drop_kappa_set", "braking_keys",
        "non_braking_keys"}
    assert set(out["sensitivity"]) == want | {"pullback:held"}
    assert not any("error" in v for v in out["sensitivity"].values())
    assert out["sensitivity"]["partition3"]["nulls_omitted"] == ["value_ridge"]
    assert out["sensitivity"]["drop_small"]["nulls_omitted"] == []
    # review finding: the pre-registered accel-held pullback (Section 3.2) was never scored.
    # It is C1 on crit['g_held'] with every null left out (a null never enters R2_c: with
    # g_held = g its R2_c, CI and Spearman are C1's own), descriptive
    held = out["sensitivity"]["pullback:held"]
    assert held["nulls_omitted"] == ["qss", "random_critics", "value_ridge"]
    assert held["null_diff"] == {} and held["label_scope"] == "descriptive"
    assert held["pullback"] == "held" and held["descriptive"]
    # without nulls C1's null rule is vacuous: its band, never a decision or a label
    assert "decision" not in held and "label_at_scale" not in held
    assert held["band"] in ("confirm", "refute", "inconclusive", "not testable")
    assert held["n_rows"] == out["C1"]["n_rows"]
    for k in ("stat", "ci", "r2c", "spearman_ci"):
        np.testing.assert_allclose(held[k], out["C1"][k], rtol=1e-12, atol=1e-14, err_msg=k)
    other = cs.analyse_network("toy", world, frozen, {**crit, "g_held": g[:, :, ::-1]}, nulls,
                               flags, PB.cluster_counts(G, 50), variants=["pullback:held"],
                               strata=())
    assert set(other["sensitivity"]) == {"pullback:held"}
    assert other["sensitivity"]["pullback:held"]["stat"] != out["C1"]["stat"]
    assert len(out["robustness_C1"]["stat"]) == 10 and "C1-braking" in out
    assert out["decision_rows"]["vx"] < out["decision_rows"]["s"]      # K2b's frozen rule
    acts = {"z": rng.normal(size=(n, 6)), "m": rng.normal(size=(n, 2)),
            "actor_h3": rng.normal(size=(n, 5)),
            **{f"random{s}_h3": rng.normal(size=(n, 5)) for s in POOL}}
    pr = cs.probes_of(acts, world, frozen, world.tab["interior"], world.tab["pedal_switching"],
                      np.ones((n, 4), bool))
    assert set(pr["sensitivity"]) == want and "C2" in pr and "C2-sw" in pr
    assert all({"C2", "C2-sw"} <= set(v) for v in pr["sensitivity"].values())
    assert len(pr["robustness_C2"]["stat"]) == 10
    # C3-w: x from every generalist's width derivative, y and both covariates from the record
    for name in AN.GENERALISTS:
        p = tmp_path / "post" / "critic" / f"{PR.net_file(name)}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, dV_dwin=world.tab["Pi_in"] + rng.normal(size=n))
    rec = {"frozen": frozen,
           "c3w_inputs": {nm: {"completion": float(rng.random()), "competence": float(i),
                               "overlap": float(-i)} for i, nm in enumerate(AN.GENERALISTS)}}
    c3 = cs.c3w_of(tmp_path, world, PB.cluster_counts(G, 50), rec)
    assert c3["n"] == 31 and set(c3["partial"]) == {"competence", "overlap"}
    assert not c3["critic_missing"] and not c3["x_not_testable"]
    assert set(c3["x_n_uncalibrated"].values()) == {0}
    # finding #10's rule for C3-w's x: a network whose x keeps no scored row is 'not
    # testable' (never a zero-filled R2), left out as missing, with its reason
    gone = AN.GENERALISTS[4]
    np.savez(tmp_path / "post" / "critic" / f"{PR.net_file(gone)}.npz",
             dV_dwin=np.full(n, np.nan))
    c3 = cs.c3w_of(tmp_path, world, PB.cluster_counts(G, 50), rec)
    assert c3["n"] == 30 and gone in c3["missing"] and not c3["critic_missing"]
    assert c3["x_not_testable"] == {gone: "no row"}


def test_the_probes_stage_needs_the_control_pool_in_each_activations_file(tmp_path):
    """The primary C2 and C2-sw are scored against the control pool (random-init actors
    0-19), so the probes stage refuses, before any network is probed, an activations file
    that is missing or holds only the 5 controls' h3 (``pool_problems``); a paired
    increment (K4's plants) takes no pool."""
    cs = _script()
    from apexzero.costate import analysis as AN

    d = tmp_path / "post" / "activations"
    d.mkdir(parents=True)
    np.savez(d / f"{PR.net_file('G1')}.npz", **{f"random{s}_h3": np.zeros((2, 3))
                                                 for s in POOL})
    np.savez(d / f"{PR.net_file('G2')}.npz", **{f"random{s}_h3": np.zeros((2, 3))
                                                 for s in range(5)})
    assert cs.pool_problems(tmp_path, ["G1"]) == []
    probs = cs.pool_problems(tmp_path, ["G1", "G2", "G3"])
    assert len(probs) == 2 and "lacks the control pool's h3 of seeds [5, 6" in probs[0]
    assert probs[1].endswith("is missing")
    with pytest.raises(ValueError, match="no control pool"):
        AN.c2_network(None, [], None, None, None, None, None, None, ("s",), {"s": True},
                      np.ones(2, bool), host=np.zeros(2), pool=np.zeros((1, 1), int))
    lab = AN.controls_label(POOL, True)
    assert lab["seeds"] == list(range(20)) and lab["ci"] == "two-level" and lab["n"] == 20
    assert AN.controls_label(range(5), False)["ci"] == "geometry"


def test_c2_cross_fits_with_the_frozen_decoy_k():
    """f_D's one k (``states.KNN_K``) is frozen in the record; probes_of cross-fits every
    partition at the frozen k ('frozen'), at ``states.KNN_K`` where the frozen values lack it
    ('constant'), and records which."""
    cs = _script()
    from apexzero.costate import states as ST

    world, frozen = _toy_world()
    n = world.n
    rng = np.random.default_rng(5)
    acts = {"z": rng.normal(size=(n, 6)), "m": rng.normal(size=(n, 2)),
            "actor_h3": rng.normal(size=(n, 5)),
            **{f"random{s}_h3": rng.normal(size=(n, 5)) for s in POOL}}
    other = 3 if ST.KNN_K != 3 else 5
    fz = {**frozen, "decoy_k": {"k": other, "scope": "every partition"}}
    fk = {**frozen, "decoy_k": {"k": ST.KNN_K, "scope": "every partition"}}
    assert PR.decoy_k(fz) == other and PR.decoy_k(fk) == ST.KNN_K
    assert PR.decoy_k(frozen) is None and PR.decoy_k({"decoy_k": {"k": None}}) is None
    args = (acts, world)
    rows = (world.tab["interior"], world.tab["pedal_switching"])
    opts = {"variants": ["partition1"], "strata": {"C2": (), "C2-sw": ()}}
    got = cs.probes_of(*args, fz, *rows, **opts)
    ref = cs.probes_of(*args, fk, *rows, **opts)
    bare = cs.probes_of(*args, frozen, *rows, **opts)
    assert got["decoy_k"] == {"k": other, "source": "frozen"}
    assert ref["decoy_k"] == {"k": ST.KNN_K, "source": "frozen"}
    assert bare["decoy_k"] == {"k": ST.KNN_K, "source": "constant"}
    assert bare["C2"]["r2_base"] == ref["C2"]["r2_base"]
    # the frozen k is the one used, on partition 0 and on every robustness partition
    assert got["C2"]["r2_base"] != ref["C2"]["r2_base"]
    assert (got["sensitivity"]["partition1"]["C2"]["r2_base"]
            != ref["sensitivity"]["partition1"]["C2"]["r2_base"])


def _own_returns(tab, v, rng, crash_share: float = 0.1) -> dict:
    """A planted own stage's value records (``own_of``'s v_* fields) on the toy T: the rows
    the stage restarts from (T's interior every 20th step, and D at t = 0), the policy's own
    return ``v`` there, a share of runs crashed (their J garbage, never counted)."""
    t = np.asarray(tab["t"])
    asked = (np.asarray(tab["interior"], bool) & (t % 20 == 0) & (t > 0)) | (t == 0)
    crashed = asked & (rng.random(len(t)) < crash_share)
    det = np.where(asked, v, np.nan)
    det[crashed] = 1e6
    H = np.where(asked, 1000, 0)
    return {"v_rows": asked, "v_own_det": det, "v_own_valid": asked & ~crashed,
            "v_own_terminated": crashed, "v_own_strip_end": np.zeros(len(t), bool),
            "v_own_H": H, "v_own_tail": np.where(asked, 0.995 ** H, np.nan),
            "v_own_inf": det, "v_steps": np.int64(1000)}


def test_value_accuracy_decides_c1_own_and_every_strata_table_is_reported():
    """Section 3.3 'Value accuracy', Section 13 'Built before the tag'. The critic's value
    accuracy is the out-of-fold R2 of sigma_R V_c against its deterministic policy's own
    discounted return (``own_of``'s v_own_det) on the own stage's rows (T's interior restarts
    and D), crashed runs dropped, the frozen folds; with K3b's trained analogue it decides
    C1-own's standing. A critic that fits its own policy's value passes even where that value
    departs nonlinearly from the optimum's V*inf (C1-own a secondary); one that fits V*inf
    instead fails it; the V*inf score is reported beside it and never decides; a random
    critic fails both; without the own stage's returns it does not pass. analyse_network and
    probes_of report each hypothesis' strata table; a stratum a variant already scored reuses
    that record (C1-braking and its strata row carry one CI, with slope CIs); a recomputed C2
    or C2-sw row is computed on its own rows; a non-confirm row carries its slope CIs; a
    kappa-clipped stratum on few geometries is a row, never an exception; merge_probes puts
    C2's and C2-sw's tables beside C1's."""
    cs = _script()
    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB

    world, frozen = _toy_world()
    tab, n = world.tab, world.n
    rng = np.random.default_rng(4)
    lam8 = np.stack([tab[f"lam_{c}"] for c in ("s", "n", "xi", "vx", "vy", "r", "delta",
                                                "pedal")], 1)
    g8 = np.repeat((lam8 + 0.3 * rng.normal(size=lam8.shape))[:, None], 6, 1)
    base = {"g": g8, "g_held": g8.copy(), "pi": rng.normal(size=(n, 6, 2)),
            "random_g": rng.normal(size=(2, n, 8)), "dV_dwin": rng.normal(size=n)}
    vstar = tab["v_star_inf"]
    # the policy's own return departs nonlinearly from V*inf: an affine map of V*inf
    # explains little of it
    u = (vstar - vstar.mean()) / vstar.std()
    v_pi = 40.0 + 6.0 * u ** 2 + 0.5 * rng.normal(size=n)

    def critic(v, noise=0.05):
        return {**base, "value": np.stack([v + noise * v.std() * rng.normal(size=n)
                                           for _ in range(6)], 1)}

    fits_own, fits_vstar = critic(v_pi), critic(vstar)
    random_ = {**base, "value": rng.normal(size=(n, 6))}
    lo8 = lam8 + 0.1 * rng.normal(size=lam8.shape)
    own = {"lam_own": lo8, "lam_own_held": lo8.copy(), "valid": np.ones(n, bool),
           **_own_returns(tab, v_pi, rng)}
    ok = own["v_rows"] & own["v_own_valid"]
    nulls = {"qss": rng.normal(size=(n, 8))}
    geo = np.asarray(tab["pair_id"])
    kc = np.isin(geo, (0, 1)) & (rng.random(n) < 0.8)    # kappa clipping on 2 geometries
    flags = {"observability": SimpleNamespace(s_unobservable=np.zeros(n, bool)),
             "kappa_clipped": kc}
    counts = PB.cluster_counts(AN.geometry_clusters(tab)[1], 50)
    a = cs.analyse_network("toy", world, frozen, fits_own, nulls, flags, counts, own, True,
                           variants=["braking_keys", "non_braking_keys"])
    va, vs = a["value_accuracy"], a["value_accuracy_vstar"]
    assert va["pass"] and va["r2"] > 0.95 and len(va["heads"]) == 5 and min(va["heads"]) > 0.95
    assert va["target"] == "v_own_det" and va["rows"] == "own"
    assert va["n"] == va["n_scored"] == int(ok.sum()) and va["n_missing"] == 0
    assert va["n_asked"] == int(own["v_rows"].sum()) and va["n_terminated"] == int(
        own["v_own_terminated"].sum()) > 0                      # crashed runs: reported, dropped
    assert va["n_d"] == int((ok & (tab["t"] == 0)).sum()) > 0 and va["n_t_restarts"] > 0
    assert va["tail"]["max"] == pytest.approx(0.995 ** 1000) and va["v_steps"] == 1000
    assert vs["target"] == "v_star_inf" and vs["rows"] == "interior" and vs["descriptive"]
    assert "pass" not in vs and not vs["meets_threshold"] and vs["r2"] < 0.5
    assert vs["n"] == int(tab["interior"].sum())
    assert a["C1-own"]["status"] == "secondary"               # the V*inf score never decides
    # Section 13: lambda^own's accel-held form is reported beside C1-own, descriptively
    # (the critic's held pullback against it; with both forms equal, C1-own's numbers)
    oh = a["C1-own_held"]
    assert oh["descriptive"] and oh["pullback"] == "held"
    assert oh["status"] == "descriptive (the accel-held form)"
    assert oh["stat"] == a["C1-own"]["stat"] and oh["ci"] == a["C1-own"]["ci"]
    f = cs.analyse_network("toy", world, frozen, fits_vstar, nulls, flags, counts, own, True,
                           variants=["primary"], strata=())
    assert f["value_accuracy_vstar"]["meets_threshold"] and not f["value_accuracy"]["pass"]
    assert f["C1-own"]["status"] == "descriptive (value accuracy)"
    b = cs.analyse_network("toy", world, frozen, random_, nulls, flags, counts, own, True,
                           variants=["primary"], strata=("interior", "contact"))
    assert not b["value_accuracy"]["pass"] and b["value_accuracy"]["r2"] < 0.8
    assert not b["value_accuracy_vstar"]["meets_threshold"]
    assert b["C1-own"]["status"] == "descriptive (value accuracy)"
    assert b["C1-own"]["validity"] == {"k3b_trained": True, "value_accuracy": False}
    c = cs.analyse_network("toy", world, frozen, fits_own, nulls, flags, counts, own,
                           np.False_, variants=["primary"], strata=())
    assert c["C1-own"]["status"] == "descriptive (K3b)" and c["by_stratum"] == {"C1": {}}
    no_ret = {"lam_own": own["lam_own"], "valid": own["valid"]}
    d0 = cs.analyse_network("toy", world, frozen, fits_own, nulls, flags, counts, no_ret, True,
                            variants=["primary"], strata=())
    assert not d0["value_accuracy"]["pass"] and "own returns" in d0["value_accuracy"]["reason"]
    assert d0["C1-own"]["status"] == "descriptive (value accuracy)"
    # C1's strata table: every stratum, the primary row the decision's own record
    t1 = a["by_stratum"]["C1"]
    assert list(t1) == list(AN.C1_STRATA)
    assert t1["interior"]["primary"] and t1["interior"]["stat"] == a["C1"]["stat"]
    assert np.all(np.isfinite(t1["interior"]["slope_ci"]))       # slope CIs per stratum
    for name, r in t1.items():
        assert r["descriptive"] is (name != "interior") and "band" in r, name
        assert r["n"] == int(AN.stratum_rows(AN.strata_tab(tab, flags), name).sum()), name
        if r["band"] != "not testable":
            assert np.isfinite(r["stat"]) and set(r["null_diff"]) == {"qss", "random_critics"}, \
                name
    assert t1["contact"]["n"] == int(tab["contact"].sum())
    assert t1["interior&kappa_clipped"]["geometries"] <= 2
    assert t1["interior&W:800"]["n"] == 0 and t1["interior&W:800"]["band"] == "not testable"
    # C1-braking and its strata row are one record: the same CI and band, with slope CIs
    for sname, vname in (("interior&braking_key", "braking_keys"),
                         ("interior&~braking_key", "non_braking_keys")):
        v = a["sensitivity"][vname]
        assert t1[sname]["ci"] == v["ci"] and t1[sname]["band"] == v["decision"], sname
        assert np.all(np.isfinite(v["slope_ci"])), sname
    assert a["C1-braking"]["ci"] == t1["interior&braking_key"]["ci"]
    assert a["C1-braking"]["label_scope"] == "descriptive"
    assert set(b["by_stratum"]["C1"]) == {"interior", "contact"}
    # a non-confirm row carries its slope CIs (recomputed and primary alike)
    r = cs.analyse_network("toy", world, frozen, {**fits_own, "g": rng.normal(size=(n, 6, 8))},
                           nulls, flags, counts, None, True, variants=["primary"],
                           strata=("interior", "contact"))
    for nm, row in r["by_stratum"]["C1"].items():
        assert row["band"] != "confirm" and np.all(np.isfinite(row["slope_ci"])), nm
    # review finding: without the own stage's output C1-own was silently absent; a
    # non-confirmatory run records why (the confirmatory one requires the output)
    for k in ("C1-own", "C1-own_held"):
        assert r[k] == {"decision": "not testable", "descriptive": True,
                        "reason": "no own stage output"}, k
    # C1-pattern on the interior's pedal-saturated states, descriptive
    ps = a["C1-pattern_saturated"]
    assert ps["descriptive"] and ps["n"] == int((tab["interior"] & tab["pedal_saturated"]).sum())
    # C2's and C2-sw's tables from probes_of, then merged beside C1's
    acts = {"z": rng.normal(size=(n, 6)), "m": rng.normal(size=(n, 2)),
            "actor_h3": rng.normal(size=(n, 5)),
            **{f"random{s}_h3": rng.normal(size=(n, 5)) for s in POOL}}
    pr = cs.probes_of(acts, world, frozen, tab["interior"], tab["pedal_switching"],
                      np.ones((n, 4), bool), variants=["braking_keys"], kappa_clipped=kc)
    t2, tsw = pr["by_stratum"]["C2"], pr["by_stratum"]["C2-sw"]
    assert list(t2) == list(AN.C2_STRATA) and list(tsw) == list(AN.C2SW_STRATA)
    assert t2["interior"]["primary"] and t2["interior"]["stat"] == pr["C2"]["stat"]
    # the decisions against the control pool with the two-level CI; every other record (the
    # variants, the robustness summary, the strata tables' other rows) against the 5
    # controls with the geometry-only CI, each labelled
    pool, five = AN.controls_label(POOL, True), AN.controls_label(range(5), False)
    for h, tb, prim in (("C2", t2, "interior"), ("C2-sw", tsw, "pedal_switching")):
        assert pr[h]["controls"] == pool and pr[h]["ci_kind"] == "two-level", h
        assert len(pr[h]["r2_random"]) == 20 and pr[h]["pool"]["of"] == 21, h
        assert pr[h]["ci_geometry"] != pr[h]["ci"], h
        assert pr["sensitivity"]["braking_keys"][h]["controls"] == five, h
        assert len(pr["sensitivity"]["braking_keys"][h]["r2_random"]) == 5, h
        assert "ci_kind" not in pr["sensitivity"]["braking_keys"][h], h
        assert tb[prim]["controls"] == pool and tb[prim]["pool"] == pr[h]["pool"], h
        assert all(r["controls"] == five for k, r in tb.items()      # a row on the decided
                   if k != prim and "controls" in r), h                 # rows included
        assert sum("controls" in r for k, r in tb.items() if k != prim) >= 3, h
    assert tsw["pedal_switching"]["primary"] and tsw["pedal_switching"]["stat"] == \
        pr["C2-sw"]["stat"]
    assert t2["interior&braking_key"]["stat"] == pr["sensitivity"]["braking_keys"]["C2"]["stat"]
    assert not t2["near_limit"]["primary"] and t2["near_limit"]["n"] == int(
        tab["near_limit"].sum())
    assert all("band" in r and "stat" in r for r in (*t2.values(), *tsw.values())
               if r["band"] != "not testable")
    # a recomputed row is computed on its own rows (one direct run covers both hypotheses)
    # (its decision against the pool; the row's gain is the same fits' against the 5 controls)
    dr = cs.probes_of(acts, world, frozen, tab["near_limit"], tab["pedal_saturated"],
                      np.ones((n, 4), bool), variants=["primary"], strata={})
    for row_, d in ((t2["near_limit"], dr["C2"]), (tsw["pedal_saturated"], dr["C2-sw"])):
        assert row_["r2_base"] == d["r2_base"] and row_["raw"] == d["raw"]
        five_gain = np.subtract(d["raw"], np.median(np.subtract(
            np.asarray(d["r2_random"])[:5], d["r2_base"]), axis=0))
        np.testing.assert_allclose(row_["gain"], five_gain, rtol=0, atol=1e-15)
    assert t2["near_limit"]["stat"] != t2["interior"]["stat"]
    assert all(max(r["scored_rows"]) == r["n"] for r in t2.values() if "scored_rows" in r)
    # the kappa-clipped strata: a row on 2 geometries, not an exception; without the flags
    # 'not testable'
    assert t2["interior&kappa_clipped"]["geometries"] <= 2
    assert t2["interior&kappa_clipped"]["band"] == "not testable"     # explicitly, not a
    assert "inner cross-validation" in t2["interior&kappa_clipped"]["reason"]    # LinAlgError
    assert "inner cross-validation" in tsw["pedal_switching&kappa_clipped"]["reason"]
    assert t2["interior&W:800"]["band"] == "not testable" and t2["interior&W:800"]["n"] == 0
    assert tsw["pedal_switching&kappa_clipped"]["n"] == int((tab["pedal_switching"] & kc).sum())
    nokc = cs.probes_of(acts, world, frozen, tab["interior"], tab["pedal_switching"],
                        variants=["primary"], strata={"C2": ("interior&kappa_clipped",)})
    row = nokc["by_stratum"]["C2"]["interior&kappa_clipped"]
    assert row["band"] == "not testable" and "kappa_clipped" in row["reason"]
    small = cs.probes_of(acts, world, frozen, np.ones(n, bool), tab["pedal_switching"],
                         variants=["primary"], strata={"C2": ("interior",)})
    assert not small["by_stratum"]["C2"]["interior"]["primary"]    # not the decided rows
    assert small["by_stratum"]["C2-sw"] == {}
    merged = cs.merge_probes(dict(a), json.loads(json.dumps(pr, default=float)))
    assert set(merged["by_stratum"]) == {"C1", "C2", "C2-sw"}
    assert merged["C2"]["stat"] == pr["C2"]["stat"] and "C1" in merged
    assert merged["sensitivity"]["braking_keys"]["C2"]["stat"] == \
        pr["sensitivity"]["braking_keys"]["C2"]["stat"]
    assert merged["sensitivity"]["braking_keys"]["stat"] == a["C1-braking"]["stat"]   # C1's


def _toy_inputs(world, seed: int = 5):
    """A toy network's critic, nulls, flags and activations on ``_toy_world``'s T."""
    rng = np.random.default_rng(seed)
    tab, n = world.tab, world.n
    lam8 = np.stack([tab[f"lam_{c}"] for c in ("s", "n", "xi", "vx", "vy", "r", "delta",
                                                "pedal")], 1)
    g = np.repeat((lam8 + 0.5 * rng.normal(size=lam8.shape))[:, None], 6, 1)
    crit = {"g": g, "g_held": g.copy(), "value": rng.normal(size=(n, 6)),
            "pi": rng.normal(size=(n, 6, 2)), "random_g": rng.normal(size=(2, n, 8)),
            "dV_dwin": tab["Pi_in"] + rng.normal(size=n)}
    flags = {"observability": SimpleNamespace(s_unobservable=np.zeros(n, bool)),
             "kappa_clipped": np.zeros(n, bool)}
    acts = {"z": rng.normal(size=(n, 6)), "m": rng.normal(size=(n, 2)),
            "actor_h3": rng.normal(size=(n, 5)),
            **{f"random{s}_h3": rng.normal(size=(n, 5)) for s in POOL}}
    return crit, {"qss": rng.normal(size=(n, 8))}, flags, acts


def test_a_raising_descriptive_is_recorded_and_a_raising_decision_propagates(monkeypatch):
    """The containment rule (``analysis.guarded``; the analysis runs once after the tag and
    a crash voids the tag): a descriptive computation that raises is recorded as {'error':
    '<type>: <message>'} in its place; a primary decision or a named secondary's decision
    rule raises. Review finding: the primary C2 and C2-sw turned any ValueError (a
    LinAlgError, a broadcasting bug) into a 'not testable' verdict; only
    ``probes.NotTestable`` is one now."""
    cs = _script()
    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB

    world, frozen = _toy_world()
    tab, n = world.tab, world.n
    crit, nulls, flags, acts = _toy_inputs(world)
    counts = PB.cluster_counts(AN.geometry_clusters(tab)[1], 50)

    def boom(*a, **k):
        raise RuntimeError("boom")

    def run(**kw):
        return cs.analyse_network("toy", world, frozen, crit, nulls, flags, counts,
                                  **{"variants": ["primary"], "strata": ("interior",), **kw})

    # descriptives: C1-heads, W1, the V*inf score, a variant, a strata row, the held pullback
    for target, key in (("c1_heads", "C1-heads"), ("calibrated_agreement", "W1")):
        with monkeypatch.context() as m:
            m.setattr(AN, target, boom)
            assert run()[key] == {"error": "RuntimeError: boom"}, key
    with monkeypatch.context() as m:
        m.setattr(AN, "value_accuracy", boom)       # value accuracy itself is a decision rule
        with pytest.raises(RuntimeError, match="boom"):
            run()
    real_c1 = AN.c1
    primary_rows = np.asarray(tab["interior"], bool)

    def c1_fails_off_the_interior(g, lam, outer, w, cl, G, d, rows, *a, **k):
        if not np.array_equal(rows, primary_rows):
            raise ValueError("a descriptive bug")
        return real_c1(g, lam, outer, w, cl, G, d, rows, *a, **k)

    with monkeypatch.context() as m:
        m.setattr(AN, "c1", c1_fails_off_the_interior)
        out = run(variants=["non_braking_keys", "pullback:held"], strata=("interior", "contact"))
    assert out["sensitivity"]["non_braking_keys"] == {"error": "ValueError: a descriptive bug"}
    assert "error" not in out["sensitivity"]["pullback:held"]      # on the interior: scored
    row = out["by_stratum"]["C1"]["contact"]
    assert row["band"] == "error" and row["error"] == "ValueError: a descriptive bug"
    assert row["n"] == int(tab["contact"].sum())
    assert out["by_stratum"]["C1"]["interior"]["primary"]
    assert "error" not in out["C1"] and out["C1"]["decision"] != "error"
    with monkeypatch.context() as m:                # C1-braking is a secondary's decision rule
        m.setattr(AN, "c1", c1_fails_off_the_interior)
        with pytest.raises(ValueError, match="descriptive bug"):
            run(variants=["braking_keys"])
    for target in ("c1", "c1_limit", "c1_pattern", "c1_own"):     # decisions: they raise
        with monkeypatch.context() as m:
            m.setattr(AN, target, boom)
            with pytest.raises(RuntimeError, match="boom"):
                run(own={"lam_own": np.zeros((n, 8)), "valid": np.ones(n, bool)})
    # C2 and C2-sw: only NotTestable is a 'not testable' verdict on the primary entry
    real_c2 = AN.c2_network
    for exc, raises in ((PB.NotTestable("no component passing the power floor has a scored "
                                        "row"), False),
                        (ValueError("operands could not be broadcast"), True),
                        (np.linalg.LinAlgError("Eigenvalues did not converge"), True)):
        def c2_raising(*a, exc=exc, **k):
            raise exc

        with monkeypatch.context() as m:
            m.setattr(AN, "c2_network", c2_raising)
            if raises:
                with pytest.raises(type(exc)):
                    cs.probes_of(acts, world, frozen, primary_rows, tab["pedal_switching"],
                                 variants=["primary"], strata={})
            else:
                pr = cs.probes_of(acts, world, frozen, primary_rows, tab["pedal_switching"],
                                  variants=["primary"], strata={})
                assert pr["C2"] == {"decision": "not testable", "reason": str(exc),
                                    "controls": AN.controls_label(PB.POOL_SEEDS, True)}

    def c2_fails_off_the_interior(h3, rnd, z, m_, res, fo, w_all, d, names, passing, rows,
                                  *a, **k):
        if not np.array_equal(rows, primary_rows):
            raise ValueError("a descriptive bug")
        return real_c2(h3, rnd, z, m_, res, fo, w_all, d, names, passing, rows, *a, **k)

    with monkeypatch.context() as m:
        m.setattr(AN, "c2_network", c2_fails_off_the_interior)
        pr = cs.probes_of(acts, world, frozen, primary_rows, tab["pedal_switching"],
                          variants=["braking_keys"], strata={"C2": ("interior", "contact")})
    assert "decision" in pr["C2"] and "error" not in pr["C2"]
    assert pr["sensitivity"]["braking_keys"]["C2"] == {"error": "ValueError: a descriptive bug"}
    assert "error" not in pr["sensitivity"]["braking_keys"]["C2-sw"]
    assert pr["by_stratum"]["C2"]["contact"]["band"] == "error"
    assert pr["by_stratum"]["C2"]["interior"]["primary"]
    # C3-w and C3-R2c: a covariate's partial is descriptive, the test itself a decision
    x = {k: float(i) for i, k in enumerate(AN.GENERALISTS)}
    with monkeypatch.context() as m:
        m.setattr(AN, "_partial_c3", boom)
        c3 = AN.c3_r2c(x, x, n_perm=50, covariates={"competence": x})
    assert c3["partial"] == {"competence": {"error": "RuntimeError: boom"}}
    assert c3["rho"] == pytest.approx(1.0)
    with monkeypatch.context() as m:
        m.setattr(AN, "c3_w", boom)
        with pytest.raises(RuntimeError, match="boom"):
            AN.c3_r2c(x, x, n_perm=50)


def test_the_confirmatory_analysis_requires_the_own_stage(tmp_path):
    """Review findings: the run-once analysis checked the own stage's files only where they
    existed, so a skipped or partial own stage dropped C1-own and value accuracy for good.
    Every decision network's post/own/<net>.npz (``OWN_FIELDS``) and K3b's trained analogue
    (<net>.k3b.json, whose decision must be the npz's k3b_pass) are required and
    stamp-checked; a missing specialist's critic (its descriptive C1) is refused too."""
    cs = _script()
    from apexzero.costate import analysis as AN

    root = tmp_path / "root"
    nets = [PR.net_file(AN.network_of(a)) for a in AN.DECISION_NETWORKS]
    rec = {"code_hash": "0" * 64, "environment": None, "outputs": {}}
    prov = SimpleNamespace(code_hash="0" * 64, deviation=None, tag=None)

    def own_probs():
        return [p for p in cs.confirmatory_input_problems(rec, root, prov)
                if any(s in p for s in ("(C1-own, value accuracy)", "(K3b's trained analogue)",
                                        "has no K3b decision", "k3b_pass is not"))]

    probs = own_probs()
    assert sum("is missing (C1-own, value accuracy)" in p for p in probs) == 9
    assert sum("is missing (K3b's trained analogue)" in p for p in probs) == 9
    assert sum("specialist's descriptive C1" in p for p in cs.confirmatory_input_problems(
        rec, root, prov)) == len(AN.SPECIALISTS)
    own_dir = root / "post" / "own"
    own_dir.mkdir(parents=True)
    full = {k: np.zeros(3) for k in cs.OWN_FIELDS}
    for i, f in enumerate(nets):
        arrays = {**full, "k3b_pass": np.bool_(i != 1)}
        if i == 0:
            arrays.pop("lam_own")
        np.savez(own_dir / f"{f}.npz", **arrays)
        (own_dir / f"{f}.k3b.json").write_text(json.dumps(
            {"decision": {"pass": i not in (1, 2)}} if i != 3 else {"starts": []}))
    probs = own_probs()
    assert probs == [f"post/own/{nets[0]}.npz lacks lam_own (C1-own, value accuracy)",
                     f"post/own/{nets[2]}.npz's k3b_pass is not {nets[2]}.k3b.json's decision",
                     f"post/own/{nets[3]}.k3b.json has no K3b decision"]
    stamps = cs.confirmatory_input_problems(rec, root, prov)
    for f in nets:                                   # both files stamp-checked
        assert f"{f}.npz: no provenance stamp" in stamps
        assert f"{f}.k3b.json: no provenance stamp" in stamps
    assert "(own optional)" not in cs.analysis_paths.__doc__


def test_the_networks_outside_the_decision_set_and_c3_r2c(tmp_path, monkeypatch):
    """Review finding: C3-R2c (a named secondary), C1-heads beyond the decision networks and
    the descriptive C1 of the other 22 generalists and the specialists were never computed.
    The analysis reports, per other generalist, C1 without nulls and C1-heads, per
    specialist C1 without nulls (contained: a specialist without 'g' is an error, not a
    crash), and C3-R2c: C1's R2_c of all 31 generalists (a decision network's from its C1)
    against completion with C3-w's test on its own stream."""
    cs = _script()
    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB
    from apexzero.costate import stream_rng
    from apexzero.fields import network as NW

    world, frozen = _toy_world()
    tab = world.tab
    rng = np.random.default_rng(6)
    out = tmp_path / "o"
    decided = {AN.network_of(a) for a in AN.DECISION_NETWORKS}
    crits, flags = {}, None
    for i, name in enumerate([*AN.GENERALISTS, *AN.SPECIALISTS]):
        crit, nulls, flags, _ = _toy_inputs(world, seed=10 + i)
        if name in AN.SPECIALISTS:
            crit = {k: (v[:, :1] if k in ("g", "g_held", "value", "pi") else v)
                    for k, v in crit.items()}
            if name == AN.SPECIALISTS[-1]:
                crit.pop("g")
        crits[name] = crit
        p = out / "post" / "critic" / f"{PR.net_file(name)}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **crit)
        if name in decided:
            (out / "nulls").mkdir(exist_ok=True)
            np.savez(out / "nulls" / f"{PR.net_file(name)}.npz", **nulls)
    stat = {name: float(rng.random()) for name in decided}
    rec = {"frozen": frozen, "networks": {nm: {} for nm in crits},
           "c3w_inputs": {nm: {"completion": float(rng.random()), "competence": float(i),
                               "overlap": float(-i)} for i, nm in enumerate(AN.GENERALISTS)}}
    monkeypatch.setattr(cs, "_post_tag", lambda a, w: rec)
    monkeypatch.setattr(cs, "_load_world_frozen", lambda a, r: world)
    monkeypatch.setattr(cs, "ROOT", tmp_path / "repo")
    monkeypatch.setattr(cs, "unlocked_by", lambda path, tag, root=None: ("ab" * 32, ""))
    monkeypatch.setattr(cs, "load_agent", lambda name, pin: SimpleNamespace(obs_rms=None))
    monkeypatch.setattr(NW, "upcast32", lambda x: x)
    from apexzero.costate import inputs as IN
    monkeypatch.setattr(IN, "flags", lambda w, rms: flags)
    monkeypatch.setattr(cs, "analyse_network", lambda name, *a, **k: {
        "C1": {"stat": stat[name], "decision": "inconclusive"}})
    cs.main(["analyse", "--out", str(out)])
    d = json.loads((out / "analyse" / "decisions_dry.json").read_text())
    desc = d["descriptive_networks"]
    others = [g for g in AN.GENERALISTS if g not in decided]
    assert len(others) == 22 and set(desc) == set(others) | set(AN.SPECIALISTS)
    counts = PB.cluster_counts(AN.geometry_clusters(tab)[1])
    x = dict(stat)
    for g in others:
        c1 = desc[g]["C1"]
        assert c1["descriptive"] and c1["label_scope"] == "descriptive" and c1["null_diff"] == {}
        assert c1["nulls_omitted"] == list(AN.NULL_NAMES)
        assert set(desc[g]["C1-heads"]) == {"D", "grad_KP_minus_V", "grad_secondary_minus_KP"}
        want, x[g] = cs.describe_network(world, frozen, crits[g], flags, counts)
        assert c1["stat"] == pytest.approx(want["C1"]["stat"], abs=1e-12) == x[g]
    for s in AN.SPECIALISTS[:-1]:
        assert desc[s]["C1"]["descriptive"] and "C1-heads" not in desc[s]
    assert desc[AN.SPECIALISTS[-1]] == {"error": "KeyError: 'g'"}
    r2c = d["C3-R2c"]
    assert r2c["n"] == 31 and r2c["stream"] == ["costate_c3", AN.C3_R2C_SUBSTREAM]
    y, cov = cs.c3_y(list(x), rec)
    ref = AN.c3_w(x, y, rng=stream_rng("costate_c3", AN.C3_R2C_SUBSTREAM), covariates=cov)
    for k in ("rho", "p", "upper_bound", "null_sd", "decision"):
        assert r2c[k] == pytest.approx(ref[k]), k
    assert set(r2c["partial"]) == {"competence", "overlap"}
    # its own stream: C3-w's draws on the same x and y give another null
    assert AN.c3_w(x, y)["null_sd"] != pytest.approx(ref["null_sd"], abs=1e-12)
    assert d["C3-w"]["n"] == 31 and set(d["C3-w_sensitivity"]) == {"drop_small",
                                                                  "drop_kappa_set"}


_K3T = r"""
import dataclasses, json, sys
import importlib.util
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from types import SimpleNamespace
from apexzero.agents.ppo import RunningMeanStd
from apexzero.costate import critic as CR, probes as PB, inputs as IN
from apexzero.fields import network as NW
spec = importlib.util.spec_from_file_location("costate_script", sys.argv[1])
cs = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(spec.origin.rsplit("/", 1)[0]))
spec.loader.exec_module(cs)
rms, cfg = IN.obs_rms_of("G1")
rms32 = jax.tree.map(lambda a: np.asarray(a, np.float32), rms)
GEN, SPEC = "runs/dr-pilot-200M", "S1"
# a world of 8 keys x 5 rows (t = 0..4), raw observations around G1's normaliser's mean
rng = np.random.default_rng(0)
mean, sd = np.asarray(rms.mean, float), np.sqrt(np.asarray(rms.var, float) + 1e-8)
ob = mean + sd * rng.normal(0.0, 1.5, (40, 222))
world = SimpleNamespace(n=40, tab={"t": np.tile(np.arange(5), 8)}, keys=[
    SimpleNamespace(rows=np.arange(5 * i, 5 * i + 5), oj=SimpleNamespace(ob=ob[5 * i:5 * i + 5]))
    for i in range(8)])

def stand_in(c, seed=3, bias=0.0):
    # a random-init stand-in for a trained snapshot (G1's obs_rms; float32 leaves)
    crit = CR.random_critic_params(c, seed)
    crit["params"]["Dense_0"]["bias"] = crit["params"]["Dense_0"]["bias"] + np.float32(bias)
    snap = SimpleNamespace(actor=NW.random_actor_params(c, seed), critic=crit,
                           ret_rms=RunningMeanStd(np.zeros(()), np.float32(4.0), np.float32(1.0)))
    return SimpleNamespace(agent_id="stand-in", timesteps=0, cfg=c, obs_rms=rms32, snapshot=snap)

def run(ag, name):
    r = cs.k3_trained(ag, world, name)
    return {"decision": r["decision"], "checks": r["checks"], "critic": r["critic"],
            "extractor": r["extractor"], "float32": r["float32"], "n": r["n_states"],
            "n_d": r["n_d"], "rows": r["rows"]}

out = {}
try:                                  # before the unlock (the tag): refused, never contained
    cs.k3_trained(stand_in(cfg), world, GEN)
    out["refused"] = False
except PermissionError:
    out["refused"] = True
PB.unlock_trained(); CR.unlock_trained()    # the test's stand-in for the freeze check
out["gen"] = run(stand_in(cfg), GEN)
spec_cfg = cfg.with_(ensemble_heads=0)
out["spec"] = run(stand_in(spec_cfg), SPEC)
# planted failures: V near 1e4 (float32 resolves it to about 1e-3, float64 to 2e-12); a
# head-free critic as a generalist and 5 heads as a specialist; the probe extractor off by
# 1e-8; a float32 measurement that raises (contained: it fails its check)
out["bias"] = run(stand_in(cfg, bias=1e4), GEN)
out["spec_as_gen"] = run(stand_in(spec_cfg), GEN)
out["gen_as_spec"] = run(stand_in(cfg), SPEC)
hidden = PB.hidden
PB.hidden = lambda net, z, xp=np: dataclasses.replace(hidden(net, z, xp),
                                                      out=hidden(net, z, xp).out + 1e-8)
out["extractor_off"] = run(stand_in(cfg), GEN)
PB.hidden = hidden
gap32 = CR.float32_gap
def boom(*a, **k):
    raise ValueError("boom")
CR.float32_gap = boom
out["raised"] = run(stand_in(cfg), GEN)
CR.float32_gap = gap32
# the float32 values this x64 process computes (the test compares them with its own float32
# process's)
v32 = CR.float32_values(CR.trained_critic(stand_in(cfg)), cfg, ob[cs.d_rows(world)])
out["v32"], out["v32_dtype"] = v32.tolist(), str(v32.dtype)
print(json.dumps(out, default=float))
"""


@pytest.mark.skipif(not _have_g1(), reason="G1's normaliser and config are missing")
def test_k3_after_the_tag_on_random_init_stand_ins_with_planted_failures(tmp_path):
    """Section 7 K3 after the tag (review finding: never run on trained networks): the
    critic stage's ``k3_trained`` on random-init stand-ins at D (T's t = 0 rows), in a
    float64 process. It is refused before the unlock (the trained marks; not contained),
    passes a generalist (5 heads) and a specialist (none) with both extractors exact and the
    float32 V within 1e-4, and fails each planted defect on its own check: V near 1e4
    (float32), the wrong head count either way, a probe extractor off by 1e-8, and a float32
    measurement that raises (contained)."""
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([sys.executable, "-c", _K3T, str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=600, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["refused"]
    for k in ("gen", "spec"):
        g = out[k]
        assert g["decision"] == {"pass": True, "failed": [], "reason": None}, (k, g)
        assert g["n"] == g["n_d"] == 8 and g["rows"] == "D: T's t = 0 rows"
        assert g["critic"]["dtype"] == g["critic"]["actor_dtype"] == "float64"
        assert max(g["critic"][q] for q in ("v", "heads", "actor")) <= 1e-10
        assert max(v for q, v in g["extractor"].items()
                   if q in ("actor_mean", "critic_v", "critic_heads")) <= 1e-10
        assert g["float32"]["dtype32"] == "float32" and g["float32"]["max_abs"] < 1e-5
    assert out["gen"]["critic"]["n_heads"] == out["gen"]["extractor"]["heads"] == 5
    assert out["spec"]["critic"]["n_heads"] == out["spec"]["critic"]["flax_heads"] == 0
    b = out["bias"]
    assert b["decision"]["failed"] == ["float32"] and b["float32"]["max_abs"] > 1e-4
    assert b["float32"]["v_absmax"] > 9e3 and b["checks"]["extractor"]
    assert out["spec_as_gen"]["decision"]["failed"] == ["heads"]
    assert out["gen_as_spec"]["decision"]["failed"] == ["heads"]
    assert "expected 0" in out["gen_as_spec"]["decision"]["reason"]
    e = out["extractor_off"]
    assert e["decision"]["failed"] == ["extractor"] and "probe extractor" in e["decision"]["reason"]
    assert e["critic"]["v"] <= 1e-10                   # the critic's extractor is untouched
    ra = out["raised"]
    assert ra["decision"]["failed"] == ["float32"] and ra["float32"] == {
        "error": "ValueError: boom"}
    assert ra["decision"]["reason"].startswith("K3 trained check failed: float32 (")
    # the float32 V of the x64 process (x64 switched off around it) is bit for bit what
    # this float32 process computes: RunningMeanStd.normalize and Critic.apply in float32
    import jax
    import jax.numpy as jnp

    from apexzero.agents.ppo import Critic, RunningMeanStd
    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN

    assert not jax.config.jax_enable_x64
    rms, cfg = IN.obs_rms_of("G1")
    mean, sd = np.asarray(rms.mean, float), np.sqrt(np.asarray(rms.var, float) + 1e-8)
    ob = mean + sd * np.random.default_rng(0).normal(0.0, 1.5, (40, 222))
    r32 = RunningMeanStd(*(jnp.asarray(np.asarray(a, np.float32)) for a in rms))
    ref = np.asarray(Critic(cfg).apply(CR.random_critic_params(cfg, 3), r32.normalize(
        jnp.asarray(ob[::5].astype(np.float32)), 10.0)))
    assert out["v32_dtype"] == "float32" == str(ref.dtype)
    assert np.array_equal(np.asarray(out["v32"], np.float32), ref)


def test_the_critic_stage_writes_k3_and_the_confirmatory_analysis_requires_it(
        tmp_path, monkeypatch):
    """The critic stage writes each network's K3 after-the-tag checks (<net>.k3.json) before
    its arrays, which carry the verdict ('k3_trained_pass'); the confirmatory analysis
    refuses unless all 34 networks have a stamped verdict with a boolean decision that their
    critic's flag repeats."""
    cs = _script()
    from apexzero.costate import analysis as AN
    from apexzero.costate import inputs as IN

    gen, spec = AN.GENERALISTS[0], AN.SPECIALISTS[0]
    rec = {"networks": {gen: {}, spec: {}}}
    order = []

    def k3(agent, world, name):
        order.append(("k3", name))
        ok = name != spec
        return {"decision": {"pass": ok, "failed": [] if ok else ["heads"],
                             "reason": None if ok else "K3 trained check failed: heads (x)"}}

    def crit(agent, world, bank=None):
        order.append(("critic", agent))
        return {"g": np.zeros((3, 6, 8)), "value": np.zeros((3, 6)), "dV_dwin": np.zeros(3)}

    monkeypatch.setattr(cs, "_post_tag", lambda a, w: rec)
    monkeypatch.setattr(cs, "_load_world_frozen", lambda a, r: None)
    monkeypatch.setattr(cs, "load_agent", lambda name, pin: name)
    monkeypatch.setattr(IN, "design", lambda: (None, None))
    monkeypatch.setattr(cs, "k3_trained", k3)
    monkeypatch.setattr(cs, "critic_of", crit)
    out = tmp_path / "o"
    cs.main(["critic", "--out", str(out)])
    assert order == [("k3", gen), ("critic", gen), ("k3", spec), ("critic", spec)]
    d = out / "post" / "critic"
    for name, ok in ((gen, True), (spec, False)):
        f = PR.net_file(name)
        k = json.loads((d / f"{f}.k3.json").read_text())
        assert k["decision"]["pass"] is ok and "provenance" in k
        with np.load(d / f"{f}.npz") as z:
            assert bool(z["k3_trained_pass"]) is ok
    v = cs.k3_trained_of(out)
    assert v[gen] == {"pass": True, "reason": None}
    assert v[spec] == {"pass": False, "reason": "K3 trained check failed: heads (x)"}
    assert v[AN.GENERALISTS[1]] == {"pass": None, "reason": "not run"} and len(v) == 34
    # the confirmatory input check
    prov = SimpleNamespace(code_hash="0" * 64, deviation=None, tag=None)
    crec = {"code_hash": "0" * 64, "environment": None, "outputs": {}}

    def k3_probs():
        return [p for p in cs.confirmatory_input_problems(crec, out, prov)
                if "K3 after the tag" in p or "K3 decision" in p or "k3_trained_pass" in p]

    probs = k3_probs()
    assert sum("k3.json is missing (K3 after the tag)" in p for p in probs) == 32
    assert not any(p.startswith((f"post/critic/{PR.net_file(gen)}.",
                                 f"post/critic/{PR.net_file(spec)}.")) for p in probs)
    g1 = AN.GENERALISTS[1]
    f1, f2 = PR.net_file(g1), PR.net_file(AN.GENERALISTS[2])
    (d / f"{f1}.k3.json").write_text(json.dumps({"decision": {"pass": "yes"}}))
    np.savez(d / f"{f1}.npz", g=np.zeros(3))
    (d / f"{f2}.k3.json").write_text(json.dumps({"decision": {"pass": True}}))
    np.savez(d / f"{f2}.npz", k3_trained_pass=np.bool_(False))
    np.savez(d / f"{PR.net_file(gen)}.npz", g=np.zeros(3))
    probs = k3_probs()
    assert f"post/critic/{f1}.k3.json has no K3 decision" in probs
    assert (f"post/critic/{f2}.npz's k3_trained_pass is not {f2}.k3.json's decision"
            in probs)
    assert f"post/critic/{PR.net_file(gen)}.npz lacks k3_trained_pass (K3 after the tag)" \
        in probs
    stamps = cs.confirmatory_input_problems(crec, out, prov)
    assert f"{f2}.k3.json: no provenance stamp" in stamps        # stamp-checked
    assert str(Path("post/critic") / f"{PR.net_file(gen)}.k3.json") in cs.input_hashes(out)


def test_a_network_failing_k3_after_the_tag_is_missing_from_every_decision(
        tmp_path, monkeypatch):
    """Section 7 K3 after the tag: a network that fails it keeps its statistics, its C1,
    C1-heads, C1-own, C1-pattern, C1-limit, C1-braking, W1, C2 and C2-sw records are marked
    descriptive with the reason 'K3 trained check failed: ...', and it is missing from every
    decision it enters (per network, so R18 counts a missing seed; C3-w and C3-R2c leave it
    out and list it), as H4R's gate2-trained rule treats a failing network; a failing
    network's own analysis is contained (it decides nothing); the analysis lists every
    verdict."""
    cs = _script()
    from apexzero.costate import analysis as AN
    from apexzero.fields import network as NW

    world, frozen = _toy_world()
    out = tmp_path / "o"
    decided = {AN.network_of(a) for a in AN.DECISION_NETWORKS}
    g2, g3s2, other, s1 = (AN.network_of("G2"), AN.network_of("G3s2"), AN.GENERALISTS[-1],
                           AN.SPECIALISTS[0])
    bad = {g2: "K3 trained check failed: float32 (max |V64 - V32| 0.0003 > 0.0001)",
           g3s2: "K3 trained check failed: heads (expected 5: config 5, tree 4, Flax 4, "
                 "probe extractor 4)",
           other: "K3 trained check failed: extractor (critic v 2e-09 > 1e-10)",
           s1: "K3 trained check failed: heads (expected 0: config 5, tree 5, Flax 5, "
               "probe extractor 5)"}
    for i, name in enumerate([*AN.GENERALISTS, *AN.SPECIALISTS]):
        crit, nulls, flags, _ = _toy_inputs(world, seed=10 + i)
        if name in AN.SPECIALISTS:
            crit = {k: (v[:, :1] if k in ("g", "g_held", "value", "pi") else v)
                    for k, v in crit.items()}
        f = PR.net_file(name)
        (out / "post" / "critic").mkdir(parents=True, exist_ok=True)
        np.savez(out / "post" / "critic" / f"{f}.npz", **crit)
        ok = name not in bad
        (out / "post" / "critic" / f"{f}.k3.json").write_text(json.dumps(
            {"decision": {"pass": ok, "failed": [] if ok else ["x"], "reason": bad.get(name)}}))
        if name in decided:
            (out / "nulls").mkdir(exist_ok=True)
            np.savez(out / "nulls" / f"{f}.npz", **nulls)
    probes = {"C2": {"decision": "confirm", "stat": 0.2}, "C2-sw": {"decision": "refute"},
              "by_stratum": {"C2": {"interior": {"primary": True, "descriptive": False},
                                    "contact": {"primary": False, "descriptive": True}}}}
    (out / "post" / "probes").mkdir(parents=True)
    (out / "post" / "probes" / f"{PR.net_file(g2)}.json").write_text(json.dumps(probes))
    rng = np.random.default_rng(3)
    rec = {"frozen": frozen, "networks": {nm: {} for nm in [*AN.GENERALISTS, *AN.SPECIALISTS]},
           "c3w_inputs": {nm: {"completion": float(rng.random()), "competence": float(i),
                               "overlap": float(-i)} for i, nm in enumerate(AN.GENERALISTS)}}

    def analyse(name, *a, **k):
        if name == g3s2:
            raise FloatingPointError("a failing network's statistic raised")
        return {"C1": {"stat": 0.7, "decision": "confirm"},
                "C1-own": {"decision": "refute", "status": "secondary"},
                "C1-heads": {"D": {"decision": "confirm"}},
                "C1-pattern": {"decision": "confirm"}, "C1-limit": {"decision": "refute"},
                "W1": {"r2c": 0.3}, "value_accuracy": {"pass": True}}

    monkeypatch.setattr(cs, "_post_tag", lambda a, w: rec)
    monkeypatch.setattr(cs, "_load_world_frozen", lambda a, r: world)
    monkeypatch.setattr(cs, "ROOT", tmp_path / "repo")
    monkeypatch.setattr(cs, "unlocked_by", lambda path, tag, root=None: ("ab" * 32, ""))
    monkeypatch.setattr(cs, "load_agent", lambda name, pin: SimpleNamespace(obs_rms=None))
    monkeypatch.setattr(NW, "upcast32", lambda x: x)
    from apexzero.costate import inputs as IN
    monkeypatch.setattr(IN, "flags", lambda w, rms: flags)
    monkeypatch.setattr(cs, "analyse_network", analyse)
    cs.main(["analyse", "--out", str(out)])
    d = json.loads((out / "analyse" / "decisions_dry.json").read_text())
    k3 = d["k3_trained"]
    assert k3["failed"] == bad and k3["not_run"] == [] and len(k3["networks"]) == 34
    assert k3["networks"][AN.GENERALISTS[0]] == {"pass": True, "reason": None}
    # the failing decision network: every decision-bearing record descriptive, with the reason
    n2 = d["networks"][g2]
    for k in ("C1", "C1-own", "C1-heads", "C1-pattern", "C1-limit", "W1", "C2", "C2-sw"):
        assert n2[k]["descriptive"] is True and n2[k]["label_scope"] == "descriptive", k
        assert n2[k]["descriptive_reason"] == bad[g2], k
    assert n2["C1"]["decision"] == "confirm" and n2["C1"]["stat"] == 0.7     # kept
    assert n2["C1-own"]["status"] == f"descriptive ({bad[g2]})"
    assert "descriptive" not in n2["value_accuracy"]
    assert n2["k3_trained"] == {"pass": False, "reason": bad[g2]}
    primary = n2["by_stratum"]["C2"]["interior"]                # its primary row too
    assert primary["descriptive"] is True and primary["descriptive_reason"] == bad[g2]
    assert "descriptive_reason" not in n2["by_stratum"]["C2"]["contact"]
    # its analysis raising is contained, not a crash
    assert d["networks"][g3s2]["error"].startswith("FloatingPointError: ")
    assert d["networks"][g3s2]["k3_trained"]["pass"] is False
    # missing from every decision it enters
    for a, name in (("G2", g2), ("G3s2", g3s2)):
        assert d["per_network"][a] == {"C1": "missing", "C2": "missing", "C2-sw": "missing",
                                       "excluded": bad[name]}
    assert d["per_network"]["G1"] == {"C1": "confirm", "C2": "missing", "C2-sw": "missing"}
    assert d["variants"]["v2"]["C1"]["seeds"]["G2"] == "missing"
    assert d["variants"]["v2"]["C1"]["seeds"]["G2s1"] == "confirm"
    excluded = sorted(n for n in (g2, g3s2, other))
    for k in ("C3-w", "C3-R2c"):
        assert d[k]["n"] == 28 and sorted(d[k]["k3_trained_excluded"]) == excluded, k
        assert set(excluded) <= set(d[k]["missing"]), k
    assert all(sorted(v["k3_trained_excluded"]) == excluded
               for v in d["C3-w_sensitivity"].values())
    # outside the decision set: marked, still described (contained)
    desc = d["descriptive_networks"]
    assert desc[other]["C1"]["descriptive_reason"] == bad[other]
    assert desc[other]["C1-heads"]["descriptive_reason"] == bad[other]
    assert desc[s1]["k3_trained"] == {"pass": False, "reason": bad[s1]}
    assert desc[s1]["C1"]["descriptive_reason"] == bad[s1]
    passing = next(n for n in AN.GENERALISTS if n not in decided and n not in bad)
    assert "k3_trained" not in desc[passing] and "descriptive_reason" not in desc[passing]["C1"]
    assert "k3_trained" not in desc[AN.SPECIALISTS[1]]


def _walk_keys(d):
    if isinstance(d, dict):
        for k, v in d.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(d, list):
        for v in d:
            yield from _walk_keys(v)


def test_a_dry_analyse_withholds_every_decision(tmp_path, monkeypatch):
    """Review finding: a non-confirmatory analyse computed and wrote every decision, so the
    analysis could be read many times before the run-once confirmatory one. Until a stamped
    confirmatory decisions file is committed (``unlocked_by``) it writes coverage only."""
    cs = _script()
    world, frozen = _toy_world()
    rec = {"frozen": frozen, "networks": {}}
    monkeypatch.setattr(cs, "_post_tag", lambda a, w: rec)
    monkeypatch.setattr(cs, "_load_world_frozen", lambda a, r: world)
    monkeypatch.setattr(cs, "ROOT", tmp_path / "repo")
    out = tmp_path / "o"
    cs.main(["analyse", "--out", str(out)])
    d = json.loads((out / "analyse" / "decisions_dry.json").read_text())
    assert d["decided"] is False and d["unlocked_by"] is None and "coverage" in d
    from apexzero.costate import analysis as AN
    assert {h: set(v) for h, v in d["coverage"]["strata"].items()} == {
        h: set(v) for h, v in AN.STRATA.items()}                # row counts, no statistic
    assert d["coverage"]["strata"]["C1"]["contact"] == int(world.tab["contact"].sum())
    banned = {"per_network", "variants", "C3-w", "decision", "label_at_scale", "verdict"}
    assert not banned & set(_walk_keys({k: v for k, v in d.items() if k != "provenance"}))
    monkeypatch.setattr(cs, "unlocked_by", lambda path, tag, root=None: ("ab" * 32, ""))
    cs.main(["analyse", "--out", str(out)])
    d = json.loads((out / "analyse" / "decisions_dry.json").read_text())
    assert d["decided"] is True and d["unlocked_by"] == "ab" * 32
    assert {"per_network", "variants", "C3-w"} <= set(d)


def test_unlocked_by_needs_a_committed_stamped_confirmatory_file(tmp_path):
    cs = _script()
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    path = root / "outputs" / "costate" / "analyse" / "decisions.json"
    path.parent.mkdir(parents=True)
    tag = {"object": "o" * 40, "commit": "c" * 40}
    prov = cs.Provenance(None, "analyse")
    prov.tag, prov.argv = tag, ["costate.py", "analyse", "--confirmatory"]
    cs._use_provenance(prov)
    cs._write(path, {"confirmatory": True, "decided": True, "per_network": {}, "variants": {},
                     "C3-w": {}})
    sha = cs.confirmatory_done(path, tag)
    assert sha and cs.confirmatory_done(path, {**tag, "commit": "d" * 40}) is None
    got, why = cs.unlocked_by(path, tag, root)
    assert got is None and "not committed" in why
    _git(root, "add", "-f", str(path.relative_to(root)))
    _git(root, "commit", "-q", "-m", "decisions")
    assert cs.unlocked_by(path, tag, root)[0] == sha
    prov.argv = ["costate.py", "analyse"]
    cs._write(path, {"confirmatory": True, "decided": True, "per_network": {}, "variants": {},
                     "C3-w": {}})
    assert cs.confirmatory_done(path, tag) is None          # not a --confirmatory run


def test_load_world_frozen_checks_the_record(tmp_path):
    """Review finding: the post-tag stages loaded world.npz unbound and never compared T's
    list or the folds with the record."""
    cs = _script()
    from apexzero.costate import inputs as IN
    from apexzero.costate import states as ST

    world, _ = _toy_world()
    tab = {k: world.tab[k] for k in ("key_id", "t", "R", "theta")}
    p = IN.save_world(tmp_path / "inputs" / "world.npz", IN.World(tab, []))
    uniq = np.unique(ST.geometry_keys(tab), axis=0)
    rec = {"outputs": {"inputs/world.npz": {"sha256": PR.sha256_file(p)}},
           "t_pairs": [[str(k), int(t)] for k, t in zip(tab["key_id"], tab["t"], strict=True)],
           "frozen": {"geometries": uniq.tolist(),
                      "fold_partition0": ST.geometry_folds(uniq[:, 0], uniq[:, 1], 0).tolist()}}
    args = SimpleNamespace(out=str(tmp_path))
    assert cs._load_world_frozen(args, rec).n == len(tab["t"])
    with pytest.raises(SystemExit, match="not the world the record binds"):
        cs._load_world_frozen(args, {**rec, "outputs": {"inputs/world.npz": {"sha256": "0"}}})
    with pytest.raises(SystemExit, match="list differs"):
        cs._load_world_frozen(args, {**rec, "t_pairs": rec["t_pairs"][1:]})
    bad = {**rec["frozen"], "fold_partition0": [0] * len(uniq)}
    with pytest.raises(SystemExit, match="folds differ"):
        cs._load_world_frozen(args, {**rec, "frozen": bad})
