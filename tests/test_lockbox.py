"""scripts/lockbox.py (LB-0, engineering): the protocol-v1 tag gate, every stage's refusal of
R_TEST names without it, the roots, the one-shot analyse, the rollout's equality with
``timed_laps``, the lap profile and sector closure, the width re-solve track, the OOD buffer
and the statistics.

No test solves an OCP, opens outputs/sealed/, or loads or drives an R_TEST circuit: the gate's
success path is checked on names only, with ``lockbox_guard`` replaced by a recorder. The
checkpoint tests need runs/ (local only) and the track tests data/ (fetched); each is skipped
without them."""

import ast
import importlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from apexzero.tracks.corners import R_TEST

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CKPT_RUN = "h4r-v1-realw-s12"           # completes Norisring on every run (H4R manifest)
CKPT = ROOT / "runs" / CKPT_RUN / "ckpt" / "000199884800.pkl"
HAVE_RUNS = CKPT.exists()
HAVE_TRACKS = (ROOT / "data" / "tracks" / "tumftm" / "Norisring.csv").exists()
needs_runs = pytest.mark.skipif(not HAVE_RUNS, reason="the H4R checkpoints are not here")
needs_tracks = pytest.mark.skipif(not HAVE_TRACKS, reason="the R-dev tracks are not here")
LOCKBOX_NAMES = ["Monza", "monza", "Spa-reversed", "Silverstone-mirrored", "Mexico City",
                 "Suzuka_reversed", "data/tracks/tumftm/Austin.csv"]

OBJ = "a" * 40
COMMIT = "b" * 40
HEAD = "c" * 40


@pytest.fixture(scope="module")
def lb():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("lockbox")
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture(autouse=True)
def _keep_priority(lb, monkeypatch):
    """main() lowers its own priority; the test process keeps its own."""
    monkeypatch.setattr(lb, "lower_priority", lambda *a, **k: None)


class FakeGit:
    """A stand-in for ``git_run``: an annotated protocol-v1 on origin and locally, HEAD
    descending from it and the metric code unchanged, unless a field says otherwise."""

    def __init__(self, events=None, **kw):
        self.remote = (f"{OBJ}\trefs/tags/protocol-v1\n"
                       f"{COMMIT}\trefs/tags/protocol-v1^{{}}\n")
        self.remote_rc, self.local, self.kind, self.commit = 0, OBJ, "tag", COMMIT
        self.ancestor, self.diff_clean, self.untracked = True, True, ""
        self.__dict__.update(kw)
        self.events = events if events is not None else []

    def __call__(self, *args):
        self.events.append(("git", args[0]))

        def cp(rc, out=""):
            return subprocess.CompletedProcess(["git", *args], rc, out, "")

        cmd = args[0]
        if cmd == "ls-remote":
            return cp(self.remote_rc, self.remote if self.remote_rc == 0 else "")
        if cmd == "rev-parse":
            if args[-1] == "refs/tags/protocol-v1":
                return cp(0 if self.local else 1, self.local or "")
            if args[-1].endswith("^{commit}"):
                return cp(0, self.commit)
            return cp(0, HEAD)
        if cmd == "cat-file":
            return cp(0, self.kind)
        if cmd == "merge-base":
            return cp(0 if self.ancestor else 1)
        if cmd == "diff":
            return cp(0 if self.diff_clean else 1)
        if cmd == "ls-files":
            return cp(0, self.untracked if "--others" in args else "")
        if cmd == "status":
            return cp(0, "")
        raise AssertionError(f"unexpected git call {args}")


NO_TAG = {"remote": ""}


# ------------------------------------------------------------------------- the tag gate


def test_tag_gate_passes_only_on_the_full_tag(lb, monkeypatch):
    monkeypatch.setattr(lb, "git_run", FakeGit())
    got = lb.tag_gate()
    assert got == {"tag": "protocol-v1", "tag_object": OBJ, "commit": COMMIT, "head": HEAD}


@pytest.mark.parametrize("change", [
    NO_TAG,                                                      # not on origin
    {"remote_rc": 128},                                          # origin unreachable
    {"remote": f"{COMMIT}\trefs/tags/protocol-v1\n"},            # lightweight tag
    {"remote": f"{OBJ}\trefs/tags/other/protocol-v1\n"
               f"{COMMIT}\trefs/tags/other/protocol-v1^{{}}\n"},  # a different ref
    {"local": ""},                                               # no local tag
    {"local": "d" * 40},                                         # local tag is another object
    {"kind": "commit"},                                          # not a tag object
    {"commit": "e" * 40},                                        # tag commit differs from origin
    {"ancestor": False},                                         # HEAD does not descend
    {"diff_clean": False},                                       # metric code changed
    {"untracked": "src/apexzero/new.py\n"},                      # untracked metric code
])
def test_tag_gate_refuses(lb, monkeypatch, change):
    monkeypatch.setattr(lb, "git_run", FakeGit(**change))
    with pytest.raises(PermissionError):
        lb.tag_gate()


def test_unsealed_guard_runs_only_behind_the_gate(lb, monkeypatch):
    """Names only: no geometry is loaded and nothing runs."""
    events = []

    def guard(name, unsealed=False):
        events.append(("guard", unsealed))

    monkeypatch.setattr(lb, "lockbox_guard", guard)
    monkeypatch.setattr(lb, "git_run", FakeGit(events, **NO_TAG))
    with pytest.raises(PermissionError):
        lb.clear(R_TEST, "lockbox")
    assert not [e for e in events if e[0] == "guard"]
    events.clear()
    monkeypatch.setattr(lb, "git_run", FakeGit(events))
    clr = lb.clear(R_TEST, "lockbox")
    guards = [i for i, e in enumerate(events) if e[0] == "guard"]
    gits = [i for i, e in enumerate(events) if e[0] == "git"]
    assert len(guards) == 4 * 12 and all(events[i][1] is True for i in guards)
    assert max(gits) < min(guards)                    # every git check came first
    assert clr.mode == "lockbox" and clr.tag["commit"] == COMMIT
    with pytest.raises(ValueError):                   # all 12, no subset
        lb.clear(R_TEST[:6], "lockbox")
    with pytest.raises(ValueError):
        lb.clear([*R_TEST[:11], R_TEST[0]], "lockbox")


def test_unsealed_true_appears_once_behind_tag_gate(lb):
    """The source has exactly one ``unsealed=`` keyword, in ``clear``, after ``tag_gate()``."""
    tree = ast.parse((SCRIPTS / "lockbox.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and any(k.arg == "unsealed" for k in n.keywords)]
    assert len(calls) == 1
    (call,) = calls
    clear = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "clear")
    assert clear.lineno <= call.lineno <= clear.end_lineno
    gate = [n for n in ast.walk(clear) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "tag_gate"]
    assert len(gate) == 1 and gate[0].lineno < call.lineno


# ------------------------------------------------------------------------- every stage refuses


@pytest.mark.parametrize("name", LOCKBOX_NAMES)
def test_dryrun_freeze_refuses_lockbox_names(lb, tmp_path, name):
    root = tmp_path / "dry"
    with pytest.raises(PermissionError):
        lb.main(["freeze", "--mode", "dryrun", "--root", str(root), "--circuits", "Norisring",
                 name, "--h4r-subjects", CKPT_RUN])
    assert not root.exists()


def test_dryrun_refuses_non_rdev_names(lb):
    for bad in (["IMS"], ["Norisring-mirrored"], ["Norisring", "Norisring"], []):
        with pytest.raises(ValueError):
            lb.clear(bad, "dryrun")


def _plant_freeze(root: Path, mode: str, circuits, tag=None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rec = {"mode": mode, "freeze_id": "f" * 32, "circuits": list(circuits),
           "variants": ["original", "mirrored", "reversed"], "subjects": [],
           "circuit_files_sha256": {}, "tag": tag, "label": "planted"}
    (root / "freeze.json").write_text(json.dumps(rec))


@pytest.mark.parametrize("stage", ["ocp", "laps", "analyse"])
@pytest.mark.parametrize("name", ["Monza", "Spa-reversed", "silverstone"])
def test_later_stages_refuse_planted_lockbox_names(lb, tmp_path, stage, name):
    root = tmp_path / "dry"
    _plant_freeze(root, "dryrun", ["Norisring", name])
    with pytest.raises(PermissionError):
        lb.main([stage, "--root", str(root)])
    assert not (root / "analysis.lock").exists() and not (root / "ocp").exists()


@pytest.mark.parametrize("stage", ["freeze", "ocp", "laps", "analyse"])
def test_every_stage_refuses_the_lockbox_without_the_tag(lb, tmp_path, monkeypatch, stage):
    events = []

    def guard(name, unsealed=False):
        events.append(("guard", unsealed))
        if unsealed:
            raise AssertionError("lockbox_guard(unsealed=True) ran without the tag")
        raise PermissionError(name)

    lockbox_root = tmp_path / "outputs" / "lockbox"
    monkeypatch.setattr(lb, "LOCKBOX_ROOT", lockbox_root)
    monkeypatch.setattr(lb, "lockbox_guard", guard)
    monkeypatch.setattr(lb, "git_run", FakeGit(events, **NO_TAG))
    _plant_freeze(lockbox_root, "lockbox", R_TEST, {"commit": COMMIT})
    argv = (["freeze", "--mode", "lockbox", "--subjects", str(tmp_path / "s.json")]
            if stage == "freeze" else [stage, "--root", str(lockbox_root)])
    with pytest.raises(PermissionError):
        lb.main(argv)
    assert ("git", "ls-remote") in events and not [e for e in events if e[0] == "guard"]
    assert not (lockbox_root / "analysis.lock").exists()


def test_lockbox_freeze_takes_no_circuit_list(lb, tmp_path, monkeypatch):
    monkeypatch.setattr(lb, "git_run", FakeGit(**NO_TAG))
    with pytest.raises(lb.CheckFailed):
        lb.main(["freeze", "--mode", "lockbox", "--circuits", "Norisring", "--subjects", "x"])


def test_modes_must_match_their_roots(lb, tmp_path, monkeypatch):
    lockbox_root = tmp_path / "outputs" / "lockbox"
    monkeypatch.setattr(lb, "LOCKBOX_ROOT", lockbox_root)
    monkeypatch.setattr(lb, "git_run", FakeGit())
    other = tmp_path / "elsewhere"
    _plant_freeze(other, "lockbox", R_TEST, {"commit": COMMIT})
    with pytest.raises(lb.CheckFailed):
        lb.load_freeze(other)
    _plant_freeze(lockbox_root, "dryrun", ["Norisring"])
    with pytest.raises(lb.CheckFailed):
        lb.load_freeze(lockbox_root)


def test_load_variant_admits_only_cleared_names(lb):
    names = frozenset({"Norisring", "Norisring-mirrored", "Norisring-reversed"})
    clr = lb.Clearance("dryrun", ("Norisring",), names)
    with pytest.raises(PermissionError):
        lb.load_variant("Monza", "original", clr)
    forged = lb.Clearance("dryrun", ("Monza",), frozenset({"Monza", "Monza-mirrored"}))
    with pytest.raises(PermissionError):                 # the guard still runs per name
        lb.load_variant("Monza", "mirrored", forged)
    no_tag = lb.Clearance("lockbox", ("Monza",), frozenset({"Monza"}), None)
    with pytest.raises(PermissionError):
        lb.load_variant("Monza", "original", no_tag)


# ------------------------------------------------------------------------- roots and files


def test_roots(lb, tmp_path, monkeypatch):
    with pytest.raises(lb.CheckFailed):
        lb.resolve_root("dryrun", None)
    for bad in (lb.LOCKBOX_ROOT, lb.LOCKBOX_ROOT / "x", lb.LOCKBOX_ROOT.parent,
                lb.SEALED, lb.SEALED / "x"):
        with pytest.raises(lb.CheckFailed):
            lb.resolve_root("dryrun", bad)
    assert lb.resolve_root("dryrun", lb.DRYRUN_PARENT / "x") == lb.DRYRUN_PARENT / "x"
    assert lb.resolve_root("lockbox", None) == lb.LOCKBOX_ROOT
    with pytest.raises(lb.CheckFailed):
        lb.resolve_root("lockbox", lb.DRYRUN_PARENT / "x")
    monkeypatch.setattr(lb, "SEALED", tmp_path / "sealed")
    with pytest.raises(PermissionError):
        lb.safe_path(tmp_path, "sealed", "a.pkl")
    with pytest.raises(PermissionError):
        lb.safe_path(tmp_path / "root", "..", "x.json")
    assert lb.safe_path(tmp_path, "ocp", "a.json") == tmp_path / "ocp" / "a.json"


def test_write_new_never_overwrites(lb, tmp_path):
    p = tmp_path / "a" / "b.json"
    sha = lb.write_json_new(p, {"x": float("nan"), "y": float("inf"), "z": np.float32(1.5)})
    assert sha == lb.sha256_file(p)
    assert json.loads(p.read_text()) == {"x": None, "y": "inf", "z": 1.5}
    with pytest.raises(lb.CheckFailed):
        lb.write_json_new(p, {})
    assert [lb.unclean(v) for v in (None, "inf", "-inf", 2)][1:] == [np.inf, -np.inf, 2.0]


def _fake_checkpoint(path: Path, timesteps: int = 7) -> str:
    raw = pickle.dumps({"timesteps": timesteps, "config": {}})
    path.write_bytes(raw)
    return lb_sha(raw)


def lb_sha(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def test_verify_subjects(lb, tmp_path, monkeypatch):
    ck = tmp_path / "a.pkl"
    sha = _fake_checkpoint(ck)
    good = {"name": "s1", "checkpoint": str(ck), "sha256": sha}
    (out,) = lb.verify_subjects([good])
    assert out["timesteps"] == 7 and out["checkpoint"] == str(ck)
    for bad in ([], [good, good], [dict(good, sha256="0" * 64)], [dict(good, name="../x")]):
        with pytest.raises(lb.CheckFailed):
            lb.verify_subjects(bad)
    monkeypatch.setattr(lb, "SEALED", tmp_path)
    with pytest.raises(PermissionError):
        lb.verify_subjects([good])


@needs_tracks
def test_freeze_writes_a_fresh_root_once(lb, tmp_path):
    ck = tmp_path / "a.pkl"
    sha = _fake_checkpoint(ck)
    spath = tmp_path / "subjects.json"
    spath.write_text(json.dumps({"subjects": [{"name": "s1", "checkpoint": str(ck),
                                               "sha256": sha}]}))
    root = tmp_path / "dry"
    lb.main(["freeze", "--mode", "dryrun", "--root", str(root), "--circuits", "Norisring",
             "--subjects", str(spath), "--runs", "4"])
    rec = json.loads((root / "freeze.json").read_text())
    assert rec["mode"] == "dryrun" and rec["label"].startswith("DRY RUN")
    assert rec["circuits"] == ["Norisring"] and rec["protocol"]["laps"]["runs"] == 4
    assert rec["tag"] is None and len(rec["freeze_id"]) == 32
    assert list(rec["circuit_files_sha256"]) == ["data/tracks/tumftm/Norisring.csv"]
    assert len(lb.ocp_tasks(rec)) == 9
    assert lb.ocp_key("Norisring", "reversed", -0.5) == "Norisring-reversed__w-0.5"
    with pytest.raises(lb.CheckFailed):                  # never into a used root
        lb.main(["freeze", "--mode", "dryrun", "--root", str(root), "--circuits", "Norisring",
                 "--subjects", str(spath)])


# ------------------------------------------------------------------------- analyse runs once


def test_analyse_refuses_existing_outputs_and_locks_once(lb, tmp_path):
    root = tmp_path / "dry"
    for f in ("analysis.lock", "analysis.json", "corners.csv"):
        _plant_freeze(root, "dryrun", ["Norisring"])
        (root / f).write_text("{}")
        with pytest.raises(lb.CheckFailed, match="runs once"):
            lb.main(["analyse", "--root", str(root)])
        (root / f).unlink()
    lb.acquire_lock(root, "f" * 32)
    with pytest.raises(lb.CheckFailed, match="runs once"):
        lb.acquire_lock(root, "f" * 32)


def test_analyse_checks_inputs_before_the_lock(lb, tmp_path):
    """Missing or foreign inputs stop analyse before it locks, so it can run after a fix."""
    root = tmp_path / "dry"
    _plant_freeze(root, "dryrun", ["Norisring"])
    rec = json.loads((root / "freeze.json").read_text())
    rec["protocol"] = {"widths": {"deltas_m": [0.0, -0.5, 0.5]}}
    (root / "freeze.json").write_text(json.dumps(rec))
    with pytest.raises(lb.CheckFailed, match="missing"):
        lb.main(["analyse", "--root", str(root)])
    assert not (root / "analysis.lock").exists()
    for key in (lb.ocp_key(*t) for t in lb.ocp_tasks(rec)):
        sha = lb.write_new(root / "ocp" / f"{key}.pkl", b"x")
        lb.write_json_new(root / "ocp" / f"{key}.json", {"freeze_id": "0" * 32,
                                                         "pkl_sha256": sha})
    with pytest.raises(lb.CheckFailed, match="another freeze"):
        lb.load_inputs(root, rec)
    key = lb.ocp_key("Norisring", "original", 0.0)
    (root / "ocp" / f"{key}.pkl").unlink()
    (root / "ocp" / f"{key}.pkl").write_bytes(b"y")
    with pytest.raises(lb.CheckFailed, match="does not match"):
        lb.load_inputs(root, rec)
    assert not (root / "analysis.lock").exists()


# ------------------------------------------------------------------------- laps and timing


def test_crossing_and_lap_profile_on_a_synthetic_trace(lb):
    L, dt, v = 1000.0, 0.05, 41.3
    t = (np.arange(2000) + 1) * dt
    s = (v * t).astype(np.float32)
    stations = np.r_[np.arange(0.0, L, 2.0), L]
    tau = lb.lap_profile(s, t, stations, L, 2, dt)
    t1, t2 = lb.crossing_time(s, t, L, dt), lb.crossing_time(s, t, 2 * L, dt)
    assert tau[0] == 0.0 and tau[-1] == t2 - t1
    assert np.all(np.diff(tau) > 0)
    assert np.allclose(tau, stations / v, atol=1e-3)
    assert lb.crossing_time(s, t, 1e9, dt) is None


def test_sector_dt_sums_to_the_lap_delta(lb):
    from apexzero.viz.sectors import Sector

    L = 1000.0
    secs = [Sector("C1", 100.0, 400.0, 250.0, 50.0, 1.0, [0], 150.0, 350.0),
            Sector("C2", 400.0, 900.0, 600.0, 80.0, -1.0, [1], 450.0, 800.0),
            Sector("C3", 900.0, 1100.0, 1000.0, 30.0, 1.0, [2], 950.0, 1050.0)]   # wraps
    stations = np.r_[np.arange(0.0, L, 2.0), L]
    rng = np.random.default_rng(0)
    tau = np.vstack([np.cumsum(np.r_[0.0, rng.uniform(0.04, 0.06, len(stations) - 1)]),
                     np.full(len(stations), np.nan)])
    t_ocp = np.cumsum(np.r_[0.0, rng.uniform(0.04, 0.05, len(stations) - 1)])
    dt = lb.sector_dt(stations, tau, t_ocp, secs, L)
    assert np.isclose(dt[0].sum(), (tau[0, -1] - t_ocp[-1]) - (tau[0, 0] - t_ocp[0]),
                      atol=1e-12)
    assert np.all(np.isnan(dt[1]))


def test_outcomes_and_causes(lb):
    T, R = 5, 3
    s = np.tile(np.arange(1, T + 1, dtype=np.float32)[:, None] * 100.0, (1, R))
    alive = np.ones((T, R), bool)
    term = np.zeros((T, R), bool)
    causes = np.zeros((T, R, 4), bool)
    alive[2:, 1] = False
    term[2, 1] = True
    causes[2, 1] = [False, True, True, False]       # slipped implies spun: slip comes first
    ro = lb.LapRollout(np.array([True, False, False]), np.array([1.0, np.nan, np.nan]),
                       s.max(axis=0), np.arange(1, T + 1) * 0.05, s, alive, term, causes,
                       None, None, None, 250.0, 2, 0.05)
    oc = lb.outcomes(ro)
    assert oc["crash"].tolist() == [False, True, False]
    assert oc["timeout"].tolist() == [False, False, True]
    assert oc["cause"][1] == "slip" and oc["term_lap"][1] == 2 and oc["term_s"][1] == 50.0


@needs_tracks
def test_rollout_matches_timed_laps_on_a_fixed_policy(lb):
    import jax.numpy as jnp

    from apexzero.env.adapters import reference_speed
    from apexzero.eval.laps import timed_laps

    def policy(obs):
        return jnp.stack([-0.8 * obs[:, 204], 0.3 + 0.0 * obs[:, 200]], axis=-1)

    clr = lb.clear(["Norisring"], "dryrun")
    tr = lb.load_variant("Norisring", "original", clr)
    v0 = float(reference_speed(tr)[0]) * 0.9
    a = timed_laps(policy, tr, v0, runs=4)
    b = lb.rollout_laps(policy, tr, v0, runs=4, obs_fn=lambda o: o * 2.0)
    assert np.array_equal(a.completed, b.completed)
    assert np.array_equal(a.lap_time, b.lap_time, equal_nan=True)
    assert np.array_equal(a.s_reached, b.s_reached)
    assert b.obs0.shape == (b.s.shape[0], 222) and b.s_pre0[0] == 0.0
    oc = lb.outcomes(b)
    assert np.array_equal(oc["crash"] | oc["timeout"], ~b.completed)


@needs_runs
@needs_tracks
def test_checkpoint_rollout_profile_buffer_and_d_ood(lb):
    """A real subject on Norisring (it completes): parity with timed_laps, the lap profile's
    exact closure, and a small OOD buffer whose own rows are at distance 0."""
    from apexzero.env.adapters import reference_speed
    from apexzero.interp.controls import d_ood

    man = json.loads((ROOT / "outputs" / "h4r" / "manifest.json").read_text())
    sha = man["networks"][f"runs/{CKPT_RUN}"]["sha256"]
    sub = {"name": CKPT_RUN, "checkpoint": str(CKPT.relative_to(ROOT)), "sha256": sha}
    policy, cfg, snap = lb.load_subject(sub)
    clr = lb.clear(["Norisring"], "dryrun")
    tr = lb.load_variant("Norisring", "original", clr)
    v0 = float(reference_speed(tr)[0]) * 0.9
    lp = {"runs": 4, "laps": 2, "seed": 0}
    ro = lb.rollout_laps(policy, tr, v0, runs=4, obs_fn=snap.obs_rms.normalize)
    lb.parity_check(policy, tr, v0, lp, ro)
    assert ro.completed.all()
    stations = np.r_[tr.s, ro.length]
    for r in range(4):
        tau = lb.lap_profile(ro.s[:, r], ro.t, stations, ro.length, 2, ro.dt)
        assert tau[-1] == ro.lap_time[r] and tau[0] == 0.0 and np.all(np.diff(tau) >= 0)
    ro.lap_time[0] += 1e-9
    with pytest.raises(lb.CheckFailed):
        lb.parity_check(policy, tr, v0, lp, ro)

    spec = dict(lb.PROTOCOL["ood"], levels=2, steps=40, n_obs=60, chunk=2)
    buf = lb.build_buffer(snap, cfg, spec, CKPT_RUN)
    assert buf.obs.shape == (60, 222) and set(np.unique(buf.lap)) <= {0, 1}
    assert buf.lolo_median_nn > 0
    near, far = d_ood(buf.obs, buf), d_ood(buf.obs + 1.0, buf)
    assert near.max() < 0.1 and far.min() > near.max()
    again = lb.build_buffer(snap, cfg, spec, CKPT_RUN)
    assert np.array_equal(buf.obs, again.obs) and buf.lolo_median_nn == again.lolo_median_nn


# ------------------------------------------------------------------------- geometry


def test_widen_moves_both_sides(lb):
    from apexzero.tracks.track import from_centreline

    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    xy = np.stack([300 * np.cos(th), 200 * np.sin(th)], 1)
    tr = from_centreline("oval", xy, np.full(200, 5.0), np.full(200, 6.0), ds=2.0)
    assert lb.widen(tr, 0.0) is tr
    w = lb.widen(tr, 0.5)
    assert np.allclose(w.w_left, tr.w_left + 0.5) and np.allclose(w.w_right, tr.w_right + 0.5)
    assert np.array_equal(w.kappa, tr.kappa) and np.array_equal(w.x, tr.x)
    assert w.meta["width_delta_m"] == 0.5 and "width_delta_m" not in tr.meta
    with pytest.raises(lb.CheckFailed):
        lb.widen(tr, -4.5)


@needs_tracks
def test_variants_are_named_and_mirrored(lb):
    clr = lb.clear(["Norisring"], "dryrun")
    o = lb.load_variant("Norisring", "original", clr)
    m = lb.load_variant("Norisring", "mirrored", clr)
    r = lb.load_variant("Norisring", "reversed", clr)
    assert (o.name, m.name, r.name) == ("Norisring", "Norisring-mirrored", "Norisring-reversed")
    assert np.isclose(m.length, o.length) and np.isclose(r.length, o.length, rtol=1e-3)
    assert np.isclose(np.sum(m.kappa), -np.sum(o.kappa), rtol=1e-3)


# ------------------------------------------------------------------------- statistics


def test_corner_types(lb):
    assert [lb.corner_type(v) for v in (119.9, 120.0, 200.0, 200.1, float("nan"))] == \
        ["slow", "medium", "medium", "fast", "unknown"]


def test_iqm(lb):
    from scipy.stats import trim_mean

    x = np.random.default_rng(1).normal(size=37)
    assert np.isclose(lb.iqm(x), trim_mean(x, 0.25))
    assert lb.iqm([1.0, 2.0, 3.0, np.inf], keep_inf=True) == 2.5
    assert lb.iqm([1.0, np.inf, np.inf, np.inf], keep_inf=True) == np.inf
    assert lb.iqm([1.0, 2.0, np.nan, np.inf]) == 1.5
    M = np.array([[0.05, 0.06, np.nan], [0.07, 0.05, 0.08], [0.06, np.inf, 0.07]])
    a = lb.iqm_stratified(np.where(np.isinf(M), np.nan, M), 200, 0)
    b = lb.iqm_stratified(M, 200, 0, keep_inf=True)
    assert a["n_missing"] == 2 and b["n_inf"] == 1 and b["n_missing"] == 1
    assert a["ci95"][0] <= a["iqm"] <= a["ci95"][1]
    assert a == lb.iqm_stratified(np.where(np.isinf(M), np.nan, M), 200, 0)
    assert lb._pct([1.0, 2.0, np.inf], 97.5) == np.inf and lb._pct([3.0, 1.0], 2.5) == 1.0


def test_family_summary(lb):
    def row(completion, g0, gn, gw):
        n = round(completion * 10)
        return {"completed": [True] * n + [False] * (10 - n), "completion": completion,
                "gap": {"+0.0": g0, "-0.5": gn, "+0.5": gw}}

    rows = [row(1.0, 0.05, 0.06, 0.04), row(0.6, 0.07, 0.08, 0.06), row(0.2, 0.09, None, 0.1)]
    out = lb.family_summary(rows, 0.5)
    assert out["circuits_completed"] == 2 and np.isclose(out["laps_completed_share"], 0.6)
    assert np.isclose(out["median_gap"], 0.06) and np.isclose(out["median_gap_narrow"], 0.07)
    assert np.isclose(out["median_gap_crash_inf"], 0.07)
    rows[1]["gap"]["+0.0"] = None                    # a missing reference: not smaller
    assert np.isnan(lb.family_summary(rows, 0.5)["median_gap_crash_inf"])


# ------------------------------------------------------------------------- analyse end to end


def _planted_reference(track, scale: float = 1.0, success: bool = True):
    """An OCPResult-shaped reference on the track's own stations: QSS speeds, t from ds/v."""
    from apexzero.env.adapters import reference_speed
    from apexzero.ocp.min_time import OCPResult

    n = round(track.length / 3.0)
    h = track.length / n
    s = np.arange(n) * h
    v = np.interp(s, track.s, reference_speed(track)) * scale
    seg = h / v
    t = np.r_[0.0, np.cumsum(seg)[:-1]]
    z = np.zeros((n, 7))
    z[:, 2] = v
    zeros = np.zeros(n)
    return OCPResult(s=s, z=z, u=np.zeros((n, 2)), t=t, lap_time=float(seg.sum()),
                     success=success, status="planted", iterations=0, solve_seconds=0.0,
                     kappa=zeros, w_left=zeros, w_right=zeros, dual_left=zeros,
                     dual_right=zeros, meta={"ds": h})


@needs_tracks
def test_analyse_end_to_end_on_a_planted_root(lb, tmp_path):
    """Planted references and laps on Norisring's three variants: one reference failed, one
    subject completes everything, one nothing. analyse runs to the end, its sectors close,
    its gaps are the planted ones, and it refuses a second run."""
    root = tmp_path / "dry"
    root.mkdir()
    csv_rel = "data/tracks/tumftm/Norisring.csv"
    fid = "f" * 32
    protocol = json.loads(json.dumps(lb.PROTOCOL))
    protocol["laps"]["runs"] = 4
    protocol["stats"]["n_boot"] = 50
    subs = [{"name": "good", "sha256": "1" * 64}, {"name": "bad", "sha256": "2" * 64}]
    rec = {"mode": "dryrun", "freeze_id": fid, "circuits": ["Norisring"],
           "variants": list(lb.VARIANTS), "protocol": protocol, "subjects": subs,
           "circuit_files_sha256": {csv_rel: lb.sha256_file(ROOT / csv_rel)}, "tag": None,
           "label": "DRY RUN: planted"}
    lb.write_json_new(root / "freeze.json", rec)
    clr = lb.clear(["Norisring"], "dryrun")
    failed = lb.ocp_key("Norisring", "reversed", 0.5)
    for c, v, d in lb.ocp_tasks(rec):
        key = lb.ocp_key(c, v, d)
        tr = lb.widen(lb.load_variant(c, v, clr), d)
        res = _planted_reference(tr, 1.0 + 0.02 * d, success=key != failed)
        sha = lb.write_new(root / "ocp" / f"{key}.pkl", pickle.dumps(res))
        lb.write_json_new(root / "ocp" / f"{key}.json", {
            "freeze_id": fid, "pkl_sha256": sha, "success": key != failed,
            "attempts": [{}] if key != failed else [{}, {}],
            "shooting_rel_err": 1e-5 if key != failed else float("nan")})
    nominal = {}
    for v in lb.VARIANTS:
        tr = lb.load_variant("Norisring", v, clr)
        res = _planted_reference(tr)
        nominal[v] = res.lap_time
        stations = np.r_[tr.s, tr.length]
        t_ocp = lb.ocp_time_at(res, stations)
        for sub in subs:
            ok = sub["name"] == "good"
            tau = np.full((4, len(stations)), np.nan)
            if ok:
                for r in range(4):
                    tau[r] = t_ocp * (1.05 + 0.001 * r)
            lap = tau[:, -1]
            npz = lb.npz_bytes(stations=stations, tau=tau, ood_s=stations[:-1:5].copy(),
                               ood_d=np.linspace(0.5, 3.0, len(stations[:-1:5]),
                                                 dtype=np.float32))
            name = tr.name
            sha = lb.write_new(root / "laps" / sub["name"] / f"{name}.npz", npz)
            lb.write_json_new(root / "laps" / sub["name"] / f"{name}.json", {
                "freeze_id": fid, "checkpoint_sha256": sub["sha256"],
                "completed": [ok] * 4, "lap_time_s": lap, "crash": [not ok] * 4,
                "timeout": [False] * 4, "term_s": [np.nan] * 4 if ok else [100.0] * 4,
                "cause": [""] * 4 if ok else ["corridor"] * 4,
                "npz": f"laps/{sub['name']}/{name}.npz", "npz_sha256": sha})
    lb.main(["analyse", "--root", str(root)])
    out = json.loads((root / "analysis.json").read_text())
    assert out["checks"]["references_converged"] == "8/9"
    assert out["checks"]["unconverged"] == [failed] and out["checks"]["retried"] == [failed]
    assert out["checks"]["sector_sum_closure_s"] < 1e-6
    good, bad = out["per_subject"]["good"], out["per_subject"]["bad"]
    assert good["all"]["circuits_completed"] == 3 and bad["all"]["circuits_completed"] == 0
    assert np.isclose(good["original"]["median_gap"], 1.0515 - 1.0, atol=1e-9)
    assert bad["all"]["median_gap"] is None and bad["all"]["median_gap_crash_inf"] == "inf"
    assert good["reversed"]["median_gap_wide"] is None          # its reference failed
    assert good["h1_as_written"] == {"laps_ok": True, "gap_ok": False}
    rows = {(r["subject"], r["name"]): r for r in out["rows"]}
    assert rows[("good", "Norisring")]["gap"]["+0.0"] == pytest.approx(0.0515)
    assert rows[("bad", "Norisring-mirrored")]["causes"]["corridor"] == 4
    corners = list(csv_rows(root / "corners.csv"))
    assert {r["subject"] for r in corners} == {"good", "bad"}
    for v in lb.VARIANTS:
        dts = [float(r["dt_median_s"]) for r in corners
               if r["subject"] == "good" and r["variant"] == v]
        # per sector the median of the 4 runs' (0.050 ... 0.053) x sector time: 0.0515 x it
        assert np.isclose(sum(dts), 0.0515 * nominal[v], rtol=1e-9)
    assert any(int(r["crashes_here"]) for r in corners if r["subject"] == "bad")
    assert out["iqm_gap"]["all"]["uncompleted_as_inf"]["n_inf"] == 3
    with pytest.raises(lb.CheckFailed, match="runs once"):
        lb.main(["analyse", "--root", str(root)])


def csv_rows(path: Path):
    import csv

    with open(path, newline="") as fh:
        yield from csv.DictReader(fh)
