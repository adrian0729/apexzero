"""scripts/lockbox.py (LB-0, engineering): the protocol-v1 tag gate, every stage's refusal of
R_TEST names without it, the roots, the one-shot analyse, the rollout's equality with
``timed_laps``, the lap profile and sector closure, the width re-solve track, the OOD buffer
and the statistics.

No test solves an OCP, opens outputs/sealed/, or loads or drives an R_TEST circuit: the gate's
success path is checked on names only, with ``lockbox_guard`` replaced by a recorder. The
lockbox-mode rehearsal swaps ``R_TEST`` for an R-dev circuit, points ``LOCKBOX_ROOT``, the
repository and the track directory into tmp_path, and runs against a scratch git repo with a
real annotated tag; the solver and the checkpoint are fakes. The
checkpoint tests need runs/ (local only) and the track tests data/ (fetched); each is skipped
without them."""

import ast
import importlib
import json
import os
import pickle
import shutil
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
        self.tracked, self.blobs = True, {}          # ls-files --error-unmatch; <commit>:<path>
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
            if ":" in args[-1]:
                blob = self.blobs.get(args[-1].split(":", 1)[1])
                return cp(0 if blob else 128, blob or "")
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
            if "--error-unmatch" in args:
                return cp(0 if self.tracked else 1)
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


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
SUBJECTS_REL = "docs/lockbox-subjects.json"


class ScratchRepo:
    """A bare ``origin`` and a clone of it in tmp_path, with real git and no user config: a
    metric-code file and a committed subjects file. ``lb.ROOT`` points at the clone."""

    def __init__(self, tmp_path: Path, monkeypatch, lb, subjects: bytes | None = None):
        cfg = tmp_path / "gitconfig"
        cfg.write_text("")
        for k, v in {"GIT_CONFIG_GLOBAL": str(cfg), "GIT_CONFIG_NOSYSTEM": "1",
                     "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                     "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}.items():
            monkeypatch.setenv(k, v)
        self.origin, self.clone = tmp_path / "origin.git", tmp_path / "clone"
        subprocess.run(["git", "init", "-q", "--bare", str(self.origin)], check=True)
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.clone)], check=True,
                       capture_output=True)
        (self.clone / "scripts").mkdir()
        (self.clone / "scripts" / "lockbox.py").write_text("# metric code\n")
        (self.clone / "docs").mkdir()
        (self.clone / "uv.lock").write_text("# lock\n")
        (self.clone / "pyproject.toml").write_text("# project\n")
        self.subjects = subjects or json.dumps(
            {"subjects": [{"name": "s1", "checkpoint": "runs/s1/a.pkl",
                           "sha256": "0" * 64}]}).encode()
        (self.clone / SUBJECTS_REL).write_bytes(self.subjects)
        self.git("add", "-A")
        self.git("commit", "-qm", "first")
        self.git("tag", "-a", "protocol-v10", "-m", "a different tag")
        (self.clone / "notes.txt").write_text("second\n")
        self.git("add", "-A")
        self.git("commit", "-qm", "second")
        self.git("push", "-q", "origin", "HEAD:refs/heads/main", "--tags")
        monkeypatch.setattr(lb, "ROOT", self.clone)

    def git(self, *args) -> str:
        return subprocess.run(["git", *args], cwd=self.clone, check=True, capture_output=True,
                              text=True).stdout.strip()

    def tag(self, annotated: bool = True) -> None:
        self.git("tag", *(["-a", "-m", "the lockbox"] if annotated else []), "protocol-v1")
        self.git("push", "-q", "origin", "refs/tags/protocol-v1")


@needs_git
def test_tag_gate_passes_on_a_real_annotated_tag(lb, tmp_path, monkeypatch):
    """Real git, not FakeGit: ``ls-remote --tags origin protocol-v1`` prints no peeled line,
    so the gate must name both refs in full. protocol-v10 must not stand in for protocol-v1."""
    repo = ScratchRepo(tmp_path, monkeypatch, lb)
    with pytest.raises(PermissionError, match="not on origin"):   # only protocol-v10 exists
        lb.tag_gate()
    repo.tag()
    got = lb.tag_gate()
    assert got["tag_object"] == repo.git("rev-parse", "refs/tags/protocol-v1")
    assert got["commit"] == repo.git("rev-parse", "HEAD") == got["head"]
    assert got["tag_object"] != got["commit"]
    (repo.clone / "scripts" / "lockbox.py").write_text("# changed\n")
    with pytest.raises(PermissionError, match="differs"):
        lb.tag_gate()


@needs_git
def test_tag_gate_refuses_a_real_lightweight_tag(lb, tmp_path, monkeypatch):
    repo = ScratchRepo(tmp_path, monkeypatch, lb)
    repo.tag(annotated=False)
    with pytest.raises(PermissionError, match="not an annotated tag"):
        lb.tag_gate()


@needs_git
def test_lockbox_subjects_file_is_the_tagged_blob(lb, tmp_path, monkeypatch):
    """The file freeze reads is the file git checked: inside the repository, not a symlink,
    tracked, and byte-identical to the tag commit's blob at the same path."""
    repo = ScratchRepo(tmp_path, monkeypatch, lb)
    repo.tag()
    commit = lb.tag_gate()["commit"]
    path = repo.clone / SUBJECTS_REL
    r, raw = lb.tagged_subjects_file(path, commit)
    assert (r, raw) == (SUBJECTS_REL, repo.subjects)
    subs = lb.subjects_from_file(raw, r)
    assert subs[0]["source"] == SUBJECTS_REL and subs[0]["name"] == "s1"
    monkeypatch.chdir(repo.clone)                        # relative to the clone: the same file
    assert lb.tagged_subjects_file(SUBJECTS_REL, commit) == (SUBJECTS_REL, repo.subjects)

    elsewhere = tmp_path / "elsewhere"                   # same relative path, another tree
    (elsewhere / "docs").mkdir(parents=True)
    (elsewhere / SUBJECTS_REL).write_bytes(repo.subjects)
    monkeypatch.chdir(elsewhere)
    with pytest.raises(lb.CheckFailed, match="inside the repository"):
        lb.tagged_subjects_file(SUBJECTS_REL, commit)
    with pytest.raises(lb.CheckFailed, match="inside the repository"):
        lb.tagged_subjects_file(tmp_path / "elsewhere" / "docs" / "lockbox-subjects.json",
                                commit)
    link = repo.clone / "docs" / "link.json"
    link.symlink_to(path)
    with pytest.raises(lb.CheckFailed, match="symlink"):
        lb.tagged_subjects_file(link, commit)
    new = repo.clone / "docs" / "new.json"               # untracked, and not in the tag
    new.write_bytes(repo.subjects)
    with pytest.raises(lb.CheckFailed, match="byte-identical"):
        lb.tagged_subjects_file(new, commit)
    path.write_bytes(repo.subjects.replace(b"s1", b"s2"))  # changed since the tag
    with pytest.raises(lb.CheckFailed, match="byte-identical"):
        lb.tagged_subjects_file(path, commit)


def test_lockbox_subjects_file_checks_with_fake_git(lb, tmp_path, monkeypatch):
    monkeypatch.setattr(lb, "ROOT", tmp_path)
    raw = b'{"subjects": []}'
    (tmp_path / "s.json").write_bytes(raw)
    blob = lb.git_blob_id(raw)
    monkeypatch.setattr(lb, "git_run", FakeGit(blobs={"s.json": blob}))
    assert lb.tagged_subjects_file(tmp_path / "s.json", COMMIT) == ("s.json", raw)
    for change in ({"tracked": False}, {"blobs": {}}, {"blobs": {"s.json": "0" * 40}}):
        monkeypatch.setattr(lb, "git_run", FakeGit(**({"blobs": {"s.json": blob}} | change)))
        with pytest.raises(lb.CheckFailed):
            lb.tagged_subjects_file(tmp_path / "s.json", COMMIT)


def test_git_blob_id_matches_git(lb, tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    raw = b"{\r\n \"a\": 1}\n\x00"
    (tmp_path / "f").write_bytes(raw)
    out = subprocess.run(["git", "hash-object", "--no-filters", str(tmp_path / "f")],
                         capture_output=True, text=True, check=True, env=dict(os.environ))
    assert lb.git_blob_id(raw) == out.stdout.strip()


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


PLANTED_SUBJECT = {"name": "s1", "checkpoint": "runs/s1/agent.pkl", "sha256": "0" * 64}


def _plant_freeze(root: Path, mode: str, circuits, tag=None, **change) -> None:
    """A freeze.json as freeze writes it, except that no file is hashed (placeholder sha256s,
    so no test reads a circuit file it does not need)."""
    root.mkdir(parents=True, exist_ok=True)
    rec = {"mode": mode, "freeze_id": "f" * 32, "circuits": list(circuits),
           "variants": ["original", "mirrored", "reversed"], "subjects": [PLANTED_SUBJECT],
           "circuit_files_sha256": {f"data/tracks/tumftm/{c}.csv": "0" * 64 for c in circuits},
           "tag": tag, "label": "planted", **change}
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


@pytest.mark.parametrize("stage", ["ocp", "laps", "analyse"])
@pytest.mark.parametrize("change", [
    {"circuit_files_sha256": {"outputs/sealed/x.csv": "0" * 64}},              # a planted key
    {"circuit_files_sha256": {"data/tracks/tumftm/Norisring.csv": "0" * 64,
                              "outputs/sealed/x.csv": "0" * 64}},              # one extra key
    {"circuit_files_sha256": {}},                                              # one missing
    {"circuit_files_sha256": ["data/tracks/tumftm/Norisring.csv"]},            # not a mapping
    {"subjects": []},
    {"subjects": [dict(PLANTED_SUBJECT, name="../s1")]},
    {"subjects": [PLANTED_SUBJECT, PLANTED_SUBJECT]},
    {"subjects": [{k: v for k, v in PLANTED_SUBJECT.items() if k != "checkpoint"}]},
    {"subjects": [dict(PLANTED_SUBJECT, checkpoint="outputs/sealed/s1.pkl")]},
])
def test_later_stages_recheck_freeze_paths_before_reading(lb, tmp_path, monkeypatch, stage,
                                                         change):
    """A hand-edited freeze.json cannot point ocp, laps or analyse at another file: the
    circuit-file keys and the subjects' paths are checked again before any file is read."""
    reads = []
    monkeypatch.setattr(lb, "sha256_file", lambda p: reads.append(p) or "0" * 64)
    root = tmp_path / "dry"
    _plant_freeze(root, "dryrun", ["Norisring"], **change)
    with pytest.raises((lb.CheckFailed, PermissionError)):
        lb.main([stage, "--root", str(root)])
    assert not reads
    assert not (root / "analysis.lock").exists() and not (root / "ocp").exists()


def test_load_freeze_accepts_what_freeze_writes(lb, tmp_path):
    root = tmp_path / "dry"
    _plant_freeze(root, "dryrun", ["Norisring", "Hockenheim"])
    rec, clr = lb.load_freeze(root)
    assert clr.circuits == ("Norisring", "Hockenheim") and rec["subjects"] == [PLANTED_SUBJECT]


def test_load_subject_refuses_a_sealed_checkpoint(lb, tmp_path, monkeypatch):
    monkeypatch.setattr(lb, "SEALED", tmp_path)
    ck = tmp_path / "a.pkl"
    sha = _fake_checkpoint(ck)
    with pytest.raises(PermissionError):
        lb.load_subject({"name": "s1", "checkpoint": str(ck), "sha256": sha})


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
    assert rec["subjects_file"] == {"source": "(outside the repository) subjects.json",
                                    "sha256": lb.sha256_file(spath)}
    assert rec["subjects"][0]["source"] == rec["subjects_file"]["source"]
    lb.load_freeze(root)                                  # later stages accept it
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


@needs_tracks
def test_analyse_checks_inputs_before_the_lock(lb, tmp_path):
    """Missing or foreign inputs stop analyse before it locks, so it can run after a fix."""
    root = tmp_path / "dry"
    _plant_freeze(root, "dryrun", ["Norisring"])
    rec = json.loads((root / "freeze.json").read_text())
    rec["protocol"] = {"widths": {"deltas_m": [0.0, -0.5, 0.5]}}
    csv_rel = "data/tracks/tumftm/Norisring.csv"
    rec["circuit_files_sha256"] = {csv_rel: lb.sha256_file(ROOT / csv_rel)}
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


def test_lap_state_interpolates_where_crossing_time_does(lb):
    """n (and v_x) at the stations of a lap: exact for a quantity linear in time, including a
    station crossed on the very first sample (the start value stands in before it)."""
    L, dt, v = 1000.0, 0.05, 41.3
    t = (np.arange(2000) + 1) * dt
    s = (v * t).astype(np.float32)
    a, b = 0.3, -1.5
    x = a * t + b
    stations = np.r_[0.5, np.arange(2.0, L, 97.0), L]
    for lap in (1, 2):
        got = lb.lap_state(s, x, b, stations, L, lap)
        want = a * ((lap - 1) * L + stations) / v + b
        assert np.allclose(got, want, atol=2e-3)
        # the same crossing as the lap profile's: n at the station is n at its crossing time
        tau = lb.lap_profile(s, t, stations, L, lap, dt)
        t0 = lb.crossing_time(s, t, (lap - 1) * L, dt)
        assert np.allclose(got, a * (tau + t0) + b, atol=2e-3)
    with pytest.raises(ValueError):
        lb.lap_state(s[:100], x[:100], b, stations, L, 2)


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
    assert b.n.shape == b.vx.shape == b.s.shape and b.n_start.shape == b.vx_start.shape == (4,)
    assert b.n_start[0] == 0.0 and np.allclose(b.vx_start[0], v0, rtol=1e-6)
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


CSV_REL = "data/tracks/tumftm/Norisring.csv"
PLANTED_SUBJECTS = [{"name": "good", "checkpoint": "runs/good/agent.pkl", "sha256": "1" * 64},
                    {"name": "bad", "checkpoint": "runs/bad/agent.pkl", "sha256": "2" * 64}]


def _plant_norisring_freeze(lb, root: Path):
    """A dry-run freeze.json for Norisring and two subjects; returns (rec, clearance)."""
    root.mkdir()
    protocol = json.loads(json.dumps(lb.PROTOCOL))
    protocol["laps"]["runs"] = 4
    protocol["stats"]["n_boot"] = 50
    rec = {"mode": "dryrun", "freeze_id": "f" * 32, "circuits": ["Norisring"],
           "variants": list(lb.VARIANTS), "protocol": protocol, "subjects": PLANTED_SUBJECTS,
           "circuit_files_sha256": {CSV_REL: lb.sha256_file(ROOT / CSV_REL)}, "tag": None,
           "label": "DRY RUN: planted"}
    lb.write_json_new(root / "freeze.json", rec)
    return rec, lb.clear(["Norisring"], "dryrun")


def _plant_refs(lb, root: Path, rec: dict, clr, failed=(), shooting=None) -> None:
    """Planted references: ``failed`` keys did not converge; ``shooting`` maps a key to its
    shooting error (default 1e-5)."""
    for c, v, d in lb.ocp_tasks(rec):
        key = lb.ocp_key(c, v, d)
        ok = key not in failed
        tr = lb.widen(lb.load_variant(c, v, clr), d)
        res = _planted_reference(tr, 1.0 + 0.02 * d, success=ok)
        sha = lb.write_new(root / "ocp" / f"{key}.pkl", pickle.dumps(res))
        lb.write_json_new(root / "ocp" / f"{key}.json", {
            "freeze_id": rec["freeze_id"], "pkl_sha256": sha, "success": ok,
            "attempts": [{}] if ok else [{}, {}],
            "shooting_rel_err": (shooting or {}).get(key, 1e-5) if ok else float("nan")})


def _plant_laps(lb, root: Path, rec: dict, clr) -> dict:
    """Planted laps: "good" completes every run at 1.05 x the nominal planted reference, "bad"
    none. Returns the nominal lap time of each variant."""
    nominal = {}
    for v in lb.VARIANTS:
        tr = lb.load_variant("Norisring", v, clr)
        res = _planted_reference(tr)
        nominal[v] = res.lap_time
        stations = np.r_[tr.s, tr.length]
        t_ocp = lb.ocp_time_at(res, stations)
        for sub in rec["subjects"]:
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
                "freeze_id": rec["freeze_id"], "checkpoint_sha256": sub["sha256"],
                "completed": [ok] * 4, "lap_time_s": lap, "crash": [not ok] * 4,
                "timeout": [False] * 4, "term_s": [np.nan] * 4 if ok else [100.0] * 4,
                "cause": [""] * 4 if ok else ["corridor"] * 4,
                "npz": f"laps/{sub['name']}/{name}.npz", "npz_sha256": sha})
    return nominal


@needs_tracks
def test_analyse_end_to_end_on_a_planted_root(lb, tmp_path):
    """Planted references and laps on Norisring's three variants: one reference failed, one
    subject completes everything, one nothing. analyse runs to the end, its sectors close,
    its gaps are the planted ones, and it refuses a second run."""
    root = tmp_path / "dry"
    rec, clr = _plant_norisring_freeze(lb, root)
    failed = lb.ocp_key("Norisring", "reversed", 0.5)
    _plant_refs(lb, root, rec, clr, failed=(failed,))
    nominal = _plant_laps(lb, root, rec, clr)
    lb.main(["analyse", "--root", str(root)])
    out = json.loads((root / "analysis.json").read_text())
    assert out["checks"]["references_converged"] == "8/9"
    assert out["checks"]["unconverged"] == [failed] and out["checks"]["retried"] == [failed]
    assert out["checks"]["sector_sum_closure_s"] < 1e-6
    assert out["checks"]["shooting_failed"]["n"] == 0
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


def _fake_ocp(monkeypatch, fail=()):
    """Fast stand-ins for the solver and the shooting check (the planted reference of
    ``_planted_reference``); ``fail`` maps a (name, width delta) to the exception to raise."""
    import types

    from apexzero.ocp import check, min_time

    def solve(track, vehicle, ds, margin, tol, max_iter, init=None):
        d = track.meta.get("width_delta_m", 0.0)
        if (track.name, d) in fail:
            raise fail[(track.name, d)]
        return _planted_reference(track, 1.0 + 0.02 * d)

    monkeypatch.setattr(min_time, "solve_min_time", solve)
    monkeypatch.setattr(check, "shooting_check", lambda res, veh: types.SimpleNamespace(
        rel_lap_error=1e-5, max_speed_error=0.1))


def _serial_pool(monkeypatch, lb):
    """run_pool in this process (a spawned worker would not see the fakes)."""
    monkeypatch.setattr(lb, "run_pool", lambda fn, tasks, workers: [fn(*t) for t in tasks])


@needs_tracks
def test_a_raising_solve_fails_its_reference_only(lb, tmp_path, monkeypatch):
    """solve_min_time raising, and a widening that leaves no corridor, each write the
    reference's normal record (success false, the error text, a pickle of None); an interrupt
    is not swallowed."""
    root = tmp_path / "dry"
    rec, clr = _plant_norisring_freeze(lb, root)
    _fake_ocp(monkeypatch, fail={("Norisring", 0.0): RuntimeError("IPOPT crashed"),
                                 ("Norisring-mirrored", 0.0): KeyboardInterrupt()})
    out = lb.solve_task(str(root), rec, clr, "Norisring", "original", 0.0)
    assert out["converged"] is False and "IPOPT crashed" in out["error"]
    r = lb.ocp_done(root, lb.ocp_key("Norisring", "original", 0.0))   # its pickle is checked
    assert r["success"] is False and r["status"] == "error" and r["attempts"] == []
    assert "IPOPT crashed" in r["error"] and r["freeze_id"] == rec["freeze_id"]
    assert pickle.loads((root / "ocp" / "Norisring__w+0.0.pkl").read_bytes()) is None
    assert r["lap_time_s"] is None and r["shooting_rel_err"] is None
    assert lb.shooting_ok(r) is False

    out = lb.solve_task(str(root), rec, clr, "Norisring", "original", -50.0)     # no corridor
    assert out["converged"] is False and "no corridor" in out["error"]
    assert pickle.loads((root / "ocp" / "Norisring__w-50.0.pkl").read_bytes()) is None

    with pytest.raises(KeyboardInterrupt):
        lb.solve_task(str(root), rec, clr, "Norisring", "mirrored", 0.0)
    assert not (root / "ocp" / "Norisring-mirrored__w+0.0.json").exists()
    ok = lb.solve_task(str(root), rec, clr, "Norisring", "reversed", 0.0)         # a normal one
    assert ok["converged"] is True and ok["error"] is None


@needs_tracks
def test_a_pickle_left_without_its_record_is_solved_again(lb, tmp_path, monkeypatch):
    """A solve killed between its pickle and its record leaves an orphan pickle: the key is
    not done, and solving it again replaces the orphan instead of refusing it; a recorded
    reference is still never overwritten."""
    root = tmp_path / "dry"
    rec, clr = _plant_norisring_freeze(lb, root)
    _fake_ocp(monkeypatch)
    key = lb.ocp_key("Norisring", "original", 0.0)
    (root / "ocp").mkdir(parents=True, exist_ok=True)
    (root / "ocp" / f"{key}.pkl").write_bytes(b"half-written")
    assert lb.ocp_done(root, key) is None
    out = lb.solve_task(str(root), rec, clr, "Norisring", "original", 0.0)
    assert out["converged"] is True
    assert lb.ocp_done(root, key)["success"] is True      # the new pickle matches its record
    with pytest.raises(lb.CheckFailed, match="never overwrites"):
        lb.solve_task(str(root), rec, clr, "Norisring", "original", 0.0)


@needs_tracks
def test_analyse_runs_to_the_end_after_a_raising_solve(lb, tmp_path, monkeypatch):
    """The ocp stage goes on past a nominal and a width reference that raise, laps are planted,
    and analyse runs to the end with exactly those gaps missing."""
    root = tmp_path / "dry"
    rec, clr = _plant_norisring_freeze(lb, root)
    _fake_ocp(monkeypatch, fail={("Norisring", 0.0): RuntimeError("IPOPT crashed"),
                                 ("Norisring-reversed", 0.5): ValueError("bad grid")})
    _serial_pool(monkeypatch, lb)
    lb.main(["ocp", "--root", str(root)])
    assert len(list((root / "ocp").glob("*.json"))) == 9
    _plant_laps(lb, root, rec, clr)
    lb.main(["analyse", "--root", str(root)])
    out = json.loads((root / "analysis.json").read_text())
    assert out["checks"]["references_converged"] == "7/9"
    assert out["checks"]["unconverged"] == ["Norisring__w+0.0", "Norisring-reversed__w+0.5"]
    assert out["checks"]["shooting_failed"]["n"] == 0
    rows = {(r["subject"], r["name"]): r for r in out["rows"]}
    nom = rows[("good", "Norisring")]["gap"]
    assert nom["+0.0"] is None and nom["-0.5"] is not None and nom["+0.5"] is not None
    rev = rows[("good", "Norisring-reversed")]["gap"]
    assert rev["+0.5"] is None and rev["+0.0"] is not None and rev["-0.5"] is not None
    assert rows[("good", "Norisring-mirrored")]["gap"]["+0.0"] == pytest.approx(0.0515)
    assert out["per_subject"]["good"]["original"]["median_gap"] is None
    corners = list(csv_rows(root / "corners.csv"))
    assert {r["type"] for r in corners if r["variant"] == "original"} == {"unknown"}
    assert out["checks"]["mirror_ocp_lap_rel_diff"]["Norisring"] is None


@needs_tracks
def test_analyse_runs_to_the_end_with_a_failed_nominal_reference(lb, tmp_path):
    """A nominal reference that did not converge: its variant's gaps and sector times are
    missing, the other variants' stand, and analyse finishes."""
    root = tmp_path / "dry"
    rec, clr = _plant_norisring_freeze(lb, root)
    bad = lb.ocp_key("Norisring", "mirrored", 0.0)
    _plant_refs(lb, root, rec, clr, failed=(bad,))
    nominal = _plant_laps(lb, root, rec, clr)
    lb.main(["analyse", "--root", str(root)])
    out = json.loads((root / "analysis.json").read_text())
    assert out["checks"]["unconverged"] == [bad]
    rows = {(r["subject"], r["name"]): r for r in out["rows"]}
    assert rows[("good", "Norisring-mirrored")]["gap"]["+0.0"] is None
    assert rows[("good", "Norisring-mirrored")]["gap"]["+0.5"] is not None
    assert rows[("good", "Norisring")]["gap"]["+0.0"] == pytest.approx(0.0515)
    corners = list(csv_rows(root / "corners.csv"))
    assert all(r["dt_median_s"] == "" and r["type"] == "unknown"
               for r in corners if r["variant"] == "mirrored" and r["subject"] == "good")
    dts = [float(r["dt_median_s"]) for r in corners
           if r["variant"] == "original" and r["subject"] == "good"]
    assert np.isclose(sum(dts), 0.0515 * nominal["original"], rtol=1e-9)
    assert out["checks"]["mirror_ocp_lap_rel_diff"]["Norisring"] is None


@needs_tracks
def test_a_reference_at_the_shooting_limit_is_failed(lb, tmp_path):
    """|shooting error| >= P2_SHOOTING_LIMIT: the reference counts as failed for the gap (the
    limit itself fails; just under it stands), and analysis.json counts them."""
    lim = lb.P2_SHOOTING_LIMIT
    root = tmp_path / "dry"
    rec, clr = _plant_norisring_freeze(lb, root)
    over, edge, under = ("Norisring-reversed__w+0.0", "Norisring__w+0.5", "Norisring__w-0.5")
    _plant_refs(lb, root, rec, clr, shooting={over: -2 * lim, edge: lim, under: lim * 0.999})
    _plant_laps(lb, root, rec, clr)
    lb.main(["analyse", "--root", str(root)])
    out = json.loads((root / "analysis.json").read_text())
    sf = out["checks"]["shooting_failed"]
    assert sf["n"] == 2 and sorted(sf["keys"]) == sorted([over, edge]) and sf["limit"] == lim
    assert out["checks"]["references_converged"] == "9/9" and out["checks"]["references_failed"] == 2
    rows = {(r["subject"], r["name"]): r for r in out["rows"]}
    assert rows[("good", "Norisring-reversed")]["gap"]["+0.0"] is None
    assert rows[("good", "Norisring")]["gap"]["+0.5"] is None
    assert rows[("good", "Norisring")]["gap"]["-0.5"] is not None
    assert rows[("good", "Norisring")]["gap"]["+0.0"] == pytest.approx(0.0515)
    assert lb.shooting_ok({"shooting_rel_err": lim * 0.999}) is True
    assert lb.shooting_ok({"shooting_rel_err": -lim}) is False
    assert lb.shooting_ok({"shooting_rel_err": None}) is False


@needs_tracks
def test_ood_buffer_generator_has_real_widths_as_train_dr_builds_it(lb, monkeypatch):
    """The buffer's levels come from fit_grammar(R-dev, real_widths=True), the construction of
    scripts/train_dr.py --real-widths (no --real-widths-from, no stat widths, no tilt), and
    differ from the width-less generator's."""
    import jax

    from apexzero.tracks import generator
    from apexzero.tracks.corners import R_DEV
    from apexzero.tracks.levels import build, make_levels
    from apexzero.tracks.track import load_tumftm

    spec = dict(lb.PROTOCOL["ood"], levels=2)
    assert spec["real_widths"] is True and "real_widths" in spec["generator"]
    calls, real_fit = [], generator.fit_grammar
    monkeypatch.setattr(generator, "fit_grammar",
                        lambda *a, **k: calls.append((a, k)) or real_fit(*a, **k))
    tracks, x0 = lb.ood_levels(spec)
    ((args, kw),) = calls
    assert len(args) == 1 and len(args[0]) == len(R_DEV) and kw == {"real_widths": True}

    dev = [load_tumftm(ROOT / "data" / "tracks" / "tumftm" / f"{n}.csv") for n in R_DEV]
    lv = make_levels(jax.random.PRNGKey(spec["root_seed"]), spec["stream"], n=2,
                     randomize_car=False)

    def levels(g):
        return build(lv, g)[0], build(lv, g)[2]

    # train_dr.py: fit_grammar(dev, real_widths=args.real_widths, width_tracks=None)
    want, want_x0 = levels(real_fit(dev, real_widths=True, width_tracks=None))
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(tracks),
                                                    jax.tree.leaves(want), strict=True))
    assert np.array_equal(x0, want_x0)
    plain, _ = levels(real_fit(dev))
    assert not all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(tracks),
                                                        jax.tree.leaves(plain), strict=True))
    src = (SCRIPTS / "train_dr.py").read_text()
    assert "fit_grammar(dev, real_widths=args.real_widths, width_tracks=width_tracks)" in src


@pytest.mark.skipif(not (ROOT / "runs" / "e0-s50" / "config.json").exists(),
                    reason="the E0 runs are not here")
def test_the_subjects_trained_with_real_widths_and_nothing_else(lb):
    a = json.loads((ROOT / "runs" / "e0-s50" / "config.json").read_text())["args"]
    assert a["real_widths"] is True and a["real_widths_from"] == ""
    assert a["stat_widths"] is False and a["stat_widths_from"] == "" and a["tilt_fraction"] == 0.0


def test_protocol_constants_are_pinned(lb):
    """The values the lockbox run freezes; a dry-run setting must not be left in."""
    P = lb.PROTOCOL
    assert {k: P["laps"][k] for k in ("runs", "laps", "start_fraction", "seed",
                                      "completion_min")} == {
        "runs": 10, "laps": 2, "start_fraction": 0.9, "seed": 0, "completion_min": 0.5}
    assert {k: P["ocp"][k] for k in ("ds", "margin", "tol", "max_iter")} == {
        "ds": 3.0, "margin": 1.0, "tol": 1e-6, "max_iter": 3000}
    assert P["widths"]["deltas_m"] == [0.0, -0.5, 0.5] and lb.WIDTH_DELTAS == (0.0, -0.5, 0.5)
    assert P["sectors"]["merge_gap_m"] == 40.0 and P["sectors"]["type_kph"] == [120.0, 200.0]
    assert {k: P["ood"][k] for k in ("root_seed", "stream", "levels", "steps", "n_obs",
                                     "subsample_seed", "chunk", "real_widths")} == {
        "root_seed": 7707, "stream": "a_dr", "levels": 100, "steps": 1200, "n_obs": 40_000,
        "subsample_seed": 7707, "chunk": 25, "real_widths": True}
    assert (P["stats"]["n_boot"], P["stats"]["boot_seed"], P["stats"]["ci"]) == (
        2000, 0, [2.5, 97.5])
    assert lb.P2_SHOOTING_LIMIT == 3e-3
    assert lb.parse_args(["freeze", "--mode", "lockbox"]).runs == 10


def test_metric_paths_cover_the_lock_and_the_project_file(lb):
    assert {"uv.lock", "pyproject.toml", "src/apexzero", "scripts/lockbox.py"} <= set(
        lb.METRIC_PATHS)


@needs_git
def test_tag_gate_refuses_a_changed_lock_file(lb, tmp_path, monkeypatch):
    repo = ScratchRepo(tmp_path, monkeypatch, lb)
    repo.tag()
    assert lb.tag_gate()["commit"] == repo.git("rev-parse", "HEAD")
    for name in ("uv.lock", "pyproject.toml"):
        (repo.clone / name).write_text("# changed\n")
        with pytest.raises(PermissionError, match="differs"):
            lb.tag_gate()
        repo.git("checkout", "--", name)
    lb.tag_gate()


def _json_bytes(*subs: dict) -> bytes:
    return json.dumps({"subjects": list(subs)}).encode()


def test_timesteps_must_equal_the_checkpoints(lb, tmp_path):
    ck = tmp_path / "a.pkl"
    sha = _fake_checkpoint(ck, 7)
    sub = {"name": "s1", "checkpoint": str(ck), "sha256": sha}
    (out,) = lb.verify_subjects(lb.subjects_from_file(_json_bytes(dict(sub, timesteps=7)), "x"))
    assert out["timesteps"] == 7
    assert lb.verify_subjects(lb.subjects_from_file(_json_bytes(sub), "x"))[0]["timesteps"] == 7
    for bad in (8, "7", True):
        with pytest.raises(lb.CheckFailed, match="timesteps"):
            lb.verify_subjects(lb.subjects_from_file(_json_bytes(dict(sub, timesteps=bad)),
                                                     "x"))
    # a 12-digit file name is the checkpoint's timesteps too
    named = tmp_path / "000000000007.pkl"
    named.write_bytes(ck.read_bytes())
    lb.verify_subjects([dict(sub, checkpoint=str(named))])
    wrong = tmp_path / "000000000008.pkl"
    wrong.write_bytes(ck.read_bytes())
    with pytest.raises(lb.CheckFailed, match="file name"):
        lb.verify_subjects([dict(sub, checkpoint=str(wrong))])


def test_eval_rdev_is_for_the_dry_run_only(lb, tmp_path, monkeypatch):
    ev = tmp_path / "eval_rdev.json"
    ev.write_text('{"Norisring": {"completion": 1.0, "median_lap_s": 40.5, "gap": 0.1},'
                  ' "note": "x"}')
    sub = {"name": "s1", "checkpoint": "runs/s1/a.pkl", "sha256": "0" * 64,
           "eval_rdev": str(ev), "eval_rdev_sha256": lb.sha256_file(ev)}
    (got,) = lb.subjects_from_file(_json_bytes(sub), "x", "dryrun")
    assert got["stored_eval_rdev"] == {"Norisring": {"completion": 1.0, "median_lap_s": 40.5}}
    assert got["eval_rdev_sha256"] == sub["eval_rdev_sha256"]
    for mode in ("lockbox", None):         # lockbox, and the default (the strict one)
        args = (mode,) if mode else ()
        with pytest.raises(lb.CheckFailed, match="dry run only"):
            lb.subjects_from_file(_json_bytes(sub), "x", *args)
    for half in ({"eval_rdev": str(ev)}, {"eval_rdev_sha256": sub["eval_rdev_sha256"]}):
        s = {k: v for k, v in sub.items() if not k.startswith("eval_rdev")} | half
        with pytest.raises(lb.CheckFailed):
            lb.subjects_from_file(_json_bytes(s), "x", "dryrun")
        with pytest.raises(lb.CheckFailed, match="dry run only"):
            lb.subjects_from_file(_json_bytes(s), "x", "lockbox")
    with pytest.raises(lb.CheckFailed, match="sha256"):
        lb.subjects_from_file(_json_bytes(dict(sub, eval_rdev_sha256="0" * 64)), "x", "dryrun")
    monkeypatch.setattr(lb, "SEALED", tmp_path)
    with pytest.raises(PermissionError):
        lb.subjects_from_file(_json_bytes(sub), "x", "dryrun")
    plain = {k: v for k, v in sub.items() if not k.startswith("eval_rdev")}
    assert "stored_eval_rdev" not in lb.subjects_from_file(_json_bytes(plain), "x", "lockbox")[0]


@needs_tracks
def test_dryrun_freeze_keeps_runs_and_stored_evals_reach_the_agreement(lb, tmp_path):
    ck = tmp_path / "a.pkl"
    sha = _fake_checkpoint(ck)
    ev = tmp_path / "eval_rdev.json"
    ev.write_text(json.dumps({"Norisring": {"completion": 1.0, "median_lap_s": 40.5}}))
    spath = tmp_path / "subjects.json"
    spath.write_bytes(_json_bytes({"name": "s1", "checkpoint": str(ck), "sha256": sha,
                                   "timesteps": 7, "eval_rdev": str(ev),
                                   "eval_rdev_sha256": lb.sha256_file(ev)}))
    root = tmp_path / "dry"
    lb.main(["freeze", "--mode", "dryrun", "--root", str(root), "--circuits", "Norisring",
             "--subjects", str(spath), "--runs", "4"])
    rec = json.loads((root / "freeze.json").read_text())
    assert rec["protocol"]["laps"]["runs"] == 4 and rec["protocol"]["ood"]["real_widths"] is True
    (sub,) = rec["subjects"]
    assert sub["timesteps"] == 7 and sub["stored_eval_rdev"]["Norisring"]["completion"] == 1.0
    rows = [{"subject": "s1", "variant": "original", "circuit": "Norisring", "completion": 1.0,
             "median_lap_s": 40.25}]
    assert lb.stored_eval_agreement(rec, rows) == {
        "s1": {"circuits": 1, "completion_equal": 1, "max_abs_median_lap_diff_s": 0.25}}


def test_lockbox_freeze_refuses_other_runs_and_eval_rdev(lb, tmp_path, monkeypatch):
    """Behind a (fake) passing gate on an R-dev stand-in for R_TEST: --runs other than the
    protocol's 10 is refused before anything is read; a subjects file with eval_rdev fields is
    refused; neither leaves a root."""
    events = []
    monkeypatch.setattr(lb, "ROOT", tmp_path)
    monkeypatch.setattr(lb, "R_TEST", ["Norisring"])
    monkeypatch.setattr(lb, "LOCKBOX_ROOT", tmp_path / "outputs" / "lockbox")
    monkeypatch.setattr(lb, "lockbox_guard", lambda name, unsealed=False: events.append(name))
    sub = {"name": "s1", "checkpoint": "runs/s1/a.pkl", "sha256": "0" * 64,
           "eval_rdev": "e.json", "eval_rdev_sha256": "0" * 64}
    raw = _json_bytes(sub)
    (tmp_path / "s.json").write_bytes(raw)
    monkeypatch.setattr(lb, "git_run", FakeGit(blobs={"s.json": lb.git_blob_id(raw)}))
    for runs in ("4", "11", "3"):
        with pytest.raises(lb.CheckFailed, match="takes 10 runs"):
            lb.main(["freeze", "--mode", "lockbox", "--subjects", str(tmp_path / "s.json"),
                     "--runs", runs])
    with pytest.raises(lb.CheckFailed, match="dry run only"):
        lb.main(["freeze", "--mode", "lockbox", "--subjects", str(tmp_path / "s.json")])
    assert events and not (tmp_path / "outputs").exists()


def test_the_committed_subjects_file_lists_e0(lb):
    d = json.loads((ROOT / "docs" / "prereg" / "protocol-v1-subjects.json").read_text())
    subs = lb.subjects_from_file((ROOT / "docs" / "prereg" / "protocol-v1-subjects.json")
                                 .read_bytes(), "x")
    assert [s["name"] for s in subs] == [f"e0-s5{i}" for i in range(5)]
    for s in d["subjects"]:
        assert set(s) == {"name", "checkpoint", "sha256", "timesteps"}
        assert s["checkpoint"] == f"runs/{s['name']}/ckpt/000999948288.pkl"
        assert s["timesteps"] == 999948288 and len(s["sha256"]) == 64
    lb.check_subject_paths(subs)
    man = ROOT / "outputs" / "e0" / "manifest.json"
    if man.exists():
        runs = {r["name"]: r["final"] for r in json.loads(man.read_text())["runs"]}
        assert all(runs[s["name"]]["sha256"] == s["sha256"] for s in d["subjects"])
    for s in d["subjects"]:
        ck = ROOT / s["checkpoint"]
        if ck.exists():
            assert lb.sha256_file(ck) == s["sha256"]


# ------------------------------------------------------------------------- lockbox rehearsal


@needs_git
@needs_tracks
def test_lockbox_mode_rehearsal_freeze_ocp_laps_analyse(lb, tmp_path, monkeypatch):
    """Every lockbox-mode stage on Norisring standing in for R_TEST: a scratch repo with a real
    annotated protocol-v1 on its origin, LOCKBOX_ROOT, the repository and the track directory
    under tmp_path, ``lockbox_guard`` a recorder. The solver, the checkpoint and its OOD buffer
    are fakes; the tag gate, the path checks, the one-shot analyse and the rollout (with its
    parity check against ``timed_laps``) are the real ones. No real R_TEST name can reach the
    recorder (it raises), and the guard runs only behind the gate."""
    import types

    import jax.numpy as jnp

    from apexzero.interp.controls import KnnBuffer

    ck = tmp_path / "a.pkl"
    subjects = _json_bytes({"name": "fake", "checkpoint": str(ck), "sha256": _fake_checkpoint(ck),
                            "timesteps": 7})
    repo = ScratchRepo(tmp_path, monkeypatch, lb, subjects=subjects)
    track_dir = repo.clone / "data" / "tracks" / "tumftm"
    track_dir.mkdir(parents=True)
    shutil.copy(ROOT / "data" / "tracks" / "tumftm" / "Norisring.csv", track_dir)
    lockbox = tmp_path / "outputs" / "lockbox"
    monkeypatch.setattr(lb, "R_TEST", ["Norisring"])
    monkeypatch.setattr(lb, "TRACK_DIR", track_dir)
    monkeypatch.setattr(lb, "LOCKBOX_ROOT", lockbox)
    monkeypatch.setattr(lb, "SEALED", tmp_path / "outputs" / "sealed")
    assert not set(lb.R_TEST) & set(R_TEST) and lb.ROOT == repo.clone

    events = []
    real_gate = lb.tag_gate

    def gate():
        events.append(("gate", "start"))
        got = real_gate()
        events.append(("gate", "ok"))
        return got

    def guard(name, unsealed=False):
        if name.split("-")[0] in R_TEST:
            raise AssertionError(f"a real R_TEST name reached the guard: {name}")
        events.append(("guard", name, unsealed))

    monkeypatch.setattr(lb, "tag_gate", gate)
    monkeypatch.setattr(lb, "lockbox_guard", guard)
    _fake_ocp(monkeypatch)
    _serial_pool(monkeypatch, lb)

    def policy(obs):
        return jnp.stack([-0.8 * obs[:, 204], 0.3 + 0.0 * obs[:, 200]], axis=-1)

    snap = types.SimpleNamespace(obs_rms=types.SimpleNamespace(normalize=lambda o: o))
    monkeypatch.setattr(lb, "load_subject", lambda sub: (policy, None, snap))
    buf = KnnBuffer(obs=np.random.default_rng(0).normal(size=(64, 222)).astype(np.float32),
                    act=np.zeros((64, 2), np.float32), lap=np.zeros(64, int),
                    lolo_median_nn=1.0, agent="fake")
    monkeypatch.setattr(lb, "subject_buffer", lambda root, rec, sub, s, c: (buf, "0" * 64))

    def stage(*argv):
        events.clear()
        lb.main(list(argv))
        guards = [i for i, e in enumerate(events) if e[0] == "guard"]
        gate_ok = [i for i, e in enumerate(events) if e == ("gate", "ok")]
        assert events[0] == ("gate", "start") and gate_ok and guards
        assert min(guards) > min(gate_ok)                 # the guard only behind the gate
        assert all(events[i][2] is True for i in guards)
        assert {events[i][1] for i in guards} <= {"Norisring", "Norisring-mirrored",
                                                  "Norisring-reversed"}

    sfile = str(repo.clone / SUBJECTS_REL)
    with pytest.raises(PermissionError, match="not on origin"):    # no tag yet: sealed
        lb.main(["freeze", "--mode", "lockbox", "--subjects", sfile])
    assert not [e for e in events if e[0] == "guard"] and not lockbox.exists()
    repo.tag()
    with pytest.raises(lb.CheckFailed, match="takes 10 runs"):
        lb.main(["freeze", "--mode", "lockbox", "--subjects", sfile, "--runs", "4"])
    assert not lockbox.exists()

    stage("freeze", "--mode", "lockbox", "--subjects", sfile)
    rec = json.loads((lockbox / "freeze.json").read_text())
    assert rec["mode"] == "lockbox" and rec["label"].startswith("LOCKBOX")
    assert rec["circuits"] == ["Norisring"] and rec["protocol"]["laps"]["runs"] == 10
    assert rec["protocol"]["ood"]["real_widths"] is True
    assert rec["subjects_file"]["source"] == SUBJECTS_REL
    assert rec["subjects_file"]["sha256"] == lb.sha256_bytes(subjects)
    assert rec["subjects"][0]["timesteps"] == 7 and rec["tag"]["commit"] == repo.git("rev-parse", "HEAD")
    assert list(rec["circuit_files_sha256"]) == ["data/tracks/tumftm/Norisring.csv"]
    stage("ocp", "--root", str(lockbox))
    assert len(list((lockbox / "ocp").glob("*.json"))) == 9
    stage("laps", "--root", str(lockbox))
    for v in lb.VARIANTS:
        name = lb.variant_name("Norisring", v)
        r = json.loads((lockbox / "laps" / "fake" / f"{name}.json").read_text())
        assert len(r["completed"]) == 10
        with np.load(lockbox / "laps" / "fake" / f"{name}.npz") as z:
            n_st = len(z["stations"])
            assert z["path_n"].shape == z["path_v"].shape == z["tau"].shape == (10, n_st)
            done = np.array(r["completed"], bool)
            assert np.isfinite(z["path_n"][done]).all() and np.isfinite(z["path_v"][done]).all()
            assert np.isnan(z["path_n"][~done]).all() and np.isnan(z["path_v"][~done]).all()
            assert (z["path_v"][done] > 0).all()
    stage("analyse", "--root", str(lockbox))
    out = json.loads((lockbox / "analysis.json").read_text())
    assert out["mode"] == "lockbox" and out["label"].startswith("LOCKBOX")
    assert out["checks"]["references_converged"] == "9/9" and len(out["rows"]) == 3
    assert (lockbox / "corners.csv").exists() and (lockbox / "analysis.lock").exists()
    with pytest.raises(lb.CheckFailed, match="runs once"):
        lb.main(["analyse", "--root", str(lockbox)])


def csv_rows(path: Path):
    import csv

    with open(path, newline="") as fh:
        yield from csv.DictReader(fh)
