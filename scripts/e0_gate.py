"""E0's run manifest and gate (docs/plan/e0.md, Sections 2 and 3).

    uv run python scripts/e0_gate.py manifest --tag e0-train-v1 --box-sums runs/e0-box-sha256-*.txt
    uv run python scripts/e0_gate.py gate
    uv run python -c "import sys; sys.path.insert(0, 'scripts'); import e0_gate; e0_gate.box_command()"

**The tag gate** (``tag_gate``, run first by both subcommands): e0-train-v1 is an annotated tag
on origin (both refs named in full in ``git ls-remote``) and locally, the same object; HEAD
descends from its commit; E0's code (``E0_CODE_PATHS``: train_dr.py's import closure under
src/apexzero by directory, src/apexzero/ocp (the end-of-run evaluation unpickles the OCP laps),
the trainer, the queue, the environment's pins and these two scripts; untracked files included)
is the tag's, byte for byte (``code_byte_problems``: a file marked assume-unchanged or
skip-worktree passes git diff, not this); generator-v1 and sim-v1 both peel to
``GENERATOR_COMMIT``, locally and on origin, and the tag's commit descends from it; the same
paths less the two E0 scripts, which do not exist there, are identical in ``GENERATOR_COMMIT``
and the tag's commit; and the 12 R-dev centreline files match ``RDEV_SHA256`` (docs/findings/
gw4.md Section 3). The centrelines are opened by name, never by listing data/tracks/tumftm/,
which also holds the lockbox circuits.

**manifest** (on the Mac, after ``gpu_sync.sh pull-round e0 scripts/e0_runs.py --phase main``,
with the box's own ``sha256sum`` lines, from ``box_command()``, in ``--box-sums``): runs
``round_manifest.py check`` on round e0 against the tag (``--runs-args --phase main``, the
queue's form). Its only tolerated problems are those of a run outputs/e0/failures.json lists
(committed; a run that failed twice: status "failed" and a reason; the queue re-runs a failed
run once). Then, per run: every checkpoint's sha256 and timesteps, checked against the box's
sums (a done or retried run's are the 16 of ``CHECKPOINT_TIMESTEPS``, the last at
``FINAL_TIMESTEPS``); every file the box's sums list under the run is here, so a file the
pull-back missed refuses the manifest (run it before the box is deleted, and pull again);
config.json, metrics.csv, eval_rdev.json and the logs' sha256; the config's args (``config_problems``: --real-widths, no --real-widths-from,
no --stat-widths, no --tilt-*, --vg-coef 0, --steps 1e9, the run's seed, every other argument
train_dr.py's default at the tag); the box's status, times, lane and GPU. The box's sums must
also hold the 12 R-dev centrelines and the 12 R-dev OCP files, each equal to ``RDEV_SHA256`` /
``OCP_SHA256``; they are recorded. A run with no eval_rdev.json, here or in the box's sums, is
recorded as it is (the gate counts it 0). Writes outputs/e0/manifest.json, write-once, and only
when there is no problem.

**gate** (Section 3), once, after the manifest is committed: refuses if outputs/e0/gate.json or
gate.lock exists or was ever committed; re-checks the committed manifest and every input's
sha256, and each eval_rdev.json's structure (``eval_problems``: its circuits are R_DEV's in
order, each row a dict whose ``completion`` is a number or null), so that a malformed file does
not use up the single run; takes the lock (never removed); then, per seed, counts the R-dev
circuits with completion > 0.5 in its eval_rdev.json (null and NaN are not completed; a seed
failures.json lists, or one with no eval_rdev.json, counts 0). The gate passes if the median of
the 5 counts is at least ``PASS_MEDIAN`` (10). Descriptive: each completed circuit's gap to the
s-OCP lap (eval_rdev.json's ``gap``, as vg1_analysis reads it) and the number of seeds at 10 or
more. Every number is labelled "in-sample for widths". Writes outputs/e0/gate.json, write-once,
and prints it.

Nothing here reads R_TEST, outputs/sealed or any run outside E0's list. Nothing here imports
another study's script: the helpers copied from scripts/vg1_analysis.py are marked.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import e0_runs as RL
import numpy as np
import round_manifest as RM  # the shared layer's trainer-parser helpers (stdlib only)

ROOT = Path(__file__).resolve().parents[1]
ROUND = "e0"
TAG = "e0-train-v1"
UPSTREAM_TAGS = ("generator-v1", "sim-v1")
GENERATOR_COMMIT = "f7c924002927374423ba44a476a23272a5ba2691"     # both tags peel to it
RUNS_SCRIPT = "scripts/e0_runs.py"
OUT = "outputs/e0"
MANIFEST = f"{OUT}/manifest.json"
GATE = f"{OUT}/gate.json"
LOCK = f"{OUT}/gate.lock"
FAILURES = f"{OUT}/failures.json"
E0_SCRIPTS = ("scripts/e0_runs.py", "scripts/e0_gate.py")
# train_dr.py's import closure under src/apexzero, by directory (tests/test_e0.py runs it, with
# the end-of-run evaluation's OCP unpickle, and checks every loaded file is under one of these;
# src/apexzero/ocp is loaded by that unpickle), the trainer, the queue, the environment's pins
# and E0's scripts
E0_CODE_PATHS = ("src/apexzero/__init__.py", "src/apexzero/agents", "src/apexzero/env",
                 "src/apexzero/eval", "src/apexzero/oracle", "src/apexzero/tracks",
                 "src/apexzero/vehicle", "src/apexzero/ocp", "scripts/train_dr.py",
                 "scripts/_bootstrap.py", "scripts/gpu_round.sh", "scripts/round_manifest.py",
                 "uv.lock", "pyproject.toml", *E0_SCRIPTS)
# the same, less E0's own scripts: what generator-v1 and the tag's commit must share
UPSTREAM_CODE_PATHS = tuple(p for p in E0_CODE_PATHS if p not in E0_SCRIPTS)
# docs/findings/gw4.md Section 3: the R-dev centrelines, in R_DEV's order
RDEV_SHA256 = {
    "BrandsHatch": "9c69b09cef717d8abffc7c9450e59d7e339ac0dd0c7ed8121d73fe2f373145bb",
    "Hockenheim": "5eb6b7d8ff7b9fc905b38caed9a65d8e76e4bd7af5af9ba82aecdf4a6e0eacc5",
    "MoscowRaceway": "843c0f47e7210b5c9bfb3bf2f55cfee02cced0970f9b711a951b650c5cad3753",
    "Norisring": "8857d3c362ad2923c1f93c8d257498f50459770b9021adcc7969b71085c31d9a",
    "Nuerburgring": "9fea2a9daa25311bb251be416140b6c55cb2e53bb4f69c28f512fbfce25a3f87",
    "Oschersleben": "9ce893ee1fde37cc0626d654b765e65cd7159ffca922496ab881d61cad6ab304",
    "Sepang": "89ee026fd439e9b41311fc742b44efb120c3c1148f55e5963b38b62971352eaf",
    "Sochi": "caf3f00d78a7b54aeab131d2a8f263131c33d5431292a05dcbaaad49a19c351c",
    "YasMarina": "befab5a3be5a6ab320ce16d5ac5ec9016d633c5d6e739b9483f4d6364f0fe017",
    "Melbourne": "f9ea4395a25dc4189d8b8c692ce9656682de62ad7b784dfd7c024ac8d4f245e9",
    "Catalunya": "8d6dc9d50e98bdecd818cb880a4b12bb5cec25d699a6b774e0b593c0f8a9da5e",
    "Zandvoort": "f3f2a82c9f41bcee8d7161a6ad6c0b8e6d4b902dd772459918c9fe7f7451bc59",
}
TRACKS_DIR = "data/tracks/tumftm"
OCP_DIR = "outputs/ocp"
# docs/plan/e0.md Section 2, "OCP reference files": the R-dev s-OCP laps train_dr.py's
# end-of-run evaluation unpickles, measured on the Mac on 28 Oct 2026, in R_DEV's order
OCP_SHA256 = {
    "BrandsHatch": "2c8addbe5bd296b6b1b6356c6081b313a4704f14adb923cdca6ab2b71222a939",
    "Hockenheim": "2cc8b8113bde3645fd6c4a068c2573acaf5a5ad7bf957d877a28f70effac8dc5",
    "MoscowRaceway": "d5e42ef26bdb26dabd387042fea2e7d0ca50dfc6666b5c455ade9c8c382dc1fe",
    "Norisring": "19289175beba94c1df6b1c45e9e074339776bc08b233fc35fd34e3e011ef6aa5",
    "Nuerburgring": "c80fc5f860ee5c6cc0443a3f619aec552c978333d5c041fd9064af2019bd2877",
    "Oschersleben": "dfbaeee52d78438d83194f55a25763e51c3161c34f3662c05a9e4c7c71d8a2bd",
    "Sepang": "941c32b5219a94dfa21653344d732e0de3e469af61b62bd874ca1ce4d3c03039",
    "Sochi": "451725a45f4e5747a84679565f36b96279bb59b7ca6850621efc154eb441c388",
    "YasMarina": "d08853c4c1a35d10dd908b6c728b20e0d9c72624dee59b099140008aaf07d60e",
    "Melbourne": "7395c3f8094ea20ecaf70aeefddb6a71b44237f8c4588434a4518610c474411e",
    "Catalunya": "dc6ca50f225de45506780abc9e35edcfe8c5a63e79faacca475f44ee9f110ff8",
    "Zandvoort": "3e2ef9a00167dd25a64fff0af3cc02ef5924b53f5463fd41722833e056dba646",
}
# train_dr.py's last update at --steps 1e9: floor(1e9 / (2048 x 64)) = 7,629 updates of 131,072
# steps (checkpoints.checkpoint_schedule names the final snapshot by the timesteps it ends on)
FINAL_TIMESTEPS = 999_948_288
# checkpoints.checkpoint_schedule(1e9, 131072): the 16 snapshots a finished run keeps, by name
# (the log-spaced targets, then the final update)
CHECKPOINT_TIMESTEPS = (10_000_000, 13_593_563, 18_478_497, 25_118_864, 34_145_488, 46_415_888,
                        63_095_734, 85_769_589, 116_591_440, 158_489_319, 215_443_469,
                        292_864_456, 398_107_170, 541_169_526, 735_642_254, FINAL_TIMESTEPS)
# the two defaults train_dr.py computes (ppo.ValueGradient's lam and winsor)
COMPUTED_DEFAULTS = {"vg_lambda": 0.95, "vg_winsor": 0.005}
COMPLETED = 0.5                    # a circuit is completed when completion > 0.5 over 10 laps
PASS_MEDIAN = 10                   # the gate: the median seed completes at least 10 of 12
LABEL = "in-sample for widths"


# ------------------------------------------------------------------------- small helpers


# Copied from scripts/vg1_analysis.py (bodies unchanged), so that E0's logic lives in files the
# tag pins: CheckFailed, sha256_file, clean, num, box_sums and read_list.


class CheckFailed(SystemExit):
    """A built-in check failed: nothing more is written."""


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(x):
    """JSON-safe: NaN as null, infinities as "inf" / "-inf", NumPy values as Python ones."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def num(x) -> float:
    """``clean``'s inverse for one number: null is NaN, "inf" / "-inf" are infinities."""
    if x is None:
        return math.nan
    return float(x)


def box_sums(paths: list[str]) -> dict[str, str]:
    """{repo-relative path: sha256} from ``sha256sum`` lines taken on the box."""
    out = {}
    for p in paths:
        for k, line in enumerate(Path(p).read_text().splitlines(), 1):
            if not line.strip():
                continue
            m = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line.strip())
            if not m:
                raise CheckFailed(f"{Path(p).name}, line {k}: not a sha256sum line")
            path = m[2].removeprefix("./")
            if path in out and out[path] != m[1]:
                raise CheckFailed(f"{path}: two different box sha256 values")
            out[path] = m[1]
    return out


def read_list(path: Path, name: str) -> list[str]:
    """A module-level list of strings in a source file, read with ast (no import)."""
    for node in ast.parse(Path(path).read_text()).body:
        targets = [getattr(t, "id", None) for t in getattr(node, "targets", [])]
        if isinstance(node, ast.Assign) and targets == [name]:
            return list(ast.literal_eval(node.value))
    raise CheckFailed(f"{name} not found in {path}")


def box_command() -> str:
    """The box's sha256 command (printed, and returned): the checkpoints and evaluations of
    every run, then the 12 R-dev centrelines and the 12 R-dev OCP files, each by explicit name
    (data/tracks/tumftm/ also holds the lockbox circuits and is never globbed)."""
    files = [f"{TRACKS_DIR}/{n}.csv" for n in RDEV_SHA256] + [f"{OCP_DIR}/{n}.pkl"
                                                              for n in OCP_SHA256]
    cmd = " ".join(["sha256sum", "runs/e0-*/ckpt/*.pkl", "runs/e0-*/eval_rdev.json", *files])
    print(cmd)
    return cmd


def git_run(*args: str) -> subprocess.CompletedProcess:
    """Every git call goes through here, in ROOT (the tests replace it)."""
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def git(*args: str) -> str:
    out = git_run(*args)
    return out.stdout.strip() if out.returncode == 0 else ""


def rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return f"<outside>/{p.name}"


def write_new(path: Path, obj) -> str:
    """Write JSON to a path that must not exist yet (atomic); returns its sha256."""
    path = Path(path)
    if path.exists():
        raise CheckFailed(f"{rel(path)} exists: outputs are write-once")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(clean(obj), indent=1, allow_nan=False) + "\n").encode()
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()
    print(f"wrote {rel(path)}", flush=True)
    return hashlib.sha256(data).hexdigest()


def committed(path: str) -> bool:
    """Tracked and unchanged against HEAD."""
    return (git_run("ls-files", "--error-unmatch", "--", path).returncode == 0
            and git_run("diff", "--quiet", "HEAD", "--", path).returncode == 0)


def ever_committed(path: str) -> bool:
    return bool(git("log", "--all", "--format=%H", "--", path))


def read_json(path: str, inputs: dict | None = None) -> dict:
    p = ROOT / path
    if not p.is_file():
        raise CheckFailed(f"{path} not found")
    if inputs is not None:
        inputs[path] = sha256_file(p)
    return json.loads(p.read_text())


def r_dev() -> list[str]:
    return read_list(ROOT / "src" / "apexzero" / "tracks" / "corners.py", "R_DEV")


def provenance(start: float, gate: dict, inputs: dict[str, str]) -> dict:
    return {"script": "scripts/e0_gate.py", "script_sha256": sha256_file(Path(__file__)),
            "runs_script_sha256": sha256_file(Path(__file__).with_name("e0_runs.py")),
            "tag_gate": gate, "head": git("rev-parse", "HEAD"),
            "changed": [ln for ln in git("status", "--porcelain", "--untracked-files=no")
                        .splitlines() if ln.strip()],
            "inputs": dict(sorted(inputs.items())),
            "argv": ["scripts/e0_gate.py", *[rel(a) if os.path.isabs(a) else a
                                             for a in sys.argv[1:]]],
            "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
            "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "environment": {"python": platform.python_version(), "machine": platform.machine(),
                            "system": platform.system()}}


# ------------------------------------------------------------------------- the tag gate


def _tree(commit: str, paths: tuple[str, ...]) -> dict[str, str] | str:
    """{file: blob id} of ``paths`` in ``commit``'s tree, or the error text."""
    ls = git_run("ls-tree", "-r", commit, "--", *paths)
    if ls.returncode != 0:
        return f"cannot list {commit[:12]}'s tree ({ls.stderr.strip()})"
    out = {}
    for line in ls.stdout.splitlines():
        meta, _, name = line.partition("\t")
        if name:
            out[name] = meta.split()[-1]
    return out


def code_byte_problems(commit: str, paths: tuple[str, ...] = E0_CODE_PATHS) -> list[str]:
    """Tracked files under ``paths`` that are not ``commit``'s on disk, byte for byte: each
    file's blob id hashed from its bytes (``git hash-object --no-filters``) against the commit's
    tree, a file of the commit missing on disk, and a file the index lists that the commit lacks.
    git diff and git status trust the index for a file marked assume-unchanged or skip-worktree,
    so an edited file there passes them; this does not (vg1_analysis.code_byte_problems)."""
    want = _tree(commit, paths)
    if isinstance(want, str):
        return [want]
    index = git_run("ls-files", "--", *paths)
    out = [f"{p} is tracked here but not in {commit[:12]}"
           for p in sorted(set(index.stdout.splitlines()) - set(want)) if p]
    present = [p for p in sorted(want) if (ROOT / p).is_file()]
    out += [f"{p} is missing on disk" for p in sorted(want) if p not in present]
    if present:
        h = git_run("hash-object", "--no-filters", "--", *present)
        got = h.stdout.split()
        if h.returncode != 0 or len(got) != len(present):
            return out + [f"git hash-object failed ({h.stderr.strip()})"]
        out += [f"{p} on disk is not {commit[:12]}'s" for p, g in zip(present, got, strict=True)
                if g != want[p]]
    return out


def tree_problems(a: str, b: str, paths: tuple[str, ...]) -> list[str]:
    """How ``paths`` differ between commits ``a`` and ``b`` (a file in one only, or a blob)."""
    ta, tb = _tree(a, paths), _tree(b, paths)
    if isinstance(ta, str) or isinstance(tb, str):
        return [t for t in (ta, tb) if isinstance(t, str)]
    return [f"{p} differs" if p in ta and p in tb else f"{p} is in only one"
            for p in sorted(set(ta) | set(tb)) if ta.get(p) != tb.get(p)]


def rdev_problems() -> list[str]:
    """The 12 R-dev centrelines against ``RDEV_SHA256``, opened by name from R_DEV's list (the
    directory also holds the lockbox circuits and is never listed)."""
    names = r_dev()
    out = []
    if names != list(RDEV_SHA256):
        out.append(f"R_DEV {names} is not the sha256 table's circuits")
    for name, want in RDEV_SHA256.items():
        p = ROOT / TRACKS_DIR / f"{name}.csv"
        if not p.is_file():
            out.append(f"{rel(p)} is missing")
        elif sha256_file(p) != want:
            out.append(f"{rel(p)} is not gw4.md's (sha256)")
    return out


def tag_gate(tag: str = TAG, paths: tuple[str, ...] = E0_CODE_PATHS) -> dict:
    """PermissionError unless every check of the module docstring's 'tag gate' holds."""
    ls = git_run("ls-remote", "origin", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}")
    if ls.returncode != 0:
        raise PermissionError(f"cannot list origin's tags ({ls.stderr.strip()})")
    refs = {}
    for line in ls.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            refs[parts[1]] = parts[0]
    obj, peeled = refs.get(f"refs/tags/{tag}"), refs.get(f"refs/tags/{tag}^{{}}")
    if obj is None:
        raise PermissionError(f"{tag} is not on origin: E0 is not launched or read before it")
    if peeled is None or peeled == obj:
        raise PermissionError(f"{tag} on origin is not an annotated tag")
    local = git_run("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}")
    if local.returncode != 0 or local.stdout.strip() != obj:
        raise PermissionError(f"the local {tag} is missing or is not origin's tag object")
    kind = git_run("cat-file", "-t", obj)
    if kind.returncode != 0 or kind.stdout.strip() != "tag":
        raise PermissionError(f"{tag} ({obj}) is not a tag object")
    commit = git_run("rev-parse", "--verify", "--quiet", f"{obj}^{{commit}}")
    if commit.returncode != 0 or commit.stdout.strip() != peeled:
        raise PermissionError(f"{tag}'s commit does not match origin's")
    if git_run("merge-base", "--is-ancestor", peeled, "HEAD").returncode != 0:
        raise PermissionError(f"HEAD does not descend from {tag} ({peeled})")
    absent = [p for p in paths if git_run("cat-file", "-e", f"{peeled}:{p}").returncode != 0]
    if absent:
        raise PermissionError(f"not in {tag}: {', '.join(absent)}")
    if git_run("diff", "--quiet", peeled, "--", *paths).returncode != 0:
        raise PermissionError(f"E0's code ({', '.join(paths)}) differs from {tag}'s")
    untracked = git_run("ls-files", "--others", "--exclude-standard", "--", *paths)
    if untracked.returncode != 0 or untracked.stdout.strip():
        raise PermissionError(f"untracked files under E0's code: {untracked.stdout!r}")
    unequal = code_byte_problems(peeled, paths)
    if unequal:
        raise PermissionError(f"E0's code on disk is not {tag}'s: {'; '.join(unequal[:6])}")
    up_refs = [r for up in UPSTREAM_TAGS for r in (f"refs/tags/{up}", f"refs/tags/{up}^{{}}")]
    ls = git_run("ls-remote", "origin", *up_refs)
    if ls.returncode != 0:
        raise PermissionError(f"cannot list origin's tags ({ls.stderr.strip()})")
    refs = {}
    for line in ls.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            refs[parts[1]] = parts[0]
    for up in UPSTREAM_TAGS:
        there = refs.get(f"refs/tags/{up}^{{}}") or refs.get(f"refs/tags/{up}")
        if there != GENERATOR_COMMIT:
            raise PermissionError(f"{up} on origin is {there}, not {GENERATOR_COMMIT}")
        c = git_run("rev-parse", "--verify", "--quiet", f"refs/tags/{up}^{{commit}}")
        if c.returncode != 0 or c.stdout.strip() != GENERATOR_COMMIT:
            raise PermissionError(f"the local {up} is missing or is not {GENERATOR_COMMIT}")
    if git_run("merge-base", "--is-ancestor", GENERATOR_COMMIT, peeled).returncode != 0:
        raise PermissionError(f"{tag}'s commit does not descend from {GENERATOR_COMMIT} "
                              f"({' and '.join(UPSTREAM_TAGS)})")
    base = tuple(p for p in paths if p not in E0_SCRIPTS)
    moved = tree_problems(GENERATOR_COMMIT, peeled, base)
    if moved:
        raise PermissionError(f"the code differs between generator-v1 and {tag}: "
                              f"{'; '.join(moved[:6])}")
    bad = rdev_problems()
    if bad:
        raise PermissionError(f"the R-dev centrelines are not gw4.md's: {'; '.join(bad[:4])}")
    return {"tag": tag, "tag_object": obj, "commit": peeled, "head": git("rev-parse", "HEAD"),
            "upstream": dict.fromkeys(UPSTREAM_TAGS, GENERATOR_COMMIT)}


# ------------------------------------------------------------------------- the round check


def check_report() -> tuple[int, str]:
    """(return code, report) of ``round_manifest.py check`` on round e0 against the tag."""
    cmd = [sys.executable, "scripts/round_manifest.py", "check", "--round", ROUND,
           "--runs-script", RUNS_SCRIPT, "--tag", TAG, "--runs-args", "--phase", "main"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout + proc.stderr


def round_check(failures: dict) -> dict:
    """``round_manifest.py check`` against the tag; the problems of a run ``failures`` lists
    are tolerated (and listed), any other one raises."""
    rc, report = check_report()
    problems = []
    if rc:
        lines = report.splitlines()
        if not any(ln.startswith("FAILED:") for ln in lines):
            raise CheckFailed(f"round_manifest.py check did not run:\n{report.strip()}")
        k = next(i for i, ln in enumerate(lines) if ln.startswith("FAILED:"))
        problems = [ln.strip() for ln in lines[k + 1:] if ln.strip()]
    tolerated = [p for p in problems if p.split(":", 1)[0] in set(failures)]
    other = [p for p in problems if p not in tolerated]
    if other:
        raise CheckFailed("round_manifest.py check failed:\n  " + "\n  ".join(other))
    return {"returncode": rc, "tolerated": tolerated,
            "report_sha256": hashlib.sha256(report.encode()).hexdigest(), "report": report}


def read_failures(inputs: dict) -> dict:
    """outputs/e0/failures.json's runs ({} if there is no file): committed, each with status
    "failed" (failed twice) and a reason."""
    if not (ROOT / FAILURES).exists():
        return {}
    if not committed(FAILURES):
        raise CheckFailed(f"{FAILURES} must be committed before any outcome is read")
    rec = read_json(FAILURES, inputs)
    if rec.get("record") != "e0-failures":
        raise CheckFailed(f"{FAILURES} is not E0's failure record")
    runs = rec.get("runs") or {}
    bad = [k for k, v in runs.items()
           if (v or {}).get("status") != "failed" or not (v or {}).get("reason")]
    if bad:
        raise CheckFailed(f"{FAILURES}: {bad} need status 'failed' (failed twice) and a reason")
    return runs


# ------------------------------------------------------------------------- the manifest


def box_records() -> dict[str, dict]:
    """{run: its record in the box manifests, with the manifest's file name as 'box'}."""
    out = {}
    for p in sorted((ROOT / "runs").glob(f"{ROUND}-manifest-*of*.json")):
        m = json.loads(p.read_text())
        for r in m.get("runs", []):
            out[r["name"]] = {**r, "box": p.name, "gpu_model": m.get("gpu_model"),
                              "gpu_count": m.get("gpu_count"), "box_type": m.get("box_type")}
    return out


def round_files() -> dict[str, str]:
    """{path: sha256} of the round's box manifests, queue logs and status files."""
    pats = (f"{ROUND}-manifest-*of*.json", f"{ROUND}-queue-*of*.log",
            f"{ROUND}-status-*of*.tsv")
    return {f"runs/{p.name}": sha256_file(p) for pat in pats
            for p in sorted((ROOT / "runs").glob(pat))}


def checkpoints(name: str) -> list[dict]:
    """[{path, sha256, timesteps}] of runs/<name>/ckpt/*.pkl in timestep order."""
    out = []
    for p in sorted((ROOT / "runs" / name / "ckpt").glob("*.pkl")):
        if not re.fullmatch(r"\d{12}", p.stem):
            raise CheckFailed(f"runs/{name}/ckpt/{p.name}: not a checkpoint name")
        out.append({"path": f"runs/{name}/ckpt/{p.name}", "sha256": sha256_file(p),
                    "timesteps": int(p.stem)})
    return out


def trainer_defaults(source: str) -> dict:
    """train_dr.py's argument defaults, from its parser rebuilt from ``source`` (the tag's) by
    round_manifest.py's ``trainer_parser``; the two it computes are ``COMPUTED_DEFAULTS``."""
    out = vars(RM.trainer_parser(source).parse_args([]))
    for k, v in out.items():
        if v is RM.COMPUTED:
            if k not in COMPUTED_DEFAULTS:
                raise CheckFailed(f"train_dr.py computes the default of --{k}: not checkable")
            out[k] = COMPUTED_DEFAULTS[k]
    return out


def tag_defaults(commit: str) -> dict:
    show = git_run("show", f"{commit}:scripts/train_dr.py")
    if show.returncode != 0:
        raise CheckFailed(f"scripts/train_dr.py is not in {commit[:12]}")
    return trainer_defaults(show.stdout)


def expected_args(seed: int, defaults: dict) -> dict:
    """One run's full args: Section 1's (--steps 1e9, the seed and name, --real-widths with no
    --real-widths-from, no --stat-widths, no tilt, no VG term) over every other argument at
    train_dr.py's default at the tag (``defaults``)."""
    return {**defaults, "steps": 1e9, "name": RL.run_name(seed), "seed": int(seed),
            "real_widths": True, "real_widths_from": "", "stat_widths": False,
            "tilt_fraction": 0.0, "vg_coef": 0.0}


def config_problems(cfg: dict, seed: int, defaults: dict) -> list[str]:
    """How a run's config.json args depart from ``expected_args``; [] when they match."""
    got, want = cfg.get("args"), expected_args(seed, defaults)
    if not isinstance(got, dict):
        return ['config has no "args" object']
    out = []
    for k in sorted(set(got) | set(want)):
        if k not in want:
            out.append(f"config records --{k}, which train_dr.py does not define")
        elif k not in got:
            out.append(f"config does not record --{k}")
        elif not RM._same(got[k], want[k]):
            out.append(f"--{k} is {got[k]!r} in the config, want {want[k]!r}")
    return out


def box_data(sums: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """({path: the box's sha256}, problems) for the 12 R-dev centrelines and the 12 R-dev OCP
    files in the box's sums: each must be there and equal ``RDEV_SHA256`` / ``OCP_SHA256``."""
    got, problems = {}, []
    pinned = [(f"{TRACKS_DIR}/{n}.csv", h) for n, h in RDEV_SHA256.items()]
    pinned += [(f"{OCP_DIR}/{n}.pkl", h) for n, h in OCP_SHA256.items()]
    for path, want in pinned:
        if path not in sums:
            problems.append(f"{path} has no box sha256 (--box-sums)")
            continue
        got[path] = sums[path]
        if sums[path] != want:
            problems.append(f"{path} on the box is not the pinned file (sha256)")
    return got, problems


def build_manifest(sums: dict[str, str], runs: list[dict], failures: dict, check: dict,
                   defaults: dict, gate: dict) -> dict:
    """The manifest (module docstring 'manifest'); raises on any problem."""
    boxes = box_records()
    problems, entries = [], []
    for run in runs:
        name = run["name"]
        box = boxes.get(name, {})
        fail = failures.get(name)
        ent = {"name": name, "seed": run["seed"], "status": box.get("status"), "failure": fail,
               "box": {k: box.get(k) for k in ("box", "lane", "gpu", "start", "end", "gpu_model",
                                               "gpu_count", "box_type")}}
        d = ROOT / "runs" / name
        cks = checkpoints(name) if d.is_dir() else []
        ent["checkpoints"] = cks
        done = box.get("status") in ("done", "retried")
        final = cks[-1] if cks and cks[-1]["timesteps"] == FINAL_TIMESTEPS else None
        if done and final is None:
            problems.append(f"{name}: no final checkpoint at {FINAL_TIMESTEPS} timesteps")
        got = {c["timesteps"] for c in cks}
        if done and got != set(CHECKPOINT_TIMESTEPS):
            problems.append(f"{name}: its checkpoints are not train_dr.py's schedule at --steps "
                            f"1e9 (missing {sorted(set(CHECKPOINT_TIMESTEPS) - got)}, extra "
                            f"{sorted(got - set(CHECKPOINT_TIMESTEPS))})")
        ent["final"] = final
        for f in ("config.json", "metrics.csv", "eval_rdev.json"):
            p = d / f
            ent[f.split(".")[0] + "_sha256"] = sha256_file(p) if p.is_file() else None
        ent["logs_sha256"] = {lg: sha256_file(ROOT / lg) for lg in
                              (f"runs/{name}.log", f"runs/{name}.log.1") if (ROOT / lg).is_file()}
        for c in cks + ([{"path": f"runs/{name}/eval_rdev.json",
                          "sha256": ent["eval_rdev_sha256"]}] if ent["eval_rdev_sha256"] else []):
            want = sums.get(c["path"])
            if want is None:
                problems.append(f"{name}: {c['path']} has no box sha256 (--box-sums)")
            elif want != c["sha256"]:
                problems.append(f"{name}: {c['path']} is not the box's file (sha256)")
        # the other way: every file the box's sums list under the run is here (a missing
        # eval_rdev.json would otherwise be recorded as none, and the gate would count it 0)
        for path in sorted(p for p in sums if p.startswith(f"runs/{name}/")):
            if not (ROOT / path).is_file():
                problems.append(f"{name}: {path} is in the box's sums but not here (pull again)")
        if done and fail is not None:
            problems.append(f"{name}: listed in failures.json but its status is {box['status']!r}")
        if done:
            cfg_p = d / "config.json"
            if not cfg_p.is_file():
                problems.append(f"{name}: no config.json")
            else:
                problems += [f"{name}: {p}" for p in
                             config_problems(json.loads(cfg_p.read_text()), run["seed"],
                                             defaults)]
            if ent["metrics_sha256"] is None:
                problems.append(f"{name}: no metrics.csv")
        elif fail is None:
            problems.append(f"{name}: status {box.get('status')!r} and not in failures.json")
        entries.append(ent)
    data, data_problems = box_data(sums)
    problems += data_problems
    unlisted = sorted(set(failures) - {r["name"] for r in runs})
    if unlisted:
        problems.append(f"failures.json lists runs outside the round: {unlisted}")
    if problems:
        raise CheckFailed(f"the manifest has {len(problems)} problem(s):\n  "
                          + "\n  ".join(problems))
    return {"record": "e0-manifest", "round": ROUND, "tag": gate["tag"],
            "tag_object": gate["tag_object"], "commit": gate["commit"], "runs": entries,
            "box_data_sha256": data, "round_files": round_files(), "round_check": check,
            "box_sums_used": len(sums)}


def cmd_manifest(a) -> None:
    if a.tag != TAG:
        raise CheckFailed(f"E0's tag is {TAG}, not {a.tag}")
    gate = tag_gate()
    t0 = time.time()
    path = ROOT / MANIFEST
    if path.exists():
        raise CheckFailed(f"{rel(path)} exists: write-once")
    inputs: dict[str, str] = {}
    failures = read_failures(inputs)
    check = round_check(failures)
    runs = RL.runs("main")
    sums = box_sums(a.box_sums)
    for p in a.box_sums:
        inputs[rel(p)] = sha256_file(p)
    m = build_manifest(sums, runs, failures, check, tag_defaults(gate["commit"]), gate)
    m["provenance"] = provenance(t0, gate, inputs)
    write_new(path, m)


# ------------------------------------------------------------------------- the gate


def eval_problems(rows, names: list[str]) -> list[str]:
    """How an eval_rdev.json departs from train_dr.py's: its circuits are not R_DEV's in order,
    or a row is not a dict, or its ``completion`` (or ``gap``, where present) is not a number
    or null. A NaN is a number: it is not completed."""
    if not isinstance(rows, dict) or list(rows) != names:
        return ["its circuits are not R-dev's (R_DEV, in order)"]
    out = []
    for n in names:
        row = rows[n]
        if not isinstance(row, dict):
            out.append(f"{n}: the row is not an object")
            continue
        for key in ("completion", "gap"):
            v = row.get(key)
            if key in row and v is not None and (isinstance(v, bool)
                                                 or not isinstance(v, int | float)):
                out.append(f"{n}: {key} {v!r} is not a number or null")
        if "completion" not in row:
            out.append(f"{n}: no completion")
    return out


def load_manifest(gate: dict, inputs: dict) -> dict:
    """The committed manifest, of this tag and run list, with each run's config.json,
    metrics.csv and eval_rdev.json re-hashed against it, and failures.json as it was then."""
    if not committed(MANIFEST):
        raise CheckFailed(f"{MANIFEST} must be committed before the gate runs")
    m = read_json(MANIFEST, inputs)
    if m.get("record") != "e0-manifest":
        raise CheckFailed(f"{MANIFEST} is not E0's manifest")
    if m.get("tag") != TAG or m.get("commit") != gate["commit"]:
        raise CheckFailed(f"the manifest is of {m.get('tag')} ({str(m.get('commit'))[:12]}), "
                          f"not {TAG}")
    if [e["name"] for e in m["runs"]] != [r["name"] for r in RL.runs("main")]:
        raise CheckFailed("the manifest's runs are not the run list's")
    for ent in m["runs"]:
        for f in ("config", "metrics", "eval_rdev"):
            want = ent.get(f"{f}_sha256")
            p = ROOT / "runs" / ent["name"] / f"{f}.{'csv' if f == 'metrics' else 'json'}"
            if want is not None:
                if not p.is_file() or sha256_file(p) != want:
                    raise CheckFailed(f"{ent['name']}: {rel(p)} is not the manifest's")
                inputs[rel(p)] = want
                if f == "eval_rdev":
                    bad = eval_problems(json.loads(p.read_text()), r_dev())
                    if bad:
                        raise CheckFailed(f"{ent['name']}: {rel(p)}: {'; '.join(bad[:4])}")
            elif p.is_file():
                raise CheckFailed(f"{ent['name']}: {rel(p)} exists but the manifest has none")
    then = (m.get("provenance") or {}).get("inputs") or {}
    have = {}
    failures = read_failures(have)
    if have.get(FAILURES) != then.get(FAILURES):
        raise CheckFailed(f"{FAILURES} is not as it was at the manifest")
    inputs.update(have)
    if set(failures) != {e["name"] for e in m["runs"] if e.get("failure")}:
        raise CheckFailed("failures.json and the manifest list different runs")
    return m


def seed_record(ent: dict, names: list[str]) -> dict:
    """One seed's count of completed R-dev circuits (completion > COMPLETED in its eval_rdev.json)
    and the per-circuit completion and, on completed circuits, the gap to the s-OCP lap. A seed
    failures.json lists, or one with no eval_rdev.json, counts 0."""
    rec = {"name": ent["name"], "seed": ent["seed"], "count": 0, "counted_as_zero": None,
           "completion_by_circuit": None, "gap_by_circuit_completed": None}
    if ent.get("failure"):
        rec["counted_as_zero"] = f"failed twice ({ent['failure'].get('reason')})"
        return rec
    if not ent.get("eval_rdev_sha256"):
        rec["counted_as_zero"] = "no eval_rdev.json"
        return rec
    rows = json.loads((ROOT / "runs" / ent["name"] / "eval_rdev.json").read_text())
    bad = eval_problems(rows, names)
    if bad:
        raise CheckFailed(f"{ent['name']}: eval_rdev.json: {'; '.join(bad[:4])}")
    comp = {n: num(rows[n].get("completion")) for n in names}
    done = [n for n in names if comp[n] > COMPLETED]
    rec.update(count=len(done), completion_by_circuit=comp,
               gap_by_circuit_completed={n: num(rows[n].get("gap")) for n in done})
    return rec


def decide(counts: list[int]) -> dict:
    """The gate: the median of the seeds' counts, and pass when it is at least PASS_MEDIAN."""
    med = statistics.median(counts)
    return {"median": med, "pass": med >= PASS_MEDIAN}


def compute_gate(m: dict, names: list[str]) -> dict:
    seeds = [seed_record(e, names) for e in m["runs"]]
    counts = [s["count"] for s in seeds]
    return {"label": LABEL, "threshold": {"completed_if_completion_above": COMPLETED,
                                          "pass_if_median_at_least": PASS_MEDIAN},
            "counts": counts, **decide(counts),
            "seeds_at_least_pass_median": sum(c >= PASS_MEDIAN for c in counts),
            "per_seed": {s["name"]: s for s in seeds}}


def cmd_gate(a) -> None:
    gate = tag_gate()
    t0 = time.time()
    for p in (GATE, LOCK):
        if (ROOT / p).exists() or ever_committed(p):
            raise CheckFailed(f"{p} exists or was committed: the gate runs once")
    inputs: dict[str, str] = {}
    m = load_manifest(gate, inputs)
    fd = os.open(ROOT / LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    os.write(fd, f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {gate['commit']}\n".encode())
    os.close(fd)
    rec = {"record": "e0-gate", **compute_gate(m, r_dev()), "tag": gate,
           "manifest_sha256": inputs[MANIFEST]}
    rec["provenance"] = provenance(t0, gate, inputs)
    write_new(ROOT / GATE, rec)
    print(f"E0 gate ({LABEL}): counts {rec['counts']}, median {rec['median']}, "
          f"{'PASS' if rec['pass'] else 'FAIL'} (>= {PASS_MEDIAN}); seeds at "
          f"{PASS_MEDIAN}+: {rec['seeds_at_least_pass_median']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="the run manifest, after pull-round")
    m.add_argument("--tag", required=True, help=f"E0's tag, {TAG}")
    m.add_argument("--box-sums", nargs="+", required=True,
                   help="the box's sha256sum lines, taken before it is deleted")
    sub.add_parser("gate", help="the gate, once, after the manifest is committed")
    a = ap.parse_args(argv)
    {"manifest": cmd_manifest, "gate": cmd_gate}[a.cmd](a)


if __name__ == "__main__":
    main()
