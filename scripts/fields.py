"""Receptive fields, step 2: the stages of the design (docs/interp/receptive-fields-design.md).

    uv run python scripts/fields.py <stage> [options]

Pre-tag stages (design Sections 8, 10 and 15; no trained network's field is computed):

    keys            the key lists (A2 set, anchored finals and earlier snapshots) with their
                    strata and the loop audit (stored horizons against the loop length),
                    and Gate 0 on them (no solves)
    repolish        one REFINE_IPOPT re-polish per key, its movement recorded (Section 5);
                    a key that fails to re-polish is a listed 'numerical' exclusion; the
                    stage passes when every key was processed without a crash
    gate1           KKT fields against Richardson FD of warm re-solves on the stratified
                    sample, with the planted bugs
    gate2           network AD against FD for a random actor, the planted edge keeper, the
                    linear driver and the hybrid control (the trained architecture in the loop)
    gate3           teacher-forced wiring of trained snapshots (no derivatives) with the
                    planted wiring errors; --all checks every key with a det row
    gate4           G* of the optimum on every A2 key (all 481; --no-twins leaves out the 6
                    horizon twins); keys outside the band are listed exclusions
    gate5           the KKT replay through the network-side loop, and CasADi against JAX
                    state tangents, on the Gate 1 sample
    controls        RF-C1..C5 and the unlawful readers through the network pipeline
    opt-fields      the optimum's fields per key (A2 and anchored), persisted with their
                    exclusion flags (agent-independent on the A2 keys)
    qp-fields       Qp's fields on the primary A2 keys (H1-typical's lawful-reader reference;
                    Qp is untrained)
    phase-ceiling   the optimum's fields across each A2 cell's phases (from opt-fields)
    calibrate       branch consistency's value tolerance from the controls (the linear driver,
                    the hybrid control and the lane keeper) and Gate 3's every-key errors
    prereg          the pre-registration record src/apexzero/fields/prereg.json (key lists,
                    regime cell sets, gate2-trained and dose keys, network checkpoints, the
                    stamped passing outputs above, the calibrated tolerance, design and code
                    hashes), from a clean tree: the last step before the tag

Post-tag stages (refused unless the freeze check passes; ``--post-tag-deviation REASON``
logs a dated deviation and waives only a changed-code or changed-design-document problem):

    harness-f32     the float32 harness runs (x64 off) of every network without step 1
                    rollouts on the A2 keys: their Gate 3 and branch-consistency reference
    gate2-trained   Gate 2's rule on every H2', H4 and H4-transfer network at the record's
                    12 keys (a failing network is descriptive)
    fields-a2       the networks' fields on the A2 keys (H2', H4, H4-transfer, H1-typical)
    fields-anchored the networks' fields on the anchored finals (H1-fast)
    fields-earlier  each earlier snapshot's fields on its own anchored keys (descriptive)
    fields-anchored-h4
                    the 18 H4 networks on the pooled anchored finals (the anchored-key H4,
                    descriptive; off-policy for every network but a key's owner)
    dose            the width dose-response subsample (the record's keys), optimum and the
                    fixed dose networks
    h3-keys         H3's new key solves (step 2's own key cache)
    h3-lambda       G2's lambda-grid rollouts and lambda*_lin at H3's units
    analyse         the analysis: --confirmatory runs once, on complete recorded inputs, into
                    the fixed outputs/fields/analyse/decisions.json; before it, a
                    non-confirmatory run gives the coverage and plumbing only

Each stage writes under outputs/fields/<stage>/ (``--out`` moves the root; the confirmatory
analysis refuses it) and logs to runs/fields-<stage>.log (``--log-dir``), with a done/total
line at least once a minute and a note in runs/activity.md while it runs (only when the log
is under runs/). Every output carries a provenance stamp (``Provenance``: the code hash at
the start and just before writing, which must agree, HEAD, the changed paths against HEAD,
argv, the partial options in effect, start and end times, any deviation), embedded in a JSON
object or as a <file>.provenance.json sidecar. Network stages run in float64: this script
switches x64 on before a pre-tag stage runs (a post-tag stage after its freeze check), and
runs step 1's float32 harness only in the x64-off stages (calibrate, harness-f32). XLA is held
to 2 threads unless --threads says otherwise.

Freeze rule (design Sections 9 and 15; ``freeze_problems``): the tag receptive-fields-prereg-v1
must exist as an annotated tag, be on origin as the same object, and match the write-once
outputs/fields/tag.json the first post-tag run writes, which must then be committed
(``git add -f``) before the next post-tag stage and is compared with its first committed
version from then on; step 1's psychophysics-prereg-v2 must be its pinned object; nothing
under src/ (all of it), scripts/fields.py or scripts/_bootstrap.py may differ from the tag
(working tree, later commits or untracked files, and byte by byte, so an assume-unchanged or
skip-worktree edit or a .py file in an ignored directory is caught; later commits touching
only docs or findings are fine); step 1's frozen code must be its own tag's; the
pre-registration record at the tag must exist, register the design document's blob at the
tag (its bytes), hash the code the tag holds (``prereg.tree_code_hash``) and come from a base
commit that differs from the tag only by the record; and it must match the disk: the key
files and lists, the re-polish movement, step 1's design and support file, the design
document, the code hash, the package versions, every network's checkpoint, the bound pre-tag
outputs and the calibration. Post-tag stages iterate the record's frozen key lists and load
the recorded checkpoints, never a fresh glob, and refuse a module imported from inside the
repository but outside the frozen code. After the check, ``network.unlock_trained`` allows
trained networks' fields. Pre-tag stages refuse once the tag exists, unless
``--post-tag-deviation`` logs it, and even then never write into outputs/fields ('prereg'
always refuses).

Run-once (design Section 15): ``analyse --confirmatory`` refuses when
outputs/fields/analyse/decisions.json exists or was ever committed; its printed ``git add -f``
line commits it, and a non-confirmatory analysis computes the decisions only after that
committed, stamped confirmatory file exists (``stages.confirmatory_done``).
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _threads(argv: list[str]) -> int | None:
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return None


_T = _threads(sys.argv)
if _T is not None or "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               f"intra_op_parallelism_threads={_T or 2}")

import _bootstrap  # noqa: F401  (makes src/ importable)

ROOT = Path(__file__).resolve().parents[1]
TAG = "receptive-fields-prereg-v1"
STEP1_TAG = "psychophysics-prereg-v2"
STEP1_TAG_OBJECT = "f68d0430cccf538097d08d3d07970ce464a1dc89"   # pushed; never moves
STEP1_FROZEN = ("src/apexzero/interp", "scripts/psychophysics.py")
# nothing here may change after the tag (prereg.CODE_ROOTS: the code hash's roots); all of
# src/ (anything there is importable: scripts/_bootstrap.py puts it first on sys.path)
CODE_PATHS = ("src", "scripts/fields.py", "scripts/_bootstrap.py")
DESIGN_DOC = "docs/interp/receptive-fields-design.md"
REMOTE = "origin"
PRE_TAG = ("keys", "repolish", "gate1", "gate2", "gate3", "gate4", "gate5", "controls",
           "opt-fields", "qp-fields", "phase-ceiling", "calibrate", "prereg")
# calibrate builds float32 references in this process (the harness needs x64 off) and runs the
# float64 dry runs in subprocesses; harness-f32 runs the float32 harness after the tag. They
# are the two stages that leave x64 off here.
X64_OFF = ("calibrate", "harness-f32")
# Its controls: the planted linear driver on the A2 cells where it survives the window
# (mirror pairs), the hybrid control (network.HYBRID_CELLS) and step 1's lane keeper
# (generic-callable path), as in tests/.
CAL_LINEAR_CELLS = (344, 345, 62, 63, 66, 67)
CAL_LINEAR_STEPS = 200
CAL_KEEPER_STARTS = ((380, 0), (51, 1))
CAL_RANDOM_STARTS = ((62, 0), (345, 1))
POST_TAG = ("harness-f32", "gate2-trained", "fields-a2", "fields-anchored", "fields-earlier",
            "fields-anchored-h4", "dose", "h3-keys", "h3-lambda", "analyse")
STAGES = PRE_TAG + POST_TAG
OUT = ROOT / "outputs" / "fields"
PREREG_REL = "src/apexzero/fields/prereg.json"
DECISIONS_REL = "outputs/fields/analyse/decisions.json"      # the run-once output (committed)
HARNESS_CHUNK = 64          # float32 harness runs per compiled call (harness-f32)
DEFAULT_KINDS = ["a2", "finals", "earlier"]


# ------------------------------------------------------------------------- git and the freeze


def _git(*args: str, root: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def _git_bytes(*args: str, root: Path = ROOT) -> subprocess.CompletedProcess:
    """git with its output as bytes (a hash of a blob must not go through text decoding)."""
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)


def _tag_object(tag: str, root: Path = ROOT) -> tuple[str, str] | None:
    """(object sha, object type) of refs/tags/<tag> ('tag' for an annotated tag, 'commit'
    for a lightweight one); None when the tag does not exist."""
    r = _git("rev-parse", "--verify", "-q", f"refs/tags/{tag}", root=root)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    obj = r.stdout.strip()
    t = _git("cat-file", "-t", obj, root=root)
    return obj, t.stdout.strip()


def _tag_commit(tag: str, root: Path = ROOT) -> str | None:
    """The commit a tag points to (None when the tag does not exist)."""
    r = _git("rev-parse", "--verify", "-q", f"refs/tags/{tag}^{{commit}}", root=root)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _head(root: Path = ROOT) -> str:
    r = _git("rev-parse", "HEAD", root=root)
    return r.stdout.strip() if r.returncode == 0 else f"<git error: {r.stderr.strip()[:200]}>"


def remote_tag_object(tag: str, root: Path = ROOT, remote: str = REMOTE) -> str | None:
    """The object refs/tags/<tag> names on ``remote`` (``git ls-remote``); None when the
    remote does not have it; raises RuntimeError when the remote cannot be read."""
    r = _git("ls-remote", remote, f"refs/tags/{tag}", root=root)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:200] or f"git ls-remote {remote} failed")
    for ln in r.stdout.splitlines():
        sha, _, ref = ln.partition("\t")
        if ref.strip() == f"refs/tags/{tag}":
            return sha.strip()
    return None


def tag_identity(tag: str = TAG, root: Path = ROOT) -> dict | None:
    """The tag's object, type, commit and tagger date (None when it does not exist)."""
    t = _tag_object(tag, root)
    if t is None:
        return None
    d = _git("for-each-ref", "--format=%(taggerdate:iso-strict)", f"refs/tags/{tag}", root=root)
    return {"tag": tag, "object": t[0], "type": t[1], "commit": _tag_commit(tag, root),
            "tagger_date": d.stdout.strip()}


def changed_since(tag: str, paths: tuple[str, ...], root: Path = ROOT) -> list[str]:
    """Files under ``paths`` that differ from the tagged commit (working tree or later
    commits), plus untracked ones; [] when they are the tagged code. A git failure is itself
    reported (as '<git error: ...>'), never read as 'no change'."""
    r = _git("diff", "--name-only", tag, "--", *paths, root=root)
    if r.returncode != 0:
        return [f"<git error: {r.stderr.strip()[:200]}>"]
    changed = [ln for ln in r.stdout.splitlines() if ln.strip()]
    u = _git("ls-files", "--others", "--exclude-standard", "--", *paths, root=root)
    if u.returncode != 0:
        return changed + [f"<git error: {u.stderr.strip()[:200]}>"]
    return changed + [f"{ln} (untracked)" for ln in u.stdout.splitlines() if ln.strip()]


def tagged_record(tag: str = TAG, root: Path = ROOT) -> dict | None:
    """The pre-registration record as the tagged commit holds it (None if absent)."""
    r = _git("show", f"{tag}:{PREREG_REL}", root=root)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return json.loads(r.stdout)


def tagged_doc_sha256(tag: str = TAG, root: Path = ROOT) -> str | None:
    """sha256 of the design document's blob at the tag, from its bytes (None if absent)."""
    r = _git_bytes("cat-file", "blob", f"{tag}:{DESIGN_DOC}", root=root)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def _rel(path: Path, root: Path = ROOT) -> str | None:
    """``path`` relative to ``root`` (POSIX), None when it lies outside."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return None


def committed_versions(rel: str, root: Path = ROOT) -> list[tuple[str, bytes | None]]:
    """[(commit, bytes or None where the commit deletes it)] of every commit on any ref that
    touched ``rel``, oldest first; [] when it was never committed. Raises RuntimeError when
    git fails."""
    r = _git("log", "--all", "--reverse", "--format=%H", "--", rel, root=root)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:200] or "git log failed")
    out = []
    for c in r.stdout.split():
        b = _git_bytes("show", f"{c}:{rel}", root=root)
        out.append((c, b.stdout if b.returncode == 0 else None))
    return out


def _disk_files(paths: tuple[str, ...], root: Path) -> dict[str, str]:
    """{repo-relative path: sha256} of the .py files on disk under ``paths`` (every one,
    tracked, untracked or ignored)."""
    from apexzero.fields import prereg

    root = Path(root)
    out = {}
    for r in paths:
        p = root / r
        files = p.rglob("*.py") if p.is_dir() else ([p] if p.is_file() else [])
        for f in files:
            if f.is_file():
                out[f.relative_to(root).as_posix()] = prereg.sha256_file(f)
    return out


def byte_changes(rev: str, paths: tuple[str, ...], root: Path = ROOT) -> dict[str, list[str]]:
    """{path: [sha256 on disk or 'missing', sha256 at ``rev`` or 'missing']} of every .py file
    under ``paths`` whose bytes on disk differ from its blob at ``rev`` (or exist on one side
    only): what git's own diff misses (assume-unchanged, skip-worktree, ignored directories).
    Raises RuntimeError when git fails."""
    from apexzero.fields import prereg

    disk = _disk_files(paths, root)
    blobs = prereg.tree_blobs(Path(root), rev, paths)
    tree = {k: prereg.sha256_bytes(v) for k, v in blobs.items()
            if k.endswith(".py") or k in paths}
    out = {}
    for k in sorted(set(disk) | set(tree)):
        a, b = disk.get(k, "missing"), tree.get(k, "missing")
        if a != b:
            out[k] = [a, b]
    return out


ENV_COMPARED = ("python", "packages")      # the environment items a freeze check compares


def env_problems(recorded: dict | None, now: dict | None) -> list[str]:
    """The Python and package versions that differ between two ``environment()`` blocks (not
    the machine: the design expects a separate computing machine); [] when ``recorded`` is
    None."""
    if recorded is None:
        return []
    now = now or {}
    out = []
    if recorded.get("python") != now.get("python"):
        out.append(f"python {recorded.get('python')} -> {now.get('python')}")
    rp, np_ = recorded.get("packages") or {}, now.get("packages") or {}
    for k in sorted(set(rp) | set(np_)):
        if rp.get(k) != np_.get(k):
            out.append(f"{k} {rp.get(k)} -> {np_.get(k)}")
    return out


def step1_problems(root: Path = ROOT) -> list:
    """Step 1's freeze (``prereg.Problem`` 'step1'; a git failure 'git'): its tag must exist,
    be its pinned object (this repository), and its frozen code (src/apexzero/interp,
    scripts/psychophysics.py) must not differ from it, in git or byte by byte."""
    from apexzero.fields import prereg

    P = prereg.Problem
    t1 = _tag_object(STEP1_TAG, root)
    if t1 is None:
        return [P("step1", f"step 1's tag {STEP1_TAG} is missing: its frozen code cannot be "
                  "checked")]
    out = []
    if Path(root).resolve() == ROOT.resolve() and t1[0] != STEP1_TAG_OBJECT:
        out.append(P("step1", f"step 1's tag {STEP1_TAG} is {t1[0][:12]}, not its pinned "
                     f"object {STEP1_TAG_OBJECT[:12]}"))
    ch1 = changed_since(STEP1_TAG, STEP1_FROZEN, root)
    if ch1:
        code = "git" if any(c.startswith("<git error") for c in ch1) else "step1"
        out.append(P(code, f"step 1's frozen code differs from {STEP1_TAG}: {ch1[:8]}"))
    try:
        b1 = byte_changes(STEP1_TAG, STEP1_FROZEN, root)
    except RuntimeError as e:
        return out + [P("git", f"git failed reading {STEP1_TAG}: {e}")]
    hidden = sorted(set(b1) - {c.split(" ")[0] for c in ch1})
    if hidden:
        out.append(P("step1", f"step 1's frozen code differs from {STEP1_TAG} byte by byte "
                     f"(not in git's diff): {hidden[:8]}"))
    return out


def _fields_out(root: Path = ROOT) -> Path:
    """outputs/fields of a repository root (the fixed home of tag.json and deviations.jsonl)."""
    return Path(root) / "outputs" / "fields"


def _pins_problems(pins: dict, root: Path) -> list[str]:
    """Each recorded checkpoint re-hashed, and whether its name still resolves to it (a newer
    checkpoint in the run directory means the run changed)."""
    from apexzero.fields import prereg
    from apexzero.fields import stages as S

    out = []
    for name, pin in sorted(pins.items()):
        p = Path(root) / pin["path"]
        if not p.exists() or prereg.sha256_file(p) != pin["sha256"]:
            out.append(f"{name}: the checkpoint {pin['path']} is not the recorded one")
            continue
        now = S.checkpoint_path(name)
        if now is None or Path(now).resolve() != p.resolve():
            out.append(f"{name}: its last checkpoint is now {now}, not the recorded {pin['path']}")
    return out


def _rollout_problems(recorded: dict, root: Path) -> list[str]:
    """Each recorded rollout file's det group re-hashed."""
    from apexzero.fields import prereg

    out = []
    for who, v in sorted(recorded.items()):
        p = Path(root) / v["path"]
        if not p.exists() or prereg.det_hash(p) != v["det_sha256"]:
            out.append(f"{who}: the stored rollouts {v['path']} are not the recorded ones")
    return out


def rollout_record(final_timesteps: dict, root: Path = ROOT, earlier=()) -> dict:
    """{agent@timesteps: {path, det_sha256}} of every registry agent's final rollout file and
    of the earlier snapshots ``earlier`` ((agent, timesteps) pairs: fields-earlier's
    restarts) (Gate 3, branch consistency and the anchored restarts read their det groups)."""
    from apexzero.fields import prereg
    from apexzero.interp import io

    out = {}
    for a, ts in sorted(list(final_timesteps.items()) + [(a, int(t)) for a, t in earlier]):
        p = io.rollout_path(a, int(ts))
        if p.exists():
            out[f"{a}@{ts}"] = {"path": str(Path(p).resolve().relative_to(Path(root).resolve())),
                                "det_sha256": prereg.det_hash(p)}
    return out


def _verify_on_disk(rec: dict, root: Path, out_root: Path | None = None) -> list[str]:
    """``prereg.verify_record`` against this repository's key store, re-polish rows, design,
    support file, checkpoints and the bound outputs under ``out_root`` (the root the stage
    reads; default outputs/fields)."""
    from apexzero.fields import keys as K
    from apexzero.fields import prereg
    from apexzero.interp import io, stimuli

    out_root = Path(out_root or OUT)
    rows_path = out_root / "repolish" / "repolish.jsonl"
    rows = ([json.loads(ln) for ln in rows_path.read_text().splitlines() if ln.strip()]
            if rows_path.exists() else None)
    bank, cells = stimuli.load_design(io.STIMULI_DIR)
    ft = {a: K.final_timesteps(a) for a in rec.get("final_timesteps", {})}
    sup = io.SUPPORT_DIR / "support.json"
    probs = prereg.verify_record(rec, root, io.KEYS_DIR, lambda: prereg.key_sets(),
                                 rows, stimuli.design_hash(bank, cells), ft, out_root=out_root,
                                 networks=lambda pins: _pins_problems(pins, root),
                                 support_sha256=prereg.sha256_file(sup) if sup.exists() else "",
                                 rollouts=lambda rr: _rollout_problems(rr, root))
    if rows is None:
        probs.append(prereg.Problem("repolish", f"no re-polish rows at {rows_path} to check "
                                    "against the record"))
    return probs


def freeze_problems(tag: str = TAG, root: Path = ROOT, verify=_verify_on_disk,
                    remote: str | None = REMOTE, pin_path: Path | None = None) -> list:
    """Why a post-tag stage may not run on what is checked out (module docstring), as
    ``prereg.Problem`` strings with a category: a missing, lightweight, unpushed or moved tag,
    or a tag.json pin that is uncommitted, missing or not its first committed version
    ('tag'); step 1's tag missing, moved or its frozen code changed ('step1'); any change
    since the tag under the code paths, in git or byte by byte, or other package versions
    than the record's ('code_changed'; a git failure: 'git'); a missing record, one that does
    not register the design document's blob at the tag, whose code hash is not the tagged
    code's, or whose base commit is missing, not an ancestor of the tag or differs from it by
    more than the record ('record'); and every way the disk no longer matches the record
    (``verify(record, root)``). ``remote`` None skips the origin check (tests); ``pin_path``:
    the write-once tag.json (default outputs/fields/tag.json; its committed version is
    checked only when it lies inside ``root``)."""
    from apexzero.fields import prereg

    P = prereg.Problem
    t = _tag_object(tag, root)
    if t is None:
        return [P("tag", f"the git tag {tag} does not exist yet (design Section 9: the fields "
                  "of trained networks are computed only after it)")]
    out = []
    if t[1] != "tag":
        out.append(P("tag", f"{tag} is a lightweight tag: it must be annotated"))
    if remote is not None:
        try:
            on = remote_tag_object(tag, root, remote)
            if on != t[0]:
                out.append(P("tag", f"{tag} is not on {remote} as {t[0][:12]} (there: {on}); "
                             "push the tag before any post-tag stage"))
        except RuntimeError as e:
            out.append(P("tag", f"cannot confirm {tag} on {remote}: {e}"))
    pin_path = Path(pin_path) if pin_path is not None else _fields_out(root) / "tag.json"
    tag_commit = _tag_commit(tag, root)
    if pin_path.exists():
        pin = json.loads(pin_path.read_text())
        if pin.get("object") != t[0] or pin.get("commit") != tag_commit:
            out.append(P("tag", f"the tag moved since the first post-tag run ({pin_path}: "
                         f"{str(pin.get('object'))[:12]}, now {t[0][:12]})"))
    out += _pin_committed_problems(pin_path, root, t[0], tag_commit)
    ch = changed_since(tag, CODE_PATHS, root)
    if any(c.startswith("<git error") for c in ch):
        out.append(P("git", f"git failed on the code paths: {ch[:2]}"))
    elif ch:
        out.append(P("code_changed", f"code differs from the tag {tag}: {ch[:8]}"))
    try:
        hidden = sorted(set(byte_changes(tag, CODE_PATHS, root))
                        - {c.split(" ")[0] for c in ch})
        if hidden:
            out.append(P("code_changed", f"code differs from the tag {tag} byte by byte (not "
                         f"in git's diff: assume-unchanged, skip-worktree or ignored): "
                         f"{hidden[:8]}"))
    except RuntimeError as e:
        out.append(P("git", f"git failed reading the tag's code: {e}"))
    out += step1_problems(root)
    rec = tagged_record(tag, root)
    if rec is None:
        out.append(P("record", f"no pre-registration record {PREREG_REL} at the tag {tag}"))
        return out
    doc = tagged_doc_sha256(tag, root)
    if not rec.get("design_doc_sha256") or doc != rec["design_doc_sha256"]:
        out.append(P("record", "the design document at the tag is not the one the record "
                     "registers"))
    out += _record_base_problems(rec, tag, root)
    env = env_problems(rec.get("environment"), environment())
    if env:
        out.append(P("code_changed", f"package versions differ from the record's: {env[:6]}"))
    if verify is not None:
        out.extend(p if hasattr(p, "code") else P("", str(p)) for p in verify(rec, root))
    return out


def _pin_committed_problems(pin_path: Path, root: Path, obj: str, commit: str | None) -> list:
    """The tag.json pin anchored in git (design Section 15: write-once): once committed, its
    first committed version must name the tag (object and commit) and equal the file on disk;
    a pin on disk that was never committed must be committed before the next post-tag stage.
    Skipped for a pin outside ``root`` (tests)."""
    from apexzero.fields import prereg

    P = prereg.Problem
    rel = _rel(pin_path, root)
    if rel is None:
        return []
    try:
        vers = [v for v in committed_versions(rel, root) if v[1] is not None]
    except RuntimeError as e:
        return [P("git", f"git failed reading {rel}'s history: {e}")]
    if not vers:
        if Path(pin_path).exists():
            return [P("tag", f"{rel} is not committed: run `git add -f {rel} && git commit` "
                      "before the next post-tag stage")]
        return []
    first = json.loads(vers[0][1])
    out = []
    if first.get("object") != obj or first.get("commit") != commit:
        out.append(P("tag", f"the tag moved since its pin was committed ({rel} at "
                     f"{vers[0][0][:12]}: {str(first.get('object'))[:12]}, now {obj[:12]})"))
    if not Path(pin_path).exists():
        out.append(P("tag", f"{rel} is missing on disk but committed at {vers[0][0][:12]}"))
    elif json.loads(Path(pin_path).read_text()) != first:
        out.append(P("tag", f"{rel} on disk is not its first committed version"))
    return out


def _record_base_problems(rec: dict, tag: str, root: Path) -> list:
    """The record against the tagged code ('record', never waivable): its code hash must be
    the code the tag holds (``prereg.tree_code_hash``), and its base commit must exist, be an
    ancestor of the tag and differ from it under the code paths and the design document by
    the record alone."""
    from apexzero.fields import prereg

    P = prereg.Problem
    out = []
    try:
        if prereg.tree_code_hash(Path(root), tag) != rec.get("code_hash"):
            out.append(P("record", "the code at the tag is not the code the record was built "
                         "from (re-run prereg on the tagged code and re-tag)"))
    except RuntimeError as e:
        out.append(P("git", f"git failed hashing the code at the tag: {e}"))
    base = rec.get("prereg_base_commit")
    if not base or _git("cat-file", "-e", f"{base}^{{commit}}", root=root).returncode != 0:
        out.append(P("record", f"the record's base commit {base!r} is missing or unknown"))
        return out
    if _git("merge-base", "--is-ancestor", base, tag, root=root).returncode != 0:
        out.append(P("record", f"the record's base commit {base[:12]} is not an ancestor of "
                     f"the tag {tag}"))
        return out
    d = _git("diff", "--name-only", base, tag, "--", *CODE_PATHS, DESIGN_DOC, root=root)
    if d.returncode != 0:
        out.append(P("git", f"git diff {base[:12]} {tag}: {d.stderr.strip()[:200]}"))
    else:
        extra = [ln for ln in d.stdout.splitlines() if ln.strip() and ln != PREREG_REL]
        if extra:
            out.append(P("record", f"the tag differs from the record's base commit "
                         f"{base[:12]} by more than the record: {extra[:8]}"))
    return out


def diff_evidence(tag: str = TAG, root: Path = ROOT) -> dict:
    """What a deviation records: the sha256 of ``git diff <tag>`` over the code paths and the
    design document, the untracked files under them with their sha256, every .py file whose
    bytes differ from the tag's ('byte_changes': [on disk, at the tag], so a hidden edit
    leaves evidence too), HEAD, the tag's commit and object and the record's base commit."""
    from apexzero.fields import prereg

    d = _git_bytes("diff", tag, "--", *CODE_PATHS, DESIGN_DOC, root=root)
    u = _git("ls-files", "--others", "--exclude-standard", "--", *CODE_PATHS, DESIGN_DOC,
             root=root)
    untracked = {ln: prereg.sha256_file(Path(root) / ln) for ln in u.stdout.splitlines()
                 if ln.strip() and (Path(root) / ln).is_file()}
    try:
        changes = byte_changes(tag, CODE_PATHS, root)
    except RuntimeError as e:
        changes = {"<git error>": [str(e), ""]}
    t = _tag_object(tag, root)
    rec = tagged_record(tag, root) if t else None
    return {"diff_sha256": hashlib.sha256(d.stdout).hexdigest(), "diff_bytes": len(d.stdout),
            "untracked": untracked, "byte_changes": changes, "head": _head(root),
            "tag_commit": _tag_commit(tag, root), "tag_object": t[0] if t else None,
            "prereg_base_commit": (rec or {}).get("prereg_base_commit")}


def log_deviation(what: str, reason: str, problems: list, category: str, root: Path = ROOT,
                  log_root: Path | None = None, tag: str = TAG) -> dict:
    """Append a dated deviation to <log_root>/deviations.jsonl (default outputs/fields) with
    the problems it waives and ``diff_evidence``; returns the entry (it also goes into every
    output stamp of the run)."""
    from apexzero.fields import prereg

    rec = {"date": time.strftime("%Y-%m-%d"), "time": time.strftime("%H:%M:%S"),
           "stage": what, "category": category, "reason": reason,
           "problems": [{"code": prereg.problem_code(p), "message": str(p)} for p in problems],
           **diff_evidence(tag, root)}
    path = Path(log_root or _fields_out(root)) / "deviations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"deviation logged in {path}: {reason}", flush=True)
    return rec


def pin_tag(tag: str = TAG, root: Path = ROOT, pin_path: Path | None = None) -> Path:
    """Write the write-once tag.json (the tag's identity at the first post-tag run); a later
    run compares against it (``freeze_problems``) and never overwrites it."""
    pin_path = Path(pin_path) if pin_path is not None else _fields_out(root) / "tag.json"
    ident = tag_identity(tag, root)
    if ident is not None and not pin_path.exists():
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pin_path, "x") as fh:
            fh.write(json.dumps({**ident, "pinned_at": _now()}, indent=1) + "\n")
        rel = _rel(pin_path, root)
        if rel is not None:
            print(f"pinned the tag in {rel}: commit it before the next post-tag stage "
                  f"(git add -f {rel} && git commit -m 'Step 2: tag pin')", flush=True)
    return pin_path


def require_tag(what: str, deviation: str | None = None, log_root: Path | None = None,
                root: Path = ROOT, problems=None, pin_path: Path | None = None) -> dict | None:
    """Refuse a post-tag stage unless ``freeze_problems`` is empty. With ``deviation`` (a
    reason), problems that are all waivable (``prereg.WAIVABLE``: changed code, a changed
    design document) are logged as a dated deviation and the stage proceeds; any other
    problem (a missing, lightweight, unpushed or moved tag, a missing record, changed key
    sets, checkpoints, outputs, step 1's code, a git error) always refuses. The first run
    that proceeds writes the write-once tag.json. Returns the deviation entry or None."""
    from apexzero.fields import prereg

    probs = freeze_problems(root=root, pin_path=pin_path) if problems is None else problems
    if not probs:
        pin_tag(root=root, pin_path=pin_path)
        return None
    if deviation and prereg.waivable(probs):
        dev = log_deviation(what, deviation, probs, "post-tag deviation", root, log_root)
        pin_tag(root=root, pin_path=pin_path)
        return dev
    hard = [p for p in probs if prereg.problem_code(p) not in prereg.WAIVABLE]
    why = ("; ".join(hard) + " (a deviation cannot waive these)") if deviation else \
        "; ".join(probs)
    raise SystemExit(f"refusing to run {what}: {why}")


def refuse_after_tag(what: str, deviation: str | None = None, always: bool = False,
                     root: Path = ROOT, log_root: Path | None = None,
                     out_root: Path | None = None, canon: Path | None = None) -> dict | None:
    """Pre-tag stages refuse once the tag exists (design Section 15) unless ``deviation``
    gives a reason, which is logged; ``always`` (the prereg stage) refuses whatever the
    reason. Even with a deviation, a run whose output root ``out_root`` is, contains or lies
    inside the canonical outputs/fields (``canon``) is refused before anything is logged or
    written: it would overwrite the pre-registered outputs the record binds. Returns the
    deviation entry or None."""
    if _tag_object(TAG, root) is None:
        return None
    if always or not deviation:
        extra = "" if always else (" (pass --post-tag-deviation REASON with --out outside "
                                   "outputs/fields to log it as a deviation)")
        raise SystemExit(f"refusing to run {what} after the tag {TAG}: it would change a "
                         f"pre-registered output{extra}")
    if out_root is not None:
        out = Path(out_root).resolve()
        c = Path(canon if canon is not None else _fields_out(root)).resolve()
        if out == c or c in out.parents or out in c.parents:
            raise SystemExit(f"refusing to run {what} after the tag into {out_root}: it would "
                             "overwrite pre-registered outputs; give --out outside "
                             "outputs/fields")
    return log_deviation(what, deviation, [], "pre-tag stage after the tag", root, log_root)


# ------------------------------------------------------------------------- provenance


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _partial(args) -> list[str]:
    """The options that make a run partial (its output cannot be bound by the record or read
    by the confirmatory analysis)."""
    out = []
    if getattr(args, "out", None) is not None and Path(args.out).resolve() != OUT.resolve():
        out.append("--out")
    for name in ("limit", "n", "network"):
        if getattr(args, name, None) not in (None, [], False):
            out.append(f"--{name}")
    stage = getattr(args, "stage", "")
    if stage == "gate2":            # Gate 2's fixed rows (gates.GATE2_CELLS; None: the default)
        given = {k: getattr(args, k, None) for k in ("random", "keeper", "linear", "hybrid")}
        if any(v is not None for v in given.values()):
            from apexzero.fields import gates as G

            out += [f"--{k}" for k, v in given.items()
                    if v is not None and sorted(v) != sorted(G.GATE2_CELLS[k])]
    if stage == "gate3":
        from apexzero.interp import agents

        if not getattr(args, "all", False) or set(args.agent or ["G2"]) != set(agents.AGENTS):
            out.append("gate3 without --all on every agent")
        given = {k: getattr(args, k, None) for k in ("n_anchored", "n_a2")}
        if any(v is not None for v in given.values()):
            from apexzero.fields import gates as G

            least = {"n_anchored": G.GATE3_N_ANCHORED, "n_a2": G.GATE3_N_A2}
            out += [f"--{k.replace('_', '-')}" for k, v in given.items()
                    if v is not None and v < least[k]]
    elif getattr(args, "agent", None):
        out.append("--agent")
    if getattr(args, "quick", False):
        out.append("--quick")
    if getattr(args, "no_twins", False):
        out.append("--no-twins")
    if hasattr(args, "kinds") and sorted(args.kinds) != sorted(DEFAULT_KINDS):
        out.append("--kinds")
    if getattr(args, "stage", "") == "controls" and sorted(getattr(args, "which", [])) != sorted(
            ("RF-C1", "RF-C2", "RF-C3", "RF-C4", "RF-C5", "readers")):
        out.append("--which")
    return out


class Provenance:
    """A stage run's provenance (design Section 15): the code hash at the start, HEAD, the
    paths changed against HEAD, argv, the partial options, the start time and the
    environment (Python and package versions); ``stamp`` re-takes the code hash just before
    an output is written and refuses to write when it differs (the code changed while the
    stage ran), and with ``check_imports`` (post-tag stages) also refuses when a module was
    imported from inside the repository but outside the frozen code (``import_problems``)."""

    def __init__(self, args=None, stage: str = "", root: Path = ROOT):
        from apexzero.fields import prereg

        self.root = Path(root)
        self.stage = stage
        self.code_hash = prereg.code_hash(self.root)
        self.head = _head(self.root)
        self.changed = changed_since("HEAD", CODE_PATHS + (DESIGN_DOC,), self.root)
        self.argv = list(sys.argv)
        self.partial = _partial(args) if args is not None else []
        self.start = _now()
        self.deviation: dict | None = None
        self.tag: dict | None = None
        self.environment = environment(self.root)
        self.check_imports = False

    def stamp(self, files: dict | None = None, content_sha256: str | None = None) -> dict:
        from apexzero.fields import prereg

        now = prereg.code_hash(self.root)
        if now != self.code_hash:
            raise SystemExit(f"refusing to write the {self.stage} output: the code changed "
                             f"while the stage ran ({self.code_hash[:12]} at the start, "
                             f"{now[:12]} now)")
        bad = import_problems(self.root) if self.check_imports else []
        if bad:
            raise SystemExit(f"refusing to write the {self.stage} output: {'; '.join(bad[:4])}")
        out = {"stage": self.stage, "code_hash": self.code_hash, "head": self.head,
               "changed_vs_head": self.changed, "argv": self.argv, "partial": self.partial,
               "start": self.start, "end": _now(), "deviation": self.deviation,
               "tag": self.tag, "environment": self.environment, "files": dict(files or {})}
        if content_sha256 is not None:
            out["content_sha256"] = content_sha256
        return out


_PROV: list[Provenance] = []          # the running stage's provenance (main sets it)


def _prov() -> Provenance:
    if not _PROV:
        _PROV.append(Provenance(None, "?"))
    return _PROV[-1]


def stage_log(args, stage: str, total: int, note: str = ""):
    """io.StageLog writing runs/fields-<stage>.log (or under --log-dir)."""
    from apexzero.interp import io

    class FieldsLog(io.StageLog):
        def __init__(self):
            super().__init__(stage, total, note=note)
            self.log_dir = Path(args.log_dir)
            self.path = self.log_dir / f"fields-{stage}.log"
            self._note = f"fields {stage} [pid {os.getpid()}]"

        def _activity(self, text):
            if self.log_dir.resolve() == io.RUNS_DIR.resolve():
                super()._activity(text)

    return FieldsLog()


def _out(args, stage: str) -> Path:
    d = Path(args.out) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def content_sha256(obj: dict) -> str:
    """sha256 of a JSON object output's content without its 'provenance' stamp, as
    ``io.write_json`` serialises it (a file read back and re-serialised gives the same
    bytes): the embedded stamp's binding of the content."""
    from apexzero.fields import prereg
    from apexzero.interp import io

    return prereg.sha256_bytes(io.json_bytes({k: v for k, v in obj.items() if k != "provenance"}))


def _write(path: Path, obj) -> Path:
    """A JSON output with its stamp: embedded ('provenance', binding the content's sha256,
    ``content_sha256``) in an object, else a sidecar (``_stamped_write``). The stamp (the code
    hash check) is taken before anything is written; the write is atomic."""
    from apexzero.interp import io

    path = Path(path)
    if isinstance(obj, dict):
        io.write_json(path, {**obj, "provenance": _prov().stamp(
            content_sha256=content_sha256(obj))})
    else:
        _stamped_write(path, lambda q: io.write_json(q, obj))
    print(f"wrote {path}", flush=True)
    return path


def _stamp_file(path: Path) -> Path:
    """The sidecar stamp of a non-object output (jsonl, npz, list JSON) with its sha256 (the
    code hash re-checked first: ``Provenance.stamp``)."""
    from apexzero.fields import prereg

    path = Path(path)
    sc = prereg.stamp_path(path)
    sc.write_text(json.dumps(_prov().stamp({path.name: prereg.sha256_file(path)}), indent=1,
                             default=str) + "\n")
    return sc


def _stamped_write(path: Path, write) -> Path:
    """A non-object output written only after the code hash check (design Section 15: taken
    before writing): ``Provenance.stamp`` first (refusing leaves the previous output as it
    was), the old sidecar removed, ``write(tmp)`` into a temporary file beside it (its
    suffix kept: np.savez appends '.npz' otherwise), renamed into place, then stamped.
    Outputs streamed while a stage runs (re-polish's appended rows, opt-fields' per-key
    arrays, calibrate's float64 dry runs) are stamped after the run instead."""
    from apexzero.fields import prereg

    _prov().stamp()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prereg.stamp_path(path).unlink(missing_ok=True)
    tmp = path.with_name(".tmp-" + path.name)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _stamp_file(path)
    return path


def _write_rows(path: Path, rows: list[dict]) -> Path:
    from apexzero.fields import stages as S

    _stamped_write(path, lambda q: S.write_rows(q, rows))
    print(f"wrote {path} ({len(rows)} rows)", flush=True)
    return Path(path)


def _x64() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)


def _limit(refs: list, n: int | None, spread: bool = True) -> list:
    """The first n keys, or n spread evenly over the list (``spread``)."""
    if n is None or n >= len(refs):
        return list(refs)
    if not spread:
        return list(refs[:n])
    import numpy as np

    return [refs[i] for i in np.linspace(0, len(refs) - 1, int(n)).astype(int)]


def _refs(kinds: list[str]):
    """Key refs of the requested kinds: 'a2', 'finals', 'earlier' (anchored)."""
    from apexzero.fields import keys as K

    out = []
    if "a2" in kinds:
        out += list(K.a2_refs())
    if "finals" in kinds:
        out += [r for refs in K.anchored_finals().values() for r in refs]
    if "earlier" in kinds:
        out += [r for refs in K.anchored_earlier().values() for r in refs]
    return out


def _pre_tag(args, what: str) -> None:
    """A pre-tag stage after the tag: refused, or logged as a deviation (``refuse_after_tag``)
    when its --out lies outside outputs/fields."""
    dev = refuse_after_tag(what, getattr(args, "post_tag_deviation", None),
                           always=args.stage == "prereg", out_root=Path(args.out), canon=OUT)
    if dev is not None:
        _prov().deviation = dev


# ------------------------------------------------------------------------- pre-tag stages


def stage_keys(args) -> None:
    """The key lists with their strata, then Gate 0 on the requested kinds."""
    import dataclasses

    from apexzero.fields import gates as G
    from apexzero.fields import keys as K

    out = _out(args, "keys")
    a2 = K.a2_refs()
    finals = K.anchored_finals()
    earlier = K.anchored_earlier()
    rows = []
    for r in a2:
        st = K.strata(r)
        rows.append({**dataclasses.asdict(r), **{f"strata_{k}": v for k, v in
                                                 dataclasses.asdict(st).items()}})
    lists = {"a2": {"n": len(a2), "by_kind_W": _count(a2, lambda r: f"{r.kind}/{r.W}"),
                    "w_twins": [r.key_id for r in a2 if not K.strata(r).w_is_cell],
                    "cap_binds": [r.key_id for r in a2 if K.strata(r).cap_binds],
                    "keys": rows},
             "anchored_finals": {a: {"n": len(v), "snapshot": v[0].snapshot if v else "",
                                     "keys": [r.key_id for r in v]} for a, v in finals.items()},
             "anchored_earlier": {f"{a}@{s}": len(v) for (a, s), v in earlier.items()}}
    lists["n_finals"] = sum(v["n"] for v in lists["anchored_finals"].values())
    lists["a2_primary"] = [r.key_id for r in K.a2_primary_refs()]
    _write(out / "lists.json", lists)
    if not args.no_audit:
        from apexzero.fields import network as N
        from apexzero.fields import stages as S

        audit = N.loop_audit()
        audit["h3"] = S.h3_loop_audit()
        _write(out / "loop_audit.json", audit)
        print(f"loop audit: max stored horizon {audit['max']} (limit {audit['limit']}), "
              f"over {audit['n_over']}, open {audit['n_open']}, ok {audit['ok']}; H3 max "
              f"{audit['h3']['max']} (limit {audit['h3']['limit']}), ok {audit['h3']['ok']}")
        if not (audit["ok"] and audit["h3"]["ok"]):
            raise SystemExit("the loop is too short for a stored horizon: raise network.STEPS "
                             "or stages.H3_STEPS")
    if args.no_gate0:
        return
    refs = _limit(_refs(args.kinds), args.limit)
    with stage_log(args, "keys", len(refs), note=f"gate 0 on {len(refs)} keys") as log:
        res = G.gate0(refs, workers=args.workers, progress=lambda n: log.step(n))
    _write(out / "gate0.json", res)
    s = res["summary"]
    print(f"Gate 0: {s['n_ok']}/{s['n']} ok, max |r_free| {s['max_r_free']:.2e}, "
          f"pass {s['pass']}")


def _count(refs, key) -> dict:
    out: dict = {}
    for r in refs:
        out[key(r)] = out.get(key(r), 0) + 1
    return dict(sorted(out.items()))


def _repolish_chunk(refs) -> list[dict]:
    """One re-polish per key: a row with 'ok' False (IPOPT's status or an RK stage outside its
    kappa block) is a numerical exclusion; an exception is a crash ('crashed': the stage
    does not pass)."""
    from apexzero.fields import keys as K

    bank, _ = K.design()
    out = []
    for r in refs:
        try:
            out.append(K.summary(K.repolish(K.rebuild(K.load_key(r), bank))))
        except Exception as e:  # noqa: BLE001  (recorded as a crash; the stage fails)
            out.append({"key_id": r.key_id, "kind": r.kind, "ok": False, "crashed": True,
                        "error": f"{type(e).__name__}: {e}"})
    return out


def stage_repolish(args) -> None:
    """One re-polish per key; resumes from outputs/fields/repolish/repolish.jsonl."""
    from apexzero.fields import gates as G

    out = _out(args, "repolish")
    path = out / "repolish.jsonl"
    done = set()
    if path.exists() and not args.force:
        done = {json.loads(ln)["key_id"] for ln in path.read_text().splitlines() if ln.strip()}
    elif path.exists():
        path.unlink()
    refs = [r for r in _limit(_refs(args.kinds), args.limit) if r.key_id not in done]
    if done:
        _prov().partial.append(f"resumed {len(done)} rows (--force redoes them)")
    with stage_log(args, "repolish", len(refs)) as log, open(path, "a") as fh:
        def progress(n):
            log.step(n)

        for i in range(0, len(refs), 50 * max(args.workers, 1)):
            rows = G.map_keys(_repolish_chunk, refs[i: i + 50 * max(args.workers, 1)],
                              args.workers, chunk=50, progress=progress)
            fh.writelines(json.dumps(_jsonl(row)) + "\n" for row in rows)
            fh.flush()
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    _stamp_file(path)
    summ = {"n": len(rows), "n_ok": sum(bool(r.get("ok")) for r in rows),
            "n_flagged": sum(bool(r.get("flagged")) for r in rows),
            "max_dx": max((r.get("max_dx", 0.0) for r in rows), default=0.0),
            "max_dJ_rel": max((abs(r.get("dJ_rel", 0.0)) for r in rows), default=0.0),
            "max_iterations": max((r.get("iterations", 0) for r in rows), default=0),
            "not_ok": [r["key_id"] for r in rows if not r.get("ok") and not r.get("crashed")],
            "crashed": [r["key_id"] for r in rows if r.get("crashed")]}
    # design Section 5: a failed re-polish is a listed 'numerical' exclusion, not a failure
    # of the stage; the stage passes when every key was processed without a crash
    summ["pass"] = bool(rows and not summ["crashed"])
    _write(out / "summary.json", summ)
    print(f"re-polish: {summ['n_ok']}/{summ['n']} ok, {summ['n_flagged']} flagged, "
          f"pass {summ['pass']}")


def _jsonl(row: dict) -> dict:
    from apexzero.interp import io

    return json.loads(io.json_bytes(row))


def _gate1_refs(args):
    from apexzero.fields import gates as G

    sample = G.gate1_sample()
    if args.n is None or args.n >= len(sample):
        return sample
    # round-robin over the strata so a small run still spans them
    by: dict[str, list] = {}
    for r in sample:
        by.setdefault(G.family(r), []).append(r)
    out, k = [], 0
    while len(out) < args.n:
        for s in by.values():
            if k < len(s) and len(out) < args.n:
                out.append(s[k])
        k += 1
    return out


def stage_gate1(args) -> None:
    from apexzero.fields import gates as G

    refs = _gate1_refs(args)
    with stage_log(args, "gate1", len(refs)) as log:
        res = G.gate1(refs, workers=args.workers, progress=lambda n: log.step(n))
    _write(_out(args, "gates") / "gate1.json", res)
    d = res["decision"]
    print(f"Gate 1 on {len(refs)} keys: median {d['clean']['median']:.2e}, p95 "
          f"{d['clean']['p95']:.2e}, pass {d['pass']}; plants fail "
          f"{ {p: v['fails'] for p, v in d['plants'].items()} }")


def _gate2_cells(args) -> dict[str, tuple[int, ...]]:
    """Gate 2's cells per policy: the options given, else ``gates.GATE2_CELLS``."""
    from apexzero.fields import gates as G

    return {k: tuple(getattr(args, k)) if getattr(args, k, None) is not None else tuple(v)
            for k, v in G.GATE2_CELLS.items()}


def stage_gate2(args) -> None:
    from apexzero.fields import gates as G

    cells = _gate2_cells(args)
    rows = G.gate2_policies(cells["random"], cells["keeper"], cells["linear"],
                            cells_hybrid=cells["hybrid"])
    with stage_log(args, "gate2", len(rows)) as log:
        res = []
        for label, net, rd, st, steps in rows:
            res.append(G.gate2_row(net, rd, st, steps, label=label))
            log.step(note=label)
    out = {"rows": res, "tol": G.GATE2_TOL, "pass": bool(res) and all(r["pass"] for r in res),
           "cells": {k: list(v) for k, v in cells.items()}}
    _write(_out(args, "gates") / "gate2.json", out)
    for r in res:
        print(f"Gate 2 {r['label']}: rel_err {r['rel_err']} share {r['share_counted']}")


def stage_gate3(args) -> None:
    from apexzero.fields import gates as G

    agents = args.agent or ["G2"]
    n_anch = G.GATE3_N_ANCHORED if args.n_anchored is None else args.n_anchored
    n_a2 = G.GATE3_N_A2 if args.n_a2 is None else args.n_a2
    with stage_log(args, "gate3", len(agents)) as log:
        res = {}
        for a in agents:
            res[a] = G.gate3_agent(a, n_anch, n_a2, all_keys=args.all, limit=args.limit)
            log.step(note=a)
    out = {"agents": res, "pass": all(v["pass"] for v in res.values())}
    _write(_out(args, "gates") / "gate3.json", out)
    for a, v in res.items():
        errs = [r.get("mean_err") for r in v["rows"]]
        print(f"Gate 3 {a}: pass {v['pass']}, mean errors {errs}")


def stage_gate4(args) -> None:
    from apexzero.fields import gates as G
    from apexzero.fields import keys as K

    refs = _limit(list(K.a2_primary_refs() if args.no_twins else K.a2_refs()), args.limit)
    with stage_log(args, "gate4", len(refs)) as log:
        res = G.gate4(refs, workers=args.workers, progress=lambda n: log.step(n))
    _write(_out(args, "gates") / "gate4.json", res)
    print(f"Gate 4 on {res['n']} A2 keys: {res['n_outside']} outside, max |G* - 1| "
          f"{res['max_abs_dev']:.2e}, pass {res['pass']}")


def stage_gate5(args) -> None:
    from apexzero.fields import gates as G

    refs = _gate1_refs(args)
    with stage_log(args, "gate5", len(refs)) as log:
        res = G.gate5(refs, progress=lambda n: log.step(n))
    _write(_out(args, "gates") / "gate5.json", res)
    for r in res["rows"]:
        print(f"Gate 5 {r['key_id']} ({r['kind']}, W {r['W']}): err {r['err']}, state "
              f"{r['state_tangent_err']:.1e}, pass {r['pass']}")


def stage_controls(args) -> None:
    from apexzero.fields import controls as C

    which = tuple(args.which)
    cells = C.QUICK_CELLS if args.quick else None
    with stage_log(args, "controls", len(which)) as log:
        res = {}
        for name in which:
            res.update({k: v for k, v in C.run_all((name,), cells).items() if k != "pass"})
            log.step(note=name)
    res["pass"] = {k: bool(v["pass"]) for k, v in res.items()}
    _write(_out(args, "controls") / "controls.json", C.jsonable(res))
    print(f"controls: {res['pass']}")


def _opt_chunk(items) -> list[dict]:
    """``stages.opt_fields`` on [(arrays dir, key), ...] (one map_keys chunk)."""
    from apexzero.fields import keys as K
    from apexzero.fields import stages as S

    bank, _ = K.design()
    return [row for out_dir, ref in items for row in S.opt_fields([ref], Path(out_dir), bank)]


def stage_opt_fields(args) -> None:
    """The optimum's fields per key (pre-tag; ``stages.opt_fields_key``)."""
    from apexzero.fields import gates as G

    out = _out(args, "opt-fields")
    refs = _limit(_refs(args.kinds), args.limit)
    with stage_log(args, "opt-fields", len(refs)) as log:
        items = [(str(out / "arrays"), r) for r in refs]
        rows = G.map_keys(_opt_chunk, items, args.workers, chunk=10,
                          progress=lambda n: log.step(n))
    _write_rows(out / "rows.jsonl", rows)
    bad = [r["key_id"] for r in rows if r.get("failed")]
    # a failed row (no re-polish) is a listed 'numerical' exclusion; any other exception
    # stops the stage (stages.opt_fields_key), so a written summary has no crash
    summ = {"n": len(rows), "n_failed": len(bad), "failed": bad, "crashed": [],
            "n_outside_gate4": sum(not r.get("in_band", False) for r in rows
                                   if r["kind"] != "anchored" and not r.get("failed")),
            "n_inertia_failed": sum(bool(r.get("inertia_failed")) for r in rows),
            "n_adjoint_failed": sum(r.get("adjoint_ok") is False for r in rows),
            "n_off_plateau": sum(not r.get("on_plateau", False) for r in rows
                                 if not r.get("failed") and r["valid"]["c_min"])}
    from apexzero.fields import prereg

    summ["rows_sha256"] = prereg.sha256_file(out / "rows.jsonl")
    summ["arrays_hash"], summ["n_arrays"] = prereg.dir_hash(out / "arrays")
    _write(out / "summary.json", summ)
    print(f"opt-fields: {summ}")


def _qp_chunk(items) -> list[dict]:
    """``stages.reader_fields`` on [(arrays dir, key), ...] (one map_keys chunk; a spawned
    worker switches x64 on itself)."""
    _x64()
    from apexzero.fields import stages as S

    return [row for out_dir, ref in items for row in S.reader_fields([ref], Path(out_dir), "Qp")]


def stage_qp_fields(args) -> None:
    """Qp's fields on the primary A2 keys (``stages.reader_fields``; Qp is untrained, so this
    runs before the tag): H1-typical's lawful-reader reference."""
    from apexzero.fields import gates as G
    from apexzero.fields import keys as K
    from apexzero.fields import prereg

    out = _out(args, "qp-fields")
    refs = _limit(list(K.a2_primary_refs()), args.limit)
    with stage_log(args, "qp-fields", len(refs)) as log:
        rows = G.map_keys(_qp_chunk, [(str(out / "arrays"), r) for r in refs], args.workers,
                          chunk=10, progress=lambda n: log.step(n))
    _write_rows(out / "rows.jsonl", rows)
    fails = [r["key_id"] for r in rows if r.get("pipeline")]
    h, n = prereg.dir_hash(out / "arrays" / "reader-Qp")
    summ = {"n": len(rows), "pipeline_failures": fails,
            "valid": {o: sum(bool(r.get("valid", {}).get(o)) for r in rows)
                      for o in ("c_min", "v_apex", "s_b")},
            "rows_sha256": prereg.sha256_file(out / "rows.jsonl"), "arrays_hash": h,
            "n_arrays": n}
    _write(out / "summary.json", summ)
    print(f"qp-fields: {len(rows)} keys, {len(fails)} pipeline failures, valid {summ['valid']}")




def calibration_errors(g3: dict, agent_ids) -> tuple[dict, list[str]]:
    """Per agent, the largest teacher-forced mean error of Gate 3's clean rows that passed,
    over the planted sample and every key (``gate3 --all``), and why the calibration may not
    use them: an agent missing from gates/gate3.json, one without the every-key check or with
    no checked key, a non-finite error."""
    import numpy as np

    probs, per = [], {}
    for a in agent_ids:
        v = g3.get("agents", {}).get(a)
        if v is None:
            probs.append(f"{a} is missing from gates/gate3.json")
            continue
        ek = v.get("every_key")
        if ek is None:
            probs.append(f"{a}: gates/gate3.json lacks the every-key check (gate3 --all)")
            continue
        errs = [r["mean_err"] for r in v.get("rows", []) if r.get("ok") and "mean_err" in r]
        errs += [r["mean_err"] for r in ek.get("rows", []) if "skipped" not in r and r.get("ok")]
        if not ek.get("n_checked") or not errs:
            probs.append(f"{a}: no key checked by Gate 3")
            continue
        if not all(np.isfinite(float(e)) for e in errs):
            probs.append(f"{a}: a non-finite teacher-forced error")
            continue
        per[a] = float(max(errs))
    return per, probs


def control_rows(reps: dict) -> list[dict]:
    """The calibration's control rows from the float64 dry runs' reports (``reps``: 'linear',
    'hybrid' and 'random' with its 'keeper' rows): per row its source, branch values,
    teacher-forced mean error and branch verdict. Refuses (SystemExit) when a control's own
    branch check fails."""
    rows = reps["linear"]["rows"] + reps["hybrid"]["rows"] + reps["random"].get("keeper", [])
    out = [{"source": r["source"], "values": r["branch"]["values"],
            "mean_err": r["wiring"]["mean_err"], "branch_ok": r["branch"]["ok"]} for r in rows]
    if not all(r["branch_ok"] for r in out):
        raise SystemExit("calibrate refuses: a control's own branch check failed "
                         f"({[r['source'] for r in out if not r['branch_ok']]})")
    return out


def stage_calibrate(args) -> None:
    """Branch consistency's value tolerance (``network.branch_tolerance``; design Sections 14
    and 15), derived before the tag without running a trained network's free loop: the
    controls' float32/float64 outcome spread per unit of teacher-forced action error (the
    linear driver, the hybrid control with the trained architecture in the loop, the lane
    keeper), times the trained agents' largest teacher-forced error over every key that passed
    Gate 3 (``gate3 --all`` on every agent, read from gates/gate3.json), times BRANCH_FACTOR.
    Writes calibration/branch_tol.json (with the sha256 of the gates/gate3.json it read); the
    pre-registration record binds that file, and every post-tag stage reads the tolerance
    from the record. Refuses when an agent is missing or unchecked, an error is not finite,
    or a control's own branch check fails."""
    import jax
    import numpy as np

    from apexzero.fields import keys as K
    from apexzero.fields import network as N
    from apexzero.fields import prereg
    from apexzero.interp import agents as A
    from apexzero.interp import rollout

    if jax.config.jax_enable_x64:
        raise SystemExit("calibrate builds float32 references: x64 must be off in this process")
    g3_path = _out(args, "gates") / "gate3.json"
    if not g3_path.exists():
        raise SystemExit("run `fields.py gate3 --all --agent ...` on every agent first")
    g3 = json.loads(g3_path.read_text())
    per_agent, probs = calibration_errors(g3, A.AGENTS)
    if probs:
        raise SystemExit("calibrate refuses: " + "; ".join(probs))
    out = _out(args, "calibration")
    with stage_log(args, "calibrate", 4) as log:
        bank, cells = K.design()
        g1 = N._final_agent("G1")
        starts = [(cells[c], k) for c, k in CAL_RANDOM_STARTS]
        det = N.float32_reference(rollout.agent_policy(g1.cfg), N.random_actor_params(g1.cfg, 0),
                                  g1.obs_rms, bank, starts, 200)

        def save(name: str, det: dict, meta: dict) -> Path:
            return _stamped_write(out / name, lambda q: N.save_det(q, det, meta))

        refs = {"random": save("random_f32.npz", det,
                               {"policy": "random", "seed": 0, "loop_steps": 200})}
        starts = [(cells[c], 0) for c in CAL_LINEAR_CELLS]
        Ws, bs = zip(*(N.linear_driver(bank.stimuli[c.stim_id].direction) for c, _ in starts),
                     strict=True)
        det = N.float32_reference(N.linear_policy, (np.stack(Ws), np.stack(bs)), None, bank,
                                  starts, CAL_LINEAR_STEPS, aux=list(range(len(starts))))
        det["lin_W"], det["lin_b"] = np.stack(Ws), np.stack(bs)
        refs["linear"] = save("linear_f32.npz", det,
                              {"policy": "linear_driver", "loop_steps": CAL_LINEAR_STEPS})
        starts = [(cells[c], 0) for c in N.HYBRID_CELLS]
        prm = N.hybrid_reference_params([bank.stimuli[c.stim_id].direction for c, _ in starts],
                                        g1.cfg, g1.obs_rms)
        det = N.float32_reference(N.hybrid_policy(g1.cfg), prm, g1.obs_rms, bank, starts,
                                  N.HYBRID_STEPS, aux=list(range(len(starts))))
        det["hyb_Wn"] = prm[0]
        refs["hybrid"] = save("hybrid_f32.npz", det,
                              {"policy": "hybrid", "eps": N.HYBRID_GAIN,
                               "loop_steps": N.HYBRID_STEPS})
        starts = [(cells[c], k) for c, k in CAL_KEEPER_STARTS]
        det = N.float32_reference(rollout._lane_keeper, N.KEEPER_PARAMS, None, bank, starts, 300)
        refs["keeper"] = save("keeper_f32.npz", det,
                              {"policy": "lane_keeper", "loop_steps": 300})
        log.step(note="float32 references")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]))
        cmds = {"linear": ["--f32", str(refs["linear"])],
                "hybrid": ["--f32", str(refs["hybrid"])],
                "random": ["--f32", str(refs["random"]), "--keeper", str(refs["keeper"])]}
        procs = {k: subprocess.Popen([sys.executable, "-m", "apexzero.fields.network", "dry-run",
                                      k, *a, "--out", str(out / f"{k}_f64")], env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                 for k, a in cmds.items()}
        reps = {}
        for k, p in procs.items():
            txt, _ = p.communicate(timeout=3600)
            if p.returncode != 0:
                raise SystemExit(f"dry-run {k} failed:\n{txt[-4000:]}")
            reps[k] = json.loads((out / f"{k}_f64.json").read_text())
            for ext in (".json", ".npz"):
                _stamp_file(out / f"{k}_f64{ext}")
        log.step(note="float64 dry runs")
        crows = control_rows(reps)
        trained = max(per_agent.values())
        tol = N.branch_tolerance(crows, trained)
        if not all(np.isfinite(v) for v in tol["tol"].values()):
            raise SystemExit(f"calibrate refuses: a non-finite tolerance {tol['tol']}")
        log.step(note="tolerance")
        res = {"tolerance": tol["tol"], "factor": tol["factor"], "trained_mean_err": trained,
               "trained_mean_err_by_agent": per_agent, "agents_missing_from_gate3": [],
               "argmax_agent": max(per_agent, key=per_agent.get),
               "gate3_keys_checked": {a: v["every_key"]["n_checked"]
                                      for a, v in g3["agents"].items()},
               "gate3_failures_left_out": {a: v["every_key"]["failures"]
                                           for a, v in g3["agents"].items()
                                           if v["every_key"]["failures"]},
               "gate3_sha256": prereg.sha256_file(g3_path),
               "control_rows": crows, "n_control_rows": len(crows),
               "hybrid_mlp_share": [r.get("mlp_share") for r in reps["hybrid"]["rows"]],
               "slopes": tol["slopes"], "provisional_BRANCH_TOL": dict(N.BRANCH_TOL),
               "controls_branch_ok": True}
        _write(out / "branch_tol.json", res)
        log.step(note="written")
    print(f"calibrate: tolerance {tol['tol']} (trained max teacher-forced error {trained:.2e}; "
          f"{len(crows)} control rows)")


def stage_phase_ceiling(args) -> None:
    from apexzero.fields import stages as S

    src = Path(args.out) / "opt-fields"
    rows = S.read_rows(src / "rows.jsonl")
    res = S.phase_ceiling_rows(rows, src / "arrays")
    _write(_out(args, "phase-ceiling") / "phase_ceiling.json", res)
    print(f"phase ceiling on {len(res)} cells")


def dirty_paths(root: Path = ROOT) -> list[str]:
    """Paths under the code paths and the design document that differ from HEAD or are
    untracked, other than the record itself (the prereg stage needs a clean tree)."""
    r = _git("status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS, DESIGN_DOC,
             root=root)
    if r.returncode != 0:
        return [f"<git error: {r.stderr.strip()[:200]}>"]
    out = []
    for ln in r.stdout.splitlines():
        path = ln[3:].strip().strip('"')
        if path and path != PREREG_REL:
            out.append(path)
    return out


def network_pins(names, root: Path = ROOT) -> tuple[dict, list[str]]:
    """{name: {path, sha256, timesteps}} for every network a post-tag stage loads, and the
    problems: a registry agent's checkpoint that is not step 1's final snapshot, or an earlier
    snapshot ('<agent>@<timesteps>') that is not step 1's snapshot of those timesteps (their
    manifest entries' path and sha256; the loaded timesteps must match)."""
    from apexzero.fields import stages as S
    from apexzero.interp import agents, io

    pins, probs = {}, []
    man = json.loads(io.MANIFEST_PATH.read_text()) if io.MANIFEST_PATH.exists() else {}
    finals = {s["agent"]: s for s in man.get("snapshots", []) if s.get("is_final")}
    for name in dict.fromkeys(names):
        snap = S.split_snapshot(name)
        try:
            pins[name] = S.network_pin(name, root)
        except FileNotFoundError as e:
            probs.append(str(e))
            continue
        ent = finals.get(name) if name in agents.AGENTS else (
            S.step1_snapshot(*snap) if snap is not None else None)
        if name not in agents.AGENTS and snap is None:
            continue
        if (ent is None or ent["path"] != pins[name]["path"]
                or ent["sha256"] != pins[name]["sha256"]
                or (snap is not None and int(pins[name]["timesteps"]) != snap[1])):
            probs.append(f"{name}: its checkpoint is not step 1's ({pins[name]['path']})")
    return pins, probs


def earlier_networks() -> list[str]:
    """The earlier snapshots with anchored keys, as network names '<agent>@<timesteps>'
    (fields-earlier; their checkpoints are pinned like every other network's)."""
    from apexzero.fields import keys as K

    return [f"{a}@{int(s)}" for a, s in sorted(K.anchored_earlier())]


def pinned_networks() -> list[str]:
    """Every network a post-tag stage loads: H2', H4, H4-transfer and the dose networks,
    every registry agent with a final snapshot (fields-anchored, H3, controls) and every
    earlier snapshot with anchored keys (fields-earlier)."""
    from apexzero.fields import analysis as A
    from apexzero.interp import agents

    reg = [a for a in agents.AGENTS if agents.final_snapshot(agents.AGENTS[a]) is not None]
    return list(dict.fromkeys(list(A.A2_NETWORKS) + list(A.DOSE_NETWORKS) + reg
                              + earlier_networks()))


def regime_record(sets: dict) -> tuple[dict, list[str]]:
    """The per-regime A2 cell sets of every agent of step 1's support (over the primary A2
    cells of the frozen 'a2' set minus 'a2_twins'), H4's typing per variant, and support.json's
    sha256, which must be step 1's manifest's."""
    from apexzero.fields import analysis as A
    from apexzero.fields import keys as K
    from apexzero.fields import prereg
    from apexzero.interp import io, support

    probs = []
    sup = io.SUPPORT_DIR / "support.json"
    sha = prereg.sha256_file(sup)
    man = json.loads(io.MANIFEST_PATH.read_text()) if io.MANIFEST_PATH.exists() else {}
    if (man.get("support") or {}).get("sha256") != sha:
        probs.append("support.json is not the one step 1's manifest records")
    bank, cells = K.design()
    twins = {tuple(t) for t in sets.get("a2_twins", [])}
    by = {(r.kind, r.key_id): r.cell_id for kind in K.A2_KINDS for r in K.key_index(kind)}
    primary = sorted({by[tuple(t)] for t in sets["a2"] if tuple(t) not in twins})
    regimes = support.load_support(io.SUPPORT_DIR).regimes
    cells_by = {a: {k: list(v) for k, v in A.a2_cells_by_regime(a, bank, cells, regimes,
                                                                 primary).items()}
                for a in sorted(regimes)}
    typing = {"v1": A.H4_VARIANTS["v1"]["regime_of"], "v3": A.H4_VARIANTS["v3"]["regime_of"],
              "transfer": A.H4_TRANSFER["regime_of"]}
    return {"support_sha256": sha, "primary_cells": primary, "a2_regime_cells": cells_by,
            "h4_typing": typing,
            "variant_regime_cells": {v: cells_by[a] for v, a in typing.items()}}, probs


def environment(root: Path = ROOT) -> dict:
    """The Python, platform and package versions (and uv.lock's sha256, for information) a
    stage runs with: in the record and in every stamp. The freeze check and the confirmatory
    input check compare the Python and package versions ('code_changed': a logged deviation
    may waive them), never the machine (another computing machine is expected)."""
    import importlib.metadata as md
    import platform

    from apexzero.fields import prereg

    pk = {}
    for name in ("jax", "jaxlib", "numpy", "scipy", "casadi", "flax", "optax"):
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pk[name] = None
    lock = Path(root) / "uv.lock"
    return {"python": sys.version.split()[0], "machine": platform.machine(),
            "system": platform.system(), "packages": pk,
            "uv_lock_sha256": prereg.sha256_file(lock) if lock.exists() else None}


def record_extra(head: str, pins: dict, regimes: dict, rollouts: dict, outputs: dict,
                 cal: dict, gate2_ids, dose_ids, env: dict | None = None) -> dict:
    """The items the prereg stage adds to the record (``prereg.build_record``'s ``extra``):
    the base commit, the network pins, the regime cell sets, the stored rollouts, the
    environment, the gate2-trained and dose key lists ((kind, key id) pairs), the bound
    outputs, the calibration and step 1's tag the record was built on."""
    return {"prereg_base_commit": head, "networks": pins, "regimes": regimes,
            "rollouts": rollouts, "environment": environment() if env is None else env,
            "gate2_trained": [list(t) for t in gate2_ids], "dose_sample": [list(t) for t in dose_ids],
            "outputs": outputs, "calibration": cal,
            "step1_tag": {"tag": STEP1_TAG, "object": STEP1_TAG_OBJECT}}


def head_problems(root: Path = ROOT) -> list[str]:
    """Why the bytes on disk are not HEAD's (the record must hash only what HEAD holds): the
    code hash against ``prereg.tree_code_hash`` of HEAD (an assume-unchanged or skip-worktree
    edit, an ignored .py file), the design document against its blob at HEAD."""
    from apexzero.fields import prereg

    out = []
    try:
        if prereg.code_hash(Path(root)) != prereg.tree_code_hash(Path(root), "HEAD"):
            out.append("the code on disk is not HEAD's (an assume-unchanged or skip-worktree "
                       "edit, or a .py file git does not track)")
    except RuntimeError as e:
        out.append(f"git failed hashing HEAD's code: {e}")
    b = _git_bytes("cat-file", "blob", f"HEAD:{DESIGN_DOC}", root=root)
    doc = Path(root) / DESIGN_DOC
    if b.returncode != 0 or not doc.exists() or \
            prereg.sha256_bytes(b.stdout) != prereg.sha256_file(doc):
        out.append(f"{DESIGN_DOC} on disk is not HEAD's")
    return out


def stage_prereg(args) -> None:
    """The pre-registration record (``fields.prereg``; design Section 15): the last step
    before the tag, from a clean tree (nothing changed under the code paths or the design
    document but the record, in git or byte by byte against HEAD: ``head_problems``) with
    step 1's frozen code its own tag's (``step1_problems``), every pre-tag output bound
    (stamped with this code, passing, covering exactly the listed keys) and the calibrated
    tolerance. Records HEAD (``record_extra``)."""
    from apexzero.fields import keys as K
    from apexzero.fields import prereg
    from apexzero.fields import stages as S
    from apexzero.interp import agents, io, stimuli

    dirty = dirty_paths()
    if dirty:
        raise SystemExit(f"prereg refuses a dirty tree: {dirty[:8]} (commit the code and the "
                         "final design document, run prereg, commit prereg.json alone, then "
                         "tag)")
    if not (ROOT / DESIGN_DOC).exists():
        raise SystemExit(f"{DESIGN_DOC} is missing")
    s1 = step1_problems()
    if s1:
        raise SystemExit("prereg refuses: step 1's frozen code is not its tag's: "
                         + "; ".join(s1))
    hp = head_problems()
    if hp:
        raise SystemExit("prereg refuses: " + "; ".join(hp))
    root = Path(args.out)
    rows_path = root / "repolish" / "repolish.jsonl"
    if not rows_path.exists():
        raise SystemExit(f"no re-polish rows at {rows_path}: run the repolish stage first")
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines() if ln.strip()]
    sets = prereg.key_sets()
    bank, cells = stimuli.load_design(io.STIMULI_DIR)
    ft = {a: K.final_timesteps(a) for a in agents.AGENTS
          if agents.final_snapshot(agents.AGENTS[a]) is not None}
    code = prereg.code_hash(ROOT)
    qp_ids = {r.key_id for r in K.a2_primary_refs()}
    outputs, probs = prereg.bind_outputs(root, code, sets, qp_ids)
    g3 = root / "gates" / "gate3.json"
    cal, p2 = prereg.calibration_binding(root, prereg.sha256_file(g3) if g3.exists() else None)
    pins, p3 = network_pins(pinned_networks())
    regimes, p4 = regime_record(sets)
    probs += p2 + p3 + p4
    rp = prereg.repolish_rows(rows)
    missing = sorted({k for v in sets.values() for _, k in v} - set(rp))
    if missing:
        probs.append(f"{len(missing)} listed keys have no re-polish row")
    if probs:
        raise SystemExit("prereg refuses: " + "; ".join(probs))
    rollouts = rollout_record(ft, earlier=[(a, int(t)) for a, t in K.anchored_earlier()])
    missing_r = sorted(set(ft) - {k.split("@")[0] for k in rollouts})
    if missing_r:
        raise SystemExit(f"prereg refuses: no stored final rollouts for {missing_r}")
    extra = record_extra(_head(), pins, regimes, rollouts, outputs, cal,
                         [(r.kind, r.key_id) for r in S.gate2_trained_sample()],
                         [(r.kind, r.key_id) for r in S.dose_sample()])
    rec = prereg.build_record(ROOT, sets, io.KEYS_DIR, rows, stimuli.design_hash(bank, cells),
                              ft, prereg.tolerances(cal["tolerance"]), extra)
    path = prereg.save(rec)
    print(f"wrote {path}: {rec['key_counts']}; code {code[:12]}, HEAD {extra['prereg_base_commit'][:12]}")
    print(f"next: git add {PREREG_REL} && git commit -m 'Step 2: pre-registration record'; "
          f"git tag -a {TAG} -m '...'; git push {REMOTE} {TAG}")


# ------------------------------------------------------------------------- post-tag stages


def import_problems(root: Path = ROOT, modules: dict | None = None) -> list[str]:
    """Why this process's code is not the checked-out code the freeze check hashes: the
    imported apexzero must be <root>/src/apexzero (scripts/_bootstrap.py puts src/ first),
    never an installed copy elsewhere; and every module loaded from inside the repository
    (outside the environment's own prefix, e.g. .venv) must be frozen code: under
    src/apexzero, or scripts/fields.py or scripts/_bootstrap.py (a scripts/numpy.py or a
    src/sitecustomize.py would shadow a package)."""
    import apexzero

    root = Path(root).resolve()
    want = (root / "src" / "apexzero").resolve()
    got = Path(apexzero.__file__).resolve().parent
    out = [] if got == want else [f"apexzero is imported from {got}, not {want}"]
    prefixes = {Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    allowed = {(root / "scripts" / n).resolve() for n in ("fields.py", "_bootstrap.py")}
    for name, mod in sorted((modules if modules is not None else dict(sys.modules)).items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        fp = Path(f).resolve()
        if root not in fp.parents or any(p == fp or p in fp.parents for p in prefixes):
            continue
        if want in fp.parents or fp in allowed:
            continue
        out.append(f"{name} imported from {fp}, outside the frozen code")
    return out


def _post_tag(args, what: str, x64: bool = True) -> dict:
    """The freeze check (the record against the disk and the canonical outputs/fields; a run
    into another root is stamped partial and never read by the confirmatory analysis), then
    float64 and trained
    networks' fields allowed (``x64`` False: the float32 harness stage keeps x64 off and
    computes no field). Returns the tagged record; refuses when it has no calibrated
    tolerance or no checkpoint pins."""
    from apexzero.fields import prereg

    out_root = Path(args.out)
    if getattr(args, "limit", None) is not None and out_root.resolve() == OUT.resolve():
        raise SystemExit("--limit writes a partial run: give --out outside outputs/fields")
    bad = import_problems()
    if bad:
        raise SystemExit(f"refusing to run {what}: {'; '.join(bad)}")
    probs = freeze_problems(verify=lambda r, rt: _verify_on_disk(r, rt, OUT))
    dev = require_tag(what, getattr(args, "post_tag_deviation", None), problems=probs)
    rec = tagged_record()
    if rec is None:
        raise SystemExit("no pre-registration record at the tag")
    if prereg.record_tolerance(rec) is None:
        raise SystemExit("refusing: the record holds no calibrated branch tolerance")
    if not rec.get("networks"):
        raise SystemExit("refusing: the record pins no network checkpoint")
    p = _prov()
    p.deviation, p.tag, p.check_imports = dev, tag_identity(), True
    if x64:
        _x64()
        from apexzero.fields import network as N

        N.unlock_trained()
    return rec


def _frozen_refs(names: tuple[str, ...], rec: dict | None = None, ids=None):
    """The tagged record's key lists (never a fresh glob), as KeyRefs; or the record's own
    (kind, key_id) list ``ids`` (gate2-trained, dose)."""
    from apexzero.fields import keys as K
    from apexzero.fields import prereg

    rec = tagged_record() if rec is None else rec
    if rec is None:
        raise SystemExit("no pre-registration record at the tag")
    want = ({tuple(t) for name in names for t in prereg.frozen_ids(rec, name)} if ids is None
            else {tuple(t) for t in ids})
    by = {(r.kind, r.key_id): r for kind in K.KINDS for r in K.key_index(kind)}
    missing = sorted(want - set(by))
    if missing:
        raise SystemExit(f"{len(missing)} tagged keys are not in the key store: {missing[:3]}")
    return [by[t] for t in sorted(want)]


def _pin(rec: dict, name: str) -> dict:
    pin = rec.get("networks", {}).get(name)
    if pin is None:
        raise SystemExit(f"{name} is not a recorded network (the record pins its checkpoint)")
    return pin


def _load(rec: dict, name: str):
    from apexzero.fields import stages as S

    return S.load_network(name, _pin(rec, name), ROOT)


def harness_path(root: Path, name: str) -> Path:
    return Path(root) / "harness-f32" / f"{name.replace('/', '_')}.npz"


def _det_of(agent_id: str | None, own: bool = False, rec: dict | None = None,
            harness_root: Path | None = None, require: bool = False):
    """det groups, loaded on first use: of each anchored key's own snapshot (``own`` False),
    or of ``agent_id``'s final snapshot for every key (``own`` True: A2 keys, the network's
    own runs). A run path of a registry agent's run (an H4 baseline) reads that agent's step 1
    rollouts when its recorded checkpoint is the agent's; any other network reads its
    float32 harness runs (``harness-f32``) under ``harness_root``, which ``require`` makes
    mandatory. None for a network with neither."""
    from apexzero.fields import keys as K
    from apexzero.fields import network as N
    from apexzero.fields import stages as S
    from apexzero.interp import io

    cache: dict = {}
    a_own = agent_id
    if own and agent_id is not None and agent_id not in K.agents.AGENTS:
        reg = S.registry_of_run(agent_id)
        pins = (rec or {}).get("networks", {})
        same = (reg is not None and agent_id in pins and reg in pins
                and pins[agent_id]["sha256"] == pins[reg]["sha256"])
        a_own = reg if same or (reg is not None and not pins) else None
        if a_own is None and harness_root is not None:
            hp = harness_path(harness_root, agent_id)
            if not hp.exists():
                if require:
                    raise SystemExit(f"{agent_id}: no float32 harness runs ({hp}): run the "
                                     "harness-f32 stage first")
                return lambda ref: None
            det = N.load_reference(hp)[0]
            return lambda ref: det

    def get(ref):
        a = a_own if own else (ref.agent or agent_id)
        if a is None or a not in K.agents.AGENTS:
            return None
        ts = K.final_timesteps(a) if own or not ref.snapshot else int(ref.snapshot)
        path = io.rollout_path(a, ts)
        if not path.exists():
            if require:
                raise SystemExit(f"{a}: its step 1 rollouts {path} are missing")
            return None
        if (a, ts) not in cache:
            cache[(a, ts)] = N.load_det(path)
        return cache[(a, ts)]

    return get


def without_rollouts(names) -> list[str]:
    """The networks with no step 1 rollouts of their own checkpoint (run paths of no registry
    agent): harness-f32 runs them."""
    from apexzero.fields import stages as S
    from apexzero.interp import agents

    return [n for n in names if n not in agents.AGENTS and S.registry_of_run(n) is None]


def _exploratory(args, default) -> list[str]:
    """The stage's fixed network list; ``--network`` only into a root other than
    outputs/fields (an exploratory run never lands where the analysis reads)."""
    if args.network:
        if Path(args.out).resolve() == OUT.resolve():
            raise SystemExit("--network writes an exploratory run: give --out outside "
                             "outputs/fields")
        return list(args.network)
    return list(default)


def harness_det(fn, params, obs_rms, bank, cells, starts, steps: int,
                chunk: int = HARNESS_CHUNK) -> dict:
    """Step 1's float32 harness (``network.float32_reference``, x64 off) from (cell, phase)
    starts in chunks of ``chunk`` runs, as one det group."""
    import numpy as np

    from apexzero.fields import network as N

    parts = [N.float32_reference(fn, params, obs_rms, bank,
                                 [(cells[c], k) for c, k in starts[i: i + chunk]], steps)
             for i in range(0, len(starts), chunk)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def stage_harness_f32(args) -> None:
    """The float32 harness runs (x64 off, step 1's frozen harness) of every H4 and
    H4-transfer network without step 1 rollouts, from each A2 key's (cell, phase) start: their
    Gate 3 and branch-consistency reference in fields-a2 (design Section 15)."""
    rec = _post_tag(args, "the float32 harness runs", x64=False)
    from apexzero.fields import analysis as A
    from apexzero.fields import keys as K
    from apexzero.fields import network as N
    from apexzero.fields import prereg
    from apexzero.fields import stages as S
    from apexzero.interp import agents, rollout

    refs = _limit(_frozen_refs(("a2",), rec), args.limit)
    starts = S.harness_starts(refs)
    names = without_rollouts(_exploratory(args, A.A2_NETWORKS))
    bank, cells = K.design()
    out = _out(args, "harness-f32")
    with stage_log(args, "harness-f32", len(names)) as log:
        for name in names:
            pin = _pin(rec, name)
            path = ROOT / pin["path"]
            if prereg.sha256_file(path) != pin["sha256"]:
                raise SystemExit(f"{name}: {pin['path']} is not the recorded checkpoint")
            ag = agents.load_agent(path, Path(name).name)
            if int(ag.timesteps) != int(pin["timesteps"]):
                raise SystemExit(f"{name}: timesteps {ag.timesteps}, recorded {pin['timesteps']}")
            det = harness_det(rollout.agent_policy(ag.cfg), ag.actor_params, ag.obs_rms, bank,
                              cells, starts, N.STEPS)
            meta = {"net": f"{Path(name).name}@{ag.timesteps}", "ckpt_sha256": pin["sha256"],
                    "loop_steps": N.STEPS, "n_runs": len(starts)}
            _stamped_write(harness_path(args.out, name),
                           lambda q, det=det, meta=meta: N.save_det(q, det, meta))
            log.step(note=name)
    _write(out / "summary.json", {"networks": names, "n_starts": len(starts),
                                  "loop_steps": N.STEPS})


def stage_gate2_trained(args) -> None:
    """Gate 2's rule on every H2', H4 and H4-transfer network at the record's gate2-trained
    keys (``stages.gate2_trained_network``; design Section 15): runs before fields-a2; a
    failing network is descriptive."""
    rec = _post_tag(args, "gate2-trained")
    from apexzero.fields import analysis as A
    from apexzero.fields import keys as K
    from apexzero.fields import stages as S

    if not rec.get("gate2_trained"):
        raise SystemExit("the record lists no gate2-trained keys")
    refs = _frozen_refs((), rec, rec["gate2_trained"])
    names = _exploratory(args, A.A2_NETWORKS)
    bank, cells = K.design()
    res = {}
    with stage_log(args, "gate2-trained", len(names) * len(refs)) as log:
        for name in names:
            res[name] = S.gate2_trained_network(_load(rec, name), refs, bank, cells)
            log.step(len(refs), note=name)
    _write(_out(args, "gate2-trained") / "gate2_trained.json",
           {"keys": [r.key_id for r in refs], "networks": res,
            "failed": [n for n, v in res.items() if not v["pass"]],
            "pass": all(v["pass"] for v in res.values())})


def stage_fields_a2(args) -> None:
    rec = _post_tag(args, "the networks' fields on the A2 keys")
    from apexzero.fields import analysis as A
    from apexzero.fields import prereg
    from apexzero.fields import stages as S

    tol = prereg.record_tolerance(rec)
    refs = _limit(_frozen_refs(("a2",), rec), args.limit)
    names = _exploratory(args, A.A2_NETWORKS)
    out = _out(args, "fields-a2")
    with stage_log(args, "fields-a2", len(refs) * len(names)) as log:
        for name in dict.fromkeys(names):
            net = _load(rec, name)
            det_of = _det_of(name, own=True, rec=rec, harness_root=Path(args.out), require=True)
            rows = S.network_fields(net, refs, out / "arrays", det_of=det_of, branch_tol=tol,
                                    progress=lambda n: log.step(n))
            _write_rows(out / f"{name.replace('/', '_')}.jsonl", rows)


def stage_fields_anchored(args) -> None:
    rec = _post_tag(args, "the networks' fields on the anchored finals")
    from apexzero.fields import prereg
    from apexzero.fields import stages as S

    tol = prereg.record_tolerance(rec)
    agents_ = args.agent or None
    if agents_ and Path(args.out).resolve() == OUT.resolve():
        raise SystemExit("--agent writes a partial run: give --out outside outputs/fields")
    out = _out(args, "fields-anchored")
    sets = [k for k in rec["key_sets"] if k.startswith("finals:")
            and (agents_ is None or k.split(":", 1)[1] in agents_)]
    for name in sets:
        agent = name.split(":", 1)[1]
        refs = _limit(_frozen_refs((name,), rec), args.limit)
        net = _load(rec, agent)
        with stage_log(args, f"fields-anchored-{agent}", len(refs)) as log:
            rows = S.network_fields(net, refs, out / "arrays", det_of=_det_of(agent),
                                    blind=False, branch_tol=tol, progress=lambda n: log.step(n))
        _write_rows(out / f"{agent}.jsonl", rows)


def stage_fields_earlier(args) -> None:
    """Each earlier snapshot's fields on its own anchored keys (design Section 7, descriptive:
    the 'earlier snapshots'): the record's 'earlier:<agent>@<timesteps>' key sets, each run by
    that snapshot's recorded checkpoint ('<agent>@<timesteps>', pinned from step 1's
    manifest) from its own det rows (option A). Gate 3 runs on every key with a det row;
    branch consistency uses the record's calibrated tolerance, which was derived from the
    finals' Gate 3 errors (the earlier snapshots' rows are descriptive)."""
    rec = _post_tag(args, "the earlier snapshots' fields")
    from apexzero.fields import prereg
    from apexzero.fields import stages as S

    tol = prereg.record_tolerance(rec)
    agents_ = args.agent or None
    if agents_ and Path(args.out).resolve() == OUT.resolve():
        raise SystemExit("--agent writes a partial run: give --out outside outputs/fields")
    out = _out(args, "fields-earlier")
    sets = [k for k in rec["key_sets"] if k.startswith("earlier:")
            and (agents_ is None or k.split(":", 1)[1].split("@")[0] in agents_)]
    for sname in sets:
        name = sname.split(":", 1)[1]
        agent = name.split("@")[0]
        refs = _limit(_frozen_refs((sname,), rec), args.limit)
        net = _load(rec, name)
        with stage_log(args, f"fields-earlier-{name}", len(refs)) as log:
            rows = S.network_fields(net, refs, out / "arrays", det_of=_det_of(agent),
                                    blind=False, branch_tol=tol, progress=lambda n: log.step(n))
        _write_rows(out / f"{name}.jsonl", rows)


def stage_fields_anchored_h4(args) -> None:
    """The anchored-key version of H4 (design Section 7, descriptive): every H4 network on the
    pooled anchored finals of every agent (the record's 'finals:*' key sets), each key under
    option A from its owner's det row. The states are off-policy for every network but the
    key's owner: Gate 3 and branch consistency run only where the network's recorded
    checkpoint is the owner's (an H4 baseline that is a registry agent's run), and every
    other row is marked 'off_policy' (no check, never a Gate 3 exclusion). Rows only, no
    arrays (H4 reads G). About 18 x 8,136 runs and VJPs: hours."""
    rec = _post_tag(args, "the anchored-key H4")
    from apexzero.fields import analysis as A
    from apexzero.fields import prereg
    from apexzero.fields import stages as S

    tol = prereg.record_tolerance(rec)
    finals = [k for k in rec["key_sets"] if k.startswith("finals:")]
    refs = _limit(_frozen_refs(tuple(finals), rec), args.limit)
    pins = rec["networks"]
    out = _out(args, "fields-anchored-h4")
    names = _exploratory(args, A.h4_networks())
    with stage_log(args, "fields-anchored-h4", len(refs) * len(names)) as log:
        for name in names:
            net = _load(rec, name)
            sha = _pin(rec, name)["sha256"]

            def own(ref, sha=sha):
                return ref.agent in pins and pins[ref.agent]["sha256"] == sha

            rows = S.network_fields(net, refs, out / "arrays", det_of=_det_of(None),
                                    blind=False, paths=False, branch_tol=tol, save=False,
                                    checks_of=own, progress=lambda n: log.step(n))
            _write_rows(out / f"{name.replace('/', '_')}.jsonl", rows)


def stage_dose(args) -> None:
    rec = _post_tag(args, "the dose-response subsample")
    from apexzero.fields import analysis as A
    from apexzero.fields import gates as G
    from apexzero.fields import keys as K
    from apexzero.fields import kkt as KK
    from apexzero.fields import network as N
    from apexzero.fields import stages as S

    if not rec.get("dose_sample"):
        raise SystemExit("the record lists no dose keys")
    out = _out(args, "dose")
    refs = _limit(_frozen_refs((), rec, rec["dose_sample"]), args.limit)
    bank, cells = K.design()
    with stage_log(args, "dose", len(refs)) as log:
        opt = []
        for ref in refs:
            try:
                prob, stim, _ = G.problem(ref, bank)
                opt.append(KK.dose_optimum(prob, stim.landmarks, int(stim.direction),
                                           K.decision_step(prob.rec)))
            except Exception as e:  # noqa: BLE001  (recorded; the key is a numerical exclusion)
                opt.append({"key_id": ref.key_id, "failed": True, "error": str(e)})
            log.step()
    _write(out / "optimum.json", opt)
    res = {}
    for name in _exploratory(args, A.DOSE_NETWORKS):
        net = _load(rec, name)
        det = _det_of(name)
        rows = []
        for ref in refs:
            if ref.kind == "anchored" and ref.agent != name:
                continue
            key = K.load_key(ref)
            st = S.network_start(ref, key, bank, cells, det(ref) if ref.kind == "anchored"
                                 else None)
            rd = N.road(bank, st.stim_id)
            try:
                rows.append({"key_id": ref.key_id, **S.dose_network(net, rd, st)})
            except N.LoopExhausted as e:
                rows.append({"key_id": ref.key_id, "pipeline": S.PIPELINE_LOOP, "error": str(e)})
        res[name] = rows
    _write(out / "networks.json", res)


def stage_h3_keys(args) -> None:
    _post_tag(args, "H3's key solves")
    from apexzero.fields import stages as S

    cache = S.h3_cache(_out(args, "h3-keys") / "keys")
    base = S.h3_base_keys(cache)
    anch = S.h3_anchored_keys(cache)
    _write(_out(args, "h3-keys") / "keys.json",
           {"base": [r.key_id for r in base], "anchored": [r.key_id for r in anch],
            "status": {r.key_id: r.status for r in base + anch}})


def stage_h3_lambda(args) -> None:
    rec = _post_tag(args, "H3's lambda-grid rollouts")
    from apexzero.fields import analysis as A
    from apexzero.fields import keys as K
    from apexzero.fields import network as N
    from apexzero.fields import stages as S

    net = _load(rec, A.H3_AGENT)
    _, cells = K.design()
    res = []
    for cid in A.H3_CELLS:
        for k in range(cells[cid].n_phases):
            try:
                res.append(S.h3_lambda(net, cid, k))
            except N.LoopExhausted as e:        # that unit's pipeline failure; the stage goes on
                res.append({"cell": int(cid), "phase": int(k), "net": net.name,
                            "pipeline": S.PIPELINE_LOOP, "error": str(e)})
    _write(_out(args, "h3-lambda") / "h3.json", res)


def input_stamp_problems(paths, code_hash_now: str, deviation: dict | None,
                         tag_now: dict | None = None, env: dict | None = None,
                         check_tag: bool = True) -> list[str]:
    """Why the confirmatory analysis may not read these inputs: one without a stamp, whose
    stamp binds no content hash (a sidecar's file sha256, or an embedded stamp's
    'content_sha256'), changed after its stamp hashed it, written by other code, by a partial
    run, under a deviation this run does not declare, under another tag than ``tag_now``
    (object and commit; ``check_tag``), or (``env``: the record's environment) with other
    Python or package versions than the record's unless this run declares a deviation."""
    from apexzero.fields import prereg

    out = []
    for p in paths:
        st = prereg.read_stamp(p)
        if st is None:
            out.append(f"{p.name}: no provenance stamp")
            continue
        files = st.get("files") or {}
        if p.name in files:
            if prereg.sha256_file(p) != files[p.name]:
                out.append(f"{p.name}: changed after it was stamped")
        elif st.get("content_sha256"):
            if content_sha256(json.loads(Path(p).read_text())) != st["content_sha256"]:
                out.append(f"{p.name}: changed after it was stamped")
        else:
            out.append(f"{p.name}: its stamp binds no content hash")
        if st.get("code_hash") != code_hash_now:
            out.append(f"{p.name}: written by other code ({str(st.get('code_hash'))[:12]})")
        if st.get("partial"):
            out.append(f"{p.name}: a partial run ({st['partial']})")
        if st.get("deviation") and deviation is None:
            out.append(f"{p.name}: written under a deviation this run does not declare")
        t = st.get("tag") or {}
        if check_tag and (tag_now is None or t.get("object") != tag_now.get("object")
                          or t.get("commit") != tag_now.get("commit")):
            out.append(f"{p.name}: written under another tag ({str(t.get('object'))[:12]})")
        if env is not None and deviation is None and env_problems(env, st.get("environment")):
            out.append(f"{p.name}: computed with other package versions than the record's "
                       f"({env_problems(env, st.get('environment'))[:3]})")
    return out


def analysis_inputs(root: Path, rec: dict) -> list[Path]:
    """The post-tag files the analysis reads (the descriptive fields-earlier and
    fields-anchored-h4 rows included: stamp-checked when present, never required)."""
    from apexzero.fields import analysis as A

    out = [root / "fields-a2" / f"{n.replace('/', '_')}.jsonl" for n in A.A2_NETWORKS]
    out += [root / "fields-anchored" / f"{k.split(':', 1)[1]}.jsonl" for k in rec["key_sets"]
            if k.startswith("finals:")]
    out += [root / "gate2-trained" / "gate2_trained.json"]
    out += [harness_path(root, n) for n in without_rollouts(A.A2_NETWORKS)]
    out += [root / "fields-earlier" / f"{k.split(':', 1)[1]}.jsonl" for k in rec["key_sets"]
            if k.startswith("earlier:")]
    out += [root / "fields-anchored-h4" / f"{n.replace('/', '_')}.jsonl"
            for n in A.h4_networks()]
    return out


def run_once_problems(path: Path, root: Path = ROOT) -> list[str]:
    """Why the confirmatory analysis may not run again (design Section 15: it runs once): its
    decisions file exists, or was ever committed (any ref; a deleted file leaves its commit),
    when it lies inside ``root``."""
    out = [f"{path} exists"] if Path(path).exists() else []
    rel = _rel(path, root)
    if rel is not None:
        try:
            if committed_versions(rel, root):
                out.append(f"{rel} was committed before")
        except RuntimeError as e:
            out.append(f"git failed reading {rel}'s history: {e}")
    return out


def unlocked_by(path: Path, tag: dict | None, root: Path = ROOT) -> tuple[str | None, str]:
    """(sha256 of the confirmatory decisions that unlock a non-confirmatory analysis's
    decisions, or None; why not): a genuine stamped confirmatory output
    (``stages.confirmatory_done``) that, when it lies inside ``root``, is also committed as
    those bytes (so a peek against a fake or later-deleted file cannot happen unseen)."""
    from apexzero.fields import stages as S

    if not Path(path).exists():
        return None, "no confirmatory run yet"
    sha = S.confirmatory_done(path, tag)
    if sha is None:
        return None, f"{path} is not a stamped confirmatory output"
    rel = _rel(path, root)
    if rel is not None:
        from apexzero.fields import prereg

        try:
            shas = {prereg.sha256_bytes(b) for _, b in committed_versions(rel, root) if b}
        except RuntimeError as e:
            return None, f"git failed reading {rel}'s history: {e}"
        if sha not in shas:
            return None, f"{rel} is not committed (git add -f {rel} && git commit)"
    return sha, ""


def _jsonl_rows(path: Path) -> list[dict]:
    from apexzero.fields import stages as S

    return S.read_rows(path) if Path(path).exists() else []


def stage_analyse(args) -> None:
    """The analysis (design Sections 9 and 15). ``--confirmatory`` runs once: it refuses
    --agent and any --out but outputs/fields, refuses when outputs/fields/analyse/
    decisions.json exists or was ever committed (``run_once_problems``), unless every
    required network has a row for every frozen A2 key from its recorded checkpoint with the
    arrays its rows bind, every key an optimum row, gate2-trained a verdict per network on
    the record's keys, and every input a stamp of this code, this tag and the record's
    package versions (``input_stamp_problems``); it writes its decisions once to that file
    (with the deviation log and the tag pin it ran under) and prints its sha256 and the
    ``git add -f`` that commits it. Without it, the coverage and plumbing only
    (decisions_dry.json), never the decisions, until a committed, stamped confirmatory file
    exists (``unlocked_by``; its sha256 is then recorded as 'unlocked_by'); its --agent
    limits the per-agent field analyses (H1-typical, H2-speed, H1-fast and the earlier
    snapshots) to those agents."""
    rec = _post_tag(args, "the analysis")
    from apexzero.fields import analysis as A
    from apexzero.fields import prereg
    from apexzero.fields import stages as S

    fixed = OUT / "analyse" / "decisions.json"
    unlock = None
    if args.confirmatory:
        if args.agent:
            raise SystemExit("--agent is not allowed with --confirmatory (it reads every network)")
        if Path(args.out).resolve() != OUT.resolve():
            raise SystemExit("--confirmatory writes to outputs/fields only: drop --out")
        ro = run_once_problems(fixed)
        if ro:
            raise SystemExit("refusing to run the confirmatory analysis again (design Section "
                             f"9: the analysis runs once): {'; '.join(ro)}")
        root = OUT
        probs = S.confirmatory_input_problems(rec, root)
        probs += input_stamp_problems([p for p in analysis_inputs(root, rec) if p.exists()],
                                      _prov().code_hash, _prov().deviation, _prov().tag,
                                      rec.get("environment"))
        if probs:
            raise SystemExit("the confirmatory analysis runs once, on complete inputs: "
                             + "; ".join(probs[:20]))
        path = S.confirmatory_guard(fixed.parent)
        path.parent.mkdir(parents=True, exist_ok=True)
        decide = True
    else:
        root = Path(args.out)
        unlock, why = unlocked_by(fixed, _prov().tag)
        if unlock is None and fixed.exists():
            print(f"warning: {why}: coverage and plumbing only", flush=True)
        decide = unlock is not None
        path = _out(args, "analyse") / "decisions_dry.json"
    opt = {r["key_id"]: r for r in S.read_rows(root / "opt-fields" / "rows.jsonl")}
    opt_dir = root / "opt-fields" / "arrays"
    net_rows, net_dirs = {}, {}
    for name in A.A2_NETWORKS:
        rows = _jsonl_rows(root / "fields-a2" / f"{name.replace('/', '_')}.jsonl")
        if rows:
            net_rows[name] = rows
            net_dirs[name] = S.arrays_dir(root / "fields-a2" / "arrays", rows[0]["net"])
    g2 = root / "gate2-trained" / "gate2_trained.json"
    g2d = json.loads(g2.read_text()) if g2.exists() else None
    gate2_ok = ({n: bool(v["pass"]) for n, v in g2d["networks"].items()}
                if g2d is not None else None)
    regimes = (rec.get("regimes") or {}).get("a2_regime_cells", {})
    inputs = {str(p.relative_to(root)): prereg.sha256_file(p)
              for p in analysis_inputs(root, rec) if p.exists()}
    for stage, sub in (("opt-fields", "arrays"), ("qp-fields", "arrays/reader-Qp")):
        rp = root / stage / "rows.jsonl"
        if rp.exists():
            inputs[f"{stage}/rows.jsonl"] = prereg.sha256_file(rp)
            inputs[f"{stage}/{sub}"] = prereg.dir_hash(root / stage / sub)[0]
    res = {"confirmatory": bool(args.confirmatory), "decided": decide, "unlocked_by": unlock,
           "inputs": inputs,
           "gate2_trained": {n: {k: v.get(k) for k in ("pass", "n_keys", "n_checked",
                                                      "n_exhausted")}
                             for n, v in (g2d or {}).get("networks", {}).items()}}
    if args.confirmatory:          # what the run ran under, bound by decisions.json's sha256
        dl = OUT / "deviations.jsonl"
        tp = OUT / "tag.json"
        res["deviations"] = ([json.loads(ln) for ln in dl.read_text().splitlines() if ln.strip()]
                             if dl.exists() else [])
        res["tag_pin"] = json.loads(tp.read_text()) if tp.exists() else None
    res.update(S.analyse_edges(net_rows, opt, gate2_ok=gate2_ok, regime_cells=regimes,
                               decide=decide))
    qrows = _jsonl_rows(root / "qp-fields" / "rows.jsonl")
    qp = ({r["key_id"]: r for r in qrows}, root / "qp-fields" / "arrays" / "reader-Qp") \
        if qrows else None
    only = set(args.agent or ())          # non-confirmatory only: these agents' field analyses

    def wanted(agent: str) -> bool:
        return not only or agent in only

    res["fields"] = {}
    for name in A.H2_NETWORKS:
        if name in net_rows and wanted(name):
            bulk = set(regimes.get(name, {}).get("bulk", ()))
            res["fields"][name] = S.analyse_fields(net_rows[name], opt, net_dirs[name], opt_dir,
                                                   bulk, qp=qp, decide=decide)
    res["H1_fast"], res["anchored_edges"], res["earlier"] = {}, {}, {}
    for k in rec["key_sets"]:
        if not k.startswith(("finals:", "earlier:")):
            continue
        name = k.split(":", 1)[1]
        agent = name.split("@")[0]
        if not wanted(agent):
            continue
        stage = "fields-anchored" if k.startswith("finals:") else "fields-earlier"
        f = root / stage / f"{name}.jsonl"
        rows = _jsonl_rows(f)
        if not rows:
            continue
        d = S.arrays_dir(root / stage / "arrays", rows[0]["net"])
        if stage == "fields-anchored":
            res["H1_fast"][agent] = S.h1_fast_rows(rows, opt, d, opt_dir, agent, decide=decide)
            if decide:                                    # descriptive: H2' on anchored keys
                res["anchored_edges"][agent] = S.anchored_edges(rows, opt)
        elif decide:                                      # descriptive: earlier snapshots
            res["earlier"][name] = {"anchored_edges": S.anchored_edges(rows, opt),
                                    "H1_fast": S.h1_fast_rows(rows, opt, d, opt_dir, agent)}
    if decide:                                            # descriptive: the anchored-key H4
        h4 = {n: _jsonl_rows(root / "fields-anchored-h4" / f"{n.replace('/', '_')}.jsonl")
              for n in A.h4_networks()}
        h4 = {n: v for n, v in h4.items() if v}
        if h4:
            res["anchored_H4"] = S.anchored_h4(h4, opt)
    if args.confirmatory and run_once_problems(path):    # written once (another run got there)
        raise SystemExit(f"refusing to overwrite the confirmatory decisions {path}")
    _write(path, res)
    sha = prereg.sha256_file(path)
    print(f"{path.name} sha256 {sha}"
          + ("" if decide else " (coverage and plumbing only: no committed confirmatory run)"))
    if args.confirmatory:
        rel = _rel(path) or str(path)
        print(f"commit it now: git add -f {rel} && git commit -m 'Step 2: confirmatory "
              f"decisions {sha[:12]}'", flush=True)


# ------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=2, help="XLA intra-op threads")
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in STAGES:
        p = sub.add_parser(name)
        p.add_argument("--out", default=str(OUT), help="output root (default outputs/fields)")
        p.add_argument("--log-dir", default=str(ROOT / "runs"),
                       help="where fields-<stage>.log goes (default runs/)")
        p.add_argument("--workers", type=int, default=1, help="key worker processes")
        p.add_argument("--force", action="store_true", help="redo instead of resuming")
        p.add_argument("--post-tag-deviation", default=None, metavar="REASON",
                       help="after the tag: log a dated deviation and run (post-tag stages: "
                            "waives only changed code or a changed design document; pre-tag "
                            "stages: only with --out outside outputs/fields)")
        if name in ("keys", "repolish", "opt-fields"):
            p.add_argument("--kinds", nargs="+", default=list(DEFAULT_KINDS),
                           choices=("a2", "finals", "earlier"))
        if name in ("keys", "repolish", "gate3", "gate4", "opt-fields", "qp-fields",
                    "harness-f32", "fields-a2", "fields-anchored", "fields-earlier",
                    "fields-anchored-h4", "dose"):
            p.add_argument("--limit", type=int, default=None,
                           help="this many keys, spread evenly over the list (a partial run)")
        if name == "keys":
            p.add_argument("--no-gate0", action="store_true", help="the lists only")
            p.add_argument("--no-audit", action="store_true", help="skip the loop audit")
        if name == "gate4":
            p.add_argument("--no-twins", action="store_true",
                           help="leave out the 6 tier V horizon twins (default: all 481)")
        if name in ("harness-f32", "gate2-trained", "fields-a2", "fields-anchored-h4", "dose"):
            p.add_argument("--network", action="append", default=None,
                           help="network (registry id or runs/<dir>; repeatable): an "
                                "exploratory run, only with --out outside outputs/fields")
        if name in ("gate1", "gate5"):
            p.add_argument("--n", type=int, default=None,
                           help="the first n keys of the stratified sample (round-robin over "
                                "strata); default all 50")
        if name in ("gate3", "fields-anchored", "fields-earlier", "analyse"):
            p.add_argument("--agent", action="append", default=None,
                           help="agent id (repeatable); gate3 default G2; analyse: only "
                                "these agents' field analyses, refused by --confirmatory")
        if name == "gate3":
            p.add_argument("--n-anchored", type=int, default=None,
                           help="planted sample's anchored finals (default "
                                "gates.GATE3_N_ANCHORED = 3; fewer is partial)")
            p.add_argument("--n-a2", type=int, default=None,
                           help="planted sample's A2 keys (default gates.GATE3_N_A2 = 2; "
                                "fewer is partial)")
            p.add_argument("--all", action="store_true",
                           help="the clean check on every key with a det row (exclusions)")
        if name == "controls":
            p.add_argument("--which", nargs="+",
                           default=["RF-C1", "RF-C2", "RF-C3", "RF-C4", "RF-C5", "readers"],
                           choices=("RF-C1", "RF-C2", "RF-C3", "RF-C4", "RF-C5", "readers"))
            p.add_argument("--quick", action="store_true",
                           help="the multi-cell controls on fewer cells (controls.QUICK_CELLS)")
        if name == "gate2":             # defaults: gates.GATE2_CELLS (other cells: partial)
            p.add_argument("--random", nargs="*", type=int, default=None,
                           help="A2 cells of the random-init actor (steer; default 62 345)")
            p.add_argument("--keeper", nargs="*", type=int, default=None,
                           help="cells of the planted edge keeper (plan-level outcomes; "
                                "default 344)")
            p.add_argument("--linear", nargs="*", type=int, default=None,
                           help="cells of the planted linear driver (plan-level outcomes; "
                                "default 344 345 66)")
            p.add_argument("--hybrid", nargs="*", type=int, default=None,
                           help="cells of the hybrid control (linear driver + small-gain "
                                "random-init MLP through G1's obs_rms; plan-level outcomes; "
                                "default network.HYBRID_CELLS, 344 66)")
        if name == "analyse":
            p.add_argument("--confirmatory", action="store_true",
                           help="the run-once analysis into outputs/fields/analyse/decisions.json")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _PROV[:] = [Provenance(args, args.stage)]
    if args.stage in PRE_TAG:
        _pre_tag(args, args.stage)
        if args.stage not in X64_OFF:
            _x64()                   # post-tag stages switch it on after the freeze check
    t0 = time.time()
    globals()["stage_" + args.stage.replace("-", "_")](args)
    print(f"{args.stage}: {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
