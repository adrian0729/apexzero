"""Idea 2, the costate: the stages of the design (docs/interp/costate-design.md, v2.1).

    uv run --no-sync python scripts/costate.py <stage> [options]

Pre-tag stages (design Sections 1 and 7; no trained network's activation, value or gradient
is computed; refused once the tag exists):

    inputs      T's rows over the A2 keys with their targets (re-polished), tail values and
                the network-independent observation maps (inputs/world.npz), the frozen
                values (winsorisation bounds, f_D's k, residual shares, power floor, folds)
                and, per --network (default G1), the observability and kappa flags
                (inputs/summary.json).
                The 4 multi-basin keys are left out of T (design Section 3.2) and listed;
                run it after k2 (its t_H is bound) and again before prereg
    k0          K0 on the A2 keys and N's anchored finals (--kinds a2: the A2 keys only, a
                partial run)
    k1          K1 on step 2's Gate 1 sample (50 keys; --n fewer, a partial run), with the
                planted bugs and pi on the anchored keys
    k2          K2 on the horizon twins, and --n-extend W = 800 insurance solves
    k2b         K2b on --n keys (default 30) stratified by R and theta
    k3          K3: the critic's pullback dry runs on 200 T states (--n fewer keys: a
                partial run) and the extractor's self-consistency (float64 subprocesses,
                each report removed before its subprocess runs, their exit statuses
                checked and recorded)
    k3b         K3b: the policy's own costate loop on step 2's linear driver, its accel-reading
                plant and the hybrid, with the on-manifold check
    k4          K4's planted representations through the probe pipeline on T (G1's obs_rms;
                PR-6b and PR-6c on C2's decision rows: run after inputs and k2b)
    decoy-k     f_D's k re-derived on this world: PR-6b's and PR-6c's validation seeds at
                every candidate k, the rule's choice, and the reproduction of the record's
                table (``gates.decoy_k``; gates/decoy_k.json; after inputs and k2b)
    c2-null     C2's random-only null, descriptive: 20 random-init actors (seeds 20-39)
                scored as the network through C2's primary statistic against C2's control
                pool (seeds 0-19) with the two-level CI (``gates.c2_null``;
                gates/c2_null.json; after inputs and k2b)
    k5          K5: the permuted-target wiring check and the control task
    k6          K6: coverage per component and reason
    nulls       C1's input-side nulls per network obs_rms (default: the 9 decision networks;
                fewer is partial): value-fit ridge and MLP on the horizon-corrected value, the
                window value's ridge, QSS, barrier, far-geometry lambda (nulls/<run>.npz named
                by run directory: the value fits and far-geometry lambda [6, n, 8] per
                held-out fold, QSS and barrier [n, 8]; nulls/summary.json with each array's
                sha256 and shapes)
    prereg      the pre-registration record src/apexzero/costate/prereg.json, from a clean
                tree, binding the stamped passing outputs above: the last step before the tag

Post-tag stages (refused unless the freeze check passes: ``freeze_problems``):

    activations the actors' h1-h3 and m and the critics' layers and heads at T, with the
                random-init actors' h3 and m on each network's z (a decision network's: the
                control pool's 20, seeds 0-19; every other network's: the 5 controls)
    critic      first K3's after-the-tag checks on each network at D, T's t = 0 rows
                (``k3_trained``, post/critic/<net>.k3.json: the extractor's float64
                self-consistency against Actor.apply and Critic.apply(with_heads=True) to
                1e-10, 5 heads per generalist and none for S1-S3, the float64 V against the
                float32 one to 1e-4); then the trained critics' values and pullbacks (V and
                heads) and uniform inside-widening derivatives at T, and the 20 random-init
                critics' pullbacks (post/critic/<net>.npz, with 'k3_trained_pass')
    own         the policy's own costate (C1-own) at T's interior restarts, every 20th step,
                K3b's trained analogue, and the deterministic policy's own discounted returns
                from those restarts and from D (up to 1,000 steps or to the strip's end):
                value accuracy's target
    probes      C2 and C2-sw per decision network, with the pre-registered sensitivity and
                robustness variants (partitions 1-10, geometry and state weights, the small
                geometries and the kappa set dropped, braking and non-braking sub-scores)
                and their descriptive strata tables (a primary entry is 'not testable' only
                on ``probes.NotTestable``); the primary C2 and C2-sw against the control
                pool with the two-level CI, every other record against the 5 controls with
                the geometry-only CI (``probes_of``)
    analyse     the analysis (C1 and its secondaries with the accel-held pullback, the
                critic's value accuracy against its own policy's return, which with K3b's
                trained analogue decides C1-own's standing, the per-network strata tables of
                every hypothesis, C3-w and C3-R2c over the 31 generalists, the
                descriptive C1 and C1-heads of the networks outside the decision set, K4's
                notes beside C2 decisions near the band (``analysis.band_notes``) and C2's
                random-only null beside each primary C2 (``analysis.c2_null_note``); every
                descriptive contained, ``analysis.guarded``; a network that failed K3's
                after-the-tag checks is descriptive and missing from every decision,
                ``analysis.k3_trained_failed``, and listed): --confirmatory runs once, into
                outputs/costate/analyse/decisions.json, only on complete inputs stamped by
                this code under this tag; a non-confirmatory run computes no decision at all
                (coverage and plumbing only) until a stamped confirmatory decisions file is
                committed (``unlocked_by``)

Each stage writes under outputs/costate/<stage group>/ (``--out`` moves the root; any root
other than outputs/costate, a --limit or --n or a --network makes the run partial, as does
an option that changes a bound output from its pre-registered form: ``_partial``) and logs to
outputs/costate/logs/costate-<stage>.log (never under runs/, which is the main checkout's).
Every output carries a provenance stamp (``F.Provenance``'s fields on idea 2's code paths:
the code hash at the start and just before writing, HEAD, the changed paths, argv, the
partial options, times, any deviation and the tag), embedded in a JSON object or as a
<file>.provenance.json sidecar. An npz is written with fixed zip timestamps (``savez``), so
its sha256 depends on its arrays alone.

**Worker pools** (``POOLED``: probes, own, nulls, decoy-k; ``apexzero.costate.workers``).
Their units (a network each; decoy-k's (k, world, seed)) are independent, and --workers N
(default ``workers.default_workers``: the cores over --threads, less one, 5 on the 12-core
Mac at 2 threads) runs them in N spawned processes; --workers 1 is the serial path, in this
process. Each unit's seeds depend on the unit alone; every worker shares this process' thread
pins (--threads sets XLA's intra-op threads; a pooled stage pins the BLAS libraries to one
thread, ``BLAS_THREADS``, before numpy loads), its x64 setting and its trained-network locks;
this process loads the checkpoints and checks their pins, gathers the results in the serial
order and only then writes and stamps every output, exactly as the serial path does. The
outputs are therefore the serial run's to the bit; the stamp records the pool ('workers',
'threads'), which is never partial and enters no content hash. A unit that fails in a worker
fails the stage, and nothing is written. One BLAS thread, not --threads: on this Mac
Accelerate gives decoy-k's units the same bits at 1, 6 and 12 threads as unpinned (every
earlier run) but not at 2 (decoy-k's full-size Delta2 up to 8e-11 apart paired, 1.5e-10
unpaired: design Section 16.6, well inside the 1e-6 reproduction tolerance), and a worker's
numpy work runs on one core either way.

**Freeze rule** (design Section 7 'Freeze'; ``freeze_problems``, step 2's rule simplified):
the tag costate-prereg-v1 must exist as an annotated tag, be on origin as the same object and
be an ancestor of HEAD; the write-once outputs/costate/tag.json the first post-tag run writes
must be committed before the next post-tag stage and stay its first committed version;
nothing under src/ (all of it), scripts/costate.py or scripts/_bootstrap.py may differ from
the tag (git diff, untracked files, and byte by byte); step 1's and step 2's frozen code must
be their own tags', and those local tags must be the objects the record registers
(``steps``; step 1's also its pinned ``F.STEP1_TAG_OBJECT``: scripts/fields.py is outside the
code hash, frozen only by step 2's tag); the record at the tag must exist, register the
design document's blob at the tag, hash the code the tag holds (``prereg.tree_code_hash``),
come from a base commit that differs from the tag by the record alone; package versions must
be the record's; and the disk must match the record (key files and lists, T's list, design
document, code hash, bound outputs, checkpoints). Only then are ``critic.unlock_trained``,
``probes.unlock_trained`` and ``network.unlock_trained`` called. A logged
``--post-tag-deviation REASON`` waives only a changed-code or changed-design-document
problem.

**Run-once** (Section 7: the analysis runs once): ``analyse --confirmatory`` refuses when its
decisions file exists or was ever committed, refuses unless every input is present (the own
stage's arrays, K3b's trained analogue and every network's K3 after-the-tag checks among
them) and stamped by this code under this
tag (the pre-tag nulls arrays: by the record's code, and
the arrays the record binds; ``confirmatory_input_problems``), and prints the
``git add -f`` that commits it. Before that file is committed, a non-confirmatory analyse
writes coverage and plumbing only (``unlocked_by``, as step 2's).

**Where it runs** (step 2's and H4R's confirmatory decisions are already committed on main):
idea 2 runs in the main checkout and is tagged on main. The pre-tag stages write
outputs/costate there and prereg writes the record there; then commit
src/apexzero/costate/prereg.json on main, ``git tag -a costate-prereg-v1`` on that commit and
``git push origin main costate-prereg-v1``. Every post-tag stage runs in the main checkout,
where the bound pre-tag outputs are and where tag.json and decisions.json are committed.
"""

import argparse
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


def _stage_arg(argv: list[str]) -> str | None:
    """The stage on the command line: the first argument that is neither an option nor
    --threads' value."""
    skip = False
    for a in argv[1:]:
        if skip:
            skip = False
        elif a == "--threads":
            skip = True
        elif not a.startswith("-"):
            return a
    return None


# The stages with worker pools (``workers.POOLED``) and the BLAS libraries' thread variables
# (``workers.BLAS_THREADS``; the tests check both copies): a pooled stage pins them to one
# thread before numpy loads, in this process and so in every worker it spawns (module
# docstring 'Worker pools').
POOLED = ("probes", "own", "nulls", "decoy-k")
BLAS_THREADS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS")

_T = _threads(sys.argv)
if _T is not None or "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               f"intra_op_parallelism_threads={_T or 2}")
if _stage_arg(sys.argv) in POOLED:
    os.environ.update(dict.fromkeys(BLAS_THREADS, "1"))

import _bootstrap  # noqa: F401  (makes src/ importable)
import fields as F

ROOT = Path(__file__).resolve().parents[1]
TAG = "costate-prereg-v1"
REMOTE = F.REMOTE
STEP1_TAG, STEP1_FROZEN = F.STEP1_TAG, F.STEP1_FROZEN
STEP2_TAG = F.TAG
STEP2_FROZEN = ("src/apexzero/fields", "scripts/fields.py")
CODE_PATHS = ("src", "scripts/costate.py", "scripts/_bootstrap.py")
DESIGN_DOC = "docs/interp/costate-design.md"
PREREG_REL = "src/apexzero/costate/prereg.json"
OUT_REL = "outputs/costate"
OUT = ROOT / OUT_REL
DECISIONS_REL = f"{OUT_REL}/analyse/decisions.json"
PRE_TAG = ("inputs", "k0", "k1", "k2", "k2b", "k3", "k3b", "k4", "decoy-k", "c2-null", "k5", "k6",
           "nulls", "prereg")
POST_TAG = ("activations", "critic", "own", "probes", "analyse")
STAGES = PRE_TAG + POST_TAG
X64_OFF = ("k3",)               # its float64 work runs in subprocesses
GROUP = {"inputs": "inputs", "nulls": "nulls", "prereg": "prereg", "activations": "post",
         "critic": "post", "own": "post", "probes": "post", "analyse": "analyse"}
K3B_CELLS = (("linear", 344), ("linear", 345), ("linear", 66), ("accel", 344), ("hybrid", 344),
             ("hybrid", 66))
K3B_TRAINED_CELLS = (344, 66)   # the trained analogue of K3b (C1-own's validity), per network
OWN_EVERY = 20                  # Section 3.3: T's interior restarts every 20th step
                                # (= own.RESTART_EVERY, the record's; checked in the tests)


# ------------------------------------------------------------------------- the freeze


def _P(code: str, msg: str):
    from apexzero.fields.prereg import Problem

    return Problem(code, msg)


def tagged_record(tag: str = TAG, root: Path = ROOT) -> dict | None:
    r = F._git("show", f"{tag}:{PREREG_REL}", root=root)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return json.loads(r.stdout)


def tagged_doc_sha256(tag: str = TAG, root: Path = ROOT) -> str | None:
    import hashlib

    r = F._git_bytes("cat-file", "blob", f"{tag}:{DESIGN_DOC}", root=root)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def pin_path_of(root: Path = ROOT) -> Path:
    return Path(root) / OUT_REL / "tag.json"


def frozen_steps_problems(root: Path = ROOT) -> list:
    """Step 1's and step 2's frozen code against their own tags (git and byte by byte); in
    this repository step 1's tag must also be its pinned object (``F.STEP1_TAG_OBJECT``, as
    ``fields.step1_problems``)."""
    out = []
    for tag, paths, label in ((STEP1_TAG, STEP1_FROZEN, "step 1"),
                              (STEP2_TAG, STEP2_FROZEN, "step 2")):
        t = F._tag_object(tag, root)
        if t is None:
            out.append(_P("frozen", f"{label}'s tag {tag} is missing"))
            continue
        if (tag == STEP1_TAG and Path(root).resolve() == ROOT.resolve()
                and t[0] != F.STEP1_TAG_OBJECT):
            out.append(_P("frozen", f"{label}'s tag {tag} is {t[0][:12]}, not its pinned object "
                          f"{F.STEP1_TAG_OBJECT[:12]}"))
        ch = F.changed_since(tag, paths, root)
        if ch:
            out.append(_P("frozen", f"{label}'s frozen code differs from {tag}: {ch[:6]}"))
        try:
            hidden = sorted(set(F.byte_changes(tag, paths, root)) - {c.split(" ")[0] for c in ch})
        except RuntimeError as e:
            out.append(_P("git", f"git failed reading {tag}: {e}"))
            continue
        if hidden:
            out.append(_P("frozen", f"{label}'s frozen code differs from {tag} byte by byte: "
                          f"{hidden[:6]}"))
    return out


def step_tags_problems(rec: dict, root: Path = ROOT) -> list:
    """Step 1's and step 2's local tags against the objects the record registers
    (``steps``, written by prereg): ``frozen_steps_problems`` diffs against the local tag
    names, so a moved tag would let changed frozen code through, scripts/fields.py among it
    (outside the code hash). Never waivable ('frozen')."""
    steps = rec.get("steps") or {}
    out = []
    for key, tag in (("step1", STEP1_TAG), ("step2", STEP2_TAG)):
        want = (steps.get(key) or {}).get("object")
        t = F._tag_object(tag, root)
        if not want:
            out.append(_P("frozen", f"the record registers no object for {key}'s tag {tag}"))
        elif t is None or t[0] != want:
            out.append(_P("frozen", f"{key}'s tag {tag} is {(t or ['missing'])[0][:12]}, not "
                          f"the record's {want[:12]}"))
    return out


def _verify_on_disk(rec: dict, root: Path) -> list:
    from apexzero.costate import keys_root
    from apexzero.costate import prereg as PR

    return PR.verify_record(rec, root, keys_root(), lambda: PR.key_sets(), Path(root) / OUT_REL,
                            lambda pins: PR.pins_problems(pins, root),
                            design_hash=PR.stimulus_design_hash())


def freeze_problems(tag: str = TAG, root: Path = ROOT, verify=_verify_on_disk,
                    remote: str | None = REMOTE, pin_path: Path | None = None) -> list:
    """Why a post-tag stage may not run (module docstring 'Freeze rule'), as
    ``fields.prereg.Problem`` strings with a code."""
    from apexzero.costate import prereg as PR

    t = F._tag_object(tag, root)
    if t is None:
        return [_P("tag", f"the git tag {tag} does not exist yet (design Section 7: no trained "
                   "network's activation or gradient before it)")]
    out = []
    if t[1] != "tag":
        out.append(_P("tag", f"{tag} is a lightweight tag: it must be annotated"))
    if remote is not None:
        try:
            on = F.remote_tag_object(tag, root, remote)
            if on != t[0]:
                out.append(_P("tag", f"{tag} is not on {remote} as {t[0][:12]} (there: {on})"))
        except RuntimeError as e:
            out.append(_P("tag", f"cannot confirm {tag} on {remote}: {e}"))
    commit = F._tag_commit(tag, root)
    if F._git("merge-base", "--is-ancestor", commit or tag, "HEAD", root=root).returncode != 0:
        out.append(_P("tag", f"the tag {tag} is not an ancestor of HEAD"))
    pin_path = Path(pin_path) if pin_path is not None else pin_path_of(root)
    if pin_path.exists():
        pin = json.loads(pin_path.read_text())
        if pin.get("object") != t[0] or pin.get("commit") != commit:
            out.append(_P("tag", f"the tag moved since the first post-tag run ({pin_path})"))
    out += list(F._pin_committed_problems(pin_path, root, t[0], commit))
    ch = F.changed_since(tag, CODE_PATHS, root)
    if any(c.startswith("<git error") for c in ch):
        out.append(_P("git", f"git failed on the code paths: {ch[:2]}"))
    elif ch:
        out.append(_P("code_changed", f"code differs from the tag {tag}: {ch[:8]}"))
    try:
        hidden = sorted(set(F.byte_changes(tag, CODE_PATHS, root)) - {c.split(" ")[0] for c in ch})
        if hidden:
            out.append(_P("code_changed", f"code differs from the tag {tag} byte by byte: "
                          f"{hidden[:8]}"))
    except RuntimeError as e:
        out.append(_P("git", f"git failed reading the tag's code: {e}"))
    out += frozen_steps_problems(root)
    rec = tagged_record(tag, root)
    if rec is None:
        out.append(_P("record", f"no pre-registration record {PREREG_REL} at the tag {tag}"))
        return out
    out += step_tags_problems(rec, root)
    if not rec.get("design_doc_sha256") or tagged_doc_sha256(tag, root) != rec["design_doc_sha256"]:
        out.append(_P("record", "the design document at the tag is not the one the record "
                      "registers"))
    try:
        if PR.tree_code_hash(Path(root), tag) != rec.get("code_hash"):
            out.append(_P("record", "the code at the tag is not the code the record was built "
                          "from"))
    except RuntimeError as e:
        out.append(_P("git", f"git failed hashing the code at the tag: {e}"))
    base = rec.get("prereg_base_commit")
    if not base or F._git("cat-file", "-e", f"{base}^{{commit}}", root=root).returncode != 0:
        out.append(_P("record", f"the record's base commit {base!r} is missing or unknown"))
    else:
        d = F._git("diff", "--name-only", base, tag, "--", *CODE_PATHS, DESIGN_DOC, root=root)
        extra = [ln for ln in d.stdout.splitlines() if ln.strip() and ln != PREREG_REL]
        if d.returncode != 0 or extra:
            out.append(_P("record", f"the tag differs from the record's base commit by more "
                          f"than the record: {extra[:6]}"))
    env = F.env_problems(rec.get("environment"), F.environment(root))
    if env:
        out.append(_P("code_changed", f"package versions differ from the record's: {env[:6]}"))
    if verify is not None:
        out.extend(p if hasattr(p, "code") else _P("", str(p)) for p in verify(rec, root))
    return out


def log_deviation(what: str, reason: str, problems: list, category: str, root: Path = ROOT,
                  log_root: Path | None = None) -> dict:
    from apexzero.fields import prereg as FP

    rec = {"date": time.strftime("%Y-%m-%d"), "time": time.strftime("%H:%M:%S"),
           "stage": what, "category": category, "reason": reason,
           "problems": [{"code": FP.problem_code(p), "message": str(p)} for p in problems],
           "head": F._head(root)}
    path = Path(log_root or (Path(root) / OUT_REL)) / "deviations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def pin_tag(root: Path = ROOT, pin_path: Path | None = None) -> Path:
    """The write-once tag.json (the tag's identity at the first post-tag run)."""
    pin_path = Path(pin_path) if pin_path is not None else pin_path_of(root)
    ident = F.tag_identity(TAG, root)
    if ident is not None and not pin_path.exists():
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pin_path, "x") as fh:
            fh.write(json.dumps({**ident, "pinned_at": F._now()}, indent=1) + "\n")
        print(f"pinned the tag in {pin_path}: commit it (git add -f) before the next post-tag "
              "stage", flush=True)
    return pin_path


def require_tag(what: str, deviation: str | None = None, root: Path = ROOT, problems=None,
                pin_path: Path | None = None) -> dict | None:
    """Refuse a post-tag stage unless ``freeze_problems`` is empty; a deviation waives only
    the waivable codes."""
    from apexzero.fields import prereg as FP

    probs = freeze_problems(root=root, pin_path=pin_path) if problems is None else problems
    if not probs:
        pin_tag(root, pin_path)
        return None
    if deviation and FP.waivable(probs):
        dev = log_deviation(what, deviation, probs, "post-tag deviation", root)
        pin_tag(root, pin_path)
        return dev
    raise SystemExit(f"refusing to run {what}: " + "; ".join(str(p) for p in probs))


def refuse_after_tag(what: str, deviation: str | None = None, always: bool = False,
                     root: Path = ROOT, out_root: Path | None = None) -> dict | None:
    """Pre-tag stages refuse once the tag exists; with a deviation only into a root outside
    outputs/costate (never for 'prereg')."""
    if F._tag_object(TAG, root) is None:
        return None
    if always or not deviation:
        raise SystemExit(f"refusing to run {what} after the tag {TAG}: it would change a "
                         "pre-registered output")
    canon = (Path(root) / OUT_REL).resolve()
    out = Path(out_root).resolve() if out_root is not None else canon
    if out == canon or canon in out.parents or out in canon.parents:
        raise SystemExit(f"refusing to run {what} after the tag into {out}: give --out outside "
                         f"{OUT_REL}")
    return log_deviation(what, deviation, [], "pre-tag stage after the tag", root)


def import_problems(root: Path = ROOT, modules: dict | None = None) -> list[str]:
    """Every module loaded from inside the repository must be frozen code: under
    src/apexzero, or scripts/costate.py, scripts/fields.py (step 2's helpers, frozen at its
    tag) or scripts/_bootstrap.py; apexzero itself must be this checkout's."""
    import apexzero

    root = Path(root).resolve()
    want = (root / "src" / "apexzero").resolve()
    got = Path(apexzero.__file__).resolve().parent
    out = [] if got == want else [f"apexzero is imported from {got}, not {want}"]
    prefixes = {Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    allowed = {(root / "scripts" / n).resolve() for n in ("costate.py", "fields.py",
                                                          "_bootstrap.py")}
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


# ------------------------------------------------------------------------- provenance


def _partial(args) -> list[str]:
    """The options that make a run partial: another --out, any --limit, --n or --n-extend
    given (0 included), and --network except on the nulls stage when the networks given
    cover the decision networks (named by run directory: G1 and runs/dr-pilot-200M are the
    same); and those that change a bound output from its pre-registered form: k0 without 'n'
    in --kinds (Section 13: K0 on every A2 key and every anchored final), any k4 --which (its
    pass covers only the plants that ran), a k5 --ceiling other than 'mlp' (ridge is the dry
    runs' stand-in), k3b --steps other than ``own.K3B_STEPS``, nulls --mlp-alphas or
    --mlp-maxiter other than the record's (``probes.MLP_ALPHAS``, ``probes.MLP_MAXITER``),
    and inputs --partitions other than all of them (``states.N_PARTITIONS``)."""
    out = []
    stage = getattr(args, "stage", None)
    if Path(getattr(args, "out", OUT)).resolve() != OUT.resolve():
        out.append("--out")
    for name in ("limit", "n", "network", "n_extend"):
        v = getattr(args, name, None)
        if v is None or v is False or (isinstance(v, list) and not v):   # 0 is given
            continue
        if name == "network" and stage == "nulls":
            from apexzero.costate import prereg as PR

            if {PR.net_file(n) for n in PR.null_networks()} <= {PR.net_file(n) for n in v}:
                continue
        out.append(f"--{name.replace('_', '-')}")
    if stage == "k0" and "n" not in (getattr(args, "kinds", None) or []):
        out.append("--kinds")
    if stage == "k4" and getattr(args, "which", None):
        out.append("--which")
    if stage == "k5" and getattr(args, "ceiling", "mlp") != "mlp":
        out.append("--ceiling")
    if stage == "k3b":
        from apexzero.costate import own as OW

        s = getattr(args, "steps", None)
        if s is not None and int(s) != OW.K3B_STEPS:
            out.append("--steps")
    if stage == "nulls":
        from apexzero.costate import probes as PB

        a = getattr(args, "mlp_alphas", None)
        if a is not None and tuple(map(float, a)) != tuple(map(float, PB.MLP_ALPHAS)):
            out.append("--mlp-alphas")
        if int(getattr(args, "mlp_maxiter", PB.MLP_MAXITER)) != PB.MLP_MAXITER:
            out.append("--mlp-maxiter")
    if stage == "inputs":
        from apexzero.costate import states as ST

        p = getattr(args, "partitions", None)
        if p is not None and int(p) != ST.N_PARTITIONS:
            out.append("--partitions")
    return out


class Provenance(F.Provenance):
    """step 2's ``Provenance`` on idea 2's code paths (module docstring)."""

    def __init__(self, args=None, stage: str = "", root: Path = ROOT):
        from apexzero.costate import prereg as PR

        self.root = Path(root)
        self.stage = stage
        self.code_hash = PR.code_hash(self.root)
        self.head = F._head(self.root)
        self.changed = F.changed_since("HEAD", CODE_PATHS + (DESIGN_DOC,), self.root)
        self.argv = list(sys.argv)
        self.partial = _partial(args) if args is not None else []
        self.start = F._now()
        self.deviation = None
        self.tag = None
        self.environment = F.environment(self.root)
        self.check_imports = False
        self.workers = None             # a pooled stage's worker processes (``_workers``)

    def stamp(self, files: dict | None = None, content_sha256: str | None = None) -> dict:
        from apexzero.costate import prereg as PR
        from apexzero.costate import workers as WK

        now = PR.code_hash(self.root)
        if now != self.code_hash:
            raise SystemExit(f"refusing to write the {self.stage} output: the code changed "
                             "while the stage ran")
        bad = import_problems(self.root) if self.check_imports else []
        if bad:
            raise SystemExit(f"refusing to write the {self.stage} output: {'; '.join(bad[:4])}")
        out = {"stage": self.stage, "code_hash": self.code_hash, "head": self.head,
               "changed_vs_head": self.changed, "argv": self.argv, "partial": self.partial,
               "start": self.start, "end": F._now(), "deviation": self.deviation,
               "tag": self.tag, "environment": self.environment, "files": dict(files or {})}
        if self.workers is not None:
            out["workers"], out["threads"] = int(self.workers), WK.thread_env()
        if content_sha256 is not None:
            out["content_sha256"] = content_sha256
        return out


def _use_provenance(p: Provenance) -> None:
    """Our provenance for this run, in step 2's helpers too (``F._write`` stamps through
    ``F._prov``)."""
    F._PROV[:] = [p]


def _write(path: Path, obj) -> Path:
    return F._write(Path(path), obj)


def savez(path: Path, arrays: dict) -> Path:
    """``np.savez_compressed``'s file (one deflated '<name>.npy' member per array, in order)
    with every member's zip timestamp fixed at 1980-01-01, so its bytes, and the sha256 that a
    summary or a stamp records, depend on the arrays alone: the same arrays written twice (by
    a serial and a pooled run, ``workers``) are the same file."""
    import zipfile

    import numpy as np
    from numpy.lib import format as npf

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for name, v in arrays.items():
            zi = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(zi, "w", force_zip64=True) as fh:
                npf.write_array(fh, np.asanyarray(v), allow_pickle=True)   # as numpy's
    return Path(path)


def _write_npz(path: Path, arrays: dict) -> Path:
    return F._stamped_write(Path(path), lambda q: savez(q, arrays))


def _workers(args, units: int | None = None) -> int:
    """A pooled stage's worker processes (``POOLED``): --workers, by default
    ``workers.default_workers`` (the cores over --threads, less one: 5 on the 12-core Mac at
    2 threads), at most one per unit; 1 runs the stage's serial path in this process. The
    count is recorded in every stamp the stage writes ('workers', beside the thread pins,
    'threads'); it is never partial (``_partial``) and enters no output's content."""
    from apexzero.costate import workers as WK

    w = getattr(args, "workers", None)
    n = WK.default_workers(getattr(args, "threads", None) or 2) if w is None else int(w)
    n = max(1, n if units is None else min(n, int(units)))
    F._prov().workers = n
    return n


def _out(args, group: str) -> Path:
    d = Path(args.out) / group
    d.mkdir(parents=True, exist_ok=True)
    return d


def _x64() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)


def _log(args, stage: str, msg: str) -> None:
    d = Path(args.log_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"costate-{stage}.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def _pre_tag(args, what: str) -> None:
    dev = refuse_after_tag(what, getattr(args, "post_tag_deviation", None),
                           always=args.stage == "prereg", out_root=Path(args.out))
    if dev is not None:
        F._prov().deviation = dev


# ------------------------------------------------------------------------- inputs


def _spread(items: list, n: int | None) -> list:
    return F._limit(items, n)


def _a2(args):
    from apexzero.costate import states as ST

    return _spread(sorted(ST.a2_refs(), key=lambda r: r.key_id), getattr(args, "limit", None))


def multi_basin(ref, root=None) -> bool:
    """A key flagged multi-basin in its file (design Section 3.2: excluded)."""
    from apexzero.costate import keys_root
    from apexzero.fields import keys as K

    return bool(getattr(K.load_key(ref, root=keys_root(root)), "multi_basin", False))


def _analysable(refs) -> tuple[list, list]:
    """(refs without the multi-basin keys, the left-out refs)."""
    keep, out = [], []
    for r in refs:
        (out if multi_basin(r) else keep).append(r)
    return keep, out


def _world_path(args) -> Path:
    return Path(args.out) / "inputs" / "world.npz"


def _load_world(args):
    from apexzero.costate import inputs as IN

    p = _world_path(args)
    if not p.exists():
        raise SystemExit(f"no {p}: run the inputs stage first")
    return IN.load_world(p)


def _t_H(args) -> int | None:
    p = Path(args.out) / "gates" / "k2.json"
    if p.exists():
        return int(json.loads(p.read_text())["horizon"]["t_H"])
    return None


def stage_inputs(args) -> None:
    """T's world, the frozen values and the per-network flags (module docstring)."""
    import numpy as np

    from apexzero.costate import inputs as IN
    from apexzero.costate import keys_root
    from apexzero.costate import prereg as PR
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG
    from apexzero.fields import keys as K
    from apexzero.fields.prereg import sha256_file

    bank, _ = IN.design()
    refs, mb = _analysable(_a2(args))
    t_H = _t_H(args)
    if t_H is None:
        print("warning: no gates/k2.json yet: T is uncut and inputs/summary.json will not "
              "bind (rerun inputs after k2)", flush=True)
    excluded = {"multi_basin": []}
    for r in mb:
        rec = K.load_key(r, root=keys_root())
        lm = bank.stimuli[r.stim_id].landmarks
        n_t = len(ST.t_steps(rec.x, lm, int(r.W), t_H))
        excluded["multi_basin"].append({"key_id": r.key_id, "rows": int(n_t)})
    kds, failed = [], []
    for i, r in enumerate(refs):
        try:
            kds.append(IN.key_data(r, bank, True, t_H))
        except (TG.RepolishFailed, ValueError) as e:   # a listed exclusion
            failed.append({"key_id": r.key_id, "error": f"{type(e).__name__}: {e}"})
        if i % 25 == 0:
            _log(args, "inputs", f"{i + 1}/{len(refs)}")
    world = IN.build_world(kds)
    path = _world_path(args)
    F._stamped_write(path, lambda q: IN.save_world(q, world))
    flags = {}
    for name in args.network or ["G1"]:
        rms, _ = IN.obs_rms_of(name)
        fl = IN.flags(world, rms)
        ob = fl["observability"]
        flags[name] = {"s_unobservable": int(ob.s_unobservable.sum()),
                       "kappa_clipped": int(fl["kappa_clipped"].sum()),
                       "clipped_any": int(fl["clipped_any"].sum()),
                       "s_ratio_exit_max": float(np.max(ob.s_ratio[np.asarray(
                           world.tab["ds_c1"]) > 25], initial=0.0))}
    frozen = PR.frozen_values(world.tab, t_H, partitions=getattr(args, "partitions", None))
    _write(Path(args.out) / "inputs" / "summary.json",
           {"n": world.n, "keys": len(kds), "failed": failed, "t_H": t_H, "frozen": frozen,
            "flags": flags, "excluded": excluded, "world_sha256": sha256_file(path)})


# ------------------------------------------------------------------------- gates


def stage_k0(args) -> None:
    from apexzero.costate import gates as GT
    from apexzero.costate import states as ST

    refs = _a2(args)
    if "n" in (args.kinds or []):
        refs += [r for v in ST.n_refs().values() for r in v]
    res = GT.k0(refs, progress=lambda k: _log(args, "k0", k))
    _write(_out(args, "gates") / "k0.json", res)


def stage_k1(args) -> None:
    from apexzero.costate import gates as GT

    refs = GT.k1_sample()
    if args.n is not None:
        refs = F._limit(refs, args.n)
    rows, pi_rows = [], []
    anchored = [r for r in refs if r.kind == "anchored"][:GT.K1_N_PI]
    for r in refs:
        rows += GT.k1_key(r)
        _log(args, "k1", r.key_id)
    for r in anchored:
        pi_rows += GT.k1_pi(r)
    _write(_out(args, "gates") / "k1.json", {"rows": rows, "pi": pi_rows,
                                             "decision": GT.k1_decision(rows, pi_rows)})


def _sd(args):
    from apexzero.costate import gates as GT

    return GT.lam_sd(_load_world(args).tab)


def stage_k2(args) -> None:
    import numpy as np

    from apexzero.costate import gates as GT
    from apexzero.costate import states as ST
    from apexzero.costate import stream_rng

    world = _load_world(args)
    sd = GT.lam_sd(world.tab)
    res = GT.k2_twins(GT.twin_pairs(), sd, tab=world.tab)
    ext = []
    n = 30 if args.n_extend is None else int(args.n_extend)
    if n:
        pool, _ = _analysable([r for r in ST.a2_refs()
                               if r.W == 400 and ST.strata(r).w_is_cell])
        R = np.array([ST.strata(r).R for r in pool])
        order = np.argsort(R, kind="mergesort")
        rng = stream_rng("costate_horizon")
        bins = np.array_split(order, max(n, 1))
        for b in bins[:n]:
            ref = pool[int(b[rng.integers(len(b))])]
            r = GT.k2_extend(ref)
            if r.get("success"):
                d = GT.k2_compare(r.pop("cs400"), r.pop("cs800"), sd, r.pop("t"))
                r["T_max"] = d.max(0).tolist()
            ext.append(r)
            _log(args, "k2", ref.key_id)
    res["extend"] = ext
    res["extend_ok"] = bool(all(e.get("success") and not e.get("different_basin")
                                and max(e.get("T_max", [0])) <= GT.K2_TOL for e in ext))
    _write(_out(args, "gates") / "k2.json", res)


def stage_k2b(args) -> None:
    import numpy as np

    from apexzero.costate import gates as GT
    from apexzero.costate import states as ST
    from apexzero.costate import stream_rng

    sd = _sd(args)
    pool, _ = _analysable([r for r in ST.a2_refs() if ST.strata(r).w_is_cell
                           and r.kind == "nominal"])
    geo = np.array([(ST.strata(r).R, ST.strata(r).theta_deg) for r in pool])
    uniq, inv = np.unique(geo, axis=0, return_inverse=True)
    stratum = ST.r_tercile(uniq[:, 0], uniq[:, 1]) * 10 + ST.theta_class(uniq[:, 1])
    key = stratum[inv.ravel()]
    rng = stream_rng("costate_horizon", 1)
    n = 30 if args.n is None else int(args.n)
    pick = []
    for i in range(n):
        groups = sorted(set(key.tolist()))
        g = groups[i % len(groups)]
        idx = np.flatnonzero(key == g)
        pick.append(pool[int(idx[rng.integers(len(idx))])])
    keys = []
    for r in dict.fromkeys(pick):
        keys.append(GT.k2b_key(r, sd))
        _log(args, "k2b", r.key_id)
    rule = GT.k2b_rule(keys)
    ok = all(all(v.get("ok") for v in k["severities"].values()) for k in keys)
    _write(_out(args, "gates") / "k2b.json", {"keys": keys, "rule": rule, "pass": bool(ok)})


def _subprocess(args, module: str, argv: list[str], ok: tuple[int, ...] = (0,)) -> int:
    """``python -m module argv`` (src/ on the path) and its exit status; SystemExit unless
    the status is in ``ok`` (Python exits 1 on an uncaught exception too, so 1 is accepted
    only where the caller can tell a fail from a crash)."""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, "-m", module, *argv], cwd=ROOT, env=env,
                       capture_output=True, text=True, check=False)
    if r.returncode not in ok:
        raise SystemExit(f"{module} {argv[:2]} exited {r.returncode}: {r.stderr[-1500:]}")
    return int(r.returncode)


def stage_k3(args) -> None:
    """K3 from its float64 subprocesses' reports. Each report is removed right before its
    subprocess runs, so only this run's are judged: the critic's dry runs must exit 0
    (``critic.main`` returns 0 only: 1 is a crash) and write PREFIX.json; an extractor check
    must rewrite its report and exit 0 on a pass, 1 on a fail (any other pairing is a crash).
    Every exit status is recorded ('returncodes')."""
    from apexzero.costate import critic as CR
    from apexzero.costate import gates as GT

    d = _out(args, "gates") / "k3"
    d.mkdir(parents=True, exist_ok=True)
    size = (["--keys", str(args.n)] if args.n is not None
            else ["--min-states", str(CR.K3_MIN_STATES)])
    rc: dict[str, int] = {}
    for what, extra in (("plants", []), ("random", size)):
        prefix = d / what
        for q in (prefix.with_suffix(".json"), prefix.with_suffix(".npz")):
            q.unlink(missing_ok=True)
        rc[what] = _subprocess(args, "apexzero.costate.critic",
                               ["dry-run", what, "--out", str(prefix), *extra])
        if not prefix.with_suffix(".json").exists():
            raise SystemExit(f"critic dry-run {what} exited 0 without writing {what}.json")
    ext = []
    for heads in ("5", "0"):
        p = d / f"extractor_{heads}.json"
        p.unlink(missing_ok=True)
        rc[f"extractor_{heads}"] = r = _subprocess(
            args, "apexzero.costate.probes", ["extractor-check", "--out", str(p), "--heads",
                                              heads], ok=(0, 1))
        if not p.exists():
            raise SystemExit(f"extractor-check --heads {heads} exited {r} without writing "
                             f"{p.name}")
        e = json.loads(p.read_text())
        if r != (0 if e.get("pass") else 1):            # probes.main's own exit rule
            raise SystemExit(f"extractor-check --heads {heads} exited {r} with a report whose "
                             f"pass is {e.get('pass')!r}: a crash, not its verdict")
        ext.append(e)
    plants = json.loads((d / "plants.json").read_text())
    random = json.loads((d / "random.json").read_text())
    res = GT.k3_judge(plants, random, ext,
                      min_states=CR.K3_MIN_STATES if args.n is None else 0)
    res["reports"] = {"plants": plants, "random": random, "extractor": ext}
    res["returncodes"] = rc
    _write(_out(args, "gates") / "k3.json", res)


def stage_k3b(args) -> None:
    from apexzero.costate import gates as GT
    from apexzero.costate import inputs as IN

    rms, cfg = IN.obs_rms_of("G1")
    cells = K3B_CELLS if args.n is None else K3B_CELLS[: int(args.n)]
    res = GT.k3b(cells, cfg, rms, steps=args.steps)
    res["pass"] = res["decision"]["pass"]
    _write(_out(args, "gates") / "k3b.json", res)


def _probe_world(args, c2_setup: bool = False):
    """(probe world, G1's config, inputs) of K4, K5 and the decoy-k and c2-null stages,
    under G1's obs_rms; with ``c2_setup`` also C2's decision mask (G1's observability and
    K2b's exclusions from gates/k2b.json) and the frozen power floor (inputs/summary.json),
    which C2's primary statistic is scored with. 'inputs' records what the world was built
    from (``prereg.PROBE_WORLD_OUTPUTS``: prereg checks it against the bound files): the sha256
    of inputs/world.npz and, with ``c2_setup``, the power floor and f_D's k read from
    inputs/summary.json and the canonical hash of K2b's exclusions read from gates/k2b.json."""
    from apexzero.costate import gates as GT
    from apexzero.costate import inputs as IN
    from apexzero.costate import prereg as PR
    from apexzero.costate import states as ST
    from apexzero.fields.prereg import sha256_file

    summ = Path(args.out) / "inputs" / "summary.json"
    k2b = Path(args.out) / "gates" / "k2b.json"
    if c2_setup and (not summ.exists() or not k2b.exists()):
        raise SystemExit(f"{getattr(args, 'stage', None) or 'k4'} scores C2's primary "
                         "statistic on C2's decision rows: run inputs and k2b first "
                         "(inputs/summary.json, gates/k2b.json)")
    world = _load_world(args)
    inputs = {"world_sha256": sha256_file(_world_path(args))}
    rms, cfg = IN.obs_rms_of("G1")
    z, _, _ = IN.z_and_jz(world, rms)
    if not c2_setup:
        return GT.probe_world(world.tab, z, cfg), cfg, inputs
    fl = IN.flags(world, rms)
    fr, exc = _json(summ)["frozen"], _json(k2b)["rule"]["excluded"]
    inputs.update(power_floor=fr["power_floor"], decoy_k=(fr.get("decoy_k") or {}).get("k"),
                  k2b_excluded_sha256=PR._canon({p: list(v) for p, v in exc.items()}))
    mask = ST.decision_mask(world.tab, ~fl["observability"].s_unobservable, exc)
    return GT.probe_world(world.tab, z, cfg, mask=mask,
                          power_floor=fr["power_floor"]), cfg, inputs


def stage_k4(args) -> None:
    from apexzero.costate import gates as GT

    pw, cfg, inputs = _probe_world(args, c2_setup=not args.which or "PR-6" in args.which)
    res = GT.k4(pw, which=args.which, readout_net=GT.pr4(cfg, pw) if (
        not args.which or "PR-4" in args.which) else None)
    res["inputs"] = inputs
    _write(_out(args, "gates") / "k4.json", res)


def stage_decoy_k(args) -> None:
    """f_D's k re-derived on this world (``gates.decoy_k``): its pass, the rule's choice equal
    to the code's k and the record's table reproduced, binds at prereg. Its units, one per
    candidate k, world and validation seed (``gates.decoy_k_unit``), run in --workers
    processes (``_workers``); the stamp records the pool that ran them."""
    from apexzero.costate import gates as GT

    pw, _, inputs = _probe_world(args, c2_setup=True)
    res = GT.decoy_k(pw, progress=lambda m: _log(args, "decoy-k", m), workers=_workers(args))
    _workers(args, len(res["candidates"]) * len(GT.DECOY_K_WORLDS) * len(res["seeds"]))
    res["inputs"] = inputs
    _write(_out(args, "gates") / "decoy_k.json", res)


def stage_c2_null(args) -> None:
    """C2's random-only null (``gates.c2_null``; descriptive), frozen at prereg."""
    from apexzero.costate import gates as GT

    pw, cfg, inputs = _probe_world(args, c2_setup=True)
    res = GT.c2_null(pw, cfg, progress=lambda m: _log(args, "c2-null", m))
    res["inputs"] = inputs
    _write(_out(args, "gates") / "c2_null.json", res)


def stage_k5(args) -> None:
    from apexzero.costate import gates as GT

    pw, _, inputs = _probe_world(args)
    res = GT.k5(pw, ceiling=args.ceiling)
    res["inputs"] = inputs
    _write(_out(args, "gates") / "k5.json", res)


def stage_k6(args) -> None:
    from apexzero.costate import gates as GT
    from apexzero.costate import inputs as IN

    world = _load_world(args)
    rms, _ = IN.obs_rms_of("G1")
    fl = IN.flags(world, rms)
    k2b = Path(args.out) / "gates" / "k2b.json"
    exc = json.loads(k2b.read_text())["rule"]["excluded"] if k2b.exists() else None
    summ = Path(args.out) / "inputs" / "summary.json"
    mb = json.loads(summ.read_text()).get("excluded", {}).get("multi_basin") if \
        summ.exists() else None
    _write(_out(args, "gates") / "k6.json",
           GT.k6(world.tab, ~fl["observability"].s_unobservable, fl["kappa_clipped"], exc, mb))


def stage_nulls(args) -> None:
    """C1's input-side nulls per network obs_rms (no network evaluated): the fold-dependent
    ones per held-out fold (``analysis.FOLD_NULLS``, [F, n, 8], cross-fitted: ``nulls``
    module docstring 'Per held-out fold'), the summary with their shapes, the value fits'
    penalties and training R2 (the nested fits' too) and whether every slice is finite (the
    far-geometry lambda's slices may hold rows without a far geometry: recorded, not
    required). One unit per network (``nulls_unit``, its obs_rms read in this process) in
    --workers processes (``_workers``); every array and the summary are written here once all
    have returned."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import inputs as IN
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST
    from apexzero.costate import workers as WK
    from apexzero.fields.prereg import read_stamp, sha256_file

    world = _load_world(args)
    tab = world.tab
    fo = PB.folds(tab)
    w = ST.weights(tab, "key")
    summ = {"networks": {}, "failed": []}
    names = list(dict.fromkeys(AN.network_of(n) for n in (args.network or PR.null_networks())))
    alphas = PB.MLP_ALPHAS if args.mlp_alphas is None else tuple(args.mlp_alphas)
    units = [(name, IN.obs_rms_of(name)[0]) for name in names]     # obs_rms read here
    got = WK.run(nulls_unit, units, _workers(args, len(units)),
                 state=(world, fo, w, alphas, args.mlp_maxiter), names=names)
    for name, res in zip(names, got, strict=True):
        arrays = {k: np.asarray(v) for k, v in res.items() if k != "fits"}
        path = _write_npz(_out(args, "nulls") / f"{PR.net_file(name)}.npz", arrays)
        summ["networks"][name] = {"fits": res["fits"], "sha256": sha256_file(path),
                                  "code_hash": (read_stamp(path) or {}).get("code_hash"),
                                  "finite": {k: bool(np.isfinite(v).all()) for k, v in
                                             arrays.items() if k != "far_geometry"},
                                  "shapes": {k: list(v.shape) for k, v in arrays.items()}}
        if "far_geometry" in arrays:
            fg = arrays["far_geometry"]
            summ["networks"][name]["far_geometry_nonfinite"] = {
                "slices": int((~np.isfinite(fg)).any(2).sum()),
                "out_of_fold": int((~np.isfinite(fg[fo.outer, np.arange(len(w))])).any(1).sum())}
    _write(_out(args, "nulls") / "summary.json", summ)


def nulls_unit(state, unit) -> dict:
    """One network's nulls (``nulls.null_gradients``): ``state`` the stage's (world, folds,
    key weights, MLP penalties, MLP iterations), ``unit`` (name, obs_rms)."""
    from apexzero.costate import nulls as NL

    world, fo, w, alphas, maxiter = state
    return NL.null_gradients(world.keys, world.tab, world.v_star, fo, w, unit[1], world.lam,
                             mlp_alphas=alphas, mlp_maxiter=maxiter,
                             v_star_inf=world.v_star_inf)


def dirty_paths(root: Path = ROOT) -> list[str]:
    r = F._git("status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS, DESIGN_DOC,
               root=root)
    if r.returncode != 0:
        return [f"<git error: {r.stderr.strip()[:200]}>"]
    return [ln[3:].strip() for ln in r.stdout.splitlines() if ln[3:].strip() != PREREG_REL]


def all_networks() -> list[str]:
    from apexzero.costate import analysis as AN

    return list(AN.GENERALISTS) + list(AN.SPECIALISTS)


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def prereg_frozen(out_root: Path, frozen: dict) -> dict:
    """The record's frozen values: the inputs stage's, plus K2b's exclusion rule (and its
    hash), K2's horizon and insurance verdicts, K4's band-edge notes (``gates.band_notes``,
    which the analysis prints beside a C2 decision near the band) and C2's random-only null
    (``gates.c2_null_frozen``, printed beside each primary C2), read from the bound gate
    outputs."""
    from apexzero.costate import gates as GT
    from apexzero.costate import prereg as PR

    out = dict(frozen)
    out["k4_band_notes"] = GT.band_notes(_json(Path(out_root) / "gates" / "k4.json"))
    out["c2_null"] = GT.c2_null_frozen(_json(Path(out_root) / "gates" / "c2_null.json"))
    k2b = _json(Path(out_root) / "gates" / "k2b.json")["rule"]
    out["k2b_excluded"] = {p: list(v) for p, v in k2b["excluded"].items()}
    out["k2b_excluded_sha256"] = PR._canon(out["k2b_excluded"])
    k2 = _json(Path(out_root) / "gates" / "k2.json")
    out["k2"] = {"t_H": k2["horizon"]["t_H"], "drops_T": k2.get("drops_T"),
                 "fallback_keys": k2.get("fallback_keys"), "extend_ok": k2.get("extend_ok"),
                 "different_basin": [e["key_id"] for e in k2.get("extend", [])
                                     if e.get("different_basin")],
                 "extend_T_max": max((max(e.get("T_max", [0.0])) for e in k2.get("extend", [])),
                                     default=None)}
    return out


def bound_env_problems(out_root: Path, rels, env_now: dict) -> list[str]:
    """The bound pre-tag outputs whose stamps record other Python or package versions than
    ``env_now`` (``F.env_problems``), or none: the record takes this run's environment, which
    the freeze check and the confirmatory input check hold every later stage to, so a package
    change after a stage ran is refused here, not at the last post-tag stage."""
    from apexzero.fields.prereg import read_stamp

    out = []
    for rel in rels:
        p = Path(out_root) / rel
        st = read_stamp(p) if p.exists() else None
        if st is None:
            continue                       # reported by bind_outputs and bind_arrays
        if not st.get("environment"):
            out.append(f"{rel}'s stamp records no environment")
            continue
        env = F.env_problems(st["environment"], env_now)
        if env:
            out.append(f"{rel} was computed with other package versions than this run's "
                       f"({env[:3]}): rerun it")
    return out


def stage_prereg(args) -> None:
    """The record (``costate.prereg``), from a clean tree with every bound output passing,
    stamped by this code under this run's package versions."""
    from apexzero.costate import analysis as AN
    from apexzero.costate import keys_root
    from apexzero.costate import prereg as PR

    dirty = dirty_paths()
    if dirty:
        raise SystemExit(f"prereg refuses a dirty tree: {dirty[:8]}")
    probs = [str(p) for p in frozen_steps_problems()]
    code = PR.code_hash(ROOT)
    try:
        if code != PR.tree_code_hash(ROOT, "HEAD"):
            probs.append("the code on disk is not HEAD's")
    except RuntimeError as e:
        probs.append(f"git: {e}")
    env_now = F.environment()
    outputs, p2 = PR.bind_outputs(Path(args.out), code)
    arrays, p2b = PR.bind_arrays(Path(args.out), code)
    p2 += bound_env_problems(Path(args.out), [*outputs, *arrays], env_now)
    pins, p3 = PR.network_pins(all_networks(), ROOT)
    # the deferred C1-dev's snapshots (``prereg.DEFERRED``): pinned, not pre-registered
    snaps, p4 = {}, []
    for n in AN.DECISION_NETWORKS:
        snaps[n], q = PR.snapshot_pins(n, ROOT)
        p4 += q
    c3w, p5 = PR.c3w_inputs(AN.GENERALISTS, ROOT, completion, competence)
    for n, v in c3w.items():
        try:
            v["overlap"], v["overlap_file"], v["overlap_sha256"] = overlap(n)
        except (OSError, ValueError, KeyError) as e:
            v["overlap"] = None
            v["overlap_error"] = str(e)
    probs += p2 + p2b + p3 + p4 + p5
    world = _load_world(args)
    summ = _json(Path(args.out) / "inputs" / "summary.json")
    if probs:
        raise SystemExit("prereg refuses: " + "; ".join(probs))
    t_pairs = list(zip(world.tab["key_id"].tolist(), world.tab["t"].tolist(), strict=True))
    extra = {"prereg_base_commit": F._head(), "networks": pins, "snapshots": snaps,
             "outputs": {**outputs, **arrays}, "environment": env_now,
             "c3w_inputs": c3w, "design_hash": PR.stimulus_design_hash(),
             "excluded": summ.get("excluded"),
             "steps": {"step1": {"tag": STEP1_TAG, "object": F.STEP1_TAG_OBJECT},
                       "step2": {"tag": STEP2_TAG,
                                 "object": (F._tag_object(STEP2_TAG) or [None])[0]}}}
    rec = PR.build_record(ROOT, PR.key_sets(), keys_root(), t_pairs,
                          prereg_frozen(Path(args.out), summ["frozen"]), extra)
    path = PR.save(rec)
    print(f"wrote {path}; next, on main in this checkout: git add {PREREG_REL} && git commit; "
          f"git tag -a {TAG}; git push {REMOTE} main {TAG}; then run every post-tag stage in "
          "this checkout (module docstring 'Where it runs')")


# ------------------------------------------------------------------------- post-tag


def _post_tag(args, what: str) -> dict:
    """The freeze check, then trained networks allowed in this process (module docstring)."""
    from apexzero.costate import critic as CR
    from apexzero.costate import probes as PB
    from apexzero.fields import network as NW

    if getattr(args, "limit", None) is not None and Path(args.out).resolve() == OUT.resolve():
        raise SystemExit("--limit writes a partial run: give --out outside outputs/costate")
    bad = import_problems()
    if bad:
        raise SystemExit(f"refusing to run {what}: {'; '.join(bad)}")
    dev = require_tag(what, getattr(args, "post_tag_deviation", None))
    rec = tagged_record()
    if rec is None or not rec.get("networks"):
        raise SystemExit("refusing: no record at the tag, or it pins no network")
    p = F._prov()
    p.deviation, p.tag, p.check_imports = dev, F.tag_identity(TAG), True
    _x64()
    CR.unlock_trained()
    PB.unlock_trained()
    NW.unlock_trained()
    return rec


def load_agent(name: str, pin: dict, root: Path = ROOT):
    """A network's snapshot from its recorded checkpoint (sha256 checked before loading)."""
    from apexzero.fields import prereg as FP
    from apexzero.fields import stages as S
    from apexzero.interp import agents

    path = Path(root) / pin["path"]
    if FP.sha256_file(path) != pin["sha256"]:
        raise SystemExit(f"{name}: {pin['path']} is not the recorded checkpoint")
    ag = agents.load_agent(path, S.network_agent_id(name))
    if int(ag.timesteps) != int(pin["timesteps"]):
        raise SystemExit(f"{name}: timesteps {ag.timesteps}, recorded {pin['timesteps']}")
    return ag


def _load_world_frozen(args, rec: dict):
    """T's world for a post-tag stage: the bound inputs/world.npz (its sha256 against the
    record), T's (key id, t) list against the record's (``prereg.frozen_t``), and T's
    geometries and partition 0's folds against the frozen ones."""
    import numpy as np

    from apexzero.costate import prereg as PR
    from apexzero.costate import states as ST
    from apexzero.fields.prereg import sha256_file

    p = _world_path(args)
    if not p.exists():
        raise SystemExit(f"no {p}: the record binds T's world")
    want = (rec.get("outputs") or {}).get("inputs/world.npz", {}).get("sha256")
    if not want or sha256_file(p) != want:
        raise SystemExit(f"refusing: {p} is not the world the record binds")
    world = _load_world(args)
    pairs = sorted(zip(map(str, world.tab["key_id"]), map(int, world.tab["t"]), strict=True))
    if pairs != sorted(PR.frozen_t(rec)):
        raise SystemExit("refusing: T's (key, t) list differs from the record's")
    uniq = np.unique(ST.geometry_keys(world.tab), axis=0)
    fr = rec.get("frozen") or {}
    if (uniq.tolist() != fr.get("geometries")
            or ST.geometry_folds(uniq[:, 0], uniq[:, 1], 0).tolist() != fr.get("fold_partition0")):
        raise SystemExit("refusing: T's geometries or partition 0's folds differ from the frozen "
                         "ones")
    return world


def activations_of(agent, world, n_random: int = 5) -> dict:
    """The actor's h1-h3 and m, the critic's h1-h3, V and heads at T (z under the agent's own
    obs_rms), and the random-init actors' h3 and m on the same z (seeds 0 to ``n_random`` -
    1: the stage gives a decision network the control pool, ``probes.N_POOL``, and every
    other network the 5 controls, ``probes.N_RANDOM_ACTORS``)."""
    import numpy as np

    from apexzero.costate import inputs as IN
    from apexzero.costate import probes as PB
    from apexzero.fields import network as NW

    rms = NW.upcast32(agent.obs_rms)
    z, cl, _ = IN.z_and_jz(world, rms)
    o = np.clip(z, -10.0, 10.0)
    actor, critic = PB.agent_nets(agent)
    ha, hc = PB.hidden(actor, o), PB.hidden(critic, o)
    out = {"z": z, "clipped": cl, "m": ha["m"], "V": hc["V"]}
    for k in ("h1", "h2", "h3"):
        out[f"actor_{k}"], out[f"critic_{k}"] = ha[k], hc[k]
    if hc.heads is not None:
        out["heads"], out["head_units"] = hc.heads, hc.u
    for s in range(n_random):
        hr = PB.hidden(PB.random_actor(agent.cfg, s), o)
        out[f"random{s}_h3"], out[f"random{s}_m"] = hr["h3"], hr["m"]
    return out


def critic_of(agent, world, seeds=None, bank=None) -> dict:
    """The trained critic's pullbacks (V and heads) at T, its uniform inside-widening
    derivative (``inputs.width_derivative``: C3-w's x, with ``bank``) and the random-init
    critics' main-V pullbacks (Section 5's 20 seeds, the agent's obs_rms and sigma_R)."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN
    from apexzero.costate import nulls as NL
    from apexzero.fields import network as NW

    net = CR.trained_critic(agent, getattr(agent, "checkpoint", ""))
    n = world.n
    E = net.n_heads
    g = np.full((n, 1 + E, 8), np.nan)
    pi = np.full((n, 1 + E, 2), np.nan)
    gh = np.full((n, 1 + E, 8), np.nan)
    y = np.full((n, 1 + E), np.nan)
    acc = np.full((n, 1 + E, 2), np.nan)
    for k in world.keys:
        pb = CR.pullback(net, None, k.inp, k.oj)
        g[k.rows], pi[k.rows], gh[k.rows] = pb.g, pb.pi, pb.g_held
        y[k.rows], acc[k.rows] = pb.value, pb.accel_sens
    seeds = CR.RANDOM_SEEDS if seeds is None else seeds
    rg = np.stack([NL.gradients(c, world.keys, n).g for c in
                   CR.random_critics(agent.cfg, net.obs_rms, net.sigma, seeds)])
    out = {"g": g, "pi": pi, "g_held": gh, "value": y, "accel_sens": acc, "random_g": rg,
           "sigma": np.float64(net.sigma)}
    if bank is not None:
        dw = np.full(n, np.nan)
        tab = world.tab
        for k in world.keys:
            i0 = int(k.rows[0])
            rd = NW.road(bank, int(tab["stim_id"][i0]))
            dw[k.rows] = IN.width_derivative(net, rd, k.inp, int(tab["direction"][i0]))
        out["dV_dwin"] = dw
    return out


def stage_activations(args) -> None:
    """Each network's activations (``activations_of``): a decision network's with the
    control pool's random-init actors (``probes.POOL_SEEDS``, which its primary C2 and C2-sw
    are scored against), every other network's with the 5 controls, stored as before (one
    compressed npz per network). Measured at T's 9,887 rows on a random-init stand-in: the
    pool's h3 is 20 x 9,887 x 256 float64 (405 MB; 575 MB with every array), the npz 535 MB
    against 242 MB with the 5 controls (4.8 GB over the 9 decision networks, 2.6 GB more),
    the process' peak resident memory 1.4 GB, about 16 s per network."""
    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB

    rec = _post_tag(args, "activations")
    world = _load_world_frozen(args, rec)
    decided = {PR.net_file(a) for a in AN.DECISION_NETWORKS}
    for name in args.network or list(rec["networks"]):
        ag = load_agent(name, rec["networks"][name])
        n = PB.N_POOL if PR.net_file(name) in decided else PB.N_RANDOM_ACTORS
        _write_npz(_out(args, "post") / "activations" / f"{PR.net_file(name)}.npz",
                   activations_of(ag, world, n_random=n))


def d_rows(world):
    """D (design Section 3.2 'D (secondary)', the decision states): T's t = 0 rows, one per
    A2 key in T (the 4 multi-basin keys are left out of T, so 477 of the 481)."""
    import numpy as np

    return np.flatnonzero(np.asarray(world.tab["t"]) == 0)


def k3_trained(agent, world, name: str) -> dict:
    """K3's checks on a trained network after the tag (Section 7 K3: 'trained networks after
    it, on D' and 'After the tag'): ``gates.k3_trained`` at D's raw observations (the bound
    world's ``ObsJac.ob``, the observations every pullback reads) with ``name``'s head rule
    (``gates.expected_heads``). A network that fails is treated as missing in every decision
    it enters (``analysis.k3_trained_failed``)."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import gates as GT

    ob = np.full((world.n, CR.OBS_DIM), np.nan)
    for k in world.keys:
        ob[k.rows] = k.oj.ob
    rows = d_rows(world)
    have = rows[np.isfinite(ob[rows]).all(axis=1)]
    out = GT.k3_trained(agent, ob[have], GT.expected_heads(name))
    out.update(rows="D: T's t = 0 rows", n_d=len(rows),
               n_d_without_observation=int(len(rows) - len(have)))
    return out


def stage_critic(args) -> None:
    """Per network (every one the record pins): K3's after-the-tag checks on D
    (``k3_trained``, post/critic/<net>.k3.json), then its critic's arrays (``critic_of``,
    post/critic/<net>.npz, with the check's verdict as 'k3_trained_pass')."""
    import numpy as np

    from apexzero.costate import inputs as IN
    from apexzero.costate import prereg as PR

    rec = _post_tag(args, "critic gradients")
    world = _load_world_frozen(args, rec)
    bank, _ = IN.design()
    for name in args.network or list(rec["networks"]):
        ag = load_agent(name, rec["networks"][name])
        f = PR.net_file(name)
        k3 = k3_trained(ag, world, name)
        _write(_out(args, "post") / "critic" / f"{f}.k3.json", k3)
        if not k3["decision"]["pass"]:
            print(f"{name}: {k3['decision']['reason']} (descriptive, missing from every "
                  "decision)", flush=True)
        out = critic_of(ag, world, bank=bank)
        out["k3_trained_pass"] = np.bool_(k3["decision"]["pass"])
        _write_npz(_out(args, "post") / "critic" / f"{f}.npz", out)


def own_of(agent, world, every: int = OWN_EVERY, bank=None, net=None, keys=None,
           v_steps: int | None = None) -> dict:
    """lambda^own and pi^own at T's interior states every ``every``-th plan step (restarts
    from x_t with prev_action a*_{t-1}, H = W - t; Section 3.3): on the manifold, with the
    key row's own M_x and M_a (``own.own_costate``'s maps0), the held forms beside them.

    Value accuracy's target (Section 3.3 'Value accuracy'): from the same restarts and from D
    (every key's t = 0 state with its reset inputs, prev_action = accel = 0), the
    deterministic policy's own discounted return (``own.own_return``: delta = 0, s_stop =
    +inf, up to ``v_steps`` = ``own.V_STEPS`` = 1,000 steps or to the strip's end, not W),
    per row: v_own_det (J), v_own_H, v_own_tail (gamma^H), v_own_valid (no termination
    within H; a crash is recorded, 'v_own_terminated', and dropped), v_own_strip_end, the
    descriptive horizon-corrected v_own_inf, and 'v_rows' (the rows asked for). N's states
    and the D keys' own-run states (Section 3.3 (ii), (iii)) are not built here.

    ``net`` replaces the agent's actor (a random-init or planted control in tests; the
    agent's own, marked trained, by default); ``keys`` restricts the rows to those key ids."""
    import numpy as np

    from apexzero.costate import inputs as IN
    from apexzero.costate import own as OW
    from apexzero.fields import network as NW

    if bank is None:
        bank, _ = IN.design()
    net = NW.agent_net(agent) if net is None else net
    v_steps = OW.V_STEPS if v_steps is None else int(v_steps)
    tab = world.tab
    t = np.asarray(tab["t"])
    pick = (np.ones(world.n, bool) if keys is None
            else np.isin(np.asarray(tab["key_id"]).astype(str), [str(k) for k in keys]))
    lam_rows = np.asarray(tab["interior"], bool) & (t % every == 0) & (t > 0) & pick
    v_rows = (lam_rows | (t == 0)) & pick
    rows = np.flatnonzero(lam_rows)
    lam, lam_h = np.full((world.n, 8), np.nan), np.full((world.n, 8), np.nan)
    pi, pi_h = np.full((world.n, 2), np.nan), np.full((world.n, 2), np.nan)
    valid = np.zeros(world.n, bool)
    by_key = {k.key_id: k for k in world.keys}
    for i in rows:
        k = by_key[str(tab["key_id"][i])]
        j = int(np.flatnonzero(k.rows == i)[0])
        inp = k.inp.take([j])
        W, ti = int(tab["W"][i]), int(tab["t"][i])
        li = NW.LoopIn(inp.x[0], inp.prev_action[0], inp.accel[0], np.zeros((W - ti, 2)),
                       np.int32(0), np.float64(np.inf), NW.ctx64(), NW.context_channels())
        rd = NW.road(bank, int(tab["stim_id"][i]))
        oc = OW.own_costate(net, rd.base, li, W - ti, W - ti, reset=bool(k.oj.reset[j]),
                            maps0=(k.oj.Mx[j], k.oj.Ma[j]))
        lam[i], pi[i], valid[i] = oc.lam[0], oc.pi, oc.valid_all
        lam_h[i], pi_h[i] = oc.lam_held[0], oc.pi_held
    # value accuracy's own returns (Section 3.3): T's interior restarts and D
    vo = {"v_own_det": np.full(world.n, np.nan), "v_own_inf": np.full(world.n, np.nan),
          "v_own_tail": np.full(world.n, np.nan), "v_own_H": np.zeros(world.n, int),
          "v_own_valid": np.zeros(world.n, bool), "v_own_terminated": np.zeros(world.n, bool),
          "v_own_strip_end": np.zeros(world.n, bool)}
    li0 = NW.LoopIn(np.zeros(8), np.zeros(2), np.zeros(2), np.zeros((v_steps, 2)), np.int32(0),
                    np.float64(np.inf), NW.ctx64(), NW.context_channels())
    by_stim: dict = {}                 # one batched loop per road (keys sharing a stimulus)
    for k in world.keys:
        js = np.flatnonzero(v_rows[k.rows])
        if len(js):
            inp = k.inp.take(js)
            by_stim.setdefault(int(tab["stim_id"][int(k.rows[js[0]])]), []).append(
                (np.asarray(k.rows)[js], inp.x, inp.prev_action, inp.accel))
    for sid, parts in by_stim.items():
        idx = np.concatenate([q[0] for q in parts])
        r = OW.own_return(net, NW.road(bank, sid).base, li0,
                          *(np.concatenate([q[i] for q in parts]) for i in (1, 2, 3)), v_steps)
        for name, v in (("v_own_det", r.J), ("v_own_inf", r.J_inf), ("v_own_tail", r.tail),
                        ("v_own_H", r.H), ("v_own_valid", r.valid),
                        ("v_own_terminated", r.terminated), ("v_own_strip_end", r.strip_end)):
            vo[name][idx] = v
    return {"lam_own": lam, "pi_own": pi, "lam_own_held": lam_h, "pi_own_held": pi_h,
            "valid": valid, "rows": rows, "v_rows": v_rows, "v_steps": np.int64(v_steps), **vo}


def k3b_trained(agent, bank=None, cells=K3B_TRAINED_CELLS, steps: int | None = None) -> dict:
    """K3b's trained analogue (Section 7: after the tag each network runs one; a network
    that fails has descriptive C1-own): ``own.k3b_start`` on the network's own policy from
    A2 cell starts, closed loops of ``steps`` (default ``own.K3B_STEPS``, K3b's), with the
    on-manifold check, pooled by ``own.k3b_decision``."""
    from apexzero.costate import inputs as IN
    from apexzero.costate import own as OW
    from apexzero.fields import network as NW

    steps = OW.K3B_STEPS if steps is None else int(steps)
    bank_, cells_ = IN.design()
    bank = bank_ if bank is None else bank
    net = NW.agent_net(agent)
    starts = []
    for cid in cells:
        st = NW.cell_start(cells_[int(cid)], 0, bank)
        rd = NW.road(bank, st.stim_id)
        starts.append(OW.k3b_start(net, rd.base, NW.loop_inputs(st, steps), steps,
                                   name=f"trained@{cid}"))
    return {"starts": [s.record() for s in starts], "decision": OW.k3b_decision(starts)}


def stage_own(args) -> None:
    """Each network's K3b trained analogue and own costate (``own_unit``): one unit per
    network, its checkpoint loaded and its pin checked in this process (``load_agent``), in
    --workers processes (``_workers``); the files are written here once all have returned."""
    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR
    from apexzero.costate import workers as WK

    rec = _post_tag(args, "own costate")
    world = _load_world_frozen(args, rec)
    names = args.network or [AN.network_of(a) for a in AN.DECISION_NETWORKS]
    units = [load_agent(name, rec["networks"][name]) for name in names]
    got = WK.run(own_unit, units, _workers(args, len(units)), state=world, names=names)
    for name, (k3, out) in zip(names, got, strict=True):
        _write(_out(args, "post") / "own" / f"{PR.net_file(name)}.k3b.json", k3)
        _write_npz(_out(args, "post") / "own" / f"{PR.net_file(name)}.npz", out)


def own_unit(world, agent) -> tuple[dict, dict]:
    """One network's own stage: K3b's trained analogue (``k3b_trained``) and its own costate
    and returns on ``world`` (``own_of``), with 'k3b_pass'."""
    import numpy as np

    k3 = k3b_trained(agent)
    out = own_of(agent, world)
    out["k3b_pass"] = np.bool_(k3["decision"]["pass"])
    return k3, out


# ------------------------------------------------------------------------- variants


VARIANT_SCHEMES = ("geometry", "state")    # Section 4 'Weights' sensitivities (i), (ii)


def variant_specs(tab, frozen: dict, names=None) -> dict[str, dict]:
    """The pre-registered sensitivity and robustness variants of the decision statistics
    (Sections 4 and 6): 'primary' (partition 0, key weights, every row); partitions 1-10;
    geometry and state weights; the 11 geometries with 3 keys or fewer dropped; the kappa
    sensitivity set dropped (frozen 'kappa_set'); the braking and non-braking keys'
    sub-scores. Each: {partition, scheme, keep [n] or None}."""
    import numpy as np

    from apexzero.costate import states as ST

    specs = {"primary": {"partition": 0, "scheme": "key", "keep": None}}
    for p in range(1, ST.N_PARTITIONS):
        specs[f"partition{p}"] = {"partition": p, "scheme": "key", "keep": None}
    for sc in VARIANT_SCHEMES:
        specs[f"weights:{sc}"] = {"partition": 0, "scheme": sc, "keep": None}
    specs["drop_small"] = {"partition": 0, "scheme": "key", "keep": ~ST.small_geometry(tab)}
    kappa = {tuple(map(float, g)) for g in frozen.get("kappa_set", [])}
    G = ST.geometry_keys(tab)
    specs["drop_kappa_set"] = {"partition": 0, "scheme": "key",
                               "keep": np.array([tuple(g) not in kappa for g in G.tolist()])}
    brk = np.asarray(tab["braking_key"], bool)
    specs["braking_keys"] = {"partition": 0, "scheme": "key", "keep": brk}
    specs["non_braking_keys"] = {"partition": 0, "scheme": "key", "keep": ~brk}
    if names is not None:
        specs = {k: v for k, v in specs.items() if k in set(names) | {"primary"}}
    return specs


def _subclusters(tab, rows, seed: int = 0):
    """(cluster index [n] re-indexed over the geometries present in ``rows``, their number,
    bootstrap counts over them): a subset's bootstrap resamples its own geometries only."""
    import numpy as np

    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST

    g = ST.geometry_index(tab)
    present = np.unique(g[np.asarray(rows, bool)])
    remap = np.full(int(g.max()) + 1, -1)
    remap[present] = np.arange(len(present))
    cl = np.where(remap[g] >= 0, remap[g], 0)
    G = max(len(present), 1)
    return cl, G, PB.cluster_counts(G, rng=PB.boot_rng(900 + int(seed)))


def _winsorised(tab, frozen: dict, names):
    import numpy as np

    from apexzero.costate import states as ST

    return np.stack([ST.winsorise(np.asarray(tab[f"lam_{c}"], float),
                                  frozen["winsor_bounds"][f"lam_{c}"]) for c in names], 1)


def _bounds(frozen: dict, names):
    import numpy as np

    return np.array([frozen["winsor_bounds"][f"lam_{c}"] for c in names], float)


def probes_of(acts: dict, world, frozen: dict, rows_c2, rows_sw, valid=None,
              variants=None, strata=None, kappa_clipped=None) -> dict:
    """C2 and C2-sw of one network from its activations (``costate.analysis``), with
    ``valid`` [n, 4] each K_P component's decision rows (``states.decision_mask``), the
    pre-registered variants (``variant_specs``; the decoy residuals re-cross-fitted on each
    partition's folds at f_D's frozen k, ``prereg.decoy_k``, or ``states.KNN_K`` where the
    frozen values lack it, recorded under 'decoy_k'; weights renormalised on each variant's
    stratum), and C2's and C2-sw's descriptive strata tables (``by_stratum``:
    ``analysis.STRATA``, or ``strata`` {'C2': names, 'C2-sw': names}; partition 0, key
    weights renormalised on each stratum; a stratum whose
    rows a variant already scored reuses that record; ``kappa_clipped`` [n] the network's own
    kappa flags, ``inputs.flags``, for the kappa-clipped strata, 'not testable' without
    them). The primary C2 and C2-sw record 'not testable' only for ``probes.NotTestable`` (no
    component passes the power floor or keeps a scored row; no row): any other error
    propagates. The variants, the strata tables and the robustness summary are descriptive
    (``analysis.guarded``: an error is recorded in place of the record). Only the primary C2
    and C2-sw enter a decision, so only they are scored against the control pool
    (``probes.POOL_SEEDS``, 'random0_h3' to 'random19_h3' in ``acts``) with the two-level CI
    (``probes.pool_draws``); every other record keeps the 5 controls and the geometry-only CI
    (``analysis`` module docstring 'C2's control pool'), and each says which ('controls'),
    the 'not testable' ones included (a strata row with no row or no per-network column); a
    contained error records the error alone. A strata table's primary row is the decision's
    own record."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    tab = world.tab
    lam = _winsorised(tab, frozen, TG.K_P)
    five = tuple(range(PB.N_RANDOM_ACTORS))
    rnd = [acts[f"random{s}_h3"] for s in five]
    pool = [acts[f"random{s}_h3"] for s in PB.POOL_SEEDS]     # the decisions' controls
    z = np.clip(acts["z"], -10, 10)
    sig = ST.winsorise(np.asarray(tab["sigma_pedal"], float),
                       frozen["winsor_bounds"]["sigma_pedal"])
    specs = variant_specs(tab, frozen, variants)
    # f_D's k: the record's frozen one ('decoy_k'), else the code's (states.KNN_K)
    k_d = PR.decoy_k(frozen)
    out: dict = {"sensitivity": {}, "decoy_k": {
        "k": ST.KNN_K if k_d is None else k_d, "source": "constant" if k_d is None else "frozen"}}
    res_by_p: dict = {}
    fo_by_p: dict = {}

    def residuals(p: int, fo):
        if p not in res_by_p:
            res_by_p[p] = ST.cross_fit(ST.decoy_d1(tab), lam, fo.outer, ST.weights(tab, "key"),
                                       k=out["decoy_k"]["k"])
        return res_by_p[p]

    def folds(p: int):
        if p not in fo_by_p:
            fo_by_p[p] = PB.folds(tab, p)
        return fo_by_p[p]

    def refs(primary: bool) -> dict:
        """The controls (and, for a decision, the two-level CI's draws) of one record."""
        if primary:
            return {"random_h3": pool, "pool": PB.pool_draws(len(pool)),
                    "controls": PB.POOL_SEEDS}
        return {"random_h3": rnd, "pool": None, "controls": five}

    def c2_on(r2, fo, res, scheme="key", primary=False) -> dict:
        w2 = ST.stratum_weights(tab, r2, scheme)
        rf = refs(primary)
        try:
            return AN.c2_network(acts["actor_h3"], rf["random_h3"], z, acts["m"], res.by_fold,
                                 fo, None, tab["direction"], TG.K_P, frozen["power_floor"],
                                 r2, weights=w2[r2], mask=valid, pool=rf["pool"],
                                 controls=rf["controls"])
        except PB.NotTestable as e:
            return {"decision": "not testable", "reason": str(e),
                    "controls": AN.controls_label(rf["controls"], primary)}

    def sw_on(rs, fo, scheme="key", primary=False) -> dict:
        ws = ST.stratum_weights(tab, rs, scheme)
        rf = refs(primary)
        try:
            return AN.c2_sw_network(acts["actor_h3"], rf["random_h3"], z, sig, acts["m"][:, 1],
                                    fo, ws[rs], rs, pool=rf["pool"], controls=rf["controls"])
        except PB.NotTestable as e:
            return {"decision": "not testable", "reason": str(e),
                    "controls": AN.controls_label(rf["controls"], primary)}

    def on(h: str, sp: dict, rows, primary: bool = False) -> dict:
        fo = folds(sp["partition"])
        if h == "C2":
            return c2_on(rows, fo, residuals(sp["partition"], fo), sp["scheme"], primary)
        return sw_on(rows, fo, sp["scheme"], primary)

    # (hypothesis, rows) -> a 5-control record, partition 0 under key weights (the decisions,
    # against the pool, are reused by their own strata row only)
    done: dict = {}
    for vname, sp in specs.items():
        keep = np.ones(world.n, bool) if sp["keep"] is None else np.asarray(sp["keep"], bool)
        rows = {"C2": np.asarray(rows_c2, bool) & keep, "C2-sw": np.asarray(rows_sw, bool) & keep}
        if vname == "primary":            # the decisions: the control pool; they raise
            out.update({h: on(h, sp, r, primary=True) for h, r in rows.items()})
            continue
        rec = {h: AN.guarded(on, h, sp, r) for h, r in rows.items()}
        out["sensitivity"][vname] = rec
        if sp["partition"] == 0 and sp["scheme"] == "key":
            for h, r in rows.items():
                done.setdefault((h, r.tobytes()), rec[h])
    # the descriptive strata tables (Sections 6, 12 and 13)
    decided = {"C2": np.asarray(rows_c2, bool), "C2-sw": np.asarray(rows_sw, bool)}
    tab_s = AN.strata_tab(tab, None if kappa_clipped is None
                          else {"kappa_clipped": kappa_clipped})
    canon = {"partition": 0, "scheme": "key"}

    def row(h: str, sname: str, rows) -> dict:
        # never the primary row (its stratum needs no per-network column, and it reuses out[h])
        lab = {"controls": AN.controls_label(five, False)}
        if rows is None:
            return {**AN.stratum_record(h, {"reason": "no per-network column " + ", ".join(
                AN.missing_columns(tab_s, sname))}, None, tab_s), **lab}
        primary = bool(sname == AN.PRIMARY_STRATUM[h] and np.array_equal(rows, decided[h]))
        rec_s = out[h] if primary else done.get((h, rows.tobytes()))
        if rec_s is None and not rows.any():
            rec_s = {"decision": "not testable", "reason": "no row", **lab}
        if rec_s is None:
            rec_s = done[(h, rows.tobytes())] = AN.guarded(on, h, canon, rows)
        return AN.stratum_record(h, rec_s, rows, tab_s, primary=primary)

    def table(h: str) -> dict:
        names = AN.STRATA[h] if strata is None else strata.get(h, ())
        return {sname: AN.guarded(row, h, sname, rows)
                for sname, rows in AN.strata_rows(tab_s, names).items()}

    out["by_stratum"] = {h: AN.guarded(table, h) for h in ("C2", "C2-sw")}
    parts = [out["sensitivity"][k]["C2"].get("stat") for k in out["sensitivity"]
             if k.startswith("partition")]
    parts = [v for v in parts if v is not None]
    if parts:
        out["robustness_C2"] = AN.guarded(lambda: {
            "stat": parts, "min": min(parts), "max": max(parts), "sd": float(np.std(parts)),
            "controls": AN.controls_label(five, False)})
    return out


def pool_problems(out_root: Path, names) -> list[str]:
    """Why the probes stage cannot score these networks' primary C2 and C2-sw against the
    control pool: an activations file missing, or without every pool actor's h3
    ('random0_h3' to 'random19_h3', ``probes.POOL_SEEDS``; an activations file written for
    the 5 controls). Checked before any network is probed."""
    import numpy as np

    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB

    probs = []
    for name in names:
        p = Path(out_root) / "post" / "activations" / f"{PR.net_file(name)}.npz"
        if not p.exists():
            probs.append(f"{p.relative_to(out_root)} is missing")
            continue
        with np.load(p) as d:
            lacks = [s for s in PB.POOL_SEEDS if f"random{s}_h3" not in d.files]
        if lacks:
            probs.append(f"{p.relative_to(out_root)} lacks the control pool's h3 of seeds "
                         f"{lacks} (run activations again)")
    return probs


def stage_probes(args) -> None:
    """Each decision network's C2 and C2-sw (``probes_unit``): one unit per network, its
    checkpoint loaded and its pin checked in this process for its decision mask and kappa
    flags, in --workers processes (``_workers``), each loading its own activations file; the
    files are written here once all have returned."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import inputs as IN
    from apexzero.costate import prereg as PR
    from apexzero.costate import states as ST
    from apexzero.costate import workers as WK
    from apexzero.fields import network as NW

    rec = _post_tag(args, "probes")
    world = _load_world_frozen(args, rec)
    tab = world.tab
    inner = np.asarray(tab["interior"], bool)
    sw = np.asarray(tab[ST.SW_STRATUM], bool)
    names = args.network or [AN.network_of(a) for a in AN.DECISION_NETWORKS]
    probs = pool_problems(Path(args.out), names)
    if probs:
        raise SystemExit("the probes stage scores each primary C2 and C2-sw against the "
                         "control pool: " + "; ".join(probs))
    units = []
    for name in names:
        p = Path(args.out) / "post" / "activations" / f"{PR.net_file(name)}.npz"
        ag = load_agent(name, rec["networks"][name])
        fl = IN.flags(world, NW.upcast32(ag.obs_rms))
        valid = ST.decision_mask(tab, ~fl["observability"].s_unobservable,
                                 rec["frozen"].get("k2b_excluded"))
        units.append((str(p), valid, fl["kappa_clipped"]))
    got = WK.run(probes_unit, units, _workers(args, len(units)),
                 state=(_table_world(world), rec["frozen"], inner, sw), names=names)
    for name, out in zip(names, got, strict=True):
        _write(_out(args, "post") / "probes" / f"{PR.net_file(name)}.json", out)


def _table_world(world):
    """``world`` without its per-key inputs (``probes_of`` reads T's table only): what the
    probes stage hands its workers."""
    import dataclasses

    return dataclasses.replace(world, keys=[]) if dataclasses.is_dataclass(world) else world


def probes_unit(state, unit) -> dict:
    """One network's C2 and C2-sw with their variants and strata tables (``probes_of``):
    ``state`` the stage's (world, frozen values, C2's and C2-sw's rows), ``unit`` (its
    activations file, its decision mask, its kappa flags)."""
    import numpy as np

    world, frozen, inner, sw = state
    path, valid, kappa = unit
    with np.load(path) as d:
        acts = {k: d[k] for k in d.files}
    return probes_of(acts, world, frozen, inner, sw, valid, kappa_clipped=kappa)


def _summary(v) -> dict | None:
    import numpy as np

    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}


def value_accuracy_of(crit: dict, own: dict | None, tab, fo, cl, G, counts) -> dict:
    """The critic's value accuracy (``analysis.value_accuracy``; Section 3.3 'Value
    accuracy', Section 12 decision 7, Section 13): sigma_R V_c (main V; each head
    descriptively) against the deterministic policy's own return ``own['v_own_det']`` on the
    own stage's rows (``own['v_rows']``: T's interior restarts and D) where the run did not
    crash (``own['v_own_valid']``), key weights renormalised there, with the gamma^H tails,
    the horizons and the crashed and strip-end counts reported. Without the own stage's
    returns it does not pass ('reason'). The main V's score decides (it raises); the heads'
    scores, the horizons, the tails and the horizon-corrected score are descriptive
    (``analysis.guarded``)."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import states as ST

    v = crit["value"]
    if own is None or AN.VALUE_TARGET not in own:
        return {**AN.value_accuracy(v[:, 0], None, fo.outer, np.zeros(len(v))),
                "reason": f"no own returns ({AN.VALUE_TARGET}: the own stage)", "heads": []}
    asked = np.asarray(own["v_rows"], bool)
    ok = asked & np.asarray(own["v_own_valid"], bool)
    t = np.asarray(own[AN.VALUE_TARGET], float)
    w = ST.stratum_weights(tab, ok)
    va = AN.value_accuracy(v[:, 0], t, fo.outer, w, ok, cl, G, counts)
    va["heads"] = AN.guarded(lambda: [AN.value_accuracy(v[:, i], t, fo.outer, w, ok)["r2"]
                                      for i in range(1, v.shape[1])])
    va["n_asked"] = int(asked.sum())
    va["n_t_restarts"] = int((ok & (np.asarray(tab["t"]) > 0)).sum())
    va["n_d"] = int((ok & (np.asarray(tab["t"]) == 0)).sum())
    va["n_terminated"] = int((asked & np.asarray(own["v_own_terminated"], bool)).sum())
    va["n_strip_end"] = int((ok & np.asarray(own["v_own_strip_end"], bool)).sum())
    va["v_steps"] = int(own["v_steps"]) if "v_steps" in own else None
    va["H"] = AN.guarded(lambda: _summary(np.asarray(own["v_own_H"], float)[ok]))
    va["tail"] = AN.guarded(lambda: _summary(np.asarray(own["v_own_tail"], float)[ok]))
    if "v_own_inf" in own:            # descriptive: J plus V*inf's constant-speed tail at x_H
        va["horizon_corrected_r2"] = AN.guarded(lambda: AN.value_accuracy(
            v[:, 0], np.asarray(own["v_own_inf"], float), fo.outer, w, ok,
            target_name="v_own_inf")["r2"])
    return va


def c1_frame(world, frozen: dict, flags: dict):
    """C1's inputs on T that do not depend on the critic: partition 0's folds ('fo'), the
    geometry clusters ('cl', 'G'), K_P's columns of the 8 state components ('idx'), the
    winsorised K_P targets and their bounds ('lam', 'bnd'), the interior ('inner') and its
    key weights ('w_in'), and the network's decision mask ('valid', ``states.decision_mask``:
    s's observability under its own obs_rms, K2b's frozen exclusions)."""
    from types import SimpleNamespace

    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    tab = world.tab
    cl, G = AN.geometry_clusters(tab)
    inner = np.asarray(tab["interior"], bool)
    return SimpleNamespace(
        tab=tab, fo=PB.folds(tab), cl=cl, G=G, idx=[TG.STATE_NAMES.index(c) for c in TG.K_P],
        lam=_winsorised(tab, frozen, TG.K_P), bnd=_bounds(frozen, TG.K_P), inner=inner,
        w_in=ST.stratum_weights(tab, inner),
        valid=ST.decision_mask(tab, ~flags["observability"].s_unobservable,
                               frozen.get("k2b_excluded")))


def c1_bare(fr, g, counts):
    """C1 without nulls (``analysis.c1`` on ``c1_frame``'s rows, decision mask, calibration,
    bounds and bootstrap, every slope CI) of a pullback ``g`` [n, 8]: the accel-held
    sensitivity, C3-R2c's x (its R2_c) and the descriptive C1 of the networks outside the
    decision set."""
    from apexzero.costate import analysis as AN
    from apexzero.costate import targets as TG

    return AN.c1(g[:, fr.idx], fr.lam, fr.fo.outer, fr.w_in, fr.cl, fr.G, fr.tab["direction"],
                 fr.inner, None, fr.valid, TG.K_P, counts, slope_boot="always", bounds=fr.bnd)


def heads_of(crit: dict, fr, counts) -> dict:
    """C1-heads (``analysis.c1_heads``, Section 6: each generalist's 5 heads) on T's interior
    from a critic's outputs (``critic_of``): per head V, the 8 gradient components and pi."""
    from apexzero.costate import analysis as AN
    from apexzero.costate import targets as TG

    if crit["g"].shape[1] < 3:
        return {"decision": "not testable", "reason": "the critic has fewer than 2 heads"}
    q = {"V": crit["value"][:, 1:]}
    for i, c in enumerate(TG.STATE_NAMES):
        q[f"g_{c}"] = crit["g"][:, 1:, i]
    q["pi_steer"], q["pi_pedal"] = crit["pi"][:, 1:, 0], crit["pi"][:, 1:, 1]
    return AN.c1_heads(q, fr.w_in, fr.cl, fr.G, fr.inner, counts)


def describe_network(world, frozen: dict, crit: dict, flags: dict, counts,
                     heads: bool = True) -> tuple[dict, float]:
    """A network outside the decision set (Section 6: descriptive for C1, C2 and C2-sw; the
    specialists throughout): its record, {'C1': C1 without nulls (``c1_bare``), and with
    ``heads`` 'C1-heads' (``heads_of``, contained: ``analysis.guarded``)}, and C1's R2_c
    (C3-R2c's x for a generalist). Its C2 and C2-sw are deferred (``prereg.DEFERRED``)."""
    from apexzero.costate import analysis as AN

    fr = c1_frame(world, frozen, flags)
    c = c1_bare(fr, crit["g"][:, 0], counts)
    out = {"C1": {**c.record(), "descriptive": True, "label_scope": "descriptive",
                  "nulls_omitted": list(AN.NULL_NAMES)}}
    if heads:
        out["C1-heads"] = AN.guarded(heads_of, crit, fr, counts)
    return out, c.stat


def analyse_network(name: str, world, frozen: dict, crit: dict, nulls: dict, flags: dict,
                    counts, own: dict | None = None, k3b_trained: bool = False,
                    variants=None, strata=None) -> dict:
    """C1 and its secondaries of one network (``costate.analysis``): every statistic on a
    stratum with the key weights renormalised there (``states.stratum_weights``), each K_P
    component on its decision rows (``states.decision_mask``: s's observability, K2b's
    frozen exclusions), the calibrated predictions clipped to the frozen winsorisation
    bounds, the pooled slopes' CIs per component whatever the decision (Section 6; the label
    still needs a confirm); the critic's value accuracy against its own policy's return
    (``value_accuracy_of``: main V and, descriptively, each head), which with K3b's trained
    analogue decides whether C1-own is a secondary or descriptive, and against the optimum's
    V*inf on T's interior ('value_accuracy_vstar', descriptive); C1-pattern, and on the
    interior's pedal-saturated states descriptively with pedal kept there; C1-own (without
    the own stage's output: 'not testable', descriptive, with the reason) and its accel-held
    form ('C1-own_held': the critic's accel-held pullback against lambda^own_held,
    descriptive); C1's pre-registered variants (``variant_specs``; on partitions 1-10 the
    fold-dependent nulls, fitted on partition 0, are left out and said so; partition 0's
    key-weighted variants carry their slope CIs), C1-braking and its non-braking counterpart
    bootstrapped over their own geometries, and the accel-held pullback ('pullback:held',
    Section 3.2's pre-registered sensitivity: ``crit['g_held']`` scored as C1 with every null
    left out, ``nulls_omitted``, descriptive: its 'band', no decision or label); and C1's
    descriptive strata table
    (``analysis.STRATA['C1']``, or ``strata``: each stratum with its own geometries'
    bootstrap, ``by_stratum``; a stratum whose rows the decision or a variant already scored
    reuses that record, so a named secondary has one CI; the kappa-clipped strata from
    ``flags['kappa_clipped']``). ``variants`` limits the variants (``variant_specs``' names,
    'pullback:held' among them; all by default).

    Containment (``analysis`` module docstring): C1 and the decision rules of C1-pattern,
    value accuracy, C1-own, C1-limit and C1-braking raise; every other entry is descriptive
    and goes through ``analysis.guarded``, which records {'error': ...} in its place."""
    import hashlib

    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    tab = world.tab
    fr = c1_frame(world, frozen, flags)
    fo, cl, G, idx, lam, bnd = fr.fo, fr.cl, fr.G, fr.idx, fr.lam, fr.bnd
    inner, w_in, valid = fr.inner, fr.w_in, fr.valid
    g = crit["g"][:, 0, idx]
    # [n, 8], or [F, n, 8] per held-out fold (``analysis.FOLD_NULLS``): K_P on the last axis
    nl = {k: np.asarray(v)[..., idx] for k, v in nulls.items() if k != "fits"}
    nl["random_critics"] = [r[:, idx] for r in crit["random_g"]]
    c1 = AN.c1(g, lam, fo.outer, w_in, cl, G, tab["direction"], inner, nl, valid, TG.K_P,
               counts, slope_boot="always", bounds=bnd)
    out = {"C1": c1.record(), "decision_rows": {c: int((inner & valid[:, k]).sum())
                                                for k, c in enumerate(TG.K_P)},
           "decision_mask_sha256": hashlib.sha256(
               np.ascontiguousarray(valid).tobytes()).hexdigest()}
    # value accuracy (Sections 3.3, 12 decision 7, 13): C1-own's validity, against the
    # network's own deterministic return; against V*inf on T's interior descriptively
    va = value_accuracy_of(crit, own, tab, fo, cl, G, counts)
    out["value_accuracy"] = va

    def vstar_score() -> dict:
        vstar = (np.asarray(tab[AN.VALUE_TARGET_VSTAR], float) if AN.VALUE_TARGET_VSTAR in tab
                 else None)
        vs = AN.value_accuracy(crit["value"][:, 0], vstar, fo.outer, w_in, inner, cl, G,
                               counts, target_name=AN.VALUE_TARGET_VSTAR,
                               rows_name=AN.VALUE_ROWS_VSTAR)
        vs["heads"] = [AN.value_accuracy(crit["value"][:, i], vstar, fo.outer, w_in,
                                         inner)["r2"] for i in range(1, crit["value"].shape[1])]
        vs["descriptive"] = True                     # never C1-own's precondition
        vs["meets_threshold"] = vs.pop("pass")
        return vs

    out["value_accuracy_vstar"] = AN.guarded(vstar_score)
    # C1-pattern: all 8 components, delta and pedal where their bounds are inactive
    lam8 = _winsorised(tab, frozen, TG.STATE_NAMES)
    bnd8 = _bounds(frozen, TG.STATE_NAMES)
    valid8 = np.ones((len(w_in), 8), bool)
    valid8[:, idx] = valid
    valid8[:, 6] &= ~np.asarray(tab["delta_bound"], bool)
    sat = np.asarray(tab["pedal_saturated"], bool)
    pedal_free = valid8.copy()
    valid8[:, 7] &= ~sat
    out["C1-pattern"] = AN.c1_pattern(crit["g"][:, 0], lam8, fo.outer, w_in, cl, G,
                                      tab["direction"], inner, valid8, TG.STATE_NAMES,
                                      counts=counts, bounds=bnd8)

    # Section 6 'the saturated stratum is reported separately' (descriptive): C1-pattern on
    # the interior's pedal-saturated states, pedal scored there
    def pattern_saturated() -> dict:
        rs = inner & sat
        cp = {"decision": "not testable", "reason": "no row"}
        if rs.any():
            cl_p, G_p, cnt_p = _subclusters(tab, rs, AN.STRATUM_SEED - 1)
            cp = AN.c1_pattern(crit["g"][:, 0], lam8, fo.outer, ST.stratum_weights(tab, rs),
                               cl_p, G_p, tab["direction"], rs, pedal_free, TG.STATE_NAMES,
                               counts=cnt_p, bounds=bnd8)
        cp["n"], cp["descriptive"] = int(rs.sum()), True
        return cp

    out["C1-pattern_saturated"] = AN.guarded(pattern_saturated)
    out["C1-heads"] = AN.guarded(heads_of, crit, fr, counts)   # the generalists' 5 heads
    if own is None:          # a non-confirmatory run only (the confirmatory one requires it)
        out["C1-own"] = {"decision": "not testable", "descriptive": True,
                         "reason": "no own stage output"}
        out["C1-own_held"] = dict(out["C1-own"])
    else:
        lo = own["lam_own"][:, idx]
        ok = np.isfinite(lo).all(1) & np.asarray(own["valid"], bool)
        out["C1-own"] = AN.c1_own(g, lo, fo.outer, ST.stratum_weights(tab, ok), cl, G,
                                  tab["direction"], ok, k3b_trained, TG.K_P, counts, valid,
                                  value_accuracy_pass=va["pass"])

        def own_held() -> dict:
            # Section 13: lambda^own's accel-held form beside it, against the critic's
            # accel-held pullback (the same derivative on both sides); descriptive
            lh = own["lam_own_held"][:, idx]
            okh = np.isfinite(lh).all(1) & np.asarray(own["valid"], bool)
            r = AN.c1_own(crit["g_held"][:, 0, idx], lh, fo.outer, ST.stratum_weights(tab, okh),
                          cl, G, tab["direction"], okh, k3b_trained, TG.K_P, counts, valid,
                          value_accuracy_pass=va["pass"])
            r.update(status="descriptive (the accel-held form)", descriptive=True,
                     pullback="held")
            return r

        out["C1-own_held"] = AN.guarded(own_held)
    if "dV_dwin" in crit:                           # W1: the critic's width price against Pi^in
        w1 = inner if AN.W1_ROWS == "interior" else np.ones(world.n, bool)
        out["W1"] = AN.guarded(lambda: AN.calibrated_agreement(
            crit["dV_dwin"], np.asarray(tab["Pi_in"], float), fo.outer,
            ST.stratum_weights(tab, w1), cl, G, w1, counts,
            bounds=[frozen["winsor_bounds"]["Pi_in"]]))
    sd = np.array([np.sqrt(ST.wvar(lam[inner, k], w_in[inner])) for k in range(4)])
    wall = np.asarray(tab["wall_side"])
    out["C1-limit"] = AN.c1_limit(c1.calibrated, g, lam, fo.outer,
                                  ST.stratum_weights(tab, wall != 0), cl, G, wall, sd, TG.K_P,
                                  counts, direction=tab["direction"], valid=valid)
    out["sensitivity"] = {}
    # partition 0's per-fold nulls have no slices for partitions 1-10's folds (the random
    # critics are a list of [n, 4] arrays, fold-free)
    fold_nulls = set(AN.FOLD_NULLS) | {k for k, v in nl.items()
                                       if isinstance(v, np.ndarray) and v.ndim == 3}
    done = {inner.tobytes(): out["C1"]}   # rows -> record (partition 0, key weights)

    def variant(i: int, sp: dict, rows) -> dict:
        fo_v = PB.folds(tab, sp["partition"])
        cl_v, G_v, cnt_v = _subclusters(tab, rows, i)
        nl_v = {k: v for k, v in nl.items() if sp["partition"] == 0 or k not in fold_nulls}
        canon = sp["partition"] == 0 and sp["scheme"] == "key"
        r = AN.c1(g, lam, fo_v.outer, ST.stratum_weights(tab, rows, sp["scheme"]), cl_v, G_v,
                  tab["direction"], rows, nl_v, valid, TG.K_P, cnt_v,
                  slope_boot="always" if canon else False, bounds=bnd).record()
        r["nulls_omitted"] = sorted(set(nl) - set(nl_v))
        # Section 12 decision 2: the "at scale" label gates C1's primary record only; a
        # variant's (C1-braking's among them) is reported, never a label claim
        r["label_scope"] = "descriptive"
        return r

    for i, (vname, sp) in enumerate(variant_specs(tab, frozen, variants).items()):
        if vname == "primary":
            continue
        keep = np.ones(world.n, bool) if sp["keep"] is None else np.asarray(sp["keep"], bool)
        rows = inner & keep
        # C1-braking (a secondary's decision rule) raises; every other variant is descriptive
        r = (variant(i, sp, rows) if vname == "braking_keys"
             else AN.guarded(variant, i, sp, rows))
        out["sensitivity"][vname] = r
        if sp["partition"] == 0 and sp["scheme"] == "key":
            done.setdefault(rows.tobytes(), r)
    if "braking_keys" in out["sensitivity"]:
        out["C1-braking"] = out["sensitivity"]["braking_keys"]
    if variants is None or "pullback:held" in variants:
        def held() -> dict:
            # without nulls C1's null rule is vacuous: the record holds its band, never a
            # decision or a label (as a strata row's)
            r = c1_bare(fr, crit["g_held"][:, 0], counts).record()
            r["band"] = r.pop("decision")
            r.pop("label_at_scale")
            r.update(nulls_omitted=sorted(nl), descriptive=True, label_scope="descriptive",
                     pullback="held")
            return r

        out["sensitivity"]["pullback:held"] = AN.guarded(held)
    parts = [v.get("stat") for k, v in out["sensitivity"].items() if k.startswith("partition")]
    parts = [v for v in parts if v is not None]
    if parts:
        out["robustness_C1"] = AN.guarded(lambda: {
            "stat": parts, "min": float(np.nanmin(parts)), "max": float(np.nanmax(parts)),
            "sd": float(np.nanstd(parts))})
    # C1's descriptive strata table (Sections 4, 6, 12 and 13; ``analysis.STRATA``)
    tab_s = AN.strata_tab(tab, flags)

    def stratum(j: int, sname: str, rows) -> dict:
        if rows is None:
            return AN.stratum_record("C1", {"reason": "no per-network column " + ", ".join(
                AN.missing_columns(tab_s, sname))}, None, tab_s)
        rec_s = done.get(rows.tobytes())
        if rec_s is None and not rows.any():
            rec_s = {"decision": "not testable", "reason": "no row"}
        if rec_s is None:
            def compute() -> dict:
                cl_s, G_s, cnt_s = _subclusters(tab, rows, AN.STRATUM_SEED + j)
                return AN.c1(g, lam, fo.outer, ST.stratum_weights(tab, rows), cl_s, G_s,
                             tab["direction"], rows, nl, valid, TG.K_P, cnt_s,
                             slope_boot="always", bounds=bnd).record()

            rec_s = done[rows.tobytes()] = AN.guarded(compute)
        return AN.stratum_record(
            "C1", rec_s, rows, tab_s,
            primary=bool(sname == AN.PRIMARY_STRATUM["C1"] and np.array_equal(rows, inner)))

    def table() -> dict:
        names = AN.STRATA["C1"] if strata is None else strata
        return {sname: AN.guarded(stratum, j, sname, rows)
                for j, (sname, rows) in enumerate(AN.strata_rows(tab_s, names).items())}

    out["by_stratum"] = {"C1": AN.guarded(table)}
    return out


def completion(run: str, root: Path = ROOT) -> float:
    """C3-w's y: the mean over eval_rdev.json's 12 R-dev circuits of the completion."""
    from apexzero.costate import analysis as AN

    d = json.loads((Path(root) / AN.network_of(run) / "eval_rdev.json").read_text())
    return float(sum(float(v["completion"]) for v in d.values()) / len(d))


def competence(run: str, root: Path = ROOT) -> float:
    """C3-w's covariate: the final training episode_return (the last row of metrics.csv)."""
    import csv

    from apexzero.costate import analysis as AN

    with open(Path(root) / AN.network_of(run) / "metrics.csv") as fh:
        rows = list(csv.DictReader(fh))
    return float(rows[-1]["episode_return"])


def fields_out(root: Path = ROOT) -> Path:
    """Step 2's outputs (outputs/fields, or through a worktree's outputs/outputs link)."""
    d = Path(root) / "outputs" / "fields"
    if (d / "fields-a2").exists():
        return d
    return Path(root) / "outputs" / "outputs" / "fields"


def overlap(run: str, root: Path = ROOT) -> tuple[float, str, str]:
    """C3-w's overlap covariate (Section 6): step 2's B on the A2 cells, the network's
    median over geometries of its per-key B (``fields.stages.edge_key_measures``, H4's
    statistic), with the rows file read (repo-relative without resolving symlinks, as the
    checkpoint pins: through a worktree's outputs/outputs link it is still a relative path,
    never an absolute local one in the record) and its sha256. Reads step 2's outputs
    only."""
    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR
    from apexzero.fields import stages as S
    from apexzero.fields.prereg import sha256_file
    from apexzero.interp import registry

    run_ = AN.network_of(run)
    d = fields_out(root)
    cands = [run_.replace("/", "_")] + [a for a, sp in registry.AGENTS.items() if sp.run == run_]
    for c in cands:
        p = d / "fields-a2" / f"{c}.jsonl"
        if p.exists():
            break
    else:
        raise FileNotFoundError(f"{run}: no step 2 fields-a2 rows")
    opt = {r["key_id"]: r for r in S.read_rows(d / "opt-fields" / "rows.jsonl")}
    b = S.median_statistic(S.edge_key_measures(S.read_rows(p), opt, "B"), n=1)
    return float(b), PR._repo_rel(p, root), sha256_file(p)


def c3w_of(out_root: Path, world, counts, rec: dict | None = None, keep=None,
           exclude=()) -> dict:
    """C3-w over the generalists with critic outputs: x the calibrated R2 (C1's calibration,
    on ``analysis.C3W_ROWS``, key weights renormalised there) of sigma_R dV_c/dDelta_in
    against Pi^in_t, y the mean completion and the covariates (competence, overlap) from the
    record's frozen C3-w inputs (read from runs/ only without a record: a dry run);
    ``keep`` restricts the rows (a sensitivity variant); the networks in ``exclude`` (those
    that failed K3 after the tag) are left out, listed in 'missing' and in
    'k3_trained_excluded'. A network whose x keeps no row with an out-of-fold calibration
    (``analysis.calibrated_agreement``: 'not testable', x NaN) is left out as missing, its
    reason in 'x_not_testable'; 'x_n_uncalibrated' counts each network's dropped rows."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST

    tab = world.tab
    fo = PB.folds(tab)
    rows = (np.asarray(tab["interior"], bool) if AN.C3W_ROWS == "interior"
            else np.ones(world.n, bool))
    if keep is not None:
        rows = rows & np.asarray(keep, bool)
    cl, G, cnt = (*AN.geometry_clusters(tab), counts) if keep is None else _subclusters(tab, rows)
    w = ST.stratum_weights(tab, rows)
    bnd = (rec or {}).get("frozen", {}).get("winsor_bounds", {}).get("Pi_in")
    x, missing, untestable, n_uncal = {}, [], {}, {}
    for name in AN.GENERALISTS:
        p = Path(out_root) / "post" / "critic" / f"{PR.net_file(name)}.npz"
        if name in exclude:
            continue
        if not p.exists():
            missing.append(name)
            continue
        with np.load(p) as d:
            if "dV_dwin" not in d.files:
                missing.append(name)
                continue
            dw = d["dV_dwin"]
        xr = AN.calibrated_agreement(dw, np.asarray(tab["Pi_in"], float), fo.outer, w, cl, G,
                                     rows, cnt, bounds=None if bnd is None else [bnd])
        x[name], n_uncal[name] = xr["r2c"], xr["n_uncalibrated"]
        if xr.get("decision") == "not testable":
            untestable[name] = xr["reason"]
    y, cov = c3_y(list(x), rec)
    out = AN.c3_w(x, y, covariates=cov)
    out["critic_missing"] = missing
    out["x_not_testable"] = untestable
    out["x_n_uncalibrated"] = n_uncal
    out["k3_trained_excluded"] = [n for n in AN.GENERALISTS if n in exclude]
    return out


def c3_y(names, rec: dict | None) -> tuple[dict, dict]:
    """C3's y (the mean completion) and covariates ({'competence': ..., 'overlap': ...}) of
    ``names`` from the record's frozen C3-w inputs (read from runs/ only without a record: a
    dry run); a network without step 2's overlap is left out of that covariate."""
    inputs = (rec or {}).get("c3w_inputs") or {}
    y, comp, ov = {}, {}, {}
    for name in names:
        if inputs:
            v = inputs[name]
            y[name], comp[name] = v["completion"], v["competence"]
            if v.get("overlap") is not None:
                ov[name] = v["overlap"]
        else:
            y[name], comp[name] = completion(name), competence(name)
    cov = {"competence": comp}
    if ov:
        cov["overlap"] = ov
    return y, cov


def c3r2c_of(x: dict, rec: dict | None) -> dict:
    """C3-R2c (``analysis.c3_r2c``, a named secondary): x each generalist's C1 R2_c without
    nulls (``c1_bare``; a decision network's C1 R2_c is the same number, its nulls never
    entering R2_c), y and the covariates as C3-w's (``c3_y``)."""
    from apexzero.costate import analysis as AN

    y, cov = c3_y(list(x), rec)
    return AN.c3_r2c(x, y, covariates=cov)


# ------------------------------------------------------------------------- the analysis


DECISION_FIELDS = ("per_network", "variants", "C3-w")
# the arrays the analysis reads outside ``analysis.guarded`` (``confirmatory_input_problems``
# requires them): the own stage's (C1-own, value accuracy) and every generalist's critic
# (C3-w's and C3-R2c's x; a decision network's also its random critics, C1's sanity floor)
OWN_FIELDS = ("lam_own", "valid", "v_rows", "v_own_det", "v_own_valid", "v_own_terminated",
              "v_own_strip_end", "k3b_pass")
CRITIC_FIELDS = ("g", "value", "dV_dwin")


def analysis_paths(root: Path) -> dict[str, list[Path]]:
    """The files the analysis reads: per decision network its critic, activations, probes,
    nulls, own stage arrays and K3b's trained analogue ('own', 'k3b'); every generalist's
    critic (C3-w's and C3-R2c's x, C1-heads) and every specialist's (its descriptive C1); and
    for each of those 34 networks K3's after-the-tag checks ('k3': post/critic/<net>.k3.json,
    generalists then specialists)."""
    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR

    root = Path(root)
    nets = [PR.net_file(AN.network_of(a)) for a in AN.DECISION_NETWORKS]
    out = {"post": [], "nulls": [], "own": [], "k3b": [], "c3w": [], "specialists": [],
           "k3": []}
    for f in nets:
        out["post"] += [root / "post" / "critic" / f"{f}.npz",
                        root / "post" / "activations" / f"{f}.npz",
                        root / "post" / "probes" / f"{f}.json"]
        out["nulls"].append(root / "nulls" / f"{f}.npz")
        out["own"].append(root / "post" / "own" / f"{f}.npz")
        out["k3b"].append(root / "post" / "own" / f"{f}.k3b.json")
    out["c3w"] = [root / "post" / "critic" / f"{PR.net_file(n)}.npz" for n in AN.GENERALISTS]
    out["specialists"] = [root / "post" / "critic" / f"{PR.net_file(n)}.npz"
                          for n in AN.SPECIALISTS]
    out["k3"] = [root / "post" / "critic" / f"{PR.net_file(n)}.k3.json"
                 for n in [*AN.GENERALISTS, *AN.SPECIALISTS]]
    return out


def _json_decision(path: Path):
    """A verdict file's decision.pass (K3b's trained analogue, K3 after the tag), or None
    when the file cannot be read or holds none."""
    try:
        return json.loads(Path(path).read_text())["decision"]["pass"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _npz_flag(path: Path, key: str) -> bool | None:
    """An npz's one-value boolean flag (k3b_pass, k3_trained_pass), or None without it."""
    import numpy as np

    with np.load(path) as d:
        if key in d.files and np.asarray(d[key]).size == 1:
            return bool(np.asarray(d[key]).item())
    return None


def k3_trained_of(out_root: Path) -> dict[str, dict]:
    """Every network's K3 after-the-tag verdict (``analysis_paths``' 'k3' files): {'pass':
    bool, 'reason': None or 'K3 trained check failed: ...'}; without a readable decision,
    {'pass': None, 'reason': 'not run'} (a non-confirmatory run only: the confirmatory one
    requires every file, ``confirmatory_input_problems``)."""
    from apexzero.costate import analysis as AN

    out = {}
    for name, p in zip([*AN.GENERALISTS, *AN.SPECIALISTS], analysis_paths(out_root)["k3"],
                       strict=True):
        dec = _json_decision(p) if p.exists() else None
        if not isinstance(dec, bool):
            out[name] = {"pass": None, "reason": "not run"}
            continue
        why = None
        if not dec:
            why = (json.loads(p.read_text())["decision"].get("reason")
                   or "K3 trained check failed (no reason recorded)")
        out[name] = {"pass": dec, "reason": why}
    return out


def confirmatory_input_problems(rec: dict, root: Path, prov=None) -> list[str]:
    """Why the confirmatory analysis may not run on these inputs (as step 2's
    ``confirmatory_input_problems`` and ``input_stamp_problems``): a decision network's
    critic, activations, probes, nulls, own stage arrays (``OWN_FIELDS``: C1-own and value
    accuracy) or K3b's trained analogue missing, or its k3b_pass not that file's decision; a
    generalist's or specialist's critic missing, or a generalist's without ``CRITIC_FIELDS``
    (a decision network's also without its random critics); any of the 34 networks' K3
    after-the-tag checks (post/critic/<net>.k3.json) missing or without a boolean decision,
    or its critic's k3_trained_pass missing or not that decision; a post-tag input not
    stamped by this code under this tag (or partial, or changed since); a nulls array not
    stamped by the record's code (``rec['code_hash']``: pre-tag stages cannot rewrite it
    after the tag, so under a logged code deviation this run's hash would refuse it for
    good) or not the one the record binds."""
    import numpy as np

    from apexzero.fields.prereg import sha256_file

    prov = F._prov() if prov is None else prov
    root = Path(root)
    paths = analysis_paths(root)
    probs = [f"{p.relative_to(root)} is missing" for k in ("post", "nulls")
             for p in paths[k] if not p.exists()]
    decided = set(paths["post"])
    for p in paths["c3w"]:
        if not p.exists():
            probs.append(f"{p.relative_to(root)} is missing (C3-w, C3-R2c)")
            continue
        need = CRITIC_FIELDS + (("random_g",) if p in decided else ())
        with np.load(p) as d:
            lacks = [k for k in need if k not in d.files]
        if lacks:
            probs.append(f"{p.relative_to(root)} lacks {', '.join(lacks)} (C3-w, C3-R2c, C1)")
    probs += [f"{p.relative_to(root)} is missing (the specialist's descriptive C1)"
              for p in paths["specialists"] if not p.exists()]
    for p, pk in zip(paths["own"], paths["k3b"], strict=True):
        k3b = None
        if not p.exists():
            probs.append(f"{p.relative_to(root)} is missing (C1-own, value accuracy)")
        else:
            with np.load(p) as d:
                lacks = [k for k in OWN_FIELDS if k not in d.files]
                if "k3b_pass" in d.files and np.asarray(d["k3b_pass"]).size == 1:
                    k3b = bool(np.asarray(d["k3b_pass"]).item())
            if lacks:
                probs.append(f"{p.relative_to(root)} lacks {', '.join(lacks)} (C1-own, value "
                             "accuracy)")
        if not pk.exists():
            probs.append(f"{pk.relative_to(root)} is missing (K3b's trained analogue)")
            continue
        dec = _json_decision(pk)
        if not isinstance(dec, bool):
            probs.append(f"{pk.relative_to(root)} has no K3b decision")
        elif k3b is not None and dec != k3b:
            probs.append(f"{p.relative_to(root)}'s k3b_pass is not {pk.name}'s decision")
    # K3 after the tag (the critic stage): every network whose critic the analysis reads
    for p, pk in zip([*paths["c3w"], *paths["specialists"]], paths["k3"], strict=True):
        if not pk.exists():
            probs.append(f"{pk.relative_to(root)} is missing (K3 after the tag)")
            continue
        dec = _json_decision(pk)
        if not isinstance(dec, bool):
            probs.append(f"{pk.relative_to(root)} has no K3 decision")
            continue
        if p.exists():
            flag = _npz_flag(p, "k3_trained_pass")
            if flag is None:
                probs.append(f"{p.relative_to(root)} lacks k3_trained_pass (K3 after the tag)")
            elif flag != dec:
                probs.append(f"{p.relative_to(root)}'s k3_trained_pass is not {pk.name}'s "
                             "decision")
    post = [p for k in ("post", "own", "k3b", "c3w", "specialists", "k3") for p in paths[k]
            if p.exists()]
    probs += F.input_stamp_problems(list(dict.fromkeys(post)), prov.code_hash, prov.deviation,
                                    prov.tag, rec.get("environment"))
    nl = [p for p in paths["nulls"] if p.exists()]
    probs += F.input_stamp_problems(nl, rec.get("code_hash"), prov.deviation, None,
                                    rec.get("environment"), check_tag=False)
    for p in nl:
        rel = f"nulls/{p.name}"
        want = (rec.get("outputs") or {}).get(rel, {}).get("sha256")
        if not want or sha256_file(p) != want:
            probs.append(f"{rel} is not the array the record binds")
    return probs


def input_hashes(root: Path) -> dict[str, str]:
    from apexzero.fields.prereg import sha256_file

    paths = analysis_paths(root)
    return {str(p.relative_to(root)): sha256_file(p) for k in paths for p in paths[k]
            if p.exists()}


def confirmatory_done(path: Path, tag: dict | None) -> str | None:
    """The sha256 of a genuine confirmatory decisions file, else None (as step 2's
    ``stages.confirmatory_done``): a JSON object with 'confirmatory' and 'decided' true,
    the decided results (``DECISION_FIELDS``), and an embedded stamp of the analyse stage,
    not partial, with '--confirmatory' in its argv, written under ``tag``."""
    from apexzero.fields.prereg import sha256_file

    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("confirmatory") is not True or d.get("decided") is not True:
        return None
    if not set(DECISION_FIELDS) <= set(d):
        return None
    pv = d.get("provenance")
    if not isinstance(pv, dict) or pv.get("stage") != "analyse" or pv.get("partial"):
        return None
    if "--confirmatory" not in (pv.get("argv") or []):
        return None
    t = pv.get("tag") or {}
    if tag is None or t.get("object") != tag.get("object") or t.get("commit") != tag.get("commit"):
        return None
    return sha256_file(Path(path))


def unlocked_by(path: Path, tag: dict | None, root: Path = ROOT) -> tuple[str | None, str]:
    """(sha256 of the confirmatory decisions that unlock a non-confirmatory analysis's
    decisions, or None; why not): a genuine stamped confirmatory output
    (``confirmatory_done``) that, inside ``root``, is committed as those bytes (step 2's
    rule, scripts/fields.py:2090-2110)."""
    from apexzero.fields import prereg

    if not Path(path).exists():
        return None, "no confirmatory run yet"
    sha = confirmatory_done(path, tag)
    if sha is None:
        return None, f"{path} is not a stamped confirmatory output"
    rel = F._rel(path, root)
    if rel is not None:
        try:
            shas = {prereg.sha256_bytes(b) for _, b in F.committed_versions(rel, root) if b}
        except RuntimeError as e:
            return None, f"git failed reading {rel}'s history: {e}"
        if sha not in shas:
            return None, f"{rel} is not committed (git add -f {rel} && git commit)"
    return sha, ""


def _npz(path: Path) -> dict:
    import numpy as np

    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def coverage(out_root: Path, world, flags_of) -> dict:
    """Coverage and plumbing only (no decision): which inputs exist, their arrays' shapes,
    the decision rows per component and the geometries."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR
    from apexzero.costate import states as ST

    tab = world.tab
    inner = np.asarray(tab["interior"], bool)
    out = {"n": world.n, "interior": int(inner.sum()),
           "braking_keys": len(np.unique(np.asarray(tab["key_id"])[
               np.asarray(tab["braking_key"], bool)])),
           "geometries": int(AN.geometry_clusters(tab)[1]), "networks": {}}
    def strata() -> dict:      # the strata tables' row counts (no statistic); kappa
        # clipping is per network: G1's flags here, every network's count below
        tab_s = AN.strata_tab(tab, flags_of(AN.network_of(AN.DECISION_NETWORKS[0])))
        return {h: AN.strata_counts(tab_s, names) for h, names in AN.STRATA.items()}

    out["strata"] = AN.guarded(strata)
    for a in AN.DECISION_NETWORKS:
        name = AN.network_of(a)
        f = PR.net_file(name)
        rec: dict = {}
        for kind, p in (("critic", Path(out_root) / "post" / "critic" / f"{f}.npz"),
                        ("nulls", Path(out_root) / "nulls" / f"{f}.npz"),
                        ("own", Path(out_root) / "post" / "own" / f"{f}.npz"),
                        ("activations", Path(out_root) / "post" / "activations" / f"{f}.npz")):
            rec[kind] = ({k: list(v.shape) for k, v in _npz(p).items()} if p.exists()
                         else None)
        rec["probes"] = (Path(out_root) / "post" / "probes" / f"{f}.json").exists()
        fl = flags_of(name)
        if fl is not None:
            valid = ST.decision_mask(tab, ~fl["observability"].s_unobservable)
            rec["decision_rows"] = {c: int((inner & valid[:, k]).sum())
                                    for k, c in enumerate(("s", "n", "xi", "vx"))}
            rec["kappa_clipped"] = int(np.asarray(fl["kappa_clipped"], bool).sum())
        out["networks"][name] = rec
    return out


def merge_probes(rec_n: dict, pr: dict) -> dict:
    """One network's probes output (``probes_of``) merged into its analysis record: C2 and
    C2-sw beside C1, each variant's C2 and C2-sw into that variant's entry, and the C2 and
    C2-sw strata tables beside C1's (``by_stratum``)."""
    pr = dict(pr)
    pr.pop("provenance", None)
    sens = pr.pop("sensitivity", {})
    tables = pr.pop("by_stratum", {})
    rec_n.update(pr)
    rec_n.setdefault("by_stratum", {}).update(tables)
    for k, v in sens.items():
        rec_n.setdefault("sensitivity", {}).setdefault(k, {}).update(
            {"C2": v.get("C2"), "C2-sw": v.get("C2-sw")})
    return rec_n


def stage_analyse(args) -> None:
    """The analysis (module docstring 'Run-once'). A confirmatory run checks its inputs
    before computing anything; a non-confirmatory run computes decisions only once a stamped
    confirmatory decisions file is committed (``unlocked_by``), and otherwise writes coverage
    and plumbing only."""

    from apexzero.costate import analysis as AN
    from apexzero.costate import inputs as IN
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB
    from apexzero.fields import network as NW

    rec = _post_tag(args, "the analysis")
    fixed = ROOT / DECISIONS_REL
    prov = F._prov()
    if args.confirmatory:
        if Path(args.out).resolve() != OUT.resolve():
            raise SystemExit("--confirmatory writes to outputs/costate only: drop --out")
        ro = F.run_once_problems(fixed)
        if ro:
            raise SystemExit("refusing to run the confirmatory analysis again: " + "; ".join(ro))
        probs = confirmatory_input_problems(rec, Path(args.out), prov)
        if probs:
            raise SystemExit("the confirmatory analysis runs once, on complete inputs: "
                             + "; ".join(probs[:20]))
        path, decide, unlock = fixed, True, None
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        unlock, why = unlocked_by(fixed, prov.tag)
        if unlock is None and fixed.exists():
            print(f"warning: {why}: coverage and plumbing only", flush=True)
        decide = unlock is not None
        path = _out(args, "analyse") / "decisions_dry.json"
    world = _load_world_frozen(args, rec)
    frozen = rec["frozen"]
    rms_cache: dict = {}

    def flags_of(name):
        if name not in rms_cache:
            pin = rec["networks"].get(name)
            if pin is None:
                return None
            rms_cache[name] = IN.flags(world, NW.upcast32(load_agent(name, pin).obs_rms))
        return rms_cache[name]

    res: dict = {"confirmatory": bool(args.confirmatory), "decided": bool(decide),
                 "unlocked_by": unlock, "tag_pin": prov.tag, "deviation": prov.deviation,
                 "inputs": input_hashes(Path(args.out))}
    if not decide:
        res["coverage"] = coverage(Path(args.out), world, flags_of)
        _write(path, res)
        return
    G = AN.geometry_clusters(world.tab)[1]
    counts = PB.cluster_counts(G)
    # K3 after the tag (the critic stage): a failing network keeps its statistics, marked
    # descriptive, and is missing from every decision it enters (``analysis.k3_trained_failed``)
    k3t = k3_trained_of(Path(args.out))
    if args.confirmatory and any(v["pass"] is None for v in k3t.values()):
        raise SystemExit("a K3 after-the-tag verdict vanished during the analysis")
    failed = {n: v["reason"] for n, v in k3t.items() if v["pass"] is False}
    res["k3_trained"] = {"networks": k3t, "failed": failed,
                         "not_run": [n for n, v in k3t.items() if v["pass"] is None],
                         "rule": "a failing network's decision-bearing records are "
                                 "descriptive and it is missing from every decision it "
                                 "enters (C1, C2, C2-sw, the variants, C3-w, C3-R2c)"}
    res["networks"] = {}
    for name in [AN.network_of(a) for a in AN.DECISION_NETWORKS]:
        f = PR.net_file(name)
        cp = Path(args.out) / "post" / "critic" / f"{f}.npz"
        npth = Path(args.out) / "nulls" / f"{f}.npz"
        if not cp.exists() or not npth.exists():
            if args.confirmatory:
                raise SystemExit(f"{name}: an input vanished during the analysis")
            res["networks"][name] = {"missing": True}
            continue
        crit, nulls = _npz(cp), _npz(npth)
        op = Path(args.out) / "post" / "own" / f"{f}.npz"
        if args.confirmatory and not op.exists():
            raise SystemExit(f"{name}: an input vanished during the analysis")
        own = _npz(op) if op.exists() else None
        an_args = (name, world, frozen, crit, nulls, flags_of(name), counts, own,
                   bool(own is not None and own.get("k3b_pass", False)))
        an_kw = {"variants": getattr(args, "variants", None)}
        # a network that failed K3 after the tag decides nothing, so it is contained
        rec_n = (AN.guarded(analyse_network, *an_args, **an_kw) if name in failed
                 else analyse_network(*an_args, **an_kw))
        pp = Path(args.out) / "post" / "probes" / f"{f}.json"
        if pp.exists():
            merge_probes(rec_n, json.loads(pp.read_text()))
        # K4's notes beside a C2 decision near the band, and C2's random-only null beside
        # the primary C2 (descriptive, contained)
        rec_n["k4_band_notes"] = AN.guarded(AN.band_notes, rec_n, frozen.get("k4_band_notes"))
        rec_n["c2_null_note"] = AN.guarded(AN.c2_null_note, rec_n, frozen.get("c2_null"))
        if name in failed:
            AN.k3_trained_failed(rec_n, failed[name])
        res["networks"][name] = rec_n
    # the networks outside the decision set (Section 6: descriptive; ``describe_network``):
    # an other generalist's C1 R2_c is C3-R2c's x, so it raises (unless the network failed
    # K3 after the tag: then it is out of C3-R2c and contained); a specialist's record is
    # descriptive throughout (``analysis.guarded``)
    decided = {AN.network_of(a) for a in AN.DECISION_NETWORKS}
    x_r2c = {n: res["networks"][n]["C1"]["stat"] for n in decided
             if n not in failed and "C1" in res["networks"].get(n, {})}
    res["descriptive_networks"] = {}
    for name in [*AN.GENERALISTS, *AN.SPECIALISTS]:
        if name in decided:
            continue
        cp = Path(args.out) / "post" / "critic" / f"{PR.net_file(name)}.npz"
        if not cp.exists():
            if args.confirmatory:
                raise SystemExit(f"{name}: an input vanished during the analysis")
            res["descriptive_networks"][name] = {"missing": True}
            continue
        crit = _npz(cp)
        if name in AN.GENERALISTS and name not in failed:
            rec_d, x_r2c[name] = describe_network(world, frozen, crit, flags_of(name), counts)
        else:
            rec_d = AN.guarded(lambda c=crit, n=name: describe_network(
                world, frozen, c, flags_of(n), counts, heads=n in AN.GENERALISTS)[0])
        if name in failed:
            AN.k3_trained_failed(rec_d, failed[name])
        res["descriptive_networks"][name] = rec_d
    res["C3-w"] = c3w_of(Path(args.out), world, counts, rec, exclude=failed)
    res["C3-R2c"] = c3r2c_of(x_r2c, rec)
    res["C3-R2c"]["k3_trained_excluded"] = [n for n in AN.GENERALISTS if n in failed]
    res["C3-w_sensitivity"] = {}
    for vname in ("drop_small", "drop_kappa_set"):
        res["C3-w_sensitivity"][vname] = AN.guarded(lambda v=vname: c3w_of(
            Path(args.out), world, counts, rec, variant_specs(world.tab, frozen, [v])[v]["keep"],
            exclude=failed))
    per = {a: AN.network_decisions(res["networks"].get(AN.network_of(a), {}).get("C1"),
                                   res["networks"].get(AN.network_of(a), {}).get("C2"),
                                   res["networks"].get(AN.network_of(a), {}).get("C2-sw"),
                                   excluded=failed.get(AN.network_of(a)))
           for a in AN.DECISION_NETWORKS}
    res["per_network"] = per
    res["variants"] = AN.variant_decisions(per)
    if args.confirmatory and F.run_once_problems(path):
        raise SystemExit(f"refusing to overwrite the confirmatory decisions {path}")
    _write(path, res)
    if args.confirmatory:
        print(f"commit it now: git add -f {DECISIONS_REL} && git commit", flush=True)


# ------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=2, help="XLA intra-op threads")
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in STAGES:
        p = sub.add_parser(name)
        p.add_argument("--out", default=str(OUT), help=f"output root (default {OUT_REL})")
        p.add_argument("--log-dir", default=str(OUT / "logs"))
        p.add_argument("--post-tag-deviation", default=None, metavar="REASON")
        if name in ("inputs", "k0"):
            p.add_argument("--limit", type=int, default=None,
                           help="this many A2 keys, spread over the list (a partial run)")
        if name == "inputs":
            p.add_argument("--partitions", type=int, default=None,
                           help="robustness partitions hashed (default all 11)")
        if name == "k0":
            p.add_argument("--kinds", nargs="+", default=["a2", "n"], choices=("a2", "n"),
                           help="the A2 keys and N's anchored finals (without n: partial)")
        if name in ("k1", "k2b", "k3", "k3b"):
            p.add_argument("--n", type=int, default=None, help="fewer items (a partial run)")
        if name == "k2":
            p.add_argument("--n-extend", type=int, default=None,
                           help="insurance solves (default 30; fewer is partial)")
        if name == "k3b":
            p.add_argument("--steps", type=int, default=None,
                           help="closed-loop steps (default own.K3B_STEPS = 160; another "
                                "is partial)")
        if name == "k4":
            p.add_argument("--which", nargs="*", default=None)
        if name == "k5":
            p.add_argument("--ceiling", default="mlp", choices=("mlp", "ridge"))
        if name in ("inputs", "nulls", "activations", "critic", "own", "probes"):
            p.add_argument("--network", action="append", default=None)
        if name == "nulls":
            p.add_argument("--mlp-alphas", type=float, nargs="+", default=None)
            p.add_argument("--mlp-maxiter", type=int, default=500)
        if name == "analyse":
            p.add_argument("--confirmatory", action="store_true")
        if name in POOLED:
            p.add_argument("--workers", type=_positive, default=None,
                           help="worker processes, each with --threads XLA threads and one "
                                "BLAS thread (default: the cores over --threads, less one; 1: "
                                "the serial path in this process); never partial")
    return ap


def _positive(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{v}: at least 1")
    return n


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _use_provenance(Provenance(args, args.stage))
    if args.stage in PRE_TAG:
        _pre_tag(args, args.stage)
        if args.stage not in X64_OFF:
            _x64()
    t0 = time.time()
    globals()["stage_" + args.stage.replace("-", "_")](args)
    print(f"{args.stage}: {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
