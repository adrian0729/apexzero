"""H4R: a fresh, pre-registered test of real-width training on edge tracking
(docs/interp/h4r-design.md), driving step 2's frozen code on 20 new networks.

    uv run --no-sync python scripts/h4r.py <stage> [options]

Pre-tag stages (refused once the tag h4r-prereg-v1 exists; no network is run):

    manifest        the 20 runs of design Section 2 (runs/h4r-v{1,3}-{base,realw}-s{10..14}):
                    each run's final checkpoint (path, timesteps, sha256; it must be train_dr.py's
                    last update, FINAL_TIMESTEPS), its config.json against Section 2 (``--steps
                    200e6 --seed s``, v3's tilt, --real-widths on all 12 R-dev profiles, every
                    other argument and the PPO and env blocks train_dr.py's defaults), its
                    eval_rdev.json (circuits completed: completion > 0.5) and the queue log's
                    retries and failures (runs/h4r.log). A run without a finished final
                    checkpoint and evaluation, or that failed twice in the queue, is listed as
                    missing with its reason, never silently dropped. Design Section 2's only way
                    to be missing is failing twice: with any run missing, prereg needs the queue
                    log pulled back, and a missing run the log does not show failing twice is
                    'unexplained' (never started, killed, or not pulled back; 'finished_missing'
                    when the log shows it done on the box), which prereg refuses unless that run
                    is named with --accept-missing RUN=REASON (repeatable; a name that is not
                    unexplained is refused). Each accepted run enters the record as a Section 2
                    deviation with its reason, its queue events and whether it finished on the
                    box. Writes outputs/h4r/manifest.json, committed (git add -f) before prereg.
    prereg          the pre-registration record scripts/h4r_prereg.json (tracked), from a clean
                    tree (scripts/h4r.py, the design document and the manifest committed and
                    equal to HEAD byte by byte) with step 2's freeze intact (``fields.py``'s full
                    freeze check: step 1's and step 2's code, record, keys and bound outputs) and
                    the manifest still matching the runs on disk. It binds the design document's
                    and this script's sha256, step 2's tag object and record sha256, step 2's
                    frozen A2 and gate2-trained key ids copied from step 2's record with their
                    content hashes, step 2's calibrated branch tolerance, the manifest (path,
                    sha256, every checkpoint and evaluation), the permutation design, Holm's
                    settings, the gate rule and the secondary tests. Then: commit it alone,
                    place the annotated tag, push both.

Post-tag stages (refused unless ``freeze_problems`` is empty: the annotated tag on origin as
the same object, HEAD's and the disk's scripts/h4r.py, design document, manifest and record
the tagged ones, the record's networks, missing runs, accepted runs and design block those of
the tagged manifest and this script (``record_manifest_problems``), the record's base commit
differing from the tag by the record alone, the write-once tag pin
outputs/h4r/tag-h4r-prereg-v1.json committed, every recorded checkpoint and evaluation
re-hashed, the record rebuilt from the tagged manifest and step 2's record unchanged, step 2's
tag its pinned object with its record and step 2's full freeze check passing). Only after it
do they switch x64 on and call ``network.unlock_trained``:

    harness-f32     the float32 harness runs (x64 off, step 2's ``fields.harness_det``) of every
                    present H4R network from each frozen A2 key's (cell, phase) start: the Gate 3
                    and branch-consistency reference of the fields stage. The first post-tag run
                    writes the tag pin, which must be committed before the next stage.
    gate2-trained   step 2's ``stages.gate2_trained_network`` on the record's 12 keys for every
                    present network. H4R's rule (design Section 4): a failing key is excluded for
                    that network as a pipeline exclusion counted in coverage; the network is
                    descriptive only if more than 2 of its 12 keys fail.
    fields          step 2's ``stages.network_fields`` on the frozen A2 keys (481, step 2's
                    fields-a2 body: base run, Gate 3 and branch consistency against the harness
                    reference with step 2's calibrated tolerance, VJP fields, edge measures);
                    resumes a network whose stamped rows are complete (--force redoes it).
    analyse         once, into the fixed outputs/h4r/analyse/decisions.json: per network G, B
                    and T as the median over geometries of the per-geometry medians of its
                    mirror-pair values (step 2's ``stages.median_statistic``) with step 2's
                    aggregation (pair completeness, exclusions in step 2's order with the
                    gate2-trained keys after step 2's pipeline failures, 30% coverage; the
                    harness file's checkpoint re-checked), H4R-G (real < baseline) and H4R-B
                    (real > baseline) by the exact within-variant permutation test (step 2's
                    ``analysis.permutation_test``: 63,504 relabellings for 5 vs 5, every
                    relabelling of the remaining sets when networks drop; not testable with
                    fewer than 3 per arm in either variant), Holm across the two at 0.05, refute
                    on the opposite one-sided p <= 0.05; the secondary S1 (completion, the same
                    test), S2 (paired by seed, sign flips) and S3 (Spearman of B and completion
                    within each arm, bootstrap CIs); and the descriptive tables (G, B, T with
                    CIs, left and right, the per-regime breakdown, the pooled picture with step
                    2's 18 H4 networks). Refuses when the file exists or was ever committed, or
                    when an input is incomplete or not stamped by this code under this tag;
                    prints its sha256 and the git command that commits it. --check runs the
                    input checks only and writes nothing.

Every output carries a provenance stamp (embedded, or a <file>.provenance.json sidecar, as step
2's): step 2's code hash and this script's sha256 at the start and just before writing (they
must agree), HEAD, the changed paths, argv, the partial options (--out elsewhere, --limit,
--network), start and end times, the H4R and step 2 tag identities and the environment; a write
is refused, naming the changed files, when the code changed while the stage ran. So src/,
scripts/fields.py, scripts/_bootstrap.py and scripts/h4r.py stay untouched in every session
(no new file either) until the decisions are committed. Stages log to runs/h4r-<stage>.log
(``--log-dir``). XLA is held to 2 threads unless --threads says otherwise. Nothing here changes
step 1's or step 2's code; their modules are imported read-only.
"""

import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import re
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
import fields as F  # step 2's frozen stage helpers (scripts/fields.py), read-only

ROOT = Path(__file__).resolve().parents[1]
TAG = "h4r-prereg-v1"
REMOTE = F.REMOTE
STEP2_TAG = F.TAG                                               # receptive-fields-prereg-v1
STEP2_TAG_OBJECT = "9ec48f30c5c46babbd1f173b2b82560b88a96610"   # pushed; never moves
STEP2_RECORD_REL = F.PREREG_REL                                 # src/apexzero/fields/prereg.json
STEP2_DECISIONS_REL = F.DECISIONS_REL                           # step 2's committed decisions
STEP2_DECISIONS_SHA256 = "7f5941453539cc0071686b9c041286f783cc3ae8744f7d1f107a76e599656fcf"
STEP2_OUT = F.OUT                                               # outputs/fields (opt-fields rows)
DESIGN_DOC = "docs/interp/h4r-design.md"
SCRIPT_REL = "scripts/h4r.py"
TESTS_REL = "tests/test_h4r.py"
PREREG_REL = "scripts/h4r_prereg.json"
OUT_REL = "outputs/h4r"
OUT = ROOT / OUT_REL
MANIFEST_REL = f"{OUT_REL}/manifest.json"
DECISIONS_REL = f"{OUT_REL}/analyse/decisions.json"            # the run-once output (committed)
KEYS_DIR: Path | None = None    # step 2's key files (None: io.KEYS_DIR; tests redirect it)
QUEUE_LOG = "runs/h4r.log"
H4R_PATHS = (SCRIPT_REL, DESIGN_DOC)            # H4R's code and design (in every stamp)
BOUND = (SCRIPT_REL, DESIGN_DOC, MANIFEST_REL)  # committed before prereg; the tag holds them
PRE_TAG = ("manifest", "prereg")
POST_TAG = ("harness-f32", "gate2-trained", "fields", "analyse")
STAGES = PRE_TAG + POST_TAG
RECORD_VERSION = 1

# design Section 2 (fixed before training)
VARIANTS = {"v1": {"tilt_fraction": 0.0, "tilt_alpha": 1.0},
            "v3": {"tilt_fraction": 0.5, "tilt_alpha": 2.0}}
REAL, BASE = "realw", "base"
ARMS = {BASE: False, REAL: True}                # arm -> --real-widths
SEEDS = (10, 11, 12, 13, 14)
TRAIN_STEPS = 200e6
NUM_ENVS, NUM_STEPS = 2048, 64
# train_dr.py's last update: (200e6 // (2048 * 64)) * 2048 * 64 (the final ckpt file's name)
FINAL_TIMESTEPS = (int(TRAIN_STEPS) // (NUM_ENVS * NUM_STEPS)) * NUM_ENVS * NUM_STEPS
# design Sections 4 and 5 (fixed now)
ALPHA = 0.05                    # Holm's family level and the refute level (one-sided)
MIN_PER_ARM = 3                 # not testable with fewer in any arm of either variant
GATE2_MAX_FAILING = 2           # descriptive only when more than this many of 12 keys fail
GATE2_REASON = "pipeline: gate2-trained"     # H4R's exclusion of a failing gate2-trained key
COMPLETION_MIN = 0.5            # a circuit is completed when more than half its runs finish
S3_STREAM = 201                 # S3's bootstrap seed stream (step 2's are 101-105)
BOOTSTRAP_N = None              # None: step 2's measures.BOOTSTRAP_N (20,000); tests lower it
PRIMARY = {"H4R-G": ("G", -1), "H4R-B": ("B", 1)}    # measure, direction (real - baseline)
# a manifest entry's items the record copies per present network (the freeze compares them)
NET_FIELDS = ("variant", "arm", "seed", "path", "sha256", "timesteps", "eval_path",
              "eval_sha256", "completed", "config_sha256")


def pin_rel(tag: str = TAG) -> str:
    """The write-once tag pin of ``tag`` (named by the tag, so a re-registration under a new
    tag never trips over an earlier tag's pin)."""
    return f"{OUT_REL}/tag-{tag}.json"


def run_name(variant: str, arm: str, seed: int) -> str:
    return f"runs/h4r-{variant}-{arm}-s{int(seed)}"


NETWORKS = tuple(run_name(v, a, s) for v in VARIANTS for a in (BASE, REAL) for s in SEEDS)
_NAME = re.compile(r"^runs/h4r-(v1|v3)-(base|realw)-s(\d+)$")


def parse_name(name: str) -> tuple[str, str, int]:
    """(variant, arm, seed) of an H4R run name."""
    m = _NAME.match(name)
    if m is None:
        raise ValueError(f"{name} is not an H4R run name")
    return m[1], m[2], int(m[3])


def _file(name: str) -> str:
    return name.replace("/", "_")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(path: Path) -> str:
    from apexzero.fields import prereg

    return prereg.sha256_file(path)


def _script_sha(root: Path = ROOT) -> str:
    p = Path(root) / SCRIPT_REL
    return _sha_file(p) if p.exists() else "missing"


# ------------------------------------------------------------------------- git


def blob(rev: str, rel: str, root: Path = ROOT) -> bytes | None:
    """The bytes of ``rel`` in commit (or tag) ``rev``; None when it has no such file."""
    r = F._git_bytes("cat-file", "blob", f"{rev}:{rel}", root=root)
    return r.stdout if r.returncode == 0 else None


def tree_problems(root: Path = ROOT, paths: tuple[str, ...] = BOUND) -> list[str]:
    """Why the files the record binds are not exactly HEAD's (the prereg stage needs a clean
    tree): each must be committed at HEAD with its bytes on disk equal to that blob (so an
    untracked or ignored file, an edit, an assume-unchanged or skip-worktree file is caught),
    and git status must show nothing under them."""
    out = []
    st = F._git("status", "--porcelain", "--untracked-files=all", "--", *paths, root=root)
    if st.returncode != 0:
        out.append(f"<git error: {st.stderr.strip()[:200]}>")
    else:
        out += [f"{ln[3:].strip()} differs from HEAD (git status)"
                for ln in st.stdout.splitlines() if ln.strip()]
    for rel in paths:
        b = blob("HEAD", rel, root)
        p = Path(root) / rel
        if b is None:
            ign = F._git("check-ignore", "-q", rel, root=root).returncode == 0
            out.append(f"{rel} is not committed at HEAD"
                       + ((f" (git add -f {rel})" if ign else "") if p.exists() else " (missing)"))
        elif not p.exists():
            out.append(f"{rel} is missing on disk")
        elif _sha_file(p) != _sha(b):
            out.append(f"{rel} on disk is not HEAD's")
    return list(dict.fromkeys(out))


def refuse_after_tag(what: str, root: Path = ROOT) -> None:
    """Pre-tag stages refuse once the tag exists: they write what the record binds."""
    if F._tag_object(TAG, root) is not None:
        raise SystemExit(f"refusing to run {what} after the tag {TAG}: it would change a "
                         "pre-registered input")


# ------------------------------------------------------------------------- the manifest


def expected_args(variant: str, arm: str, seed: int) -> dict:
    """train_dr.py's args of one run (design Section 2): ``--steps 200e6 --seed s``, v3's
    ``--tilt-fraction 0.5 --tilt-alpha 2``, ``--real-widths`` on all 12 R-dev profiles for the
    real-width arm, every other argument train_dr.py's default."""
    return {"steps": TRAIN_STEPS, "name": Path(run_name(variant, arm, seed)).name,
            "num_envs": NUM_ENVS, "num_steps": NUM_STEPS, "network": "mlp", "bank": 4096,
            "refresh": 256, "fixed_car": False, **VARIANTS[variant],
            "real_widths": ARMS[arm], "real_widths_from": "", "ensemble_heads": 5,
            "seed": int(seed)}


def expected_ppo() -> dict:
    """The PPO block of config.json that train_dr.py writes with its defaults."""
    from apexzero.agents.ppo import PPOConfig

    return dataclasses.asdict(PPOConfig(
        num_envs=NUM_ENVS, num_steps=NUM_STEPS, total_timesteps=int(TRAIN_STEPS),
        network="mlp", gamma=0.995, gae_lambda=0.95, actor_width=256, critic_width=256,
        mlp_layers=3, activation="tanh", ensemble_heads=5))


def expected_env() -> dict:
    from apexzero.env import racing

    return racing.EnvParams()._asdict()


def _same(a, b) -> bool:
    """Equal values of one JSON item (a bool only equals a bool; ints and floats compare as
    numbers)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return float(a) == float(b)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _diff(block: str, got: dict, want: dict) -> list[str]:
    out = []
    for k in sorted(set(got) | set(want)):
        if k not in got:
            out.append(f"{block}.{k} missing (want {want[k]!r})")
        elif k not in want:
            out.append(f"{block}.{k} = {got[k]!r} is not in design Section 2")
        elif not _same(got[k], want[k]):
            out.append(f"{block}.{k} = {got[k]!r}, want {want[k]!r}")
    return out


def config_problems(cfg: dict, variant: str, arm: str, seed: int, ppo: dict | None = None,
                    env: dict | None = None) -> list[str]:
    """How a run's config.json departs from design Section 2: its args (``expected_args``),
    its PPO block (``expected_ppo``: train_dr.py's defaults, ending at FINAL_TIMESTEPS) and its
    env block (``expected_env``); [] when it matches."""
    ppo = expected_ppo() if ppo is None else ppo
    env = expected_env() if env is None else env
    out = _diff("args", cfg.get("args") or {}, expected_args(variant, arm, seed))
    out += _diff("ppo", cfg.get("ppo") or {}, ppo)
    out += _diff("env", cfg.get("env") or {}, env)
    p = cfg.get("ppo") or {}
    try:
        b = int(p["num_envs"]) * int(p["num_steps"])
        if (int(p["total_timesteps"]) // b) * b != FINAL_TIMESTEPS:
            out.append(f"ppo ends at {(int(p['total_timesteps']) // b) * b} timesteps, not "
                       f"{FINAL_TIMESTEPS}")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        out.append("ppo: no num_envs, num_steps or total_timesteps")
    return out


_QUEUE = re.compile(r"^\[(?P<time>[^\]]+)\]\s+(?P<event>start|done|retry|FAILED)\s+(?P<run>\S+)")


def queue_events(text: str) -> dict[str, list[dict]]:
    """{run (directory name): [{time, event}]} from scripts/gpu_queue_h4r.sh's log lines
    ('start', 'done', 'retry', 'FAILED')."""
    out: dict[str, list[dict]] = {}
    for ln in text.splitlines():
        m = _QUEUE.match(ln.strip())
        if m:
            out.setdefault(m["run"], []).append({"time": m["time"], "event": m["event"]})
    return out


def completed(ev: dict) -> tuple[int, dict[str, float], list[str]]:
    """(circuits completed, completion per R-dev circuit, circuits absent) of an
    eval_rdev.json: a circuit is completed when its completion exceeds 0.5 (the width test's
    rule)."""
    from apexzero.tracks.corners import R_DEV

    absent = [c for c in R_DEV if c not in ev or "completion" not in (ev[c] or {})]
    comp = {c: float(ev[c]["completion"]) for c in R_DEV if c not in absent}
    return sum(1 for v in comp.values() if v > COMPLETION_MIN), comp, absent


def _pickle_timesteps(path: Path, name: str) -> int:
    """The timesteps a checkpoint holds (``agents.load_agent``: reads the pickle only)."""
    from apexzero.interp import agents

    return int(agents.load_agent(path, Path(name).name).timesteps)


def manifest_entry(name: str, root: Path = ROOT, queue: dict | None = None, timesteps_of=None,
                   ppo: dict | None = None, env: dict | None = None
                   ) -> tuple[dict, str | None, list[str]]:
    """(entry, why the run is missing or None, its config problems) for one run under
    ``root``: its queue events and retry log, its config.json and its problems, its
    eval_rdev.json (sha256, completion per circuit, circuits completed) and its final
    checkpoint (the last ckpt/*.pkl, as step 2 resolves a run's final snapshot: path, sha256,
    timesteps, which must be FINAL_TIMESTEPS and the file's name). Missing: no run directory,
    no config.json, failed twice in the queue, no eval_rdev.json (training did not finish) or
    one without every R-dev circuit, no checkpoint, or a last checkpoint at other timesteps."""
    timesteps_of = _pickle_timesteps if timesteps_of is None else timesteps_of
    root = Path(root)
    v, arm, seed = parse_name(name)
    run = root / name
    ent: dict = {"variant": v, "arm": arm, "seed": seed}
    q = (queue or {}).get(run.name)
    if queue is not None:
        q = q or []
        ent["queue"] = q
        ent["retried"] = any(e["event"] == "retry" for e in q)
        ent["failed_twice"] = any(e["event"] == "FAILED" for e in q)
    ent["first_attempt_log"] = (run.parent / f"{run.name}.log.1").exists()
    if not run.is_dir():
        return ent, "no run directory", []
    cp = run / "config.json"
    if not cp.exists():
        return ent, "no config.json", []
    cfg = json.loads(cp.read_text())
    ent["config_path"] = f"{name}/config.json"
    ent["config_sha256"] = _sha_file(cp)
    ent["args"] = cfg.get("args")
    probs = config_problems(cfg, v, arm, seed, ppo, env)
    if ent.get("failed_twice"):
        return ent, f"failed twice in the queue ({QUEUE_LOG})", probs
    ep = run / "eval_rdev.json"
    if not ep.exists():
        return ent, "no eval_rdev.json (training did not finish)", probs
    k, comp, absent = completed(json.loads(ep.read_text()))
    ent.update(eval_path=f"{name}/eval_rdev.json", eval_sha256=_sha_file(ep), completion=comp,
               completed=k)
    if absent:
        return ent, f"eval_rdev.json lacks R-dev circuits {absent}", probs
    ckpts = sorted((run / "ckpt").glob("*.pkl"))
    if not ckpts:
        return ent, "no checkpoint", probs
    final = ckpts[-1]
    ts = int(timesteps_of(final, name))
    ent.update(path=f"{name}/ckpt/{final.name}", sha256=_sha_file(final), timesteps=ts,
               n_checkpoints=len(ckpts))
    if ts != FINAL_TIMESTEPS or final.stem != f"{FINAL_TIMESTEPS:012d}":
        return ent, (f"the last checkpoint {final.name} holds {ts} timesteps, not "
                     f"{FINAL_TIMESTEPS} (training did not finish)"), probs
    return ent, None, probs


def build_manifest(root: Path = ROOT, timesteps_of=None, ppo: dict | None = None,
                   env: dict | None = None, resolve=None) -> dict:
    """The manifest of the 20 runs (module docstring) under ``root``. ``resolve(name)``: the
    checkpoint step 2's loader resolves for a run (``stages.checkpoint_path``; default only
    for this repository), which must be the manifest's. 'unexplained_missing': the missing
    runs the queue log does not show failing twice (design Section 2's only way to be
    missing): a run never started, killed, or not pulled back; 'finished_missing': those of
    them the queue log shows done on the box (it must be pulled back)."""
    root = Path(root)
    if resolve is None and root.resolve() == ROOT.resolve():
        from apexzero.fields import stages as S

        resolve = S.checkpoint_path
    qp = root / QUEUE_LOG
    queue = queue_events(qp.read_text()) if qp.exists() else None
    ppo = expected_ppo() if ppo is None else ppo
    env = expected_env() if env is None else env
    nets, missing, cfgp = {}, {}, {}
    for name in NETWORKS:
        ent, why, probs = manifest_entry(name, root, queue, timesteps_of, ppo, env)
        if why is None and resolve is not None:
            got = resolve(name)
            if got is None or Path(got).resolve() != (root / ent["path"]).resolve():
                probs = probs + [f"step 2's loader resolves {got}, not {ent['path']}"]
        nets[name] = ent
        if why:
            missing[name] = why
        if probs:
            cfgp[name] = probs
    known = {Path(n).name for n in NETWORKS}
    unexplained = [n for n in missing if not nets[n].get("failed_twice")]
    finished = [n for n in unexplained if _finished_on_box(nets[n].get("queue") or [])]
    return {"design_doc": DESIGN_DOC, "final_timesteps": FINAL_TIMESTEPS,
            "networks": nets, "present": [n for n in NETWORKS if n not in missing],
            "missing": missing, "unexplained_missing": unexplained,
            "finished_missing": finished, "config_problems": cfgp,
            "queue_log": {"path": QUEUE_LOG, "present": queue is not None,
                          "sha256": _sha_file(qp) if qp.exists() else None,
                          "unexpected_runs": sorted(set(queue or {}) - known)}}


def _finished_on_box(events: list[dict]) -> bool:
    """The queue log shows the run done (its last event not a failure)."""
    return any(e["event"] == "done" for e in events) and events[-1]["event"] != "FAILED"


def parse_accept(items: list[str] | None) -> dict[str, str]:
    """{run name: reason} from --accept-missing RUN=REASON items (RUN as 'runs/h4r-...' or its
    directory name); refuses a malformed item, an empty reason, an unknown or repeated run."""
    out: dict[str, str] = {}
    for it in items or []:
        run, sep, why = str(it).partition("=")
        run = run.strip()
        run = run if run.startswith("runs/") else f"runs/{run}"
        if not sep or not why.strip():
            raise SystemExit(f"--accept-missing {it!r}: give RUN=REASON (a non-empty reason)")
        if run not in NETWORKS:
            raise SystemExit(f"--accept-missing {it!r}: {run} is not an H4R run")
        if run in out:
            raise SystemExit(f"--accept-missing names {run} twice")
        out[run] = why.strip()
    return out


def manifest_problems(man: dict, accept: dict[str, str] | None = None) -> list[str]:
    """Why the prereg stage may not bind this manifest: a present run whose config departs
    from design Section 2; no run present; a missing run without the queue log (Section 2's
    only evidence for a missing run); a missing run the queue log does not explain (it failed
    twice) unless ``accept`` ({run: reason}, --accept-missing) names it, a run the log shows
    done on the box with its own message (pull it back); an accepted run that is not
    unexplained missing. A missing run's config problems are reported only."""
    accept = dict(accept or {})
    present = man.get("present", [])
    out = [f"{n}: {'; '.join(p[:3])}" for n, p in (man.get("config_problems") or {}).items()
           if n in present]
    if not present:
        out.append("no run is present")
    if man.get("missing") and not (man.get("queue_log") or {}).get("present"):
        out.append(f"runs are missing and the queue log {QUEUE_LOG} is not here: pull it back "
                   "and re-run the manifest stage (design Section 2's only evidence for a "
                   "missing run)")
    un = list(man.get("unexplained_missing", []))
    fin = set(man.get("finished_missing", []))
    done = [n for n in un if n in fin and n not in accept]
    other = [n for n in un if n not in fin and n not in accept]
    if done:
        out.append(f"runs the queue log shows finished on the box are missing here: {done} "
                   "(pull them back and re-run the manifest stage; if one is lost, name it "
                   "with --accept-missing RUN=REASON: a Section 2 deviation in the record)")
    if other:
        out.append(f"missing runs the queue log does not show failing twice: {other} (pull "
                   f"them and {QUEUE_LOG} back and re-run the manifest stage, or name each "
                   "with --accept-missing RUN=REASON)")
    stale = sorted(n for n in accept if n not in un)
    if stale:
        out.append(f"--accept-missing names runs that are not unexplained missing: {stale}")
    return out


def accepted_block(man: dict, accept: dict[str, str] | None) -> dict:
    """The record's 'missing_accepted': per unexplained missing run its reason (None when not
    accepted: ``manifest_problems`` refuses it), its queue events and whether it finished on
    the box (a design Section 2 deviation)."""
    accept = dict(accept or {})
    fin = set(man.get("finished_missing", []))
    return {n: {"reason": accept.get(n), "queue": (man["networks"].get(n) or {}).get("queue"),
                "finished_on_box": n in fin} for n in man.get("unexplained_missing", [])}


def stage_manifest(args) -> None:
    """The checkpoint manifest (module docstring), written to outputs/h4r/manifest.json."""
    man = build_manifest()
    _write(OUT / "manifest.json", man)
    print(f"manifest: {len(man['present'])}/{len(NETWORKS)} runs present")
    for n, why in man["missing"].items():
        print(f"  missing {n}: {why}")
    for n, probs in man["config_problems"].items():
        print(f"  CONFIG {n}: {'; '.join(probs[:6])}")
    if man["config_problems"]:
        print("the prereg stage refuses a present run whose config departs from design "
              "Section 2")
    if man["missing"] and not man["queue_log"]["present"]:
        print(f"  NO QUEUE LOG: pull {QUEUE_LOG} back (prereg refuses missing runs without it)")
    if man["unexplained_missing"]:
        print(f"  UNEXPLAINED missing (not failed twice in {QUEUE_LOG}): "
              f"{man['unexplained_missing']}")
    if man["finished_missing"]:
        print(f"  FINISHED ON THE BOX but missing here (pull them back): "
              f"{man['finished_missing']}")
    print(f"next: git add -f {MANIFEST_REL} && git commit -m 'H4R: checkpoint manifest'")


# ------------------------------------------------------------------------- step 2's identity


def step2_identity(root: Path = ROOT) -> tuple[dict, list[str]]:
    """Step 2's tag (object and commit), its record at the tag (sha256 of the blob, code hash)
    and its committed decisions' pinned sha256, with the problems: the tag missing or not its
    pinned object (this repository), the record absent at the tag or differing on disk."""
    out = []
    t = F._tag_object(STEP2_TAG, root)
    if t is None:
        return {}, [f"step 2's tag {STEP2_TAG} is missing"]
    if Path(root).resolve() == ROOT.resolve() and t[0] != STEP2_TAG_OBJECT:
        out.append(f"step 2's tag is {t[0][:12]}, not its pinned object {STEP2_TAG_OBJECT[:12]}")
    b = blob(STEP2_TAG, STEP2_RECORD_REL, root)
    if b is None:
        return {}, out + [f"no record {STEP2_RECORD_REL} at {STEP2_TAG}"]
    disk = Path(root) / STEP2_RECORD_REL
    if not disk.exists() or _sha_file(disk) != _sha(b):
        out.append(f"{STEP2_RECORD_REL} on disk is not the one at {STEP2_TAG}")
    rec = json.loads(b)
    return {"tag": STEP2_TAG, "tag_object": t[0], "tag_commit": F._tag_commit(STEP2_TAG, root),
            "record": STEP2_RECORD_REL, "record_sha256": _sha(b),
            "code_hash": rec.get("code_hash"),
            "decisions": {"path": STEP2_DECISIONS_REL, "sha256": STEP2_DECISIONS_SHA256}}, out


def step2_freeze_problems() -> list[str]:
    """Step 2's own full freeze check (``fields.freeze_problems`` with its disk verification
    against outputs/fields: step 1's and step 2's code, tags, record, keys, re-polish rows,
    checkpoints, rollouts, bound outputs and calibration, package versions)."""
    return [f"step 2: {p}" for p in F.freeze_problems(
        verify=lambda r, rt: F._verify_on_disk(r, rt, F.OUT))]


# ------------------------------------------------------------------------- the record


def design_block() -> dict:
    """The tests and rules the record fixes (design Sections 4 and 5)."""
    from math import comb

    return {
        "networks": {"variants": dict(VARIANTS), "arms": {"real": REAL, "baseline": BASE},
                     "seeds": list(SEEDS), "final_timesteps": FINAL_TIMESTEPS},
        "statistic": "each network's median over geometries of the per-geometry medians of "
                     "its mirror-pair values (stages.median_statistic on "
                     "stages.edge_key_measures)",
        "permutation": {"strata": "variant", "exchanged": "real/baseline labels within a "
                        "variant", "T": "sum over variants of mean(real) - mean(baseline)",
                        "relabellings_5v5": comb(10, 5) ** 2,
                        "dropped": "every relabelling of the remaining sets",
                        "min_per_arm": MIN_PER_ARM, "code": "analysis.permutation_test"},
        "primary": {"H4R-G": {"measure": "G", "direction": "real < baseline"},
                    "H4R-B": {"measure": "B", "direction": "real > baseline"}},
        "holm": {"alpha": ALPHA, "family": list(PRIMARY), "m": len(PRIMARY),
                 "not_testable_p": 1.0,
                 "confirm": "p <= its Holm level (alpha / (m - rank + 1), step-down)"},
        "refute": {"opposite_one_sided_p_max": ALPHA, "holm_adjusted": False},
        "gate2_trained": {"keys": 12, "failing_key": GATE2_REASON + " (after step 2's "
                          "pipeline reasons, before 'invalid outcome'), counted in coverage",
                          "descriptive_if_failing_more_than": GATE2_MAX_FAILING},
        "coverage_max": 0.30, "pair_rule": "undefined when more than 3/8 of its listed keys "
                                           "are excluded (measures.max_excluded)",
        "secondary": {
            "S1": {"measure": "R-dev circuits completed (completion > 0.5)",
                   "direction": "real > baseline", "test": "the primary permutation test",
                   "networks": "every present network (field exclusions do not apply)",
                   "decision": "analysis.decide at 0.05, unadjusted"},
            "S2": {"pairs": "same seed, both arms of a variant, both networks kept",
                   "statistic": "mean of the within-pair differences (real - baseline)",
                   "test": "every sign flip (1,024 for 10 pairs)", "measures": ["G", "B"],
                   "directions": {"G": "real < baseline", "B": "real > baseline"},
                   "min_pairs_per_variant": MIN_PER_ARM,
                   "decision": "analysis.decide at 0.05, unadjusted"},
            "S3": {"measure": "Spearman of median B and circuits completed",
                   "within": "each arm (both variants pooled; per variant reported)",
                   "networks": "the networks kept for H4R-B",
                   "ci": "95% percentile bootstrap over networks", "stream": S3_STREAM,
                   "descriptive": True}},
        "bootstrap_n": "measures.BOOTSTRAP_N", "completion_min": COMPLETION_MIN}


def build_record(root: Path, manifest: dict, manifest_sha: str, step2: dict, step2_rec: dict,
                 head: str, env: dict | None = None, keys_root: Path | None = None,
                 accept: dict[str, str] | None = None) -> dict:
    """The pre-registration record (module docstring) from the committed manifest, step 2's
    identity and its record (``step2_rec``: the frozen key lists, gate2-trained keys,
    calibrated tolerance, opt-fields binding and regime cell sets are copied from it), with
    the accepted missing runs (``accept``: {run: reason}; 'missing_accepted' only when the
    manifest has unexplained missing runs)."""
    from apexzero.fields import prereg
    from apexzero.interp import io

    root = Path(root)
    keys_root = io.KEYS_DIR if keys_root is None else keys_root
    sets = step2_rec["key_sets"]
    g2 = [list(t) for t in step2_rec["gate2_trained"]]
    tests = blob("HEAD", TESTS_REL, root)
    nets = {name: {k: manifest["networks"][name][k] for k in NET_FIELDS}
            for name in manifest["present"]}
    cal = step2_rec.get("calibration") or {}
    regimes = step2_rec.get("regimes") or {}
    extra = ({"missing_accepted": accepted_block(manifest, accept)}
             if manifest.get("unexplained_missing") else {})
    return {
        **extra,
        "version": RECORD_VERSION, "tag": TAG, "prereg_base_commit": head,
        "design_doc": DESIGN_DOC, "design_doc_sha256": _sha_file(root / DESIGN_DOC),
        "script": SCRIPT_REL, "script_sha256": _sha_file(root / SCRIPT_REL),
        "tests": {TESTS_REL: _sha(tests) if tests is not None else None},
        "manifest": {"path": MANIFEST_REL, "sha256": manifest_sha},
        "networks": nets, "missing": dict(manifest["missing"]),
        "step2": step2,
        "keys": {"a2": {"ids": [list(t) for t in sets["a2"]], "n": len(sets["a2"]),
                        "content": step2_rec["content"]["a2"],
                        "content_now": prereg.content_hash([tuple(t) for t in sets["a2"]],
                                                           keys_root)},
                 "a2_twins": {"ids": [list(t) for t in sets.get("a2_twins", [])],
                              "content": step2_rec["content"].get("a2_twins")},
                 "gate2_trained": {"ids": g2, "n": len(g2),
                                   "content": prereg.content_hash([tuple(t) for t in g2],
                                                                  keys_root)}},
        "branch_tolerance": {"tolerance": dict(cal.get("tolerance") or {}),
                             "path": cal.get("path"), "sha256": cal.get("sha256"),
                             "factor": cal.get("factor"),
                             "trained_mean_err": cal.get("trained_mean_err")},
        "opt_fields": (step2_rec.get("outputs") or {}).get("opt_fields"),
        "regime_cells": {v: (regimes.get("variant_regime_cells") or {}).get(v)
                         for v in VARIANTS},
        "regime_typing": {v: (regimes.get("h4_typing") or {}).get(v) for v in VARIANTS},
        "design": design_block(),
        "environment": F.environment(root) if env is None else env}


def stage_prereg(args) -> None:
    """The pre-registration record (module docstring): the last step before the tag."""
    from apexzero.interp import io

    accept = parse_accept(args.accept_missing)
    probs = tree_problems(ROOT)
    if probs:
        raise SystemExit("prereg refuses a dirty tree: " + "; ".join(probs[:8]) + " (commit "
                         f"{SCRIPT_REL}, {DESIGN_DOC} and git add -f {MANIFEST_REL} first)")
    s2, p2 = step2_identity(ROOT)
    if p2:
        raise SystemExit("prereg refuses: " + "; ".join(p2))
    p2 = step2_freeze_problems()
    if p2:
        raise SystemExit("prereg refuses: step 2's freeze does not hold: " + "; ".join(p2[:8]))
    man_path = ROOT / MANIFEST_REL
    man = json.loads(man_path.read_text())
    fresh = build_manifest(ROOT)
    if F.content_sha256(man) != F.content_sha256(fresh):
        raise SystemExit("prereg refuses: the committed manifest no longer matches the runs on "
                         "disk (re-run the manifest stage and commit it)")
    mp = manifest_problems(man, accept)
    if mp:
        raise SystemExit("prereg refuses the manifest: " + "; ".join(mp))
    rec2 = json.loads(blob(STEP2_TAG, STEP2_RECORD_REL, ROOT))
    rec = build_record(ROOT, man, _sha_file(man_path), s2, rec2, F._head(ROOT),
                       keys_root=io.KEYS_DIR if KEYS_DIR is None else KEYS_DIR, accept=accept)
    if rec["keys"]["a2"]["content_now"] != rec["keys"]["a2"]["content"]:
        raise SystemExit("prereg refuses: the A2 key files differ from step 2's record")
    if not rec["branch_tolerance"]["tolerance"]:
        raise SystemExit("prereg refuses: step 2's record holds no calibrated branch tolerance")
    path = ROOT / PREREG_REL
    path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(f"wrote {PREREG_REL}: {len(rec['networks'])} networks, {len(rec['missing'])} missing; "
          f"script {rec['script_sha256'][:12]}, HEAD {rec['prereg_base_commit'][:12]}")
    for n, a in (rec.get("missing_accepted") or {}).items():
        print(f"  ACCEPTED missing {n} (a Section 2 deviation"
              + (", finished on the box" if a["finished_on_box"] else "") + f"): {a['reason']}")
    print(f"next: git add {PREREG_REL} && git commit -m 'H4R: pre-registration record'; "
          f"git tag -a {TAG} -m 'H4R pre-registration'; git push {REMOTE} HEAD {TAG}")


# ------------------------------------------------------------------------- the freeze


def tagged_record(tag: str = TAG, root: Path = ROOT) -> dict | None:
    b = blob(tag, PREREG_REL, root)
    return json.loads(b) if b else None


def _record_base_problems(rec: dict, tag: str, root: Path) -> list[str]:
    """The record's base commit must exist, be an ancestor of the tag and differ from it
    under the bound files and step 2's code paths by the record alone."""
    base = rec.get("prereg_base_commit")
    if not base or F._git("cat-file", "-e", f"{base}^{{commit}}", root=root).returncode != 0:
        return [f"the record's base commit {base!r} is missing or unknown"]
    if F._git("merge-base", "--is-ancestor", base, tag, root=root).returncode != 0:
        return [f"the record's base commit {base[:12]} is not an ancestor of the tag {tag}"]
    d = F._git("diff", "--name-only", base, tag, "--", *BOUND, PREREG_REL, *F.CODE_PATHS,
               root=root)
    if d.returncode != 0:
        return [f"<git error: {d.stderr.strip()[:200]}>"]
    extra = [ln for ln in d.stdout.splitlines() if ln.strip() and ln != PREREG_REL]
    if not extra:
        return []
    return [f"the tag differs from the record's base commit {base[:12]} by more than the record: "
            + str(extra[:8])]


def record_manifest_problems(rec: dict, man: dict) -> list[str]:
    """The record against the tagged manifest (git content only): its networks exactly the
    manifest's present runs with their NET_FIELDS, its missing runs the manifest's, its
    accepted runs exactly the manifest's unexplained missing runs (each with a reason, its
    queue events and whether it finished on the box), its design block this script's, and the
    manifest still passing ``manifest_problems`` with the recorded acceptances. Which networks
    enter the tests, and why one is dropped, comes from the manifest, never from an edit."""
    out = []
    want = {n: {k: (man.get("networks") or {}).get(n, {}).get(k) for k in NET_FIELDS}
            for n in man.get("present", [])}
    if rec.get("networks") != want:
        diff = sorted(set(rec.get("networks") or {}) ^ set(want)) or sorted(
            n for n in want if (rec.get("networks") or {}).get(n) != want[n])
        out.append(f"the record's networks are not the tagged manifest's present runs: {diff[:6]}")
    if rec.get("missing") != man.get("missing"):
        out.append("the record's missing runs are not the tagged manifest's")
    acc = rec.get("missing_accepted") or {}
    un = list(man.get("unexplained_missing", []))
    if not isinstance(acc, dict) or not all(isinstance(a, dict) for a in acc.values()):
        out.append("the record's accepted missing runs are not {run: {reason, queue, "
                   "finished_on_box}}")
    elif sorted(acc) != sorted(un):
        out.append(f"the record's accepted missing runs {sorted(acc)} are not the tagged "
                   f"manifest's unexplained missing runs {sorted(un)}")
    else:
        want_acc = accepted_block(man, {n: a.get("reason") for n, a in acc.items()})
        if json.loads(json.dumps(want_acc)) != acc or \
                not all(isinstance(a.get("reason"), str) and a["reason"].strip()
                        for a in acc.values()):
            out.append("the record's accepted missing runs do not carry a reason, the "
                       "manifest's queue events and whether they finished on the box")
        out += [f"the tagged manifest: {p}" for p in manifest_problems(
            man, {n: a["reason"] for n, a in acc.items()
                  if isinstance(a.get("reason"), str) and a["reason"].strip()})]
    if rec.get("design") != json.loads(json.dumps(design_block())):
        out.append("the record's design block is not this script's")
    return out


def freeze_problems(tag: str = TAG, root: Path = ROOT, remote: str | None = REMOTE,
                    pin_path: Path | None = None, verify=None) -> list[str]:
    """Why a post-tag stage may not run (module docstring): the tag missing, lightweight, not
    on ``remote`` as the same object (None skips: tests) or not an ancestor of HEAD; the
    write-once tag pin (``pin_rel(tag)``) moved, or on disk but never committed (``fields``'s
    rule); a bound file (this script, the design document, the manifest, the record) at HEAD
    or on disk other than the tagged one; a record whose script, design document or manifest
    sha256 is not the tagged blob's, whose networks, missing runs, accepted runs or design
    block are not the tagged manifest's and this script's (``record_manifest_problems``),
    whose base commit differs from the tag by more than the record, or whose environment's
    package versions differ; and ``verify(record, root)``'s problems (the checkpoints,
    evaluations, the record rebuilt and step 2's freeze: ``_verify_on_disk``)."""
    t = F._tag_object(tag, root)
    if t is None:
        return [f"the git tag {tag} does not exist yet (design Section 6: the H4R networks' "
                + "fields are computed only after it)"]
    out = []
    if t[1] != "tag":
        out.append(f"{tag} is a lightweight tag: it must be annotated")
    if remote is not None:
        try:
            on = F.remote_tag_object(tag, root, remote)
            if on != t[0]:
                out.append(f"{tag} is not on {remote} as {t[0][:12]} (there: {on}); push the "
                           "tag before any post-tag stage")
        except RuntimeError as e:
            out.append(f"cannot confirm {tag} on {remote}: {e}")
    commit = F._tag_commit(tag, root)
    if F._git("merge-base", "--is-ancestor", commit or tag, "HEAD", root=root).returncode != 0:
        out.append(f"the tag {tag} is not an ancestor of HEAD")
    pin_path = Path(pin_path) if pin_path is not None else Path(root) / pin_rel(tag)
    if pin_path.exists():
        pin = json.loads(pin_path.read_text())
        if pin.get("object") != t[0] or pin.get("commit") != commit:
            out.append(f"the tag moved since the first post-tag run ({pin_path}: "
                       f"{str(pin.get('object'))[:12]}, now {t[0][:12]})")
    out += [str(p) for p in F._pin_committed_problems(pin_path, root, t[0], commit)]
    tagged = {}
    for rel in (*BOUND, PREREG_REL):
        tb = blob(tag, rel, root)
        p = Path(root) / rel
        if tb is None:
            out.append(f"{rel} is not in the tag {tag}")
            continue
        tagged[rel] = _sha(tb)
        if blob("HEAD", rel, root) != tb:
            out.append(f"HEAD's {rel} is not the tagged one")
        if not p.exists() or _sha_file(p) != tagged[rel]:
            out.append(f"{rel} on disk is not the tagged one")
    rec = tagged_record(tag, root)
    if rec is None:
        return out + [f"no pre-registration record {PREREG_REL} at the tag {tag}"]
    for rel, have in ((SCRIPT_REL, rec.get("script_sha256")),
                      (DESIGN_DOC, rec.get("design_doc_sha256")),
                      (MANIFEST_REL, (rec.get("manifest") or {}).get("sha256"))):
        if rel in tagged and have != tagged[rel]:
            out.append(f"the record does not register the tagged {rel}")
    if rec.get("tag") != tag or rec.get("version") != RECORD_VERSION:
        out.append(f"the record names tag {rec.get('tag')!r}, version {rec.get('version')}")
    mb = blob(tag, MANIFEST_REL, root)
    if mb is not None:
        out += record_manifest_problems(rec, json.loads(mb))
    out += _record_base_problems(rec, tag, root)
    env = F.env_problems(rec.get("environment"), F.environment(root))
    if env:
        out.append(f"package versions differ from the record's: {env[:6]}")
    if verify is not None:
        out += [str(p) for p in verify(rec, root)]
    return out


def network_problems(rec: dict, root: Path = ROOT) -> list[str]:
    """Each recorded checkpoint and evaluation re-hashed, and the run's last checkpoint still
    the recorded one (a newer one means the run changed)."""
    out = []
    for name, pin in sorted(rec.get("networks", {}).items()):
        p = Path(root) / pin["path"]
        if not p.exists() or _sha_file(p) != pin["sha256"]:
            out.append(f"{name}: the checkpoint {pin['path']} is not the recorded one")
            continue
        last = sorted((Path(root) / name / "ckpt").glob("*.pkl"))
        if not last or last[-1].resolve() != p.resolve():
            out.append(f"{name}: its last checkpoint is now {last[-1] if last else None}")
        e = Path(root) / pin["eval_path"]
        if not e.exists() or _sha_file(e) != pin["eval_sha256"]:
            out.append(f"{name}: {pin['eval_path']} is not the recorded evaluation")
    return out


def step2_record_problems(rec: dict, root: Path = ROOT) -> list[str]:
    """Step 2 against the record's copy: its tag object and record sha256 (``step2_identity``)
    and the copied key ids, gate2-trained keys and tolerance equal to step 2's record's."""
    s2, out = step2_identity(root)
    want = rec.get("step2") or {}
    for k in ("tag_object", "tag_commit", "record_sha256"):
        if s2.get(k) != want.get(k):
            out.append(f"step 2's {k} is {str(s2.get(k))[:12]}, the record's "
                       f"{str(want.get(k))[:12]}")
    if want.get("decisions") != {"path": STEP2_DECISIONS_REL, "sha256": STEP2_DECISIONS_SHA256}:
        out.append(f"step 2's decisions are pinned at {STEP2_DECISIONS_SHA256[:12]}, the "
                   f"record's {str((want.get('decisions') or {}).get('sha256'))[:12]}")
    b = blob(STEP2_TAG, STEP2_RECORD_REL, root)
    if b is not None:
        r2 = json.loads(b)
        keys = rec.get("keys") or {}
        if [list(t) for t in r2["key_sets"]["a2"]] != (keys.get("a2") or {}).get("ids"):
            out.append("the record's A2 keys are not step 2's")
        if [list(t) for t in r2.get("gate2_trained", [])] != \
                (keys.get("gate2_trained") or {}).get("ids"):
            out.append("the record's gate2-trained keys are not step 2's")
        tol = (r2.get("calibration") or {}).get("tolerance")
        if tol != (rec.get("branch_tolerance") or {}).get("tolerance"):
            out.append("the record's branch tolerance is not step 2's")
    return out


def record_rebuild_problems(rec: dict, root: Path = ROOT, tag: str = TAG,
                            keys_root: Path | None = None) -> list[str]:
    """The record rebuilt (``build_record``) from the tagged manifest, step 2's record at its
    tag, step 2's identity now and the key files (``keys_root``; default step 2's), with the
    record's own base commit, environment and acceptances: every item but 'tests' (informative)
    must be the recorded one (the copied key lists and hashes, tolerance, opt-fields binding,
    regime cells and typing included)."""
    mb, b2 = blob(tag, MANIFEST_REL, root), blob(STEP2_TAG, STEP2_RECORD_REL, root)
    if mb is None or b2 is None:
        return [(f"cannot rebuild the record: no {MANIFEST_REL} at {tag} or no "
                 f"{STEP2_RECORD_REL} at {STEP2_TAG}")]
    keys_root = KEYS_DIR if keys_root is None else keys_root
    try:
        acc = {n: (a or {}).get("reason")
               for n, a in (rec.get("missing_accepted") or {}).items()}
        want = build_record(root, json.loads(mb), (rec.get("manifest") or {}).get("sha256"),
                            step2_identity(root)[0], json.loads(b2),
                            rec.get("prereg_base_commit"), env=rec.get("environment"),
                            keys_root=keys_root, accept=acc)
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        return [f"cannot rebuild the record from the tagged manifest and step 2's record: {e!r}"]
    a, b = json.loads(json.dumps(rec)), json.loads(json.dumps(want))
    diff = sorted(k for k in set(a) | set(b) if k != "tests" and a.get(k) != b.get(k))
    if not diff:
        return []
    return [("the record is not what build_record gives from the tagged manifest and step 2's "
             f"record: {diff}")]


def _verify_on_disk(rec: dict, root: Path) -> list[str]:
    """The record against this repository: checkpoints and evaluations, step 2's identity and
    copies, the record rebuilt, and step 2's own full freeze check."""
    return network_problems(rec, root) + step2_record_problems(rec, root) + \
        record_rebuild_problems(rec, root) + step2_freeze_problems()


def pin_tag(tag: str = TAG, root: Path = ROOT, pin_path: Path | None = None) -> Path:
    """The write-once tag pin ``pin_rel(tag)`` (the tag's identity at the first post-tag
    run)."""
    pin_path = Path(pin_path) if pin_path is not None else Path(root) / pin_rel(tag)
    ident = F.tag_identity(tag, root)
    if ident is not None and not pin_path.exists():
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pin_path, "x") as fh:
            fh.write(json.dumps({**ident, "pinned_at": F._now()}, indent=1) + "\n")
        rel = F._rel(pin_path, root)
        if rel is not None:
            print(f"pinned the tag in {rel}: commit it before the next post-tag stage "
                  f"(git add -f {rel} && git commit -m 'H4R: tag pin')", flush=True)
    return pin_path


def import_problems(root: Path = ROOT, modules: dict | None = None) -> list[str]:
    """``fields.import_problems`` with this script allowed: every module loaded from inside
    the repository must be step 2's frozen code, scripts/fields.py, scripts/_bootstrap.py or
    scripts/h4r.py."""
    me = (Path(root) / SCRIPT_REL).resolve()
    mods = dict(sys.modules) if modules is None else dict(modules)
    keep = {k: m for k, m in mods.items()
            if not (getattr(m, "__file__", None) and Path(m.__file__).resolve() == me)}
    return F.import_problems(root, keep)


def _post_tag(args, what: str, x64: bool = True) -> dict:
    """The freeze check (``freeze_problems`` with ``_verify_on_disk``), the tag pin, then (x64)
    float64 and trained networks' fields allowed. Returns the tagged record."""
    from apexzero.fields import network as N

    if getattr(args, "limit", None) is not None and Path(args.out).resolve() == OUT.resolve():
        raise SystemExit("--limit writes a partial run: give --out outside outputs/h4r")
    bad = import_problems()
    if bad:
        raise SystemExit(f"refusing to run {what}: {'; '.join(bad)}")
    probs = freeze_problems(verify=_verify_on_disk)
    if probs:
        raise SystemExit(f"refusing to run {what}: " + "; ".join(probs[:12]))
    pin_tag()
    rec = tagged_record()
    p = _prov()
    p.tag, p.step2_tag, p.check_imports = F.tag_identity(TAG), F.tag_identity(STEP2_TAG), True
    if x64:
        F._x64()
        N.unlock_trained()
    return rec


# ------------------------------------------------------------------------- provenance


def _partial(args) -> list[str]:
    """The options that make a run partial (never read by the analysis)."""
    out = []
    if getattr(args, "out", None) is not None and Path(args.out).resolve() != OUT.resolve():
        out.append("--out")
    for name in ("limit", "network"):
        if getattr(args, name, None) not in (None, [], False):
            out.append(f"--{name}")
    return out


class Provenance:
    """A stage run's provenance (module docstring; step 2's stamp fields, so that
    ``fields.input_stamp_problems`` reads it, plus this script's sha256 and step 2's tag)."""

    def __init__(self, args=None, stage: str = "", root: Path = ROOT):
        from apexzero.fields import prereg

        self.root = Path(root)
        self.stage = stage
        self.code_hash = prereg.code_hash(self.root)
        self.h4r_sha256 = _script_sha(self.root)
        self.head = F._head(self.root)
        self.changed = F.changed_since("HEAD", F.CODE_PATHS + H4R_PATHS, self.root)
        self.argv = list(sys.argv)
        self.partial = _partial(args) if args is not None else []
        self.start = F._now()
        self.tag: dict | None = None
        self.step2_tag: dict | None = None
        self.environment = F.environment(self.root)
        self.check_imports = False

    def stamp(self, files: dict | None = None, content_sha256: str | None = None) -> dict:
        from apexzero.fields import prereg

        now = prereg.code_hash(self.root)
        if now != self.code_hash or _script_sha(self.root) != self.h4r_sha256:
            what = F.changed_since(self.head, F.CODE_PATHS + H4R_PATHS, self.root)[:8]
            raise SystemExit(f"refusing to write the {self.stage} output: the code changed while "
                             f"the stage ran (changed since {str(self.head)[:12]}: {what}); "
                             "remove or revert that and re-run (complete networks resume)")
        bad = import_problems(self.root) if self.check_imports else []
        if bad:
            raise SystemExit(f"refusing to write the {self.stage} output: {'; '.join(bad[:4])}")
        out = {"stage": self.stage, "code_hash": self.code_hash, "h4r_sha256": self.h4r_sha256,
               "head": self.head, "changed_vs_head": self.changed, "argv": self.argv,
               "partial": self.partial, "start": self.start, "end": F._now(),
               "deviation": None, "tag": self.tag, "step2_tag": self.step2_tag,
               "environment": self.environment, "files": dict(files or {})}
        if content_sha256 is not None:
            out["content_sha256"] = content_sha256
        return out


_PROV: list[Provenance] = []


def _prov() -> Provenance:
    if not _PROV:
        _PROV.append(Provenance(None, "?"))
    return _PROV[-1]


def _write(path: Path, obj: dict) -> Path:
    """A JSON object output with its embedded stamp (binding its content's sha256)."""
    from apexzero.interp import io

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    io.write_json(path, {**obj, "provenance": _prov().stamp(
        content_sha256=F.content_sha256(obj))})
    print(f"wrote {path}", flush=True)
    return path


def _stamp_file(path: Path) -> Path:
    from apexzero.fields import prereg

    path = Path(path)
    sc = prereg.stamp_path(path)
    sc.write_text(json.dumps(_prov().stamp({path.name: _sha_file(path)}), indent=1,
                             default=str) + "\n")
    return sc


def _stamped_write(path: Path, write) -> Path:
    """A non-object output (rows, npz) written after the code check, atomically, then its
    sidecar stamp (``fields._stamped_write``'s order)."""
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


def stage_log(args, stage: str, total: int, note: str = ""):
    """io.StageLog writing runs/h4r-<stage>.log (or under --log-dir)."""
    from apexzero.interp import io

    class H4RLog(io.StageLog):
        def __init__(self):
            super().__init__(stage, total, note=note)
            self.log_dir = Path(args.log_dir)
            self.path = self.log_dir / f"h4r-{stage}.log"
            self._note = f"h4r {stage} [pid {os.getpid()}]"

        def _activity(self, text):
            if self.log_dir.resolve() == io.RUNS_DIR.resolve():
                super()._activity(text)

    return H4RLog()


def _out(args, stage: str) -> Path:
    d = Path(args.out) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------- post-tag stages


def _networks(args, rec: dict) -> list[str]:
    """The record's present networks in design order; ``--network`` (an exploratory run)
    only into a root other than outputs/h4r."""
    have = [n for n in NETWORKS if n in rec.get("networks", {})]
    if getattr(args, "network", None):
        if Path(args.out).resolve() == OUT.resolve():
            raise SystemExit("--network writes an exploratory run: give --out outside "
                             "outputs/h4r")
        bad = [n for n in args.network if n not in have]
        if bad:
            raise SystemExit(f"not recorded networks: {bad}")
        return list(args.network)
    return have


def _pin(rec: dict, name: str) -> dict:
    pin = rec.get("networks", {}).get(name)
    if pin is None:
        raise SystemExit(f"{name} is not a recorded network")
    return pin


def frozen_refs(rec: dict, which: str) -> list:
    """The record's frozen key list (step 2's ids, never a fresh glob), as KeyRefs."""
    return F._frozen_refs((), rec, rec["keys"][which]["ids"])


def stage_harness_f32(args) -> None:
    """The float32 harness runs (x64 off) of every present network from each frozen A2 key's
    start (step 2's ``fields.harness_det``)."""
    rec = _post_tag(args, "the float32 harness runs", x64=False)
    from apexzero.fields import keys as K
    from apexzero.fields import network as N
    from apexzero.fields import stages as S
    from apexzero.interp import agents, rollout

    refs = F._limit(frozen_refs(rec, "a2"), args.limit)
    starts = S.harness_starts(refs)
    names = _networks(args, rec)
    bank, cells = K.design()
    out = _out(args, "harness-f32")
    with stage_log(args, "harness-f32", len(names)) as log:
        for name in names:
            pin = _pin(rec, name)
            path = ROOT / pin["path"]
            if _sha_file(path) != pin["sha256"]:
                raise SystemExit(f"{name}: {pin['path']} is not the recorded checkpoint")
            ag = agents.load_agent(path, Path(name).name)
            if int(ag.timesteps) != int(pin["timesteps"]):
                raise SystemExit(f"{name}: timesteps {ag.timesteps}, recorded {pin['timesteps']}")
            det = F.harness_det(rollout.agent_policy(ag.cfg), ag.actor_params, ag.obs_rms, bank,
                                cells, starts, N.STEPS)
            meta = {"net": f"{Path(name).name}@{ag.timesteps}", "ckpt_sha256": pin["sha256"],
                    "loop_steps": N.STEPS, "n_runs": len(starts)}
            _stamped_write(F.harness_path(args.out, name),
                           lambda q, det=det, meta=meta: N.save_det(q, det, meta))
            log.step(note=name)
    _write(out / "summary.json", {"networks": names, "n_starts": len(starts),
                                  "loop_steps": N.STEPS})


def gate_verdict(result: dict) -> dict:
    """H4R's gate2-trained rule (design Section 4) on ``stages.gate2_trained_network``'s
    result: the keys whose row did not pass (an exhausted loop included) are excluded for the
    network; it is descriptive only if more than GATE2_MAX_FAILING of them fail."""
    fail = sorted(str(r["key_id"]) for r in result.get("rows", []) if r.get("pass") is not True)
    return {"failing_keys": fail, "n_failing": len(fail), "n_keys": len(result.get("rows", [])),
            "descriptive": len(fail) > GATE2_MAX_FAILING}


def stage_gate2_trained(args) -> None:
    """Step 2's gate2-trained check on every present network at the record's 12 keys, with
    H4R's per-key rule (``gate_verdict``)."""
    rec = _post_tag(args, "gate2-trained")
    from apexzero.fields import keys as K
    from apexzero.fields import stages as S

    refs = frozen_refs(rec, "gate2_trained")
    names = _networks(args, rec)
    bank, cells = K.design()
    res = {}
    with stage_log(args, "gate2-trained", len(names) * len(refs)) as log:
        for name in names:
            v = S.gate2_trained_network(S.load_network(name, _pin(rec, name), ROOT), refs, bank,
                                        cells)
            v["h4r"] = gate_verdict(v)
            res[name] = v
            log.step(len(refs), note=name)
    _write(_out(args, "gate2-trained") / "gate2_trained.json",
           {"keys": [r.key_id for r in refs], "networks": res,
            "rule": {"failing_key": GATE2_REASON,
                     "descriptive_if_failing_more_than": GATE2_MAX_FAILING},
            "descriptive": [n for n, v in res.items() if v["h4r"]["descriptive"]],
            "failing_keys": {n: v["h4r"]["failing_keys"] for n, v in res.items()}})
    for n, v in res.items():
        print(f"gate2-trained {n}: {v['h4r']['n_failing']}/{v['h4r']['n_keys']} keys fail"
              + (" (descriptive)" if v["h4r"]["descriptive"] else ""))


def _rows_done(path: Path, want: set[str], name: str, pin: dict, arrays_root: Path) -> bool:
    """A network's stamped rows are complete (resume): a stamp of this code and script under
    this tag, not partial, a row from the recorded checkpoint for every key with its checks,
    and the arrays the rows bind."""
    from apexzero.fields import prereg
    from apexzero.fields import stages as S

    st = prereg.read_stamp(path) if Path(path).exists() else None
    if st is None or st.get("partial") or st.get("h4r_sha256") != _prov().h4r_sha256:
        return False
    if F.input_stamp_problems([Path(path)], _prov().code_hash, None, _prov().tag):
        return False
    rows = S.read_rows(path)
    return not (S.rows_problems(rows, want, name, pin, checked=True)
                or S.arrays_problems(rows, S.arrays_dir(arrays_root, S.expected_net(name, pin)),
                                     name))


def stage_fields(args) -> None:
    """Step 2's ``stages.network_fields`` (its fields-a2 body) on the frozen A2 keys for every
    present network, with the harness reference and step 2's calibrated branch tolerance."""
    rec = _post_tag(args, "the H4R networks' fields on the A2 keys")
    from apexzero.fields import network as N
    from apexzero.fields import stages as S

    tol = {k: float(v) for k, v in rec["branch_tolerance"]["tolerance"].items()}
    refs = F._limit(frozen_refs(rec, "a2"), args.limit)
    want = {r.key_id for r in refs}
    names = _networks(args, rec)
    out = _out(args, "fields")
    todo = []
    for name in names:
        f = out / f"{_file(name)}.jsonl"
        if not args.force and not _prov().partial and \
                _rows_done(f, want, name, _pin(rec, name), out / "arrays"):
            print(f"{name}: complete stamped rows in {f.name} (--force redoes them)")
            continue
        todo.append(name)
    with stage_log(args, "fields", len(refs) * len(todo)) as log:
        for name in todo:
            pin = _pin(rec, name)
            hp = F.harness_path(Path(args.out), name)
            if not hp.exists():
                raise SystemExit(f"{name}: no float32 harness runs ({hp}): run harness-f32 first")
            det, meta = N.load_reference(hp)
            if meta.get("ckpt_sha256") != pin["sha256"]:
                raise SystemExit(f"{name}: {hp.name} is not this checkpoint's harness reference")
            net = S.load_network(name, pin, ROOT)
            rows = S.network_fields(net, refs, out / "arrays", det_of=lambda ref, det=det: det,
                                    branch_tol=tol, progress=lambda n: log.step(n))
            _write_rows(out / f"{_file(name)}.jsonl", rows)


# ------------------------------------------------------------------------- the decisions


def h4r_key_measures(rows: list[dict], opt_rows: dict, measure: str, failing=()) -> list:
    """Step 2's per-key edge measure (``stages.edge_key_measures``: primary non-capped A2
    keys, exclusions in step 2's order) with H4R's gate2-trained exclusion: a failing key of
    this network is excluded as 'pipeline: gate2-trained' unless step 2's own pipeline
    failure (loop, no optimum row, branch) already excludes it."""
    from apexzero.fields import measures as MS
    from apexzero.fields import stages as S

    fail = {str(k) for k in failing}
    return [dataclasses.replace(k, value=float("nan"), reason=GATE2_REASON)
            if k.key_id in fail and k.reason not in MS.PIPELINE_REASONS else k
            for k in S.edge_key_measures(rows, opt_rows, measure)]


def holm(pvals: dict[str, float | None], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm's step-down over the family ``pvals`` (m = its size; a test that is not testable,
    p None, enters as p = 1): sorted by p (ties by name), the i-th (from 0) is rejected when
    every earlier one was and p <= alpha / (m - i); the adjusted p is the running maximum of
    min(1, (m - i) p)."""
    m = len(pvals)
    ps = {k: 1.0 if v is None else float(v) for k, v in pvals.items()}
    out, run, going = {}, 0.0, True
    for i, k in enumerate(sorted(ps, key=lambda k: (ps[k], k))):
        level = alpha / (m - i)
        run = max(run, min(1.0, (m - i) * ps[k]))
        going = going and ps[k] <= level
        out[k] = {"p": ps[k], "rank": i + 1, "level": level, "p_adjusted": run,
                  "reject": bool(going), "testable": pvals[k] is not None}
    return out


def _oriented(res: dict, direction: int) -> dict:
    """``analysis.permutation_test``'s result (p: real > baseline) for a test of
    ``direction`` (+1 real > baseline, -1 real < baseline): p is the predicted direction's,
    p_opposite the other's, per stratum too."""
    def flip(d):
        d = dict(d)
        if direction < 0:
            d["p"], d["p_opposite"] = d["p_opposite"], d["p"]
        return d

    out = flip(res)
    out["strata"] = {k: flip(v) for k, v in res["strata"].items()}
    return out


def kept_sets(values: dict[str, float], status: dict[str, str]) -> tuple[dict, dict]:
    """({variant: {arm: [kept networks]}}, {dropped network: why}) over the 20 networks:
    a network is dropped when ``status`` gives a reason (missing, descriptive) or its value is
    missing or not finite."""
    import numpy as np

    kept = {v: {BASE: [], REAL: []} for v in VARIANTS}
    dropped = {}
    for name in NETWORKS:
        v, arm, _ = parse_name(name)
        why = status.get(name)
        if not why and (name not in values or not np.isfinite(float(values[name]))):
            why = "statistic undefined"
        if why:
            dropped[name] = why
        else:
            kept[v][arm].append(name)
    return kept, dropped


def _short(kept: dict) -> list[str]:
    return [f"{v}/{arm}: {len(kept[v][arm])}" for v in VARIANTS for arm in (REAL, BASE)
            if len(kept[v][arm]) < MIN_PER_ARM]


def within_variant_test(values: dict[str, float], kept: dict, direction: int) -> dict:
    """The exact within-variant permutation test (step 2's ``analysis.permutation_test``: T =
    sum over variants of mean(real) - mean(baseline), every joint relabelling of the kept
    sets) in ``direction``; not testable with fewer than MIN_PER_ARM in any arm of either
    variant."""
    n = {v: {arm: len(kept[v][arm]) for arm in (REAL, BASE)} for v in VARIANTS}
    base = {"n": n, "networks": {v: {a: list(x) for a, x in d.items()} for v, d in kept.items()},
            "direction": "real > baseline" if direction > 0 else "real < baseline"}
    short = _short(kept)
    if short:
        return {**base, "testable": False,
                "why": f"fewer than {MIN_PER_ARM} networks remain in {', '.join(short)}"}
    from apexzero.fields import analysis as A

    res = A.permutation_test({v: ([float(values[x]) for x in kept[v][REAL]],
                                  [float(values[x]) for x in kept[v][BASE]]) for v in VARIANTS})
    return {**base, "testable": True, **_oriented(res, direction)}


def primary_decision(test: dict, h: dict) -> str:
    """Confirm when Holm rejects; refute when the opposite one-sided p <= 0.05; else
    inconclusive ('not testable (...)' when the test could not run)."""
    if not test.get("testable"):
        return f"not testable ({test.get('why', '')})"
    if h["reject"]:
        return "confirm"
    if test["p_opposite"] <= ALPHA:
        return "refute"
    return "inconclusive"


def sign_flip_test(diffs, direction: int) -> dict:
    """The paired sign-flip test of the mean within-pair difference (real - baseline): every
    one of the 2^n sign flips; p in ``direction``, p_opposite the other way (ties count)."""
    import numpy as np

    d = np.asarray(diffs, float)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    dist = (signs * d).mean(axis=1)
    obs = float(d.mean())
    eps = 1e-12 * max(1.0, float(np.max(np.abs(dist))))
    up, down = float(np.mean(dist >= obs - eps)), float(np.mean(dist <= obs + eps))
    p, q = (up, down) if direction > 0 else (down, up)
    return {"n_pairs": len(d), "n_flips": len(dist), "mean_diff": obs, "p": p, "p_opposite": q}


def paired_test(values: dict[str, float], kept: dict, direction: int) -> dict:
    """S2: the same seed in both arms of a variant, both kept, forms a pair; the sign-flip
    test of their differences pooled over variants (not testable with fewer than
    MIN_PER_ARM pairs in either variant)."""
    from apexzero.fields import analysis as A

    pairs = {v: [s for s in SEEDS if run_name(v, REAL, s) in kept[v][REAL]
                 and run_name(v, BASE, s) in kept[v][BASE]] for v in VARIANTS}
    diffs = {f"{v}/s{s}": float(values[run_name(v, REAL, s)]) - float(values[run_name(v, BASE, s)])
             for v in VARIANTS for s in pairs[v]}
    base = {"pairs": {v: list(p) for v, p in pairs.items()}, "differences": diffs}
    short = [v for v in VARIANTS if len(pairs[v]) < MIN_PER_ARM]
    if short:
        return {**base, "testable": False,
                "why": f"fewer than {MIN_PER_ARM} complete pairs in {', '.join(short)}"}
    res = sign_flip_test(list(diffs.values()), direction)
    return {**base, "testable": True, **res, "decision": A.decide(res["p"], res["p_opposite"])}


def spearman(x, y) -> float:
    """Spearman's rho (average ranks); NaN with fewer than 3 points or a constant side."""
    import numpy as np
    from scipy import stats

    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def spearman_ci(x, y, n: int, rng) -> dict:
    """rho with its 95% percentile bootstrap interval over points (resampled with
    replacement; a draw with a constant side is undefined and left out, counted)."""
    import numpy as np
    from scipy import stats

    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearman(x, y)
    if len(x) < 3:
        return {"n": len(x), "rho": rho, "lo": float("nan"), "hi": float("nan"),
                "n_undefined": int(n)}
    idx = rng.integers(0, len(x), size=(int(n), len(x)))
    rx, ry = stats.rankdata(x[idx], axis=1), stats.rankdata(y[idx], axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, (rx * ry).sum(axis=1) / den, np.nan)
    ok = np.isfinite(r)
    lo, hi = (np.percentile(r[ok], (2.5, 97.5)) if ok.any() else (np.nan, np.nan))
    return {"n": len(x), "rho": rho, "lo": float(lo), "hi": float(hi),
            "n_undefined": int((~ok).sum())}


def _s3_rng():
    import numpy as np

    from apexzero.interp.constants import ROOT_SEED, STREAM_OFFSET

    return np.random.default_rng([ROOT_SEED, STREAM_OFFSET + S3_STREAM])


def margin_story(b: dict[str, float], completion: dict[str, int], kept: dict, n: int) -> dict:
    """S3 (descriptive): within each arm, Spearman's rho of a network's median B and its
    circuits completed, both variants pooled and per variant, with bootstrap CIs."""
    rng = _s3_rng()
    out = {}
    for arm in (REAL, BASE):
        per = {}
        for label, vs in (("pooled", tuple(VARIANTS)), *((v, (v,)) for v in VARIANTS)):
            names = [x for v in vs for x in kept[v][arm] if completion.get(x) is not None]
            per[label] = {**spearman_ci([b[x] for x in names], [completion[x] for x in names],
                                        n, rng), "networks": names}
        out[arm] = per
    return out


def regime_breakdown(rows: list[dict], opt_rows: dict, failing, cells: dict | None,
                     n: int) -> dict:
    """A network's median G and B per regime of its variant's seed-0 typing (step 2's record:
    regime -> A2 cell ids; descriptive)."""
    from apexzero.fields import stages as S

    out = {}
    for reg, cids in sorted((cells or {}).items()):
        cs = {int(c) for c in cids}
        sub = [r for r in rows if int(r["cell"]) in cs]
        km = {m: h4r_key_measures(sub, opt_rows, m, failing) for m in ("G", "B")}
        out[reg] = {"n_keys": len(km["G"]),
                    **{m: S.median_statistic(v, n) if v else float("nan")
                       for m, v in km.items()}}
    return out


def step2_pooled(d: dict | None) -> dict:
    """Step 2's 18 H4 networks from its committed confirmatory decisions (source 'step2'):
    median G (its H4 statistic), B and T, variant, arm, and step 2's descriptive label."""
    if d is None:
        return {}
    from apexzero.fields import analysis as A

    out = {}
    for v, spec in A.H4_VARIANTS.items():
        for arm, names in ((REAL, spec["real"]), (BASE, spec["baseline"])):
            for name in names:
                ed = (d.get("edge_descriptive") or {}).get(name) or {}
                out[name] = {"source": "step2", "variant": v, "arm": arm,
                             "G": (d.get("statistics") or {}).get(name),
                             "B": ((ed.get("B") or {}).get("pooled") or {}).get("median"),
                             "T": ((ed.get("T") or {}).get("pooled") or {}).get("median"),
                             "descriptive": (d.get("descriptive") or {}).get(name, "")}
    return out


def _group_medians(rows: dict[str, dict]) -> dict:
    """The median of each measure per source / variant / arm (descriptive; NaN when no
    network of the group has a finite value)."""
    import numpy as np

    groups: dict = {}
    for r in rows.values():
        g = groups.setdefault(f"{r['source']}/{r['variant']}/{r['arm']}", {})
        for m in ("G", "B", "T"):
            g.setdefault(m, []).append(np.nan if r.get(m) is None else float(r[m]))
    out = {}
    for k, g in sorted(groups.items()):
        out[k] = {"n": len(g["G"])}
        for m, v in g.items():
            a = np.asarray(v, float)
            out[k][m] = float(np.median(a[np.isfinite(a)])) if np.isfinite(a).any() \
                else float("nan")
    return out


def analyse_h4r(net_rows: dict[str, list[dict]], opt_rows: dict[str, dict],
                failing: dict[str, list[str]], completion: dict[str, int],
                missing: dict[str, str], regime_cells: dict | None = None,
                step2: dict | None = None, n: int | None = None) -> dict:
    """H4R's decisions and descriptive tables (module docstring) from the present networks'
    fields rows, step 2's optimum rows, each network's failing gate2-trained keys, the
    circuits each present network completed and the missing networks."""
    import numpy as np

    from apexzero.fields import analysis as A
    from apexzero.fields import measures as MS
    from apexzero.fields import stages as S

    n = int(n if n is not None else (BOOTSTRAP_N or MS.BOOTSTRAP_N))
    nets, stats, status = {}, {"G": {}, "B": {}, "T": {}}, {}
    for name in NETWORKS:
        if name in missing:
            status[name] = f"missing: {missing[name]}"
            continue
        if name not in net_rows:
            raise ValueError(f"{name} is neither missing nor given rows")
        v, arm, seed = parse_name(name)
        rows, fail = net_rows[name], list(failing.get(name, ()))
        km = {m: h4r_key_measures(rows, opt_rows, m, fail) for m in ("G", "B", "T")}
        cov = MS.coverage(A.coverage_records(km["G"], name, "final", "G"))[f"{name}@final"]["G"]
        cov["gate2_keys"] = sorted(k.key_id for k in km["G"] if k.reason == GATE2_REASON)
        why = ("gate2-trained" if len(fail) > GATE2_MAX_FAILING
               else "coverage" if cov["descriptive"] else "")
        if why:
            status[name] = f"descriptive ({why})"
        for m, per in stats.items():
            per[name] = S.median_statistic(km[m], n)
        t = A.summarise(km["T"], None, n)
        t["label"] = A.t_label(t["pooled"]["lo"], t["pooled"]["hi"])
        nets[name] = {"variant": v, "arm": arm, "seed": seed, "descriptive": why,
                      "gate2": {"failing_keys": sorted(fail), "n_failing": len(fail),
                                "descriptive": len(fail) > GATE2_MAX_FAILING},
                      "coverage": cov, "statistics": {m: stats[m][name] for m in stats},
                      "completed": completion.get(name),
                      "G": A.summarise(km["G"], None, n), "B": A.summarise(km["B"], None, n),
                      "T": t, "n_rows": len(rows), "n_keys": len(km["G"]),
                      "near_ties": S.near_tie_counts(rows),
                      "regimes": regime_breakdown(rows, opt_rows, fail,
                                                  (regime_cells or {}).get(v), n)}
    tests, kept = {}, {}
    for key, (m, direction) in PRIMARY.items():
        kept[m], dropped = kept_sets(stats[m], status)
        tests[key] = {"measure": m, "dropped": dropped,
                      **within_variant_test(stats[m], kept[m], direction)}
    hm = holm({k: t["p"] if t.get("testable") else None for k, t in tests.items()})
    for k, t in tests.items():
        t["holm"] = hm[k]
        t["decision"] = primary_decision(t, hm[k])
    # S1: completion on every present network (the field exclusions do not apply)
    comp = {k: float(c) for k, c in completion.items() if c is not None}
    k1, d1 = kept_sets(comp, {x: f"missing: {missing[x]}" for x in missing})
    s1 = {"measure": "circuits completed", "dropped": d1, "values": comp,
          **within_variant_test(comp, k1, 1)}
    if s1.get("testable"):
        s1["decision"] = A.decide(s1["p"], s1["p_opposite"])
    s2 = {m: paired_test(stats[m], kept[m], d) for m, d in PRIMARY.values()}
    s3 = margin_story(stats["B"], completion, kept["B"], n)
    pooled = {**step2_pooled(step2),
              **{name: {"source": "h4r", "variant": nets[name]["variant"],
                        "arm": nets[name]["arm"], **{m: stats[m][name] for m in stats},
                        "descriptive": nets[name]["descriptive"]} for name in nets}}
    return {"h4r": True, "networks": nets, "statistics": stats,
            "status": {name: status.get(name, "kept") for name in NETWORKS},
            "missing": dict(missing), "H4R_G": tests["H4R-G"], "H4R_B": tests["H4R-B"],
            "holm": {"alpha": ALPHA, "m": len(PRIMARY), "tests": hm},
            "secondary": {"S1": s1, "S2": s2, "S3": s3},
            "descriptive": {"pooled_with_step2": {"networks": pooled,
                                                  "groups": _group_medians(pooled)},
                            "optimum": S.optimum_edges(opt_rows, n)},
            "n_bootstrap": n,
            "finite": {m: int(sum(np.isfinite(float(x)) for x in stats[m].values()))
                       for m in stats}}


def _inputs(rec: dict, root: Path) -> tuple[list[Path], list[str]]:
    """The stamped files the analysis reads under ``root`` (fields rows, harness runs,
    gate2-trained) and why they are not complete: every present network's rows for exactly
    the frozen A2 keys from its recorded checkpoint, with its checks and the arrays its rows
    bind; its harness runs, of its recorded checkpoint; its gate2-trained result on the
    record's keys from its recorded checkpoint with a verdict."""
    from apexzero.fields import stages as S

    a2 = {k for _, k in rec["keys"]["a2"]["ids"]}
    g2want = sorted(k for _, k in rec["keys"]["gate2_trained"]["ids"])
    paths, out = [], []
    for name in NETWORKS:
        if name not in rec["networks"]:
            continue
        pin = rec["networks"][name]
        f = root / "fields" / f"{_file(name)}.jsonl"
        hp = F.harness_path(root, name)
        for p in (f, hp):
            if not p.exists():
                out.append(f"{name}: no {p.parent.name}/{p.name}")
            else:
                paths.append(p)
        if hp.exists() and _harness_meta(hp).get("ckpt_sha256") != pin["sha256"]:
            out.append(f"{name}: {hp.name} is not this checkpoint's harness reference")
        if f.exists():
            rows = S.read_rows(f)
            out += S.rows_problems(rows, a2, name, pin, checked=True)
            out += S.arrays_problems(rows, S.arrays_dir(root / "fields" / "arrays",
                                                        S.expected_net(name, pin)), name)
    g2 = root / "gate2-trained" / "gate2_trained.json"
    if not g2.exists():
        out.append("no gate2-trained results")
    else:
        paths.append(g2)
        d = json.loads(g2.read_text())
        if sorted(d.get("keys", [])) != g2want:
            out.append("gate2-trained: its keys are not the record's")
        for name in rec["networks"]:
            v, pin = (d.get("networks") or {}).get(name), rec["networks"][name]
            if v is None:
                out.append(f"gate2-trained lacks {name}")
            elif (v.get("net") != S.expected_net(name, pin) or v.get("ckpt_sha256") != pin["sha256"]
                  or sorted(str(r.get("key_id")) for r in v.get("rows", [])) != g2want
                  or not all(isinstance(r.get("pass"), bool) for r in v.get("rows", []))):
                out.append(f"gate2-trained: {name} is not from its recorded checkpoint on the "
                           "record's keys with a verdict per key")
    return paths, out


def _harness_meta(path: Path) -> dict:
    """A harness file's meta (``network.save_det``'s meta_json) without reading its arrays;
    {} when it has none or cannot be read."""
    import zipfile

    import numpy as np

    try:
        with np.load(Path(path), allow_pickle=False) as d:
            return json.loads(str(d["meta_json"])) if "meta_json" in d.files else {}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return {}


def _stamp_extra_problems(paths: list[Path]) -> list[str]:
    """This script's sha256 in each input's stamp (step 2's check covers the rest)."""
    from apexzero.fields import prereg

    out = []
    for p in paths:
        st = prereg.read_stamp(p) or {}
        if st.get("h4r_sha256") != _prov().h4r_sha256:
            out.append(f"{p.name}: written by another scripts/h4r.py")
    return out


def stage_analyse(args) -> None:
    """The run-once analysis (module docstring) into outputs/h4r/analyse/decisions.json."""
    rec = _post_tag(args, "the analysis")
    from apexzero.fields import stages as S

    fixed = OUT / "analyse" / "decisions.json"
    ro = F.run_once_problems(fixed)
    if ro:
        raise SystemExit(f"refusing to run the H4R analysis again (it runs once): {'; '.join(ro)}")
    paths, probs = _inputs(rec, OUT)
    probs += F.input_stamp_problems(paths, _prov().code_hash, None, _prov().tag,
                                    rec.get("environment"))
    probs += _stamp_extra_problems(paths)
    opt_path = STEP2_OUT / "opt-fields" / "rows.jsonl"
    if not opt_path.exists() or _sha_file(opt_path) != (rec.get("opt_fields") or {}).get(
            "rows_sha256"):
        probs.append("step 2's opt-fields rows are not the recorded ones")
    s2d = STEP2_OUT / "analyse" / "decisions.json"
    if not s2d.exists() or _sha_file(s2d) != rec["step2"]["decisions"]["sha256"]:
        probs.append("step 2's decisions.json is not its committed confirmatory file")
    completion = {}
    for name, pin in rec["networks"].items():
        e = ROOT / pin["eval_path"]
        if not e.exists() or _sha_file(e) != pin["eval_sha256"]:
            probs.append(f"{name}: {pin['eval_path']} is not the recorded evaluation")
            continue
        completion[name] = completed(json.loads(e.read_text()))[0]
    a2 = {k for _, k in rec["keys"]["a2"]["ids"]}
    opt = {}
    if opt_path.exists():
        opt = {r["key_id"]: r for r in S.read_rows(opt_path) if r["key_id"] in a2}
        miss = sorted(a2 - set(opt))
        if miss:
            probs.append(f"{len(miss)} frozen A2 keys without an optimum row (e.g. {miss[:3]})")
    if probs:
        raise SystemExit("the H4R analysis runs once, on complete recorded inputs: "
                         + "; ".join(probs[:20]))
    if args.check:
        print(f"analyse --check: the inputs of {len(rec['networks'])} networks are complete "
              "and stamped; nothing written")
        return
    g2 = json.loads((OUT / "gate2-trained" / "gate2_trained.json").read_text())
    failing = {name: gate_verdict(v)["failing_keys"] for name, v in g2["networks"].items()
               if name in rec["networks"]}
    net_rows = {name: S.read_rows(OUT / "fields" / f"{_file(name)}.jsonl")
                for name in rec["networks"]}
    res = analyse_h4r(net_rows, opt, failing, completion, rec.get("missing", {}),
                      rec.get("regime_cells"), json.loads(s2d.read_text()))
    res["confirmatory"] = True
    res["inputs"] = {str(p.relative_to(OUT)): _sha_file(p) for p in paths}
    res["inputs"]["step2:opt-fields/rows.jsonl"] = _sha_file(opt_path)
    res["inputs"]["step2:analyse/decisions.json"] = _sha_file(s2d)
    res["record_sha256"] = _sha_file(ROOT / PREREG_REL) if (ROOT / PREREG_REL).exists() else None
    tp = ROOT / pin_rel(TAG)
    res["tag_pin"] = json.loads(tp.read_text()) if tp.exists() else None
    if F.run_once_problems(fixed):                 # written once (another run got there)
        raise SystemExit(f"refusing to overwrite the H4R decisions {fixed}")
    _write(fixed, res)
    sha = _sha_file(fixed)
    for k in ("H4R_G", "H4R_B"):
        t = res[k]
        print(f"{k}: {t['decision']}" + (f" (p {t['p']:.3g}, opposite {t['p_opposite']:.3g}, "
                                         f"{t['n_relabellings']} relabellings)"
                                         if t.get("testable") else ""))
    print(f"decisions.json sha256 {sha}")
    rel = F._rel(fixed, ROOT) or str(fixed)
    print(f"commit it now: git add -f {rel} && git commit -m 'H4R: confirmatory decisions "
          f"{sha[:12]}'", flush=True)


# ------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=2, help="XLA intra-op threads")
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in STAGES:
        p = sub.add_parser(name)
        p.add_argument("--log-dir", default=str(ROOT / "runs"),
                       help="where h4r-<stage>.log goes (default runs/)")
        if name in ("harness-f32", "gate2-trained", "fields"):
            p.add_argument("--out", default=str(OUT), help="output root (default outputs/h4r; "
                           "elsewhere: a partial run the analysis never reads)")
            p.add_argument("--network", action="append", default=None,
                           help="a recorded network (repeatable): an exploratory run, only "
                                "with --out outside outputs/h4r")
        if name in ("harness-f32", "fields"):
            p.add_argument("--limit", type=int, default=None,
                           help="this many keys, spread evenly over the list (a partial run)")
        if name == "fields":
            p.add_argument("--force", action="store_true",
                           help="redo networks whose stamped rows are complete")
        if name == "prereg":
            p.add_argument("--accept-missing", action="append", default=None,
                           metavar="RUN=REASON",
                           help="bind one missing run the queue log does not show failing "
                                "twice (repeatable, one per run; the reason, its queue events "
                                "and whether it finished on the box go into the record as a "
                                "design Section 2 deviation)")
        if name == "analyse":
            p.add_argument("--check", action="store_true",
                           help="the input checks only; nothing is computed or written")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _PROV[:] = [Provenance(args, args.stage)]
    if args.stage in PRE_TAG:
        refuse_after_tag(args.stage)
    t0 = time.time()
    globals()["stage_" + args.stage.replace("-", "_")](args)
    print(f"{args.stage}: {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
