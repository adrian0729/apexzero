"""GW-3, the combined width round: its run manifest and its pre-registered analysis
(docs/prereg/gw3-width-round.md; Sections 2 and 8 for the manifest, 3, 5 and 6 for the
analysis).

    nice -n 10 uv run python scripts/gw3_analysis.py manifest [--accept-missing RUN=REASON]
    nice -n 10 uv run python scripts/gw3_analysis.py analyse [--check]

**The tag gate** (``tag_gate``, scripts/lockbox.py's pattern). Both stages raise
PermissionError unless the annotated tag gw3-prereg-v1 is on origin (both refs named in full in
``git ls-remote``), the local tag is the same tag object, HEAD descends from its commit, and the
files this script reads (``GATED``: this script, the run list, the draft, (2a)'s width report,
H4R's frozen harness with its imports, and tracks/corners.py; untracked files included) are the
tagged ones in the working tree, byte for byte (so a file marked assume-unchanged or
skip-worktree is compared too). scripts/round_manifest.py, the shared layer, is not gated, as
later rounds may fix it: the manifest records its sha256 and whether it is the tag's.

**manifest** (Section 8, step 5: on the Mac, once every box's runs are pulled back with
``gpu_sync.sh pull-round`` and its sha256 list with ``gpu_sync.sh <box> pull
gw3-sums-<i>of<K>.txt``). It runs ``round_manifest.py check`` against the tag and takes its
report only when every problem it lists names a run that failed twice (Section 2: the only way
to be missing) or a run named with ``--accept-missing RUN=REASON`` (missing for any other
reason: a deviation, recorded with its reason). For every other run it records the box's
status, lane, GPU and times and the retry record; its config.json (sha256 and args, which must
be the cell's: Section 2's flags and train_dr.py's defaults at the tag, H4R's PPO and env
blocks, no VG block); its eval_rdev.json (sha256, completion per R-dev circuit, circuits
completed, a finite gap on every completed circuit); and its final checkpoint (the last of 16
in ckpt/, 000199884800.pkl, holding 199,884,800 timesteps). Each of the three files' sha256
must equal the box's list. Per box it records the manifest, queue log, status file and sha256
list (each sha256), the box type, GPU model and count, lanes and shard, and the launch commit,
which must be the tag's. It writes outputs/gw3/manifest.json (refused once a manifest was
committed); commit it with git add -f.

**analyse** (Section 8, step 6: once). It refuses if outputs/gw3/analysis.json exists or was
ever committed, and unless the manifest is committed at HEAD and unchanged on disk, of this tag
and of this run list. It reads only the manifest's evaluations, each re-hashed, and checks each
one's s-OCP lap against outputs/ocp/<circuit>.pkl (every name through ``lockbox_guard``) and its
gap against median lap / s-OCP lap - 1. Then it computes Section 3 (P1; S1, S1-v1, S1-v3, S2
and S3; the lap time; the descriptive tables), Section 5 (GW-4's rule, criterion 4 read from
the width report only after its sha256 is checked; #12's fallback; the in-sample label) and
Section 6 (the gate's definition, recorded and not read), and writes outputs/gw3/analysis.json
once, with a provenance block like the width report's. --check runs the input checks only and
writes nothing.

**Counting the exact p** (Section 3). Per variant, a dynamic programme over the integer units
counts the relabellings that give each sum of the treated arm's units (``subset_sum_counts``).
T is linear in the variants' treated sums, so the relabellings with T* >= T (or <= T) are summed
over every combination of sums, with T scaled by the least common multiple of n_treated x
n_baseline over the variants: every comparison is between integers. tests/test_gw3_analysis.py
checks the counts against listing every relabelling. A test is not testable with fewer than 3
units in either arm of any of its variants.

**Readings fixed here where the draft leaves a detail open:**
- the secondary tests take P1's labels (confirm, refute, inconclusive, not testable);
- S3's bootstrap resamples the seeds (20-29) with replacement, both arms of a cell at the same
  drawn seeds (a seed with a missing run is skipped in that arm's mean), 20,000 draws from
  numpy's default_rng(0) for each cell. A cell with no network in an arm has no difference;
  D2 is then undefined and criterion 2 does not hold, as Section 3 has it for Delta-gap;
- the paired lap statistics take, per same-seed (sa, base) pair, the circuits both networks
  complete: the difference of the two networks' median gaps there, and the mean of the
  per-circuit differences (interp-leftovers.md Section 6.2's form). A pair with no common
  circuit is left out and counted;
- criterion 5 is read at the generator-v1 commit, not here, so the rule's decision is recorded
  for both of its outcomes;
- a width report whose sha256 is not the pinned one stops GW-4 (no decision is recorded).
  Section 3's results do not read the report and are written either way.
"""

import argparse
import calendar
import hashlib
import json
import math
import os
import pickle
import platform
import re
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import gw3_runs as RL
import h4r  # H4R's frozen harness (scripts/h4r.py), read-only
import numpy as np
import round_manifest as RM  # the shared layer's run-list and config helpers (stdlib only)

from apexzero.tracks.corners import R_DEV

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()   # the code this process loaded

TAG = "gw3-prereg-v1"
REMOTE = "origin"
ROUND = "gw3"
SCRIPT_REL = "scripts/gw3_analysis.py"
RUNS_REL = "scripts/gw3_runs.py"
DRAFT_REL = "docs/prereg/gw3-width-round.md"
WIDTH_REPORT_REL = "outputs/width-report/report.json"
WIDTH_REPORT_SHA256 = "5871664be5a6ff6ac882e3a38e131660fe16676f37ddd0a99ffba7294a42f014"
CHECK_REL = "scripts/round_manifest.py"
TRAINER_REL = "scripts/train_dr.py"
OUT_REL = "outputs/gw3"
MANIFEST_REL = f"{OUT_REL}/manifest.json"
ANALYSIS_REL = f"{OUT_REL}/analysis.json"
GATED = (SCRIPT_REL, RUNS_REL, DRAFT_REL, WIDTH_REPORT_REL, "scripts/h4r.py", "scripts/fields.py",
         "scripts/_bootstrap.py", "src/apexzero/tracks/corners.py")
STATUS = ("PRE-REGISTERED (gw3-prereg-v1, docs/prereg/gw3-width-round.md): P1 primary; S1, "
          "S1-v1, S1-v3 and S2 secondary (unadjusted); S3, the lap time's extras and the tables "
          "descriptive")

# Section 2
VARIANTS = tuple(RL.VARIANT_FLAGS)                    # v1, v3
ARMS = tuple(RL.ARM_FLAGS)                            # base, cl-f1, cl-f2, sl-f1, sl-f2, sa
SEEDS = RL.SEEDS
FOLD_OF = {"cl-f1": "f1", "cl-f2": "f2", "sl-f1": "f1", "sl-f2": "f2"}
HELD_OUT = {"f1": RL.FOLDS["f2"], "f2": RL.FOLDS["f1"]}   # a fold-f network's held-out circuits
TILT = {v: h4r.VARIANTS[v] for v in VARIANTS}         # v1 (0, 1) and v3 (0.5, 2), as in H4R
FINAL_TIMESTEPS = h4r.FINAL_TIMESTEPS                 # 199,884,800: train_dr.py's last update
FINAL_CKPT = f"{FINAL_TIMESTEPS:012d}.pkl"
N_CHECKPOINTS = 16
VG_LAMBDA, VG_WINSOR = 0.95, 0.005                    # ppo.ValueGradient's defaults at the tag
# config.json's "ppo" block with train_dr.py's defaults (H4R's; tests check it against
# PPOConfig as train_dr.py builds it)
PPO = {"num_envs": 2048, "num_steps": 64, "total_timesteps": 200_000_000, "update_epochs": 4,
       "num_minibatches": 8, "gamma": 0.995, "gae_lambda": 0.95, "clip_eps": 0.2,
       "clip_value": True, "vf_coef": 0.5, "ent_coef": 0.001, "ent_coef_final": 0.0,
       "lr": 0.0003, "anneal_lr": True, "max_grad_norm": 0.5, "normalize_obs": True,
       "normalize_reward": True, "log_std_init": -0.5, "network": "mlp", "actor_width": 256,
       "critic_width": 256, "mlp_layers": 3, "activation": "tanh", "obs_noise": 0.0,
       "ensemble_heads": 5}
# config.json's "env" block: racing.EnvParams()'s defaults, H4R's (tests check it against them)
ENV = {"dt": 0.05, "substeps": 10, "window_steps": 1200, "edge_margin": 1.0, "slow_speed": 2.0,
       "slow_time": 2.0, "action_cost": 0.02, "progress_scale": 10.0, "obs_noise": 0.0,
       "strip_tail": 520.0}

# Section 3
COMPLETION_MIN = h4r.COMPLETION_MIN                   # completed when completion > 0.5
ALPHA = Fraction(5, 100)
MIN_UNITS = 3
MAX_UNIT = len(R_DEV)
TESTS = {"P1": ("sl", "base", VARIANTS), "S1": ("cl", "base", VARIANTS),
         "S1-v1": ("cl", "base", ("v1",)), "S1-v3": ("cl", "base", ("v3",)),
         "S2": ("sa", "base", VARIANTS)}
BOOT_N = 20_000
BOOT_SEED = 0
POINTS = 100.0
GAP_TOL = 1e-12
NAMED = ("Norisring", "YasMarina", "Sochi", "Melbourne")
NOT_THE_GATE_MIN = 10

# Section 5
D2_MIN = Fraction(-1, 2)
DGAP_MAX = 1.0
C4_V3_ALLOWED = ("f1/v3/share_1m", "f2/v3/share_1m", "f1/v3/spectrum:8-16")
C4_N_CHECKS = 90
_VG = "tests/test_value_gradient.py::"
C5_TESTS = ("tests/test_stat_widths.py::test_default_is_bit_identical",
            "tests/test_stat_widths.py::test_default_bank_is_bit_identical",
            "tests/test_stat_widths.py::test_train_dr_default_path_is_unchanged",
            "tests/test_real_widths.py::test_default_is_unchanged",
            "tests/test_reexpress.py::test_default_and_2a_unchanged_and_2b_fit (if (2b) is there)",
            _VG + "test_default_racing_update_reproduces_the_pre_term_values",
            _VG + "test_default_rollout_records_no_capture_and_defaults_are_off",
            _VG + "test_train_dr_default_path_passes_no_term",
            "tests/test_ppo.py::test_make_train_reproduces_the_pre_split_update")
DECISIONS = {
    "adopt": "generator-v1 sets stat_widths (M = (2a), fitted on all 12 R-dev circuits) as E0's "
             "training width model; the default stays the frozen default",
    "copied": "generator-v1 uses copied R-dev width profiles (--real-widths); every R-dev "
              "number for E0 is labelled 'in-sample for widths'",
    "default": "generator-v1 uses the default widths"}
IN_SAMPLE_LABEL = "in-sample for widths"
IN_SAMPLE_COVERS = (
    "the 80% gate's result (#10: the median seed's completed R-dev circuits) and each seed's",
    "per-circuit R-dev completion, lap times and gaps to the s-OCP lap",
    "every comparison or figure that uses E0's R-dev results")

# Section 6 (recorded, not read)
GATE_80 = {
    "read_here": False,
    "rule": "E0 (5 seeds x 1B steps in v1, on generator-v1 and sim-v1) passes if its median "
            "seed completes at least 10 of the 12 R-dev circuits at its final checkpoint",
    "checkpoint": "the final one; no checkpoint is chosen on R-dev",
    "label_if_copied_widths": IN_SAMPLE_LABEL,
    "gw3": "GW-3's 200M runs are not E0; its count of networks with C >= 10 is descriptive and "
           "labelled 'not the gate'"}

if (list(RL.FOLDS["f1"]), list(RL.FOLDS["f2"])) != (R_DEV[0::2], R_DEV[1::2]):
    raise SystemExit("gw3_runs.FOLDS are not R_DEV[0::2] and R_DEV[1::2]")


class CheckFailed(SystemExit):
    """An input check failed: the stage stops and writes nothing."""


# ------------------------------------------------------------------------- git and files


def git_run(*args: str) -> subprocess.CompletedProcess:
    """Every git call goes through here, in ROOT (the tests point ROOT at scratch repos)."""
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def git(*args: str) -> str:
    out = git_run(*args)
    return out.stdout.strip() if out.returncode == 0 else ""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def tag_gate(paths: tuple[str, ...] = GATED) -> dict:
    """PermissionError unless the annotated tag gw3-prereg-v1 is on origin, the local tag is
    the same tag object, HEAD descends from its commit, and ``paths`` (untracked files
    included) in the working tree are the tagged ones byte for byte, each present at the tag.

    The two refs are named in full: ``git ls-remote --tags origin gw3-prereg-v1`` prints only
    the tag line, never the peeled ``^{}`` one, and a ``gw3-prereg-v1*`` glob also matches
    gw3-prereg-v10. The parse below takes exact names only."""
    ls = git_run("ls-remote", REMOTE, f"refs/tags/{TAG}", f"refs/tags/{TAG}^{{}}")
    if ls.returncode != 0:
        raise PermissionError(f"cannot list {REMOTE}'s tags ({ls.stderr.strip()})")
    refs = {}
    for line in ls.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            refs[parts[1]] = parts[0]
    obj, peeled = refs.get(f"refs/tags/{TAG}"), refs.get(f"refs/tags/{TAG}^{{}}")
    if obj is None:
        raise PermissionError(f"{TAG} is not on {REMOTE}: GW-3 is not pre-registered")
    if peeled is None or peeled == obj:
        raise PermissionError(f"{TAG} on {REMOTE} is not an annotated tag")
    local = git_run("rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}")
    if local.returncode != 0 or local.stdout.strip() != obj:
        raise PermissionError(f"the local {TAG} is missing or is not {REMOTE}'s tag object")
    kind = git_run("cat-file", "-t", obj)
    if kind.returncode != 0 or kind.stdout.strip() != "tag":
        raise PermissionError(f"{TAG} ({obj}) is not a tag object")
    commit = git_run("rev-parse", "--verify", "--quiet", f"{obj}^{{commit}}")
    if commit.returncode != 0 or commit.stdout.strip() != peeled:
        raise PermissionError(f"{TAG}'s commit does not match {REMOTE}'s")
    anc = git_run("merge-base", "--is-ancestor", peeled, "HEAD")
    if anc.returncode != 0:
        raise PermissionError(f"HEAD does not descend from {TAG} ({peeled})")
    absent = [p for p in paths if git_run("cat-file", "-e", f"{peeled}:{p}").returncode != 0]
    if absent:
        raise PermissionError(f"not in {TAG}: {', '.join(absent)}")
    diff = git_run("diff", "--name-only", peeled, "--", *paths)
    if diff.returncode != 0 or diff.stdout.strip():
        raise PermissionError(f"these differ from {TAG}'s: "
                              f"{', '.join(diff.stdout.split()) or diff.stderr.strip()}")
    untracked = git_run("ls-files", "--others", "--exclude-standard", "--", *paths)
    if untracked.returncode != 0 or untracked.stdout.strip():
        raise PermissionError(f"untracked files under the tagged paths: {untracked.stdout!r}")
    # git diff does not look at a file marked assume-unchanged or skip-worktree: the bytes too
    unequal = [p for p in paths if not (ROOT / p).is_file()
               or sha256_file(ROOT / p) != sha256_bytes(git_blob(peeled, p) or b"")]
    if unequal:
        raise PermissionError(f"these differ from {TAG}'s on disk: {', '.join(unequal)}")
    return {"tag": TAG, "tag_object": obj, "commit": peeled, "head": git("rev-parse", "HEAD")}


def git_blob(rev: str, rel: str) -> bytes | None:
    """The bytes of ``rel`` in commit ``rev`` (None when it has no such file)."""
    out = subprocess.run(["git", "cat-file", "blob", f"{rev}:{rel}"], cwd=ROOT,
                         capture_output=True, check=False)
    return out.stdout if out.returncode == 0 else None


def ever_committed(rel: str) -> bool:
    return bool(git("log", "--all", "--format=%H", "--", rel))


def write_new(path: Path, data: bytes) -> str:
    """Write ``data`` to a path that must not exist yet (atomic); returns its sha256."""
    path = Path(path)
    if path.exists():
        raise CheckFailed(f"{path.relative_to(ROOT)} exists: it is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    try:
        os.link(tmp, path)        # fails if the path appeared meanwhile
    finally:
        tmp.unlink()
    return sha256_bytes(data)


def clean(x):
    """JSON-safe: NaN as null, fractions as floats, NumPy values as Python ones."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, Fraction):
        return float(x)
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def dumps(obj) -> bytes:
    return (json.dumps(clean(obj), indent=1, allow_nan=False) + "\n").encode()


def provenance(start: float, gate: dict, inputs: dict[str, str]) -> dict:
    """Like the width report's: the plan (the draft) and tag, this script at load and now (they
    must agree), HEAD, the changed paths, every input's sha256, argv, times, environment."""
    now = sha256_file(SCRIPT)
    if now != SCRIPT_SHA256:
        raise CheckFailed(f"{SCRIPT_REL} changed while it ran: nothing is written")
    return {
        "label": STATUS, "plan": DRAFT_REL, "plan_sha256": sha256_file(ROOT / DRAFT_REL),
        "tag": gate["tag"], "tag_object": gate["tag_object"], "tag_commit": gate["commit"],
        "script": SCRIPT_REL, "script_sha256": SCRIPT_SHA256,
        "runs_script_sha256": sha256_file(ROOT / RUNS_REL),
        "git_head": git("rev-parse", "HEAD"),
        "changed_paths": [ln for ln in git_run("status", "--porcelain", "--untracked-files=no")
                          .stdout.splitlines() if ln.strip()],
        "inputs_sha256": dict(sorted(inputs.items())), "argv": sys.argv[1:],
        "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
        "end": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - start, 1),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "machine": platform.machine(), "system": platform.system(),
                        "nice": os.getpriority(os.PRIO_PROCESS, 0)}}


# ------------------------------------------------------------------------- the manifest


def parse_name(name: str) -> tuple[str, str, int]:
    """(variant, arm, seed) of a GW-3 run name."""
    m = re.fullmatch(r"gw3-(v1|v3)-(base|cl-f1|cl-f2|sl-f1|sl-f2|sa)-s(\d+)", name)
    if m is None:
        raise ValueError(f"{name} is not a GW-3 run name")
    return m[1], m[2], int(m[3])


def trainer_defaults(source: str) -> dict:
    """train_dr.py's argument defaults, from its parser rebuilt from ``source`` (the tag's, by
    round_manifest.py's ``trainer_parser``, as its check rebuilds it). The two defaults it
    computes, --vg-lambda and --vg-winsor, are ppo.ValueGradient's (VG_LAMBDA, VG_WINSOR)."""
    out = vars(RM.trainer_parser(source).parse_args([]))
    computed = {"vg_lambda": VG_LAMBDA, "vg_winsor": VG_WINSOR}
    for k, v in out.items():
        if v is RM.COMPUTED:
            if k not in computed:
                raise CheckFailed(f"train_dr.py computes the default of --{k}: not checkable")
            out[k] = computed[k]
    return out


def design_args(variant: str, arm: str, seed: int) -> dict:
    """The train_dr.py arguments Section 2 fixes for one run: ``--steps 200e6 --seed s``, the
    variant's tilt, the arm's width option (a fold's circuits, comma-separated, for a leave-out
    arm; no --reexpress), no VG term, and H4R's defaults (2,048 envs, 64 steps, MLP, bank 4,096,
    refresh 256, randomised car, 5 heads)."""
    fold = ",".join(RL.FOLDS[FOLD_OF[arm]]) if arm in FOLD_OF else ""
    copied, stat = arm.startswith("cl-"), arm.startswith("sl-") or arm == "sa"
    return {"steps": float(RL.STEPS), "name": RL.run_name(variant, arm, seed),
            "num_envs": 2048, "num_steps": 64, "network": "mlp", "bank": 4096, "refresh": 256,
            "fixed_car": False, **TILT[variant], "real_widths": copied,
            "real_widths_from": fold if copied else "", "stat_widths": stat,
            "stat_widths_from": fold if stat else "", "reexpress": False, "vg_coef": 0.0,
            "ensemble_heads": 5, "seed": int(seed)}


def expected_args(variant: str, arm: str, seed: int, defaults: dict) -> dict:
    """One run's full args: Section 2's (``design_args``) over every other argument at
    train_dr.py's default at the tag (``defaults``, from ``trainer_defaults``)."""
    return {**defaults, **design_args(variant, arm, seed)}


def config_problems(cfg: dict, variant: str, arm: str, seed: int, defaults: dict) -> list[str]:
    """How a run's config.json departs from its cell (Section 2): its args (``expected_args``),
    its PPO block (``PPO``) and env block (``ENV``), H4R's, no VG block (train_dr.py writes one
    only with --vg-coef > 0) and no other block than ppo, args and env; [] when it matches."""
    out = []
    if set(cfg) != {"ppo", "args", "env"}:
        out.append(f"config blocks {sorted(cfg)}, not ['args', 'env', 'ppo']")
    out += h4r._diff("args", cfg.get("args") or {}, expected_args(variant, arm, seed, defaults))
    out += h4r._diff("ppo", cfg.get("ppo") or {}, PPO)
    out += h4r._diff("env", cfg.get("env") or {}, ENV)
    return out


def parse_sums(text: str) -> dict[str, str]:
    """{path under runs/: sha256} from ``sha256sum``'s lines ('<hex>  <path>', or '<hex>
    *<path>' in binary mode)."""
    out = {}
    for k, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        m = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line.rstrip("\n"))
        if m is None:
            raise CheckFailed(f"line {k} of a sha256 list is not '<sha256>  <path>': {line!r}")
        out[m[2]] = m[1]
    return out


def eval_summary(ev: dict) -> tuple[dict, list[str]]:
    """An eval_rdev.json's completion per circuit and circuits completed (H4R's rule,
    ``h4r.completed``), and what it lacks: absent circuits, completed circuits without a
    finite gap."""
    k, comp, absent = h4r.completed(ev)
    no_gap = [c for c, x in comp.items() if x > COMPLETION_MIN
              and not math.isfinite(float((ev[c] or {}).get("gap", float("nan"))))]
    return {"completion": comp, "completed": k}, [*(f"lacks {c}" for c in absent),
                                                  *(f"{c} is completed but has no gap (no "
                                                    f"outputs/ocp/{c}.pkl on the box?)"
                                                    for c in no_gap)]


def box_manifests() -> list[tuple[str, dict]]:
    """The shard manifests round_manifest.py wrote on the boxes (runs/gw3-manifest-<i>of<K>)."""
    pat = re.compile(rf"{ROUND}-manifest-(\d+)of(\d+)\.json")
    out = []
    for p in sorted((ROOT / "runs").glob(f"{ROUND}-manifest-*of*.json")):
        if pat.fullmatch(p.name):
            out.append((f"runs/{p.name}", json.loads(p.read_text())))
    return out


def run_check(runs_args: list[str]) -> tuple[int, str]:
    """``round_manifest.py check`` against the tag: (return code, its report)."""
    cmd = [sys.executable, CHECK_REL, "check", "--round", ROUND, "--runs-script", RUNS_REL,
           "--tag", TAG, *(["--runs-args", *runs_args] if runs_args else [])]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return p.returncode, p.stdout + p.stderr


def check_problems(rc: int, report: str) -> list[str]:
    """The problems ``round_manifest.py check`` listed ([] when it passed); CheckFailed when it
    stopped before checking (a bad tag or run list)."""
    if rc == 0:
        return []
    lines = report.splitlines()
    starts = [k for k, ln in enumerate(lines) if ln.startswith("FAILED: ")]
    if rc != 1 or not starts:
        raise CheckFailed(f"round_manifest.py check stopped (exit {rc}):\n{report.strip()}")
    return [ln[2:] for ln in lines[starts[-1] + 1:] if ln.startswith("  ")]


def parse_accept(items: list[str] | None) -> dict[str, str]:
    """{run: reason} from --accept-missing RUN=REASON items (RUN with or without 'runs/')."""
    names = {r["name"] for r in RL.runs()}
    out: dict[str, str] = {}
    for it in items or []:
        run, sep, why = str(it).partition("=")
        run = run.strip().removeprefix("runs/")
        if not sep or not why.strip():
            raise SystemExit(f"--accept-missing {it!r}: give RUN=REASON (a non-empty reason)")
        if run not in names:
            raise SystemExit(f"--accept-missing {it!r}: {run} is not a GW-3 run")
        if run in out:
            raise SystemExit(f"--accept-missing names {run} twice")
        out[run] = why.strip()
    return out


def run_entry(run: dict, rec: dict | None, sums: dict[str, str] | None, timesteps_of,
              defaults: dict) -> tuple[dict, str | None, list[str]]:
    """(entry, why the run is incomplete here or None, problems that refuse the manifest) for
    one run: the box's record, the config, the evaluation and the final checkpoint, each file's
    sha256 against the box's list."""
    name = run["name"]
    v, arm, seed = parse_name(name)
    d = ROOT / "runs" / name
    ent: dict = {"variant": v, "arm": arm, "seed": seed}
    if rec is not None:
        ent.update({k: rec.get(k) for k in ("status", "shard", "lane", "gpu", "start", "end")})
    ent["first_attempt_log"] = (ROOT / "runs" / f"{name}.log.1").exists()
    if not d.is_dir():
        return ent, "no run directory", []
    probs = []
    cp = d / "config.json"
    if not cp.exists():
        return ent, "no config.json", []
    cfg = json.loads(cp.read_text())
    ent.update(config_path=f"runs/{name}/config.json", config_sha256=sha256_file(cp),
               args=cfg.get("args"))
    probs += [f"config: {p}" for p in config_problems(cfg, v, arm, seed, defaults)]
    ep = d / "eval_rdev.json"
    if not ep.exists():
        return ent, "no eval_rdev.json (training did not finish)", probs
    summary, lacks = eval_summary(json.loads(ep.read_text()))
    ent.update(eval_path=f"runs/{name}/eval_rdev.json", eval_sha256=sha256_file(ep), **summary)
    if lacks:
        return ent, f"eval_rdev.json: {'; '.join(lacks)}", probs
    ckpts = sorted((d / "ckpt").glob("*.pkl"))
    if not ckpts:
        return ent, "no checkpoint", probs
    final = ckpts[-1]
    ts = int(timesteps_of(final, name))
    ent.update(checkpoint_path=f"runs/{name}/ckpt/{final.name}",
               checkpoint_sha256=sha256_file(final), timesteps=ts, n_checkpoints=len(ckpts))
    if final.name != FINAL_CKPT or ts != FINAL_TIMESTEPS:
        return ent, (f"the last checkpoint {final.name} holds {ts} timesteps, not "
                     f"{FINAL_TIMESTEPS} (training did not finish)"), probs
    if len(ckpts) != N_CHECKPOINTS:
        probs.append(f"{len(ckpts)} checkpoints, not {N_CHECKPOINTS}")
    if sums is None:
        probs.append("no sha256 list from its box")
    else:
        for key, rel in (("config_sha256", "config.json"), ("eval_sha256", "eval_rdev.json"),
                         ("checkpoint_sha256", f"ckpt/{FINAL_CKPT}")):
            box = sums.get(f"{name}/{rel}")
            if box != ent[key]:
                probs.append(f"{rel}: sha256 {ent[key][:12]} here, "
                             f"{box[:12] if box else 'none'} on the box")
    return ent, None, probs


def _queue_hours(records: list[dict]) -> float | None:
    """The box's queue time (first start to last end, UTC), as round_manifest.py check counts
    it for the stop rule."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        starts = [calendar.timegm(time.strptime(r["start"], fmt)) for r in records
                  if r.get("start")]
        ends = [calendar.timegm(time.strptime(r["end"], fmt)) for r in records if r.get("end")]
    except (TypeError, ValueError):
        return None
    return (max(ends) - min(starts)) / 3600 if starts and ends else None


def build_manifest(gate: dict, accept: dict[str, str] | None = None, check=None,
                   timesteps_of=None) -> tuple[dict, list[str]]:
    """(the manifest, the problems that refuse it). ``check(runs_args)`` runs
    round_manifest.py's check (default ``run_check``); ``timesteps_of(path, name)`` reads a
    checkpoint's timesteps (default H4R's loader, ``h4r._pickle_timesteps``)."""
    check = run_check if check is None else check
    timesteps_of = h4r._pickle_timesteps if timesteps_of is None else timesteps_of
    accept = dict(accept or {})
    runs = RL.runs()
    boxes, problems, records = [], [], {}
    found = box_manifests()
    if not found:
        problems.append(f"no runs/{ROUND}-manifest-<i>of<K>.json here (pull-round each box)")
    runs_args = {json.dumps(m.get("runs_args")) for _, m in found}
    if len(runs_args) > 1:
        problems.append(f"the boxes ran the run list with different options: {sorted(runs_args)}")
    args = json.loads(next(iter(runs_args))) if len(runs_args) == 1 else []
    sums_by_shard = {}
    for path, m in found:
        i, k = (int(x) for x in str(m.get("shard")).split("/"))
        tag = f"{i}of{k}"
        files = {"manifest": path, "queue_log": f"runs/{ROUND}-queue-{tag}.log",
                 "status": f"runs/{ROUND}-status-{tag}.tsv", "sums": f"runs/{ROUND}-sums-{tag}.txt"}
        box = {"shard": m.get("shard"), **{k_: m.get(k_) for k_ in (
            "box_type", "gpu_model", "gpu_count", "gpus", "lanes_per_gpu", "xla_mem_fraction",
            "head", "tree_clean", "written", "run_list_sha256")}}
        for key, rel in files.items():
            p = ROOT / rel
            box[key] = rel
            box[f"{key}_sha256"] = sha256_file(p) if p.exists() else None
            if not p.exists():
                problems.append(f"{rel} is not here"
                                + (f" (scripts/gpu_sync.sh <box> pull {p.name})"
                                   if key == "sums" else ""))
        if (ROOT / files["sums"]).exists():
            sums_by_shard[m.get("shard")] = parse_sums((ROOT / files["sums"]).read_text())
        hours = _queue_hours(m.get("runs") or [])
        box["queue_hours"] = hours
        box["gpu_hours"] = (hours * m["gpu_count"] if hours is not None
                            and isinstance(m.get("gpu_count"), int) else None)
        if m.get("head") != gate["commit"]:
            problems.append(f"{path}: HEAD {str(m.get('head'))[:12]} is not {TAG}'s commit")
        boxes.append(box)
        for rec in m.get("runs") or []:
            records[rec.get("name")] = rec
    rc, report = check(args)
    listed = check_problems(rc, report)
    show = git_run("show", f"{gate['commit']}:{TRAINER_REL}")
    if show.returncode != 0:
        raise CheckFailed(f"{TRAINER_REL} is not in {TAG}")
    defaults = trainer_defaults(show.stdout)
    entries, missing, incomplete, tolerated = {}, {}, {}, []
    for run in runs:
        name = run["name"]
        rec = records.get(name)
        status = rec.get("status") if rec else "in no manifest"
        ent, why, probs = run_entry(run, rec, _box_sums(name, found, sums_by_shard),
                                    timesteps_of, defaults)
        ent["status"], ent["retried"] = status, status == "retried"
        entries[name] = ent
        if status == "failed":
            missing[name] = f"failed twice in the queue ({_status_file(name, found)})"
        elif status not in ("done", "retried"):
            incomplete[name] = f"status {status!r} on the box" + (f"; here: {why}" if why else "")
        elif why is not None:
            incomplete[name] = f"{status} on the box, but here: {why}"
        else:
            problems += [f"{name}: {p}" for p in probs]
    failed = list(missing)
    for name, why in incomplete.items():
        if name in accept:
            missing[name] = f"accepted as missing ({accept[name]}); here: {why}"
        else:
            problems.append(f"{name}: {why} (pull it back and re-run the manifest stage, or "
                            "name it with --accept-missing RUN=REASON: a Section 2 deviation)")
    stale = sorted(n for n in accept if n not in incomplete)
    if stale:
        problems.append(f"--accept-missing names runs that are not missing for another reason "
                        f"than failing twice: {stale}")
    for p in listed:
        owner = next((n for n in missing if p.startswith(f"{n}: ")), None)
        if owner is None:
            problems.append(f"round_manifest.py check: {p}")
        else:
            tolerated.append(p)
    present = [r["name"] for r in runs if r["name"] not in missing | incomplete]
    man = {"round": ROUND, "draft": DRAFT_REL, "tag": {k: gate[k] for k in
                                                       ("tag", "tag_object", "commit")},
           "launch_commit": sorted({b["head"] for b in boxes if b["head"]}),
           "runs_script": RUNS_REL, "runs_args": args,
           "run_list_sha256": run_list_sha256(runs),
           "check": {"script": CHECK_REL, "returncode": rc, "report": report,
                     "tolerated": tolerated},
           "boxes": boxes, "final_timesteps": FINAL_TIMESTEPS, "final_checkpoint": FINAL_CKPT,
           "trainer_defaults": defaults, "ppo": PPO, "env": ENV,
           "runs": entries, "present": present, "missing": missing, "failed_twice": failed,
           "accepted_missing": {n: accept[n] for n in accept if n in missing},
           "retries": {n: {"first_attempt_log": entries[n]["first_attempt_log"]}
                       for n in entries if entries[n].get("retried")}}
    return man, problems


def _box_sums(name: str, found: list[tuple[str, dict]], sums_by_shard: dict) -> dict | None:
    """The sha256 list of the box whose manifest records ``name``."""
    for _, m in found:
        if any(r.get("name") == name for r in m.get("runs") or []):
            return sums_by_shard.get(m.get("shard"))
    return None


def _status_file(name: str, found: list[tuple[str, dict]]) -> str:
    for _, m in found:
        if any(r.get("name") == name for r in m.get("runs") or []):
            i, k = str(m.get("shard")).split("/")
            return f"runs/{ROUND}-status-{i}of{k}.tsv"
    return "no manifest"


def run_list_sha256(runs: list[dict]) -> str:
    """round_manifest.py's run_list_sha256 (the same text, hashed)."""
    return sha256_bytes(json.dumps(runs, sort_keys=True, separators=(",", ":")).encode())


def stage_manifest(args) -> None:
    gate = tag_gate()
    if ever_committed(MANIFEST_REL):
        raise CheckFailed(f"{MANIFEST_REL} was committed: the manifest is not rebuilt")
    start = time.time()
    man, problems = build_manifest(gate, parse_accept(args.accept_missing))
    print(f"manifest: {len(man['present'])} of {len(man['runs'])} runs present, "
          f"{len(man['missing'])} missing, {len(man['retries'])} retried")
    for n, why in man["missing"].items():
        print(f"  missing {n}: {why}")
    if problems:
        print(f"REFUSED: {len(problems)} problem(s)")
        for p in problems:
            print(f"  {p}")
        raise CheckFailed(1)
    man["provenance"] = provenance(start, gate, {
        CHECK_REL: sha256_file(ROOT / CHECK_REL),
        **{b[k]: b[f"{k}_sha256"] for b in man["boxes"]
           for k in ("manifest", "queue_log", "status", "sums")}})
    man["provenance"]["check_script_is_tagged"] = not git_run(
        "diff", "--quiet", gate["commit"], "--", CHECK_REL).returncode
    out = ROOT / MANIFEST_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(dumps(man))
    print(f"wrote {MANIFEST_REL} ({sha256_file(out)[:12]}); next: git add -f {MANIFEST_REL} "
          "&& git commit -m 'GW-3: run manifest'")


# ------------------------------------------------------------------------- the exact test


def subset_sum_counts(values: list[int], k: int) -> dict[int, int]:
    """{s: the number of k-element subsets of ``values`` (non-negative integers) summing to s}."""
    table: list[dict[int, int]] = [{} for _ in range(k + 1)]
    table[0][0] = 1
    for v in values:
        for j in range(k, 0, -1):       # downwards, so each value enters a subset once
            cur = table[j]
            for s, c in table[j - 1].items():
                cur[s + v] = cur.get(s + v, 0) + c
    return table[k]


def exact_counts(groups: dict) -> dict:
    """The exact within-variant permutation test of T = sum over variants of mean(treated) -
    mean(baseline), counted. ``groups`` maps a variant to (treated units, baseline units),
    integers 0..12. Returns the relabellings with T* >= T (``ge``), with T* <= T (``le``), all
    of them (``total``), and T as a fraction. Scaled by L = lcm over variants of n_t * n_b,
    T = sum_v [S_t,v * L (1/n_t + 1/n_b) - L * (S_t,v + S_b,v) / n_b], an integer."""
    scale = 1
    for t, b in groups.values():
        scale = math.lcm(scale, len(t) * len(b))
    dist, obs = {0: 1}, 0
    for t, b in groups.values():
        t, b = [int(x) for x in t], [int(x) for x in b]
        nt, nb = len(t), len(b)
        weight = scale * (nt + nb) // (nt * nb)
        offset = scale // nb * (sum(t) + sum(b))
        obs += sum(t) * weight - offset
        counts = subset_sum_counts(t + b, nt)
        new: dict[int, int] = {}
        for a, ca in dist.items():
            for s, cs in counts.items():
                key = a + s * weight - offset
                new[key] = new.get(key, 0) + ca * cs
        dist = new
    total = sum(dist.values())
    if total != math.prod(math.comb(len(t) + len(b), len(t)) for t, b in groups.values()):
        raise AssertionError("the relabellings counted are not all of them")
    return {"ge": sum(c for key, c in dist.items() if key >= obs),
            "le": sum(c for key, c in dist.items() if key <= obs), "total": total,
            "T": Fraction(obs, scale)}


def permutation_test(units: dict, treated: str, baseline: str, variants) -> dict:
    """One of Section 3's tests on the present units: one-sided exact p (T* >= T, ties and the
    observed labelling included) and the opposite p (T* <= T); confirm at p <= 0.05, refute at
    opposite p <= 0.05, otherwise inconclusive; not testable with fewer than 3 units in either
    arm of any variant."""
    groups = {v: ([units[treated][v][s] for s in sorted(units[treated][v])],
                  [units[baseline][v][s] for s in sorted(units[baseline][v])]) for v in variants}
    rec = {"treated": treated, "baseline": baseline, "variants": list(variants),
           "n_units": {v: {"treated": len(t), "baseline": len(b)} for v, (t, b) in groups.items()},
           "means": {v: {"treated": Fraction(sum(t), len(t)) if t else None,
                         "baseline": Fraction(sum(b), len(b)) if b else None}
                     for v, (t, b) in groups.items()}}
    for t, b in groups.values():
        if any(not isinstance(x, int) or not 0 <= x <= MAX_UNIT for x in t + b):
            raise ValueError(f"units must be integers 0..{MAX_UNIT}: {t}, {b}")
    if any(len(t) < MIN_UNITS or len(b) < MIN_UNITS for t, b in groups.values()):
        return {**rec, "outcome": "not testable", "p": None, "p_opposite": None, "T": None,
                "note": f"fewer than {MIN_UNITS} units in an arm: counts as not confirmed"}
    c = exact_counts(groups)
    p, q = Fraction(c["ge"], c["total"]), Fraction(c["le"], c["total"])
    outcome = "confirm" if p <= ALPHA else "refute" if q <= ALPHA else "inconclusive"
    return {**rec, "T": c["T"], "p": p, "p_opposite": q, "count_ge": c["ge"],
            "count_le": c["le"], "relabellings": c["total"], "outcome": outcome}


# ------------------------------------------------------------------------- Section 3


def completed_set(net: dict) -> set[str]:
    return {c for c in R_DEV if float(net["completion"][c]) > COMPLETION_MIN}


def count(net: dict, circuits) -> int:
    done = completed_set(net)
    return sum(c in done for c in circuits)


def by_cell(nets: dict) -> dict:
    """{(variant, arm, seed): network}."""
    return {(n["variant"], n["arm"], int(n["seed"])): n for n in nets.values()}


def make_units(nets: dict) -> dict:
    """{unit: {variant: {seed: value}}} (Section 3): ``base`` b_s and ``sa`` a_s count all 12
    circuits; ``sl`` and ``cl`` are the leave-out units u_s = C(f1, H_f1) + C(f2, H_f2), present
    only when both halves of the pair are."""
    cell = by_cell(nets)
    out = {u: {v: {} for v in VARIANTS} for u in ("base", "sl", "cl", "sa")}
    for v in VARIANTS:
        for s in SEEDS:
            for arm in ("base", "sa"):
                if (v, arm, s) in cell:
                    out[arm][v][s] = count(cell[v, arm, s], R_DEV)
            for x in ("sl", "cl"):
                f1, f2 = cell.get((v, f"{x}-f1", s)), cell.get((v, f"{x}-f2", s))
                if f1 is not None and f2 is not None:
                    out[x][v][s] = count(f1, HELD_OUT["f1"]) + count(f2, HELD_OUT["f2"])
    return out


def _arm_means(values: dict[int, int], idx: np.ndarray) -> np.ndarray:
    x = np.array([values.get(s, np.nan) for s in SEEDS], float)[idx]
    n = np.sum(~np.isnan(x), axis=1)
    return np.where(n > 0, np.nansum(x, axis=1) / np.maximum(n, 1), np.nan)


def s3(nets: dict) -> dict:
    """S3 (descriptive, two-sided): per (variant, fold) cell, mean_s C(sl-f, H_f) - mean_s
    C(cl-f, H_f) in circuits of 6, with a 95% percentile bootstrap over seeds; and D2, their
    mean over the 4 cells (GW-4's criterion 2)."""
    cell = by_cell(nets)
    cells = {}
    for v in VARIANTS:
        for f in ("f1", "f2"):
            arm = {x: {s: count(cell[v, f"{x}-{f}", s], HELD_OUT[f]) for s in SEEDS
                       if (v, f"{x}-{f}", s) in cell} for x in ("sl", "cl")}
            rec = {"sl": arm["sl"], "cl": arm["cl"], "n_sl": len(arm["sl"]),
                   "n_cl": len(arm["cl"]), "diff": None, "ci95": None}
            if arm["sl"] and arm["cl"]:
                rec["mean_sl"] = Fraction(sum(arm["sl"].values()), len(arm["sl"]))
                rec["mean_cl"] = Fraction(sum(arm["cl"].values()), len(arm["cl"]))
                rec["diff"] = rec["mean_sl"] - rec["mean_cl"]
                idx = np.random.default_rng(BOOT_SEED).integers(0, len(SEEDS),
                                                                (BOOT_N, len(SEEDS)))
                d = _arm_means(arm["sl"], idx) - _arm_means(arm["cl"], idx)
                ok = ~np.isnan(d)
                rec["ci95"] = [float(x) for x in np.percentile(d[ok], [2.5, 97.5])]
                rec["draws_used"] = int(ok.sum())
            cells[f"{v}/{f}"] = rec
    diffs = [c["diff"] for c in cells.values()]
    d2 = sum(diffs) / len(diffs) if all(x is not None for x in diffs) else None
    return {"label": "descriptive (two-sided); GW-4's criterion 2 reads D2", "cells": cells,
            "D2": d2, "bootstrap": {"draws": BOOT_N, "rng": f"numpy default_rng({BOOT_SEED}) "
                                    "per cell", "over": "seeds 20-29 with replacement, both "
                                    "arms at the same drawn seeds"}}


def gaps(net: dict) -> dict[str, float]:
    """{circuit: gap in points} over the circuits a network completes."""
    return {c: POINTS * float(net["gap"][c]) for c in R_DEV if c in completed_set(net)}


def median_gap(g: dict[str, float], circuits=None) -> float | None:
    v = [x for c, x in g.items() if circuits is None or c in circuits]
    return float(np.median(v)) if v else None


def _mean(x: list[float]) -> float | None:
    return float(np.mean(x)) if x else None


def lap_time(nets: dict) -> dict:
    """GW-4's criterion 3 (IN-4's form): each network's median gap over its completed R-dev
    circuits; Delta-gap = mean over sa networks - mean over base networks, both variants pooled,
    networks with no completed circuit left out (and counted); undefined if every sa or every
    base network is left out. Descriptive: per variant, paired by seed on the circuits both
    networks complete, and the per-circuit table of interp-leftovers.md Section 6.1."""
    cell = by_cell(nets)
    g = {name: gaps(n) for name, n in nets.items()}
    per_net = {name: {"variant": n["variant"], "arm": n["arm"], "seed": int(n["seed"]),
                      "n_completed": len(g[name]), "median_gap": median_gap(g[name])}
               for name, n in sorted(nets.items())}

    def arm(a: str, variants) -> dict:
        names = [nm for nm, r in per_net.items() if r["arm"] == a and r["variant"] in variants]
        kept = [per_net[nm]["median_gap"] for nm in names if per_net[nm]["median_gap"] is not None]
        return {"n": len(names), "n_left_out": len(names) - len(kept), "mean": _mean(kept),
                "sd": float(np.std(kept, ddof=1)) if len(kept) > 1 else None,
                "left_out_per_variant": {v: sum(1 for nm in names if per_net[nm]["variant"] == v
                                                and per_net[nm]["median_gap"] is None)
                                         for v in variants}}

    def delta(variants) -> dict:
        sa, base = arm("sa", variants), arm("base", variants)
        d = sa["mean"] - base["mean"] if sa["mean"] is not None and base["mean"] is not None \
            else None
        return {"sa": sa, "base": base, "delta": d}

    pooled = delta(VARIANTS)
    pairs, left = {}, []
    for v in VARIANTS:
        for s in SEEDS:
            if (v, "sa", s) not in cell or (v, "base", s) not in cell:
                continue
            ga = g[RL.run_name(v, "sa", s)]
            gb = g[RL.run_name(v, "base", s)]
            common = sorted(set(ga) & set(gb))
            if not common:
                left.append(f"{v}/s{s}")
                continue
            pairs[f"{v}/s{s}"] = {"variant": v, "n_common": len(common),
                                  "median_sa": median_gap(ga, common),
                                  "median_base": median_gap(gb, common),
                                  "diff_of_medians": median_gap(ga, common) - median_gap(gb,
                                                                                         common),
                                  "mean_of_diffs": float(np.mean([ga[c] - gb[c]
                                                                  for c in common]))}
    paired = {"pairs": pairs, "left_out": left,
              "mean_over_pairs": {k: {sc: _mean([p[k] for p in pairs.values()
                                                 if sc == "pooled" or p["variant"] == sc])
                                      for sc in ("pooled", *VARIANTS)}
                                  for k in ("diff_of_medians", "mean_of_diffs")}}
    per_circuit = {}
    for v in VARIANTS:
        gs = [g[RL.run_name(v, "sa", s)] for s in SEEDS if (v, "sa", s) in cell]
        gb = [g[RL.run_name(v, "base", s)] for s in SEEDS if (v, "base", s) in cell]
        both = [c for c in R_DEV if any(c in x for x in gs) and any(c in x for x in gb)]
        rows = {}
        for c in both:
            ms = float(np.median([x[c] for x in gs if c in x]))
            mb = float(np.median([x[c] for x in gb if c in x]))
            rows[c] = {"base": mb, "sa": ms, "delta": ms - mb,
                       "n_base": sum(c in x for x in gb), "n_sa": sum(c in x for x in gs)}
        d = [r["delta"] for r in rows.values()]
        per_circuit[v] = {"circuits": rows, "n_circuits": len(rows),
                          "median_delta": float(np.median(d)) if d else None,
                          "n_positive": sum(x > 0 for x in d)}
    return {"units": "points (100 x (median lap / s-OCP lap - 1))", "per_network": per_net,
            "delta_gap": pooled["delta"], "pooled": pooled,
            "per_variant": {v: delta((v,)) for v in VARIANTS},
            "paired_by_seed": {"label": "descriptive", **paired},
            "per_circuit": {"label": "descriptive (interp-leftovers.md Section 6.1's table)",
                            **per_circuit}}


def _summary(x: dict[int, int]) -> dict:
    v = list(x.values())
    return {"per_seed": x, "n": len(v), "mean": Fraction(sum(v), len(v)) if v else None,
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}


def descriptive(nets: dict, units: dict) -> dict:
    """Section 3's descriptive tables."""
    cell = by_cell(nets)
    counts, shares, gate = {}, {}, {}
    for v in VARIANTS:
        for a in ARMS:
            present = {s: cell[v, a, s] for s in SEEDS if (v, a, s) in cell}
            rec = {"all_12": _summary({s: count(n, R_DEV) for s, n in present.items()})}
            if a in FOLD_OF:
                f = FOLD_OF[a]
                rec["held_out"] = _summary({s: count(n, HELD_OUT[f]) for s, n in present.items()})
                rec["in_sample_fold"] = _summary({s: count(n, RL.FOLDS[f])
                                                  for s, n in present.items()})
            counts[f"{v}/{a}"] = rec
            shares[f"{v}/{a}"] = {
                c: {"share": (sum(c in completed_set(n) for n in present.values())
                              / len(present)) if present else None, "n": len(present),
                    "held_out": (c in HELD_OUT[FOLD_OF[a]]) if a in FOLD_OF else None}
                for c in R_DEV}
            gate[f"{v}/{a}"] = {"n": len(present), "c_at_least_10": sum(
                count(n, R_DEV) >= NOT_THE_GATE_MIN for n in present.values())}
    pairs = {v: {s: {"sl": units["sl"][v].get(s), "base": units["base"][v].get(s),
                     "diff": (units["sl"][v][s] - units["base"][v][s])
                     if s in units["sl"][v] and s in units["base"][v] else None}
                 for s in SEEDS} for v in VARIANTS}
    return {"counts": counts, "units": units,
            "per_circuit_share": {"named": list(NAMED), "shares": shares},
            "per_seed_pairs_sl_base": pairs,
            "c_at_least_10_of_12": {"label": "not the gate (Section 6)", "per_cell": gate}}


# ------------------------------------------------------------------------- Section 5


def criterion_4(report: bytes | None, pin: str = WIDTH_REPORT_SHA256) -> dict:
    """GW-4's criterion 4, restated (27 Oct 2026): every v1 check in (2a)'s committed width
    report passes in both folds, the v3 misses are limited to C4_V3_ALLOWED, and every no-copy
    check (f1, f2 and all 12, both variants) passes. The report is read only after its sha256 is
    checked against ``pin``; a mismatch stops GW-4 for an amendment."""
    sha = sha256_bytes(report) if report is not None else None
    if sha != pin:
        return {"holds": None, "sha256": sha, "pinned_sha256": pin,
                "stopped": f"{WIDTH_REPORT_REL} is not the pinned report (sha256 "
                           f"{(sha or 'missing')[:12]}, pinned {pin[:12]}): GW-4 stops for an "
                           "amendment (Section 5)"}
    r = json.loads(report)
    m = r["models"]["2a"]
    band = {f"{f}/{v}/{c['stat']}": c["ok"] for f in ("f1", "f2") for v in VARIANTS
            for c in m[f][v]["checks"] if c["judged"]}
    nocopy = {f"{f}/{v}/nocopy": bool(m[f][v]["nocopy"]["ok"]) for f in ("f1", "f2", "all")
              for v in VARIANTS}
    failed = [k for k, ok in {**band, **nocopy}.items() if not ok]
    if sorted(failed) != sorted(r["verdict"]["2a"]["failed"]) or \
            len(band) + len(nocopy) != C4_N_CHECKS or r["verdict"]["2a"]["n_checks"] != C4_N_CHECKS:
        raise CheckFailed("the width report's checks do not reproduce its own verdict")
    v1_failed = [k for k in failed if k.split("/")[1] == "v1"]
    v3_band_failed = [k for k in failed if k.split("/")[1] == "v3" and not k.endswith("/nocopy")]
    nocopy_failed = [k for k in failed if k.endswith("/nocopy")]
    holds = (not v1_failed and set(v3_band_failed) <= set(C4_V3_ALLOWED) and not nocopy_failed)
    return {"holds": holds, "sha256": sha, "pinned_sha256": pin, "n_checks": C4_N_CHECKS,
            "failed": failed, "v1_failed": v1_failed,
            "v3_failed_outside_allowed": sorted(set(v3_band_failed) - set(C4_V3_ALLOWED)),
            "nocopy_failed": nocopy_failed, "v3_allowed": list(C4_V3_ALLOWED)}


def gw4(tests: dict, s3res: dict, lap: dict, c4: dict) -> dict:
    """Section 5's rule: GW-4 adopts M if all five criteria hold; otherwise #12's rule (copied
    widths if and only if (2a) fails the rule and S1-v1 confirms; else the default widths).
    Criterion 5 is read at the generator-v1 commit, so the decision is given for both of its
    outcomes."""
    d2, dgap = s3res["D2"], lap["delta_gap"]
    crit = {
        "1": {"holds": tests["P1"]["outcome"] == "confirm", "P1": tests["P1"]["outcome"]},
        "2": {"holds": d2 is not None and d2 >= D2_MIN, "D2": d2, "min": D2_MIN},
        "3": {"holds": dgap is not None and dgap < DGAP_MAX, "delta_gap": dgap,
              "max_exclusive": DGAP_MAX},
        "4": c4,
        "5": {"holds": None, "read": "by GW-4 at the generator-v1 commit, not here",
              "tests": list(C5_TESTS)}}
    s1v1 = tests["S1-v1"]["outcome"]
    out = {"criteria": crit, "s1_v1": s1v1, "decisions": DECISIONS}
    if c4["holds"] is None:
        return {**out, "stopped": c4["stopped"], "criteria_1_to_4_hold": None,
                "if_criterion_5_holds": None, "if_criterion_5_fails": None}
    c14 = all(crit[k]["holds"] for k in ("1", "2", "3", "4"))
    fallback = "copied" if s1v1 == "confirm" else "default"
    res = {"criteria_1_to_4_hold": c14,
           "if_criterion_5_holds": "adopt" if c14 else fallback,
           "if_criterion_5_fails": fallback,
           "fallback_rule": "#12: copied R-dev widths if and only if (2a) fails the rule and "
                            "S1-v1 confirms (not testable counts as not confirmed); otherwise "
                            "the default widths"}
    labels = {k: (IN_SAMPLE_LABEL if res[k] == "copied" else None)
              for k in ("if_criterion_5_holds", "if_criterion_5_fails")}
    return {**out, **res, "in_sample_label": labels, "in_sample_label_covers": IN_SAMPLE_COVERS,
            "e0_variant": "v1 (decided 27 Oct 2026; not chosen on GW-3's results)",
            "lockbox": "R_TEST is unaffected either way"}


def analyse_runs(nets: dict, missing: dict[str, str], report: bytes | None,
                 pin: str = WIDTH_REPORT_SHA256) -> dict:
    """Sections 3, 5 and 6 from the present networks ({name: {variant, arm, seed, completion,
    gap}}: completion and gap per R-dev circuit, as eval_rdev.json gives them) and the missing
    runs ({name: reason})."""
    units = make_units(nets)
    tests = {k: permutation_test(units, *spec) for k, spec in TESTS.items()}
    s3res = s3(nets)
    lap = lap_time(nets)
    c4 = criterion_4(report, pin)
    return {"status": STATUS,
            "runs": {"expected": len(RL.runs()), "present": len(nets), "missing": missing},
            "primary": {"P1": {**tests["P1"], "question": "sl above base on held-out "
                               "circuits, both variants (one-sided)"}},
            "secondary": {"label": "secondary: exact one-sided tests at 0.05, unadjusted",
                          "S1": {**tests["S1"], "question": "cl above base on held-out "
                                 "circuits, both variants"},
                          "S1-v1": {**tests["S1-v1"], "question": "cl above base within v1 "
                                    "(#12's transfer condition)"},
                          "S1-v3": {**tests["S1-v3"], "question": "cl above base within v3 "
                                    "(reported; does not decide #12)"},
                          "S2": {**tests["S2"], "question": "sa above base on all 12 "
                                 "(in-sample for M's summaries)"}},
            "S3": s3res, "lap_time": lap, "descriptive": descriptive(nets, units),
            "gw4": gw4(tests, s3res, lap, c4), "gate_80pct": GATE_80}


# ------------------------------------------------------------------------- analyse


def ocp_laps() -> tuple[dict[str, float], dict[str, str]]:
    """The s-OCP lap of each R-dev circuit (outputs/ocp/<circuit>.pkl, each name through
    ``lockbox_guard`` first) and each file's sha256."""
    from apexzero.viz.export import lockbox_guard

    laps, shas = {}, {}
    for c in R_DEV:
        lockbox_guard(c)
        p = ROOT / "outputs" / "ocp" / f"{c}.pkl"
        raw = p.read_bytes()
        laps[c] = float(pickle.loads(raw).lap_time)
        shas[f"outputs/ocp/{c}.pkl"] = sha256_bytes(raw)
    return laps, shas


def load_manifest(gate: dict) -> tuple[dict, str]:
    """The committed manifest (committed at HEAD, unchanged on disk) and its sha256, after
    checking that it is this tag's and this run list's."""
    probs = h4r.tree_problems(ROOT, (MANIFEST_REL,))
    if probs:
        raise CheckFailed(f"the manifest must be committed and unchanged: {probs}")
    raw = (ROOT / MANIFEST_REL).read_bytes()
    man = json.loads(raw)
    if man.get("tag") != {k: gate[k] for k in ("tag", "tag_object", "commit")}:
        raise CheckFailed(f"the manifest is not {TAG}'s ({man.get('tag')})")
    if man.get("run_list_sha256") != run_list_sha256(RL.runs()):
        raise CheckFailed("the manifest's run list is not scripts/gw3_runs.py's")
    if sorted([*man["present"], *man["missing"]]) != sorted(man["runs"]) or \
            sorted(man["runs"]) != sorted(r["name"] for r in RL.runs()):
        raise CheckFailed("the manifest's runs are not the run list, each present or missing")
    return man, sha256_bytes(raw)


def load_networks(man: dict, ocp: dict[str, float]) -> tuple[dict, dict[str, str]]:
    """The present networks' evaluations, each re-hashed against the manifest; every row's s-OCP
    lap must be outputs/ocp's and its gap median lap / s-OCP lap - 1."""
    nets, inputs = {}, {}
    for name in man["present"]:
        ent = man["runs"][name]
        raw = (ROOT / ent["eval_path"]).read_bytes()
        if sha256_bytes(raw) != ent["eval_sha256"]:
            raise CheckFailed(f"{ent['eval_path']} is not the manifest's (sha256)")
        inputs[ent["eval_path"]] = ent["eval_sha256"]
        ev = json.loads(raw)
        if set(ev) != set(R_DEV):
            raise CheckFailed(f"{name}: its evaluation's circuits are not the 12 R-dev circuits")
        summary, lacks = eval_summary(ev)
        if lacks or summary["completion"] != ent["completion"]:
            raise CheckFailed(f"{name}: its evaluation is not the manifest's ({lacks})")
        for c in R_DEV:
            row = ev[c]
            if "ocp_lap_s" not in row:      # train_dr.py's row without outputs/ocp/<c>.pkl
                if float(row["completion"]) > COMPLETION_MIN:
                    raise CheckFailed(f"{name} {c}: completed, but no s-OCP lap")
                continue
            if float(row["ocp_lap_s"]) != ocp[c]:
                raise CheckFailed(f"{name} {c}: ocp_lap_s {row['ocp_lap_s']} is not "
                                  f"outputs/ocp's {ocp[c]}")
            lap = float(row["median_lap_s"])
            if math.isfinite(lap) and abs(float(row["gap"]) - (lap / ocp[c] - 1.0)) > GAP_TOL:
                raise CheckFailed(f"{name} {c}: gap is not median lap / s-OCP lap - 1")
        v, arm, seed = parse_name(name)
        if (ent["variant"], ent["arm"], ent["seed"]) != (v, arm, seed):
            raise CheckFailed(f"{name}: the manifest's cell is not its name's")
        nets[name] = {"variant": v, "arm": arm, "seed": seed,
                      "completion": {c: float(ev[c]["completion"]) for c in R_DEV},
                      "gap": {c: float(ev[c].get("gap", float("nan"))) for c in R_DEV}}
    return nets, inputs


def stage_analyse(args) -> None:
    gate = tag_gate()
    out = ROOT / ANALYSIS_REL
    if out.exists() or ever_committed(ANALYSIS_REL):
        raise CheckFailed(f"{ANALYSIS_REL} exists or was committed: the analysis runs once")
    start = time.time()
    man, man_sha = load_manifest(gate)
    ocp, ocp_sha = ocp_laps()
    nets, inputs = load_networks(man, ocp)
    report_path = ROOT / WIDTH_REPORT_REL
    report = report_path.read_bytes() if report_path.exists() else None
    inputs |= {MANIFEST_REL: man_sha, **ocp_sha}
    if report is not None:
        inputs[WIDTH_REPORT_REL] = sha256_bytes(report)
    print(f"inputs checked: {len(nets)} networks present, {len(man['missing'])} missing")
    if args.check:
        print("--check: nothing computed or written")
        return
    res = analyse_runs(nets, man["missing"], report)
    res["provenance"] = provenance(start, gate, inputs)
    sha = write_new(out, dumps(res))
    p1, g = res["primary"]["P1"], res["gw4"]
    print(f"P1: {p1['outcome']} (p {p1['p']}); S1-v1: {g['s1_v1']}; GW-4: "
          f"{g.get('stopped') or g['if_criterion_5_holds']} if criterion 5 holds")
    print(f"wrote {ANALYSIS_REL} ({sha}); next: git add -f {ANALYSIS_REL} && git commit")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    m = sub.add_parser("manifest", help="the run manifest, after pull-back (Section 8, step 5)")
    m.add_argument("--accept-missing", action="append", metavar="RUN=REASON",
                   help="a run missing for another reason than failing twice (a deviation)")
    a = sub.add_parser("analyse", help="the analysis, once (Section 8, step 6)")
    a.add_argument("--check", action="store_true", help="input checks only; writes nothing")
    args = ap.parse_args(argv)
    {"manifest": stage_manifest, "analyse": stage_analyse}[args.stage](args)


if __name__ == "__main__":
    main()
