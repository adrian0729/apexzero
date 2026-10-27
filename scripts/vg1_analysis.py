"""VG-1's manifests, pilot rule and analysis (docs/prereg/vg1-value-gradient.md, Sections 4-6).

    uv run python scripts/vg1_analysis.py manifest --phase pilot --tag <pilot commit> \\
        --box-sums runs/vg1p-box-sha256.txt
    uv run python scripts/vg1_analysis.py m0 --phase pilot
    uv run python scripts/vg1_analysis.py pilot --box-gpu-hours <H>
    uv run python scripts/vg1_analysis.py manifest --phase main --tag vg1-prereg-v1 \\
        --box-sums runs/vg1-box-sha256.txt
    uv run python scripts/vg1_analysis.py m0 --phase main
    uv run python scripts/vg1_analysis.py analyse

**manifest** (on the Mac, after ``gpu_sync.sh pull-round``): runs ``round_manifest.py check``
on the round (vg1p for the pilot, vg1 for the main run) against ``--tag`` (the pilot commit's
sha for the pilot, the tag for the main run). Its only tolerated problems are those of a run
outputs/vg1/failures.json classifies (committed; a run that failed twice: 'missing' for the
box, memory, I/O or the harness, 'failed' for non-finite numbers; Section 5.1). Then, per run:
every checkpoint's sha256 and timesteps, the final one at ``FINAL_TIMESTEPS``, checked against
the box's own sha256 (``--box-sums``: ``sha256sum`` lines taken on the box before it is deleted);
config.json, metrics.csv, eval_rdev.json and the logs' sha256; the box's status, times, lane
and GPU. A pilot run must have --skip-eval-rdev in its config, no eval_rdev.json, and no R-dev
circuit's name and no 'completion' in its logs. A main run's config must carry the arm's
flags: ``vg`` = {coef beta*, lam 0.95, winsor 0.005} for vgl, no ``vg`` for base (Section 4.2).
Writes outputs/vg1/manifest-<phase>.json, write-once.

**m0**: ``scripts/vgl.py residual`` on each run's final checkpoint at its fixed settings
(``M0_SIZES``), into outputs/vg1/m0/<run>.json, with VG-1's code unchanged against the
manifest's commit before and after each run (tracked changes only: other studies' untracked
files, which vgl.py's own clean flag counts, do not). A sidecar <run>.check.json binds the
record's sha256, HEAD and that check.

**pilot** (Section 4.1): reads the pilot manifest, outputs/vg1/c1/pilot.json (M1, from
scripts/vg1_c1.py), outputs/vg1/m0/<run>.json (M0, from ``m0``) and each run's last
metrics.csv row (EV), each checked against the manifest's sha256 and against the pilot
commit's code. A beta is eligible if both its runs
finished with finite M1, M0 and EV, EV >= 0.90 and M0 < 13.99. beta* is the eligible beta with
the highest median M1 over its two seeds, a tie to the smaller beta; with none eligible,
beta* is null and the main run does not start. The cost check, in GPU-hours with idle GPUs
included: the pilot's box GPU-hours (``--box-gpu-hours``: box lifetime x its GPUs, setup and
pull-back included; at least the queue's) plus the main run's projection, 20 x 6.5 min + 20 x
beta*'s mean pilot GPU minutes per run + 15 min x the main box's GPUs (``--main-gpus``,
default 4); above 14 it goes to the user before the tag, and the stop rule pauses VG-1 above
16.8. A run's GPU minutes are its share of its GPU (``gpu_minutes``): the integral over its
time of 1 / (the runs on that GPU then). Writes outputs/vg1/pilot.json, write-once.

**analyse** (Sections 5 and 6), once, after the tag: the tag gate (``tag_gate``: vg1-prereg-v1
on origin as an annotated tag, the local tag the same object, HEAD descended from it, VG-1's
code ``CODE_PATHS`` the tagged code, untracked files included); refuses if analysis.json or its
lock exists or was ever committed; re-checks the manifest (``round_manifest.py check`` and every
sha256), then takes the lock (outputs/vg1/analysis.lock, never removed) before any number. Per
network (``network_values``): R2_c and Spearman rho (vg1_c1.py), M0 and S4's steer and pedal
residuals (vgl.py), EV, completion C (circuits with completion > 0.5 of 12). The failed-network
rule: a network with a non-finite parameter, a non-finite last metrics row, a numerical K3
failure, a non-finite R2_c, or classified 'failed' is failed; each non-finite statistic takes
its test's worst value (``WORST``: R2_c and rho -inf, M0 and S4 +inf, EV -inf), C is 0 without
eval_rdev.json; missing networks are dropped. P is the exact one-sided Wilcoxon rank-sum
(``ranksum_exact``: mid-ranks, the exact permutation distribution of the rank sum by the
recursion over ranks, a tie counting toward p), confirm at p <= 0.05, refute at the opposite
p <= 0.05, not testable with fewer than 15 networks in an arm; with the Hodges-Lehmann shift and
its 95% percentile bootstrap (20,000 draws, default_rng(20261026), nearest rank). S1 (two-sided,
doubled smaller tail), S2, S3, S4 (steer, pedal), S5 (two-sided) are the same test; S6 the 20
same-seed differences by all 2^n sign flips (``signflip_exact``), -inf replaced by the lowest
finite R2_c of the 40. The reading is Section 6's first matching row. The descriptives of Section
5.3 follow. Writes outputs/vg1/analysis.json.

Every file this writes is write-once (no overwrite), JSON with NaN as null and infinities as
"inf" / "-inf". Nothing here reads R_TEST, outputs/sealed or any run outside VG-1's lists.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = "outputs/vg1"
RUNS_SCRIPT = "scripts/vg1_runs.py"
ROUNDS = {"pilot": "vg1p", "main": "vg1"}
TAG = "vg1-prereg-v1"
CODE_PATHS = ("scripts/vg1_analysis.py", "scripts/vg1_c1.py", "scripts/vg1_runs.py",
              "scripts/vgl.py", "scripts/train_dr.py", "scripts/round_manifest.py",
              "scripts/gpu_round.sh", "scripts/_bootstrap.py", "src/apexzero")
FINAL_TIMESTEPS = 199_884_800
FAILURE_CLASSES = ("missing", "failed")
# the pilot rule (Section 4.1)
EV_MIN = 0.90
M0_MAX = 13.99                      # strictly below: the median M0 of H4R's five v1 baselines
BETAS = ("0.1", "0.3", "1.0")
# scripts/vgl.py residual's fixed settings (Sections 4.1 and 5.2: S3)
M0_SIZES = {"rollouts": 4, "num_envs": 64, "num_steps": 64, "bank": 256, "seed": 20261026}
M0_VG = {"lam": 0.95, "winsor": 0.005}
VG_FLAGS = {"lam": 0.95, "winsor": 0.005}
# the cost check (Sections 4.1 and 8), GPU-hours with idle GPUs included
BASE_GPU_MIN = 6.5                  # per 200M default-path run at 4 lanes per GPU
MAIN_RUNS_PER_ARM = 20
SETUP_MIN = 15.0                    # setup and pull-back, per box, in wall minutes
COST_LIMIT_H = 14.0
STOP_RULE_H = 16.8                  # 14 h + 20%
# the analysis (Section 5)
ALPHA = 0.05
MIN_PER_ARM = 15
N_BOOT = 20_000
BOOT_SEED = 20261026
PERM_TIE = 1e-12
COMPLETED = 0.5
WORST = {"r2c": -math.inf, "rho": -math.inf, "m0": math.inf, "s4_steer": math.inf,
         "s4_pedal": math.inf, "ev": -math.inf}


class CheckFailed(SystemExit):
    """A built-in check failed: nothing more is written."""


# ------------------------------------------------------------------------- small helpers


def git_run(*args: str) -> subprocess.CompletedProcess:
    """Every git call goes through here (the tests replace it)."""
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def git(*args: str) -> str:
    out = git_run(*args)
    return out.stdout.strip() if out.returncode == 0 else ""


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


def rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return f"<outside>/{p.name}"


def provenance(start: float, inputs: dict[str, str]) -> dict:
    return {"script": "scripts/vg1_analysis.py",
            "script_sha256": sha256_file(Path(__file__)), "head": git("rev-parse", "HEAD"),
            "changed": [ln for ln in git("status", "--porcelain", "--untracked-files=no")
                        .splitlines() if ln.strip()],
            "inputs": dict(sorted(inputs.items())),
            "argv": ["scripts/vg1_analysis.py",
                     *[rel(a) if os.path.isabs(a) else a for a in sys.argv[1:]]],
            "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
            "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "machine": platform.machine(), "system": platform.system()}}


def read_list(path: Path, name: str) -> list[str]:
    """A module-level list of strings in a source file, read with ast (no import)."""
    for node in ast.parse(Path(path).read_text()).body:
        targets = [getattr(t, "id", None) for t in getattr(node, "targets", [])]
        if isinstance(node, ast.Assign) and targets == [name]:
            return list(ast.literal_eval(node.value))
    raise CheckFailed(f"{name} not found in {path}")


def r_dev() -> list[str]:
    return read_list(ROOT / "src" / "apexzero" / "tracks" / "corners.py", "R_DEV")


def load_runs_script():
    spec = importlib.util.spec_from_file_location("vg1_runs", ROOT / RUNS_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


# ------------------------------------------------------------------------- the tag gate


def tag_gate(tag: str = TAG, paths: tuple[str, ...] = CODE_PATHS) -> dict:
    """PermissionError unless the annotated tag exists on origin, the local tag is the same tag
    object, HEAD descends from its commit, and VG-1's code in the working tree (``paths``,
    untracked files included) is the tagged code (scripts/lockbox.py's ``tag_gate``)."""
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
        raise PermissionError(f"{tag} is not on origin: VG-1's main run is not analysed before it")
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
    if git_run("diff", "--quiet", peeled, "--", *paths).returncode != 0:
        raise PermissionError(f"VG-1's code ({', '.join(paths)}) differs from {tag}'s")
    untracked = git_run("ls-files", "--others", "--exclude-standard", "--", *paths)
    if untracked.returncode != 0 or untracked.stdout.strip():
        raise PermissionError(f"untracked files under VG-1's code: {untracked.stdout!r}")
    return {"tag": tag, "tag_object": obj, "commit": peeled, "head": git("rev-parse", "HEAD")}


def code_at(head: str | None, commit: str, paths: tuple[str, ...] = CODE_PATHS) -> bool:
    """Whether ``head``'s VG-1 code is ``commit``'s (a measurement's provenance)."""
    if not head:
        return False
    return git_run("diff", "--quiet", commit, head, "--", *paths).returncode == 0


# ------------------------------------------------------------------------- the round check


def check_report(phase: str, tag: str) -> tuple[int, str]:
    """(return code, report) of ``round_manifest.py check`` on the phase's round."""
    cmd = [sys.executable, "scripts/round_manifest.py", "check", "--round", ROUNDS[phase],
           "--runs-script", RUNS_SCRIPT, "--tag", tag, "--runs-args", "--phase", phase]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout + proc.stderr


def round_check(phase: str, tag: str, failures: dict) -> dict:
    """``round_manifest.py check`` on the phase's round against ``tag``; the problems of a run
    ``failures`` classifies are tolerated (and listed), any other one raises."""
    rc, report = check_report(phase, tag)
    problems = []
    if rc:
        lines = report.splitlines()
        if not any(ln.startswith("FAILED:") for ln in lines):
            raise CheckFailed(f"round_manifest.py check did not run:\n{report.strip()}")
        k = next(i for i, ln in enumerate(lines) if ln.startswith("FAILED:"))
        problems = [ln.strip() for ln in lines[k + 1:] if ln.strip()]
    classified = set(failures)
    tolerated = [p for p in problems if p.split(":", 1)[0] in classified]
    other = [p for p in problems if p not in tolerated]
    if other:
        raise CheckFailed("round_manifest.py check failed:\n  " + "\n  ".join(other))
    return {"returncode": rc, "tolerated": tolerated,
            "report_sha256": hashlib.sha256(report.encode()).hexdigest(), "report": report}


def read_failures(phase: str, inputs: dict) -> dict:
    """outputs/vg1/failures.json's runs of this phase ({} if there is no file); committed."""
    path = f"{OUT}/failures.json"
    if not (ROOT / path).exists():
        return {}
    if not committed(path):
        raise CheckFailed(f"{path} must be committed before any outcome is read (Section 5.1)")
    rec = read_json(path, inputs)
    if rec.get("record") != "vg1-failures":
        raise CheckFailed(f"{path} is not VG-1's failure record")
    runs = {k: v for k, v in (rec.get("runs") or {}).items()
            if k.startswith(ROUNDS[phase] + "-")}
    bad = [k for k, v in runs.items() if (v or {}).get("class") not in FAILURE_CLASSES
           or not (v or {}).get("reason")]
    if bad:
        raise CheckFailed(f"{path}: {bad} need a class ({FAILURE_CLASSES}) and a reason")
    return runs


# ------------------------------------------------------------------------- manifest


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


def box_records(round_: str) -> dict[str, dict]:
    """{run: its record in the box manifests, with the manifest's file name as 'box'}."""
    out = {}
    for p in sorted((ROOT / "runs").glob(f"{round_}-manifest-*of*.json")):
        m = json.loads(p.read_text())
        for r in m.get("runs", []):
            out[r["name"]] = {**r, "box": p.name, "gpu_model": m.get("gpu_model"),
                              "gpu_count": m.get("gpu_count"), "box_type": m.get("box_type")}
    return out


def round_files(round_: str) -> dict[str, str]:
    """{path: sha256} of the round's box manifests, queue logs and status files."""
    pats = (f"{round_}-manifest-*of*.json", f"{round_}-queue-*of*.log",
            f"{round_}-status-*of*.tsv")
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


def leak_problems(name: str, names: list[str]) -> list[str]:
    """A pilot run's R-dev leaks: an eval_rdev.json, or an R-dev name or 'completion' in a log."""
    out = []
    if (ROOT / "runs" / name / "eval_rdev.json").exists():
        out.append(f"{name}: runs/{name}/eval_rdev.json exists")
    for log in (f"runs/{name}.log", f"runs/{name}.log.1"):
        p = ROOT / log
        if not p.exists():
            continue
        low = p.read_text(errors="replace").casefold()
        hit = [n for n in names if n.casefold() in low] + (["completion"] if "completion" in low
                                                            else [])
        if hit:
            out.append(f"{name}: {log} names {hit[:3]}")
    return out


def config_problems(run: dict, cfg: dict, phase: str, beta: str | None) -> list[str]:
    """The arm's flags in a run's config.json (Sections 4.1 and 4.2)."""
    name, args = run["name"], cfg.get("args") or {}
    out = []
    if phase == "pilot":
        if args.get("skip_eval_rdev") is not True:
            out.append(f"{name}: --skip-eval-rdev is not in its config")
        want = {"coef": run["beta"], **VG_FLAGS}
    elif run.get("arm") == "vgl":
        want = {"coef": float(beta), **VG_FLAGS}
    else:
        want = None
        if args.get("vg_coef", 0.0) != 0.0:
            out.append(f"{name}: a base run with --vg-coef {args.get('vg_coef')}")
    if phase == "main" and args.get("skip_eval_rdev") is not False:
        out.append(f"{name}: a main run must evaluate R-dev (skip_eval_rdev False)")
    got = cfg.get("vg")
    if want is None and got is not None:
        out.append(f"{name}: a base run's config has vg {got}")
    if want is not None and got != want:
        out.append(f"{name}: config vg {got}, want {want}")
    if int(args.get("seed", -1)) != run["seed"]:
        out.append(f"{name}: config seed {args.get('seed')}")
    return out


def build_manifest(phase: str, tag: str, sums: dict[str, str], runs: list[dict],
                   failures: dict, check: dict, beta: str | None) -> dict:
    """The phase's manifest (module docstring 'manifest'); raises on any problem."""
    round_ = ROUNDS[phase]
    commit = git("rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}")
    if not commit:
        raise CheckFailed(f"{tag} is not a commit or tag here")
    boxes = box_records(round_)
    names = r_dev()
    problems, entries = [], []
    for run in runs:
        name = run["name"]
        box = boxes.get(name, {})
        fail = failures.get(name)
        ent = {"name": name, "seed": run["seed"], "arm": run.get("arm"), "beta": run.get("beta"),
               "status": box.get("status"), "failure": fail,
               "box": {k: box.get(k) for k in ("box", "lane", "gpu", "start", "end", "gpu_model",
                                               "gpu_count", "box_type")}}
        d = ROOT / "runs" / name
        cks = checkpoints(name) if d.is_dir() else []
        ent["checkpoints"] = cks
        done = box.get("status") in ("done", "retried")
        final = cks[-1] if cks and cks[-1]["timesteps"] == FINAL_TIMESTEPS else None
        if done and final is None:
            problems.append(f"{name}: no final checkpoint at {FINAL_TIMESTEPS} timesteps")
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
        if done:
            cfg_p = d / "config.json"
            if not cfg_p.is_file():
                problems.append(f"{name}: no config.json")
            else:
                problems += config_problems(run, json.loads(cfg_p.read_text()), phase, beta)
            if ent["metrics_sha256"] is None:
                problems.append(f"{name}: no metrics.csv")
            if phase == "main" and ent["eval_rdev_sha256"] is None:
                problems.append(f"{name}: no eval_rdev.json")
        elif fail is None:
            problems.append(f"{name}: status {box.get('status')!r} and not in failures.json")
        if phase == "pilot":
            problems += leak_problems(name, names)
        entries.append(ent)
    unlisted = sorted(set(failures) - {r["name"] for r in runs})
    if unlisted:
        problems.append(f"failures.json lists runs outside the round: {unlisted}")
    if problems:
        raise CheckFailed(f"the {phase} manifest has {len(problems)} problem(s):\n  "
                          + "\n  ".join(problems))
    return {"record": "vg1-manifest", "phase": phase, "round": round_, "tag": tag,
            "commit": commit, "beta_star": beta, "runs": entries,
            "round_files": round_files(round_), "round_check": check,
            "box_sums_used": len(sums)}


def cmd_manifest(a) -> None:
    t0 = time.time()
    phase = a.phase
    path = ROOT / OUT / f"manifest-{phase}.json"
    if path.exists():
        raise CheckFailed(f"{rel(path)} exists: write-once")
    inputs: dict[str, str] = {}
    failures = read_failures(phase, inputs)
    check = round_check(phase, a.tag, failures)
    beta = None
    if phase == "main":
        read_json(f"{OUT}/pilot.json", inputs)
        beta = load_runs_script().beta_star(ROOT)
    runs = load_runs_script().runs(phase, ROOT)
    sums = box_sums(a.box_sums)
    for p in a.box_sums:
        inputs[rel(p)] = sha256_file(p)
    m = build_manifest(phase, a.tag, sums, runs, failures, check, beta)
    m["provenance"] = provenance(t0, inputs)
    write_new(path, m)


# ------------------------------------------------------------------------- GPU minutes


def _t(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def gpu_minutes(entries: list[dict]) -> dict[str, float]:
    """{run: its share of its GPU in minutes}: the integral over its time of 1 / (the runs on
    the same box and GPU then). The shares of one GPU sum to its busy time."""
    spans = {e["name"]: ((e["box"]["box"], e["box"]["gpu"]), _t(e["box"]["start"]),
                         _t(e["box"]["end"])) for e in entries
             if (e.get("box") or {}).get("start") and (e.get("box") or {}).get("end")}
    out = {n: 0.0 for n in spans}
    for g in {s[0] for s in spans.values()}:
        mine = {n: s for n, s in spans.items() if s[0] == g}
        cuts = sorted({t for s in mine.values() for t in s[1:]})
        for lo, hi in itertools.pairwise(cuts):
            on = [n for n, s in mine.items() if s[1] <= lo and s[2] >= hi]
            for n in on:
                out[n] += (hi - lo) / len(on) / 60.0
    return out


def queue_gpu_hours(entries: list[dict]) -> float:
    """Each box's queue wall time x its listed GPUs, summed (round_manifest.py check's line)."""
    by = {}
    for e in entries:
        b = e.get("box") or {}
        if b.get("start") and b.get("end"):
            lo, hi, n = by.get(b["box"], (math.inf, -math.inf, b.get("gpu_count") or 1))
            by[b["box"]] = (min(lo, _t(b["start"])), max(hi, _t(b["end"])), n)
    return sum((hi - lo) / 3600.0 * n for lo, hi, n in by.values())


# ------------------------------------------------------------------------- per-network inputs


def last_metrics(name: str) -> dict[str, float]:
    with open(ROOT / "runs" / name / "metrics.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    return {k: float(v) if v not in ("", None) else math.nan for k, v in rows[-1].items()}


def vg1_code_changed() -> list[str]:
    """Tracked changes under VG-1's code (``git status``; other studies' untracked files do
    not count, as vgl.py's own 'src_or_scripts_dirty' would count them)."""
    return [ln for ln in git_run("status", "--porcelain", "--untracked-files=no", "--",
                                 *CODE_PATHS).stdout.splitlines() if ln.strip()]


def m0_sidecar(name: str) -> str:
    return f"{OUT}/m0/{name}.check.json"


def cmd_m0(a) -> None:
    """M0 and S4's inputs: ``scripts/vgl.py residual`` on each run's final checkpoint at its
    fixed settings, into outputs/vg1/m0/<run>.json, with VG-1's code unchanged against the
    manifest's commit before and after; a sidecar <run>.check.json binds the record's sha256,
    HEAD and that check (``check_m0`` accepts it in place of vgl.py's own clean flag)."""
    if a.phase == "main":
        tag_gate()
    inputs: dict[str, str] = {}
    m = load_manifest(a.phase, inputs)
    want = set(a.run or [])
    for ent in m["runs"]:
        name = ent["name"]
        if not ent.get("final") or (want and name not in want):
            continue
        out, side = f"{OUT}/m0/{name}.json", m0_sidecar(name)
        if (ROOT / out).exists() or (ROOT / side).exists():
            raise CheckFailed(f"{out} or its sidecar exists: write-once")
        head = git("rev-parse", "HEAD")
        if vg1_code_changed() or not code_at(head, m["commit"]):
            raise CheckFailed(f"VG-1's code is not {m['commit'][:12]}'s: {vg1_code_changed()}")
        (ROOT / out).parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "scripts/vgl.py", "--threads", str(a.threads), "--nice", "10",
               "--out", out, "residual", f"runs/{name}", "--checkpoint", ent["final"]["path"]]
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        if vg1_code_changed() or git("rev-parse", "HEAD") != head:
            raise CheckFailed(f"VG-1's code or HEAD changed while {name}'s M0 ran")
        write_new(ROOT / side, {"record": "vg1-m0-check", "run": name,
                                "m0_sha256": sha256_file(ROOT / out), "head": head,
                                "commit": m["commit"], "vg1_code_clean": True,
                                "manifest_sha256": m["_sha256"]})


def check_m0(rec: dict, ent: dict, commit: str, side: dict | None = None,
             rec_sha256: str | None = None) -> list[str]:
    """An M0 record (vgl.py residual) against the run's final checkpoint, the fixed settings
    and the reference commit's code: vgl.py's own clean flag, or ``cmd_m0``'s sidecar binding
    the record's sha256."""
    out = []
    final = ent.get("final") or {}
    if rec.get("run") != ent["name"] or rec.get("checkpoint") != Path(final.get("path", "")).name:
        out.append(f"{ent['name']}: its M0 is of {rec.get('run')}/{rec.get('checkpoint')}")
    if rec.get("checkpoint_sha256") != final.get("sha256"):
        out.append(f"{ent['name']}: its M0's checkpoint sha256 is not the manifest's")
    sizes = {k: (rec.get("sizes") or {}).get(k) for k in M0_SIZES}
    if sizes != M0_SIZES or rec.get("vg") != M0_VG:
        out.append(f"{ent['name']}: M0 at {sizes} {rec.get('vg')}, not {M0_SIZES} {M0_VG}")
    env = rec.get("environment") or {}
    vouched = (side is not None and side.get("vg1_code_clean") is True
               and side.get("m0_sha256") == rec_sha256 and side.get("head") == env.get("head"))
    clean = env.get("src_or_scripts_dirty") is False or vouched
    if not clean or not code_at(env.get("head"), commit):
        out.append(f"{ent['name']}: its M0 was not computed at {commit[:12]}'s code, clean")
    return out


def read_m0(name: str, inputs: dict) -> tuple[dict, dict | None, str]:
    """(the run's M0 record, its sidecar or None, the record's sha256)."""
    rec = read_json(f"{OUT}/m0/{name}.json", inputs)
    side = (read_json(m0_sidecar(name), inputs) if (ROOT / m0_sidecar(name)).is_file()
            else None)
    return rec, side, inputs[f"{OUT}/m0/{name}.json"]


def check_c1(c1: dict, manifest: dict, commit: str) -> list[str]:
    """vg1_c1.py's record against the manifest (its sha256 and every checkpoint) and the
    reference commit's code."""
    out = []
    pv = c1.get("provenance") or {}
    if c1.get("manifest_sha256") != manifest["_sha256"]:
        out.append("the C1 record was not computed from this manifest")
    if c1.get("partial"):
        out.append("the C1 record is partial")
    if not (c1.get("gate") or {}).get("pass"):
        out.append("the C1 record's wiring gate did not pass")
    if not code_at(pv.get("head"), commit):
        out.append(f"the C1 record was not computed at {commit[:12]}'s code")
    dirty = [c for c in pv.get("changed", []) if any(c.split()[-1].startswith(p)
                                                     for p in CODE_PATHS)]
    if dirty:
        out.append(f"the C1 record was computed with VG-1's code changed: {dirty[:4]}")
    for ent in manifest["runs"]:
        if not ent.get("final"):
            continue
        r = (c1.get("networks") or {}).get(ent["name"])
        if r is None:
            out.append(f"{ent['name']}: not in the C1 record")
        elif (r.get("checkpoint") or {}).get("sha256") != ent["final"]["sha256"]:
            out.append(f"{ent['name']}: the C1 record's checkpoint is not the manifest's")
    return out


def load_manifest(phase: str, inputs: dict) -> dict:
    path = f"{OUT}/manifest-{phase}.json"
    if phase == "main" and not committed(path):
        raise CheckFailed(f"{path} must be committed (Section 9, step 6)")
    m = read_json(path, inputs)
    if m.get("record") != "vg1-manifest" or m.get("phase") != phase:
        raise CheckFailed(f"{path} is not VG-1's {phase} manifest")
    m["_sha256"] = inputs[path]
    for ent in m["runs"]:
        for f in ("config", "metrics", "eval_rdev"):
            want = ent.get(f"{f}_sha256")
            p = ROOT / "runs" / ent["name"] / f"{f}.{'csv' if f == 'metrics' else 'json'}"
            if want is not None and (not p.is_file() or sha256_file(p) != want):
                raise CheckFailed(f"{ent['name']}: {rel(p)} is not the manifest's")
    return m


def network_values(ent: dict, c1: dict | None, m0: dict | None, names: list[str]) -> dict:
    """One network's statistics with the failed-network rule (module docstring 'analyse')."""
    name = ent["name"]
    fail = ent.get("failure") or {}
    if fail.get("class") == "missing":
        return {"name": name, "missing": True, "reason": fail.get("reason")}
    why = []
    if fail.get("class") == "failed":
        why.append(f"failed twice on non-finite numbers ({fail.get('reason')})")
    raw = {"r2c": math.nan, "rho": math.nan, "m0": math.nan, "s4_steer": math.nan,
           "s4_pedal": math.nan, "ev": math.nan}
    if c1 is not None:
        cc = c1.get("c1") or {}
        raw["r2c"], raw["rho"] = num(cc.get("stat")), num(cc.get("spearman_mean"))
        if c1.get("params_finite") is False:
            why.append("a non-finite parameter in its final checkpoint")
        k3 = c1.get("k3") or {}
        if k3.get("pass") is False:
            why.append(f"a numerical K3 failure ({k3.get('failed')})")
    if m0 is not None:
        raw["m0"] = num(m0.get("normalised_residual_median"))
        ap = m0.get("action_projected_residual") or {}
        raw["s4_steer"], raw["s4_pedal"] = num(ap.get("steer")), num(ap.get("pedal"))
    if ent.get("metrics_sha256"):
        last = last_metrics(name)
        raw["ev"] = last.get("explained_var", math.nan)
        if not last or not all(math.isfinite(v) for v in last.values()):
            why.append("a non-finite value in its last metrics.csv row")
    if not math.isfinite(raw["r2c"]):
        why.append("a non-finite R2_c")
    val = {k: (v if math.isfinite(v) else WORST[k]) for k, v in raw.items()}
    comp = None
    if ent.get("eval_rdev_sha256"):
        rows = json.loads((ROOT / "runs" / name / "eval_rdev.json").read_text())
        if list(rows) != names:
            raise CheckFailed(f"{name}: eval_rdev.json's circuits are not R-dev's")
        comp = {n: num(rows[n].get("completion")) for n in names}
    val["completion"] = (0.0 if comp is None else
                         float(sum(1 for v in comp.values() if v > COMPLETED)))
    return {"name": name, "seed": ent["seed"], "arm": ent.get("arm"), "missing": False,
            "failed": bool(why), "failed_because": why, "raw": raw, "value": val,
            "substituted": sorted(k for k in raw if not math.isfinite(raw[k])),
            "completion_by_circuit": comp,
            "gaps": (None if comp is None else
                     {n: num(r.get("gap")) for n, r in json.loads(
                         (ROOT / "runs" / name / "eval_rdev.json").read_text()).items()})}


# ------------------------------------------------------------------------- the tests


def midranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties (infinities included) at their mid-rank; NaN refused."""
    x = np.asarray(x, float)
    if np.isnan(x).any():
        raise ValueError("NaN in a ranked statistic: apply the failed-network rule first")
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def ranksum_exact(x, y) -> dict:
    """The exact Wilcoxon rank-sum test of x against y: W, the sum of x's mid-ranks; its exact
    permutation distribution over the C(n, n_x) ways to choose x's ranks, by the recursion
    over ranks (counts of subsets of each size and doubled sum); p_greater = P(W >= w) and
    p_less = P(W <= w), a tie counting toward p; two_sided = min(1, 2 min of the two)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, n = len(x), len(x) + len(y)
    if nx == 0 or len(y) == 0:
        return {"n_x": nx, "n_y": len(y), "w": math.nan, "p_greater": math.nan,
                "p_less": math.nan, "two_sided": math.nan}
    if n > 60:
        raise ValueError("the exact counts are int64: at most 60 networks")
    r2 = np.rint(2 * midranks(np.concatenate([x, y]))).astype(np.int64)    # doubled: integers
    total = int(r2.sum())
    dp = np.zeros((nx + 1, total + 1), dtype=np.int64)    # [subset size, doubled rank sum]
    dp[0, 0] = 1
    for i, v in enumerate(r2):
        for k in range(min(i + 1, nx), 0, -1):
            dp[k, v:] += dp[k - 1, :total + 1 - v]
    dist = dp[nx]
    if int(dist.sum()) != comb(n, nx):
        raise ArithmeticError("the rank-sum distribution does not count every subset")
    w2 = int(r2[:nx].sum())
    all_ = comb(n, nx)
    pg = int(dist[w2:].sum()) / all_
    pl = int(dist[:w2 + 1].sum()) / all_
    return {"n_x": nx, "n_y": len(y), "w": w2 / 2.0, "expected_w": nx * (n + 1) / 2.0,
            "p_greater": pg, "p_less": pl, "two_sided": min(1.0, 2.0 * min(pg, pl))}


def _diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a - b with a difference of two equal infinities counted as 0."""
    with np.errstate(invalid="ignore"):
        d = a - b
    return np.where(np.isinf(a) & (a == b), 0.0, d)


def _median_sorted(s: np.ndarray) -> np.ndarray:
    """The median along the last axis of sorted values: the mean of the middle two for an even
    count, where an infinite one wins over a finite one."""
    m = s.shape[-1]
    if m % 2:
        return s[..., m // 2]
    lo, hi = s[..., m // 2 - 1], s[..., m // 2]
    with np.errstate(invalid="ignore"):
        mid = (lo + hi) / 2.0
    return np.where(lo == hi, lo, mid)


def hodges_lehmann(x, y) -> float:
    """The median of the pairwise differences x_i - y_j."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x) or not len(y):
        return math.nan
    return float(_median_sorted(np.sort(_diff(x[:, None], y[None, :]).ravel())))


def nearest_rank(draws: np.ndarray, q: float) -> float:
    """The q-quantile by nearest rank (no interpolation, so infinities are safe)."""
    s = np.sort(np.asarray(draws, float))
    return float(s[max(0, math.ceil(q * len(s)) - 1)])


def bootstrap_ci(x, y, stat: str, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """95% percentile CI of the Hodges-Lehmann shift ('hl') or the mean difference ('mean'):
    each draw resamples x, then y, with replacement (default_rng(seed), one stream)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x) or not len(y):
        return {"lo": math.nan, "hi": math.nan, "n_boot": n_boot, "seed": seed}
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(x), size=(n_boot, len(x)))
    iy = rng.integers(0, len(y), size=(n_boot, len(y)))
    xs, ys = x[ix], y[iy]
    if stat == "hl":
        d = _diff(xs[:, :, None], ys[:, None, :]).reshape(n_boot, -1)
        draws = _median_sorted(np.sort(d, axis=1))
    elif stat == "mean":
        with np.errstate(invalid="ignore"):
            draws = xs.mean(1) - ys.mean(1)
    else:
        raise ValueError(stat)
    return {"lo": nearest_rank(draws, 0.025), "hi": nearest_rank(draws, 0.975),
            "n_boot": n_boot, "seed": seed}


def signflip_exact(d) -> dict:
    """The exact sign-flip test of the mean of d over all 2^n flips: p_greater = P(mean >=
    observed), p_less = P(mean <= observed), a tie (within ``PERM_TIE``) counting toward p."""
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        return {"n": 0, "mean": math.nan, "p_greater": math.nan, "p_less": math.nan}
    if not np.isfinite(d).all():
        raise ValueError("the sign-flip test needs finite differences")
    sums = np.zeros(1)
    for v in d:
        sums = np.concatenate([sums + v, sums - v])
    obs = float(d.sum())
    return {"n": n, "mean": obs / n, "flips": len(sums),
            "p_greater": float(np.mean(sums >= obs - PERM_TIE)),
            "p_less": float(np.mean(sums <= obs + PERM_TIE))}


def one_sided(x, y, higher: bool) -> dict:
    """The rank-sum of x (vgl) against y (base) in the direction ``higher``: its p, the
    opposite p, and confirm / refute / inconclusive at ALPHA."""
    t = ranksum_exact(x, y)
    p, q = (t["p_greater"], t["p_less"]) if higher else (t["p_less"], t["p_greater"])
    res = ("confirm" if p <= ALPHA else "refute" if q <= ALPHA else "inconclusive")
    return {**t, "direction": "vgl above" if higher else "vgl below", "p": p, "p_opposite": q,
            "result": res}


def reading(p_result: str, s3_confirm: bool) -> dict:
    """Section 6's first matching row."""
    rows = [(1, lambda: p_result == "refute"),
            (2, lambda: p_result == "confirm" and s3_confirm),
            (3, lambda: p_result == "confirm" and not s3_confirm),
            (4, lambda: p_result in ("inconclusive", "not testable") and s3_confirm),
            (5, lambda: p_result in ("inconclusive", "not testable") and not s3_confirm)]
    row = next(k for k, ok in rows if ok())
    return {"row": row, "P": p_result, "S3": "confirm" if s3_confirm else "not confirm"}


def primary(vgl: list[float], base: list[float]) -> dict:
    """P (Section 5.1): not testable with fewer than MIN_PER_ARM networks in an arm."""
    if len(vgl) < MIN_PER_ARM or len(base) < MIN_PER_ARM:
        return {"result": "not testable", "n_vgl": len(vgl), "n_base": len(base),
                "reason": f"fewer than {MIN_PER_ARM} networks in an arm"}
    out = one_sided(vgl, base, higher=True)
    out["hodges_lehmann"] = hodges_lehmann(vgl, base)
    out["hodges_lehmann_ci"] = bootstrap_ci(vgl, base, "hl")
    return out


def paired(vals: list[dict]) -> dict:
    """S6: the same-seed differences vgl - base in R2_c, -inf replaced by the lowest finite
    R2_c of the networks (reported)."""
    r = [v["value"]["r2c"] for v in vals]
    finite = [x for x in r if math.isfinite(x)]
    floor = min(finite) if finite else math.nan
    by = {}
    for v in vals:
        by.setdefault(v["seed"], {})[v["arm"]] = v["value"]["r2c"]
    subs, diffs = [], []
    for s in sorted(by):
        if set(by[s]) != {"base", "vgl"}:
            continue
        a, b = by[s]["vgl"], by[s]["base"]
        if not math.isfinite(a) or not math.isfinite(b):
            subs.append(s)
        a, b = (x if math.isfinite(x) else floor for x in (a, b))
        diffs.append(a - b)
    out = signflip_exact(diffs)
    out.update(seeds=[s for s in sorted(by) if set(by[s]) == {"base", "vgl"}],
               substituted_seeds=subs, floor=floor,
               result=("confirm" if out["p_greater"] <= ALPHA else
                       "refute" if out["p_less"] <= ALPHA else "inconclusive"))
    return out


def analysis_tests(vals: list[dict]) -> dict:
    """P, S1-S6 and the reading on the networks' values (missing ones already dropped)."""
    arm = {a: [v for v in vals if v["arm"] == a] for a in ("vgl", "base")}
    def col(a, k):
        return [v["value"][k] for v in arm[a]]

    p = primary(col("vgl", "r2c"), col("base", "r2c"))
    s1 = ranksum_exact(col("vgl", "completion"), col("base", "completion"))
    s1.update(direction="two-sided", p=s1["two_sided"],
              result="differs" if s1["two_sided"] <= ALPHA else "no difference shown",
              mean_difference=(float(np.mean(col("vgl", "completion")))
                               - float(np.mean(col("base", "completion"))))
              if arm["vgl"] and arm["base"] else math.nan,
              mean_difference_ci=bootstrap_ci(col("vgl", "completion"),
                                              col("base", "completion"), "mean"))
    s5 = ranksum_exact(col("vgl", "ev"), col("base", "ev"))
    s5.update(direction="two-sided", p=s5["two_sided"],
              result="differs" if s5["two_sided"] <= ALPHA else "no difference shown",
              hodges_lehmann=hodges_lehmann(col("vgl", "ev"), col("base", "ev")))
    s3 = one_sided(col("vgl", "m0"), col("base", "m0"), higher=False)
    out = {"P": p, "S1": s1, "S2": one_sided(col("vgl", "rho"), col("base", "rho"), True),
           "S3": s3, "S4": {"steer": one_sided(col("vgl", "s4_steer"), col("base", "s4_steer"),
                                               False),
                            "pedal": one_sided(col("vgl", "s4_pedal"), col("base", "s4_pedal"),
                                               False)},
           "S5": s5, "S6": paired(vals)}
    out["reading"] = reading(p["result"], s3["result"] == "confirm")
    return out


# ------------------------------------------------------------------------- pilot


def pilot_rule(per: dict[str, dict]) -> dict:
    """Section 4.1's rule on {run: {beta (str), seed, finished, m1, m0, ev}}: eligibility per
    beta, the median M1 of each, beta* (null if none is eligible)."""
    elig, med, why = {}, {}, {}
    for b in BETAS:
        rs = [r for r in per.values() if r["beta"] == b]
        reasons = []
        if len(rs) != 2:
            reasons.append(f"{len(rs)} runs, not 2")
        for r in rs:
            vals = [num(r.get(k)) for k in ("m1", "m0", "ev")]
            if not r.get("finished"):
                reasons.append(f"{r['name']} did not finish")
            elif not all(math.isfinite(v) for v in vals):
                reasons.append(f"{r['name']} has a non-finite metric")
            else:
                if vals[2] < EV_MIN:
                    reasons.append(f"{r['name']} EV {vals[2]:.4f} < {EV_MIN}")
                if not vals[1] < M0_MAX:
                    reasons.append(f"{r['name']} M0 {vals[1]:.3f} >= {M0_MAX}")
        elig[b], why[b] = not reasons, reasons
        m1 = [num(r.get("m1")) for r in rs]
        med[b] = float(np.median(m1)) if rs and all(math.isfinite(v) for v in m1) else math.nan
    ok = [b for b in BETAS if elig[b]]
    star = None
    if ok:
        best = max(med[b] for b in ok)
        star = min((b for b in ok if med[b] == best), key=float)    # a tie: the smaller beta
    return {"eligible": elig, "ineligible_because": why, "median_m1": med,
            "beta_star": None if star is None else float(star), "beta_star_text": star}


def cost_check(beta_star: str | None, per: dict[str, dict], box_gpu_hours: float,
               main_gpus: int) -> dict:
    """Section 4.1's cost check, in GPU-hours with idle GPUs included."""
    if beta_star is None:
        return {"applies": False, "reason": "no beta is eligible: the main run does not start"}
    t = [r["gpu_minutes"] for r in per.values() if r["beta"] == beta_star]
    t_star = float(np.mean(t))
    main = (MAIN_RUNS_PER_ARM * BASE_GPU_MIN + MAIN_RUNS_PER_ARM * t_star
            + SETUP_MIN * main_gpus) / 60.0
    total = box_gpu_hours + main
    return {"applies": True, "pilot_gpu_hours": box_gpu_hours, "beta_star_gpu_minutes": t_star,
            "main_projected_gpu_hours": main, "main_gpus": main_gpus, "total": total,
            "limit": COST_LIMIT_H, "to_the_user": bool(total > COST_LIMIT_H),
            "stop_rule": STOP_RULE_H, "formula": "20 x 6.5 min + 20 x beta*'s mean GPU minutes "
                                                 "+ 15 min x the main box's GPUs"}


def cmd_pilot(a) -> None:
    t0 = time.time()
    path = ROOT / OUT / "pilot.json"
    if path.exists() or ever_committed(f"{OUT}/pilot.json"):
        raise CheckFailed(f"{OUT}/pilot.json exists or was committed: the pilot rule runs once")
    inputs: dict[str, str] = {}
    m = load_manifest("pilot", inputs)
    commit = m["commit"]
    c1 = read_json(f"{OUT}/c1/pilot.json", inputs)
    bad = check_c1(c1, m, commit)
    gm = gpu_minutes(m["runs"])
    qh = queue_gpu_hours(m["runs"])
    if a.box_gpu_hours < qh:
        bad.append(f"--box-gpu-hours {a.box_gpu_hours} is below the queue's {qh:.2f} GPU-hours")
    per = {}
    for ent in m["runs"]:
        name = ent["name"]
        r = {"name": name, "beta": next(b for b in BETAS if float(b) == ent["beta"]),
             "seed": ent["seed"], "status": ent["status"],
             "finished": ent["status"] in ("done", "retried") and bool(ent.get("final")),
             "gpu_minutes": gm.get(name, math.nan), "m1": None, "m0": None, "ev": None}
        if r["finished"]:
            cn = (c1.get("networks") or {}).get(name) or {}
            r["m1"] = (cn.get("c1") or {}).get("stat")
            r["k3_pass"], r["params_finite"] = (cn.get("k3") or {}).get("pass"), cn.get(
                "params_finite")
            m0, side, msha = read_m0(name, inputs)
            bad += check_m0(m0, ent, commit, side, msha)
            r["m0"] = m0.get("normalised_residual_median")
            r["s4"] = m0.get("action_projected_residual")
            r["ev"] = last_metrics(name).get("explained_var")
        per[name] = r
    if bad:
        raise CheckFailed("the pilot's inputs do not check:\n  " + "\n  ".join(bad))
    rule = pilot_rule(per)
    cost = cost_check(rule["beta_star_text"], per, a.box_gpu_hours, a.main_gpus)
    rec = {"record": "vg1-pilot", "label": "exploratory (the pilot; never confirmatory)",
           "pilot_commit": commit, "runs": per, **rule, "cost_check": cost,
           "queue_gpu_hours": qh, "rule": {"ev_min": EV_MIN, "m0_max_exclusive": M0_MAX,
                                           "betas": list(BETAS), "tie": "the smaller beta"},
           "provenance": provenance(t0, inputs)}
    write_new(path, rec)
    print(f"beta* = {rule['beta_star_text']}; eligible {rule['eligible']}; cost "
          f"{cost.get('total', math.nan):.2f} GPU-h (to the user: {cost.get('to_the_user')})")


# ------------------------------------------------------------------------- analyse


def descriptives(vals: list[dict], c1: dict, m0s: dict, gm: dict, names: list[str]) -> dict:
    """Section 5.3 (decides nothing)."""
    def med(xs):
        xs = [x for x in xs if x is not None and math.isfinite(x)]
        return float(np.median(xs)) if xs else math.nan

    out = {"per_network": {v["name"]: {k: v.get(k) for k in ("arm", "seed", "failed",
                                                              "failed_because", "raw",
                                                              "completion_by_circuit")}
                           for v in vals}}
    arms = {a: [v for v in vals if v["arm"] == a] for a in ("base", "vgl")}
    per_arm = {}
    for a, vs in arms.items():
        cn = [(c1.get("networks") or {}).get(v["name"], {}).get("c1") or {} for v in vs]
        mm = [m0s.get(v["name"]) or {} for v in vs]
        comps = next((x["components"] for x in mm if x.get("components")), [])
        per_arm[a] = {
            "n": len(vs), "failed": sum(v["failed"] for v in vs),
            "median": {k: med([v["raw"][k] for v in vs]) for k in WORST},
            "median_completion": med([v["value"]["completion"] for v in vs]),
            "r2c_component": {c: med([(x.get("r2c") or {}).get(c) for x in cn])
                              for c in ("s", "xi")},
            "spearman_component": {c: med([(x.get("spearman") or {}).get(c) for x in cn])
                                   for c in ("s", "xi")},
            "m0_component": {k: med([(x.get("normalised_residual") or {}).get(k) for x in mm])
                             for k in comps},
            "action_projection_share_target": {
                act: {k: med([((x.get("action_projection_share_target") or {}).get(act) or {})
                              .get(k) for x in mm]) for k in comps}
                for act in ("steer", "pedal")},
            "completion_share": {n: (float(np.mean([v["completion_by_circuit"][n] > COMPLETED
                                                    for v in vs if v["completion_by_circuit"]]))
                                     if any(v["completion_by_circuit"] for v in vs) else math.nan)
                                 for n in names},
            "gpu_minutes_per_update": med([gm.get(v["name"], math.nan)
                                           / (FINAL_TIMESTEPS / (2048 * 64)) for v in vs])}
    out["per_arm"] = per_arm
    both = [n for n in names if all(
        any(v["completion_by_circuit"] and v["completion_by_circuit"][n] > COMPLETED for v in vs)
        for vs in arms.values())]
    out["gap_on_circuits_both_arms_complete"] = {
        n: {a: med([v["gaps"][n] for v in vs if v["completion_by_circuit"]
                    and v["completion_by_circuit"][n] > COMPLETED]) for a, vs in arms.items()}
        for n in both}
    return out


def cmd_analyse(a) -> None:
    t0 = time.time()
    out_p, lock_p = ROOT / OUT / "analysis.json", ROOT / OUT / "analysis.lock"
    for p in (out_p, lock_p):
        if p.exists() or ever_committed(f"{OUT}/{p.name}"):
            raise CheckFailed(f"{rel(p)} exists or was committed: the analysis runs once")
    tag = tag_gate()
    inputs: dict[str, str] = {}
    pilot = read_json(f"{OUT}/pilot.json", inputs)
    if not committed(f"{OUT}/pilot.json"):
        raise CheckFailed(f"{OUT}/pilot.json is not committed")
    failures = read_failures("main", inputs)
    m = load_manifest("main", inputs)
    if m.get("tag") != TAG or m["commit"] != tag["commit"]:
        raise CheckFailed(f"the manifest is of {m.get('tag')} ({m['commit'][:12]}), not {TAG}")
    listed = {e["name"]: (e.get("failure") or {}).get("class") for e in m["runs"]
              if e.get("failure")}
    if {k: v.get("class") for k, v in failures.items()} != listed:
        raise CheckFailed("failures.json and the manifest classify different runs")
    check = round_check("main", TAG, failures)
    want = [r["name"] for r in load_runs_script().runs("main", ROOT)]
    if [e["name"] for e in m["runs"]] != want:
        raise CheckFailed("the manifest's runs are not the run list's")
    for ent in m["runs"]:
        if ent.get("final"):
            p = ROOT / ent["final"]["path"]
            if not p.is_file() or sha256_file(p) != ent["final"]["sha256"]:
                raise CheckFailed(f"{ent['name']}: its final checkpoint is not the manifest's")
    c1 = read_json(f"{OUT}/c1/main.json", inputs)
    bad = check_c1(c1, m, tag["commit"])
    m0s = {}
    for ent in m["runs"]:
        p = f"{OUT}/m0/{ent['name']}.json"
        if ent.get("final") and (ROOT / p).is_file():
            m0s[ent["name"]], side, msha = read_m0(ent["name"], inputs)
            bad += check_m0(m0s[ent["name"]], ent, tag["commit"], side, msha)
        elif ent.get("final"):
            bad.append(f"{ent['name']}: no M0 record {p}")
    if bad:
        raise CheckFailed("the main run's inputs do not check:\n  " + "\n  ".join(bad))
    fd = os.open(lock_p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    os.write(fd, f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {tag['commit']}\n".encode())
    os.close(fd)
    names = r_dev()
    allv = [network_values(e, (c1.get("networks") or {}).get(e["name"]), m0s.get(e["name"]),
                           names) for e in m["runs"]]
    vals = [v for v in allv if not v["missing"]]
    tests = analysis_tests(vals)
    counts = {a_: {"failed": sum(v["failed"] for v in vals if v["arm"] == a_),
                   "missing": sum(v["missing"] for v in allv
                                  if v["name"].startswith(f"vg1-{a_}-"))}
              for a_ in ("base", "vgl")}
    rec = {"record": "vg1-analysis", "label": "confirmatory (P) and secondary (S1-S6, "
                                              "unadjusted); descriptives decide nothing",
           "tag": tag, "beta_star": pilot.get("beta_star"), "counts": counts,
           "missing": [v["name"] for v in allv if v["missing"]],
           "tests": tests, "values": {v["name"]: v.get("value") for v in vals},
           "descriptives": descriptives(vals, c1, m0s, gpu_minutes(m["runs"]), names),
           "pilot": {k: pilot.get(k) for k in ("runs", "beta_star", "median_m1", "eligible")},
           "round_check": {k: check[k] for k in ("returncode", "tolerated", "report_sha256")},
           "provenance": provenance(t0, inputs)}
    write_new(out_p, rec)
    print(f"P: {tests['P']['result']} (p = {tests['P'].get('p')}); reading row "
          f"{tests['reading']['row']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="the phase's manifest, after pull-round")
    m.add_argument("--phase", choices=sorted(ROUNDS), required=True)
    m.add_argument("--tag", required=True, help="the pilot commit's sha, or vg1-prereg-v1")
    m.add_argument("--box-sums", nargs="+", required=True,
                   help="sha256sum output taken on the box(es) before deletion")
    p = sub.add_parser("pilot", help="the pilot rule and the cost check")
    p.add_argument("--box-gpu-hours", type=float, required=True,
                   help="the pilot box's lifetime x its GPUs, setup and pull-back included")
    p.add_argument("--main-gpus", type=int, default=4, help="the main run's box GPUs")
    r = sub.add_parser("m0", help="vgl.py residual on each run of the manifest (M0 and S4)")
    r.add_argument("--phase", choices=sorted(ROUNDS), required=True)
    r.add_argument("--run", action="append", help="only this run (repeatable)")
    r.add_argument("--threads", type=int, default=4)
    sub.add_parser("analyse", help="the main analysis (runs once, after the tag)")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    {"manifest": cmd_manifest, "m0": cmd_m0, "pilot": cmd_pilot, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
