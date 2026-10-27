"""Maze E5 (PA-3): JaxUED's and SFL's own maze agents, their level scores against exact regret.

    uv run python scripts/maze_e5.py train --arm sfl --view 13 --seed 1 --updates 30000 \\
        --name maze-e5-sfl-v13-s1                         # one run (the run list's cmd)
    uv run python scripts/maze_e5.py timing --rounds maze-e5t4 maze-e5t8 \\
        --audit-round maze-e5at --capture-round maze-e5c  # the calibration's projection
    uv run python scripts/maze_e5.py manifest --train-tag maze-e5-train-v1
    uv run python scripts/maze_e5.py audit --run maze-e5-sfl-v13-s1 --name maze-e5a-sfl-v13-s1
    uv run python scripts/maze_e5.py analyse --workers 4

The pre-registration is docs/prereg/maze-e5.md; scripts/maze_e5_runs.py holds the run lists
and scripts/gpu_round.sh runs them on the boxes. Every stage first checks the vendored
files against their pinned git blobs (``apexzero.maze.sfl.check_vendored``).

- ``train`` (on the box, from the run list): runs JaxUED's maze_plr.py (through
  run_jaxued_maze.py) or SFL's minigrid_sfl.py (through run_sfl_maze.py) in runs/<name>/,
  with exactly the flags Section 2 fixes for the arm (``trainer_cmd``). JaxUED saves every
  100-update block (DEV-E5-1); once it exits, every orbax step but the selected ones is
  deleted, so only the 16 come back. It writes runs/<name>/e5-run.json: the trainer's
  command, the vendored blobs, start and end, wall and CPU time, each 100-update block's end
  (the calibration's raw data) and each kept checkpoint's update, path and sha256 (over its
  files). It imports no JAX, so the queue's lanes hold only the trainers' GPU memory.
- ``timing`` (on the Mac, after the calibration rounds are pulled back): the projection rule
  of Section 9 (``apexzero.maze.e5.project_queue``). It writes runs/maze-e5-timing.json and
  prints the projected GPU-hours per lane count and box layout; the user decides on them.
- ``manifest`` (on the Mac, after pull-round, before the tag): round_manifest.py's check of
  round ``maze-e5`` at the training tag (a run that failed twice is accepted as missing,
  Section 8), then E5's own: each run's e5-run.json and trainer command are what its cell
  fixes, its recorded config equals that command (maze_plr.py's parser is rebuilt from the
  vendored source; SFL's config is recomposed), and each checkpoint's files match their
  sha256. Writes outputs/maze-e5/manifest.json (never overwritten), to be committed.
- ``audit`` (on a box, behind the tag gate, one job per run: round ``maze-e5a``): the
  committed manifest at the tag, every checkpoint's sha256 checked before it is loaded; the
  fresh audit on 2,000 mazes (and, for SFL, on 2,000 of its own generator's), the in-buffer
  audit for PLR-MaxMC and SFL. One npz per checkpoint in runs/<name>/, never overwritten.
  ``--timing-only`` audits one calibration checkpoint and writes only its timing.
- ``analyse`` (on the Mac, behind the tag gate, once): round_manifest.py's check of round
  ``maze-e5a`` at the tag, every audit file against its recorded sha256, then a lock that is
  never removed, then Sections 4-5 (``apexzero.maze.e5``). It refuses if the lock or an
  output exists. Writes outputs/maze-e5/analyse/{analysis.json, decisions.md}.

**The tag gate** (``tag_gate``, scripts/lockbox.py's rule for ``maze-e5-prereg-v1``):
PermissionError unless the annotated tag exists on origin, the local tag is the same tag
object, HEAD descends from its commit, and E5's code (``CODE_PATHS``, untracked files
included) is the tagged code. The manifest the audit reads must be tracked and byte-identical
to the tag commit's copy.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)

from apexzero.maze import sfl

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "outputs" / "maze-e5"
MANIFEST = OUT / "manifest.json"
TAG = "maze-e5-prereg-v1"
TRAIN_TAG = "maze-e5-train-v1"
CODE_PATHS = ("scripts/maze_e5.py", "scripts/maze_e5_runs.py", "scripts/run_sfl_maze.py",
              "scripts/run_jaxued_maze.py", "scripts/round_manifest.py", "scripts/_bootstrap.py",
              "src/apexzero/maze", "src/apexzero/audit/estimators.py", "src/apexzero/compat.py",
              "third_party/jaxued_examples", "third_party/sfl")
EVAL_FREQ = 100                      # updates per block (JaxUED --eval_freq; SFL's EVAL_FREQ)
JAXUED_FLAGS = {"dr": ["--replay_prob", "0", "--exploratory_grad_updates"], "plr": []}
TIME = "%Y-%m-%dT%H:%M:%SZ"
MAX_WORKERS = 4


def checkpoint_updates() -> list[int]:
    return [100 * round(3 * 100 ** ((k - 1) / 15)) for k in range(1, 17)]


def now() -> str:
    return datetime.now(UTC).strftime(TIME)


def rel(path: Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def tree_sha256(d: Path) -> str:
    """sha256 over a directory's files: each relative path and the sha256 of its bytes."""
    h = hashlib.sha256()
    for p in sorted(x for x in Path(d).rglob("*") if x.is_file()):
        h.update(p.relative_to(d).as_posix().encode() + b"\0"
                 + sha256_bytes(p.read_bytes()).encode() + b"\n")
    return h.hexdigest()


def write_new(path: Path, data: bytes) -> str:
    """Write ``data`` to a path that must not exist yet (atomic); returns its sha256."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{rel(path)} exists: E5 never overwrites an output")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()
    return sha256_bytes(data)


def write_json_new(path: Path, obj) -> str:
    from apexzero.maze.e5 import finite_or_none

    return write_new(path, (json.dumps(finite_or_none(obj), indent=1, allow_nan=False)
                            + "\n").encode())


def lower_priority(level: int = 10) -> None:
    try:
        if os.getpriority(os.PRIO_PROCESS, 0) < level:
            os.setpriority(os.PRIO_PROCESS, 0, level)
    except OSError:
        pass


# ------------------------------------------------------------------------- git and the tag


def git_run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def tag_gate(tag: str = TAG, paths=CODE_PATHS) -> dict:
    """scripts/lockbox.py's ``tag_gate`` for E5's tag and code: PermissionError unless the
    annotated tag exists on origin, the local tag is the same tag object, HEAD descends from
    its commit, and ``paths`` (untracked files included) are the tagged code."""
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
        raise PermissionError(f"{tag} is not on origin")
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
        raise PermissionError(f"E5's code ({', '.join(paths)}) differs from {tag}'s")
    untracked = git_run("ls-files", "--others", "--exclude-standard", "--", *paths)
    if untracked.returncode != 0 or untracked.stdout.strip():
        raise PermissionError(f"untracked files under E5's code: {untracked.stdout!r}")
    return {"tag": tag, "tag_object": obj, "commit": peeled,
            "head": git_run("rev-parse", "HEAD").stdout.strip()}


def tagged_file(path: Path, commit: str) -> bytes:
    """``path``'s bytes, which must be tracked and identical to ``commit``'s copy."""
    r = rel(path)
    if Path(path).is_symlink() or not Path(path).is_file():
        raise SystemExit(f"{r} must be a regular file")
    tracked = git_run("ls-files", "--error-unmatch", "--", r).returncode == 0
    shown = subprocess.run(["git", "show", f"{commit}:{r}"], cwd=ROOT, capture_output=True,
                           check=False)
    raw = Path(path).read_bytes()
    if not tracked or shown.returncode != 0 or shown.stdout != raw:
        raise SystemExit(f"{r} must be tracked and byte-identical to the tagged copy")
    return raw


# ------------------------------------------------------------------------- the trainers


def selected(updates: int) -> list[int]:
    return [u for u in checkpoint_updates() if u <= updates]


def trainer_cmd(name: str, arm: str, view: int, seed: int, updates: int,
                save: list[int] | None = None, capture: bool = True) -> list[str]:
    """The trainer's command (relative to the repo) that Section 2 fixes for a run."""
    if updates % EVAL_FREQ:
        raise SystemExit(f"--updates must be a multiple of {EVAL_FREQ}")
    save = selected(updates) if save is None else save
    if arm in JAXUED_FLAGS:
        if not capture:
            raise SystemExit("--no-capture is for SFL runs")
        return ["scripts/run_jaxued_maze.py", "--run_name", name, "--seed", str(seed),
                "--agent_view_size", str(view), "--num_updates", str(updates),
                "--eval_freq", str(EVAL_FREQ), "--checkpoint_save_interval", "1",
                "--max_number_of_checkpoints", str(updates // EVAL_FREQ), *JAXUED_FLAGS[arm]]
    if arm == "sfl":
        return ["scripts/run_sfl_maze.py", *sfl.e5_overrides(seed, view, updates),
                "--save-updates", ",".join(map(str, save)), *([] if capture else ["--no-capture"])]
    raise SystemExit(f"unknown arm {arm!r}")


def jaxued_dir(run_dir: Path, name: str, seed: int) -> Path:
    return Path(run_dir) / "checkpoints" / name / str(seed)


def jaxued_steps(ckdir: Path) -> dict[int, Path]:
    models = Path(ckdir) / "models"
    if not models.is_dir():
        return {}
    return {int(p.name): p for p in models.iterdir()
            if p.name.isdigit() and (p / "commit_success.txt").is_file()}


def child_cpu() -> float:
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime


def stage_train(a) -> int:
    blobs = sfl.check_vendored(network=a.arm == "sfl")
    save = [int(u) for u in a.save.split(",")] if a.save else selected(a.updates)
    cmd = trainer_cmd(a.name, a.arm, a.view, a.seed, a.updates, save, not a.no_capture)
    run_dir = RUNS / a.name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{rel(run_dir)} exists and is not empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    print("E5 train:", " ".join(cmd), flush=True)
    t0, c0, start = time.time(), child_cpu(), now()
    proc = subprocess.run([sys.executable, str(ROOT / cmd[0]), *cmd[1:]], cwd=run_dir,
                          check=False, stdin=subprocess.DEVNULL)
    wall, cpu = time.time() - t0, child_cpu() - c0
    record = {"name": a.name, "arm": a.arm, "view": a.view, "seed": a.seed,
              "updates": a.updates, "selected": save, "capture": not a.no_capture,
              "trainer_cmd": cmd, "vendored": blobs, "start": start, "end": now(),
              "start_unix": t0, "wall_s": wall, "cpu_s": cpu, "exit": proc.returncode,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "xla_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")}
    problems = [] if proc.returncode == 0 else [f"the trainer exited {proc.returncode}"]
    if a.arm == "sfl":
        log = run_dir / "sfl-eval.jsonl"
        lines = [json.loads(x) for x in log.read_text().splitlines()] if log.is_file() else []
        record["block_ends_unix"] = [x["time"] for x in lines]
        record["eval"] = lines
        found = sfl.checkpoint_updates(run_dir)
        record["checkpoints"] = [{"update": u, "path": rel(sfl.checkpoint_dir(run_dir, u)),
                                  "sha256": tree_sha256(sfl.checkpoint_dir(run_dir, u))}
                                 for u in found]
    else:
        steps = jaxued_steps(jaxued_dir(run_dir, a.name, a.seed))
        record["block_ends_unix"] = [(steps[k] / "commit_success.txt").stat().st_mtime
                                     for k in sorted(steps)]
        keep = {u // EVAL_FREQ - 1 for u in save}
        if proc.returncode == 0:
            for k in sorted(set(steps) - keep):          # DEV-E5-1: only the 16 come back
                shutil.rmtree(steps[k])
            record["pruned"] = len(set(steps) - keep)
        found = sorted((k + 1) * EVAL_FREQ for k in steps if k in keep)
        record["checkpoints"] = [{"update": u, "step": u // EVAL_FREQ - 1,
                                  "path": rel(steps[u // EVAL_FREQ - 1]),
                                  "sha256": tree_sha256(steps[u // EVAL_FREQ - 1])}
                                 for u in found]
    if proc.returncode == 0 and found != sorted(save):
        problems.append(f"checkpoints at {found}, expected {sorted(save)}")
    record["problems"] = problems
    write_new(run_dir / "e5-run.json", (json.dumps(record, indent=1) + "\n").encode())
    print(f"E5 train: {wall / 60:.1f} min wall, {len(found)} checkpoints"
          + (f"; FAILED: {problems}" if problems else ""), flush=True)
    return 1 if problems else 0


# ------------------------------------------------------------------------- config checks


def maze_plr_parser():
    """maze_plr.py's argument parser, rebuilt from the vendored source's ``add_argument``
    calls (literal arguments; ``argparse.BooleanOptionalAction`` is the one non-literal),
    and the dests of its 'Training params' group, which make its ``group_name``."""
    src = (ROOT / "third_party" / "jaxued_examples" / "maze_plr.py").read_text()
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    group = []
    types = {"int": int, "float": float, "str": str}
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"]
    for call in sorted(calls, key=lambda n: (n.lineno, n.col_offset)):
        flags = [ast.literal_eval(x) for x in call.args]
        kw = {}
        for k in call.keywords:
            if k.arg == "type":
                kw["type"] = types[k.value.id]
            elif k.arg == "action":
                if ast.unparse(k.value) != "argparse.BooleanOptionalAction":
                    raise ValueError(f"{flags[0]}: unexpected action {ast.unparse(k.value)}")
                kw["action"] = argparse.BooleanOptionalAction
            elif k.arg != "help":
                kw[k.arg] = ast.literal_eval(k.value)
        action = parser.add_argument(*flags, **kw)
        receiver = getattr(call.func.value, "id", "")
        if receiver in ("group", "mut_group"):
            group.append(action.dest)
    return parser, sorted(group)


def _same(a, b) -> bool:
    return isinstance(a, bool) == isinstance(b, bool) and a == b


def expected_jaxued_config(trainer_args: list[str]) -> dict:
    """What maze_plr.py writes to config.json for these arguments: vars(args), num_updates
    from num_env_steps if given, and group_name."""
    parser, group = maze_plr_parser()
    cfg = vars(parser.parse_args(trainer_args))
    if cfg["num_env_steps"] is not None:
        cfg["num_updates"] = cfg["num_env_steps"] // (cfg["num_train_envs"] * cfg["num_steps"])
    cfg["group_name"] = "".join(str(cfg[k]) for k in group)
    return cfg


def config_problems(run: dict, record: dict) -> tuple[list[str], str | None]:
    """A run's e5-run.json against its run-list entry, and its trainer's recorded config
    against that trainer command; returns (problems, the config's path)."""
    p = []
    for k in ("arm", "view", "seed", "updates"):
        if record.get(k) != run[k]:
            p.append(f"e5-run.json records {k} {record.get(k)!r}, the run list {run[k]!r}")
    if p:
        return p, None
    want = trainer_cmd(run["name"], run["arm"], run["view"], run["seed"], run["updates"])
    if record.get("trainer_cmd") != want:
        p.append(f"trainer command {record.get('trainer_cmd')} is not Section 2's {want}")
    if record.get("exit") != 0 or record.get("problems"):
        p.append(f"the run recorded exit {record.get('exit')} and {record.get('problems')}")
    if record.get("selected") != selected(run["updates"]):
        p.append(f"checkpoints selected {record.get('selected')}")
    run_dir = RUNS / run["name"]
    if run["arm"] == "sfl":
        path = run_dir / "config.json"
        if not path.is_file():
            return p + [f"{rel(path)} is not here"], None
        cfg = json.loads(path.read_text())
        if cfg.get("overrides") != want[1:5] or cfg.get("config") != sfl.compose_config(
                want[1:5]) or cfg.get("network") != "vendored":
            p.append("config.json is not SFL's composed config with the run's overrides")
        if cfg.get("save_updates") != selected(run["updates"]) or not cfg.get(
                "capture_learnable_set"):
            p.append("config.json: save_updates or capture differ from Section 2's")
        return p, rel(path)
    path = jaxued_dir(run_dir, run["name"], run["seed"]) / "config.json"
    if not path.is_file():
        return p + [f"{rel(path)} is not here"], None
    got, exp = json.loads(path.read_text()), expected_jaxued_config(want[1:])
    for k in sorted(set(got) | set(exp)):
        if k not in got or k not in exp or not _same(got[k], exp[k]):
            p.append(f"config.json {k}: {got.get(k)!r}, the command gives {exp.get(k)!r}")
    return p, rel(path)


# ------------------------------------------------------------------------- rounds


def runs_list(phase: str, *args: str) -> list[dict]:
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "maze_e5_runs.py"),
                          "--phase", phase, *args], cwd=ROOT, capture_output=True, text=True,
                         check=True)
    return [json.loads(x) for x in out.stdout.splitlines() if x.strip()]


def round_check(round_: str, tag: str, runs_args: list[str], allow_failed: bool) -> list[str]:
    """round_manifest.py check; returns the runs that failed twice (accepted as missing only
    with ``allow_failed``: every other problem stops the stage)."""
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "round_manifest.py"), "check",
                           "--round", round_, "--runs-script", "scripts/maze_e5_runs.py",
                           "--tag", tag, "--runs-args", *runs_args],
                          cwd=ROOT, capture_output=True, text=True, check=False)
    print(proc.stdout.rstrip())
    return parse_check(round_, proc.returncode, proc.stdout, proc.stderr, allow_failed)


FAILED_RUN = re.compile(r"(\S+): status 'failed' in .+")


def parse_check(round_: str, code: int, stdout: str, stderr: str, allow_failed: bool) -> list[str]:
    """The runs check reports as failed twice, if those are its only problems and
    ``allow_failed``; otherwise SystemExit with the report."""
    if code == 0:
        return []
    problems = [x.strip() for x in stdout.split("FAILED:", 1)[-1].splitlines()[1:] if x.strip()]
    failed = [m[1] for x in problems if (m := FAILED_RUN.fullmatch(x))]
    other = [x for x in problems if not FAILED_RUN.fullmatch(x)]
    if not allow_failed or other or code != 1 or not failed:
        raise SystemExit(f"round {round_} does not pass round_manifest.py check:\n"
                         + "\n".join(other or problems or [stderr.strip()]))
    return failed


def shard_manifests(round_: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RUNS.glob(f"{round_}-manifest-*of*.json"))]


def stage_manifest(a) -> None:
    sfl.check_vendored()
    args = ["--phase", "main", "--drop", str(a.drop)]
    failed = round_check("maze-e5", a.train_tag, args, allow_failed=True)
    queue = {r["name"]: r for m in shard_manifests("maze-e5") for r in m["runs"]}
    boxes = [{k: m.get(k) for k in ("shard", "head", "box_type", "gpu_model", "gpu_count",
                                    "gpus", "lanes_per_gpu", "written")}
             for m in shard_manifests("maze-e5")]
    entries, problems = {}, []
    for run in runs_list("main", "--drop", str(a.drop)):
        name = run["name"]
        entry = {k: run[k] for k in ("arm", "view", "seed", "updates")}
        entry["queue"] = {k: queue.get(name, {}).get(k) for k in ("status", "start", "end",
                                                                  "lane", "gpu")}
        if name in failed:
            entries[name] = {**entry, "status": "missing", "checkpoints": []}
            continue
        path = RUNS / name / "e5-run.json"
        if not path.is_file():
            problems.append(f"{name}: {rel(path)} is not here")
            continue
        record = json.loads(path.read_text())
        found, cfg = config_problems(run, record)
        problems += [f"{name}: {x}" for x in found]
        for ck in record.get("checkpoints", []):
            d = ROOT / ck["path"]
            if not d.is_dir() or tree_sha256(d) != ck["sha256"]:
                problems.append(f"{name}: {ck['path']} does not match its sha256")
        if [c["update"] for c in record.get("checkpoints", [])] != selected(run["updates"]):
            problems.append(f"{name}: not the 16 selected checkpoints")
        entries[name] = {**entry, "status": "trained", "config": cfg,
                         "config_sha256": sha256_bytes((ROOT / cfg).read_bytes()) if cfg else None,
                         "e5_run_sha256": sha256_bytes(path.read_bytes()),
                         "wall_s": record.get("wall_s"), "checkpoints": record["checkpoints"],
                         "eval_tail": (record.get("eval") or [])[-1:]}
    if problems:
        raise SystemExit(f"manifest: {len(problems)} problem(s):\n  " + "\n  ".join(problems))
    head = git_run("rev-parse", f"{a.train_tag}^{{commit}}").stdout.strip()
    manifest = {"round": "maze-e5", "train_tag": a.train_tag, "train_commit": head,
                "drop": a.drop, "written": now(), "vendored": sfl.check_vendored(),
                "boxes": boxes, "missing": failed, "runs": entries}
    write_json_new(MANIFEST, manifest)
    n = sum(e["status"] == "trained" for e in entries.values())
    print(f"wrote {rel(MANIFEST)}: {n} runs trained, {len(failed)} missing; commit it before "
          f"the tag {TAG}")


# ------------------------------------------------------------------------- the audit


def load_any(arm: str, ckpt: dict, run_dir: Path, seed: int, name: str):
    """(checkpoint, policy, config) of one manifest checkpoint."""
    from apexzero.maze import audit as maze

    if arm == "sfl":
        ck = sfl.load_sfl_checkpoint(run_dir, ckpt["update"])
        return ck, sfl.sfl_network_policy(ck.params), ck.config
    ck = maze.load_checkpoint(jaxued_dir(run_dir, name, seed), ckpt["step"])
    if ck.num_updates != ckpt["update"]:
        raise SystemExit(f"{name}: step {ckpt['step']} holds {ck.num_updates} updates")
    return ck, maze.network_policy(ck.params), ck.config


def audit_one(arm: str, ck, policy, config, run_index: int, update: int, levels: dict,
              oracles: dict) -> dict:
    """Everything the analysis reads for one checkpoint, as arrays (fresh, own, in-buffer)."""
    import jax

    from apexzero.maze import e5_audit as ea

    st = ea.settings(arm, config)
    key = ea.audit_key(run_index, update)
    out = {"fresh": ea.audit_policy(policy, st, levels["jaxued"], oracles[st["gamma"], "jaxued"],
                                    key)}
    if arm == "sfl":
        out["own"] = ea.audit_policy(policy, st, levels["own"], oracles[st["gamma"], "own"],
                                     jax.random.fold_in(key, 4))
    buf = ea.inbuffer_levels(arm, ck, run_index, update)
    if buf is not None:
        out["fresh"].update(ea.audit_inbuffer(policy, st, *buf, jax.random.fold_in(key, 3)))
    return out


def audit_levels(arms, configs) -> tuple[dict, dict]:
    from apexzero.maze import audit as maze
    from apexzero.maze import e5_audit as ea

    levels = {"jaxued": ea.fresh_levels()}
    oracles = {}
    for arm, cfg in zip(arms, configs):
        st = ea.settings(arm, cfg)
        if (st["gamma"], "jaxued") not in oracles:
            oracles[st["gamma"], "jaxued"] = maze.bfs_optimum(levels["jaxued"], st["gamma"],
                                                              st["max_steps"])
        if arm == "sfl":
            levels.setdefault("own", ea.fresh_levels(n_walls=st["n_walls"]))
            if (st["gamma"], "own") not in oracles:
                oracles[st["gamma"], "own"] = maze.bfs_optimum(levels["own"], st["gamma"],
                                                               st["max_steps"])
    return levels, oracles


def npz_bytes(**arrays) -> bytes:
    import io

    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def stage_audit(a) -> None:
    sfl.check_vendored()
    out_dir = RUNS / a.name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{rel(out_dir)} exists and is not empty")
    if a.timing_only:
        return audit_timing(a, out_dir)
    gate = tag_gate()
    manifest = json.loads(tagged_file(MANIFEST, gate["commit"]))
    names = list(manifest["runs"])
    entry = manifest["runs"].get(a.run)
    if entry is None or entry["status"] != "trained":
        raise SystemExit(f"{a.run} is not a trained run in the manifest")
    run_dir = RUNS / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    first = entry["checkpoints"][0]
    if tree_sha256(ROOT / first["path"]) != first["sha256"]:
        raise SystemExit(f"{first['path']} does not match the manifest's sha256")
    _, _, cfg0 = load_any(entry["arm"], first, run_dir, entry["seed"], a.run)
    levels, oracles = audit_levels([entry["arm"]], [cfg0])
    files = {}
    for ckpt in entry["checkpoints"]:
        if tree_sha256(ROOT / ckpt["path"]) != ckpt["sha256"]:
            raise SystemExit(f"{ckpt['path']} does not match the manifest's sha256")
        t0 = time.time()
        ck, policy, cfg = load_any(entry["arm"], ckpt, run_dir, entry["seed"], a.run)
        res = audit_one(entry["arm"], ck, policy, cfg, names.index(a.run), ckpt["update"],
                        levels, oracles)
        for part, arrays in res.items():
            path = out_dir / f"u{ckpt['update']:05d}-{part}.npz"
            files[path.name] = write_new(path, npz_bytes(
                **arrays, update=ckpt["update"], checkpoint_sha256=ckpt["sha256"],
                wall_s=time.time() - t0))
        print(f"{a.run} update {ckpt['update']}: solve "
              f"{res['fresh']['success_rate_tgt'][res['fresh']['j_star'] > 0].mean():.1%}, "
              f"{time.time() - t0:.0f} s", flush=True)
    write_new(out_dir / "audit-run.json", (json.dumps(
        {"run": a.run, "arm": entry["arm"], "view": entry["view"], "seed": entry["seed"],
         "tag": gate, "files": files, "written": now()}, indent=1) + "\n").encode())


def audit_timing(a, out_dir: Path) -> None:
    """Section 9's timing-only audit of one calibration checkpoint: the whole audit, timed,
    and no score written."""
    src = RUNS / a.source
    record = json.loads((src / "e5-run.json").read_text())
    ckpt = next(c for c in record["checkpoints"] if c["update"] == a.update)
    t0, c0 = time.time(), time.process_time()
    ck, policy, cfg = load_any(record["arm"], ckpt, src, record["seed"], a.source)
    levels, oracles = audit_levels([record["arm"]], [cfg])
    t1 = time.time()
    res = audit_one(record["arm"], ck, policy, cfg, 0, a.update, levels, oracles)
    t2 = time.time()
    steps = {part: int(r["stream_steps"].max()) for part, r in res.items()}
    if "inbuf_stream_steps" in res["fresh"]:
        steps["inbuf"] = int(res["fresh"]["inbuf_stream_steps"].max())
    out_dir.mkdir(parents=True, exist_ok=True)
    write_new(out_dir / "audit-timing.json", (json.dumps(
        {"source": a.source, "arm": record["arm"], "view": record["view"], "update": a.update,
         "setup_s": t1 - t0, "audit_s": t2 - t1, "cpu_s": time.process_time() - c0,
         "stream_steps": steps, "levels": {k: int(v.wall_map.shape[0]) for k, v in levels.items()},
         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "written": now()},
        indent=1) + "\n").encode())
    print(f"timing-only audit of {a.source} update {a.update}: {t2 - t1:.0f} s "
          f"(+{t1 - t0:.0f} s setup), streams {steps}; no scores written", flush=True)


# ------------------------------------------------------------------------- timing


def stage_timing(a) -> None:
    """Section 9's projection rule from the calibration rounds."""
    from apexzero.maze import e5

    rates, failed = {}, {}
    for round_ in a.rounds:
        lanes = int(round_.removeprefix("maze-e5t"))
        ms = shard_manifests(round_)
        if not ms:
            raise SystemExit(f"no manifest for round {round_} (pull-round first)")
        gpus = ms[0]["gpus"]
        if ms[0]["lanes_per_gpu"] != lanes:
            raise SystemExit(f"{round_} ran {ms[0]['lanes_per_gpu']} lanes per GPU, not {lanes}")
        status = {r["name"]: r["status"] for m in ms for r in m["runs"]}
        per_cell: dict = {}
        for run in runs_list("timing", "--lanes-per-gpu", str(lanes), "--gpus", str(gpus),
                             "--arms", a.arms):
            cell = f"{run['arm']}-v{run['view']}"
            path = RUNS / run["name"] / "e5-run.json"
            if status.get(run["name"]) not in ("done", "retried") or not path.is_file():
                failed.setdefault(lanes, []).append(run["name"])
                continue
            rec = json.loads(path.read_text())
            per_cell.setdefault(cell, []).append(
                e5.block_rates(rec["start_unix"], rec["block_ends_unix"], EVAL_FREQ))
        rates[lanes] = {c: {k: float(sorted(x[k] for x in v)[len(v) // 2]) for k in
                            ("per_update_s", "startup_s")} | {"runs": len(v)}
                        for c, v in per_cell.items()}
    audit_s = {}
    if a.audit_round:
        for p in sorted(RUNS.glob(f"{a.audit_round}-*/audit-timing.json")):
            t = json.loads(p.read_text())
            audit_s[f"{t['arm']}-v{t['view']}"] = t["audit_s"]
    capture = capture_check(a.capture_round) if a.capture_round else None
    main = runs_list("main")
    cells = [f"{r['arm']}-v{r['view']}" for r in main]
    shards = [[f"{r['arm']}-v{r['view']}" for r in main if r["seed"] == s]
              for s in dict.fromkeys(r["seed"] for r in main)]
    proj = {}
    for lanes, rt in sorted(rates.items()):
        if failed.get(lanes):
            proj[lanes] = {"error": f"runs failed at {lanes} lanes per GPU: {failed[lanes]}"}
            continue
        x4 = e5.project_queue(cells, rt, 30000, 4, lanes, audit_s or None,
                              overhead_h=a.overhead_h)
        singles = [e5.project_queue(c, rt, 30000, 1, lanes, audit_s or None,
                                    overhead_h=a.overhead_h) for c in shards]
        proj[lanes] = {"x4": x4, "single_gpu_boxes": {
            "boxes": len(singles),
            "total_gpu_h": sum(s.get("total_gpu_h", float("nan")) for s in singles),
            "longest_box_wall_h": max(s.get("train_wall_h", float("nan")) for s in singles)}}
    ok = {k: v for k, v in proj.items() if "x4" in v and "error" not in v["x4"]}
    choice = min(ok, key=lambda k: ok[k]["x4"]["total_gpu_h"]) if ok else None
    result = {"rates": rates, "audit_s": audit_s, "capture_check": capture, "projection": proj,
              "lanes_per_gpu": choice, "overhead_h_per_box": a.overhead_h,
              "rule": "maze-e5.md Section 9: the lane count with the shortest projected total"}
    path = RUNS / "maze-e5-timing.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    for lanes, v in proj.items():
        if "error" in v or "error" in v["x4"]:
            print(f"{lanes} lanes per GPU: {v.get('error') or v['x4']['error']}")
            continue
        x = v["x4"]
        print(f"{lanes} lanes per GPU, one x4 box: train {x['train_wall_h']:.2f} h wall = "
              f"{x['train_gpu_h']:.1f} GPU-h, audit {x['audit_gpu_h'] or 0:.1f}, overhead "
              f"{x['overhead_gpu_h']:.1f}: total {x['total_gpu_h']:.1f} GPU-h (stop rule "
              f"{x['stop_rule_gpu_h']:.1f}); {v['single_gpu_boxes']['boxes']} single-GPU boxes: "
              f"{v['single_gpu_boxes']['total_gpu_h']:.1f} GPU-h")
    print(f"chosen: {choice} lanes per GPU; wrote {rel(path)}"
          + (f"; capture check {capture['verdict']}" if capture else ""))


def capture_check(round_: str) -> dict:
    """The GPU check of DEV-E5-2: two captured runs (determinism) and one without (the
    capture leaves the parameters bit-identical)."""
    p = {tag: RUNS / f"{round_}-sfl-v5-{tag}" / sfl.CKPT_DIR / "u00200" / "params.msgpack"
         for tag in ("on-a", "on-b", "off")}
    if not all(x.is_file() for x in p.values()):
        return {"verdict": "missing", "files": {k: v.is_file() for k, v in p.items()}}
    raw = {k: v.read_bytes() for k, v in p.items()}
    det, eq = raw["on-a"] == raw["on-b"], raw["on-a"] == raw["off"]
    verdict = ("bit-identical" if det and eq else "differs" if det else
               "inconclusive (two captured runs differ: the GPU run is not deterministic)")
    return {"verdict": verdict, "deterministic": det, "identical_without_capture": eq}


# ------------------------------------------------------------------------- the analysis


def load_audits(manifest: dict, audit_names: dict) -> list:
    """The audited runs as apexzero.maze.e5.Run, every file checked against audit-run.json."""
    import io

    import numpy as np

    from apexzero.maze import e5

    runs = []
    for name, entry in manifest["runs"].items():
        if entry["status"] != "trained":
            continue
        d = RUNS / audit_names[name]
        rec = json.loads((d / "audit-run.json").read_text())
        cks = []
        for ckpt in entry["checkpoints"]:
            path = d / f"u{ckpt['update']:05d}-fresh.npz"
            raw = path.read_bytes()
            if sha256_bytes(raw) != rec["files"].get(path.name):
                raise SystemExit(f"{rel(path)} does not match audit-run.json")
            z = dict(np.load(io.BytesIO(raw)))
            if str(z["checkpoint_sha256"]) != ckpt["sha256"]:
                raise SystemExit(f"{rel(path)} audits another checkpoint")
            scores = {s: z[f"s_{s}"] for s in e5.SCORES if f"s_{s}" in z}
            scores.update({k[2:]: z[k] for k in z if k.startswith(("s_pvl_lam", "s_l1_lam"))})
            inbuf = None
            if "inbuf_stored" in z:
                inbuf = {"stored": z["inbuf_stored"], "regret": z["inbuf_regret"],
                         "regret_norm": z["inbuf_regret_norm"],
                         "reachable": z["inbuf_reachable"].astype(bool),
                         "age": z["inbuf_age"]}
            cks.append(e5.Checkpoint(
                ckpt["update"], scores, z["regret"], z["regret_norm"], z["j_star"] > 0,
                z["success_rate_tgt"], inbuf,
                {k: z[k] for k in ("j_star", "j_pi", "j_pi_odd", "j_pi_even")}))
        runs.append(e5.Run(name, entry["arm"], entry["view"], entry["seed"], cks))
    return runs


def decisions_md(res: dict) -> str:
    from apexzero.maze import e5

    def f(x, d=3):
        ok = isinstance(x, int | float) and math.isfinite(x)
        return f"{x:+.{d}f}" if ok else "n/a"

    def pval(x):
        if not (isinstance(x, int | float) and math.isfinite(x)):
            return "n/a"
        return "<0.001" if x < 0.001 else f"{x:.3f}"

    L = ["# Maze E5: decisions (generated by scripts/maze_e5.py analyse)", ""]
    for fam, title in (("primary", "Primary family (Holm over 36)"),
                       ("secondary", "Secondary family (Holm over 28)")):
        L += [f"## {title}", "", ("| test | arm | score | target | phase | estimate | 95% CI "
                                  "| p | p (Holm) | significant |"),
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in res[fam]:
            L.append(f"| {r['test']} | {r['arm']} | {r.get('score', 'maxmc')} | "
                     f"{r.get('target', 'abs-norm')} | {r.get('phase') or '-'} | "
                     f"{f(r.get('estimate'))} | [{f(r.get('lo'))}, {f(r.get('hi'))}] | "
                     f"{pval(r.get('p'))} | {pval(r.get('p_holm'))} | "
                     f"{'yes' if r.get('significant') else 'no'} |")
        L.append("")
    L += ["## Labels (reached phases)", "",
          "| arm | score | target | phase | ρ̄ | nRC | Δ [CI] | label |",
          "|---|---|---|---|---|---|---|---|"]
    for arm, a in res["arms"].items():
        for (s, t, ph), lab in a["labels"].items():
            c = a["cells"][(s, t, ph)]
            L.append(f"| {arm} | {s} | {t} | {ph} | {f(c['rho'], 2)} | {f(c['nrc'], 2)} | "
                     f"{f(c['delta'], 2)} [{f(c['delta_lo'], 2)}, {f(c['delta_hi'], 2)}] | "
                     f"{lab['label']}{' (nRC undefined)' if lab['nrc_flagged'] else ''}"
                     f"{' (Holm)' if lab['holm_delta'] else ''} |")
    L += ["", "## Outcome rows", "", "| arm | score | rows | reading |", "|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {e5.ARM_LABEL[r['arm']]} | {e5.LABEL[r['score']]} | {', '.join(r['rows'])} "
                 f"| {'; '.join(r['reading'])} |")
    L += ["", "S1 (the plan's MaxMC prediction) holds for: "
          + (", ".join(a for a, v in res["s1_holds"].items() if v) or "no arm"), ""]
    L += [f"Note: {n}" for n in res["notes"]] + [""]
    return "\n".join(L)


def stage_analyse(a) -> None:
    lower_priority()
    sfl.check_vendored()
    gate = tag_gate()
    out = OUT / "analyse"
    lock = out / "analysis.lock"
    targets = [out / "analysis.json", out / "decisions.md"]
    if lock.exists() or any(p.exists() for p in targets):
        raise SystemExit(f"{rel(out)} holds a lock or an output: analyse runs once")
    manifest = json.loads(tagged_file(MANIFEST, gate["commit"]))
    drop = ["--drop", str(manifest["drop"])]
    round_check("maze-e5a", TAG, ["--phase", "audit", *drop], allow_failed=False)
    audit_names = {r["source"]: r["name"] for r in runs_list("audit", *drop)}
    runs = load_audits(manifest, audit_names)
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f"{now()} {gate['head']}\n".encode())
    os.close(fd)
    from apexzero.maze import e5

    t0 = time.time()
    pt = e5.point(runs)
    boot = e5.bootstrap(runs, a.n_boot, e5.BOOT_SEED, min(a.workers, MAX_WORKERS))
    res = e5.analyse(runs, boot, pt)
    res["describe"] = e5.describe(runs, pt)
    res["provenance"] = {"tag": gate, "manifest_sha256": sha256_bytes(MANIFEST.read_bytes()),
                         "n_boot": a.n_boot, "boot_seed": e5.BOOT_SEED, "runs": len(runs),
                         "missing": manifest["missing"], "seconds": round(time.time() - t0, 1)}
    write_json_new(targets[0], res)
    write_new(targets[1], decisions_md(res).encode())
    print(f"wrote {rel(targets[0])} and {rel(targets[1])} ({time.time() - t0:.0f} s)")


# ------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("train", help="one run (the run list's cmd)")
    p.add_argument("--arm", choices=["dr", "plr", "sfl"], required=True)
    p.add_argument("--view", type=int, choices=[5, 13], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--updates", type=int, default=30000)
    p.add_argument("--save", default="", help="updates to keep (default: the E5 checkpoints)")
    p.add_argument("--no-capture", action="store_true", help="SFL: do not capture D")
    p.add_argument("--name", required=True)
    p = sub.add_parser("timing", help="the calibration's projection (Mac)")
    p.add_argument("--rounds", nargs="+", required=True, help="e.g. maze-e5t4 maze-e5t8")
    p.add_argument("--audit-round", default=None)
    p.add_argument("--capture-round", default=None)
    p.add_argument("--arms", default="sfl,plr,dr")
    p.add_argument("--overhead-h", type=float, default=0.25,
                   help="setup and pull-back per box, wall hours")
    p = sub.add_parser("manifest", help="check the pulled-back runs, write the manifest (Mac)")
    p.add_argument("--train-tag", default=TRAIN_TAG)
    p.add_argument("--drop", type=int, default=0)
    p = sub.add_parser("audit", help="audit one run's checkpoints (box, behind the tag)")
    p.add_argument("--run", help="the training run")
    p.add_argument("--timing-only", action="store_true")
    p.add_argument("--source", help="--timing-only: the calibration run")
    p.add_argument("--update", type=int, default=400, help="--timing-only: its checkpoint")
    p.add_argument("--name", required=True)
    p = sub.add_parser("analyse", help="Sections 4-5, once (Mac, behind the tag)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=4000)
    a = ap.parse_args(argv)
    if a.stage == "train":
        return stage_train(a)
    if a.stage == "audit" and not (a.run or (a.timing_only and a.source)):
        raise SystemExit("audit needs --run, or --timing-only with --source")
    {"timing": stage_timing, "manifest": stage_manifest, "audit": stage_audit,
     "analyse": stage_analyse}[a.stage](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
