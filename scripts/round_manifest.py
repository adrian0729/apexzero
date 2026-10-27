"""Run lists, schedules and manifests for Phase 3's GPU rounds (with scripts/gpu_round.sh).

    uv run python scripts/round_manifest.py check --round gw3 --runs-script scripts/gw3_runs.py \\
        --tag gw3-v1 --runs-args --phase main

**The run-list contract.** Each round has a run-list script, scripts/<round>_runs.py, that uses
only the standard library and pure-Python repo imports (never JAX). Its ``runs(**opts)`` returns
the round's runs in the pre-registered order, each a dict with

- ``name``: the run directory under runs/ (letters, digits and ``. _ + -``; unique). Names
  should start with ``<round>-``: ``check`` looks for stray directories with that prefix.
- ``seed``: an int. ``SHARD=i/K`` keeps the runs whose seed's index among the round's distinct
  seeds, in first-appearance order, is i mod K, so each box takes whole seeds.
- ``cmd``: the trainer's path relative to the repo and its full arguments, ``--name <name>``
  included once, e.g. ``["scripts/train_dr.py", "--steps", "200e6", "--seed", "20", "--name",
  "gw3-v1-base-s20"]``. The queue runs it as ``uv run --no-sync python <cmd...>``.
- optionally ``config``: the JSON file, inside runs/<name>/, where the run records its
  arguments (``runs/<name>/config.json`` for train_dr.py and train.py).

Other keys are kept and ignored. Run as a script it prints one JSON object per line, in order,
and takes round-specific options (e.g. ``--phase pilot|main``). Every subcommand runs it from
the repo root with this interpreter and checks every entry before anything else: the fields
above, unique names, the trainer in the repo, and no argument naming a lockbox (R_TEST)
circuit, by ``viz.export.is_lockbox``'s rule.

**Subcommands.** Each takes ``--round R --runs-script S``; ``--runs-args``, if given, comes
last, and everything after it goes to the runs script.

- ``schedule --shard i/K --gpus G --lanes-per-gpu N`` (gpu_round.sh's): job j of the shard's
  list goes to lane j mod L (L = G x N), lane l to GPU l mod G, at position j // L in its lane.
  ``--format tsv`` prints lane, gpu, position and name (DRY_RUN); ``--format nul`` the same
  fields, the argument count and the cmd, NUL-terminated, for the queue. ``--fresh`` refuses if
  any of the shard's run directories or logs, or its status file or manifest, exists.
- ``names [--shard i/K]``: the shard's run names (default the whole round), for gpu_sync.sh's
  pull-round.
- ``write --shard i/K [--gpus G --lanes-per-gpu N]`` (on the box, at the end of the queue):
  writes runs/<round>-manifest-<i>of<K>.json from the queue's status file
  (runs/<round>-status-<i>of<K>.tsv): HEAD, whether the tracked tree is clean, the tags at
  HEAD, the GPUs nvidia-smi lists (model and count), the box type (env ``BREV_TYPE``, else
  "unknown"), the GPUs and lanes per GPU used, the shard, the run list's sha256 and, per run,
  its status (done, retried, failed; "unfinished" or "not run" if the queue stopped early),
  start and end times (UTC), lane, GPU and its config file's sha256.
- ``check --tag T`` (on the Mac, after pull-round): exits 1 unless the shard manifests in runs/
  cover every run exactly once with status done or retried; every manifest's HEAD is T's
  commit, its tree was clean and its run list is this one (the runs script unchanged since T);
  the boxes all list one GPU model; each run's directory and config file are here, the config
  is the box's (sha256) and its recorded arguments equal its cmd's; and no directory in runs/
  with the prefix ``<round>-`` is outside the run list. The comparison is per trainer, with
  the trainer's argument parser rebuilt from its source at T: train_dr.py records every
  argument (``args``), train.py its tracks, seed and PPO settings; any other trainer's config
  format is unknown and reported as a problem, not passed.
"""

import argparse
import ast
import hashlib
import json
import os
import posixpath
import re
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePath, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("runs")  # every path is relative to ROOT (main changes directory there)
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
ROUND = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
OK = ("done", "retried")
EVENTS = ("start", "retry", "done", "retried", "failed")
TIME = "%Y-%m-%dT%H:%M:%SZ"


# --------------------------------------------------------------------------- run lists


def _run(*cmd: str) -> subprocess.CompletedProcess:
    """``cmd`` from the repo root, its output captured; the caller reads the return code."""
    return subprocess.run(cmd, check=False, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL)


def parse_shard(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(0|[1-9][0-9]*)/([1-9][0-9]*)", text)
    if not m or int(m[1]) >= int(m[2]):
        raise SystemExit(f"shard must be i/K with 0 <= i < K: {text!r}")
    return int(m[1]), int(m[2])


def shard_file(round_: str, kind: str, i: int, k: int, suffix: str) -> Path:
    """runs/<round>-<kind>-<i>of<K><suffix>: the queue log, status file and manifest."""
    return RUNS / f"{round_}-{kind}-{i}of{k}{suffix}"


def _squash(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", name.casefold())


def lockbox_prefixes() -> tuple[str, ...]:
    """R_TEST's names, squashed as ``viz.export.is_lockbox`` squashes them, read from
    tracks/corners.py without importing it (that needs numpy and scipy; this needs the stdlib)."""
    path = Path("src/apexzero/tracks/corners.py")
    if not path.is_file():
        raise SystemExit(f"{path} not found: cannot check the run list against R_TEST")
    for node in ast.parse(path.read_text()).body:
        targets = [getattr(t, "id", None) for t in getattr(node, "targets", [])]
        if isinstance(node, ast.Assign) and targets == ["R_TEST"]:
            return tuple(_squash(n) for n in ast.literal_eval(node.value))
    raise SystemExit(f"R_TEST not found in {path}")


def names_lockbox(piece: str, prefixes: tuple[str, ...]) -> bool:
    """``viz.export.is_lockbox``'s rule: directory and suffix dropped, case and separators
    ignored, any name that starts with an R_TEST circuit."""
    stem = re.sub(r"\.(csv|pkl|json)$", "", PurePath(piece).name, flags=re.IGNORECASE)
    return bool(prefixes) and _squash(stem).startswith(prefixes)


def option_values(cmd: list[str], flag: str) -> list[str | None]:
    """Every value ``flag`` takes in ``cmd`` (``--flag value`` or ``--flag=value``)."""
    out = []
    for k, a in enumerate(cmd):
        if a == flag:
            out.append(cmd[k + 1] if k + 1 < len(cmd) else None)
        elif a.startswith(flag + "="):
            out.append(a[len(flag) + 1:])
    return out


def _repo_file(path: str) -> bool:
    p = PurePosixPath(path)
    return not p.is_absolute() and ".." not in p.parts and Path(path).is_file()


def run_list_problems(runs: list) -> list[str]:
    """Every way the run list breaks the contract (empty if none)."""
    if not runs:
        return ["the run list is empty"]
    problems, seen = [], set()
    prefixes = lockbox_prefixes()
    for k, run in enumerate(runs):
        name = run.get("name") if isinstance(run, dict) else None
        if not isinstance(name, str) or not NAME.fullmatch(name):
            problems.append(f"run {k}: name {name!r} is not a plain directory name")
            continue
        if name in seen:
            problems.append(f"{name}: listed twice")
        seen.add(name)
        seed = run.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            problems.append(f"{name}: seed {seed!r} is not an int")
        cmd = run.get("cmd")
        if not (isinstance(cmd, list) and cmd and all(isinstance(a, str) for a in cmd)):
            problems.append(f"{name}: cmd is not a non-empty list of strings")
            continue
        if any("\0" in a for a in cmd):
            problems.append(f"{name}: cmd contains a NUL character")
        if not _repo_file(cmd[0]):
            problems.append(f"{name}: cmd[0] {cmd[0]!r} is not a script in the repo")
        if option_values(cmd, "--name") != [name]:
            problems.append(f"{name}: cmd must pass --name {name} exactly once "
                            f"(passes {option_values(cmd, '--name')})")
        for a in cmd[1:]:
            if a.startswith("-") and "=" not in a:
                continue
            value = a.split("=", 1)[1] if a.startswith("-") else a
            for piece in value.split(","):
                if names_lockbox(piece, prefixes):
                    problems.append(f"{name}: argument {a!r} names a lockbox (R_TEST) circuit")
                if "outputs/sealed" in piece:
                    problems.append(f"{name}: argument {a!r} points into outputs/sealed/")
        config = run.get("config")
        if config is not None:
            parts = PurePosixPath(config).parts if isinstance(config, str) else ()
            if len(parts) < 3 or parts[:2] != ("runs", name) or ".." in parts:
                problems.append(f"{name}: config {config!r} is not a file inside runs/{name}/")
    return problems


def load_runs(round_: str, runs_script: str, runs_args: list[str]) -> list[dict]:
    """The round's run list, in order, from its runs script; every entry checked."""
    if not ROUND.fullmatch(round_):
        raise SystemExit(f"round name {round_!r}: letters, digits and . _ - only")
    if not _repo_file(runs_script):
        raise SystemExit(f"runs script {runs_script!r} is not a file in the repo")
    proc = _run(sys.executable, runs_script, *runs_args)
    if proc.returncode:
        raise SystemExit(f"{runs_script} exited {proc.returncode}:\n{proc.stderr.strip()}")
    runs = []
    for k, line in enumerate(proc.stdout.splitlines(), 1):
        if line.strip():
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{runs_script}, line {k} of its output: not JSON ({e})") from None
    problems = run_list_problems(runs)
    if problems:
        raise SystemExit(f"{runs_script}: {len(problems)} problem(s) in the run list:\n  "
                         + "\n  ".join(problems))
    return runs


def seed_order(runs: list[dict]) -> list[int]:
    """The round's distinct seeds in first-appearance order."""
    return list(dict.fromkeys(r["seed"] for r in runs))


def in_shard(runs: list[dict], i: int, k: int) -> list[dict]:
    """Shard i of K: the runs whose seed's index (first-appearance order) is i mod K."""
    index = {s: n for n, s in enumerate(seed_order(runs))}
    return [r for r in runs if index[r["seed"]] % k == i]


def schedule(jobs: list[dict], gpus: int, lanes_per_gpu: int) -> list[tuple[int, int, int, dict]]:
    """(lane, gpu, position, run) per job: job j to lane j mod L, lane l to GPU l mod gpus."""
    lanes = gpus * lanes_per_gpu
    return [(j % lanes, (j % lanes) % gpus, j // lanes, run) for j, run in enumerate(jobs)]


def run_list_sha256(runs: list[dict]) -> str:
    text = json.dumps(runs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- the box


def git(*args: str) -> str:
    proc = _run("git", *args)
    if proc.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def gpu_names() -> list[str] | None:
    """The GPUs nvidia-smi lists (None without nvidia-smi)."""
    try:
        proc = _run("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    except FileNotFoundError:
        return None
    if proc.returncode:
        print(f"warning: nvidia-smi failed: {proc.stderr.strip()}", file=sys.stderr)
        return None
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def read_status(path: Path) -> dict[str, list[dict]]:
    """The queue's events per run, in order: name, event, UTC time, lane, GPU per line."""
    events = defaultdict(list)
    if not path.exists():
        return events
    for k, line in enumerate(path.read_text().splitlines(), 1):
        f = line.split("\t")
        if len(f) != 5 or f[1] not in EVENTS:
            raise SystemExit(f"{path}, line {k}: not a status line: {line!r}")
        events[f[0]].append({"event": f[1], "time": f[2], "lane": int(f[3]), "gpu": int(f[4])})
    return events


def run_record(run: dict, events: list[dict]) -> dict:
    status = events[-1]["event"] if events else "not run"
    if status in ("start", "retry"):
        status = "unfinished"
    config = run.get("config")
    return {"name": run["name"], "seed": run["seed"], "status": status,
            "start": events[0]["time"] if events else None,
            "end": events[-1]["time"] if status in ("done", "retried", "failed") else None,
            "lane": events[0]["lane"] if events else None,
            "gpu": events[0]["gpu"] if events else None,
            "config": config,
            "config_sha256": sha256_file(config) if config and Path(config).is_file() else None}


def _env_int(name: str) -> int | None:
    value = os.environ.get(name, "")
    return int(value) if value.isdigit() else None


# --------------------------------------------------------------------------- configs


class _Computed:
    """A default the trainer computes (not a literal): compared only where a cmd sets it."""

    def __repr__(self) -> str:
        return "<computed default>"


COMPUTED = _Computed()
_MISSING = object()
_TYPES = {"float": float, "int": int, "str": str}


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def trainer_parser(source: str) -> _Parser:
    """The trainer's argument parser, rebuilt from its ``add_argument`` calls (in source order)
    with literal arguments; a default that is not a literal becomes ``COMPUTED``."""
    parser = _Parser(add_help=False)
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"]
    for call in sorted(calls, key=lambda n: (n.lineno, n.col_offset)):
        flags = [ast.literal_eval(a) for a in call.args]
        kw = {}
        for k in call.keywords:
            if k.arg == "help":
                continue
            if k.arg == "type":
                if not (isinstance(k.value, ast.Name) and k.value.id in _TYPES):
                    raise ValueError(f"{flags[0]}: type is not float, int or str")
                kw["type"] = _TYPES[k.value.id]
                continue
            try:
                kw[k.arg] = ast.literal_eval(k.value)
            except ValueError:
                if k.arg != "default":
                    raise ValueError(f"{flags[0]}: {k.arg} is not a literal") from None
                kw["default"] = COMPUTED
        parser.add_argument(*flags, **kw)
    return parser


def _flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def _same(recorded, value) -> bool:
    value = json.loads(json.dumps(value))
    return isinstance(recorded, bool) == isinstance(value, bool) and recorded == value


def compare_train_dr(config: dict, args: dict) -> tuple[list[str], list[str]]:
    """train_dr.py records ``vars(args)`` under "args": every argument is compared."""
    recorded = config.get("args")
    if not isinstance(recorded, dict):
        return ['config has no "args" object'], []
    problems, notes = [], []
    for key in sorted(set(recorded) | set(args)):
        if key not in args:
            problems.append(f"config records {_flag(key)}, which the trainer does not define")
        elif key not in recorded:
            problems.append(f"config does not record {_flag(key)}")
        elif args[key] is COMPUTED:
            notes.append(f"{_flag(key)} has a computed default: compared only where a cmd sets it")
        elif not _same(recorded[key], args[key]):
            problems.append(f"{_flag(key)} is {recorded[key]!r} in the config, "
                            f"{args[key]!r} by the cmd")
    return problems, notes


# train.py's config: tracks and seed at the top level, the rest through PPOConfig.
TRAIN_FIELDS = {"tracks": ("tracks",), "seed": ("seed",), "steps": ("ppo", "total_timesteps"),
                "num_envs": ("ppo", "num_envs"), "num_steps": ("ppo", "num_steps"),
                "lr": ("ppo", "lr"), "network": ("ppo", "network")}


def compare_train(config: dict, args: dict) -> tuple[list[str], list[str]]:
    """train.py records its tracks and seed, and its steps (as int total_timesteps), envs,
    rollout length, learning rate and network in PPOConfig. --name is the run directory, and
    anything else (--log-every) is not recorded."""
    problems, notes = [], []
    for key, value in args.items():
        if key == "name":
            continue
        if key not in TRAIN_FIELDS:
            notes.append(f"{_flag(key)} is not recorded in train.py's config: not compared")
            continue
        if value is COMPUTED:
            notes.append(f"{_flag(key)} has a computed default: compared only where a cmd sets it")
            continue
        recorded = config
        for part in TRAIN_FIELDS[key]:
            recorded = recorded.get(part, _MISSING) if isinstance(recorded, dict) else _MISSING
        want = int(value) if key == "steps" else value
        if recorded is _MISSING:
            problems.append(f"config does not record {_flag(key)} ({'.'.join(TRAIN_FIELDS[key])})")
        elif not _same(recorded, want):
            problems.append(f"{_flag(key)} is {recorded!r} in the config, {want!r} by the cmd")
    return problems, notes


TRAINERS = {"scripts/train_dr.py": compare_train_dr, "scripts/train.py": compare_train}


def compare_config(run: dict, config: dict, parser: _Parser) -> tuple[list[str], list[str]]:
    """The run's recorded arguments against its cmd, parsed by its trainer's parser."""
    try:
        args = vars(parser.parse_args(run["cmd"][1:]))
    except ValueError as e:
        return [f"cmd does not parse with the trainer: {e}"], []
    problems = []
    if "seed" in args and args["seed"] != run["seed"]:
        problems.append(f"the run list's seed is {run['seed']}, the cmd's --seed {args['seed']}")
    if "name" in args and args["name"] != run["name"]:
        problems.append(f"the cmd's --name is {args['name']!r}")
    found, notes = TRAINERS[posixpath.normpath(run["cmd"][0])](config, args)
    return problems + found, notes


# --------------------------------------------------------------------------- subcommands


def cmd_schedule(a):
    runs = load_runs(a.round, a.runs_script, a.runs_args)
    i, k = parse_shard(a.shard)
    if a.gpus < 1 or a.lanes_per_gpu < 1:
        raise SystemExit("--gpus and --lanes-per-gpu must be positive")
    jobs = in_shard(runs, i, k)
    if not jobs:
        raise SystemExit(f"shard {i}/{k} has no runs: the round has {len(seed_order(runs))} seeds")
    leftovers = [p for r in jobs for p in (RUNS / r["name"], RUNS / f"{r['name']}.log",
                                           RUNS / f"{r['name']}.log.1")]
    leftovers += [shard_file(a.round, "status", i, k, ".tsv"),
                  shard_file(a.round, "manifest", i, k, ".json")]
    existing = [str(p) for p in leftovers if p.exists()]
    if existing:
        if a.fresh:
            raise SystemExit("refusing to start: these exist already (move them away first): "
                             + ", ".join(existing))
        print("warning: these exist already: " + ", ".join(existing), file=sys.stderr)
    plan = schedule(jobs, a.gpus, a.lanes_per_gpu)
    if a.format == "tsv":
        print("lane\tgpu\tposition\tname")
        for lane, gpu, pos, run in plan:
            print(f"{lane}\t{gpu}\t{pos}\t{run['name']}")
        shard_seeds = seed_order(jobs)
        print(f"{a.round} shard {i}/{k}: {len(jobs)} of {len(runs)} runs ({len(shard_seeds)} of "
              f"{len(seed_order(runs))} seeds) on {a.gpus} GPU(s) x {a.lanes_per_gpu} lanes",
              file=sys.stderr)
    else:
        out = sys.stdout.buffer
        for lane, gpu, pos, run in plan:
            fields = [str(lane), str(gpu), str(pos), run["name"], str(len(run["cmd"])), *run["cmd"]]
            out.write(b"".join(f.encode() + b"\0" for f in fields))
        out.flush()


def cmd_names(a):
    runs = load_runs(a.round, a.runs_script, a.runs_args)
    i, k = parse_shard(a.shard)
    for run in in_shard(runs, i, k):
        print(run["name"])


def cmd_write(a):
    runs = load_runs(a.round, a.runs_script, a.runs_args)
    i, k = parse_shard(a.shard)
    jobs = in_shard(runs, i, k)
    events = read_status(shard_file(a.round, "status", i, k, ".tsv"))
    records = [run_record(run, events.get(run["name"], [])) for run in jobs]
    unlisted = sorted(set(events) - {r["name"] for r in jobs})
    names = gpu_names()
    model = None if not names else names[0] if len(set(names)) == 1 else "mixed"
    manifest = {
        "round": a.round, "shard": f"{i}/{k}", "runs_script": a.runs_script,
        "runs_args": a.runs_args, "run_list_sha256": run_list_sha256(runs),
        "head": git("rev-parse", "HEAD").strip(),
        "tree_clean": git("status", "--porcelain", "--untracked-files=no") == "",
        "tags": git("tag", "--points-at", "HEAD").split(),
        "gpu_model": model, "gpu_count": None if names is None else len(names), "gpu_names": names,
        "box_type": os.environ.get("BREV_TYPE") or "unknown",
        "gpus": _env_int("GPUS") if a.gpus is None else a.gpus,
        "lanes_per_gpu": _env_int("LANES_PER_GPU") if a.lanes_per_gpu is None else a.lanes_per_gpu,
        "xla_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "written": datetime.now(UTC).strftime(TIME),
        "status_unlisted": unlisted,
        "runs": records,
    }
    path = shard_file(a.round, "manifest", i, k, ".json")
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    counts = {s: sum(r["status"] == s for r in records) for s in dict.fromkeys(
        r["status"] for r in records)}
    print(f"wrote {path}: {len(records)} runs, "
          + ", ".join(f"{n} {s}" for s, n in counts.items()))


def _hours(records: list[dict]) -> float | None:
    times = [(r.get("start"), r.get("end")) for r in records]
    try:
        starts = [datetime.strptime(s, TIME) for s, _ in times if s]
        ends = [datetime.strptime(e, TIME) for _, e in times if e]
    except (TypeError, ValueError):
        return None
    if not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds() / 3600


def cmd_check(a):
    runs = load_runs(a.round, a.runs_script, a.runs_args)
    by_name = {r["name"]: r for r in runs}
    script = posixpath.normpath(a.runs_script)
    problems, notes = [], {}
    proc = _run("git", "rev-parse", "--verify", "--quiet", f"{a.tag}^{{commit}}")
    if proc.returncode:
        raise SystemExit(f"tag {a.tag!r} not found")
    commit = proc.stdout.strip()
    if _run("git", "cat-file", "-e", f"{commit}:{script}").returncode:
        problems.append(f"{script} is not in {a.tag}")
    elif _run("git", "diff", "--quiet", commit, "--", script).returncode:
        problems.append(f"{script} differs from its version at {a.tag}")
    expected_sha = run_list_sha256(runs)

    print(f"round {a.round}: {script} {' '.join(a.runs_args)}".rstrip()
          + f": {len(runs)} runs, {len(seed_order(runs))} seeds")
    print(f"tag {a.tag}: {commit[:12]}")
    pattern = re.compile(rf"{re.escape(a.round)}-manifest-(\d+)of(\d+)\.json")
    paths = sorted(p for p in RUNS.glob(f"{a.round}-manifest-*of*.json")
                   if pattern.fullmatch(p.name))
    if not paths:
        problems.append(f"no manifest runs/{a.round}-manifest-<i>of<K>.json here (pull-round?)")
    entries = defaultdict(list)
    models, shard_counts = set(), set()
    for path in paths:
        try:
            m = json.loads(path.read_text())
            i, k = parse_shard(m["shard"])
            records = list(m["runs"])
        except (OSError, ValueError, KeyError, TypeError, SystemExit) as e:
            problems.append(f"{path}: unreadable manifest ({e})")
            continue
        fi, fk = pattern.fullmatch(path.name).groups()
        if (i, k) != (int(fi), int(fk)):
            problems.append(f"{path}: records shard {i}/{k}")
        shard_counts.add(k)
        hours, gpus = _hours(records), m.get("gpus")
        billed = m.get("gpu_count") if isinstance(m.get("gpu_count"), int) else gpus
        listed = (f"{m['gpu_count']} x {m.get('gpu_model')}" if m.get("gpu_count")
                  else "no GPU listed")
        clean = "clean" if m.get("tree_clean") is True else "NOT clean"
        print(f"  {path}: shard {i}/{k}, {len(records)} runs, {m.get('box_type')}, {listed} "
              f"({gpus} used, {m.get('lanes_per_gpu')} lanes per GPU), HEAD "
              f"{str(m.get('head'))[:12]}, {clean}"
              + (f"; queue {hours:.2f} h x {billed} GPUs = {hours * billed:.1f} GPU-h (idle "
                 "included; setup and pull-back not)"
                 if hours is not None and isinstance(billed, int) else ""))
        if m.get("round") != a.round:
            problems.append(f"{path}: round {m.get('round')!r}")
        if m.get("head") != commit:
            problems.append(f"{path}: HEAD {str(m.get('head'))[:12]} is not {a.tag} "
                            f"({commit[:12]})")
        if m.get("tree_clean") is not True:
            problems.append(f"{path}: the box's tracked tree was not clean")
        if a.tag not in (m.get("tags") or []):
            notes[f"{path}: {a.tag} was not among the tags at the box's HEAD"] = None
        if (posixpath.normpath(str(m.get("runs_script"))) != script
                or m.get("runs_args") != a.runs_args):
            problems.append(f"{path}: the box ran {m.get('runs_script')} {m.get('runs_args')}")
        if m.get("run_list_sha256") != expected_sha:
            problems.append(f"{path}: the box's run list is not this one (run_list_sha256)")
        if m.get("status_unlisted"):
            problems.append(f"{path}: queue status for runs not in its list: "
                            f"{m['status_unlisted']}")
        model = m.get("gpu_model")
        if not model or model == "mixed":
            problems.append(f"{path}: GPU model {model!r} (nvidia-smi listed {m.get('gpu_names')})")
        models.add(model)
        if isinstance(gpus, int) and isinstance(m.get("gpu_count"), int) and gpus > m["gpu_count"]:
            problems.append(f"{path}: used {gpus} GPUs, nvidia-smi listed {m['gpu_count']}")
        shard_names = {r["name"] for r in in_shard(runs, i, k)}
        for rec in records:
            name = rec.get("name") if isinstance(rec, dict) else None
            if name not in by_name:
                problems.append(f"{path}: {name!r} is not in the run list")
                continue
            if name not in shard_names:
                problems.append(f"{path}: {name} is not in shard {i}/{k}")
            entries[name].append((path, rec))
    if len(models) > 1:
        problems.append(f"the boxes' GPU models differ: {sorted(map(str, models))}")
    if len(shard_counts) > 1:
        problems.append(f"manifests from different shard counts K: {sorted(shard_counts)}")

    parsers, statuses, compared = {}, defaultdict(int), 0
    for run in runs:
        name = run["name"]
        found = entries.get(name, [])
        if not found:
            problems.append(f"{name}: in no manifest")
            continue
        if len(found) > 1:
            where = ", ".join(str(p) for p, _ in found)
            problems.append(f"{name}: in {len(found)} manifests ({where})")
        for path, rec in found:
            if rec.get("status") not in OK:
                problems.append(f"{name}: status {rec.get('status')!r} in {path}")
        statuses[str(found[0][1].get("status"))] += 1
        if not (RUNS / name).is_dir():
            problems.append(f"{name}: runs/{name} is not here")
        if not run.get("config"):
            notes["runs without a config file: arguments not checked"] = None
            continue
        config = Path(run["config"])
        if not config.is_file():
            problems.append(f"{name}: {config} is not here")
            continue
        if found[0][1].get("config_sha256") != sha256_file(config):
            problems.append(f"{name}: {config} is not the box's (sha256)")
        trainer = posixpath.normpath(run["cmd"][0])
        if trainer not in TRAINERS:
            problems.append(f"{name}: {trainer}'s config format is unknown: arguments not checked")
            continue
        if trainer not in parsers:
            show = _run("git", "show", f"{commit}:{trainer}")
            try:
                parsers[trainer] = (trainer_parser(show.stdout) if not show.returncode
                                    else f"{trainer} is not in {a.tag}")
            except (ValueError, SyntaxError) as e:
                parsers[trainer] = f"cannot rebuild {trainer}'s parser at {a.tag}: {e}"
        parser = parsers[trainer]
        if isinstance(parser, str):
            problems.append(f"{name}: {parser}")
            continue
        try:
            recorded = json.loads(config.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{name}: {config} is not JSON ({e})")
            continue
        found_problems, found_notes = compare_config(run, recorded, parser)
        problems += [f"{name}: {p}" for p in found_problems]
        notes.update(dict.fromkeys(f"{trainer}: {n}" for n in found_notes))
        compared += 1
    for d in sorted(RUNS.glob(f"{a.round}-*")):
        if d.is_dir() and d.name not in by_name:
            problems.append(f"runs/{d.name}: has the round's prefix but is not in the run list")
    outside = [n for n in by_name if not n.startswith(f"{a.round}-")]
    if outside:
        notes[f"{len(outside)} run name(s) do not start with {a.round}-: stray directories of "
              "theirs are not looked for"] = None

    print("runs: " + ", ".join(f"{n} {s}" for s, n in sorted(statuses.items()))
          + f"; {compared} configs compared")
    for note in notes:
        print(f"note: {note}")
    if problems:
        print(f"FAILED: {len(problems)} problem(s)")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    print(f"OK: every run of {a.round} is done, once, at {a.tag}, with the arguments of its cmd")


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runs_args = []
    if "--runs-args" in argv:
        k = argv.index("--runs-args")
        argv, runs_args = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    commands = {}
    for name, fn, help_ in [("schedule", cmd_schedule, "the shard's jobs, lane by lane"),
                            ("names", cmd_names, "the shard's run names"),
                            ("write", cmd_write, "write the shard's manifest (on the box)"),
                            ("check", cmd_check, "check the round's manifests (on the Mac)")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--round", required=True)
        p.add_argument("--runs-script", required=True)
        commands[name] = (p, fn)
    p = commands["schedule"][0]
    p.add_argument("--shard", default="0/1")
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--lanes-per-gpu", type=int, required=True)
    p.add_argument("--format", choices=["tsv", "nul"], default="tsv")
    p.add_argument("--fresh", action="store_true",
                   help="refuse if the shard's run directories, logs, status or manifest exist")
    commands["names"][0].add_argument("--shard", default="0/1")
    p = commands["write"][0]
    p.add_argument("--shard", required=True)
    p.add_argument("--gpus", type=int, default=None, help="GPUs used (default: env GPUS)")
    p.add_argument("--lanes-per-gpu", type=int, default=None,
                   help="lanes per GPU (default: env LANES_PER_GPU)")
    commands["check"][0].add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    a.runs_args = runs_args
    os.chdir(ROOT)
    commands[a.command][1](a)


if __name__ == "__main__":
    main()
