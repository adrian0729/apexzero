#!/usr/bin/env python3
"""Live terminal dashboard of everything running in this repo.

    uv run python scripts/dashboard.py            # refresh every 5 s (Ctrl-C to quit)
    uv run python scripts/dashboard.py --once     # print one frame and exit

The live view restarts itself when this file changes, so edits show up without a restart.

Shows:
- the training queue (scripts/queue_overnight.sh): each job's status, progress, speed,
  return, crash rate, checkpoints and ETA, and the queue's estimated finish;
- other experiment processes started from scripts/ (solvers, evaluations, analyses);
- run directories and logs under runs/ updated in the last 15 minutes;
- machine load, memory, power and whether idle sleep is prevented;
- runs/activity.md: free-form notes on work that has no process of its own here
  (agent workflows, analyses in progress), kept up to date by whoever starts them.

Standard library only, so it starts instantly and never touches the training env. The
queue script is only read: editing a running bash script corrupts its execution.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
QUEUE = ROOT / "scripts" / "queue_overnight.sh"
NOTES = RUNS / "activity.md"
RECENT_S = 15 * 60

RESET, BOLD, DIM = "\x1b[0m", "\x1b[1m", "\x1b[2m"
RED, GREEN, YELLOW, BLUE, CYAN = "\x1b[31m", "\x1b[32m", "\x1b[33m", "\x1b[34m", "\x1b[36m"
STATUS_COLOUR = {"running": GREEN, "compiling": GREEN, "evaluating": CYAN, "done": BLUE,
                 "pending": YELLOW, "failed": RED, "stopped": YELLOW}
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


# ---------------------------------------------------------------------------- parsing

def parse_etime(s: str) -> int:
    """ps elapsed time "[[dd-]hh:]mm:ss" -> seconds."""
    days, _, rest = s.strip().rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return (int(days) if days else 0) * 86400 + h * 3600 + m * 60 + sec


@dataclass
class Proc:
    pid: int
    ppid: int
    elapsed: int
    cpu: float
    rss_mb: float
    cmd: str


def processes() -> list[Proc]:
    out = _run(["ps", "-axo", "pid=,ppid=,etime=,pcpu=,rss=,command="])
    procs = []
    for line in out.splitlines():
        f = line.split(None, 5)
        if len(f) == 6:
            try:
                procs.append(Proc(int(f[0]), int(f[1]), parse_etime(f[2]), float(f[3]),
                                  int(f[4]) / 1024, f[5]))
            except ValueError:
                continue
    return procs


@dataclass
class Job:
    name: str
    script: str
    args: str


def parse_queue(text: str) -> list[Job]:
    """The ``run <name> <script> <args>`` lines of a queue script, in order."""
    jobs = []
    for line in text.replace("\\\n", " ").splitlines():
        m = re.match(r"\s*run\s+(\S+)\s+(scripts/\S+\.py)\s*(.*)$", line)
        if m:
            jobs.append(Job(m.group(1), m.group(2), m.group(3)))
    return jobs


def tail_rows(path: Path, n: int = 2) -> list[dict]:
    """The last n rows of a CSV with a header line (reads only the file's end)."""
    try:
        with open(path, "rb") as fh:
            header = fh.readline().decode(errors="ignore").strip()
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 16384))
            lines = fh.read().decode(errors="ignore").splitlines()
    except OSError:
        return []
    if not header:
        return []
    body = [ln for ln in lines[1 if size > 16384 else 0:] if ln and ln != header][-n:]
    rows = []
    for rec in csv.DictReader(io.StringIO(header + "\n" + "\n".join(body))):
        rows.append({k: _num(v) for k, v in rec.items()})
    return rows


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _config(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "config.json").read_text())
    except (OSError, ValueError):
        return {}


def script_kind(cfg: dict) -> str | None:
    """Which training script wrote a run's config.json."""
    if "ued" in cfg:
        return "scripts/train_ued.py"
    if "tracks" in cfg:
        return "scripts/train.py"
    if "bank" in cfg.get("args", {}):
        return "scripts/train_dr.py"
    return None


# ---------------------------------------------------------------------------- state

@dataclass
class JobState:
    job: Job
    status: str
    steps: float = 0.0
    total: float = 0.0
    sps: float | None = None          # recent speed (steps / s)
    eta_s: float | None = None        # remaining seconds (estimated for pending jobs)
    estimated: bool = False
    ret: float | None = None
    crash: float | None = None
    ckpts: int = 0
    proc: Proc | None = None
    note: str = ""


def history_sps(min_total: float = 50e6) -> dict[str, list[float]]:
    """Average training speed of earlier full-size runs, by training script."""
    out: dict[str, list[float]] = {}
    for cfg_path in list(RUNS.glob("*/config.json")) + list(RUNS.glob("*/*/config.json")):
        cfg = _config(cfg_path.parent)
        kind = script_kind(cfg)
        total = cfg.get("ppo", {}).get("total_timesteps", 0)
        rows = tail_rows(cfg_path.parent / "metrics.csv", 1)
        if kind and total >= min_total and rows and isinstance(rows[-1].get("sps"), float):
            out.setdefault(kind, []).append(rows[-1]["sps"])
    return out


def _steps_arg(args: str) -> float:
    m = re.search(r"--steps\s+(\S+)", args)
    try:
        return float(m.group(1)) if m else 0.0
    except ValueError:
        return 0.0


def job_state(job: Job, procs: list[Proc], queue_alive: bool, hist: dict) -> JobState:
    run_dir, log = RUNS / job.name, RUNS / f"{job.name}.log"
    pat = re.compile(rf"--name\s+{re.escape(job.name)}(\s|$)")
    mine = [p for p in procs if pat.search(p.cmd) and not p.cmd.startswith("uv ")]
    proc = max(mine, key=lambda p: p.cpu) if mine else None
    cfg = _config(run_dir)
    ppo = cfg.get("ppo", {})
    total = float(ppo.get("total_timesteps") or _steps_arg(job.args))
    batch = float(ppo.get("num_envs", 0) * ppo.get("num_steps", 0))
    rows = tail_rows(run_dir / "metrics.csv", 2)
    st = JobState(job, "pending", total=total, proc=proc)
    st.ckpts = len(list((run_dir / "ckpt").glob("*.pkl"))) if (run_dir / "ckpt").is_dir() else 0
    if rows:
        last = rows[-1]
        st.steps = float(last.get("timesteps") or 0.0)
        st.ret = _first(last, "episode_return", "replay_return", "explore_return")
        st.crash = _first(last, "terminated_frac")
        st.sps = last.get("sps") if isinstance(last.get("sps"), float) else None
        if len(rows) == 2:
            dt = _f(last.get("wall_s")) - _f(rows[0].get("wall_s"))
            ds = st.steps - _f(rows[0].get("timesteps"))
            if dt > 0 and ds > 0:
                st.sps = ds / dt
    complete = total > 0 and st.steps >= total - max(batch, 1.0)
    failed_log = log.exists() and "Traceback" in _tail_text(log)

    if proc:
        st.status = "evaluating" if complete else ("running" if rows else "compiling")
    elif complete:
        st.status = "done"
    elif failed_log:
        st.status = "failed"
    elif log.exists() or rows:
        st.status = "stopped" if not queue_alive else "failed"
    else:
        st.status = "pending"

    if st.status == "running" and st.sps:
        st.eta_s = (total - st.steps) / st.sps
    elif st.status in ("pending", "compiling"):
        known = hist.get(job.script)
        if known and total:
            st.eta_s, st.estimated = total / statistics.median(known), True
            st.sps = statistics.median(known)
    return st


def _first(row: dict, *keys):
    for k in keys:
        if isinstance(row.get(k), float):
            return row[k]
    return None


def _f(v) -> float:
    return v if isinstance(v, float) else 0.0


def _tail_text(path: Path, nbytes: int = 4096) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - nbytes))
            return fh.read().decode(errors="ignore")
    except OSError:
        return ""


def machine() -> str:
    load = os.getloadavg()[0]
    ncpu = os.cpu_count() or 1
    parts = [f"load {load:4.1f} / {ncpu} cores"]
    total = _run(["sysctl", "-n", "hw.memsize"]).strip()
    vm = _run(["vm_stat"])
    page = re.search(r"page size of (\d+)", vm)
    if total.isdigit() and page:
        def pages(label):
            m = re.search(rf"{label}:\s+(\d+)", vm)
            return int(m.group(1)) if m else 0
        used = (pages("Pages active") + pages("Pages wired down")
                + pages("Pages occupied by compressor")) * int(page.group(1))
        parts.append(f"mem {used / 2**30:4.1f} / {int(total) / 2**30:.0f} GB")
    batt = _run(["pmset", "-g", "batt"])
    src = re.search(r"'(.+?)'", batt)
    pct = re.search(r"(\d+)%", batt)
    if src:
        parts.append(src.group(1) + (f" {pct.group(1)}%" if pct else ""))
    asrt = _run(["pmset", "-g", "assertions"])
    m = re.search(r"PreventUserIdleSystemSleep\s+(\d)", asrt)
    if m:
        parts.append("idle sleep " + (f"{GREEN}prevented{RESET}" if m.group(1) == "1"
                                      else f"{YELLOW}allowed{RESET}"))
    return "   ".join(parts)


# ---------------------------------------------------------------------------- render

def fmt_dur(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(s)
    if s < 3600:
        return f"{s // 60} min"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


def clock(epoch: float) -> str:
    return time.strftime("%H:%M", time.localtime(epoch))


def bar(frac: float, width: int = 18) -> str:
    frac = min(max(frac, 0.0), 1.0)
    full = round(frac * width)
    return "█" * full + "░" * (width - full)


def fit(line: str, width: int) -> str:
    """Truncate to a visible width, keeping ANSI codes intact."""
    out, seen, i = [], 0, 0
    while i < len(line):
        m = ANSI.match(line, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        if seen >= width:
            break
        out.append(line[i])
        seen += 1
        i += 1
    return "".join(out) + RESET


PIPELINE = ("manifest", "support", "stimuli", "keys", "validate", "controls", "dryrun",
            "rollouts", "anchored-keys", "probes", "analyse", "plots")
PREREG_TAG = "psychophysics-prereg-v2"
_STAGE = re.compile(r"^(\S+ \S+) (start|done|failed) (\S+): (\d+)/(\d+)(.*)$")
_PROGRESS = re.compile(r"^(\S+ \S+) (\d+)/(\d+) (.*)$")


@dataclass
class StageState:
    stage: str
    status: str                    # running | done | failed | stopped
    done: int = 0
    total: int = 0
    when: str = ""                 # time of the last line
    summary: str = ""              # the last informative line


def stage_state(log: Path, running: bool) -> StageState | None:
    """The latest run of a psychophysics stage from runs/psychophysics-<stage>.log: its
    start / progress / done / failed lines (``io.StageLog``) and the last other line."""
    stage = log.stem.removeprefix("psychophysics-")
    lines = _tail_text(log, 32768).splitlines()
    st = None
    for ln in lines:
        m = _STAGE.match(ln)
        if m and m.group(3) == stage:
            if m.group(2) == "start":
                st = StageState(stage, "running" if running else "stopped", int(m.group(4)),
                                int(m.group(5)), m.group(1)[11:16])
            elif st is not None or m.group(2) != "start":
                st = st or StageState(stage, "")
                st.status = "running" if running else m.group(2)
                st.done, st.total, st.when = int(m.group(4)), int(m.group(5)), m.group(1)[11:16]
                if m.group(2) == "failed":
                    st.summary = m.group(6).strip()
            continue
        if st is None:
            continue
        m = _PROGRESS.match(ln)
        if m:
            st.done, st.total, st.when = int(m.group(2)), int(m.group(3)), m.group(1)[11:16]
        elif len(ln) > 20 and st.status != "failed":
            st.summary, st.when = ln[20:], ln[11:16]
    return st



# Agent workflows run by Claude Code for this repo keep a journal per run (started / result
# events per agent) under ~/.claude/projects/<repo path as a slug>/<session>/subagents/.
CLAUDE_PROJECT = Path.home() / ".claude" / "projects" / re.sub(r"[/ ]", "-", str(ROOT))
STALL_S = 20 * 60


@dataclass
class WorkflowState:
    name: str
    running: list[str]            # labels of agents started without a result
    done: int
    started: int
    last_s: float                 # seconds since any of its files changed


def workflows(base: Path = CLAUDE_PROJECT, within_s: float = 12 * 3600,
              now: float | None = None) -> list[WorkflowState]:
    """Recent agent workflows: their running agents, progress and last activity."""
    now = time.time() if now is None else now
    names = {f.stem.rsplit("-wf_", 1)[1]: f.stem.rsplit("-wf_", 1)[0]
             for f in base.glob("*/workflows/scripts/*-wf_*.js")}
    out = []
    for j in base.glob("*/subagents/workflows/wf_*/journal.jsonl"):
        files = [j, *j.parent.glob("agent-*.jsonl")]
        try:
            last = max(f.stat().st_mtime for f in files)
        except OSError:
            continue
        if now - last > within_s:
            continue
        started, results = {}, set()
        for ln in _tail_text(j, 1 << 20).splitlines():
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            if e.get("type") == "started":
                started[e.get("agentId")] = e.get("label") or e.get("phase") or "agent"
            elif e.get("type") == "result":
                results.add(e.get("agentId"))
        run_id = j.parent.name.removeprefix("wf_")
        out.append(WorkflowState(names.get(run_id, j.parent.name), [
            lab for a, lab in started.items() if a not in results],
            len(results & set(started)), len(started), now - last))
    return sorted(out, key=lambda w: w.last_s)



REMOTE_JOBS = RUNS / "remote_jobs.json"     # [{"host", "log", "label"}]: jobs on rented boxes
_REMOTE: dict = {"t": 0.0, "lines": {}}


def remote_jobs(every: float = 60.0) -> list[tuple[str, str, str]]:
    """(label, host, last log line) for each job listed in runs/remote_jobs.json, read over
    ssh at most once a minute (5 s timeout per host)."""
    try:
        jobs = json.loads(REMOTE_JOBS.read_text())
    except (OSError, ValueError):
        return []
    if time.time() - _REMOTE["t"] > every:
        for j in jobs:
            out = _run(["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", j["host"],
                        f"tail -n 1 {j['log']}"])
            _REMOTE["lines"][j["host"] + j["log"]] = out.strip().splitlines()[-1] if out.strip() else "(no reply)"
        _REMOTE["t"] = time.time()
    return [(j.get("label", j["log"]), j["host"], _REMOTE["lines"].get(j["host"] + j["log"], "..."))
            for j in jobs]


_BREV: dict = {"t": 0.0, "rows": None}


def gpu_instances(every: float = 60.0) -> list[str] | None:
    """``brev ls`` rows (name, status, machine, GPU), refreshed at most once a minute;
    None if the Brev CLI is absent."""
    if shutil.which("brev") is None:
        return None
    if time.time() - _BREV["t"] > every or _BREV["rows"] is None:
        out = _run(["brev", "ls"])
        rows = []
        for ln in out.splitlines():
            f = ln.split()
            if len(f) >= 6 and f[0] not in ("NAME", "You") and not ln.startswith(" NAME"):
                rows.append(f)
        _BREV.update(t=time.time(), rows=rows)
    return _BREV["rows"]


def render(width: int) -> list[str]:
    now = time.time()
    procs = processes()
    lines = [f"{BOLD}ApexZero: what's running{RESET}".ljust(60) + time.strftime("%a %d %b %H:%M:%S"),
             machine(), ""]

    # --- queue
    queue_proc = next((p for p in procs if "queue_overnight.sh" in p.cmd and "bash" in p.cmd), None)
    jobs = parse_queue(QUEUE.read_text()) if QUEUE.exists() else []
    hist = history_sps()
    states = [job_state(j, procs, queue_proc is not None, hist) for j in jobs]
    t, unknown = now, False
    finish = {}
    for st in states:
        if st.status in ("running", "pending", "compiling", "evaluating"):
            if st.eta_s is None:
                unknown = True
            else:
                t += st.eta_s
            finish[st.job.name] = None if unknown else t
    if queue_proc:
        head = f"{GREEN}● running{RESET} {fmt_dur(queue_proc.elapsed)}"
        known = [(n, f) for n, f in finish.items() if f]
        if finish and len(known) == len(finish):
            head += f"   est. finish {BOLD}~{clock(known[-1][1])}{RESET}"
        elif known:
            nxt = list(finish)[len(known)]
            head += (f"   est. ~{clock(known[-1][1])} until {nxt} starts"
                     f" {DIM}(no speed history for it yet){RESET}")
    else:
        head = f"{DIM}not running{RESET}"
    lines.append(f"{BOLD}TRAINING QUEUE{RESET} {DIM}(scripts/queue_overnight.sh){RESET}   {head}")
    idle = queue_proc is None and all(st.status in ("done", "failed", "stopped") for st in states)
    if idle and states:
        n_done = sum(st.status == "done" for st in states)
        lines.append(f"  {DIM}last queue: {n_done} of {len(states)} jobs done "
                     f"({', '.join(st.job.name for st in states)}){RESET}")
    else:
        lines.append(f"{DIM}  #  {'job':<34} {'status':<10} {'progress':<17} {'eta':<20} {'steps':>11} "
                     f"{'speed':>9} {'return':>7} {'crash':>7} {'ckpt':>4}{RESET}")
    for i, st in enumerate(states if not (idle and states) else [], 1):
        col = STATUS_COLOUR.get(st.status, "")
        frac = st.steps / st.total if st.total else 0.0
        prog = f"{bar(frac, 12)} {frac * 100:3.0f}%" if st.steps else ""
        steps = (f"{st.steps / 1e6:5.1f}/{st.total / 1e6:.0f}M" if st.steps
                 else f"{st.total / 1e6:.0f}M" if st.total else "")
        speed = (("~" if st.estimated else "") + f"{st.sps / 1e3:.1f}k/s") if st.sps else ""
        ret = f"{st.ret:7.1f}" if st.ret is not None else ""
        crash = f"{st.crash * 100:6.2f}%" if st.crash is not None else ""
        if st.status in ("running", "pending", "compiling") and st.job.name in finish:
            f_at = finish[st.job.name]
            eta = (("~" if st.estimated else "") + clock(f_at) + f" ({fmt_dur(st.eta_s)})"
                   if f_at else "?")
        elif st.status == "evaluating":
            eta = "final evaluation"
        elif st.status == "done":
            eta = ""
        else:
            eta = st.status
        lines.append(f"  {i}  {st.job.name:<34} {col}{st.status:<10}{RESET} {prog:<17} {eta:<20} {steps:>11} "
                     f"{speed:>9} {ret:>7} {crash:>7} {st.ckpts or '':>4}")
    lines.append("")

    # --- psychophysics pipeline
    tagged = _run(["git", "-C", str(ROOT), "tag", "-l", PREREG_TAG]).strip()
    tag_txt = (f"{GREEN}tagged {PREREG_TAG}{RESET}" if tagged
               else f"{YELLOW}not tagged yet{RESET} {DIM}(no agent is rolled out on stimuli before it){RESET}")
    lines.append(f"{BOLD}PSYCHOPHYSICS PIPELINE{RESET} {DIM}(scripts/psychophysics.py){RESET}   {tag_txt}")
    shown = 0
    for stage in PIPELINE:
        log = RUNS / f"psychophysics-{stage}.log"
        if not log.exists():
            continue
        running = any(f"psychophysics.py {stage}" in p.cmd for p in procs)
        st = stage_state(log, running)
        if st is None:
            continue
        col = STATUS_COLOUR.get(st.status, "")
        frac = st.done / st.total if st.total else 0.0
        prog = f"{bar(frac, 12)} {st.done}/{st.total}" if st.total else ""
        lines.append(f"  {stage:<14} {col}{st.status:<8}{RESET} {prog:<24} {st.when:>5}  {DIM}{st.summary}{RESET}")
        shown += 1
    if not shown:
        lines.append(f"  {DIM}no stage has run{RESET}")
    lines.append("")

    # --- agent workflows
    wfs = workflows(now=now)[:4]
    lines.append(f"{BOLD}AGENT WORKFLOWS{RESET} {DIM}(agents done/started, last activity){RESET}")
    if wfs:
        for w in wfs:
            if w.running and w.last_s > STALL_S:
                st = f"{RED}stalled?{RESET}"
            elif w.running:
                st = f"{GREEN}running{RESET} "
            else:
                st = f"{BLUE}done{RESET}    "
            now_doing = ", ".join(w.running) if w.running else ""
            lines.append(f"  {w.name.removeprefix('psychophysics-')[:24]:<24} {st} {w.done}/{w.started} "
                         f"{fmt_dur(w.last_s):>9} ago  {DIM}{now_doing}{RESET}")
    else:
        lines.append(f"  {DIM}none{RESET}")
    lines.append("")

    # --- GPU instances
    rows = gpu_instances()
    if rows is not None:
        lines.append(f"{BOLD}GPU (Brev){RESET} {DIM}(brev ls, refreshed each minute){RESET}")
        if rows:
            for f in rows:
                col = GREEN if f[1] == "RUNNING" else YELLOW
                lines.append(f"  {f[0]:<28} {col}{f[1]:<10}{RESET} {f[2]:<10} {f[3]:<10} "
                             f"{' '.join(f[5:])}")
        else:
            lines.append(f"  {DIM}none{RESET}")
        for label, host, last in remote_jobs():
            lines.append(f"  {CYAN}{label}{RESET} {DIM}on {host}:{RESET} {last}")
        lines.append("")

    # --- other experiment processes
    queue_pids = {st.proc.pid for st in states if st.proc}
    others = [p for p in procs if "scripts/" in p.cmd and "python" in p.cmd.lower()
              and not p.cmd.startswith("uv ") and "dashboard.py" not in p.cmd
              and p.pid not in queue_pids]
    lines.append(f"{BOLD}OTHER EXPERIMENTS{RESET} {DIM}(python processes running scripts/){RESET}")
    if others:
        lines.append(f"{DIM}  {'pid':>6} {'elapsed':>10} {'cpu':>6} {'mem':>7}  command{RESET}")
        for p in sorted(others, key=lambda p: -p.elapsed):
            cmd = p.cmd[p.cmd.find("scripts/"):]
            lines.append(f"  {p.pid:>6} {fmt_dur(p.elapsed):>10} {p.cpu:5.0f}% "
                         f"{p.rss_mb / 1024:5.1f}GB  {cmd}")
    else:
        lines.append(f"  {DIM}none{RESET}")
    lines.append("")

    # --- recently updated runs outside the queue
    queue_names = {j.name for j in jobs}
    recent = []
    for m in list(RUNS.glob("*/metrics.csv")) + list(RUNS.glob("*/*/metrics.csv")) + list(RUNS.glob("*.log")):
        name = m.parent.relative_to(RUNS).as_posix() if m.name == "metrics.csv" else m.stem
        try:
            age = now - m.stat().st_mtime
        except OSError:
            continue
        if age < RECENT_S and name not in queue_names:
            recent.append((age, name, m.relative_to(ROOT).as_posix()))
    lines.append(f"{BOLD}OTHER RUNS UPDATED IN THE LAST 15 MIN{RESET}")
    if recent:
        for age, _, path in sorted(recent):
            lines.append(f"  {fmt_dur(age):>8} ago  {path}")
    else:
        lines.append(f"  {DIM}none{RESET}")
    lines.append("")

    # --- notes
    if NOTES.exists():
        age = now - NOTES.stat().st_mtime
        lines.append(f"{BOLD}NOTES{RESET} {DIM}(runs/activity.md, updated {fmt_dur(age)} ago){RESET}")
        for ln in NOTES.read_text().splitlines()[:20]:
            if ln.strip():
                lines.append("  " + ln)
    return [fit(ln, width) for ln in lines]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    ap.add_argument("--every", type=float, default=5.0, help="refresh interval (s)")
    args = ap.parse_args()
    if args.once or not sys.stdout.isatty():
        frame = "\n".join(render(shutil.get_terminal_size((160, 40)).columns))
        print(frame if sys.stdout.isatty() else ANSI.sub("", frame))
        return
    sys.stdout.write("\x1b[?25l\x1b[2J")
    src = Path(__file__).resolve()
    born = src.stat().st_mtime
    try:
        while True:
            frame = render(shutil.get_terminal_size((160, 40)).columns)
            sys.stdout.write("\x1b[H" + "\x1b[K\n".join(frame) + "\x1b[K\n\x1b[J")
            sys.stdout.flush()
            time.sleep(args.every)
            if src.stat().st_mtime != born:          # edited: restart on the new code
                sys.stdout.write("\x1b[?25h")
                os.execv(sys.executable, [sys.executable, str(src), *sys.argv[1:]])
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?25h\n")


if __name__ == "__main__":
    main()
