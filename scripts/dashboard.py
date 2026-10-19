#!/usr/bin/env python3
"""Live terminal dashboard of everything running in this repo.

    uv run python scripts/dashboard.py            # refresh every 5 s (Ctrl-C to quit)
    uv run python scripts/dashboard.py --once     # print one frame and exit

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
                 "pending": YELLOW, "failed": RED, "stopped": RED}
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
    lines.append(f"{DIM}  #  {'job':<36} {'status':<11} {'progress':<24} {'steps':>11} "
                 f"{'speed':>9} {'return':>8} {'crash':>7} {'ckpt':>4}  eta{RESET}")
    for i, st in enumerate(states, 1):
        col = STATUS_COLOUR.get(st.status, "")
        frac = st.steps / st.total if st.total else 0.0
        prog = f"{bar(frac)} {frac * 100:3.0f}%" if st.steps else ""
        steps = (f"{st.steps / 1e6:5.1f}/{st.total / 1e6:.0f}M" if st.steps
                 else f"{st.total / 1e6:.0f}M" if st.total else "")
        speed = (("~" if st.estimated else "") + f"{st.sps / 1e3:.1f}k/s") if st.sps else ""
        ret = f"{st.ret:8.1f}" if st.ret is not None else ""
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
        lines.append(f"  {i}  {st.job.name:<36} {col}{st.status:<11}{RESET} {prog:<24} {steps:>11} "
                     f"{speed:>9} {ret:>8} {crash:>7} {st.ckpts or '':>4}  {eta}")
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
    try:
        while True:
            frame = render(shutil.get_terminal_size((160, 40)).columns)
            sys.stdout.write("\x1b[H" + "\x1b[K\n".join(frame) + "\x1b[K\n\x1b[J")
            sys.stdout.flush()
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?25h\n")


if __name__ == "__main__":
    main()
