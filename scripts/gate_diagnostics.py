"""Gate diagnostics (GW-1): EXPLORATORY (docs/findings/gate-diagnostics-plan.md, committed in
916edaa before anything here was computed; dated amendments in its Section 10). Every number
this script writes is exploratory (or descriptive). Nothing is trained.

    nice -n 15 uv run python scripts/gate_diagnostics.py time      # D0: one evaluation timed
    nice -n 15 uv run python scripts/gate_diagnostics.py curves    # D1 (and D3's traces)
    nice -n 15 uv run python scripts/gate_diagnostics.py adr       # D2
    uv run python scripts/gate_diagnostics.py crashes              # D3 (seconds, from traces)
    uv run python scripts/gate_diagnostics.py report               # the readings, E1-E5

**Lockbox.** Only the 12 R-dev circuits and generated A_DR levels are used. Every circuit name
passes ``viz.export.lockbox_guard`` before any rollout (R_TEST names and their mirrored or
reversed variants raise PermissionError), and any name outside R_DEV is refused. Gaps are
measured only against outputs/ocp/<R-dev circuit>.pkl. outputs/sealed/ is never read.

**Networks** (plan Section 2): the 28 same-code runs, H4R's 20 (``outputs/h4r/manifest.json``)
and the 8 leave-out runs (``src/apexzero/fields/prereg.json``). Every final is checked against
its bound sha256 before it is read; the sha256 of every other checkpoint and of every file read
is recorded.

**D1** (plan Section 4). Checkpoint indices 0, 3, 6, 9, 12, 15 of each run's 16, evaluated with
``scripts/eval_rdev.py``'s protocol. The rollout (``rollout_laps``) is ``eval.laps.timed_laps``
line for line with more outputs (every run's s, n, v_x and termination flags at every step); the
tests check that it reproduces ``timed_laps`` and ``scripts/eval_rdev.py``'s JSON exactly.
C counts circuits with completion fraction > 0.5; c_bar is the mean completion fraction.

**D2** (plan Section 5). The 1,000 A_DR levels (``make_levels(PRNGKey(2026), 'a_dr')``, grammar
fitted on R-dev, as ``scripts/pilot_audit.py`` builds them). Primary: the mean action without
observation noise over one 1,200-step window per level; a level is completed when the window
ends without termination. Secondary: ``audit.rollouts.rollout_windows`` at its defaults, 4
windows per level, key ``fold_in(PRNGKey(2026), 1)``.

**D3** (plan Section 6) works on the traces of H4R's 20 finals only. Corners are
``tracks.corners.segment`` at its defaults on each refined track; W_k = [s_start - 50,
s_end + 50] m; clearance c = w_in - 1.0 - sigma n, with the environment's own width spline.

**Workers.** At most 4 processes at a time; each worker lowers its priority to nice 15 and runs
with 2 XLA CPU threads (``--threads``). Another workflow shares the Mac.

**Outputs** (outputs/gate-diagnostics/): timing.json, evals/<run>/<step>.json, traces/<run>.npz,
adr/<run>.json, curves.json, adr.json, crashes.json, each with a provenance block.
"""

from __future__ import annotations

import os
import sys
import time

_T0 = time.time()


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

import argparse
import dataclasses
import hashlib
import json
import multiprocessing
import pickle
import platform
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import Actor, PPOConfig
from apexzero.audit.rollouts import rollout_windows, snapshot_from_pickle
from apexzero.env import racing
from apexzero.env.adapters import reference_speed
from apexzero.tracks.corners import R_DEV, segment
from apexzero.tracks.generator import fit_grammar
from apexzero.tracks.levels import build, make_levels
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.track import Track, load_tumftm
from apexzero.vehicle.sim import track_arrays, widths_at
from apexzero.viz.export import lockbox_guard

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "outputs" / "gate-diagnostics"
PLAN = ROOT / "docs" / "findings" / "gate-diagnostics-plan.md"
PLAN_COMMIT = "916edaa"
MANIFEST = ROOT / "outputs" / "h4r" / "manifest.json"
PREREG = ROOT / "src" / "apexzero" / "fields" / "prereg.json"
TRACK_DIR = ROOT / "data" / "tracks" / "tumftm"
OCP_DIR = ROOT / "outputs" / "ocp"
PROTOCOL_VERSION = 1      # bump when an evaluation's content changes

# ------------------------------------------------------------------------- the networks


@dataclasses.dataclass(frozen=True)
class Net:
    name: str
    arm: str
    variant: str


ARMS = ("v1-base", "v1-realw", "v3-base", "v3-realw", "v3-loo")
H4R_ARMS = ARMS[:4]
NETWORKS: tuple[Net, ...] = (
    *(Net(f"h4r-{v}-{a}-s{s}", f"{v}-{a}", v)
      for v in ("v1", "v3") for a in ("base", "realw") for s in range(10, 15)),
    *(Net(f"dr-realw-loo-f{f}-s{s}", "v3-loo", "v3") for f in (1, 2) for s in range(4)),
)
BY_NAME = {n.name: n for n in NETWORKS}
VARIANTS = ("v1", "v3")

# ------------------------------------------------------------------------- the protocol

GRID_IDX = (0, 3, 6, 9, 12, 15)
GRID_STEPS = (10_000_000, 18_205_642, 33_144_540, 60_341_763, 109_856_054, 199_884_800)
FINAL = GRID_STEPS[-1]
N_CHECKPOINTS = 16
TIMED = ("h4r-v1-base-s10", GRID_STEPS[0])      # D0's evaluation, kept as a D1 grid point
N_EVAL = len(NETWORKS) * len(GRID_STEPS)         # 168
N_RUNS = 10
LAPS = 2
START_FRACTION = 0.9
COMPLETION_MIN = 0.5      # a circuit is completed when MORE than half its runs finish
MAX_WORKERS = 4
NICE = 15
W_MAC_MAX_H = 6.0         # D0's schedule rule
CAUSES = ("corridor", "slip", "spin", "stall")   # diagnose_crashes.first_crash's order

N_BOOT = 20_000
BOOT_SEED = 0
FLAT_BAND = 0.5           # circuits (the late label)
LATE_FROM, MID_FROM = GRID_STEPS[4], GRID_STEPS[3]

ADR_ROOT_SEED = 2026
ADR_WINDOWS = 4
ADR_KEY_FOLD = 1
ADR_CHUNK = 100
LATE_PHASE = 0.9
WILSON_Z = 1.959963984540054

WINDOW_PAD = 50.0         # m, step 2's c_min window
EDGE_MARGIN = racing.EnvParams().edge_margin     # 1.0 m, the evaluation's termination boundary
FORMER_MIN = 3            # baseline terminations (of 50) that make a former crash corner
DELTA_B = {"v1": 0.44, "v3": 0.27}               # H4R's confirmatory B shift (arm means, m)
BAND = (0.5, 2.0)         # "about the B shift": dc in [0.5, 2] x dB


class CheckFailed(SystemExit):
    """A built-in check failed: the stage stops and writes nothing more."""


# ------------------------------------------------------------------------- provenance


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def provenance(inputs: dict[str, str], start: float, argv=None) -> dict:
    return {
        "label": "exploratory (docs/findings/gate-diagnostics-plan.md)",
        "plan": rel(PLAN), "plan_commit": PLAN_COMMIT, "plan_sha256": sha256_file(PLAN),
        "script": rel(SCRIPT), "script_sha256": sha256_file(SCRIPT),
        "git_head": git("rev-parse", "HEAD"),
        "changed_paths": [ln for ln in git("status", "--porcelain", "--untracked-files=no")
                          .splitlines() if ln.strip()],
        "src_clean": not git("status", "--porcelain", "--", "src").strip(),
        "inputs_sha256": dict(sorted(inputs.items())),
        "argv": [rel(SCRIPT), *(sys.argv[1:] if argv is None else argv)],
        "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
        "end": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - start, 1),
        "environment": environment(),
    }


def environment() -> dict:
    return {"python": platform.python_version(), "jax": jax.__version__,
            "numpy": np.__version__, "machine": platform.machine(),
            "system": platform.system(), "node_kind": "mac" if platform.system() == "Darwin"
            else "other", "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "nice": os.nice(0), "cpu_count": os.cpu_count()}


def clean(x):
    """JSON-safe: NaN and infinities as null, NumPy values as Python ones."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=1, allow_nan=False))
    tmp.replace(path)


def lower_priority(level: int = NICE) -> None:
    cur = os.nice(0)
    if cur < level:
        os.nice(level - cur)


# ------------------------------------------------------------------------- bindings


def bound_finals() -> dict[str, dict]:
    """{run: {'sha256', 'eval_sha256' (H4R only)}} for the 28 finals, from the committed
    records; each entry's path must be the run's final checkpoint."""
    man = json.loads(MANIFEST.read_text())["networks"]
    pre = json.loads(PREREG.read_text())["networks"]
    out = {}
    for net in NETWORKS:
        key = f"runs/{net.name}"
        ent = (man if net.arm in H4R_ARMS else pre)[key]
        if ent["path"] != f"{key}/ckpt/{FINAL:012d}.pkl" or int(ent["timesteps"]) != FINAL:
            raise CheckFailed(f"{key}: bound final is {ent['path']} ({ent['timesteps']})")
        out[net.name] = {"sha256": ent["sha256"], "eval_sha256": ent.get("eval_sha256")}
    return out


def checkpoint_paths(name: str) -> list[Path]:
    """The run's 16 checkpoints; refuses a run whose schedule is not the plan's."""
    paths = sorted((ROOT / "runs" / name / "ckpt").glob("*.pkl"))
    steps = tuple(int(p.stem) for p in paths)
    if len(paths) != N_CHECKPOINTS or tuple(steps[i] for i in GRID_IDX) != GRID_STEPS:
        raise CheckFailed(f"{name}: checkpoint schedule {steps} is not the plan's")
    return paths


def checkpoint_path(name: str, step: int) -> Path:
    if name not in BY_NAME or step not in GRID_STEPS:
        raise ValueError(f"{name} @ {step} is not on D1's grid")
    return ROOT / "runs" / name / "ckpt" / f"{step:012d}.pkl"


def load_checkpoint(path: Path, bound: str | None = None) -> tuple[dict, str]:
    """(snapshot dict, sha256 of the bytes read); refuses a final whose sha256 is not bound."""
    raw = Path(path).read_bytes()
    sha = sha256_bytes(raw)
    if bound is not None and sha != bound:
        raise CheckFailed(f"{path}: sha256 {sha} is not the bound {bound}")
    return pickle.loads(raw), sha


def make_policy(d: dict):
    """(policy, cfg, snapshot): ``scripts/eval_rdev.py``'s deterministic policy."""
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = PPOConfig(**{k: v for k, v in d["config"].items() if k in known})
    snap = snapshot_from_pickle(d)
    actor = Actor(cfg, 2, False)

    def policy(obs):
        return actor.apply(snap.actor, snap.obs_rms.normalize(obs))[0]

    return policy, cfg, snap


# ------------------------------------------------------------------------- circuits

_TRACKS: dict[str, Track] = {}


def check_circuit(name: str) -> None:
    """The lockbox guard (PermissionError on R_TEST and its variants), then R-dev only."""
    lockbox_guard(name)
    if name not in R_DEV:
        raise ValueError(f"{name} is not an R-dev circuit")


def circuit(name: str) -> Track:
    check_circuit(name)
    if name not in _TRACKS:
        _TRACKS[name] = refine_reference(load_tumftm(TRACK_DIR / f"{name}.csv"))
    return _TRACKS[name]


def ocp_lap(name: str) -> tuple[float, str]:
    check_circuit(name)
    raw = (OCP_DIR / f"{name}.pkl").read_bytes()
    return float(pickle.loads(raw).lap_time), sha256_bytes(raw)


def circuit_inputs(names=R_DEV) -> dict[str, str]:
    out = {}
    for name in names:
        check_circuit(name)
        out[rel(TRACK_DIR / f"{name}.csv")] = sha256_file(TRACK_DIR / f"{name}.csv")
        out[rel(OCP_DIR / f"{name}.pkl")] = sha256_file(OCP_DIR / f"{name}.pkl")
    return out


# ------------------------------------------------------------------------- the rollout


@dataclasses.dataclass
class LapRollout:
    """``eval.laps.LapResult`` plus every run's arrays (post-step states, as timed_laps)."""
    completed: np.ndarray      # [R]
    lap_time: np.ndarray       # [R]
    s_reached: np.ndarray      # [R]
    trace: dict                # run 0, exactly as timed_laps
    t: np.ndarray              # [T]
    s: np.ndarray              # [T, R]
    n: np.ndarray
    vx: np.ndarray
    alive: np.ndarray          # [T, R] after the step
    term: np.ndarray           # [T, R] raw flags of step_no_reset
    off: np.ndarray
    slip: np.ndarray
    spun: np.ndarray
    stall: np.ndarray
    length: float
    laps: int

    @property
    def completion_rate(self) -> float:
        return float(self.completed.mean())

    @property
    def median_lap(self) -> float:
        lt = self.lap_time[self.completed]
        return float(np.median(lt)) if len(lt) else float("nan")


def rollout_laps(policy_fn, track: Track, v_start: float, runs: int = 20, laps: int = 2,
                 params: racing.EnvParams = racing.EnvParams(), ctx=None,
                 seed: int = 0) -> LapRollout:
    """``eval.laps.timed_laps``, line for line, keeping every run's s, n, v_x, alive and
    termination flags at every step (the tests check exact equality)."""
    tr = track_arrays(track)
    ctx = racing.default_context() if ctx is None else ctx
    L = float(track.length)
    max_steps = int(laps * 3.0 * L / max(v_start, 20.0) / params.dt) + 200
    eval_params = params._replace(window_steps=10**9)

    rng = np.random.default_rng(seed)
    n0 = np.where(np.arange(runs) == 0, 0.0, rng.uniform(-1.0, 1.0, runs))
    v0 = v_start * np.where(np.arange(runs) == 0, 1.0, rng.uniform(0.95, 1.05, runs))
    x0 = np.zeros((runs, 8), np.float32)
    x0[:, 1], x0[:, 3] = n0, v0

    def init_state(x):
        return racing.EnvState(x=x, prev_action=jnp.zeros(2), accel=jnp.zeros(2), ctx=ctx,
                               hidden_mu=jnp.array(1.0),
                               t=jnp.zeros((), jnp.int32), slow=jnp.zeros(()), s0=x[0])

    states = jax.vmap(init_state)(jnp.asarray(x0))
    p = racing.params_from_context(ctx)
    obs = jax.vmap(lambda st: racing.observe(st, tr, p))(states)

    @jax.jit
    def run(states, obs):
        def body(c, _):
            st, ob, alive = c
            a = policy_fn(ob)
            ob2, st2, _, term, _, info = jax.vmap(
                lambda s, aa: racing.step_no_reset(s, aa, tr, eval_params))(st, a)
            alive2 = alive & ~term
            st2 = jax.tree.map(lambda new, old: jnp.where(
                alive.reshape((-1,) + (1,) * (new.ndim - 1)), new, old), st2, st)
            return (st2, ob2, alive2), (st2.x[:, 0], st2.x[:, 1], st2.x[:, 3], a, alive2, term,
                                        info["off_track"], info["slipped"], info["spun"],
                                        info["stalled"])
        _, out = jax.lax.scan(body, (states, obs, jnp.ones(runs, bool)), None, max_steps)
        return out

    s, n, vx, acts, alive, term, off, slip, spun, stall = (np.asarray(a) for a in run(states, obs))
    t = (np.arange(max_steps) + 1) * params.dt
    completed = np.zeros(runs, bool)
    lap_time = np.full(runs, np.nan)
    for r in range(runs):
        def cross(target, r=r):
            idx = np.nonzero(s[:, r] >= target)[0]
            if not len(idx):
                return None
            k = idx[0]
            s_prev = s[k - 1, r] if k else 0.0
            return t[k] - params.dt + (target - s_prev) / max(s[k, r] - s_prev, 1e-6) * params.dt
        t1, t2 = cross((laps - 1) * L), cross(laps * L)
        if t1 is not None and t2 is not None and alive[np.nonzero(s[:, r] >= laps * L)[0][0], r]:
            completed[r] = True
            lap_time[r] = t2 - t1
    first_end = int(np.argmax(s[:, 0] >= laps * L)) or max_steps
    trace = {"t": t[:first_end], "s": s[:first_end, 0], "n": n[:first_end, 0],
             "vx": vx[:first_end, 0], "action": acts[:first_end, 0]}
    return LapRollout(completed, lap_time, s.max(axis=0), trace, t, s, n, vx, alive, term,
                      off, slip, spun, stall, L, laps)


def outcomes(ro: LapRollout) -> dict[str, np.ndarray]:
    """Per run: crash (terminated without completing), timeout (neither), the terminating step
    and its s (mod L), lap, n, v_x and cause, and ``end``, the last sample of the run's
    passage through its two laps (the termination step, else the last sample before s = 2L)."""
    T, R = ro.s.shape
    prev = np.vstack([np.ones((1, R), bool), ro.alive[:-1]])
    hit = ro.term & prev
    out = {k: np.full(R, np.nan) for k in ("term_s", "term_n", "term_vx")}
    out.update(term_step=np.full(R, -1), term_lap=np.full(R, -1), end=np.full(R, T - 1),
               crash=np.zeros(R, bool), timeout=np.zeros(R, bool),
               cause=np.array([""] * R, dtype=object))
    for r in range(R):
        ks = np.nonzero(hit[:, r])[0]
        if ro.completed[r]:
            out["end"][r] = int(np.nonzero(ro.s[:, r] >= ro.laps * ro.length)[0][0]) - 1
        if ks.size and not ro.completed[r]:
            k = int(ks[0])
            out["crash"][r] = True
            out["term_step"][r] = k
            out["end"][r] = k
            out["term_s"][r] = float(ro.s[k, r] % ro.length)
            out["term_lap"][r] = int(ro.s[k, r] // ro.length) + 1
            out["term_n"][r] = float(ro.n[k, r])
            out["term_vx"][r] = float(ro.vx[k, r])
            out["cause"][r] = ("corridor" if ro.off[k, r] else "slip" if ro.slip[k, r] else
                               "spin" if ro.spun[k, r] else "stall" if ro.stall[k, r] else
                               "nonfinite")
        elif not ro.completed[r]:
            out["timeout"][r] = True
    return out


def trace_arrays(ro: LapRollout, oc: dict, track: Track) -> dict[str, np.ndarray]:
    """D3's trace of one circuit: s, n, v_x and the environment's widths at every sample up to
    the last run's end, with each run's end and outcome."""
    T_end = int(oc["end"].max()) + 1
    s = ro.s[:T_end]
    tr = track_arrays(track)
    wl, wr = jax.jit(jax.vmap(lambda x: widths_at(x, tr)))(jnp.asarray(s.ravel()))
    return {"s": s, "n": ro.n[:T_end], "vx": ro.vx[:T_end],
            "wl": np.asarray(wl).reshape(s.shape), "wr": np.asarray(wr).reshape(s.shape),
            "end": oc["end"].astype(np.int64), "crash": oc["crash"],
            "completed": ro.completed, "length": np.array(ro.length)}


def evaluate(policy_fn, circuits=tuple(R_DEV), runs: int = N_RUNS, traces: bool = False):
    """(rows, traces, inputs): ``scripts/eval_rdev.py``'s evaluation of one policy, with every
    run's outcome. Every name passes the lockbox guard before any rollout."""
    for name in circuits:
        check_circuit(name)
    rows, trs, inputs = {}, {}, {}
    for name in circuits:
        track = circuit(name)
        v_start = float(reference_speed(track)[0]) * START_FRACTION
        ro = rollout_laps(policy_fn, track, v_start, runs=runs, laps=LAPS)
        oc = outcomes(ro)
        t_ocp, sha = ocp_lap(name)
        inputs[rel(OCP_DIR / f"{name}.pkl")] = sha
        rows[name] = {
            "completion": ro.completion_rate, "median_lap_s": ro.median_lap,
            "ocp_lap_s": t_ocp, "gap": ro.median_lap / t_ocp - 1.0,
            "completed": ro.completed, "lap_time": ro.lap_time, "s_reached": ro.s_reached,
            "length": ro.length, "v_start": v_start, "max_steps": int(ro.s.shape[0]),
            **{k: oc[k] for k in ("crash", "timeout", "term_step", "term_s", "term_lap",
                                  "term_n", "term_vx", "cause")}}
        if traces:
            trs[name] = trace_arrays(ro, oc, track)
    return rows, trs, inputs


def summary_of(rows: dict) -> tuple[int, float]:
    """(C, c_bar) over the circuits in ``rows``."""
    fr = np.array([rows[n]["completion"] for n in R_DEV if n in rows])
    return int((fr > COMPLETION_MIN).sum()), float(fr.mean())


# ------------------------------------------------------------------------- D1 tasks


def eval_path(name: str, step: int) -> Path:
    return OUT / "evals" / name / f"{step:012d}.json"


def trace_path(name: str) -> Path:
    return OUT / "traces" / f"{name}.npz"


def eval_one(name: str, step: int, out: Path, traces: Path | None = None) -> dict:
    """Evaluate one grid checkpoint, write its JSON (and traces), return the record."""
    t_start = time.time()
    t0 = time.perf_counter()
    path = checkpoint_path(name, step)
    bound = bound_finals()[name]["sha256"] if step == FINAL else None
    d, sha = load_checkpoint(path, bound)
    if int(d["timesteps"]) != step:
        raise CheckFailed(f"{path}: stores {d['timesteps']} steps")
    policy, _, _ = make_policy(d)
    t_load = time.perf_counter() - t0
    rows, trs, inputs = evaluate(policy, traces=traces is not None)
    t_eval = time.perf_counter() - t0
    C, cbar = summary_of(rows)
    net = BY_NAME[name]
    inputs[rel(path)] = sha
    inputs.update(circuit_inputs())
    rec = {"run": name, "arm": net.arm, "variant": net.variant, "timesteps": step,
           "checkpoint": rel(path), "checkpoint_sha256": sha, "bound": bound is not None,
           "protocol_version": PROTOCOL_VERSION,
           "protocol": {"runs": N_RUNS, "seed": 0, "laps": LAPS, "start_fraction": START_FRACTION,
                        "completion_min": COMPLETION_MIN, "car": "default"},
           "C": C, "c_bar": cbar, "circuits": rows,
           "timing": {"load_s": round(t_load, 3), "eval_s": round(t_eval, 3),
                      "since_process_start_s": round(time.time() - _T0, 3)},
           "provenance": provenance(inputs, t_start, argv=[])}
    write_json(out, rec)
    if traces is not None:
        traces.parent.mkdir(parents=True, exist_ok=True)
        flat = {f"{c}__{k}": v for c, tr in trs.items() for k, v in tr.items()}
        tmp = traces.with_name(traces.stem + ".tmp.npz")
        np.savez_compressed(tmp, **flat)
        tmp.replace(traces)
    return rec


def eval_done(name: str, step: int) -> bool:
    p = eval_path(name, step)
    if not p.exists() or (step == FINAL and not trace_path(name).exists()):
        return False
    rec = json.loads(p.read_text())
    return (rec.get("protocol_version") == PROTOCOL_VERSION
            and rec["checkpoint_sha256"] == sha256_file(checkpoint_path(name, step)))


def _worker_init() -> None:
    lower_priority()


def _eval_task(name: str, step: int) -> tuple[str, int, int, float, float]:
    rec = eval_one(name, step, eval_path(name, step),
                   trace_path(name) if step == FINAL else None)
    return name, step, rec["C"], rec["c_bar"], rec["timing"]["eval_s"]


def run_pool(fn, tasks: list[tuple], workers: int) -> list:
    workers = max(1, min(int(workers), MAX_WORKERS))
    out = []
    if not tasks:
        return out
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_worker_init) as ex:
        futs = {ex.submit(fn, *t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            out.append(res)
            print(f"[{i}/{len(tasks)}] {res}", flush=True)
    return out


# ------------------------------------------------------------------------- D0


def stage_time(args) -> None:
    """D0: the timed evaluation alone, then as one of 4 concurrent processes (each nice 15,
    2 XLA threads). t4 is the longest of the 4 evaluations' wall times (loading the
    checkpoint to the 12th circuit, compile included); W = 168 t4 / 4."""
    start = time.time()
    name, step = TIMED
    tdir = OUT / "timing"
    tdir.mkdir(parents=True, exist_ok=True)

    def cmd(out: Path) -> list[str]:
        return [sys.executable, str(SCRIPT), "eval-one", "--run", name, "--step", str(step),
                "--out", str(out), "--threads", "2"]

    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    t0 = time.perf_counter()
    subprocess.run(cmd(eval_path(name, step)), cwd=ROOT, env=env, check=True)
    alone_wall = time.perf_counter() - t0
    alone = json.loads(eval_path(name, step).read_text())

    t0 = time.perf_counter()
    procs, starts = [], []
    for i in range(MAX_WORKERS):
        starts.append(time.perf_counter())
        procs.append(subprocess.Popen(cmd(tdir / f"concurrent-{i}.json"), cwd=ROOT, env=env))
    walls = []
    for p, s0 in zip(procs, starts, strict=True):
        if p.wait() != 0:
            raise CheckFailed("a concurrent timing process failed")
        walls.append(time.perf_counter() - s0)
    conc = [json.loads((tdir / f"concurrent-{i}.json").read_text()) for i in range(MAX_WORKERS)]
    same = all(np.array_equal(np.asarray(c["circuits"][k]["completed"]),
                              np.asarray(alone["circuits"][k]["completed"]))
               and np.array_equal(np.asarray(c["circuits"][k]["lap_time"], float),
                                  np.asarray(alone["circuits"][k]["lap_time"], float),
                                  equal_nan=True)
               for c in conc for k in R_DEV)
    t1 = alone["timing"]["eval_s"]
    t4 = max(c["timing"]["eval_s"] for c in conc)
    W = N_EVAL * t4 / MAX_WORKERS
    rec = {
        "label": "D0 (exploratory): timing only; the rule uses seconds, never a completion",
        "evaluation": {"run": name, "timesteps": step},
        "alone": {"eval_s": t1, "load_s": alone["timing"]["load_s"],
                  "process_wall_s": round(alone_wall, 3)},
        "concurrent": [{"eval_s": c["timing"]["eval_s"], "load_s": c["timing"]["load_s"],
                        "process_wall_s": round(w, 3)} for c, w in zip(conc, walls, strict=True)],
        "t1_s": t1, "t4_s": t4,
        "t4_mean_s": float(np.mean([c["timing"]["eval_s"] for c in conc])),
        "t4_process_max_s": round(max(walls), 3),
        "n_evaluations": N_EVAL, "W_s": W, "W_h": W / 3600.0,
        "W_h_with_process_start": N_EVAL * max(walls) / MAX_WORKERS / 3600.0,
        "rule": f"W <= {W_MAC_MAX_H:g} h: Mac; otherwise one A6000",
        "decision": "mac" if W / 3600.0 <= W_MAC_MAX_H else "a6000",
        "concurrent_results_identical_to_alone": bool(same),
        "provenance": provenance({rel(eval_path(name, step)): sha256_file(eval_path(name, step))},
                                 start),
    }
    write_json(OUT / "timing.json", rec)
    print(json.dumps({k: rec[k] for k in ("t1_s", "t4_s", "W_h", "decision",
                                          "concurrent_results_identical_to_alone")}))


# ------------------------------------------------------------------------- D1 summaries


def boot_mean(x, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Mean with a 95% percentile bootstrap CI over the units of ``x``."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": len(x)}


def late_label(lo: float, hi: float, band: float = FLAT_BAND) -> dict:
    """The plan's late-curve labels (Section 4), every one that holds, and the reading's
    label: a directional label (rising, falling) before flat (amendment 1)."""
    holds = {"rising": lo > 0.0, "flat": lo >= -band and hi <= band, "falling": hi < 0.0}
    label = ("rising" if holds["rising"] else "falling" if holds["falling"] else
             "flat" if holds["flat"] else "unresolved")
    return {"label": label, "holds": [k for k, v in holds.items() if v] or ["unresolved"]}


def change_summary(d) -> dict:
    d = np.asarray(d, float)
    return {**boot_mean(d), "median": float(np.median(d)),
            "rise": int((d > 0).sum()), "level": int((d == 0).sum()), "fall": int((d < 0).sum())}


def best_checkpoint(C_by_step: dict[int, int]) -> dict:
    """Best C of the six; a tie with the final counts as the final."""
    final = C_by_step[FINAL]
    best = max(C_by_step.values())
    at = min(s for s, c in C_by_step.items() if c == best)
    return {"best": best, "final": final, "best_minus_final": best - final,
            "best_before_final": best > final, "best_step": FINAL if best == final else at}


def calls(fracs) -> np.ndarray:
    return np.asarray(fracs, float) > COMPLETION_MIN


def platform_check(stored: dict[str, dict], mac: dict[str, dict]) -> dict:
    """Mac re-evaluation of each final against the stored end-of-training evaluation."""
    per, dis = {}, []
    agree = total = 0
    for name in stored:
        st = np.array([stored[name][c]["completion"] for c in R_DEV])
        mc = np.array([mac[name][c]["completion"] for c in R_DEV])
        a = calls(st) == calls(mc)
        agree += int(a.sum())
        total += len(a)
        dC = int(calls(mc).sum() - calls(st).sum())
        per[name] = {"C_stored": int(calls(st).sum()), "C_mac": int(calls(mc).sum()), "dC": dC,
                     "calls_agree": int(a.sum()), "fractions_equal": int((st == mc).sum())}
        for c, ok, fs, fm in zip(R_DEV, a, st, mc, strict=True):
            if not ok:
                dis.append({"run": name, "circuit": c, "stored": float(fs), "mac": float(fm)})
    dCs = np.array([v["dC"] for v in per.values()])
    return {"calls_agree": agree, "calls_total": total, "share": agree / total,
            "networks_C_within_1": int((np.abs(dCs) <= 1).sum()),
            "networks_C_equal": int((dCs == 0).sum()), "n_networks": len(per),
            "per_network": per, "disagreements": dis}


def load_evals(steps=GRID_STEPS) -> dict[tuple[str, int], dict]:
    out = {}
    for net in NETWORKS:
        for step in steps:
            p = eval_path(net.name, step)
            if not p.exists():
                raise CheckFailed(f"{rel(p)} is missing: run the curves stage")
            out[net.name, step] = json.loads(p.read_text())
    return out


def stored_evals() -> tuple[dict[str, dict], dict[str, str]]:
    bound = bound_finals()
    out, shas = {}, {}
    for net in NETWORKS:
        p = ROOT / "runs" / net.name / "eval_rdev.json"
        sha = sha256_file(p)
        want = bound[net.name]["eval_sha256"]
        if want is not None and sha != want:
            raise CheckFailed(f"{rel(p)}: sha256 {sha} is not the manifest's {want}")
        out[net.name] = json.loads(p.read_text())
        shas[rel(p)] = sha
    return out, shas


def curves_summary(ev: dict[tuple[str, int], dict]) -> dict:
    C = {n.name: {s: ev[n.name, s]["C"] for s in GRID_STEPS} for n in NETWORKS}
    cb = {n.name: {s: ev[n.name, s]["c_bar"] for s in GRID_STEPS} for n in NETWORKS}
    per_arm = {}
    for arm in ARMS:
        names = [n.name for n in NETWORKS if n.arm == arm]
        per_arm[arm] = {str(s): {
            "C_mean": float(np.mean([C[k][s] for k in names])),
            "C_min": int(min(C[k][s] for k in names)), "C_max": int(max(C[k][s] for k in names)),
            "c_bar_mean": float(np.mean([cb[k][s] for k in names])),
            "c_bar_min": float(min(cb[k][s] for k in names)),
            "c_bar_max": float(max(cb[k][s] for k in names)), "n": len(names)}
            for s in GRID_STEPS}
    per_variant = {v: {str(s): {
        "C": per_arm[f"{v}-realw"][str(s)]["C_mean"] - per_arm[f"{v}-base"][str(s)]["C_mean"],
        "c_bar": (per_arm[f"{v}-realw"][str(s)]["c_bar_mean"]
                  - per_arm[f"{v}-base"][str(s)]["c_bar_mean"])} for s in GRID_STEPS}
        for v in VARIANTS}
    all_mean = {str(s): {"C_mean": float(np.mean([C[k][s] for k in C])),
                         "c_bar_mean": float(np.mean([cb[k][s] for k in cb]))}
                for s in GRID_STEPS}
    names = [n.name for n in NETWORKS]
    d_late_C = [C[k][FINAL] - C[k][LATE_FROM] for k in names]
    d_mid_C = [C[k][FINAL] - C[k][MID_FROM] for k in names]
    d_late_cb = [cb[k][FINAL] - cb[k][LATE_FROM] for k in names]
    d_mid_cb = [cb[k][FINAL] - cb[k][MID_FROM] for k in names]
    changes = {"d_late_C": change_summary(d_late_C), "d_mid_C": change_summary(d_mid_C),
               "d_late_c_bar": change_summary(d_late_cb), "d_mid_c_bar": change_summary(d_mid_cb)}
    label = late_label(changes["d_late_C"]["lo"], changes["d_late_C"]["hi"])
    best = {k: best_checkpoint(C[k]) for k in names}
    n_before = sum(b["best_before_final"] for b in best.values())
    per_circuit = {k: {str(s): {c: {"completion": ev[k, s]["circuits"][c]["completion"],
                                    "gap": ev[k, s]["circuits"][c]["gap"]} for c in R_DEV}
                       for s in GRID_STEPS} for k in names}
    m0, m3, m5 = (all_mean[str(GRID_STEPS[i])]["C_mean"] for i in (0, 3, 5))
    rise = m5 - m0
    readings = {
        "E1": {
            "C_mean_all_by_step": {s: all_mean[str(s)]["C_mean"] for s in GRID_STEPS},
            "rise_10M_to_final": rise,
            "share_of_rise_by_60M": (m3 - m0) / rise if rise else None,
            "late_label": label["label"], "late_labels_holding": label["holds"],
            "median_d_late_C": changes["d_late_C"]["median"],
            "median_abs_d_late_C": float(np.median(np.abs(d_late_C))),
            "v1_realw_minus_base_C_by_step": {s: per_variant["v1"][str(s)]["C"]
                                              for s in GRID_STEPS},
            "share_best_before_final": n_before / len(names),
            "expected": ("C rises with log steps, most of the rise before 60.3M; late 'flat' or "
                         "'unresolved' (median network moves <= 1 circuit); v1 real-width "
                         "advantage visible by 60.3M; >= 1/3 best before the final")},
    }
    return {"C": C, "c_bar": cb, "per_arm": per_arm, "per_variant_realw_minus_base": per_variant,
            "all_networks": all_mean, "changes": changes, "late": label,
            "best_checkpoint": {"per_network": best, "n_best_before_final": n_before,
                                "share": n_before / len(names),
                                "best_minus_final_mean": float(np.mean(
                                    [b["best_minus_final"] for b in best.values()]))},
            "per_circuit": per_circuit, "readings": readings}


def stage_curves(args) -> None:
    start = time.time()
    for net in NETWORKS:
        checkpoint_paths(net.name)
    bound_finals()
    tasks = [(n.name, s) for n in NETWORKS for s in GRID_STEPS if not eval_done(n.name, s)]
    print(f"{N_EVAL - len(tasks)} of {N_EVAL} evaluations on disk; {len(tasks)} to run "
          f"({min(args.workers, MAX_WORKERS)} workers)", flush=True)
    run_pool(_eval_task, tasks, args.workers)
    ev = load_evals()
    stored, stored_sha = stored_evals()
    inputs = dict(stored_sha)
    for (name, step), rec in ev.items():
        inputs[rec["checkpoint"]] = rec["checkpoint_sha256"]
        inputs[rel(eval_path(name, step))] = sha256_file(eval_path(name, step))
    for net in NETWORKS:
        inputs[f"runs/{net.name}/config.json"] = sha256_file(ROOT / "runs" / net.name
                                                             / "config.json")
    inputs[rel(MANIFEST)] = sha256_file(MANIFEST)
    inputs[rel(PREREG)] = sha256_file(PREREG)
    inputs.update(circuit_inputs())
    machines = sorted({"{system}-{machine}".format(**rec["provenance"]["environment"])
                       for rec in ev.values()})
    summ = curves_summary(ev)
    mac_final = {n.name: ev[n.name, FINAL]["circuits"] for n in NETWORKS}
    plat = platform_check(stored, mac_final)
    summ["platform_check"] = plat
    summ["readings"]["E2"] = {
        "calls_agree_share": plat["share"], "calls_agree": plat["calls_agree"],
        "networks_C_within_1": plat["networks_C_within_1"],
        "networks_C_equal": plat["networks_C_equal"],
        "disagreements": plat["disagreements"],
        "expected": (">= 90% of 336 calls agree; C within +-1 for most networks; "
                     "disagreements near half completion")}
    rec = {"label": "D1 (exploratory)", "grid_steps": GRID_STEPS, "machines": machines,
           **summ, "provenance": provenance(inputs, start)}
    write_json(OUT / "curves.json", rec)
    r = summ["readings"]["E1"]
    print(json.dumps({"late": summ["late"], "d_late_C": summ["changes"]["d_late_C"],
                      "C_mean_by_step": r["C_mean_all_by_step"],
                      "platform_share": plat["share"]}, indent=1))


# ------------------------------------------------------------------------- D2

_LEVELS: dict = {}


def adr_levels():
    """(levels, tracks, x0) of the 1,000 A_DR levels, as scripts/pilot_audit.py builds them."""
    if "lv" not in _LEVELS:
        for name in R_DEV:
            check_circuit(name)
        dev = [load_tumftm(TRACK_DIR / f"{name}.csv") for name in R_DEV]
        g = fit_grammar(dev)
        lv = make_levels(jax.random.PRNGKey(ADR_ROOT_SEED), "a_dr")
        tracks, _, x0 = build(lv, g)
        _LEVELS.update(lv=lv, tracks=tracks, x0=x0)
    return _LEVELS["lv"], _LEVELS["tracks"], _LEVELS["x0"]


def make_det_window(cfg: PPOConfig, steps: int = 1200,
                    params: racing.EnvParams = racing.EnvParams()):
    """One deterministic window from a level's start: the mean action, no observation noise
    (``audit.rollouts.make_window_fn`` otherwise). Returns (crashed, truncated, cause flags at
    the crash, distance, crash step, first truncation step)."""
    actor = Actor(cfg, racing.ACTION_DIM, False)
    params = params._replace(window_steps=steps)

    def one(snap, tr, x0_l, ctx_l, mu_l):
        st = racing.EnvState(x=x0_l, prev_action=jnp.zeros(2), accel=jnp.zeros(2), ctx=ctx_l,
                             hidden_mu=mu_l, t=jnp.zeros((), jnp.int32), slow=jnp.zeros(()),
                             s0=x0_l[0])
        obs = racing.observe(st, tr, racing.params_from_context(ctx_l, hidden_mu=mu_l))

        def body(c, _):
            st, ob, alive = c
            mean, _ = actor.apply(snap.actor, snap.obs_rms.normalize(ob))
            ob2, st2, _, term, trunc, info = racing.step_no_reset(st, mean, tr, params)
            alive2 = alive & ~term & ~trunc
            st2 = jax.tree.map(lambda n, old: jnp.where(alive, n, old), st2, st)
            ob2 = jnp.where(alive, ob2, ob)
            return (st2, ob2, alive2), (term & alive, trunc & alive, info["off_track"],
                                        info["slipped"], info["spun"], info["stalled"])

        (stT, _, _), (term, trunc, off, slip, spun, stall) = jax.lax.scan(
            body, (st, obs, jnp.array(True)), None, steps)
        k = jnp.argmax(term)
        return (term.any(), trunc.any(), jnp.stack([off[k], slip[k], spun[k], stall[k]]),
                stT.x[0] - x0_l[0], k, jnp.argmax(trunc))

    return one


def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def adr_path(name: str) -> Path:
    return OUT / "adr" / f"{name}.json"


def adr_one(name: str) -> dict:
    t_start = time.time()
    t0 = time.perf_counter()
    path = checkpoint_path(name, FINAL)
    d, sha = load_checkpoint(path, bound_finals()[name]["sha256"])
    _, cfg, snap = make_policy(d)
    lv, tracks, x0 = adr_levels()
    L = int(x0.shape[0])
    one = jax.jit(jax.vmap(make_det_window(cfg), in_axes=(None, 0, 0, 0, 0)))
    parts = []
    for a in range(0, L, ADR_CHUNK):
        sl = slice(a, min(a + ADR_CHUNK, L))
        tr = jax.tree.map(lambda v, sl=sl: v[sl], tracks)
        parts.append(jax.tree.map(np.asarray, one(snap, tr, x0[sl], lv.ctx[sl], lv.hidden_mu[sl])))
    crashed, trunc, flags, dist, kstep, tstep = (np.concatenate([p[i] for p in parts])
                                                 for i in range(6))
    t_det = time.perf_counter() - t0
    cause = np.where(~crashed, "", np.where(flags[:, 0], "corridor", np.where(
        flags[:, 1], "slip", np.where(flags[:, 2], "spin", np.where(flags[:, 3], "stall",
                                                                    "nonfinite")))))
    key = jax.random.fold_in(jax.random.PRNGKey(ADR_ROOT_SEED), ADR_KEY_FOLD)
    ro = rollout_windows(snap, cfg, tracks, x0, lv.ctx, lv.hidden_mu, ADR_WINDOWS, key)
    t_all = time.perf_counter() - t0
    k_det = int((~crashed).sum())
    win_done = ~np.asarray(ro.terminated, bool)
    rec = {"run": name, "arm": BY_NAME[name].arm, "variant": BY_NAME[name].variant,
           "checkpoint": rel(path), "checkpoint_sha256": sha, "n_levels": L,
           "deterministic": {"completed": k_det, "share": k_det / L, "wilson95": wilson(k_det, L),
                             "truncated_before_window_end": int((trunc & (tstep < 1199)).sum()),
                             "causes": {c: int((cause == c).sum()) for c in (*CAUSES, "nonfinite")},
                             "per_level_completed": (~crashed).astype(int),
                             "crash_step": np.where(crashed, kstep, -1),
                             "distance_m": np.round(dist, 1)},
           "stochastic": {"windows": int(win_done.size), "completed": int(win_done.sum()),
                          "share": float(win_done.mean()),
                          "phase": "late" if win_done.mean() >= LATE_PHASE else "not late",
                          "per_level_completed_windows": win_done.sum(axis=1)},
           "timing": {"deterministic_s": round(t_det, 2), "total_s": round(t_all, 2)},
           "provenance": provenance({rel(path): sha}, t_start, argv=[])}
    write_json(adr_path(name), rec)
    return rec


def _adr_task(name: str) -> tuple[str, float, float, float]:
    rec = adr_one(name)
    return (name, rec["deterministic"]["share"], rec["stochastic"]["share"],
            rec["timing"]["total_s"])


def adr_summary(adr: dict[str, dict], cbar: dict[str, float]) -> dict:
    per = {k: {"arm": v["arm"], "det": v["deterministic"]["share"],
               "det_wilson95": v["deterministic"]["wilson95"],
               "stoch": v["stochastic"]["share"], "stoch_phase": v["stochastic"]["phase"],
               "rdev_c_bar": cbar[k], "det_minus_c_bar": v["deterministic"]["share"] - cbar[k]}
           for k, v in adr.items()}
    arms = {}
    for arm in ARMS:
        ks = [k for k in per if per[k]["arm"] == arm]
        arms[arm] = {m: {"mean": float(np.mean([per[k][m] for k in ks])),
                         "median": float(np.median([per[k][m] for k in ks])),
                         "min": float(min(per[k][m] for k in ks)),
                         "max": float(max(per[k][m] for k in ks))}
                     for m in ("det", "stoch", "rdev_c_bar")}
        arms[arm]["n"] = len(ks)
    allk = list(per)
    med_det = float(np.median([per[k]["det"] for k in allk]))
    med_cb = float(np.median([per[k]["rdev_c_bar"] for k in allk]))
    return {"per_network": per, "per_arm": arms,
            "all": {"det_median": med_det, "rdev_c_bar_median": med_cb,
                    "stoch_median": float(np.median([per[k]["stoch"] for k in allk]))},
            "readings": {"E3": {
                "arm_median_det": {a: arms[a]["det"]["median"] for a in ARMS},
                "every_arm_median_det_ge_0.90": all(arms[a]["det"]["median"] >= 0.9
                                                    for a in ARMS),
                "arm_median_rdev_c_bar": {a: arms[a]["rdev_c_bar"]["median"] for a in ARMS},
                "arm_mean_stoch": {a: arms[a]["stoch"]["mean"] for a in ARMS},
                "n_networks_det_gt_c_bar": int(sum(per[k]["det_minus_c_bar"] > 0 for k in allk)),
                "section8_row": ("median A_DR >= 0.90 while R-dev c_bar <= 0.7"
                                 if med_det >= 0.9 and med_cb <= 0.7 else
                                 "median A_DR < 0.90" if med_det < 0.9 else
                                 "median A_DR >= 0.90 and R-dev c_bar > 0.7 (no row)"),
                "expected": ("deterministic >= 0.90 for the median network in every arm, well "
                             "above its R-dev c_bar; stochastic lower, about 0.8-0.9")}}}


def stage_adr(args) -> None:
    start = time.time()
    bound_finals()
    tasks = [(n.name,) for n in NETWORKS
             if not (adr_path(n.name).exists() and json.loads(adr_path(n.name).read_text())
                     ["checkpoint_sha256"] == bound_finals()[n.name]["sha256"])]
    print(f"{len(NETWORKS) - len(tasks)} of {len(NETWORKS)} on disk; {len(tasks)} to run",
          flush=True)
    run_pool(_adr_task, tasks, args.workers)
    adr = {n.name: json.loads(adr_path(n.name).read_text()) for n in NETWORKS}
    cbar = {}
    for n in NETWORKS:
        p = eval_path(n.name, FINAL)
        if not p.exists():
            raise CheckFailed(f"{rel(p)} is missing: run the curves stage first")
        cbar[n.name] = json.loads(p.read_text())["c_bar"]
    inputs = {rel(adr_path(k)): sha256_file(adr_path(k)) for k in adr}
    inputs.update({rel(eval_path(k, FINAL)): sha256_file(eval_path(k, FINAL)) for k in adr})
    inputs.update({v["checkpoint"]: v["checkpoint_sha256"] for v in adr.values()})
    inputs.update({rel(TRACK_DIR / f"{c}.csv"): sha256_file(TRACK_DIR / f"{c}.csv") for c in R_DEV})
    rec = {"label": "D2 (exploratory)",
           "levels": {"root": f"PRNGKey({ADR_ROOT_SEED})", "stream": "a_dr", "n": 1000,
                      "grammar": "fit_grammar(R-dev)", "window_steps": 1200,
                      "stochastic": {"windows": ADR_WINDOWS, "key": "fold_in(PRNGKey(2026), 1)",
                                     "obs_noise": 0.01}},
           **adr_summary(adr, cbar), "provenance": provenance(inputs, start)}
    write_json(OUT / "adr.json", rec)
    print(json.dumps(rec["readings"], indent=1))


# ------------------------------------------------------------------------- D3: geometry


@dataclasses.dataclass(frozen=True)
class Window:
    """Corner k's window W_k = [s_start - pad, s_end + pad] on a closed track of length L, in
    cyclic arc length (a corner may cross s = 0)."""
    circuit: str
    k: int
    sigma: int
    s_start: float
    span: float          # (s_end - s_start) mod L
    length: float        # L
    pad: float = WINDOW_PAD

    @property
    def a(self) -> float:
        return self.s_start - self.pad

    @property
    def width(self) -> float:
        return self.span + 2.0 * self.pad

    def contains(self, s) -> np.ndarray:
        return np.mod(np.asarray(s, float) - self.a, self.length) <= self.width

    def core_distance(self, s) -> np.ndarray:
        """Distance (m) from s to the corner proper [s_start, s_end] (0 inside), cyclic."""
        d = np.mod(np.asarray(s, float) - self.s_start, self.length)
        after = d - self.span
        before = self.length - d
        return np.where(d <= self.span, 0.0, np.minimum(after, before))


def corner_windows(name: str, track: Track) -> list[Window]:
    seg = segment(track)
    L = float(track.length)
    return [Window(name, k, int(c.sign), float(c.s_start), float((c.s_end - c.s_start) % L), L)
            for k, c in enumerate(seg.corners)]


def assign_corner(s_mod: float, wins: list[Window]) -> int | None:
    """The corner whose window contains s (amendment 1: the nearest corner proper when windows
    overlap, the earlier corner on a tie); None on a straight."""
    best, best_d = None, np.inf
    for w in wins:
        if w.contains(s_mod):
            d = float(w.core_distance(s_mod))
            if d < best_d:
                best, best_d = w.k, d
    return best


def clearance(wl, wr, n, sigma: int, margin: float = EDGE_MARGIN) -> np.ndarray:
    """c = w_in - margin - sigma n, w_in = w_left for sigma = +1 and w_right for sigma = -1."""
    w_in = np.asarray(wl, float) if sigma > 0 else np.asarray(wr, float)
    return w_in - margin - sigma * np.asarray(n, float)


def side_of(n: float, sigma: int | None) -> str:
    if sigma is None:
        return "straight"
    return "inside" if sigma * n > 0 else "outside"


def run_passages(s, n, wl, wr, end: int, crash: bool, win: Window, laps: int = LAPS) -> list[dict]:
    """One run's passages through ``win``: maximal stretches of consecutive samples up to the
    run's end whose s (mod L) lies in the window, before the end of lap 2 (the terminating
    sample of a crash always counts). Each: entry/exit sample, lap at entry, c_k (minimum
    clearance), and whether the run terminates in it."""
    s = np.asarray(s, float)[: end + 1]
    idx = np.arange(len(s))
    valid = (s < laps * win.length) | ((idx == end) & crash)
    m = win.contains(s) & valid
    if not m.any():
        return []
    c = clearance(np.asarray(wl)[: end + 1], np.asarray(wr)[: end + 1],
                  np.asarray(n)[: end + 1], win.sigma)
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts, stops = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
    return [{"i0": int(a), "i1": int(b), "lap": int(s[a] // win.length) + 1,
             "c": float(c[a: b + 1].min()), "terminated": bool(crash and b == end)}
            for a, b in zip(starts, stops, strict=True)]


def former_crash_corners(counts: dict[tuple[str, int], int], k_min: int = FORMER_MIN) -> list:
    """Corners where at least ``k_min`` of the baseline arm's runs terminate."""
    return sorted(key for key, v in counts.items() if v >= k_min)


def gain_circuits(base_calls: dict[str, list[bool]], realw_calls: dict[str, list[bool]]) -> list:
    """Circuits that more real-width than baseline networks complete."""
    return [c for c in R_DEV if sum(realw_calls[c]) > sum(base_calls[c])]


def boot_corner_diff(base: np.ndarray, realw: np.ndarray, n_boot: int = N_BOOT,
                     seed: int = BOOT_SEED) -> dict:
    """dc: the mean over corners of (real-width arm mean - baseline arm mean) of per-network
    medians [networks, corners] (NaN where a network has no passage). The CI resamples
    networks within each arm, corners fixed; a corner undefined in a draw is left out of that
    draw (amendment 1)."""
    base, realw = np.asarray(base, float), np.asarray(realw, float)
    if base.shape[1] == 0:
        return {"dc": None, "lo": None, "hi": None, "n_corners": 0, "draws_with_dropped": 0}

    def stat(b, r):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.nanmean(r, axis=-2) - np.nanmean(b, axis=-2)
            return np.nanmean(d, axis=-1), np.isnan(d).any(axis=-1)

    point, _ = stat(base, realw)
    rng = np.random.default_rng(seed)
    nb, nr = base.shape[0], realw.shape[0]
    ib = rng.integers(0, nb, (n_boot, nb))
    ir = rng.integers(0, nr, (n_boot, nr))
    vals, dropped = [], 0
    for a in range(0, n_boot, 2000):
        v, dr = stat(base[ib[a:a + 2000]], realw[ir[a:a + 2000]])
        vals.append(v)
        dropped += int(dr.sum())
    vals = np.concatenate(vals)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return {"dc": float(point), "lo": float(lo), "hi": float(hi), "n_corners": int(base.shape[1]),
            "draws_with_dropped": dropped}


# ------------------------------------------------------------------------- D3: the stage


def load_traces(name: str) -> dict[str, dict[str, np.ndarray]]:
    p = trace_path(name)
    if not p.exists():
        raise CheckFailed(f"{rel(p)} is missing: run the curves stage")
    z = np.load(p, allow_pickle=False)
    out: dict[str, dict] = {}
    for key in z.files:
        c, k = key.split("__", 1)
        check_circuit(c)
        out.setdefault(c, {})[k] = z[key]
    return out


def network_crash_data(name: str, wins: dict[str, list[Window]]) -> dict:
    """Per circuit: the crash census and every run's passages through every corner window."""
    trs = load_traces(name)
    ev = json.loads(eval_path(name, FINAL).read_text())["circuits"]
    out = {}
    for c in R_DEV:
        tr = trs[c]
        crashes, passages = [], {w.k: [] for w in wins[c]}
        R = tr["s"].shape[1]
        for r in range(R):
            end, crash = int(tr["end"][r]), bool(tr["crash"][r])
            if crash != bool(ev[c]["crash"][r]):
                raise CheckFailed(f"{name} {c} run {r}: trace and evaluation disagree")
            for w in wins[c]:
                for p in run_passages(tr["s"][:, r], tr["n"][:, r], tr["wl"][:, r],
                                      tr["wr"][:, r], end, crash, w):
                    passages[w.k].append({**p, "run": r})
            if crash:
                s_mod = float(ev[c]["term_s"][r])
                k = assign_corner(s_mod, wins[c])
                sigma = None if k is None else wins[c][k].sigma
                crashes.append({"run": r, "s": s_mod, "lap": int(ev[c]["term_lap"][r]),
                                "n": float(ev[c]["term_n"][r]), "vx": float(ev[c]["term_vx"][r]),
                                "cause": ev[c]["cause"][r], "corner": k,
                                "side": side_of(float(ev[c]["term_n"][r]), sigma)})
        out[c] = {"crashes": crashes, "passages": passages,
                  "completion": ev[c]["completion"]}
    return out


def is_inside_corridor(cr: dict) -> bool:
    return cr["cause"] == "corridor" and cr["side"] == "inside"


def network_medians(data: dict, names: list[str], corners: list[tuple[str, int]],
                    clear_only: bool = False) -> np.ndarray:
    """[networks, corners] median c_k over each network's passages (NaN if none)."""
    M = np.full((len(names), len(corners)), np.nan)
    for i, nm in enumerate(names):
        for j, (c, k) in enumerate(corners):
            v = [p["c"] for p in data[nm][c]["passages"][k]
                 if not (clear_only and p["terminated"])]
            if v:
                M[i, j] = float(np.median(v))
    return M


def crashes_analysis(data: dict, wins: dict[str, list[Window]], stored: dict[str, dict]) -> dict:
    res = {"variants": {}}
    census_all = {}
    for v in VARIANTS:
        base = [n.name for n in NETWORKS if n.arm == f"{v}-base"]
        realw = [n.name for n in NETWORKS if n.arm == f"{v}-realw"]
        counts = {(c, w.k): 0 for c in R_DEV for w in wins[c]}
        for nm in base:
            for c in R_DEV:
                for cr in data[nm][c]["crashes"]:
                    if cr["corner"] is not None:
                        counts[c, cr["corner"]] += 1
        former = former_crash_corners(counts)
        fset = set(former)
        mac_calls = {arm: {c: [data[nm][c]["completion"] > COMPLETION_MIN for nm in arm_names]
                           for c in R_DEV} for arm, arm_names in (("base", base),
                                                                  ("realw", realw))}
        st_calls = {arm: {c: [stored[nm][c]["completion"] > COMPLETION_MIN for nm in arm_names]
                          for c in R_DEV} for arm, arm_names in (("base", base),
                                                                 ("realw", realw))}
        gains = gain_circuits(mac_calls["base"], mac_calls["realw"])
        gains_stored = gain_circuits(st_calls["base"], st_calls["realw"])

        def n_crash(names, circuits, pred=lambda cr, c: True):
            return sum(1 for nm in names for c in circuits for cr in data[nm][c]["crashes"]
                       if pred(cr, c))

        def inside_former(cr, c, fset=fset):
            return is_inside_corridor(cr) and (c, cr["corner"]) in fset

        dT = n_crash(base, gains) - n_crash(realw, gains)
        dT_in = n_crash(base, gains, inside_former) - n_crash(realw, gains, inside_former)
        q1_circuit = {}
        for c in gains:
            d = n_crash(base, [c]) - n_crash(realw, [c])
            di = n_crash(base, [c], inside_former) - n_crash(realw, [c], inside_former)
            q1_circuit[c] = {"dT": d, "dT_in": di, "s_in": di / d if d > 0 else None,
                             "base_terminations": n_crash(base, [c]),
                             "realw_terminations": n_crash(realw, [c])}
        rates = {}
        for c, k in former:
            row = {"sigma": wins[c][k].sigma, "baseline_terminations": counts[c, k]}
            for arm, names in (("base", base), ("realw", realw)):
                npass = sum(len(data[nm][c]["passages"][k]) for nm in names)
                nterm = sum(1 for nm in names for cr in data[nm][c]["crashes"]
                            if cr["corner"] == k)
                nin = sum(1 for nm in names for cr in data[nm][c]["crashes"]
                          if cr["corner"] == k and is_inside_corridor(cr))
                row[arm] = {"passages": npass, "terminations": nterm, "inside_corridor": nin,
                            "rate": nterm / npass if npass else None}
            rates[f"{c}#{k}"] = row
        # Census of every baseline (and real-width) termination.
        census = {}
        for arm, names in (("base", base), ("realw", realw)):
            crs = [(c, cr) for nm in names for c in R_DEV for cr in data[nm][c]["crashes"]]
            n = len(crs)
            census[arm] = {
                "terminations": n,
                "inside_corridor_at_corner": sum(is_inside_corridor(cr) and cr["corner"] is not None
                                                 for _, cr in crs),
                "inside_corridor_at_former": sum(inside_former(cr, c) for c, cr in crs),
                "by_cause": {k: sum(cr["cause"] == k for _, cr in crs)
                             for k in (*CAUSES, "nonfinite")},
                "by_side": {k: sum(cr["side"] == k for _, cr in crs)
                            for k in ("inside", "outside", "straight")},
                "by_lap": {str(k): sum(cr["lap"] == k for _, cr in crs) for k in (1, 2, 3)}}
            census[arm]["share_inside_corridor_at_corner"] = (
                census[arm]["inside_corridor_at_corner"] / n if n else None)
        census_all[v] = census
        # Q2.
        Mb = network_medians(data, base, former)
        Mr = network_medians(data, realw, former)
        q2 = boot_corner_diff(Mb, Mr)
        lo_b, hi_b = BAND[0] * DELTA_B[v], BAND[1] * DELTA_B[v]
        q2.update(band=[lo_b, hi_b], in_band=(q2["dc"] is not None and lo_b <= q2["dc"] <= hi_b),
                  ci_includes_0=(q2["lo"] is not None and q2["lo"] <= 0.0 <= q2["hi"]),
                  delta_B=DELTA_B[v])
        q2_corner = {f"{c}#{k}": {"base_mean": float(np.nanmean(Mb[:, j])) if np.isfinite(
                         Mb[:, j]).any() else None,
                         "realw_mean": float(np.nanmean(Mr[:, j])) if np.isfinite(
                         Mr[:, j]).any() else None,
                         "base_networks": Mb[:, j], "realw_networks": Mr[:, j]}
                     for j, (c, k) in enumerate(former)}
        q2_clear = boot_corner_diff(network_medians(data, base, former, clear_only=True),
                                    network_medians(data, realw, former, clear_only=True))
        all_c = [(c, w.k) for c in R_DEV for w in wins[c]]
        q2_all = boot_corner_diff(network_medians(data, base, all_c),
                                  network_medians(data, realw, all_c))
        q2_circuit = {}
        for c in R_DEV:
            cc = [ck for ck in former if ck[0] == c]
            if cc:
                q2_circuit[c] = boot_corner_diff(network_medians(data, base, cc),
                                                 network_medians(data, realw, cc))
        s_in = dT_in / dT if dT > 0 else None
        res["variants"][v] = {
            "former_crash_corners": [f"{c}#{k}" for c, k in former],
            "baseline_terminations_per_corner": {f"{c}#{k}": n for (c, k), n in counts.items()
                                                 if n},
            "gain_circuits": gains, "gain_circuits_stored": gains_stored,
            "completing_networks": {arm: {c: int(sum(mac_calls[arm][c])) for c in R_DEV}
                                    for arm in mac_calls},
            "Q1": {"dT": dT, "dT_in": dT_in, "s_in": s_in,
                   "base_terminations_on_gain": n_crash(base, gains),
                   "realw_terminations_on_gain": n_crash(realw, gains),
                   "per_circuit": q1_circuit},
            "termination_rates_at_former": rates,
            "census": census,
            "Q2": q2, "Q2_per_corner": q2_corner,
            "secondary": {"Q2_clearing_passages_only": q2_clear, "Q2_all_corners": q2_all,
                          "Q2_per_circuit": q2_circuit}}
    v1, v3 = res["variants"]["v1"], res["variants"]["v3"]
    pooled_base = sum(census_all[v]["base"]["terminations"] for v in VARIANTS)
    pooled_in = sum(census_all[v]["base"]["inside_corridor_at_corner"] for v in VARIANTS)
    res["readings"] = {
        "E4": {"baseline_share_inside_corridor_at_corner": {
                   v: census_all[v]["base"]["share_inside_corridor_at_corner"] for v in VARIANTS},
               "baseline_share_pooled": pooled_in / pooled_base if pooled_base else None,
               "s_in": {v: res["variants"][v]["Q1"]["s_in"] for v in VARIANTS},
               "n_gain_circuits": {v: len(res["variants"][v]["gain_circuits"]) for v in VARIANTS},
               "expected": ">= 80% inside-edge corridor exits at corners; v1 s_in >= 2/3"},
        "E5": {"dc": {v: res["variants"][v]["Q2"] for v in VARIANTS},
               "expected": "dc_v1 > 0 and in [0.22, 0.88] m; dc_v3 unresolved (CI includes 0)"},
        "section8_row": section8_d3(v1["Q1"]["s_in"], v1["Q2"], v3["Q1"]["s_in"], v3["Q2"]),
    }
    return res


def section8_d3(s_in_v1, q2_v1, s_in_v3, q2_v3) -> dict:
    """Section 8's D3 rows, read per variant (the margin-consistent row needs v1 at least)."""
    def row(s_in, q2):
        if s_in is None:
            return "s_in undefined (no termination difference on gain circuits)"
        if s_in < 1 / 3:
            return "s_in < 1/3"
        if s_in < 2 / 3:
            return "1/3 <= s_in < 2/3 (no row)"
        lo, hi = q2["band"]
        if q2["dc"] is None:
            return "s_in >= 2/3, dc undefined"
        if q2["dc"] < lo:
            return "s_in >= 2/3 but dc below the band"
        if q2["dc"] > hi:
            return "s_in >= 2/3 but dc above the band"
        return "margin-consistent: s_in >= 2/3 and dc in the band"
    return {"v1": row(s_in_v1, q2_v1), "v3": row(s_in_v3, q2_v3)}


def stage_crashes(args) -> None:
    start = time.time()
    names = [n.name for n in NETWORKS if n.arm in H4R_ARMS]
    wins = {c: corner_windows(c, circuit(c)) for c in R_DEV}
    data = {nm: network_crash_data(nm, wins) for nm in names}
    stored, stored_sha = stored_evals()
    res = crashes_analysis(data, wins, stored)
    inputs = {rel(trace_path(nm)): sha256_file(trace_path(nm)) for nm in names}
    inputs.update({rel(eval_path(nm, FINAL)): sha256_file(eval_path(nm, FINAL)) for nm in names})
    inputs.update({k: v for k, v in stored_sha.items()
                   if any(k == f"runs/{nm}/eval_rdev.json" for nm in names)})
    inputs.update(circuit_inputs())
    corners = {c: [{"k": w.k, "sigma": w.sigma, "s_start": w.s_start,
                    "s_end": (w.s_start + w.span) % w.length, "window": [w.a, w.a + w.width]}
                   for w in wins[c]] for c in R_DEV}
    census = {nm: {c: data[nm][c]["crashes"] for c in R_DEV} for nm in names}
    rec = {"label": "D3 (exploratory)", "networks": names, "corners": corners, **res,
           "crash_census": census, "provenance": provenance(inputs, start)}
    write_json(OUT / "crashes.json", rec)
    print(json.dumps(res["readings"], indent=1, default=str))


# ------------------------------------------------------------------------- report


def descriptive_d1(ev: dict[tuple[str, int], dict]) -> dict:
    """DESCRIPTIVE, added after D1's results were seen (not in the plan): lap gaps by
    checkpoint, circuits completed by checkpoint, and networks at the gate's 10 of 12."""
    names = [n.name for n in NETWORKS]
    gaps, per_circuit, ge10 = {}, {}, {}
    for s in GRID_STEPS:
        g = [ev[k, s]["circuits"][c]["gap"] for k in names for c in R_DEV
             if ev[k, s]["circuits"][c]["completion"] > COMPLETION_MIN]
        gaps[s] = {"median": float(np.median(g)), "n_pairs": len(g)}
        per_circuit[s] = {c: int(sum(ev[k, s]["circuits"][c]["completion"] > COMPLETION_MIN
                                     for k in names)) for c in R_DEV}
        ge10[s] = sorted(k for k in names if ev[k, s]["C"] >= 10)
    first, last = GRID_STEPS[0], FINAL
    paired = [(ev[k, first]["circuits"][c]["gap"], ev[k, last]["circuits"][c]["gap"])
              for k in names for c in R_DEV
              if min(ev[k, first]["circuits"][c]["completion"],
                     ev[k, last]["circuits"][c]["completion"]) > COMPLETION_MIN]
    p = np.array(paired)
    lost = {c: int(sum(ev[k, first]["circuits"][c]["completion"] > COMPLETION_MIN
                       and ev[k, last]["circuits"][c]["completion"] <= COMPLETION_MIN
                       for k in names)) for c in R_DEV}
    gained = {c: int(sum(ev[k, first]["circuits"][c]["completion"] <= COMPLETION_MIN
                         and ev[k, last]["circuits"][c]["completion"] > COMPLETION_MIN
                         for k in names)) for c in R_DEV}
    return {"label": "descriptive (added after D1's results were seen; not in the plan)",
            "median_gap_of_completed_pairs": gaps,
            "networks_completing_each_circuit": per_circuit,
            "networks_C_ge_10": {s: {"n": len(v), "runs": v} for s, v in ge10.items()},
            "networks_C_ge_10_at_any_checkpoint": sorted({k for v in ge10.values() for k in v}),
            "paired_gap_10M_vs_final": {"n_pairs": len(p),
                                        "median_10M": float(np.median(p[:, 0])),
                                        "median_final": float(np.median(p[:, 1])),
                                        "median_change": float(np.median(p[:, 1] - p[:, 0])),
                                        "n_final_faster": int((p[:, 1] < p[:, 0]).sum())},
            "circuits_lost_10M_to_final": lost, "circuits_gained_10M_to_final": gained}


def stage_report(args) -> None:
    if (OUT / "curves.json").exists():
        start = time.time()
        ev = load_evals()
        rec = descriptive_d1(ev)
        rec["provenance"] = provenance({rel(eval_path(k, s)): sha256_file(eval_path(k, s))
                                        for k, s in ev}, start)
        write_json(OUT / "descriptive.json", rec)
    for f in ("timing.json", "curves.json", "adr.json", "crashes.json"):
        p = OUT / f
        if not p.exists():
            print(f"{f}: missing")
            continue
        rec = json.loads(p.read_text())
        print(f"== {f} ({rec.get('label', '')})")
        if f == "timing.json":
            print(json.dumps({k: rec[k] for k in ("t1_s", "t4_s", "W_h", "decision")}))
        else:
            print(json.dumps(rec["readings"], indent=1))


# ------------------------------------------------------------------------- main


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("time", "curves", "adr", "crashes", "report", "eval-one"))
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--threads", type=int, default=2, help="XLA CPU threads per process")
    ap.add_argument("--run", help="eval-one: the run")
    ap.add_argument("--step", type=int, help="eval-one: the checkpoint's timesteps")
    ap.add_argument("--out", type=Path, help="eval-one: the JSON to write")
    ap.add_argument("--traces", type=Path, help="eval-one: also write traces here")
    args = ap.parse_args(argv)
    if args.stage == "eval-one":
        lower_priority()
        rec = eval_one(args.run, args.step, args.out, args.traces)
        print(f"{args.run} @ {args.step}: C {rec['C']}, c_bar {rec['c_bar']:.3f}, "
              f"{rec['timing']['eval_s']:.1f} s", flush=True)
        return
    {"time": stage_time, "curves": stage_curves, "adr": stage_adr, "crashes": stage_crashes,
     "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
