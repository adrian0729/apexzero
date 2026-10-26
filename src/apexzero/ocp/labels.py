"""Paper labels: J*_20 per level, the replay-verified best-known optimum of the agent's own
window problem (two-track plan 2.2, "Targets"; follow-through plan PA-1).

A level is theta = (strip, car context, hidden grip, x0) of a paper level set
(``tracks.levels``). Its label is the discounted return J* of the training-objective OCP in
window mode (``ocp.window``: 1,200 steps at 20 Hz, gamma 0.995, the env's reward), found from
the plan's uniform three-start budget and accepted only if it replays. Per level:

1. **s-OCP.** The level's own minimum-time OCP (``ocp.min_time``) on the part of the strip a
   window can reach (``window.reachable_strip``). It is the time-mapped guess of the ``socp``
   start and the extension every continuation stage needs. If it does not converge it is
   re-solved once from its iterate at tol 1e-5 (``scripts/solve_all_ocp.py``'s rule); a
   finite iterate that still has not converged is used anyway and flagged, because a guess
   decides only where IPOPT starts, never whether a label counts. Without a finite s-OCP the
   ``socp`` start is skipped and every chain extends with the QSS line instead.
2. **Three starts** (``STARTS``), each a horizon continuation (10, 20, 40, 60 s; to-ocp-spike.md):
   - ``socp``: the s-OCP mapped to time (``window.min_time_guess``), extended by it;
   - ``qss``: the QSS line (``qss_line``): the minimum-curvature line inside the corridor
     (``audit.qss_distance``, the d_QSS line) driven at the level's own car's QSS speed
     profile from x0's speed, mapped to time with a kinematic steer and a pedal from the
     longitudinal force balance; extended by itself, so it is an independent start and not
     the s-OCP start with a different first 10 s;
   - ``agent``: the best of one deterministic and ``agent_windows`` stochastic rollouts of a
     checkpoint (``AgentStart``; plan: "the best rollout of the final checkpoint"), by
     discounted return with the env's crash penalty, cut where it terminated and continued
     by the s-OCP mapping; extended by the s-OCP (the spike measured a policy extension at
     ~4x the iterations).
3. **Verification.** ``replay.verify_window``: the solve succeeded (Solve_Succeeded, no
   window bound active, every RK stage in its kappa block) and the controls, replayed
   open-loop in the float64 env with the level's hidden grip, survive every step and
   reproduce J* within ``REPLAY_RTOL`` (1e-6, stricter than the plan's 0.05%). The corridor
   in the solve is tightened by 3 cm (``WindowSpec.safety``), inside the plan's 2-5 cm.
4. **The label** is the highest J* among the verified primary and retry starts (the
   ``replay.best_verified`` rule: a failed start never wins on J).
5. **Retry policy** (``LabelPolicy``; a draft for decision #17). A start stops before its
   next stage once it has spent ``start_cpu_budget_s`` (it then counts as failed, "capped"),
   and each IPOPT call stops at ``call_cpu_cap_s``. Only when *no* primary start verifies,
   each start is retried once, in the order socp, qss, agent, from its own guess, with the
   finer schedule ``retry_stages_s``, ``retry_max_iter`` and ``retry_cpu_budget_s``; the first
   retry that verifies ends the retries (its level is flagged ``retried``). A level with no
   verified start after that is ``unsolved``: it has no label, and the exports refuse it.
   The budget is checked between stages, so a start can overrun it by one stage: at most
   ``max_rounds`` (6) re-centring calls of ``call_cpu_cap_s`` each, plus the refinement;
   IPOPT's ``max_iter`` usually stops a call first (3,000 iterations of a 60 s window are
   ~400 CPU-s).
   An exception is not an outcome: the level's record is an ``error``, its cache is kept,
   and the next run of the same output directory labels it again from the cached s-OCP and
   starts. A worker that dies leaves no record, so its level is labelled again too.
6. **Replicates** (the repeatability pilot, plan "Label noise": "200 pilot levels get at
   least 3 extra starts"). ``replicates`` = R runs the three starts R more times, each from
   its own guess with a seeded smooth jitter (``jitter_guess``), so the label procedure's
   own spread can be measured. Replicates never change the label; they are reported as
   ``j_star_replicates`` and ``j_star_all``.

**Levels are generated with x64 off** (``make_paper_levels``), whatever the caller's x64
state. Generating under x64 draws different random numbers: another car, another grip and
another strip under the same key (the spike's ``pool:i`` levels were made that way). The
consumers (``train_ued.py``, ``pilot_audit.py``) build their levels in float32, so a label
made on an x64 draw would belong to a different level. The float32 draws (car, grip, x0,
strip) are upcast exactly to float64 for the solve and the replay; the strip's node values
are its float32 spline coefficients evaluated in float64 (``generator.to_track``).

**Schema** (``SCHEMA``): one JSON record per level with its theta (keys, context, grip, x0 and
the strip's sha256), the policy, every start's outcome and stage log, the label, CPU times,
and the code identity (``code_identity``): the sha256 of every source file a label depends
on and of the R-dev track files the grammar is fitted on, the git commit, the files that
differ from it, and the ``sim-v1`` tag's commit. The tag does not exist yet, so records say
so (``sim_v1: null``) and carry the commit and the code hash instead.

**Resumable shards.** Each level writes ``records/<id>.json`` (and ``.npz``: the label's states
and controls, and every start's controls) atomically when it is done; the s-OCP and every
finished start are cached under ``work/<id>/`` until then, so a killed run resumes mid-level.
A run skips levels whose record is verified or unsolved, and relabels errors.

**Exports.** ``export_j_opt`` writes the .npy that ``scripts/pilot_audit.py --j-opt`` reads (J*
of the first n levels, in level order); ``export_oracle`` writes the .npz of
``ued.pool.save_oracle_labels`` for ``scripts/train_ued.py --labels``. Both check every level's
encoding against the level set they label and refuse missing, unsolved or mixed-code labels.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import platform
import subprocess
import tempfile
import time
import traceback
import warnings
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from apexzero.ocp.replay import REPLAY_RTOL, verify_window
from apexzero.ocp.window import (
    WindowResult,
    WindowSolver,
    WindowSpec,
    min_time_guess,
    reachable_strip,
    solve_continuation,
    window_return,
)
from apexzero.tracks.track import Track
from apexzero.vehicle.params import VehicleParams

SCHEMA = "apexzero-labels/1"
STARTS = ("socp", "qss", "agent")
OUTCOMES = ("verified", "unsolved", "error")
ROOT = Path(__file__).resolve().parents[3]
TRACKS_DIR = Path("data/tracks/tumftm")
DT = 0.05

# Files a label depends on (relative to the repository root). The code hash covers every .py
# file under the directories, tracked or not, so an uncommitted change shows up.
LABEL_SOURCES = (
    "src/apexzero/ocp",
    "src/apexzero/vehicle",
    "src/apexzero/env",
    "src/apexzero/tracks",
    "src/apexzero/oracle/qss.py",
    "src/apexzero/audit/qss_distance.py",
    "src/apexzero/agents/ppo.py",
)


# ----------------------------------------------------------------------------- policy


@dataclass(frozen=True)
class LabelPolicy:
    """Everything that decides a label besides the level and the code (stored in every
    record; a run refuses an output directory made under another policy)."""

    seconds: float = 60.0                         # the window (1,200 steps)
    stages_s: tuple[float, ...] = (10.0, 20.0, 40.0, 60.0)
    radius: float = 60.0                          # m, the window bound (spike: best for chains)
    overlap: int = 20                             # steps of a stage's optimum not reused
    max_iter: int = 3000                          # IPOPT, per stage (re-centring round)
    call_cpu_cap_s: float = 600.0                 # IPOPT max_cpu_time, per call
    start_cpu_budget_s: float = 900.0             # a chain stops before a stage past this
    retry: bool = True
    retry_stages_s: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    retry_max_iter: int = 6000
    retry_cpu_budget_s: float = 1800.0
    socp_retry_tol: float = 1e-5
    socp_ds: float = 3.0
    agent_windows: int = 8                        # stochastic rollouts (plus one deterministic)
    agent_obs_noise: float = 0.01                 # as the audit's rollouts
    agent_seed: int = 0
    replicates: int = 0                           # extra rounds of the three starts
    jitter: tuple[float, float, float] = (0.05, 0.005, 0.02)   # n (m), vx (relative), actions
    replay_rtol: float = REPLAY_RTOL

    def steps(self, seconds: float | None = None) -> int:
        return round((self.seconds if seconds is None else seconds) / DT)

    def horizons(self, retry: bool = False) -> list[int]:
        """Continuation stages in control steps, ending at the window."""
        stages = self.retry_stages_s if retry else self.stages_s
        return [self.steps(s) for s in stages if s < self.seconds] + [self.steps()]

    def to_json(self) -> dict:
        return _jsonable(dataclasses.asdict(self))


# ----------------------------------------------------------------------------- levels


@dataclass
class PaperLevel:
    """One paper level, regenerated from its encoding (float32 draws, upcast exactly)."""

    stream: str
    root: int
    index: int
    generator: str
    keys: np.ndarray          # [2] uint32
    ctx: np.ndarray           # [6] float64 (float32 values)
    hidden_mu: float
    x0: np.ndarray            # [8] float64
    track: Track              # float64 node values of the float32 strip

    @property
    def id(self) -> str:
        return level_id(self.stream, self.root, self.index)

    @property
    def p(self) -> VehicleParams:
        """The car the env drives: observed context times the hidden grip, in float64."""
        from apexzero.vehicle.params import F1_2025, with_context

        c = [float(v) for v in self.ctx]
        return with_context(F1_2025, c[0], c[1], c[2], c[3], c[4] * self.hidden_mu, c[5])

    def encoding(self) -> dict:
        return {"id": self.id, "stream": self.stream, "root": int(self.root), "index": int(self.index),
                "generator": self.generator, "keys": [int(k) for k in self.keys],
                "ctx": [float(v) for v in self.ctx], "hidden_mu": float(self.hidden_mu),
                "x0": [float(v) for v in self.x0], "strip_sha256": strip_sha256(self.track),
                "strip_length_m": float(self.track.length)}


def level_id(stream: str, root: int, index: int) -> str:
    return f"{stream}-r{int(root)}-{int(index):05d}"


def strip_sha256(track: Track) -> str:
    """sha256 of the strip as the solve and the replay read it (float64 node values)."""
    h = hashlib.sha256()
    h.update(f"{track.n_points} {track.ds!r} {track.closed}".encode())
    for a in (track.kappa, track.w_left, track.w_right, track.x, track.y, track.theta):
        h.update(np.ascontiguousarray(a, np.float64).tobytes())
    return h.hexdigest()


def r_dev_grammar():
    """The generator grammar every paper level set uses: fitted on the R-dev circuits
    (``fit_grammar`` refuses lockbox circuits), with x64 off."""
    import jax

    from apexzero.tracks.corners import R_DEV
    from apexzero.tracks.generator import fit_grammar
    from apexzero.tracks.track import load_tumftm

    with jax.enable_x64(False):
        return fit_grammar([load_tumftm(str(ROOT / TRACKS_DIR / f"{n}.csv")) for n in R_DEV])


def make_paper_levels(stream: str, root: int, indices: Sequence[int], grammar=None) -> list[PaperLevel]:
    """Levels ``indices`` of ``make_levels(PRNGKey(root), stream)``, built with x64 off as
    ``train_ued.py`` and ``pilot_audit.py`` build them (module docstring), then upcast."""
    import jax
    import jax.numpy as jnp

    from apexzero.tracks.generator import to_track
    from apexzero.tracks.levels import build, make_levels

    idx = np.asarray(sorted({int(i) for i in indices}), int)
    if idx.size == 0:
        return []
    if idx[0] < 0:
        raise ValueError(f"level indices must be >= 0, got {idx[0]}")
    with jax.enable_x64(False):
        g = grammar if grammar is not None else r_dev_grammar()
        lv = make_levels(jax.random.PRNGKey(int(root)), stream, n=int(idx[-1]) + 1)
        sel = jnp.asarray(idx)
        sub = lv._replace(keys=lv.keys[sel], ctx=lv.ctx[sel], hidden_mu=lv.hidden_mu[sel])
        tracks, _, x0 = build(sub, g)
        tracks = jax.tree.map(np.asarray, tracks)
        keys, ctx = np.asarray(sub.keys, np.uint32), np.asarray(sub.ctx)
        mu, x0 = np.asarray(sub.hidden_mu), np.asarray(x0)
    if ctx.dtype != np.float32:
        raise RuntimeError(f"level draws came out {ctx.dtype}, not float32")
    out = []
    for j, i in enumerate(idx):
        tr = to_track(jax.tree.map(lambda a, j=j: a[j], tracks), name=level_id(stream, root, i))
        tr = dataclasses.replace(tr, **{k: np.asarray(getattr(tr, k), np.float64)
                                        for k in ("s", "x", "y", "theta", "kappa", "w_left", "w_right")})
        out.append(PaperLevel(stream=stream, root=int(root), index=int(i), generator=lv.generator,
                              keys=keys[j].copy(), ctx=ctx[j].astype(np.float64),
                              hidden_mu=float(mu[j]), x0=x0[j].astype(np.float64), track=tr))
    return out


# ----------------------------------------------------------------------------- code identity


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_source_files(root: Path = ROOT) -> list[str]:
    files = set()
    for rel in LABEL_SOURCES:
        p = root / rel
        if p.is_dir():
            files |= {str(f.relative_to(root)) for f in p.rglob("*.py") if "__pycache__" not in f.parts}
        elif p.exists():
            files.add(rel)
    return sorted(files)


def _git(args: list[str], root: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def code_identity(root: Path = ROOT) -> dict:
    """What a label was computed with: sha256 per source file (``LABEL_SOURCES``) and per
    R-dev track file, one hash over all of them, the git commit and the hashed files that
    differ from it, and the ``sim-v1`` tag (None until it exists)."""
    from apexzero.tracks.corners import R_DEV

    files = {f: _sha256_file(root / f) for f in label_source_files(root)}
    data = {}
    for n in R_DEV:
        p = root / TRACKS_DIR / f"{n}.csv"
        data[str(TRACKS_DIR / f"{n}.csv")] = _sha256_file(p) if p.exists() else None
    h = hashlib.sha256()
    for k, v in sorted({**files, **data}.items()):
        h.update(f"{k}\0{v}\n".encode())
    commit = _git(["rev-parse", "HEAD"], root)
    dirty = _git(["status", "--porcelain", "--", *LABEL_SOURCES], root)
    sim = _git(["rev-parse", "-q", "--verify", "refs/tags/sim-v1^{commit}"], root)
    note = None if sim else ("sim-v1 is not tagged yet: the git commit and this code hash identify "
                             "the simulator instead")
    return {"sha256": h.hexdigest(), "files": files, "data": data, "git_commit": commit,
            "git_dirty": None if dirty is None else [ln[3:] for ln in dirty.splitlines() if ln],
            "sim_v1": sim, "sim_v1_note": note}


def environment() -> dict:
    import casadi
    import jax

    return {"python": platform.python_version(), "machine": platform.machine(),
            "system": platform.system(), "casadi": casadi.__version__, "jax": jax.__version__,
            "numpy": np.__version__}


# ----------------------------------------------------------------------------- guesses


@dataclass
class LineGuess:
    """A time-parametrised line in the fields ``window.min_time_guess`` reads from an
    ``OCPResult`` (s, t, z, u; lap_time for closed tracks), so it maps and extends the same way."""

    s: np.ndarray       # [N] centreline stations
    t: np.ndarray       # [N] time along the line
    z: np.ndarray       # [N, 7] n, xi, vx, vy, r, delta, pedal
    u: np.ndarray       # [N, 2] steer and pedal commands
    lap_time: float = float("nan")
    meta: dict = field(default_factory=dict)


def qss_line(track: Track, x0: np.ndarray, p: VehicleParams, margin: float) -> LineGuess:
    """The QSS line start: the minimum-curvature line of the strip inside the corridor
    tightened by ``margin`` (starting at x0's lateral offset), with the forward-backward QSS
    speed profile of car ``p`` along it from x0's speed, timed by constant acceleration
    between nodes. States: n = the line's offset, xi its heading to the centreline, vx the
    QSS speed, vy = 0, r = v kappa_line, delta the kinematic angle atan(L kappa_line), pedal
    the force balance m a + drag (throttle F max(v, v_floor) / P_max, brake F / F_brake_max).
    The commands equal the steer and pedal states. Not dynamically consistent (neither is
    the s-OCP guess): it says where and how fast, and IPOPT does the rest."""
    from apexzero.audit.qss_distance import line_geometry, min_curvature_offsets
    from apexzero.oracle.qss import GG, speed_profile

    if track.closed:
        raise ValueError("qss_line is for open strips (paper levels)")
    k = np.asarray(track.kappa, float)
    ds = track.ds
    alpha = min_curvature_offsets(track.x, track.y, track.theta, track.w_left, track.w_right,
                                  margin, start_offset=float(x0[1]))
    seg, k_line = line_geometry(k, alpha, ds)
    gg = GG.from_params(p)
    v = speed_profile(k_line, seg, gg, closed=False, v_start=float(x0[3]))
    t = np.concatenate([[0.0], np.cumsum(2.0 * seg / (v[:-1] + v[1:]))])
    sigma = np.concatenate([[0.0], np.cumsum(seg)])
    a_long = np.gradient(0.5 * v * v, sigma)
    xi = np.arctan2(np.gradient(alpha, ds), 1.0 - alpha * k)
    delta = np.arctan(p.wheelbase * k_line)
    force = p.m * a_long + gg.drag(v)
    pedal = np.where(force >= 0.0, force * np.maximum(v, p.v_drive_floor) / p.p_max,
                     force / p.f_brake_max)
    pedal = np.clip(pedal, -1.0, 1.0)
    steer = np.clip(delta / p.delta_max, -1.0, 1.0)
    z = np.stack([alpha, xi, v, np.zeros_like(v), v * k_line, steer * p.delta_max, pedal], 1)
    return LineGuess(s=np.asarray(track.s, float), t=t, z=z, u=np.stack([steer, pedal], 1),
                     meta={"margin": margin, "v_start": float(x0[3]), "line": "min-curvature"})


def smooth_noise(rng: np.random.Generator, shape: tuple[int, ...], width: int = 20) -> np.ndarray:
    """Unit-variance noise smoothed over ``width`` steps (a moving average along axis 0)."""
    raw = rng.standard_normal((shape[0] + width - 1, *shape[1:]))
    kern = np.ones(width) / np.sqrt(width)
    return np.apply_along_axis(lambda c: np.convolve(c, kern, mode="valid"), 0, raw)


def jitter_guess(guess: tuple[np.ndarray, np.ndarray], seed: Sequence[int],
                 scales: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """A replicate start's guess: smooth seeded jitter on n (m), vx (relative) and the
    controls; x0 and the action box are kept."""
    xs, acts = (np.array(v, float) for v in guess)
    rng = np.random.default_rng([int(s) for s in seed])
    sn, sv, sa = scales
    T = len(acts)
    xs[1:, 1] += sn * smooth_noise(rng, (T,))
    xs[1:, 3] *= 1.0 + sv * smooth_noise(rng, (T,))
    acts = np.clip(acts + sa * smooth_noise(rng, (T, 2)), -1.0, 1.0)
    return xs, acts


def splice(prefix_xs: np.ndarray, prefix_acts: np.ndarray, k: int, steps: int,
           extend: Callable) -> tuple[np.ndarray, np.ndarray]:
    """The first k steps of a rollout continued from its state x_k by ``extend``."""
    if k >= steps:
        return np.array(prefix_xs[: steps + 1], float), np.array(prefix_acts[:steps], float)
    xe, ae = extend(np.asarray(prefix_xs[k], float), steps - k)
    xs = np.vstack([prefix_xs[: k + 1], np.asarray(xe)[1:]])
    acts = np.vstack([prefix_acts[:k], ae]) if k else np.asarray(ae, float)
    return xs, acts


class AgentStart:
    """The agent-rollout start: rollouts of a checkpoint (``agents.checkpoints`` format) in
    the float64 env from the level's x0, one deterministic (mean action) and ``windows``
    stochastic (Gaussian actions, ``obs_noise`` on the normalised observation, as the audit).
    Keys are folded from ``seed`` and the level id, so a level's rollouts do not depend on
    the order or the worker."""

    def __init__(self, path: str | Path, windows: int = 8, obs_noise: float = 0.01, seed: int = 0):
        import jax.numpy as jnp

        from apexzero.agents.ppo import Actor, PPOConfig, RunningMeanStd
        from apexzero.env import racing

        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        d = pickle.loads(raw)
        names = {f.name for f in dataclasses.fields(PPOConfig)}
        self.cfg = PPOConfig(**{k: v for k, v in d["config"].items() if k in names})
        self.actor = Actor(self.cfg, racing.ACTION_DIM, False)
        self.params = d["actor"]
        self.rms = RunningMeanStd(**{k: jnp.asarray(v) for k, v in d["obs_rms"].items()})
        self.timesteps = d.get("timesteps")
        self.path = Path(path)
        self.windows, self.obs_noise, self.seed = int(windows), float(obs_noise), int(seed)
        self._fns: dict[int, Callable] = {}

    def describe(self) -> dict:
        p = self.path.resolve()
        rel = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name
        return {"checkpoint": rel, "sha256": self.sha256, "timesteps": self.timesteps,
                "windows": self.windows, "obs_noise": self.obs_noise, "seed": self.seed}

    def _fn(self, steps: int):
        if steps not in self._fns:
            import jax
            import jax.numpy as jnp

            from apexzero.env import racing

            actor, env_params = self.actor, racing.EnvParams(window_steps=10**9)

            def one(params, rms, tr, x0, ctx, mu, key, scale):
                z = jnp.zeros(2, jnp.float64)
                st = racing.EnvState(x=x0, prev_action=z, accel=z, ctx=ctx, hidden_mu=mu,
                                     t=jnp.zeros((), jnp.int32), slow=jnp.zeros((), jnp.float64), s0=x0[0])
                obs = racing.observe(st, tr, racing.params_from_context(ctx, hidden_mu=mu))

                def body(c, k):
                    st, ob, alive = c
                    k1, k2 = jax.random.split(k)
                    o = rms.normalize(ob.astype(jnp.float32))
                    o = o + scale * self.obs_noise * jax.random.normal(k1, o.shape, o.dtype)
                    mean, log_std = actor.apply(params, o)
                    a = mean + scale * jnp.exp(log_std) * jax.random.normal(k2, mean.shape, mean.dtype)
                    a = jnp.clip(a, -1.0, 1.0).astype(jnp.float64)
                    ob2, st2, r, term, _, _ = racing.step_no_reset(st, a, tr, env_params)
                    return (st2, ob2, alive & ~term), (st2.x, a, r, alive, term & alive)

                _, out = jax.lax.scan(body, (st, obs, jnp.array(True)), jax.random.split(key, steps))
                return out

            self._fns[steps] = jax.jit(jax.vmap(one, in_axes=(None, None, None, None, None, None, 0, 0)))
        return self._fns[steps]

    def rollouts(self, level: PaperLevel, steps: int, gamma: float = 0.995) -> dict:
        """Every rollout's states [K, T+1, 8], controls [K, T, 2], discounted return (crash
        penalty included, nothing after a termination), and first terminating step (or T)."""
        import jax
        import jax.numpy as jnp

        from apexzero.vehicle.sim import track_arrays

        tr = track_arrays(level.track, dtype=jnp.float64)
        key = jax.random.fold_in(jax.random.PRNGKey(self.seed), zlib.crc32(level.id.encode()))
        keys = jax.random.split(key, self.windows + 1)
        scale = jnp.asarray([0.0] + [1.0] * self.windows, jnp.float32)
        xs, acts, r, alive, term = (np.asarray(a) for a in self._fn(steps)(
            self.params, self.rms, tr, jnp.asarray(level.x0, jnp.float64),
            jnp.asarray(level.ctx, jnp.float64), jnp.asarray(level.hidden_mu, jnp.float64), keys, scale))
        K = xs.shape[0]
        xs = np.concatenate([np.broadcast_to(level.x0, (K, 1, 8)), xs], 1)
        ret = np.sum(gamma ** np.arange(steps) * r * alive, 1)
        crashed = term.any(1)
        k_term = np.where(crashed, np.argmax(term, 1), steps)
        return {"xs": xs, "acts": acts, "returns": ret, "crashed": crashed, "k_term": k_term}

    def guess(self, level: PaperLevel, steps: int, extend: Callable) -> tuple[tuple, dict]:
        """The best rollout (highest return) as a guess, cut at its terminating step and
        continued by ``extend``; info has every return and the best surviving one (a feasible
        control sequence: about a lower bound on J*, up to the solve's 3 cm tightening)."""
        ro = self.rollouts(level, steps)
        b = int(np.argmax(ro["returns"]))
        surv = ~ro["crashed"]
        info = {"returns": ro["returns"].tolist(), "crashed": ro["crashed"].tolist(),
                "best": b, "best_kind": "deterministic" if b == 0 else "stochastic",
                "best_return": float(ro["returns"][b]), "best_k_term": int(ro["k_term"][b]),
                "j_survivor_best": float(np.max(ro["returns"][surv])) if surv.any() else None,
                "survivors": int(surv.sum())}
        k = int(ro["k_term"][b])
        return splice(ro["xs"][b], ro["acts"][b], k, steps, extend), info


# ----------------------------------------------------------------------------- solves


class SolverCache:
    """One ``WindowSolver`` per (window length, max_iter, CPU cap), built on first use; its
    construction CPU is counted in ``setup_cpu`` (compiled libraries are cached on disk)."""

    def __init__(self, radius: float = 60.0, compile: bool = True, cache_dir: Path | None = None):
        self.radius, self.compile, self.cache_dir = radius, compile, cache_dir
        self._solvers: dict[tuple, WindowSolver] = {}
        self.setup_cpu = 0.0

    def get(self, steps: int, max_iter: int, cpu_cap: float | None) -> WindowSolver:
        key = (int(steps), int(max_iter), cpu_cap)
        if key not in self._solvers:
            c0 = time.process_time()
            self._solvers[key] = WindowSolver(
                WindowSpec(steps=int(steps), radius=self.radius), max_iter=int(max_iter),
                compile=self.compile, cache_dir=self.cache_dir,
                ipopt=None if cpu_cap is None else {"max_cpu_time": float(cpu_cap)})
            self.setup_cpu += time.process_time() - c0
        return self._solvers[key]

    def factory(self, max_iter: int, cpu_cap: float | None) -> Callable[[int], WindowSolver]:
        return lambda steps: self.get(steps, max_iter, cpu_cap)


def continuation(solver_for, level: PaperLevel, horizons: Sequence[int], first_guess, extend,
                 overlap: int = 20, cpu_budget: float | None = None):
    """``window.solve_continuation`` one stage at a time, so a chain can stop on its CPU
    budget between stages. The solver calls and warm starts are the library's: each stage
    continues from the last successful one (else from ``first_guess``). Returns (the final
    stage's result or None if the chain stopped early, every stage's result, capped)."""
    results: list[WindowResult] = []
    last_good, cpu, capped = None, 0.0, False
    for T in horizons:
        if cpu_budget is not None and cpu >= cpu_budget:
            capped = True
            break
        res = solve_continuation(solver_for, level.track, level.x0, [T],
                                 first_guess if last_good is None else None, extend, overlap=overlap,
                                 ctx=level.ctx, p=level.p, previous=last_good,
                                 hidden_mu=level.hidden_mu)[0]
        res.meta["chain"] = [(len(r.a), r.status) for r in results] + [(len(res.a), res.status)]
        cpu += res.meta["cpu_seconds"]
        results.append(res)
        if res.success:
            last_good = res
    final = results[-1] if results and not capped and len(results[-1].a) == horizons[-1] else None
    return final, results, capped


@dataclass
class StartOutcome:
    name: str                         # socp | qss | agent
    round: str                        # primary | retry | replicate-<r>
    extension: str                    # socp | qss
    value: float | None               # J* of the final stage (verified or not)
    verified: bool
    status: str
    cpu_s: float                      # IPOPT stages
    replay_cpu_s: float
    iterations: int
    capped: bool
    stages: list[dict]
    j_guess: float
    replay: dict | None = None
    x: np.ndarray | None = None       # final stage's states [T+1, 8] and controls [T, 2]
    a: np.ndarray | None = None
    note: str | None = None

    @property
    def key(self) -> str:
        return f"{self.round}-{self.name}"

    def summary(self) -> dict:
        d = {k: getattr(self, k) for k in ("name", "round", "extension", "value", "verified", "status",
                                           "cpu_s", "replay_cpu_s", "iterations", "capped", "j_guess",
                                           "replay", "stages", "note")}
        return _jsonable(d)


def run_start(name: str, round_: str, guess, extend, extension: str, level: PaperLevel,
              policy: LabelPolicy, solvers: SolverCache, retry: bool = False) -> StartOutcome:
    """One start: the continuation chain (the retry schedule and budgets if ``retry``), then
    the float64 replay of its final stage."""
    horizons = policy.horizons(retry)
    max_iter = policy.retry_max_iter if retry else policy.max_iter
    budget = policy.retry_cpu_budget_s if retry else policy.start_cpu_budget_s
    spec = WindowSpec(steps=horizons[-1], radius=policy.radius)
    j_guess = window_return(np.asarray(guess[0]), np.asarray(guess[1]), spec)
    final, stages, capped = continuation(solvers.factory(max_iter, policy.call_cpu_cap_s), level,
                                         horizons, guess, extend, policy.overlap, budget)
    verified, replay, rcpu = False, None, 0.0
    if final is not None:
        c0 = time.process_time()
        verified = verify_window(final, level.track, level.x0, level.ctx, level.hidden_mu,
                                 policy.replay_rtol)
        rcpu = time.process_time() - c0
        replay = final.meta.get("replay")
    status = (final.status if final is not None else
              f"capped after {len(stages)} of {len(horizons)} stages" if capped else "no stages")
    log = [dict(seconds=len(r.a) * DT, iterations=r.iterations, rounds=r.rounds,
                cpu_s=r.meta["cpu_seconds"], J=r.value, status=r.status,
                warm_start=r.meta.get("warm_start")) for r in stages]
    return StartOutcome(name=name, round=round_, extension=extension,
                        value=None if final is None else float(final.value), verified=bool(verified),
                        status=status, cpu_s=float(sum(r.meta["cpu_seconds"] for r in stages)),
                        replay_cpu_s=rcpu, iterations=int(sum(r.iterations for r in stages)),
                        capped=capped, stages=log, j_guess=float(j_guess), replay=replay,
                        x=None if final is None else np.asarray(final.x),
                        a=None if final is None else np.asarray(final.a))


@dataclass
class SOCP:
    result: object | None             # min_time.OCPResult on ``strip``
    strip: Track
    info: dict

    @property
    def usable(self) -> bool:
        return bool(self.info.get("usable"))


def solve_socp(level: PaperLevel, policy: LabelPolicy) -> SOCP:
    """The level's s-OCP on its reachable strip, with one retry from the iterate (module
    docstring, step 1). ``usable``: finite, with time increasing along the strip."""
    from apexzero.ocp.min_time import solve_min_time

    strip = reachable_strip(level.track, level.x0, policy.seconds, level.p)
    c0 = time.process_time()
    res = solve_min_time(strip, level.p, ds=policy.socp_ds)
    first = {"status": res.status, "iterations": res.iterations}
    retried = False
    if not res.success:
        retried = True
        res = solve_min_time(strip, level.p, ds=policy.socp_ds, init=(res.z, res.u),
                             tol=policy.socp_retry_tol)
    cpu = time.process_time() - c0
    finite = bool(np.all(np.isfinite(res.z)) and np.all(np.isfinite(res.u)) and np.all(np.isfinite(res.t)))
    usable = finite and bool(np.all(np.diff(res.t) > 0))
    info = {"success": bool(res.success), "status": res.status, "iterations": int(res.iterations),
            "retried": retried, "first": first if retried else None, "usable": usable,
            "cpu_s": cpu, "strip_m": float(strip.length), "of_m": float(level.track.length),
            "time_s": float(res.t[-1]) if finite else None}
    return SOCP(result=res if usable else None, strip=strip, info=info)


# ----------------------------------------------------------------------------- a level


@dataclass
class LevelLabel:
    record: dict
    arrays: dict


def _primary_order(starts: Sequence[str]) -> list[str]:
    bad = set(starts) - set(STARTS)
    if bad:
        raise ValueError(f"unknown starts {sorted(bad)}; known: {STARTS}")
    return [s for s in STARTS if s in starts]


def label_level(level: PaperLevel, policy: LabelPolicy, solvers: SolverCache,
                agent: AgentStart | None = None, starts: Sequence[str] = STARTS,
                work: Path | None = None, code: dict | None = None,
                log: Callable[[str], None] | None = None) -> LevelLabel:
    """Label one level (module docstring, steps 1-6). ``work``: the level's cache directory
    (s-OCP and finished starts), for resuming."""
    import jax

    from apexzero.env.racing import EnvParams

    if not jax.config.jax_enable_x64:
        raise RuntimeError("label_level needs jax_enable_x64 (float64 guesses and replay)")
    starts = _primary_order(starts)
    if "agent" in starts and agent is None:
        raise ValueError("the agent start needs a checkpoint (AgentStart)")
    say = log or (lambda m: None)
    c0, w0, setup0 = time.process_time(), time.perf_counter(), solvers.setup_cpu
    H = policy.steps()
    if work is not None:
        work.mkdir(parents=True, exist_ok=True)
    hits: list[str] = []

    socp = _cached(work, "socp.pkl", lambda: solve_socp(level, policy), hits)
    say(f"{level.id}: s-OCP {socp.info['status']} on {socp.info['strip_m']:.0f} m, "
        f"{socp.info['cpu_s']:.0f} CPU-s{'' if socp.usable else ' (not usable)'}")
    g0 = time.process_time()
    line = qss_line(level.track, level.x0, level.p, EnvParams().edge_margin + WindowSpec().safety)

    def ext_qss(state, n):
        return min_time_guess(line, level.track, state, n)

    def ext_socp(state, n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # past the cut strip's end: held, as designed
            return min_time_guess(socp.result, socp.strip, state, n)

    extend = {"qss": ext_qss, "socp": ext_socp if socp.usable else ext_qss}
    ext_name = {"qss": "qss", "socp": "socp" if socp.usable else "qss"}
    guesses: dict[str, tuple] = {}
    agent_info = None
    notes = {}
    for name in starts:
        if name == "socp":
            if socp.usable:
                guesses[name] = (extend["socp"](level.x0, H), "socp")
            else:
                notes[name] = "skipped: no finite s-OCP"
        elif name == "qss":
            guesses[name] = (ext_qss(level.x0, H), "qss")
        else:
            g, agent_info = agent.guess(level, H, extend["socp"])
            guesses[name] = (g, ext_name["socp"])
    guess_cpu = time.process_time() - g0

    outcomes: list[StartOutcome] = []

    def go(name, round_, guess, retry=False):
        g, e = guess
        out = _cached(work, f"start-{round_}-{name}.pkl",
                      lambda: run_start(name, round_, g, extend[e], e, level, policy, solvers, retry), hits)
        outcomes.append(out)
        say(f"{level.id}: {round_} {name}: {out.status}, J {out.value}, verified {out.verified}, "
            f"{out.cpu_s:.0f} CPU-s, {out.iterations} it")
        return out

    for name in starts:
        if name in guesses:
            go(name, "primary", guesses[name])
    retried = False
    if policy.retry and not any(o.verified for o in outcomes):
        retried = True
        for name in starts:
            if name in guesses and go(name, "retry", guesses[name], retry=True).verified:
                break
    for r in range(1, policy.replicates + 1):
        for name in starts:
            if name in guesses:
                g, e = guesses[name]
                seed = (zlib.crc32(level.id.encode()), r, STARTS.index(name))
                go(name, f"replicate-{r}", (jitter_guess(g, seed, policy.jitter), e))

    label_pool = [o for o in outcomes if o.round in ("primary", "retry") and o.verified]
    best = max(label_pool, key=lambda o: o.value) if label_pool else None
    verified_all = [o for o in outcomes if o.verified]
    reps = []
    for r in range(policy.replicates + 1):
        rnd = ("primary", "retry") if r == 0 else (f"replicate-{r}",)
        vals = [o.value for o in outcomes if o.round in rnd and o.verified]
        reps.append(max(vals) if vals else None)
    prim = [o.value for o in outcomes if o.round == "primary" and o.verified]
    j_surv = None if agent_info is None else agent_info["j_survivor_best"]
    label = {
        "status": "verified" if best is not None else "unsolved",
        "j_star": None if best is None else best.value,
        "start": None if best is None else best.key,
        "j_replay": None if best is None else best.replay["return"],
        "retried": retried,
        "n_primary": sum(o.round == "primary" for o in outcomes),
        "n_verified_primary": len(prim),
        "primary_spread": float(max(prim) - min(prim)) if len(prim) > 1 else None,
        "j_star_replicates": reps,
        "j_star_all": max(o.value for o in verified_all) if verified_all else None,
        "j_agent_survivor_best": j_surv,
        "agent_beats_label": (None if best is None or j_surv is None
                              else bool(j_surv > best.value + policy.replay_rtol * abs(best.value))),
    }
    solves = sum(o.cpu_s for o in outcomes)
    replays = sum(o.replay_cpu_s for o in outcomes)
    cpu = {
        # The level's cost: s-OCP + guesses + every IPOPT call + every replay. Each part
        # carries its own CPU, so this is the same whether or not the level was resumed
        # from its cache; one-off setup (building the solvers) is reported apart.
        "level_s": socp.info["cpu_s"] + guess_cpu + solves + replays,
        "level_process_s": time.process_time() - c0,    # this call only (cache hits excluded)
        "wall_s": time.perf_counter() - w0,
        "socp_s": socp.info["cpu_s"],
        "guesses_s": guess_cpu,
        "solves_s": solves,
        "replays_s": replays,
        "setup_s": solvers.setup_cpu - setup0,
        "primary_s": sum(o.cpu_s + o.replay_cpu_s for o in outcomes if o.round == "primary"),
        "retry_s": sum(o.cpu_s + o.replay_cpu_s for o in outcomes if o.round == "retry"),
        "replicates_s": sum(o.cpu_s + o.replay_cpu_s for o in outcomes if o.round.startswith("replicate")),
        "cached": hits,       # parts reused from an interrupted run
    }
    record = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level.encoding(),
        "problem": {"window_steps": H, "dt": DT, "spec": _jsonable(WindowSpec(steps=H, radius=policy.radius).__dict__),
                    "replay_rtol": policy.replay_rtol, "objective": "sum gamma^t [ds/10 - 0.02 |a_t - a_{t-1}|^2]"},
        "policy": policy.to_json(),
        "starts_requested": list(starts),
        "notes": notes,
        "agent": None if agent is None else {**agent.describe(), "rollouts": agent_info},
        "socp": socp.info,
        "starts": [o.summary() for o in outcomes],
        "label": label,
        "cpu": cpu,
        "code": code,
        "environment": environment(),
    }
    arrays = {"x0": level.x0}
    if best is not None:
        arrays.update(label_x=best.x, label_a=best.a)
    for o in outcomes:
        if o.a is not None:
            arrays[f"a_{o.key}"] = o.a
    return LevelLabel(record=_jsonable(record), arrays=arrays)


def _cached(work: Path | None, name: str, make: Callable, hits: list[str]):
    """``make()``, cached as a pickle in ``work`` (None: no cache); a reuse is noted in ``hits``."""
    if work is None:
        return make()
    path = work / name
    if path.exists():
        hits.append(name)
        with open(path, "rb") as fh:
            return pickle.load(fh)
    out = make()
    _atomic_write(path, pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
    return out


# ----------------------------------------------------------------------------- records


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    return x


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o644)            # mkstemp's 0600 would hide records from other users
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def record_path(out: Path, lid: str) -> Path:
    return Path(out) / "records" / f"{lid}.json"


def write_record(out: Path, lab: LevelLabel) -> Path:
    """The level's record and arrays, each written atomically (arrays first, so a record
    on disk always has its arrays). Once the level has an outcome (``is_done``) its work
    cache is removed; an ``error`` record keeps it, so the next run resumes from it."""
    import io
    import shutil

    lid = lab.record["level"]["id"]
    path = record_path(out, lid)
    buf = io.BytesIO()
    np.savez_compressed(buf, **lab.arrays)
    _atomic_write(path.with_suffix(".npz"), buf.getvalue())
    _atomic_write(path, json.dumps(lab.record, indent=1).encode())
    work = Path(out) / "work" / lid
    if is_done(lab.record) and work.exists():
        shutil.rmtree(work)
    return path


def error_record(level: PaperLevel, exc: BaseException, policy: LabelPolicy, code: dict | None,
                 cpu_s: float) -> LevelLabel:
    rec = {"schema": SCHEMA, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "level": level.encoding(), "policy": policy.to_json(),
           "label": {"status": "error", "j_star": None},
           "error": {"type": type(exc).__name__, "message": str(exc),
                     "traceback": traceback.format_exception(exc)},
           "cpu": {"level_process_s": cpu_s}, "code": code}
    return LevelLabel(record=_jsonable(rec), arrays={"x0": level.x0})


def load_record(out: Path, lid: str) -> dict | None:
    p = record_path(out, lid)
    return json.loads(p.read_text()) if p.exists() else None


def iter_records(out: Path) -> Iterable[dict]:
    for p in sorted((Path(out) / "records").glob("*.json")):
        yield json.loads(p.read_text())


def is_done(rec: dict | None) -> bool:
    """A level is done once it has an outcome (verified or unsolved); errors are redone."""
    return rec is not None and rec.get("label", {}).get("status") in ("verified", "unsolved")


def select_shard(indices: Sequence[int], shard: tuple[int, int] | None) -> list[int]:
    """Indices with index % n == i for shard (i, n): disjoint shards cover every index."""
    idx = sorted({int(i) for i in indices})
    if shard is None:
        return idx
    i, n = shard
    if not 0 <= i < n:
        raise ValueError(f"shard {i}/{n}: need 0 <= i < n")
    return [k for k in idx if k % n == i]


# ----------------------------------------------------------------------------- exports


class LabelError(ValueError):
    """Labels that cannot be exported as asked (missing, unsolved, mismatched, mixed)."""


def collect(out: Path, levels_encoding: Sequence[dict], allow_mixed_code: bool = False
            ) -> tuple[np.ndarray, list[dict]]:
    """J* in the order of ``levels_encoding`` (dicts with stream, root, index, keys, ctx,
    hidden_mu), from verified records only, each checked against its level's encoding (keys
    exactly, ctx and hidden_mu as float32). Raises ``LabelError`` naming every problem."""
    problems, js, recs = [], [], []
    for enc in levels_encoding:
        lid = level_id(enc["stream"], enc["root"], enc["index"])
        rec = load_record(out, lid)
        if rec is None:
            problems.append(f"{lid}: no record")
            continue
        st = rec.get("label", {}).get("status")
        if st != "verified":
            problems.append(f"{lid}: {st}")
            continue
        lv = rec["level"]
        same = (np.array_equal(np.asarray(lv["keys"], np.uint32), np.asarray(enc["keys"], np.uint32))
                and np.array_equal(np.asarray(lv["ctx"], np.float32), np.asarray(enc["ctx"], np.float32))
                and np.float32(lv["hidden_mu"]) == np.float32(enc["hidden_mu"]))
        if not same:
            problems.append(f"{lid}: the record's level differs from the level set's (keys, ctx or grip)")
            continue
        js.append(float(rec["label"]["j_star"]))
        recs.append(rec)
    codes = sorted({(r.get("code") or {}).get("sha256") for r in recs}, key=str)
    if len(codes) > 1 and not allow_mixed_code:
        problems.append(f"labels from {len(codes)} code versions: {', '.join(str(c)[:12] for c in codes)}")
    if problems:
        head = "; ".join(problems[:20]) + (f"; and {len(problems) - 20} more" if len(problems) > 20 else "")
        raise LabelError(f"{len(problems)} problem(s): {head}")
    return np.asarray(js, np.float64), recs


def levels_encoding(levels, stream: str, root: int) -> list[dict]:
    """Encodings of a ``tracks.levels.Levels`` set (in its order) for ``collect``."""
    keys, ctx, mu = (np.asarray(getattr(levels, k)) for k in ("keys", "ctx", "hidden_mu"))
    return [{"stream": stream, "root": int(root), "index": i, "keys": keys[i], "ctx": ctx[i],
             "hidden_mu": mu[i]} for i in range(len(mu))]


def float32_levels(stream: str, root: int, n: int):
    """``make_levels(PRNGKey(root), stream, n)`` with x64 off (the consumers' draws)."""
    import jax

    from apexzero.tracks.levels import make_levels

    with jax.enable_x64(False):
        return make_levels(jax.random.PRNGKey(int(root)), stream, n=int(n))


def _provenance(recs: list[dict], **extra) -> dict:
    codes = sorted({(r.get("code") or {}).get("sha256") for r in recs}, key=str)
    commits = sorted({(r.get("code") or {}).get("git_commit") for r in recs}, key=str)
    return _jsonable({"schema": SCHEMA, "levels": len(recs), "code_sha256": codes, "git_commits": commits,
                      "sim_v1": sorted({(r.get("code") or {}).get("sim_v1") for r in recs}, key=str),
                      "policy": recs[0]["policy"] if recs else None,
                      "retried": sum(bool(r["label"]["retried"]) for r in recs),
                      "agent": sorted({(r.get("agent") or {}).get("sha256") for r in recs}, key=str),
                      **extra})


def export_j_opt(out: Path, path: Path, n: int, stream: str = "a_dr", root: int = 2026,
                 allow_mixed_code: bool = False) -> np.ndarray:
    """``pilot_audit.py --j-opt``: J* of levels 0..n-1 of ``stream`` at ``root`` (its levels
    are ``make_levels(PRNGKey(2026), 'a_dr', n)``), float64 [n], plus ``<path>.json`` with the
    provenance."""
    lv = float32_levels(stream, root, n)
    js, recs = collect(out, levels_encoding(lv, stream, root), allow_mixed_code)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, js)
    Path(f"{path}.json").write_text(json.dumps(_provenance(recs, consumer="pilot_audit --j-opt",
                                                           stream=stream, root=root), indent=1))
    return js


def export_oracle(out: Path, path: Path, levels, stream: str = "pool", root: int = 0,
                  allow_mixed_code: bool = False) -> np.ndarray:
    """``train_ued.py --labels``: ``ued.pool.save_oracle_labels`` for the pool ``levels``
    (``tracks.levels.Levels``, e.g. ``levels.load(pool_file)``), plus ``<path>.json``."""
    from apexzero.ued.pool import save_oracle_labels

    js, recs = collect(out, levels_encoding(levels, stream, root), allow_mixed_code)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_oracle_labels(path, levels, js)
    Path(f"{path}.json").write_text(json.dumps(_provenance(recs, consumer="train_ued --labels",
                                                           stream=stream, root=root), indent=1))
    return js


def summarise(records: Iterable[dict]) -> dict:
    """Counts by outcome, CPU per level, and the starts' verification rates."""
    recs = list(records)
    by = {s: [r for r in recs if r.get("label", {}).get("status") == s] for s in OUTCOMES}
    done = by["verified"] + by["unsolved"]

    def stats(key):
        a = np.array([r["cpu"][key] for r in done], float)
        return None if not a.size else {"mean": float(a.mean()), "median": float(np.median(a)),
                                        "max": float(a.max()), "total": float(a.sum())}

    rates = {}
    for r in done:
        for o in r["starts"]:
            k = f"{o['round'].split('-')[0]}:{o['name']}"
            n, v = rates.get(k, (0, 0))
            rates[k] = (n + 1, v + bool(o["verified"]))
    return {"levels": len(recs), **{s: len(v) for s, v in by.items()},
            "retried": sum(bool(r["label"].get("retried")) for r in done),
            "cpu_level_s": stats("level_s"), "cpu_socp_s": stats("socp_s"),
            "cpu_primary_s": stats("primary_s"),
            "start_verified": {k: f"{v}/{n}" for k, (n, v) in sorted(rates.items())}}
