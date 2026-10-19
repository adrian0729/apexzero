"""Agents, snapshots and the manifest (protocol 8, step 0).

Snapshots load with ``audit.rollouts.snapshot_from_pickle``, the PPOConfig filtered to its
known fields (as viz/export.load_policy does) and ``Actor(cfg, 2, False)``. The manifest
(``outputs/psychophysics/manifest.json``) records every snapshot's sha256, timesteps, size
and final exp(log_std), and for every prospective snapshot that it has not been evaluated
on any stimulus. Runs still training are listed only once finished (agent.pkl exists for
specialists, written by train.py when training ends; eval_rdev.json for generalists, written
by train_dr.py at the end), and a pickle that
fails to load (``save_snapshot`` is not atomic) is skipped with its error recorded.

Freeze rule: before the tag, no snapshot is rolled out on any protocol stimulus. Loading a
snapshot and one forward pass on a dummy observation is allowed (manifest step 0).
``real_lap`` drives a snapshot round a real R_DEV circuit (the standard deterministic
evaluation of diagnose_crashes.py, not a stimulus); only the Integrate phase calls it, see
IMPLEMENTATION.md.

Timesteps: checkpoints carry ``timesteps``; S1's agent.pkl does not, and train.py ends on
the last whole update, so its timesteps are ``num_updates * batch_size`` of its config
(199,884,800 for the 200M specialists, the same number S2's and S3's final checkpoints are
named by), not ``total_timesteps``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import math
import pickle
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import Actor, PPOConfig, RunningMeanStd
from apexzero.audit.rollouts import Snapshot, snapshot_from_pickle
from apexzero.env import racing
from apexzero.interp.io import (
    MANIFEST_PATH,
    REPO,
    assert_x64_off,
    code_hash,
    read_json,
    sha256_arrays,
    sha256_file,
    write_json,
)
from apexzero.interp.stimuli import RealState
from apexzero.interp.support import GRAMMARS, GrammarSpec


@dataclass(frozen=True)
class AgentSpec:
    """One trained agent of protocol 8."""

    agent_id: str                 # 'S1', 'S2', 'S3', 'G1', 'G2', 'G3'
    run: str                      # repo-relative run directory
    kind: str                     # 'specialist' | 'generalist'
    pilot_informed: bool          # seen in the pilots (secondary family)
    prospective_after: float      # snapshots with timesteps > this are prospective (inf: none)
    final_file: str               # 'agent.pkl' (S1) or 'ckpt' (last ckpt/*.pkl)
    done_file: str                # file whose presence marks the run finished
    grammar: GrammarSpec | None   # training grammar (generalists)


AGENTS = {
    "S1": AgentSpec("S1", "runs/hockenheim-specialist-mlp-200M", "specialist", True,
                    float("inf"), "agent.pkl", "agent.pkl", None),
    "S2": AgentSpec("S2", "runs/hockenheim-specialist-mlp-200M-s2", "specialist", False,
                    122.3e6, "ckpt", "agent.pkl", None),
    "S3": AgentSpec("S3", "runs/hockenheim-specialist-mlp-200M-s3", "specialist", False,
                    0.0, "ckpt", "agent.pkl", None),
    "G1": AgentSpec("G1", "runs/dr-pilot-200M", "generalist", True, float("inf"), "ckpt",
                    "eval_rdev.json", GRAMMARS["base"]),
    "G2": AgentSpec("G2", "runs/dr-pilot-v2-tilt30", "generalist", True, float("inf"),
                    "ckpt", "eval_rdev.json", GRAMMARS["v2mix"]),
    "G3": AgentSpec("G3", "runs/dr-pilot-v3-tilt50a2", "generalist", False, 0.0, "ckpt",
                    "eval_rdev.json", GRAMMARS["v3mix"]),
}
EXCLUDED_RUNS = ("runs/hockenheim-specialist-mlp",)


@dataclass
class LoadedAgent:
    """A snapshot ready for the harness."""

    agent_id: str
    path: Path
    timesteps: int
    cfg: PPOConfig                # filtered to PPOConfig's fields
    snapshot: Snapshot            # actor, critic, obs_rms, ret_rms
    log_std: np.ndarray           # [2] actor log_std (steer, pedal)
    is_final: bool

    @property
    def actor_params(self):
        return self.snapshot.actor

    @property
    def obs_rms(self) -> RunningMeanStd:
        return self.snapshot.obs_rms


class SnapshotError(RuntimeError):
    """A snapshot that cannot be loaded (partially written, corrupt)."""


def _run_dir(spec: AgentSpec) -> Path:
    return REPO / spec.run          # an absolute ``run`` (tests) replaces REPO


def _rel(path: Path) -> str:
    """Repo-relative path string (absolute if outside the repo)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def load_agent(path: str | Path, agent_id: str = "") -> LoadedAgent:
    """Load one pickle (catching ``pickle.UnpicklingError`` and EOFError as a
    ``SnapshotError``). timesteps from the pickle's 'timesteps' key, else num_updates *
    batch_size of its config (agent.pkl; see the module docstring)."""
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        known = {f.name for f in dataclasses.fields(PPOConfig)}
        cfg = PPOConfig(**{k: v for k, v in d["config"].items() if k in known})
        snap = snapshot_from_pickle(d)
        log_std = np.asarray(d["actor"]["params"]["log_std"], np.float32).reshape(2)
    except (pickle.UnpicklingError, EOFError, KeyError, TypeError, ValueError,
            AttributeError) as e:
        raise SnapshotError(f"{path}: {type(e).__name__}: {e}") from e
    timesteps = int(d["timesteps"]) if "timesteps" in d else cfg.num_updates * cfg.batch_size
    # The harness normalises observations as training did and adds no observation noise
    # (protocol 1.4: obs_noise = 0); a snapshot trained otherwise cannot be run as is.
    if not getattr(cfg, "normalize_obs", True) or float(getattr(cfg, "obs_noise", 0.0)) != 0.0:
        raise SnapshotError(f"{path}: config normalize_obs={getattr(cfg, 'normalize_obs', None)}, "
                            f"obs_noise={getattr(cfg, 'obs_noise', None)}: the harness assumes "
                            "normalised observations without noise")
    spec = AGENTS.get(agent_id)
    final = final_snapshot(spec) if spec is not None else None
    is_final = final is not None and final.resolve() == path.resolve()
    return LoadedAgent(agent_id, path, timesteps, cfg, snap, log_std, is_final)


def list_snapshots(spec: AgentSpec) -> list[Path]:
    """Snapshots of a finished run in timestep order (ckpt/*.pkl, or agent.pkl for S1);
    [] while the run is unfinished (no ``done_file``)."""
    run = _run_dir(spec)
    if not (run / spec.done_file).exists():
        return []
    if spec.final_file == "agent.pkl":
        return [run / "agent.pkl"]
    # ckpt names are zero-padded timesteps, so the lexical order is the timestep order.
    return sorted((run / "ckpt").glob("*.pkl"))


def final_snapshot(spec: AgentSpec) -> Path | None:
    snaps = list_snapshots(spec)
    return snaps[-1] if snaps else None


def last_three(spec: AgentSpec) -> list[Path]:
    """The final and the two snapshots before it (P6 stability rule, protocol 8)."""
    return list_snapshots(spec)[-3:]


def forward_pass_ok(agent: LoadedAgent) -> bool:
    """One actor forward pass on a dummy (all-zero) normalised observation gives a finite
    mean and log_std. This is the only use of a snapshot allowed before the tag."""
    actor = Actor(agent.cfg, racing.ACTION_DIM, False)
    mean, log_std = actor.apply(agent.actor_params, jnp.zeros(racing.OBS_DIM, jnp.float32))
    return bool(np.all(np.isfinite(np.asarray(mean))) and np.all(np.isfinite(np.asarray(log_std))))


# ------------------------------------------------------------------------- manifest


@dataclass
class SnapshotEntry:
    """One snapshot in the manifest."""

    agent: str
    path: str                     # repo-relative
    timesteps: int
    is_final: bool
    sha256: str
    size_bytes: int
    log_std: tuple[float, float]  # (steer, pedal)
    std: tuple[float, float]      # exp(log_std)
    prospective: bool
    evaluated_on_stimuli: bool    # False for every prospective snapshot at the tag
    first_evaluated: str = ""     # ISO time and stage of the first rollout on a stimulus
    forward_pass_ok: bool = False # finite output on a dummy observation
    obs_rms_sha256: str = ""
    gate: dict = field(default_factory=dict)   # {'completion', 'osc', 'passed'} after rollouts
    error: str = ""               # load error, if any


@dataclass
class Manifest:
    """``outputs/psychophysics/manifest.json`` (protocol 10, steps 0, 1, 4, 5)."""

    version: str = "psychophysics-prereg-v2"
    created: str = ""
    updated: str = ""
    root_seed: int = 20261019
    git_commit: str = ""
    code_hash: str = ""
    stimulus_hash: str = ""
    key_cache_hash: str = ""
    tag: str = ""                 # set when the tag is made
    agents: dict[str, dict] = field(default_factory=dict)    # agent_id -> AgentSpec fields
    snapshots: list[SnapshotEntry] = field(default_factory=list)
    support: dict = field(default_factory=dict)      # path + hash of support.json; regimes
    band: float | None = None     # b from V8
    tau_table: str = ""           # path of the tau table
    lists: dict[str, list] = field(default_factory=dict)     # infeasible, unsolved, multi_basin, ...
    validation: dict[str, dict] = field(default_factory=dict)  # V1..V11 results
    budget: dict = field(default_factory=dict)       # timing, projections, cuts (with reasons)
    excluded_runs: list[str] = field(default_factory=lambda: list(EXCLUDED_RUNS))
    deviations: list[dict] = field(default_factory=list)  # {date, item, reason}

    def entry(self, path: str) -> SnapshotEntry | None:
        """The snapshot entry with this (repo-relative) path."""
        for e in self.snapshots:
            if e.path == path:
                return e
        return None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _timesteps_from_name(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return -1


def snapshot_entry(spec: AgentSpec, path: Path, is_final: bool) -> SnapshotEntry:
    """Hash, size, timesteps, log_std and a forward pass on a zero observation. A snapshot
    that does not load gives an entry with ``error`` set (timesteps from its file name)."""
    path = Path(path)
    sha, size = sha256_file(path), path.stat().st_size
    try:
        ag = load_agent(path, spec.agent_id)
    except SnapshotError as e:
        nan = (float("nan"), float("nan"))
        ts = _timesteps_from_name(path)
        return SnapshotEntry(spec.agent_id, _rel(path), ts, is_final, sha, size, nan, nan,
                             ts > spec.prospective_after, False, error=str(e))
    ls = tuple(float(v) for v in ag.log_std)
    rms = ag.obs_rms
    return SnapshotEntry(
        agent=spec.agent_id, path=_rel(path), timesteps=ag.timesteps, is_final=is_final,
        sha256=sha, size_bytes=size, log_std=ls, std=tuple(math.exp(v) for v in ls),
        prospective=ag.timesteps > spec.prospective_after, evaluated_on_stimuli=False,
        forward_pass_ok=forward_pass_ok(ag),
        obs_rms_sha256=sha256_arrays(*(np.asarray(a) for a in (rms.mean, rms.var, rms.count))))


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _agent_record(spec: AgentSpec, snaps: list[Path]) -> dict:
    rec = {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)}
    rec["finished"] = bool(snaps)
    rec["n_snapshots"] = len(snaps)
    rec["final"] = _rel(snaps[-1]) if snaps else ""
    if snaps:
        rec["status"] = "finished"
    else:
        # Not listed until the run is done (protocol 8): say why, so a pending agent is
        # visible in the manifest rather than silently absent.
        run = _run_dir(spec)
        n_ckpt = len(list((run / "ckpt").glob("*.pkl"))) if (run / "ckpt").is_dir() else 0
        rec["status"] = (f"pending: no {spec.done_file} yet ({n_ckpt} checkpoints written so far)"
                         if run.is_dir() else "pending: run directory missing")
    return rec


@contextlib.contextmanager
def manifest_lock(path: Path = MANIFEST_PATH) -> Iterator[None]:
    """Exclusive lock on ``<manifest>.lock`` (several stages update the manifest)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def update_manifest(fn: Callable[[Manifest], object], path: Path = MANIFEST_PATH) -> Manifest:
    """Re-read the manifest under the lock, apply ``fn(m)`` (in place) and save: no stage
    overwrites another's fields with a stale in-memory copy."""
    path = Path(path)
    with manifest_lock(path):
        m = load_manifest(path) if path.exists() else Manifest(created=_now())
        fn(m)
        m.updated = _now()
        save_manifest(m, path)
    return m


def freeze_manifest(out: Path = MANIFEST_PATH, agents: dict[str, AgentSpec] = AGENTS,
                    existing: Manifest | None = None) -> Manifest:
    """Step 0: list every finished run's snapshots (adding new ones to ``existing``, never
    changing an entry already there), verify each loads and gives a finite forward pass, and
    write the manifest. G3 and S3 enter when they finish.

    ``existing`` defaults to the manifest already at ``out``. Two exceptions to "never
    changed": an entry that failed to load (``error`` set) is retried and replaced, so a
    file caught mid-write does not stay unlisted; and ``is_final`` (derived, not measured)
    follows the run's current last snapshot. Snapshots are only *read*: nothing is rolled
    out, and every new entry has ``evaluated_on_stimuli`` False."""
    out = Path(out)
    if existing is None:
        existing = load_manifest(out) if out.exists() else Manifest(created=_now())
    # Load and forward-pass the new snapshots outside the lock (seconds each) ...
    records, new_entries = {}, {}
    for agent_id, spec in agents.items():
        snaps = list_snapshots(spec)
        records[agent_id] = (_agent_record(spec, snaps), _rel(snaps[-1]) if snaps else None)
        for path in snaps:
            old = existing.entry(_rel(path))
            if old is not None and not old.error:
                continue
            new_entries[_rel(path)] = snapshot_entry(spec, path, is_final=path == snaps[-1])

    # ... and merge them into the manifest as it is on disk now, under the lock.
    def merge(m: Manifest) -> None:
        for agent_id, (rec, final) in records.items():
            m.agents[agent_id] = rec
            for rel, entry in new_entries.items():
                if entry.agent != agent_id:
                    continue
                old = m.entry(rel)
                if old is not None and not old.error:
                    continue                    # another process listed it meanwhile
                if old is not None:
                    m.snapshots[m.snapshots.index(old)] = entry
                else:
                    m.snapshots.append(entry)
            for e in m.snapshots:
                if e.agent == agent_id and final is not None:
                    e.is_final = e.path == final
        m.snapshots.sort(key=lambda e: (e.agent, e.timesteps, e.path))
        m.created = m.created or _now()
        if m.tag:
            # After the tag the tagged commit and code hash stand; later runs are listed.
            m.lists.setdefault("post_tag_manifest_runs", []).append(
                dict(time=_now(), git_commit=_git_commit(), code_hash=code_hash()))
        else:
            m.git_commit = _git_commit()
            m.code_hash = code_hash()

    with manifest_lock(out):
        m = load_manifest(out) if out.exists() else existing
        merge(m)
        m.updated = _now()
        save_manifest(m, out)
    return m


def _pair(v) -> tuple[float, float]:
    def f(x):
        if x is None:
            return float("nan")
        return float(x)

    return (f(v[0]), f(v[1]))


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    d = read_json(path)
    names = {f.name for f in dataclasses.fields(Manifest)}
    snap_names = {f.name for f in dataclasses.fields(SnapshotEntry)}
    snaps = []
    for s in d.get("snapshots", []):
        s = {k: v for k, v in s.items() if k in snap_names}
        s["log_std"], s["std"] = _pair(s["log_std"]), _pair(s["std"])
        snaps.append(SnapshotEntry(**s))
    kw = {k: v for k, v in d.items() if k in names and k != "snapshots"}
    return Manifest(snapshots=snaps, **kw)


def save_manifest(m: Manifest, path: Path = MANIFEST_PATH) -> None:
    write_json(path, m)


def mark_evaluated_locked(snapshot_path: str, stage: str, path: Path = MANIFEST_PATH) -> None:
    """``mark_evaluated`` through ``update_manifest``: called by a rollout job before it
    saves its first group, so a snapshot can never be rolled out without being marked."""
    update_manifest(lambda m: mark_evaluated(m, snapshot_path, stage), path)


def mark_evaluated(m: Manifest, snapshot_path: str, stage: str) -> None:
    """Record the first rollout of a snapshot on a stimulus (rollout stage only)."""
    e = m.entry(snapshot_path) or m.entry(_rel(Path(snapshot_path)))
    if e is None:
        raise KeyError(f"{snapshot_path} is not in the manifest")
    if not e.evaluated_on_stimuli:
        e.evaluated_on_stimuli = True
        e.first_evaluated = f"{_now()} {stage}"
    m.updated = _now()


# ------------------------------------------------------------------------- real laps


@dataclass
class RealLap:
    """A deterministic two-lap run on a real circuit with the evaluation's convention
    (rolling start s = 0, n = 0, v0 = 0.9 v_QSS(0), default car; diagnose_crashes.py)."""

    agent: str
    track: str
    v_start: float                # m/s
    x: np.ndarray                 # [K, 8] states, s unwrapped (laps continue past L)
    prev_action: np.ndarray       # [K, 2]
    accel: np.ndarray             # [K, 2]
    valid: np.ndarray             # [K] bool
    term_step: int                # -1: completed two laps
    cause: str
    length: float                 # m, circuit length


def real_lap(agent: LoadedAgent, track_name: str, laps: int = 2) -> RealLap:
    """Run the evaluation lap and keep every state (``lockbox_guard`` first). Integrate
    phase only: this is an agent rollout on a real R_DEV circuit.

    The convention of scripts/diagnose_crashes.py: ``refine_reference(load_tumftm(name))``,
    rolling start x0 = [0, 0, 0, 0.9 v_QSS(0), 0, ...], default car, deterministic mean
    action, ``laps * 3 L / max(v0, 20) / dt`` steps, through the stimulus harness with a
    one-track bank (so it records what stimulus runs record). term_step is -1 when the run
    did not terminate within those steps."""
    from apexzero.env.adapters import reference_speed
    from apexzero.interp import rollout
    from apexzero.tracks.reference import refine_reference
    from apexzero.tracks.track import load_tumftm
    from apexzero.vehicle.sim import track_arrays
    from apexzero.viz.export import lockbox_guard

    lockbox_guard(track_name)
    assert_x64_off()
    track = refine_reference(load_tumftm(REPO / "data" / "tracks" / "tumftm" / f"{track_name}.csv"))
    L = float(track.length)
    v0 = float(reference_speed(track)[0]) * 0.9
    steps = int(laps * 3 * L / max(v0, 20.0) / racing.EnvParams().dt)
    bank = jax.tree.map(lambda a: jnp.asarray(a)[None], track_arrays(track))
    fn = rollout.make_stimulus_rollout(rollout.agent_policy(agent.cfg), steps, "full")
    batch = rollout.batch_from_states(0, rollout.approach_state(0.0, v0))
    tr = rollout.trace_arrays(fn(agent.actor_params, agent.obs_rms, bank, batch), 1, "full")
    k = int(tr["term_step"][0])
    return RealLap(agent=agent.agent_id, track=track_name, v_start=v0, x=tr["x"][0],
                   prev_action=tr["prev_action"][0], accel=tr["accel"][0],
                   valid=tr["valid"][0], term_step=k,
                   cause=rollout.cause_of(tr["flags"][0][k]) if k >= 0 else "", length=L)


def real_state(lap: RealLap, s_target: float, lap_index: int) -> RealState:
    """The first recorded state with unwrapped s >= (lap_index - 1) L + s_target; if that
    precedes the run's start (crash-lap corners within 800 m of s = 0), the run's initial
    state. Raises if the run terminated before reaching it.

    Only live states count: a terminated run's terminal state (the crash) is never used as
    an initial state. The state's s is returned modulo L (circuit coordinates)."""
    target = (lap_index - 1) * lap.length + s_target
    s = lap.x[:, 0].astype(np.float64)
    live = lap.valid.copy()
    if lap.term_step >= 0:
        live[lap.term_step + 1:] = False      # the terminal state and the frozen copies
    if target <= s[0]:
        k = 0
    else:
        hit = np.nonzero(live & (s >= target))[0]
        if not len(hit):
            raise ValueError(f"{lap.agent} on {lap.track}: the run ends at s = "
                             f"{s[live].max():.0f} m before reaching {target:.0f} m"
                             + (f" ({lap.cause})" if lap.cause else ""))
        k = int(hit[0])
    x = lap.x[k].astype(np.float64).copy()
    x[0] = x[0] % lap.length
    return RealState(agent=lap.agent, track=lap.track, lap=lap_index, s_real=float(x[0]),
                     x=tuple(float(v) for v in x),
                     prev_action=tuple(float(v) for v in lap.prev_action[k]),
                     accel=tuple(float(v) for v in lap.accel[k]))
