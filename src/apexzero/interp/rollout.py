"""The stimulus harness: initial states, phases and the batched closed-loop rollout.

Protocol 4 (initial states, phases, policy modes, recording, batching) and step 2 (harness
checks). One jitted function serves every agent, checkpoint and scripted control: the
policy's parameters and the stimulus bank are jit *arguments*, and each run gathers its
track from the bank by ``track_id`` inside a vmap over (track_id, x0, noise key). Chunks
hold at most 2048 episodes and are grouped by window length W (one compile per W; a short
last chunk is padded to a power-of-two bucket so it does not compile a new shape).

Policy modes (protocol 4): o = obs_rms.normalize(observe(st)) clipped to +-10; deterministic
a = mean; native noise a = mean + exp(log_std) eps; matched noise a = mean + (0.04, 0.08) eps;
the env clips a to [-1, 1]. eps_t = normal(fold_in(run_key, t), (2,)) with run_key =
fold_in(fold_in(PRNGKey(20261019), pair_id), sample), so every snapshot sees the same noise
(common random numbers), and eps_steer is negated on the mirrored member of a pair. After
termination the state is frozen with ``jnp.where``. float64 must be off (``replay`` and
``check`` switch it on globally): keys and rollouts run in separate processes.

Stochastic sample j of a cell starts at phase j mod n_phases, so crash rates average over
the aliasing phases.

Rollout files (``io.rollout_path(agent, timesteps)``), one .npz per snapshot, with three
groups of flat arrays named ``<group>_<field>``; group 'det' (deterministic, full traces),
'nat' (native noise) and 'mat' (matched noise). R runs per group, T = max W in the group
(runs with a shorter W are padded by repeating their last state, ``valid`` False):

    <g>_cell [R] int32, <g>_stim [R] int32, <g>_pair [R] int32, <g>_phase [R] int16,
    <g>_sample [R] int32 (-1 deterministic), <g>_steer_sign [R] int8, <g>_W [R] int16,
    <g>_s0 [R] float64, <g>_term_step [R] int32, <g>_trunc_step [R] int32,
    <g>_valid [R, T+1] bool, <g>_flags [R, T] uint8 (constants.FLAG_BITS),
    <g>_util_f, <g>_util_r, <g>_beta_f [R, T+1] float32 (float16 would round every beta_f
      in (0.99976, 1) to 1.0 and util 0.98 to 0.97998, flipping crash typing's thresholds)
  det only:
    det_x [R, T+1, 8] float32, det_prev_action [R, T+1, 2], det_accel [R, T+1, 2],
    det_a_mean [R, T, 2], det_a [R, T, 2] (before the env's clip), det_reward [R, T],
    det_clip_frac [R, T+1, 4], det_clip_max [R, T+1, 4] (constants.OBS_GROUPS)
  nat / mat only:
    <g>_s [R, T+1] float32, <g>_vx [R, T+1] float32, <g>_n [R, T+1] float16,
    <g>_pedal [R, T+1] float16, <g>_xi [R, T+1] float16 (first-sight xi, protocol 4)
  top level:
    log_std [2] float32, agent str, timesteps int64, code_hash str, created str
    (plus meta_json: every meta entry as JSON, with 'groups': {group: {code_hash, created,
    design_hash, ...}} recording what produced each group)

Conventions fixed here (see docs/interp/IMPLEMENTATION.md):
- ``valid[k]`` is True for every state actually reached, the terminal state included
  (D27); later states are frozen copies. term_step / trunc_step is the step k whose
  transition terminated / truncated (state k + 1 is the terminal state), -1 if none.
  Step k was taken iff ``valid[k + 1]``.
- Per-step arrays of steps not taken (after the end) hold NaN (a, a_mean), 0 (reward,
  flags). ``a`` is the policy's sample before the env's [-1, 1] clip.
- ``nonfinite`` is not reported by ``step_no_reset``; it is inferred as
  term & ~(off_track | spun | stalled) (non-finite states compare False everywhere).
- Clipping statistics use the *unclipped* normalised observation z = (ob - mean) /
  sqrt(var + 1e-8): clip_frac is the fraction of |z| >= 10 (the channels the clip
  saturates) and clip_max is max |z|, so a saturating channel shows how far beyond the clip
  it is. Controls (obs_rms None) record NaN.
- beta_f is ``measures.front_brake_demand`` (looked up when the harness is first traced).
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import Actor, PPOConfig, RunningMeanStd
from apexzero.env import racing
from apexzero.interp.constants import (
    CAUSE_PRECEDENCE,
    CHUNK,
    DT,
    FLAG_BITS,
    MATCHED_NOISE_STD,
    N_PHASES,
    OBS_CLIP,
    OBS_CONTEXT,
    OBS_EGO,
    OBS_GROUPS,
    OBS_PREVIEW,
    PILOT_S0,
    PILOT_S_ENTRY,
    PILOT_STEPS,
    PILOT_V0,
    ROOT_SEED,
    STREAM_OFFSET,
)
from apexzero.interp.io import Table, _atomic, assert_x64_off
from apexzero.interp.stimuli import Cell, Stimulus, StimulusBank
from apexzero.vehicle.dynamics import tyre_forces
from apexzero.vehicle.ops import JAX
from apexzero.vehicle.params import CONTEXT_RANGES, F1_2025
from apexzero.vehicle.sim import TrackArrays

# ------------------------------------------------------------------------- policies


class Policy(Protocol):
    """``policy(params, o, state, tr, aux) -> (mean [2], log_std [2])`` for one run.

    ``o`` is the normalised, clipped observation when the rollout is given an obs_rms
    (agents), else the raw ``racing.observe`` output (controls). ``state`` and ``tr`` are the
    run's EnvState and TrackArrays (controls read the road from them); ``aux`` is the run's
    int32 index into per-run controller tables (tape row for T, target-profile row for Q;
    0 for agents). Controls without native noise return log_std = -inf."""

    def __call__(self, params: Any, o: jax.Array, state: racing.EnvState, tr: TrackArrays,
                 aux: jax.Array) -> tuple[jax.Array, jax.Array]: ...


@functools.cache
def agent_policy(cfg: PPOConfig) -> Policy:
    """``Actor(cfg, 2, False).apply(params, o)`` with the Policy signature (params = the
    snapshot's actor params). Cached per config, so every snapshot of an agent family shares
    one Policy object and therefore one compiled harness."""
    actor = Actor(cfg, racing.ACTION_DIM, False)

    def policy(params, o, state, tr, aux):
        mean, log_std = actor.apply(params, o)
        return mean, log_std

    return policy


def _tape_policy(params, o, state, tr, aux):
    """Open-loop tape: ``params`` [n_rows, steps, 2], mean = params[aux, state.t]; no native
    noise. Used by the mirror gate (controls.make_tape_policy is the protocol's T)."""
    t = jnp.minimum(state.t, params.shape[1] - 1)
    return params[aux, t], jnp.full(2, -jnp.inf, params.dtype)


# m, the lane keeper's speed lookahead (NumPy float32: a jnp constant made at import would
# be float64 whenever another module switched x64 on first)
_KEEPER_D = np.arange(0.0, 240.0, 4.0, dtype=np.float32)
_KEEPER_A_LAT, _KEEPER_A_BRAKE = 12.0, 15.0


def _lane_keeper(params, o, state, tr, aux):
    """A scripted closed-loop driver for harness checks (never an analysed control): steer
    to the centreline with curvature feed-forward; pedal towards min(v_cap, the speed it
    can still brake to every corner within 240 m at 15 m/s^2, cornering at 12 m/s^2).
    ``params`` = (v_cap, log_std [2]). Deliberately slow, so check runs stay alive."""
    from apexzero.vehicle.sim import kappa_at

    v_cap, log_std = params
    s, n, xi, vx = state.x[0], state.x[1], state.x[2], state.x[3]
    p = F1_2025
    k_ahead = kappa_at(s + 0.3 * jnp.maximum(vx, 1.0), tr)
    steer = ((p.lf + p.lr) * k_ahead - 0.05 * n - 0.6 * xi) / p.delta_max
    k = jnp.abs(kappa_at(s + _KEEPER_D, tr))
    v_corner = jnp.sqrt(_KEEPER_A_LAT / jnp.maximum(k, 1e-4))
    v_target = jnp.minimum(v_cap, jnp.min(jnp.sqrt(v_corner**2 + 2 * _KEEPER_A_BRAKE * _KEEPER_D)))
    pedal = 0.4 * (v_target - vx)
    return jnp.clip(jnp.stack([steer, pedal]), -1.0, 1.0), log_std


def _const_policy(params, o, state, tr, aux):
    """mean = params[0] (constant), log_std = params[1] (common-random-number check)."""
    return params[0], params[1]


# ------------------------------------------------------------------------- states, phases


def approach_state(s0: float, v0: float) -> np.ndarray:
    """Synthetic initial state [s0, 0, 0, v0, 0, 0, 0, 0] as float32 (protocol 4)."""
    return np.array([s0, 0.0, 0.0, v0, 0.0, 0.0, 0.0, 0.0], np.float32)


def default_ctx() -> np.ndarray:
    """``racing.default_context()`` as exact float32 (computed with x64 off; protocol 1.4),
    hidden_mu = 1. Any other rounding blows the specialist's context channels to the clip.

    ``default_context`` builds a jnp array from Python floats, so with x64 off each entry is
    the float64 value rounded once to float32; this does the same without touching JAX."""
    p = F1_2025
    cla = p.cla_f + p.cla_r
    return np.array([p.m, p.p_max, cla, p.cda, p.mu_scale, p.cla_f / cla], np.float32)


def init_env_state(x0, prev_action=None, accel=None, ctx=None, hidden_mu: float = 1.0
                   ) -> racing.EnvState:
    """EnvState at reset-like conditions (racing.py has no such helper; the levels
    convention): t = 0, slow = 0, s0 = x0[0], prev_action and accel default to 0, ctx to
    ``default_ctx()``. float32 throughout."""
    f32 = jnp.float32
    x = jnp.asarray(x0, f32)
    pa = jnp.zeros(2, f32) if prev_action is None else jnp.asarray(prev_action, f32)
    ac = jnp.zeros(2, f32) if accel is None else jnp.asarray(accel, f32)
    c = jnp.asarray(default_ctx() if ctx is None else ctx, f32)
    return racing.EnvState(x=x, prev_action=pa, accel=ac, ctx=c,
                           hidden_mu=jnp.asarray(hidden_mu, f32),
                           t=jnp.zeros((), jnp.int32), slow=jnp.zeros((), f32), s0=x[0])


def phases(v_sight: float, n: int = N_PHASES, dt: float = DT) -> np.ndarray:
    """u_k = (k / n) v_sight dt (m), k = 0..n-1 (protocol 4)."""
    return np.arange(n) / n * float(v_sight) * dt


def d_max(vx) -> np.ndarray:
    """Preview horizon clip(6 vx, 150, 500) m (``racing.preview_offsets``)."""
    return np.clip(6.0 * np.maximum(np.asarray(vx, float), 0.0), 150.0, 500.0)


def sight_speed(s: np.ndarray, vx: np.ndarray, s_c0: float) -> float:
    """vx at the first step where the corner start enters the preview (s_c0 - s <=
    d_max(vx)); from the nominal key, else the pinned s-OCP, else the QSS profile (the
    caller passes that trajectory). Raises if the corner never comes into view."""
    s, vx = np.asarray(s, float), np.asarray(vx, float)
    seen = np.nonzero(s_c0 - s <= d_max(vx))[0]
    if not len(seen):
        raise ValueError(f"the corner at s = {s_c0} never enters the preview")
    return float(vx[seen[0]])


def recompute_observations(x: np.ndarray, prev_action: np.ndarray, accel: np.ndarray,
                           tr: TrackArrays, obs_rms: RunningMeanStd | None = None,
                           ctx: np.ndarray | None = None) -> np.ndarray:
    """[K, 222] observations of recorded states (``racing.observe``), normalised and clipped
    when ``obs_rms`` is given. Traces store states, not observations (0.5 GB per 1,000 runs
    otherwise); d_OOD and the aliasing probe recompute them."""
    c = jnp.asarray(default_ctx() if ctx is None else ctx, jnp.float32)
    out = _observe_batch(jnp.asarray(x, jnp.float32), jnp.asarray(prev_action, jnp.float32),
                         jnp.asarray(accel, jnp.float32), c, tr)
    if obs_rms is not None:
        out = obs_rms.normalize(out, clip=OBS_CLIP)
    return np.asarray(out, np.float32)


@jax.jit
def _observe_batch(x, pa, ac, ctx, tr):
    def one(xk, pk, ak):
        st = init_env_state(xk, pk, ak, ctx)
        return racing.observe(st, tr, racing.params_from_context(ctx, hidden_mu=st.hidden_mu))

    return jax.vmap(one)(x, pa, ac)


def noise_key(pair_id: int, sample: int) -> np.ndarray:
    """Raw uint32 [2] key data of fold_in(fold_in(PRNGKey(ROOT_SEED), pair_id), sample)."""
    if not 0 <= pair_id < STREAM_OFFSET:
        raise ValueError(f"noise id {pair_id} outside [0, {STREAM_OFFSET}) (seed streams)")
    if sample < 0:
        raise ValueError("deterministic runs (sample -1) have no noise key")
    k = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(ROOT_SEED), pair_id), sample)
    return np.asarray(k, np.uint32)


def noise_eps(key: np.ndarray, steps: int, steer_sign: float = 1.0) -> np.ndarray:
    """[steps, 2] eps_t of one run as the harness draws it (eps_steer times steer_sign)."""
    k = jnp.asarray(key, jnp.uint32)
    eps = jax.vmap(lambda t: jax.random.normal(jax.random.fold_in(k, t), (2,)))(
        jnp.arange(steps))
    return np.asarray(eps) * np.array([steer_sign, 1.0], np.float32)


# ------------------------------------------------------------------------- batches

MODES = ("det", "nat", "mat")
SYNTHETIC_KINDS = ("synthetic", "pilot", "twin", "straight")


class RolloutBatch(NamedTuple):
    """Inputs of one compiled call; every field has leading axis [B]."""

    track_id: jax.Array      # int32, row of the bank
    x0: jax.Array            # [B, 8] float32 initial state
    prev_action: jax.Array   # [B, 2] float32 a_{-1} (0 for synthetic stimuli)
    accel: jax.Array         # [B, 2] float32 (0 for synthetic stimuli)
    ctx: jax.Array           # [B, 6] float32 car context (default_ctx for every protocol run)
    hidden_mu: jax.Array     # [B] float32 (1.0)
    noise_key: jax.Array     # [B, 2] uint32 run key data (noise_key)
    noise_std: jax.Array     # [B, 2] float32 fixed std: MATCHED_NOISE_STD in 'mat', else 0
    native: jax.Array        # [B] bool: add exp(log_std) eps ('nat')
    steer_sign: jax.Array    # [B] float32 +1, -1 on the mirrored member (negates eps_steer)
    aux: jax.Array           # [B] int32 per-run controller-table row (0 for agents)


class RunSpec(NamedTuple):
    """One run to do (a row of the run index)."""

    cell_id: int
    stim_id: int
    pair_id: int
    phase: int
    sample: int              # -1 deterministic
    mode: str                # 'det' | 'nat' | 'mat'
    steer_sign: int
    W: int
    aux: int = 0


def steer_sign(stim: Stimulus) -> int:
    """Fallback steer sign from a bank record, for cells without ``Cell.mirrored``: synthetic
    rows (with a CornerSpec) take their direction (left +1, right -1); real-road rows +1
    for the original and -1 for the row built as its mirror image (the higher stim_id of the
    two, ``mirror_id``). Either rule gives a mirror pair's members opposite signs."""
    if stim.spec is not None and stim.direction in (-1, 1):
        return int(stim.direction)
    if stim.mirror_id >= 0 and stim.stim_id > stim.mirror_id:
        return -1
    return 1


def cell_steer_sign(cell: Cell, bank: StimulusBank | None = None) -> int:
    """-1 on the mirrored member of a pair (``Cell.mirrored``: its eps_steer is negated,
    protocol 4), else +1; ``steer_sign`` of the bank record for a cell object without the
    field."""
    mirrored = getattr(cell, "mirrored", None)
    if mirrored is not None:
        return -1 if mirrored else 1
    if bank is None:
        raise ValueError(f"cell {cell.cell_id} has no 'mirrored' field and no bank was given")
    return steer_sign(bank.stimuli[cell.stim_id])


def check_pair_signs(cells: list[Cell], bank: StimulusBank | None = None) -> None:
    """Raise if the two cells of a mirror pair (same pair_id) have the same steer sign: their
    noise would then not be mirrored (P5). Pairs with one or more than two cells in the
    list are not judged."""
    by_pair: dict[int, list[Cell]] = {}
    for c in cells:
        by_pair.setdefault(c.pair_id, []).append(c)
    for pid, members in by_pair.items():
        if len(members) == 2 and len({cell_steer_sign(c, bank) for c in members}) != 2:
            raise ValueError(f"pair {pid}: cells {[c.cell_id for c in members]} have the same "
                             "steer sign (exactly one member must be Cell.mirrored)")


def _samples_for(cell: Cell, samples: dict[str, int] | int) -> int:
    """Stochastic samples of one cell: a number, or per tier ('A', 'E' matches 'E1'..'E3',
    'other' for the rest), the largest over the cell's tiers."""
    if isinstance(samples, int):
        return samples
    best = 0
    for t in cell.tiers:
        if t in samples:
            n = samples[t]
        elif t[:1] in samples:
            n = samples[t[:1]]
        else:
            n = samples.get("other", 0)
        best = max(best, int(n))
    return best


def det_phase_indices(n_phases: int, phases_det: int | None) -> list[int]:
    """Phase indices of a cell's deterministic runs: every phase, or ``phases_det`` of them
    spread evenly over the cycle (a 16-phase P7 cell run at 4 phases takes k = 0, 4, 8, 12,
    the 4-phase grid u = 0, P/4, P/2, 3P/4)."""
    if phases_det is None or phases_det >= n_phases:
        return list(range(n_phases))
    step = max(1, n_phases // max(1, int(phases_det)))
    return list(range(0, n_phases, step))[: int(phases_det)]


def plan_runs(cells: list[Cell], bank: StimulusBank, mode: str, samples: dict[str, int] | int,
              phases_det: int | None = None, aux: dict[int, int] | None = None,
              agent: str | None = None) -> list[RunSpec]:
    """Runs of one mode over ``cells``: deterministic runs at every phase (or
    ``phases_det`` of them, ``det_phase_indices``); stochastic sample j at phase j mod
    n_phases. ``samples`` per tier (REPLICATION) or one number (ignored for 'det'). ``aux``
    maps cell_id -> controller-table row (controls; default 0).
    ``agent`` skips cells whose ``runners`` list other agents only. The mirrored member of
    each pair (``Cell.mirrored``) gets steer_sign -1; a pair whose two cells share a sign is
    an error (``check_pair_signs``)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    if agent is not None:
        cells = [c for c in cells if not getattr(c, "runners", ()) or agent in c.runners]
    check_pair_signs(cells, bank)
    runs = []
    for c in cells:
        if c.W is None:
            raise ValueError(f"cell {c.cell_id} has no window W yet (key layout)")
        sign = cell_steer_sign(c, bank)
        a = 0 if aux is None else int(aux[c.cell_id])
        if mode == "det":
            for k in det_phase_indices(c.n_phases, phases_det):
                runs.append(RunSpec(c.cell_id, c.stim_id, c.pair_id, k, -1, mode, sign, c.W, a))
        else:
            for j in range(_samples_for(c, samples)):
                runs.append(RunSpec(c.cell_id, c.stim_id, c.pair_id, j % c.n_phases, j, mode,
                                    sign, c.W, a))
    return runs


INDEX_DTYPES = {"cell": np.int32, "stim": np.int32, "pair": np.int32, "phase": np.int16,
                "sample": np.int32, "steer_sign": np.int8, "W": np.int16, "s0": np.float64}


def make_batch(runs: list[RunSpec], cells: dict[int, Cell]) -> tuple[RolloutBatch, Table]:
    """Batch arrays and the run index table (columns cell, stim, pair, phase, sample,
    steer_sign, W, s0) for a list of runs (one compiled call runs them all for the largest
    W). x0, prev_action and accel come from ``Cell.initial_state`` / the cell's real-state
    fields; ctx is ``default_ctx()``, hidden_mu 1."""
    B = len(runs)
    x0 = np.zeros((B, 8), np.float32)
    pa = np.zeros((B, 2), np.float32)
    ac = np.zeros((B, 2), np.float32)
    keys = np.zeros((B, 2), np.uint32)
    std = np.zeros((B, 2), np.float32)
    native = np.zeros(B, bool)
    for i, r in enumerate(runs):
        c = cells[r.cell_id]
        if c.stim_id != r.stim_id or c.pair_id != r.pair_id:
            raise ValueError(f"run {i} does not match cell {c.cell_id}")
        x0[i] = np.asarray(c.initial_state(r.phase), np.float32)
        pa[i] = c.prev_action0
        ac[i] = c.accel0
        if r.mode != "det":
            keys[i] = noise_key(r.pair_id, r.sample)
        if r.mode == "mat":
            std[i] = MATCHED_NOISE_STD
        native[i] = r.mode == "nat"
    batch = RolloutBatch(
        track_id=np.array([r.stim_id for r in runs], np.int32), x0=x0, prev_action=pa,
        accel=ac, ctx=np.tile(default_ctx(), (B, 1)), hidden_mu=np.ones(B, np.float32),
        noise_key=keys, noise_std=std, native=native,
        steer_sign=np.array([r.steer_sign for r in runs], np.float32),
        aux=np.array([r.aux for r in runs], np.int32))
    table = {"cell": [r.cell_id for r in runs], "stim": [r.stim_id for r in runs],
             "pair": [r.pair_id for r in runs], "phase": [r.phase for r in runs],
             "sample": [r.sample for r in runs], "steer_sign": [r.steer_sign for r in runs],
             "W": [r.W for r in runs], "s0": x0[:, 0].astype(np.float64)}
    table = {k: np.asarray(v, INDEX_DTYPES[k]) for k, v in table.items()}
    return batch, table


def batch_from_states(track_id, x0, *, prev_action=None, accel=None, mode: str = "det",
                      pair_id=None, sample=None, steer_sign=None, aux=None) -> RolloutBatch:
    """A RolloutBatch straight from arrays (pilot replication, real laps, checks): x0 [B, 8],
    optional real-state auxiliaries; noise from (pair_id, sample) for 'nat' / 'mat'."""
    x0 = np.atleast_2d(np.asarray(x0, np.float32))
    B = len(x0)
    tid = np.broadcast_to(np.asarray(track_id, np.int32), (B,)).copy()
    keys = np.zeros((B, 2), np.uint32)
    if mode != "det":
        pid = np.broadcast_to(np.asarray(pair_id), (B,))
        smp = np.broadcast_to(np.asarray(sample), (B,))
        keys = np.stack([noise_key(int(p), int(s)) for p, s in zip(pid, smp, strict=True)])
    std = np.tile(np.asarray(MATCHED_NOISE_STD, np.float32), (B, 1)) * (mode == "mat")
    sign = np.ones(B, np.float32) if steer_sign is None else np.broadcast_to(
        np.asarray(steer_sign, np.float32), (B,)).copy()
    return RolloutBatch(
        track_id=tid, x0=x0,
        prev_action=np.zeros((B, 2), np.float32) if prev_action is None
        else np.broadcast_to(np.asarray(prev_action, np.float32), (B, 2)).copy(),
        accel=np.zeros((B, 2), np.float32) if accel is None
        else np.broadcast_to(np.asarray(accel, np.float32), (B, 2)).copy(),
        ctx=np.tile(default_ctx(), (B, 1)), hidden_mu=np.ones(B, np.float32), noise_key=keys,
        noise_std=std.astype(np.float32), native=np.full(B, mode == "nat"), steer_sign=sign,
        aux=np.zeros(B, np.int32) if aux is None
        else np.broadcast_to(np.asarray(aux, np.int32), (B,)).copy())


# ------------------------------------------------------------------------- the harness


class Trace(NamedTuple):
    """Per-run outputs of one compiled call (leading axis [B]); see the module docstring for
    names, shapes and dtypes. Fields a ``record`` mode does not keep are None."""

    x: jax.Array | None             # [B, T+1, 8]          (full)
    prev_action: jax.Array | None   # [B, T+1, 2]          (full)
    accel: jax.Array | None         # [B, T+1, 2]          (full)
    a_mean: jax.Array | None        # [B, T, 2]            (full)
    a: jax.Array | None             # [B, T, 2]            (full)
    reward: jax.Array | None        # [B, T]               (full)
    clip_frac: jax.Array | None     # [B, T+1, 4]          (full)
    clip_max: jax.Array | None      # [B, T+1, 4]          (full)
    s: jax.Array | None             # [B, T+1]             (stoch)
    vx: jax.Array | None            # [B, T+1]             (stoch)
    n: jax.Array | None             # [B, T+1]             (stoch)
    pedal: jax.Array | None         # [B, T+1]             (stoch)
    xi: jax.Array | None            # [B, T+1]             (stoch)
    util_f: jax.Array               # [B, T+1]
    util_r: jax.Array               # [B, T+1]
    beta_f: jax.Array               # [B, T+1]
    valid: jax.Array                # [B, T+1] bool
    flags: jax.Array                # [B, T] uint8
    term_step: jax.Array            # [B] int32
    trunc_step: jax.Array           # [B] int32


RECORDS = ("full", "stoch")
_KAPPA = np.zeros(racing.OBS_DIM, bool)
_KAPPA[4:OBS_PREVIEW:5] = True
_GROUP_MASKS = np.stack([
    _KAPPA,                                                          # kappa * 50 (40)
    np.r_[~_KAPPA[:OBS_PREVIEW], np.zeros(OBS_EGO + OBS_CONTEXT, bool)],   # edges (160)
    np.r_[np.zeros(OBS_PREVIEW, bool), np.ones(OBS_EGO, bool), np.zeros(OBS_CONTEXT, bool)],
    np.r_[np.zeros(OBS_PREVIEW + OBS_EGO, bool), np.ones(OBS_CONTEXT, bool)],
])
assert _GROUP_MASKS.sum(1).tolist() == [40, 160, OBS_EGO, OBS_CONTEXT] and len(OBS_GROUPS) == 4


def _clip_stats(z: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per OBS_GROUPS: fraction of |z| >= OBS_CLIP and max |z| (unclipped z)."""
    m = jnp.asarray(_GROUP_MASKS)
    az = jnp.abs(z)
    frac = jnp.sum(m & (az >= OBS_CLIP), axis=1) / jnp.sum(m, axis=1)
    mx = jnp.max(jnp.where(m, az, 0.0), axis=1)
    return frac.astype(jnp.float32), mx.astype(jnp.float32)


def _flag_bits(off, slipped, spun, stalled, nonfinite, truncated) -> jax.Array:
    bits = (off.astype(jnp.uint8) << FLAG_BITS["off_track"]
            | slipped.astype(jnp.uint8) << FLAG_BITS["slipped"]
            | spun.astype(jnp.uint8) << FLAG_BITS["spun"]
            | stalled.astype(jnp.uint8) << FLAG_BITS["stalled"]
            | nonfinite.astype(jnp.uint8) << FLAG_BITS["nonfinite"]
            | truncated.astype(jnp.uint8) << FLAG_BITS["truncated"])
    return bits.astype(jnp.uint8)


_CAUSE_BITS = {"corridor": "off_track", "slip": "slipped", "spin": "spun", "stall": "stalled",
               "nonfinite": "nonfinite"}


def cause_of(flags: int) -> str:
    """First termination cause by precedence (corridor > slip > spin > stall > non-finite)
    from one step's FLAG_BITS; '' if none fired (measures.crash_cause is the measures copy)."""
    for cause in CAUSE_PRECEDENCE:
        if int(flags) >> FLAG_BITS[_CAUSE_BITS[cause]] & 1:
            return cause
    return ""


def _episode(policy: Policy, steps: int, record: str, env_params: racing.EnvParams,
             brake_demand: Callable) -> Callable:
    """One run: (policy_params, obs_rms, bank, b) -> dict of per-run outputs."""
    full = record == "full"

    def run(policy_params, obs_rms, bank, b: RolloutBatch):
        tr = jax.tree.map(lambda arr: arr[b.track_id], bank)
        st0 = racing.EnvState(x=b.x0, prev_action=b.prev_action, accel=b.accel, ctx=b.ctx,
                              hidden_mu=b.hidden_mu, t=jnp.zeros((), jnp.int32),
                              slow=jnp.zeros((), b.x0.dtype), s0=b.x0[0])
        p = racing.params_from_context(b.ctx, hidden_mu=b.hidden_mu)
        ob0 = racing.observe(st0, tr, p)
        sign = jnp.stack([b.steer_sign, jnp.ones_like(b.steer_sign)])

        def policy_input(ob):
            if obs_rms is None:
                nan = jnp.full(len(OBS_GROUPS), jnp.nan, jnp.float32)
                return ob, (nan, nan)
            z = (ob - obs_rms.mean) / jnp.sqrt(obs_rms.var + 1e-8)
            return obs_rms.normalize(ob, clip=OBS_CLIP), _clip_stats(z)

        def state_record(st, stats):
            fo = tyre_forces(st.x, p, JAX)
            rec = {"util_f": fo.util_f, "util_r": fo.util_r,
                   "beta_f": brake_demand(st.x, p, JAX)[0]}
            if full:
                rec.update(x=st.x, prev_action=st.prev_action, accel=st.accel,
                           clip_frac=stats[0], clip_max=stats[1])
            else:
                rec.update(s=st.x[0], vx=st.x[3], n=st.x[1], pedal=st.x[7], xi=st.x[2])
            return rec

        def body(carry, t):
            st, o, stats, alive = carry
            mean, log_std = policy(policy_params, o, st, tr, b.aux)
            eps = jax.random.normal(jax.random.fold_in(b.noise_key, t), (2,), mean.dtype) * sign
            std = jnp.where(b.native, jnp.exp(log_std), 0.0) + b.noise_std
            a = mean + std * eps
            ob2, st2, r, term, trunc, info = racing.step_no_reset(st, a, tr, env_params)
            nonfinite = term & ~(info["off_track"] | info["spun"] | info["stalled"])
            flags = _flag_bits(info["off_track"], info["slipped"], info["spun"],
                               info["stalled"], nonfinite, trunc)
            o2, stats2 = policy_input(ob2)
            keep = lambda new, old: jnp.where(alive, new, old)
            st2 = jax.tree.map(keep, st2, st)
            o2 = keep(o2, o)
            stats2 = jax.tree.map(keep, stats2, stats)
            out = state_record(st2, stats2)
            out.update(valid=alive, flags=jnp.where(alive, flags, jnp.uint8(0)),
                       term=alive & term, trunc=alive & trunc)
            if full:
                nan2 = jnp.full(2, jnp.nan, mean.dtype)
                out.update(a_mean=jnp.where(alive, mean, nan2), a=jnp.where(alive, a, nan2),
                           reward=jnp.where(alive, r, 0.0))
            return (st2, o2, stats2, alive & ~term & ~trunc), out

        o0, stats0 = policy_input(ob0)
        rec0 = state_record(st0, stats0)
        _, outs = jax.lax.scan(body, (st0, o0, stats0, jnp.array(True)),
                               jnp.arange(steps, dtype=jnp.int32))

        def first(mask):
            return jnp.where(mask.any(), jnp.argmax(mask), -1).astype(jnp.int32)

        res = {k: jnp.concatenate([jnp.asarray(v)[None], outs[k]]) for k, v in rec0.items()}
        res["valid"] = jnp.concatenate([jnp.array([True]), outs["valid"]])
        res["flags"] = outs["flags"]
        res["term_step"], res["trunc_step"] = first(outs["term"]), first(outs["trunc"])
        if full:
            res.update(a_mean=outs["a_mean"], a=outs["a"], reward=outs["reward"])
        return res

    return run


def _resolve_brake_demand() -> Callable:
    from apexzero.interp import measures

    return measures.front_brake_demand


@functools.lru_cache(maxsize=32)
def _compiled(policy: Policy, steps: int, record: str, params: racing.EnvParams):
    def fn(policy_params, obs_rms, bank, batch):
        run = _episode(policy, steps, record, params, _resolve_brake_demand())
        out = jax.vmap(run, in_axes=(None, None, None, 0))(policy_params, obs_rms, bank, batch)
        fields = {f: out.get(f) for f in Trace._fields}
        return Trace(**fields)

    return jax.jit(fn)


def make_stimulus_rollout(policy: Policy, steps: int, record: str = "full",
                          params: racing.EnvParams = racing.EnvParams()) -> Callable:
    """``jit(fn(policy_params, obs_rms | None, bank: TrackArrays [NB, ...], batch:
    RolloutBatch) -> Trace)`` for ``steps`` control steps (protocol 4). ``record`` is 'full'
    (deterministic runs) or 'stoch'. The env runs with window_steps = 10**9, so only the
    strip end truncates. With ``obs_rms`` None the policy gets raw observations and the
    clipping statistics are NaN. The returned function asserts float64 is off before each
    call (``.jitted`` is the bare jitted function); compiled functions are cached per
    (policy, steps, record, params)."""
    if record not in RECORDS:
        raise ValueError(f"record {record!r} not in {RECORDS}")
    params = params._replace(window_steps=10**9)
    jitted = _compiled(policy, int(steps), record, params)

    def fn(policy_params, obs_rms, bank, batch):
        assert_x64_off()
        return jitted(policy_params, obs_rms, bank, batch)

    fn.jitted = jitted
    return fn


def bank_arrays(bank: StimulusBank | TrackArrays) -> TrackArrays:
    """Stacked float32 TrackArrays of a bank (``StimulusBank.arrays()``), or the arrays
    themselves; cached on the bank object, since every chunk gathers from the same bank."""
    if isinstance(bank, TrackArrays):
        return bank
    cached = getattr(bank, "_rollout_arrays", None)
    if cached is None:
        cached = jax.tree.map(jnp.asarray, bank.arrays())
        try:
            bank._rollout_arrays = cached
        except AttributeError:
            pass
    return cached


def _bucket(n: int, chunk: int) -> int:
    """Padded batch size of a chunk of n runs: the next power of two >= max(n, 16), at most
    ``chunk``. Fewer distinct shapes means fewer compiles."""
    b = 16
    while b < n:
        b *= 2
    return max(min(b, chunk), n)


def _pad_batch(batch: RolloutBatch, size: int) -> RolloutBatch:
    n = len(batch.track_id)
    if size == n:
        return batch
    return RolloutBatch(*(np.concatenate([np.asarray(a), np.repeat(np.asarray(a)[:1], size - n, 0)])
                          for a in batch))


STOCH_F16 = ("n", "pedal", "xi")        # descriptive traces; util and beta_f stay float32


def trace_arrays(trace: Trace, n: int, record: str) -> dict[str, np.ndarray]:
    """First n runs of a Trace as numpy arrays with the file dtypes (module docstring)."""
    out = {}
    for f in Trace._fields:
        v = getattr(trace, f)
        if v is None:
            continue
        a = np.asarray(v)[:n]
        if f == "valid":
            a = a.astype(bool)
        elif f == "flags":
            a = a.astype(np.uint8)
        elif f in ("term_step", "trunc_step"):
            a = a.astype(np.int32)
        elif record == "stoch" and f in STOCH_F16:
            a = a.astype(np.float16)
        else:
            a = a.astype(np.float32)
        out[f] = a
    return out


def _pad_time(arrs: dict[str, np.ndarray], T: int) -> dict[str, np.ndarray]:
    """Pad per-run arrays from their window to T steps: states repeat their last value
    (valid False), per-step arrays get NaN (a, a_mean), 0 (reward, flags)."""
    out = {}
    for k, a in arrs.items():
        if a.ndim < 2:
            out[k] = a
            continue
        per_step = k in ("a_mean", "a", "reward", "flags")
        L = T if per_step else T + 1
        extra = L - a.shape[1]
        if extra <= 0:
            out[k] = a
            continue
        if k == "valid":
            pad = np.zeros((a.shape[0], extra), bool)
        elif per_step:
            fill = np.nan if k in ("a_mean", "a") else 0
            pad = np.full((a.shape[0], extra) + a.shape[2:], fill, a.dtype)
        else:
            pad = np.repeat(a[:, -1:], extra, axis=1)
        out[k] = np.concatenate([a, pad], axis=1)
    return out


def run_rollouts(policy: Policy, policy_params: Any, obs_rms: RunningMeanStd | None,
                 bank: StimulusBank | TrackArrays, cells: list[Cell], runs: list[RunSpec],
                 record: str, chunk: int = CHUNK, log=None) -> dict[str, np.ndarray]:
    """Chunked execution of ``runs`` (grouped by W, at most ``chunk`` per call), returning
    the flat arrays of one npz group (without the group prefix), rows in the order of
    ``runs``. Asserts x64 off. ``log`` is an ``io.StageLog`` (or anything with ``step(n,
    note=...)``) advanced per chunk."""
    assert_x64_off()
    if not runs:
        raise ValueError("no runs")
    if chunk > CHUNK:
        raise ValueError(f"chunks hold at most {CHUNK} episodes")
    cell_map = {c.cell_id: c for c in cells}
    arrays = bank_arrays(bank)
    T = max(r.W for r in runs)
    by_w: dict[int, list[int]] = {}
    for i, r in enumerate(runs):
        by_w.setdefault(int(r.W), []).append(i)
    parts: list[tuple[np.ndarray, dict]] = []
    for W in sorted(by_w):
        fn = make_stimulus_rollout(policy, W, record)
        idx = by_w[W]
        for start in range(0, len(idx), chunk):
            sel = idx[start:start + chunk]
            batch, table = make_batch([runs[i] for i in sel], cell_map)
            padded = _pad_batch(batch, _bucket(len(sel), chunk))
            trace = fn(policy_params, obs_rms, arrays, padded)
            arrs = _pad_time(trace_arrays(trace, len(sel), record), T)
            parts.append((np.asarray(sel), {**table, **arrs}))
            if log is not None:
                log.step(len(sel), note=f"W={W}")
    order = np.argsort(np.concatenate([p[0] for p in parts]), kind="stable")
    keys = parts[0][1].keys()
    return {k: np.concatenate([p[1][k] for p in parts])[order] for k in keys}


def save_rollouts(path: Path, groups: dict[str, dict[str, np.ndarray]], meta: dict) -> Path:
    """Write the per-snapshot .npz (module docstring layout), atomically. ``meta`` needs
    log_std, agent and timesteps; code_hash and created are filled in when missing."""
    from apexzero.interp.io import _jsonable, code_hash

    meta = dict(meta)
    meta.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
    if "code_hash" not in meta:
        meta["code_hash"] = code_hash()
    # Per group: what produced it (a file can hold groups from several runs of the stage,
    # and a caller re-saving a loaded file passes the first group's meta).
    gm = dict(meta.get("groups", {}) or {})
    new = [g for g in groups if g not in gm]
    if new:
        now = {"code_hash": code_hash(), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if "design_hash" in meta:
            now["design_hash"] = meta["design_hash"]
        for g in new:
            gm[g] = dict(now)
    meta["groups"] = gm
    missing = {"log_std", "agent", "timesteps"} - set(meta)
    if missing:
        raise ValueError(f"rollout meta lacks {sorted(missing)}")
    arrays: dict[str, np.ndarray] = {}
    for g, cols in groups.items():
        if g not in MODES:
            raise ValueError(f"group {g!r} not in {MODES}")
        for k, v in cols.items():
            arrays[f"{g}_{k}"] = np.asarray(v)
    arrays["log_std"] = np.asarray(meta["log_std"], np.float32).reshape(2)
    arrays["agent"] = np.asarray(str(meta["agent"]))
    arrays["timesteps"] = np.asarray(int(meta["timesteps"]), np.int64)
    arrays["code_hash"] = np.asarray(str(meta["code_hash"]))
    arrays["created"] = np.asarray(str(meta["created"]))
    arrays["meta_json"] = np.asarray(json.dumps(_jsonable(meta), sort_keys=True))
    return _atomic(Path(path), lambda fh: np.savez_compressed(fh, **arrays))


def load_rollouts(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """(groups, meta) of a rollout file."""
    groups: dict[str, dict[str, np.ndarray]] = {}
    with np.load(Path(path), allow_pickle=False) as d:
        meta = json.loads(str(d["meta_json"])) if "meta_json" in d.files else {}
        for k in d.files:
            g, _, field = k.partition("_")
            if g in MODES and field:
                groups.setdefault(g, {})[field] = d[k]
        meta.update(log_std=d["log_std"], agent=str(d["agent"]), timesteps=int(d["timesteps"]),
                    code_hash=str(d["code_hash"]), created=str(d["created"]))
    return groups, meta


def matched_std() -> np.ndarray:
    return np.asarray(MATCHED_NOISE_STD, np.float32)


# ------------------------------------------------------------------------- step-2 checks

_MIRROR = np.array([1, -1, -1, 1, -1, -1, -1, 1], np.float32)


def _env_params() -> racing.EnvParams:
    return racing.EnvParams()._replace(window_steps=10**9)


def _diff(a: np.ndarray, b: np.ndarray, relative: bool) -> float:
    """max |a - b|, or max |a - b| / max(1, |b|) (float32 states: s ~ 1000 m has an ulp of
    6e-5 m, so an absolute 1e-6 is below the precision of the s channel)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not a.size:
        return 0.0
    d = np.abs(a - b)
    if relative:
        d = d / np.maximum(1.0, np.abs(b))
    return float(d.max())


def check_step_loop(policy: Policy, policy_params: Any, obs_rms: RunningMeanStd | None,
                    bank: StimulusBank | TrackArrays, cell: Cell, steps: int = 400,
                    phase: int = 0, aux: int = 0, relative: bool = True) -> float:
    """Max |state difference| between the harness's mean-action run and a plain Python loop
    of ``racing.step_no_reset`` (must be <= 1e-6). Use a scripted control or a randomly
    initialised network before the tag. The loop stops at the first termination or
    truncation; both must end on the same step (else inf). ``relative`` scales each
    difference by max(1, |x|): the harness is vmapped and the loop is not, so XLA may round
    differently by an ulp (an ulp of vx = 80 m/s is 7.6e-6)."""
    arrays = bank_arrays(bank)
    run = RunSpec(cell.cell_id, cell.stim_id, cell.pair_id, phase, -1, "det", 1, steps, aux)
    batch, _ = make_batch([run], {cell.cell_id: cell})
    tr = jax.tree.map(lambda a: jnp.asarray(a)[cell.stim_id], arrays)
    trace = trace_arrays(make_stimulus_rollout(policy, steps, "full")(
        policy_params, obs_rms, arrays, batch), 1, "full")

    env = _env_params()
    st = init_env_state(batch.x0[0], batch.prev_action[0], batch.accel[0], batch.ctx[0])
    p = racing.params_from_context(st.ctx, hidden_mu=st.hidden_mu)
    step = jax.jit(lambda s_, a_: racing.step_no_reset(s_, a_, tr, env))
    act = jax.jit(lambda o_, s_: policy(policy_params, o_, s_, tr, jnp.int32(aux))[0])
    ob = jax.jit(racing.observe)(st, tr, p)
    xs, end = [np.asarray(st.x)], -1
    for k in range(steps):
        o = ob if obs_rms is None else obs_rms.normalize(ob, clip=OBS_CLIP)
        ob, st, _, term, trunc, _ = step(st, act(o, st))
        xs.append(np.asarray(st.x))
        if bool(term) or bool(trunc):
            end = k
            break
    ref = np.stack(xs)
    stop = trace["term_step"][0] if trace["term_step"][0] >= 0 else trace["trunc_step"][0]
    if stop != end:
        return float("inf")
    return _diff(trace["x"][0][: len(ref)], ref, relative)


def check_batch_equals_single(policy: Policy, policy_params: Any,
                              obs_rms: RunningMeanStd | None, bank: StimulusBank | TrackArrays,
                              cells: list[Cell], steps: int = 200, mode: str = "det",
                              samples: int = 1, isolated: bool = True,
                              relative: bool = True) -> float:
    """Max difference between a batched call and the same runs done one at a time (every
    phase of every cell; stochastic modes with ``samples`` samples each), over states, util
    and beta_f of every state reached; inf if a run ends on a different step.

    ``isolated`` (the default, an exact test): run i is redone alone among decoys, in a
    batch of the same shape where every other row holds a different run (run i + 1's
    inputs), and must match the full batch's row i bit for bit (returns 0.0). This is what
    "runs do not interact" means, without float noise. ``isolated=False`` redoes each run
    in a batch of one: XLA vectorises a different shape differently, so the two agree only
    to float32 rounding, and closed-loop gains carry an ulp of vx into the pedal (about
    4e-6 after 400 steps of the lane keeper); ``relative`` as in ``check_step_loop``."""
    arrays = bank_arrays(bank)
    cell_map = {c.cell_id: c for c in cells}
    runs = []
    for c in cells:
        sign = cell_steer_sign(c, bank if hasattr(bank, "stimuli") else None)
        if mode == "det":
            runs += [RunSpec(c.cell_id, c.stim_id, c.pair_id, k, -1, mode, sign, steps)
                     for k in range(c.n_phases)]
        else:
            runs += [RunSpec(c.cell_id, c.stim_id, c.pair_id, j % c.n_phases, j, mode, sign,
                             steps) for j in range(samples)]
    if len(runs) < 2:
        raise ValueError("need at least two runs")
    fn = make_stimulus_rollout(policy, steps, "full")
    batch, _ = make_batch(runs, cell_map)
    whole = trace_arrays(fn(policy_params, obs_rms, arrays, batch), len(runs), "full")
    worst = 0.0
    for i, r in enumerate(runs):
        if isolated:
            decoy = runs[(i + 1) % len(runs)]
            b1, _ = make_batch([r if j == i else decoy for j in range(len(runs))], cell_map)
            one = {k: v[i:i + 1] for k, v in trace_arrays(
                fn(policy_params, obs_rms, arrays, b1), len(runs), "full").items()}
        else:
            b1, _ = make_batch([r], cell_map)
            one = trace_arrays(fn(policy_params, obs_rms, arrays, b1), 1, "full")
        if one["term_step"][0] != whole["term_step"][i]:
            return float("inf")
        v = one["valid"][0]
        for f in ("x", "util_f", "beta_f", "a"):
            m = v if one[f].shape[1] == len(v) else v[1:]     # step k taken iff valid[k + 1]
            worst = max(worst, _diff(one[f][0][m], whole[f][i][m], relative))
    return worst


def _cell_x0(cell: Cell, phase: int = 0) -> np.ndarray:
    return np.asarray(cell.initial_state(phase), np.float32)


def mirror_gate(bank: StimulusBank | TrackArrays, pair: tuple[Cell, Cell], steps: int = 400,
                seed: int = ROOT_SEED) -> float:
    """P5 harness gate: an action sequence on a stimulus and the steer-negated sequence on
    its mirror give mirrored states to 1e-5 over 400 steps (also with mirrored noise).

    The sequence is the scripted lane keeper's closed-loop run on the first member (so the
    car stays on the road for the whole window) plus a seeded random steering wobble. It
    is then replayed open-loop: deterministic, and in the matched-noise mode with the
    pair's common random numbers (eps_steer negated on the mirror). Returns the largest
    |x_first - mirror(x_second)| over both modes (inf if they end on different steps)."""
    arrays = bank_arrays(bank)
    c1, c2 = pair
    x01, x02 = _cell_x0(c1), _cell_x0(c2)
    if np.max(np.abs(x01 * _MIRROR - x02)) > 0:
        raise ValueError("the pair's initial states are not mirror images")
    v_target = float(max(x01[3], 30.0))
    keeper = make_stimulus_rollout(_lane_keeper, steps, "full")
    kp = (jnp.float32(v_target), jnp.full(2, -jnp.inf, jnp.float32))
    base = trace_arrays(keeper(kp, None, arrays, batch_from_states(c1.stim_id, x01)), 1, "full")
    acts = np.nan_to_num(base["a_mean"][0], nan=0.0)
    rng = np.random.default_rng(seed)
    t = np.arange(steps) * DT
    wobble = sum(rng.uniform(0.005, 0.02) * np.sin(2 * np.pi * rng.uniform(0.1, 1.0) * t
                                                   + rng.uniform(0, 2 * np.pi)) for _ in range(3))
    acts[:, 0] = np.clip(acts[:, 0] + wobble, -1.0, 1.0)
    tape = np.stack([acts, acts * np.array([-1.0, 1.0])]).astype(np.float32)
    fn = make_stimulus_rollout(_tape_policy, steps, "full")
    worst = 0.0
    for mode in ("det", "mat"):
        b = batch_from_states([c1.stim_id, c2.stim_id], np.stack([x01, x02]), mode=mode,
                              pair_id=c1.pair_id, sample=0, steer_sign=[1.0, -1.0],
                              aux=[0, 1])
        tr = trace_arrays(fn(jnp.asarray(tape), None, arrays, b), 2, "full")
        if tr["term_step"][0] != tr["term_step"][1] or tr["trunc_step"][0] != tr["trunc_step"][1]:
            return float("inf")
        v = tr["valid"][0]
        d = np.abs(tr["x"][0][v].astype(np.float64) - tr["x"][1][v] * _MIRROR)
        worst = max(worst, float(d.max()))
    return worst


def check_common_noise(bank: StimulusBank | TrackArrays, cells: list[Cell],
                       steps: int = 40, samples: int = 2, tol: float = 1e-4,
                       signs: dict[int, int] | None = None) -> bool:
    """Common random numbers: the same (pair, sample) gives identical eps for any params;
    the mirrored member gets -eps_steer and the same eps_pedal; and the harness draws
    exactly ``noise_eps(noise_key(pair, sample))``. Checked with a constant-mean policy at
    two parameter settings in native-noise mode: eps = (a - mean) / exp(log_std) on every
    step taken (``tol`` absorbs the float32 rounding of that quotient). ``signs`` (cell_id
    -> steer sign) overrides ``cell_steer_sign``."""
    arrays = bank_arrays(bank)
    fn = make_stimulus_rollout(_const_policy, steps, "full")
    signs = signs or {c.cell_id: cell_steer_sign(c, bank if hasattr(bank, "stimuli") else None)
                      for c in cells}
    settings = ((np.array([0.0, 0.3], np.float32), np.log(np.array([0.05, 0.1], np.float32))),
                (np.array([0.02, 0.6], np.float32), np.log(np.array([0.2, 0.02], np.float32))))
    for c in cells:
        x0 = np.repeat(_cell_x0(c)[None], samples, 0)
        b = batch_from_states(c.stim_id, x0, mode="nat", pair_id=c.pair_id,
                              sample=np.arange(samples), steer_sign=signs[c.cell_id])
        for params in settings:
            tr = trace_arrays(fn(params, None, arrays, b), samples, "full")
            for j in range(samples):
                taken = tr["valid"][j][1:]                # step k taken iff valid[k + 1]
                eps = (tr["a"][j] - tr["a_mean"][j]) / np.exp(params[1])
                want = noise_eps(noise_key(c.pair_id, j), steps, signs[c.cell_id])
                if not taken[0] or np.max(np.abs(eps[taken] - want[taken])) > tol:
                    return False
    for members in _group_pairs(cells).values():
        if len(members) != 2:
            continue
        a, b = members
        if signs[a.cell_id] == signs[b.cell_id]:
            return False
        for j in range(samples):
            ea = noise_eps(noise_key(a.pair_id, j), steps, signs[a.cell_id])
            eb = noise_eps(noise_key(b.pair_id, j), steps, signs[b.cell_id])
            if not (np.array_equal(ea[:, 0], -eb[:, 0]) and np.array_equal(ea[:, 1], eb[:, 1])):
                return False
    return True


def _group_pairs(cells: list[Cell]) -> dict[int, list[Cell]]:
    out: dict[int, list[Cell]] = {}
    for c in cells:
        out.setdefault(c.pair_id, []).append(c)
    return out


CONTEXT_TOL = 1e-3


def check_context(snapshots: dict[str, RunningMeanStd]) -> dict[str, float]:
    """S1's normalised default context within 1e-3 of 0, and default_context inside the
    generalists' CONTEXT_RANGES; returns max |normalised context| per agent. Raises
    ValueError when a check fails. Every specialist ('S*') is held to the 1e-3 bound (all
    three trained on the default car only)."""
    ctx = default_ctx()
    lo = np.array([v[0] for v in CONTEXT_RANGES.values()])
    hi = np.array([v[1] for v in CONTEXT_RANGES.values()])
    if not np.all((lo <= ctx) & (ctx <= hi)):
        raise ValueError(f"default context {ctx} outside CONTEXT_RANGES")
    st = init_env_state(approach_state(0.0, 50.0), ctx=ctx)
    raw = jnp.asarray(2.0 * (st.ctx - racing._CTX_LO) / (racing._CTX_HI - racing._CTX_LO) - 1.0)
    out = {}
    for agent, rms in snapshots.items():
        sl = slice(OBS_PREVIEW + OBS_EGO, OBS_PREVIEW + OBS_EGO + OBS_CONTEXT)
        z = (raw - jnp.asarray(rms.mean)[sl]) / jnp.sqrt(jnp.asarray(rms.var)[sl] + 1e-8)
        out[agent] = float(jnp.max(jnp.abs(z)))
        if agent.startswith("S") and out[agent] > CONTEXT_TOL:
            raise ValueError(f"{agent}: normalised default context reaches {out[agent]:.3g} "
                             f"> {CONTEXT_TOL} (context rounding?)")
    return out


def benchmark(policy: Policy, policy_params: Any, obs_rms: RunningMeanStd | None,
              bank: StimulusBank | TrackArrays, batch_size: int = 2048, steps: int = 800,
              record: str = "full", repeats: int = 1) -> float:
    """Steps per second at B = 2048, T = 800 with 2 XLA threads (step 2): the bank's rows
    in turn, deterministic, from [100, 0, 0, 60, ...]; compile excluded (timed separately,
    printed). XLA threads are set by the calling process (scripts/psychophysics.py)."""
    arrays = bank_arrays(bank)
    nb = int(jnp.asarray(arrays.ds).shape[0])
    batch = batch_from_states(np.arange(batch_size) % nb,
                              np.repeat(approach_state(100.0, 60.0)[None], batch_size, 0))
    fn = make_stimulus_rollout(policy, steps, record).jitted
    t0 = time.perf_counter()
    compiled = fn.lower(policy_params, obs_rms, arrays, batch).compile()
    t_compile = time.perf_counter() - t0
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(policy_params, obs_rms, arrays, batch))
        best = min(best, time.perf_counter() - t0)
    sps = batch_size * steps / best
    print(f"benchmark B={batch_size} T={steps} {record}: compile {t_compile:.1f} s, "
          f"run {best:.1f} s, {sps:,.0f} steps/s", flush=True)
    return sps


# ------------------------------------------------------------------------- pilot replication

PILOT_EXPECTED = {"G1": frozenset(), "G2": frozenset({(9.0, 180.0, -1)})}


def _pilot_key(stim: Stimulus) -> tuple[float, float, int]:
    """(R, theta_deg, direction) of a pilot mini-grid row."""
    if stim.spec is not None:
        return float(stim.spec.R), float(stim.spec.theta_deg), int(stim.spec.direction)
    parts = {p[0]: p[1:] for p in stim.name.split("/") if p[:1] in ("R", "T") and p[1:2].isdigit()}
    return float(parts["R"]), float(parts["T"]), int(stim.direction)


def pilot_verdict(agent_id: str, results: dict[tuple[float, float, int], bool]) -> bool:
    """True if the crashed set matches the pilot: G1 none of 24, G2 only R9/180/right."""
    crashed = frozenset(k for k, v in results.items() if v)
    return len(results) == 24 and crashed == PILOT_EXPECTED[agent_id]


def pilot_replication(agent_id: str) -> dict:
    """Step 2's blocking check (one of the protocol's two allowed pre-tag agent rollouts;
    Integrate phase only): G1 completes 24/24 pilot mini-grid cells and G2 crashes only
    R9/180/right. Returns {cell name: (crashed, cause, v_entry, s_crash, v_crash, v_min)}
    and 'match': bool.

    The pilot's own layout (D19): every cell starts at [100, 0, 0, 80, 0, ...], 500
    deterministic steps; v_entry is vx where s first reaches the corner start (800 m),
    NaN if never reached; s_crash and v_crash are the terminal state's s and vx (NaN when
    the run did not terminate); v_min is the lowest vx over the states reached."""
    from apexzero.interp import agents, stimuli

    if agent_id not in PILOT_EXPECTED:
        raise ValueError("the pilot replication covers G1 and G2 only")
    path = agents.final_snapshot(agents.AGENTS[agent_id])
    if path is None:
        raise FileNotFoundError(f"{agent_id} has no final snapshot")
    ag = agents.load_agent(path, agent_id)
    bank, cells = stimuli.pilot_minigrid()
    stim_ids = [c.stim_id for c in cells]
    x0 = np.repeat(approach_state(PILOT_S0, PILOT_V0)[None], len(cells), 0)
    fn = make_stimulus_rollout(agent_policy(ag.cfg), PILOT_STEPS, "full")
    tr = trace_arrays(fn(ag.actor_params, ag.obs_rms, bank_arrays(bank),
                         batch_from_states(stim_ids, x0)), len(cells), "full")
    out: dict = {}
    keyed: dict[tuple[float, float, int], bool] = {}
    for i, c in enumerate(cells):
        stim = bank.stimuli[c.stim_id]
        k = int(tr["term_step"][i])
        crashed = k >= 0
        cause = cause_of(tr["flags"][i][k]) if crashed else ""
        s, vx, v = tr["x"][i][:, 0], tr["x"][i][:, 3], tr["valid"][i]
        hit = np.nonzero(v & (s >= PILOT_S_ENTRY))[0]
        v_entry = float(np.interp(PILOT_S_ENTRY, s[hit[0] - 1:hit[0] + 1],
                                  vx[hit[0] - 1:hit[0] + 1])) if len(hit) and hit[0] > 0 \
            else float("nan")
        s_crash = float(s[k + 1]) if crashed else float("nan")
        v_crash = float(vx[k + 1]) if crashed else float("nan")
        out[stim.name] = (crashed, cause, v_entry, s_crash, v_crash, float(vx[v].min()))
        keyed[_pilot_key(stim)] = crashed
    out["match"] = pilot_verdict(agent_id, keyed)
    return out
