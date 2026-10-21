"""The network's side of step 2: a float64 closed loop, its outcome VJPs, and Gates 2 and 3
(design Sections 3, 5 'Network' and 8).

**A float64 process.** Everything here that evaluates a network needs x64 on
(``require_x64``). Step 1's harness asserts x64 off (rollout.py:640-650, io.py:241), so the
two never share a process: ``python -m apexzero.fields.network dry-run ...`` switches x64 on
at its start; ``float32_reference`` (the frozen harness, for branch consistency) runs in an
x64-off process. Module constants are NumPy, so importing this module is safe in either.

**The loop** (``closed_loop``) is the harness's deterministic step (``_episode``,
rollout.py:547-612) on T64-style inputs (ruling R20, rollout.py:795-848): exact float64
upcasts of the float32 values the agents' harness used (``upcast32``: each floating leaf is
rounded to float32, as the harness's jit saw it, then widened exactly). That covers the
stimulus row's TrackArrays (``perturb.upcast_row``, rollout.py:808), the actor params and
obs_rms of the snapshot, and ``default_ctx()`` (rollout.py:179) with hidden_mu = 1. Per step:

    o    = clip((observe(st) - mean) / sqrt(var + 1e-8), -10, 10)   RunningMeanStd.normalize
    mean = actor(o)                                                  deterministic mode
    st'  = racing.step_no_reset(st, mean, tr, EnvParams(window_steps=10**9))   clips to [-1, 1]

(racing.py:146-187; ``_env_params``, rollout.py:912). One deviation from computing everything
in float64: the observation's six context channels (racing.py:119) are constant, and they are
set to the float32 harness's values, upcast (``context_channels``). A specialist's normaliser
divides them by sqrt(var + 1e-8) with var ~ 0, so float64 arithmetic on them would move its
input by ~1e-3 (rollout.py:179-187 makes the same point about the context itself).

The scan has a fixed length (``STEPS`` = 256). Step 1's stored runs need up to 180 steps from
x_{L-1} on the anchored finals and up to 210 on the A2 set to reach the outcome window's end
(the optimum's plans 70-199), so 256 leaves a margin of at least ``HORIZON_MARGIN`` = 40 steps
(``loop_audit`` checks it on the stored runs). After the step that terminates, truncates, or
first reaches the outcome window's end s_stop = s_c1 + 50 m (``outcomes.window``), the state is
frozen with a **safe double-where**: the next step is computed from the start state instead of
the frozen one, and its result is discarded, so its derivatives stay finite. ``Trace.valid``
follows the harness convention (rollout.py:46-51): state k + 1 is valid iff step k was taken;
frozen copies are invalid, so ``measures._View`` (measures.py:829) cuts the run there. A run
still alive at its last step without reaching s_stop has **exhausted the loop** (``Run.exhausted``):
its plans carry the reason 'loop length exhausted' (never 'window end not reached'), which
``measures.coverage`` treats as a pipeline failure, not as the network's exclusion; ``fields``
and ``branch_consistency`` refuse such a run (``LoopExhausted``) unless the caller reads only
the decision step's steer (``allow_exhausted``: the instantaneous fields' 1-step loops).

**Two road arrays** (``closed_loop(..., tr_obs=...)``): the observation (and a scripted
policy's own reads of the road) uses ``tr_obs``, the dynamics, termination and the outcome
readers use ``tr``; by default they are the same arrays and the loop is unchanged.
``path_fields`` perturbs one path at a time, with the base arrays elsewhere: the observation's
kappa channel (kappa_c), its pose (x, y, theta: every body-frame edge point), its widths, and
the dynamics (``tr``: control_step's kappa_c, termination, and the outcome's own reading of the
road, c_min's edge). The parts sum to the full VJP (``PATH_TOL`` = 1e-9; design Section 4).

**Restarts** (design Section 5). An A2 key's state (``a2_start``) is the cell's designed x0
at the key's phase (``Cell.initial_state``, stimuli.py:372; the key's x0 is that float32 state
cast, asserted bit for bit) with prev_action and accel from ``Cell.initial_aux`` (0). Its
decision is at loop step 0 and s_1 = x0[0]. An anchored key (``anchored_start``) restarts
from its agent's deterministic run in the det group of the key's ``anchor['snapshot']`` file
(io.rollout_path, rollout.py:23-43), at the row of the key's cell and phase. Under option A
(the default) the loop starts at x_{L-1} with det_prev_action[L-1] and det_accel[L-1], and
step 0 applies the agent's a_{L-1} = clip(det_a[L-1]) (the key's ``anchor['a_prev']``,
answer_key.py:1671, asserted equal); then the policy drives. The key's x0 must equal
det_x[L-1] bit for bit, and s_1 = det_x[L, 0] must lie within 1e-3 m of the key's x[1, 0].
Option 'L' starts at det_x[L], det_prev_action[L], det_accel[L]. Always slow = 0 (vx > 2 m/s
asserted) and t = 0.

**Outcomes and fields.** The base run's plans (``outcomes.plans``: window, validity, argmin
and onset picks, near-ties) fix the picks; the JAX readers then give c_min, v_apex and s_b as
functions of the closed loop's states (Danskin), and ``steer`` is the policy's steer mean at
the decision step (the instantaneous field; design Section 3). ``fields`` takes one
``jax.vjp`` of this outcome vector with respect to the three float64 sample arrays (kappa,
w_left, w_right) through ``perturb.perturbed_arrays`` (the strip_arrays tangent, so pose,
kappa_c and the widths move together) and pulls back the four unit cotangents with a vmap:
per-sample impulse responses of every outcome in one pass.

**Gate 2** (``gate2``): AD against central finite differences along given directions. A
comparison counts only when the -h, -h/2, 0, +h/2 and +h runs share their discrete structure
(``Structure``: the env clip's saturated (step, channel) pairs, the obs clip's clipped
channels, the observation's own kinks, the argmin pick, the onset run, the end of the run)
and FD(h) and FD(h/2) agree to O(h^2) (their difference within ``RICH_RTOL`` of the
Richardson value, or at the round-off floor). h = 0.1 x the smallest kink radius upstream of
the outcome (``kink_radius``: each kink quantity's margin over its AD tangent, one JVP),
capped. The error is the field-vector relative error ||AD - FD_R|| / ||FD_R|| over the
directions that count; pass < 1e-3. The 2 m knots of the pose lookups (``sim.lookup``) are
kinks too, of size kappa ds in heading; they are left out of the radius (there are thousands
per run, each tiny), and the O(h^2) check catches any that matter.

**Gate 3** (``wiring_check``, teacher-forced): at every step of a stored float32 run, the
float64 actor mean on the stored state (with the stored prev_action and accel) must match
det_a_mean to 1e-4, and one float64 env step from (det_x[t], det_a[t]) must match det_x[t+1]:
s within substeps x ulp32(s) / 2, the other states within ``STEP_TOL`` (provisional bounds
from the float32/float64 spread of random-actor and linear-readout runs, to be recalibrated
in the dry run). The next state's prev_action must be clip(det_a[t]) exactly.
``branch_consistency`` compares a free-running float64 run with the float32 file's run in
what the design allows: the end before the window (termination step), the saturation set,
the outcome events (validity, argmin pick, onset run) and the outcome values within a
tolerance. s is never compared step by step.

**Policies** (``NetPolicy``): a LoadedAgent's actor (``agent_net``), a random-init actor of
the same architecture (``random_net``; step 1's ``_random_actor`` recipe,
scripts/psychophysics.py:537), a linear readout of the raw observation (``linear_net``), or
any callable with the harness's Policy signature (rollout.py:107; ``callable_net``).

**Exclusion flags.** ``kappa_clipped`` (H2''s confirmatory set and the coverage reason
'kappa clipping'): any preview kappa channel z[:, 5k + 4] at the obs clip |z| >= 10 over the
observations the outcome reads (``_obs_end``: the steps before its read end; the decision
step's own for steer), on the float64 base run. NaN-safe: raw-observation controls (z = NaN)
are never clipped. This is step 2's rule; step 1's binding pair-level rule is 'before s_c1' on
float32 det runs (interp/measures.py:1157, analysis.kappa_clipped_pairs), which does not exist
for networks without step 1 rollouts (H4, H4-transfer); ``kappa_clip_steps`` also reports the
'before s_c1' window so both are recorded.

**Blind-pedal control** (``blind_pedal_net``, H1-typical's paired control, design Section 7):
the network with its pedal replaced by its own base run's pedal tape (``Run.mean[:, 1]``, so it
exists for every network); the steer stays the network's.

**Pre-registration** (design Sections 8 and 10): before the tag no field, gain, derivative or
finite difference of a trained network is computed. Trained snapshots enter only Gate 3's
teacher-forced wiring check (no derivatives, no perturbations). ``agent_net`` marks its
policy ``trained``; ``fields``, ``path_fields``, ``kink_radius``, ``gate2``,
``tangent_consistency`` and a perturbed ``run`` refuse a trained policy (PermissionError) until
``unlock_trained`` is called, which scripts/fields.py does only after its freeze check. The
base run (delta None) and ``wiring_check`` stay open (Gate 3, branch consistency).

**Lockbox**: ``road`` applies step 1's guard to the stimulus (``answer_key._guard``,
answer_key.py:297), as ``keys.rebuild`` and ``kkt.problem`` do; R_TEST circuits never reach the
loop.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import Actor, PPOConfig, RunningMeanStd
from apexzero.env import racing
from apexzero.fields import measures as MS
from apexzero.fields import outcomes as O
from apexzero.fields import perturb as PT
from apexzero.interp import answer_key as ak
from apexzero.interp import io, rollout
from apexzero.interp import measures as M
from apexzero.interp.constants import BRAKE_ON, OBS_CLIP, OBS_CONTEXT, OBS_EGO, OBS_PREVIEW
from apexzero.interp.rollout import Policy, agent_policy, default_ctx
from apexzero.interp.stimuli import Cell, Landmarks, StimulusBank
from apexzero.vehicle.dynamics import tyre_forces
from apexzero.vehicle.ops import JAX
from apexzero.vehicle.params import CONTEXT_RANGES
from apexzero.vehicle.sim import TrackArrays

STEPS = 256                       # loop length: stored horizons up to 180 (anchored) / 210 (A2)
HORIZON_MARGIN = 40                # steps a stored horizon must leave free (``loop_audit``)
EXHAUSTED = MS.LOOP_EXHAUSTED        # plan reason: alive at the loop's end, s_stop not reached
PATH_TOL = 1e-9                    # path_fields: the parts sum to the total (relative)
ZERO_ATOL = 1e-6                   # Gate 2: a field below this on both sides is a zero field
ENV = racing.EnvParams()._replace(window_steps=10**9)   # rollout._env_params, rollout.py:912
CTX = slice(OBS_PREVIEW + OBS_EGO, OBS_PREVIEW + OBS_EGO + OBS_CONTEXT)   # context channels
OUTCOMES = (*O.OUTCOMES, "steer")  # the outcome vector: c_min, v_apex, s_b, steer
MIN_VX = 2.0                       # m/s, a restart must be above the stall speed (slow = 0)
S1_TOL = 1e-3                      # m, |det_x[L, 0] - key x[1, 0]| (design Section 3, 'Names')
MEAN_TOL = 1e-4                    # Gate 3: |float64 actor mean - det_a_mean|
GATE2_TOL = 1e-3                   # Gate 2: field-vector relative error
RICH_RTOL = 1e-4                   # Gate 2: |FD(h) - FD(h/2)| within this of the Richardson value
KINK_FRACTION = 0.1                # Gate 2: h = 0.1 x the smallest kink radius
H_MIN = 1e-9                       # Gate 2: below this, round-off: the comparison does not count
H_MAX = {"kappa": 1e-5, "width": 1e-3}   # Gate 2: cap on h per unit max |direction| (1/m, m)
# Gate 3: one float64 step from a stored float32 state against the stored next state, per
# state: |dx| <= STEP_TOL for n, xi, vx, vy, r, delta, pedal (s has its own rule, see
# ``wiring_check``). Provisional: about 10x the largest spread on 4 random-actor, 4
# linear-readout and 5 G2 rows (n 9e-7 m, xi 9e-7, vx 1.9e-5 m/s, vy 6e-7 m/s, r 2.3e-7 1/s,
# delta 3.9e-8 rad, pedal 1.9e-7, accel 3.7e-4 m/s^2), to be recalibrated on the gate keys.
STEP_TOL = np.array([np.nan, 1e-5, 1e-5, 2e-4, 1e-5, 3e-6, 5e-7, 2e-6])
ACCEL_TOL = 5e-3                   # m/s^2, the trailing-step acceleration (dt = 0.05 amplifies)
# Branch consistency: outcome values within this of the float32 file's (provisional; the
# design's rule is 5x the float32/float64 spread on the gate keys. Seen so far: the linear
# driver 1e-5 m (c_min), 9e-6 m/s (v_apex), 7e-4 m (s_b); the lane keeper 1.6e-6 m, 7e-5 m/s,
# 3.4e-3 m over 114-190 steps).
BRANCH_TOL = {"c_min": 1e-3, "v_apex": 1e-3, "s_b": 1e-2}
BRANCH_FACTOR = 5.0                # design Section 8: 5 x the float32/float64 spread
DET_FIELDS = ("cell", "stim", "phase", "sample", "W", "x", "prev_action", "accel", "a_mean", "a",
              "valid", "term_step", "trunc_step")

# ------------------------------------------------------------------------- float64 inputs


def require_x64() -> None:
    """Raise unless x64 is on (a float64 process; never step 1's harness process)."""
    if jax.dtypes.canonicalize_dtype(jnp.float64) != jnp.float64:
        raise RuntimeError("the float64 loop runs in its own process with x64 on "
                           "(python -m apexzero.fields.network), never beside the harness")


def upcast32(tree):
    """Every floating leaf rounded to float32 (what the harness's jit saw) and widened exactly
    to float64 (``rollout._upcast``, rollout.py:798); other leaves unchanged. NumPy leaves."""
    def up(a):
        a = np.asarray(a)
        if np.issubdtype(a.dtype, np.floating):
            return a.astype(np.float32).astype(np.float64)
        return a

    return jax.tree.map(up, tree)


def ctx64(ctx: np.ndarray | None = None) -> np.ndarray:
    """``default_ctx()`` (float32, rollout.py:179) upcast exactly."""
    return np.asarray(default_ctx() if ctx is None else ctx, np.float32).astype(np.float64)


def context_channels(ctx: np.ndarray | None = None) -> np.ndarray:
    """The observation's context channels as the float32 harness computes them, upcast
    exactly: racing.py:119, 2 (ctx - lo) / (hi - lo) - 1 on float32 constants, compiled by
    XLA with x64 off. XLA folds hi - lo and divides by multiplying with its reciprocal, so
    NumPy's IEEE arithmetic would differ by an ulp (the default car's mass channel is
    -2.2e-8, not 0); the same jit gives the same bits in a float32 and a float64 process."""
    c = np.asarray(default_ctx() if ctx is None else ctx, np.float32)
    return np.array(_context_channels(tuple(c.tolist())))


@functools.lru_cache(maxsize=64)
def _context_channels(ctx: tuple[float, ...]) -> tuple[float, ...]:
    lo = np.array([v[0] for v in CONTEXT_RANGES.values()], np.float32)
    hi = np.array([v[1] for v in CONTEXT_RANGES.values()], np.float32)
    with jax.enable_x64(False):
        f = jax.jit(lambda c_: 2.0 * (c_ - jnp.asarray(lo)) / (jnp.asarray(hi) - jnp.asarray(lo))
                    - 1.0)
        out = np.asarray(f(jnp.asarray(np.array(ctx, np.float32))))
    return tuple(out.astype(np.float32).astype(np.float64).tolist())


# ------------------------------------------------------------------------- policies


@dataclass(frozen=True, eq=False)
class NetPolicy:
    """A policy for the float64 loop. ``fn`` has the harness's Policy signature
    (rollout.py:107; only the mean is used); ``params`` and ``obs_rms`` are float64 exact
    upcasts; ``obs_rms`` None feeds the raw observation (controls). ``trained`` marks a
    trained snapshot (``agent_net``): its fields wait for the tag (``unlock_trained``)."""

    fn: Policy
    params: Any
    obs_rms: RunningMeanStd | None
    name: str = ""
    trained: bool = False


class LoopExhausted(RuntimeError):
    """A run alive at the loop's last step that never reached s_stop: a pipeline failure (the
    loop is too short), never the network's own exclusion."""


_TRAINED_OK = False


def unlock_trained() -> None:
    """Allow fields of trained networks in this process. scripts/fields.py calls it only after
    its freeze check (the tag ``receptive-fields-prereg-v1`` at HEAD, the code unchanged)."""
    global _TRAINED_OK
    _TRAINED_OK = True


def _refuse_trained(net: NetPolicy, what: str) -> None:
    if net.trained and not _TRAINED_OK:
        raise PermissionError(f"{what} of the trained network {net.name} waits for the tag "
                              "(design Sections 8 and 10: no trained network's field, gain or "
                              "finite difference before it)")


def agent_net(agent, obs_rms: RunningMeanStd | None = None) -> NetPolicy:
    """A ``agents.LoadedAgent``'s actor (``rollout.agent_policy``) with its own obs_rms, or
    ``obs_rms`` (Gate 3's planted wrong normaliser). Marked ``trained``."""
    rms = agent.obs_rms if obs_rms is None else obs_rms
    return NetPolicy(agent_policy(agent.cfg), upcast32(agent.actor_params), upcast32(rms),
                     f"{agent.agent_id}@{agent.timesteps}", trained=True)


def random_actor_params(cfg: PPOConfig, seed: int = 0):
    """Float32 params of a randomly initialised actor with ``cfg``'s architecture: step 1's
    ``_random_actor`` (scripts/psychophysics.py:537), built with x64 off so that a float32
    and a float64 process draw the same weights. NumPy leaves."""
    with jax.enable_x64(False):
        actor = Actor(cfg, racing.ACTION_DIM, False)
        params = actor.init(jax.random.PRNGKey(seed), jnp.zeros(racing.OBS_DIM, jnp.float32))
        return jax.tree.map(np.asarray, params)


def random_net(cfg: PPOConfig, obs_rms: RunningMeanStd, seed: int = 0) -> NetPolicy:
    """A random-init actor (never a trained one) normalised by ``obs_rms`` (read only)."""
    return NetPolicy(agent_policy(cfg), upcast32(random_actor_params(cfg, seed)),
                     upcast32(obs_rms), f"random-{seed}")


def linear_policy(params, o, state, tr, aux):
    """mean = W o + b on the raw observation (Policy signature, rollout.py:107; no native
    noise). ``params`` = (W [2, 222], b [2]), or row-stacked (W [R, 2, 222], b [R, 2]) with
    the run's ``aux`` picking the row (one harness call, several readouts)."""
    W, b = params
    if W.ndim == 3:
        W, b = W[aux], b[aux]
    return W @ o + b, jnp.full(2, -jnp.inf, o.dtype)


def linear_net(W: np.ndarray, b: np.ndarray, name: str = "linear") -> NetPolicy:
    """A linear readout of the raw observation (``linear_policy``); W, b are rounded to
    float32 first, so the float32 harness and this loop read the same weights."""
    return NetPolicy(linear_policy, upcast32((np.asarray(W), np.asarray(b))), None, name)


def callable_net(fn: Policy, params: Any, obs_rms: RunningMeanStd | None = None,
                 name: str = "callable") -> NetPolicy:
    """Any callable with the Policy signature (scripted controllers, planted policies)."""
    return NetPolicy(fn, upcast32(params), None if obs_rms is None else upcast32(obs_rms), name)


def _preview(k: int, channel: int) -> int:
    """Observation index of preview point k (0..39) channel (0 lx, 1 ly, 2 rx, 3 ry, 4 kappa*50;
    racing.py:97-107)."""
    return 5 * int(k) + int(channel)


KAPPA_CHANNELS = np.array([_preview(k, 4) for k in range(racing.N_PREVIEW)])


@functools.lru_cache(maxsize=32)
def blind_pedal_policy(fn: Policy) -> Policy:
    """``fn`` with its pedal mean replaced by a tape (params = (fn's params, tape [S])): the
    command at loop step t is tape[min(t, S - 1)], as ``gates._tape_policy`` indexes it. One
    closure per ``fn`` (so the loop's jit cache is shared)."""
    def policy(params, o, state, tr, aux):
        inner, tape = params
        mean, log_std = fn(inner, o, state, tr, aux)
        t = jnp.minimum(state.t, tape.shape[0] - 1)
        return mean.at[1].set(tape[t].astype(mean.dtype)), log_std

    return policy


def blind_pedal_net(net: NetPolicy, tape: np.ndarray) -> NetPolicy:
    """H1-typical's blind-pedal control (design Section 7): ``net`` steering, its pedal the
    constant tape (the network's own float64 base run's pedal means, ``Run.mean[:, 1]``). On
    the base road the run reproduces the network's own base run bit for bit."""
    tape = np.asarray(tape, np.float64)
    return NetPolicy(blind_pedal_policy(net.fn), (net.params, tape), net.obs_rms,
                     f"{net.name}+pedal-tape", trained=net.trained)


def steer_tape_policy(params, o, state, tr, aux):
    """A planted policy whose steering reads no road: steer = tape[min(t, S - 1)], pedal =
    w . o + b on the raw observation (params = (tape [S], w [222], b)). The blind-pedal
    control's known answer: with the pedal on a tape too nothing reads the road but the
    dynamics, so every observation path's field vanishes and v_apex's and s_b's width fields
    are exactly 0."""
    tape, w, b = params
    t = jnp.minimum(state.t, tape.shape[0] - 1)
    steer = tape[t].astype(o.dtype)
    return jnp.stack([steer, w @ o + b]).astype(o.dtype), jnp.full(2, -jnp.inf, o.dtype)


def linear_driver(sigma: int, steer_k: int = 7, steer_gain: float = 1.3, ff_k: int = 2,
                  ff_gain: float = 0.16, pedal0: float = 11.2, speed_gain: float = 8.0,
                  kappa_gain: float = 5.3, kappa_ks: tuple[int, int] = (3, 15)
                  ) -> tuple[np.ndarray, np.ndarray]:
    """A planted closed-loop linear readout of the raw observation that drives some A2 cells
    with D = 250 m through the corner (cells 344/345, R 44.7 m / 45 deg, in 122 steps; 62/63
    and 66/67 in 154-159): the dry run's control with valid plan-level outcomes (a random
    actor does not survive to s_c1; from the A2 starts 40-65 m before the corner the readout
    crashes). Steer: pure pursuit to the centre of preview
    point ``steer_k`` (body-frame y of the left and right edges) plus curvature feed-forward
    at point ``ff_k``. Pedal: pedal0 - speed_gain vx / 50 - kappa_gain sigma (mean kappa * 50
    over preview points kappa_ks), so it brakes when the corner comes into view; sigma is the
    corner's direction, which makes the readout mirror-symmetric. Float32 values."""
    W = np.zeros((2, racing.OBS_DIM))
    W[0, _preview(steer_k, 1)] += steer_gain
    W[0, _preview(steer_k, 3)] += steer_gain
    W[0, _preview(ff_k, 4)] += ff_gain
    W[1, OBS_PREVIEW] = -speed_gain                       # vx / 50, the first ego channel
    ks = range(*kappa_ks)
    for k in ks:
        W[1, _preview(k, 4)] -= int(sigma) * kappa_gain / len(ks)
    return W.astype(np.float32), np.array([0.0, pedal0], np.float32)


# ------------------------------------------------------------------------- restarts


@dataclass(frozen=True, eq=False)
class Start:
    """Where the loop starts and what it reads (design Section 5, 'Restart')."""

    x0: np.ndarray            # [8] float64 (an exact float32 value)
    prev_action: np.ndarray   # [2]
    accel: np.ndarray         # [2]
    a_forced: np.ndarray      # [n, 2] commands applied before the policy (option A: a_{L-1})
    decision_step: int        # loop step of the free decision (0; 1 under option A)
    s1: float                 # m, the bumps' origin s_1
    stim_id: int
    sigma: int                # the corner's direction: +1 left, -1 right
    landmarks: Landmarks
    offset: int = 0           # the file run's state index of loop state 0 (L - 1 under option A)
    source: str = ""

    @property
    def s_stop(self) -> float:
        """The outcome window's end s_c1 + 50 m: the loop freezes after reaching it."""
        return float(O.window(self.landmarks)[1])


def _check_vx(x0: np.ndarray, what: str) -> None:
    if not float(x0[3]) > MIN_VX:
        raise ValueError(f"{what}: vx = {float(x0[3]):.3f} m/s is not above {MIN_VX} (slow = 0 "
                         "would not hold)")


def cell_start(cell: Cell, phase: int, bank: StimulusBank) -> Start:
    """The designed start of a cell at a phase (A2 and any designed cell): x0, prev_action and
    accel as the harness's ``make_batch`` takes them (rollout.py:421-422), decision at step 0."""
    x0 = np.asarray(cell.initial_state(phase), np.float32).astype(np.float64)
    pa, ac = (np.asarray(a, np.float32).astype(np.float64) for a in cell.initial_aux(phase))
    _check_vx(x0, f"cell {cell.cell_id} phase {phase}")
    st = bank.stimuli[cell.stim_id]
    return Start(x0=x0, prev_action=pa, accel=ac, a_forced=np.zeros((0, 2)), decision_step=0,
                 s1=float(x0[0]), stim_id=int(cell.stim_id), sigma=int(st.direction),
                 landmarks=st.landmarks, offset=0, source=f"cell {cell.cell_id} phase {phase}")


def a2_start(rec, bank: StimulusBank, cells: dict[int, Cell]) -> Start:
    """An A2 (nominal or phase) key's state: the cell's start at the key's phase, which must be
    the key's x0 bit for bit (keys cast the float32 harness state, stimuli.py:372-381)."""
    if rec.kind == "anchored":
        raise ValueError(f"key {rec.key_id} is anchored: use anchored_start")
    start = cell_start(cells[rec.cell_id], int(rec.phase), bank)
    if not np.array_equal(start.x0, np.asarray(rec.x0, np.float64)):
        raise ValueError(f"key {rec.key_id}: x0 is not cell {rec.cell_id}'s state at phase "
                         f"{rec.phase}")
    if int(start.stim_id) != int(rec.stim_id):
        raise ValueError(f"key {rec.key_id}: stimulus {rec.stim_id} is not the cell's")
    return start


def load_det(path: Path, fields: tuple[str, ...] = DET_FIELDS) -> dict[str, np.ndarray]:
    """The det group of a rollout file (rollout.py:23-43), only ``fields`` (each array is
    decompressed once; ``rollout.load_rollouts`` would also read the stochastic groups)."""
    with np.load(Path(path), allow_pickle=False) as d:
        return {f: d[f"det_{f}"] for f in fields if f"det_{f}" in d.files}


def det_row(det: dict[str, np.ndarray], cell_id: int, phase: int) -> int:
    """The deterministic run of (cell, phase) in a det group; exactly one must exist."""
    hit = np.flatnonzero((det["cell"] == int(cell_id)) & (det["phase"] == int(phase)))
    if "sample" in det:
        hit = hit[det["sample"][hit] == -1]
    if len(hit) != 1:
        raise ValueError(f"{len(hit)} deterministic runs of cell {cell_id} phase {phase}")
    return int(hit[0])


def anchored_start(rec, det: dict[str, np.ndarray], bank: StimulusBank, option: str = "A"
                   ) -> Start:
    """An anchored key's restart from its agent's deterministic run (``det``: the det group
    of the file of the key's ``anchor['snapshot']``). Option 'A' starts at x_{L-1} and applies
    a_{L-1} at step 0 (decision at step 1); option 'L' starts at x_L (decision at step 0)."""
    if rec.kind != "anchored":
        raise ValueError(f"key {rec.key_id} is {rec.kind}, not anchored")
    if option not in ("A", "L"):
        raise ValueError(f"option {option!r} is not 'A' or 'L'")
    i = det_row(det, rec.cell_id, rec.phase)
    if int(det["stim"][i]) != int(rec.stim_id):
        raise ValueError(f"key {rec.key_id}: row {i} drove stimulus {det['stim'][i]}")
    L = int(rec.anchor["step"])
    x = np.asarray(det["x"][i], np.float32).astype(np.float64)
    if not det["valid"][i][L]:
        raise ValueError(f"key {rec.key_id}: the run did not reach step {L}")
    if not np.array_equal(x[L - 1], np.asarray(rec.x0, np.float64)):
        raise ValueError(f"key {rec.key_id}: x0 is not det_x[{L - 1}] of row {i}")
    a_prev = np.clip(np.asarray(det["a"][i][L - 1], np.float32).astype(np.float64), -1.0, 1.0)
    if not np.array_equal(a_prev, np.asarray(rec.anchor["a_prev"], np.float64)):
        raise ValueError(f"key {rec.key_id}: a_prev is not clip(det_a[{L - 1}]) of row {i}")
    s1 = float(x[L, 0])
    if abs(s1 - float(np.asarray(rec.x)[1, 0])) > S1_TOL:
        raise ValueError(f"key {rec.key_id}: s_1 = {s1} is not x[1, 0] = {rec.x[1][0]}")
    pa = np.asarray(det["prev_action"][i], np.float32).astype(np.float64)
    ac = np.asarray(det["accel"][i], np.float32).astype(np.float64)
    k0, forced, dstep = (L - 1, a_prev[None], 1) if option == "A" else (L, np.zeros((0, 2)), 0)
    _check_vx(x[k0], f"key {rec.key_id} restart")
    st = bank.stimuli[rec.stim_id]
    return Start(x0=x[k0], prev_action=pa[k0], accel=ac[k0], a_forced=forced, decision_step=dstep,
                 s1=s1, stim_id=int(rec.stim_id), sigma=int(st.direction),
                 landmarks=st.landmarks, offset=k0,
                 source=f"{rec.agent}@{rec.anchor['snapshot']} row {i} L={L} option {option}")


# ------------------------------------------------------------------------- the loop


class LoopIn(NamedTuple):
    """The loop's array inputs (a jit argument)."""

    x0: Any                   # [8]
    prev_action: Any          # [2]
    accel: Any                # [2]
    a_forced: Any             # [steps, 2], rows >= n_forced unused
    n_forced: Any             # int
    s_stop: Any               # m
    ctx: Any                  # [6] physical context
    ctx_obs: Any              # [6] the float32 harness's context channels, upcast


def loop_inputs(start: Start, steps: int = STEPS) -> LoopIn:
    n = len(start.a_forced)
    if n > steps:
        raise ValueError("more forced commands than steps")
    forced = np.zeros((steps, 2))
    forced[:n] = start.a_forced
    return LoopIn(np.asarray(start.x0, np.float64), np.asarray(start.prev_action, np.float64),
                  np.asarray(start.accel, np.float64), forced, np.int32(n),
                  np.float64(start.s_stop), ctx64(), context_channels())


class Trace(NamedTuple):
    """One closed-loop run (S = steps)."""

    x: Any        # [S+1, 8] states; frozen after the end
    valid: Any    # [S+1] bool: reached (state 0, and state k + 1 iff step k was taken)
    mean: Any     # [S, 2] policy mean (pre-clip) at the step's input state
    a: Any        # [S, 2] command given to the env (the mean, or a forced command), pre-clip
    z: Any        # [S, 222] unclipped normalised observation (NaN on raw observations)
    term: Any     # [S] bool: step k terminated (state k + 1 is terminal)
    trunc: Any    # [S] bool: step k truncated (end of strip)
    stop: Any     # [S] bool: step k first reached s >= s_stop (alive)


def env_state(x, prev_action, accel, ctx) -> racing.EnvState:
    """EnvState at restart conditions in x's dtype: t = 0, slow = 0, hidden_mu = 1, s0 = x[0]
    (``rollout.init_env_state`` without its float32 cast, rollout.py:190-202)."""
    dt = x.dtype
    return racing.EnvState(x=x, prev_action=prev_action, accel=accel, ctx=ctx,
                           hidden_mu=jnp.ones((), dt), t=jnp.zeros((), jnp.int32),
                           slow=jnp.zeros((), dt), s0=x[0])


def observe(st: racing.EnvState, tr: TrackArrays, ctx_obs):
    """``racing.observe`` with the context channels set to ``ctx_obs`` (module docstring)."""
    p = racing.params_from_context(st.ctx, hidden_mu=st.hidden_mu)
    return racing.observe(st, tr, p).at[CTX].set(ctx_obs)


def policy_input(ob, obs_rms):
    """(o, z): the normalised, clipped observation the policy reads (``normalize``,
    ppo.py:282) and the unclipped z (rollout.py:556-561); raw ob and NaN without obs_rms."""
    if obs_rms is None:
        return ob, jnp.full_like(ob, jnp.nan)
    z = (ob - obs_rms.mean) / jnp.sqrt(obs_rms.var + 1e-8)
    return obs_rms.normalize(ob, clip=OBS_CLIP), z


def closed_loop(fn: Policy, params, obs_rms, tr: TrackArrays, li: LoopIn, steps: int = STEPS,
                tr_obs: TrackArrays | None = None) -> Trace:
    """The deterministic closed loop (module docstring), traceable: jit, vjp and jvp go
    through it. Step k applies ``li.a_forced[k]`` for k < n_forced, else the policy mean.
    ``tr_obs`` (default ``tr``) is the road the observation and the policy read; ``tr`` is
    the dynamics' (``racing.step_no_reset``: control_step, termination)."""
    tro = tr if tr_obs is None else tr_obs
    st0 = env_state(li.x0, li.prev_action, li.accel, li.ctx)
    aux = jnp.zeros((), jnp.int32)

    def keep(alive, new, old):
        return jax.tree.map(lambda a, b: jnp.where(alive, a, b), new, old)

    def body(carry, t):
        st, alive = carry
        safe = keep(alive, st, st0)                 # double-where: never step a frozen state
        o, z = policy_input(observe(safe, tro, li.ctx_obs), obs_rms)
        mean = fn(params, o, safe, tro, aux)[0]
        a = jnp.where(t < li.n_forced, li.a_forced[t], mean)
        _, nxt, _, term, trunc, _ = racing.step_no_reset(safe, a, tr, ENV)
        reached = nxt.x[0] >= li.s_stop
        new = keep(alive, nxt, st)
        out = dict(x=new.x, valid=alive, mean=mean, a=a, z=z, term=alive & term,
                   trunc=alive & trunc, stop=alive & reached & ~term & ~trunc)
        return (new, alive & ~term & ~trunc & ~reached), out

    _, outs = jax.lax.scan(body, (st0, jnp.array(True)), jnp.arange(steps))
    return Trace(x=jnp.concatenate([li.x0[None], outs["x"]]),
                 valid=jnp.concatenate([jnp.array([True]), outs["valid"]]), mean=outs["mean"],
                 a=outs["a"], z=outs["z"], term=outs["term"], trunc=outs["trunc"],
                 stop=outs["stop"])


def _loop_on(fn, steps, params, obs_rms, base, samples, li, delta):
    tr = PT.perturbed_arrays(base, samples, delta)
    return tr, closed_loop(fn, params, obs_rms, tr, li, steps)


@functools.lru_cache(maxsize=16)
def _run_jit(fn: Policy, steps: int, mode: str = "tangent"):
    """jit((params, obs_rms, base, samples, li, delta) -> (TrackArrays, Trace)): the loop on
    the base arrays plus the strip_arrays tangent (or secant) along delta."""
    def run(params, obs_rms, base, samples, li, delta):
        tr = PT.perturbed_arrays(base, samples, delta, mode=mode)
        return tr, closed_loop(fn, params, obs_rms, tr, li, steps)

    return jax.jit(run)


# ------------------------------------------------------------------------- runs and outcomes


def _first(mask: np.ndarray) -> int:
    hit = np.flatnonzero(mask)
    return int(hit[0]) if len(hit) else -1


@dataclass
class Run:
    """A loop run as NumPy, with its outcome plans."""

    start: Start
    x: np.ndarray             # [S+1, 8]
    valid: np.ndarray         # [S+1]
    mean: np.ndarray          # [S, 2]
    a: np.ndarray             # [S, 2]
    z: np.ndarray             # [S, 222]
    term_step: int            # -1: none
    trunc_step: int
    stop_step: int            # the step that reached s_stop (-1: not reached)
    plans: dict[str, O.Plan]
    arrays: TrackArrays       # the run's road (NumPy float64)

    @property
    def n_steps(self) -> int:
        """Steps taken."""
        return int(self.valid[1:].sum())

    @property
    def exhausted(self) -> bool:
        """Alive at the loop's last step with no termination, truncation or stop: the loop was
        too short for this run (module docstring)."""
        return bool(self.valid[-1] and self.term_step < 0 and self.trunc_step < 0
                    and self.stop_step < 0)

    @property
    def steer(self) -> float:
        """The steer mean at the decision step (NaN if that step was not taken)."""
        d = self.start.decision_step
        return float(self.mean[d, 0]) if d < self.n_steps else float("nan")

    @property
    def traj(self) -> M.Trajectory:
        return O.trajectory(self.x, self.valid, self.term_step, source="network")

    def values(self) -> np.ndarray:
        """[4] outcome values (NaN where not valid), in ``OUTCOMES`` order."""
        out = [self.plans[o].value if self.plans[o].valid else np.nan for o in O.OUTCOMES]
        return np.array([*out, self.steer])

    def valid_outcomes(self) -> np.ndarray:
        return np.array([self.plans[o].valid for o in O.OUTCOMES] + [np.isfinite(self.steer)])


def as_run(start: Start, tr: TrackArrays, trc: Trace) -> Run:
    """NumPy view of a loop run plus its plans (``outcomes.plans`` on the run's own road). A
    run that exhausted the loop gets EXHAUSTED as the reason of every plan the window end
    invalidated (a pipeline failure, never 'window end not reached')."""
    t = jax.tree.map(np.asarray, trc)
    arrays = jax.tree.map(np.asarray, tr)
    term, trunc, stop = _first(t.term), _first(t.trunc), _first(t.stop)
    traj = O.trajectory(t.x, t.valid, term, source="network")
    plans = O.plans(traj, start.landmarks, start.sigma, arrays)
    r = Run(start=start, x=t.x, valid=t.valid.astype(bool), mean=t.mean, a=t.a, z=t.z,
            term_step=term, trunc_step=trunc, stop_step=stop, plans=plans, arrays=arrays)
    if r.exhausted:
        r.plans = {k: (dataclasses.replace(v, reason=EXHAUSTED) if not v.valid else v)
                   for k, v in plans.items()}
    return r


@dataclass(frozen=True, eq=False)
class Road:
    """The road a start drives on: the exact float64 upcast of the agents' float32 arrays and
    the float32-cast samples the tangent linearises at (``perturb.upcast_row``,
    ``perturb.samples32``)."""

    base: TrackArrays
    samples: PT.Samples

    @property
    def n(self) -> int:
        return len(self.samples.kappa)


def road(bank: StimulusBank, stim_id: int) -> Road:
    """The road of a stimulus, through step 1's lockbox guard (answer_key.py:297)."""
    ak._guard(bank.stimuli[int(stim_id)])
    return Road(PT.upcast_row(bank, stim_id), PT.samples32(bank, stim_id))


def run(net: NetPolicy, rd: Road, start: Start, delta: PT.Samples | None = None,
        steps: int = STEPS, mode: str = "tangent") -> Run:
    """One closed-loop run on the road plus ``delta`` (tangent by default), analysed. A
    perturbed run of a trained network waits for the tag (``unlock_trained``)."""
    require_x64()
    if delta is not None:
        _refuse_trained(net, "a perturbed run")
    d = PT.zeros(rd.n) if delta is None else delta
    tr, trc = _run_jit(net.fn, int(steps), mode)(net.params, net.obs_rms, rd.base, rd.samples,
                                                  loop_inputs(start, steps), d)
    return as_run(start, tr, trc)


class Picks(NamedTuple):
    """The base run's picks as arrays (a jit argument): invalid outcomes get a dummy pick."""

    c_min: O.Pick
    v_apex: O.Pick
    s_b: O.OnsetPick
    decision: Any             # int
    sigma: Any                # +1 / -1


def _pick(plan: O.Plan) -> O.Pick:
    if plan.valid:
        p = plan.pick
        return O.Pick(np.int32(p.i0), np.int32(p.i1), np.float64(p.s_star), np.bool_(p.interp))
    return O.Pick(np.int32(0), np.int32(0), np.float64(0.0), np.bool_(False))


def picks_of(base: Run) -> Picks:
    sb = base.plans["s_b"]
    k = int(sb.pick.k) if sb.valid else 0
    return Picks(_pick(base.plans["c_min"]), _pick(base.plans["v_apex"]),
                 O.OnsetPick(np.int32(k), np.float64(BRAKE_ON)),
                 np.int32(base.start.decision_step), np.int32(base.start.sigma))


def outcome_vector(x, mean, tr: TrackArrays, picks: Picks, p) -> jax.Array:
    """[c_min, v_apex, s_b, steer] read at the base run's picks (outcomes' JAX readers)."""
    return jnp.stack([O.c_min(x, tr, picks.sigma, picks.c_min), O.v_apex(x, picks.v_apex),
                      O.s_b(x, picks.s_b, p), mean[picks.decision, 0]])


@functools.lru_cache(maxsize=16)
def _fields_jit(fn: Policy, steps: int):
    p = O.drive_params()

    def vec(params, obs_rms, base, samples, li, picks, delta):
        tr, trc = _loop_on(fn, steps, params, obs_rms, base, samples, li, delta)
        return outcome_vector(trc.x, trc.mean, tr, picks, p)

    def fields(params, obs_rms, base, samples, li, picks, delta0):
        y, pull = jax.vjp(lambda d: vec(params, obs_rms, base, samples, li, picks, d), delta0)
        imp = jax.vmap(pull)(jnp.eye(len(OUTCOMES), dtype=y.dtype))[0]
        return y, imp

    return jax.jit(fields)


@dataclass
class Fields:
    """Per-sample impulse responses of the outcome vector at one start (design Section 4).
    Invalid outcomes have NaN values and NaN impulse rows (as ``kkt.fields``, kkt.py
    ``Fields``), so every measure of them is NaN."""

    values: np.ndarray        # [4] outcome values from the VJP's forward pass
    valid: np.ndarray         # [4] bool (the base run's plans; steer: decision step taken)
    impulse: PT.Samples       # kappa, w_left, w_right: [4, N] d outcome / d sample
    seconds: float            # wall time of the VJP call (forward + 4 pullbacks), compiled
    near_tie: np.ndarray      # [4] bool

    def of(self, outcome: str) -> PT.Samples:
        i = OUTCOMES.index(outcome)
        return PT.Samples(*(a[i] for a in self.impulse))


def _check_run(net: NetPolicy, base: Run, what: str, allow_exhausted: bool) -> None:
    _refuse_trained(net, what)
    if base.exhausted and not allow_exhausted:
        raise LoopExhausted(f"{what}: the run from {base.start.source} is alive at the loop's "
                            f"end ({len(base.valid) - 1} steps) without reaching s_stop")


def fields(net: NetPolicy, rd: Road, base: Run, steps: int = STEPS, repeats: int = 1,
           allow_exhausted: bool = False) -> Fields:
    """One ``jax.vjp`` of the outcome vector with respect to the three sample arrays at the
    base run's picks. ``seconds`` is the best of ``repeats`` calls after the compile. Refuses
    a trained network before the tag, and an exhausted base run unless ``allow_exhausted``
    (only the decision step's steer is then read)."""
    require_x64()
    _check_run(net, base, "fields", allow_exhausted)
    fn = _fields_jit(net.fn, int(steps))
    args = (net.params, net.obs_rms, rd.base, rd.samples, loop_inputs(base.start, steps),
            picks_of(base), PT.zeros(rd.n))
    y, imp = jax.block_until_ready(fn(*args))
    best = np.inf
    for _ in range(max(int(repeats), 1)):
        t0 = time.perf_counter()
        y, imp = jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    valid = base.valid_outcomes()
    tie = np.array([base.plans[o].near_tie for o in O.OUTCOMES] + [False])
    return Fields(values=np.where(valid, np.asarray(y), np.nan), valid=valid,
                  impulse=PT.Samples(*(np.where(valid[:, None], np.asarray(a), np.nan)
                                       for a in imp)),
                  seconds=float(best), near_tie=tie)


@functools.lru_cache(maxsize=4)
def _explicit_jit():
    def explicit(x, base, samples, sigma, pick, delta0):
        def c(d):
            return O.c_min(x, PT.perturbed_arrays(base, samples, d), sigma, pick)

        return jax.vjp(c, delta0)[1](jnp.ones((), jnp.float64))[0]

    return jax.jit(explicit)


def explicit_c_min(rd: Road, base: Run) -> PT.Samples:
    """c_min's own read of the road at its pick with the states held (the edge's move, as
    ``kkt.Fields.explicit['c_min']`` on the optimum side): d c_min / d sample through the
    reader alone, no loop. NaN rows when c_min is invalid. The line's part is the impulse
    minus this."""
    require_x64()
    pl = base.plans["c_min"]
    if not pl.valid:
        return PT.Samples(*(np.full(rd.n, np.nan) for _ in PT.FEATURES))
    g = _explicit_jit()(jnp.asarray(base.x), rd.base, rd.samples, jnp.int32(base.start.sigma),
                        _pick(pl), PT.zeros(rd.n))
    return PT.Samples(*(np.asarray(a) for a in g))


# ------------------------------------------------------------------------- paths of the road

PATHS = {"obs_kappa": ("kappa_c",), "obs_pose": PT.POSE_FIELDS,
         "obs_widths": ("w_left_c", "w_right_c"), "dynamics": None}
PART_FIELDS = {**PATHS, "obs": PT.FLOAT_FIELDS}   # 'obs': the three observation paths at once


def _arrays64(base: TrackArrays) -> TrackArrays:
    """The base arrays as ``perturb.perturbed_arrays`` lays them out at delta = 0 (JAX)."""
    return TrackArrays(**{k: jnp.asarray(getattr(base, k), jnp.float64) for k in PT.FLOAT_FIELDS},
                       ds=jnp.asarray(base.ds, jnp.float64),
                       n_points=jnp.asarray(base.n_points, jnp.int32),
                       closed=jnp.asarray(base.closed))


@functools.lru_cache(maxsize=32)
def _path_jit(fn: Policy, steps: int, part: str):
    p = O.drive_params()
    keep = PART_FIELDS[part]

    def vec(params, obs_rms, base, samples, li, picks, delta):
        pert = PT.perturbed_arrays(base, samples, delta)
        b = _arrays64(base)
        if keep is None:                     # the dynamics (and the outcome's road reading)
            tr, tro = pert, b
        else:                                # one path of the observation
            tr, tro = b, b._replace(**{k: getattr(pert, k) for k in keep})
        trc = closed_loop(fn, params, obs_rms, tr, li, steps, tr_obs=tro)
        return outcome_vector(trc.x, trc.mean, tr, picks, p)

    def fields_(params, obs_rms, base, samples, li, picks, delta0):
        y, pull = jax.vjp(lambda d: vec(params, obs_rms, base, samples, li, picks, d), delta0)
        return jax.vmap(pull)(jnp.eye(len(OUTCOMES), dtype=y.dtype))[0]

    return jax.jit(fields_)


def path_fields(net: NetPolicy, rd: Road, base: Run, steps: int = STEPS,
                total: Fields | None = None, allow_exhausted: bool = False,
                parts: tuple[str, ...] = tuple(PATHS)) -> dict:
    """The plan-level impulse responses split by the road's path into the closed loop
    (``PATHS``; module docstring): {part: Samples of [4, N]}, plus 'sum_rel_err', the
    largest relative error of the parts' sum against the full VJP (``fields``) over the
    valid outcomes (PATH_TOL = 1e-9; NaN unless ``parts`` partitions the paths: all of
    ``PATHS``, or 'obs' with 'dynamics'). Invalid outcomes' rows are NaN. A scripted
    callable that reads ``tr`` itself (the edge keeper, Q) counts those reads in the
    observation's parts."""
    require_x64()
    _check_run(net, base, "path fields", allow_exhausted)
    unknown = set(parts) - set(PART_FIELDS)
    if unknown:
        raise ValueError(f"unknown parts {sorted(unknown)}; expected {sorted(PART_FIELDS)}")
    total = fields(net, rd, base, steps, allow_exhausted=allow_exhausted) if total is None \
        else total
    args = (net.params, net.obs_rms, rd.base, rd.samples, loop_inputs(base.start, steps),
            picks_of(base), PT.zeros(rd.n))
    valid = base.valid_outcomes()
    out = {}
    for part in parts:
        imp = jax.block_until_ready(_path_jit(net.fn, int(steps), part)(*args))
        out[part] = PT.Samples(*(np.where(valid[:, None], np.asarray(a), np.nan) for a in imp))
    partition = set(parts) in (set(PATHS), {"obs", "dynamics"})
    err = 0.0 if partition else float("nan")
    for f in PT.FEATURES if partition else ():
        tot = np.asarray(getattr(total.impulse, f))[valid]
        sm = sum(np.asarray(getattr(out[k], f))[valid] for k in parts)
        scale = max(float(np.abs(tot).max(initial=0.0)), 1e-300)
        err = max(err, float(np.abs(sm - tot).max(initial=0.0)) / scale)
    return {"parts": out, "sum_rel_err": err, "valid": valid}


# ------------------------------------------------------------------------- kappa clipping


def _obs_end(r: Run, outcome: str) -> int:
    """How many of the run's observations (steps 0..t-1) an outcome reads: the steps before
    its read end (``_read_end``), the decision step included for steer; every step taken for
    an invalid outcome."""
    n = r.n_steps
    if outcome == "steer":
        return min(r.start.decision_step + 1, n)
    pl = r.plans.get(outcome)
    if pl is None or not pl.valid:
        return n
    return min(_read_end(r, outcome), n)


def kappa_clip_steps(r: Run, t_end: int | None = None) -> list[tuple[int, int]]:
    """(step, preview point k) where a kappa channel is at the obs clip, |z| >= 10, over
    steps < t_end (all taken by default). NaN z (a raw-observation policy) never clips."""
    n = r.n_steps if t_end is None else min(int(t_end), r.n_steps)
    z = np.asarray(r.z[:n][:, KAPPA_CHANNELS], float)
    with np.errstate(invalid="ignore"):
        hit = np.abs(z) >= OBS_CLIP
    return [(int(t), int(k)) for t, k in zip(*np.nonzero(hit), strict=True)]


def kappa_clipped(r: Run, outcome: str) -> bool:
    """Step 2's kappa-clipping flag of one outcome (module docstring): a kappa channel at the
    clip over the observations the outcome reads."""
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}; expected one of {OUTCOMES}")
    return bool(kappa_clip_steps(r, _obs_end(r, outcome)))


def kappa_clip_record(r: Run) -> dict:
    """Per outcome the flag, and the clipped (step, k) set both over the outcome's reads and
    before s_c1 (step 1's window, recorded beside it)."""
    s = r.x[:, 0]
    k_c1 = _first(s[: r.n_steps] >= r.start.landmarks.s_c1)
    before = kappa_clip_steps(r, r.n_steps if k_c1 < 0 else k_c1)
    return {"flag": {o: kappa_clipped(r, o) for o in OUTCOMES},
            "steps": {o: kappa_clip_steps(r, _obs_end(r, o)) for o in OUTCOMES},
            "before_s_c1": bool(before), "steps_before_s_c1": before}


def directional(fl: Fields, direction: PT.Samples) -> np.ndarray:
    """[4] derivatives of the outcomes along a sample-space direction (``perturb.combine``)."""
    return sum(PT.combine(getattr(fl.impulse, f), np.asarray(getattr(direction, f))[None])[:, 0]
               for f in PT.FEATURES)


# ------------------------------------------------------------------------- discrete structure


@dataclass(frozen=True)
class Structure:
    """The discrete structure of a run that finite differences must not cross (Gate 2; branch
    consistency compares its end, saturation set and events with the float32 file's)."""

    end: tuple                # (term_step, trunc_step, stop_step)
    saturated: frozenset      # (step, channel): |a| > 1 on a policy step (env clip active)
    clipped: frozenset        # (step, channel): |z| >= 10 (obs clip active)
    obs_kinks: frozenset      # (step, kink): the observation's own kinks (``_obs_kinks``)
    events: tuple             # per outcome: (valid, pick) of the plans; steer: taken


OBS_KINKS = ("preview_lo", "preview_hi", "util_f", "util_r", "vx_lt_1")


def _obs_kinks(x: np.ndarray, p) -> list[np.ndarray]:
    """Per state, on which side of each observation kink it lies (racing.py:86-117): the
    preview horizon clip(6 max(vx, 0), 150, 500), min(util, 2) per axle, max(vx, 1) in the
    sideslip channel."""
    vx = x[:, 3]
    fo = tyre_forces(x, p, M.NP)
    d = 6.0 * np.maximum(vx, 0.0)
    return [d <= 150.0, d >= 500.0, np.asarray(fo.util_f) >= 2.0, np.asarray(fo.util_r) >= 2.0,
            vx < 1.0]


def _events(r: Run) -> tuple:
    out = []
    for o in O.OUTCOMES:
        pl = r.plans[o]
        pk = () if not pl.valid else tuple(np.asarray(v).item() for v in pl.pick)
        out.append((bool(pl.valid), pk))
    return (*out, bool(np.isfinite(r.steer)))


def structure(r: Run) -> Structure:
    n = r.n_steps
    nf = len(r.start.a_forced)
    sat = {(int(t), int(c)) for t, c in zip(*np.nonzero(np.abs(r.a[:n]) > 1.0), strict=True)
           if t >= nf}
    clip = {(int(t), int(i)) for t, i in zip(*np.nonzero(np.abs(r.z[:n]) >= OBS_CLIP),
                                             strict=True)}
    flags = _obs_kinks(r.x[:n], O.drive_params())
    kinks = {(int(t), OBS_KINKS[j]) for j, f in enumerate(flags) for t in np.flatnonzero(f)}
    return Structure(end=(r.term_step, r.trunc_step, r.stop_step), saturated=frozenset(sat),
                     clipped=frozenset(clip), obs_kinks=frozenset(kinks), events=_events(r))


# ------------------------------------------------------------------------- kink radius


class SegPicks(NamedTuple):
    """Every element pick of the c_min and v_apex windows (``outcomes.segment``), padded."""

    c_min: O.Pick
    v_apex: O.Pick
    sigma: Any


def _seg_picks(base: Run, steps: int) -> tuple[SegPicks, dict[str, int]]:
    n_pad = steps + 3
    out, counts = {}, {}
    v = O.view(base.traj)
    a, b = O.window(base.start.landmarks)
    ys = {"c_min": O.clearance_np(base.x, base.arrays, base.start.sigma), "v_apex": base.x[:, 3]}
    for name, y in ys.items():
        sg = O.segment(v, a, b, y) if base.plans[name].valid else None
        picks = [] if sg is None else sg[2]
        counts[name] = len(picks)
        picks = picks + [O.Pick(0, 0, 0.0, False)] * (n_pad - len(picks))
        out[name] = O.Pick(np.array([p.i0 for p in picks], np.int32),
                           np.array([p.i1 for p in picks], np.int32),
                           np.array([p.s_star for p in picks], np.float64),
                           np.array([p.interp for p in picks], bool))
    return SegPicks(out["c_min"], out["v_apex"], np.int32(base.start.sigma)), counts


@functools.lru_cache(maxsize=16)
def _kink_jit(fn: Policy, steps: int):
    p = O.drive_params()

    def quantities(params, obs_rms, base, samples, li, seg, delta):
        tr, trc = _loop_on(fn, steps, params, obs_rms, base, samples, li, delta)
        x = trc.x
        fo = tyre_forces(x, p, JAX)
        c = O.clearance(x, tr, seg.sigma)
        return dict(mean=trc.mean, z=trc.z, s=x[:, 0], vx=x[:, 3], util_f=fo.util_f,
                    util_r=fo.util_r, beta=O.brake_demand(x, p),
                    c_seg=jax.vmap(lambda pk: O.read(c, x[:, 0], pk))(seg.c_min),
                    v_seg=jax.vmap(lambda pk: O.read(x[:, 3], x[:, 0], pk))(seg.v_apex))

    def jvp(params, obs_rms, base, samples, li, seg, direction):
        zero = jax.tree.map(jnp.zeros_like, direction)
        return jax.jvp(lambda d: quantities(params, obs_rms, base, samples, li, seg, d),
                       (zero,), (direction,))

    return jax.jit(jvp)


def _radius(margin: np.ndarray, tangent: np.ndarray) -> float:
    """min margin / |tangent| (inf where the tangent vanishes)."""
    m = np.abs(np.asarray(margin, float)).ravel()
    t = np.abs(np.asarray(tangent, float)).ravel()
    ok = np.isfinite(m) & np.isfinite(t) & (t > 0)
    return float(np.min(m[ok] / t[ok])) if ok.any() else np.inf


def _read_end(base: Run, outcome: str) -> int:
    """The last state an outcome depends on (its upstream ends there)."""
    pl = base.plans.get(outcome)
    if outcome == "steer":
        return base.start.decision_step
    if outcome == "s_b":
        s = base.x[:, 0]
        k_apex = _first(s >= base.start.landmarks.s_apex)
        return max(int(pl.pick.k) + 3, k_apex if k_apex >= 0 else base.n_steps)
    k_end = _first(base.x[:, 0] >= O.window(base.start.landmarks)[1])
    return k_end if k_end >= 0 else base.n_steps


def kink_radius(net: NetPolicy, rd: Road, base: Run, direction: PT.Samples,
                steps: int = STEPS) -> dict[str, float]:
    """Per outcome: the smallest amplitude along ``direction`` at which a kink upstream of it
    is crossed, to first order (margin / |AD tangent| of each kink quantity: the env clip,
    the obs clip, the observation's kinks, the argmin and window-end brackets, the onset
    threshold)."""
    _refuse_trained(net, "kink radii")
    seg, counts = _seg_picks(base, steps)
    vals, tans = _kink_jit(net.fn, int(steps))(net.params, net.obs_rms, rd.base, rd.samples,
                                               loop_inputs(base.start, steps), seg, direction)
    v = {k: np.asarray(a) for k, a in vals.items()}
    t = {k: np.asarray(a) for k, a in tans.items()}
    nf, n = len(base.start.a_forced), base.n_steps
    a_win, b_win = O.window(base.start.landmarks)
    out = {}
    for o in OUTCOMES:
        if not base.valid_outcomes()[OUTCOMES.index(o)]:
            out[o] = np.nan
            continue
        r_end = min(_read_end(base, o), n)
        # state r_end depends on the commands of steps < r_end and on the observations of
        # those steps; the steer mean at step d also reads the observation of step d
        t_obs = r_end + 1 if o == "steer" else r_end
        t_act = slice(nf, max(r_end, nf))
        r = [_radius(1.0 - np.abs(v["mean"][t_act]), t["mean"][t_act]),
             _radius(OBS_CLIP - np.abs(v["z"][:t_obs]), t["z"][:t_obs]),
             _radius(np.minimum(np.abs(6 * v["vx"][:t_obs] - 150), np.abs(6 * v["vx"][:t_obs] - 500)),
                     6 * t["vx"][:t_obs]),
             _radius(2.0 - v["util_f"][:t_obs], t["util_f"][:t_obs]),
             _radius(2.0 - v["util_r"][:t_obs], t["util_r"][:t_obs])]
        if o in ("c_min", "v_apex"):
            key = "c_seg" if o == "c_min" else "v_seg"
            y, dy = v[key][: counts[o]], t[key][: counts[o]]
            j = int(np.argmin(y))
            others = np.arange(len(y)) != j
            r.append(_radius((y - y[j])[others], (dy - dy[j])[others]))
            k = np.arange(min(r_end + 1, len(v["s"])))
            r.append(_radius(np.minimum(np.abs(v["s"][k] - a_win), np.abs(v["s"][k] - b_win)),
                             t["s"][k]))
        if o == "s_b":
            k = slice(0, r_end + 1)
            r.append(_radius(v["beta"][k] - BRAKE_ON, t["beta"][k]))
        out[o] = float(min(r))
    return out


# ------------------------------------------------------------------------- Gate 2


@dataclass
class Gate2:
    """AD against finite differences along directions, per outcome (design Section 8)."""

    names: list[str]
    ad: np.ndarray            # [M, 4] directional derivatives from the impulse responses
    fd: np.ndarray            # [M, 4] Richardson value (4 FD(h/2) - FD(h)) / 3
    fd_h: np.ndarray          # [M, 4] central FD(h)
    fd_h2: np.ndarray         # [M, 4] central FD(h/2)
    h: np.ndarray             # [M, 4]
    radius: np.ndarray        # [M, 4] kink radius (amplitude units)
    counted: np.ndarray       # [M, 4] bool
    reason: list[list[str]]   # [M][4] why not counted ('' when counted)
    rel_err: np.ndarray       # [4] ||AD - FD|| / ||FD|| over counted directions (NaN: none)
    seconds: float

    @property
    def share_counted(self) -> np.ndarray:
        """[4] fraction of the valid (direction, outcome) comparisons that count."""
        tried = np.array([[r != "outcome not valid" for r in row] for row in self.reason])
        n = tried.sum(axis=0)
        return np.where(n > 0, self.counted.sum(axis=0) / np.maximum(n, 1), np.nan)

    def valid_outcomes(self) -> list[bool]:
        """[4] whether each outcome was valid on the base run (the reason share_counted
        uses)."""
        return [r != "outcome not valid" for r in self.reason[0]] if self.reason else []

    def zero_rule(self, zero_atol: float = ZERO_ATOL) -> np.ndarray:
        """``zero_rule_error`` of this result."""
        return zero_rule_error(self, zero_atol)

    def outcome_ok(self, tol: float = GATE2_TOL, zero_atol: float = ZERO_ATOL) -> list[bool]:
        """Per outcome: an invalid one is skipped (True); a valid one passes only with a
        finite zero-rule error below ``tol`` (inf: FD exactly 0 where AD is not, the
        signature of a wiring bug; NaN: nothing counted)."""
        err = self.zero_rule(zero_atol)
        return [bool(not v or (np.isfinite(e) and e < tol))
                for v, e in zip(self.valid_outcomes(), err, strict=True)]

    def ok(self, tol: float = GATE2_TOL, zero_atol: float = ZERO_ATOL) -> bool:
        """Gate 2's pass (one rule for the gates and the dry runs): every valid outcome
        passes ``outcome_ok`` and at least one is valid."""
        return bool(any(self.valid_outcomes()) and all(self.outcome_ok(tol, zero_atol)))


def zero_rule_error(g: Gate2, zero_atol: float = ZERO_ATOL) -> np.ndarray:
    """[4] field-vector relative error over the counted directions, 0 for a zero field
    (both AD and FD below ``zero_atol`` on every counted direction), inf when FD is exactly
    zero on every counted direction but AD is not, NaN when none counts."""
    out = np.full(g.ad.shape[1], np.nan)
    for k in range(g.ad.shape[1]):
        c = g.counted[:, k]
        if not c.any():
            continue
        ad, fd = g.ad[c, k], g.fd[c, k]
        if max(np.abs(ad).max(), np.abs(fd).max()) <= zero_atol:
            out[k] = 0.0
            continue
        den = float(np.linalg.norm(fd))
        out[k] = float(np.linalg.norm(ad - fd) / den) if den > 0 else np.inf
    return out


def _h_max(direction: PT.Samples) -> float:
    big = {f: float(np.max(np.abs(np.asarray(getattr(direction, f))))) for f in PT.FEATURES}
    caps = [H_MAX["kappa"] / big["kappa"]] if big["kappa"] > 0 else []
    caps += [H_MAX["width"] / big[f] for f in ("w_left", "w_right") if big[f] > 0]
    return min(caps) if caps else 1.0


def _scaled(direction: PT.Samples, h: float) -> PT.Samples:
    return PT.Samples(*(float(h) * np.asarray(a, float) for a in direction))


def gate2(net: NetPolicy, rd: Road, start: Start, directions: dict[str, PT.Samples],
          steps: int = STEPS, base: Run | None = None, fl: Fields | None = None,
          rich_rtol: float = RICH_RTOL, allow_exhausted: bool = False) -> Gate2:
    """Gate 2 at one start (module docstring). ``base`` and ``fl`` are recomputed if not
    given; ``allow_exhausted`` as in ``fields`` (a 1-step loop's steer)."""
    require_x64()
    _refuse_trained(net, "Gate 2")
    t0 = time.perf_counter()
    base = run(net, rd, start, steps=steps) if base is None else base
    fl = fields(net, rd, base, steps, allow_exhausted=allow_exhausted) if fl is None else fl
    s0 = structure(base)
    y0 = base.values()
    names = list(directions)
    M_, K_ = len(names), len(OUTCOMES)
    ad, fd, fd_h, fd_h2, hh, rad = (np.full((M_, K_), np.nan) for _ in range(6))
    counted = np.zeros((M_, K_), bool)
    reason = [["" for _ in OUTCOMES] for _ in names]
    for m, name in enumerate(names):
        d = directions[name]
        ad[m] = directional(fl, d)
        radius = kink_radius(net, rd, base, d, steps)
        h_cap = _h_max(d)
        for k, o in enumerate(OUTCOMES):
            if not fl.valid[k]:
                reason[m][k] = "outcome not valid"
                continue
            rad[m, k] = radius[o]
            h = min(KINK_FRACTION * radius[o], h_cap)
            hh[m, k] = h
            if not h >= H_MIN:
                reason[m][k] = "kink within round-off of the base point"
                continue
            runs = {s: run(net, rd, start, _scaled(d, s * h), steps)
                    for s in (-1.0, -0.5, 0.5, 1.0)}
            if any(structure(r) != s0 for r in runs.values()):
                reason[m][k] = "discrete structure changes within h"
                continue
            yv = {s: r.values()[k] for s, r in runs.items()}
            f1 = (yv[1.0] - yv[-1.0]) / (2 * h)
            f2 = (yv[0.5] - yv[-0.5]) / h
            fd_h[m, k], fd_h2[m, k] = f1, f2
            fr = (4 * f2 - f1) / 3
            fd[m, k] = fr
            floor = 10 * np.finfo(float).eps * max(abs(y0[k]), 1.0) / h
            if abs(f1 - f2) > max(rich_rtol * abs(fr), floor):
                reason[m][k] = "FD(h) and FD(h/2) disagree beyond O(h^2)"
                continue
            counted[m, k] = True
    rel = np.full(K_, np.nan)
    for k in range(K_):
        c = counted[:, k]
        if c.any():
            num = np.linalg.norm(ad[c, k] - fd[c, k])
            den = np.linalg.norm(fd[c, k])
            rel[k] = num / den if den > 0 else (0.0 if num == 0 else np.inf)
    return Gate2(names, ad, fd, fd_h, fd_h2, hh, rad, counted, reason, rel,
                 time.perf_counter() - t0)


def gate2_directions(rd: Road, start: Start, near=PT.GATE_NEAR_D, corner=PT.GATE_CORNER,
                     sigma_b: float = PT.GATE_SIGMA_B) -> dict[str, PT.Samples]:
    """Gate 1's directions on the network side (``perturb.gate_directions``, the one
    definition): each feature's bumps near the car at s_1 + d and at the corner's landmarks,
    and the inside and outside trapezoid shifts (design Section 8)."""
    return PT.gate_directions(rd.n, start.s1, start.landmarks, start.sigma, near, corner,
                              sigma_b)


# ------------------------------------------------------------------------- tangent consistency


def tangent_consistency(net: NetPolicy, rd: Road, start: Start, direction: PT.Samples,
                        eps: float, steps: int = STEPS) -> dict[str, float]:
    """The closed loop on the strip_arrays tangent against the secant along ``eps``
    ``direction`` (pose, kappa_c and widths moved consistently by both): max |dx| between
    them (second order in eps) and from the base run (first order), over the states both
    runs reached."""
    _refuse_trained(net, "tangent consistency")
    b = run(net, rd, start, steps=steps)
    tan = run(net, rd, start, _scaled(direction, eps), steps, mode="tangent")
    sec = run(net, rd, start, _scaled(direction, eps), steps, mode="secant")
    n = min(b.n_steps, tan.n_steps, sec.n_steps) + 1
    return {"tangent_vs_secant": float(np.max(np.abs(tan.x[:n] - sec.x[:n]))),
            "tangent_vs_base": float(np.max(np.abs(tan.x[:n] - b.x[:n]))), "states": int(n)}


# ------------------------------------------------------------------------- Gate 3


def _f64(a) -> np.ndarray:
    """A stored float32 array widened exactly."""
    return np.asarray(a, np.float32).astype(np.float64)


def _ulp32(s: np.ndarray) -> np.ndarray:
    return np.spacing(np.abs(np.asarray(s, np.float32))).astype(np.float64)


@functools.lru_cache(maxsize=16)
def _means_jit(fn: Policy):
    def means(params, obs_rms, tr, x, pa, ac, ctx, ctx_obs):
        def one(xk, pk, ak):
            st = env_state(xk, pk, ak, ctx)
            o, _ = policy_input(observe(st, tr, ctx_obs), obs_rms)
            return fn(params, o, st, tr, jnp.zeros((), jnp.int32))[0]

        return jax.vmap(one)(x, pa, ac)

    return jax.jit(means)


@jax.jit
def _steps_jit(tr, x, pa, ac, a, ctx):
    def one(xk, pk, ak, a_k):
        _, st, *_ = racing.step_no_reset(env_state(xk, pk, ak, ctx), a_k, tr, ENV)
        return st.x, st.prev_action, st.accel

    return jax.vmap(one)(x, pa, ac, a)


def policy_means(net: NetPolicy, tr: TrackArrays, x: np.ndarray, prev_action: np.ndarray,
                 accel: np.ndarray) -> np.ndarray:
    """[K, 2] float64 policy means on states (teacher-forced), as the loop computes them."""
    require_x64()
    return np.asarray(_means_jit(net.fn)(net.params, net.obs_rms, tr, x, prev_action, accel,
                                         ctx64(), context_channels()))


@dataclass
class Wiring:
    """Gate 3's teacher-forced check on one stored run."""

    steps: tuple[int, int]    # [t0, t1): the steps checked
    mean_err: float           # max |float64 mean - det_a_mean|
    mean_err_at: int          # its step
    step_err: np.ndarray      # [8] max |float64 step - det_x[t+1]| per state
    step_excess: np.ndarray   # [8] max |dx| / tolerance per state (s: substeps ulp32(s) / 2)
    accel_err: float          # max |float64 accel - det_accel[t+1]|
    prev_action_ok: bool      # det_prev_action[t+1] == clip(det_a[t]) exactly

    @property
    def mean_ok(self) -> bool:
        return bool(self.mean_err <= MEAN_TOL)

    @property
    def step_ok(self) -> bool:
        return bool(np.all(self.step_excess <= 1.0) and self.accel_err <= ACCEL_TOL
                    and self.prev_action_ok)

    @property
    def ok(self) -> bool:
        return self.mean_ok and self.step_ok


def horizon(det: dict[str, np.ndarray], i: int, s_stop: float) -> int:
    """Steps of row i up to the outcome horizon: the step that first reaches s_stop, the
    terminating or truncating step, or the last step taken (exclusive end)."""
    x, valid = det["x"][i], det["valid"][i]
    taken = int(valid[1:].sum())
    ends = [taken]
    k = _first(valid[1:] & (x[1:, 0] >= s_stop))
    if k >= 0:
        ends.append(k + 1)
    for name in ("term_step", "trunc_step"):
        if int(det[name][i]) >= 0:
            ends.append(int(det[name][i]) + 1)
    return min(ends)


def wiring_check(net: NetPolicy, det: dict[str, np.ndarray], i: int, tr: TrackArrays,
                 t0: int = 0, t1: int | None = None, s_stop: float | None = None,
                 zero_prev_action: bool = False, zero_accel: bool = False) -> Wiring:
    """Gate 3 on row i of a det group, steps t0 .. t1 - 1 (default: up to ``horizon`` at
    s_stop, or every step taken). ``zero_prev_action`` and ``zero_accel`` plant wiring
    errors (so does a ``net`` with another snapshot's obs_rms); they must fail."""
    require_x64()
    if t1 is None:
        t1 = horizon(det, i, np.inf if s_stop is None else s_stop)
    ts = np.arange(int(t0), int(t1))
    if not len(ts):
        raise ValueError(f"row {i}: no steps in [{t0}, {t1})")
    x, pa, ac = _f64(det["x"][i]), _f64(det["prev_action"][i]), _f64(det["accel"][i])
    a, a_mean = _f64(det["a"][i]), _f64(det["a_mean"][i])
    pa_in = np.zeros_like(pa[ts]) if zero_prev_action else pa[ts]
    ac_in = np.zeros_like(ac[ts]) if zero_accel else ac[ts]
    means = policy_means(net, tr, x[ts], pa_in, ac_in)
    err = np.max(np.abs(means - a_mean[ts]), axis=1)
    j = int(np.argmax(err))
    x1, _, ac1 = (np.asarray(v) for v in _steps_jit(tr, x[ts], pa[ts], ac[ts], a[ts], ctx64()))
    dx = np.abs(x1 - x[ts + 1])
    tol = np.tile(STEP_TOL, (len(ts), 1))
    tol[:, 0] = ENV.substeps * _ulp32(x[ts + 1, 0]) / 2      # each substep rounds s once
    clipped = np.clip(np.asarray(det["a"][i][ts], np.float32), -1, 1).astype(np.float64)
    return Wiring(steps=(int(t0), int(t1)), mean_err=float(err[j]), mean_err_at=int(ts[j]),
                  step_err=dx.max(axis=0), step_excess=(dx / tol).max(axis=0),
                  accel_err=float(np.max(np.abs(ac1 - ac[ts + 1]))),
                  prev_action_ok=bool(np.array_equal(pa[ts + 1], clipped)))


# ------------------------------------------------------------------------- branch consistency


@dataclass
class Branch:
    """A free-running float64 run against the float32 file's run (design Section 8)."""

    end64: tuple              # (crashed before the window end, step)
    end32: tuple
    saturated64: frozenset
    saturated32: frozenset
    events64: tuple
    events32: tuple
    values: dict[str, tuple[float, float]]   # outcome -> (float64, float32)
    tol: dict[str, float]

    @property
    def end_ok(self) -> bool:
        return self.end64 == self.end32

    @property
    def saturation_ok(self) -> bool:
        return self.saturated64 == self.saturated32

    @property
    def events_ok(self) -> bool:
        return self.events64 == self.events32

    @property
    def values_ok(self) -> bool:
        return all(abs(a - b) <= self.tol[o] for o, (a, b) in self.values.items()
                   if np.isfinite(a) or np.isfinite(b))

    @property
    def ok(self) -> bool:
        return self.end_ok and self.saturation_ok and self.events_ok and self.values_ok


def branch_tolerance(control_rows: list[dict], trained_mean_err: float,
                     factor: float = BRANCH_FACTOR, floor: dict | None = None) -> dict:
    """Branch consistency's per-outcome value tolerance, derived before the tag from data the
    rules allow (design Section 8's '5 x the float32/float64 spread', which cannot be measured
    on a trained network's free-running loop before the tag): for each outcome, the controls'
    float32/float64 outcome spread per unit of their teacher-forced per-step action error
    (``control_rows``: dicts with 'values' {outcome: (float64, float32)} as
    ``Branch.values`` and 'mean_err' from ``wiring_check``), its maximum over the rows, scaled
    by the trained networks' largest teacher-forced mean error over every key
    (``trained_mean_err``, Gate 3 on every key), times ``factor``; never below ``floor`` (per
    outcome; default none). Returns {outcome: tolerance} and the per-row slopes."""
    slopes: dict[str, list[float]] = {o: [] for o in O.OUTCOMES}
    for r in control_rows:
        e = float(r["mean_err"])
        for o, (a, b) in r["values"].items():
            if np.isfinite(a) and np.isfinite(b) and e > 0:
                slopes[o].append(abs(float(a) - float(b)) / e)
    tol = {o: (factor * max(v) * float(trained_mean_err) if v else float("nan"))
           for o, v in slopes.items()}
    if floor:
        tol = {o: max(t, floor.get(o, 0.0)) if np.isfinite(t) else floor.get(o, t)
               for o, t in tol.items()}
    return {"tol": tol, "slopes": slopes, "trained_mean_err": float(trained_mean_err),
            "factor": float(factor)}


def file_run(det: dict[str, np.ndarray], i: int, start: Start, arrays: TrackArrays) -> Run:
    """Row i of a float32 det group as a ``Run`` from the start's offset, cut at the outcome
    horizon as the loop is (``horizon``), with plans on ``arrays`` (the agents' road,
    upcast). Its z is not recorded (NaN)."""
    k0 = start.offset
    x = np.asarray(det["x"][i], np.float32).astype(np.float64)[k0:]
    valid = np.asarray(det["valid"][i], bool)[k0:].copy()
    h = horizon(det, i, start.s_stop) - k0
    valid[h + 1:] = False

    def rel(name: str) -> int:
        k = int(det[name][i]) - k0
        return k if 0 <= k < h else -1

    a = np.asarray(det["a"][i], np.float32).astype(np.float64)[k0:]
    mean = np.asarray(det["a_mean"][i], np.float32).astype(np.float64)[k0:]
    term = rel("term_step")
    plans = O.plans(O.trajectory(x, valid, term, source="file"), start.landmarks, start.sigma,
                    arrays)
    return Run(start=start, x=x, valid=valid, mean=mean, a=a,
               z=np.full((len(a), racing.OBS_DIM), np.nan), term_step=term,
               trunc_step=rel("trunc_step"), stop_step=_first(valid[1:] & (x[1:, 0] >= start.s_stop)),
               plans=plans, arrays=arrays)


def branch_consistency(r64: Run, det: dict[str, np.ndarray], i: int, tol: dict[str, float]
                       ) -> Branch:
    """Compare a float64 run (on the unperturbed road) with row i of a float32 det group
    from the same start: termination before the window end, the saturation set over the
    steps both took, the outcome events and the outcome values within ``tol`` (per
    outcome). Pure NumPy. Refuses an exhausted float64 run (``LoopExhausted``): a loop too
    short for the run is a pipeline failure, not a branch difference."""
    if r64.exhausted:
        raise LoopExhausted(f"branch consistency: the float64 run from {r64.start.source} "
                            "exhausted the loop")
    r32 = file_run(det, i, r64.start, r64.arrays)
    n = min(r64.n_steps, r32.n_steps)
    nf = len(r64.start.a_forced)

    def sat(r: Run) -> frozenset:
        return frozenset((int(t), int(c)) for t, c in zip(*np.nonzero(np.abs(r.a[:n]) > 1.0),
                                                         strict=True) if t >= nf)

    def end(r: Run) -> tuple:
        return (r.term_step >= 0, r.term_step)

    v64, v32 = r64.values(), r32.values()
    vals = {o: (float(v64[k]), float(v32[k])) for k, o in enumerate(O.OUTCOMES)}
    return Branch(end64=end(r64), end32=end(r32), saturated64=sat(r64), saturated32=sat(r32),
                  events64=_events(r64)[:3], events32=_events(r32)[:3], values=vals,
                  tol=dict(tol))


# ------------------------------------------------------------------------- the loop's length


def det_horizon(det: dict[str, np.ndarray], i: int, s_stop: float, offset: int = 0
                ) -> tuple[int, bool]:
    """(steps from state ``offset`` that row i's stored run needs, ended): ended when it
    reached s_stop or terminated, else it ran out of its own stored steps alive (a loop of
    any length could then exhaust)."""
    x, valid = det["x"][i], det["valid"][i]
    reached = _first(valid[1:] & (x[1:, 0] >= s_stop)) >= 0
    ended = bool(reached or int(det["term_step"][i]) >= 0)
    return int(horizon(det, i, s_stop) - offset), ended


def loop_audit(agent_ids=None, steps: int = STEPS, margin: int = HORIZON_MARGIN,
               earlier: bool = True, keys_root=None) -> dict:
    """Stored horizons against the loop's length (file reads only, no network): for each
    agent's final det group, the steps from the network's restart to s_stop (or the run's
    end) on the A2 set (from x0) and the agent's anchored finals (from x_{L-1}), and on the
    earlier snapshots' anchored keys from their own det groups. ``ok`` when every ended
    horizon is at most ``steps - margin`` and no stored run ran out alive."""
    from apexzero.fields import keys as K

    bank, _ = K.design()
    a2 = K.a2_refs(keys_root)
    finals = K.anchored_finals(keys_root)
    early = K.anchored_earlier(keys_root) if earlier else {}
    ids = list(finals) if agent_ids is None else list(agent_ids)
    limit = int(steps) - int(margin)
    out: dict[str, Any] = {"steps": int(steps), "margin": int(margin), "limit": limit,
                           "agents": {}}

    def scan(det, refs, anchored: bool) -> dict:
        hs, open_ = [], []
        for r in refs:
            try:
                i = det_row(det, r.cell_id, r.phase)
            except ValueError:
                continue
            s_stop = float(O.window(bank.stimuli[int(det["stim"][i])].landmarks)[1])
            h, ended = det_horizon(det, i, s_stop, r.anchor_step - 1 if anchored else 0)
            (hs if ended else open_).append(h)
        return {"n": len(hs) + len(open_), "max": int(max(hs, default=0)),
                "n_over": int(sum(h > limit for h in hs)), "n_open": len(open_),
                "max_open": int(max(open_, default=0))}

    fields_ = ("cell", "stim", "phase", "sample", "x", "valid", "term_step", "trunc_step")
    for aid in ids:
        ts = K.final_timesteps(aid)
        det = load_det(io.rollout_path(aid, ts), fields_)
        rec = {"a2": scan(det, a2, False), "finals": scan(det, finals.get(aid, ()), True)}
        for (a, snap), refs in early.items():
            if a == aid:
                d2 = load_det(io.rollout_path(aid, int(snap)), fields_)
                rec[f"earlier@{snap}"] = scan(d2, refs, True)
        out["agents"][aid] = rec
    groups = [g for rec in out["agents"].values() for g in rec.values()]
    out["max"] = int(max((g["max"] for g in groups), default=0))
    out["n_over"] = int(sum(g["n_over"] for g in groups))
    out["n_open"] = int(sum(g["n_open"] for g in groups))
    out["ok"] = bool(out["n_over"] == 0 and out["max"] <= limit)
    return out


# ------------------------------------------------------------------------- float32 references


def float32_reference(fn: Policy, params, obs_rms: RunningMeanStd | None, bank: StimulusBank,
                      starts: list[tuple[Cell, int]], steps: int, aux: list[int] | None = None
                      ) -> dict[str, np.ndarray]:
    """Deterministic runs of the frozen float32 harness (``make_batch``, the compiled
    ``_episode`` and ``trace_arrays``, rollout.py:403-712) from (cell, phase) starts, as a det
    group (``DET_FIELDS``). An x64-off process only: the harness asserts it. ``aux`` gives
    each run's controller row (``linear_policy`` with row-stacked params)."""
    runs = [rollout.RunSpec(c.cell_id, c.stim_id, c.pair_id, int(k), -1, "det", 1, int(steps),
                            0 if aux is None else int(aux[j])) for j, (c, k) in enumerate(starts)]
    batch, table = rollout.make_batch(runs, {c.cell_id: c for c, _ in starts})
    out = rollout.make_stimulus_rollout(fn, int(steps), "full")(
        params, obs_rms, rollout.bank_arrays(bank), batch)
    tr = rollout.trace_arrays(out, len(runs), "full")
    det = {k: np.asarray(v) for k, v in table.items() if k in DET_FIELDS}
    det.update({k: v for k, v in tr.items() if k in DET_FIELDS})
    det["aux"] = np.asarray(batch.aux)
    return det


def save_det(path: Path, det: dict[str, np.ndarray], meta: dict) -> Path:
    """A det group as an npz in the rollout files' layout (``det_<field>``) plus meta_json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"det_{k}": np.asarray(v) for k, v in det.items()}
    np.savez(path, meta_json=np.asarray(json.dumps(meta)), **arrays)
    return path


def load_reference(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """(det group with every field, meta) of a ``save_det`` file."""
    with np.load(Path(path), allow_pickle=False) as d:
        det = {k[4:]: d[k] for k in d.files if k.startswith("det_")}
        meta = json.loads(str(d["meta_json"])) if "meta_json" in d.files else {}
    return det, meta


# ------------------------------------------------------------------------- dry runs

def _design():
    from apexzero.fields import keys as K

    return K.design()


def _final_agent(agent_id: str):
    from apexzero.interp import agents

    path = agents.final_snapshot(agents.AGENTS[agent_id])
    if path is None:
        raise FileNotFoundError(f"{agent_id} has no final snapshot")
    return agents.load_agent(path, agent_id)


def _row_starts(det: dict[str, np.ndarray], bank: StimulusBank, cells: dict[int, Cell]
                ) -> list[Start]:
    return [cell_start(cells[int(c)], int(k), bank) for c, k in zip(det["cell"], det["phase"],
                                                                    strict=True)]


def _diag(r64: Run, det: dict[str, np.ndarray], i: int) -> dict:
    """Float32/float64 spread over the steps both runs took (diagnostic only)."""
    n = min(r64.n_steps, int(det["valid"][i][1:].sum()))
    x32 = np.asarray(det["x"][i], np.float64)[: n + 1]
    m32 = np.asarray(det["a_mean"][i], np.float64)[:n]
    return {"steps": n, "max_dx": np.abs(r64.x[: n + 1] - x32).max(axis=0).tolist(),
            "max_dmean": float(np.abs(r64.mean[:n] - m32).max()) if n else 0.0,
            "first_dmean": np.abs(r64.mean[0] - m32[0]).tolist() if n else []}


def _branch_record(b: Branch) -> dict:
    return {"ok": b.ok, "end_ok": b.end_ok, "saturation_ok": b.saturation_ok,
            "events_ok": b.events_ok, "values_ok": b.values_ok, "end64": list(b.end64),
            "end32": list(b.end32), "n_saturated": [len(b.saturated64), len(b.saturated32)],
            "events64": repr(b.events64), "events32": repr(b.events32),
            "values": {o: list(v) for o, v in b.values.items()}}


def _wiring_record(w: Wiring) -> dict:
    return {"ok": w.ok, "mean_ok": w.mean_ok, "step_ok": w.step_ok, "steps": list(w.steps),
            "mean_err": w.mean_err, "mean_err_at": w.mean_err_at,
            "step_err": w.step_err.tolist(), "step_excess": w.step_excess.tolist(),
            "accel_err": w.accel_err, "prev_action_ok": w.prev_action_ok}


def _gate2_record(g: Gate2) -> dict:
    return {"ok": g.ok(), "rel_err": g.rel_err.tolist(), "zero_rule": g.zero_rule().tolist(),
            "outcome_ok": g.outcome_ok(), "share_counted": g.share_counted.tolist(),
            "names": g.names, "ad": g.ad.tolist(), "fd": g.fd.tolist(), "h": g.h.tolist(),
            "counted": g.counted.tolist(), "reason": g.reason, "seconds": g.seconds}


def _planted(net: NetPolicy, wrong: NetPolicy, det, i, tr, t0, s_stop) -> dict:
    """Gate 3 with each planted wiring error: another snapshot's obs_rms, prev_action zeroed,
    accel zeroed. Each must fail (mean error above MEAN_TOL)."""
    out = {}
    for name, kw, n in (("obs_rms", {}, wrong), ("prev_action", {"zero_prev_action": True}, net),
                        ("accel", {"zero_accel": True}, net)):
        w = wiring_check(n, det, i, tr, t0=t0, s_stop=s_stop, **kw)
        out[name] = {"mean_err": w.mean_err, "fails": not w.mean_ok}
    return out


KEEPER_PARAMS = (np.float32(60.0), np.full(2, -np.inf, np.float32))   # psychophysics.py:607


def keeper_net() -> NetPolicy:
    """Step 1's scripted lane keeper (``rollout._lane_keeper``, a harness-check controller that
    reads the state and the road, not the observation) as a generic callable."""
    return callable_net(rollout._lane_keeper, KEEPER_PARAMS, None, "lane_keeper")


def dry_run_keeper(f32_path: Path) -> list[dict]:
    """The generic-callable path: the lane keeper against its float32 reference (branch
    consistency, valid outcomes, Gate 3's step check)."""
    bank, cells = _design()
    det, meta = load_reference(f32_path)
    net = keeper_net()
    steps = int(meta.get("loop_steps", STEPS))
    rows = []
    for i, start in enumerate(_row_starts(det, bank, cells)):
        rd = road(bank, start.stim_id)
        r = run(net, rd, start, steps=steps)
        rows.append({"source": start.source, "n_steps": r.n_steps, "term_step": r.term_step,
                     "valid": r.valid_outcomes().tolist(), "values": r.values().tolist(),
                     "branch": _branch_record(branch_consistency(r, det, i, BRANCH_TOL)),
                     "diag": _diag(r, det, i),
                     "wiring": _wiring_record(wiring_check(net, det, i, rd.base,
                                                           s_stop=start.s_stop))})
    return rows


def dry_run_random(f32_path: Path, seed: int = 0, gate2_rows: int = 1,
                   keeper_path: Path | None = None) -> tuple[dict, dict]:
    """The random-init actor (G1's architecture and obs_rms, read only; RF-C3) against its
    float32 reference: branch consistency, Gate 3 with planted errors (G2's obs_rms as the
    wrong one), the steer field's VJP timing and Gate 2 on steer (the only outcome a random
    actor keeps: it does not survive to s_c1). With ``keeper_path``, also ``dry_run_keeper``."""
    require_x64()
    bank, cells = _design()
    det, meta = load_reference(f32_path)
    g1, g2 = _final_agent("G1"), _final_agent("G2")
    net = random_net(g1.cfg, g1.obs_rms, seed)
    wrong = random_net(g1.cfg, g2.obs_rms, seed)
    steps = int(meta.get("loop_steps", STEPS))
    rows, arrays = [], {}
    for i, start in enumerate(_row_starts(det, bank, cells)):
        rd = road(bank, start.stim_id)
        r = run(net, rd, start, steps=steps)
        b = branch_consistency(r, det, i, BRANCH_TOL)
        w = wiring_check(net, det, i, rd.base, s_stop=start.s_stop)
        fl = fields(net, rd, r, steps, repeats=3)
        nan_rows = [bool(np.isnan(np.asarray(fl.impulse.kappa[k])).all()) for k in range(4)]
        rec = {"source": start.source, "n_steps": r.n_steps, "term_step": r.term_step,
               "exhausted": r.exhausted, "valid": r.valid_outcomes().tolist(),
               "branch": _branch_record(b), "diag": _diag(r, det, i), "wiring": _wiring_record(w),
               "planted": _planted(net, wrong, det, i, rd.base, 0, start.s_stop),
               "vjp_seconds": fl.seconds, "steer": r.steer, "steer_vjp": float(fl.values[3]),
               "impulse_nan_rows": nan_rows,
               "edge_gain": MS.edge_gain(fl.of("c_min"), start.landmarks, start.sigma),
               "kappa_clip": {k: v for k, v in kappa_clip_record(r).items()
                              if k in ("flag", "before_s_c1")}}
        if i < gate2_rows:
            rec["gate2"] = _gate2_record(gate2(net, rd, start, gate2_directions(rd, start),
                                               steps, base=r, fl=fl))
        arrays[f"steer_{i}"] = np.stack(fl.of("steer"))
        rows.append(rec)
    rep = {"policy": net.name, "steps": steps, "rows": rows}
    if keeper_path is not None:
        rep["keeper"] = dry_run_keeper(keeper_path)
    return rep, arrays


def kappa_readout(k: int, c: float) -> tuple[np.ndarray, np.ndarray]:
    """steer = c kappa * 50 at preview point k (RF-C1's shape, on the raw observation)."""
    W = np.zeros((2, racing.OBS_DIM), np.float32)
    W[0, _preview(k, 4)] = c
    return W, np.zeros(2, np.float32)


def edge_readout(k1: int, c1: float, k2: int, c2: float) -> tuple[np.ndarray, np.ndarray]:
    """steer = c1 ly(k1) + c2 ry(k2): body-frame y of the left edge at preview point k1 and
    of the right edge at k2 (RF-C4's shape, on the raw observation)."""
    W = np.zeros((2, racing.OBS_DIM), np.float32)
    W[0, _preview(k1, 1)] = c1
    W[0, _preview(k2, 3)] = c2
    return W, np.zeros(2, np.float32)


READOUTS = {"kappa": (10, 0.3), "edges": (6, 0.8, 9, -0.5)}
TANGENT_DIRECTIONS = (("kappa@near+10", 1e-5), ("w_left@near+10", 1e-3),
                      ("w_right@corner:c0", 1e-3))


def _paths_record(net: NetPolicy, rd: Road, r: Run, steps: int, fl: Fields) -> dict:
    """The plan-level path split at a start: the parts' sum against the full VJP, each part's
    size per outcome, and the dynamics part of the decision steer (exactly 0 at an A2 start's
    decision step 0, where x_0 is fixed)."""
    pf = path_fields(net, rd, r, steps, total=fl)
    k = OUTCOMES.index("steer")
    size = {part: [float(np.nanmax(np.abs(np.stack([np.asarray(a)[j] for a in imp]))))
                   if np.isfinite(np.asarray(imp.kappa)[j]).all() else float("nan")
                   for j in range(len(OUTCOMES))] for part, imp in pf["parts"].items()}
    dyn = pf["parts"]["dynamics"]
    return {"sum_rel_err": pf["sum_rel_err"], "max_abs": size,
            "steer_dynamics_max": float(max(np.abs(np.asarray(a)[k]).max() for a in dyn))}


def _exhaustion_record(net: NetPolicy, rd: Road, start: Start, steps: int = 20) -> dict:
    """A loop too short for the run: the plans say 'loop length exhausted' (not 'window end
    not reached'), and ``fields`` and ``branch_consistency`` refuse it."""
    r = run(net, rd, start, steps=steps)
    out = {"exhausted": r.exhausted, "reasons": {k: v.reason for k, v in r.plans.items()}}
    try:
        fields(net, rd, r, steps)
        out["fields_refused"] = False
    except LoopExhausted:
        out["fields_refused"] = True
    try:
        branch_consistency(r, {}, 0, BRANCH_TOL)
        out["branch_refused"] = False
    except LoopExhausted:
        out["branch_refused"] = True
    return out


def _blind_record(net: NetPolicy, rd: Road, r: Run, steps: int) -> dict:
    """The blind-pedal control on the network's own base run (its run reproduces the base run
    bit for bit) and its known answer on a planted policy: the run's own steer as a tape
    with the network's pedal row (a linear readout; ``steer_tape_policy``), whose blind-pedal
    control reads no road at all. There the blind fields' observation paths are exactly 0,
    its kappa field is its dynamics path, v_apex's and s_b's width fields are exactly 0, and
    the unblinded planted policy's s_b reads the preview's kappa (its obs_kappa part)."""
    blind = blind_pedal_net(net, r.mean[:, 1])
    rb = run(blind, rd, r.start, steps=steps)
    n = r.n_steps + 1
    out = {"reproduces": bool(np.array_equal(rb.x[:n], r.x[:n]) and rb.n_steps == r.n_steps),
           "n_steps": rb.n_steps}
    W, b = net.params
    planted = callable_net(steer_tape_policy, (r.mean[:, 0], W[1], b[1]), None, "steer-tape")
    rp = run(planted, rd, r.start, steps=steps)
    out["planted_max_dx"] = float(np.abs(rp.x[:n] - r.x[:n]).max())
    bl = blind_pedal_net(planted, rp.mean[:, 1])
    rbl = run(bl, rd, r.start, steps=steps)
    fb = fields(bl, rd, rbl, steps)
    pb = path_fields(bl, rd, rbl, steps, total=fb, parts=("obs", "dynamics"))
    fp = fields(planted, rd, rp, steps)
    iv, ib = OUTCOMES.index("v_apex"), OUTCOMES.index("s_b")
    out["blind_obs_max"] = float(max(np.nanmax(np.abs(np.asarray(getattr(pb["parts"]["obs"], f))))
                                     for f in PT.FEATURES))
    out["blind_width_vs_max"] = float(max(np.abs(np.asarray(getattr(fb.impulse, f))[j]).max()
                                          for f in ("w_left", "w_right") for j in (iv, ib)))
    dyn = np.asarray(pb["parts"]["dynamics"].kappa)[:3]
    tot = np.asarray(fb.impulse.kappa)[:3]
    out["blind_kappa_is_dynamics"] = float(np.nanmax(np.abs(dyn - tot))
                                           / max(float(np.nanmax(np.abs(tot))), 1e-300))
    out["blind_kappa_max"] = float(np.nanmax(np.abs(tot)))
    # the unblinded planted policy's pedal reads the preview: its s_b kappa field differs
    d_sb = np.asarray(fp.impulse.kappa)[ib] - np.asarray(fb.impulse.kappa)[ib]
    out["pedal_reading_share"] = float(np.abs(d_sb).max()
                                       / max(float(np.abs(np.asarray(fp.impulse.kappa)[ib]).max()),
                                             1e-300))
    return out


def dry_run_linear(f32_path: Path, gate2_rows: int = 2) -> tuple[dict, dict]:
    """The planted linear driver (``linear_driver``) against its float32 reference: branch
    consistency, Gate 3 (teacher-forced), the VJP of all four outcomes, Gate 2 on Gate 1's
    directions, the tangent against the secant through the closed loop, and the instantaneous
    steer fields of the RF-C1 / RF-C4 shaped readouts (their closed forms are the tests')."""
    require_x64()
    bank, cells = _design()
    det, meta = load_reference(f32_path)
    steps = int(meta.get("loop_steps", STEPS))
    rows, arrays = [], {}
    starts = _row_starts(det, bank, cells)
    for i, start in enumerate(starts):
        W, b = linear_driver(start.sigma)
        if not (np.array_equal(W, det["lin_W"][i]) and np.array_equal(b, det["lin_b"][i])):
            raise ValueError(f"row {i}: the file's readout is not linear_driver({start.sigma})")
        net = linear_net(W, b, "linear_driver")
        rd = road(bank, start.stim_id)
        r = run(net, rd, start, steps=steps)
        b_ = branch_consistency(r, det, i, BRANCH_TOL)
        w = wiring_check(net, det, i, rd.base, s_stop=start.s_stop)
        fl = fields(net, rd, r, steps, repeats=3)
        rec = {"source": start.source, "sigma": start.sigma, "n_steps": r.n_steps,
               "term_step": r.term_step, "valid": r.valid_outcomes().tolist(),
               "values": r.values().tolist(), "near_tie": fl.near_tie.tolist(),
               "vjp_values": fl.values.tolist(), "branch": _branch_record(b_),
               "diag": _diag(r, det, i), "wiring": _wiring_record(w), "vjp_seconds": fl.seconds}
        if i < gate2_rows:
            dirs = gate2_directions(rd, start)
            rec["gate2"] = _gate2_record(gate2(net, rd, start, dirs, steps, base=r, fl=fl))
            tc = {}
            for name, eps in TANGENT_DIRECTIONS:
                tc[name] = [tangent_consistency(net, rd, start, dirs[name], e, steps)
                            for e in (eps, eps / 2)]
            rec["tangent"] = tc
        if i == 0:
            rec["paths"] = _paths_record(net, rd, r, steps, fl)
            rec["blind_pedal"] = _blind_record(net, rd, r, steps)
            rec["exhaustion"] = _exhaustion_record(net, rd, start)
        for o in OUTCOMES:
            arrays[f"{o}_{i}"] = np.stack(fl.of(o))
        rows.append(rec)
    # instantaneous steer fields of the planted readouts at the first start
    start = starts[0]
    rd = road(bank, start.stim_id)
    ro = {}
    for name, args in READOUTS.items():
        W, b = kappa_readout(*args) if name == "kappa" else edge_readout(*args)
        net = linear_net(W, b, name)
        r = run(net, rd, start, steps=steps)
        fl = fields(net, rd, r, steps, allow_exhausted=True)     # the decision step's steer only
        arrays[f"readout_{name}"] = np.stack(fl.of("steer"))
        ro[name] = {"args": list(args), "steer": r.steer, "seconds": fl.seconds}
    return {"policy": "linear_driver", "steps": steps, "rows": rows, "readouts": ro,
            "readout_start": {"cell": int(det["cell"][0]), "phase": int(det["phase"][0])}}, arrays


def _anchored_refs(agent_id: str, snapshot: int, n: int) -> list:
    """The first n verified anchored keys of an agent's snapshot (key id order; JSON only)."""
    from apexzero.fields import keys as K

    out = []
    for p in sorted((io.KEYS_DIR / "anchored").glob("*.json")):
        d = json.loads(p.read_text())
        an = d.get("anchor") or {}
        if (d.get("agent") == agent_id and str(an.get("snapshot")) == str(snapshot)
                and d.get("status") == "verified"):
            out.append(K.load_key("anchored", d["key_id"]))
            if len(out) >= n:
                break
    return out


def _restart_record(rnd: NetPolicy, rd: Road, start: Start, det: dict[str, np.ndarray], i: int,
                    rec) -> dict:
    """Option A's restart, driven by a random-init actor (no trained network): the forced
    float64 step from x_{L-1} lands on det_x[L] within Gate 3's step tolerance and on the
    key's x[1] in s within S1_TOL; the steer field at the decision step (x_1, whose road
    dependence enters through the forced step's dynamics) passes Gate 2."""
    r = run(rnd, rd, start)
    x_file = _f64(det["x"][i][start.offset + 1])
    fl = fields(rnd, rd, r)
    dirs = {k: v for k, v in gate2_directions(rd, start).items()
            if k.startswith(("kappa", "w_"))}
    g = gate2(rnd, rd, start, dirs, base=r, fl=fl)
    k = OUTCOMES.index("steer")
    dyn = path_fields(rnd, rd, r, total=fl, parts=("dynamics",))["parts"]["dynamics"]
    return {"steer_dynamics_max": float(max(np.abs(np.asarray(a)[k]).max() for a in dyn)),
            "forced_step_err": np.abs(r.x[1] - x_file).tolist(),
            "forced_step_excess": float(np.max((np.abs(r.x[1] - x_file) / STEP_TOL)[1:])),
            "s1_err": abs(float(r.x[1, 0]) - float(np.asarray(rec.x)[1, 0])),
            "applied": r.a[0].tolist(), "a_prev": list(rec.anchor["a_prev"]),
            "n_steps": r.n_steps, "steer": r.steer, "vjp_seconds": fl.seconds,
            "steer_field_norm": float(np.linalg.norm(np.stack(fl.of("steer")))),
            "gate2": _gate2_record(g)}


def dry_run_gate3(agent_id: str = "G2", wrong_rms_from: str = "G1", n_anchored: int = 3,
                  n_a2: int = 2, seed: int = 0) -> tuple[dict, dict]:
    """Gate 3 (teacher-forced, no derivatives, no perturbations) on a trained agent's stored
    rows: anchored keys from x_{L-1} (option A's forced step included) and A2 rows from x0,
    each to the outcome horizon; then each planted wiring error, which must fail. On the
    anchored keys, option A's restart is also driven by a random-init actor of the agent's
    architecture (``_restart_record``): the only derivatives taken here."""
    require_x64()
    from apexzero.fields import keys as K

    bank, cells = _design()
    ag, other = _final_agent(agent_id), _final_agent(wrong_rms_from)
    net, wrong = agent_net(ag), agent_net(ag, obs_rms=other.obs_rms)
    rnd = random_net(ag.cfg, ag.obs_rms, seed)
    det = load_det(io.rollout_path(agent_id, ag.timesteps))
    rows = []
    for rec in _anchored_refs(agent_id, ag.timesteps, n_anchored):
        start = anchored_start(rec, det, bank, "A")
        i = det_row(det, rec.cell_id, rec.phase)
        rd = road(bank, start.stim_id)
        w = wiring_check(net, det, i, rd.base, t0=start.offset, s_stop=start.s_stop)
        rows.append({"key": rec.key_id, "kind": "anchored", "source": start.source,
                     "wiring": _wiring_record(w),
                     "planted": _planted(net, wrong, det, i, rd.base, start.offset,
                                         start.s_stop),
                     "restart_random": _restart_record(rnd, rd, start, det, i, rec)})
    a2 = [r for r in K.a2_refs() if r.kind == "nominal"][:: 37][:n_a2]
    for ref in a2:
        rec = K.load_key(ref)
        start = a2_start(rec, bank, cells)
        i = det_row(det, rec.cell_id, rec.phase)
        rd = road(bank, start.stim_id)
        w = wiring_check(net, det, i, rd.base, t0=0, s_stop=start.s_stop)
        rows.append({"key": rec.key_id, "kind": rec.kind, "source": start.source,
                     "wiring": _wiring_record(w),
                     "planted": _planted(net, wrong, det, i, rd.base, 0, start.s_stop)})
    return {"agent": net.name, "wrong_obs_rms": wrong_rms_from, "rows": rows,
            "loop_audit": loop_audit((agent_id,), earlier=False)}, {}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return repr(o)
    return o


def main(argv: list[str] | None = None) -> int:
    """``python -m apexzero.fields.network dry-run {random,linear,gate3} --out PREFIX``: a dry
    run in a float64 process (x64 switched on here), writing PREFIX.json and PREFIX.npz."""
    ap = argparse.ArgumentParser(prog="python -m apexzero.fields.network")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("what", choices=("random", "linear", "gate3"))
    d.add_argument("--f32", type=Path, help="float32 reference (save_det) for random / linear")
    d.add_argument("--keeper", type=Path, help="the lane keeper's float32 reference (random)")
    d.add_argument("--agent", default="G2")
    d.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    t0 = time.perf_counter()
    if args.what == "random":
        rep, arrays = dry_run_random(args.f32, keeper_path=args.keeper)
    elif args.what == "linear":
        rep, arrays = dry_run_linear(args.f32)
    else:
        rep, arrays = dry_run_gate3(args.agent)
    rep["seconds"] = time.perf_counter() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(_jsonable(rep), indent=1))
    np.savez(args.out.with_suffix(".npz"), **arrays)
    return 0


if __name__ == "__main__":
    sys.exit(main())
