"""The critic's estimate of the costate: its gradient pulled back to the car's state (design
v2.1, Section 3.2 'The pullback' and 'The accel channels'; Section 4 'Features and
extraction'; Section 5 'Random networks'; Section 6 C1 and C1-heads; Section 7 K3).

**A float64 process.** As in step 2 (network.py:4-8), everything here that evaluates a
critic needs x64 on (``fields.network.require_x64``, network.py:210-214): ``python -m
apexzero.costate.critic dry-run ...`` switches it on at its start. Random-init parameters are
drawn with x64 off (``random_critic_params``, as ``network.random_actor_params``,
network.py:306-313), so a float32 and a float64 process draw the same weights, and every
floating leaf is upcast exactly (T64, ``network.upcast32``, network.py:217-226). Importing
this module is safe in either kind of process.

**The observation map** (Section 3.2). The critic reads o = clip((ob - mean) /
sqrt(var + 1e-8), -10, 10) (``RunningMeanStd.normalize``, ppo.py:282-283; the actor's input,
ppo.py:390-391) of ob = observe(x, prev_action, accel, track, ctx) (racing.py:91-120), always
through step 2's ``network.observe`` with ``ctx_obs = context_channels()`` (network.py:234-252,
616-619), at restart conditions (hidden_mu = 1, ``network.env_state``, network.py:607-613), on
the stimulus's T64 road (``network.road``, network.py:766-769). ``Inputs`` holds one row per
state: (x_t, x_{t-1}, a_{t-1}, reset).

- Rows with t >= 1 (T at t >= 5, N) see prev_action = clip(a_{t-1}) (racing.py:150, 181) and
  accel = A(x_t, x_{t-1}) = ((vx_t - vx_{t-1})/dt - vy_t r_t, (vy_t - vy_{t-1})/dt + vx_t r_t)
  (``accel_of``, racing.py:155-156), x_{t-1} and a_{t-1} from the key (``key_inputs``).
- Reset rows (an A2 key's t = 0: D and T's t = 0 rows) see accel = prev_action = 0, as
  training resets (racing.py:136-139) and ``Cell.initial_aux`` (network.py:52-55).

``obs_jacobians`` (independent of the network, so shared by every critic and obs_rms) gives
ob, J = dob/dx_t at fixed accel and prev_action, J_prev = dob/dprev_action and J_acc =
dob/daccel (``jax.jacfwd``), dA/dx_t, and the on-manifold maps

    M_x = dA/dx_t + dA/dx_{t-1} Phi_x(x_{t-1}, a_{t-1})^-1,   M_a = -dA/dx_{t-1} Phi_x^-1 Phi_a,

with Phi_x and Phi_a the ``jacfwd`` of ``sim.control_step`` (sim.py:122-132, the env's step,
racing.py:149-152) at the clipped command, no Newton step, and linear solves rather than an
explicit inverse. The body dynamics do not read s, n or xi (dynamics.py:116-153), so Phi_x is
block triangular and M_x's (s, n, xi) columns vanish (``ObsJac.frenet_leak`` reports their
size): only g_vx and the secondary components move (Section 3.2). ``ObsJac.defect`` =
Phi(x_{t-1}, a_{t-1}) - x_t says how far a row is from the data manifold.

**The pullback** (``pullback``). With dY/do the gradient of each critic output (V, then the
heads), w = dY/do / sqrt(var + 1e-8) with the clipped channels zeroed (|z| >= 10: the clip
zeroes those rows, Section 3.2) and sigma_R = sqrt(ret_rms.var + 1e-8) (ppo.py:425-429;
``Snapshot.sigma``, audit/rollouts.py:30-34):

    g_held  = sigma_R J' w         pi_held = sigma_R J_prev' w     (accel held: sensitivity)
    s_acc   = sigma_R J_acc' w                       (accel-channel sensitivity, per m/s^2)
    g       = g_held + M_x' s_acc  pi      = pi_held + M_a' s_acc  (t >= 1: the primary)
    g       = g_held               pi      = pi_held               (reset rows: accel held)
    g_xprev = g_held + (dA/dx_t)' s_acc                           (x_{t-1} held: descriptive)

If V_c o z equals V* on the manifold of consistent observations, the primary g equals lambda
(Section 3.2 'What is claimed'). ``Pullback.accel_term`` = g - g_held is the accel term, and
``Pullback.accel_share`` its share of g_vx per state, |term| / (|g_held| + |term|), bounded in
[0, 1] (Section 3.2 'Sensitivities'). Every output is pulled back the same way, so the heads
give per-head g and pi (Section 6 C1-heads); C1 uses only the main V (output 0).

**Critics** (``CriticNet``): a callable fn(params, o) -> [1 + E] (V first, in units of reward
/ sigma_R, then E heads), its params, the obs_rms it reads, sigma_R and a ``trained`` mark.
``trained_critic`` wraps a snapshot's critic (marked trained); ``random_critic`` a random-init
critic of the network's own PPOConfig (``Critic(cfg).init(PRNGKey(seed), zeros(222))``,
Section 5; seeds 0-19, ``RANDOM_SEEDS``; V only unless ``heads``), read through the trained
network's obs_rms; ``linear_critic``, ``state_critic`` and ``accel_normal_critic`` are K3's
planted critics with known pullbacks (Section 7 K3: a planted linear critic; the accel-normal
plant V(x, a) + c phi(z_acc - A(x, a)), whose primary pullback is grad V and the right pi for
every c and phi, while the accel-held pullback is not).

**The extractor** (``layers``, ``extract``; Section 4 'Features and extraction'). One float64
extractor over the parameter tree serves the actor, the critic trunk and the heads:
h_k = tanh(h_{k-1} W_k + b_k) from ``MLPEncoder_0/Dense_k`` (ppo.py:147-157), the output
``Dense_0`` (the critic's V, the actor's pre-clip mean, ppo.py:190-191, 208-209), and head i's
hidden units u_i = relu(h_3 Dense_{2i+2} + b) and output u_i Dense_{2i+1} + b. Flax names a
compact module's submodules in construction order, and ppo.py:212-215 builds each head's
output Dense before the hidden one inside the call ``_dense(1)(relu(_dense(64)(h)))``, so
the pairing (2i + 2 hidden, 2i + 1 output) is by construction order, checked on shapes, not
inferred from them. Head gradients are taken only through the extractor: through
``Critic.apply`` they are exactly 0 (stop_gradient, ppo.py:211). ``apply_gap`` is K3's
self-consistency check against ``Critic.apply`` / ``Actor.apply`` in float64.

**C1-heads** (``head_quantities``, ``head_disagreement``, ``head_reliability``). For q in {V,
g_s, ..., g_pedal, pi_steer, pi_pedal}, D_q = mean_t var_k(q^k_t) / var_t(mean_k q^k_t) (heads
k (variance with ddof 1), states t, weighted over t), and the inter-head reliability rho_q, the
mean over head pairs of their correlation over states (sqrt(rho_q) bounds a head's |corr| with
any target).

**K3's checks** (Section 7). ``fd_pullback``: Richardson central differences (h, h/2) of
sigma_R V_c along each state component (K1's steps, ``targets.FD_H``) and each prev-action
component (``PI_H``), for the primary (x_t moved, x_{t-1} recovered by Newton on the control
step at fixed a_{t-1} (``prev_state``), accel recomputed; a_{t-1} moved the same way), the
accel-held and the x_{t-1}-held variants. ``fd_jacobian``: J_z by the same differences. A
comparison counts only where the discrete structure is the same at -h, -h/2, 0, h/2 and h
(``_signature``: the obs clip mask, the tyre min(util, 2) (racing.py:116), the preview's
d_max clip (racing.py:86), vx at 1 m/s in the sideslip channel (racing.py:113), the pedal's
C1 relu knots at 0 and +-0.02 (ops.py:49-52, dynamics.py:75-78) at x_t and x_{t-1}, the
segment of every pose lookup (the car's and the 40 preview points'; ``sim.lookup`` is
piecewise linear on the 2 m knots, sim.py:64-73, and a knot near the centre passes the O(h^2)
check while the difference averages the two one-sided slopes), and each head's ReLU signs for
that head) and FD(h) and FD(h/2) agree to O(h^2) (step 2's Gate 2 rule, network.py:76-80).
The gate is step 2's field-vector error (network.py:81-82): ||AD - FD|| / ||FD|| over the
counted state directions of a row and output (and separately over the two pi directions), or
the relative Frobenius error of J_z over its counted columns; K3 passes below ``FD_TOL`` =
1e-6. Per-component errors, |AD - FD| / max(|FD|, ``ZERO_REL`` ||FD||), are descriptive: a
component far below the others (g_s on an exit straight, about 1e-5 of the field) sits at the
round-off of the large global pose coordinates.

**Dry runs** (``python -m apexzero.costate.critic dry-run {plants,random} --out PREFIX``,
writing PREFIX.json and PREFIX.npz). 'plants': a synthetic float32 strip (a smooth corner,
exactly constant widths) with rows built by the env's own step, a synthetic normaliser, and
the planted critics against their closed forms and finite differences. 'random': three A2
keys' T states (the example key, a W = 800 key, R = 6 m; stored x and a only) through G1's
obs_rms (read, as step 2 does; G1's own networks are never built or evaluated), the
extractor against Flax, J_z and a random-init critic's pullbacks (main V and heads) against
finite differences, and Section 5's 20 random-init critics.

**Pre-registration** (Section 1, Section 7): before the tag ``costate-prereg-v1`` no trained
critic's value, gradient or finite difference is computed. ``trained_critic`` marks its critic
``trained``; ``pullback``, ``values`` and ``fd_pullback`` refuse it (PermissionError) until
``unlock_trained``, which only scripts/costate.py calls, after its freeze check. Reading a
snapshot's parameter tree (``layers``) evaluates nothing.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import Actor, Critic, PPOConfig, RunningMeanStd
from apexzero.costate.targets import FD_H, STATE_NAMES
from apexzero.env import racing
from apexzero.fields import network as NW
from apexzero.interp.constants import OBS_CLIP, OBS_PREVIEW
from apexzero.interp.io import sha256_arrays
from apexzero.vehicle.dynamics import body_derivatives, tyre_forces
from apexzero.vehicle.ops import JAX
from apexzero.vehicle.sim import TrackArrays, control_step

OBS_DIM = racing.OBS_DIM                 # 222
EGO = {nm: OBS_PREVIEW + i for i, nm in enumerate(
    ("vx", "vy", "r", "sideslip", "lateral", "xi", "delta", "pedal", "ax", "ay", "prev_steer",
     "prev_pedal", "util_f", "util_r", "alpha_f", "alpha_r"))}       # racing.py:112-118
ACCEL = slice(EGO["ax"], EGO["ay"] + 1)            # 208-209: accel / 50
PREV = slice(EGO["prev_steer"], EGO["prev_pedal"] + 1)   # 210-211: prev_action
ACCEL_SCALE = 50.0                       # racing.py:114
RMS_EPS = 1e-8                           # ppo.py:282 (normalize), ppo.py:425 (sigma_R)
DT = float(NW.ENV.dt)                    # s, one control step (racing.py:50)
SUBSTEPS = int(NW.ENV.substeps)
HEAD_HIDDEN = 64                         # ppo.py:213
N_RANDOM = 20                            # random-init critics (Section 5)
RANDOM_SEEDS = tuple(range(N_RANDOM))
OUTPUT_NAMES = ("V",)                    # output 0; heads follow as 'head0'..'head4'
PI_NAMES = ("pi_steer", "pi_pedal")
VARIANTS = ("manifold", "held", "xprev")
PI_H = 1e-5                              # FD step on a_{t-1} (K1's pi step, Section 7)
FD_STEPS = (-1.0, -0.5, 0.0, 0.5, 1.0)   # multiples of h: Richardson (h, h/2) and the centre
FD_TOL = 1e-6                            # K3: AD against finite differences (field-vector)
ZERO_REL = 1e-4                          # per-component error floor, x ||FD|| of the row (tau_s)
RICH_RTOL = NW.RICH_RTOL                 # |FD(h) - FD(h/2)| within this of the Richardson value
ROUNDOFF = 64.0                          # x eps |Y| / h: the round-off floor of FD(h) - FD(h/2)
PLANT_TOL = 1e-10                        # K3: the accel-normal plant's pullback
LINEAR_TOL = 1e-12                       # K3: a planted linear critic's pullback
NEWTON_ITERS = 30                        # prev_state: Newton on the control step
PLANT_ITERS = 8                          # the accel-normal plant's own Newton (no gradient)
PEDAL_KNOTS = (-2e-2, 0.0, 2e-2)         # ops.relu's knots on +-pedal (ops.py:49-52)
BATCH = 128                              # rows per jit call (one compile per function)
K3_MIN_STATES = 200                      # Section 7 K3: J_z and the pullbacks on 200 T states
OBS_F32_RTOL = 1e-3                      # K3: s's ratio on float32-upcast key arrays (report)

# ------------------------------------------------------------------------- pre-registration

_TRAINED_OK = False


def unlock_trained() -> None:
    """Allow trained critics in this process. Only scripts/costate.py calls it, after its
    freeze check of the tag ``costate-prereg-v1`` (design Section 7 'Freeze')."""
    global _TRAINED_OK
    _TRAINED_OK = True


def _refuse_trained(net: CriticNet, what: str) -> None:
    if net.trained and not _TRAINED_OK:
        raise PermissionError(f"{what} of the trained critic {net.name} waits for the tag "
                              "costate-prereg-v1 (design Sections 1 and 7: no trained network's "
                              "value, gradient or finite difference before it)")


# ------------------------------------------------------------------------- the extractor


@dataclass(frozen=True)
class Layers:
    """A parameter tree's layers in evaluation order (see the module docstring)."""

    trunk: tuple              # ((W [in, width], b [width]), ...): MLPEncoder_0/Dense_k
    out: tuple                # (W [width, d], b [d]): Dense_0
    heads: tuple              # ((W_hidden, b_hidden, W_out, b_out), ...): Dense_{2i+2}, _{2i+1}

    @property
    def n_heads(self) -> int:
        return len(self.heads)


def _dense_index(name: str) -> int:
    if not name.startswith("Dense_") or not name[6:].isdigit():
        raise ValueError(f"unexpected module {name!r}: not an MLP actor or critic")
    return int(name[6:])


def _kb(d) -> tuple:
    return d["kernel"], d["bias"]


def layers(params) -> Layers:
    """Parse an MLP actor's or critic's parameter tree (ppo.py:147-157, 183-217). Heads are
    paired by construction order: head i's hidden layer is Dense_{2i+2} and its output
    Dense_{2i+1} (Flax counts submodules as they are constructed, and ppo.py:213 constructs
    the output Dense before the hidden one). Raises on SimBa trees or anything else."""
    p = params.get("params", params)
    if "MLPEncoder_0" not in p:
        raise ValueError(f"no MLPEncoder_0 in {sorted(p)}: not an MLP trunk (SimBa?)")
    enc = p["MLPEncoder_0"]
    ks = sorted(_dense_index(k) for k in enc)
    if ks != list(range(len(ks))) or not ks:
        raise ValueError(f"MLPEncoder_0 layers {sorted(enc)} are not Dense_0..Dense_{len(ks) - 1}")
    trunk = tuple(_kb(enc[f"Dense_{k}"]) for k in ks)
    top = sorted(_dense_index(k) for k in p if k not in ("MLPEncoder_0", "log_std"))
    if top != list(range(len(top))) or len(top) % 2 != 1:
        raise ValueError(f"top-level layers {sorted(p)}: expected Dense_0 and 2 per head")
    width = np.shape(trunk[-1][0])[1]
    out = _kb(p["Dense_0"])
    if np.shape(out[0])[0] != width:
        raise ValueError(f"Dense_0 kernel {np.shape(out[0])} does not read the trunk's {width}")
    heads = []
    for i in range((len(top) - 1) // 2):
        Wh, bh = _kb(p[f"Dense_{2 * i + 2}"])
        Wo, bo = _kb(p[f"Dense_{2 * i + 1}"])
        if np.shape(Wh) != (width, HEAD_HIDDEN) or np.shape(Wo) != (HEAD_HIDDEN, 1):
            raise ValueError(f"head {i}: Dense_{2 * i + 2} {np.shape(Wh)} and Dense_{2 * i + 1} "
                             f"{np.shape(Wo)} are not [{width}, {HEAD_HIDDEN}] and "
                             f"[{HEAD_HIDDEN}, 1]")
        heads.append((Wh, bh, Wo, bo))
    return Layers(trunk, out, tuple(heads))


class Activations(NamedTuple):
    """The extractor's output for o [..., 222]."""

    h: tuple                  # trunk activations h_1..h_L, each [..., width]
    out: Any                  # [..., d] Dense_0: the critic's V (d = 1), the actor's mean (2)
    pre: Any                  # [..., E, 64] heads' hidden pre-activations
    u: Any                    # [..., E, 64] heads' hidden units relu(pre)
    heads: Any                # [..., E] head outputs


def extract(params, o, activation: str = "tanh") -> Activations:
    """Every activation of an MLP actor or critic on o (float64 when x64 is on)."""
    L = layers(params)
    act = jnp.tanh if activation == "tanh" else jax.nn.relu
    h, hs = o, []
    for W, b in L.trunk:
        h = act(h @ W + b)
        hs.append(h)
    out = h @ L.out[0] + L.out[1]
    if L.heads:
        pre = jnp.stack([h @ Wh + bh for Wh, bh, _, _ in L.heads], axis=-2)
        u = jax.nn.relu(pre)
        heads = jnp.stack([(u[..., i, :] @ Wo + bo)[..., 0]
                           for i, (_, _, Wo, bo) in enumerate(L.heads)], axis=-1)
    else:
        pre = u = jnp.zeros((*o.shape[:-1], 0, HEAD_HIDDEN), o.dtype)
        heads = jnp.zeros((*o.shape[:-1], 0), o.dtype)
    return Activations(tuple(hs), out, pre, u, heads)


def mlp_critic(params, o):
    """[1 + E]: V and the heads of an MLP tanh critic (``CriticNet.fn``)."""
    a = extract(params, o, "tanh")
    return jnp.concatenate([a.out[..., :1], a.heads], axis=-1)


def mlp_value(params, o):
    """[1]: V only (the random-init controls, Section 5)."""
    return extract(params, o, "tanh").out[..., :1]


def mlp_head_signs(params, o):
    """[E, 64] bool: the heads' ReLU signs (the FD structure of each head)."""
    return extract(params, o, "tanh").pre > 0.0


def apply_gap(cfg: PPOConfig, params, o, kind: str = "critic") -> dict:
    """K3's extractor self-consistency in float64 (Section 7): max |extractor - Flax apply|
    of the critic's V and heads (``Critic.apply(..., with_heads=True)``) or the actor's mean,
    on observations o [n, 222]."""
    NW.require_x64()
    o = jnp.asarray(o, jnp.float64)
    act = extract(params, o, cfg.activation)
    if kind == "actor":
        mean, _ = Actor(cfg, racing.ACTION_DIM, False).apply(params, o)
        return {"mean": float(jnp.max(jnp.abs(act.out - mean))), "dtype": str(mean.dtype)}
    mod = Critic(cfg)
    if cfg.ensemble_heads:
        v, hs = mod.apply(params, o, with_heads=True)
    else:
        v, hs = mod.apply(params, o), jnp.zeros((o.shape[0], 0))
    return {"v": float(jnp.max(jnp.abs(act.out[:, 0] - v))),
            "heads": float(jnp.max(jnp.abs(act.heads - hs))) if hs.shape[-1] else 0.0,
            "n_heads": int(act.heads.shape[-1]), "flax_heads": int(hs.shape[-1]),
            "dtype": str(v.dtype)}


# ------------------------------------------------------------------------- critics


@dataclass(frozen=True, eq=False)
class CriticNet:
    """A critic for the pullback (see the module docstring). ``fn`` maps (params, o [222]) to
    [1 + n_heads], V first, in units of reward / sigma; ``kinks`` (optional) maps (params, o)
    to the heads' discrete structure [n_heads, m] (bool) for the FD rule."""

    fn: Callable
    params: Any
    obs_rms: RunningMeanStd
    sigma: float = 1.0
    n_heads: int = 0
    name: str = ""
    trained: bool = False
    checkpoint: str = ""      # sha256 of the snapshot's .pkl
    kinks: Callable | None = None

    @property
    def n_out(self) -> int:
        return 1 + self.n_heads

    @property
    def output_names(self) -> tuple[str, ...]:
        return OUTPUT_NAMES + tuple(f"head{i}" for i in range(self.n_heads))


def sigma_r(ret_rms: RunningMeanStd) -> float:
    """sigma_R = sqrt(ret_rms.var + 1e-8) (ppo.py:425-429) on the exact upcast of var."""
    return float(np.sqrt(float(np.float32(np.asarray(ret_rms.var))) + RMS_EPS))


def _require_mlp(cfg: PPOConfig) -> None:
    if cfg.network != "mlp" or cfg.activation != "tanh":
        raise ValueError(f"config network={cfg.network!r} activation={cfg.activation!r}: idea "
                         "2's networks are MLPs with tanh layers (design Section 2)")


def random_critic_params(cfg: PPOConfig, seed: int = 0):
    """Float32 params of a random-init critic with ``cfg``'s architecture (heads included
    when cfg has them): ``Critic(cfg).init(PRNGKey(seed), zeros(222))`` with x64 off, so a
    float32 and a float64 process draw the same weights (Section 5). NumPy leaves."""
    with jax.enable_x64(False):
        params = Critic(cfg).init(jax.random.PRNGKey(int(seed)), jnp.zeros(OBS_DIM, jnp.float32))
        return jax.tree.map(np.asarray, params)


def params_hash(params) -> str:
    """sha256 of a parameter tree's leaves in tree order."""
    return sha256_arrays(*jax.tree_util.tree_leaves(params))


def random_critic(cfg: PPOConfig, obs_rms: RunningMeanStd, seed: int = 0, sigma: float = 1.0,
                  heads: bool = False) -> CriticNet:
    """A random-init critic of the network's own config (never ``PPOConfig()``), read
    through the trained network's obs_rms (read only) and scaled by its sigma_R; V only
    unless ``heads`` (Section 5)."""
    _require_mlp(cfg)
    prm = NW.upcast32(random_critic_params(cfg, seed))
    use_heads = bool(heads and cfg.ensemble_heads)
    return CriticNet(mlp_critic if use_heads else mlp_value, prm, NW.upcast32(obs_rms),
                     float(sigma), cfg.ensemble_heads if use_heads else 0, f"random-critic-{seed}",
                     kinks=mlp_head_signs if use_heads else None)


def random_critics(cfg: PPOConfig, obs_rms: RunningMeanStd, sigma: float = 1.0,
                   seeds=RANDOM_SEEDS, heads: bool = False) -> list[CriticNet]:
    """Section 5's 20 random-init critics (seeds 0-19) of one network."""
    return [random_critic(cfg, obs_rms, s, sigma, heads) for s in seeds]


def trained_critic(agent, checkpoint: str = "") -> CriticNet:
    """A ``LoadedAgent``'s critic with its heads, obs_rms and sigma_R, marked ``trained``
    (nothing is evaluated here)."""
    _require_mlp(agent.cfg)
    E = int(agent.cfg.ensemble_heads)
    prm = NW.upcast32(agent.snapshot.critic)
    if layers(prm).n_heads != E:
        raise ValueError(f"{agent.agent_id}: {layers(prm).n_heads} heads in the tree, config {E}")
    return CriticNet(mlp_critic, prm, NW.upcast32(agent.obs_rms),
                     sigma_r(agent.snapshot.ret_rms), E, f"{agent.agent_id}@{agent.timesteps}",
                     trained=True, checkpoint=checkpoint, kinks=mlp_head_signs if E else None)


# ------------------------------------------------------------------------- planted critics


def _linear_fn(params, o):
    W, b = params[0], params[1]
    return W @ o + b


def linear_critic(w: np.ndarray, b=0.0, obs_rms: RunningMeanStd | None = None,
                  sigma: float = 1.0, name: str = "linear") -> CriticNet:
    """V = w'o + b (w [222], or [1 + E, 222] for V and E planted heads): K3's planted linear
    critic, whose pullback is the closed form of its channels."""
    W = np.atleast_2d(np.asarray(w, np.float64))
    bb = np.broadcast_to(np.asarray(b, np.float64), (W.shape[0],)).copy()
    return CriticNet(_linear_fn, (W, bb), _rms(obs_rms), float(sigma), W.shape[0] - 1, name)


def _rms(obs_rms: RunningMeanStd | None) -> RunningMeanStd:
    if obs_rms is None:
        return RunningMeanStd(np.zeros(OBS_DIM), np.ones(OBS_DIM), np.float64(1.0))
    return RunningMeanStd(*(np.asarray(a, np.float64) for a in obs_rms))


def decode(o, mean, var, widths: tuple[float, float], delta_max: float):
    """(x [8] with s = 0, prev_action [2], accel [2]) read back from an unclipped o: the ego
    channels (racing.py:112-115) with n from the lateral channel on a road of constant
    ``widths`` (w_left, w_right). For planted critics only."""
    ob = o * jnp.sqrt(var + RMS_EPS) + mean
    wl, wr = widths
    n = ob[EGO["lateral"]] * 0.5 * (wl + wr) + 0.5 * (wl - wr)
    x = jnp.stack([jnp.zeros((), o.dtype), n, ob[EGO["xi"]], ob[EGO["vx"]] * 50.0,
                   ob[EGO["vy"]] * 5.0, ob[EGO["r"]], ob[EGO["delta"]] * delta_max,
                   ob[EGO["pedal"]]])
    return x, ob[PREV], ob[ACCEL] * ACCEL_SCALE


def _vehicle(ctx=None):
    return racing.params_from_context(jnp.asarray(NW.ctx64(ctx)), hidden_mu=1.0)


def manifold_accel(x, a, p, iters: int = PLANT_ITERS):
    """A(x_t, x_{t-1}) with x_{t-1} = Phi^-1(x_t; a), from the body dynamics alone
    (dynamics.py:116-153, RK2 substeps as sim.rk2_step, sim.py:107-111): the accel the env
    reports for x_t reached under a, as a differentiable function of (x_t, a). Newton
    (``iters`` steps, no gradient) finds x_{t-1}; one more Newton step from it carries the
    derivative, which at the converged point is the implicit one, J^-1 (dx_t - Phi_a da)
    (the residual multiplies the rest). The accel-normal plant's own route, independent of
    ``obs_jacobians``."""
    h = DT / SUBSTEPS
    pad = jnp.zeros(3, x.dtype)

    def step(xb, u):
        def rate(xc):
            return jnp.stack(body_derivatives(jnp.concatenate([pad, xc]), u, p, JAX))

        def sub(xc, _):
            return xc + h * rate(xc + 0.5 * h * rate(xc)), None

        return jax.lax.scan(sub, xb, None, length=SUBSTEPS)[0]

    def newton(y, xb, u):
        return y - jnp.linalg.solve(jax.jacfwd(step)(y, u), step(y, u) - xb)

    xb = x[3:]
    xs, us = jax.lax.stop_gradient(xb), jax.lax.stop_gradient(a)
    y = jax.lax.stop_gradient(jax.lax.fori_loop(0, iters, lambda _, y_: newton(y_, xs, us), xs))
    y = newton(y, xb, a)
    return jnp.stack([(x[3] - y[0]) / DT - x[4] * x[5], (x[4] - y[1]) / DT + x[3] * x[5]])


@functools.lru_cache(maxsize=16)
def _state_fn(f: Callable, widths: tuple[float, float]):
    dmax = float(_vehicle().delta_max)

    def fn(params, o):
        mean, var, _, sigma = params
        x, pa, _ = decode(o, mean, var, widths, dmax)
        return jnp.reshape(f(x, pa) / sigma, (1,))

    return fn


@functools.lru_cache(maxsize=16)
def _accel_normal_fn(f: Callable, phi: Callable, widths: tuple[float, float]):
    p = _vehicle()
    dmax = float(p.delta_max)

    def fn(params, o):
        mean, var, c, sigma = params
        x, pa, acc = decode(o, mean, var, widths, dmax)
        return jnp.reshape((f(x, pa) + c * phi(acc - manifold_accel(x, pa, p))) / sigma, (1,))

    return fn


def _plant_params(obs_rms, c: float, sigma: float) -> tuple:
    rms = _rms(obs_rms)
    return rms, (rms.mean, rms.var, np.float64(c), np.float64(sigma))


def state_critic(f: Callable, obs_rms: RunningMeanStd, widths: tuple[float, float],
                 sigma: float = 1.0, name: str = "planted-state") -> CriticNet:
    """V_c(o) = f(x, a_prev) / sigma with (x, a_prev) decoded from o (``decode``; f must not
    read s): a critic whose pullback is grad f, pi = df/da_prev, with or without accel."""
    rms, prm = _plant_params(obs_rms, 0.0, sigma)
    return CriticNet(_state_fn(f, tuple(map(float, widths))), prm, rms, float(sigma), 0, name)


def accel_normal_critic(f: Callable, phi: Callable, c: float, obs_rms: RunningMeanStd,
                        widths: tuple[float, float], sigma: float = 1.0,
                        name: str = "accel-normal") -> CriticNet:
    """K3's accel-normal plant: V_c = [f(x, a) + c phi(z_acc - A(x, a))] / sigma, where
    z_acc is the observed accel and A(x, a) the accel consistent with (x_t, a_{t-1})
    (``manifold_accel``). On the manifold its primary pullback is grad f and pi = df/da for
    every c and phi; the accel-held pullback carries c phi'(0) and fails. c and sigma are
    parameters, so one compile serves every c."""
    rms, prm = _plant_params(obs_rms, c, sigma)
    return CriticNet(_accel_normal_fn(f, phi, tuple(map(float, widths))), prm, rms,
                     float(sigma), 0, f"{name}(c={c:g})")


# ------------------------------------------------------------------------- the observation map


class Inputs(NamedTuple):
    """Rows of states to pull back at (see the module docstring)."""

    x: np.ndarray             # [n, 8] x_t
    x_prev: np.ndarray        # [n, 8] x_{t-1} (reset rows: x_t, unused)
    a_prev: np.ndarray        # [n, 2] a_{t-1} (reset rows: 0)
    reset: np.ndarray         # [n] bool: accel = prev_action = 0 (an A2 key's t = 0)

    def __len__(self) -> int:
        return len(self.x)

    @property
    def prev_action(self) -> np.ndarray:
        """What the network reads in channels 210-211 (racing.py:150, 181)."""
        return np.where(self.reset[:, None], 0.0, np.clip(self.a_prev, -1.0, 1.0))

    @property
    def accel(self) -> np.ndarray:
        return np.where(self.reset[:, None], 0.0, accel_of(self.x, self.x_prev))

    def take(self, idx) -> Inputs:
        return Inputs(*(np.asarray(a)[idx] for a in self))


def accel_of(x, x_prev, dt: float = DT):
    """The env's body-frame acceleration over the step x_{t-1} -> x_t (racing.py:155-156);
    NumPy or JAX, rows [..., 8]."""
    xp = jnp if isinstance(x, jax.Array) else np
    return xp.stack([(x[..., 3] - x_prev[..., 3]) / dt - x[..., 4] * x[..., 5],
                     (x[..., 4] - x_prev[..., 4]) / dt + x[..., 3] * x[..., 5]], axis=-1)


def key_inputs(x: np.ndarray, a: np.ndarray, steps, anchored: bool = False) -> Inputs:
    """Rows at a key's plan steps t: x_t, x_{t-1}, a_{t-1} from the key's arrays (x [W+1, 8],
    a [W, 2]). t = 0 of an A2 key is a reset row; an anchored key's x_0 had the agent's own
    a_{L-2} and accel, which the key does not hold (design Section 2), so its t = 0 is
    refused (N uses t = 1, with a_0 = the fixed a_{L-1})."""
    t = np.atleast_1d(np.asarray(steps, int))
    x = np.asarray(x, np.float64)
    a = np.asarray(a, np.float64)
    if np.any(t < 0) or np.any(t > len(a)):
        raise ValueError(f"steps outside 0..{len(a)}")
    reset = t == 0
    if anchored and reset.any():
        raise ValueError("an anchored key's x_0 is not a reset state; use t >= 1")
    tp = np.maximum(t - 1, 0)
    return Inputs(x[t].copy(), np.where(reset[:, None], x[t], x[tp]),
                  np.where(reset[:, None], 0.0, a[tp]), reset)


def _ctx():
    return jnp.asarray(NW.ctx64()), jnp.asarray(NW.context_channels())


def _obs_fn(tr, ctx, ctx_obs):
    def ob(x, pa, acc):
        return NW.observe(NW.env_state(x, pa, acc, ctx), tr, ctx_obs)

    return ob


def _phi_fn(tr, ctx):
    p = racing.params_from_context(ctx, hidden_mu=1.0)
    return lambda x, a: control_step(x, a, tr, p, DT, SUBSTEPS)


@functools.lru_cache(maxsize=1)
def _obs_jit():
    def one(tr, x, xp, ap, reset, ctx, ctx_obs):
        pa = jnp.where(reset, 0.0, jnp.clip(ap, -1.0, 1.0))
        acc = jnp.where(reset, 0.0, accel_of(x, xp))
        obf = _obs_fn(tr, ctx, ctx_obs)
        ob = obf(x, pa, acc)
        J, Jp, Ja = jax.jacfwd(obf, argnums=(0, 1, 2))(x, pa, acc)
        phi = _phi_fn(tr, ctx)
        Px, Pa = jax.jacfwd(phi, argnums=(0, 1))(xp, pa)
        Ax, Axp = jax.jacfwd(accel_of, argnums=(0, 1))(x, xp)
        Mx = Ax + jnp.linalg.solve(Px.T, Axp.T).T
        Ma = -Axp @ jnp.linalg.solve(Px, Pa)
        z = jnp.zeros_like
        return (ob, J, Jp, Ja, jnp.where(reset, z(Ax), Ax), jnp.where(reset, z(Mx), Mx),
                jnp.where(reset, z(Ma), Ma), jnp.where(reset, z(x), phi(xp, pa) - x))

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, 0, None, None)))


def _batched(f, n: int, *arrays, batch: int = BATCH):
    """f over rows in chunks of ``batch`` (the last padded with its first row), so each jit
    compiles once; returns the concatenated outputs (a tuple when f returns one)."""
    if n <= 0:
        raise ValueError("no rows")
    outs = []
    for i in range(0, n, batch):
        idx = np.arange(i, min(i + batch, n))
        pad = np.r_[idx, np.full(batch - len(idx), idx[0])]
        res = f(*(np.asarray(a)[pad] for a in arrays))
        res = res if isinstance(res, tuple) else (res,)
        outs.append(tuple(np.asarray(r)[:len(idx)] for r in res))
    cat = tuple(np.concatenate(parts) for parts in zip(*outs, strict=True))
    return cat if len(cat) > 1 else cat[0]


@dataclass(frozen=True)
class ObsJac:
    """The network-independent part of the pullback at rows of states."""

    ob: np.ndarray            # [n, 222] raw observation
    J: np.ndarray             # [n, 222, 8] dob/dx_t, accel and prev_action held
    J_prev: np.ndarray        # [n, 222, 2] dob/dprev_action
    J_acc: np.ndarray         # [n, 222, 2] dob/daccel
    Ax: np.ndarray            # [n, 2, 8] dA/dx_t (0 on reset rows)
    Mx: np.ndarray            # [n, 2, 8] on-manifold dA/dx_t (0 on reset rows)
    Ma: np.ndarray            # [n, 2, 2] on-manifold dA/da_{t-1} (0 on reset rows)
    defect: np.ndarray        # [n, 8] Phi(x_{t-1}, a_{t-1}) - x_t (0 on reset rows)
    reset: np.ndarray         # [n] bool

    @property
    def frenet_leak(self) -> float:
        """max |M_x| on the (s, n, xi) columns (0 up to round-off; Section 3.2)."""
        return float(np.max(np.abs(self.Mx[:, :, :3]), initial=0.0))

    def z(self, obs_rms: RunningMeanStd) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(z unclipped [n, 222], clipped mask, sd [222]) under a network's obs_rms."""
        mean, var = (np.asarray(a, np.float64) for a in (obs_rms.mean, obs_rms.var))
        sd = np.sqrt(var + RMS_EPS)
        z = (self.ob - mean) / sd
        return z, np.abs(z) >= OBS_CLIP, sd

    def jz(self, obs_rms: RunningMeanStd) -> tuple[np.ndarray, np.ndarray]:
        """(J_z = dz/dx_t [n, 222, 8], clipped [n, 222]): the observability input
        (``states.observability``)."""
        _, clipped, sd = self.z(obs_rms)
        return self.J / sd[None, :, None], clipped


def obs_jacobians(tr: TrackArrays, inp: Inputs) -> ObsJac:
    """ob, its Jacobians and the on-manifold maps at every row (float64; the module
    docstring)."""
    NW.require_x64()
    ctx, ctx_obs = _ctx()
    f = _obs_jit()
    res = _batched(lambda *a: f(tr, *a, ctx, ctx_obs), len(inp), inp.x, inp.x_prev, inp.a_prev,
                   inp.reset)
    return ObsJac(*res, reset=np.asarray(inp.reset, bool))


# ------------------------------------------------------------------------- the pullback


@functools.lru_cache(maxsize=64)
def _grad_jit(fn: Callable):
    def one(params, o):
        y, vjp = jax.vjp(lambda o_: fn(params, o_), o)
        return y, jax.vmap(lambda c: vjp(c)[0])(jnp.eye(y.shape[0], dtype=o.dtype))

    return jax.jit(jax.vmap(one, in_axes=(None, 0)))


@functools.lru_cache(maxsize=64)
def _forward_jit(fn: Callable):
    return jax.jit(jax.vmap(fn, in_axes=(None, 0)))


@dataclass(frozen=True)
class Pullback:
    """A critic's outputs and their pullbacks at rows of states (module docstring). Output
    axis: 0 = V, 1.. = heads; g and pi in reward units per state / action unit (sigma_R
    applied), y in the critic's own units (reward / sigma_R)."""

    name: str
    sigma: float
    reset: np.ndarray         # [n]
    y: np.ndarray             # [n, 1 + E]
    g: np.ndarray             # [n, 1 + E, 8] primary (on-manifold; accel-held on reset rows)
    pi: np.ndarray            # [n, 1 + E, 2]
    g_held: np.ndarray        # [n, 1 + E, 8] accel held (sensitivity analysis)
    pi_held: np.ndarray       # [n, 1 + E, 2]
    g_xprev: np.ndarray       # [n, 1 + E, 8] x_{t-1} held (descriptive)
    accel_sens: np.ndarray    # [n, 1 + E, 2] sigma_R dV/daccel (per m/s^2)
    z: np.ndarray             # [n, 222] unclipped
    clipped: np.ndarray       # [n, 222] bool

    @property
    def value(self) -> np.ndarray:
        """[n, 1 + E] in reward units (sigma_R V_c)."""
        return self.sigma * self.y

    @property
    def n_heads(self) -> int:
        return self.y.shape[1] - 1

    @property
    def accel_term(self) -> np.ndarray:
        """[n, 1 + E, 8] g - g_held: the accel term of the primary pullback."""
        return self.g - self.g_held

    @property
    def accel_share(self) -> np.ndarray:
        """[n, 1 + E] the accel term's share of g_vx, |term| / (|g_held| + |term|)."""
        t = np.abs(self.accel_term[..., 3])
        d = np.abs(self.g_held[..., 3]) + t
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 0, t / d, 0.0)


def values(net: CriticNet, tr: TrackArrays, x: np.ndarray, prev_action: np.ndarray,
           accel: np.ndarray) -> np.ndarray:
    """[n, 1 + E] critic outputs (reward / sigma_R) at states with the given prev_action and
    accel (no clip of prev_action: the caller's values are read as given)."""
    _refuse_trained(net, "a value")
    NW.require_x64()
    ob = _observations(tr, x, prev_action, accel)
    o = np.clip((ob - np.asarray(net.obs_rms.mean)) / np.sqrt(np.asarray(net.obs_rms.var)
                                                              + RMS_EPS), -OBS_CLIP, OBS_CLIP)
    f = _forward_jit(net.fn)
    return _batched(lambda o_: f(net.params, o_), len(o), o)


def pullback(net: CriticNet, tr: TrackArrays, inp: Inputs, oj: ObsJac | None = None
             ) -> Pullback:
    """Every output's primary, accel-held and x_{t-1}-held pullbacks and the accel-channel
    sensitivity at the rows of ``inp`` (``oj``: their ``obs_jacobians``, shared by critics)."""
    _refuse_trained(net, "a pullback")
    NW.require_x64()
    if oj is None:
        oj = obs_jacobians(tr, inp)
    z, clipped, sd = oj.z(net.obs_rms)
    o = np.clip(z, -OBS_CLIP, OBS_CLIP)
    f = _grad_jit(net.fn)
    y, dy = _batched(lambda o_: f(net.params, o_), len(o), o)
    w = np.where(clipped[:, None, :], 0.0, dy) / sd           # [n, K, 222]: dY/dob
    s = net.sigma
    g_held = s * np.einsum("nkc,ncd->nkd", w, oj.J)
    pi_held = s * np.einsum("nkc,ncd->nkd", w, oj.J_prev)
    acc = s * np.einsum("nkc,ncd->nkd", w, oj.J_acc)
    keep = ~oj.reset[:, None, None]
    g = g_held + keep * np.einsum("nke,ned->nkd", acc, oj.Mx)
    pi = pi_held + keep * np.einsum("nke,ned->nkd", acc, oj.Ma)
    g_xprev = g_held + keep * np.einsum("nke,ned->nkd", acc, oj.Ax)
    return Pullback(net.name, float(s), oj.reset.copy(), y, g, pi, g_held, pi_held, g_xprev,
                    acc, z, clipped)


# ------------------------------------------------------------------------- C1-heads


def head_quantities(pb: Pullback) -> dict[str, np.ndarray]:
    """q -> [n, E] per head: V (reward units), g_k for every state component, pi_steer and
    pi_pedal (Section 6 C1-heads)."""
    if not pb.n_heads:
        raise ValueError(f"{pb.name} has no heads")
    q = {"V": pb.value[:, 1:]}
    q.update({f"g_{nm}": pb.g[:, 1:, k] for k, nm in enumerate(STATE_NAMES)})
    q.update({nm: pb.pi[:, 1:, j] for j, nm in enumerate(PI_NAMES)})
    return q


def _wts(n: int, w) -> np.ndarray:
    w = np.ones(n) if w is None else np.asarray(w, float)
    return w / w.sum()


def head_disagreement(q: np.ndarray, w: np.ndarray | None = None) -> float:
    """D_q = mean_t var_k(q^k_t) / var_t(mean_k q^k_t) (q [n, E]): the variance over heads
    with ddof 1 (unbiased for the head-level variance), states t weighted by ``w``."""
    q = np.asarray(q, float)
    w = _wts(len(q), w)
    within = float(w @ q.var(axis=1, ddof=1))
    m = q.mean(axis=1)
    between = float(w @ (m - w @ m) ** 2)
    return within / between if between > 0 else float("inf") if within > 0 else 0.0


def head_reliability(q: np.ndarray, w: np.ndarray | None = None) -> float:
    """rho_q: the mean over head pairs of their weighted correlation over states (q [n, E])."""
    q = np.asarray(q, float)
    w = _wts(len(q), w)
    c = q - w @ q
    cov = (c * w[:, None]).T @ c
    sd = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.outer(sd, sd)
    iu = np.triu_indices(q.shape[1], 1)
    return float(np.nanmean(r[iu]))


# ------------------------------------------------------------------------- K3: finite differences


@functools.lru_cache(maxsize=1)
def _ob_jit():
    def one(tr, x, pa, acc, ctx, ctx_obs):
        ob = _obs_fn(tr, ctx, ctx_obs)(x, pa, acc)
        fo = tyre_forces(x, racing.params_from_context(ctx, hidden_mu=1.0), JAX)
        sk = x[0] + jnp.concatenate([jnp.zeros(1, x.dtype),
                                     racing.preview_offsets(jnp.maximum(x[3], 0.0))])
        return ob, jnp.stack([fo.util_f, fo.util_r]), jnp.floor(sk / tr.ds)

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, None, None)))


@functools.lru_cache(maxsize=1)
def _phi_jit():
    def one(tr, x, a, ctx):
        phi = _phi_fn(tr, ctx)
        return phi(x, a), jax.jacfwd(phi)(x, a)

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, None)))


def _observations(tr, x, pa, acc, with_kinks: bool = False):
    """ob [n, 222] (and, with ``with_kinks``, the tyre utilisations [n, 2] and the pose
    lookups' segment indices [n, 41]: the car's own and the 40 preview points')."""
    ctx, ctx_obs = _ctx()
    f = _ob_jit()
    ob, util, seg = _batched(lambda *a: f(tr, *a, ctx, ctx_obs), len(x), x, pa, acc)
    return (ob, util, seg) if with_kinks else ob


def prev_state(tr: TrackArrays, x: np.ndarray, a: np.ndarray, guess: np.ndarray | None = None,
               iters: int = NEWTON_ITERS) -> tuple[np.ndarray, np.ndarray]:
    """x_{t-1} with Phi(x_{t-1}, a) = x_t (rows [n, 8], a [n, 2], unclipped), by Newton on
    ``sim.control_step`` from ``guess`` (default x_t); returns (x_{t-1}, max |residual| per
    row). The K3 finite differences' route back to the manifold (Section 7)."""
    NW.require_x64()
    ctx, _ = _ctx()
    f = _phi_jit()
    x = np.asarray(x, np.float64)
    a = np.asarray(a, np.float64)
    xp = np.array(x if guess is None else guess, np.float64)
    for _ in range(iters):
        F, Jx = _batched(lambda *v: f(tr, *v, ctx), len(x), xp, a)
        r = F - x
        step = np.linalg.solve(Jx, r[..., None])[..., 0]
        xp = xp - step
        if np.all(np.abs(step) <= 4 * np.finfo(float).eps * (1.0 + np.abs(xp))):
            break
    F, _ = _batched(lambda *v: f(tr, *v, ctx), len(x), xp, a)
    return xp, np.max(np.abs(F - x), axis=1)


def _region(v: np.ndarray, knots) -> np.ndarray:
    return np.stack([v <= k for k in knots], axis=-1)


def _signature(z: np.ndarray, util: np.ndarray, seg: np.ndarray, x: np.ndarray,
               xp: np.ndarray | None) -> np.ndarray:
    """[..., m]: the discrete structure of an evaluation (module docstring), with the pose
    lookups' segments (sim.lookup is piecewise linear on the 2 m knots, sim.py:64-73: a
    central difference across a knot near the centre passes the O(h^2) check but averages
    the two one-sided slopes)."""
    vx = x[..., 3]
    parts = [np.abs(z) >= OBS_CLIP, util >= 2.0, np.stack([6.0 * vx <= 150.0, 6.0 * vx >= 500.0,
                                                           vx <= 1.0], axis=-1),
             _region(x[..., 7], PEDAL_KNOTS)]
    if xp is not None:
        parts.append(_region(xp[..., 7], PEDAL_KNOTS))
    return np.concatenate([np.concatenate(parts, axis=-1).astype(float), seg], axis=-1)


class _Points(NamedTuple):
    x: np.ndarray             # [n, D, P, 8]
    pa: np.ndarray            # [n, D, P, 2]
    acc: np.ndarray           # [n, D, P, 2]
    xp: np.ndarray | None     # [n, D, P, 8] (manifold and xprev variants)
    h: np.ndarray             # [D]
    newton: float             # max Newton residual (manifold)


def _points(tr: TrackArrays, inp: Inputs, variant: str, h: np.ndarray, with_pi: bool
            ) -> _Points:
    """Perturbed inputs: direction d < 8 moves x_t[d], d = 8, 9 moves a_{t-1}; FD_STEPS
    multiples of h[d]."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} is not one of {VARIANTS}")
    n, D = len(inp), 8 + 2 * with_pi
    st = np.asarray(FD_STEPS)
    P = len(st)
    x = np.broadcast_to(inp.x[:, None, None], (n, D, P, 8)).copy()
    pa = np.broadcast_to(inp.prev_action[:, None, None], (n, D, P, 2)).copy()
    for d in range(8):
        x[:, d, :, d] += st * h[d]
    for j in range(2 * with_pi):
        pa[:, 8 + j, :, j] += st * h[8 + j]
    reset = np.broadcast_to(inp.reset[:, None, None], (n, D, P))
    acc0 = np.broadcast_to(inp.accel[:, None, None], (n, D, P, 2))
    xp = np.broadcast_to(inp.x_prev[:, None, None], (n, D, P, 8)).copy()
    newton = 0.0
    if variant == "manifold":
        live = ~reset
        if live.any():
            sol, res = prev_state(tr, x[live], pa[live], xp[live])
            xp[live] = sol
            newton = float(res.max())
    acc = acc0 if variant == "held" else np.where(reset[..., None], 0.0, accel_of(x, xp))
    return _Points(x, pa, acc, None if variant == "held" else xp, h, newton)


def _richardson(Y: np.ndarray, h: np.ndarray):
    """Y [n, D, P, K] at FD_STEPS -> (Richardson value, |FD(h) - FD(h/2)|, smooth) [n, D, K]."""
    hh = h[None, :, None]
    d1 = (Y[:, :, 4] - Y[:, :, 0]) / (2.0 * hh)
    d2 = (Y[:, :, 3] - Y[:, :, 1]) / hh
    R = (4.0 * d2 - d1) / 3.0
    disc = np.abs(d1 - d2)
    noise = ROUNDOFF * np.finfo(float).eps * np.max(np.abs(Y), axis=2) / hh
    return R, disc, disc <= RICH_RTOL * np.abs(R) + noise


def _same(sig: np.ndarray) -> np.ndarray:
    """sig [n, D, P, m] -> [n, D]: one structure at every point."""
    return np.all(sig == sig[:, :, 2:3], axis=(2, 3))


def _field_error(ad: np.ndarray, fd: np.ndarray, counted: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(||AD - FD|| / ||FD|| over the counted components of the last axis, any counted):
    step 2's field-vector error (network.py:81-82)."""
    m = np.asarray(counted, bool)
    num = np.sqrt(np.sum(np.where(m, (ad - fd) ** 2, 0.0), axis=-1))
    den = np.sqrt(np.sum(np.where(m, fd ** 2, 0.0), axis=-1))
    some = m.any(axis=-1)
    return np.where(some, num / np.maximum(den, 1e-300), 0.0), some


def _component_error(ad: np.ndarray, fd: np.ndarray) -> np.ndarray:
    """|AD - FD| / max(|FD|, ZERO_REL ||FD||) per component (descriptive)."""
    floor = ZERO_REL * np.linalg.norm(fd, axis=-1, keepdims=True)
    return np.abs(ad - fd) / np.maximum(np.abs(fd), np.maximum(floor, 1e-300))


@dataclass(frozen=True)
class FDCheck:
    """AD against Richardson finite differences per (row, output, direction); directions are
    the 8 state components (g), then pi_steer and pi_pedal (pi). The gate is the field-vector
    error of g and of pi over their counted directions (``max_err`` < ``FD_TOL``);
    ``comp_err`` is per component, descriptive."""

    variant: str
    ad: np.ndarray            # [n, K, 10]
    fd: np.ndarray            # [n, K, 10]
    counted: np.ndarray       # [n, K, 10] bool
    err_g: np.ndarray         # [n, K] field-vector error over the counted state directions
    err_pi: np.ndarray        # [n, K] over the counted pi directions
    comp_err: np.ndarray      # [n, K, 10]
    newton: float             # max Newton residual of the recovered x_{t-1}

    def max_err_of(self, outputs=slice(None)) -> float:
        e = np.concatenate([self.err_g[:, outputs], self.err_pi[:, outputs]], axis=-1)
        return float(np.max(e, initial=0.0))

    @property
    def max_err(self) -> float:
        return self.max_err_of()

    @property
    def share_counted(self) -> float:
        return float(np.mean(self.counted))

    @property
    def ok(self) -> bool:
        return bool(self.max_err < FD_TOL)


def fd_pullback(net: CriticNet, tr: TrackArrays, inp: Inputs, variant: str = "manifold",
                pb: Pullback | None = None, h: dict[str, float] | None = None,
                h_pi: float = PI_H) -> FDCheck:
    """K3: ``pullback``'s ``variant`` (manifold: g and pi; held: g_held and pi_held; xprev:
    g_xprev, with pi_held) against finite differences of sigma_R x every output."""
    _refuse_trained(net, "finite differences")
    NW.require_x64()
    hs = np.array([(h or FD_H)[nm] for nm in STATE_NAMES] + [h_pi, h_pi])
    if pb is None:
        pb = pullback(net, tr, inp)
    pts = _points(tr, inp, variant, hs, True)
    n, D, P = pts.x.shape[:3]
    flat = (lambda a: a.reshape(n * D * P, a.shape[-1]))
    ob, util, seg = _observations(tr, flat(pts.x), flat(pts.pa), flat(pts.acc), True)
    mean, var = (np.asarray(a, np.float64) for a in (net.obs_rms.mean, net.obs_rms.var))
    z = (ob - mean) / np.sqrt(var + RMS_EPS)
    o = np.clip(z, -OBS_CLIP, OBS_CLIP)
    f = _forward_jit(net.fn)
    Y = _batched(lambda o_: f(net.params, o_), len(o), o).reshape(n, D, P, -1) * net.sigma
    R, _, smooth = _richardson(Y, hs)                          # [n, D, K]
    sig = _signature(z, util, seg, flat(pts.x), None if pts.xp is None else flat(pts.xp))
    same = _same(sig.reshape(n, D, P, -1))                      # [n, D]
    counted = smooth & same[..., None]
    if net.kinks is not None and net.n_heads:
        kf = _forward_jit(net.kinks)
        hk = _batched(lambda o_: kf(net.params, o_), len(o), o).reshape(n, D, P, net.n_heads, -1)
        head_same = np.all(hk == hk[:, :, 2:3], axis=(2, 4))  # [n, D, E]
        counted[..., 1:] &= head_same
    g = {"manifold": (pb.g, pb.pi), "held": (pb.g_held, pb.pi_held),
         "xprev": (pb.g_xprev, pb.pi_held)}[variant]
    ad = np.concatenate(g, axis=-1)                             # [n, K, D]
    fd = np.moveaxis(R, 1, 2)
    cnt = np.moveaxis(counted, 1, 2)
    eg, _ = _field_error(ad[..., :8], fd[..., :8], cnt[..., :8])
    ep, _ = _field_error(ad[..., 8:], fd[..., 8:], cnt[..., 8:])
    ce = np.concatenate([_component_error(ad[..., :8], fd[..., :8]),
                         _component_error(ad[..., 8:], fd[..., 8:])], axis=-1)
    return FDCheck(variant, ad, fd, cnt, eg, ep, ce, pts.newton)


@dataclass(frozen=True)
class JacCheck:
    """J_z (accel and prev_action held) against Richardson differences: the gate is the
    relative Frobenius error over the counted columns per row (``max_err``); ``comp_err``
    per column, ||dJ_k|| / max(||J_k||, ZERO_REL ||J||), descriptive."""

    counted: np.ndarray       # [n, 8] bool
    err: np.ndarray           # [n]
    comp_err: np.ndarray      # [n, 8]

    @property
    def max_err(self) -> float:
        return float(np.max(self.err, initial=0.0))

    @property
    def share_counted(self) -> float:
        return float(np.mean(self.counted))


def fd_jacobian(tr: TrackArrays, inp: Inputs, obs_rms: RunningMeanStd, oj: ObsJac | None = None,
                h: dict[str, float] | None = None) -> JacCheck:
    """K3: J_z by AD against float64 central differences of z (Section 7), a column counted
    where the structure (``_signature``: clip mask, tyre min(., 2), d_max clip, pedal knots,
    pose segments) is unchanged within +-h and the Richardson pair agrees on every channel."""
    NW.require_x64()
    hs = np.array([(h or FD_H)[nm] for nm in STATE_NAMES])
    if oj is None:
        oj = obs_jacobians(tr, inp)
    Jz, _ = oj.jz(obs_rms)
    pts = _points(tr, inp, "held", hs, False)
    n, D, P = pts.x.shape[:3]
    flat = (lambda a: a.reshape(n * D * P, a.shape[-1]))
    ob, util, seg = _observations(tr, flat(pts.x), flat(pts.pa), flat(pts.acc), True)
    mean, var = (np.asarray(a, np.float64) for a in (obs_rms.mean, obs_rms.var))
    z = (ob - mean) / np.sqrt(var + RMS_EPS)
    R, _, smooth = _richardson(z.reshape(n, D, P, -1), hs)     # [n, 8, 222]
    same = _same(_signature(z, util, seg, flat(pts.x), None).reshape(n, D, P, -1))
    counted = same & smooth.all(axis=2)                         # [n, 8]
    fdJ = np.moveaxis(R, 1, 2)                                  # [n, 222, 8]
    err, _ = _field_error(np.moveaxis(Jz, 1, 2).reshape(n, -1),
                          np.moveaxis(fdJ, 1, 2).reshape(n, -1),
                          np.repeat(counted, OBS_DIM, axis=1))
    norms = np.linalg.norm(fdJ, axis=1)
    floor = ZERO_REL * np.linalg.norm(norms, axis=1, keepdims=True)
    comp = np.linalg.norm(Jz - fdJ, axis=1) / np.maximum(norms, np.maximum(floor, 1e-300))
    return JacCheck(counted, err, comp)


def fd_manifold_maps(tr: TrackArrays, inp: Inputs, oj: ObsJac | None = None,
                     h: dict[str, float] | None = None, h_pi: float = PI_H) -> dict:
    """K3: M_x and M_a against differences of A(x_t, x_{t-1}(x_t, a_{t-1})) along the
    manifold (x_{t-1} by Newton). Non-reset rows; relative error per row over the matrix."""
    NW.require_x64()
    live = ~np.asarray(inp.reset, bool)
    sub = inp.take(np.flatnonzero(live))
    if oj is None:
        oj = obs_jacobians(tr, sub)
    else:
        oj = ObsJac(*(np.asarray(getattr(oj, f))[live] for f in ObsJac.__dataclass_fields__))
    hs = np.array([(h or FD_H)[nm] for nm in STATE_NAMES] + [h_pi, h_pi])
    pts = _points(tr, sub, "manifold", hs, True)
    R, _, smooth = _richardson(pts.acc, hs)                     # [n, 10, 2]
    ad = np.concatenate([oj.Mx, oj.Ma], axis=-1)                # [n, 2, 10]
    fd = np.moveaxis(R, 1, 2)
    scale = np.max(np.abs(fd), axis=(1, 2), keepdims=True)
    err = np.abs(ad - fd) / np.maximum(scale, 1e-300)
    return {"max_err": float(err.max(initial=0.0)), "smooth": float(smooth.mean()),
            "newton": pts.newton, "frenet_leak": oj.frenet_leak}


# ------------------------------------------------------------------------- dry runs (float64)


def _synthetic_road(w_left: float = 7.0, w_right: float = 5.5, n: int = 700, ds: float = 2.0
                    ) -> TrackArrays:
    """A float32 strip (T64-upcast) with a smooth corner and exactly constant widths."""
    from apexzero.tracks.generator import strip_arrays

    s = np.arange(n) * ds
    kappa = (1.0 / 70.0) * np.exp(-0.5 * ((s - 420.0) / 60.0) ** 2)
    with jax.enable_x64(False):
        tr = strip_arrays(jnp.asarray(kappa, jnp.float32), jnp.full(n, w_left, jnp.float32),
                          jnp.full(n, w_right, jnp.float32), ds)
    tr = NW.upcast32(jax.tree.map(np.asarray, tr))
    # the float32 spline solve leaves ~1e-7 wiggles on a constant width: equal coefficients
    # make the widths exactly constant (a B-spline's basis sums to 1), so n decodes exactly
    return tr._replace(w_left_c=np.full_like(tr.w_left_c, w_left),
                       w_right_c=np.full_like(tr.w_right_c, w_right))


PLANT_STARTS = (   # x_{t-1} and a_{t-1} of the planted rows (t >= 1)
    ((100.0, 0.5, 0.02, 60.0, 0.3, 0.05, 0.02, 0.7), (0.1, 0.9)),
    ((330.0, -1.0, -0.03, 35.0, -0.5, -0.2, -0.05, -0.6), (-0.3, -0.8)),
    ((420.0, 2.0, 0.05, 20.0, 0.2, 0.4, 0.1, 0.3), (0.4, 0.5)),
    ((250.0, 0.0, 0.0, 70.0, 0.0, 0.0, 0.0, 1.0), (0.0, 1.0)),
    ((380.0, -0.3, 0.04, 45.0, 0.6, 0.25, 0.06, -0.9), (0.6, -1.0)),
)
PLANT_RESET = ((151.3, 0.8, -0.01, 55.0, 0.0, 0.0, 0.0, 0.4),)


def plant_inputs(tr: TrackArrays) -> Inputs:
    """Rows exactly on the env's manifold: x_t = control_step(x_{t-1}, a_{t-1}) in float64,
    plus a reset row."""
    xp = np.array([s for s, _ in PLANT_STARTS])
    a = np.array([u for _, u in PLANT_STARTS])
    F, _ = _phi_jit()(tr, jnp.asarray(xp), jnp.asarray(a), _ctx()[0])
    xr = np.array(PLANT_RESET)
    return Inputs(np.r_[np.asarray(F), xr], np.r_[xp, xr], np.r_[a, np.zeros((len(xr), 2))],
                  np.r_[np.zeros(len(xp), bool), np.ones(len(xr), bool)])


def plant_rms(seed: int = 5) -> RunningMeanStd:
    """A synthetic normaliser (means near 0, variances in [0.5, 3])."""
    rng = np.random.default_rng(seed)
    return RunningMeanStd(rng.normal(0.0, 0.05, OBS_DIM), rng.uniform(0.5, 3.0, OBS_DIM),
                          np.float64(1e6))


def plant_f(x, a):
    """The planted value: a smooth function of (n, xi, vx, vy, r, delta, pedal, a_{t-1})."""
    n, xi, vx, vy, r, d, p = (x[i] for i in range(1, 8))
    return (0.3 * n - 0.8 * xi + 0.02 * vx + 0.001 * vx ** 2 - 0.4 * vy + 0.7 * r + 1.5 * d
            + 0.25 * p + 0.1 * n * vx / 50.0 + jnp.sin(xi) * vy + 0.05 * a[0] - 0.12 * a[1]
            + 0.3 * a[0] * d + 0.2 * a[1] * p)


def plant_phi(u):
    return jnp.sin(0.7 * u[0]) + 0.5 * u[1] + 0.01 * u[1] ** 2


def _composed(net: CriticNet, tr: TrackArrays, inp: Inputs) -> dict:
    """g_held and g_xprev by autodiff of the composition V(clip(z(observe(x, ...)))) (no
    chain rule by hand): an independent route to two of ``pullback``'s variants."""
    ctx, ctx_obs = _ctx()
    mean, var = (jnp.asarray(a, jnp.float64) for a in (net.obs_rms.mean, net.obs_rms.var))
    trj = jax.tree.map(jnp.asarray, tr)

    def v_of(x, pa, acc):
        ob = _obs_fn(trj, ctx, ctx_obs)(x, pa, acc)
        o = jnp.clip((ob - mean) / jnp.sqrt(var + RMS_EPS), -OBS_CLIP, OBS_CLIP)
        return net.fn(net.params, o)[0] * net.sigma

    held = jax.jit(jax.vmap(jax.grad(v_of, argnums=(0, 1))))
    xprev = jax.jit(jax.vmap(jax.grad(
        lambda x, pa, xp, rs: v_of(x, pa, jnp.where(rs, 0.0, accel_of(x, xp))))))
    gh, ph = held(inp.x, inp.prev_action, inp.accel)
    gx = xprev(inp.x, inp.prev_action, inp.x_prev, inp.reset)
    return {"g_held": np.asarray(gh), "pi_held": np.asarray(ph), "g_xprev": np.asarray(gx)}


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def dry_run_plants() -> tuple[dict, dict]:
    """K3 on the synthetic strip: the manifold maps, the planted linear critics' closed
    forms (V and planted heads), the state-function critic and the accel-normal plant (for
    several c), each against its known pullback and against finite differences."""
    t0 = time.perf_counter()
    wl, wr = 7.0, 5.5
    tr = _synthetic_road(wl, wr)
    inp = plant_inputs(tr)
    rms = plant_rms()
    oj = obs_jacobians(tr, inp)
    _, clipped, sd = oj.z(rms)
    rep: dict = {"n": len(inp), "reset": inp.reset.tolist(), "any_clipped": bool(clipped.any())}
    rep["compile_seconds"] = time.perf_counter() - t0
    # the env's own accel and step on the planted rows
    trj = jax.tree.map(jnp.asarray, tr)
    _, st, *_ = jax.vmap(lambda x, pa, ac, a: racing.step_no_reset(
        NW.env_state(x, pa, ac, _ctx()[0]), a, trj, NW.ENV))(
        jnp.asarray(inp.x_prev), jnp.zeros((len(inp), 2)), jnp.zeros((len(inp), 2)),
        jnp.asarray(inp.a_prev))
    live = ~inp.reset
    rep["env_accel_err"] = float(np.max(np.abs(np.asarray(st.accel)[live] - inp.accel[live])))
    rep["env_step_err"] = float(np.max(np.abs(np.asarray(st.x)[live] - inp.x[live])))
    rep["defect"] = float(np.max(np.abs(oj.defect)))
    rep["frenet_leak"] = oj.frenet_leak
    rep["maps"] = fd_manifold_maps(tr, inp, oj)
    rep["J_acc_err"] = float(np.max(np.abs(oj.J_acc[:, ACCEL] - np.eye(2) / ACCEL_SCALE)))
    rep["J_acc_other"] = float(np.max(np.abs(np.delete(oj.J_acc, np.r_[208:210], axis=1))))
    rep["J_prev_err"] = float(np.max(np.abs(oj.J_prev[:, PREV] - np.eye(2))))
    rep["J_prev_other"] = float(np.max(np.abs(np.delete(oj.J_prev, np.r_[210:212], axis=1))))
    rep["J_accel_rows"] = float(np.max(np.abs(oj.J[:, 208:212])))
    # planted linear critics: closed forms
    sigma = 1.7
    lin = {}
    half = 0.5 * (wl + wr)
    for ch, comp, scale in (("vx", 3, 1.0 / 50.0), ("lateral", 1, 1.0 / half), ("xi", 2, 1.0),
                            ("pedal", 7, 1.0)):
        c = EGO[ch]
        pb = pullback(linear_critic(np.eye(OBS_DIM)[c], 0.3, rms, sigma), tr, inp, oj)
        want = np.zeros((len(inp), 8))
        want[:, comp] = sigma * scale / sd[c]
        lin[ch] = {"g": _rel(pb.g[:, 0], want), "g_held": _rel(pb.g_held[:, 0], want),
                   "pi": float(np.max(np.abs(pb.pi))), "accel": float(np.max(np.abs(pb.accel_sens)))}
    rng = np.random.default_rng(11)
    w = np.zeros(OBS_DIM)
    w[208:212] = rng.normal(size=4)
    pb = pullback(linear_critic(w, 0.0, rms, sigma), tr, inp, oj)
    acc = sigma * w[ACCEL] / (ACCEL_SCALE * sd[ACCEL])
    pih = sigma * w[PREV] / sd[PREV]
    keep = ~inp.reset[:, None]
    lin["accel_prev"] = {
        "g_held": float(np.max(np.abs(pb.g_held))),
        "accel_sens": _rel(pb.accel_sens[:, 0], np.broadcast_to(acc, (len(inp), 2))),
        "pi_held": _rel(pb.pi_held[:, 0], np.broadcast_to(pih, (len(inp), 2))),
        "g": _rel(pb.g[:, 0], keep * np.einsum("e,ned->nd", acc, oj.Mx)),
        "pi": _rel(pb.pi[:, 0], pih + keep * np.einsum("e,ned->nd", acc, oj.Ma)),
        "g_reset": float(np.max(np.abs(pb.g[inp.reset]))),
        "accel_share_vx": pb.accel_share[~inp.reset, 0].tolist()}
    Wh = np.stack([np.eye(OBS_DIM)[EGO["vx"]], np.eye(OBS_DIM)[EGO["xi"]], w])
    pbh = pullback(linear_critic(Wh, 0.0, rms, sigma), tr, inp, oj)
    lin["heads"] = {"n_heads": pbh.n_heads,
                    "head0_xi": _rel(pbh.g[:, 1, 2], sigma / sd[EGO["xi"]] * np.ones(len(inp))),
                    "head1": _rel(pbh.g[:, 2], pb.g[:, 0]),
                    "main": _rel(pbh.g[:, 0, 3], sigma / (50.0 * sd[EGO["vx"]]) * np.ones(len(inp)))}
    # a random linear critic over every channel: the composition's own autodiff
    wr_ = rng.normal(size=OBS_DIM) * 0.1
    net = linear_critic(wr_, 0.0, rms, sigma)
    pb = pullback(net, tr, inp, oj)
    comp_ = _composed(net, tr, inp)
    lin["composed"] = {"g_held": _rel(pb.g_held[:, 0], comp_["g_held"]),
                       "pi_held": _rel(pb.pi_held[:, 0], comp_["pi_held"]),
                       "g_xprev": _rel(pb.g_xprev[:, 0], comp_["g_xprev"])}
    fdc = {v: fd_pullback(net, tr, inp, v, pb) for v in VARIANTS}
    lin["fd"] = {v: {"max_err": c.max_err, "share": c.share_counted} for v, c in fdc.items()}
    rep["linear"] = lin
    rep["linear_seconds"] = time.perf_counter() - t0
    # the state-function critic and the accel-normal plant
    grad_f = jax.vmap(jax.grad(plant_f, argnums=(0, 1)))
    gx, ga = (np.asarray(v) * 1.0 for v in grad_f(jnp.asarray(inp.x), jnp.asarray(inp.prev_action)))
    plants = {}
    st_net = state_critic(plant_f, rms, (wl, wr), sigma)
    pb = pullback(st_net, tr, inp, oj)
    plants["state"] = {"g": _rel(pb.g[:, 0], gx), "pi": _rel(pb.pi[:, 0], ga),
                       "g_held": _rel(pb.g_held[:, 0], gx), "y": _rel(pb.value[:, 0], np.asarray(
                           jax.vmap(plant_f)(jnp.asarray(inp.x), jnp.asarray(inp.prev_action))))}
    for c in (0.0, 0.5, 20.0):
        net = accel_normal_critic(plant_f, plant_phi, c, rms, (wl, wr), sigma)
        pb = pullback(net, tr, inp, oj)
        live = ~inp.reset
        rec = {"g": _rel(pb.g[live, 0], gx[live]), "pi": _rel(pb.pi[live, 0], ga[live]),
               "g_held": _rel(pb.g_held[live, 0], gx[live]),
               "pi_held": _rel(pb.pi_held[live, 0], ga[live]),
               "g_xprev": _rel(pb.g_xprev[live, 0], gx[live]),
               "accel_sens": float(np.max(np.abs(pb.accel_sens[live, 0])))}
        if c == 20.0:
            for v in ("manifold", "held"):
                ck = fd_pullback(net, tr, inp, v, pb)
                rec[f"fd_{v}"] = {"max_err": ck.max_err, "share": ck.share_counted,
                                  "newton": ck.newton}
        plants[f"c={c:g}"] = rec
    rep["plants"] = plants
    rep["seconds"] = time.perf_counter() - t0
    return rep, {"x": inp.x, "Mx": oj.Mx, "Ma": oj.Ma}


EXAMPLE_KEY = "03b6076863e3adfe335e6757"     # design Section 2's example key


def _dry_keys(n_keys: int | None = None, min_states: int = K3_MIN_STATES):
    """A2 keys for K3 (stored arrays; only x and a are read): the example key, a W = 800 key
    and the smallest radius, then one nominal key of each further geometry in a permuted
    order (stream ``costate_k3``) until the rows reach ``min_states`` (Section 7 K3: 200 T
    states); ``n_keys`` instead takes exactly that many (a partial run). Each item: (ref,
    road, T's rows, s_t - s_c1, s_t - s_apex per row). Multi-basin keys are left out."""
    from apexzero.costate import keys_root, stimuli_dir, stream_rng
    from apexzero.costate import states as ST
    from apexzero.fields import keys as K

    bank, _ = K.design(stimuli_dir())
    refs = ST.a2_refs()
    pick = [r for r in refs if r.key_id == EXAMPLE_KEY]
    pick += [next(r for r in refs if r.W == 800 and r not in pick)]
    pick += [next(r for r in refs if r.stim_name.startswith("syn/R6/") and r not in pick)]
    seen = {(ST.strata(r).R, ST.strata(r).theta_deg) for r in pick}
    pool: dict[tuple, K.KeyRef] = {}
    for r in sorted(refs, key=lambda r: r.key_id):
        st = ST.strata(r)
        g = (st.R, st.theta_deg)
        if st.w_is_cell and r.kind == "nominal" and g not in seen and g not in pool:
            pool[g] = r
    order = sorted(pool)
    perm = stream_rng("costate_k3").permutation(len(order))
    more = [pool[order[i]] for i in perm]
    out, n_rows = [], 0
    for ref in [*pick, *more]:
        if n_keys is not None and len(out) >= n_keys:
            break
        if n_keys is None and n_rows >= min_states:
            break
        rec = K.load_key(ref, root=keys_root())
        if bool(getattr(rec, "multi_basin", False)):
            continue
        x, a = np.asarray(rec.x, np.float64), np.asarray(rec.a, np.float64)
        lm = bank.stimuli[rec.stim_id].landmarks
        t = ST.t_steps(x, lm, int(rec.W))
        out.append((ref, NW.road(bank, rec.stim_id).base, key_inputs(x, a, t),
                    x[t, 0] - lm.s_c1, x[t, 0] - lm.s_apex))
        n_rows += len(t)
    return out


def _f32(inp: Inputs) -> Inputs:
    """The key arrays rounded to float32 and upcast (K3: observability on float32-upcast
    against float64 key arrays)."""
    def r(a):
        return np.asarray(np.asarray(a, np.float32), np.float64)

    return Inputs(r(inp.x), r(inp.x_prev), r(inp.a_prev), inp.reset)


def observability_report(keys, ojs, rms, ojs32=None) -> dict:
    """K3's observability sub-checks (Section 7): s flagged on the exit straight (ds > 25 m
    past s_c1) and on none before s_c1; the transitional rows (0 < ds <= 10 m) and the apex
    rows (the T row nearest s_apex per key) reported with their s ratio; on the exit rows
    xi's and vx's squared column residuals with and without the s column and their relative
    difference (reported); and, with ``ojs32`` (``obs_jacobians`` on ``_f32`` inputs), the
    flags and ratios on float32-upcast key arrays."""
    from apexzero.costate import states as ST

    obs = [ST.observability(*oj.jz(rms)) for oj in ojs]
    ds = np.concatenate([k[3] for k in keys])
    da = [np.abs(k[4]) for k in keys]
    apex = np.concatenate([np.arange(len(d)) == int(np.argmin(d)) for d in da])
    ratio = np.concatenate([o.s_ratio for o in obs])
    flag = np.concatenate([o.s_unobservable for o in obs])
    exit_, before, trans = ds > 25.0, ds <= 0.0, (ds > 0.0) & (ds <= 10.0)
    rep = {"n_exit": int(exit_.sum()), "exit_flagged": int(flag[exit_].sum()),
           "n_before": int(before.sum()), "before_flagged": int(flag[before].sum()),
           "ratio_exit_max": float(ratio[exit_].max()) if exit_.any() else None,
           "ratio_before_min": float(ratio[before].min()) if before.any() else None,
           "n_transitional": int(trans.sum()), "transitional_flagged": int(flag[trans].sum()),
           "ratio_transitional": [float(ratio[trans].min()), float(ratio[trans].max())]
           if trans.any() else None,
           "n_apex": int(apex.sum()), "apex_flagged": int(flag[apex].sum()),
           "ratio_apex_min": float(ratio[apex].min()) if apex.any() else None}
    # xi's and vx's residuals with and without the s column on the exit rows
    J = np.concatenate([np.where(oj.jz(rms)[1][..., None], 0.0, oj.jz(rms)[0]) for oj in ojs])
    if exit_.any():
        Je = J[exit_]
        rel = {}
        for k, nm in ((2, "xi"), (3, "vx")):
            with_s = ST._project_out(Je[:, :, [j for j in range(8) if j != k]], Je[:, :, k],
                                     ST.LSTSQ_RCOND)
            without = ST._project_out(Je[:, :, [j for j in range(1, 8) if j != k]], Je[:, :, k],
                                      ST.LSTSQ_RCOND)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel[nm] = float(np.nanmax(np.abs(with_s - without) / np.maximum(without, 1e-300)))
        rep["exit_with_without_s"] = rel
    if ojs32 is not None:
        o32 = [ST.observability(*oj.jz(rms)) for oj in ojs32]
        r32 = np.concatenate([o.s_ratio for o in o32])
        f32 = np.concatenate([o.s_unobservable for o in o32])
        keep = before | trans | apex
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = np.abs(r32 - ratio) / np.maximum(ratio, 1e-300)
        rep["f32_same_flags"] = bool(np.array_equal(f32, flag))
        rep["f32_ratio_rel"] = float(rr[keep].max()) if keep.any() else 0.0
    return rep


def dry_run_random(agent: str = "G1", n_keys: int | None = 3, seeds=RANDOM_SEEDS,
                   fd_seeds=(0,), min_states: int = K3_MIN_STATES) -> tuple[dict, dict]:
    """K3 with random-init critics of ``agent``'s architecture through its obs_rms (read
    only; its own critic is never built or evaluated): the extractor against Flax (critic
    with heads, actor), the heads' stop-gradient, J_z and the pullbacks (main V and heads;
    primary, accel-held, x_{t-1}-held) against finite differences on a few keys' T states,
    and the 20 random critics' pullbacks."""
    t0 = time.perf_counter()
    g = NW._final_agent(agent)
    cfg, rms = g.cfg, NW.upcast32(g.obs_rms)
    rep: dict = {"agent": agent, "cfg": {k: getattr(cfg, k) for k in (
        "network", "critic_width", "actor_width", "mlp_layers", "activation", "ensemble_heads")}}
    keys = _dry_keys(n_keys, min_states)
    rep["keys"] = [{"key_id": r.key_id, "W": r.W, "stim": r.stim_name, "n": len(i)}
                   for r, _, i, *_ in keys]
    rep["n_states"] = int(sum(len(k[2]) for k in keys))
    rep["min_states"] = None if n_keys is not None else int(min_states)
    ojs = [obs_jacobians(tr, inp) for _, tr, inp, *_ in keys]
    rep["defect_body"] = float(max(np.max(np.abs(o.defect[:, 3:])) for o in ojs))
    rep["defect_frenet"] = float(max(np.max(np.abs(o.defect[:, :3])) for o in ojs))
    rep["frenet_leak"] = float(max(o.frenet_leak for o in ojs))
    rep["obs_seconds"] = time.perf_counter() - t0
    # observability (states.observability) on this J_z: s is structurally unobservable on the
    # exit straight, observable while the corner is in the preview (Section 3.2, K3)
    ojs32 = [obs_jacobians(tr, _f32(inp)) for _, tr, inp, *_ in keys]
    rep["observability"] = observability_report(keys, ojs, rms, ojs32)
    # extractor against Flax, critic (heads) and actor, float64
    o_all = np.concatenate([np.clip(o.z(rms)[0], -OBS_CLIP, OBS_CLIP) for o in ojs])
    gaps = []
    for s in range(3):
        prm = NW.upcast32(random_critic_params(cfg, s))
        gaps.append({"seed": s, **apply_gap(cfg, prm, o_all, "critic"),
                     "actor": apply_gap(cfg, NW.upcast32(NW.random_actor_params(cfg, s)), o_all,
                                        "actor")["mean"]})
    rep["apply_gap"] = gaps
    prm = NW.upcast32(random_critic_params(cfg, 0))
    flax_head = jax.grad(lambda o_: Critic(cfg).apply(prm, o_, with_heads=True)[1][0])(o_all[0])
    ours = jax.grad(lambda o_: mlp_critic(prm, o_)[1])(o_all[0])
    rep["stop_gradient"] = {"flax": float(jnp.max(jnp.abs(flax_head))),
                            "extractor": float(jnp.max(jnp.abs(ours)))}
    rep["param_hash"] = {str(s): params_hash(random_critic_params(cfg, s)) for s in (0, 1)}
    # J_z and the pullbacks against finite differences
    jac, fd = [], {v: [] for v in VARIANTS}
    for (_, tr, inp, *_), oj in zip(keys, ojs, strict=True):
        jac.append(fd_jacobian(tr, inp, rms, oj))
        for s in fd_seeds:
            net = random_critic(cfg, rms, s, sigma=2.5, heads=True)
            pb = pullback(net, tr, inp, oj)
            for v in VARIANTS:
                fd[v].append(fd_pullback(net, tr, inp, v, pb))
            if s == fd_seeds[0]:
                comp_ = _composed(net, tr, inp)
                rep.setdefault("composed", []).append(
                    {"g_held": _rel(pb.g_held[:, 0], comp_["g_held"]),
                     "g_xprev": _rel(pb.g_xprev[:, 0], comp_["g_xprev"])})
    rep["jz"] = {"max_err": max(j.max_err for j in jac),
                 "share": float(np.mean([j.share_counted for j in jac])),
                 "n": int(sum(j.counted.size for j in jac)),
                 "n_states": int(sum(len(j.err) for j in jac)),
                 "component_p99": np.quantile(np.concatenate(
                     [np.where(j.counted, j.comp_err, 0.0) for j in jac]), 0.99, axis=0).tolist()}
    rep["fd"] = {}
    for v, cks in fd.items():
        ce = np.concatenate([np.where(c.counted, c.comp_err, 0.0) for c in cks])
        rep["fd"][v] = {"n_states": int(sum(c.counted.shape[0] for c in cks) // len(fd_seeds)),
                        "main": max(c.max_err_of(slice(0, 1)) for c in cks),
                        "heads": max(c.max_err_of(slice(1, None)) for c in cks),
                        "share_main": float(np.mean([c.counted[:, 0].mean() for c in cks])),
                        "share_heads": float(np.mean([c.counted[:, 1:].mean() for c in cks])),
                        "newton": max(c.newton for c in cks),
                        "component_p99": np.quantile(ce.reshape(-1, ce.shape[-1]), 0.99,
                                                     axis=0).tolist()}
    # the 20 random critics (V only) on every row, timed
    t1 = time.perf_counter()
    gs, ys, share = [], [], []
    for net in random_critics(cfg, rms, 2.5, seeds):
        pbs = [pullback(net, tr, inp, oj) for (_, tr, inp, *_), oj in zip(keys, ojs,
                                                                          strict=True)]
        gs.append(np.concatenate([p.g[:, 0] for p in pbs]))
        ys.append(np.concatenate([p.y[:, 0] for p in pbs]))
        share.append(np.concatenate([p.accel_share[~p.reset, 0] for p in pbs]))
    G = np.stack(gs)
    rep["random"] = {"n_critics": len(gs), "n_rows": int(G.shape[1]),
                     "finite": bool(np.isfinite(G).all() and np.isfinite(ys).all()),
                     "min_pair_gap": float(min(np.max(np.abs(G[i] - G[j]))
                                               for i in range(len(G)) for j in range(i))),
                     "seconds_per_critic": (time.perf_counter() - t1) / len(gs),
                     "accel_share_median": float(np.median(np.concatenate(share)))}
    rep["seconds"] = time.perf_counter() - t0
    return rep, {"G": G, "o": o_all}


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
    """``python -m apexzero.costate.critic dry-run {plants,random} --out PREFIX``: K3's dry
    runs in a float64 process (x64 switched on here), writing PREFIX.json and PREFIX.npz.
    No trained critic is built or evaluated (``random`` reads only the agent's config and
    obs_rms)."""
    ap = argparse.ArgumentParser(prog="python -m apexzero.costate.critic")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("what", choices=("plants", "random"))
    d.add_argument("--agent", default="G1")
    d.add_argument("--keys", type=int, default=None,
                   help="exactly this many keys (a partial run); default: K3_MIN_STATES rows")
    d.add_argument("--min-states", type=int, default=K3_MIN_STATES)
    d.add_argument("--seeds", type=int, default=N_RANDOM)
    d.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    if args.what == "plants":
        rep, arrays = dry_run_plants()
    else:
        rep, arrays = dry_run_random(args.agent, args.keys, tuple(range(args.seeds)),
                                     min_states=args.min_states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(_jsonable(rep), indent=1))
    np.savez(args.out.with_suffix(".npz"), **arrays)
    return 0


if __name__ == "__main__":
    sys.exit(main())
