"""Idea 2's probes: the hidden-unit extractor, ridge and MLP probes, the direction-aware R2,
C2's conditional probe and C2-sw's gain, the control task, permuted-target nulls, the
fold-wise cross-validation on the frozen folds and the cluster bootstrap over geometries
(design v2.1, Sections 4, 5, 6 and 7 K3-K5).

**Extractor** (``hidden``; Section 4 'Features and extraction', K3 'Extractor
self-consistency'). One float64 NumPy pass over a flax parameter tree serves the actor, the
critic trunk and the critic's heads:

- trunk: h_k = act(h_{k-1} W_k + b_k), k = 1..L, from ``MLPEncoder_0/Dense_{k-1}``
  (``MLPEncoder``, ppo.py:147-157: act is tanh when ``cfg.activation`` is 'tanh', else relu);
  h_0 = z;
- actor: the mean m = h_L W + b from the top-level ``Dense_0`` (ppo.py:183-196), before the
  env's clip to [-1, 1] (racing.py:150); log_std is a parameter, not an activation;
- critic: V = h_L W + b from the top-level ``Dense_0`` (ppo.py:199-208), in units of
  reward / sigma_R;
- heads (ppo.py:209-215): head i (i = 0..E-1) is ``_dense(1)(relu(_dense(64)(hs)))`` inside a
  list comprehension. Python evaluates the outer call's callee before its argument, so flax
  names head i's output layer ``Dense_{2i+1}`` and its hidden layer ``Dense_{2i+2}``: u_i =
  relu(h_L Dense_{2i+2}) and head_i = u_i Dense_{2i+1}. Heads are paired by this construction
  order (``head_layers``), never by shape or by sorting names (``Dense_10`` sorts before
  ``Dense_2``). The stop-gradient at ppo.py:211 does not change values; it zeroes the heads'
  input gradients through ``Critic.apply``, so (Section 4) head gradients are taken only
  through this extractor: with ``xp = jax.numpy`` in an x64 process ``hidden`` is traceable.

``Net`` holds float64 exact upcasts of float32 params (``fields.network.upcast32``,
network.py:217-227). Random-init networks: actors from ``fields.network.random_actor_params``
(network.py:306-313) and critics from ``random_critic_params`` (Section 5: ``Critic(cfg).init(
PRNGKey(seed), zeros(OBS_DIM, float32))`` with x64 off), always with the network's own
PPOConfig (``run_config``: the run's config.json filtered to PPOConfig's fields, as
agents.py:130-131 does; G1's run is registry.py:88). ``extractor_check`` is K3's float64
self-consistency against ``Actor.apply`` and ``Critic.apply(..., with_heads=True)``; it needs
x64, so it runs in its own process (``python -m apexzero.costate.probes extractor-check``), as
step 2's float64 work does (network.py:4-8). ``extractor_gap`` is its record for one network:
after the tag it runs on each trained network at D inside the critic stage's x64 process
(``gates.k3_trained``, Section 7 K3 'trained networks after it, on D').

**Pre-registration** (design Section 1; Section 7 'Freeze'): before the tag
``costate-prereg-v1`` no trained network's activation, hidden unit, value or gradient is
computed. ``agent_nets`` marks a snapshot's networks ``trained`` and ``hidden`` refuses them
(PermissionError) until ``unlock_trained`` is called, as step 2's network.py:279-294 does; the
analysis script calls it only after its freeze check. Everything else here works on arrays.

**Features** (Section 4 'Preprocessing and penalties'). Every feature is z-scored on the outer
training fold by the metric's weights; a column whose weighted training SD is below
``MIN_SD`` = 1e-6 is dropped for that fold (``Scaler``), so constant observation channels
(the context) and dead or saturated units never get inflated. z is treated the same way in
every z-only probe, so R2_z does not depend on a run's obs_rms scale (PR-7).

**Ridge with two penalties** (``ridge_cv``). The features are a base block B ([z, m], plus the
decoys where present) and an optional hidden block H (h3, or another layer). For each target
column the fit minimises sum_i w~_i (y_i - c - B_i b - H_i h)^2 + alpha_B |b|^2 + alpha_H |h|^2
with w~ the metric's weights normalised to sum 1 over the fit's own rows and an unpenalised
intercept, so alpha is relative to unit-variance features. alpha_B is from ``GRID`` (1e-4 to 1e4
in half-decade steps, 17 values) and alpha_H from ``H_GRID`` = GRID plus +inf: alpha_H = inf is
the baseline itself, so the full model nests it (Section 4: 'h3's penalty -> inf'). The pair
is chosen per component by inner grouped CV (``Folds.inner``: 5 folds over the training
geometries on stream ``costate_inner``, ``states.inner_folds``, states.py:332-341) as the
smallest pooled weighted SSE; near-ties (relative 1e-12) go to the larger penalties.
``ridge_path`` solves every penalty pair from one eigendecomposition of G_BB and one of the
Schur complement M(alpha_B) = G_HH - G_HB (G_BB + alpha_B)^-1 G_BH per alpha_B:
h = (M + alpha_H)^-1 (c_H - G_HB (G_BB + alpha_B)^-1 c_B), b = (G_BB + alpha_B)^-1 (c_B - G_BH h).
Inner fits use the outer fold's scaling and centre on their own training rows.

**Targets per fold.** ``ridge_cv`` takes targets as [n, m] or as [F, n, m], the targets used
while outer fold f is held out: ``states.Residuals.by_fold`` (states.py:466-499; C2's
cross-fitted lambda_res: training rows carry residuals from models that never saw fold f, the
held-out rows the model fitted on the other folds) or ``permuted_targets``. Held-out rows are
scored against row f of the array (``Probe.y``). No held-out row enters a fit, a scaling or a
penalty choice.

**Folds** (``folds``; Section 4 'Splits'): the outer fold of each row is its geometry's
(``states.geometry_folds``, states.py:324-329, the same deal as ``states.row_folds``,
states.py:344-351), the inner folds ``states.inner_folds``, and the bootstrap cluster is the
geometry. They must be dealt on all of T's 48 geometries; a stratum (interior, pedal-saturated)
takes ``Folds.subset``, never a re-deal on its own rows.

**Metrics** (Section 4 'Metrics'). ``r2``: out-of-fold R2 = 1 - SSE/SST pooled over folds, with
the metric's weights. For a mirror-odd component (n, xi, vy, r, delta, steer;
``targets.MIRROR_ODD``; a column name's last '_' part) SST is about the weighted mean within each
turn direction (R2_wd); for s, vx, pedal R2_wd is the raw R2. ``direction_only`` is the
per-direction training-fold mean, the floor for raw R2 (Section 5).

**Cluster bootstrap** (``cluster_counts``, ``boot_sums``, ``boot_r2``): geometries are
resampled with replacement (``N_BOOT`` = 2,000), the out-of-fold predictions are kept, and
every R2 is recomputed with each row's weight times its geometry's multiplicity, from
per-(geometry, direction) sums. CIs are percentile intervals (``ci``).

**C2** (``c2``, ``gain``; Section 6): Delta2_k = [R2(h3 + z + m) - R2(z + m)] on lambda_res,k
for the network minus the median over the random-init actors of the same gain (each reads the
trained network's z and is given the trained network's m, Section 5), averaged over the
components that pass the power floor; ``decide`` applies the bands (confirm: CI lower end > 0
and point >= ``C2_BAND`` = 0.05; refute: CI upper end < 0.05); with no component passing
(or none passing with a scored row) it raises ``NotTestable``, the one error a caller records
as 'not testable'. ``Gain.partial`` is Delta R2 / (1 - R2_base). **Paired increment**
(``c2_paired``, ``paired``; K4's no-gain plants, gates module docstring): a plant's effect is
the change it causes in the same network, Delta2(host + plant) - Delta2(host), both gains from
one fit of the base and the controls on the same rows, components and bootstrap counts, so the
controls' median cancels: per component R2(host + plant) - R2(host), the CI from the paired
draws. The host's own gain (its chance offset against the other random networks) is kept
beside it. **C2-sw** (``c2_sw``): the
same gain with the base [z] (not conditioned on m) on sigma^ped over the pedal-saturated rows;
``wspearman`` gives the co-reported rho(m_pedal - 1, sigma^ped).

**Two-level CI** (``gain``'s ``pool``; the lead's decision before the tag, on random-init
evidence only: C2's random-only null on its decision rows had mean -0.019 and SD 0.028 over
20 random networks against 5 fixed controls, 7 refuting, none confirming). The primary C2 and
C2-sw score each decision network against a pool of ``N_POOL`` = 20 random-init actors
(``POOL_SEEDS`` 0-19, built as the 5 controls are: on the network's own z, the base given the
network's m). The point is the statistic above with the median over the 20 in place of the
median over 5. Each of the ``N_BOOT`` draws resamples the geometry clusters as before (the
same counts) and, independently, the 20 controls with replacement (``pool_draws`` on stream
``POOL_STREAM``, 'costate_c2_pool'), and its statistic takes the median over the resampled
controls under the draw's geometry counts. The CI is the percentile interval of those draws,
so it carries the controls' own spread (the between-network variance that the
geometry-only CI, controls fixed, leaves out) as well as the geometries'. The bands do not
change. Two descriptives stand beside it in the record: ``ci_geometry``, the same point's
geometry-only CI, and ``pool`` (``pool_rank``), the network's rank among the 20 controls'
own gains over the base.

**The single-geometry robustness rule** (``LOGO_RULE``; ``gain``'s ``logo``, ``logo_gain``,
``logo_counts``, ``logo_rule``; the lead's decision before the tag, on random-init evidence:
on C2's decision rows one geometry held about 5% of the weight but 64-70% of the base's
out-of-fold squared error, and leaving it out moved a random host's statistic by 0.09-0.17).
Every decision computed with a geometry-cluster bootstrap (the primary C1, C2 and C2-sw) is
also computed on each leave-one-geometry-out scoring set: geometry g's rows weighted 0 in
scoring, the fitted probes, out-of-fold predictions and controls unchanged, so set g's point
is the statistic of the same predictions without those rows (from the per-geometry sums,
``boot_r2`` of the counts 1 - e_g). Its CI is the same percentile bootstrap over the other
G - 1 geometries, ``N_BOOT`` draws of ``logo_counts`` on stream ``LOGO_STREAM``
('costate_logo', sub-stream the geometry's index in T), vectorised over the draws; a
two-level CI keeps the pool's control draws (``pool_draws``), the same for every set, so it
stays two-level. A confirm or refute stands only if every set decides the same; otherwise
the record says 'inconclusive' with reason 'geometry-sensitive', names the geometries whose
removal changes it (``ruled``: each one's index in T, R, theta and its set's decision) and
keeps the full set's decision as 'decision_all_geometries'. An inconclusive full set stays
inconclusive. Beside it (``logo_record``): each set's point, move, CI and decision, the most
influential geometry (the largest absolute move) and each passing component's geometries'
shares of the weight and of the base's squared error (``cluster_shares``). The descriptive
records take a 'summary' (the sets' points, the most influential geometry and the shares; no
CI, no rule). ``fwl`` is Section 4's
two-stage (Frisch-Waugh-Lovell) check: within each training fold, the target and every H
column residualised on B by ridge at the full model's alpha_B, then a ridge of the residual
target on the residual H at its alpha_H; with alpha_B -> 0 it reproduces the full model
exactly.

**MLP probe** (``mlp_cv``; secondary, a decodability ceiling only): one hidden layer of
``MLP_HIDDEN`` = 64 tanh units on z-scored features, weighted least squares on the
standardised target with an L2 penalty on the weights, L-BFGS from a seeded init (stream
``MLP_STREAM``), the penalty chosen from ``MLP_ALPHAS`` by the same inner CV.

**Controls and nulls** (Section 5; Section 7 K5).
- ``control_task``: for component k, f_k is a fixed random 2-layer tanh function of
  standardised z (``CONTROL_HIDDEN`` units, weights N(0, gain^2 / fan-in), stream
  ``costate_control`` sub-stream (k, 0)), standardised on the given rows; y = f_k + s eta,
  eta i.i.d. N(0, 1) drawn once (sub-stream (k, 1)), s^2 = (1 - R2*_k) / R2*_k, and, given a
  ceiling (a callable y -> R2, the MLP-on-z ceiling of the real pipeline), s is adjusted by
  log-bisection for at most ``CONTROL_STEPS`` evaluations until the ceiling lands within
  ``CONTROL_TOL`` = 0.02 of R2*_k. ``selectivity`` = R2(target) - R2(control) (descriptive).
- ``permuted_targets``: while outer fold f is held out, the training targets (every row
  outside f) are permuted state by state across all training folds, never within each
  training fold (that keeps each fold's mean, an eighth of each of its geometries' own mean:
  real training signal that transfers to held-out geometries). The mirror-odd columns are
  permuted within each turn direction, since R2_wd already credits the per-direction mean
  (sub-stream (draw, f, 1 + j), j the direction's index in ``np.unique(direction)``); s, vx
  and the other even columns across both directions (sub-stream (draw, f, 0)); stream
  ``costate_perm``. Held-out rows keep their own targets. ``direction`` is required.
- ``wiring_null`` (K5's wiring check, a one-sided test for a positive R2): ``WIRING_DRAWS`` =
  10 permuted-target probes (one ``ridge_cv`` per draw), their out-of-fold predictions
  averaged and scored by R2_wd against the unpermuted held-out targets, with the cluster-
  bootstrap CI. A component passes when its CI's lower end is at or below 0 and its point at
  or below ``WIRING_TOL`` = 0.01 (gates.K4_ZERO_TOL); K5's check passes when every one does.
  Under correct wiring the expected R2 is that of the unweighted leave-fold-out training mean
  (the permutation moves targets between rows of unequal weight), slightly below 0. One draw
  adds a chance term (the probe's small prediction spread times its chance correlation with
  the target) that the geometry bootstrap holds fixed, so a single draw can break both bounds
  with no leak; hence the average. The per-draw points are reported. Not seen: a leak
  confined to the intercept or the fold means (bounded by the between-fold share of the
  target's variance) and a single leaked geometry, near the floor; ``Folds.check`` and
  ``ridge_cv``'s training masks guard both.

**Plants** (Section 7 K4): ``plant_units`` (PR-1a, PR-1b, PR-8: a scaled unit u = t + eta per
target), ``decoy_units`` (PR-6: tanh units of random biased mixes of decoy variables),
``plant_readout`` (PR-4: a trained-architecture network whose h2 holds tanh of a linear
readout of z and whose h3 holds it linearly), ``one_hot`` and ``grouped_folds`` (PR-3's codes
and its deliberately mis-grouped split).

**Streams**: ``costate_inner``, ``costate_perm`` and ``costate_control`` are idea 2's
(``stream_rng``, costate/__init__.py:52-55). The bootstrap, the MLP inits, the control
pool's resampling and the leave-one-geometry-out sets' use ``BOOT_STREAM`` = 308,
``MLP_STREAM`` = 307, ``POOL_STREAM`` = 312 and ``LOGO_STREAM`` = 313 with the same seeding
(ROOT_SEED, STREAM_OFFSET + index, ids; interp/constants.py:51-56); ``STREAMS`` names them
'costate_boot', 'costate_mlp', 'costate_c2_pool' and 'costate_logo'.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from apexzero.agents.ppo import Actor, Critic, PPOConfig
from apexzero.costate import states as ST
from apexzero.costate import stream_rng
from apexzero.costate.targets import MIRROR_ODD
from apexzero.env import racing
from apexzero.fields import network as NW
from apexzero.interp import io
from apexzero.interp.constants import ROOT_SEED, STREAM_OFFSET

KINDS = ("actor", "critic")
HEAD_WIDTH = 64                 # ppo.py:213
N_HEADS_GENERALIST = 5          # design Section 2: every generalist; S1-S3 have none
EXTRACTOR_TOL = 1e-10           # K3: float64 extractor against Actor.apply / Critic.apply
MIN_SD = 1e-6                   # Section 4: features with a training-fold SD below this drop
GRID = tuple(float(v) for v in 10.0 ** np.arange(-4.0, 4.01, 0.5))   # 1e-4 .. 1e4, 17 values
H_GRID = (*GRID, float("inf"))  # the hidden block's penalties; inf nests the baseline
TIE_RTOL = 1e-12                # penalty choice: near-ties go to the larger penalties
N_BOOT = 2000                   # Section 4 'Metrics': cluster bootstrap over geometries
CI_LEVEL = 0.95
C2_BAND = 0.05                  # Section 6 C2 (and C2-sw): the band
N_RANDOM_ACTORS = 5             # Section 5: random-init actors per trained network (seeds 0-4)
N_POOL = 20                     # C2's and C2-sw's primary controls: a pool of random-init actors
POOL_SEEDS = tuple(range(N_POOL))   # ... seeds 0-19 (0-4 are the N_RANDOM_ACTORS controls)
N_RANDOM_CRITICS = 20           # Section 5: random-init critics (seeds 0-19)
MLP_HIDDEN = 64                 # Section 4: one hidden layer of 64 tanh units
MLP_ALPHAS = (1e-4, 1e-3, 1e-2)  # L2 on the MLP's weights (provisional: fixed at the tag)
MLP_MAXITER = 500               # L-BFGS iterations (provisional)
CONTROL_HIDDEN = 64             # K5: the control function's width
CONTROL_GAIN = 1.0              # K5: the stated gain (weights N(0, gain^2 / fan-in))
CONTROL_TOL = 0.02              # K5: the control's ceiling within this of R2*_k
CONTROL_STEPS = 8               # K5: bisection evaluations at most
WIRING_DRAWS = 10               # K5: permutation draws whose out-of-fold predictions average
WIRING_TOL = 0.01               # K5: the largest permuted-target R2_wd point (gates.K4_ZERO_TOL)
BOOT_STREAM = 308               # costate.STREAMS['costate_boot']
MLP_STREAM = 307                # costate.STREAMS['costate_mlp']
POOL_STREAM = 312               # costate.STREAMS['costate_c2_pool']: the pool's resampling
LOGO_STREAM = 313               # costate.STREAMS['costate_logo']: the LOGO sets' resampling
LOGO_RULE = ("the single-geometry robustness rule: the decision is recomputed on each "
             "leave-one-geometry-out scoring set (the geometry's rows weighted 0 in scoring; "
             "fits, calibrations, out-of-fold predictions, nulls and controls unchanged), its CI "
             "the same bootstrap over the other G - 1 geometries (stream costate_logo, "
             "sub-stream the geometry, N_BOOT draws; a two-level CI keeps the pool's control "
             "draws); a confirm or refute stands only if every set's decision is the same, "
             "otherwise it is 'inconclusive' (reason 'geometry-sensitive', the geometries named, "
             "the full-set decision kept as 'decision_all_geometries'); an inconclusive "
             "full-set decision stays inconclusive")


def _rng(index: int, *ids: int) -> np.random.Generator:
    return np.random.default_rng([ROOT_SEED, STREAM_OFFSET + int(index), *map(int, ids)])


def boot_rng(*ids: int) -> np.random.Generator:
    """The bootstrap's generator (``BOOT_STREAM``; ids: e.g. network, partition)."""
    return _rng(BOOT_STREAM, *ids)


def pool_rng(*ids: int) -> np.random.Generator:
    """The control pool's resampling generator in the two-level bootstrap (``POOL_STREAM``),
    independent of the geometry resampling (``boot_rng``)."""
    return _rng(POOL_STREAM, *ids)


def mlp_rng(*ids: int) -> np.random.Generator:
    return _rng(MLP_STREAM, *ids)


def logo_rng(*ids: int) -> np.random.Generator:
    """The leave-one-geometry-out sets' resampling generator (``LOGO_STREAM``; ids: the
    geometry left out)."""
    return _rng(LOGO_STREAM, *ids)


# ------------------------------------------------------------------------- networks


_TRAINED_OK = False


def unlock_trained() -> None:
    """Allow hidden units of trained networks in this process (only after the freeze check
    of the tag ``costate-prereg-v1``; module docstring)."""
    global _TRAINED_OK
    _TRAINED_OK = True


@dataclass(frozen=True, eq=False)
class Net:
    """An actor or a critic for the extractor. ``params`` is the flax tree ({'params': ...})
    with float64 leaves (exact upcasts of float32); ``trained`` marks a snapshot's network."""

    kind: str
    params: Any
    cfg: PPOConfig
    name: str = ""
    trained: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind {self.kind!r} not in {KINDS}")


def _refuse_trained(net: Net, what: str) -> None:
    if net.trained and not _TRAINED_OK:
        raise PermissionError(f"{what} of the trained network {net.name} wait for the tag "
                              "costate-prereg-v1 (design Section 1: no trained network's "
                              "activation, value or gradient before it)")


def run_config(run: str | Path) -> PPOConfig:
    """A run's PPOConfig from its config.json ('ppo' section), filtered to PPOConfig's fields
    (agents.py:130-131). ``run`` is a run directory (relative to the repo or absolute)."""
    path = Path(run)
    path = path if path.is_absolute() else io.REPO / path
    d = json.loads((path / "config.json").read_text())
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    return PPOConfig(**{k: v for k, v in d["ppo"].items() if k in known})


def g1_config() -> PPOConfig:
    """G1's config (registry.py:88), read from its config.json; no checkpoint is opened."""
    from apexzero.interp.registry import AGENTS

    return run_config(AGENTS["G1"].run)


def _check_mlp(cfg: PPOConfig) -> None:
    if cfg.network != "mlp":
        raise ValueError(f"the extractor reads MLPEncoder_0; network {cfg.network!r} is not an "
                         "idea 2 network (pass the run's own PPOConfig, never PPOConfig())")


def random_critic_params(cfg: PPOConfig, seed: int = 0):
    """Float32 params of a randomly initialised critic with ``cfg``'s architecture (design
    Section 5), built with x64 off so that a float32 and a float64 process draw the same
    weights. NumPy leaves; ``ensemble_heads`` heads are created (``is_initializing``)."""
    with jax.enable_x64(False):
        critic = Critic(cfg)
        params = critic.init(jax.random.PRNGKey(seed), jnp.zeros(racing.OBS_DIM, jnp.float32))
        return jax.tree.map(np.asarray, params)


def random_actor(cfg: PPOConfig, seed: int = 0) -> Net:
    """A random-init actor (network.py:306-313), float64 upcast."""
    _check_mlp(cfg)
    return Net("actor", NW.upcast32(NW.random_actor_params(cfg, seed)), cfg,
               f"random-actor-{seed}")


def random_critic(cfg: PPOConfig, seed: int = 0) -> Net:
    _check_mlp(cfg)
    return Net("critic", NW.upcast32(random_critic_params(cfg, seed)), cfg,
               f"random-critic-{seed}")


def agent_nets(agent) -> tuple[Net, Net]:
    """A ``LoadedAgent``'s (agents.py:85-103) actor and critic, marked ``trained``. Loading
    evaluates nothing; ``hidden`` refuses them before the tag."""
    name = f"{agent.agent_id}@{agent.timesteps}"
    snap = agent.snapshot
    return (Net("actor", NW.upcast32(snap.actor), agent.cfg, name, trained=True),
            Net("critic", NW.upcast32(snap.critic), agent.cfg, name, trained=True))


def _tree(params) -> dict:
    return params.get("params", params)


def _dense(layer: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(layer["kernel"], np.float64), np.asarray(layer["bias"], np.float64)


def encoder_layers(params, cfg: PPOConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """The trunk's (W, b) from ``MLPEncoder_0/Dense_0..L-1``, in order."""
    _check_mlp(cfg)
    enc = _tree(params)["MLPEncoder_0"]
    if len(enc) != cfg.mlp_layers:
        raise ValueError(f"MLPEncoder_0 has {len(enc)} layers, config says {cfg.mlp_layers}")
    return [_dense(enc[f"Dense_{k}"]) for k in range(cfg.mlp_layers)]


def n_heads(params) -> int:
    """Number of value heads in a critic tree (top-level Dense layers beyond Dense_0, in
    pairs)."""
    top = [k for k in _tree(params) if k.startswith("Dense_")]
    if (len(top) - 1) % 2:
        raise ValueError(f"odd number of head layers: {sorted(top)}")
    return (len(top) - 1) // 2


def head_layers(params, cfg: PPOConfig) -> list[tuple[tuple[np.ndarray, np.ndarray],
                                                     tuple[np.ndarray, np.ndarray]]]:
    """Head i's ((W, b) hidden, (W, b) output) = (Dense_{2i+2}, Dense_{2i+1}): construction
    order (module docstring)."""
    E = n_heads(params)
    if E != cfg.ensemble_heads:
        raise ValueError(f"critic tree has {E} heads, config says {cfg.ensemble_heads}")
    tree = _tree(params)
    width = cfg.critic_width
    out = []
    for i in range(E):
        hid, top = _dense(tree[f"Dense_{2 * i + 2}"]), _dense(tree[f"Dense_{2 * i + 1}"])
        if hid[0].shape != (width, HEAD_WIDTH) or top[0].shape != (HEAD_WIDTH, 1):
            raise ValueError(f"head {i}: hidden {hid[0].shape}, output {top[0].shape}")
        out.append((hid, top))
    return out


@dataclass(frozen=True)
class Hidden:
    """One network's activations on n inputs (float64): ``layers`` h1..hL [n, width];
    ``out`` the actor's pre-clip mean m [n, 2] or the critic's V [n]; for a critic with heads,
    ``u`` the heads' hidden units [E, n, 64] and ``heads`` their outputs [n, E]."""

    kind: str
    layers: tuple[np.ndarray, ...]
    out: np.ndarray
    u: np.ndarray | None = None
    heads: np.ndarray | None = None

    def __getitem__(self, name: str) -> np.ndarray:
        if name.startswith("h") and name[1:].isdigit():
            return self.layers[int(name[1:]) - 1]
        if name in ("m", "V"):
            if (name == "m") != (self.kind == "actor"):
                raise KeyError(f"{name!r} is not a {self.kind} output")
            return self.out
        if name == "heads" and self.heads is not None:
            return self.heads
        if name.startswith("u") and name[1:].isdigit() and self.u is not None:
            return self.u[int(name[1:])]
        raise KeyError(name)


def hidden(net: Net, z: np.ndarray, xp=np) -> Hidden:
    """The extractor (module docstring) on z [n, OBS_DIM] (normalised, clipped inputs).
    ``xp`` = ``jax.numpy`` (in an x64 process) makes it traceable: input gradients of V, of
    the heads (which ``Critic.apply`` stop-gradients to exactly 0, ppo.py:211) and of hidden
    units are taken through this function."""
    _refuse_trained(net, "hidden units")
    z = xp.asarray(z, xp.float64)
    z = z[None] if z.ndim == 1 else z
    relu = lambda a: xp.maximum(a, 0.0)
    act = xp.tanh if net.cfg.activation == "tanh" else relu
    h, hs = z, []
    for W, b in encoder_layers(net.params, net.cfg):
        h = act(h @ W + b)
        hs.append(h)
    W, b = _dense(_tree(net.params)["Dense_0"])
    out = h @ W + b
    if net.kind == "actor":
        return Hidden("actor", tuple(hs), out)
    u, heads = None, None
    if net.cfg.ensemble_heads:
        pairs = head_layers(net.params, net.cfg)
        u = xp.stack([relu(h @ Wh + bh) for (Wh, bh), _ in pairs])
        heads = xp.stack([u[i] @ Wo[:, 0] + bo[0] for i, (_, (Wo, bo)) in enumerate(pairs)],
                         axis=1)
    return Hidden("critic", tuple(hs), out[:, 0], u, heads)


def normalise(obs: np.ndarray, obs_rms) -> tuple[np.ndarray, np.ndarray]:
    """(z, clipped): clip((o - mean) / sqrt(var + 1e-8), -10, 10) in float64 on the exact
    upcast of obs_rms (``RunningMeanStd.normalize``, ppo.py:282-283), and where |z| >= 10."""
    rms = NW.upcast32(obs_rms)
    zr = (np.asarray(obs, np.float64) - np.asarray(rms.mean)) / np.sqrt(
        np.asarray(rms.var) + 1e-8)
    return np.clip(zr, -10.0, 10.0), np.abs(zr) >= 10.0


def extractor_check(cfg: PPOConfig, seeds: Sequence[int] = (0, 1), n: int = 64,
                    z_seed: int = 0) -> dict:
    """K3's extractor self-consistency on random-init networks (a float64 process): the
    extractor's m, V and heads against ``Actor.apply`` and ``Critic.apply(...,
    with_heads=True)`` in float64, to ``EXTRACTOR_TOL``; head count as the config says. Input
    gradients: dV/dz through the JAX extractor against ``jax.grad`` of ``Critic.apply``; the
    heads' through ``Critic.apply`` are exactly 0 (stop_gradient), through the extractor not."""
    NW.require_x64()
    rng = np.random.default_rng(z_seed)
    z = np.clip(rng.normal(0.0, 2.0, (n, racing.OBS_DIM)), -10.0, 10.0)
    zj = jnp.asarray(z, jnp.float64)
    rows = []
    for s in seeds:
        c = random_critic(cfg, s)
        rec = {"seed": int(s), **extractor_gap(random_actor(cfg, s), c, z)}
        rec.update(_grad_record(c, zj))
        rows.append(rec)
    errs = ("actor_mean", "critic_v", "critic_heads", "grad_v")
    worst = max(max(v for k, v in r.items() if k in errs) for r in rows)
    heads_ok = all(r.get("grad_heads_apply_max", 0.0) == 0.0
                   and r.get("grad_heads_extractor_max", 1.0) > 0.0 for r in rows)
    ok = (worst <= EXTRACTOR_TOL and all(r["dtype"] == "float64" for r in rows)
          and all(r["heads"] == cfg.ensemble_heads for r in rows) and heads_ok)
    return {"rows": rows, "max_error": worst, "tol": EXTRACTOR_TOL, "pass": bool(ok)}


def extractor_gap(actor: Net, critic: Net, z: np.ndarray) -> dict:
    """One network's extractor against Flax in float64 on inputs z [n, OBS_DIM] (normalised,
    clipped; K3's self-consistency, ``extractor_check``'s record without the input
    gradients): max |m - ``Actor.apply``'s mean| ('actor_mean'), max |V - ``Critic.apply``'s
    V| ('critic_v'), with heads max |heads - ``Critic.apply(..., with_heads=True)``'s|
    ('critic_heads'), the tree's head count ('heads') and the dtype. A trained network waits
    for the tag (refused before any evaluation, as ``hidden`` refuses it)."""
    _refuse_trained(actor, "the extractor check")
    _refuse_trained(critic, "the extractor check")
    NW.require_x64()
    ha, hc = hidden(actor, z), hidden(critic, z)
    zj = jnp.asarray(z, jnp.float64)
    cfg = critic.cfg
    mean = np.asarray(Actor(actor.cfg, racing.ACTION_DIM, False).apply(actor.params, zj)[0])
    rec = {"dtype": str(mean.dtype), "actor_mean": float(np.max(np.abs(mean - ha.out))),
           "heads": int(n_heads(critic.params))}
    if cfg.ensemble_heads:
        v, hd = Critic(cfg).apply(critic.params, zj, with_heads=True)
        rec["critic_heads"] = float(np.max(np.abs(np.asarray(hd) - hc.heads)))
    else:
        v = Critic(cfg).apply(critic.params, zj)
    rec["critic_v"] = float(np.max(np.abs(np.asarray(v) - hc.out)))
    return rec


def _grad_record(c: Net, zj) -> dict:
    """dV/dz through the JAX extractor against ``jax.grad`` of ``Critic.apply`` (relative
    max error), and the largest head input gradient by either route. A trained network waits
    for the tag before the ``Critic.apply`` gradient is taken."""
    _refuse_trained(c, "an input gradient")
    cfg = c.cfg
    gv = np.asarray(jax.grad(lambda x: Critic(cfg).apply(c.params, x).sum())(zj))
    ge = np.asarray(jax.grad(lambda x: hidden(c, x, jnp).out.sum())(zj))
    rec = {"grad_v": float(np.max(np.abs(gv - ge)) / np.max(np.abs(gv)))}
    if cfg.ensemble_heads:
        gh = jax.grad(lambda x: Critic(cfg).apply(c.params, x, with_heads=True)[1].sum())(zj)
        gx = jax.grad(lambda x: hidden(c, x, jnp).heads.sum())(zj)
        rec["grad_heads_apply_max"] = float(np.max(np.abs(np.asarray(gh))))
        rec["grad_heads_extractor_max"] = float(np.max(np.abs(np.asarray(gx))))
    return rec


def plant_readout(net: Net, v: np.ndarray, c: float = 0.0, eps: float = 1e-4,
                  unit: int = 0) -> Net:
    """PR-4 (design Section 7 K4): a copy of a random-init network whose h2[unit] holds
    tanh(z v + c) and whose h3[unit] holds eps * tanh(z v + c), each to O(eps^2) relative:
    h1[unit] = tanh(eps (z v + c)), h2[unit] = tanh(h1[unit] / eps), h3[unit] =
    tanh(eps h2[unit]). The other units keep their random weights (they read the planted
    units through their own small weights). Needs a tanh trunk of at least 3 layers."""
    if net.trained:
        raise ValueError("plants are built on random-init networks only")
    if net.cfg.activation != "tanh" or net.cfg.mlp_layers < 3:
        raise ValueError("PR-4 needs a tanh trunk of at least 3 layers")
    params = jax.tree.map(lambda a: np.array(a, np.float64, copy=True), net.params)
    enc = _tree(params)["MLPEncoder_0"]
    v = np.asarray(v, np.float64)
    W0, b0 = enc["Dense_0"]["kernel"], enc["Dense_0"]["bias"]
    W0[:, unit], b0[unit] = eps * v, eps * c
    W1, b1 = enc["Dense_1"]["kernel"], enc["Dense_1"]["bias"]
    W1[:, unit], b1[unit] = 0.0, 0.0
    W1[unit, unit] = 1.0 / eps
    W2, b2 = enc["Dense_2"]["kernel"], enc["Dense_2"]["bias"]
    W2[:, unit], b2[unit] = 0.0, 0.0
    W2[unit, unit] = eps
    return Net(net.kind, params, net.cfg, f"{net.name}+readout", trained=False)


# ------------------------------------------------------------------------- folds


@dataclass(frozen=True)
class Folds:
    """Outer fold [n] of each row, the inner folds [F, n] while fold f is held out (-1 on its
    held-out rows) and the bootstrap cluster [n] (the geometry)."""

    outer: np.ndarray
    inner: np.ndarray
    cluster: np.ndarray
    partition: int = 0

    @property
    def n_folds(self) -> int:
        return int(self.inner.shape[0])

    def subset(self, mask: np.ndarray) -> Folds:
        """The same folds on a subset of rows (a stratum): never re-dealt."""
        m = np.asarray(mask, bool)
        return Folds(self.outer[m], self.inner[:, m], self.cluster[m], self.partition)

    def check(self) -> None:
        """Held-out rows carry no inner fold, training rows one; each cluster sits whole in
        one outer fold."""
        for f in range(self.n_folds):
            held = self.outer == f
            if np.any(self.inner[f][held] != -1) or np.any(self.inner[f][~held] < 0):
                raise ValueError(f"inner folds of outer fold {f} overlap its held-out rows")
        for g in np.unique(self.cluster):
            if len(np.unique(self.outer[self.cluster == g])) != 1:
                raise ValueError(f"cluster {g} is split across outer folds")


def folds(tab: io.Table, partition: int = 0) -> Folds:
    """T's frozen folds (module docstring): outer = ``states.row_folds``' deal, inner =
    ``states.inner_folds`` per held-out fold, cluster = the geometry index."""
    G = ST.geometry_keys(tab)
    uniq, inv = np.unique(G, axis=0, return_inverse=True)
    inv = inv.ravel()
    outer_g = ST.geometry_folds(uniq[:, 0], uniq[:, 1], partition)
    inner = np.stack([ST.inner_folds(uniq[:, 0], uniq[:, 1], outer_g, f, partition)[inv]
                      for f in range(ST.N_FOLDS)])
    out = Folds(outer_g[inv], inner, inv.astype(int), int(partition))
    out.check()
    return out


def grouped_folds(group: np.ndarray, rng: np.random.Generator, n_folds: int = ST.N_FOLDS,
                  n_inner: int = ST.N_INNER, cluster: np.ndarray | None = None) -> Folds:
    """Folds dealt over arbitrary groups (random order, round robin): PR-3's deliberately
    mis-grouped split (by key or cell, mirror partners separated). ``cluster`` defaults to
    the group."""
    _, g = np.unique(np.asarray(group), return_inverse=True)
    g = g.ravel()
    G = int(g.max()) + 1
    outer_g = np.empty(G, int)
    outer_g[rng.permutation(G)] = np.arange(G) % n_folds
    inner = np.full((n_folds, len(g)), -1, int)
    for f in range(n_folds):
        tr = np.flatnonzero(outer_g != f)
        ig = np.full(G, -1, int)
        ig[tr[rng.permutation(len(tr))]] = np.arange(len(tr)) % n_inner
        inner[f] = ig[g]
    cl = g if cluster is None else np.unique(np.asarray(cluster), return_inverse=True)[1].ravel()
    return Folds(outer_g[g], inner, cl.astype(int), -1)


# ------------------------------------------------------------------------- features


@dataclass(frozen=True)
class Scaler:
    """Weighted z-scoring fitted on a fold's training rows; ``keep`` drops columns whose
    weighted SD is below ``MIN_SD``."""

    mean: np.ndarray
    sd: np.ndarray
    keep: np.ndarray

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)
        return (X[:, self.keep] - self.mean[self.keep]) / self.sd[self.keep]


def scaler(X: np.ndarray, w: np.ndarray, min_sd: float = MIN_SD) -> Scaler:
    X, w = np.asarray(X, np.float64), np.asarray(w, np.float64)
    mean = ST.wmean(X, w)
    sd = np.sqrt(np.maximum(ST.wmean((X - mean) ** 2, w), 0.0))
    keep = sd >= min_sd
    return Scaler(mean, np.where(keep, sd, 1.0), keep)


def _as2d(a: np.ndarray | None, n: int) -> np.ndarray:
    if a is None:
        return np.zeros((n, 0))
    a = np.asarray(a, np.float64)
    return a[:, None] if a.ndim == 1 else a


def _by_fold(Y: np.ndarray, F: int) -> np.ndarray:
    """Targets as [F, n, m] (module docstring 'Targets per fold')."""
    Y = np.asarray(Y, np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim == 2:
        return np.broadcast_to(Y, (F, *Y.shape))
    if Y.ndim != 3 or Y.shape[0] != F:
        raise ValueError(f"targets {Y.shape}: expected [n, m] or [{F}, n, m]")
    return Y


def is_odd(name: str | None) -> bool:
    """Mirror-odd target column (its last '_' part in ``MIRROR_ODD``: 'lam_n', 'xi',
    'sigma_steer'; not 'Pi_in' or 'lam_vx')."""
    return name is not None and str(name).split("_")[-1] in MIRROR_ODD


def _names(names, m: int) -> tuple:
    names = tuple(names) if names is not None else (None,) * m
    if len(names) != m:
        raise ValueError(f"{len(names)} names for {m} target columns")
    return names


# ------------------------------------------------------------------------- ridge


@dataclass(frozen=True)
class _Sums:
    """Weighted raw sums of one set of rows."""

    w: float
    x: np.ndarray
    xx: np.ndarray
    y: np.ndarray
    xy: np.ndarray

    def __sub__(self, o: _Sums) -> _Sums:
        return _Sums(self.w - o.w, self.x - o.x, self.xx - o.xx, self.y - o.y, self.xy - o.xy)

    def __add__(self, o: _Sums) -> _Sums:
        return _Sums(self.w + o.w, self.x + o.x, self.xx + o.xx, self.y + o.y, self.xy + o.xy)

    def moments(self):
        """(mu_x, y_bar, G, c): the weighted means, covariance and cross-covariance with the
        weights normalised to sum 1."""
        mu, yb = self.x / self.w, self.y / self.w
        G = self.xx / self.w - np.outer(mu, mu)
        c = self.xy / self.w - np.outer(mu, yb)
        return mu, yb, 0.5 * (G + G.T), c


def _sums(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> _Sums:
    Xw = X * w[:, None]
    return _Sums(float(w.sum()), Xw.sum(0), Xw.T @ X, w @ Y, Xw.T @ Y)


def ridge_path(G: np.ndarray, c: np.ndarray, pb: int, a_b: Sequence[float],
               a_h: Sequence[float]) -> np.ndarray:
    """Coefficients [len(a_b), len(a_h), p, m] of the two-penalty ridge for every pair
    (module docstring): the first ``pb`` columns form the base block. alpha_H = inf gives the
    base-only fit (zero hidden coefficients); with ``pb`` = 0 only alpha_H matters."""
    c = np.asarray(c, np.float64)
    p, m = c.shape
    a_b = np.atleast_1d(np.asarray(a_b, np.float64))
    a_h = np.atleast_1d(np.asarray(a_h, np.float64))
    fin = np.isfinite(a_h)
    beta = np.zeros((len(a_b), len(a_h), p, m))
    ph = p - pb
    if pb == 0:
        if ph:
            d, U = np.linalg.eigh(G)
            Ur = U.T @ c
            beta[:, fin] = np.einsum("pk,jkm->jpm", U, Ur[None] / (d[None, :, None]
                                                                + a_h[fin, None, None]))[None]
        return beta
    e, V = np.linalg.eigh(G[:pb, :pb])
    Vc = V.T @ c[:pb]
    C = G[pb:, :pb] @ V
    for i, ab in enumerate(a_b):
        inv = 1.0 / (e + ab)
        bB0 = V @ (inv[:, None] * Vc)
        beta[i, :, :pb] = bB0
        if ph == 0 or not fin.any():
            continue
        M = G[pb:, pb:] - (C * inv) @ C.T
        r = c[pb:] - C @ (inv[:, None] * Vc)
        d, U = np.linalg.eigh(0.5 * (M + M.T))
        bH = np.einsum("pk,jkm->jpm", U, (U.T @ r)[None] / (d[None, :, None]
                                                           + a_h[fin, None, None]))
        beta[i, fin, pb:] = bH
        beta[i, fin, :pb] = bB0[None] - np.einsum("pk,jkm->jpm", V,
                                                  inv[None, :, None] * (C.T @ bH))
    return beta


def _choose(sse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per component, the (i_b, i_h) of the smallest SSE [nb, nh, m]; near-ties to the larger
    penalties (the larger i_h first, then the larger i_b)."""
    nb, nh, m = sse.shape
    ib, ih = np.empty(m, int), np.empty(m, int)
    for k in range(m):
        s = sse[:, :, k]
        ok = s <= s.min() * (1.0 + TIE_RTOL) + 1e-300
        cand = [(j, i) for i in range(nb) for j in range(nh) if ok[i, j]]
        j, i = max(cand)
        ib[k], ih[k] = i, j
    return ib, ih


def _predict(X: np.ndarray, mu: np.ndarray, yb: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """[n, nb, nh, m] predictions of every coefficient set."""
    nb, nh, p, m = beta.shape
    P = (X - mu) @ beta.transpose(2, 0, 1, 3).reshape(p, -1)
    return P.reshape(len(X), nb, nh, m) + yb


def _select(X: np.ndarray, Y: np.ndarray, w: np.ndarray, inner: np.ndarray, pb: int,
            a_b: Sequence[float], a_h: Sequence[float]) -> tuple[np.ndarray, np.ndarray,
                                                                 np.ndarray]:
    """Inner grouped CV over the penalty pairs on one outer fold's training rows."""
    ids = np.unique(inner[inner >= 0])
    parts = {j: _sums(X[inner == j], Y[inner == j], w[inner == j]) for j in ids}
    total = sum(parts.values(), start=_Sums(0.0, 0.0, 0.0, 0.0, 0.0))
    sse = np.zeros((len(a_b), len(a_h), Y.shape[1]))
    for j in ids:
        val = inner == j
        mu, yb, G, c = (total - parts[j]).moments()
        pred = _predict(X[val], mu, yb, ridge_path(G, c, pb, a_b, a_h))
        sse += np.einsum("n,nabm->abm", w[val], (Y[val][:, None, None, :] - pred) ** 2)
    ib, ih = _choose(sse)
    return ib, ih, sse


@dataclass(frozen=True)
class Probe:
    """Out-of-fold predictions of one probe. ``pred`` and ``y`` [n, m] (each row scored
    against its own held-out fold's target); ``alpha`` [F, m, 2] the chosen (base, hidden)
    penalties (hidden = inf: the baseline; nan: no hidden block); ``kept`` [F, 2] the
    columns kept per block; ``names`` the target columns."""

    pred: np.ndarray
    y: np.ndarray
    alpha: np.ndarray
    kept: np.ndarray
    names: tuple

    def subset(self, mask: np.ndarray) -> Probe:
        m = np.asarray(mask, bool)
        return dataclasses.replace(self, pred=self.pred[m], y=self.y[m])


def ridge_cv(base: np.ndarray, Y: np.ndarray, fo: Folds, w: np.ndarray,
             hidden: np.ndarray | None = None, names=None, grid: Sequence[float] = GRID,
             h_grid: Sequence[float] = H_GRID) -> Probe:
    """The ridge probe over the outer folds (module docstring): z-scoring per training fold,
    per-component penalties by inner grouped CV, out-of-fold predictions. ``base`` [n, pb]
    (may be None with a hidden block), ``hidden`` [n, ph] or None (then the base-only ridge,
    one penalty), ``Y`` [n, m] or [F, n, m], ``w`` the metric's weights."""
    w = np.asarray(w, np.float64)
    n = len(w)
    B, H = _as2d(base, n), _as2d(hidden, n)
    X = np.hstack([B, H])
    F = fo.n_folds
    Yf = _by_fold(Y, F)
    m = Yf.shape[2]
    a_b = tuple(grid) if B.shape[1] else (np.nan,)
    a_h = tuple(h_grid) if H.shape[1] else (np.inf,)
    pred, y = np.full((n, m), np.nan), np.full((n, m), np.nan)
    alpha = np.full((F, m, 2), np.nan)
    kept = np.zeros((F, 2), int)
    for f in range(F):
        test, train = fo.outer == f, fo.outer != f
        if not test.any():
            continue
        sc = scaler(X[train], w[train])
        Xs = sc.apply(X)
        pb = int(sc.keep[:B.shape[1]].sum())
        kept[f] = pb, Xs.shape[1] - pb
        ab_f = a_b if pb else (np.nan,)
        ah_f = a_h if Xs.shape[1] > pb else (np.inf,)
        Yt = Yf[f][train]
        ib, ih, _ = _select(Xs[train], Yt, w[train], fo.inner[f][train], pb, ab_f, ah_f)
        mu, yb, G, c = _sums(Xs[train], Yt, w[train]).moments()
        beta = ridge_path(G, c, pb, ab_f, ah_f)
        P = _predict(Xs[test], mu, yb, beta)
        for k in range(m):
            pred[test, k] = P[:, ib[k], ih[k], k]
            alpha[f, k] = ab_f[ib[k]], (ah_f[ih[k]] if H.shape[1] else np.nan)
        y[test] = Yf[f][test]
    return Probe(pred, y, alpha, kept, _names(names, m))


def fwl(base: np.ndarray, hidden: np.ndarray, Y: np.ndarray, fo: Folds, w: np.ndarray,
        alpha: np.ndarray, names=None) -> Probe:
    """Section 4's two-stage check (module docstring) at the penalties ``alpha`` [F, m, 2] of
    the full model (``Probe.alpha``)."""
    w = np.asarray(w, np.float64)
    n = len(w)
    B, H = _as2d(base, n), _as2d(hidden, n)
    X = np.hstack([B, H])
    F = fo.n_folds
    Yf = _by_fold(Y, F)
    m = Yf.shape[2]
    pred, y = np.full((n, m), np.nan), np.full((n, m), np.nan)
    for f in range(F):
        test, train = fo.outer == f, fo.outer != f
        if not test.any():
            continue
        sc = scaler(X[train], w[train])
        Xs = sc.apply(X)
        pb = int(sc.keep[:B.shape[1]].sum())
        wt = w[train] / w[train].sum()
        mu = wt @ Xs[train]
        Xc = Xs - mu
        Bc, Hc = Xc[:, :pb], Xc[:, pb:]
        Gbb = (Bc[train] * wt[:, None]).T @ Bc[train]
        Gbh = (Bc[train] * wt[:, None]).T @ Hc[train]
        for k in range(m):
            ab, ah = alpha[f, k]
            yk = Yf[f][:, k]
            yb = float(wt @ yk[train])
            yc = yk - yb
            if pb:
                A = Gbb + ab * np.eye(pb)
                gam = np.linalg.solve(A, (Bc[train] * wt[:, None]).T @ yc[train])
                Gam = np.linalg.solve(A, Gbh)
            else:
                gam, Gam = np.zeros(0), np.zeros((0, Hc.shape[1]))
            Ht = Hc - Bc @ Gam
            yt = yc - Bc @ gam
            if np.isfinite(ah) and Ht.shape[1]:
                Gt = (Ht[train] * wt[:, None]).T @ Ht[train]
                bh = np.linalg.solve(Gt + ah * np.eye(Gt.shape[0]),
                                     (Ht[train] * wt[:, None]).T @ yt[train])
            else:
                bh = np.zeros(Ht.shape[1])
            pred[test, k] = yb + Bc[test] @ gam + Ht[test] @ bh
        y[test] = Yf[f][test]
    return Probe(pred, y, np.asarray(alpha, float), np.zeros((F, 2), int), _names(names, m))


def direction_only(Y: np.ndarray, fo: Folds, w: np.ndarray, direction: np.ndarray,
                   names=None) -> Probe:
    """The per-direction training-fold mean (design Section 5), out of fold."""
    w = np.asarray(w, np.float64)
    direction = np.asarray(direction)
    F = fo.n_folds
    Yf = _by_fold(Y, F)
    n, m = Yf.shape[1:]
    pred, y = np.full((n, m), np.nan), np.full((n, m), np.nan)
    for f in range(F):
        test, train = fo.outer == f, fo.outer != f
        for d in np.unique(direction[test]):
            tr, te = train & (direction == d), test & (direction == d)
            pred[te] = ST.wmean(Yf[f][tr], w[tr])
        y[test] = Yf[f][test]
    return Probe(pred, y, np.full((F, m, 2), np.nan), np.zeros((F, 2), int), _names(names, m))


# ------------------------------------------------------------------------- metrics


def r2(y: np.ndarray, pred: np.ndarray, w: np.ndarray, direction: np.ndarray | None = None,
       names=None) -> np.ndarray:
    """[m] weighted R2 = 1 - SSE/SST; SST within direction for mirror-odd names when
    ``direction`` is given (R2_wd; module docstring)."""
    y, pred = _as2d(y, len(w)), _as2d(pred, len(w))
    w = np.asarray(w, np.float64)
    names = _names(names, y.shape[1])
    out = np.empty(y.shape[1])
    for k, nm in enumerate(names):
        grp = direction if (direction is not None and is_odd(nm)) else None
        out[k] = 1.0 - ST.wmean((y[:, k] - pred[:, k]) ** 2, w) / ST.wvar(y[:, k], w, grp)
    return out


def probe_r2(pr: Probe, w: np.ndarray, direction: np.ndarray | None = None,
             rows: np.ndarray | None = None) -> np.ndarray:
    """R2_wd of a probe's out-of-fold predictions (raw R2 without ``direction``), on
    ``rows``."""
    m = np.ones(len(w), bool) if rows is None else np.asarray(rows, bool)
    d = None if direction is None else np.asarray(direction)[m]
    return r2(pr.y[m], pr.pred[m], np.asarray(w)[m], d, pr.names)


def fold_r2(pr: Probe, fo: Folds, w: np.ndarray, direction: np.ndarray | None = None
            ) -> np.ndarray:
    """[F, m] R2 on each held-out fold (the spread over folds)."""
    return np.stack([probe_r2(pr, w, direction, fo.outer == f) for f in range(fo.n_folds)])


def wrank(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mid-ranks: the weight below each value plus half the weight at it (ties
    share one rank)."""
    x, w = np.asarray(x, np.float64), np.asarray(w, np.float64)
    u, inv = np.unique(x, return_inverse=True)
    tot = np.bincount(inv.ravel(), weights=w, minlength=len(u))
    return (np.cumsum(tot) - 0.5 * tot)[inv.ravel()]


def wspearman(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted Spearman rho: the weighted Pearson correlation of weighted mid-ranks."""
    x = np.asarray(x, np.float64)
    w = np.ones(len(x)) if w is None else np.asarray(w, np.float64)
    rx, ry = wrank(x, w), wrank(y, w)
    rx, ry = rx - ST.wmean(rx, w), ry - ST.wmean(ry, w)
    return float(ST.wmean(rx * ry, w) / np.sqrt(ST.wmean(rx * rx, w) * ST.wmean(ry * ry, w)))


# ------------------------------------------------------------------------- bootstrap


def cluster_counts(n_clusters: int, n_boot: int = N_BOOT,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """[B, G] multiplicities of G clusters drawn with replacement, B times."""
    rng = boot_rng() if rng is None else rng
    G = int(n_clusters)
    return rng.multinomial(G, np.full(G, 1.0 / G), size=int(n_boot))


def pool_draws(n_controls: int, n_boot: int = N_BOOT,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """[B, J] indices of J controls drawn with replacement, B times: the control level of
    the two-level bootstrap (``gain``'s ``pool``; default stream ``POOL_STREAM``, drawn the
    same way for every network, as ``cluster_counts`` draws the geometries)."""
    rng = pool_rng() if rng is None else rng
    J = int(n_controls)
    return rng.integers(0, J, size=(int(n_boot), J))


@dataclass(frozen=True)
class BootSums:
    """Per-(cluster, direction) weighted sums of one target set and K prediction sets."""

    w: np.ndarray          # [G, D]
    wy: np.ndarray         # [G, D, m]
    wyy: np.ndarray        # [G, D, m]
    wee: np.ndarray        # [G, D, K, m]
    odd: np.ndarray        # [m] bool: SST within direction


def boot_sums(y: np.ndarray, preds: np.ndarray, w: np.ndarray, cluster: np.ndarray,
              direction: np.ndarray | None = None, names=None,
              mask: np.ndarray | None = None) -> BootSums:
    """``preds`` [K, n, m] (or [n, m]); ``cluster`` [n] in 0..G-1 (``Folds.cluster``);
    ``mask`` [n, m] scores component k on its own rows only (its weight 0 elsewhere: the
    per-component decision masks, e.g. s's observability and K2b's exclusions)."""
    w = np.asarray(w, np.float64)
    n = len(w)
    y = _as2d(y, n)
    P = np.asarray(preds, np.float64)
    P = P[None] if P.ndim == 2 else P
    if mask is not None:
        wm = w[:, None] * np.asarray(mask, bool).reshape(n, y.shape[1])
        y = np.where(wm > 0, y, 0.0)
        P = np.where(wm[None] > 0, P, 0.0)
    names = _names(names, y.shape[1])
    _, cl = np.unique(np.asarray(cluster), return_inverse=True)
    cl = cl.ravel()
    G = int(cl.max()) + 1
    if direction is None:
        d = np.zeros(n, int)
    else:
        _, d = np.unique(np.asarray(direction), return_inverse=True)
        d = d.ravel()
    D = int(d.max()) + 1
    idx = cl * D + d
    e2 = (y[None] - P) ** 2                          # [K, n, m]
    odd = np.array([direction is not None and is_odd(nm) for nm in names])
    if mask is not None:
        return BootSums(_acc(wm, idx, G, D), _acc(wm * y, idx, G, D),
                        _acc(wm * y * y, idx, G, D),
                        _acc(np.moveaxis(wm[None] * e2, 1, 0), idx, G, D), odd)
    return BootSums(_acc(w, idx, G, D), _acc(w[:, None] * y, idx, G, D),
                    _acc(w[:, None] * y * y, idx, G, D),
                    _acc(np.moveaxis(w[None, :, None] * e2, 1, 0), idx, G, D), odd)


def _acc(v: np.ndarray, idx: np.ndarray, G: int, D: int) -> np.ndarray:
    """Sums of v's rows per (cluster, direction) cell ``idx`` = cluster * D + direction."""
    out = np.zeros((G * D, *v.shape[1:]))
    np.add.at(out, idx, v)
    return out.reshape(G, D, *v.shape[1:])


def boot_r2(bs: BootSums, counts: np.ndarray) -> np.ndarray:
    """[B, K, m] R2 of every prediction set under each row of ``counts`` [B, G]."""
    c = np.atleast_2d(np.asarray(counts, np.float64))
    if bs.w.ndim == 3:                               # per-component weights (a mask)
        sw = np.einsum("bg,gdm->bdm", c, bs.w)
    else:
        sw = np.einsum("bg,gd->bd", c, bs.w)[..., None]
    sy = np.einsum("bg,gdm->bdm", c, bs.wy)
    syy = np.einsum("bg,gdm->bdm", c, bs.wyy)
    sse = np.einsum("bg,gdkm->bkm", c, bs.wee)
    with np.errstate(invalid="ignore", divide="ignore"):
        within = np.where(sw > 0, syy - sy ** 2 / np.where(sw > 0, sw, 1.0), 0.0).sum(1)
        pooled = syy.sum(1) - sy.sum(1) ** 2 / sw.sum(1)
    sst = np.where(bs.odd[None], within, pooled)              # [B, m]
    return 1.0 - sse / sst[:, None, :]


def ci(samples: np.ndarray, level: float = CI_LEVEL) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval along axis 0."""
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(np.asarray(samples, np.float64), [a, 1.0 - a], axis=0)
    return lo, hi


# ------------------------------------------------------------------------- leave one geometry out


def logo_counts(n_clusters: int, left_out: int, n_boot: int = N_BOOT,
                sub: int | None = None) -> np.ndarray:
    """[B, G] the cluster bootstrap of the leave-one-geometry-out set without cluster
    ``left_out`` (``LOGO_RULE``): the G - 1 others drawn with replacement, G - 1 per draw, on
    stream ``LOGO_STREAM`` sub-stream ``sub`` (default ``left_out``; the callers pass the
    geometry's index in T, so a set's draws do not depend on the stratum's other
    geometries' positions); column ``left_out`` is 0."""
    G = int(n_clusters)
    if G < 2:
        raise ValueError("a leave-one-geometry-out set needs 2 clusters or more")
    rng = logo_rng(int(left_out) if sub is None else int(sub))
    c = rng.multinomial(G - 1, np.full(G - 1, 1.0 / (G - 1)), size=int(n_boot))
    return np.insert(c, int(left_out), 0, axis=1)


def logo_rule(decision: str, set_decisions: Sequence[str]) -> tuple[str, list[int]]:
    """``LOGO_RULE``: (the recorded decision, the positions of the sets whose decision differs
    from the full set's). A confirm or refute stands only when no set differs; any other
    decision stands as it is, and no set is named."""
    if decision not in ("confirm", "refute"):
        return decision, []
    changed = [i for i, d in enumerate(set_decisions) if d != decision]
    return ("inconclusive" if changed else decision), changed


def cluster_shares(bs: BootSums, k: int = 0) -> dict[str, np.ndarray]:
    """Each cluster's share [G, m] of the metric's weight ('weight'), of prediction set k's
    squared error ('sse') and of the total sum of squares about the full set's mean ('sst',
    within direction for a mirror-odd component: R2's denominator) per component; NaN where
    a component's total is 0."""
    m = bs.wy.shape[2]
    w = bs.w if bs.w.ndim == 3 else np.repeat(bs.w[..., None], m, 2)      # [G, D, m]
    sw, sy = w.sum(0), bs.wy.sum(0)                                         # [D, m]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_d = np.where(sw > 0, sy / np.where(sw > 0, sw, 1.0), 0.0)
        mu_p = sy.sum(0) / sw.sum(0)
    mu = np.where(bs.odd[None], mu_d, mu_p[None])[None]                    # [1, D, m]
    sst = (bs.wyy - 2.0 * mu * bs.wy + mu ** 2 * w).sum(1)
    out = {}
    for name, a in (("weight", w.sum(1)), ("sse", bs.wee[:, :, k, :].sum(1)), ("sst", sst)):
        tot = a.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(tot > 0, a / np.where(tot > 0, tot, 1.0), np.nan)
    return out


def _geometry(gid, labels) -> dict:
    """A geometry's record: its index in T and, with ``labels`` ([G_T, 2], each index's
    (R, theta)), its R and theta."""
    out = {"geometry": int(gid)}
    if labels is not None:
        R, th = np.asarray(labels, float)[int(gid)]
        out.update(R=float(R), theta=float(th))
    return out


def logo_record(ids, labels, point: float, stats: np.ndarray, shares: dict | None = None,
                ci_: np.ndarray | None = None, decisions: Sequence[str] | None = None,
                extra: dict | None = None, n_boot: int | None = None) -> dict:
    """The leave-one-geometry-out record of one statistic: the geometries left out (their
    index in T, R and theta), each set's point and move (point minus the full set's), the
    most influential geometry (the largest absolute finite move) and the clusters' shares
    ({component: {share: [G]}}); with the rule, each set's CI and decision and the stream.
    A 'summary' (no CI) is the descriptive records' (``LOGO_RULE``: exempt from the rule)."""
    ids = [int(v) for v in ids]
    st = np.asarray(stats, float)
    move = st - float(point)
    lab = np.asarray(labels, float)[ids] if labels is not None else None
    out = {"kind": "rule" if decisions is not None else "summary", "geometry": ids,
           "R": None if lab is None else lab[:, 0].tolist(),
           "theta": None if lab is None else lab[:, 1].tolist(),
           "stat": st.tolist(), "move": move.tolist()}
    fin = np.flatnonzero(np.isfinite(move))
    if len(fin):
        i = int(fin[np.argmax(np.abs(move[fin]))])
        out["most_influential"] = {**_geometry(ids[i], labels), "stat": float(st[i]),
                                   "move": float(move[i])}
    if decisions is not None:
        out.update(ci=np.asarray(ci_, float).tolist(), decision=list(decisions),
                   stream="costate_logo", n_boot=int(n_boot))
    if shares is not None:
        out["shares"] = shares
    if extra:
        out.update(extra)
    return out


def logo_gain(bs: BootSums, passing: np.ndarray, point: float, ids, labels=None,
              n_boot: int = N_BOOT, pool: np.ndarray | None = None, band: float = C2_BAND,
              rule: bool = True, names=None) -> dict:
    """A gain's leave-one-geometry-out sets (``LOGO_RULE``) from its sums ``bs`` (prediction
    sets: the base, the network, the controls) over clusters ``ids`` (each one's index in T,
    the sums' order): set g weights cluster g's rows 0 in scoring, so its point is the
    statistic of the same out-of-fold predictions without them, over the components passing
    the power floor that keep a scored row outside g. With ``rule`` its CI is the same
    bootstrap over the other G - 1 clusters (``logo_counts``; ``pool``: the same control
    draws, so a two-level CI stays two-level), vectorised over the draws, and its decision
    the bands'; a set with no component left is 'not testable'. 'shares': each passing
    component's clusters' shares of the weight and of the base's squared error."""
    G = bs.w.shape[0]
    m = bs.wy.shape[2]
    names = [str(i) for i in range(m)] if names is None else list(names)
    passing = np.asarray(passing, bool)
    wk = (bs.w if bs.w.ndim == 3 else np.repeat(bs.w[..., None], m, 2)).sum(1)    # [G, m]
    keep = passing[None] & ((wk.sum(0)[None] - wk) > 0)                           # [G, m]
    g0 = _gain_of(boot_r2(bs, 1.0 - np.eye(G)), passing)                          # [G, m]
    stats = np.array([g0[g, keep[g]].mean() if keep[g].any() else np.nan for g in range(G)])
    sh = cluster_shares(bs, 0)
    shares = {names[k]: {"weight": sh["weight"][:, k].tolist(),
                         "sse_base": sh["sse"][:, k].tolist()} for k in np.flatnonzero(passing)}
    if not rule:
        return logo_record(ids, labels, point, stats, shares)
    cis = np.full((G, 2), np.nan)
    decs = []
    for g in range(G):
        if not keep[g].any():
            decs.append("not testable")
            continue
        rb = boot_r2(bs, logo_counts(G, g, n_boot, ids[g]))                       # [B, K, m]
        gb = _gain_of(rb, passing, pool)[:, keep[g]].mean(1)
        lo, hi = ci(gb)
        cis[g] = lo, hi
        decs.append(decide(float(stats[g]), float(lo), float(hi), band))
    return logo_record(ids, labels, point, stats, shares, cis, decs, n_boot=n_boot)


# ------------------------------------------------------------------------- C2 and C2-sw


def decide(point: float, lo: float, hi: float, band: float = C2_BAND) -> str:
    """C2's (and C2-sw's) bands (design Section 6)."""
    if lo > 0.0 and point >= band:
        return "confirm"
    if hi < band:
        return "refute"
    return "inconclusive"


@dataclass(frozen=True)
class Gain:
    """A conditional probe's gain, the network minus the median random-init control (module
    docstring 'C2'). Per component: ``r2_base``, ``r2_full``, ``r2_random`` [J, m],
    ``gain`` (the statistic's components), ``raw`` (network's own gain), ``partial``
    (raw / (1 - r2_base)); ``stat`` the mean of ``gain`` over ``passing``, ``boot`` its
    bootstrap draws, ``ci`` its interval, ``decision`` the band's verdict. With a control
    pool (module docstring 'Two-level CI'): ``ci_geometry`` the interval of the same
    statistic with the controls fixed (descriptive) and ``pool`` the network's rank among
    the controls' own gains (``pool_rank``; descriptive). With the leave-one-geometry-out
    sets (``logo``: ``logo_gain``'s record): under the rule (``LOGO_RULE``) ``decision`` is
    the ruled one, ``decision_all`` the full set's and ``sensitive`` the geometries whose
    removal changes it; a summary (descriptive) leaves the decision as it is."""

    names: tuple
    passing: np.ndarray
    r2_base: np.ndarray
    r2_full: np.ndarray
    r2_random: np.ndarray
    gain: np.ndarray
    raw: np.ndarray
    partial: np.ndarray
    stat: float
    boot: np.ndarray
    ci: tuple[float, float]
    decision: str
    band: float
    ci_geometry: tuple[float, float] | None = None
    pool: dict | None = None
    logo: dict | None = None
    decision_all: str | None = None
    sensitive: tuple = ()

    def record(self) -> dict:
        out = {"names": list(self.names), "passing": self.passing.tolist(),
               "r2_base": self.r2_base.tolist(), "r2_full": self.r2_full.tolist(),
               "r2_random": self.r2_random.tolist(), "gain": self.gain.tolist(),
               "raw": self.raw.tolist(), "partial": self.partial.tolist(),
               "stat": self.stat, "ci": list(self.ci), "decision": self.decision,
               "band": self.band}
        if self.ci_geometry is not None:
            out.update(ci_kind="two-level", ci_geometry=list(self.ci_geometry),
                       pool=self.pool)
        out.update(logo_fields(self.logo, self.decision_all, self.sensitive))
        return out


def logo_fields(logo: dict | None, decision_all: str | None, sensitive: Sequence[dict]
                ) -> dict:
    """A record's leave-one-geometry-out fields: 'logo' (the sets) and, under the rule,
    'decision_all_geometries' (the full set's decision), 'geometry_sensitive' (the
    geometries whose removal changes it, each with the set's decision) and, when it is not
    empty, 'reason' 'geometry-sensitive'."""
    if logo is None:
        return {}
    out: dict = {"logo": logo}
    if logo.get("kind") == "rule":
        out.update(decision_all_geometries=decision_all, geometry_sensitive=list(sensitive))
        if sensitive:
            out["reason"] = "geometry-sensitive"
    return out


def ruled(decision: str, logo: dict, labels=None) -> tuple[str, tuple]:
    """``logo_rule`` on a rule record (``logo_record``): (the recorded decision, the changing
    geometries, each with R, theta and the set's decision)."""
    dec, changed = logo_rule(decision, logo["decision"])
    return dec, tuple({**_geometry(logo["geometry"][i], labels),
                       "decision": logo["decision"][i]} for i in changed)


def _gain_of(r: np.ndarray, passing: np.ndarray, pick: np.ndarray | None = None
             ) -> np.ndarray:
    """From R2 [..., 2 + J, m] (base, full, randoms): per-component gains [..., m]. ``pick``
    [B, J'] (with r [B, 2 + J, m]): the controls whose median draw b uses (the two-level
    bootstrap's resampled pool); None: every control."""
    own = r[..., 1, :] - r[..., 0, :]
    if r.shape[-2] > 2:
        rand = r[..., 2:, :] - r[..., :1, :]
        if pick is not None:
            rand = np.take_along_axis(rand, np.asarray(pick)[:, :, None], axis=-2)
        own = own - np.median(rand, axis=-2)
    return own


def pool_rank(raw: np.ndarray, r2_random: np.ndarray, r2_base: np.ndarray,
              passing: np.ndarray) -> dict:
    """The network's rank among its controls (descriptive): each one's own gain over the
    base, R2(control + base) - R2(base), and the network's, each the mean over the
    statistic's components ``passing``; 'rank' 1 is the largest of the J + 1 ('of'), with
    'n_above' controls strictly above the network, 'n_below' strictly below and 'n_tied'."""
    p = np.asarray(passing, bool)
    net = float(np.mean(np.asarray(raw, float)[p]))
    ctl = (np.asarray(r2_random, float) - np.asarray(r2_base, float)[None])[:, p].mean(1)
    above, below = int((ctl > net).sum()), int((ctl < net).sum())
    return {"rank": 1 + above, "of": len(ctl) + 1, "n_above": above, "n_below": below,
            "n_tied": len(ctl) - above - below, "network_gain": net,
            "control_gains": ctl.tolist()}


def _passing(passing, names: tuple) -> np.ndarray:
    """[m] bool from None (all), a sequence, or a dict keyed by column name or by its
    component (``states.power_floor``'s keys: 'lam_vx' -> 'vx')."""
    if passing is None:
        return np.ones(len(names), bool)
    if isinstance(passing, dict):
        out = []
        for nm in names:
            key = nm if nm in passing else str(nm).split("_")[-1]
            if key not in passing:
                raise KeyError(f"no power-floor verdict for {nm!r}")
            out.append(bool(passing[key]))
        return np.asarray(out, bool)
    return np.asarray(passing, bool)


class NotTestable(ValueError):
    """The design's 'not testable' conditions for C2 and C2-sw (Section 12 decision 5: no
    component passes the power floor; no component passing it has a scored row): a verdict
    the analysis records. Every other error is a failure and propagates (a ValueError
    subclass, so callers that caught ValueError still catch it)."""


def gain(base: Probe, full: Probe, randoms: Sequence[Probe], w: np.ndarray, fo: Folds,
         direction: np.ndarray | None = None, passing=None, rows: np.ndarray | None = None,
         counts: np.ndarray | None = None, band: float = C2_BAND,
         mask: np.ndarray | None = None, pool: np.ndarray | None = None,
         logo: str | None = None, labels: np.ndarray | None = None) -> Gain:
    """The gain statistic of probes fitted on the same rows and targets (``rows``: the
    stratum scored, e.g. the interior; ``counts`` [B, G] over ``fo.cluster``'s clusters;
    ``mask`` [n, m] each component's own scored rows within them). ``pool`` [B, J] (the
    two-level CI, module docstring; ``pool_draws``): draw b's statistic takes the median
    over the controls ``randoms[pool[b]]`` under draw b's geometry counts, and the point
    stays the median over all J; None: the geometry-only CI with the controls fixed.
    ``logo`` (``logo_gain``; ``fo.cluster`` holds each row's geometry index in T, and
    ``labels`` [G_T, 2] each index's (R, theta)): 'rule' the leave-one-geometry-out sets
    with their CIs and decisions, and the decision ruled by ``LOGO_RULE``; 'summary' their
    points only (descriptive); None neither. ``NotTestable`` when no component passes the
    power floor or none passing it keeps a scored row."""
    m = np.ones(len(w), bool) if rows is None else np.asarray(rows, bool)
    names = base.names
    for p in (full, *randoms):
        if p.names != names or not np.array_equal(p.y, base.y, equal_nan=True):
            raise ValueError("the probes were not fitted on the same targets")
    passing = _passing(passing, names)
    if not passing.any():
        raise NotTestable("no component passes the power floor: the statistic is not testable")
    P = np.stack([base.pred, full.pred, *[p.pred for p in randoms]])[:, m]
    d = None if direction is None else np.asarray(direction)[m]
    mk = None if mask is None else np.asarray(mask, bool).reshape(len(w), -1)[m]
    if mk is not None:
        passing = passing & mk.any(0)
        if not passing.any():
            raise NotTestable("no component passing the power floor has a scored row")
    bs = boot_sums(base.y[m], P, np.asarray(w)[m], fo.cluster[m], d, names, mk)
    G = bs.w.shape[0]
    r0 = boot_r2(bs, np.ones((1, G)))[0]                                   # [2 + J, m]
    counts = cluster_counts(G) if counts is None else counts
    if counts.shape[1] != G:
        raise ValueError(f"counts over {counts.shape[1]} clusters, the rows have {G}")
    rb = boot_r2(bs, counts)                                                # [B, 2 + J, m]
    g0 = _gain_of(r0, passing)
    gb = _gain_of(rb, passing)[:, passing].mean(1)
    stat = float(g0[passing].mean())
    raw = r0[1] - r0[0]
    ci_geo, rank = None, None
    if pool is not None:
        pool = np.asarray(pool)
        J = len(randoms)
        if (pool.ndim != 2 or pool.shape[0] != counts.shape[0] or not J
                or not np.issubdtype(pool.dtype, np.integer)
                or pool.min() < 0 or pool.max() >= J):
            raise ValueError(f"the pool's draws {pool.shape} are not [{counts.shape[0]}, J] "
                             f"indices of the {J} controls")
        lo_g, hi_g = ci(gb)
        ci_geo = (float(lo_g), float(hi_g))
        gb = _gain_of(rb, passing, pool)[:, passing].mean(1)
        rank = pool_rank(raw, r0[2:], r0[0], passing)
    lo, hi = ci(gb)
    dec = decide(stat, float(lo), float(hi), band)
    lg, dec_all, sens = None, None, ()
    if logo is not None:
        if logo not in ("rule", "summary"):
            raise ValueError(f"logo is 'rule', 'summary' or None, not {logo!r}")
        ids = np.unique(np.asarray(fo.cluster)[m])           # boot_sums' cluster order
        if G < 2:                    # removing the one geometry leaves no row to score
            lg = {"kind": logo, "geometry": ids.tolist(), "reason": "one geometry",
                  **({"decision": ["not testable"]} if logo == "rule" else {})}
        else:
            lg = logo_gain(bs, passing, stat, ids, labels, counts.shape[0], pool, band,
                           logo == "rule", names)
        if logo == "rule":
            dec_all = dec
            dec, sens = ruled(dec, lg, labels)
    return Gain(names, passing, r0[0], r0[1], r0[2:], g0, raw, raw / (1.0 - r0[0]), stat, gb,
                (float(lo), float(hi)), dec, band, ci_geo, rank, lg, dec_all, sens)


def conditional(hidden_layers: Sequence[np.ndarray], base: np.ndarray, Y: np.ndarray,
                fo: Folds, w: np.ndarray, names=None, grid: Sequence[float] = GRID,
                h_grid: Sequence[float] = H_GRID) -> tuple[Probe, list[Probe]]:
    """The baseline probe and one full probe per hidden layer (conditional probing, Hewitt et
    al. 2021): [h + base] against [base], same rows, targets and folds."""
    pb = ridge_cv(base, Y, fo, w, None, names, grid, h_grid)
    return pb, [ridge_cv(base, Y, fo, w, h, names, grid, h_grid) for h in hidden_layers]


def c2(h3: np.ndarray, random_h3: Sequence[np.ndarray], z: np.ndarray, m: np.ndarray,
       y_res: np.ndarray, fo: Folds, w: np.ndarray, direction: np.ndarray, names,
       passing=None, rows: np.ndarray | None = None, counts: np.ndarray | None = None,
       grid: Sequence[float] = GRID, h_grid: Sequence[float] = H_GRID,
       mask: np.ndarray | None = None, pool: np.ndarray | None = None,
       logo: str | None = None, labels: np.ndarray | None = None) -> Gain:
    """C2 (design Section 6): ``y_res`` lambda_res (``states.Residuals.by_fold`` [F, n, m] or
    [n, m]), the base [z, m] of the network, the network's h3 and the random-init actors'
    h3 on the same z; the probes are fitted on every row given and component k is scored on
    ``mask[:, k]`` (default all); ``pool``: the two-level CI's control draws, ``logo`` and
    ``labels`` the leave-one-geometry-out sets (``gain``)."""
    base = np.hstack([_as2d(z, len(w)), _as2d(m, len(w))])
    pb, fulls = conditional([h3, *random_h3], base, y_res, fo, w, names, grid, h_grid)
    return gain(pb, fulls[0], fulls[1:], w, fo, direction, passing, rows, counts, mask=mask,
                pool=pool, logo=logo, labels=labels)


@dataclass(frozen=True)
class Paired:
    """A plant's paired increment in its own host (module docstring 'Paired increment'):
    ``gain`` per component R2(host + plant) - R2(host), ``stat`` its mean over the
    statistic's components, ``boot`` the paired bootstrap draws (the plant's minus the
    host's under the same counts), ``ci`` their interval; ``plant`` and ``host`` the two
    unpaired gains (``host``: the host's own offset against the random controls)."""

    plant: Gain
    host: Gain
    gain: np.ndarray
    stat: float
    boot: np.ndarray
    ci: tuple[float, float]

    def record(self) -> dict:
        return {"names": list(self.plant.names), "passing": self.plant.passing.tolist(),
                "gain": self.gain.tolist(), "stat": self.stat, "ci": list(self.ci),
                "unpaired": self.plant.record(), "host": self.host.record()}


def paired(plant: Gain, host: Gain) -> Paired:
    """The paired increment of ``plant`` over ``host``: two gains of the same probes' base and
    random controls on the same rows, components and bootstrap counts (``c2_paired``);
    ValueError otherwise."""
    same = (plant.names == host.names and np.array_equal(plant.passing, host.passing)
            and plant.boot.shape == host.boot.shape
            and np.array_equal(plant.r2_base, host.r2_base, equal_nan=True)
            and np.array_equal(plant.r2_random, host.r2_random, equal_nan=True))
    if not same:
        raise ValueError("the gains do not share a base, controls, rows and components")
    boot = plant.boot - host.boot
    lo, hi = ci(boot)
    return Paired(plant, host, plant.gain - host.gain, float(plant.stat - host.stat), boot,
                  (float(lo), float(hi)))


def c2_paired(h3: np.ndarray, host: np.ndarray, random_h3: Sequence[np.ndarray], z: np.ndarray,
              m: np.ndarray, y_res: np.ndarray, fo: Folds, w: np.ndarray,
              direction: np.ndarray | None, names, passing=None, rows: np.ndarray | None = None,
              counts: np.ndarray | None = None, grid: Sequence[float] = GRID,
              h_grid: Sequence[float] = H_GRID, mask: np.ndarray | None = None) -> Paired:
    """``c2`` of the layer ``h3`` (the host with a plant) paired with ``c2`` of ``host``
    alone: one fit of the base [z, m] and of the random controls, the two gains on the same
    rows, mask and bootstrap counts (``counts`` None: ``cluster_counts`` over the rows'
    geometries, as ``gain`` draws them). ``plant`` and ``host`` equal ``c2``'s own gains of
    the two layers."""
    base = np.hstack([_as2d(z, len(w)), _as2d(m, len(w))])
    pb, fulls = conditional([h3, host, *random_h3], base, y_res, fo, w, names, grid, h_grid)
    if counts is None:
        r = np.ones(len(w), bool) if rows is None else np.asarray(rows, bool)
        counts = cluster_counts(len(np.unique(np.asarray(fo.cluster)[r])))
    g_plant = gain(pb, fulls[0], fulls[2:], w, fo, direction, passing, rows, counts, mask=mask)
    g_host = gain(pb, fulls[1], fulls[2:], w, fo, direction, passing, rows, counts, mask=mask)
    return paired(g_plant, g_host)


def c2_sw(h3: np.ndarray, random_h3: Sequence[np.ndarray], z: np.ndarray, sigma: np.ndarray,
          fo: Folds, w: np.ndarray, rows: np.ndarray | None = None,
          counts: np.ndarray | None = None, grid: Sequence[float] = GRID,
          h_grid: Sequence[float] = H_GRID, pool: np.ndarray | None = None,
          logo: str | None = None, labels: np.ndarray | None = None) -> Gain:
    """C2-sw (design Section 6): Delta_sw = [R2(h3 + z) - R2(z)] on sigma^ped, not conditioned
    on m, on the pedal-saturated rows given (with their ``Folds.subset``); ``pool``: the
    two-level CI's control draws, ``logo`` and ``labels`` the leave-one-geometry-out sets
    (``gain``)."""
    pb, fulls = conditional([h3, *random_h3], z, sigma, fo, w, ("sigma_pedal",), grid, h_grid)
    return gain(pb, fulls[0], fulls[1:], w, fo, None, None, rows, counts, pool=pool,
                logo=logo, labels=labels)


# ------------------------------------------------------------------------- MLP probe


@dataclass(frozen=True)
class MLP:
    """A fitted one-hidden-layer tanh MLP on standardised features and target."""

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: float
    sc: Scaler
    y_mean: float
    y_sd: float
    converged: bool

    def predict(self, X: np.ndarray) -> np.ndarray:
        Hh = np.tanh(self.sc.apply(X) @ self.W1 + self.b1)
        return self.y_mean + self.y_sd * (Hh @ self.W2 + self.b2)


def _mlp_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray, wt: np.ndarray, alpha: float,
              h: int) -> tuple[float, np.ndarray]:
    p = X.shape[1]
    W1 = theta[:p * h].reshape(p, h)
    b1 = theta[p * h:p * h + h]
    W2 = theta[p * h + h:p * h + 2 * h]
    b2 = theta[-1]
    Hh = np.tanh(X @ W1 + b1)
    r = Hh @ W2 + b2 - y
    loss = float(wt @ (r * r) + alpha * (np.sum(W1 * W1) + np.sum(W2 * W2)))
    g = 2.0 * wt * r
    dA = np.outer(g, W2) * (1.0 - Hh * Hh)
    grad = np.concatenate([(X.T @ dA + 2.0 * alpha * W1).ravel(), dA.sum(0),
                           Hh.T @ g + 2.0 * alpha * W2, [g.sum()]])
    return loss, grad


def mlp_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float,
            rng: np.random.Generator, hidden: int = MLP_HIDDEN, maxiter: int = MLP_MAXITER,
            sc: Scaler | None = None) -> MLP:
    """Weighted least squares on the standardised target plus alpha (|W1|^2 + |W2|^2), the
    weights normalised to sum 1; L-BFGS from W1 ~ N(0, 1/p), W2 ~ N(0, 1/h)."""
    w = np.asarray(w, np.float64)
    sc = scaler(X, w) if sc is None else sc
    Xs = sc.apply(X)
    wt = w / w.sum()
    ym = float(wt @ y)
    ysd = float(np.sqrt(wt @ (y - ym) ** 2)) or 1.0
    ys = (np.asarray(y, np.float64) - ym) / ysd
    p = Xs.shape[1]
    theta0 = np.concatenate([rng.normal(0.0, 1.0 / np.sqrt(max(p, 1)), p * hidden),
                             np.zeros(hidden), rng.normal(0.0, 1.0 / np.sqrt(hidden), hidden),
                             [0.0]])
    res = minimize(_mlp_loss, theta0, args=(Xs, ys, wt, float(alpha), hidden), jac=True,
                   method="L-BFGS-B", options={"maxiter": int(maxiter)})
    t = res.x
    return MLP(t[:p * hidden].reshape(p, hidden), t[p * hidden:p * hidden + hidden],
               t[p * hidden + hidden:p * hidden + 2 * hidden], float(t[-1]), sc, ym, ysd,
               bool(res.success))


def mlp_cv(X: np.ndarray, Y: np.ndarray, fo: Folds, w: np.ndarray, names=None,
           alphas: Sequence[float] = MLP_ALPHAS, hidden: int = MLP_HIDDEN,
           maxiter: int = MLP_MAXITER) -> Probe:
    """The MLP probe over the outer folds; alpha per component by inner CV when several are
    given. Seeds: ``mlp_rng(partition, outer fold, component, inner fold + 1 or 0)``."""
    w = np.asarray(w, np.float64)
    X = np.asarray(X, np.float64)
    F = fo.n_folds
    Yf = _by_fold(Y, F)
    n, m = Yf.shape[1:]
    pred, y = np.full((n, m), np.nan), np.full((n, m), np.nan)
    alpha = np.full((F, m, 2), np.nan)
    part = max(fo.partition, 0)
    for f in range(F):
        test, train = fo.outer == f, fo.outer != f
        if not test.any():
            continue
        sc = scaler(X[train], w[train])
        inner = fo.inner[f]
        for k in range(m):
            yk = Yf[f][:, k]
            best = alphas[0]
            if len(alphas) > 1:
                sse = np.zeros(len(alphas))
                for j in np.unique(inner[train]):
                    tr, va = train & (inner != j), train & (inner == j)
                    for a, al in enumerate(alphas):
                        mdl = mlp_fit(X[tr], yk[tr], w[tr], al, mlp_rng(part, f, k, j + 1),
                                      hidden, maxiter, sc)
                        sse[a] += w[va] @ (yk[va] - mdl.predict(X[va])) ** 2
                best = alphas[int(np.argmin(sse))]
            mdl = mlp_fit(X[train], yk[train], w[train], best, mlp_rng(part, f, k, 0), hidden,
                          maxiter, sc)
            pred[test, k] = mdl.predict(X[test])
            alpha[f, k, 0] = best
        y[test] = Yf[f][test]
    return Probe(pred, y, alpha, np.zeros((F, 2), int), _names(names, m))


# ------------------------------------------------------------------------- controls, nulls


def control_function(Z: np.ndarray, w: np.ndarray, k: int, gain: float = CONTROL_GAIN,
                     hidden: int = CONTROL_HIDDEN) -> np.ndarray:
    """f_k(z) (module docstring), standardised (weighted) on the given rows."""
    sc = scaler(Z, w)
    Zs = sc.apply(Z)
    rng = stream_rng("costate_control", int(k), 0)
    p = Zs.shape[1]
    A = rng.normal(0.0, gain / np.sqrt(p), (p, hidden))
    Bm = rng.normal(0.0, gain / np.sqrt(hidden), (hidden, hidden))
    v = rng.normal(0.0, 1.0 / np.sqrt(hidden), hidden)
    f = np.tanh(np.tanh(Zs @ A) @ Bm) @ v
    return (f - ST.wmean(f, w)) / np.sqrt(ST.wvar(f, w))


@dataclass(frozen=True)
class Control:
    """A control task's target ``y`` (standardised), its function ``f``, the noise scale
    ``s``, the achieved ceiling (nan without one) and the bisection's evaluations."""

    y: np.ndarray
    f: np.ndarray
    s: float
    r_star: float
    ceiling: float
    evaluations: int
    ok: bool


def control_task(Z: np.ndarray, w: np.ndarray, k: int, r_star: float,
                 ceiling: Callable[[np.ndarray], float] | None = None,
                 gain: float = CONTROL_GAIN, steps: int = CONTROL_STEPS,
                 tol: float = CONTROL_TOL) -> Control:
    """K5's control task for component k with the real target's ceiling ``r_star``."""
    if not 0.0 < r_star < 1.0:
        raise ValueError(f"R2* = {r_star} outside (0, 1)")
    w = np.asarray(w, np.float64)
    f = control_function(Z, w, k, gain)
    eta = stream_rng("costate_control", int(k), 1).standard_normal(len(f))
    s0 = float(np.sqrt((1.0 - r_star) / r_star))

    def target(s):
        y = f + s * eta
        return (y - ST.wmean(y, w)) / np.sqrt(ST.wvar(y, w))

    if ceiling is None:
        return Control(target(s0), f, s0, r_star, float("nan"), 0, True)
    lo, hi = np.log(0.25), np.log(4.0)
    t, best = 0.0, None
    for i in range(int(steps)):
        s = s0 * np.exp(t)
        got = float(ceiling(target(s)))
        if best is None or abs(got - r_star) < abs(best[1] - r_star):
            best = (s, got)
        if abs(got - r_star) <= tol:
            return Control(target(s), f, s, r_star, got, i + 1, True)
        lo, hi = (t, hi) if got > r_star else (lo, t)    # too predictable: more noise
        t = 0.5 * (lo + hi)
    s, got = best
    return Control(target(s), f, s, r_star, got, int(steps), abs(got - r_star) <= tol)


def selectivity(r2_target: np.ndarray, r2_control: np.ndarray) -> np.ndarray:
    """Hewitt and Liang (2019): probe R2 on the target minus on its control task."""
    return np.asarray(r2_target, np.float64) - np.asarray(r2_control, np.float64)


def permuted_targets(Y: np.ndarray, fo: Folds, direction: np.ndarray, names=None,
                     draw: int = 0) -> np.ndarray:
    """[F, n, m]: while outer fold f is held out, the training targets (every row outside f)
    permuted state by state across all training folds, the mirror-odd columns of ``names``
    within each turn direction, the others across both (module docstring); the held-out rows
    keep their own. ``draw`` is the permutation draw (sub-streams (draw, f, j))."""
    if direction is None:
        raise ValueError("permuted_targets needs the turn direction: the mirror-odd columns "
                         "are permuted within it")
    F = fo.n_folds
    Yf = np.array(_by_fold(Y, F), copy=True)
    direction = np.asarray(direction)
    if direction.shape != (Yf.shape[1],):
        raise ValueError(f"direction {direction.shape} for {Yf.shape[1]} rows")
    odd = np.array([is_odd(nm) for nm in _names(names, Yf.shape[2])], bool)
    dirs = np.unique(direction)
    for f in range(F):
        train = fo.outer != f
        tr = np.flatnonzero(train)
        p = tr[stream_rng("costate_perm", draw, f, 0).permutation(len(tr))]
        Yf[f][np.ix_(tr, ~odd)] = Yf[f][np.ix_(p, ~odd)]
        for j, d in enumerate(dirs):
            idx = np.flatnonzero(train & (direction == d))
            p = idx[stream_rng("costate_perm", draw, f, 1 + j).permutation(len(idx))]
            Yf[f][np.ix_(idx, odd)] = Yf[f][np.ix_(p, odd)]
    return Yf


def wiring_null(base: np.ndarray, Y: np.ndarray, fo: Folds, w: np.ndarray,
                hidden: np.ndarray | None = None, names=None, direction=None,
                counts: np.ndarray | None = None, n_draws: int = WIRING_DRAWS,
                tol: float = WIRING_TOL, grid: Sequence[float] = GRID,
                h_grid: Sequence[float] = H_GRID) -> dict:
    """K5's permuted-target wiring check (module docstring): one ``ridge_cv`` on
    ``permuted_targets`` per draw 0 .. n_draws - 1, the out-of-fold predictions averaged
    over the draws and scored by R2_wd against the unpermuted held-out targets, with the
    cluster-bootstrap CI (``counts``). A component passes when the CI's lower end is at or
    below 0 and the point at or below ``tol``; 'per_draw' [n_draws][m] holds each draw's own
    point. ``direction`` is required."""
    if int(n_draws) < 1:
        raise ValueError(f"n_draws = {n_draws}: at least one permutation draw")
    w = np.asarray(w, np.float64)
    F = fo.n_folds
    Y0 = _by_fold(Y, F)
    y = np.full(Y0.shape[1:], np.nan)
    for f in range(F):
        y[fo.outer == f] = Y0[f][fo.outer == f]          # the unpermuted held-out targets
    preds = []
    for d in range(int(n_draws)):
        pr = ridge_cv(base, permuted_targets(Y, fo, direction, names, d), fo, w, hidden, names,
                      grid, h_grid)
        preds.append(pr.pred)
    bs = boot_sums(y, np.stack([*preds, np.mean(preds, 0)]), w, fo.cluster, direction,
                   pr.names)
    counts = cluster_counts(bs.w.shape[0]) if counts is None else counts
    points = boot_r2(bs, np.ones((1, bs.w.shape[0])))[0]      # [n_draws + 1, m]
    point = points[-1]
    lo, hi = ci(boot_r2(bs, counts)[:, -1])
    ok = (lo <= 0.0) & (point <= tol)
    return {"r2": point.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
            "per_draw": points[:-1].tolist(), "n_draws": int(n_draws), "tol": float(tol),
            "pass_by_component": ok.tolist(), "pass": bool(np.all(ok))}


# ------------------------------------------------------------------------- plants (K4)


def expected_r2(snr: float) -> float:
    """SNR / (1 + SNR): the R2 of a target from one unit t + eta with Var eta = Var t / SNR."""
    return float(snr) / (1.0 + float(snr))


def plant_units(H: np.ndarray, targets: np.ndarray, snr, w: np.ndarray,
                rng: np.random.Generator, scale: float | None = None) -> np.ndarray:
    """[n, ph + m]: ``H`` with one unit per target column appended, u = scale (t~ + eta),
    t~ the weighted-standardised target and eta ~ N(0, 1/SNR) i.i.d. per row (``snr``
    scalar or per column). ``scale`` defaults to the median weighted SD of H's units (PR-1b's
    alpha)."""
    w = np.asarray(w, np.float64)
    H = _as2d(H, len(w))
    T = _as2d(targets, len(w))
    snr = np.broadcast_to(np.asarray(snr, np.float64), (T.shape[1],))
    t = (T - ST.wmean(T, w)) / np.sqrt(ST.wvar(T, w))
    u = t + rng.standard_normal(T.shape) / np.sqrt(snr)
    if scale is None:
        scale = float(np.median(np.sqrt(ST.wvar(H, w)))) if H.shape[1] else 1.0
    return np.hstack([H, scale * u])


def decoy_units(D: np.ndarray, w: np.ndarray, rng: np.random.Generator, n_units: int = 32,
                gain: float = 1.0, bias: float = 1.0) -> np.ndarray:
    """PR-6: tanh(gain (Ds A + b)) with Ds the weighted-standardised decoy variables (e.g.
    s - s_apex, vx, sigma, log R), A ~ N(0, 1/q) and b ~ N(0, bias^2) per unit (drawn after
    A, so A does not depend on ``bias``; bias 0 gives tanh(gain Ds A)). Without b every unit
    is an odd function of the centred decoys, and a ridge on 32 of them peaks at R2 0.36 on
    lambda (K4's review). The probe z-scores each unit, so ``gain`` acts only through
    saturation; the caller picks it (``gates.pr6``)."""
    w = np.asarray(w, np.float64)
    Ds = scaler(D, w).apply(D)
    A = rng.normal(0.0, 1.0 / np.sqrt(Ds.shape[1]), (Ds.shape[1], int(n_units)))
    b = rng.normal(0.0, float(bias), int(n_units)) if bias else 0.0
    return np.tanh(gain * (Ds @ A + b))


def one_hot(labels: np.ndarray) -> np.ndarray:
    """[n, L] indicator columns of the labels (PR-3's geometry and pair codes)."""
    _, inv = np.unique(np.asarray(labels), return_inverse=True)
    inv = inv.ravel()
    out = np.zeros((len(inv), int(inv.max()) + 1))
    out[np.arange(len(inv)), inv] = 1.0
    return out


# ------------------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> int:
    """``extractor-check``: K3's float64 extractor self-consistency on random-init networks
    (this process switches x64 on first)."""
    ap = argparse.ArgumentParser(prog="python -m apexzero.costate.probes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extractor-check")
    e.add_argument("--out", required=True)
    e.add_argument("--run", default=None, help="run directory whose config.json to use "
                                                "(default: G1's)")
    e.add_argument("--heads", type=int, default=None, help="override ensemble_heads (0: the "
                                                           "specialists' head-free critic)")
    e.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    t0 = time.time()
    cfg = run_config(args.run) if args.run else g1_config()
    if args.heads is not None:
        cfg = cfg.with_(ensemble_heads=int(args.heads))
    rec = extractor_check(cfg, args.seeds)
    rec.update(seconds=time.time() - t0, config={"network": cfg.network,
                                                 "actor_width": cfg.actor_width,
                                                 "critic_width": cfg.critic_width,
                                                 "mlp_layers": cfg.mlp_layers,
                                                 "ensemble_heads": cfg.ensemble_heads})
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("max_error", "pass", "seconds")}))
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
