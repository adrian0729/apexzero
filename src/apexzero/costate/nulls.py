"""C1's nulls: value-gradient comparators put through the critic's own pullback (design v2.1,
Section 6 C1 'Nulls'; Section 5 'Value-gradient comparators'; Section 12 decision 7 fixes
the QSS and clearance-barrier definitions in code).

Every null is a function of the observation, wrapped as a ``critic.CriticNet`` whose output
is already in reward units (sigma = 1), so ``critic.pullback`` gives its g and pi exactly as
it gives the critic's (the on-manifold total derivative for t >= 1 rows, accel held on reset
rows, clipped channels zeroed; critic.py's module docstring), and ``analysis.c1`` calibrates
them as it calibrates the critic. None reads a trained network; each reads one network's
obs_rms (the input side, design Section 1: computed before the tag).

**Per held-out fold** (review finding on the fold-dependent nulls; C2's standard,
``states.Residuals.by_fold``). A null fitted on targets (the value fits, on V*inf or V*) or
built from them (the far-geometry lambda) is stored as [F, n, 8] (``analysis.FOLD_NULLS``):
slice f is what every row carries while outer fold f is held out, built without fold f's
data, and ``analysis.calibrate`` fits fold f's calibration on slice f's training rows and
predicts fold f's rows from slice f. One predictor per row, built without the row's own fold,
let fold f's targets into the training rows' predictors that fit fold f's calibration, and
calibrated a value fit on a mix of the other folds' fits (on the dry-run world the interior
K_P-mean R2_c so built against per held-out fold: far-geometry lambda 0.585 and 0.591,
G1's value-fit ridge 0.211 and 0.255; a weaker null made C1's paired rule easier to pass).
The QSS and barrier values and the random critics read no target and stay [n, 8].

Each slice is **cross-fitted** as ``states.cross_fit`` cross-fits C2's decoys (v2.3, before
the tag): in slice f, fold f's rows carry the fit without fold f, and a training row of fold
g carries the fit without folds f and g (``fold_gradients``, ``FoldNets.net_for``; the far
geometries of neither f nor the row's own fold for the far-geometry lambda), so no row's
predictor in any slice comes from a fit that saw its own fold. The nested value fit is the
same fit (z-scored on its own rows) at fold f's penalty, as the nested kNN fits are f_D's
own kNN on fewer rows; one fit per pair of folds and penalty serves both slices (F(F - 1)/2
= 15 extra fits when every fold's penalty agrees, as the ridge fits' do on the smoke run;
20 for G1's MLP, whose fold 1 chose another penalty). Before it, slice f pulled every
row back through fold f's fit, so fold f's calibration was fitted on in-sample training-row
gradients and applied to fold f's out-of-sample ones. On the smoke run's world under G1's
obs_rms the interior K_P-mean null R2_c (every K_P component scored) moves from 0.2546 to
0.2172 for the value-fit ridge (n 0.370 to 0.224: its in-sample calibration slopes were
9.6-120 by fold, cross-fitted 2.4-9.5), from 0.3344 to 0.3328 for the window fit and from
0.4070 to 0.3978 for the MLP. On G1's decision rows (s and xi under K2b's exclusions) both
ridge fits' slopes clamp at 0 in both layouts (R2_c -0.0128 either way) and the MLP's s
slope does not (0.016-0.035 by fold, cross-fitted 0.012-0.029): its R2_c there moves from
-0.0278 to -0.0164. The nested fits add about a fifth to the nulls stage (G1: 345 s, was
283 s), nearly all of it the MLP's extra L-BFGS fits.

- **Tail values** (``tail_values``): V*_t = sum_{tau >= t} gamma^(tau - t) r_tau of the key's
  plan, r_tau = (s_{tau+1} - s_tau)/10 - 0.02 ||a_tau - a_{tau-1}||^2 with a_{-1} = 0
  (window.py:470-490; the value the post-jump costate differentiates, targets.py's module
  docstring), so V*_0 = J_W.
- **The value-fit target** (``horizon_values``; settled before the tag, review finding on
  the window length): V*_t has no terminal value, so it depends on W, which z does not hold
  (on the horizon twins V*_t of the W = 400 key is 0.75-0.87 of its W = 800 twin's). The
  primary value fits regress on the horizon-corrected value V*inf_t = V*_t +
  gamma^(W - t) PROGRESS vx_W DT / (1 - gamma): the infinite-horizon value the critic
  regresses on, under the design's own truncation argument (Section 3.1: straight road after
  the corner, x_W near top speed, vx_W 86.9-87.2 m/s on every key); it removes the W
  dependence (a W indicator adds no R2 after it). The as-coded fit on V*_t stays as an extra
  value-fit null (``value_ridge_window``: the stronger comparator on the dry-run world, so
  keeping it keeps C1 conservative). Value fits choose the ridge penalty from
  ``VALUE_GRID`` (``probes.GRID`` floored at 1e-2: the corrected target let one fold pick
  1e-3 and extrapolate), and C1 clips every calibrated prediction to the target's bounds.
- **Value-fit ridge** (``fit_value_ridge``): per outer fold, a ridge of V*inf_t on the clipped z
  of the training folds' T rows (all of T: value regression sees every state, as the critic
  does), z-scored by the weighted training SD with units below ``probes.MIN_SD`` dropped
  (``probes.scaler``), the penalty from ``probes.GRID`` by inner grouped CV on the frozen
  inner folds (``probes._select``, the probes' own selection); re-expressed on o as the
  linear critic w'o + b (``critic.linear_critic``). Its nested fits (the fit without folds
  f and g, at fold f's penalty) give slice f's training rows (``fold_gradients``).
- **Value-fit MLP** (``fit_value_mlp``): the 64-unit tanh MLP probe (``probes.mlp_fit``:
  weighted least squares on the standardised target, L-BFGS, penalty by inner CV over
  ``VALUE_MLP_ALPHAS``) on the same rows, as a JAX function of o (``_mlp_fn``), with nested
  fits as the ridge's (seeds: ``fit_value_mlp``).
- **QSS value** (``qss_critic``; Section 6: 'the discounted progress of the preview's
  quasi-steady speed profile, the law step 1's Qp reads'): from the raw observation (o
  decoded through the network's obs_rms), the preview curvature rebuilt on Qp's 2 m grid over
  500 m (``controls.preview_profile``'s rule, controls.py:619-628: kappa linear in d between
  the 40 preview samples at ``racing.preview_offsets(vx)``, held beyond), the backward QSS
  pass (``grip_profile``: controls.py:489-505's, with a floored sqrt) capped at v_top
  (c_v = 1: the QSS limit itself, not Q's tracking margin), and a forward pass from the
  car's speed at the full-throttle acceleration a_acc(v) limited by that target. Time along
  the profile is t(d) = integral dd / u; V_QSS = 0.1 sum_j ds gamma^(t_mid,j / dt)
  + 0.1 u_end gamma^(t_end/dt) dt / (-ln gamma) (the remaining progress at the last speed,
  discounted). Model tables are step 1's cached ``ModelTables`` (``car``; read only, never
  built here).
- **Clearance-barrier value** (``barrier_critic``): the clearances to the termination
  boundary (edge margin 1.0 m, racing.py:160) at the car and at the 40 preview points with the
  car's normalised lateral position held: c^L_k = w_k (1 - l)/2 - 1, c^R_k = w_k (1 + l)/2 - 1,
  with l the lateral channel (204; racing.py:111) and w_k the road width at preview point k
  (the distance between its left and right edge points, channels 5k..5k+3 x 100 m; the car's
  own from point 0); V_bar = ``BARRIER_SCALE`` sum_k gamma^(d_k / (max(vx, 1) dt))
  [log c~^L_k + log c~^R_k], c~ = (c + sqrt(c^2 + eps^2))/2 a smooth floor at
  ``BARRIER_EPS``. It reproduces lambda_n's sign structure (the price of moving toward the
  nearer wall) and depends on vx through the discounting of the preview.
- **Phase-matched far-geometry lambda** (``far_lambda``; as step 2's Delta rho,
  receptive-fields-design.md:438-440, measures.py:416-422): each T state's phase coordinate
  phi (approach 0-1 from the key's first T state to the optimum's brake onset s_b, braking
  1-2 to s_apex, corner 2-3 to s_c1, exit 3-4 over the 50 m after s_c1; ``phase_coordinate``);
  for every far geometry (``measures.far_geometry``: |ln R ratio| >= 0.4 or |delta theta| >=
  45 deg) the key-weighted mean over its keys of their lambda interpolated at phi
  (mirror-odd components sign-aligned to the row's direction); the prediction is the median
  over those geometries. Slice f takes only far geometries outside fold f and outside the
  row's own fold (``states.cross_fit``'s rule: a held-out row's slice is its own-fold-free
  median, a training row's also leaves fold f out), so no geometry of fold f lends its lambda
  to slice f.
- **Random-init critics**: ``critic.random_critics`` (Section 5's 20 seeds of the network's
  own config, its obs_rms and sigma_R); ``analysis.c1`` takes their per-draw maximum.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import RunningMeanStd
from apexzero.costate import critic as CR
from apexzero.costate import data_dirs
from apexzero.costate import probes as PB
from apexzero.costate import states as ST
from apexzero.costate.targets import MIRROR_ODD, STATE_NAMES
from apexzero.env import racing
from apexzero.fields import measures as FMS
from apexzero.interp import io
from apexzero.interp import measures as M
from apexzero.interp.constants import OBS_PREVIEW
from apexzero.interp.controls import (
    _AY_FLOOR,
    QP_DS,
    QP_N,
    V_FREE,
    CarTables,
    _uinterp,
    car_tables,
)

GAMMA = 0.995                 # every run's gamma (design Section 2; ppo.py PPOConfig)
DT = CR.DT
PROGRESS = 0.1                # reward per m of progress (racing.py:174)
RATE = 0.02                   # action-change cost (racing.py:175)
EDGE_MARGIN = 1.0             # m, the env's termination margin (racing.py:54, 160)
BARRIER_SCALE = 0.1           # reward units per unit log-clearance (the calibration absorbs it)
BARRIER_EPS = 0.05            # m: the smooth floor of a clearance
VALUE_MLP_ALPHAS = PB.MLP_ALPHAS
VALUE_GRID_MIN = 1e-2         # the value fits' smallest ridge penalty (module docstring)
VALUE_GRID = tuple(a for a in PB.GRID if a >= VALUE_GRID_MIN)
EXIT_SPAN = ST.EXIT_MARGIN    # m: the exit phase's length in phi
LAT = OBS_PREVIEW + 4         # the lateral channel (204)
VX = OBS_PREVIEW              # vx / 50 (200)


# ------------------------------------------------------------------------- tail values


def tail_values(x: np.ndarray, a: np.ndarray, gamma: float = GAMMA,
                a_prev: np.ndarray | None = None) -> np.ndarray:
    """[W+1] V*_t of a plan (module docstring); V*_W = 0. ``a_prev`` is a_{-1} (0)."""
    x, a = np.asarray(x, float), np.asarray(a, float)
    W = len(a)
    prev = np.vstack([np.zeros((1, 2)) if a_prev is None else np.asarray(a_prev, float)[None],
                      a[:-1]])
    r = PROGRESS * (x[1:, 0] - x[:-1, 0]) - RATE * np.sum((a - prev) ** 2, axis=1)
    V = np.zeros(W + 1)
    for t in range(W - 1, -1, -1):
        V[t] = r[t] + gamma * V[t + 1]
    return V


def horizon_values(x: np.ndarray, a: np.ndarray, gamma: float = GAMMA,
                   a_prev: np.ndarray | None = None) -> np.ndarray:
    """[W+1] the horizon-corrected value V*inf_t = V*_t + gamma^(W - t) PROGRESS vx_W DT /
    (1 - gamma) (module docstring 'The value-fit target')."""
    x = np.asarray(x, float)
    W = len(a)
    V = tail_values(x, a, gamma, a_prev)
    tail = PROGRESS * float(x[-1, 3]) * DT / (1.0 - gamma)
    return V + gamma ** (W - np.arange(W + 1, dtype=float)) * tail


# ------------------------------------------------------------------------- value fits


@dataclass(frozen=True)
class FoldNets:
    """A value fit's null critics: ``nets[f]`` fitted without outer fold f (or one shared,
    fold-free) and, for the full cross-fit (module docstring 'Per held-out fold'),
    ``nested[f][g]`` fitted without folds f and g at fold f's penalty (None for g = f), the
    critic slice f pulls fold g's rows back through."""

    nets: tuple[CR.CriticNet, ...]
    alpha: np.ndarray            # [F] chosen penalty per fold (nan: none)
    r2_train: np.ndarray         # [F] training R2 of the fit (descriptive)
    nested: tuple[tuple[CR.CriticNet | None, ...], ...] = ()
    nested_r2_train: np.ndarray | None = None   # [F, F] training R2 (nan on the diagonal)

    def net(self, f: int) -> CR.CriticNet:
        return self.nets[f if len(self.nets) > 1 else 0]

    def net_for(self, f: int, g: int) -> CR.CriticNet:
        """The critic slice f pulls fold g's rows back through: fold f's own fit for g = f,
        the fit without folds f and g otherwise (a shared critic for every pair)."""
        if len(self.nets) == 1 or g == f:
            return self.net(f)
        if not self.nested or self.nested[f][g] is None:
            raise ValueError(f"no fit without folds {f} and {g}: the full cross-fit needs "
                             "the nested fits")
        return self.nested[f][g]

    def record(self) -> dict:
        """The summary's 'fits' entry: the penalties, the training R2 of the fits without fold
        f and of the nested fits ([F][F], None on the diagonal)."""
        nr = None if self.nested_r2_train is None else [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in self.nested_r2_train]
        return {"alpha": self.alpha.tolist(), "r2_train": self.r2_train.tolist(),
                "nested_r2_train": nr}


def _o(z: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(z, float), -10.0, 10.0)


def _pairs(n_folds: int):
    """(f, g, a, b) for every ordered pair of distinct folds, (a, b) = (min, max)."""
    for f in range(n_folds):
        for g in range(n_folds):
            if g != f:
                yield f, g, min(f, g), max(f, g)


def _ridge_critic(X: np.ndarray, v: np.ndarray, w: np.ndarray, tr: np.ndarray, alpha: float,
                  obs_rms: RunningMeanStd, name: str,
                  sc: PB.Scaler | None = None) -> tuple[CR.CriticNet, float]:
    """(linear critic on o, training R2) of the ridge of ``v`` [n, 1] on X's ``tr`` rows at
    ``alpha``, z-scored on those rows (``sc``: their scaler, if already fitted)."""
    sc = PB.scaler(X[tr], w[tr]) if sc is None else sc
    Xs = sc.apply(X[tr])
    mu, yb, G, c = PB._sums(Xs, v[tr], w[tr]).moments()
    beta = PB.ridge_path(G, c, Xs.shape[1], (alpha,), (np.inf,))[0, 0, :, 0]
    wo = np.zeros(X.shape[1])
    wo[sc.keep] = beta / sc.sd[sc.keep]
    b = float(yb[0] - mu @ beta - np.sum(sc.mean[sc.keep] * beta / sc.sd[sc.keep]))
    return (CR.linear_critic(wo, b, obs_rms, 1.0, name=name),
            float(PB.r2(v[tr, 0], X[tr] @ wo + b, w[tr])[0]))


def fit_value_ridge(z: np.ndarray, v: np.ndarray, fo: PB.Folds, w: np.ndarray,
                    obs_rms: RunningMeanStd, grid: Sequence[float] = VALUE_GRID) -> FoldNets:
    """The value-fit ridge per outer fold and its nested fits (module docstring), as linear
    critics on o: fold f's penalty by inner CV, every nested fit serving slice f at it."""
    X, v, w = _o(z), np.asarray(v, float)[:, None], np.asarray(w, float)
    F = fo.n_folds
    nets, alpha, r2s = [], np.full(F, np.nan), np.full(F, np.nan)
    for f in range(F):
        tr = fo.outer != f
        sc = PB.scaler(X[tr], w[tr])
        Xs = sc.apply(X)
        p = Xs.shape[1]
        ib, _, _ = PB._select(Xs[tr], v[tr], w[tr], fo.inner[f][tr], p, tuple(grid), (np.inf,))
        alpha[f] = grid[ib[0]]
        net, r2s[f] = _ridge_critic(X, v, w, tr, alpha[f], obs_rms, f"value-ridge-f{f}", sc)
        nets.append(net)
    nested: list[list] = [[None] * F for _ in range(F)]
    r2n = np.full((F, F), np.nan)
    done: dict = {}
    for f, g, a, b in _pairs(F):
        key = (a, b, float(alpha[f]))
        if key not in done:
            tr = (fo.outer != a) & (fo.outer != b)
            done[key] = _ridge_critic(X, v, w, tr, alpha[f], obs_rms, f"value-ridge-f{a}f{b}")
        nested[f][g], r2n[f, g] = done[key]
    return FoldNets(tuple(nets), alpha, r2s, tuple(map(tuple, nested)), r2n)


def _mlp_fn(params, o):
    mean, sd, W1, b1, W2, b2, ym, ysd = params
    h = jnp.tanh(((o - mean) / sd) @ W1 + b1)
    return jnp.reshape(ym + ysd * (h @ W2 + b2), (1,))


def mlp_net(mdl: PB.MLP, obs_rms: RunningMeanStd, name: str = "value-mlp") -> CR.CriticNet:
    """A fitted ``probes.MLP`` on o as a critic (dropped columns get zero weight)."""
    sc = mdl.sc
    p = len(sc.keep)
    W1 = np.zeros((p, mdl.W1.shape[1]))
    W1[sc.keep] = mdl.W1
    sd = np.where(sc.keep, sc.sd, 1.0)
    mean = np.where(sc.keep, sc.mean, 0.0)
    prm = (mean, sd, W1, np.asarray(mdl.b1, float), np.asarray(mdl.W2, float),
           np.float64(mdl.b2), np.float64(mdl.y_mean), np.float64(mdl.y_sd))
    return CR.CriticNet(_mlp_fn, prm, CR._rms(obs_rms), 1.0, 0, name)


def nested_seed(a: int, b: int) -> int:
    """The last id of the value-fit MLP's nested fit without folds a < b (``fit_value_mlp``):
    past the inner folds' 1..N_INNER and the outer fit's 0."""
    if not a < b:
        raise ValueError(f"a nested fit's folds in order: {a} >= {b}")
    return ST.N_INNER + 1 + int(b)


def fit_value_mlp(z: np.ndarray, v: np.ndarray, fo: PB.Folds, w: np.ndarray,
                  obs_rms: RunningMeanStd, alphas: Sequence[float] = VALUE_MLP_ALPHAS,
                  hidden: int = PB.MLP_HIDDEN, maxiter: int = PB.MLP_MAXITER) -> FoldNets:
    """The value-fit MLP per outer fold and its nested fits (module docstring); seeds
    ``probes.mlp_rng`` (partition, fold, 99, inner fold + 1 or 0): component index 99 marks
    the value fit; the nested fit without folds a < b (one per pair and penalty, serving
    slice a's rows of fold b and slice b's of fold a when their penalties agree) draws
    (partition, a, 99, ``nested_seed(a, b)``) and is z-scored on its own training rows."""
    X, v, w = _o(z), np.asarray(v, float), np.asarray(w, float)
    part = max(fo.partition, 0)
    F = fo.n_folds
    nets, alpha, r2s = [], np.full(F, np.nan), np.full(F, np.nan)
    for f in range(F):
        tr = fo.outer != f
        sc = PB.scaler(X[tr], w[tr])
        best = alphas[0]
        if len(alphas) > 1:
            inner = fo.inner[f]
            sse = np.zeros(len(alphas))
            for j in np.unique(inner[tr]):
                a_, b_ = tr & (inner != j), tr & (inner == j)
                for i, al in enumerate(alphas):
                    mdl = PB.mlp_fit(X[a_], v[a_], w[a_], al, PB.mlp_rng(part, f, 99, j + 1),
                                     hidden, maxiter, sc)
                    sse[i] += w[b_] @ (v[b_] - mdl.predict(X[b_])) ** 2
            best = alphas[int(np.argmin(sse))]
        mdl = PB.mlp_fit(X[tr], v[tr], w[tr], best, PB.mlp_rng(part, f, 99, 0), hidden, maxiter,
                         sc)
        nets.append(mlp_net(mdl, obs_rms, f"value-mlp-f{f}"))
        alpha[f] = best
        r2s[f] = PB.r2(v[tr], mdl.predict(X[tr]), w[tr])[0]
    nested: list[list] = [[None] * F for _ in range(F)]
    r2n = np.full((F, F), np.nan)
    done: dict = {}
    for f, g, a, b in _pairs(F):
        key = (a, b, float(alpha[f]))
        if key not in done:
            tr = (fo.outer != a) & (fo.outer != b)
            mdl = PB.mlp_fit(X[tr], v[tr], w[tr], float(alpha[f]),
                             PB.mlp_rng(part, a, 99, nested_seed(a, b)), hidden, maxiter)
            done[key] = (mlp_net(mdl, obs_rms, f"value-mlp-f{a}f{b}"),
                         float(PB.r2(v[tr], mdl.predict(X[tr]), w[tr])[0]))
        nested[f][g], r2n[f, g] = done[key]
    return FoldNets(tuple(nets), alpha, r2s, tuple(map(tuple, nested)), r2n)


# ------------------------------------------------------------------------- QSS and barrier


@functools.lru_cache(maxsize=1)
def car() -> CarTables:
    """The Q family's car tables (``controls.car_tables``) from step 1's cached model tables,
    read from step 1's outputs (``data_dirs``; never built or written here), as float64."""
    name = M.tables_path().name
    path = data_dirs()[0].parent / "tables" / name
    if not path.exists():
        raise FileNotFoundError(f"step 1's model tables {path} are missing")
    return CarTables(*(np.asarray(a, np.float64) for a in car_tables(M.ModelTables.load(path))))


SQRT_FLOOR = 1e-12           # the braking pass's sqrt(1 - lat^2) floor (finite gradients)


def grip_profile(kap_abs, ds, car_: CarTables):
    """``controls.grip_profile`` (controls.py:489-505) with sqrt(1 - lat^2) floored at
    ``SQRT_FLOOR``: the same values, but where the lateral use is clipped at 1 the frozen
    pass's gradient is 0 x inf = nan; here it is 0."""
    vlim = jnp.interp(kap_abs, car_.kap_ss, car_.v_ss, left=V_FREE)

    def body(v_next, inp):
        k_next, vl = inp
        ay = jnp.maximum(_uinterp(v_next, car_.v, car_.a_y), _AY_FLOOR)
        lat = jnp.clip(v_next * v_next * k_next / ay, 0.0, 1.0)
        a = _uinterp(v_next, car_.v, car_.a_dec) * jnp.sqrt(jnp.maximum(1.0 - lat * lat,
                                                                          SQRT_FLOOR))
        v = jnp.minimum(vl, jnp.sqrt(v_next * v_next + 2.0 * a * ds))
        return v, v

    _, vb = jax.lax.scan(body, vlim[-1], (kap_abs[1:], vlim[:-1]), reverse=True)
    return jnp.concatenate([vb, vlim[-1:]])


def _decode(o, mean, var):
    return o * jnp.sqrt(var + CR.RMS_EPS) + mean


def qss_value(ob, car_: CarTables, gamma: float = GAMMA):
    """V_QSS of one raw observation (module docstring); JAX, differentiable."""
    v = jnp.maximum(ob[VX] * 50.0, 0.0)
    d_k = racing.preview_offsets(v)
    kap_k = ob[4:OBS_PREVIEW:5] / 50.0
    d_j = jnp.arange(QP_N, dtype=ob.dtype) * QP_DS
    kap = jnp.abs(jnp.interp(d_j, d_k, kap_k))
    target = jnp.minimum(car_.v_top, grip_profile(kap, QP_DS, car_))

    def fwd(u, tj):
        a = _uinterp(u, car_.v, car_.a_acc)
        un = jnp.minimum(tj, jnp.sqrt(jnp.maximum(u * u + 2.0 * a * QP_DS, 1e-6)))
        return un, un

    _, us = jax.lax.scan(fwd, v, target[1:])
    u = jnp.concatenate([v[None], us])
    u = jnp.maximum(u, 1.0)
    seg = 2.0 * QP_DS / (u[1:] + u[:-1])
    t_end = jnp.cumsum(seg)
    t_mid = t_end - 0.5 * seg
    lg = jnp.log(gamma)
    prog = PROGRESS * QP_DS * jnp.sum(jnp.exp(lg * t_mid / DT))
    tail = PROGRESS * u[-1] * jnp.exp(lg * t_end[-1] / DT) * DT / (-lg)
    return prog + tail


def _qss_fn(params, o):
    mean, var, car_, gamma = params
    return jnp.reshape(qss_value(_decode(o, mean, var), car_, gamma), (1,))


def qss_critic(obs_rms: RunningMeanStd, gamma: float = GAMMA) -> CR.CriticNet:
    rms = CR._rms(obs_rms)
    tab = CarTables(*(jnp.asarray(a, jnp.float64) for a in car()))
    return CR.CriticNet(_qss_fn, (rms.mean, rms.var, tab, np.float64(gamma)), rms, 1.0, 0, "qss")


def barrier_value(ob, gamma: float = GAMMA, scale: float = BARRIER_SCALE,
                  eps: float = BARRIER_EPS, margin: float = EDGE_MARGIN):
    """V_bar of one raw observation (module docstring); JAX, differentiable."""
    vx = ob[VX] * 50.0
    lat = ob[LAT]
    pts = ob[:OBS_PREVIEW].reshape(-1, 5) * 100.0
    w = jnp.sqrt((pts[:, 0] - pts[:, 2]) ** 2 + (pts[:, 1] - pts[:, 3]) ** 2)
    w = jnp.concatenate([w[:1], w])
    d = jnp.concatenate([jnp.zeros(1, ob.dtype), racing.preview_offsets(jnp.maximum(vx, 0.0))])
    cl = 0.5 * w * (1.0 - lat) - margin
    cr = 0.5 * w * (1.0 + lat) - margin

    def soft(c):
        return 0.5 * (c + jnp.sqrt(c * c + eps * eps))

    om = jnp.exp(jnp.log(gamma) * d / (jnp.maximum(vx, 1.0) * DT))
    return scale * jnp.sum(om * (jnp.log(soft(cl)) + jnp.log(soft(cr))))


def _barrier_fn(params, o):
    mean, var, gamma = params
    return jnp.reshape(barrier_value(_decode(o, mean, var), gamma), (1,))


def barrier_critic(obs_rms: RunningMeanStd, gamma: float = GAMMA) -> CR.CriticNet:
    rms = CR._rms(obs_rms)
    return CR.CriticNet(_barrier_fn, (rms.mean, rms.var, np.float64(gamma)), rms, 1.0, 0,
                        "barrier")


# ------------------------------------------------------------------------- gradients by key


@dataclass(frozen=True)
class KeyInput:
    """One key's T rows for the network side: their row indices into T's table, their
    ``critic.Inputs`` and the network-independent ``critic.ObsJac``."""

    key_id: str
    rows: np.ndarray
    inp: CR.Inputs
    oj: CR.ObsJac


@dataclass(frozen=True)
class Grads:
    """A null's (or a critic's) outputs at T's rows: y [n] (reward units), g [n, 8] the
    primary pullback, pi [n, 2], g_held [n, 8] (accel held)."""

    y: np.ndarray
    g: np.ndarray
    pi: np.ndarray
    g_held: np.ndarray


def gradients(net: CR.CriticNet, keys: Sequence[KeyInput], n: int, output: int = 0) -> Grads:
    """Pull back one net on every key's rows."""
    y, g = np.full(n, np.nan), np.full((n, 8), np.nan)
    pi, gh = np.full((n, 2), np.nan), np.full((n, 8), np.nan)
    for k in keys:
        pb = CR.pullback(net, None, k.inp, k.oj)
        y[k.rows] = pb.value[:, output]
        g[k.rows], pi[k.rows], gh[k.rows] = pb.g[:, output], pb.pi[:, output], \
            pb.g_held[:, output]
    return Grads(y, g, pi, gh)


def fold_gradients(fits: FoldNets, keys: Sequence[KeyInput], outer: np.ndarray,
                   n_folds: int = ST.N_FOLDS, output: int = 0) -> np.ndarray:
    """[F, n, 8] a value fit's primary pullback per held-out fold (module docstring 'Per
    held-out fold'): slice f pulls fold f's keys back through fold f's fit and a training
    fold g's keys through the fit without folds f and g (``FoldNets.net_for``), so no row's
    slice-f gradient comes from a fit that saw its own fold. ``outer`` [n] the rows' outer
    folds (a key's rows share one: folds are dealt by geometry)."""
    outer = np.asarray(outer, int)
    out = np.full((n_folds, len(outer), 8), np.nan)
    fold_of = []
    for k in keys:
        fk = np.unique(outer[k.rows])
        if len(fk) != 1:
            raise ValueError(f"key {k.key_id} spans outer folds {fk.tolist()}")
        fold_of.append(int(fk[0]))
    for f in range(n_folds):
        for k, g in zip(keys, fold_of, strict=True):
            out[f, k.rows] = CR.pullback(fits.net_for(f, g), None, k.inp, k.oj).g[:, output]
    return out


def z_rows(keys: Sequence[KeyInput], obs_rms: RunningMeanStd, n: int
           ) -> tuple[np.ndarray, np.ndarray]:
    """(z unclipped [n, 222], clipped mask) of every T row under one obs_rms."""
    z, cl = np.full((n, CR.OBS_DIM), np.nan), np.zeros((n, CR.OBS_DIM), bool)
    for k in keys:
        zz, cc, _ = k.oj.z(obs_rms)
        z[k.rows], cl[k.rows] = zz, cc
    return z, cl


# ------------------------------------------------------------------------- far geometry


def phase_coordinate(tab: io.Table) -> np.ndarray:
    """[n] phi of every T row (module docstring)."""
    s = np.asarray(tab["s"], float)
    key = np.asarray(tab["key_id"])
    s0 = np.empty_like(s)
    for k in np.unique(key):
        m = key == k
        s0[m] = s[m].min()
    s_b = np.asarray(tab["s_b"], float)
    start = np.where(np.isfinite(s_b), s_b, np.asarray(tab["s_c0"], float))
    apex, c1 = np.asarray(tab["s_apex"], float), np.asarray(tab["s_c1"], float)

    def frac(a, lo, hi):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.clip(np.where(hi > lo, (a - lo) / (hi - lo), 0.0), 0.0, 1.0)

    phi = np.where(s < start, frac(s, s0, start),
                   np.where(s < apex, 1.0 + frac(s, start, apex),
                            np.where(s < c1, 2.0 + frac(s, apex, c1),
                                     3.0 + frac(s, c1, c1 + EXIT_SPAN))))
    return phi


def far_lambda(tab: io.Table, lam: np.ndarray, outer: np.ndarray, w: np.ndarray,
               names: Sequence[str] = STATE_NAMES, n_folds: int = ST.N_FOLDS) -> np.ndarray:
    """[F, n, m] the phase-matched far-geometry lambda of every row per held-out fold (module
    docstring): slice f from the far geometries in neither fold f nor the row's own fold."""
    lam = np.asarray(lam, float).reshape(len(w), -1)
    outer = np.asarray(outer, int)
    phi = phase_coordinate(tab)
    key = np.asarray(tab["key_id"])
    t = np.asarray(tab["t"])
    sig = np.asarray(tab["direction"], float)
    geo = ST.geometry_index(tab)
    GK = ST.geometry_keys(tab)
    G = int(geo.max()) + 1
    gR = np.array([GK[geo == g][0] for g in range(G)])
    odd = np.array([nm in MIRROR_ODD for nm in names])
    acc = np.zeros((G, len(w), lam.shape[1]))
    wsum = np.zeros(G)
    for k in np.unique(key):
        m = np.flatnonzero(key == k)
        m = m[np.argsort(t[m])]
        g, sk = int(geo[m[0]]), float(sig[m[0]])
        kw = float(w[m].sum())
        for j in range(lam.shape[1]):
            v = np.interp(phi, phi[m], lam[m, j])
            acc[g, :, j] += kw * (np.where(odd[j], sk * sig, 1.0) * v)
        wsum[g] += kw
    mean = acc / np.maximum(wsum, 1e-300)[:, None, None]             # [G, n, m]
    fold_g = np.array([int(outer[geo == g][0]) for g in range(G)])
    far = np.array([[FMS.far_geometry(gR[a], gR[b]) for b in range(G)] for a in range(G)])
    own = far[geo] & (fold_g[None, :] != outer[:, None])              # [n, G]
    out = np.full((n_folds, len(w), lam.shape[1]), np.nan)
    for f in range(n_folds):
        use = own & (fold_g[None, :] != f)
        vals = np.where(use.T[:, :, None], mean, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f] = np.nanmedian(vals, axis=0)
    return out


# ------------------------------------------------------------------------- all nulls


NULLS = ("value_ridge", "value_mlp", "value_ridge_window", "qss", "barrier", "far_geometry")


def null_gradients(keys: Sequence[KeyInput], tab: io.Table, v_star: np.ndarray, fo: PB.Folds,
                   w: np.ndarray, obs_rms: RunningMeanStd, lam: np.ndarray | None = None,
                   which: Sequence[str] = NULLS,
                   mlp_alphas: Sequence[float] = VALUE_MLP_ALPHAS,
                   mlp_maxiter: int = PB.MLP_MAXITER,
                   v_star_inf: np.ndarray | None = None) -> dict:
    """{null: g} (and 'fits': the value fits' penalties and training R2, the nested fits'
    too, ``FoldNets.record``) for one network's obs_rms on T (``keys`` cover T's rows;
    ``lam`` [n, 8] for the far-geometry null): g is [F, n, 8] per held-out fold for the value
    fits and the far-geometry lambda (cross-fitted), [n, 8] for QSS and the barrier (module
    docstring 'Per held-out fold'). 'value_ridge' and 'value_mlp' fit
    ``v_star_inf`` (the horizon-corrected value; default T's 'v_star_inf' column, else
    ``v_star``), 'value_ridge_window' fits ``v_star`` (module docstring). A float64 process."""
    n, F = len(w), fo.n_folds
    if v_star_inf is None:
        v_star_inf = np.asarray(tab["v_star_inf"], float) if "v_star_inf" in tab else v_star
    out: dict = {"fits": {}}
    z = None
    if {"value_ridge", "value_mlp", "value_ridge_window"} & set(which):
        z, _ = z_rows(keys, obs_rms, n)
    if "value_ridge" in which:
        fr = fit_value_ridge(z, v_star_inf, fo, w, obs_rms)
        out["value_ridge"] = fold_gradients(fr, keys, fo.outer, F)
        out["fits"]["value_ridge"] = fr.record()
    if "value_ridge_window" in which:
        fr = fit_value_ridge(z, v_star, fo, w, obs_rms)
        out["value_ridge_window"] = fold_gradients(fr, keys, fo.outer, F)
        out["fits"]["value_ridge_window"] = fr.record()
    if "value_mlp" in which:
        fm = fit_value_mlp(z, v_star_inf, fo, w, obs_rms, mlp_alphas, maxiter=mlp_maxiter)
        out["value_mlp"] = fold_gradients(fm, keys, fo.outer, F)
        out["fits"]["value_mlp"] = fm.record()
    if "qss" in which:
        out["qss"] = gradients(qss_critic(obs_rms), keys, n).g
    if "barrier" in which:
        out["barrier"] = gradients(barrier_critic(obs_rms), keys, n).g
    if "far_geometry" in which and lam is not None:
        out["far_geometry"] = far_lambda(tab, lam, fo.outer, w, n_folds=F)
    return out
