"""The optimum's side of idea 2: the post-jump discrete costate of a stored key, its pedal fold,
the previous-action component, the switching function and the shadow prices of the walls,
the slip limit, grip and the road (design v2.1, Sections 3.1 and 3.4; K0 and the shifts of
K1 in Section 7).

**The NLP** is a key's TO-OCP as step 2 rebuilds and re-polishes it (``keys.rebuild``,
keys.py:423-454; ``keys.repolish``, keys.py:545-571): w = [vec(X) (W+1 states of 8);
vec(A) (W actions of 2)], g = [defects x_{t+1} - Phi_t (rows 8t+i); corridor rows at x_{t+1}
(8W+2t+k: n - (w_l - 1.03), -(w_r - 1.03) - n); slip rows alpha_f/alpha_max,
alpha_r/alpha_max at x_{t+1} (10W+2t+k)] (window.py:498-501, 319-342). f = -J with
J = sum_t gamma^t [(s_{t+1} - s_t)/10 - 0.02 ||a_t - a_{t-1}||^2], a_{-1} = 0
(window.py:470-490). CasADi's convention L = f + lam_g'g + lam_x'w (keys.py:464-469;
kkt.py:4-9): a bound shift c has dJ*/dc = +lam. x_0 is fixed (and a_0 on anchored keys,
keys.py:414-416); IPOPT's default ``fixed_variable_treatment = make_parameter`` fills a fixed
variable's multiplier from the Lagrangian gradient, which ``assert_parameter_duals`` checks
the solver options allow (design Section 7, K0). Keys are unscaled (kkt.py:679-680).

**The pedal fold** (``pedal_fold``, design Section 3.1). The pedal row of Phi is linear:
each of the ``substeps`` RK2 substeps (window.py:210-235) of dpedal = (a - pedal)/tau
(dynamics.py:132) maps pedal to (1 - h/tau + (h/tau)^2/2) pedal + ..., so
pedal_{t+1} = alpha pedal_t + (1 - alpha) a_t with alpha = 0.905^10 (``pedal_alpha``;
checked against jac_g's pedal rows in ``fold_check``). The pedal-state box is therefore
redundant (actions are clipped to [-1, 1]) and its multipliers are carried into the defect
and command multipliers: c_{W-1} = lam_x[8W+7], c_{t-1} = lam_x[8t+7] + alpha c_t;
lam_g'[8t+7] = lam_g[8t+7] + c_t; the pedal command of step t gains (1 - alpha) c_t; the
pedal-state multipliers of x_1..x_W become 0. x_0's pedal multiplier (a fixed variable's)
gains alpha c_0, so that stationarity holds on every variable with the folded multipliers
(``stationarity``) and lambda_0 = lam_x'[0:8] (the design's 'the fold adds alpha c_0').

**The post-jump costate** (``costate``; design Section 3.1). For 0 <= t < W

    lambda_t = gamma^-t Phi_x(x_t, a_t)' lam_g'[8t:8t+8] - e_s/10          ('jac' route)
             = lambda-_t + gamma^-t (J_path(x_t)' nu_{t-1} + lam_x'[8t:8t+8])   ('cheap')
    lambda-_t = gamma^-t (lam_g'[8(t-1):8t] - gamma^(t-1) e_s/10)           (pre-jump)

with lambda_0 = lam_x'[0:8] on the cheap route. The jac route reads every Phi_x block from one
``jac_g`` evaluation (kkt.py:327-331, 433-446); the cheap route needs only the path
Jacobian (window.py:319-342) at x_t, from step t-1's window, and the box multipliers, and the
two agree by stationarity on x_t (K0: to 1e-10). At t = W the value is 0, so lambda_W = 0, and
the cheap route's lambda-_W + jump_W must vanish too (``Costate.end_gap``).
pi_t = dV_t/da_{t-1} = 0.04 (a*_t - a*_{t-1}), a*_{-1} = 0, pi_W = 0. The switching function
is sigma_t = gamma^-t lam_x'(a_t) (design Section 6, C2); ``action_identity`` checks
sigma_t = Phi_a' (e_s/10 + gamma lambda-_{t+1}) + gamma pi_{t+1} - 0.04 (a_t - a_{t-1}).
On anchored keys lam_g[0:8] is not unique when a constraint on x_1 is active (x_1 is pinned by
the fixed x_0 and a_0), so lambda_0 and lambda-_1 are not either (``Costate.unique_from``);
lambda_1 is (it uses lam_g[8:16]).

**Activity and margins** (``Costate``): a row or bound is active when its multiplier exceeds
``ACTIVE_TOL`` in magnitude. x_t's own rows are the corridor and slip rows of step t - 1 and
x_t's box (folded, so the pedal box never counts; ``pedal_box`` keeps the raw flag of design
Section 3.2's 'pedal-saturated'). The clearance to the env's termination boundary (the edge
margin of 1.0 m, racing.py:160, not the OCP's 3 cm tighter bound) is -g_corridor + safety per
side, and the slip margin alpha_max - max(|alpha_f|, |alpha_r|) (racing.py:162), from the path
function at every x_t (x_0 with step 0's window).

**Shadow prices** (``prices``; design Section 3.4). nu^L, nu^R >= 0 are the corridor
multipliers on x_{t+1}; inside is left for a left turn (sigma = +1). For t <= W:
Pi^X_t = gamma^-t sum_{tau >= t} nu^X_tau (dV_t/dDelta for a uniform widening of side X
ahead), Pi^slip_t = gamma^-t sum_{tau >= t} (|nu^f| + |nu^r|) (dV_t/d ln alpha_max, the 1e-4
rad margin held), and the context prices Pi^ctx_t = gamma^-t sum_{tau >= t}
[lam_g'[defect tau]' dPhi_tau/dctx - nu_tau' dg_path(x_{tau+1})/dctx] (ctx as P holds it,
keys.py:375-384; column 4 is mu_scale, the grip price Pi^mu; the path part is zero for mu,
since slip angles do not depend on it). d^X_t is the nu^X-weighted mean of s_{tau+1} - s_t.
``road_field`` gives dV_t/d(samples) of the strip's kappa, w_left and w_right through step
2's F_p blocks (``kkt.param_blocks``, kkt.py:652-713) with the multipliers of the rows
tau >= t: dV_t/dp = -gamma^-t lam_g'(tau >= t)' dg/dp (the envelope theorem; the
stationarity rows do not enter). Its sum over the inside width samples equals Pi^in_t
(a uniform shift of the node values is a uniform shift of the spline). Where the road the
window reads ahead of x_t is straight (``straight_ahead``), lambda_n = Pi^R - Pi^L (K0).

**K1's shifts** (``pinned_bounds``, ``fd_costate``, ``fd_costate_k1``, ``fd_pi``; design
Section 7). For a plan state t >= 1 the prefix x_0..x_{t-1}, a_0..a_{t-1} is pinned at the
optimum, x_t's own corridor and slip rows and its box (not s's window bound) are relaxed, and
defect row t - 1 is shifted by c e_i, so dJ_W/dc = gamma^t lambda_{t,i} + gamma^(t-1)
[i = s]/10; for t = 0 the x_0 bounds move (dJ_W/dc = lambda_0). The prefix's own defect and
path rows, which then hold fixed variables only, are relaxed too (as constant rows they
would leave IPOPT a rank-deficient Jacobian; they cannot bind). The warm re-solves use step
2's Gate 1 solver (``kkt.fd_solver``: the key solver's options, REFINE_IPOPT's warm start,
tol 1e-10, kkt.py:980-995). ``fd_costate_k1`` applies K1's 'smaller c' where the Richardson
pair disagrees, and ``FD.kink`` K1's kink rule. Every estimate carries K1's zero-rule floor
(``gates.k1_error``) as ``FD.noise`` = 6 KINK_J_NOISE / (h gamma^t): ``fd_pi``'s at t = 1
and its step ``PI_H``, ``fd_pedal_below``'s from its ladder's J spread. Planted bugs
(``PLANTS``, ``ACTION_PLANTS``, ``PI_PLANTS``) must fail; 'unfolded' is the raw (un-folded)
multipliers through the same formula (``costate`` with an identity ``Fold``): it moves only
lambda_pedal, by alpha-weighted sums of the pedal-state box multipliers ahead (c_t), and K1
scores it on the pedal rows where c_t != 0.

**The pedal step at pedal-saturated states** (``fd_pedal_below``; a design clarification
recorded before the tag: the design's 1e-5 is replaced there). Shifting pedal at a pedal-
saturated state leaves about 300 near-active pedal-state bounds in the warm re-solves, whose
J then scatters by about 1.5e-8, a thousand times the 1e-11 the other components see; at
c = 1e-5 the one-sided difference is noise-limited (and shrinking c only adds noise). The
pedal component there is differenced one-sided from below on its own ladder
``PEDAL_LADDER`` = (1e-3, 5e-4, 2.5e-4): the largest converged c whose Richardson pair (c, c/2)
agrees within ``SHRINK_RTOL`` is accepted; a failed re-solve falls back to the next c; a
discrepancy that grows as c shrinks (the noise signature) moves c up once (``PEDAL_UP``)
instead; c never goes below ``PEDAL_MIN_H``. The J noise of that row is estimated from the
same solves (the spread of the converged rungs' Richardson values x the smallest c x
gamma^t / 6, at least ``KINK_J_NOISE``) and enters ``FD.noise``, K1's zero-rule floor.

**K0's verdict** (``k0_record``): the x_0 identity (``X0_TOL``), the anchored a_0 companion,
the two routes on T's rows (``ROUTE_TOL``), the cheap route's end gap (``ROUTE_TOL``), the
folded stationarity (``FOLD_TOL``) and command signs (``SIGN_TOL``), the action rows
(``ACTION_TOL``) with every planted action-row bug above ``PLANT_MIN``, sigma read from the
folded command multipliers exactly, lambda_n = Pi^R - Pi^L where the road ahead is straight
(``STRAIGHT_TOL``), and finiteness.

Units: values in reward units (1 = 10 m of progress); lambda per m (s, n), per rad (xi,
delta), per m/s (vx, vy), per rad/s (r), per unit (pedal) and pi per unit command (``UNITS``).
"""

from __future__ import annotations

import copy
import dataclasses
import functools
from dataclasses import dataclass

import casadi as ca
import numpy as np

from apexzero.costate import keys_root, stimuli_dir
from apexzero.fields import keys as K
from apexzero.fields import kkt as KK
from apexzero.fields.perturb import Samples
from apexzero.ocp.window import NA, NX, WindowSpec, path_functions
from apexzero.ocp.window import step_functions as _step_functions
from apexzero.vehicle.params import F1_2025, VehicleParams

STATE_NAMES = K.STATE_NAMES                      # s, n, xi, vx, vy, r, delta, pedal
ACTION_NAMES = K.ACTION_NAMES                    # steer, pedal
S, N, XI, VX, VY, R, DELTA, PEDAL = range(NX)
K_P = ("s", "n", "xi", "vx")                     # design Section 3.2: the primary components
SECONDARY = ("vy", "r", "delta", "pedal")
MIRROR_ODD = ("n", "xi", "vy", "r", "delta", "steer")   # odd under mirroring (Section 3.2)
UNITS = {"s": "reward/m", "n": "reward/m", "xi": "reward/rad", "vx": "reward/(m/s)",
         "vy": "reward/(m/s)", "r": "reward/(rad/s)", "delta": "reward/rad",
         "pedal": "reward/unit", "steer": "reward/unit command",
         "Pi_wall": "reward/m", "Pi_slip": "reward per unit ln alpha_max",
         "Pi_mu": "reward per unit mu_scale", "d": "m"}
REWARD_UNIT = "reward (1 = 10 m of progress)"
PATH_ROWS = ("left", "right", "slip_f", "slip_r")
CTX_NAMES = ("m", "p_max", "cla", "cda", "mu_scale", "aero_balance")   # answer_key.py:275-281
MU = 4                                           # ctx index of mu_scale
ACTIVE_TOL = 1e-6           # |multiplier| above this: the row / bound is active
SATURATED_TOL = ACTIVE_TOL  # |lam_x[8t+7]| above this: pedal-saturated. The pedal state nears
                            # 1 geometrically (ratio alpha per step), so the box multiplier has
                            # no gap: T's share is 0.28 / 0.24 / 0.19 at 1e-9 / 1e-8 / 1e-6
                            # (stored A2 multipliers), while the pedal command's is 0.294 from
                            # 1e-9 to 1e-5. Fixed at the tag (design Section 7 'Freeze').
ROUTE_TOL = 1e-10           # K0: jac_g and cheap routes agree
X0_TOL = 1e-12              # K0: the x_0 identity (exact by construction)
FOLD_TOL = 1e-10            # K0: stationarity with the pedal-state box removed
COMMAND_AT_BOUND = 1e-5     # |a| >= 1 - this: a command is at its bound (sign check;
                            # barrier slacks at z ~ 1e-5 are ~1e-6)
SIGN_TOL = 1e-10            # K0: a folded command multiplier's wrong-signed part
ACTION_TOL = 1e-9           # K0: the action rows (7e-11 on the 481 A2 keys; gamma^-t
                            # amplifies the stationarity residual on W = 800 keys)
STRAIGHT_TOL = 1e-12        # K0: lambda_n = Pi^R - Pi^L where the road ahead is straight
PLANT_MIN = 1e-4            # K0: every planted action-row bug must exceed this
ROUTES = ("jac", "cheap")
PLANTS = ("no_es", "no_discount", "flip_sign", "off_by_one", "pre_jump", "unfolded")
FD_H = {"s": 1e-4, "n": 1e-4, "vx": 1e-4, "vy": 1e-4, "xi": 1e-6, "r": 1e-6, "delta": 1e-6,
        "pedal": 1e-5}      # K1's steps (design Section 7); pedal-saturated states: PEDAL_LADDER
PEDAL_LADDER = (1e-3, 5e-4, 2.5e-4)   # K1's one-sided pedal steps at pedal-saturated states
PEDAL_UP = 2e-3             # tried once when the pair's discrepancy grows as c shrinks (noise)
PEDAL_MIN_H = 1e-4          # never below this for that row type


def rate(spec: WindowSpec) -> float:
    """d/d a_{t-1} of the action cost: 2 x 0.02 (window.py:470-490)."""
    return 2.0 * float(spec.action_cost)


def pedal_alpha(spec: WindowSpec, p: VehicleParams = F1_2025) -> float:
    """alpha of pedal_{t+1} = alpha pedal_t + (1 - alpha) a_t: RK2 (midpoint) substeps of
    dpedal = (a - pedal)/tau_pedal (window.py:224-229; dynamics.py:132)."""
    k = (float(spec.dt) / int(spec.substeps)) / float(p.tau_pedal)
    return float((1.0 - k + 0.5 * k * k) ** int(spec.substeps))


def assert_parameter_duals(prob: K.KeyProblem) -> None:
    """K0: the solver fills the fixed variables' multipliers (IPOPT's default
    ``make_parameter``; ``make_parameter_nodual`` would leave lam_x[0:8] empty)."""
    opt = prob.solver._opts.get("ipopt.fixed_variable_treatment", "make_parameter")
    if opt == "make_parameter_nodual":
        raise ValueError("fixed_variable_treatment = make_parameter_nodual: x_0 has no dual")


def with_multipliers(prob: K.KeyProblem, lam_g: np.ndarray, lam_x: np.ndarray) -> K.KeyProblem:
    """A shallow copy of ``prob`` carrying other multipliers (same point, P and bounds)."""
    new = copy.copy(prob)
    new.lam_g = np.asarray(lam_g, float).copy()
    new.lam_x = np.asarray(lam_x, float).copy()
    return new


def stationarity(prob: K.KeyProblem, lam_g: np.ndarray | None = None,
                 lam_x: np.ndarray | None = None) -> np.ndarray:
    """grad f + J_g' lam_g + lam_x [nw] on every variable, fixed ones included
    (``keys.stationarity``, keys.py:464-469), with other multipliers if given."""
    if lam_g is not None:
        prob = with_multipliers(prob, lam_g, prob.lam_x if lam_x is None else lam_x)
    return K.stationarity(prob)


# ------------------------------------------------------------------------- the pedal fold


@dataclass(frozen=True)
class Fold:
    """The multipliers of the TO-OCP without the pedal-state box (module docstring)."""

    alpha: float
    c: np.ndarray             # [W] c_t, t = 0..W-1
    lam_g: np.ndarray         # [ng] lam_g'
    lam_x: np.ndarray         # [nw] lam_x'
    pedal_box: np.ndarray     # [W+1] the raw pedal-state multipliers lam_x[8t+7]

    @property
    def saturated(self) -> np.ndarray:
        """[W+1] the raw pedal-state bound is active at x_t (x_0: fixed, never)."""
        out = np.abs(self.pedal_box) > SATURATED_TOL
        out[0] = False
        return out


def pedal_fold(prob: K.KeyProblem, alpha: float | None = None) -> Fold:
    """The fold of the module docstring (O(W)); ``alpha`` defaults to ``pedal_alpha``."""
    T = prob.T
    a = pedal_alpha(prob.spec) if alpha is None else float(alpha)
    lx = np.asarray(prob.lam_x, float)
    box = lx[NX * np.arange(T + 1) + PEDAL].copy()
    c = np.zeros(T)
    c[T - 1] = box[T]
    for t in range(T - 1, 0, -1):
        c[t - 1] = box[t] + a * c[t]
    lg = np.asarray(prob.lam_g, float).copy()
    lg[NX * np.arange(T) + PEDAL] += c
    lxf = lx.copy()
    lxf[NX * np.arange(1, T + 1) + PEDAL] = 0.0
    lxf[PEDAL] += a * c[0]
    lxf[NX * (T + 1) + NA * np.arange(T) + 1] += (1.0 - a) * c
    return Fold(alpha=a, c=c, lam_g=lg, lam_x=lxf, pedal_box=box)


def identity_fold(prob: K.KeyProblem) -> Fold:
    """The raw multipliers as a ``Fold`` with c = 0: K1's 'unfolded' planted bug."""
    T = prob.T
    lx = np.asarray(prob.lam_x, float).copy()
    return Fold(alpha=pedal_alpha(prob.spec), c=np.zeros(T),
                lam_g=np.asarray(prob.lam_g, float).copy(), lam_x=lx,
                pedal_box=lx[NX * np.arange(T + 1) + PEDAL].copy())


def fold_check(prob: K.KeyProblem, fold: Fold, A: np.ndarray, B: np.ndarray) -> dict:
    """K0's pedal fold: the pedal rows of Phi are the linear lag (max deviation), the
    stationarity residual with the folded multipliers on every variable, and the sign of the
    folded pedal-command multipliers (positive only at the upper bound, negative only at the
    lower; interior ones are reported by size). A fixed command (an anchored key's a_0,
    lbx = ubx) carries a fixed variable's multiplier, which has no sign: it is left out of
    the sign check."""
    T = prob.T
    lin = np.zeros((NX,))
    lin[PEDAL] = fold.alpha
    dev = max(float(np.max(np.abs(A[:, PEDAL, :] - lin))),
              float(np.max(np.abs(B[:, PEDAL, :] - np.array([0.0, 1.0 - fold.alpha])))))
    r = stationarity(prob, fold.lam_g, fold.lam_x)
    ic = NX * (T + 1) + NA * np.arange(T) + 1
    cmd = fold.lam_x[ic]
    a_ped = prob.a[:, 1]
    free = np.asarray(prob.lbx, float)[ic] < np.asarray(prob.ubx, float)[ic]
    up = (a_ped >= 1.0 - COMMAND_AT_BOUND) & free
    lo = (a_ped <= -1.0 + COMMAND_AT_BOUND) & free
    wrong = np.where(up, np.maximum(-cmd, 0.0), 0.0) + np.where(lo, np.maximum(cmd, 0.0), 0.0)
    inner = np.abs(np.where(up | lo, 0.0, cmd))
    return {"pedal_row_deviation": dev, "stationarity": float(np.max(np.abs(r))),
            "stationarity_raw": float(np.max(np.abs(stationarity(prob)))),
            "command_sign_violation": float(wrong.max(initial=0.0)),
            "command_interior_max": float(inner.max(initial=0.0)),
            "n_saturated": int(fold.saturated.sum()), "c_max": float(np.max(np.abs(fold.c)))}


# ------------------------------------------------------------------------- per-step functions


def dynamics_blocks(prob: K.KeyProblem) -> tuple[np.ndarray, np.ndarray]:
    """(Phi_x [W, 8, 8], Phi_a [W, 8, 2]) of every step from one ``jac_g`` evaluation."""
    return KK._dynamics_blocks(KK.jacobian(prob), prob.T)


def _window_params(prob: K.KeyProblem):
    T, Kc = prob.T, prob.spec.n_coef
    sl = K.parameter_slices(T, Kc)
    P = prob.P
    Pk, Pl, Pr = (P[sl[k]].reshape(T, Kc).T for k in ("kappa", "w_left", "w_right"))
    return Pk, Pl, Pr, P[sl["j0"]], P[sl["q"]], P[sl["ctx"]], P[sl["alpha_max"]]


@functools.cache
def _path(spec: WindowSpec) -> tuple[ca.Function, ca.Function]:
    path, path_jac, _ = path_functions(spec)
    return path, path_jac


def path_values(prob: K.KeyProblem) -> np.ndarray:
    """[W+1, 4] the path function (left, right corridor rows, slip rows) at every x_t: x_0
    with step 0's window, x_t (t >= 1) with step t-1's, as the NLP's rows read it."""
    T = prob.T
    _, Pl, Pr, j0, q, ctx, am = _window_params(prob)
    cols = np.r_[0, np.arange(T)]
    path, _ = _path(dataclasses.replace(prob.spec, steps=1))
    out = path.map(T + 1)(prob.x.T, Pl[:, cols], Pr[:, cols], j0[None, cols],
                          np.tile(q[:, None], (1, T + 1)), np.tile(ctx[:, None], (1, T + 1)),
                          np.tile(am[:, None], (1, T + 1)))
    return out.full().T


def path_jacobians(prob: K.KeyProblem) -> np.ndarray:
    """[W, 4, 8] d(path rows of step t)/d x_{t+1}."""
    T = prob.T
    _, Pl, Pr, j0, q, ctx, am = _window_params(prob)
    _, path_jac = _path(dataclasses.replace(prob.spec, steps=1))
    out = path_jac.map(T)(prob.x[1:].T, Pl, Pr, j0[None, :], np.tile(q[:, None], (1, T)),
                          np.tile(ctx[:, None], (1, T)), np.tile(am[:, None], (1, T)))
    return out.full().reshape(4, T, NX).transpose(1, 0, 2)


@functools.cache
def _ctx_functions(spec: WindowSpec) -> tuple[ca.Function, ca.Function]:
    """dPhi/dctx [8, 6] of one step and d path/dctx [4, 6] at a state (CasADi AD on the
    frozen, uncompiled step and path functions, as kkt.py:574-604 builds F_p)."""
    phi = _step_functions(spec)[0].expand()
    path = path_functions(spec)[0]
    x = ca.SX.sym("x", NX)
    a = ca.SX.sym("a", NA)
    W = ca.SX.sym("W", spec.n_coef)
    Wr = ca.SX.sym("Wr", spec.n_coef)
    j0 = ca.SX.sym("j0")
    q = ca.SX.sym("q", 4)
    ctx = ca.SX.sym("ctx", 6)
    am = ca.SX.sym("alpha_max")
    xn = phi(x, a, W, j0, q, ctx)
    dyn = ca.Function("dphi_dctx", [x, a, W, j0, q, ctx], [ca.jacobian(xn, ctx)], {"cse": True})
    gp = path(x, W, Wr, j0, q, ctx, am)
    pth = ca.Function("dpath_dctx", [x, W, Wr, j0, q, ctx, am], [ca.jacobian(gp, ctx)],
                      {"cse": True})
    return dyn, pth


def ctx_jacobians(prob: K.KeyProblem) -> tuple[np.ndarray, np.ndarray]:
    """(dPhi_t/dctx [W, 8, 6], d path_t(x_{t+1})/dctx [W, 4, 6])."""
    T = prob.T
    Pk, Pl, Pr, j0, q, ctx, am = _window_params(prob)
    dyn, pth = _ctx_functions(dataclasses.replace(prob.spec, steps=1))
    qT, cT, aT = (np.tile(v[:, None], (1, T)) for v in (q, ctx, am))
    d = dyn.map(T)(prob.x[:T].T, prob.a.T, Pk, j0[None, :], qT, cT).full()
    p = pth.map(T)(prob.x[1:].T, Pl, Pr, j0[None, :], qT, cT, aT).full()
    return (d.reshape(NX, T, 6).transpose(1, 0, 2), p.reshape(4, T, 6).transpose(1, 0, 2))


# ------------------------------------------------------------------------- the costate


@dataclass(frozen=True)
class Costate:
    """Targets of one key at every plan state t = 0..W (index t; value 0 at t = W where the
    value is 0). Folded multipliers throughout (``fold``)."""

    key_id: str
    kind: str
    W: int
    gamma: float
    decision: int             # 0 (A2: a*_0 at x_0) or 1 (anchored: a*_1 at x_1)
    unique_from: int          # lambda_t is unique for t >= this (anchored: 1)
    route: str
    plant: str | None
    x: np.ndarray             # [W+1, 8]
    a: np.ndarray             # [W, 2]
    lam: np.ndarray           # [W+1, 8] post-jump lambda_t (lambda_W = 0)
    lam_pre: np.ndarray       # [W+1, 8] pre-jump lambda-_t (t = 0: = lambda_0, no own rows)
    pi: np.ndarray            # [W+1, 2] dV_t/da_{t-1} (pi_W = 0)
    switch: np.ndarray        # [W, 2] sigma_t = gamma^-t lam_x'(a_t)
    nu: np.ndarray            # [W, 4] raw lam_g of the path rows of step t (at x_{t+1})
    box: np.ndarray           # [W+1, 8] folded lam_x' on x_t (x_0: its fixed-variable dual)
    command: np.ndarray       # [W, 2] raw lam_x on a_t (bound activity)
    pedal_box: np.ndarray     # [W+1] raw lam_x[8t+7]
    path: np.ndarray          # [W+1, 4] path rows at x_t (``path_values``)
    end_gap: float            # cheap route: |lambda-_W + jump_W| (0 by stationarity)
    edge_safety: float        # the OCP corridor's margin inside the env's edge (0.03 m)
    alpha_max: float          # rad
    fold: Fold
    multi_basin: bool = False

    @property
    def T(self) -> int:
        return self.W

    @property
    def jump(self) -> np.ndarray:
        """[W+1, 8] lambda_t - lambda-_t (nonzero only where x_t's own rows are active)."""
        return self.lam - self.lam_pre

    @property
    def own_corridor(self) -> np.ndarray:
        """[W+1, 2] x_t's own left / right corridor row is active (t = 0: none)."""
        return _own(np.abs(self.nu[:, :2]) > ACTIVE_TOL)

    @property
    def own_slip(self) -> np.ndarray:
        """[W+1, 2] x_t's own front / rear slip row is active."""
        return _own(np.abs(self.nu[:, 2:]) > ACTIVE_TOL)

    @property
    def own_box(self) -> np.ndarray:
        """[W+1, 8] x_t's own (folded) box is active (x_0 is fixed: never)."""
        out = np.abs(self.box) > ACTIVE_TOL
        out[0] = False
        return out

    @property
    def contact(self) -> np.ndarray:
        """[W+1] x_t's own corridor, slip or (folded) box row is active (design Section
        3.2's contact stratum: where the target jumps)."""
        return (self.own_corridor.any(1) | self.own_slip.any(1) | self.own_box.any(1))

    @property
    def pedal_saturated(self) -> np.ndarray:
        """[W+1] raw pedal-state bound active at x_t (design Section 3.2)."""
        return self.fold.saturated

    @property
    def clearance(self) -> np.ndarray:
        """[W+1, 2] distance (m) from x_t to the env's termination boundary, left / right."""
        return -self.path[:, :2] + self.edge_safety

    @property
    def slip_margin(self) -> np.ndarray:
        """[W+1] alpha_max - max(|alpha_f|, |alpha_r|) (rad) at x_t."""
        return self.alpha_max * (1.0 - np.max(np.abs(self.path[:, 2:]), axis=1))

    @property
    def command_saturated(self) -> np.ndarray:
        """[W, 2] a_t's bound is active (raw multiplier above ACTIVE_TOL): where the
        switching function is nonzero."""
        return np.abs(self.command) > ACTIVE_TOL

    @property
    def sigma_pedal(self) -> np.ndarray:
        """[W] gamma^-t x the folded pedal-command multiplier (C2-sw's target)."""
        return self.switch[:, 1]


def _own(rows: np.ndarray) -> np.ndarray:
    """Rows of step t (at x_{t+1}) -> x_t's own rows, [W+1, k] (x_0 has none)."""
    return np.vstack([np.zeros((1, rows.shape[1]), bool), rows])


def _check(route: str, plant: str | None) -> None:
    if route not in ROUTES:
        raise ValueError(f"unknown route {route!r}; expected one of {ROUTES}")
    if plant is not None and plant not in PLANTS:
        raise ValueError(f"unknown plant {plant!r}; expected one of {PLANTS}")
    if plant is not None and route != "jac":
        raise ValueError("planted bugs are built on the jac route")


def costate(prob: K.KeyProblem, route: str = "jac", plant: str | None = None,
            fold: Fold | None = None, blocks: tuple[np.ndarray, np.ndarray] | None = None
            ) -> Costate:
    """The targets of ``prob``'s point by the given route (module docstring). ``blocks``
    reuses (Phi_x, Phi_a) from ``dynamics_blocks``; ``plant`` builds one of K1's planted
    bugs (jac route only)."""
    _check(route, plant)
    if prob.solver.scaled:
        raise ValueError("step 1's keys are unscaled")
    assert_parameter_duals(prob)
    T, spec = prob.T, prob.spec
    gam = float(spec.gamma)
    if plant == "unfolded":
        fd = identity_fold(prob)
    else:
        fd = pedal_fold(prob) if fold is None else fold
    lg, lx = fd.lam_g, fd.lam_x
    es = np.zeros(NX)
    es[S] = 1.0 / float(spec.progress_scale)
    t = np.arange(T + 1)
    inv = gam ** (-t.astype(float))
    dyn = lg[: NX * T].reshape(T, NX)
    nu = np.asarray(prob.lam_g, float)[NX * T:].reshape(2, T, 2).transpose(1, 0, 2).reshape(T, 4)
    pre = np.zeros((T + 1, NX))
    pre[1:] = inv[1:, None] * (dyn - gam ** (t[1:, None] - 1.0) * es)
    box = lx[: NX * (T + 1)].reshape(T + 1, NX)
    Jp = path_jacobians(prob)
    jump = np.zeros((T + 1, NX))
    jump[1:] = inv[1:, None] * (np.einsum("tki,tk->ti", Jp, nu) + box[1:])
    lam = np.zeros((T + 1, NX))
    end_gap = float(np.max(np.abs(pre[T] + jump[T])))
    if route == "jac":
        A, _ = dynamics_blocks(prob) if blocks is None else blocks
        blk = dyn
        if plant == "off_by_one":
            blk = np.vstack([np.zeros((1, NX)), dyn[:-1]])
        raw = np.einsum("tij,ti->tj", A, blk)
        if plant == "no_discount":
            lam[:T] = raw - es
        elif plant == "no_es":
            lam[:T] = inv[:T, None] * raw
        else:
            lam[:T] = inv[:T, None] * raw - es
        if plant == "flip_sign":
            lam = -lam
    else:
        lam[0] = box[0]
        lam[1:T] = pre[1:T] + jump[1:T]
    pre[0] = lam[0]
    if plant == "pre_jump":
        lam = pre.copy()
        lam[T] = 0.0
    a = prob.a
    a_prev = np.vstack([np.zeros((1, NA)), a[:-1]])
    pi = np.zeros((T + 1, NA))
    pi[:T] = rate(spec) * (a - a_prev)
    cmd = lx[NX * (T + 1):].reshape(T, NA)
    switch = inv[:T, None] * cmd
    dec = K.decision_step(prob.rec)
    return Costate(key_id=prob.rec.key_id, kind=prob.rec.kind, W=T, gamma=gam, decision=dec,
                   unique_from=dec, route=route, plant=plant, x=prob.x.copy(), a=a.copy(),
                   lam=lam, lam_pre=pre, pi=pi, switch=switch, nu=nu, box=box.copy(),
                   command=np.asarray(prob.lam_x, float)[NX * (T + 1):].reshape(T, NA).copy(),
                   pedal_box=fd.pedal_box, path=path_values(prob), end_gap=end_gap,
                   edge_safety=float(spec.safety), alpha_max=float(prob.p.alpha_max), fold=fd,
                   multi_basin=bool(getattr(prob.rec, "multi_basin", False)))


# ------------------------------------------------------------------------- K0 identities


def x0_identity(prob: K.KeyProblem, A0: np.ndarray) -> float:
    """max |lam_x[0:8] - (Phi_x(x_0, a*_0)' lam_g[0:8] - e_s/10)| on the raw multipliers
    (exact by construction under make_parameter; design Section 7, K0)."""
    es = np.zeros(NX)
    es[S] = 1.0 / float(prob.spec.progress_scale)
    return float(np.max(np.abs(prob.lam_x[:NX] - (A0.T @ prob.lam_g[:NX] - es))))


def a0_identity(prob: K.KeyProblem, B0: np.ndarray) -> float:
    """Anchored keys: max |lam_x(a_0) - (Phi_a(x_0, a_0)' lam_g[0:8] - 0.04 a_0
    + 0.04 gamma (a*_1 - a_0))| with a_{-1} = 0 (K0's a_0 companion)."""
    T = prob.T
    a0, a1 = prob.a[0], prob.a[1]
    k, gam = rate(prob.spec), float(prob.spec.gamma)
    lam_a0 = prob.lam_x[NX * (T + 1): NX * (T + 1) + NA]
    return float(np.max(np.abs(lam_a0 - (B0.T @ prob.lam_g[:NX] - k * a0 + k * gam * (a1 - a0)))))


ACTION_PLANTS = ("no_es", "no_pi", "no_discount")


def action_rows(cs: Costate, B: np.ndarray, spec: WindowSpec, plant: str | None = None
                ) -> np.ndarray:
    """[W, 2] Phi_a' (e_s/10 + gamma lambda-_{t+1}) + gamma pi_{t+1} - 0.04 (a_t - a_{t-1})
    (design Section 6, C2; pi_W = 0). ``plant``: 'no_es', 'no_pi' or 'no_discount' (the
    identity is then compared with lam_x'(a_t), not gamma^-t lam_x'(a_t))."""
    if plant is not None and plant not in ACTION_PLANTS:
        raise ValueError(f"unknown plant {plant!r}; expected one of {ACTION_PLANTS}")
    gam = cs.gamma
    es = np.zeros(NX)
    if plant != "no_es":
        es[S] = 1.0 / float(spec.progress_scale)
    rhs = np.einsum("tia,ti->ta", B, es[None, :] + gam * cs.lam_pre[1:])
    if plant != "no_pi":
        rhs = rhs + gam * cs.pi[1:]
    a_prev = np.vstack([np.zeros((1, NA)), cs.a[:-1]])
    return rhs - rate(spec) * (cs.a - a_prev)


def action_identity(cs: Costate, B: np.ndarray, spec: WindowSpec, plant: str | None = None
                    ) -> float:
    """max |sigma_t - action_rows| over t < W and both actions (K0's action rows)."""
    lhs = cs.switch
    if plant == "no_discount":
        lhs = cs.switch * cs.gamma ** np.arange(cs.W, dtype=float)[:, None]
    return float(np.max(np.abs(lhs - action_rows(cs, B, spec, plant))))


def switch_gap(cs: Costate) -> float:
    """max |sigma_t - gamma^-t lam_x'(a_t)| relative to the latter's scale: 0 when C2-sw's
    sigma is read from the folded command multipliers (the ones stationarity verified)."""
    T = cs.W
    inv = cs.gamma ** (-np.arange(T, dtype=float))
    want = inv[:, None] * cs.fold.lam_x[NX * (T + 1):].reshape(T, NA)
    return float(np.max(np.abs(cs.switch - want)) / max(float(np.max(np.abs(want))), 1.0))


def route_gap(a: Costate, b: Costate) -> np.ndarray:
    """[W] max over components of |lambda_t^a - lambda_t^b|, t < W."""
    return np.max(np.abs(a.lam[:-1] - b.lam[:-1]), axis=1)


def straight_ahead(prob: K.KeyProblem, atol: float = 1e-12,
                   width_atol: float | None = None) -> np.ndarray:
    """[W+1] the road the window reads from x_t's spline interval on is straight: every
    packed kappa coefficient up to the last one a lookup at s_W reads is below ``atol`` (the
    natural spline's wiggles past a corner decay by about 0.27 per knot, so 'straight'
    starts some knots after s_c1). That suffices for lambda_n = Pi^R - Pi^L: with kappa = 0
    the dynamics are invariant to shifting n, and n_tau + d <= w_l(s_tau) - e is
    n_tau <= (w_l(s_tau) - d) - e whatever the widths' shape. The design's 'and of constant
    width' (Section 3.4) is checked too when ``width_atol`` is given (the long strips' widths
    carry float32-level noise of ~2e-7 m along the straight)."""
    from apexzero.tracks.spline import packed_coeffs

    tr = prob.track
    n, ds = tr.n_points, float(tr.ds)
    # packed index j holds c_{j-1}; a lookup at s reads c_{i-1}..c_{i+2}, i = floor(s/ds)
    i = np.clip(np.floor(prob.x[:, 0] / ds).astype(int), 0, n - 2)
    reach = int(i.max()) + 3
    ck = np.abs(packed_coeffs(np.asarray(tr.kappa, float), False, n))[: reach + 1]
    ok = np.maximum.accumulate(ck[::-1])[::-1] <= atol
    if width_atol is not None:
        for w in (tr.w_left, tr.w_right):
            c = packed_coeffs(np.asarray(w, float), False, n)[: reach + 1]
            hi = np.maximum.accumulate(c[::-1])[::-1]
            lo = np.minimum.accumulate(c[::-1])[::-1]
            ok &= (hi - lo) <= width_atol
    return ok[i]


@dataclass(frozen=True)
class Prices:
    """Shadow prices of the walls, the slip limit and the car context (module docstring),
    at t = 0..W (0 at t = W)."""

    direction: int            # sigma: +1 left turn (inside = left), -1 right
    left: np.ndarray          # [W+1]
    right: np.ndarray
    slip: np.ndarray
    ctx: np.ndarray           # [W+1, 6] (column MU: the grip price)
    ctx_path: np.ndarray      # [W+1, 6] the path rows' part of ctx (0 for mu)
    d_left: np.ndarray        # [W+1] nu-weighted mean distance ahead (m); nan without mass
    d_right: np.ndarray

    @property
    def inside(self) -> np.ndarray:
        return self.left if self.direction > 0 else self.right

    @property
    def outside(self) -> np.ndarray:
        return self.right if self.direction > 0 else self.left

    @property
    def d_in(self) -> np.ndarray:
        return self.d_left if self.direction > 0 else self.d_right

    @property
    def d_out(self) -> np.ndarray:
        return self.d_right if self.direction > 0 else self.d_left

    @property
    def mu(self) -> np.ndarray:
        return self.ctx[:, MU]


def _tail(v: np.ndarray) -> np.ndarray:
    """[W+1(, k)] sum_{tau >= t} v_tau for t = 0..W of v [W(, k)] (0 at t = W)."""
    c = np.cumsum(v[::-1], axis=0)[::-1]
    return np.concatenate([c, np.zeros((1, *v.shape[1:]))], axis=0)


def prices(prob: K.KeyProblem, cs: Costate, direction: int | None = None,
           ctx_blocks: tuple[np.ndarray, np.ndarray] | None = None) -> Prices:
    T, gam = cs.W, cs.gamma
    inv = gam ** (-np.arange(T + 1, dtype=float))
    sig = int(prob.rec.direction if direction is None else direction)
    if sig not in (1, -1):
        raise ValueError(f"turn direction {sig}: expected +1 or -1")
    nuL, nuR = cs.nu[:, 0], cs.nu[:, 1]
    sl = np.abs(cs.nu[:, 2]) + np.abs(cs.nu[:, 3])
    dPhi, dPath = ctx_jacobians(prob) if ctx_blocks is None else ctx_blocks
    dyn = cs.fold.lam_g[: NX * T].reshape(T, NX)
    by_dyn = np.einsum("tic,ti->tc", dPhi, dyn)
    by_path = -np.einsum("tkc,tk->tc", dPath, cs.nu)
    s = cs.x[:, 0]

    def mean_ahead(nu):
        m = _tail(nu)
        ms = _tail(nu * s[1:])
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(m > ACTIVE_TOL, ms / np.where(m > ACTIVE_TOL, m, 1.0) - s, np.nan)
        return d

    return Prices(direction=sig, left=inv * _tail(nuL), right=inv * _tail(nuR),
                  slip=inv * _tail(sl), ctx=inv[:, None] * _tail(by_dyn + by_path),
                  ctx_path=inv[:, None] * _tail(by_path), d_left=mean_ahead(nuL),
                  d_right=mean_ahead(nuR))


def road_field(prob: K.KeyProblem, cs: Costate, t: int, blocks: KK.ParamBlocks | None = None
               ) -> Samples:
    """dV_t/d(samples) of kappa, w_left and w_right: -gamma^-t lam_g'(rows tau >= t)' dg/dp
    through step 2's F_p blocks (module docstring)."""
    T = prob.T
    if not 0 <= t < T:
        raise ValueError(f"t = {t} outside 0..{T - 1}")
    blk = KK.param_blocks(prob) if blocks is None else blocks
    lg = np.zeros_like(cs.fold.lam_g)
    lg[NX * t: NX * T] = cs.fold.lam_g[NX * t: NX * T]
    lg[NX * T + 2 * t: 10 * T] = cs.fold.lam_g[NX * T + 2 * t: 10 * T]
    out = blk.transpose(np.zeros(prob.solver.nw), lg)
    return Samples(*(-(cs.gamma ** -t) * np.asarray(v, float) for v in out))


def k0_record(prob: K.KeyProblem, jac: Costate, cheap: Costate, pr: Prices,
              blocks: tuple[np.ndarray, np.ndarray], rows: np.ndarray | None = None) -> dict:
    """K0 on one key (design Section 7): the x_0 identity, the a_0 companion (anchored keys),
    the two routes on the state rows ``rows`` (T's steps; default every t from
    ``unique_from``; the maximum over all t < W is reported beside it, where gamma^-t
    amplifies the stationarity residual up to ~20 on W = 800 keys), the pedal fold, the
    action rows, lambda_n = Pi^R - Pi^L where the road ahead is straight, the jump
    lambda - lambda- per component, and finiteness. JSON-ready."""
    A, B = blocks
    gap = route_gap(jac, cheap)
    rows = np.arange(jac.unique_from, jac.W) if rows is None else np.asarray(rows, int)
    rows = rows[rows >= jac.unique_from]
    st = straight_ahead(prob)
    st[-1] = False
    dn = np.abs(jac.lam[:, N] - (pr.right - pr.left))
    anch = K.is_anchored(prob.rec)
    out = {"key_id": prob.rec.key_id, "kind": prob.rec.kind, "W": prob.T,
           "x0_identity": x0_identity(prob, A[0]),
           "a0_identity": a0_identity(prob, B[0]) if anch else None,
           "route_gap": float(gap[rows].max()) if len(rows) else 0.0,
           "route_gap_all": float(gap[jac.unique_from:].max()), "end_gap": cheap.end_gap,
           "action_identity": action_identity(jac, B, prob.spec),
           "action_plants": {p: action_identity(jac, B, prob.spec, p) for p in ACTION_PLANTS},
           "switch_gap": switch_gap(jac),
           **fold_check(prob, jac.fold, A, B),
           "n_straight": int(st.sum()),
           "straight_identity": float(dn[st].max()) if st.any() else None,
           "jump_max": [float(v) for v in np.max(np.abs(jac.jump), axis=0)],
           "finite": bool(np.all(np.isfinite(jac.lam)) and np.all(np.isfinite(jac.pi))
                          and np.all(np.isfinite(jac.switch)))}
    out["plants_fail"] = bool(all(v > PLANT_MIN for v in out["action_plants"].values()))
    out["ok"] = bool(out["x0_identity"] <= X0_TOL and out["route_gap"] <= ROUTE_TOL
                     and (not anch or out["a0_identity"] <= X0_TOL)
                     and out["end_gap"] <= ROUTE_TOL
                     and out["stationarity"] <= FOLD_TOL
                     and out["command_sign_violation"] <= SIGN_TOL
                     and out["action_identity"] <= ACTION_TOL
                     and out["switch_gap"] <= 1e-12
                     and (out["straight_identity"] is None
                          or out["straight_identity"] <= STRAIGHT_TOL)
                     and out["plants_fail"] and out["finite"])
    return out


# ------------------------------------------------------------------------- loading


@dataclass(frozen=True)
class KeyTargets:
    """One stored key, re-polished, with its targets by both routes and its prices."""

    problem: K.KeyProblem
    repolish: K.Repolish
    jac: Costate
    cheap: Costate
    prices: Prices
    blocks: tuple[np.ndarray, np.ndarray]


class RepolishFailed(ValueError):
    """The re-polish did not succeed (``keys.Repolish.ok``): no target is read there."""


def load_problem(ref: K.KeyRef, root=None, bank=None) -> tuple[K.KeyProblem, K.Repolish]:
    """The re-polished problem of a stored key (``keys.repolish`` of ``keys.rebuild``; design
    Section 2: idea 2 re-polishes every key)."""
    if bank is None:
        bank, _ = K.design(stimuli_dir())
    rp = K.repolish(K.rebuild(K.load_key(ref, root=keys_root(root)), bank))
    if not rp.ok:
        raise RepolishFailed(f"key {ref.key_id}: re-polish status {rp.status}, block_ok "
                             f"{rp.block_ok}")
    return rp.problem, rp


def key_targets(ref: K.KeyRef, root=None, bank=None) -> KeyTargets:
    prob, rp = load_problem(ref, root, bank)
    blocks = dynamics_blocks(prob)
    jac = costate(prob, "jac", blocks=blocks)
    cheap = costate(prob, "cheap", fold=jac.fold)
    return KeyTargets(problem=prob, repolish=rp, jac=jac, cheap=cheap, prices=prices(prob, jac),
                      blocks=blocks)


def targets_change(stored: Costate, polished: Costate) -> dict:
    """K0's report of the re-polish: max |change| of lambda, pi and sigma (over t < W)."""
    return {"lam": float(np.max(np.abs(stored.lam - polished.lam))),
            "pi": float(np.max(np.abs(stored.pi - polished.pi))),
            "switch": float(np.max(np.abs(stored.switch - polished.switch)))}


# ------------------------------------------------------------------------- K1's shifts


Bounds = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def pinned_bounds(prob: K.KeyProblem, t: int) -> Bounds:
    """(lbx, ubx, lbg, ubg) with the prefix x_0..x_{t-1}, a_0..a_{t-1} pinned at the
    optimum, x_t's own corridor and slip rows (step t - 1's) and x_t's box relaxed, s's
    window bound kept (design Section 7, K1). t >= 1."""
    T = prob.T
    if not 1 <= t < T:
        raise ValueError(f"t = {t} outside 1..{T - 1}")
    lbx, ubx, lbg, ubg = (np.array(v, float) for v in (prob.lbx, prob.ubx, prob.lbg, prob.ubg))
    pin = np.r_[np.arange(NX * t), NX * (T + 1) + np.arange(NA * t)]
    lbx[pin] = ubx[pin] = prob.w[pin]
    own = NX * t + np.arange(1, NX)
    lbx[own], ubx[own] = -np.inf, np.inf
    # x_t's own path rows (step t - 1's) are relaxed, and so are the prefix's defect and path
    # rows (steps 0..t-2): they involve fixed variables only, cannot bind, and as constant
    # rows would leave IPOPT a rank-deficient constraint Jacobian
    path = np.r_[NX * T + np.arange(2 * t), 10 * T + np.arange(2 * t)]
    lbg[path], ubg[path] = -np.inf, np.inf
    lbg[: NX * (t - 1)], ubg[: NX * (t - 1)] = -np.inf, np.inf
    return lbx, ubx, lbg, ubg


def resolve_bounds(prob: K.KeyProblem, b: Bounds, solver=None) -> KK.Resolve:
    """A warm re-solve of the key's NLP (same P) under other bounds, from the problem's
    primal-dual point with the multipliers of relaxed sides zeroed."""
    import time

    rs = KK.fd_solver(prob) if solver is None else solver
    lbx, ubx, lbg, ubg = b
    lam_x = np.where(np.isfinite(lbx) | np.isfinite(ubx), prob.lam_x, 0.0)
    lam_g = np.where(np.isfinite(lbg) | np.isfinite(ubg), prob.lam_g, 0.0)
    t0 = time.perf_counter()
    sol = rs(x0=prob.w, p=prob.P, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, lam_g0=lam_g, lam_x0=lam_x)
    secs = time.perf_counter() - t0
    st = rs.stats()
    return KK.Resolve(w=np.asarray(sol["x"]).ravel(), lam_g=np.asarray(sol["lam_g"]).ravel(),
                      lam_x=np.asarray(sol["lam_x"]).ravel(), status=str(st["return_status"]),
                      iterations=int(st["iter_count"]), seconds=secs, J=-float(sol["f"]))


def shifted_bounds(prob: K.KeyProblem, t: int, i: int, c: float, base: Bounds | None = None
                   ) -> Bounds:
    """K1's shift of state component i of x_t by c: the x_0 bounds for t = 0, else defect
    row t - 1 on the pinned problem (``pinned_bounds``)."""
    if t == 0:
        lbx, ubx, lbg, ubg = (np.array(v, float) for v in (prob.lbx, prob.ubx, prob.lbg,
                                                            prob.ubg))
        lbx[i] = ubx[i] = prob.w[i] + c
        return lbx, ubx, lbg, ubg
    lbx, ubx, lbg, ubg = (np.array(v, float) for v in (pinned_bounds(prob, t) if base is None
                                                        else base))
    lbg[NX * (t - 1) + i] = ubg[NX * (t - 1) + i] = c
    return lbx, ubx, lbg, ubg


@dataclass(frozen=True)
class FD:
    """A finite-difference estimate of one target from warm re-solves (target units)."""

    value: float              # Richardson estimate (lambda_{t,i} or pi_{1,j})
    fd_h: float               # the estimate at h
    fd_h2: float              # at h/2
    discrepancy: float        # |fd_h2 - fd_h|
    side: str                 # 'central' or 'below' (one-sided, c < 0)
    h: float
    ok: bool                  # every re-solve converged
    iterations: int
    forward: float = float("nan")    # Richardson one-sided estimates (central side only)
    backward: float = float("nan")
    noise: float = 0.0        # K1's floor and kink noise term: 6 KINK_J_NOISE / (h gamma^t)
    ladder: tuple = ()        # pedal-saturated rows: (h, value, discrepancy, ok) per c tried

    @property
    def kink(self) -> bool:
        """K1's kink rule: the one-sided estimates disagree by more than KINK_FACTOR x
        max(Richardson discrepancy, ``noise``); such a comparison does not count toward the
        pass. Forward and backward contain the unshifted re-solve J(0) with weight 6/h
        (forward - backward = (4 J(h/2) - J(h) - 6 J(0) + 4 J(-h/2) - J(-h)) / (h gamma^t))
        and the central pair does not, so a J(0) offset of KINK_J_NOISE between re-solves
        moves forward - backward by ``noise`` = 6 KINK_J_NOISE / (h gamma^t). Section 7's
        KINK_FACTOR x discrepancy alone is no kink test under re-solve scatter, since both
        statistics scale with it: on a smooth J(c) with K1's stencil it flags about 31% of
        comparisons for independent J scatter of any size, and 84% or more when J(0) carries an
        offset of its own (``gates.literal_kink`` reports that reading). The noise term only
        removes flags: every kink by this rule is one by Section 7's."""
        if self.side != "central":
            return False
        return bool(abs(self.forward - self.backward)
                    > KINK_FACTOR * max(self.discrepancy, self.noise))


KINK_FACTOR = 10.0          # K1 (design Section 7)
KINK_J_NOISE = 1e-11        # J's spread between tol-1e-10 re-solves (seen: ~1e-11)
K1_MEDIAN = 1e-3            # K1's pass: per component, median relative error ...
K1_P95 = 1e-2               # ... and p95
ZERO_ATOL = KK.ZERO_ATOL    # Gate 1's zero-field rule (kkt.py:163)


def fd_error(target: float, fd: FD, zero_atol: float = ZERO_ATOL) -> float:
    """|target - FD_R| / max(|FD_R|, 3 discrepancy) (Gate 1's score, kkt.py:1068-1092); 0 when
    both are below ``zero_atol`` (a zero field)."""
    if abs(target) < zero_atol and abs(fd.value) < zero_atol:
        return 0.0
    return abs(target - fd.value) / max(abs(fd.value), 3.0 * fd.discrepancy, 1e-300)


def _richardson(d_h: float, d_h2: float, side: str) -> float:
    return (4.0 * d_h2 - d_h) / 3.0 if side == "central" else 2.0 * d_h2 - d_h


def fd_costate(prob: K.KeyProblem, t: int, i: int, h: float | None = None,
               side: str = "central", solver=None) -> FD:
    """lambda_{t,i} from re-solves of the shifted problem: Richardson over (h, h/2), central
    or one-sided from below (pedal-saturated states, design Section 7), with the one-sided
    Richardson estimates of the same solves for the kink rule."""
    if side not in ("central", "below"):
        raise ValueError(f"side {side!r}")
    h = FD_H[STATE_NAMES[i]] if h is None else float(h)
    gam = float(prob.spec.gamma)
    base = pinned_bounds(prob, t) if t >= 1 else None
    ok, its = True, 0
    cache: dict[float, float] = {}

    def J(c):
        nonlocal ok, its
        if c not in cache:
            r = resolve_bounds(prob, shifted_bounds(prob, t, i, c, base), solver)
            ok &= r.ok
            its += r.iterations
            cache[c] = r.J
        return cache[c]

    def d(hh):
        if side == "central":
            return (J(hh) - J(-hh)) / (2.0 * hh)
        return (J0 - J(-hh)) / hh

    J0 = J(0.0)              # re-solved too: the one-sided estimates' offset cancels
    d_h, d_h2 = d(h), d(0.5 * h)
    shift = gam ** (t - 1) / float(prob.spec.progress_scale) if (t >= 1 and i == S) else 0.0
    scale = gam ** t

    def to_lam(v):
        return (v - shift) / scale

    fwd = bwd = float("nan")
    if side == "central":
        fwd = to_lam(2.0 * (J(0.5 * h) - J0) / (0.5 * h) - (J(h) - J0) / h)
        bwd = to_lam(2.0 * (J0 - J(-0.5 * h)) / (0.5 * h) - (J0 - J(-h)) / h)
    l_h, l_h2 = to_lam(d_h), to_lam(d_h2)
    return FD(value=to_lam(_richardson(d_h, d_h2, side)), fd_h=l_h, fd_h2=l_h2,
              discrepancy=abs(l_h2 - l_h), side=side, h=h, ok=bool(ok), iterations=its,
              forward=fwd, backward=bwd, noise=6.0 * KINK_J_NOISE / (h * scale))


SHRINK_RTOL = 1e-3          # K1: Richardson discrepancy above this x |FD| -> a smaller c
MAX_SHRINK = 2              # at most two tenfold reductions of c


def fd_costate_k1(prob: K.KeyProblem, t: int, i: int, side: str = "central",
                  rtol: float = SHRINK_RTOL, max_shrink: int = MAX_SHRINK, solver=None) -> FD:
    """``fd_costate`` at K1's step, retried with c / 10 (at most ``max_shrink`` times) while
    the Richardson pair disagrees by more than ``rtol`` of the estimate and by more than the
    round-off of the re-solves (``FD.noise``; a smaller c would only add round-off): K1's
    'or a smaller c' where a shifted x_t pushes past a nearby contact (design Section 7).
    The last attempt is returned (its kink flag and discrepancy decide whether it
    counts)."""
    if i == PEDAL and side == "below":
        return fd_pedal_below(prob, t, solver=solver, rtol=rtol)
    h = FD_H[STATE_NAMES[i]]
    fd = fd_costate(prob, t, i, h, side, solver)
    for _ in range(max_shrink):
        if fd.kink or fd.discrepancy <= max(rtol * abs(fd.value), fd.noise):
            break
        h *= 0.1
        fd = fd_costate(prob, t, i, h, side, solver)
    return fd


def fd_pedal_below(prob: K.KeyProblem, t: int, ladder=PEDAL_LADDER, rtol: float = SHRINK_RTOL,
                   solver=None) -> FD:
    """lambda_pedal at a pedal-saturated state, one-sided from below on its own ladder
    (module docstring 'The pedal step at pedal-saturated states'). Returns the accepted
    estimate (or, with none accepted, the converged one with the smallest relative
    discrepancy, which K1 then scores as unresolved), its ``noise`` from the J noise the
    ladder's solves show, and every rung in ``FD.ladder``."""
    scale = float(prob.spec.gamma) ** t
    tried: list[FD] = []
    accepted, prev, noisy = None, None, False
    for h in ladder:
        if h < PEDAL_MIN_H:
            break
        fd = fd_costate(prob, t, PEDAL, h, "below", solver)
        tried.append(fd)
        if not fd.ok:
            continue                       # a failed re-solve: the next (smaller) c
        if fd.discrepancy <= rtol * abs(fd.value):
            accepted = fd
            break
        if prev is not None and fd.discrepancy > prev:
            noisy = True                   # grows as c shrinks: noise, so go up instead
            break
        prev = fd.discrepancy
    if accepted is None and noisy:
        fd = fd_costate(prob, t, PEDAL, PEDAL_UP, "below", solver)
        tried.append(fd)
        if fd.ok and fd.discrepancy <= rtol * abs(fd.value):
            accepted = fd
    good = [f for f in tried if f.ok]
    eps = KINK_J_NOISE
    if len(good) >= 2:
        vals = [f.value for f in good]
        eps = max(eps, (max(vals) - min(vals)) * min(f.h for f in good) * scale / 6.0)
    if accepted is not None:
        best = accepted
    elif good:
        best = min(good, key=lambda f: f.discrepancy / max(abs(f.value), 1e-300))
    else:
        best = tried[-1]
    return dataclasses.replace(best, noise=6.0 * eps / (best.h * scale),
                               ladder=tuple((f.h, f.value, f.discrepancy, f.ok) for f in tried))


PI_PLANTS = ("plus_shift", "r0_prev", "j_w")
PI_H = 1e-5                 # K1's step for pi (design Section 7)


def fd_pi(prob: K.KeyProblem, j: int, h: float = PI_H, plant: str | None = None,
          a_prev: np.ndarray | None = None, solver=None) -> FD:
    """pi_{1,j} on an anchored key (design Section 7, K1 'pi on 10 anchored keys'): a_0's
    bounds moved to a_0 + c e_j, defect row 0 set to g_0 at (x_0, a_0 + c, x_1) so x_1 is
    exactly unchanged, and J_tail = (J_W - r_0)/gamma with r_0 = (s_1 - s_0)/10
    - 0.02 ||a_0 + c - a_{-1}||^2 from the re-solve's own w, a_{-1} = 0. Planted bugs:
    'plus_shift' (defect row 0 shifted the other way), 'r0_prev' (r_0 with ``a_prev`` as
    a_{-1}, e.g. the key's anchor['a_prev'] or the agent's a_{L-2}) and 'j_w' (J_W in place
    of J_tail).

    ``FD.noise`` is K1's floor built as a lambda_1 comparison's (``fd_costate``): the
    estimate is the same central Richardson stencil over (h, h/2), four warm re-solves and
    no J(0), and J errors of KINK_J_NOISE reach J_tail divided by gamma, as they reach
    lambda_1 = dJ_W/dc / gamma, so the floor is 6 KINK_J_NOISE / (h gamma) (twice the
    largest error they can give the estimate, ``gates.k1_error``). There is no kink test
    (forward and backward are not formed). A planted FD carries the same floor."""
    if not K.is_anchored(prob.rec):
        raise ValueError("pi's shift is defined on anchored keys (a_0 fixed)")
    if plant is not None and plant not in PI_PLANTS:
        raise ValueError(f"unknown plant {plant!r}; expected one of {PI_PLANTS}")
    T, spec = prob.T, prob.spec
    gam, ia = float(spec.gamma), NX * (T + 1)
    prev = np.zeros(NA) if plant != "r0_prev" else np.asarray(a_prev, float)
    _, g_base = K.constraints(prob)
    ok, its = True, 0

    def J_tail(c):
        nonlocal ok, its
        lbx, ubx, lbg, ubg = (np.array(v, float) for v in (prob.lbx, prob.ubx, prob.lbg,
                                                            prob.ubg))
        w = prob.w.copy()
        w[ia + j] += c
        lbx[ia: ia + NA] = ubx[ia: ia + NA] = w[ia: ia + NA]
        g0 = K.constraints(prob, w)[1][:NX]
        if plant == "plus_shift":
            g0 = 2.0 * g_base[:NX] - g0
        lbg[:NX] = ubg[:NX] = g0
        r = resolve_bounds(prob, (lbx, ubx, lbg, ubg), solver)
        ok &= r.ok
        its += r.iterations
        x = r.w[: NX * (T + 1)].reshape(T + 1, NX)
        a0 = r.w[ia: ia + NA]
        r0 = (x[1, 0] - x[0, 0]) / float(spec.progress_scale) - float(spec.action_cost) * float(
            np.sum((a0 - prev) ** 2))
        return r.J if plant == "j_w" else (r.J - r0) / gam

    def d(hh):
        return (J_tail(hh) - J_tail(-hh)) / (2.0 * hh)

    d_h, d_h2 = d(h), d(0.5 * h)
    return FD(value=_richardson(d_h, d_h2, "central"), fd_h=d_h, fd_h2=d_h2,
              discrepancy=abs(d_h2 - d_h), side="central", h=h, ok=bool(ok), iterations=its,
              noise=6.0 * KINK_J_NOISE / (h * gam))
