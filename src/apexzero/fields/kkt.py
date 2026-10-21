"""The optimum's receptive fields: barrier sensitivity at a re-polished key (design Section 5
'Optimum', Gates 1 and 4).

**The NLP** is a key's TO-OCP as ``keys.rebuild`` (keys.py:407) and ``keys.repolish``
(keys.py:529) give it: w = [vec(X) (T+1 states of 8); vec(A) (T actions of 2)], and g
(window.py:498-502) holds the defects x_{t+1} - Phi_t (rows 8t+i), the corridor rows 8T+2t+k
(n - (w_l - edge), -(w_r - edge) - n at x_{t+1}) and the slip rows 10T+2t+k. f = -J. CasADi's
convention is L = f + lam_g'g + lam_x'w, with lam = z_U - z_L (positive at an upper bound).
Gate 0 (``keys.stationarity``) pins it.

**The barrier KKT system** is IPOPT's primal-dual system at a fixed mu, with the slacks
condensed out. Every bound side and every inequality row keeps its barrier term:

- slacks s come from w against lbx/ubx, and from g(w) against lbg/ubg;
- z on the side that lam's sign points to is |lam| plus the hidden side's value; the hidden
  side has z = mu/s. So z_U - z_L = lam exactly (``Barrier``);
- mu is the median of s z over the strongly active sides (z >= 1e-4). On step 1's keys it is
  1.0e-11 to 1.7e-11 (IPOPT's mu_min is 1e-11).

The matrix is K = [[H + Sigma_x, J'], [J, -D]] over the free variables, those with
lbx < ubx (IPOPT's fixed variables are parameters). So x_0 is held on A2 keys, and x_0 and a_0
on anchored keys (design 'option A'; no x_1 hold).
- H = hess_lag(lam_f = 1, lam_g), symmetrised from its upper triangle (window.py:533-570):
  U + U' - diag U.
- Sigma_x = z_L/s_L + z_U/s_U.
- D = 1 / (z_L/s_L + z_U/s_U) on the inequality rows, and 0 on the defects.

Differentiating [grad_w L; g - s] = 0 at fixed mu along a parameter change dp gives
K [dw; dlam] = -F_p dp, with F_p = [d(grad_w L)/dp; dg/dp]. For a scalar outcome y(w, p),
the adjoint is K m = [grad_w y; 0], and then dy/dp = dy/dp|_explicit - m' F_p. Solves use a
SuperLU factorisation of K with symmetric Jacobi scaling (every large entry of K is on its
diagonal: Sigma_x of the bound-active variables and D of the inactive rows) and two steps of
iterative refinement; adjoint residuals are ~1e-14 relative.

**Parameters.** These are the bank's 2 m samples of kappa, w_left and w_right, the Track's node
values (``perturb.perturbed_track``). The map runs samples -> packed coefficients
(``packed_coeffs``, linear, natural ends; spline.py:50-84) -> the window entries of P
(``KeyProblem.coefficient_index``, window.py:187-203). F_p is built from per-step blocks with
respect to the window entries (``ParamBlocks``), by CasADi AD on an *uncompiled* build. The
compiled externals the solver runs have no p-derivatives.

- Curvature: the frozen step map ``step_functions(spec)[0]`` (window.py:249), expanded to SX.
  dg_t/dW_t = -dPhi/dW_t (8 x K) and d(J'lam)_{z_t}/dW_t = -d2(lam_t'Phi)/dz_t dW_t (10 x K).
  Only the kappa block that the node selects carries a derivative (window.py:156-168).
- Widths: the frozen path function (window.py:319-342) at x_{t+1}. Both terms are kept. One is
  the corridor rows' -b_k(t). The other is the stationarity term on x_{t+1}: d(J'nu)/dW =
  d2(nu'g_path)/dx dW, whose s-entry is the design's -nu_side b_k'(t)/ds. Dropping the second
  term is one of Gate 1's planted bugs.

**Outcomes** (``Readout``): c_min, v_apex and s_b from ``outcomes`` on the optimum's plan (the
pick fixed: Danskin), and the decision's steer a*_dec[0] (a*_0 on A2 keys, a*_1 on anchored
keys; raw sign, recoded by turn direction downstream). grad_w y comes from the jitted JAX
readers (float64 inside ``jax.enable_x64`` blocks; the global flag is never touched). c_min
also depends on the width samples directly (the edge itself moves), and this explicit term is
added to the adjoint term. On an active inside contact the two cancel: the optimum's c_min
field is zero to ~1e-10 and G* = 1. 'c_line', the adjoint term alone (the line's move with the
edge held), is what carries the information there (``Fields.implicit``).

**Inertia** (``inertia``) is (n_free, m, 0), computed by exact congruences and never by
thresholding a condition number:
- inactive inequality rows (Sigma < 1) are condensed into the (1,1) block (one negative
  eigenvalue each); strongly active bound sides (Sigma >= 1) are un-condensed into rows
  [e_i', -1/Sigma] (each removes one), so every entry of size 1/mu sits on a diagonal of -D_C;
- Gould's reduction on the defects with an orthonormal basis Q of their null space (the plan's
  open-loop instability grows the action-parametrised basis to ~1e5 over 800 steps) leaves
  M = [[R, (C Q)'], [C Q, -D_C]], R = Q'W Q, of size n_u + rows, with O(1) entries;
- if R is positive definite (Cholesky), inertia(M) = (n_u, rows, 0) exactly; otherwise
  Bunch-Kaufman LDL' of M (``scipy.linalg.ldl``) counts the pivot signs. Bound rows are
  linearly dependent along every full-throttle stretch (a pedal state held at 1 by a pedal
  command at 1), so M has genuine eigenvalues of the size of D_C. A pivot within
  size * eps * max|pivot| of zero (``n_small``) has a sign at the rounding level: the
  reduced matrix is then recounted by its eigenvalues (``certified_counts``), and the count
  stands only when every |lambda| > size * eps * ||M||_2 (``Inertia.determined``); an
  uncertified count fails ``Inertia.ok`` ('inertia undetermined', excluded as an inertia
  failure).

**Gate 1** (``richardson``, ``gate1``) uses warm re-solves on P(+-h delta) and P(+-h/2 delta)
with the same bounds and window centres. The re-solver (``fd_solver``) is an IPOPT on the key
solver's own frozen functions and options with REFINE_IPOPT's warm start (window.py:405) and
tol tightened from 1e-7 to 1e-10: at 1e-7 a perturbation whose effect on the KKT residual is
below tolerance stops after 0 iterations, which made small fields one-sided. y is read at the
base plan's picks (c_min on the perturbed track arrays, c_line on the base arrays).
FD(h) = (y(h) - y(-h)) / 2h, FD_R = (4 FD(h/2) - FD(h)) / 3, and the Richardson-pair
discrepancy is |FD(h/2) - FD(h)|. The directions (``perturb.gate_directions``, shared with
Gates 2 and 5) are each feature's bumps near the car (s_1 - 10, s_1, s_1 + 10) and at the
corner's landmarks (s_c0 - 20, s_c0, s_apex, s_c1, s_c1 + 20), and both trapezoids; a field
vector is one feature's bumps in one region, or one trapezoid (``direction_group``), so the
in-corner fields (40-300 times the near-car ones) cannot swamp the near-car errors, and every
key, anchored keys included (s_1 is 108-171 m before s_c0 there), has its in-corner fields
checked. err = ||KKT - FD_R|| / max(||FD_R||, 3 ||discrepancy||, ||scale||) per field vector
(``gate1_score``), where scale is c_min's explicit edge term along the same directions (c_min's
true field is zero on an active contact; the edge's own move is its scale) and absent for the
other quantities. A vector that both sides put below 1e-6 (``ZERO_ATOL``) with no scale above
it is a zero field: it scores 0 and is flagged, and the decision pools only the others
(``gates.gate1_decision``). Step sizes per unit-peak direction (``FD_H``): 1e-6 for curvature
and 1e-4 for widths; larger steps cross active-set kinks at the mm scale.

**Gate 4** (``edge_gain``): G* = 1 - dc_min/dDelta along the inside trapezoid shift
(``perturb.edge_shift``). It must lie in [0.98, 1.02].

**Dose-response** (``dose_optimum``, design Section 4): warm continuation in the trapezoid
shift's amplitude, each branch (widening +0.1, +0.3, +1 m; narrowing -0.1, -0.3, -1, -2 m) out
from the re-polished key on the dose solver (``dose_solver``: the key solver's frozen options,
REFINE_IPOPT's warm start, tol 1e-8). A step restarts from the stored key when it does not
converge, takes more than 200 iterations, or breaks J's monotonicity in the amplitude (a
widening relaxes the corridor, so J is non-decreasing in it); ``continuation`` holds the rule.

**Checks recorded per key** (fixed at the tag): ``Fields.max_adjoint_residual`` against
ADJOINT_RTOL = 1e-10 (seen ~1e-14) and the barrier's fallback count (``Fields.barrier_fallback``,
recorded, not thresholded).

Planted bugs (``PLANTS``, Gate 1 must fail them): 'drop_width_mixed' (the stationarity term of
the widths), 'swap_widths' (w_left and w_right exchanged in F_p) and 'flip_lam_g' (L = f -
lam_g'g in H and in F_p's mixed terms; the barrier terms keep the correct sides).
"""
from __future__ import annotations

import dataclasses
import functools
import time
from dataclasses import dataclass, field

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from apexzero.fields import keys as K
from apexzero.fields import outcomes as O
from apexzero.fields import perturb as PT
from apexzero.fields.perturb import Samples
from apexzero.interp import answer_key as ak
from apexzero.interp.stimuli import Landmarks
from apexzero.ocp.window import (
    NA,
    NX,
    REFINE_IPOPT,
    WindowResult,
    WindowSpec,
    _window_centres,
    path_functions,
)
from apexzero.ocp.window import step_functions as _step_functions
from apexzero.tracks.spline import packed_coeffs
from apexzero.tracks.track import Track
from apexzero.vehicle.params import VehicleParams

OUTCOMES = ("c_min", "v_apex", "s_b", "steer")
STRONG_Z = 1e-4            # a side is strongly active for the mu estimate when z >= this
MU_FALLBACK = 1e-11        # IPOPT's mu_min, when no side is strongly active
ROW_SIGMA = 1.0            # inertia partition: barrier curvature above this -> a row in M
REFINE_STEPS = 2           # iterative refinement steps of each KKT solve
PLANTS = ("drop_width_mixed", "swap_widths", "flip_lam_g")
GATE1_MEDIAN = 1e-3        # design Section 8, Gate 1
GATE1_P95 = 1e-2
GATE4_BAND = (0.98, 1.02)  # design Section 8, Gate 4
FD_H = {"kappa": 1e-6, "w_left": 1e-4, "w_right": 1e-4}   # per unit-peak direction
GATE1_OUTCOMES = ("c_min", "c_line", "v_apex", "s_b", "steer")
GATE1_TOL = 1e-10          # IPOPT tol of Gate 1's re-solves (REFINE_IPOPT's own is 1e-7)
ZERO_ATOL = 1e-6           # Gate 1: a field below this on both sides is a zero field
ADJOINT_RTOL = 1e-10       # max relative adjoint residual of a key's fields (seen: ~1e-14)
EIG_CERT = 1.0             # inertia: |lambda| > EIG_CERT * size * eps * ||M||_2 certifies a sign
DOSE_AMPLITUDES = (0.1, 0.3, 1.0, -0.1, -0.3, -1.0, -2.0)   # m (+ widens): design Section 4
DOSE_MAX_ITER = 200        # dose continuation: a warm step above this restarts from the key
DOSE_TOL = 1e-8            # IPOPT tol of the dose solves
DOSE_SOLVER_MAX_ITER = 3000

# ------------------------------------------------------------------------- the coefficient map


def coeffs(f: np.ndarray) -> np.ndarray:
    """Packed natural-spline coefficients of node values f [n] (``packed_coeffs`` of an open
    strip, pad_to = n): length n + 2, linear in f."""
    f = np.asarray(f, float)
    return packed_coeffs(f, False, len(f))


def _tri_solve(rhs: np.ndarray) -> np.ndarray:
    """Solve the interpolation system tridiag(1, 4, 1) x = rhs (coeffs_open_np's band)."""
    m = len(rhs)
    ab = np.zeros((3, m))
    ab[0, 1:] = 1.0
    ab[1, :] = 4.0
    ab[2, :-1] = 1.0
    return sla.solve_banded((1, 1), ab, rhs)


def coeffs_transpose(cbar: np.ndarray, n: int) -> np.ndarray:
    """The transpose of ``coeffs`` for n nodes: cotangents of the n + 2 packed coefficients ->
    cotangents of the node values. ``coeffs_open_np`` (spline.py:50-67) is c_0 = f_0,
    c_{n-1} = f_{n-1}, the inner c = tri^{-1}(6 f[1:-1] - c_0 e_1 - c_{n-1} e_m), and the
    natural ends c_{-1} = 2c_0 - c_1, c_n = 2c_{n-1} - c_{n-2}. The tridiagonal is symmetric."""
    cb = np.asarray(cbar, float)
    if cb.shape != (n + 2,):
        raise ValueError(f"expected {n + 2} coefficient cotangents, got {cb.shape}")
    c = cb[1: n + 1].copy()                       # c_0 .. c_{n-1}
    c[0] += 2.0 * cb[0]
    c[1] -= cb[0]
    c[n - 1] += 2.0 * cb[n + 1]
    c[n - 2] -= cb[n + 1]
    r = _tri_solve(c[1: n - 1])
    out = np.empty(n)
    out[1: n - 1] = 6.0 * r
    out[0] = c[0] - r[0]
    out[n - 1] = c[n - 1] - r[-1]
    return out


# ------------------------------------------------------------------------- the barrier point


@dataclass(frozen=True)
class Barrier:
    """The barrier multipliers and slacks of a primal-dual point (``barrier``). Arrays over w
    [nw] (bounds) and over g [ng] (rows). A missing side has z = 0 and s = inf, and fixed
    variables and equality rows have neither side."""

    mu: float
    n_strong: int             # strongly active sides behind the mu estimate (0: MU_FALLBACK)
    zl: np.ndarray
    zu: np.ndarray
    sl: np.ndarray
    su: np.ndarray
    gzl: np.ndarray
    gzu: np.ndarray
    gsl: np.ndarray
    gsu: np.ndarray
    sigma_x: np.ndarray       # [nw] z_L/s_L + z_U/s_U (0 on fixed variables)
    sigma_g: np.ndarray       # [ng] the same on the inequality rows (0 on equality rows)
    fallback: int             # active sides whose measured slack was <= 0 (s = mu/z used)

    @property
    def d_g(self) -> np.ndarray:
        """[ng] D of the (2,2) block -D: 1/Sigma_g on inequality rows, 0 on equality rows."""
        out = np.zeros_like(self.sigma_g)
        nz = self.sigma_g > 0
        out[nz] = 1.0 / self.sigma_g[nz]
        return out


def _slacks(v: np.ndarray, lo: np.ndarray, hi: np.ndarray, use: np.ndarray):
    """(s_L, s_U): v - lo and hi - v where the bound is finite and ``use`` holds, inf
    elsewhere."""
    sl = np.where(use & np.isfinite(lo), v - lo, np.inf)
    su = np.where(use & np.isfinite(hi), hi - v, np.inf)
    return sl, su


def _strong_products(lam: np.ndarray, sl: np.ndarray, su: np.ndarray) -> np.ndarray:
    """s z of the strongly active sides (z = |lam| on the side lam points to, s > 0)."""
    up = (lam >= STRONG_Z) & np.isfinite(su) & (su > 0)
    lo = (-lam >= STRONG_Z) & np.isfinite(sl) & (sl > 0)
    return np.concatenate([su[up] * lam[up], sl[lo] * -lam[lo]])


def _sides(lam: np.ndarray, sl: np.ndarray, su: np.ndarray, mu: float):
    """(z_L, z_U, s_L, s_U, fallbacks) with z_U - z_L = lam. The hidden side has z = mu/s, and
    the side lam points to has |lam| plus the hidden one. An active side whose measured slack
    is not positive takes s = mu/z."""
    has_l, has_u = np.isfinite(sl), np.isfinite(su)
    hid_l = np.where(has_l & (sl > 0), mu / np.where(sl > 0, sl, 1.0), 0.0)
    hid_u = np.where(has_u & (su > 0), mu / np.where(su > 0, su, 1.0), 0.0)
    up = (lam > 0) & has_u
    lo = (lam < 0) & has_l
    zu = np.where(up, lam + hid_l, hid_u)
    zl = np.where(lo, -lam + hid_u, hid_l)
    bad_l = has_l & (sl <= 0) & (zl > 0)
    bad_u = has_u & (su <= 0) & (zu > 0)
    sl = np.where(bad_l, mu / np.where(zl > 0, zl, 1.0), sl)
    su = np.where(bad_u, mu / np.where(zu > 0, zu, 1.0), su)
    return zl, zu, sl, su, int(bad_l.sum() + bad_u.sum())


def _sigma(z: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(s) & (z > 0), z / np.where(np.isfinite(s) & (s > 0), s, 1.0), 0.0)


def barrier(prob: K.KeyProblem, mu: float | None = None) -> Barrier:
    """The barrier point of ``prob``'s primal-dual point (see the module docstring)."""
    T = prob.T
    free = prob.free
    sl, su = _slacks(prob.w, prob.lbx, prob.ubx, free)
    _, g = K.constraints(prob)
    ineq = prob.lbg < prob.ubg
    gsl, gsu = _slacks(g, prob.lbg, prob.ubg, ineq)
    if mu is None:
        sz = np.concatenate([_strong_products(prob.lam_x, sl, su),
                             _strong_products(prob.lam_g, gsl, gsu)])
        n_strong = len(sz)
        mu = float(np.median(sz)) if n_strong else MU_FALLBACK
    else:
        n_strong = 0
    zl, zu, sl, su, fb = _sides(np.where(free, prob.lam_x, 0.0), sl, su, mu)
    gzl, gzu, gsl, gsu, gfb = _sides(np.where(ineq, prob.lam_g, 0.0), gsl, gsu, mu)
    sigma_x = np.where(free, _sigma(zl, sl) + _sigma(zu, su), 0.0)
    sigma_g = np.where(ineq, _sigma(gzl, gsl) + _sigma(gzu, gsu), 0.0)
    if np.any(ineq[: NX * T]):
        raise ValueError("defect rows must be equalities")
    return Barrier(mu=float(mu), n_strong=n_strong, zl=zl, zu=zu, sl=sl, su=su, gzl=gzl, gzu=gzu,
                   gsl=gsl, gsu=gsu, sigma_x=sigma_x, sigma_g=sigma_g, fallback=fb + gfb)


# ------------------------------------------------------------------------- the KKT matrix


def _check_plant(plant: str | None) -> None:
    if plant is not None and plant not in PLANTS:
        raise ValueError(f"unknown plant {plant!r}; expected one of {PLANTS}")


def _lagrangian_lam(prob: K.KeyProblem, plant: str | None) -> np.ndarray:
    return -prob.lam_g if plant == "flip_lam_g" else prob.lam_g


def hessian(prob: K.KeyProblem, lam_g: np.ndarray | None = None) -> sp.csc_matrix:
    """H = Hess_w L with lam_f = 1 [nw, nw], the full symmetric matrix from hess_lag's upper
    triangle U: U + U' - diag U."""
    lam = prob.lam_g if lam_g is None else lam_g
    U = prob.solver.funcs["hess_lag"](prob.w, prob.P, 1.0, lam).sparse()
    U = sp.csc_matrix(U)
    return (U + U.T - sp.diags(U.diagonal())).tocsc()


def jacobian(prob: K.KeyProblem) -> sp.csc_matrix:
    """J = dg/dw [ng, nw]."""
    _, jac = prob.solver.funcs["jac_g"](prob.w, prob.P)
    return sp.csc_matrix(jac.sparse())


@dataclass
class KKT:
    """The condensed barrier KKT system at a primal-dual point, factorised."""

    prob: K.KeyProblem
    barrier: Barrier
    free: np.ndarray          # indices of the free variables (lbx < ubx)
    H: sp.csc_matrix          # [nw, nw]
    J: sp.csc_matrix          # [ng, nw]
    matrix: sp.csc_matrix     # [nf + ng]^2: [[H + Sigma_x, J'], [J, -D]] (free variables)
    scale: np.ndarray         # Jacobi scaling 1/sqrt(max(|K_ii|, 1))
    lu: spla.SuperLU          # of diag(scale) K diag(scale)
    plant: str | None = None
    seconds: dict = field(default_factory=dict)

    @property
    def n_free(self) -> int:
        return len(self.free)

    @property
    def m(self) -> int:
        return self.J.shape[0]

    def solve(self, rhs: np.ndarray) -> tuple[np.ndarray, float]:
        """K u = rhs [nf + ng, k] (or [nf + ng]), with REFINE_STEPS steps of iterative
        refinement; returns (u, max relative residual ||K u - rhs|| / ||rhs||)."""
        b = np.asarray(rhs, float)
        one = b.ndim == 1
        b = b[:, None] if one else b
        d = self.scale[:, None]
        u = d * self.lu.solve(d * b)
        for _ in range(REFINE_STEPS):
            u = u + d * self.lu.solve(d * (b - self.matrix @ u))
        res = np.linalg.norm(self.matrix @ u - b, axis=0) / np.maximum(np.linalg.norm(b, axis=0),
                                                                       1e-300)
        return (u[:, 0] if one else u), float(res.max())

    def lift(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(dw [nw] with 0 on fixed variables, dlam [ng]) from a solution vector."""
        nw = self.prob.solver.nw
        dw = np.zeros(nw)
        dw[self.free] = u[: self.n_free]
        return dw, u[self.n_free:]


def kkt_system(prob: K.KeyProblem, plant: str | None = None, mu: float | None = None) -> KKT:
    """Assemble and factorise the condensed barrier KKT matrix at ``prob``'s point. ``plant``
    'flip_lam_g' evaluates H with -lam_g (Gate 1's planted bug)."""
    _check_plant(plant)
    t0 = time.perf_counter()
    bar = barrier(prob, mu)
    H = hessian(prob, _lagrangian_lam(prob, plant))
    J = jacobian(prob)
    free = np.flatnonzero(prob.free)
    Hf = H[free][:, free] + sp.diags(bar.sigma_x[free])
    Jf = J[:, free]
    mat = sp.bmat([[Hf, Jf.T], [Jf, -sp.diags(bar.d_g)]], format="csc")
    t1 = time.perf_counter()
    scale = 1.0 / np.sqrt(np.maximum(np.abs(mat.diagonal()), 1.0))
    S = sp.diags(scale)
    lu = spla.splu((S @ mat @ S).tocsc())
    t2 = time.perf_counter()
    return KKT(prob=prob, barrier=bar, free=free, H=H, J=J, matrix=mat, scale=scale, lu=lu,
               plant=plant, seconds={"assemble": t1 - t0, "factorise": t2 - t1})


# ------------------------------------------------------------------------- inertia


@dataclass(frozen=True)
class Inertia:
    """Inertia of the condensed KKT matrix (numbers of positive, negative and zero
    eigenvalues) by exact congruences. ``ok`` when it is (n_free, m, 0) *and* certified
    (``determined``): the Cholesky path always is; on the LDL path a pivot whose sign is at
    the rounding level (``n_small`` > 0) sends the reduced matrix to a dense eigenvalue
    count, which certifies the signs when every |lambda| exceeds EIG_CERT size eps ||M||_2
    (no element growth there). An uncertified count is 'inertia undetermined', an exclusion
    under the inertia reason."""

    n_pos: int
    n_neg: int
    n_zero: int
    expected: tuple[int, int, int]
    method: str               # 'cholesky' (R positive definite), 'ldl' (Bunch-Kaufman of M),
                              # or 'ldl+eig' (LDL had rounding-level pivots: eigvalsh recount)
    reduced_size: int         # size of the reduced matrix M = n_u + rows
    n_active_rows: int        # inequality rows kept as rows of M
    n_bound_rows: int         # bound sides un-condensed into rows of M
    r_min_eig: float          # smallest eigenvalue of R = Q'W Q (the u-block of M)
    n_small: int              # 'ldl': pivots within size * eps * max|pivot| of zero
    seconds: float
    determined: bool = True   # the signs are certified (see the class docstring)
    eig_min_abs: float = float("nan")    # 'ldl+eig': min |lambda(M)| (its certificate)

    @property
    def ok(self) -> bool:
        return bool(self.determined
                    and (self.n_pos, self.n_neg, self.n_zero) == self.expected)


def _dynamics_blocks(J: sp.csc_matrix, T: int) -> tuple[np.ndarray, np.ndarray]:
    """A_t = dPhi_t/dx_t [T, 8, 8] and B_t = dPhi_t/da_t [T, 8, 2] from the defect rows of J
    (g_t = x_{t+1} - Phi_t, so J holds -A_t and -B_t)."""
    C = J[: NX * T].tocoo()
    t = C.row // NX
    A = np.zeros((T, NX, NX))
    B = np.zeros((T, NX, NA))
    ia = NX * (T + 1)
    on_x = (C.col < ia) & (C.col // NX == t)
    A[t[on_x], C.row[on_x] % NX, C.col[on_x] % NX] = -C.data[on_x]
    ca_ = C.col - ia
    on_a = (C.col >= ia) & (ca_ // NA == t)
    B[t[on_a], C.row[on_a] % NX, ca_[on_a] % NA] = -C.data[on_a]
    return A, B


def null_space(kkt: KKT) -> np.ndarray:
    """Z [n_free, n_u]: a basis of the defect Jacobian's null space over the free variables,
    parametrised by the free actions (dx_0 = 0, dx_{t+1} = A_t dx_t + B_t da_t). Requires
    x_1..x_T free and x_0 fixed (every key). Its entries grow with the open-loop instability
    of the plan (to ~1e5 over 800 steps), so ``inertia`` orthonormalises it."""
    prob, T = kkt.prob, kkt.prob.T
    nw = prob.solver.nw
    ia = NX * (T + 1)
    pos = np.full(nw, -1)
    pos[kkt.free] = np.arange(kkt.n_free)
    if np.any(pos[NX: ia] < 0) or np.any(pos[:NX] >= 0):
        raise ValueError("expected x_0 fixed and x_1..x_T free")
    acts = kkt.free[kkt.free >= ia]
    n_u = len(acts)
    col = {int(i): k for k, i in enumerate(acts)}
    A, B = _dynamics_blocks(kkt.J, T)
    Z = np.zeros((kkt.n_free, n_u))
    X = np.zeros((NX, n_u))
    for t in range(T):
        X = A[t] @ X
        for j in range(NA):
            k = col.get(ia + NA * t + j)
            if k is not None:
                X[:, k] += B[t][:, j]
        Z[pos[NX * (t + 1): NX * (t + 2)]] = X
    Z[pos[acts], np.arange(n_u)] = 1.0
    return Z


def _ldl_counts(M: np.ndarray) -> tuple[int, int, int, int]:
    """(n_pos, n_neg, n_zero, n_small) of a symmetric matrix from the signs of its
    Bunch-Kaufman LDL' pivots (Sylvester); n_zero counts exact zeros, n_small the pivots
    within size * eps * max|pivot| of zero (their sign is at the rounding level)."""
    _, D, _ = sla.ldl(M, lower=True, hermitian=True)
    ev = []
    i, n = 0, len(D)
    while i < n:
        if i + 1 < n and D[i + 1, i] != 0.0:
            ev.extend(np.linalg.eigvalsh(D[i: i + 2, i: i + 2]))
            i += 2
        else:
            ev.append(D[i, i])
            i += 1
    ev = np.asarray(ev)
    big = float(np.max(np.abs(ev))) if n else 0.0
    small = np.abs(ev) <= n * np.finfo(float).eps * big
    return int(np.sum(ev > 0)), int(np.sum(ev < 0)), int(np.sum(ev == 0)), int(small.sum())


def certified_counts(M: np.ndarray) -> tuple[int, int, int, int, bool, float, str]:
    """(n_pos, n_neg, n_zero, n_small, determined, min |lambda|, method) of a symmetric
    matrix: the LDL' pivot signs (``_ldl_counts``), certified. With no rounding-level pivot
    the count stands ('ldl'); otherwise the matrix's eigenvalues are counted instead
    (``scipy.linalg.eigvalsh``, backward stable with no element growth), and the count is
    determined only when every |lambda| exceeds EIG_CERT size eps ||M||_2 ('ldl+eig')."""
    pos, neg, zero, n_small = _ldl_counts(M)
    if n_small == 0:
        return pos, neg, zero, 0, True, float("nan"), "ldl"
    ev = sla.eigvalsh(M)
    norm = float(np.max(np.abs(ev))) if len(ev) else 0.0
    floor = EIG_CERT * len(ev) * np.finfo(float).eps * norm
    mn = float(np.min(np.abs(ev))) if len(ev) else float("inf")
    return (int(np.sum(ev > 0)), int(np.sum(ev < 0)), int(np.sum(ev == 0)), n_small,
            bool(mn > floor), mn, "ldl+eig")


def inertia(kkt: KKT) -> Inertia:
    """Inertia of ``kkt.matrix`` by exact congruences (module docstring):
    1. the inactive inequality rows (D > 1) are condensed into the (1,1) block, +1 negative
       each; the strongly active bound sides (Sigma >= 1) are un-condensed into rows
       [e_i', -1/Sigma], -1 negative each;
    2. Gould's reduction on the defects with an orthonormal basis Q of their null space
       (``null_space``, QR): (8T, 8T, 0) plus the inertia of M = [[R, (C Q)'], [C Q, -D_C]],
       R = Q'W Q;
    3. M: when R is positive definite (Cholesky), inertia(M) = (n_u, rows, 0) exactly, since
       -D_C - (C Q) R^{-1} (C Q)' is then negative definite; otherwise Bunch-Kaufman LDL' of M.
    Every barrier entry of size 1/mu sits in -D_C as a tiny diagonal, so R and C Q are O(1).
    Bound rows can be linearly dependent (a pedal state held at 1 by a pedal command at 1:
    LICQ fails along every full-throttle stretch), which makes M's own small eigenvalues of
    the size of D_C; step 3's Cholesky path never pivots on them."""
    t0 = time.perf_counter()
    bar, T = kkt.barrier, kkt.prob.T
    f = kkt.free
    Hf = kkt.H[f][:, f]
    Jf = kkt.J[:, f]
    n_e = NX * T
    sig_g = bar.sigma_g
    rows = np.arange(len(sig_g))
    act = rows[(rows >= n_e) & (sig_g >= ROW_SIGMA)]
    cond = rows[(rows >= n_e) & (sig_g > 0) & (sig_g < ROW_SIGMA)]
    sxl = _sigma(bar.zl, bar.sl)[f]
    sxu = _sigma(bar.zu, bar.su)[f]
    small = np.where(sxl < ROW_SIGMA, sxl, 0.0) + np.where(sxu < ROW_SIGMA, sxu, 0.0)
    b_idx = np.concatenate([np.flatnonzero(sxl >= ROW_SIGMA), np.flatnonzero(sxu >= ROW_SIGMA)])
    b_sig = np.concatenate([sxl[sxl >= ROW_SIGMA], sxu[sxu >= ROW_SIGMA]])
    Jc = Jf[cond]
    W1 = (Hf + sp.diags(small) + Jc.T @ sp.diags(sig_g[cond]) @ Jc).tocsr()
    Q, _ = np.linalg.qr(null_space(kkt))
    n_u = Q.shape[1]
    C = sp.vstack([Jf[act], sp.csr_matrix((np.ones(len(b_idx)), (np.arange(len(b_idx)), b_idx)),
                                          shape=(len(b_idx), len(f)))]).tocsr()
    CQ = C @ Q
    d_c = np.concatenate([1.0 / sig_g[act], 1.0 / b_sig])
    R = Q.T @ (W1 @ Q)
    R = 0.5 * (R + R.T)
    n_rows = len(d_c)
    r_min = float(sla.eigvalsh(R, subset_by_index=[0, 0])[0]) if n_u else np.inf
    determined, eig_min = True, float("nan")
    try:
        sla.cholesky(R, lower=True)
        pos, neg, zero, n_small, method = n_u, n_rows, 0, 0, "cholesky"
    except np.linalg.LinAlgError:
        M = np.block([[R, CQ.T], [CQ, -np.diag(d_c)]])
        pos, neg, zero, n_small, determined, eig_min, method = certified_counts(M)
    return Inertia(n_pos=n_e + pos, n_neg=n_e + len(cond) + neg - len(b_idx), n_zero=zero,
                   expected=(kkt.n_free, kkt.m, 0), method=method, reduced_size=n_u + n_rows,
                   n_active_rows=len(act), n_bound_rows=len(b_idx), r_min_eig=r_min,
                   n_small=n_small, seconds=time.perf_counter() - t0, determined=determined,
                   eig_min_abs=eig_min)


# ------------------------------------------------------------------------- parameter derivatives


@functools.cache
def _fp_functions(spec: WindowSpec) -> tuple[ca.Function, ca.Function]:
    """Per-step CasADi functions of the window entries' derivatives, by AD on the frozen
    (uncompiled) step and path functions:
    dyn(x, a, W, j0, q, ctx, lam) -> (dPhi/dW [8, K], d2(lam'Phi)/d(x, a) dW [10, K]);
    path(x, Wl, Wr, j0, q, ctx, amax, nu) -> (d g_corridor / d[Wl; Wr] [2, 2K],
    d2(nu'g_path)/dx d[Wl; Wr] [8, 2K])."""
    Kc = spec.n_coef
    phi = _step_functions(spec)[0].expand()
    path = path_functions(spec)[0]
    x = ca.SX.sym("x", NX)
    a = ca.SX.sym("a", NA)
    W = ca.SX.sym("W", Kc)
    j0 = ca.SX.sym("j0")
    q = ca.SX.sym("q", 4)
    ctx = ca.SX.sym("ctx", 6)
    lam = ca.SX.sym("lam", NX)
    xn = phi(x, a, W, j0, q, ctx)
    z = ca.vertcat(x, a)
    dyn = ca.Function("fp_dyn", [x, a, W, j0, q, ctx, lam],
                      [ca.jacobian(xn, W), ca.jacobian(ca.gradient(ca.dot(lam, xn), z), W)],
                      {"cse": True})
    Wl = ca.SX.sym("Wl", Kc)
    Wr = ca.SX.sym("Wr", Kc)
    amax = ca.SX.sym("alpha_max")
    nu = ca.SX.sym("nu", 4)
    gp = path(x, Wl, Wr, j0, q, ctx, amax)
    Wlr = ca.vertcat(Wl, Wr)
    pth = ca.Function("fp_path", [x, Wl, Wr, j0, q, ctx, amax, nu],
                      [ca.jacobian(gp[0:2], Wlr), ca.jacobian(ca.gradient(ca.dot(nu, gp), x), Wlr)],
                      {"cse": True})
    return dyn, pth


@dataclass
class ParamBlocks:
    """F_p as per-step blocks with respect to the window entries of P (rows of F_p: the
    stationarity rows of w, then the rows of g), and the map to the samples."""

    T: int
    n: int                    # strip nodes (samples per feature)
    nw: int
    ng: int
    cidx: np.ndarray          # [T, K] packed-coefficient index of each window entry
    dyn_g: np.ndarray         # [T, 8, K]   d g_defect_t / d Wk_t
    dyn_w: np.ndarray         # [T, 10, K]  d (J'lam)_{x_t, a_t} / d Wk_t
    path_g: np.ndarray        # [T, 2, 2K]  d g_corridor_t / d [Wl_t; Wr_t]
    path_w: np.ndarray        # [T, 8, 2K]  d (J'lam)_{x_{t+1}} / d [Wl_t; Wr_t]
    plant: str | None = None
    seconds: float = 0.0

    def _layout(self):
        T = self.T
        ix = NX * np.arange(T)[:, None] + np.arange(NX)[None, :]          # x_t
        ia = NX * (T + 1) + NA * np.arange(T)[:, None] + np.arange(NA)[None, :]
        return ix, ia

    def _windows(self, delta: Samples) -> dict[str, np.ndarray]:
        out = {f: coeffs(np.asarray(getattr(delta, f), float))[self.cidx] for f in PT.FEATURES}
        if self.plant == "swap_widths":
            out["w_left"], out["w_right"] = out["w_right"], out["w_left"]
        return out

    def apply(self, delta: Samples) -> tuple[np.ndarray, np.ndarray]:
        """F_p delta: (stationarity rows [nw], constraint rows [ng]) for a sample-space
        perturbation."""
        T = self.T
        W = self._windows(delta)
        ix, ia = self._layout()
        rw, rg = np.zeros(self.nw), np.zeros(self.ng)
        dk = np.einsum("trk,tk->tr", self.dyn_w, W["kappa"])
        np.add.at(rw, ix, dk[:, :NX])
        np.add.at(rw, ia, dk[:, NX:])
        rg[: NX * T] += np.einsum("trk,tk->tr", self.dyn_g, W["kappa"]).reshape(-1)
        Wlr = np.concatenate([W["w_left"], W["w_right"]], axis=1)
        np.add.at(rw, ix + NX, np.einsum("trk,tk->tr", self.path_w, Wlr))
        rg[NX * T: NX * T + 2 * T] += np.einsum("trk,tk->tr", self.path_g, Wlr).reshape(-1)
        return rw, rg

    def transpose(self, mu_w: np.ndarray, mu_g: np.ndarray) -> Samples:
        """mu' F_p as sample cotangents (one [n] array per feature)."""
        T = self.T
        ix, ia = self._layout()
        mz = np.concatenate([mu_w[ix], mu_w[ia]], axis=1)                  # [T, 10]
        wk = (np.einsum("trk,tr->tk", self.dyn_w, mz)
              + np.einsum("trk,tr->tk", self.dyn_g, mu_g[: NX * T].reshape(T, NX)))
        corr = mu_g[NX * T: NX * T + 2 * T].reshape(T, 2)
        wlr = (np.einsum("trk,tr->tk", self.path_w, mu_w[ix + NX])
               + np.einsum("trk,tr->tk", self.path_g, corr))
        Kc = self.cidx.shape[1]
        win = {"kappa": wk, "w_left": wlr[:, :Kc], "w_right": wlr[:, Kc:]}
        if self.plant == "swap_widths":
            win["w_left"], win["w_right"] = win["w_right"], win["w_left"]
        out = {}
        for f in PT.FEATURES:
            cb = np.bincount(self.cidx.ravel(), weights=win[f].ravel(), minlength=self.n + 2)
            out[f] = coeffs_transpose(cb, self.n)
        return Samples(**out)


def param_blocks(prob: K.KeyProblem, plant: str | None = None) -> ParamBlocks:
    """F_p's per-step blocks at ``prob``'s point (CasADi AD on the uncompiled build)."""
    _check_plant(plant)
    t0 = time.perf_counter()
    T, spec = prob.T, prob.spec
    Kc = spec.n_coef
    if prob.solver.scaled:
        raise ValueError("step 1's keys are unscaled; a scaled solver's P has offsets")
    if prob.track.closed:
        raise ValueError("sample-space fields are defined on open strips")
    sl = K.parameter_slices(T, Kc)
    P = prob.P
    Pk, Pl, Pr = (P[sl[k]].reshape(T, Kc).T for k in ("kappa", "w_left", "w_right"))
    J0 = P[sl["j0"]][None, :]
    qT = np.tile(P[sl["q"]][:, None], (1, T))
    cT = np.tile(P[sl["ctx"]][:, None], (1, T))
    aT = np.tile(P[sl["alpha_max"]][:, None], (1, T))
    lam = _lagrangian_lam(prob, plant)
    lam_dyn = lam[: NX * T].reshape(T, NX).T
    nu = np.vstack([lam[NX * T: 10 * T].reshape(T, 2).T, lam[10 * T:].reshape(T, 2).T])
    dyn, pth = _fp_functions(dataclasses.replace(spec, steps=1))   # per step: T-independent
    x, a = prob.x, prob.a
    jw, hw = dyn.map(T)(x[:T].T, a.T, Pk, J0, qT, cT, lam_dyn)
    jp, hp = pth.map(T)(x[1:].T, Pl, Pr, J0, qT, cT, aT, nu)

    def blocks(m, rows):
        return np.asarray(m).reshape(rows, T, -1).transpose(1, 0, 2)

    path_w = blocks(hp, NX)
    if plant == "drop_width_mixed":
        path_w = np.zeros_like(path_w)
    n = int(prob.track.n_points)
    return ParamBlocks(T=T, n=n, nw=prob.solver.nw, ng=prob.solver.ng,
                       cidx=prob.coefficient_index(), dyn_g=-blocks(jw, NX),
                       dyn_w=-blocks(hw, NX + NA), path_g=blocks(jp, 2), path_w=path_w,
                       plant=plant, seconds=time.perf_counter() - t0)


# ------------------------------------------------------------------------- outcomes


@dataclass(frozen=True)
class Readout:
    """One scalar outcome of the optimum's plan: its value, grad_w y [nw] and the explicit
    sample derivative (c_min's widths; zero otherwise)."""

    name: str
    valid: bool
    reason: str
    value: float
    grad_w: np.ndarray | None
    grad_p: Samples | None
    plan: O.Plan | None = None


def _y(name: str, x, a, wl, wr, ta, sigma, pick, dec, p: VehicleParams):
    """Outcome ``name`` of states x [T+1, 8] and actions a [T, 2] at a fixed pick, with the
    width coefficients wl, wr in the track arrays ta (JAX)."""
    if name == "c_min":
        return O.c_min(x, ta._replace(w_left_c=wl, w_right_c=wr), sigma, pick)
    if name == "v_apex":
        return O.v_apex(x, pick)
    if name == "s_b":
        return O.s_b(x, pick, p)
    if name == "steer":
        return a[dec, 0]
    raise ValueError(f"unknown outcome {name!r}; expected one of {OUTCOMES}")


@functools.partial(jax.jit, static_argnums=(0, 9))
def _value_and_grad(name, x, a, wl, wr, ta, sigma, pick, dec, p):
    return jax.value_and_grad(lambda *v: _y(name, *v[:4], ta, sigma, pick, dec, p),
                              argnums=(0, 1, 2, 3))(x, a, wl, wr)


@functools.partial(jax.jit, static_argnums=(0, 9))
def _value(name, x, a, wl, wr, ta, sigma, pick, dec, p):
    return _y(name, x, a, wl, wr, ta, sigma, pick, dec, p)


def _pick_arrays(name: str, plan: O.Plan | None):
    """The plan's pick as JAX arrays (traced, so one compilation serves every key)."""
    if name == "steer":
        return O.Pick(jnp.int32(0), jnp.int32(0), jnp.float64(0.0), jnp.bool_(False))
    pk = plan.pick
    if isinstance(pk, O.OnsetPick):
        return O.OnsetPick(jnp.int32(pk.k), jnp.float64(pk.thr))
    return O.Pick(jnp.int32(pk.i0), jnp.int32(pk.i1), jnp.float64(pk.s_star), jnp.bool_(pk.interp))


def _jax_args(x: np.ndarray, a: np.ndarray, ta):
    tj = jax.tree.map(jnp.asarray, ta)
    return jnp.asarray(x, jnp.float64), jnp.asarray(a, jnp.float64), tj.w_left_c, tj.w_right_c, tj


def outcome_plans(prob: K.KeyProblem, lm: Landmarks, sigma: int) -> dict[str, O.Plan]:
    """The three outcome plans on the problem's planned states (picks fixed here)."""
    ta = K.key_arrays(prob.track)
    return O.plans(K.trajectory(prob), lm, sigma, ta, None, O.drive_params(prob.ctx))


def readouts(prob: K.KeyProblem, lm: Landmarks, sigma: int, decision: int,
             plans: dict[str, O.Plan] | None = None) -> dict[str, Readout]:
    """c_min, v_apex, s_b and the decision's steer with their gradients at the problem's point
    (float64 JAX inside ``jax.enable_x64``; jitted with the picks traced)."""
    T, n = prob.T, int(prob.track.n_points)
    plans = outcome_plans(prob, lm, sigma) if plans is None else plans
    p = O.drive_params(prob.ctx)
    ia = NX * (T + 1) + NA * decision
    out = {}
    for name in OUTCOMES:
        plan = plans.get(name)
        if name == "steer":
            if not prob.free[ia]:
                out[name] = Readout(name, False, "decision steer is fixed", np.nan, None, None)
                continue
        elif plan is None or not plan.valid:
            out[name] = Readout(name, False, plan.reason if plan else "no plan", np.nan, None,
                                None, plan)
            continue
        with jax.enable_x64(True):
            args = _jax_args(prob.x, prob.a, K.key_arrays(prob.track))
            val, grads = _value_and_grad(name, *args, jnp.int32(sigma), _pick_arrays(name, plan),
                                         jnp.int32(decision), p)
            val = float(val)
            gx, ga, gl, gr = (np.asarray(g, float) for g in grads)
        gw = np.concatenate([gx.reshape(-1), ga.reshape(-1)])
        gp = Samples(np.zeros(n), coeffs_transpose(gl, n), coeffs_transpose(gr, n))
        out[name] = Readout(name, True, "", val, gw, gp, plan)
    return out


def outcome_values(prob: K.KeyProblem, reads: dict[str, Readout], sigma: int, decision: int,
                   w: np.ndarray | None = None, delta: Samples | None = None) -> dict[str, float]:
    """The valid outcomes at a point w (default: the problem's) on the track plus ``delta``,
    read at the base picks: c_min on the perturbed track's arrays, and 'c_line' (c_min's line
    part) on the base arrays."""
    T = prob.T
    w = prob.w if w is None else np.asarray(w, float)
    x = w[: NX * (T + 1)].reshape(T + 1, NX)
    a = w[NX * (T + 1):].reshape(T, NA)
    p = O.drive_params(prob.ctx)
    tracks = {"base": K.key_arrays(prob.track),
              "pert": K.key_arrays(PT.perturbed_track(prob.track, delta))}
    out = {}
    with jax.enable_x64(True):
        args = {k: _jax_args(x, a, ta) for k, ta in tracks.items()}
        for name, r in reads.items():
            if not r.valid:
                continue
            for label, which in ((name, "pert"), ("c_line", "base")):
                if label == "c_line" and name != "c_min":
                    continue
                out[label] = float(_value(name, *args[which], jnp.int32(sigma),
                                          _pick_arrays(name, r.plan), jnp.int32(decision), p))
    return out


# ------------------------------------------------------------------------- fields


@dataclass
class Fields:
    """The optimum's impulse responses at one key: dy/d(sample) per outcome and feature (the
    adjoint term plus c_min's explicit width term); NaN arrays for invalid outcomes."""

    key_id: str
    decision: int
    reads: dict[str, Readout]
    impulse: dict[str, Samples]       # total: adjoint (implicit) + explicit
    explicit: dict[str, Samples]      # the outcome's own sample dependence (c_min's edge)
    adjoint_residual: dict[str, float]
    mu: float
    inertia: Inertia | None
    plant: str | None = None
    seconds: dict = field(default_factory=dict)
    barrier_fallback: int = 0         # active sides whose measured slack was <= 0 (recorded)

    def valid(self, name: str) -> bool:
        return self.reads["c_min" if name == "c_line" else name].valid

    @property
    def max_adjoint_residual(self) -> float:
        """The largest finite relative adjoint residual over the outcomes (0 if none)."""
        r = [v for v in self.adjoint_residual.values() if np.isfinite(v)]
        return float(max(r)) if r else 0.0

    @property
    def adjoint_ok(self) -> bool:
        """Every adjoint solve within ADJOINT_RTOL (a gate and stage check, fixed at the tag)."""
        return self.max_adjoint_residual <= ADJOINT_RTOL

    def implicit(self, name: str) -> Samples:
        """The adjoint term alone: the plan's response (for c_min, the line's part 'c_line')."""
        return Samples(*(np.asarray(a) - np.asarray(b) for a, b in zip(
            self.impulse[name], self.explicit[name], strict=True)))

    def along(self, name: str, delta: Samples) -> float:
        """dy/d(amplitude) along a sample-space direction (sum over features and samples).
        'c_line' is c_min's implicit part: the line's move with the edge held."""
        imp = self.implicit("c_min") if name == "c_line" else self.impulse[name]
        return float(sum(np.dot(np.asarray(getattr(imp, f)), np.asarray(getattr(delta, f), float))
                         for f in PT.FEATURES))


def adjoint(kkt: KKT, blocks: ParamBlocks, grad_w: np.ndarray) -> tuple[Samples, float]:
    """-m' F_p for K m = [grad_w y (free entries); 0]: the adjoint (implicit) term of dy/dp."""
    rhs = np.concatenate([grad_w[kkt.free], np.zeros(kkt.m)])
    u, res = kkt.solve(rhs)
    mw, mg = kkt.lift(u)
    s = blocks.transpose(mw, mg)
    return Samples(*(-np.asarray(v) for v in s)), res


def forward(kkt: KKT, blocks: ParamBlocks, delta: Samples) -> tuple[np.ndarray, np.ndarray, float]:
    """(dw [nw], dlam [ng], residual): the first-order change of the primal-dual point along a
    sample-space perturbation, K [dw; dlam] = -F_p delta."""
    rw, rg = blocks.apply(delta)
    u, res = kkt.solve(-np.concatenate([rw[kkt.free], rg]))
    dw, dl = kkt.lift(u)
    return dw, dl, res


def fields(prob: K.KeyProblem, lm: Landmarks, sigma: int, decision: int,
           plant: str | None = None, check_inertia: bool = True,
           plans: dict[str, O.Plan] | None = None) -> Fields:
    """The optimum's impulse responses for every outcome at a (re-polished) key."""
    t0 = time.perf_counter()
    kkt = kkt_system(prob, plant)
    t1 = time.perf_counter()
    blocks = param_blocks(prob, plant)
    t2 = time.perf_counter()
    reads = readouts(prob, lm, sigma, decision, plans)
    t3 = time.perf_counter()
    n = blocks.n
    nan = Samples(*(np.full(n, np.nan) for _ in PT.FEATURES))
    impulse, explicit, res = {}, {}, {}
    for name, r in reads.items():
        if not r.valid:
            impulse[name], explicit[name], res[name] = nan, nan, np.nan
            continue
        imp, res[name] = adjoint(kkt, blocks, r.grad_w)
        explicit[name] = r.grad_p
        impulse[name] = Samples(*(np.asarray(a) + np.asarray(b)
                                  for a, b in zip(imp, r.grad_p, strict=True)))
    t4 = time.perf_counter()
    inr = inertia(kkt) if check_inertia else None
    t5 = time.perf_counter()
    return Fields(key_id=str(prob.rec.key_id), decision=decision, reads=reads, impulse=impulse,
                  explicit=explicit, adjoint_residual=res, mu=kkt.barrier.mu, inertia=inr,
                  plant=plant,
                  seconds={"kkt": t1 - t0, "param_blocks": t2 - t1, "readouts": t3 - t2,
                           "adjoints": t4 - t3, "inertia": t5 - t4, "total": t5 - t0},
                  barrier_fallback=int(kkt.barrier.fallback))


def key_fields(prob: K.KeyProblem, plant: str | None = None, check_inertia: bool = True,
               bank=None) -> Fields:
    """``fields`` at a stored key: landmarks and turn direction from its stimulus, the
    decision step from its kind (``keys.decision_step``)."""
    if bank is None:
        bank, _ = K.design()
    stim = bank.stimuli[prob.rec.stim_id]
    return fields(prob, stim.landmarks, int(stim.direction), K.decision_step(prob.rec), plant,
                  check_inertia)


# ------------------------------------------------------------------------- Gate 4


def edge_gain(fl: Fields, lm: Landmarks, sigma: int, side: str = "inside") -> float:
    """G = 1 - dc_min/dDelta along the trapezoid shift of that side's width (design Section 4):
    G* at the optimum (1 when the inside contact stays active)."""
    if not fl.valid("c_min"):
        return float("nan")
    n = len(fl.impulse["c_min"].kappa)
    return 1.0 - fl.along("c_min", PT.edge_shift(n, lm, sigma, side, 1.0))


def gate4(g_star: float) -> bool:
    """G* within [0.98, 1.02] (design Section 8, Gate 4)."""
    return bool(GATE4_BAND[0] <= g_star <= GATE4_BAND[1])


# ------------------------------------------------------------------------- Gate 1


@dataclass(frozen=True)
class Resolve:
    """A warm re-solve of the key's NLP on a perturbed P (same bounds and window centres)."""

    w: np.ndarray
    lam_g: np.ndarray
    lam_x: np.ndarray
    status: str
    iterations: int
    seconds: float
    J: float = float("nan")   # the window return (f = -J)

    @property
    def ok(self) -> bool:
        return self.status == "Solve_Succeeded"


_FD_SOLVERS: dict[tuple[int, float], tuple[object, ca.Function]] = {}


def fd_solver(prob: K.KeyProblem, tol: float = GATE1_TOL) -> ca.Function:
    """Gate 1's re-solver: an IPOPT on the key solver's own (frozen) functions and options,
    REFINE_IPOPT's warm start (window.py:405), with tol tightened to ``tol``. At the refine
    solver's tol = 1e-7 a perturbation whose effect on the KKT residual is below 1e-7 stops
    after 0 iterations, so a small field's FD is one-sided (seen on A2 keys at 1e-5 fields);
    at 1e-10 those match the KKT field to 1e-7."""
    key = (id(prob.solver), float(tol))
    hit = _FD_SOLVERS.get(key)
    if hit is None or hit[0] is not prob.solver:
        opts = dict(prob.solver._opts)
        opts.update({"ipopt." + k: v for k, v in REFINE_IPOPT.items()})
        opts["ipopt.tol"] = float(tol)
        hit = (prob.solver, ca.nlpsol("to_ocp_fd", "ipopt", prob.solver.funcs["nlp"], opts))
        _FD_SOLVERS[key] = hit
    return hit[1]


def resolve(prob: K.KeyProblem, delta: Samples | None, solver=None) -> Resolve:
    """A warm re-solve (from the problem's primal and dual) on P(delta)
    (``keys.key_parameters``) with the problem's bounds and window centres; ``solver``
    defaults to ``fd_solver(prob)``."""
    return resolve_from(prob, delta, prob.w, prob.lam_g, prob.lam_x, solver)


def resolve_from(prob: K.KeyProblem, delta: Samples | None, w: np.ndarray, lam_g: np.ndarray,
                 lam_x: np.ndarray, solver=None) -> Resolve:
    """``resolve`` warm-started from a given primal-dual point (the dose continuation's
    previous amplitude) instead of the problem's own."""
    rs = fd_solver(prob) if solver is None else solver
    P = K.key_parameters(prob.solver, prob.track, prob.sigma, prob.ctx, prob.p, delta)
    t0 = time.perf_counter()
    sol = rs(x0=np.asarray(w, float), p=P, lbx=prob.lbx, ubx=prob.ubx, lbg=prob.lbg,
             ubg=prob.ubg, lam_g0=np.asarray(lam_g, float), lam_x0=np.asarray(lam_x, float))
    secs = time.perf_counter() - t0
    st = rs.stats()
    return Resolve(w=np.asarray(sol["x"]).ravel(), lam_g=np.asarray(sol["lam_g"]).ravel(),
                   lam_x=np.asarray(sol["lam_x"]).ravel(), status=str(st["return_status"]),
                   iterations=int(st["iter_count"]), seconds=secs, J=-float(sol["f"]))


def scaled(delta: Samples, c: float) -> Samples:
    return Samples(*(c * np.asarray(a, float) for a in delta))


@dataclass(frozen=True)
class FD:
    """Richardson-extrapolated central differences along one direction (Gate 1)."""

    name: str
    h: float
    values: dict[str, np.ndarray]     # outcome -> [y(-h), y(-h/2), y(h/2), y(h)]
    fd_h: dict[str, float]
    fd_h2: dict[str, float]
    fd_r: dict[str, float]
    disc: dict[str, float]            # |FD(h/2) - FD(h)|, the Richardson-pair discrepancy
    statuses: tuple[str, ...]
    iterations: tuple[int, ...]
    seconds: float

    @property
    def ok(self) -> bool:
        return all(s == "Solve_Succeeded" for s in self.statuses)


def richardson(prob: K.KeyProblem, reads: dict[str, Readout], sigma: int, decision: int,
               delta: Samples, h: float, name: str = "", solver=None) -> FD:
    """FD(h), FD(h/2) and their Richardson extrapolation for every valid outcome, from four
    warm re-solves (Gate 1; design Section 8)."""
    amps = (-h, -0.5 * h, 0.5 * h, h)
    vals: dict[str, list[float]] = {}
    st, it, secs = [], [], 0.0
    for c in amps:
        d = scaled(delta, c)
        rs = resolve(prob, d, solver)
        st.append(rs.status)
        it.append(rs.iterations)
        secs += rs.seconds
        for k, v in outcome_values(prob, reads, sigma, decision, rs.w, d).items():
            vals.setdefault(k, []).append(v)
    values = {k: np.asarray(v) for k, v in vals.items()}
    fd_h = {k: float((v[3] - v[0]) / (2 * h)) for k, v in values.items()}
    fd_h2 = {k: float((v[2] - v[1]) / h) for k, v in values.items()}
    fd_r = {k: (4.0 * fd_h2[k] - fd_h[k]) / 3.0 for k in values}
    disc = {k: abs(fd_h2[k] - fd_h[k]) for k in values}
    return FD(name=name, h=float(h), values=values, fd_h=fd_h, fd_h2=fd_h2, fd_r=fd_r, disc=disc,
              statuses=tuple(st), iterations=tuple(it), seconds=secs)


def gate1_score(kkt_vals, fd_r, disc, scale=None, zero_atol: float = ZERO_ATOL
                ) -> tuple[float, bool]:
    """(err, zero) of one field vector (design Section 8, Gate 1):
    err = ||KKT - FD_R|| / max(||FD_R||, 3 ||discrepancy||, ||scale||).

    ``scale`` is a field the quantity is known to cancel against: c_min's explicit edge
    term along the same directions (its true field is zero on an active contact, so
    ||FD_R|| is no scale; the edge's own move is). A vector that both sides put below
    ``zero_atol`` (per unit amplitude), with no such scale above it, is a *zero field*:
    ``zero`` True and err 0. Zero fields cannot fail by construction, so the decision pools
    only the others (``gates.gate1_decision``) and reports their count."""
    k, r, d = (np.atleast_1d(np.asarray(v, float)) for v in (kkt_vals, fd_r, disc))
    sc = 0.0 if scale is None else float(np.linalg.norm(np.atleast_1d(np.asarray(scale, float))))
    num = float(np.linalg.norm(k - r))
    if max(float(np.linalg.norm(k)), float(np.linalg.norm(r)), sc) <= zero_atol:
        return 0.0, True
    den = max(float(np.linalg.norm(r)), 3.0 * float(np.linalg.norm(d)), sc)
    if den == 0.0:
        return (0.0 if num == 0.0 else float("inf")), False
    return num / den, False


def gate1_error(kkt_vals, fd_r, disc, zero_atol: float = ZERO_ATOL, scale=None) -> float:
    """``gate1_score``'s err alone (0 for a zero field)."""
    return gate1_score(kkt_vals, fd_r, disc, scale, zero_atol)[0]


def gate1_directions(n: int, s1: float, lm: Landmarks, sigma: int, near=PT.GATE_NEAR_D,
                     corner=PT.GATE_CORNER, sigma_b: float = PT.GATE_SIGMA_B,
                     ds: float = PT.DS) -> dict[str, Samples]:
    """Gate 1's directions (``perturb.gate_directions``, shared with Gates 2 and 5): unit-peak
    Gaussian bumps of each feature near the car (s_1 + d) and at the corner's landmarks
    (s_c0 - 20, s_c0, s_apex, s_c1, s_c1 + 20), and the inside and outside trapezoid shifts."""
    return PT.gate_directions(n, s1, lm, sigma, near, corner, sigma_b, ds)


def direction_h(delta: Samples, h: dict[str, float] | float | None = None) -> float:
    """FD amplitude for a direction: the per-feature amplitude (``FD_H``) of its feature (the
    smallest when it mixes features)."""
    if isinstance(h, (int, float)):
        return float(h)
    hh = FD_H if h is None else h
    used = [f for f in PT.FEATURES if np.any(np.asarray(getattr(delta, f)) != 0)]
    return float(min(hh[f] for f in used)) if used else float(min(hh.values()))


def direction_group(name: str) -> str:
    """Gate 1's field vectors (``perturb.direction_group``): one feature's bumps in one region
    ('kappa@near+10' -> 'kappa@near', 'w_left@corner:apex' -> 'w_left@corner'); any other
    direction is its own group."""
    return PT.direction_group(name)


def score_scale(fl: Fields, q: str, directions: dict[str, Samples], names) -> np.ndarray | None:
    """The cancellation scale of quantity q along directions (``gate1_score``): c_min's
    explicit edge term, None for every other quantity."""
    if q != "c_min" or not fl.valid("c_min"):
        return None
    ex = fl.explicit["c_min"]
    return np.array([sum(float(np.dot(np.asarray(getattr(ex, f)),
                                      np.asarray(getattr(directions[nm], f), float)))
                         for f in PT.FEATURES) for nm in names])


@dataclass
class Gate1:
    """KKT directional derivatives against Richardson FD re-solves (Gate 1): per compared
    quantity (``GATE1_OUTCOMES``) and per group of directions (a feature's bumps in one
    region, each trapezoid), err of the field vector and whether it is a zero field."""

    names: tuple[str, ...]
    kkt: dict[str, np.ndarray]        # quantity -> [n_dir]
    fd_r: dict[str, np.ndarray]       # quantity -> [n_dir]
    disc: dict[str, np.ndarray]
    fd: list[FD]
    err: dict[tuple[str, str], float]     # (quantity, group) -> err
    zero: dict[tuple[str, str], bool] = field(default_factory=dict)

    @property
    def ok_solves(self) -> bool:
        return all(f.ok for f in self.fd)

    def worst(self) -> float:
        return float(max(self.err.values())) if self.err else float("nan")

    def by_quantity(self) -> dict[str, float]:
        """The worst group per quantity."""
        out: dict[str, float] = {}
        for (q, _), e in self.err.items():
            out[q] = max(out.get(q, 0.0), e)
        return out


def groups_of(names) -> dict[str, list[int]]:
    """Direction indices by field vector (``direction_group``)."""
    groups: dict[str, list[int]] = {}
    for i, nm in enumerate(names):
        groups.setdefault(direction_group(nm), []).append(i)
    return groups


def score_fields(fl: Fields, names, directions: dict[str, Samples], fd_r: dict, disc: dict
                 ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], bool]]:
    """(err, zero) per (quantity, group) of ``fl``'s fields against re-solve FDs."""
    groups = groups_of(names)
    err, zero = {}, {}
    for q in fd_r:
        if not fl.valid(q):
            continue
        kk = np.array([fl.along(q, directions[nm]) for nm in names])
        sc = score_scale(fl, q, directions, names)
        for g, ix in groups.items():
            e, z = gate1_score(kk[ix], fd_r[q][ix], disc[q][ix], None if sc is None else sc[ix])
            err[q, g], zero[q, g] = e, z
    return err, zero


def gate1(prob: K.KeyProblem, fl: Fields, directions: dict[str, Samples], sigma: int,
          h: dict[str, float] | float | None = None, solver=None) -> Gate1:
    """Gate 1 at one key: each direction's KKT derivative (``Fields.along``) against the
    Richardson FD of warm re-solves, for c_min, its line part c_line, v_apex, s_b and the
    decision's steer."""
    names = tuple(directions)
    fds = [richardson(prob, fl.reads, sigma, fl.decision, directions[nm],
                      direction_h(directions[nm], h), nm, solver) for nm in names]
    qs = [q for q in GATE1_OUTCOMES if fl.valid(q)]
    kk = {q: np.array([fl.along(q, directions[nm]) for nm in names]) for q in qs}
    r = {q: np.array([f.fd_r[q] for f in fds]) for q in qs}
    d = {q: np.array([f.disc[q] for f in fds]) for q in qs}
    err, zero = score_fields(fl, names, directions, r, d)
    return Gate1(names=names, kkt=kk, fd_r=r, disc=d, fd=fds, err=err, zero=zero)


# ------------------------------------------------------------------------- dose-response


_DOSE_SOLVERS: dict[int, tuple[object, ca.Function]] = {}


def dose_solver(prob: K.KeyProblem, tol: float = DOSE_TOL) -> ca.Function:
    """The dose continuation's solver: the key solver's own (frozen) functions and options
    with REFINE_IPOPT's warm start, tol ``tol`` and room for DOSE_SOLVER_MAX_ITER iterations
    (so that a step over DOSE_MAX_ITER is seen, then restarted)."""
    key = id(prob.solver)
    hit = _DOSE_SOLVERS.get(key)
    if hit is None or hit[0] is not prob.solver:
        opts = dict(prob.solver._opts)
        opts.update({"ipopt." + k: v for k, v in REFINE_IPOPT.items()})
        opts["ipopt.tol"] = float(tol)
        opts["ipopt.max_iter"] = int(DOSE_SOLVER_MAX_ITER)
        hit = (prob.solver, ca.nlpsol("to_ocp_dose", "ipopt", prob.solver.funcs["nlp"], opts))
        _DOSE_SOLVERS[key] = hit
    return hit[1]


@dataclass(frozen=True)
class DoseStep:
    """One amplitude of a warm continuation."""

    amplitude: float
    result: Resolve
    restarted: bool           # re-solved from the stored key (``reasons`` says why)
    reasons: tuple[str, ...]
    warm_from: float          # the amplitude the (first) solve was warm-started from


def dose_branches(amplitudes=DOSE_AMPLITUDES) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(widening, narrowing): each branch in increasing |amplitude|, continued from 0."""
    a = [float(x) for x in amplitudes]
    return (tuple(sorted(x for x in a if x > 0)), tuple(sorted((x for x in a if x < 0),
                                                               reverse=True)))


def continuation(solve, J0: float, amplitudes=DOSE_AMPLITUDES, max_iter: int = DOSE_MAX_ITER,
                 j_rtol: float = K.MOVE_J_RTOL) -> list[DoseStep]:
    """Warm continuation in amplitude with the design's restart rule (Section 4). ``solve(a,
    warm)`` solves at amplitude a from ``warm`` (a Resolve, or None: the stored key) and
    returns a Resolve. Each branch (widening, narrowing) starts at the stored key (J0) and
    steps out in |a|; a step restarts from the stored key when it did not converge, took
    more than ``max_iter`` iterations, or broke monotonicity of J (the trapezoid widening
    relaxes the corridor, so J is non-decreasing in a, within ``j_rtol`` |J0|). A restarted
    step is kept whatever it gives (its status is recorded); the branch continues from the
    last converged point."""
    out = []
    tol = abs(float(J0)) * float(j_rtol)
    for branch in dose_branches(amplitudes):
        warm, a_prev, J_prev = None, 0.0, float(J0)
        for a in branch:
            r = solve(a, warm)
            why = []
            if not r.ok:
                why.append(f"status {r.status}")
            if r.iterations > max_iter:
                why.append(f"{r.iterations} iterations > {max_iter}")
            mono = r.J >= J_prev - tol if a > 0 else r.J <= J_prev + tol
            if r.ok and not mono:
                why.append("J non-monotone in |amplitude|")
            restarted = bool(why) and warm is not None
            if restarted:
                r = solve(a, None)
            out.append(DoseStep(float(a), r, restarted, tuple(why), float(a_prev)))
            if r.ok:
                warm, a_prev, J_prev = r, a, r.J
    return out


def dose_optimum(prob: K.KeyProblem, lm: Landmarks, sigma: int, decision: int,
                 side: str = "inside", amplitudes=DOSE_AMPLITUDES, solver=None) -> dict:
    """The optimum's dose-response at a key along the trapezoid shift of one side's width:
    the warm continuation (``continuation``) on the dose solver, and per amplitude the
    outcomes read at the base plan's picks (``outcome_values``; c_min on the perturbed
    track) plus the re-solved plan's own outcomes (``outcome_plans`` on the moved point)."""
    rs = dose_solver(prob) if solver is None else solver
    n = int(prob.track.n_points)
    direction = PT.edge_shift(n, lm, sigma, side, 1.0)
    reads = readouts(prob, lm, sigma, decision)

    def solve(a, warm):
        d = scaled(direction, a)
        if warm is None:
            return resolve_from(prob, d, prob.w, prob.lam_g, prob.lam_x, rs)
        return resolve_from(prob, d, warm.w, warm.lam_g, warm.lam_x, rs)

    J0 = -float(K.constraints(prob)[0])          # the re-polished point's own J
    steps = continuation(solve, J0, amplitudes)
    rows = []
    for st in steps:
        r = st.result
        row = {"amplitude": st.amplitude, "status": r.status, "iterations": r.iterations,
               "J": r.J, "restarted": st.restarted, "reasons": list(st.reasons),
               "warm_from": st.warm_from, "seconds": r.seconds}
        if r.ok:
            d = scaled(direction, st.amplitude)
            row["at_base_picks"] = outcome_values(prob, reads, sigma, decision, r.w, d)
            moved = dataclasses.replace(prob, track=PT.perturbed_track(prob.track, d), w=r.w,
                                        lam_g=r.lam_g, lam_x=r.lam_x)
            pl = outcome_plans(moved, lm, sigma)
            row["plans"] = {k: (v.value if v.valid else float("nan")) for k, v in pl.items()}
        rows.append(row)
    return {"key_id": str(prob.rec.key_id), "side": side, "J0": J0,
            "base": {k: r.value for k, r in reads.items()}, "steps": rows,
            "n_restarts": int(sum(s.restarted for s in steps)),
            "all_ok": bool(all(s.result.ok for s in steps))}


# ------------------------------------------------------------------------- non-stored problems


def problem(solver: ak.AnchoredSolver, track: Track, res: WindowResult, ctx: np.ndarray,
            p: VehicleParams, x0: np.ndarray, kind: str = "nominal", key_id: str = "",
            stim_id: int = -1, direction: int = 1) -> K.KeyProblem:
    """A KeyProblem from a fresh ``AnchoredSolver.solve`` result, built as ``keys.rebuild``
    builds a stored key's: window centres of the solution, the same P and bounds (a_0 fixed
    when ``kind`` is 'anchored'). For synthetic windows and new solves (H3). The track passes
    step 1's lockbox guard (``answer_key._guard_track``: its name and source circuit)."""
    ak._guard_track(track)
    lam_g, lam_x = res.meta["lam"]
    spec = solver.spec
    rec = ak.KeyRecord(key_id=key_id or f"{kind}-{track.name}", kind=kind, status="verified",
                       stim_id=stim_id, stim_name=track.name, cell_id=-1, phase=0,
                       direction=direction, x0=np.asarray(x0, float), W=int(spec.steps),
                       spec=dataclasses.asdict(spec), alpha_max=float(p.alpha_max), shrink=0.0,
                       J=float(res.value), x=np.asarray(res.x, float), a=np.asarray(res.a, float),
                       lam_g=np.asarray(lam_g, float), lam_x=np.asarray(lam_x, float))
    sigma = _window_centres(rec.x[:, 0])
    P = K.key_parameters(solver, track, sigma, ctx, p)
    lbx, ubx, lbg, ubg = K.bounds(rec, spec, sigma, p)
    w = np.concatenate([rec.x.reshape(-1), rec.a.reshape(-1)])
    return K.KeyProblem(rec=rec, spec=spec, solver=solver, track=track, p=p,
                        ctx=np.asarray(ctx, float), sigma=sigma, P=P, lbx=lbx, ubx=ubx, lbg=lbg,
                        ubg=ubg, w=w, lam_g=rec.lam_g.copy(), lam_x=rec.lam_x.copy())
