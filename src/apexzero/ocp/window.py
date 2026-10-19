"""Training-objective OCP (TO-OCP), window mode: the agent's own problem, solved exactly.

Same everything as the RL environment:
- dynamics: the shared ODE, RK2 with ``substeps`` steps per 50 ms control period;
- controls: zero-order hold at 20 Hz, a_t in [-1, 1]^2, a_{-1} = 0 (env reset);
- track: the same C2 cubic B-spline the JAX simulator evaluates (see below);
- constraints at every post-step state: the env's termination set, tightened by small
  margins (``safety`` 3 cm, ``slip_safety`` 1e-4 rad) so a float64 replay of a
  constraint-riding optimum does not terminate: -(w_right(s) - edge) <= n <= w_left(s) - edge,
  |alpha_f|, |alpha_r| <= alpha_max - slip_safety, |xi| <= pi/2, vx >= 2 m/s;
- objective: maximise sum_t gamma^t [ (s_{t+1} - s_t) / 10 - 0.02 ||a_t - a_{t-1}||^2 ].

**Exact, floor-indexed track lookup.** casadi.bspline only exists for MX graphs, which made
exact Hessians ~50x slower. Each control step t instead carries a *window* of K spline
coefficients around its warm-start position sigma_t (coefficient m is c_{j0+m}), and a
lookup repeats the simulator's own arithmetic (``tracks.spline.eval_jax``): u = s / ds,
i = floor(u), t = u - i, f = b0(t) c_{i-1} + b1(t) c_i + b2(t) c_{i+1} + b3(t) c_{i+2}, with
the four coefficients picked from the window by equality masks (i - j0 == m). floor and ==
have zero derivative, so first and second derivatives flow only through the cubic basis of
t. (The previous lookup summed K smooth-kernel terms, all of which entered every
derivative; it cost ~4x more per Hessian at the same K.) The 20 RK2 stages of a control step
look up kappa from a small *block* of coefficients picked once per step at the node's
interval, so a wider window costs little. The window bounds |s_t - sigma_t| <= radius; an
outer loop re-centres the windows if the solution reaches a bound. (A very wide window is
not a substitute for a good guess: the bound also acts as a trust region, and from a poor
guess a 300 m radius converged to worse local optima. ``solve_continuation`` supplies
guesses within about a metre of the optimum instead.)

**Structured exact derivatives.** The step map is a composition Phi = psi^10 of RK2
substeps psi. IPOPT gets the constraint Jacobian from forward sensitivities and the
Hessian of the Lagrangian from the second-order adjoint of the composition,

    Hess(lam' Phi) = sum_k Y_k' Hess(mu_{k+1}' psi)(y_k) Y_k,
    y_k = (x_k, a),  Y_k = dy_k / d(x_0, a),  mu_10 = lam,  mu_k = psi_x(y_k)' mu_{k+1},

assembled into fixed sparsity patterns (custom ``jac_g`` / ``hess_lag``). This is the same
derivative CasADi's AD computes (tests check it to ~1e-14), but only small per-substep SX
functions are generated, so the C stays small enough to compile in about a minute and to
run from cache: compiled, it is ~10x faster than CasADi's generic AD in its virtual
machine. Compiled libraries are cached by a hash of their source (one per window length;
every level reuses it).

The track, the car context and x0 are all *parameters*, so one solver serves every level
with the same window length. The optimum J*_20 is a best-known value (IPOPT finds local
optima), verified by replaying its controls in the float64 simulator.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np

from apexzero.tracks.spline import basis, packed_coeffs
from apexzero.tracks.track import Track
from apexzero.vehicle import dynamics
from apexzero.vehicle.ops import CASADI
from apexzero.vehicle.params import F1_2025, VehicleParams, with_context

NX, NA = 8, 2
NZ = NX + NA
BIG = 1e12          # "no clamp" for closed tracks in the lookup's index arithmetic
MAX_TRAVEL = 6.5    # m per 50 ms control step (vx <= 120 m/s), sizes windows and blocks


@dataclass(frozen=True)
class WindowSpec:
    steps: int = 1200
    dt: float = 0.05
    substeps: int = 10
    gamma: float = 0.995
    progress_scale: float = 10.0
    action_cost: float = 0.02
    edge_margin: float = 1.0
    safety: float = 0.03        # m, corridor tightening
    slip_safety: float = 1e-4   # rad, slip tightening (IPOPT may exceed a bound by ~1e-8)
    radius: float = 60.0        # m: s_t may move this far from its window centre
    ds: float = 2.0             # nominal knot spacing (sizes the window; the true per-track
                                # spacing is a solver parameter)
    block: int = 12             # coefficients per step for the dynamics' lookups

    @property
    def k_behind(self) -> int:
        return int(np.ceil(self.radius / self.ds)) + 3

    @property
    def n_coef(self) -> int:
        # behind + ahead (radius + one control step of travel) + the block + spline support
        return self.k_behind + int(np.ceil((self.radius + MAX_TRAVEL) / self.ds)) + self.block + 2


@dataclass
class WindowResult:
    x: np.ndarray          # [T+1, 8]
    a: np.ndarray          # [T, 2]
    value: float           # discounted return J*
    success: bool
    status: str
    iterations: int
    solve_seconds: float
    rounds: int
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- lookups

def track_scalars(track: Track) -> list[float]:
    """q = [ds, u_lo, u_hi, i_max] for the lookups: no clamping on closed tracks, eval_jax's
    clamping (u in [0, n-1], last interval n-2) on open strips."""
    n = track.n_points
    return [track.ds, -BIG, BIG, BIG] if track.closed else [track.ds, 0.0, n - 1.0, n - 2.0]


def spline_index(s, q):
    """Interval index i (a float holding an integer) and fraction t, as eval_jax computes them."""
    u = ca.fmin(ca.fmax(s / q[0], q[1]), q[2])
    i = ca.fmin(ca.floor(u), q[3])
    return i, u - i


def _combine(t, c):
    b0, b1, b2, b3 = basis(t)
    return b0 * c[0] + b1 * c[1] + b2 * c[2] + b3 * c[3]


def window_lookup(s, windows, j0, q):
    """Splines at s, one per coefficient window (W[m] = c_{j0+m}, all of length K).

    Exact (identical arithmetic to eval_jax) while 1 <= i - j0 <= K - 3; zero outside.
    """
    i, t = spline_index(s, q)
    il = i - j0
    K = windows[0].numel()
    mask = ca.vertcat(*[il == m for m in range(1, K - 2)])
    return [_combine(t, [ca.dot(mask, W[k: k + K - 3]) for k in range(4)]) for W in windows]


def block_select(s, W, j0, q, L: int):
    """The L coefficients c_{i-1}, ..., c_{i+L-2} around s's interval i, and i itself.

    Valid while 1 <= i - j0 <= K - L + 1.
    """
    i, _ = spline_index(s, q)
    il = i - j0
    K = W.numel()
    mask = ca.vertcat(*[il == m for m in range(1, K - L + 2)])
    return ca.vertcat(*[ca.dot(mask, W[k: k + K - L + 1]) for k in range(L)]), i


def block_lookup(s, blk, i0, q):
    """Spline at s from a block selected at interval i0; exact while 0 <= i - i0 <= L - 4."""
    i, t = spline_index(s, q)
    li = i - i0
    L = blk.numel()
    mask = ca.vertcat(*[li == m for m in range(L - 3)])
    return _combine(t, [ca.dot(mask, blk[k: k + L - 3]) for k in range(4)])


def coefficient_windows(track: Track, centres: np.ndarray, spec: WindowSpec):
    """Per-step windows [K, T] of the kappa, w_left, w_right coefficients, and their j0."""
    n = track.n_points
    j0 = np.floor(np.asarray(centres) / track.ds).astype(int) - spec.k_behind
    idx = j0[:, None] + np.arange(spec.n_coef)[None, :]
    k = np.mod(idx, n) + 1 if track.closed else np.clip(idx, -1, n) + 1
    coefs = [packed_coeffs(a, track.closed, n) for a in (track.kappa, track.w_left, track.w_right)]
    return [c[k].T for c in coefs], j0.astype(float)


# ----------------------------------------------------------------------- step functions

def _context_params(ctx) -> VehicleParams:
    return with_context(F1_2025, ctx[0], ctx[1], ctx[2], ctx[3], ctx[4], ctx[5])


def substep_functions(spec: WindowSpec):
    """psi (one RK2 substep, as sim.rk2_step), its Jacobian, and Hess(mu' psi) w.r.t. (x, a)."""
    L = spec.block
    x = ca.SX.sym("x", NX)
    a = ca.SX.sym("a", NA)
    blk = ca.SX.sym("blk", L)
    i0 = ca.SX.sym("i0")
    q = ca.SX.sym("q", 4)
    ctx = ca.SX.sym("ctx", 6)
    mu = ca.SX.sym("mu", NX)
    p = _context_params(ctx)
    h = spec.dt / spec.substeps

    def rhs(xx):
        return dynamics.f(xx, a, block_lookup(xx[0], blk, i0, q), p, CASADI)

    k1 = rhs(x)
    xn = x + h * rhs(x + 0.5 * h * k1)
    hess, _ = ca.hessian(ca.dot(mu, xn), ca.vertcat(x, a))
    opts = {"cse": True}
    args = [x, a, blk, i0, q, ctx]
    return (ca.Function("psi", args, [xn], opts),
            ca.Function("psi_jac", args, [xn, ca.jacobian(xn, x), ca.jacobian(xn, a)], opts),
            ca.Function("psi_hess", [x, a, mu, blk, i0, q, ctx], [ca.densify(hess)], opts))


def _block_function(spec: WindowSpec) -> ca.Function:
    x = ca.SX.sym("x", NX)
    W = ca.SX.sym("W", spec.n_coef)
    j0 = ca.SX.sym("j0")
    q = ca.SX.sym("q", 4)
    blk, i = block_select(x[0], W, j0, q, spec.block)
    return ca.Function("block", [x, W, j0, q], [blk, i], {"cse": True})


def step_functions(spec: WindowSpec):
    """Per control step: Phi(x, a, W, j0, q, ctx) -> x_{t+1}; its Jacobian [8 x 10] w.r.t.
    (x, a); and Hess(lam' Phi) [10 x 10] by the second-order adjoint over the substeps."""
    psi, psi_jac, psi_hess = substep_functions(spec)
    block = _block_function(spec)
    x = ca.MX.sym("x", NX)
    a = ca.MX.sym("a", NA)
    W = ca.MX.sym("W", spec.n_coef)
    j0 = ca.MX.sym("j0")
    q = ca.MX.sym("q", 4)
    ctx = ca.MX.sym("ctx", 6)
    lam = ca.MX.sym("lam", NX)
    blk, i0 = block(x, W, j0, q)
    args = [x, a, W, j0, q, ctx]

    xc = x
    for _ in range(spec.substeps):
        xc = psi(xc, a, blk, i0, q, ctx)
    phi = ca.Function("phi", args, [xc])

    xs, jx, ja = [x], [], []
    for _ in range(spec.substeps):
        xn, jxk, jak = psi_jac(xs[-1], a, blk, i0, q, ctx)
        xs.append(xn)
        jx.append(jxk)
        ja.append(jak)
    sx, sa = [ca.MX.eye(NX)], [ca.MX.zeros(NX, NA)]        # d x_k / d(x_0, a)
    for k in range(spec.substeps):
        sx.append(ca.mtimes(jx[k], sx[k]))
        sa.append(ca.mtimes(jx[k], sa[k]) + ja[k])
    phi_jac = ca.Function("phi_jac", args, [xs[-1], ca.horzcat(sx[-1], sa[-1])])

    mu = [None] * spec.substeps + [lam]
    for k in range(spec.substeps - 1, -1, -1):
        mu[k] = ca.mtimes(jx[k].T, mu[k + 1])
    bottom = ca.horzcat(ca.MX.zeros(NA, NX), ca.MX.eye(NA))
    hess = psi_hess(x, a, mu[1], blk, i0, q, ctx)
    for k in range(1, spec.substeps):
        Y = ca.vertcat(ca.horzcat(sx[k], sa[k]), bottom)
        hess = hess + ca.mtimes(Y.T, ca.mtimes(psi_hess(xs[k], a, mu[k + 1], blk, i0, q, ctx), Y))
    phi_hess = ca.Function("phi_hess", [x, a, lam, W, j0, q, ctx], [hess])
    return phi, phi_jac, phi_hess


def step_function(spec: WindowSpec) -> ca.Function:
    """x_{t+1} = Phi(x_t, a_t; coefficient window, j0, q, context): RK2 as in sim.py."""
    return step_functions(spec)[0]


def path_functions(spec: WindowSpec):
    """Constraints at a post-step state, scaled: [n - (wl - edge), -(wr - edge) - n,
    alpha_f / alpha_max, alpha_r / alpha_max]; their Jacobian and Hess(nu' path) (upper)."""
    K = spec.n_coef
    x = ca.SX.sym("x", NX)
    Wl = ca.SX.sym("Wl", K)
    Wr = ca.SX.sym("Wr", K)
    j0 = ca.SX.sym("j0")
    q = ca.SX.sym("q", 4)
    ctx = ca.SX.sym("ctx", 6)
    amax = ca.SX.sym("alpha_max")
    nu = ca.SX.sym("nu", 4)
    edge = spec.edge_margin + spec.safety
    wl, wr = window_lookup(x[0], [Wl, Wr], j0, q)
    fo = dynamics.tyre_forces(x, _context_params(ctx), CASADI)
    g = ca.vertcat(x[1] - (wl - edge), -(wr - edge) - x[1], fo.alpha_f / amax, fo.alpha_r / amax)
    hess, _ = ca.hessian(ca.dot(nu, g), x)
    args = [x, Wl, Wr, j0, q, ctx, amax]
    opts = {"cse": True}
    return (ca.Function("path", args, [g], opts),
            ca.Function("path_jac", args, [ca.jacobian(g, x)], opts),
            ca.Function("path_hess", args + [nu], [ca.triu(hess)], opts))


# ----------------------------------------------------------------------------- compiling

def default_cache_dir() -> Path:
    return Path(os.environ.get("APEXZERO_CACHE", Path.home() / ".cache" / "apexzero")) / "to_ocp"


def compile_functions(funcs: dict[str, ca.Function], cache_dir: Path | None = None,
                      flags: tuple[str, ...] = ("-O2",)) -> dict[str, ca.Function]:
    """Generate C for ``funcs``, compile it into a shared library (cached by a hash of the
    source and flags) and load it. No fast-math: the compiled code must keep IEEE semantics."""
    cache = Path(cache_dir or default_cache_dir())
    cg = ca.CodeGenerator("to_ocp.c", {"with_header": False})
    for f in funcs.values():
        cg.add(f)
    with tempfile.TemporaryDirectory(prefix="to_ocp_") as tmp:
        cwd = os.getcwd()
        try:
            os.chdir(tmp)           # CodeGenerator writes into the working directory
            cg.generate()
        finally:
            os.chdir(cwd)
        src = Path(tmp) / "to_ocp.c"
        tag = hashlib.sha256(src.read_bytes() + " ".join(flags).encode()).hexdigest()[:20]
        lib = cache / f"to_ocp_{tag}.so"
        if not lib.exists():
            cache.mkdir(parents=True, exist_ok=True)
            part = cache / f"{lib.name}.{os.getpid()}.part"     # atomic publish: concurrent
            cc = os.environ.get("CC", "cc")                     # builders write separate files
            subprocess.run([cc, *flags, "-fPIC", "-shared", str(src), "-o", str(part), "-lm"],
                           check=True, capture_output=True)
            os.replace(part, lib)
    return {k: ca.external(f.name(), str(lib)) for k, f in funcs.items()}


# -------------------------------------------------------------------------------- solver

class WindowSolver:
    """One IPOPT solver for every level with the same window length (and spec).

    ``scale`` (8 state scales, optional) makes IPOPT work in scaled variables: x = D x~ + o
    with s offset by its window centre, and the defect rows divided by the same D. IPOPT
    does not scale variables itself, so this changes its path (regularisation, barrier,
    bound pushes); see docs/findings/to-ocp-spike.md for what it was worth.
    """

    def __init__(self, spec: WindowSpec = WindowSpec(), max_iter: int = 3000, tol: float = 1e-7,
                 print_level: int = 0, compile: bool = True, ipopt: dict | None = None,
                 warm_ipopt: dict | None = None, cache_dir: Path | None = None,
                 scale: tuple[float, ...] | None = None):
        self.spec = spec
        T, K = spec.steps, spec.n_coef
        self.T, self.K = T, K
        nw = NX * (T + 1) + NA * T
        ng = NX * T + 4 * T
        self.nw, self.ng = nw, ng

        def ix(i, t):
            return NX * t + i

        def iz(e, t):                 # (x_t, a_t) -> index in w = [vec(X); vec(A)]
            return NX * t + e if e < NX else NX * (T + 1) + NA * t + (e - NX)

        self.scaled = scale is not None
        sx = np.ones(NX) if scale is None else np.asarray(scale, float)
        self.var_scale = np.concatenate([np.tile(sx, T + 1), np.ones(NA * T)])
        self.con_scale = np.concatenate([np.tile(1.0 / sx, T), np.ones(4 * T)])
        wt = ca.MX.sym("x", nw)                 # IPOPT's variables
        if self.scaled:
            offset = ca.MX.sym("offset", nw)
            w = ca.DM(self.var_scale) * wt + offset
        else:
            w = wt
        X = ca.reshape(w[: NX * (T + 1)], NX, T + 1)
        A = ca.reshape(w[NX * (T + 1):], NA, T)
        Pk, Pl, Pr = (ca.MX.sym(nm, K, T) for nm in ("Pk", "Pl", "Pr"))
        J0 = ca.MX.sym("J0", 1, T)
        q = ca.MX.sym("q", 4)
        ctx = ca.MX.sym("ctx", 6)
        amax = ca.MX.sym("alpha_max")
        P = ca.vertcat(ca.vec(Pk), ca.vec(Pl), ca.vec(Pr), ca.vec(J0), q, ctx, amax)
        if self.scaled:
            P = ca.vertcat(P, offset)
        qT, cT, aT = ca.repmat(q, 1, T), ca.repmat(ctx, 1, T), ca.repmat(amax, 1, T)
        phi, phi_jac, phi_hess = step_functions(spec)
        path, path_jac, path_hess = path_functions(spec)

        # Objective f = -J = b'w + w'Hf w / 2 (progress is linear, the action cost quadratic).
        disc = spec.gamma ** np.arange(T)
        b = np.zeros(nw)
        b[[ix(0, t + 1) for t in range(T)]] -= disc / spec.progress_scale
        b[[ix(0, t) for t in range(T)]] += disc / spec.progress_scale
        hr, hc, hv = [], [], []
        c2 = 2.0 * spec.action_cost
        for t in range(T):
            for j in range(NA):
                hr.append(iz(NX + j, t))
                hc.append(iz(NX + j, t))
                hv.append(c2 * disc[t] + (c2 * disc[t + 1] if t + 1 < T else 0.0))
                if t + 1 < T:
                    for r, c in ((iz(NX + j, t), iz(NX + j, t + 1)), (iz(NX + j, t + 1), iz(NX + j, t))):
                        hr.append(r)
                        hc.append(c)
                        hv.append(-c2 * disc[t + 1])
        order = np.lexsort((hr, hc))              # DM(sparsity, values) takes CCS order
        Hf = ca.DM(ca.Sparsity.triplet(nw, nw, [hr[k] for k in order], [hc[k] for k in order]),
                   [hv[k] for k in order])
        Hfw = ca.mtimes(Hf, w)
        f = ca.dot(ca.DM(b), w) + 0.5 * ca.dot(w, Hfw)
        grad_f = ca.DM(b) + Hfw

        # Constraints: defects (rows 8t+i), corridor (8T+2t+k), slip (10T+2t+k).
        g_path = path(X[:, 1:], Pl, Pr, J0, qT, cT, aT)
        g = ca.vertcat(ca.vec(X[:, 1:] - phi.map(T)(X[:, :T], A, Pk, J0, qT, cT)),
                       ca.vec(g_path[0:2, :]), ca.vec(g_path[2:4, :]))

        # Jacobian: nonzeros gathered from [ones (x_{t+1}); -dPhi_t (8x10 dense); path jac].
        rows, cols, src = [], [], []
        for t in range(T):
            for i in range(NX):
                rows.append(NX * t + i)
                cols.append(ix(i, t + 1))
                src.append(NX * t + i)
        base = NX * T
        for t in range(T):
            for c in range(NZ):
                for r in range(NX):
                    rows.append(NX * t + r)
                    cols.append(iz(c, t))
                    src.append(base + NX * NZ * t + NX * c + r)
        base += NX * NZ * T
        pr_, pc_ = path_jac.sparsity_out(0).get_triplet()
        npj = len(pr_)
        for t in range(T):
            for e in range(npj):
                r = pr_[e]
                rows.append(NX * T + 2 * t + r if r < 2 else 10 * T + 2 * t + r - 2)
                cols.append(ix(pc_[e], t + 1))
                src.append(base + npj * t + e)
        order = np.lexsort((rows, cols))          # CCS order: by column, then row
        jac_sp = ca.Sparsity.triplet(ng, nw, rows, cols)
        _, dphi = phi_jac.map(T)(X[:, :T], A, Pk, J0, qT, cT)
        dpath = path_jac.map(T)(X[:, 1:], Pl, Pr, J0, qT, cT, aT)
        jsrc = ca.vertcat(ca.DM.ones(NX * T), -ca.vec(dphi), dpath.nz[:])
        jac_g = ca.MX(jac_sp, jsrc[[int(k) for k in np.asarray(src)[order]]])

        # Hessian of the Lagrangian (upper triangle): contributions summed by a 0/1 matrix.
        lam_f = ca.MX.sym("lam_f")
        lam_g = ca.MX.sym("lam_g", ng)
        rows, cols, src = [], [], []
        for t in range(T):
            for c in range(NZ):
                for r in range(c + 1):
                    rows.append(iz(r, t))
                    cols.append(iz(c, t))
                    src.append(NZ * NZ * t + NZ * c + r)
        base = NZ * NZ * T
        qr_, qc_ = path_hess.sparsity_out(0).get_triplet()
        nph = len(qr_)
        for t in range(T):
            for e in range(nph):
                rows.append(ix(qr_[e], t + 1))
                cols.append(ix(qc_[e], t + 1))
                src.append(base + nph * t + e)
        base += nph * T
        Hf_u = ca.triu(Hf)
        fr, fc = Hf_u.sparsity().get_triplet()
        for e in range(len(fr)):
            rows.append(fr[e])
            cols.append(fc[e])
            src.append(base + e)
        n_src = base + len(fr)
        key = np.asarray(cols) * nw + np.asarray(rows)
        uniq, slot = np.unique(key, return_inverse=True)
        hess_sp = ca.Sparsity.triplet(nw, nw, [int(k) for k in uniq % nw], [int(k) for k in uniq // nw])
        gather = ca.DM(ca.Sparsity.triplet(len(uniq), n_src, [int(k) for k in slot], src), 1.0)
        lam = ca.DM(self.con_scale) * lam_g if self.scaled else lam_g   # multipliers of g
        lam_dyn = ca.reshape(lam[: NX * T], NX, T)
        nu = ca.vertcat(ca.reshape(lam[NX * T: 10 * T], 2, T), ca.reshape(lam[10 * T:], 2, T))
        # g = x_{t+1} - Phi, so the defects contribute Hess((-lam)' Phi).
        hdyn = phi_hess.map(T)(X[:, :T], A, -lam_dyn, Pk, J0, qT, cT)
        hpath = path_hess.map(T)(X[:, 1:], Pl, Pr, J0, qT, cT, aT, nu)
        hsrc = ca.vertcat(ca.vec(hdyn), hpath.nz[:], lam_f * ca.DM(Hf_u.nonzeros()))
        hess_l = ca.MX(hess_sp, ca.mtimes(gather, hsrc))
        if self.scaled:     # chain rule for x = D x~ + o and g~ = S g (diagonal D, S)
            Dm, Sm = ca.diag(ca.DM(self.var_scale)), ca.diag(ca.DM(self.con_scale))
            g = ca.DM(self.con_scale) * g
            grad_f = ca.DM(self.var_scale) * grad_f
            jac_g = ca.mtimes(ca.mtimes(Sm, jac_g), Dm)
            hess_l = ca.mtimes(ca.mtimes(Dm, hess_l), Dm)

        funcs = {
            "nlp": ca.Function("to_ocp_nlp", [wt, P], [f, g], ["x", "p"], ["f", "g"]),
            "grad_f": ca.Function("to_ocp_grad_f", [wt, P], [f, grad_f], ["x", "p"], ["f", "grad_f_x"]),
            "jac_g": ca.Function("to_ocp_jac_g", [wt, P], [g, jac_g], ["x", "p"], ["g", "jac_g_x"]),
            "hess_lag": ca.Function("to_ocp_hess_lag", [wt, P, lam_f, lam_g], [hess_l],
                                    ["x", "p", "lam_f", "lam_g"], ["hess_gamma_x_x"]),
        }
        self.compiled = False
        t0 = time.perf_counter()
        if compile:
            try:
                funcs = compile_functions(funcs, cache_dir)
                self.compiled = True
            except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
                warnings.warn(f"TO-OCP: C compilation failed ({e}); using CasADi's VM", stacklevel=2)
        self.compile_seconds = time.perf_counter() - t0
        self.funcs = funcs
        self.n_param = P.numel()
        opts = {"ipopt.max_iter": max_iter, "ipopt.tol": tol, "ipopt.print_level": print_level,
                "print_time": False, "ipopt.mu_strategy": "adaptive", "ipopt.linear_solver": "mumps",
                "calc_lam_p": False, "no_nlp_grad": True, "grad_f": funcs["grad_f"],
                "jac_g": funcs["jac_g"], "hess_lag": funcs["hess_lag"]}
        opts.update({"ipopt." + k: v for k, v in (ipopt or {}).items()})
        self._opts = opts
        self.solver = ca.nlpsol("to_ocp", "ipopt", funcs["nlp"], opts)
        self._warm_ipopt = warm_ipopt
        self._warm_solver = None

    def warm_solver(self):
        """Solver for re-centring rounds (IPOPT warm-start options), if any were given."""
        if not self._warm_ipopt:
            return self.solver
        if self._warm_solver is None:
            opts = dict(self._opts)
            opts.update({"ipopt." + k: v for k, v in self._warm_ipopt.items()})
            self._warm_solver = ca.nlpsol("to_ocp_warm", "ipopt", self.funcs["nlp"], opts)
        return self._warm_solver

    def parameters(self, track: Track, centres: np.ndarray, ctx: np.ndarray, p: VehicleParams) -> np.ndarray:
        """Track windows (centred on ``centres``, one per step), track scalars and car."""
        (Ck, Cl, Cr), j0 = coefficient_windows(track, centres, self.spec)
        return np.concatenate([Ck.T.reshape(-1), Cl.T.reshape(-1), Cr.T.reshape(-1), j0,
                               track_scalars(track), ctx, [p.alpha_max]])

    def offset(self, sigma: np.ndarray) -> np.ndarray:
        """o in x = D x~ + o: each node's s relative to its centre sigma [T+1] (scaled solvers)."""
        o = np.zeros(self.nw)
        o[: NX * (self.T + 1): NX] = sigma
        return o

    def solve(self, track: Track, x0: np.ndarray, init: tuple[np.ndarray, np.ndarray],
              ctx: np.ndarray | None = None, p: VehicleParams = F1_2025,
              max_rounds: int = 6, lam0: tuple[np.ndarray, np.ndarray] | None = None) -> WindowResult:
        spec, T = self.spec, self.T
        if abs(track.ds - spec.ds) > 0.05:
            raise ValueError(f"track ds {track.ds} too far from the nominal {spec.ds} the window is sized for")
        if ctx is None:
            cla = p.cla_f + p.cla_r
            ctx = np.array([p.m, p.p_max, cla, p.cda, p.mu_scale, p.cla_f / cla])
        xs, acts = (np.array(v, float) for v in init)
        xs[0] = x0
        lam_g, lam_x = lam0 if lam0 is not None else (None, None)
        R = spec.radius
        wall, cpu, iters, log = 0.0, 0.0, 0, []
        for rnd in range(1, max_rounds + 1):
            # Step t's window is centred on its start sigma_t; node t+1 must stay inside the
            # windows of both steps it belongs to. (The car only moves forward, so a warm start
            # that steps back, e.g. at a junction of two guesses, is centred monotonically.)
            sigma = np.maximum.accumulate(xs[:, 0])
            P = self.parameters(track, sigma[:-1], ctx, p)
            o, d = np.zeros(self.nw), self.var_scale
            if self.scaled:
                o = self.offset(sigma)
                P = np.concatenate([P, o])
            lo = np.maximum(sigma[1:], sigma[:-1]) - R
            hi = np.minimum(sigma[1:], sigma[:-1] + MAX_TRAVEL - 0.5) + R
            lbX = np.tile(np.array([-np.inf, -np.inf, -0.5 * np.pi, 2.0, -25.0, -5.0, -p.delta_max, -1.0])[:, None], (1, T + 1))
            ubX = np.tile(np.array([np.inf, np.inf, 0.5 * np.pi, 120.0, 25.0, 5.0, p.delta_max, 1.0])[:, None], (1, T + 1))
            lbX[0, 1:], ubX[0, 1:] = lo, hi
            lbX[:, 0] = ubX[:, 0] = x0
            lbx = (np.concatenate([lbX.T.reshape(-1), -np.ones(NA * T)]) - o) / d
            ubx = (np.concatenate([ubX.T.reshape(-1), np.ones(NA * T)]) - o) / d
            slip = 1.0 - spec.slip_safety / p.alpha_max        # |alpha| / alpha_max bound
            lbg = np.concatenate([np.zeros(NX * T), np.full(2 * T, -np.inf), np.full(2 * T, -slip)])
            ubg = np.concatenate([np.zeros(NX * T), np.zeros(2 * T), np.full(2 * T, slip)])
            w0 = np.clip((np.concatenate([xs.reshape(-1), acts.reshape(-1)]) - o) / d, lbx, ubx)
            kw = dict(x0=w0, p=P, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            solver = self.solver
            if lam_g is not None:
                kw.update(lam_g0=lam_g, lam_x0=lam_x)
                solver = self.warm_solver()
            t0, c0 = time.perf_counter(), time.process_time()
            sol = solver(**kw)
            wall += time.perf_counter() - t0
            cpu += time.process_time() - c0
            st = solver.stats()
            iters += int(st["iter_count"])
            log.append({"round": rnd, "iterations": int(st["iter_count"]), "status": st["return_status"],
                        **{k: v for k, v in st.items() if k.startswith("t_proc_nlp")}})
            wv = np.asarray(sol["x"]).ravel() * d + o
            xs = wv[: NX * (T + 1)].reshape(T + 1, NX)
            acts = wv[NX * (T + 1):].reshape(T, NA)
            lam_g, lam_x = np.asarray(sol["lam_g"]).ravel(), np.asarray(sol["lam_x"]).ravel()
            at_bound = bool(np.any(xs[1:, 0] < lo + 1.0) or np.any(xs[1:, 0] > hi - 1.0))
            if not at_bound:
                break
        return WindowResult(x=xs, a=acts, value=float(-sol["f"]), success=bool(st["success"]) and not at_bound,
                            status=str(st["return_status"]) + (" (window bound active)" if at_bound else ""),
                            iterations=iters, solve_seconds=wall, rounds=rnd,
                            meta={"cpu_seconds": cpu, "rounds": log, "lam": (lam_g, lam_x),
                                  "track": track.name, "ctx": ctx.tolist(), "spec": spec.__dict__,
                                  "compiled": self.compiled})


def window_return(xs: np.ndarray, acts: np.ndarray, spec: WindowSpec = WindowSpec()) -> float:
    T = len(acts)
    prev = np.vstack([np.zeros((1, 2)), acts[:-1]])
    r = (xs[1:, 0] - xs[:-1, 0]) / spec.progress_scale - spec.action_cost * np.sum((acts - prev) ** 2, 1)
    return float(np.sum(spec.gamma ** np.arange(T) * r))


# ------------------------------------------------------------------- warm starts, continuation

def min_time_guess(ocp, track: Track, x_start: np.ndarray, steps: int, dt: float = 0.05):
    """The s-domain minimum-time optimum (``min_time.OCPResult``) mapped to time from
    x_start's arc length on: close to the TO-OCP optimum in position and speed, though not
    dynamically consistent with x_start itself. Returns (xs [steps+1, 8], acts [steps, 2])."""
    s0 = float(x_start[0])
    if track.closed:
        L, lap = track.length, ocp.lap_time
        t_grid, s_grid = np.r_[ocp.t, lap], np.r_[ocp.s, L]
        z_grid, u_grid = np.vstack([ocp.z, ocp.z[:1]]), np.vstack([ocp.u, ocp.u[:1]])
        laps = np.floor(s0 / L)
        tau = np.interp(s0 - laps * L, s_grid, t_grid) + dt * np.arange(steps + 1)
        s = np.interp(np.mod(tau, lap), t_grid, s_grid) + L * (laps + np.floor(tau / lap))
        s_mod = np.mod(s, L)
    else:
        s_grid, z_grid, u_grid = ocp.s, ocp.z, ocp.u
        tau = np.interp(s0, ocp.s, ocp.t) + dt * np.arange(steps + 1)
        s = s_mod = np.interp(tau, ocp.t, ocp.s)
    z = np.stack([np.interp(s_mod, s_grid, z_grid[:, k]) for k in range(NX - 1)], 1)
    acts = np.stack([np.interp(s_mod[:-1], s_grid, u_grid[:, k]) for k in range(NA)], 1)
    xs = np.hstack([s[:, None], z])
    xs[0] = x_start
    return xs, acts


def prefix_multipliers(lam: tuple[np.ndarray, np.ndarray], T_prev: int, N: int, T: int):
    """Multipliers of a T_prev-step solution kept for its first N steps, zero elsewhere,
    laid out for a T-step window (warm start of the next continuation stage)."""
    lam_g, lam_x = lam
    g = np.zeros(12 * T)
    g[: NX * N] = lam_g[: NX * N]
    g[NX * T: NX * T + 2 * N] = lam_g[NX * T_prev: NX * T_prev + 2 * N]
    g[10 * T: 10 * T + 2 * N] = lam_g[10 * T_prev: 10 * T_prev + 2 * N]
    x = np.zeros(NX * (T + 1) + NA * T)
    x[: NX * (N + 1)] = lam_x[: NX * (N + 1)]
    x[NX * (T + 1): NX * (T + 1) + NA * N] = lam_x[NX * (T_prev + 1): NX * (T_prev + 1) + NA * N]
    return g, x


def solve_continuation(solver_for, track: Track, x0: np.ndarray, horizons, first_guess,
                       extend, overlap: int = 20, ctx: np.ndarray | None = None,
                       p: VehicleParams = F1_2025, warm_multipliers: bool = False,
                       previous: WindowResult | None = None) -> list[WindowResult]:
    """Horizon continuation: solve a short window, then each next one from the previous
    optimum.

    Optima of nested windows agree except near the shorter window's end (with no terminal
    value, it does not brake for what lies beyond), so stage k is warm-started with stage
    k-1's optimum, cut ``overlap`` steps before the end of the shorter of the two windows,
    and continued by ``extend(state, n)`` -> (xs [n+1, 8] starting at state, acts [n, 2]).
    ``horizons`` (control steps) usually ascend; a stage may also be *shorter* than its
    predecessor, which re-solves that window from a longer optimum (a "lookahead polish":
    a different start that can reach a better local optimum). The first stage starts from
    ``first_guess``, or continues from ``previous`` if given. ``solver_for(steps)`` returns
    the WindowSolver for a window length. With ``warm_multipliers`` the kept prefix's
    multipliers are passed on too (the solver's ``warm_ipopt`` options decide how IPOPT
    uses them).
    """
    results: list[WindowResult] = []
    for T in horizons:
        lam0 = None
        prev = results[-1] if results else previous
        if prev is None:
            xs, acts = (np.asarray(v, float) for v in first_guess)
            xs, acts = xs[: T + 1], acts[:T]
        else:
            N = max(min(len(prev.a), T) - overlap, 1)
            xe, ae = extend(prev.x[N], T - N)
            xs = np.vstack([prev.x[: N + 1], np.asarray(xe)[1:]])
            acts = np.vstack([prev.a[:N], ae])
            if warm_multipliers and "lam" in prev.meta:
                lam0 = prefix_multipliers(prev.meta["lam"], len(prev.a), N, T)
        results.append(solver_for(T).solve(track, x0, init=(xs, acts), ctx=ctx, p=p, lam0=lam0))
    return results
