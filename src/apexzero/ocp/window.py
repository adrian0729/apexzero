"""Training-objective OCP (TO-OCP), window mode: the agent's own problem, solved exactly.

Same everything as the RL environment:
- dynamics: the shared ODE, RK2 with ``substeps`` steps per 50 ms control period;
- controls: zero-order hold at 20 Hz, a_t in [-1, 1]^2, a_{-1} = 0 (env reset);
- track: the same C2 cubic B-spline the JAX simulator evaluates (see below);
- constraints at every post-step state: the env's termination set, tightened by a small
  ``safety`` margin so a float64 replay of a constraint-riding optimum does not terminate:
  -(w_right(s) - edge) <= n <= w_left(s) - edge, |alpha_f|, |alpha_r| <= alpha_max,
  |xi| <= pi/2, vx >= 2 m/s;
- objective: maximise sum_t gamma^t [ (s_{t+1} - s_t) / 10 - 0.02 ||a_t - a_{t-1}||^2 ].

**Fast exact track lookup.** casadi.bspline only exists for MX graphs, which made exact
Hessians ~50x slower. Here each control step t carries a *window* of K spline
coefficients around its warm-start position sigma_t, and kappa(s), w(s) are evaluated
as the exact B-spline sum  f(s) = sum_m C[m] B(s/ds - (j0 + m))  in SX. Inside the window
this is identical to the simulator's spline; s_t is bounded to stay inside it, and an
outer loop re-centres the windows if the solution reaches a bound. The track, the car
context and x0 are all *parameters*, so one compiled solver serves every level with the
same window length.

The optimum J*_20 is a best-known value (IPOPT finds local optima), verified by replaying
its controls in the float64 simulator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from apexzero.tracks.spline import packed_coeffs
from apexzero.tracks.track import Track
from apexzero.vehicle import dynamics
from apexzero.vehicle.ops import CASADI
from apexzero.vehicle.params import F1_2025, VehicleParams, with_context


@dataclass(frozen=True)
class WindowSpec:
    steps: int = 1200
    dt: float = 0.05
    substeps: int = 10
    gamma: float = 0.995
    progress_scale: float = 10.0
    action_cost: float = 0.02
    edge_margin: float = 1.0
    safety: float = 0.03
    radius: float = 15.0        # m: s_t may move this far from its window centre
    ds: float = 2.0             # nominal knot spacing (sizes the window; the true per-track
                                # spacing is a solver parameter)

    @property
    def k_behind(self) -> int:
        return int(np.ceil(self.radius / self.ds)) + 3

    @property
    def n_coef(self) -> int:
        # behind + ahead (radius + one control step of travel at up to 120 m/s) + support
        return self.k_behind + int(np.ceil((self.radius + 6.5) / self.ds)) + 4


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


def kernel(x):
    """Uniform cubic B-spline basis centred at 0 (support [-2, 2])."""
    ax = ca.fabs(x)
    return (ca.fmax(0.0, 2.0 - ax) ** 3 - 4.0 * ca.fmax(0.0, 1.0 - ax) ** 3) / 6.0


def window_eval(s, C, j0, inv_ds, K):
    u = s * inv_ds - j0
    out = 0.0
    for m in range(K):
        out = out + C[m] * kernel(u - m)
    return out


def _context_params(ctx) -> VehicleParams:
    return with_context(F1_2025, ctx[0], ctx[1], ctx[2], ctx[3], ctx[4], ctx[5])


def step_function(spec: WindowSpec) -> ca.Function:
    """x_{t+1} = Phi(x_t, a_t; kappa window, j0, context): RK2 as in sim.py, in SX."""
    K = spec.n_coef
    x = ca.SX.sym("x", 8)
    a = ca.SX.sym("a", 2)
    Ck = ca.SX.sym("Ck", K)
    j0 = ca.SX.sym("j0")
    inv_ds = ca.SX.sym("inv_ds")
    ctx = ca.SX.sym("ctx", 6)
    p = _context_params(ctx)
    h = spec.dt / spec.substeps

    def rhs(xx):
        return dynamics.f(xx, a, window_eval(xx[0], Ck, j0, inv_ds, K), p, CASADI)

    xc = x
    for _ in range(spec.substeps):
        k1 = rhs(xc)
        xc = xc + h * rhs(xc + 0.5 * h * k1)
    return ca.Function("Phi", [x, a, Ck, j0, inv_ds, ctx], [xc])


def path_function(spec: WindowSpec) -> ca.Function:
    """Constraint values at a post-step state: [n - (wl - edge), -(wr - edge) - n, alpha_f, alpha_r]."""
    K = spec.n_coef
    x = ca.SX.sym("x", 8)
    Cl = ca.SX.sym("Cl", K)
    Cr = ca.SX.sym("Cr", K)
    j0 = ca.SX.sym("j0")
    inv_ds = ca.SX.sym("inv_ds")
    ctx = ca.SX.sym("ctx", 6)
    p = _context_params(ctx)
    edge = spec.edge_margin + spec.safety
    wl = window_eval(x[0], Cl, j0, inv_ds, K)
    wr = window_eval(x[0], Cr, j0, inv_ds, K)
    fo = dynamics.tyre_forces(x, p, CASADI)
    return ca.Function("path", [x, Cl, Cr, j0, inv_ds, ctx],
                       [ca.vertcat(x[1] - (wl - edge), -(wr - edge) - x[1], fo.alpha_f, fo.alpha_r)])


class WindowSolver:
    """One compiled IPOPT solver for every level with the same window length."""

    def __init__(self, spec: WindowSpec = WindowSpec(), max_iter: int = 3000, tol: float = 1e-7,
                 print_level: int = 0, threads: int = 1):
        self.spec = spec
        T, K = spec.steps, spec.n_coef
        mode = ("thread", threads) if threads > 1 else ("serial",)
        phi = step_function(spec).map(T, *mode)
        path = path_function(spec).map(T, *mode)

        # MX outer graph over mapped SX per-step functions: derivatives stay per step
        # (map nodes) instead of expanding T copies of the dynamics into one SX graph.
        X = ca.MX.sym("X", 8, T + 1)
        A = ca.MX.sym("A", 2, T)
        Pk = ca.MX.sym("Pk", K, T)
        Pl = ca.MX.sym("Pl", K, T)
        Pr = ca.MX.sym("Pr", K, T)
        J0 = ca.MX.sym("J0", 1, T)
        inv_ds = ca.MX.sym("inv_ds")
        ctx = ca.MX.sym("ctx", 6)
        alpha_max = ca.MX.sym("alpha_max")

        rep_ds = ca.repmat(inv_ds, 1, T)
        nxt = phi(X[:, :T], A, Pk, J0, rep_ds, ca.repmat(ctx, 1, T))
        defects = X[:, 1:] - nxt
        g_path = path(X[:, 1:], Pl, Pr, J0, rep_ds, ca.repmat(ctx, 1, T))

        disc = ca.DM(spec.gamma ** np.arange(T)).T
        progress = (X[0, 1:] - X[0, :T]) / spec.progress_scale
        A_prev = ca.horzcat(ca.DM.zeros(2, 1), A[:, : T - 1])
        act_cost = spec.action_cost * ca.sum1((A - A_prev) ** 2)
        J = ca.sum2(disc * (progress - act_cost))

        g = ca.vertcat(ca.vec(defects), ca.vec(g_path[0:2, :]),
                       ca.vec(g_path[2:4, :]) / alpha_max)
        self._n_def = 8 * T
        P = ca.vertcat(ca.vec(Pk), ca.vec(Pl), ca.vec(Pr), ca.vec(J0), inv_ds, ctx, alpha_max)
        self.solver = ca.nlpsol("to_ocp", "ipopt", {"x": ca.veccat(X, A), "f": -J, "g": g, "p": P}, {
            "ipopt.max_iter": max_iter, "ipopt.tol": tol, "ipopt.print_level": print_level,
            "print_time": False, "ipopt.mu_strategy": "adaptive", "ipopt.linear_solver": "mumps",
        })
        self.T, self.K = T, K

    def _windows(self, coefs, sigma, n, closed):
        spec = self.spec
        j0 = np.floor(sigma / self._ds).astype(int) - spec.k_behind
        idx = j0[:, None] + np.arange(self.K)[None, :]
        if closed:
            k = np.mod(idx, n) + 1
        else:
            k = np.clip(idx, -1, n) + 1
        return [c[k].T for c in coefs], j0.astype(float)

    def solve(self, track: Track, x0: np.ndarray, init: tuple[np.ndarray, np.ndarray],
              ctx: np.ndarray | None = None, p: VehicleParams = F1_2025,
              max_rounds: int = 6) -> WindowResult:
        spec, T = self.spec, self.T
        if abs(track.ds - spec.ds) > 0.05:
            raise ValueError(f"track ds {track.ds} too far from the nominal {spec.ds} the window is sized for")
        self._ds = track.ds
        n = track.n_points
        coefs = [packed_coeffs(a, track.closed, n) for a in (track.kappa, track.w_left, track.w_right)]
        if ctx is None:
            cla = p.cla_f + p.cla_r
            ctx = np.array([p.m, p.p_max, cla, p.cda, p.mu_scale, p.cla_f / cla])
        xs, acts = (np.array(a, float) for a in init)
        xs[0] = x0
        lam_g = lam_x = None
        t_total, iters = 0.0, 0
        R = spec.radius
        for rnd in range(1, max_rounds + 1):
            sigma = xs[1:, 0]
            (Ck, Cl, Cr), j0 = self._windows(coefs, sigma, n, track.closed)
            P = np.concatenate([Ck.T.reshape(-1), Cl.T.reshape(-1), Cr.T.reshape(-1), j0,
                                [1.0 / track.ds], ctx, [p.alpha_max]])
            # Bounds: x0 fixed; s_t within its window; env termination set as boxes.
            lbX = np.tile(np.array([-np.inf, -np.inf, -0.5 * np.pi, 2.0, -25.0, -5.0, -p.delta_max, -1.0])[:, None], (1, T + 1))
            ubX = np.tile(np.array([np.inf, np.inf, 0.5 * np.pi, 120.0, 25.0, 5.0, p.delta_max, 1.0])[:, None], (1, T + 1))
            lbX[0, 1:] = sigma - R
            ubX[0, 1:] = sigma + R
            lbX[:, 0] = ubX[:, 0] = x0
            lbx = np.concatenate([lbX.T.reshape(-1), -np.ones(2 * T)])
            ubx = np.concatenate([ubX.T.reshape(-1), np.ones(2 * T)])
            lbg = np.concatenate([np.zeros(self._n_def), -np.inf * np.ones(2 * T), -np.ones(2 * T)])
            ubg = np.concatenate([np.zeros(self._n_def), np.zeros(2 * T), np.ones(2 * T)])
            w0 = np.clip(np.concatenate([xs.reshape(-1), acts.reshape(-1)]), lbx, ubx)
            kw = dict(x0=w0, p=P, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            if lam_g is not None:
                kw.update(lam_g0=lam_g, lam_x0=lam_x)
            t0 = time.perf_counter()
            sol = self.solver(**kw)
            t_total += time.perf_counter() - t0
            st = self.solver.stats()
            iters += int(st["iter_count"])
            w = np.asarray(sol["x"]).ravel()
            xs = w[: 8 * (T + 1)].reshape(T + 1, 8)
            acts = w[8 * (T + 1):].reshape(T, 2)
            lam_g, lam_x = np.asarray(sol["lam_g"]).ravel(), np.asarray(sol["lam_x"]).ravel()
            at_bound = np.max(np.abs(xs[1:, 0] - sigma)) > R - 1.0
            if not at_bound:
                break
        return WindowResult(x=xs, a=acts, value=float(-sol["f"]), success=bool(st["success"]) and not at_bound,
                            status=str(st["return_status"]) + (" (window bound active)" if at_bound else ""),
                            iterations=iters, solve_seconds=t_total, rounds=rnd,
                            meta={"spec": spec.__dict__, "track": track.name, "ctx": ctx.tolist()})


def window_return(xs: np.ndarray, acts: np.ndarray, spec: WindowSpec = WindowSpec()) -> float:
    T = len(acts)
    prev = np.vstack([np.zeros((1, 2)), acts[:-1]])
    r = (xs[1:, 0] - xs[:-1, 0]) / spec.progress_scale - spec.action_cost * np.sum((acts - prev) ** 2, 1)
    return float(np.sum(spec.gamma ** np.arange(T) * r))
