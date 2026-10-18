"""Uniform cubic B-spline lookups for track quantities (kappa, widths): C2 in s.

The simulator and the training-objective OCP must evaluate *the same* smooth function of
arc length, so the dynamics are identical in JAX and CasADi and IPOPT sees continuous
second derivatives. Samples f_j at s_j = j * ds are interpolated by a uniform cubic
B-spline with coefficients c_j:

    f(s) = b0(t) c_{i-1} + b1(t) c_i + b2(t) c_{i+1} + b3(t) c_{i+2},  i = floor(s/ds), t = s/ds - i

with the standard uniform basis. Interpolation requires (c_{j-1} + 4 c_j + c_{j+1}) / 6 = f_j:
periodic for closed tracks (solved by FFT), natural ends (zero second derivative) for open
strips (a tridiagonal solve, done in JAX so generated strips stay on-device).

Coefficients are stored with an offset of one: ``coef[j + 1] = c_j`` for j = -1 .. N, in an
array of length P + 2 (P = padded point count).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def coeffs_closed_np(f: np.ndarray) -> np.ndarray:
    """Periodic interpolation coefficients c_0..c_{N-1} (NumPy, FFT)."""
    n = len(f)
    k = np.arange(n)
    eig = (4.0 + 2.0 * np.cos(2 * np.pi * k / n)) / 6.0
    return np.real(np.fft.ifft(np.fft.fft(f) / eig))


def coeffs_open_jax(f: jax.Array) -> jax.Array:
    """Natural-end interpolation coefficients c_{-1}..c_N (JAX, tridiagonal)."""
    n = f.shape[0]
    c0, cn1 = f[0], f[-1]
    rhs = 6.0 * f[1:-1]
    rhs = rhs.at[0].add(-c0).at[-1].add(-cn1)
    m = n - 2
    dl = jnp.concatenate([jnp.zeros(1, f.dtype), jnp.ones(m - 1, f.dtype)])
    du = jnp.concatenate([jnp.ones(m - 1, f.dtype), jnp.zeros(1, f.dtype)])
    d = jnp.full(m, 4.0, f.dtype)
    inner = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, None])[:, 0]
    c = jnp.concatenate([c0[None], inner, cn1[None]])
    cm1 = 2.0 * c[0] - c[1]
    cn = 2.0 * c[-1] - c[-2]
    return jnp.concatenate([cm1[None], c, cn[None]])


def coeffs_open_np(f: np.ndarray) -> np.ndarray:
    """Natural-end coefficients c_{-1}..c_N in float64 (SciPy banded solve)."""
    from scipy.linalg import solve_banded

    f = np.asarray(f, float)
    n = len(f)
    c = np.empty(n)
    c[0], c[-1] = f[0], f[-1]
    rhs = 6.0 * f[1:-1]
    rhs[0] -= c[0]
    rhs[-1] -= c[-1]
    m = n - 2
    ab = np.zeros((3, m))
    ab[0, 1:] = 1.0
    ab[1, :] = 4.0
    ab[2, :-1] = 1.0
    c[1:-1] = solve_banded((1, 1), ab, rhs)
    return np.r_[2.0 * c[0] - c[1], c, 2.0 * c[-1] - c[-2]]


def packed_coeffs(f: np.ndarray, closed: bool, pad_to: int) -> np.ndarray:
    """Coefficients in the stored layout: length pad_to + 2, coef[j + 1] = c_j."""
    n = len(f)
    out = np.zeros(pad_to + 2)
    if closed:
        c = coeffs_closed_np(np.asarray(f, float))
        out[1: n + 1] = c
        out[0] = c[-1]
        out[n + 1] = c[0]
        if n + 2 < len(out):
            out[n + 2] = c[1]
    else:
        out[: n + 2] = coeffs_open_np(f)
        out[n + 2:] = out[n + 1]
    return out


def basis(t):
    t2, t3 = t * t, t * t * t
    b0 = (1.0 - t) ** 3 / 6.0
    b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
    b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
    b3 = t3 / 6.0
    return b0, b1, b2, b3


def eval_jax(coef: jax.Array, s: jax.Array, ds, n, closed) -> jax.Array:
    """Evaluate a packed spline at arc length(s) s."""
    u = s / ds
    nf = n.astype(u.dtype) if hasattr(n, "astype") else jnp.asarray(n, u.dtype)
    u = jnp.where(closed, jnp.mod(u, nf), jnp.clip(u, 0.0, nf - 1.0))
    i = jnp.floor(u)
    t = u - i
    i = i.astype(jnp.int32)
    i = jnp.where(closed, i, jnp.minimum(i, n - 2))   # last interval for open strips
    t = jnp.where(closed, t, u - i.astype(u.dtype))
    b0, b1, b2, b3 = basis(t)

    def at(j):
        k = jnp.where(closed, jnp.mod(j, n), j)      # closed: wrap to 0..n-1; open: -1..n
        return coef[k + 1]

    return b0 * at(i - 1) + b1 * at(i) + b2 * at(i + 1) + b3 * at(i + 2)


def eval_casadi(coef: np.ndarray, ds: float, n: int, closed: bool):
    """CasADi Function s -> f(s) identical to ``eval_jax`` (uses casadi.bspline)."""
    import casadi as ca

    if closed:
        c = np.r_[coef[0], coef[1: n + 1], coef[1], coef[2]]   # c_{-1}..c_{n+1}
    else:
        c = coef[: n + 2]                                        # c_{-1}..c_n
    m = len(c)
    knots = [(k - 3) * ds for k in range(m + 4)]
    s = ca.MX.sym("s")
    if closed:
        L = n * ds
        x = s - L * ca.floor(s / L)
    else:
        x = ca.fmin(ca.fmax(s, 0.0), (n - 1) * ds)
    val = ca.bspline(x, ca.DM(c), [knots], [3], 1, {})
    return ca.Function("spline", [s], [val])


def values_from_coeffs(coef: np.ndarray, n: int) -> np.ndarray:
    """Spline values at the n nodes: (c_{j-1} + 4 c_j + c_{j+1}) / 6 (open or closed layout)."""
    c = np.asarray(coef, float)
    return (c[0:n] + 4.0 * c[1:n + 1] + c[2:n + 2]) / 6.0


def spline_np(coef: np.ndarray, s: np.ndarray, ds: float, n: int, closed: bool) -> np.ndarray:
    """NumPy evaluation (float64), identical to ``eval_jax``."""
    u = np.asarray(s, float) / ds
    u = np.mod(u, n) if closed else np.clip(u, 0.0, n - 1.0)
    i = np.floor(u).astype(int)
    if not closed:
        i = np.minimum(i, n - 2)
    t = u - i
    b0, b1, b2, b3 = basis(t)

    def at(j):
        k = np.mod(j, n) if closed else j
        return coef[k + 1]

    return b0 * at(i - 1) + b1 * at(i) + b2 * at(i + 1) + b3 * at(i + 2)
