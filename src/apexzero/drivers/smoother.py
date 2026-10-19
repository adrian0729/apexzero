"""Rauch-Tung-Striebel smoother for progress along the lap, state [s, v, a].

The feed gives two noisy views of one car's progress, on two different ~3.7 Hz clocks:
positions (projected to s, centimetre-precise but sparse) and speed (whole km/h, and
measured along the car's own line rather than the centreline). Official lap and sector
times add a few exact constraints. A constant-acceleration model driven by white jerk
fuses all of them, and the RTS backward pass makes every estimate use the whole lap.

Measurements are scalar and processed in time order, so the two clocks never have to be
resampled onto each other:

- ``POS``:    z = s                         (projected position, unwrapped)
- ``SPEED``:  z = gain * v                  (gain = per-lap scale * path ratio at s)
- ``ANCHOR``: z = s with ~1 mm noise        (a timing line crossed at a known time)

Innovations beyond ``gate`` standard deviations are rejected (position glitches). Anchors
are never gated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POS, SPEED, ANCHOR = 0, 1, 2


@dataclass(frozen=True)
class SmootherConfig:
    jerk_psd: float = 400.0      # white-jerk spectral density (m^2/s^5); see tests and synth
    sigma_pos: float = 2.2       # along-track position noise (m), measured at Monza 2025
    sigma_speed: float = 0.35    # speed noise (m/s): 1 km/h quantisation plus model error
    sigma_anchor: float = 1e-3   # a timing line, effectively exact
    gate: float = 6.0            # innovation gate (standard deviations); anchors exempt
    p0: tuple[float, float, float] = (25.0**2, 10.0**2, 20.0**2)


DEFAULT_CONFIG = SmootherConfig()


@dataclass
class Smoothed:
    t: np.ndarray        # [K] epoch times (sorted)
    x: np.ndarray        # [K, 3] smoothed [s, v, a]
    p: np.ndarray        # [K, 3, 3] smoothed covariances
    used: np.ndarray     # [K] whether each measurement passed the gate
    kind: np.ndarray     # [K]
    z: np.ndarray        # [K]
    nis: float           # mean normalised innovation squared of accepted measurements

    def s_at(self, tq: np.ndarray) -> np.ndarray:
        """s(t) by cubic Hermite interpolation of (s, v) between epochs."""
        return hermite(self.t, self.x[:, 0], self.x[:, 1], np.asarray(tq, dtype=float))

    def v_at(self, tq: np.ndarray) -> np.ndarray:
        return np.interp(tq, self.t, self.x[:, 1])

    def sigma_s_at(self, tq: np.ndarray) -> np.ndarray:
        return np.sqrt(np.interp(tq, self.t, self.p[:, 0, 0]))

    def t_at(self, sq: np.ndarray, dt: float = 0.005) -> np.ndarray:
        """Inverse of s(t): the time at which the car reached each s (monotone in s)."""
        tt = np.arange(self.t[0], self.t[-1], dt)
        ss = np.maximum.accumulate(self.s_at(tt))
        return np.interp(np.asarray(sq, dtype=float), ss, tt)


def hermite(t: np.ndarray, s: np.ndarray, v: np.ndarray, tq: np.ndarray) -> np.ndarray:
    i = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 2)
    h = t[i + 1] - t[i]
    h = np.where(h > 0, h, 1.0)
    u = np.clip((tq - t[i]) / h, 0.0, 1.0)
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    return h00 * s[i] + h10 * h * v[i] + h01 * s[i + 1] + h11 * h * v[i + 1]


def _transition(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    f = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    d2, d3, d4, d5 = dt**2, dt**3, dt**4, dt**5
    qm = q * np.array([[d5 / 20, d4 / 8, d3 / 6], [d4 / 8, d3 / 3, d2 / 2], [d3 / 6, d2 / 2, dt]])
    return f, qm


def smooth(t: np.ndarray, kind: np.ndarray, z: np.ndarray, gain: np.ndarray | None = None,
           cfg: SmootherConfig = DEFAULT_CONFIG, x0: np.ndarray | None = None) -> Smoothed:
    """Filter forward, then RTS backward, over scalar measurements sorted by time.

    ``gain`` multiplies v in SPEED measurements (ignored for the others); it may depend on
    s, but it is passed in already evaluated (a fixed-point pass in the caller handles
    that), which keeps the measurement model linear.
    """
    order = np.argsort(t, kind="stable")
    t, kind, z = np.asarray(t, float)[order], np.asarray(kind)[order], np.asarray(z, float)[order]
    g = np.ones(len(t)) if gain is None else np.asarray(gain, float)[order]
    n = len(t)
    sig = np.where(kind == POS, cfg.sigma_pos, np.where(kind == SPEED, cfg.sigma_speed,
                                                        cfg.sigma_anchor))
    if x0 is None:
        first_pos = z[kind != SPEED][0] if np.any(kind != SPEED) else 0.0
        first_v = z[kind == SPEED][0] / g[kind == SPEED][0] if np.any(kind == SPEED) else 50.0
        x0 = np.array([first_pos, first_v, 0.0])
    xf = np.zeros((n, 3))
    pf = np.zeros((n, 3, 3))
    xp = np.zeros((n, 3))
    pp = np.zeros((n, 3, 3))
    fs = np.zeros((n, 3, 3))
    used = np.zeros(n, bool)
    x = np.asarray(x0, float).copy()
    p = np.diag(cfg.p0).astype(float)
    nis_sum, nis_n = 0.0, 0
    t_prev = t[0]
    for k in range(n):
        f, qm = _transition(t[k] - t_prev, cfg.jerk_psd)
        x = f @ x
        p = f @ p @ f.T + qm
        fs[k], xp[k], pp[k] = f, x, p
        h = np.array([1.0, 0.0, 0.0]) if kind[k] != SPEED else np.array([0.0, g[k], 0.0])
        innov = z[k] - h @ x
        ph = p @ h
        s_var = h @ ph + sig[k] ** 2
        if kind[k] == ANCHOR or innov * innov <= cfg.gate**2 * s_var:
            kg = ph / s_var
            x = x + kg * innov
            p = p - np.outer(kg, ph)
            p = 0.5 * (p + p.T)
            used[k] = True
            if kind[k] != ANCHOR:
                nis_sum += innov * innov / s_var
                nis_n += 1
        xf[k], pf[k] = x, p
        t_prev = t[k]
    xs = xf.copy()
    ps = pf.copy()
    for k in range(n - 2, -1, -1):
        f = fs[k + 1]
        c = pf[k] @ f.T @ np.linalg.inv(pp[k + 1])
        xs[k] = xf[k] + c @ (xs[k + 1] - xp[k + 1])
        ps[k] = pf[k] + c @ (ps[k + 1] - pp[k + 1]) @ c.T
    return Smoothed(t, xs, ps, used, kind, z, nis_sum / max(nis_n, 1))
