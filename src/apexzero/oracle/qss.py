"""Quasi-steady-state (QSS) lap-time estimate for a point mass on a given path.

The car is reduced to a speed-dependent g-g diagram built from the same vehicle
parameters as the full model:

    a_lat_max(v)   = mu_y * (m g + DF(v)) / m
    a_brake_max(v) = mu_x * (m g + DF(v)) / m + drag(v) / m
    a_acc_max(v)   = min(P / (m v), mu_x_r * Fz_r(v) / m) - drag(v) / m

combined through a friction ellipse. The speed profile is the classic forward-backward
pass: cornering limit from curvature, then acceleration forward and braking backward.
This is the fast oracle (milliseconds per track); the CasADi OCP is the ground truth it
is calibrated against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexzero.vehicle.params import VehicleParams


@dataclass(frozen=True)
class GG:
    """Point-mass g-g-v envelope derived from VehicleParams."""

    mu_x: float
    mu_y: float
    mu_x_r: float
    p: VehicleParams

    @classmethod
    def from_params(cls, p: VehicleParams) -> GG:
        # Friction at a representative per-tyre load (static + downforce at ~60 m/s).
        fz_tyre = 0.25 * (p.m * p.g + 0.5 * p.rho * (p.cla_f + p.cla_r) * 60.0**2)
        dmu = p.kmu * (fz_tyre - p.fz0)
        mu_x = p.mu_scale * (0.5 * (p.mux_f + p.mux_r) + dmu)
        mu_y = p.mu_scale * (0.5 * (p.muy_f + p.muy_r) + dmu)
        mu_x_r = p.mu_scale * (p.mux_r + dmu)
        return cls(mu_x, mu_y, mu_x_r, p)

    def normal_force(self, v):
        p = self.p
        return p.m * p.g + 0.5 * p.rho * (p.cla_f + p.cla_r) * v * v

    def drag(self, v):
        p = self.p
        return 0.5 * p.rho * p.cda * v * v + p.f_roll * self.normal_force(v)

    def a_lat_max(self, v):
        return self.mu_y * self.normal_force(v) / self.p.m

    def a_brake_max(self, v):
        return self.mu_x * self.normal_force(v) / self.p.m + self.drag(v) / self.p.m

    def a_acc_max(self, v):
        p = self.p
        fz_r = p.m * p.g * p.lf / (p.lf + p.lr) + 0.5 * p.rho * p.cla_r * v * v
        traction = self.mu_x_r * fz_r
        power = p.p_max / np.maximum(v, p.v_drive_floor)
        return (np.minimum(power, traction) - self.drag(v)) / p.m

    def v_corner(self, kappa, v_cap: float = 120.0):
        """Largest v with v^2 |kappa| <= a_lat_max(v) (closed form, downforce included)."""
        p = self.p
        c = 0.5 * p.rho * (p.cla_f + p.cla_r) / p.m
        k = np.abs(kappa)
        denom = k - self.mu_y * c
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.sqrt(self.mu_y * p.g / denom)
        return np.where(denom > 1e-9, np.minimum(v, v_cap), v_cap)


def speed_profile(kappa: np.ndarray, ds: np.ndarray | float, gg: GG, closed: bool = True,
                  v_start: float | None = None, v_cap: float = 120.0) -> np.ndarray:
    """Forward-backward QSS speed profile along a path with curvature kappa[i].

    ``ds`` is the spacing between consecutive samples (scalar or per-segment array of
    length N for closed paths / N-1 for open). Closed paths are swept twice so the
    profile is periodic.
    """
    kappa = np.asarray(kappa, dtype=float)
    n = len(kappa)
    dsv = np.broadcast_to(np.asarray(ds, dtype=float), (n if closed else n - 1,))
    v_lim = gg.v_corner(kappa, v_cap)

    def ellipse(v, k, a_max):
        lat = np.minimum((v * v * abs(k)) / gg.a_lat_max(v), 1.0)
        return a_max * np.sqrt(1.0 - lat * lat)

    reps = 2 if closed else 1
    m = n * reps
    idx = np.arange(m) % n

    # Forward pass (acceleration-limited).
    v = np.empty(m)
    v[0] = v_lim[0] if v_start is None else min(v_start, v_lim[0])
    for j in range(1, m):
        i_prev, i = idx[j - 1], idx[j]
        a = max(ellipse(v[j - 1], kappa[i_prev], gg.a_acc_max(v[j - 1])), 0.0)
        v[j] = min(v_lim[i], np.sqrt(v[j - 1] ** 2 + 2.0 * a * dsv[i_prev]))
    v_fwd = v[-n:] if closed else v

    # Backward pass (braking-limited).
    vb = np.empty(m)
    vb[-1] = v_fwd[-1]
    for j in range(m - 2, -1, -1):
        i, i_next = idx[j], idx[j + 1]
        a = ellipse(vb[j + 1], kappa[i_next], gg.a_brake_max(vb[j + 1]))
        vb[j] = min(v_fwd[i] if (not closed or j >= m - n) else v_lim[i],
                    np.sqrt(vb[j + 1] ** 2 + 2.0 * a * dsv[i]))
    v_bwd = vb[-n:] if closed else vb
    return np.minimum(v_fwd, v_bwd)


def lap_time(v: np.ndarray, ds: np.ndarray | float, closed: bool = True) -> float:
    v = np.asarray(v, dtype=float)
    if closed:
        v_next = np.roll(v, -1)
        dsv = np.broadcast_to(np.asarray(ds, dtype=float), v.shape)
    else:
        v_next = v[1:]
        v = v[:-1]
        dsv = np.broadcast_to(np.asarray(ds, dtype=float), v.shape)
    return float(np.sum(2.0 * dsv / (v + v_next)))
