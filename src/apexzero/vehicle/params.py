"""Vehicle parameters for an F1-style car (2022-25 regulations).

Sources:
- TUMFTM laptime-simulation F1 parameter set (geometry, aero, power, tyre friction and
  load sensitivity). TUMFTM's own file uses m = 733 kg (2017 car).
- m = 800 kg is the FIA 2025 minimum mass (car + driver, no fuel).
- Iz, Pacejka B/C, steering limits, actuator lags, brake bias and maximum brake force are
  assumptions; each gets a +/-30% sensitivity run before any result is reported.
"""

from __future__ import annotations

from typing import NamedTuple


class VehicleParams(NamedTuple):
    m: float = 800.0             # kg, FIA 2025 minimum mass
    iz: float = 1100.0           # kg m^2 (assumed)
    lf: float = 1.968            # m, CoG to front axle (TUMFTM)
    lr: float = 1.632            # m, CoG to rear axle (TUMFTM)
    h_cog: float = 0.335         # m (TUMFTM)
    cda: float = 1.56            # m^2, drag area, DRS closed (TUMFTM)
    cla_f: float = 2.20          # m^2, front downforce area (TUMFTM)
    cla_r: float = 2.68          # m^2, rear downforce area (TUMFTM)
    rho: float = 1.18            # kg/m^3 (TUMFTM)
    p_max: float = 695e3         # W (TUMFTM)
    f_roll: float = 0.03         # rolling resistance coefficient on total normal load (TUMFTM)
    mux_f: float = 1.65          # longitudinal friction, front, at fz0 (TUMFTM)
    muy_f: float = 1.85          # lateral friction, front, at fz0 (TUMFTM)
    mux_r: float = 1.95          # longitudinal friction, rear, at fz0 (TUMFTM)
    muy_r: float = 2.15          # lateral friction, rear, at fz0 (TUMFTM)
    fz0: float = 3000.0          # N per tyre, nominal load for the friction values (TUMFTM)
    kmu: float = -5e-5           # 1/N, friction load sensitivity per tyre (TUMFTM)
    mu_scale: float = 1.0        # global grip multiplier (track condition, calibration)
    pac_b: float = 12.0          # Pacejka stiffness factor (assumed; peak slip ~7 deg)
    pac_c: float = 1.6           # Pacejka shape factor (assumed)
    delta_max: float = 0.45      # rad, road-wheel steering lock (assumed)
    tau_steer: float = 0.05      # s, steering first-order lag (assumed)
    steer_rate_max: float = 1.5  # rad/s, steering rate limit (assumed)
    tau_pedal: float = 0.05      # s, throttle/brake first-order lag (assumed)
    brake_bias: float = 0.58     # front share of brake force (assumed)
    f_brake_max: float = 60e3    # N, total brake force demand at full pedal (assumed)
    v_drive_floor: float = 5.0   # m/s, P/v drive force is evaluated at max(v, floor)
    alpha_max: float = 0.175     # rad (10 deg), tyre slip beyond which control is lost:
                                 # the env terminates and the OCP constrains at this limit
    g: float = 9.81

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    def replace(self, **kw) -> VehicleParams:
        return self._replace(**kw)


F1_2025 = VehicleParams()

# Car-context features observed by the policy, with training ranges (plan section 4).
CONTEXT_FIELDS = ("m", "p_max", "cla", "cda", "mu_scale", "aero_balance")
CONTEXT_RANGES = {
    "m": (750.0, 850.0),
    "p_max": (550e3, 800e3),
    "cla": (3.0, 6.0),
    "cda": (1.0, 1.9),
    "mu_scale": (0.85, 1.10),
    "aero_balance": (F1_2025.cla_f / (F1_2025.cla_f + F1_2025.cla_r) - 0.05,
                     F1_2025.cla_f / (F1_2025.cla_f + F1_2025.cla_r) + 0.05),
}


def with_context(base: VehicleParams, m, p_max, cla, cda, mu_scale, aero_balance):
    """Build parameters from the six context features (works with floats or arrays)."""
    return base._replace(
        m=m, p_max=p_max, cda=cda, mu_scale=mu_scale,
        cla_f=cla * aero_balance, cla_r=cla * (1.0 - aero_balance),
    )
