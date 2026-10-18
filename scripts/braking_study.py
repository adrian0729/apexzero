"""K3: the 1-D braking problem, Pontryagin vs value iteration across discount factors.

    uv run python scripts/braking_study.py

Writes docs/findings/braking-gamma.md and docs/findings/braking-gamma.png.
"""

from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from apexzero.learn.braking import BrakingProblem, pontryagin_switch, rollout, value_iteration
from apexzero.oracle.qss import GG
from apexzero.vehicle.params import F1_2025

GAMMAS = (0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)


def main():
    prob = BrakingProblem(n_actions=41)
    gg = GG.from_params(F1_2025)
    s_star = pontryagin_switch(prob, gg)
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for g in GAMMAS:
        sg, vg, _, pol = value_iteration(prob, gg, g, ns=801, nv=361)
        traj, brake_at, crashed = rollout(prob, sg, vg, pol)
        rows.append((g, brake_at, traj[-1, 1], crashed))
        print(f"gamma {g}: brake at {brake_at}, corner {traj[-1, 1] * 3.6:.1f} km/h, crashed {crashed}", flush=True)
        if g in (0.8, 0.9, 0.95, 0.995):
            ax[0].plot(traj[:, 0], traj[:, 1] * 3.6, label=f"gamma {g} ({0.05 / (1 - g):.1f} s horizon)")
    ax[0].axvline(prob.s_c, color="k", lw=0.8)
    ax[0].axhline(prob.v_c * 3.6, color="r", lw=0.8, ls="--", label="corner limit")
    ax[0].axvline(s_star, color="g", lw=0.8, ls=":", label=f"Pontryagin switch ({s_star:.1f} m)")
    ax[0].set_xlabel("distance (m)")
    ax[0].set_ylabel("speed (km/h)")
    ax[0].set_xlim(250, prob.s_c + 5)
    ax[0].legend(fontsize=7)
    g_arr = np.array([r[0] for r in rows])
    b_arr = np.array([np.nan if r[1] is None else r[1] for r in rows])
    ax[1].semilogx(0.05 / (1 - g_arr), b_arr, "o-")
    for (g, b, v, c), h in zip(rows, 0.05 / (1 - g_arr)):
        if c:
            ax[1].plot(h, b, "rx", ms=10)
    ax[1].axhline(s_star, color="g", ls=":", label="Pontryagin switch")
    ax[1].set_xlabel("effective horizon 0.05 / (1 - gamma) (s)")
    ax[1].set_ylabel("braking point (m)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = Path("docs/findings")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "braking-gamma.png", dpi=150)

    intro = (f"Point mass on the car's g-g envelope, {prob.v0 * 3.6:.0f} km/h at s = 0, a "
             f"{prob.v_c * 3.6:.0f} km/h corner at {prob.s_c:.0f} m. Pontryagin's bang-bang "
             f"solution switches from full throttle to full braking at **{s_star:.1f} m**.")
    lines = ["# The 1-D braking problem: Pontryagin vs value iteration", "", intro,
             "Value iteration solves the discounted RL objective the agent uses (progress / 10 per",
             "50 ms step, the env's crash penalty, full-throttle progress after a safe corner) on an",
             "801 x 361 (s, v) grid with 41 acceleration levels; red crosses mark crashes.", "",
             "| gamma | Horizon | Braking point | Corner speed | Crashed |", "|---|---|---|---|---|"]
    for g, b, v, c in rows:
        lines.append(f"| {g} | {0.05 / (1 - g):.1f} s | {'-' if b is None else f'{b:.1f} m'} | {v * 3.6:.1f} km/h | {'yes' if c else 'no'} |")
    lines += ["", "![braking](braking-gamma.png)", "",
              "Corner speeds within about 1 km/h of the limit are at the grid's resolution (0.25 m/s),",
              "so a crash flagged at ~91 km/h is a resolution effect, not a horizon effect.", "",
              "Reading: a crash forfeits all future progress, so its true cost is the discounted",
              "value of everything after the corner. With a short horizon that value is small, the",
              "progress gained by braking later wins, and the agent arrives too fast. The RL",
              "environment's gamma = 0.995 (10 s horizon) is long enough that the discounted",
              "optimum brakes where the minimum-time optimum does, up to grid resolution."]
    (out / "braking-gamma.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
