"""Throughput of the vectorised simulator: control steps per second vs batch size.

    uv run python scripts/bench_sim.py
"""

import time

import _bootstrap  # noqa: F401  (makes src/ importable)
import jax
import jax.numpy as jnp
import numpy as np

from apexzero.tracks.track import load_tumftm
from apexzero.vehicle.params import F1_2025
from apexzero.vehicle.sim import control_step, track_arrays


def bench(batch: int, steps: int = 200, substeps: int = 10) -> float:
    tr = track_arrays(load_tumftm("data/tracks/tumftm/Monza.csv"))
    rng = np.random.default_rng(0)
    x = np.zeros((batch, 8), dtype=np.float32)
    x[:, 0] = rng.uniform(0, 5000, batch)
    x[:, 3] = rng.uniform(30, 80, batch)
    x = jnp.asarray(x)
    u = jnp.asarray(rng.uniform(-0.2, 0.5, (batch, 2)).astype(np.float32))
    p = F1_2025

    @jax.jit
    def rollout(x):
        def body(xc, _):
            return jax.vmap(lambda xi, ui: control_step(xi, ui, tr, p, 0.05, substeps))(xc, u), None
        return jax.lax.scan(body, x, None, length=steps)[0]

    rollout(x).block_until_ready()
    t0 = time.perf_counter()
    rollout(x).block_until_ready()
    dt = time.perf_counter() - t0
    return batch * steps / dt


if __name__ == "__main__":
    print(f"devices: {jax.devices()}")
    for b in (256, 1024, 2048, 4096, 8192):
        print(f"batch {b:5d}: {bench(b) / 1e6:6.2f} M control steps/s (10 RK2 substeps each)")
