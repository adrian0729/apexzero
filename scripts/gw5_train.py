"""GW-5's trainer: scripts/train_dr.py with two opt-in, training-only levers (EXPLORATORY,
docs/findings/gw5-plan.md).

    uv run python scripts/gw5_train.py [--train-edge-margin M] [--local-bumps] <train_dr.py args>

train_dr.py runs unchanged. For this process only, the wrapper replaces two names that it
calls:
- ``--train-edge-margin M``: ``train_dr.procedural_env`` gets ``EnvParams(edge_margin=M)``,
  so the training environment terminates M metres from each edge (and draws its start offsets
  inside that margin). The end-of-run R-dev evaluation calls ``timed_laps`` with the default
  EnvParams, edge_margin 1.0, as for every other run.
- ``--local-bumps``: ``adapters.generate`` adds local Gaussian width bumps to every generated
  strip. Each side gets ``BUMPS_PER_SIDE`` bumps with centres uniform on the strip, sigma
  log-uniform in ``SIGMA_M`` (so a bump's +-sigma span is 4-20 m) and amplitude uniform in
  +-``AMP_M`` m. A width is never pushed below min(its own value, ``W_MIN_M``), and the
  generator's inside cap is applied again. The base strip is the one the generator would
  give for the same key (the bumps use ``fold_in(key, BUMP_STREAM)``).

No file under src/ and no frozen script changes, so the E0 and lockbox code paths stay
byte-identical. The levers are recorded in runs/<name>/gw5.json; train_dr.py's own
config.json records its arguments as usual (its "env" block is the default EnvParams).
"""

import argparse
import functools
import json
import math
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import jax
import jax.numpy as jnp
import train_dr

from apexzero.env import adapters, racing
from apexzero.tracks import generator

BUMPS_PER_SIDE = 96        # about one per 85 m per side on a 4,096-point, 2 m strip
SIGMA_M = (2.0, 10.0)      # log-uniform
AMP_M = 1.0                # uniform in [-AMP_M, AMP_M]
W_MIN_M = 2.5
BUMP_STREAM = 0x6757


def bump_profile(key, n: int, ds: float):
    """[2, n]: the sum of BUMPS_PER_SIDE Gaussian bumps per side, accumulated one bump at a
    time (a scan, so a vmapped bank never holds a [bumps, n] array per strip)."""
    kc, ks, ka = jax.random.split(key, 3)
    shape = (BUMPS_PER_SIDE, 2)
    centre = jax.random.uniform(kc, shape, minval=0.0, maxval=n * ds)
    sigma = jnp.exp(jax.random.uniform(ks, shape, minval=math.log(SIGMA_M[0]),
                                       maxval=math.log(SIGMA_M[1])))
    amp = jax.random.uniform(ka, shape, minval=-AMP_M, maxval=AMP_M)
    s = jnp.arange(n) * ds

    def add(acc, b):
        c, sg, a = b
        return acc + a[:, None] * jnp.exp(-0.5 * ((s[None, :] - c[:, None]) / sg[:, None]) ** 2), None

    out, _ = jax.lax.scan(add, jnp.zeros((2, n)), (centre, sigma, amp))
    return out


def bumped(generate):
    def gen(key, g, n_points=generator.N_POINTS, ds=generator.DS, *args, **kw):
        kappa, wl, wr = generate(key, g, n_points, ds, *args, **kw)
        b = bump_profile(jax.random.fold_in(key, BUMP_STREAM), kappa.shape[0], ds)
        wl2 = jnp.maximum(wl + b[0], jnp.minimum(wl, W_MIN_M))
        wr2 = jnp.maximum(wr + b[1], jnp.minimum(wr, W_MIN_M))
        cap = 1.0 + g.margin / jnp.maximum(jnp.abs(kappa), 1e-6)
        wl2 = jnp.where(kappa > 0, jnp.minimum(wl2, cap), wl2)
        wr2 = jnp.where(kappa < 0, jnp.minimum(wr2, cap), wr2)
        return kappa, wl2, wr2
    return gen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], add_help=False)
    ap.add_argument("--train-edge-margin", type=float, default=None)
    ap.add_argument("--local-bumps", action="store_true")
    mine, rest = ap.parse_known_args()
    levers = {"train_edge_margin": mine.train_edge_margin, "local_bumps": mine.local_bumps}
    if mine.train_edge_margin is not None:
        if mine.train_edge_margin <= 0:
            raise SystemExit("--train-edge-margin must be positive")
        params = racing.EnvParams(edge_margin=mine.train_edge_margin)
        train_dr.procedural_env = functools.partial(adapters.procedural_env, params=params)
        levers["train_env_params"] = params._asdict()
    if mine.local_bumps:
        adapters.generate = bumped(adapters.generate)
        levers["bumps"] = {"per_side": BUMPS_PER_SIDE, "sigma_m": list(SIGMA_M), "amp_m": AMP_M,
                           "w_min_m": W_MIN_M, "stream": BUMP_STREAM}
    name = rest[rest.index("--name") + 1] if "--name" in rest else "dr-pilot"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "gw5.json").write_text(json.dumps({"levers": levers, "train_dr_args": rest},
                                             indent=1))
    sys.argv = [str(Path(train_dr.__file__)), *rest]
    train_dr.main()


if __name__ == "__main__":
    main()
