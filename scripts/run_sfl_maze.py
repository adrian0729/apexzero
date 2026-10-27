"""Run SFL's own minigrid_sfl.py (vendored, unmodified) with a compatibility shim, for maze E5.

    uv run python scripts/run_sfl_maze.py SEED=1 env.AGENT_VIEW_SIZE=13 \\
        learning.TOTAL_TIMESTEPS=1966080000 learning.NUM_CHECKPOINTS=300 \\
        --save-updates 300,400,600,...,30000

The positional ``key.path=value`` arguments are Hydra's overrides of SFL's composed config
(``minigrid-sfl.yaml`` with its ``env`` and ``learning`` groups; ``apexzero.maze.sfl``
composes it). The working directory is the run directory: the shim writes ``config.json``
(the overrides and the composed config), ``sfl-eval.jsonl`` (one line per refresh block: the
update count, its wall-clock time and the script's own eval and learnability statistics),
and, at each ``--save-updates`` update, ``sfl-checkpoints/u<update>/`` (the parameters and the
learnable set D in force; DEV-E5-2). It exits 1 if a selected update was not saved.

What the shim does instead of editing the file (as run_jaxued_maze.py does for JaxUED):
- Hydra and OmegaConf are not installed. ``hydra.main`` becomes the identity, the shim calls
  ``main(config)`` with the composed config, and ``OmegaConf.to_container`` copies it.
- ``sfl.train.train_utils`` (not vendored) is a stand-in whose ``save_params`` is E5's
  checkpoint writer. With ``learning.NUM_CHECKPOINTS`` = updates / EVAL_FREQ the script calls
  it after every refresh block, and the writer keeps the selected updates only.
- ``sfl.train.minigrid_plr`` (SFL's network and ``evaluate_rnn``) is the vendored file,
  hash-checked, with the three SFL files it imports (``sfl.MINIGRID_PLR_IMPORTS``).
- ``jax.tree_map`` (removed from JAX) comes from ``apexzero.compat``; the script's 0/1
  integer ``where=`` masks, which today's JAX refuses, are cast to booleans
  (``sfl.install_mask_where``: what older JAX did with them); and matplotlib's
  ``FigureCanvasAgg.tostring_rgb`` (removed in matplotlib 3.10), which the script's buffer
  figure calls, is restored from ``buffer_rgba``. Each buffer figure is closed once logged
  (the script never closes them: 300 refreshes would hold 300 figures).
- ``wandb.Artifact`` is a no-op: the script uploads each saved model, and ``add_file`` would
  fail on the safetensors path the stand-in ``save_params`` does not write. wandb stays in
  the config's own mode (``WANDB_MODE: disabled``).
- **The learnable set D.** The script never returns D from its jitted step, so the shim gives
  the module a view of ``jax`` whose ``jit`` wraps one function, ``get_learnability_set``,
  so that its outputs (D's scores and levels) also go to an ``io_callback``. Nothing else is
  traced differently. The scores captured each block must equal the
  ``learnability_set_scores`` the script logs for that block, or the run stops.
  ``--no-capture`` turns this off, for the test that the parameters are bit-identical with
  and without it (DEV-E5-2).
"""

import argparse
import copy
import functools
import json
import os
import time
import types
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)

os.environ.setdefault("WANDB_SILENT", "true")

import jax
import jax.experimental
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg

import apexzero.compat  # noqa: F401  (restores jax.tree_map before jaxued is imported)
from apexzero.maze import sfl

SCRIPT = sfl.VENDOR / "sfl" / "train" / "minigrid_sfl.py"
EVAL_KEYS = ("solve_rate/mean", "return/mean", "learnability_set_mean_score", "time_delta",
             "steps_per_section")

if not hasattr(FigureCanvasAgg, "tostring_rgb"):
    FigureCanvasAgg.tostring_rgb = lambda self: np.asarray(self.buffer_rgba())[..., :3].tobytes()


sfl.install_mask_where()       # SFL's integer where= masks under today's JAX


class _NoArtifact:
    def __init__(self, *args, **kwargs):
        pass

    def add_file(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


class Recorder:
    """What the hooks share: the last logged update, the D captured for its block, the
    updates to save and the per-block log."""

    def __init__(self, out: Path, save_updates, eval_freq: int, capture: bool):
        self.out, self.eval_freq, self.capture = out, eval_freq, capture
        self.save_updates, self.saved = set(save_updates), []
        self.update, self.learnable, self.blocks = None, None, 0
        self.log = open(out / "sfl-eval.jsonl", "a")   # noqa: SIM115 (closed in close())

    def close(self):
        self.log.close()

    def on_learnable(self, scores, levels):
        self.learnable = (np.asarray(scores), {k: np.asarray(getattr(levels, k))
                                               for k in sfl.LEVEL_FIELDS})

    def on_log(self, data: dict):
        if "maps" in data:
            plt.close("all")
        if "update_count" not in data:
            return
        self.update, self.blocks = int(data["update_count"]), self.blocks + 1
        logged = np.asarray(data["learnability_set_scores"])
        if self.capture and (self.learnable is None
                             or not np.array_equal(self.learnable[0], logged)):
            raise RuntimeError(f"update {self.update}: the captured D is not the logged one")
        line = {"update": self.update, "time": time.time(),
                **{k: float(np.asarray(data[k])) for k in EVAL_KEYS if k in data}}
        self.log.write(json.dumps(line) + "\n")
        self.log.flush()

    def save_params(self, params, path):
        u = self.update
        if u is None or u not in self.save_updates or u in self.saved:
            return
        learnable = None
        if self.capture:
            scores, levels = self.learnable
            learnable = {**levels, "scores": scores, "refresh_update": u - self.eval_freq,
                         "update": u}
        sfl.save_checkpoint(self.out, u, params, learnable)
        self.saved.append(u)
        print(f"E5: saved update {u}" + (" with D" if self.capture else ""), flush=True)


class _JaxView(types.ModuleType):
    """``jax`` as minigrid_sfl.py sees it: everything is jax's own, except that ``jit`` of
    ``get_learnability_set`` also sends its outputs to the recorder."""

    def __init__(self, rec: Recorder):
        super().__init__("jax")
        self._rec = rec

    def __getattr__(self, name):
        return getattr(jax, name)

    def jit(self, fun=None, **kwargs):
        if fun is None:
            return functools.partial(self.jit, **kwargs)
        jitted = jax.jit(fun, **kwargs)
        if getattr(fun, "__name__", "") != "get_learnability_set" or not self._rec.capture:
            return jitted

        @functools.wraps(fun)
        def get_learnability_set(*args):
            scores, levels = jitted(*args)
            jax.experimental.io_callback(self._rec.on_learnable, None, scores, levels,
                                         ordered=True)
            return scores, levels

        return get_learnability_set


def load_module(rec: Recorder, network_module=None):
    """minigrid_sfl.py as a module, with the stand-ins above (``sfl.stub_modules``).
    ``network_module`` replaces sfl.train.minigrid_plr (an alternative network for a test; none
    uses it now that SFL's own is complete); otherwise the vendored file is used."""
    stubs = sfl.stub_modules(rec.save_params)
    if network_module is not None:
        stubs["sfl.train.minigrid_plr"] = network_module
    else:
        sfl.check_minigrid_plr()
    mod = sfl.load_vendored(SCRIPT, "sfl_minigrid_sfl", stubs)
    mod.jax = _JaxView(rec)
    return mod


def run(overrides: list[str], save_updates=(), capture: bool = True, out: Path | None = None,
        network_module=None) -> list[int]:
    """Compose the config, record it, and run the vendored ``main``; returns the updates
    saved."""
    out = Path.cwd() if out is None else Path(out)
    blobs = sfl.check_vendored(network=network_module is None)
    cfg = sfl.compose_config(overrides)
    n, freq = sfl.num_updates(cfg), int(cfg["learning"]["EVAL_FREQ"])
    # The script calls save_params after every block but the first (eval_step > 0, L736).
    bad = sorted(u for u in save_updates if u % freq or not freq < u <= n)
    if bad:
        raise SystemExit(f"--save-updates {bad}: not multiples of EVAL_FREQ {freq} in "
                         f"({freq}, {n}]")
    if save_updates and n // freq // int(cfg["learning"]["NUM_CHECKPOINTS"]) != 1:
        raise SystemExit("--save-updates needs learning.NUM_CHECKPOINTS = NUM_UPDATES // "
                         f"EVAL_FREQ = {n // freq}, so save_params is called every block")
    out.mkdir(parents=True, exist_ok=True)
    record = {"overrides": list(overrides), "save_updates": sorted(save_updates),
              "capture_learnable_set": capture, "num_updates": n, "vendored": blobs,
              "network": "vendored" if network_module is None else "stand-in", "config": cfg}
    (out / "config.json").write_text(json.dumps(record, indent=1) + "\n")
    rec = Recorder(out, save_updates, freq, capture)
    real_init, real_artifact = wandb.init, wandb.Artifact

    def init(*args, **kwargs):
        result = real_init(*args, **kwargs)
        real_log = wandb.log

        def log(data, *a, **kw):
            rec.on_log(data)
            return real_log(data, *a, **kw)

        wandb.log = log
        return result

    wandb.init, wandb.Artifact = init, _NoArtifact
    try:
        mod = load_module(rec, network_module)
        cwd = os.getcwd()
        os.chdir(out)
        try:
            mod.main(copy.deepcopy(cfg))
        finally:
            os.chdir(cwd)
    finally:
        wandb.init, wandb.Artifact = real_init, real_artifact
        rec.close()
        plt.close("all")
    missing = sorted(set(save_updates) - set(rec.saved))
    if missing:
        raise SystemExit(f"updates {missing} were not saved")
    return rec.saved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("overrides", nargs="*", help="Hydra-style key.path=value overrides")
    ap.add_argument("--save-updates", default="",
                    help="comma-separated updates at which to save the parameters and D")
    ap.add_argument("--no-capture", action="store_true",
                    help="do not capture D (the bit-identity test's other arm)")
    a = ap.parse_args(argv)
    updates = [int(u) for u in a.save_updates.split(",") if u]
    saved = run(a.overrides, updates, capture=not a.no_capture)
    print(f"E5: done, {len(saved)} checkpoints saved", flush=True)


if __name__ == "__main__":
    main()
