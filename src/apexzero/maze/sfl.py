"""SFL's MiniGrid maze code for maze E5: the vendored source, its config, its checkpoints.

The subject is SFL's own training script (``third_party/sfl/sfl/train/minigrid_sfl.py``,
vendored unmodified from amacrutherford/sampling-for-learnability at the commit below), run
through ``scripts/run_sfl_maze.py``. This module holds what the run, the manifest and the audit
share:

- **The vendored files and their git blob SHA-1s** (``check_vendored``), checked before every
  E5 stage, as the draft's Section 2 requires. ``minigrid_sfl.py`` imports two modules of the
  SFL repository that are not vendored (``NOT_VENDORED``): ``sfl.train.minigrid_plr``, for the
  network (``ActorCritic``) and ``evaluate_rnn``, and ``sfl.train.train_utils``, for
  ``save_params``. The shim replaces ``save_params`` with E5's checkpoint writer, so
  ``train_utils`` is never needed. ``minigrid_plr.py`` is: without it there is no SFL network,
  and its download needs the user's OK (``minigrid_plr`` raises until it is vendored).
- **The config** (``compose_config``): Hydra's composition of ``minigrid-sfl.yaml`` (defaults
  ``env: minigrid``, ``learning: ippo-minigrid``, then ``_self_``), with OmegaConf's YAML rules
  (``3e8`` is a float, as OmegaConf reads it; plain PyYAML would read a string), and
  ``key.path=value`` overrides of existing keys, as Hydra's command line takes them. Neither
  Hydra nor OmegaConf is installed; nothing else of them is used.
- **The checkpoints** E5 adds (DEV-E5-2): at each selected update, the parameters
  (``params.msgpack``, flax's msgpack) and the learnable set D in force (``learnable.npz``:
  the K levels, their learnability from the last refresh and that refresh's update), under
  ``<run dir>/sfl-checkpoints/u<update>/``.
- **SFL's outcome counting** (``sfl_outcomes_reference``, a line-for-line port of
  ``_calc_outcomes_by_agent``, minigrid_sfl.py L218-244) and the audit's vectorised version
  (``native_learnability``), which must agree on test vectors; and SFL's top-K
  (``reference_top_k``: ``argsort(l)[-K:]``, ties to the highest index, the score spec's S-1
  and S-2; NaN raises, S-3).
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "third_party" / "sfl"
SFL_REPO = "amacrutherford/sampling-for-learnability"
SFL_COMMIT = "8b14e2ec07b38dc33a9df476b84faa2f3ca5fd3b"

# Path (relative to third_party/sfl/) -> git blob SHA-1, from the tree at SFL_COMMIT.
VENDORED = {
    "LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
    "sfl/__init__.py": "b3c06d488393abb3b3829e5590d42409c995b4cf",
    "sfl/train/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "sfl/train/minigrid_sfl.py": "6436f5f659fa5fd0992c3ce479054eacb0353fdd",
    "sfl/train/config/minigrid-sfl.yaml": "41a016967485abcc8001b24ec91ddca783e54e6e",
    "sfl/train/config/learning/ippo-minigrid.yaml": "633e1793c235cce1f3b046075d7f14a1d68ac67d",
    "sfl/train/config/env/minigrid.yaml": "8df39e2c6881321897748e2450b51d43f3ad5b5c",
}
# Imported by minigrid_sfl.py but not vendored (not in the 27 Oct 2026 approval): path ->
# (git blob SHA-1, bytes), from the same tree listing. Only minigrid_plr.py is needed.
NOT_VENDORED = {
    "sfl/train/minigrid_plr.py": ("9537a794b7df09c76bf816215b366b03eb10b682", 38737),
    "sfl/train/train_utils/__init__.py": ("b5437dc2e9f5ed3a6700b5414702fe5ba7902394", 44),
    "sfl/train/train_utils/common.py": ("be4f1f26619dc302aae1a23882ed287c93ebc01c", 6308),
}
MINIGRID_PLR = "sfl/train/minigrid_plr.py"
# JaxUED's maze example (third_party/jaxued_examples/README.md), checked with SFL's files.
JAXUED = {"third_party/jaxued_examples/maze_plr.py": "a7b6164dc9d9772613d9cc2dab0ef7e9603dc7d0"}

CONFIG_DIR = VENDOR / "sfl" / "train" / "config"
CKPT_DIR = "sfl-checkpoints"
LEVEL_FIELDS = ("wall_map", "goal_pos", "agent_pos", "agent_dir", "width", "height")


class VendorMismatch(RuntimeError):
    """A vendored file is missing or is not the pinned commit's blob."""


class MissingSFLSource(RuntimeError):
    """A file SFL's code imports is not vendored (its download needs the user's OK)."""


def git_blob_sha1(data: bytes) -> str:
    """``git hash-object`` of ``data``: SHA-1 over ``blob <size>\\0`` and the bytes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check_files(expected: dict[str, str], base: Path) -> dict[str, str]:
    """Every file in ``expected`` (relative to ``base``) is present with its blob SHA-1."""
    bad = []
    for rel, blob in expected.items():
        path = base / rel
        if not path.is_file():
            bad.append(f"{rel}: missing")
        elif (got := git_blob_sha1(path.read_bytes())) != blob:
            bad.append(f"{rel}: blob {got[:12]}, expected {blob[:12]}")
    if bad:
        raise VendorMismatch("vendored files differ from their pinned blobs: " + "; ".join(bad))
    return dict(expected)


def check_vendored(network: bool = False) -> dict[str, str]:
    """JaxUED's maze_plr.py and SFL's vendored files, each against its pinned blob; with
    ``network``, also SFL's minigrid_plr.py (``MissingSFLSource`` if it is not vendored)."""
    out = dict(check_files(JAXUED, ROOT))
    out.update({f"third_party/sfl/{k}": v for k, v in check_files(VENDORED, VENDOR).items()})
    if network:
        out[f"third_party/sfl/{MINIGRID_PLR}"] = check_minigrid_plr()
    return out


def check_minigrid_plr() -> str:
    blob, size = NOT_VENDORED[MINIGRID_PLR]
    if not (VENDOR / MINIGRID_PLR).is_file():
        raise MissingSFLSource(
            f"third_party/sfl/{MINIGRID_PLR} is not vendored: minigrid_sfl.py imports SFL's "
            f"network (ActorCritic) and evaluate_rnn from it. Its download ({size:,} bytes, blob "
            f"{blob[:12]}, {SFL_REPO} at {SFL_COMMIT[:12]}) needs the user's OK (maze-e5.md "
            "Section 10)")
    check_files({MINIGRID_PLR: blob}, VENDOR)
    return blob


# --------------------------------------------------------------------------- the config

_FLOAT = re.compile(r"""^(?:
     [-+]?[0-9]+(?:_[0-9]+)*\.[0-9]*(?:[eE][-+]?[0-9]+)?
    |[-+]?[0-9]+(?:_[0-9]+)*(?:[eE][-+]?[0-9]+)
    |\.[0-9]+(?:_[0-9]+)*(?:[eE][-+][0-9]+)?
    |[-+]?[0-9]+(?:_[0-9]+)*(?::[0-5]?[0-9])+\.[0-9]*
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""", re.VERBOSE)


class OmegaConfLoader(yaml.SafeLoader):
    """PyYAML's safe loader with OmegaConf's changes (omegaconf ``get_yaml_loader``): floats
    without a dot or without an exponent sign (``3e8``, ``2.5e7``) are floats, timestamps are
    strings, and a repeated key is an error."""

    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            if key_node.value in keys:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key_node.value!r}", key_node.start_mark)
            keys.add(key_node.value)
        return super().construct_mapping(node, deep=deep)


OmegaConfLoader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT,
                                      list("-+0123456789."))
OmegaConfLoader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:timestamp"]
    for k, v in OmegaConfLoader.yaml_implicit_resolvers.items()}


def load_yaml(text: str) -> Any:
    return yaml.load(text, Loader=OmegaConfLoader)  # a SafeLoader subclass


def _merge(base: dict, extra: dict) -> dict:
    """OmegaConf.merge for plain dicts: nested dicts merge, anything else replaces."""
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_override(text: str) -> tuple[list[str], Any]:
    """``a.b=value`` (Hydra's form for an existing key) -> (["a", "b"], value)."""
    key, sep, value = text.partition("=")
    if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", key):
        raise ValueError(f"override {text!r} is not key.path=value")
    return key.split("."), load_yaml(value) if value != "" else ""


def compose_config(overrides: list[str] = (), config_dir: Path = CONFIG_DIR,
                   name: str = "minigrid-sfl") -> dict:
    """The config Hydra hands ``minigrid_sfl.main`` (after ``OmegaConf.to_container``): the
    primary config's ``defaults`` groups, then the primary config itself (``_self_`` is
    last), then ``overrides`` (``key.path=value``, existing keys only, as Hydra requires
    without ``+``)."""
    primary = load_yaml((config_dir / f"{name}.yaml").read_text())
    defaults = primary.pop("defaults", [])
    cfg: dict = {}
    seen_self = False
    for entry in defaults:
        if entry == "_self_":
            cfg = _merge(cfg, primary)
            seen_self = True
            continue
        (group, option), = entry.items()
        cfg = _merge(cfg, {group: load_yaml((config_dir / group / f"{option}.yaml").read_text())})
    if not seen_self:      # Hydra 1.1+: _self_ defaults to last when it is not listed
        cfg = _merge(cfg, primary)
    for text in overrides:
        path, value = parse_override(text)
        node = cfg
        for part in path[:-1]:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"override {text!r}: {part!r} is not a config group")
            node = node[part]
        if path[-1] not in node:
            raise KeyError(f"override {text!r}: {'.'.join(path)} is not in the config")
        node[path[-1]] = value
    return cfg


def e5_overrides(seed: int, view: int, updates: int, base: dict | None = None) -> list[str]:
    """The overrides of an E5 run (maze-e5.md Section 2). DEV-E5-3: ``updates`` PPO updates
    (SFL's NUM_UPDATES is TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS). DEV-E5-4: the view.
    DEV-E5-2: NUM_CHECKPOINTS = updates // EVAL_FREQ, so the script calls save_params after
    every refresh block (its checkpoint_steps becomes 1) and E5's writer keeps the selected
    updates; this changes nothing the script computes."""
    cfg = base or compose_config()
    lc = cfg["learning"]
    per_update = int(lc["NUM_STEPS"]) * int(lc["NUM_ENVS"])
    if updates % int(lc["EVAL_FREQ"]):
        raise ValueError(f"updates {updates} is not a multiple of EVAL_FREQ {lc['EVAL_FREQ']}")
    return [f"SEED={int(seed)}", f"env.AGENT_VIEW_SIZE={int(view)}",
            f"learning.TOTAL_TIMESTEPS={updates * per_update}",
            f"learning.NUM_CHECKPOINTS={updates // int(lc['EVAL_FREQ'])}"]


def num_updates(cfg: dict) -> int:
    """minigrid_sfl.py L93-95."""
    lc = cfg["learning"]
    return int(lc["TOTAL_TIMESTEPS"] // lc["NUM_STEPS"] // lc["NUM_ENVS"])


def run_settings(cfg: dict) -> dict:
    """What the audit shares with an SFL run's training: generator walls, discount, lambda,
    PPO rollout length and view (the time limit is the Maze default, 250 steps)."""
    return dict(n_walls=int(cfg["env"]["N_WALLS"]), gamma=float(cfg["learning"]["GAMMA"]),
                gae_lambda=float(cfg["learning"]["GAE_LAMBDA"]),
                num_steps=int(cfg["learning"]["NUM_STEPS"]),
                agent_view_size=int(cfg["env"]["AGENT_VIEW_SIZE"]),
                rollout_steps=int(cfg["ROLLOUT_STEPS"]),
                eval_freq=int(cfg["learning"]["EVAL_FREQ"]))


# --------------------------------------------------------------------------- outcomes


def install_mask_where():
    """SFL's code passes 0/1 integer masks as ``where=`` to mean and std (minigrid_sfl.py
    L238-243, L392, L457-475). JAX now accepts only boolean ``where``; before, a mask entered
    the reduction as its truth value, so a non-boolean mask is cast to ``mask != 0``, the
    computation the code was written for. Returns the check it replaced (None if the cast is
    already installed); run_sfl_maze.py installs it for its whole process."""
    import jax._src.numpy.reductions as red
    import jax.numpy as jnp

    orig = red.check_where
    if getattr(orig, "_e5_mask_where", False):
        return None

    def check_where(name, where):
        if where is not None and getattr(where, "dtype", None) != bool:
            where = jnp.asarray(where) != 0
        return orig(name, where)

    check_where._e5_mask_where = True
    red.check_where = check_where
    return orig


@contextmanager
def mask_where():
    """``install_mask_where`` for the duration only."""
    import jax._src.numpy.reductions as red

    orig = install_mask_where()
    try:
        yield
    finally:
        if orig is not None:
            red.check_where = orig



def sfl_outcomes_reference(max_steps: int, dones, returns) -> dict:
    """minigrid_sfl.py L218-244 (``_calc_outcomes_by_agent``), line for line, for arrays
    [max_steps, N]: at most 10 completed episodes per level, the open last one dropped,
    success = a positive reward in the episode."""
    from functools import partial

    import jax
    import jax.numpy as jnp

    @partial(jax.vmap, in_axes=(None, 1, 1))
    @partial(jax.jit, static_argnums=(0,))
    def _calc_outcomes_by_agent(max_steps: int, dones, returns):
        idxs = jnp.arange(max_steps)

        @partial(jax.vmap, in_axes=(0, 0))
        def __ep_outcomes(start_idx, end_idx):
            mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
            r = jnp.sum(returns * mask)
            goal_r = (returns > 0) * 1.0
            success = jnp.sum(goal_r * mask)
            l = end_idx - start_idx  # the reference's name
            return r, success, l

        done_idxs = jnp.argwhere(dones, size=10, fill_value=max_steps).squeeze()
        mask_done = jnp.where(done_idxs == max_steps, 0, 1)
        ep_return, success, length = __ep_outcomes(
            jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)
        return {"ep_return": ep_return.mean(where=mask_done),
                "num_episodes": mask_done.sum(),
                "success_rate": success.mean(where=mask_done),
                "ep_len": length.mean(where=mask_done)}

    with jax.enable_x64(False), mask_where():
        o = _calc_outcomes_by_agent(int(max_steps), jnp.asarray(dones), jnp.asarray(returns))
        o = {k: np.asarray(v) for k, v in o.items()}
    o["learnability"] = o["success_rate"] * (1 - o["success_rate"])   # L279-280, one agent
    return o


def native_learnability(done: np.ndarray, reward: np.ndarray, rollout_steps: int = 2000,
                        max_episodes: int = 10) -> dict[str, np.ndarray]:
    """SFL's native learnability on the first ``rollout_steps`` steps of streams [T, N] from a
    reset with a fresh carry: the first ``max_episodes`` completed episodes, the open last
    one dropped, success = a positive reward in the episode, l = p(1 - p). NaN where no
    episode completes (SFL's argsort would rank it top; with the 250-step time limit and 2,000
    steps it cannot happen)."""
    done = np.asarray(done, bool)[:rollout_steps]
    reward = np.asarray(reward, float)[:rollout_steps]
    if done.shape[0] < rollout_steps:
        raise ValueError(f"stream has {done.shape[0]} steps; SFL counts {rollout_steps}")
    k = np.cumsum(done, axis=0)
    # Step t belongs to episode ep[t], the number of episode ends before it; the first
    # max_episodes completed episodes count. Success sums the positive-reward steps of the
    # counted episodes, as the reference does (in the maze, at most one per episode).
    ep = np.r_[np.zeros((1, done.shape[1]), int), k[:-1]]
    n_eps = np.minimum(k[-1], max_episodes)
    succ = ((reward > 0) & (ep < n_eps[None, :])).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n_eps > 0, succ / np.where(n_eps > 0, n_eps, 1), np.nan)
    return {"success_rate": p, "learnability": p * (1 - p), "episodes": n_eps}


def reference_top_k(scores, k: int) -> np.ndarray:
    """SFL's top-K (minigrid_sfl.py L289): ``argsort(l)[-K:]``, ascending; ties go to the
    highest index (S-1, S-2). NaN raises (S-3): SFL's argsort would rank it first."""
    s = np.asarray(scores, float)
    if np.isnan(s).any():
        raise ValueError("learnability has NaN entries (S-3)")
    return np.argsort(s, kind="stable")[-k:]


# --------------------------------------------------------------------------- checkpoints


class SFLCheckpoint(NamedTuple):
    params: Any          # flax variables {"params": ...} for minigrid_plr.ActorCritic
    config: dict         # the run's config.json (the composed config)
    update: int          # PPO updates at save time
    learnable: dict      # D in force: level fields, "scores", "refresh_update"


def checkpoint_dir(run_dir: str | Path, update: int) -> Path:
    return Path(run_dir) / CKPT_DIR / f"u{int(update):05d}"


def save_checkpoint(run_dir: str | Path, update: int, params, learnable: dict | None) -> Path:
    """Write ``params`` (flax msgpack) and D (npz) for ``update``; refuses to overwrite."""
    from flax import serialization

    d = checkpoint_dir(run_dir, update)
    d.mkdir(parents=True, exist_ok=False)
    (d / "params.msgpack").write_bytes(serialization.to_bytes(params))
    if learnable is not None:
        buf = io.BytesIO()
        np.savez(buf, **{k: np.asarray(v) for k, v in learnable.items()})
        (d / "learnable.npz").write_bytes(buf.getvalue())
    return d


def checkpoint_updates(run_dir: str | Path) -> list[int]:
    base = Path(run_dir) / CKPT_DIR
    if not base.is_dir():
        return []
    return sorted(int(p.name[1:]) for p in base.iterdir()
                  if re.fullmatch(r"u\d{5}", p.name) and (p / "params.msgpack").is_file())


def load_sfl_checkpoint(run_dir: str | Path, update: int) -> SFLCheckpoint:
    from flax import serialization

    run_dir = Path(run_dir)
    d = checkpoint_dir(run_dir, update)
    params = serialization.msgpack_restore((d / "params.msgpack").read_bytes())
    learnable = {}
    if (d / "learnable.npz").is_file():
        with np.load(d / "learnable.npz") as z:
            learnable = {k: z[k] for k in z.files}
    config = json.loads((run_dir / "config.json").read_text())
    return SFLCheckpoint(params, config, int(update), learnable)


def learnable_levels(learnable: dict):
    """D's levels as a batched jaxued ``Level``."""
    import jax.numpy as jnp
    from jaxued.environments.maze import Level

    return Level(**{k: jnp.asarray(learnable[k]) for k in LEVEL_FIELDS})


# --------------------------------------------------------------------------- the network


def stub_modules(save_params=None) -> dict:
    """The stand-ins for what SFL's files import and this repo does not install or vendor:
    ``hydra.main`` is the identity (the caller passes the composed config itself),
    ``OmegaConf.to_container`` a deep copy, and ``sfl.train.train_utils.save_params`` the
    given writer (a no-op by default)."""
    from types import SimpleNamespace

    return {
        "hydra": SimpleNamespace(main=lambda **kw: (lambda fn: fn)),
        "omegaconf": SimpleNamespace(OmegaConf=SimpleNamespace(to_container=copy.deepcopy)),
        "sfl.train.train_utils": SimpleNamespace(
            save_params=save_params or (lambda params, path: None)),
    }


@contextmanager
def import_context(stubs: dict):
    """Import SFL's vendored files: third_party/sfl on sys.path and ``stubs`` in sys.modules
    for the duration only (the imported modules keep their references)."""
    import sys

    saved = {k: sys.modules.get(k) for k in [*stubs, "sfl", "sfl.train"]}
    sys.modules.update(stubs)
    sys.path.insert(0, str(VENDOR))
    try:
        yield
    finally:
        sys.path.remove(str(VENDOR))
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def load_vendored(path: Path, name: str, stubs: dict):
    import importlib.util

    import apexzero.compat  # noqa: F401  (jax.tree_map, which SFL's code calls)

    with import_context(stubs):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def minigrid_plr():
    """SFL's sfl/train/minigrid_plr.py as a module (hash-checked; raises MissingSFLSource
    while it is not vendored). Its own imports are not known yet: the file was not read."""
    check_minigrid_plr()
    return load_vendored(VENDOR / MINIGRID_PLR, "sfl_minigrid_plr", stub_modules())


def _sfl_network_act(params, key, hstate, obs, last_done):
    # minigrid_sfl.py L179-186 (get_learnability_set's _env_step), one step.
    import jax

    net = minigrid_plr().ActorCritic(7)
    x = jax.tree.map(lambda a: a[None, ...], (obs, last_done))
    hstate, pi, value = net.apply(params, x, hstate)
    action = pi.sample(seed=key)
    return hstate, action.squeeze(0), value.squeeze(0)


def sfl_network_policy(params):
    """An SFL checkpoint's stochastic policy, with a fresh carry as in every SFL rollout."""
    from apexzero.maze.audit import Policy

    return Policy(params, lambda n: minigrid_plr().ActorCritic.initialize_carry((n,)),
                  _sfl_network_act)
