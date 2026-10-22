"""Idea 2, the costate: does the network encode the TO-OCP's costate?

Design: ``docs/interp/costate-design.md`` (draft v2.1; not frozen until the tag
``costate-prereg-v1``). Step 1's code (``src/apexzero/interp``) and step 2's
(``src/apexzero/fields``) are frozen and only imported here, read-only (design Section 1,
'The rules of steps 1 and 2 carry over').

Module map (built in stages; each module's docstring cites what it relies on):

- ``targets``: the optimum side of Section 3. The post-jump discrete costate lambda_t at
  every plan state of a key, by the ``jac_g`` route and by the cheap route (path Jacobians
  and the box multipliers, no ``jac_g``), with the pedal fold that makes lambda_pedal
  unique; the previous-action component pi_t; the switching function sigma_t; the pre-jump
  form; the wall, slip, grip and context shadow prices and road fields of Section 3.4; the
  K0 identities (Section 7) as functions of one key; and the pinned-prefix defect shifts of
  K1, used by the finite-difference tests.
- ``states``: the state sets of Section 4 (T, D, N), their strata (interior with k = 10
  steps, Section 12 decision 4; near-limit, contact, pedal-saturated, phase), the six
  geometry folds and the inner folds, the decoys D1 with the cross-fitted kNN
  residualisation, observability by column residual (the tau_s rule of Section 3.2) and
  the key weights (Section 4 'Weights').

**Seed streams** (``STREAMS``): idea 2's named NumPy streams, disjoint from step 1's
(constants.py:57-67, indices 1-9), step 2's (measures.py:129-130, 101-105) and H4R's S3
(scripts/h4r.py:160, 201). ``stream_rng`` seeds them as step 1 and step 2 do (io.py:236-238,
measures.py:154-157).

**Data directories** (``data_dirs``): step 1's ``io.KEYS_DIR`` and ``io.STIMULI_DIR``
(io.py:28-33), or, in a git worktree whose ``outputs/`` holds the main checkout's outputs as
the link ``outputs/outputs``, the same directories through that link. The keys are read,
never written.

Pre-registration rule (design Section 7, Section 1): before the tag no trained network's
activation, hidden unit, critic value or gradient is computed. Nothing in ``targets`` or
``states`` evaluates a network; input-side quantities (J_z) enter ``states`` as arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apexzero.interp import io
from apexzero.interp.constants import ROOT_SEED, STREAM_OFFSET

TAG = "costate-prereg-v1"
STREAMS = {"costate_folds": 301, "costate_inner": 302, "costate_horizon": 303,
           "costate_perm": 304, "costate_control": 305, "costate_c3": 306,
           "costate_mlp": 307, "costate_boot": 308, "costate_k3": 309, "costate_k4": 310}
# 307, 308 and 310 are the indices probes.MLP_STREAM, probes.BOOT_STREAM and
# gates.K4_STREAM already seed with this rule (named here, their draws unchanged); 309 is
# K3's key sample (critic._dry_keys).


def stream_rng(name: str, *ids: int) -> np.random.Generator:
    """A NumPy generator of one of idea 2's seed streams; ``ids`` index sub-streams (the
    robustness partitions of the folds, design Section 4 'Splits')."""
    return np.random.default_rng([ROOT_SEED, STREAM_OFFSET + STREAMS[name], *map(int, ids)])


def data_dirs() -> tuple[Path, Path]:
    """(keys root, stimuli directory) of step 1's outputs (see the module docstring)."""
    if (io.KEYS_DIR / "nominal").exists():
        return io.KEYS_DIR, io.STIMULI_DIR
    link = io.REPO / "outputs" / "outputs" / "psychophysics"
    if (link / "keys" / "nominal").exists():
        return link / "keys", link / "stimuli"
    return io.KEYS_DIR, io.STIMULI_DIR


def keys_root(root: Path | None = None) -> Path:
    return Path(root) if root is not None else data_dirs()[0]


def stimuli_dir(path: Path | None = None) -> Path:
    return Path(path) if path is not None else data_dirs()[1]
