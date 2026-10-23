"""The state sets of idea 2, their strata, folds, decoys, observability and weights (design
v2.1, Sections 3.2, 4 and 12).

**State sets** (Section 4 'State sets'). Rows are column tables (``io.Table``, io.py:60-110:
equal-length 1-d columns), one row per (key, plan step t):

- **T** (``key_rows``, ``t_steps``): an A2 key's (``keys.a2_refs``, keys.py:170-174) plan states
  every ``EVERY`` = 5 steps from t = 0 to t_end, the first t with s_t >= s_c1 + 50 m
  (``t_end``), i.e. the multiples of 5 up to t_end (9,967 states on the 481 keys). K2's
  horizon cut (``horizon_keep``) keeps t <= t_H on W = 400 keys and t <= 400 + t_H on W = 800
  keys (remaining horizon, Section 7 K2); ``rule='absolute'`` keeps t <= t_H on every key (the
  reading behind the design's '9,170 at t_H = 100').
- **D**: T at t = 0 (481 states; ``rows['t'] == 0``).
- **N** (``n_refs``, ``key_rows(..., steps=[1])``): step 1's 12 finals' anchored keys
  (``keys.anchored_finals``, keys.py:202-212), at x_1 (8,136 states).

**Strata** per row, from the key's targets (``targets.Costate``) and its stimulus landmarks
(stimuli.py:205-219):

- *interior* (``interior``; Section 4 and Section 12 decision 4): for every tau in
  [t, min(t + k, W)], x_tau's own corridor and slip rows are inactive, its clearance to the
  env's termination boundary is at least ``CLEARANCE_MIN`` on both sides, and its slip
  margin to alpha_max at least ``SLIP_MARGIN_MIN``. k = ``INTERIOR_K`` = 10 steps is
  primary; ``INTERIOR_K_DESCRIPTIVE`` = 20 is descriptive. The window looks ahead, as
  Section 4 states it ('from t to t + k'); decision 4's 'k steps from the nearest contact'
  only changes k. The two thresholds are provisional (fixed in code at the tag after the dry
  run, Section 12 decision 7);
- *near-limit*: the rest of T (``~interior``);
- *contact*: x_t's own corridor, slip or (folded) box row is active (``Costate.contact``);
- *pedal-saturated*: the raw pedal-state bound is active at x_t (|lam_x[8t+7]| above
  ``targets.SATURATED_TOL``); a reporting stratum only: that raw multiplier is split
  arbitrarily between the defect, the state bound and the command bound (LICQ fails), and
  its share of T moves from 0.28 to 0.19 as the tolerance goes from 1e-9 to 1e-6;
  *command-saturated*: the pedal command's raw bound multiplier is active at t;
- *pedal-switching* (``SW_STRATUM``, C2-sw's decision stratum and PR-8's rows): C2-sw's
  target sigma^ped_t = gamma^-t x the folded pedal-command multiplier is above
  ``targets.ACTIVE_TOL``: a unique quantity with a gap (the same rows for tolerances 1e-8 to
  1e-5), the target's support (the raw flag drops 35% of it), t = 0 included. Design Section
  3.1 keeps the raw flag "for reporting" and C2's theory puts sigma on command-saturated
  steps; Sections 3.2 and 6 C2-sw's 'pedal-saturated steps' wording is read as this stratum
  (flagged to the design's author);
- *phase* (``phases``): 'approach' before the optimum's brake onset s_b (step 1's
  ``brake_onset`` on the plan's front brake demand, as ``outcomes.s_b_plan`` reads it,
  outcomes.py:242-255, measures.py:1444-1477), 'braking' from s_b to the geometric apex,
  'corner' from the apex to s_c1, 'exit' from s_c1. A key without a sustained brake before
  the apex starts its 'braking' phase at s_c0 (its corner entry). *corner_band* is Section
  12's separate s_c0 <= s < s_c1 stratum;
- *braking key* (Section 4 'Sub-scores'): lambda_vx changes sign exactly twice over the key's
  T range (``sign_changes``).

**Folds** (``geometry_folds``; Section 4 'Splits'). The unit is the geometry (R, theta)
(``keys.Strata.geometry``, keys.py:247-250), so mirror partners, phases, tier C widths, the
horizon twins and every plan state of a geometry share a fold. Strata are R tercile (by the
geometries' rank in (R, theta)) x theta class {90, 180, other}; within each stratum the
geometries, taken in (R, theta) order, are permuted on stream ``costate_folds`` (sub-stream =
the partition), the strata are visited sorted by (theta class, R tercile), and the
concatenated list is dealt to the 6 folds in serpentine order (0..5, 5..0, ...), 8
geometries per fold. (This reads the design's 'permuted ..., sorted by (theta, R) and dealt'
as a sort of the strata: (R, theta) is unique per geometry, so sorting the geometries
themselves would undo the permutation.) Partition 0 is primary; partitions 1..10 are the
robustness partitions. ``inner_folds`` deals the training geometries to 5 inner folds the
same way on stream ``costate_inner`` (sub-streams: partition, held-out fold).

**Decoys** (``decoy_d1``, ``cross_fit``; Section 4 'Decoys'). D1 = (s - s_apex, vx, log R,
theta, w, sigma). f_D is the ``KNN_K`` = 8 nearest-neighbour mean (weighted by the
metric's weights) on D1 standardised by the (weighted) mean and SD of its own training
states, the same k in every fit, outer fold, partition and K_P component. For outer fold f
the held-out states get the residual of the model fitted on the other five folds; the
training states of fold f get cross-fitted residuals (fold g's states from the model fitted
on the four folds other than f and g), so no state is ever its own neighbour.

f_D's k (v2.3, the second decision before the tag; ``gates.decoy_k_choice``). In C2 the
residualiser serves validity: what leaks into C2 is f_D's bias (E[f_D] differing from the
decoy function), which a network holding that function can recover; its variance (noise
borrowed from other geometries) no network can recover, and it only adds unexplainable
variance to lambda_res, a power cost in the conservative direction. An out-of-geometry SSE
trades the two, so it is the wrong criterion for choosing k (the nested-CV choice per fold
and component it replaces took k = 15-30 on K4's smooth known-truth world, which then
leaked). The rule, fixed before any result: among ``gates.DECOY_K_CANDIDATES`` = (3, 5, 8,
15) the largest k for which every one of 6 validation seeds (stream ``costate_decoy_k``,
disjoint from K4's gate seeds) of K4's PR-6b and PR-6c worlds gives C2's primary Delta2 <=
0.01 with y residualised at k, C2's power floor recomputed at k; none: k = 3, a blocker. On
the full-size T (9,887 states, G1's obs_rms, random-init actors) only k = 8 qualifies
(largest leak 0.0053 on PR-6b, 0.0096 on PR-6c; k = 3, 5 and 15 each have a seed above
0.01; ``gates.DECOY_K_VALIDATION``). At k = 8 the interior residual shares of (s, n, xi, vx)
are (0.224, 0.091, 0.248, 0.076) (0.289, 0.107, 0.213, 0.070 at the design's k = 15):
lambda_n is below the power floor, so C2 decides on s and xi.

``residual_share`` is Var_w(lambda_res) / Var_w(lambda) (about within-direction means for the
mirror-odd components, as R2_wd); ``knn_stability`` the spread of a state's six
predictions; ``POWER_FLOOR`` = 0.10 (Section 12 decision 5).

**Observability** (``observability``; Section 3.2 'Exclusions and flags'). From J_z [.., m, 8]
(dz/dx at fixed accel and prev_action, rows zeroed where the obs clip binds): s is dropped as
structurally unobservable when ||J_z e_s|| / median_k ||J_z e_k|| < ``TAU_S`` = 1e-4; every
other component's squared residual after least-squares projection of its column onto the
other columns (s's excluded when dropped; an SVD least squares with relative cutoff
``LSTSQ_RCOND``, no explicit inverse) is compared with a per-component threshold set in the
dry run (``thresholds``; none yet, so only the s rule decides).

**Weights** (``weights``; Section 4 'Weights', Section 12 decision 1): 'key' (primary): each
key equal, states equal within a key; 'geometry': geometries equal, mirror pairs equal within
a geometry, states equal within a pair; 'state': states equal. ``small_geometry`` marks the 11
geometries with 3 keys or fewer (the third sensitivity drops them). Weights sum to 1 over the
rows given, so a subset (the interior stratum) is reweighted on its own rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from apexzero.costate import keys_root, stimuli_dir, stream_rng
from apexzero.costate.targets import (
    ACTIVE_TOL,
    K_P,
    MIRROR_ODD,
    STATE_NAMES,
    VX,
    Costate,
    Prices,
)
from apexzero.fields import keys as K
from apexzero.fields import outcomes as O
from apexzero.interp import io
from apexzero.interp.stimuli import Landmarks

EVERY = 5                    # T: every 5th plan step (Section 4)
EXIT_MARGIN = 50.0           # m: t_end is the first t with s_t >= s_c1 + this
INTERIOR_K = 10              # steps (0.5 s): primary interior horizon (Section 12 decision 4)
INTERIOR_K_DESCRIPTIVE = 20  # steps (1 s): descriptive
CLEARANCE_MIN = 0.25         # m to the termination boundary (provisional: dry run, Section 12)
SLIP_MARGIN_MIN = 0.01       # rad below alpha_max (provisional: dry run, Section 12)
PHASES = ("approach", "braking", "corner", "exit")
N_FOLDS = 6
N_INNER = 5
N_PARTITIONS = 11            # partition 0 (frozen at the tag) + 10 robustness partitions
THETA_CLASSES = (90.0, 180.0)   # class 0, 1; every other theta is class 2
KNN_K = 8                    # f_D's one k (module docstring; gates.decoy_k_choice)
POWER_FLOOR = 0.10           # Section 12 decision 5
TAU_S = 1e-4                 # Section 3.2: s's column-norm ratio
LSTSQ_RCOND = 1e-12          # relative singular-value cutoff of the column projections
SMALL_GEOMETRY_KEYS = 3      # Section 4: sensitivity (iii) drops geometries with <= this
WINSOR = (0.005, 0.995)      # Section 4 'Metrics': winsorisation quantiles on T
SCHEMES = ("key", "geometry", "state")
SW_STRATUM = "pedal_switching"   # C2-sw's decision stratum (module docstring)
D1_NAMES = ("ds_apex", "vx", "log_R", "theta", "width", "direction")

# ------------------------------------------------------------------------- per key


def t_end(x: np.ndarray, lm: Landmarks, margin: float = EXIT_MARGIN) -> int:
    """The first t with s_t >= s_c1 + margin (Section 4)."""
    s = np.asarray(x, float)[:, 0]
    hit = np.flatnonzero(s >= lm.s_c1 + margin)
    if not len(hit):
        raise ValueError(f"the plan never reaches s_c1 + {margin} m (s_max {s.max():.1f})")
    return int(hit[0])


def horizon_keep(t: np.ndarray, W: int, t_H: int | None, rule: str = "remaining") -> np.ndarray:
    """K2's retention (Section 7): t <= t_H on W = 400 keys and t <= (W - 400) + t_H on longer
    ones ('remaining'), or t <= t_H on every key ('absolute'); all kept when t_H is None."""
    t = np.asarray(t)
    if t_H is None:
        return np.ones(t.shape, bool)
    if rule == "remaining":
        return t <= (int(W) - 400) + int(t_H)
    if rule == "absolute":
        return t <= int(t_H)
    raise ValueError(f"unknown horizon rule {rule!r}")


def t_steps(x: np.ndarray, lm: Landmarks, W: int, t_H: int | None = None,
            rule: str = "remaining", every: int = EVERY) -> np.ndarray:
    """T's plan steps of one key: multiples of ``every`` up to t_end, after K2's cut."""
    t = np.arange(0, t_end(x, lm) + 1, every)
    return t[horizon_keep(t, W, t_H, rule)]


def _window_all(ok: np.ndarray, k: int) -> np.ndarray:
    """[n] ok[t..min(t + k, n - 1)] all True."""
    bad = np.r_[0, np.cumsum(~np.asarray(ok, bool))]
    n = len(ok)
    hi = np.minimum(np.arange(n) + k + 1, n)
    return (bad[hi] - bad[:n]) == 0


def interior(cs: Costate, k: int = INTERIOR_K, clearance_min: float = CLEARANCE_MIN,
             slip_margin_min: float = SLIP_MARGIN_MIN) -> np.ndarray:
    """[W+1] the interior stratum (module docstring)."""
    ok = ~(cs.own_corridor.any(1) | cs.own_slip.any(1))
    ok &= np.min(cs.clearance, axis=1) >= clearance_min
    ok &= cs.slip_margin >= slip_margin_min
    return _window_all(ok, int(k))


def brake_onset(cs: Costate, lm: Landmarks) -> float:
    """s_b of the optimum's plan (nan without a sustained brake before s_apex)."""
    v = O.view(O.trajectory(cs.x))
    beta = np.asarray(O.M.front_brake_demand(v.x, v.p, O.M.NP)[0], float)
    return float(O.M.brake_onset(v.s, v.t, beta, np.ones(v.n, bool), lm.s_apex, v.vx).s_b)


def phases(s: np.ndarray, lm: Landmarks, s_b: float) -> np.ndarray:
    """Phase labels of arc lengths s (module docstring)."""
    s = np.asarray(s, float)
    start = s_b if np.isfinite(s_b) else lm.s_c0
    idx = np.searchsorted(np.array([start, lm.s_apex, lm.s_c1]), s, side="right")
    return np.asarray(PHASES)[idx]


def sign_changes(v: np.ndarray, atol: float = 0.0) -> int:
    """Sign changes of a series, ignoring entries with |v| <= atol."""
    v = np.asarray(v, float)
    sg = np.sign(v[np.abs(v) > atol])
    return int(np.sum(sg[1:] != sg[:-1]))


TARGET_COLUMNS = (*(f"lam_{c}" for c in STATE_NAMES), *(f"lam_pre_{c}" for c in STATE_NAMES),
                  "pi_steer", "pi_pedal", "sigma_steer", "sigma_pedal", "Pi_in", "Pi_out",
                  "Pi_slip", "Pi_mu", "d_in")


def key_rows(cs: Costate, pr: Prices, ref: K.KeyRef, st: K.Strata, lm: Landmarks,
             steps: np.ndarray | None = None, t_H: int | None = None,
             rule: str = "remaining", clearance_min: float = CLEARANCE_MIN,
             slip_margin_min: float = SLIP_MARGIN_MIN) -> io.Table:
    """One key's rows: T's steps by default (``t_steps``), or the given ``steps`` (N: [1])."""
    if steps is None:
        te = t_end(cs.x, lm)
        t = t_steps(cs.x, lm, cs.W, t_H, rule)
    else:
        try:
            te = t_end(cs.x, lm)
        except ValueError:
            te = cs.W - 1                    # a plan that never leaves the corner window
        t = np.asarray(steps, int)
    if np.any((t < 0) | (t >= cs.W)):
        raise ValueError("plan steps must lie in [0, W)")
    n = len(t)
    s = cs.x[t, 0]
    s_b = brake_onset(cs, lm)
    inner = interior(cs, INTERIOR_K, clearance_min, slip_margin_min)
    inner20 = interior(cs, INTERIOR_K_DESCRIPTIVE, clearance_min, slip_margin_min)
    sig = int(st.direction)
    side_in = 0 if sig > 0 else 1
    rng_T = np.arange(0, te + 1)
    out: io.Table = {
        "key_id": np.full(n, ref.key_id), "kind": np.full(n, ref.kind),
        "stim_id": np.full(n, ref.stim_id), "cell_id": np.full(n, ref.cell_id),
        "pair_id": np.full(n, st.pair_id), "key_phase": np.full(n, ref.phase),
        "W": np.full(n, cs.W), "direction": np.full(n, sig), "R": np.full(n, st.R),
        "theta": np.full(n, st.theta_deg), "width": np.full(n, st.width),
        "w_is_cell": np.full(n, st.w_is_cell), "multi_basin": np.full(n, cs.multi_basin),
        "agent": np.full(n, ref.agent), "t": t, "t_end": np.full(n, te),
        "s": s, "n": cs.x[t, 1], "vx": cs.x[t, VX], "s_apex": np.full(n, lm.s_apex),
        "s_c0": np.full(n, lm.s_c0), "s_c1": np.full(n, lm.s_c1), "s_b": np.full(n, s_b),
        "ds_apex": s - lm.s_apex, "ds_c1": s - lm.s_c1, "phase": phases(s, lm, s_b),
        "corner_band": (s >= lm.s_c0) & (s < lm.s_c1),
        "interior": inner[t], "interior20": inner20[t], "near_limit": ~inner[t],
        "contact": cs.contact[t], "pedal_saturated": cs.pedal_saturated[t],
        "command_saturated": cs.command_saturated[t, 1],
        SW_STRATUM: cs.switch[t, 1] > ACTIVE_TOL,
        "braking_key": np.full(n, sign_changes(cs.lam[rng_T, VX]) == 2),
        "clearance_in": cs.clearance[t, side_in], "clearance_out": cs.clearance[t, 1 - side_in],
        "slip_margin": cs.slip_margin[t],
    }
    for i, c in enumerate(STATE_NAMES):
        out[f"lam_{c}"] = cs.lam[t, i]
    for i, c in enumerate(STATE_NAMES):
        out[f"lam_pre_{c}"] = cs.lam_pre[t, i]
    out["pi_steer"], out["pi_pedal"] = cs.pi[t, 0], cs.pi[t, 1]
    out["sigma_steer"], out["sigma_pedal"] = cs.switch[t, 0], cs.switch[t, 1]
    out["Pi_in"], out["Pi_out"] = pr.inside[t], pr.outside[t]
    out["Pi_slip"], out["Pi_mu"], out["d_in"] = pr.slip[t], pr.mu[t], pr.d_in[t]
    io.check_table(out)
    return out


def concat(tables: list[io.Table]) -> io.Table:
    """Rows of several keys (``io.concat_tables``), with each row's geometry index."""
    tab = io.concat_tables(tables)
    tab["geometry"] = geometry_index(tab)
    return tab


def a2_refs(root=None) -> tuple[K.KeyRef, ...]:
    return K.a2_refs(keys_root(root), stimuli_dir())


def n_refs(root=None) -> dict[str, tuple[K.KeyRef, ...]]:
    """N's keys: the verified anchored keys at each of step 1's agents' final snapshots
    (reads each final snapshot's timesteps; no network is evaluated)."""
    return K.anchored_finals(keys_root(root))


def strata(ref: K.KeyRef) -> K.Strata:
    return K.strata(ref, stimuli_dir())


# ------------------------------------------------------------------------- geometry and folds


def geometry_keys(tab: io.Table) -> np.ndarray:
    """[n, 2] (R, theta) per row."""
    return np.stack([np.asarray(tab["R"], float), np.asarray(tab["theta"], float)], axis=1)


def geometry_index(tab: io.Table) -> np.ndarray:
    """Each row's geometry as an index into the sorted unique (R, theta)."""
    _, inv = np.unique(geometry_keys(tab), axis=0, return_inverse=True)
    return inv.ravel().astype(int)


def theta_class(theta: np.ndarray) -> np.ndarray:
    th = np.asarray(theta, float)
    out = np.full(th.shape, len(THETA_CLASSES), int)
    for k, v in enumerate(THETA_CLASSES):
        out[np.isclose(th, v)] = k
    return out


def r_tercile(R: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Tercile 0..2 of each geometry by its rank in (R, theta)."""
    order = np.lexsort((np.asarray(theta, float), np.asarray(R, float)))
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    return (3 * rank) // len(order)


def _serpentine(n: int, k: int) -> np.ndarray:
    base = np.arange(k)
    seq = np.concatenate([base if j % 2 == 0 else base[::-1] for j in range(-(-n // k))])
    return seq[:n]


def deal(R: np.ndarray, theta: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Fold of each geometry (module docstring's algorithm)."""
    R, theta = np.asarray(R, float), np.asarray(theta, float)
    cls, ter = theta_class(theta), r_tercile(R, theta)
    order = []
    for c in sorted(set(cls.tolist())):
        for q in sorted(set(ter[cls == c].tolist())):
            members = np.flatnonzero((cls == c) & (ter == q))
            members = members[np.lexsort((theta[members], R[members]))]
            order.extend(members[rng.permutation(len(members))].tolist())
    fold = np.empty(len(R), int)
    fold[np.asarray(order, int)] = _serpentine(len(order), n_folds)
    return fold


def geometry_folds(R: np.ndarray, theta: np.ndarray, partition: int = 0,
                   n_folds: int = N_FOLDS) -> np.ndarray:
    """Outer fold of each geometry, for one partition (0: primary; 1..10: robustness)."""
    if not 0 <= partition < N_PARTITIONS:
        raise ValueError(f"partition {partition} outside 0..{N_PARTITIONS - 1}")
    return deal(R, theta, n_folds, stream_rng("costate_folds", partition))


def inner_folds(R: np.ndarray, theta: np.ndarray, outer: np.ndarray, held_out: int,
                partition: int = 0, n_inner: int = N_INNER) -> np.ndarray:
    """Inner fold (0..n_inner-1) of each training geometry of outer fold ``held_out``; -1 for
    the held-out geometries."""
    outer = np.asarray(outer, int)
    train = outer != held_out
    out = np.full(len(outer), -1, int)
    out[train] = deal(np.asarray(R)[train], np.asarray(theta)[train], n_inner,
                      stream_rng("costate_inner", partition, held_out))
    return out


def row_folds(tab: io.Table, partition: int = 0, n_folds: int = N_FOLDS) -> np.ndarray:
    """Outer fold of each row (its geometry's)."""
    G = geometry_keys(tab)
    uniq, inv = np.unique(G, axis=0, return_inverse=True)
    return geometry_folds(uniq[:, 0], uniq[:, 1], partition, n_folds)[inv.ravel()]


# ------------------------------------------------------------------------- weights


def _group_sizes(labels: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    return cnt[inv.ravel()]


def weights(tab: io.Table, scheme: str = "key") -> np.ndarray:
    """Row weights summing to 1 over the given rows (module docstring)."""
    n = io.check_table(tab)
    if scheme == "state":
        return np.full(n, 1.0 / n)
    key = np.asarray(tab["key_id"])
    if scheme == "key":
        n_keys = len(np.unique(key))
        return 1.0 / (n_keys * _group_sizes(key))
    if scheme == "geometry":
        g = geometry_index(tab)
        pair = np.char.add(np.char.add(g.astype(str), "/"), np.asarray(tab["pair_id"]).astype(str))
        n_geo = len(np.unique(g))
        _, gi = np.unique(g, return_inverse=True)
        pairs_per_geo = np.array([len(np.unique(pair[g == v])) for v in np.unique(g)])[gi.ravel()]
        return 1.0 / (n_geo * pairs_per_geo * _group_sizes(pair))
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")


def decision_mask(tab: io.Table, observable_s: np.ndarray | None = None,
                  k2b_excluded: dict[str, list[str]] | None = None, names=K_P) -> np.ndarray:
    """[n, m] the rows on which each component may decide (C1, its secondaries, C2): s only
    where observable (the tau_s rule, Section 3.2), no component in a phase stratum where K2b
    excludes it (Section 7 K2b 'Rule'; the bound gates/k2b.json's rule, frozen in the
    record), and never a multi-basin key's row (Section 3.2; such keys are also left out of
    T when it is built)."""
    n = io.check_table(tab)
    names = list(names)
    ok = np.ones((n, len(names)), bool)
    if observable_s is not None and "s" in names:
        ok[:, names.index("s")] &= np.asarray(observable_s, bool)
    if k2b_excluded:
        ph = np.asarray(tab["phase"])
        for p, comps in k2b_excluded.items():
            for c in comps:
                if c in names:
                    ok[ph == p, names.index(c)] = False
    if "multi_basin" in tab:
        ok &= ~np.asarray(tab["multi_basin"], bool)[:, None]
    return ok


def stratum_weights(tab: io.Table, rows: np.ndarray, scheme: str = "key") -> np.ndarray:
    """[n] the scheme's weights renormalised on the scored stratum ``rows`` (each key equal
    on the stratum, states equal within a key), 0 elsewhere: the one convention every
    statistic scored on a stratum uses (Section 4 'Weights')."""
    n = io.check_table(tab)
    rows = np.asarray(rows, bool)
    w = np.zeros(n)
    if rows.any():
        w[rows] = weights({k: np.asarray(v)[rows] for k, v in tab.items()
                           if np.ndim(v) == 1 and len(v) == n}, scheme)
    return w


def small_geometry(tab: io.Table, max_keys: int = SMALL_GEOMETRY_KEYS) -> np.ndarray:
    """[n] the row's geometry has at most ``max_keys`` keys."""
    g = geometry_index(tab)
    key = np.asarray(tab["key_id"])
    n_keys = {v: len(np.unique(key[g == v])) for v in np.unique(g)}
    return np.array([n_keys[v] <= max_keys for v in g])


def wmean(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float)
    return np.tensordot(w, np.asarray(y, float), axes=(0, 0)) / w.sum()


def wvar(y: np.ndarray, w: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Weighted variance (about the weighted mean, or within ``groups``' own means)."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    if groups is None:
        return wmean((y - wmean(y, w)) ** 2, w)
    dev = np.empty_like(y)
    for gval in np.unique(groups):
        m = groups == gval
        dev[m] = y[m] - wmean(y[m], w[m])
    return wmean(dev ** 2, w)


def wquantile(y: np.ndarray, w: np.ndarray, q) -> np.ndarray:
    """Weighted quantiles of a 1-d sample (midpoint interpolation of the weighted CDF)."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    o = np.argsort(y)
    cw = np.cumsum(w[o])
    cdf = (cw - 0.5 * w[o]) / cw[-1]
    return np.interp(q, cdf, y[o])


def winsor_bounds(y: np.ndarray, w: np.ndarray, q=WINSOR) -> tuple[float, float]:
    lo, hi = wquantile(y, w, np.asarray(q, float))
    return float(lo), float(hi)


def winsorise(y: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return np.clip(np.asarray(y, float), bounds[0], bounds[1])


# ------------------------------------------------------------------------- decoys


def decoy_d1(tab: io.Table) -> np.ndarray:
    """[n, 6] D1 = (s - s_apex, vx, log R, theta, w, sigma)."""
    return np.stack([np.asarray(tab["ds_apex"], float), np.asarray(tab["vx"], float),
                     np.log(np.asarray(tab["R"], float)), np.asarray(tab["theta"], float),
                     np.asarray(tab["width"], float), np.asarray(tab["direction"], float)],
                    axis=1)


@dataclass(frozen=True)
class KNN:
    """A fitted k-nearest-neighbour mean on standardised features."""

    mean: np.ndarray
    sd: np.ndarray
    tree: cKDTree
    y: np.ndarray             # [n_train, m]
    w: np.ndarray             # [n_train]
    k: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, float) - self.mean) / self.sd
        _, idx = self.tree.query(Z, k=self.k)
        idx = idx.reshape(len(Z), self.k)
        ww = self.w[idx]
        return np.einsum("nk,nkm->nm", ww, self.y[idx]) / ww.sum(1, keepdims=True)


def knn_fit(X: np.ndarray, Y: np.ndarray, w: np.ndarray | None = None, k: int = KNN_K) -> KNN:
    """Standardise X by its (weighted) mean and SD and index it (a constant feature keeps
    SD 1)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    if len(X) < k:
        raise ValueError(f"{len(X)} training states < k = {k}")
    mean = wmean(X, w)
    sd = np.sqrt(wvar(X, w))
    sd = np.where(sd > 0, sd, 1.0)
    return KNN(mean=mean, sd=sd, tree=cKDTree((X - mean) / sd), y=Y, w=w, k=int(k))


@dataclass(frozen=True)
class Residuals:
    """Cross-fitted decoy residuals lambda_res = lambda - f_D (module docstring), with f_D's
    k (the same for every fold and column)."""

    by_fold: np.ndarray       # [F, n, m]: row i's residual when fold f is held out
    fold: np.ndarray          # [n]
    k: int = KNN_K

    @property
    def oof(self) -> np.ndarray:
        """[n, m] each row's residual from the model that held its own fold out."""
        return self.by_fold[self.fold, np.arange(len(self.fold))]

    def training(self, f: int) -> np.ndarray:
        """[n, m] the residuals used when fold f is held out (held-out rows included)."""
        return self.by_fold[f]


def cross_fit(X: np.ndarray, Y: np.ndarray, fold: np.ndarray, w: np.ndarray | None = None,
              k: int = KNN_K) -> Residuals:
    """lambda_res = Y - f_D cross-fitted (module docstring), f_D the ``k``-neighbour mean
    (``KNN_K`` in the pipeline) in every fit, fold and column."""
    if k is None or isinstance(k, bool) or np.ndim(k) or int(k) != k or int(k) < 1:
        raise ValueError(f"f_D's k must be one positive integer, got {k!r}")
    k = int(k)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    fold = np.asarray(fold, int)
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    F = int(fold.max()) + 1
    out = np.full((F, *Y.shape), np.nan)
    for f in range(F):
        test, train = fold == f, fold != f
        out[f, test] = Y[test] - knn_fit(X[train], Y[train], w[train], k).predict(X[test])
        for g in range(F):
            if g == f:
                continue
            fit = train & (fold != g)
            pred = fold == g
            out[f, pred] = Y[pred] - knn_fit(X[fit], Y[fit], w[fit], k).predict(X[pred])
    return Residuals(by_fold=out, fold=fold, k=k)


def residual_share(res: np.ndarray, Y: np.ndarray, w: np.ndarray,
                   direction: np.ndarray | None = None, names=None) -> np.ndarray:
    """[m] Var_w(res) / Var_w(Y) per target column; for mirror-odd components (``names``)
    with ``direction`` given, both about the within-direction means (R2_wd's convention)."""
    res = np.asarray(res, float)
    Y = np.asarray(Y, float)
    res, Y = (res[:, None], Y[:, None]) if Y.ndim == 1 else (res, Y)
    names = [None] * Y.shape[1] if names is None else list(names)
    out = np.empty(Y.shape[1])
    for j, nm in enumerate(names):
        grp = direction if (direction is not None and nm in MIRROR_ODD) else None
        out[j] = wvar(res[:, j], w, grp) / wvar(Y[:, j], w, grp)
    return out


def decoy_shares(tab: io.Table, lam: np.ndarray, oof: np.ndarray,
                 w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(T's, T's interior's) K_P residual shares (``residual_share``) of the own-fold decoy
    residuals ``oof`` [n, 4] of ``lam``: T with its key weights ``w`` (default the metric's),
    the interior, C2's decision stratum and the power floor's, with its own key weights."""
    lam, oof = np.asarray(lam, float), np.asarray(oof, float)
    w = weights(tab, "key") if w is None else np.asarray(w, float)
    d = np.asarray(tab["direction"])
    inner = np.asarray(tab["interior"], bool)
    wi = weights({c: np.asarray(v)[inner] for c, v in tab.items()}, "key")
    return (residual_share(oof, lam, w, d, K_P),
            residual_share(oof[inner], lam[inner], wi, d[inner], K_P))


def knn_stability(r: Residuals, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """[m] the weighted mean over rows of the SD of a row's F predictions (one per held-out
    fold), relative to the SD of the target."""
    Y = np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    pred = Y[None] - r.by_fold                                    # [F, n, m]
    sd = pred.std(axis=0)
    return wmean(sd, w) / np.sqrt(wvar(Y, w))


def power_floor(share: np.ndarray, names=K_P, floor: float = POWER_FLOOR) -> dict[str, bool]:
    """Which components pass C2's power floor (Section 12 decision 5)."""
    return {nm: bool(v >= floor) for nm, v in zip(names, np.asarray(share), strict=True)}


# ------------------------------------------------------------------------- observability


@dataclass(frozen=True)
class Observability:
    """Per state: s's column-norm ratio and flag, each column's squared residual after
    projection on the others (absolute and relative to its own squared norm), and the
    verdict per component."""

    s_ratio: np.ndarray       # [n]
    s_unobservable: np.ndarray  # [n] bool
    resid_sq: np.ndarray      # [n, 8] (nan for s where dropped)
    rel_resid: np.ndarray     # [n, 8]
    observable: np.ndarray    # [n, 8] bool


def _project_out(A: np.ndarray, b: np.ndarray, rcond: float) -> np.ndarray:
    """||b - P_A b||^2 per state, P_A the projector on A's column space (SVD, relative
    cutoff ``rcond``; the residual vector is formed explicitly, so a column in the others'
    span gives ~eps^2 ||b||^2, not eps ||b||^2). A [n, m, p], b [n, m]."""
    U, sv, _ = np.linalg.svd(A, full_matrices=False)
    keep = sv > rcond * np.maximum(sv[:, :1], 1e-300)
    c = np.einsum("nmp,nm->np", U, b) * keep
    r = b - np.einsum("nmp,np->nm", U, c)
    return np.einsum("nm,nm->n", r, r)


def observability(Jz: np.ndarray, clipped: np.ndarray | None = None, tau_s: float = TAU_S,
                  thresholds: dict[str, float] | None = None,
                  rcond: float = LSTSQ_RCOND) -> Observability:
    """The rule of the module docstring on J_z [n, m, 8] (or one state [m, 8]); ``clipped``
    [n, m] zeroes the rows where the obs clip binds."""
    J = np.array(Jz, float)
    one = J.ndim == 2
    J = J[None] if one else J
    if clipped is not None:
        J = np.where(np.asarray(clipped, bool).reshape(J.shape[:2])[..., None], 0.0, J)
    n, _, p = J.shape
    norms = np.linalg.norm(J, axis=1)                             # [n, 8]
    ratio = norms[:, 0] / np.maximum(np.median(norms, axis=1), 1e-300)
    drop = ratio < tau_s
    resid = np.full((n, p), np.nan)
    for k in range(p):
        if k == 0:
            m = ~drop
            if m.any():
                resid[m, 0] = _project_out(J[m][:, :, 1:], J[m][:, :, 0], rcond)
            continue
        for use_s, m in ((True, ~drop), (False, drop)):
            if not m.any():
                continue
            cols = [j for j in range(p) if j != k and (use_s or j != 0)]
            resid[m, k] = _project_out(J[m][:, :, cols], J[m][:, :, k], rcond)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = resid / np.where(norms > 0, norms, np.nan) ** 2
    obs = np.ones((n, p), bool)
    obs[:, 0] = ~drop
    for k, nm in enumerate(STATE_NAMES):
        if thresholds and nm in thresholds:
            obs[:, k] &= np.nan_to_num(resid[:, k], nan=-1.0) >= float(thresholds[nm])
    if one:
        return Observability(ratio[0:1], drop[0:1], resid[0:1], rel[0:1], obs[0:1])
    return Observability(ratio, drop, resid, rel, obs)
