"""Idea 2's pre-registration record (design v2.1, Section 7 'Freeze'): what the tag
``costate-prereg-v1`` fixes besides the code.

The record is ``src/apexzero/costate/prereg.json``, inside the frozen paths, so git freezes it
with the code at the tag and scripts/costate.py's freeze check reads it from the tagged commit
(``git show <tag>:<path>``). It is written last before the tag, from a clean tree, by the
'prereg' stage (``build_record``), and holds (modelled on step 2's ``fields.prereg``,
fields/prereg.py:1-60, simplified where idea 2 needs less):

- **the state lists**: the A2 key set (``keys.a2_refs``: 481 sorted (kind, key id) pairs) with
  the content hash of its .json and .npz files (``fields.prereg.content_hash``), N's anchored
  finals per agent (``states.n_refs``) with theirs, and T's (key id, t) list after K2's cut
  with its sha256 (``t_hash``); D is T at t = 0;
- **the frozen values** computed before the tag (``frozen_values``): the winsorisation bounds
  of every target column on T, the K_P residual shares and the power-floor verdicts, partition
  0's fold of each geometry and the sha256 of the 10 robustness partitions, t_H, the kappa
  sensitivity set (the geometries with R <= ``KAPPA_SET_R``), the interior share by phase;
- **the constants** the gates, nulls and decisions use (``constants``: a copy for the reader;
  the code hash binds them): tolerances, thresholds (interior k, clearance and slip margins,
  tau_s, the pedal-saturated tolerance), the penalty grid, Huber's constant, the bands, the
  seeds and streams, the null definitions, C3's strata, covariates and pre-signed pairs, the
  value-accuracy threshold;
- **the deferred analyses** (``deferred``, ``DEFERRED``; design Section 13 'Deferred'):
  C3-int, C1-dev, Delta C1, C2-pairs and C1-own on its own-run and N states are not part of
  costate-prereg-v1 (exploratory if run later). C3-int's permutations and pre-signed pairs
  (in the constants) and C1-dev's snapshot pins stay in the record, labelled there as kept
  without pre-registering those analyses;
- **every network** a post-tag stage loads (``network_pins``: the 31 generalists of C3-w's
  strata and the 3 specialists; ``snapshot_pins``: the snapshots of the 9 decision networks
  that the deferred C1-dev would read): its checkpoint's repo-relative path (built without
  resolving symlinks, so a worktree whose runs/ links to the main checkout pins runs/...),
  sha256 (which covers the head subtrees Dense_1-Dense_10) and timesteps; post-tag loaders
  load the recorded path and check its hash (``pins_problems`` re-hashes networks and
  snapshots alike);
- **the bound pre-tag outputs** (``bind_outputs``): every ``BOUND_OUTPUTS`` file under
  outputs/costate with its sha256 and its stamp's code hash; each must carry a stamp of this
  code, not be partial, and pass (K1 on its ``pass_with_plants``; the nulls summary with
  every decision network, every null finite and the sha256 of each array file; inputs with
  K2's t_H); and the arrays the post-tag analysis reads (``bind_arrays``: inputs/world.npz
  and each decision network's nulls/<net>.npz, with their stamps' sha256);
- **C3-w's inputs** (``c3w_inputs``): per generalist the sha256 of eval_rdev.json and
  metrics.csv and the computed y (completion) and competence, which the post-tag C3-w reads
  from the record, and step 1's stimulus design hash (``stimuli.design_hash``), which the
  post-tag roads are rebuilt from;
- the design document's sha256, ``code_hash`` over src/ (all of it), scripts/costate.py and
  scripts/_bootstrap.py (``fields.prereg.code_hash``'s rule on ``CODE_ROOTS``), the commit the
  record was built on (``prereg_base_commit``), step 1's and step 2's tags and the
  environment (Python and package versions).

``verify_record`` recomputes each item from disk and lists every mismatch as a
``fields.prereg.Problem`` with a code; only 'code_changed' and 'design_doc_changed' may be
waived by a logged deviation (``WAIVABLE``). Post-tag stages iterate ``frozen_ids`` and
``frozen_t``, never a fresh glob.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from apexzero.fields import prereg as FP
from apexzero.fields.prereg import Problem, read_stamp, sha256_file

PREREG_NAME = "prereg.json"
PREREG_PATH = Path(__file__).with_name(PREREG_NAME)
PREREG_REL = f"src/apexzero/costate/{PREREG_NAME}"
DESIGN_DOC = "docs/interp/costate-design.md"
CODE_ROOTS = ("src", "scripts/costate.py", "scripts/_bootstrap.py")
VERSION = 1
WAIVABLE = FP.WAIVABLE
KAPPA_SET_R = 8.94            # m: Section 4's kappa-clipping sensitivity set (R <= this)
STAMP_SUFFIX = FP.STAMP_SUFFIX
# The pre-tag outputs the record binds (relative to outputs/costate) and their pass keys.
BOUND_OUTPUTS = ("inputs/summary.json", "gates/k0.json", "gates/k1.json", "gates/k2.json",
                 "gates/k2b.json", "gates/k3.json", "gates/k3b.json", "gates/k4.json",
                 "gates/k5.json", "gates/k6.json", "nulls/summary.json")
BOUND_ARRAYS = ("inputs/world.npz",)   # plus nulls/<net>.npz per decision network
# Design Section 13 'Deferred': the analyses that are not part of costate-prereg-v1 (if run
# later, they are exploratory), as (name, what, the record's items kept for them, which pin
# their inputs without pre-registering them). The record lists them ('deferred',
# ``deferred``); a later deferral is added here and to Section 13.
DEFERRED = (
    ("C3-int", "the network x geometry interaction test (Section 6; ``analysis.c3_int``)",
     ("constants.analysis.c3_int_perm", "constants.analysis.c3_int_pairs")),
    ("C1-dev", ("C1's R2_c across each decision network's snapshots (Section 6; "
                "``analysis.c1_dev``)"), ("snapshots",)),
    ("DeltaC1", "R2(z + g) - R2(z), conditional probing (Section 6; ``analysis.delta_c1``)",
     ()),
    ("C2-pairs", "matched-decoy row pairs within a geometry (Section 6; ``analysis.c2_pairs``)",
     ()),
    ("C1-own on own-run and N states",
     ("C1-own on every state of each network's own run from each D key and on N (Section 3.3 "
      "(ii), (iii)): it needs critic pullbacks at own-run states; C1-own is decided on T's "
      "interior restarts only"), ()),
)


def net_file(name: str) -> str:
    """A network's output file stem: its run directory (``analysis.network_of``), '/' -> '_'
    (so 'G1' and 'runs/dr-pilot-200M' name the same file)."""
    from apexzero.costate import analysis as AN

    return AN.network_of(name).replace("/", "_")


def null_networks() -> list[str]:
    """The networks C1's nulls are computed for: the decision networks' run directories."""
    from apexzero.costate import analysis as AN

    return [AN.network_of(a) for a in AN.DECISION_NETWORKS]


def code_hash(repo: Path) -> str:
    return FP.code_hash(Path(repo), CODE_ROOTS)


def tree_code_hash(repo: Path, rev: str) -> str:
    return FP.tree_code_hash(Path(repo), rev, CODE_ROOTS)


def _canon(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_default).encode()).hexdigest()


def t_hash(pairs: list[tuple[str, int]]) -> str:
    """sha256 of T's sorted (key id, t) list."""
    return _canon(sorted([str(k), int(t)] for k, t in pairs))


def key_sets(keys_root: Path | None = None) -> dict[str, list[tuple[str, str]]]:
    """{'a2': A2 keys, 'n:<agent>': each agent's anchored finals} as sorted (kind, id)."""
    from apexzero.costate import states as ST

    out = {"a2": sorted((r.kind, r.key_id) for r in ST.a2_refs(keys_root))}
    for a, refs in ST.n_refs(keys_root).items():
        out[f"n:{a}"] = sorted((r.kind, r.key_id) for r in refs)
    return out


def constants() -> dict:
    """The gates', nulls' and decisions' constants (a copy for the reader)."""
    from apexzero.costate import STREAMS
    from apexzero.costate import analysis as AN
    from apexzero.costate import critic as CR
    from apexzero.costate import gates as GT
    from apexzero.costate import nulls as NL
    from apexzero.costate import own as OW
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    return {
        "targets": {"active_tol": TG.ACTIVE_TOL, "saturated_tol": TG.SATURATED_TOL,
                    "route_tol": TG.ROUTE_TOL, "x0_tol": TG.X0_TOL, "fold_tol": TG.FOLD_TOL,
                    "sign_tol": TG.SIGN_TOL, "action_tol": TG.ACTION_TOL,
                    "straight_tol": TG.STRAIGHT_TOL, "plant_min": TG.PLANT_MIN,
                    "fd_h": TG.FD_H, "pedal_ladder": list(TG.PEDAL_LADDER),
                    "pedal_up": TG.PEDAL_UP, "pedal_min_h": TG.PEDAL_MIN_H,
                    "k1": [TG.K1_MEDIAN, TG.K1_P95], "plants": list(TG.PLANTS),
                    "kink_factor": TG.KINK_FACTOR, "zero_atol": TG.ZERO_ATOL},
        "states": {"every": ST.EVERY, "exit_margin": ST.EXIT_MARGIN,
                   "interior_k": ST.INTERIOR_K, "interior_k_descriptive": ST.INTERIOR_K_DESCRIPTIVE,
                   "clearance_min": ST.CLEARANCE_MIN, "slip_margin_min": ST.SLIP_MARGIN_MIN,
                   "folds": ST.N_FOLDS, "inner": ST.N_INNER, "partitions": ST.N_PARTITIONS,
                   "knn_k": ST.KNN_K, "power_floor": ST.POWER_FLOOR, "tau_s": ST.TAU_S,
                   "winsor": list(ST.WINSOR), "small_geometry": ST.SMALL_GEOMETRY_KEYS,
                   "sw_stratum": ST.SW_STRATUM},
        "probes": {"grid": list(PB.GRID), "h_grid_inf": True, "min_sd": PB.MIN_SD,
                   "n_boot": PB.N_BOOT, "ci": PB.CI_LEVEL, "c2_band": PB.C2_BAND,
                   "random_actors": PB.N_RANDOM_ACTORS, "random_critics": PB.N_RANDOM_CRITICS,
                   "mlp": [PB.MLP_HIDDEN, list(PB.MLP_ALPHAS), PB.MLP_MAXITER],
                   "control": [PB.CONTROL_HIDDEN, PB.CONTROL_GAIN, PB.CONTROL_TOL,
                               PB.CONTROL_STEPS], "streams": [PB.BOOT_STREAM, PB.MLP_STREAM]},
        "analysis": {"huber_c": AN.HUBER_C, "cal_slope": AN.CAL_SLOPE, "k_lev": AN.K_LEV,
                     "c3w_rows": AN.C3W_ROWS, "w1_rows": AN.W1_ROWS,
                     "c1": [AN.C1_CONFIRM, AN.C1_REFUTE],
                     "scale_band": list(AN.SCALE_BAND), "probe_fit": AN.PROBE_FIT,
                     "c3": [AN.C3_PERMUTATIONS, AN.C3_ALPHA, AN.C3_STRONG],
                     # C3-int is deferred (``DEFERRED``): kept, not pre-registered
                     "c3_int_perm": AN.C3_INT_PERMUTATIONS, "c3_strata": AN.C3_STRATA,
                     "c3_int_pairs": [list(p) for p in AN.C3_INT_PAIRS],
                     "value_accuracy": {
                         "r2_min": AN.VALUE_ACCURACY_R2, "target": AN.VALUE_TARGET,
                         "target_is": "the deterministic policy's own discounted return "
                                      "(own.own_return: delta = 0, s_stop = +inf)",
                         "rows": AN.VALUE_ROWS,
                         "rows_are": f"T's interior restarts every {OW.RESTART_EVERY}th step and D "
                                     "(every key's t = 0); crashed runs recorded and dropped",
                         "steps": OW.V_STEPS, "to": "the strip's end at most",
                         "every_row_scored": True,
                         "fit": "out of fold: C1's per-fold affine calibration (slope >= 0) "
                                "on the frozen folds, key weights on the rows",
                         "gates": "C1-own's validity",
                         "descriptive": {"target": AN.VALUE_TARGET_VSTAR,
                                         "rows": AN.VALUE_ROWS_VSTAR}},
                     "strata": {h: list(v) for h, v in AN.STRATA.items()},
                     "strata_per_network_columns": list(AN.PER_NETWORK_COLUMNS),
                     "strata_seed": AN.STRATUM_SEED,
                     "decision_networks": list(AN.DECISION_NETWORKS),
                     "c3_covariates": ["final training episode_return", "step 2's B"]},
        "nulls": {"names": list(AN.NULL_NAMES), "gamma": NL.GAMMA,
                  "barrier": [NL.BARRIER_SCALE, NL.BARRIER_EPS, NL.EDGE_MARGIN],
                  "qss_cv": 1.0, "value_mlp_alphas": list(NL.VALUE_MLP_ALPHAS),
                  "value_target": "horizon-corrected V*inf (value_ridge_window: V*_t)",
                  "value_grid_min": NL.VALUE_GRID_MIN},
        "critic": {"fd_tol": CR.FD_TOL, "plant_tol": CR.PLANT_TOL, "linear_tol": CR.LINEAR_TOL,
                   "random_seeds": list(CR.RANDOM_SEEDS), "pi_h": CR.PI_H,
                   "k3_min_states": CR.K3_MIN_STATES,
                   # K3's pass rule (design v2.2 Section 13): per form, and the floor every
                   # planted wrong pullback must exceed
                   "fd_tol_manifold": CR.FD_TOL_MANIFOLD, "fd_tols": dict(CR.FD_TOLS),
                   "jz_tol": CR.FD_TOL, "wrong_min": CR.WRONG_MIN,
                   "wrong_pairs": list(CR.WRONG_PAIRS),
                   "wrong_closed_forms": list(CR.WRONG_CLOSED_FORMS)},
        "own": {"k3b": [OW.K3B_MEDIAN, OW.K3B_P95], "plants": list(OW.OWN_PLANTS),
                "v_steps": OW.V_STEPS, "restart_every": OW.RESTART_EVERY,
                "manifold_at": OW.MANIFOLD_AT, "accel_weights": OW.ACCEL_WEIGHTS},
        "gates": {"k1_kink_share": GT.K1_KINK_SHARE, "k2_tol": GT.K2_TOL,
                  "k2b": [GT.K2B_MARGIN, GT.K2B_SEVERITIES, list(GT.K2B_RAMP), GT.K2B_TOL],
                  "k4": [list(GT.K4_SNR), GT.K4_TOL, GT.K4_ZERO_TOL, GT.K4_READOUT_TOL],
                  "k4_c1": [list(GT.K4_C1_SNR), list(GT.K4_FLOOR), GT.K4_DIRECTION_FLOOR]},
        "streams": dict(STREAMS), "kappa_set_r": KAPPA_SET_R}


def deferred() -> list[dict]:
    """The record's 'deferred' list (``DEFERRED``): each analysis with the record items kept
    for it that do not make it pre-registered."""
    return [{"name": n, "what": w, "kept_not_preregistered": list(kept),
             "status": "not part of costate-prereg-v1: exploratory if run later"}
            for n, w, kept in DEFERRED]


def frozen_values(tab, t_H: int | None, partitions: int | None = None) -> dict:
    """The values computed on T before the tag (module docstring)."""
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    w = ST.weights(tab, "key")
    cols = [c for c in ST.TARGET_COLUMNS if c in tab]
    bounds = {c: list(ST.winsor_bounds(np.asarray(tab[c], float), w)) for c in cols}
    lam = np.stack([ST.winsorise(np.asarray(tab[f"lam_{c}"], float), bounds[f"lam_{c}"])
                    for c in TG.K_P], 1)
    fold = ST.row_folds(tab)
    res = ST.cross_fit(ST.decoy_d1(tab), lam, fold, w)
    d = np.asarray(tab["direction"])
    share = ST.residual_share(res.oof, lam, w, d, TG.K_P)
    inner = np.asarray(tab["interior"], bool)
    wi = ST.weights({k: np.asarray(v)[inner] for k, v in tab.items()}, "key")
    share_int = ST.residual_share(res.oof[inner], lam[inner], wi, d[inner], TG.K_P)
    from apexzero.costate import analysis as AN
    G = ST.geometry_keys(tab)
    uniq = np.unique(G, axis=0)
    parts = {p: ST.geometry_folds(uniq[:, 0], uniq[:, 1], p).tolist()
             for p in range(partitions or ST.N_PARTITIONS)}
    return {"winsor_bounds": bounds, "residual_share": dict(zip(TG.K_P, share.tolist(),
                                                                strict=True)),
            "residual_share_interior": dict(zip(TG.K_P, share_int.tolist(), strict=True)),
            # C2 is decided on the interior stratum, so its floor is applied there (the
            # T-wide verdicts are descriptive)
            "power_floor": ST.power_floor(share_int), "power_floor_T": ST.power_floor(share),
            "c3w_rows": AN.C3W_ROWS, "w1_rows": AN.W1_ROWS, "t_H": t_H,
            "geometries": uniq.tolist(), "fold_partition0": parts[0],
            "partitions_sha256": _canon(parts),
            "kappa_set": [g for g in uniq.tolist() if g[0] <= KAPPA_SET_R],
            "interior_share": float(w @ inner.astype(float)),
            "t_count": len(w)}


def build_record(repo: Path, sets: dict[str, list[tuple[str, str]]], keys_root: Path,
                 t_pairs: list[tuple[str, int]], frozen: dict, extra: dict | None = None) -> dict:
    """The record (module docstring); ``extra`` holds the base commit, network pins, bound
    outputs, environment and the steps' tags."""
    repo = Path(repo)
    if not (repo / DESIGN_DOC).exists():
        raise FileNotFoundError(f"{DESIGN_DOC} is missing: the record registers it")
    rec = {"version": VERSION, "tag": "costate-prereg-v1",
           "key_sets": {k: [list(t) for t in v] for k, v in sets.items()},
           "key_counts": {k: len(v) for k, v in sets.items()},
           "content": {k: FP.content_hash(v, keys_root) for k, v in sets.items()},
           "t_pairs": [[str(k), int(t)] for k, t in sorted(t_pairs)],
           "t_sha256": t_hash(t_pairs), "frozen": frozen, "constants": constants(),
           "deferred": deferred(), "design_doc_sha256": sha256_file(repo / DESIGN_DOC),
           "code_hash": code_hash(repo)}
    rec.update(extra or {})
    return rec


def _repo_rel(path: Path, repo: Path) -> str:
    """``path`` relative to ``repo`` without resolving symlinks (a worktree's runs/ may link
    to the main checkout: the pin stays runs/..., which the post-tag loader opens through the
    same link and whose sha256 guards it); ValueError when it is not under the repo."""
    import os

    p, r = Path(os.path.abspath(path)), Path(os.path.abspath(repo))
    try:
        return p.relative_to(r).as_posix()
    except ValueError:
        try:
            return p.relative_to(r.resolve()).as_posix()
        except ValueError as e:
            raise ValueError(f"{path} is not under the repository {repo}") from e


def pin_of(path: Path, agent_id: str, repo: Path) -> dict:
    """{path (repo-relative), sha256, timesteps} of one checkpoint (reads its pickle only)."""
    from apexzero.interp import agents

    ag = agents.load_agent(path, agent_id)
    return {"path": _repo_rel(path, repo), "sha256": sha256_file(path),
            "timesteps": int(ag.timesteps)}


def network_pins(names, repo: Path) -> tuple[dict, list[str]]:
    """{name: {path, sha256, timesteps}} (``fields.stages.checkpoint_path`` and the pickle;
    evaluates nothing) and the problems: no finished checkpoint, or one outside the
    repository."""
    from apexzero.fields import stages as S

    pins, probs = {}, []
    for n in dict.fromkeys(names):
        try:
            path = S.checkpoint_path(n)
            if path is None:
                raise FileNotFoundError(f"{n}: no finished run with a checkpoint")
            pins[n] = pin_of(path, S.network_agent_id(n), repo)
        except (FileNotFoundError, ValueError) as e:
            probs.append(str(e))
    return pins, probs


def snapshot_files(run: str, repo: Path) -> list[str]:
    """Every checkpoint of a run directory (C1-dev's 16 snapshots), repo-relative, sorted."""
    from apexzero.interp import registry

    spec = registry.AGENTS.get(run)
    d = Path(repo) / (spec.run if spec is not None else run) / "ckpt"
    return sorted(_repo_rel(p, repo) for p in d.glob("*.pkl")) if d.is_dir() else []


def snapshot_pins(run: str, repo: Path) -> tuple[dict, list[str]]:
    """C1-dev's snapshots of one network pinned like the networks: {'<agent>@<timesteps>':
    {path, sha256, timesteps}} (``pins_problems`` re-hashes them; their loader checks the
    hash) and the problems."""
    from apexzero.fields import stages as S

    out, probs = {}, []
    for rel in snapshot_files(run, repo):
        try:
            pin = pin_of(Path(repo) / rel, S.network_agent_id(run), repo)
        except (OSError, ValueError) as e:
            probs.append(f"{run}: {rel}: {e}")
            continue
        out[f"{run}@{pin['timesteps']}"] = pin
    if not out:
        probs.append(f"{run}: no snapshot to pin")
    return out, probs


def c3w_inputs(nets, repo: Path, completion, competence) -> tuple[dict, list[str]]:
    """C3-w's y and covariate per generalist, with the sha256 of the files they come from
    (``completion(net)``, ``competence(net)``: scripts/costate.py's readers)."""
    from apexzero.costate import analysis as AN

    out, probs = {}, []
    for n in nets:
        run = Path(repo) / AN.network_of(n)
        try:
            out[n] = {"eval_rdev_sha256": sha256_file(run / "eval_rdev.json"),
                      "metrics_sha256": sha256_file(run / "metrics.csv"),
                      "completion": float(completion(n)), "competence": float(competence(n))}
        except (OSError, ValueError, KeyError) as e:
            probs.append(f"{n}: C3-w input: {e}")
    return out, probs


def c3w_problems(rec: dict, repo: Path) -> list[str]:
    """C3-w's recorded input files re-hashed."""
    from apexzero.costate import analysis as AN

    out = []
    for n, v in sorted((rec.get("c3w_inputs") or {}).items()):
        run = Path(repo) / AN.network_of(n)
        for f, k in (("eval_rdev.json", "eval_rdev_sha256"), ("metrics.csv", "metrics_sha256")):
            p = run / f
            if not p.exists() or sha256_file(p) != v.get(k):
                out.append(f"{n}: {f} is not the recorded one")
    return out


def stimulus_design_hash() -> str:
    """Step 1's stimulus design hash (``stimuli.design_hash`` of the bank and cells the
    post-tag roads are rebuilt from): the cells as ``stimuli.load_design``'s list, in
    ``keys._design``'s order, as step 2 hashes them (scripts/fields.py:410-414);
    ``keys.design`` gives them as a dict by id, which ``design_hash`` cannot take."""
    from apexzero.costate import stimuli_dir
    from apexzero.interp import stimuli

    bank, cells = stimuli.load_design(stimuli_dir())
    return stimuli.design_hash(bank, cells)


def _passes(rel: str, d) -> bool:
    if not isinstance(d, dict):
        return False
    if rel == "inputs/summary.json":
        return bool(d.get("n")) and not d.get("failed") and d.get("t_H") is not None
    if rel == "nulls/summary.json":
        nets = d.get("networks") or {}
        need = {net_file(n) for n in null_networks()}
        have = {net_file(n) for n in nets}
        return bool(need <= have and not d.get("failed")
                    and all(all(v.get("finite", {}).values()) and v.get("sha256")
                            for v in nets.values()))
    if rel == "gates/k1.json":
        dec = d.get("decision")
        return bool(isinstance(dec, dict) and dec.get("pass_with_plants"))
    if rel in ("gates/k6.json",):
        return bool(d.get("n"))
    for k in ("pass", "summary"):
        if k in d:
            v = d[k]
            return bool(v.get("pass") if isinstance(v, dict) else v)
    dec = d.get("decision")
    return bool(dec.get("pass")) if isinstance(dec, dict) else False


def bind_outputs(out_root: Path, code_hash_now: str) -> tuple[dict, list[str]]:
    """({rel: {sha256, code_hash}}, problems) of the bound pre-tag outputs (module docstring)."""
    root = Path(out_root)
    out: dict = {}
    probs: list[str] = []
    for rel in BOUND_OUTPUTS:
        p = root / rel
        if not p.exists():
            probs.append(f"{rel} is missing")
            continue
        st = read_stamp(p)
        if st is None:
            probs.append(f"{rel} has no provenance stamp")
        else:
            if st.get("code_hash") != code_hash_now:
                probs.append(f"{rel} was written by other code")
            if st.get("partial"):
                probs.append(f"{rel} is a partial run ({st['partial']})")
        try:
            ok = _passes(rel, json.loads(p.read_text()))
        except (ValueError, KeyError, TypeError):
            ok = False
        if not ok:
            probs.append(f"{rel} does not pass")
        out[rel] = {"sha256": sha256_file(p), "code_hash": None if st is None
                    else st.get("code_hash")}
    probs += _cross_problems(root)
    return out, probs


def _cross_problems(root: Path) -> list[str]:
    """Consistency between bound outputs: inputs' t_H is K2's, and the nulls summary's array
    hashes are the files on disk."""
    out = []
    try:
        inp = json.loads((root / "inputs/summary.json").read_text())
        k2 = json.loads((root / "gates/k2.json").read_text())
        if inp.get("t_H") is None or inp.get("t_H") != k2["horizon"]["t_H"]:
            out.append(f"inputs/summary.json's t_H {inp.get('t_H')} is not K2's "
                       f"{k2['horizon']['t_H']}: rerun inputs after k2")
    except (OSError, ValueError, KeyError, TypeError):
        pass                       # a missing file is reported by bind_outputs
    try:
        nl = json.loads((root / "nulls/summary.json").read_text())
        for n, v in (nl.get("networks") or {}).items():
            p = root / "nulls" / f"{net_file(n)}.npz"
            if not p.exists() or sha256_file(p) != v.get("sha256"):
                out.append(f"nulls/{net_file(n)}.npz is not the array its summary hashed")
    except (OSError, ValueError, TypeError):
        pass
    return out


def bind_arrays(out_root: Path, code_hash_now: str, nets=None) -> tuple[dict, list[str]]:
    """({rel: {sha256, code_hash}}, problems) of the arrays the post-tag analysis reads
    (``BOUND_ARRAYS`` and nulls/<net>.npz for every decision network, ``nets``): each must
    exist with a sidecar stamp of this code, not partial, whose recorded sha256 is the
    file's."""
    root = Path(out_root)
    nets = null_networks() if nets is None else nets
    rels = list(BOUND_ARRAYS) + [f"nulls/{net_file(n)}.npz" for n in nets]
    out: dict = {}
    probs: list[str] = []
    for rel in rels:
        p = root / rel
        if not p.exists():
            probs.append(f"{rel} is missing")
            continue
        st = read_stamp(p)
        sha = sha256_file(p)
        if st is None:
            probs.append(f"{rel} has no provenance stamp")
        else:
            if st.get("code_hash") != code_hash_now:
                probs.append(f"{rel} was written by other code")
            if st.get("partial"):
                probs.append(f"{rel} is a partial run ({st['partial']})")
            if (st.get("files") or {}).get(p.name) != sha:
                probs.append(f"{rel} changed after it was stamped")
        out[rel] = {"sha256": sha, "code_hash": None if st is None else st.get("code_hash")}
    return out, probs


def verify_record(rec: dict, repo: Path, keys_root: Path,
                  current_sets: Callable[[], dict] | None = None,
                  out_root: Path | None = None,
                  networks: Callable[[dict], list[str]] | None = None,
                  design_hash: str | None = None) -> list[Problem]:
    """Every way the disk differs from the tagged record; [] when it matches (module
    docstring)."""
    out: list[Problem] = []
    repo = Path(repo)
    if int(rec.get("version", -1)) != VERSION:
        out.append(Problem("record", f"record version {rec.get('version')} is not {VERSION}"))
    sets = {k: [tuple(t) for t in v] for k, v in rec.get("key_sets", {}).items()}
    for k, ids in sets.items():
        if FP.content_hash(ids, keys_root) != rec.get("content", {}).get(k):
            out.append(Problem("keys", f"key set {k}: the key files differ from the tagged ones"))
    if current_sets is not None:
        now = current_sets()
        for k in sorted(set(now) | set(sets)):
            if [tuple(t) for t in now.get(k, [])] != sets.get(k, []):
                out.append(Problem("keys", f"key set {k}: the index no longer gives the tagged "
                                   "list"))
    if t_hash([tuple(p) for p in rec.get("t_pairs", [])]) != rec.get("t_sha256"):
        out.append(Problem("record", "T's list does not match its hash"))
    doc = repo / DESIGN_DOC
    if not rec.get("design_doc_sha256"):
        out.append(Problem("record", "the record registers no design document"))
    elif not doc.exists() or sha256_file(doc) != rec["design_doc_sha256"]:
        out.append(Problem("design_doc_changed", f"{DESIGN_DOC} differs from the tagged design"))
    if code_hash(repo) != rec.get("code_hash"):
        out.append(Problem("code_changed", "code changed after the record was written"))
    if out_root is not None:
        for rel, v in (rec.get("outputs") or {}).items():
            p = Path(out_root) / rel
            if not p.exists() or sha256_file(p) != v["sha256"]:
                out.append(Problem("outputs", f"{rel} differs from the tagged output"))
        if not rec.get("outputs"):
            out.append(Problem("record", "the record binds no pre-tag outputs"))
    if networks is not None:
        pins = rec.get("networks")
        if not pins:
            out.append(Problem("record", "the record pins no network checkpoint"))
        else:
            out += [Problem("checkpoints", m) for m in networks(pins)]
        snaps = {k: v for d in (rec.get("snapshots") or {}).values() for k, v in d.items()}
        if snaps:
            out += [Problem("checkpoints", m) for m in networks(snaps)]
        out += [Problem("c3w", m) for m in c3w_problems(rec, repo)]
    if design_hash is not None and design_hash != rec.get("design_hash"):
        out.append(Problem("design", "step 1's stimulus design differs from the recorded one"))
    return out


def pins_problems(pins: dict, repo: Path) -> list[str]:
    """Each recorded checkpoint re-hashed (path and sha256)."""
    out = []
    for name, pin in sorted(pins.items()):
        p = Path(repo) / pin["path"]
        if not p.exists() or sha256_file(p) != pin["sha256"]:
            out.append(f"{name}: the checkpoint {pin['path']} is not the recorded one")
    return out


def frozen_ids(rec: dict, name: str) -> list[tuple[str, str]]:
    return [tuple(t) for t in rec["key_sets"][name]]


def frozen_t(rec: dict) -> list[tuple[str, int]]:
    return [(str(k), int(t)) for k, t in rec["t_pairs"]]


def save(rec: dict, path: Path = PREREG_PATH) -> Path:
    path = Path(path)
    path.write_text(json.dumps(rec, indent=1, sort_keys=True, default=_default) + "\n")
    return path


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")
