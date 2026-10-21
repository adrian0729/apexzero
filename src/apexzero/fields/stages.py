"""Bodies of step 2's batch stages (design Sections 4, 7 and 10), as functions of keys and
networks, so that each is built and tested before the tag; scripts/fields.py only adds the
freeze check, logs and output paths.

- **Optimum fields** (``opt_fields_key``; pre-tag: agent-independent on the A2 keys, the
  optimum side of the anchored keys): per key the impulse responses (``kkt.key_fields`` on
  the re-polished key) of c_min, v_apex, s_b and the decision steer, with c_min's explicit
  edge term (c_line = impulse - explicit), and the exclusion flags fixed with them: validity
  and reason per outcome, near-ties, the certified inertia, Gate 4's G* and band, the edge's
  own move at the argmin e(s*) (``measures.edge_move``: 'argmin off plateau'), the adjoint
  residual against ``kkt.ADJOINT_RTOL``, the barrier fallback count and the re-polish record.
  A key that fails to re-polish is a row with ``failed`` (``gates.RepolishFailed``): decided
  before the tag as a separate numerical-failure exclusion ('numerical'), never silently
  replaced by the stored point.
- **Phase ceiling** (``phase_ceiling_rows``): ``measures.phase_ceiling`` over one cell's phase
  keys from the stored optimum fields.
- **Network fields** (``network_key``; post-tag for trained networks, tested here on
  random-init actors and planted policies): the float64 base run from the network's restart
  (A2: the cell's x0; anchored: option A from the det row), Gate 3's teacher-forced check and
  branch consistency where the network has a det row, the VJP fields, the edge measures (G,
  G_out, B, e(s*)), the kappa-clipping flags, the plan-level path split of the curvature
  fields, and the blind-pedal control's fields (H1-typical). A run that exhausts the loop
  raises (``network.LoopExhausted``): a pipeline failure stops the stage.
- **Dose-response** (``dose_sample``, ``dose_network``; the optimum side is
  ``kkt.dose_optimum``): a stratified, seeded 400-key subsample, and per network the secant
  closed-loop runs at the dose amplitudes with the crash amplitude against B / (1 - G).
- **H3** (``h3_base_keys``, ``h3_anchored_keys``, ``h3_lambda``): the full-recipe nominal and
  phase keys on cells 584, 605 and 612 and the W = 400 anchored keys at G2's stored states
  (step 1's ``answer_key`` recipes, into a KeyCache of step 2's own outputs, never step 1's),
  and G2's lambda-grid rollouts with lambda*_lin from the impulse sums
  (``analysis.lambda_star``, ``analysis.lambda_lin``). H3's units start 800 m before the
  corner (G2's stored runs need 320-439 steps), so their loop is ``H3_STEPS`` = 512, checked
  on the stored runs by ``h3_loop_audit``.
- **Analysis** (``edge_key_measures``, ``analyse_edges``): per-key measures with their
  exclusion reasons into H2', H4 and H4-transfer and the coverage table; the run-once guard
  (``confirmatory_guard``) refuses a second confirmatory run.

Network loading (``load_network``): a registry agent's final snapshot or a run directory's
last checkpoint (``agents.load_agent``), as a trained ``network.NetPolicy``: its fields need
``network.unlock_trained``, which only scripts/fields.py's post-tag path calls.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from apexzero.fields import analysis as A
from apexzero.fields import gates as G
from apexzero.fields import keys as K
from apexzero.fields import kkt as KK
from apexzero.fields import measures as MS
from apexzero.fields import network as N
from apexzero.fields import outcomes as O
from apexzero.fields import perturb as PT
from apexzero.fields.perturb import Samples
from apexzero.interp import agents, io, registry

OPT_OUTCOMES = KK.OUTCOMES               # c_min, v_apex, s_b, steer
NET_OUTCOMES = N.OUTCOMES                # the same order
SIGMA_BS = (8.0, 4.0, 16.0)
DOSE_N = 400                             # dose-response subsample (design: 300-500 keys)
DOSE_STRATA = {"A2/400": 100, "A2/800": 40, "v1": 60, "v2": 60, "v3": 60, "S": 80}
NUMERICAL = "numerical"                  # an optimum key that failed to re-polish (excluded)
H3_STEPS = 512     # H3's loop: G2's stored runs need 320-439 steps from the real cells' x0


def _stack(fields: dict[str, Samples], names) -> np.ndarray:
    """[len(names), 3, N] from {outcome: Samples}."""
    return np.stack([np.stack([np.asarray(getattr(fields[o], f), float) for f in PT.FEATURES])
                     for o in names])


def _meta(ref: K.KeyRef) -> dict:
    st = K.strata(ref)
    return {"key_id": ref.key_id, "kind": ref.kind, "agent": ref.agent, "snapshot": ref.snapshot,
            "cell": ref.cell_id, "phase": ref.phase, "stim": ref.stim_id, "W": ref.W,
            "direction": st.direction, "pair_id": st.pair_id, "approach": st.approach,
            "R": st.R, "theta_deg": st.theta_deg, "n_phases": st.n_phases,
            "w_is_cell": st.w_is_cell, "cap_binds": st.cap_binds, "no_cap": st.no_cap}


# ------------------------------------------------------------------------- optimum fields


def opt_fields_key(ref: K.KeyRef, bank=None) -> tuple[dict, dict[str, np.ndarray]]:
    """(row, arrays) of the optimum's fields at one stored key (module docstring); arrays:
    'impulse' and 'explicit' [4 outcomes, 3 features, N]. A key that cannot be re-polished
    or rebuilt gives a ``failed`` row and no arrays."""
    if bank is None:
        bank, _ = K.design()
    meta = _meta(ref)
    try:
        prob, stim, rps = G.problem(ref, bank)
    except Exception as e:  # noqa: BLE001  (decided before the tag: a 'numerical' exclusion)
        return {**meta, "failed": True, "reason": NUMERICAL,
                "error": f"{type(e).__name__}: {e}"}, {}
    fl = KK.key_fields(prob, bank=bank)
    lm, sig = stim.landmarks, int(stim.direction)
    g_star = KK.edge_gain(fl, lm, sig)
    pick = fl.reads["c_min"].plan.pick if fl.valid("c_min") else None
    e_star = MS.edge_move(lm, prob.track.n_points, prob.x[:, 0], pick)
    inr = fl.inertia
    row = {**meta, "failed": False, **G._repolish_record(rps), **G._field_checks(fl),
           "decision": fl.decision, "mu": fl.mu,
           "valid": {o: bool(fl.valid(o)) for o in OPT_OUTCOMES},
           "reasons": {o: fl.reads[o].reason for o in OPT_OUTCOMES},
           "values": {o: float(fl.reads[o].value) for o in OPT_OUTCOMES},
           "near_tie": {o: bool(fl.reads[o].plan.near_tie) if fl.reads[o].plan else False
                        for o in OPT_OUTCOMES},
           "s_star": float(fl.reads["c_min"].plan.s_at) if fl.valid("c_min") else float("nan"),
           "G_star": g_star, "in_band": KK.gate4(g_star), "e_star": e_star,
           "on_plateau": bool(np.isfinite(e_star) and e_star >= MS.EDGE_PLATEAU),
           "inertia_failed": bool(inr is None or not inr.ok),
           "s1": K.bump_origin(prob.rec), "s_c0": lm.s_c0, "s_c1": lm.s_c1,
           "n_samples": int(prob.track.n_points)}
    arrays = {"impulse": _stack(fl.impulse, OPT_OUTCOMES),
              "explicit": _stack(fl.explicit, OPT_OUTCOMES)}
    return row, arrays


def save_key_arrays(out_dir: Path, key_id: str, arrays: dict[str, np.ndarray]) -> Path | None:
    """<out_dir>/<key_id>.npz (compressed; the impulse responses are zero past the
    support)."""
    if not arrays:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{key_id.replace('/', '_')}.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_key_arrays(out_dir: Path, key_id: str) -> dict[str, np.ndarray]:
    with np.load(Path(out_dir) / f"{key_id.replace('/', '_')}.npz") as z:
        return {k: z[k] for k in z.files}


def opt_fields(refs, out_dir: Path, bank=None, progress=None) -> list[dict]:
    """``opt_fields_key`` over keys, arrays saved under ``out_dir``; the rows."""
    if bank is None:
        bank, _ = K.design()
    rows = []
    for ref in refs:
        row, arrays = opt_fields_key(ref, bank)
        save_key_arrays(out_dir, ref.key_id, arrays)
        rows.append(row)
        if progress:
            progress(1)
    return rows


def samples_of(arr: np.ndarray, j: int) -> Samples:
    """Outcome j's Samples from a [4, 3, N] array."""
    return Samples(*(np.asarray(arr[j, k], float) for k in range(3)))


def phase_ceiling_rows(rows: list[dict], out_dir: Path, outcomes=("s_b", "v_apex", "steer"),
                       sigma_b: float = PT.SIGMA_B[0]) -> list[dict]:
    """``measures.phase_ceiling`` per A2 cell with at least two phase keys (the horizon twins
    left out), per outcome, from the stored optimum fields."""
    by: dict[int, list[dict]] = {}
    for r in rows:
        if not r.get("failed") and r["kind"] != "anchored" and r["w_is_cell"]:
            by.setdefault(int(r["cell"]), []).append(r)
    out = []
    for cid, rs in sorted(by.items()):
        if len(rs) < 2:
            continue
        arr = {r["phase"]: load_key_arrays(out_dir, r["key_id"])["impulse"] for r in rs}
        rec = {"cell": cid, "phases": sorted(arr), "R": rs[0]["R"], "theta_deg": rs[0]["theta_deg"]}
        for o in outcomes:
            j = OPT_OUTCOMES.index(o)
            f = {r["phase"]: (r["s_c0"], r["direction"],
                              samples_of(arr[r["phase"]], j) if r["valid"][o] else None)
                 for r in rs}
            pc = MS.phase_ceiling(f, o, sigma_b)
            rec[o] = {"rho_median": pc["rho_median"], "rho_min": pc["rho_min"],
                      "n_pairs": pc["n_pairs"]}
        out.append(rec)
    return out


# ------------------------------------------------------------------------- networks


def load_network(name: str) -> N.NetPolicy:
    """A registry agent's final snapshot, or a run directory's last checkpoint ('runs/<dir>',
    the H4 and H4-transfer networks), as a trained NetPolicy (marked ``trained``)."""
    if name in agents.AGENTS:
        return N.agent_net(N._final_agent(name))
    spec = registry.AgentSpec(Path(name).name, name, "generalist", False, 0.0, "ckpt",
                              "eval_rdev.json", None)
    path = agents.final_snapshot(spec)
    if path is None:
        raise FileNotFoundError(f"{name}: no finished run with a checkpoint")
    return N.agent_net(agents.load_agent(path, Path(name).name))


def network_start(ref: K.KeyRef, rec, bank, cells, det: dict | None) -> N.Start:
    """The network's restart at a key: A2 keys from the cell's x0, anchored keys under
    option A from the det row of the key's snapshot (``det`` required)."""
    if ref.kind != "anchored":
        return N.a2_start(rec, bank, cells)
    if det is None:
        raise ValueError(f"anchored key {ref.key_id} needs its snapshot's det group")
    return N.anchored_start(rec, det, bank, "A")


def network_key(net: N.NetPolicy, ref: K.KeyRef, bank, cells, det: dict | None = None,
                steps: int = N.STEPS, blind: bool = True, paths: bool = True,
                branch_tol: dict | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    """(row, arrays) of one network's fields at one key (module docstring). arrays:
    'impulse' [4, 3, N]; with ``paths`` 'paths_kappa' [4 paths, 4 outcomes, N] (the
    curvature fields by path); with ``blind`` 'blind_impulse' [4, 3, N]."""
    rec = K.load_key(ref)
    st = network_start(ref, rec, bank, cells, det)
    rd = N.road(bank, st.stim_id)
    base = N.run(net, rd, st, steps=steps)
    if base.exhausted:
        raise N.LoopExhausted(f"{net.name} at {ref.key_id}: alive after {steps} steps")
    row: dict[str, Any] = {**_meta(ref), "net": net.name, "n_steps": base.n_steps,
                           "term_step": base.term_step, "stop_step": base.stop_step,
                           "decision_step": st.decision_step, "s1": st.s1}
    row["gate3"], row["branch"] = None, None
    if det is not None:
        try:
            i = N.det_row(det, rec.cell_id, rec.phase)
        except ValueError:
            i = None
        if i is not None:
            w = N.wiring_check(net, det, i, rd.base, t0=st.offset, s_stop=st.s_stop)
            row["gate3"] = {"ok": w.ok, "mean_err": w.mean_err,
                            "step_excess": w.step_excess.tolist()}
            b = N.branch_consistency(base, det, i, branch_tol or N.BRANCH_TOL)
            row["branch"] = {"ok": b.ok, "end_ok": b.end_ok, "saturation_ok": b.saturation_ok,
                             "events_ok": b.events_ok, "values_ok": b.values_ok,
                             "values": {o: list(v) for o, v in b.values.items()}}
    fl = N.fields(net, rd, base, steps)
    valid = {o: bool(v) for o, v in zip(NET_OUTCOMES, fl.valid, strict=True)}
    row.update(valid=valid,
               reasons={o: base.plans[o].reason for o in O.OUTCOMES} | {"steer": ""},
               values={o: float(v) for o, v in zip(NET_OUTCOMES, fl.values, strict=True)},
               near_tie={o: bool(v) for o, v in zip(NET_OUTCOMES, fl.near_tie, strict=True)},
               vjp_seconds=fl.seconds)
    kc = N.kappa_clip_record(base)
    row["kappa_clip"], row["kappa_clip_before_s_c1"] = kc["flag"], kc["before_s_c1"]
    pick = base.plans["c_min"].pick if valid["c_min"] else None
    em = MS.edge_measures(fl.of("c_min"), fl.of("v_apex"), row["values"]["c_min"], st.landmarks,
                          st.sigma, s_states=base.x[:, 0], pick=pick)
    row.update(G=em.G, G_out=em.G_out, B=em.B, dc_in=em.dc_in, dc_out=em.dc_out,
               dv_in=em.dv_in, e_star=em.e_star, on_plateau=em.on_plateau)
    arrays = {"impulse": np.stack([np.stack([np.asarray(fl.of(o)[k]) for k in range(3)])
                                   for o in NET_OUTCOMES]),
              "explicit_c_min": np.stack([np.asarray(a) for a in N.explicit_c_min(rd, base)])}
    if paths:
        pf = N.path_fields(net, rd, base, steps, total=fl)
        row["paths_sum_rel_err"] = pf["sum_rel_err"]
        arrays["paths_kappa"] = np.stack([np.asarray(pf["parts"][p].kappa) for p in N.PATHS])
    if blind:
        bnet = N.blind_pedal_net(net, base.mean[:, 1])
        bb = N.run(bnet, rd, st, steps=steps)
        bf = N.fields(bnet, rd, bb, steps)
        row["blind"] = {"reproduces": bool(np.array_equal(bb.x, base.x)),
                        "valid": {o: bool(v) for o, v in zip(NET_OUTCOMES, bf.valid,
                                                             strict=True)}}
        arrays["blind_impulse"] = np.stack([np.stack([np.asarray(bf.of(o)[k]) for k in range(3)])
                                            for o in NET_OUTCOMES])
    return row, arrays


def network_fields(net: N.NetPolicy, refs, out_dir: Path, det_of=None, steps: int = N.STEPS,
                   progress=None, **kw) -> list[dict]:
    """``network_key`` over keys; ``det_of(ref)`` gives the det group of the key's snapshot
    (None: no rollouts); arrays saved under <out_dir>/<net name>/."""
    bank, cells = K.design()
    rows = []
    sub = Path(out_dir) / net.name.replace("/", "_").replace("@", "_")
    for ref in refs:
        row, arrays = network_key(net, ref, bank, cells, det_of(ref) if det_of else None,
                                  steps, **kw)
        save_key_arrays(sub, ref.key_id, arrays)
        rows.append(row)
        if progress:
            progress(1)
    return rows


# ------------------------------------------------------------------------- dose-response


def dose_sample(strata: dict[str, int] = DOSE_STRATA, rng: np.random.Generator | None = None
                ) -> list[K.KeyRef]:
    """The dose-response subsample (design Section 4: 300-500 keys): per stratum (Gate 1's
    families) that many keys without replacement on step 2's 'dose_sample' stream, from the
    primary A2 keys and the anchored finals; sorted by stratum, then key id."""
    return G.gate1_sample(strata, MS.stream_rng("dose_sample") if rng is None else rng)


def dose_network(net: N.NetPolicy, rd: N.Road, st: N.Start, amplitudes=KK.DOSE_AMPLITUDES,
                 side: str = "inside", steps: int = N.STEPS, fl: N.Fields | None = None,
                 base: N.Run | None = None) -> dict:
    """The network's dose-response along the trapezoid shift (design Section 4): one
    closed-loop run per amplitude on the secant road (``network.run(..., mode='secant')``),
    its outcomes and whether it crashed before the window's end; the crash amplitude (the
    smallest narrowing of the grid that crashes) against the first-order prediction
    B / (1 - G) from the base fields."""
    base = N.run(net, rd, st, steps=steps) if base is None else base
    fl = N.fields(net, rd, base, steps) if fl is None else fl
    direction = PT.edge_shift(rd.n, st.landmarks, st.sigma, side, 1.0)
    rows = []
    for a in amplitudes:
        r = N.run(net, rd, st, KK.scaled(direction, float(a)), steps, mode="secant")
        if r.exhausted:
            raise N.LoopExhausted(f"dose {a}: alive after {steps} steps")
        rows.append({"amplitude": float(a), "values": r.values().tolist(),
                     "valid": r.valid_outcomes().tolist(),
                     "crashed": bool(r.term_step >= 0 and not r.plans["c_min"].valid),
                     "reason": r.plans["c_min"].reason})
    em = MS.edge_measures(fl.of("c_min"), fl.of("v_apex"), float(fl.values[0]), st.landmarks,
                          st.sigma)
    narrow = sorted(-r["amplitude"] for r in rows if r["amplitude"] < 0 and r["crashed"])
    pred = em.B / (1.0 - em.G) if np.isfinite(em.G) and em.G < 1.0 else float("inf")
    return {"side": side, "rows": rows, "G": em.G, "B": em.B, "crash_narrowing_predicted": pred,
            "crash_narrowing": narrow[0] if narrow else float("inf")}


# ------------------------------------------------------------------------- H3


def h3_cache(out_root: Path):
    """Step 2's own key cache for H3's new solves (never step 1's KEYS_DIR)."""
    from apexzero.interp import answer_key as ak

    root = Path(out_root)
    if root.resolve() == io.KEYS_DIR.resolve():
        raise ValueError("H3's keys go under step 2's outputs, not step 1's key store")
    return ak.KeyCache(root)


def h3_base_keys(cache, cells_ids=A.H3_CELLS, tables=None, q_cv: float = 0.9) -> list:
    """H3's nominal and phase keys (design Section 2): ``answer_key.full_recipe`` per cell,
    its phase cycle, then ``warm_key`` per phase u_1..u_{n-1} (as step 1's phase stage);
    lockbox-guarded by step 1's recipe."""
    from apexzero.interp import answer_key as ak
    from apexzero.interp import measures as M

    bank, cells = K.design()
    tables = M.model_tables() if tables is None else tables
    out = []
    for cid in cells_ids:
        cell = cells[int(cid)]
        ak._guard(bank.stimuli[cell.stim_id])
        nom = ak.full_recipe(bank, cell, cache=cache, tables=tables, q_cv=q_cv)
        out.append(nom)
        P = ak.phase_cycle(bank, cell, nominal=nom)
        c = dataclasses.replace(cell, P=P)
        for k in range(1, int(cell.n_phases)):
            out.append(ak.warm_key(nom, bank, c, "phase", phase=k, cache=cache, tables=tables))
    return out


def h3_anchored_keys(cache, agent: str = A.H3_AGENT, cells_ids=A.H3_CELLS, tables=None,
                     W: int = 400) -> list:
    """H3's W = 400 anchored keys at the agent's stored deterministic states (step 1's
    ``anchored_key`` on the agent's det run per phase, from that phase's base key in
    ``cache``); file reads of the stored rollouts only."""
    from apexzero.interp import answer_key as ak
    from apexzero.interp import measures as M
    from apexzero.interp import rollout

    bank, cells = K.design()
    tables = M.model_tables() if tables is None else tables
    ts = K.final_timesteps(agent)
    groups, _ = rollout.load_rollouts(io.rollout_path(agent, ts))
    det = groups["det"]
    out = []
    for cid in cells_ids:
        cell = cells[int(cid)]
        nom = ak.key_for(cell, "nominal", 0, cache=cache, bank=bank)
        if nom is None:
            raise ValueError(f"cell {cid}: no base key in the H3 cache (run h3_base_keys first)")
        for k in range(int(cell.n_phases)):
            try:
                i = N.det_row(det, cell.cell_id, k)
            except ValueError:
                continue
            traj = M.trajectory_from_rollout(det, i, "det")
            base = ak.anchor_base(cell, k, nom, cache=cache, bank=bank)
            out.append(ak.anchored_key(bank, cell, base, agent, traj, cache, snapshot=str(ts),
                                       phase=k, tables=tables, W=W))
    return out


def h3_cells_with_twins(bank, cells, h3_cells=A.H3_CELLS) -> list[int]:
    """H3's const_width cells and the cells of their real-width twins."""
    twins = set(A.h3_pairs(bank, cells, h3_cells).values())
    return sorted({int(c) for c in h3_cells}
                  | {c for c, cl in cells.items() if cl.stim_id in twins})


def h3_loop_audit(agent: str = A.H3_AGENT, steps: int = H3_STEPS,
                  margin: int = N.HORIZON_MARGIN) -> dict:
    """``network.loop_audit``'s rule for H3's loop: the agent's stored det runs on the H3
    cells and their real-width twins (file reads only) end within ``steps - margin``."""
    bank, cells = K.design()
    det = N.load_det(io.rollout_path(agent, K.final_timesteps(agent)),
                     ("cell", "stim", "phase", "sample", "x", "valid", "term_step", "trunc_step"))
    hs, open_ = [], []
    for cid in h3_cells_with_twins(bank, cells):
        for k in range(int(cells[cid].n_phases)):
            try:
                i = N.det_row(det, cid, k)
            except ValueError:
                continue
            s_stop = float(O.window(bank.stimuli[int(det["stim"][i])].landmarks)[1])
            h, ended = N.det_horizon(det, i, s_stop)
            (hs if ended else open_).append(h)
    limit = int(steps) - int(margin)
    return {"steps": int(steps), "limit": limit, "n": len(hs) + len(open_),
            "max": int(max(hs, default=0)), "n_open": len(open_),
            "ok": bool(hs and not open_ and max(hs) <= limit)}


def h3_lambda(net: N.NetPolicy, cell_id: int, phase: int = 0, lams=A.H3_LAMBDAS,
              steps: int = H3_STEPS) -> dict:
    """H3 at one unit (design Section 7): the closed loop on w(lambda) = w_cw + lambda dw
    (``analysis.width_change``; widths enter strip_arrays linearly, so the secant is exact)
    over the lambda grid, lambda* (the smallest crashing lambda) and lambda*_lin from the
    impulse responses at lambda = 0, with the clearance change split into the edge's and
    the line's moves."""
    bank, cells = K.design()
    pairs = A.h3_pairs(bank, cells, (int(cell_id),))
    st = N.cell_start(cells[int(cell_id)], int(phase), bank)
    rd = N.road(bank, st.stim_id)
    N.road(bank, pairs[int(cell_id)])                    # the twin through the lockbox too
    dw = A.width_change(bank, st.stim_id, pairs[int(cell_id)])
    runs = []
    for lam in lams:
        r = N.run(net, rd, st, KK.scaled(dw, float(lam)), steps, mode="secant")
        if r.exhausted:
            raise N.LoopExhausted(f"H3 lambda {lam}: alive after {steps} steps")
        runs.append({"lambda": float(lam), "crashed": bool(r.term_step >= 0
                                                           and not r.plans["c_min"].valid),
                     "c_min": float(r.values()[0]), "term_step": r.term_step})
    base = N.run(net, rd, st, steps=steps)
    out = {"cell": int(cell_id), "phase": int(phase), "net": net.name, "runs": runs,
           "lambda_star": A.lambda_star([r["lambda"] for r in runs], [r["crashed"] for r in runs])}
    if base.plans["c_min"].valid:
        fl = N.fields(net, rd, base, steps)
        out.update(A.lambda_lin(float(fl.values[0]), fl.of("c_min"), N.explicit_c_min(rd, base),
                                dw))
    return out


# ------------------------------------------------------------------------- analysis


def edge_key_measures(net_rows: list[dict], opt_rows: dict[str, dict], measure: str = "G",
                      require_non_capped: bool = True) -> list[MS.KeyMeasure]:
    """One network's per-key edge measure (G, G_out, B, dv_in) as KeyMeasures with their
    exclusion reasons (``analysis.key_measure``): c_min invalid, Gate 3 failed, the
    optimum's inertia or Gate 4 failed at the key (``opt_rows``: key_id -> optimum row; the
    lawful band's reference), kappa clipping over c_min's reads, argmin off plateau. Horizon
    twins are left out; capped stimuli too unless ``require_non_capped`` is False."""
    out = []
    for r in net_rows:
        if not r.get("w_is_cell", True) or (require_non_capped and r.get("cap_binds")):
            continue
        opt = opt_rows.get(r["key_id"], {})
        numerical = bool(opt.get("failed") or opt.get("adjoint_ok") is False)
        gate4_bad = bool(opt) and not numerical and not (opt.get("in_band")
                                                         and not opt.get("inertia_failed"))
        flags = {"invalid": not r["valid"]["c_min"], "why_invalid": r["reasons"]["c_min"],
                 "gate3_failed": bool(r.get("gate3") and not r["gate3"]["ok"]),
                 "numerical": numerical, "inertia_failed": gate4_bad,
                 "kappa_clipped": bool(r["kappa_clip"]["c_min"]),
                 "off_plateau": bool(r["valid"]["c_min"] and not r["on_plateau"])}
        meta = {"key_id": r["key_id"], "pair_id": r["pair_id"], "approach": r["approach"],
                "direction": r["direction"], "phase": r["phase"], "n_phases": r["n_phases"],
                "R": r["R"], "theta_deg": r["theta_deg"]}
        out.append(A.key_measure(meta, r[measure], flags))
    return out


def median_statistic(rows: list[MS.KeyMeasure], n: int = MS.BOOTSTRAP_N) -> float:
    """A network's median over geometries (H4's statistic), from its key measures."""
    return float(MS.agent_summary(rows, (1, -1), n).median)


def analyse_edges(net_rows: dict[str, list[dict]], opt_rows: dict[str, dict],
                  n: int = MS.BOOTSTRAP_N) -> dict:
    """H2' per step-1 generalist (bands on the median G's CI, per direction and pooled), the
    R18 variant rule, H4 and H4-transfer on the networks' median G, and the coverage table.
    ``net_rows`` maps a network name (registry id or run path) to its A2 rows."""
    km = {name: edge_key_measures(rows, opt_rows) for name, rows in net_rows.items()}
    out: dict[str, Any] = {"H2": {}, "coverage": MS.coverage(
        rec for name, rows in km.items() for rec in A.coverage_records(rows, name, "final", "G"))}
    for name in A.H2_NETWORKS:
        if name in km:
            out["H2"][name] = A.summarise(km[name], MS.h2_decision, n)
    out["H2_variants"] = {v: A.variant_rule({a: d["decision"] for a, d in out["H2"].items()}, v)
                          for v in registry.variants()}
    stats = {name: median_statistic(rows, n) for name, rows in km.items()}
    have = set(stats)
    need4 = {r for d in A.H4_VARIANTS.values() for r in d["real"] + d["baseline"]}
    if need4 <= have:
        out["H4"] = A.h4_test(stats)
    needt = set(A.H4_TRANSFER["real"]) | set(A.H4_TRANSFER["baseline"])
    if needt <= have:
        out["H4_transfer"] = A.h4_transfer(stats)
    out["statistics"] = stats
    return out


def _lm(row: dict):
    """The landmarks the trapezoid needs (s_c0, s_c1) from a row."""
    from apexzero.interp.stimuli import Landmarks

    a, b = float(row["s_c0"]), float(row["s_c1"])
    return Landmarks(a, a, 0.5 * (a + b), b, b)


def _meta_of(r: dict) -> dict:
    return {k: r[k] for k in ("key_id", "pair_id", "approach", "direction", "phase", "n_phases",
                              "R", "theta_deg")}


def field_comparisons(net_rows: list[dict], opt_rows: dict[str, dict], net_dir: Path,
                      opt_dir: Path, outcome: str, impulse_key: str = "impulse",
                      sigma_b: float = PT.SIGMA_B[0]) -> list[dict]:
    """Per key: the network's (``impulse_key``: 'impulse', or 'blind_impulse' for the
    blind-pedal control) and the optimum's curvature field of one outcome, recoded toward the
    turn on the common s_c0-aligned grid (``measures.aligned_field``): gain, rho, location,
    and the shape specificity delta_rho against the optimum's fields at far geometries among
    the same keys (``measures.shape_specificity``), with the exclusion flags."""
    j = NET_OUTCOMES.index(outcome)
    jo = OPT_OUTCOMES.index(outcome)
    d = PT.d_grid(MS.NULL_D[1], MS.NULL_D[0])
    net_f, opt_f, rows = {}, {}, {}
    for r in net_rows:
        o = opt_rows.get(r["key_id"])
        if o is None or o.get("failed"):
            continue
        geo = (r["R"], r["theta_deg"])
        a = load_key_arrays(net_dir, r["key_id"])
        b = load_key_arrays(opt_dir, r["key_id"])["impulse"]
        sn, so = samples_of(a[impulse_key], j), samples_of(b, jo)
        net_f[r["key_id"]] = (geo, o["s_c0"], r["direction"], sn)
        opt_f[r["key_id"]] = (geo, o["s_c0"], r["direction"], so)
        Fn = MS.aligned_field(sn, o["s_c0"], r["direction"], outcome, "kappa_turn", sigma_b)
        Fo = MS.aligned_field(so, o["s_c0"], r["direction"], outcome, "kappa_turn", sigma_b)
        rows[r["key_id"]] = {"row": r, "opt": o, **MS.compare(Fn, Fo, d)}
    for sp in MS.shape_specificity(net_f, opt_f, outcome, sigma_b):
        rows[sp["label"]]["delta_rho"] = sp["delta_rho"]
        rows[sp["label"]]["rho_far"] = sp["rho_far"]
    return list(rows.values())


def _flags(r: dict, o: dict, outcome: str) -> dict:
    return {"invalid": not (r["valid"][outcome] and o["valid"][outcome]),
            "why_invalid": r["reasons"].get(outcome, "") if not r["valid"][outcome] else "",
            "gate3_failed": bool(r.get("gate3") and not r["gate3"]["ok"]),
            "numerical": bool(o.get("failed") or o.get("adjoint_ok") is False),
            "inertia_failed": bool(o.get("inertia_failed")),
            "kappa_clipped": bool(r["kappa_clip"][outcome])}


def h1_typical_rows(cmp: list[dict], ctrl: list[dict], outcome: str, cells=None
                    ) -> tuple[dict, dict]:
    """H1-typical's per-key measures (gain, location, delta_rho) for a network and its
    blind-pedal control, restricted to ``cells`` (the agent's bulk A2 cells; None: all)."""
    def rows_of(items):
        out = {"gain": [], "location": [], "shape": []}
        for c in items:
            r, o = c["row"], c["opt"]
            if cells is not None and r["cell"] not in cells:
                continue
            fl = _flags(r, o, outcome)
            for name, v in (("gain", c["g"]), ("location", c["location"]),
                            ("shape", c.get("delta_rho", float("nan")))):
                out[name].append(A.key_measure(_meta_of(r), v, fl))
        return out

    return rows_of(cmp), rows_of(ctrl)


def analyse_fields(net_rows: list[dict], opt_rows: dict[str, dict], net_dir: Path,
                   opt_dir: Path, cells=None, n: int = MS.BOOTSTRAP_N) -> dict:
    """One network's H1-typical (per outcome, s_b then v_apex, with the paired blind-pedal
    control) and H2-speed (r_in = dv_apex,net / dv_apex,opt along the inside shift) on its A2
    rows (horizon twins left out; ``cells``: its bulk A2 cells for H1-typical)."""
    rows = [r for r in net_rows if r.get("w_is_cell", True)]
    out: dict[str, Any] = {"H1_typical": {}}
    for o in ("s_b", "v_apex"):
        cmp = field_comparisons(rows, opt_rows, net_dir, opt_dir, o)
        ctrl = field_comparisons(rows, opt_rows, net_dir, opt_dir, o, "blind_impulse")
        a, b = h1_typical_rows(cmp, ctrl, o, cells)
        out["H1_typical"][o] = A.h1_typical(a["gain"], a["location"], a["shape"], b, n)
    r_in = []
    for r in rows:
        opt = opt_rows.get(r["key_id"])
        if opt is None or opt.get("failed"):
            continue
        imp = samples_of(load_key_arrays(opt_dir, r["key_id"])["impulse"],
                         OPT_OUTCOMES.index("v_apex"))
        dv_opt = MS.edge_derivative(imp, _lm(opt), r["direction"], "inside")
        fl = _flags(r, opt, "v_apex")
        r_in.append(A.key_measure(_meta_of(r), MS.speed_ratio(r["dv_in"], dv_opt), fl))
    out["H2_speed"] = A.summarise(r_in, A.h2_speed, n)
    return out


def h1_fast_rows(net_rows: list[dict], opt_rows: dict[str, dict], net_dir: Path, opt_dir: Path,
                 agent: str, n: int = MS.BOOTSTRAP_N) -> dict:
    """H1-fast for one agent on its anchored finals: the differences s_b,net - s_b* and
    v_apex,net - v_apex* and the curvature gain of the s_b field (``analysis.h1_fast``)."""
    sb, va = [], []
    for r in net_rows:
        o = opt_rows.get(r["key_id"])
        if o is None or o.get("failed"):
            continue
        for name, store in (("s_b", sb), ("v_apex", va)):
            store.append(A.key_measure(_meta_of(r), r["values"][name] - o["values"][name],
                                       _flags(r, o, name)))
    cmp = field_comparisons(net_rows, opt_rows, net_dir, opt_dir, "s_b")
    gain = [A.key_measure(_meta_of(c["row"]), c["g"], _flags(c["row"], c["opt"], "s_b"))
            for c in cmp]
    return A.h1_fast(sb, va, gain, agent, n)


def confirmatory_guard(out_dir: Path) -> Path:
    """The analysis runs once (design Section 9): refuses when the confirmatory decisions
    file already exists; returns its path."""
    path = Path(out_dir) / "decisions.json"
    if path.exists():
        raise SystemExit(f"refusing to run the confirmatory analysis again: {path} exists "
                         "(design Section 9: the analysis runs once)")
    return path


def write_rows(path: Path, rows: list[dict]) -> Path:
    """Rows as JSON lines (``io.json_bytes`` per row)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.writelines(json.dumps(json.loads(io.json_bytes(r))) + "\n" for r in rows)
    return path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


# ------------------------------------------------------------------------- dry run


def dry_run(cell_a2: int = 344, h3_cell: int = A.H3_CELLS[0], lams=(0.0, 0.5, 1.0)) -> dict:
    """The network stage bodies end to end on policies that are not trained networks (a
    float64 process): ``network_key`` for the planted linear driver (paths and the blind-pedal
    control) and for a random-init actor (G1's architecture and obs_rms, read only) at an A2
    key, ``dose_network`` for the linear driver, ``h3_lambda`` for the planted edge keeper on
    a lambda subgrid, and ``analyse_edges`` on the rows."""
    from apexzero.fields import controls as C

    N.require_x64()
    bank, cells = K.design()
    ref = next(r for r in K.a2_primary_refs() if r.cell_id == cell_a2 and r.phase == 0)
    sigma = int(bank.stimuli[ref.stim_id].direction)
    W, b = N.linear_driver(sigma)
    lin = N.linear_net(W, b, "linear_driver")
    g1 = C.g1_reference()
    rnd = N.random_net(g1.cfg, g1.obs_rms, 0)
    out: dict[str, Any] = {}
    row_l, arr_l = network_key(lin, ref, bank, cells)
    row_r, arr_r = network_key(rnd, ref, bank, cells, paths=False)
    out["linear"] = row_l
    out["random"] = row_r
    out["arrays"] = {"linear": {k: list(v.shape) for k, v in arr_l.items()},
                     "random_impulse_nan_rows": [bool(np.isnan(arr_r["impulse"][j]).all())
                                                 for j in range(4)]}
    rec = K.load_key(ref)
    st = network_start(ref, rec, bank, cells, None)
    rd = N.road(bank, st.stim_id)
    out["dose"] = dose_network(lin, rd, st)
    hst = N.cell_start(cells[h3_cell], 0, bank)
    hrd = N.road(bank, hst.stim_id)
    keeper = C.keeper_net(hst, hrd, True)
    out["h3"] = h3_lambda(keeper, h3_cell, 0, lams)
    out["h3_loop_audit"] = h3_loop_audit()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        orow, oarr = opt_fields_key(ref, bank)
        save_key_arrays(tmp / "opt", ref.key_id, oarr)
        save_key_arrays(tmp / "net", ref.key_id, arr_l)
        opt = {ref.key_id: orow}
        out["analysis"] = analyse_edges({"linear_driver": [row_l], "random": [row_r]}, opt,
                                        n=200)
        out["fields_analysis"] = analyse_fields([row_l], opt, tmp / "net", tmp / "opt", n=200)
        out["comparison"] = {o: {k: v for k, v in c.items() if k not in ("row", "opt")}
                             for o in ("s_b", "v_apex")
                             for c in field_comparisons([row_l], opt, tmp / "net", tmp / "opt", o)}
    return out


def main(argv: list[str] | None = None) -> int:
    """``python -m apexzero.fields.stages dry-run --out FILE``: ``dry_run`` in a float64
    process (x64 switched on here)."""
    import argparse

    import jax

    ap = argparse.ArgumentParser(prog="python -m apexzero.fields.stages")
    ap.add_argument("what", choices=("dry-run",))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    res = dry_run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(io.json_bytes(res))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
