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
  branch consistency (per outcome, ``branch_record``) where the network has a det row (step
  1's rollouts, or for a network without them its float32 harness runs from the post-tag
  'harness-f32' stage), the VJP fields, the edge measures (G, G_out, T = G + G_out, B,
  e(s*)), the kappa-clipping flags, each outcome's support end (``support_ends``), the
  plan-level path split of the curvature fields, and the blind-pedal control's fields
  (H1-typical). Each row carries its arrays file's sha256, so the stamped rows bind the
  arrays the analysis reads (``arrays_problems``). A run that exhausts the loop is that key's
  pipeline failure (``failure_row``, design Section 15); the stage goes on. The same body
  runs each earlier snapshot on its own anchored keys ('fields-earlier') and, without Gate 3
  and branch checks on another network's states ('off_policy'), the H4 networks on the
  pooled anchored finals ('fields-anchored-h4').
- **gate2-trained** (``gate2_trained_sample``, ``gate2_trained_network``; post-tag): Gate 2's
  rule on every H2', H4 and H4-transfer network at 12 primary non-capped A2 keys drawn on the
  'gate2_trained' stream (listed in the record), along both trapezoids and the corner bumps
  of kappa and the inside width; a key whose loop is exhausted fails the network's check
  (Gate 2's rule: every row passes) without aborting the stage; a failing network is
  descriptive.
- **Readers** (``reader_fields``; pre-tag, untrained): Qp's fields on the primary A2 keys,
  H1-typical's lawful-reader reference.
- **Dose-response** (``dose_sample``, ``dose_network``; the optimum side is
  ``kkt.dose_optimum``): a stratified, seeded 400-key subsample (listed in the record), and
  per network of ``analysis.DOSE_NETWORKS`` the secant closed-loop runs at the dose
  amplitudes with the crash amplitude against B / (1 - G).
- **H3** (``h3_base_keys``, ``h3_anchored_keys``, ``h3_lambda``): the full-recipe nominal and
  phase keys on cells 584, 605 and 612 and the W = 400 anchored keys at G2's stored states
  (step 1's ``answer_key`` recipes, into a KeyCache of step 2's own outputs, never step 1's),
  and G2's lambda-grid rollouts with lambda*_lin from the impulse sums
  (``analysis.lambda_star``, ``analysis.lambda_lin``). H3's units start 800 m before the
  corner (G2's stored runs need 320-439 steps), so their loop is ``H3_STEPS`` = 512, checked
  on the stored runs by ``h3_loop_audit``.
- **Analysis** (``key_flags``, ``edge_key_measures``, ``analyse_edges``, ``analyse_fields``,
  ``h1_fast_rows``): per-key measures with their exclusion reasons (the pipeline failures
  first: loop exhausted, no optimum row, branch consistency) into H2', H4, H4-transfer,
  H2-speed and the descriptive H1-typical and H1-fast, the coverage tables, T, the capped
  stratum and H4's per-regime breakdown, and the other descriptive items of design Section
  7 (``fields_descriptive``: sigma_b 4 and 16 m and steer_0; ``anchored_edges``: H2' on the
  anchored finals and the earlier snapshots; ``anchored_h4``: H4 on the pooled anchored
  finals; the asymmetry index of every reported measure, ``analysis.summarise``;
  ``near_tie_counts``; the dose ratios in ``dose_network``); ``decide`` False gives the
  coverage and plumbing only (the non-confirmatory analysis before the confirmatory run).
  H1-typical's fields are compared on the s_c0-aligned grid (Section 15), H1-fast's gain and
  steer_0 on the s_1-anchored one (Sections 4 and 6; ``field_comparisons`` 'grid'). The
  run-once rules: ``confirmatory_input_problems`` (complete, recorded inputs, with the arrays
  their rows bind and gate2-trained against the record), ``confirmatory_done`` (what unlocks
  a non-confirmatory analysis's decisions) and ``confirmatory_guard``.

Network loading (``load_network``): a network's recorded checkpoint (post-tag: the record's
path, its sha256 checked right before the load), or before the tag a registry agent's final
snapshot or a run directory's last checkpoint (``agents.load_agent``), as a trained
``network.NetPolicy``: its fields need ``network.unlock_trained``, which only
scripts/fields.py's post-tag path calls.
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
    'impulse' and 'explicit' [4 outcomes, 3 features, N]. A key that fails to re-polish
    (``gates.RepolishFailed``) gives a ``failed`` row and no arrays; any other exception is a
    crash and propagates (the stage fails)."""
    if bank is None:
        bank, _ = K.design()
    meta = _meta(ref)
    try:
        prob, stim, rps = G.problem(ref, bank)
    except G.RepolishFailed as e:     # decided before the tag: a 'numerical' exclusion
        return {**meta, "failed": True, "reason": NUMERICAL,
                "error": f"{type(e).__name__}: {e}"}, {}
    fl = KK.key_fields(prob, bank=bank)
    lm, sig = stim.landmarks, int(stim.direction)
    g_star = KK.edge_gain(fl, lm, sig)
    g_out_star = 1.0 - KK.edge_gain(fl, lm, sig, "outside")     # dc_min / dDelta_out
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
           "G_star": g_star, "G_out_star": g_out_star, "T_star": g_star + g_out_star,
           "in_band": KK.gate4(g_star), "e_star": e_star,
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


def _run_spec(name: str) -> registry.AgentSpec:
    """A run directory ('runs/<dir>') as an AgentSpec of its last checkpoint."""
    return registry.AgentSpec(Path(name).name, name, "generalist", False, 0.0, "ckpt",
                              "eval_rdev.json", None)


def split_snapshot(name: str) -> tuple[str, int] | None:
    """(agent, timesteps) of an earlier snapshot's network name '<agent>@<timesteps>' (a
    registry agent: 'G1@109856054'); None for any other name."""
    a, sep, ts = name.partition("@")
    return (a, int(ts)) if sep and a in agents.AGENTS and ts.isdigit() else None


def step1_snapshot(agent: str, timesteps: int) -> dict | None:
    """Step 1's manifest entry of one snapshot of a registry agent (path, sha256, ...)."""
    man = json.loads(io.MANIFEST_PATH.read_text()) if io.MANIFEST_PATH.exists() else {}
    return next((s for s in man.get("snapshots", []) if s.get("agent") == agent
                 and int(s.get("timesteps", -1)) == int(timesteps)), None)


def network_agent_id(name: str) -> str:
    """The agent id a network's rows carry before '@': the registry id (an earlier
    snapshot's agent), or the run directory's name."""
    snap = split_snapshot(name)
    if snap is not None:
        return snap[0]
    return name if name in agents.AGENTS else Path(name).name


def checkpoint_path(name: str) -> Path | None:
    """The checkpoint a network name resolves to on disk: a registry agent's final snapshot,
    an earlier snapshot's file as step 1's manifest records it ('<agent>@<timesteps>'), or a
    run directory's last checkpoint (a fresh glob: before the tag only; after it
    ``load_network`` loads the recorded path)."""
    if name in agents.AGENTS:
        return agents.final_snapshot(agents.AGENTS[name])
    snap = split_snapshot(name)
    if snap is not None:
        ent = step1_snapshot(*snap)
        return None if ent is None else Path(io.REPO) / ent["path"]
    return agents.final_snapshot(_run_spec(name))


def network_pin(name: str, repo: Path) -> dict:
    """{path (repo-relative), sha256, timesteps} of a network's final checkpoint, for the
    pre-registration record (design Section 15). Reads the snapshot's pickle only."""
    from apexzero.fields import prereg

    path = checkpoint_path(name)
    if path is None:
        raise FileNotFoundError(f"{name}: no finished run with a checkpoint")
    ag = agents.load_agent(path, network_agent_id(name))
    return {"path": str(Path(path).resolve().relative_to(Path(repo).resolve())),
            "sha256": prereg.sha256_file(path), "timesteps": int(ag.timesteps)}


def registry_of_run(name: str) -> str | None:
    """The registry agent whose run directory a run path names (runs/dr-pilot-200M -> G1),
    so an H4 baseline shares its step-1 rollouts (Gate 3 and branch consistency); None for
    a registry id or a run of no registry agent."""
    if name in agents.AGENTS:
        return None
    return next((a for a, sp in agents.AGENTS.items() if sp.run == name), None)


def load_network(name: str, pin: dict | None = None, repo: Path | None = None) -> N.NetPolicy:
    """A network as a trained NetPolicy (marked ``trained``). With ``pin`` (the record's
    {path, sha256, timesteps}; post-tag stages always pass it) the recorded path is loaded,
    its sha256 checked right before the load and its timesteps after it; without one, a
    registry agent's final snapshot or a run directory's last checkpoint ('runs/<dir>', the
    H4 and H4-transfer networks)."""
    from apexzero.fields import prereg

    if pin is None:
        path = checkpoint_path(name)
        if path is None:
            raise FileNotFoundError(f"{name}: no finished run with a checkpoint")
    else:
        path = Path(repo if repo is not None else io.REPO) / pin["path"]
    sha = prereg.sha256_file(path)
    if pin is not None and sha != pin["sha256"]:
        raise ValueError(f"{name}: {pin['path']} is not the recorded checkpoint (sha256 "
                         f"{sha[:12]}, recorded {pin['sha256'][:12]})")
    ag = agents.load_agent(path, network_agent_id(name))
    if pin is not None and int(ag.timesteps) != int(pin["timesteps"]):
        raise ValueError(f"{name}: timesteps {ag.timesteps}, recorded {pin['timesteps']}")
    return N.agent_net(ag, checkpoint=sha)


def network_start(ref: K.KeyRef, rec, bank, cells, det: dict | None) -> N.Start:
    """The network's restart at a key: A2 keys from the cell's x0, anchored keys under
    option A from the det row of the key's snapshot (``det`` required)."""
    if ref.kind != "anchored":
        return N.a2_start(rec, bank, cells)
    if det is None:
        raise ValueError(f"anchored key {ref.key_id} needs its snapshot's det group")
    return N.anchored_start(rec, det, bank, "A")


def branch_record(b: N.Branch) -> dict:
    """A row's branch-consistency record: the aggregate checks, the verdict per outcome
    (``network.Branch.ok_by``) and the tolerance used."""
    return {"ok": b.ok, "ok_by": b.ok_by(), "end_ok": b.end_ok,
            "saturation_ok": b.saturation_ok, "events_ok": b.events_ok,
            "values_ok": b.values_ok, "tol": dict(b.tol),
            "values": {o: list(v) for o, v in b.values.items()}}


def support_ends(base: N.Run) -> dict[str, float]:
    """Per outcome, the last arc length the loop can read before the outcome is decided
    (``measures.support_end`` up to its read state: c_min and v_apex at the pick's later
    state, s_b at the onset, steer at the decision step); NaN for an invalid outcome. The far
    end of the s_1-anchored d grid (design Sections 4 and 6; ``field_comparisons``)."""
    out = {}
    for o in O.OUTCOMES:
        p = base.plans[o]
        if not p.valid or p.pick is None:
            out[o] = float("nan")
            continue
        k = int(p.pick.k) if o == "s_b" else int(p.pick.i1)
        out[o] = MS.support_end(base.x, min(k, len(base.x) - 1))
    out["steer"] = MS.support_end(base.x, int(base.start.decision_step))
    return out


def network_key(net: N.NetPolicy, ref: K.KeyRef, bank, cells, det: dict | None = None,
                steps: int = N.STEPS, blind: bool = True, paths: bool = True,
                branch_tol: dict | None = None, checks: bool = True
                ) -> tuple[dict, dict[str, np.ndarray]]:
    """(row, arrays) of one network's fields at one key (module docstring). arrays:
    'impulse' [4, 3, N]; with ``paths`` 'paths_kappa' [4 paths, 4 outcomes, N] (the
    curvature fields by path); with ``blind`` 'blind_impulse' [4, 3, N]. ``branch_tol``: the
    value tolerance of branch consistency: for a trained network the record's calibrated one,
    which must be passed (design Section 15: never a code constant); for the pre-tag controls
    the provisional ``network.BRANCH_TOL`` by default. ``checks`` False (another network's
    anchored state, off-policy: the anchored-key H4) takes the restart from ``det`` but runs
    no Gate 3 or branch check, and marks the row 'off_policy'. A run that exhausts the loop
    raises ``network.LoopExhausted`` (``network_fields`` records it as that key's pipeline
    failure). A trained network is refused before the tag (``network.unlock_trained``)."""
    if net.trained and branch_tol is None:
        raise ValueError(f"{net.name}: a trained network's branch tolerance is the record's "
                         "calibrated one (pass branch_tol)")
    N._refuse_trained(net, "network fields")
    rec = K.load_key(ref)
    st = network_start(ref, rec, bank, cells, det)
    rd = N.road(bank, st.stim_id)
    base = N.run(net, rd, st, steps=steps)
    if base.exhausted:
        raise N.LoopExhausted(f"{net.name} at {ref.key_id}: alive after {steps} steps")
    row: dict[str, Any] = {**_meta(ref), "net": net.name, "ckpt_sha256": net.checkpoint,
                           "loop_steps": int(steps), "n_steps": base.n_steps,
                           "term_step": base.term_step, "stop_step": base.stop_step,
                           "decision_step": st.decision_step, "s1": st.s1,
                           "support_end": support_ends(base)}
    row["gate3"], row["branch"] = None, None
    if not checks:
        row["off_policy"] = True
    if det is not None and checks:
        try:
            i = N.det_row(det, rec.cell_id, rec.phase)
        except ValueError:
            i = None
        if i is not None:
            w = N.wiring_check(net, det, i, rd.base, t0=st.offset, s_stop=st.s_stop)
            row["gate3"] = {"ok": w.ok, "mean_err": w.mean_err,
                            "step_excess": w.step_excess.tolist()}
            row["branch"] = branch_record(N.branch_consistency(
                base, det, i, N.BRANCH_TOL if branch_tol is None else branch_tol))
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
    row.update(G=em.G, G_out=em.G_out, T=em.T, B=em.B, dc_in=em.dc_in, dc_out=em.dc_out,
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


PIPELINE_LOOP = "loop exhausted"            # a failure row's 'pipeline' (design Section 15)


def arrays_dir(out_dir: Path, net_name: str) -> Path:
    """<out_dir>/<net name with '/' and '@' as '_'>: a network's per-key arrays."""
    return Path(out_dir) / net_name.replace("/", "_").replace("@", "_")


def _saved_sha(path: Path | None) -> str | None:
    from apexzero.fields import prereg

    return None if path is None else prereg.sha256_file(path)


def failure_row(net: N.NetPolicy, ref: K.KeyRef, why: str, error: str = "") -> dict:
    """The row of a key whose computation failed in the pipeline (design Section 15: the
    loop exhausted at it): that key only is excluded ('pipeline: loop exhausted'); the
    network's stage goes on."""
    return {**_meta(ref), "net": net.name, "ckpt_sha256": net.checkpoint, "pipeline": why,
            "error": error}


def network_fields(net: N.NetPolicy, refs, out_dir: Path, det_of=None, steps: int = N.STEPS,
                   progress=None, save: bool = True, checks_of=None, **kw) -> list[dict]:
    """``network_key`` over keys; ``det_of(ref)`` gives the det group of the key's snapshot
    (None: no rollouts); arrays saved under <out_dir>/<net name>/ (``arrays_dir``), each
    row carrying its file's sha256 ('arrays_sha256': the stamped rows bind the arrays the
    analysis reads, ``arrays_problems``); ``save`` False keeps no arrays (the anchored-key
    H4 reads rows only). ``checks_of(ref)`` False runs that key without Gate 3 and branch
    consistency (``network_key``'s ``checks``: another network's anchored state). A key whose
    loop is exhausted gets a ``failure_row`` and the stage goes on."""
    bank, cells = K.design()
    rows = []
    sub = arrays_dir(out_dir, net.name)
    for ref in refs:
        try:
            row, arrays = network_key(net, ref, bank, cells, det_of(ref) if det_of else None,
                                      steps, checks=checks_of(ref) if checks_of else True, **kw)
            row["arrays_sha256"] = (_saved_sha(save_key_arrays(sub, ref.key_id, arrays))
                                    if save else None)
        except N.LoopExhausted as e:
            row = failure_row(net, ref, PIPELINE_LOOP, str(e))
        rows.append(row)
        if progress:
            progress(1)
    return rows


# ------------------------------------------------------------------------- gate2-trained


GATE2_TRAINED_N = 12        # primary non-capped A2 keys (design Section 15)


def gate2_trained_sample(n: int = GATE2_TRAINED_N, rng: np.random.Generator | None = None
                         ) -> list[K.KeyRef]:
    """gate2-trained's keys (design Section 15): n primary, non-capped A2 keys drawn without
    replacement on step 2's 'gate2_trained' stream from the key-id-sorted list, in key-id
    order; the prereg stage writes them into the record, and the stage reads them there."""
    pool = sorted((r for r in K.a2_primary_refs() if not K.strata(r).cap_binds),
                  key=lambda r: r.key_id)
    rng = MS.stream_rng("gate2_trained") if rng is None else rng
    pick = rng.choice(len(pool), size=int(n), replace=False)
    return [pool[i] for i in sorted(pick)]


def gate2_trained_directions(rd: N.Road, st: N.Start) -> dict:
    """gate2-trained's directions (design Section 15): both trapezoids and the corner bumps of
    kappa and of the inside width (``perturb.gate_directions``' names)."""
    w_in = "w_left" if int(st.sigma) > 0 else "w_right"
    return {k: v for k, v in N.gate2_directions(rd, st).items()
            if k.startswith(("trapezoid_", "kappa@corner", f"{w_in}@corner"))}


def gate2_trained_network(net: N.NetPolicy, refs, bank=None, cells=None, steps: int = N.STEPS
                          ) -> dict:
    """Gate 2's rule (``gates.gate2_row``, ``network.Gate2.ok``) for one network on
    gate2-trained's keys and directions, from the A2 starts. Gate 2's rule is every row
    passing: a key whose loop is exhausted cannot be checked, so its row fails ('exhausted')
    without aborting the stage. The network passes only when every key was checked and
    passed; a network that fails is descriptive for H2', H4 and H4-transfer. The share
    counted is averaged over the checked keys."""
    if bank is None or cells is None:
        bank, cells = K.design()
    rows = []
    for ref in refs:
        rec = K.load_key(ref)
        st = N.a2_start(rec, bank, cells)
        rd = N.road(bank, st.stim_id)
        try:
            row = G.gate2_row(net, rd, st, steps, gate2_trained_directions(rd, st),
                              label=f"{net.name}@{ref.key_id}")
        except N.LoopExhausted as e:
            row = {"label": f"{net.name}@{ref.key_id}", "pass": False, "exhausted": True,
                   "error": str(e)}
        rows.append({"key_id": ref.key_id, **row})
    checked = [r for r in rows if not r.get("exhausted")]
    share = np.array([r["share_counted"] for r in checked], float).reshape(-1, len(NET_OUTCOMES))
    return {"net": net.name, "ckpt_sha256": net.checkpoint, "rows": rows,
            "n_keys": len(rows), "n_checked": len(checked),
            "n_exhausted": len(rows) - len(checked),
            "share_counted_mean": [float(np.nanmean(c)) if np.isfinite(c).any() else float("nan")
                                   for c in share.T] if len(checked) else [],
            "pass": bool(rows and all(r["pass"] is True for r in rows))}


def harness_starts(refs) -> list[tuple[int, int]]:
    """The distinct (cell, phase) starts of keys, sorted: the float32 harness runs of a network
    without step 1 rollouts (post-tag 'harness-f32'; one run per start, as step 1's det
    group)."""
    return sorted({(int(r.cell_id), int(r.phase)) for r in refs})


# ------------------------------------------------------------------------- readers (Qp)


def reader_fields(refs, out_dir: Path, variant: str = "Qp", bank=None, cells=None,
                  progress=None) -> list[dict]:
    """An unlawful-reader control's fields on keys (pre-tag: it is untrained; design Section
    7, H1-typical's reference 'against Qp'): ``controls.reader_net`` per stimulus through
    ``network_key`` (no blind-pedal control, no path split), arrays under
    <out_dir>/reader-<variant>/; an exhausted loop is a ``failure_row``."""
    from apexzero.fields import controls as C

    if bank is None or cells is None:
        bank, cells = K.design()
    rows = []
    sub = Path(out_dir) / f"reader-{variant}"
    for ref in refs:
        net = C.reader_net(variant, ref.stim_id)
        try:
            row, arrays = network_key(net, ref, bank, cells, None, blind=False, paths=False)
            row["arrays_sha256"] = _saved_sha(save_key_arrays(sub, ref.key_id, arrays))
        except N.LoopExhausted as e:
            row = failure_row(net, ref, PIPELINE_LOOP, str(e))
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
    B / (1 - G) from the base fields. Descriptive (design Section 4): per amplitude a and
    outcome (c_min, v_apex) the linearity ratio (y(a) - y(0)) / (a dy/dDelta), 1 for a
    linear response, and per |a| with both signs on the grid the narrowing/widening
    asymmetry (y(-|a|) - y(0)) / -(y(|a|) - y(0)), 1 for a linear response (NaN where a value
    or the derivative is undefined or zero)."""
    base = N.run(net, rd, st, steps=steps) if base is None else base
    fl = N.fields(net, rd, base, steps) if fl is None else fl
    direction = PT.edge_shift(rd.n, st.landmarks, st.sigma, side, 1.0)
    y0 = np.asarray(base.values(), float)
    slope = {o: MS.along(fl.of(o), direction) for o in DOSE_OUTCOMES}
    rows = []
    for a in amplitudes:
        r = N.run(net, rd, st, KK.scaled(direction, float(a)), steps, mode="secant")
        if r.exhausted:
            raise N.LoopExhausted(f"dose {a}: alive after {steps} steps")
        y = np.asarray(r.values(), float)
        rows.append({"amplitude": float(a), "values": y.tolist(),
                     "valid": r.valid_outcomes().tolist(),
                     "crashed": bool(r.term_step >= 0 and not r.plans["c_min"].valid),
                     "reason": r.plans["c_min"].reason,
                     "linearity": {o: _ratio(y[j] - y0[j], float(a) * slope[o])
                                   for j, o in enumerate(O.OUTCOMES) if o in DOSE_OUTCOMES}})
    by = {r["amplitude"]: np.asarray(r["values"], float) for r in rows}
    asym = {f"{a:g}": {o: _ratio(by[-a][j] - y0[j], -(by[a][j] - y0[j]))
                       for j, o in enumerate(O.OUTCOMES) if o in DOSE_OUTCOMES}
            for a in sorted(x for x in by if x > 0 and -x in by)}
    em = MS.edge_measures(fl.of("c_min"), fl.of("v_apex"), float(fl.values[0]), st.landmarks,
                          st.sigma)
    narrow = sorted(-r["amplitude"] for r in rows if r["amplitude"] < 0 and r["crashed"])
    pred = em.B / (1.0 - em.G) if np.isfinite(em.G) and em.G < 1.0 else float("inf")
    return {"side": side, "rows": rows, "G": em.G, "B": em.B, "crash_narrowing_predicted": pred,
            "crash_narrowing": narrow[0] if narrow else float("inf"),
            "base_values": y0.tolist(), "slope": slope, "asymmetry": asym}


DOSE_OUTCOMES = ("c_min", "v_apex")         # the dose-response's descriptive ratios


def _ratio(num: float, den: float) -> float:
    """num / den, NaN when either is not finite or den is 0."""
    if not (np.isfinite(num) and np.isfinite(den)) or den == 0.0:
        return float("nan")
    return float(num / den)


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


def branch_failed(r: dict, outcome: str) -> bool:
    """A row's branch-consistency verdict for one outcome (``branch_record``'s ok_by; a row
    without a det row is not checked)."""
    b = r.get("branch")
    if not b:
        return False
    by = b.get("ok_by")
    return (not b["ok"]) if by is None else not bool(by.get(outcome, True))


def key_flags(r: dict, opt: dict | None, outcome: str, gate4: bool = False,
              ref: dict | None = None, use_ref: bool = False) -> dict:
    """The exclusion flags of one network row for one outcome (``analysis.key_measure``):
    the pipeline failures (the row's loop exhausted, no optimum row, branch consistency), the
    network's validity (and the optimum's for field comparisons, ``gate4`` False; with
    ``use_ref`` a reference control's row ``ref`` too), Gate 3, the optimum's numerical,
    inertia and (``gate4``: the edge measures) Gate 4 failures, kappa clipping over the
    outcome's reads and, for the edge measures, the argmin off plateau."""
    if r.get("pipeline"):
        return {"loop_exhausted": True}
    o = opt or {}
    numerical = bool(opt is not None and (o.get("failed") or o.get("adjoint_ok") is False))
    net_ok = bool(r["valid"][outcome])
    other_ok = True
    if not gate4 and opt is not None and not numerical:
        other_ok = bool(o["valid"][outcome])
    if use_ref:
        other_ok = other_ok and bool(ref is not None and not ref.get("pipeline")
                                     and ref["valid"][outcome])
    out = {"no_optimum": opt is None, "branch_failed": branch_failed(r, outcome),
           "invalid": not (net_ok and other_ok),
           "why_invalid": r["reasons"].get(outcome, "") if not net_ok else "",
           "gate3_failed": bool(r.get("gate3") and not r["gate3"]["ok"]),
           "numerical": numerical,
           "inertia_failed": bool(opt is not None and not numerical and o.get("inertia_failed")),
           "gate4_failed": bool(gate4 and opt is not None and not numerical
                                and not o.get("in_band")),
           "kappa_clipped": bool(r["kappa_clip"][outcome])}
    if gate4:
        out["off_plateau"] = bool(net_ok and not r["on_plateau"])
    return out


def _num(v) -> float:
    """A row value as a float (JSON writes NaN as null)."""
    return float("nan") if v is None else float(v)


def _value(r: dict, measure: str) -> float:
    if r.get("pipeline"):
        return float("nan")
    if measure == "T":
        return _num(r["G"]) + _num(r["G_out"])
    return _num(r[measure])


def edge_key_measures(net_rows: list[dict], opt_rows: dict[str, dict], measure: str = "G",
                      require_non_capped: bool = True, capped_only: bool = False
                      ) -> list[MS.KeyMeasure]:
    """One network's per-key edge measure (G, G_out, T = G + G_out, B, dv_in) as KeyMeasures
    with their exclusion reasons (``key_flags`` with Gate 4 and the plateau rule; reasons in
    ``measures.EXCLUSION_REASONS``' order): the pipeline failures (loop exhausted at the key,
    no optimum row, branch consistency of c_min), c_min invalid, Gate 3, the optimum's
    numerical failure, inertia or Gate 4 (the lawful band's reference), kappa clipping over
    c_min's reads, argmin off plateau. Horizon twins are left out; capped stimuli too unless
    ``require_non_capped`` is False (``capped_only``: the capped stratum alone,
    descriptive)."""
    out = []
    for r in net_rows:
        if not r.get("w_is_cell", True):
            continue
        capped = bool(r.get("cap_binds"))
        if (capped_only and not capped) or (require_non_capped and capped and not capped_only):
            continue
        flags = key_flags(r, opt_rows.get(r["key_id"]), "c_min", gate4=True)
        out.append(A.key_measure(_meta_of(r), _value(r, measure), flags))
    return out


def median_statistic(rows: list[MS.KeyMeasure], n: int = MS.BOOTSTRAP_N) -> float:
    """A network's median over geometries (H4's statistic), from its key measures."""
    return float(MS.agent_summary(rows, (1, -1), n).median)


def _cov_entry(cov: dict, name: str, outcome: str) -> dict | None:
    return cov.get(f"{name}@final", {}).get(outcome)


def analyse_edges(net_rows: dict[str, list[dict]], opt_rows: dict[str, dict],
                  n: int = MS.BOOTSTRAP_N, gate2_ok: dict[str, bool] | None = None,
                  regime_cells: dict[str, dict] | None = None, decide: bool = True) -> dict:
    """H2', H4 and H4-transfer from the networks' A2 rows (``net_rows``: network name ->
    rows; ``opt_rows``: key id -> optimum row). Always: the coverage table of G per network
    (every reason, the pipeline failures listed) and which listed networks are missing. With
    ``decide`` (never before the confirmatory run, design Section 15): H2' per step-1
    generalist (``analysis.h2_network``: bands on the median G's CI per direction and pooled,
    'descriptive (coverage)' above 30% excluded, 'descriptive (gate2-trained)' when
    ``gate2_ok`` is False for it), R18's variant rule over the non-descriptive seeds, H4 and
    H4-transfer on the networks' median G ('not testable (...)' when any of their networks is
    descriptive), and the descriptive measures: G, T = G + G_out with its label, G_out and B
    for every network, each with its left-right asymmetry index (``analysis.summarise``;
    'asymmetry_G' is G's), the capped stratum, the optimum's G* and T*, and each H4 and
    H4-transfer network's median G per regime of its variant's seed-0 typing
    (``regime_cells``: agent -> regime -> A2 cell ids, from the record)."""
    km = {name: edge_key_measures(rows, opt_rows) for name, rows in net_rows.items()}
    cov = MS.coverage(rec for name, rows in km.items()
                      for rec in A.coverage_records(rows, name, "final", "G"))
    have = set(km)
    out: dict[str, Any] = {
        "coverage": cov, "decided": bool(decide),
        "networks": {name: {"n_rows": len(net_rows[name]), "n_keys": len(km[name]),
                            "pipeline_keys": (_cov_entry(cov, name, "G") or {}).get(
                                "pipeline_keys", []),
                            "near_ties": near_tie_counts(net_rows[name])} for name in km},
        "missing": {"H2": [a for a in A.H2_NETWORKS if a not in have],
                    "H4": sorted(set(A.h4_networks()) - have),
                    "H4_transfer": sorted(set(A.transfer_networks()) - have)}}
    if not decide:
        return out
    g2 = gate2_ok or {}
    out["H2"] = {name: A.h2_network(A.summarise(km[name], MS.h2_decision, n),
                                    _cov_entry(cov, name, "G"), g2.get(name))
                 for name in A.H2_NETWORKS if name in km}
    out["H2_variants"] = {v: A.h2_variant(out["H2"], v) for v in registry.variants()}
    stats = {name: median_statistic(rows, n) for name, rows in km.items()}
    desc = {}
    for name in km:
        c = _cov_entry(cov, name, "G")
        if g2.get(name) is False:
            desc[name] = "gate2-trained"
        elif c is None or c["descriptive"]:
            desc[name] = "coverage"
    for key, need, fn in (("H4", A.h4_networks(), A.h4_test),
                          ("H4_transfer", A.transfer_networks(), A.h4_transfer)):
        miss = sorted(set(need) - have)
        out[key] = fn(stats, descriptive=desc) if not miss else {"decision": "not computed",
                                                                 "missing": miss}
    out["statistics"] = stats
    out["descriptive"] = desc
    edge = {}
    for name, rows in net_rows.items():
        t = A.summarise(edge_key_measures(rows, opt_rows, "T"), None, n)
        t["label"] = A.t_label(t["pooled"]["lo"], t["pooled"]["hi"])
        g = out["H2"][name] if name in out["H2"] else A.summarise(km[name], None, n)
        edge[name] = {"G": {k: v for k, v in g.items() if k in ("left", "right", "pooled",
                                                                 "asymmetry")},
                      "T": t, "G_out": A.summarise(edge_key_measures(rows, opt_rows, "G_out"),
                                                    None, n),
                      "B": A.summarise(edge_key_measures(rows, opt_rows, "B"), None, n)}
        edge[name]["asymmetry_G"] = g["asymmetry"]
        capped = edge_key_measures(rows, opt_rows, capped_only=True)
        if capped:
            edge[name]["capped_G"] = A.summarise(capped, None, n)
    out["edge_descriptive"] = edge
    out["optimum"] = optimum_edges(opt_rows, n)
    if regime_cells:
        out["H4_regimes"] = h4_regime_breakdown(net_rows, opt_rows, regime_cells, n)
    return out


def near_tie_counts(rows: list[dict]) -> dict[str, int]:
    """Per outcome, the keys whose base run has a near-tie (design Section 3: flagged keys are
    reported and kept) among a network's rows that have a run."""
    return {o: int(sum(bool((r.get("near_tie") or {}).get(o)) for r in rows
                       if not r.get("pipeline")))
            for o in NET_OUTCOMES}


def anchored_edges(net_rows: list[dict], opt_rows: dict[str, dict], n: int = MS.BOOTSTRAP_N
                   ) -> dict:
    """The anchored-key version of H2' for one agent (design Section 7, descriptive): G, T and
    B on its non-capped anchored finals, with their exclusions and coverage (the edge
    measures' rules, ``edge_key_measures``)."""
    out = {}
    for m in ("G", "T", "B"):
        km = edge_key_measures(net_rows, opt_rows, m)
        out[m] = A.summarise(km, None, n)
        if m == "G":
            out["coverage"] = _coverage_of(km, "net", "G")
    out["T"]["label"] = A.t_label(out["T"]["pooled"]["lo"], out["T"]["pooled"]["hi"])
    return out


def anchored_h4(net_rows: dict[str, list[dict]], opt_rows: dict[str, dict],
                n: int = MS.BOOTSTRAP_N) -> dict:
    """The anchored-key version of H4 (design Section 7, descriptive): each H4 network's G on
    the pooled anchored finals (``edge_key_measures``' rules; off-policy for every network
    but a key's owner, whose rows carry no Gate 3 or branch check), its median statistic and
    coverage, and H4's test on those statistics (``analysis.h4_test``; 'not testable (...)'
    as H4), reported, never decided."""
    km = {name: edge_key_measures(rows, opt_rows) for name, rows in net_rows.items()}
    cov = MS.coverage(rec for name, rows in km.items()
                      for rec in A.coverage_records(rows, name, "final", "G"))
    stats = {name: median_statistic(rows, n) for name, rows in km.items()}
    desc = {name: "coverage" for name in km
            if (_cov_entry(cov, name, "G") or {"descriptive": True})["descriptive"]}
    miss = sorted(set(A.h4_networks()) - set(km))
    test = (A.h4_test(stats, descriptive=desc) if not miss
            else {"decision": "not computed", "missing": miss})
    return {"descriptive": True, "statistics": stats, "coverage": cov, "test": test,
            "n_off_policy": {name: int(sum(bool(r.get("off_policy")) for r in rows))
                             for name, rows in net_rows.items()}}


def optimum_edges(opt_rows: dict[str, dict], n: int = MS.BOOTSTRAP_N) -> dict:
    """The optimum's G* and T* (= G* + G_out*) over the primary, non-capped A2 keys
    (descriptive: T* ~ 1)."""
    rows = [r for r in opt_rows.values() if r.get("kind") != "anchored" and r.get("w_is_cell")
            and not r.get("cap_binds")]
    out = {}
    for name in ("G_star", "T_star"):
        km = [A.key_measure(_meta_of(r), _num(r.get(name)) if not r.get("failed")
                            else float("nan"), {"numerical": bool(r.get("failed"))})
              for r in rows if name in r or r.get("failed")]
        if km:
            out[name] = A.summarise(km, None, n)
    return out


def h4_regime_breakdown(net_rows: dict[str, list[dict]], opt_rows: dict[str, dict],
                        regime_cells: dict[str, dict], n: int = MS.BOOTSTRAP_N) -> dict:
    """Each H4 and H4-transfer network's median G per regime of its variant's seed-0 typing
    (``analysis.h4_regime_agent``; descriptive, design Section 7)."""
    out = {}
    for name, rows in net_rows.items():
        ag = A.h4_regime_agent(name)
        if ag is None or ag not in regime_cells:
            continue
        per = {}
        for reg, cids in regime_cells[ag].items():
            cs = {int(c) for c in cids}
            km = edge_key_measures([r for r in rows if int(r["cell"]) in cs], opt_rows)
            per[reg] = {"n_keys": len(km), "median": median_statistic(km, n) if km
                        else float("nan")}
        out[name] = {"typed_by": ag, "regimes": per}
    return out


def _lm(row: dict):
    """The landmarks the trapezoid needs (s_c0, s_c1) from a row."""
    from apexzero.interp.stimuli import Landmarks

    a, b = float(row["s_c0"]), float(row["s_c1"])
    return Landmarks(a, a, 0.5 * (a + b), b, b)


def _meta_of(r: dict) -> dict:
    return {k: r[k] for k in ("key_id", "pair_id", "approach", "direction", "phase", "n_phases",
                              "R", "theta_deg")}


_NAN_CMP = {"g": float("nan"), "rho": float("nan"), "location": float("nan")}


GRIDS = ("aligned", "s1")


def s1_grid(r: dict, o: dict, outcome: str) -> tuple[float, np.ndarray]:
    """(s_1, d): the design's s_1-anchored grid of one key (Sections 4 and 6): bumps at s_1 + d,
    d from -20 m to the network's support end for the outcome (the row's 'support_end'; for a
    row without one, s_c0 + 400 m, the aligned grid's far end)."""
    s1 = float(r["s1"]) if r.get("s1") is not None else float(o["s1"])
    end = (r.get("support_end") or {}).get(outcome)
    end = float(end) if end is not None and np.isfinite(float(end)) \
        else float(o["s_c0"]) + MS.NULL_D[1]
    return s1, MS.d_grid_to(end, s1)


def _s1_field(imp: Samples, s1: float, d: np.ndarray, sigma: int, outcome: str,
              sigma_b: float) -> np.ndarray:
    return MS.bump_field(MS.turn_fields(imp, sigma, outcome)["kappa_turn"], s1, d, sigma_b)


def field_comparisons(net_rows: list[dict], opt_rows: dict[str, dict], net_dir: Path,
                      opt_dir: Path, outcome: str, impulse_key: str = "impulse",
                      sigma_b: float = PT.SIGMA_B[0], ref: tuple | None = None,
                      grid: str = "aligned") -> list[dict]:
    """Per key: the network's (``impulse_key``: 'impulse', or 'blind_impulse' for the
    blind-pedal control) and the reference's curvature field of one outcome, recoded toward
    the turn, compared on a grid: ``grid`` 'aligned' (H1-typical, design Section 15) the
    common s_c0-aligned grid (``measures.aligned_field``, d in [-150, 400] m); 's1' (H1-fast's
    gain and steer_0, Sections 4 and 6) bumps at s_1 + d, d from -20 m to the network's
    support end (``s1_grid``), so the near field ahead of the car is kept wherever the corner
    lies. Gain, rho, location, and the shape specificity delta_rho against the reference's
    fields at far geometries among the same keys (``measures.shape_specificity``, always on
    the common s_c0-aligned grid, Section 8), with the exclusion flags (``key_flags``). The
    reference is the optimum (``opt_dir``), or with ``ref`` = (rows by key id, arrays dir) a
    control's network-side fields (Qp); s_c0 and the optimum's flags come from ``opt_rows``
    either way. A key with a pipeline failure, no optimum row or a failed optimum is kept as
    an excluded entry (NaN), never dropped."""
    if grid not in GRIDS:
        raise ValueError(f"grid {grid!r} is not one of {GRIDS}")
    j = NET_OUTCOMES.index(outcome)
    jo = OPT_OUTCOMES.index(outcome)
    d = PT.d_grid(MS.NULL_D[1], MS.NULL_D[0])
    net_f, ref_f, rows = {}, {}, {}
    for r in net_rows:
        o = opt_rows.get(r["key_id"])
        rr = ref[0].get(r["key_id"]) if ref is not None else None
        fl = key_flags(r, o, outcome, ref=rr, use_ref=ref is not None)
        entry = {"row": r, "opt": o, "flags": fl}
        bad = (r.get("pipeline") or o is None or fl["numerical"]
               or (ref is not None and (rr is None or rr.get("pipeline"))))
        if bad:
            rows[r["key_id"]] = {**entry, **_NAN_CMP}
            continue
        geo = (r["R"], r["theta_deg"])
        a = load_key_arrays(net_dir, r["key_id"])
        if ref is None:
            sr = samples_of(load_key_arrays(opt_dir, r["key_id"])["impulse"], jo)
        else:
            sr = samples_of(load_key_arrays(ref[1], r["key_id"])["impulse"], j)
        sn = samples_of(a[impulse_key], j)
        net_f[r["key_id"]] = (geo, o["s_c0"], r["direction"], sn)
        ref_f[r["key_id"]] = (geo, o["s_c0"], r["direction"], sr)
        if grid == "aligned":
            Fn = MS.aligned_field(sn, o["s_c0"], r["direction"], outcome, "kappa_turn", sigma_b)
            Fr = MS.aligned_field(sr, o["s_c0"], r["direction"], outcome, "kappa_turn", sigma_b)
            rows[r["key_id"]] = {**entry, **MS.compare(Fn, Fr, d)}
        else:
            s1, dk = s1_grid(r, o, outcome)
            Fn = _s1_field(sn, s1, dk, int(r["direction"]), outcome, sigma_b)
            Fr = _s1_field(sr, s1, dk, int(r["direction"]), outcome, sigma_b)
            rows[r["key_id"]] = {**entry, **MS.compare(Fn, Fr, dk)}
    for sp in MS.shape_specificity(net_f, ref_f, outcome, sigma_b):
        rows[sp["label"]]["delta_rho"] = sp["delta_rho"]
        rows[sp["label"]]["rho_far"] = sp["rho_far"]
    return list(rows.values())


def _non_capped(rows: list[dict]) -> list[dict]:
    """Primary (no horizon twin), non-capped rows (design Section 15: the capped stimuli are
    a descriptive stratum of every hypothesis)."""
    return [r for r in rows if r.get("w_is_cell", True) and not r.get("cap_binds")]


def _capped(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("w_is_cell", True) and r.get("cap_binds")]


def h1_typical_rows(cmp: list[dict], ctrl: list[dict], outcome: str, cells=None
                    ) -> tuple[dict, dict]:
    """H1-typical's per-key measures (gain, location, delta_rho) for a network and its
    blind-pedal control, restricted to ``cells`` (the agent's bulk A2 cells; None: all) and
    to non-capped stimuli."""
    def rows_of(items):
        out = {"gain": [], "location": [], "shape": []}
        for c in items:
            r = c["row"]
            if r.get("cap_binds") or (cells is not None and r["cell"] not in cells):
                continue
            for name, v in (("gain", c["g"]), ("location", c["location"]),
                            ("shape", c.get("delta_rho", float("nan")))):
                out[name].append(A.key_measure(_meta_of(r), v, c["flags"]))
        return out

    return rows_of(cmp), rows_of(ctrl)


def _coverage_of(rows: list[MS.KeyMeasure], who: str, outcome: str) -> dict | None:
    return MS.coverage(A.coverage_records(rows, who, "final", outcome)).get(
        f"{who}@final", {}).get(outcome)


def analyse_fields(net_rows: list[dict], opt_rows: dict[str, dict], net_dir: Path,
                   opt_dir: Path, cells=None, n: int = MS.BOOTSTRAP_N,
                   qp: tuple | None = None, decide: bool = True) -> dict:
    """One network's H1-typical (descriptive: per outcome, s_b then v_apex, against the
    optimum and, given ``qp`` = (Qp's rows by key id, its arrays dir), against Qp, each with
    the paired blind-pedal control) and H2-speed (r_in = dv_apex,net / dv_apex,opt along the
    inside shift; 'descriptive (coverage)' above 30% excluded) on its primary, non-capped A2
    rows (``cells``: its bulk A2 cells for H1-typical), with the coverage per outcome, the
    capped stratum's r_in and the other bump widths and steer_0 (``fields_descriptive``; all
    descriptive). Without ``decide``: the coverage only."""
    rows = _non_capped(net_rows)
    out: dict[str, Any] = {"H1_typical": {}, "coverage": {}}
    cmps = {}
    for o in ("s_b", "v_apex"):
        cmp = field_comparisons(rows, opt_rows, net_dir, opt_dir, o)
        ctrl = field_comparisons(rows, opt_rows, net_dir, opt_dir, o, "blind_impulse")
        a, b = h1_typical_rows(cmp, ctrl, o, cells)
        cmps[o] = (a, b)
        out["coverage"][f"H1_typical/{o}"] = _coverage_of(a["gain"], "net", o)
    r_in = h2_speed_rows(rows, opt_rows, opt_dir)
    cov = _coverage_of(r_in, "net", "v_apex")
    out["coverage"]["H2_speed/v_apex"] = cov
    if not decide:
        return {"coverage": out["coverage"]}
    for o, (a, b) in cmps.items():
        out["H1_typical"][o] = A.h1_typical(a["gain"], a["location"], a["shape"], b, n)
    if qp is not None:
        out["H1_typical_vs_Qp"], out["Qp_vs_optimum"] = {}, {}
        for o in ("s_b", "v_apex"):
            cmp = field_comparisons(rows, opt_rows, net_dir, opt_dir, o, ref=qp)
            ctrl = field_comparisons(rows, opt_rows, net_dir, opt_dir, o, "blind_impulse", ref=qp)
            a, b = h1_typical_rows(cmp, ctrl, o, cells)
            out["H1_typical_vs_Qp"][o] = A.h1_typical(a["gain"], a["location"], a["shape"], b, n)
            mine = {r["key_id"] for r in rows}
            q_rows = [r for k, r in qp[0].items() if k in mine]
            qc = field_comparisons(q_rows, opt_rows, qp[1], opt_dir, o)
            qa, _ = h1_typical_rows(qc, [], o, cells)
            out["Qp_vs_optimum"][o] = {k: A.summarise(v, None, n) for k, v in qa.items()}
    out["H2_speed"] = A.with_coverage(A.summarise(r_in, A.h2_speed, n), cov)
    capped = h2_speed_rows(_capped(net_rows), opt_rows, opt_dir)
    if capped:
        out["H2_speed_capped"] = A.summarise(capped, None, n)
    out["descriptive"] = fields_descriptive(rows, opt_rows, net_dir, opt_dir, cells, n)
    return out


def fields_descriptive(rows: list[dict], opt_rows: dict[str, dict], net_dir: Path,
                       opt_dir: Path, cells=None, n: int = MS.BOOTSTRAP_N) -> dict:
    """Design Section 7's descriptive field items for one network on its primary, non-capped
    A2 rows: H1-typical's gain, location and delta_rho against the optimum at the other bump
    widths (sigma_b 4 and 16 m, on its bulk cells ``cells``; reaches compare only at equal
    sigma_b), and the instantaneous steer field steer_0 (the decision step's steer mean
    against the optimum's decision steer, curvature recoded toward the turn) on every row, on
    the s_1-anchored grid (``field_comparisons`` grid 's1': the near field ahead of the car is
    kept wherever the corner lies)."""
    out = {}
    for sb in SIGMA_BS[1:]:
        for o in ("s_b", "v_apex"):
            a, _ = h1_typical_rows(field_comparisons(rows, opt_rows, net_dir, opt_dir, o,
                                                     sigma_b=sb), [], o, cells)
            out[f"sigma_b={sb:g}/{o}"] = {k: A.summarise(v, None, n) for k, v in a.items()}
    a, _ = h1_typical_rows(field_comparisons(rows, opt_rows, net_dir, opt_dir, "steer",
                                             grid="s1"), [], "steer", None)
    out["steer_0"] = {k: A.summarise(v, None, n) for k, v in a.items()}
    return out


def h2_speed_rows(rows: list[dict], opt_rows: dict[str, dict], opt_dir: Path
                  ) -> list[MS.KeyMeasure]:
    """H2-speed's per-key r_in with its exclusion flags (``key_flags`` on v_apex; an optimum
    response below the floor is an invalid outcome)."""
    out = []
    for r in rows:
        o = opt_rows.get(r["key_id"])
        fl = key_flags(r, o, "v_apex")
        v = float("nan")
        if not (r.get("pipeline") or o is None or fl["numerical"]):
            imp = samples_of(load_key_arrays(opt_dir, r["key_id"])["impulse"],
                             OPT_OUTCOMES.index("v_apex"))
            v = MS.speed_ratio(_num(r["dv_in"]), MS.edge_derivative(imp, _lm(o),
                                                                    r["direction"], "inside"))
            if not np.isfinite(v) and not fl.get("invalid"):
                fl = {**fl, "invalid": True, "why_invalid": "optimum response below floor"}
        out.append(A.key_measure(_meta_of(r), v, fl))
    return out


def h1_fast_rows(net_rows: list[dict], opt_rows: dict[str, dict], net_dir: Path, opt_dir: Path,
                 agent: str, n: int = MS.BOOTSTRAP_N, decide: bool = True) -> dict:
    """H1-fast for one agent on its non-capped anchored finals (descriptive,
    ``analysis.h1_fast``): the differences s_b,net - s_b* and v_apex,net - v_apex* and the
    curvature gain of the s_b field (on the s_1-anchored grid, ``field_comparisons`` grid
    's1'), with the coverage per outcome and the capped stratum reported apart. Without
    ``decide``: the coverage only."""
    def diffs(rows):
        sb, va = [], []
        for r in rows:
            o = opt_rows.get(r["key_id"])
            for name, store in (("s_b", sb), ("v_apex", va)):
                fl = key_flags(r, o, name)
                ok = not (r.get("pipeline") or o is None or fl["numerical"])
                v = _num(r["values"][name]) - _num(o["values"][name]) if ok else float("nan")
                store.append(A.key_measure(_meta_of(r), v, fl))
        return sb, va

    def gains(rows):                   # on the s_1-anchored grid (design Sections 4 and 6)
        cmp = field_comparisons(rows, opt_rows, net_dir, opt_dir, "s_b", grid="s1")
        return [A.key_measure(_meta_of(c["row"]), c["g"], c["flags"]) for c in cmp]

    rows = [r for r in net_rows if not r.get("cap_binds")]
    sb, va = diffs(rows)
    gain = gains(rows)
    cov = {"s_b": _coverage_of(sb, agent, "s_b"), "v_apex": _coverage_of(va, agent, "v_apex"),
           "gain": _coverage_of(gain, agent, "s_b-gain")}
    if not decide:
        return {"coverage": cov}
    out = A.h1_fast(sb, va, gain, agent, n)
    out["coverage"] = cov
    capped = [r for r in net_rows if r.get("cap_binds")]
    if capped:
        csb, cva = diffs(capped)
        out["capped"] = {"s_b": A.summarise(csb, None, n), "v_apex": A.summarise(cva, None, n),
                         "gain": A.summarise(gains(capped), None, n)}
    return out


def confirmatory_done(path: Path, tag: dict | None) -> str | None:
    """The sha256 of a genuine confirmatory decisions file at ``path``, else None (design
    Section 15: before the confirmatory run a non-confirmatory analysis gives the coverage and
    plumbing only, so a stray, hand-made or touched file must never unlock the decisions). It
    must parse as an object with 'confirmatory' and 'decided' true, the decided results (H2,
    H4, H4_transfer) and inputs, and an embedded stamp of the analyse stage, not partial, with
    '--confirmatory' in its argv, written under ``tag`` (the current tag identity: object and
    commit). The code hash is not compared: a logged deviation may change the code after the
    confirmatory run."""
    from apexzero.fields import prereg

    path = Path(path)
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("confirmatory") is not True or d.get("decided") is not True:
        return None
    if not {"H2", "H4", "H4_transfer"} <= set(d) or not d.get("inputs"):
        return None
    pv = d.get("provenance")
    if not isinstance(pv, dict) or pv.get("stage") != "analyse" or pv.get("partial"):
        return None
    if "--confirmatory" not in (pv.get("argv") or []):
        return None
    t = pv.get("tag") or {}
    if tag is None or t.get("object") != tag.get("object") or t.get("commit") != tag.get("commit"):
        return None
    return prereg.sha256_file(path)


def confirmatory_guard(out_dir: Path) -> Path:
    """The analysis runs once (design Sections 9 and 15): refuses when the confirmatory
    decisions file already exists; returns its path (scripts/fields.py passes the fixed
    outputs/fields/analyse, never an --out root)."""
    path = Path(out_dir) / "decisions.json"
    if path.exists():
        raise SystemExit(f"refusing to run the confirmatory analysis again: {path} exists "
                         "(design Section 9: the analysis runs once)")
    return path


def expected_net(name: str, pin: dict) -> str:
    """The 'net' a network's rows must carry: its agent id at the recorded timesteps."""
    return f"{network_agent_id(name)}@{int(pin['timesteps'])}"


def arrays_problems(rows: list[dict], arrays: Path, name: str) -> list[str]:
    """Why a network's per-key arrays are not the ones its stamped rows bind: a row (other
    than a pipeline failure) without 'arrays_sha256', without its <arrays>/<key>.npz, or whose
    file's sha256 differs. Files of no row (an earlier run's) are ignored."""
    from apexzero.fields import prereg

    miss, bad = [], []
    for r in rows:
        if r.get("pipeline"):
            continue
        f = Path(arrays) / f"{str(r['key_id']).replace('/', '_')}.npz"
        if not r.get("arrays_sha256") or not f.exists():
            miss.append(r["key_id"])
        elif prereg.sha256_file(f) != r["arrays_sha256"]:
            bad.append(r["key_id"])
    out = []
    if miss:
        out.append(f"{name}: {len(miss)} rows without their arrays (e.g. {miss[:3]})")
    if bad:
        out.append(f"{name}: {len(bad)} arrays differ from their rows' sha256 (e.g. {bad[:3]})")
    return out


def rows_problems(rows: list[dict], want: set[str], name: str, pin: dict | None,
                  checked: bool = False) -> list[str]:
    """Why one network's rows are not the complete, recorded input of the confirmatory
    analysis: duplicate key ids, a key set other than the frozen one (``want``), a row from
    another snapshot or checkpoint than the record's (``pin``), and with ``checked`` a row
    whose wiring and branch checks did not run (no det row)."""
    out = []
    ids = [r["key_id"] for r in rows]
    if len(ids) != len(set(ids)):
        out.append(f"{name}: {len(ids) - len(set(ids))} duplicate key rows")
    if set(ids) != want:
        miss, extra = sorted(want - set(ids)), sorted(set(ids) - want)
        out.append(f"{name}: {len(miss)} frozen keys without a row (e.g. {miss[:3]}), "
                   f"{len(extra)} rows of unlisted keys (e.g. {extra[:3]})")
    if pin is None:
        out.append(f"{name}: no recorded checkpoint")
    else:
        net = expected_net(name, pin)
        bad = [r["key_id"] for r in rows if r.get("net") != net
               or r.get("ckpt_sha256") != pin["sha256"]]
        if bad:
            out.append(f"{name}: {len(bad)} rows not from the recorded checkpoint {net} "
                       f"(e.g. {bad[:3]})")
    if checked:
        bad = [r["key_id"] for r in rows if not r.get("pipeline") and not r.get("branch")]
        if bad:
            out.append(f"{name}: {len(bad)} rows without the wiring and branch checks "
                       f"(e.g. {bad[:3]})")
    return out


def confirmatory_input_problems(rec: dict, root: Path, networks=None,
                                gate2_file: str = "gate2-trained/gate2_trained.json"
                                ) -> list[str]:
    """Why the run-once confirmatory analysis may not run on what is under ``root`` (design
    Section 15): every required network (``analysis.A2_NETWORKS`` by default) must have a
    fields-a2 row for exactly the frozen A2 keys, from its recorded checkpoint, each with its
    wiring and branch checks, and the per-key arrays its rows bind (``arrays_problems``); every
    anchored finals set its fields-anchored rows and arrays; every frozen A2 and finals key an
    optimum row (once); gate2-trained a result for every required network, on exactly the
    record's gate2-trained keys, from its recorded checkpoint, with a boolean verdict; and the
    record the bulk A2 cells of every H2' network (H1-typical's data). [] when complete."""
    from apexzero.fields import prereg

    root = Path(root)
    networks = A.A2_NETWORKS if networks is None else tuple(networks)
    pins = rec.get("networks", {})
    out = []
    a2 = {k for _, k in prereg.frozen_ids(rec, "a2")}
    regimes = (rec.get("regimes") or {}).get("a2_regime_cells", {})
    no_bulk = [a for a in A.H2_NETWORKS if a in networks and not regimes.get(a, {}).get("bulk")]
    if no_bulk:
        out.append(f"the record has no bulk A2 cells (H1-typical's data) for {no_bulk}")
    for name in networks:
        f = root / "fields-a2" / f"{name.replace('/', '_')}.jsonl"
        if not f.exists():
            out.append(f"{name}: no fields-a2 rows ({f.name})")
            continue
        rows = read_rows(f)
        out += rows_problems(rows, a2, name, pins.get(name), checked=True)
        if name in pins:
            out += arrays_problems(rows, arrays_dir(root / "fields-a2" / "arrays",
                                                    expected_net(name, pins[name])), name)
    finals = [k for k in rec.get("key_sets", {}) if k.startswith("finals:")]
    for sname in finals:
        agent = sname.split(":", 1)[1]
        f = root / "fields-anchored" / f"{agent}.jsonl"
        if not f.exists():
            out.append(f"{agent}: no fields-anchored rows")
            continue
        rows = read_rows(f)
        out += rows_problems(rows, {k for _, k in prereg.frozen_ids(rec, sname)},
                             agent, pins.get(agent))
        if agent in pins:
            out += arrays_problems(rows, arrays_dir(root / "fields-anchored" / "arrays",
                                                    expected_net(agent, pins[agent])), agent)
    opt_path = root / "opt-fields" / "rows.jsonl"
    if not opt_path.exists():
        out.append("no optimum rows (opt-fields/rows.jsonl)")
    else:
        ids = [r["key_id"] for r in read_rows(opt_path)]
        want = a2 | {k for sname in finals for _, k in prereg.frozen_ids(rec, sname)}
        if len(ids) != len(set(ids)):
            out.append("opt-fields: duplicate key rows")
        miss = sorted(want - set(ids))
        if miss:
            out.append(f"{len(miss)} frozen keys without an optimum row (e.g. {miss[:3]})")
    g2 = root / gate2_file
    if not g2.exists():
        out.append(f"no gate2-trained results ({gate2_file})")
    else:
        g2d = json.loads(g2.read_text())
        have = g2d.get("networks", {})
        miss = [nm for nm in networks if nm not in have]
        if miss:
            out.append(f"gate2-trained lacks {len(miss)} networks (e.g. {miss[:3]})")
        want = sorted(k for _, k in rec.get("gate2_trained", []))
        if sorted(g2d.get("keys", [])) != want:
            out.append("gate2-trained: its keys are not the record's gate2-trained keys")
        bad = []
        for nm in networks:
            v, pin = have.get(nm), pins.get(nm)
            if v is None:
                continue
            if (pin is None or v.get("net") != expected_net(nm, pin)
                    or v.get("ckpt_sha256") != pin["sha256"] or not isinstance(v.get("pass"), bool)
                    or sorted(r.get("key_id") for r in v.get("rows", [])) != want):
                bad.append(nm)
        if bad:
            out.append(f"gate2-trained: {len(bad)} networks not from their recorded checkpoint "
                       f"on the record's keys, or without a verdict (e.g. {bad[:3]})")
    return out


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


def dry_run(cell_a2: int = 344, h3_cell: int = A.H3_CELLS[0], lams=(0.0, 0.5, 1.0),
            harness: Path | None = None) -> dict:
    """The network stage bodies end to end on policies that are not trained networks (a
    float64 process): ``network_key`` for the planted linear driver (paths and the blind-pedal
    control) and for a random-init actor (G1's architecture and obs_rms, read only) at an A2
    key, a loop exhausted at one key (a failure row, the stage going on), ``dose_network``
    for the linear driver, ``gate2_trained_network`` for the linear driver at that key,
    ``h3_lambda`` for the planted edge keeper on a lambda subgrid, ``reader_fields`` for Qp,
    and ``analyse_edges`` (with and without ``decide``) and ``analyse_fields`` (against the
    optimum and against Qp) on the rows. With ``harness`` (a float32 harness file of the
    hybrid control from that key's (cell, phase) start, made in an x64-off process as the
    post-tag 'harness-f32' stage makes them; ``network.save_det``), the wiring of a network
    without step 1 rollouts (design Section 15): ``network_key`` of the hybrid control with
    that det group (Gate 3 and branch consistency per outcome), once at the default
    tolerance and once at a zero tolerance, whose branch failures are pipeline exclusions,
    and once with ``checks`` False (another network's state, the anchored-key H4): no Gate 3
    or branch check, marked 'off_policy'."""
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
    harness_rows = {}
    if harness is not None:
        det, meta = N.load_reference(harness)
        hyb = N.hybrid_net(sigma, g1.cfg, g1.obs_rms, 0)
        harness_rows["hybrid"], _ = network_key(hyb, ref, bank, cells, det, blind=False,
                                                paths=False)
        harness_rows["hybrid_strict"], _ = network_key(
            hyb, ref, bank, cells, det, blind=False, paths=False,
            branch_tol=dict.fromkeys(N.BRANCH_TOL, 0.0))
        out["harness"] = {"meta": meta, **harness_rows}
        off, _ = network_key(hyb, ref, bank, cells, det, blind=False, paths=False,
                             branch_tol=dict.fromkeys(N.BRANCH_TOL, 0.0), checks=False)
        out["off_policy"] = {k: off.get(k) for k in ("off_policy", "gate3", "branch", "G")}
    out["arrays"] = {"linear": {k: list(v.shape) for k, v in arr_l.items()},
                     "random_impulse_nan_rows": [bool(np.isnan(arr_r["impulse"][j]).all())
                                                 for j in range(4)]}
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        short = network_fields(lin, [ref], tmp / "short", steps=20, blind=False, paths=False)
    out["short_loop"] = short
    rec = K.load_key(ref)
    st = network_start(ref, rec, bank, cells, None)
    rd = N.road(bank, st.stim_id)
    out["dose"] = dose_network(lin, rd, st)
    out["gate2_trained"] = gate2_trained_network(lin, [ref], bank, cells)
    out["gate2_trained_directions"] = sorted(gate2_trained_directions(rd, st))
    hst = N.cell_start(cells[h3_cell], 0, bank)
    hrd = N.road(bank, hst.stim_id)
    keeper = C.keeper_net(hst, hrd, True)
    out["h3"] = h3_lambda(keeper, h3_cell, 0, lams)
    out["h3_loop_audit"] = h3_loop_audit()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        orow, oarr = opt_fields_key(ref, bank)
        save_key_arrays(tmp / "opt", ref.key_id, oarr)
        save_key_arrays(tmp / "net", ref.key_id, arr_l)
        opt = {ref.key_id: orow}
        out["optimum_row"] = {k: orow[k] for k in ("G_star", "G_out_star", "T_star")}
        nets = {"linear_driver": [row_l], "random": [row_r]}
        out["plumbing"] = analyse_edges(nets, opt, n=200, decide=False)
        out["analysis"] = analyse_edges(nets, opt, n=200)
        out["analysis_no_opt"] = analyse_edges({"linear_driver": [row_l]}, {}, n=200,
                                               decide=False)["coverage"]
        if harness_rows:
            out["harness_coverage"] = analyse_edges(
                {k: [v] for k, v in harness_rows.items()}, opt, n=200, decide=False)["coverage"]
            out["harness_reasons"] = {
                name: {o: A.key_measure(_meta_of(r), 0.0,
                                        key_flags(r, orow, o, gate4=o == "c_min")).reason
                       for o in O.OUTCOMES}
                for name, r in harness_rows.items()}
        out["fields_analysis"] = analyse_fields([row_l], opt, tmp / "net", tmp / "opt", n=200)
        (qrow,) = reader_fields([ref], tmp / "qp", "Qp", bank, cells)
        out["qp_row"] = qrow
        out["fields_analysis_qp"] = analyse_fields([row_l], opt, tmp / "net", tmp / "opt",
                                                   n=200, qp=({ref.key_id: qrow},
                                                              tmp / "qp" / "reader-Qp"))
        out["comparison"] = {o: {k: v for k, v in c.items() if k not in ("row", "opt")}
                             for o in ("s_b", "v_apex")
                             for c in field_comparisons([row_l], opt, tmp / "net", tmp / "opt", o)}
    return out


def main(argv: list[str] | None = None) -> int:
    """``python -m apexzero.fields.stages dry-run --out FILE [--harness F32]``: ``dry_run`` in a
    float64 process (x64 switched on here)."""
    import argparse

    import jax

    ap = argparse.ArgumentParser(prog="python -m apexzero.fields.stages")
    ap.add_argument("what", choices=("dry-run",))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--harness", type=Path, default=None,
                    help="the hybrid control's float32 harness file at the dry run's A2 key")
    args = ap.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    res = dry_run(harness=args.harness)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(io.json_bytes(res))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
