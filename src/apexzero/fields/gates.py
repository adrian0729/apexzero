"""Gates 0-5 of step 2 (design Section 8), each returning a JSON-ready result dict.

- **Gate 0** (``gate0_key``, ``gate0``): stationarity of the stored multipliers under the
  rebuilt P on the free variables, max |r_free| <= 1e-6 (``keys.gate0``, keys.py:474), on every
  key; no solves.
- **Gate 1** (``gate1_sample``, ``gate1_key``, ``gate1_decision``): the KKT fields against
  Richardson FD of warm re-solves (``kkt.gate1``) on a key sample stratified by approach and
  agent family (``GATE1_STRATA``: A2 keys at W = 400 and 800, anchored finals of the v1, v2
  and v3 generalists and of the specialists), along ``perturb.gate_directions``: each
  feature's bumps near the car (s_1 - 10, s_1, s_1 + 10) and at the corner's landmarks (s_c0 -
  20, s_c0, s_apex, s_c1, s_c1 + 20; an anchored key's s_1 is 108-171 m before s_c0, so
  without them its in-corner fields would go unchecked), and both trapezoids. Every (key,
  quantity, feature-and-region group) field vector is one error (``kkt.gate1_score``; c_min
  is scored against its explicit edge term). Zero fields (both sides below 1e-6) cannot fail
  and are counted, never pooled. Pass: median < 1e-3 and p95 < 1e-2 on the non-zero vectors,
  pooled and within each quantity; each stratum has scored corner vectors of v_apex and s_b
  in kappa and in the inside width (``GATE1_COVERAGE``); every key re-polished, re-solved and
  passed the adjoint check. Each planted bug (``kkt.PLANTS``) is scored against the same
  re-solves and must fail the pooled rule.
- **Gate 2** (``gate2_row``, ``gate2``): AD against FD through the float64 closed loop
  (``network.gate2``) for policies that are not trained networks: the random actor (steer
  only), the planted edge keeper, the linear driver and the hybrid control (the linear driver
  plus a small-gain random-init MLP through G1's obs_rms; plan-level outcomes), along the same
  directions. After the tag the same rule runs on every H2', H4 and H4-transfer network
  ('gate2-trained', ``stages.gate2_trained_network``). The rule is ``network.Gate2.ok``: per
  valid outcome a field-vector relative error < 1e-3 over the comparisons that count (a field
  that both sides put below 1e-6 on every counted direction is a zero field and scores 0; a
  valid outcome with nothing counted, or an infinite or NaN error, fails); at least one
  outcome valid. The share counted is reported. Gate 2's rows are fixed (``GATE2_CELLS``,
  ``gate2_labels``): the record binds a Gate 2 only with exactly those rows, each passing.
- **Gate 3** (``gate3_key``, ``gate3_agent``): teacher-forced wiring of a trained snapshot on
  its stored rows (``network.wiring_check``; no derivative, no perturbation): anchored finals
  from x_{L-1} and A2 rows from x0 to the outcome horizon. Each planted wiring error (another
  agent's obs_rms, prev_action zeroed, accel zeroed) must fail (mean error above 1e-4) on a
  spread sample; with ``all_keys`` the clean check runs on every key with a det row, and its
  failures are the per-key 'gate 3' exclusion list.
- **Gate 4** (``gate4_key``, ``gate4``): G* = 1 - dc_min/dDelta_in from the KKT field on every A2
  key (all 481, the tier V horizon twins included) within [0.98, 1.02] (``kkt.edge_gain``,
  ``kkt.gate4``), with the certified inertia and the adjoint check; keys outside (and keys
  that failed to re-polish) are listed and excluded (design Sections 5 and 8). Gate 4 passes
  when every key was processed without a crash (an exception other than ``RepolishFailed``,
  listed as 'crashed').
- **Gate 5** (``gate5_key``, ``gate5``; float64 process): the KKT replay. The KKT forward
  sensitivity (dw, dlam) along each of Gate 1's directions (``kkt.forward``) gives the plan's
  action tangent da*; the tape a* + eps da* is driven through the network side's closed loop
  (``network.closed_loop`` with a tape policy, ``network.outcome_vector``) on the road
  perturbed by eps delta (``perturb.perturbed_arrays``), and d outcome / d eps at 0 (one JVP)
  must match the KKT field along delta (``kkt.Fields.along``) to a relative L2 of 1e-3 per
  outcome over the directions (a field both sides put below 1e-6 everywhere scores 0, as in
  Gate 1; an infinite or NaN error of a valid outcome fails, ``gate5_outcomes_ok``), both read
  at the optimum's picks. The cross-pipeline outcomes are c_min, v_apex, s_b and c_line: a
  fixed tape makes vx independent of the road, so v_apex tests the action-to-vx linearisation
  and the road's d axes are tested by s_b and c_line (and c_min's edge term). The steer is a
  separate *decision check* (the tape's steer is da*[decision], whose agreement with the
  adjoint field is K's symmetry, plus the decision index both sides read). A planted negative
  control rolls the replay's road perturbation by 10 m while the tape keeps da*: the road
  outcomes must fail and the steer must not move. The replay's own plans (the network side's
  outcome definitions) must give the optimum's outcome values to 1e-8; their picks may differ
  only at a tie (on an active inside arc c_min's argmin is a tie at the 1e-10 level, and the
  replay's ~1e-10 drift can move it). The env's action clip is kept; da* ~ 0 is asserted where
  |a*| = 1 with a strongly active multiplier (the barrier leaves ~1e-7 of max |da*| there, and
  the clip's derivative is then irrelevant). Separately the state tangent is propagated through
  CasADi's step Jacobians (A_t, B_t from the defect rows, ``kkt._dynamics_blocks``; the
  curvature windows' dPhi/dW from ``kkt.param_blocks``) and through JAX AD of
  ``sim.control_step`` at the plan's own states; they must agree to 1e-8 (relative to the
  tangent's size). The replay starts where the network starts (``network.a2_start`` asserts
  the key's x0 is the cell's state bit for bit; an anchored key's tape starts with its fixed
  a_0 at x_{L-1}), and the key's car context must be the network's (``network.ctx64``).

**Numerical checks on every optimum-side row** (Gates 1, 4 and 5; fixed at the tag): the key
must have re-polished (``problem`` raises ``RepolishFailed`` otherwise, and the row fails), its
adjoint residuals must be within ``kkt.ADJOINT_RTOL``, and its inertia must be certified
(``kkt.Inertia.determined``). The re-polish record, the largest adjoint residual and the
barrier's fallback count are copied into each row.

Lockbox and pre-registration: keys are rebuilt through ``keys.rebuild`` (step 1's guard on the
stimulus); no trained network is differentiated or perturbed (Gate 3 is teacher-forced).
"""

from __future__ import annotations

import dataclasses
import functools
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.env import racing
from apexzero.fields import keys as K
from apexzero.fields import kkt as KK
from apexzero.fields import measures as MS
from apexzero.fields import network as N
from apexzero.fields import outcomes as O
from apexzero.fields import perturb as PT
from apexzero.fields.perturb import Samples
from apexzero.interp import io, registry
from apexzero.ocp.window import NA, NX
from apexzero.vehicle.sim import control_step

GATE0_TOL = K.GATE0_TOL
GATE1_STRATA = {"A2/400": 14, "A2/800": 6, "v1": 8, "v2": 8, "v3": 8, "S": 6}   # 50 keys
GATE2_TOL = N.GATE2_TOL
GATE2_ZERO = KK.ZERO_ATOL
# Gate 2's fixed rows (design Sections 8 and 15): the policies' A2 cells, the defaults of
# gate2_policies and of scripts/fields.py's gate2 stage; a run on other cells is partial.
GATE2_CELLS = {"random": (62, 345), "keeper": (344,), "linear": (344, 345, 66),
               "hybrid": N.HYBRID_CELLS}
GATE3_N_ANCHORED = 3       # Gate 3's planted sample: anchored finals and A2 keys per agent
GATE3_N_A2 = 2
GATE5_TOL = 1e-3           # relative L2 of the replayed fields against the KKT fields
GATE5_STATE_TOL = 1e-8     # CasADi against JAX state tangents (relative)
GATE5_BOUND_TOL = 1e-3     # |da*| where |a*| = 1, relative to max |da*| (seen: <= 3.1e-5)
GATE5_TIE = 1e-8          # m, m/s: the replay's outcome values against the optimum's
REPLAY_OUTCOMES = ("c_min", "v_apex", "s_b", "steer", "c_line")   # the replay JVP's order
GATE5_OUTCOMES = ("c_min", "v_apex", "s_b", "c_line")    # the cross-pipeline comparisons
GATE5_ROLL = 5             # samples (10 m): the planted road misalignment of Gate 5
LOOP_ROUND = 50            # Gate 5's loop lengths are rounded up to this (fewer compiles)


# ------------------------------------------------------------------------- Gate 0


def gate0_key(ref: K.KeyRef, bank=None) -> dict:
    """Gate 0 at one stored key (rebuild only, no solve)."""
    t0 = time.perf_counter()
    rec = K.load_key(ref)
    g = K.gate0(K.rebuild(rec, bank))
    return {"key_id": ref.key_id, "kind": ref.kind, "agent": ref.agent, "W": ref.W,
            "r_free": g.r_free, "r_at": g.r_at, "defect": g.defect,
            "corridor_violation": g.corridor_violation, "slip_violation": g.slip_violation,
            "J_rel": (g.J_rebuilt - g.J_stored) / abs(g.J_stored), "ok": g.ok,
            "seconds": time.perf_counter() - t0}


def _gate0_chunk(refs: list[K.KeyRef]) -> list[dict]:
    bank, _ = K.design()
    out = []
    for r in refs:
        try:
            out.append(gate0_key(r, bank))
        except Exception as e:  # noqa: BLE001  (a key that cannot be rebuilt fails the gate)
            out.append({"key_id": r.key_id, "kind": r.kind, "agent": r.agent, "ok": False,
                        "error": f"{type(e).__name__}: {e}"})
    return out


def map_keys(fn, refs: list, workers: int = 1, chunk: int = 50, progress=None) -> list:
    """``fn`` over chunks of keys, in ``workers`` spawned processes (1: in this process);
    ``progress(n_done)`` after each chunk."""
    chunks = [list(refs[i: i + chunk]) for i in range(0, len(refs), chunk)]
    out: list = []
    if workers <= 1:
        for c in chunks:
            out.extend(fn(c))
            if progress:
                progress(len(c))
        return out
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=get_context("spawn")) as ex:
        for c, res in zip(chunks, ex.map(fn, chunks), strict=True):
            out.extend(res)
            if progress:
                progress(len(c))
    return out


def gate0_summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    rf = np.array([r["r_free"] for r in rows if "r_free" in r], float)
    by: dict[str, dict] = {}
    for r in rows:
        k = r["kind"] if r["kind"] != "anchored" else f"anchored/{r['agent']}"
        b = by.setdefault(k, {"n": 0, "n_ok": 0, "max_r_free": 0.0})
        b["n"] += 1
        b["n_ok"] += int(bool(r.get("ok")))
        if "r_free" in r:
            b["max_r_free"] = max(b["max_r_free"], float(r["r_free"]))
    return {"tol": GATE0_TOL, "n": len(rows), "n_ok": len(ok), "max_r_free":
            float(rf.max()) if len(rf) else float("nan"),
            "max_defect": float(max((r.get("defect", 0.0) for r in rows), default=np.nan)),
            "failures": [{k: r.get(k) for k in ("key_id", "kind", "agent", "r_free", "r_at",
                                                "error")} for r in bad],
            "by_group": by, "pass": not bad}


def gate0(refs: list[K.KeyRef], workers: int = 1, progress=None) -> dict:
    """Gate 0 on every key of ``refs``; per-key rows and the summary."""
    rows = map_keys(_gate0_chunk, list(refs), workers, progress=progress)
    return {"summary": gate0_summary(rows), "rows": rows}


# ------------------------------------------------------------------------- Gate 1


def family(ref: K.KeyRef) -> str:
    """Gate 1's stratum of a key: 'A2/<W>' for A2 keys, the generalist variant ('v1', 'v2',
    'v3') or 'S' (specialists) for anchored keys."""
    if ref.kind != "anchored":
        return f"A2/{ref.W}"
    spec = registry.AGENTS[ref.agent]
    return spec.variant if spec.kind == "generalist" else "S"


def gate1_sample(strata: dict[str, int] = GATE1_STRATA, rng: np.random.Generator | None = None
                 ) -> list[K.KeyRef]:
    """Gate 1's keys: per stratum, that many keys drawn without replacement (step 2's
    'gate1_sample' stream) from the A2 set (by W) or the anchored finals (by family);
    sorted by stratum, then key id."""
    rng = MS.stream_rng("gate1_sample") if rng is None else rng
    pools: dict[str, list[K.KeyRef]] = {k: [] for k in strata}
    for r in K.a2_refs():
        if K.strata(r).w_is_cell and family(r) in pools:
            pools[family(r)].append(r)
    for refs in K.anchored_finals().values():
        for r in refs:
            if family(r) in pools:
                pools[family(r)].append(r)
    out = []
    for name, n in strata.items():
        pool = sorted(pools[name], key=lambda r: r.key_id)
        if len(pool) < n:
            raise ValueError(f"stratum {name}: {len(pool)} keys < {n}")
        pick = rng.choice(len(pool), size=n, replace=False)
        out.extend(pool[i] for i in sorted(pick))
    return out


class RepolishFailed(ValueError):
    """A key whose re-polish did not succeed (IPOPT status, or an RK stage left its kappa
    block): its point may not be a KKT point, so no field is computed there."""


def problem(ref: K.KeyRef, bank=None) -> tuple[K.KeyProblem, Any, dict]:
    """The re-polished problem of a stored key, its stimulus and the re-polish summary.
    Raises ``RepolishFailed`` unless the re-polish succeeded (``keys.Repolish.ok``: IPOPT's
    Solve_Succeeded and every RK stage inside its kappa block); the gates record such a key as
    a failed row."""
    if bank is None:
        bank, _ = K.design()
    rp = K.repolish(K.rebuild(K.load_key(ref), bank))
    if not rp.ok:
        raise RepolishFailed(f"key {ref.key_id}: re-polish status {rp.status}, block_ok "
                             f"{rp.block_ok}")
    return rp.problem, bank.stimuli[ref.stim_id], K.summary(rp)


def _repolish_record(rps: dict) -> dict:
    """The re-polish fields every gate row carries."""
    return {"repolish_ok": bool(rps["ok"]), "repolish_status": str(rps["status"]),
            "repolish_block_ok": bool(rps["block_ok"]),
            "repolish_iterations": int(rps["iterations"]),
            "repolish_flagged": bool(rps["flagged"])}


def _field_checks(fl: KK.Fields) -> dict:
    """The numerical checks of a key's fields every gate row carries (fixed at the tag)."""
    inr = fl.inertia
    return {"max_adjoint_residual": fl.max_adjoint_residual, "adjoint_ok": bool(fl.adjoint_ok),
            "barrier_fallback": int(fl.barrier_fallback),
            "inertia_ok": None if inr is None else bool(inr.ok),
            "inertia_method": None if inr is None else inr.method,
            "inertia_n_small": None if inr is None else int(inr.n_small),
            "inertia_determined": None if inr is None else bool(inr.determined)}


def _failed_row(ref: K.KeyRef, e: Exception, **extra) -> dict:
    return {"key_id": ref.key_id, "kind": ref.kind, "agent": ref.agent, "W": ref.W,
            "failed": True, "error": f"{type(e).__name__}: {e}", **extra}


def directions(prob: K.KeyProblem, stim) -> dict[str, Samples]:
    """Gate 1's (and Gate 5's) directions at a key (``perturb.gate_directions``)."""
    return KK.gate1_directions(prob.track.n_points, K.bump_origin(prob.rec), stim.landmarks,
                               int(stim.direction))


GATE1_COVERAGE = (("v_apex", "kappa"), ("v_apex", "w_in"), ("s_b", "kappa"), ("s_b", "w_in"))


def _errors(fl: KK.Fields, g1: KK.Gate1, dirs: dict[str, Samples]
            ) -> tuple[dict[str, float], list[str]]:
    """({'quantity|group': err}, [the zero fields' keys]) of ``fl``'s fields against g1's
    re-solves (``kkt.score_fields``)."""
    err, zero = KK.score_fields(fl, g1.names, dirs, g1.fd_r, g1.disc)
    return ({f"{q}|{g}": e for (q, g), e in err.items()},
            sorted(f"{q}|{g}" for (q, g), z in zero.items() if z))


def _corner_coverage(err: dict[str, float], zero: list[str], sigma: int) -> dict[str, bool]:
    """Whether the key has a non-zero, scored corner vector of v_apex and s_b in kappa and in
    its inside width (``GATE1_COVERAGE``)."""
    inside = PT.side_feature(int(sigma), "inside")
    out = {}
    for q, f in GATE1_COVERAGE:
        k = f"{q}|{inside if f == 'w_in' else f}@corner"
        out[f"{q}|{f}"] = bool(k in err and k not in zero)
    return out


def gate1_key(ref: K.KeyRef, plants: tuple[str, ...] = KK.PLANTS, bank=None) -> dict:
    """Gate 1 at one key: the clean fields' errors and each planted bug's, against one set
    of re-solves."""
    t0 = time.perf_counter()
    prob, stim, rps = problem(ref, bank)
    fl = KK.key_fields(prob, bank=bank)
    dirs = directions(prob, stim)
    g1 = KK.gate1(prob, fl, dirs, int(stim.direction))
    t1 = time.perf_counter()
    err, zero = _errors(fl, g1, dirs)
    planted = {}
    for plant in plants:
        flp = KK.fields(prob, stim.landmarks, int(stim.direction), K.decision_step(prob.rec),
                        plant=plant, check_inertia=False)
        pe, pz = _errors(flp, g1, dirs)
        planted[plant] = {"err": pe, "zero": pz}
    nonzero = [e for k, e in err.items() if k not in zero]
    return {"key_id": ref.key_id, "kind": ref.kind, "agent": ref.agent, "W": ref.W,
            "stratum": family(ref), "direction": int(stim.direction),
            "ok_solves": g1.ok_solves,
            "statuses": sorted({s for f in g1.fd for s in f.statuses}),
            "iterations": int(sum(sum(f.iterations) for f in g1.fd)), "n_solves": 4 * len(dirs),
            "n_directions": len(dirs), **_field_checks(fl), **_repolish_record(rps),
            "err": err, "zero": zero, "corner_coverage": _corner_coverage(err, zero,
                                                                          int(stim.direction)),
            "planted": planted, "worst": max(nonzero) if nonzero else float("nan"),
            "seconds_gate1": t1 - t0, "seconds": time.perf_counter() - t0}


def _gate1_chunk(refs: list[K.KeyRef]) -> list[dict]:
    bank, _ = K.design()
    out = []
    for r in refs:
        try:
            out.append(gate1_key(r, bank=bank))
        except Exception as e:  # noqa: BLE001  (a key that cannot be scored fails the gate)
            out.append(_failed_row(r, e, stratum=family(r)))
    return out


def _pooled(errs: list[float], n_zero: int = 0) -> dict:
    """Median, p95 and max of the scored (non-zero) errors, the pass rule on them, and how
    many zero fields were left out (they cannot fail, so they carry no information)."""
    e = np.asarray(errs, float)
    if not len(e):
        return {"n": 0, "n_zero": int(n_zero), "median": float("nan"), "p95": float("nan"),
                "max": float("nan"), "pass": False}
    with np.errstate(invalid="ignore"):          # plants can score inf (inf - inf in the
        med, p95 = float(np.median(e)), float(np.percentile(e, 95))      # interpolation)
    if np.isnan(p95):
        p95 = float(np.percentile(e, 95, method="higher"))
    return {"n": len(e), "n_zero": int(n_zero), "median": med, "p95": p95,
            "max": float(np.max(e)), "pass": bool(med < KK.GATE1_MEDIAN and p95 < KK.GATE1_P95)}


def _scored(rows: list[dict], which=None, select=lambda k: True) -> dict:
    """``_pooled`` over the rows' non-zero errors whose key passes ``select`` (``which``: a
    plant's errors instead of the clean ones)."""
    vals, nz = [], 0
    for r in rows:
        src = r["err"] if which is None else r["planted"].get(which, {}).get("err", {})
        zero = set(r["zero"] if which is None else r["planted"].get(which, {}).get("zero", []))
        for k, e in src.items():
            if not select(k):
                continue
            if k in zero:
                nz += 1
            else:
                vals.append(e)
    return _pooled(vals, nz)


def gate1_decision(rows: list[dict]) -> dict:
    """Gate 1's decision over every (key, quantity, group) field vector (design Section 8):
    the median < 1e-3 and p95 < 1e-2 rule on the non-zero vectors, pooled *and* within each
    quantity (so a bug confined to one quantity cannot hide in the pooled tail); each plant
    must fail the pooled rule; every key must have re-polished, solved its re-solves and
    passed the adjoint check; and each stratum must have scored corner vectors of v_apex and
    s_b in kappa and in the inside width (``GATE1_COVERAGE``). Zero fields are counted, per
    quantity and per (quantity, group), and never pooled."""
    good = [r for r in rows if not r.get("failed")]
    out = {"n_keys": len(rows), "failed": [{k: r.get(k) for k in ("key_id", "error")}
                                           for r in rows if r.get("failed")],
           "clean": _scored(good), "ok_solves": all(r["ok_solves"] for r in good),
           "inertia_ok": all(r["inertia_ok"] is not False for r in good),
           "repolish_ok": all(r["repolish_ok"] for r in good),
           "adjoint_ok": all(r["adjoint_ok"] for r in good), "plants": {}}
    for plant in sorted({p for r in good for p in r["planted"]}):
        pe = _scored(good, plant)
        pe["fails"] = not pe["pass"]
        out["plants"][plant] = pe
    keys = sorted({k for r in good for k in r["err"]})
    qs = sorted({k.split("|")[0] for k in keys})
    out["by_quantity"] = {q: _scored(good, select=lambda k, q=q: k.split("|")[0] == q)
                          for q in qs}
    out["zero_by_group"] = {k: sum(k in r["zero"] for r in good) for k in keys}
    strata = sorted({r["stratum"] for r in rows})
    out["by_stratum"] = {}
    for st in strata:
        sr = [r for r in good if r["stratum"] == st]
        d = _scored(sr)
        d["corner_coverage"] = {c: bool(any(r["corner_coverage"].get(c) for r in sr))
                                for c in (f"{q}|{f}" for q, f in GATE1_COVERAGE)}
        d["covered"] = bool(sr and all(d["corner_coverage"].values()))
        out["by_stratum"][st] = d
    out["quantities_pass"] = bool(qs and all(v["pass"] for v in out["by_quantity"].values()))
    out["coverage_pass"] = all(v["covered"] for v in out["by_stratum"].values())
    out["pass"] = bool(good and not out["failed"] and out["clean"]["pass"]
                       and out["quantities_pass"] and out["ok_solves"] and out["repolish_ok"]
                       and out["adjoint_ok"] and out["coverage_pass"]
                       and all(p["fails"] for p in out["plants"].values()))
    return out


def gate1(refs: list[K.KeyRef], workers: int = 1, progress=None) -> dict:
    rows = map_keys(_gate1_chunk, list(refs), workers, chunk=1, progress=progress)
    return {"decision": gate1_decision(rows), "rows": rows}


# ------------------------------------------------------------------------- Gate 2


def zero_rule_error(g: N.Gate2, zero_atol: float = GATE2_ZERO) -> np.ndarray:
    """``network.zero_rule_error`` (the rule lives beside ``network.Gate2``)."""
    return N.zero_rule_error(g, zero_atol)


def gate2_row(net: N.NetPolicy, rd: N.Road, st: N.Start, steps: int = N.STEPS,
              dirs: dict[str, Samples] | None = None, label: str = "") -> dict:
    """Gate 2 on one (policy, start) with Gate 1's directions by default. The pass rule is
    ``network.Gate2.ok``: every valid outcome finite and below 1e-3 under the zero rule, at
    least one outcome valid, and no valid outcome without a counted comparison."""
    N.require_x64()
    t0 = time.perf_counter()
    dirs = N.gate2_directions(rd, st) if dirs is None else dirs
    g = N.gate2(net, rd, st, dirs, steps=steps)
    return {"label": label or net.name, "policy": net.name, "start": st.source, "steps": steps,
            "outcomes": list(N.OUTCOMES), "valid": g.valid_outcomes(),
            "rel_err": g.zero_rule().tolist(), "rel_err_network": g.rel_err.tolist(),
            "share_counted": g.share_counted.tolist(), "names": g.names, "reasons": g.reason,
            "outcome_ok": g.outcome_ok(), "pass": bool(g.ok()),
            "seconds": time.perf_counter() - t0}


def gate2_labels(cells: dict = GATE2_CELLS) -> set[str]:
    """The labels of Gate 2's rows (``gate2_policies``) for ``cells``."""
    return ({f"random@{c}" for c in cells["random"]} | {f"edge-keeper@{c}" for c in cells["keeper"]}
            | {f"linear@{c}" for c in cells["linear"]} | {f"hybrid@{c}" for c in cells["hybrid"]})


def gate2_policies(cells_random=GATE2_CELLS["random"], cells_keeper=GATE2_CELLS["keeper"],
                   cells_linear=GATE2_CELLS["linear"], seed: int = 0, agent=None,
                   cells_hybrid=GATE2_CELLS["hybrid"]
                   ) -> list[tuple[str, N.NetPolicy, N.Road, N.Start, int]]:
    """Gate 2's (label, policy, road, start, steps) rows before the tag: a random-init actor
    (G1's architecture and obs_rms, read only; the instantaneous steer), the planted edge
    keeper (plan-level c_min, v_apex and s_b), the planted linear driver (345 is 344's
    mirror; 66 another geometry), and the hybrid control (``network.hybrid_net``: the linear
    driver plus a small-gain random-init actor through G1's obs_rms, the trained
    architecture inside the loop on plan-level outcomes; design Section 15)."""
    from apexzero.fields import controls as C

    g1 = C.g1_reference() if agent is None else agent
    rows = []
    for cid in cells_random:
        st, rd = C.start(cid)
        rows.append((f"random@{cid}", N.random_net(g1.cfg, g1.obs_rms, seed), rd, st, N.STEPS))
    for cid in cells_keeper:
        st, rd = C.start(cid)
        rows.append((f"edge-keeper@{cid}", C.keeper_net(st, rd, True), rd, st, C.KEEPER_STEPS))
    for cid in cells_linear:
        st, rd = C.start(cid)
        W, b = N.linear_driver(st.sigma)
        rows.append((f"linear@{cid}", N.linear_net(W, b, "linear_driver"), rd, st, 200))
    for cid in cells_hybrid:
        st, rd = C.start(cid)
        rows.append((f"hybrid@{cid}", N.hybrid_net(st.sigma, g1.cfg, g1.obs_rms, seed), rd, st,
                     N.HYBRID_STEPS))
    return rows


def gate2(rows=None) -> dict:
    """Gate 2 on ``gate2_policies`` (or given rows); pass when every row passes."""
    rows = gate2_policies() if rows is None else rows
    res = [gate2_row(net, rd, st, steps, label=label) for label, net, rd, st, steps in rows]
    return {"rows": res, "tol": GATE2_TOL, "pass": all(r["pass"] for r in res)}


# ------------------------------------------------------------------------- Gate 3


def gate3_key(ref: K.KeyRef, net: N.NetPolicy, det: dict, bank, cells,
              wrong: N.NetPolicy | None = None) -> dict:
    """Gate 3 at one key with a det row of ``net``'s snapshot (teacher-forced, no derivative,
    no perturbation): the wiring check from the network's restart (x_{L-1} under option A
    for anchored keys, x0 for A2 keys) to the outcome horizon, and, with ``wrong`` (the
    planted wrong obs_rms), each planted wiring error, which must fail. A key with no det
    row of this snapshot is 'skipped'; its row then carries no verdict."""
    rec = K.load_key(ref)
    try:
        i = N.det_row(det, rec.cell_id, rec.phase)
    except ValueError as e:
        return {"key_id": ref.key_id, "kind": ref.kind, "skipped": str(e)}
    if ref.kind == "anchored":
        st = N.anchored_start(rec, det, bank, "A")
    else:
        st = N.a2_start(rec, bank, cells)
    rd = N.road(bank, st.stim_id)
    w = N.wiring_check(net, det, i, rd.base, t0=st.offset, s_stop=st.s_stop)
    row = {"key_id": ref.key_id, "kind": ref.kind, "row": i, "steps": list(w.steps),
           "horizon": int(w.steps[1] - w.steps[0]), "mean_err": w.mean_err,
           "step_excess": w.step_excess.tolist(), "accel_err": w.accel_err,
           "prev_action_ok": w.prev_action_ok, "ok": w.ok}
    if wrong is not None:
        planted = {}
        for name, kw, n in (("obs_rms", {}, wrong),
                            ("prev_action", {"zero_prev_action": True}, net),
                            ("accel", {"zero_accel": True}, net)):
            wp = N.wiring_check(n, det, i, rd.base, t0=st.offset, s_stop=st.s_stop, **kw)
            planted[name] = {"mean_err": wp.mean_err, "fails": not wp.mean_ok}
        row["planted"] = planted
        row["planted_fail"] = all(p["fails"] for p in planted.values())
    return row


def gate3_refs(agent_id: str) -> list[K.KeyRef]:
    """Every key Gate 3 checks for an agent: its anchored finals and the A2 set (each is
    skipped where the agent's final det group has no row of its cell and phase)."""
    return list(K.anchored_finals().get(agent_id, ())) + list(K.a2_refs())


def gate3_agent(agent_id: str, n_anchored: int = GATE3_N_ANCHORED, n_a2: int = GATE3_N_A2,
                wrong_from: str | None = None,
                all_keys: bool = False, limit: int | None = None) -> dict:
    """Gate 3 on one trained snapshot's stored rows (teacher-forced; module docstring). The
    planted wiring errors run on a spread sample (n_anchored anchored finals, n_a2 A2 keys);
    with ``all_keys`` the clean check also runs on every key of ``gate3_refs`` (or ``limit``
    of them, spread), and its failures are the per-key 'gate 3' exclusion list."""
    N.require_x64()
    bank, cells = K.design()
    ag = N._final_agent(agent_id)
    other_id = wrong_from or ("G2" if agent_id == "G1" else "G1")
    other = N._final_agent(other_id)
    net, wrong = N.agent_net(ag), N.agent_net(ag, obs_rms=other.obs_rms)
    det = N.load_det(io.rollout_path(agent_id, ag.timesteps))
    finals = K.anchored_finals().get(agent_id, ())
    picks = [finals[i] for i in np.linspace(0, len(finals) - 1, n_anchored).astype(int)] \
        if len(finals) and n_anchored else []
    a2 = [r for r in K.a2_refs() if K.strata(r).w_is_cell]
    picks += [a2[i] for i in np.linspace(0, len(a2) - 1, n_a2).astype(int)] if n_a2 else []
    rows = [gate3_key(ref, net, det, bank, cells, wrong) for ref in picks]
    done = [r for r in rows if "skipped" not in r]
    out = {"agent": net.name, "wrong_obs_rms": other_id, "rows": rows,
           "planted_pass": bool(done and all(r["ok"] and r["planted_fail"] for r in done))}
    if all_keys:
        refs = gate3_refs(agent_id)
        if limit is not None and limit < len(refs):
            refs = [refs[i] for i in np.linspace(0, len(refs) - 1, int(limit)).astype(int)]
        every = [gate3_key(ref, net, det, bank, cells) for ref in refs]
        checked = [r for r in every if "skipped" not in r]
        out["every_key"] = {"n": len(every), "n_checked": len(checked),
                            "n_skipped": len(every) - len(checked),
                            "failures": [r["key_id"] for r in checked if not r["ok"]],
                            "max_mean_err": float(max((r["mean_err"] for r in checked),
                                                      default=float("nan"))),
                            "max_horizon": int(max((r["horizon"] for r in checked), default=0)),
                            "rows": every}
    out["pass"] = bool(out["planted_pass"] and (not all_keys or checked))
    return out


# ------------------------------------------------------------------------- Gate 4


def gate4_key(ref: K.KeyRef, bank=None) -> dict:
    """G*, the inertia and the adjoint check at one A2 key; ``ok`` when G* is in the band,
    the inertia is right and certified, and every adjoint residual is within ADJOINT_RTOL."""
    t0 = time.perf_counter()
    prob, stim, rps = problem(ref, bank)
    fl = KK.key_fields(prob, bank=bank)
    g = KK.edge_gain(fl, stim.landmarks, int(stim.direction))
    st = K.strata(ref)
    chk = _field_checks(fl)
    return {"key_id": ref.key_id, "kind": ref.kind, "cell": ref.cell_id, "phase": ref.phase,
            "W": ref.W, "w_is_cell": st.w_is_cell, "direction": st.direction, "R": st.R,
            "theta": st.theta_deg, "cap_binds": st.cap_binds, "G_star": g,
            "in_band": KK.gate4(g), **chk, "mu": fl.mu, "c_min": fl.reads["c_min"].value,
            **_repolish_record(rps),
            "ok": bool(KK.gate4(g) and chk["inertia_ok"] and chk["adjoint_ok"]),
            "seconds": time.perf_counter() - t0}


def _gate4_chunk(refs: list[K.KeyRef]) -> list[dict]:
    bank, _ = K.design()
    out = []
    for r in refs:
        try:
            out.append(gate4_key(r, bank))
        except RepolishFailed as e:      # a numerical exclusion (design Section 5), listed
            out.append({**_failed_row(r, e), "in_band": False, "inertia_ok": False,
                        "ok": False, "repolish_failed": True})
        except Exception as e:  # noqa: BLE001  (a crash: listed, and Gate 4 does not pass)
            out.append({**_failed_row(r, e), "in_band": False, "inertia_ok": False,
                        "ok": False, "crashed": True})
    return out


def gate4(refs: list[K.KeyRef], workers: int = 1, progress=None) -> dict:
    """Gate 4 on every key of ``refs`` (the stage passes all 481 A2 keys, horizon twins
    included). Keys not ``ok`` (G* outside the band, inertia or adjoint failing, or no
    re-polish) are listed in 'outside' as exclusions (design Sections 5 and 8); Gate 4 passes
    when every key was processed and none crashed ('crashed': any other exception)."""
    refs = list(refs)
    rows = map_keys(_gate4_chunk, refs, workers, chunk=10, progress=progress)
    g = np.array([r.get("G_star", np.nan) for r in rows], float)
    outside = [r for r in rows if not r.get("ok")]
    crashed = [r["key_id"] for r in rows if r.get("crashed")]
    return {"band": list(KK.GATE4_BAND), "adjoint_rtol": KK.ADJOINT_RTOL, "n": len(rows),
            "n_outside": len(outside),
            "n_undetermined": sum(r.get("inertia_determined") is False for r in rows),
            "outside": [{k: r.get(k) for k in ("key_id", "G_star", "inertia_ok",
                                               "inertia_determined", "adjoint_ok", "error",
                                               "repolish_failed")}
                        for r in outside],
            "crashed": crashed,
            "G_star_min": float(np.nanmin(g)) if np.isfinite(g).any() else np.nan,
            "G_star_max": float(np.nanmax(g)) if np.isfinite(g).any() else np.nan,
            "max_abs_dev": float(np.nanmax(np.abs(g - 1.0))) if np.isfinite(g).any() else np.nan,
            "rows": rows, "pass": bool(len(rows) == len(refs) and not crashed)}


# ------------------------------------------------------------------------- Gate 5


def _tape_policy(params, o, state, tr, aux):
    """The plan's commands as a policy: mean = tape[t] (t = steps since the restart)."""
    t = jnp.minimum(state.t, params.shape[0] - 1)
    return params[t], jnp.full((2,), -jnp.inf, params.dtype)


@functools.lru_cache(maxsize=8)
def _replay_jit(steps: int):
    p = O.drive_params()

    def outs(tape, dtape, base, samples, li, picks, delta, eps):
        tr = PT.perturbed_arrays(base, samples, Samples(*(eps * d for d in delta)))
        trc = N.closed_loop(_tape_policy, tape + eps * dtape, None, tr, li, steps)
        y = N.outcome_vector(trc.x, trc.mean, tr, picks, p)
        base_j = jax.tree.map(jnp.asarray, base)
        c_line = O.c_min(trc.x, base_j, picks.sigma, picks.c_min)
        return jnp.concatenate([y, c_line[None]])

    def jvp(tape, dtape, base, samples, li, picks, delta):
        return jax.jvp(lambda e: outs(tape, dtape, base, samples, li, picks, delta, e),
                       (jnp.float64(0.0),), (jnp.float64(1.0),))

    def run(tape, base, samples, li, delta):
        tr = PT.perturbed_arrays(base, samples, delta)
        return tr, N.closed_loop(_tape_policy, tape, None, tr, li, steps)

    return jax.jit(jvp), jax.jit(run)


@functools.lru_cache(maxsize=4)
def _state_tangent_jit(dt: float, substeps: int):
    """dx_{t+1} = J_x dx_t + J_a da_t + J_kc dkappa_c through ``control_step`` at the plan's
    states (JAX AD), as a scan over t."""

    def prop(xs, acts, das, base, dkc, p):
        def body(dx, inp):
            x, a, da = inp

            def step(x_, a_, kc_):
                return control_step(x_, a_, base._replace(kappa_c=kc_), p, dt, substeps)

            _, dnext = jax.jvp(step, (x, a, base.kappa_c), (dx, da, dkc))
            return dnext, dnext

        _, out = jax.lax.scan(body, jnp.zeros(NX, xs.dtype), (xs, acts, das))
        return out

    return jax.jit(prop)


def replay_start(ref: K.KeyRef, rec) -> N.Start:
    """Where the replay starts: an A2 key's network start (``network.a2_start``, asserting
    x0), or an anchored key's x_{L-1} = x0 with its fixed a_0 at the head of the tape; the
    decision step is the optimum's (``keys.decision_step``)."""
    bank, cells = K.design()
    if rec.kind != "anchored":
        return N.a2_start(rec, bank, cells)
    stim = bank.stimuli[rec.stim_id]
    return N.Start(x0=np.asarray(rec.x0, np.float64), prev_action=np.zeros(2), accel=np.zeros(2),
                   a_forced=np.zeros((0, 2)), decision_step=K.decision_step(rec),
                   s1=K.bump_origin(rec), stim_id=int(rec.stim_id), sigma=int(stim.direction),
                   landmarks=stim.landmarks, offset=0, source=f"key {rec.key_id} x0")


def _picks_equal(a: O.Plan, b: O.Plan) -> bool:
    if a.valid != b.valid:
        return False
    if not a.valid:
        return True
    return tuple(np.asarray(v).item() for v in a.pick) == tuple(np.asarray(v).item()
                                                               for v in b.pick)


def _rel_l2(got: np.ndarray, want: np.ndarray, zero_atol: float = GATE2_ZERO) -> float:
    """||got - want|| / ||want||: 0 for a zero field (both below ``zero_atol``), inf when the
    reference is exactly zero and ``got`` is not, NaN when either holds a NaN."""
    got, want = np.asarray(got, float), np.asarray(want, float)
    if np.isnan(got).any() or np.isnan(want).any():
        return float("nan")
    if max(np.abs(got).max(initial=0.0), np.abs(want).max(initial=0.0)) <= zero_atol:
        return 0.0
    den = float(np.linalg.norm(want))
    return float(np.linalg.norm(got - want) / den) if den > 0 else float("inf")


def gate5_outcomes_ok(err: dict[str, float], valid: dict[str, bool]) -> dict[str, bool]:
    """Per cross-pipeline outcome: a valid outcome passes only with a finite error <=
    GATE5_TOL (inf: the KKT field is zero and the replay's is not; NaN: a NaN in either
    field); an invalid one is skipped (True)."""
    return {q: bool(not valid[q] or (np.isfinite(err[q]) and err[q] <= GATE5_TOL))
            for q in GATE5_OUTCOMES}


def _rolled(delta: Samples, k: int) -> Samples:
    return Samples(*(np.roll(np.asarray(a, float), int(k)) for a in delta))


def gate5_key(ref: K.KeyRef, bank=None, dirs: dict[str, Samples] | None = None) -> dict:
    """Gate 5 at one key (module docstring); a float64 process. Besides the cross-pipeline
    errors (c_min, v_apex, s_b, c_line), it reports the decision check (steer: a KKT
    forward/adjoint identity plus the decision index, not cross-pipeline evidence) and a
    planted negative control: the replay's road perturbed by delta rolled GATE5_ROLL samples
    (10 m) while its tape keeps the KKT's da*, which must fail on the road-reading outcomes
    and leave the tape-only steer unchanged."""
    N.require_x64()
    t0 = time.perf_counter()
    prob, stim, rps = problem(ref, bank)
    rec, T = prob.rec, prob.T
    fl = KK.key_fields(prob, check_inertia=False, bank=bank)
    kkt = KK.kkt_system(prob)
    blocks = KK.param_blocks(prob)
    dirs = directions(prob, stim) if dirs is None else dirs
    st = replay_start(ref, rec)
    if not np.array_equal(prob.ctx, N.ctx64()):
        raise ValueError(f"key {ref.key_id}: its car context is not the network's")
    rd = N.road(bank if bank is not None else K.design()[0], rec.stim_id)
    s_stop = st.s_stop
    k_stop = int(np.argmax(prob.x[:, 0] >= s_stop)) if np.any(prob.x[:, 0] >= s_stop) else T
    steps = min(T, LOOP_ROUND * int(np.ceil((k_stop + 1) / LOOP_ROUND)))
    tape = np.asarray(prob.a, float)[:steps]
    jvp_fn, run_fn = _replay_jit(steps)
    li = N.loop_inputs(st, steps)
    tr0, trc = run_fn(tape, rd.base, rd.samples, li, PT.zeros(rd.n))
    base_run = N.as_run(st, tr0, trc)
    opt_plans = KK.outcome_plans(prob, stim.landmarks, int(stim.direction))
    picks_match = {o: _picks_equal(base_run.plans[o], opt_plans[o]) for o in O.OUTCOMES}
    # the replay's outcome values must be the optimum's; a different pick is then a tie (the
    # replay's ~1e-10 drift moves the argmin along an active arc). Fields are compared at
    # the optimum's picks either way.
    value_diff = {o: abs(base_run.plans[o].value - opt_plans[o].value) if opt_plans[o].valid
                  else 0.0 for o in O.OUTCOMES}
    picks_ok = all(value_diff[o] <= GATE5_TIE for o in O.OUTCOMES)
    n_cmp = min(base_run.n_steps + 1, k_stop + 1)
    replay_dx = float(np.max(np.abs(base_run.x[:n_cmp] - prob.x[:n_cmp])))
    picks = N.picks_of(dataclasses.replace(base_run, plans=opt_plans))
    a_star = np.asarray(prob.a, float)
    lam_a = np.asarray(prob.lam_x, float)[NX * (T + 1):].reshape(T, NA)
    free_a = prob.free[NX * (T + 1):].reshape(T, NA)
    # |a*| = 1 up to the barrier's slack (mu / z ~ 1e-11), with a strongly active multiplier
    at_bound = free_a & (np.abs(a_star) >= 1.0 - 1e-6) & (np.abs(lam_a) >= KK.STRONG_Z)
    names = list(dirs)
    got = np.zeros((len(names), len(REPLAY_OUTCOMES)))
    rolled = np.zeros_like(got)
    want = np.zeros_like(got)
    bound_da, state_err, kkt_res = 0.0, 0.0, 0.0
    p_env = jax.tree.map(jnp.asarray, racing.params_from_context(jnp.asarray(N.ctx64()),
                                                                 hidden_mu=1.0))
    A, B = KK._dynamics_blocks(kkt.J, T)
    prop = _state_tangent_jit(float(N.ENV.dt), int(N.ENV.substeps))
    base_j = jax.tree.map(jnp.asarray, rd.base)
    for m, nm in enumerate(names):
        delta = dirs[nm]
        dw, _, res = KK.forward(kkt, blocks, delta)
        kkt_res = max(kkt_res, res)
        da = dw[NX * (T + 1):].reshape(T, NA)
        scale = max(float(np.abs(da).max()), 1e-300)
        if at_bound.any():
            bound_da = max(bound_da, float(np.abs(da[at_bound]).max()) / scale)
        _, dy = jvp_fn(tape, da[:steps], rd.base, rd.samples, li, picks, delta)
        got[m] = np.asarray(dy)
        _, dyr = jvp_fn(tape, da[:steps], rd.base, rd.samples, li, picks,
                        _rolled(delta, GATE5_ROLL))
        rolled[m] = np.asarray(dyr)
        want[m] = [fl.along(q, delta) if fl.valid(q) else np.nan for q in REPLAY_OUTCOMES]
        # state tangents: CasADi's step Jacobians against JAX AD of control_step
        Wk = KK.coeffs(np.asarray(delta.kappa, float))[blocks.cidx]           # [T, K]
        dphi_w = -np.einsum("trk,tk->tr", blocks.dyn_g, Wk)
        dx_c = np.zeros((T + 1, NX))
        for t in range(T):
            dx_c[t + 1] = A[t] @ dx_c[t] + B[t] @ da[t] + dphi_w[t]
        dkc = PT.strip_tangent(rd.samples, delta).kappa_c
        dx_j = np.asarray(prop(jnp.asarray(prob.x[:T]), jnp.asarray(a_star), jnp.asarray(da),
                               base_j, dkc, p_env))
        sz = max(float(np.abs(dx_c[1:]).max()), 1e-300)
        state_err = max(state_err, float(np.abs(dx_c[1:] - dx_j).max()) / sz)
    plan_of = {"c_line": "c_min"}
    col = {q: k for k, q in enumerate(REPLAY_OUTCOMES)}
    valid = {q: bool(fl.valid(q) and opt_plans[plan_of.get(q, q)].valid) for q in GATE5_OUTCOMES}
    err = {q: (_rel_l2(got[:, col[q]], want[:, col[q]]) if valid[q] else float("nan"))
           for q in GATE5_OUTCOMES}
    outcome_ok = gate5_outcomes_ok(err, valid)
    # the decision check: the tape's steer at the decision step is the KKT's da*; its error
    # against the adjoint field is K's symmetry, plus the index the two sides read
    k_dec = col["steer"]
    dec = {"decision_net": int(st.decision_step), "decision_kkt": int(K.decision_step(rec)),
           "steer_valid": bool(fl.valid("steer"))}
    dec["steer_err"] = (_rel_l2(got[:, k_dec], want[:, k_dec]) if dec["steer_valid"]
                        else float("nan"))
    dec["ok"] = bool(dec["decision_net"] == dec["decision_kkt"] == fl.decision
                     and (not dec["steer_valid"] or (np.isfinite(dec["steer_err"])
                                                     and dec["steer_err"] <= GATE5_TOL)))
    # the planted misalignment: road-reading outcomes must fail, the tape-only steer must not
    # move (v_apex reads vx, which a fixed tape makes independent of the road, except at an
    # interpolated window-end pick)
    roll_err = {q: (_rel_l2(rolled[:, col[q]], want[:, col[q]]) if valid[q] else float("nan"))
                for q in GATE5_OUTCOMES}
    road = [q for q in ("c_min", "s_b", "c_line") if valid[q]]
    planted = {"roll": GATE5_ROLL, "err": roll_err,
               "road_fails": bool(any(not (roll_err[q] <= GATE5_TOL) for q in road)),
               "steer_unchanged": bool(np.array_equal(rolled[:, k_dec], got[:, k_dec]))}
    planted["fails"] = bool(planted["road_fails"] and planted["steer_unchanged"])
    ok = (all(outcome_ok.values()) and any(valid.values()) and dec["ok"]
          and state_err <= GATE5_STATE_TOL and bound_da <= GATE5_BOUND_TOL and picks_ok
          and fl.adjoint_ok and planted["fails"])
    return {"key_id": ref.key_id, "kind": ref.kind, "W": T, "loop_steps": steps,
            "n_directions": len(names), "err": err, "valid": valid, "outcome_ok": outcome_ok,
            "decision_check": dec, "planted_roll": planted,
            "state_tangent_err": state_err, "bound_da": bound_da,
            "n_at_bound": int(at_bound.sum()), "picks_match": picks_match,
            "near_tie": {o: bool(opt_plans[o].near_tie) for o in O.OUTCOMES},
            "value_diff": value_diff, "picks_ok": picks_ok,
            "replay_dx": replay_dx, "kkt_residual": kkt_res, **_field_checks(fl),
            **_repolish_record(rps),
            "max_field": {q: float(np.nanmax(np.abs(want[:, k]))) for k, q in
                          enumerate(REPLAY_OUTCOMES)},
            "pass": bool(ok), "seconds": time.perf_counter() - t0}


def gate5(refs: list[K.KeyRef], progress=None) -> dict:
    """Gate 5 on keys (one process: the replay's jit is shared); a key that cannot be
    replayed is a failed row."""
    bank, _ = K.design()
    rows = []
    for r in refs:
        try:
            rows.append(gate5_key(r, bank))
        except Exception as e:  # noqa: BLE001  (a key that cannot be replayed fails the gate)
            rows.append({**_failed_row(r, e), "pass": False})
        if progress:
            progress(1)
    return {"tol": GATE5_TOL, "state_tol": GATE5_STATE_TOL, "rows": rows,
            "pass": bool(rows and all(r["pass"] for r in rows))}
