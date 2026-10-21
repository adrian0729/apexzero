"""Step 1's answer keys as read-only data for step 2 (design Sections 2 and 5, Gate 0).

Records live under ``outputs/psychophysics/keys/<kind>/<key_id>.{json,npz}`` (``KeyCache``,
answer_key.py:2647; arrays x0, x, a, lam_g, lam_x, active_steps, answer_key.py:2614). They are
read here and never written. Three kinds are used:

- **A2 state set** (``a2_refs``): verified ``nominal`` (u = 0) and ``phase`` (u_1..u_3) keys
  whose cell's approach is 'A2', the cells from ``stimuli.load_design(outputs/psychophysics/
  stimuli)`` as scripts/psychophysics.py:257 ``_design`` loads them. 142 + 339 = 481 keys. Six
  tier V cells hold both a W = 400 and a W = 800 nominal key (step 1's horizon check); both
  are listed, as in the design's count, and ``Strata.w_is_cell`` tells them apart; the
  per-pair summaries use ``a2_primary_refs`` (475 keys, the twins left out and reported
  descriptively), while every gate runs on all 481. Each is
  solved from the cell's designed x0 (``Cell.initial_state``, stimuli.py:372) with
  a_{-1} = 0; the free decision is a*_0.
- **Anchored finals** (``anchored_finals``): verified ``anchored`` keys whose
  ``anchor['snapshot']`` is the agent's final snapshot's timesteps (``agents.final_snapshot``
  and ``agents.load_agent``, agents.py:122-164; read from the pickle, never hard-coded).
  Earlier snapshots (step 1's P6c keys) are listed separately (``anchored_earlier``). An
  anchored key is solved from the agent's x_{L-1} with a_0 fixed to a_{L-1}
  (``AnchoredSolver``, answer_key.py:562-680; the fixed bound at :613-616), so its free
  decision is a*_1 at x_1.

**The rebuilt NLP** (``rebuild``). P = ``WindowSolver.parameters`` (window.py:631-635) of
``bank.track(stim_id)`` (stimuli.py:476), with window centres ``_window_centres`` (window.py:410)
of the stored states, the car context ``key_ctx`` (answer_key.py:275: float32 values cast,
hidden_mu = 1) and the record's alpha_max; the solver is ``make_solver(W, spec=...)``
(answer_key.py:698), unscaled. The bounds copy ``AnchoredSolver.solve`` (answer_key.py:598-
619). The original solve centred its windows on its last warm start, not on the solution; the
lookups are exact while a node stays inside its window (window.py:141-184), so any centring
that holds the solution gives the same constraint function, and only the inactive window
bounds |s_t - sigma_t| <= 60 m move.

**Gate 0** (design Section 8): with CasADi's convention f = -J and L = f + lam_g' g + lam_x' w,
the stored multipliers make r = grad f + J_g' lam_g + lam_x vanish on the free variables
(lbx < ubx: all but x_0, and a_0 for anchored keys). ``gate0`` reports max |r_free| and the
dynamics defects g[0:8T] of the stored primal under the rebuilt P.

**Re-polish** (design Section 5). Keys solved on the rented machine rebuild with defects of
1e-8 to 2e-7 here (its float32 track arrays differ in the last bits), keys solved here to
1e-10. ``repolish`` runs one ``REFINE_IPOPT`` solve (window.py:405) warm-started from the stored
primal and dual on the rebuilt P with the same bounds, and records the movement; a key is
flagged when any state moves more than 1e-6 or J by more than 1e-9 relative. All keys are kept.

Lockbox: every rebuild calls step 1's guard on the stimulus (``answer_key._guard``,
answer_key.py:297); R_TEST circuits never reach the solver.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apexzero.fields.perturb import Samples, perturbed_track
from apexzero.interp import agents, io
from apexzero.interp import answer_key as ak
from apexzero.interp.answer_key import KeyRecord
from apexzero.interp.constants import DT
from apexzero.interp.measures import Trajectory
from apexzero.interp.stimuli import Cell, StimulusBank, load_design
from apexzero.ocp.window import (
    MAX_TRAVEL,
    NA,
    NX,
    REFINE_IPOPT,
    WindowSpec,
    _window_centres,
    block_offset_range,
    coefficient_windows,
    track_scalars,
)
from apexzero.tracks.spline import packed_coeffs
from apexzero.tracks.track import Track
from apexzero.vehicle.params import F1_2025, VehicleParams
from apexzero.vehicle.sim import TrackArrays

KINDS = ("nominal", "phase", "anchored")
A2_KINDS = ("nominal", "phase")
A2_APPROACH = "A2"
MOVE_X_TOL = 1e-6          # m (any state), re-polish flag (design Section 5)
MOVE_J_RTOL = 1e-9         # relative J change, re-polish flag
GATE0_TOL = 1e-6           # max |r_free| (design Section 8, Gate 0)
STATE_NAMES = ("s", "n", "xi", "vx", "vy", "r", "delta", "pedal")
ACTION_NAMES = ("steer", "pedal")

# ------------------------------------------------------------------------- the index


@dataclass(frozen=True)
class KeyRef:
    """One key's JSON metadata (no arrays): enough to select, stratify and load it."""

    kind: str
    key_id: str
    status: str
    stim_id: int
    stim_name: str
    cell_id: int
    phase: int
    W: int
    direction: int            # stimulus direction: +1 left, -1 right
    pair_id: int
    agent: str = ""           # anchored: whose state
    snapshot: str = ""        # anchored: anchor['snapshot'] (timesteps, a string)
    anchor_step: int = -1     # anchored: L
    measures_hash: str = ""

    @property
    def verified(self) -> bool:
        return self.status == "verified"


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else io.KEYS_DIR


@functools.cache
def _index(kind: str, root: str) -> tuple[KeyRef, ...]:
    out = []
    for j in sorted((Path(root) / kind).glob("*.json")):
        d = json.loads(j.read_text())
        an = d.get("anchor") or {}
        out.append(KeyRef(kind=kind, key_id=d["key_id"], status=d["status"],
                          stim_id=int(d["stim_id"]), stim_name=d["stim_name"],
                          cell_id=int(d["cell_id"]), phase=int(d["phase"]), W=int(d["W"]),
                          direction=int(d["direction"]), pair_id=int(d.get("pair_id", -1)),
                          agent=d.get("agent", ""), snapshot=str(an.get("snapshot", "")),
                          anchor_step=int(an.get("step", -1)),
                          measures_hash=d.get("measures_hash", "")))
    return tuple(out)


def key_index(kind: str, root: Path | None = None) -> tuple[KeyRef, ...]:
    """Every record of one kind (JSON only, sorted by key id; cached per process)."""
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} is not one of {KINDS}")
    return _index(kind, str(_root(root)))


def load_key(kind: str | KeyRef, key_id: str | None = None, root: Path | None = None
             ) -> KeyRecord:
    """The full record (``KeyCache.get``: JSON and arrays)."""
    if isinstance(kind, KeyRef):
        kind, key_id = kind.kind, kind.key_id
    rec = ak.KeyCache(_root(root)).get(kind, key_id)
    if rec is None:
        raise KeyError(f"no {kind} key {key_id} under {_root(root)}")
    return rec


@functools.cache
def _design(stimuli_dir: str) -> tuple[StimulusBank, tuple[Cell, ...]]:
    bank, cells = load_design(Path(stimuli_dir))
    return bank, tuple(cells)


def design(stimuli_dir: Path | None = None) -> tuple[StimulusBank, dict[int, Cell]]:
    """Step 1's stimulus design: the bank and the cells by id (cached per process)."""
    bank, cells = _design(str(Path(stimuli_dir) if stimuli_dir is not None else io.STIMULI_DIR))
    return bank, {c.cell_id: c for c in cells}


def a2_refs(root: Path | None = None, stimuli_dir: Path | None = None) -> tuple[KeyRef, ...]:
    """The A2 state set: verified nominal and phase keys whose cell approach is A2."""
    _, cells = design(stimuli_dir)
    return tuple(r for kind in A2_KINDS for r in key_index(kind, root)
                 if r.verified and cells[r.cell_id].approach == A2_APPROACH)


def a2_primary_refs(root: Path | None = None, stimuli_dir: Path | None = None
                    ) -> tuple[KeyRef, ...]:
    """The A2 keys the per-pair summaries use (H2', H4, H1-typical): the A2 set without the 6
    tier V horizon twins (``Strata.w_is_cell``: the key's W is the cell's), 475 keys, one per
    (cell, phase). The twins are reported descriptively (horizon-check duplicates); every
    gate (Gate 4 included) still runs on all 481 keys."""
    return tuple(r for r in a2_refs(root, stimuli_dir) if strata(r, stimuli_dir).w_is_cell)


def a2_twins(root: Path | None = None, stimuli_dir: Path | None = None) -> tuple[KeyRef, ...]:
    """The tier V horizon twins of the A2 set (``a2_refs`` minus ``a2_primary_refs``)."""
    return tuple(r for r in a2_refs(root, stimuli_dir) if not strata(r, stimuli_dir).w_is_cell)


@functools.cache
def final_timesteps(agent_id: str) -> int:
    """Timesteps of the agent's final snapshot, read from the snapshot itself (S1's agent.pkl
    carries none; ``load_agent`` gives num_updates * batch_size, agents.py:122-147)."""
    spec = agents.AGENTS[agent_id]
    path = agents.final_snapshot(spec)
    if path is None:
        raise ValueError(f"{agent_id}: run not finished, no final snapshot")
    return int(agents.load_agent(path, agent_id).timesteps)


def anchored_finals(root: Path | None = None) -> dict[str, tuple[KeyRef, ...]]:
    """Verified anchored keys at each agent's final snapshot, in registry order (agents with
    no anchored key are left out)."""
    refs = [r for r in key_index("anchored", root) if r.verified]
    out = {}
    for aid in agents.AGENTS:
        mine = [r for r in refs if r.agent == aid]
        if mine:
            final = str(final_timesteps(aid))
            out[aid] = tuple(r for r in mine if r.snapshot == final)
    return out


def anchored_earlier(root: Path | None = None) -> dict[tuple[str, str], tuple[KeyRef, ...]]:
    """Verified anchored keys at earlier snapshots, by (agent, snapshot): descriptive."""
    out: dict[tuple[str, str], list[KeyRef]] = {}
    for r in key_index("anchored", root):
        if r.verified and r.snapshot != str(final_timesteps(r.agent)):
            out.setdefault((r.agent, r.snapshot), []).append(r)
    return {k: tuple(v) for k, v in sorted(out.items())}


# ------------------------------------------------------------------------- strata


@dataclass(frozen=True)
class Strata:
    """What a key is stratified by (design Sections 2, 4 and 6)."""

    approach: str             # the cell's: 'A2', 'A1', 'B', ...
    direction: int            # sigma: +1 left corner, -1 right
    R: float                  # CornerSpec.R (m); nan off the synthetic grid
    theta_deg: float          # CornerSpec.theta_deg
    width: float              # total width w (m): 12, or tier C's 9 / 15
    pair_id: int              # mirror pair (with approach: protocol 9.1's unit)
    phase: int                # k of u_k
    n_phases: int
    W: int                    # the key's window (steps)
    w_is_cell: bool           # W equals the cell's W (False: a tier V horizon twin)
    cap_binds: bool           # the generator's inside cap narrows the inside (Geometry)
    no_cap: bool              # tier A no-cap variant
    tiers: tuple[str, ...]
    v0: float                 # m/s
    D: float                  # m, s_c0 - s0 at u = 0

    @property
    def geometry(self) -> tuple[float, float]:
        """(R, theta): the cluster of the bootstrap (design Section 6)."""
        return self.R, self.theta_deg


def strata(ref: KeyRef | KeyRecord, stimuli_dir: Path | None = None) -> Strata:
    bank, cells = design(stimuli_dir)
    cell = cells[ref.cell_id]
    stim = bank.stimuli[ref.stim_id]
    spec = stim.spec
    nan = float("nan")
    return Strata(approach=cell.approach, direction=int(stim.direction),
                  R=float(spec.R) if spec else nan, theta_deg=float(spec.theta_deg) if spec else nan,
                  width=float(spec.width) if spec else nan, pair_id=int(cell.pair_id),
                  phase=int(ref.phase), n_phases=int(cell.n_phases), W=int(ref.W),
                  w_is_cell=cell.W is None or int(cell.W) == int(ref.W),
                  cap_binds=bool(stim.geometry.cap_binds), no_cap=bool(spec.no_cap) if spec else False,
                  tiers=tuple(cell.tiers), v0=float(cell.v0),
                  D=nan if cell.D is None else float(cell.D))


def is_anchored(rec: KeyRecord | KeyRef) -> bool:
    return rec.kind == "anchored"


def decision_step(rec: KeyRecord | KeyRef) -> int:
    """Index of the optimum's free decision: a*_1 at x_1 for anchored keys (a_0 is fixed), a*_0
    at x_0 otherwise (design Section 2)."""
    return 1 if is_anchored(rec) else 0


def bump_origin(rec: KeyRecord) -> float:
    """s_1, the bumps' origin (design Section 3 'Names'): x[1, 0] for anchored keys, x[0, 0]
    for A2 keys."""
    return float(np.asarray(rec.x)[decision_step(rec), 0])


# ------------------------------------------------------------------------- the rebuilt NLP


def variable_name(i: int, T: int) -> str:
    """Name of entry i of w = [vec(X) (T+1 states of 8); vec(A) (T actions of 2)]."""
    if i < NX * (T + 1):
        return f"x[{i // NX}].{STATE_NAMES[i % NX]}"
    j = i - NX * (T + 1)
    return f"a[{j // NA}].{ACTION_NAMES[j % NA]}"


def key_car(rec: KeyRecord) -> VehicleParams:
    """The problem's car: F1_2025 with the record's alpha_max (backed-off keys are not used)."""
    return F1_2025._replace(alpha_max=float(rec.alpha_max))


@dataclass
class KeyProblem:
    """A stored key's NLP rebuilt on this machine, with its stored primal-dual point."""

    rec: KeyRecord
    spec: WindowSpec
    solver: ak.AnchoredSolver
    track: Track
    p: VehicleParams
    ctx: np.ndarray           # [6] driven context (key_ctx; hidden_mu = 1)
    sigma: np.ndarray         # [T+1] window centres
    P: np.ndarray             # solver parameters
    lbx: np.ndarray
    ubx: np.ndarray
    lbg: np.ndarray
    ubg: np.ndarray
    w: np.ndarray             # primal [nw] = [vec(x); vec(a)]
    lam_g: np.ndarray
    lam_x: np.ndarray

    @property
    def T(self) -> int:
        return int(self.spec.steps)

    @property
    def free(self) -> np.ndarray:
        """[nw] bool: variables IPOPT optimises (lbx < ubx)."""
        return self.lbx < self.ubx

    @property
    def x(self) -> np.ndarray:
        return self.w[: NX * (self.T + 1)].reshape(self.T + 1, NX)

    @property
    def a(self) -> np.ndarray:
        return self.w[NX * (self.T + 1):].reshape(self.T, NA)

    def coefficient_index(self) -> np.ndarray:
        """[T, K] index into the packed coefficient arrays (``packed_coeffs``, length n + 2) of
        each window entry of P: window m of step t is packed coefficient
        clip(j0_t + m, -1, n) + 1 (window.py:187-201)."""
        n = self.track.n_points
        _, j0 = coefficient_windows(self.track, self.sigma[:-1], self.spec)
        idx = j0.astype(int)[:, None] + np.arange(self.spec.n_coef)[None, :]
        return np.clip(idx, -1, n) + 1


def key_arrays(track: Track) -> TrackArrays:
    """The optimum's splines as NumPy float64 TrackArrays: ``packed_coeffs`` of the Track's
    node values (what the coefficient windows hold, window.py:200), laid out as
    ``sim.track_arrays`` (sim.py:37-56) lays them out, without JAX (so no float32 truncation
    in an x64-off process). For ``outcomes`` on the optimum's plan."""
    n, closed = track.n_points, bool(track.closed)

    def coef(a):
        return packed_coeffs(np.asarray(a, float), closed, n)

    return TrackArrays(kappa_c=coef(track.kappa), w_left_c=coef(track.w_left),
                       w_right_c=coef(track.w_right), x=np.asarray(track.x, float),
                       y=np.asarray(track.y, float), theta=np.asarray(track.theta, float),
                       ds=np.asarray(track.ds, float), n_points=np.asarray(n, np.int32),
                       closed=np.asarray(closed))


def trajectory(prob: KeyProblem) -> Trajectory:
    """The problem's planned states (stored or re-polished) as a ``measures.Trajectory`` at
    20 Hz, every state valid, commands attached (a_{-1} = 0 for A2 keys; anchored keys start
    at x_{L-1}, and their a_0 is the agent's a_{L-1})."""
    x = prob.x.copy()
    K = len(x)
    return Trajectory(source="to_ocp", t=np.arange(K) * DT, x=x, valid=np.ones(K, bool),
                      a=prob.a.copy(), dt=DT, meta={"key_id": prob.rec.key_id})


def parameter_slices(T: int, K: int) -> dict[str, slice]:
    """Where each block sits in P (window.py:462-467): the kappa, w_left and w_right windows
    (step t's K coefficients contiguous), j0 [T], the track scalars q [4], ctx [6] and
    alpha_max [1]."""
    out, at = {}, 0
    for name, size in (("kappa", K * T), ("w_left", K * T), ("w_right", K * T), ("j0", T),
                       ("q", 4), ("ctx", 6), ("alpha_max", 1)):
        out[name] = slice(at, at + size)
        at += size
    return out


def key_parameters(solver: ak.AnchoredSolver, track: Track, sigma: np.ndarray, ctx: np.ndarray,
                   p: VehicleParams, delta: Samples | None = None) -> np.ndarray:
    """P for the track's samples plus ``delta`` (``perturb.perturbed_track``): the
    optimum-side entry of a sample-space perturbation (design Section 4)."""
    return solver.parameters(perturbed_track(track, delta), np.asarray(sigma)[:-1], ctx, p)


def bounds(rec: KeyRecord, spec: WindowSpec, sigma: np.ndarray, p: VehicleParams
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(lbx, ubx, lbg, ubg) as ``AnchoredSolver.solve`` builds them (answer_key.py:604-619):
    state box, s within +-radius of the window centres, x_0 fixed, actions in [-1, 1] (a_0
    fixed to the stored a_0 for anchored keys), defects = 0, corridor <= 0, |slip| <= 1 -
    slip_safety / alpha_max."""
    T, R = int(spec.steps), spec.radius
    x0 = np.asarray(rec.x0, float)
    lo = np.maximum(sigma[1:], sigma[:-1]) - R
    hi = np.minimum(sigma[1:], sigma[:-1] + MAX_TRAVEL - 0.5) + R
    if not np.all(lo < hi):
        raise ValueError("empty window bounds")
    lbX = np.tile(np.array([-np.inf, -np.inf, -0.5 * np.pi, 2.0, -25.0, -5.0, -p.delta_max,
                            -1.0])[:, None], (1, T + 1))
    ubX = np.tile(np.array([np.inf, np.inf, 0.5 * np.pi, 120.0, 25.0, 5.0, p.delta_max,
                            1.0])[:, None], (1, T + 1))
    lbX[0, 1:], ubX[0, 1:] = lo, hi
    lbX[:, 0] = ubX[:, 0] = x0
    lbx = np.concatenate([lbX.T.reshape(-1), -np.ones(NA * T)])
    ubx = np.concatenate([ubX.T.reshape(-1), np.ones(NA * T)])
    if is_anchored(rec):
        ia = NX * (T + 1)
        lbx[ia: ia + NA] = ubx[ia: ia + NA] = np.asarray(rec.a, float)[0]
    slip = 1.0 - spec.slip_safety / p.alpha_max
    lbg = np.concatenate([np.zeros(NX * T), np.full(2 * T, -np.inf), np.full(2 * T, -slip)])
    ubg = np.concatenate([np.zeros(NX * T), np.zeros(2 * T), np.full(2 * T, slip)])
    return lbx, ubx, lbg, ubg


def rebuild(rec: KeyRecord, bank: StimulusBank | None = None, track: Track | None = None
            ) -> KeyProblem:
    """The key's NLP on this machine: solver, rebuilt P, bounds and the stored point."""
    if rec.status != "verified" or rec.x is None or rec.lam_g is None:
        raise ValueError(f"{rec.kind} key {rec.key_id} is {rec.status}: nothing to rebuild")
    if rec.kind not in KINDS or float(rec.shrink) != 0.0:
        raise ValueError(f"{rec.kind} keys (shrink {rec.shrink}) are not step 2 keys")
    if bank is None:
        bank, _ = design()
    ak._guard(bank.stimuli[rec.stim_id])
    if track is None:
        track = bank.track(rec.stim_id)
    spec = WindowSpec(**rec.spec)
    if int(spec.steps) != int(rec.W):
        raise ValueError(f"key {rec.key_id}: spec steps {spec.steps} != W {rec.W}")
    solver = ak.make_solver(int(rec.W), spec=spec)
    if solver.scaled:
        raise ValueError("step 1's keys were solved unscaled")
    p = key_car(rec)
    ctx = ak.key_ctx(p)
    x = np.asarray(rec.x, float)
    a = np.asarray(rec.a, float)
    if not np.array_equal(x[0], np.asarray(rec.x0, float)):
        raise ValueError(f"key {rec.key_id}: x[0] is not x0")
    sigma = _window_centres(x[:, 0])
    P = key_parameters(solver, track, sigma, ctx, p)
    lbx, ubx, lbg, ubg = bounds(rec, spec, sigma, p)
    w = np.concatenate([x.reshape(-1), a.reshape(-1)])
    return KeyProblem(rec=rec, spec=spec, solver=solver, track=track, p=p, ctx=ctx, sigma=sigma,
                      P=P, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, w=w,
                      lam_g=np.asarray(rec.lam_g, float).copy(),
                      lam_x=np.asarray(rec.lam_x, float).copy())


def constraints(prob: KeyProblem, w: np.ndarray | None = None, P: np.ndarray | None = None
                ) -> tuple[float, np.ndarray]:
    """(f, g) at w (default: the problem's point) under P (default: the rebuilt one)."""
    f, g = prob.solver.funcs["nlp"](prob.w if w is None else w, prob.P if P is None else P)
    return float(f), np.asarray(g).ravel()


def stationarity(prob: KeyProblem) -> np.ndarray:
    """r = grad f + J_g' lam_g + lam_x [nw] at the problem's primal-dual point (CasADi's
    convention: L = f + lam_g' g + lam_x' w, f = -J)."""
    _, gf = prob.solver.funcs["grad_f"](prob.w, prob.P)
    _, jac = prob.solver.funcs["jac_g"](prob.w, prob.P)
    return np.asarray(gf).ravel() + np.asarray(jac.sparse().T @ prob.lam_g).ravel() + prob.lam_x


@dataclass(frozen=True)
class Gate0:
    """Gate 0 at the problem's point: stationarity on the free variables and feasibility."""

    r_free: float             # max |r| over the free variables
    r_at: str                 # the variable where it is attained
    n_free: int
    defect: float             # max |g| over the dynamics rows (x_{t+1} - Phi_t)
    corridor_violation: float  # max(g_corridor, 0)
    slip_violation: float     # max(|g_slip| - bound, 0)
    J_rebuilt: float          # -f at the point under the rebuilt P
    J_stored: float

    @property
    def ok(self) -> bool:
        return self.r_free <= GATE0_TOL


def gate0(prob: KeyProblem) -> Gate0:
    T = prob.T
    r = stationarity(prob)
    free = prob.free
    i = int(np.argmax(np.where(free, np.abs(r), -1.0)))
    f, g = constraints(prob)
    corr = g[NX * T: 10 * T]
    slip = g[10 * T:]
    return Gate0(r_free=float(np.max(np.abs(r[free]))), r_at=variable_name(i, T),
                 n_free=int(free.sum()), defect=float(np.max(np.abs(g[: NX * T]))),
                 corridor_violation=float(max(np.max(corr - prob.ubg[NX * T: 10 * T]), 0.0)),
                 slip_violation=float(max(np.max(np.abs(slip) - prob.ubg[10 * T:]), 0.0)),
                 J_rebuilt=-f, J_stored=float(prob.rec.J))


@dataclass
class Repolish:
    """One REFINE_IPOPT solve from the stored point on the rebuilt P, and how far it moved."""

    problem: KeyProblem       # the re-polished primal-dual point (same P and bounds)
    status: str
    iterations: int
    seconds: float
    J: float
    dJ_rel: float             # (J - J_stored) / |J_stored|
    max_dx: float             # max |x - x_stored| over all states and steps
    max_dx_state: np.ndarray  # [8] per state component
    max_dx_step: np.ndarray   # [8] step of each component's maximum
    dx_step: np.ndarray       # [T+1] max |x - x_stored| over the components, per step (the
                              # movement within an outcome's horizon; late steps move most)
    max_da: float
    defect_before: float      # rebuilt defects at the stored point
    defect_after: float
    block_ok: bool            # every RK stage inside its kappa block (window.py:736-743)

    @property
    def ok(self) -> bool:
        return self.status == "Solve_Succeeded" and self.block_ok

    @property
    def flagged(self) -> bool:
        """Moved more than the design's tolerances (kept either way)."""
        return bool(self.max_dx > MOVE_X_TOL or abs(self.dJ_rel) > MOVE_J_RTOL)


def _block_ok(prob: KeyProblem) -> bool:
    spec, T, s = prob.spec, prob.T, prob.solver
    (Ck, _, _), j0 = coefficient_windows(prob.track, prob.sigma[:-1], spec)
    q_t = np.tile(np.array(track_scalars(prob.track))[:, None], (1, T))
    lo, hi = s.block_offsets(prob.x[:T].T, prob.a.T, Ck, j0[None, :], q_t,
                             np.tile(prob.ctx[:, None], (1, T)))
    rng = block_offset_range(spec.block)
    return bool(rng[0] <= float(np.min(np.asarray(lo))) and float(np.max(np.asarray(hi))) <= rng[1])


def repolish(prob: KeyProblem) -> Repolish:
    """One ``REFINE_IPOPT`` solve (the solver's ``refine_solver``: its IPOPT options with
    REFINE_IPOPT's warm start, mu_init 1e-9, constr_viol_tol 1e-10, no bound relaxation)
    warm-started from the stored primal and dual, on the rebuilt P and the same bounds."""
    T = prob.T
    rs = prob.solver.refine_solver()
    assert REFINE_IPOPT["warm_start_init_point"] == "yes"
    defect_before = float(np.max(np.abs(constraints(prob)[1][: NX * T])))
    t0 = time.perf_counter()
    sol = rs(x0=prob.w, p=prob.P, lbx=prob.lbx, ubx=prob.ubx, lbg=prob.lbg, ubg=prob.ubg,
             lam_g0=prob.lam_g, lam_x0=prob.lam_x)
    secs = time.perf_counter() - t0
    st = rs.stats()
    new = copy.copy(prob)
    new.w = np.asarray(sol["x"]).ravel()
    new.lam_g = np.asarray(sol["lam_g"]).ravel()
    new.lam_x = np.asarray(sol["lam_x"]).ravel()
    J = -float(sol["f"])
    dx = np.abs(new.x - prob.x)
    J0 = float(prob.rec.J)
    return Repolish(problem=new, status=str(st["return_status"]), iterations=int(st["iter_count"]),
                    seconds=secs, J=J, dJ_rel=(J - J0) / abs(J0), max_dx=float(dx.max()),
                    max_dx_state=dx.max(axis=0), max_dx_step=dx.argmax(axis=0),
                    dx_step=dx.max(axis=1),
                    max_da=float(np.max(np.abs(new.a - prob.a))), defect_before=defect_before,
                    defect_after=float(np.max(np.abs(constraints(new)[1][: NX * T]))),
                    block_ok=_block_ok(new))


def summary(rp: Repolish) -> dict:
    """The re-polish record fixed at the tag (design Section 9), JSON-ready."""
    d = {f.name: getattr(rp, f.name) for f in dataclasses.fields(rp) if f.name != "problem"}
    d["max_dx_state"] = [float(v) for v in rp.max_dx_state]
    d["max_dx_step"] = [int(v) for v in rp.max_dx_step]
    d["dx_step"] = [float(v) for v in rp.dx_step]
    d.update(key_id=rp.problem.rec.key_id, kind=rp.problem.rec.kind, flagged=rp.flagged, ok=rp.ok)
    return d
