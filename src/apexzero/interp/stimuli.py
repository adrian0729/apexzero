"""Stimuli: single-corner strips, real-corner transplants, crash ladders and tier D conditions.

Protocol 3 (construction, transplants, tiers D and E, grid), 1.4 (layout) and 4 (cells).

Every stimulus is an open strip of N_KNOTS = 2560 knots at DS = 2 m with its corner start at
S_C = 1000 m, so the whole design stacks into one bank without padding, and an approach
(v0, D) only sets the initial state s0 = S_C - D - u_k. Strips are built from knot samples
with ``generator.strip_arrays`` (the C2 spline interpolates them; no splprep), and the
solvers get the same spline through ``generator.to_track``. The reference line is the
centreline: ``refine_reference`` is never called on a synthetic stimulus.

Data contract (shared by every module):
- ``StimulusBank``: float64 knot samples kappa / w_left / w_right [B, N] plus one
  ``Stimulus`` record per row; row index == stim_id == track_id in rollouts. The float32
  TrackArrays the agents and keys use are rebuilt from the samples by ``bank.arrays()``.
- ``Cell``: one stimulus at one approach (or one real initial state). A mirror pair is two
  cells (left, right) with the same ``pair_id``; ``pair_id`` is also the noise id of
  common random numbers, and ``Cell.mirrored`` marks the member whose steering noise is
  negated (protocol 4). ``Cell.runners`` lists the agents that run a cell (empty: all).
- ``Landmarks``: s_c0, s_a0, s_apex, s_a1, s_c1 in strip coordinates (m).

Transplants use the Track the real-lap evaluation uses,
``refine_reference(load_tumftm(name))`` (scripts/train_dr.py, diagnose_crashes.py), and
always call ``lockbox_guard(name)`` first. Knot samples are copied index for index from the
source (real tracks have ds within 3e-4 of 2 m), so transplant kappa equals the source
samples exactly over the excised range, and the arc length is rescaled by 2 / ds_source
(< 0.03%). The 20 m kappa tapers of a cut lie just outside the excised range, so the road
under a real initial state is the real road.

Real-road conditions (tier D, the E1 ladders and E2) share one rule: the first 100 m from
the initial state (the lead-in) is always the real road with real width, so every version
of a corner starts from the same consistent state; every substitution (straight road,
constant width, scaled or swapped corner, perturbation) blends in over 20 m on the
substituted side, and the inside-width cap is applied (and flagged) on modified knots only.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.interp import io
from apexzero.interp.constants import (
    A1_D,
    A1_V0_KPH,
    COMPLETE_AFTER_EXIT,
    D_AFTER,
    D_CONST_WIDTH,
    D_DOSES,
    D_GAP_CLIP,
    D_LEAD_IN,
    D_REGION_MARGIN,
    D_SWAP_LOG_RATIO,
    D_SWAP_MIN_LOG_RATIO,
    D_WIDTH_DOSES,
    DS,
    DT,
    E2_MIN_DISTANCE,
    E3_R,
    E3_W_IN,
    E_CUT_AFTER,
    E_CUT_BEFORE,
    E_FAST_D,
    E_FAST_V0_KPH,
    F6_KAPPA_BAND,
    F6_KAPPA_NOISE,
    F6_WOBBLE_AMPL,
    F6_WOBBLE_WAVELENGTH,
    G_V0_KPH,
    G_WIDTHS_EXTRA,
    INSIDE_CAP_MARGIN,
    N_KNOTS,
    N_PHASES,
    N_PHASES_P7,
    O_R_TIGHT,
    O_R_WIDE,
    O_V_KPH,
    P7_R,
    PILOT_N_KNOTS,
    PILOT_R,
    PILOT_S0,
    PILOT_S_ENTRY,
    PILOT_STEPS,
    PILOT_THETA_DEG,
    PILOT_V0,
    PILOT_WIDTH,
    R_180_MAX,
    R_BRAKING,
    R_CONTROLS,
    R_NOCAP,
    RAMP_MIN,
    RECOVERY_D,
    RECOVERY_R,
    RECOVERY_V0_KPH,
    ROOT_SEED,
    S_C,
    S_TRUNC,
    STREAM_OFFSET,
    STREAMS,
    TAPER,
    THETA_A_DEG,
    TIER_B_D,
    TIER_B_R,
    TIER_B_THETA_DEG,
    TIER_B_V0_KPH,
    TIER_C_45_R,
    TIER_C_135_R,
    TIER_C_WIDTH_R,
    TIER_C_WIDTHS,
    TWIN_TOL,
    V_TOP_KPH,
    VALIDATION_IMPOSED_RT,
    VALIDATION_RT,
    W_LONG,
    WIDTH_CORE,
    kph,
)
from apexzero.interp.support import (
    GRAMMARS,
    LTypTable,
    SupportTables,
    knn_log_density,
    l_typ,
    stimulus_v_peak_batch,
    support_features,
    table_features,
    theta_class,
    v_typ,
)
from apexzero.tracks.corners import R_DEV, R_TEST, Corner, segment, smooth
from apexzero.tracks.generator import strip_arrays, to_track
from apexzero.tracks.spline import packed_coeffs, spline_np
from apexzero.tracks.track import Track
from apexzero.vehicle.sim import TrackArrays, kappa_at

# ------------------------------------------------------------------------- records


@dataclass(frozen=True)
class CornerSpec:
    """One synthetic corner (protocol 3.1). ``l_r`` is resolved at construction
    (``corner_spec``), so a spec fully determines its strip."""

    R: float                          # nominal radius (m)
    theta_deg: float                  # |turning angle| (deg)
    direction: int                    # +1 left, -1 right
    width: float = WIDTH_CORE         # total width w (m), split equally between the sides
    l_r: float = RAMP_MIN             # ramp length (m): max(10, min(l_typ(R), theta R)) or 10
    ramp_kind: str = "typical"        # 'typical' | 'abrupt' (E3, l_r = 10) | 'pilot' | 'twin'
    no_cap: bool = False              # tier A no-cap variant: constant w/2, no inside cap
    w_in_corner: float | None = None  # tier E3 inside half-width at the corner, w_in^c (m)
    s_c: float = S_C                  # corner start s_c0 (m); shift_corner moves it

    @property
    def theta(self) -> float:
        """|turning angle| (rad)."""
        return float(np.radians(self.theta_deg))

    @property
    def kappa_peak(self) -> float:
        """min(1/R, theta / l_r) (1/m); theta / 10 for the generator's 10 m triangle."""
        return min(1.0 / self.R, self.theta / self.l_r)

    @property
    def L_p(self) -> float:
        """Plateau length theta / kappa_peak - l_r >= 0 (m)."""
        return max(self.theta / self.kappa_peak - self.l_r, 0.0)

    @property
    def L_c(self) -> float:
        """Corner length L_p + 2 l_r (m)."""
        return self.L_p + 2.0 * self.l_r

    @property
    def is_triangle(self) -> bool:
        return self.L_p <= 1e-12

    def landmarks(self) -> Landmarks:
        """Protocol 3.1: s_c0, s_c0 + l_r, s_c0 + L_c / 2, s_c0 + L_c - l_r, s_c0 + L_c."""
        return Landmarks(self.s_c, self.s_c + self.l_r, self.s_c + 0.5 * self.L_c,
                         self.s_c + self.L_c - self.l_r, self.s_c + self.L_c)


@dataclass(frozen=True)
class Landmarks:
    """Fixed points of a stimulus in strip coordinates (m), protocol 3.1. For real corners
    (transplants, tier D): s_c0 / s_c1 from ``corners.segment`` (start, end + ds),
    s_apex = (s_c0 + s_c1) / 2, and s_a0 / s_a1 the first / last knot in [s_c0, s_c1] with
    |kappa_smooth| >= 0.9 max|kappa_smooth| (equal to the apex for a single peak)."""

    s_c0: float    # corner start (start of ramp in)
    s_a0: float    # arc start (s_c0 + l_r)
    s_apex: float  # geometric apex (s_c0 + L_c / 2)
    s_a1: float    # arc end (s_c1 - l_r)
    s_c1: float    # corner exit (s_c0 + L_c)

    @property
    def L_c(self) -> float:
        return self.s_c1 - self.s_c0


@dataclass(frozen=True)
class Geometry:
    """Realised geometry of a stimulus, recorded for every row (protocol 3.1)."""

    R_eff: float          # m, 1 / max|kappa_at| every 0.05 m over [s_c0 - 20, s_c1 + 20]
    theta_eff_deg: float  # deg, signed integral of kappa_at over the same window
    onset_rate: float     # 1/m^2, kappa_peak / l_r (nan for real corners)
    R_seg: float          # m, corners.segment r_min of the corner holding s_apex
    theta_seg_deg: float  # deg, |segment angle|
    L_seg: float          # m, segment length
    ringing: float        # 1/m, max |kappa_at| of the opposite sign in the window (>= 0)
    s_kmax: float         # m, position of max |kappa_at|
    n_segments: int       # corners segment() finds in [s_c0 - 50, s_c1 + 50] (1 expected)
    cap_binds: bool       # the inside-width cap reduced some knot's width
    w_in_min: float       # m, min inside half-width over [s_c0, s_c1]


@dataclass(frozen=True)
class Stimulus:
    """One row of the bank. ``source`` records the provenance of real-road rows:
    {'track', 's_c_real', 's_e_real', 'cut': [a, b], 'version' or 'condition', 'lambda',
    'corner_index', 'swap_with', 'ds_source', 'n_source', 'j_offset', 'exact': [i_lo, i_hi],
    'mirrored', 'twin_l_r', ...}; empty for synthetic rows."""

    stim_id: int                     # row in the bank == track_id in rollouts
    name: str                        # unique, e.g. 'syn/R13.1/T180/W12/typical/L'
    kind: str                        # 'synthetic' | 'straight' | 'transplant' | 'hockenheim'
                                     # | 'twin' | 'pilot'
    tiers: tuple[str, ...]           # tiers using it: 'A','B','C','D','E1','E2','E3','G','O','V'
    direction: int                   # +1 left, -1 right, 0 straight
    mirror_id: int                   # stim_id of its mirror image in the bank (-1 if none)
    landmarks: Landmarks
    geometry: Geometry
    spec: CornerSpec | None = None
    source: dict = field(default_factory=dict)
    flags: tuple[str, ...] = ()      # 'cap_binds', 'no_cap', 'off_distribution', 'abrupt', ...


@dataclass(frozen=True)
class Profile:
    """Knot samples of one strip (float64, [n]) before it enters a bank."""

    kappa: np.ndarray     # 1/m
    w_left: np.ndarray    # m
    w_right: np.ndarray   # m
    landmarks: Landmarks
    capped: np.ndarray    # bool [n], the inside cap reduced this knot's width


@dataclass(frozen=True)
class RealState:
    """An agent's real deterministic lap state, used as a transplant's initial state
    (protocol 3.2): the first recorded state at or after the target arc length on the lap
    being reproduced. Produced by ``agents.real_lap_states``."""

    agent: str                    # whose lap ('G1', 'G2', 'S1', ...)
    track: str                    # R_DEV circuit
    lap: int                      # 1-based lap of the deterministic two-lap run
    s_real: float                 # m, arc length of the state on the circuit (mod L)
    x: tuple[float, ...]          # [8] state in circuit coordinates
    prev_action: tuple[float, float]
    accel: tuple[float, float]


@dataclass
class Cell:
    """One stimulus at one approach or one real initial state (protocol 3.5, 4)."""

    cell_id: int
    stim_id: int
    pair_id: int                  # shared by a mirror pair's two cells; the noise id
    tiers: tuple[str, ...]
    approach: str                 # 'A1' | 'A2' | 'B' | 'E-fast' | 'REC' | 'G' | 'real' | 'V'
                                  # | 'imposed' | 'pilot'
    v0: float                     # m/s (real-state cells: x0[3])
    D: float | None               # m, s_c0 - s0 at u = 0; None until the A2 layout sets it
    W: int | None = None          # steps (400 | 800), set by the key layout
    P: float | None = None        # m, phase cycle v_sight * dt, set by the key layout
    n_phases: int = N_PHASES      # 4, or 16 on P7 cells
    x0: tuple[float, ...] | None = None   # explicit [8] state in strip coords (real states)
    prev_action0: tuple[float, float] = (0.0, 0.0)
    accel0: tuple[float, float] = (0.0, 0.0)
    state_source: str = ""        # e.g. 'G2:lap1@s_c-800' for real states
    null_cell: int = -1           # tier G cell with the same v0, width and W
    flags: tuple[str, ...] = ()   # 'recovery', 'imposed', 'braking', 'flat', 'no_key', ...
    mirrored: bool = False        # the pair member whose steering noise is negated
    runners: tuple[str, ...] = ()  # agents that run this cell; () = every agent

    def phase_offsets(self) -> np.ndarray:
        """u_k = (k / n_phases) P (m), k = 0..n_phases-1 (protocol 4)."""
        if self.P is None:
            raise ValueError(f"cell {self.cell_id} has no phase cycle P yet (key layout)")
        return np.arange(self.n_phases) * self.P / self.n_phases

    def initial_state(self, k: int = 0) -> np.ndarray:
        """x0 [8] float32 at phase k: synthetic cells [S_C - D - u_k, 0, 0, v0, 0, 0, 0, 0]
        (stimuli place every corner at S_C), real-state cells their state with s shifted by
        -u_k. Built in float32; keys cast this to float64 so both start bit-identically.
        Phase 0 (u = 0) needs no P, so the nominal key can be solved before the layout."""
        u = 0.0 if k == 0 else float(self.phase_offsets()[k])
        if self.x0 is not None:
            x = np.asarray(self.x0, np.float64).copy()
            x[0] -= u
            return x.astype(np.float32)
        if self.D is None:
            raise ValueError(f"cell {self.cell_id} has no approach distance D yet (A2 layout)")
        return np.array([S_C - self.D - u, 0.0, 0.0, self.v0, 0.0, 0.0, 0.0, 0.0], np.float32)


class StimulusBank:
    """The stimulus set: float64 knot samples [B, N] and one Stimulus per row."""

    def __init__(self, kappa: np.ndarray, w_left: np.ndarray, w_right: np.ndarray,
                 stimuli: list[Stimulus], ds: float = DS):
        self.kappa = np.asarray(kappa, np.float64)
        self.w_left = np.asarray(w_left, np.float64)
        self.w_right = np.asarray(w_right, np.float64)
        self.stimuli, self.ds = list(stimuli), float(ds)
        if self.kappa.shape != self.w_left.shape or self.kappa.shape != self.w_right.shape:
            raise ValueError("kappa and widths must have the same [B, N] shape")
        if len(self.stimuli) != self.kappa.shape[0]:
            raise ValueError(f"{len(self.stimuli)} records for {self.kappa.shape[0]} rows")
        for i, st in enumerate(self.stimuli):
            if st.stim_id != i:
                raise ValueError(f"stimulus {st.name} has stim_id {st.stim_id} at row {i}")
        self._names = {s.name: s for s in self.stimuli}
        if len(self._names) != len(self.stimuli):
            raise ValueError("stimulus names are not unique")
        self._arrays: dict[str, TrackArrays] = {}

    def __len__(self) -> int:
        return len(self.stimuli)

    def arrays(self, dtype=jnp.float32) -> TrackArrays:
        """Stacked TrackArrays [B, ...]: ``jax.vmap(strip_arrays)`` of the samples cast to
        ``dtype`` (float32 for agents, as in training). Cached per dtype."""
        name = jnp.dtype(dtype).name
        if name not in self._arrays:
            ds = self.ds
            fn = jax.jit(jax.vmap(lambda k, wl, wr: strip_arrays(k, wl, wr, ds)))
            self._arrays[name] = fn(jnp.asarray(self.kappa, dtype), jnp.asarray(self.w_left, dtype),
                                    jnp.asarray(self.w_right, dtype))
        return self._arrays[name]

    def row_arrays(self, stim_id: int, dtype=jnp.float32) -> TrackArrays:
        """Unbatched TrackArrays of one row (the same values as ``arrays()[stim_id]``)."""
        return strip_arrays(jnp.asarray(self.kappa[stim_id], dtype),
                            jnp.asarray(self.w_left[stim_id], dtype),
                            jnp.asarray(self.w_right[stim_id], dtype), self.ds)

    def track(self, stim_id: int) -> Track:
        """NumPy Track of one row, ``to_track`` of its float32 TrackArrays (what the TO-OCP,
        s-OCP, replay and QSS use), named after the stimulus."""
        return to_track(self.row_arrays(stim_id, jnp.float32), self.stimuli[stim_id].name)

    def by_name(self, name: str) -> Stimulus:
        return self._names[name]

    def save(self, out_dir: Path) -> None:
        """``bank.npz`` (kappa, w_left, w_right) and ``stimuli.json``."""
        out_dir = Path(out_dir)
        arrays = {"kappa": self.kappa, "w_left": self.w_left, "w_right": self.w_right,
                  "ds": np.asarray(self.ds)}
        io._atomic(out_dir / "bank.npz", lambda fh: np.savez(fh, **arrays))
        io.write_json(out_dir / "stimuli.json", {"stimuli": self.stimuli,
                                                 "content_hash": self.content_hash()})

    @classmethod
    def load(cls, out_dir: Path) -> StimulusBank:
        out_dir = Path(out_dir)
        with np.load(out_dir / "bank.npz", allow_pickle=False) as d:
            k, wl, wr, ds = d["kappa"], d["w_left"], d["w_right"], float(d["ds"])
        recs = io.read_json(out_dir / "stimuli.json")["stimuli"]
        return cls(k, wl, wr, [stimulus_from_dict(r) for r in recs], ds)

    def content_hash(self) -> str:
        """sha256 of the samples and the stimulus records (the manifest's stimulus hash
        combines it with the cells')."""
        h_arr = io.sha256_arrays(self.kappa, self.w_left, self.w_right, np.asarray(self.ds))
        return io.sha256_json({"arrays": h_arr, "stimuli": self.stimuli})


# ------------------------------------------------------------------------- (de)serialising


def _num(x) -> float:
    """JSON number back to float (null -> nan, "inf" / "-inf" -> inf)."""
    return float("nan") if x is None else float(x)


def _dc_from_dict(cls, d: dict):
    kw = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        t = str(f.type)
        if t.startswith("tuple"):
            if v is None:
                kw[f.name] = None
            elif "float" in t:
                kw[f.name] = tuple(_num(x) for x in v)
            else:
                kw[f.name] = tuple(v)
        elif t.startswith("float") and "None" not in t:
            kw[f.name] = _num(v)
        elif t.startswith("float | None") or t == "float | None":
            kw[f.name] = None if v is None else _num(v)
        else:
            kw[f.name] = v
    return cls(**kw)


def stimulus_from_dict(d: dict) -> Stimulus:
    spec = None if d.get("spec") is None else _dc_from_dict(CornerSpec, d["spec"])
    return Stimulus(
        stim_id=int(d["stim_id"]), name=d["name"], kind=d["kind"], tiers=tuple(d["tiers"]),
        direction=int(d["direction"]), mirror_id=int(d["mirror_id"]),
        landmarks=_dc_from_dict(Landmarks, d["landmarks"]),
        geometry=_dc_from_dict(Geometry, d["geometry"]), spec=spec,
        source=dict(d.get("source", {})), flags=tuple(d.get("flags", ())))


def save_cells(path: Path, cells: list[Cell]) -> None:
    io.write_json(path, {"cells": cells})


def load_cells(path: Path) -> list[Cell]:
    recs = io.read_json(path)["cells"]
    out = []
    for r in recs:
        c = _dc_from_dict(Cell, r)
        c.v0 = _num(r["v0"])
        c.D = None if r.get("D") is None else _num(r["D"])
        c.P = None if r.get("P") is None else _num(r["P"])
        c.W = None if r.get("W") is None else int(r["W"])
        c.x0 = None if r.get("x0") is None else tuple(_num(x) for x in r["x0"])
        c.prev_action0 = tuple(_num(x) for x in r.get("prev_action0", (0.0, 0.0)))
        c.accel0 = tuple(_num(x) for x in r.get("accel0", (0.0, 0.0)))
        out.append(c)
    return out


# ------------------------------------------------------------------------- synthetic


def corner_spec(R: float, theta_deg: float, direction: int, width: float = WIDTH_CORE,
                ramp: str = "typical", l_typ_table: LTypTable | None = None,
                no_cap: bool = False, w_in_corner: float | None = None,
                s_c: float = S_C) -> CornerSpec:
    """Resolve l_r: 'typical' -> max(10, min(l_typ(R), theta R)) (needs ``l_typ_table``),
    'abrupt' and 'pilot' -> 10 m (a theta R < 10 corner is then the generator's 10 m
    triangle). An E3 ``w_in_corner`` equal to w / 2 is no narrowing and is dropped, so the
    strip deduplicates with its tier A twin. Protocol 3.1."""
    theta = math.radians(theta_deg)
    if ramp == "typical":
        if l_typ_table is None:
            raise ValueError("a typical ramp needs the frozen l_typ table")
        l_r = max(RAMP_MIN, min(l_typ(R, l_typ_table), theta * R))
    elif ramp in ("abrupt", "pilot"):
        l_r = RAMP_MIN
    else:
        raise ValueError(f"unknown ramp kind {ramp!r}")
    if w_in_corner is not None and abs(w_in_corner - 0.5 * width) < 1e-12:
        w_in_corner = None
    return CornerSpec(float(R), float(theta_deg), int(direction), float(width), float(l_r), ramp,
                      bool(no_cap), None if w_in_corner is None else float(w_in_corner), float(s_c))


def trapezoid(u: np.ndarray, l_r: float, L_p: float) -> np.ndarray:
    """prof(u): u / l_r on [0, l_r), 1 on [l_r, l_r + L_p), (2 l_r + L_p - u) / l_r after,
    clipped at 0, and 0 before the corner (protocol 3.1)."""
    u = np.asarray(u, np.float64)
    return np.where(u < 0.0, 0.0, np.where(u < l_r, u / l_r, np.where(
        u < l_r + L_p, 1.0, np.clip((2.0 * l_r + L_p - u) / l_r, 0.0, 1.0))))


def apply_inside_cap(kappa: np.ndarray, w_left: np.ndarray, w_right: np.ndarray,
                     margin: float = INSIDE_CAP_MARGIN, mask: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The generator's cap: w_in <= 1 + margin / max(|kappa|, 1e-6) on the inside of each
    turn (``mask``: only these knots). Returns (w_left, w_right, capped [n] bool)."""
    kappa = np.asarray(kappa, np.float64)
    cap = 1.0 + margin / np.maximum(np.abs(kappa), 1e-6)
    m = np.ones(kappa.shape, bool) if mask is None else np.asarray(mask, bool)
    wl = np.where(m & (kappa > 0), np.minimum(w_left, cap), w_left)
    wr = np.where(m & (kappa < 0), np.minimum(w_right, cap), w_right)
    return wl, wr, (wl < w_left) | (wr < w_right)


def corner_profile(spec: CornerSpec, n_points: int = N_KNOTS, ds: float = DS) -> Profile:
    """Knot samples of a synthetic corner (protocol 3.1): u = s - s_c, prof(u) = u / l_r on
    [0, l_r), 1 on [l_r, l_r + L_p), (2 l_r + L_p - u) / l_r after, clipped at 0;
    kappa = direction * kappa_peak * prof at s_i = i ds. Widths w/2 per side with the inside
    cap min(w_in, 1 + 0.6 / |kappa|) at every knot (not for no_cap); E3 narrows the inside to
    w_in(s) = w/2 - (w/2 - w_in^c) prof(u) before the cap. Exact float64; kappa matches the
    formula to 1e-12."""
    s = np.arange(n_points) * ds
    prof = trapezoid(s - spec.s_c, spec.l_r, spec.L_p)
    kappa = spec.direction * spec.kappa_peak * prof
    half = 0.5 * spec.width
    wl = np.full(n_points, half)
    wr = np.full(n_points, half)
    if spec.w_in_corner is not None:
        inside = half - (half - spec.w_in_corner) * prof
        if spec.direction > 0:
            wl = inside
        else:
            wr = inside
    if spec.no_cap:
        capped = np.zeros(n_points, bool)
    else:
        wl, wr, capped = apply_inside_cap(kappa, wl, wr)
    return Profile(kappa, wl, wr, spec.landmarks(), capped)


def straight_profile(width: float = WIDTH_CORE, n_points: int = N_KNOTS) -> Profile:
    """Constant-width straight (tier G). Landmarks are the virtual corner at S_C (L_c = 0),
    so null runs align with corner runs by distance travelled."""
    half = np.full(n_points, 0.5 * width)
    return Profile(np.zeros(n_points), half.copy(), half.copy(),
                   Landmarks(S_C, S_C, S_C, S_C, S_C), np.zeros(n_points, bool))


def shift_corner(spec: CornerSpec, delta: float, n_points: int = N_KNOTS,
                 ds: float = DS) -> Profile:
    """The same corner translated by ``delta`` metres (any real value), resampled exactly
    from the trapezoid formula (P7 on-policy aliasing). Landmarks move with it."""
    return corner_profile(replace(spec, s_c=spec.s_c + float(delta)), n_points, ds)


def mirror(p: Profile) -> Profile:
    """Mirror image: kappa -> -kappa, w_left <-> w_right; landmarks unchanged. Exact."""
    return Profile(-p.kappa, p.w_right.copy(), p.w_left.copy(), p.landmarks, p.capped.copy())


def mirror_state(x: np.ndarray) -> np.ndarray:
    """[s, n, xi, vx, vy, r, delta, pedal] -> [s, -n, -xi, vx, -vy, -r, -delta, pedal]."""
    x = np.array(x, copy=True)
    x[..., [1, 2, 4, 5, 6]] *= -1.0
    return x


def mirror_aux(prev_action: np.ndarray, accel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Steer of prev_action and ay of accel negated."""
    pa = np.array(prev_action, copy=True)
    ac = np.array(accel, copy=True)
    pa[..., 0] *= -1.0
    ac[..., 1] *= -1.0
    return pa, ac


def _segment_corner(track: Track, lm: Landmarks) -> tuple[Corner | None, int]:
    """The segment() corner holding s_apex (else the largest one overlapping the corner
    window), and the number of corners overlapping [s_c0 - 50, s_c1 + 50]."""
    cs = segment(track).corners
    ds = track.ds
    lo, hi = lm.s_c0 - 50.0, lm.s_c1 + 50.0
    near = [c for c in cs if c.s_start <= hi and c.s_end + ds >= lo]
    hold = [c for c in cs if c.s_start <= lm.s_apex <= c.s_end + ds]
    if hold:
        return hold[0], len(near)
    win = [c for c in cs if c.s_start <= lm.s_c1 + 20.0 and c.s_end + ds >= lm.s_c0 - 20.0]
    if win:
        return max(win, key=lambda c: abs(c.angle)), len(near)
    return None, len(near)


_kappa_at_jit = jax.jit(kappa_at)


def _dense_kappa(s: np.ndarray, tr: TrackArrays) -> np.ndarray:
    """``sim.kappa_at`` on a grid, padded to a power-of-two length so one compile serves
    every window."""
    n = len(s)
    m = max(1024, 1 << (n - 1).bit_length())
    pad = np.concatenate([s, np.full(m - n, s[-1])])
    return np.asarray(_kappa_at_jit(jnp.asarray(pad, tr.ds.dtype), tr), np.float64)[:n]


def realised_geometry(tr: TrackArrays, landmarks: Landmarks, kappa_samples: np.ndarray,
                      w_left: np.ndarray, w_right: np.ndarray, capped: np.ndarray,
                      l_r: float | None = None, kappa_peak: float | None = None,
                      direction: int | None = None) -> Geometry:
    """Realised geometry of one strip (protocol 3.1): ``sim.kappa_at`` every 0.05 m over
    [s_c0 - 20, s_c1 + 20] on ``tr`` (one unbatched TrackArrays), R_seg from
    ``corners.segment(to_track(tr))`` with its defaults. ``kappa_peak`` / ``l_r`` give the
    onset rate of a synthetic corner; ``direction`` (default: the sign of the window's
    integral) decides which side is inside and which curvature sign is ringing."""
    lm = landmarks
    s = np.arange(lm.s_c0 - 20.0, lm.s_c1 + 20.0 + 1e-9, 0.05)
    km = _dense_kappa(s, tr)
    i_max = int(np.argmax(np.abs(km)))
    kmax = float(np.abs(km[i_max]))
    theta_eff = float(np.degrees(np.trapezoid(km, s)))
    if direction is None:
        direction = int(np.sign(theta_eff))
    ringing = float(max(0.0, np.max(-direction * km))) if direction != 0 else 0.0
    corner, n_seg = _segment_corner(to_track(tr), lm)
    ds = float(tr.ds)
    i0 = max(math.floor(lm.s_c0 / ds), 0)
    i1 = min(math.ceil(lm.s_c1 / ds), len(kappa_samples) - 1)
    if direction > 0:
        inside = w_left[i0:i1 + 1]
    elif direction < 0:
        inside = w_right[i0:i1 + 1]
    else:
        inside = np.minimum(w_left, w_right)[i0:i1 + 1]
    onset = (kappa_peak / l_r) if (kappa_peak is not None and l_r) else float("nan")
    return Geometry(
        R_eff=1.0 / kmax if kmax > 0 else float("inf"), theta_eff_deg=theta_eff,
        onset_rate=float(onset),
        R_seg=float(corner.r_min) if corner else float("inf"),
        theta_seg_deg=float(abs(math.degrees(corner.angle))) if corner else 0.0,
        L_seg=float(corner.length) if corner else 0.0, ringing=ringing, s_kmax=float(s[i_max]),
        n_segments=int(n_seg), cap_binds=bool(np.any(capped)),
        w_in_min=float(np.min(inside)) if len(inside) else float("nan"))


def stimulus_track(tr_b: TrackArrays, i: int, name: str) -> Track:
    """``to_track`` of row i of a stacked bank (protocol 3.1)."""
    return to_track(jax.tree.map(lambda a: a[i], tr_b), name)


def bank_from_profiles(profiles: list[Profile], stimuli: list[Stimulus]) -> StimulusBank:
    """Stack profiles into a bank; stimuli[i].stim_id must equal i."""
    return StimulusBank(np.stack([p.kappa for p in profiles]), np.stack([p.w_left for p in profiles]),
                        np.stack([p.w_right for p in profiles]), stimuli)


def stimulus_name(spec: CornerSpec) -> str:
    """'syn/R{R:g}/T{theta:g}/W{w:g}/{ramp_kind}[/nocap][/win{w_in:g}]/{L|R}' (s_c != S_C
    appends '/dc{s_c - S_C:+.3f}')."""
    parts = ["syn", f"R{spec.R:g}", f"T{spec.theta_deg:g}", f"W{spec.width:g}", spec.ramp_kind]
    if spec.ramp_kind == "twin":
        parts.append(f"lr{spec.l_r:.4g}")
    if spec.no_cap:
        parts.append("nocap")
    if spec.w_in_corner is not None:
        parts.append(f"win{spec.w_in_corner:g}")
    parts.append("L" if spec.direction > 0 else "R")
    if abs(spec.s_c - S_C) > 1e-12:
        parts.append(f"dc{spec.s_c - S_C:+.3f}")
    return "/".join(parts)


# ------------------------------------------------------------------------- real roads

TAPER_KNOTS = round(TAPER / DS)          # 10 knots: every taper and blend
LEAD_KNOTS = round(D_LEAD_IN / DS)       # 50 knots: the real lead-in
MARGIN_KNOTS = round(D_REGION_MARGIN / DS)  # 10 knots: corner region margin
I_C = round(S_C / DS)                    # 500: strip knot of every corner start


@functools.cache
def _real_track_cached(name: str) -> Track:
    from apexzero.tracks.reference import refine_reference
    from apexzero.tracks.track import load_tumftm

    return refine_reference(load_tumftm(io.REPO / "data" / "tracks" / "tumftm" / f"{name}.csv"))


def real_track(name: str) -> Track:
    """The Track the real-lap evaluation uses: ``lockbox_guard(name)`` first, then
    ``refine_reference(load_tumftm('data/tracks/tumftm/<name>.csv'))``. R_DEV only."""
    from apexzero.viz.export import lockbox_guard

    lockbox_guard(name)
    if name not in R_DEV:
        raise ValueError(f"{name!r} is not an R_DEV circuit; stimuli are cut from R_DEV only")
    return _real_track_cached(name)


def real_corners(track: Track) -> list[Corner]:
    """``corners.segment(track)`` with its defaults (the single radius estimator)."""
    return segment(track).corners


def _guard_named(track_name: str, track: Track | None) -> Track:
    from apexzero.viz.export import lockbox_guard

    lockbox_guard(track_name)
    if track is None:
        return real_track(track_name)
    lockbox_guard(track.name)
    return track


@dataclass(frozen=True)
class Excision:
    """Map between a closed source circuit and a strip: source knot j (mod n) lands on
    strip knot i = j - j_c + i_c, with i_c = S_C / DS; s_strip = i * DS. The excised range
    [j_a, j_b] is copied exactly; the cut tapers lie outside it."""

    track: str
    n_source: int        # knots on the source circuit
    ds_source: float     # m
    j_c: int             # source knot of the corner start
    j_a: int             # first excised source knot (s_c - before), unwrapped (may be < 0)
    j_b: int             # last excised source knot (s_c + after), unwrapped

    @property
    def i_c(self) -> int:
        return I_C

    def to_strip(self, s_real: float, lap_offset: int | None = None) -> float:
        """Strip arc length of a circuit arc length (knot-index map, rescaled to DS). With
        ``lap_offset`` None the source position is unwrapped to the lap nearest the corner;
        an integer adds that many laps to s_real / ds_source explicitly."""
        j = float(s_real) / self.ds_source
        if lap_offset is None:
            j += self.n_source * round((self.j_c - j) / self.n_source)
        else:
            j += lap_offset * self.n_source
        return (j - self.j_c + self.i_c) * DS


class _Source:
    """A closed source circuit sampled at fractional, unwrapped knot positions: integer
    positions copy the samples exactly, others evaluate the periodic C2 spline."""

    def __init__(self, track: Track):
        self.track = track
        self.n = track.n_points
        self.ds = track.ds
        self._c = {k: packed_coeffs(np.asarray(getattr(track, k), float), True, self.n)
                   for k in ("kappa", "w_left", "w_right")}

    def at(self, name: str, j: np.ndarray) -> np.ndarray:
        j = np.asarray(j, float)
        ji = np.round(j)
        exact = np.abs(j - ji) < 1e-9
        out = np.asarray(getattr(self.track, name), float)[np.mod(ji.astype(np.int64), self.n)]
        if not exact.all():
            val = spline_np(self._c[name], j * self.ds, self.ds, self.n, True)
            out = np.where(exact, out, val)
        return out


def _ramp_down(i: np.ndarray, start: float, nt: int = TAPER_KNOTS) -> np.ndarray:
    """1 up to knot ``start``, linear to 0 over nt knots after it, 0 beyond."""
    return np.clip(1.0 - (i - start) / nt, 0.0, 1.0)


def _ramp_up(i: np.ndarray, end: float, nt: int = TAPER_KNOTS) -> np.ndarray:
    """0 up to nt knots before ``end``, linear to 1 at ``end``, 1 beyond."""
    return np.clip(1.0 - (end - i) / nt, 0.0, 1.0)


def _real_landmarks(kappa: np.ndarray, i_first: int, m: int) -> Landmarks:
    """D14 landmarks of a real corner occupying strip knots [i_first, i_first + m)."""
    s_c0 = i_first * DS
    s_c1 = (i_first + m) * DS
    ks = np.abs(smooth(kappa, DS, 10.0, False)[i_first:i_first + m])
    hi = np.flatnonzero(ks >= 0.9 * ks.max()) if len(ks) and ks.max() > 0 else np.array([0])
    return Landmarks(s_c0, (i_first + int(hi[0])) * DS, 0.5 * (s_c0 + s_c1),
                     (i_first + int(hi[-1])) * DS, s_c1)


class _RealRoad:
    """One real corner excised onto a strip with the tapers, lead-in and region weights the
    real-road conditions share (module docstring)."""

    def __init__(self, track_name: str, corner: Corner, before: float, after: float,
                 n_points: int = N_KNOTS, track: Track | None = None,
                 lead_start: int | None = None):
        self.track = _guard_named(track_name, track)
        self.name = track_name
        self.corner = corner
        self.src = _Source(self.track)
        self.n_points = n_points
        n, dsr = self.track.n_points, self.track.ds
        self.j_c = round(corner.s_start / dsr)
        self.m = round(corner.length / dsr)                 # knots inside the corner
        nb, na = round(before / DS), round(after / DS)
        self.exc = Excision(track_name, n, dsr, self.j_c, self.j_c - nb, self.j_c + na)
        if (na + nb + 2 * TAPER_KNOTS) >= n:
            raise ValueError(f"excision of {before + after} m does not fit on {track_name}")
        self.i = np.arange(n_points)
        self.i_a, self.i_b = I_C - nb, I_C + na
        if self.i_a - TAPER_KNOTS < 0 or self.i_b + TAPER_KNOTS >= n_points:
            raise ValueError("excision does not fit on the strip")
        self.i_last = I_C + self.m - 1                        # last knot inside the corner
        self.c0 = I_C - MARGIN_KNOTS                          # corner region [c0, c1]
        self.c1 = self.i_last + MARGIN_KNOTS
        self.lead0 = self.i_a if lead_start is None else int(lead_start)
        self.lead1 = self.lead0 + LEAD_KNOTS
        if self.lead1 + TAPER_KNOTS > self.c0:
            raise ValueError(f"lead-in [{self.lead0}, {self.lead1}] leaves no room before the "
                             f"corner region at {self.c0}")
        self.identity = self.i - I_C + self.j_c               # unwrapped source knot per strip knot

    # -- sampling
    def sample(self, src: np.ndarray, j_a: float | None = None, j_b: float | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """kappa, w_left, w_right at source positions ``src`` with the cut tapers outside
        [j_a, j_b] (default: the excision) and widths held beyond the cut."""
        j_a = self.exc.j_a if j_a is None else j_a
        j_b = self.exc.j_b if j_b is None else j_b
        cut = _ramp_up(src, j_a) * _ramp_down(src, j_b)
        held = np.clip(src, j_a, j_b)
        return (self.src.at("kappa", src) * cut, self.src.at("w_left", held),
                self.src.at("w_right", held))

    def real(self):
        return self.sample(self.identity.astype(float))

    # -- weights (1 = keep the real road)
    def w_lead(self) -> np.ndarray:
        """1 up to the end of the lead-in, blending to 0 over 20 m after it."""
        return _ramp_down(self.i, self.lead1)

    def w_before_straight(self) -> np.ndarray:
        """0 on the road between the lead-in and the corner region, 20 m blends inside it."""
        w = np.maximum(_ramp_down(self.i, self.lead1), _ramp_up(self.i, self.c0))
        return np.where((self.i > self.lead1) & (self.i < self.c0), w, 1.0)

    def w_after_straight(self) -> np.ndarray:
        """0 after the corner region, with a 20 m blend after it."""
        return _ramp_down(self.i, self.c1)

    def landmarks(self, kappa: np.ndarray) -> Landmarks:
        return _real_landmarks(kappa, I_C, self.m)

    def source_info(self, **extra) -> dict:
        d = dict(track=self.name, s_c_real=float(self.corner.s_start),
                 s_e_real=float(self.corner.s_end + self.track.ds),
                 cut=[float(self.exc.j_a * self.track.ds), float(self.exc.j_b * self.track.ds)],
                 ds_source=float(self.track.ds), n_source=int(self.track.n_points),
                 j_offset=int(self.j_c - I_C), lead_in=[self.lead0 * DS, self.lead1 * DS])
        d.update(extra)
        return d


def _const_width(rr: _RealRoad, wl: np.ndarray, wr: np.ndarray, half: float,
                 kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Width blended from real (lead-in) to ``half`` per side over 20 m after the lead-in,
    then the inside cap on the modified knots."""
    w = rr.w_lead()
    wl2 = w * wl + (1.0 - w) * half
    wr2 = w * wr + (1.0 - w) * half
    return apply_inside_cap(kappa, wl2, wr2, mask=w < 1.0)


def transplant(track_name: str, corner: Corner, before: float = E_CUT_BEFORE,
               after: float = E_CUT_AFTER, taper: float = TAPER, n_points: int = N_KNOTS,
               track: Track | None = None) -> tuple[Profile, Excision]:
    """Excise [s_c - before, s_c + after] from a real circuit (wrapping across s = 0), taper
    kappa to 0 over ``taper`` m beyond each cut, hold the cut-edge widths constant beyond the
    cut, pad with straight road so the corner start lands at S_C. Calls
    ``lockbox_guard(track_name)`` first. Protocol 3.2."""
    if abs(taper - TAPER) > 1e-12:
        raise ValueError("the protocol's tapers are 20 m")
    rr = _RealRoad(track_name, corner, before, after, n_points, track)
    k, wl, wr = rr.real()
    return Profile(k, wl, wr, rr.landmarks(k), np.zeros(n_points, bool)), rr.exc


LADDER_VERSIONS = ("real", "before_straight", "after_straight", "const_width", "isolated",
                   "twin", "mirror")
E2_VERSIONS = ("real", "isolated", "twin")


def ladder(track_name: str, corner: Corner, track: Track | None = None,
           lead_start: float | None = None, versions: tuple[str, ...] = LADDER_VERSIONS,
           twin: TwinFit | None = None) -> tuple[dict[str, Profile], Excision]:
    """The tier E1 versions of a crash corner (protocol 3.4): real road; road before the
    corner straightened; road after straightened; constant width at the corner's median
    width; isolated (all three); trapezoid twin (``twin_fit``); mirror of the real road.
    Straightening and width changes blend over 20 m on the substituted side of the corner
    region [s_c - 20, s_e + 20] and of the 100 m real lead-in that starts at ``lead_start``
    (strip m; default the excision start s_c - 800, where the initial state sits). The twin
    is the fitted trapezoid at S_C on straight road at the real apex width (equal sides,
    inside cap), behind the same lead-in. E2 uses ``versions=E2_VERSIONS``."""
    lead = None if lead_start is None else math.floor(lead_start / DS + 1e-9)
    rr = _RealRoad(track_name, corner, E_CUT_BEFORE, E_CUT_AFTER, N_KNOTS, track, lead)
    k, wl, wr = rr.real()
    lm = rr.landmarks(k)
    zeros = np.zeros(N_KNOTS, bool)
    out: dict[str, Profile] = {}
    corner_knots = slice(I_C, rr.i_last + 1)
    half_med = 0.5 * float(np.median(wl[corner_knots] + wr[corner_knots]))
    for v in versions:
        if v == "real":
            out[v] = Profile(k, wl, wr, lm, zeros)
        elif v == "mirror":
            out[v] = mirror(Profile(k, wl, wr, lm, zeros))
        elif v == "before_straight":
            out[v] = Profile(k * rr.w_before_straight(), wl, wr, lm, zeros)
        elif v == "after_straight":
            out[v] = Profile(k * rr.w_after_straight(), wl, wr, lm, zeros)
        elif v == "const_width":
            wl2, wr2, cap = _const_width(rr, wl, wr, half_med, k)
            out[v] = Profile(k, wl2, wr2, lm, cap)
        elif v == "isolated":
            k2 = k * rr.w_before_straight() * rr.w_after_straight()
            wl2, wr2, cap = _const_width(rr, wl, wr, half_med, k2)
            out[v] = Profile(k2, wl2, wr2, lm, cap)
        elif v == "twin":
            if twin is None:
                twin = twin_for(rr.track, corner)
            tp = corner_profile(twin.spec)
            w = rr.w_lead()
            k2 = w * k + tp.kappa
            wl2 = w * wl + (1.0 - w) * tp.w_left
            wr2 = w * wr + (1.0 - w) * tp.w_right
            wl2, wr2, cap = apply_inside_cap(k2, wl2, wr2, mask=w < 1.0)
            out[v] = Profile(k2, wl2, wr2, tp.landmarks, cap | tp.capped)
        else:
            raise ValueError(f"unknown ladder version {v!r}")
    return out, rr.exc


@dataclass(frozen=True)
class TwinFit:
    """Least-squares synthetic twin of a real corner (protocol 3.4)."""

    spec: CornerSpec       # fitted (R, theta, l_r) with the real apex width, inside cap
    target: dict           # {'R_seg', 'theta_deg', 'L_seg'} of the real corner
    achieved: dict         # the same measured on the twin with corners.segment
    ok: bool               # within TWIN_TOL (2%, 1 deg, 2 m)


def _twin_spec(theta: float, l_r: float, L_p: float, direction: int, width: float,
               s_c: float = S_C) -> CornerSpec:
    """Trapezoid with ramp l_r and plateau L_p turning theta (rad): R = (l_r + L_p) / theta,
    so kappa_peak = 1 / R <= theta / l_r and the plateau is exactly L_p."""
    return CornerSpec(R=(l_r + L_p) / theta, theta_deg=math.degrees(theta), direction=direction,
                      width=width, l_r=l_r, ramp_kind="twin", s_c=s_c)


def _measure_twin(spec: CornerSpec, n_points: int | None = None, float32: bool = False
                  ) -> dict[str, float]:
    """segment() of the twin's strip: the corner with the largest |angle|."""
    if n_points is None:
        n_points = math.ceil((spec.s_c + spec.L_c + 200.0) / DS) + 1
    p = corner_profile(spec, n_points)
    if float32:
        tr = to_track(strip_arrays(jnp.asarray(p.kappa, jnp.float32), jnp.asarray(p.w_left, jnp.float32),
                                   jnp.asarray(p.w_right, jnp.float32), DS))
    else:
        z = np.zeros(n_points)
        tr = Track(name="twin", s=np.arange(n_points) * DS, x=z, y=z, theta=z, kappa=p.kappa,
                   w_left=z, w_right=z, closed=False)
    cs = segment(tr).corners
    if not cs:
        return {"R_seg": float("inf"), "theta_deg": 0.0, "L_seg": 0.0}
    c = max(cs, key=lambda c: abs(c.angle))
    return {"R_seg": float(c.r_min), "theta_deg": float(abs(math.degrees(c.angle))),
            "L_seg": float(c.length)}


def _twin_residual(meas: dict, R_seg: float, theta_deg: float, L_seg: float) -> np.ndarray:
    return np.array([(meas["R_seg"] / R_seg - 1.0) / TWIN_TOL["R_seg_rel"],
                     (meas["theta_deg"] - theta_deg) / TWIN_TOL["angle_deg"],
                     (meas["L_seg"] - L_seg) / TWIN_TOL["length"]])


def _twin_match(f: float, R_seg: float, theta_deg: float, direction: int, width: float,
                s_c: float, iters: int = 12) -> tuple[CornerSpec, dict]:
    """For a ramp fraction f = l_r / (theta R) in (0, 1] (1: triangle), adjust (theta, R) by
    fixed-point iteration until segment() on the twin returns R_seg and the angle."""
    th, R = math.radians(theta_deg), R_seg
    spec = meas = None
    for _ in range(iters):
        l_r = max(f * th * R, 2.0)
        spec = _twin_spec(th, l_r, max(th * R - l_r, 0.0), direction, width, s_c)
        meas = _measure_twin(spec)
        eR, eth = meas["R_seg"] / R_seg, meas["theta_deg"] - theta_deg
        if abs(eR - 1.0) < 1e-4 and abs(eth) < 1e-3:
            break
        R = R / eR
        th = th - math.radians(eth)
    return spec, meas


def twin_fit(R_seg: float, theta_deg: float, L_seg: float, direction: int,
             apex_width: float) -> TwinFit:
    """Fit (R, theta, l_r) so that ``corners.segment`` on the twin's strip matches the real
    corner's R_seg, angle and length within 2%, 1 deg and 2 m. A trapezoid's shape is its
    ramp fraction f = l_r / (theta R) (f = 1 is the triangle): for each f, (R, theta) are
    solved so R_seg and the angle match (fixed point), and f is chosen by a grid and a
    bisection so the segment length is nearest the real one (the least-squares solution
    with the radius and angle residuals driven to ~0). A real corner longer than the
    triangle with its R_seg and angle (long low-curvature shoulders) cannot be matched by
    any trapezoid; the fit then returns the closest twin with ``ok`` False and says so in
    ``achieved['note']``. ``ramp_kind`` = 'twin'; the width is the real apex width split
    equally, inside cap applied (corner_profile)."""
    sc = 200.0   # short strips while fitting; segment() is local
    th0 = math.radians(theta_deg)
    f_min = min(1.0, 2.0 / (th0 * R_seg))
    grid = np.unique(np.r_[np.geomspace(f_min, 1.0, 24), 1.0])
    tried = []
    for f in grid:
        spec, meas = _twin_match(float(f), R_seg, theta_deg, direction, apex_width, sc)
        tried.append((float(f), spec, meas))
    Ls = np.array([t[2]["L_seg"] for t in tried])
    k = int(np.argmin(np.abs(Ls - L_seg)))
    # Bisection between the grid neighbours that bracket the target length, if any.
    for lo, hi in ((k - 1, k), (k, k + 1)):
        if 0 <= lo and hi < len(tried) and (Ls[lo] - L_seg) * (Ls[hi] - L_seg) < 0:
            a, b = tried[lo][0], tried[hi][0]
            La = Ls[lo]
            for _ in range(14):
                mid = math.sqrt(a * b)
                spec, meas = _twin_match(mid, R_seg, theta_deg, direction, apex_width, sc)
                tried.append((mid, spec, meas))
                if (meas["L_seg"] - L_seg) * (La - L_seg) > 0:
                    a, La = mid, meas["L_seg"]
                else:
                    b = mid
    def score(t):
        r = _twin_residual(t[2], R_seg, theta_deg, L_seg)
        return float(10.0 * r[0] ** 2 + 10.0 * r[1] ** 2 + r[2] ** 2)

    f, spec, _ = min(tried, key=score)
    spec = replace(spec, s_c=S_C)
    achieved = _measure_twin(spec, N_KNOTS, float32=True)
    r = _twin_residual(achieved, R_seg, theta_deg, L_seg)
    ok = bool(np.all(np.abs(r) <= 1.0 + 1e-9))
    L_tri = float(max(Ls))
    note = ""
    if not ok and L_seg > L_tri + TWIN_TOL["length"]:
        note = (f"L_seg {L_seg:.0f} m exceeds the longest trapezoid with this R_seg and angle "
                f"(triangle, {L_tri:.0f} m)")
    elif not ok and L_seg < float(min(Ls)) - TWIN_TOL["length"]:
        note = f"L_seg {L_seg:.0f} m is shorter than the 2 m-ramp twin ({min(Ls):.0f} m)"
    achieved = dict(achieved, f=f, note=note, L_triangle=L_tri)
    target = {"R_seg": float(R_seg), "theta_deg": float(theta_deg), "L_seg": float(L_seg)}
    return TwinFit(spec, target, achieved, ok)


def twin_for(track: Track, corner: Corner) -> TwinFit:
    """``twin_fit`` of one real corner: its segment() R_seg, angle and length, its sign, and
    the real total width at the apex knot (the corner's midpoint)."""
    from apexzero.viz.export import lockbox_guard

    lockbox_guard(track.name)
    n, ds = track.n_points, track.ds
    j_apex = (round(corner.s_start / ds) + round(corner.length / ds) // 2) % n
    width = float(track.w_left[j_apex] + track.w_right[j_apex])
    return twin_fit(corner.r_min, abs(math.degrees(corner.angle)), corner.length, corner.sign,
                    width)


@dataclass(frozen=True)
class CrashPair:
    """One (agent, real crash) unit of P8, from the deterministic real-lap crash tables
    (docs/findings/<run>-crashes.md: lap and crash s). A unit carries no radius label:
    its corner is the ``corners.segment`` corner containing the crash (``crash_corner``,
    D15) and its regime comes from that corner's R_seg against the agent's frozen q1 / q10
    (ruling R2, ``analysis.p8_set``)."""

    agent: str
    track: str
    lap: int
    s_crash: float       # m on the circuit


CRASH_PAIRS = (
    CrashPair("G1", "YasMarina", 1, 2832.0),
    CrashPair("G1", "Norisring", 1, 480.0),
    CrashPair("G1", "Hockenheim", 1, 265.0),
    CrashPair("G1", "Melbourne", 1, 377.0),
    CrashPair("G1", "Sochi", 1, 2058.0),
    CrashPair("G1", "Zandvoort", 1, 369.0),
    CrashPair("G1", "BrandsHatch", 1, 2750.0),
    CrashPair("G1", "Oschersleben", 1, 474.0),
    CrashPair("G2", "Norisring", 1, 910.0),
    CrashPair("G2", "YasMarina", 1, 1369.0),
    CrashPair("G2", "Melbourne", 1, 2562.0),
    CrashPair("G2", "Catalunya", 1, 4182.0),
    CrashPair("G2", "Zandvoort", 1, 3228.0),
    CrashPair("G2", "Sochi", 1, 2060.0),
    CrashPair("G2", "Oschersleben", 1, 469.0),
)
EXTRA_CRASH_PAIRS_FILE = "crash_pairs_extra.json"   # under io.STIMULI_DIR (G3's own crashes)
_CRASH_PAIR_FIELDS = ("agent", "track", "lap", "s_crash")


def crash_pairs(path: Path | None = None) -> tuple[CrashPair, ...]:
    """P8's units: ``CRASH_PAIRS`` (G1's and G2's, from the crash docs) plus G3's own R_DEV
    crash corners, which protocol 2.4 P8 adds "by the same construction rule" from G3's
    standard R_DEV evaluation. Those come from G3's deterministic real-lap runs
    (``crash_pairs_from_laps``, written by ``rollouts --real-laps`` to
    stimuli/crash_pairs_extra.json); until then only the named pairs exist. A radius label
    in an older file is ignored (R2)."""
    f = io.STIMULI_DIR / EXTRA_CRASH_PAIRS_FILE if path is None else Path(path)
    extra = ()
    if f.exists():
        extra = tuple(CrashPair(**{k: d[k] for k in _CRASH_PAIR_FIELDS})
                      for d in io.read_json(f)["pairs"])
    return CRASH_PAIRS + tuple(p for p in extra if p not in CRASH_PAIRS)


def crash_pairs_from_laps(agent: str, laps: dict) -> list[CrashPair]:
    """An agent's crash units from its deterministic real laps ({circuit: lap} with
    ``x`` [K, 8] (s unwrapped), ``valid``, ``term_step``, ``length``; ``agents.RealLap``):
    each R_DEV circuit where the run terminated gives one pair at the crash s (the terminal
    state's s modulo the lap length), on that lap. Its corner and regime follow from the
    crash s (``crash_corner``, R2)."""
    out = []
    for name, lap in sorted(laps.items()):
        if name not in R_DEV or int(lap.term_step) < 0:
            continue
        k = min(int(lap.term_step) + 1, len(lap.x) - 1)
        s_unwrapped = float(lap.x[k, 0])
        L = float(lap.length)
        lap_index = int(s_unwrapped // L) + 1
        s = s_unwrapped % L
        out.append(CrashPair(agent, name, lap_index, round(s, 1)))
    return out


def crash_unit_R_seg(pair: CrashPair) -> float:
    """R_seg (m) of a crash unit's corner: the ``corners.segment`` corner containing the
    crash point (D15). The unit's regime is this radius against the crashing agent's
    frozen q1 / q10 (ruling R2: one estimator for every crash corner)."""
    return float(crash_corner(real_track(pair.track), pair.s_crash).r_min)


def _contains(c: Corner, s: float, ds: float, length: float) -> bool:
    a, e = c.s_start, c.s_end + ds
    if a <= e:
        return a <= s <= e
    return s >= a or s <= e - length


def crash_corner(track: Track, s_crash: float) -> Corner:
    """The ``corners.segment`` corner containing s_crash (s_start <= s_crash <= s_end + ds,
    wrapping), else the last corner starting before it (cyclically). This is the crash
    corner of ruling R2 (D15); its R_seg sets the unit's regime. A tighter neighbour inside
    the ladder's [s_c - 800, s_c + 400] window still acts through the context rungs."""
    from apexzero.viz.export import lockbox_guard

    lockbox_guard(track.name)
    cs = real_corners(track)
    s = float(s_crash) % track.length
    for c in cs:
        if _contains(c, s, track.ds, track.length):
            return c
    before = [c for c in cs if c.s_start <= s]
    return max(before, key=lambda c: c.s_start) if before else max(cs, key=lambda c: c.s_start)


def crash_corner_key(track_name: str, corner: Corner) -> str:
    """'E1/<circuit>/<s_start:.0f>', the key of a crash corner's states and strips."""
    return f"E1/{track_name}/{corner.s_start:.0f}"


def distinct_crash_corners() -> dict[str, tuple[str, Corner, list[CrashPair]]]:
    """The distinct crash corners of CRASH_PAIRS (13 expected), keyed like
    ``crash_corner_key``: (circuit, Corner, pairs crashing there)."""
    out: dict[str, tuple[str, Corner, list[CrashPair]]] = {}
    for p in crash_pairs():
        c = crash_corner(real_track(p.track), p.s_crash)
        key = crash_corner_key(p.track, c)
        if key not in out:
            out[key] = (p.track, c, [])
        out[key][2].append(p)
    return out


def _cyclic_gap(a: Corner, b: Corner, ds: float, length: float) -> float:
    """Distance (m) between two corners' intervals on a closed circuit (0 if they overlap)."""
    def iv(c):
        return c.s_start, c.s_start + c.length
    (a0, a1), (b0, b1) = iv(a), iv(b)
    best = float("inf")
    for shift in (-length, 0.0, length):
        lo, hi = b0 + shift, b1 + shift
        best = min(best, max(0.0, lo - a1, a0 - hi))
    return best


BRAKING_QSS_DROP = 0.05   # E2 braking corner: QSS v_apex <= (1 - 0.05) QSS v_peak


def matched_corners(rdev: dict[str, Track], crash: list[tuple[str, Corner]],
                    reachable: dict[str, float]) -> list[tuple[str, Corner, int]]:
    """E2 (protocol 3.4): for each crash corner, the R_DEV braking corner where no generalist
    crashed, in the same theta class, nearest in log R_seg, >= 1 km from any crash corner on
    its circuit, and reached alive by G2's real lap; matches are distinct. ``rdev``: circuit
    -> real Track; ``crash``: every crash corner (circuit, Corner) of G1 and G2;
    ``reachable``: circuit -> the largest unwrapped s G2's two-lap run reached alive (inf if
    it finished). A candidate qualifies when s_end + 250 m is reached on lap 2 (its state
    then comes from lap 2, D18) or else on lap 1. "Braking" uses the circuit's QSS profile:
    v_apex <= 0.95 v_peak. Crash corners are matched tightest first; exact distance ties
    are broken with ``io.stream_rng('e2_ties')``. Returns (circuit, Corner, lap) per crash
    corner, in the order of ``crash``."""
    from apexzero.oracle.qss import GG, speed_profile
    from apexzero.vehicle.params import F1_2025
    from apexzero.viz.export import lockbox_guard

    for name, tr in rdev.items():
        lockbox_guard(name)
        lockbox_guard(tr.name)
        if name not in R_DEV:
            raise ValueError(f"{name!r} is not an R_DEV circuit")
    gg = GG.from_params(F1_2025)
    rng = io.stream_rng("e2_ties")
    cands = []
    for name in sorted(rdev):
        tr = rdev[name]
        v = speed_profile(tr.kappa, tr.ds, gg, closed=True)
        cs = real_corners(tr)
        L = tr.length
        crashes_here = [c for (t, c) in crash if t == name]
        reach = reachable.get(name, float("inf"))
        for j, c in enumerate(cs):
            n, ds = tr.n_points, tr.ds
            a = round(c.s_start / ds)
            idx = (a + np.arange(round(c.length / ds))) % n
            pe = round(cs[j - 1].s_end / ds)
            gk = (a - pe - 1) % n
            prev = (pe + 1 + np.arange(gk + 1)) % n
            v_peak, v_apex = float(v[prev].max()), float(v[idx].min())
            if v_apex > (1.0 - BRAKING_QSS_DROP) * v_peak:
                continue
            if any(abs(c.s_start - x.s_start) < 1e-9 for x in crashes_here):
                continue
            if any(_cyclic_gap(c, x, ds, L) < E2_MIN_DISTANCE for x in crashes_here):
                continue
            end = c.s_start + c.length + COMPLETE_AFTER_EXIT
            if L + end <= reach:
                lap = 2
            elif end <= reach:
                lap = 1
            else:
                continue
            cands.append((name, c, lap, rng.random()))
    order = sorted(range(len(crash)), key=lambda k: crash[k][1].r_min)
    used: set[tuple[str, float]] = set()
    out: list[tuple[str, Corner, int] | None] = [None] * len(crash)
    for k in order:
        _, cc = crash[k]
        cls = theta_class(math.degrees(abs(cc.angle)))
        pool = [(abs(math.log(c.r_min / cc.r_min)), tie, name, c, lap) for name, c, lap, tie in cands
                if (name, c.s_start) not in used and theta_class(math.degrees(abs(c.angle))) == cls]
        if not pool:
            raise ValueError(f"no E2 candidate for crash corner {crash[k][0]} @ {cc.s_start}")
        _, _, name, c, lap = min(pool, key=lambda t: (t[0], t[1]))
        used.add((name, c.s_start))
        out[k] = (name, c, lap)
    return [o for o in out if o is not None]


# ------------------------------------------------------------------------- tier D

D_CONDITIONS = (
    # F1 factorial: before x after x width; (a) = real/real/real, iso = straight/straight/const
    "F1/real/real/real", "F1/real/real/const", "F1/real/straight/real", "F1/real/straight/const",
    "F1/straight/real/real", "F1/straight/real/const", "F1/straight/straight/real",
    "F1/straight/straight/const",
    # F2 dose: corner scaled by lambda in place (road after shifted), and whole-strip width
    "F2/lambda0.8", "F2/lambda0.9", "F2/lambda0.95", "F2/lambda1.05", "F2/lambda1.1",
    "F2/lambda1.25", "F2/width-0.5", "F2/width+0.5",
    # F4 mirror (from the mirrored state), F5 swaps, F6 irrelevant perturbations
    "F4/mirror", "F5/swapA", "F5/swapB",
    "F6/wobble0.3", "F6/wobble0.6", "F6/knoise0.5e-3", "F6/knoise1e-3",
)
COND_A = "F1/real/real/real"
COND_ISO = "F1/straight/straight/const"
assert len(D_CONDITIONS) == 23, "protocol 3.3: 8 + 8 + 1 + 2 + 4 conditions"
D_TRACK = "Hockenheim"


@dataclass(frozen=True)
class DCorner:
    """One tier D corner: its segment() corner, D_gap and the strip extent (protocol 3.3)."""

    index: int
    corner: Corner
    gap: float           # m, straight before it (segment() gaps, cyclic)
    D_gap: float         # m, clip(gap, 100, 600)

    @property
    def before(self) -> float:
        """s_c - s_start = D_gap + 100 m."""
        return self.D_gap + D_LEAD_IN

    @property
    def after(self) -> float:
        """s_end_strip - s_c = L_seg + 1200 m."""
        return self.corner.length + D_AFTER


def hockenheim_corners(track: Track | None = None) -> list[DCorner]:
    """The 20 corners of Hockenheim (``corners.segment``) with their D_gap."""
    track = _guard_named(D_TRACK, track)
    seg = segment(track)
    cs = seg.corners
    n, ds = track.n_points, track.ds
    out = []
    for i, c in enumerate(cs):
        pe = round(cs[i - 1].s_end / ds)
        gap = ((round(c.s_start / ds) - pe - 1) % n) * ds
        out.append(DCorner(i, c, float(gap), float(np.clip(gap, *D_GAP_CLIP))))
    return out


def _f6_rng(i: int, cond: str) -> np.random.Generator:
    """Per (corner, condition) generator inside the 'f6' stream."""
    return np.random.default_rng([ROOT_SEED, STREAM_OFFSET + STREAMS["f6"], i,
                                  D_CONDITIONS.index(cond)])


def _kappa_noise(rng: np.random.Generator, i: np.ndarray, start: int, end: int,
                 level: float) -> np.ndarray:
    """Band-limited curvature noise (wavelengths 80-200 m) on strip knots (start, end]:
    random Fourier coefficients on the region's frequency grid, windowed by 20 m ramps,
    mean removed under the window (zero net heading) and scaled to peak |noise| = level."""
    L = (end - start) * DS
    m = np.arange(1, int(L / F6_KAPPA_BAND[0]) + 2)
    f = m / L
    f = f[(f >= 1.0 / F6_KAPPA_BAND[1]) & (f <= 1.0 / F6_KAPPA_BAND[0])]
    if len(f) == 0:
        f = np.array([2.0 / (F6_KAPPA_BAND[0] + F6_KAPPA_BAND[1])])
    a, b = rng.normal(size=len(f)), rng.normal(size=len(f))
    s = (i - start) * DS
    raw = (a[:, None] * np.cos(2 * np.pi * f[:, None] * s) + b[:, None] * np.sin(2 * np.pi * f[:, None] * s)).sum(0)
    w = _ramp_up(i, start + TAPER_KNOTS) * _ramp_down(i, end - TAPER_KNOTS)
    w = np.where((i > start) & (i < end), w, 0.0)
    c = np.sum(w * raw) / np.sum(w)
    noise = w * (raw - c)
    return level * noise / np.max(np.abs(noise))


def _scaled_corner(rr: _RealRoad, lam: float):
    """F2 dose: the corner region [c0, c1] scaled by lam about c0 (kappa(s) -> kappa(s /
    lam) / lam), the road after shifted by (lam - 1)(c1 - c0), 20 m blends at both ends."""
    i = rr.i.astype(float)
    c0, c1 = rr.c0, rr.c1
    Lr = c1 - c0
    end = c0 + lam * Lr
    ident = i - I_C + rr.j_c
    scaled = (c0 - I_C + rr.j_c) + (i - c0) / lam
    after = ident - (lam - 1.0) * Lr
    j_b = rr.exc.j_b
    k_id, wl_id, wr_id = rr.sample(ident)
    k_sc, wl_sc, wr_sc = rr.sample(scaled)
    k_sc = k_sc / lam
    k_af, wl_af, wr_af = rr.sample(after, j_b=j_b)
    t_in = np.clip((i - c0) / TAPER_KNOTS, 0.0, 1.0)
    t_out = np.clip((i - (end - TAPER_KNOTS)) / TAPER_KNOTS, 0.0, 1.0)
    region = (i >= c0) & (i <= end)

    def mix(a_id, a_sc, a_af):
        inner = (1.0 - t_out) * ((1.0 - t_in) * a_id + t_in * a_sc) + t_out * a_af
        return np.where(i < c0, a_id, np.where(region, inner, a_af))

    return mix(k_id, k_sc, k_af), mix(wl_id, wl_sc, wl_af), mix(wr_id, wr_sc, wr_af), region


def _swapped_corner(rr: _RealRoad, rj: _RealRoad):
    """F5 swap A: corner i's approach up to its region start c0, then corner j's road from
    its own region start (20 m blend), straightened after j's region."""
    i = rr.i.astype(float)
    ident = i - I_C + rr.j_c
    jmap = i - I_C + rj.j_c
    k_i, wl_i, wr_i = rr.sample(ident)
    k_j, wl_j, wr_j = rj.sample(jmap)
    t = np.clip((i - rr.c0) / TAPER_KNOTS, 0.0, 1.0)
    w_after = _ramp_down(rr.i, rj.c1)

    def mix(a, b):
        return (1.0 - t) * a + t * b

    return mix(k_i, k_j) * w_after, mix(wl_i, wl_j), mix(wr_i, wr_j)


def hockenheim_conditions(i: int, track: Track | None = None, swap_j: int | None = None,
                          f6_scale: float = 1.0) -> tuple[dict[str, Profile], Excision]:
    """The 23 tier D conditions of Hockenheim corner i (protocol 3.3). The strip runs from
    s_start = s_c - (D_gap + 100) to s_e + 1200 with D_gap = clip(gap to the previous corner,
    100, 600), placed so s_c lands at S_C; the first 100 m is always the real road with real
    width. Regions: before [s_start + 100, s_c - 20], corner [s_c - 20, s_e + 20], after
    [s_e + 20, s_e + 1200]; every substitution blends over 20 m on the substituted side. F1
    width 'const' is 12 m outside the lead-in. F2: kappa(s) -> kappa(s / lambda) / lambda
    over the corner region scaled about its start (corner start moves by 20 (lambda - 1) m,
    the road after shifts), and the width doses change the total width by +-0.5 m outside
    the lead-in; the inside cap is applied to modified knots and flagged. F5 swap A inserts
    corner ``swap_j`` (``swap_partner``; omitted while it is None) and straightens the road
    after it; swap B straightens the corner region and the road after it. F6 wobble moves
    both edges outward and inward together (w_left and w_right += A sin(2 pi s / 60 + phi));
    the curvature noise is ``_kappa_noise``; both outside the lead-in, seeded from the 'f6'
    stream per (corner, condition). ``f6_scale`` scales both F6 amplitudes (0.5: protocol
    7's one halving when the K control lands in class 1); condition names keep the nominal
    amplitudes and the rows record the scale."""
    track = _guard_named(D_TRACK, track)
    dcs = hockenheim_corners(track)
    dc = dcs[i]
    rr = _RealRoad(D_TRACK, dc.corner, dc.before, dc.after, N_KNOTS, track)
    k, wl, wr = rr.real()
    lm = rr.landmarks(k)
    zeros = np.zeros(N_KNOTS, bool)
    out: dict[str, Profile] = {}
    w_lead = rr.w_lead()
    mod = w_lead < 1.0
    for before in ("real", "straight"):
        for after in ("real", "straight"):
            for width in ("real", "const"):
                kk = k.copy()
                if before == "straight":
                    kk = kk * rr.w_before_straight()
                if after == "straight":
                    kk = kk * rr.w_after_straight()
                if width == "const":
                    wl2, wr2, cap = _const_width(rr, wl, wr, 0.5 * D_CONST_WIDTH, kk)
                else:
                    wl2, wr2, cap = wl, wr, zeros
                out[f"F1/{before}/{after}/{width}"] = Profile(kk, wl2, wr2, lm, cap)
    c0m = rr.c0 * DS
    for lam in D_DOSES:
        kk, wl2, wr2, region = _scaled_corner(rr, lam)
        wl2, wr2, cap = apply_inside_cap(kk, wl2, wr2, mask=region)
        lm_l = Landmarks(*(c0m + lam * (x - c0m) for x in (lm.s_c0, lm.s_a0, lm.s_apex, lm.s_a1,
                                                             lm.s_c1)))
        out[f"F2/lambda{lam:g}"] = Profile(kk, wl2, wr2, lm_l, cap)
    for dw in D_WIDTH_DOSES:
        wl2 = wl + 0.5 * dw * (1.0 - w_lead)
        wr2 = wr + 0.5 * dw * (1.0 - w_lead)
        wl2, wr2, cap = apply_inside_cap(k, wl2, wr2, mask=mod)
        out[f"F2/width{dw:+g}"] = Profile(k, wl2, wr2, lm, cap)
    out["F4/mirror"] = mirror(out[COND_A])
    if swap_j is not None:
        rj = _RealRoad(D_TRACK, dcs[swap_j].corner, dc.before, dcs[swap_j].after, N_KNOTS, track,
                       rr.lead0)
        kk, wl2, wr2 = _swapped_corner(rr, rj)
        wl2, wr2, cap = apply_inside_cap(kk, wl2, wr2, mask=rr.i >= rr.c0)
        out["F5/swapA"] = Profile(kk, wl2, wr2, _real_landmarks(kk, I_C, rj.m), cap)
    out["F5/swapB"] = Profile(k * _ramp_down(rr.i, rr.c0), wl, wr, lm, zeros)
    for amp in F6_WOBBLE_AMPL:
        cond = f"F6/wobble{amp:g}"
        phi = _f6_rng(i, cond).uniform(0.0, 2.0 * np.pi)
        wob = f6_scale * amp * np.sin(2.0 * np.pi * rr.i * DS / F6_WOBBLE_WAVELENGTH + phi) \
            * (1.0 - w_lead)
        wl2, wr2, cap = apply_inside_cap(k, wl + wob, wr + wob, mask=mod)
        out[cond] = Profile(k, wl2, wr2, lm, cap)
    for level, tag in zip(F6_KAPPA_NOISE, ("0.5e-3", "1e-3")):
        cond = f"F6/knoise{tag}"
        kk = k + _kappa_noise(_f6_rng(i, cond), rr.i, rr.lead1, rr.i_b, f6_scale * level)
        wl2, wr2, cap = apply_inside_cap(kk, wl, wr, mask=mod)
        out[cond] = Profile(kk, wl2, wr2, lm, cap)
    return out, rr.exc


def swap_partner(i: int, v_min_star: dict[int, float], braking: set[int]) -> int | None:
    """F5 swap A partner j of corner i: the braking corner whose |log(v_i* / v_j*)| is
    closest to 0.35 and at least 0.15 (min-time v_min* of condition (a)); None if none.
    Ties go to the lower index."""
    if i not in v_min_star:
        return None
    best = None
    for j in sorted(braking):
        if j == i or j not in v_min_star:
            continue
        r = abs(math.log(v_min_star[i] / v_min_star[j]))
        if r < D_SWAP_MIN_LOG_RATIO:
            continue
        d = abs(r - D_SWAP_LOG_RATIO)
        if best is None or d < best[0] - 1e-12:
            best = (d, j)
    return None if best is None else best[1]


# ------------------------------------------------------------------------- the design


@dataclass
class _Row:
    """A bank row while the design is being assembled."""

    name: str
    profile: Profile
    kind: str
    direction: int
    spec: CornerSpec | None = None
    tiers: set = field(default_factory=set)
    source: dict = field(default_factory=dict)
    flags: set = field(default_factory=set)
    mirror_name: str | None = None
    l_r: float | None = None
    kappa_peak: float | None = None


@dataclass
class _CellReq:
    """A cell (or mirror pair of cells) requested by one or more tiers."""

    names: tuple[str, ...]        # (L, R) for a corner pair, (name,) for a straight
    approach: str
    v0: float | None              # m/s; None: A2, from v_typ once R_seg is known
    D: float | None
    tiers: set = field(default_factory=set)
    flags: set = field(default_factory=set)
    n_phases: int = N_PHASES


def _spec_flags(spec: CornerSpec) -> set:
    f = set()
    if spec.no_cap:
        f |= {"no_cap", "off_distribution"}
    if spec.ramp_kind == "abrupt":
        f.add("abrupt")
    if spec.w_in_corner is not None:
        f.add("narrowed")
    if spec.is_triangle:
        f.add("triangle")
    if spec.theta * spec.R < RAMP_MIN:
        f.add("generator_triangle")
    return f


class _Design:
    """Accumulates rows and cell requests, deduplicating by name and by (strip, approach)."""

    def __init__(self, support: SupportTables | None):
        self.support = support
        self.rows: dict[str, _Row] = {}
        self.reqs: dict[tuple, _CellReq] = {}

    def corner(self, R: float, theta: float, tier: str, width: float = WIDTH_CORE,
               ramp: str = "typical", no_cap: bool = False, w_in: float | None = None
               ) -> tuple[str, str]:
        lt = None if self.support is None else self.support.l_typ
        names = []
        for d in (1, -1):
            spec = corner_spec(R, theta, d, width, ramp, lt, no_cap, w_in)
            name = stimulus_name(spec)
            if name not in self.rows:
                self.rows[name] = _Row(name, corner_profile(spec), "synthetic", d, spec,
                                       flags=_spec_flags(spec), l_r=spec.l_r,
                                       kappa_peak=spec.kappa_peak)
            self.rows[name].tiers.add(tier)
            names.append(name)
        self.rows[names[0]].mirror_name, self.rows[names[1]].mirror_name = names[1], names[0]
        return names[0], names[1]

    def straight(self, width: float, tier: str = "G") -> str:
        name = f"straight/W{width:g}"
        if name not in self.rows:
            self.rows[name] = _Row(name, straight_profile(width), "straight", 0)
        self.rows[name].tiers.add(tier)
        return name

    def cell(self, names: tuple[str, ...], approach: str, v0: float | None, D: float | None,
             tier: str, flags: tuple[str, ...] = (), n_phases: int = N_PHASES) -> None:
        key = (names, approach if approach != "B" or (v0, D) != (kph(A1_V0_KPH), A1_D) else "A1",
               None if v0 is None else round(v0, 9), D)
        if approach == "B" and key[1] == "A1":
            approach = "A1"
        req = self.reqs.get(key)
        if req is None:
            req = self.reqs[key] = _CellReq(names, approach, v0, D)
        req.tiers.add(tier)
        req.flags |= set(flags)
        req.n_phases = max(req.n_phases, n_phases)


def _geometry_rows(rows: list[_Row], tr_b: TrackArrays) -> list[Geometry]:
    out = []
    for k, r in enumerate(rows):
        tr = jax.tree.map(lambda a, k=k: a[k], tr_b)
        p = r.profile
        out.append(realised_geometry(tr, p.landmarks, p.kappa, p.w_left, p.w_right, p.capped,
                                     r.l_r, r.kappa_peak, r.direction))
    return out


def _assemble(rows: list[_Row], start_id: int = 0) -> tuple[list[Stimulus], TrackArrays]:
    """Stimulus records (with realised geometry) for new rows; ids start at ``start_id``."""
    k = np.stack([r.profile.kappa for r in rows])
    wl = np.stack([r.profile.w_left for r in rows])
    wr = np.stack([r.profile.w_right for r in rows])
    tr_b = jax.jit(jax.vmap(lambda a, b, c: strip_arrays(a, b, c, DS)))(
        jnp.asarray(k, jnp.float32), jnp.asarray(wl, jnp.float32), jnp.asarray(wr, jnp.float32))
    geoms = _geometry_rows(rows, tr_b)
    ids = {r.name: start_id + i for i, r in enumerate(rows)}
    stims = []
    for i, (r, g) in enumerate(zip(rows, geoms)):
        flags = set(r.flags)
        if g.cap_binds:
            flags.add("cap_binds")
        stims.append(Stimulus(
            stim_id=start_id + i, name=r.name, kind=r.kind, tiers=tuple(sorted(r.tiers)),
            direction=r.direction, mirror_id=ids.get(r.mirror_name, -1) if r.mirror_name else -1,
            landmarks=r.profile.landmarks, geometry=g, spec=r.spec, source=r.source,
            flags=tuple(sorted(flags))))
    return stims, tr_b


def build_design(support: SupportTables) -> tuple[StimulusBank, list[Cell]]:
    """Every synthetic stimulus and cell (protocol 3.5): tiers A (incl. no-cap), B (incl.
    recovery), C, E3, G, validation cells and imposed cells, both directions, deduplicated
    (a strip used by several tiers is one row; its cells list every tier: tier B's (310 km/h,
    600 m) cells are the A1 cells, E3's (6.0 m, typical, A1) cells are tier A's). A2 cells
    get v0 = v_typ(R_seg, theta_seg) and D = None (the key layout sets it); W and P are set
    by the key layout. Tier O needs no rows of its own beyond ``tier_o_specs``."""
    d = _Design(support)
    a1 = (kph(A1_V0_KPH), A1_D)
    # Tier A: 20 geometries at 90 deg, 15 at 180 deg (R <= 55.9), A1 and A2; no-cap at A2.
    for theta in THETA_A_DEG:
        radii = R_BRAKING + R_CONTROLS if theta == 90.0 else tuple(r for r in R_BRAKING if r <= R_180_MAX)
        for R in radii:
            names = d.corner(R, theta, "A")
            p7 = R in P7_R
            d.cell(names, "A1", *a1, "A", ("p7",) if p7 else (), N_PHASES_P7 if p7 else N_PHASES)
            d.cell(names, "A2", None, None, "A")
    for theta in THETA_A_DEG:
        for R in R_NOCAP:
            d.cell(d.corner(R, theta, "A", no_cap=True), "A2", None, None, "A")
    # Tier B: crash map and recovery.
    for R in TIER_B_R:
        for theta in TIER_B_THETA_DEG:
            names = d.corner(R, theta, "B")
            for v0 in TIER_B_V0_KPH:
                for D in TIER_B_D:
                    d.cell(names, "B", kph(v0), D, "B")
    for R in RECOVERY_R:
        for theta in TIER_B_THETA_DEG:
            names = d.corner(R, theta, "B")
            for v0 in RECOVERY_V0_KPH:
                d.cell(names, "REC", kph(v0), RECOVERY_D, "B", ("recovery",))
    # Tier C: theta and width at A2.
    for R in TIER_C_45_R:
        d.cell(d.corner(R, 45.0, "C"), "A2", None, None, "C")
    for R in TIER_C_135_R:
        d.cell(d.corner(R, 135.0, "C"), "A2", None, None, "C")
    for R in TIER_C_WIDTH_R:
        for theta in THETA_A_DEG:
            for w in TIER_C_WIDTHS:
                d.cell(d.corner(R, theta, "C", width=w), "A2", None, None, "C")
    # Tier E3: crash shapes at A1 and E-fast.
    for R in E3_R:
        for w_in in E3_W_IN:
            for ramp in ("abrupt", "typical"):
                names = d.corner(R, 90.0, "E3", ramp=ramp, w_in=w_in)
                d.cell(names, "A1", *a1, "E3")
                d.cell(names, "E-fast", kph(E_FAST_V0_KPH), E_FAST_D, "E3")
    # Validation (V1-V11) and imposed cells.
    for R, theta in VALIDATION_RT:
        names = d.corner(R, theta, "V")
        d.cell(names, "A1", *a1, "V")
        d.cell(names, "A2", None, None, "V")
    for R, theta in VALIDATION_IMPOSED_RT:
        d.cell(d.corner(R, theta, "V"), "imposed", kph(A1_V0_KPH), RECOVERY_D, "V", ("imposed",))

    rows = list(d.rows.values())
    stims, _ = _assemble(rows)
    by_name = {s.name: s for s in stims}

    # Cells: corners first (L, R per pair), then the null straights tier C needs.
    cells: list[Cell] = []
    pair = 0

    def a2_v0(s: Stimulus) -> float:
        return v_typ(s.geometry.R_seg, s.geometry.theta_seg_deg, support.v_typ)

    c_widths: dict[float, set] = {}
    for req in d.reqs.values():
        sL = by_name[req.names[0]]
        v0 = a2_v0(sL) if req.v0 is None else req.v0
        if "C" in req.tiers and sL.spec is not None and sL.spec.width != WIDTH_CORE:
            c_widths.setdefault(sL.spec.width, set()).add(round(v0 * 3.6, 6))
        for k, name in enumerate(req.names):
            s = by_name[name]
            cells.append(Cell(
                cell_id=len(cells), stim_id=s.stim_id, pair_id=pair,
                tiers=tuple(sorted(req.tiers)), approach=req.approach, v0=float(v0),
                D=req.D, n_phases=req.n_phases,
                flags=tuple(sorted(set(req.flags) | set(s.flags))), mirrored=k == 1))
        pair += 1

    # Tier G null straights (one strip per width; cells per (v0, width), s0 = 400 m).
    g_rows: list[_Row] = []
    g_specs = [(WIDTH_CORE, v) for v in G_V0_KPH]
    for w in G_WIDTHS_EXTRA:
        g_specs += [(w, v) for v in sorted(c_widths.get(w, ()))]
    widths_needed = sorted({w for w, _ in g_specs})
    for w in widths_needed:
        g_rows.append(_Row(f"straight/W{w:g}", straight_profile(w), "straight", 0, tiers={"G"}))
    g_stims, _ = _assemble(g_rows, start_id=len(stims))
    stims += g_stims
    g_id = {s.name: s.stim_id for s in g_stims}
    null_of: dict[tuple[float, float], int] = {}
    for w, v in g_specs:
        cell = Cell(cell_id=len(cells), stim_id=g_id[f"straight/W{w:g}"], pair_id=pair,
                    tiers=("G",), approach="G", v0=kph(v), D=S_C - 400.0)
        null_of[(w, round(v, 6))] = cell.cell_id
        cells.append(cell)
        pair += 1
    for c in cells:
        s = stims[c.stim_id]
        if s.spec is not None:
            c.null_cell = null_of.get((s.spec.width, round(c.v0 * 3.6, 6)), -1)

    bank = StimulusBank(np.stack([r.profile.kappa for r in rows + g_rows]),
                        np.stack([r.profile.w_left for r in rows + g_rows]),
                        np.stack([r.profile.w_right for r in rows + g_rows]), stims)
    return bank, cells


def tier_o_specs(support: SupportTables) -> list[tuple[CornerSpec, float]]:
    """(spec, v in m/s) of the 52 tier O scans: R {9.4, 11.7} x {90, 180} and R {14.6, 28.6,
    55.9} x {45, 90, 180} at v in {150, 200, 250, 300} km/h, left-hand, typical ramps,
    width 12."""
    geoms = [(R, t) for R in O_R_TIGHT for t in (90.0, 180.0)]
    geoms += [(R, t) for R in O_R_WIDE for t in (45.0, 90.0, 180.0)]
    return [(corner_spec(R, t, 1, WIDTH_CORE, "typical", support.l_typ), kph(v))
            for R, t in geoms for v in O_V_KPH]


def pilot_minigrid() -> tuple[StimulusBank, list[Cell]]:
    """The v1 pilot mini-grid in the pilot's own layout (step 2 replication, protocol 1.1):
    R {9, 11, 13, 16, 20, 30} x {90, 180} deg x {L, R}, width 11 as 5.5 / 5.5 with no cap,
    10 m ramps, corner start 800 m on a 1280-knot strip, x0 = [100, 0, 0, 80, 0, 0, 0, 0],
    500 deterministic steps. Its own small bank (N = 1280), not part of the design."""
    rows: list[_Row] = []
    for R in PILOT_R:
        for theta in PILOT_THETA_DEG:
            names = []
            for d in (1, -1):
                spec = corner_spec(R, theta, d, PILOT_WIDTH, "pilot", no_cap=True,
                                   s_c=PILOT_S_ENTRY)
                name = f"pilot/R{R:g}/T{theta:g}/{'L' if d > 0 else 'R'}"
                rows.append(_Row(name, corner_profile(spec, PILOT_N_KNOTS), "pilot", d, spec,
                                 tiers={"pilot"}, flags=_spec_flags(spec), l_r=spec.l_r,
                                 kappa_peak=spec.kappa_peak))
                names.append(name)
            rows[-2].mirror_name, rows[-1].mirror_name = names[1], names[0]
    stims, _ = _assemble(rows)
    x0 = (PILOT_S0, 0.0, 0.0, PILOT_V0, 0.0, 0.0, 0.0, 0.0)
    cells = [Cell(cell_id=i, stim_id=i, pair_id=i // 2, tiers=("pilot",), approach="pilot",
                  v0=PILOT_V0, D=PILOT_S_ENTRY - PILOT_S0, W=PILOT_STEPS, P=0.0, n_phases=1,
                  x0=x0, mirrored=i % 2 == 1) for i in range(len(rows))]
    bank = StimulusBank(np.stack([r.profile.kappa for r in rows]),
                        np.stack([r.profile.w_left for r in rows]),
                        np.stack([r.profile.w_right for r in rows]), stims)
    return bank, cells


# ------------------------------------------------------------------------- real design

RUNNERS_FROM_S1 = ("S1", "G1", "G2", "G3")   # generalists run tier D from S1's state


def _strip_state(exc: Excision, st: RealState, mirrored: bool
                 ) -> tuple[tuple[float, ...], tuple[float, float], tuple[float, float]]:
    x = np.asarray(st.x, np.float64).copy()
    x[0] = exc.to_strip(st.s_real)
    pa, ac = np.asarray(st.prev_action, np.float64), np.asarray(st.accel, np.float64)
    if mirrored:
        x = mirror_state(x)
        pa, ac = mirror_aux(pa, ac)
    return tuple(float(v) for v in x), (float(pa[0]), float(pa[1])), (float(ac[0]), float(ac[1]))


def _excision_of(track: Track, track_name: str, corner: Corner, before: float, after: float
                 ) -> Excision:
    dsr = track.ds
    j_c = round(corner.s_start / dsr)
    return Excision(track_name, track.n_points, dsr, j_c, j_c - round(before / DS),
                    j_c + round(after / DS))


_LADDER_EXACT = {   # knot ranges where kappa is the source's, per version (see _exact)
    "real": ("a", "b"), "mirror": ("a", "b"), "const_width": ("a", "b"),
    "before_straight": ("a", "lead1", "c0", "b"), "after_straight": ("a", "c1"),
    "isolated": ("a", "lead1", "c0", "c1"), "twin": ("a", "lead1"),
}
_D_EXACT = {"F1/real/real": ("a", "b"), "F1/real/straight": ("a", "c1"),
            "F1/straight/real": ("a", "lead1", "c0", "b"),
            "F1/straight/straight": ("a", "lead1", "c0", "c1"),
            "F2/lambda": ("a", "c0"), "F2/width": ("a", "b"), "F4": ("a", "b"),
            "F5": ("a", "c0"), "F6/wobble": ("a", "b"), "F6/knoise": ("a", "lead1")}


def _exact(marks: tuple[str, ...], pts: dict[str, int]) -> list[list[int]]:
    v = [pts[m] for m in marks]
    return [[v[k], v[k + 1]] for k in range(0, len(v), 2)]


def _d_exact_key(cond: str) -> tuple[str, ...]:
    for prefix, marks in _D_EXACT.items():
        if cond.startswith(prefix):
            return marks
    raise KeyError(cond)


def build_real_design(bank: StimulusBank, cells: list[Cell],
                      states: dict[str, dict[str, RealState]],
                      swap: dict[int, int] | None = None, f6_scale: float = 1.0
                      ) -> tuple[StimulusBank, list[Cell]]:
    """Append tiers D, E1 and E2 (protocol 3.3, 3.4): strips from ``hockenheim_conditions``,
    ``ladder`` and E2 matched corners; cells from the real states ({agent: {key: state}},
    keyed as ``real_state_requests`` names them: 'E1/<circuit>/<s_c>', 'E2/<circuit>/<s_c>'
    (G2's) and 'D/<i>' (each specialist's)), each mapped into strip coordinates. Every
    agent runs every ladder from every state of that corner; generalists run tier D from
    S1's state (``Cell.runners``), each specialist from its own; F4 and the E1 mirror start
    from the mirrored state and share their pair (and noise) with the real road. ``swap``
    maps a Hockenheim corner to its F5 swap A partner (``swap_partner``; rows are added
    only for corners present there). The lead-in of a ladder starts at its earliest state."""
    rows: list[_Row] = []
    new_cells: list[dict] = []          # cell kwargs, finalised once stim ids exist
    pair = max((c.pair_id for c in cells), default=-1) + 1

    def add_cell(name, st_info, approach, tiers, runners, flags, pair_id, mirrored, x0, pa, ac,
                 src):
        new_cells.append(dict(name=name, approach=approach, tiers=tiers, runners=runners,
                              flags=flags, pair_id=pair_id, mirrored=mirrored, x0=x0, pa=pa,
                              ac=ac, state_source=src))

    # E1 ladders and E2 matched corners.
    jobs = []
    for key, (tname, corner, pairs) in distinct_crash_corners().items():
        sts = [(p.agent, states.get(p.agent, {}).get(key)) for p in pairs]
        sts = [(a, s) for a, s in sts if s is not None]
        if sts:
            jobs.append(("E1", key, tname, corner, sts, LADDER_VERSIONS))
    for key, st in sorted(states.get("G2", {}).items()):
        if not key.startswith("E2/"):
            continue
        _, tname, s_c = key.split("/")
        cs = [c for c in real_corners(real_track(tname)) if abs(c.s_start - float(s_c)) < 1.0]
        if len(cs) != 1:
            raise ValueError(f"E2 state {key} matches {len(cs)} corners")
        jobs.append(("E2", key, tname, cs[0], [("G2", st)], E2_VERSIONS))
    for tier, key, tname, corner, sts, versions in jobs:
        track = real_track(tname)
        exc = _excision_of(track, tname, corner, E_CUT_BEFORE, E_CUT_AFTER)
        lead = min(exc.to_strip(s.s_real) for _, s in sts)
        twin = twin_for(track, corner)
        profs, exc = ladder(tname, corner, track, lead_start=lead, versions=versions, twin=twin)
        rr_pts = _RealRoad(tname, corner, E_CUT_BEFORE, E_CUT_AFTER, N_KNOTS, track,
                           math.floor(lead / DS + 1e-9))
        pts = dict(a=rr_pts.i_a, b=rr_pts.i_b, lead1=rr_pts.lead1, c0=rr_pts.c0, c1=rr_pts.c1)
        base = f"{tier}/{tname}/{corner.s_start:.0f}"
        # R4: a twin that matches R_seg and the angle (to < 1e-3) but is shorter than its
        # corner is flagged 'twin_length_short' and listed with its length shortfall
        # (``twin_length_shortfall``); any other miss keeps 'twin_fit_fail'.
        twin_flags = set()
        if not twin.ok:
            short = (twin_r4_match(twin.achieved["R_seg"], twin.achieved["theta_deg"], twin.target)
                     and twin.achieved["L_seg"] < twin.target["L_seg"])
            twin_flags = {"twin_length_short" if short else "twin_fit_fail"}
        for v, p in profs.items():
            src = rr_pts.source_info(version=v, exact=_exact(_LADDER_EXACT[v], pts),
                                     mirrored=v == "mirror", twin_l_r=twin.spec.l_r,
                                     twin_ok=twin.ok, twin_target=twin.target,
                                     twin_achieved=twin.achieved, key=key)
            sign = -corner.sign if v == "mirror" else corner.sign
            row = _Row(f"{base}/{v}", p, "twin" if v == "twin" else "transplant", sign,
                       twin.spec if v == "twin" else None, {tier}, src,
                       set(twin_flags) if v == "twin" else set())
            if v == "twin":
                row.l_r, row.kappa_peak = twin.spec.l_r, twin.spec.kappa_peak
            if v in ("real", "mirror"):
                row.mirror_name = f"{base}/{'mirror' if v == 'real' else 'real'}"
            rows.append(row)
        for agent, st in sts:
            pair_real = pair
            pair += 1
            start = corner.s_start - E_CUT_BEFORE < 0.0 and tier == "E1"
            for v in versions:
                mir = v == "mirror"
                x0, pa, ac = _strip_state(exc, st, mir)
                pid = pair_real if v in ("real", "mirror") else pair
                if v not in ("real", "mirror"):
                    pair += 1
                fl = {"start_state"} if start else set()
                add_cell(f"{base}/{v}", st, "real", (tier,), (), fl, pid, mir, x0, pa, ac,
                         f"{agent}:lap{st.lap}@{key}")
    # Tier D.
    if any(k.startswith("D/") for a in ("S1", "S2", "S3") for k in states.get(a, {})):
        track = real_track(D_TRACK)
        dcs = hockenheim_corners(track)
        for dc in dcs:
            key = f"D/{dc.index}"
            sts = [(a, states[a][key]) for a in ("S1", "S2", "S3") if key in states.get(a, {})]
            if not sts:
                continue
            j = None if swap is None else swap.get(dc.index)
            conds, exc = hockenheim_conditions(dc.index, track, j, f6_scale)
            rr_pts = _RealRoad(D_TRACK, dc.corner, dc.before, dc.after, N_KNOTS, track)
            pts = dict(a=rr_pts.i_a, b=rr_pts.i_b, lead1=rr_pts.lead1, c0=rr_pts.c0, c1=rr_pts.c1)
            base = f"D/{dc.index}"
            for cond, p in conds.items():
                lam = float(cond.split("lambda")[1]) if "lambda" in cond else 1.0
                src = rr_pts.source_info(condition=cond, corner_index=dc.index, D_gap=dc.D_gap,
                                         exact=_exact(_d_exact_key(cond), pts),
                                         mirrored=cond == "F4/mirror", key=key,
                                         **({"lambda": lam} if "lambda" in cond else {}),
                                         **({"f6_scale": f6_scale} if cond.startswith("F6/")
                                            else {}),
                                         **({"swap_with": j} if cond == "F5/swapA" else {}))
                sign = -dc.corner.sign if cond == "F4/mirror" else dc.corner.sign
                if cond == "F5/swapA":
                    sign = dcs[j].corner.sign
                flags = {"cap_binds"} if p.capped.any() else set()
                row = _Row(f"{base}/{cond}", p, "hockenheim", sign, None, {"D"}, src, flags)
                if cond in (COND_A, "F4/mirror"):
                    row.mirror_name = f"{base}/{'F4/mirror' if cond == COND_A else COND_A}"
                rows.append(row)
            for agent, st in sts:
                runners = RUNNERS_FROM_S1 if agent == "S1" else (agent,)
                pair_a = pair
                pair += 1
                for cond in conds:
                    mir = cond == "F4/mirror"
                    x0, pa, ac = _strip_state(exc, st, mir)
                    pid = pair_a if cond in (COND_A, "F4/mirror") else pair
                    if cond not in (COND_A, "F4/mirror"):
                        pair += 1
                    add_cell(f"{base}/{cond}", st, "real", ("D",), runners, set(), pid, mir, x0,
                             pa, ac, f"{agent}:lap{st.lap}@{key}")
    if not rows:
        return bank, list(cells)
    stims, _ = _assemble(rows, start_id=len(bank))
    by_name = {s.name: s for s in stims}
    out_cells = list(cells)
    for c in new_cells:
        s = by_name[c["name"]]
        out_cells.append(Cell(
            cell_id=len(out_cells), stim_id=s.stim_id, pair_id=c["pair_id"], tiers=c["tiers"],
            approach=c["approach"], v0=float(c["x0"][3]), D=float(S_C - c["x0"][0]),
            x0=c["x0"], prev_action0=c["pa"], accel0=c["ac"], state_source=c["state_source"],
            flags=tuple(sorted(set(c["flags"]) | set(s.flags))), mirrored=c["mirrored"],
            runners=tuple(c["runners"])))
    new = StimulusBank(np.concatenate([bank.kappa, np.stack([r.profile.kappa for r in rows])]),
                       np.concatenate([bank.w_left, np.stack([r.profile.w_left for r in rows])]),
                       np.concatenate([bank.w_right, np.stack([r.profile.w_right for r in rows])]),
                       list(bank.stimuli) + stims, bank.ds)
    return new, out_cells


def split_nulls(cells: list[Cell]) -> list[Cell]:
    """One tier G null cell per (straight, v0, W): every corner cell points to the null of
    its own null family (the same straight and v0) with its own W (protocol 5: the catch
    trial and the corner time use the tier G straight with the same v0, width and W).

    Idempotent, and safe to re-run after any W change (V2 moving a class to W = 800): a
    family's existing null with the wanted W is reused, a null that no corner cell needs
    any more takes a new W before a copy is made, and a copy gets the next cell and pair
    ids. Cell ids must equal positions."""
    out = list(cells)
    if any(c.cell_id != i for i, c in enumerate(out)):
        raise ValueError("cell ids must equal their positions")
    nulls = [c for c in out if "G" in c.tiers]
    if not nulls:
        return out
    family = {c.cell_id: (c.stim_id, round(float(c.v0), 6)) for c in nulls}
    members: dict[tuple, list[int]] = {}
    for c in nulls:
        members.setdefault(family[c.cell_id], []).append(c.cell_id)
    need: dict[tuple, set[int]] = {}
    for c in out:
        if "G" in c.tiers or c.null_cell < 0 or c.W is None or c.null_cell not in family:
            continue
        need.setdefault(family[c.null_cell], set()).add(int(c.W))
    pair = max(c.pair_id for c in out) + 1
    for fam, ws in sorted(need.items()):
        ids = members[fam]
        have = {int(out[i].W): i for i in ids if out[i].W is not None and int(out[i].W) in ws}
        spare = [i for i in ids if i not in have.values()]
        for w in sorted(ws - set(have)):
            if spare:
                i = spare.pop(0)
                out[i] = replace(out[i], W=w)
            else:
                i = len(out)
                out.append(replace(out[ids[0]], cell_id=i, pair_id=pair, W=w))
                pair += 1
                members[fam].append(i)
                family[i] = fam
            have[w] = i
        for k, c in enumerate(out):
            if "G" in c.tiers or c.null_cell < 0 or c.W is None or c.null_cell not in family:
                continue
            if family[c.null_cell] == fam and c.null_cell != have[int(c.W)]:
                out[k] = replace(c, null_cell=have[int(c.W)])
    return out


def real_state_requests(reachable: dict[str, float] | None = None) -> list[dict]:
    """The real-lap states ``build_real_design`` needs, as {key, agent, track, s_target,
    lap} (for ``agents.real_state(lap, s_target, lap)``): E1 at s_c - 800 on each crash lap
    (a negative s_target means the run's start state, D16); tier D at s_start = s_c - D_gap
    - 100 on lap 2 for S1, S2 and S3; and, once G2's reach per circuit is known
    (``reachable``), E2 at s_c - 800 on the lap ``matched_corners`` picks."""
    out = []
    for p in crash_pairs():
        c = crash_corner(real_track(p.track), p.s_crash)
        out.append(dict(key=crash_corner_key(p.track, c), agent=p.agent, track=p.track,
                        s_target=float(c.s_start - E_CUT_BEFORE), lap=p.lap))
    for dc in hockenheim_corners():
        for agent in ("S1", "S2", "S3"):
            out.append(dict(key=f"D/{dc.index}", agent=agent, track=D_TRACK,
                            s_target=float(dc.corner.s_start - dc.before), lap=2))
    if reachable is not None:
        crash = [(t, c) for (t, c, _) in distinct_crash_corners().values()]
        rdev = {name: real_track(name) for name in R_DEV}
        for tname, c, lap in matched_corners(rdev, crash, reachable):
            out.append(dict(key=f"E2/{tname}/{c.s_start:.0f}", agent="G2", track=tname,
                            s_target=float(c.s_start - E_CUT_BEFORE), lap=lap))
    return out


# ------------------------------------------------------------------------- checks


THETA_EFF_TOL_DEG = 0.3            # step 1: realised angle within 0.3 deg of nominal
THETA_SAMPLE_AREA_TOL_DEG = 0.02   # R3: theta_eff equals the knot samples' area (spline)
THETA_SPEC_AREA_TOL_DEG = 1e-6     # the continuous trapezoid turns exactly its nominal angle
# Ruling R4: a twin shorter than its real corner is accepted only if it matches R_seg
# (relative) and the angle (deg) to < 1e-3, the twin fit's own convergence (``_twin_match``).
TWIN_R4_TOL = {"R_seg_rel": 1e-3, "angle_deg": 1e-3}


def theta_sampling_bound_deg(spec: CornerSpec, ds: float = DS) -> float:
    """Ruling R3's bound on the sampling miss (deg): how far the knot samples' area
    (sum kappa ds, which the C2 spline integrates exactly) can lie from the continuous
    trapezoid's area when its kinks fall between knots. The trapezoid rule on a piecewise
    linear profile errs by |slope change| ds^2 t (1 - t) / 2 at a kink a fraction t into its
    knot interval, at most |slope change| ds^2 / 8; the trapezoid's four kinks change the
    slope by kappa_peak / l_r each (the triangle's apex twice that, once), so the miss is at
    most kappa_peak ds^2 / (2 l_r)."""
    return float(np.degrees(spec.kappa_peak * ds * ds / (2.0 * spec.l_r)))


def spec_area_deg(spec: CornerSpec) -> float:
    """The continuous trapezoid's own turning angle (deg): kappa_peak times the area under its
    unit profile, integrated from its vertices (the ones ``kappa_formula`` checks the samples
    against). Equals theta_deg for a consistent spec."""
    a = spec.s_c
    xs = [a, a + spec.l_r, a + spec.l_r + spec.L_p, a + spec.L_c]
    return float(np.degrees(spec.kappa_peak * np.trapezoid([0.0, 1.0, 1.0, 0.0], xs)))


def twin_r4_match(R_seg: float, theta_deg: float, target: dict) -> bool:
    """Ruling R4: the twin matches its real corner's R_seg and angle to < 1e-3
    (``TWIN_R4_TOL``)."""
    return bool(abs(R_seg / target["R_seg"] - 1.0) < TWIN_R4_TOL["R_seg_rel"]
                and abs(theta_deg - target["theta_deg"]) < TWIN_R4_TOL["angle_deg"])


def check_design_details(bank: StimulusBank, cells: list[Cell]) -> dict[str, dict]:
    """Step 1's builder checks with their numbers (``check_design`` keeps the verdicts)."""
    from apexzero.vehicle.sim import track_arrays

    res: dict[str, dict] = {}
    syn = [s for s in bank.stimuli if s.spec is not None and s.kind in ("synthetic", "pilot")]
    # kappa against an independent evaluation of the trapezoid (np.interp on its vertices).
    worst = 0.0
    for s in syn:
        sp = s.spec
        a = sp.s_c
        xs = [a - 1.0, a, a + sp.l_r, a + sp.l_r + sp.L_p, a + sp.L_c, a + sp.L_c + 1.0]
        ys = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        grid = np.arange(bank.kappa.shape[1]) * bank.ds
        ref = sp.direction * sp.kappa_peak * np.interp(grid, xs, ys)
        worst = max(worst, float(np.max(np.abs(bank.kappa[s.stim_id] - ref))))
    res["kappa_formula"] = dict(ok=worst <= 1e-12, max_abs=worst, rows=len(syn))
    # Realised angle within 0.3 deg of nominal (synthetic rows, including twins), and every
    # spec's continuous trapezoid turning exactly its nominal angle (``spec_area_deg``: the
    # samples follow the spec, so only this catches a spec whose parameters disagree with
    # its angle). Ruling R3: a miss from the 2 m knot sampling of the trapezoid's kinks is
    # accepted and listed; theta_eff stays recorded per row (Geometry). A miss is sampling
    # only on a synthetic or pilot row (its samples are the trapezoid's, ``kappa_formula``)
    # whose spec area is exact, whose theta_eff is the knot samples' own area (the spline
    # integrates as the samples' trapezoid sum) and whose samples' area lies within the
    # kinks' sampling bound of the nominal angle (``theta_sampling_bound_deg``). Any other
    # miss fails.
    dev, fails, accepted = [], [], []
    worst_spec = 0.0
    for s in bank.stimuli:
        if s.spec is None:
            continue
        sp = s.spec
        sa = spec_area_deg(sp)
        spec_miss = abs(sa - sp.theta_deg)
        worst_spec = max(worst_spec, spec_miss)
        nominal = s.direction * sp.theta_deg
        d = abs(s.geometry.theta_eff_deg - nominal)
        dev.append(d)
        if d <= THETA_EFF_TOL_DEG and spec_miss <= THETA_SPEC_AREA_TOL_DEG:
            continue
        area = float(np.degrees(np.sum(bank.kappa[s.stim_id]) * bank.ds))
        bound = theta_sampling_bound_deg(sp, float(bank.ds))
        if spec_miss > THETA_SPEC_AREA_TOL_DEG:
            why = "spec area is not the nominal angle"
        elif s.kind not in ("synthetic", "pilot"):
            why = "not a synthetic row"
        elif abs(s.geometry.theta_eff_deg - area) > THETA_SAMPLE_AREA_TOL_DEG:
            why = "theta_eff is not the knot samples' area"
        elif abs(area - nominal) > bound + 1e-9:
            why = "samples' area beyond the kinks' sampling bound"
        else:
            why = ""
        row = dict(name=s.name, theta_eff=s.geometry.theta_eff_deg, sample_area_deg=area,
                   spec_area_deg=sa, sampling_bound_deg=bound, l_r=sp.l_r, L_p=sp.L_p)
        if why:
            fails.append(dict(row, why=why))
        else:
            accepted.append(row)
    res["theta_eff"] = dict(ok=not fails, max_deg=max(dev, default=0.0), fails=fails,
                            accepted=accepted, accepted_by="R3" if accepted else "",
                            spec_area_max_deg=worst_spec)
    # track_arrays(to_track(.)) equals the bank's float32 arrays, every row.
    tr_b = bank.arrays()
    worst, bad = 0.0, []
    for i in range(len(bank)):
        a = track_arrays(bank.track(i))
        b = jax.tree.map(lambda x, i=i: x[i], tr_b)
        for f in ("kappa_c", "w_left_c", "w_right_c", "x", "y", "theta"):
            d = float(np.max(np.abs(np.asarray(getattr(a, f), np.float64) - np.asarray(getattr(b, f), np.float64))))
            worst = max(worst, d)
            if d > 0.0:
                bad.append((i, f, d))
        if float(a.ds) != float(b.ds) or int(a.n_points) != int(b.n_points) or bool(a.closed) != bool(b.closed):
            bad.append((i, "meta", 1.0))
    res["track_arrays_roundtrip"] = dict(ok=not bad, max_abs=worst, mismatches=bad[:10])
    # Mirrors exact, on the samples and on the float32 arrays.
    worst = 0.0
    for s in bank.stimuli:
        m = s.mirror_id
        if m < 0:
            continue
        i = s.stim_id
        worst = max(worst, float(np.max(np.abs(bank.kappa[i] + bank.kappa[m]))),
                    float(np.max(np.abs(bank.w_left[i] - bank.w_right[m]))),
                    float(np.max(np.abs(bank.w_right[i] - bank.w_left[m]))))
        for f, sgn in (("kappa_c", -1.0), ("theta", -1.0), ("y", -1.0), ("x", 1.0)):
            ai = np.asarray(getattr(tr_b, f)[i], np.float64)
            am = np.asarray(getattr(tr_b, f)[m], np.float64)
            worst = max(worst, float(np.max(np.abs(ai - sgn * am))))
        worst = max(worst, float(np.max(np.abs(np.asarray(tr_b.w_left_c[i], np.float64)
                                               - np.asarray(tr_b.w_right_c[m], np.float64)))))
    res["mirror_exact"] = dict(ok=worst == 0.0, max_abs=worst)
    # Inside cap on synthetic rows (not the no-cap variants); twins after their real lead-in
    # (real roads keep their own widths: refined Norisring itself reaches 0.68).
    worst = 0.0
    for s in bank.stimuli:
        if s.spec is None or s.spec.no_cap:
            continue
        k = bank.kappa[s.stim_id]
        inside = np.where(k > 0, bank.w_left[s.stim_id], bank.w_right[s.stim_id])
        m = (inside - 1.0) * np.abs(k)
        if s.kind == "twin":
            m = m[round(s.source["lead_in"][1] / DS) + TAPER_KNOTS:]
        worst = max(worst, float(np.max(m)))
    res["inside_cap"] = dict(ok=worst <= INSIDE_CAP_MARGIN + 1e-9, max=worst)
    # Lengths: s_c1 + 250 < 4598 m; L_c <= 521 m; every cell's window ends before 4598 m.
    worst_exit = max(s.landmarks.s_c1 + COMPLETE_AFTER_EXIT for s in bank.stimuli)
    longest = max((s.landmarks.L_c for s in bank.stimuli if s.kind == "synthetic"), default=0.0)
    reach = []
    for c in cells:
        if c.D is None:
            s0 = S_C - 50.0          # A2 before layout: D >= 50 m is the most that can apply
        else:
            s0 = float(c.initial_state(0)[0])
        W = c.W or W_LONG
        reach.append(s0 + W * DT * kph(V_TOP_KPH))
    res["lengths"] = dict(ok=bool(worst_exit < S_TRUNC and longest <= 521.0
                                  and (max(reach, default=0.0) < S_TRUNC)),
                          max_exit_plus_250=worst_exit, longest_corner=longest,
                          max_window_reach=max(reach, default=0.0))
    # Lockbox: an R_TEST name raises PermissionError before anything loads.
    try:
        real_track(R_TEST[0])
        lock_ok = False
    except PermissionError:
        lock_ok = True
    res["lockbox"] = dict(ok=lock_ok, name=R_TEST[0])
    # Transplant kappa equals the source over every exact range.
    worst, nrows = 0.0, 0
    for s in bank.stimuli:
        ex = s.source.get("exact")
        if not ex:
            continue
        tr = real_track(s.source["track"])
        sign = -1.0 if s.source.get("mirrored") else 1.0
        for lo, hi in ex:
            i = np.arange(lo, hi + 1)
            j = np.mod(i + int(s.source["j_offset"]), tr.n_points)
            worst = max(worst, float(np.max(np.abs(bank.kappa[s.stim_id, i] - sign * tr.kappa[j]))))
        nrows += 1
    # ``vacuous``: no row to check (a synthetic-only design); reported, not a pass.
    res["transplant_source"] = dict(ok=worst <= 1e-9, max_abs=worst, rows=nrows,
                                    vacuous=nrows == 0)
    # Twin fits within tolerance, re-measured on the bank row. Ruling R4: a twin that matches
    # R_seg and the angle to < 1e-3 (``twin_r4_match``) but is shorter than its real corner
    # (the real corner is longer than the longest symmetric trapezoid with that R_seg and
    # angle) is accepted and listed with its length shortfall, which the twin-transfer
    # analysis carries as a covariate. A twin outside the protocol's 2% / 1 deg, a length
    # miss on a twin that matches R_seg or the angle only to 1e-3 or worse, or a twin longer
    # than its corner, fails.
    fails, accepted, shortfall, n_twins = [], [], {}, 0
    worst_r = worst_a = 0.0
    for s in bank.stimuli:
        if s.kind != "twin":
            continue
        n_twins += 1
        t = s.source["twin_target"]
        g = s.geometry
        dr, da = abs(g.R_seg / t["R_seg"] - 1.0), abs(g.theta_seg_deg - t["theta_deg"])
        short = twin_length_shortfall(s)
        shortfall[s.name] = short
        r_ok = dr <= TWIN_TOL["R_seg_rel"] + 1e-9 and da <= TWIN_TOL["angle_deg"] + 1e-9
        if r_ok and abs(short) <= TWIN_TOL["length"] + 1e-9:
            continue
        row = dict(name=s.name, target=t, R_seg=g.R_seg, theta=g.theta_seg_deg, L_seg=g.L_seg,
                   length_shortfall=short,
                   note=s.source.get("twin_achieved", {}).get("note", ""))
        if r_ok and short > 0.0 and twin_r4_match(g.R_seg, g.theta_seg_deg, t):
            accepted.append(row)
            worst_r, worst_a = max(worst_r, dr), max(worst_a, da)
        else:
            fails.append(row)
    res["twin_fit"] = dict(ok=not fails, fails=fails, accepted=accepted,
                           accepted_by="R4" if accepted else "", rows=n_twins,
                           vacuous=n_twins == 0, length_shortfall=shortfall,
                           accepted_max_R_rel=worst_r, accepted_max_angle_deg=worst_a)
    return res


def twin_length_shortfall(stim: Stimulus) -> float:
    """Ruling R4: the real corner's L_seg minus its twin's, re-measured on the bank row (m;
    > 0 when the twin is shorter). NaN for a row that is not a twin."""
    if stim.kind != "twin":
        return float("nan")
    return float(stim.source["twin_target"]["L_seg"] - stim.geometry.L_seg)


def check_design(bank: StimulusBank, cells: list[Cell]) -> dict[str, bool]:
    """Step 1's builder checks (protocol 10): kappa matches the trapezoid formula to 1e-12;
    theta_eff within 0.3 deg of nominal and every spec's trapezoid turning its nominal angle;
    ``track_arrays(to_track(.))`` equals the bank's arrays; mirrors exact;
    (w_in - 1)|kappa| <= 0.6 (capped rows); s_c1 + 250 < 4598 and every key window ends
    before 4598 m; R_TEST names raise PermissionError; transplant kappa equals the source to
    1e-9; twin fits within tolerance. The theta_eff misses of ruling R3 (knot sampling of
    the kinks, within its bound) and the twin length shortfalls of R4 (R_seg and angle
    matched to < 1e-3) pass and are listed in ``check_design_details``. Returns
    {check: passed}."""
    return {k: bool(v["ok"]) for k, v in check_design_details(bank, cells).items()}


# ------------------------------------------------------------------------- support covariate

CELL_SUPPORT_COLUMNS = ("cell", "stim", "R_seg", "theta_deg", "l_r", "L_c", "v_peak") + tuple(
    f"logdens_{g}" for g in GRAMMARS)


def cell_support_table(bank: StimulusBank, cells: list[Cell], support: SupportTables) -> io.Table:
    """Joint training-support covariate per cell (protocol 2.3): features (log R_seg, theta,
    l_r, L_c, log v_peak^QSS) with the stimulus's segment() geometry, its ramp (spec l_r;
    real roads the fitted twin's l_r, tier D none) and v_peak^QSS from the cell's start
    state, and the kNN log density under each grammar's training sample, every feature
    divided by that sample's own SD. Cells without D yet (A2 before layout) get nan."""
    ok = [c for c in cells if c.D is not None]
    vp = np.full(len(cells), np.nan)
    if ok:
        idx = [c.cell_id for c in ok]
        vp_ok = stimulus_v_peak_batch(bank.kappa[[c.stim_id for c in ok]],
                                      np.array([float(c.initial_state(0)[0]) for c in ok]),
                                      np.array([c.v0 for c in ok]),
                                      np.array([bank.stimuli[c.stim_id].landmarks.s_c0 for c in ok]))
        pos = {cid: k for k, cid in enumerate(c.cell_id for c in cells)}
        for cid, v in zip(idx, vp_ok):
            vp[pos[cid]] = v
    st = [bank.stimuli[c.stim_id] for c in cells]
    R = np.array([s.geometry.R_seg for s in st])
    th = np.array([s.geometry.theta_seg_deg for s in st])
    L = np.array([s.geometry.L_seg for s in st])
    lr = np.array([s.spec.l_r if s.spec is not None else s.source.get("twin_l_r", np.nan) for s in st],
                  float)
    feats = support_features(R, th, lr, L, vp)
    out = {"cell": np.array([c.cell_id for c in cells]), "stim": np.array([c.stim_id for c in cells]),
           "R_seg": R, "theta_deg": th, "l_r": lr, "L_c": L, "v_peak": vp}
    for g in GRAMMARS:
        train = table_features(support.table(g))
        # Each feature standardised by that grammar's own training SD (protocol 2.3).
        fin = train[np.all(np.isfinite(train), axis=1)]
        out[f"logdens_{g}"] = knn_log_density(train, feats, np.std(fin, axis=0, ddof=1))
    return out


def save_design(out_dir: Path, bank: StimulusBank, cells: list[Cell]) -> str:
    """bank.npz, stimuli.json and cells.json under ``out_dir``; returns the design hash
    (stimulus samples and records plus the cells), the manifest's stimulus hash."""
    out_dir = Path(out_dir)
    bank.save(out_dir)
    save_cells(out_dir / "cells.json", cells)
    return design_hash(bank, cells)


def design_hash(bank: StimulusBank, cells: list[Cell]) -> str:
    return io.sha256_json({"bank": bank.content_hash(), "cells": cells})


def load_design(out_dir: Path) -> tuple[StimulusBank, list[Cell]]:
    out_dir = Path(out_dir)
    return StimulusBank.load(out_dir), load_cells(out_dir / "cells.json")
