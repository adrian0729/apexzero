"""Training-support statistics: what corners each agent saw, and what counts as typical.

Protocol 1.4 (l_typ, single radius estimator), 2.3 (regimes, clipping, joint support
covariate) and 3.5 (v_typ for A2). Everything is computed from 300 generated strips per
training grammar, with the generator's own per-corner parameters, and frozen in step 1:

- ``l_typ(R)``: median generator ramp ``l_r`` of base-grammar corners per R_seg bin,
  interpolated linearly in log R through bin medians placed at 9.4, 14.6, 22.9, 40 and 75 m.
- ``v_typ``: median QSS peak approach speed of base-grammar corners per (R_seg bin, theta
  class), rounded to the A2 speed choices.
- regimes per agent: q1 / q10 of R_seg from that agent's own grammar, and R_clip from the
  final snapshot's obs_rms (the radius below which the kappa * 50 preview input clips).
- the joint kNN support density over (log R_seg, theta, l_r, L_c, log v_peak^QSS).

Grammars are built exactly as scripts/train_dr.py builds them: ``fit_grammar`` on the 12
raw (unrefined) R_DEV centrelines, a per-strip mixture with ``tilted(g, alpha)`` at the run's
tilt fraction (``adapters.generate_bank``), and QSS reference speeds from
``speed_profile_jax`` as the training bank computes them. ``generate()`` does not return its
per-corner ramp, so ``generate_with_corners`` is a copy that also returns the corner
parameters; a test checks its kappa and widths equal ``generate``'s. ``qss_profile_jax`` is a
copy of ``speed_profile_jax`` with an optional start speed, so training corners (no start
speed: identical to the repo function) and stimuli (start speed v0 at s0) share one QSS
estimator. R_seg is always ``1 / max|kappa_smooth|`` from ``corners.segment`` with its
defaults, for training strips, stimuli, R_DEV corners and crash corners alike.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import RunningMeanStd
from apexzero.interp import io
from apexzero.interp.constants import (
    L_TYP_BIN_EDGES,
    L_TYP_R_NODES,
    OBS_CLIP,
    STRIP_TAIL,
    SUPPORT_N_STRIPS,
    THETA_CLASS_EDGE_DEG,
    V_TYP_CHOICES_KPH,
)
from apexzero.interp.io import Table
from apexzero.interp.registry import GRAMMARS, GrammarSpec, agent_grammar
from apexzero.tracks.corners import R_DEV, segment, smooth
from apexzero.tracks.generator import (
    DS,
    MAX_SEGMENTS,
    N_POINTS,
    Grammar,
    _lowfreq,
    fit_grammar,
    tilted,
)
from apexzero.tracks.track import Track, load_tumftm
from apexzero.vehicle.params import F1_2025, VehicleParams

# ------------------------------------------------------------------------- grammars

# GrammarSpec and GRAMMARS live in ``registry`` (with the agents that train on them) and are
# re-exported here. Each generalist's own training grammar, read from the agent registry:
# G1-G3 and, by ruling R18, their replicates (a replicate trains on its variant's grammar, so
# its q1 / q10 equal its seed-0 agent's; its R_clip comes from its own obs_rms).
AGENT_GRAMMAR = agent_grammar()
SPECIALIST_TRACK = "Hockenheim"


class GenCorners(NamedTuple):
    """Per generated corner of one strip (``max_segments`` slots; slots past the strip end
    are present but lie beyond n_points * ds). Units: m, rad, 1/m."""

    start: jax.Array    # [M] s of the ramp-in start
    angle: jax.Array    # [M] |turning angle| (rad) after jitter
    sign: jax.Array     # [M] +1 left, -1 right
    k_peak: jax.Array   # [M] peak curvature after the triangle rule (1/m)
    l_r: jax.Array      # [M] ramp length (m), >= ramp_min
    l_c: jax.Array      # [M] constant-radius arc length (m), 0 for triangles
    length: jax.Array   # [M] l_c + 2 l_r (m)


class Strips(NamedTuple):
    """``training_strips`` output, every field stacked over S strips."""

    kappa: jax.Array        # [S, N] 1/m (float32, as in training)
    w_left: jax.Array       # [S, N] m
    w_right: jax.Array      # [S, N] m
    corners: GenCorners     # each [S, M]
    use_alt: jax.Array      # [S] bool, strip drawn from the tilted grammar
    v_qss: jax.Array        # [S, N] m/s, the bank's QSS reference speed


@functools.lru_cache(maxsize=1)
def dev_centrelines() -> tuple[Track, ...]:
    """The 12 raw R_DEV centrelines the trainers fit their grammar on (``load_tumftm``, no
    refine). ``lockbox_guard`` runs on every name first."""
    from apexzero.viz.export import lockbox_guard

    out = []
    for name in R_DEV:
        lockbox_guard(name)
        out.append(load_tumftm(io.REPO / "data" / "tracks" / "tumftm" / f"{name}.csv"))
    return tuple(out)


def grammar(spec: GrammarSpec) -> tuple[Grammar, Grammar | None]:
    """(g, g_alt) for a spec, fitted on the R_DEV circuits exactly as train_dr.py does
    (``fit_grammar`` defaults; ``tilted(g, alpha=spec.tilt_alpha)``). Loads only R_DEV
    tracks. ``g_alt`` is None for a grammar without a tilted share (G1's base grammar)."""
    g = fit_grammar(list(dev_centrelines()))
    g_alt = tilted(g, alpha=spec.tilt_alpha) if spec.tilt_fraction > 0.0 else None
    return g, g_alt


def generate_with_corners(key: jax.Array, g: Grammar, n_points: int = N_POINTS,
                          ds: float = DS, max_segments: int = MAX_SEGMENTS):
    """Copy of ``generator.generate`` that also returns the generator's corners.

    Returns (kappa [n_points], w_left [n_points], w_right [n_points], GenCorners). The
    first three equal ``generator.generate(key, g, n_points, ds, max_segments)`` exactly
    (test): the body is the repo function line for line, plus the corner record. Protocol
    1.4 (l_typ needs the generator's ``l_r``)."""
    keys = jax.random.split(key, 10)
    m = max_segments
    jit = lambda k, shape: jnp.exp(g.jitter * jax.random.normal(k, shape))

    ci = jax.random.categorical(keys[0], g.corner_logits, shape=(m,))
    angle = g.corner_angle[ci] * jit(keys[1], (m,))
    radius = jnp.maximum(g.corner_rmin[ci] * jit(keys[2], (m,)), g.r_floor)
    clen = g.corner_len[ci] * jit(keys[3], (m,))

    si = jax.random.randint(keys[4], (m,), 0, g.straight_len.shape[0])
    gap = g.gap_scale * g.straight_len[si] * jit(keys[5], (m,))
    first = jax.random.uniform(keys[6], (), minval=500.0, maxval=900.0)
    gap = gap.at[0].set(first)

    p_alt = jnp.where(gap[1:] < 20.0, 1.0, jnp.where(gap[1:] < 60.0, g.p_alt_short, g.p_alt_long))
    flips = jax.random.uniform(keys[7], (m - 1,)) < p_alt
    s0 = jnp.where(jax.random.uniform(keys[8], ()) < 0.5, -1.0, 1.0)
    sign = s0 * jnp.concatenate([jnp.ones(1), jnp.cumprod(jnp.where(flips, -1.0, 1.0))])

    k_peak = 1.0 / radius
    arc_equiv = angle / k_peak
    l_r = jnp.clip(clen - arc_equiv, g.ramp_min, None)
    tri = arc_equiv < l_r
    k_peak = jnp.where(tri, angle / l_r, k_peak)
    l_c = jnp.where(tri, 0.0, arc_equiv - l_r)
    c_total = l_c + 2.0 * l_r

    seg_len = jnp.stack([gap, c_total], axis=1).reshape(-1)
    starts = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg_len)[:-1]])
    s = jnp.arange(n_points) * ds
    j = jnp.clip(jnp.searchsorted(starts, s, side="right") - 1, 0, 2 * m - 1)
    is_corner = (j % 2) == 1
    c = j // 2
    u = s - starts[j]
    lr, lc, kp = l_r[c], l_c[c], k_peak[c]
    prof = jnp.where(u < lr, u / lr, jnp.where(u < lr + lc, 1.0, jnp.clip((2 * lr + lc - u) / lr, 0.0, 1.0)))
    kappa = jnp.where(is_corner, sign[c] * kp * prof, 0.0)

    ti = jax.random.randint(keys[9], (), 0, g.width.shape[0])
    kw = jax.random.split(keys[9], 3)
    w_total = g.width_scale * g.width[ti] * jnp.exp(0.08 * jax.random.normal(kw[0], ()))
    w_total = w_total + g.width_scale * _lowfreq(kw[1], s, 1.5)
    w_total = jnp.clip(w_total, 7.0 * g.width_scale, 18.0)
    asym = _lowfreq(kw[2], s, 0.5)
    wl = 0.5 * w_total + asym
    wr = 0.5 * w_total - asym
    cap = 1.0 + g.margin / jnp.maximum(jnp.abs(kappa), 1e-6)
    wl = jnp.where(kappa > 0, jnp.minimum(wl, cap), wl)
    wr = jnp.where(kappa < 0, jnp.minimum(wr, cap), wr)
    corners = GenCorners(start=starts[1::2], angle=angle, sign=sign, k_peak=k_peak, l_r=l_r,
                         l_c=l_c, length=c_total)
    return kappa, wl, wr, corners


def qss_profile_jax(kappa, ds, p: VehicleParams = F1_2025, v_start=jnp.inf,
                    v_cap: float = 120.0, knot_cap=None):
    """Copy of ``qss.speed_profile_jax`` with a start speed and an optional per-knot speed
    cap: the forward pass starts at min(v_lim[0], v_start), and ``knot_cap`` [n] (inf where
    uncapped) lowers v_lim knot by knot. With neither it is the repo function (test), which
    is how the training bank computes v_ref."""
    fz_tyre = 0.25 * (p.m * p.g + 0.5 * p.rho * (p.cla_f + p.cla_r) * 60.0**2)
    dmu = p.kmu * (fz_tyre - p.fz0)
    mu_x = p.mu_scale * (0.5 * (p.mux_f + p.mux_r) + dmu)
    mu_y = p.mu_scale * (0.5 * (p.muy_f + p.muy_r) + dmu)
    mu_x_r = p.mu_scale * (p.mux_r + dmu)
    cla = p.cla_f + p.cla_r

    def normal(v):
        return p.m * p.g + 0.5 * p.rho * cla * v * v

    def drag(v):
        return 0.5 * p.rho * p.cda * v * v + p.f_roll * normal(v)

    def a_lat(v):
        return mu_y * normal(v) / p.m

    def a_brake(v):
        return mu_x * normal(v) / p.m + drag(v) / p.m

    def a_acc(v):
        fz_r = p.m * p.g * p.lf / (p.lf + p.lr) + 0.5 * p.rho * p.cla_r * v * v
        return (jnp.minimum(p.p_max / jnp.maximum(v, p.v_drive_floor), mu_x_r * fz_r) - drag(v)) / p.m

    def bisect(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        up = p.p_max / mid > drag(mid)
        return (jnp.where(up, mid, lo), jnp.where(up, hi, mid))

    lo, hi = jax.lax.fori_loop(0, 60, bisect, (jnp.asarray(1.0), jnp.asarray(200.0)))
    v_cap = jnp.minimum(v_cap, 0.5 * (lo + hi))

    c = 0.5 * p.rho * cla / p.m
    k = jnp.abs(kappa)
    denom = k - mu_y * c
    v_lim = jnp.where(denom > 1e-9, jnp.minimum(jnp.sqrt(mu_y * p.g / jnp.maximum(denom, 1e-9)), v_cap), v_cap)
    if knot_cap is not None:
        v_lim = jnp.minimum(v_lim, jnp.asarray(knot_cap, v_lim.dtype))

    def ellipse(v, kk, a_max):
        lat = jnp.minimum(v * v * jnp.abs(kk) / a_lat(v), 1.0)
        return a_max * jnp.sqrt(1.0 - lat * lat)

    def fwd(v_prev, inp):
        kk_prev, vl = inp
        a = jnp.maximum(ellipse(v_prev, kk_prev, a_acc(v_prev)), 0.0)
        v = jnp.minimum(vl, jnp.sqrt(v_prev**2 + 2.0 * a * ds))
        return v, v

    v0 = jnp.minimum(v_lim[0], jnp.asarray(v_start, v_lim.dtype))
    _, vf = jax.lax.scan(fwd, v0, (kappa[:-1], v_lim[1:]))
    vf = jnp.concatenate([v0[None], vf])

    def bwd(v_next, inp):
        kk_next, vfi = inp
        a = ellipse(v_next, kk_next, a_brake(v_next))
        v = jnp.minimum(vfi, jnp.sqrt(v_next**2 + 2.0 * a * ds))
        return v, v

    _, vb = jax.lax.scan(bwd, vf[-1], (kappa[1:], vf[:-1]), reverse=True)
    return jnp.concatenate([vb, vf[-1:]])


@functools.partial(jax.jit, static_argnames=("mix",))
def _strips(keys, g, g_alt, alt_fraction, mix: bool):
    """vmapped generate_bank-style draw: mixture split only when the run mixes grammars."""

    def one(k):
        if mix:
            k, kc = jax.random.split(k)
            use_alt = jax.random.uniform(kc) < alt_fraction
            gg = jax.tree.map(lambda a, b: jnp.where(use_alt, b, a), g, g_alt)
        else:
            use_alt, gg = jnp.asarray(False), g
        kappa, wl, wr, corners = generate_with_corners(k, gg)
        return kappa, wl, wr, corners, use_alt, qss_profile_jax(kappa, DS, F1_2025)

    return jax.vmap(one)(keys)


def training_strips(spec: GrammarSpec, n_strips: int = SUPPORT_N_STRIPS,
                    key: jax.Array | None = None) -> Strips:
    """``n_strips`` strips of a grammar with the same per-strip mixture rule as
    ``adapters.generate_bank``. Default key: ``io.stream_key('support')`` folded with the
    grammar name's index in GRAMMARS. Returns a ``Strips`` (kappa, w_left, w_right [S, 4096],
    GenCorners stacked [S, M], use_alt [S], v_qss [S, 4096]). Protocol 2.3."""
    if key is None:
        key = jax.random.fold_in(io.stream_key("support"), list(GRAMMARS).index(spec.name))
    g, g_alt = grammar(spec)
    mix = g_alt is not None
    out = _strips(jax.random.split(key, n_strips), g, g if g_alt is None else g_alt,
                  jnp.asarray(spec.tilt_fraction, jnp.float32), mix)
    return Strips(*out)


# ------------------------------------------------------------------------- corner tables

# Columns of a corner table (training strips, R_DEV circuits): name -> unit / meaning.
CORNER_COLUMNS = {
    "source": "str: grammar name, or the R_DEV circuit name",
    "strip": "int: strip index (-1 for R_DEV)",
    "alt": "bool: strip drawn from the tilted grammar (False for R_DEV)",
    "s_start": "m: segment() corner start",
    "s_end": "m: segment() corner end (last knot inside)",
    "R_seg": "m: segment() r_min (10 m-smoothed), the single radius estimator",
    "R_eff": "m: 1 / max|kappa| of the exact samples over the corner",
    "theta_deg": "deg: |segment() angle|",
    "sign": "int: +1 left, -1 right",
    "L_c": "m: segment() length",
    "l_r": "m: generator ramp of the generated corner holding the R_seg peak (nan for R_DEV)",
    "onset_rate": "1/m^2: generator k_peak / l_r (nan for R_DEV)",
    "gap_before": "m: straight before the corner (segment() gaps)",
    "v_peak": "m/s: max centreline QSS speed (default car) over the straight before the corner",
    "v_apex": "m/s: min centreline QSS speed over the corner",
    "w_in_min": "m: min inside half-width over the corner",
    "strip_length": "m: driven length of the corner's strip or circuit (rates per km)",
}


def _bare_track(kappa: np.ndarray, ds: float, closed: bool, name: str = "strip") -> Track:
    n = len(kappa)
    z = np.zeros(n)
    return Track(name=name, s=np.arange(n) * ds, x=z, y=z, theta=z, kappa=kappa, w_left=z,
                 w_right=z, closed=closed)


def _corner_rows(kappa: np.ndarray, w_left: np.ndarray, w_right: np.ndarray, v: np.ndarray,
                 ds: float, closed: bool, s_limit: float | None = None) -> list[dict]:
    """Per-corner geometry and QSS speeds of one strip or circuit (indices wrap if closed)."""
    n = len(kappa)
    seg = segment(_bare_track(kappa, ds, closed))
    ks = smooth(kappa, ds, 10.0, closed)
    cs = seg.corners
    rows = []
    for j, c in enumerate(cs):
        if s_limit is not None and c.s_start >= s_limit:
            continue
        a = round(c.s_start / ds)
        m = round(c.length / ds)
        idx = (a + np.arange(m)) % n if closed else np.arange(a, min(a + m, n))
        if j == 0 and not closed:
            prev = np.arange(0, a + 1)
            gap = a * ds
        else:
            pe = round(cs[j - 1].s_end / ds)
            gk = (a - pe - 1) % n if closed else a - pe - 1
            prev = (pe + 1 + np.arange(max(gk, 0) + 1)) % n if closed else np.arange(pe + 1, a + 1)
            gap = max(gk, 0) * ds
        i_pk = idx[int(np.argmax(np.abs(ks[idx])))]
        inside = w_left[idx] if c.sign > 0 else w_right[idx]
        rows.append(dict(
            s_start=c.s_start, s_end=c.s_end, R_seg=c.r_min,
            R_eff=1.0 / max(float(np.abs(kappa[idx]).max()), 1e-9),
            theta_deg=abs(math.degrees(c.angle)), sign=c.sign, L_c=c.length, gap_before=gap,
            v_peak=float(v[prev].max()), v_apex=float(v[idx].min()),
            w_in_min=float(inside.min()), s_peak=i_pk * ds))
    return rows


def _holding_corner(gc: dict[str, np.ndarray], s_peak: float, n_points: int, ds: float) -> int:
    """Index of the generated corner containing s_peak, else the nearest one (D12)."""
    end = gc["start"] + gc["length"]
    live = gc["start"] < n_points * ds
    inside = live & (gc["start"] <= s_peak) & (s_peak < end)
    if inside.any():
        return int(np.argmax(inside))
    dist = np.where(live, np.maximum(gc["start"] - s_peak, s_peak - end), np.inf)
    return int(np.argmin(dist))


def _to_table(rows: list[dict]) -> Table:
    cols = list(CORNER_COLUMNS)
    out = {}
    for k in cols:
        vals = [r[k] for r in rows]
        if k == "source":
            out[k] = np.asarray(vals, dtype=str)
        elif k in ("strip", "sign"):
            out[k] = np.asarray(vals, dtype=np.int64)
        elif k == "alt":
            out[k] = np.asarray(vals, dtype=bool)
        else:
            out[k] = np.asarray(vals, dtype=np.float64)
    return out


def corner_table(spec: GrammarSpec, n_strips: int = SUPPORT_N_STRIPS,
                 key: jax.Array | None = None) -> Table:
    """Every ``corners.segment`` corner of ``training_strips(spec)`` that starts before the
    strip tail (s < L - 520), with the columns of ``CORNER_COLUMNS``. A segmented corner that
    merges several generated corners takes ``l_r`` of the generated corner containing its
    max|kappa_smooth| (D12). QSS profile: the training bank's ``speed_profile_jax(kappa, 2,
    F1_2025)`` (``qss_profile_jax`` without a start speed). Protocol 2.3."""
    st = jax.tree.map(np.asarray, training_strips(spec, n_strips, key))
    n = st.kappa.shape[1]
    reach = (n - 1) * DS - STRIP_TAIL
    rows = []
    for i in range(st.kappa.shape[0]):
        kap = st.kappa[i].astype(np.float64)
        gc = {f: np.asarray(getattr(st.corners, f)[i], np.float64) for f in GenCorners._fields}
        for r in _corner_rows(kap, st.w_left[i].astype(np.float64), st.w_right[i].astype(np.float64),
                              st.v_qss[i].astype(np.float64), DS, False, reach):
            m = _holding_corner(gc, r["s_peak"], n, DS)
            r.update(source=spec.name, strip=i, alt=bool(st.use_alt[i]), l_r=gc["l_r"][m],
                     onset_rate=gc["k_peak"][m] / gc["l_r"][m], strip_length=reach)
            rows.append(r)
    return _to_table(rows)


def rdev_corner_table() -> Table:
    """The same columns for every corner of the 12 R_DEV circuits on the real-lap Track
    (``stimuli.real_track``: ``lockbox_guard`` on each name, then the refined reference the
    evaluation drives), with the closed-circuit QSS profile (``adapters.reference_speed``).
    Used for R_DEV quantile marks on the curve axes and E2 matching. Never touches R_TEST."""
    from apexzero.interp.stimuli import real_track
    from apexzero.oracle.qss import GG, speed_profile

    gg = GG.from_params(F1_2025)
    rows = []
    for name in R_DEV:
        tr = real_track(name)
        v = speed_profile(tr.kappa, tr.ds, gg, closed=True)
        for r in _corner_rows(tr.kappa, tr.w_left, tr.w_right, v, tr.ds, True):
            r.update(source=name, strip=-1, alt=False, l_r=np.nan, onset_rate=np.nan,
                     strip_length=tr.length)
            rows.append(r)
    return _to_table(rows)


# ------------------------------------------------------------------------- frozen tables


# Corners turning less than this are left out of the "typical" statistics (l_typ, v_typ):
# with them the open top R_seg bin is dominated by near-straight kinks at the generator's
# 10 m ramp floor (median l_r 13 m above 55.9 m instead of >= 44 m), which makes l_typ fall
# from 42 m at R 40 to 13 m at R 75. Every stimulus turns >= 45 deg. Bins below 55.9 m hold
# no such kinks, so their medians are unchanged. Decision S1 in IMPLEMENTATION.md.
TYPICAL_MIN_THETA_DEG = 10.0


@dataclass(frozen=True)
class LTypTable:
    """l_typ(R): medians of ``l_r`` per R_seg bin (bins split at L_TYP_BIN_EDGES; the top
    bin is open), placed at L_TYP_R_NODES, over corners turning >= ``min_theta_deg``.
    ``l_nodes_all`` keeps the medians over every corner (the protocol's literal rule) for
    the record. Protocol 1.4."""

    r_nodes: tuple[float, ...] = L_TYP_R_NODES   # m
    l_nodes: tuple[float, ...] = ()              # m, median l_r per bin
    counts: tuple[int, ...] = ()                 # corners per bin
    min_theta_deg: float = TYPICAL_MIN_THETA_DEG
    l_nodes_all: tuple[float, ...] = ()          # m, medians over all corners


@dataclass(frozen=True)
class VTypTable:
    """v_typ per (R_seg bin, theta class): median base-grammar v_peak^QSS (m/s) and the
    rounded A2 choice (km/h, one of V_TYP_CHOICES_KPH). Bins as LTypTable; theta classes
    'lt120' and 'ge120'; keys "<bin index>|<class>". A (bin, class) cell with fewer than
    ``V_TYP_MIN_COUNT`` corners takes the pooled median of its bin (listed in ``pooled``).
    Protocol 3.5 (A2)."""

    bin_edges: tuple[float, ...] = L_TYP_BIN_EDGES
    median: dict[str, float] = field(default_factory=dict)      # "<bin>|<class>" -> m/s
    choice_kph: dict[str, float] = field(default_factory=dict)  # "<bin>|<class>" -> km/h
    counts: dict[str, int] = field(default_factory=dict)
    pooled: tuple[str, ...] = ()
    min_theta_deg: float = TYPICAL_MIN_THETA_DEG


V_TYP_MIN_COUNT = 20


@dataclass(frozen=True)
class RegimeTable:
    """One agent's regime boundaries (protocol 2.3), frozen into the manifest."""

    agent: str
    grammar: str                     # GrammarSpec.name ('' for specialists)
    q1: float                        # m, 1st percentile of R_seg on the agent's grammar
    q10: float                       # m, 10th percentile
    r_clip: tuple[float, ...] = ()   # m, R_clip(k) for the 40 preview points (worse sign)
    r_clip_max: float = float("nan")  # m, max over k
    snapshot: str = ""               # path of the snapshot whose obs_rms gave r_clip


@dataclass
class SupportTables:
    """Everything step 1 freezes (``outputs/psychophysics/support/support.json`` plus the
    corner tables as .npz). Feature SDs are those of the base-grammar training sample,
    shared by every density query."""

    l_typ: LTypTable
    v_typ: VTypTable
    regimes: dict[str, RegimeTable]
    feature_sd: tuple[float, ...]    # SDs of SUPPORT_FEATURES on the base training sample
    corner_tables: dict[str, str]    # grammar name -> path of its corner table (.npz)
    rdev_table: str                  # path of the R_DEV corner table
    hashes: dict[str, str] = field(default_factory=dict)
    quantiles: dict[str, dict[str, float]] = field(default_factory=dict)  # R_seg q's per set
    rates: dict[str, dict[str, float]] = field(default_factory=dict)      # per 100 km, per set

    def table(self, name: str) -> Table:
        """A frozen corner table by grammar name, or 'rdev'."""
        return io.load_table(self.rdev_table if name == "rdev" else self.corner_tables[name])


def _bin(R_seg) -> np.ndarray:
    return np.digitize(np.asarray(R_seg, float), L_TYP_BIN_EDGES)


def _bin_medians(base: Table, col: str, keep: np.ndarray) -> tuple[list[float], list[int]]:
    b = _bin(base["R_seg"])
    meds, counts = [], []
    for k in range(len(L_TYP_R_NODES)):
        sel = base[col][(b == k) & keep]
        if len(sel) == 0:
            raise ValueError(f"l_typ bin {k} has no base-grammar corners")
        meds.append(float(np.median(sel)))
        counts.append(len(sel))
    return meds, counts


def l_typ_table(base: Table, min_theta_deg: float = TYPICAL_MIN_THETA_DEG) -> LTypTable:
    """Bin medians of ``l_r`` from the base-grammar corner table, over corners turning at
    least ``min_theta_deg`` (0 gives the protocol's literal rule). Protocol 1.4."""
    keep = np.abs(base["theta_deg"]) >= min_theta_deg
    meds, counts = _bin_medians(base, "l_r", keep)
    all_meds, _ = _bin_medians(base, "l_r", np.ones(len(keep), bool))
    return LTypTable(L_TYP_R_NODES, tuple(meds), tuple(counts), float(min_theta_deg),
                     tuple(all_meds))


def l_typ(R: float, table: LTypTable) -> float:
    """l_typ(R) in m: linear in log R between the nodes, constant outside. Protocol 1.4."""
    return float(np.interp(math.log(R), np.log(table.r_nodes), table.l_nodes))


def _round_choice(v_ms: float) -> float:
    c = np.asarray(V_TYP_CHOICES_KPH)
    return float(c[int(np.argmin(np.abs(c - v_ms * 3.6)))])


def v_typ_table(base: Table, min_theta_deg: float = TYPICAL_MIN_THETA_DEG) -> VTypTable:
    """Median v_peak per (R_seg bin, theta class) over corners turning at least
    ``min_theta_deg``, rounded to the nearest A2 choice. Protocol 3.5."""
    b = _bin(base["R_seg"])
    keep = np.abs(base["theta_deg"]) >= min_theta_deg
    cls = np.array([theta_class(t) for t in base["theta_deg"]])
    median, choice, counts, pooled = {}, {}, {}, []
    for k in range(len(L_TYP_BIN_EDGES) + 1):
        for c in ("lt120", "ge120"):
            key = f"{k}|{c}"
            sel = (b == k) & (cls == c) & keep
            counts[key] = int(sel.sum())
            if counts[key] < V_TYP_MIN_COUNT:
                sel = (b == k) & keep
                pooled.append(key)
            median[key] = float(np.median(base["v_peak"][sel]))
            choice[key] = _round_choice(median[key])
    return VTypTable(L_TYP_BIN_EDGES, median, choice, counts, tuple(pooled), float(min_theta_deg))


def v_typ(R_seg: float, theta_deg: float, table: VTypTable) -> float:
    """A2 approach speed (m/s) for a stimulus: the rounded choice of its bin and class."""
    key = f"{int(_bin(R_seg))}|{theta_class(theta_deg)}"
    return table.choice_kph[key] / 3.6


def theta_class(theta_deg: float) -> str:
    """'lt120' or 'ge120' (THETA_CLASS_EDGE_DEG)."""
    return "lt120" if abs(theta_deg) < THETA_CLASS_EDGE_DEG else "ge120"


def _rms_arrays(obs_rms) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(obs_rms, dict):
        return np.asarray(obs_rms["mean"], np.float64), np.asarray(obs_rms["var"], np.float64)
    return np.asarray(obs_rms.mean, np.float64), np.asarray(obs_rms.var, np.float64)


KAPPA_CHANNELS = np.arange(40) * 5 + 4      # kappa * 50 in racing.observe's preview block


def r_clip(obs_rms: RunningMeanStd | dict) -> np.ndarray:
    """R_clip(k) (m) for the 40 preview kappa channels (obs index 5k + 4): the radius below
    which kappa * 50 normalises past the +-10 clip on the worse sign. With o = (50 kappa -
    mu) / sigma (sigma = sqrt(var + 1e-8), as ``RunningMeanStd.normalize``), the clip is hit
    at |kappa| = (10 sigma - |mu|) / 50 on the side of mu's opposite sign, so R_clip =
    50 / (10 sigma - |mu|) (inf if 10 sigma <= |mu|). The protocol prints 50 / (|mu| +
    10 sigma), which is the better sign; the two differ by < 2% on the finals (|mu| <=
    0.16, sigma >= 0.54). Reading obs_rms is not an evaluation of the agent. Protocol 2.3."""
    mu, var = _rms_arrays(obs_rms)
    sd = np.sqrt(var[KAPPA_CHANNELS] + 1e-8)
    room = OBS_CLIP * sd - np.abs(mu[KAPPA_CHANNELS])
    with np.errstate(divide="ignore"):
        return np.where(room > 0.0, 50.0 / np.maximum(room, 1e-300), np.inf)


def regime_table(agent: str, spec: GrammarSpec | None, corners: Table,
                 obs_rms: RunningMeanStd | dict | None, snapshot: str = "") -> RegimeTable:
    """q1 / q10 of R_seg from ``corners`` (the agent's own grammar; for a specialist, its
    circuit's rows) and R_clip from ``obs_rms`` (None before the final snapshot enters the
    manifest). Protocol 2.3."""
    q1, q10 = (float(q) for q in np.quantile(corners["R_seg"], [0.01, 0.10]))
    rc = () if obs_rms is None else tuple(float(r) for r in r_clip(obs_rms))
    return RegimeTable(agent=agent, grammar="" if spec is None else spec.name, q1=q1, q10=q10,
                       r_clip=rc, r_clip_max=max(rc) if rc else float("nan"),
                       snapshot=str(snapshot))


def regime(R_seg: float, rt: RegimeTable) -> str:
    """Design-time regime of a stimulus for one agent, from its R_seg (protocol 2.3):
    'clip' (R_seg <= r_clip_max), 'below' (r_clip_max < R_seg < q1), 'tail'
    (q1 <= R_seg < q10) or 'bulk' (R_seg >= q10). Without R_clip (nan) nothing is 'clip'.
    The binding kappa-clipped flag is the run-time one (measures: any preview kappa channel
    at |o| >= 10 before s_c1)."""
    if np.isfinite(rt.r_clip_max) and R_seg <= rt.r_clip_max:
        return "clip"
    if R_seg < rt.q1:
        return "below"
    if R_seg < rt.q10:
        return "tail"
    return "bulk"


# ------------------------------------------------------------------------- joint support

SUPPORT_FEATURES = ("log_R_seg", "theta_deg", "l_r", "L_c", "log_v_peak")
KNN_K = 10


def support_features(R_seg, theta_deg, l_r, L_c, v_peak) -> np.ndarray:
    """[n, 5] features in SUPPORT_FEATURES order (log of m and m/s; deg; m; m)."""
    cols = [np.log(np.asarray(R_seg, float)), np.abs(np.asarray(theta_deg, float)),
            np.asarray(l_r, float), np.asarray(L_c, float), np.log(np.asarray(v_peak, float))]
    return np.stack([np.atleast_1d(c) for c in cols], axis=1)


def table_features(t: Table) -> np.ndarray:
    """``support_features`` of a corner table's rows."""
    return support_features(t["R_seg"], t["theta_deg"], t["l_r"], t["L_c"], t["v_peak"])


def knn_log_density(train: np.ndarray, query: np.ndarray, sd: np.ndarray, k: int = KNN_K,
                    exclude_self: bool = False) -> np.ndarray:
    """log kNN density (k = 10) of ``query`` [m, d] under ``train`` [n, d], each feature
    divided by ``sd``: log(k / (n V_d r_k^d)) with V_d the unit-ball volume. With
    ``exclude_self`` the queries are training rows: each one's nearest neighbour (itself)
    is dropped and n becomes n - 1. Query rows with a non-finite feature get nan.
    Protocol 2.3."""
    from scipy.spatial import cKDTree

    sd = np.asarray(sd, float)
    tr = np.asarray(train, float) / sd
    tr = tr[np.all(np.isfinite(tr), axis=1)]
    q = np.atleast_2d(np.asarray(query, float)) / sd
    d = tr.shape[1]
    out = np.full(len(q), np.nan)
    ok = np.all(np.isfinite(q), axis=1)
    if not ok.any():
        return out
    kk = k + int(exclude_self)
    dist, _ = cKDTree(tr).query(q[ok], k=kk)
    r_k = np.maximum(np.atleast_2d(dist)[:, -1], 1e-12)
    n = len(tr) - int(exclude_self)
    log_vd = 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1.0)
    out[ok] = math.log(k) - math.log(n) - log_vd - d * np.log(r_k)
    return out




@jax.jit
def _v_peak_batch(kappa, i0, v0, i_end):
    """Rows of ``kappa`` [B, N]: the QSS profile with the road before knot i0 straightened
    and the speed pinned to min(v_lim, v0) at i0 (a start speed there: the forward pass
    reaches i0 at the cap from the straight and accelerates from v0 after it; values at and
    after i0 do not depend on the road before it). Returns max v over knots [i0, i_end]."""

    def one(k, a, v, b):
        idx = jnp.arange(k.shape[0])
        k_eff = jnp.where(idx < a, 0.0, k)
        prof = qss_profile_jax(k_eff, DS, F1_2025, knot_cap=jnp.where(idx == a, v, jnp.inf))
        return jnp.max(jnp.where((idx >= a) & (idx <= b), prof, -jnp.inf))

    return jax.vmap(one)(kappa, i0, v0, i_end)


def stimulus_v_peak_batch(kappa: np.ndarray, s0: np.ndarray, v0: np.ndarray,
                          s_c0: np.ndarray, ds: float = DS) -> np.ndarray:
    """``stimulus_v_peak`` for many (strip, approach) rows at once (kappa [B, N] samples,
    s0, v0, s_c0 [B]); one vmapped QSS call."""
    kappa = np.atleast_2d(np.asarray(kappa, np.float32))
    n = kappa.shape[1]
    i0 = np.clip(np.floor(np.asarray(s0, float) / ds), 0, n - 1).astype(np.int32)
    ic = np.clip(np.ceil(np.asarray(s_c0, float) / ds), 0, n - 1).astype(np.int32)
    out = _v_peak_batch(jnp.asarray(kappa), jnp.asarray(np.broadcast_to(i0, len(kappa))),
                        jnp.asarray(np.broadcast_to(np.asarray(v0, np.float32), len(kappa))),
                        jnp.asarray(np.broadcast_to(ic, len(kappa))))
    return np.asarray(out, np.float64)


def stimulus_v_peak(kappa: np.ndarray, s0: float, v0: float, s_c0: float,
                    ds: float = DS) -> float:
    """v_peak^QSS of a stimulus at one approach: the max of the centreline QSS profile
    (default car, the training bank's ``speed_profile_jax`` envelope, started at v0 on the
    knot at or before s0) over [s0, s_c0]. Same estimator as the training corners' v_peak.
    Protocol 2.3."""
    return float(stimulus_v_peak_batch(np.asarray(kappa)[None], np.array([s0]), np.array([v0]),
                                       np.array([s_c0]), ds)[0])


def tight_fast_rate(corners: Table, r_max: float = 20.0, v_min_kph: float = 250.0) -> float:
    """Corners per 100 km of strip that are tight (R_seg < r_max) and reached at
    v_peak >= v_min_kph (P1-tail rationale; secondary support covariate, 9.3). The driven
    length is the sum of ``strip_length`` over the table's distinct (source, strip)."""
    sel = (corners["R_seg"] < r_max) & (corners["v_peak"] >= v_min_kph / 3.6)
    keys = {}
    for src, st, ln in zip(corners["source"], corners["strip"], corners["strip_length"]):
        keys[(str(src), int(st))] = float(ln)
    km = sum(keys.values()) / 1000.0
    return float(sel.sum()) / km * 100.0 if km > 0 else float("nan")


def _quantiles(t: Table) -> dict[str, float]:
    qs = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
    return {f"q{round(100 * q)}": float(v) for q, v in zip(qs, np.quantile(t["R_seg"], qs))}


def _rates(t: Table) -> dict[str, float]:
    return {"tight_fast_r12": tight_fast_rate(t, 12.0), "tight_fast_r16": tight_fast_rate(t, 16.0),
            "tight_fast_r20": tight_fast_rate(t, 20.0),
            "tight_r20": tight_fast_rate(t, 20.0, 0.0)}


def _table_hash(t: Table) -> str:
    return io.sha256_arrays(*(np.asarray(t[k]) for k in sorted(t)))


def build_support(out_dir: Path, agents_obs_rms: dict[str, tuple[str, RunningMeanStd | dict]],
                  n_strips: int = SUPPORT_N_STRIPS, log=None) -> SupportTables:
    """Step 1: corner tables for the three grammars and R_DEV, l_typ, v_typ, every
    generalist's regime table (``AGENT_GRAMMAR``: G1-G3 and R18's replicates, each from its
    variant's grammar; R_clip for agents whose final is in ``agents_obs_rms`` as
    {agent: (snapshot path, obs_rms)}; a specialist listed there gets q1 / q10 of its
    circuit's corners and its R_clip), feature SDs, R_seg quantiles and tight-and-fast rates;
    written under ``out_dir`` (support.json, corners_<set>.npz). ``log`` is an optional
    ``io.StageLog`` (5 steps). About 1 CPU-minute."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, Table] = {}
    paths: dict[str, str] = {}
    for name, spec in GRAMMARS.items():
        tables[name] = corner_table(spec, n_strips)
        paths[name] = str(io.save_table(out_dir / f"corners_{name}.npz", tables[name]))
        if log is not None:
            log.step(note=f"corner table {name}: {len(tables[name]['R_seg'])} corners")
    rdev = rdev_corner_table()
    rdev_path = str(io.save_table(out_dir / "corners_rdev.npz", rdev))
    if log is not None:
        log.step(note=f"R_DEV corner table: {len(rdev['R_seg'])} corners")
    base = tables["base"]
    lt, vt = l_typ_table(base), v_typ_table(base)
    regimes = {}
    for agent, gname in AGENT_GRAMMAR.items():
        snap, rms = agents_obs_rms.get(agent, ("", None))
        regimes[agent] = regime_table(agent, GRAMMARS[gname], tables[gname], rms, snap)
    for agent, (snap, rms) in agents_obs_rms.items():
        if agent not in AGENT_GRAMMAR:
            own = io.select(rdev, rdev["source"] == SPECIALIST_TRACK)
            regimes[agent] = regime_table(agent, None, own, rms, snap)
    sd = np.std(table_features(base), axis=0, ddof=1)
    sets = dict(tables, rdev=rdev)
    st = SupportTables(
        l_typ=lt, v_typ=vt, regimes=regimes, feature_sd=tuple(float(x) for x in sd),
        corner_tables=paths, rdev_table=rdev_path,
        hashes={k: _table_hash(t) for k, t in sets.items()},
        quantiles={k: _quantiles(t) for k, t in sets.items()},
        rates={k: _rates(t) for k, t in sets.items()},
    )
    io.write_json(out_dir / "support.json", st)
    if log is not None:
        log.step(note="support.json")
    return st


def load_support(path: Path) -> SupportTables:
    """Read ``support.json`` (a file, or the directory holding it) back into SupportTables."""
    path = Path(path)
    if path.is_dir():
        path = path / "support.json"
    d = io.read_json(path)

    def f(x):
        if x is None:
            return float("nan")
        return float(x)   # numbers, and the strings "inf" / "-inf"

    lt = d["l_typ"]
    vt = d["v_typ"]
    regimes = {k: RegimeTable(agent=r["agent"], grammar=r["grammar"], q1=f(r["q1"]),
                              q10=f(r["q10"]), r_clip=tuple(f(x) for x in r["r_clip"]),
                              r_clip_max=f(r["r_clip_max"]), snapshot=r["snapshot"])
               for k, r in d["regimes"].items()}
    return SupportTables(
        l_typ=LTypTable(tuple(lt["r_nodes"]), tuple(lt["l_nodes"]), tuple(lt["counts"]),
                        f(lt["min_theta_deg"]), tuple(lt["l_nodes_all"])),
        v_typ=VTypTable(tuple(vt["bin_edges"]), {k: f(v) for k, v in vt["median"].items()},
                        {k: f(v) for k, v in vt["choice_kph"].items()},
                        {k: int(v) for k, v in vt["counts"].items()}, tuple(vt["pooled"]),
                        f(vt["min_theta_deg"])),
        regimes=regimes, feature_sd=tuple(f(x) for x in d["feature_sd"]),
        corner_tables=dict(d["corner_tables"]), rdev_table=d["rdev_table"],
        hashes=dict(d.get("hashes", {})), quantiles=d.get("quantiles", {}),
        rates=d.get("rates", {}))


V_TYP_CHOICES = tuple(v / 3.6 for v in V_TYP_CHOICES_KPH)   # m/s
