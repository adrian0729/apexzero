"""Procedural F1-like track strips, generated on-device in JAX.

A strip is an open, 8 km curvature profile kappa(s) on a 2 m grid, built from tokens:

    long straight (500-900 m), then [corner, straight] repeated

Corners are drawn *jointly* (turning angle, minimum radius, length) from the corners of the
development circuits (R-dev only; the lockbox circuits are never used), with log-normal
jitter. Each corner is a trapezoidal curvature profile: a clothoid ramp in, a constant-
radius arc, a clothoid ramp out. The turning direction alternates with a probability that
depends on the gap to the previous corner, which reproduces chicanes and esses. Widths are
drawn from the development circuits' median widths with low-frequency variation and
left/right asymmetry, and the inside width is capped so that (w_in - 1) * kappa <= 0.6
keeps the Frenet frame non-singular wherever the car can reach.

Because the simulator is local in Frenet coordinates, open strips need no closure or
self-intersection checks.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.tracks.corners import segment
from apexzero.tracks.spline import coeffs_open_jax, values_from_coeffs
from apexzero.tracks.track import Track
from apexzero.vehicle.sim import TrackArrays

N_POINTS = 4096
DS = 2.0
MAX_SEGMENTS = 128


class Grammar(NamedTuple):
    corner_angle: jax.Array   # [Nc] |turning angle| (rad)
    corner_rmin: jax.Array    # [Nc] minimum radius (m)
    corner_len: jax.Array     # [Nc] corner length (m)
    straight_len: jax.Array   # [Ns] gap lengths between corners (m)
    p_alt_short: jax.Array    # P(direction alternates | 20 m <= gap < 60 m); forced below 20 m
    p_alt_long: jax.Array     # P(direction alternates | gap >= 60 m)
    width: jax.Array          # [Nt] median total width per development circuit (m)
    jitter: jax.Array         # log-normal sigma on angle, radius, length, straights
    r_floor: jax.Array        # minimum allowed radius (m)
    margin: jax.Array         # cap on (w_in - 1) * kappa
    ramp_min: jax.Array       # minimum clothoid length (m)
    gap_scale: jax.Array      # calibration: sampled gaps -> gaps as the segmentation sees them
    corner_logits: jax.Array  # [Nc] sampling log-weights over the corner pool (0 = uniform)
    width_scale: jax.Array    # multiplies the sampled total width


def fit_grammar(tracks: list[Track], min_angle_deg: float = 0.0, jitter: float = 0.15,
                r_floor: float = 8.0, margin: float = 0.6, ramp_min: float = 10.0,
                gap_scale: float = 1.0) -> Grammar:
    ang, rmin, clen, straights, alt_s, alt_l, widths = [], [], [], [], [], [], []
    for tr in tracks:
        seg = segment(tr)
        cs = [c for c in seg.corners if abs(np.degrees(c.angle)) >= min_angle_deg]
        ang += [abs(c.angle) for c in cs]
        rmin += [c.r_min for c in cs]
        clen += [c.length for c in cs]
        straights += list(seg.straights)
        signs = np.array([c.sign for c in seg.corners])
        gaps = seg.straights[: len(signs) - 1]
        alt = signs[1:] != signs[:-1]
        # Gaps under the segmentation's 20 m merge distance always alternate by
        # construction (same-direction corners that close were merged), so the generator
        # forces alternation there and only fits the probability for longer gaps.
        alt_s += list(alt[(gaps >= 20.0) & (gaps < 60.0)])
        alt_l += list(alt[gaps >= 60.0])
        widths.append(float(np.median(tr.w_left + tr.w_right)))
    f = lambda a: jnp.asarray(np.asarray(a, dtype=np.float32))
    return Grammar(
        corner_angle=f(ang), corner_rmin=f(rmin), corner_len=f(clen), straight_len=f(straights),
        p_alt_short=f(np.mean(alt_s)), p_alt_long=f(np.mean(alt_l)), width=f(widths),
        jitter=f(jitter), r_floor=f(r_floor), margin=f(margin), ramp_min=f(ramp_min),
        gap_scale=f(gap_scale), corner_logits=f(np.zeros(len(ang))), width_scale=f(1.0),
    )


def tilted(g: Grammar, alpha: float = 1.0, gap_scale: float = 0.8) -> Grammar:
    """Hairpin-heavy, technical variant still inside the training support: corners are
    re-weighted by (1 / r_min)^alpha and gaps shortened. Used for the A_tail level set."""
    logits = -alpha * jnp.log(g.corner_rmin)
    return g._replace(corner_logits=logits - logits.max(), gap_scale=jnp.asarray(gap_scale, jnp.float32))


def out_of_distribution(g: Grammar) -> Grammar:
    """Shifted generator for P-OOD / A_OOD: narrower tracks (about 6.5-8 m), tighter hairpins
    (radius floor 6 m, heavier weight on slow corners) and longer straights."""
    logits = -1.5 * jnp.log(g.corner_rmin)
    return g._replace(
        corner_logits=logits - logits.max(), r_floor=jnp.asarray(6.0, jnp.float32),
        width_scale=jnp.asarray(0.65, jnp.float32), gap_scale=jnp.asarray(1.6, jnp.float32),
    )


def _lowfreq(key, s, amplitude, wl_min=300.0, wl_max=1500.0, n=3):
    k1, k2, k3 = jax.random.split(key, 3)
    wl = jax.random.uniform(k1, (n,), minval=wl_min, maxval=wl_max)
    ph = jax.random.uniform(k2, (n,), minval=0.0, maxval=2 * jnp.pi)
    amp = jax.random.uniform(k3, (n,), minval=0.3, maxval=1.0)
    sig = jnp.sum(amp[:, None] * jnp.sin(2 * jnp.pi * s[None, :] / wl[:, None] + ph[:, None]), 0)
    return amplitude * sig / jnp.sum(amp)


def generate(key: jax.Array, g: Grammar, n_points: int = N_POINTS, ds: float = DS,
             max_segments: int = MAX_SEGMENTS):
    """Return (kappa, w_left, w_right), each [n_points], for one open strip."""
    keys = jax.random.split(key, 10)
    m = max_segments
    jit = lambda k, shape: jnp.exp(g.jitter * jax.random.normal(k, shape))

    # Corners: joint draw of (angle, r_min, length) from the empirical pool, jittered.
    ci = jax.random.categorical(keys[0], g.corner_logits, shape=(m,))
    angle = g.corner_angle[ci] * jit(keys[1], (m,))
    radius = jnp.maximum(g.corner_rmin[ci] * jit(keys[2], (m,)), g.r_floor)
    clen = g.corner_len[ci] * jit(keys[3], (m,))

    # Straights: empirical gaps, jittered; the first one is a long run-up.
    si = jax.random.randint(keys[4], (m,), 0, g.straight_len.shape[0])
    gap = g.gap_scale * g.straight_len[si] * jit(keys[5], (m,))
    first = jax.random.uniform(keys[6], (), minval=500.0, maxval=900.0)
    gap = gap.at[0].set(first)

    # Turning direction: alternation probability depends on the preceding gap.
    p_alt = jnp.where(gap[1:] < 20.0, 1.0, jnp.where(gap[1:] < 60.0, g.p_alt_short, g.p_alt_long))
    flips = jax.random.uniform(keys[7], (m - 1,)) < p_alt
    s0 = jnp.where(jax.random.uniform(keys[8], ()) < 0.5, -1.0, 1.0)
    sign = s0 * jnp.concatenate([jnp.ones(1), jnp.cumprod(jnp.where(flips, -1.0, 1.0))])

    # Trapezoidal curvature profile per corner: ramp l_r, arc l_c, ramp l_r.
    k_peak = 1.0 / radius
    arc_equiv = angle / k_peak                     # length of an equivalent pure arc
    l_r = jnp.clip(clen - arc_equiv, g.ramp_min, None)
    tri = arc_equiv < l_r                          # too little angle for the ramps: triangle
    k_peak = jnp.where(tri, angle / l_r, k_peak)
    l_c = jnp.where(tri, 0.0, arc_equiv - l_r)
    c_total = l_c + 2.0 * l_r

    # Lay out [gap_0, corner_0, gap_1, corner_1, ...] along s.
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

    # Widths: development-circuit median, low-frequency variation, asymmetry, Frenet cap.
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
    return kappa, wl, wr


def strip_arrays(kappa, w_left, w_right, ds: float = DS) -> TrackArrays:
    """Integrate heading and position for an open strip and package it for the simulator."""
    theta = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * ds)])
    c, s = jnp.cos(theta), jnp.sin(theta)
    x = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (c[1:] + c[:-1]) * ds)])
    y = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (s[1:] + s[:-1]) * ds)])
    n = kappa.shape[0]
    return TrackArrays(
        kappa_c=coeffs_open_jax(kappa), w_left_c=coeffs_open_jax(w_left),
        w_right_c=coeffs_open_jax(w_right), x=x, y=y, theta=theta,
        ds=jnp.asarray(ds, kappa.dtype), n_points=jnp.asarray(n, jnp.int32),
        closed=jnp.asarray(False),
    )


def generate_strip(key: jax.Array, g: Grammar) -> TrackArrays:
    return strip_arrays(*generate(key, g))


def to_track(arr: TrackArrays, name: str = "strip") -> Track:
    """NumPy Track view of a generated strip (for analysis, OCP and plotting)."""
    a = jax.tree.map(np.asarray, arr)
    n = int(a.n_points)
    s = np.arange(n) * float(a.ds)
    return Track(name=name, s=s, x=a.x[:n], y=a.y[:n], theta=a.theta[:n],
                 kappa=values_from_coeffs(a.kappa_c, n), w_left=values_from_coeffs(a.w_left_c, n),
                 w_right=values_from_coeffs(a.w_right_c, n), closed=False,
                 meta={"source": "procedural"})
