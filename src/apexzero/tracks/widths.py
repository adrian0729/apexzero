"""Statistical width model for procedural strips (GW-2, option 2a). Opt-in; off by default.

Real roads, as the R-dev evaluation measures them (per-side widths from the
``refine_reference`` line), differ from the default generator's slowly varying corridor in
two ways: the inside edge closes in toward a tight apex while the outside opens out, and
both sides vary on every scale from metres to kilometres. The copied option
(``fit_grammar(..., real_widths=True)``) shows the agent those profiles, but it copies the
evaluation circuits. This model draws statistically similar roads without copying any
circuit: it stores only summaries fitted on the fitting circuits (no profile), and each
strip draws its widths from them.

Per strip (``stat_widths``):

1. **Level.** A fitting circuit's median total width (uniform draw), times a log-normal
   jitter (``LEVEL_JITTER``), split evenly between the sides.
2. **Slow variation and multi-scale residual noise per side.** Two stationary Gaussian
   processes on the strip, one common to both sides (half the total-width variation) and
   one differential (half the left-right difference), so each side is level / 2 + common
   +/- differential. Their SDs per octave band of wavelength (4 m to 4,096 m,
   ``BAND_EDGES``) are the fitting circuits' pooled residual SDs; the bands from
   ``SLOW_FROM`` (256 m) up are the slow variation, the shorter ones the residual noise.
   One log-normal roughness multiplier per strip spreads strips from smooth to rough, as
   circuits are (it keeps the pooled variance).
3. **Corner coupling.** At every generated corner the reference line moves toward the
   inside by ``delta`` and the road widens by ``tau`` in total, along the corner's own
   curvature profile (the generator's trapezoid, 0 to 1): the inside width changes by
   ``-delta + tau / 2``, the outside by ``+delta + tau / 2``. ``delta`` and ``tau`` are
   drawn per corner from two-piece normals (q16, median, q84) fitted per radius bin
   (``RADIUS_EDGES``) on the fitting circuits' corners.
4. **Clip** each side to [``SIDE_MIN``, ``SIDE_MAX``]; everything is multiplied by the
   grammar's ``width_scale``. The generator then applies its usual inside cap.

The fit (``fit_width_model``) is a fixed procedure on the fitting circuits as evaluated
(2 m grid, ``refine_reference`` geometry): each corner of ``corners.segment`` gets least-
squares amplitudes of its inside and outside widths on its normalised |kappa| profile,
relative to the median of that side over the 100 m before and after it; delta and tau
follow from the two amplitudes. The fitted corner terms are removed and the residuals'
band variances are pooled over circuits (weighted by length). Lockbox circuits are refused.

No profile is stored: the model's arrays are one median per fitting circuit, two 5 x 3
quantile tables, two 10-band SD vectors and two scalars (``max_leaf_size``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.tracks.corners import segment, smooth
from apexzero.tracks.track import Track

GRID_DS = 2.0                                    # the generator's grid (generator.DS)
RADIUS_EDGES = (25.0, 50.0, 100.0, 200.0)        # radius bins: <25, 25-50, 50-100, 100-200, >=200 m
BAND_EDGES = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0)
SLOW_FROM = 256.0                                # bands from here up are the "slow variation"
SIDE_MIN, SIDE_MAX = 2.5, 16.0                   # per-side clip (m), before width_scale
LEVEL_JITTER = 0.08                              # as the default model's level jitter
MIN_CORNERS = 10                                 # a radius bin with fewer borrows a neighbour's
BASE_KNOTS = 50                                  # 100 m before and after a corner: its base
N_RADIUS = len(RADIUS_EDGES) + 1
N_BANDS = len(BAND_EDGES) - 1


class WidthModel(NamedTuple):
    median_width: jax.Array   # [Nt] median total width per fitting circuit (m)
    level_jitter: jax.Array   # log-normal sigma on the level
    delta_q: jax.Array        # [N_RADIUS, 3] inside shift (m): q16, q50, q84 per radius bin
    tau_q: jax.Array          # [N_RADIUS, 3] total widening (m): q16, q50, q84 per radius bin
    common_sd: jax.Array      # [N_BANDS] SD of half the total-width residual per band (m)
    diff_sd: jax.Array        # [N_BANDS] SD of half the left-right residual per band (m)
    roughness: jax.Array      # log-normal sigma of the per-strip multiplier on both SDs


def max_leaf_size(wm: WidthModel) -> int:
    """Largest number of values in any array of the model (the no-copy check's bound: a
    circuit's profile has over 1,000 knots on the 2 m grid)."""
    return max(int(np.size(leaf)) for leaf in jax.tree.leaves(wm))


# ------------------------------------------------------------------------------- fitting


def grid_profile(tr: Track, ds: float = GRID_DS):
    """(kappa, w_left, w_right) of a closed circuit on a ``ds`` grid (periodic linear
    interpolation, round(length / ds) knots), as ``generator.width_profiles`` resamples."""
    if not tr.closed:
        raise ValueError(f"{tr.name}: the width model is fitted on closed circuits")
    n = round(tr.length / ds)
    s = np.arange(n) * ds
    f = lambda a: np.interp(s, tr.s, a, period=tr.length)
    return f(tr.kappa), f(tr.w_left), f(tr.w_right)


def profile_track(kappa, w_left, w_right, closed: bool, ds: float = GRID_DS,
                  name: str = "profile") -> Track:
    """A Track view of a (kappa, w_left, w_right) profile for ``corners.segment``, which
    reads only s, kappa and closed (positions are not needed and are left at 0)."""
    kappa = np.asarray(kappa, float)
    s = np.arange(len(kappa)) * ds
    z = np.zeros_like(s)
    return Track(name, s, z, z, z, kappa, np.asarray(w_left, float), np.asarray(w_right, float),
                 closed=closed)


def corner_terms(kappa: np.ndarray, w_left: np.ndarray, w_right: np.ndarray,
                 ds: float = GRID_DS):
    """Per-corner rows (r_min, delta, tau) of a closed profile, and the fitted corner terms
    (comp_left, comp_right) per knot. For each corner of ``segment`` (start a, end b):
    g = |kappa smoothed over 10 m| on [a, b], divided by its maximum; a_in and a_out are the
    least-squares amplitudes of (w - base) on g, where base is that side's median over the
    BASE_KNOTS knots before a and after b; delta = (a_out - a_in) / 2, tau = a_in + a_out."""
    kappa, w_left, w_right = (np.asarray(a, float) for a in (kappa, w_left, w_right))
    n = len(kappa)
    ks = smooth(kappa, ds, 10.0, True)
    comp_l, comp_r = np.zeros(n), np.zeros(n)
    rows = []
    for c in segment(profile_track(kappa, w_left, w_right, True, ds)).corners:
        a, b = round(c.s_start / ds), round(c.s_end / ds)
        if b < a:                                   # the corner spans s = 0
            b += n
        idx = np.arange(a, b + 1) % n
        g = np.abs(ks[idx])
        if g.max() <= 0:
            continue
        g = g / g.max()
        around = np.r_[(a - np.arange(1, BASE_KNOTS + 1)) % n, (b + np.arange(1, BASE_KNOTS + 1)) % n]
        w_in, w_out = (w_left, w_right) if c.sign > 0 else (w_right, w_left)
        gg = float(np.sum(g * g))
        a_in = float(np.sum((w_in[idx] - np.median(w_in[around])) * g) / gg)
        a_out = float(np.sum((w_out[idx] - np.median(w_out[around])) * g) / gg)
        if c.sign > 0:
            comp_l[idx] += a_in * g
            comp_r[idx] += a_out * g
        else:
            comp_r[idx] += a_in * g
            comp_l[idx] += a_out * g
        rows.append((c.r_min, 0.5 * (a_out - a_in), a_in + a_out))
    return np.asarray(rows, dtype=float).reshape(-1, 3), comp_l, comp_r


def band_variance(x: np.ndarray, ds: float = GRID_DS) -> np.ndarray:
    """Variance of a periodic profile per wavelength band (``BAND_EDGES``), from its FFT
    (mean removed); the bands' sum is the variance between 4 m and 4,096 m."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    p = 2.0 * np.abs(np.fft.rfft(x)) ** 2 / n**2
    k = np.arange(len(p))
    if n % 2 == 0:
        p[-1] /= 2.0
    lam = np.where(k > 0, n * ds / np.maximum(k, 1), np.inf)
    band = np.searchsorted(np.asarray(BAND_EDGES), lam, side="right") - 1
    ok = (k > 0) & (band >= 0) & (band < N_BANDS)
    return np.bincount(band[ok], weights=p[ok], minlength=N_BANDS)


def _bin_quantiles(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(q16, q50, q84) of v per radius bin; a bin with fewer than MIN_CORNERS corners takes
    in its neighbours (the one with more corners first) until it has enough."""
    b = np.searchsorted(np.asarray(RADIUS_EDGES), r, side="right")
    counts = np.bincount(b, minlength=N_RADIUS)
    out = np.zeros((N_RADIUS, 3))
    for i in range(N_RADIUS):
        lo = hi = i
        while counts[lo:hi + 1].sum() < MIN_CORNERS and (lo > 0 or hi < N_RADIUS - 1):
            left = counts[lo - 1] if lo > 0 else -1
            right = counts[hi + 1] if hi < N_RADIUS - 1 else -1
            if right >= left:
                hi += 1
            else:
                lo -= 1
        sel = v[(b >= lo) & (b <= hi)]
        out[i] = np.quantile(sel, [0.16, 0.5, 0.84]) if len(sel) else 0.0
    return out


def fit_width_model(tracks: list[Track], ds: float = GRID_DS) -> WidthModel:
    """Fit the model on closed circuits as evaluated (pass ``refine_reference`` tracks).
    Refuses lockbox circuits. Every fitted value is a summary: one median per circuit, two
    5 x 3 quantile tables and two 10-band SD vectors (no profile is kept)."""
    from apexzero.tracks.generator import _refuse_lockbox  # generator imports this module

    _refuse_lockbox(tracks)
    medians, corners, v_c, v_d, n_knots = [], [], [], [], []
    for tr in tracks:
        k, wl, wr = grid_profile(tr, ds)
        medians.append(float(np.median(wl + wr)))
        rows, comp_l, comp_r = corner_terms(k, wl, wr, ds)
        corners.append(rows)
        rl, rr = wl - comp_l, wr - comp_r
        v_c.append(band_variance(0.5 * (rl + rr), ds))
        v_d.append(band_variance(0.5 * (rl - rr), ds))
        n_knots.append(len(k))
    w = np.asarray(n_knots, float) / np.sum(n_knots)
    v_c, v_d = np.asarray(v_c), np.asarray(v_d)
    log_sd = 0.5 * np.log(v_c.sum(axis=1) + v_d.sum(axis=1))
    rows = np.concatenate(corners)
    f = lambda a: jnp.asarray(np.asarray(a, dtype=np.float32))
    return WidthModel(
        median_width=f(medians), level_jitter=f(LEVEL_JITTER),
        delta_q=f(_bin_quantiles(rows[:, 0], rows[:, 1])),
        tau_q=f(_bin_quantiles(rows[:, 0], rows[:, 2])),
        common_sd=f(np.sqrt(w @ v_c)), diff_sd=f(np.sqrt(w @ v_d)),
        roughness=f(np.std(log_sd) if len(tracks) > 1 else 0.0),
    )


# ------------------------------------------------------------------------------ sampling


def _band_gains(n: int, ds: float, sd: jax.Array) -> jax.Array:
    """Per-rfft-bin gains that turn unit white noise of length n into a periodic process
    with SD ``sd[b]`` in band b (bins outside every band, the mean and Nyquist get 0)."""
    k = np.arange(n // 2 + 1)
    lam = np.where(k > 0, n * ds / np.maximum(k, 1), np.inf)
    band = np.searchsorted(np.asarray(BAND_EDGES), lam, side="right") - 1
    ok = (k > 0) & (2 * k < n) & (band >= 0) & (band < N_BANDS)
    counts = np.bincount(band[ok], minlength=N_BANDS)
    scale = np.zeros(len(k))
    scale[ok] = np.sqrt(n / (2.0 * counts[band[ok]]))
    return jnp.asarray(scale, sd.dtype) * sd[np.where(ok, band, 0)]


def band_noise(key: jax.Array, sd: jax.Array, n: int, ds: float = GRID_DS) -> jax.Array:
    """Zero-mean Gaussian process of length n with SD ``sd[b]`` in wavelength band b."""
    z = jax.random.normal(key, (n,), sd.dtype)
    return jnp.fft.irfft(_band_gains(n, ds, sd) * jnp.fft.rfft(z), n)


def two_piece(z: jax.Array, q: jax.Array) -> jax.Array:
    """Two-piece normal with quantiles (q16, q50, q84) = q (per row) from z ~ N(0, 1)."""
    return q[..., 1] + jnp.where(z < 0, q[..., 1] - q[..., 0], q[..., 2] - q[..., 1]) * z


def stat_widths(key: jax.Array, wm: WidthModel, width_scale: jax.Array, corner: jax.Array,
                shape: jax.Array, sign: jax.Array, k_peak: jax.Array, ds: float = GRID_DS):
    """(w_left, w_right) of one strip before the inside cap.

    ``corner`` [n] is each knot's corner index, ``shape`` [n] its normalised curvature
    profile (0 off corners, up to 1 on the arc), ``sign`` [m] and ``k_peak`` [m] each
    corner's turning direction and peak curvature (the generator's own values)."""
    n = shape.shape[0]
    k = jax.random.split(key, 7)
    ti = jax.random.randint(k[0], (), 0, wm.median_width.shape[0])
    level = wm.median_width[ti] * jnp.exp(wm.level_jitter * jax.random.normal(k[1], ()))
    rough = wm.roughness * jax.random.normal(k[2], ())
    mult = jnp.exp(rough - wm.roughness**2)                  # E[mult^2] = 1
    common = mult * band_noise(k[3], wm.common_sd, n, ds)
    diff = mult * band_noise(k[4], wm.diff_sd, n, ds)

    m = k_peak.shape[0]
    rbin = jnp.searchsorted(jnp.asarray(RADIUS_EDGES, k_peak.dtype), 1.0 / k_peak, side="right")
    delta = two_piece(jax.random.normal(k[5], (m,)), wm.delta_q[rbin])
    tau = two_piece(jax.random.normal(k[6], (m,)), wm.tau_q[rbin])
    d_in = (-delta + 0.5 * tau)[corner] * shape
    d_out = (delta + 0.5 * tau)[corner] * shape
    left_in = sign[corner] > 0
    wl = 0.5 * level + common + diff + jnp.where(left_in, d_in, d_out)
    wr = 0.5 * level + common - diff + jnp.where(left_in, d_out, d_in)
    clip = lambda w: width_scale * jnp.clip(w, SIDE_MIN, SIDE_MAX)
    return clip(wl), clip(wr)
