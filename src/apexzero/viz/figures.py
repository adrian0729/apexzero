"""Static figures from a trajectory export (``apexzero.viz.export`` documents).

The figures read only the exported document, never live objects, so the exact figures
for a release can be redrawn from the committed JSON, and re-running them on a final
agent's export needs no code change.

- :func:`track_map` - the agent's and the optimum's lines coloured by speed (one shared
  scale), plus a zoom on the corner that costs the agent the most time.
- :func:`speed_and_delta` - speed against distance for both ghosts, and the cumulative
  time delta Δt(s) beneath it (two panels, one shared distance axis, never two y-scales).
- :func:`sector_losses` - time lost or gained per corner, charged with the carry-over of
  a slow exit when the export has it, against what each sector measured; the bars add
  up to the lap delta.

An optimum without the CERTIFIED badge is labelled "not certified" wherever it is named.

Colours follow the data-viz reference palette: categorical slots for ghost identity, one
blue ramp for speed (magnitude), and the blue/red diverging pair for gain/loss (polarity).

Figures are built on ``matplotlib.figure.Figure`` directly, not pyplot, so importing this
module never switches the caller's backend or leaks open figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from apexzero.viz.export import ghost_arrays

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TRACK_FILL = "#ecebe6"
BAND = "#f3f2ee"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4, fixed order
GAIN, LOSS = "#2a78d6", "#e34948"                       # diverging poles
SPEED_RAMP = LinearSegmentedColormap.from_list(
    "speed", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])

KPH = 3.6
TUMFTM_ATTRIBUTION = "Track geometry: TUMFTM racetrack-database (LGPL-3.0)"


def _style() -> dict:
    return {
        # Arial before Helvetica: matplotlib cannot load some glyphs (U+2212) from
        # macOS's Helvetica.ttc. DejaVu Sans is the fallback everywhere else.
        "font.family": ["Arial", "DejaVu Sans"], "font.size": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 8.5, "text.color": INK,
    }


def _comparison(doc: dict, ghost: str | None):
    comps = doc.get("comparisons", [])
    if not comps:
        raise ValueError("document has no comparison (export it with a reference ghost)")
    if ghost is None:
        return comps[0]
    for c in comps:
        if c["ghost"] == ghost:
            return c
    raise KeyError(ghost)


def _label(doc: dict, gid: str) -> str:
    """A ghost's label, tagged "not certified" for an optimum without the CERTIFIED badge
    (a continuous-control or best-known lap, not the replay-verified 20 Hz answer key)."""
    g = next(g for g in doc["ghosts"] if g["id"] == gid)
    if g["kind"] == "ocp" and g.get("badge") != "CERTIFIED":
        return f"{g['label']} · not certified"
    return g["label"] if not g.get("badge") else f"{g['label']} · {g['badge']}"


def _sector_costs(comp: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    """(charged, measured, split available): what each corner costs, carry-over-aware when
    the export has a ``sector_split``, and what the clock measured inside each sector."""
    measured = np.asarray(comp["sector_delta_s"], float)
    split = comp.get("sector_split")
    if split is None:
        return measured, measured, False
    return np.asarray(split["charged_s"], float), measured, True


def _lap(doc: dict, gid: str) -> float:
    return next(g["lap_time_s"] for g in doc["ghosts"] if g["id"] == gid)


def _track_patch(ax, tr: dict, **kw) -> None:
    left = np.c_[tr["left"]["x"], tr["left"]["y"]]
    right = np.c_[tr["right"]["x"], tr["right"]["y"]]
    if tr["closed"]:
        # Ribbon as quads between the edges (one polygon would self-overlap where the
        # circuit crosses itself, e.g. a figure of eight).
        j = np.r_[1:len(left), 0]
        quads = np.stack([left, left[j], right[j], right], axis=1)
        ax.add_collection(PolyCollection(quads, facecolors=TRACK_FILL, edgecolors=TRACK_FILL,
                                         linewidths=0.3, **kw))
    else:
        ax.add_patch(Polygon(np.vstack([left, right[::-1]]), closed=True,
                             facecolor=TRACK_FILL, edgecolor="none", **kw))
    for e in (left, right):
        xy = np.vstack([e, e[:1]]) if tr["closed"] else e
        ax.plot(xy[:, 0], xy[:, 1], color=AXIS, lw=0.5)


def _speed_line(ax, g: dict, norm, lw=2.0):
    pts = np.c_[g["x"], g["y"]]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=SPEED_RAMP, norm=norm, linewidths=lw, capstyle="round")
    lc.set_array(0.5 * (g["speed"][:-1] + g["speed"][1:]) * KPH)
    ax.add_collection(lc)
    return lc


def _start_line(ax, tr: dict, overhang: float = 25.0) -> None:
    """The start line, drawn past both edges so it reads at map scale."""
    x0, y0, th = tr["x"][0], tr["y"][0], tr["heading"][0]
    nx, ny = -np.sin(th), np.cos(th)
    a = tr["w_left"][0] + overhang if "w_left" in tr else overhang
    b = tr["w_right"][0] + overhang if "w_right" in tr else overhang
    ax.plot([x0 + a * nx, x0 - b * nx], [y0 + a * ny, y0 - b * ny], color=INK, lw=1.6,
            solid_capstyle="butt", zorder=5)


def _sector_labels(ax, tr: dict, offset: float = 45.0) -> None:
    s = np.asarray(tr["s"])
    x, y, th = (np.asarray(tr[k]) for k in ("x", "y", "heading"))
    kap = np.asarray(tr["kappa"])
    for sec in tr["sectors"]:
        i = int(np.argmin(np.abs(s - sec["s_apex"])))
        # Outside of the turn: away from the centre of curvature.
        side = -np.sign(kap[i]) or 1.0
        nx, ny = -np.sin(th[i]), np.cos(th[i])
        ax.text(x[i] + side * offset * nx, y[i] + side * offset * ny, sec["name"],
                ha="center", va="center", fontsize=7, color=INK_2)


def track_map(doc: dict, ghost: str | None = None, zoom_margin: float = 50.0) -> Figure:
    """Both lines coloured by speed, and the corner complex of the costliest sector with
    both lines overlaid (``zoom_margin`` metres of track before its entry and after its
    exit)."""
    comp = _comparison(doc, ghost)
    gid, rid = comp["ghost"], comp["reference"]
    tr = doc["track"]
    g, r = ghost_arrays(doc, gid), ghost_arrays(doc, rid)
    vmin = min(g["speed"].min(), r["speed"].min()) * KPH
    vmax = max(g["speed"].max(), r["speed"].max()) * KPH
    norm = Normalize(vmin, vmax)
    loss, _, split = _sector_costs(comp)
    worst = int(np.argmax(loss))
    sec = tr["sectors"][worst]

    with rc_context(_style()):
        fig = Figure(figsize=(15, 6.0), layout="constrained")
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.62])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        lc = None
        for ax, (gg, key) in zip(axes[:2], ((g, gid), (r, rid)), strict=True):
            _track_patch(ax, tr)
            lc = _speed_line(ax, gg, norm)
            _start_line(ax, tr)
            _sector_labels(ax, tr)
            ax.set_title(f"{_label(doc, key)}  ·  {_lap(doc, key):.3f} s", loc="left")
            ax.set_aspect("equal")
            ax.axis("off")
            ax.autoscale_view()
        cb = fig.colorbar(lc, ax=axes[:2], orientation="horizontal", fraction=0.04,
                          pad=0.02, aspect=50)
        cb.set_label("speed (km/h)", color=INK_2)
        cb.outline.set_visible(False)

        # Zoom on the costliest corner complex: identity by categorical colour, not speed.
        az = axes[2]
        _track_patch(az, tr)
        for gg, key, col in ((r, rid, SERIES[1]), (g, gid, SERIES[0])):
            az.plot(gg["x"], gg["y"], color=col, lw=1.8, label=_label(doc, key),
                    solid_capstyle="round")
        sl = np.asarray(tr["s"])
        L = tr["length_m"]
        a = sec.get("s_entry", sec["s_start"]) - zoom_margin
        b = sec.get("s_exit", sec["s_end"]) + zoom_margin
        rel = np.mod(sl - a, L) if tr["closed"] else sl - a
        inside = (rel >= 0) & (rel <= b - a)
        pts = np.r_[np.c_[tr["left"]["x"], tr["left"]["y"]][inside],
                    np.c_[tr["right"]["x"], tr["right"]["y"]][inside]]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = 0.08 * max(hi - lo)
        az.set_xlim(lo[0] - pad, hi[0] + pad)
        az.set_ylim(lo[1] - pad, hi[1] + pad)
        az.set_aspect("equal")
        az.axis("off")
        az.set_title(f"Costliest corner {sec['name']}: {loss[worst]:+.3f} s"
                     + (" charged" if split else ""), loc="left")
        az.legend(loc="upper left", bbox_to_anchor=(0.0, -0.01), ncols=1)
        fig.suptitle(f"{doc['track']['name']}: racing lines coloured by speed", x=0.01, ha="left",
                     fontsize=12, fontweight="bold", color=INK)
    return fig


def _sector_bands(ax, doc: dict, label: bool) -> None:
    tr = doc["track"]
    L = tr["length_m"]
    for i, sec in enumerate(tr["sectors"]):
        a, b = sec["s_start"], sec["s_end"]
        spans = [(a, min(b, L))] + ([(0.0, b - L)] if b > L else [])
        for lo, hi in spans:
            if i % 2 == 0:
                ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
        if label:
            mid = 0.5 * (a + min(b, L))
            ax.text(mid, 1.0, sec["name"], transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7, color=MUTED)


def speed_and_delta(doc: dict, ghost: str | None = None) -> Figure:
    """Speed vs distance (both ghosts) over the cumulative time delta of ``ghost``."""
    comp = _comparison(doc, ghost)
    gid, rid = comp["ghost"], comp["reference"]
    g, r = ghost_arrays(doc, gid), ghost_arrays(doc, rid)
    L = doc["track"]["length_m"]
    with rc_context(_style()):
        fig = Figure(figsize=(12, 6.2), layout="constrained")
        a1, a2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
        for ax in (a1, a2):
            _sector_bands(ax, doc, label=ax is a1)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        a1.plot(r["s"], r["speed"] * KPH, color=SERIES[1], lw=1.3,
                label=f"{_label(doc, rid)}  {_lap(doc, rid):.3f} s")
        a1.plot(g["s"], g["speed"] * KPH, color=SERIES[0], lw=1.3,
                label=f"{_label(doc, gid)}  {_lap(doc, gid):.3f} s")
        a1.set_ylabel("speed (km/h)")
        a1.legend(loc="lower right", ncols=2, bbox_to_anchor=(1.0, 1.06))
        a1.set_title(f"{doc['track']['name']}: speed and time delta along the lap", pad=22)

        s = np.asarray(comp["s"])
        d = np.asarray(comp["delta_t"])
        a2.axhline(0.0, color=AXIS, lw=0.8)
        a2.plot(s, d, color=SERIES[0], lw=1.4)
        a2.annotate(f"{d[-1]:+.3f} s", (s[-1], d[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
        a2.set_ylabel(f"Δt (s)\n> 0: {gid} behind")
        a2.set_xlabel("distance from the start line (m)")
        a2.set_xlim(0, L)
    return fig


def sector_losses(doc: dict, ghost: str | None = None, labelled: int = 3) -> Figure:
    """Time lost (> 0) or gained (< 0) per corner; bars sum to the lap delta.

    With a carry-over split in the export, the filled bar is what the corner is
    *charged* (time lost on flat-out stretches after it moved back to it) and the
    outline is what the clock *measured* inside the sector, so a flat-out bend that
    only inherits a slow exit shows an empty outline with no bar. Tick labels give the
    reference ghost's minimum speed in the sector (the reference-line radius would
    call a straightened kink a hairpin)."""
    comp = _comparison(doc, ghost)
    secs = doc["track"]["sectors"]
    vals, measured, split = _sector_costs(comp)
    speed = comp.get("sector_speed")
    if speed is not None:
        names = [f"{s['name']}\n{v * KPH:.0f} km/h"
                 for s, v in zip(secs, speed["ref_vmin_mps"], strict=True)]
    else:
        names = [s["name"] if s.get("r_min") is None else f"{s['name']}\n{s['r_min']:.0f} m"
                 for s in secs]
    x = np.arange(len(vals))
    with rc_context(_style()):
        fig = Figure(figsize=(max(6.0, 0.62 * len(vals) + 2.0), 4.4), layout="constrained")
        ax = fig.subplots()
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.bar(x, vals, width=0.42, color=np.where(vals >= 0, LOSS, GAIN), lw=0,
               label="charged to the corner" if split else None)
        if split:
            ax.bar(x, measured, width=0.66, facecolor="none", edgecolor=INK_2, lw=0.8,
                   label="measured inside the sector")
            ax.legend(loc="upper left", ncols=2)
        ax.axhline(0.0, color=AXIS, lw=0.8)
        # Label selectively: the largest losses and every gain.
        top = set(np.argsort(-vals)[:labelled].tolist()) | set(np.nonzero(vals < -5e-4)[0].tolist())
        both = np.r_[vals, measured]
        span = max(np.abs(both).max(), 1e-6)
        for i in sorted(top):
            v = vals[i]
            y = max(v, measured[i]) if v >= 0 else min(v, measured[i])
            ax.text(i, y + np.sign(v or 1.0) * 0.03 * span, f"{v:+.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
        lo, hi = min(both.min(), 0.0), max(both.max(), 0.0)
        ax.set_ylim(lo - 0.12 * span * (lo < 0), hi + (0.22 if split else 0.12) * span)
        ax.set_xticks(x, names, fontsize=7.5)
        ax.set_ylabel(f"time vs {comp['reference']} (s)\n> 0: lost")
        what = "charged per corner" if split else "per sector"
        ax.set_title(f"{doc['track']['name']}: where {_label(doc, comp['ghost'])} loses time",
                     pad=24)
        ax.text(0.0, 1.02, f"against {_label(doc, comp['reference'])};  {what}, sums to "
                f"{vals.sum():+.3f} s (lap delta {comp['lap_delta_s']:+.3f} s)",
                transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
        lines = [("Sector = corner complex plus half of each neighbouring straight. Names are "
                  "this segmentation's, not official turn numbers.")]
        if split:
            lines.append("Charged: time gained or lost while both cars are flat out is moved "
                         "back to the corner where that stretch began (a slow exit costs time "
                         "on the straight after it).")
        lines.append("Label: the reference's minimum speed in the sector." if speed is not None
                     else "Label: minimum radius of the reference line.")
        ax.text(0.0, -0.2, "\n".join(lines), transform=ax.transAxes, fontsize=7, color=MUTED,
                va="top")
    return fig


def add_footer(fig: Figure, text: str, height_in: float = 0.24) -> None:
    """Reserve a strip of ``height_in`` inches along the bottom of a constrained-layout
    figure and write ``text`` in it (e.g. the track data's licence)."""
    strip = height_in / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0.0, strip, 1.0, 1.0 - strip))
    with rc_context(_style()):
        fig.text(0.006, 0.5 * strip, text, fontsize=7, color=MUTED, ha="left", va="center")


def save_figures(doc: dict, out_dir: str | Path, stem: str, dpi: int = 160,
                 footer: str | None = None) -> list[Path]:
    """Render all three figures to ``<out_dir>/<stem>-{map,speed-delta,sectors}.png``, each
    with ``footer`` along its bottom edge when given."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, fn in (("map", track_map), ("speed-delta", speed_and_delta),
                       ("sectors", sector_losses)):
        fig = fn(doc)
        if footer:
            add_footer(fig, footer)
        p = out / f"{stem}-{suffix}.png"
        with rc_context(_style()):
            fig.savefig(p, dpi=dpi)
        paths.append(p)
    return paths
