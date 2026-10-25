"""Render mode: a trajectory export drawn frame by frame into an image sequence or a clip.

Like the figures and the viewer, the renderer reads only an ``apexzero.trajectory``
document (``apexzero.viz.export``), so a clip can be redrawn from an export without
re-running a policy or an optimiser. It needs nothing beyond the ``viz`` group: frames are
drawn on matplotlib's Agg canvas and animations are written with Pillow, which matplotlib
already depends on. No browser and no ffmpeg.

- :func:`write_frames` - a lossless PNG sequence (``frame_00000.png``, ...) and a
  ``frames.json`` manifest. This is the master copy: any encoder can make a video of it
  later, e.g. ``ffmpeg -framerate 30 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p
  clip.mp4``.
- :func:`write_animation` - an animated WebP, GIF or APNG. Pillow holds every frame of
  these in memory, so keep them small (the call refuses above ``PIL_MEMORY_LIMIT``). MP4
  works only when an ``ffmpeg`` binary is already on PATH; the project installs none.
- :func:`write_still` - one frame as a PNG. Stills carry the attribution footer, so they
  are what gets committed.

A frame shows the cars top-down (the camera follows one ghost, or shows the whole
circuit), a minimap, the lap clock, each ghost's speed with its badges, and the live time
delta: how far the followed ghost is behind the reference *at the distance it has
reached*, t - t_ref(s(t)), the quantity the Rerun log calls ``delta/<ghost>``. It ends at
the comparison's lap delta. Badges are drawn as chips. An optimum without the CERTIFIED
badge gets a NOT CERTIFIED chip, the same rule as the figures' "not certified" tag.

Refuses lockbox (R_TEST) circuits, by the document's track name, unless ``unsealed``.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MPath
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D

from apexzero.viz.export import ghost_arrays, lockbox_guard, validate_document
from apexzero.viz.figures import (
    AXIS,
    GAIN,
    INK,
    INK_2,
    KPH,
    LOSS,
    MUTED,
    SERIES,
    SURFACE,
    TUMFTM_ATTRIBUTION,
    _start_line,
    _style,
    _track_patch,
)

CAMERAS = ("follow", "full")
PIL_SUFFIXES = (".webp", ".gif", ".png", ".apng")
FRAME_GLOB = "frame_*.png"
PIL_MEMORY_LIMIT = 4e9      # bytes of decoded frames Pillow may hold for one animation
# A chevron pointing along +x: the car marker, rotated to the car's heading every frame.
CAR = MPath([(1.0, 0.0), (-0.7, 0.55), (-0.35, 0.0), (-0.7, -0.55), (1.0, 0.0)],
            [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY])

# Layout in figure fractions. The figure is 7.2 in tall at dpi = height / 7.2, so every
# font and line scales with the frame and a 640x360 preview looks like the 1280x720 master.
VIEW = (0.012, 0.075, 0.600, 0.800)      # main view: left, bottom, width, height
SIDE = (0.635, 0.988)                    # right column: x from, x to
MINI = (0.635, 0.560, 0.353, 0.315)      # minimap (follow camera) or speed panel (full)
STRIP = (0.668, 0.075, 0.320, 0.105)     # Δt(s) strip


@dataclass(frozen=True)
class RenderSpec:
    """What to draw, and how fast. Times are lap times in seconds (0 at the start line)."""

    fps: float = 30.0
    rate: float = 1.0               # lap seconds per second of clip (1.0 = real time)
    width: int = 1280
    height: int = 720
    camera: str = "follow"          # "follow" one ghost, or the "full" circuit
    follow: str | None = None       # the ghost the camera and Δt follow; default the first
                                    # ghost that is not the reference
    window_m: float = 300.0         # follow camera: metres shown from bottom to top
    start: float = 0.0
    end: float | None = None        # default: the slowest ghost's lap time
    hold: float = 2.0               # seconds of clip frozen on the last frame
    trail_s: float = 3.0            # seconds of path drawn behind each car
    title: str | None = None        # default: the document's title
    subtitle: str | None = None
    attribution: str | None = None  # footer, e.g. the track data's licence

    def __post_init__(self):
        if not (self.fps > 0 and self.rate > 0):
            raise ValueError("fps and rate must be positive")
        if self.width < 64 or self.height < 64:
            raise ValueError("frames must be at least 64 x 64 pixels")
        if self.camera not in CAMERAS:
            raise ValueError(f"camera {self.camera!r} not in {CAMERAS}")
        if not (self.window_m > 0 and self.hold >= 0 and self.trail_s >= 0):
            raise ValueError("window_m must be positive; hold and trail_s not negative")


# --------------------------------------------------------------------------- timing


def frame_times(start: float, end: float, fps: float, rate: float = 1.0,
                hold: float = 0.0) -> np.ndarray:
    """The lap time shown on each frame: ``rate`` lap seconds per second of clip from
    ``start``, the last moving frame exactly at ``end``, then ``hold`` seconds of clip
    frozen there."""
    if not (fps > 0 and rate > 0):
        raise ValueError("fps and rate must be positive")
    if not end > start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    step = rate / fps
    n = int(np.ceil((end - start) / step - 1e-9))
    t = np.minimum(start + step * np.arange(n + 1), end)
    return np.r_[t, np.full(round(hold * fps), float(end))]


def sample(ch: dict[str, np.ndarray], t: float) -> dict[str, float]:
    """A ghost's channels at lap time ``t``, linear in time and held at the first and last
    sample: a finished car waits on the line."""
    tt = float(np.clip(t, ch["t"][0], ch["t"][-1]))
    return {k: float(np.interp(tt, ch["t"], v)) for k, v in ch.items()}


def live_delta(ch: dict[str, np.ndarray], ref: dict[str, np.ndarray], t: float) -> float:
    """t - t_ref(s(t)): how far ghost ``ch`` is behind ``ref`` where it is at lap time
    ``t`` (> 0: behind). Once the ghost has finished it is the lap delta."""
    tt = float(np.clip(t, ch["t"][0], ch["t"][-1]))
    s = float(np.interp(tt, ch["t"], ch["s"]))
    return tt - float(np.interp(s, ref["s"], ref["t"]))


# --------------------------------------------------------------------------- labels


def badges(ghost: dict) -> list[str]:
    """The chips drawn under a ghost's label: its badge, plus NOT CERTIFIED for an optimum
    that is not the replay-verified 20 Hz answer key (the figures' "not certified")."""
    chips = [ghost["badge"]] if ghost.get("badge") else []
    if ghost["kind"] == "ocp" and ghost.get("badge") != "CERTIFIED":
        chips.append("NOT CERTIFIED")
    return chips


def default_attribution(doc: dict) -> str | None:
    """The TUMFTM licence line when the document's track came from that database (by its
    source field or its notes), else None."""
    text = " ".join([str(doc["track"].get("source", "")), *map(str, doc.get("notes", []))])
    return TUMFTM_ATTRIBUTION if "tumftm" in text.casefold() else None


def ghost_colours(doc: dict) -> dict[str, str]:
    """The figures' assignment: the reference in categorical slot 2, the others in 1, 3, 4."""
    ref = doc.get("reference")
    free = itertools.cycle([c for i, c in enumerate(SERIES) if ref is None or i != 1])
    return {g["id"]: SERIES[1] if g["id"] == ref else next(free) for g in doc["ghosts"]}


def _follow_id(doc: dict, follow: str | None) -> str:
    ids = [g["id"] for g in doc["ghosts"]]
    if follow is not None:
        if follow not in ids:
            raise KeyError(f"no ghost {follow!r} in the document (ghosts: {ids})")
        return follow
    return next((i for i in ids if i != doc.get("reference")), ids[0])


def _delta_comparison(doc: dict, follow: str) -> dict | None:
    """The comparison the Δt readout shows: the followed ghost's, or the first one when the
    camera follows the reference. None for a document without a reference."""
    comps = doc.get("comparisons", [])
    return next((c for c in comps if c["ghost"] == follow), comps[0] if comps else None)


def still_time(doc: dict, token: str | float, follow: str | None = None,
               end: float | None = None) -> float:
    """A lap time from a number, ``"end"`` (the slowest ghost's finish, or ``end``), or
    ``"worst"``: when the followed ghost reaches the apex of the corner that costs it the
    most (charged time when the export has the carry-over split, else measured)."""
    if isinstance(token, (int, float)):
        return float(token)
    if token == "end":
        return end if end is not None else max(g["lap_time_s"] for g in doc["ghosts"])
    if token == "worst":
        gid = _follow_id(doc, follow)
        comp = _delta_comparison(doc, gid)
        if comp is None:
            raise ValueError("'worst' needs a document with a reference ghost")
        split = comp.get("sector_split")
        cost = np.asarray(split["charged_s"] if split else comp["sector_delta_s"], float)
        sec = doc["track"]["sectors"][int(np.argmax(cost))]
        ch = ghost_arrays(doc, comp["ghost"])
        return float(np.interp(sec["s_apex"] % doc["track"]["length_m"], ch["s"], ch["t"]))
    try:
        return float(token)
    except ValueError:
        raise ValueError(f"still time {token!r}: a number of seconds, 'end' or 'worst'") from None


# --------------------------------------------------------------------------- drawing


def _view(lo: np.ndarray, hi: np.ndarray, aspect: float, pad: float = 0.04):
    """(centre, height) of the smallest view of ``aspect`` (width / height) that shows the
    box [lo, hi] with a margin of ``pad`` of each side."""
    w, h = (hi - lo) * (1.0 + 2.0 * pad)
    return 0.5 * (lo + hi), max(h, w / aspect)


def _set_view(ax, centre, height: float, aspect: float) -> None:
    ax.set_xlim(centre[0] - 0.5 * height * aspect, centre[0] + 0.5 * height * aspect)
    ax.set_ylim(centre[1] - 0.5 * height, centre[1] + 0.5 * height)


class Renderer:
    """Draws the frames of one document: build it once, then call :meth:`frame`."""

    def __init__(self, doc: dict, spec: RenderSpec | None = None, unsealed: bool = False):
        validate_document(doc)
        lockbox_guard(doc["track"]["name"], unsealed)
        self.doc, self.spec = doc, spec or RenderSpec()
        self.ghosts = {g["id"]: g for g in doc["ghosts"]}
        self.ch = {gid: ghost_arrays(doc, gid) for gid in self.ghosts}
        self.colour = ghost_colours(doc)
        self.follow = _follow_id(doc, self.spec.follow)
        self.comp = _delta_comparison(doc, self.follow)
        self.end = (self.spec.end if self.spec.end is not None
                    else max(g["lap_time_s"] for g in doc["ghosts"]))
        if not self.end > self.spec.start:
            raise ValueError(f"end ({self.end}) must be after start ({self.spec.start})")
        with rc_context(_style()):
            self._build()

    def times(self) -> np.ndarray:
        s = self.spec
        return frame_times(s.start, self.end, s.fps, s.rate, s.hold)

    # ---- layout

    def _build(self) -> None:
        sp, doc, tr = self.spec, self.doc, self.doc["track"]
        dpi = sp.height / 7.2
        # The epsilon keeps int(inches * dpi) from truncating to one pixel short.
        fig = Figure(figsize=((sp.width + 1e-3) / dpi, (sp.height + 1e-3) / dpi), dpi=dpi,
                     facecolor=SURFACE)
        self.fig, self.canvas = fig, FigureCanvasAgg(fig)
        self.aspect = VIEW[2] * sp.width / (VIEW[3] * sp.height)
        edges = np.r_[np.c_[tr["left"]["x"], tr["left"]["y"]],
                      np.c_[tr["right"]["x"], tr["right"]["y"]]]
        self.box = edges.min(axis=0), edges.max(axis=0)

        ax = self.ax = fig.add_axes(VIEW)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(AXIS)
        _track_patch(ax, tr)
        _start_line(ax, tr)
        self._sector_names(ax, tr)
        self.trail, self.car = {}, {}
        for gid, ch in self.ch.items():
            col = self.colour[gid]
            top = 1 if gid == self.follow else 0
            ax.plot(ch["x"], ch["y"], color=col, lw=0.9, alpha=0.35, zorder=3)
            self.trail[gid] = ax.plot([], [], color=col, lw=2.4, solid_capstyle="round",
                                      zorder=4 + top)[0]
            self.car[gid] = ax.plot([], [], ls="none", marker=CAR, ms=16, mfc=col,
                                    mec=SURFACE, mew=0.8, zorder=6 + top)[0]
        if sp.camera == "full":
            _set_view(ax, *_view(*self.box, self.aspect), self.aspect)
            self._build_speed_panel()
        else:
            self._build_minimap()

        title = sp.title or doc.get("title") or tr["name"]
        fig.text(VIEW[0], 0.955, title, fontsize=15, fontweight="bold", color=INK, va="center")
        if sp.subtitle:
            fig.text(VIEW[0], 0.912, sp.subtitle, fontsize=9.5, color=INK_2, va="center")
        if sp.attribution:
            fig.text(VIEW[0], 0.030, sp.attribution, fontsize=7.2, color=MUTED, va="center")
        fig.text(SIDE[1], 0.030, "Simulated laps, ApexZero", fontsize=7.2, color=MUTED,
                 ha="right", va="center")
        self._build_hud()

    def _sector_names(self, ax, tr: dict, offset: float = 40.0) -> None:
        s, kap = np.asarray(tr["s"]), np.asarray(tr["kappa"])
        x, y, th = (np.asarray(tr[k]) for k in ("x", "y", "heading"))
        for sec in tr["sectors"]:
            i = int(np.argmin(np.abs(s - sec["s_apex"])))
            side = -np.sign(kap[i]) or 1.0      # outside of the turn
            ax.text(x[i] - side * offset * np.sin(th[i]), y[i] + side * offset * np.cos(th[i]),
                    sec["name"], ha="center", va="center", fontsize=8, color=INK_2,
                    clip_on=True, zorder=2)

    def _build_minimap(self) -> None:
        tr, sp = self.doc["track"], self.spec
        am = self.fig.add_axes(MINI)
        am.set_axis_off()
        _track_patch(am, tr)
        _start_line(am, tr, overhang=10.0)
        aspect = MINI[2] * sp.width / (MINI[3] * sp.height)
        _set_view(am, *_view(*self.box, aspect, pad=0.02), aspect)
        self.dot = {}
        for gid, ch in self.ch.items():
            col = self.colour[gid]
            am.plot(ch["x"], ch["y"], color=col, lw=0.6, alpha=0.3)
            self.dot[gid] = am.plot([], [], "o", ms=5.5, mfc=col, mec=SURFACE, mew=0.6,
                                    zorder=5 + (gid == self.follow))[0]
        self.window_box = am.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=INK_2, lw=0.7,
                                                 zorder=6))

    def _build_speed_panel(self) -> None:
        """With the whole circuit in view, the minimap's slot shows speed against distance."""
        L = self.doc["track"]["length_m"]
        a = self.fig.add_axes((MINI[0] + 0.035, MINI[1], MINI[2] - 0.035, MINI[3] - 0.03))
        a.grid(axis="y")
        a.set_axisbelow(True)
        self.speed_line, self.speed_dot = {}, {}
        top = max(ch["speed"].max() for ch in self.ch.values()) * KPH
        for gid, ch in self.ch.items():
            col = self.colour[gid]
            a.plot(ch["s"], ch["speed"] * KPH, color=col, lw=0.7, alpha=0.25)
            self.speed_line[gid] = a.plot([], [], color=col, lw=1.2)[0]
            self.speed_dot[gid] = a.plot([], [], "o", ms=3.5, color=col)[0]
        a.set_xlim(0, L)
        a.set_ylim(0, top * 1.05)
        a.set_xticks([0, L / 2, L], ["0", f"{L / 2000:.1f}", f"{L / 1000:.1f} km"])
        a.tick_params(labelsize=7)
        a.set_ylabel("speed (km/h)", fontsize=7)

    def _build_hud(self) -> None:
        fig, sp = self.fig, self.spec
        x0, x1 = SIDE
        self.canvas.draw()      # text extents, to place the badge chips
        renderer = self.canvas.get_renderer()
        self.clock = fig.text(x0, 0.510, "", fontsize=20, fontweight="bold", color=INK,
                              va="center")
        pace = "real time" if sp.rate == 1 else f"played at {sp.rate:g}x real time"
        fig.text(x1, 0.510, f"lap time, {pace}", fontsize=8, color=MUTED, ha="right",
                 va="center")
        n = len(self.ghosts)
        step = 0.080 if n <= 2 else 0.160 / (n - 1)
        self.value = {}
        for k, (gid, g) in enumerate(self.ghosts.items()):
            y = 0.445 - step * k
            fig.text(x0, y, "●", fontsize=11, color=self.colour[gid], va="center")
            fig.text(x0 + 0.022, y, g["label"], fontsize=9.5, color=INK, va="center")
            self.value[gid] = fig.text(x1, y, "", fontsize=9.5, color=INK, ha="right",
                                       va="center")
            x = x0 + 0.024
            for chip in badges(g):
                t = fig.text(x, y - 0.034, chip, fontsize=6.8, fontweight="bold", color=INK_2,
                             va="center", bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE,
                                                    ec=INK_2, lw=0.6))
                x += t.get_window_extent(renderer).width / sp.width + 0.016
        self.delta = self.dline = None
        if self.comp is None:
            return
        gid, rid = self.comp["ghost"], self.comp["reference"]
        y = 0.445 - step * (n - 1) - 0.095
        self.delta = fig.text(x0, y, "", fontsize=15, fontweight="bold", va="center")
        fig.text(x0, y - 0.040, f"{self.ghosts[gid]['label']} against "
                 f"{self.ghosts[rid]['label']} at the same distance; > 0: behind",
                 fontsize=7.2, color=MUTED, va="center")
        ad = fig.add_axes(STRIP)
        cs, cd = np.asarray(self.comp["s"], float), np.asarray(self.comp["delta_t"], float)
        self.cs, self.cd = cs, cd
        L = self.doc["track"]["length_m"]
        ad.axhline(0.0, color=AXIS, lw=0.6)
        col = self.colour[gid]
        self.dline = ad.plot([], [], color=col, lw=1.4)[0]
        self.ddot = ad.plot([], [], "o", ms=3.5, color=col)[0]
        lo, hi = min(cd.min(), 0.0), max(cd.max(), 0.0)
        pad = 0.1 * max(hi - lo, 1e-3)
        ad.set_xlim(0, L)
        ad.set_ylim(lo - pad, hi + pad)
        ad.set_xticks([0, L / 2, L], ["0", f"{L / 2000:.1f}", f"{L / 1000:.1f} km"])
        ad.yaxis.set_major_locator(MaxNLocator(3))
        ad.tick_params(labelsize=7)
        ad.set_ylabel("Δt (s)", fontsize=7)

    # ---- per frame

    def _update(self, t: float) -> None:
        now = {gid: sample(ch, t) for gid, ch in self.ch.items()}
        for gid, ch in self.ch.items():
            p, g = now[gid], self.ghosts[gid]
            self.car[gid].set_data([p["x"]], [p["y"]])
            self.car[gid].set_marker(MarkerStyle(CAR, transform=Affine2D().rotate(p["heading"])))
            k = (ch["t"] > t - self.spec.trail_s) & (ch["t"] < t)
            self.trail[gid].set_data(np.r_[ch["x"][k], p["x"]], np.r_[ch["y"][k], p["y"]])
            done = t >= g["lap_time_s"]
            self.value[gid].set_text(f"lap {g['lap_time_s']:.3f} s" if done
                                     else f"{p['speed'] * KPH:.0f} km/h")
            if self.spec.camera == "follow":
                self.dot[gid].set_data([p["x"]], [p["y"]])
            else:
                k = ch["t"] <= t
                self.speed_line[gid].set_data(np.r_[ch["s"][k], p["s"]],
                                              np.r_[ch["speed"][k], p["speed"]] * KPH)
                self.speed_dot[gid].set_data([p["s"]], [p["speed"] * KPH])
        self.clock.set_text(f"{min(t, self.end):.2f} s")
        if self.spec.camera == "follow":
            f = now[self.follow]
            h, w = self.spec.window_m, self.spec.window_m * self.aspect
            _set_view(self.ax, (f["x"], f["y"]), h, self.aspect)
            self.window_box.set_bounds(f["x"] - w / 2, f["y"] - h / 2, w, h)
        if self.comp is not None:
            gid = self.comp["ghost"]
            d = live_delta(self.ch[gid], self.ch[self.comp["reference"]], t)
            self.delta.set_text(f"Δt {d:+.3f} s")
            self.delta.set_color(LOSS if d > 5e-4 else GAIN if d < -5e-4 else INK)
            s_now = now[gid]["s"]
            k = self.cs <= s_now
            d_now = float(np.interp(s_now, self.cs, self.cd))
            self.dline.set_data(np.r_[self.cs[k], s_now], np.r_[self.cd[k], d_now])
            self.ddot.set_data([s_now], [d_now])

    def frame(self, t: float) -> np.ndarray:
        """The frame at lap time ``t``, as RGB [height, width, 3] uint8."""
        self._update(float(t))
        with rc_context(_style()):
            self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()


def _frames(r: Renderer, times: Iterable[float]) -> Iterator[np.ndarray]:
    """Frames at ``times``; a repeated time (the hold) reuses the previous frame."""
    last_t, last = None, None
    for t in times:
        if t != last_t:
            last_t, last = t, r.frame(t)
        yield last


def render_frames(doc: dict, spec: RenderSpec | None = None,
                  unsealed: bool = False) -> Iterator[tuple[float, np.ndarray]]:
    """(lap time, RGB frame) for every frame of the clip."""
    r = Renderer(doc, spec, unsealed)
    times = r.times()
    yield from zip(times.tolist(), _frames(r, times), strict=True)


# --------------------------------------------------------------------------- writers


def _manifest(r: Renderer, times: np.ndarray, **extra) -> dict:
    doc, sp = r.doc, r.spec
    return {
        "generator": "apexzero.viz.render",
        "source": {k: doc.get(k) for k in ("schema", "version", "title", "created")}
                  | {"track": doc["track"]["name"]},
        "spec": dataclasses.asdict(sp) | {"end": r.end, "follow": r.follow},
        "frames": len(times), "duration_s": len(times) / sp.fps,
        "lap_times_s": np.round(times, 4).tolist(),
        "ghosts": {gid: {"label": g["label"], "badges": badges(g), "lap_time_s": g["lap_time_s"]}
                   for gid, g in r.ghosts.items()},
        **extra,
    }


def write_frames(doc: dict, out_dir: str | Path, spec: RenderSpec | None = None,
                 overwrite: bool = False, unsealed: bool = False) -> Path:
    """Write the clip as ``<out_dir>/frame_00000.png``, ... and ``frames.json`` (frame rate,
    the lap time of every frame, the spec and the source document). Returns the manifest.

    Refuses a directory that already holds frames unless ``overwrite``, which removes only
    the old ``frame_*.png`` files."""
    from PIL import Image

    r = Renderer(doc, spec, unsealed)
    out = Path(out_dir)
    old = sorted(out.glob(FRAME_GLOB)) if out.exists() else []
    if old and not overwrite:
        raise FileExistsError(f"{out} already holds {len(old)} frames (pass overwrite=True)")
    out.mkdir(parents=True, exist_ok=True)
    for p in old:
        p.unlink()
    times = r.times()
    for k, rgb in enumerate(_frames(r, times)):
        Image.fromarray(rgb).save(out / f"frame_{k:05d}.png")
    manifest = out / "frames.json"
    manifest.write_text(json.dumps(_manifest(
        r, times, pattern="frame_%05d.png",
        encode=(f"ffmpeg -framerate {r.spec.fps:g} -i frame_%05d.png -c:v libx264 "
                "-pix_fmt yuv420p clip.mp4")), indent=1))
    return manifest


def _durations_ms(n: int, fps: float) -> list[int]:
    """Integer frame durations whose running sum stays on the exact clip clock."""
    edges = np.round(np.arange(n + 1) * 1000.0 / fps).astype(int)
    return np.diff(edges).tolist()


def write_animation(doc: dict, path: str | Path, spec: RenderSpec | None = None,
                    unsealed: bool = False) -> Path:
    """Write the clip as an animated ``.webp``, ``.gif`` or ``.png``/``.apng`` (Pillow), or
    as ``.mp4`` when an ``ffmpeg`` binary is on PATH."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        return _write_mp4(doc, path, spec, unsealed)
    if suffix not in PIL_SUFFIXES:
        raise ValueError(f"unsupported clip format {suffix!r}: use {PIL_SUFFIXES} or .mp4")
    from PIL import Image

    r = Renderer(doc, spec, unsealed)
    sp, times = r.spec, r.times()
    if suffix == ".gif" and abs(100.0 / sp.fps - round(100.0 / sp.fps)) > 1e-9:
        raise ValueError(f"GIF stores frame durations in 1/100 s, so {sp.fps:g} fps would "
                         "drift: use 10, 20, 25 or 50 fps, or WebP")
    held = len(times) * sp.width * sp.height * 4
    if held > PIL_MEMORY_LIMIT:
        raise ValueError(f"Pillow would hold about {held / 1e9:.1f} GB of frames for this "
                         "animation: lower the size or fps, or write an image sequence")
    frames = (Image.fromarray(rgb) for rgb in _frames(r, times))
    first = next(frames)
    kw: dict = {"save_all": True, "append_images": list(frames), "loop": 0,
                "duration": _durations_ms(len(times), sp.fps)}
    if suffix == ".webp":
        kw |= {"quality": 85, "method": 4}
    elif suffix in (".png", ".apng"):
        kw["format"] = "PNG"
    path.parent.mkdir(parents=True, exist_ok=True)
    first.save(path, **kw)
    return path


def _write_mp4(doc: dict, path: Path, spec: RenderSpec | None, unsealed: bool) -> Path:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("MP4 needs an ffmpeg binary on PATH, and this project installs "
                           "none: write an image sequence (or a .webp) and encode it later")
    r = Renderer(doc, spec, unsealed)
    sp = r.spec
    if sp.width % 2 or sp.height % 2:
        raise ValueError("H.264 in yuv420p needs an even width and height")
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{sp.width}x{sp.height}", "-framerate", f"{sp.fps:g}", "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for rgb in _frames(r, r.times()):
            proc.stdin.write(rgb.tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code:
        raise RuntimeError(f"ffmpeg exited with status {code}")
    return path


def write_still(doc: dict, t: float, path: str | Path, spec: RenderSpec | None = None,
                unsealed: bool = False) -> Path:
    """One frame, at lap time ``t``, as a PNG."""
    from PIL import Image

    rgb = Renderer(doc, spec, unsealed).frame(t)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path
