"""Write a trajectory export as a Rerun recording (``.rrd``): the plan's viewer v0.

Rerun's own viewer (native, or the web viewer with the file dropped on it) then gives
scrubbing, synced plots and zoom for free, from the same export the static viewer reads.
Optional: ``rerun-sdk`` lives in the ``rerun`` dependency group (``uv sync --group rerun``),
because the wheel is large and nothing else depends on it.

Layout of the recording:
    track/edges/{left,right}, track/sectors   static geometry and corner-complex labels
    track/lines/<ghost>                        each ghost's whole lap, static
    track/cars/<ghost>                         the car position on the ``lap_time`` timeline
    speed/<ghost>, delta/<ghost>, grip/<ghost> scalars on the same timeline
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apexzero.viz.export import ghost_arrays, validate_document

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
EDGE = "#a9a89f"


def _rgb(hex_colour: str) -> list[int]:
    h = hex_colour.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def log_document(doc: dict, path: str | Path, application_id: str = "apexzero") -> Path:
    """Write ``doc`` to ``path`` (.rrd). Returns the path."""
    import rerun as rr
    import rerun.blueprint as rrb

    validate_document(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blueprint = rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial2DView(origin="/track", name=doc["track"]["name"]),
        rrb.Vertical(
            rrb.TimeSeriesView(origin="/speed", name="speed (km/h)"),
            rrb.TimeSeriesView(origin="/delta", name="Δt vs reference (s)"),
            rrb.TimeSeriesView(origin="/grip", name="friction-ellipse use"),
        ),
        column_shares=[3, 2],
    ))
    rec = rr.RecordingStream(application_id)
    rec.save(path, default_blueprint=blueprint)

    tr = doc["track"]
    for side in ("left", "right"):
        xy = np.c_[tr[side]["x"], tr[side]["y"]]
        if tr["closed"]:
            xy = np.vstack([xy, xy[:1]])
        rec.log(f"track/edges/{side}", rr.LineStrips2D([xy], colors=[_rgb(EDGE)]), static=True)
    s = np.asarray(tr["s"])
    apex = [int(np.argmin(np.abs(s - sec["s_apex"]))) for sec in tr["sectors"]]
    if apex:
        rec.log("track/sectors", rr.Points2D(
            np.c_[np.asarray(tr["x"])[apex], np.asarray(tr["y"])[apex]],
            labels=[sec["name"] for sec in tr["sectors"]], radii=-2.0, show_labels=True),
            static=True)

    ref_id = doc.get("reference")
    ref = ghost_arrays(doc, ref_id) if ref_id else None
    for k, g in enumerate(doc["ghosts"]):
        gid, col = g["id"], _rgb(SERIES[k % len(SERIES)])
        ch = ghost_arrays(doc, gid)
        t = ch["t"]
        xy = np.c_[ch["x"], ch["y"]]
        times = [rr.TimeColumn("lap_time", duration=t)]
        rec.log(f"track/lines/{gid}", rr.LineStrips2D([xy], colors=[col], radii=-0.75),
                static=True)
        # Negative radii are UI points: the car stays visible at any zoom.
        rec.send_columns(f"track/cars/{gid}", indexes=times, columns=rr.Points2D.columns(
            positions=xy, radii=np.full(len(t), -5.0), colors=np.tile(col, (len(t), 1))))
        for root, values in (("speed", ch["speed"] * 3.6),
                             ("grip", ch.get("ellipse_use"))):
            if values is None:
                continue
            rec.log(f"{root}/{gid}", rr.SeriesLines(colors=[col], names=[g["label"]]),
                    static=True)
            rec.send_columns(f"{root}/{gid}", indexes=times,
                             columns=rr.Scalars.columns(scalars=values))
        if ref is not None and gid != ref_id:
            # Live delta: how far behind the reference this ghost is when it reaches s.
            d = t - np.interp(ch["s"], ref["s"], ref["t"])
            rec.log(f"delta/{gid}", rr.SeriesLines(colors=[col], names=[g["label"]]),
                    static=True)
            rec.send_columns(f"delta/{gid}", indexes=times, columns=rr.Scalars.columns(scalars=d))
    rec.flush()
    return path
