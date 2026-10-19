"""Synthetic tracks, OCP results and documents shared by the viz tests."""

import numpy as np

from apexzero.ocp.min_time import OCPResult
from apexzero.tracks.track import from_centreline
from apexzero.viz import export


def rounded_square(side=200.0, radius=40.0, width=12.0, start_frac=0.0):
    """Four 90-degree corners joined by straights. ``start_frac`` rotates where s = 0 is."""
    pts = []
    centres = [(side, 0.0), (side, side), (0.0, side), (0.0, 0.0)]
    for k, (cx, cy) in enumerate(centres):
        a0 = -np.pi / 2 + k * np.pi / 2
        x0, y0 = centres[k - 1]
        x0, y0 = x0 + radius * np.cos(a0), y0 + radius * np.sin(a0)
        x1, y1 = cx + radius * np.cos(a0), cy + radius * np.sin(a0)
        for f in np.linspace(0, 1, 40, endpoint=False):
            pts.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
        for a in np.linspace(a0, a0 + np.pi / 2, 30, endpoint=False):
            pts.append((cx + radius * np.cos(a), cy + radius * np.sin(a)))
    xy = np.roll(np.array(pts), -int(start_frac * len(pts)), axis=0)
    w = np.full(len(xy), width / 2)
    return from_centreline("square", xy, w, w, ds=2.0, smoothing=0.05)


def circle(radius=150.0, width=14.0, n=400):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)
    w = np.full(n, width / 2)
    return from_centreline("circle", xy, w, w, ds=2.0)


def synthetic_ocp(track, ds=3.0, v0=60.0, dv=15.0):
    """An OCPResult with a known speed profile, its time from the trapezoidal dt/ds rule
    exactly as ``solve_min_time`` computes it."""
    L = track.length
    n = round(L / ds)
    h = L / n
    s = np.arange(n) * h
    v = v0 + dv * np.sin(2 * np.pi * s / L)
    z = np.zeros((n, 7))
    z[:, 0] = 2.0 * np.sin(4 * np.pi * s / L)
    z[:, 2] = v
    z[:, 6] = 0.3
    u = np.zeros((n, 2))
    nxt = np.r_[1:n, 0]
    seg = 0.5 * h * (1 / v + 1 / v[nxt])
    t = np.r_[0.0, np.cumsum(seg)][:n]
    return OCPResult(s=s, z=z, u=u, t=t, lap_time=float(seg.sum()), success=True,
                     status="synthetic", iterations=0, solve_seconds=0.0,
                     kappa=np.interp(s, track.s, track.kappa), w_left=np.full(n, 7.0),
                     w_right=np.full(n, 7.0), dual_left=np.zeros(n), dual_right=np.zeros(n),
                     meta={"ds": h, "margin": 1.0, "track": track.name})


def make_doc(track=None):
    track = track or rounded_square()
    res = synthetic_ocp(track)
    ocp = export.ocp_ghost(res, track)
    slow = synthetic_ocp(track, v0=55.0)
    agent = export.ocp_ghost(slow, track, id="agent", label="slower")
    agent.kind = "policy"
    return export.build_document(track, [agent, ocp], reference="ocp", title="test",
                                 notes=["a note with </script> inside"])
