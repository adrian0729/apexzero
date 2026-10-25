"""Render a trajectory export (scripts/export_demo.py) into a clip, an image sequence or stills.

    uv run python scripts/render_demo.py docs/demo/hockenheim-specialist.json \
        --name hockenheim-mvp1 --rate 1.3 --frames --clip webp --stills worst,end

Writes, under --out (default: the export's directory):
    <name>-frames/frame_00000.png ...  with --frames: the lossless master sequence at --size
                                       and --fps, plus frames.json (lap time of every frame)
    <name>.<fmt>                       with --clip webp,gif,apng,mp4: an animation at
                                       --clip-size and --clip-fps (Pillow; mp4 only if an
                                       ffmpeg binary is on PATH, which the project does not
                                       install)
    <name>-<token>.png                 with --stills: one frame each at --size, for a lap
                                       time in seconds, 'end' or 'worst' (the followed
                                       ghost at the apex of its costliest corner)

The frames show the export's own track geometry (TUMFTM-derived, LGPL-3.0), so every
frame carries the attribution footer. Only stills are committed; the frames and clips
are git-ignored under docs/demo and regenerated with this script.

Lockbox: refuses R_TEST circuits (and their mirrored/reversed variants) unless
--unseal-lockbox is given, which protocol-v1 allows only after the unseal.
"""

import argparse
import dataclasses
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)

from apexzero.viz import export, render


def size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc", help="an apexzero.trajectory export (.json)")
    ap.add_argument("--out", default=None, help="output directory (default: the export's)")
    ap.add_argument("--name", default=None, help="output stem (default: the export's)")
    ap.add_argument("--frames", action="store_true", help="write the PNG sequence")
    ap.add_argument("--clip", default="", help="comma-separated: webp, gif, apng, mp4")
    ap.add_argument("--stills", default="", help="comma-separated lap times, 'end', 'worst'")
    ap.add_argument("--size", type=size, default=(1280, 720), help="frames and stills, WxH")
    ap.add_argument("--fps", type=float, default=30.0, help="frame sequence rate")
    ap.add_argument("--clip-size", type=size, default=(854, 480))
    ap.add_argument("--clip-fps", type=float, default=20.0, help="GIF needs 10/20/25/50")
    ap.add_argument("--rate", type=float, default=1.0, help="lap seconds per clip second")
    ap.add_argument("--camera", choices=render.CAMERAS, default="follow")
    ap.add_argument("--follow", default=None, help="ghost id (default: the first that is "
                                                   "not the reference)")
    ap.add_argument("--window", type=float, default=300.0, help="follow view height (m)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--hold", type=float, default=2.5, help="seconds frozen at the end")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--attribution", default=None,
                    help="footer (default: the TUMFTM licence line when the export's track "
                         "comes from TUMFTM)")
    ap.add_argument("--overwrite", action="store_true", help="replace an old frame sequence")
    ap.add_argument("--unseal-lockbox", action="store_true")
    args = ap.parse_args()

    wall0 = time.perf_counter()
    src = Path(args.doc)
    doc = export.read_document(src)
    export.lockbox_guard(doc["track"]["name"], unsealed=args.unseal_lockbox)
    out = Path(args.out) if args.out else src.parent
    name = args.name or src.stem
    attribution = args.attribution or render.default_attribution(doc)
    if attribution is None:
        print("note: no attribution footer (the export names no TUMFTM source); "
              "pass --attribution")
    w, h = args.size
    spec = render.RenderSpec(
        fps=args.fps, rate=args.rate, width=w, height=h, camera=args.camera,
        follow=args.follow, window_m=args.window, start=args.start, end=args.end,
        hold=args.hold, title=args.title, subtitle=args.subtitle, attribution=attribution)

    paths = []
    for token in filter(None, args.stills.split(",")):
        t = render.still_time(doc, token.strip(), args.follow, args.end)
        tag = token.strip().replace(".", "p")
        paths.append(render.write_still(doc, t, out / f"{name}-{tag}.png", spec,
                                        unsealed=args.unseal_lockbox))
        print(f"still '{token}' at lap time {t:.2f} s")
    if args.frames:
        paths.append(render.write_frames(doc, out / f"{name}-frames", spec,
                                         overwrite=args.overwrite,
                                         unsealed=args.unseal_lockbox))
    cw, ch = args.clip_size
    clip_spec = dataclasses.replace(spec, fps=args.clip_fps, width=cw, height=ch)
    for fmt in filter(None, args.clip.split(",")):
        paths.append(render.write_animation(doc, out / f"{name}.{fmt.strip()}", clip_spec,
                                            unsealed=args.unseal_lockbox))

    r = render.Renderer(doc, spec, unsealed=args.unseal_lockbox)
    n = len(r.times())
    print(f"{doc['track']['name']}: {n} frames at {args.fps:g} fps = {n / args.fps:.1f} s of "
          f"clip, lap time {spec.start:.2f}-{r.end:.2f} s at {args.rate:g}x, "
          f"following {r.follow}")
    for p in paths:
        # A sequence is reported by its manifest; its size is the whole directory's.
        files = list(p.parent.iterdir()) if p.name == "frames.json" else [p]
        print(f"  {p}  ({sum(f.stat().st_size for f in files) / 1e3:.0f} kB)")
    print(f"wall {time.perf_counter() - wall0:.1f} s")


if __name__ == "__main__":
    main()
