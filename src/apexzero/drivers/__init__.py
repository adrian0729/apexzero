"""Drivers-lite: where F1 teammates gain or lose time, from public 3.7 Hz data.

The public position feed has no lateral line (docs/findings/snapping-audit.md), so the
study lives in the distance domain: time as a function of lap distance along the TUMFTM
centreline. Modules, in pipeline order:

- ``circuits``: OpenF1 <-> TUMFTM names, and which TUMFTM layouts are out of date;
- ``registration``: similarity transform of the feed onto the centreline (ICP), with the
  3 m RMS / 1% length gate;
- ``session``: loading an archived session; race-control parsing;
- ``progress`` + ``smoother``: s(t) per lap, an RTS smoother on [s, v, a] anchored to the
  official lap and sector times;
- ``laps``: the push-lap filter, including traffic;
- ``segments``: additive corner complexes and entry / apex / exit phases;
- ``deltas``: R1 envelope, teammate deltas, placebo and split-half checks;
- ``synth``: synthetic laps for the injection-recovery gate;
- ``pipeline``: the steps above, end to end, for real and synthetic sessions.

Everything here is a pilot (plan section 3.4). Nothing in it ranks drivers.
"""
