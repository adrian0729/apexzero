# Demo exports

Viewer v0 and the ghost exports from plan section 3.2, built on what exists now (the K4
Hockenheim specialist and the s-domain OCP). Re-run them on final agents with the same script.

| File | What |
|---|---|
| `hockenheim-specialist.json` | `apexzero.trajectory` 1.0 export: track centreline and edges, corner sectors, two ghosts (agent, optimum), Δt(s) and the time per sector |
| `hockenheim-specialist.html` | The static viewer with that export embedded. Open it straight from disk |
| `hockenheim-specialist-map.png` | Both racing lines coloured by speed, plus a zoom on the costliest corner complex |
| `hockenheim-specialist-speed-delta.png` | Speed against distance for both ghosts, and Δt(s) below |
| `hockenheim-specialist-sectors.png` | Time lost per corner sector. The bars sum to the lap delta |

Regenerate (about 15 s of CPU):

```bash
uv run python scripts/export_demo.py
uv run --group rerun python scripts/export_demo.py --rrd /tmp/hockenheim.rrd   # plus a Rerun recording
```

Other exports load into `src/apexzero/viz/viewer/index.html`. Drop the file on the page, or
serve the repository and open `.../viewer/index.html?data=/docs/demo/<name>.json`. The URL
also takes `t=<s>`, `follow=<ghost id>`, `rate=<x>`, `play=1` and `speed=1`.

## What the numbers are, and are not

- **Hockenheim is an R-dev circuit.** No agent or optimum lap is computed on a lockbox
  (R_TEST) circuit. `export_demo.py` refuses those unless `--unseal-lockbox` is passed.
- **The optimum is the s-domain minimum-time OCP**, with continuous controls (T_cont), on the
  same ODE, car and corridor. It is **not** the CERTIFIED 20 Hz answer key T*_20, because
  TO-OCP lap mode does not exist yet. Since T_cont ≤ T*_20, the 4.8% gap overstates the time
  a 20 Hz policy could still find. No badge is attached to either ghost for that reason.
- **The agent lap** uses the evaluation protocol of `apexzero.eval.laps`: a deterministic
  policy, a rolling start at s = 0 at 90% of the QSS reference speed, and lap 2 timed. The
  export re-runs that lap and keeps the full vehicle state. Its lap time (74.679 s) matches
  run 0 of the training script's `eval.json` to 1e-6 s, and a fresh one-run `timed_laps` call
  to 2e-4 s. The float32 network rounds slightly differently at different batch sizes;
  the script fails if the two differ by more than 1 ms.
- **Sectors (C1, C2, ...)** are corner complexes from `tracks.corners.segment`, each with
  half of the straight on either side. They are not official turn numbers.
- Track geometry comes from the TUMFTM racetrack-database (LGPL-3.0).
