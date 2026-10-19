# Demo exports

Viewer v0 and the ghost exports from plan section 3.2, built on what exists now (the K4
Hockenheim specialist and the s-domain OCP). Re-run them on final agents with the same script.

| File | What | In git |
|---|---|---|
| `hockenheim-specialist-map.png` | Both racing lines coloured by speed, plus a zoom on the corner that costs the agent the most | yes |
| `hockenheim-specialist-speed-delta.png` | Speed against distance for both ghosts, and Δt(s) below | yes |
| `hockenheim-specialist-sectors.png` | Time per corner: the bar is what the corner is charged, the outline what the clock measured inside the sector. Both sum to the lap delta | yes |
| `hockenheim-specialist.json` | `apexzero.trajectory` 1.1 export: track centreline and edges, corner sectors, two ghosts (agent, optimum), Δt(s), and per sector the measured and charged time and the entry, exit and minimum speeds | no, regenerate |
| `hockenheim-specialist.html` | The static viewer with that export embedded. Open it straight from disk | no, regenerate |

The JSON and the HTML are not committed because they contain the TUMFTM track geometry
(LGPL-3.0), and the repository's policy is to fetch that data by script, never to commit
it. Regenerate them (about 10 s of CPU; needs the TUMFTM tracks under `data/`, the agent
under `runs/` and the OCP under `outputs/`):

```bash
uv run python scripts/export_demo.py
uv run --group rerun python scripts/export_demo.py --rrd /tmp/hockenheim.rrd   # plus a Rerun recording
```

Other exports load into `src/apexzero/viz/viewer/index.html`. Drop the file on the page, or
serve the repository and open `.../viewer/index.html?data=/docs/demo/<name>.json`. The URL
also takes `t=<s>`, `follow=<ghost id>`, `rate=<x>`, `play=1` and `speed=1`.

## What the numbers are, and are not

- **Hockenheim is an R-dev circuit.** No agent or optimum lap is computed on a lockbox
  (R_TEST) circuit. The export library refuses those by the track's name (case, path and
  separators ignored, mirrored and reversed variants included) unless it is explicitly
  unsealed, and `export_demo.py` only unseals with `--unseal-lockbox`.
- **The optimum is not certified.** It is the s-domain minimum-time OCP with continuous
  controls (T_cont), on the same ODE, car and corridor. It is **not** the CERTIFIED 20 Hz
  answer key T*_20, because TO-OCP lap mode does not exist yet. Since T_cont ≤ T*_20, the
  4.8% gap (74.679 s against 71.282 s) overstates the time a 20 Hz policy could still find.
  No badge is attached to either ghost; the figures and the viewer tag the optimum
  "not certified" wherever it is named.
- **The agent lap** uses the evaluation protocol of `apexzero.eval.laps`: a deterministic
  policy, a rolling start at s = 0 at 90% of the QSS reference speed, and lap 2 timed. The
  export re-runs that lap and keeps the full vehicle state. Its lap time differs from run 0
  of the training script's `eval.json` by 1.1e-6 s and from a fresh one-run
  `timed_laps` call by 1.6e-4 s. Both are float32 rounding in differently compiled
  rollouts; the cause is not isolated (it is not the batch size: the one-run harness call
  and the export both run one car). The script fails if the harness differs by more than
  1 ms. The exact values are in the export's notes and provenance.
- **Measured against charged.** Sectors (C1, C2, ...) are corner complexes from
  `tracks.corners.segment`, each with half of the straight on either side; they are not
  official turn numbers. What a sector measures includes a speed deficit carried in from
  the corner before it: both cars are flat out through all of C2, a 253 m-radius bend,
  and the agent enters it 33 km/h slower after braking to 221 km/h at C1 (the optimum
  keeps 277 km/h). C2 measures +0.265 s, all of it C1's. The charged column moves time
  gained or lost while both cars are flat out (throttle ≥ 0.99) back to the corner where
  that stretch began: C1 is charged +0.665 s and C2 0.000 s. Charged times are still an
  attribution rule, not a counterfactual; they also move any line difference on those
  flat-out stretches upstream.
- **Speeds, not radii.** The figure and the viewer label each sector with the optimum's
  minimum speed. The reference line's minimum radius (`r_min` in the export) can mislead:
  at C1 the refined line has a short curvature spike (about 20 m) that both cars straighten.
- `cmd_steer` and `cmd_pedal` are the commands the simulator applied (the policy output
  clipped to [-1, 1]), not the raw policy mean.
- Track geometry comes from the TUMFTM racetrack-database (LGPL-3.0).
