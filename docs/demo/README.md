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
| `hockenheim-mvp1-worst.png`, `hockenheim-mvp1-end.png` | MVP-1 lite stills: the specialist at the apex of its costliest corner (C12-C13), and the finish | yes |
| `hockenheim-mvp1.webp`, `hockenheim-mvp1-frames/` | The MVP-1 lite clip: an animated WebP preview, and the lossless PNG master sequence with `frames.json` | no, regenerate |
| `catalunya-generalist-{map,speed-delta,sectors}.png` | The same three figures for the R-dev generalist ghost on Catalunya, badged POST HOC | yes |
| `catalunya-generalist-worst.png`, `catalunya-generalist-end.png` | Stills of the generalist ghost against the optimum: at the apex station of its costliest corner complex (C13-C15), and the finish | yes |
| `catalunya-generalist.json`, `.html`, `.webp`, `-frames/` | Its export, viewer page and clip | no, regenerate |

Every committed figure and still carries the TUMFTM attribution in its footer.

The JSON, the HTML and the clips are not committed because they contain the TUMFTM track
geometry (LGPL-3.0), and the repository's policy is to fetch that data by script, never to
commit it; the clips are large as well. Regenerate the exports (about 10 s of CPU each;
needs the TUMFTM tracks under `data/`, the agent under `runs/` and the OCP under
`outputs/`):

```bash
uv run python scripts/export_demo.py
uv run --group rerun python scripts/export_demo.py --rrd /tmp/hockenheim.rrd   # plus a Rerun recording
```

Other exports load into `src/apexzero/viz/viewer/index.html`. Drop the file on the page, or
serve the repository and open `.../viewer/index.html?data=/docs/demo/<name>.json`. The URL
also takes `t=<s>`, `follow=<ghost id>`, `rate=<x>`, `play=1` and `speed=1`.

## Render mode (clips, frame sequences, stills)

`scripts/render_demo.py` draws an export frame by frame (`apexzero.viz.render`). It reads
only the JSON, like the figures, and needs nothing beyond the default `viz` group: frames
are drawn on matplotlib's Agg canvas and animations are written with Pillow, which
matplotlib already depends on. No browser and no ffmpeg are involved.

- **The frame:** a camera that follows one ghost (or `--camera full` for the whole
  circuit), a minimap, the lap clock, each ghost's speed with its badges as chips, and the
  live Δt: how far the followed ghost is behind the reference at the distance it has
  reached, t − t_ref(s(t)). The strip below draws Δt(s) up to that point. Δt ends at the
  export's lap delta. An optimum without the CERTIFIED badge carries a NOT CERTIFIED chip.
- **Outputs:** `--frames` writes the lossless PNG master (`<name>-frames/frame_00000.png`
  …, plus `frames.json` with the lap time of every frame). `--clip webp` (or `gif`,
  `apng`) writes an animated preview, and `--stills worst,end,<seconds>` writes single
  frames. `worst` puts the followed ghost at the apex station (minimum reference radius)
  of its costliest corner complex, by charged time.
- **Pillow holds every frame of an animation in memory**, so previews default to 854×480
  at 20 fps, about 2 GB for a 60 s clip. The call refuses above 4 GB. GIF stores frame
  durations in 1/100 s, so it accepts only 10, 20, 25 or 50 fps.
- **MP4.** ffmpeg is not installed, and nothing here installs it. `--clip mp4` works only
  if an `ffmpeg` binary is already on PATH. Otherwise encode the PNG master later with the
  command in `frames.json`:
  `ffmpeg -framerate 30 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p clip.mp4`.
- **Cost on the Mac.** For the 60 s MVP-1 lite clip, the 1280×720, 30 fps master is 1,800
  frames, 182 MB and about 3 min. The 854×480 WebP preview is 10.9 MB and another 3 min.
- Lockbox circuits are refused by the export's track name, as everywhere else.

### Regenerating each output

Run from the repository root, in this order: each render reads the export written just
before it.

```bash
# Hockenheim specialist against the optimum: export, viewer page and the three figures
uv run python scripts/export_demo.py

# MVP-1 lite: the full lap at 1.3x real time (57.4 s, plus a 2.5 s hold), stills, master, preview
uv run python scripts/render_demo.py docs/demo/hockenheim-specialist.json --name hockenheim-mvp1 \
    --rate 1.3 --hold 2.5 --frames --clip webp --stills worst,end \
    --title "Hockenheim: the RL specialist against the optimum's ghost" \
    --subtitle "MVP-1 lite. Lap 2 from a rolling start, deterministic policy. The optimum is the continuous-control s-domain OCP (T_cont): not certified."

# The R-dev generalist ghost on Catalunya, badged POST HOC
uv run python scripts/export_demo.py --track Catalunya \
    --agent runs/dr-realw-loo-f2-s1/ckpt/000199884800.pkl \
    --agent-label "RL generalist (DR, 200M steps)" --agent-badge "POST HOC" \
    --name catalunya-generalist \
    --note "POST HOC: the agent is the final checkpoint (199,884,800 steps) of dr-realw-loo-f2-s1. It is shown because it completed the most R-dev circuits (11 of 12) of the eight leave-circuits-out width runs (docs/findings/width-test.md, Section 7); it was picked after those results were seen, not by a pre-registered rule." \
    --note "Training: domain randomisation on procedural strips (v3 grammar) with real width profiles from fold 2's circuits only (Hockenheim, Norisring, Oschersleben, Sochi, Melbourne, Zandvoort). Catalunya's widths were not in training, but the curvature grammar was fitted on all 12 R-dev circuits, so Catalunya is held out for widths only, not unseen."
uv run python scripts/render_demo.py docs/demo/catalunya-generalist.json \
    --name catalunya-generalist --rate 1.45 --hold 2.5 --frames --clip webp --stills worst,end \
    --title "Catalunya: an R-dev generalist ghost against the optimum" \
    --subtitle "POST HOC: checkpoint picked after the R-dev results, not by a pre-registered rule. Optimum: T_cont, not certified."
```

`render_demo.py` refuses to overwrite an existing frame sequence unless given `--overwrite`.

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
- **The MVP-1 lite clip is sped up.** It plays the timed lap at 1.3× real time so it stays
  under 60 s, and the frame says so next to the lap clock. The clock and Δt are lap times:
  Δt ends at +3.397 s, the lap delta above.
- **The Catalunya generalist is POST HOC.** It is the final checkpoint of
  `dr-realw-loo-f2-s1`. It is shown because it completed the most R-dev circuits (11 of
  12) of the eight leave-circuits-out width runs (`docs/findings/width-test.md`, Section
  7). It was picked after those results were seen, not by a pre-registered rule, so the
  ghost is badged POST HOC in the export, the figures, the viewer and the frames.
  - Catalunya is held out for widths only: the run's real width profiles came from fold 2
    (Hockenheim, Norisring, Oschersleben, Sochi, Melbourne, Zandvoort). Its curvature
    grammar was fitted on all 12 R-dev circuits, so Catalunya is not an unseen circuit.
  - Its lap is 81.154 s against the optimum's 75.647 s, a gap of 7.28% (T_cont, not
    certified). The run's `eval_rdev.json` gives a median of 81.153 s over its 10 laps
    and the same 7.28%. The exported lap differs from a fresh one-run `timed_laps` call
    by 1.7e-5 s.
  - It is a 200M-step DR network from the width study, not E0. The demo's real generalist
    waits for E0 and `protocol-v1`.
- Track geometry comes from the TUMFTM racetrack-database (LGPL-3.0).
