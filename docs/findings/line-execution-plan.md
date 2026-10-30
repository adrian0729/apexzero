# Line or execution: plan (POST HOC, descriptive; written before any R_TEST split number is computed, 29 Oct 2026)

**Status.** This is the line / execution split that `protocol-v1` left for later (Section 5: "descriptive and is not computed in LB-1 … Anything computed from them is POST HOC and says so"; Section 14). It runs after the lockbox findings ([`lockbox.md`](lockbox.md)) and the provenance check ([`lockbox-provenance.md`](lockbox-provenance.md)) were committed. It is not pre-registered with a tag. This plan is committed before any R_TEST quantity below is computed, so the measures and the reading are fixed in advance. It decides nothing and changes no number in `lockbox.md`.

## Question

At the median subject, E0's completed laps on R_TEST are 5.16% slower than the optimum. Is that time lost mostly in **the line** (where the car is across the track) or in **the execution** (how it drives that line: speed, braking and throttle points, steering timing)?

## Method: the optimum held near the car's own line

- **The fixed-line optimum.** For a completed lap with lateral offset n_a(s), solve the same minimum-time problem as the lap's reference, with one change. The offset may move at most ε from the car's line:
  max(n_a − ε, min(lo, n_a)) ≤ n ≤ min(n_a + ε, max(hi, n_a)),
  where lo and hi are the reference's own corridor bounds, −(w_right − 1) and w_left − 1. Its lap time T_ε is the fastest lap possible within ε of the car's line, with the environment's car, corridor and slip limit.
- **The split.** The car's lost time T_a − T* is exactly (T_a − T_ε) + (T_ε − T*). The first part can be recovered by better driving within ε of the car's line. The second needs the line itself to move by more than ε.
- **The recovery curve.** R(ε) = (T_a − T_ε) / (T_a − T*) is the share of the lost time that comes back within ε. The band grid is **ε = 0.05, 0.25, 0.5, 1 and 2 m**. ε = 0.05 m stands for the car's own line.
- **Why a curve and not one number.** In development on R-dev, the recovered share went from 27% at 0.05 m to 94% at 2 m on one lap. Any single ε would be arbitrary, so the whole curve is the result.

## Data

- **Rows.** The (subject, circuit) pairs of the **original** circuits with `circuit_completed` true in `outputs/lockbox/analysis.json`. These are the rows that enter the headline median gap: 41 rows over the five subjects. The mirrored and reversed variants are out of scope.
- **The car's lap.** `outputs/lockbox/laps/<subject>/<circuit>.npz` (`stations`, `tau`, `path_n`, `path_v`) and its `.json` (`completed`, `lap_time_s`), checked against the record's `npz_sha256`. Each row uses one **representative run**: the completed run whose lap time is closest to the row's median lap time (ties: the lowest run index). Laps are near-deterministic (median lap SD 0.002 s, `lockbox.md`). T_a is that run's lap time.
- **The reference and the geometry.** `outputs/lockbox/ocp/<circuit>__w+0.0.pkl`, checked against its record's `pkl_sha256`. T* is its lap time. Curvature and widths come from the pickle, at its own 3 m grid. **No file in `data/tracks/tumftm/` is opened, and no track is loaded.**
- **Alignment.** The lap's stations are the track's 2 m grid on [0, L], and the reference's grid spans the same L. The car's line on the reference's grid is the linear interpolation of `path_n` at the reference's stations.
- **Corners.** The sectors (`s_start_m`, `s_end_m`, `type`) come from `outputs/lockbox/corners.csv`; they tile the lap. The type (slow, medium, fast) is the lockbox's.
- **Comparison.** The R-dev dry run (`outputs/lockbox-dryrun/lb1-rdev-e0`) is computed the same way, on its 36 completed originals. Those numbers are **in-sample for widths**.

## Solver

- `apexzero.ocp.min_time.solve_min_time`, unchanged, with the lockbox's settings: ds 3 m, margin 1 m, tol 1e-6, max_iter 3,000, the default initial guess, and the lockbox's one retry from the failed iterate at tol 1e-5.
- It runs on a `Track` built from the pickle's own stations and curvature, with the band as its widths (w_left = upper bound + 1, w_right = 1 − lower bound). Its x, y and heading are zeros, which the solver does not use. Resampling such a track returns the pickle's arrays to within 1e-13 (checked on R-dev).
- Each solve gets the lockbox's shooting check (`apexzero.ocp.check.shooting_check`, without the track).
- The code is `scripts/line_execution.py`; nothing under `src/apexzero` or `scripts/lockbox.py` changes. It runs on the Mac, on 4 workers at nice 15. That is about 217 solves on R_TEST and 192 on R-dev, at roughly 45 s each: about 1.5 hours of wall time, $0.

## What is reported

Every number is read from the script's output, `outputs/posthoc/line-execution/<root>/split.json` (write-once).
1. **The recovery curve.** For each ε: R(ε) at the median subject (each subject's median over its rows, then the median over the five subjects, as in the headline), pooled over all rows (Σ(T_a − T_ε) / Σ(T_a − T*)), and its range over rows.
2. **The gap split.** G = G_exec(ε) + G_line(ε) per row, with G_exec = (T_a − T_ε)/T* and G_line = (T_ε − T*)/T*, at the median subject.
3. **The half-recovery band ε₅₀.** It is the ε at which R reaches 0.5, by linear interpolation on the grid ("below 0.05 m" or "above 2 m" outside it). It is given per row, with the median subject and the range.
4. **By corner type, at ε = 0.05 and 0.5 m.** For each sector, the execution part is the sector delta of t_a(s) − t_ε(s), and the line part the sector delta of t_ε(s) − t*(s) (`viz.sectors.sector_deltas`). Both are pooled over rows within each type, as shares of that type's lost time.
5. **How far the car's line is from the optimum's.** The median and 90th percentile of |n_a − n*| per row, and by corner type, pooled over the stations in that type's sectors.
6. **R-dev, the same,** labelled in-sample for widths.

## The reading, fixed now

- Take R(0.05) at the median subject, the share recoverable without moving the line:
  - **below 1/3:** the loss is "mostly line";
  - **above 2/3:** it is "mostly execution";
  - **otherwise:** "mixed".
- The curve and ε₅₀ are reported whatever the reading. Nothing else is read as a result.

## Checks (all reported)

- **The solver's noise floor.** Each of the 12 references is re-solved from its pickle's arrays with its own widths; |T_recon/T* − 1| is reported. On R-dev BrandsHatch it was 2.8e-5, with the inputs reproduced to 2e-14, so the solver lands on a slightly different point from near-identical input.
- **Each fixed-line solve** must converge, with a shooting error below 3e-3 (the lockbox's P2 limit). A failed solve leaves that row's value at that ε missing. It is listed, not imputed.
- **Consistency.** T_ε ≥ T* (1 − 1e-4), and T_ε does not increase with ε (to 1e-4 relative). T_a ≥ T_0.05 (1 − 1e-4): the car cannot beat the optimum on its own line, up to the grid. Violations are listed.
- **The band holds the line** at every station. The wrap jump |n_a(L) − n_a(0)| is reported per row.
- **The representative run's gap** is compared with the lockbox's gap for that row; the largest difference is reported.
- **Sector closure:** the sector parts sum to the lap totals within 1e-6 s; the analysis stops otherwise.

## Limits, stated now

- **"Line" means the lateral offset n(s) only.** Execution covers the rest: speed, braking and throttle points, steering timing, heading and slip.
- **ε = 0.05 m is not exactly the line.** The optimum may tidy the line by up to 5 cm.
- **The grid.** At 3 m, the discretisation is about 0.075% of a lap (Monza, `lockbox-provenance.md`). That is about 1.5% of the lost time.
- **The optimum knows the whole track in advance;** the car sees only its observations.
- **One run per row.**

## How the method was developed (R-dev only)

- The method was tried on one R-dev lap (BrandsHatch, `e0-s52`), with 10 solves at several bands and on a smoothed line.
- **Smoothing the line is not used.** The car's line is no wigglier than the optimum's: the RMS of the part above a ±10 m moving average is 0.065 m against 0.078 m. A line smoothed over ±10 m was slower than the raw one at every band tried.
- **Before this plan, on R_TEST,** the lead read only array shapes, the lap stations' spacing, and one reference's grid length, offset range and width range (Austin), to check the alignment. No agent line, speed or split number was read.

## Outputs

- `outputs/posthoc/line-execution/<root>/solves/` (one write-once JSON and pickle per solve, resumable), `recon/` (the noise-floor solves) and `split.json`.
- The findings note will be `docs/findings/line-execution.md`.
