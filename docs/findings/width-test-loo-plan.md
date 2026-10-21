# Width test, leave-circuits-out: design fixed before the runs (21 Oct 2026)

The first width test (`docs/findings/width-test.md`) trained generalists on strips whose
per-side widths are segments of the R-dev circuits' own profiles. R-dev completion rose from
5.4 to 8.2 of 12 (v1) and from 5.8 to 8.5 (v3). But the profiles came from the same circuits
the agents are evaluated on, so part of the gain could be seeing those exact width patterns
rather than learning to handle varying width. This test separates the two.

## Design

- **Folds**, fixed by alternating through `apexzero.tracks.corners.R_DEV` in its fixed order,
  so nothing is chosen by hand:
  - fold 1 widths: `R_DEV[0::2]` = Brands Hatch, Moscow Raceway, Nürburgring, Sepang,
    Yas Marina, Catalunya;
  - fold 2 widths: `R_DEV[1::2]` = Hockenheim, Norisring, Oschersleben, Sochi,
    Melbourne, Zandvoort.
- **Runs**: the v3 configuration (`--tilt-fraction 0.5 --tilt-alpha 2`) with
  `--real-widths --real-widths-from <fold>`, 200M steps, seeds 0–3 for each fold: 8 runs
  (`dr-realw-loo-f{1,2}-s{0..3}`). Curvature is generated as before, from all 12 circuits;
  only the widths are held out.
- **Primary outcome**: each run's completion on the **held-out** circuits, those of the other
  fold (0–6), from the standard R-dev evaluation (`eval_rdev.json`; a circuit counts as
  completed when more than half its runs finish).
- **Comparison**: the v3 baseline seeds 0–4 (`dr-pilot-v3-tilt50a2`, `-s1` to `-s4`) on the
  same held-out circuits, fold by fold (10 values: 5 seeds × 2 folds), against the 8
  leave-out runs. One-sided exact permutation test on the mean (leave-out above baseline).
  Labels are exchanged within each fold only, since the two folds' held-out circuits
  differ in difficulty. The analysis is `scripts/width_loo_analysis.py`, committed
  before any leave-out run finished.
- **Secondary**:
  - the leakage estimate: the in-sample real-width v3 runs (`dr-realw-v3-s0..3`) on the same
    held-out circuits minus the leave-out runs;
  - per-circuit completion, with Sochi, Norisring, Yas Marina and Melbourne named;
  - the median gap to the optimal lap on completed circuits.
- **Reading**: if the leave-out runs beat the baseline on held-out circuits, the gain is width
  robustness and not only memorised width patterns. If they match the baseline while the
  in-sample runs beat it, the gain was mostly leakage. Anything between is reported as the
  split of the effect.

The R_TEST lockbox is untouched: no run is evaluated on it.
