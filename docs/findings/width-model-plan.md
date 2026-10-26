# Width model (GW-2): the report's bands, fixed before the report (26 Oct 2026)

**Exploratory.** This plan fixes what the width report measures and what counts as a generator matching R-dev (the bands), before the report is computed for any width model. It is item GW-2 of the follow-through plan ([`docs/plan/follow-through.md`](../plan/follow-through.md), Section 7.4), under decision #13: build model (2a) first, and (2b) only if (2a) misses these bands. The bands are also GW-4's criterion 4 ("the width-report bands hold").

**What existed when this was written.**
- The (2a) model, `src/apexzero/tracks/widths.py`, opt-in through `fit_grammar(..., stat_widths=True)` and `train_dr.py --stat-widths [--stat-widths-from ...]`, with `tests/test_stat_widths.py`.
- The report, `scripts/width_report.py`, with `tests/test_width_report.py`. Its statistics were tested only on planted synthetic profiles.
- No report statistic had been computed for any generator against R-dev, and the fitted model's values had not been looked at. The only statistic computed on (2a) strips fitted on R-dev was a validity test: the median within-strip range of the left width is larger than the default model's. It was not compared with R-dev.

The report refuses to run unless this file is committed and unchanged, and it refuses to overwrite its output, so it runs once per model.

## 1. The (2a) model in brief

Per strip, from summaries fitted on the fitting circuits as evaluated (`refine_reference` geometry, 2 m grid). No profile is stored; the model's largest array has 15 values.
1. **Level:** a fitting circuit's median total width, chosen uniformly, × a log-normal jitter (σ 0.08), split evenly between the sides.
2. **Slow variation and multi-scale residual noise per side:**
   - two Gaussian processes: a common one (half the total-width variation) and a differential one (half the left–right difference);
   - their SDs per octave band (4–4,096 m) are the fitting circuits' pooled residual SDs, after the corner terms are removed. The bands from 256 m up are the slow variation;
   - one log-normal roughness multiplier per strip.
3. **Corner coupling:**
   - at each generated corner the reference line moves inside by δ and the road widens by τ, along the corner's own curvature profile;
   - δ and τ are drawn from two-piece normals (q16, median, q84) fitted per radius bin (< 25, 25–50, 50–100, 100–200, ≥ 200 m).
4. **Clip** each side to 2.5–16 m, then the generator's existing inside cap.

Curvature is unchanged. With the option off, the generator, the training bank and `train_dr.py` are bit-identical to before (tests).

## 2. What the report compares

- **Folds:** the committed leave-out folds of `width-test-loo-plan.md`.
  - A model fitted on fold f's circuits F is judged on the other fold's circuits H, which it never saw.
  - Fold f1 fits on `R_DEV[0::2]` and is judged on `R_DEV[1::2]`; fold f2 the reverse.
  - The fit on all 12 circuits, the candidate for `generator-v1`, is reported in-sample and is not judged.
- **Models:**
  - the default model (no width option);
  - the copied model (`--real-widths --real-widths-from F`, as in the leave-out runs);
  - (2a), fitted on F;
  - later (2b), if it is built.
- **Real circuits:** the 12 R-dev circuits as every evaluation drives them, from `refine_reference` on a 2 m grid. Each name passes `lockbox_guard`. Nothing on R_TEST is read.
- **Strips:**
  - 400 strips per (model, fold, variant), keys `split(PRNGKey(20261026), 400)`, the same keys for every model;
  - curvature from the grammar fitted on all 12 circuits, as in training;
  - **v1:** the base grammar. **v3:** the odd strips from `tilted(g, alpha=2)`, the even ones from the base grammar.

## 3. Statistics

All on the 2 m grid. A **tight-corner knot** has |κ| ≥ 1/50 m⁻¹. Its **inside** is the side it turns toward.

| Name | Definition |
|---|---|
| `w1_all` | W1 between per-side widths at all knots, sides pooled |
| `w1_inside_tight`, `w1_outside_tight` | W1 between inside (outside) widths at tight-corner knots |
| `share_1m`, `share_2m` | Share of tight-corner knots whose inside width ranges over ≥ 1 m (≥ 2 m) in the 100 m up to and including the knot. Open strips skip their first 50 knots |
| `apex_dip:<bin>`, `apex_widen:<bin>` | For each `corners.segment` corner: the inside (outside) width at the apex (max of \|κ\| smoothed over 10 m) minus that side's median over the 100 m before and the 100 m after the corner. The median per radius bin. Strip corners whose base would leave the strip are skipped |
| `spectrum:<band>` | Per-side width variance per octave band, 8–16 m to 512–1,024 m. Welch method: 2,048 m segments every 1,024 m (wrapped on circuits), linear detrend, periodic Hann window, both sides pooled |
| no-copy | (a) Structural: the largest array of the width model. (b) Empirical: of the strips' 200 m windows that vary (SD ≥ 0.25 m on either side; first 100 strips, every 200 m), the share within 0.05 m RMS of any 200 m window of any of the 12 R-dev circuits, in either side order |

Descriptive only: the total-width and half-difference spectra, per-circuit values, the in-sample all-12 comparison, the fitted model values, and the time to generate one 256-strip bank refresh.

## 4. The bands

**Rule.** The fitting fold F is the reference. A generator X fitted on F should be no further from the held-out circuits H than F's own roads are, plus a fixed slack.

| Statistic | Passes when | Slack |
|---|---|---|
| `w1_all`, `w1_inside_tight`, `w1_outside_tight` | W1(X, H) ≤ W1(F, H) + slack | 0.25 m |
| `share_1m`, `share_2m` | \|X − H\| ≤ \|F − H\| + slack | 0.10 |
| `apex_dip`, `apex_widen`, per radius bin | \|X − H\| ≤ \|F − H\| + slack. Judged only where H has ≥ 5 corners in the bin and X ≥ 20. If F has none, \|F − H\| is taken as 0 | 0.5 m |
| `spectrum`, per band 8–16 m to 256–512 m (6 bands; 512–1,024 m reported only) | \|log₂(X/H)\| ≤ \|log₂(F/H)\| + slack | 1 (a factor of 2) |
| no-copy (folds f1, f2 and all 12) | Largest array ≤ 64 values, and the empirical share ≤ max(0.001, the share for H's real windows against F's; for all 12, each circuit against the other 11) | — |

**Why these slacks.**
- **0.25 m (W1).** It is about the smallest width-related margin effect measured on these agents: H4R's v3 buffer shift was +0.27 m, and GW-1's clearance shifts at former crash corners were +0.27 and +0.28 m.
- **0.10 (shares).** The default generator's share is about 0.005 and the real crash roads' 0.79 (width-test, Section 2). 0.10 is about an eighth of that contrast.
- **0.5 m (apex).** A bin median rests on far fewer corners than a W1, and 0.5 m is half the termination margin (1.0 m).
- **Factor 2 (spectrum).** One octave band's power may be off by up to a factor of 2 beyond the fold-to-fold difference.
- **512–1,024 m is reported only.** Window leakage from slower variation dominates it.
- **No-copy.** A copied window matches to float32 rounding (about 1e-6 m). Two different roads that vary rarely agree within 0.05 m over 200 m, and the real H-against-F rate measures how rarely.

**Verdict.** (2a) **meets the bands** only if every judged check passes in both folds and both variants, and the no-copy check passes for f1, f2 and all 12 in both variants. That is up to 21 checks per (fold, variant), plus 6 no-copy checks. The report lists every failed check by name. Nothing is averaged across checks.

**What follows.**
- **(2a) meets the bands:** GW-3's stat-model arms use (2a), and (2b) is not built (#13).
- **(2a) misses:** build (2b) and report it once with these bands (`--models 2b --tag 2b`). Bands and statistics are unchanged.
  - (2b) generates a strip as (2a) does, then re-expresses it on a smoothed reference, as `refine_reference` does for 6 of the 12 circuits. That needs a JAX port of the smoothing: the box-constrained offset problem of `_offsets_qp` solved with a fixed number of iterations, then the re-measured widths (`rebase`) and the new reference curvature, then the inside cap.
  - Generated strips never exceed the drivable margin, so `refine_reference`'s own trigger never fires on them. (2b) therefore re-expresses a share of strips regardless (6 of the 12 circuits are re-expressed). That share and the regularisation weight are fixed in an amendment to this plan before (2b)'s report is computed.
- **(2b) meets the bands:** GW-3 uses (2b).
- **Neither meets:** the lead takes the report to the user before GW-3's pre-registration is tagged. GW-4's criterion 4 could not then hold for either model.

## 5. Expected directions (stated before computing)

| | Expected |
|---|---|
| E1, default | Misses `share_1m` and `share_2m` in all four (fold, variant) cells: its share stays near 0.005. Misses the apex dip in the bins under 50 m, and several spectrum bands from 8 to 128 m, where it is more than 2× below H. May pass `w1_all` |
| E2, copied | Passes `w1_all` and the spectra (they are real widths). Misses the apex dip in the tight bins, because its width changes are not placed at corners. Its tight-knot shares lie below H. Fails no-copy by construction (share near 1) |
| E3, (2a) | Meets the bands in both folds and both variants (held with moderate confidence). The likeliest misses are the < 25 m apex bins, which have few corners, and fold f1 (judged on Norisring, Sochi and Melbourne, the hardest width cases; width-test, Section 7). No-copy passes: 15 values, and an empirical share near 0 |
| E4, (2a) against default | In every cell, (2a)'s tight-knot inside-range shares lie much closer to H than the default model's |

## 6. What this does not do

- **No agent.** The report trains and evaluates nothing, and it says nothing about completion. GW-3 tests that.
- **R-dev erosion.** It reads R-dev geometry only, as the generator fit already does, and evaluates no agent there. So it adds no use of R-dev under #11's erosion bound.
- **Cost.** A few minutes of one Mac process at nice 10.
