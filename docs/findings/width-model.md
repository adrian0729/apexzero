# Width model (GW-2): results (exploratory, 26 Oct 2026)

Plan: [width-model-plan.md](width-model-plan.md). It fixed the report's statistics and bands in `bdfd442`, before the report was computed for any model. Its amendment 1 fixed model (2b) in `1bd85ec`, after (2a)'s report and before (2b)'s.

Code:
- `src/apexzero/tracks/widths.py` (2a) and `src/apexzero/tracks/reexpress.py` (2b);
- `scripts/width_report.py`;
- tests in `tests/test_stat_widths.py`, `tests/test_reexpress.py` and `tests/test_width_report.py`.

Results:
- `outputs/width-report/report.json`: default, copied and (2a). Computed once at `bdfd442`.
- `outputs/width-report/report-2b.json`: (2b). Computed once at `1bd85ec`.
- Each has a provenance block: plan commit and sha256, script sha256, HEAD, changed paths, and the sha256 of every R-dev centreline read. `src/` was clean for both.
- The two reports are force-added to git (26 Oct 2026, after review), because GW-3's criterion 4 reads (2a)'s. Their sha256s, which GW-4 checks before reading:
  - `report.json`: `5871664be5a6ff6ac882e3a38e131660fe16676f37ddd0a99ffba7294a42f014`;
  - `report-2b.json`: `89ffd2a3a1c49de902e880c44f4e0dfec38ed790ebd2551435778e2c89f37755`.
- Every other file under `outputs/` is local and not in git.

**Every number here is exploratory**, or descriptive where it says so. Post hoc readings are labelled as such. No agent was trained or evaluated. Only the 12 R-dev circuits were read, each through `lockbox_guard`. R_TEST and `outputs/sealed/` were not touched.

## 1. Summary

- **(2a) does not meet the bands.** 3 of its 90 checks fail, all in v3:
  - `share_1m`, in both folds. (2a) has 0.61 and 0.63 against H's 0.39 and 0.45, above the band's 0.55 and 0.61.
  - `spectrum:8-16`, in fold f1. A width SD of 2.2 cm against H's 0.9 cm, a factor of 5.8 in power against the band's 4.4.
  - **Every v1 check passes in both folds,** and the no-copy check passes everywhere.
- **(2b) does not meet the bands either.** 3 of 90 fail, all in fold f1:
  - `spectrum:8-16` in v1 (2.3 cm) and v3 (3.7 cm);
  - `share_1m` in v3 (0.61).
  - Fold f2 passes every check, and no-copy passes everywhere.
- **What follows, by the plan** (Section 4): neither model meets the bands. GW-4's criterion 4 cannot hold for either as the bands stand, so the lead takes this to the user before GW-3's pre-registration is tagged. Section 8 sets out the choice. The user chose option 1 on 27 Oct 2026: (2a), with criterion 4 restated (Section 8).
- **What both models do achieve.** They bring the corner-coupled statistics from the default model's near zero to R-dev's level, and they copy nothing:

  | Statistic (in-sample, all 12; v1 / v3) | R-dev | Default | Copied | (2a) | (2b) |
  |---|---|---|---|---|---|
  | Tight-knot inside range ≥ 1 m | 0.43 | 0.002 / 0.005 | 0.23 / 0.23 | 0.51 / 0.58 | 0.47 / 0.57 |
  | Tight-knot inside range ≥ 2 m | 0.16 | 0.000 / 0.001 | 0.07 / 0.08 | 0.09 / 0.14 | 0.08 / 0.17 |
  | W1, inside width at tight knots (m) | — | 0.16 / 0.18 | 0.30 / 0.31 | 0.26 / 0.25 | 0.20 / 0.20 |
  | No-copy: windows within 0.05 m RMS of an R-dev window | 0 | 0.0025 | 0.997 / 0.989 | 0 | 0 |

  The default model's share of 0.002–0.005 reproduces the width test's 0.4–0.5% (width-test, Section 2).
- **The failures are narrow, and of two kinds** (post hoc; Section 6). The 8–16 m band holds centimetre-scale width variation, where R-dev's widths are heavily smoothed. The v3 share overshoot is consistent with the tilted grammar's hairpin mix. Neither failure involves the inside dips or the widening at apexes, which every model passes.

## 2. What was computed

- **Real circuits.** The 12 R-dev circuits through `refine_reference`, on a 2 m grid: 54.9 km, with 2,469 tight-corner knots (|κ| ≥ 1/50 m⁻¹) and 267 corners.
- **Strips.**
  - 400 per (model, fold, variant), with the same keys for every model.
  - Curvature from the grammar fitted on all 12 circuits.
  - v3: the odd strips from `tilted(g, 2)`.
- **Folds.** The committed leave-out folds. Fold f1 fits on Brands Hatch, Moscow Raceway, Nürburgring, Sepang, Yas Marina and Catalunya, and is judged on Hockenheim, Norisring, Oschersleben, Sochi, Melbourne and Zandvoort. Fold f2 is the reverse.
- **Compute.** One Mac process at nice 10 with 2 threads, while another workflow shared the machine.
  - The default, copied and (2a) report took 71 s; the (2b) report took 329 s.
  - One 256-strip bank refresh (the training bank's per-update work) took 0.08 s for the default model, 0.07 s for copied, 0.08 s for (2a) and 15.9 s for (2b).
  - (2b) solves a 4-rung QP ladder per strip. Its cost on the A6000 was not measured.

## 3. R-dev itself (descriptive)

| | f1's circuits (F for fold f1) | f2's circuits | All 12 |
|---|---|---|---|
| Tight-corner knots | 1,369 | 1,100 | 2,469 |
| Inside range ≥ 1 m / ≥ 2 m | 0.45 / 0.13 | 0.39 / 0.19 | 0.43 / 0.16 |
| Apex dip / widening, r < 25 m (median; corners) | −0.36 / +0.88 m (32) | −0.60 / +0.72 m (26) | −0.55 / +0.78 m (58) |
| Apex dip, 25–50 m | −0.26 m (36) | −0.23 m (34) | −0.24 m (70) |
| Per-side power, 8–16 m (SD) | 1.35 cm | 0.91 cm | 1.14 cm |
| Per-side power, 256–512 m (SD) | 0.23 m | 0.32 m | 0.28 m |

Per circuit, the share of tight-corner knots whose inside width ranges over ≥ 1 m:
- Norisring 1.00 (≥ 2 m: 0.91);
- Moscow Raceway 0.71;
- Yas Marina 0.54;
- 0.41–0.44 at Brands Hatch, Nürburgring, Sochi, Melbourne, Catalunya and Zandvoort;
- Hockenheim 0.34, Oschersleben 0.13, Sepang 0.06.

Circuits differ far more from each other than the folds do.

## 4. Results per check

The band rule is fixed in plan Section 4: the generator X may be no further from the held-out circuits H than the fitting fold F is, plus a slack.

**`share_1m`** (slack 0.10):

| Fold / variant | H | F | Band | Default | Copied | (2a) | (2b) |
|---|---|---|---|---|---|---|---|
| f1 / v1 | 0.392 | 0.452 | 0.23–0.55 | 0.002 ✗ | 0.217 ✗ | 0.530 ✓ | 0.489 ✓ |
| f1 / v3 | 0.392 | 0.452 | 0.23–0.55 | 0.005 ✗ | 0.211 ✗ | **0.613 ✗** | **0.615 ✗** |
| f2 / v1 | 0.452 | 0.392 | 0.29–0.61 | 0.002 ✗ | 0.243 ✗ | 0.560 ✓ | 0.495 ✓ |
| f2 / v3 | 0.452 | 0.392 | 0.29–0.61 | 0.005 ✗ | 0.246 ✗ | **0.630 ✗** | 0.566 ✓ |

**Per-side power at 8–16 m** (band: |log₂(X/H)| ≤ |log₂(F/H)| + 1), as an SD in cm:

| Fold / variant | H | F | Band | Default | Copied | (2a) | (2b) |
|---|---|---|---|---|---|---|---|
| f1 / v1 | 0.91 | 1.35 | 0.43–1.91 | 0.21 ✗ | 1.36 ✓ | 1.82 ✓ | **2.29 ✗** |
| f1 / v3 | 0.91 | 1.35 | 0.43–1.91 | 0.45 ✓ | 1.42 ✓ | **2.18 ✗** | **3.67 ✗** |
| f2 / v1 | 1.35 | 0.91 | 0.64–2.85 | 0.21 ✗ | 1.06 ✓ | 1.52 ✓ | 1.49 ✓ |
| f2 / v3 | 1.35 | 0.91 | 0.64–2.85 | 0.45 ✗ | 1.24 ✓ | 1.70 ✓ | 2.33 ✓ |

The default model fails at the band's lower end; (2a) and (2b) at its upper end.

**Every other check passes for (2a) and (2b)** in both folds and both variants:
- the three W1s;
- `share_2m`;
- the apex dip and widening in all five radius bins (all bins judged: H has ≥ 14 corners in each);
- the spectrum from 16 m to 512 m;
- no-copy.

The largest structural array is 15 values for both models. The empirical share is 0 of 2,791–3,567 windows in every cell, against a real reference of 0. Nearest-window RMS medians are 0.27–0.32 m.

**The reference models.**
- **Default:** fails 8–10 checks per (fold, variant). It fails `share_1m` everywhere and `share_2m` in f1, and `apex_widen:<25` everywhere. In the spectrum bands from 16 to 256 m it is 50–4,700 times below H; at 8–16 m 4–42 times, and it passes one cell there; at 256–512 m it is 5 times below H in f1 (a miss) and 2.5 times in f2 (a pass). Its empirical no-copy share is 0.0025 (3 of 1,207 smooth windows), against a real reference of 0. See Section 6.
- **Copied:** fails `share_1m` and `apex_widen:<25` in every cell, and no-copy by construction (0.98–1.00).

## 5. Expected directions (plan Section 5 and amendment 1) against the results

| | Expected (before computing) | Result | Borne out? |
|---|---|---|---|
| E1 | Default misses both shares in all four cells, the apex dip under 50 m, and several spectrum bands 8–128 m; may pass `w1_all` | Misses `share_1m` in all four and `share_2m` in f1. Passes `share_2m` in f2 and every apex dip. Misses `apex_widen:<25`, and the spectrum bands 16–256 m by factors of 50–4,700 (8–16 m in 3 of 4 cells). Passes `w1_all` | **In part** (the apex dip: it is near 0, but so is H's beyond 25 m, and the < 25 m bin's F–H distance absorbs it) |
| E2 | Copied passes `w1_all` and the spectra, misses the tight-bin apex dip, has shares below H, and fails no-copy | Passes `w1_all` and every spectrum band. Misses `apex_widen:<25` but passes `apex_dip:<25`. Shares 0.21–0.25 against H's 0.39–0.45. No-copy 0.98–1.00 | **Yes**, except the dip itself (the bin's F–H distance absorbs it) |
| E3 | (2a) meets the bands in both folds and variants; likeliest misses the < 25 m bins and fold f1 | Misses 3 checks, all in v3 (`share_1m` in both folds, `spectrum:8-16` in f1). Passes every apex bin. No-copy passes | **No** (narrowly; not where expected) |
| E4 | (2a)'s shares much closer to H than the default model's | 0.53–0.63 against H's 0.39–0.45, with the default at 0.002–0.005 | **Yes** |
| E5 | (2b)'s v3 shares below (2a)'s; possible v1 undershoot | f2/v3 0.566 against 0.630 (cured); f1/v3 0.615 against 0.613 (not cured). v1 0.49–0.50 against 0.53–0.56, no undershoot | **In part** |
| E6 | (2b) meets the bands (held weakly) | Misses 3 checks, all in fold f1 | **No** |

## 6. Post hoc readings (after the results were seen; none changes the verdict)

- **The 8–16 m band is centimetre-scale.**
  - R-dev's per-side power there is an SD of 0.9–1.35 cm. `from_centreline` smooths the TUMFTM widths with a spline of 5 cm RMS tolerance, so this band is mostly what the data pipeline leaves.
  - The failing generator values are SDs of 2.2–3.7 cm. For the agent, whose termination margin is 1 m, that is physically negligible.
  - The band's log-ratio rule has no absolute floor, which was an oversight in the plan.
  - With a floor of 5 cm (both powers below (0.05 m)² passing), (2a) would fail 2 checks (the two v3 shares) and (2b) 1 (f1/v3 `share_1m`). This is computed from the report's values and is not a band.
- **The v3 share overshoot is consistent with the tilted grammar's corner mix.**
  - `tilted(g, 2)` weights corners by (1/r_min)², so v3's tight-corner knots sit in tighter corners than R-dev's. Those corners have the largest drawn dips and widenings.
  - In-sample, (2a)'s share is 0.51 in v1 and 0.58 in v3 under the same width model. The difference comes from curvature alone.
  - A share conditioned on radius would separate the two effects. It was not computed.
- **The copied model trained on half of R-dev's tight-knot width variation.** It shows 0.22–0.25 against 0.43, and no apex coupling: its median apex dip and widening are within 0.005 m of zero in every bin. The completion gains of the width test and H4R came from training on it. So corner-coupled width changes were not what produced those gains, and they may or may not add to them. GW-3 measures this.
- **The empirical no-copy check has false positives on smooth roads.** The default model, which stores no profile, has 3 of 1,207 varying windows within 0.05 m RMS of an R-dev window. These are slowly varying 200 m windows that happen to match a slow stretch of a real road. The structural check is the one that shows a model cannot copy. The empirical one only separates copying (about 1.0) from not copying (about 0).

## 7. Model (2b): what the port adds, and its cost

- **The port matches numpy's `refine_reference`.** It was compared with refine's own pieces on 80 raw strips (plan amendment 1):
  - the same regularisation weight on 75 of 78;
  - the 3 others are threshold cases one rung apart;
  - widths agree to p99 ≤ 0.07 m and curvature to p99 ≤ 0.003 m⁻¹.
- **How often it acts.** In that sample, 13 of 80 strips triggered re-expression. Most strips' raw drivable margin is already ≤ 0.6. 6 of the 12 R-dev circuits trigger.
- **What it changes,** against (2a): smaller dips and widenings in some bins (f1 r < 25 m: dip −0.12 against −0.26 m, widening +0.65 against +1.09 m), because its width model is fitted on raw profiles. More 8–16 m power, from the re-measured widths.
- **Cost.** 15.9 s per 256-strip refresh on the Mac CPU, against 0.08 s for (2a), about 200 times slower. On the A6000 the QP's 6,000 sequential FISTA iterations per refresh would add an unmeasured share to each 200M run. That must be measured before any GW-3 arm uses (2b).

## 8. For the user: the choice the plan left open (decided 27 Oct 2026)

**Decided by the user on 27 Oct 2026: option 1.**
- GW-3's stat-model arms use (2a).
- GW-4's criterion 4 is restated: every v1 check in (2a)'s committed report (`report.json`, sha256 above) passes in both folds, and the v3 misses are limited to `share_1m` (both folds) and `spectrum:8-16` (fold f1).
- This is a deviation from the plan's bands. It is logged, dated 27 Oct 2026 and with its reason, in the GW-3 draft's Section 10 (`docs/prereg/gw3-width-round.md`), before the tag.
- The readings in Section 6 that bear on it stay post hoc. No result here changes: neither model meets the bands.

The plan's branch is "neither meets: the lead takes the report to the user before GW-3's pre-registration is tagged". The options as set out on 26 Oct 2026, with my recommendation first:

1. **GW-3 with (2a), with GW-4's criterion 4 restated before the tag** (recommended).
   - For example: "the bands hold in v1 in both folds, and every v3 check passes except `share_1m` (overshoot ≤ 0.25) and the 8–16 m band".
   - (2a) is the direct fit, costs nothing extra to generate, passes every v1 check, and its misses are the two kinds above.
   - It must be logged as a deviation from this plan's verdict rule, decided before any GW-3 run.
2. **GW-3 with (2b),** restated the same way. It passes fold f2 fully and misses f1 at 8–16 m in both variants. It costs about 200 times more to generate, its GPU cost is unmeasured, and it rests on a port validated to p99 0.07 m. Not recommended over (2a).
3. **GW-3 without a stat-model arm:** baseline against copied leave-out only.
   - It still answers the v1 transfer question (follow-through Section 9.1).
   - But `generator-v1` could then adopt no statistical model, and GW-4 falls to default widths unless #12 (copying R-dev widths) is decided for copying.
4. **One more model iteration before GW-3,** for example (2a) with its corner coupling conditioned on the radius mix, or a short-scale noise floor.
   - It would be tuned after seeing the held-out folds, which weakens the bands as a test.
   - It is arguably outside #11's budget (GW-2 has run).

Whichever is chosen, the GW-3 draft (`docs/prereg/gw3-width-round.md`) fixes everything else. Only its stat-model arm (2a or 2b) and the wording of criterion 4 waited on this; both are now fixed there.

## 9. Procedure log

| When (26 Oct 2026) | What | Effect |
|---|---|---|
| Before `bdfd442` | (2a), the report and their tests were written. Tests on R-dev-fitted (2a) checked validity only (the cap, clips, finite values, more within-strip range than the default model). The fitted values were not looked at | As planned |
| `bdfd442` | Plan, (2a), report and tests committed together | Bands fixed |
| After `bdfd442` | The report ran once (71 s). (2a) missed 3 checks | (2b) to be built |
| Before `1bd85ec` | (2b) built and validated against numpy on 80 raw strips (not report statistics). The report script gained the (2b) model: the same statistics and bands, with (2b)'s strips generated in batches | As amendment 1 records |
| `1bd85ec` | Amendment 1, (2b), the report change and tests committed | (2b) fixed |
| After `1bd85ec` | The (2b) report ran once (329 s). (2b) missed 3 checks | Neither meets |
| After the (2b) report | `REEXPRESS_BATCH` raised from 64 to 256 (one batch per 256-strip refresh), to cut GPU latency. It only touches `generate_bank`, which the report used only for the timing line | No band statistic changed; the 15.9 s timing used 64 |

## 10. What this does not show

- **Nothing about driving.** No agent saw these strips. Whether (2a) training transfers to held-out circuits is GW-3's question. (2b) was not chosen for GW-3 (Section 8), so its transfer is not tested.
- **Not whether the bands are the right target.** They measure resemblance to R-dev, not what makes an agent robust. The copied model, which trained the gains so far, fails them too.
- **No R_TEST geometry was used.** The comparison is within R-dev, fold against fold.
