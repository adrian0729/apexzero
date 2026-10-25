# Gate diagnostics (GW-1): results (exploratory, 25 Oct 2026)

Plan: [gate-diagnostics-plan.md](gate-diagnostics-plan.md), committed in `916edaa` before anything here was computed. Its amendment 1 (Section 10, commit `245a2c4`) fixed the points the plan left open; it too was committed before any evaluation. Code: `scripts/gate_diagnostics.py` (stages `time`, `curves`, `adr`, `crashes`, `report`), tests: `tests/test_gate_diagnostics.py`.

Results are in `outputs/gate-diagnostics/`, each file with a provenance block (commit, changed paths, the sha256 of every checkpoint and evaluation read):
- `timing.json` (D0);
- `curves.json` (D1, with the platform check);
- `adr.json` (D2);
- `crashes.json` (D3);
- `descriptive.json` (Section 4.4);
- `evals/<run>/<step>.json`, `adr/<run>.json` and `traces/<run>.npz` (the per-network records).

Files under `outputs/` are local and not in git.

**Every number in this note is exploratory**, or descriptive where it says so. Each is reported against the direction the plan stated before computing (its Section 7). Nothing was trained, nothing was pre-registered with a tag, and the readings in Section 8 steer the bounded lever slots of decision #11 without deciding anything confirmatory. Only the 12 R-dev circuits and generated A_DR levels were used. Every circuit name passed `viz.export.lockbox_guard` before any rollout, gaps were measured only against `outputs/ocp/<R-dev circuit>.pkl`, and `outputs/sealed/` was not read. R_TEST stays sealed. Under #11's erosion bound, GW-1 counts as one use of R-dev.

## 1. Summary

- **D0: everything ran on the Mac.** One evaluation took 88 s alone and at most 68 s as one of 4 concurrent processes (t₄). That gives W = 168 × 68.4 / 4 s = 0.80 h, under the 6 h limit. The 4 concurrent results were identical to the alone one. Every curve point and every D3 trace comes from one machine (Darwin arm64).
- **D1: the late curve is "rising", but the curve as a whole is U-shaped.**
  - Mean circuits completed C over the 28 networks:

    | 10.0M | 18.2M | 33.1M | 60.3M | 109.9M | 199.9M |
    |---|---|---|---|---|---|
    | 8.79 | 8.36 | 7.07 | 6.36 | 5.54 | 7.11 |

  - The late change d_late = C(199.9M) − C(109.9M) has mean +1.57 [+0.96, +2.18]. 19 networks rise, 6 stay level and 3 fall. By the plan's rule that is **rising**.
  - The mid change d_mid = C(199.9M) − C(60.3M) is +0.75 [−0.25, +1.75], not resolved from 0.
  - 22 of 28 networks have their best C before the final (13 of them at 10.0M), by +3.4 circuits on average among those 22 (+2.6 over all 28).
  - Descriptive, added after the results were seen: the early completions are slow laps. The median lap gap of completed circuits falls from 54% at 10.0M to 7.8% at the final. All 157 (network, circuit) pairs completed at both 10.0M and the final are faster at the final.
- **E2 (platform) is borne out, more strongly than expected.** The Mac re-evaluation of the 28 finals matches the stored A6000 evaluations on 336 of 336 completion calls, with C equal for every network. 334 of 336 completion fractions are identical, but no lap time is identical bit for bit (largest difference 0.40 s).
- **D2: the finals have mastered their own distribution.**
  - Deterministic A_DR completion is 0.961–0.989 (arm medians 0.969–0.985). The stochastic window completion is 0.955–0.982, all 28 "late".
  - Every network's A_DR completion is above its R-dev c̄, by +0.06 to +0.74. The medians over the 28 are 0.976 (A_DR) and 0.667 (R-dev c̄).
  - This is Section 8's row "median A_DR ≥ 0.90 while R-dev c̄ ≤ 0.7": the R-dev shortfall is transfer.
- **D3: margin-consistent in both variants on the primary measures. A secondary measure weakens the v1 margin reading.**
  - **Inside-edge exits.** 97% (v1) and 92% (v3) of the baselines' terminations are inside-edge corridor exits at corners.
  - **Q1.** On v1's 9 gain circuits, real-width networks terminate 134 times fewer than baselines (301 against 167). Inside-edge corridor exits at former crash corners fall by 163, so s_in = 1.22: they account for all of the drop and more. In v3, 3 gain circuits give s_in = 1.00 (ΔT = 40).
  - **Q2.**
    - Δc_v1 = +0.27 m [+0.03, +0.49], inside v1's band (0.22–0.88 m).
    - Δc_v3 = +0.28 m [+0.20, +0.36], inside v3's band (0.135–0.54 m).
    - On clearing passages only (secondary, selection-biased): Δc_v1 = +0.11 m [−0.08, +0.29], below the band with a CI that includes 0, and Δc_v3 = +0.16 m [+0.10, +0.28].
- **Expected directions.**

  | | Borne out | Not borne out |
  |---|---|---|
  | E1 | 2 of 4 parts: the v1 real-width advantage by 60.3M; ≥ 1/3 best before the final | C rising from 10M, with most of the rise before 60.3M; a flat or unresolved late curve |
  | E2 | all of it | |
  | E3 | deterministic A_DR | the stochastic level (0.96–0.98, not 0.8–0.9) |
  | E4 | all of it | |
  | E5 | v1 | v3 (its CI excludes 0) |

- **For the two levers (Section 8 of the plan).**
  - D1 "rising" points to the **budget lever** (GW-6's own condition, "curves still rising"). D3 "margin-consistent" points to the **margin lever** (GW-5's training-margin arm). #11 allows one slot of each.
  - D2 qualifies the budget lever: more steps on the same generator are unlikely to close the R-dev half alone.
  - Post hoc, the late rise recovers only part of an earlier fall, and the curve ends below its 10M level (Section 8). The evidence for the budget lever is therefore weaker than its label.

## 2. What was computed

- **Networks.** The 28 same-code runs of the plan's Section 2: H4R's 20 (v1 and v3, baseline and real width, seeds 10–14) and the 8 v3 leave-out runs. Each final was checked against its bound sha256 (`outputs/h4r/manifest.json` and `src/apexzero/fields/prereg.json`) before it was read. The stored `eval_rdev.json` files of the H4R runs were checked against the manifest. The sha256 of every other file read is recorded.
- **D1 rollout.** `rollout_laps` is `eval.laps.timed_laps` line for line, with more outputs: every run's s, n, v_x, alive flag and termination flags at every step. The tests check it:
  - against `timed_laps` on the timed checkpoint (Norisring and Zandvoort, 10 runs; every field equal);
  - against `scripts/eval_rdev.py`'s JSON on the same checkpoint (all 12 circuits; completion, median lap and gap equal);
  - against `timed_laps` with a fixed hand-written policy.
- **D1 grid.** Six checkpoints per run (indices 0, 3, 6, 9, 12, 15): 168 evaluations, `scripts/eval_rdev.py`'s protocol (default car, 10 runs, seed 0, two laps, lap 2 timed). C counts circuits with completion > 0.5; c̄ is the mean completion fraction.
- **D2.** The 1,000 A_DR levels (`make_levels(PRNGKey(2026), 'a_dr')`, grammar fitted on R-dev; the keys are checked against the pilot audit's construction).
  - Primary: the mean action without observation noise over one 1,200-step window per level.
  - Secondary: `audit.rollouts.rollout_windows` at its defaults, 4 windows per level, key `fold_in(PRNGKey(2026), 1)`.
- **D3.** The final traces of H4R's 20 networks.
  - Corners come from `tracks.corners.segment` at its defaults on each refined track: 265 corners on the 12 circuits. W_k = [s_start − 50, s_end + 50] m.
  - Clearance c = w_in − 1.0 − σn, using the environment's own width spline.
  - Amendment 1's rules: passages, overlapping windows, crash, time-out and the Q2 bootstrap.
- **Compute.** All on the Mac, with at most 4 processes at a time for this item, each at nice 15 with 2 XLA threads. PS-P4's 4-process pool ran at the same time (its s-OCP solves during D1, its generalist and specialist runs during D2), and another workflow also shared the machine. The load was not recorded.

  | Stage | Wall time |
  |---|---|
  | D0 | 165 s |
  | D1 (167 evaluations after D0's) | 47.5 min |
  | D2 | 20.4 min |
  | D3 | 9 s |

## 3. D0: timing (`timing.json`)

| | Evaluation wall time (s) | Process wall time (s) |
|---|---|---|
| Alone | 88.1 | 91.5 |
| 4 concurrent (each) | 68.4, 65.8, 65.8, 67.2 | 72.3–72.4 |

- t₄ = 68.4 s (the longest of the four, amendment 1). W = 2,871 s = **0.80 h** (0.84 h with process start), so the rule (≤ 6 h) sends everything to the Mac. The A6000 was not used.
- The reported estimate was about 30 s per evaluation and W ≈ 21 min. The measured time was about 2.2–2.9 times that. Under the other workflow's load the alone run was slower than the concurrent ones; this was not investigated, and the rule does not depend on it.
- The four concurrent results were identical to the alone run's: completion flags and lap times on all 12 circuits. The alone run is D1's 10.0M point for `h4r-v1-base-s10`.

## 4. D1: R-dev completion against training steps (`curves.json`)

### 4.1 C and c̄ by arm (mean, with the range over networks)

| Arm (n) | 10.0M | 18.2M | 33.1M | 60.3M | 109.9M | 199.9M |
|---|---|---|---|---|---|---|
| v1 baseline (5) | 8.8 (8–10) | 7.0 (3–9) | 6.8 (4–9) | 5.4 (4–7) | 3.8 (2–6) | 5.0 (3–7) |
| v1 real width (5) | 7.2 (6–8) | 6.0 (3–7) | 6.6 (2–9) | 6.6 (5–9) | 5.0 (3–6) | 7.6 (4–9) |
| v3 baseline (5) | 8.2 (5–10) | 8.2 (7–9) | 6.8 (4–11) | 5.8 (2–9) | 6.4 (6–8) | 7.2 (6–8) |
| v3 real width (5) | 9.0 (7–11) | 9.0 (6–12) | 7.0 (4–10) | 5.8 (4–9) | 6.2 (3–8) | 7.8 (6–9) |
| v3 leave-out (8) | 10.0 (8–12) | 10.4 (9–11) | 7.8 (5–11) | 7.5 (5–10) | 6.0 (3–8) | 7.6 (4–11) |
| **All 28** | **8.79** | **8.36** | **7.07** | **6.36** | **5.54** | **7.11** |

| c̄ | 10.0M | 18.2M | 33.1M | 60.3M | 109.9M | 199.9M |
|---|---|---|---|---|---|---|
| v1 baseline | 0.74 (0.67–0.83) | 0.59 (0.26–0.75) | 0.57 (0.33–0.76) | 0.45 (0.33–0.58) | 0.31 (0.18–0.50) | 0.41 (0.24–0.58) |
| v1 real width | 0.60 (0.50–0.67) | 0.50 (0.25–0.58) | 0.55 (0.17–0.75) | 0.55 (0.42–0.75) | 0.42 (0.25–0.50) | 0.64 (0.33–0.78) |
| v3 baseline | 0.68 (0.42–0.83) | 0.68 (0.58–0.75) | 0.57 (0.33–0.92) | 0.48 (0.17–0.75) | 0.54 (0.50–0.67) | 0.60 (0.50–0.67) |
| v3 real width | 0.75 (0.59–0.92) | 0.75 (0.50–1.00) | 0.58 (0.33–0.83) | 0.48 (0.33–0.75) | 0.52 (0.25–0.67) | 0.65 (0.50–0.75) |
| v3 leave-out | 0.83 (0.64–1.00) | 0.86 (0.75–0.92) | 0.65 (0.42–0.92) | 0.62 (0.39–0.83) | 0.50 (0.26–0.67) | 0.64 (0.33–0.92) |
| **All 28** | **0.733** | **0.697** | **0.593** | **0.529** | **0.462** | **0.593** |

**Real width minus baseline, per variant (C / c̄):**

| Variant | 10.0M | 18.2M | 33.1M | 60.3M | 109.9M | 199.9M |
|---|---|---|---|---|---|---|
| v1 | −1.6 / −0.142 | −1.0 / −0.090 | −0.2 / −0.020 | +1.2 / +0.100 | +1.2 / +0.103 | +2.6 / +0.223 |
| v3 | +0.8 / +0.068 | +0.8 / +0.067 | +0.2 / +0.010 | +0.0 / −0.002 | −0.2 / −0.020 | +0.6 / +0.050 |

The finals reproduce H4R's S1 arm means: 5.0, 7.6, 7.2 and 7.8 (h4r.md, Section 6.1).

### 4.2 Late and mid changes (28 networks; 95% percentile bootstrap over networks, 20,000 draws, seed 0)

| Change | Mean [95% CI] | Median | Rise / level / fall |
|---|---|---|---|
| d_late, C (199.9M − 109.9M) | **+1.57 [+0.96, +2.18]** | +1.5 | 19 / 6 / 3 |
| d_mid, C (199.9M − 60.3M) | +0.75 [−0.25, +1.75] | +1.0 | 17 / 2 / 9 |
| d_late, c̄ | +0.132 [+0.080, +0.185] | +0.125 | 20 / 4 / 4 |
| d_mid, c̄ | +0.064 [−0.018, +0.148] | +0.083 | 17 / 2 / 9 |

- **Late label: rising.** The CI lies above 0. It does not lie inside [−0.5, +0.5], so only one label holds and amendment 1's precedence rule did not apply.
- **The late rise follows a fall.** From 10.0M to 109.9M, mean C falls by 3.25 circuits. The rise to the final recovers 1.57 of them, leaving the final 1.68 circuits below 10.0M.
- **The plan's caveat applies in reverse.** The plan fixed that "a curve that still rises late, despite the anneal, is direct evidence for the budget lever". The learning rate and the entropy coefficient reach zero at 200M, and the rise falls in the last half of that anneal. Whether it comes from the extra steps or from the anneal cannot be told apart here (Section 8).

### 4.3 Best checkpoint (bears on #10)

- 22 of 28 networks (79%) have their best C at a checkpoint before the final; a tie counts as the final. Best minus final is +2.6 circuits on average (0 to +7).
- The best is at 10.0M for 13 networks, at 18.2M for 5, at 33.1M for 4 and at the final for 6.

| Network | 10.0M | 18.2M | 33.1M | 60.3M | 109.9M | 199.9M | Best − final |
|---|---|---|---|---|---|---|---|
| v1 base s10–s14 | 9, 9, 8, 8, 10 | 7, 9, 3, 8, 8 | 9, 6, 4, 9, 6 | 5, 7, 5, 4, 6 | 4, 3, 4, 6, 2 | 6, 4, 7, 5, 3 | +3, +5, +1, +4, +7 |
| v1 realw s10–s14 | 8, 6, 7, 8, 7 | 7, 3, 7, 6, 7 | 9, 2, 8, 5, 9 | 7, 7, 9, 5, 5 | 5, 3, 6, 5, 6 | 4, 8, 9, 9, 8 | +5, 0, 0, 0, +1 |
| v3 base s10–s14 | 5, 9, 8, 10, 9 | 8, 9, 7, 9, 8 | 11, 4, 7, 4, 8 | 9, 8, 2, 5, 5 | 6, 6, 6, 6, 8 | 6, 8, 8, 6, 8 | +5, +1, 0, +4, +1 |
| v3 realw s10–s14 | 9, 9, 9, 7, 11 | 11, 6, 12, 7, 9 | 8, 4, 6, 7, 10 | 6, 9, 5, 5, 4 | 8, 3, 7, 5, 8 | 8, 8, 6, 8, 9 | +3, +1, +6, 0, +2 |
| loo f1 s0–s3 | 9, 11, 9, 12 | 9, 10, 11, 11 | 7, 8, 11, 9 | 6, 9, 6, 9 | 6, 8, 6, 4 | 7, 8, 8, 7 | +2, +3, +3, +5 |
| loo f2 s0–s3 | 8, 11, 9, 11 | 10, 10, 11, 11 | 8, 5, 5, 9 | 5, 7, 8, 10 | 6, 8, 3, 7 | 9, 11, 4, 7 | +1, 0, +7, +4 |

### 4.4 What the early completions are (descriptive; added after the results were seen, not in the plan)

Source: `descriptive.json`, written by the `report` stage from the evaluation records.

| | 10.0M | 18.2M | 33.1M | 60.3M | 109.9M | 199.9M |
|---|---|---|---|---|---|---|
| Median lap gap of completed (network, circuit) pairs (n) | 54.2% (246) | 29.8% (234) | 17.3% (198) | 11.5% (178) | 8.6% (155) | 7.8% (199) |
| Networks with C ≥ 10 | 7 | 9 | 3 | 1 | 0 | 1 |

- **Early completions are slow laps.** 157 (network, circuit) pairs are completed at both 10.0M and the final. Their median gap is 53.6% at 10.0M and 7.8% at the final, and every one of the 157 is faster at the final.
- **Early checkpoints reach the gate's count.** 13 networks reach C ≥ 10 at some checkpoint: 7 leave-out runs, 5 v3 H4R runs (2 baseline, 3 real width) and 1 v1 baseline.
  - Of the 21 network-checkpoints with C ≥ 10, 20 are at 60.3M or earlier and 16 at 18.2M or earlier. There, the median lap gap is 11–54%.
  - The one final is `dr-realw-loo-f2-s1` (C 11).

**Networks (of 28) completing each circuit:**

| Circuit | 10.0M | 18.2M | 33.1M | 60.3M | 109.9M | 199.9M | Lost by the final | Gained by the final |
|---|---|---|---|---|---|---|---|---|
| Brands Hatch | 27 | 16 | 9 | 18 | 14 | 21 | 7 | 1 |
| Hockenheim | 28 | 25 | 20 | 13 | 11 | 17 | 11 | 0 |
| Moscow Raceway | 18 | 14 | 12 | 15 | 12 | 19 | 6 | 7 |
| Norisring | 4 | 9 | 6 | 6 | 2 | 3 | 3 | 2 |
| Nürburgring | 15 | 25 | 24 | 25 | 27 | 27 | 1 | 13 |
| Oschersleben | 19 | 3 | 1 | 5 | 8 | 16 | 10 | 7 |
| Sepang | 28 | 28 | 28 | 28 | 27 | 28 | 0 | 0 |
| Sochi | 26 | 28 | 22 | 11 | 6 | 6 | 20 | 0 |
| Yas Marina | 8 | 14 | 10 | 0 | 2 | 2 | 8 | 2 |
| Melbourne | 26 | 26 | 18 | 13 | 3 | 11 | 16 | 1 |
| Catalunya | 19 | 19 | 25 | 17 | 22 | 26 | 2 | 9 |
| Zandvoort | 28 | 27 | 23 | 27 | 21 | 23 | 5 | 0 |

"Lost" counts networks that complete the circuit at 10.0M and not at the final; "gained" counts the reverse. Sochi (20 lost) and Melbourne (16) account for most of the fall. Norisring is rarely completed at any checkpoint (at most 9 of 28). Yas Marina peaks at 14 of 28 (18.2M) and is almost never completed from 60.3M on.

### 4.5 Platform check (Mac against the stored A6000 end-of-training evaluations)

- **Calls and C.** 336 of 336 completion calls agree, and C is equal for all 28 networks, so no disagreement exists to fall near half completion.
- **Fractions.** 334 of 336 completion fractions are identical. The two that differ keep their call: `h4r-v1-realw-s11` at Norisring is 0.1 stored against 0.0 here, and `dr-realw-loo-f2-s0` at Yas Marina 1.0 against 0.7.
- **Lap times.** None of the 202 median lap times that both platforms have is bit-equal. The largest difference is 0.40 s. The two platforms do not compute identically, but every reading here is the same on either.

## 5. D2: A_DR completion of the finals (`adr.json`)

| Arm | Deterministic (median; range) | Stochastic windows (mean; range) | R-dev c̄ (median) |
|---|---|---|---|
| v1 baseline | 0.969 (0.965–0.983) | 0.961 (0.958–0.963) | 0.417 |
| v1 real width | 0.975 (0.973–0.986) | 0.968 (0.955–0.980) | 0.667 |
| v3 baseline | 0.972 (0.968–0.978) | 0.961 (0.956–0.968) | 0.667 |
| v3 real width | 0.985 (0.975–0.986) | 0.973 (0.967–0.982) | 0.667 |
| v3 leave-out | 0.976 (0.961–0.989) | 0.970 (0.955–0.982) | 0.637 |
| **All 28 (median)** | **0.976** | **0.966** | **0.667** |

- **Deterministic completion.**
  - Every network completes between 961 and 989 of the 1,000 levels. Their Wilson 95% intervals lie within 0.947–0.994.
  - The failures are mostly corridor exits (9–31 per network), plus 0–8 slips and 2 stalls. No window ended early at a strip's end.
- **Stochastic window completion.** It is lower than deterministic for every network, by 0.002–0.020, and is "late" (≥ 90%) for all 28.
- **Against R-dev.** Every network's deterministic A_DR completion exceeds its R-dev c̄, by +0.06 (`dr-realw-loo-f2-s1`, c̄ 0.92) to +0.74 (`h4r-v1-base-s14`, c̄ 0.24).
- **Arms.** Real-width arms are slightly higher on A_DR than their baselines (arm medians +0.006 in v1 and +0.013 in v3), although A_DR uses the default width model.
- **Section 8 row.** The median over the 28 is 0.976 on A_DR and 0.667 for R-dev c̄: "median A_DR ≥ 0.90 while R-dev c̄ ≤ 0.7".
- **Comparison with the pilot audit.** The plan's 0.8–0.9 for the stochastic measure came from the pilot audit's single seed at 60–110M, measured on 150 levels. These are 200M finals on all 1,000 levels.

## 6. D3: the crash comparison on H4R's 20 finals (`crashes.json`)

### 6.1 Crash census (all 12 circuits)

| Variant / arm | Terminations (of 600 runs) | Inside-edge corridor exits at a corner | At a former crash corner | Other |
|---|---|---|---|---|
| v1 baseline | 351 | 341 (97.2%) | 340 | 10 corridor exits on a Hockenheim straight (one network, n ≈ +3.1 m) |
| v1 real width | 217 | 207 (95.4%) | 147 | 10 slips at Norisring (one network, at about 0.1 m/s) |
| v3 baseline | 240 | 220 (91.7%) | 220 | 20 outside-edge exits at Yas Marina corner 0 (two networks) |
| v3 real width | 210 | 210 (100%) | 160 | none |

- **Lap and spread.** All but 9 terminations happen on lap 1. The policy is deterministic and the 10 starts differ only slightly, so a network's runs nearly always crash together: termination counts move in blocks of 10. The independent units are networks, 5 per arm.
- **Former crash corners.** 22 corners in v1 and 15 in v3 have at least 3 of the baseline arm's 50 runs terminating in their window.

### 6.2 Q1: are the extra completions avoided inside-edge crashes?

**Gain circuits.** Circuits that more real-width than baseline networks complete. The Mac re-evaluation and the stored table agree exactly (h4r.md, Section 6.1):
- **v1 (9):** Catalunya, Hockenheim, Melbourne, Moscow Raceway, Norisring, Nürburgring, Sochi, Yas Marina and Zandvoort.
- **v3 (3):** Hockenheim, Melbourne and Zandvoort.

| Variant | Baseline terminations on gain circuits | Real-width | ΔT | Δ of inside-edge corridor exits at former crash corners | s_in |
|---|---|---|---|---|---|
| v1 | 301 | 167 | 134 | 163 (290 → 127) | **1.22** |
| v3 | 80 | 40 | 40 | 40 (80 → 40) | **1.00** |

- **v1: s_in > 1.** Real-width networks avoid more inside-edge crashes at the baselines' crash corners than they avoid terminations overall. On the gain circuits they add 30 inside-edge exits at corners that are not former crash corners, and 10 slips at Norisring. Per circuit, s_in is 1.0 on 5 of the 9 circuits. It is 0.23 on Hockenheim, 0.91 on Melbourne, 2.0 on Yas Marina and 4.0 on Norisring.
- **v3** rests on 3 circuits and 4 blocks of 10 runs, so Q1 is weak there, as the plan expected.

**Termination rates at former crash corners** (terminations per passage through the window, baseline → real width):
- **v1.** The rate falls at 19 of 22 corners, to 0 at 14 of them. It is unchanged at Brands Hatch corner 4 (20/80 in both arms). It rises at 2: Moscow Raceway corner 12 (10/70 → 20/80) and Yas Marina corner 1 (10/40 → 20/60).
- **v3.** The rate falls at 12 of 15 corners, to 0 at 7. At Hockenheim corner 5 the fall is only marginal (10/88 → 10/90). It is unchanged at 3: Moscow Raceway corner 12 (10/90), Sochi corner 1 (40/60), and Norisring corner 0, where every passage terminates in both arms (40/40 → 50/50).

### 6.3 Q2: is the clearance larger by about the B shift?

Δc is the mean over the variant's former crash corners of the real-width arm mean minus the baseline arm mean of each network's median c_k over its passages (m). The 95% CI resamples networks within each arm (20,000 draws, seed 0) with the corners fixed.

| Measure | v1 | v3 |
|---|---|---|
| **Δc, primary (all passages)** | **+0.27 [+0.03, +0.49]** (22 corners) | **+0.28 [+0.20, +0.36]** (15 corners) |
| H4R's ΔB, and the band [0.5, 2] × ΔB | +0.44; 0.22–0.88 | +0.27; 0.135–0.54 |
| In the band | yes (point; the CI extends below it) | yes (the whole CI) |
| CI includes 0 | no | no |
| Draws with a corner left out (amendment 1) | 7,871 of 20,000 | 11,600 of 20,000 |
| Secondary: clearing passages only (selection-biased) | +0.11 [−0.08, +0.29] | +0.16 [+0.10, +0.28] |
| Secondary: all 265 corners | +0.215 [+0.12, +0.31] (just below the band) | +0.07 [+0.01, +0.20] |
| Secondary measures: draws with a corner left out | 20,000 of 20,000 (both measures) | 20,000 of 20,000 (both measures) |

- **What the primary Δc mixes.** It counts every passage, and a passage that ends in an inside-edge exit has c ≤ 0. At former crash corners, much of the arm difference is therefore the difference in how often the corner is survived (Section 6.2), not clearance on a clean pass.
  - On clearing passages only, the v1 shift is +0.11 m. That is below v1's band, with a CI that includes 0.
  - In v3 the clearing-only shift (+0.16 m) stays inside its band.
  - The clearing-only measure is selection-biased toward zero difference: baselines that clear a corner are the ones that keep margin there, and networks that never clear it drop out. It cannot settle the question either way.
- **Many dropped draws.** Several corners are passed by only one or two networks of an arm, because the rest crash earlier on lap 1: for example Yas Marina corner 12 in the v1 baselines and corner 9 in the v3 baselines. Draws that miss those networks drop the corner. That is why so many draws leave a corner out. In both secondary measures, in both variants, every draw (20,000 of 20,000) leaves out at least one corner.
- **Across all corners** (the general margin on real roads), the shift is +0.215 m in v1, just below its band (0.215 against 0.22). In v3 it is +0.07 m, below its band.
- **Per circuit** (secondary; `crashes.json`, `Q2_per_circuit`). In v1 the shift is positive on 10 of 11 circuits with former crash corners; the exception is Oschersleben (−0.14, CI −1.33 to +0.60). Its CI excludes 0 on Nürburgring, Sochi, Yas Marina and Catalunya. In v3 it is positive on all 8, with the CI excluding 0 on Norisring, Yas Marina, Melbourne and Zandvoort.

## 7. Expected directions (plan Section 7) against the results

| | Expected (before computing) | Result | Borne out? |
|---|---|---|---|
| E1 | C rises with log steps from 10.0M, most of the rise before 60.3M | C falls from 8.79 (10.0M) to 5.54 (109.9M), then rises to 7.11 | **No** |
| E1 | Late curve "flat" or "unresolved"; median network moves ≤ 1 circuit | "Rising", +1.57 [+0.96, +2.18]; median d_late +1.5 | **No** |
| E1 | v1 real-width advantage visible by 60.3M | +1.2 circuits at 60.3M (and +1.2, +2.6 later); negative at 10.0M, 18.2M and 33.1M | Yes |
| E1 | ≥ 1/3 of networks best before the final | 22 of 28 (79%) | Yes |
| E2 | ≥ 90% of 336 calls agree; C within ±1 for most; disagreements near half completion | 336 of 336; C equal for 28 of 28; no disagreements | Yes |
| E3 | Deterministic A_DR ≥ 0.90 for the median network of every arm, well above its R-dev c̄ | Arm medians 0.969–0.985 against c̄ medians 0.42–0.67; every network above its c̄ | Yes |
| E3 | Stochastic window completion lower, about 0.8–0.9 | Lower for all 28, but 0.955–0.982 | **In part** (lower, not 0.8–0.9) |
| E4 | ≥ 80% of baseline terminations are inside-edge corridor exits at corners | 97.2% (v1), 91.7% (v3), 94.9% pooled | Yes |
| E4 | v1 s_in ≥ 2/3; v3 weak (few gain circuits) | v1 1.22; v3 1.00 on 3 gain circuits | Yes |
| E5 | Δc_v1 > 0 and inside its band | +0.27 [+0.03, +0.49], band 0.22–0.88 | Yes (the CI reaches below the band) |
| E5 | Δc_v3 unresolved (CI includes 0) | +0.28 [+0.20, +0.36] | **No**: resolved, and inside its band |

E5 was held weakly because H4R found the B shift equal in v1 and v3 while the completion gain was four times larger in v1 (h4r.md, Section 8). Here the clearance shift at former crash corners is also about equal in the two variants (+0.27 and +0.28 m), while v1 has 9 gain circuits and v3 has 3.

## 8. Readings for the two levers (plan Section 8)

| Diagnostic | Outcome | Section 8 row | Where it points |
|---|---|---|---|
| D1 | Late CI above 0 | "rising" | **Budget lever**: the GW-6 1B pilot's condition ("curves still rising") is met |
| D2 | Median A_DR 0.976, median R-dev c̄ 0.667 | "median A_DR ≥ 0.90 while R-dev c̄ ≤ 0.7" | The R-dev shortfall is transfer. More steps on the same generator are unlikely to close the R-dev half alone; GW-2's width model and the margin lever are the candidates |
| D3 | v1: s_in 1.22, Δc +0.27 m in the band; v3: s_in 1.00, Δc +0.28 m in the band | "margin-consistent" in both variants | **Margin lever**: GW-5's training-margin arm (`edge_margin` 1.3–1.5 m in training only) |

**By the plan's rule.** D1 and D3 each point to their own lever, so both bounded slots of #11 can be used: one GW-6 pilot and one GW-5 round. D2 points away from relying on the budget lever alone.

**What qualifies these readings** (post hoc, after the results were seen; none changes a planned label):
- **The D1 rise is a partial recovery.** Mean C falls by 3.25 circuits from 10.0M to 109.9M while the lap gap falls from 54% to 8.6%, then rises by 1.57 as the anneal ends. The plan read a late rise "despite the anneal" as direct evidence for more steps. Here the rise coincides with the anneal's last half, and the 200M curve never returns to its 10M level.
  - A 1B run anneals over 1B steps. Whether it would end higher than a 200M run is not shown by these curves; that is what GW-6 would measure.
  - The budget reading is therefore weaker than the "rising" label suggests, and D2 says the networks already master their own distribution.
- **The D3 margin reading is firmer in v3 than in v1.**
  - In v1 the primary Δc is just inside the band, with a CI from +0.03 to +0.49 m.
  - On clearing passages only it is +0.11 m [−0.08, +0.29], which on its own would be the row "s_in ≥ 2/3 but Δc below the band". That row reads: crashes avoided without much more clearance, the line placed or timed differently, and it points to GW-5's local-perturbation arm.
  - The clearing-only measure is selection-biased, so the planned primary stands as the reading. But the margin lever is held weakly in v1, the variant where the completion gain is.
- **The R-dev completion count rewards slow driving early in training** (Section 4.4). For #10: an R-dev-selected checkpoint chosen on completion alone would mostly pick 10–33M checkpoints, 17–54% off the optimal lap. Any such rule needs a lap-time condition beside completion.

## 9. What this does not show

- **What causes the U shape.** No run was trained. The early fall coincides with large speed gains, and the late rise with the end of the anneal. Neither is tested.
- **Whether a 1B run ends higher.** That is GW-6's question. A 1B run is not a continued 200M run.
- **That the margin causes the completions.** D3 shows that the extra completions are inside-edge crashes avoided at the baselines' crash corners, with more clearance there. It does not show that a training margin would produce them. Both arms come from one commit and one GPU type, with 5 networks per arm. Runs within a network are nearly identical, so run-level counts overstate the information.
- **Anything about E0, `generator-v1` or R_TEST.** The gate is read at E0 (#10). R_TEST stays sealed and was not touched.

## 10. Procedure log

| Date (25 Oct 2026, local) | What | Effect |
|---|---|---|
| Before 02:28 | Script and tests written. Amendment 1 committed with them (`245a2c4`), before any network was evaluated. The only rollouts before it were unit tests with planted traces and one fixed hand-written policy | As planned |
| 02:28–02:31 | **D0**: the first evaluation of any network (`h4r-v1-base-s10` at 10.0M), alone and then 4 concurrent | W = 0.80 h: Mac |
| About 02:31 | The checkpoint tests ran after D0, as the plan's ordering requires: equality with `timed_laps` and `scripts/eval_rdev.py` on the D0 checkpoint, and the binding of the 28 finals | All passed |
| 02:34–03:21 | **D1**: 167 evaluations in a 4-process pool. All 168 records carry the sha256 of the committed script (`245a2c4`). HEAD moved during the run (`5226e00`, `5befa05`, `9249e46`), because the parallel agents committed their own files; none touched `src/` or this script | None |
| About 02:36 (during D1) | Planted D3 tests were added to `tests/test_gate_diagnostics.py` (86 lines: `_planted_d3`, `test_network_crash_data_from_planted_files`, `test_crashes_analysis_on_planted_data`), after `245a2c4`. No analysis code changed: every D1 and D3 record carries the committed script's sha256. The provenance blocks written from 02:36:50 on, `curves.json` and `crashes.json` among them, list the test file as modified | None |
| 03:22 | **D3**, from D1's final traces | As planned |
| 03:22–03:42 | **D2**: 28 networks in a 4-process pool | As planned |
| About 03:23 | **Added after D1's results were seen:** the `report` stage's descriptive block (`descriptive_d1`, Section 4.4). The edit touched only the report stage, while D2 was running; D2's code path is unchanged. The D2 records carry the sha256 of the edited file, because provenance reads the file when it writes | Labelled descriptive. No planned measure or label changed |
| 03:42:51 | After the last computation (D2 ended 03:42:31), two lane merges by other agents changed `src/` (`0f1773d` drivers, `34f4161` demo). Of the modules the script imports, only `viz/export.py` changed, and only in its docstring | None: every result was computed with `src/` as at `245a2c4` |
| This note | Post hoc readings (Section 8, "What qualifies these readings") written after all results were seen | Labelled post hoc |

**Checks with nothing to report.**
- All 28 finals matched their bound sha256, and the 20 stored H4R evaluations matched the manifest.
- `src/` was clean in every provenance block.
- No time-out runs: every non-completing run terminated.
- No R_TEST name was evaluated.
