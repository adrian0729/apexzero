# X-1: on H4R's same-code networks, real-width training lowers the slow, far-upstream part of edge tracking. No change was detected for short bumps at the apex, and short ramps are followed slightly more (exploratory)

Results of 30 Oct 2026. Plan: [locality-h4r-plan.md](locality-h4r-plan.md), committed and pushed in `3e71926` before anything here was computed. Code: `scripts/locality_h4r_exploratory.py`, which imports `scripts/locality_exploratory.py`, `scripts/h4r.py` and step 2's frozen `src/apexzero/fields` read-only. Results: `outputs/locality-h4r/results.json` (committed) and `results.csv` (local). Background: [locality.md](locality.md) and [h4r.md](h4r.md).

**Every number in this note is exploratory** and comes from `results.json`, except where another source is named. Only the primary outcome had a decision rule fixed in the plan. No network, closed loop, gradient or solver was run; only synthetic A2 keys were used, and R_TEST is not involved. The run took 68 s on the Mac.

## 1. Summary

- **The primary outcome: no change detected at short scales** (the plan's reading). Real-width minus baseline, in L/e (the line's move per metre the edge moves at s\*), mean over the two variants:

  | Bump at s\* | Difference | v1 | v3 | Permutation p (two-sided) | Paired p | Pairs with real-width lower |
  |---|---|---|---|---|---|---|
  | σ_b = 4 m | +0.020 | +0.033 | +0.007 | 0.065 | 0.12 | 3 of 10 |
  | σ_b = 8 m | +0.026 | +0.045 | +0.008 | 0.12 | 0.18 | 3 of 10 |

  Holm at 0.05 rejects neither. If anything the sign is up, the opposite of the corner-wide effect. On the same networks, the trapezoid's (step 2's corner-wide G) difference is −0.059: H4R-G, reproduced exactly (Section 3).
- **X1-a (no detectable difference at σ_b = 4 and 8 m): borne out.** The differences are almost exactly those on step 2's confounded arms (+0.019 and +0.026 in G_loc there; +0.020 and +0.026 here). So the code and hardware confound did not create that result.
- **X1-b (real-width lower at long scales and for changes far upstream): partly borne out, in 5 of the 7 profiles it names.** All seven differences are negative.
  - **Detected at long scales:** σ_b = 64 m −0.059 (p 0.0010) and L = 80 m −0.047 (p 0.0046). The trapezoid, not one of the seven, gives −0.059 (p 0.0015).
  - **Detected for bumps before s\*:** −0.045 at 20 m (p 1.9e-4), −0.038 at 40 m and −0.005 at 60 m (both p 3.1e-5, the smallest possible). Real-width is lower in all 10 same-seed pairs at 40 and 60 m. The 60 m difference is tiny because nearly nothing responds that far upstream.
  - **Not detected** at σ_b = 32 m (−0.027, p 0.087) or X = 80 m (−0.023, p 0.099). L = 40 m, outside the seven, gives −0.019 (p 0.33).
- **X1-c (gain falls as the change becomes more local): borne out in all 20 networks,** monotone in σ_b, X and L. The response kernel peaks at d = −10 m in all 20: the line trails the edge by about 10 m, as in `locality.md`.
- **A descriptive surprise: real-width networks follow short ramps slightly more.** These are ramps ending at s\*, the plan's model of a real-road intrusion. Real-width is higher by +0.028 at X = 5 m, +0.039 at 10 m and +0.037 at 20 m (L/e; p 0.033, 0.0037, 0.0089), in both variants. This is not the primary outcome, and it is not corrected for the many tests here (Section 5).
- **For the follow-through's open question:** width realism does not improve short-scale following, beyond a nudge of about 0.02–0.04 on a scale where the networks sit at 0.1–0.4 and the optimum at 1. By the follow-through's own rule (Section 9.2), local perturbations (GW-5) stay the lever to test.

## 2. What was computed

- **Profiles and forms:** `locality.md`'s 24 profiles in its three forms, from its own functions:
  - bumps at s\*, σ_b = 2–64 m;
  - ramps ending at s\*, X = 5–80 m;
  - plateaus, L = 10–80 m, plus step 2's corner-wide trapezoid;
  - σ_b = 8 m bumps at s\* + d, d = −80 to +20 m.

  They were run on H4R's stored impulse responses for all 20 networks: v1 and v3, real-width and baseline, seeds 10–14.
- **Keys: H4R's own** (`h4r.h4r_key_measures`), 358–426 kept of 427 per network. H4R's gate2-trained rule excluded one key in each of five networks (three real-width, two baseline), as in `h4r.md`.
- **Each network's statistic** is its pooled median over geometries (20,000 bootstrap draws on step 2's stream).
- **The tests: H4R's own.** The stratified permutation test on the per-network medians (v1 and v3, 5 against 5: 63,504 relabellings), and the paired sign-flip test over the 10 same-seed pairs (1,024 flips). Both are two-sided, as twice the smaller one-sided p.
- **"Difference"** is the mean over the two variants of (mean real-width − mean baseline), which is half the permutation statistic T.

## 3. Checks (all passed; the script stops and writes nothing on any failure)

| Check | Result |
|---|---|
| Impulse-array files against their rows' sha256 | 9,620 of 9,620 match |
| Trapezoid G_loc against each kept key's G (required: 1e-9) | max \|difference\| 0 |
| Each network's pooled trapezoid summary (median, lo, hi) and statistic against H4R's `decisions.json` | max \|difference\| 4.4e-16, all 20 networks |
| The trapezoid's permutation test against H4R-G | T (−0.1181) and both one-sided p (7.56e-4, 0.99926) identical |
| s\* recovery (`locality.md`'s checks) | basis sums within 6.7e-16 of 1; e_star within 7.8e-16; no kept s\* off the trapezoid's plateau |
| Frozen code and `scripts/locality_exploratory.py` | unchanged from HEAD |

## 4. Real-width against baseline, every profile

The real and baseline columns are medians over the 10 networks of each arm, pooling the variants. The difference and the per-variant columns are means. The bumps of σ_b = 2–8 m and the ramps with X ≤ 20 m are given in L/e: these are the primary form, and the one `locality.md` reads where the forms differ. Every other profile is given in G_loc. At σ_b = 4–8 m the two forms agree to 0.001. Every form of every profile is in `results.json`.

| Profile | Form | Real-width | Baseline | Difference | v1 | v3 | Permutation p | Paired p | Real lower (of 10) |
|---|---|---|---|---|---|---|---|---|---|
| bump σ_b = 2 m | L/e | 0.094 | 0.082 | +0.011 | +0.018 | +0.005 | 0.054 | 0.10 | 3 |
| σ_b = 4 m | L/e | 0.176 | 0.161 | +0.020 | +0.033 | +0.007 | 0.065 | 0.12 | 3 |
| σ_b = 8 m | L/e | 0.321 | 0.309 | +0.026 | +0.045 | +0.008 | 0.12 | 0.18 | 3 |
| σ_b = 16 m | G_loc | 0.524 | 0.530 | +0.008 | +0.029 | −0.012 | 0.64 | 0.65 | 4 |
| σ_b = 32 m | G_loc | 0.646 | 0.684 | −0.027 | −0.020 | −0.034 | 0.087 | 0.051 | 8 |
| σ_b = 64 m | G_loc | 0.714 | 0.770 | −0.059 | −0.047 | −0.071 | 0.0010 | 0.0020 | 10 |
| ramp X = 5 m | L/e | 0.265 | 0.246 | +0.028 | +0.030 | +0.025 | 0.033 | 0.037 | 2 |
| X = 10 m | L/e | 0.329 | 0.294 | +0.039 | +0.044 | +0.034 | 0.0037 | 0.0059 | 1 |
| X = 20 m | L/e | 0.426 | 0.392 | +0.037 | +0.048 | +0.027 | 0.0089 | 0.018 | 2 |
| X = 40 m | G_loc | 0.563 | 0.555 | +0.014 | +0.028 | −0.001 | 0.38 | 0.35 | 4 |
| X = 80 m | G_loc | 0.654 | 0.679 | −0.023 | −0.016 | −0.031 | 0.099 | 0.039 | 8 |
| plateau L = 10 m | G_loc | 0.451 | 0.448 | +0.022 | +0.043 | +0.001 | 0.24 | 0.28 | 3 |
| L = 20 m | G_loc | 0.561 | 0.566 | +0.012 | +0.034 | −0.011 | 0.54 | 0.57 | 4 |
| L = 40 m | G_loc | 0.662 | 0.697 | −0.019 | −0.012 | −0.026 | 0.33 | 0.27 | 7 |
| L = 80 m | G_loc | 0.729 | 0.778 | −0.047 | −0.030 | −0.064 | 0.0046 | 0.0039 | 9 |
| trapezoid (H4R-G) | G_loc | 0.746 | 0.805 | −0.059 | −0.049 | −0.069 | 0.0015 | 0.0020 | 10 |
| bump at d = −80 m | G_loc | −0.000 | 0.000 | −0.000 | −0.001 | −0.000 | 0.074 | 0.10 | 7 |
| d = −60 m | G_loc | −0.000 | 0.002 | −0.005 | −0.006 | −0.003 | 3.1e-5 | 0.0020 | 10 |
| d = −40 m | G_loc | 0.019 | 0.057 | −0.038 | −0.038 | −0.037 | 3.1e-5 | 0.0020 | 10 |
| d = −20 m | G_loc | 0.255 | 0.314 | −0.045 | −0.050 | −0.040 | 1.9e-4 | 0.0039 | 9 |
| d = −10 m | G_loc | 0.384 | 0.383 | +0.008 | +0.021 | −0.006 | 0.62 | 0.64 | 3 |
| d = 0 (= σ_b 8 m) | G_loc | 0.321 | 0.309 | +0.026 | +0.045 | +0.008 | 0.12 | 0.18 | 3 |
| d = +10 m | G_loc | 0.179 | 0.164 | +0.015 | +0.021 | +0.009 | 0.070 | 0.063 | 3 |
| d = +20 m | G_loc | 0.063 | 0.055 | +0.007 | +0.003 | +0.011 | 0.11 | 0.15 | 3 |

- Across all 20 networks the short-scale values are 0.063–0.120 at σ_b = 2 m, 0.130–0.226 at 4 m and 0.257–0.397 at 8 m (L/e). Both ends of each range are a little above `locality.md`'s 29 networks (0.055–0.103, 0.111–0.198 and 0.227–0.359).
- **Medians and means can disagree within a variant.** For example, v3's real-width median at σ_b = 4 m (0.170) is below its baseline median (0.190), while v3's mean difference is +0.007. The tests use the means.

## 5. Against `locality.md`'s confounded arms (step 2's networks; its Section 4.3)

| Profile | Difference there | Difference here |
|---|---|---|
| bump σ_b = 2 / 4 / 8 m (G_loc) | +0.010 / +0.019 / +0.026 | +0.010 / +0.020 / +0.026 |
| ramp X = 5 / 10 / 20 m (G_loc) | +0.012 / +0.014 / +0.015 (p 0.36–0.38) | +0.026 / +0.036 / +0.037 (p 0.0065–0.034) |
| σ_b = 64 m | −0.066 | −0.059 |
| trapezoid | −0.079 | −0.059 |
| bump at d = −40 m | −0.048 | −0.038 |

- **The pattern replicates on same-code networks.** Real-width networks are lower for slow and far-upstream changes, and not lower for short ones.
- **The two sets differ in size and detail:**
  - **The long-scale effects are weaker here.** σ_b = 32 m is −0.047 (p 0.0012) there against −0.027 (p 0.087) here, X = 80 m −0.038 (p 0.0020) against −0.023 (p 0.099), and the trapezoid −0.079 against −0.059.
  - **The ramps' positive difference is about twice as large here.** Its v3 part changed sign: −0.004 and −0.003 at X = 5 and 10 m there, +0.023 and +0.030 here (G_loc).
- **Multiplicity.**
  - There are 64 permutation tests here (24 profiles in G_loc and L, 16 in L/e) and 64 paired tests. Only the two primary tests were corrected. Nine of the 24 G_loc tests have p below 0.05. All nine are long-scale or upstream profiles (real-width lower) or the X = 5–20 m ramps (real-width higher).

## 6. What it means (the lead's reading, not tested)

- **H4R's lower corner-wide G comes from the slow, far-upstream part of the response.** Real-width-trained lines answer a widening 20–60 m before the closest point less, and an edge change confined to a few metres at the apex the same or slightly more. Real-width training changes how the line responds to the corner's approach, not how precisely it follows the edge near the apex.
- **Together with the line / execution split** ([line-execution.md](line-execution.md)):
  - On R_TEST, only about 30% of E0's lost time comes back on its own line (R(0.05) = 0.295), and half comes back within 0.37 m of it. So small line moves are where much of the time is.
  - The networks follow local edge changes with a gain of only 0.1–0.3, and training on real widths alone does not raise that gain.
  - The split measures distance from the car's own line, not edge tracking, so this link is an inference, not a test. So, as the follow-through framed it, local perturbations (GW-5) are the lever to test if precise placement is the goal. E0 itself was trained with copied real widths; this result is from H4R's networks, not E0's.
- **Limits:**
  - The analysis is first order, at the base argmin, on synthetic A2 corners (mostly 12 m wide) from the shared start states. It covers neither real crash corners nor the agents' own states (`locality.md`, Section 7).
  - There are five seeds per arm per variant. The short-scale differences (+0.02) are about a third of the size of the corner-wide one (−0.059). Differences of that size are near what five seeds per arm can detect. A small short-scale effect in either direction is therefore not ruled out. No interval or power figure was computed.
  - The ramp result is descriptive and uncorrected.
