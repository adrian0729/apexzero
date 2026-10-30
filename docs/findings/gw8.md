# GW-8: in a sweep of the training margin, completion, lap gap and clearance all rise with the margin (1.2, 1.4, 1.6 m: 9.6, 10.8, 11.4 circuits; 8.45%, 9.01%, 9.53%); the plan's rule keeps 1.4 m (exploratory)

Results of 30 Oct 2026. Plan: [gw8-plan.md](gw8-plan.md), committed and tagged `gw8-v1` (`c213107`) before any run.
- **Code:**
  - `scripts/gw8_runs.py` (tag `gw8-v1`);
  - `scripts/gw5_train.py`, unchanged since `gw5-v1`;
  - `scripts/gw8_analysis.py`, written while the round trained and committed (`822062d`) before any run came back. Before the runs, it was checked on GW-5's runs relabelled in a scratch copy: it reproduced GW-5's margin-against-control numbers.
- **Results:** `outputs/gw8/completion.json` and `edges.json` (committed).
- **One deviation from the plan: the box.**
  - `massedcompute_A6000_plus` was not available, so the round ran on `massedcompute_A6000`: the same GPU and price, with 6 vCPUs instead of 12.
  - All three margins shared that box, so the comparison between them is unaffected. Throughput is not comparable with GW-5's: about 111k steps per second per run, against about 128k.
  - The round took 1.72 h (`round_manifest.py check` OK at `gw8-v1`). The box was up about 3.5 h, about $2.40, because it sat idle after the round while the Mac was asleep, and the first pull-back broke off when the Mac's network dropped.
- **Everything here is exploratory.**
  - The plan fixed the margins, the outcomes, the tests and the recommendation rule in advance.
  - The R-dev numbers are in-sample for widths, as E0's are.
  - R_TEST was not used.
  - `generator-v1`, `sim-v1`, E0 and the lockbox are unchanged.

## 1. Summary

- **The plan's rule keeps 1.4 m.**
  - 1.6 m completes 0.6 circuits more than 1.4 m; the rule needed 1.0.
  - 1.2 m completes 1.2 fewer; the rule allowed 0.4 fewer. Its lap-time saving (0.40 points on shared circuits) would have passed the rule's 0.3.
- **The margin acts as a dial.** Over 1.2–1.6 m, completion, lap gap and clearance at the real crash corners all rise with it:

  | Margin | Circuits completed (mean of 12) | Mean of the runs' median gaps | Clearance at the crash corners (median) |
  |---|---|---|---|
  | 1.2 m | 9.6 | 8.45% | 1.36 m |
  | 1.4 m | 10.8 | 9.01% | 1.73 m |
  | 1.6 m | 11.4 | 9.53% | 1.99 m |

  - **Each 0.2 m costs about half a point of lap gap.** On the circuits both same-seed runs complete, 1.2 m is 0.40 points faster than 1.4 m (faster in 28 of 45 circuit pairs), and 1.6 m is 0.52 points slower (slower in 36 of 51).
  - **The completion gain shrinks:** +1.2 circuits from 1.2 to 1.4 m, then +0.6 from 1.4 to 1.6 m.
  - **No step is significant at five seeds.** Against 1.4 m, the two-sided p is 0.17 and 0.56 for completion, and 0.15 and 0.14 for the mean of median gaps.
- **1.4 m reproduces GW-5 on new seeds:** 10.8 circuits against GW-5's 10.6, and a gap of 9.01% against 9.39%.
- **1.6 m completes the most.** Its median is 12: three of five seeds complete every circuit. It completes Yas Marina in 5 of 5 seeds, against 2 of 5 at 1.2 and 1.4 m.
- **Edge following does not change with the margin.** At the 12 real crash unit-phases, L/e for the 4 m and 8 m bumps, the trapezoid G and the real change's L/e stay within 0.031 of 1.4 m's on average (every p ≥ 0.38). The margin works through clearance, as GW-5 found.

## 2. Completion and lap time (primary outcome: R-dev circuits completed, of 12)

| Margin | Seeds 90–94 | Mean | Median | vs 1.4 m | Two-sided p | Mean of the runs' median gaps | vs 1.4 m (two-sided p) | Same-seed gap on shared circuits vs 1.4 m | Steps/s per run |
|---|---|---|---|---|---|---|---|---|---|
| 1.2 m | 9, 11, 10, 9, 9 | 9.6 | 9 | −1.2 | 0.17 | 8.45% | −0.56 (0.15) | −0.40 points (17 of 45 pairs slower) | 110.6k |
| 1.4 m | 9, 11, 12, 11, 11 | 10.8 | 11 | – | – | 9.01% | – | – | 111.7k |
| 1.6 m | 12, 11, 12, 12, 10 | **11.4** | **12** | +0.6 | 0.56 | 9.53% | +0.52 (0.14) | +0.52 points (36 of 51 pairs slower) | 111.9k |

**Reference (descriptive only; seeds 70–74, another box and day):** GW-5's `ctrl`, no margin, completed 8.0 circuits at a mean of median gaps of 8.17%. Its 1.4 m `margin` arm completed 10.6 at 9.39%.

**Seeds completing each circuit (of 5).**

| Circuit | 1.2 m | 1.4 m | 1.6 m |
|---|---|---|---|
| Brands Hatch | 4 | 5 | 5 |
| Catalunya | 4 | 5 | 5 |
| Hockenheim | 5 | 5 | 5 |
| Melbourne | 5 | 5 | 5 |
| Moscow Raceway | 5 | 5 | 5 |
| Norisring | 2 | 3 | 4 |
| Nürburgring | 5 | 5 | 5 |
| Oschersleben | 3 | 5 | 4 |
| Sepang | 5 | 5 | 5 |
| Sochi | 3 | 4 | 4 |
| Yas Marina | 2 | 2 | **5** |
| Zandvoort | 5 | 5 | 5 |

- **The same-seed gap is like for like:** each margin against 1.4 m with the same seed, on the circuits both complete.
  - Per seed, 1.2 m is faster in four of five (−1.10 to −0.10 points; seed 93 +0.44).
  - 1.6 m is slower in all five (+0.09 to +1.14 points).

## 3. Clearance and edge following at the real crash corners (secondary, descriptive)

Measured as in GW-5 ([gw5.md](gw5.md) Section 3), at the 12 real R-dev crash unit-phases from step 2's shared start states. Each network's value is its median over the unit-phases; all 15 networks loaded, each with all 12. Each cell gives the margin's median over its five networks, then in brackets the **mean** difference from 1.4 m and the two-sided p.

| Measure | 1.2 m | 1.4 m | 1.6 m |
|---|---|---|---|
| base clearance c_min, m | 1.360 (−0.121, p 0.56) | 1.728 | 1.988 (+0.263, p 0.14) |
| L/e, bump σ_b = 4 m | 0.224 (+0.020, p 0.49) | 0.219 | 0.250 (+0.022, p 0.38) |
| L/e, bump σ_b = 8 m | 0.382 (+0.012, p 0.85) | 0.408 | 0.431 (+0.029, p 0.44) |
| trapezoid G (corner-wide) | 0.652 (−0.031, p 0.43) | 0.685 | 0.717 (−0.003, p 0.91) |
| L/e, the real change | 0.176 (−0.010, p 0.75) | 0.192 | 0.185 (+0.015, p 0.90) |

- **Clearance rises with the margin** (medians 1.36, 1.73 and 1.99 m), though the networks overlap: 1.15–2.02 m at 1.2 m, 1.24–1.83 m at 1.4 m and 1.56–2.12 m at 1.6 m. GW-5's no-margin control had a median of 1.21 m.
- **Following does not move.** No following measure differs from 1.4 m by more than 0.031, and none is close to significant.

## 4. The recommendation rule (fixed in the plan)

- **1.6 m:** +0.6 circuits over 1.4 m, below the required +1.0. Not recommended.
- **1.2 m:** −1.2 circuits, beyond the allowed −0.4. Its same-seed gap, −0.40 points, would have met the −0.3 bar. Not recommended.
- **So 1.4 m stays.**

## 5. What it means (the lead's reading, not tested)

- **The margin is a dial, not a switch.** Between 1.2 and 1.6 m, each 0.2 m adds about 0.1–0.4 m of clearance at the crash corners (means 1.51, 1.63 and 1.89 m; medians 1.36, 1.73 and 1.99 m) and about half a point of lap gap. The completion it buys shrinks from about 1.2 circuits to about 0.6. None of these steps is significant at five seeds; the pattern is that all three outcomes move in the same direction, in order.
- **1.4 m remains a sound choice** by the plan's rule. It sits in the middle of the trade, and GW-6 showed it holds at 1B (11.4 of 12 against E0's 7.2).
- **If completion matters more than lap time, 1.6 m is the next candidate.** At 200M it already has a median of 12 and completes Yas Marina in every seed. This sweep does not show that it beats 1.4 m, and it was not run at 1B.
- **Limits:**
  - five seeds per margin;
  - 200M steps only;
  - R-dev completion is in-sample for widths, with no held-out set;
  - the box differs from GW-5's and GW-6's.
