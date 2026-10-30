# GW-5, GW-6 and GW-7: a 1.4 m training-only margin lifts R-dev completion from 8.0 to 10.6 of 12 at 200M; local width bumps do not help, and SimBa follows edges far more but keeps less room and tends to complete less (exploratory)

Results of 30 Oct 2026. Plan: [gw5-plan.md](gw5-plan.md), committed with its code and tagged `gw5-v1` (`c5963fc`) before any run.
- **Code:**
  - `scripts/gw5_train.py` (the levers, wrapping `train_dr.py` unchanged) and `scripts/gw5_runs.py`, both tagged `gw5-v1`;
  - `scripts/gw6_runs.py` (tag `gw6-v1`);
  - `scripts/gw5_analysis.py`, written after the runs (first committed in `7961d56`).
- **Results:** `outputs/gw5/completion.json` and `edges.json` (committed). GW-6's will be `gw6.json`. The plan named a single `analysis.json`.
- **Beyond the plan, added after the analysis was first run (on review):**
  - the real crash change's L/e and the permutation test on the gap;
  - the gap on the circuits both same-seed runs complete;
  - the clearance (base c_min) at the real crash corners.

  They are marked where they appear.
- **Everything here is exploratory.**
  - The plan fixed the arms, the primary outcome, the tests and GW-6's rule in advance.
  - The R-dev numbers are in-sample for widths, as E0's are.
  - R_TEST was not used.
  - `generator-v1`, `sim-v1`, E0 and the lockbox are unchanged.

## 1. Summary

- **GW-5's margin works.** Training with the car terminated 1.4 m from the edge, then evaluating at the usual 1.0 m, raises R-dev completion:
  - **10.6 circuits of 12 against 8.0 for the control** (+2.6; one-sided permutation p 0.016, that is 4 of 252 relabellings). Holm over the three arms rejects (level 0.0167).
  - All five margin seeds complete 9–12 circuits; the control's complete 6–9. The median is 11, above E0's gate of 10, at 200M steps.
  - It completes the hard circuits: Norisring in 5 of 5 seeds (control 1 of 5), Sochi and Yas Marina in 2 of 5 each (control 0).
  - Two-sided, p is 0.032.
  - **The cost is lap time.**
    - The mean of the runs' median gaps rises from 8.2% to 9.4% (+1.2 points, two-sided p 0.008).
    - On the circuits both same-seed runs complete, the margin arm is 1.1 points slower: 36 of 39 circuit pairs are slower (added).
  - **How it works:** it keeps more room. At the 12 real crash unit-phases, its base clearance is 1.79 m against 1.21 m (+0.47 m, two-sided p 0.032; added).
- **GW-5's local width bumps do not help.**
  - Completion is 7.8 against 8.0 (one-sided p 0.69).
  - The mean of median gaps is unchanged (+0.07 points). But on the circuits both same-seed runs complete, the bumps arm is 0.58 points slower (24 of 33 circuit pairs; added).
  - At the 12 real crash unit-phases of X-2, short-scale edge following is unchanged (L/e at 4 m +0.006, p 0.83; at 8 m +0.018, p 0.62).
- **GW-7's SimBa follows edges far more, keeps much less room, and tends to complete less.**
  - Completion is 6.6 against 8.0 (two-sided p 0.12, not significant).
  - It is faster on every circuit it shares with its same-seed control: 1.56 points, 24 of 24 circuit pairs (added).
  - Its base clearance at the real crash corners is 0.64 m against 1.21 m (two-sided p 0.016; added).
  - It follows edges much more at every scale, in all five seeds against all five controls (p 0.008, the smallest possible):
    - L/e for a 4 m bump: 0.42 against 0.21;
    - for an 8 m bump: 0.64 against 0.36;
    - the corner-wide trapezoid: 0.85 against 0.72;
    - the real crash changes: 0.48 against 0.19.
  - It trains at about 73k steps per second against about 130k. These are cumulative averages that vary with what shared the GPU (116–150k within one MLP arm), but the arms were staggered evenly over the lanes.
- **GW-6** (Section 5) is the margin recipe at 1B steps, chosen by the plan's rule, against E0's five 1B runs.

## 2. Completion and lap time (primary outcome: R-dev circuits completed, of 12)

| Arm | Seeds 70–74 | Mean | Median | vs `ctrl` | One-sided p | Holm | Mean of the runs' median gaps | Steps/s per run |
|---|---|---|---|---|---|---|---|---|
| `ctrl` (E0's recipe at 200M) | 6, 9, 9, 7, 9 | 8.0 | 9 | – | – | – | 8.17% | 129.8k |
| `margin` (1.4 m in training) | 9, 12, 11, 10, 11 | **10.6** | 11 | **+2.6** | **0.016** | **rejects** | 9.39% | 127.2k |
| `bumps` (local width bumps) | 8, 6, 9, 8, 8 | 7.8 | 8 | −0.2 | 0.69 | – | 8.24% | 128.9k |
| `simba` (GW-7) | 5, 7, 7, 7, 7 | 6.6 | 7 | −1.4 | 0.96 (two-sided 0.12) | – | 6.79% | 73.0k |

**Seeds completing each circuit (of 5).** Norisring, Yas Marina and Sochi were the three hardest in earlier runs (pooled completion 6%, 6% and 39%, follow-through Section 7.4).

| Circuit | ctrl | margin | bumps | simba |
|---|---|---|---|---|
| Brands Hatch | 4 | 5 | 5 | 2 |
| Catalunya | 4 | 5 | 5 | 0 |
| Hockenheim | 3 | 5 | 3 | 1 |
| Melbourne | 5 | 5 | 4 | 2 |
| Moscow Raceway | 5 | 5 | 4 | 2 |
| Norisring | **1** | **5** | 2 | 3 |
| Nürburgring | 5 | 5 | 5 | 5 |
| Oschersleben | 5 | 4 | 2 | 5 |
| Sepang | 5 | 5 | 5 | 5 |
| Sochi | **0** | **2** | 0 | 3 |
| Yas Marina | **0** | **2** | 0 | 0 |
| Zandvoort | 3 | 5 | 4 | 5 |

- **The gap comparison is not like for like.** Each run's gap is its median over the circuits it completes, and the arms complete different circuits. Within the margin arm, the gap is 9.3–9.5% in every seed against 7.9–8.7% for the control. SimBa's lower gap comes on a different, smaller set of circuits.
- **All 20 runs finished.** They ran on one A6000 at 5 lanes in 2.54 h (`round_manifest.py check` OK at `gw5-v1`).

## 3. Edge following at the real crash corners (secondary, descriptive)

Measured with X-2's machinery ([x2x3.md](x2x3.md)) at the 12 real R-dev crash unit-phases, from step 2's shared start states. Each network's value is its median over the unit-phases. Each cell gives the arm's median over its five networks, then in brackets the **mean** difference from `ctrl` (the permutation statistic) and the two-sided p.

| Measure | ctrl | margin | bumps | simba |
|---|---|---|---|---|
| L/e, bump σ_b = 4 m | 0.210 | 0.215 (+0.017, p 0.48) | 0.228 (+0.006, p 0.83) | **0.419** (+0.193, p 0.008) |
| L/e, bump σ_b = 8 m | 0.362 | 0.402 (+0.029, p 0.46) | 0.397 (+0.018, p 0.62) | **0.640** (+0.255, p 0.008) |
| trapezoid G (corner-wide) | 0.722 | 0.588 (−0.061, p 0.36) | 0.699 (−0.023, p 0.60) | **0.850** (+0.149, p 0.008) |
| L/e, the real change (added) | 0.189 | 0.209 (+0.030, p 0.48) | 0.139 (−0.038, p 0.38) | **0.480** (+0.247, p 0.008) |
| base clearance c_min, m (added) | 1.212 | **1.791** (+0.466, p 0.032) | 1.473 (+0.152, p 0.44) | **0.644** (−0.591, p 0.016) |

- **SimBa follows edges far more at every scale.** Its range over the five networks does not overlap the control's on any of the four following measures. It also keeps far less room at these corners (0.46–0.83 m against 0.82–1.63 m).
- **The margin and the bumps change no following measure detectably** (at n = 5; the margin's trapezoid median is lower, p 0.36).
- **The margin networks keep more room** at the real crash corners: 1.35–1.82 m against 0.82–1.63 m. That fits the margin working through clearance rather than better following.
- **One control network missed four unit-phases.** `gw5-ctrl-s72` crashed in its base run at all four Catalunya phases (c_min invalid), so its value is over 8 unit-phases. Every other network has all 12.

## 4. What it means (the lead's reading, not tested)

- **The margin is the lever.** It is the only one of the three that moves completion, and by a lot: one seed completes all 12 circuits at 200M steps.
  - It works through clearance (+0.47 m at the real crash corners), not better edge following.
  - It costs about 1.1–1.2 points of lap gap, the same order as the width model GW-4 turned down for its +1.5-point gap. Whether that trade is worth it is a judgment for the next recipe.
- **Local bumps, as built here, are not.** They change neither completion nor short-scale following, and they cost about 0.6 points of lap time on shared circuits. The edge-following deficit X-2 found at the real corners is not cured by exposing the policy to random local width changes of ±1 m. This one design does not rule out other densities or amplitudes.
- **SimBa's high edge following does not translate into completion** (6.6 against 8.0, p 0.12), although it is 1.6 points faster on every shared circuit.
  - It keeps about half the control's clearance at the real crash corners (0.64 m against 1.21 m). It drives a tighter, faster line with less room for error, which fits completing less; whether that is the cause is not tested.
  - The interp tooling did load SimBa, which the plan had doubted.

## 5. GW-6: the margin recipe at 1B (pending)

Running on the same box since 08:39 UTC (5 runs, seeds 80–84). This section is filled in from `outputs/gw5/gw6.json` when the runs are back.
