# GW-8, the training-margin sweep: plan (exploratory; written before any run, 30 Oct 2026)

**Status.**
- **Why it runs.** After GW-6, the user asked on 30 Oct 2026 for a sweep of the training-only margin. GW-5 tried one setting, 1.4 m, picked inside the follow-through's range of 1.3–1.5 m ([gw5.md](gw5.md)). At 200M it lifted R-dev completion from 8.0 to 10.6 of 12, for about 1.1 points of lap gap.
- **The question.** Does a smaller margin (1.2 m) keep most of that completion gain for less lap time? Does a larger one (1.6 m) complete more?
- **The order.** This plan and its run list are committed and tagged (`gw8-v1`) before any run. Before that, the only runs were two 41k-step smoke runs (1.2 and 1.6 m, no evaluation), as plumbing.
- **What it can change.** It is exploratory and changes no reported result: `generator-v1`, `sim-v1`, E0 and the lockbox are done. Only R-dev data and one A6000 are used. R_TEST is not involved.

## The round: 3 margins × 5 seeds at 200M

- **The recipe** is GW-5's margin arm: `scripts/gw5_train.py --steps 200e6 --real-widths --train-edge-margin M`, with M = 1.2, 1.4 and 1.6 m. Evaluation stays at 1.0 m.
  - `gw5_train.py` is used as tagged at `gw5-v1`, unchanged. It takes any positive margin.
  - The seeds are 90–94, unused so far.
  - The 1.4 m arm repeats GW-5's margin arm on new seeds, so the three margins are compared within one round, on one box and day.
- **Every margin leaves room to drive.** R-dev's narrowest side is 3.36 m (Brands Hatch), so even 1.6 m leaves at least 1.76 m on each side.
- **The run list** is `scripts/gw8_runs.py`, run by `scripts/gpu_round.sh`. Names are `gw8-m<M>-s<seed>`, seed-major with the arms rotated. As in GW-5, the runs carry no "config" entry. Each run records its margin in `runs/<name>/gw5.json`.
- **The box.** One `massedcompute_A6000_plus` (the standing permission) at 5 lanes: about 2 h for 15 runs, roughly $1.5. It is deleted once the runs are back.

## Outcomes

- **Primary: R-dev circuits completed per run** (completion > 0.5, of 12), from each run's end-of-run `eval_rdev.json`.
  - 1.2 m and 1.6 m are each compared with 1.4 m by the exact permutation test of the mean difference (5 against 5, 252 relabellings), two-sided.
  - The mean difference is reported with it.
- **Lap gap:**
  - each run's median gap on the circuits it completes, with the arm means, compared with 1.4 m in the same way;
  - like for like: the gap on the circuits both same-seed runs complete (1.2 against 1.4 m and 1.6 against 1.4 m), as the pooled mean difference and the share of circuit pairs in which the arm is slower.
- **Secondary (descriptive):**
  - training throughput (steps per second, from `metrics.csv`);
  - at the 12 real R-dev crash unit-phases, with X-2's machinery as in GW-5's `edges` stage, each network's base clearance c_min and its L/e for σ_b = 4 and 8 m bumps and trapezoid G (the median over the unit-phases). Each is compared with 1.4 m, two-sided.
- **Reference (descriptive only, other seeds and day):** GW-5's `ctrl` (8.0 circuits, mean of median gaps 8.17%) and `margin` (10.6, 9.39%), seeds 70–74.

## The recommendation rule (fixed now; a selection rule, not a test)

- **1.6 m** is recommended if its mean circuits completed is at least 1.0 above 1.4 m's.
- **Otherwise 1.2 m** is recommended if both hold:
  - its mean circuits completed is at most 0.4 below 1.4 m's;
  - its like-for-like gap against 1.4 m is at least 0.3 points lower (pooled mean difference ≤ −0.3 points).
- **Otherwise 1.4 m** stays.
- A 1B run of the result is not part of this plan.

## Limits, stated now

- **Small samples.** There are 5 seeds per arm, so differences of a fraction of a circuit are within noise. The rule selects; it does not establish a difference.
- **No held-out set.** R-dev completion is in-sample for widths, and R_TEST is spent.
- **One budget.** The sweep is at 200M. The ranking may differ at 1B, where GW-6 ran 1.4 m.
- **The edge measures** use step 2's shared start states at real corners, not each network's own lap states.

## Outputs

- **Runs:** `runs/gw8-*`, pulled back with `gpu_sync.sh pull-round`.
- **The analysis:** `scripts/gw8_analysis.py`, written while the round trains and committed before the runs come back, writes `outputs/gw8/{completion,edges}.json` (committed).
- **The findings note:** `docs/findings/gw8.md`.
