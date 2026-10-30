# GW-5, GW-6 and GW-7: plan (exploratory; written before any run, 30 Oct 2026)

**Status.**
- **What this covers.** These are the follow-through's optional training levers (`docs/plan/follow-through.md`, Section 7.4): GW-5, the training-only margin and local width perturbations; GW-6, a 1B pilot of the chosen recipe; and GW-7, SimBa.
- **Why they run now.** The user asked for them on 30 Oct 2026, after E0 and the lockbox. Decision #13 had put SimBa out of scope; the user's request brings it back.
- **What they can and cannot change.** All three are exploratory and change no reported result: `generator-v1`, `sim-v1`, E0 and the lockbox are done.
- **The order.** This plan and its code are committed and tagged (`gw5-v1`) before any run. Before that, the only things run were three 41k-step smoke runs (one per new arm, no evaluation) and a check of the bumps on three strips, as plumbing.
- **What is used.** Only R-dev data and the A6000 are used. R_TEST is not involved.

## The round (GW-5 and GW-7): 4 arms × 5 seeds at 200M

- **The recipe** is E0's: copied R-dev widths (`--real-widths`), v1, MLP 3 × 256, `train_dr.py`'s defaults, at 200M steps. The seeds are 70–74, unused so far.
- **The arms:**
  - **`ctrl`**: no lever;
  - **`margin`** (GW-5): a training-only termination margin of 1.4 m (`--train-edge-margin 1.4`). The plan's range was 1.3–1.5 m. Evaluation stays at 1.0 m;
  - **`bumps`** (GW-5): local width bumps in training (`--local-bumps`). Each side of every generated strip gets 96 Gaussian bumps:
    - centres uniform along the strip;
    - σ log-uniform in 2–10 m, so a bump's ±σ span is 4–20 m;
    - amplitude uniform in ±1 m;
    - widths never pushed below min(own, 2.5 m), and the generator's inside cap applied again.

    The bumps land on both sides, so the inside edge of every corner gets them. On three test strips they changed the widths by 0.18–0.19 m RMS.
  - **`simba`** (GW-7): the SimBa network already in `agents/ppo.py` (`--network simba`, critic width 384).
- **The code.**
  - `scripts/gw5_train.py` runs `scripts/train_dr.py` unchanged. For its own process only, it replaces the two functions the levers need (`procedural_env`'s parameters and `adapters.generate`). No file under `src/` and no frozen script changes.
  - The run list is `scripts/gw5_runs.py`, run by `scripts/gpu_round.sh`. Its runs carry no "config" entry, because `round_manifest.py` cannot parse `gw5_train.py`'s arguments. Each run records its levers in `runs/<name>/gw5.json`.
- **The box.** One `massedcompute_A6000_plus` (the standing permission) at 5 lanes, about 2.5–3 h, roughly $2. It is set up and synced with `gpu_setup.sh` and `gpu_sync.sh`, which push R-dev data only. It is deleted when GW-6 is done, or when no GW-6 run is needed.

## Outcomes

- **Primary: R-dev circuits completed** (completion > 0.5, of 12), from each run's end-of-run `eval_rdev.json`. This is E0's gate measure, and it is in-sample for widths.
  - Each lever arm is compared with `ctrl` by the exact permutation test of the mean difference (5 against 5, 252 relabellings).
  - The test is one-sided (arm > `ctrl`), with Holm over the three arms at 0.05. The mean difference and the two-sided p are also reported.
- **Secondary (descriptive):**
  - **The lap-time gap:** each run's median gap on the circuits it completes, from `eval_rdev.json`, with the arm means.
  - **Training throughput** (steps per second, from `metrics.csv`).
  - **Short-scale edge following,** the outcome `locality.md` named for a width lever:
    - What is measured: each network's L/e for σ_b = 4 and 8 m bumps at s\*, and the trapezoid G, at the 12 real R-dev crash unit-phases of step 2's H3, with X-2's machinery (its plan, `docs/findings/x2-plan.md`, follows this one). Each network's value is the median over the unit-phases.
    - The comparison: `margin` and `bumps` against `ctrl`, by the same permutation test, two-sided.
    - SimBa is included only if the frozen fields code can load it; otherwise it is reported as not measured.

## GW-6: the rule for the 1B pilot (fixed now)

- **The plan's condition holds.** GW-6 runs "only if GW-1's curves are still rising", and GW-1 labelled the late curve rising (+1.57 circuits from 110M to 200M; `gate-diagnostics.md`).
- **Choosing the recipe.** The candidate is the lever arm with the largest mean gain in circuits completed over `ctrl`. Ties go to the smaller one-sided p, then to the order bumps, margin, simba.
  - It is **chosen** if its gain is at least +1.0 circuit and its one-sided p is at most 0.10, unadjusted. This is a selection rule, not a test.
- **If an arm is chosen,** GW-6 is that recipe at 1B (`--steps 1e9`), seeds 80–84, on the same box at 5 lanes (about 3.5–4 h, roughly $2.5).
  - Its comparison is E0's five 1B runs: the same recipe without the lever, the same code (E0's code paths unchanged), on another box and day. It is compared by R-dev circuits completed per run, with the two-sided permutation test (5 against 5), and by whether the median reaches E0's gate of 10 of 12.
  - Its R-dev numbers are in-sample for widths, as E0's are.
- **If no arm is chosen,** the chosen recipe is E0's own, and its 1B pilot already exists: E0 itself (5 × 1B, median 8 of 12). GW-6 then trains nothing new and says so.

## Limits, stated now

- **Small samples.** There are 5 seeds per arm. Each lever has one setting, chosen within the plan's range (margin) or by judgment (the bumps' density, scale and amplitude). A null result does not rule out other settings.
- **No held-out set.** R-dev completion is in-sample for widths, and R_TEST is spent, so no held-out completion is measured.
- **GW-6 against E0** differs in box and day as well as the lever.
- **The edge-following measure** uses step 2's shared start states at real corners, not each network's own lap states.

## Outputs

- **Runs:** `runs/gw5-<arm>-s<seed>` (and `runs/gw6-*` if GW-6 trains), pulled back with `gpu_sync.sh pull-round`.
- **The analysis:** `scripts/gw5_analysis.py` writes `outputs/gw5/analysis.json` (committed).
- **The findings note:** `docs/findings/gw5.md`.
