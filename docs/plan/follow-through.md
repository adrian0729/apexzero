# Follow-through plan (25 Oct 2026)

This plan collects everything discussed but not yet done, puts it in order, and costs it. It merges six read-only scoping reports with my own check of the "What next" sections in the findings. Nothing in it has been run. Two of the exploratory results are suggestive and come up for discussion afterwards: the width test's leave-circuits-out transfer (p 0.063) and the locality study. Section 9 sets up that discussion.

Sources:
- `docs/STATUS.md` and `docs/plan/two-track-plan.md`;
- your notes `ApexZero-plan.md` and `ApexZero-novelty-and-outputs.md`, which I read but did not edit;
- the findings in `docs/findings/`: `costate.md`, `width-test.md`, `width-test-loo-plan.md`, `h4r.md`, `locality.md`, `receptive-fields-step2.md`, `psychophysics-step1.md`, `generalist-seeds.md`, `to-ocp-spike.md`, `pilot-audit.md`, `maze-audit-pilot.md` and `drivers-pilot.md`;
- `docs/interp/costate-design.md`, `docs/interp/psychophysics-protocol.md`, `paper/sections/*.tex`, and the code under `src/apexzero/` and `scripts/`.

The scoping did not read `outputs/sealed/`, and it evaluated nothing on R_TEST.

## 1. Conventions

**Labels**
- **Confirmatory:** a pre-registration and a tag come before any outcome is computed (the H4R pattern).
- **Exploratory:** a plan with its expected directions is committed first, and the result is labelled exploratory (the `locality-plan.md` pattern).
- **Engineering:** code, hygiene or infrastructure that makes no claim.

**Machines**
- **Mac:** the M3 Pro with 12 cores. "Mac" times are wall time, taken from timings recorded in the docs unless marked *(est.)*. "Build" is developer time, given only where a report estimated it.
- **A6000:** the cheapest Massed Compute box. You have given standing permission to start it. It has 12 vCPU, so it does not speed up CPU solves, and the account seems limited to one instance at a time.
  - Recorded rate: about 6.5 min of box time per 200M run at 4 lanes. For example, H4R trained 20 runs in 2 h 13 min.
  - Its hourly price is not recorded in the repo. The reports assumed $0.5–1 per box-hour, so check Brev.
- **Big CPU box:** quotes of $3.16/h for 128 vCPU and $5.53/h for 224 vCPU, about $0.025 per vCPU-hour. It needs your OK each time.

**Lockbox rule.** Nothing is evaluated on `apexzero.tracks.corners.R_TEST`, and `outputs/sealed/` is not opened, until `protocol-v1` is tagged. R_TEST geometry never leaves the Mac.

## 2. The short version

- **Phase 0, now (hours, Mac, $0):**
  - lockbox hygiene: three stray lockbox files sit outside `outputs/sealed/`;
  - STATUS corrections;
  - paper upkeep;
  - the driver-registration rule that was decided but never applied in code.
- **Phase 1, analyses of what is already on disk (Mac, $0, a plan first for each):**
  - Idea 2's cheap leftovers;
  - the zero-training diagnostics for the 80% gate;
  - step 2's two unwritten summaries;
  - the H4R lap gap;
  - P4 made testable;
  - the drivers power simulation;
  - a targets-only pilot for braking and corner states.
- **Phase 2, builds on the Mac:**
  - the statistical width model and width report: the critical path;
  - the paper's labeller;
  - the lockbox harness;
  - the value-gradient term, if you want that study;
  - Idea 2's larger exploratory builds.
- **Phase 3, training on one A6000 at a time:**
  - one combined, pre-registered width round: the critical path;
  - optionally the value-gradient study, a bounded lever round, a 1B pilot and maze E5.
- **Phase 4, freeze and train the final agent:**
  - apply the generator-v1 rule and tag `generator-v1` and `sim-v1`;
  - train E0 (5 × 1B);
  - if the paper is un-paused: G1, production labels and the curriculum runs.
- **Phase 5, the lockbox, once:** `protocol-v1`, the unseal on the Mac, one analysis.

**Critical path to the lockbox.** HY-1 → GW-1 → GW-2 → GW-3 pre-registration → GW-3 round → GW-4 rule (with GW-5 and GW-6 optional and bounded) → tags → E0 → checkpoint choice on R-dev → LB-0 dry run on R-dev → `protocol-v1` → unseal.
- Compute on that path is roughly 15–25 A6000 box-hours plus a few Mac hours.
- Build and review rounds will dominate the elapsed time, not compute.

**Totals, roughly**

| Scope | A6000 box-hours | Big CPU box | Mac |
|---|---|---|---|
| Without the paper | 15–40 | none | $0 |
| With the paper | 40–85 | about $40–85 for production labels (your OK needed) | about 90 CPU-h more, $0 |

- At the assumed A6000 rate, the GPU side is tens of dollars.
- Idea 2's leftovers are about 10–14 h of Mac time and $0.

**One new observation changes how the leave-out result reads.** This scoping found it from tables already in the findings; it is post hoc and descriptive (Section 9.1).
- Against H4R's v3 baselines, which have the same code as the leave-out runs, the v3 transfer gap is about zero.
- The width effect is reliable in v1, where no leave-out runs were trained.

## 3. Summary table

Phases: 0 is now; 1 is existing data; 2 is Mac builds; 3 is A6000 rounds; 4 is freeze and final agent; 5 is the lockbox; "D" means held for the discussion after this plan. "#n" points to a decision in Section 10.

| ID | Item | Label | Mac | A6000 h | Other $ | Depends on | Risk | Value | Ph |
|---|---|---|---|---|---|---|---|---|---|
| HY-1 | Seal 3 stray lockbox files; `solve_all_ocp.py` defaults to R_DEV; replace STATUS's Monza K2a figure | Eng | minutes | – | – | #3 | low | required | 0 |
| HY-2 | STATUS fixes: TO-OCP timing (drop lockbox windows), drivers-lite numbers | Eng | 15 min | – | – | – | low | upkeep | 0 |
| PA-0 | Paper upkeep: stale TODOs, R-dev-only spike paragraph | Eng | 1–2 h (est.) | – | – | HY-2 | low | upkeep | 0 |
| DV-A | Registration gate (p99 excursion ≤ 8 m) in code; pin the 4 pilot sessions first | Eng | 1–2 h build | – | – | – | low | unblocks drivers | 0 |
| DV-D2 | Regenerate the snapping-audit panel with legible labels | Eng | ~1 h | – | – | – | low | low | 0 |
| CS-0 | Idea 2 exploratory scaffold and committed plan | Eng + plan | build, small | – | – | #9 | low | needed by CS-* | 1 |
| CS-1 | ξ-tail investigation | Exp | < 30 min | – | – | CS-0 | low | moderate–high | 1 |
| CS-2 | Descriptives pack (unwinsorised R², per-head ρ_k, …) | Exp | < 15 min | – | – | CS-0 | low | low | 1 |
| CS-3 | C3-int (network × geometry) | Exp | < 15 min | – | – | CS-0, #8, #16 | medium | low–moderate | 1 |
| CS-4 | ΔC1, R²_z and the MLP-on-z ceiling | Exp | ~20 min pooled | – | – | CS-0 | medium | moderate | 1 |
| CS-5 | W1 on the Π targets | Exp | < 1 h | – | – | CS-0 | medium | low–moderate | 1 |
| GW-1 | Zero-training gate diagnostics | Exp | ~1.5–2 h (est.) | 0 (opt. ~1.5) | – | plan | low | high | 1 |
| IN-1 | Step 2: pair-weighted anchored-key G | Exp | < 1 h (est.) | – | – | plan | low | low–moderate | 1 |
| IN-2 | Step 2: common-key H2′ summary | Exp | < 1 h (est.) | – | – | plan | low | moderate | 1 |
| IN-3 | Step 1 outputs never written up (P2d, P3 ratio, P6d, P8-F, P6 curves) | Exp (descr.) | 2–3 h (est.) | – | – | – | low | low | 1 |
| IN-4 | H4R lap gap on circuits both arms complete | Exp (descr.) | < 1 h (est.) | – | – | – | low | moderate (feeds GW-4) | 1 |
| PS-P4 | Step 1's P4 made testable (option A) | Exp | < 3 h | (B2: ~1) | – | plan, #7 | medium | low–moderate | 1 |
| DV-B0 | Drivers power simulation | Exp | 2–4 h build, seconds | – | – | DV-A | low | decides DV-B | 1 |
| BC-a0 | Braking/corner targets pilot (no network outputs) | Exp | 4–8 CPU-h | – | – | plan, #6 | medium | decides BC-1 | 1 |
| GW-2 | Statistical width model (2a; 2b optional) and width report | Exp | minutes compute; build medium (2a), large (2b) | – | – | GW-1 | medium | high | 2 |
| LB-0 | Lockbox harness, tests, and an R-dev dry run | Eng | build medium | – | – | HY-1 | low | required for LB-1 | 2 |
| PA-1 | Labeller, 200 pilot labels, repeatability | Eng + pilot | ~80 CPU-h (8–12 h wall, nights) | – | – | #4 | medium | essential for the paper | 2 |
| PA-2 | Pilot audit with real labels (5 seeds) | Exp (pilot) | ~4 h wall | – | – | PA-1 | medium | high per CPU-h | 2 |
| VG-0 | Value-gradient design, PPO term and tests | Eng → Conf design | build medium–large | – | – | #5 | medium | see VG-1 | 2 |
| CS-10 | C1-own on own-run states, then on N | Exp | ~45 min pooled (+4 min/net) | – | – | CS-1; CS-9 for (iii) | medium | moderate–high | 2 |
| CS-11 | Own-policy width price Π^{in,own} | Exp | 15–50 min pooled | – | – | CS-0; CS-9 for N | medium | moderate–high | 2 |
| CS-7 | Critic-layer and head-unit probes | Exp | ~30 min pooled | – | – | CS-0 | low | moderate | 2 |
| CS-8 | C2-sw random-only null, residualised variant, C2 decoy | Exp | 20 min – 3 h, plus ~2 h for (b) and (c) | – | – | CS-0, #16 | low | moderate | 2 |
| CS-9 | N: per-agent replication of C1/C2/C2-sw | Exp | 45–60 min pooled + build | – | – | CS-0, #16 | medium | moderate–high | 2 |
| CS-6 | C1-dev over 16 snapshots | Exp | ~1 h | – | – | CS-1 | low | low–moderate | 2 |
| CS-12 | C2/C2-sw beyond the 9 decision networks | Exp | 45–60 min (primary) / ~4.5 h (full) | – | – | CS-8 | low | low–moderate | 2 |
| CS-13 | C2-pairs | Exp | ~30 min | – | – | CS-9 for N | low | low | 2 |
| CS-14 | q_c, the action-projected critic signal | Exp | ~1.5 h (+53 min optional solves) | – | – | CS-0, #8 | high | moderate–high (conceptual) | 2 |
| CS-15 | FWL check, actor h1/h2 probes, C2 on D | Exp | ~1 h | – | – | CS-0 | low | low | 2 |
| DV-B | Drivers: pre-registration, blind QA, run once, report | Conf | 15–25 h effort; < 1.5 h CPU | – | – | DV-B0 go, #18 | medium (gate likely fails) | moderate | 2 |
| DM-1 | Demo: render mode, MVP-1 lite clip, R-dev generalist ghost | Exp (demo) | 8–11 h effort | – | – | #2, #19 | low | moderate | 2 |
| PA-3a | Maze E5 pre-registration; vendor SFL | Conf (design) | build medium | – | – | #2, #17 | low | see PA-3 | 2 |
| GW-3 | Combined width round: baseline, copied leave-out, stat-model leave-out, stat-model all-12 | Conf | analysis minutes | ~7 (v1) / ~14 (v1 + v3) | – | GW-2, #7, #13 | medium (batch, power) | high | 3 |
| VG-1 | Value-gradient pilot, then 20 or 40 runs | Exp → Conf | ~45 min to measure | 4–8 (7–14) | – | VG-0, #5 | high | moderate (only causal test) | 3 |
| GW-5 | Lever round: training margin, local width perturbations | Exp | fields stage ~3 h per 20 networks | ~1.7 | – | GW-3/GW-4 | medium | medium | 3 |
| GW-6 | 1B pilot of the chosen recipe | Exp | – | ~2.8 | – | GW-1 curves | low | medium | 3 |
| GW-7 | SimBa trial | Exp | – | 1–2 | – | #13 | medium | low–medium | 3 |
| PA-3 | Maze E5 at scale (30 runs and the audit) | Conf | audit ~38 h if on the Mac | 3–6 (unmeasured) | – | PA-3a | medium | moderate | 3 |
| GW-4 | Apply the generator-v1 rule; tag `generator-v1` and `sim-v1` | Eng | minutes | – | – | GW-3 (+GW-5/6) | medium | critical path | 4 |
| E0 | E0: 5 seeds × 1B on the tags | Training | checkpoint choice ~40 min | ~3.5–4 | – | GW-4, #15 | medium | critical path | 4 |
| PA-4 | E0-sym, E0-const, E0-short | Training | – | ~4–5 more | – | GW-4, #17 | medium | paper only | 4 |
| PA-5 | Production labels (5,400 levels × 3 starts, plus subsets) | Measurement | 5–8 days if on the Mac | – | $40–85 CPU box | PA-1, PA-2/G1, E0, #1 | medium | contribution 1 | 4 |
| PA-6 | 21 curriculum runs (PLR⊥, SFL, DR-pool) | Conf | – | 12–18 | – | PA-5, G1 | medium (power) | RQ3 | 4 |
| PA-7 | E1/E2, buffer audit, RQ3, fill the paper | Conf | E1 ~70 h if on the Mac | 2–5 (+4–10 for E1) | – | PA-6 | medium | the results | 4 |
| BC-1 | Braking/corner confirmatory study | Conf | 9–35 h wall | – | ~$5–20 on a 64-vCPU box (my arithmetic) | BC-a0, #6 | medium | low–moderate | 4 |
| LB-1 | `protocol-v1`: pre-register, tag, unseal, analyse once | Conf | ~3–6 h | – | – | E0, PS-P4, LB-0, #14, #15 | high (one shot) | high | 5 |
| X-1 | Locality profiles on H4R's 20 same-code networks | Exp | cheap (stored arrays) | – | – | plan | low | moderate | D |
| X-2 | Locality on the real crash profiles (12 unit-phase VJPs) | Exp | small (est.) | – | – | plan | medium | moderate | D |
| X-3 | Finite-amplitude closed-loop locality checks (R-dev roads) | Exp | small (est.) | – | – | plan | low | low–moderate | D |

Later or optional: DV-B5 (M1 hierarchical model, only if the drivers gate passes), DM-C4/C5, DV-D3/D4/D6/D7, LB-opt (TO-OCP lap mode, fixed-path OCP). See Section 7.

## 4. Phases

### Phase 0: hygiene and recorded decisions (now; Mac; hours; $0)
1. **HY-1** (needs your OK, #3). It changes the sealed set, so do it first.
2. **HY-2 and PA-0.**
3. **DV-A.** Pin the four pilot session keys before changing the gate.
4. **DV-D2**, if convenient.
5. Before Phase 1, decide #8 (the braking and corner hygiene rule) and #9 (the costate freeze route). They decide how Idea 2's exploratory work may run.

### Phase 1: analyses of what is already on disk (Mac; $0; each with a plan committed first)
- **Idea 2:** CS-0, then CS-1, then CS-2 to CS-5 in one session. Together they take under about 2 h of compute.
- **Gate diagnostics:** GW-1. It decides between the training-budget lever and the margin lever, and it shapes GW-2.
- **Small write-ups:** IN-1, IN-2 and IN-4; IN-3 only if wanted.
- **PS-P4, option A.** It has no dependency, but it must be done before `protocol-v1`.
- **DV-B0**, then your go or no-go on DV-B (#4, #18).
- **BC-a0.** It reads targets only, so it keeps the confirmatory option of BC-1 open.
- **Decision point after Phase 1:** whether to do VG (#5), BC-1 (#6) and DV-B, and which paused tracks to un-pause (#4).

### Phase 2: builds on the Mac (parallel lanes, see Section 5)
- **Critical path:** GW-2 with its width report, then the GW-3 pre-registration.
- **Off the path:**
  - LB-0, on R-dev only;
  - PA-1 labels on Mac nights, then PA-2, if the paper is un-paused;
  - VG-0, if VG is chosen;
  - CS-7 to CS-15 in the order of Section 7.2;
  - DV-B;
  - DM-1;
  - PA-3a.

### Phase 3: A6000 rounds (one box at a time)
Order on the box:
1. **GW-3.** It is on the critical path.
2. **GW-5 and GW-6,** only if the gate is still far and GW-1's learning curves are still rising.
3. **VG-1 and PA-3**, in the gaps while the critical path waits on Mac work.

- If a positive VG result could change E0's recipe (#5), VG-1 must finish before GW-4.
- A big CPU box cannot run while the A6000 does.

### Phase 4: freeze and train the final agent
- **GW-4.** Apply the rule committed before GW-3, then tag `generator-v1` and `sim-v1`.
- **E0, 5 × 1B.** The gate is formally read here (#10). PA-4 runs alongside it if the paper is un-paused.
- **If the paper is un-paused:** PA-2 on the final generator, then G1 (`audit-v1`), then PA-5 on a big CPU box (#1), then PA-6 and PA-7.
- **BC-1 and CS-14** can run here or at any later point. They do not touch the agent.

### Phase 5: the lockbox (once)
**LB-1.**
1. Choose checkpoints on R-dev and run the full R-dev dry run.
2. Write, review and tag `protocol-v1` (#14, #15).
3. Unseal on the Mac: references, then laps, then `analyse` once, then a findings document.
4. Optionally, open `outputs/sealed/` for a provenance comparison.

## 5. What can run in parallel, and what shares files

| Lane | Items | Files it touches | Conflicts |
|---|---|---|---|
| A: Idea 2 exploratory | CS-0…CS-15 | new `scripts/costate_exploratory.py`, `docs/findings/costate-followups-plan.md`, `outputs/costate/exploratory/` | The costate freeze hash covers all of `src/`, untracked files included, so any `src/` edit on main makes it refuse. Run it from a worktree at `a48841e` with `outputs/` pointed at the main checkout (outputs are not in git), or byte-check only the modules it imports (#9) |
| B: generator | GW-1, GW-2, GW-3 scripts, GW-5 | `src/apexzero/tracks/{generator.py, widths.py, reference_jax.py}`, `scripts/train_dr.py`, `scripts/width_report.py`, new queue and analysis scripts, tests | `train_dr.py` with lane C; `env/racing.py` with GW-5's margin |
| C: value gradient | VG-0, VG-1 | `src/apexzero/agents/ppo.py`, `src/apexzero/env/adapters.py`, `scripts/train_dr.py`, new `scripts/vgl.py`, tests | `train_dr.py` with lane B. Land one lane's flags first; each GPU round trains on one pinned commit |
| D: paper labels | PA-1, PA-2, later PA-5 | new `src/apexzero/ocp/labels.py`, `scripts/label_levels.py`; reads `ocp/window.py`, `costate/workers.py` | Mac CPU at night (see below) |
| E: drivers | DV-A, DV-B0, DV-B | `src/apexzero/drivers/*`, `scripts/drivers_*.py`, `tests/test_drivers_timing.py` | none |
| F: lockbox | HY-1, LB-0, LB-1 | `scripts/solve_all_ocp.py`, new `scripts/lockbox.py`, `viz/export.lockbox_guard` | `lockbox_guard` with DM-C4: keep both in one lane |
| G: interp small | PS-P4, IN-1…IN-4, X-1…X-3 | new scripts outside the frozen harnesses (the `fields_exploratory_h4.py` precedent) | Check each frozen harness's guard before writing |
| H: demo | DM-1, DM-C4/C5 | `src/apexzero/viz/*`, `scripts/export_demo.py` | `lockbox_guard` (lane F) |
| I: maze | PA-3a, PA-3 | `third_party/`, `scripts/maze_audit.py`, `src/apexzero/maze/` | none |

Shared resources:
- **`docs/STATUS.md`.** Every lane will want to edit it. Update it only at merge, from one session.
- **The A6000.** One instance at a time, so GW-3, GW-5, GW-6, VG-1, PA-3, E0 and PA-6 queue behind each other, and the big CPU box waits for them.
- **Mac CPU.** Lane A's pooled stages use 5 workers × 2 threads. PA-1's label nights, BC-a0 and BC-1 each want most of the machine. Do not overlap them.

## 6. Constraints that cut across items

- **Stray lockbox files.**
  - `outputs/monza_ocp.pkl` and `outputs/monza_ocp_ds2.pkl` are Monza optima outside `outputs/sealed/`.
  - `outputs/ocp/summary.csv` has 26 lines but only 13 R-dev/IMS `.pkl` files beside it, so it very likely holds the 12 lockbox rows. These were checked by name, size and line count only.
  - STATUS says all 24 such files are sealed.
  - `solve_all_ocp.py` with no `--tracks` still solves every circuit. The core library (`load_tumftm`, `solve_min_time`, `timed_laps`) has no `lockbox_guard`.
- **STATUS quotes lockbox numbers.**
  - The K2a row gives "Monza: 0.0003%".
  - The TO-OCP speed row's "46–232 CPU-s, mean 97" appears to include the circuit windows, which contain Monza, Spa, Silverstone and Budapest (`to-ocp-spike.md`, around lines 17 and 110–113). Budget labels from the pool:0–7 rows instead: about 775 CPU-s per level with 3 starts.
- **Braking and corner rows.** The critic gradients `g` and the λ targets on those rows are on disk and have never been compared.
  - If BC-1 might be confirmatory, no exploratory item may compare them first. That affects CS-3's n/vx parts, the braking-phase rows of CS-9 and CS-10, and CS-14.
  - Until #8 is decided, keep lane A on K2b's decision rows.
- **Default generator bit-identical.** New width options must default to `None`, so the frozen psychophysics, fields and costate harnesses are unchanged.
- **Driver pilot list.** `pilot_sessions()` picks the first four accepted sessions. Under the new gate it would silently switch the pilot from Suzuka/Montreal/Spa/Monza to Shanghai/Suzuka/Sakhir/Montreal. Pin the list before DV-A.
- **R-dev erosion.** Each recipe round chosen on R-dev weakens R-dev as a proxy for R-test. Keep the fold-held-out reading as the honest metric and bound the number of rounds (#11).
- **Idea 2 as the last study.** On 24 Oct you said Idea 2 was meant to be the last planned study, and big-box stage parallelism stays dropped.
  - The CS, BC and VG items are therefore optional follow-through, not commitments.
  - PA-5's per-level worker pool is a different thing, but the box still needs your OK.

## 7. Items by workstream

### 7.1 Hygiene and upkeep
- **HY-1.**
  - Move the three files into `outputs/sealed/` whole, and add a dated addendum to STATUS's deviation section.
  - For R-dev figures, use `outputs/paper-draft/ocp-rdev/summary.csv`.
  - Make `solve_all_ocp.py` default to R_DEV and guard its arguments.
  - Replace the Monza figure in STATUS with the R-dev recomputation. `environment.tex` has a TODO asking for the same.
- **HY-2.**
  - The TO-OCP figure, as above.
  - The drivers-lite row: the findings say a held-out error of 11–19 ms and claim limits of 27–101 ms, while STATUS says 10–18 and about 24–100.
- **PA-0.** `estimators.tex` says the samplers are not built, but they are. The `answer_key.tex` TODO on μ_h is stale, because `solve(hidden_mu=)` exists. Replace "the spike is running" with the pool:0–7 and Hockenheim numbers only.
- **DV-A.**
  - In `drivers/registration.py`, the gate becomes p99 excursion ≤ 8 m. RMS and length become a reported diagnostic.
  - Update `drivers_report.py`, the comment in `circuits.py`, and the tests (a 25 m displaced section rejected; a line 2 m inside with RMS over 3 m accepted).
  - Check: rerun registration for 2023–25 (about 47 sessions × 2 s) and diff it against the archived JSON. Only the accepted flags should change.
- **DV-D2.** `run_audit.py` does not cache its results, so this means a rerun (minutes) or adding a cache.

### 7.2 Idea 2: leftovers on existing outputs (all exploratory, $0, Mac)
One plan, `docs/findings/costate-followups-plan.md`, committed before any computation, with expected directions and an amendment rule. The code goes in a new script outside the hash. It checks the sha256 of every bound array it reads (`decisions.json` binds 113) and calls `unlock_trained()` itself, as `_post_tag` does.
- None of these can be confirmatory on these 34 networks.
- K2b's frozen exclusions still apply: only λ_s (1,643 rows) and λ_ξ (1,581 rows) are scored. A score that ignores them is a labelled sensitivity.

**Recommended order**
- **CS-1, the ξ tail.**
  - Locate the 14–21% of ξ rows with the wrong sign, by:
    - geometry: geometry 38 holds 10–25% of the error;
    - observation-clip flags;
    - the preview-spacing kinks at vx 25 and 83.3 m/s;
    - the tyre-channel kinks.
  - Split g_ξ into critic and map parts with a fresh VJP.
  - Report R²_c under the Huber and rank calibrations beside the frozen one.
  - It decides which R²_c later items co-report. It cannot change the refute: bulk R²_c is at most 0.092.
- **CS-2 to CS-5 in one pass.**
  - **CS-2:** per-head ρ_k is its only interpretive part.
  - **CS-3:** C3-int. Double-centring over networks removes T's λ, so it tests network-specific gradient deviation against behaviour, not mispricing.
  - **CS-4:** the MLP ceiling says whether λ_s and λ_ξ are decodable from what the network sees at all.
  - **CS-5:** the Π targets are already in `world.npz`. K2b was never measured for them.
- **CS-10 and CS-11 next.**
  - **CS-10:** λ^own at every state of each network's own runs gives far more rows than C1-own's roughly 162 per component.
  - **CS-11:** the own-policy inside-width price is the only comparison that can support "the networks do not value the inside wall the way the optimum does".
  - Measure the cost on one network first. The report estimated about 10–15 min per network.
- **CS-7 and CS-8.**
  - **CS-7:** do the critics hold λ in another layer?
  - **CS-8:** a random-only null for C2-sw. Only its variant (b), with σ^ped residualised on position and speed, removes the named confound.
- **CS-9 (N), then CS-10 (iii), N's CS-11 and CS-13.**
  - N is the only score at the networks' own states.
  - All 30 of N's geometries are among T's 48, so the frozen folds and winsor bounds apply.
  - Exclude the 6 K0 keys (#16). They are at most 2 of about 700 states per agent.
- **CS-6, then CS-12 (primary only), then CS-14 (only before VG, if wanted), then CS-15 on request.**

Total for CS-0 to CS-13: about 10–14 h of Mac time over a few sessions, with no new OCP solves. Three optional new solves are #16's.

### 7.3 Idea 2: new studies (findings Section 10, items 4 and 5)

**BC: braking and corner states**
- **Why:** 86% of λ_s's variance lies in corner states, and C1 was decided where λ_s barely varies. The prior for a confirm is low, so a refute is the likely result. It would extend the conclusion, not change it, and it does nothing for driving performance.
- **BC-a0 (exploratory):**
  - K2b's 26 keys × 8 grammar continuations, plus per-state placement on about 50 states;
  - about 4–8 CPU-h on the Mac;
  - it measures how much the unseen road moves λ under realistic continuations.
- **If that share is small:** pre-register C1 on braking and corner rows against the existing λ, with a stated bound. That is cheap.
- **If it is large:** build the preview-matched average, option (ii).
  - 2,298 interior states × M draws × about 60–75 s gives 150–380 CPU-h.
  - That is 9–35 h of wall time on the Mac, or a few hours on a 64-vCPU box (#1).
- **Integrity:** bind the sha256s of the existing g and λ on these rows. Approach-phase s and ξ stay decided under `costate-prereg-v1`.

**VG: value-gradient term in the critic's loss**
- **The test:** a GDHP-style term with VGL(λ) targets through the JAX simulator, tested like H4R. Baseline and VGL arms on fresh seeds (15 and up), one box and one commit. The outcomes are C1's frozen R²_c (with a rank test) and R-dev completion.
- **VG-0, on the Mac:**
  - `PPOConfig` fields, off by default;
  - per-transition env-state capture, and the pre-reset state in `adapters._bank_step`;
  - the target scan and the double-backprop term;
  - tests: β = 0 reproduces the current loss; finite differences; on 1-D braking, the gradient lands nearer the exact value gradient;
  - the design and its pre-registration.
- **VG-1, on the A6000:**
  - a pilot of about 6 runs (1–2 box-h), with β chosen on manipulation metrics only;
  - then 20 runs (2.5–5.5 h) or 40 runs (5–11 h), because 5 per arm has weak power for completion;
  - the overhead per update is unmeasured; the report guessed 1.5–3×.
- **Value:** it is the only causal test of the critic-to-actor path. If the manipulation works and completion does not move, "fix the critic's gradient" is closed as a lever.
  - The prior for a completion gain is low: the critic reaches the actor only through advantages, C3-w found no strong association, and the crash evidence points to width perception.
- **A design tension for you (#5):** the one-step action projection barely touches s and ξ, the components C1 scores.

### 7.4 The generator's width model and the 80% gate

**Where the gate stands**
- It is defined at E0, 5 × 1B: R-dev ≥ 80%, in practice 10 of 12 circuits.
- No run on disk has reached it reliably. All 52 runs use one recipe (MLP 3×256, 2,048 envs, 200M steps).
  - Baselines: 0 of 25 reach 10 or more.
  - In-sample real-width runs: 2 of 26.
- Pooled per-circuit completion for in-sample real widths: Norisring 6%, Yas Marina 6%, Sochi 39%, Melbourne 50%.
- The generator report's judgment of the odds:
  - under 10% that the median seed reaches 10 of 12 at 200M with today's levers;
  - perhaps 15–30% at E0 with a working width model, 1B steps and one more lever that works.
- The plan's fallback is already written: early-versus-mid in `audit-v1`, and the showcase seed extended to 2–3B. The pivot comes only below 50% at 2B.

**The items**
- **GW-1 (exploratory, high value, near-zero cost).**
  - R-dev completion against training steps from the 16 checkpoints already stored per run, for example 6 checkpoints × 28 same-code runs, about 170 evaluations.
    - An evaluation takes about 30 s on the Mac by the lockbox report, or about 3.5 min of A6000 time by the generator report. Time one on the Mac first.
  - Deterministic A_DR completion of the finals.
  - H4R Section 10.2's crash comparison: are the extra completions avoided inside-edge crashes, with clearance larger by about the B shift? `diagnose_crashes.py` needs an output-path change.
  - It decides between the budget lever and the margin lever.
- **GW-2 (exploratory build).** An opt-in width model that copies no evaluation circuit.
  - **(2a), recommended first.** Circuit-level median width with jitter, slow variation, a corner-coupled inside dip and outside widening, multi-scale residual noise per side, then the existing inside cap.
  - **(2b).** Generate the road, then re-express it on a smoothed reference as `refine_reference` does. Evaluation pairs reference curvature with a non-parallel inside edge on 6 of 12 circuits, including Norisring and Yas Marina, and the generator never does. That is an untested hypothesis for the inside-edge crashes. (2b) needs a JAX port of the smoothing.
  - **Width report** (`scripts/width_report.py`): generator against R-dev per fold, covering per-side W1, the share of tight-corner knots with an inside-width range ≥ 1 m or ≥ 2 m over 100 m, apex dip by radius, spectra, and a no-copy check.
- **GW-3 (confirmatory).** One queue, one commit, one box, with lanes interleaved, fresh seeds 20 and up, and the committed folds.
  - Arms: baseline, copied leave-out, stat-model leave-out, stat-model all-12.
  - v1 must be in it, because that is where the width effect replicates (Section 9.1).
  - v1 alone is about 60–70 runs and about 7 h. Adding v3 doubles that.
  - It absorbs "more leave-out seeds". Run on its own, that would be about 40 runs and 4.5 h, and it is low value without the stat-model arm.
- **GW-4.** The rule is committed before GW-3. Adopt the stat model fitted on all 12 if:
  1. it beats baseline on held-out circuits by the pre-registered one-sided test;
  2. it is no more than about 0.5 of 6 below the copied arm;
  3. the median lap gap rises by under about 1 point (IN-4 informs this);
  4. the width-report bands hold;
  5. the default stays bit-identical.

  Otherwise iterate the model, or tag `generator-v1` with default widths.
- **GW-5 (exploratory, only if the gate is still far).** One 3-arm × 5-seed round at 200M, about 1.7 h:
  - a training-only termination margin, for example `edge_margin` 1.3–1.5 m against 1.0 m at evaluation. #13 decides whether it is part of `sim-v1`;
  - local inside-edge perturbations of 2–20 m, judged on locality's short-scale L/e, not corner-wide G.
- **GW-6.** A 1B pilot of the chosen recipe, about 2.8 h. Run it only if GW-1's curves are still rising. The learning rate and entropy anneal to zero, so a 1B run is not a continued 200M run.
- **GW-7 (SimBa), and the hairpin and curriculum levers:** see Section 8.

### 7.5 The paper (currently paused, #4)
- **PA-1:**
  - per level: an s-OCP, TO-OCP continuation from 3 starts, float64 replay, keep the best verified result;
  - a retry policy, a schema stamped with the `sim-v1` hash, resumable shards, exports for `pilot_audit --j-opt` and `train_ued --labels`;
  - to build: the QSS-line start and the agent-rollout start;
  - then 200 pilot labels plus 3 extra starts each, about 80 CPU-h. The spike had one failure in 8 levels, and that rate needs measuring.
- **PA-2:** `pilot_audit.py --j-opt` over the DR-pilot checkpoints, extended to seeds 1–4. It sizes G1's thresholds and the MDE, and tests whether late-phase claims can pass the 3× label-SD rule. Its levels must be disjoint from `audit-v1`'s.
- **PA-3:**
  - JaxUED maze at its defaults: 5 seeds × {DR, PLR-MaxMC, SFL} × 2 view sizes, 30,000 updates each, 16 log-spaced checkpoints, BFS regret on 2,000 mazes, about 32 target episodes;
  - GPU time per run is unmeasured; measure it on the first run;
  - it is the plan's first public output, and less novel than the racing audit.
- **PA-4:**
  - E0 needs the asymmetric critic, which is not built;
  - add a constant-LR option and a halved-preview option;
  - resolve the chunk-length conflict: 64-step rollouts against the scorer's 128.
- **PA-5:**
  - about 1,300–1,700 CPU-h in all;
  - rent the big CPU box only after the labeller is tested, the tags exist, the label rules are frozen in `audit-v1` and E0's final checkpoint exists;
  - set a measured stop rule, for example stop above $150.
- **PA-6 and PA-7:** see the table. Filling the `\RESULT`s has no value before then.

### 7.6 Drivers and demo
- **DV-B0** simulates P(split-half ρ ≥ 0.5) from the recorded noise budgets, at ×1 and ×1.36. It sweeps τ = 0–40 ms, the raw and profile statistics, and 2 and 3 phases. It touches no new real sessions.
  - At the pilot's estimate (ρ 0.11, p 0.237), the 35 usable 2023–25 sessions probably fail the 0.5 gate. The two longest pairs reach about 0.43.
  - A pass would not prove driver skill either: persistent per-car errors also survive across halves.
- **DV-B** (if go):
  - B1, the pre-registration, 6–10 h. It fixes the sessions, phases, statistic, filters, a held-out anchor criterion, the claim wording ("teammate-label effects") and the fallback case-study rule.
  - B2, a blind QA pass, 2–4 h.
  - B3, run once, under 1.5 h of CPU.
  - B4, the report, which ships either way.
  - B5, the M1 hierarchical model, only on a pass.
- **DM-1:**
  - a render mode: ffmpeg is not installed (#2);
  - an MVP-1 lite Hockenheim clip, with the optimum tagged "not certified";
  - an R-dev generalist ghost, for example `dr-realw-loo-f2-s1` on Catalunya, badged POST HOC.
- **Later:**
  - DM-C4, human-only timing ghosts. It needs a `lockbox_guard` change (#19).
  - DM-C5, the R0 audit post. Its grid-slot control needs a new OpenF1 fetch.
  - The rest of the demo is blocked on E0, `protocol-v1` and builds the plan costs at 25–50 h each.
- **Small items:**
  - DV-D3: a clearer message for Baku 2025's empty laps table.
  - DV-D4: a per-car breakdown of the Silverstone offsets, about 1 h.
  - DV-D6: the tyre cross-check cannot be done from the archive.
  - DV-D7: the official circuit lengths are low value.

### 7.7 P4 and the lockbox
- **PS-P4, option A:** certified-irrelevant perturbations that reach d_OOD of about 2–3, run through the frozen builder's `f6_scale`.
  - Candidate families: larger κ noise on straights only, widening-only width changes, and a larger in-phase wobble where it is certified.
  - Each stimulus is certified by s-OCP (corner v_min within about 1%) and by the generalists completing it (C ≥ 0.9).
  - The classification is reproduced by a new script.
  - The likely answer for S2 is "fragile", not "memorised". S1 cannot move under any F6 design.
  - Option B2, 5 fresh specialists (about 1 h on the A6000) under a new pre-registration, is the clean prospective test (#7).
- **LB-0:**
  - `scripts/lockbox.py` with the stages `freeze`, `ocp`, `laps` and `analyse`. `analyse` runs once and refuses if its output exists.
  - It reads references only from a fresh `outputs/lockbox/`, and calls `lockbox_guard(unsealed=True)` only at the tag.
  - It needs per-corner timing, ±0.5 m width re-solves and an OOD-distance statistic.
  - A dry run with R_DEV standing in for R_TEST means no bug first surfaces after the unseal.
- **LB-1, before the tag:**
  - subjects and checkpoint sha256s, with any later run labelled POST HOC;
  - the selection rule;
  - 36 variants: 12 circuits, mirrored and reversed;
  - starts, runs per circuit and the completion unit;
  - re-solved s-OCP references under the tagged code, and a convergence fallback;
  - metrics, how crashes enter the gap, IQM with a stratified bootstrap, and calibrated predictions from the R-dev dry run.
  - Compute on the Mac: references 3–4.5 CPU-h (1–1.5 h wall), laps under 1 h for about 20 agents.
  - **Likely outcome with today's agents:** H1 (≥ 90% of laps, median gap ≤ 4%) would be refuted on both counts. Best R-dev means are 7.6–8.5 of 12, with gaps of 6.7–7.3%.
- **LB-opt:**
  - TO-OCP lap mode, for a certified G_20;
  - the fixed-path OCP, for the line/execution split, about 25 h by the plan.

  Build them only if `protocol-v1` declares them (#15).

### 7.8 Interpretability leftovers no workstream scoped (all exploratory)
- **IN-1.** Step 2, Section 10.2: the pre-registered pair weighting and CIs for the anchored-key G. That summary is empty today; only per-key medians exist.
- **IN-2.** Step 2, Section 10.3: H2′ on the A2 keys valid for all nine generalists. It checks how much of G3's lawful verdict comes from its own exclusion set. H4R's common-key check is a different analysis.
- **IN-3.** Step 1's "Not covered here": P2d, the P3 ratio, P6d, P8-F and the per-checkpoint P6 curves, all from `decisions_all.json`. Write-up only.
- **IN-4.** H4R Section 10.3: the lap gap on circuits both arms complete, from existing evaluations. It feeds GW-4's lap-time criterion.
- **Step 1b's two-corner stimuli** (protocol 9.6) were deprioritised by a logged deviation, because the ladders pointed to width, not a neighbouring corner. Not proposed.

## 8. Items with low value that could be dropped (none is dropped here)

| Item | Why the value is low |
|---|---|
| CS-2's accel share and C1 on D | vx is not scored. The held forms equal the primary ones to about 1e-14 |
| CS-13 C2-pairs | It qualifies a C2 confirm, and C2 never confirmed |
| CS-15 (FWL, actor h1/h2, C2 on D) | Machinery and completeness checks only |
| CS-12 full variant (~4.5 h) | Mostly more inconclusive results; the primary-only run carries the cross-network C2-sw pattern |
| CS-6 C1-dev | Every final R²_c is ≤ 0.025, so only the rank trend is likely to be informative |
| BC-1 at full size | 150–380 CPU-h for a likely refute that extends, not changes, the conclusion. Decide after BC-a0 |
| More leave-out seeds on their own | Low value for the generator decision unless run as GW-3's comparison arm |
| GW-7 SimBa | No evidence in the repo, and the interp tooling assumes MLP agents |
| Hairpin tilt and curriculum levers | v1/v2/v3 do not differ (5.4/5.8/5.8). E0 is DR by definition, and the PLR⊥ pilot scored 2 against DR's 4 |
| PS-P4 | The likely answer, "fragile", changes how the demo's specialist beat reads, not whether it happens |
| PA-4's E0-const and E0-short | The plan's own cut candidates |
| The PLR/SFL throughput pilot before G1 | It only measures throughput and output size |
| Drivers: jolpica external validity, sprint qualifying, DV-D3/D6/D7 | Phase deltas sum exactly to lap deltas; SOFT tyres appear only in SQ3; the data gaps cannot be fixed or do not matter |
| IN-3 | Descriptive write-ups of outputs nobody is waiting on |
| X-3 | Low to moderate; useful only if X-1 or X-2 is acted on |

VG is not low value as interpretation: it is the only causal test. Its expected effect on driving performance is low for its cost, though.

Not in this plan:
- the outputs list O1–O11 in your notes, except where the demo touches it;
- the architecture ladder in `ApexZero-plan.md` Section 8;
- big-box stage parallelism, dropped on 24 Oct.

Say if any of these should be scoped.

## 9. For the follow-up discussion: the suggestive exploratory results

### 9.1 Leave-circuits-out width transfer (p 0.063)
- **What it found** (`width-test.md` Section 7), on held-out circuits (of 6):
  - leave-out runs 3.62, v3 baseline 2.90, in-sample 4.25;
  - one-sided p 0.063 with 4 seeds per fold;
  - by fold: fold 1's circuits gained fully, fold 2's (Norisring, Sochi, Melbourne) little.
- **New and post hoc.** I checked this arithmetic against `h4r.md` Section 6.1's per-circuit table. It is descriptive only.
  - H4R's v3 baselines (seeds 10–14) have the same training code as the leave-out runs. The generator report checked that the `git diff` over the training paths is empty, and both ran on an A6000.
  - On the leave-out basis they score 4.8 on fold 1 and 2.4 on fold 2, a mean of 3.6. That equals the leave-out runs' 3.62.
  - Pooling all 10 v3 baselines gives 3.25, a gap of +0.37.
  - H4R's in-sample v3 real-width arm scores 3.9 on the same basis, so in v3 there was little width effect to transfer (H4R S1: +0.6 of 12, p 0.32).
  - In v1 the in-sample effect replicates: +1.0 and +1.6 per fold (S1: +2.6 of 12, per-variant p 0.044). No leave-out runs exist in v1.
- **Reading.**
  - The test's baseline came from an older batch: v3 seeds 0–4 averaged 5.8 of 12, against H4R's 7.2 for seeds 10–14. That batch difference is as large as the effect.
  - STATUS's "about half the gain transfers" is therefore weaker than it reads for v3.
  - The question stays open, and v1 is where it can be answered.
- **What would settle it:** GW-3 in v1 with all arms in one queue, about 7 h on the A6000, pre-registered.
  - It also tests the statistical model, which is the generator-v1 decision itself. So this is not a side study: it sits on the critical path.
  - Another exploratory leave-out round would add little.

### 9.2 Locality of edge tracking
- **What it found** (`locality.md`):
  - Gain falls as the inside-edge change becomes more local, in all 29 networks: about 0.095 at a 2 m bump, 0.166 at 4 m and 0.305 at 8 m, against 0.760 for step 2's corner-wide change.
  - The line trails the edge by about 10 m.
  - Real-width and baseline networks do not differ at short scales (E4 not borne out, confounded by code and hardware).
  - Locality does not take G2 as low as its real-road 0.07.
- **Why it matters for the follow-through.** It is the best current mechanism for the inside-edge crashes, and it names the outcome a width-training change should target: short-scale L/e, not corner-wide G. That bears directly on GW-5's local-perturbation lever and on GW-2's corner-coupled dip.
- **Cheap next steps, all exploratory with a plan first:**
  - **X-1.** The same profiles on H4R's 20 same-code networks, from impulse responses already on disk (`outputs/h4r/fields/arrays/`). No network is run. It tests E4 without the code and hardware confound. Best information per hour.
  - **X-2.** The real crash profiles of G2's three crash units, with new VJPs at 12 unit-phases. It asks whether locality reproduces the 0.07.
  - **X-3.** Finite-amplitude closed-loop checks of a 4 m bump and a 5 m ramp, on R-dev roads only.
- **Question for the discussion.** Should X-1 run before GW-5? If real-width training already leaves short-scale following unchanged on same-code networks, local perturbations are the lever to test. If it does change it, width realism alone may be enough.

## 10. Decisions only you can make

**Spend and permissions**
1. **Spend beyond the cheapest A6000.**
   - A big CPU box for PA-5 (about $40–85, under the conditions in Section 7.5).
   - Optionally a 64-vCPU box for BC-1.
   - Nothing else needs a bigger box. Brev allows one instance at a time.
2. **Downloads and installs:**
   - the SFL source at its pinned commit (PA-3a);
   - ffmpeg or Puppeteer (DM-1);
   - new OpenF1 fetches (DM-C5, DV-D3).
3. **HY-1:** move the three stray files into `outputs/sealed/` and add a dated deviation addendum.

**Scope**

4. **Which paused tracks to un-pause now:** the paper, the drivers study (after DV-B0) and the demo. Also whether Idea 2's follow-ups (CS, BC, VG) go ahead at all, given that Idea 2 was meant to be the last study.
5. **VG:**
   - whether to do it;
   - the loss form and Ω, given the per-component outcome against an action-projected error;
   - 20 or 40 runs, and completion primary or secondary;
   - whether a positive result may change E0's recipe. That would put VG before GW-4.
6. **BC:**
   - whether to run BC-a0;
   - afterwards, the target option, one grammar or one per variant, M, and Mac or box.

**Pre-registration**

7. **Which new studies get a pre-registration and tag.** Recommended for GW-3, VG-1's main run, DV-B, PA-3, and BC-1 if it is to decide anything. P4: option A, exploratory, or B2, confirmatory with fresh specialists.
8. **From now on, no exploratory comparison of critic gradients against λ on braking or corner rows.** This keeps BC-1's confirmatory option open. Or give that option up.
9. **The costate freeze.**
   - Recommended: keep `src/` frozen for lane A, through a worktree or byte checks.
   - The alternative: edit `src/` and log a deviation.

**The generator and the 80% gate**

10. **The gate's operational definition:**
    - 10 of 12 per seed, or on the median seed;
    - read at 1B;
    - whether an R-dev-selected checkpoint counts, which plan Section 4 allows.
11. **How far to push before freezing.**
    - Suggested budget: GW-1, GW-2, one GW-3 round, at most one GW-5 lever round and one GW-6 pilot.
    - Then freeze `generator-v1` whatever the result, and use the plan's written fallback.
12. **Whether `generator-v1` may copy R-dev width profiles.** Every R-dev number would then be in-sample.
13. **Width-model details:**
    - (2a), (2b) or both;
    - whether to refit the curvature grammar on the refined geometry;
    - v1, v3 or both, and seeds per cell;
    - whether a training margin is part of `sim-v1`;
    - whether SimBa is in scope;
    - whether the paper's level sets adopt the new model.

**The lockbox**

14. **When to spend it.** Recommended: only after `generator-v1`, `sim-v1` and E0 are final and P4 is settled. Also decide:
    - whether to decouple `protocol-v1` from G1 and `audit-v1`. The plan ties them, which puts the unseal around Oct 2027;
    - whether to go ahead if the gate is not met;
    - whether to split R_TEST 6 + 6.
15. **What it declares:**
    - subjects: E0 5 × 1B, recommended, or existing 200M networks;
    - the asymmetric critic or E0-sym only;
    - the gap reference: s-OCP, or a certified G_20, which needs LB-opt;
    - whether the line/execution split is declared or descriptive;
    - H1 as written, or calibrated predictions;
    - 10 or 20 runs per circuit, and whether to add a context sweep;
    - whether to open `outputs/sealed/` afterwards;
    - whether to declare the scaling curve.

**Smaller choices (these can wait until the item starts)**

16. **Costate:**
    - C3-int's vx pairs;
    - the 6 K0 keys: exclude (recommended) or re-polish;
    - the optional new solves: K2b on N and K2b for q_c, about 53 min each;
    - C2-sw null scaling: G1 only or per network;
    - CS-12: primary only or full, and whether to include the specialists.
17. **Paper:**
    - the labeller's retry policy and the `audit-v1` level root;
    - the G1 reliability metric and the S2 reference;
    - the E0 cuts and observation noise;
    - SFL's N/M/T, min-fill, mode F and chunk length;
    - E5 now, or inside `protocol-v1`, and whether to add PLR-PVL;
    - before `protocol-v1`: the Holm family, RC@10%, the admission quantile, the thresholds, and geometric difficulty.
18. **Drivers:**
    - 2 or 3 phases. The two-track plan says 2; the code uses 3;
    - raw or profile statistic;
    - blind behind a pre-registration (recommended) or open;
    - whether to include sprint qualifying and 2026;
    - whether to keep `layout_matches`.
19. **Demo:**
    - posting publicly before arXiv;
    - a `lockbox_guard` change that allows human-only exports on R-test circuits;
    - licensing (TUMFTM LGPL-3.0, OpenF1 CC BY-NC-SA).

## 11. Decisions taken (25 Oct 2026)

- **#3, HY-1:** seal the three stray lockbox files, unread, and log a dated deviation addendum.
  Done: they are in `outputs/sealed/stray-2026-10-25/`; STATUS's Monza and lockbox-window
  figures are replaced; `scripts/solve_all_ocp.py` defaults to R_DEV and refuses lockbox circuits.
- **#4, scope:** follow through on every track: Idea 2's leftovers and new studies (CS, BC, VG),
  the generator and gate path, the drivers study, the demo and the paper.
- **#14, the lockbox:** spend it after the final agent (`generator-v1`, `sim-v1` and E0 fixed,
  P4 settled), decoupled from the paper's `audit-v1`.
- **#11, the 80% gate:** a bounded budget (GW-1, GW-2, one GW-3 round, at most one GW-5 lever
  round and one GW-6 pilot), then freeze `generator-v1` whatever the result.
- Still to ask when the item starts (updated 27 Oct 2026, after the third round; on 25 Oct this
  listed #1, #2, #5-#10, #12, #13 and #15-#19):
  - spend beyond the cheapest A6000 (#1: a big CPU box for PA-5, optionally a 64-vCPU box for
    BC-1), and maze E5's full queue on its measured estimate;
  - installs and downloads not yet approved (#2): anything the vendored SFL code needs, ffmpeg or
    Puppeteer, new OpenF1 fetches;
  - BC-1's choices after BC-a0 (#6);
  - #13's last two parts: whether to refit the curvature grammar on the refined geometry, and
    whether the paper's level sets adopt the new model;
  - what `protocol-v1` declares (#15);
  - #17's remaining parts: the `audit-v1` level root and PA-1's replicate design (at the label
    pilot's pre-registration), G1's reliability metric and S2 reference, E0's cuts and observation
    noise, the racing curriculum's SFL settings (N/M/T, min-fill), mode F and the chunk length,
    and the items due before `protocol-v1`;
  - the demo (#19);
  - to confirm, because this section does not record them as your decisions: #8, #9 and #16,
    which lane A's plan (`docs/findings/costate-followups-plan.md`, on its branch) has applied
    since 25 Oct (#8's rule for braking and corner rows, #9's recommended route, and its own choices
    for #16's items, of which Section 10 recommends only excluding the 6 K0 keys); and DV-B's
    no-go, reached by its plan's committed rule (`docs/findings/drivers-power.md`), which leaves
    #18 unneeded unless DV-B goes ahead.

**Decisions taken (25 Oct 2026, second round, after Phase 1)**
- **#10, the 80% gate:** E0 (5 x 1B) passes if its median seed completes at least 10 of 12 R-dev
  circuits at its final checkpoint. No checkpoint is chosen on R-dev; GW-1 showed that R-dev
  completion peaks at slow early checkpoints.
- **#7, P4:** option A was vacuous (`docs/findings/p4.md`), so P4 is settled by option B2: 5 fresh
  specialists under a new pre-registration, before `protocol-v1`.
- **#5, VG:** the manipulation (C1's R^2_c) is primary and completion secondary; 40 runs; a
  positive result does not change E0's recipe, so VG stays off the lockbox's critical path.
- **#13, the width model:** build (2a) first and (2b) only if (2a) misses the width-report bands;
  GW-3 runs v1 and v3 at 5 seeds per cell (about 14 A6000 h); the training margin is tested only
  in GW-5; SimBa is out of scope.

**Decisions taken (27 Oct 2026, third round, after Phase 2)**

The detail is in the four drafts: `docs/prereg/gw3-width-round.md`, `vg1-value-gradient.md`,
`p4-b2-specialists.md` and `maze-e5.md`.
- **#13, GW-3's width model and GW-4's criterion 4:** M = (2a). Criterion 4 becomes: every v1
  check in (2a)'s committed width report (`outputs/width-report/report.json`, sha256
  `5871664be5a6ff6ac882e3a38e131660fe16676f37ddd0a99ffba7294a42f014`) passes in both folds, and the
  v3 misses are limited to `share_1m` (both folds) and `spectrum:8-16` (fold f1), with every
  no-copy check still required (the lead kept it from the drafted criterion). This deviates
  from the bands in `docs/findings/width-model-plan.md`, so it is logged as a dated deviation
  (27 Oct 2026), with its reason, before the GW-3 tag. The reasons in `docs/findings/width-model.md`
  are post hoc readings and stay labelled post hoc (`gw3-width-round.md` Section 10).
- **#13, seeds per cell:** GW-3 runs 10 seeds per cell, seeds 20-29: 6 arms x 2 variants x 10 seeds
  = 120 runs, about 14 A6000 h (120 x 6.4-6.65 min per 200M run at 4 lanes is 12.8-13.3 h, plus
  setup and pull-back). This supersedes the second round's 5 seeds per cell; #13's "about 14 A6000
  h" now matches. The draft's power table is recomputed at 10 seeds by its own simulation method
  (H4R baselines: v1 5.0 ± 1.58, v3 7.2 ± 1.10), after first checking that the method reproduces
  the 5-seed figures (0.03, 0.25, 0.37, 0.70) within simulation error (`gw3-width-round.md`
  Section 3).
- **E0's curvature variant** (no Section 10 number; `gw3-width-round.md` Section 9, item 5): v1,
  `train_dr.py`'s default, fixed now, before GW-3's tag and without GW-3 data.
- **#12, copied widths as `generator-v1`'s fallback:** used if and only if (2a) fails GW-4's
  adoption rule and GW-3 shows that copied widths transfer to held-out circuits, by S1-v1, a test
  pre-specified in `gw3-width-round.md` (Section 5). If copied widths are used, every R-dev number
  for E0, the 80% gate (#10) included, is labelled in-sample for widths wherever it is reported.
  If (2a) fails and S1-v1 does not confirm, `generator-v1` uses the default widths. The lockbox
  (R_TEST) is unaffected either way. The lead wrote the rule as deterministic, so nothing is left
  to choose after GW-3's data.
- **#5, VG-1's loss form and Ω:** the GDHP-style squared error with full-state diagonal Ω, equal
  over the 12 standardised components of y = (x, prev_action, accel), as built in VG-0. Not the
  action-projected Ω = BB^T (`vg1-value-gradient.md`).
- **#2, the SFL download for maze E5 (PA-3a):** approved. From
  `github.com/amacrutherford/sampling-for-learnability` at commit
  `8b14e2ec07b38dc33a9df476b84faa2f3ca5fd3b`, the files and git blob SHA-1s listed in `maze-e5.md`
  (46,359 B, or 60,890 B with `sfl/train/common/network.py`), Apache-2.0, vendored with its
  LICENSE and each blob checked. Not downloaded yet; it happens when E5 is built. Any install
  (Hydra, OmegaConf or anything else) needs a separate OK from you.
- **Phase 3's A6000 budget (on the cheapest box; #1 stays open):** GW-3 (about 14 h), VG-1 (8-14 h)
  and P4-B2 (about 1 h) are approved, about 23-29 box-hours in all, on the cheapest Massed Compute
  A6000, one box at a time, deleted when done. The lead checks the hourly price on Brev before
  creating the box.
  - Stop rule (the lead's): if a round's projected box time exceeds its approved estimate by more
    than 20%, the lead pauses that round and asks you.
  - VG-1's cost check now returns to you if the pilot projects the pilot plus the main run above
    14 h (it was 11 h).
  - Maze E5: only a short timing run on the box is approved now. Its full queue comes back to you
    with the measured estimate.
- **The smaller choices: the drafts' defaults are accepted.**
  - **#5, VG-1** (`vg1-value-gradient.md` Section 10): P is an exact one-sided Wilcoxon rank-sum on
    C1's R^2_c, the single primary, with C1's Spearman ρ as secondary S2 (no Holm family); the
    pilot's β grid {0.1, 0.3, 1.0} with its eligibility rule (finite metrics, EV ≥ 0.90,
    M0 < 13.99); C1-own stays optional and descriptive. The lead's decision (engineering): the VG
    settings stay in a separate `ValueGradient` object, not `PPOConfig` fields as Section 7.3 has
    it, because new `PPOConfig` fields would break frozen `scripts/h4r.py`'s config comparison.
  - **#7, P4-B2** (`p4-b2-specialists.md` Section 11), all five as drafted: the corner-matched
    reading is included and decides only after a vacuous class 2; 3 of 5 specialists to claim a
    reading; the registered F6 conditions stay out of the CM comparators; seeds 4-8, run names as
    drafted, 5 lanes; no new certified family.
  - **#17 and #2, maze E5** (`maze-e5.md` Section 11): its own tag, `maze-e5-prereg-v1`, not folded
    into `protocol-v1`; no PLR-PVL arm; DR through `maze_plr.py` with replay off as drafted
    (`maze_dr.py` is not downloaded).
  - **#17, PA-1's labeller** (`src/apexzero/ocp/labels.py`): the retry policy and budgets as
    drafted (900 s per start, 600 s per IPOPT call, max_iter 3,000; on failure one retry per start
    with stages every 10 s, max_iter 6,000, 1,800 s); the 200-label pilot's agent start uses
    `runs/dr-pilot-200M` at 200M. Still open, decided when the pilot is pre-registered: the
    `audit-v1` level root (disjoint from PA-2's levels) and the replicate design (3 extra jittered
    starts).

The lead's notes (27 Oct 2026):
- **Phase 2 is done.** GW-2 found that neither width model meets the width-report bands
  (`docs/findings/width-model.md`). VG-0, LB-0 and PA-1's labeller are built; PA-1's 200-label
  pilot waits for its pre-registration. The four drafts above are written and reviewed, and the
  review's 16 findings are fixed. Off the critical path, lane A restarts from CS-9.
- **The full R-dev lockbox dry run** is left to LB-1 step 1 in Phase 5 (Section 4). LB-0 built the
  harness and its tests.
- **LB-1 must declare how uncompleted circuits enter the gap** (Section 7.7; #15).
- **Not in the approved A6000 budget:** GW-5 and GW-6 (#11's optional rounds), E0 (5 x 1B) and
  PA-4, PA-6 and PA-7. Each comes back to you with its estimate before it runs.
- **Still open from #14:** whether to go ahead with the lockbox if E0 misses the 80% gate, and
  whether to split R_TEST 6 + 6. They are asked before `protocol-v1` is written.
- **Lockbox incident (27 Oct 2026):** while recording these decisions, one agent's recursive
  search for "gw3|GW-3" over `outputs/` traversed `outputs/sealed/`. It printed nothing: its
  output was filtered to drop any sealed path and it was stopped with empty output. Logged in
  `docs/STATUS.md`.

**Decision taken (27 Oct 2026, fourth round): more boxes.**
- You allowed more than one box at a time ("you can use even more boxes if you want"). The
  approved GPU time is unchanged: about 23-29 A6000 GPU-hours for GW-3, VG-1 and P4-B2, with the
  20% stop rule counted in GPU-hours, idle GPUs included.
- Brev's prices, checked 27 Oct 2026: `massedcompute_A6000_plus` $0.68/h (1 GPU, 12 vCPU),
  `massedcompute_A6000_plusx2` $1.37/h and `massedcompute_A6000_plusx4` $2.74/h (2 or 4 GPUs,
  the same A6000 and the same price per GPU). About $16-20 for the approved rounds.
- The lead's plan: one `massedcompute_A6000_plusx4` box (the account has allowed one instance at a
  time), each run on one GPU at 4 lanes per GPU; or several single-GPU boxes of the same type, each
  taking whole seeds, if the x4 type is unavailable. The queue scripts work on any number of GPUs
  and boxes, and each round's manifest records the box type, GPU count and shard.


**Decisions taken (27 Oct 2026, fifth round): P4-B2 and E5 details.**
- P4-B2: the shared queue's immediate retry in the same lane is accepted. The five d_OOD buffers
  are bound by sha256 and not committed (about 182 MB kept out of the public repo).
- E5: SFL's `sfl/train/minigrid_plr.py` (38,737 bytes) and `sfl/util/jaxued/jaxued_utils.py`
  (2,942 bytes, with two empty `__init__.py` files) downloaded, each checked against its blob
  hash at commit `8b14e2ec`. No install was needed. Section 11 item 11: a score faithful in every
  reached phase that also shifts is row A, not B or B'.

The lead's notes (27 Oct 2026, launch):
- `gw3-prereg-v1` and `p4-b2-train-v1` are tagged on `860939a`, which is also VG-1's pilot
  commit. VG-1's main run uses 5 lanes per GPU on the x4 box if a memory check on the pilot box
  passes, so every lane runs one run of each arm and the modelled cost tail of 4 lanes does not
  arise (VG-1 draft, Section 10, item 9).
- The account allowed two instances at once: one `massedcompute_A6000_plusx4` box for GW-3 (then
  VG-1's main run after its tag) and one `massedcompute_A6000_plus` box for the memory check, the
  VG-1 pilot and then P4-B2.

The lead's notes (28 Oct 2026, Phase 3 done):
- **GW-3** (`docs/findings/gw3.md`): P1 confirms (v1 9.8 against 4.6, v3 9.9 against 5.0 held-out
  circuits of 12; p = 2.9e-11). GW-4's criterion 3 fails (Δgap +1.52 points), so by #12
  `generator-v1` uses copied R-dev widths, and every R-dev number for E0 is labelled "in-sample
  for widths". Box 14.7 GPU-hours.
- **P4-B2** (`docs/findings/p4-b2.md`): no recipe-level description; no reading reaches 3 of the
  5 specialists (one mixed, two undecided, one fragile, one context-keyed). Box 2.06 GPU-hours,
  1.74 of them idle while the lead waited on an answer about VG-1.
- **VG-1** (`docs/findings/vg1.md`): P confirms (C1's R²_c, p = 0.0105, Hodges–Lehmann +0.35);
  reading row 2. S1 shows no completion difference (−0.5 circuits, CI −1.2 to +0.2), so "fix the
  critic's gradient" is closed as a completion lever at 200M, v1. E0 is unchanged. A second pilot
  at smaller β (Amendment 1, the user's choice) was needed; 10.31 GPU-hours in all.
- **Phase 3's A6000 time:** 27.1 GPU-hours (14.7 + 2.06 + 10.31) against the approved 23–29.
- **Still to come from Phase 3:** maze E5's timing run (PA-3), after a review of the changes made
  since its build review (the vendored network and Section 11 item 11) and its dry run on the
  pilot's checkpoints.

The lead's notes (28 Oct 2026, lane A and E5's calibration):
- **Lane A** (`docs/findings/costate-followups.md`, merged into main as `0bbf966`): CS-1 to CS-15
  and BC-a0 ran, except CS-13 on T; all exploratory. The C1 refute holds at the policies' own
  states, on N and at all 144 training snapshots. λ is decodable from z (R² 0.35 for λ_s, 0.67 for
  λ_ξ), but the critics' input gradients do not express it. BC-a0: road beyond the preview moves
  braking and corner λ by 9–103% of its phase variance, so BC-1 would need the preview-matched
  average target (option (ii)), at about 2.3–2.9 times the planned 150–380 CPU-h. The note's
  Section 6 lists what is not done and the questions for the lead.
- **E5's calibration** (`docs/prereg/maze-e5.md` Section 8 step 5): 1.74 GPU-h on one A6000. The
  rule projects 87.6 GPU-h on an x4 box and 106.8 on five single-GPU boxes. The x4 figure is too
  low, because the load changed within the calibration runs; under the measured fair-share model
  the x4 box needs about 170 GPU-h and the single-GPU boxes about 108. The GPU capture check is
  inconclusive (GPU training is not bit-deterministic). The full queue goes to the user at about
  108 GPU-h, with the drop levels.
- **Phase 3's A6000 time** is now 28.8 GPU-hours (27.1 + 1.74).

The user's decisions (28 Oct 2026, sixth round):
- **E0** (`docs/plan/e0.md`): the budget is approved, on one `massedcompute_A6000_plus` at 5 lanes,
  about 2.8–3.2 GPU-h; the stop rule's threshold is 3.84 GPU-h. Tag `e0-train-v1`, then the run.
- **E5** (`docs/prereg/maze-e5.md` Section 8 step 5): the full queue (drop 0) is approved on five
  single-GPU boxes at 8 lanes per GPU, about 108 GPU-h with the audit; the stop rule's threshold
  is 130 GPU-h. Tag `maze-e5-train-v1`, then the run.
- **BC-1 is dropped** (#6). BC-a0 showed that road beyond the preview moves braking and corner λ
  too much for the existing λ to be its target. #8's ban, which kept BC-1's confirmatory option
  open, is lifted: critic gradients may now be compared with λ on braking and corner rows,
  labelled exploratory.
- **Lane A:** #8, #9 and #16 are confirmed as lane A ran them, so its results stand. CS-13 on T is
  dropped (Section 8 allows it).

