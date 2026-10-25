# P4 made testable (PS-P4, option A): plan (exploratory; written before any computation, 25 Oct 2026)

**Status.** This is item PS-P4 of the follow-through plan, option A ([`docs/plan/follow-through.md`](../plan/follow-through.md), Section 7.7). It is exploratory, and it comes after step 1's one confirmatory run. It cannot change step 1's confirmatory P4 verdicts (S1 class 6; S2 and S3 "not testable"; the recipe claim "inconclusive"), nor the post-tag rerun's vacuous class 2 for S2 and S3 ([psychophysics-step1.md](psychophysics-step1.md), Sections 4.1 and 4.4).

Option B2 is not part of this plan. It would train 5 fresh specialists under a new pre-registration, about 1 h on the A6000, and it remains the clean prospective test. Your choice between A and B2 (#7) stays open.

This plan is committed before anything below is computed, so the stimuli, certification rules, classification and expected directions are fixed in advance. Later changes are dated amendments (Section 10). Only Hockenheim, an R-dev circuit, is used. R_TEST stays sealed.

## 1. Question

P4 asks whether a Hockenheim specialist reads the road or has memorised Hockenheim. Its class 2, "context-keyed memorisation", has two requirements:
- a single-factor F1 condition fails (C ≤ 0.5);
- every F6 irrelevant perturbation at least as far off-manifold (d_OOD) as that condition completes (C ≥ 0.8).

Class 1, "off-manifold fragile", holds instead when some F6 perturbation at or below that d_OOD also fails (protocol, P4).

The registered F6 perturbations reach d_OOD of only 0.7–1.0, against 2.0–2.75 for S2's and S3's failing conditions. So S2's and S3's class 2 held vacuously, with no F6 perturbation at or above the failing d_OOD. Memorisation and fragility could not be told apart (ruling R10; step 1, Section 3 item 3).

Option A builds perturbations that are certified irrelevant and that reach d_OOD of about 2–3, then reapplies P4's ordered rules with them.

## 2. What step 1 left

Source: `outputs/psychophysics/analysis/decisions_all.json`, P4 rows, sha256 `14c674863e78b68cf01986bddcaec34480700ea21811aa0971550711feca4ea1`. All three specialists have 9 braking corners of 20.

| Specialist | Class in the post-tag rerun | Single-factor F1 conditions at C ≤ 0.5 (C, d_OOD) | Registered F6 (C; d_OOD) |
|---|---|---|---|
| S1 | 6, mixed | none; the lowest is constant width (0.56, 2.64) | 0.89–1.0; 0.69–1.01 |
| S2 | 2, vacuous | road after straightened (0.22, 2.37); constant width (0.22, 2.01) | 0.67–1.0; 0.74–1.02. Wobble 0.6 (0.3 m as built): C 0.67 at 1.02 |
| S3 | 2, vacuous | constant width (0.44, 2.75) | 0.89–1.0; 0.72–0.97 |

- **Condition (a)** has C = 1.0 and d_OOD 0.68–0.72 for all three specialists.
- **S2 and S3 both crash on a tier G null straight.** The protocol's tier G rule reads that as evidence for class 1 on synthetic stimuli.
- **S1 cannot move under any F6 design.**
  - F6 enters only classes 1 and 2, and both need a single-factor F1 condition at C ≤ 0.5, which S1 does not have.
  - Classes 3–5 need all eight F1 conditions at C ≥ 0.8, which S1 also fails.
  - S1 is carried through as a check only.

## 3. Stimuli: the candidate families (fixed now)

Every new condition is a tier D condition. It uses the same 20 Hockenheim corners, strips, landmarks and starts as condition (a), and it changes the road only outside the 100 m real lead-in, as F6 does.

**F6s: the frozen builder at larger amplitude.**
- The family uses `interp.stimuli.hockenheim_conditions(i, track, swap_j, f6_scale)`, called read-only. Its F6 rows are kept.
- The scales are f6_scale ∈ {1, 2, 3, 4, 6, 8}. Step 1 used 0.5, after ruling R27's halving.
- Each scale gives the builder's four conditions:
  - the in-phase wobble, at 0.3·s and 0.6·s m (60 m wavelength; both edges move outward and inward together);
  - band-limited κ noise, at peak 0.5e-3·s and 1e-3·s m⁻¹ (wavelengths 80–200 m, zero net heading);
  - all drawn on the frozen `f6` stream.
- That makes 24 conditions, with wobble amplitudes 0.3–4.8 m and κ noise 0.5e-3 to 8e-3 m⁻¹.
- The wobble members are the "larger in-phase wobble"; certification decides where it applies.

**K-str: κ noise on straights only** (new code that composes the frozen builder's pieces).
- **Straights** are the knots of the condition (a) strip outside every `tracks.corners.segment` corner, after each corner is extended by 50 m on both sides, and after the lead-in.
- **The noise.** The frozen `_kappa_noise`, read-only, is applied separately on each maximal straight run longer than 200 m (one wavelength at the band's long end). Each run gets its own zero-net-heading window and 20 m tapers.
- **Levels:** peak 1e-3, 2e-3, 4e-3 and 8e-3 m⁻¹. That makes 4 conditions.

**W+: widening only** (new code).
- w_left and w_right both increase by A·max(0, sin(2πs/60 + φ))·(1 − w_lead): the frozen wobble's footprint and wavelength, with its narrowing half removed.
- The frozen `apply_inside_cap` is then applied to the modified knots.
- **Amplitudes:** A ∈ {0.5, 1, 2, 3, 4} m. That makes 5 conditions.
- **Contingency** (fixed now, decided on certification alone, before any specialist run): if W+ is certified at fewer than 7 of the 9 braking corners at every A ≥ 1 m, it is rebuilt on K-str's straights footprint.

**Random draws for K-str and W+.** They use a new stream, `np.random.default_rng([ROOT_SEED, offset, corner, family, level])`. The offset is fixed in the script and differs from every frozen stream offset, and the frozen `f6` stream's draws are never reused for them.

**Built-in check.** Rebuilt at f6_scale 0.5, the frozen builder's F6 strips must equal step 1's stored ones (`outputs/psychophysics/stimuli/bank.npz`) bit for bit. A failure stops the item for an amendment.

In all: 33 conditions (38 if the contingency applies), each on 20 corners. Only the 9 braking corners are scored.

## 4. Certification (before any specialist sees a new condition)

**1. s-OCP: the corner's optimum is unchanged.**
- For each braking corner i and each condition, the pinned s-OCP is solved from the tier D start that step 1's keys use: S1's real-lap state at that corner, the start every generalist's tier D runs share (`analysis.cond_table`).
- Step 1's frozen key code (`interp.answer_key`) is used read-only.
- v*_min,i is the s-OCP's minimum speed over corner i's region [s_c0 − 50, s_c1 + 50], as tier D's `v_min_star` is defined.
- **Certified at corner i** when both hold:
  - the solve converges and passes its checks;
  - |v*_min,i(cond) / v*_min,i(a) − 1| ≤ 0.01.
- Condition (a) is re-solved with the same code. Each value must match the stored tier D key to a relative 1e-6 (a built-in check).

**2. The generalists complete it.**
- The 9 step-1 generalist finals run each condition on its certified corners from S1's state, with 4 deterministic phases (tier D's protocol). C_G is P4's C: the share of corners where at least 3 of 4 phase runs complete, completion meaning no termination before s_c1 + 250 m.
- A condition is certified if the median of the nine C_G is at least 0.9.
- **Fallback** (fixed now, read on condition (a) before any new condition is run): if the generalists' median C_G(a) is itself below 0.9, the criterion becomes median C_G(cond) ≥ median C_G(a) − 0.1.

**3. Corner count.** A condition enters P4 only if it is certified at 7 or more of the 9 braking corners. Its C and d_OOD are then taken over its certified corners.

**4. Freeze.** The certified list is written to `outputs/p4/certified.json` and committed before any specialist is rolled out on a new condition. It holds:
- every condition;
- its certified corners;
- the sha256 of its strips.

No specialist output on a new condition exists until that commit.

## 5. Specialist runs and d_OOD

**Runs.**
- S1, S2 and S3 run every certified condition on its certified corners, each from its own tier D states, with 4 deterministic phases.
- They use the frozen rollout code (`interp.rollout`), read-only. C is computed as in P4.

**d_OOD.**
- For each run: the median, over steps from the start to s_c + 50 m, of the normalised nearest-neighbour distance from the run's observations to that specialist's own Hockenheim buffer (`outputs/psychophysics/controls/knn_S{1,2,3}.npz`).
- It uses the frozen `interp.controls.d_ood`. Per condition, the value is the median over its runs, P4's D.

**Built-in check.** The registered conditions, (a) included, are rerun through the same path. They must reproduce `decisions_all.json`'s C exactly and its d_OOD to 1e-6, for S1–S3. A failure stops the item for an amendment.

## 6. Classification (reproduced by a new script)

**The rule.**
- The new script implements P4's ordered rules exactly: the protocol's P4 table, `interp.analysis.p4_classify` and AD16.
- The thresholds are unchanged: gate 0.9, fail 0.5, pass 0.8, main effect 0.4.
- The F6 set is extended to the four registered conditions plus every certified new condition.
- Classes 0 and 3–6 do not involve F6, so they come out as the frozen classifier gives them.

**Checks.**
- With the extended set equal to the registered four, the script must reproduce `decisions_all.json`'s P4 rows for S1–S3 exactly: class, label, failing condition, `vacuous`, `n_f6_at_or_above`, C and d_OOD.
- Its tests also compare it with the frozen `p4_classify` on planted tables, including the dry run's simulated context-keyed agent and reader (`SIM_D_OOD`).

**Reported per specialist:**
- the class and label under the extended set, and whether it is non-vacuous (`n_f6_at_or_above` ≥ 1);
- the class under each family on its own (the registered four plus that family);
- C against d_OOD for each family;
- the certified condition whose d_OOD is nearest each failing F1 condition's.

**The recipe claim.** P4's recipe rule applies: a class is claimed for the recipe only if at least 2 of the 3 seeds share it.

**Multiplicity, stated now.** More F6 conditions make class 1 easier: one near condition at C ≤ 0.5 is enough. They make class 2 harder: one far condition below 0.8 blocks it. The per-family classes show whether the answer depends on the family. No threshold is adjusted for this.

## 7. Expected directions (stated before computing)

- **E1, certification.**
  - The κ-noise members (F6s κ noise and K-str) are certified to larger amplitudes than the wobble.
  - W+ is certified at its small amplitudes and loses corners at the larger ones.
  - The F6s wobble at 1.8 m or more fails at 3 or more of the 9 corners and drops out.
- **E2, reach.**
  - d_OOD rises with amplitude in every family.
  - At least one certified condition reaches d_OOD ≥ 2.37 for S2 and ≥ 2.75 for S3. Those conditions are wobble or widening members, not κ noise.
- **E3, S2: class 1, off-manifold fragile.** Some certified condition at d_OOD ≤ 2.37 has C ≤ 0.5.
  - Basis: S2 already completes only 0.67 under a 0.3 m wobble at d_OOD 1.02, and it crashes on a null straight.
- **E4, S3: class 1 or class 6.** This is held weakly. Its registered F6 completions are all at least 0.89, but it too crashes on a null straight.
- **E5, S1: class 6, unchanged** (structural; Section 2).
- **E6, the recipe:** class 1 from 2 of 3 seeds, if E3 and E4 both land on class 1.

## 8. What each outcome means

| Outcome for S2 or S3 | Reading (protocol, interpretation table) | Consequence |
|---|---|---|
| Class 1 | Fragile off its manifold. Memorisation cannot be diagnosed with irrelevant perturbations | The demo's specialist beat and `protocol-v1` describe the specialists as fragile off Hockenheim's road, not as memorisers |
| Class 2, non-vacuous | Context-keyed memorisation (the protocol's prediction), with the key named by the F1 main effects | Its "4.8% from optimal" is a fact about Hockenheim only |
| Class 6 | Mixed: neither rule holds over the extended set | No clean label. Report it corner by corner |
| Class 2, still vacuous | No certified condition reaches the failing d_OOD, so option A cannot test P4 within certification | Only B2 can settle it |

- Each reading is exploratory and is reported as such.
- `protocol-v1` is to be written only once P4 is settled (#14). It takes its wording for the specialists from this result, unless B2 is run.

## 9. Compute, code, outputs and integrity

**Compute.** Under about 3 h of Mac wall time (est.). At most 4 processes at a time, each under `nice -n 15`, because another workflow shares the Mac.
- **s-OCP:** up to 33 × 9 solves for the new conditions plus 9 for condition (a), 306 in all (351 if the contingency applies).
  - The first 10 solves are timed.
  - If they average more than 120 s, the F6s ladder is cut to f6_scale ∈ {1, 2, 4, 8} (16 conditions) before any further solve. This is decided on time alone.
- **Rollouts:** 12 networks × at most 38 conditions × 20 corners × 4 deterministic phases. Minutes.
- **No A6000.**

**Code.**
- `scripts/p4_exploratory.py` (new), with stages `build`, `certify-socp`, `certify-generalists`, `freeze`, `specialists`, `classify` and `report`.
- `tests/test_p4_exploratory.py` (new). It covers:
  - the classifier against the frozen `p4_classify`;
  - the K-str and W+ builders: zero net heading, the footprints, and widening only;
  - the certification rules;
  - the lockbox refusal.

**Integrity.**
- `src/apexzero/interp` and `scripts/psychophysics.py` are frozen and unchanged since `psychophysics-prereg-v2`. They are imported read-only, and the script refuses to run when either differs from HEAD.
- Nothing is written under `outputs/psychophysics/`.
- No file under `src/` is added or changed: the costate freeze hash covers `src/`.

**Outputs.**
- `outputs/p4/`: strips, certification, runs, d_OOD, classes, and a provenance block with the commit, changed paths and input hashes.
- `outputs/p4/certified.json`, committed at the freeze (Section 4).
- A findings note, `docs/findings/p4.md`. Every number in it is labelled exploratory.

**Lockbox.**
- The track name passes `viz.export.lockbox_guard`, and the frozen builder's own `_guard_named` admits Hockenheim only.
- No R_TEST circuit and no mirrored or reversed variant of one is built or driven.
- `outputs/sealed/` is not read.

## 10. Amendments

Any change to this plan after its commit is a dated amendment appended here, and it is committed before the computation it changes. Each amendment states:
- what changed, and why;
- which results, if any, had been seen when it was written.

Results computed under the earlier version are kept and reported beside the amended ones. An amendment written after a result was seen is labelled post hoc.

**Extending a ladder.** If no certified condition reaches the failing F1 condition's d_OOD for a specialist, one ladder extension is allowed. It must be:
- fixed before its runs;
- within certification;
- labelled post hoc, because d_OOD values would by then be known.

### Amendment 1 (25 Oct 2026): implementation details, fixed before any computation

**What had been seen.** No new condition had been built into a bank, solved or driven. Before writing this, I read the frozen code and reran 16 stored deterministic runs of S2 on four registered cells (corner 0: (a), road after straightened, F6 wobble 0.6; corner 3: (a)), to see whether the frozen rollout code reproduces step 1's runs. Those are registered conditions with results already in `decisions_all.json`. The unit tests of `scripts/p4_exploratory.py` build corner 0's new strips as stimuli only; nothing is solved or driven on them.

**What changed, and why.** Nothing in Sections 1–8 changes. The items below fix details the plan leaves open, so that none of them is chosen after a result is seen.

1. **Batch shape.** Every rollout chunk is padded to exactly 2048 runs (step 1's `CHUNK`) with copies of its own runs, which are then dropped. XLA's float32 results depend on the batch shape. In the check above, a 16-run batch differed from the stored runs by up to 5e-4 m in s, while 2048-run batches matched them bit for bit. All of step 1's tier D runs, for all 12 agents, were made in 2048-run batches. No definition changes.
2. **d_OOD per run.** It is computed on the deterministic phase-0 run, as step 1's code does (`scripts/psychophysics.py`, `_d_ood_from_det`, imported read-only). The "median over its runs" of Section 5 is then `cond_table`'s median over the corner's runs that have a d_OOD, which is the phase-0 value, and P4's D is the median of that over the condition's corners. The reproduction check (Section 5) needs exactly this.
3. **C.** C is the frozen `cond_table`'s `complete` (at least 3 of 4 phases) on run rows computed by `measures.response_metrics`, as `analysis.run_table` computes them. Tier D cells have no null run. `completion_by_condition` then averages it over the corners present, which for a new condition are its certified corners.
4. **K-str's straights.**
   - `corners.segment` runs, with its defaults, on the float32 Track of condition (a)'s strip (what `StimulusBank.track` returns).
   - Each corner spans [s_start, s_end + 2 m], extended by 50 m on both sides.
   - Straight knots lie after the lead-in (knot > `lead1`) and no later than the excision's last knot `i_b`, the range the frozen κ noise uses.
   - A maximal run of straight knots a..b is kept when (b − a)·2 m > 200 m. It gets `_kappa_noise(rng, i, a, b, level)`: noise on the open run, 20 m tapers inside it, zero net heading over it, and peak `level`.
   - Runs are drawn in order of position from one generator per (corner, family, level). The frozen inside cap is then applied with `mask = w_lead < 1`, as the frozen κ noise does.
   - The build stops (an amendment would be needed) if K-str's footprint reaches [s_c0 − 50, s_c1 + 50] of the scored corner.
5. **W+.**
   - φ = `rng.uniform(0, 2π)` from the new stream, with s = knot index × 2 m, as the frozen wobble uses.
   - "The modified knots" for the cap are the knots where the widening is positive.
   - The build stops if the cap leaves any knot narrower than condition (a)'s road on either side.
6. **W+str, the contingency's rebuild.** W+ again, with the footprint (1 − w_lead) multiplied by K-str's window: the frozen κ noise's taper window on each straight run, 0 at the run's ends, with 20 m ramps. W+str is built with the rest. It is solved and driven only if the contingency triggers.
7. **Streams and names.**
   - The new stream is `default_rng([ROOT_SEED, STREAM_OFFSET + 401, corner, family, level])`, with family 1 for K-str, 2 for W+ and 3 for W+str, and level the position in the family's list.
   - Offset 401 differs from step 1's offsets (1–9), step 2's (101–105) and the costate's (3xx).
   - Conditions are named `P4A/F6s/<registered tail>/x<scale>`, `P4A/Kstr/<peak>`, `P4A/Wplus/<A>m` and `P4A/WplusStr/<A>m`.
8. **The timing rule (Section 9).**
   - The first 10 solves, in job order, are condition (a) on the 9 braking corners and then `P4A/F6s/wobble0.3/x1` on corner 0, run in the pool of 4.
   - The time of a solve is the wall-clock seconds of its IPOPT call in its worker (`OCPResult.solve_seconds`). CPU seconds are recorded beside it.
   - Condition (a)'s reproduction check (Section 4.1) is read on the same 9 solves, before any further solve.
9. **Generalists.**
   - The generalists run condition (a) and the isolated road (F1/straight/straight/const) on the 9 braking corners first. (a) fixes the rule or its fallback before any new condition is run. Both feed the memorisation index MI that the classifier reports, as in step 1 (G1–G3).
   - They then run every condition with at least one s-OCP-certified corner, on those corners.
   - For the contingency, a W+ member's count is its number of s-OCP-certified corners if it passes the generalists' rule, and 0 otherwise.
   - As an extra check, the reference runs are compared bit for bit with step 1's stored runs.
10. **Specialists.**
    - S1–S3 rerun all 23 registered conditions on the 9 braking corners. P4 reads all of them: classes 3–6, S_spec, φ and m.
    - Nothing on a new condition is run unless the check of Section 5 passes.
    - As an extra check, the runs are also compared bit for bit with step 1's stored runs.
11. **Inputs to the classifier.**
    - The tier G crash flag is taken from `decisions_all.json` (S1 none, S2 and S3 a crash).
    - The tier A table is not loaded. It matters only for class 4 against class 5, which need all eight F1 conditions at C ≥ 0.8, and no specialist has that on the registered conditions, which the extended set does not change. The script stops if it ever reaches class 4 or 5.
12. **How the expected directions are read** (Section 7). "Enters P4" means that the generalists pass the condition and it is certified at 7 or more corners.
    - **E1, part 1.** The κ-noise members reach a higher rung than the wobble. A rung is the f6_scale for F6s members and level / 1e-3 for K-str, so K-str at 8e-3 sits on the same rung as F6s knoise1e-3 at scale 8. E1 part 1 holds if the highest rung entering P4 among the F6s κ-noise members, and the one among the K-str members, both exceed the highest among the F6s wobble members. None entering counts as 0.
    - **E1, part 2.** W+ at A = 0.5 and 1 m enters P4, and W+ at 4 m has fewer s-OCP-certified corners than at 0.5 m.
    - **E1, part 3.** Every evaluated F6s wobble member of 1.8 m or more fails the s-OCP rule at 3 or more corners and does not enter P4.
    - **E2, part 1.** For each specialist and each sub-ladder with 2 or more members entering P4 (F6s by registered tail, K-str, W+, W+str), the Spearman ρ between amplitude and d_OOD is positive. With 2 members, the larger amplitude has the larger d_OOD. It holds if this is true for every pair of specialist and sub-ladder.
    - **E2, part 2.** At least one condition entering P4 has d_OOD ≥ 2.37 for S2 and one has d_OOD ≥ 2.75 for S3, and every such condition is a wobble or widening member.
    - **E3–E6** are read off the extended-set classes and the recipe rule as Section 7 states them.

### Amendment 2 (25 Oct 2026): the one ladder extension (POST HOC)

**What had been seen.** Everything in the planned run, so this amendment is post hoc, as Section 10's extension clause requires:
- the certification, `outputs/p4/certified.json`, committed in `7ad8db2`: 10 conditions certified;
- the specialists' C and d_OOD on those conditions;
- their classes.

No certified condition reaches the d_OOD of S2's failing F1 conditions (2.01 and 2.37) or of S3's (2.75). The highest is W+ at 1 m, with 1.34–1.53. S2 and S3 stay class 2, vacuous.

**What changed, and why.** The one ladder extension Section 10 allows: K-str continued at 16e-3, 32e-3 and 64e-3 m⁻¹, positions 4–6 of its ladder, each rung doubling as before.
- K-str is the only family certified at its top rung (8e-3 at all 9 corners), and it leaves the corner region untouched.
- F6s κ noise loses corners from scale 3 on, because its noise reaches the corner region. W+ fails the generalists from 2 m. No wobble member is certified.
- K-str's d_OOD up to 8e-3 (0.75–0.92) was known when these rungs were chosen, and so was S3's falling C on it (1.0, 1.0, 0.875, 0.78).

**Fixed before its runs.**
- **Construction.** The same footprint, noise, stream and cap as K-str (Amendment 1, items 4 and 7), with level index 4–6.
- **Separate outputs.** The extension has its own build, certification, freeze and runs, under `outputs/p4/extension/` (`--extension`).
- **Checks.** The same checks as the planned run:
  - condition (a)'s s-OCP re-solved against step 1's keys;
  - the generalists' reference runs, whose criterion must equal the planned run's;
  - the specialists' registered reruns, checked against `decisions_all.json`.
- **Certification.** Section 4, unchanged: s-OCP within 1% at 7 or more of the 9 braking corners, and the generalists' median C_G ≥ 0.9. The list, `outputs/p4/extension/certified.json`, is committed before any specialist runs an extension rung.
- **Classification.** P4's ordered rules, with the F6 set made up of the registered four, the 10 conditions certified in the planned run and every certified extension rung. Classes are also given per family as before, and the recipe rule is applied.
- **Labels.** Every extension result is labelled post hoc and reported beside the planned results, which stand as this item's result. If no extension rung is certified, the planned result stands unchanged.
