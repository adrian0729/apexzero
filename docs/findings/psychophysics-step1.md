# Network psychophysics, step 1: results (21 Oct 2026)

Protocol: [`docs/interp/psychophysics-protocol.md`](../interp/psychophysics-protocol.md) (v2, rulings R1–R31 in its Section 12), implementation map: [`docs/interp/IMPLEMENTATION.md`](../interp/IMPLEMENTATION.md). Pre-registration tag: `psychophysics-prereg-v2`.

Every number below was read from a file named next to it. Files under `outputs/` and `runs/` are local experiment outputs and are not in git. Labels used throughout:

- **Confirmatory**: a pre-registered prediction in the primary or secondary family (protocol Section 8), decided once by `analyse --confirmatory` and read from `outputs/psychophysics/analysis/decisions.json`. The primary family covers the prospective agents; the secondary family covers the pilot-informed ones.
- **Replication**: the same decisions on the seed-1 and seed-2 replicates of each generalist variant (ruling R18), from the same run, in `outputs/psychophysics/analysis/replication.json`. R18 keeps primary decisions on the seed-0 agents.
- **Post-confirmatory**: pre-registered secondary analyses, controls and exploratory analyses outside the confirmatory families. The harness decides them only in a non-confirmatory run. That run came after the confirmatory one (01:45–01:48) and wrote `outputs/psychophysics/analysis/decisions_all.json` (Section 4.5).
- **Descriptive**: numbers recomputed here from the rollout files. They decide nothing.

## 1. Summary

Step 1 showed controlled single-corner stimuli to 12 trained racing policies and scored each closed-loop response against the optimum of the agent's own training objective, solved from the same state. The stimuli were synthetic corners, real crash corners with their context removed one factor at a time, and Hockenheim corners under up to 23 conditions. The design, stimuli, pre-tag key cache and analysis code were frozen and tagged on 20 Oct at 19:15, before the protocol rollouts began. The only earlier agent runs were the pre-registered exceptions listed in Section 2. The confirmatory analysis ran once (21 Oct, 01:31–01:42).

**Primary family (confirmatory; prospective agents).**
- G3 (variant v3):
  - P1a: **confirm**. Every bulk pair is in band, the slope is 0.935, and there is no deterministic crash.
  - P1-tail: **inconclusive**. After a 310 km/h approach no tail pair overspeeds, and 29% are too slow.
  - P1b: **inconclusive**, within the blind spot of ruling R24.
  - P2: **not applicable**, because no pair overspeeds.
  - P3: **confirm**. b = 0.947, and the physics model wins on held-out RMSE.
- S2 and S3: P4 is **not testable**, because their d_OOD buffers had not been built before the run (a procedural miss, Section 7.2). The recipe claim is therefore **inconclusive**.

**Secondary family (confirmatory; pilot-informed agents).**
- G1:
  - P1a: **confirm**.
  - P1-tail: **inconclusive**. G1 is cautious, not fast.
  - P1b: **inconclusive**.
  - P2: **not applicable**.
  - P3: **confirm**.
- G2:
  - P1a: **confirm**.
  - P1-tail: **refute**, because its A1 tail pairs track the optimum. It becomes inconclusive in the pre-registered sensitivity analysis without unsettled runs.
  - P1b: **inconclusive**.
  - P2: **not applicable**.
  - P3: **inconclusive**, and not stable. b = 1.14, above the band.
- S1: P4 **class 6** ("mixed", the protocol's none-of-the-above class).
- G2's P8-tail: **refute**. G2's three real tail crashes all reproduce on the transplanted road, but the isolated corner crashes in none of them. None is a single-corner failure (0, against a refutation threshold of 0). The ladder names constant width as a necessary factor in all three, and the straightened road after the corner in one.

**Replication (R18, 9 networks).**
- P1a confirms for all three variants and all 9 networks.
- P3 confirms for v1 and v3.
- P1-tail is refuted for v2 by 2 of 3 seeds. Only 1 of 3 seeds refutes once unsettled runs are excluded.
- Every other variant verdict is inconclusive or not applicable.

**Descriptive.**
- No tail pair overspeeds after the 310 km/h approach (A1) in any of the 9 networks.
- So the failure the study was built to detect, entering tight corners too fast after a fast approach, was not seen.
- Formally, P1-tail is refuted only for G2 and variant v2. It is inconclusive for G1 and for the primary agent G3.
- The crash ladders point to the road's width profile. Real-road crashes happen at the inside edge where the inside width is changing, and constant-width versions are mostly completed. That led to a separate training test ([width-test.md](width-test.md)).

**Post-confirmatory (pre-registered, decided after the confirmatory run; Section 4.5).**
- The mirror-symmetry control P5 fails for 8 of 9 generalists and for S1. The harness mirror gate, re-run as the protocol requires, still passes, and P5 fails just the same on exactly mirrored pairs. The asymmetry is therefore reported as learned (Sections 4.5 and 7.2).
- P6a (larger corners learned first) is refuted for G1, G2 and G3.
- P7 (preview sampling) is refuted for 11 of 12 finals.
- P8-twin is refuted for G1 and G3.
- P8-bulk and P1c confirm. P1c-approach is refuted for G2.
- The same run re-decided P4 for S2 and S3 with their buffers (post-tag; Section 4.4). Both land in class 2, but vacuously, and both crash on a null straight. Memorisation and fragility therefore cannot be told apart for them.

## 2. Design in brief

**Question.** Do trained PPO racing policies read corner severity the way the optimum does, and where does the reading fail? The agents' crashes on real circuits motivated the question. Norisring, Sochi and Yas Marina defeat all 15 trained generalists ([generalist-seeds.md](generalist-seeds.md)).

**Stimuli** (`outputs/psychophysics/stimuli/stimuli.json`, `cells.json`): 762 strips (160 synthetic corners, 449 Hockenheim conditions, 120 real-corner transplants, 30 trapezoid twins, 3 null straights) and 2,036 cells, where a cell is a strip plus an initial state.
- Tier A (P1, P2): 17 braking radii from 6.0 to 87.3 m, plus 109, 171 and 300 m controls, at 90°, and the same radii up to 55.9 m at 180°. Width is 12 m and ramps are training-typical. Two approaches:
  - A1: 310 km/h from 600 m, so the corner is out of the preview at the start.
  - A2: the typical approach speed for the corner's size.
- Tier B (crash map, P3): v0 ∈ {160, 220, 280, 310} km/h × D ∈ {250, 600} m.
- Tier D (P4): 20 Hockenheim corners under up to 23 conditions, 449 strips in all. Swap A exists for 9 of the 20 corners; every other condition exists for all 20.
  - A 2×2×2 factorial: road before the corner real or straight, road after real or straight, width real or constant 12 m.
  - A corner-size dose series.
  - A mirror.
  - Corner swaps.
  - Small irrelevant perturbations (F6), at half amplitude per R27.
- Tier E1 (P8): every real crash corner cut at [s_c − 800, s_c + 400] m in 7 rungs:
  - real road;
  - road before straightened;
  - road after straightened;
  - constant width;
  - isolated (all three changes);
  - trapezoid twin;
  - mirror.
- Tier G: null straights and catch trials.
- Other tiers: C, E2, E3, O.

**Agents** (`outputs/psychophysics/manifest.json`, `agents`):

| Agent | Run | Training | Status in the study |
|---|---|---|---|
| G1 | `runs/dr-pilot-200M` | v1: base grammar | Pilot-informed (secondary family) |
| G2 | `runs/dr-pilot-v2-tilt30` | v2: 30% of strips from the tilted grammar | Pilot-informed (secondary) |
| G3 | `runs/dr-pilot-v3-tilt50a2` | v3: 50% from a (1/r_min)^2 tilt | Prospective (primary) |
| G1s1, G1s2, G2s1, G2s2, G3s1, G3s2 | `runs/dr-pilot-*-s1`, `-s2` | Seeds 1 and 2 with the seed-0 arguments (R18), GPU | Prospective (replication) |
| S1 | `runs/hockenheim-specialist-mlp-200M` | Hockenheim only | Pilot-informed (secondary) |
| S2, S3 | `runs/hockenheim-specialist-mlp-200M-s2`, `-s3` | Hockenheim only | Prospective (primary) |

Regimes are frozen per agent from 300 strips of its own training grammar (`manifest.json`, `support.regimes`). The table gives q1 / q10 of R_seg and the radius below which the curvature input clips:

| Agent | q1 / q10 (m) | Clip radius (m) |
|---|---|---|
| G1 | 12.0 / 21.3 | 8.9 |
| G2 | 11.1 / 17.7 | 7.6 |
| G3 | 10.2 / 14.5 | 6.3 |
| Specialists | 11.7 / 17.9 | 9.4–9.5 |

**Answer keys.** The reference for every claim is the training-objective OCP (TO-OCP). It is the exact optimum of the agent's own discounted reward on the environment's own dynamics, at 20 Hz, from the stimulus's initial state, and it is replay-verified in the environment (return within 1e-6, alive at every step).

| Key | What it is | What it judges |
|---|---|---|
| Nominal | The TO-OCP from the stimulus's initial state | The overspeed side of the band |
| Backed-off | Slip limit 8.5° instead of 10°, corridor 0.35 m narrower on each side | The caution side of the band |
| Phase-matched | Nominal keys solved at each control-grid phase | Every braking cell, because V7 failed (R15) |
| Re-anchored | Solved at the agent's own state, per phase | A1 and tier B 600 m approaches, so that a different approach speed is not scored as error (R17) |
| Viability | Solved from a crashed run's states before the crash | Crash typing (R30) |
| s-domain minimum-time OCP | Minimum time over the strip | "Physics requires" statements, tier D |

- The speed band is b = 10%. The pre-registered V8 rule clipped it at its ceiling (R15; `manifest.json`, `band`).
- The manifest lists no unsolved or infeasible P0 keys and 26 multi-basin cells, which are excluded from key-based errors (`manifest.json`, `lists`).
- The key table marks 5 tier D cells, all at one Hockenheim corner, with an unsolved TO-OCP key (`analysis/key_table.npz`; this file was rewritten by the non-confirmatory run, Section 7.2). Tier D's decisions read completion and s-OCP keys.
- The post-tag anchored-keys stage ran in six local invocations and on a rented cloud machine (Sections 7.2 and 7.3).
  - Its last invocation records 2,171 jobs (re-anchored and viability) with 0 errors (`manifest.json`, `validation.anchored_keys`).
  - Of the 9,621 re-anchored keys in the cache, 96 are unsolved (Section 7.3).
  - A run without a verified re-anchored key at its phase is judged against the nominal key at that phase (protocol 6.1; R12, R17a).

**Controls and dry run.** Scripted controllers went through the same harness with required verdicts that blocked the tag until met:

| Control | What it is |
|---|---|
| T | Tape: the key's actions replayed |
| T64 | Tape in float64 |
| Q | A QSS controller on the centreline |
| Qp | Q with preview only |
| c1 | Brakes 15 m late against the key |
| c2 | Curvature-saturating reader |
| c3 | Pure time-to-corner trigger |
| K | kNN memoriser of S1 |
| OB, LS, LSK, LSKF | Planted crash controls for crash typing |

- All 12 controls passed (pre-registration record in `IMPLEMENTATION.md`).
- The dry run on the real key design returned all 13 planted verdicts of its simulated agents (`outputs/psychophysics/analysis/dry_run.npz`). The simulated agents were lawful, overspeeding below 12 m, late onset, time-to-corner braking, and context-keyed.
- The dry run wrote the power table (`power.npz`, Section 3).

**Pre-registration.**
- `psychophysics-prereg-v2` tags commit `1338394` (20 Oct 2026, 19:15 EDT).
- The record frozen at the tag (`IMPLEMENTATION.md`):
  - stimulus hash `e7846c92…`;
  - key-cache hash `6d79b163…`;
  - frozen-code hash `8650965a…` (`src/apexzero/interp` and `scripts/psychophysics.py`);
  - broad code hash `b4072de7…` (all of `src/apexzero` and the script; the manifest's `code_hash`);
  - dry run passed;
  - P3 not demoted.
- Agent rollouts on protocol stimuli began at 19:15:22 the same evening (`runs/psychophysics-rollouts.log`).
- Before the tag the only agent runs were the pre-registered exceptions (`manifest.json`, `lists.pre_tag_agent_rollouts`, 7 entries, and `lists.standard_evaluations`):
  - the step-2 pilot replication of G1 and G2 on the 24-cell pilot mini-grid, run three times on 19 Oct (05:12, 05:14, 07:13);
  - S1's kNN-buffer laps (19 Oct, 08:49);
  - 35 R1 standard evaluations (19 Oct, 08:43–08:44).

**Families** (protocol Section 8):
- Primary: G3's P1a, P1-tail, P1b, P2 and P3; S2's and S3's P4 class; the specialist recipe claim.
- Secondary: G1's and G2's P1a–P3; G2's P8-tail; S1's P4.
- Replication (R18): P1a–P3 on the six replicates. A variant-level claim needs the same decision in at least 2 of the variant's 3 seeds.
- Stability: a network-level claim must also hold at one of the two snapshots before the final (134.1M and 163.8M steps).

**Volume.** Each final ran 31,704 runs: 4,792 deterministic, 20,576 native-noise and 6,336 matched-noise (`runs/psychophysics-analyse.log`; G2's final rollout file).

## 3. Power limits (read these before the verdicts)

These limits were written before the tag (rulings R24, R26 and R27, and `outputs/psychophysics/analysis/power.npz`). They bound what an "inconclusive" or "refute" can mean.

1. **P1b cannot see a curvature-saturating reader (R24).** P1b reads entry speed v_e. A reader that saturates its curvature reading moves v_min strongly but v_e by only 0–0.7%.
   - In the dry run, a planted saturating reader made P1b confirm only if it saturated at q1 ≥ 17.5 m (G1's grid) or ≥ 14.4 m (G2's). For G3 it never confirmed on the tested grid up to 14.4 m.
   - Saturating at 9.7–11.3 m on G1's grid even returned "refute".
   - So an inconclusive or refuted P1b does not exclude a reader that saturates curvature below support.
2. **P3 cannot refute a hybrid braking rule (R26).** A planted driver that brakes at the earlier of a physical and a time-to-corner trigger gives b ≈ 0.78–0.79 for G1–G3 in the power table (0.77–0.79 across all 12 agents; `power.npz`). R26's own text gives b ≈ 0.82. In every agent's power row the preview-index model wins on RMSE for the planted hybrid. That is inconclusive under R25's calibrated band, not a refutation.
   - A P3 confirm still requires the physics model to win the held-out comparison, which the hybrid did not.
   - A time-to-corner rule is also invisible where braking is capacity-limited (R26).
3. **P4's class 2 can be reached vacuously (R10, R27).** Class 2 requires every F6 perturbation that is at least as far off-manifold (d_OOD) as the failing F1 condition to complete.
   - In S1's numbers every F6 perturbation sits at d_OOD ≤ 1.01, while the single-factor F1 conditions sit at 1.07–5.48 (`decisions.json`, S1 P4 details). For S1, then, any class 2 would hold vacuously.
   - The control K reached class 2 vacuously, and it validates only the competence gate and the class-1 path (R27). Only the simulated context-keyed agent validated class 2 against class 1.
4. **Detection sizes** (`power.npz`):

   | Test | Smallest planted overspeed detected |
   |---|---|
   | P1a, P1-tail | 12.5% |
   | P1b, P2 | 10% (S3's P1b: 15%) |

   - The band is 10%, so overspeeds of a few percent are inside it by design.
   - P2 needs at least 3 overspeeding pairs in its set O; with fewer it is "not applicable".
5. **One network per generalist variant** in the primary and secondary families. R18's seed replicates are the check. Differences between variants smaller than the seed spread are not interpreted.

## 4. Decisions

Sections 4.1–4.3 come from the one confirmatory run: `outputs/psychophysics/analysis/decisions.json` and `replication.json`. The run log is `runs/psychophysics-analyse.log`: start 2026-10-21 01:31:22, done 01:42:39, 11.3 min. "Stable" is the 2-of-last-3-snapshots rule. Section 4.4 is the post-tag P4 rerun. Section 4.5 lists the pre-registered analyses that only the later, non-confirmatory run decided.

### 4.1 Confirmatory decision table

| Family | Agent | Prediction | Verdict | Stable | Key numbers |
|---|---|---|---|---|---|
| Primary | G3 | P1a | **confirm** | yes | 15 bulk pairs at A2, 100% in band; S_bulk 0.935 (leave-one-R-out 0.923–0.950); 0 deterministic crash pairs; native-noise P_crash 99% upper bound 0.52% |
| Primary | G3 | P1-tail | **inconclusive** | yes | A2: 6 tail pairs, 83% in band, 17% under. A1: 7 pairs, 0% over, 71% in band, 29% under, 0 crashes. Contrast e_ve^RA(A1) − e_ve(A2) positive in 0 of 6 geometries |
| Primary | G3 | P1b | **inconclusive** | yes | 11 pairs; 0% departing (r > log 1.1 or overspeed crash); 55% lawful; 0 crash pairs; Kendall τ(r, R_seg) +0.18 at A1, +0.06 at A2 (prediction: ≤ −0.3) |
| Primary | G3 | P2 | **not applicable** | yes | Set O empty (0 overspeeding tail or below-support pairs at A1) |
| Primary | G3 | P3 | **confirm** | yes | b = 0.947 (b* = 0.765; confirm band 0.665–1.10); leave-one-geometry-out 0.937–0.954; held-out RMSE of log d_b: M_phys 0.037, M_idx 0.164, M_fd 0.309; 1,165 runs in 33 geometries |
| Primary | S2 | P4 | **not testable** | — | d_OOD missing (no buffer). Single-factor F1 failures: road after straightened C = 0.22, constant width C = 0.22. MI 1.04. Crashes on a null straight (tier G rule: evidence for class 1 on synthetic stimuli) |
| Primary | S3 | P4 | **not testable** | — | d_OOD missing. Single-factor F1 failure: constant width C = 0.44. MI 0.70. Crashes on a null straight |
| Primary | S1–S3 | P4 recipe | **inconclusive** | — | Classes: S1 6, S2 and S3 not testable. "Class 2 for all three" is not met |
| Secondary | G1 | P1a | **confirm** | yes (a) | 8 pairs, 100% in band; S_bulk 0.900 (0.872–0.913); 0 crash pairs; P_crash bound 0.92% |
| Secondary | G1 | P1-tail | **inconclusive** | yes (a) | A2: 10 pairs, 80% in band, 20% under (S at A2 1.32). A1: 10 pairs, 0% over, 70% in band, 30% under, 0 crashes. Contrast positive in 0 of 10 |
| Secondary | G1 | P1b | **inconclusive** | yes (a) | 11 pairs; 0% departing; 55% lawful; 0 crash pairs; τ +0.38 (A1), +0.40 (A2) |
| Secondary | G1 | P2 | **not applicable** | yes (a) | Set O empty |
| Secondary | G1 | P3 | **confirm** | yes (a) | b = 1.006 (b* = 0.872; band 0.772–1.10); leave-one-out 0.994–1.033; RMSE 0.045 / 0.108 / 0.199; 1,049 runs in 31 geometries |
| Secondary | G2 | P1a | **confirm** | yes | 10 pairs, 100% in band; S_bulk 0.910 (0.894–0.927); 0 crash pairs; P_crash bound 0.75% |
| Secondary | G2 | P1-tail | **refute** | yes | A2: 9 pairs, 89% in band, 11% under (S 1.04). A1: 9 pairs, 100% in band, 0% over, 0 crashes. Contrast positive in 0 of 9. Pre-registered sensitivity without unsettled runs: inconclusive (the A1 tail set is flagged, > 20% unsettled) |
| Secondary | G2 | P1b | **inconclusive** | yes | 12 pairs; 8% departing (1 pair); 50% lawful; 1 crash pair; τ +0.30 (A1), +0.13 (A2) |
| Secondary | G2 | P2 | **not applicable** | yes | Set O empty |
| Secondary | G2 | P3 | **inconclusive** | **no** | b = 1.142, above 1.10 (b* = 0.873); leave-one-out 1.133–1.152; RMSE 0.041 / 0.129 / 0.241 (M_phys wins); both earlier snapshots: confirm |
| Secondary | S1 | P4 | **class 6 (mixed)** | — | C(a) = 1.0 over 9 braking corners. Single-factor F1: constant width 0.56, before straightened 0.78, after straightened 0.78; isolated 0.22; mirror 0.11. S_spec 0.93; phantom index φ 0.04; swap index m 0.44; MI 0.81. No tier G crash |
| Secondary | G2 | P8-tail | **refute** | — | n = 3 tail units, 3 reproduced, 0 single-corner with twin agreement (confirm needs ≥ 2, refute ≤ 0). Labels: width necessary in all 3; Melbourne also needs the road after |

Notes on the table:
- (a) G1's "stable" rests on its 163.8M snapshot alone. Its 134.1M snapshot returned "not testable" for all five predictions (`history_verdicts` in the details).
- In this seed-0 table, every verdict is unchanged when lock-limited cells or unsettled runs are excluded (`without_lock_limited`, `without_unsettled_runs` in the details), except G2's P1-tail, noted in its row.
- In P1b, a few A1 pairs had at least one run without a verified re-anchored key at its phase, either because the key is unsolved or because re-anchoring did not apply. Those runs were judged against the nominal key at that phase: G1 4 pairs, G2 2, G3 3 (`ra_fallback_pairs`).

### 4.2 P8-tail: G2's real tail crashes

By R2, a crash corner is the `corners.segment` corner that contains the crash, and its regime comes from its R_seg under G2's frozen q1 / q10 (11.1 / 17.7 m). That leaves three tail units. G2's other two former "tail" crashes, Yas Marina 1356 (R_seg 20.8 m) and Zandvoort 3205 (28.0 m), are bulk. Thresholds are proportional (R2): confirm at ⌈0.6n⌉ = 2 or more, refute at ⌊0.2n⌋ = 0 or fewer. All three transplants passed the fidelity gate (`fidelity_failed` and `fidelity_missing` are empty).

| Unit: corner start, m (unit id in `decisions.json`) | R_seg (m) | Angle (°) | Reproduced | Single-corner with twin | Necessary factor |
|---|---|---|---|---|---|
| Norisring 874 (`G2/Norisring/910`) | 15.1 | 94 | yes | no | constant width |
| Melbourne 2531 (`G2/Melbourne/2562`) | 15.1 | 90 | yes | no | constant width; road after straightened |
| Catalunya 4168 (`G2/Catalunya/4182`) | 14.8 | 85 | yes | no | constant width |

Sources: `decisions.json` (P8-tail details) and `outputs/psychophysics/stimuli/stimuli.json` (geometry).

The refutation follows from the count of single-corner units: in all three, the isolated rung completes, so `single_with_twin` is 0, at the refutation threshold `refute_max` 0. The necessary-factor labels are a separate output of the same ladder.

G2's deterministic completion of each rung, over all crash states of the corner × 4 phases (recomputed from G2's final rollout file; descriptive), is:
- Catalunya: 0 of 4 on real, before straightened, after straightened and mirror; 4 of 4 on constant width, isolated and twin.
- Melbourne: the same pattern, except that the road-after-straightened rung also completes 4 of 4.
- Norisring 874: the same pattern on both of its crash states (G2's and G3's), 8 of 8 or 0 of 8.

### 4.3 Replication across seeds (R18) and stability

Source: `outputs/psychophysics/analysis/replication.json` (`variants`, `spread`, `decisions`). The variant verdict is the decision shared by at least 2 of the 3 seeds, and a claim is a shared confirm or refute.

| Variant | Prediction | Seed 0 | Seed 1 | Seed 2 | Variant verdict | Claim |
|---|---|---|---|---|---|---|
| v1 | P1a | confirm | confirm | confirm | **confirm** | yes |
| v1 | P1-tail | inconclusive | unstable (final: refute) | inconclusive (not stable) | inconclusive | no |
| v1 | P1b | inconclusive | inconclusive | inconclusive | inconclusive | no |
| v1 | P2 | not applicable | not applicable | not applicable | not applicable | no |
| v1 | P3 | confirm | inconclusive | confirm | **confirm** | yes |
| v2 | P1a | confirm | confirm | confirm | **confirm** | yes |
| v2 | P1-tail | refute | refute | unstable (final: refute) | **refute** (1 of 3 seeds without unsettled runs) | yes |
| v2 | P1b | inconclusive | inconclusive | inconclusive | inconclusive | no |
| v2 | P2 | not applicable | not applicable | not applicable | not applicable | no |
| v2 | P3 | inconclusive | inconclusive | confirm | inconclusive | no |
| v3 | P1a | confirm | confirm | confirm | **confirm** | yes |
| v3 | P1-tail | inconclusive | inconclusive | inconclusive | inconclusive | no |
| v3 | P1b | inconclusive | inconclusive | inconclusive | inconclusive | no |
| v3 | P2 | not applicable | not applicable | not applicable | not applicable | no |
| v3 | P3 | confirm | confirm | inconclusive | **confirm** | yes |

Key numbers per network. R_dep is the smallest grid radius from which every larger braking radius is in band with no deterministic crash. The P1c ordering decision built on it is in Section 4.5.

| Network | S_bulk (P1a) | A1 tail in band / under / over (P1-tail) | b (P3) | R_dep A1 / A2 (m) |
|---|---|---|---|---|
| G1 | 0.900 | 70% / 30% / 0% | 1.006 | 14.6 / 13.1 |
| G1s1 | 0.971 | 80% / 20% / 0% | 1.261 | 13.1 / 13.1 |
| G1s2 | 0.928 | 70% / 30% / 0% | 1.004 | 14.6 / 13.1 |
| G2 | 0.910 | 100% / 0% / 0% | 1.142 | 10.5 / 11.7 |
| G2s1 | 0.974 | 89% / 11% / 0% | 1.148 | 11.7 / 11.7 |
| G2s2 | 0.969 | 89% / 11% / 0% | 1.011 | 11.7 / 11.7 |
| G3 | 0.935 | 71% / 29% / 0% | 0.947 | 10.5 / 10.5 |
| G3s1 | 0.964 | 71% / 29% / 0% | 1.056 | 10.5 / 11.7 |
| G3s2 | 0.972 | 43% / 57% / 0% | 1.220 | 13.1 / 11.7 |

**P1a.** All 9 networks have every bulk pair in band and no deterministic crash pair. The 99% bound on native-noise P_crash is at most 0.92%.

**P3.**
- The physics model has the lowest held-out RMSE in all 9 networks.
- All four inconclusive P3 verdicts (G2, G1s1, G2s1, G3s2) have b above the upper edge, at 1.14–1.26. None sits on the time-to-corner side.

**Stability.**
- Every seed-0 decision is stable except G2's P3, which confirmed at both earlier snapshots and is inconclusive at the final.
- G1's stability rests on its 163.8M snapshot alone, because its 134.1M snapshot was "not testable" for all five predictions. G1s2's rests on its 134.1M snapshot alone, because its 163.8M snapshot was "not testable" for all five.
- Among the replicates, G1s1's and G2s2's final P1-tail refutations did not recur at either earlier snapshot, so they are reported as "unstable".
- G1s2's P1-tail is also not stable. It is inconclusive at the final, but its history is refute at 134.1M and not testable at 163.8M (`spread`: `"P1-tail": "no"`). G1's and G1s2's shared "inconclusive" is what makes v1's P1-tail variant verdict (2 of 3 agree).
- Excluding unsettled runs, a pre-registered sensitivity analysis, changes three P1-tail verdicts to inconclusive: G2's refutation and G1s1's and G2s2's final-snapshot refutations. It also turns G2s2's P3 confirm into inconclusive.
- G2s1's P1-tail refutation is the only one that survives every sensitivity analysis. In that analysis, v2's P1-tail refutation stands on 1 of 3 seeds.

### 4.4 Post-tag P4 rerun for S2 and S3 (not confirmatory)

**Status.**
- S2's and S3's kNN d_OOD buffers were built after the confirmatory run (`runs/psychophysics-controls.log`: 2026-10-21 01:43:50–01:44:01; `manifest.json`, `lists.post_tag_buffers`).
- The non-confirmatory `analyse` (01:45:09–01:48:12, `runs/psychophysics-analyse.log`) then re-decided P4 with them (`outputs/psychophysics/analysis/decisions_all.json`).
- This is a post-tag analysis made after the one confirmatory run, and it is logged as a deviation (Section 7.2). The confirmatory table above keeps "not testable" for S2 and S3 and "inconclusive" for the recipe claim.
- The code labels these rows `"primary"`, which is their protocol family. The label does not make the rerun confirmatory.

| Agent | Class | Label | Failing F1 condition | MI | S_spec | φ | Flags |
|---|---|---|---|---|---|---|---|
| S2 | 2 | context-keyed memorisation | road after straightened (`F1/real/straight/real`) | 1.04 | 0.88 | 0.55 | `vacuous`, `n_f6_at_or_above` 0, `tier_g_crash` |
| S3 | 2 | context-keyed memorisation | constant width (`F1/real/real/const`) | 0.70 | 0.79 | 0.02 | `vacuous`, `n_f6_at_or_above` 0, `tier_g_crash` |
| S1 (for reference; confirmatory) | 6 | mixed | none at ≤ 0.5 | 0.81 | 0.93 | 0.04 | none |
| Recipe (S1–S3) | "class 2" from 2 of 3 seeds | | | | | | `predicted_class_2_all: false` |

**Reading under the protocol's own rules.**
- **Vacuous class 2 (R10, R27).** Class 2 requires every F6 perturbation at least as far off-manifold (d_OOD) as the failing F1 condition to complete. For S2 and S3 no F6 perturbation is that far off-manifold (`n_f6_at_or_above` 0). The condition therefore holds trivially, and the classification says nothing about whether irrelevant changes are tolerated. The dry run validated class 2 against class 1 only for the simulated context-keyed agent (R27). Section 3 item 3 warned in advance that class 2 could be reached this way.
- **The tier G rule.** S2 and S3 both crash on a plain null straight. The protocol reads that as evidence for class 1 (fragility off the training manifold) on synthetic stimuli, and fragility cannot be told apart from context keying. Their class 2 therefore does not establish memorisation over fragility.
- **The recipe claim.** The recipe-level "class 2" rests on those two vacuous classifications. Its pre-registered prediction, class 2 for all three seeds, is not met (S1 is class 6).
- **What it does say.** Each specialist's completion collapses under one single-factor change of Hockenheim's road: S2's when the road after a corner is straightened, S3's when the width is made constant. S2 also brakes for corners replaced by straight road (φ 0.55; S1 0.04, S3 0.02). All three lose far more than the generalists when Hockenheim is isolated and made constant-width (MI 0.70–1.04 against the generalists' −0.04). The specialists are tied to Hockenheim's context. Whether that is memorisation or fragility, these stimuli cannot decide.

### 4.5 Pre-registered analyses decided after the confirmatory run (post-confirmatory)

**Source and status.**
- The source is `outputs/psychophysics/analysis/decisions_all.json`. The non-confirmatory `analyse` wrote it (2026-10-21 01:45:09–01:48:12, `runs/psychophysics-analyse.log`) from the same rollouts and key cache as the confirmatory run.
- By design, the confirmatory mode decides only the family rows of Section 4.1. The protocol's secondary analyses, its control (P5) and its exploratory analyses are decided only in the non-confirmatory mode.
- These verdicts are pre-registered but not confirmatory, and they were decided after the confirmatory results were known.
- The "Family" column is the one `decisions_all.json` gives.

| Prediction (protocol 2.4) | Family | Verdicts | Key numbers |
|---|---|---|---|
| P1c: R_dep order v1 ≥ v2 ≥ v3 at A1 (R18: each variant's median over 3 seeds) | Secondary | **confirm** | Medians 14.6, 11.7, 10.5 m; exact null probability 1/6 (`p_null` 0.167) |
| P1c-approach: R_dep(A1) ≥ R_dep(A2) | Secondary | G1 confirm, **G2 refute**, G3 confirm | G2: 10.5 m at A1 against 11.7 m at A2 |
| P2b: onset or depth | Secondary; replication | not applicable for all 9 generalists | Set O empty |
| P3-β, corners under 120°: \|β − β*\| ≤ 0.1 | Secondary | **refute** for G1, G2 and G3 | β 0.79, 0.74, 0.70 against β* 0.54 |
| P3-β, corners of 120° or more | Secondary | G1 confirm, **G2 refute**, G3 confirm | β 0.77, 0.82, 0.80 against β* 0.72 |
| P3-β, replicates | Replication | Under 120°: refute G1s1, G3s1, G3s2; confirm G1s2, G2s1, G2s2. 120° or more: refute G1s1, G3s2; confirm the other four | — |
| P5: mirror symmetry | Control | **refute** for 8 of 9 generalists (all but G2) and for S1. G2 confirms. S2 and S3 are descriptive under the tier G rule (rule verdict: refute) | Criteria: median \|Δd_b\| ≤ 5 m, median \|Δe_ve\| ≤ 2 points, crash concordance ≥ 90%. Median \|Δd_b\|: G1 9.6 m, G2 3.2 m, G3 5.7 m. S1: concordance 64%, median \|Δd_b\| 62.8 m |
| P6a: larger R competent first | Secondary | **refute** for G1, G2 and G3 | Spearman ρ(R, t_R) +0.12 to +0.13 (prediction ≤ −0.5); p 0.69–0.72 |
| P6b: caution first | Exploratory | refute for G1, G2 and G3 | Median bulk e_ve^bo at each agent's first gate-passing checkpoint (73.7M, 134.1M, 109.9M): −0.06, −0.03, −0.02 (prediction < −0.10) |
| P6c: overspeed appears late | Secondary | not applicable for G1, G2 and G3 | Tail e_ve^RA at A1 is below +b at the final; the checkpoint keys behind it are discussed in Section 7.2 |
| P7: preview sampling | Secondary; replication | **refute** for 11 of 12 finals; G1 inconclusive | On-policy aliasing on short cells 0.005–0.010 against Qp's 0.076; G1's share of cells with a flat closed-loop excess is 0.7 (`frac_all_flat`; refutation needs 75%) |
| P8-bulk: context in ≥ 2/3 of reproduced bulk units | Secondary | **confirm** (G1 and G2 units) | 10 of 12 reproduced units are context |
| P8-twin: twin transfer | Secondary | **refute** for G1 and G3; G2 inconclusive | Isolated real corner and twin agree in 79%, 80%, 70% of corners (prediction ≥ 80%); Spearman 0.29, 0.04, 0.05 (prediction ≥ 0.5) |
| P2d, P3 ratio, P6d, P8-F | Exploratory or descriptive | descriptive | Not written up here |

**What the refutations say,** in the protocol's own terms (Section 9.6):
- **P5.** These networks are not mirror-symmetric (see Section 6.3). As the protocol requires, the harness mirror gate was re-run (21 Oct, 02:45), and it still passes: maximum |Δ| of 0.0 over 400 steps on both gate pairs, deterministic and with mirrored noise. A post-hoc check (not pre-registered) also recomputed P5 on exactly mirrored pairs only. At A2, each member of a pair has its own start distance, laid out from its own key, so 43 of the 68 A2 pairs start up to 208 m apart, most by under 5 m. Without them, every verdict is unchanged, and the median |Δd_b| moves by at most 0.4 m (Section 7.2). By the protocol's rule the asymmetry is therefore reported as learned. It shows mostly in where the networks brake: median |Δd_b| is 5.0–12.4 m for the eight that fail (above the 5 m limit in seven; G2s1 fails on speed error), while their speed errors differ by 1.5–2.9 points.
- **P6a.** Competence does not arrive in order of corner size.
- **P7.** The preview behaves like an ideal sampled observer by the protocol's measures.
- **P8-twin.** The trapezoid twin does not stand in for the isolated real corner as well as predicted for G1 and G3.
- **P3-β.** For corners under 120° the agents' braking distance grows with v_b² − v_e*² more steeply than the key's. This qualifies P3's slope result (Section 5).

## 5. What the results mean

**Severity is read lawfully on typical approaches (P1a: confirmatory for G1–G3; replication for all 9 networks).**
- On corners like the ones they saw often in training, each generalist's entry speed is within the band in every pair: at most 10% above its own optimum, and at most 10% below the backed-off optimum.
- Entry speed scales with the optimum's at slopes of 0.90–0.97, with no deterministic crashes and a noisy crash rate bounded below 1%.
- This holds for all three variants and all nine networks.
- In plain terms, the networks are not misjudging ordinary corners.

**Braking distance scales like physics; a pure preview-index rule is disfavoured, a hybrid is not excluded (P3).**
- Confirmatory: P3 confirms for G3 (primary) and G1 (secondary). G2 is inconclusive.
- Replication: P3 confirms for variants v1 and v3 and is inconclusive for v2. It is also inconclusive for G1s1, G2s1 and G3s2.
- In all 9 networks the physics model predicts held-out braking distances best. Held-out RMSE of log d_b:
  - physics model: 0.04–0.08;
  - preview-index rule: 0.11–0.19, which is 1.4–4.5 times the physics model's, network by network (`rmse` in `decisions.json` and `replication.json`);
  - fixed distance: 0.19–0.32.
- A pure time-to-corner or fixed-distance rule is therefore disfavoured.
- A hybrid rule is not excluded:
  - P3 cannot refute one (R26), and the planted hybrid returned "inconclusive".
  - A time-to-corner rule is invisible where braking is capacity-limited.
  - What the data add is that none of the nine networks looks like the planted hybrid, for which the preview-index model won.
- Where P3 is inconclusive, the slope is above the band (1.14–1.26), not on the time-to-corner side.
  - b is a log–log slope with a separate intercept per geometry. b > 1 says braking distance grows faster than the physical distance as the speed drop grows. Equally, it says braking distance is relatively shorter for small drops.
  - It says nothing about whether onsets are early or late in absolute terms. Safety was not tested here.
- The post-confirmatory β fits (Section 4.5) disagree with the key's for corners under 120° in G1, G2 and G3.

**No fast-approach overspeed was seen (P1-tail, P2).**
- The study's main hypothesis was that the generalists crash tight corners because they rarely saw them at high speed. They would then enter too fast and brake too late after a 310 km/h approach.
- Descriptively, that did not happen. Across all nine networks, 0% of tail pairs overspeed at A1. The within-geometry contrast (fast approach minus typical approach, each against its own key) was positive in no geometry for eight networks, and in 1 of 6 for G3s2.
- Formally, P1-tail is:
  - inconclusive for G3 (primary) and G1 (secondary);
  - refuted for G2 (secondary) and for variant v2 (replication, 2 of 3 seeds).
- Both refutations are fragile. Without unsettled runs, G2's becomes inconclusive and v2's rests on 1 of 3 seeds.
- Where the agents depart from the band after a fast approach, they are too slow: up to 30% of tail pairs underspeed for the seed-0 networks, and 57% for G3s2. G1 and G3 are inconclusive because they are cautious, not because they overspeed.
- With no overspeeding pair, P2 was not applicable: its late-braking mechanism had nothing to explain.

**Below training support the result is open (P1b: confirmatory, inconclusive everywhere).**
- No generalist departs upward: at most 1 of 12 pairs, in G2.
- About half the seed-0 pairs are lawful (50–55%; 27–64% across all nine networks), and the rest are slower than the bulk law predicts.
- Kendall τ(r, R_seg) is positive at both approaches in every network (seed 0: +0.06 to +0.40). Relative speed falls as R falls, the opposite of the predicted sign.
- Section 3's limit applies: P1b cannot see a reader that saturates curvature, so these verdicts say nothing for or against one.

**G2's three tail crashes are context failures (P8-tail: confirmatory, secondary family).**
- G2's three real tail crashes reproduce on the transplanted road from G2's own lap state.
- In none of them does the isolated corner crash, so none is a single-corner failure, and P8-tail is refuted.
- The ladder labels constant width as a necessary factor in all three, and the straightened road after the corner as a second one at Melbourne. In protocol 9.6's terms, these crashes are not explained by reading a single corner.
- By the pre-registered rule, P1-tail and P2 could have explained G2's real crashes only if P8-tail confirmed, and it was refuted.
- This verdict covers G2's three tail units only. The descriptive ladders in Section 6 extend the picture to G1 and G3 (not confirmatory), with exceptions noted there.

**Specialists (P4): S1 is class 6; S2 and S3 were not testable in the confirmatory run.**
- Confirmatory:
  - S1 (secondary family) falls in class 6, "mixed", the protocol's class for a specialist that meets none of classes 0–5.
  - S2 and S3 (primary family) are not testable, because d_OOD was missing.
  - The recipe prediction (class 2 for all three seeds) is not met.
- S1's profile:
  - it reads where corners are (φ 0.04: it does not brake for a corner replaced by straight road);
  - its speed scales with corner size (S_spec 0.93);
  - but it completes only 1 of 9 corners when the road is mirrored;
  - and 5 of 9 when only the width is made constant.
- Why class 6: no single-factor F1 condition drops S1's completion to 0.5 or below, which classes 1 and 2 need. Not all eight F1 conditions are at 0.8 or above, which classes 3–5 need.
- The memorisation index (MI) is the specialist's completion loss from real Hockenheim to the isolated, constant-width version, minus the generalists' loss on the same change.
  - It is 0.81 (S1), 1.04 (S2) and 0.70 (S3).
  - The generalists' mean loss is −0.04, which means they do slightly better on the isolated version.
  - The protocol reports MI with every class but sets no threshold on it, so calling these values large is a judgement, not a decision.
- S2 and S3 crash on a null straight. Under the tier G rule that is evidence for class 1 on synthetic stimuli: fragility off their manifold, which the protocol says cannot be told apart from context keying. Their synthetic-tier results count as completion rates only.
- The post-tag P4 rerun for S2 and S3 (Section 4.4) puts both in class 2, vacuously and with a tier G crash, so it does not decide between memorisation and fragility.

## 6. Descriptive results (not confirmatory)

These numbers were recomputed from the final deterministic rollouts (`outputs/psychophysics/rollouts/<agent>/<final>.npz`) with the stimulus and cell tables (`outputs/psychophysics/stimuli/`). Each subsection states its method so that it can be reproduced. They decide nothing.

### 6.1 Crash-ladder completion by rung

**Method.** A run completes if it does not terminate, or terminates only beyond s_c1 + 250 m. Every agent runs every rung from every crash state of that corner. There are 21 crash states over 15 corners: 8 from G1, 7 from G2 and 6 from G3, the last from G3's own R-dev crashes (`stimuli/crash_pairs_extra.json`). With 4 phases that gives 84 runs per agent and rung.

| Agent | Real | Before straightened | After straightened | Constant width | Isolated | Twin | Mirror |
|---|---|---|---|---|---|---|---|
| G1 | 13/84 | 16/84 | 8/84 | **84/84** | 76/84 | 76/84 | 24/84 |
| G2 | 26/84 | 34/84 | 27/84 | **84/84** | 84/84 | 84/84 | 48/84 |
| G3 | 36/84 | 36/84 | 32/84 | **75/84** | 72/84 | 76/84 | 40/84 |
| G1s1 | 32/84 | 32/84 | 28/84 | 80/84 | 84/84 | 84/84 | 31/84 |
| G1s2 | 52/84 | 48/84 | 52/84 | 84/84 | 76/84 | 84/84 | 40/84 |
| G2s1 | 48/84 | 36/84 | 46/84 | 76/84 | 76/84 | 76/84 | 56/84 |
| G2s2 | 37/84 | 49/84 | 29/84 | 82/84 | 81/84 | 81/84 | 44/84 |
| G3s1 | 35/84 | 40/84 | 31/84 | 76/84 | 84/84 | 84/84 | 52/84 |
| G3s2 | 51/84 | 56/84 | 54/84 | 80/84 | 84/84 | 84/84 | 35/84 |

From each seed-0 agent's own crash states only:

| Agent | Real | Before straightened | After straightened | Constant width | Isolated | Twin | Mirror |
|---|---|---|---|---|---|---|---|
| G1 | 2/32 | 2/32 | 0/32 | 32/32 | 28/32 | 28/32 | 8/32 |
| G2 | 0/28 | 8/28 | 8/28 | 28/28 | 28/28 | 28/28 | 8/28 |
| G3 | 0/24 | 0/24 | 0/24 | 20/24 | 16/24 | 20/24 | 8/24 |

Per corner, for the seed-0 agents: completed runs on each rung, in the order real, before straightened, after straightened, constant width, isolated, twin, mirror. Corners are sorted by R_seg.

| Corner | R_seg (m) | Angle (°) | Crash states from (runs per rung) | G1 | G2 | G3 |
|---|---|---|---|---|---|---|
| Yas Marina 2817 | 11.0 | 85 | G1, G3 (8) | 0 1 0 8 8 8 0 | 0 0 0 8 8 8 8 | 0 0 0 8 8 8 0 |
| Melbourne 1873 | 13.8 | 82 | G3 (4) | 0 0 0 4 0 0 0 | 4 4 4 4 4 4 4 | 0 0 0 4 0 0 0 |
| Norisring 446 | 13.9 | 184 | G1 (4) | 0 0 0 4 4 4 0 | 4 4 4 4 4 4 4 | 4 4 4 4 4 4 4 |
| Catalunya 4168 | 14.8 | 85 | G2 (4) | 4 4 4 4 4 4 0 | 0 0 0 4 4 4 0 | 4 4 4 4 4 4 4 |
| Melbourne 2531 | 15.1 | 90 | G2 (4) | 0 0 0 4 4 4 0 | 0 0 4 4 4 4 0 | 0 0 0 4 4 4 0 |
| Norisring 874 | 15.1 | 94 | G2, G3 (8) | 0 0 0 8 8 8 0 | 0 0 0 8 8 8 0 | 0 0 0 8 8 8 0 |
| Melbourne 348 | 20.0 | 73 | G1 (4) | 1 1 0 4 4 4 0 | 4 4 4 4 4 4 4 | 4 4 0 4 4 4 0 |
| Hockenheim 248 | 20.3 | 53 | G1 (4) | 0 0 0 4 4 4 0 | 4 4 1 4 4 4 0 | 4 4 4 4 4 4 0 |
| Yas Marina 1356 | 20.8 | 105 | G2 (4) | 3 4 0 4 4 4 4 | 0 4 4 4 4 4 4 | 4 4 4 4 4 4 4 |
| Sochi 2026 | 22.2 | 85 | G1, G2, G3 (12) | 0 0 0 12 12 12 0 | 0 0 0 12 12 12 0 | 0 0 0 11 8 8 0 |
| Zandvoort 336 | 25.2 | 180 | G1, G3 (8) | 0 0 0 8 8 8 8 | 2 2 2 8 8 8 8 | 0 0 0 8 8 8 8 |
| Zandvoort 3205 | 28.0 | 165 | G2 (4) | 0 2 0 4 4 4 4 | 0 0 0 4 4 4 0 | 4 4 4 0 4 4 4 |
| Brands Hatch 1136 | 34.0 | 138 | G3 (4) | 4 4 4 4 4 4 0 | 4 4 4 4 4 4 4 | 0 0 0 0 0 4 4 |
| Brands Hatch 2724 | 34.9 | 45 | G1 (4) | 1 0 0 4 0 0 0 | 4 0 0 4 4 4 4 | 4 4 4 4 4 4 4 |
| Oschersleben 418 | 44.3 | 227 | G1, G2 (8) | 0 0 0 8 8 8 8 | 0 8 0 8 8 8 8 | 8 8 8 8 8 8 8 |

What the ladders show:
- Straightening the road before or after a corner rarely rescues a crash.
- Holding the width constant at the corner's median width rescues all of G1's and G2's runs and 75 of G3's 84.
- Constant width does not always help. At Zandvoort 3205, G3 completes the real road (4 of 4) but crashes on the constant-width rung (0 of 4). At Brands Hatch 1136, G3 is 0 of 4 on constant width.
- The isolated corner and its twin mostly complete. The exceptions are where the corner itself, or the isolated version's changed approach, defeats the agent:
  - G1 at Melbourne 1873 and Brands Hatch 2724: isolated and twin 0 of 4, constant width 4 of 4.
  - G3 at Melbourne 1873: isolated and twin 0 of 4.
  - G3 at Sochi: isolated and twin 8 of 12.
  - G3 at Brands Hatch 1136: crashes on every rung except the twin and the mirror. That corner is 206 m long, longer than the longest trapezoid with its R_seg and angle (154 m), so its twin is shorter (R4a). Variant v3 completes Brands Hatch with 0 of 5 seeds ([generalist-seeds.md](generalist-seeds.md)).

### 6.2 Crash positions against the width profile

**Method.**
- Take every real-rung deterministic crash of G1, G2 and G3 (177 runs: 71, 58 and 48).
- Read the widths of the stimulus bank (`stimuli/bank.npz`, 2 m knots) at the crash point and over the 100 m before it.
- The crash side is the sign of the lateral offset n at termination. The inside edge is σn > 0 (protocol Section 5).

Results:
- 160 of the 177 crashes fall within [s_c0 − 50, s_c1 + 100], at a median of 32 m after the corner start.
- The other 17 happen on the road leading to the corner:
  - 8 are G1's on the Norisring 874 strip, about 395 m before the corner;
  - 8 are G1's and G3's on the Melbourne 2531 strip, about 640 m before;
  - 1 is G3's on the Sochi strip, 783 m before.
- **159 of those 160 leave the road on the inside edge.**
- The total width at the crash is typical of the corner: median 12.1 m against the corner's median 12.0 m. Only 31 of 160 crash where the road is more than 0.25 m narrower than the corner's median.
- The inside width is moving. The crash-side width varied by at least 1 m over the last 100 m in 146 of 160 crashes, and by at least 2 m in 107. The median range is 2.6 m.
- The crash-side width at the crash was more than 0.25 m below that side's corner median in 90 of 160.
- Per corner, the median crash-side width range over the last 100 m runs from 0.5 m (Brands Hatch 1136) to 3.2 m (Norisring 874, Yas Marina 2817). The total-width range reaches 5.4 m at Sochi and Oschersleben.
- Context: transplants use the track the R-dev evaluation drives (protocol 3.2), whose per-side widths are measured from the refined reference line. On 6 of the 12 R-dev circuits that line moves toward hairpin apexes, so one side narrows while the other widens (`src/apexzero/tracks/generator.py` docstring; `tests/test_real_widths.py`).

**Training roads for comparison.** The default training generator varies width only slowly: an R-dev circuit's median width with low-frequency variation of wavelength 300–1,500 m, amplitude 1.5 m in total width and 0.5 m in asymmetry, plus the inside cap (`generator.py`). The method was:
- Generate 300 default strips (grammar fitted on R-dev, PRNG keys 10000–10299).
- At every knot of a tight corner (|κ| ≥ 1/50 m⁻¹), take the range of the inside-edge width over the 100 m up to that knot.
- Do the same at the corner knots of the 15 real crash roads.

| Roads | Knots | Median range | Range ≥ 1 m | Range ≥ 2 m |
|---|---|---|---|---|
| Training strips, v1 grammar | 88,054 | 0.23 m | 0.4% | 0.0% |
| Training strips, v3's tilted grammar | 233,206 | 0.23 m | 0.5% | 0.0% |
| Real crash roads (E1 real rung) | 302 | 1.78 m | 79% | 40% |

**Reading (descriptive).**
- The total width at the crash is typical of the corner, but the crash-side (almost always inside) width is not. It is more than 0.25 m below that side's corner median in 90 of 160 crashes, and it has moved by at least 1 m over the last 100 m in 146.
- The agents cut the inside of corners whose inside edge moves by metres within the last 100 m. Changes that large almost never occur in a tight training corner.
- Holding the width constant removes most of these crashes. But the constant-width rung changes the width level, the left-right asymmetry and the variation at once, so the ladders cannot say which of the three matters.

### 6.3 Mirror asymmetry in the ladders

- The mirror rung is the exact mirror of the real rung, road and initial state together. Lateral states, the steering-related states and the left and right widths are all mirrored (checked on `stimuli/cells.json`).
- The harness mirror gate was exact when it ran on 19 Oct: maximum |Δ| of 0.0 over 400 steps, with mirrored noise (`runs/psychophysics-rollouts.log`, step 2). A left-right symmetric policy would therefore complete a real rung and its mirror equally.
- The generalists do not:

  | Agent | Real (of 84) | Mirror (of 84) |
  |---|---|---|
  | G1 | 13 | 24 |
  | G2 | 26 | 48 |
  | G3 | 36 | 40 |

- Real and mirror completion differ by at least 50 points on 5 of 15 corners for G1 (mirror better on 3), 5 of 15 for G2 (mirror better on 4) and 4 of 15 for G3 (2 each way). Among the replicates the count ranges from 1 to 5 of 15.
- The real crash roads are mildly lopsided: the median |w_left − w_right| over [s_c0 − 300, s_c1 + 100] is 0.27 m, with a maximum of 0.74 m.
- On the same window the median is 0.00 m on the constant-width, isolated and twin rungs and on every synthetic stimulus. Locally, not all of them are exactly symmetric (`stimuli/bank.npz`):
  - 28 of the 160 synthetic strips have left-right differences of up to 1.4 m. They come from the inside cap (R 6.0 and 7.5 m, and R 9.4 m at width 15 m) and from tier E3's `win4.6` variants.
  - The constant-width, isolated and twin rungs have local differences of up to 0.48 m.
  - Each mirror pair is an exact mirror of its partner.
- So on real roads these networks behave asymmetrically, by an amount that changes outcomes. The pre-registered mirror control agrees: P5 fails for 8 of the 9 generalists and for S1 (Section 4.5).
- The protocol's response to a P5 failure is to re-run the harness mirror gate, and to call the asymmetry learned only if the gate still passes. The re-run passes, and the post-hoc layout check rules out unequal A2 starts as the cause (Section 7.2). The asymmetry is learned.

## 7. Deviations log

### 7.1 Before the tag: rulings R1–R31 (all pre-registered)

Every ruling below was written into protocol Section 12 and committed before the tag (the R30 and R31 commits precede `1338394`). Each was decided before any non-pilot agent saw a protocol stimulus. They are part of the design, not deviations, and are summarised by category:

| Category | Rulings | Substance |
|---|---|---|
| Agent runs allowed before the tag | R1 | Real-lap rollouts for initial states repeat the standard R-dev evaluation. 35 are logged in the manifest, all before the tag |
| Scope: replicate agents and stability | R18, R19 | Seeds 1 and 2 of each generalist variant join as prospective agents, and variant claims need 2 of 3 seeds. History snapshots also run tier G so the gate can judge them. 6.3 covers replicates. Stability compares decision classes |
| Estimators and definitions | R2, R3, R4, R4a, R5, R6, R7, R8, R9, R12 | The crash corner is the containing segment and P8 thresholds are proportional, leaving G2 with 3 tail units (R2). θ_eff, twin-length and twin-angle tolerances are accepted and listed (R3, R4, R4a). l_typ uses corners of at least 10° (R5). The unsettled flag is off for cells with the corner in view at the start (R6). Budget-rule count (R7). The crash-typing search starts at s_c0 − 50 (R8). Clip-radius sign (R9). Re-anchored keys for below-support cells (R12) |
| Analysis decisions | R10, R11, R13 | Vacuous class 2 is flagged (R10). P6a also needs p ≤ 0.05 (R11). Analysis decisions AD1–AD25 signed off (R13) |
| Validation outcomes and key exactness | R14, R15, R16, R17, R17c, R25a | Q tuning rule (R14). V2, V5, V7 and V9 failures handled as the protocol prescribes, with b = 10% (R15). 523 P0 keys, 0 unsolved, 7 multi-basin, 3 V6 violations listed (R16). Per-phase re-anchored keys and 16-phase keys on P7 cells (R17). Tier B keys at D = 600 (R17c) and D = 250 (R25a) |
| Dry-run calibrations | R20–R29 | Float64 tape check (R20). Phase starts on real laps (R21). Q's pedal law (R22). c1 is 15 m late against the key (R23). c2 and P1b's power statement (R24). P3 calibrated to the key's own slope b* (R25). c3 is a pure trigger, with the hybrid power statement (R26). F6 halved; K validates class 1 only (R27). Viability solves for c1 (R28). Implementation notes (R29) |
| Crash typing | R30, R31 | Overspeed is defined causally: a crash is overspeed if some tested state is unrecoverable, faster than the key, and recoverable at the key's speed. Notes from the reviews (R31) |

### 7.2 After the tag (dated)

| Date (EDT) | What | Effect on results |
|---|---|---|
| 2026-10-20 19:38 → 2026-10-21 01:02 | **6.3 "agent beats the key" re-solves.** On cell 476, the null straight at width 12 and 280 km/h, S1's deterministic return (83.29153) exceeds the null key's J* (83.29084) by 6.9e-4, which is 8.2e-6 relative and above 6.3's 1e-6 threshold. Each of the six local `anchored-keys` invocations (Section 7.3) checked the finals whose rollouts were on disk and re-solved the cell from them. The six `manifest.json` entries (`deviations`) list all 12 finals (19:38, 22:02, 01:01), S1 only (21:37, 23:25), and G1–G3 with S1–S3 (21:38). The first replaced the key with one better by 5e-10; the others found nothing better. S1 still beats this key, and replicates alone never triggered a re-solve. Any 6.3 checks run on the rented machine (below) are not recorded | Null keys back catch-trial onsets, corner regret and corner time. No confirmatory decision here used them (P2 was not applicable) |
| 2026-10-20 19:38 → 2026-10-21 01:29 | **Anchored-keys stage: six local invocations, with the job lists steered by moving rollout files.** The stage builds its job list from the rollout files on disk. Rollout files were moved between `outputs/psychophysics/rollouts/` and `outputs/psychophysics/rollouts_parked/` outside the harness (no code refers to that directory). `rollouts_parked/` was created at 21:37:24, 12 s before an invocation that saw only S1's rollouts. It was emptied at 23:58:06, 18 s after the second such invocation finished. Every agent rollout directory except S1's was last modified at 23:58:04–06 (S1's at 19:53). Section 7.3 lists the six invocations | None expected on key values: the frozen code is unchanged (`git diff psychophysics-prereg-v2 -- src/apexzero/interp scripts/psychophysics.py` is empty), and each key is cached under a hash of its inputs (protocol 6.2). But the stage did not run as one pass, and its job selection was steered outside the harness |
| 2026-10-20 19:41 | **P6's exception decided on no data, then the other way.** The first invocation logged "re-anchoring immaterial on the finals (frac 1.0, n 159): nominal keys for P6". No re-anchored key existed yet (the first in the cache was written at 20:17:33). A run without one is compared against the key behind its own e_ve (`analysis.py`, R17a fallback), so the difference was 0 by construction. Every later invocation that had generalist rollouts (21:38, 22:02, 01:01) found the exception not met and selected P6c checkpoints. The manifest records the last invocation's `p6: "checkpoints re-anchored"` (`validation.anchored_keys`), and that is the determination in force | The exception's outcome depends on whether the finals' re-anchored keys exist when it is evaluated, that is, on the order of runs. It affects P6c only, a secondary analysis (not applicable for G1–G3, Section 4.5). No confirmatory decision uses it |
| 2026-10-20 19:45 → 2026-10-21 01:56 | **Eight commits after the tag.** Five are dashboard-only (`scripts/dashboard.py` and its tests): `cb3b198` 19:45, `67d7e07` 21:32, `00f4808` 21:40, `bb93d0d` 23:50, `3aaf1bb` 00:23. `83ebe56` (00:20) adds the opt-in real-width generator option of [width-test.md](width-test.md), in `src/apexzero/tracks/generator.py`, `src/apexzero/env/adapters.py` (a guard in `generate_bank`), `scripts/train_dr.py` and a test. `2143a59` (01:54) adds `--real-widths-from` (`generator.py` again, `train_dr.py`, a queue script, a test and the leave-out plan). `00c1980` (01:56) adds the leave-out analysis script. The frozen-code hash (`src/apexzero/interp` and the script) is `8650965a…` at every commit. The broad code hash (all of `src/apexzero` and the script) went `b4072de7…` → `b82c7730…` (`83ebe56`) → `cb025fa4…` (`2143a59`) | Probably none, but not none by construction. The harness imports `generator.py` (`strip_arrays`, `to_track`) and `adapters.py` (`reference_speed`). Stimuli and rollouts predate `83ebe56`. The last anchored-keys invocation (01:01–01:29) and both `analyse` runs (01:31 and 01:45) ran on its tree, and the 20 keys that invocation solved carry `b82c7730`: 4 nominal keys (cells 763, 1044 and 1100 verified, 964 unsolved) and 16 viability keys (G3s1). With the option off the generator is bit-identical (`tests/test_real_widths.py::test_default_is_unchanged`), and the adapters guard only raises when exactly one of two grammars has real widths. No psychophysics stage ran after `2143a59` |
| 2026-10-20 20:52 → 2026-10-21 00:46 | **Part of the anchored-keys stage ran on a rented cloud GPU box, which the local log does not record.** In the key cache, 5,939 of the 9,621 re-anchored keys, 287 viability keys and 4 nominal keys carry the tagged code hash `b4072de7…` and `created` stamps of 21 Oct 00:52–04:46 (`outputs/psychophysics/keys/*/*.json`). These stamps are UTC. The later ones are ahead of the local clock at the time of writing, and the local machine's only invocation in the 00:52–01:45 window ran on a later tree and wrote a different code hash. In EDT the box worked 20:52–21:45 and 23:48–00:46, and no record was created between 21:45 and 23:48; why is not recorded. The box solved every P6c checkpoint key, all of S2's and S3's keys, and part of every other final's. S1's 768 re-anchored keys were all solved locally. What was copied to the box, and how its cache was merged into the local one, were not recorded. The box's stage log, and any manifest entries it wrote (6.3 checks included), are not among the local outputs. Where both machines solved the same key id, the merged cache holds one record, and which machine's cannot be told | None expected on key values: the same tagged code, and keys cached under a hash of their inputs. The confirmatory analysis read the merged cache. The gap is in provenance: the box's own checks cannot be audited |
| 2026-10-21 01:31 | **S2 and S3 kNN d_OOD buffers missing at the confirmatory run.** The implementation map (D33) planned them for after the tag, but they were not built before `analyse --confirmatory`. P4 therefore returned "not testable" for S2 and S3 and "inconclusive" for the recipe claim (`runs/psychophysics-analyse.log`, 01:41:34 and 01:41:58). The buffers were built at 01:43:50–01:44:01 (`manifest.json`, `lists.post_tag_buffers`), and the non-confirmatory `analyse` (01:45–01:48) re-decided P4 with them | The confirmatory table stands as written. The rerun goes in Section 4.4 and is labelled as post-tag |
| 2026-10-21 01:45–01:48 | **The non-confirmatory `analyse` rewrote shared analysis files.** It rewrote `analysis/key_table.npz` (01:45:51) and the per-agent pairs, cell-rate, R50 and support tables (01:48:12). The confirmatory run's own files are `decisions.json`/`.npz`, `replication.json` and `replication_*.npz` (01:42:39). The non-confirmatory run did not touch them; it writes `decisions_all.*` and `replication_all*` | Section 2's count of unsolved tier D keys is read from the rewritten key table. The confirmatory decisions are unaffected |
| 2026-10-21 02:45 | **P5's harness-gate re-run, as the protocol requires after a P5 failure.** `scripts/mirror_gate_rerun.py` runs the frozen `rollout.mirror_gate` on the same two pairs as the pre-tag gate (tier A R 22.9 m / 90° at A1, R 44.7 m / 90° at A2). Both give a maximum \|Δ\| of 0.0 over 400 steps, deterministic and with mirrored noise (`outputs/psychophysics/post/mirror_gate_rerun.json`). As an extension it tried every other tier A pair. All pairs at A1 pass. At A2, 43 of the 68 pairs cannot be gated, because their two members start at different places: each member's start distance D is laid out from its own key, and the mirror shortcut is off (V5). The offsets are under 0.1 m for 23 pairs, under 5 m for 16, and 10 m or more for 4 (R 6.0 and 7.5 m at 180°, and R 22.9 m / 90°, whose members start 250 m and 41.5 m before the corner). Every pair outside A2 is an exact mirror | The gate passes, so by the protocol's rule P5's asymmetry is learned. The unequal A2 starts are a property of the frozen layout, not of the harness |
| 2026-10-21 02:47 | **Post-hoc check of P5 against the A2 layout (not pre-registered).** `scripts/p5_layout_check.py` recomputes P5 with the frozen `analysis.p5` on the cached run tables. With all pairs it reproduces every recorded P5 number. Dropping the pairs that start 5 m or more apart, or keeping only exactly mirrored pairs (197–200 of about 240 pairs per generalist), changes no verdict. The generalists' median \|Δd_b\| moves by at most 0.4 m (`outputs/psychophysics/post/p5_layout_check.json`) | P5's failures are not caused by the layout |

**Checks with nothing to report.**
- `budget.cuts` is empty, so no budget rule fired.
- The manifest's `stale_key_measures`, `key_errors`, `validation_errors` and `anchored_errors` lists are empty.
- No log mentions stale rollouts.
- Section 9.4's clipping sensitivity analysis never applied. No run had clipping in the edge, ego or context groups, so no decision carries a `without_clipped_runs` result.

`manifest.json`'s `deviations` list contains only the 6.3 entries. Every other row is recorded here, as the protocol's freeze rule asks.

### 7.3 Anchored-keys stage: timeline and counts

Local invocations (`runs/psychophysics-anchored-keys.log`). The finals present come from each invocation's 6.3 entry in `manifest.json` (`deviations`).

| Start (EDT) | Finals on disk | Jobs selected | Outcome |
|---|---|---|---|
| 20 Oct 19:38:34 | All 12 | 1,810 | No "done" line. Last progress was 307 of 1,810 at 21:34:56. Logged the vacuous P6 determination at 19:41:17 |
| 21:37:36 | S1 only | none | Ended after the pre-flight (21:38:17), before any job. P6c checkpoints: none |
| 21:38:48 | G1–G3 and S1–S3 (no replicates) | 1,297 | Last progress was 232 of 1,297 at 22:02:24 |
| 22:02:56 | All 12 | 2,176 | Last progress was 487 of 2,176 at 23:25:19 |
| 23:25:49 | S1 only | 143 | Done, 143 of 143, at 23:57:48 |
| 21 Oct 01:01:41 | All 12 | 2,176 | Done, 2,176 of 2,176, at 01:29:54. No re-anchored key carries its code hash: every re-anchored job was a cache hit. It solved 4 nominal and 16 viability keys (Section 7.2) |

**Reconciliation.** The last invocation's log total of 2,176 is the 2,171 jobs that `manifest.json` records (`validation.anchored_keys`: re-anchored and viability jobs, 0 errors), plus the 6.3 re-solve, plus 4 nominal keys solved before the jobs (`viability_nominal_inline`: cells 763, 964, 1044, 1100).

**Who solved what** (`outputs/psychophysics/keys/anchored/*.json`, `created` and `code_hash`):
- Local machine: 3,682 re-anchored keys (20 Oct 20:17–23:57), all for finals, S1's 768 among them.
- Rented machine: 5,939 re-anchored keys:
  - G1's checkpoints at 73.7M, 109.9M and 134.1M (160 each);
  - G2's at 134.1M and 163.8M (168 each);
  - G3's at 109.9M, 134.1M and 163.8M (192 each);
  - S2's 404 and S3's 393;
  - part of each other final.

**Unsolved re-anchored keys** (`status`). 96 of 9,621 are unsolved:
- finals: G1 19, G2 14, G3 9, G1s1 10, G1s2 15, G2s1 2, G2s2 21, G3s2 1, S1 2;
- checkpoints: 1 each at G1 109.9M, G2 163.8M and G3 134.1M.

The affected runs fall back to the nominal key at their phase (protocol 6.1; R12, R17a). P1b's `ra_fallback_pairs` (Section 4.1) count such pairs.

## 8. What next

1. **Step 2: receptive fields.** Measure which preview inputs drive braking and steering through closed-loop sensitivities, Jacobians and counterfactual edits, as the protocol's step 2 plans.
   - The crash ladders give it a target: what the edge and width channels contribute near width transitions at the inside edge, against the curvature channels. Synthetic width-transition stimuli would extend tier E3.
   - **This departs from the protocol.** Protocol 9.6 says that a low F for G2's tail crashes, the outcome here, means step 1b's two-corner stimuli are prioritised. The reason for proposing step 2 first is that the necessary context factor in the ladders was width, not a neighbouring corner. If adopted, the change of order will be logged as a deviation.
2. **Idea 2: costate.** Test whether the networks' hidden activations encode the optimum's costate. The answer-key cache stores the solver's multipliers for every verified key.
3. **Width test: leakage check.** Training on real width profiles raised R-dev completion ([width-test.md](width-test.md)), but those profiles come from the same 12 circuits used for evaluation. A leave-circuits-out test, specified in [width-test-loo-plan.md](width-test-loo-plan.md) and committed with its analysis script before its runs, found that about half of the in-sample gain transfers to circuits whose widths were not seen (p = 0.063; [width-test.md](width-test.md), Section 7).
4. **Protocol steps after the confirmatory run, now done.** The harness mirror gate re-run and the P5 layout check (Section 7.2), and the P4 rerun write-up (Section 4.4).

## Not covered here

From the non-confirmatory run (`outputs/psychophysics/analysis/decisions_all.json`), the following are not written up:
- the descriptive and exploratory outputs P2d, P3 ratio, P6d and P8-F;
- the per-checkpoint P6 curves.

Plots are in `outputs/psychophysics/analysis/plots/`.
