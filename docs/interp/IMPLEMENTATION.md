# Psychophysics step 1: implementation map

This file goes with `docs/interp/psychophysics-protocol.md` (v2). It covers:
- which function implements each protocol section;
- the data contract shared by the module builders;
- decisions where the protocol is ambiguous;
- mismatches between the protocol's assumed APIs and the repo.

**Rules every builder follows:**
- New files only: `src/apexzero/interp/*.py`, `scripts/psychophysics.py`, `tests/test_interp_*.py` and this file. The training queue imports `src/apexzero` at each job start, so no other file changes. A repo function that needs changing is copied into `interp/`, and the copy is noted here.
- Lockbox: `lockbox_guard(name)` runs before any named track is loaded. Nothing is ever loaded from `R_TEST` or from `outputs/sealed/`.
- Freeze rule: no trained agent is rolled out on any stimulus. Tests use scripted controls or randomly initialised networks. One forward pass on a dummy observation is allowed.
- Compute: each check stays under about 3 CPU-minutes, with XLA at 2 threads. Only validation-sized solves run in tests. Long stages run later through `scripts/psychophysics.py`.
- Precision: keys run in float64, and `replay` switches x64 on globally. Rollouts run in float32, and `io.assert_x64_off()` checks this. The two never share a process.

## 1. Module dependency order

```
constants ─► io ─► support ─► stimuli ─► measures ─► rollout ─► agents ─► controls ─► answer_key
                                                         └──────────────► probe
                                              (measures, answer_key, support) ─► analysis
registry (no interp import) ◄── support, stimuli, rollout, agents, analysis (ruling R18)
scripts/psychophysics.py imports everything lazily, inside each stage function.
```

| Module | Role | Status in the skeleton |
|---|---|---|
| `constants` | Every fixed number in protocol 1.4, 3.5, 4, 5, 6 and 10 | Complete |
| `registry` | `AgentSpec`, `AGENTS` (S1-S3, G1-G3, R18's six replicates), `GrammarSpec`, `GRAMMARS`, the id helpers | Added for ruling R18 (see "Ruling R18") |
| `io` | Paths, column tables (.npz), JSON, hashes, seed streams, `StageLog`, `assert_x64_off` | Complete, with tests |
| `support` | Training corner tables, `l_typ`, `v_typ`, regimes, `R_clip`, kNN density | Implemented (§9) |
| `stimuli` | `CornerSpec`, profiles, `StimulusBank`, `Cell`, transplants, ladders, twins, tier D, the design | Implemented (§9) |
| `measures` | `ModelTables`, `front_brake_demand`, `Trajectory`, `Marks`, adapters, `ResponseMetrics`, `KeyErrors`, `crash_type` | Implemented (§8) |
| `rollout` | `Policy`, initial states, phases, `RolloutBatch`, `Trace`, the harness, npz I/O, step-2 checks | Implemented (§7) |
| `agents` | `AGENTS`, `load_agent`, `Manifest`, `freeze_manifest`, real laps | Implemented (§7) |
| `controls` | T, Q, Qp, c1–c3, K as Policies; `QParams`; c_v tuning; kNN buffer | Implemented (§ controls/probe notes) |
| `answer_key` | `solve_min_time_pinned`, `AnchoredSolver`/`solve_from_state`, recipe, every key type, cache, V1–V11 | Implemented (answer_key §9) |
| `probe` | Tier O scans, Π/A, A_alias | Implemented (§ controls/probe notes) |
| `analysis` | Run, pair and condition tables; P1–P8; intervals; dry run; power table | Implemented (§ analysis notes) |
| `scripts/psychophysics.py` | Every stage of protocol 10 | Implemented (§ integration); steps 0–3 run |

## 2. Protocol → implementation

| Protocol | Definition | Implemented by |
|---|---|---|
| 1.4 | Every fixed choice | `constants` |
| 1.4, 2.3 | Single radius estimator R_seg | `corners.segment` defaults, called by `support.corner_table`, `stimuli.realised_geometry`, `stimuli.crash_corner` |
| 1.4 | l_typ(R) | `support.l_typ_table`, `support.l_typ` (from `support.generate_with_corners`) |
| 2.2 | e_ve, e_ve^bo, e_ve^RA | `measures.key_errors` (nominal); the run-table columns `e_ve_bo` and `e_ve_ra` in `analysis.run_table` |
| 2.2 | Band membership | `analysis.band_side` |
| 2.2 | S, its estimability, leave-one-R-out | `analysis.sensitivity` → `SlopeFit` |
| 2.2 | Extrapolation residual r | `analysis.extrapolation_residual` |
| 2.3 | q1/q10 and regimes | `support.regime_table`, `support.regime` |
| 2.3 | R_clip(k) | `support.r_clip` |
| 2.3 | κ-clipped (binding, at run time) | `measures.ResponseMetrics.kappa_clipped`, from `Trace.clip_frac` |
| 2.3 | Joint support density | `support.support_features`, `support.knn_log_density`, `support.stimulus_v_peak` |
| 2.3 | Braking or flat | `analysis.pair_values` (`braking` column: the key has a sustained brake and v_min* < v0 − τ) |
| 2.4 P1a, P1-tail, P1b, P1c | Decision rules | `analysis.p1a`, `p1_tail`, `p1b`, `p1c`, `r_dep` |
| 2.4 P2, P2b | Late braking; onset/depth split | `analysis.p2`, `analysis.p2b_decompose` |
| 2.4 P3 | Braking law, model comparison, separation check | `analysis.p3_fit` → `P3Result`, `analysis.p3`, `analysis.p3_separation` |
| 2.4 P4 | Specialist classes, φ, m, d_OOD, MI | `analysis.p4_classify` → `P4Result`, `phantom_index`, `swap_index`; `controls.d_ood`, `controls.KnnBuffer` |
| 2.4 P5 | Gates and symmetry | `rollout.mirror_gate`, `answer_key.v5_mirror`, `analysis.p5` |
| 2.4 P6 | Competence, t_R, P6a–P6d | `analysis.t_r`, `analysis.p6`, `analysis.permutation_spearman` |
| 2.4 P7 | Phase SD, nulls, A_alias, tier O | `analysis.p7`, `analysis.p7_null`, `probe.aliasing_amplitude`, `probe.open_loop_scan` |
| 2.4 P8 | Ladder and twin agreement | `analysis.p8`; `stimuli.ladder`, `stimuli.twin_fit` |
| 3.1 | Trapezoid κ, widths, inside cap, E3 narrowing | `stimuli.corner_spec`, `corner_profile`, `apply_inside_cap` |
| 3.1 | Landmarks | `stimuli.Landmarks` (synthetic: from `CornerSpec`; real: see decision D14) |
| 3.1 | Realised geometry | `stimuli.realised_geometry` → `Geometry` |
| 3.1 | Arrays and Tracks | `StimulusBank.arrays()` (vmapped `strip_arrays`), `StimulusBank.track()`, `stimuli.stimulus_track` |
| 3.2 | Transplants, fidelity gate inputs | `stimuli.real_track`, `stimuli.transplant` → (`Profile`, `Excision`) |
| 3.2 | YasMarina diagnosis | `rollout.run_rollouts` on the real closed track and the transplant (Integrate phase) |
| 3.3 | Tier D conditions | `stimuli.hockenheim_conditions`, `stimuli.swap_partner`, `D_CONDITIONS` |
| 3.4 | E1 ladders, twins, E2 matches, E3 | `stimuli.ladder`, `twin_fit`, `matched_corners`, `CRASH_PAIRS`, `crash_corner`; E3 through `corner_spec(w_in_corner=…)` |
| 3.5 | The grid, cells and counts | `stimuli.build_design`, `stimuli.build_real_design`, `stimuli.tier_o_specs` |
| 3.5 | A2 layout (D and v_typ) | `answer_key.layout_a2`, `support.v_typ` |
| 4 | Synthetic initial state | `rollout.approach_state`, `Cell.initial_state`, `rollout.init_env_state` |
| 4 | Phases and v_sight | `rollout.phases`, `rollout.sight_speed`, `Cell.phase_offsets` |
| 4 | Window W | `answer_key.window_length` |
| 4 | Policy modes and common noise | `rollout.make_stimulus_rollout`, `rollout.noise_key`, `RolloutBatch` |
| 4 | Recorded quantities | `rollout.Trace`, `rollout.save_rollouts` (layout in its docstring) |
| 4 | Tier O states | `probe.scan_state` |
| 5 | Model tables | `measures.build_model_tables` → `ModelTables` |
| 5 | Brake demand β_f | `measures.front_brake_demand` |
| 5 | Every measure | `measures.response_metrics` → `ResponseMetrics`; pieces in `brake_onset`, `response_onset`, `line_eta`, `path_curvature`, `first_sight`, `oscillation_index`, `crash_cause` |
| 5 | Errors against a key | `measures.key_errors` → `KeyErrors`; `measures.delta_v` |
| 5 | Trajectory adapters | `measures.trajectory_from_rollout`, `_from_window`, `_from_replay`, `_from_socp` |
| 6.1 | Nominal, backed-off, phase, null, re-anchored, viability, actuation keys | `answer_key.full_recipe`, `warm_key`, `backoff_problem`, `null_key`, `anchored_key` (one per deterministic phase, from `anchor_base` and `anchor_for`: R17a), `viability_keys`, `actuation_key` |
| 6.1 | `solve_from_state` | `answer_key.AnchoredSolver.solve(a0_fixed=…)`, `answer_key.solve_from_state` |
| 6.2 | Pinned s-OCP | `answer_key.pinned_strip`, `solve_min_time_pinned`, `socp_from_state`, `socp_end` |
| 6.2 | Guess, stages and starts | `answer_key.socp_guess`, `extend_straight`, `stages_for`, `retime`, `make_solver` |
| 6.2 | Cache | `answer_key.KeyCache`, `key_id`, `key_for`, `mirror_key`; records `KeyRecord`, `StartRecord`, `SocpKey` |
| 6.3 | Near-equal set, multi-basin cells, τ, onset floor, an agent beating the key | `answer_key.near_equal`, `multi_basin`, `onset_floor`, `agent_beats_key` |
| 6.4 | Active sets, infeasibility bound | `answer_key.active_set` → `ActiveSet`; `answer_key.infeasibility_bound` → `Bound` |
| 6.5 | V1–V11 | `answer_key.v1_replay` … `v11_active_set` → `ValidationResult` |
| 7 | Controls and required verdicts | `controls.make_q_policy`, `make_tape_policy`, `make_knn_policy`, `CONTROLS`, `T64`, `tune_cv`, `tau_c_k1`, `c1_shifts`, `c2_planted_law`, `c3_checks`, `knn_fallback`, `offline_prediction`; `rollout.run_tape_rollouts_x64` (R20) |
| 7 | Dry run and power table | `analysis.simulate_pairs`, `analysis.power_table` |
| 8 | Agents, loading, manifest, gate | `registry.AGENTS` (re-exported as `agents.AGENTS`), `load_agent`, `freeze_manifest`, `Manifest`, `SnapshotEntry`; gate via `ResponseMetrics.osc_index` and completion |
| 9.1 | Pair values, crash typing | `analysis.pair_values`, `measures.crash_type`, `answer_key.no_return` → `measures.NoReturn` |
| 9.2 | Beta-binomial, Clopper–Pearson, permutations, bootstrap | `analysis.beta_binomial`, `clopper_pearson`, `permutation_spearman`, `bootstrap_band` |
| 9.3 | R50 and Firth | `analysis.r50`, `analysis.firth_logistic` |
| 10 | Stages, logging, budget | `scripts/psychophysics.py`, `io.StageLog`, `answer_key.project_budget` |

## 3. The shared data contract

### 3.1 Stimuli (`stimuli.py`)

**Strip layout.** Every stimulus strip has `N_KNOTS = 2560` knots at `DS = 2.0` m, with its corner start at `S_C = 1000` m. One bank therefore holds the whole design with no padding.

**`StimulusBank`** has two parts:
- `bank.npz`: `kappa`, `w_left` and `w_right`, each `[B, 2560]` float64 knot samples.
- `stimuli.json`: one `Stimulus` per row. The row index is the `stim_id`, and it is also the `track_id` used in rollouts.

`bank.arrays()` returns the stacked float32 `TrackArrays` that agents use, as in training. `bank.track(i)` returns the NumPy `Track` (`to_track` of the float32 arrays) that the TO-OCP, s-OCP, replay and QSS use.

**`Stimulus`** fields: `stim_id`, `name`, `kind`, `tiers`, `direction`, `mirror_id`, `landmarks`, `geometry`, `spec`, `source`, `flags`. Names follow `syn/R{R}/T{θ}/W{w}/{ramp_kind}[/nocap][/win{w}]/{L|R}`.

**`CornerSpec`** fields:

| Field | Meaning |
|---|---|
| `R` | Nominal radius (m) |
| `theta_deg` | Turning angle (deg) |
| `direction` | +1 left, −1 right |
| `width` | Total width (m) |
| `l_r` | Ramp length (m), resolved when the spec is built |
| `ramp_kind` | `typical`, `abrupt`, `pilot` or `twin` |
| `no_cap` | Inside-width cap switched off |
| `w_in_corner` | E3 inside half-width at the corner (m) |
| `s_c` | Corner start (m) |

Derived properties: `theta`, `kappa_peak`, `L_p`, `L_c`, `is_triangle`.

**`Landmarks`**: `(s_c0, s_a0, s_apex, s_a1, s_c1)` in strip metres.

**`Geometry`**: `R_eff`, `theta_eff_deg`, `onset_rate`, `R_seg`, `theta_seg_deg`, `L_seg`, `ringing`, `s_kmax`, `n_segments`, `cap_binds`, `w_in_min`.

**`Cell`** is one stimulus at one approach, or from one real initial state. It is saved in `stimuli/cells.json`.

| Field | Meaning |
|---|---|
| `cell_id`, `stim_id` | Identity |
| `pair_id` | Shared by the left and right members; also the noise id |
| `tiers`, `approach` | Design labels |
| `v0` | m/s |
| `D` | m; `None` for A2 until `layout_a2` sets it |
| `W` | Steps |
| `P` | Phase cycle (m) |
| `n_phases` | 4, or 16 on P7 cells |
| `x0`, `prev_action0`, `accel0` | Real initial states, in strip coordinates |
| `state_source`, `null_cell`, `flags` | Provenance and links |

Methods: `phase_offsets()` returns u_k; `initial_state(k)` returns a float32 x0.

### 3.2 Rollouts (`rollout.py`)

**Policy signature.** `policy(params, o, state, tr, aux) -> (mean[2], log_std[2])`.
- For agents, `o` is the normalised, clipped observation.
- For controls, `o` is the raw observation (the harness is called with `obs_rms=None`).
- `aux` is an int32 row into a per-run controller table: the tape row for T, the target-profile row for Q.

**Harness.** `make_stimulus_rollout(policy, steps, record)` returns `jit(fn(policy_params, obs_rms|None, bank_arrays, RolloutBatch) -> Trace)`.

**`RolloutBatch`** fields, each with a leading `[B]`: `track_id`, `x0[8]`, `prev_action[2]`, `accel[2]`, `ctx[6]`, `hidden_mu`, `noise_key[2]` (uint32), `noise_std[2]`, `native`, `steer_sign`, `aux`.

**Noise.**
- run key = `fold_in(fold_in(PRNGKey(20261019), pair_id), sample)`
- ε_t = `normal(fold_in(run_key, t), (2,))`
- On the mirrored member, ε_steer is multiplied by `steer_sign`.

**File layout.** One npz per snapshot at `rollouts/<agent>/<timesteps:012d>.npz`, with groups `det_*`, `nat_*` and `mat_*`. The exact arrays, shapes and dtypes are in the `rollout.py` module docstring.
- `valid[k]` is True for every state actually reached, including the terminal state.
- `flags` uses `constants.FLAG_BITS`.
- `log_std` is stored once per snapshot.

### 3.3 Measures (`measures.py`)

**`Trajectory`** holds:
- `t[K]`, `x[K,8]` and `valid[K]`;
- per-step arrays `a`, `a_mean` and `reward`, each of length K−1;
- `util_f`, `util_r` and `beta_f[K]`;
- `flags[K−1]`, `term_step` and `trunc_step`;
- `clip_frac` and `clip_max[K,4]`, `log_std`, `dt`, `prev_action0`, `meta`.

Channels a source does not record are NaN. Stochastic traces record only s, vx, n, pedal, util and β_f.

**`Marks`**: `landmarks`, `direction`, `s0`, `W`, `w_left[N]`, `w_right[N]`, `triangle`.

**Metrics.** `response_metrics(traj, marks, null, tables) -> ResponseMetrics`. Its field names are the run-table columns (`METRIC_COLUMNS`).

**Key errors.** `key_errors(...) -> KeyErrors` (`KEY_ERROR_COLUMNS`). `crash_type(traj, marks, key, NoReturn)` returns `overspeed`, `overbrake`, `line_steer` or `other`.

### 3.4 Keys (`answer_key.py`)

**`KeyRecord`**: one per TO-OCP key, stored as `keys/<kind>/<key_id>.json` plus `.npz`. Kinds are `nominal`, `backoff`, `phase`, `null`, `anchored`, `viability` and `actuation`.
- Arrays: `x[W+1,8]`, `a[W,2]`, `lam_g`, `lam_x`, `active_steps[W+1,5]`, `x0[8]`.
- JSON: status, J, replay, starts (`StartRecord`), near_equal, logged, spread, multi_basin, tau, measures, active flags, anchor, socp_id, CPU time, flags.

**`SocpKey`**: the pinned s-OCP. It holds an `OCPResult` with s in strip coordinates and t = 0 at the pin.

**Cache.** `key_id` = sha256 of (stimulus samples, x0, W, car parameters, spec, kind, extras). `KeyCache.content_hash()` hashes array contents, not file bytes, so the manifest's key-cache hash is reproducible.

### 3.5 Tables (`analysis.py`)

Tables are column dicts saved as `.npz` through `io.save_table`. The column sets are fixed:
- `RUN_COLUMNS` = run index + `METRIC_COLUMNS` + `KEY_ERROR_COLUMNS` + `analysis.RUN_EXTRA_COLUMNS` (`e_ve_bo`, `e_ve_ra`, `key_at_phase`, `crash_type`, the key values `v_e_star_ra`, `v_e_star_k`, `v_b_star_k`, `d_b_star_k`, the key-choice flags `ra_key`, `speed_at_phase`, `bo_at_phase`, and the re-anchored key behind each run, `ra_phase` and `ra_key_id` (R17a));
- `PAIR_COLUMNS`;
- `COND_COLUMNS` (tier D);
- `DECISION_COLUMNS`.

Prediction thresholds are module constants in `analysis.py`.

### 3.6 Manifest (`agents.py`)

`Manifest` is written as `outputs/psychophysics/manifest.json`.

| Field | Contents |
|---|---|
| `version`, `created`, `updated`, `root_seed` | Header |
| `git_commit`, `code_hash`, `stimulus_hash`, `key_cache_hash`, `tag` | Hashes and tag |
| `agents` | One entry per agent: every `AgentSpec` field (variant, seed and hardware included), finished, snapshot count, final, status; a replicate also `replicate_of` and `ruling` (R18) |
| `snapshots` | List of `SnapshotEntry`: path, timesteps, is_final, sha256, size, log_std, std, prospective, evaluated_on_stimuli, first_evaluated, forward_pass_ok, obs_rms hash, gate, error |
| `support` | Paths, hashes and frozen regimes |
| `band`, `tau_table` | From validation |
| `lists` | Infeasible, unsolved and multi-basin cells |
| `validation` | V1–V11 results |
| `budget` | Timings, projections and cuts |
| `excluded_runs`, `deviations` | Bookkeeping |

## 4. Decisions where the protocol is ambiguous

| # | Decision | Reason |
|---|---|---|
| D1 | Transplants and real laps use `refine_reference(load_tumftm(name))`, the Track that `train_dr.py`, `train.py` and `diagnose_crashes.py` evaluate on. The "never refine" rule in 1.4 applies to synthetic stimuli. | The fidelity gate compares with the real lap, which ran on the refined track. Hockenheim is unchanged by refine. |
| D2 | Transplant knot samples are copied index for index from the source and not resampled. Real tracks have ds = 2 ± 3e-4 m, so arc length is rescaled by < 0.03%. | This makes "κ matches the source to 1e-9" exact. The fidelity gate judges the stretch. |
| D3 | The pinned s-OCP resamples the full strip's spline on nodes s0 − 20 + 2i, so a node lands exactly on s0. | Phase offsets put s0 off the 2 m grid. Values at the nodes equal the full strip's by construction, which is the 1e-9 assert. |
| D4 | s-OCP results are returned with s in strip coordinates and t = 0 at the pinned node. The 20 m lead-in nodes are dropped. | `min_time_guess` and `trajectory_from_socp` then need no offsets. |
| D5 | The bank stores float64 samples. Agents get float32 `TrackArrays`, as in training. The 1e-12 and 1e-9 checks run on the float64 samples. | Float32 coefficients cannot meet 1e-9. `track_arrays(to_track(·))` still equals the bank arrays exactly. |
| D6 | x0 is built in float32. Keys use its float64 cast. | Agent and key then start from bit-identical states. |
| D7 | The noise id in protocol 4's `fold_in(…, stim_id)` is the cell's `pair_id`: shared by L and R, distinct across approaches. | P5 requires the two members of a pair to see mirrored noise. A per-strip id would give them different noise. |
| D8 | ε_t = `normal(fold_in(run_key, t), (2,))`; ε_steer is negated on the mirror. | Pins the common-random-number stream for every snapshot. |
| D9 | Stochastic sample j starts at phase j mod n_phases. | Crash rates then average over the aliasing phases instead of resting on u = 0. |
| D10 | An error's pair value is the median of per-run errors. Speed errors (e_ve, e_vmin, e_vmin^line) are against the nominal u = 0 key; onset errors (Δd_b, Δd_resp), regret and Δt_corner against the key at the run's own phase if one exists, else the u = 0 key (amended in the review fixes, RF9). **When the manifest's validation V7 failed on speeds** (6.5's remedy, ruling R15), the speed errors also go to the key at the run's own phase when one exists, else the u = 0 key, and e_ve_bo to a backed-off key at the run's phase when one exists, else the u = 0 one (see "V7 remedy" below). e_ve^RA is against the re-anchored key anchored on the run's own phase, and P7's 16-phase cells have phase keys at every phase (ruling R17). | Protocol 2.2 defines e_ve against the u = 0 key; 6.1 lists phase-matched keys as backing Δd_b, P2b and the key phase SD only. V7 measured how far v_e* moves with the phase; its failure means the u = 0 key is the wrong reference for a run at another phase. |
| D11 | l_typ's top bin is open (R_seg ≥ 55.9 m). v_typ uses the same R_seg bins, with θ classes < 120° and ≥ 120° (the E2 classes). | The protocol places the top median at 75 m and holds it constant beyond. It names "θ class" for v_typ without defining it. |
| D12 | When a segmented corner merges several generated corners, it takes the l_r of the generated corner that holds its max\|κ_smooth\|. | Gives one ramp per R_seg. |
| D13 | Support strips use the root-seed stream `support`, not `PRNGKey(7)` as in the briefs. | Protocol 1.4 fixes the root seed. The frozen values will differ slightly from the expected numbers in 2.3, which are only expectations. |
| D14 | Real-corner landmarks: s_c0 and s_c1 come from `segment` (s_start and s_end + ds); s_apex is their midpoint; s_a0 and s_a1 are the first and last knots with \|κ_smooth\| ≥ 0.9·max. | The protocol defines arcs only for trapezoids. |
| D15 | A crash corner is the `segment` corner that contains the crash point; otherwise the last corner starting before it. | This is the corner the car left the road in. Ruling R2 confirms it for every crash corner, G2's YasMarina and Zandvoort included. |
| D16 | A transplant's initial state is the real-lap state at s_c − 800 on the crash lap. If that point lies before the run's start, the run's own start state is used. The strip still covers [s_c − 800, s_c + 400]; the road before s = 0 is the end of the lap. | Every G1/G2 crash is on lap 1, and six crash corners lie within 800 m of s = 0 (Hockenheim 265, Melbourne 377, Zandvoort 369, Norisring 480, Oschersleben 474/469). |
| D17 | E2 candidates are limited to corners that G2's real lap reaches alive. | G2 crashes on lap 1 at 7 circuits, so beyond its crash there is no G2 state. |
| D18 | States for corners that did not crash (tier D, E2) come from lap 2 of the deterministic two-lap evaluation. | Lap 2 is the timed, settled lap in `eval/laps.py`. |
| D19 | The pilot replication uses the pilot's own layout: N = 1280, entry at 800 m, start at 100 m and 80 m/s, width 5.5/5.5 with no cap, 500 steps. | "Replicate" means the same stimulus. |
| D20 | Tier G strips are straight over their whole length. Null cells start at s0 = 400 m. | Nulls are aligned by distance travelled, so s0 does not matter. 400 m leaves more than 4 km before truncation. |
| D21 | A run is finished when `agent.pkl` exists (specialists) or `eval_rdev.json` exists (generalists). | S1's run directory has no eval.json. `train.py` writes agent.pkl when training ends. |
| D22 | The final snapshot is the last `ckpt/*.pkl` for S2, S3 and G1–G3, and `agent.pkl` for S1, which has no ckpt directory. | Checkpoints carry `timesteps`; agent.pkl does not. |
| D23 | Tables are numpy column dicts in `.npz`. | pandas is not a declared dependency (only transitive). These hash reproducibly. |
| D24 | One rollout npz per snapshot, with det/nat/mat groups. Stochastic traces store n, pedal, util and β_f as float16, and s and vx as float32. | About 25k stochastic runs per final would otherwise take about 0.6 GB per file. |
| D25 | Phases of a real-state cell shift only s, by −u_k. | The state lies on the 100 m real lead-in or the approach straight, so the shift is consistent. |
| D26 | The phase cycle P comes from the nominal key's sighting speed, else the pinned s-OCP's, else QSS. | Tier D braking corners have only s-OCP keys. |
| D27 | After termination, the terminal state is kept (`valid` True) and later states are frozen copies (`valid` False). | s_crash and v_crash are the env's terminal state. |
| D28 | `io.py` is implemented in the skeleton, with tests. | Every module writes through it. Six builders would otherwise race to write it. |
| D29 | Q targets are precomputed per stimulus and indexed by `aux`. Qp recomputes its preview target every step. | A 2560-knot forward-backward pass every step would dominate the rollout. |
| D30 | Controls get raw observations. Their clipping statistics are NaN. | Controls have no normaliser. K normalises with S1's obs_rms itself. |
| D31 | `log_std` is recorded once per snapshot, `[2]`. | It is a state-independent parameter of every actor. |
| D32 | Right-hand keys are mirrored on the fly from the left-hand solve once V5 has passed (`answer_key.mirror_key`). | This is protocol 3.5's "one direction" key count. |
| D33 | The specialists' d_OOD buffers are built per specialist. S1's is built before the tag (the protocol's exception); S2's and S3's after the tag. | d_OOD uses "that specialist's own Hockenheim buffer". |

## 5. API check (task (a)): mismatches with the protocol's assumptions

All of these were checked by reading the source and running `inspect.signature`. Nothing was executed against agents.

| API | Finding | Handling |
|---|---|---|
| `racing.init_env_state` | **Does not exist.** Protocol 4 says states are "built with `init_env_state`". | `rollout.init_env_state`, following the levels convention: t = 0, slow = 0, s0 = x0[0]. |
| `min_time.solve_min_time(track, p, ds=3.0, margin, init, max_iter, tol, print_level, u_rate_weight)` | Start state is free: no z0 pin. The guess has no `v_start`. No vx bounds; vx is boxed to [5, 120]. No CPU cap. Default ds is 3.0. | Copied as `answer_key.solve_min_time_pinned`, with `z_start`, `i_start`, `v_start`, `vx_upper` and `max_cpu_time`. V10 checks it against the repo solver. |
| `WindowSolver.solve(track, x0, init, ctx, p, max_rounds, lam0, hidden_mu, refine)` | No `a0_fixed`. a₋₁ = 0 is hard-coded, as in `window_return` and `replay_window`. IPOPT caps (`max_iter`, `ipopt={'max_cpu_time': …}`) are constructor options. Tracks with \|ds − 2\| > 0.05 are rejected. | `AnchoredSolver(WindowSolver)` overrides `solve`. `make_solver` caches one instance per cap set; the compiled library is shared through the disk cache. DS = 2.0 everywhere. |
| `replay.replay_window`, `verify_window`, `best_verified` | Signatures as assumed. `replay_window` sets `jax_enable_x64` globally and starts with prev_action = accel = 0. `best_verified` mutates `res.meta['replay']`. `replay_window` uses the env's default reward weights (action cost 0.02), not the result's spec. | Keys and rollouts run in separate processes. Keys verify with `answer_key.verify_key` (K15). Anchored replays stay consistent: the anchored problem also has a₋₁ = 0 at x_{L−1}, so the step-0 term is a constant. |
| `window.min_time_guess(ocp, track, x_start, steps, dt)` | On an open strip it stops where the s-OCP ends (with a warning). | `answer_key.socp_guess` adds a straight full-throttle tail (`extend_straight`). |
| `window.solve_continuation`, `WindowSpec`, `WindowResult` | As assumed. `meta['lam']` = (lam_g, lam_x). `alpha_max` is a runtime parameter. | — |
| `audit.rollouts.snapshot_from_pickle(d)` | Returns only (actor, critic, obs_rms, ret_rms). Config and timesteps must be read from the dict. agent.pkl has no `timesteps`. `diagnose_crashes.py` does not filter the config. | `agents.load_agent` filters PPOConfig to its fields, as `export.load_policy` does, and falls back to `total_timesteps`. |
| `ppo.Actor(cfg, 2, False)`, `PPOConfig`, `RunningMeanStd.normalize(x, clip=10)` | As assumed. log_std is a state-independent parameter. | — |
| `generator.strip_arrays(kappa, w_left, w_right, ds)`, `to_track(arr, name)` | As assumed. No padding option; dtype follows the input. | Fixed N = 2560 and float32 inputs. |
| `generator.generate` | Does not expose the per-corner ramp l_r that l_typ needs. | Copied as `support.generate_with_corners`; a test checks its κ equals `generate`'s. |
| `sim.track_arrays(track, pad_to, dtype)`, `kappa_at(s, tr)`, `edges_at(s, tr)` | As assumed. `edges_at` returns ((lx, ly), (rx, ry)). | — |
| `racing.EnvState`, `step_no_reset`, `observe(state, tr, p)`, `default_context()`, `params_from_context(ctx, base, hidden_mu)` | As assumed. `observe` needs VehicleParams. `info` has no non-finite flag (infer it as term & ~others), and `spun` includes `slipped`. `default_context()` comes back float64 when x64 is on. | `measures.crash_cause` tests slip before spin. `rollout.default_ctx()` pins exact float32. |
| `qss.GG.from_params`, `speed_profile(kappa, ds, gg, closed, v_start, v_cap)` | As assumed. A point mass at the 60 m/s tyre load, so not a key. | Used for layout, `support.stimulus_v_peak` and the QSS null only. |
| `corners.segment(track, k_on, k_off, merge_gap, smooth_m)`, `R_DEV`, `R_TEST` | As assumed. `Corner.s_end` is the last knot inside, so the corner exit is s_end + ds. 12 + 12 circuits. | D14. |
| `export.lockbox_guard(track_name, unsealed)` | As assumed. Also catches mirrored, reversed and path variants. | Called by `stimuli.real_track` and `agents.real_lap`. |
| `eval.laps.timed_laps(policy_fn, track, v_start, runs, laps, params, ctx, seed)` | Takes the Track from the caller and keeps only run 0's (t, s, n, vx, action). No full state, prev_action or accel. | `agents.real_lap` re-runs the same convention (s = 0, v0 = 0.9·v_QSS(0), default car, two laps) and records full EnvStates. |
| Real circuits (`load_tumftm`) | ds_eff is 1.9998–2.0005 m, not exactly 2. | D2. |

## 6. Open questions for the orchestrator (answered by protocol 12, rulings R1-R13)

Question 1 is answered by ruling R1, question 2 by R2; question 3 needed no ruling. The
implementation of every ruling is in "Rulings R1-R13" at the end of this file.

1. **Real-lap rollouts for initial states.** The E1 real and isolated keys are P0 and solved before the tag. They need G1's and G2's deterministic real-lap states, and tier D's s-OCP keys need S1's. These runs repeat the standard R_DEV evaluation already done for `docs/findings/*-crashes.md` and `eval_rdev.json`; they are not stimuli. But they are agent rollouts, and the protocol lists only two exceptions. **Recommendation:** allow them as already-seen evaluation, identical to `diagnose_crashes.py`, run them only in the Integrate phase, and log them in the manifest. G3's crash corners (P8 addition) need the same evaluation on a prospective agent.
2. **G2 YasMarina and Zandvoort crash corners.** The protocol labels them 12.7 and 12.1 m. Those are the crash docs' raw radius within ±60 m, which picks up the adjacent tighter corner. The `segment` corner that contains each crash has R_seg ≈ 20.4 and 27.5 m (brief-support §3; computed: 20.78 and 28.02 m). P8's tail set is named by the labels. D15 uses the containing corner. If the labelled neighbour is intended instead, change `stimuli.crash_corner` before the tag.
3. **S3 has finished.** `runs/hockenheim-specialist-mlp-200M-s3/agent.pkl` and eval.json now exist. Under D21 it enters the manifest at the next `manifest` stage, unevaluated.

## 7. Builder notes: `rollout.py` and `agents.py`

Implemented in full (note IDs RO1-RO13, so they do not clash with the rulings R1-R13); tests in `tests/test_interp_rollout.py` (random actor with the agents' MLP 3×256 tanh architecture, scripted lane keeper, hand-built strips; no trained snapshot is rolled out).

| # | Decision | Reason |
|---|---|---|
| RO1 | The steer sign comes from `Cell.mirrored` (−1 on the mirrored member). `plan_runs` raises if both cells of a pair share a sign. Cells without the field fall back to `rollout.steer_sign(stimulus)`. | P5 needs opposite signs within each pair; a silent mismatch would break the mirror test. |
| RO2 | Deterministic runs carry a zero key and zero noise std; `nat` sets `native`; `mat` sets `noise_std = (0.04, 0.08)`. One compiled function serves all three modes. | Fewer compiles; a = mean + 0·ε is exactly the mean. |
| RO3 | `a` is recorded before the env's clip. Steps not taken hold NaN (`a`, `a_mean`) or 0 (`reward`, `flags`). Step k was taken iff `valid[k + 1]`. | The adapter clips; NaN keeps post-crash commands from leaking into measures. |
| RO4 | Clip statistics use the unclipped z = (ob − mean)/√(var + 1e-8): `clip_frac` is the fraction of \|z\| ≥ 10, and `clip_max` is max\|z\| (so it can exceed 10). | Shows how far past the clip a channel is. |
| RO5 | Stochastic traces also keep ξ (`<g>_xi`, float16). | Protocol 4 records ξ at first sight for every run; measures are computed from the traces afterwards. |
| RO6 | The step-loop check reports max\|Δx\|/max(1, \|x\|). Measured: 1.2e-7 (random actor) and 8.9e-8 (lane keeper, 400 steps alive). | float32 cannot hold s ≈ 1000 m to 1e-6 absolute (its ulp is 6e-5 m). |
| RO7 | Batch = single is checked exactly: each run is redone among decoys in a batch of the same shape, and the result must match bit for bit (measured 0.0). A batch-of-one variant (`isolated=False`) agrees only to float32 rounding (5.8e-7 for the actor over 60 steps; 4.4e-6 for the lane keeper over 400 steps). | XLA vectorises different batch shapes differently, and closed-loop gains carry one ulp of vx into the pedal. Isolation tests what matters (runs do not interact) without float noise. |
| RO8 | Mirror gate: the lane keeper drives the first member closed-loop, a seeded steering wobble is added, and the result is replayed as a tape on both members (steer negated on the mirror), deterministic and with matched noise under CRN. Measured: 0.0 over 400 steps. A negative control (no negation) differs by more than 1e-2. | The gate needs a sequence that keeps the car on the road for 400 steps. |
| RO9 | A short last chunk is padded to a power-of-two bucket of at least 16 runs. | One compile per bucket instead of one per remainder size. |
| RO10 | agent.pkl has no `timesteps`, so it gets `num_updates × batch_size` of its config: 199,884,800 for S1, the number S2's and S3's final checkpoints carry. The skeleton's fallback was `total_timesteps`. | train.py ends on the last whole update. |
| RO11 | `freeze_manifest` retries entries with `error` set. `is_final` (derived) follows each run's current last snapshot. Every other field of an existing entry is never changed. | A run is listed only once finished, so in practice neither exception fires. |
| RO12 | `real_lap` runs through the harness with a one-track (closed) bank, following diagnose_crashes.py's convention: `steps = laps·3L/max(v0, 20)/dt`. `real_state` uses live states only; the terminal crash state is never used. | It records everything a stimulus run records. |
| RO13 | `pilot_replication` builds x0 = [100, 0, 0, 80, …] directly (D19), not through `Cell.initial_state`. | The pilot layout puts the corner at 800 m, not S_C. |

**Throughput** (`rollout.benchmark`, 2 XLA threads, random 3×256 actor with obs_rms, 64-row bank of 2560-knot strips, load average about 32 from the trainer and other agents):
- B = 2048, T = 800, `full`: 104k steps/s (15.7 s per call; 3.1 s to compile);
- `stoch`: 112k steps/s.

This is above the protocol's 30k assumption, so budget rule 3 (below 15k) does not fire.

**API additions** (all backward compatible):
- `plan_runs(..., aux=, agent=)`, `cell_steer_sign`, `check_pair_signs`;
- `batch_from_states`, `noise_eps`, `trace_arrays`, `bank_arrays`, `cause_of`, `d_max`;
- `make_stimulus_rollout(...)` returns a wrapper that asserts x64 off, with `.jitted` as the bare jitted function;
- extra keyword arguments on the checks: `relative`, `isolated`, `phase`, `aux`, `signs`;
- `benchmark(..., record=, repeats=)`, `pilot_verdict`;
- `agents.forward_pass_ok` and `Manifest.entry`.

`run_rollouts` and the checks also accept a stacked `TrackArrays` in place of a `StimulusBank`.

## 8. Builder notes: `measures.py`

Implemented in full; tests in `tests/test_interp_measures.py` (25 tests, about 8 CPU-s; synthetic trajectories with known answers, one 24-step TO-OCP solve, no agent). `response_metrics` costs 0.5 ms per 400-step run.

**Model tables** are cached at `outputs/psychophysics/tables/model_tables_<car hash>.npz` by `measures.model_tables()` (20 CPU-s to build, 2 ms to load).
- a_dec(310 km/h) = 65.75 m/s² (protocol 1.2: 65.4). The front axle saturates at pedal 0.689, 0.455, 0.295 and 0.26 at 310, 200, 100 and 60 km/h (protocol: 0.70, 0.46, 0.30, 0.26).
- a_y,max is the centripetal |v|·r of a true steady state. On the (δ, sideslip) grid, r comes from the yaw-moment balance and the pedal from dvx = 0. Steady states are the sign changes of dvy along the sideslip axis. The best one is polished by Newton on (sideslip, r, pedal) over ±1 δ step.
- Because lock and power are in the model, the table is lock-limited at 20–40 km/h, tyre-limited at 45–265 km/h and power-limited at 270–310 km/h; at 315 km/h it is 0. Relative to QSS that gives 0.77 at 40 km/h, 0.935 at 100, 0.87 at 265 and 0.55 at 300. Protocol 1.2 quotes 0.99 at 40 and 0.87 at 300, which matches only the tyre-limited band.
- v_ss(ρ) consequently caps fast corners: 261 km/h at ρ = 109 m, 297 at 171 and 308 at 300.

| # | Decision | Reason |
|---|---|---|
| M1 | A point measure needs its point reached alive. A window measure (v_min, util peaks, ρ_min, line apex, clearance, RMS(n − n*), Δv) needs the window's end reached, else NaN. `completed` means s_c1 + 250 m was reached before any termination. `time_censored` means alive at W without reaching it. | "Censored, never imputed." This also matches 9.1's "non-overspeed crash after s_c0 loses its v_min". |
| M2 | Everything "at s" uses first passage on the strictly increasing prefix of s. | A spin or stop makes s non-monotone. |
| M3 | Release s_r is interpolated at the 0.1 crossing. s_dec is interpolated between the two commands that bracket 0 (a₋₁ = `prev_action0`). Blip speed lost is vx at the blip's start minus vx at the first state after it. Forced means t_b ≤ 0.5 s. | Same treatment as the onset. P3's "more than 0.5 s after the start". |
| M4 | The legacy onset uses the primary rule's structure (last run of ≥ 3 states beginning before s_apex) on pedal ≤ −0.2, inclusive. | The v2 draft's −0.2 definition. The v1 "first after the lift" is not used. |
| M5 | Depth removes aerodynamic drag only, as written; rolling resistance (0.3–1 m/s²) stays in. Peak is over states k_b…k_r (central differences). Mean is the exact Δv/Δt over [t_b, t_r] minus the time-averaged drag. | The protocol says "minus drag/m". |
| M6 | κ_path is signed toward the corner (σκ), and \|κ\| on straights. ρ_min includes the interpolated window ends, so a short plateau still has samples. | A short arc may hold no state. |
| M7 | η on a straight uses σ = +1 (left = inside). Widths come from the strip's C2 spline, exactly as the env reads them. η = ±1 is the termination boundary. | Clearance near 0 must be exact. |
| M8 | `clip_frac_<g>` is the mean over valid states of the per-state fraction; `clip_max_<g>` is the max. `kappa_clipped` follows protocol 2.3. | Run-level summaries of per-step channels. |
| M9 | J re-prices step 0 against a₋₁ = 0 and sums only the steps taken (to termination or truncation, at most W). | Protocol 5's regret convention for transplants with a real prev_action. |
| M10 | Δt_corner aligns each null by its own counterpart's distance travelled: the run's null by s_ff − s0, the key's null by s_ff − s0*. It is NaN for crashed or time-censored runs. `dt_corner_socp` repeats it against the s-OCP. | Fair to keys solved at another phase (s0* ≠ s0). 9.4 excludes time-censored runs. |
| M11 | Crash typing tries overspeed, then over-brake, then line/steer, else other. "≥ 3 steps" means consecutive. Δv is taken at the first step of the run, for over-brake too. Only a *certified* point of no return decides. Otherwise the ellipse rule searches from s_c0 − 50 m to the crash, or the whole run when the crash comes first. `NoReturn.s_last_viable ≥ s_a0` marks "every tested state through s_a0 viable", which is resolved and so is not overspeed. | The window keeps a survived earlier corner of a real strip from typing the crash. |
| M12 | `sight_at_start` is added. On A2 cells D < d_max(v0), so the corner is visible at state 0, where the pedal state is 0. The protocol's rule then flags every A2 run as unsettled. | Ruling R6: the unsettled flag does not apply when `sight_at_start`. |
| M13 | The stochastic adapter also maps ξ (`<g>_xi`); other unrecorded channels are NaN. | Rollout note RO5. |

**Step 3, measured.** The adapters were checked on a genuine 80-step TO-OCP solution that brakes (one-off, 32 CPU-s) against its float64 env replay: the largest measure difference was 6.3e-13 and J agreed to 3e-15. The test suite repeats this on a 24-step solve.

**API additions** (all backward compatible):
- `ResponseMetrics` gains `sight_at_start` and `blip_s_last`. `KeyErrors` gains `corner_regret`, `dt_corner_socp` and `dt_flag_socp`.
- New keyword arguments: `key_errors(..., socp=, null_socp=)`, `crash_type(..., p=)`, `brake_onset(..., vx=)`, `trajectory_from_rollout(..., source=)` and `trajectory_from_replay(..., p=, gamma=, substeps=)`.
- `NoReturn` gains `s_last_viable`. `BrakeOnset` gains `t_b`, `v_b` and `t_r`. `ModelTables` gains the steady-state arrays `delta_y`, `beta_y`, `r_y`, `pedal_y`, `alpha_f_y`, `alpha_r_y` and `limit_y`.
- New helpers: `NP` (NumPy ops), `car_params`, `make_marks`, `model_tables` / `tables_path`, `steady_lateral_max`, `discounted_return`, `corner_time_lost`, `metrics_max_diff` and `check_window_replay` (step 3).
- The `ocp` imports are now type-only, so rollout processes no longer pull in casadi through measures.

**Requests:**
- `answer_key.no_return` should set `s_last_viable` when every tested state is viable (it now returns `NoReturn(None, False)`, which crash typing reads as unresolved).
- `analysis.run_table` can pass `socp=` / `null_socp=` to fill `dt_corner_socp`.
- M12 needed a ruling (R6, implemented).

## 9. Builder notes: `answer_key.py`

**Solver copies (V10, measured exact).** `solve_min_time_pinned` and `AnchoredSolver.solve` differ from `solve_min_time` / `WindowSolver.solve` only by the pin, `v_start`, `vx_upper`, CPU cap and `a0_fixed` lines (checked with a source diff). With the extras unset both reproduce the repo solver bit for bit (max difference 0.0 on a 600 m strip and a 24-step window). The pin holds to 0.0 and the cut strip's κ and widths equal the full strip's to 9e-16.

**alpha_max needs no recompile.** `WindowSolver.parameters` passes `p.alpha_max` as the last solver parameter, and the slip bound 1 − slip_safety/alpha_max is computed per solve. Widths are data (coefficient windows), and narrowing a natural spline's samples by c narrows the spline by c. So the backed-off key (`backoff_problem`) uses the nominal compiled library; a test checks the parameter vector. Only `WindowSpec` fields are compiled in, so the min-time-in-actuation key (`ACTUATION_SPEC`, γ = 1, action cost 1e-4) needs one compile per window length.

**Decisions:**

| # | Decision | Reason |
|---|---|---|
| K1 | Each IPOPT call of a start gets the start's remaining CPU budget, rounded up to 50 s; a start whose summed CPU passes 400 s, or whose last stage hit a cap, is capped and counts as failed. | IPOPT's `max_cpu_time` is per call, a start is several calls. |
| K2 | Every key is solved for the car the rollouts drive: `racing.default_context()` as exact float32 values, cast (`key_ctx`). | Key and agent then drive the identical car. |
| K3 | The guess tail beyond the s-OCP is a full-throttle run with a weak lane-keeping steer (0.5 rad/s, at most 0.3 of the lock). | Steer 0 drifts off the strip over 20 s. |
| K4 | A start fails without being solved if its guess is non-finite. A start that fails counts as a disagreement for the Q rule, so Q is added. | Never feed NaN to IPOPT; a missing start cannot agree. |
| K5 | A verified start must also keep max\|α\| ≤ alpha_max + 1e-6 on its own states (`replay['max_slip']`). | The replay env terminates at the nominal 10°, which does not check a backed-off key's 8.5°. |
| K6 | Backed-off and anchored keys are verified on the problem's own Track. For a backed-off key that is the narrowed corridor, which is stricter than the stimulus's. | — |
| K7 | Phase-matched and anchored warm starts re-time the nominal key by aligning in s (`retime`). A start before the source holds x0's speed until it reaches the source's first s. | This is the u_k shift, done in s. |
| K8 | `KeyRecord.direction` is the stimulus's direction (+1, −1, 0). Keys mirrored on the fly carry the flag `mirrored` and the id `<key_id>m`. | The skeleton's "+1 solved / −1 mirrored" reading breaks when V5 solves a right-hand key. |
| K9 | `layout_a2` measures the onset distance from v0 with a cruise s-OCP: pinned 600 m out at v0, with vx ≤ v0 up to s_c0. Then D = d_b + 25. v_b* for the ±10% rule is read from the pinned s-OCP at that D, since the TO-OCP keys need W and P first. Cells outside the band move 10 m per round (shorter if v_b is high) until 90% are in band, for at most 10 rounds. | The protocol's "onset distance from v0" needs a cruise phase. The brief measured a 3–4 m gap between s-OCP and TO-OCP onsets. |
| K10 | The actuation key solves one stage at W, from the s-OCP guess, the nominal key re-timed and start (3), Q's rollout (added in the V3 review: 6.2 adds start (3) on the validation set, and actuation keys exist only there; skipped and flagged without model tables). | A continuation would compile every stage length. The two-start recipe left cell 442 (R13/180 A1) with its starts 2.3e-4 apart in J, past 6.2's 1e-4 trigger for start (3). |
| K11 | Anchored and viability keys get the same caps and one retry at max_iter 3000 as agent-free keys. Viability keys are single-stage at W = 200 from the nominal and backed-off keys re-timed. | 6.2's caps applied uniformly. |
| K12 | The infeasibility bound's ρ_ub is 1.1 × the largest minimum radius over [s_c0, s_c1] among three lines: the repo's min-curvature line (`audit.qss_distance`), a Frenet-curvature min-curvature line (`frenet_min_curvature_offsets`), and the centreline. All three are recorded in `Bound.rho_lines`. | The repo's QP weights the inside of a turn by (1 − nκ)⁴, so it wraps hairpins round the inner edge: ρ = 8.0 m at R13/180, below the centreline's 13 m. Used alone, it would certify feasible hairpin cells. The Frenet line gives 13.6 m. |
| K13 | `no_return` fills `NoReturn.s_last_viable` (the measures builder's request). Tested states are the agent's x_L, the first state at or after each target, which the anchored solve from x_{L−1} reproduces exactly. | — |
| K14 | Key ids hash kind, the bank row's float64 samples, x0's exact bytes, W, every car parameter, the float32 context, the spec and extras (shrink, a0, agent, snapshot, target, s_end, vx bounds; a re-anchored key also its phase, R17a, `anchored_key_id`). Aliases are per cell (`cell<id>_ph<k>_<agent>[@snapshot][#target]`) and per mirror pair (`pair<id>_dir±1_ph<k>_…`). The s-OCP alias is written only for the nominal problem. | `key_for` can find a key without re-deriving it, and a backed-off s-OCP can never shadow the nominal one. |
| K15 | Starts are verified with `answer_key.verify_key`: `replay.verify_window` with the env's `action_cost` and `progress_scale` taken from the key's `WindowSpec` (`replay_key`). For the default spec it is the repo replay bit for bit (test). | `replay_window` builds `EnvParams` with the env's action cost 0.02, so the actuation key (1e-4) replayed to a return ~3e-4 below its J* and every one came back 'unsolved' in validation (V3), though IPOPT converged and the replay survived (state deviation 5e-9). Under its own weights R40/90 A2 verifies to 1.6e-12, and its two starts agree on J* to 1e-12. Test: `test_actuation_key_verifies_under_its_own_reward`. |
| K16 | An actuation record carries the flag `ACTUATION_REPLAY_FLAG` once its starts were verified with `verify_key`. A cached actuation record without it is re-solved under the same id, with or without `--force`. | Key ids hash the problem, not the code, so the 12 records solved before K15 (all 'unsolved') kept their ids, and `actuation_key` returned them on every later run. Only V3 reads actuation keys; no other kind changed its verification (K15: the default spec's replay is the repo's). Test: `test_actuation_key_re_solves_records_from_before_its_own_replay`. |
| K17 | V3 judges a cell only when both its keys are verified. A cell without them, or an A1/A2 pair job that returned no V3, is unevaluated, and V3 then fails (`v3_verdict`, script `_v3_merge`). The merged verdict is judged at V8's b (the expected 0.075 only when V8 gave none). Multi-basin keys are listed (`multi_basin`), with the objective effect of every start in the actuation key's near-equal set (`objective_e_ve_near_equal`); blocking stays on the selected start. | V3 passed with all 12 cells ('verified', 'unsolved'), so R15 did not list it. The pair jobs run before V8, so their own verdict is at 0.075; 6.5 says b/2. Cell 448 (R13/90 A2) is multi-basin by 6.3: its two near-equal starts give v_e 33.58 and 34.29 m/s, an objective effect of −2.46% or −0.38%. Tests: `test_v3_is_not_passed_by_cells_it_could_not_judge`, `test_v3_merge_judges_at_v8_band_and_fails_cells_it_could_not_judge`. |

**Interface changes** (all additive; existing positional arguments unchanged):
- `full_recipe`: `shrink` now defaults to None (kind `backoff` uses 0.35 and 8.5°). New keywords: `alpha_max`, `tables`, `q_guess`, `q_cv`, `W`, `spec`, `force`, `prior`.
- `warm_key`, `null_key`, `anchored_key`, `viability_keys`, `actuation_key` take `tables=`, `p=` and `force=`. `anchored_key` and `viability_keys` also take `snapshot=`; `viability_keys` takes `agent=`; `anchored_key` takes `W=` and `phase=` (R17a); `actuation_key` takes `nominal=`.
- `socp_from_state` gains `stim_id`, `cell_id`, `phase`, `key_id` and `max_iter`.
- `solve_from_state` accepts `a_prev=None`, which is a plain `WindowSolver.solve`.
- `retime` gains `track`, used to extend past the source's end.
- `active_set` gains `landmarks` (the v_min window) and `spec`.
- `make_solver` gains `compile`.
- `key_for` gains `snapshot`, `solve` and recipe keywords.
- `KeyCache.put` gains `aliases`; new `alias`, `set_alias`, `get_flag`, `set_flag` and `records`.
- `v2`, `v3`, `v5` and `v8` take `cache=`, `tables=` and `log=`. `v6_monotone` returns `(result, records)` and re-solves when given `bank` and `cells`.
- `KeyRecord` gains `pair_id`, `labels`, `stats` and `bound`. `StartRecord` gains `rounds`. `Bound` gains `rho_lines`.
- New functions: `key_ctx`, `key_metrics`, `socp_metrics`, `run_metrics`, `q_rollout_guess`, `series_of`, `solve_series`, `phase_cycle`, `refresh_phase_cycles`, `resolve_beaten_key`, `slip_steps`, `frenet_min_curvature_offsets`, `mirror_state`, `profile_gap`, `band_from_costs`, `row_hash`.

**Measured cost** (load average 18–26, 2 XLA threads):

| Solve | Size | Cost |
|---|---|---|
| Pinned s-OCP | 442 nodes, R40/90 from 160 km/h at D = 200 m, 98 iterations | 3.1 CPU-s |
| 200-step TO-OCP from the s-OCP guess, plus replay | 67 iterations | 2.2–2.4 CPU-s |
| Perturbed start | 165 iterations | 4.0–4.3 CPU-s |
| Backed-off warm key | — | 3.6–4.8 CPU-s |
| Whole nominal recipe (s-OCP and 2 starts) | — | 9–11 CPU-s |
| V10 | — | 20–22 CPU-s |
| Building a cached 200-step solver | — | 4 s wall |

**Requests:**
- `controls.q_target_profile` returns NaN everywhere on a plain R40/90 strip with the cached `model_tables_d1177829d920.npz`. The cause is the cached `a_y_max`, which ends in 0 at 87.5 m/s: with that entry set to 1 m/s² the profile is finite. Until that is fixed, start (3) is recorded as "non-finite first guess (not solved)".
- The `keys` stage should call `layout_a2`, then `solve_series` over `series_of(...)` (descending R), then `refresh_phase_cycles`, then `warm_key` for the backed-off and phase-matched keys. V5 must pass before right-hand keys are taken from `key_for` (mirror shortcut).

## 9. Builder notes: `support.py` and `stimuli.py`

Implemented in full; tests in `tests/test_interp_support.py` (14) and `tests/test_interp_stimuli.py` (22), about 37 CPU-s together. No agent is rolled out; the finals' obs_rms are only read (R_clip). Real-lap states in the tests are stand-ins.

**Measured cost** (2 XLA threads, load average about 25):

| Step | CPU time |
|---|---|
| `build_support` (3 grammars × 300 strips plus 12 R_DEV circuits) | 12 s |
| `build_design` (163 rows, 490 cells) | 1.7 s |
| `build_real_design` (13 E1 + 13 E2 corners, 20 tier D corners; 590 rows, 604 cells) | 2.8 s |
| `check_design_details` on all 753 rows | 6.9 s |
| `cell_support_table` | 0.5 s |

**Frozen values, from the root-seed stream** (D13; `outputs/psychophysics/support/` once the stage runs):

| Quantity | Value | Protocol's expectation |
|---|---|---|
| q1 / q10 of R_seg | G1 12.0 / 21.3; G2 11.1 / 17.7; G3 10.2 / 14.5 m | 11.4 / 20.2, 10.4 / 16.4, 9.4 / 13.4 (these are the briefs' R_raw quantiles; their R_seg row reads 12.1 / 21.1) |
| R_clip^max (worse sign) | G1 8.94, G2 7.64, S1 9.39 m | 8.9, 7.6, 9.4 |
| l_typ at 9.4, 14.6, 22.9, 40, 75 m | 13.5, 21.1, 31.8, 42.4, 45.7 m (θ ≥ 10°, S1) | 26, 33, 40, 44, ≥ 44 |
| Onset rate, base corners with R_eff < 20 m | 0.0030 m⁻² | 0.0028–0.0031 |
| v_typ (km/h) per R_seg bin, θ < 120 / ≥ 120 | 180/140, 160/200, 180/200, 220/250, 250/250 | about 160 below 18.3 m |
| Tight and fast corners per 100 km (R_seg < 12 m, v_peak ≥ 250 km/h) | base 0.96, v2 1.52, v3 2.82, R_DEV 7.3 | 1.8 / 2.4 / 4.1 vs 20 (raw radius < 12 m) |

The l_typ expectation came from the grammar pool binned by pool r_min (brief-critB). The protocol defines l_typ on generated corners binned by R_seg. A corner only lands in a tight R_seg bin if its ramp is short, so the tight bins' medians are lower. The onset rate confirms that the generated strips are the ones critB measured.

| # | Decision | Reason |
|---|---|---|
| S1 | **Accepted (ruling R5).** l_typ and v_typ use base corners turning ≥ 10° (`TYPICAL_MIN_THETA_DEG`). Over every corner, the literal rule, the top bin's median is 13.2 m: 2,738 of its 5,674 corners are near-straight kinks at the 10 m ramp floor. l_typ would then fall from 42 m at R 40 to 13 m at R 75. The filter changes only the top bin. `LTypTable.l_nodes_all` keeps the literal medians, and `l_typ_table(base, 0)` restores the literal rule. | Protocol 1.4 expects ≥ 44 m at 75 m. Every stimulus turns ≥ 45°. |
| S2 | R_clip(k) = 50/(10σ_k − \|μ_k\|) (inf if 10σ ≤ \|μ\|), with σ = √(var + 1e-8) as in `normalize`. | This is the "worse sign" the protocol text asks for; its printed formula, 50/(\|μ\| + 10σ), is the better sign. The two differ by < 2% on the finals. |
| S3 | v_peak^QSS uses one estimator for training corners and stimuli: `support.qss_profile_jax`, a copy of `speed_profile_jax` (the training bank's v_ref) that adds a start speed and a per-knot cap. A stimulus is profiled from the knot at or before s0 at v0, with the road before it treated as straight. A test checks the copy equals the repo function bit for bit, and the NumPy envelope to 2e-4. | Protocol 2.3: "the same estimator ... v_peak from the QSS profile for both". |
| S4 | The R_DEV corner table is built on the refined real-lap Track (`stimuli.real_track`). The grammar is fitted on the raw centrelines, as train_dr.py does. | Crash corners, transplants and the fidelity gate all live on the refined track. Refinement merges some corners (MoscowRaceway 21 vs 36 raw), so R_DEV quantiles differ from the briefs' raw ones. |
| S5 | Cut tapers lie just outside the excised range ([s_c − 820, s_c − 800] and after s_c + 400). The road under a real initial state is therefore real, and transplant κ equals the source exactly on the whole excised range (measured 0.0 on 590 rows). | A taper inside the cut would put the start state on a road it was not computed on. |
| S6 | The tier D lead-in rule (the first 100 m from the state is real road with real width; substitutions blend in over 20 m on the substituted side) also applies to every E1 and E2 version. The lead-in starts at the corner's earliest state: s_c − 800 m, or s = 0 for the six crash corners within 800 m of the line (D16). | Straightening the road under a real state (r, δ, vy ≠ 0) would add a start transient to every ablation rung. |
| S7 | F2 λ scales the corner region [s_c − 20, s_e + 20] about its start: the corner start moves by 20(λ − 1) m (−4 to +5 m), and the landmarks scale with it. The road after it shifts by (λ − 1)·(L_seg + 40), with 20 m blends at both junctions. | "In place" while the real approach stays untouched. |
| S8 | F2 width doses (±0.25 m per side), F6 and F1 "const" apply outside the lead-in. | The lead-in rule; protocol 3.3's "whole strip" for the width doses conflicts with it. |
| S9 | F5 swap B straightens the corner region and the road after it, so A and B differ only in the presence of corner j. Swap A copies corner j's κ and widths from its region start and straightens after its region. | The phantom index measures v_min over [s_c − 50, s_c + L_c + 50]; a following real corner would contaminate it. |
| S10 | F6 wobble moves both edges out and in together: w_l and w_r += A·sin(2πs/60 + φ). κ noise is a random Fourier sum over the 80–200 m band, windowed by 20 m ramps, with the mean removed under the window (zero net heading) and scaled to peak \|Δκ\| = level. Seeds: `[root, 10⁶ + 4, corner, condition]`. | The protocol does not fix the phase convention or the amplitude norm. |
| S11 | The E2 "braking corner" uses the circuit's QSS: v_apex ≤ 0.95 v_peak. Crash corners are matched tightest first and matches are distinct; distance to crash corners is the cyclic gap between intervals. A candidate needs s_e + 250 m reached alive on lap 2 (or on lap 1 if G2's run ends there). | No key exists before matching. |
| S12 | Noise pairs on real roads: E1 real and mirror share a `pair_id` (the mirror is `Cell.mirrored`), and so do tier D (a) and F4. Every other version or condition has its own pair. | P5-style mirrored noise where a mirror exists, without pooling different ablation rungs. |
| S13 | `Cell.runners`: tier D cells from S1's state list (S1, G1, G2, G3); those from S2's or S3's state list only that specialist. Empty means every agent. | Protocol 3.3. |
| S14 | Tier G has 20 null cells, not ≤ 14: tier C's A2 cells use 6 different v_typ per width, and protocol 3.5 wants a null at each. | Count follows the rule. |
| S15 | The inside cap is applied knot by knot, as in the generator. A triangle's peak can fall between knots, so the cap may not bind just under R 8.33 m, and the realised R_eff of a triangle can sit up to about 3% above the nominal R (recorded in `Geometry`). | Exact protocol formula at the knots. |
| S16 | Tier D rows have no fitted twin, so their l_r, and hence their joint-support density, is NaN. E1 and E2 rows use their twin's l_r. | Protocol 2.3 asks for l_r. A real corner has no generator ramp. |

**Step-1 checks that fail as specified (accepted by rulings R3 and R4: the rows pass and are listed; R3 only within the kinks' sampling bound and with an exact spec area, R4 only for twins matching R_seg and the angle to < 1e-3, which are flagged `twin_length_short`; see "Rulings R1-R13"):**
1. **θ_eff within 0.3°** fails on 6 of 160 synthetic rows: R 6.0/180° (typical and no-cap; 0.44°) and R 8.4/90° (0.34°). The deviation is the knot samples' own area: kinks of the exact trapezoid fall between 2 m knots, and the spline integrates exactly as the samples' trapezoid sum. `check_design_details` lists each failing row with that area. Possible rulings:
   - accept, since θ_eff is recorded and the rows are clip-saturated or below support;
   - round l_r and L_p to the 2 m grid;
   - scale κ_peak to preserve the area.
2. **Twin fits within 2%, 1°, 2 m** fail on 5 of 13 crash corners and 10 of 13 E2 matches (E2 measured with stand-in reachability). In each failure the real L_seg exceeds the longest symmetric trapezoid with the same R_seg and angle, which is the triangle. Examples: Norisring 874, 76 m against at most 50 m; Hockenheim 248, 48 against 38; Sochi 4795, 198 against 118. The fit matches R_seg and the angle to < 1e-3 and takes the closest length; `twin_fit_fail` and the note record the shortfall. Possible rulings:
   - accept R_seg and angle twins with the length shortfall listed (P8's twin transfer reads R_seg and θ class);
   - allow a shouldered twin, which is a new shape.

**Interface changes** (callers adapt):
- `Cell`: new fields `mirrored` (which member gets ε_steer negated) and `runners` (empty means all). `initial_state(k=0)` works before the layout for phase 0.
- `training_strips` returns `Strips(kappa, w_left, w_right, corners, use_alt, v_qss)` instead of a 4-tuple.
- `matched_corners(rdev: {circuit: Track}, crash, reachable: {circuit: max unwrapped s alive})` returns (circuit, Corner, lap) triples.
- `Excision.to_strip(s_real, lap_offset=None)`: None unwraps to the lap nearest the corner.
- `realised_geometry(..., kappa_peak=, direction=)`.
- `ladder(..., lead_start=, versions=, twin=)`.
- `build_real_design(..., swap=None)`: F5 swap A rows exist only for corners in `swap` (`swap_partner` needs the condition (a) s-OCP keys).
- `build_support(..., n_strips=, log=)`.
- `SupportTables` gains `quantiles`, `rates` and `.table(name)`. `LTypTable` gains `min_theta_deg` and `l_nodes_all`; `VTypTable` gains `pooled` and `min_theta_deg`.
- Pilot rows are named `pilot/R<R>/T<θ>/<L|R>`.

**New helpers:**
- stimuli: `real_state_requests(reachable=None)` (the states the Integrate phase must collect, with keys `E1/<circuit>/<s_c>`, `E2/<circuit>/<s_c>`, `D/<i>`), `distinct_crash_corners`, `crash_corner_key`, `hockenheim_corners` / `DCorner`, `twin_for`, `check_design_details`, `cell_support_table`, `save_design` / `load_design` / `design_hash`, `StimulusBank.row_arrays`, `stimulus_from_dict`, `trapezoid`;
- support: `stimulus_v_peak_batch`, `qss_profile_jax`, `table_features`, `AGENT_GRAMMAR`, `dev_centrelines`.

**Wiring for `scripts/psychophysics.py`:**
- **support stage:** call `support.build_support(io.SUPPORT_DIR, {agent: (path, pickle['obs_rms'])})` for the finals in the manifest.
- **stimuli stage:**
  1. `stimuli.build_design(load_support(SUPPORT_DIR))`, then `check_design`, then `save_design(io.STIMULI_DIR, ...)`.
  2. With `--real`: `build_real_design(bank, cells, states, swap)` on the states from `agents.real_state`, following `real_state_requests`.
  3. After `answer_key.layout_a2` sets D, re-run `check_design` (lengths) and `cell_support_table`.


## Builder notes: `controls.py` and `probe.py`

Implemented in full. Tests are in `tests/test_interp_controls.py`: 22 tests, about 70 s, of which about 25 s is `measures.build_model_tables`. They use scripted controls and one randomly initialised MLP; no snapshot is loaded. The trapezoid test double equals `stimuli.corner_profile` and `shift_corner` exactly, and one test runs `tune_cv`, `completion` and `fit_tau_c` through the real harness on `stimuli.pilot_minigrid()`.

**Decisions.** Each one was forced by a failure of the literal reading, measured in a plain step loop on 42 tier-A-like cells (A1 and A2-like, L and R, plus no-cap R6/R7.5).

| # | Decision | Reason |
|---|---|---|
| C1 | The QSS pass is the classic one in `oracle/qss.py` (friction ellipse in both passes), on the model tables: v_ss from a_y,max, braking a_dec(v)·√(1−lat²), and the model's full-throttle acceleration in the forward pass. One JAX function, `grip_profile`, serves Q (2560 knots, precomputed per stimulus) and Qp (251-point, 2 m grid to 500 m, every step). | The protocol names the tables but not the pass. Sharing one function makes Q and Qp identical when the preview resolves the corner (tested). |
| C2 | a_y,max is used as a cornering limit (`lateral_limit`). Table entries with `limit_y == 'power'` (above about 265 km/h; 0 at 315 km/h) are replaced by the tyres' grip, extrapolated with normal load (μ_eff linear in N, fitted on the top six tyre-limited entries). Lock-limited entries are kept. | With the raw table, v_ss(κ) < v_top for every κ > 0 and a_y = 0 gives 0/0. This produced the NaN profile that §9 reports for `answer_key` start (3). With c_v applied, Q held 0.9·v_top ≈ 283 km/h on straights. The power limit belongs in the forward pass. |
| C3 | c_v scales the grip-limited backward profile only: target = min(v_top, c_v·V_grip). | Read literally, c_v·v_QSS also scales the power-limited straight-line speed, so Q would brake from the first step at A1 and on every null straight. The braking branches then decelerate at about c_v²·a_dec, the margin a P controller needs. |
| C4 | The policy's target has no forward pass. `forward_profile` (the forward pass from (s0, v0)) gives `offline_prediction`. **Amended by R22 B and C** ("Dry-run rulings R20-R28" below): the forward pass uses the model's signed steady-turn acceleration, and the offline onset replays Q's law on the 1-D model. | When the target is out of reach, the pedal saturates at the traction limit, which is what the forward pass describes. Keeping it out of the target lets one row per stimulus serve every approach and phase. |
| C5 | **Replaced by R22 A** (below: the speed is predicted through the pedal lag, φ = 0.482, and the command solved by bisection; the measured-acceleration predictor had loop gain 2.8). As built: the pedal is 0.5 per m/s on [target read 0.075 s ahead] − [v + 0.075·a_x (the env's measured accel)], plus feed-forward. The feed-forward is the model's steady straight-line pedal for the target's acceleration at the read-ahead point (`pedal_map`). | Plain P on T(s) is pinned at full throttle until 3 m/s from the braking branch. The 50 ms pedal lag then cost 2–3 steps, and Q entered 5% above its own target at c_v = 0.95, which would fail the 2-point `offline_match`. With C5, v_e is within 0.5% of the target and v_min within 2% in all 42 cells. |
| C6 | "Clipped" means the per-axle friction ellipse. The pedal lies between the brake-saturation pedal for a_dec·room_f and the full-throttle pedal for a_acc·room_r, where room = √(1−sin²(C·atan(B·α))) at each axle's current slip angle (`axle_room`). It never goes beyond saturation; the key never commands −1. | Without the clip, full throttle out of a hairpin spins the car. A table-based ellipse is unusable because a_y,max → 0 at top speed. |
| C7 | Steering uses pure pursuit from the car's reference point to the centreline point at s + max(8 m, 0.6 s·v). The curvature 2 sin α/l_d is converted with the model's steady-state map (kinematic angle plus `steer_correction`, an understeer table from Newton on dvy = dr = 0) and clipped to the front tyre's peak slip (7.15°) and the lock. | The rear-axle kinematic version left the road on R100/90 at 215 km/h (\|n\| 5.26 m) and R171/90 at A1. Full lock on R6/180 put the front at 14–22° slip, a slip termination. |
| C8 | Qp holds κ_1 for d < d_1, interpolates linearly in d, and holds κ_40 to 500 m. The profile's last point is at v_ss(κ_40), the steady state of a held curvature. | Protocol 7 wording. |
| C9 | In c1, braking-branch knots (T falling, below the cornering limit) take T(s − shift), capped by the cornering limit at s. The corner and exit are unchanged. **Amended by R23:** the shift is per cell, 15 m plus Q's offline-predicted onset lead over the u = 0 nominal key (`c1_shifts`), so c1 is 15 m late against the key. | "Braking branch moved 15 m towards the corner". |
| C10 | In c2, \|κ\| ≤ 1/12 applies to the speed target only; steering follows the real road. | "Reads R < 12 m as 12 m". |
| C11 | c3 goes flat out until d_c ≤ τ_c·v (and v > v_e), then brakes at the constant fraction k = (D(v) − D(v_e))/d_c of its braking capacity, where D is the a_dec braking distance. It hands over to Q once Q's read-ahead point is in the corner. **Amended by R26:** a pure trigger braking on feed-forward only, v_e Q's own (lawful) entry speed; τ_c is the smallest value with k ≤ 1 on every tier A/B braking cell (`tau_c_k1`, replaying c3's own law on the 1-D model); `fit_tau_c` (τ fitted to the key's d_b) is gone. | Constant deceleration demanded more than a_dec at low speed: 11% over v_e at τ = 1 s. Without the hand-over, c3 spun braking on the ramp of R87–R109 corners, where Q's braking lies inside the ramp and T(s_c0) = v_top. Gentle braking (long τ) passes β_f > 0.3 late, so d_b(τ) is not monotone. |
| C12 | K returns the buffer's mean action; the sampled action is kept as `act_sampled`. `k` is static (`make_knn_policy(k)`). The buffer starts from `racing.reset` (random s, 60–100% of `adapters.reference_speed`), uses stochastic actions, and runs one lap or until termination. The seed is `stream_key('knn_buffer')` folded with the agent id. | A memoriser of the deterministic policy, which the classes compare with. A traced k cannot drive `top_k`. |
| C13 | `tune_cv` completion means no termination before s_c1 + 250 m (`completion`). Time-censored runs count as complete. | The P4 definition. |
| C14 | The detrending spline is the penalised cubic B-spline of R's `smooth.spline`: `.nknots.smspl` knots, exact ∫B''B'' penalty, and λ bisected to trace 6 (checked by `hat_trace`). | It is the standard "smoothing spline with 6 df". The n-knot version needs an n × n eigenproblem and is indistinguishable at 6 df. |
| C15 | Π uses the Lomb–Scargle periodogram of the residual against k(d) (non-uniform in d), on a 0.01 cycle/index grid over 0.2–2. | k(d) ∝ d^(2/3). |
| C16 | Scans store the raw policy mean (the network's read-out). A_alias uses the command the plant receives, the mean clipped to [−1, 1]; both are switchable with `clip`. | With raw means, a ripple inside saturation would count as aliasing for agents but not for Q or Qp, whose commands are already clipped. |
| C17 | A_alias states are those in [t_b − 1 s, t_b] before s_c0. prev_action and accel come from the trace (`det_prev_action`, `det_accel`) when given, otherwise they are rebuilt as the env does: to 1e-6 and to 2e-2 m/s² (float32 states). | Trajectory has no accel field. |
| C18 | Shifted strips are built with `stimuli.shift_corner` and are deduplicated across states. Q's target rows for them come from `params_fn` (`q_params_for_kappa`). | Q's precomputed row belongs to the unshifted strip. |

**Interface changes (callers must adapt):**
- `QParams` = (v_target, s_corner, car: `CarTables`, c_v, k_v, lookahead_min, lookahead_time, lead_time, kappa_cap, tau_c). The skeleton's v_grid, a_dec and a_y_max moved into `car`. Build it with `q_params(bank, cells, tables, c_v, variant, tau_c)` or `q_params_for_kappa(kappa_rows, s_corner, tables, ...)`. `tables` may be `ModelTables` or a prebuilt `car_tables(tables)`; prebuilding saves 0.4 s per call.
- `KnnParams` lost `k` and gained `sq_norm`. Build it with `knn_params(buffer, obs_rms.mean, obs_rms.var)`. `make_knn_policy(k=1)`; use `KNN_K_FALLBACK` = 5 for the fallback.
- `q_target_profile(kappa, tables, c_v, ds, kappa_cap, brake_shift)` lost `v_start`; the forward pass is `forward_profile(v_target, kappa, s0, v0, tables)`.
- The signature is now `offline_prediction(v_target, kappa, landmarks, s0, v0, key_metrics, tables)`. `key_metrics` has v_e, v_min and d_b. The verdict is `offline_match(pred, measured)`.
- `tune_cv(bank, cells, tables, completion_fn=None)` returns `(c_v or None, {c_v: completion rate})`.
- `fit_tau_c(..., d_b_fn=None, tol, bracket, scan_step, max_iter)`. (Removed by R26: `tau_c_k1(bank, cells, tables, c_v)`.)
- `KnnBuffer` gained `act_sampled`, `meta`, `save` and `load`. `build_knn_buffer(agent, n_laps, n_obs, track=None)` builds on `collect_laps` and `assemble_buffer`; it must run only in the Integrate phase.
- The probe signatures are `open_loop_scan(..., aux, profile_fn, clip, ctx, params_fn)` and `aliasing_amplitude(..., deltas, **kw)`. The keywords are `profile_fn`, `params_fn`, `prev_action`, `accel`, `clip` and `ctx`. `aliasing_detail` returns `AliasResult` with per-state residuals.

**Findings for analysis:**
- Q's A_alias under the pre-registered definition is about 0.10, and Qp's about 0.09–0.10 on R9.4 and R22.9 at A1. All of it comes from the 2–3 states next to t_b, where the throttle-to-brake switch crosses the state as the corner moves (a step in δ). Every earlier state fits to < 1e-5. The protocol's "Q ≈ 0" therefore holds only away from the onset. I suggest reporting A_alias both as pre-registered and without states within 0.75Δ/v of t_b (from `AliasResult.residual`). The decision rule itself is unchanged.
- With the lock-limited low-speed a_y,max, v_ss for R < 7.7 m is about 20 km/h, so Q crawls through R6 and R7.5 at 19–21 km/h. That is the minimum kinematic turning radius of 7.45 m, not a bug.
- Q completes all 42 tier-A-like cells at all 4 phases at c_v 0.95 and 0.90. On Hockenheim condition (a) strips from a crude start (Q's own target speed 400 m before s_c0, zero yaw rate), it completes 17/20 at 0.95, with rear-slip spins in R45–50 corners, and 19/20 at 0.90. The one failure at 0.90 starts mid-corner at zero yaw rate. Expect `tune_cv` to choose 0.90 once tier D uses the real states.
- At c_v 0.90, c2 crashes only R6/180 (lock-limited, read as R12, so 48 km/h) among the 42 cells. c1 crashes all 38 braking cells; only the flat R171 and R300 cells complete. Its 15 m late branch meets the cornering limit as a step that the controller cannot absorb, so every crash is an overspeed crash with a late onset, which is the planted defect. Analysis should expect c1's P2 verdict to rest on crashes at every R, not only the tail.
- Timings with 2 XLA threads: `car_tables` 0.4 s; 74 target rows 0.25 s; Q 230k steps/s and Qp 49k steps/s (B 256 × 800, plain loop); K with a 40k buffer about 9k steps/s (B 64); `open_loop_scan` (1,400 d) 0.4 s; `aliasing_amplitude` 0.03 s after compile.


## Builder notes: `analysis.py`

Implemented in full (protocol 2.2–2.4, 7, 9): pair values, every decision rule P1–P8 with the
9.4 variants and the 2-of-last-3 stability rule, beta-binomial (exact likelihood, profile
interval; Clopper–Pearson when ρ̂ = 0), Firth logistic, Kendall τ-b, permutation Spearman,
bootstrap bands, the dry run (five simulated agents) and the power table. Its module docstring
lists decisions AD1–AD25, signed off as written (ruling R13; examples: an overspeed crash
is +∞ even before s_c0; the 3-of-8 and 5-of-8 rules scale with the number of runs; undefined
pairs count as not in band; P2(b) is not met without crash cells).

Interface (callers adapted in the integration): `run_table(agent, timesteps, rollouts, cells,
bank, keys: KeySource, tables, nulls=None)`; `CachedKeys(bank, cells, cache, anchor_id,
viability)` wraps `answer_key.key_for`; `key_table(cells, bank, keys)`; `AnalysisInputs`;
`decision_table(inp, confirmatory)`; `dry_run(key_table, tables, b)`; `power_table(key_table,
support, b, tables)`; `cond_table(runs, keys, d_ood)`; `phase_sd_table(runs, keys, qp_runs,
key_sd)`. Schemas: `RUN_COLUMNS` gains `v_e_star_ra`; `PAIR_COLUMNS`, `COND_COLUMNS` and
`DECISION_COLUMNS` as in the module.

## Integration (`scripts/psychophysics.py` and cross-module fixes)

**Stages.** Every stage of protocol 10 is a subcommand. Each logs to
`runs/psychophysics-<stage>.log`, keeps a one-line note in `runs/activity.md` while it runs,
takes `--workers` and `--force`, and resumes from what is on disk. Key stages solve in a pool of
spawned processes with a fixed worker count (default 4; rollouts and probes default 3).

| Stage | Step | What it does |
|---|---|---|
| `manifest` | 0 | `freeze_manifest` per agent; pending runs listed with the reason |
| `support` | 1 | `build_support` with R_clip from every final; a rebuild must reproduce the frozen corner tables |
| `stimuli` | 1 | `build_design`, `check_design_details`, the stand-in geometry build of tiers D/E1/E2 (I9), `cell_support_table`; `--real` appends tiers D/E from `real_states.json` |
| `rollouts --pilot-replication` | 2 | Harness gates, then the pilot replication of G1 and G2, then the throughput benchmark; writes `harness/step2.json` |
| `validate --measures` | 3 | Model tables, brake demand vs `tyre_forces`, the known-answer measures tests, one TO-OCP solution vs its float64 replay; writes `harness/step3.json` |
| `keys` | 4 | Layout (pool, then the global D rule), then P0/P1/P2 keys under the budget rule (P1 includes tier B D = 600's nominal keys, R17c; the phase cycles are refreshed from the nominal keys after each priority's pool, D26; with validation V7 failed: phase-matched keys for every braking cell, tier B D = 600's included, each item's after its nominal keys; at every phase but u = 0, so all 16 on P7's cells, R17b), then V1/V4/V6/V7 over every key; `--layout-only`, `--smoke` |
| `validate` | 4 | V1–V11 in the pool; V3 merged at V8's b; V5 sets `mirror_ok`; V2 classes get W = 800; timings → budget. `--force` re-solves every key of the pair and V7 jobs once (I21) |
| `controls` | 4/5 | c_v, τ_c, every control through the harness (`rollouts/control-<name>/0.npz`); `--knn-buffer` (exception 2) |
| `dryrun` | 5 | Key table, dry run, power table, P3 separation, the controls' required verdicts, the three hashes |
| `rollouts` | 6 | Post-tag only: per snapshot and group, protocol 4's replication table, budget rule 3; `--real-laps` (the standard R_DEV evaluation, allowed before the tag by ruling R1, see I12) |
| `anchored-keys` | 7 | Post-tag: re-anchored keys, one per deterministic phase of each cell (R17a; P0 generalist A1 tail/bulk and non-clipped below-support (R12), P1 specialists and tier B D = 600, P2 checkpoints) and viability solves. The nominal and phase-matched keys they start from come from the keys stage (R17c); a pre-flight reports any not cached, and a missing nominal key is solved once per cell in a pool before the jobs (the fallback) |
| `probes` | 6 | Post-tag: tier O scans and A_alias for agents, Q and Qp |
| `analyse` | 8 | Run tables (cached), key table, conditions, checkpoints and gates, P7 and P8 tables, density, decision table (`--confirmatory` once, needs G3 and S3) |
| `plots` | 8 | Psychometric curves per final |

**Decisions.**

| # | Decision | Reason |
|---|---|---|
| I1 | `answer_key.key_for` finds agent-free keys (nominal, backoff, phase, null, actuation, socp) by `expected_key_id` of the cell's current problem (x0 at the phase, W), not by the cell alias; mirrored keys likewise from the left member's id. Aliases remain for agent-dependent kinds. | An alias names the key solved last for a cell. V2 solves A2 cells at 400 and 800 steps, so the alias could name the wrong window. Test: `test_key_for_looks_up_the_cells_current_problem`. |
| I2 | After the layout, and on every save of the cells (V2 W changes included), tier G null cells are split per W: one null cell per (straight, v0, W), and each corner cell points to the null with its own W (`stimuli.split_nulls`, idempotent). | Protocol 5's catch trial and Δt_corner use the null with the same v0, width and W; the design had one null per (v0, width). |
| I3 | `CachedKeys.anchor_id` defaults to `'<agent>@<timesteps>'`, the alias `anchored_key(agent=…, snapshot=str(timesteps))` writes. | The default `'<agent>/<timesteps>'` matched no alias (and put a slash in a file name). |
| I4 | Viability solves carry the run's phase in their snapshot tag (`'<timesteps>-ph<k>'`); `analyse` maps them back to (agent, timesteps, cell, phase). | `viability_keys` records are not indexed by the run's phase. |
| I5 | `io.StageLog` writes the done/total line from a heartbeat thread at least once a minute, and edits `runs/activity.md` under an exclusive file lock. | One key solve can take longer than a minute; several stages and agents edit the notes file. |
| I6 | The layout runs `layout_a2(max_rounds=0)` per cell in the pool (warming the cruise and pinned s-OCPs), then the global rule once over the warm cache. | Same code path and result as a serial `layout_a2`, at pool speed. |
| I7 | Validation runs one job per validation pair (V2 both horizons, V3, V5, backed-off key) so no key is solved twice concurrently; `mirror_ok` is set once from the merged V5. | Each `v5_mirror` call sets the flag for its own pairs only. |
| I8 | The pilot replication and the kNN buffer are recorded in `manifest.lists['pre_tag_agent_rollouts']`, not as deviations. | They are the protocol's two exceptions, not departures from it. |
| I9 | Step 1's transplant and twin checks run on a geometry-only build of tiers D/E1/E2 from stand-in states placed at each requested s_target (G2's reach from its documented lap-1 crashes). These rows are not saved. | Ruling R1 allows the real-lap runs before the tag; until `rollouts --real-laps` has written the real states, step 1 checks these stand-ins. The strips depend on a state only through the lead-in start knot. |
| I10 | Step 2's gates use a randomly initialised actor with G1's obs_rms (read only) and the scripted lane keeper, on tier A R22.9/90 at A1 and R44.7/90 at A2 (D = 200 m, W = 400, P = v0·dt). | No trained agent may run on a stimulus before the tag. |
| I11 | Step 3's TO-OCP check solves the validation cell R40/90 at A2 with D = 200 m and W = 200 (nothing cached) through `full_recipe`, which also exercises the Q start. | A validation-sized solve, the same problem the answer-key tests solve. |
| I12 | `rollouts --real-laps` (the real-lap states of tiers D, E1, E2, the fidelity-gate traces and G3's crash corners) runs before the tag (`real_laps_gate`); after the tag it runs only on the tagged code. Each (agent, circuit) run is logged as a "standard evaluation (R1)" entry in `manifest.lists['standard_evaluations']` as soon as its lap file is written. Every other agent rollout stays tag-gated. | Ruling R1: these runs repeat the standard R_DEV evaluation and put no protocol stimulus in front of an agent. E1's P0 keys can now be solved before the tag. |
| I13 | Superseded by ruling R2: P8's sets are the regimes of the crash corners' R_seg under each agent's frozen q1 / q10 (`analysis.p8_set`); the named lists and the radius labels are gone. | G1's Sochi corner has R_seg 22.2 m, above G1's frozen q10 (21.3 m), so it stays bulk. G2's YasMarina (20.8 m) and Zandvoort (28.0 m) move to bulk. |
| I14 | New analysis builders `ladder_table`, `twin_table`, `density_table`, `alias_table` (tested on synthetic tables). The fidelity gate compares distance travelled on the strip with the real lap's, scaled by ds_source / 2 (D2). | AnalysisInputs needed them and no module built them. |
| I15 | Tier O scans start at d_b^agent(v) + 50 m, with d_b^agent(v) the agent's median deterministic onset on tier A/B cells of the same geometry at the approach speed nearest v, else d_b^phys(v, v_ss(R)). | Protocol 4 does not say where d_b^agent(v) comes from. |
| I16 | `rollout.pilot_replication` also returns s_crash, v_crash and v_min per cell; `agents` records why a run is pending. | Report the pilot's crash speed; make pending agents visible. |
| I17 | The float32 interp test modules switch x64 off for themselves (module-scoped autouse fixture) and restore it; `rollout._KEEPER_D` is a NumPy constant. | `tests/test_dynamics.py` and `tests/test_env.py` switch x64 on at import, so in a full-suite run 22 interp tests failed on float64 state. |
| I18 | `analysis.p5` is "not testable" when no pair has both members' e_ve (no keys yet). | It returned "refute" from an empty set. |
| I19 | Start (3)'s Q rollout uses Q's tuned c_v (controls/tuning.json) once the controls stage has tuned it, else `answer_key.Q_CV_DEFAULT` (0.95), and the keys stage logs which (amended in the review fixes, RF20). | Protocol 7 defines Q with the tuned c_v; the stage order tunes c_v before validate and keys. |
| I20 | The controls stage retunes c_v once tier D condition (a) exists, retries τ_c until the nominal key at R22.9/90/A1 exists, and redoes a control's runs when c_v or τ_c changed. | Protocol 7 tunes on tiers A and D (a); τ_c needs the key. |
| I21 | `validate --force` reaches the pair and V7 jobs: `_key_job` passes `force` to `_validate_pair` and `_validate_v7_cell`, which read the cache through `answer_key.FreshCache`. Records cached before the job read as absent, so each key (V2's horizons, nominal, actuation, the right-hand member, backed-off, phase) is solved once and later reads in the job see the new record; s-OCPs, aliases and flags read through, as with `full_recipe(force=True)`. | `_key_job` built `force` and dropped it for both jobs, and `v3_objective` had no way to pass it on, so `--force` re-solved nothing there. Forcing each call instead would solve the left nominal key up to four times per pair. Tests: `test_validate_jobs_pass_force_to_the_pair_and_v7_jobs`, `test_validate_pair_force_hands_every_key_call_one_fresh_view`, `test_fresh_cache_hides_records_cached_before_it`. |

**Not yet exercised** (needs keys or rollouts that do not exist yet): every key-solving stage
beyond the one smoke solve through the pool, the dry run on the real design, and every post-tag
stage. Two plumbing runs in a temporary output root (never under `outputs/`) did exercise the
controls stage on a fake layout (Q, Qp, c1, c2 through the harness: 1648 / 432 / 1648 / 860
runs; Q and Qp complete 100%, c1 4.9%, c2 94.4%) with `_control_verdicts` on an empty key
cache, and the post-tag rollout job with a randomly initialised actor (2216 det and 3264
matched-noise runs) through `analysis.run_table` (5480 runs in 4 s) and `decision_table`. Their pure helpers are tested
(`_split_nulls`, key sets and the budget projection were checked on the real design: tier A 74
nominal keys, E3 28, tier C 23, recovery 12, 20 phase geometries; P0 24.6 CPU-h at the
protocol's default timings).

## Review fixes (fidelity, numerics and hygiene reviews)

Three adversarial reviews of the harness found 3 blockers, 19 majors and 39 minors. Every
finding was checked against the code; the fixes below each have a regression test
(`tests/test_interp_review.py`, `tests/test_interp_script.py`, and additions to the existing
test modules). Refuted and deferred findings, with their evidence, are in the integration
report of this round.

| # | Fix | Where |
|---|---|---|
| RF1 | Key measures are recomputed from the cached solution in `CachedKeys` with the current measures code and tables, and a corner key's catch-trial onset d_resp* is measured against the null key of its null cell (same v0, width, W). Δd_resp, and so P2(a), is now defined. | `analysis.CachedKeys` (fidelity B1, numerics M4) |
| RF2 | Tier D's s-OCP columns are shared by every runner's cell of a strip from the S1-state cell that holds the solved s-OCP; `cond_table` takes the braking class and v_min* from rows with a solved s-OCP, never from whichever row comes last. | `analysis.key_table`, `_share_tier_d_socp`, `cond_table` (fidelity B2) |
| RF3 | P4 is "not testable" (class -1) when a single-factor F1 condition fails and its or an F6 condition's d_OOD is missing; the recipe claim ignores such seeds. S2's and S3's buffers are built after the tag (`controls --knn-buffer --agent S2 --agent S3`); K's and Q's P4 use S1's buffer with S1's obs_rms. | `analysis.p4_classify`, script `_knn_buffer`, `_d_ood_from_det` (fidelity B3) |
| RF4 | T's "every error within τ" (`analysis.tape_errors`, against the key's τ for v_e, v_min, d_b) and Q's "offline_match" (`controls.offline_match_runs`, every deterministic tier A and B run against Q's own forward profile) are computed. | fidelity M1 |
| RF5 | Protocol 6.3 runs first in `anchored-keys`: every final's u = 0 return against J*; beaten cells are re-solved warm from every final's u = 0 run (`resolve_beaten_key`; a mirrored key is re-solved as the right-hand cell's own problem) and a deviation is logged per cell. The run-table cache keys on the key cache, so errors are recomputed. | script `_beaten_jobs`, `answer_key.resolve_beaten_key` (fidelity M2, numerics M1) |
| RF6 | Re-anchored keys: P1 adds the generalist finals' tier B D = 600 cells; P2 re-anchors four checkpoints per generalist (first gate-passing, two log-spaced, final) unless the finals show re-anchoring immaterial or budget rule 1 cut P2. | script `anchored_cells`, `select_p6_checkpoints` (fidelity M3, M4) |
| RF7 | A failed P3 separation check demotes G3's P3 to secondary (`AnalysisInputs.demote_p3`, from the manifest). | fidelity M5 |
| RF8 | G3's own R_DEV crash corners: `rollouts --real-laps` runs G3's deterministic laps on R_DEV and writes `stimuli/crash_pairs_extra.json` (`stimuli.crash_pairs_from_laps`); `stimuli.crash_pairs()` adds them to every consumer; their P8 set comes from G3's frozen q1 / q10 (`analysis.p8_set`; by ruling R2, every agent's units now follow this rule). | fidelity M6 |
| RF9 | YasMarina diagnosis (3.2): `rollouts --yasmarina` (post-tag) writes `analysis/yasmarina.json`; a knife-edge unit stays in P8 as unreproduced (`p8_units(knife_edge=)`). | fidelity M7 |
| RF10 | Completion on a tier G straight is no termination within W (not reaching the virtual corner's s_c1 + 250 m). New metric `v_max` (top speed over the run) is the v_top covariate. | `measures.response_metrics`, `analysis.v_top` (fidelity M8, minor 7) |
| RF11 | Protocol 8's gate is applied to the finals and the two history snapshots before P1–P3 are read; gentle cells and tier G straights must each reach 90%, and an undefined oscillation index fails. | `analysis.decisions`, `checkpoint_gate` (fidelity M9, minor 8) |
| RF12 | V8's bound at s_c0 is on the speed interpolated between nodes (a linear constraint), not at the nearest node; the s-OCP has no free lead-in (the pin is node 0, `SOCP_LEAD_IN` = 0), so nothing before the pin leaks into the solution; s-OCP ids carry the formulation. | `answer_key.solve_min_time_pinned(vx_lin=)`, `socp_from_state`, `socp_extra` (numerics M2, m2) |
| RF13 | V6 groups by approach state: d_b* only at a fixed (v0, D); at A2 v_e* and v_min* across the bins' v0; never across W, direction or equal R. | `answer_key._group_key` (numerics M3) |
| RF14 | Cached measures carry `measures_hash`; `refresh_stale` recomputes stale ones before the acceptance checks and lists keys whose multi-basin measures moved (re-solve with `--force`); a cached s-OCP's measures are recomputed on every hit with the caller's tables. | `answer_key.refresh_measures` (numerics M4) |
| RF15 | Post-tag stages check the frozen code against the tag and the design against the pre-registration record that `dryrun` writes into this file (below; a logged design deviation may replace it); stimuli, keys, validate and controls refuse after the tag unless run with `--post-tag-deviation REASON`, which logs a dated deviation (and every design they save as a design deviation, whose hash the checks then accept); dryrun always refuses; runs whose s0 or W is not their cell's raise in `run_table`; rollout files record each group's code and design hash. | script `_require_tag`, `prereg_problems`, `analysis.check_runs_match_cells` (hygiene M1, minor 13) |
| RF16 | The manifest is updated under a lock with a re-read (`agents.update_manifest`); a rollout job marks its snapshot evaluated before running; rollout files are merged under a per-file lock; after the tag `manifest` no longer overwrites the tagged commit and code hash. | hygiene M2, M3, minor 3 |
| RF17 | Analysis run tables are cached with a stamp of the rollout file, key cache, design and code, and always rebuilt under `--confirmatory`; the confirmatory run refuses `--agent` and partial inputs (`confirmatory_problems`). | hygiene M4, M5 |
| RF18 | Cornering limit: v_ss and the infeasibility bound use a_y,max with the power-limited entries replaced by tyre grip (`measures.cornering_limit`, shared with controls C2). | numerics m1 |
| RF19 | `_was_capped` ignores the refine round; V9 counts slip-active steps on one scale (≥ 8.0°) for both keys; the guess tail follows a curved road (curvature feed-forward, 1 rad/s loop, integral on n); failed s-OCPs are flagged in the layout (flat, `layout_socp_failed`), give W = 800, and leave V8's medians (listed). | numerics m3, m4, m5, m8 |
| RF20 | Start (3) uses the tuned c_v (I19); stochastic traces keep util and β_f in float32; checkpoints run REPLICATION's 4 phases (k = 0, 4, 8, 12 on P7 cells); controls run tier D from S1's state, K falls back to k = 5 when C(a) < 0.9 and the F6 halving is a stage option (`stimuli --f6-scale 0.5`); P2b uses the phase-matched key values; P2d gets the centreline QSS v_min; Δt_corner against the s-OCP is computed; the lock-limited sensitivity reaches P3's runs; P7's spreads share one estimator (ddof 0); each grammar's density uses its own SD; the tier G rule for specialists is applied. | numerics m6, m9; hygiene minor 8; fidelity minors 1–6, 9–12 |
| RF21 | Hygiene: the step-2 random actor has the agents' architecture (G1's MLP config, not SimBa); the x64 gate tests that the harness refuses float64; `--threads` overrides an inherited XLA_FLAGS; key stages default to 5 workers; atomic writes keep mode 0644 (mkstemp's 0600 had spread to outputs and runs/activity.md); one activity line per process, without markup; `--force` reaches the solvers and rollouts; probes merge per agent and assert float32; step 2 and step 3 gate the later stages; the buffer and real laps are written atomically; `load_agent` refuses a config with obs noise or without normalisation. | hygiene minors 1–7, 9, 10, 12 |


## Rulings R1-R13 (protocol 12)

How the harness implements each pre-tag ruling of protocol section 12. Tests named `test_R<n>_*` pin each behaviour change; they are in `tests/test_interp_rulings.py` unless noted.

| # | Ruling | Implemented by | Tests |
|---|---|---|---|
| R1 | Real-lap rollouts are allowed before the tag and logged as standard evaluations | `scripts/psychophysics.py`: `real_laps_gate` (no tag needed; after the tag, the tagged code is required), `stage_rollouts` (only `--real-laps` skips the tag; `--yasmarina` and step 6 stay gated), `_real_laps` writes one `r1_entry` per (agent, circuit) run to `manifest.lists['standard_evaluations']` (category "standard evaluation (R1)", snapshot, outcome, lap file, before/after the tag) when its lap file is written; a resume that loads a lap file with no entry (the run stopped between the two writes) logs it once, marked `logged_on_resume` (`_log_r1_once`); `R1_CONVENTION` names eval/laps.py's convention as run by diagnose_crashes.py (`agents.real_lap`, without eval's 200-step margin); CLI help of `--real-laps`; note I12 | `test_R1_real_laps_run_before_the_tag_everything_else_stays_gated`, `test_R1_each_real_lap_is_logged_as_a_standard_evaluation`, `test_R1_a_lap_written_but_not_logged_is_logged_on_resume`, `test_R1_cli_help_names_the_ruling` |
| R2 | The crash corner is the segment corner containing the crash; its regime comes from its R_seg; proportional P8 thresholds | `stimuli.CrashPair` has no radius label (the 12.7 / 12.1 m labels, `LABEL_MISMATCH` and the `label_mismatch` cell flags are gone; an old `label_R` in crash_pairs_extra.json is ignored); `stimuli.crash_corner` (D15), `stimuli.crash_unit_R_seg`; `analysis.p8_set(agent, R_seg, regimes)` is the regime under the agent's frozen q1 / q10 for every agent (`P8_SETS` removed); script `_p8_units`; `analysis.p8_tail_thresholds` (ceil(3n/5), floor(n/5) in integer arithmetic) and `p8_bulk_minimum` (ceil(2n/3)); `analysis.p8`: P8-tail is "not testable" for n < 3; P8-bulk keeps the protocol's rule (context in ≥ 2/3 of the reproduced bulk units, as ≥ ceil(2n/3)) and is not testable only with none reproduced (R2 changes the tail rule only). P8-tail's n leaves out transplants that fail the fidelity gate (3.2: excluded from P8, listed as `fidelity_failed`); knife-edge units and units whose gate could not be evaluated (no real lap, state index entry, real rung or u = 0 run: script `_fidelity` returns None, `ladder_table` column `fidelity_missing`, a unit without cells keeps an empty real row) stay in as unreproduced, are listed as `fidelity_missing` and logged by `analyse`. With step 1's frozen values, G2's tail is Norisring 874 (R_seg 15.1), Melbourne 2531 (15.1) and Catalunya 4168 (14.8), so n = 3; YasMarina (20.8) and Zandvoort (28.0) are bulk. The computed R_seg of these two (20.78 and 28.02 m) differ from the ruling's approximate 20.4 and 27.5 m (brief-support §3); both are above G2's q10 (17.67 m), so the regime is the same | `test_R2_p8_thresholds_are_proportional` (n = 5 gives 3 and 1), `test_R2_bulk_context_share_is_proportional`, `test_R2_leaves_the_bulk_rule_testable_below_3_units`, `test_R2_a_gate_with_missing_inputs_keeps_the_unit_in_p8_tail`, `test_R2_ladder_table_marks_unevaluated_gates_and_keeps_empty_units`, `test_R2_crash_corner_regimes_come_from_their_R_seg`; `test_interp_stimuli.py::test_crash_corners_carry_no_radius_labels`; `test_interp_review.py::test_M6_g3_crash_units_join_p8` |
| R3 | theta_eff misses from 2 m knot sampling are accepted and listed | `stimuli.check_design_details`: every spec's continuous trapezoid must turn its nominal angle (`spec_area_deg`, kappa_peak times the area under its vertices, to 1e-6°; this is the check that catches a spec whose parameters disagree with its angle, since the samples follow the spec). A miss of more than 0.3° is accepted (`accepted`, `accepted_by='R3'`) only on a synthetic or pilot row whose theta_eff equals the knot samples' own area (within `THETA_SAMPLE_AREA_TOL_DEG` = 0.02°; the C2 spline integrates exactly as the samples) and whose samples' area lies within the kinks' sampling bound of the nominal angle (`theta_sampling_bound_deg` = kappa_peak ds² / (2 l_r): each of the four kinks between knots moves the trapezoid sum by at most \|Δslope\| ds² / 8). Any other miss fails, with its reason (`why`). On the saved design the 6 known rows miss 0.34° (bound 1.03°) and 0.44° (bound 1.42°); the worst miss/bound over all 160 rows is 0.49; theta_eff stays in `Geometry`; script `check_line` logs "N rows accepted per R3 (listed)" | `test_R3_R4_accepted_rows_pass_and_are_listed`, `test_R3_a_spec_that_disagrees_with_its_angle_still_fails`; `test_interp_stimuli.py::test_design_checks` (the 6 known rows, their spec area and bound) |
| R4 | Twins match R_seg and angle; the length shortfall is listed and is a covariate | `stimuli.check_design_details`: a twin shorter than its corner by more than 2 m is accepted (`accepted_by='R4'`, `length_shortfall` per twin, worst R and angle residuals of the accepted rows) only if it matches R_seg (relative) and the angle (deg) to < 1e-3 (`TWIN_R4_TOL`, `twin_r4_match`; the twin fit converges to 1e-4 and 1e-3); a longer twin, a miss outside 2% / 1°, or a length miss on a twin matched only within 2% / 1° fails. On the stand-in build the 16 short twins match to ≤ 6.0e-5 and ≤ 5.5e-4°; `stimuli.twin_length_shortfall`; `build_real_design` flags an accepted twin `twin_length_short` (any other miss `twin_fit_fail`); script `_p8_units` carries `length_shortfall` per corner; `analysis.TWIN_COLUMNS['length_shortfall']`, `analysis.twin_table`, and `analysis.twin_shortfall_covariate` in P8-twin's details (twins short by more than 2 m, agreement among them and among the rest, Spearman of the shortfall with \|e_ve(iso) − e_ve(twin)\|; the verdict does not use it) | `test_R3_R4_accepted_rows_pass_and_are_listed`, `test_R4_shortfall_is_a_twin_transfer_covariate`; `test_interp_analysis.py::test_twin_table_uses_the_real_rung_v_b_for_the_approach`; `test_interp_stimuli.py::test_build_real_design_with_stand_in_states` (Norisring 874's twin) |
| R5 | The ≥ 10° filter and v_typ pooling stay | No change: `support.TYPICAL_MIN_THETA_DEG` = 10 (defaults of `l_typ_table` and `v_typ_table`), `V_TYP_MIN_COUNT` = 20 | `test_R5_typical_ramp_and_speed_keep_the_10_deg_filter`; `test_interp_support.py::test_l_typ_table_and_interpolation`, `::test_v_typ_rounds_to_the_a2_choices` |
| R6 | No unsettled flag when the corner is in view at the start | `measures.response_metrics`: `unsettled` is False whenever `sight_at_start`; unchanged otherwise. Changing measures.py changes `answer_key.measures_hash`, so cached key measures are refreshed by `refresh_stale` | `test_interp_measures.py::test_R6_unsettled_does_not_apply_when_the_corner_is_in_view_at_the_start` |
| R7 | Budget rule 2 cuts 3 geometries × 2 approaches, keys only | script `budget_cut` (tier A A1/A2, 180°, R ≥ 35.8), `_cut`, `flag_budget_cuts` (flags `budget_cut` and `no_key`, idempotent; the flags follow the current projection and are cleared when rule 2 no longer fires, so no stale `no_key` hides a key `_cut` now solves), `rollout_cells` and `_rollout_job` (cut cells are rolled out), `anchored_cells` (no re-anchored key for them); `answer_key.DEFAULT_CUTS` (6 nominal and 6 backed-off keys, one direction) and the rule 2 message; the keys stage credits rule 2 on its own item sets (`_rule2_cut_counts`: 6 nominal keys with the mirror shortcut, 12 without, and 3 phase-matched keys per non-180° phase cell; backed-off stays 6 against its fixed one-direction 66); `analysis.NO_KEY_FLAG`, `has_no_key`, `_NoKeyFilter` (hides every key of a cut cell in `run_table` and `key_table`), `key_status = 'no_key'`, `excluded = 'no_key'` in `pair_values` (in `EXCL_KEY`, not `EXCL_ALL`), no viability solves for them (`viability_targets`) | `test_R7_rule_2_cuts_keys_of_3_geometries_x_2_approaches_only` (with the clearing), `test_R7_rule_2_credit_counts_the_cut_keys_in_both_directions`, `test_R7_cut_cells_carry_no_key_and_leave_key_based_errors` |
| R8 | Crash typing's ellipse search starts at s_c0 − 50 | Already so: `measures.crash_type` (`V_MIN_MARGIN` = 50, M11) | `test_interp_measures.py::test_R8_crash_typing_ellipse_search_starts_at_s_c0_minus_50` |
| R9 | R_clip = 50 / (10σ − \|μ\|) | Already so: `support.r_clip` (S2) | `test_R9_r_clip_uses_the_sign_that_clips_first`; `test_interp_support.py::test_r_clip_is_the_worse_sign` |
| R10 | P4 reads single-factor F1 conditions; a vacuous class 2 is flagged and reported | Already so: `analysis.p4_classify` (`details['vacuous']`), passed into the P4 decision's details by `analysis.p4` (AD16) | `test_interp_analysis.py::test_R10_vacuous_class_2_is_flagged_in_the_p4_decision` |
| R11 | P6a confirm also needs permutation p ≤ 0.05 | Already so: `analysis.p6` (`P6A_ALPHA`, AD18) | `test_interp_analysis.py::test_R11_p6a_confirm_needs_the_permutation_p` |
| R12 | P0 re-anchors the non-clipped below-support A1 cells too | script `anchored_cells`: at P0 a generalist's A1 tail, bulk and below-support (R_seg < q1) cells that are not κ-clipped. With rollouts present, the anchored-keys stage passes each final's κ-clipped pairs (`analysis.kappa_clipped_pairs`, the binding run-time definition of 2.3 and AD9). Without them the design-time regime 'below' is used. P2 (P6c checkpoints) keeps tail and bulk; `stage_anchored_keys` builds the finals' run tables first; the budget projection counts P0's re-anchored keys on the design once the regimes are frozen (`_anchored_g_count`: tail, bulk and R12's below-support cells alike, 196 = 160 + 36 cells on the current design; the protocol's 96 before then; ruling R17a makes it one key per deterministic phase of each cell, 1504 keys; `projection['anchored_G']` keeps the count, its source and R12's share, `_r12_extra_anchored`). Analysis: `uses_ra` (A1, tier B D = 600) picks e_ve^RA (`band_errors`, `_pair_row`). New run column `ra_key` and pair column `ra_fallback` record any run that fell back to the nominal key; P1b reports `ra_fallback_pairs` over its A1 Kendall-tau scope (0 once the keys exist), and `history_ra_fallback_pairs` for the two history snapshots its stability check re-reads: R12 is scoped to P0, so a history snapshot that is also a P6c checkpoint (P2 keys: tail and bulk only) still mixes keys in its tau; this is reported, not changed (open for the study owner); AD5 is updated | `test_R12_p0_re_anchors_the_non_clipped_below_support_a1_cells`, `test_R12_run_and_pair_tables_record_the_re_anchored_key`, `test_R12_budget_counts_the_extra_anchored_keys`; `test_interp_analysis.py::test_R12_p1b_reads_e_ve_ra_throughout_a1`, `::test_R12_p1b_reports_the_history_snapshots_ra_fallback` |
| R13 | AD1–AD25 signed off as written | The `analysis` module docstring says so; no decision changed | `test_R13_analysis_decisions_are_signed_off` |

**Review of the rulings implementation.** A review found two majors and seven minors; each was reproduced and is fixed with a regression test (named in the table above) unless noted.

| # | Finding | Fix |
|---|---|---|
| M1 | R3's acceptance could never fail: theta_eff always equals the knot samples' area, so any miss (a ×1.10 curvature, a plateau 5 m too long) passed | Spec area to 1e-6°, the kinks' sampling bound, synthetic and pilot rows only (R3 row) |
| M2 | P8-bulk became "not testable" for n < 3, which R2 does not rule | Bulk rule as pre-registered: not testable only with no reproduced unit (R2 row) |
| m1 | R4 accepted twins within 2% / 1° instead of < 1e-3; accepted twins kept `twin_fit_fail` | `TWIN_R4_TOL`; flag `twin_length_short` (R4 row) |
| m2 | Missing fidelity inputs counted as a failed gate and shrank P8-tail's n | `fidelity_missing`: kept in n as unreproduced, listed and logged (R2 row) |
| m3 | R12 holds at the finals only; P1b's history snapshots that are P6c checkpoints still mix keys | Reported (`history_ra_fallback_pairs`), not changed: R12 is scoped to P0. Open for the study owner |
| m4 | R7's flags were never cleared when rule 2 stopped firing | Flags follow the current projection (R7 row) |
| m5 | Rule 2 credited 6 nominal keys without the mirror shortcut (12 cut); `anchored_G` added design-counted R12 cells to the protocol's fixed 96 | `_rule2_cut_counts`; `_anchored_g_count` (196 on the current design) (R7, R12 rows) |
| m6 | A lap written but not logged before a crash was never logged; `R1_CONVENTION` cited eval/laps.py's step budget | Logged once on resume; convention text (R1 row) |
| m7 | I9 still said "pending ruling Q1"; section 7's note IDs R1-R13 clashed with the rulings; the ruling's approximate R_seg | I9 text; notes renamed RO1-RO13; computed R_seg noted (R2 row, section 6) |

## Ruling R14 and the Q braking clip (orchestrator, 2026-10-19)

- `controls.pedal_command`: the brake clip is a_dec(v) × min(front room, rear room). It was the front room only, and braking from S1's mid-corner real-lap states (yaw rate −0.7 rad/s, sideslip 2.9 m/s at 197 km/h) spun Q on 5 of 20 tier D condition-(a) corners. With the fix, 19 of 20 complete at every c_v.
- `controls.tune_cv(..., gate_cells=)` and `gate_completion`: R14's rule is 100% on tier A plus P4's C ≥ 0.9 on tier D (a). The script passes tier D (a) as `gate_cells`. Test: `test_R14_tune_cv_gates_tier_d_by_p4_competence`.
- Q's remaining failure is Hockenheim corner 6 (D/6). It starts at 295 km/h, 85 m before a kink whose centreline radius is 19 m; Q's own envelope there is 285 km/h. It is reported as a limit of centreline driving.

## V7 remedy: phase-matched keys for every braking cell (protocol 6.5, ruling R15)

Validation V7 failed on speeds: on 6 of its 10 tier A cells, v_e* at phases u_1–u_3 differs from u_0's by up to 0.67 m/s (the limit is 0.5 km/h). Protocol 6.5's remedy is implemented as below. While V7 passes, the phase sets, the jobs, the projection and the errors are as before. Three things change whatever V7 says: P7's key null no longer reads the validation phase keys on the R 28.6 and 55.9 V7 cells (see **P7**), run and pair tables gain the columns below, and the run-table stamp gains `speed_at_phase` (the stamp's code hash changes with any code change, so tables are rebuilt once either way).

- **Verdict.** `analysis.v7_speeds_failed(manifest.validation)` (script `_v7_failed`) is True when validation V7 ran, failed and lists failed cells (`details['failed']`; the real manifest lists 6). V7's only criterion is on speeds, so a listed failure is a speed failure. A V7 that could not be judged (no cell with phase keys, or an error: passed False with nothing listed) does not trigger the remedy; the keys stage logs it and keeps 6.1's set until validation judges V7 (review fix). The keys stage's acceptance V7 over every key is reported and is not the trigger.
- **Candidates** (script `_key_sets(..., v7_failed)`, `PHASE_ITEMS`): every cell with an agent-free nominal key, at that key's priority:
  - `phase` (P0): tier A at A1 and A2 (this includes the 12 tier B cells shared with A1), E1's real and isolated rungs, and the validation cells (tier V; review fix: the finals roll them out and P5 reads every deterministic pair, so they were the one class of braking cells with a nominal key and no phase keys; their nominal keys come from the validate stage; 20 of 28 brake, 60 keys);
  - `phase_P1`: E1 twins, E2 and E3;
  - `phase_P2`: tier C and tier B's recovery cells.

  Tier G nulls are flat. Tier D has only s-OCP keys, so there is no TO-OCP key to warm-start from. Tier B's other cells had only re-anchored keys and got no phase keys (ruling R17c later made tier B D = 600 a P1 nominal and phase item, `phase_B600`: see "Ruling R17c" below). None of the others get phase keys. Protocol 6.1's set (A1, R ≤ 18.3, `_core_phase_cell`) keeps its keys in every case.
- **Braking only** (`_nominal_braking`, `_phase_split`, `_phase_targets`): a candidate gets keys only if its cached nominal key brakes by 2.3 (`_braking`: a sustained brake and v_min* < v0 − τ). Flat cells, and cells whose nominal key is unsolved, are skipped.
- **Scheduling** (`stage_keys`, `_phase_jobs`): each item's jobs are planned after its nominal keys are solved, in the same run:
  - P0's jobs follow the tier A series and E1, after the existing phase-cycle refresh;
  - P1's jobs follow E1 twins, E2, E3 and tier D;
  - P2's jobs follow tier C and recovery.

  Before the P1 and P2 jobs, the phase cycles are refreshed from the new nominal keys (`_refresh_phase_cycles`, D26). The design is saved only when a P changed. Phase indices come from `_phase_indices` (u_1..u_3; 4, 8, 12 on 16-phase cells). `_key_job` is unchanged.
- **Budget** (`_budget(..., cache=)`): each phase item counts 3 keys per cell that brakes or whose nominal key is not solved yet, an upper bound. `projection['phase_keys']` lists, per item, the candidates, the counted cells and the cells not known yet. `phase_P1` and `phase_P2` are new `answer_key.BUDGET_ITEMS` at P1 and P2 (timing item "warm"), so rule 1 drops `phase_P2` along with the other P2 items. While V7 passes they are not in the projection.
- **Rule 2** (`_rule2_keeps_phase`, `_cut`, `_rule2_cut_counts(bank, sets, phase)`): "keep phase-matched keys for 180° cells only" applies to every phase item. A cell without a CornerSpec (a real road) is not a 180° cell. A `budget_cut` cell loses its phase keys along with its nominal key. The credit is 3 × the counted cells that `_cut` drops, item by item, so it equals what the stage then skips. On 6.1's set it is the old credit (3 per non-180° cell).
- **Errors** (`analysis.run_table(speed_at_phase=)`, AD23):
  - e_ve, e_vmin and e_vmin^line are against the onset errors' key: the key at the run's phase when one exists, else the u = 0 key.
  - e_ve_bo uses a backed-off key at the run's phase when one exists, else the u = 0 one. `KeySource.backoff_at` is optional; `CachedKeys.backoff_at` looks one up and never solves it. Backed-off keys are solved at u = 0 only, so none exists.
  - Without re-anchoring, e_ve_ra and v_e_star_ra follow e_ve's key, and e_ve_bo_ra is the run's own e_ve_bo (review fix, AD5). It was (1 + e_ve_ra)·v_e*/v_e*bo − 1 on every run, which on a fallback run scaled the backed-off edge by the phase key's ratio v_e*_k/v_e*, unlike every other approach; before the remedy the two were equal there.
  - New run columns `speed_at_phase` (the switch) and `bo_at_phase`. Together with `key_at_phase` and `v_e_star_k`, they name the key behind every error.
  - New pair column `speed_at_phase`. S and r (`v_e_star_band`) use the median v_e*_k of the runs' keys where e_ve is not re-anchored (AD5). `reanchor_immaterial` compares v_e*RA with that same key.
- **Where the switch is read.** The script's `_run_table` reads the manifest's verdict for the controls' verdicts, K's fallback, the anchored-keys stage and the probes. `stage_analyse` passes it to `_run_table_cached`, whose stamp includes it, so cached run tables are rebuilt when the verdict changes.
- **P7.** `_key_phase_sd` keeps P7's key null on R ≤ 18.3, as P7 defines it ("the key's own SD across its 4 phase-matched solutions (R ≤ 18.3 only)"), although the remedy also solves phase keys on P7's R 28.6 and 55.9 cells. This also applies while V7 passes: before, the left-hand R 28.6 and 55.9 A1 V7 cells contributed through their validation phase keys.
- **Logging.** `_phase_jobs` reports, per item, the cells that get keys, the flat or unsolved ones, and apart from those the cells with no cached nominal key (a job that errored or did not run), which a resumed `keys` run picks up (review fix).
- **Running stage.** A `keys` run started before this change plans with the code it loaded, so it solves 6.1's set only. Its later pools spawn workers that load `_key_job`, which is unchanged. The expanded set is solved by the next `keys` run, which resumes from the cache.
- **Tests** (`tests/test_interp_v7_remedy.py`): `test_v7_verdict_is_read_from_the_manifest_validation`, `test_v7_remedy_sets_every_nominal_key_cell_and_old_set_when_v7_passes`, `test_v7_remedy_skips_non_braking_and_unsolved_cells`, `test_v7_remedy_budget_counts_the_expanded_set_and_rule_2_credit`, `test_v7_remedy_keys_stage_solves_phase_keys_after_their_nominal_keys` (the stage with its pools stubbed), `test_v7_remedy_run_table_judges_speeds_against_the_phase_key`, `test_v7_holding_keeps_speed_errors_on_the_u0_key`, `test_reanchoring_is_judged_against_the_key_behind_e_ve`, `test_v7_remedy_reaches_the_cached_analysis_run_tables`, `test_p7_key_phase_sd_stays_on_r_up_to_18_3`; review fixes `test_v7_that_could_not_be_judged_does_not_trigger_the_remedy`, `test_v7_remedy_log_counts_cells_without_a_nominal_key_apart` and (`test_interp_analysis.py`) `test_a1_runs_without_a_reanchored_key_fall_back_to_their_own_e_ve_bo`.
- **Resolved by ruling R17** (the two review findings below were open for the owner; the code had followed R15's "when one exists" literally; see "Ruling R17" below):
  - *Re-anchored keys.* At A1 and on tier B D = 600 the band reads e_ve^RA, whose key is solved from the agent's u = 0 run (6.1), so runs at u_1..u_3 are judged against a key on the u = 0 control grid. The remedy therefore leaves P1b at A1, P2's set O and P2b's agent side as they were; it changes A2 (P1a, P1b at A2), tiers C and E (P8), P5, P3's β and the controls. Phase-matching them needs re-anchored keys from each phase's run (about 4× the A1 anchored keys). **Ruling R17a:** one re-anchored key per deterministic phase, each anchored on that phase's run.
  - *P7's 16-phase cells.* Phase keys exist at u_1..u_3 only (6.1; 4, 8 and 12 of 16), so on the 20 P7 cells 24 of each pair's 32 deterministic runs (26 with the two u = 0 runs) are judged against the u = 0 key, and the pair median mixes keys. At A1 the band uses e_ve^RA, so this reaches e_ve fallbacks and P5. Keys at every phase would add 12 per cell (240 keys on the 20 cells). **Ruling R17b:** keys at all 16 phases on the 20 P7 cells.

## Ruling R17: re-anchored keys per phase, and keys at every phase of P7's cells (protocol 12)

R17 resolves the V7 remedy's two open items (above). (a) Re-anchored keys are solved per deterministic phase, each anchored on that phase's own run, and every run is judged against the key of its phase. (b) P7's 20 cells get phase-matched keys at all 16 phases; P7's pre-registered null keeps its definition (the key's d_b SD over phases 0, 4, 8 and 12).

| # | Ruling | Implemented by | Tests (`tests/test_interp_r17.py`) |
|---|---|---|---|
| R17a | One re-anchored key per deterministic phase, anchored on that phase's own run; each run, and each stochastic run (phase j mod n, D9), uses the key of its phase; 6.1's fallbacks at that phase | **Keys** (`answer_key`): `anchor_base(cell, k, nominal)` is the key at phase k, i.e. the phase-matched key at u_k when one is cached and verified, else the u = 0 nominal key. Its d_b* and v_b* place s_L, its sustained brake gates re-anchoring, it is start (1), and it is what the run falls back to (6.1: the nominal key at that phase). `anchor_for(...)` finds x_{L−1} and a_{L−1} on the phase's run and returns an `Anchor`, or None when re-anchoring does not apply. `anchored_key(..., phase=k)` solves it with a0_fixed as before. A cached record is reused only when it was started from the same base key (`anchor['nominal']`; review finding 4): the id holds the problem, not the warm start, so a key anchored from the u = 0 key before the phase key was cached is re-solved from the phase key once it exists, and the new record replaces the old under the same id and alias. **Identity**: `anchored_key_id` adds `phase` to the id's extras (with a0, agent, snapshot), so the same state at two phases gives two keys. `KeyRecord.phase` = k, so the alias is `cell<id>_ph<k>_<agent>@<snapshot>`; the anchor dict records `phase`, `base_kind` and `base_phase`. **Stage**: `anchored_jobs` gives one key per (cell, deterministic phase) of the snapshot's run file (`_det_rows`). That is every phase at the finals (16 on P7 cells) and the 4-phase grid at the checkpoints (0, 4, 8, 12 on P7 cells). A cell's phases share one job (`phases`: (phase, row) pairs, in phase order; review finding 1), so a missing u = 0 nominal key (tier B D = 600's, which no keys-stage item solved before R17c; since R17c only a fallback) is solved once per (agent, cell) rather than by 4 or 16 concurrent workers (since the R17c review once per cell, in a pool before the jobs). `_anchored_job` solves each phase from `anchor_base` at that phase; a phase that raises is reported (`error`, one entry per failed phase) and the others keep their keys. 6.3's u = 0 rows (`_u0_rows`) are unchanged. **Analysis**: `KeySource.anchored(cell, agent, timesteps, phase)`, `CachedKeys.anchored` and `_NoKeyFilter` look up the run's phase only, never another phase's key; `KeyView` gains `key_id` and `phase`. New run columns are `ra_phase` (phase of the run the key is anchored on; −1 on a fallback) and `ra_key_id`. New pair columns are `ra_keys` (distinct re-anchored keys behind the deterministic runs) and `ra_off_phase` (runs whose key was anchored at another phase; 0 by construction). e_ve_ra, v_e_star_ra, the pair's v_e*RA median and e_ve_bo_ra (still scaled by the cell's u = 0 ratio v_e*/v_e*bo, 2.2) follow each run's own key. So do P1-tail's contrast, P1b at A1, P2's set O, P5 at A1, the P6 exception (`reanchor_immaterial`) and P6c. AD5's text records it. A run without its phase's key falls back to its own e_ve, as under R12: under the V7 remedy (the manifest's state) that is the nominal key at its phase, the phase-matched key where one is solved, else the u = 0 key (AD23); were V7 to hold, e_ve and so the fallback stay on the u = 0 key (2.2), consistent with the fallback's e_ve_bo. Stochastic runs at a checkpoint whose phase has no deterministic run (P7 cells: phases other than 0, 4, 8, 12) have no re-anchored key and fall back the same way; the band reads the deterministic runs only. **Budget**: `_anchored_g_count` counts one key per deterministic phase of each design cell (`_ra_phases_final`: 4, and 16 on P7 cells) and keeps `cells`, `u0_keys`, `p7_cells`, `p7_keys` and R12's `r12_below_support` (cells) and `r12_below_support_keys`; before the regimes are frozen it uses 4 × 96. P1 and P2 are counted on the design too (`_anchored_p12_count`, review finding 2), as `anchored_cells` selects them: `anchored_S` the specialists' A1 and tier B D = 600 cells at every phase of a final (16 on P7 cells), `anchored_B` the generalists' tier B D = 600 cells, `anchored_ckpt` the generalists' A1 tail and bulk cells × `P6C_CHECKPOINTS` (3; the final is P0) × 4 phases once the regimes are frozen, else the protocol's 216 × 4. The protocol's u = 0 estimates (50, 108, 216; `RA_U0_PROTOCOL`) were already below the stage's selection before R17. A new P1 item `nominal_B600` budgeted the anchored stage's cold u = 0 nominal solves of the tier B D = 600 cells (one per cell); since R17c it is a keys-stage item (below). `projection['anchored_phases']` records each item's keys, u = 0 count, P7 share and source. Viability solves are unchanged. | `test_R17a_anchored_key_identity_carries_the_phase`, `test_R17a_anchor_base_is_the_key_at_the_runs_phase`, `test_R17a_anchored_key_falls_back_to_the_key_at_that_phase`, `test_R17a_anchored_key_stores_one_record_per_phase` (the solver stubbed: two phases of one run give two records, ids and aliases; a repeat is a cache hit), `test_R17a_cache_and_analysis_find_the_key_of_each_phase`, `test_R17a_anchored_key_cache_hit_requires_the_same_base`, `test_R17a_anchored_jobs_one_per_deterministic_phase`, `test_R17a_anchored_job_solves_a_missing_nominal_key_once_per_cell`, `test_R17a_anchored_job_anchors_on_the_phase_run_from_the_phase_key`, `test_R17a_anchored_keys_stage_solves_a_key_per_phase` (the stage with its pool stubbed), `test_R17a_run_table_judges_each_run_against_its_phases_re_anchored_key`, `test_R17a_a_key_anchored_on_another_phases_run_is_counted`, `test_R17a_budget_counts_one_re_anchored_key_per_phase`, `test_R17a_budget_counts_p1_anchored_keys_as_the_stage_selects_them`; updated `test_interp_rulings.py::test_R12_budget_counts_the_extra_anchored_keys` (keys = 4 × cells) and the key-source doubles (`anchored(..., phase=0)`) |
| R17b | Phase-matched keys at all 16 phases of P7's 20 cells, warm from the nominal key; P7's null keeps phases 0, 4, 8, 12 | **Keys stage**: `_phase_key_indices(c)` gives every phase but u = 0: u_1..u_3 on a 4-phase cell, as before, and 1..15 on a 16-phase cell. `_phase_jobs` plans these phases; `_key_job` is unchanged (warm from the u = 0 nominal key, the shifted retime of K7). `_p7_cell` (n_phases > 4) cells always get keys, like 6.1's core cells: they are in 'phase' while V7 holds (R 28.6 and 55.9 besides the core), and kept whatever their braking class when V7 failed. Budget rule 2 (180° only) still applies. `_phase_indices` keeps u_1..u_3 of the 4-phase grid (4, 8, 12 of 16) for validation V7 (`_validate_v7_cell`, 6.5 unchanged) and for P7's null (`_p7_null_phases` = 0, 4, 8, 12; `_key_phase_sd`). **Budget**: each phase cell counts `len(_phase_key_indices(c))` keys, in the item counts and in rule 2's credit (`_rule2_cut_counts`); `projection['phase_keys']` adds `keys`, `p7_cells`, `p7_keys` and `p7_beyond_u1_u3` per item. **Analysis**: `CachedKeys.nominal(cell, k)` already finds the phase key at any k by the problem id. So all 32 deterministic runs of a P7 pair (16 phases × 2 directions) are judged against the key at their own phase (`key_at_phase`, `v_e_star_k`, Δd_b, P2b, and the speed errors under AD23). Their re-anchored keys at A1 follow R17a with 16 phases. AD23 records it. The keys stage's acceptance V7 (reported, never the trigger) now lists every phase key of P7 cells. | `test_R17b_phase_key_indices_cover_every_phase_and_v7_keeps_u1_u3`, `test_R17b_validation_v7_still_solves_u1_u3`, `test_R17b_p7_cells_get_phase_keys_at_every_phase` (jobs, log, budget, rule 2 credit), `test_R17b_p7_null_keeps_phases_0_4_8_12`, `test_R17b_analysis_reads_the_key_at_each_of_the_16_phases` |

**Projection on the real design** (read only: `_budget` on the saved design, manifest and key cache, nothing written; 2026-10-19 after the `keys --priority P2` run; V5 failed, so both directions; V7 failed; validation timings 8.36 CPU-s per warm or anchored key, 34.3 per nominal). "Before R17" is the same cache counted without R17's additions (3 phase keys per P7 cell, one re-anchored key per cell):

| Item | Before R17 | With R17 |
|---|---|---|
| `phase` (P0) | 537 keys, 1.25 CPU-h | 777 keys, 1.81 CPU-h: the 20 P7 cells carry 300 (15 each), 240 more than u_1..u_3 |
| `anchored_G` (P0) | 196 keys (196 cells), 0.46 CPU-h | 1504 keys, 3.49 CPU-h: 136 four-phase (generalist, cell) pairs × 4 + 60 P7 pairs × 16 (960); R12's 36 below-support cells carry 288 |
| `anchored_S` (P1) | 50 (protocol; the stage selects 426 (agent, cell) pairs) | 2424 keys, 5.63 CPU-h: 3 specialists × (122 four-phase cells × 4 + 20 P7 cells × 16); P7 carries 960 |
| `anchored_B` (P1) | 108 (protocol; the stage selects 216) | 864 keys, 2.01 CPU-h: 3 generalists × 72 tier B D = 600 cells × 4 |
| `anchored_ckpt` (P2) | 216 (protocol; the stage selects 480) | 1920 keys, 4.46 CPU-h: 160 (generalist, cell) tail and bulk pairs × 3 checkpoints × 4 phases |
| `nominal_B600` (P1, new) | not budgeted | 72 cold nominal keys, 0.69 CPU-h (solved by the anchored stage; since R17c by the keys stage) |
| `phase_P1`, `phase_P2` | 324, 189 (no P7 cells) | unchanged |
| Total (P0 / P1 / P2) | 8.38 CPU-h (4.48 / 2.29 / 1.61) | 23.89 CPU-h (8.08 / 10.24 / 5.57) |

No budget rule fires (P0 well below 35 CPU-h, total below 60). R17's "about 4×" holds for the 4-phase cells. On P7's cells the re-anchored keys are 16 per cell, so `anchored_G` is 7.7× its u = 0 count: about 3.5 CPU-h, against the ruling's "roughly 1.5". With P1 and P2 counted on the design (review finding 2) the re-anchored keys total 6712 (1504 + 2424 + 864 + 1920), 15.6 CPU-h; the earlier "R17 adds 2670 keys, 6.2 CPU-h" scaled the protocol's u = 0 estimates and was too low. Part of the gap predates R17: counted the same way, the u = 0 selection is 1318 keys (196 + 426 + 216 + 480) against the protocol's 470 (96 + 50 + 108 + 216). (The manifest's stored projection, 8.43 CPU-h, was taken before the P1 and P2 nominal keys were solved: since then `phase_P1` went from 327 to 324 keys and `phase_P2` from 210 to 189 as flat cells were found; that change is not R17's.)

**Cache state.** The `keys --priority P2` run finished before R17 was implemented: the cache holds phase keys at phases 1, 2, 3 (about 330 cells) and at 4, 8 and 12 only on the 20 P7 cells. The next `keys` run resumes from the cache and adds the other 12 phases of each P7 cell (240 keys, `_phase_key_indices`). Re-anchored keys are post-tag (`anchored-keys`), and none was cached before R17, so no cached id changes meaning (the phase in the id is new).

**R17 review (2026-10-19).** Findings 1, 2 and 4 and the `ra_keys` column note are fixed (above; the column doc now says `ra_keys` = n_runs − ra_fallback on a uses_ra pair and 0 elsewhere). The two items below were open for the owner; ruling R17c resolved both (next section):
- *Tier B D = 600 has no phase-matched keys* (review finding 3, confirmed; resolved by R17c). The V7 remedy's sets are `phase` (A1, A2), `phase_P1` (E1–E3) and `phase_P2` (tier C, recovery); tier B is not a nominal-key item at all (its u = 0 nominal key is solved inline by the anchored stage). So on the 72 tier B D = 600 cells every RA key at phases 1–3 places s_L from, starts from and falls back to the u = 0 key. That is R17a's fallback read literally ("the nominal key at that phase", which never exists there), not a code defect. Phase keys for those cells would be about 72 × 3 = 216 warm keys (0.5 CPU-h) at P1 and need a ruling to add tier B to the remedy's sets.
- *Order of stages* (ruled by R17c: the `keys` run comes first). With finding 4's fix a re-anchored key solved before its phase key existed is re-solved once the phase key is cached, so the `keys` run that adds P7's other 12 phases should precede `anchored-keys` to avoid paying twice (correctness does not depend on it).

## Ruling R17c: tier B D = 600 keys in the keys stage (protocol 12)

R17c accepts R17's corrected cost (about 24 CPU-h for all keys; no budget rule fires) and resolves the R17 review's two open items. Tier B D = 600 cells, where the band reads e_ve^RA (`analysis` `uses_ra`), get their nominal keys in the keys stage at P1 and, because V7 failed, phase-matched keys at their phases, so R17a's per-phase re-anchored keys there start from, and fall back to, the key at their own phase. The keys run (with P7's other 12 phases) comes before `anchored-keys`.

| # | Ruling | Implemented by | Tests |
|---|---|---|---|
| R17c | Tier B D = 600 cells get nominal and phase-matched keys in the keys stage (P1), about 72 cold nominal and 216 warm phase keys; the keys run precedes `anchored-keys` | **Cells**: `_b600_cell(c)` (approach 'B', D = 600; tier B's 310 km/h / 600 m cells are the A1 cells, so these are 160, 220 and 280 km/h × 6 R × 2 θ × 2 directions = 72); `anchored_cells` uses the same test. **Key sets** (`_key_sets`): new nominal item `nominal_B600` (one direction with the mirror shortcut, both without it; V5 failed, so 72), whatever V7 says; with V7 failed, new phase item `phase_B600` (`PHASE_ITEMS` = phase, phase_P1, phase_B600, phase_P2), so `_phase_split` keeps the braking cells only (2.3, from the cached nominal key), `_phase_key_indices` gives u_1..u_3 (tier B cells have 4 phases), and budget rule 2 (180° only) applies as to every phase item. While V7 holds `phase_B600` is empty, as `phase_P1` and `phase_P2` are. **Jobs** (`stage_keys`): P1's pool adds tier B D = 600's nominal keys as protocol 6.2's descending-R series (`_series_jobs`: 12 series of 6 radii per (θ, direction, v0)), cold (`full_recipe` with start (2) from the larger-R neighbour; no warm start from another key kind). After the P1 pool and tier D's doses, the phase cycles are refreshed from the new nominal keys (D26: tier B D = 600's P now comes from its nominal key, as every other cell's does; since the R17c review, finding 6, whatever V7 says, and likewise after P2's pool) and, with V7 failed, `phase_P1` and `phase_B600` are planned and solved in one pool (`_phase_after` takes several items). Tier B D = 600 gets no job at P0 or P2. `_key_job` is unchanged. **Budget** (`answer_key.BUDGET_ITEMS`): `nominal_B600` ('nominal', P1) is a keys-stage item counted on `_key_sets` (it was a tuple item counted by the anchored stage's selection); `phase_B600` ('warm', P1) counts 3 keys per braking or not-yet-solved cell, only while V7 failed; `projection['phase_keys']` reports it like the other phase items. **Anchored stage**: solving a missing nominal key is only a fallback, now with Q's tuned c_v (`job['q_cv']`, I19), marked `nominal_inline`; since the R17c review (finding 5) the stage solves each missing one once, in a pool of 'nominal' jobs before the re-anchored and viability jobs, and the worker's inline solve remains a last resort. `_phase_key_expected(bank, c, nom, v7_failed, rule2)` is the keys stage's rule for phase-matched keys on a re-anchored cell (core and P7 cells always; with V7 failed every cell whose nominal key brakes by 2.3; budget rule 2 keeps 180° cells only). `_anchored_preflight(bank, jobs, cache, v7_failed, cuts)` runs in the main process before the pools and lists the job cells without a cached nominal key and the (cell, phase) pairs without a verified phase-matched key where that rule expects one, with tier B D = 600's share, and apart (`viability_nominal_missing`) the viability jobs' other cells without a nominal key. It only looks up. The worker reports `u0_base_phases`: the phases k > 0 anchored from the u = 0 key where the same rule expects a phase-matched key (the job carries `v7_failed` and `rule2`; review finding 1), so after a complete keys run it is 0, like the pre-flight. A missing key is logged, not fatal: the anchored keys are post-tag and the keys stage refuses after the tag unless run as a logged deviation (`--post-tag-deviation`), so the jobs fall back (the nominal solve above; R17a's u = 0 fallback, re-solved by finding 4's rule once the phase key exists). The status in `manifest.validation['anchored_keys'][priority]` records the pre-flight counts, `nominal_inline` (re-anchored cells' ids), `viability_nominal_inline` (viability-only cells' ids: not an R17c fallback) and `u0_base_phases` (count). | `test_interp_r17.py`: `test_R17c_tier_b_d600_cells_are_a_p1_nominal_and_phase_item` (sets in both mirror modes, never D = 250 or recovery, the same cells as `anchored_cells`), `test_R17c_budget_counts_b600_nominal_and_phase_keys` (items, priorities, timings, braking filter, upper bound without a cache, none while V7 holds, rule 2's credit), `test_R17c_keys_stage_solves_b600_nominal_then_phase_keys_at_p1` (the stage with its pool stubbed: series in P1's pool, phase keys after them, flat cell skipped, nothing at P0 or P2, nominal keys but no phase keys while V7 holds), `test_R17c_anchored_job_finds_the_keys_cached_and_solves_inline_only_as_fallback`, `test_R17c_anchored_preflight_reports_missing_nominal_and_phase_keys` (a real `KeyCache` in a tmp dir), `test_R17c_anchored_stage_reports_the_preflight_and_fallbacks`; review fixes `test_R17c_review_u0_based_phases_are_the_preflights_missing_phase_keys`, `test_R17c_review_missing_nominal_keys_are_solved_once_and_viability_apart`, `test_R17c_review_p1_phase_cycles_follow_the_nominal_keys_whatever_v7_says`; updated `test_interp_v7_remedy.py` (cell 13, tier B D = 600, is now a P1 nominal and phase cell: the sets, the budget's `phase_B600` and rule 2 credit, and the stage's P1 phase pool) and `test_interp_r17.py` (`nominal_inline` and `u0_base_phases` in the fallback test; the stage test stubs the pre-flight) |

**Decision.** The ruling's "cold" is read as the budget's timing item (a nominal key, no warm start from another key of the cell), and the keys as protocol 6.2's recipe for agent-free keys, whose start (2) is the neighbouring larger-R key of the same series ("each series is solved in descending R"), as for tier A, E3 and tier C. The anchored stage's fallback solves a single cell without start (2). Key ids hold the problem, not the starts, so either path caches the same key.

**Projection on the real design** (read only: `_budget` on the saved design, manifest and key cache, nothing written; 2026-10-19; V5 failed, so both directions; V7 failed; validation timings 34.3 CPU-s per nominal key, 8.36 per warm key):

| Item | Before R17c | With R17c |
|---|---|---|
| `nominal_B600` (P1) | 72 cold nominal keys, 0.69 CPU-h (anchored stage, inline) | 72 keys, 0.69 CPU-h (keys stage, 12 series) |
| `phase_B600` (P1, new) | none | 216 warm keys, 0.50 CPU-h: 72 candidates × 3, an upper bound (no tier B D = 600 nominal key is cached, so every cell counts as braking until its key exists) |
| Total (P0 / P1 / P2) | 23.89 CPU-h (8.08 / 10.24 / 5.57) | 24.39 CPU-h (8.08 / 10.75 / 5.57) |

No budget rule fires (P0 8.08 against 35 CPU-h, total 24.39 against 60). All 72 tier B D = 600 cells have W = 800, so their nominal keys may cost more than the validation mean the projection uses.

**Cache state and order of stages.** None of the 72 tier B D = 600 cells has a nominal or phase key cached, and 240 of P7's phase keys (the 12 phases besides 4, 8 and 12 on each of 20 cells) are missing. The next `keys --priority P2` run, before the tag, resumes from the cache and solves both. The design is saved again when the phase-cycle refresh changes tier B D = 600's P (from the s-OCP's to its nominal key's v_sight), so the design hash changes, as after every keys run. `anchored-keys` (post-tag) comes after it; its pre-flight then should report nothing missing.

**R17c review (2026-10-19).** No finding at medium severity or above; six low or informational ones, all confirmed. Five are fixed; finding 4 is an intended side effect, recorded here:

| # | Finding | Verdict and evidence | Fix | Test |
|---|---|---|---|---|
| 1 | `u0_base_phases` counted phases that are not fallbacks | Confirmed. The worker counted any k > 0 anchored from a u = 0 key with a sustained brake; the pre-flight expects a phase key only by the keys stage's rule. Reproduced: has_brake with v_min* 85.9 at v0 86 (not braking by 2.3) gave a pre-flight `phase_missing` of [] and `u0_base_phases` [1, 2, 3]; likewise every non-core A1 cell while V7 holds and rule 2's 90° cells | `_phase_key_expected` is the one rule for both; the stage hands each job `v7_failed` and `rule2` | `test_R17c_review_u0_based_phases_are_the_preflights_missing_phase_keys` (5 cases, a real `KeyCache`: the worker's count equals the pre-flight's) |
| 2 | Viability solves inflated `nominal_inline` | Confirmed. `viability_targets` drops only EXCL_ALL and no_key, so a crashed run on a cell with key_status 'missing' (tier B D = 250: no keys-stage item) solved its nominal key inline and was logged as an R17c fallback, while the pre-flight ignores viability jobs | `nominal_inline` lists re-anchored cells only; viability-only cells go to `viability_nominal_inline` and the pre-flight's `viability_nominal_missing`, logged apart as "not an R17c fallback" | `test_R17c_review_missing_nominal_keys_are_solved_once_and_viability_apart` |
| 3 | The pre-flight's advice could not be followed | Confirmed, with a nuance: it said "`keys --priority P1`, before the tag" from a post-tag stage. The keys stage does run after the tag, but only with `--post-tag-deviation REASON` (`_refuse_after_tag`) | The message says the keys stage should have solved them before the tag (P1 or P2), names the deviation route, and states the fallbacks | `test_R17c_anchored_stage_reports_the_preflight_and_fallbacks` (the logged text) |
| 4 | 6.3's "beaten" check now covers the 72 tier B D = 600 cells | Confirmed; intended. Protocol 6.3 applies to every cell with a key and an agent's u = 0 return. Before R17c these cells' nominal keys were solved inline after 6.3, so the first anchored run skipped them and a later run (e.g. P2 after P1) checked them after their re-anchored keys were solved; since the base check compares key ids, which a 6.3 re-solve keeps, those keys would not have been re-solved. Now 6.3 sees the keys first, as for A1. 6.3 re-solves are budgeted for no cell (A1 included); phase keys warm-started from a nominal key that 6.3 later replaces stay valid (their id holds the problem, not the start), as for A1 | None (recorded) | None |
| 5 | Fallback concurrency | Confirmed. At P1 a tier B D = 600 cell has a re-anchored job from each of up to 6 finals, so a missing nominal key could be cold-solved by up to 6 workers at once | The stage solves each missing nominal key (re-anchored and viability cells) once, in a pool before the jobs | `test_R17c_review_missing_nominal_keys_are_solved_once_and_viability_apart` (G1 and S1 both re-anchor cell 1; one 'nominal' job) |
| 6 | P refresh after P1 only when V7 failed | Confirmed. `_refresh_phase_cycles` ran only inside `_phase_after`, called only with V7 failed, so while V7 holds the P1 and P2 cells' P waited for a later keys run's P0 refresh, and a last keys run before the tag would leave it stale (moot today: V7 failed) | `_phase_after` refreshes whatever V7 says and runs the P1 / P2 phase pools only with V7 failed | `test_R17c_review_p1_phase_cycles_follow_the_nominal_keys_whatever_v7_says` |

Each fix was checked by mutation on a scratch copy of the script: undoing it makes its test fail (finding 3's by its logged text). The fixes change no budget item, key id or key-stage job while V7 failed (the manifest's state), so the projection (24.39 CPU-h) and the cache state above stand. With V7 holding, a `keys` run now saves the design after P1 (and P2) when their nominal keys move a P.

## Ruling R18: seed replicates of the generalists (protocol 12)

R18 adds seeds 1 and 2 of each generalist variant as prospective agents: G1s1, G1s2 (v1, G1's grammar), G2s1, G2s2 (v2), G3s1, G3s2 (v3). They were trained by `scripts/gpu_queue_replicates.sh` with the seed-0 runs' arguments; only the seed differs. Primary decisions stay on G1–G3 as pre-registered. A variant-level claim needs the same decision in at least 2 of the variant's 3 seeds; otherwise it is seed-specific. P1c orders the variants by each variant's median R_dep, P8 stays on the seed-0 crash units, the replicates run the finals and the last three checkpoints, and P6's checkpoint history stays on seed 0. Seeds 3 and 4 of the same queue are not agents of this study. Nothing below rolls out an agent: the replicates are listed, loaded for a forward pass and read for obs_rms only.

| # | Requirement | Implemented by | Tests |
|---|---|---|---|
| 1 | Registry | New module `registry` (no interp import; `agents` re-exports `AGENTS`, `AgentSpec` and the helpers, `support` re-exports `GrammarSpec` and `GRAMMARS`). `AgentSpec` gains `variant` ('v1'/'v2'/'v3', '' for specialists), `seed` (G1–G3 0; S1, S2, S3 1, 2, 3 as their config.json records; replicates 1, 2) and `hardware` (R18: "recorded, not corrected": G1–G3 "Mac CPU", replicates "GPU (NVIDIA L40S, …)", '' where the ruling does not say). The six replicates are `prospective_after` 0, not pilot-informed, finished when eval_rdev.json exists, final = last ckpt. Helpers: `generalists()`, `seed0_generalists()`, `replicates(variant)`, `specialists()`, `variants()`, `variant_seeds(v)`, `variant_of`, `seed_of`, `seed0_of`, `is_generalist`, `is_specialist`, `is_replicate`, `agent_grammar()`, `runs_cell`. Every hard-coded ("G1", "G2", "G3") or ("S1", "S2", "S3") list is gone: `analysis.GENERALISTS` / `SPECIALISTS`, `support.AGENT_GRAMMAR` (now derived: replicates included), `stimuli.RUNNERS_FROM_S1` and the two tier D specialist loops and `real_state_requests`, `rollout.check_context` (specialists by kind), and in the script `_r12_extra_anchored`, `_anchored_g_count`, `_anchored_p12_count`, `stage_anchored_keys`, `_knn_buffer`, `confirmatory_problems`, `stage_analyse` (P6, P8 ladder, twins) and the pilot (`rollout.PILOT_EXPECTED`). Values for the existing agents are unchanged | `test_interp_r18.py`: `test_R18_registry_lists_the_replicates_with_their_variant_and_seed`, `test_R18_replicate_runs_differ_from_seed0_only_in_the_seed` (reads the runs' config.json: ppo and env equal, args differ only in seed, name and options the seed-0 runs left unrecorded at the grammar's values), `test_R18_no_hard_coded_agent_lists` (source scan); updated `test_interp_skeleton.py::test_agent_registry_matches_the_protocol` |
| 2 | Scope | Tiers A, B, C, E3, G and E1/E2 (cells with no runners, built from the same real states) are run by every agent. Tier D's S1-state cells list S1 and G1–G3 as runners in the saved design. `registry.runs_cell` lets a replicate run every cell its seed-0 agent runs (`rollout.plan_runs`), so the design and its hash are unchanged. Probes run per final, replicates included. Keys: the agent-free keys are shared. `stage_anchored_keys` gives the replicates' finals G1–G3's P0 keys (A1 tail, bulk and R12's non-clipped below-support cells, from each replicate's own regimes and run-time clipping) and P1 keys (tier B D = 600), and their own viability solves; P2 (P6c's checkpoints) and `reanchor_immaterial` stay on G1–G3. The stage refuses when a replicate final has no frozen regimes with R_clip, because `anchored_cells` would treat every A1 cell as bulk. Regimes: `support.AGENT_GRAMMAR` includes the replicates, so the support stage (which rebuilds when a final lacks R_clip) gives each one its variant's q1 / q10 and R_clip from its own obs_rms. Rollouts: `_snapshot_kinds` gives a replicate its final and the two before it, even under `--snapshots all`. P8: crash units and twins stay G1–G3's (`stage_analyse`; `_real_laps` never runs a replicate) | `test_R18_replicates_run_their_seed0_agents_cells`, `test_R18_replicates_roll_out_their_finals_and_last_three_only`, `test_R18_replicate_regimes_are_their_own_or_the_seed0_stand_in`, `test_R18_anchored_keys_stage_gives_replicates_the_p0_and_p1_keys_but_never_p2` (the stage at P2 with its pool stubbed: P0 and P1 jobs for G1s1, P2 only for G1's checkpoints, viability selected per family), `test_R18_anchored_keys_stage_needs_the_replicates_own_regimes`; updated `test_interp_support.py::test_build_and_load_support` (a replicate's q1 / q10 equal its seed-0 agent's; its R_clip is its own) |
| 3 | Analysis | `family_of`: a replicate's decisions are family 'replication' (P5 'control', exploratory ones 'exploratory'). `decisions` runs P1a, P1-tail, P1b, P2 and P3 per replicate through `_generalist_decisions`, the seed-0 code path (gate, 9.4 variants, 2-of-last-3 stability on the replicate's own history), in the exploratory set only. The confirmatory table's rows are unchanged. `replication_tables` / `decision_table` write the replication family apart: `replication_decisions.npz` (the replicates' rows), `replication_variants.npz` (`VARIANT_COLUMNS`, per (variant, prediction) `variant_verdict`: the verdict shared by ≥ 2 of the 3 seeds, else the decision shared by ≥ 2 (`decision_class`: 'cautious but lawful' is a P1a refutation, protocol 2.4); 'seed-specific' when all three decided and no 2 agree; 'not testable' when a seed has no decision and no 2 agree; `claim` when the shared decision is confirm, refute or cautious but lawful), `replication_spread.npz` (`SPREAD_COLUMNS`: each seed's five verdicts, stability, `key_numbers` and R_dep at A1 and A2) and `replication.json`. Under `--confirmatory` these are written once, like decisions.npz (`replication_all_*` otherwise). `p1c`: each variant's R_dep is the median over its 3 seeds (+∞ counts), defined only when the known seeds fix it (`_seed_median`: all 3 known, or 2 known and equal; a seed absent from the inputs or with an undefined R_dep is unknown), else P1c is 'not testable'; the exact null permutes the 3 medians. With no replicate in the inputs (seed 0 alone) this is the pre-registered test; the row keeps the label 'G1,G2,G3' and adds `r_dep_A1_by_seed`, `n_seeds` (known seeds), `r_dep_A1_seed0`, `undetermined` and `rule`. Unchanged and on G1–G3: P1c-approach, P4's memorisation index (the mean over G finals), P6, P8. `power_table` appends replicate rows after the other agents, so each agent's simulation streams and rows stay the same | `test_R18_replicates_form_the_replication_family`, `test_R18_variant_verdict_needs_two_of_three_seeds`, `test_R18_variant_table_and_seed_spread`, `test_R18_p1c_orders_variants_by_their_median_r_dep`; `test_interp_analysis.py::test_R18_decision_table_writes_the_replication_family` (confirmatory rows identical with and without replicates; the family's files; the one-run rule; the exploratory set and P1c on medians), `::test_R18_power_table_keeps_the_other_agents_rows` (checked by mutation: the old sorted order fails it) |
| 4 | Budget | `answer_key.BUDGET_ITEMS` gains `anchored_R` (P0, 'anchored'), `viability_R` (P0, 9.1's cap of 200 applied to the replicates apart, so G1–G3's seeded subset is unchanged) and `anchored_B_R` (P1). There is no P2 item, because P6 stays on seed 0. `_anchored_rep_count` counts the items on the design, one key per deterministic phase (R17a). It uses each replicate's own frozen regimes or, until the support stage has frozen them, its seed-0 agent's as a stand-in (same grammar, so same q1 / q10; only R_clip, and so R12's below-support edge, can differ). Before any regimes exist it uses the protocol's 96 cells per 3 generalists. `proj['replicates']` lists the items and the totals without them. `_rollout_projection` (`proj['rollouts']`) counts protocol 4's runs and steps per agent, snapshot kind and mode, with each run W steps and tier D by `runs_cell`, at the measured steps/s; budget rule 3 stays a throughput rule | `test_R18_budget_counts_the_replicates_keys_and_rollouts`, `test_R18_budget_before_the_regimes_are_frozen`; `test_interp_r17.py`'s keys-stage stub also stubs `_anchored_rep_count` (at the protocol's default timings the replicate fallback count would fire rule 1 on its toy design) |
| 5 | Manifest | `freeze_manifest` lists a replicate once eval_rdev.json exists. Every snapshot is prospective (`prospective_after` 0) and `evaluated_on_stimuli` False. Its agent record carries variant, seed, hardware, `replicate_of` and `ruling: R18`. The manifest stage logs each replicate as "R18 replicate of G<n>" plus a summary line (finished, snapshots, prospective, evaluated). `confirmatory_problems` (step 8's gate) waits for every registry agent's final with det, nat and mat groups, the replicates included, and (review finding 1) for the rollouts of the manifest's two snapshots before every generalist final (`_history_entries`; det and nat, det alone once budget rule 3 cut the checkpoint stochastic runs) | `test_R18_manifest_lists_the_replicates_prospective_and_not_evaluated` (tmp runs, one finished and one pending), `test_R18_manifest_stage_names_the_replicates`; updated `test_interp_script.py::test_confirmatory_analysis_needs_complete_inputs` |

**Decisions.**
- **The registry is its own module.** `support`, `stimuli`, `rollout` and `analysis` must name agents, and `agents` imports `stimuli` and `support`, so the registry sits below them all. `agents.AGENTS` is the same object.
- **Tier D by `runs_cell`, not new runners.** Adding the replicates to `RUNNERS_FROM_S1` would change the saved design and its hash (a stimuli-stage rebuild). Mapping a replicate to its seed-0 agent's runners gives the same runs.
- **P1 anchored keys for the replicates.** R18 says "the same tiers, keys and anchored keys as their seed-0 agent". G1–G3 get tier B D = 600 re-anchored keys at P1, so the replicates do too. No replication-family decision reads tier B D = 600 (P1a–P2 read tiers A, P3 reads no re-anchored key), so dropping them would cost only descriptive curves and save about 4 CPU-h.
- **Viability apart.** 9.1 caps the solves at 200 "in all", and beyond the cap a seeded subset of 10 crashed pairs per agent is kept. Pooling the replicates with G1–G3 would change G1–G3's subset, so each family gets the cap and the seeded rule (`viability_R`).
- **6.3 over every final.** `_beaten_jobs` checks every final's u = 0 return, the replicates included: a replicate that beats a key gets that key re-solved and a deviation logged, as for any agent. Keys are shared, so such a re-solve (or the replicates' extra warm starts when another agent beats a key) reaches G1–G3's key-based errors too; `resolve_beaten_key` replaces a key only with a strictly better verified one. Each job, its deviation (`r18`: `beaten_by_replicates_only`, `replicate_starts`) and the stage's summary (`beaten_by_replicates_only`) record the replicates' part (review finding 2).
- **The seed-spread R_dep is the final's.** P1c reads finals only.

**Projection on the real design** (read only: `_budget` on the saved design, manifest and key cache, nothing written; 2026-10-19; V5 failed, so both directions; V7 failed; validation timings: 34.35 CPU-s per nominal key, 8.36 per warm, anchored or viability key). The support stage has not yet frozen the replicates' regimes, so each is counted with its seed-0 agent's:

| Item | Before R18 | With R18 |
|---|---|---|
| `anchored_R` (P0, new) | none | 3008 keys, 6.99 CPU-h: 392 (replicate, cell) pairs, 2 × G1–G3's 196. P7 cells carry 1920 keys, R12's below-support cells 576 |
| `viability_R` (P0, new) | none | 200 solves (cap), 0.46 CPU-h |
| `anchored_B_R` (P1, new) | none | 1728 keys, 4.01 CPU-h: 6 × 72 tier B D = 600 cells × 4 phases |
| G1–G3's and the specialists' items | as in R17c | unchanged |
| Total (P0 / P1 / P2) | 24.39 CPU-h (8.08 / 10.75 / 5.57) | 35.86 CPU-h (15.53 / 14.76 / 5.57); about 7.2 h wall on 5 workers |

No budget rule fires: P0 is 15.53 against 35 CPU-h, and the total is 35.86 against 60. At the protocol's default timings (before validation), the same items would add about 21 CPU-h, and rule 1 could fire there. The manifest holds the validation timings.

**Agent rollouts** (protocol 4's replication table on the saved design, every snapshot `--snapshots all` lists): 370.2M steps against 222.0M before R18. The six replicates add 148.2M: 24.7M each, their final and the two checkpoints before it; tier D's S1-state cells are included. At the measured 196k (deterministic) / 205k (stochastic) steps/s per process, that is 0.51 process-hours in all (0.20 for the replicates), about 10 minutes wall on 3 processes. Budget rule 3 (below 15k steps/s) does not fire.

**Order of stages before the tag.** The replicates are not in the manifest or support.json yet. `manifest` lists them, which loads each snapshot and runs one forward pass. `support` then rebuilds the corner tables (they must reproduce the frozen ones) and adds the replicates' regimes and R_clip. `dryrun`'s power table gains their rows. The design and key cache do not change. After the tag, `rollouts` covers them by default, `anchored-keys` refuses until their regimes are frozen, and `analyse --confirmatory` waits for their finals.

**Not implemented as written, and open items.**
- *Pre-existing; affects G1–G3 and the replicates alike.* Protocol 4's "last 3 checkpoints" rows run tiers A and B only. Protocol 8's gate needs the tier G width-12 straights, so `checkpoint_gate` cannot judge the two history snapshots. AD24 then makes their P1–P3 decisions 'not testable', and `_stabilise` turns every claim at the final into 'unstable' (2-of-last-3 cannot hold). The simulated tests pass the gate through `inputs.gates`, so this never surfaced. Fixing it needs a ruling, for example tier G on the last-3 rollouts (the null straights are cheap) or a history gate on the gentle tier A cells. R18 relies on the same rule for the replicates. Until then the confirmatory analysis refuses to run (review finding 1, below). **Settled by ruling R19a** (section below): the last-3 rollouts run tier G too.
- *P1 anchored keys and viability for the replicates* (Decisions above) go beyond "their own P0 anchored keys"; dropping them is one line each.

### R18 review

Each finding was reproduced before it was fixed. Each fix was then checked by mutation: undoing it on the working copy, one at a time, makes its test fail. No fix changes a confirmatory row of G1–G3, a key, a budget item or the design. The read-only budget projection still gives P0 / P1 / P2 = 15.53 / 14.76 / 5.57, 35.86 CPU-h in all, and no cut.

| # | Finding | Verdict and evidence | Fix | Tests |
|---|---|---|---|---|
| 1 | The stability rule can never hold (High, pre-existing). Missing history rollouts let claims stand unchecked | **Confirmed.** The saved design has no cell that is both in tier G and in tier A or B (27 cells are tier G only). The `last3` rows run tiers A and B, so `checkpoint_gate` on a history table gives `completion_null` NaN and `evaluable` False. The history verdicts become `['not testable', 'not testable']` and a 'confirm' at the final becomes 'unstable'. With no history, `_stabilise` gives 'n/a', and `confirmatory_problems` did not ask for the history rollouts | **Needs a ruling** on what is actually decided: tier G on the last-3 rollouts, a history gate on the gentle tier A cells, or something else. Until then, the one-shot confirmatory run is protected. `confirmatory_problems` now needs the manifest's two snapshots before every generalist final (`_history_entries`) with det and nat groups (det alone once budget rule 3 has cut the checkpoint stochastic runs). `analysis.history_gate_problems` names every generalist final whose history is missing or that the gate cannot judge (`history_judgeable`). `stage_analyse --confirmatory` refuses on it, and exploratory runs log it. Each P1–P3 decision with history records `history_gate_judgeable`. Decisions are unchanged. **Ruling R19a** then added tier G to the last-3 rollouts (section below) | `test_interp_script.py::test_confirmatory_analysis_needs_complete_inputs` (history cases), `::test_history_entries_are_the_two_snapshots_before_the_final`; `test_interp_analysis.py::test_R18_review_history_snapshots_without_tier_g_cannot_pass_the_gate` (since R19a, the incomplete-input case: a history without its tier G runs is 'not testable', the claim at the final 'unstable', and `history_gate_problems` names it; complete inputs: `::test_R19a_history_snapshots_with_tier_g_are_judged_and_claims_stay_stable`) |
| 2 | Replicates can change G1–G3's keys through 6.3 | **Confirmed as behaviour; kept.** Protocol 6.3 says "any agent", and the re-solve is warm-started "from every final agent's u = 0 rollout". R18 makes the replicates agents of the study, and keys are shared, as S1–S3's 6.3 re-solves already reach G1–G3. `resolve_beaten_key` replaces a key only with a strictly better verified solution. More starts can, however, flag a cell multi-basin. Restricting 6.3 to the agents from before R18 would leave a key that a replicate beats in place for everyone | Recorded, not changed. Each 6.3 job and its deviation carry `r18` (`beaten_by_replicates_only`, `replicate_starts`), the summary lists `beaten_by_replicates_only`, and the stage log counts those cells, so any change to G1–G3's keys that the replicates caused is listed. The study owner may still rule otherwise. **Ruling R19b** kept it as is (section below) | `test_interp_r18.py::test_R18_review_beaten_keys_record_the_replicates_part` |
| 3 | P1c edge cases | **Confirmed.** {18.3, +∞, undefined} gave a median of +∞ (`np.median` averages two values). A NaN G3 with G3s1 defined was still tested. Replicates alone gave 'confirm' under the label 'G1,G2,G3' | `_seed_median(values, n_seeds)`: a median of 3 is defined only when the known seeds fix it (all 3 known, or 2 known and equal); a seed absent from the inputs or with an undefined value is unknown; otherwise P1c is 'not testable'. With no replicate in the inputs, it is the pre-registered seed-0 test. Details add `undetermined` and `rule` | `test_interp_r18.py::test_R18_review_p1c_medians_need_the_seeds_that_fix_them`; `::test_R18_p1c_orders_variants_by_their_median_r_dep` updated |
| 4 | Verdicts compared as plain strings | **Confirmed.** {refute, cautious but lawful, confirm} gave 'seed-specific'. Protocol 2.4 makes 'cautious but lawful' the label of a P1a refutation | `decision_class` (`REFUTE_LABELS`): `variant_verdict` first looks for a verdict shared by 2 seeds, then for a decision shared by 2. {refute, cautious but lawful, confirm} now gives 'refute' (2); two 'cautious but lawful' give that label. Not changed: `_stabilise` (AD21, signed off) still compares verdicts exactly, so a final 'cautious but lawful' after a 'refute' counts as unstable. This is open for the study owner. **Ruling R19c** settled it: `_stabilise` compares decision classes too (section below) | `test_interp_r18.py::test_R18_review_cautious_but_lawful_is_a_p1a_refutation_at_the_variant_level` |
| 5 | Write order in the confirmatory run | **Confirmed.** decisions.npz was written before the replication files were serialised and checked. A bad replication table (for example an object column) raised after decisions.npz existed, and the one-run rule then refused a rerun | `decision_table` serialises every output first (`io.table_bytes` and `io.json_bytes`, with `save_table`'s and `write_json`'s checks). It then writes the one-run markers last: replication.json, then decisions.npz. A failure while building leaves no file. An I/O failure between two atomic writes can still leave a marker | `test_interp_analysis.py::test_R18_review_decision_table_serialises_before_writing` |
| 6 | The two stages checked replicate regimes differently | **Confirmed.** `stage_analyse` checked only that the regimes exist, while `stage_anchored_keys` also required R_clip. A regime without R_clip has no 'clip' set, so P1b / P2 read clip-saturated pairs as below support | One rule, `_replicate_regime_problems` (regimes exist and carry a finite R_clip). The anchored-keys stage refuses on it, `stage_analyse --confirmatory` refuses on it, and exploratory runs log it | `test_interp_r18.py::test_R18_review_both_stages_need_the_replicates_regimes_with_r_clip` |
| 7 | Budget rules are shared (information) | **Confirmed; no action.** Re-run read-only on the saved design (validation timings): no cut; P0 15.53 against 35 CPU-h, total 35.86 against 60. The toy design in `test_interp_r17.py` stubs `_anchored_rep_count` because rule 1 would fire at default timings | None | None |
| 8 | `registry.py` and `test_interp_r18.py` are untracked (information) | **Confirmed** (`git status`) | They must be added with `git add` when this is committed | None |

## Ruling R19: the history gate, 6.3 with the replicates, and decision classes (protocol 12)

R19 settles the three items the R18 review left open. Nothing below rolls out an agent, solves a key or writes under `outputs/psychophysics/`. The tests use hand-built designs, a stubbed policy and rollout engine, and tmp dirs.

| # | Ruling | Implemented by | Tests |
|---|---|---|---|
| R19a | The last-3 checkpoint rollouts (the two snapshots before each final of G1–G3 and of R18's replicates) also run tier G: the null straights, 4 deterministic phases. Protocol 8's gate can then judge them | **Plan**: `constants.REPLICATION['last3']['det_tiers']` is ('A', 'B', 'G'). `det_phases` stays 4. `native` stays {'A': 8}, so tier G gets no stochastic run, and there is no matched noise; the other checkpoints ('other_generalist') already ran A and G the same way. `_snapshot_kinds` is unchanged; `_rollout_job` plans the tiers through `rollout_cells` and `plan_runs`. One helper, `_det_phases(c, kind)`, gives a cell's deterministic phases per kind, for the projection and the input check. **Resume**: `_rollout_job` keeps a group on disk only when it holds every planned (cell, phase, sample) run, or more (`_group_runs` reads the index arrays only). The rule is global: it applies to every snapshot kind and mode, finals and other checkpoints included, not only the last-3 rows. A group that lacks planned runs is run again whole (every planned run, not only the missing ones), replaces the one on disk, and is reported as `rerun`, which the stage logs. A last-3 det group made before R19a is therefore re-run in full (tiers A, B and G) by `rollouts --snapshots last3` without `--force`. The job writes whole groups, so on an unchanged design no other group is ever re-run, and a group that holds more than its plan is kept. No agent rollout has run yet, so none exists. **Projection**: `_rollout_runs` uses `_det_phases`. `_rollout_projection` counts R19a's runs per agent (`r19`) and in total (`proj['rollouts']['r19']`: cells, runs, steps, `steps_without`, `process_h`; `_r19_cell`: tier G and not in tier A or B). **Confirmatory inputs**: `confirmatory_problems(..., history_tier_g)` takes, per generalist final and per history snapshot, how many of its planned tier G det runs the det group lacks (`_tier_g_missing`: every tier G cell the agent runs × `_det_phases(c, 'last3')`, read from the det index). A non-zero count is a problem, and so is a snapshot that was not checked. `stage_analyse --confirmatory` reads the design for the tier G cells once a history snapshot has a det group. **Analysis**: `checkpoint_gate`, `history_judgeable` and AD24 are unchanged. Once the history run tables hold tier G runs, the gate judges each history snapshot from its own runs. `history_gate_problems` is then empty on complete inputs, and its message no longer asks for a ruling. AD24's text and the module docstring record R19a | `test_interp_r19.py`: `test_R19a_last3_rows_run_tier_g_at_four_phases`, `test_R19a_rollout_job_runs_tier_g_and_reruns_a_pre_r19_det_group` (also a complete group kept, `--force`, and `_tier_g_missing`), `test_R19a_rollout_projection_counts_tier_g_at_the_history_snapshots` (and the same design without R19a), `test_R19a_confirmatory_stage_needs_the_history_tier_g_runs` (the stage reads each history file); `test_interp_analysis.py::test_R19a_history_snapshots_with_tier_g_are_judged_and_claims_stay_stable` (no gate given for the history snapshots: the gate is evaluable and passed, `history_judgeable` [True, True], no `history_gate_problems` for the agent, P1a 'confirm' with stable 'yes'; every history verdict equals the one with the gate given; tier G straights that do not complete fail the gate); updated `test_interp_script.py::test_confirmatory_analysis_needs_complete_inputs` (tier G cases) and `test_interp_analysis.py::test_R18_review_history_snapshots_without_tier_g_cannot_pass_the_gate` (now the incomplete-input case). Added by the R19 review (below): `test_interp_r19.py::test_R19_review_resume_rule_covers_every_kind`, `::test_R19_review_history_snapshots_get_their_null_runs_so_d_resp_is_defined`, `::test_R19_review_p6_judges_the_last3_checkpoints`; `test_interp_analysis.py::test_R19_review_p2_claim_recurs_at_history_snapshots_judged_from_their_tier_g_runs` |
| R19b | Kept: 6.3 covers every final agent. A re-solve only replaces a key with a strictly better verified one, and every such re-solve is logged, with whether replicates alone triggered it | Nothing changes in code. The R18 review already implemented the log: each 6.3 job and its deviation carry `r18` (`beaten_by_replicates_only`, `replicate_starts`), the stage summary lists `beaten_by_replicates_only`, and the stage log counts those cells | `test_interp_r18.py::test_R18_review_beaten_keys_record_the_replicates_part` (unchanged) |
| R19c | The stability check compares decision classes, as the variant rule does. The verdict label is still reported | `_stabilise` (AD21): a final's decision recurs when `decision_class(verdict)` is among the classes of the two history verdicts. So a final 'cautious but lawful' after a 'refute' (or the reverse) is stable, and the final keeps its own label. A different class gives 'unstable', with `final_verdict` in details, as before. Non-claims are never relabelled. The AD21 text in the `analysis` docstring and `decision_class`'s docstring record it | `test_interp_analysis.py::test_R19c_stability_compares_decision_classes` (through `_generalist_decisions` with a history the gate judges from its own tier G runs, and the rule on its own) |

**Consequences beyond the gate (R19a).**
- The history snapshots now have their own null runs (`analysis._null_runs` takes them from the det group). Their catch-trial onsets d_resp and Δd_resp, corner regret and Δt_corner are therefore defined, as at the finals. Before R19a these were NaN at a history snapshot, so P2(a) (Δd_resp < 0 in ≥ 75% of O) could not hold there, and a P2 'confirm' at a final could not recur even with the gate given. Tested since the R19 review: `test_interp_r19.py::test_R19_review_history_snapshots_get_their_null_runs_so_d_resp_is_defined` (d_resp, Δd_resp, the pair Δd_resp and corner regret at a history snapshot planned by the last-3 row; all NaN with the row before R19a) and `test_interp_analysis.py::test_R19_review_p2_claim_recurs_at_history_snapshots_judged_from_their_tier_g_runs` (a P2 'confirm' recurs and is stable).
- `run_table` checks each run's s0 and W against its cell. Tier G cells are in the saved design, so the extra runs need no design change, and the design hash is unchanged.
- P6 (seed 0) is affected too. The stage gates every manifest snapshot of G1–G3 with `checkpoint_gate`, and P6 counts a checkpoint that fails the gate as not competent (AD18). Before R19a the two snapshots before each final had no tier G runs, so they failed the gate as not evaluable. Every R level would then have looked not competent there, which pushes each t_R to the final or to +∞. With R19a they are judged like the other checkpoints, which run tiers A and G. Tested since the R19 review: `test_interp_r19.py::test_R19_review_p6_judges_the_last3_checkpoints` (checkpoint kinds from `_snapshot_kinds`, pairs as each kind's row plans them: P6a 'confirm' with each level's own t_R; with the row before R19a, checkpoints 5 and 6 are not evaluable, every t_R is the final and P6a is 'not testable').
- Specialists have no last-3 rows, so nothing changes for S1–S3.

**Rollout projection** (protocol 4's replication table on the saved design, every snapshot `--snapshots all` lists, read-only with `_rollout_projection` and `_budget`, nothing written; 2026-10-19; measured 196k deterministic and 205k stochastic steps/s per process, 3 processes). The design has 27 tier G cells, all tier G only, 4 phases each, every agent a runner. Sixteen have W = 400 and eleven W = 800.

| | Before R19 | With R19a |
|---|---|---|
| Deterministic runs per generalist | G1–G3: 16,576; each replicate: 7,476 | +216 each (2 snapshots × 27 cells × 4 phases): 16,792 and 7,692 |
| Steps per generalist | G1–G3: 40.53M; each replicate: 24.70M | +121,600 each: 40.65M and 24.82M |
| R19a's runs (`r19`) | none | 1,944 deterministic runs (9 generalists × 216), 1.09M steps, 0.0015 process-h (about 6 process-s) |
| All agent rollouts | 370.20M steps, 0.508 process-h, about 10.2 min wall | 371.29M steps (+0.30%), 0.510 process-h, about 10.2 min wall |
| Of which the replicates | 148.20M steps, 0.203 process-h | 148.92M steps, 0.204 process-h |
| Specialists | unchanged | unchanged |

Key items are unchanged. The read-only `_budget` gives P0 / P1 / P2 = 15.53 / 14.76 / 5.57, 35.86 CPU-h in all, and no budget rule fires. Budget rule 3 is a throughput rule and is not affected.

**Checks.** Each fix was checked by mutation: undoing it on the working copy, one at a time, makes its test fail. The six mutations were: last3 without tier G; `_stabilise` comparing labels exactly; resume by group name only; no tier G check in `confirmatory_problems`; the analysis stage passing no tier G counts; and the projection not counting R19a's runs. All interp tests pass in two halves (204 and 147 tests, about 132 and 100 CPU-s), and `ruff check` is clean on the harness. `tests/test_interp_r19.py` is new and must be added with `git add` when this is committed.

### R19 review

The review found no correctness defect in the R19 diff. It raised three low-severity points and one note. Each was checked on the working copy before anything changed. No code path changed: one docstring and this document were edited, and four regression tests were added. Nothing was rolled out or solved, and nothing was written under `outputs/psychophysics/` (its newest files date from before R19).

| # | Finding | Verdict and evidence | Fix | Tests |
|---|---|---|---|---|
| 1 | Test gap: the "same verdicts with or without the gate given" loop never tests P2, and R19a's side effects on P2 (d_resp and Δd_resp at history snapshots) and P6 (last-3 checkpoints judged by the gate) are documented but untested | **Confirmed.** In the `sims` fixture P2 is 'not applicable' for G1 at the final and at both history snapshots, so the loop compared 'not applicable' with 'not applicable'. The test builds run tables directly, so `_null_runs` (phase-0, unmirrored tier G det runs) is never reached. A probe also showed that the late-onset simulation (P2 'confirm') fails the gate at a history snapshot from its own runs: it crashes the gentle A1 cells (R 87.3) | Tests only; no code change. (a) P2, gate side: a G2 history from the late-onset runs, with the gentle cells completed (P2 reads only the A1 tail and below-support pairs) and R19a's tier G runs, and no gate given. P2 'confirm' recurs at both snapshots and is stable. The same history without tier G runs is 'not testable', so P2 is 'unstable'. With the gate given instead, every G2 verdict is the same. (b) P2, run side: a history snapshot's det group planned from `REPLICATION['last3']` (`rollout_cells` and `plan_runs`) on a real corner strip and its null straight, with scripted drives. `_null_runs` finds the null run, and d_resp, Δd_resp, the pair Δd_resp and corner regret are finite. With the row before R19a (tiers A and B) they are all NaN. (c) P6: checkpoint kinds come from `_snapshot_kinds`, and each checkpoint's pairs are what its row plans. The stage's gates pass everywhere, and P6a is 'confirm' with each level's own t_R. With the row before R19a, checkpoints 5 and 6 are not evaluable, every t_R is the final, and P6a is 'not testable' | `test_interp_analysis.py::test_R19_review_p2_claim_recurs_at_history_snapshots_judged_from_their_tier_g_runs`; `test_interp_r19.py::test_R19_review_history_snapshots_get_their_null_runs_so_d_resp_is_defined`, `::test_R19_review_p6_judges_the_last3_checkpoints` |
| 2 | Wording: the new resume check applies to every kind, not only 'last3', and an incomplete pre-R19a det group is re-run in full, not just its tier G runs | **Confirmed as wording; behaviour kept.** `_rollout_job` checks every mode of every kind against `_group_runs`, and a short group is re-planned and replaced whole. This is harmless. `run_rollouts` returns every planned run, so a group the job wrote is never short. A group holding more than its plan (for example a final's groups read against an other-checkpoint row) is kept. Budget rule 3's flag comes from the manifest's pre-tag throughput, so plans do not change between invocations. No agent rollout exists yet | `_rollout_job`'s docstring and the R19a row above now state the global scope, that a short group is re-run whole and replaces the one on disk, and that a superset is kept. The stage log line already said "lacked planned runs (e.g. …)" | `test_interp_r19.py::test_R19_review_resume_rule_covers_every_kind`: a final's complete groups are kept; a final nat group one run short is re-run whole (112 runs) while det and mat are kept; the final's groups are kept for an other-checkpoint row |
| 3 | Stale doc line: R18 review row 1 still says its test "pins the current 'unstable' outcome until the ruling" | **Confirmed** (row 1 of the R18 review table) | That row's Tests column now says that since R19a the test covers the incomplete-input case, and it names the complete-input test | None (documentation) |
| Note | R19a says tier G gets stochastic runs "as for the other history tiers if any", but it gets none | **Refuted as a defect.** The ruling itself says "4 deterministic phases". The last-3 row's native noise is {'A': 8}, which covers tier A only, and the gate reads deterministic completion only. 'other_generalist' already treats tier G the same way | None | `test_interp_r19.py::test_R19a_last3_rows_run_tier_g_at_four_phases` (unchanged: no nat or mat runs on tier G) |

**Checks.** Each new test fails under a mutation of the code it guards, applied to the working copy one at a time and then undone (each file compared with its backup afterwards):
- the last-3 row without tier G fails the null-run and P6 tests;
- `_null_runs` reading a phase other than 0 fails the null-run test;
- the resume check limited to 'last3' (a group name alone for the other kinds) fails the resume-scope test;
- a gate that ignores the tier G straights fails the P2 test.

All interp tests pass in two halves:
- `analysis`, `answer_key`, `controls`, `measures`, `r17`, `r18`, `r19`: 212 tests, about 130 CPU-s;
- `review`, `rollout`, `rulings`, `script`, `skeleton`, `stimuli`, `support`, `v3_validation`, `v7_remedy`: 143 tests, about 95 CPU-s.

That is 355 in all, the earlier 351 plus these four. `ruff check` is clean on `src/apexzero/interp`, `scripts/psychophysics.py` and the interp tests. `tests/test_interp_r19.py` is still untracked and must be added with `git add` when this is committed.

## Dry-run rulings R20-R28 (protocol 12, 2026-10-20)

The step-5 dry run blocked the tag: six of seven controls missed their required verdicts, and the lawful simulated agent missed P3. Rulings R20-R28 were written before any rerun. They were implemented in three parallel parts (`analysis.py`; `controls.py` and `rollout.py`; `stimuli.py`), then wired in `scripts/psychophysics.py` with the cross-file fixes listed below.

Nothing here ran a pipeline stage, rolled out an agent or solved a key, and nothing was written under `outputs/psychophysics/`. The real-design numbers come from read-only checks in memory: cached keys, the saved design and saved control rollouts. Every calibration comes from keys, Q's offline profile or simulated data, never from a control's own P-verdict.

### Implementation map

| # | Ruling | Implemented by | Tests |
|---|---|---|---|
| R20 | T64 (float64, blocking) and T32 (float32: its measure check blocks, crashes reported); regret over \|J*\| | **`analysis`**: `tape_regret` divides the regret by the key table's \|J*\| (the old code at `psychophysics.py:2478` divided by the run's J). `tape_check(runs, keys, 'f64' \| 'f32')`: f64 blocks on regret/\|J*\| ≤ 1e-6 (`TAPE_REGRET_RTOL` = `REPLAY_RTOL`), every error within τ and no termination; f32 blocks only on errors within τ, judged on the runs that reach s_c1 + 250 m (`tape_errors(reached_only=True)`), with crashes listed. **`rollout`**: `run_tape_rollouts_x64` is the only exemption from the x64 assertion. It runs the tape policy only, needs x64 on, and goes through the same chunk loop (`_run_chunks`), `make_batch`, compiled `_episode` and `trace_arrays(float_dtype)`. x0, prev_action, accel, ctx and the track arrays are exact upcasts (`_upcast`, `float32_bank_arrays`). **`controls`**: `CONTROLS['T']` is T32 (required errors; reported termination and regret); `T64` is its own spec; `T64_REGRET_TOL = REPLAY_RTOL`. **Script**: `_tape_keys` builds T32's float32 tapes and T64's uncast float64 tapes from the same keys. `_t64` runs `_t64_job` in a one-worker spawned pool whose initializer switches x64 on before anything is traced. The job checks that the worker's design hash and tape key ids equal the stage's, then writes `rollouts/control-T64/0.npz`. `_control_verdicts` judges T with `tape_check(…, 'f32')` and lists each crashed cell with its key's margins (`_key_margins`: corridor to the termination edge, overall and after s_c1; slip to α_max). It judges T64 with `tape_check(…, 'f64')` against `controls.T64` | `test_interp_dryrun_analysis.py::test_R20_tape_regret_is_divided_by_J_star`, `::test_R20_t64_blocks_on_everything_and_t32_on_the_measures_of_runs_that_reach_the_exit`; `test_interp_dryrun_controls.py::test_t64_replays_a_float64_key_like_tape_exactly`, `::test_t64_is_the_only_exemption_from_the_x64_assertion`, `::test_t64_track_arrays_are_the_agents_float32_arrays`; `test_interp_dryrun_script.py::test_R20_to_R28_dry_run_required_verdicts` |
| R21 | Phase k on a real-lap state starts on the recorded lap, (k/4)·dt earlier, interpolated | **`stimuli`**: `Cell.x0_before`, `prev_action_before` and `accel_before`, taken from the recorded step one control step before x0. `initial_state(k)` gives x0 + w(x0_before − x0), w = k/n_phases; the new `initial_aux(k)` interpolates accel with the same weight and keeps prev_action at prev_action0, the command held over the step that ends at x0 (review finding A2: it was blended toward the step before's command). `phase_rule()` returns 'lap', 's_shift' or 'missing'; a 'missing' cell raises `LapStepMissingError`, a `RuntimeError`, so no `except ValueError` swallows it. Also new: `mirrored_start()`, the `RealState` fields, `attach_lap_step`, `load_real_states`, `build_real_design` carrying the earlier step (flag 'run_start' for a state at its run's start), `cell_record` (designed cells hash as before) and the design check `real_phase_starts` (`real_phase_check`). **Cross-file**: `rollout.make_batch` takes prev_action and accel from `initial_aux(phase)`. `answer_key.q_rollout_guess(…, phase)` starts Q from the phase's aux, and `full_recipe` passes its phase. `answer_key.key_for`'s mirror shortcut builds its mate from `cell.mirrored_start()` (it mirrored x0 only). **Script**: `_append_real` reads the states with `stimuli.load_real_states`. The stimuli stage logs the rebuild instruction when real cells are 'missing'. `_require_r21` refuses the keys, controls and dryrun stages on a pre-R21 design | `test_interp_dryrun_stimuli.py` (9 tests); `test_interp_dryrun_script.py::test_R21_controls_and_dryrun_refuse_a_pre_R21_design`, `::test_R21_append_real_loads_the_lap_steps_and_keeps_the_f6_scale` |
| R22 | Q's offline match: (A) pedal law through the pedal lag; (B) signed steady-turn acceleration; (C) offline onset by replaying Q's law, rule implemented independently; tolerance from the predicted v_b; c_v re-tuned under R14 | **`controls`**: `law_a` (φ = 0.482, bisection), `turn_accel_table` and `forward_profiles`, `replay_1d`, `brake_demand_1d`, `onset_1d`, `_offline_batch`, and `offline_match` with the predicted v_b. `offline_match_runs` and `c2_planted_law` start each prediction from the run's own phase start (speed and pedal from `initial_state(phase)`), a cross-file fix for R21. `Q_LAW = 'R22'`. **Script**: the controls stage re-tunes c_v (`tune_cv`, R14) when `tuning.json` was made under another `q_law` or when its tier A / tier D (a) cells changed (`_cells_hash`: cell records and bank rows). So the next `controls` run re-tunes after R22 A and after R21 | `test_interp_dryrun_controls.py::test_law_a_*`, `::test_q_*`, `::test_offline_*`; `test_interp_dryrun_script.py::test_R20_to_R28_controls_stage_order_and_resume` |
| R23 | c1 is 15 m late against the key; its shifts go to the manifest before c1 is rolled out | **`controls`**: `c1_shifts(bank, cells, keys, tables, c_v)` gives 15 m plus Q's offline-predicted onset lead over the u = 0 key's d_b*, with a status per cell. `q_params(…, shifts=)` and `q_params_for_kappa(brake_shift=)`; c1 refuses to build without shifts. **Script**: the controls stage computes the shifts from the key table of c1's cells and writes them to `controls/c1_shifts.json` and to the manifest (`budget['c1_shifts']`: rule, c_v, Q_LAW, design hash, shift hash, status counts, lead range, per-cell shifts) **before** the control loop. c1 gets them through `_run_one_control(shifts=)`, and its rollout meta carries the shift hash, so changed shifts re-run c1 | `test_interp_dryrun_controls.py::test_c1_*`; `test_interp_dryrun_script.py::test_R20_to_R28_controls_stage_order_and_resume` |
| R24 | c2's P1a equals Q's; planted-law check; P1b reported; power statement | **`controls`**: `c2_planted_law(c2_runs, q_runs, bank, cells, tables, c_v, r_lock)` compares the c2/Q v_min ratio with the offline prediction within 2 points on cells with max\|κ\| > 1/12, and requires 1 ± 2 points elsewhere. Lock-limited cells (1/max\|κ\| ≤ `lock_radius` = 8.16 m, from the model table's 'lock' rows) are listed, not judged. A crash of either control on a judged cell fails. `CONTROLS['c2']` requires P1a `SAME_AS_Q` and planted_law; P1b is reported. **`analysis`**: `power_table` adds the P1b-sat row (`_saturation_plant`, `min_q1`). **Script**: c2's P1a is judged under Q's regime (G1's q1, q10), so it can differ from Q's only through the driving, and must equal Q's verdict. P1b is computed under c2's own (12, 20) and reported. The dry run logs each agent's P1b-sat row | `test_interp_dryrun_controls.py::test_c2_planted_law_*`; `test_interp_dryrun_analysis.py::test_R24_*`; `test_interp_dryrun_script.py::test_R20_to_R28_dry_run_required_verdicts` |
| R25 | Simulator forced onsets; P3 calibrated to b*; AD15 folds by mirror-pair condition; separation against b* | **`analysis`** (by its builder): the key table gains `t_b_star` and `forced_star`. `simulate_runs` sets forced onsets. Also `p3_b_star`, `p3_bands`, `p3` (confirm, refute and judge as written in R25; not testable without b*), `p3_folds` (AD15) and `p3_separation` against b*. `p3_fit`, `p3_separation` and `simulate_runs` refuse a key table without the two columns. **Script**: the dry run rebuilds the key table (so the stale saved one is replaced) and logs b*, \|b_idx − b*\| and \|b_fd − b*\|. The probes stage rebuilds a saved key table that lacks `forced_star` | `test_interp_dryrun_analysis.py::test_R25_*`, `::test_AD15_*`; updated `test_interp_analysis.py` (P3 rule cases at b* = 1, power-table row counts) |
| R26 | c3: pure time-to-corner trigger; τ_c the smallest with k ≤ 1 on every tier A/B braking cell from Q's offline profile; pre-P3 checks; a missed refute is a power finding | **`controls`**: `_c3_target` (pure trigger, feed-forward only, no "earlier of" rule) and `tau_c_k1` (bisection on the 1-D replay of c3's own law, pedal lag included; infeasible starts listed; τ_c is the bisected threshold plus `TAU_C_GUARD` = 1e-3 s, with the threshold, the guard and the binding runs' trigger margin recorded: review finding C1, W12). `c3_fraction`, plus the new `c3_checks`: k_le_1 (the largest k on harness states before s_c0 is ≤ 1); entry (\|v_e − T(s_c0)\| ≤ a_dec(v_e)·DT, T being Q's target row); overspeed_crashes (none on tiers A/B). Checks are on runs of verified braking tier A/B keys whose own onset is not forced; no judged run returns 'no judged run', which is not a pass. `CONTROLS['c3']` requires the three checks and expects P3 `refute:M_idx`; `TAU_RULE = 'R26+guard'`. **`analysis`**: the P3-hybrid power row. **Script**: τ_c comes from `tau_c_k1` on c3's cells whose u = 0 key is verified and braking, and is refitted when the rule, Q's law, c_v or those cells change. `tuning.json` holds τ_c and a summary, `controls/tau_c.json` the per-run rows; the old `d_b_key` is dropped. A missed expected P3 is reported as a power finding, never blocking | `test_interp_dryrun_controls.py::test_c3_*` (the harness test also covers `c3_checks`: pass, an entry 6 m/s fast, an overspeed crash, forced and flat runs left out); `test_interp_dryrun_analysis.py::test_R26_*` |
| R27 | F6 halving as pre-registered; vacuity flagged; statement | **Script**: if K lands in class 1 at the registered F6 amplitude, `_control_verdicts` gives status **'halve F6'**. It blocks, and its action names the rebuild (`stimuli --real --force --f6-scale 0.5`, which rebuilds F6 for every agent) and the reruns. Class 2 or 3 at the registered amplitude passes, and no halving is triggered. After the halving (`_f6_scales` < 1) class 2 or 3 is required, and class 1 fails, since protocol 7 halves once. A class 2 reached vacuously (R10's `vacuous` in P4's details) is flagged. R27's statement is recorded with K's verdict. `--f6-scale` now defaults to the saved design's scale (`_saved_f6_scale`), so a later rebuild never undoes the halving | `test_interp_dryrun_script.py::test_R20_to_R28_dry_run_required_verdicts` (class 1 at 1.0 asks for the halving; class 2 vacuous at 0.5 passes, flagged; class 1 at 0.5 fails), `::test_R21_append_real_loads_the_lap_steps_and_keeps_the_f6_scale` |
| R28 | Viability solves for c1's crashed u = 0 runs before the tag; the dry run checks they type as overspeed | **`analysis`**: `no_return_resolved` (the test `measures.crash_type` applies before its ellipse fallback) and the run columns `nr_resolved` and `nr_s`. `viability_typing(runs, keys=None)` passes when every judged crashed u = 0 run is resolved by the viability solves and typed overspeed. With the key table it judges only cells whose u = 0 nominal key is verified, and lists the others (`no_key_cells`); see W3. **`controls`**: `CONTROLS['c1']` requires `viability_typing`. **Script**: `_c1_viability` takes `viability_targets` with no cap, keeps phase 0 and the verified-key cells, and solves them in the key workers through `_anchored_job` ('viability', agent `control-c1`, snapshot '0-ph0'). Cached solves are reused, and the summary goes to the manifest (`validation['c1_viability']`). `_viability_map` (shared with `analyse`) feeds them to the dry run's `CachedKeys`, so c1's run table carries the typing path | `test_interp_dryrun_analysis.py::test_R28_*` (including the key-table scope); `test_interp_dryrun_script.py::test_R20_to_R28_*` |
| K | C(a) logged at k = 5 too | `controls.knn_fallback(run_k, c_a_at=)` records `C_a_k1`, `C_a_k5`, `fallback` and `k`. C(a) at k = 5 is logged on both paths: from the fallback's own runs, or, when k = 1 passes, from `c_a_at(5)`, which the script's `_knn_fallback` runs on the condition (a) cells only (`_run_one_control(only=)`; C(a) reads no other condition). The script merges the record into K's meta. The dry run reports both C(a) with K's P4 | `test_interp_dryrun_controls.py::test_knn_fallback_logs_c_a_at_both_k` |

**The controls stage, in order** (`stage_controls`):
1. c_v under R14.
2. τ_c under R26.
3. c1's shifts under R23, written to the manifest.
4. The seven in-process controls: T32, Q, Qp, c1, c2, c3, K.
5. c1's viability solves under R28.
6. T64 under R20.

Saved runs are reused only when everything in `want` (`_control_want`) is unchanged: c_v, the design hash, the frozen code hash (`src/apexzero/interp` and the script), `Q_LAW` for the Q family, τ_c for c3, the shift hash for c1, and the tape key ids for T and T64. The dry run judges only rollouts whose meta matches the inputs the controls stage would use now (`_current_control_wants`: `tuning.json`, `c1_shifts.json`, the design, the code and T's tapes); any other control gets the blocking status 'stale rollouts' (W14). An unchanged c1 shift record is kept with its first `written` time (W15).

**The dry run's required verdicts** (`_control_verdicts`, `controls.CONTROLS` and `controls.T64`):

| Control | Required (blocks the tag) | Expected or reported (never blocks) |
|---|---|---|
| T (T32) | errors within τ on runs reaching s_c1 + 250 m | termination; regret/\|J*\|; crashed cells with their key's margins |
| T64 | regret/\|J*\| ≤ 1e-6; errors within τ; no termination | — |
| Q | offline_match; P1a ∈ {confirm, cautious but lawful}; P3 confirm; P4 class 4 or 5; P5 symmetric | — |
| Qp | no crash | — |
| c1 | P2 confirm; P2b onset share > 0.5; viability typing (R28) | — |
| c2 | P1a equals Q's verdict; planted law passes | P1b |
| c3 | k ≤ 1; entry within one step; no overspeed crash on tiers A/B | P3 refute:M_idx (a miss is reported as a power finding) |
| K | P4 class 2 or 3; class 1 at the registered F6 gives 'halve F6' | vacuity flag; R27's statement; C(a) at k = 1 and 5 |
| any | rollouts made with the current inputs (otherwise 'stale rollouts') | — |

### Decisions made while wiring

| # | Decision | Reason |
|---|---|---|
| W1 | **One planted-law check for c2:** `controls.c2_planted_law`. The duplicate `analysis.c2_planted_law` (its prediction supplied by the caller) and its test were removed | The two builders wrote checks with different crash rules: a per-run fail, against a per-cell median over comparable runs. The prediction needs Q's offline profile, which lives in `controls`. The controls version was checked on the real design (780 runs pass; 96 lock-limited runs listed) |
| W2 | **c2's P1a is judged under Q's regime** (G1's q1 = 12.02, q10 = 21.32); P1b under c2's own (12, 20) | R24 requires c2's P1a to equal Q's. Under different q10 the bulk sets could differ for reasons other than the driving. On tier A both bounds give the same bulk geometries (R ≥ 22.9 m) |
| W3 | **R28's scope:** c1's crashed u = 0 runs on cells with a verified u = 0 nominal key | The viability solves start from that key, and crash typing compares against it. 88 of c1's 354 cells (tier B, D = 250) have no agent-free nominal key, because no keys-stage item solves them. Cold-solving them would cost about 8.5 CPU-h for a control check. Their crashed u = 0 runs are counted and listed (`no_key_cells`), not judged. Every judged crash must be resolved: an unresolved solve fails, the analysis builder's strict reading |
| W4 | **c3's entry check** is \|v_e − T(s_c0)\| ≤ a_dec(v_e)·DT (one control step of full braking), with T Q's target row. **k ≤ 1** is strict, with no float32 tolerance; the real-design check reached 0.9951 (at the guarded τ_c of W12: 0 of 480 A1 runs above 1). A check with no judged run is not a pass | The ruling says "entry within one step of Q's target" and "k ≤ 1" |
| W5 | **τ_c ≈ 1.19 s, not the plan's ≈ 1.07 s** (A1 cells at c_v 0.90: threshold 1.18790 s, τ_c 1.18890 s with W12's guard) | `tau_c_k1` replays c3's own law (flat out to the trigger, pedal lag included). The plan's figure used Q's speed at Q's onset, under which k can exceed 1 in c3's own runs. The ruling's condition is c3's k ≤ 1 |
| W6 | **Resume keys of the controls stage** (above). c_v is re-tuned on a new `Q_LAW` or a change of its cells; τ_c is refitted on a change of rule, law, c_v or cells | So R22 A's re-tune and R21's new tier D starts are never skipped by a resumed stage |
| W7 | **T64 runs in a spawned one-worker pool** with x64 switched on in its initializer. It refuses when its design hash or tape key ids differ from the stage's | R20 asks for "its own float64 process". The main stage keeps `assert_x64_off` |
| W8 | `offline_match_runs` and `c2_planted_law` take v0 and the pedal from the run's phase start (`initial_state(phase)`) | R21 makes a real-lap start's speed depend on the phase. Tiers A and B are designed starts, where nothing changes |
| W9 | `--f6-scale` defaults to the saved design's F6 scale, else 1.0. Two scales in one design are an error | A rebuild for any other reason (R21 again, swap.json) must not silently undo the one halving |
| W10 | `_require_r21` refuses keys, controls and dryrun on a design with 'missing' real cells | They would otherwise fail deep inside with `LapStepMissingError`, or solve keys at the old frozen starts |
| W11 | The probes stage rebuilds a saved key table that lacks `forced_star` | The table saved before R25 lacks the onset columns |
| W12 | **τ_c is R26's bisected threshold plus `TAU_C_GUARD` = 1e-3 s** (`TAU_RULE = 'R26+guard'`, so an unguarded τ_c is refitted). `tau_c_k1` records `threshold`, `guard` and `trigger_margin_min_m` (τ_c v − d_c of the binding runs at their trigger step, 1-D replay) | Review finding C1. c3 is flat out until it triggers, so k changes only when τ moves a trigger to another control step, and the smallest feasible τ is exactly such a step. At the threshold the binding run's trigger margin is about 1e-5 m, below float32 resolution of s (6.1e-5 m at 900 m): on the real A1 cells the harness at τ_c (1.1879034 s) passed with k 0.9951, and one float32 step of τ lower gave k 1.365 on cells 0 and 1. "Smallest" can only hold at control-step resolution; 1e-3 s is 2% of a step (about 9 cm at the binding run's 88 m/s). A larger τ only makes runs trigger earlier, so k can only fall. With the guard, τ_c is 1.18890 s and all 480 A1 runs keep k ≤ 1 through the harness (max 0.9951) |
| W13 | **R21's prev_action at phase k is prev_action0**, the command held over the control step that ends at x0; accel stays interpolated with the state | Review finding A2. The env's prev_action is the command being applied (`racing.step_no_reset`); every start (k/4)·DT before x0 lies inside the step whose command is prev_action0. The blend toward the step before's command was off by up to 0.23 in pedal at k = 3 (the pedal command changes by up to 0.309 between the two steps on the 89 real states with a step before; 11 above 0.1). The K diagnosis's prototype, which gave the gate 9/9, used the held command. accel is the env's trailing-step acceleration, which the interpolation estimates |
| W14 | **The dry run judges only current control rollouts**: `_control_verdicts` compares each rollout's meta with `_current_control_wants` (the same `_control_want` the controls stage resumes on). Any mismatch is the blocking status 'stale rollouts', listing the inputs that differ | Review finding C2. A skipped control kept its older file (e.g. the as-built c3 at τ 0.788 while τ_c is None), and a failed T64 returns 'error' without failing the stage, so an older design's T64 file would have been judged |
| W15 | **c1's shift record is kept when unchanged** (same shift hash, design, c_v and Q law): `controls/c1_shifts.json` and the manifest keep its first `written` time | Review finding C5. It was rewritten on every controls run, so the manifest could show a time after c1's rollout and R23's "written before c1 is rolled out" rested on the hash alone |
| W16 | **K's C(a) at k = 5 is logged on both paths.** When k = 1 passes the gate, K is also run at k = 5 on the condition (a) cells only, for its C(a); k = 1's runs are kept | Review finding C4: C_a_k5 was None on the path expected after R21, so plan step 1's "log K's C(a) at k = 5" was not met there. C(a) reads only condition (a), so the extra run is 20 corners × 4 phases |
| W17 | **`real_phase_check` after the layout.** Phase 0 and every 'lap' phase must start from exact mirrors. An s-shifted mirror pair whose members' P differ (each from its own direction's key) is listed (`p_mismatch`, `p_mismatch_max`), not failed. `_save_cells` records the check after the layout in `checks.json` (`real_phase_starts_after_layout`) and logs it | Review finding A3. The check skipped s-shifted phases while P was None, and nothing re-ran it after the layout; with the saved P the E1 run-start pair 504/510 (G1 Norisring/446, P 3.70774 against 3.70883) starts up to 0.8 mm apart at phases 1–3. This predates R21 and is the same per-member P that synthetic mirror pairs have (below) |
| W18 | **b*'s record counts the included cells without a verified key** (`n_no_key`) and P3's details the agent's runs on them (`n_runs_no_key` of `n_runs`). The cell sets are unchanged pending a ruling | Review finding A1 (below, points for a ruling note) |

### Points for a ruling note (from the three builders; not changed here)

- **Run-start states.** 49 E1 cells from 7 states are their run's own designed rolling start (G1 Hockenheim/248, Melbourne/348, Norisring/446, Oschersleben/418 and Zandvoort/336; G2 Oschersleben/418; G3 Zandvoort/336). No recorded step comes before them, so they keep the s-shift ('run_start').
- **Interpolation against integration.** Linear interpolation between the two recorded steps differs from integrating the simulator under the held action by up to:
  - 1.5 cm in s and 1.7 cm in n;
  - 0.0016 rad in heading;
  - 0.09 m/s in vx;
  - 0.036 in pedal.

  That is 0.038 at the observation scales, against the frozen shift's 0.67 m and 1.52 m/s. R21 says "interpolated", so that is what is implemented.
- **Δs_k/u_k on real cells** runs from 0.41 to 1.80, median 1.00. P is a layout quantity at the sighting speed, while the time shift moves s by the start speed's distance.
- **Mirror members with different D.** In 31 of 170 tier A/B pairs the two members' D differ, by up to 208.5 m (pair 21, R22.9/T90 A2: 250.0 against 41.5 m). AD15's folds now follow the pair, so the leak is closed either way; the layout question stays open.
- **Onset knife-edge.** On a synthetic R9.4/180 hairpin, brake demand hovers at 0.3 in the ramp. It did not occur in the real design's 1,248 Q runs, but could after the c_v re-tune.
- **P1b-sat** scans the support edge q1 itself, keeping q10 and the clip radius. Holding the agent's edge and moving only the saturation point gives the same verdict at the agent's own q1.
- **Q's start (3)** for keys solved from now on uses Q's R22 law with the c_v in `tuning.json` at the time. It is a starting guess, not part of the key id; cached keys keep their ids.

Added by the review of the R20-R28 changes (below):

- **b* and the agent's b are on different cells (finding A1; needs a ruling before the tag).** P3 drops only EXCL_ALL (`infeasible_bound`, `unsolved`), so it keeps runs on cells whose key is 'missing'; `p3_b_star` can use only verified keys. On the saved key table, 88 of tier B's 96 D = 250 cells are 'missing' (no keys-stage item solves them, W3): 26% of the 340 tier A/B cells. b* depends on its cells (read-only on the saved key table, A2 left out as a stand-in for forced key onsets: 0.870 on the 164 keyed cells, 0.907 without the recovery cells, 1.140 on A1 alone), and the dry run cannot see the mismatch, because `simulate_runs` builds runs only on verified keys. After the tag it also drifts: `anchored-keys` cold-solves the missing nominal key of every viability job, so a D = 250 cell enters b* only where some agent's u = 0 run crashed there, and b* would depend on other agents' crashes. Options: (a) solve the 88 nominal keys in the keys stage before the tag (about 8.5 CPU-h; the only option that meets R25's "on the agent's included cells" as written and removes the drift; R17c is the precedent); (b) restrict P3's comparison to keyed cells (a change to P3); (c) accept and report the mismatch, with b*'s cells frozen at the tag. W18 makes the counts visible; nothing else is changed until the ruling.
- **R24's planted-law scope (finding C3).** Lock-limited cells (raw radius 1/max\|κ\| ≤ r_lock = 8.157 m) are listed, not judged; plan step 9 named this exclusion ("raw R > 7.6 m; Q is lock-limited below"), R24's text does not. All 96 listed runs would fail if judged (read-only rerun on the saved design, R22 law, c_v 0.90): c2 crashes in all 24 runs at raw R 6.0 (R6/180) and all 24 at raw R 7.5 (R7.5/180), where the cap's overspeed shows as crashes rather than through Q's lock limit; at raw R 6.37 and 7.64 (the 90° cells) it completes but misses the prediction by up to 0.026 and 0.057. The 224 judged runs pass (largest difference 0.0066). W1's "96 lock-limited runs listed" should say that they would fail.
- **R28's scope (W3)** is a similar narrowing: the 88 tier B D = 250 cells' crashed u = 0 runs are listed, not judged.
- **Phase-k starts on the cut taper (finding A4).** On the R21 design the phase-3 start lies before the lead-in's exact range in 1,121 of 1,490 'lap' cells (988 tier D, 91 E1, 42 E2), by up to 3.16 m; x0 itself is at least 0.03 m inside it (median 1.56 m). The road there is identical across every cell that shares a start, so no condition is confounded, and the old s-shift did the same. R21's "on the recorded lap" holds for the car's state, not for the road under the first metres.
- **Mirror pairs started by s-shift (finding A3, W17).** Each member's P comes from its own direction's key, so phase k > 0 of a mirror pair starts (k/4)\|ΔP\| apart in s. On the saved design 75 synthetic pairs (40 A1, 35 B) have different P, by up to 2.3 cm, so their phase-3 starts are up to 1.7 cm apart; after R21 the only real pair affected is E1's run-start pair 504/510 (0.8 mm). This predates R21. 'lap' cells (F4 included) are exact mirrors.
- **P3 is 'not testable' when b* is undefined (finding A5).** `analysis.p3` adds this path, which R25's text does not mention. Without it, an undefined b* would read "b outside the band" as true and give a spurious refute.

### Review of the R20-R28 changes (2026-10-20)

Two reviews (agent-facing: A1–A5; controls: C1–C5) were each verified before any change. Nothing ran a pipeline stage or wrote under `outputs/psychophysics/`; the real-design checks were read-only, in memory.

| # | Finding | Verified | Action | Test |
|---|---|---|---|---|
| C1 | c3's τ_c sits on a trigger threshold, so `k_le_1` is a float32 coin flip | Confirmed: on real cells 0/1 the 1-D threshold is 1.1879033 s, the binding trigger margin 1.05e-5 m at s ≈ 897 m; the harness at the threshold gives k 1.365, at τ_c 0.9951. Pilot mini-grid: margin 8.5e-6 m, k 1.69 just below | Fixed (W12): τ_c = threshold + 1e-3 s; real A1: τ_c 1.18890 s, 480 runs k ≤ 1 | `test_c3_tau_c_guard_keeps_the_binding_trigger_clear_of_float32`; strict k ≤ 1 in `test_c3_tau_c_through_the_harness_on_the_pilot_minigrid` |
| A1 | b* is computed on other cells than the agent's b | Confirmed: 88 of 96 tier B D = 250 cells 'missing' (26% of tier A/B); b* 0.870 / 0.907 / 1.140 by cell set; post-tag drift through viability keys | Needs a ruling before the tag (points for a ruling note); counts recorded (W18) | `test_R25_b_star_counts_the_included_cells_without_a_key` |
| A2 | R21 blends prev_action toward the step before | Confirmed: pedal command changes up to 0.309 between the steps (11 of 89 states above 0.1) | Fixed (W13): held command prev_action0 at every phase; accel interpolated | `test_phase_k_keeps_the_held_command_and_interpolates_accel`; updated `test_phase_k_starts_on_the_recorded_run` |
| C2 | The dry run judges whatever control rollouts are on disk | Confirmed: `_control_verdicts` read no input from the meta | Fixed (W14): 'stale rollouts' blocks | `test_dry_run_judges_only_rollouts_made_with_the_current_inputs` |
| C4 | K's C(a) at k = 5 logged only on the fallback | Confirmed (`C_a_k5` None when k = 1 passes) | Fixed (W16) | `test_knn_fallback_logs_c_a_at_both_k`; `test_knn_fallback_logs_k5_on_the_condition_a_cells_when_k1_passes` |
| C5 | c1's shift record rewritten on every controls run | Confirmed | Fixed (W15) | `test_R20_to_R28_controls_stage_order_and_resume` |
| A3 | The R21 mirror check passes only before the layout | Confirmed (predates R21; synthetic pairs have the same per-member P, up to 1.7 cm at phase 3 on the saved design) | Fixed (W17): listed after the layout, recorded by `_save_cells`; ruling-note point | `test_real_phase_check_after_the_layout_lists_per_member_p` |
| C3 | R24's lock-limited exclusion and R28's narrowing are not in the ruling text | Confirmed: all 96 listed c2 runs would fail if judged (48 crashes, misses 0.026 and 0.057) | Ruling-note point (no code change: plan step 9 named the exclusion) | — |
| A4 | Phase-k starts sit on the cut taper | Confirmed: 1,121 of 1,490 lap cells at phase 3, up to 3.16 m; road identical across cells sharing a start | Ruling-note point | — |
| A5 | New 'not testable' when b* is undefined | Confirmed, informational | Ruling-note point | existing R25 rule-table test |

No finding was refuted.

### Read-only checks on the real design (2026-10-20)

- **The saved design is pre-R21.** `real_phase_check`: 1,539 real cells 'missing'. `_require_r21` refuses it.
- **`stimuli --real --force` reproduces the design.** In memory, `build_design(sup)` reproduces the 490 pre-layout cell records of `cells_prelayout.json` and the 163 synthetic rows bit for bit. `build_real_design` on top, with `load_real_states`, `swap.json` and F6 1.0, gives the saved real rows bit for bit, and all 1,539 real cells keep their id, stimulus, pair and x0. `real_phase_check` passes: 1,490 'lap', 49 'run_start', 0 'missing', mirrors exact. About 5 CPU-s.
- **Keys to re-solve.** 255 phase-matched keys on real cells get new ids (V7 failed, so every braking cell has them):
  - 99 in 'phase' (P0: E1 real and isolated rungs, 33 cells);
  - 156 in 'phase_P1' (E1 twins and E2, 52 cells);
  - none in 'phase_B600' or 'phase_P2'; tier D has s-OCP keys only.

  At about 85 CPU-s each (warm) that is about 6 CPU-h, about 1.2 h wall on 5 workers.
- **T32's verdict path on the saved `control-T` rollouts.**
  - Errors within τ on all 173 runs that reach s_c1 + 250 m; worst ratios to τ: v_e 0.0043, v_min 0.0093, d_b 0.0006.
  - 11 crashes reported, on cells 26, 27, 92, 93, 108, 109, 119, 123, 125, 130 and 131. Each tape key sits on its solver margins: corridor 3.0 cm (the key's `safety`) and slip 1.0e-4 rad (`slip_safety`).
  - Regret/\|J*\|: at most 1.006 (cell 130; the old denominator gave 166).
  - T32 passes. T64 needs its own run.
- **The three builders' checks** (summarised from their reports):
  - **Q on the old design:** offline_match passes all 1,248 compared runs under R22.
  - **c1:** measured onset against the key is −16.5 to −13.7 m on the A1 cells, against the 15 m planted.
  - **c3:** τ_c 1.188 s, every braking run feasible, maximum k 0.9951, entry error ≤ 0.2 step, no crashes. That τ_c sat on the knife edge of W12; with the guard τ_c is 1.18890 s and all 480 A1 runs keep k ≤ 1 (max 0.9951).
  - **c2:** the planted-law check passes on 780 runs.
  - **Simulated agents:** the dry run returns 13 of 13 planted verdicts on the real key table; the lawful agent's P3 confirms (b 0.864 against b* 0.854).
  - **Separation check:** passes (gaps 0.397 and 0.854).

### Tests

All 417 interp tests pass (411 plus the review's 6 regression tests), and `ruff check` is clean on `src/apexzero/interp`, `scripts/psychophysics.py` and the interp tests. The model tables are built once per session for `controls` and `dryrun_controls` (`test_interp_controls.model_tables_once`), and the three c3 tests share one τ_c fit, so the suite runs in two halves under 3 CPU-min each:
- `controls`, `dryrun_controls`, `dryrun_stimuli`, `dryrun_script`, `dryrun_analysis`: 85 tests, about 166 CPU-s;
- `analysis`, `answer_key`, `measures`, `r17`, `r18`, `r19`, `review`, `rollout`, `rulings`, `script`, `skeleton`, `stimuli`, `support`, `v3_validation`, `v7_remedy`: 332 tests, about 178 CPU-s (close to the limit; the next added test belongs in the first half).

New files, all untracked until `git add`:
- `tests/test_interp_dryrun_analysis.py` (24);
- `tests/test_interp_dryrun_controls.py` (21);
- `tests/test_interp_dryrun_stimuli.py` (11);
- `tests/test_interp_dryrun_script.py` (6).

### What must be re-run, in this order

Nothing below has been run. Never pass `--force` to `keys`: it would re-solve every key.

0. **A ruling on b*'s cells (review finding A1) is needed before the tag**, and best before step 2: if it chooses to solve the 88 tier B D = 250 nominal keys, the keys stage must first be given that item (a code change not made here), so one keys run covers it before the dry run. The other options change no stage.

1. **`uv run python scripts/psychophysics.py stimuli --real --force`** (R21). Rebuilds the design with the recorded steps attached, keeping the saved F6 scale (1.0). Check that the log shows `real_phase_starts` PASS (1,490 lap, 49 run_start, 0 missing) and a new design hash.
2. **`uv run python scripts/psychophysics.py keys --priority P2`** (no `--force`). Re-lays out every cell from the cached s-OCPs (W, P and A2's D; nulls per W; all cache hits expected), refreshes the phase cycles and, resuming from the cache, re-solves the 255 phase-matched keys of the real cells (new ids; about 6 CPU-h), then the acceptance checks. Every other key is a cache hit. The controls must come after this run, because the phase cycles (P) are refreshed only here.
3. **`uv run python scripts/psychophysics.py controls`**, in the stage order above:
   - c_v is re-tuned under R14, because `tuning.json` has no `q_law`;
   - τ_c under R26;
   - c1's shifts are written to the manifest;
   - every control is re-run (the design and code changed); K at k = 1, plus k = 5: on every K cell if C(a) < 0.9, else on the condition (a) cells only, for its logged C(a) (W16);
   - c1's viability solves (on the earlier diagnosis about 300 solves, about 40 CPU-min);
   - T64.

   S1's kNN buffer already exists. `--knn-buffer` is not needed.
4. **`uv run python scripts/psychophysics.py dryrun`**: the key table with R25's columns, the dry run, the power table, the separation check, the controls' verdicts (T64 included) and the pre-registration record.
5. **Only if K's status is 'halve F6'** (R27): `stimuli --real --force --f6-scale 0.5`, then `keys --priority P2` (layout of the new F6 rows; F6 has no keys), `controls` (c_v resumes unless its cells changed, since F6 does not touch tier A or tier D (a); the controls re-run on the new design hash; cached viability solves are reused) and `dryrun`.
6. **Commit and tag** if every planted verdict and control passes. That is the user's call.

## Pre-registration record (written by `dryrun`; frozen at the tag)

<!-- prereg-record:begin -->
```json
{
 "band": 0.1,
 "code_hash": "df06efe940abe0744e9208b951042dc2c9c8bb4ff0d171be941d51790142e58b",
 "controls": {
  "K": "FAIL",
  "Q": "FAIL",
  "Qp": "pass",
  "T": "FAIL",
  "c1": "FAIL",
  "c2": "FAIL",
  "c3": "FAIL"
 },
 "dry_run_passed": false,
 "frozen_code_hash": "8e0fa113ee7f5accb3262677e497dc095b360627e48140532d95ec18ac6bf702",
 "key_cache_hash": "54ff23e29a720441cb5d8349341af5041474ea7a220febf5b7f28c5fa28bb847",
 "p3_demoted": false,
 "stimulus_hash": "90a24294fea9c0a845bfb3609fe14a8590aca01acac424ce46a7e4751901b403",
 "tag": "psychophysics-prereg-v2",
 "written": "2026-10-20T00:27:11"
}
```
<!-- prereg-record:end -->
