# Drivers-lite pilot: 2025 qualifying, four circuits (PILOT, no rankings)

*Pilot of plan section 3.4 (method: plan-v1 section 6). It tests the measurement pipeline, not the drivers. Every teammate number below is an example of what the pipeline outputs, with its error bars; none of it is a claim about who is faster.*

Code: `src/apexzero/drivers/`, run by `scripts/drivers_pilot.py` (stages `registration`, `pilot`, `tune`, `injection`, `report`). Data: OpenF1 (CC BY-NC-SA 4.0) and the TUMFTM centrelines.

**Lockbox.** The circuits that pass registration are all R-test circuits. Only the public telemetry is registered and timed on them. No RL agent, OCP or QSS lap is evaluated on them. Injection-recovery laps on these circuits use each session's own pooled speed profile. The QSS-profile check and the smoother tuning run on R-dev geometry only (`qss_model` refuses R-test names in code). The synthetic noise *levels* are calibrated against the real pilot sessions (section 2). That measures the feed; it does not fit anything to those circuits.

## 1. Registration onto the TUMFTM centreline

The fit is a similarity transform: a 5-degree rotation grid with both handednesses, then trimmed ICP. The gate is the plan's: RMS <= 3 m and lap-length mismatch <= 1%. The length is the feed's own driven-line length (a spline through the samples of the fastest laps) times the fitted scale, compared with the centreline length. Fitted rotations are -0.1 to +0.2 degrees and scales 0.9940 to 1.0029, with no reflection. So the feed and TUMFTM frames share orientation and units up to a translation.

| Circuit (OpenF1) | TUMFTM | Layout | RMS (m) | p95 (m) | Length | Scale | Outside corridor | Excursion p99 (m) | Inside offset (m) | Plan gate |
|---|---|---|---|---|---|---|---|---|---|---|
| Melbourne | Melbourne | **old** | 5.42 | 12.7 | -1.38% | 0.9960 | 12.7% | +16.0 | +1.68 | fail |
| Shanghai | Shanghai | current | 4.46 | 8.4 | -1.36% | 0.9983 | 12.2% | +7.4 | +1.60 | fail |
| Suzuka | Suzuka | current | 1.72 | 3.4 | -0.36% | 1.0006 | 0.5% | -0.2 | +0.58 | pass |
| Sakhir | Sakhir | current | 3.35 | 6.2 | -1.31% | 0.9984 | 4.0% | +2.5 | +1.94 | fail |
| Catalunya | Catalunya | **old** | 6.90 | 16.9 | -0.83% | 0.9987 | 11.3% | +27.5 | +0.14 | fail |
| Montreal | Montreal | current | 2.46 | 4.5 | -0.83% | 1.0013 | 1.9% | +0.2 | +1.36 | pass |
| Spielberg | Spielberg | current | 3.48 | 7.0 | -1.07% | 0.9940 | 13.3% | +3.8 | +2.77 | fail |
| Silverstone | Silverstone | current | 4.00 | 7.9 | -1.16% | 0.9981 | 9.0% | +3.1 | +2.64 | fail |
| Spa-Francorchamps | Spa | current | 2.62 | 5.3 | -0.72% | 0.9995 | 5.6% | +3.1 | +1.69 | pass |
| Hungaroring | Budapest | current | 2.66 | 5.0 | -1.16% | 0.9992 | 7.2% | +1.3 | +1.73 | fail |
| Zandvoort | Zandvoort | **old** | 3.62 | 6.4 | -1.80% | 1.0029 | 7.2% | +11.3 | +2.72 | fail |
| Monza | Monza | current | 2.36 | 4.5 | -0.46% | 0.9996 | 4.5% | +1.2 | +1.49 | pass |
| Austin | Austin | current | 3.46 | 6.3 | -1.65% | 0.9984 | 2.7% | +0.8 | +1.98 | fail |
| Mexico City | MexicoCity | current | 3.48 | 6.3 | -1.44% | 0.9970 | 6.3% | +0.9 | +1.55 | fail |
| Interlagos | SaoPaulo | current | 3.40 | 6.5 | -1.51% | 1.0004 | 6.5% | +0.9 | +2.78 | fail |
| Yas Marina Circuit | YasMarina | **old** | 9.17 | 18.9 | -5.98% | 0.9957 | 19.1% | +35.0 | +2.40 | fail |

No TUMFTM centreline: Jeddah, Miami, Imola, Monte Carlo, Baku, Singapore, Las Vegas, Lusail. *Inside offset* is the mean lateral offset towards the inside of corners tighter than 150 m radius. *Excursion* is the distance beyond the nearer TUMFTM track edge (negative = inside the track).

**Reading.**

- The plan's gate rejects all 4 outdated TUMFTM layouts, as it should. It also rejects 8 of the 12 current layouts. It passes only Suzuka, Montreal, Spa-Francorchamps, Monza.
- The likely cause is the feed's track model, which is not the centreline. In corners tighter than 150 m radius it sits 0.6 to 2.8 m to the inside, and its lap is 0.4 to 1.6% shorter. That looks like a racing line. So both gate statistics measure the line as well as any registration error, and a twisty circuit can fail on the line alone.
- The excursion beyond the TUMFTM edges separates the two groups much more sharply. At the 99th percentile it is at most +7.4 m on current layouts and at least +11.3 m on old ones. **Proposal, not applied here:** gate on excursion p99 <= 8 m and report RMS and length as diagnostics. On this table that would pass 12 of 12 current layouts and 0 of 4 old ones. The pilot keeps the plan's gate.
- Projecting a racing-line model onto the centreline is still a fixed, monotone map, the same for every car. So s(t) stays comparable between teammates. The cost is a speed / centreline-rate ratio that varies along the lap. The smoother estimates it from all laps (section 2).

## 2. s(t) per lap, the held-out sector anchor, and synthetic calibration

What the feed looks like (measured on Monza 2025 while building the smoother):

- Projected positions scatter about 2 m *along* the track, with heavy tails. A ~0.7 m pattern sits on top, the same for every car. The snapping audit's 13 cm is the lateral scatter only.
- Speed comes in whole km/h, on its own ~3.7 Hz clock. The location and car_data clocks show no lag (best lag within +/-0.05 s).
- `date_start` is off by up to ~0.1 s: consecutive laps disagree with start + duration by up to 0.2 s. Lap and sector durations are exact. The estimated start-time corrections below have a robust sd of 53 to 75 ms, close to the 58 ms of a uniform +/-0.1 s error.

The smoother is an RTS smoother on [s, v, a] driven by white jerk. It fuses positions, speed and hard anchors at the official lap and sector times. Speed is modelled as a per-lap scale times a pooled path-ratio profile times ds/dt. The timing-line positions and each lap's start-time correction come from unanchored fits first. The median correction is pinned to zero, because a common shift is not identifiable. Hyperparameters were tuned on QSS-profile synthetic laps on R-dev geometry (Catalunya, Hockenheim). The result is jerk PSD 100 m^2/s^5 and speed sigma 1.2 m/s (pair-delta RMS 17.0 ms). The surface is flat: every grid point with speed sigma >= 0.7 m/s scored 17 to 22 ms.

**Held-out sector anchor (plan gate 3).** Each lap is refitted without the S2/S3 line, the crossing time is predicted, and it is compared with the official S1 + S2. The same check on synthetic laps calibrates the synthetic feed. The line wobble (`SynthConfig.wobble_sd`) is the one chosen from [0.005, 0.009, 0.015, 0.022, 0.03] whose synthetic held-out spread is closest to the real one. Natural lap-to-lap variation is then set per corner type x phase, from the real within-driver spread minus the synthetic measurement error (table below).

| Session | Laps fitted | Held-out bias (ms) | Held-out robust sd (ms) | p95 abs (ms) | Synthetic robust sd (ms) | Wobble | Start corr. sd (ms) | Position sigma (m) |
|---|---|---|---|---|---|---|---|---|
| 2025 Suzuka Qualifying | 93 | -28 | 9.9 | 49 | 9.0 | 0.009 | 65 | 1.55 |
| 2025 Montreal Qualifying | 88 | -3 | 15.4 | 35 | 17.4 | 0.022 | 53 | 1.77 |
| 2025 Spa-Francorchamps Qualifying | 84 | +2 | 18.5 | 31 | 15.8 | 0.022 | 75 | 1.73 |
| 2025 Monza Qualifying | 92 | -1 | 11.1 | 22 | 16.7 | 0.022 | 68 | 1.69 |

Within-driver spread of phase times (real, laps of one driver in one Q segment) against the synthetic measurement error at the calibrated wobble, median over a circuit's cells of each type, in ms (real / synthetic measurement). Where the synthetic number is larger, the synthetic feed is noisier than the real one there, which makes the budget conservative; natural variation is then set to zero.

| Session | slow entry | slow apex | slow exit | medium entry | medium apex | medium exit | fast entry | fast apex | fast exit |
|---|---|---|---|---|---|---|---|---|---|
| 2025 Suzuka | 44 / 17 | 44 / 20 | 52 / 18 | 32 / 10 | 32 / 8 | 28 / 8 | 14 / 4 | 12 / 5 | 8 / 3 |
| 2025 Montreal | 39 / 22 | 45 / 23 | 49 / 27 | 30 / 32 | 30 / 12 | 51 / 22 | 2 / 4 | 9 / 11 | 26 / 19 |
| 2025 Spa-Francorchamps | 55 / 29 | 63 / 31 | 52 / 21 | 26 / 22 | 33 / 14 | 46 / 24 | 7 / 9 | 9 / 11 | 12 / 18 |
| 2025 Monza | 50 / 23 | 43 / 24 | 32 / 21 | 28 / 24 | 21 / 10 | 46 / 29 | 18 / 17 | 11 / 9 | 20 / 19 |

In 6 of 36 circuit x cell combinations the synthetic measurement error alone exceeds the real total spread. Real lap-to-lap variation is several times larger in slow phases than in fast ones, which is why natural variation is set per cell and not as one number.

A single crossing time far from the anchors is good to 10 to 18 ms (robust sd). The synthetic column comes from the injection runs themselves. The per-session biases (-28 to +2 ms) are common to every lap of a session, so they cancel between teammates. They do show a model mismatch near that line that a physics reference would inherit.

**Segmentation.** Complexes per circuit: Suzuka 9, Montreal 9, Spa-Francorchamps 12, Monza 7. The plan asks for 12-20. Corners less than 80 m apart always merge, because at 3.7 Hz (one sample every 6-25 m) a chicane's halves cannot be timed apart. Below 12, the code keeps the natural count rather than splitting chicanes or straights to fill a quota. Each complex is split into entry / apex / exit at positions on the pooled reference speed, and typed by its apex speed (slow < 120, medium 120-200, fast > 200 km/h).

![pilot](drivers-pilot.png)

## 3. Lap filter

| Session | Laps fitted (checks 1-7 passed) | Traffic (< 1.0 s ahead in a corner) | Pass every check | Exclusions before fitting (all laps) | Teammate pairs | Laps per driver in pairs |
|---|---|---|---|---|---|---|
| 2025 Suzuka Qualifying | 93 | 50 (54%) | 43 | untimed 104, out-lap 96, in-lap 76, not push 19, tyre MEDIUM 4, deleted 1 | 5 | 1: 3, 2: 7 |
| 2025 Montreal Qualifying | 88 | 61 (69%) | 27 | out-lap 103, untimed 99, in-lap 83, tyre MEDIUM 78, not push 53, deleted 7 | 4 | 1: 3, 2: 5 |
| 2025 Spa-Francorchamps Qualifying | 84 | 35 (42%) | 49 | out-lap 102, untimed 100, in-lap 82, not push 14, deleted 4, flag 2, tyre MEDIUM 2 | 6 | 1: 3, 2: 9 |
| 2025 Monza Qualifying | 92 | 28 (30%) | 64 | untimed 104, out-lap 103, in-lap 83, not push 20, tyre MEDIUM 13, flag 7, deleted 4 | 10 | 1: 5, 2: 15 |

Traffic removes 30 to 69% of otherwise clean push laps. That is the plan's rule taken literally: any car less than 1.0 s ahead anywhere in any corner window. Spot checks at Montreal found real slow cars on out-laps and cool-down laps, on the racing line (the same lateral offset as fast cars, so not the pit lane). With the 102% rule per Q segment and shared Q segments required, pairs end up with 1-2 laps per driver (last column: laps -> number of drivers). That is the binding constraint on everything below. Tyre compounds come from the OpenF1 stints table only. Montreal's MEDIUM count is large and was not cross-checked.

## 4. Injection-recovery error budget (plan gate 2)

For each pilot circuit: 20 synthetic teammate pairs (10 pairs x 2 sessions), 3 laps per driver, comparing each driver's top-2 laps as in the real analysis. Every B driver got injected effects of +40 ms in each slow-corner entry, +20 ms in each medium-corner apex and -20 ms in each fast-corner exit. Every phase of every lap also carries natural variation at the calibrated per-cell level (section 2).

How the laps are degraded:

- resampled on the empirical location and car_data interval distributions;
- positions with Student-t (3 dof) along-track noise at each session's measured scale, 13 cm lateral noise and decimetre rounding;
- speed with a per-lap scale (sd 0.2%) and a smooth line wobble at the calibrated level (150 m correlation), then rounded to whole km/h;
- brake binarised (the timing pipeline does not use it);
- start times jittered by +/-0.1 s.

Then the laps go through the real pipeline.

*Measurement* = estimated minus true delta over the same laps: the pipeline's own error. *Recovered - injected* adds natural variation over 2 laps per driver. Its 2 sigma is the **claim limit**, the plan's "2 sigma of the resulting error".

| Corner type | Phase | Cells | Per-lap meas. sd (ms) | Meas. bias (ms) | Meas. 2 sigma (ms) | Injected (ms) | Recovered mean (ms) | Recovered - injected sd (ms) | **Claim limit (ms)** | QSS R-dev claim limit (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| slow | entry | 180 | 31 | -0.2 | 43 | +40 | +39.3 | 45.8 | **92** | 98 |
| slow | apex | 180 | 29 | -0.4 | 46 | +0 | +1.2 | 50.0 | **100** | 96 |
| slow | exit | 180 | 27 | -0.1 | 47 | +0 | -1.0 | 45.2 | **90** | 87 |
| medium | entry | 260 | 23 | -1.3 | 42 | +0 | +1.1 | 28.8 | **58** | 59 |
| medium | apex | 260 | 14 | +0.8 | 24 | +20 | +20.9 | 32.2 | **64** | 53 |
| medium | exit | 260 | 21 | +0.4 | 40 | +0 | -2.3 | 41.2 | **82** | 91 |
| fast | entry | 300 | 13 | -0.8 | 26 | +0 | -0.7 | 15.0 | **30** | 7 |
| fast | apex | 300 | 11 | +0.1 | 22 | +0 | +0.5 | 11.8 | **24** | 18 |
| fast | exit | 300 | 18 | +1.2 | 33 | -20 | -18.9 | 18.8 | **38** | 37 |

Complex level (entry + apex + exit), claim limit by corner type: slow 155 ms (measurement only 34); medium 108 ms (measurement only 40); fast 51 ms (measurement only 45).

Corner type x phase summed over the lap (the driver-profile cells of the plan's model):

| Cell | Pairs | Meas. sd (ms) | Injected total (ms) | Recovered - injected mean (ms) | Recovered - injected sd (ms) | Claim limit (ms) |
|---|---|---|---|---|---|---|
| slow entry | 80 | 29.2 | +90 | -1.5 | 63.4 | 127 |
| slow apex | 80 | 38.2 | +0 | +2.7 | 69.2 | 138 |
| slow exit | 80 | 36.2 | +0 | -2.3 | 74.8 | 150 |
| medium entry | 80 | 35.7 | +0 | +3.7 | 49.5 | 99 |
| medium apex | 80 | 17.6 | +65 | +2.8 | 55.2 | 110 |
| medium exit | 80 | 34.5 | +0 | -7.4 | 71.4 | 143 |
| fast entry | 80 | 24.2 | +0 | -2.7 | 27.4 | 55 |
| fast apex | 80 | 20.3 | +0 | +1.8 | 22.2 | 44 |
| fast exit | 80 | 32.0 | -75 | +4.0 | 33.4 | 67 |

False positives: 5.8% of 1480 zero-injected phase cells exceed their claim limit (about 5% is expected at 2 sigma; neighbouring injected phases also leak into them). Detection: 11% of 180 injected slow-entry cells (+40 ms) exceed the limit with the right sign. The rates are 10% for medium-apex (+20 ms) and 14% for fast-exit (-20 ms).

**Reading.** The injected effects come back unbiased within their error: recovered means are -2.3 to +1.2 ms off the injected values. The smallest claimable per-phase teammate effect, from one car-session with 2 laps per driver, is 24 to 100 ms depending on corner type and phase. The pipeline's own share is 22 to 47 ms. A +40 ms slow-corner entry is 0.4x its limit (92 ms), and one synthetic car-session flags it in only 11% of cases. Natural lap-to-lap variation, not the pipeline, dominates the limit in slow and medium phases. The pipeline's own error dominates in fast phases. Both average down with more laps per driver or more sessions per pair, as long as they are independent between laps. A per-car systematic error would not average down, and none is modelled. If car-sessions were independent, a season of 20 would bring the per-phase limits to 5 to 22 ms, and the corner type x phase lap sums to 10 to 33 ms (limit / sqrt(20)). That independence is an assumption; the split-half gate is what tests it. Ratios of QSS-profile limits (R-dev geometry, median calibration: wobble 0.022, natural 0 to 41 ms by cell) to pilot limits are 0.23 to 1.11, median 0.97. 8 of 9 cells agree within 25%. The exception is fast entry (0.23). There the calibrated natural variation is 0 ms, so the limit is measurement error alone, and that does depend on the profile. So the budget mostly does not hinge on the speed-profile shape. Per-lap errors also carry a part common to all laps. It cancels in teammate deltas, but it would not cancel against a physics reference (R2).

## 5. Placebo and split-half reliability (real data)

**Placebo (gate 4).** Driver labels were shuffled within each car-session (449 shuffles, top-2 laps, shared Q segments, 669 phase cells). Shuffled deltas average +0.1 +/- 0.1 ms, so neither the pipeline nor the top-2 selection creates a driver effect by itself. Mean |delta| per cell is 29 ms shuffled versus 32 ms with the true labels. A permutation test on the sum of squared deltas over all cells gives p = 0.027. That is the probability that random labels explain at least as much phase-level variance as the real ones. Read it as weak, pooled evidence that the true labels carry some phase-level structure beyond lap-to-lap variation, not as a statement about any driver. Driver, setup and per-car measurement offsets are confounded in it. A per-cell test is not possible here: with 2 laps per driver there are only 3 distinct splits.

**Split-half (gate 5).** The gate is Spearman rho >= 0.5.

| Split | Level | Cells | Spearman rho | p |
|---|---|---|---|---|
| odd vs even ranked laps, within session | phase | 372 | +0.29 | 1.2e-08 |
| odd vs even ranked laps, within session | corner type x phase | 126 | +0.39 | 5.1e-06 |
| alternate sessions (pairs in both halves) | corner type x phase | 72 | +0.11 | 0.37 |

No split reaches the 0.5 gate. The within-session halves compare one lap with one lap, and only for pairs with 2+ clean laps each. Driver, setup and any per-car measurement offset (antenna position, transponder) are confounded in them. The across-session split has only the pairs that appear in both halves of four sessions. Both are underpowered pilots of the gate. The plan's version (odd vs even rounds over a season) needs the full calendar.

## 6. Example teammate per-corner table (pilot, not a ranking)

2025 Spa-Francorchamps Qualifying, McLaren. Delta = PIA minus NOR, so positive means PIA took longer. Each driver contributes the mean of their top-2 clean laps in the shared Q segment(s) Q1, Q2, Q3. The lap delta is +17 ms, and the phase deltas below add up to it exactly. A cell would be **bold** only if it cleared both the claim limit for its corner type and phase and twice this car's own placebo spread.

| Complex | Type | Entry (ms) | Apex (ms) | Exit (ms) | Complex (ms) | Above both limits |
|---|---|---|---|---|---|---|
| C01 | slow | -42 | -4 | +19 | -27 | - |
| C02 | fast | +4 | +0 | +9 | +14 | - |
| C03 | fast | -6 | -1 | +17 | +9 | - |
| C04 | fast | +1 | +2 | -14 | -11 | - |
| C05 | medium | +64 | +9 | +13 | +85 | - |
| C06 | medium | +29 | +85 | -102 | +13 | - |
| C07 | fast | -12 | -11 | -13 | -36 | - |
| C08 | medium | +21 | +45 | -23 | +43 | - |
| C09 | medium | +1 | +11 | -6 | +6 | - |
| C10 | fast | +1 | +2 | +1 | +4 | - |
| C11 | fast | +8 | +8 | -2 | +15 | - |
| C12 | slow | -78 | -9 | -10 | -96 | - |

With 1-2 laps per driver, single cells mostly sit inside the error budget. That is the expected pilot outcome. It is why the plan pools over a season with a hierarchical model before saying anything about a driver.

## 7. Not done, or uncertain

- **Registration gate.** As written it passes 4 of 12 current-layout 2025 circuits (section 1). The corridor-excursion gate is a proposal, checked only against this table.
- **Scope.** Four circuits, one qualifying session each. Sprint qualifying was not run. Most circuits stay below the plan's 12 complexes because chicanes are never split.
- **Synthetic realism.** The synthetic feed is calibrated on two statistics per session (held-out S2 spread; within-driver spread per corner type x phase). The single wobble knob stands in for every unmodelled error source, so how the measurement error splits between phases is a model assumption (section 2 table). Natural variation is Gaussian and independent between phases. The ~0.7 m position pattern common to all cars is not synthesised; it is common-mode, but that it cancels is untested. No clock latency (none was measured) and no per-car antenna offsets are modelled. The QSS check profiles are computed on the centreline, not a racing line.
- **Smoother.** Tuning picked the stiffest acceleration model on the grid (jerk PSD 100), on a flat error surface. Held-out biases of -28 to +2 ms show it misfits some real braking zones in a way common to all laps.
- **Lap filter.** Compounds come from the stints table only, with no new/used flag. Traffic uses other cars' raw projections; samples more than 7 m from the centreline (pit lane, garage) are ignored. The tow flag is computed but unused.
- **Not built (later plan steps).** The hierarchical model (M1/M2), R2 calibration, the circularity check, the multiverse and external validity. Brake and throttle are loaded but unused.
- **Underpowered gates.** The claim limits assume 2 laps per driver, and many real pairs have 1. The placebo tests label exchangeability within car-sessions, not across a season.

Compute: the pilot stage took 80 s of CPU over four sessions. Registration takes ~2 s of CPU per session; calibration, injection-recovery and tuning take a few minutes each. Everything ran single-process at nice 15 on the M3 Pro alongside other jobs.
