# Drivers DV-B0: power of the split-half gate (plan, EXPLORATORY)

*Committed before any simulation is run. Follow-through plan item DV-B0 (`docs/plan/follow-through.md`, sections 3, 7.6 and 10 #18). The result decides whether DV-B, the confirmatory drivers study, goes ahead. It is a simulation from noise budgets already recorded in `docs/findings/drivers-pilot.md`; it touches no real session beyond the four pilot sessions.*

## Question

If DV-B runs the plan's split-half gate on 2023-25 qualifying, how likely is it to pass, as a function of the true size of the teammate-label effects? The gate (plan-v1, gate 5): Spearman rho >= 0.5 between teammate deltas from odd and even rounds.

## Inputs (all on disk)

- `outputs/drivers-lite/injection.pkl`: 80 synthetic teammate pair-sessions (20 per pilot circuit), estimated minus injected per-phase delta at 2 laps per driver in one Q segment. This is the recorded noise budget: natural lap-to-lap variation plus the pipeline's measurement error.
- `outputs/drivers-lite/persistent.pkl`: the persistent per-car error scenario (line pattern and 1 m position offset).
- `outputs/drivers-lite/pilot_*.pkl`: the four pilot lap tables. They give the yield (27 pair deltas from 40 car-sessions), the lap-count variance factors of those 27 pairs, each circuit's complex counts and time shares by corner type x phase, the pilot's lap deltas, and the pilot's own across-session split-half.
- `outputs/drivers-lite/registration*.json`: the 35 sessions that pass the corridor-excursion gate on a current layout (2023: 11, 2024: 12, 2025: 12).
- `drivers.json.gz` of those sessions, of the 2023-25 sprint qualifying sessions at the same circuits, and of 2026 qualifying at current-layout circuits: **team rosters only** (which two drivers share a car in each session). No laps, positions or timing are read from any session outside the pilot.

## Model

For pair p (two driver numbers), session s and cell c (corner type x phase, summed over the lap):

    d[p,s,c] = n[s,type(c)] * theta[p,c]  +  w[s,c] * (g[p] + eta[p,s])  +  pi[p,c]  +  k * sqrt(v[p,s]) * eps[p,s,c]

- **theta**: the teammate-label effect per complex of a corner type and phase, N(0, tau^2), constant across sessions. n is the number of complexes of that type on the session's circuit, so tau is on the scale of the injection test (+40 ms in each slow-corner entry). This is the swept quantity: **tau = 0, 5, ..., 40 ms**.
- **g + eta**: the pair's overall gap (constant g plus a per-session part eta), spread over cells by the circuit's time share w. Their sds come from the pilot's 27 lap deltas (between- and within-pair variance, minus lap-delta noise), reported with the result.
- **pi**: persistent per-car errors, constant across sessions. Off, or (scenario) one persistent-run vector per pair. With pi on, rho can pass with tau = 0: that rate is reported as the false-pass risk.
- **eps**: a recorded noise vector (one synthetic pair-session from a pilot circuit drawn at random, random sign), kept whole so the correlations between cells are kept. v is a lap-count variance factor drawn from the pilot's 27 (`PairDelta.var_factor`). **k = 1 and 1.36**, the factor by which the placebo found the synthetic budget too narrow.
- Each car-session yields a pair delta with the pilot's probability, 27/40. Each session's circuit (complex counts, time shares, noise library) is one of the four pilot circuits at random.

**Statistics.** Each pair's deltas are averaged within a half (odd or even rounds in date order, a round's qualifying and sprint qualifying in the same half). Spearman rho is taken over every (pair, cell) of the pairs present in both halves.
- **raw**: the deltas as they are, as `deltas.split_half_across` does.
- **profile**: each pair-session's deltas minus the time-share part of its lap delta, d - w * sum_c d. It asks where in the lap the time goes beyond the overall gap, which is the plan's question (Q4). Under raw, the overall gap alone can make rho pass.

**Phases.** 3 (entry, apex, exit: 9 cells, as coded) or 2 (6 cells). The recorded budgets exist only for 3 phases, so 2 phases are approximated by merging recorded phases, both ways: entry + apex | exit, and entry | apex + exit. theta is drawn per 3-phase cell and summed. Merging lets the timing errors at the dropped boundary cancel.

**Scope.** (a) the 35 qualifying sessions; (b) plus 2023-25 sprint qualifying at the same circuits; (c) plus (b) and 2026 qualifying at current-layout circuits. (b) and (c) assume those sessions register and yield like the pilot; they were not registered, and sprint qualifying has fewer laps, so (b) and (c) are optimistic.

**Calibrating tau to the pilot.** The same model on the pilot's exact design (its pairs, sessions, circuits and lap counts; halves Suzuka + Spa against Montreal + Monza, as in the pilot report) gives rho's distribution for each tau. tau_hat is where its median equals the pilot's observed rho (raw: +0.11, as recorded; profile: computed on the pilot tables); tau_80 is where the observed rho sits at its 20th percentile, an 80% upper bound. tau_hat = 0 when the observed rho is below the tau = 0 median.

**Monte Carlo.** 2,000 simulated seasons per grid point (standard error of a probability at most 0.011). Fixed seeds.

## Expected directions

P(pass) rises with tau and with more sessions, and falls with k. Raw passes more often than profile at the same tau, because the overall gap counts as signal. Two phases pass more often than three. With pi on, rho passes at tau = 0 more often than with it off. At the pilot's tau we expect the profile statistic to fail the gate (P(pass) well below 0.5), in line with the scoping estimate that even the two longest pairs reach a reliability of about 0.43.

## Decision rule (fixed now)

**Go** to DV-B if, for the profile statistic with k = 1.36 and pi off, P(rho >= 0.5) >= 0.5 at tau_80 (the profile calibration) for at least one choice the plan leaves open: 2 or 3 phases (2 phases counting only if both merges reach it) and scope (a), (b) or (c). Otherwise **no-go**: DV-B is not run, and the fallback in the two-track plan stands (2-3 pair case studies, not a gate).

The raw statistic is reported for every cell of the grid but does not enter the rule: a raw pass can come from the overall teammate gap alone and would not answer Q4. k = 1.36 because the pilot's own placebo found the k = 1 budget too narrow. tau_80 is generous on purpose: it treats setup and any persistent per-car error in the pilot as if it were driver signal, so a no-go under it is robust and a go needs DV-B's own checks.

Also reported: the tau at which P(pass) reaches 0.5 and 0.8 in each configuration, and the false-pass rate at tau = 0 with pi on.

## Outputs

`scripts/drivers_power.py` (writes `outputs/drivers-lite/power/power.json` and `docs/findings/drivers-power.md` with a figure). Labelled exploratory. Anything changed after this commit is listed as a deviation in the write-up.
