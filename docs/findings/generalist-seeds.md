# Generalist seed study: 5 seeds per DR variant (19–20 Oct 2026)

Every DR generalist (v1 base grammar; v2 30% of strips from the tilted grammar; v3 50% from a
(1/r_min)^2 tilt) was trained at 200M steps with identical arguments and seeds 0–4. Seed 0 ran on the
Mac CPU; seeds 1–4 ran on a rented NVIDIA L40S (`scripts/gpu_queue_replicates.sh`, about 430k
steps/s per run with two runs sharing the card). Evaluation: timed laps on the 12 R-dev circuits
(default car, 10 runs from 0.9 × the reference start speed). A circuit counts as completed when more
than half the runs finish.

| | v1 | v2 | v3 |
|---|---|---|---|
| Circuits completed, seeds 0–4 | 4, 4, 8, 6, 5 | 5, 7, 6, 4, 7 | 6, 5, 5, 7, 6 |
| Mean ± sd | 5.4 ± 1.7 | 5.8 ± 1.3 | 5.8 ± 0.8 |
| Median gap to the optimal lap (completed circuits) | 7.2% | 6.7% | 7.3% |

Seeds completing each circuit, of 5:

| Circuit | v1 | v2 | v3 |
|---|---|---|---|
| Brands Hatch | 3 | 5 | **0** |
| Hockenheim | 2 | 2 | 3 |
| Moscow Raceway | 2 | 4 | 3 |
| Norisring | 0 | 0 | 0 |
| Nürburgring | 5 | 5 | 4 |
| Oschersleben | 1 | 0 | **4** |
| Sepang | 5 | 5 | 5 |
| Sochi | 0 | 0 | 0 |
| Yas Marina | 0 | 0 | 0 |
| Melbourne | 1 | 1 | 1 |
| Catalunya | 4 | 4 | 5 |
| Zandvoort | 4 | 3 | 4 |

## What this says

1. **The single-seed comparison was noise.** "v1 4 → v2 5 → v3 6" (the seed-0 runs) is within the
   seed spread. Across five seeds the variants do not differ in how many circuits they complete. The
   tilt narrows the spread (sd 1.7 → 0.8), which is a consistency effect, not a capability effect.
2. **Failures are structural, not random.** Norisring, Sochi and Yas Marina are completed by 0 of 15
   generalists and Melbourne by 1 of 15. Sepang is completed by 15 of 15. Something in those circuits
   defeats this training recipe whatever the seed.
3. **The tilt trades circuits.** v3 never completes Brands Hatch (0/5; v2 5/5) but completes
   Oschersleben 4/5 (v2 0/5). The circuit totals hide systematic, variant-specific effects.

## Consequences

- Variant comparisons need several seeds; the psychophysics study now carries seeds 1–2 of each
  variant as prospective agents (protocol ruling R18).
- The psychophysics crash ladders (P8) have a sharp target: the corners that defeat all 15 agents,
  and the Brands Hatch / Oschersleben swap.
- Training more variants blind is not the way to the 80% R-dev gate. Diagnose first.
