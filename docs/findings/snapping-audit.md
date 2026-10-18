# Snapping audit: the public F1 position feed has no racing-line information

Every flying lap's 3.7 Hz location samples are compared with a cubic-spline curve through
the pole lap. Two tests: (1) distance from the pole lap's curve; (2) whether a driver's
*signed* lateral offset is consistent across their own laps, per 50 m station
(one-way ANOVA, intraclass correlation). Positive controls: pit-lane samples, laps
shifted by known 1-3 m offsets, and a planted 0.5 m line for one driver.

| Session | Laps | Median dist. | p95 | Pit-lane p95 | Injected 2 m -> | Driver ICC | Stations p<0.01 | Driver offset p95 | Planted ICC | Planted detected | Planted offset p95 | Excluded cars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023 Sakhir | 78 | 10.4 cm | 31 cm | 16.6 m | 2.00 m | -0.012 | 2.8% | 7.2 cm | +0.82 | 97% | 46 cm | - |
| 2023 Jeddah | 98 | 13.3 cm | 45 cm | 17.6 m | 2.00 m | +0.004 | 1.7% | 10.2 cm | +0.58 | 94% | 45 cm | 21 |
| 2023 Melbourne | 126 | 12.4 cm | 38 cm | 15.0 m | 1.99 m | -0.002 | 2.9% | 7.4 cm | +0.60 | 97% | 43 cm | - |
| 2023 Baku | 105 | 9.5 cm | 29 cm | 13.9 m | 2.01 m | -0.002 | 4.2% | 6.6 cm | +0.75 | 98% | 46 cm | - |
| 2023 Miami | 96 | 11.3 cm | 36 cm | 20.3 m | 2.01 m | -0.000 | 5.7% | 8.0 cm | +0.70 | 99% | 45 cm | - |
| 2023 Monte Carlo | 139 | 10.4 cm | 44 cm | 12.5 m | 2.01 m | +0.016 | 10.8% | 6.6 cm | +0.80 | 97% | 45 cm | - |
| 2023 Catalunya | 101 | 12.4 cm | 41 cm | 17.1 m | 1.99 m | +0.029 | 4.4% | 8.0 cm | +0.78 | 99% | 47 cm | - |
| 2023 Montreal | 103 | 9.5 cm | 24 cm | 24.7 m | 2.00 m | -0.004 | 4.7% | 5.8 cm | +0.87 | 100% | 45 cm | - |
| 2023 Spielberg | 117 | 11.3 cm | 32 cm | 15.4 m | 2.00 m | +0.020 | 7.1% | 8.1 cm | +0.60 | 98% | 46 cm | - |
| 2023 Silverstone | 109 | 12.2 cm | 38 cm | 31.1 m | 2.00 m | -0.002 | 2.6% | 40.9 cm | +0.80 | 93% | 56 cm | - |
| 2023 Hungaroring | 110 | 11.6 cm | 50 cm | 15.0 m | 2.01 m | +0.014 | 3.5% | 6.9 cm | +0.74 | 100% | 46 cm | - |
| 2023 Spa-Francorchamps | 32 | 11.8 cm | 37 cm | 30.3 m | 2.00 m | +0.005 | 2.2% | 10.5 cm | +0.84 | 94% | 46 cm | - |
| 2023 Monza | 108 | 11.9 cm | 39 cm | 18.5 m | 2.00 m | -0.014 | 0.9% | 8.0 cm | +0.65 | 98% | 46 cm | - |
| 2023 Singapore | 88 | 9.8 cm | 29 cm | 17.2 m | 2.01 m | +0.017 | 4.2% | 7.0 cm | +0.76 | 99% | 47 cm | - |
| 2023 Suzuka | 74 | 11.8 cm | 40 cm | 13.4 m | 2.00 m | +0.007 | 2.6% | 9.4 cm | +0.82 | 97% | 46 cm | - |
| 2023 Mexico City | 81 | 10.3 cm | 29 cm | 16.4 m | 2.00 m | +0.012 | 3.5% | 7.1 cm | +0.74 | 100% | 45 cm | - |
| 2023 Austin | 94 | 11.2 cm | 38 cm | 14.1 m | 2.01 m | +0.033 | 10.2% | 8.9 cm | +0.80 | 99% | 47 cm | - |
| 2023 Lusail | 112 | 12.3 cm | 40 cm | 16.8 m | 1.99 m | +0.011 | 5.6% | 7.5 cm | +0.76 | 99% | 44 cm | - |
| 2023 Interlagos | 92 | 11.2 cm | 35 cm | 27.1 m | 2.01 m | +0.005 | 3.6% | 7.8 cm | +0.63 | 96% | 45 cm | - |
| 2025 Suzuka | 93 | 13.1 cm | 43 cm | 14.1 m | 1.99 m | +0.015 | 1.7% | 9.4 cm | +0.59 | 97% | 43 cm | - |
| 2025 Monza | 99 | 12.9 cm | 39 cm | 18.8 m | 2.00 m | -0.002 | 3.5% | 9.4 cm | +0.67 | 97% | 45 cm | - |
| 2025 Silverstone | 90 | 13.0 cm | 42 cm | 31.2 m | 2.01 m | -0.005 | 1.7% | 8.9 cm | +0.57 | 97% | 45 cm | - |

![audit](snapping-audit.png)

**Reading.** Every driver sits ~13 cm from the pole lap's curve, with no driver-specific
offset (ICC ~ 0, significant stations at about the false-positive rate), while the same
test detects a planted 0.5 m line at almost every station and recovers injected offsets
exactly. The feed can represent off-line positions (the pit lane is metres away), so
the positions are projected onto a single track model. Consequence: driver comparisons
use speed, throttle, brake and time as functions of distance, never lateral lines.

*Excluded cars* had a broken coordinate stream: positions far from every other car's
everywhere on the lap, straights included (for example 2023 Jeddah car 21, ~3 m). That
is a per-car data fault, not line information, and is reported rather than hidden.

Data: OpenF1 (CC BY-NC-SA 4.0), aggregate statistics only.

**Flagged for follow-up** (more than 5% of stations significant, or
driver offsets above 20 cm at p95). Effects of a few centimetres are per-car
artefacts such as antenna position or latency, far below racing-line scale
(0.5-2 m); larger ones are open items, not conclusions:
- 2023 Miami: 5.7% of stations significant, driver offset p95 8.0 cm, ICC -0.000
- 2023 Monte Carlo: 10.8% of stations significant, driver offset p95 6.6 cm, ICC +0.016
- 2023 Spielberg: 7.1% of stations significant, driver offset p95 8.1 cm, ICC +0.020
- 2023 Silverstone: 2.6% of stations significant, driver offset p95 40.9 cm, ICC -0.002
- 2023 Austin: 10.2% of stations significant, driver offset p95 8.9 cm, ICC +0.033
- 2023 Lusail: 5.6% of stations significant, driver offset p95 7.5 cm, ICC +0.011

Skipped sessions:
- 2023 Zandvoort: only 0 drivers with >= 3 flying laps
