# Crash diagnostics: dr-pilot-200M (200000000 steps)

Deterministic policy, default car, rolling start, two laps; first termination per
circuit (R-dev only). 'Generator pct' is the percentile of the local corner's minimum
radius among corners of 300 generated training strips (low = tighter than most).

| Circuit | Crash | Lap | s (m) | Cause | Speed (km/h) | Corner R_min (m) | Generator pct | Width (m) |
|---|---|---|---|---|---|---|---|---|
| BrandsHatch | yes | 1 | 2750 | corridor | 204 | 33.6 | 25.9 | 8.1 |
| Hockenheim | yes | 1 | 265 | corridor | 204 | 18.3 | 7.1 | 12.0 |
| MoscowRaceway | no | - | - | - | - | - | - | - |
| Norisring | yes | 1 | 480 | corridor | 74 | 13.2 | 1.9 | 15.5 |
| Nuerburgring | no | - | - | - | - | - | - | - |
| Oschersleben | yes | 1 | 474 | corridor | 146 | 47.2 | 37.6 | 12.4 |
| Sepang | no | - | - | - | - | - | - | - |
| Sochi | yes | 1 | 2058 | corridor | 121 | 21.1 | 10.0 | 15.4 |
| YasMarina | yes | 1 | 2832 | corridor | 69 | 9.2 | 0.0 | 11.9 |
| Melbourne | yes | 1 | 377 | corridor | 138 | 19.3 | 8.3 | 13.7 |
| Catalunya | no | - | - | - | - | - | - | - |
| Zandvoort | yes | 1 | 369 | corridor | 132 | 24.4 | 14.1 | 12.0 |
