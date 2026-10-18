# TO-OCP spike (K2a)

Training-objective OCP in window mode on the RL environment's exact problem (RK2 x 10 at
5 ms, 20 Hz zero-order hold, corridor and slip limits tightened by 3 cm, discounted reward).
Warm start: the s-domain minimum-time optimum mapped to time. Replay: the optimal controls
applied open-loop in the float64 JAX environment, with its termination checks.

| Track | Window | Status | Iter (rounds) | Build | Solve | Peak RSS | J* | Replay | Diff | Survives |
|---|---|---|---|---|---|---|---|---|---|---|
| Hockenheim | 10 s | Solve_Succeeded | 169 (3) | 1 s | 44 s | 0.7 GB | 48.089 | 48.089 | -0.0000% | True |

Machine: 12 CPU threads (Apple M3 Pro); IPOPT + MUMPS, single process.
