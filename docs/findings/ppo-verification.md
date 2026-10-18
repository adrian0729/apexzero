# PPO verification against rejax

Identical hyperparameters and seeds; deterministic evaluation, 200 episodes per
seed; mean over seeds with a 95% CI.

| Task | Ours | rejax | Pass | Ours wall-clock | rejax wall-clock |
|---|---|---|---|---|---|
| CartPole-v1 | 475.3 ± 31.9 | 327.8 ± 103.7 | yes | 8 s | 24 s |
| Pendulum-v1 | -158.5 ± 5.9 | -159.5 ± 8.3 | yes | 30 s | 42 s |
