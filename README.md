# ApexZero

A generalist RL racing agent trained only on procedurally generated tracks, measured against
minimum-lap-time optimal control on *identical* physics, plus an audit of curriculum-learning
level scores against exact regret, and an analysis of public F1 telemetry.

- **Plan:** [docs/plan/two-track-plan.md](docs/plan/two-track-plan.md). One paper (*Regret,
  Measured: Auditing Curriculum Level Scores Against Optimal Control*) and one demo (*How far
  from the limit, and why?*) on a shared core.
- **Where things stand:** [docs/STATUS.md](docs/STATUS.md).
- **Findings so far:** [docs/findings/](docs/findings/).

## Setup

```bash
uv sync                                  # Python 3.12, JAX (CPU), CasADi, ...
uv run python scripts/fetch_tracks.py    # TUMFTM circuit geometry (LGPL-3.0, not vendored)
uv run python scripts/fetch_openf1.py    # OpenF1 qualifying archive (CC BY-NC-SA, ~3 h)
uv run pytest                            # ~2,000 tests in 90 files: physics gates, parity, estimators, ...
```

On some macOS setups the files in `.venv` get the "hidden" flag and Python then skips the
editable-install `.pth` file. Tests and scripts work around it; to fix the venv itself run
`chflags -R nohidden .venv`.

## Layout

| Path | What |
|---|---|
| `src/apexzero/vehicle/` | One vehicle ODE (`dynamics.f`) written against an ops shim and compiled for JAX (simulator) and CasADi (optimal control) |
| `src/apexzero/tracks/` | Track kit, C2 spline lookups, non-singular Frenet reference lines, corner segmentation, procedural generator, statistical width models, paper level sets |
| `src/apexzero/env/` | Functional, vmappable racing environment; track-bank and procedural DR adapters |
| `src/apexzero/agents/` | PPO written from scratch (verified against rejax), and checkpoints |
| `src/apexzero/eval/` | Timed laps with a deterministic policy (completion and lap times) |
| `src/apexzero/ocp/` | s-domain minimum-time OCP, training-objective OCP (window mode), consistency checks, the paper's labeller |
| `src/apexzero/oracle/` | QSS lap-time oracle (NumPy and JAX) |
| `src/apexzero/audit/` | Audit rollouts and the level-score estimators (per `docs/paper/score-spec.md`) |
| `src/apexzero/ued/` | Racing curricula: PLR⊥ on JaxUED's level sampler, and SFL |
| `src/apexzero/maze/` | Exact BFS optimum for JaxUED mazes and the maze E5 audit (the paper's external check) |
| `src/apexzero/interp/` | Network psychophysics, step 1: single-corner stimuli scored against the optimum |
| `src/apexzero/fields/` | Receptive fields, step 2: how a network's line responds to the road ahead |
| `src/apexzero/costate/` | Idea 2: critic value gradients against the window problem's costate |
| `src/apexzero/data/` | OpenF1 archiver and the position-feed snapping audit |
| `src/apexzero/drivers/` | Drivers-lite: teammate time deltas per corner phase from the public feed |
| `src/apexzero/viz/` | Trajectory export, figures, the static viewer and demo rendering |
| `src/apexzero/learn/` | Learning exercises (1-D braking: Pontryagin vs value iteration) |
| `third_party/jaxued_examples/` | JaxUED's maze PLR example, vendored unmodified (Apache-2.0) |
| `third_party/sfl/` | SFL's MiniGrid maze code, vendored unmodified (Apache-2.0) |

## Licences

Code: Apache-2.0. Data is fetched by script and never committed: TUMFTM racetrack-database
(LGPL-3.0), OpenF1 (CC BY-NC-SA 4.0), jolpica-f1.
