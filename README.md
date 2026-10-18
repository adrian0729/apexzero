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
uv run pytest                            # ~60 tests: physics gates, parity, estimators, ...
```

On some macOS setups the files in `.venv` get the "hidden" flag and Python then skips the
editable-install `.pth` file. Tests and scripts work around it; to fix the venv itself run
`chflags -R nohidden .venv`.

## Layout

| Path | What |
|---|---|
| `src/apexzero/vehicle/` | One vehicle ODE (`dynamics.f`) written against an ops shim and compiled for JAX (simulator) and CasADi (optimal control) |
| `src/apexzero/tracks/` | Track kit, C2 spline lookups, non-singular Frenet reference lines, corner segmentation, procedural generator, paper level sets |
| `src/apexzero/env/` | Functional, vmappable racing environment; track-bank and procedural DR adapters |
| `src/apexzero/agents/ppo.py` | PPO written from scratch (verified against rejax) |
| `src/apexzero/ocp/` | s-domain minimum-time OCP, training-objective OCP (window mode), consistency checks |
| `src/apexzero/oracle/` | QSS lap-time oracle (NumPy and JAX) |
| `src/apexzero/audit/` | Audit rollouts and the level-score estimators (per `docs/paper/score-spec.md`) |
| `src/apexzero/maze/` | Exact BFS optimum for JaxUED mazes (the paper's external check) |
| `src/apexzero/data/` | OpenF1 archiver and the position-feed snapping audit |
| `src/apexzero/learn/` | Learning exercises (1-D braking: Pontryagin vs value iteration) |
| `third_party/jaxued_examples/` | JaxUED's maze PLR example, vendored unmodified (Apache-2.0) |

## Licences

Code: Apache-2.0. Data is fetched by script and never committed: TUMFTM racetrack-database
(LGPL-3.0), OpenF1 (CC BY-NC-SA 4.0), jolpica-f1.
