"""Network psychophysics, step 1 (docs/interp/psychophysics-protocol.md).

Closed-loop responses of trained policies to single-corner stimuli, compared with optima
solved from the same initial state. The package is new code only: nothing else in the repo
changes, and existing functions that need changing are copied here (see
docs/interp/IMPLEMENTATION.md for the map from protocol sections to functions).

Modules, in dependency order:

    constants    fixed choices of protocol 1.4 and the grids of 3.5 (one place)
    registry     the agents (protocol 8, ruling R18's seed replicates) and their grammars
    io           paths, column tables, JSON, hashes, stage logs
    support      training-support statistics, l_typ, v_typ, regimes, R_clip, kNN density
    stimuli      corner profiles, the stimulus bank, transplants, ladders, tier D conditions
    measures     Trajectory, model tables, brake demand, response metrics, crash typing
    rollout      initial states, phases, the batched stimulus harness
    agents       loading snapshots, the manifest, real laps
    controls     scripted positive controls (T, Q, Qp, c1-c3, K)
    answer_key   pinned s-OCP, solve_from_state, every key type, validation V1-V11
    probe        tier O open-loop scans, on-policy aliasing
    analysis     pair values, decision rules P1-P8, intervals, dry run, power table

Import nothing heavy here: every module imports what it needs.
"""
