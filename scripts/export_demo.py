"""Export a policy's timed lap and the optimal lap for the demo viewer and figures.

    uv run python scripts/export_demo.py                      # Hockenheim specialist showcase
    uv run python scripts/export_demo.py --track Catalunya --agent runs/<dr-run>/ckpt/<step>.pkl \
        --name catalunya-generalist --agent-badge "POST HOC" --note "<why the badge applies>"
    uv run --group rerun python scripts/export_demo.py --rrd /tmp/hockenheim.rrd

Writes, under --out (default docs/demo):
    <name>.json                 apexzero.trajectory export (apexzero.viz.export)
    <name>.html                 the static viewer with that export embedded (opens from disk)
    <name>-map.png, <name>-speed-delta.png, <name>-sectors.png
and, with --rrd PATH, a Rerun recording (not committed: .rrd files are tied to the SDK
version that wrote them). The .json and .html carry the TUMFTM-derived track geometry
(LGPL-3.0), which the repository does not commit, so they are git-ignored under
docs/demo; only the figures, which carry the attribution in a footer, are committed.
scripts/render_demo.py turns the .json into a clip, a frame sequence or stills.

Lockbox: refuses R_TEST circuits (and their mirrored/reversed variants) unless
--unseal-lockbox is given, which protocol-v1 allows only after the unseal.
"""

import argparse
import hashlib
import json
import pickle
import resource
import subprocess
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import numpy as np

from apexzero.env.adapters import reference_speed
from apexzero.eval.laps import timed_laps
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.track import load_tumftm
from apexzero.viz import export, figures


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    """Short HEAD hash, suffixed "+dirty" when tracked files have uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="Hockenheim")
    ap.add_argument("--agent", default="runs/hockenheim-specialist-mlp-200M/agent.pkl")
    ap.add_argument("--agent-label", default="RL specialist (200M steps)")
    ap.add_argument("--agent-badge", choices=export.BADGES, default=None,
                    help="e.g. POST HOC for a checkpoint no pre-registered rule chose")
    ap.add_argument("--note", action="append", default=[],
                    help="an extra note for the export (repeatable), e.g. why a badge applies")
    ap.add_argument("--ocp", default=None, help="default: outputs/ocp/<track>.pkl")
    ap.add_argument("--out", default="docs/demo")
    ap.add_argument("--name", default="hockenheim-specialist")
    ap.add_argument("--rate", type=float, default=20.0, help="OCP ghost sample rate (Hz)")
    ap.add_argument("--v-start-frac", type=float, default=0.9,
                    help="rolling-start speed as a fraction of the QSS reference (train.py: 0.9)")
    ap.add_argument("--rrd", default=None, help="also write a Rerun recording here")
    ap.add_argument("--unseal-lockbox", action="store_true")
    args = ap.parse_args()

    unsealed = args.unseal_lockbox
    export.lockbox_guard(args.track, unsealed=unsealed)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    out = Path(args.out)
    agent_path = Path(args.agent)
    ocp_path = Path(args.ocp or f"outputs/ocp/{args.track}.pkl")

    track = refine_reference(load_tumftm(f"data/tracks/tumftm/{args.track}.csv"))
    # Again on the loaded name: "--track ./Monza" or "--track monza" resolve to a lockbox
    # circuit even though the raw string differs (the library entry points check too).
    export.lockbox_guard(track.name, unsealed=unsealed)
    with open(ocp_path, "rb") as fh:
        res = pickle.load(fh)
    # "Optimum" here is T_cont, not the certified answer key: the figures and the viewer
    # tag any optimum without the CERTIFIED badge "not certified" next to this label.
    g_ocp = export.ocp_ghost(res, track, rate_hz=args.rate, label="Optimum (T_cont)",
                             unsealed=unsealed)

    policy = export.load_policy(agent_path)
    v0 = float(reference_speed(track)[0]) * args.v_start_frac
    ro = export.rollout_policy(policy, track, v0, unsealed=unsealed)
    g_agent = export.policy_ghost(ro, track, label=args.agent_label, badge=args.agent_badge,
                                  unsealed=unsealed)

    # The export must reproduce the evaluation harness's own number for this run.
    harness = timed_laps(policy, track, v0, runs=1)
    harness_lap = float(harness.lap_time[0])
    if not np.isclose(harness_lap, g_agent.lap_time, rtol=0, atol=1e-3):
        raise SystemExit(f"export lap {g_agent.lap_time:.4f} s != harness {harness_lap:.4f} s")
    eval_run0 = None
    eval_json = agent_path.parent / "eval.json"
    if eval_json.exists():
        ev = json.loads(eval_json.read_text()).get(args.track, {})
        if ev.get("lap_times"):
            eval_run0 = float(ev["lap_times"][0])

    gap = g_agent.lap_time / g_ocp.lap_time - 1.0
    agreement = (f"It differs from a fresh one-run timed_laps call by "
                 f"{abs(harness_lap - g_agent.lap_time):.1e} s")
    if eval_run0 is not None:
        agreement += (f" and from run 0 of the training script's eval.json by "
                      f"{abs(eval_run0 - g_agent.lap_time):.1e} s")
    agreement += ("; float32 rounding in differently compiled rollouts, cause not isolated. "
                  "The script aborts if the harness differs by more than 1 ms.")
    notes = [
        f"{args.track} is a development (R-dev) circuit; no lockbox circuit is used.",
        ("The optimum is the s-domain minimum-time OCP with continuous controls (T_cont), on "
         "the same vehicle ODE, car and corridor margin. It is NOT CERTIFIED: it is not the "
         "replay-verified 20 Hz answer key T*_20 (TO-OCP lap mode is not built yet). "
         "T_cont <= T*_20, so this gap overstates the time left for a 20 Hz policy."),
        (f"Agent: deterministic mean action, rolling start at s = 0, n = 0 at "
         f"{args.v_start_frac:.0%} of the QSS reference speed, lap 2 timed (the protocol of "
         f"apexzero.eval.laps). {agreement}"),
        f"Gap to the optimum: {gap:+.2%} ({g_agent.lap_time - g_ocp.lap_time:+.3f} s).",
        ("Sectors C1, C2, ... are corner complexes from apexzero.tracks.corners.segment plus "
         "half of each neighbouring straight; they are not official turn numbers."),
        ("A sector's measured delta includes a speed deficit carried in from the corner "
         "before it. 'Charged' moves time gained or lost while both cars are flat out "
         "(throttle >= 0.99) back to the corner where that stretch began; both sum to the "
         "lap delta."),
        *args.note,
        "Track geometry: TUMFTM racetrack-database (LGPL-3.0).",
    ]
    provenance = {
        "script": "scripts/export_demo.py", "code_commit": git_commit(),
        "agent": str(agent_path), "agent_sha256": sha256(agent_path),
        "ocp": str(ocp_path), "ocp_sha256": sha256(ocp_path),
        "ocp_status": res.status, "harness_lap_time_s": harness_lap,
        "v_start_mps": v0,
    }
    if eval_run0 is not None:
        provenance["eval_json_run0_lap_time_s"] = eval_run0
    doc = export.build_document(
        track, [g_agent, g_ocp], reference="ocp",
        title=f"{args.track}: {args.agent_label} vs the optimal lap", notes=notes,
        provenance=provenance, unsealed=unsealed)

    paths = [export.write_document(doc, out / f"{args.name}.json")]
    paths.append(export.embed_in_viewer(doc, out / f"{args.name}.html"))
    paths += figures.save_figures(doc, out, args.name, footer=figures.TUMFTM_ATTRIBUTION)
    if args.rrd:
        from apexzero.viz.rerun_log import log_document

        paths.append(log_document(doc, args.rrd))

    comp = doc["comparisons"][0]
    charged = comp["sector_split"]["charged_s"]
    sp = comp["sector_speed"]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"{args.track}: agent {g_agent.lap_time:.3f} s (harness {harness_lap:.3f} s), "
          f"optimum {g_ocp.lap_time:.3f} s, gap {gap:+.2%}")
    print("  sector    measured  charged  entry dv   v min agent / optimum")
    for i, sec in enumerate(doc["track"]["sectors"]):
        print(f"  {sec['name']:8s} {comp['sector_delta_s'][i]:+8.3f} {charged[i]:+8.3f} "
              f"{sp['entry_dv_mps'][i] * 3.6:+7.1f}    {sp['vmin_mps'][i] * 3.6:4.0f} / "
              f"{sp['ref_vmin_mps'][i] * 3.6:4.0f} km/h")
    for p in paths:
        print(f"  {p}  ({p.stat().st_size / 1e3:.0f} kB)")
    print(f"wall {time.perf_counter() - wall0:.1f} s, CPU {time.process_time() - cpu0:.1f} s, "
          f"peak RSS {rss:.2f} GB")


if __name__ == "__main__":
    main()
