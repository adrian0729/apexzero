"""Network psychophysics, step 1: every stage of the pre-registered protocol.

    uv run python scripts/psychophysics.py <stage> [options]

Stages (docs/interp/psychophysics-protocol.md, section 10):

    manifest        step 0: list snapshots, hashes, forward passes (no rollouts)
    support         step 1: training-support tables, l_typ, v_typ, regimes, R_clip
    stimuli         step 1: the stimulus bank and cells, with the builder's checks
    rollouts        step 2 (--pilot-replication: harness gates, the pilot replication and the
                    throughput benchmark); step 6 agent rollouts (post-tag only)
    validate        step 3 (--measures: model tables, brake demand, known-answer measures, one
                    TO-OCP solution against its replay); step 4 V1-V11
    controls        step 4/5: tune Q's c_v, fit c3's tau_c, the Section 7 controls through
                    the harness (--knn-buffer: S1's Hockenheim laps, protocol exception 2)
    keys            step 4: A2 layout, then P0 / P1 / P2 keys under the budget rule
    dryrun          step 5: dry run on the real key design, power table, hashes
    anchored-keys   step 7: re-anchored and viability keys (post-tag)
    probes          step 6: tier O scans and on-policy aliasing (post-tag)
    analyse         step 8: run and pair tables, decisions (--confirmatory: once)
    plots           step 8: curves and figures

Each stage logs to runs/psychophysics-<stage>.log (a done/total line at least once a
minute, from a heartbeat thread), keeps a one-line note in runs/activity.md while it runs,
writes under outputs/psychophysics/ and resumes from what is already there when re-run
(--force redoes it). XLA is held to 2 threads unless --threads says otherwise. Keys run in
float64 and rollouts in float32, so they never share a process: key stages solve in a pool of
--workers processes (default 4, a fixed count: the load average is not a throttle).

Order (protocol 10): manifest; support; stimuli; rollouts --pilot-replication; validate
--measures; keys --layout-only; controls (c_v); validate; keys [--priority P1|P2]; controls
--knn-buffer; controls (tau_c and the Section 7 runs); dryrun; then commit and tag. After the
tag: rollouts (finals det, then nat/mat, then checkpoints); anchored-keys; probes; analyse
[--confirmatory]; plots.

Freeze rule: no trained agent is rolled out on a protocol stimulus before the git tag
psychophysics-prereg-v2. The rollouts stage refuses to run without the tag, except
--pilot-replication (G1 and G2 on the v1 pilot mini-grid, step 2); the controls stage's
--knn-buffer runs S1's real Hockenheim laps (step 5). Those are the protocol's two exceptions.
"""

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _threads(argv: list[str]) -> int | None:
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return None


# XLA reads its flags at import, so set them before anything imports jax. An explicit
# --threads wins over an inherited XLA_FLAGS; otherwise an inherited value is kept.
_T = _threads(sys.argv)
if _T is not None or "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               f"intra_op_parallelism_threads={_T or 2}")

import _bootstrap  # noqa: F401  (makes src/ importable)

from apexzero.interp.constants import KEY_WORKERS

ROOT = Path(__file__).resolve().parents[1]
TAG = "psychophysics-prereg-v2"
STAGES = ("manifest", "support", "stimuli", "controls", "keys", "validate", "dryrun",
          "rollouts", "anchored-keys", "probes", "analyse", "plots")
DEFAULT_WORKERS = KEY_WORKERS          # key stages (protocol 10: 5 workers); rollouts 3
FROZEN_PATHS = ("src/apexzero/interp", "scripts/psychophysics.py")
PREREG_BEGIN = "<!-- prereg-record:begin -->"
PREREG_END = "<!-- prereg-record:end -->"
IMPL_DOC = "docs/interp/IMPLEMENTATION.md"


# ------------------------------------------------------------------------- helpers


def _say(log, msg: str) -> None:
    """Print and append to the stage log."""
    print(msg, flush=True)
    if log is not None:
        log.write(msg)


def _tag_exists() -> bool:
    r = subprocess.run(["git", "tag", "--list", TAG], cwd=ROOT, capture_output=True, text=True,
                       check=True)
    return r.stdout.strip() == TAG


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def _code_changed_since_tag() -> list[str]:
    """Frozen files (the harness and this script) that differ from the tagged commit, in
    the working tree or in a later commit; [] when they are the tagged code."""
    r = _git("diff", "--name-only", TAG, "--", *FROZEN_PATHS)
    changed = [ln for ln in r.stdout.splitlines() if ln.strip()]
    u = _git("ls-files", "--others", "--exclude-standard", "--", *FROZEN_PATHS)
    return changed + [f"{ln} (untracked)" for ln in u.stdout.splitlines() if ln.strip()]


def parse_prereg(text: str) -> dict | None:
    """The pre-registration record (JSON between the prereg markers) of IMPLEMENTATION.md."""
    if PREREG_BEGIN not in text or PREREG_END not in text:
        return None
    body = text.split(PREREG_BEGIN, 1)[1].split(PREREG_END, 1)[0]
    body = body.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(body) if body else None


def write_prereg(text: str, record: dict) -> str:
    """IMPLEMENTATION.md with its pre-registration record replaced (or appended)."""
    block = (f"{PREREG_BEGIN}\n```json\n{json.dumps(record, indent=1, sort_keys=True)}\n```\n"
             f"{PREREG_END}")
    if PREREG_BEGIN in text and PREREG_END in text:
        head, rest = text.split(PREREG_BEGIN, 1)
        return head + block + rest.split(PREREG_END, 1)[1]
    return text.rstrip("\n") + ("\n\n## Pre-registration record (written by `dryrun`; frozen at "
                                 f"the tag)\n\n{block}\n")


def _tagged_prereg() -> dict | None:
    r = _git("show", f"{TAG}:{IMPL_DOC}")
    return parse_prereg(r.stdout) if r.returncode == 0 else None


def prereg_problems(prereg: dict | None, code_changed: list[str], design: str,
                    deviations: list[dict]) -> list[str]:
    """Why post-tag work may not run on what is on disk: the tagged code changed; no
    pre-registration record at the tag; or the design differs from the tagged one without a
    logged design deviation naming the new hash."""
    out = []
    if code_changed:
        out.append(f"frozen code differs from the tag: {code_changed[:8]}")
    if prereg is None:
        out.append(f"no pre-registration record in {IMPL_DOC} at the tag")
        return out
    ok = {prereg.get("stimulus_hash")} | {d.get("design_hash") for d in deviations
                                          if d.get("category") == "design"}
    if design not in ok:
        out.append(f"the design hash {design[:16]} is not the tagged one "
                   f"({str(prereg.get('stimulus_hash'))[:16]}) nor a logged design deviation")
    return out


def _require_tag(what: str, design: bool = True) -> None:
    """Post-tag stages: the tag exists, the frozen code is the tagged code, and (``design``)
    the design on disk is the tagged one or a logged deviation. Records the tag in the
    manifest the first time."""
    from apexzero.interp import agents, io, stimuli

    if not _tag_exists():
        raise SystemExit(f"refusing to run {what}: the git tag {TAG} does not exist yet "
                         "(freeze rule: no agent is rolled out on a protocol stimulus before "
                         "the tag; protocol header and section 10)")
    m = _manifest()
    h = None
    if design:
        bank, cells = stimuli.load_design(io.STIMULI_DIR)
        h = stimuli.design_hash(bank, cells)
    probs = prereg_problems(_tagged_prereg(), _code_changed_since_tag(), h or "",
                            m.deviations) if design else (
        [f"frozen code differs from the tag: {c}" for c in _code_changed_since_tag()[:8]])
    if probs:
        raise SystemExit(f"refusing to run {what}: " + "; ".join(probs))
    if not m.tag:
        commit = _git("rev-list", "-n", "1", TAG).stdout.strip()
        agents.update_manifest(lambda mm: setattr(mm, "tag", f"{TAG} {commit}"))


_POST_TAG_REASON: list[str] = []      # set when a design / key stage runs after the tag


def _refuse_after_tag(what: str, args=None) -> None:
    """Stages that build or change the design or the agent-free keys run before the tag.
    After it they run only with ``--post-tag-deviation REASON``, which logs a dated
    deviation (protocol header: every change after the tag is recorded with its reason);
    a design they save is then accepted by the post-tag checks under that deviation."""
    if not _tag_exists():
        return
    reason = getattr(args, "post_tag_deviation", None) if args is not None else None
    if not reason:
        raise SystemExit(f"refusing to run {what} after the tag {TAG}: it would change the "
                         "pre-registered design or keys (pass --post-tag-deviation REASON to "
                         "log it as a deviation)")
    from apexzero.interp import agents

    rec = dict(date=time.strftime("%Y-%m-%d"), time=time.strftime("%H:%M:%S"),
               category="stage", stage=what, reason=reason)
    agents.update_manifest(lambda m: m.deviations.append(rec))
    _POST_TAG_REASON[:] = [reason]


def _log_design_deviation(h: str, why: str) -> None:
    """After the tag, a saved design is logged as a design deviation (its hash is then what
    the post-tag checks accept)."""
    if not _POST_TAG_REASON:
        return
    from apexzero.interp import agents

    rec = dict(date=time.strftime("%Y-%m-%d"), time=time.strftime("%H:%M:%S"),
               category="design", design_hash=h, why=why, reason=_POST_TAG_REASON[0])
    agents.update_manifest(lambda m: m.deviations.append(rec))


def _require_steps(what: str, steps: tuple[str, ...] = ("step2", "step3")) -> None:
    """Protocol 10: a failed step 2 (harness gates, pilot replication) or step 3 (measures)
    blocks everything after it."""
    m = _manifest()
    bad = [s for s in steps if not (m.validation.get(s) or {}).get("passed")]
    if bad:
        raise SystemExit(f"refusing to run {what}: {', '.join(bad)} has not passed "
                         "(protocol 10: it blocks everything after it)")


def _manifest():
    from apexzero.interp import agents, io

    if not io.MANIFEST_PATH.exists():
        raise SystemExit("no manifest: run `psychophysics.py manifest` first")
    return agents.load_manifest(io.MANIFEST_PATH)


def _finals(m) -> dict:
    """agent -> final SnapshotEntry (loaded without error) of every finished run."""
    return {e.agent: e for e in m.snapshots if e.is_final and not e.error}


def _load(entry):
    from apexzero.interp import agents

    return agents.load_agent(ROOT / entry.path, entry.agent)


def _design(laid_out: bool = False):
    """The saved stimulus design (bank, cells). ``laid_out`` requires the A2 layout."""
    from apexzero.interp import io, stimuli

    if not (io.STIMULI_DIR / "cells.json").exists():
        raise SystemExit("no stimulus design: run `psychophysics.py stimuli` first")
    bank, cells = stimuli.load_design(io.STIMULI_DIR)
    if laid_out and any(c.D is None for c in cells):
        raise SystemExit("A2 cells have no D yet: run `psychophysics.py keys --layout-only`")
    return bank, cells


def _jsonable(x):
    from apexzero.interp import io

    return io._jsonable(x)


def _cpu() -> float:
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


# ------------------------------------------------------------------------- step 0


def stage_manifest(args: argparse.Namespace) -> None:
    """Step 0 (agents.freeze_manifest): every finished run's snapshots with sha256, size,
    timesteps, exp(log_std) and a finite forward pass on a dummy observation; prospective
    snapshots recorded as not evaluated; runs still training listed as pending. Re-running
    adds new snapshots and never changes an existing entry. Ruling R18's replicates (G1s1 ..
    G3s2) are listed like every agent (finished when eval_rdev.json exists; prospective; not
    evaluated), each with its variant, seed and seed-0 agent."""
    from apexzero.interp import agents, io

    specs = agents.AGENTS
    with io.StageLog("manifest", total=len(specs)) as log:
        m = agents.load_manifest(io.MANIFEST_PATH) if io.MANIFEST_PATH.exists() else None
        for aid, spec in specs.items():
            m = agents.freeze_manifest(io.MANIFEST_PATH, {aid: spec}, existing=m)
            ents = [e for e in m.snapshots if e.agent == aid]
            rec = m.agents[aid]
            log.step(note=f"{aid}: {rec['status']}, {len(ents)} snapshots")
        lines = []
        for aid in specs:
            ents = [e for e in m.snapshots if e.agent == aid]
            rec = m.agents[aid]
            fin = [e for e in ents if e.is_final]
            bad = [e.path for e in ents if e.error or not e.forward_pass_ok]
            pro = [e for e in ents if e.prospective]
            ev = [e.path for e in pro if e.evaluated_on_stimuli]
            std = fin[0].std if fin else (float("nan"), float("nan"))
            who = (f"{aid} (R18 replicate of {agents.seed0_of(aid)}: variant "
                   f"{agents.variant_of(aid)}, seed {agents.seed_of(aid)})"
                   if agents.is_replicate(aid) else aid)
            lines.append(f"{who}: {rec['status']}; {len(ents)} snapshots, final "
                         f"{fin[0].path if fin else '-'} ({fin[0].timesteps if fin else '-'}), "
                         f"std (steer, pedal) = ({std[0]:.4f}, {std[1]:.4f}); forward pass "
                         f"finite {len(ents) - len(bad)}/{len(ents)}; prospective {len(pro)}, "
                         f"evaluated on stimuli {len(ev)}")
            if bad:
                lines.append(f"  FAILED to load or non-finite: {bad}")
            if ev:
                lines.append(f"  FREEZE RULE: prospective snapshots evaluated: {ev}")
        for ln in lines:
            _say(log, ln)
        ok = all(not e.error and e.forward_pass_ok for e in m.snapshots)
        reps = agents.replicates()
        rep_ents = [e for e in m.snapshots if e.agent in reps]
        _say(log, f"R18 replicates: {sum(m.agents[a]['finished'] for a in reps if a in m.agents)}"
                  f"/{len(reps)} finished, {len(rep_ents)} snapshots, prospective "
                  f"{sum(e.prospective for e in rep_ents)}, evaluated on stimuli "
                  f"{sum(e.evaluated_on_stimuli for e in rep_ents)}")
        _say(log, f"manifest: {len(m.snapshots)} snapshots, all load with a finite forward pass: "
                  f"{ok}; written to {io.MANIFEST_PATH}")


# ------------------------------------------------------------------------- step 1


def stage_support(args: argparse.Namespace) -> None:
    """Step 1 (support.build_support): corner tables of 300 strips per grammar and R_DEV,
    l_typ, v_typ, regimes (q1 / q10 of R_seg) and R_clip from every final in the manifest
    (reading obs_rms is not an evaluation). Resumes when support.json already holds R_clip for
    every available final; a rebuild for a newly finished final must reproduce the frozen
    corner tables exactly (same seed stream), or it stops."""
    from apexzero.interp import agents, io, support

    m = _manifest()
    finals = _finals(m)
    path = io.SUPPORT_DIR / "support.json"
    with io.StageLog("support", total=6) as log:
        old = support.load_support(path) if path.exists() else None
        have = {a for a, r in (old.regimes if old else {}).items() if r.r_clip}
        if old is not None and not args.force and set(finals) <= have:
            st = old
            log.step(6, note="resumed: support.json covers every final")
        else:
            obs = {a: (e.path, _load(e).obs_rms) for a, e in sorted(finals.items())}
            st = support.build_support(io.SUPPORT_DIR, obs, log=log)
            if old is not None and old.hashes != st.hashes:
                raise SystemExit("the rebuilt corner tables differ from the frozen ones: "
                                 f"{old.hashes} vs {st.hashes}")
            log.step(max(0, 6 - log.done))
        regimes = {}
        for a, r in sorted(st.regimes.items()):
            regimes[a] = dict(q1=r.q1, q10=r.q10, r_clip_max=r.r_clip_max, snapshot=r.snapshot,
                              grammar=r.grammar)
            _say(log, f"{a}: q1 / q10 of R_seg = {r.q1:.2f} / {r.q10:.2f} m, R_clip^max = "
                      f"{r.r_clip_max:.2f} m ({r.snapshot or 'no final yet'})")
        lt, vt = st.l_typ, st.v_typ
        _say(log, "l_typ at R " + ", ".join(f"{r:g}" for r in lt.r_nodes) + " m: "
             + ", ".join(f"{v:.1f}" for v in lt.l_nodes) + f" m (corners turning >= "
             f"{lt.min_theta_deg:g} deg; every corner: "
             + ", ".join(f"{v:.1f}" for v in lt.l_nodes_all) + ")")
        _say(log, "v_typ (km/h) per bin and theta class: " + json.dumps(vt.choice_kph))
        _say(log, "tight-and-fast corners per 100 km: " + json.dumps(
            {k: round(v.get("tight_fast_r12", float("nan")), 2) for k, v in st.rates.items()}))
        sup_rec = dict(path=str(path.relative_to(ROOT)), sha256=io.sha256_file(path),
                       table_hashes=st.hashes, regimes=regimes,
                       l_typ=dict(r=list(lt.r_nodes), l=list(lt.l_nodes),
                                  min_theta_deg=lt.min_theta_deg),
                       v_typ_kph=vt.choice_kph, frozen=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       pending=[a for a in agents.AGENTS if a not in finals])
        agents.update_manifest(lambda m: setattr(m, "support", sup_rec))


def _standin_states() -> tuple[dict, dict]:
    """Stand-in real states for a geometry-only build of tiers D, E1 and E2 (step 1's
    transplant and twin checks): each requested state placed exactly at its target s on the
    circuit's centreline at 50 m/s, with no agent involved. The strips depend on a state only
    through the lead-in start (one knot either way), so the transplant kappa and the twins are
    the ones the real build will make. G2's reach per circuit comes from its documented lap-1
    crash positions (CRASH_PAIRS); the real E2 selection uses G2's real lap. Never saved as
    the design."""
    from apexzero.interp import stimuli

    reach = {p.track: p.s_crash for p in stimuli.CRASH_PAIRS if p.agent == "G2"}
    states: dict = {}
    for r in stimuli.real_state_requests(reach):
        tr = stimuli.real_track(r["track"])
        s = float(r["s_target"]) % tr.length if r["s_target"] >= 0 else 0.0
        st = stimuli.RealState(agent=r["agent"], track=r["track"], lap=r["lap"], s_real=s,
                               x=(s, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0),
                               prev_action=(0.0, 1.0), accel=(0.0, 0.0))
        states.setdefault(r["agent"], {})[r["key"]] = st
    return states, reach


def stage_stimuli(args: argparse.Namespace) -> None:
    """Step 1 (stimuli.build_design, check_design, save_design, cell_support_table). The
    real-road tiers need real-lap states (``rollouts --real-laps``, ruling R1): without them
    the D / E1 / E2 rows are built from stand-in states for the transplant and twin checks
    only. With --real, ``outputs/psychophysics/stimuli/real_states.json`` is appended to the
    design. Resumes from a saved design unless --force. The theta_eff misses of ruling R3 and
    the twin length shortfalls of R4 pass and are listed."""
    from apexzero.interp import agents, io, stimuli, support

    _refuse_after_tag("the stimuli stage", args)
    sup = support.load_support(io.SUPPORT_DIR)
    out = io.STIMULI_DIR
    with io.StageLog("stimuli", total=5) as log:
        if (out / "cells.json").exists() and not args.force:
            bank, cells = stimuli.load_design(out)
            log.step(note=f"resumed: {len(bank)} rows, {len(cells)} cells")
            if args.real and not any({"D", "E1", "E2"} & set(c.tiers) for c in cells):
                # Real tiers appended to the saved (possibly laid-out) design; the keys
                # stage lays out the new cells (W is None) on its next run.
                bank, cells = _append_real(bank, cells, args.f6_scale)
                _say(log, f"real tiers appended: {len(bank)} rows, {len(cells)} cells")
        else:
            bank, cells = stimuli.build_design(sup)
            if args.real:
                bank, cells = _append_real(bank, cells, args.f6_scale)
            log.step(note=f"built: {len(bank)} rows, {len(cells)} cells")
        det = stimuli.check_design_details(bank, cells)
        log.step(note="design checks")
        checks = {k: dict(v) for k, v in det.items()}
        if any({"D", "E1", "E2"} & set(c.tiers) for c in cells):
            # The real rows are in the design, so the checks above already cover them; a
            # stand-in build on top would duplicate every real stimulus.
            n_real, reach = 0, None
            log.step(note="real geometry checked in the design itself")
        else:
            states, reach = _standin_states()
            rb, rc = stimuli.build_real_design(bank, cells, states)
            real_det = stimuli.check_design_details(rb, rc)
            n_real = len(rb) - len(bank)
            log.step(note=f"stand-in real geometry: {n_real} rows")
            for k in ("transplant_source", "twin_fit", "inside_cap", "track_arrays_roundtrip",
                      "mirror_exact", "lengths", "theta_eff"):
                checks[f"real_standin/{k}"] = real_det[k]
        h = stimuli.save_design(out, bank, cells)
        io.write_json(out / "checks.json", dict(checks=checks, design_hash=h,
                                                 standin_real_rows=n_real,
                                                 standin_reach=reach))
        log.step(note="saved")
        cs = stimuli.cell_support_table(bank, cells, sup)
        io.save_table(out / "cell_support.npz", cs)
        log.step(note="cell support table")
        tiers: dict[str, int] = {}
        for s in bank.stimuli:
            for t in s.tiers:
                tiers[t] = tiers.get(t, 0) + 1
        _say(log, f"design: {len(bank)} rows, {len(cells)} cells, rows per tier {tiers}; "
                  f"hash {h[:16]}")
        for k, v in checks.items():
            _say(log, check_line(k, v))
        agents.update_manifest(lambda m: setattr(m, "stimulus_hash", h))
        _log_design_deviation(h, "stimuli stage after the tag")


def check_line(name: str, v: dict) -> str:
    """One step-1 check for the log. Rows accepted by a ruling (R3: theta_eff misses from
    knot sampling of the trapezoid's kinks, within their bound; R4: length shortfalls of
    twins matching R_seg and the angle to < 1e-3) pass and are listed as
    'accepted per R3/R4'."""
    skip = ("ok", "fails", "mismatches", "accepted", "accepted_by", "length_shortfall")
    num = {kk: vv for kk, vv in v.items() if kk not in skip}
    nf = len(v.get("fails", []) or [])
    acc = v.get("accepted", []) or []
    verdict = "N/A (no rows)" if v.get("vacuous") else ("PASS" if v["ok"] else "FAIL")
    out = f"check {name}: {verdict} {_jsonable(num)}" + (f" ({nf} failing rows)" if nf else "")
    if acc:
        out += (f"; {len(acc)} rows accepted per {v.get('accepted_by') or 'R3/R4'} (listed): "
                + ", ".join(str(r.get("name")) for r in acc))
    return out


def _append_real(bank, cells, f6_scale: float = 1.0):
    """Tiers D, E1 and E2 from the Integrate phase's real-lap states (real_states.json:
    {agent: {key: RealState fields}}); F5 swap A partners from the tier D s-OCP keys when the
    keys stage has written them (swap.json); F6 amplitudes scaled by ``f6_scale`` (0.5 after
    the K control landed in class 1, protocol 7)."""
    from apexzero.interp import io, stimuli

    p = io.STIMULI_DIR / "real_states.json"
    if not p.exists():
        raise SystemExit("--real needs outputs/psychophysics/stimuli/real_states.json (real-lap "
                         "states: run `psychophysics.py rollouts --real-laps` first, ruling R1)")
    raw = io.read_json(p)
    states = {a: {k: stimuli.RealState(**v) for k, v in d.items()} for a, d in raw.items()}
    sw = io.STIMULI_DIR / "swap.json"
    swap = {int(k): int(v) for k, v in io.read_json(sw).items()} if sw.exists() else None
    return stimuli.build_real_design(bank, cells, states, swap, f6_scale)


# ------------------------------------------------------------------------- step 2


def _random_actor(seed: int = 0, cfg=None):
    """A randomly initialised actor with the agents' architecture (never a trained one):
    ``cfg`` is a final's PPOConfig (its network, widths, layers and activation; only the
    config is read), by default G1's final. PPOConfig() alone is the SimBa network, which is
    not what the agents run."""
    import jax
    import jax.numpy as jnp

    from apexzero.agents.ppo import Actor
    from apexzero.env import racing

    if cfg is None:
        cfg = _load(_finals(_manifest())["G1"]).cfg
    actor = Actor(cfg, racing.ACTION_DIM, False)
    params = actor.init(jax.random.PRNGKey(seed), jnp.zeros(racing.OBS_DIM, jnp.float32))
    return cfg, params


def _harness_gates(log, bank, cells, finals) -> dict:
    """Step 2's harness gates, with a randomly initialised actor (obs_rms of G1's final, only
    read) and the scripted lane keeper: step equivalence, batch = single, mirror gate with
    mirrored noise, common random numbers, x64 off and the context checks."""
    import jax

    from apexzero.interp import io, rollout

    res: dict = {}
    # The gate: this (rollout) process starts in float32, and the harness refuses float64.
    at_start = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        io.assert_x64_off()
        refuses = False
    except RuntimeError:
        refuses = True
    finally:
        jax.config.update("jax_enable_x64", False)
    io.assert_x64_off()
    res["x64_off"] = dict(ok=not at_start and refuses, x64_at_start=at_start,
                          harness_refuses_x64=refuses)
    loaded = {a: _load(e) for a, e in sorted(finals.items())}
    rms = {a: ag.obs_rms for a, ag in loaded.items()}
    ctx = rollout.check_context(rms)
    res["context"] = dict(ok=True, max_abs_normalised_context=ctx)
    log.step(note="x64 off, context")
    cfg, actor = _random_actor(cfg=loaded["G1"].cfg)
    res["architecture"] = dict(ok=True, network=getattr(cfg, "network", ""),
                               width=getattr(cfg, "actor_width", None),
                               layers=getattr(cfg, "mlp_layers", None),
                               activation=getattr(cfg, "activation", ""))
    pol = rollout.agent_policy(cfg)
    g1 = rms.get("G1")

    def pair_of(R: float, approach: str) -> tuple:
        """The (left, right) cells of the tier A R/90 pair at an approach, given a window,
        a phase cycle (v0 dt) and, at A2, D = 200 m (the layout sets them later)."""
        out = []
        for c in cells:
            st = bank.stimuli[c.stim_id]
            if "A" in c.tiers and c.approach == approach and st.spec is not None \
                    and st.spec.R == R and st.spec.theta_deg == 90.0 and not st.spec.no_cap:
                out.append(dataclasses.replace(c, W=400, P=round(c.v0 * 0.05, 3),
                                               D=c.D if c.D is not None else 200.0))
        out.sort(key=lambda c: -bank.stimuli[c.stim_id].direction)
        if len(out) != 2 or out[0].pair_id != out[1].pair_id:
            raise RuntimeError(f"no tier A R{R}/90 {approach} pair in the design")
        return tuple(out)

    a1 = pair_of(22.9, "A1")
    a2 = pair_of(44.7, "A2")
    keeper = (np.float32(60.0), np.full(2, -np.inf, np.float32))
    d1 = rollout.check_step_loop(pol, actor, g1, bank, a1[0], steps=400)
    d1r = rollout.check_step_loop(pol, actor, g1, bank, a1[1], steps=400, phase=2)
    d2 = rollout.check_step_loop(rollout._lane_keeper, keeper, None, bank, a2[0], steps=400)
    res["step_loop"] = dict(ok=max(d1, d1r, d2) <= 1e-6, random_actor_rel=max(d1, d1r),
                            lane_keeper_rel=d2,
                            note="max |dx| / max(1, |x|) over the run (decision R6)")
    log.step(note="step loop")
    test_cells = list(a1) + list(a2)
    d3 = rollout.check_batch_equals_single(pol, actor, g1, bank, test_cells, steps=200)
    d4 = rollout.check_batch_equals_single(pol, actor, g1, bank, test_cells, steps=200,
                                           mode="nat", samples=3)
    d5 = rollout.check_batch_equals_single(rollout._lane_keeper, keeper, None, bank, list(a2),
                                           steps=400, isolated=False)
    res["batch_equals_single"] = dict(ok=d3 == 0.0 and d4 == 0.0 and d5 <= 1e-5,
                                      isolated_det=d3, isolated_native=d4,
                                      batch_of_one_keeper_rel=d5,
                                      note="isolated: bit for bit among decoys (decision R7)")
    log.step(note="batch = single")
    g_a2 = rollout.mirror_gate(bank, a2, steps=400)
    g_a1 = rollout.mirror_gate(bank, a1, steps=400)
    res["mirror_gate"] = dict(ok=max(g_a1, g_a2) <= 1e-5, max_abs_a2=g_a2, max_abs_a1=g_a1,
                              modes=["det", "mat (mirrored noise, CRN)"], steps=400)
    log.step(note="mirror gate")
    crn = rollout.check_common_noise(bank, list(a1) + list(a2), steps=40, samples=2)
    res["common_random_numbers"] = dict(ok=bool(crn))
    log.step(note="common random numbers")
    return res


def _pilot(log) -> dict:
    """The pilot replication (protocol 1.1, step 2): G1 and G2 finals on the v1 pilot
    mini-grid in the pilot's own layout. One of the protocol's two allowed pre-tag agent
    rollouts; recorded in the manifest."""
    from apexzero.interp import agents, rollout, stimuli

    out = {}
    bank, _ = stimuli.pilot_minigrid()
    # The pilot's own construction (scratch minigrid.py / stimkit.corner_kappa): peak 1/R,
    # 10 m linear ramps, plateau theta R - 10, entry 800 m on 1280 knots, width 5.5 / 5.5.
    grid = np.arange(bank.kappa.shape[1]) * 2.0
    worst = 0.0
    for st in bank.stimuli:
        sp = st.spec
        u = grid - 800.0
        lc = np.radians(sp.theta_deg) * sp.R - 10.0
        prof = np.where(u < 0, 0.0, np.where(u < 10.0, u / 10.0, np.where(
            u < 10.0 + lc, 1.0, np.clip((20.0 + lc - u) / 10.0, 0.0, 1.0))))
        worst = max(worst, float(np.max(np.abs(bank.kappa[st.stim_id] - sp.direction / sp.R * prof))),
                    float(np.max(np.abs(bank.w_left[st.stim_id] - 5.5))),
                    float(np.max(np.abs(bank.w_right[st.stim_id] - 5.5))))
    out["construction_max_abs_vs_pilot"] = worst
    _say(log, f"pilot mini-grid: {len(bank)} rows; kappa and widths vs the pilot's own "
              f"construction: max |diff| {worst:.2e}")
    recs = []
    for aid in rollout.PILOT_EXPECTED:          # G1 and G2, the pilot's agents
        res = rollout.pilot_replication(aid)
        match = res.pop("match")
        crashed = {k: v for k, v in res.items() if v[0]}
        out[aid] = dict(match=match, cells=len(res), completed=len(res) - len(crashed),
                        snapshot=str(agents.final_snapshot(agents.AGENTS[aid]).relative_to(ROOT)),
                        crashed={k: dict(cause=v[1], v_entry_kph=v[2] * 3.6, s_crash=v[3],
                                         s_crash_rel_entry=v[3] - 800.0, v_crash_kph=v[4] * 3.6)
                                 for k, v in crashed.items()},
                        v_entry_kph={k: round(v[2] * 3.6, 2) for k, v in res.items()},
                        v_min_kph={k: round(v[5] * 3.6, 2) for k, v in res.items()})
        for k, v in crashed.items():
            _say(log, f"  {aid} crash {k}: {v[1]} at s = {v[3]:.1f} m (entry + {v[3] - 800:.1f} m), "
                      f"{v[4] * 3.6:.1f} km/h; entry speed {v[2] * 3.6:.1f} km/h")
        _say(log, f"pilot {aid}: {len(res) - len(crashed)}/{len(res)} completed; crashed "
                  f"{sorted(crashed)} -> {'MATCHES' if match else 'DOES NOT MATCH'} the pilot")
        path = agents.final_snapshot(agents.AGENTS[aid])
        rec = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), agent=aid,
                   snapshot=str(path.relative_to(ROOT)), stimuli="v1 pilot mini-grid (24 cells)",
                   why="step 2 pilot replication (protocol exception 1)")
        recs.append(rec)
        log.step(note=f"pilot {aid}")
    agents.update_manifest(lambda m: m.lists.setdefault("pre_tag_agent_rollouts", []).extend(recs))
    out["minigrid"] = dict(rows=len(bank), names=[s.name for s in bank.stimuli])
    return out


def _benchmark(log) -> dict:
    """Throughput at B = 2048, T = 800 with 2 XLA threads (random actor with the agents'
    architecture and G1's obs_rms; full and stochastic records)."""
    from apexzero.interp import rollout, stimuli
    from apexzero.interp.constants import MIN_ROLLOUT_SPS

    bank, _ = _design()
    m = _manifest()
    g1 = _load(_finals(m)["G1"])
    cfg, actor = _random_actor(1, cfg=g1.cfg)
    rms = g1.obs_rms
    pol = rollout.agent_policy(cfg)
    sub = stimuli.StimulusBank(bank.kappa[:64], bank.w_left[:64], bank.w_right[:64],
                               bank.stimuli[:64], bank.ds)
    full = rollout.benchmark(pol, actor, rms, sub, 2048, 800, "full")
    stoch = rollout.benchmark(pol, actor, rms, sub, 2048, 800, "stoch")
    log.step(note="benchmark")
    return dict(full_sps=full, stoch_sps=stoch, threads=os.environ.get("XLA_FLAGS", ""),
                network=getattr(cfg, "network", ""),
                rule3_fires=bool(min(full, stoch) < MIN_ROLLOUT_SPS))


def stage_rollouts(args: argparse.Namespace) -> None:
    """Step 2 with --pilot-replication (harness gates, then the pilot replication of G1 and
    G2, then the throughput benchmark; a failed gate or a pilot mismatch blocks everything
    after step 2). With --real-laps, the standard R_DEV evaluation that gives the real-lap
    states, the fidelity-gate traces and G3's crash corners: allowed before the tag (ruling
    R1), on the frozen code after it. Otherwise step 6 (post-tag only): agent rollouts per
    snapshot, and --yasmarina."""
    from apexzero.interp import io, rollout

    if args.pilot_replication:
        bank, cells = _design()
        finals = _finals(_manifest())
        out = io.OUT / "harness"
        with io.StageLog("rollouts", total=8, note="step 2: harness gates + pilot replication") as log:
            gates = _harness_gates(log, bank, cells, finals)
            for k, v in gates.items():
                _say(log, f"gate {k}: {'PASS' if v['ok'] else 'FAIL'} {_jsonable(v)}")
            ok = all(v["ok"] for v in gates.values())
            pilot = _pilot(log) if ok else {}
            bench = _benchmark(log)
            _say(log, f"throughput B=2048 T=800: full {bench['full_sps']:,.0f} steps/s, stoch "
                      f"{bench['stoch_sps']:,.0f} steps/s (rule 3 fires: {bench['rule3_fires']})")
            verdict = ok and bool(pilot) and all(pilot[a]["match"] for a in rollout.PILOT_EXPECTED)
            io.write_json(out / "step2.json", dict(gates=gates, pilot=pilot, benchmark=bench,
                                                    passed=verdict,
                                                    created=time.strftime("%Y-%m-%dT%H:%M:%S")))
            _say(log, f"step 2 {'PASSED' if verdict else 'FAILED: everything after step 2 is blocked'}")
            from apexzero.interp import agents

            def upd(m):
                m.validation["step2"] = dict(passed=verdict,
                                             file=str((out / "step2.json").relative_to(ROOT)))
                m.budget["rollout_sps"] = dict(full=bench["full_sps"], stoch=bench["stoch_sps"])

            agents.update_manifest(upd)
        return
    if args.real_laps:
        before_tag = real_laps_gate()
        _real_laps(args, before_tag)
        return
    _require_tag("agent rollouts")
    if args.yasmarina:
        _yasmarina(args)
        return
    _stage_rollouts_post_tag(args)


# ------------------------------------------------------------------------- step 3


def _brake_demand_check(n: int = 4000, seed: int = 0) -> dict:
    """measures.front_brake_demand, rescaled by tyre_forces' friction-ellipse factor, equals
    ``tyre_forces(x).fx_f`` to 1e-10 on random states (float64 NumPy), and its JAX float32
    version matches the NumPy one (the harness records beta_f with it)."""
    import jax
    import jax.numpy as jnp

    from apexzero.interp import measures
    from apexzero.vehicle.dynamics import tyre_forces
    from apexzero.vehicle.ops import JAX
    from apexzero.vehicle.params import F1_2025

    rng = np.random.default_rng(seed)
    x = np.zeros((n, 8))
    x[:, 0] = rng.uniform(0.0, 5000.0, n)
    x[:, 1] = rng.uniform(-5.0, 5.0, n)
    x[:, 2] = rng.uniform(-0.3, 0.3, n)
    x[:, 3] = rng.uniform(-1.0, 90.0, n)
    x[:, 4] = rng.uniform(-4.0, 4.0, n)
    x[:, 5] = rng.uniform(-1.5, 1.5, n)
    x[:, 6] = rng.uniform(-0.45, 0.45, n)
    x[:, 7] = rng.uniform(-1.0, 1.0, n)
    beta, fx_dem, _ = measures.front_brake_demand(x, F1_2025, measures.NP)
    fo = tyre_forces(x, F1_2025, measures.NP)
    scale = 1.0 / measures.NP.smax(fo.util_f, 1.0, 5e-3)
    d = float(np.max(np.abs(fx_dem * scale - fo.fx_f)))
    bj = jax.jit(jax.vmap(lambda xi: measures.front_brake_demand(xi, F1_2025, JAX)[0]))(
        jnp.asarray(x, jnp.float32))
    dj = float(np.max(np.abs(np.asarray(bj, float) - beta)))
    return dict(ok=d <= 1e-10 and bool(np.all(beta >= 0)), n=n, max_abs_fx=d,
                max_abs_beta_f32_vs_f64=dj)


def stage_validate(args: argparse.Namespace) -> None:
    """Step 3 with --measures; otherwise step 4's V1-V11 (see _validate_keys)."""
    if args.measures:
        _validate_measures(args)
        return
    _validate_keys(args)


def _smoke_cell(bank, cells):
    """The validation cell R40/90 left at A2, with D = 200 m and a 200-step window: the same
    validation-sized problem the answer-key tests solve, here on the real design's strip."""
    for c in cells:
        st = bank.stimuli[c.stim_id]
        if "V" in c.tiers and c.approach == "A2" and st.spec is not None and st.spec.R == 40.0 \
                and st.spec.theta_deg == 90.0 and st.direction == 1:
            return dataclasses.replace(c, D=200.0, W=200)
    raise RuntimeError("no V-tier R40/90 A2 cell in the design")


def _validate_measures(args) -> None:
    """Step 3: the model tables (built once, cached), the brake-demand re-implementation
    against ``tyre_forces``, the known-answer measures (synthetic trajectories: brake-demand
    onset, blips, catch-trial onset, censoring after a crash, eta on both sides, crash typing:
    tests/test_interp_measures.py, run as is), and one TO-OCP solution (a validation-sized
    nominal key, nothing cached) against its float64 environment replay: identical measures
    to 1e-6. Agent-free."""
    from apexzero.interp import agents, answer_key, io, measures
    from apexzero.ocp.window import WindowResult

    out = io.OUT / "harness" / "step3.json"
    res: dict = {}
    with io.StageLog("validate", total=4, note="step 3: measures") as log:
        c0 = _cpu()
        tables = measures.model_tables()
        # The cached tables must be what a fresh build gives (they are part of the design).
        fresh = measures.build_model_tables()
        rebuild = max(float(np.max(np.abs(np.asarray(getattr(fresh, f), float)
                                          - np.asarray(getattr(tables, f), float))))
                      for f in ("v_dec", "a_dec", "pedal_dec", "v_lat", "a_y_max"))
        res["model_tables"] = dict(
            ok=rebuild == 0.0, rebuild_max_abs_diff=rebuild,
            path=str(measures.tables_path().relative_to(ROOT)),
            a_dec_310=float(tables.a_dec_at(310 / 3.6)), a_dec_200=float(tables.a_dec_at(200 / 3.6)),
            a_y_max={f"{v:g}": float(tables.a_y_at(v / 3.6)) for v in (40, 100, 200, 265, 300)},
            limits=sorted({str(x) for x in tables.limit_y}) if tables.limit_y is not None else [],
            v_ss_kph={f"{r:g}": float(tables.v_ss(r)) * 3.6 for r in (13.0, 40.0, 109.0, 300.0)},
            cpu_s=_cpu() - c0)
        log.step(note="model tables")
        res["brake_demand"] = _brake_demand_check()
        log.step(note="brake demand")
        c0 = time.time()
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            str(ROOT / "tests" / "test_interp_measures.py")], cwd=ROOT,
                           capture_output=True, text=True, check=False)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-400:]
        res["known_answer_tests"] = dict(ok=r.returncode == 0, summary=tail,
                                         wall_s=time.time() - c0)
        log.step(note="known-answer measures")
        bank, cells = _design()
        cell = _smoke_cell(bank, cells)
        answer_key._x64()
        c0 = _cpu()
        rec = answer_key.full_recipe(bank, cell, tables=tables, W=cell.W, cache=None)
        solve_cpu = _cpu() - c0
        st = bank.stimuli[cell.stim_id]
        wr = WindowResult(np.asarray(rec.x), np.asarray(rec.a), float(rec.J), True, rec.status,
                          0, 0.0, 0)
        marks = measures.make_marks(st, bank.w_left[cell.stim_id], bank.w_right[cell.stim_id],
                                    float(rec.x[0][0]), int(rec.W))
        chk = measures.check_window_replay(wr, bank.track(cell.stim_id), marks,
                                           ctx=answer_key.key_ctx(), tables=tables)
        res["replay"] = dict(
            ok=bool(chk["pass"]) and rec.status == "verified", cell=st.name,
            v0_kph=cell.v0 * 3.6, D=cell.D, W=int(rec.W), status=rec.status, J=float(rec.J),
            starts=[(s.name, s.verified, round(s.J, 6) if np.isfinite(s.J) else None,
                     s.iterations, round(s.cpu_seconds, 1)) for s in rec.starts],
            measures={k: rec.measures.get(k) for k in ("has_brake", "d_b", "v_b", "v_e", "v_min")},
            J_key_vs_metrics=abs(float(rec.J) - float(rec.measures.get("J", np.nan))),
            solve_cpu_s=solve_cpu, **{k: _jsonable(v) for k, v in chk.items()})
        log.step(note="TO-OCP vs replay")
        res["ok"] = all(v.get("ok", True) for v in res.values() if isinstance(v, dict))
        for k, v in res.items():
            _say(log, f"{k}: {_jsonable(v)}")
        io.write_json(out, res)
        agents.update_manifest(lambda m: m.validation.__setitem__(
            "step3", dict(passed=res["ok"], file=str(out.relative_to(ROOT)))))


# ------------------------------------------------------------------------- key worker pool

_W: dict = {}


def _key_worker_init() -> None:
    """Each key worker process: float64 on, the saved bank, the model tables and the key cache
    (records on disk are shared; every write is atomic and per key)."""
    from apexzero.interp import answer_key, io, measures, stimuli

    answer_key._x64()
    bank, _ = stimuli.load_design(io.STIMULI_DIR)
    _W.update(bank=bank, tables=measures.model_tables(), cache=answer_key.KeyCache())


def ak_default_cv() -> float:
    from apexzero.interp.answer_key import Q_CV_DEFAULT

    return Q_CV_DEFAULT


def _tuned_cv() -> float | None:
    """Q's tuned c_v from the controls stage (controls/tuning.json), if it has run."""
    from apexzero.interp import io

    p = io.CONTROLS_DIR / "tuning.json"
    if not p.exists():
        return None
    v = io.read_json(p).get("c_v")
    return float(v) if v is not None else None


def _rec_summary(rec) -> dict:
    from apexzero.interp import answer_key as ak

    if rec is None:
        return {"status": "missing"}
    if isinstance(rec, ak.SocpKey):
        return dict(kind="socp", key_id=rec.key_id, cell=rec.cell_id, phase=rec.phase,
                    status="verified" if rec.result.success else "failed",
                    has_brake=bool(rec.measures.get("has_brake", False)),
                    v_min=_jsonable(rec.measures.get("v_min")))
    return dict(kind=rec.kind, key_id=rec.key_id, cell=rec.cell_id, phase=rec.phase,
                status=rec.status, W=int(rec.W), cpu=float(rec.cpu_seconds), name=rec.stim_name,
                multi_basin=bool(rec.multi_basin), n_starts=len(rec.starts),
                has_brake=bool(rec.measures.get("has_brake", False)))


def _key_job(job: dict) -> dict:
    """One unit of key work in a worker (float64). ``job['what']`` names the recipe; cells
    travel with the job, so a worker always solves the main process's current cells."""
    import traceback

    from apexzero.interp import answer_key as ak

    bank, tables, cache = _W["bank"], _W["tables"], _W["cache"]
    what, cells = job["what"], job.get("cells", [])
    # Start (3) is a rollout of Q with the tuned c_v (protocol 7: Q is defined with it) when
    # the controls stage has tuned it; --force re-solves instead of reading the cache.
    rk = dict(q_cv=float(job.get("q_cv") or ak_default_cv()), force=bool(job.get("force")))
    out = dict(what=what, label=job.get("label", what), cells=[c.cell_id for c in cells],
               keys=[])
    c0, t0 = time.process_time(), time.time()
    try:
        if what == "layout":
            # One cell of the A2 layout with no D rounds: warms the cache with the cruise
            # s-OCP (A2) and the pinned s-OCP that gives W and P; the main process then runs
            # the global rule over every cell on the warm cache.
            out["layout"] = ak.layout_a2(bank, cells, tables, cache, max_rounds=0)
        elif what == "smoke":
            out["keys"] = [_rec_summary(ak.full_recipe(bank, cells[0], tables=tables,
                                                       W=cells[0].W, cache=None))]
        elif what == "series":
            out["keys"] = [_rec_summary(r) for r in ak.solve_series(bank, cells, cache,
                                                                    tables=tables, **rk)]
        elif what == "nominal":
            out["keys"] = [_rec_summary(ak.full_recipe(bank, c, cache=cache, tables=tables, **rk))
                           for c in cells]
        elif what in ("backoff", "phase"):
            for c in cells:
                base = ak.key_for(c, "nominal", 0, cache=cache, bank=bank)
                if base is None:
                    base = ak.full_recipe(bank, c, cache=cache, tables=tables, **rk)
                for k in (job.get("phases") or [0]) if what == "phase" else [0]:
                    out["keys"].append(_rec_summary(ak.warm_key(
                        base, bank, c, what, phase=k, cache=cache, tables=tables,
                        force=rk["force"], q_cv=rk["q_cv"])))
        elif what == "null":
            out["keys"] = [_rec_summary(ak.null_key(bank, c, cache, tables=tables,
                                                    force=rk["force"])) for c in cells]
        elif what == "socp":
            out["keys"] = [_rec_summary(ak.key_for(c, "socp", cache=cache, bank=bank, solve=True,
                                                   tables=tables)) for c in cells]
        elif what == "validate_pair":
            out["validation"] = _validate_pair(bank, cells, tables, cache, rk["q_cv"],
                                               force=rk["force"])
        elif what == "v7":
            out["validation"], out["P"] = _validate_v7_cell(bank, cells[0], tables, cache,
                                                            rk["q_cv"], force=rk["force"])
        elif what == "v8":
            res, b = ak.v8_band(bank, cells, cache, tables=tables)
            out["validation"] = {"V8": _jsonable(res), "b": b}
        elif what == "v10":
            out["validation"] = {"V10": _jsonable(ak.v10_solver_copies())}
        else:
            raise ValueError(f"unknown key job {what!r}")
    except Exception:  # noqa: BLE001  (a worker reports every failure to the main process)
        out["error"] = traceback.format_exc()
    out["cpu"], out["wall"] = time.process_time() - c0, time.time() - t0
    return out


def _pool(jobs: list[dict], workers: int, log, fn=None, init=None) -> list[dict]:
    """Run jobs in a fixed-size pool of spawned processes (JAX does not survive fork; XLA
    flags are inherited through the environment). Results in completion order; a job's
    exception is returned as its 'error' and logged, never raised."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if not jobs:
        return []
    fn, init = fn or _key_job, init or _key_worker_init
    out = []
    n = max(1, min(int(workers), len(jobs)))
    with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"),
                             initializer=init) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            out.append(r)
            if r.get("error"):
                note = "ERROR " + r["error"].strip().splitlines()[-1]
                log.write(f"error in {r['label']}:\n{r['error']}")
            else:
                st = [k.get("status", "") for k in r.get("keys", [])]
                note = ",".join(st) if st else "done"
            log.step(note=f"{r['label']}: {note} ({r['cpu']:.0f} CPU-s)")
    return out


def _errors(results: list[dict]) -> list[str]:
    return [f"{r['label']}: {r['error'].strip().splitlines()[-1]}" for r in results if r.get("error")]


# ------------------------------------------------------------------------- step 4: layout


def _split_nulls(cells: list) -> list:
    """``stimuli.split_nulls``: one tier G null per (straight, v0, W), every corner cell
    pointing to the null with its own W (idempotent; re-run whenever a W changes)."""
    from apexzero.interp import stimuli

    return stimuli.split_nulls(cells)


def _save_cells(bank, cells: list, log, why: str) -> tuple[str, list]:
    """Write the cells (keeping the first pre-layout copy), re-run the design checks that the
    layout can change (lengths), the support covariate, and the manifest's stimulus hash."""
    from apexzero.interp import agents, io, stimuli, support

    out = io.STIMULI_DIR
    pre = out / "cells_prelayout.json"
    if not pre.exists() and (out / "cells.json").exists():
        pre.write_bytes((out / "cells.json").read_bytes())
    # Any W change (layout, V2) must repoint corner cells to the null with their own W.
    cells = stimuli.split_nulls(cells)
    stimuli.save_cells(out / "cells.json", cells)
    h = stimuli.design_hash(bank, cells)
    det = stimuli.check_design_details(bank, cells)
    chk = io.read_json(out / "checks.json") if (out / "checks.json").exists() else {}
    chk.setdefault("checks", {})["lengths_after_layout"] = det["lengths"]
    chk["design_hash"] = h
    io.write_json(out / "checks.json", chk)
    io.save_table(out / "cell_support.npz",
                  stimuli.cell_support_table(bank, cells, support.load_support(io.SUPPORT_DIR)))
    agents.update_manifest(lambda m: setattr(m, "stimulus_hash", h))
    _log_design_deviation(h, f"cells saved after the tag ({why})")
    _say(log, f"cells saved ({why}): {len(cells)} cells, design hash {h[:16]}, lengths "
              f"{'PASS' if det['lengths']['ok'] else 'FAIL'} {_jsonable(det['lengths'])}")
    if not det["lengths"]["ok"]:
        raise SystemExit("a key window ends after the truncation point (protocol 3.1)")
    return h, cells


def _layout(bank, cells: list, tables, cache, workers: int, log) -> list:
    """Protocol 3.5's A2 layout and every cell's W and P (answer_key.layout_a2): the s-OCPs
    of every cell are solved in the pool first (one cell per job, no D rounds), then the
    global rule (D steps of 10 m until 90% of braking A2 cells have v_b* within 10% of v_typ)
    runs once over the warm cache. Nulls are split per W afterwards."""
    from apexzero.interp import answer_key as ak

    todo = [c for c in cells if c.W is None]
    jobs = [dict(what="layout", label=f"layout cell {c.cell_id} {c.approach}", cells=[c])
            for c in todo]
    log.set_total(log.done + len(jobs) + 1)
    res = _pool(jobs, workers, log)
    if _errors(res):
        raise SystemExit("layout failed: " + "; ".join(_errors(res)[:5]))
    ak._x64()
    laid = ak.layout_a2(bank, cells, tables, cache)
    a2 = [c for c in laid if c.approach == "A2"]
    brk = [c for c in a2 if "braking" in c.flags]
    out_band = [c.cell_id for c in brk if "vb_out_of_band" in c.flags]
    log.step(note=f"layout: {len(a2)} A2 cells, {len(brk)} braking, {len(out_band)} with v_b* "
                  "outside +-10% of v_typ")
    _say(log, f"A2 layout: {len(brk)} braking / {len(a2) - len(brk)} flat cells; in band "
              f"{1 - len(out_band) / max(len(brk), 1):.0%} (rule: >= 90%); W counts "
              f"{_count(c.W for c in laid)}")
    return _split_nulls(laid)


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


# ------------------------------------------------------------------------- step 4: keys


DEFAULT_TIMINGS = {"socp": 37.0, "nominal": 350.0, "warm": 85.0, "null": 40.0,
                   "anchored": 80.0, "viability": 80.0, "actuation": 350.0}   # protocol 10


def _direction(bank, c) -> int:
    return int(bank.stimuli[c.stim_id].direction)


def _one_direction(bank, cells: list, both: bool) -> list:
    """The cells whose keys are solved: every cell if the mirror shortcut is not allowed (V5
    failed), else all but the right-hand member of a pair that has its left-hand mirror in
    the design (``key_for`` mirrors that key on the fly)."""
    if both:
        return list(cells)
    lefts = {c.pair_id for c in cells if _direction(bank, c) == 1}
    return [c for c in cells if not (_direction(bank, c) == -1 and c.pair_id in lefts)]


def _spec(bank, c):
    return bank.stimuli[c.stim_id].spec


def _version(bank, c) -> str:
    return bank.stimuli[c.stim_id].name.rsplit("/", 1)[-1]


# Phase-matched keys at P0, P1 and P2; 'phase_B600' (P1) holds tier B D = 600's (ruling R17c).
PHASE_ITEMS = ("phase", "phase_P1", "phase_B600", "phase_P2")
PHASE_R_MAX = 18.3      # 6.1: phase-matched keys on the A1 cells with R <= 18.3 (P2b, P7's null)
B600_D = 600.0          # tier B's free approach whose band reads e_ve^RA (2.2, analysis uses_ra)


def _b600_cell(c) -> bool:
    """A tier B D = 600 cell (approach 'B', D = 600 m): the free approaches whose band reads
    e_ve^RA besides A1 (``analysis`` ``uses_ra``). Tier B's (310 km/h, 600 m) cells are the
    A1 cells (approach 'A1'), so these are the other three speeds, both directions."""
    return bool(c.approach == "B" and c.D is not None and abs(float(c.D) - B600_D) < 1e-6)


def _core_phase_cell(bank, c) -> bool:
    """Protocol 6.1's phase-matched cells: tier A at A1 with R <= 18.3 (20 geometries). They
    back Δd_b, P2b and P7's key phase SD, and keep their keys whatever V7 says."""
    sp = _spec(bank, c)
    return bool("A" in c.tiers and c.approach == "A1" and sp is not None and sp.R <= PHASE_R_MAX)


def _p7_cell(c) -> bool:
    """A cell run at more than the 4 deterministic phases: P7's 16-phase A1 cells (R in
    {9.4, 11.7, 14.6, 28.6, 55.9} x {90, 180} deg, both directions: 20 cells)."""
    from apexzero.interp.constants import N_PHASES

    return int(c.n_phases) > N_PHASES


def _phase_key_indices(c) -> list[int]:
    """The phases a cell's phase-matched keys are solved at: every phase but u = 0 (ruling
    R17b), so every deterministic run's errors use the key at its own phase. That is u_1..u_3
    on a 4-phase cell, as before, and all 15 on P7's 16-phase cells (they had 4, 8 and 12)."""
    return list(range(1, int(c.n_phases)))


def _p7_null_phases(c) -> list[int]:
    """P7's pre-registered key null (2.4: "the key's own SD across its 4 phase-matched
    solutions"): the keys at the 4-phase grid's phases, u = 0 and ``_phase_indices`` (0, 4,
    8, 12 of 16), whatever else is solved (R17b keeps the definition)."""
    return [0, *_phase_indices(c)]


def _v7_failed(m) -> bool:
    """Protocol 6.5: the manifest's validation V7 failed (on speeds, its only criterion).
    Its remedy: phase-matched keys for every braking cell, and speed errors judged against
    them (AD23)."""
    from apexzero.interp import analysis as A

    return A.v7_speeds_failed(getattr(m, "validation", None))


def _run_table(agent: str, ts: int, rollouts, cells: list, bank, keys, tables, m=None):
    """``analysis.run_table`` with AD23's speed-error keys set by the manifest's V7 verdict
    (``m``, read when not given): the key at each run's phase when V7 failed on speeds."""
    from apexzero.interp import analysis as A

    return A.run_table(agent, ts, rollouts, cells, bank, keys, tables,
                       speed_at_phase=_v7_failed(_manifest() if m is None else m))


def _key_sets(bank, cells: list, both: bool, v7_failed: bool = False) -> dict[str, list]:
    """The cells of every budget item (protocol 3.5's key column, one direction).

    Phase-matched keys (``PHASE_ITEMS``): while V7 holds, protocol 6.1's set (A1, R <=
    18.3, ``_core_phase_cell``) under 'phase'. When V7 failed on speeds (``v7_failed``,
    protocol 6.5's remedy), every cell with an agent-free nominal key is a candidate: 'phase'
    (P0) holds tier A (A1 and A2, the tier B cells shared with A1 included), E1's real and
    isolated rungs and the validation cells (tier V: rolled out at the finals, read by P5;
    their nominal keys come from the validate stage), 'phase_P1' E1's twins, E2 and E3,
    'phase_P2' tier C and the recovery cells (tier B); tier G (flat nulls) and tier D (s-OCP
    keys only) have none. Only braking candidates get keys (``_phase_targets``: known once
    the nominal key is solved); the core cells keep theirs regardless. Ruling R17b: P7's
    20 cells (``_p7_cell``) get keys at all 16 phases in every case, so while V7 holds
    'phase' holds them too (R 28.6 and 55.9 besides the core set).

    Ruling R17c: tier B D = 600 cells (``_b600_cell``; the band reads e_ve^RA there) are a
    P1 nominal-key item, 'nominal_B600' (one direction with the mirror shortcut, both without
    it), solved as protocol 6.2's descending-R series like tier A's; when V7 failed they are
    the phase item 'phase_B600' (P1, braking cells only), so their per-phase re-anchored
    keys (R17a) start from, and fall back to, the key at their own phase."""
    cs = _one_direction(bank, cells, both)
    sets = {
        "nominal_A": [c for c in cs if "A" in c.tiers and c.approach in ("A1", "A2")],
        "null": [c for c in cells if "G" in c.tiers],
        "nominal_E1": [c for c in cs if "E1" in c.tiers and _version(bank, c) in ("real", "isolated")],
        "E1_twins": [c for c in cs if "E1" in c.tiers and _version(bank, c) == "twin"],
        "E2": [c for c in cs if "E2" in c.tiers],
        "E3": [c for c in cs if "E3" in c.tiers and "A" not in c.tiers],
        "tierD_socp": [c for c in cells if "D" in c.tiers and "S1" in (c.runners or ("S1",))
                       and (bank.stimuli[c.stim_id].source or {}).get("condition", "").startswith(
                           ("F1/real/real/real", "F2/lambda"))],
        "tierC": [c for c in cs if "C" in c.tiers],
        "recovery": [c for c in cs if c.approach == "REC"],
        "nominal_B600": [c for c in cs if _b600_cell(c)],
    }
    sets["phase"] = [c for c in sets["nominal_A"] if _core_phase_cell(bank, c) or _p7_cell(c)]
    sets["phase_P1"], sets["phase_B600"], sets["phase_P2"] = [], [], []
    if v7_failed:
        seen: set[int] = set()

        def fresh(xs: list) -> list:
            out = [c for c in xs if c.cell_id not in seen]
            seen.update(c.cell_id for c in out)
            return out

        sets["phase"] = fresh(sets["nominal_A"] + sets["nominal_E1"]
                              + [c for c in cs if "V" in c.tiers])
        sets["phase_P1"] = fresh(sets["E1_twins"] + sets["E2"] + sets["E3"])
        sets["phase_B600"] = fresh(sets["nominal_B600"])
        sets["phase_P2"] = fresh(sets["tierC"] + sets["recovery"])
    return sets


def _nominal_braking(bank, c, cache) -> bool | None:
    """2.3's braking class of the cell's cached nominal u = 0 key (``_braking``); None while
    no nominal key is cached."""
    from apexzero.interp import answer_key as ak

    rec = ak.key_for(c, "nominal", 0, cache=cache, bank=bank)
    return None if rec is None else _braking(rec, c)


def _phase_split(bank, cs: list, cache, v7_failed: bool) -> tuple[list, list]:
    """(cells that get phase-matched keys, candidates whose braking class is not known yet).
    While V7 holds: all of them (protocol 6.1's set), none unknown. When V7 failed on speeds
    (6.5: "phase-matched keys for every braking cell"): the core cells
    (``_core_phase_cell``) and every candidate whose cached nominal key brakes (2.3: a
    sustained brake and v_min* < v0 - tau); flat cells, and cells whose nominal key is
    unsolved, get none. A candidate without a cached nominal key is unknown (every one with
    no ``cache``). P7's 16-phase cells are kept like the core cells (ruling R17b: keys at
    every phase on the 20 P7 cells)."""
    if not v7_failed:
        return list(cs), []
    keep, unknown = [], []
    for c in cs:
        if _core_phase_cell(bank, c) or _p7_cell(c):
            keep.append(c)
            continue
        b = _nominal_braking(bank, c, cache) if cache is not None else None
        if b:
            keep.append(c)
        elif b is None:
            unknown.append(c)
    return keep, unknown


def _phase_targets(bank, cs: list, cache, v7_failed: bool, unknown: bool = False) -> list:
    """The cells of a phase item that get phase-matched keys (``_phase_split``), in the
    item's order. ``unknown`` also keeps the candidates whose nominal key is not solved yet:
    the budget's upper bound before the nominal keys exist. Jobs are planned without it,
    after the nominal keys are solved."""
    keep, unk = _phase_split(bank, cs, cache, v7_failed)
    ids = {c.cell_id for c in keep + (unk if unknown else [])}
    return [c for c in cs if c.cell_id in ids]


def _phase_jobs(bank, cs: list, cache, proj: dict, item: str, v7_failed: bool,
                log=None) -> list[dict]:
    """Phase-matched key jobs (every phase but u = 0, ``_phase_key_indices``: u_1..u_3, and
    all 15 on P7's 16-phase cells by ruling R17b) for one phase item, planned once its
    cells' nominal keys are solved: ``_phase_split`` (braking cells when V7 failed), then
    budget rule 2 (``_cut``). The log counts the flat or unsolved cells apart from those
    with no cached nominal key (an errored job)."""
    tg, unk = _phase_split(bank, cs, cache, v7_failed)      # tg in the item's order
    keep = _cut(bank, tg, proj, item)
    n_keys = sum(len(_phase_key_indices(c)) for c in keep)
    if log is not None and (v7_failed or any(_p7_cell(c) for c in keep)):
        # Cells without a cached nominal key (its job errored, or was not run) are counted
        # apart from the flat and unsolved ones: a resumed run solves their keys.
        p7 = [c for c in keep if _p7_cell(c)]
        _say(log, f"{'V7 remedy' if v7_failed else 'phase-matched keys'}, {item}: {len(tg)} "
                  f"of {len(cs)} candidate cells get keys (braking, core or P7); "
                  f"{len(cs) - len(tg) - len(unk)} flat or unsolved skipped; {len(unk)} "
                  f"without a nominal key skipped (errored or not solved: a resumed keys run "
                  f"picks them up); {len(keep)} after budget rule 2: {n_keys} phase-matched "
                  f"keys, {sum(len(_phase_key_indices(c)) for c in p7)} of them on "
                  f"{len(p7)} P7 cells at every phase (R17b)")
    return [dict(what="phase", label=f"phase cell {c.cell_id}", cells=[c],
                 phases=_phase_key_indices(c)) for c in keep]


def _phase_indices(c) -> list[int]:
    """u_1..u_3 of the 4-phase grid (indices 4, 8, 12 on 16-phase cells): validation V7's
    phases (6.5) and, with u = 0, P7's key null (``_p7_null_phases``). The keys stage solves
    phase-matched keys at ``_phase_key_indices`` (every phase, ruling R17b)."""
    step = max(1, c.n_phases // 4)
    return [step, 2 * step, 3 * step]


ANCHORED_G_PROTOCOL = 96     # protocol 10: re-anchored A1 keys for the G finals, about 96


def _frozen_regimes() -> dict | None:
    """The frozen q1 / q10 per agent (support stage), or None before they exist."""
    from apexzero.interp import io, support

    try:
        return support.load_support(io.SUPPORT_DIR).regimes
    except (FileNotFoundError, OSError, KeyError, ValueError):
        return None


def _r12_extra_anchored(bank, cells: list, regimes: dict | None = None) -> int:
    """Ruling R12's addition to P0's re-anchored keys: the non-clipped below-support A1
    cells of every seed-0 generalist (design-time clipping; the stage applies the run-time
    one), from the frozen regimes. 0 before the support stage has frozen them. R18's
    replicates are counted apart (``_anchored_rep_count``)."""
    from apexzero.interp import agents

    regimes = _frozen_regimes() if regimes is None else regimes
    if not regimes:
        return 0
    return sum(len(_r12_below_cells(bank, cells, regimes, a))
               for a in agents.seed0_generalists() if a in regimes)


def _r12_below_cells(bank, cells: list, regimes: dict, agent: str) -> list:
    """One generalist's R12 cells: P0's re-anchored A1 cells below support (design-time
    regime 'below')."""
    from apexzero.interp import support

    return [c for c in anchored_cells(cells, bank, regimes, agent, "P0", True)
            if support.regime(bank.stimuli[c.stim_id].geometry.R_seg, regimes[agent])
            == "below"]


def _ra_phases_final(c) -> int:
    """Ruling R17a: re-anchored keys per cell at a final, one per deterministic phase (the
    finals run every phase: 4, and 16 on P7 cells)."""
    return int(c.n_phases)


def _anchored_g_count(bank, cells: list) -> dict:
    """P0's re-anchored A1 keys for the generalist finals, for the budget projection. Once
    the regimes are frozen, every cell is counted on the design (``anchored_cells`` at P0:
    tail, bulk and R12's non-clipped below-support cells, design-time clipping), so R12's
    addition and the rest are counted alike; before that, the protocol's 96 cells. Anchored
    keys start from each agent's own state, so both directions count (no mirror shortcut).
    Ruling R17a: one key per deterministic phase of each cell (``_ra_phases_final``: 4, and
    16 on P7's cells), so ``n`` (keys) is about 4x ``cells``; ``u0_keys`` is the count
    before R17 (one per cell, at u = 0)."""
    from apexzero.interp import agents
    from apexzero.interp.constants import N_PHASES

    gens = agents.seed0_generalists()
    regimes = _frozen_regimes()
    if not regimes or not any(a in regimes for a in gens):
        return dict(n=N_PHASES * ANCHORED_G_PROTOCOL, cells=ANCHORED_G_PROTOCOL,
                    u0_keys=ANCHORED_G_PROTOCOL, p7_cells=0, p7_keys=0,
                    source="protocol (regimes not frozen) x 4 phases (R17a)",
                    rule="R17a: one re-anchored key per deterministic phase",
                    r12_below_support=0, r12_below_support_keys=0,
                    protocol=ANCHORED_G_PROTOCOL)
    sel = [c for a in gens if a in regimes
           for c in anchored_cells(cells, bank, regimes, a, "P0", True)]
    below = [c for a in gens if a in regimes
             for c in _r12_below_cells(bank, cells, regimes, a)]
    p7 = [c for c in sel if _p7_cell(c)]
    return dict(n=sum(_ra_phases_final(c) for c in sel), cells=len(sel), u0_keys=len(sel),
                p7_cells=len(p7), p7_keys=sum(_ra_phases_final(c) for c in p7),
                source="design", rule="R17a: one re-anchored key per deterministic phase "
                                      "(4; 16 on P7 cells)",
                r12_below_support=len(below),
                r12_below_support_keys=sum(_ra_phases_final(c) for c in below),
                protocol=ANCHORED_G_PROTOCOL)


P6C_CHECKPOINTS = 3     # P6c's re-anchored checkpoints per generalist (the final is P0)
RA_U0_PROTOCOL = {"anchored_S": 50, "anchored_B": 108, "anchored_ckpt": 216}    # protocol 10


def _anchored_p12_count(bank, cells: list) -> dict[str, dict]:
    """P1's and P2's re-anchored keys for the budget projection, counted on the design as
    ``stage_anchored_keys`` will solve them (review finding 2: the protocol's u = 0
    estimates, 50 / 108 / 216, were below the stage's selection even before R17a).
    'anchored_S': the specialists' finals, A1 and tier B D = 600 cells (``anchored_cells``
    at P1); 'anchored_B': the generalist finals' tier B D = 600 cells; both one key per
    deterministic phase of a final (``_ra_phases_final``: 4, 16 on P7 cells), and needing no
    regimes. 'anchored_ckpt': the generalists' A1 tail and bulk cells at P6c's
    ``P6C_CHECKPOINTS`` checkpoints, 4 phases each (checkpoints run the 4-phase grid), once
    the generalists' regimes are frozen; before that, the protocol's 216 x 4. Each item
    records its u = 0 count (one key per cell and snapshot) and its source."""
    from apexzero.interp import agents
    from apexzero.interp.constants import N_PHASES

    out: dict[str, dict] = {}
    for item, agents_, gen in (("anchored_S", agents.specialists(), False),
                               ("anchored_B", agents.seed0_generalists(), True)):
        sel = [c for a in agents_ for c in anchored_cells(cells, bank, {}, a, "P1", gen)]
        out[item] = dict(keys=sum(_ra_phases_final(c) for c in sel), u0_keys=len(sel),
                         p7_keys=sum(_ra_phases_final(c) for c in sel if _p7_cell(c)),
                         source="design", protocol_u0=RA_U0_PROTOCOL[item])
    regimes = _frozen_regimes()
    # P6c's checkpoints (P2) stay on seed 0 (ruling R18).
    gens = [a for a in agents.seed0_generalists() if regimes and a in regimes]
    if gens:
        sel = [c for a in gens for c in anchored_cells(cells, bank, regimes, a, "P2", True)]
        u0 = P6C_CHECKPOINTS * len(sel)
        out["anchored_ckpt"] = dict(keys=N_PHASES * u0, u0_keys=u0, p7_keys=0,
                                    source=f"design: {len(sel)} cells x {P6C_CHECKPOINTS} "
                                           f"checkpoints x {N_PHASES} phases",
                                    protocol_u0=RA_U0_PROTOCOL["anchored_ckpt"])
    else:
        v = RA_U0_PROTOCOL["anchored_ckpt"]
        out["anchored_ckpt"] = dict(keys=N_PHASES * v, u0_keys=v, p7_keys=0,
                                    source=f"protocol 10 x {N_PHASES} phases (regimes not "
                                           "frozen)", protocol_u0=v)
    return out


def _replicate_regimes(regimes: dict | None) -> tuple[dict, dict[str, str]]:
    """Ruling R18: each replicate's frozen regime (q1 / q10 of its variant's grammar, R_clip
    from its own obs_rms, written by the support stage once its final is in the manifest),
    or, until then, its seed-0 agent's as a stand-in (the same grammar, so the same q1 / q10;
    only R_clip, and so the R12 below-support boundary, can differ). Returns (regimes by
    replicate, the source of each)."""
    from apexzero.interp import agents

    regimes = regimes or {}
    out, src = {}, {}
    for a in agents.replicates():
        if a in regimes:
            out[a], src[a] = regimes[a], "own"
        elif agents.seed0_of(a) in regimes:
            out[a], src[a] = regimes[agents.seed0_of(a)], f"stand-in: {agents.seed0_of(a)}'s"
    return out, src


def _anchored_rep_count(bank, cells: list) -> dict[str, dict]:
    """Ruling R18's re-anchored keys for the replicates' finals, for the budget projection:
    'anchored_R' (P0) their A1 tail, bulk and non-clipped below-support cells (R12), and
    'anchored_B_R' (P1) their tier B D = 600 cells, both as G1-G3's (``anchored_cells``),
    one key per deterministic phase of a final (R17a: 4, 16 on P7 cells). No P2 item: P6's
    checkpoint history stays on seed 0. Regimes from ``_replicate_regimes`` (a replicate
    whose own are not frozen yet is counted with its seed-0 agent's); before any regimes
    are frozen, the protocol's 96 cells per 3 generalists, x 4 phases."""
    from apexzero.interp import agents
    from apexzero.interp.constants import N_PHASES

    reps = agents.replicates()
    regs, src = _replicate_regimes(_frozen_regimes())
    out: dict[str, dict] = {}
    if regs:
        sel = [c for a in reps if a in regs
               for c in anchored_cells(cells, bank, regs, a, "P0", True)]
        below = [c for a in reps if a in regs for c in _r12_below_cells(bank, cells, regs, a)]
        out["anchored_R"] = dict(keys=sum(_ra_phases_final(c) for c in sel), u0_keys=len(sel),
                                 p7_keys=sum(_ra_phases_final(c) for c in sel if _p7_cell(c)),
                                 r12_below_support_keys=sum(_ra_phases_final(c) for c in below),
                                 agents=list(regs), regimes=src, source="design")
    else:
        n = round(ANCHORED_G_PROTOCOL * len(reps) / max(1, len(agents.seed0_generalists())))
        out["anchored_R"] = dict(keys=N_PHASES * n, u0_keys=n, p7_keys=0,
                                 r12_below_support_keys=0, agents=list(reps), regimes={},
                                 source="protocol (regimes not frozen) x 4 phases (R17a)")
    sel = [c for a in reps for c in anchored_cells(cells, bank, {}, a, "P1", True)]
    out["anchored_B_R"] = dict(keys=sum(_ra_phases_final(c) for c in sel), u0_keys=len(sel),
                               p7_keys=sum(_ra_phases_final(c) for c in sel if _p7_cell(c)),
                               agents=list(reps), source="design")
    return out


def _det_phases(c, kind: str) -> list[int]:
    """Protocol 4's replication table: the deterministic phases one snapshot of ``kind``
    runs on cell ``c`` (as ``_rollout_job`` plans them: none when the cell is in none of
    the kind's det tiers). Ruling R19a: 'last3' runs tiers A, B and G."""
    from apexzero.interp import rollout
    from apexzero.interp.constants import REPLICATION

    rep = REPLICATION[kind]
    tiers = rep.get("det_tiers")
    if tiers is not None and not set(c.tiers) & set(tiers):
        return []
    return rollout.det_phase_indices(int(c.n_phases), rep.get("det_phases"))


def _rollout_runs(c, kind: str, mode: str) -> int:
    """Protocol 4's replication table: the runs one snapshot of ``kind`` makes on cell ``c``
    in ``mode`` (as ``_rollout_job`` plans them with ``rollout.plan_runs``)."""
    from apexzero.interp import rollout
    from apexzero.interp.constants import REPLICATION

    if mode == "det":
        return len(_det_phases(c, kind))
    samples = REPLICATION[kind].get("native" if mode == "nat" else "matched")
    return rollout._samples_for(c, samples) if samples else 0


R19_TIERS = ("G",)          # ruling R19a: the tiers the last-3 rows gained (null straights)


def _r19_cell(c) -> bool:
    """A cell whose 'last3' deterministic runs ruling R19a added: tier G and not in the
    tiers protocol 4's last-3 rows ran before (A and B)."""
    return bool(set(c.tiers) & set(R19_TIERS)) and not set(c.tiers) & {"A", "B"}


def _rollout_projection(cells: list, sps: dict | None = None) -> dict:
    """Agent rollouts of step 6 counted on the design (protocol 4's replication table, every
    snapshot ``_snapshot_kinds(..., 'all')`` lists, each run W steps, tier D cells by
    ``Cell.runners`` with R18's replicates running their seed-0 agent's), per agent, kind
    and mode, with process hours at the measured steps/s (``sps``: 'full' for det, 'stoch'
    for nat and mat; protocol 10: 3 processes). Budget rule 3 is on throughput, not on this
    count. Agents whose run has no snapshot on disk count as one final (a plan). Ruling
    R19a: the 'last3' snapshots also run tier G (4 deterministic phases); those runs are
    counted in each agent's ``r19`` and in ``out['r19']``."""
    from apexzero.interp import agents, registry

    by_agent: dict[str, dict] = {}
    for a in agents.AGENTS:
        kinds = _count(k for _, k in _snapshot_kinds(a, "all")) or {"final": 1}
        mine = [c for c in cells if registry.runs_cell(tuple(c.runners or ()), a)]
        steps = {"det": 0, "nat": 0, "mat": 0}
        runs = {"det": 0, "nat": 0, "mat": 0}
        r19 = {"runs": 0, "steps": 0}
        for kind, n_snap in kinds.items():
            for c in mine:
                W = int(c.W or 800)
                for mode in steps:
                    n = n_snap * _rollout_runs(c, kind, mode)
                    runs[mode] += n
                    steps[mode] += n * W
                    if kind == "last3" and mode == "det" and _r19_cell(c):
                        r19["runs"] += n
                        r19["steps"] += n * W
        by_agent[a] = dict(snapshots=kinds, runs=runs, steps=steps,
                           total_steps=sum(steps.values()), r19=r19)
    sps = sps or {}
    full, stoch = float(sps.get("full", np.nan)), float(sps.get("stoch", np.nan))

    def hours(rows):
        det = sum(r["steps"]["det"] for r in rows)
        sto = sum(r["steps"]["nat"] + r["steps"]["mat"] for r in rows)
        return det / full / 3600.0 + sto / stoch / 3600.0 if sps else float("nan")

    reps = [by_agent[a] for a in agents.replicates()]
    rest = [v for a, v in by_agent.items() if not agents.is_replicate(a)]
    out = dict(by_agent=by_agent, steps_total=sum(v["total_steps"] for v in by_agent.values()),
               steps_replicates=sum(v["total_steps"] for v in reps),
               steps_pre_r18=sum(v["total_steps"] for v in rest),
               process_h=hours(list(by_agent.values())), process_h_replicates=hours(reps),
               sps=dict(full=full, stoch=stoch), processes=3)
    out["wall_h"] = out["process_h"] / out["processes"]
    r19_steps = sum(v["r19"]["steps"] for v in by_agent.values())
    out["r19"] = dict(
        rule="R19a: the two snapshots before each generalist final also run tier G (the null "
             "straights, 4 deterministic phases), so protocol 8's gate can judge them",
        cells=sum(1 for c in cells if _r19_cell(c)),
        runs=sum(v["r19"]["runs"] for v in by_agent.values()), steps=r19_steps,
        steps_without=out["steps_total"] - r19_steps,
        process_h=r19_steps / full / 3600.0 if sps else float("nan"))
    return out


def _rule2_keeps_phase(bank, c) -> bool:
    """Budget rule 2 keeps phase-matched keys for 180 deg cells only (a cell with no
    CornerSpec, a real road, is no 180 deg cell), and never for a cell whose nominal key the
    rule cuts (``budget_cut``: no nominal key to warm-start from)."""
    sp = _spec(bank, c)
    return bool(sp is not None and sp.theta_deg == 180.0 and not budget_cut(bank, c))


def _rule2_cut_counts(bank, sets: dict, phase: dict | None = None) -> dict[str, int]:
    """Budget rule 2's credit per item, counted on the item sets the projection counts (one
    direction with the mirror shortcut, both without it): the nominal keys of the cut cells
    (``budget_cut``: 6 with the shortcut, 12 without) and the phase-matched keys of every
    phase cell that ``_cut`` drops (``_rule2_keeps_phase``: non-180 deg cells; with V7's
    remedy also real roads and the cut cells), 3 per cell, 15 per P7 cell (R17b,
    ``_phase_key_indices``), per phase item. ``phase`` maps each phase item to
    the cells its count was taken on (default: ``sets``' own phase items). The backed-off
    item is the protocol's fixed one-direction estimate (66), so its credit stays
    ``answer_key.DEFAULT_CUTS['backoff']`` (one per cut geometry and approach)."""
    from apexzero.interp import answer_key as ak

    phase = {k: sets.get(k, []) for k in PHASE_ITEMS} if phase is None else phase
    out = {"nominal_A": sum(1 for c in sets["nominal_A"] if budget_cut(bank, c)),
           "backoff": ak.DEFAULT_CUTS["backoff"]}
    out.update({k: sum(len(_phase_key_indices(c)) for c in cs if not _rule2_keeps_phase(bank, c))
                for k, cs in phase.items()})
    return out


def _budget(bank, cells: list, both: bool, m, workers: int, cache=None) -> dict:
    """Protocol 10's projection under the budget rule. Phase-matched keys follow V7 (6.5):
    the core A1 set while it holds; when it failed on speeds, the braking cells of every
    nominal-key item at their nominal key's priority ('phase' P0, 'phase_P1' and
    'phase_B600' P1, 'phase_P2'),
    braking read from ``cache``'s nominal keys where solved and counted as braking where not
    (an upper bound; ``proj['phase_keys']``). Ruling R17b: each phase cell counts the keys
    of every phase but u = 0 (``_phase_key_indices``: 3, and 15 on P7's 16-phase cells,
    which are always counted). Ruling R17a: re-anchored keys are one per deterministic
    phase: P0's on the design (``_anchored_g_count``), and P1's and P2's on the design too
    (``_anchored_p12_count``: the specialists' A1 and tier B D = 600 cells, the generalists'
    tier B D = 600 cells, at every phase of a final; the checkpoints' 4 phases once the
    regimes are frozen) (``proj['anchored_phases']``). Ruling R17c: tier B D = 600's
    nominal keys are the keys stage's P1 item 'nominal_B600' (one per cell solved: one
    direction with the mirror shortcut, both without it) and, when V7 failed, its braking
    cells' phase-matched keys the phase item 'phase_B600' (P1).

    Ruling R18: the replicates' finals get G1-G3's re-anchored keys, 'anchored_R' (P0: A1
    tail, bulk and non-clipped below-support cells) and 'anchored_B_R' (P1: tier B D = 600),
    and their own viability solves, 'viability_R' (P0: 9.1's cap of 200 applied to the
    replicates apart, so G1-G3's seeded subset is unchanged) (``_anchored_rep_count``,
    ``proj['replicates']``). Their agent rollouts (finals and the last three checkpoints)
    are counted in ``proj['rollouts']`` (``_rollout_projection``). Ruling R19a: the last-3
    rollouts run tier G too, counted apart in ``proj['rollouts']['r19']``."""
    from apexzero.interp import answer_key as ak
    from apexzero.interp.constants import N_PHASES, VIABILITY_CAP

    timings = dict(DEFAULT_TIMINGS)
    timings.update(m.budget.get("timings", {}))
    if "rollout_sps" in m.budget:
        timings["rollout_sps"] = min(m.budget["rollout_sps"].values())
    v7_failed = _v7_failed(m)
    sets = _key_sets(bank, cells, both, v7_failed)
    split = {k: _phase_split(bank, sets[k], cache, v7_failed) for k in PHASE_ITEMS}
    ph = {}
    for k, (keep, unk) in split.items():        # counted: braking or not yet known
        ids = {c.cell_id for c in keep + unk}
        ph[k] = [c for c in sets[k] if c.cell_id in ids]
    n_keys = {k: sum(len(_phase_key_indices(c)) for c in cs) for k, cs in ph.items()}
    n_a2 = sum(1 for c in cells if c.approach == "A2")
    anchored_g = _anchored_g_count(bank, cells)
    ra12 = _anchored_p12_count(bank, cells)
    ra_rep = _anchored_rep_count(bank, cells)
    counts = {"layout_socp": n_a2, "v8_socp": 36, "validation": 30,
              "nominal_A": len(sets["nominal_A"]), "nominal_E1": len(sets["nominal_E1"]) or 30,
              "null": len(sets["null"]), "backoff": 66, "phase": n_keys["phase"],
              "anchored_G": anchored_g["n"],
              "viability": 200, "E1_twins": len(sets["E1_twins"]) or 15,
              "E2": len(sets["E2"]) or 39, "E3": len(sets["E3"]), "tierD_socp": 140,
              "tierC": len(sets["tierC"]), "recovery": len(sets["recovery"]),
              # R17c: tier B D = 600's nominal keys, a keys-stage item at P1 (the anchored
              # stage then finds them cached; its inline solve is only a fallback).
              "nominal_B600": len(sets["nominal_B600"]),
              **{k: v["keys"] for k, v in ra12.items()},
              # R18: the replicates' re-anchored keys and viability solves.
              **{k: v["keys"] for k, v in ra_rep.items()}, "viability_R": VIABILITY_CAP}
    if v7_failed:
        counts.update(phase_P1=n_keys["phase_P1"], phase_B600=n_keys["phase_B600"],
                      phase_P2=n_keys["phase_P2"])
    proj = ak.project_budget(timings, counts, cut_counts=_rule2_cut_counts(bank, sets, ph),
                             workers=workers)
    proj["anchored_G"] = anchored_g
    proj["anchored_phases"] = dict(
        rule="R17a: one re-anchored key per deterministic phase, anchored on that phase's run",
        anchored_G=dict(keys=anchored_g["n"], cells=anchored_g.get("cells"),
                        u0_keys=anchored_g.get("u0_keys"), source=anchored_g.get("source")),
        **ra12, **ra_rep)
    items = proj["items"]
    rep_items = [k for k in ("anchored_R", "anchored_B_R", "viability_R") if k in items]
    proj["replicates"] = dict(
        rule="R18: the replicates' finals get G1-G3's re-anchored keys (P0 A1, P1 tier B "
             "D = 600; no P2: P6 stays on seed 0) and their own viability solves (cap 200)",
        items={k: dict(n=items[k]["n"], cpu_h=items[k]["cpu_h"],
                       priority=items[k]["priority"]) for k in rep_items},
        cpu_h=sum(items[k]["cpu_h"] for k in rep_items),
        before_without_replicates={pr: proj["before"][pr] - sum(
            items[k]["cpu_h"] for k in rep_items if items[k]["priority"] == pr)
            for pr in ("P0", "P1", "P2")})
    proj["rollouts"] = _rollout_projection(cells, m.budget.get("rollout_sps"))
    p7 = {k: [c for c in cs if _p7_cell(c)] for k, cs in ph.items()}
    proj["phase_keys"] = dict(
        v7_failed=v7_failed, rule="every braking cell with a nominal key (V7 failed on speeds)"
        if v7_failed else "A1, R <= 18.3 (protocol 6.1)",
        phases="every phase but u = 0 (R17b): 3 per cell, 15 on P7's 16-phase cells",
        candidates={k: len(sets[k]) for k in PHASE_ITEMS},
        counted={k: len(ph[k]) for k in PHASE_ITEMS},
        keys=n_keys,
        p7_cells={k: len(v) for k, v in p7.items()},
        p7_keys={k: sum(len(_phase_key_indices(c)) for c in v) for k, v in p7.items()},
        p7_beyond_u1_u3={k: sum(len(_phase_key_indices(c)) - (N_PHASES - 1) for c in v)
                         for k, v in p7.items()},
        braking_unknown={k: len(split[k][1]) for k in PHASE_ITEMS})
    proj["timings_used"] = timings
    proj["timings_source"] = "validation" if m.budget.get("timings") else "protocol defaults"
    return proj


R7_CUT_R = 35.8      # budget rule 2: tier A 180 deg cells with R >= 35.8 lose their keys


def budget_cut(bank, c) -> bool:
    """Budget rule 2's keys-only cut (ruling R7): a tier A A1 or A2 cell at 180 deg with
    R >= 35.8 m. At 180 deg tier A reaches R 55.9, so the cut is 3 geometries (35.8, 44.7,
    55.9) x 2 approaches."""
    sp = _spec(bank, c)
    return bool("A" in c.tiers and c.approach in ("A1", "A2") and sp is not None
                and sp.theta_deg == 180.0 and sp.R >= R7_CUT_R)


def _cut(bank, cells: list, proj: dict, item: str) -> list:
    """Budget rule 2's cuts applied to one item's cells (keys only, R7)."""
    if not any(c.startswith("rule 2") for c in proj["cuts"]):
        return cells
    if item in ("nominal_A", "backoff"):
        return [c for c in cells if not budget_cut(bank, c)]
    if item in PHASE_ITEMS:
        return [c for c in cells if _rule2_keeps_phase(bank, c)]
    return cells


def _refresh_phase_cycles(bank, cells: list, cache, log, why: str) -> list:
    """D26's phase cycles from the nominal keys solved so far (``refresh_phase_cycles``),
    saved into the design only when some P changed."""
    from apexzero.interp import answer_key as ak

    ak._x64()
    new = ak.refresh_phase_cycles(bank, cells, cache)
    if any(a.P != b.P for a, b in zip(cells, new, strict=True)):
        _, new = _save_cells(bank, new, log, why)
    return new


def _series_jobs(bank, cells: list, label: str) -> list[dict]:
    from apexzero.interp import answer_key as ak

    return [dict(what="series", label=f"{label} series {i} ({len(s)} cells)", cells=s)
            for i, s in enumerate(ak.series_of(bank, cells))]


def _braking(rec, cell) -> bool:
    """2.3's braking class of a nominal key: a sustained brake and v_min* < v0 - tau."""
    from apexzero.interp.constants import MULTI_BASIN_FLOORS

    if rec is None or rec.status != "verified" or not rec.measures.get("has_brake"):
        return False
    tau = float(rec.tau.get("v_min", MULTI_BASIN_FLOORS["v_min"]))
    return float(rec.measures.get("v_min", np.inf)) < cell.v0 - tau


def stage_keys(args: argparse.Namespace) -> None:
    """Step 4 (answer_key): the A2 layout (pool of --workers, then the global D rule), then,
    once validation has run, the P0 keys (tier A nominal in descending-R series, E1 real and
    isolated, nulls; then backed-off and phase-matched keys warm from them), P1 (E1 twins,
    E2, E3, tier B D = 600 (ruling R17c), tier D s-OCPs; then, with V7 failed, their
    phase-matched keys) and P2 (tier C, recovery) up to --priority, under the budget rule
    of protocol 10. Resumes from the key cache. Ends with the acceptance checks V1, V4, V6
    and V7 over every key, and the infeasible / unsolved / multi-basin lists and tau table in
    the manifest. --smoke: one validation-sized solve through the pool, nothing cached.
    --force: re-lay out and re-solve instead of resuming from the cache. Start (3) uses
    Q's tuned c_v once the controls stage has tuned it. Refused after the tag."""
    from apexzero.interp import agents, io, measures
    from apexzero.interp import answer_key as ak

    _refuse_after_tag("the keys stage", args)
    bank, cells = _design()
    tables = measures.model_tables()
    cache = ak.KeyCache()
    q_cv = _tuned_cv()
    with io.StageLog("keys", total=1, note=f"{args.priority}, {args.workers} workers") as log:
        if args.smoke:
            res = _pool([dict(what="smoke", label="smoke R40/90 A2 W200",
                              cells=[_smoke_cell(bank, cells)])], 1, log)
            _say(log, f"smoke: {_jsonable(res[0])}")
            if res[0].get("error"):
                raise SystemExit("smoke solve failed")
            return
        _require_steps("the keys stage")
        if any(c.W is None for c in cells) or args.force:
            cells = _layout(bank, cells, tables, cache, args.workers, log)
            _, cells = _save_cells(bank, cells, log, "A2 layout, W, P, nulls per W")
        if args.layout_only:
            return
        m = _manifest()
        if "V5" not in m.validation:
            raise SystemExit("run `psychophysics.py validate` (V1-V11) before the keys: V5 "
                             "decides the mirror shortcut and the validation timings set the "
                             "budget (protocol 10, step 4)")
        _say(log, f"start (3): Q with c_v = {q_cv if q_cv is not None else ak_default_cv()} "
                  f"({'tuned' if q_cv is not None else 'default: controls not tuned yet'})")
        both = not cache.get_flag("mirror_ok", False)
        v7_failed = _v7_failed(m)
        if v7_failed:
            _say(log, "V7 failed on speeds (validation): phase-matched keys for every braking "
                      "cell with a nominal key, each item after its nominal keys (protocol 6.5); "
                      "speed errors are judged against them (AD23)")
        elif not ((getattr(m, "validation", None) or {}).get("V7") or {}).get("passed", True):
            _say(log, "V7 did not pass but listed no failed cell (not judged: no phase-matched "
                      "validation keys); protocol 6.1's phase set only until validation "
                      "judges it")
        proj = _budget(bank, cells, both, m, args.workers, cache=cache)
        agents.update_manifest(lambda mm: mm.budget.update(projection=proj, cuts=proj["cuts"],
                                                           priority=args.priority))
        cells = _flag_budget_cuts(bank, cells, proj, log)
        _say(log, f"budget ({proj['timings_source']} timings): before cuts {_jsonable(proj['before'])} "
                  f"CPU-h, after {_jsonable(proj['after'])}; cuts {proj['cuts'] or 'none'}; mirror "
                  f"shortcut {'off: both directions' if both else 'on'}")
        prios = {"P0": ["P0"], "P1": ["P0", "P1"], "P2": ["P0", "P1", "P2"]}[args.priority]
        if "P2" in prios and any(c.startswith("rule 1") for c in proj["cuts"]):
            prios.remove("P2")
            _say(log, "budget rule 1: P2 keys dropped")
        kw = dict(q_cv=q_cv, force=bool(args.force))

        def _pool_kw(jobs):
            return _pool([dict(j, **kw) for j in jobs], args.workers, log)

        sets = _key_sets(bank, cells, both, v7_failed)
        results: list[dict] = []
        # P0: nominal keys of tier A (series), E1 real / isolated, and the nulls.
        jobs = _series_jobs(bank, _cut(bank, sets["nominal_A"], proj, "nominal_A"), "tier A")
        jobs += [dict(what="nominal", label=f"E1 {bank.stimuli[c.stim_id].name}", cells=[c])
                 for c in sets["nominal_E1"]]
        jobs += [dict(what="null", label=f"null cell {c.cell_id} W{c.W}", cells=[c])
                 for c in sets["null"]]
        log.set_total(log.done + len(jobs))
        results += _pool_kw(jobs)
        # Phase cycles from the nominal keys (D26), then the warm-start keys.
        ak._x64()
        cells = ak.refresh_phase_cycles(bank, cells, cache)
        _, cells = _save_cells(bank, cells, log, "phase cycles from the nominal keys")
        sets = _key_sets(bank, cells, both, v7_failed)
        bo = [c for c in _cut(bank, sets["nominal_A"], proj, "backoff")
              if _braking(ak.key_for(c, "nominal", 0, cache=cache, bank=bank), c)]
        jobs = [dict(what="backoff", label=f"backoff cell {c.cell_id}", cells=[c]) for c in bo]
        jobs += _phase_jobs(bank, sets["phase"], cache, proj, "phase", v7_failed, log)
        log.set_total(log.done + len(jobs))
        results += _pool_kw(jobs)

        def _phase_after(items: tuple[str, ...], why: str):
            # Once the P1 / P2 nominal keys exist: their phase cycles (D26), whatever V7 says
            # (review of R17c, finding 6: the P of P1 / P2 cells followed their nominal keys
            # only when V7 failed), then V7's remedy for the P1 / P2 phase items, so the
            # phases are solved where the runs will start.
            nonlocal cells, sets
            cells = _refresh_phase_cycles(bank, cells, cache, log, why)
            sets = _key_sets(bank, cells, both, v7_failed)
            if not v7_failed:
                return []                       # no P1 / P2 phase items while V7 holds
            pj = [j for item in items
                  for j in _phase_jobs(bank, sets[item], cache, proj, item, v7_failed, log)]
            log.set_total(log.done + len(pj))
            return _pool_kw(pj)

        if "P1" in prios:
            jobs = [dict(what="nominal", label=f"P1 {bank.stimuli[c.stim_id].name}", cells=[c])
                    for c in sets["E1_twins"] + sets["E2"]]
            jobs += _series_jobs(bank, sets["E3"], "E3")
            # Ruling R17c: tier B D = 600's nominal keys (the band reads e_ve^RA there), as
            # 6.2's descending-R series, so the anchored stage finds them cached.
            jobs += _series_jobs(bank, sets["nominal_B600"], "tier B D600")
            dsets = sets["tierD_socp"]
            cond_a = [c for c in dsets if (bank.stimuli[c.stim_id].source or {}).get(
                "condition") == "F1/real/real/real"]
            jobs += [dict(what="socp", label=f"tier D (a) cell {c.cell_id}", cells=[c]) for c in cond_a]
            log.set_total(log.done + len(jobs))
            results += _pool_kw(jobs)
            results += _tier_d_doses(bank, cells, dsets, cond_a, cache, args.workers, log)
            results += _phase_after(("phase_P1", "phase_B600"),
                                    "phase cycles from the P1 nominal keys")
        if "P2" in prios:
            jobs = _series_jobs(bank, sets["tierC"], "tier C")
            jobs += [dict(what="nominal", label=f"recovery cell {c.cell_id}", cells=[c])
                     for c in sets["recovery"]]
            log.set_total(log.done + len(jobs))
            results += _pool_kw(jobs)
            results += _phase_after(("phase_P2",), "phase cycles from the P2 nominal keys")
        # Cached measures computed by older measures code are recomputed from the stored
        # solutions; keys whose multi-basin measures moved are listed for a re-solve.
        ak._x64()
        stale = ak.refresh_stale(cache, bank, tables)
        if stale["refreshed"]:
            _say(log, f"measures refreshed on {len(stale['refreshed'])} cached keys; "
                      f"{len(stale['moved'])} changed a multi-basin measure (re-solve with --force)")
        agents.update_manifest(lambda mm: mm.lists.__setitem__("stale_key_measures",
                                                               stale["moved"]))
        _acceptance(bank, cells, cache, tables, results, log)


def flag_budget_cuts(bank, cells: list, proj: dict) -> tuple[list, int]:
    """Ruling R7: budget rule 2's cells (``budget_cut``: tier A 180 deg, R >= 35.8, 3
    geometries x 2 approaches) lose their keys only. They are flagged 'budget_cut' and
    'no_key': they are still rolled out, and analysis excludes them from every key-based
    error (``analysis.has_no_key``). The flags follow the current projection: when rule 2
    no longer fires, they are cleared again, so a cell whose key ``_cut`` now solves is not
    hidden by a stale 'no_key'. Returns (cells, number of cells whose flags changed)."""
    from apexzero.interp.analysis import NO_KEY_FLAG

    cut_flags = {"budget_cut", NO_KEY_FLAG}
    rule2 = any(c.startswith("rule 2") for c in proj["cuts"])
    out, n = [], 0
    for c in cells:
        want = rule2 and budget_cut(bank, c)
        if want and not cut_flags <= set(c.flags):
            c = dataclasses.replace(c, flags=tuple(sorted({*c.flags, *cut_flags})))
            n += 1
        elif not want and "budget_cut" in c.flags:
            c = dataclasses.replace(c, flags=tuple(f for f in c.flags if f not in cut_flags))
            n += 1
        out.append(c)
    return (out, n) if n else (cells, 0)


def _flag_budget_cuts(bank, cells: list, proj: dict, log) -> list:
    """``flag_budget_cuts``, saved into the design when it changes any flag."""
    out, n = flag_budget_cuts(bank, cells, proj)
    if n:
        rule2 = any(c.startswith("rule 2") for c in proj["cuts"])
        msg = (f"budget rule 2 (R7): budget_cut / no_key updated on {n} cells (keys only; "
               "still rolled out)" if rule2 else
               f"budget rule 2 no longer fires: budget_cut / no_key cleared on {n} cells")
        _, out = _save_cells(bank, out, log, msg)
    return out


def _tier_d_doses(bank, cells, dsets, cond_a, cache, workers, log) -> list[dict]:
    """Tier D: the six lambda doses of the braking corners (condition (a)'s s-OCP brakes),
    then the F5 swap partners (stimuli.swap_partner) for the rebuild with swap A rows."""
    from apexzero.interp import answer_key as ak
    from apexzero.interp import io, stimuli

    vmin, braking = {}, set()
    for c in cond_a:
        soc = ak.key_for(c, "socp", cache=cache, bank=bank)
        if soc is None:
            continue
        i = int(bank.stimuli[c.stim_id].source["corner_index"])
        vmin[i] = float(soc.measures.get("v_min", np.nan))
        if soc.measures.get("has_brake") and vmin[i] < c.v0 - 0.5 / 3.6:
            braking.add(i)
    doses = [c for c in dsets if (bank.stimuli[c.stim_id].source or {}).get("condition", "")
             .startswith("F2/lambda") and int(bank.stimuli[c.stim_id].source["corner_index"]) in braking]
    jobs = [dict(what="socp", label=f"tier D dose cell {c.cell_id}", cells=[c]) for c in doses]
    log.set_total(log.done + len(jobs))
    res = _pool(jobs, workers, log)
    swap = {i: stimuli.swap_partner(i, vmin, braking) for i in sorted(braking)}
    swap = {i: j for i, j in swap.items() if j is not None}
    io.write_json(io.STIMULI_DIR / "swap.json", {str(i): j for i, j in swap.items()})
    _say(log, f"tier D: {len(braking)} braking corners; swap partners {swap} (swap.json: rebuild "
              "with `stimuli --real --force` to add the F5 swap A rows before the tag)")
    return res


def _all_records(cache) -> list:
    from apexzero.interp import answer_key as ak

    out = []
    for kind in ("nominal", "backoff", "phase", "null"):
        out += [r for r in cache.records(kind) if isinstance(r, ak.KeyRecord)]
    return out


def _acceptance(bank, cells, cache, tables, results, log) -> None:
    """Step 4's acceptance over every key: V1, V4, V6 (re-solving violations warm from their
    neighbours) and V7; the infeasible / unsolved / multi-basin lists, the tau table and the
    measured per-item CPU into the manifest."""
    from apexzero.interp import agents, io
    from apexzero.interp import answer_key as ak

    recs = _all_records(cache)
    cur = {c.cell_id: c for c in cells}
    v1 = ak.v1_replay(recs)
    v4 = ak.v4_spread(recs)
    v6, recs = ak.v6_monotone(recs, bank, cells, cache, tables=tables)
    v7 = ak.v7_phase(recs)
    unsolved = sorted({r.cell_id for r in recs if r.status == "unsolved"})
    infeasible = sorted({r.cell_id for r in recs if r.status == "infeasible_bound"})
    mb = sorted({r.cell_id for r in recs if r.multi_basin})
    p0 = [r for r in recs if r.kind == "nominal" and r.cell_id in cur and "A" in cur[r.cell_id].tiers]
    frac = len([r for r in p0 if r.status == "unsolved"]) / max(len(p0), 1)
    tau = {k: np.array([r.tau.get(k, np.nan) for r in recs], float)
           for k in ("d_b", "v_e", "v_min", "s_vmin", "line_apex")}
    tau.update(key_id=np.array([r.key_id for r in recs]), kind=np.array([r.kind for r in recs]),
               cell=np.array([r.cell_id for r in recs]), phase=np.array([r.phase for r in recs]))
    tpath = io.KEYS_DIR / "tau_table.npz"
    io.save_table(tpath, tau)
    cpu: dict[str, list] = {}
    for r in results:
        for k in r.get("keys", []):
            if "cpu" in k:
                item = {"nominal": "nominal", "backoff": "warm", "phase": "warm",
                        "null": "null"}.get(k.get("kind"), k.get("kind"))
                cpu.setdefault(item, []).append(k["cpu"])

    def upd(m):
        for v in (v1, v4, v6, v7):
            m.validation[f"acceptance_{v.name}"] = _jsonable(v)
        m.lists.update(unsolved=unsolved, infeasible_bound=infeasible, multi_basin=mb,
                       key_errors=_errors(results))
        m.tau_table = str(tpath.relative_to(ROOT))
        m.budget.setdefault("measured_keys_cpu_s", {}).update(
            {k: dict(n=len(v), median=float(np.median(v)), total_h=float(np.sum(v)) / 3600.0)
             for k, v in cpu.items()})

    agents.update_manifest(upd)
    _say(log, f"acceptance: V1 {'PASS' if v1.passed else 'FAIL'}, V4 reported, V6 "
              f"{'PASS' if v6.passed else 'FAIL'}, V7 {'PASS' if v7.passed else 'FAIL'}; "
              f"{len(recs)} keys; unsolved {len(unsolved)} cells ({frac:.1%} of tier A P0; > 5% "
              f"needs investigation), infeasible {len(infeasible)}, multi-basin {len(mb)}; "
              f"errors {len(_errors(results))}")


# ------------------------------------------------------------------------- step 4: validation


V7_RT = ((9.4, 90.0), (13.1, 90.0), (18.3, 90.0), (28.6, 90.0), (55.9, 90.0),
         (9.4, 180.0), (13.1, 180.0), (18.3, 180.0), (28.6, 180.0), (55.9, 180.0))


def _validate_pair(bank, pair: list, tables, cache, q_cv: float | None = None,
                   force: bool = False) -> dict:
    """One validation pair (left, right) in one worker, so no key is solved twice: V2's two
    horizons (the left member), V3's actuation key (A1 and A2 only), V5 (left and right) and
    the backed-off key (V9). ``force`` (--force) re-solves every key of the pair once,
    through ``answer_key.FreshCache`` (records cached before the job read as absent)."""
    from apexzero.interp import answer_key as ak

    cache = ak.FreshCache(cache) if force and cache is not None else cache
    q_cv = q_cv or ak_default_cv()
    left = next(c for c in pair if _direction(bank, c) == 1)
    out = {"V2": _jsonable(ak.v2_horizon(bank, [left], cache, tables=tables))}
    nom = ak.full_recipe(bank, left, cache=cache, tables=tables, q_cv=q_cv)
    if left.approach in ("A1", "A2"):
        out["V3"] = _jsonable(ak.v3_objective(bank, [left], cache, tables=tables, q_cv=q_cv))
    out["V5"] = _jsonable(ak.v5_mirror(bank, pair, cache, tables=tables))
    if _braking(nom, left):
        bo = ak.warm_key(nom, bank, left, "backoff", cache=cache, tables=tables)
        out["backoff"] = bo.key_id
    out["nominal"] = nom.key_id
    out["cpu_nominal"] = float(nom.cpu_seconds)
    return out


def _validate_v7_cell(bank, cell, tables, cache, q_cv: float | None = None,
                      force: bool = False) -> tuple[dict, float]:
    """V7 on one tier A cell: the nominal key, its phase cycle P, and phases u_1..u_3
    (``force``: every one re-solved once, as in ``_validate_pair``)."""
    from apexzero.interp import answer_key as ak

    cache = ak.FreshCache(cache) if force and cache is not None else cache
    nom = ak.full_recipe(bank, cell, cache=cache, tables=tables, q_cv=q_cv or ak_default_cv())
    P = ak.phase_cycle(bank, cell, nominal=nom)
    c = dataclasses.replace(cell, P=P)
    keys = [nom.key_id]
    cpu = []
    for k in _phase_indices(c):
        r = ak.warm_key(nom, bank, c, "phase", phase=k, cache=cache, tables=tables)
        keys.append(r.key_id)
        cpu.append(float(r.cpu_seconds))
    return {"keys": keys, "cpu_nominal": float(nom.cpu_seconds), "cpu_warm": cpu}, P


def _v3_merge(pr: list[dict], by_id: dict, b: float) -> dict:
    """V3 over the validation pair jobs (``pr``): every A1 / A2 pair's cells judged at V8's
    band b (protocol 6.5: blocks if the objective effect on v_e exceeds b / 2; the expected
    0.075 when V8 gave no b), by ``answer_key.v3_verdict``. A cell whose keys are not
    verified, and an A1 / A2 pair job that returned no V3 (an error), leave V3 unevaluated
    there, so it does not pass."""
    from apexzero.interp import answer_key as ak

    band = float(b) if np.isfinite(b) else ak.BAND_EXPECTED
    cells, missing = {}, []
    for r in pr:
        if not any(by_id[i].approach in ("A1", "A2") for i in r.get("cells", []) if i in by_id):
            continue
        d = r.get("validation", {}).get("V3")
        if d is None:
            missing.append(r["label"])
            continue
        cells.update(d.get("details", {}).get("cells", {}))
    v = ak.v3_verdict(cells, band)
    return dict(v, passed=bool(v["passed"] and not missing), missing_pairs=missing,
                cells={r["label"]: r.get("validation", {}).get("V3") for r in pr})


def _validate_keys(args) -> None:
    """Step 4's V1-V11 on the validation stimuli (protocol 6.5) in a pool of --workers: V10
    and V8 one job each; one job per validation pair (V2, V3, V5, backed-off key); one per
    V7 cell. Then V1, V4, V6, V9, V11 on the records, V3 merged and judged at V8's b
    (``_v3_merge``), V5's verdict as the cache's mirror_ok flag, b from V8, V2's W = 800
    classes applied to the cells, and the measured CPU per key item projected under the
    budget rule. Resumes from the key cache (--force re-solves every key of the pair and V7
    jobs once, ``answer_key.FreshCache``; actuation keys from before K15 are re-solved
    without it, K16). Refused after the tag."""
    from apexzero.interp import agents, io
    from apexzero.interp import answer_key as ak

    _refuse_after_tag("validation V1-V11", args)
    _require_steps("validation V1-V11")
    bank, cells = _design(laid_out=True)
    cache = ak.KeyCache()
    vcells = [c for c in cells if "V" in c.tiers]
    pairs: dict[int, list] = {}
    for c in vcells:
        pairs.setdefault(c.pair_id, []).append(c)
    v7 = []
    for R, th in V7_RT:
        for c in cells:
            sp = _spec(bank, c)
            if "A" in c.tiers and c.approach == "A1" and sp is not None and sp.R == R \
                    and sp.theta_deg == th and _direction(bank, c) == 1 and not sp.no_cap:
                v7.append(c)
    braking_v = [c for c in vcells if c.approach in ("A1", "A2") and _direction(bank, c) == 1]
    jobs = [dict(what="v10", label="V10 solver copies")]
    jobs.append(dict(what="v8", label=f"V8 band ({len(braking_v)} cells)", cells=braking_v))
    jobs += [dict(what="validate_pair", label=f"V2/V3/V5/V9 pair {pid}", cells=mem)
             for pid, mem in sorted(pairs.items()) if len(mem) == 2]
    jobs += [dict(what="v7", label=f"V7 cell {c.cell_id}", cells=[c]) for c in v7]
    jobs = [dict(j, q_cv=_tuned_cv(), force=bool(args.force)) for j in jobs]
    with io.StageLog("validate", total=len(jobs) + 1, note=f"V1-V11, {args.workers} workers") as log:
        c0 = time.time()
        res = _pool(jobs, args.workers, log)
        ak._x64()
        val: dict = {}
        by = {r["what"]: r for r in res if r["what"] in ("v8", "v10")}
        val["V10"] = by["v10"].get("validation", {}).get("V10", {"passed": False, "error": by["v10"].get("error")})
        v8 = by["v8"].get("validation", {})
        val["V8"] = v8.get("V8", {"passed": False, "error": by["v8"].get("error")})
        b = float(v8.get("b", np.nan))
        pr = [r for r in res if r["what"] == "validate_pair"]
        v2_fail = sorted({x for r in pr for x in r.get("validation", {}).get("V2", {}).get(
            "details", {}).get("use_W800_for", [])})
        v2_ok = all(r.get("validation", {}).get("V2", {}).get("passed", False) for r in pr)
        v5_ok = bool(pr) and all(r.get("validation", {}).get("V5", {}).get("passed", False) for r in pr)
        cache.set_flag("mirror_ok", v5_ok)
        val["V2"] = dict(passed=v2_ok, use_W800_for=v2_fail,
                         cells={r["label"]: r.get("validation", {}).get("V2") for r in pr})
        val["V3"] = _v3_merge(pr, {c.cell_id: c for c in vcells}, b)
        val["V5"] = dict(passed=v5_ok, pairs={r["label"]: r.get("validation", {}).get("V5") for r in pr})
        recs = {r.key_id: r for r in _all_records(cache)}
        vrecs = [recs[k] for r in pr for k in (r.get("validation", {}).get("nominal"),) if k in recs]
        brecs = [recs[k] for r in pr for k in (r.get("validation", {}).get("backoff"),) if k in recs]
        v7recs = [recs[k] for r in res if r["what"] == "v7"
                  for k in r.get("validation", {}).get("keys", []) if k in recs]
        val["V1"] = _jsonable(ak.v1_replay(vrecs + brecs + v7recs))
        val["V4"] = _jsonable(ak.v4_spread(vrecs + v7recs))
        val["V6"] = _jsonable(ak.v6_monotone(vrecs)[0])
        val["V7"] = _jsonable(ak.v7_phase(v7recs))
        val["V9"] = _jsonable(ak.v9_backoff(vrecs, brecs))
        val["V11"] = _jsonable(ak.v11_active_set(vrecs))
        log.step(note="V1-V11 merged")
        # V2: classes that need W = 800; V7 cells: their phase cycles from the nominal key.
        changed = False
        if v2_fail:
            cells = [dataclasses.replace(c, W=800) if (c.approach in v2_fail or (
                "imposed" in v2_fail and "imposed" in c.flags)) and c.W != 800 else c for c in cells]
            changed = True
        Ps = {r["cells"][0]: r["P"] for r in res if r["what"] == "v7" and "P" in r}
        if Ps:
            cells = [dataclasses.replace(c, P=Ps[c.cell_id]) if c.cell_id in Ps else c for c in cells]
            changed = True
        if changed:
            _, cells = _save_cells(bank, cells, log, "V2 windows / V7 phase cycles")
        timings = _validation_timings(res, recs)
        wall_h = (time.time() - c0) / 3600.0

        def upd(m):
            m.validation.update(val)
            if np.isfinite(b):
                m.band = b
            m.budget["timings"] = timings
            m.budget["validation_wall_h"] = wall_h
            m.budget["validation_cpu_h"] = sum(r["cpu"] for r in res) / 3600.0
            m.lists["validation_errors"] = _errors(res)

        m = agents.update_manifest(upd)
        proj = _budget(bank, cells, not v5_ok, m, args.workers, cache=cache)
        agents.update_manifest(lambda mm: mm.budget.__setitem__("projection", proj))
        io.write_json(io.OUT / "harness" / "validation.json", val)
        for k in sorted(val, key=lambda s: int(s[1:])):
            v = val[k]
            _say(log, f"{k}: {'PASS' if v.get('passed') else 'FAIL'}")
        _say(log, f"b = {b}; timings (CPU-s per unit) {timings}; projected CPU-h "
                  f"{_jsonable(proj['before'])}, cuts {proj['cuts'] or 'none'}")


def _validation_timings(res: list[dict], recs: dict) -> dict:
    """CPU-s per unit of each budget item, measured on the validation solves."""
    nom = [r["validation"]["cpu_nominal"] for r in res if r.get("validation", {}).get("cpu_nominal")]
    warm = [x for r in res if r["what"] == "v7" for x in r.get("validation", {}).get("cpu_warm", [])]
    v8 = [r for r in res if r["what"] == "v8" and not r.get("error")]
    n8 = sum(len(r.get("validation", {}).get("V8", {}).get("details", {}).get("cells", {})) for r in v8)
    out = {}
    if nom:
        out["nominal"] = float(np.median(nom))
    if warm:
        out["warm"] = float(np.median(warm))
        out["anchored"] = out["viability"] = out["warm"]
    if v8 and n8:
        out["socp"] = float(v8[0]["cpu"]) / (4 * n8)       # free + 3 constrained per cell
    if "socp" in out and "warm" in out:
        out["null"] = out["socp"] + out["warm"]
    out["actuation"] = out.get("nominal", DEFAULT_TIMINGS["actuation"])
    return out


# ------------------------------------------------------------------------- step 4/5: controls


CONTROL_AGENT = "control-{}"      # rollout files: rollouts/control-<name>/000000000000.npz


def _tier_cells(bank, cells: list, tiers: tuple[str, ...]) -> list:
    """Cells of the given tiers ('D' means condition (a) runs from S1's state for tuning,
    every condition otherwise), plus the tier G nulls they point to (catch trials)."""
    out = [c for c in cells if set(c.tiers) & set(tiers) and "G" not in c.tiers]
    if "G" in tiers or out:
        nulls = {c.null_cell for c in out} if "G" not in tiers else {
            c.cell_id for c in cells if "G" in c.tiers}
        out += [c for c in cells if c.cell_id in nulls]
    seen, uniq = set(), []
    for c in out:
        if c.cell_id not in seen:
            seen.add(c.cell_id)
            uniq.append(c)
    return uniq


def _cond_a_cells(bank, cells: list) -> list:
    from apexzero.interp.stimuli import COND_A

    return [c for c in cells if "D" in c.tiers and "S1" in (c.runners or ("S1",))
            and (bank.stimuli[c.stim_id].source or {}).get("condition") == COND_A]


def _knn_buffer(args) -> None:
    """A specialist's Hockenheim kNN buffer from 50 stochastic laps of its own final (a
    real-circuit evaluation, not a stimulus). S1's is protocol exception 2 (step 5, before
    the tag; K's memory and S1's d_OOD). S2's and S3's (their own d_OOD, P4) are built after
    the tag (``--agent S2 --agent S3``; decision D33). Recorded in the manifest."""
    from apexzero.interp import agents, controls, io

    who = args.agent or ["S1"]
    if set(who) - set(agents.specialists()):
        raise SystemExit("--knn-buffer builds specialists' buffers (S1, S2, S3)")
    if set(who) - {"S1"}:
        _require_tag("S2's / S3's kNN buffers (agent laps outside the two pre-tag exceptions)",
                     design=False)
    m = _manifest()
    finals = _finals(m)
    with io.StageLog("controls", total=len(who), note=f"kNN buffers {who}") as log:
        for a in who:
            out = io.CONTROLS_DIR / f"knn_{a}.npz"
            if out.exists() and not args.force:
                log.step(note=f"{a}: resumed (buffer exists)")
                continue
            if a not in finals:
                _say(log, f"{a}: no final in the manifest yet")
                log.step(note=f"{a}: skipped")
                continue
            e = finals[a]
            buf = controls.build_knn_buffer(_load(e))
            buf.save(out)
            rec = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), agent=a, snapshot=e.path,
                       stimuli="real Hockenheim laps (kNN buffer)",
                       why="protocol exception 2 (step 5)" if a == "S1"
                       else "d_OOD buffer (post-tag, D33)")
            key = "pre_tag_agent_rollouts" if a == "S1" else "post_tag_buffers"
            agents.update_manifest(lambda mm, rec=rec, key=key: mm.lists.setdefault(key, []).append(rec))
            log.step(note=f"{a}: buffer of {len(buf.obs)} observations")


def _control_resume_ok(meta: dict, want: dict) -> bool:
    """A control's saved runs are reused only if they were made with the same c_v, tau_c,
    design and (for T) key cache."""
    return all(meta.get(k) == v for k, v in want.items())


def stage_controls(args: argparse.Namespace) -> None:
    """Steps 4 and 5 (protocol 7): tune Q's c_v (largest of 0.95 / 0.90 / 0.85 / 0.80 with
    100% deterministic completion on tier A and tier D condition (a)), fit c3's tau_c to the
    key's d_b at R 22.9 / 90 deg / A1, and run every control through the harness on its
    tiers (deterministic, every phase; tier D from S1's state, as for the generalists),
    written as rollouts/control-<name>/0.npz for the dryrun stage's verdicts. K falls back to
    k = 5 with distance weighting when its C(a) < 0.9 (protocol 7). --knn-buffer builds the
    specialists' buffers. Controls are agent-free; the stage runs in float32. Refused after
    the tag (except S2's / S3's buffers)."""
    from apexzero.interp import agents, controls, io, measures, rollout, stimuli
    from apexzero.interp import answer_key as ak

    if args.knn_buffer:
        _knn_buffer(args)
        return
    _refuse_after_tag("the controls stage", args)
    _require_steps("the controls stage")
    bank, cells = _design(laid_out=True)
    tables = measures.model_tables()
    car = controls.car_tables(tables)
    cache = ak.KeyCache()
    io.assert_x64_off()
    out = io.CONTROLS_DIR
    design = stimuli.design_hash(bank, cells)
    with io.StageLog("controls", total=2 + len(controls.CONTROLS), note="tuning, controls") as log:
        p = out / "tuning.json"
        tun = io.read_json(p) if p.exists() and not args.force else {}
        tier_d = _cond_a_cells(bank, cells)
        if "c_v" not in tun or (tier_d and not tun.get("tier_d_cells")):
            # (Re)tuned once tier D condition (a) exists: protocol 7 tunes on A and D (a).
            tc = [c for c in cells if "A" in c.tiers]
            c_v, rates = controls.tune_cv(bank, tc, car, gate_cells=tier_d)    # ruling R14
            tun.update(c_v=c_v, rates={str(k): v for k, v in rates.items()},
                       n_cells=len(tc) + len(tier_d), tier_d_cells=len(tier_d),
                       rule="R14: 100% on tier A, P4 gate C >= 0.9 on tier D (a)",
                       note="" if tier_d else "tier D condition (a) absent (no real states yet)")
            io.write_json(p, tun)
        log.step(note=f"c_v = {tun['c_v']} ({tun['rates']})")
        if tun["c_v"] is None:
            raise SystemExit("no c_v in {0.95, 0.90, 0.85, 0.80} completes every cell: Q fails")
        c_v = float(tun["c_v"])
        if tun.get("tau_c") is None:
            a1 = [c for c in cells if c.approach == "A1" and "A" in c.tiers]
            cell = controls._tau_cell(bank, a1)
            key = ak.key_for(cell, "nominal", 0, cache=cache, bank=bank)
            if key is None or key.status != "verified":
                tun["tau_c"], tun["tau_note"] = None, "no verified nominal key at R22.9/90/A1 yet"
            else:
                tun["d_b_key"] = float(key.measures["d_b"])
                tun["tau_c"] = controls.fit_tau_c(bank, a1, car, tun["d_b_key"], c_v)
            io.write_json(p, tun)
        log.step(note=f"tau_c = {tun['tau_c']}")
        agents.update_manifest(lambda m: m.budget.__setitem__("controls_tuning", tun))
        key_hash = cache.content_hash()
        for name, spec in controls.CONTROLS.items():
            path = io.rollout_path(CONTROL_AGENT.format(name), 0)
            want = dict(c_v=c_v, tau_c=tun.get("tau_c"), design_hash=design)
            if name == "T":
                want["key_cache_hash"] = key_hash
            if path.exists() and not args.force:
                meta = rollout.load_rollouts(path)[1]
                if _control_resume_ok(meta, want):
                    log.step(note=f"{name}: resumed")
                    continue
            res = _run_one_control(name, spec, bank, cells, car, c_v, tun.get("tau_c"), cache)
            if isinstance(res, str):
                _say(log, f"control {name}: skipped ({res})")
                log.step(note=f"{name}: skipped")
                continue
            arrays, meta = res
            meta.update(want)
            if name == "K":
                arrays, meta = _knn_fallback(arrays, meta, spec, bank, cells, car, c_v, cache,
                                             tables, log)
            rollout.save_rollouts(path, {"det": arrays}, meta)
            done = controls.completion(arrays, bank)
            _say(log, f"control {name}: {len(done)} runs, completion {done.mean():.1%}")
            log.step(note=f"{name}: {len(done)} runs")


def _knn_fallback(arrays, meta, spec, bank, cells, car, c_v, cache, tables, log):
    """Protocol 7: K falls back to k = 5 with distance weighting if its C(a) < 0.9 (over
    the braking corners of tier D condition (a), from S1's state)."""
    from apexzero.interp import analysis as A
    from apexzero.interp import controls

    cs = _control_cells("K", spec, bank, cells)
    keys = A.CachedKeys(bank, cells, cache, tables=tables)
    kt = A.key_table(cs, bank, keys)
    rt = _run_table(CONTROL_AGENT.format("K"), 0, {"det": arrays}, cells, bank, keys, tables)
    C, _, _ = A.completion_by_condition(A.cond_table(rt, kt))
    c_a = C.get(A.COND_A, float("nan"))
    meta["k"], meta["C_a_k1"] = 1, c_a
    if np.isfinite(c_a) and c_a >= A.P4_GATE:
        return arrays, meta
    _say(log, f"control K: C(a) = {c_a:.2f} < 0.9 with k = 1: k = 5 fallback (protocol 7)")
    res = _run_one_control("K", spec, bank, cells, car, c_v, None, cache,
                           k=controls.KNN_K_FALLBACK)
    if isinstance(res, str):
        return arrays, meta
    arrays5, meta5 = res
    meta.update(meta5, k=controls.KNN_K_FALLBACK)
    return arrays5, meta


def _control_cells(name: str, spec, bank, cells) -> list:
    """The cells a control runs: its tiers' cells (tier O is the probes stage's) and their
    nulls; tier D only from S1's state (as for the generalists: K is S1's memoriser, Q
    reads the road); Qp on the A1 short corners and every P7 cell."""
    tiers = tuple(t for t in spec.tiers if t != "O")
    cs = _tier_cells(bank, cells, tiers)
    cs = [c for c in cs if "D" not in c.tiers or "S1" in (c.runners or ("S1",))]
    if name == "Qp":           # A1 short corners and every P7 cell (P7's null, protocol 7)
        cs = [c for c in cs if c.approach == "A1" and _spec(bank, c) is not None
              and (_spec(bank, c).R <= 18.3 or "p7" in c.flags)]
    return cs


def _run_one_control(name: str, spec, bank, cells, car, c_v: float, tau_c, cache, k: int = 1):
    """(det arrays, meta) of one control on its tiers, or the reason it cannot run yet."""
    from apexzero.interp import answer_key as ak
    from apexzero.interp import controls, io

    cs = _control_cells(name, spec, bank, cells)
    if not cs:
        return f"no cells of tiers {spec.tiers} in the design"
    meta = dict(agent=CONTROL_AGENT.format(name), timesteps=0, log_std=[-np.inf, -np.inf],
                control=name, c_v=c_v, tau_c=tau_c)
    if name == "T":
        tapes, keep = [], []
        for c in cs:
            kind = "null" if "G" in c.tiers else "nominal"
            key = ak.key_for(c, kind, 0, cache=cache, bank=bank)
            if key is not None and key.status == "verified" and key.a is not None:
                tapes.append(np.asarray(key.a, np.float32))
                keep.append(c)
        if not keep:
            return "no verified u = 0 keys yet"
        params, aux, cs = controls.tape_params(tapes), np.arange(len(keep)), keep
        policy = controls.make_tape_policy()
        meta["n_phases"] = 1
        cs = [dataclasses.replace(c, n_phases=1) for c in cs]   # the tape is the u = 0 key's
    elif name == "K":
        p = io.CONTROLS_DIR / "knn_S1.npz"
        if not p.exists():
            return "no kNN buffer (controls --knn-buffer)"
        buf = controls.KnnBuffer.load(p)
        rms = _load(_finals(_manifest())["S1"]).obs_rms
        params = controls.knn_params(buf, rms.mean, rms.var)
        policy, aux = controls.make_knn_policy(k), np.zeros(len(cs), np.int32)
        meta["knn_buffer"] = dict(buf.meta, path=str(p.relative_to(ROOT)), k=k)
    else:
        if name == "c3" and tau_c is None:
            return "tau_c not fitted yet"
        params, aux = controls.q_params(bank, cs, car, c_v, name, float(tau_c or 0.0))
        policy = controls.make_q_policy(name)
    arrays = controls.run_control(policy, params, aux, bank, cs)
    return arrays, meta


# ------------------------------------------------------------------------- step 5: dry run


def _control_verdicts(bank, cells, kt, keys, tables, b: float, sup, log) -> dict:
    """Protocol 7's required verdicts of every control whose rollouts exist. T: regret 0,
    every error within tau, no termination; Q: the harness values match Q's offline
    prediction, P1a, P3, P4 (with d_OOD against S1's buffer) and P5; K: P4 with d_OOD
    against S1's buffer, normalised with S1's obs_rms."""
    from apexzero.interp import analysis as A
    from apexzero.interp import controls, io, rollout
    from apexzero.interp.support import RegimeTable

    s1_buf = io.CONTROLS_DIR / "knn_S1.npz"
    buf = controls.KnnBuffer.load(s1_buf) if s1_buf.exists() else None
    m = _manifest()
    s1 = _finals(m).get("S1")
    s1_rms = _load(s1).obs_rms if (s1 is not None and buf is not None) else None
    out = {}
    for name, spec in controls.CONTROLS.items():
        agent = CONTROL_AGENT.format(name)
        path = rollout_path_of(agent)
        if not path.exists():
            out[name] = dict(status="missing rollouts", required=spec.required)
            continue
        groups, cmeta = rollout.load_rollouts(path)
        rt = _run_table(agent, 0, groups, cells, bank, keys, tables, m)
        base = sup.regimes["G1"]
        q1, q10 = (12.0, 20.0) if name == "c2" else (base.q1, base.q10)
        reg = {agent: RegimeTable(agent=agent, grammar="", q1=q1, q10=q10, r_clip=(),
                                  r_clip_max=float("nan"), snapshot="")}
        pairs = A.pair_values(rt, kt, reg, b)
        got: dict[str, str] = {}
        extra: dict = {}
        if "P1a" in spec.required:
            got["P1a"] = A.p1a(pairs, b).verdict
        if "P1b" in spec.required:
            got["P1b"] = A.p1b(pairs, b).verdict
        if "P2" in spec.required:
            got["P2"] = A.p2(pairs, rt, b).verdict
        if "P2b" in spec.required:
            d = A.p2b(pairs, b)
            got["P2b"] = "onset_share>0.5" if d.verdict == "confirm" else d.verdict
        if "P3" in spec.required:
            d = A.p3(A.p3_fit(rt, kt, tables), A.brake_share(rt, kt))
            got["P3"] = f"refute:{d.details.get('winner', '')}" if d.verdict == "refute" else d.verdict
        if "P5" in spec.required:
            d = A.p5(rt, kt)
            got["P5"] = "symmetric" if d.verdict == "confirm" else d.verdict
        if "P4" in spec.required:
            dood = None
            if buf is not None and s1_rms is not None and "det" in groups:
                dood = _d_ood_from_det(groups["det"], agent, bank, cells, s1_rms, buf)
            cond = A.cond_table(rt, kt, dood)
            res = A.p4_classify(cond, None) if A._n(cond) else None
            got["P4"] = A.p4(res).verdict if res is not None else "not testable"
            extra["P4"] = dict(label=res.label if res else "", C=res.C if res else {},
                               d_ood="S1 buffer" if dood is not None else "missing",
                               k=cmeta.get("k"))
            if name == "K" and got["P4"] == "class 1":
                scales = {bank.stimuli[c.stim_id].source.get("f6_scale", 1.0) for c in cells
                          if "D" in c.tiers}
                extra["P4"]["action"] = (
                    "protocol 7: halve the F6 amplitudes once (stimuli --real --force "
                    "--f6-scale 0.5), then re-run controls and dryrun" if scales <= {1.0}
                    else "already re-run with halved F6 amplitudes")
        if name == "T":
            det = rt
            crashed = A._bo(det, "crashed")
            reg_ = A._fl(det, "regret") if "regret" in det else np.full(A._n(det), np.nan)
            J = A._fl(det, "J")
            rel = np.abs(reg_) / np.maximum(np.abs(J), 1e-12)
            got["termination"] = "none" if not crashed.any() else f"{int(crashed.sum())} crashes"
            got["regret"] = "zero" if np.nanmax(rel, initial=0.0) <= 1e-6 else f"max rel {np.nanmax(rel):.2e}"
            te = A.tape_errors(rt, kt)
            got["errors"] = "within_tau" if te["passed"] else f"{te['n_beyond']} beyond tau"
            extra["errors"] = te
        if name == "Q" and "offline_match" in spec.required:
            om = controls.offline_match_runs(rt, kt, bank, cells, controls.car_tables(tables),
                                             float(cmeta.get("c_v", 0.95)))
            got["offline_match"] = "pass" if om["passed"] else f"{om['n_fail']} of {om['n']} differ"
            extra["offline_match"] = om
        if name == "Qp":
            got["runs"] = "clean" if not A._bo(rt, "crashed").any() else "crashes"
        ok = all(got.get(k) in v for k, v in spec.required.items() if k in got)
        missing = [k for k in spec.required if k not in got]
        out[name] = dict(status="pass" if ok and not missing else ("incomplete" if ok else "FAIL"),
                         got=got, required={k: list(v) for k, v in spec.required.items()},
                         not_computed=missing, runs=A._n(rt), details=extra)
        _say(log, f"control {name}: {out[name]['status']} {got}")
    return out


def rollout_path_of(agent: str, timesteps: int = 0) -> Path:
    from apexzero.interp import io

    return io.rollout_path(agent, timesteps)


def stage_dryrun(args: argparse.Namespace) -> None:
    """Step 5: the key table of the real design, the dry run (five simulated agents must get
    their planted verdicts), the power table, P3's pre-tag separation check, the controls'
    required verdicts (protocol 7; any miss blocks the tag), and the code, stimulus and
    key-cache hashes in the manifest and in the pre-registration record of
    docs/interp/IMPLEMENTATION.md (tracked, so the tag carries them; post-tag stages check
    against it). Writes analysis/{key_table,dry_run,power}.npz. Refused after the tag."""
    from apexzero.interp import agents, io, measures, stimuli, support
    from apexzero.interp import analysis as A
    from apexzero.interp import answer_key as ak
    from apexzero.interp.constants import BAND_EXPECTED

    _refuse_after_tag("the dry run (it writes the pre-registration record)")
    _require_steps("the dry run")
    bank, cells = _design(laid_out=True)
    tables = measures.model_tables()
    sup = support.load_support(io.SUPPORT_DIR)
    cache = ak.KeyCache()
    m = _manifest()
    b = float(m.band) if m.band is not None else BAND_EXPECTED
    out = io.ANALYSIS_DIR
    with io.StageLog("dryrun", total=6, note=f"b = {b}") as log:
        keys = A.CachedKeys(bank, cells, cache, tables=tables)
        kt = A.key_table(cells, bank, keys)
        io.save_table(out / "key_table.npz", kt)
        log.step(note=f"key table: {A._n(kt)} cells, {int((kt['key_status'] == 'verified').sum())} verified")
        dr = A.dry_run(kt, tables, b)
        io.save_table(out / "dry_run.npz", dr)
        n_ok = int(np.sum(dr["passed"]))
        _say(log, f"dry run: {n_ok}/{len(dr['passed'])} planted verdicts returned")
        for i in range(len(dr["passed"])):
            if not dr["passed"][i]:
                _say(log, f"  MISSED {dr['kind'][i]} {dr['prediction'][i]}: expected "
                          f"{dr['expected'][i]}, got {dr['verdict'][i]}")
        log.step(note="dry run")
        pw = A.power_table(kt, sup, b, tables)
        io.save_table(out / "power.npz", pw)
        log.step(note="power table")
        sep = A.p3_separation(kt, tables)
        _say(log, f"P3 separation check: {_jsonable(sep)}")
        log.step(note="P3 separation")
        cv = _control_verdicts(bank, cells, kt, keys, tables, b, sup, log)
        log.step(note="controls")
        h_code, h_stim, h_keys = io.code_hash(), stimuli.design_hash(bank, cells), cache.content_hash()
        h_frozen = io.code_hash(FROZEN_PATHS)

        def upd(mm):
            mm.code_hash, mm.stimulus_hash, mm.key_cache_hash = h_code, h_stim, h_keys
            mm.validation["dry_run"] = dict(passed=bool(n_ok == len(dr["passed"])),
                                            n=len(dr["passed"]), returned=n_ok)
            mm.validation["p3_separation"] = _jsonable(sep)
            mm.validation["controls"] = _jsonable(cv)

        agents.update_manifest(upd)
        # The pre-registration record goes into a tracked file, so the tagged commit holds
        # the hashes the post-tag stages are checked against (outputs/ is not tracked).
        rec = dict(tag=TAG, written=time.strftime("%Y-%m-%dT%H:%M:%S"), code_hash=h_code,
                   frozen_code_hash=h_frozen, stimulus_hash=h_stim, key_cache_hash=h_keys,
                   band=b, p3_demoted=bool(sep.get("demote_p3", False)),
                   dry_run_passed=bool(n_ok == len(dr["passed"])),
                   controls={k: v["status"] for k, v in cv.items()})
        doc = ROOT / IMPL_DOC
        doc.write_text(write_prereg(doc.read_text(), _jsonable(rec)))
        log.step(note="hashes")
        blocked = [k for k, v in cv.items() if v["status"] != "pass"]
        _say(log, f"hashes: code {h_code[:16]}, stimuli {h_stim[:16]}, keys {h_keys[:16]}. "
                  + ("Tag BLOCKED by controls: " + ", ".join(blocked) if blocked or n_ok < len(dr["passed"])
                     else f"ready to commit and tag {TAG} (the user's call)"))


# ------------------------------------------------------------------------- step 6: rollouts

_R: dict = {}


def _rollout_worker_init() -> None:
    """Each rollout worker: float32 (asserted), the saved bank and cells, and their hash."""
    from apexzero.interp import io, stimuli

    io.assert_x64_off()
    bank, cells = stimuli.load_design(io.STIMULI_DIR)
    _R.update(bank=bank, cells=cells, design_hash=stimuli.design_hash(bank, cells))


def _file_lock(path: Path):
    """An exclusive fcntl lock on '<path>.lock' (one writer per rollout file)."""
    import contextlib
    import fcntl

    @contextlib.contextmanager
    def lock():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_name(path.name + ".lock"), "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    return lock()


def _groups_in(path: Path) -> set[str]:
    """The groups (det / nat / mat) a rollout file holds, read from its index only."""
    if not path.exists():
        return set()
    with np.load(path, allow_pickle=False) as d:
        return {k.partition("_")[0] for k in d.files if k.partition("_")[0] in ("det", "nat", "mat")}


def _group_runs(path: Path, mode: str) -> set[tuple[int, int, int]] | None:
    """The (cell, phase, sample) runs that group ``mode`` of a rollout file holds, read from
    its index arrays only; None when the file or the group is absent."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as d:
        if f"{mode}_cell" not in d.files:
            return None
        return set(zip(d[f"{mode}_cell"].tolist(), d[f"{mode}_phase"].tolist(),
                       d[f"{mode}_sample"].tolist(), strict=True))


def _tier_g_missing(path: Path, cells: list, agent: str) -> int:
    """Ruling R19a: how many of the tier G deterministic runs that protocol 4's 'last3' row
    plans for ``agent`` (every tier G cell it runs, at the 4-phase grid) the det group of
    the rollout file at ``path`` lacks; all of them when the file or its det group is
    absent. Protocol 8's gate needs them to judge the two snapshots before a final."""
    from apexzero.interp import registry

    want = {(c.cell_id, k, -1) for c in cells if "G" in c.tiers
            and registry.runs_cell(tuple(c.runners or ()), agent) for k in _det_phases(c, "last3")}
    have = _group_runs(path, "det") or set()
    return len(want - have)


def rollout_cells(cells: list, tiers=None) -> list:
    """The cells a snapshot's rollout groups run (``tiers``: REPLICATION's det tiers, None for
    every tier). Budget rule 2's cut cells are among them: ruling R7 cuts their keys only,
    so they are rolled out and carry 'no_key' into the analysis."""
    return [c for c in cells if tiers is None or set(c.tiers) & set(tiers)]


def _rollout_job(job: dict) -> dict:
    """Every requested group of one snapshot (resumes groups already in its file unless
    ``force``). The snapshot is marked evaluated in the manifest before its first group is
    run, and each group is merged into the file as it is on disk under a per-file lock, so
    two rollouts processes on one snapshot never drop each other's groups. A group on disk
    is kept only when it holds every run the plan asks for, or more (``_group_runs``). This
    holds for every snapshot kind and mode, finals and other checkpoints included, not only
    the last-3 rows. A group that lacks some runs, such as a last-3 det group made before
    ruling R19a added tier G, is run again whole (every planned run, not only the missing
    ones) and replaces the one on disk (``out['rerun']``). This job writes whole groups, so
    on an unchanged design only a group made under an older plan is ever run again."""
    import traceback

    from apexzero.interp import agents, io, rollout
    from apexzero.interp.constants import REPLICATION

    out = dict(label=job["label"], path=job["path"], groups=[])
    c0 = time.process_time()
    try:
        if job.get("design_hash") and job["design_hash"] != _R["design_hash"]:
            raise RuntimeError("the design on disk changed since the stage started")
        bank, cells = _R["bank"], _R["cells"]
        cells_run = rollout_cells(cells)     # R7: budget-cut (no_key) cells included
        ag = agents.load_agent(ROOT / job["path"], job["agent"])
        target = io.rollout_path(job["agent"], ag.timesteps)
        rep = REPLICATION[job["kind"]]
        tiers = rep.get("det_tiers")
        cs = rollout_cells(cells_run, tiers)
        pol = rollout.agent_policy(ag.cfg)
        marked = False
        for mode in job["modes"]:
            if mode == "det":
                runs = rollout.plan_runs(cs, bank, "det", 0, phases_det=rep.get("det_phases"),
                                         agent=job["agent"])
                rec = "full"
            else:
                samples = rep.get("native" if mode == "nat" else "matched")
                if not samples:
                    continue
                samples = {k: v for k, v in samples.items() if not (
                    job.get("rule3") and (job["kind"] != "final" or (mode == "mat" and k == "E")))}
                if not samples:
                    continue
                runs = rollout.plan_runs(cells_run, bank, mode, samples, agent=job["agent"])
                rec = "stoch"
            if not runs:
                continue
            with _file_lock(target):
                have = _group_runs(target, mode)
            if have is not None and not job.get("force"):
                if {(r.cell_id, r.phase, r.sample) for r in runs} <= have:
                    continue
                out.setdefault("rerun", []).append(mode)    # incomplete on disk: run again
            if not marked:
                # Freeze-rule bookkeeping first: a snapshot is never rolled out unmarked.
                agents.mark_evaluated_locked(job["path"], f"rollouts-{mode}")
                marked = True
            arrays = rollout.run_rollouts(pol, ag.actor_params, ag.obs_rms, bank, cells, runs, rec)
            with _file_lock(target):
                groups, meta = rollout.load_rollouts(target) if target.exists() else ({}, {})
                groups[mode] = arrays
                gm = dict(meta.get("groups", {}) or {})
                gm.pop(mode, None)                 # this group is new: record what made it
                rollout.save_rollouts(target, groups, dict(
                    meta, agent=job["agent"], timesteps=ag.timesteps, log_std=ag.log_std.tolist(),
                    path=job["path"], kind=job["kind"], design_hash=_R["design_hash"],
                    groups=gm))
            out["groups"].append(f"{mode}:{len(runs)}")
        out["timesteps"] = ag.timesteps
    except Exception:  # noqa: BLE001  (a worker reports every failure to the main process)
        out["error"] = traceback.format_exc()
    out["cpu"] = time.process_time() - c0
    return out


def _snapshot_kinds(agent: str, which: str) -> list[tuple[str, str]]:
    """(repo-relative path, replication kind) of the snapshots to roll out (protocol 4's
    replication table): finals; the two before the final ('last3', generalists: tiers A and
    B, and since ruling R19a tier G, so protocol 8's gate can judge them); every other
    checkpoint ('other_generalist' / 'other_specialist'). Ruling R18: a replicate runs its
    final and the two before it only (protocol 8's stability rule); P6's checkpoint history
    stays on seed 0, so 'all' gives a replicate what 'last3' gives."""
    from apexzero.interp import agents

    spec = agents.AGENTS[agent]
    snaps = agents.list_snapshots(spec)
    if not snaps:
        return []
    rel = [str(p.relative_to(ROOT)) for p in snaps]
    out = [(rel[-1], "final")]
    if which in ("last3", "all") and spec.kind == "generalist":
        out += [(p, "last3") for p in rel[-3:-1]]
    if which == "all" and not agents.is_replicate(agent):
        other = "other_generalist" if spec.kind == "generalist" else "other_specialist"
        skip = rel[-3:] if spec.kind == "generalist" else rel[-1:]
        out += [(p, other) for p in rel if p not in skip]
    return out


def _stage_rollouts_post_tag(args) -> None:
    """Step 6: agent rollouts (protocol 4's replication table), one snapshot per job in a
    pool of --workers float32 processes (protocol: 3 x 2 XLA threads). Budget rule 3 (below
    15k steps/s) drops the checkpoint stochastic and the tier E matched-noise runs. Every
    snapshot rolled out is marked evaluated in the manifest. Resumes per (snapshot, group);
    a group on disk that lacks planned runs is run again (``_rollout_job``)."""
    from apexzero.interp import agents, io, stimuli
    from apexzero.interp.constants import MIN_ROLLOUT_SPS

    m = _manifest()
    bank, cells = _design(laid_out=True)
    design = stimuli.design_hash(bank, cells)
    sel = args.agent or [a for a in agents.AGENTS if a in _finals(m)]
    modes = ("det", "nat", "mat") if args.mode == "all" else (args.mode,)
    sps = min(m.budget.get("rollout_sps", {"x": 1e9}).values())
    rule3 = sps < MIN_ROLLOUT_SPS
    jobs = []
    for a in sel:
        for path, kind in _snapshot_kinds(a, args.snapshots):
            jobs.append(dict(label=f"{a} {Path(path).name} {kind}", agent=a, path=path, kind=kind,
                             modes=modes, rule3=rule3, force=bool(args.force),
                             design_hash=design))
    with io.StageLog("rollouts", total=len(jobs), note=f"{args.snapshots} {args.mode}") as log:
        res = _pool(jobs, args.workers, log, fn=_rollout_job, init=_rollout_worker_init)

        def upd(mm):
            for r in res:                   # the jobs mark before running; this is a backstop
                if r["groups"]:
                    agents.mark_evaluated(mm, r["path"], "rollouts")
            cut = "rule 3: checkpoint stochastic and tier E matched-noise runs dropped"
            if rule3 and cut not in mm.budget.setdefault("cuts", []):
                mm.budget["cuts"].append(cut)

        agents.update_manifest(upd)
        for r in res:
            if r.get("rerun"):
                _say(log, f"{r['label']}: re-ran {r['rerun']}: the group on disk lacked "
                          "planned runs (e.g. ruling R19a's tier G on a last-3 snapshot)")
        for e in _errors(res):
            _say(log, f"error: {e}")


# ------------------------------------------------------------------------- step 7: anchored


def _anchored_job(job: dict) -> dict:
    """Re-anchored keys of one (agent, snapshot, cell), one per deterministic phase of
    ``job['phases']`` ((phase, row) pairs; ruling R17a), or viability keys of one (agent,
    snapshot, cell) run, in a key worker. The keys stage solves every anchored cell's u = 0
    nominal key (tier B D = 600's at P1, ruling R17c) and, where its rule applies, the
    phase-matched keys, so the job finds them cached. Solving a missing nominal key here is
    only a fallback (``nominal_inline``; start (3) with ``job['q_cv']``). The stage solves
    each missing one once in a first pool (``what`` 'nominal': the nominal key only), so the
    jobs of several agents do not cold-solve it concurrently; a cell's phases share one job
    for the same reason. ``u0_base_phases`` lists the phases k > 0 whose phase-matched key
    the keys stage solves (``_phase_key_expected`` with ``job['v7_failed']`` and
    ``job['rule2']``, as the pre-flight) but none verified was cached, so they started from
    the u = 0 key (R17a's fallback). A phase that fails does not stop the others (``error``
    lists each failed phase)."""
    import traceback

    from apexzero.interp import answer_key as ak
    from apexzero.interp import measures, rollout

    bank, tables, cache = _W["bank"], _W["tables"], _W["cache"]
    out = dict(label=job["label"], keys=[], cells=[job["cell"].cell_id], what=job["what"])
    c0 = time.process_time()
    try:
        cell = job["cell"]
        if job["what"] not in ("beaten", "nominal"):
            groups, _ = rollout.load_rollouts(Path(job["file"]))
        if job["what"] == "beaten":
            # Protocol 6.3: re-solve warm from every final agent's u = 0 rollout.
            runs = {}
            for a, (f, row) in job["runs"].items():
                g, _ = rollout.load_rollouts(Path(f))
                runs[a] = measures.trajectory_from_rollout(g["det"], int(row), "det")
            key = ak.key_for(cell, job["kind"], 0, cache=cache, bank=bank)
            rec, dev = ak.resolve_beaten_key(bank, cell, key, runs, cache, tables=tables)
            out["keys"] = [_rec_summary(rec)]
            out["deviation"] = dict(dev, agents_beating=job["beating"], category="key",
                                    returns=job["returns"], r18=job.get("r18", {}))
            out["cpu"] = time.process_time() - c0
            return out
        nom = ak.key_for(cell, "nominal", 0, cache=cache, bank=bank)
        if nom is None:
            # Fallback only (R17c): the keys stage solves these, tier B D = 600's included;
            # the stage's first pool ('nominal' jobs) solves each missing one once.
            nom = ak.full_recipe(bank, cell, cache=cache, tables=tables,
                                 q_cv=float(job.get("q_cv") or ak_default_cv()))
            out["nominal_inline"] = True
        if job["what"] == "nominal":
            out["keys"] = [_rec_summary(nom)]
        elif job["what"] == "anchored":
            # Ruling R17a: each phase anchored on its own run, from the key at that phase
            # (the phase-matched key where one is cached, else the u = 0 key). A phase k > 0
            # counts as R17a's fallback only where the keys stage solves a phase-matched key
            # (the pre-flight's rule, ``_phase_key_expected``) and none verified was cached.
            expect = _phase_key_expected(bank, cell, nom, bool(job.get("v7_failed")),
                                         bool(job.get("rule2")))
            phases = job.get("phases") or [(int(job.get("phase", 0)), int(job["row"]))]
            failed = []
            out["phases"] = []
            for k, row in phases:
                k = int(k)
                try:
                    traj = measures.trajectory_from_rollout(groups["det"], int(row), "det")
                    base = ak.anchor_base(cell, k, nom, cache=cache, bank=bank)
                    rec = ak.anchored_key(bank, cell, base, job["agent"], traj, cache,
                                          snapshot=str(job["timesteps"]), phase=k,
                                          tables=tables, force=bool(job.get("force")))
                    out["keys"].append(_rec_summary(rec))
                    out["phases"].append(k)
                    if k and base is nom and expect:
                        out.setdefault("u0_base_phases", []).append(k)
                except Exception:  # noqa: BLE001  (one phase's failure keeps the others)
                    failed.append(f"phase {k}:\n{traceback.format_exc()}")
            if failed:
                out["error"] = "\n".join(failed)
        else:
            traj = measures.trajectory_from_rollout(groups["det"], int(job["row"]), "det")
            bo = ak.key_for(cell, "backoff", 0, cache=cache, bank=bank) or nom
            recs = ak.viability_keys(bank, cell, traj, nom, bo, cache, agent=job["agent"],
                                     snapshot=job["snapshot"], tables=tables,
                                     force=bool(job.get("force")))
            out["keys"] = [_rec_summary(r) for r in recs]
    except Exception:  # noqa: BLE001  (a worker reports every failure to the main process)
        out["error"] = traceback.format_exc()
    out["cpu"] = time.process_time() - c0
    return out


def _u0_rows(det: dict, cells: list) -> dict[int, int]:
    """cell_id -> row of its deterministic phase-0 run."""
    return {c: ph[0] for c, ph in _det_rows(det, cells).items() if 0 in ph}


def _det_rows(det: dict, cells: list) -> dict[int, dict[int, int]]:
    """cell_id -> {phase k: row of its deterministic run at k} (the first such row): the
    runs a snapshot has, every phase at the finals (16 on P7 cells), 4 at the checkpoints."""
    out: dict[int, dict[int, int]] = {}
    for i, (c, k) in enumerate(zip(det["cell"], det["phase"], strict=True)):
        out.setdefault(int(c), {}).setdefault(int(k), i)
    return out


def anchored_jobs(cells: list, file: str, det: dict, agent: str, timesteps: int, prio: str,
                  force: bool = False) -> list[dict]:
    """Ruling R17a: one re-anchored key per (cell, deterministic phase) of a snapshot's run
    file, each anchored on that phase's own run. One job per cell carries its phases'
    (phase, row) pairs in phase order (``phases``), so a nominal key the keys stage did not
    cache (the fallback, R17c) is solved once, not by 4 (16 on P7 cells) concurrent
    workers."""
    rows = _det_rows(det, cells)
    out = []
    for c in cells:
        ph = sorted(rows.get(c.cell_id, {}).items())
        if ph:
            out.append(dict(what="anchored", force=force, phases=ph, agent=agent,
                            timesteps=int(timesteps), cell=c, file=str(file),
                            label=f"RA {agent}@{timesteps} cell {c.cell_id} "
                                  f"x{len(ph)} phases ({prio})"))
    return out


def _phase_key_expected(bank, c, nom, v7_failed: bool, rule2: bool) -> bool:
    """Whether the keys stage solves phase-matched keys (at every phase but u = 0) for a
    re-anchored cell (A1 or tier B D = 600), as ``_phase_split`` and ``_cut`` decide: the core
    and P7 cells always; with V7 failed every cell whose nominal key ``nom`` brakes (2.3:
    a sustained brake and v_min* < v0 - tau); budget rule 2 (``rule2``) keeps 180 deg cells
    only. Elsewhere the u = 0 key is the key at every phase by design, not a fallback."""
    if rule2 and not _rule2_keeps_phase(bank, c):
        return False
    return bool(_core_phase_cell(bank, c) or _p7_cell(c) or (v7_failed and _braking(nom, c)))


def _anchored_preflight(bank, jobs: list[dict], cache, v7_failed: bool, cuts=()) -> dict:
    """Ruling R17c (and R17b's order of stages): the keys the re-anchored jobs start from
    should be cached by the keys stage before ``anchored-keys`` runs: every job cell's u = 0
    nominal key (tier B D = 600's included, solved at P1 since R17c) and, at each of the
    job's phases k > 0, the verified phase-matched key the keys stage solves there
    (``_phase_key_expected``; budget rule 2 read from the manifest's ``cuts``). Nothing is
    solved or written here, and a missing key is not fatal: the stage solves each missing
    nominal key once before the jobs (the fallback) and the worker anchors a phase without
    its key from the u = 0 key (R17a's fallback; ``anchored_key`` re-solves it from the phase
    key once one is cached). ``viability_nominal_missing`` lists the viability jobs' other
    cells without a cached nominal key, solved the same way but not an R17c fallback (e.g.
    tier B D = 250, which no keys-stage item solves)."""
    from apexzero.interp import answer_key as ak

    rule2 = any(str(c).startswith("rule 2") for c in cuts)
    by_cell: dict[int, tuple] = {}
    for j in jobs:
        if j.get("what") != "anchored":
            continue
        c = j["cell"]
        by_cell.setdefault(c.cell_id, (c, set()))[1].update(
            int(k) for k, _ in j.get("phases") or [] if int(k) > 0)
    nominal_missing, phase_missing = [], []
    for cid, (c, ks) in sorted(by_cell.items()):
        nom = ak.key_for(c, "nominal", 0, cache=cache, bank=bank)
        if nom is None:
            nominal_missing.append(cid)
            continue
        if not _phase_key_expected(bank, c, nom, v7_failed, rule2):
            continue
        for k in sorted(ks):
            rec = ak.key_for(c, "phase", k, cache=cache, bank=bank)
            if rec is None or rec.status != "verified":
                phase_missing.append((cid, k))
    via = {j["cell"].cell_id: j["cell"] for j in jobs
           if j.get("what") == "viability" and j["cell"].cell_id not in by_cell}
    return dict(cells=len(by_cell), nominal_missing=nominal_missing,
                b600_nominal_missing=[i for i in nominal_missing if _b600_cell(by_cell[i][0])],
                phase_missing=phase_missing,
                b600_phase_missing=[(i, k) for i, k in phase_missing
                                    if _b600_cell(by_cell[i][0])],
                viability_nominal_missing=[i for i, c in sorted(via.items()) if ak.key_for(
                    c, "nominal", 0, cache=cache, bank=bank) is None])


def anchored_cells(cells: list, bank, regimes: dict, agent: str, prio: str,
                   generalist: bool, kappa_clipped: set[int] | None = None) -> list:
    """The cells that get re-anchored keys (protocol 6.1, 10 step 7): P0, a generalist's
    final: A1 tail and bulk cells and, by ruling R12, the non-clipped below-support A1 cells,
    so P1b's Kendall tau (and P2's set O) read e_ve^RA throughout A1; P1, a specialist's
    final: A1 and tier B D = 600 cells, and a generalist's final: tier B D = 600 cells (A1
    done at P0); P2, a generalist's checkpoint: A1 tail and bulk cells (P6c).

    Below support is R_seg < q1 (regimes 'below' and 'clip'). "Non-clipped" is the binding
    run-time definition (protocol 2.3, AD9: no deterministic run of the pair drove a preview
    kappa channel to the clip) when ``kappa_clipped`` (the agent's kappa-clipped pair ids)
    is given, else the design-time one (regime 'below', R_seg > R_clip^max). Budget rule 2's
    'no_key' cells get no key (R7)."""
    from apexzero.interp import support
    from apexzero.interp.analysis import has_no_key

    rt = regimes.get(agent)
    out = []
    for c in cells:
        if has_no_key(c):
            continue
        st = bank.stimuli[c.stim_id]
        a1 = c.approach == "A1" and "A" in c.tiers
        b600 = _b600_cell(c)
        reg = support.regime(st.geometry.R_seg, rt) if rt is not None else "bulk"
        if prio == "P0":
            if kappa_clipped is None:
                below = reg == "below"
            else:
                below = reg in ("below", "clip") and int(c.pair_id) not in kappa_clipped
            want = generalist and a1 and (reg in ("tail", "bulk") or below)
        elif prio == "P2":
            want = generalist and a1 and reg in ("tail", "bulk")
        else:
            want = b600 if generalist else (a1 or b600)
        if want:
            out.append(c)
    return out


def select_p6_checkpoints(timesteps: list[int], gate: dict[int, bool]) -> list[int]:
    """P6c's four checkpoints of a generalist run: the first that passes the gate, two
    log-spaced in timesteps between it and the final, and the final (fewer when fewer
    exist). [] when no checkpoint passes the gate."""
    ts = sorted(int(t) for t in timesteps)
    if not ts:
        return []
    first = next((t for t in ts if gate.get(t)), None)
    if first is None:
        return []
    final = ts[-1]
    out = [first]
    inner = [t for t in ts if first < t < final]
    if inner and first > 0:
        for k in (1, 2):
            target = math.exp(math.log(first) + (math.log(final) - math.log(first)) * k / 3)
            left = [t for t in inner if t not in out]
            if left:
                out.append(min(left, key=lambda t: abs(math.log(t) - math.log(target))))
    if final not in out:
        out.append(final)
    return sorted(set(out))


def _replicate_regime_problems(sup, agents_in) -> list[str]:
    """R18 replicates among ``agents_in`` without frozen regimes that carry R_clip (from their
    own obs_rms, support stage): one rule for the anchored-keys stage (which refuses) and the
    analysis stage (which refuses under --confirmatory and logs otherwise). A regime without
    R_clip has no 'clip' set, so ``anchored_cells`` and P1b / P2 would read clip-saturated
    cells as below support."""
    from apexzero.interp import agents

    out = []
    for a in [x for x in agents.replicates() if x in agents_in]:
        r = sup.regimes.get(a)
        if r is None:
            out.append(f"R18 replicate {a} has no frozen regimes")
        elif not (r.r_clip and np.isfinite(r.r_clip_max)):
            out.append(f"R18 replicate {a} has frozen regimes without R_clip")
    return out


def _beaten_jobs(bank, cells, finals: dict, cache) -> tuple[list[dict], dict]:
    """Protocol 6.3: per cell with a verified u = 0 key (nominal; null on tier G), each
    final's deterministic u = 0 return; cells where an agent beats the key by > 1e-6 |J*|
    become re-solve jobs warm from every final's u = 0 run. Ruling R18's replicates are
    final agents, so they count both ways (6.3's "any agent", "every final agent"); a key
    is shared, so its re-solve reaches G1-G3's errors too. Each job, its deviation and the
    summary record the replicates' part (``r18``: beaten by replicates only; replicate
    starts), so any change to G1-G3's keys that the replicates caused is listed."""
    from apexzero.interp import agents, io, measures, rollout
    from apexzero.interp import answer_key as ak

    runs: dict[int, dict[str, tuple[str, int, float]]] = {}
    for a, e in finals.items():
        f = io.rollout_path(a, e.timesteps)
        if not f.exists():
            continue
        det = rollout.load_rollouts(f)[0].get("det")
        if not det:
            continue
        for cid, i in _u0_rows(det, cells).items():
            traj = measures.trajectory_from_rollout(det, i, "det")
            runs.setdefault(cid, {})[a] = (str(f), i, measures.discounted_return(
                traj, int(det["W"][i])))
    by = {c.cell_id: c for c in cells}
    jobs, summary = [], {"cells_checked": 0, "beaten": {}, "beaten_by_replicates_only": []}
    for cid, per in sorted(runs.items()):
        c = by[cid]
        kind = "null" if "G" in c.tiers else "nominal"
        key = ak.key_for(c, kind, 0, cache=cache, bank=bank)
        if key is None or key.status != "verified":
            continue
        summary["cells_checked"] += 1
        beat = ak.agent_beats_key(key, {a: v[2] for a, v in per.items()})
        if beat:
            summary["beaten"][cid] = beat
            r18 = dict(beaten_by_replicates_only=all(agents.is_replicate(a) for a in beat),
                       replicate_starts=sorted(a for a in per if agents.is_replicate(a)))
            if r18["beaten_by_replicates_only"]:
                summary["beaten_by_replicates_only"].append(cid)
            jobs.append(dict(what="beaten", label=f"6.3 re-solve cell {cid} ({','.join(beat)})",
                             cell=c, kind=kind, beating=beat, r18=r18,
                             returns={a: v[2] for a, v in per.items()} | {"J_star": key.J},
                             runs={a: (v[0], v[1]) for a, v in per.items()}))
    return jobs, summary


def stage_anchored_keys(args: argparse.Namespace) -> None:
    """Step 7 (post-tag; never alongside rollouts at full width): first protocol 6.3's
    'agent beats the key' re-solves on the finals' u = 0 returns (a deviation is logged per
    re-solved cell); then re-anchored keys (W = 400, protocol 6.1; ruling R17a: one per
    deterministic phase, each from the agent's own run at that phase and the key at that
    phase, ``anchored_jobs``) and viability solves (9.1: crashed u = 0 runs, or the first
    crashed phase; at most 200 solves, a seeded subset beyond the cap). P0: A1 tail, bulk
    and non-clipped below-support cells of the generalist finals (R12; clipping from each
    final's runs), plus viability; P1: the specialists' finals (A1 and
    tier B D = 600) and the generalist finals' tier B D = 600 cells; P2: the generalists'
    P6c checkpoints (first gate-passing, two log-spaced middle; the final is P0), skipped
    when the finals show re-anchoring is immaterial or budget rule 1 cut P2. Ruling R18: the
    replicates' finals get G1-G3's P0 and P1 keys (their own runs and regimes; never P2's
    checkpoint keys, P6 stays on seed 0) and their own viability solves, 9.1's cap applied
    to them apart so G1-G3's seeded subset is unchanged; a replicate needs its own frozen
    regimes (support stage). Ruling R17c:
    the nominal keys (tier B D = 600's included) and phase-matched keys the jobs start from
    come from the keys stage; ``_anchored_preflight`` reports any not cached. Solving a
    missing nominal key here is a fallback, once per cell in a pool before the jobs
    (``nominal_inline`` in the status; viability cells apart, ``viability_nominal_inline``);
    ``u0_base_phases`` counts the phases anchored from the u = 0 key where the keys stage
    solves a phase-matched key. Pool of --workers (default 5; protocol 10)."""
    from apexzero.interp import agents, io, measures, rollout, support
    from apexzero.interp import analysis as A
    from apexzero.interp import answer_key as ak

    _require_tag("anchored keys (they depend on agent rollouts)")
    bank, cells = _design(laid_out=True)
    sup = support.load_support(io.SUPPORT_DIR)
    m = _manifest()
    finals = _finals(m)
    gen = [a for a in agents.seed0_generalists() if a in finals]
    reps = [a for a in agents.replicates() if a in finals]          # R18
    spec = [a for a in agents.specialists() if a in finals]
    no_reg = _replicate_regime_problems(sup, finals)
    if no_reg:
        raise SystemExit("; ".join(no_reg) + ": run `psychophysics.py support` (it adds every "
                         "final's obs_rms)")
    tables = measures.model_tables()
    cache = ak.KeyCache()
    force = bool(args.force)
    with io.StageLog("anchored-keys", total=1, note=f"{args.priority}, {args.workers} workers") as log:
        # 6.3 first: anchored keys warm from the nominal key, which a re-solve may replace.
        bjobs, bsum = _beaten_jobs(bank, cells, finals, cache)
        _say(log, f"6.3: {bsum['cells_checked']} keys checked against the finals' u = 0 "
                  f"returns; beaten on {len(bsum['beaten'])} cells"
                  + (f" ({len(bsum.get('beaten_by_replicates_only', []))} by R18 replicates only: "
                     "their re-solve also reaches G1-G3's errors)"
                     if bsum.get("beaten_by_replicates_only") else ""))
        res_b = _pool(bjobs, args.workers, log, fn=_anchored_job) if bjobs else []
        devs = [r["deviation"] for r in res_b if r.get("deviation")]
        if devs:
            agents.update_manifest(lambda mm: mm.deviations.extend(devs))
        targets: list[tuple[str, object, str]] = []        # (agent, snapshot entry, which)
        targets += [(a, finals[a], "P0") for a in gen + reps]
        if args.priority in ("P1", "P2"):
            targets += [(a, finals[a], "P1") for a in spec + gen + reps]
        p6_note = ""
        if args.priority == "P2":
            keys = A.CachedKeys(bank, cells, cache, tables=tables)
            kt = A.key_table(cells, bank, keys)
            fin_pairs = [A.pair_values(_run_table(a, finals[a].timesteps, rollout.load_rollouts(
                io.rollout_path(a, finals[a].timesteps)), cells, bank, keys, tables, m), kt,
                sup.regimes) for a in gen if io.rollout_path(a, finals[a].timesteps).exists()]
            imm = A.reanchor_immaterial(fin_pairs)
            cut = any(str(c).startswith("rule 1") for c in m.budget.get("cuts", []))
            if imm["immaterial"]:
                p6_note = f"re-anchoring immaterial on the finals ({imm}): nominal keys for P6"
            elif cut:
                p6_note = "budget rule 1 dropped the P2 items: no checkpoint re-anchoring"
            else:
                for a in gen:
                    snaps = [x for x in m.snapshots if x.agent == a and not x.error]
                    gate = {}
                    for x in snaps:
                        f = io.rollout_path(a, x.timesteps)
                        if f.exists():
                            t = _run_table(a, x.timesteps, rollout.load_rollouts(f), cells, bank,
                                           keys, tables, m)
                            gate[x.timesteps] = A.checkpoint_gate(
                                A.pair_values(t, kt, sup.regimes))["passed"]
                    chosen = select_p6_checkpoints([x.timesteps for x in snaps], gate)
                    targets += [(a, x, "P2") for x in snaps
                                if x.timesteps in chosen and not x.is_final]
                    _say(log, f"P6c checkpoints {a}: {chosen}")
            if p6_note:
                _say(log, p6_note)
        # The generalist finals' run tables: their kappa-clipped pairs fix P0's non-clipped
        # below-support A1 cells (R12, run-time definition), and their crashed u = 0 runs get
        # viability solves (9.1).
        keys = A.CachedKeys(bank, cells, cache, tables=tables)
        kt = A.key_table(cells, bank, keys)
        final_rt: dict[str, dict] = {}
        for agent in gen + reps:
            f = io.rollout_path(agent, finals[agent].timesteps)
            if f.exists():
                groups, _ = rollout.load_rollouts(f)
                final_rt[agent] = _run_table(agent, finals[agent].timesteps, groups, cells,
                                             bank, keys, tables, m)
        jobs = []
        for agent, e, prio in targets:
            f = io.rollout_path(agent, e.timesteps)
            if not f.exists():
                continue
            groups, _ = rollout.load_rollouts(f)
            if "det" not in groups:
                continue
            clipped = (A.kappa_clipped_pairs(final_rt[agent])
                       if prio == "P0" and agent in final_rt else None)
            # R17a: one key per deterministic phase of each cell, anchored on that run.
            jobs += anchored_jobs(anchored_cells(cells, bank, sup.regimes, agent, prio,
                                                 agents.is_generalist(agent), clipped),
                                  str(f), groups["det"], agent, e.timesteps, prio, force)
        # Viability solves on the generalist finals' crashed u = 0 runs (9.1), within the cap:
        # G1-G3's as before, R18's replicates apart under their own cap.
        by = {c.cell_id: c for c in cells}
        for family in (gen, reps):
            vt_rows = [final_rt[a] for a in family if a in final_rt]
            if not vt_rows:
                continue
            vt = A.viability_targets(io.concat_tables(vt_rows), kt)
            for i in range(A._n(vt)):
                agent, ts = str(vt["agent"][i]), int(vt["timesteps"][i])
                f = io.rollout_path(agent, ts)
                groups, _ = rollout.load_rollouts(f)
                det = groups["det"]
                cid, ph = int(vt["cell"][i]), int(vt["phase"][i])
                row = next(j for j in range(len(det["cell"])) if int(det["cell"][j]) == cid
                           and int(det["phase"][j]) == ph)
                # The snapshot tag carries the run's phase, so analysis can find the run's
                # records (viability_keys' records are not indexed by phase).
                jobs.append(dict(what="viability", force=force,
                                 label=f"viability {agent}@{ts} cell {cid} ph{ph}",
                                 agent=agent, timesteps=ts, cell=by[cid], file=str(f), row=row,
                                 snapshot=f"{ts}-ph{ph}"))
        # R17c: the keys stage solved the nominal (tier B D = 600's included) and phase-matched
        # keys these jobs start from; report any it did not (the jobs fall back).
        v7, cuts = _v7_failed(m), m.budget.get("cuts", [])
        rule2 = any(str(c).startswith("rule 2") for c in cuts)
        pre = _anchored_preflight(bank, jobs, cache, v7, cuts)
        if pre["nominal_missing"] or pre["phase_missing"]:
            _say(log, f"R17c pre-flight: {len(pre['nominal_missing'])} of {pre['cells']} "
                      f"re-anchored cells have no cached nominal key "
                      f"({len(pre['b600_nominal_missing'])} tier B D = 600) and "
                      f"{len(pre['phase_missing'])} (cell, phase) pairs "
                      f"({len(pre['b600_phase_missing'])} tier B D = 600) no verified "
                      f"phase-matched key where the keys stage solves one. The keys stage "
                      f"should have solved them before the tag (`keys --priority P1` or P2; "
                      f"after the tag it runs only with --post-tag-deviation). Fallbacks: each "
                      f"missing nominal key is solved once, in a pool before the jobs; those "
                      f"phases are anchored from the u = 0 key (re-solved from the phase key "
                      f"once one is cached)")
        else:
            _say(log, f"R17c pre-flight: the nominal and phase-matched keys of all "
                      f"{pre['cells']} re-anchored cells are cached")
        if pre["viability_nominal_missing"]:
            _say(log, f"{len(pre['viability_nominal_missing'])} viability cells have no cached "
                      f"nominal key (not an R17c fallback: e.g. tier B D = 250, which no "
                      f"keys-stage item solves); solved once each before the jobs")
        q_cv = _tuned_cv()
        jobs = [dict(j, q_cv=q_cv, v7_failed=v7, rule2=rule2) for j in jobs]
        # One solve per missing nominal key (review of R17c, finding 5): at P1 a tier B
        # D = 600 cell has a job from each of up to 6 finals, which would otherwise each
        # cold-solve it at once. The jobs then find it cached.
        by_id = {j["cell"].cell_id: j["cell"] for j in jobs}
        njobs = [dict(what="nominal", cell=by_id[i], q_cv=q_cv,
                      label=f"nominal key cell {i} (missing: solved before its jobs)")
                 for i in pre["nominal_missing"] + pre["viability_nominal_missing"]]
        log.set_total(log.done + len(njobs) + len(jobs))
        res_n = _pool(njobs, args.workers, log, fn=_anchored_job) if njobs else []
        res = _pool(jobs, args.workers, log, fn=_anchored_job)
        solved = {c for r in res_n + res if r.get("nominal_inline") for c in r["cells"]}
        ra_cells = {j["cell"].cell_id for j in jobs if j["what"] == "anchored"}
        inline, via_inline = sorted(solved & ra_cells), sorted(solved - ra_cells)
        u0_base = sum(len(r.get("u0_base_phases", [])) for r in res)
        if inline or u0_base:
            _say(log, f"fallbacks: {len(inline)} re-anchored cells' nominal keys solved here; "
                      f"{u0_base} re-anchored keys at k > 0 started from the u = 0 key where "
                      f"the keys stage solves a phase-matched key (none cached)")
        if via_inline:
            _say(log, f"viability: {len(via_inline)} cells' nominal keys solved here")
        status = dict(priority=args.priority, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                      jobs=len(jobs), errors=len(_errors(res_n + res)), beaten=bsum["beaten"],
                      p6=(p6_note or "checkpoints re-anchored") if args.priority == "P2" else "",
                      preflight={k: (v if k == "cells" else len(v)) for k, v in pre.items()},
                      nominal_inline=inline, viability_nominal_inline=via_inline,
                      u0_base_phases=u0_base)

        def upd(mm):
            mm.lists["anchored_errors"] = _errors(res_n) + _errors(res) + _errors(res_b)
            mm.validation.setdefault("anchored_keys", {})[args.priority] = status

        agents.update_manifest(upd)


# ------------------------------------------------------------------------- step 6: probes


def _probe_agent(agent: str, entry, bank, cells, sup, kt, log) -> dict:
    """Tier O scans and on-policy aliasing of one final (post-tag)."""
    from apexzero.interp import analysis as A
    from apexzero.interp import io, measures, probe, rollout, stimuli

    ag = _load(entry)
    pol = rollout.agent_policy(ag.cfg)
    f = io.rollout_path(agent, ag.timesteps)
    if not f.exists():
        return {"error": f"no rollouts for {agent}"}
    groups, _ = rollout.load_rollouts(f)
    det = groups["det"]
    tables = measures.model_tables()
    # d_b^agent(v): the agent's median deterministic onset distance on tier A / B cells of
    # the same geometry at the approach speed nearest v (module docstring of probe.py).
    rt = _run_table(agent, ag.timesteps, groups, cells, bank, A.CachedKeys(bank, cells), tables)
    kj = A._join_keys(rt, kt, ("R_nom", "theta_deg", "v0"))
    scans = []
    for spec, v in stimuli.tier_o_specs(sup):
        same = (np.isclose(kj["R_nom"].astype(float), spec.R)
                & np.isclose(kj["theta_deg"].astype(float), spec.theta_deg)
                & (A._st(rt, "mode") == "det") & np.isfinite(A._fl(rt, "d_b")))
        if same.any():
            v0 = kj["v0"].astype(float)[same]
            near = same & np.isclose(kj["v0"].astype(float), v0[np.argmin(np.abs(v0 - v))])
            d_b = A._median(A._fl(rt, "d_b")[near])
        else:
            d_b = float(tables.d_b_phys(v, min(v, float(tables.v_ss(spec.R)))))
        sr = probe.open_loop_scan(pol, ag.actor_params, ag.obs_rms, spec, v, d_b + 50.0)
        scans.append(dict(agent=agent, R=spec.R, theta_deg=spec.theta_deg, v=v, d_min=d_b + 50.0,
                          Pi=sr.Pi, A=sr.A, n=len(sr.d)))
    alias = []
    rows = _u0_rows(det, cells)
    by = {c.cell_id: c for c in cells}
    for cid, i in rows.items():
        c = by[cid]
        if "p7" not in c.flags or "A" not in c.tiers:
            continue
        traj = measures.trajectory_from_rollout(det, i, "det")
        mk = measures.make_marks(bank.stimuli[c.stim_id], bank.w_left[c.stim_id],
                                 bank.w_right[c.stim_id], float(det["s0"][i]), int(det["W"][i]))
        t_b = float(measures.response_metrics(traj, mk, tables=tables).t_b)
        a = probe.aliasing_amplitude(pol, ag.actor_params, ag.obs_rms, bank.stimuli[c.stim_id].spec,
                                     traj, t_b, prev_action=det["prev_action"][i],
                                     accel=det["accel"][i])
        alias.append(dict(agent=agent, cell=cid, A_alias=a))
    log.step(note=f"{agent}: {len(scans)} scans, {len(alias)} aliasing runs")
    return dict(scans=scans, alias=alias)


def stage_probes(args: argparse.Namespace) -> None:
    """Step 6 (post-tag): tier O open-loop scans over d in [d_b^agent(v) + 50, d_max] and the
    on-policy aliasing amplitude on the P7 cells' u = 0 runs, per final; Q's and Qp's
    aliasing from their control rollouts. Writes probes/{scans,alias}.npz."""
    from apexzero.interp import analysis as A
    from apexzero.interp import io, support

    _require_tag("probes (they read agent rollouts)")
    io.assert_x64_off()
    bank, cells = _design(laid_out=True)
    sup = support.load_support(io.SUPPORT_DIR)
    m = _manifest()
    finals = _finals(m)
    sel = args.agent or sorted(finals)
    kt = io.load_table(io.ANALYSIS_DIR / "key_table.npz") if (io.ANALYSIS_DIR / "key_table.npz").exists() \
        else A.key_table(cells, bank, A.CachedKeys(bank, cells))
    scans, alias = [], []
    with io.StageLog("probes", total=len(sel) + 1) as log:
        for a in sel:
            r = _probe_agent(a, finals[a], bank, cells, sup, kt, log)
            scans += r.get("scans", [])
            alias += r.get("alias", [])
        alias += _control_alias(bank, cells, log)
        # --agent re-probes some agents: keep every other agent's rows (and Q's / Qp's only
        # from this run, which always recomputes them).
        redone = set(sel) | {"Q", "Qp"}
        scan_cols = ("agent", "R", "theta_deg", "v", "d_min", "Pi", "A", "n")
        for name, rows, cols in (("scans", scans, scan_cols),
                                 ("alias", alias, ("agent", "cell", "A_alias"))):
            f = io.PROBES_DIR / f"{name}.npz"
            new = A._to_table(rows, cols)
            if f.exists() and args.agent:
                old = io.load_table(f)
                keep = io.select(old, ~np.isin(A._st(old, "agent"), sorted(redone)))
                new = io.concat_tables([keep, new]) if A._n(keep) else new
            io.save_table(f, new)


def _control_alias(bank, cells, log) -> list[dict]:
    """A_alias of Q and Qp on the P7 cells, from their own control rollouts (C18: shifted
    strips get Q's target rows from ``q_params_for_kappa``)."""
    from apexzero.interp import controls, io, measures, probe, rollout

    tun = io.read_json(io.CONTROLS_DIR / "tuning.json")
    car = controls.car_tables(measures.model_tables())
    out = []
    for name in ("Q", "Qp"):
        f = rollout_path_of(CONTROL_AGENT.format(name))
        if not f.exists():
            continue
        groups, _ = rollout.load_rollouts(f)
        det = groups["det"]
        by = {c.cell_id: c for c in cells}
        pol = controls.make_q_policy(name)
        for cid, i in _u0_rows(det, cells).items():
            c = by[cid]
            if "p7" not in c.flags:
                continue
            spec = bank.stimuli[c.stim_id].spec
            traj = measures.trajectory_from_rollout(det, i, "det", source="control")
            mk = measures.make_marks(bank.stimuli[c.stim_id], bank.w_left[c.stim_id],
                                     bank.w_right[c.stim_id], float(det["s0"][i]), int(det["W"][i]))
            t_b = float(measures.response_metrics(traj, mk).t_b)

            def params_fn(kappa, spec=spec, name=name):
                prm = controls.q_params_for_kappa(kappa, np.full(len(kappa), spec.s_c), car,
                                                  float(tun["c_v"]), name)
                return prm, np.arange(len(kappa))

            a = probe.aliasing_amplitude(pol, None, None, spec, traj, t_b, params_fn=params_fn,
                                         prev_action=det["prev_action"][i], accel=det["accel"][i])
            out.append(dict(agent=name, cell=cid, A_alias=a))
        log.step(note=f"{name} aliasing")
    return out


# ------------------------------------------------------------------------- step 8: analysis


def _run_table_cached(agent: str, ts: int, bank, cells, keys, tables, force: bool,
                      inputs_hash: str = "", speed_at_phase: bool = False):
    """An analysis run table, cached under analysis/runs/ with the hash of what it was built
    from (the rollout file, the key cache, the design and the code: ``inputs_hash`` plus the
    file's own hash, and AD23's speed-error keys, ``speed_at_phase`` from the V7 verdict);
    rebuilt whenever any of them changed, or with ``force``."""
    from apexzero.interp import analysis as A
    from apexzero.interp import io, rollout

    p = io.ANALYSIS_DIR / "runs" / f"{agent}_{int(ts):012d}.npz"
    f = io.rollout_path(agent, ts)
    if not f.exists():
        return None
    st = f.stat()
    stamp = io.sha256_json(dict(inputs=inputs_hash, rollout=[str(f), st.st_size, st.st_mtime_ns],
                                speed_at_phase=bool(speed_at_phase)))
    side = p.with_suffix(".json")
    if p.exists() and side.exists() and not force and io.read_json(side).get("stamp") == stamp:
        return io.load_table(p)
    groups, _ = rollout.load_rollouts(f)
    rt = A.run_table(agent, ts, groups, cells, bank, keys, tables,
                     speed_at_phase=bool(speed_at_phase))
    io.save_table(p, rt)
    io.write_json(side, dict(stamp=stamp, rollout=str(f.relative_to(ROOT)),
                             built=time.strftime("%Y-%m-%dT%H:%M:%S")))
    return rt


def _history_entries(m, agent: str) -> list:
    """The manifest's two snapshots before ``agent``'s final (protocol 8's stability rule;
    protocol 4's 'last 3 checkpoints' rows), in timestep order."""
    snaps = sorted((e for e in m.snapshots if e.agent == agent and not e.error),
                   key=lambda e: e.timesteps)
    fin = [i for i, e in enumerate(snaps) if e.is_final]
    return snaps[max(0, fin[-1] - 2):fin[-1]] if fin else []


def confirmatory_problems(m, finals: dict, have_groups: dict[str, set[str]],
                          agent_arg, history_groups: dict[str, list[set[str]]],
                          history_tier_g: dict[str, list[int]]) -> list[str]:
    """Why the one-shot confirmatory analysis may not run yet (protocol 8, 10 step 8): it
    is spent on complete inputs only, which since ruling R18 include the six replicates'
    finals (their replication family is written by the same one run). ``history_groups``:
    per generalist final, the rollout groups of each of the manifest's two snapshots before
    it (``_history_entries``); protocol 8's 2-of-last-3 stability rule reads them, so each
    needs its det group, and its nat group unless budget rule 3 dropped the checkpoint
    stochastic runs. Without them every claim would stand unchecked. ``history_tier_g``:
    per generalist final, for the same snapshots, how many of their planned tier G
    deterministic runs their det group lacks (``_tier_g_missing``). Ruling R19a: protocol
    8's gate judges each history snapshot on its gentle cells and tier G width-12 straights,
    so a det group without them would make every claim at the final 'unstable'."""
    from apexzero.interp import agents

    out = []
    if agent_arg:
        out.append("--agent is not allowed with --confirmatory (it reads every final)")
    every = tuple(agents.AGENTS)      # S1-S3, G1-G3 and R18's G1s1 .. G3s2
    missing = [a for a in every if a not in finals]
    if missing:
        out.append(f"finals missing from the manifest: {missing}")
    for a in every:
        if a in finals and not {"det", "nat", "mat"} <= have_groups.get(a, set()):
            out.append(f"{a}: rollout groups {sorted(have_groups.get(a, set()))}, need det, nat, mat")
    rule3 = any(str(c).startswith("rule 3") for c in m.budget.get("cuts", []))
    need = {"det"} if rule3 else {"det", "nat"}
    for a in [x for x in agents.generalists() if x in finals]:
        hist = history_groups.get(a, [])
        if len(hist) < 2:
            out.append(f"{a}: {len(hist)} of the 2 snapshots before the final in the manifest "
                       "(protocol 8's stability rule needs both)")
        tier_g = history_tier_g.get(a, [])
        for i, g in enumerate(hist):
            if not need <= g:
                out.append(f"{a}: snapshot {i - len(hist)} before the final has rollout groups "
                           f"{sorted(g)}, need {', '.join(sorted(need))} (rollouts --snapshots "
                           "last3; protocol 8's stability rule)")
            elif i >= len(tier_g) or tier_g[i]:
                what = (f"lacks {tier_g[i]} of its tier G deterministic runs" if i < len(tier_g)
                        else "was not checked for its tier G deterministic runs")
                out.append(f"{a}: snapshot {i - len(hist)} before the final {what} (ruling "
                           "R19a: protocol 8's gate needs the null straights; rollouts "
                           "--snapshots last3 runs them)")
    if m.band is None:
        out.append("no band b from V8 in the manifest")
    for k in ("step2", "step3", "dry_run"):
        if not (m.validation.get(k) or {}).get("passed"):
            out.append(f"{k} has not passed")
    ctl = m.validation.get("controls") or {}
    bad = [k for k, v in ctl.items() if (v or {}).get("status") != "pass"]
    if not ctl or bad:
        out.append(f"controls not all passed: {bad or 'none recorded'}")
    ak_ = m.validation.get("anchored_keys") or {}
    if not any(p in ak_ for p in ("P1", "P2")):
        out.append("step 7 anchored keys (P0 and P1) have not run (anchored-keys --priority P1)")
    elif any((v or {}).get("errors") for v in ak_.values()):
        out.append("step 7 anchored keys had errors (manifest lists.anchored_errors)")
    return out


def _p8_units(bank, cells) -> tuple[list[dict], list[dict]]:
    """P8's crash units (one per CrashPair, with the E1 cells built from that agent's state)
    and the twin-transfer corners (E1 and E2, every state pooled). A unit's set is the regime
    of its crash corner's R_seg under the crashing agent's frozen q1 / q10 (ruling R2); a
    corner carries its twin's length shortfall (R4)."""
    from apexzero.interp import analysis as A
    from apexzero.interp import io, stimuli, support

    regimes = support.load_support(io.SUPPORT_DIR).regimes
    units, corners = [], {}
    for p in stimuli.crash_pairs():
        c = stimuli.crash_corner(stimuli.real_track(p.track), p.s_crash)
        key = stimuli.crash_corner_key(p.track, c)
        mine = [x for x in cells if "E1" in x.tiers and x.state_source.startswith(f"{p.agent}:")
                and x.state_source.endswith(f"@{key}")]
        vers: dict[str, list[int]] = {}
        for x in mine:
            vers.setdefault(_version(bank, x), []).append(x.cell_id)
        units.append(dict(agent=p.agent, unit=f"{p.agent}/{p.track}/{p.s_crash:.0f}",
                          set=A.p8_set(p.agent, float(c.r_min), regimes), R_seg=float(c.r_min),
                          cells=vers, key=key))
    for x in cells:
        if not ({"E1", "E2"} & set(x.tiers)):
            continue
        name = bank.stimuli[x.stim_id].name
        base = name.rsplit("/", 1)[0]
        d = corners.setdefault(base, dict(corner=base, source="E1" if "E1" in x.tiers else "E2",
                                          cells={}, R_seg=np.nan, theta_deg=np.nan,
                                          length_shortfall=np.nan))
        d["cells"].setdefault(_version(bank, x), []).append(x.cell_id)
        if _version(bank, x) == "twin":
            st = bank.stimuli[x.stim_id]
            g = st.geometry
            d["R_seg"], d["theta_deg"] = float(g.R_seg), float(abs(g.theta_seg_deg))
            d["length_shortfall"] = stimuli.twin_length_shortfall(st)
    return units, [c for c in corners.values() if "twin" in c["cells"]]


def _fidelity(bank, cells, rt, units) -> dict[str, bool | None]:
    """3.2's fidelity gate per unit: the real rung's deterministic u = 0 run from the
    crashing agent's own state matches its real lap to |dv| <= 0.1 m/s up to s_c - 100 m
    (real laps saved by the Integrate phase under stimuli/real_laps/). None when the gate
    cannot be evaluated (no real rung, real lap, state index entry or u = 0 run): missing
    data is not a failed gate, so the unit stays in P8 as unreproduced and is listed
    (``analysis.p8``), where a failed gate excludes it (3.2)."""
    from apexzero.interp import analysis as A
    from apexzero.interp import io

    out: dict[str, bool | None] = {}
    fi = io.STIMULI_DIR / "real_states_index.json"
    index = io.read_json(fi) if fi.exists() else {}
    for u in units:
        ids = u["cells"].get("real", [])
        lap_f = io.STIMULI_DIR / "real_laps" / f"{u['agent']}_{u['unit'].split('/')[1]}.npz"
        k0 = index.get(u["agent"], {}).get(u["key"])
        if not ids or not lap_f.exists() or k0 is None:
            out[u["unit"]] = None
            continue
        cell = next(c for c in cells if c.cell_id == ids[0])
        m = (A._it(rt, "cell") == cell.cell_id) & (A._st(rt, "mode") == "det") \
            & (A._it(rt, "phase", 0) == 0)
        if not m.any():
            out[u["unit"]] = None
            continue
        with np.load(lap_f) as z:
            lap = {k: z[k] for k in z.files}
        lap["k0"] = np.asarray(int(k0))
        out[u["unit"]] = _fidelity_one(bank, cell, rt, int(np.flatnonzero(m)[0]), lap)
    return out


def _fidelity_one(bank, cell, rt, row: int, lap: dict) -> bool:
    """|vx_strip(d) - vx_real(d)| <= 0.1 m/s over the distance travelled from the start state
    up to s_c - 100 (the transplant copies knots index for index, so distance on the strip is
    distance on the circuit scaled by ds_source / 2; protocol 3.2, decision D2)."""
    from apexzero.interp import io, rollout

    st = bank.stimuli[cell.stim_id]
    f = io.rollout_path(str(rt["agent"][row]), int(rt["timesteps"][row]))
    groups, _ = rollout.load_rollouts(f)
    det = groups["det"]
    j = next(i for i in range(len(det["cell"])) if int(det["cell"][i]) == cell.cell_id
             and int(det["phase"][i]) == 0)
    v = det["valid"][j]
    s, vx = det["x"][j][v, 0].astype(float), det["x"][j][v, 3].astype(float)
    k0 = int(lap["k0"][()]) if "k0" in lap else 0
    ds_src = float(lap["ds"][()]) if "ds" in lap else 2.0
    sr, vr = lap["x"][k0:, 0].astype(float), lap["x"][k0:, 3].astype(float)
    d_strip, d_real = s - s[0], (sr - sr[0]) * 2.0 / ds_src
    hi = st.landmarks.s_c0 - 100.0 - s[0]
    grid = np.arange(0.0, max(hi, 0.0), 1.0)
    if grid.size == 0 or d_strip[-1] < hi or d_real[-1] < hi:
        return False
    dv = np.abs(np.interp(grid, d_strip, vx) - np.interp(grid, d_real, vr))
    return bool(dv.max() <= 0.1)


def stage_analyse(args: argparse.Namespace) -> None:
    """Step 8: run tables per snapshot (cached under analysis/runs/, rebuilt when the
    rollouts, keys, design or code changed; always rebuilt under --confirmatory), the key
    table (key measures recomputed from the cached solutions), pair values, tier D condition
    tables (d_OOD against each specialist's own buffer), P6 checkpoint tables and gates, P7
    and P8 tables, then the decision table: --confirmatory runs once (decisions.npz, refuses
    a second run) and only on complete inputs (``confirmatory_problems``); otherwise
    decisions_all plus the descriptive tables."""
    from apexzero.interp import agents, controls, io, measures, stimuli, support
    from apexzero.interp import analysis as A
    from apexzero.interp import answer_key as ak
    from apexzero.interp.constants import BAND_EXPECTED

    _require_tag("analysis")
    m = _manifest()
    finals = _finals(m)
    sup = support.load_support(io.SUPPORT_DIR)
    if args.confirmatory:
        have = {a: _groups_in(io.rollout_path(a, e.timesteps)) for a, e in finals.items()}
        hist = {a: [_groups_in(io.rollout_path(a, h.timesteps)) for h in _history_entries(m, a)]
                for a in finals if agents.is_generalist(a)}
        # R19a: the history snapshots' det groups must hold their tier G runs (the design
        # is read for the tier G cells only once a history snapshot has a det group).
        hist_g: dict[str, list[int]] = {}
        if any("det" in g for gs in hist.values() for g in gs):
            _, d_cells = _design(laid_out=True)
            hist_g = {a: [_tier_g_missing(io.rollout_path(a, h.timesteps), d_cells, a)
                          for h in _history_entries(m, a)] for a in hist}
        probs = confirmatory_problems(m, finals, have, args.agent, hist, hist_g)
        probs += [f"{p} (support stage)" for p in _replicate_regime_problems(sup, finals)]
        if probs:
            raise SystemExit("the confirmatory analysis runs once, on complete inputs: "
                             + "; ".join(probs))
    bank, cells = _design(laid_out=True)
    tables = measures.model_tables()
    b = float(m.band) if m.band is not None else BAND_EXPECTED
    cache = ak.KeyCache()
    sel = args.agent or sorted(finals)
    force = bool(args.force or args.confirmatory)
    inputs_hash = io.sha256_json(dict(keys=cache.content_hash(),
                                      design=stimuli.design_hash(bank, cells),
                                      code=io.code_hash()))
    with io.StageLog("analyse", total=len(sel) + 4,
                     note="confirmatory" if args.confirmatory else "exploratory") as log:
        vmap: dict = {}
        for r in cache.records("viability"):
            snap = str(r.anchor.get("snapshot", ""))
            if "-ph" in snap:
                ts, ph = snap.split("-ph")
                vmap.setdefault((r.agent, int(ts), r.cell_id, int(ph), -1), []).append(r)
        keys = A.CachedKeys(bank, cells, cache, viability=vmap, tables=tables)
        kt = A.key_table(cells, bank, keys, extra=A.qss_v_min(bank, cells))
        io.save_table(io.ANALYSIS_DIR / "key_table.npz", kt)
        log.step(note="key table")

        speed_at_phase = _v7_failed(m)
        if speed_at_phase:
            _say(log, "V7 failed on speeds: e_ve, e_vmin and e_vmin^line are judged against "
                      "the key at each run's phase where one exists (AD23)")

        def rt_of(agent, ts):
            return _run_table_cached(agent, ts, bank, cells, keys, tables, force, inputs_hash,
                                     speed_at_phase)

        runs, history, ckpts, gates, log_std, vtop, conds = {}, {}, {}, {}, {}, {}, {}
        for p in _replicate_regime_problems(sup, sel):
            _say(log, f"{p} (run the support stage): without regimes its pairs get none and "
                      "its P1-P2 decisions are not testable; without R_clip no pair is "
                      "'clip' and P1b / P2 read clip-saturated pairs as below support")
        for a in sel:
            fb = io.CONTROLS_DIR / f"knn_{a}.npz"
            buffer = controls.KnnBuffer.load(fb) if fb.exists() else None
            e = finals[a]
            rt = rt_of(a, e.timesteps)
            if rt is None:
                _say(log, f"{a}: no rollouts")
                continue
            runs[a] = rt
            spec = agents.AGENTS[a]
            prev = agents.list_snapshots(spec)[-3:-1]
            history[a] = [t for t in (rt_of(a, agents.load_agent(p, a).timesteps) for p in prev)
                          if t is not None]
            if "D" in {t for c in cells for t in c.tiers}:
                if a in A.SPECIALISTS and buffer is None:
                    _say(log, f"{a}: no kNN buffer (controls --knn-buffer --agent {a}): d_OOD "
                              "missing, P4 not testable where it needs it")
                conds[a] = A.cond_table(rt, kt, _d_ood(a, rt, bank, cells, buffer))
            if a in agents.seed0_generalists():          # P6 stays on seed 0 (R18)
                ckpts[a], gates[a], log_std[a], vtop[a] = {}, {}, {}, {}
                for snap in [x for x in m.snapshots if x.agent == a]:
                    t = rt_of(a, snap.timesteps)
                    if t is None:
                        continue
                    ckpts[a][snap.timesteps] = t
                    g = A.checkpoint_gate(A.pair_values(t, kt, sup.regimes, b))
                    gates[a][snap.timesteps] = bool(g.get("passed", False))
                    log_std[a][snap.timesteps] = np.asarray(snap.log_std)
                    vtop[a][snap.timesteps] = A.v_top(t, kt)
            log.step(note=f"{a}: {A._n(rt)} runs")
        units, corners = _p8_units(bank, cells)
        ladder = []
        # P8's units are the seed-0 crash pairs (R18: no new crash units from replicates).
        for a in [x for x in agents.seed0_generalists() if x in runs]:
            mine = [u for u in units if u["agent"] == a]
            fid = _fidelity(bank, cells, runs[a], mine)
            unevaluated = sorted(k for k, v in fid.items() if v is None)
            if unevaluated:
                _say(log, f"{a}: fidelity gate not evaluated (inputs missing) for "
                          f"{unevaluated}: kept in P8 as unreproduced")
            ladder.append(A.ladder_table(runs[a], kt, mine, fid))
        twins = [A.twin_table(runs[a], kt, A.pair_values(runs[a], kt, sup.regimes, b), corners)
                 for a in agents.seed0_generalists() if a in runs]
        log.step(note="P8 tables")
        cs = io.load_table(io.STIMULI_DIR / "cell_support.npz")
        train = {g: support.table_features(sup.table(g)) for g in support.GRAMMARS}
        tf = {}
        for g in support.GRAMMARS:
            t = sup.table(g)
            sel_tf = (np.asarray(t["R_seg"]) < 20.0) & (np.asarray(t["v_peak"]) >= 250 / 3.6)
            tf[g] = support.table_features(io.select(t, sel_tf))
        # Each grammar's features standardised by its own training sample's SD (2.3).
        sds = {g: A.training_sd(train[g]) for g in support.GRAMMARS}
        dens = A.density_table(cs, cells, dict(support.AGENT_GRAMMAR), train, tf, sds)
        p7 = {}
        pf = io.PROBES_DIR / "alias.npz"
        qp = rt_of(CONTROL_AGENT.format("Qp"), 0)
        if pf.exists():
            al = A.alias_table(io.load_table(pf), kt)
            ksd = _key_phase_sd(kt, cells, cache, bank)
            for a, rt_a in runs.items():
                p7[a] = (A.phase_sd_table(rt_a, kt, qp, ksd),
                         io.select(al, A._st(al, "agent") == a))
        log.step(note="support and P7 tables")
        ym = io.ANALYSIS_DIR / "yasmarina.json"
        knife = tuple(io.read_json(ym).get("knife_edge_units", [])) if ym.exists() else ()
        sep = m.validation.get("p3_separation") or {}
        inp = A.AnalysisInputs(b=b, keys=kt, regimes=dict(sup.regimes), runs=runs,
                               history=history, tables=tables, conds=conds, checkpoints=ckpts,
                               gates=gates, log_std=log_std, v_top=vtop, p7=p7,
                               ladder=io.concat_tables(ladder) if ladder else None,
                               twins=io.concat_tables(twins) if twins else None, density=dens,
                               demote_p3=bool(sep.get("demote_p3", False)), knife_edge=knife)
        hprobs = A.history_gate_problems(inp)
        if hprobs and args.confirmatory:
            raise SystemExit("the confirmatory analysis runs once, and protocol 8's stability "
                             "rule cannot be applied: " + "; ".join(hprobs))
        for p in hprobs:
            _say(log, f"stability rule: {p}")
        dt = A.decision_table(inp, confirmatory=args.confirmatory)
        for i in range(A._n(dt)):
            _say(log, f"{dt['prediction'][i]} {dt['agent'][i]}: {dt['verdict'][i]} "
                      f"({dt['family'][i]}, stable {dt['stable'][i]})")
        rstem = "replication" if args.confirmatory else "replication_all"
        vf = io.ANALYSIS_DIR / f"{rstem}_variants.npz"
        if any(a in runs for a in agents.replicates()) and vf.exists():
            vt = io.load_table(vf)
            for i in range(A._n(vt)):
                _say(log, f"R18 {vt['variant'][i]} {vt['prediction'][i]}: {vt['verdict'][i]} "
                          f"(seeds {vt['seed_verdicts'][i]})")
        log.step(note="decision table")


def _d_ood(agent: str, rt, bank, cells, buffer) -> dict | None:
    """(agent, cell) -> median normalised NN distance of a specialist's deterministic tier D
    runs' observations (start to s_c + 50) to its own Hockenheim buffer (P4); None without
    a buffer (P4 is then 'not testable' wherever it needs d_OOD)."""
    from apexzero.interp import agents, io, rollout

    if buffer is None:
        return None
    m = _manifest()
    e = _finals(m)[agent]
    ag = agents.load_agent(ROOT / e.path, agent)
    groups, _ = rollout.load_rollouts(io.rollout_path(agent, ag.timesteps))
    return _d_ood_from_det(groups["det"], agent, bank, cells, ag.obs_rms, buffer)


def _d_ood_from_det(det: dict, label: str, bank, cells, obs_rms, buffer) -> dict:
    """d_OOD of every deterministic phase-0 tier D run of one rollout group, the
    observations rebuilt from the recorded states and normalised with ``obs_rms`` (the
    specialist's own, or S1's for the controls K and Q), keyed (label, cell_id)."""
    import jax

    from apexzero.interp import controls, rollout

    by = {c.cell_id: c for c in cells}
    arrays = rollout.bank_arrays(bank)
    out = {}
    for i in range(len(det["cell"])):
        c = by.get(int(det["cell"][i]))
        if c is None or "D" not in c.tiers or int(det["phase"][i]) != 0:
            continue
        v = det["valid"][i]
        x = det["x"][i][v]
        keep = x[:, 0] <= bank.stimuli[c.stim_id].landmarks.s_c0 + 50.0
        if not keep.any():
            continue
        tr = jax.tree.map(lambda a, s=c.stim_id: a[s], arrays)
        obs = rollout.recompute_observations(x[keep], det["prev_action"][i][v][keep],
                                             det["accel"][i][v][keep], tr, obs_rms)
        out[(label, c.cell_id)] = float(np.median(controls.d_ood(obs, buffer)))
    return out


def _key_phase_sd(kt, cells, cache, bank) -> dict[str, float]:
    """geometry -> the key's d_b SD across its 4 phase-matched solutions (P7 null, R <= 18.3
    only: V7's remedy also solves phase-matched keys on P7's larger-R cells, which P7's
    null does not read). The 4 are u = 0, 4, 8 and 12 of the 16 (``_p7_null_phases``):
    ruling R17b solves keys at every phase of P7's cells and keeps the null's definition."""
    from apexzero.interp import analysis as A
    from apexzero.interp import answer_key as ak

    out: dict[str, list] = {}
    geo = {int(c): str(g) for c, g in zip(kt["cell"], kt["geometry"], strict=True)}
    for c in cells:
        if "p7" not in c.flags or not _core_phase_cell(bank, c):
            continue
        ds = []
        for k in _p7_null_phases(c):
            r = ak.key_for(c, "nominal" if k == 0 else "phase", k, cache=cache, bank=bank)
            if r is not None and r.status == "verified":
                ds.append(float(r.measures.get("d_b", np.nan)))
        if len(ds) == 4 and np.isfinite(ds).all():
            # The same estimator as the agent's and Qp's spread (analysis.phase_spread).
            out.setdefault(geo[c.cell_id], []).append(A.phase_spread(ds))
    return {g: float(np.mean(v)) for g, v in out.items()}


# ------------------------------------------------------------------------- step 8: plots


def stage_plots(args: argparse.Namespace) -> None:
    """Step 8: psychometric curves per final (protocol 9.3): v_e, v_min and d_b per
    approach and theta against both x-axes, log R_seg and log v_min*, with the nominal key,
    the backed-off key (v_e) and the pinned s-OCP overlaid, the agent's regimes shaded on
    the R_seg axis (clip, below support, tail) and the covariate v_b / v_b* in a fourth
    panel; PNGs under analysis/plots/. (The QSS overlays and v_top are in the tables.)"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from apexzero.interp import analysis as A
    from apexzero.interp import io, support
    from apexzero.interp.constants import BAND_EXPECTED

    m = _manifest()
    sup = support.load_support(io.SUPPORT_DIR)
    b = float(m.band) if m.band is not None else BAND_EXPECTED
    kt = io.load_table(io.ANALYSIS_DIR / "key_table.npz")
    out = io.ANALYSIS_DIR / "plots"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((io.ANALYSIS_DIR / "runs").glob("*.npz"))
    finals = {e.agent: e.timesteps for e in _finals(m).values()}
    sel = [f for f in files if f.stem.split("_")[0] in finals
           and int(f.stem.split("_")[1]) == finals[f.stem.split("_")[0]]]
    star = {"v_e": ("v_e_star", "v_e_star_bo", "v_e_socp"), "v_min": ("v_min_star", None, "v_min_socp"),
            "d_b": ("d_b_star", None, "d_b_socp")}
    with io.StageLog("plots", total=len(sel)) as log:
        for f in sel:
            agent = f.stem.split("_")[0]
            pairs = A.pair_values(io.load_table(f), kt, sup.regimes, b)
            rt = sup.regimes.get(agent)
            k_by_pair = {}
            for i in range(A._n(kt)):
                if int(kt["direction"][i]) > 0:
                    k_by_pair.setdefault(int(kt["pair"][i]), i)
            for xname in ("R_seg", "v_min_star"):
                fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
                for row, approach in enumerate(("A1", "A2")):
                    for th, mk in ((90.0, "o"), (180.0, "s")):
                        msk = (A._st(pairs, "approach") == approach) & A.has_tier(pairs, "A") & \
                            np.isclose(A._fl(pairs, "theta_deg"), th) & ~A._bo(pairs, "no_cap")
                        x = A._fl(pairs, xname)[msk]
                        o = np.argsort(x)
                        kr = [k_by_pair.get(int(p), -1) for p in A._it(pairs, "pair")[msk][o]]
                        for ax, col in zip(axes[row, :3], ("v_e", "v_min", "d_b"), strict=True):
                            y = A._fl(pairs, col)[msk][o]
                            y = np.where(np.isinf(y), np.nan, y)
                            ax.semilogx(x[o], y, mk + "-", label=f"agent {th:g} deg")
                            nom, bo, so = star[col]
                            ax.semilogx(x[o], A._fl(pairs, nom)[msk][o], mk + ":", color="0.4",
                                        label=f"key {th:g} deg")
                            if bo:
                                ax.semilogx(x[o], A._fl(pairs, bo)[msk][o], mk + "--",
                                            color="0.6", label=f"backed-off {th:g} deg")
                            ys = np.array([float(kt[so][k]) if k >= 0 else np.nan for k in kr])
                            ax.semilogx(x[o], ys, mk, mfc="none", color="tab:green",
                                        label=f"s-OCP {th:g} deg")
                            ax.set_title(f"{agent} {approach}: {col}")
                            ax.set_xlabel(xname)
                            if xname == "R_seg" and rt is not None:
                                lo = getattr(rt, "r_clip_max", np.nan)
                                if np.isfinite(lo):
                                    ax.axvspan(1.0, lo, color="tab:red", alpha=0.08)
                                ax.axvspan(lo if np.isfinite(lo) else 1.0, rt.q1, color="tab:orange",
                                           alpha=0.08)
                                ax.axvspan(rt.q1, rt.q10, color="tab:blue", alpha=0.06)
                        cov = A._fl(pairs, "v_b")[msk][o] / A._fl(pairs, "v_b_star")[msk][o]
                        axes[row, 3].semilogx(x[o], cov, mk + "-", label=f"{th:g} deg")
                        axes[row, 3].set_title(f"{agent} {approach}: v_b / v_b*")
                        axes[row, 3].set_xlabel(xname)
                axes[0, 0].legend(fontsize=6)
                fig.savefig(out / f"{agent}_curves_{xname}.png", dpi=110)
                plt.close(fig)
            log.step(note=f"{agent} curves")


def _state_index(lap, s_target: float, lap_index: int) -> int:
    """The trace index ``agents.real_state`` takes for (s_target, lap): the first live state
    at or after the target, or 0 when the target precedes the run's start."""
    target = (lap_index - 1) * lap.length + s_target
    s = lap.x[:, 0].astype(np.float64)
    live = lap.valid.copy()
    if lap.term_step >= 0:
        live[lap.term_step + 1:] = False
    if target <= s[0]:
        return 0
    return int(np.nonzero(live & (s >= target))[0][0])


R1_CATEGORY = "standard evaluation (R1)"
R1_CONVENTION = ("the standard R_DEV evaluation's convention (eval/laps.py: s = 0, v0 = 0.9 "
                 "v_QSS(0), default car, deterministic, two laps on the refined R_DEV circuit), "
                 "run as scripts/diagnose_crashes.py runs it (agents.real_lap: "
                 "laps * 3 L / max(v0, 20) / dt steps, without eval/laps.py's 200-step margin)")


def _r1_outcome(rl) -> str:
    return "two laps" if rl.term_step < 0 else f"terminated: {rl.cause}"


def _log_r1_once(mm, rec: dict) -> None:
    """Append ``rec`` to the manifest's standard evaluations unless its lap file is logged."""
    ev = mm.lists.setdefault("standard_evaluations", [])
    if not any(e.get("file") == rec["file"] for e in ev):
        ev.append(rec)


def real_laps_gate() -> bool:
    """Ruling R1 (protocol 12): the real-lap rollouts of ``rollouts --real-laps`` (states for
    tiers D, E1 and E2, the fidelity-gate traces, G3's crash corners) repeat the standard
    R_DEV evaluation every agent already had (eval_rdev.json / eval.json) and put no
    protocol stimulus in front of an agent, so they run before the tag. After the tag they
    run like every post-tag stage, on the tagged code. Every other agent rollout stays
    tag-gated. Returns True when the run is before the tag."""
    if not _tag_exists():
        return True
    _require_tag("real-lap rollouts", design=False)
    return False


def r1_entry(agent: str, track: str, snapshot: str, before_tag: bool, outcome: str,
             file: str = "") -> dict:
    """The manifest record of one real-lap run: a 'standard evaluation (R1)' entry in
    ``manifest.lists['standard_evaluations']``, apart from the protocol's two pre-tag
    exceptions (``pre_tag_agent_rollouts``)."""
    return dict(category=R1_CATEGORY, agent=agent, track=track, snapshot=snapshot,
                before_tag=bool(before_tag), outcome=outcome, file=file,
                convention=R1_CONVENTION, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                why="real-lap states (tiers D, E1, E2), fidelity-gate traces and G3's crash "
                    "corners (ruling R1)")


def _real_laps(args, before_tag: bool = False) -> None:
    """The real-lap states of tiers D, E1 and E2 (stimuli.real_state_requests) and the traces
    the fidelity gate compares with: the standard deterministic two-lap evaluation of each
    named agent on its R_DEV circuit (agents.real_lap; G2 on every R_DEV circuit first, for
    its reach, which selects E2; G3 on every R_DEV circuit for its own crash corners). These
    are agent rollouts on real circuits, not stimuli: ruling R1 allows them before the tag
    (``real_laps_gate``), and each run is logged in the manifest as a 'standard evaluation
    (R1)' entry (``r1_entry``) as soon as its lap file is written. Writes
    stimuli/real_states.json, stimuli/real_states_index.json and stimuli/real_laps/*.npz."""
    from apexzero.interp import agents, io, stimuli
    from apexzero.tracks.corners import R_DEV

    m = _manifest()
    finals = _finals(m)
    out = io.STIMULI_DIR / "real_laps"
    laps: dict = {}

    def lap(agent: str, track: str):
        if (agent, track) in laps:
            return laps[(agent, track)]
        f = out / f"{agent}_{track}.npz"
        if f.exists() and not args.force:
            with np.load(f) as z:
                d = {k: z[k] for k in z.files}
            rl = agents.RealLap(agent=agent, track=track, v_start=float(d["v_start"]), x=d["x"],
                                prev_action=d["prev_action"], accel=d["accel"],
                                valid=d["valid"].astype(bool), term_step=int(d["term_step"]),
                                cause=str(d["cause"]), length=float(d["length"]))
            rel = str(f.relative_to(ROOT))
            if not any(e.get("file") == rel for e in m.lists.get("standard_evaluations", [])):
                # A lap whose run stopped between writing it and logging it: R1 logs every
                # real-lap run, so the resume logs it (when it ran is not known after the tag).
                rec = dict(r1_entry(agent, track, finals[agent].path, before_tag, _r1_outcome(rl),
                                    rel), logged_on_resume=True,
                           before_tag=True if before_tag else None)
                agents.update_manifest(lambda mm, rec=rec: _log_r1_once(mm, rec))
        else:
            rl = agents.real_lap(_load(finals[agent]), track)
            ds = float(stimuli.real_track(track).ds)
            arrays = dict(x=rl.x, prev_action=rl.prev_action, accel=rl.accel, valid=rl.valid,
                          term_step=rl.term_step, cause=rl.cause, length=rl.length,
                          v_start=rl.v_start, ds=ds)
            io._atomic(f, lambda fh: np.savez(fh, **arrays))     # a resume never reads half a file
            rec = r1_entry(agent, track, finals[agent].path, before_tag, _r1_outcome(rl),
                           str(f.relative_to(ROOT)))
            agents.update_manifest(
                lambda mm: mm.lists.setdefault("standard_evaluations", []).append(rec))
        laps[(agent, track)] = rl
        return rl

    with io.StageLog("rollouts", total=2 * len(R_DEV) + 1, note="real laps (tiers D, E1, E2)") as log:
        reach = {}
        for t in R_DEV:
            rl = lap("G2", t)
            live = rl.valid.copy()
            if rl.term_step >= 0:
                live[rl.term_step + 1:] = False
            reach[t] = float("inf") if rl.term_step < 0 else float(rl.x[live, 0].max())
            log.step(note=f"G2 {t}: {'two laps' if rl.term_step < 0 else rl.cause}")
        if "G3" in finals:
            # G3's own R_DEV crash corners join P8 by the same construction rule (2.4 P8):
            # its standard deterministic evaluation, crash by crash.
            g3 = {t: lap("G3", t) for t in R_DEV}
            pairs = stimuli.crash_pairs_from_laps("G3", g3)
            io.write_json(io.STIMULI_DIR / stimuli.EXTRA_CRASH_PAIRS_FILE,
                          dict(pairs=[dataclasses.asdict(p) for p in pairs],
                               source="G3 deterministic real laps (rollouts --real-laps)",
                               created=time.strftime("%Y-%m-%dT%H:%M:%S")))
            log.step(len(R_DEV), note=f"G3: {len(pairs)} crash units")
        states: dict = {}
        index: dict = {}
        for r in stimuli.real_state_requests(reach):
            if r["agent"] not in finals:
                continue
            rl = lap(r["agent"], r["track"])
            st = agents.real_state(rl, r["s_target"], r["lap"])
            states.setdefault(r["agent"], {})[r["key"]] = dataclasses.asdict(st)
            index.setdefault(r["agent"], {})[r["key"]] = _state_index(rl, r["s_target"], r["lap"])
        io.write_json(io.STIMULI_DIR / "real_states.json", states)
        io.write_json(io.STIMULI_DIR / "real_states_index.json", index)
        log.step(note=f"{sum(len(v) for v in states.values())} states")


YAS_UNIT = "G2/YasMarina/1369"     # P8 unit of G2's YasMarina crash (CRASH_PAIRS)
YAS_SAMPLES = 64                    # protocol 3.2: stochastic P_crash, n = 64


def _yasmarina(args) -> None:
    """Protocol 3.2's YasMarina diagnosis (post-tag: G2 on a real circuit and on its
    transplant): native-noise P_crash (n = 64) from G2's real-lap state at s_c - 800 on the
    real closed track and on the E1 real-rung transplant, with the real prev_action and
    accel and with both set to 0; a crash counts until the transplant's end (s_c + 400 m,
    the same distance travelled on both). Real-track P_crash < 0.5 (real auxiliaries) labels
    the unit knife-edge: it leaves the positive-control set and stays in P8 as unreproduced.
    Writes analysis/yasmarina.json."""
    import jax
    import jax.numpy as jnp

    from apexzero.interp import io, rollout, stimuli
    from apexzero.vehicle.sim import track_arrays

    m = _manifest()
    g2 = _load(_finals(m)["G2"])
    pair = next(p for p in stimuli.crash_pairs() if p.agent == "G2" and p.track == "YasMarina")
    track = stimuli.real_track("YasMarina")
    corner = stimuli.crash_corner(track, pair.s_crash)
    key = stimuli.crash_corner_key("YasMarina", corner)
    rs = io.read_json(io.STIMULI_DIR / "real_states.json").get("G2", {}).get(key)
    if rs is None:
        raise SystemExit(f"no G2 real-lap state for {key} (rollouts --real-laps first)")
    bank, cells = _design(laid_out=True)
    cell = next((c for c in cells if "E1" in c.tiers and c.state_source.startswith("G2:")
                 and c.state_source.endswith(f"@{key}") and _version(bank, c) == "real"), None)
    if cell is None:
        raise SystemExit(f"no E1 real-rung cell from G2's state at {key} (stimuli --real)")
    lm = bank.stimuli[cell.stim_id].landmarks
    x_strip = np.asarray(cell.initial_state(0), np.float32)
    dist = float(lm.s_c0 + stimuli.E_CUT_AFTER - x_strip[0])        # to the transplant's end
    steps = int(cell.W or 800)
    pol = rollout.agent_policy(g2.cfg)
    out: dict = dict(unit=YAS_UNIT, key=key, n=YAS_SAMPLES, distance_m=dist)
    banks = {"real": jax.tree.map(lambda a: jnp.asarray(a)[None], track_arrays(track)),
             "transplant": jax.tree.map(lambda a: jnp.asarray(a)[None],
                                        bank.row_arrays(cell.stim_id))}
    x_real = np.asarray(rs["x"], np.float32)
    starts = {"real": x_real, "transplant": x_strip}
    fn = rollout.make_stimulus_rollout(pol, steps, "stoch")
    for where in ("real", "transplant"):
        for aux in ("real_aux", "zero_aux"):
            pa = np.asarray(rs["prev_action"] if aux == "real_aux" else (0.0, 0.0), np.float32)
            ac = np.asarray(rs["accel"] if aux == "real_aux" else (0.0, 0.0), np.float32)
            x0 = np.repeat(starts[where][None], YAS_SAMPLES, 0)
            batch = rollout.batch_from_states(0, x0, prev_action=pa, accel=ac, mode="nat",
                                              pair_id=cell.pair_id, sample=np.arange(YAS_SAMPLES))
            tr = rollout.trace_arrays(fn(g2.actor_params, g2.obs_rms, banks[where], batch),
                                      YAS_SAMPLES, "stoch")
            k = tr["term_step"]
            s = tr["s"].astype(float)
            crash = [bool(k[i] >= 0 and s[i, k[i] + 1] - s[i, 0] <= dist) for i in range(YAS_SAMPLES)]
            out[f"p_crash_{where}_{aux}"] = float(np.mean(crash))
    out["knife_edge"] = bool(out["p_crash_real_real_aux"] < 0.5)
    out["knife_edge_units"] = [YAS_UNIT] if out["knife_edge"] else []
    io.write_json(io.ANALYSIS_DIR / "yasmarina.json", out)
    _say(None, f"YasMarina diagnosis: {_jsonable(out)}")


# ------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=2, help="XLA intra-op threads")
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in STAGES:
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int,
                       default=3 if name in ("rollouts", "probes") else DEFAULT_WORKERS,
                       help="worker processes (fixed, not load-throttled; protocol 10: keys "
                            "4-5, rollouts 3)")
        p.add_argument("--force", action="store_true",
                       help="redo instead of resuming (keys / validate / anchored-keys: re-solve; "
                            "rollouts: re-run groups; analyse: rebuild run tables; dryrun, probes "
                            "and plots always recompute)")
        if name in ("keys", "anchored-keys", "validate"):
            p.add_argument("--priority", choices=("P0", "P1", "P2"), default="P0")
        if name in ("rollouts", "probes", "analyse", "plots", "anchored-keys", "controls"):
            p.add_argument("--agent", action="append", default=None,
                           help="agent id (repeatable); default: every agent in the manifest "
                                "(controls --knn-buffer: S1, or S2 / S3 after the tag)")
        if name == "rollouts":
            p.add_argument("--snapshots", choices=("finals", "last3", "all"), default="finals")
            p.add_argument("--mode", choices=("det", "nat", "mat", "all"), default="det")
            p.add_argument("--pilot-replication", action="store_true",
                           help="step 2: harness gates, pilot replication, throughput")
            p.add_argument("--real-laps", action="store_true",
                           help="the standard R_DEV evaluation (real-lap states of tiers D, "
                                "E1, E2, fidelity-gate traces, G3's crash corners); allowed "
                                "before the tag by ruling R1 and logged in the manifest as "
                                "standard evaluations")
            p.add_argument("--yasmarina", action="store_true",
                           help="protocol 3.2's YasMarina diagnosis (post-tag)")
        if name == "analyse":
            p.add_argument("--confirmatory", action="store_true")
        if name == "stimuli":
            p.add_argument("--real", action="store_true",
                           help="also build tiers D and E (needs real-lap states)")
            p.add_argument("--f6-scale", type=float, default=1.0,
                           help="F6 amplitudes x this (0.5: protocol 7's one halving when K "
                                "lands in class 1)")
        if name in ("stimuli", "keys", "validate", "controls"):
            p.add_argument("--post-tag-deviation", default=None, metavar="REASON",
                           help="run this pre-tag stage after the tag, logged as a deviation")
        if name == "keys":
            p.add_argument("--layout-only", action="store_true")
            p.add_argument("--smoke", action="store_true",
                           help="one validation-sized solve, nothing cached")
        if name == "validate":
            p.add_argument("--measures", action="store_true", help="step 3 instead of V1-V11")
        if name == "controls":
            p.add_argument("--knn-buffer", action="store_true",
                           help="build S1's Hockenheim kNN buffer (protocol exception 2)")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    fn = globals()["stage_" + args.stage.replace("-", "_")]
    fn(args)


if __name__ == "__main__":
    main()
