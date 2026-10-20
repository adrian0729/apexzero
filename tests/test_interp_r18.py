"""Ruling R18 (protocol 12): seeds 1 and 2 of each generalist variant join the study as
prospective agents (G1s1, G1s2, G2s1, G2s2, G3s1, G3s2). Their registry entries, scope (the
seed-0 agent's tiers and keys, their own regimes, finals and the last three checkpoints only),
the replication family (variant-level 2-of-3 rule, P1c on median R_dep, seed spread), the
budget and the manifest. Nothing is solved or rolled out: keys and runs are doubles or
hand-built records, banks are hand-built, run configs are only read, and files go to tmp dirs
only (the analysis family's end-to-end test is in test_interp_analysis.py)."""

from __future__ import annotations

import dataclasses
import importlib
import json
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.agents.ppo import Actor, PPOConfig
from apexzero.env import racing
from apexzero.interp import agents, io, measures, registry, rollout, stimuli, support
from apexzero.interp import analysis as A
from apexzero.interp import answer_key as ak
from apexzero.interp.stimuli import Cell, Geometry, Landmarks, Stimulus

REPO = Path(__file__).resolve().parents[1]
REPLICATES = ("G1s1", "G1s2", "G2s1", "G2s2", "G3s1", "G3s2")
V7_FAILED = {"V5": {"passed": False}, "V7": {"passed": False, "name": "V7",
                                              "details": {"cells": {}, "failed": [3]}}}


@pytest.fixture(scope="module", autouse=True)
def _float32():
    """Bank arrays are float32 as in the rollouts; other modules switch x64 on at import."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def ps():
    path = str(REPO / "scripts")
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("psychophysics")
    finally:
        sys.path.remove(path)


class _NullLog:
    def __init__(self, *a, **k):
        self.done = 0
        self.lines: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def step(self, n: int = 1, note: str = ""):
        self.done += n

    def set_total(self, n: int):
        pass

    def write(self, msg: str):
        self.lines.append(msg)


LM = Landmarks(1000.0, 1010.0, 1020.0, 1030.0, 1040.0)


def _cell_design(rows):
    """rows: (R, tiers, approach, D, n_phases, runners). A hand-built bank and cells."""
    stims, cells = [], []
    for i, (R, tiers, ap, D, n_ph, runners) in enumerate(rows):
        geo = Geometry(R, 90.0, np.nan, R, 90.0, 40.0, 0.0, 1020.0, 1, False, 6.0)
        stims.append(Stimulus(i, f"s{i}", "synthetic", tiers, 1, -1, LM, geo, None))
        cells.append(Cell(i, i, 10 + i, tiers, ap, 86.0, D, W=400, P=4.0, n_phases=n_ph,
                          runners=runners))
    return SimpleNamespace(stimuli=stims), cells


# ------------------------------------------------------------------------- 1. registry


def test_R18_registry_lists_the_replicates_with_their_variant_and_seed():
    """Six prospective replicates, two per variant, trained on their seed-0 agent's grammar
    (only the seed differs); G1-G3 are seed 0 and the specialists keep their own seeds."""
    gens = registry.generalists()
    assert gens == ("G1", "G2", "G3", *REPLICATES)
    assert registry.seed0_generalists() == ("G1", "G2", "G3")
    assert registry.replicates() == REPLICATES
    assert registry.replicates("v2") == ("G2s1", "G2s2")
    assert registry.specialists() == ("S1", "S2", "S3")
    assert registry.variants() == ("v1", "v2", "v3")
    assert registry.variant_seeds("v3") == ("G3", "G3s1", "G3s2")
    runs = {"G1": "runs/dr-pilot-200M", "G2": "runs/dr-pilot-v2-tilt30",
            "G3": "runs/dr-pilot-v3-tilt50a2"}
    for a in REPLICATES:
        spec, base = registry.AGENTS[a], registry.AGENTS[registry.seed0_of(a)]
        assert registry.seed0_of(a) == a[:2] and registry.is_replicate(a)
        assert spec.kind == "generalist" and spec.variant == base.variant
        assert spec.grammar == base.grammar and spec.seed == int(a[-1])
        assert spec.run == f"{runs[a[:2]]}-s{a[-1]}"
        assert spec.prospective_after == 0.0 and not spec.pilot_informed
        assert (spec.final_file, spec.done_file) == ("ckpt", "eval_rdev.json")
        # R18: the hardware differs from seed 0's; it is recorded, not corrected
        assert spec.hardware.startswith("GPU (NVIDIA L40S") and base.hardware == "Mac CPU"
    for a, v in (("G1", "v1"), ("G2", "v2"), ("G3", "v3")):
        assert registry.AGENTS[a].variant == v and registry.AGENTS[a].seed == 0
        assert registry.seed0_of(a) == a and not registry.is_replicate(a)
    assert [registry.seed_of(a) for a in ("S1", "S2", "S3")] == [1, 2, 3]
    assert registry.seed0_of("S2") == "S2" and registry.variant_of("S2") == ""
    # unknown ids (controls, dry-run agents, test agents) are none of these
    assert not registry.is_replicate("control-Q") and registry.seed0_of("lawful") == "lawful"
    assert registry.variant_of("G9") == "" and registry.seed_of("G9") == -1
    # agents re-exports the registry, support derives its grammar map from it
    assert agents.AGENTS is registry.AGENTS and agents.AgentSpec is registry.AgentSpec
    assert agents.generalists is registry.generalists and agents.replicates is registry.replicates
    assert support.GRAMMARS is registry.GRAMMARS and support.GrammarSpec is registry.GrammarSpec
    assert support.AGENT_GRAMMAR == registry.agent_grammar()
    assert {a: support.AGENT_GRAMMAR[a] for a in REPLICATES} == {
        "G1s1": "base", "G1s2": "base", "G2s1": "v2mix", "G2s2": "v2mix", "G3s1": "v3mix",
        "G3s2": "v3mix"}
    # an AgentSpec built positionally (as before R18) gets variant '' and seed 0
    old = registry.AgentSpec("G9", "x", "generalist", False, 0.0, "ckpt", "eval_rdev.json", None)
    assert (old.variant, old.seed, old.hardware) == ("", 0, "")


def test_R18_replicate_runs_differ_from_seed0_only_in_the_seed():
    """R18: trained with exactly the seed-0 runs' arguments, only --seed differs (and the
    name; options the seed-0 runs did not record are at their defaults, the grammar's own
    values). Read from the runs' config.json only."""
    for a in REPLICATES:
        spec, base = registry.AGENTS[a], registry.AGENTS[registry.seed0_of(a)]
        f, f0 = REPO / spec.run / "config.json", REPO / base.run / "config.json"
        if not (f.exists() and f0.exists()):
            pytest.skip("replicate runs not on disk")
        c, c0 = json.loads(f.read_text()), json.loads(f0.read_text())
        assert c["ppo"] == c0["ppo"] and c["env"] == c0["env"]
        assert c["args"]["seed"] == spec.seed and c0["args"]["seed"] == 0
        g = dataclasses.asdict(spec.grammar)
        for k in set(c["args"]) | set(c0["args"]):
            v, v0 = c["args"].get(k), c0["args"].get(k)
            if k in ("seed", "name") or v == v0:
                continue
            assert v0 is None and v == g[k], (a, k, v0, v)


def test_R18_no_hard_coded_agent_lists():
    """Requirement 1: no ('G1', 'G2', 'G3') or ('S1', 'S2', 'S3') list is left in the harness;
    every module reads the registry's helpers."""
    pat = re.compile(r"""["']G1["'],\s*["']G2["'],\s*["']G3["']|["']S1["'],\s*["']S2["'],"""
                     r"""\s*["']S3["']""")
    files = sorted((REPO / "src" / "apexzero" / "interp").glob("*.py"))
    files.append(REPO / "scripts" / "psychophysics.py")
    hits = [f"{f.name}:{i + 1}" for f in files for i, ln in enumerate(f.read_text().splitlines())
            if pat.search(ln) and not ln.lstrip().startswith("#")]
    assert hits == []
    assert stimuli.RUNNERS_FROM_S1 == ("S1", "G1", "G2", "G3")      # the saved design's runners
    assert A.GENERALISTS == ("G1", "G2", "G3") and A.SPECIALISTS == ("S1", "S2", "S3")


# ------------------------------------------------------------------------- 2. scope


def test_R18_replicates_run_their_seed0_agents_cells():
    """Tier D cells built from S1's state list S1 and G1-G3 as runners (the saved design is
    unchanged); a replicate runs every cell its seed-0 agent runs, so G2s1 runs them, and a
    cell of one specialist's own state is not run by any generalist."""
    assert registry.runs_cell((), "G2s1")
    assert registry.runs_cell(stimuli.RUNNERS_FROM_S1, "G2s1")
    assert registry.runs_cell(stimuli.RUNNERS_FROM_S1, "G3s2")
    assert not registry.runs_cell(("S2",), "G1s1") and registry.runs_cell(("S2",), "S2")
    assert not registry.runs_cell(("G1",), "G2s1") and registry.runs_cell(("G1",), "G1s2")
    ids = lambda runs: sorted({r.cell_id for r in runs})
    x0 = (200.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 0.0)
    cells = [Cell(0, 0, 0, ("D",), "real", 60.0, 800.0, W=400, P=3.0, x0=x0,
                  runners=stimuli.RUNNERS_FROM_S1),
             Cell(1, 1, 1, ("D",), "real", 60.0, 800.0, W=400, P=3.0, x0=x0, runners=("S2",)),
             Cell(2, 2, 2, ("A",), "A1", 86.0, 600.0, W=400, P=4.0)]
    bank = SimpleNamespace(stimuli=[SimpleNamespace(direction=1, mirror_id=-1)] * 3)
    for agent, want in (("G1", [0, 2]), ("G1s1", [0, 2]), ("G3s2", [0, 2]), ("S2", [1, 2]),
                        ("S3", [2])):
        assert ids(rollout.plan_runs(cells, bank, "det", 0, agent=agent)) == want, agent


def test_R18_replicates_roll_out_their_finals_and_last_three_only(ps, monkeypatch):
    """Protocol 8's stability rule: a replicate's rollouts are its final and the two
    snapshots before it, whatever --snapshots says; P6's checkpoint history stays on seed 0."""
    def fake(spec):
        if spec.final_file == "agent.pkl":
            return [ps.ROOT / spec.run / "agent.pkl"]
        return [ps.ROOT / spec.run / "ckpt" / f"{i:012d}.pkl" for i in range(1, 17)]

    monkeypatch.setattr(agents, "list_snapshots", fake)
    for which in ("finals", "last3", "all"):
        k = [kind for _, kind in ps._snapshot_kinds("G2s1", which)]
        assert k == (["final"] if which == "finals" else ["final", "last3", "last3"])
    paths = [p for p, _ in ps._snapshot_kinds("G2s1", "all")]
    assert paths[0].endswith("000000000016.pkl") and paths[1:] == [
        "runs/dr-pilot-v2-tilt30-s1/ckpt/000000000014.pkl",
        "runs/dr-pilot-v2-tilt30-s1/ckpt/000000000015.pkl"]
    g2 = [kind for _, kind in ps._snapshot_kinds("G2", "all")]
    assert g2.count("other_generalist") == 13 and g2[:3] == ["final", "last3", "last3"]
    s2 = [kind for _, kind in ps._snapshot_kinds("S2", "all")]
    assert s2.count("other_specialist") == 15 and "last3" not in s2


def test_R18_replicate_regimes_are_their_own_or_the_seed0_stand_in(ps):
    """The budget counts a replicate with its own frozen regimes, or, before the support
    stage has frozen them, its seed-0 agent's (same grammar: same q1 / q10)."""
    g1, own = SimpleNamespace(q1=12.0, q10=20.0), SimpleNamespace(q1=12.0, q10=20.0)
    regs, src = ps._replicate_regimes({"G1": g1, "G1s2": own})
    assert set(regs) == {"G1s1", "G1s2"} and regs["G1s1"] is g1 and regs["G1s2"] is own
    assert src == {"G1s1": "stand-in: G1's", "G1s2": "own"}
    assert ps._replicate_regimes(None) == ({}, {})


def _stage_stubs(ps, monkeypatch, tmp_path, bank, cells, finals, regimes, snapshots=(),
                 viability=None):
    det = dict(cell=np.array([c.cell_id for c in cells for _ in range(4)]),
               phase=np.array([k for _ in cells for k in range(4)]))
    f = tmp_path / "run.npz"
    f.write_bytes(b"x")
    pooled: list[dict] = []

    def fake_pool(jobs, workers, log, fn=None, init=None):
        pooled.extend(jobs)
        return [dict(label=j["label"], keys=[], cpu=0.0) for j in jobs]

    monkeypatch.setattr(ps, "_require_tag", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_design", lambda laid_out=False: (bank, cells))
    monkeypatch.setattr(support, "load_support", lambda d: SimpleNamespace(regimes=regimes))
    monkeypatch.setattr(ps, "_manifest", lambda: SimpleNamespace(
        validation=V7_FAILED, budget={}, snapshots=list(snapshots)))
    monkeypatch.setattr(ps, "_finals", lambda m: {a: SimpleNamespace(timesteps=99)
                                                  for a in finals})
    monkeypatch.setattr(measures, "model_tables", lambda: None)
    monkeypatch.setattr(ak, "KeyCache", lambda: None)
    monkeypatch.setattr(io, "StageLog", _NullLog)
    monkeypatch.setattr(io, "rollout_path", lambda a, t: f)
    monkeypatch.setattr(rollout, "load_rollouts", lambda p: ({"det": det}, {}))
    monkeypatch.setattr(ps, "_beaten_jobs", lambda *a: ([], {"cells_checked": 0, "beaten": {}}))
    monkeypatch.setattr(A, "CachedKeys", lambda *a, **k: None)
    monkeypatch.setattr(A, "key_table", lambda *a, **k: {})
    monkeypatch.setattr(ps, "_run_table", lambda agent, ts, *a, **k: {
        "pair": np.array([10]), "agent": np.array([agent]), "timesteps": np.array([ts])})
    monkeypatch.setattr(A, "kappa_clipped_pairs", lambda rt: set())
    monkeypatch.setattr(A, "viability_targets", viability or (lambda t, kt: {}))
    monkeypatch.setattr(ps, "_pool", fake_pool)
    monkeypatch.setattr(ps, "_anchored_preflight", lambda *a, **k: dict(
        cells=2, nominal_missing=[], b600_nominal_missing=[], phase_missing=[],
        b600_phase_missing=[], viability_nominal_missing=[]))
    monkeypatch.setattr(ps, "_tuned_cv", lambda: None)
    monkeypatch.setattr(agents, "update_manifest", lambda fn: None)
    return pooled


def test_R18_anchored_keys_stage_gives_replicates_the_p0_and_p1_keys_but_never_p2(
        ps, monkeypatch, tmp_path):
    """The replicates' finals get G1-G3's re-anchored keys (P0: A1 tail, bulk and below
    support, from their own regimes; P1: tier B D = 600), never P2's checkpoint keys (P6
    stays on seed 0); their viability solves are chosen apart from G1-G3's (9.1's cap and
    seeded subset applied to each family), so G1-G3's selection is unchanged."""
    bank, cells = _cell_design([(15.0, ("A",), "A1", 600.0, 4, ()),        # tail
                                (30.0, ("A",), "A1", 600.0, 4, ()),        # bulk
                                (13.0, ("A",), "A1", 600.0, 4, ()),        # G1 tail, G1s1 below
                                (30.0, ("B",), "B", 600.0, 4, ())])        # tier B D = 600
    reg = {"G1": SimpleNamespace(q1=12.0, q10=20.0, r_clip_max=8.9, r_clip=(8.9,)),
           "G1s1": SimpleNamespace(q1=14.0, q10=20.0, r_clip_max=9.0, r_clip=(9.0,))}
    snaps = [SimpleNamespace(agent=a, timesteps=t, error="", is_final=t == 99)
             for a in ("G1", "G1s1") for t in (10, 30, 99)]
    vcalls: list[list[str]] = []

    def viability(t, kt):
        vcalls.append(sorted(set(np.asarray(t["agent"]).tolist())))
        return {}

    pooled = _stage_stubs(ps, monkeypatch, tmp_path, bank, cells, ("G1", "G1s1", "S1"), reg,
                          snaps, viability)
    monkeypatch.setattr(A, "reanchor_immaterial", lambda finals: dict(immaterial=False))
    # R 13's pair (12) is kappa-clipped at run time in G1s1's final only (R12's binding
    # definition, per agent): below support for G1s1, it gets no key there; tail for G1.
    monkeypatch.setattr(A, "kappa_clipped_pairs",
                        lambda rt: {12} if rt["agent"][0] == "G1s1" else set())
    monkeypatch.setattr(A, "pair_values", lambda *a, **k: {})
    monkeypatch.setattr(A, "checkpoint_gate", lambda p: dict(passed=True))
    ps.stage_anchored_keys(SimpleNamespace(priority="P2", workers=1, force=False))
    got = sorted((j["agent"], j["timesteps"], j["cell"].cell_id, j["label"].rsplit("(", 1)[1])
                 for j in pooled if j["what"] == "anchored")
    assert got == [
        # P2: G1's checkpoints only (first gate-passing, the middle one); the final is P0
        ("G1", 10, 0, "P2)"), ("G1", 10, 1, "P2)"), ("G1", 10, 2, "P2)"),
        ("G1", 30, 0, "P2)"), ("G1", 30, 1, "P2)"), ("G1", 30, 2, "P2)"),
        ("G1", 99, 0, "P0)"), ("G1", 99, 1, "P0)"), ("G1", 99, 2, "P0)"), ("G1", 99, 3, "P1)"),
        # the replicate's final: P0 from its own regimes and clipping, P1 tier B D = 600
        ("G1s1", 99, 0, "P0)"), ("G1s1", 99, 1, "P0)"), ("G1s1", 99, 3, "P1)"),
        ("S1", 99, 0, "P1)"), ("S1", 99, 1, "P1)"), ("S1", 99, 2, "P1)"), ("S1", 99, 3, "P1)")]
    assert vcalls == [["G1"], ["G1s1"]]


def test_R18_anchored_keys_stage_needs_the_replicates_own_regimes(ps, monkeypatch, tmp_path):
    """A replicate final without frozen regimes (support stage not rerun) would treat every
    A1 cell as bulk: the stage refuses instead."""
    bank, cells = _cell_design([(15.0, ("A",), "A1", 600.0, 4, ())])
    reg = {"G1": SimpleNamespace(q1=12.0, q10=20.0, r_clip_max=8.9, r_clip=(8.9,))}
    _stage_stubs(ps, monkeypatch, tmp_path, bank, cells, ("G1", "G2s2"), reg)
    with pytest.raises(SystemExit, match="G2s2"):
        ps.stage_anchored_keys(SimpleNamespace(priority="P0", workers=1, force=False))


def test_R18_review_both_stages_need_the_replicates_regimes_with_r_clip(
        ps, monkeypatch, tmp_path):
    """Review finding 6: one rule for the anchored-keys and analysis stages. A replicate's
    regimes must exist and carry R_clip (support stage with its obs_rms); regimes without
    R_clip have no 'clip' set. The anchored-keys stage refuses; the confirmatory analysis
    refuses too (it used to check only that the regimes exist)."""
    ok = SimpleNamespace(q1=12.0, q10=20.0, r_clip_max=8.9, r_clip=(8.9,))
    no_rc = SimpleNamespace(q1=12.0, q10=20.0, r_clip_max=float("nan"), r_clip=())
    sup = SimpleNamespace(regimes={"G1": no_rc, "G1s1": ok, "G1s2": no_rc})
    probs = ps._replicate_regime_problems(sup, ("G1", "G1s1", "G1s2", "G2s1"))
    assert probs == ["R18 replicate G1s2 has frozen regimes without R_clip",
                     "R18 replicate G2s1 has no frozen regimes"]      # seed 0 is not checked
    assert ps._replicate_regime_problems(sup, ("G1", "G1s1")) == []
    bank, cells = _cell_design([(15.0, ("A",), "A1", 600.0, 4, ())])
    _stage_stubs(ps, monkeypatch, tmp_path, bank, cells, ("G1", "G1s2"), {"G1": ok, "G1s2": no_rc})
    with pytest.raises(SystemExit, match="G1s2 has frozen regimes without R_clip"):
        ps.stage_anchored_keys(SimpleNamespace(priority="P0", workers=1, force=False))
    # the analysis stage's confirmatory gate (it stops before reading any design or rollout)
    monkeypatch.setattr(ps, "_manifest", lambda: agents.Manifest(band=0.075))
    monkeypatch.setattr(ps, "_groups_in", lambda path: {"det", "nat", "mat"})
    with pytest.raises(SystemExit, match="G1s2 has frozen regimes without R_clip"):
        ps.stage_analyse(SimpleNamespace(confirmatory=True, agent=None, force=False))


def test_R18_review_beaten_keys_record_the_replicates_part(ps, monkeypatch, tmp_path):
    """Review finding 2 (kept as protocol 6.3 reads, recorded): R18's replicates are final
    agents, so a key a replicate beats is re-solved, and every final's u = 0 run (theirs
    included) starts the re-solve; the key is shared, so G1-G3's errors follow it. Each job
    and the summary record whether only replicates beat it and which starts are theirs."""
    bank, cells = _cell_design([(15.0, ("A",), "A1", 600.0, 4, ()),
                                (30.0, ("A",), "A1", 600.0, 4, ())])
    f = tmp_path / "run.npz"
    f.write_bytes(b"x")
    J = {0: {"G1": 9.0, "G1s1": 11.0}, 1: {"G1": 11.0, "G1s1": 9.0}}   # J* = 10 on both
    monkeypatch.setattr(io, "rollout_path", lambda a, t: f)
    monkeypatch.setattr(rollout, "load_rollouts",
                        lambda p: ({"det": {"W": np.array([400, 400])}}, {}))
    monkeypatch.setattr(ps, "_u0_rows", lambda det, cells: {0: 0, 1: 1})
    seen: list[str] = []
    finals = {"G1": SimpleNamespace(timesteps=1), "G1s1": SimpleNamespace(timesteps=2)}
    order = iter([(a, cid) for a in finals for cid in (0, 1)])

    def traj(det, i, mode):
        a, cid = next(order)
        seen.append(a)
        return (a, cid)

    monkeypatch.setattr(measures, "trajectory_from_rollout", traj)
    monkeypatch.setattr(measures, "discounted_return", lambda t, W: J[t[1]][t[0]])
    monkeypatch.setattr(ak, "key_for", lambda c, kind, ph, cache=None, bank=None:
                        SimpleNamespace(status="verified", J=10.0))
    jobs, summary = ps._beaten_jobs(bank, cells, finals, None)
    assert summary["beaten"] == {0: ["G1s1"], 1: ["G1"]}
    assert summary["beaten_by_replicates_only"] == [0]
    r18 = {j["cell"].cell_id: j["r18"] for j in jobs}
    assert r18[0] == dict(beaten_by_replicates_only=True, replicate_starts=["G1s1"])
    assert r18[1] == dict(beaten_by_replicates_only=False, replicate_starts=["G1s1"])
    assert all(set(j["runs"]) == {"G1", "G1s1"} for j in jobs)     # 6.3: every final's start


# ------------------------------------------------------------------------- 3. analysis


def test_R18_replicates_form_the_replication_family():
    for a in REPLICATES:
        for p in A.CONFIRMATORY_G + ("P1c-approach", "P2b", "P3-beta-lt120", "P7"):
            assert A.family_of(p, a) == "replication"
        assert A.family_of("P5", a) == "control" and A.family_of("P2d", a) == "exploratory"
    # the pre-registered families are unchanged
    assert A.family_of("P1a", "G3") == "primary" and A.family_of("P1a", "G1") == "secondary"
    assert A.family_of("P4", "S2") == "primary"


def test_R18_variant_verdict_needs_two_of_three_seeds():
    vv = A.variant_verdict
    assert vv({"G1": "confirm", "G1s1": "confirm", "G1s2": "refute"}) == ("confirm", 2)
    assert vv({"G1": "refute", "G1s1": "refute", "G1s2": "refute"}) == ("refute", 3)
    assert vv({"G1": "confirm", "G1s1": "refute", "G1s2": "inconclusive"}) == (
        "seed-specific", 0)
    # a shared non-claim is still the shared decision; a missing seed with no 2 agreeing
    # cannot be called seed-specific (it could have made 2)
    assert vv({"G1": "not testable", "G1s1": "not testable", "G1s2": "confirm"}) == (
        "not testable", 2)
    assert vv({"G1": "confirm", "G1s1": "missing", "G1s2": "refute"}) == ("not testable", 0)
    assert vv({"G1": "confirm", "G1s1": "missing", "G1s2": "confirm"}) == ("confirm", 2)
    assert vv({"G1": "unstable", "G1s1": "unstable", "G1s2": "confirm"}) == ("unstable", 2)


def test_R18_review_cautious_but_lawful_is_a_p1a_refutation_at_the_variant_level():
    """Review finding 4: 'cautious but lawful' labels a P1a refutation (protocol 2.4), so
    {refute, cautious but lawful, confirm} shares the decision 'refute' (2 seeds), not
    'seed-specific'; two seeds with the label share the label itself."""
    vv = A.variant_verdict
    assert A.decision_class("cautious but lawful") == "refute"
    assert A.decision_class("unstable") == "unstable"
    assert vv({"G1": "refute", "G1s1": "cautious but lawful", "G1s2": "confirm"}) == (
        "refute", 2)
    assert vv({"G1": "cautious but lawful", "G1s1": "cautious but lawful",
               "G1s2": "refute"}) == ("cautious but lawful", 2)
    assert vv({"G1": "cautious but lawful", "G1s1": "missing", "G1s2": "refute"}) == (
        "refute", 2)
    assert vv({"G1": "cautious but lawful", "G1s1": "confirm", "G1s2": "inconclusive"}) == (
        "seed-specific", 0)
    t = A.variant_table([_dec("P1a", "G2", "refute"), _dec("P1a", "G2s1", "cautious but lawful"),
                         _dec("P1a", "G2s2", "confirm")])
    i = next(k for k in range(A._n(t)) if t["variant"][k] == "v2" and t["prediction"][k] == "P1a")
    assert (t["verdict"][i], bool(t["claim"][i]), int(t["n_agree"][i])) == ("refute", True, 2)


def _dec(pred, agent, verdict, stable=None, **details):
    d = A._d(pred, agent, verdict, **details)
    d.stable = stable
    return d


def test_R18_variant_table_and_seed_spread():
    """Per (variant, prediction) the decision shared by >= 2 of 3 seeds, else
    'seed-specific' with the three listed; the seed spread lists each seed's verdicts,
    stability, key numbers and R_dep."""
    ds = [_dec("P1a", "G1", "confirm", True, tracking=dict(frac_in=0.9, crash_pairs=0,
                                                          n_pairs=10,
                                                          p_crash=A.RateInterval(0.0, 0.0, 0.02, 0.0,
                                                                                 "none")),
               slope=dict(S=1.02)),
          _dec("P1a", "G1s1", "confirm"), _dec("P1a", "G1s2", "refute"),
          _dec("P1-tail", "G1", "confirm"), _dec("P1-tail", "G1s1", "refute"),
          _dec("P1-tail", "G1s2", "inconclusive"),
          _dec("P3", "G2", "refute", b=0.6, winner="M_idx"), _dec("P3", "G2s1", "refute"),
          _dec("P2", "G3", "not applicable", n_O=1)]
    vt = A.variant_table(ds)
    assert set(vt) == set(A.VARIANT_COLUMNS) and A._n(vt) == 3 * len(A.CONFIRMATORY_G)
    row = {(v, p): i for i, (v, p) in enumerate(zip(vt["variant"], vt["prediction"],
                                                    strict=True))}
    i = row[("v1", "P1a")]
    assert (vt["verdict"][i], bool(vt["claim"][i]), int(vt["n_agree"][i])) == ("confirm", True, 2)
    assert vt["seeds"][i] == "G1,G1s1,G1s2"
    i = row[("v1", "P1-tail")]
    assert vt["verdict"][i] == "seed-specific" and not vt["claim"][i]
    assert json.loads(vt["seed_verdicts"][i]) == {"G1": "confirm", "G1s1": "refute",
                                                  "G1s2": "inconclusive"}
    i = row[("v2", "P3")]                       # G2s2 missing, but 2 of 3 already agree
    assert (vt["verdict"][i], bool(vt["claim"][i])) == ("refute", True)
    assert json.loads(vt["seed_verdicts"][i])["G2s2"] == "missing"
    assert vt["verdict"][row[("v3", "P2")]] == "not testable"
    sp = A.seed_spread(ds, {"G1": {"A1": 14.6, "A2": 11.7}, "G1s1": {"A1": np.inf}})
    assert set(sp) == set(A.SPREAD_COLUMNS) and A._n(sp) == 9
    assert sp["agent"].tolist() == ["G1", "G1s1", "G1s2", "G2", "G2s1", "G2s2", "G3", "G3s1",
                                    "G3s2"]
    assert sp["seed"].tolist() == [0, 1, 2] * 3
    assert (sp["P1a"][0], sp["P1-tail"][2], sp["P3"][3]) == ("confirm", "inconclusive",
                                                             "refute")
    assert sp["P2"][1] == "missing"
    assert json.loads(sp["stable"][0])["P1a"] == "yes"
    n0 = json.loads(sp["numbers"][0])
    assert n0["P1a"]["frac_in"] == 0.9 and n0["P1a"]["S_bulk"] == 1.02
    assert n0["P1a"]["p_crash_hi"] == 0.02 and n0["P1a"]["det_crash_pairs"] == 0
    assert json.loads(sp["numbers"][3])["P3"]["winner"] == "M_idx"
    assert (sp["r_dep_A1"][0], sp["r_dep_A2"][0]) == (14.6, 11.7)
    assert np.isinf(sp["r_dep_A1"][1]) and np.isnan(sp["r_dep_A2"][1])


def test_R18_p1c_orders_variants_by_their_median_r_dep():
    """R18: each variant's R_dep is the median over its seeds; with seed 0 alone P1c is the
    pre-registered test on G1-G3's own values."""
    seed0 = {"G1": {"A1": 18.3, "A2": 11.7}, "G2": {"A1": 14.6, "A2": 14.6},
             "G3": {"A1": 11.7, "A2": 9.4}}
    d = A.p1c(seed0)[0]
    assert d.verdict == "confirm" and d.details["r_dep_A1"] == (18.3, 14.6, 11.7)
    assert d.details["r_dep_A1_seed0"] == (18.3, 14.6, 11.7) and d.agent == "G1,G2,G3"
    assert d.details["p_null"] == pytest.approx(1 / 6)
    # replicates reverse v2 and v3: medians v1 18.3, v2 median(14.6, 9.4, 9.4) = 9.4,
    # v3 median(11.7, 13.1, inf) = 13.1 -> v2 < v3: refuted although seed 0 confirms
    rd = {**seed0, "G1s1": {"A1": 18.3}, "G1s2": {"A1": 22.9}, "G2s1": {"A1": 9.4},
          "G2s2": {"A1": 9.4}, "G3s1": {"A1": 13.1}, "G3s2": {"A1": np.inf}}
    ds = A.p1c(rd)
    d = ds[0]
    assert d.verdict == "refute" and d.details["r_dep_A1"] == (18.3, 9.4, 13.1)
    assert d.details["r_dep_A1_seed0"] == (18.3, 14.6, 11.7)
    assert d.details["n_seeds"] == {"v1": 3, "v2": 3, "v3": 3}
    assert d.details["r_dep_A1_by_seed"]["v3"] == {"G3": 11.7, "G3s1": 13.1, "G3s2": np.inf}
    assert d.details["p_null"] == pytest.approx(1 / 6)
    # P1c-approach stays one row per seed-0 agent
    assert [x.agent for x in ds if x.prediction == "P1c-approach"] == ["G1", "G2", "G3"]
    # an undefined seed is unknown: two known equal seeds fix the median of 3; all equal
    # medians: inconclusive
    eq = {a: {"A1": 14.6, "A2": 14.6} for a in A.GENERALISTS + REPLICATES}
    eq["G1s1"] = {"A1": np.nan}
    d = A.p1c(eq)[0]
    assert d.verdict == "inconclusive" and d.details["n_seeds"]["v1"] == 2
    assert d.details["r_dep_A1"] == (14.6, 14.6, 14.6)
    assert A.p1c({**seed0, "G3": {"A1": np.nan}})[0].verdict == "not testable"


def test_R18_review_p1c_medians_need_the_seeds_that_fix_them():
    """Review finding 3: a variant's median R_dep over 3 seeds is defined only when its known
    seeds fix it (all 3, or 2 known and equal); never the mean of two values (the median of
    {18.3, +inf, unknown} is not +inf); with seed 0 alone P1c is the pre-registered test."""
    seed0 = {"G1": {"A1": 18.3, "A2": 11.7}, "G2": {"A1": 14.6, "A2": 14.6},
             "G3": {"A1": 11.7, "A2": 9.4}}
    full = {**seed0, "G1s1": {"A1": 18.3}, "G1s2": {"A1": 22.9}, "G2s1": {"A1": 14.6},
            "G2s2": {"A1": 9.4}, "G3s1": {"A1": 11.7}, "G3s2": {"A1": 9.4}}
    assert np.isnan(A._seed_median([18.3, np.inf], 3))
    assert A._seed_median([18.3, 18.3], 3) == 18.3 and A._seed_median([np.inf, np.inf], 3) == np.inf
    assert A._seed_median([18.3, np.inf, 9.4], 3) == 18.3 and np.isnan(A._seed_median([9.4], 3))
    assert A._seed_median([9.4], 1) == 9.4 and np.isnan(A._seed_median([], 1))
    # v1 {18.3, inf, undefined}: not fixed -> P1c not testable (was +inf and 'confirm')
    d = A.p1c({**full, "G1s1": {"A1": np.inf}, "G1s2": {"A1": np.nan}})[0]
    assert d.verdict == "not testable" and np.isnan(d.details["r_dep_A1"][0])
    assert d.details["undetermined"] == ["v1"]
    # G3 undefined, one replicate known (the other absent): not testable (was G3s1's value)
    rd = {**seed0, "G3": {"A1": np.nan}, "G3s1": {"A1": 9.4}}
    assert A.p1c(rd)[0].verdict == "not testable"
    # G3 undefined, both replicates known and equal: the median is fixed
    d = A.p1c({**full, "G3": {"A1": np.nan}, "G3s1": {"A1": 9.4}})[0]
    assert d.details["r_dep_A1"] == (18.3, 14.6, 9.4) and d.verdict == "confirm"
    # replicates only (no seed 0, e.g. --agent G1s1 G2s1 G3s1): one seed per variant
    d = A.p1c({"G1s1": {"A1": 18.3}, "G2s1": {"A1": 14.6}, "G3s1": {"A1": 11.7}})[0]
    assert d.verdict == "not testable"
    assert all(np.isnan(x) for x in d.details["r_dep_A1_seed0"])
    assert d.details["rule"].startswith("R18")
    # seed 0 alone: the pre-registered test, rule recorded
    d = A.p1c(seed0)[0]
    assert d.verdict == "confirm" and d.details["rule"].startswith("seed 0 only")
    assert d.details["n_seeds"] == {"v1": 1, "v2": 1, "v3": 1}


# ------------------------------------------------------------------------- 4. budget


def test_R18_budget_counts_the_replicates_keys_and_rollouts(ps, monkeypatch):
    """The replicates' P0 re-anchored keys (A1 tail, bulk, below support: their own regimes
    or the seed-0 stand-in), P1 tier B D = 600 keys and viability solves are their own
    budget items; no P2 item; G1-G3's items are unchanged. Their rollouts (finals and last
    three) are counted, tier D cells from S1's state included."""
    bank, cells = _cell_design([(10.0, ("A",), "A1", 600.0, 4, ()),        # G1: below q1
                                (15.0, ("A",), "A1", 600.0, 4, ()),        # tail
                                (30.0, ("A",), "A1", 600.0, 16, ()),       # bulk, P7 cell
                                (30.0, ("A",), "A2", 600.0, 4, ()),
                                (30.0, ("B",), "B", 600.0, 4, ()),         # tier B D = 600
                                (30.0, ("D",), "real", 800.0, 4, stimuli.RUNNERS_FROM_S1)])
    reg = {"G1": SimpleNamespace(q1=12.0, q10=20.0, r_clip_max=8.9),
           "G2": SimpleNamespace(q1=9.0, q10=12.0, r_clip_max=7.6),
           "G2s1": SimpleNamespace(q1=9.0, q10=12.0, r_clip_max=7.6)}
    monkeypatch.setattr(support, "load_support", lambda d: SimpleNamespace(regimes=reg))
    rep = ps._anchored_rep_count(bank, cells)
    # G1s1, G1s2 (G1's stand-in): below 10, tail 15, bulk 30 (P7) = 4 + 4 + 16 each;
    # G2s1 (own), G2s2 (stand-in): tail 10, bulk 15, bulk 30 (P7) = 4 + 4 + 16 each; G3's
    # replicates have no regimes yet (nor has G3): not counted
    assert rep["anchored_R"]["keys"] == 4 * 24 and rep["anchored_R"]["u0_keys"] == 12
    assert rep["anchored_R"]["p7_keys"] == 4 * 16
    assert rep["anchored_R"]["r12_below_support_keys"] == 2 * 4
    assert rep["anchored_R"]["regimes"] == {"G1s1": "stand-in: G1's", "G1s2": "stand-in: G1's",
                                            "G2s1": "own", "G2s2": "stand-in: G2's"}
    assert rep["anchored_B_R"]["keys"] == 6 * 4 and rep["anchored_B_R"]["u0_keys"] == 6
    g = ps._anchored_g_count(bank, cells)
    p12 = ps._anchored_p12_count(bank, cells)

    def snaps(spec):
        if spec.final_file == "agent.pkl":
            return [ps.ROOT / spec.run / "agent.pkl"]
        return [ps.ROOT / spec.run / "ckpt" / f"{i:012d}.pkl" for i in range(1, 17)]

    monkeypatch.setattr(agents, "list_snapshots", snaps)
    monkeypatch.setattr(ps, "_nominal_braking", lambda b, c, cache: True)
    m = SimpleNamespace(validation=V7_FAILED, budget={"rollout_sps": dict(full=2e4, stoch=1e4)})
    proj = ps._budget(bank, cells, True, m, 1, cache=object())
    it = proj["items"]
    assert (it["anchored_R"]["n"], it["anchored_R"]["priority"]) == (96, "P0")
    assert (it["anchored_B_R"]["n"], it["anchored_B_R"]["priority"]) == (24, "P1")
    assert (it["viability_R"]["n"], it["viability_R"]["priority"]) == (200, "P0")
    assert {i["item"] for k, i in it.items() if k.endswith("_R")} == {"anchored", "viability"}
    assert not [k for k, i in it.items() if k.endswith("_R") and i["priority"] == "P2"]
    # G1-G3's items are counted as before R18
    assert it["anchored_G"]["n"] == g["n"] and it["anchored_B"]["n"] == p12["anchored_B"]["keys"]
    assert it["anchored_ckpt"]["n"] == p12["anchored_ckpt"]["keys"]
    r = proj["replicates"]
    assert set(r["items"]) == {"anchored_R", "anchored_B_R", "viability_R"}
    assert r["cpu_h"] == pytest.approx(sum(it[k]["cpu_h"] for k in r["items"]))
    assert r["before_without_replicates"]["P0"] == pytest.approx(
        proj["before"]["P0"] - it["anchored_R"]["cpu_h"] - it["viability_R"]["cpu_h"])
    assert proj["anchored_phases"]["anchored_R"]["keys"] == 96
    # rollouts: a replicate runs its final and the two before it; G1 every snapshot
    ro = proj["rollouts"]
    by = ro["by_agent"]
    assert by["G1s1"]["snapshots"] == {"final": 1, "last3": 2}
    assert by["G1"]["snapshots"] == {"final": 1, "last3": 2, "other_generalist": 13}
    # the final's det runs: every phase of every cell, the tier D cell included (4 + 4 + 16
    # + 4 + 4 + 4), and the two last-three snapshots 4 phases on tiers A and B (5 cells)
    assert by["G1s1"]["runs"]["det"] == 36 + 2 * 5 * 4
    # S1 runs the tier D cell built from its state, S2 does not (its own D cells only)
    assert by["S1"]["runs"]["det"] == 36 and by["S2"]["runs"]["det"] == 32
    assert ro["steps_replicates"] == sum(by[a]["total_steps"] for a in REPLICATES)
    assert ro["steps_total"] == ro["steps_pre_r18"] + ro["steps_replicates"]
    det = sum(v["steps"]["det"] for v in by.values())
    sto = sum(v["steps"]["nat"] + v["steps"]["mat"] for v in by.values())
    assert ro["process_h"] == pytest.approx(det / 2e4 / 3600 + sto / 1e4 / 3600)
    assert ro["wall_h"] == pytest.approx(ro["process_h"] / 3)


def test_R18_budget_before_the_regimes_are_frozen(ps, monkeypatch):
    """Without frozen regimes the replicates' P0 keys are the protocol's 96 cells per three
    generalists, x 4 phases, for the six of them."""
    bank, cells = _cell_design([(30.0, ("B",), "B", 600.0, 4, ())])
    monkeypatch.setattr(ps, "_frozen_regimes", lambda: None)
    rep = ps._anchored_rep_count(bank, cells)
    assert rep["anchored_R"]["u0_keys"] == 192 and rep["anchored_R"]["keys"] == 4 * 192
    assert rep["anchored_R"]["source"].startswith("protocol")
    assert rep["anchored_B_R"]["keys"] == 6 * 4


# ------------------------------------------------------------------------- 5. manifest


def _write_snapshot(path, cfg, timesteps):
    params = jax.device_get(Actor(cfg, 2, False).init(jax.random.PRNGKey(timesteps),
                                                      jnp.zeros(racing.OBS_DIM)))
    rms = {"mean": np.zeros(racing.OBS_DIM, np.float32),
           "var": np.ones(racing.OBS_DIM, np.float32), "count": np.float32(10.0)}
    d = {"actor": params, "critic": {}, "obs_rms": rms, "timesteps": timesteps,
         "ret_rms": {"mean": np.float32(0), "var": np.float32(1), "count": np.float32(1)},
         "config": dataclasses.asdict(cfg)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(d, fh)


def test_R18_manifest_lists_the_replicates_prospective_and_not_evaluated(tmp_path):
    """A replicate is listed once its eval_rdev.json exists, every snapshot prospective
    (prospective_after 0) and not evaluated, its record naming its variant, seed and seed-0
    agent; one still training is pending."""
    cfg = PPOConfig(network="mlp", actor_width=32, mlp_layers=2, num_envs=64, num_steps=16,
                    total_timesteps=100_000)
    done = dataclasses.replace(registry.AGENTS["G2s1"], run=str(tmp_path / "g2s1"))
    pending = dataclasses.replace(registry.AGENTS["G3s2"], run=str(tmp_path / "g3s2"))
    for ts in (10_000, 20_000, 30_000):
        _write_snapshot(tmp_path / "g2s1" / "ckpt" / f"{ts:012d}.pkl", cfg, ts)
    (tmp_path / "g2s1" / "eval_rdev.json").write_text("{}")
    _write_snapshot(tmp_path / "g3s2" / "ckpt" / "000000010000.pkl", cfg, 10_000)
    m = agents.freeze_manifest(tmp_path / "manifest.json", {"G2s1": done, "G3s2": pending})
    rec = m.agents["G2s1"]
    assert rec["finished"] and rec["variant"] == "v2" and rec["seed"] == 1
    assert rec["replicate_of"] == "G2" and rec["ruling"] == "R18" and rec["n_snapshots"] == 3
    assert rec["hardware"] == registry.REPLICATE_HARDWARE
    assert not m.agents["G3s2"]["finished"] and m.agents["G3s2"]["status"].startswith("pending")
    ents = [e for e in m.snapshots if e.agent == "G2s1"]
    assert len(ents) == 3 and all(e.prospective and not e.evaluated_on_stimuli
                                  and e.forward_pass_ok for e in ents)
    assert [e.is_final for e in ents] == [False, False, True]
    assert not [e for e in m.snapshots if e.agent == "G3s2"]
    # the seed-0 agents' records carry no replicate fields
    g = agents._agent_record(registry.AGENTS["G2"], [])
    assert "replicate_of" not in g and g["variant"] == "v2" and g["seed"] == 0


def test_R18_manifest_stage_names_the_replicates(ps, monkeypatch, tmp_path):
    """The manifest stage's log lists each replicate as such and sums up R18's snapshots:
    prospective and none evaluated."""
    e = agents.SnapshotEntry("G1s1", "runs/x/ckpt/1.pkl", 1, True, "h", 1, (0.0, 0.0),
                             (1.0, 1.0), True, False, forward_pass_ok=True)
    m = agents.Manifest(snapshots=[e], agents={
        a: dict(status="finished" if a == "G1s1" else "pending: no eval_rdev.json yet",
                finished=a == "G1s1") for a in agents.AGENTS})
    monkeypatch.setattr(agents, "freeze_manifest", lambda *a, **k: m)
    log = _NullLog()
    monkeypatch.setattr(io, "StageLog", lambda *a, **k: log)
    monkeypatch.setattr(io, "MANIFEST_PATH", tmp_path / "manifest.json")
    ps.stage_manifest(SimpleNamespace())
    assert any(ln.startswith("G1s1 (R18 replicate of G1: variant v1, seed 1): finished")
               for ln in log.lines)
    assert "R18 replicates: 1/6 finished, 1 snapshots, prospective 1, evaluated on stimuli 0" \
        in log.lines
    assert any(ln.startswith("G1: pending") for ln in log.lines)
