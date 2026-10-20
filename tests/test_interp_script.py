"""scripts/psychophysics.py's gates and pure helpers (review regressions): the tag gate on
every post-tag stage, the stages refused after the tag, the pre-registration record, the
one-shot confirmatory analysis's input checks, P6c's checkpoint choice, which cells get
re-anchored keys, and the analysis run-table cache. Nothing is solved or rolled out."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apexzero.interp import agents, io
from apexzero.interp.stimuli import Cell, Geometry, Landmarks, Stimulus


@pytest.fixture(scope="module")
def ps():
    import sys

    path = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("psychophysics")
    finally:
        sys.path.remove(path)


# ------------------------------------------------------------------------- tag gates


def test_agent_stages_refuse_without_the_tag(ps, monkeypatch):
    """Hygiene minor 11 / freeze rule: no agent rollout, anchored key, probe or analysis
    runs before the tag (the pilot replication is an exception; the real laps are allowed by
    ruling R1, tested in test_interp_rulings.py)."""
    monkeypatch.setattr(ps, "_tag_exists", lambda: False)
    for argv in (["rollouts"], ["rollouts", "--yasmarina"],
                 ["anchored-keys"], ["probes"], ["analyse"]):
        with pytest.raises(SystemExit, match="does not exist yet"):
            ps.main(argv)


def test_design_stages_refuse_after_the_tag(ps, monkeypatch):
    """Hygiene M1: stimuli, keys, validate, controls and the dry run change the design, the
    keys or the pre-registration record, so they refuse once the tag exists."""
    monkeypatch.setattr(ps, "_tag_exists", lambda: True)
    for argv in (["stimuli"], ["keys"], ["validate"], ["controls"], ["dryrun"]):
        with pytest.raises(SystemExit, match="after the tag"):
            ps.main(argv)


def test_steps_2_and_3_block_what_follows(ps, monkeypatch):
    """Hygiene minor 6: a failed step 2 or 3 blocks the later stages."""
    m = agents.Manifest(validation={"step2": {"passed": True}, "step3": {"passed": False}})
    monkeypatch.setattr(ps, "_manifest", lambda: m)
    with pytest.raises(SystemExit, match="step3 has not passed"):
        ps._require_steps("the keys stage")
    m.validation["step3"]["passed"] = True
    ps._require_steps("the keys stage")


def test_post_tag_checks_against_the_tagged_record(ps):
    """Hygiene M1: post-tag work runs only on the tagged code and design (or a logged
    design deviation); the record lives in a tracked file."""
    text = ps.write_prereg("# doc\n", {"stimulus_hash": "abc", "frozen_code_hash": "f"})
    rec = ps.parse_prereg(text)
    assert rec == {"stimulus_hash": "abc", "frozen_code_hash": "f"}
    text2 = ps.write_prereg(text, {"stimulus_hash": "def"})
    assert text2.count(ps.PREREG_BEGIN) == 1 and ps.parse_prereg(text2)["stimulus_hash"] == "def"
    assert ps.prereg_problems(rec, [], "abc", []) == []
    assert ps.prereg_problems(rec, ["src/apexzero/interp/analysis.py"], "abc", [])
    assert ps.prereg_problems(rec, [], "xyz", [])
    assert ps.prereg_problems(rec, [], "xyz", [{"category": "design", "design_hash": "xyz"}]) == []
    assert ps.prereg_problems(None, [], "abc", [])


def test_confirmatory_analysis_needs_complete_inputs(ps):
    """Hygiene M5: the one-shot run is refused on partial inputs. Ruling R18: the six
    replicates' finals (with det, nat and mat groups) are part of the complete inputs. R18
    review finding 1: so are the rollouts of the two snapshots before every generalist final
    (protocol 8's stability rule), det and nat (det alone once budget rule 3 cut the
    checkpoint stochastic runs). Ruling R19a: their det groups must hold their tier G runs."""
    every = tuple(agents.AGENTS)
    assert len(every) == 12
    finals = {a: SimpleNamespace(timesteps=1) for a in every}
    groups = {a: {"det", "nat", "mat"} for a in every}
    hist = {a: [{"det", "nat"}, {"det", "nat"}] for a in agents.generalists()}
    tg = {a: [0, 0] for a in agents.generalists()}       # no tier G run missing (R19a)
    ok = agents.Manifest(band=0.075, validation={
        "step2": {"passed": True}, "step3": {"passed": True}, "dry_run": {"passed": True},
        "controls": {"T": {"status": "pass"}}, "anchored_keys": {"P1": {"errors": 0}}})

    def probs_of(m=ok, fin=finals, grp=groups, agent=None, h=hist, g=tg):
        return ps.confirmatory_problems(m, fin, grp, agent, h, g)

    assert probs_of() == []
    assert probs_of(agent=["G1"])                                                # --agent
    assert probs_of(fin={a: v for a, v in finals.items() if a != "G3"})
    assert probs_of(grp={**groups, "S2": {"det"}})
    # the step-8 gate waits for the replicates' finals too
    probs = probs_of(fin={a: v for a, v in finals.items() if a != "G2s1"})
    assert probs == ["finals missing from the manifest: ['G2s1']"]
    assert probs_of(grp={**groups, "G3s2": {"det", "nat"}})
    no_band = agents.Manifest(validation=ok.validation)
    assert any("band" in p for p in probs_of(m=no_band))
    no_ak = agents.Manifest(band=0.075, validation={k: v for k, v in ok.validation.items()
                                                    if k != "anchored_keys"})
    assert any("step 7" in p for p in probs_of(m=no_ak))
    # history rollouts: missing snapshots, a missing det group, a missing nat group; budget
    # rule 3 drops the checkpoint stochastic runs, so det alone is then enough
    probs = probs_of(h={**hist, "G3": [{"det", "nat"}]}, g={**tg, "G3": [0]})
    assert probs == [("G3: 1 of the 2 snapshots before the final in the manifest (protocol 8's "
                     "stability rule needs both)")]
    probs = probs_of(h={**hist, "G1s2": [set(), {"det", "nat"}]})
    assert len(probs) == 1 and probs[0].startswith("G1s2: snapshot -2 before the final")
    no_nat = {**hist, "G2": [{"det"}, {"det"}]}
    assert len(probs_of(h=no_nat)) == 2
    ok.budget["cuts"] = ["rule 3: checkpoint stochastic and tier E matched-noise runs dropped"]
    assert probs_of(h=no_nat) == []
    # specialists have no stability rule: no history needed
    assert probs_of(h={a: v for a, v in hist.items() if a != "S1"}) == []
    # R19a: a history det group without its tier G runs (made before the ruling), or one
    # whose tier G runs were not checked, is not complete input
    probs = probs_of(g={**tg, "G2s1": [4, 0]})
    assert probs == [("G2s1: snapshot -2 before the final lacks 4 of its tier G deterministic "
                      "runs (ruling R19a: protocol 8's gate needs the null straights; rollouts "
                      "--snapshots last3 runs them)")]
    probs = probs_of(g={a: v for a, v in tg.items() if a != "G1"})
    assert len(probs) == 2 and all(p.startswith("G1: snapshot -") and "not checked" in p
                                   for p in probs)


def test_history_entries_are_the_two_snapshots_before_the_final(ps):
    """R18 review finding 1: the confirmatory gate reads the manifest's two snapshots before
    each generalist final (by timesteps; errored snapshots skipped)."""
    def e(agent, ts, final=False, error=""):
        return SimpleNamespace(agent=agent, timesteps=ts, is_final=final, error=error)

    m = SimpleNamespace(snapshots=[e("G1", 30), e("G1", 99, True), e("G1", 10), e("G1", 20),
                                   e("G1s1", 5, error="bad"), e("G1s1", 7), e("G1s1", 9, True),
                                   e("S1", 1, True)])
    assert [x.timesteps for x in ps._history_entries(m, "G1")] == [20, 30]
    assert [x.timesteps for x in ps._history_entries(m, "G1s1")] == [7]
    assert ps._history_entries(m, "S1") == [] and ps._history_entries(m, "G2") == []


# ------------------------------------------------------------------------- step 7 helpers


def test_p6_checkpoints_first_gate_passing_two_log_spaced_and_final(ps):
    """Fidelity M4: four checkpoints per generalist, not every one."""
    ts = [10, 20, 40, 80, 160, 200]
    gate = {10: False, 20: True, 40: True, 80: True, 160: True, 200: True}
    assert ps.select_p6_checkpoints(ts, gate) == [20, 40, 80, 200]
    assert ps.select_p6_checkpoints(ts, {t: False for t in ts}) == []
    assert ps.select_p6_checkpoints([5, 200], {5: True, 200: True}) == [5, 200]


def _anchor_design():
    lm = Landmarks(1000.0, 1010.0, 1020.0, 1030.0, 1040.0)

    def stim(i, R):
        geo = Geometry(R, 90.0, np.nan, R, 90.0, 40.0, 0.0, 1020.0, 1, False, 6.0)
        return Stimulus(i, f"s{i}", "synthetic", ("A",), 1, -1, lm, geo, None)

    bank = SimpleNamespace(stimuli=[stim(0, 30.0), stim(1, 30.0), stim(2, 15.0)])
    cells = [Cell(0, 0, 0, ("A",), "A1", 86.0, 600.0),
             Cell(1, 1, 1, ("B",), "B", 61.0, 600.0),
             Cell(2, 1, 2, ("B",), "B", 61.0, 250.0),
             Cell(3, 2, 3, ("A",), "A1", 86.0, 600.0)]
    regimes = {a: SimpleNamespace(q1=12.0, q10=20.0, r_clip_max=np.nan, r_clip=())
               for a in ("G1", "S1")}
    return bank, cells, regimes


def test_anchored_cells_cover_tier_b_free_approaches_for_generalists(ps):
    """Fidelity M3: the generalist finals get re-anchored keys on tier B D = 600 cells (P1);
    the specialists on A1 and tier B D = 600."""
    bank, cells, reg = _anchor_design()
    ids = lambda xs: [c.cell_id for c in xs]
    assert ids(ps.anchored_cells(cells, bank, reg, "G1", "P0", True)) == [0, 3]
    assert ids(ps.anchored_cells(cells, bank, reg, "G1", "P1", True)) == [1]
    assert ids(ps.anchored_cells(cells, bank, reg, "S1", "P1", False)) == [0, 1, 3]
    assert ids(ps.anchored_cells(cells, bank, reg, "G1", "P2", True)) == [0, 3]


# ------------------------------------------------------------------------- analysis cache


def test_run_table_cache_follows_its_inputs(ps, tmp_path, monkeypatch):
    """Hygiene M4: a cached run table is rebuilt when the keys, design or rollouts change."""
    from apexzero.interp import analysis as A
    from apexzero.interp import rollout

    f = tmp_path / "roll" / "G1.npz"
    f.parent.mkdir()
    f.write_bytes(b"x")
    monkeypatch.setattr(io, "ANALYSIS_DIR", tmp_path / "analysis")
    monkeypatch.setattr(io, "rollout_path", lambda a, t: f)
    monkeypatch.setattr(ps, "ROOT", tmp_path)
    monkeypatch.setattr(rollout, "load_rollouts", lambda p: ({}, {}))
    calls = []

    def fake_rt(*a, **k):
        calls.append(1)
        return {"cell": np.array([len(calls)])}

    monkeypatch.setattr(A, "run_table", fake_rt)
    t1 = ps._run_table_cached("G1", 1, None, [], None, None, False, "keys-v1")
    t2 = ps._run_table_cached("G1", 1, None, [], None, None, False, "keys-v1")
    assert len(calls) == 1 and t2["cell"][0] == t1["cell"][0]          # reused
    ps._run_table_cached("G1", 1, None, [], None, None, False, "keys-v2")
    assert len(calls) == 2                                              # keys changed
    f.write_bytes(b"xy")
    ps._run_table_cached("G1", 1, None, [], None, None, False, "keys-v2")
    assert len(calls) == 3                                              # rollouts changed
    ps._run_table_cached("G1", 1, None, [], None, None, True, "keys-v2")
    assert len(calls) == 4                                              # --force / confirmatory


def test_a_post_tag_design_change_is_a_logged_deviation(ps, monkeypatch, tmp_path):
    """Hygiene M1: after the tag a design or key stage runs only with a stated reason, which
    is logged; the design it saves is then accepted (and only then)."""
    path = tmp_path / "manifest.json"
    agents.save_manifest(agents.Manifest(), path)
    real_update = agents.update_manifest
    monkeypatch.setattr(ps, "_tag_exists", lambda: True)
    monkeypatch.setattr(agents, "update_manifest", lambda fn, p=path: real_update(fn, p))
    with pytest.raises(SystemExit, match="post-tag-deviation"):
        ps._refuse_after_tag("the stimuli stage", SimpleNamespace(post_tag_deviation=None))
    ps._refuse_after_tag("the stimuli stage",
                         SimpleNamespace(post_tag_deviation="ruling Q1: real tiers after the tag"))
    ps._log_design_deviation("newhash", "stimuli stage after the tag")
    m = agents.load_manifest(path)
    assert [d["category"] for d in m.deviations] == ["stage", "design"]
    assert ps.prereg_problems({"stimulus_hash": "old"}, [], "newhash", m.deviations) == []
    ps._POST_TAG_REASON.clear()
