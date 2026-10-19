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
    """Hygiene M5: the one-shot run is refused on partial inputs."""
    six = ("S1", "S2", "S3", "G1", "G2", "G3")
    finals = {a: SimpleNamespace(timesteps=1) for a in six}
    groups = {a: {"det", "nat", "mat"} for a in six}
    ok = agents.Manifest(band=0.075, validation={
        "step2": {"passed": True}, "step3": {"passed": True}, "dry_run": {"passed": True},
        "controls": {"T": {"status": "pass"}}, "anchored_keys": {"P1": {"errors": 0}}})
    assert ps.confirmatory_problems(ok, finals, groups, None) == []
    assert ps.confirmatory_problems(ok, finals, groups, ["G1"])               # --agent
    assert ps.confirmatory_problems(ok, {a: v for a, v in finals.items() if a != "G3"}, groups,
                                    None)
    assert ps.confirmatory_problems(ok, finals, {**groups, "S2": {"det"}}, None)
    no_band = agents.Manifest(validation=ok.validation)
    assert any("band" in p for p in ps.confirmatory_problems(no_band, finals, groups, None))
    no_ak = agents.Manifest(band=0.075, validation={k: v for k, v in ok.validation.items()
                                                    if k != "anchored_keys"})
    assert any("step 7" in p for p in ps.confirmatory_problems(no_ak, finals, groups, None))


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
