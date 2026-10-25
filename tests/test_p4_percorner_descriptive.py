"""scripts/p4_percorner_descriptive.py (PS-P4, descriptive and post hoc): the per-cell
completion rule, the reproduction check against classes.json, the d_OOD window lengths and the
lockbox refusal, on planted files. Fast and synthetic; no stored output is read."""

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from apexzero.tracks.corners import R_DEV, R_TEST

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


@pytest.fixture(scope="module")
def pc():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("p4_percorner_descriptive")
    finally:
        sys.path.remove(str(SCRIPTS))


def runs(cells_done: dict[int, list[bool]], extra_sto: bool = True) -> dict:
    """A planted runs table: deterministic phase runs per cell, plus a stochastic row that
    must be ignored."""
    mode, cell, done = [], [], []
    for c, flags in cells_done.items():
        for f in flags:
            mode.append("det")
            cell.append(c)
            done.append(f)
        if extra_sto:
            mode.append("sto")
            cell.append(c)
            done.append(True)
    return {"mode": np.array(mode), "cell": np.array(cell), "completed": np.array(done)}


def test_per_cell_rule(pc):
    r = pc.per_cell(runs({1: [True, True, True, False], 2: [True, True, False, False],
                          3: [False] * 4}), {1: 0.5, 2: 2.0})
    assert r[1] == {"completes": 3, "n": 4, "complete": True, "d_ood": 0.5}
    assert r[2]["complete"] is False and r[2]["completes"] == 2 and r[2]["n"] == 4
    assert r[3]["complete"] is False and r[3]["d_ood"] is None
    # ceil(3n/4) for other counts, as the frozen cond_table scales it.
    assert pc.per_cell(runs({5: [True] * 3}, False), {})[5]["complete"] is True
    assert pc.per_cell(runs({5: [True, True, False]}, False), {})[5]["complete"] is False
    assert pc.per_cell(runs({5: [True] * 7 + [False]}, False), {})[5]["complete"] is True


def test_keyed_refuses_duplicates(pc):
    meta = {1: {"condition": "c", "corner": 7}, 2: {"condition": "c", "corner": 7}}
    rec = {"completes": 4, "n": 4, "complete": True, "d_ood": 1.0}
    assert pc.keyed({1: rec}, meta) == {("c", 7): rec}
    with pytest.raises(pc.CheckFailed):
        pc.keyed({1: rec, 2: rec}, meta)


def planted_recs() -> dict:
    def rec(ok, d):
        return {"completes": 4 if ok else 0, "n": 4, "complete": ok, "d_ood": d}
    return {("A", 0): rec(True, 1.0), ("A", 7): rec(False, 3.0), ("A", 14): rec(True, 2.0),
            ("A", 2): rec(False, 99.0),  # not a braking corner: ignored
            ("B", 7): rec(True, None), ("B", 14): rec(True, 4.0)}


def test_by_condition_matches_completion_by_condition(pc):
    out = pc.by_condition(planted_recs(), [0, 7, 14])
    assert out["A"]["corners"] == [0, 7, 14]
    assert math.isclose(out["A"]["C"], 2 / 3) and out["A"]["D"] == 2.0
    assert out["B"]["C"] == 1.0 and out["B"]["D"] == 4.0  # a missing d_OOD is skipped


def test_check_against_classes(pc):
    recs, braking = planted_recs(), [0, 7, 14]
    good = {"C": {"A": 2 / 3, "B": 1.0, "Z": 0.0}, "details": {"d_ood": {"A": 2.0, "B": 4.0}}}
    assert pc.check_against_classes(recs, braking, good, {"B": [7, 14]}) == 2
    bad_c = {"C": {"A": 1.0}, "details": {"d_ood": {"A": 2.0}}}
    with pytest.raises(pc.CheckFailed, match="C "):
        pc.check_against_classes(recs, braking, bad_c, {})
    bad_d = {"C": {"A": 2 / 3}, "details": {"d_ood": {"A": 2.5}}}
    with pytest.raises(pc.CheckFailed, match="d_OOD"):
        pc.check_against_classes(recs, braking, bad_d, {})
    missing_d = {"C": {"A": 2 / 3}, "details": {"d_ood": {}}}
    with pytest.raises(pc.CheckFailed, match="d_OOD"):
        pc.check_against_classes(recs, braking, missing_d, {})
    with pytest.raises(pc.CheckFailed, match="certified"):
        pc.check_against_classes(recs, braking, good, {"B": [7]})


def write_build(base: Path, track: str) -> None:
    (base / "build").mkdir(parents=True)
    stim = [{"stim_id": 0, "landmarks": {"s_c0": 1000.0}, "source": {"track": track}},
            {"stim_id": 1, "landmarks": {"s_c0": 900.0}, "source": {"track": track}}]
    (base / "build" / "stimuli.json").write_text(json.dumps({"stimuli": stim}))
    meta = [{"cell": 10, "condition": "a", "corner": 7, "owner": "S1", "registered": True},
            {"cell": 11, "condition": "a", "corner": 2, "owner": "S1", "registered": True},
            {"cell": 12, "condition": "b", "corner": 14, "owner": "S1", "registered": False}]
    (base / "build" / "cells_meta.json").write_text(json.dumps({"braking": [7, 14],
                                                               "cells": meta}))


def test_windows(pc, tmp_path):
    write_build(tmp_path, R_DEV[0])
    (tmp_path / "specialists" / "new").mkdir(parents=True)
    np.savez(tmp_path / "specialists" / "new" / "S1.npz",
             det_cell=np.array([10, 10, 11, 12], np.int32),
             det_stim=np.array([0, 0, 0, 1], np.int32),
             det_phase=np.array([0, 1, 0, 0], np.int16),
             det_s0=np.array([600.0, 500.0, 700.0, 450.0]))
    read = {}
    w = pc.windows(tmp_path, ("S1", "S2"), [7, 14], read)
    # corner 7: phase 0 only (1050 - 600); corner 2 is not braking; corner 14 uses its
    # own stimulus's s_c0 (950 - 450).
    assert w == {7: [450.0, 450.0], 14: [500.0, 500.0]}
    assert any(k.endswith("S1.npz") for k in read)


def test_lockbox_refusal(pc, tmp_path):
    ok, sealed = tmp_path / "ok", tmp_path / "sealed"
    write_build(ok, R_DEV[0])
    assert pc.guard_tracks(ok) == [R_DEV[0]]
    write_build(sealed, R_TEST[0])
    with pytest.raises(PermissionError):
        pc.guard_tracks(ok, sealed)
