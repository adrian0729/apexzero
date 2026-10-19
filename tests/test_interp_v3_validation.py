"""Validation V3 in the script (review of the V3 verify fix): ``validate --force`` reaches
every key of the pair and V7 jobs, and the merged V3 verdict is judged at V8's band and
fails when a cell could not be judged. Nothing is solved and no file is written."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apexzero.interp import answer_key as ak
from apexzero.interp.stimuli import Cell


@pytest.fixture(scope="module")
def ps():
    path = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("psychophysics")
    finally:
        sys.path.remove(path)


def test_validate_jobs_pass_force_to_the_pair_and_v7_jobs(ps, monkeypatch):
    """``_key_job`` built ``force`` and then dropped it for 'validate_pair' and 'v7', so
    ``validate --force`` re-solved nothing there (the 12 stale actuation records stayed)."""
    seen = []
    monkeypatch.setattr(ps, "_validate_pair", lambda bank, cells, tables, cache, q_cv,
                        force=False: seen.append(("pair", force)) or {})
    monkeypatch.setattr(ps, "_validate_v7_cell", lambda bank, cell, tables, cache, q_cv,
                        force=False: (seen.append(("v7", force)) or {}, 1.0))
    monkeypatch.setattr(ps, "_W", {"bank": None, "tables": None, "cache": None})
    for f in (True, False):
        for what in ("validate_pair", "v7"):
            out = ps._key_job(dict(what=what, cells=[SimpleNamespace(cell_id=0)], q_cv=0.9,
                                   force=f))
            assert "error" not in out, out.get("error")
    assert seen == [("pair", True), ("v7", True), ("pair", False), ("v7", False)]


def _pair_design():
    stims = [SimpleNamespace(direction=1), SimpleNamespace(direction=-1)]
    cells = [Cell(i, i, 1, ("V",), "A1", 86.0, 600.0, W=800, P=4.0) for i in (0, 1)]
    return SimpleNamespace(stimuli=stims), cells


def test_validate_pair_force_hands_every_key_call_one_fresh_view(ps, monkeypatch, tmp_path):
    """With force, V2, the nominal key, V3, V5 and the backed-off key all read through one
    ``FreshCache`` (each key re-solved once, later reads see the new record); without it
    they read the cache itself. The V7 job likewise."""
    bank, pair = _pair_design()
    cache = ak.KeyCache(tmp_path)
    got = []

    def cache_of(a, k, pos):
        return k["cache"] if "cache" in k else a[pos]

    nom = SimpleNamespace(key_id="n", cpu_seconds=1.0, status="verified")
    monkeypatch.setattr(ak, "v2_horizon", lambda *a, **k: got.append(cache_of(a, k, 2)) or {})
    monkeypatch.setattr(ak, "full_recipe", lambda *a, **k: got.append(cache_of(a, k, 11)) or nom)
    monkeypatch.setattr(ak, "v3_objective", lambda *a, **k: got.append(cache_of(a, k, 2)) or {})
    monkeypatch.setattr(ak, "v5_mirror", lambda *a, **k: got.append(cache_of(a, k, 2)) or {})
    monkeypatch.setattr(ak, "warm_key", lambda *a, **k: got.append(cache_of(a, k, 7)) or nom)
    monkeypatch.setattr(ak, "phase_cycle", lambda *a, **k: 4.0)
    monkeypatch.setattr(ps, "_braking", lambda rec, c: True)
    ps._validate_pair(bank, pair, None, cache, 0.9, force=True)
    assert len(got) == 5 and isinstance(got[0], ak.FreshCache)
    assert all(g is got[0] for g in got) and got[0].cache is cache
    got.clear()
    ps._validate_pair(bank, pair, None, cache, 0.9)
    assert len(got) == 5 and all(g is cache for g in got)
    got.clear()
    ps._validate_v7_cell(bank, pair[0], None, cache, 0.9, force=True)
    assert len(got) == 4 and isinstance(got[0], ak.FreshCache) and all(g is got[0] for g in got)


def _by_id():
    ap = {0: "A1", 1: "A1", 2: "imposed", 3: "imposed", 4: "A2", 5: "A2"}
    return {i: SimpleNamespace(cell_id=i, approach=a) for i, a in ap.items()}


def _v3(cells: dict) -> dict:
    return {"name": "V3", "passed": True, "details": {"cells": cells}}


def test_v3_merge_judges_at_v8_band_and_fails_cells_it_could_not_judge(ps):
    """The merged V3 used the pair jobs' expected band (0.075) and passed with every cell
    ('verified', 'unsolved'), as validation.json shows. It is now judged at V8's b, and a
    cell without a verified key, or an A1 / A2 pair job that errored, fails it."""
    pr = [dict(label="pair 1", cells=[0, 1], validation={"V3": _v3({"0": {"objective_e_ve": -0.045}})}),
          dict(label="pair 2", cells=[2, 3], validation={}),         # imposed: no V3 by design
          dict(label="pair 3", cells=[4, 5], validation={"V3": _v3({"4": {"objective_e_ve": 0.01}})})]
    v = ps._v3_merge(pr, _by_id(), 0.10)
    assert v["passed"] and v["band"] == 0.10 and v["blocking"] == [] and v["missing_pairs"] == []
    assert set(v["cells"]) == {"pair 1", "pair 2", "pair 3"}
    v = ps._v3_merge(pr, _by_id(), float("nan"))                    # no b from V8: 0.075
    assert not v["passed"] and v["blocking"] == ["0"] and v["band"] == ak.BAND_EXPECTED
    errored = [*pr[:2], dict(label="pair 3", cells=[4, 5], error="Traceback ...")]
    v = ps._v3_merge(errored, _by_id(), 0.10)
    assert not v["passed"] and v["missing_pairs"] == ["pair 3"]
    stale = [dict(label="pair 1", cells=[0, 1], validation={"V3": _v3(
        {"0": {"status": ["verified", "unsolved"]}})})]
    v = ps._v3_merge(stale, _by_id(), 0.10)
    assert not v["passed"] and v["unevaluated"] == ["0"] and v["blocking"] == []
    assert not ps._v3_merge([], _by_id(), 0.10)["passed"]           # nothing judged
    assert np.isfinite(v["band"])
