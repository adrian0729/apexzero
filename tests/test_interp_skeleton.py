"""The psychophysics package's shared contract: constants, schemas and the io helpers."""

import importlib
import math

import numpy as np
import pytest

from apexzero.interp import constants as C
from apexzero.interp import io

MODULES = ("constants", "io", "support", "stimuli", "measures", "rollout", "agents",
           "controls", "answer_key", "probe", "analysis")


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name):
    importlib.import_module(f"apexzero.interp.{name}")


def test_strip_layout_fits_the_protocol():
    assert C.STRIP_LENGTH == 5118.0
    assert C.S_TRUNC == 4598.0
    # Longest corner (R 300 m, 90 deg) ends well before truncation, with 250 m to spare.
    longest = 300.0 * math.pi / 2
    assert C.S_C + longest + C.COMPLETE_AFTER_EXIT < C.S_TRUNC
    # A1 starts at s0 = S_C - 600 - one phase cycle (< 4.4 m), inside the strip.
    assert C.S_C - C.A1_D - 310 / 3.6 * C.DT > 0.0
    assert len(C.R_BRAKING) == 17 and len(C.L_TYP_R_NODES) == len(C.L_TYP_BIN_EDGES) + 1


def test_corner_spec_geometry():
    from apexzero.interp.stimuli import CornerSpec

    tri = CornerSpec(R=6.0, theta_deg=90.0, direction=1, l_r=10.0)
    assert 1.0 / tri.kappa_peak == pytest.approx(6.366, abs=1e-3)   # protocol 3.1: R_eff 6.37
    assert tri.is_triangle and tri.L_c == pytest.approx(20.0)
    arc = CornerSpec(R=13.1, theta_deg=180.0, direction=-1, l_r=33.0)
    assert arc.kappa_peak == pytest.approx(1 / 13.1)
    assert arc.L_c == pytest.approx(math.pi * 13.1 + 33.0)


def test_schemas_are_consistent():
    from apexzero.interp.analysis import PAIR_COLUMNS, RUN_COLUMNS
    from apexzero.interp.measures import METRIC_COLUMNS, ResponseMetrics
    from apexzero.interp.stimuli import D_CONDITIONS

    assert len(set(RUN_COLUMNS)) == len(RUN_COLUMNS)
    assert len(D_CONDITIONS) == len(set(D_CONDITIONS)) == 23
    assert "e_ve" in PAIR_COLUMNS and "pair" in PAIR_COLUMNS
    assert set(METRIC_COLUMNS) <= set(RUN_COLUMNS)
    ResponseMetrics()   # every field has a default


def test_agent_registry_matches_the_protocol():
    from apexzero.interp.agents import AGENTS

    # Protocol 8's six agents and ruling R18's six seed replicates.
    assert set(AGENTS) == {"S1", "S2", "S3", "G1", "G2", "G3",
                           "G1s1", "G1s2", "G2s1", "G2s2", "G3s1", "G3s2"}
    assert AGENTS["G2"].grammar.tilt_fraction == 0.3
    assert AGENTS["G3"].grammar.tilt_alpha == 2.0
    assert AGENTS["S2"].prospective_after == 122.3e6
    assert [AGENTS[a].seed for a in ("S1", "S2", "S3", "G1", "G2", "G3")] == [1, 2, 3, 0, 0, 0]


def test_table_roundtrip(tmp_path):
    t = {"a": np.arange(3), "b": np.array(["x", "yy", "z"]), "c": np.array([1.0, np.inf, np.nan])}
    path = io.save_table(tmp_path / "t.npz", t)
    back = io.load_table(path)
    assert back["b"].tolist() == ["x", "yy", "z"]
    np.testing.assert_array_equal(back["c"], t["c"])
    with pytest.raises(ValueError):
        io.save_table(tmp_path / "bad.npz", {"a": np.arange(3), "b": np.arange(2)})


def test_json_and_hashes(tmp_path):
    from apexzero.interp.stimuli import Landmarks

    lm = Landmarks(1000.0, 1010.0, 1020.0, 1030.0, 1040.0)
    io.write_json(tmp_path / "x.json", {"lm": lm, "v": np.float32(1.5), "inf": float("inf")})
    d = io.read_json(tmp_path / "x.json")
    assert d["lm"]["s_apex"] == 1020.0 and d["inf"] == "inf"
    a = np.arange(4.0)
    assert io.sha256_arrays(a) == io.sha256_arrays(a.copy())
    assert io.sha256_arrays(a) != io.sha256_arrays(a.astype(np.float32))


def test_stage_log(tmp_path, monkeypatch):
    import os
    import stat

    monkeypatch.setattr(io, "RUNS_DIR", tmp_path)
    (tmp_path / "activity.md").write_text("- 03:30  someone else's note\n")
    with io.StageLog("unit", total=2, every=0.0) as log:
        notes = (tmp_path / "activity.md").read_text()
        # One visible line per process (pid), no markup; other notes stay.
        assert f"psychophysics unit [pid {os.getpid()}]: running" in notes
        assert "<!--" not in notes and "someone else's note" in notes
        log.step()
        log.step()
    text = (tmp_path / "psychophysics-unit.log").read_text()
    assert "2/2" in text and "done unit" in text
    assert "psychophysics unit" not in (tmp_path / "activity.md").read_text()
    # Atomic writes keep the mode a plain open() would give (mkstemp's is 0600).
    mode = stat.S_IMODE((tmp_path / "activity.md").stat().st_mode)
    assert mode == io._FILE_MODE and mode & 0o044 == 0o044 & io._FILE_MODE


def test_seed_streams_are_disjoint_from_noise():
    assert min(C.STREAMS.values()) + C.STREAM_OFFSET > 10**5
    r1, r2 = io.stream_rng("support"), io.stream_rng("support")
    assert r1.integers(1 << 30) == r2.integers(1 << 30)


def test_script_parses(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    ps = importlib.import_module("psychophysics")
    args = ps.build_parser().parse_args(["analyse", "--confirmatory"])
    assert args.stage == "analyse" and args.confirmatory
    assert set(ps.STAGES) == {n[len("stage_"):].replace("_", "-") for n in dir(ps)
                              if n.startswith("stage_")}
