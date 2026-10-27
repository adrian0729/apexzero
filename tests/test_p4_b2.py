"""scripts/p4_b2.py (P4, option B2; docs/prereg/p4-b2-specialists.md) and its run list
scripts/p4b2_runs.py: the corner-matched reading's cell types and classes, the reading table, the
recipe rule with its pooling and key wording, P4's classes on the 15-condition F6 set, the tier A
and tier G pair columns, the manifest's box sha256 list and load, the tag gate and the frozen pins
byte for byte (real git, in a scratch repository), the pre-tag refusals, the run-once write, the
lockbox refusal and the buffer streams.

Planted tables only; nothing is rolled out and no OCP is solved. Two tests read local data
(S2's run directory and the frozen certified lists' strips) and skip without it."""

import importlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from apexzero.interp import analysis as A
from apexzero.interp.stimuli import COND_A, D_CONDITIONS
from apexzero.tracks.corners import R_DEV, R_TEST

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONST, AFTER = A.F1_SINGLE["width"], A.F1_SINGLE["after"]
S2_RUN = ROOT / "runs" / "hockenheim-specialist-mlp-200M-s2"


@pytest.fixture(scope="module")
def b2():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("p4_b2")
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture(scope="module")
def rm():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("round_manifest")
    finally:
        sys.path.remove(str(SCRIPTS))


# ------------------------------------------------------------------------- the run list


def test_run_list_is_section_2(b2, rm, monkeypatch):
    runs = b2.p4b2_runs.runs()
    assert [r["agent"] for r in runs] == ["S4", "S5", "S6", "S7", "S8"]
    assert [r["seed"] for r in runs] == [4, 5, 6, 7, 8]
    for r in runs:
        k = r["seed"]
        assert r["name"] == f"hockenheim-specialist-mlp-200M-s{k}"
        assert r["cmd"] == ["scripts/train.py", "--tracks", "Hockenheim", "--steps", "200e6",
                            "--seed", str(k), "--name", r["name"]]
        assert r["config"] == f"runs/{r['name']}/config.json"
    monkeypatch.chdir(ROOT)                     # round_manifest reads tracks/corners.py there
    assert rm.run_list_problems(runs) == []
    # Every other argument is train.py's default, and a config train.py writes from it passes
    # round_manifest.py's comparison.
    parser = rm.trainer_parser((SCRIPTS / "train.py").read_text())
    args = vars(parser.parse_args(runs[0]["cmd"][1:]))
    assert args == dict(tracks=["Hockenheim"], steps=200e6, name=runs[0]["name"],
                        network="mlp", num_envs=2048, num_steps=64, lr=3e-4, seed=4,
                        log_every=10)
    config = {"tracks": ["Hockenheim"], "seed": 4,
              "ppo": {"total_timesteps": 200_000_000, "num_envs": 2048, "num_steps": 64,
                      "lr": 3e-4, "network": "mlp"}}
    problems, notes = rm.compare_config(runs[0], config, parser)
    assert problems == [] and any("--log-every" in n for n in notes)


def test_five_lanes_on_one_gpu(b2, rm):
    """The drafted 5 lanes: GPUS=1, LANES_PER_GPU=5 puts every run on GPU 0 at once, with
    gpu_round.sh's memory fraction round(0.9 / 5, 2) = 0.18, the draft's."""
    plan = rm.schedule(b2.p4b2_runs.runs(), 1, 5)
    assert [(lane, gpu, pos) for lane, gpu, pos, _ in plan] == [(k, 0, 0) for k in range(5)]
    assert round(0.9 / 5, 2) == 0.18


def test_buffer_streams_are_distinct(b2):
    ids = b2.check_streams()
    assert [ids[a] for a in ("S4", "S5", "S6", "S7", "S8")] == [135, 136, 137, 138, 139]
    assert [ids[a] for a in ("S1", "S2", "S3")] == [132, 133, 134]
    assert b2.stream_id("S4") == sum(map(ord, "S4"))


# ------------------------------------------------------------------------- Section 4.2


def test_cm_cell_types(b2):
    t = b2.cm_type
    assert t(2.0, [("g", 1.5, False)]) == "fragile"                  # a failure at or below
    assert t(2.0, [("g", 2.0, False), ("h", 3.0, True)]) == "fragile"  # at the same d: rule 1
    assert t(2.0, [("g", 1.0, True), ("h", 2.5, True), ("k", 2.0, True)]) == "keyed"
    assert t(2.0, [("g", 1.0, True), ("h", 2.5, False), ("k", 3.0, True)]) == "mixed"
    assert t(2.0, [("g", 1.0, True), ("h", 1.9, True)]) == "unmatched"  # nothing at or above
    assert t(2.0, []) == "unmatched"
    # A failure below decides before anything above, whatever it holds.
    assert t(2.0, [("g", 0.5, False), ("h", 9.0, True)]) == "fragile"


@pytest.mark.parametrize("counts,cls", [
    ({"fragile": 1, "keyed": 1}, "not testable"),               # n = 2 < 3
    ({"unmatched": 9, "keyed": 2}, "not testable"),             # unmatched cells are not in n
    ({"fragile": 2, "keyed": 2}, "CM-1"),                       # fragile exactly half
    ({"fragile": 2, "keyed": 1, "mixed": 2}, "CM-6"),           # 2 of 5 < half
    ({"keyed": 4, "mixed": 1}, "CM-2"),                         # 4 of 5 = 0.8
    ({"keyed": 3, "mixed": 1}, "CM-6"),                         # 3 of 4 = 0.75 < 0.8
    ({"keyed": 3, "unmatched": 4}, "CM-2"),
    ({"mixed": 3}, "CM-6"),
])
def test_cm_class(b2, counts, cls):
    assert b2.cm_class(counts)["cm"] == cls


def test_cm_reading_on_planted_corners(b2):
    """Failing cells are the failing condition's uncompleted corners with a finite d; each is
    typed against the certified conditions certified at its corner only."""
    comps = {0: ("g1", "g2"), 1: ("g1",), 3: ("g1", "g2"), 4: (), 7: ("g2",)}
    cells = {
        (0, CONST): (False, 2.0), (0, "g1"): (True, 2.5), (0, "g2"): (True, 3.0),   # keyed
        (1, CONST): (False, 2.0), (1, "g1"): (False, 1.0),                          # fragile
        (3, CONST): (False, 2.0), (3, "g1"): (True, 1.0), (3, "g2"): (False, 4.0),  # mixed
        (4, CONST): (False, 2.0),                                                   # unmatched
        (7, CONST): (True, 2.0), (7, "g2"): (False, 0.1),       # completed: not a failing cell
        (8, CONST): (False, float("nan")),                      # no finite d: not a cell
        (14, AFTER): (False, 1.0),                              # not in the failing list
    }
    cells[(0, "g3")] = (False, 0.1)          # certified elsewhere only: not a comparator here
    r = b2.cm_reading(cells, [CONST], comps, braking=(0, 1, 3, 4, 7, 8, 14))
    types = {c["corner"]: c["type"] for c in r["cells"]}
    assert types == {0: "keyed", 1: "fragile", 3: "mixed", 4: "unmatched"}
    assert r["counts"] == {"fragile": 1, "keyed": 1, "mixed": 1, "unmatched": 1}
    assert r["n"] == 3 and r["cm"] == "CM-6"
    # A comparator without a run or a finite d is listed as unused, not compared.
    cells[(3, "g2")] = (False, float("nan"))
    r = b2.cm_reading(cells, [CONST], comps, braking=(3,))
    assert r["cells"][0]["unused"] == ["g2"] and r["cells"][0]["type"] == "unmatched"


def test_comparators_are_section_3s_table(b2):
    cert = b2.certified()
    assert list(cert) == list(b2.CERTIFIED_CORNERS)
    comps = b2.comparators(cert)
    assert comps[14] == ("P4A/F6s/knoise0.5e-3/x1", "P4A/F6s/knoise1e-3/x1",
                         "P4A/F6s/knoise0.5e-3/x2", "P4A/Kstr/1e-3", "P4A/Kstr/8e-3",
                         "P4A/Kstr/16e-3")
    assert len(comps[0]) == 11 and "P4A/Wplus/0.5m" not in comps[16]
    assert all(not n.startswith("F6/") for v in comps.values() for n in v)
    assert b2.B2_F6 == (*A.F6_CONDS, *b2.CERTIFIED_CORNERS) and len(b2.B2_F6) == 15


# ------------------------------------------------------------------------- Section 4.3


def _res(cls, vacuous=None, keys=()):
    return dict(cls=cls, vacuous=vacuous, key_factors=list(keys))


@pytest.mark.parametrize("res,cm,reading,via", [
    (_res(0), None, "not competent", "P4"),
    (_res(-1), None, "undecided", "P4"),
    (_res(1), None, "fragile", "P4"),
    (_res(2, False, ("width",)), {"cm": "CM-1"}, "context-keyed", "P4"),   # CM does not decide
    (_res(2, True), {"cm": "CM-1"}, "fragile", "CM"),
    (_res(2, True, ("after",)), {"cm": "CM-2"}, "context-keyed", "CM"),
    (_res(2, True), {"cm": "CM-6"}, "mixed", "CM"),
    (_res(2, True), {"cm": "not testable"}, "undecided", "CM"),
    (_res(3), None, "corner-keyed", "P4"),
    (_res(4), None, "reads the road", "P4"),
    (_res(5), None, "reads the road", "P4"),
    (_res(6), {"cm": "CM-1"}, "mixed", "P4"),                              # descriptive only
])
def test_reading_table(b2, res, cm, reading, via):
    r = b2.reading_of(res, cm)
    assert (r["reading"], r["via"]) == (reading, via)
    assert r["class4"] == (res["cls"] == 4)
    if reading == "context-keyed":
        assert r["keys"] == res["key_factors"]
    if via == "CM":
        assert r["label"] == {"fragile": "fragile (CM)", "context-keyed": "context-keyed (CM)",
                              "mixed": "mixed (CM)", "undecided": "undecided"}[reading]


def test_missing_reading(b2):
    r = b2.reading_of(None, missing="failed twice in the queue (Section 9)")
    assert r["reading"] == "missing" and r["via"] is None


# ------------------------------------------------------------------------- Section 4.4


def _readings(b2, *pairs):
    return {f"S{k + 4}": b2.reading_of(res, cm) if res is not None else
            b2.reading_of(None, missing="failed twice")
            for k, (res, cm) in enumerate(pairs)}


def test_recipe_fragile_pools_cm(b2):
    rd = _readings(b2, (_res(1), None), (_res(2, True), {"cm": "CM-1"}), (_res(1), None),
                   (_res(6), None), (None, None))
    rc = b2.recipe(rd)
    assert rc["claimed"] == "fragile" and rc["n_via_cm"] == 1 and rc["missing"] == ["S8"]
    assert rc["wording"]["text"].startswith("The Hockenheim specialists are fragile")
    assert rc["n_total"] == 5


def test_recipe_context_keyed_key_wording(b2):
    rd = _readings(b2, (_res(2, False, ("width",)), None),
                   (_res(2, False, ("after", "width")), None),
                   (_res(2, True, ()), {"cm": "CM-2"}), (_res(2, True), {"cm": "CM-2"}),
                   (_res(1), None))
    rc = b2.recipe(rd)
    assert rc["claimed"] == "context-keyed" and rc["n_via_cm"] == 2
    # Most held first; ties in factor order (before, after, width), "no key named" last.
    assert rc["keys"] == [("width", 2), ("", 2), ("after", 1)]
    assert rc["wording"]["text"] == (
        "Context-keyed memorisation, keyed by constant width (2 of 4), no key named (2 of 4) "
        "and the straightened road after the corner (1 of 4). Certified-irrelevant changes at "
        "least as far off the manifold leave them intact, while one real-road factor breaks "
        "them.")
    one = b2.recipe(_readings(b2, *[(_res(2, False, ("width",)), None)] * 3,
                              (_res(6), None), (_res(6), None)))
    assert "keyed by constant width (3 of 3)." in one["wording"]["text"]


def test_recipe_reads_the_road_pools_classes_4_and_5(b2):
    within = b2.recipe(_readings(b2, (_res(4), None), (_res(4), None), (_res(5), None),
                                 (_res(6), None), (_res(6), None)))
    assert within["claimed"] == "reads the road"
    assert within["wording"]["text"] == "Reads the road within its envelope."
    beyond = b2.recipe(_readings(b2, (_res(4), None), (_res(4), None), (_res(4), None),
                                 (_res(5), None), (_res(6), None)))
    assert beyond["wording"]["text"] == "Reads the road beyond its envelope."


@pytest.mark.parametrize("pairs,claimed,key", [
    ([(_res(2, True), {"cm": "not testable"})] * 2 + [(_res(-1), None)]
     + [(_res(1), None)] * 2, "undecided", "undecided"),
    ([(_res(0), None)] * 3 + [(_res(6), None)] * 2, "not competent", "not competent"),
    ([(_res(3), None)] * 3 + [(_res(6), None)] * 2, "corner-keyed", "corner-keyed"),
    ([(_res(6), None)] * 2 + [(_res(2, True), {"cm": "CM-6"})] + [(_res(1), None)] * 2,
     "mixed", "mixed"),
    ([(_res(1), None)] * 2 + [(_res(6), None)] * 2 + [(_res(3), None)], None, "none"),
    # A missing specialist counts against every reading: 2 fragile and 3 missing is no claim.
    ([(_res(1), None)] * 2 + [(None, None)] * 3, None, "none"),
])
def test_recipe_other_readings(b2, pairs, claimed, key):
    rc = b2.recipe(_readings(b2, *pairs))
    assert rc["claimed"] == claimed and rc["wording"]["key"] == key
    assert rc["wording"]["text"] == b2.WORDING[key][0]


def test_registered_recipe_flags_vacuous_class_2(b2):
    res = {"S4": _res(2, True), "S5": _res(2, True), "S6": _res(2, False), "S7": _res(6),
           "S8": None}
    rr = b2.registered_recipe(res)
    assert rr["claimed"] == "context-keyed" and rr["n_vacuous"] == 2
    assert rr["classes"]["S8"] is None


def test_key_text(b2):
    assert b2.key_text([("width", 3)], 3) == "constant width (3 of 3)"
    assert b2.key_text([("width", 3), ("after", 1)], 4) == (
        "constant width (3 of 4) and the straightened road after the corner (1 of 4)")


# ------------------------------------------------------------------------- Section 4.1


def make_design(n_corners: int = 10, n_braking: int = 8) -> dict:
    """tests/test_interp_analysis.py's planted tier D design (as option A's tests use it)."""
    rows = []
    for i in range(n_corners):
        v_i = 20.0 + 4.0 * i
        for c in D_CONDITIONS:
            lam = float(c.split("lambda")[1]) if "lambda" in c else 1.0
            rows.append(dict(corner=i, condition=c, braking=i < n_braking,
                             v_min_star=v_i * math.sqrt(lam),
                             swap_j=(i + 3) % n_braking if c == "F5/swapA" else -1))
    return A._to_table(rows, ("corner", "condition", "braking", "v_min_star", "swap_j"))


DESIGN = make_design()


def _with(cond, mask, **cols):
    out = {k: np.array(v, copy=True) for k, v in cond.items()}
    for k, v in cols.items():
        out[k][mask] = v
    return out


def _add(cond: dict, name: str, complete, d_ood) -> dict:
    """A certified condition on every corner, copied from condition (a)'s rows (complete and
    d_ood scalars or per-corner arrays)."""
    m = A._st(cond, "condition") == COND_A
    new = {k: np.array(v, copy=True)[m] for k, v in cond.items()}
    new["condition"] = np.full(m.sum(), name)
    new["complete"] = np.broadcast_to(np.asarray(complete, bool), m.sum()).copy()
    new["d_ood"] = np.broadcast_to(np.asarray(d_ood, float), m.sum()).copy()
    return {k: np.concatenate([np.asarray(cond[k]), new[k]]) for k in cond}


CERTIFIED = ("P4A/F6s/knoise0.5e-3/x1", "P4A/F6s/knoise1e-3/x1", "P4A/F6s/knoise0.5e-3/x2",
             "P4A/F6s/knoise0.5e-3/x3", "P4A/Kstr/1e-3", "P4A/Kstr/2e-3", "P4A/Kstr/4e-3",
             "P4A/Kstr/8e-3", "P4A/Wplus/0.5m", "P4A/Wplus/1m", "P4A/Kstr/16e-3")


def _with_certified(cond: dict, complete=True, d_ood=0.8) -> dict:
    """The 11 certified conditions added on every corner."""
    for n in CERTIFIED:
        cond = _add(cond, n, complete, d_ood)
    return cond


@pytest.fixture(scope="module")
def planted(b2):
    gen = A.simulate_conds("reader", DESIGN, np.random.default_rng(1), agent="G1")
    base = A.simulate_conds("reader", DESIGN, np.random.default_rng(0), agent="S9")
    cn = base["condition"]
    # Every constant-width F1 condition fails (so width is the key), at d_OOD 2.0.
    wk = _with(base, np.isin(cn, [c for c in A.F1_CONDS if c.endswith("/const")]),
               complete=False, d_ood=2.0)
    vac = _with(wk, np.isin(cn, A.F6_CONDS), d_ood=0.5)
    tier_a = A._to_table([dict(tiers="A", approach="A2", R_seg=20.0, n_runs=8, det_complete=8,
                               excluded="", braking=True, no_cap=False)],
                         ("tiers", "approach", "R_seg", "n_runs", "det_complete", "excluded",
                          "braking", "no_cap"))
    tier_g = A._to_table([dict(tiers="G", det_crashes=0)], ("tiers", "det_crashes"))
    return dict(gen=gen, base=_with_certified(base), vac=vac, tier_a=tier_a,
                tier_g=tier_g, corner_keyed=_with_certified(A.simulate_conds(
                    "corner_keyed", DESIGN, np.random.default_rng(0), agent="S9")))


def test_p4_classes_on_the_b2_set(b2, planted):
    """Planted tables for every class on the 15-condition set, and the registered four giving
    the frozen ``p4_classify``."""
    p, gen = planted, planted["gen"]
    vac = p["vac"]
    cases = {
        0: _with(p["base"], A._st(p["base"], "condition") == COND_A, complete=False),
        -1: _with_certified(vac, True, np.nan),                  # a certified D missing
        1: _with_certified(vac, False, 1.0),                     # failing at or below 2.0
        3: p["corner_keyed"],
        6: _with_certified(vac, False, 3.0),                     # failing far: class 2 blocked
    }
    got = {k: b2.p4_class(v, gen)["cls"] for k, v in cases.items()}
    assert got == {k: k for k in cases}
    non_vac = b2.p4_class(_with_certified(vac, True, 3.0), gen)
    assert non_vac["cls"] == 2 and non_vac["vacuous"] is False
    assert non_vac["details"]["n_f6_at_or_above"] == 11 and non_vac["key_factors"] == ["width"]
    assert non_vac["failing"] == [CONST]
    vacuous = b2.p4_class(_with_certified(vac, True, 0.8), gen)
    assert vacuous["cls"] == 2 and vacuous["vacuous"] is True
    assert b2.p4_class(p["base"], gen, tier_a=p["tier_a"], tier_g=p["tier_g"])["cls"] == 4
    crash = A._to_table([dict(tiers="G", det_crashes=1)], ("tiers", "det_crashes"))
    assert b2.p4_class(p["base"], gen, tier_a=p["tier_a"], tier_g=crash)["cls"] == 5
    for cond in (p["base"], vac, p["corner_keyed"]):
        mine = b2.p4_class(cond, gen, A.F6_CONDS)
        frozen = A.p4_classify(cond, gen)
        assert (mine["cls"], mine["label"]) == (frozen.cls, frozen.label)


def test_vacuous_class_2_is_decided_by_cm(b2, planted):
    """The full chain on planted vacuous class 2 tables: constant width fails at all 8 braking
    corners at d 2.0, and the certified conditions sit at d 0.8 (P4's median D stays below 2.0
    and every certified C above 0.5, so Section 4.1 gives a vacuous class 2 each time). What the
    certified conditions do at single corners then sets the CM class and the reading."""
    braking = tuple(range(8))
    comps = {i: CERTIFIED for i in braking}

    def chain(cond):
        res = b2.p4_class(cond, planted["gen"])
        assert res["cls"] == 2 and res["vacuous"] is True and res["failing"] == [CONST]
        cm = b2.cm_reading(b2.corner_cells(cond), res["failing"], comps, braking)
        return cm, b2.reading_of(res, cm)

    def at(cond, name, corners, **cols):
        m = (A._st(cond, "condition") == name) & np.isin(A._it(cond, "corner"), corners)
        return _with(cond, m, **cols)

    plain = _with_certified(planted["vac"], True, 0.8)
    # K-str 16e-3 far (d 3.0) and completed at 3 corners: keyed there, unmatched elsewhere.
    cm, r = chain(at(plain, "P4A/Kstr/16e-3", [0, 1, 2], d_ood=3.0))
    assert cm["counts"] == {"fragile": 0, "keyed": 3, "mixed": 0, "unmatched": 5}
    assert (cm["cm"], r["reading"], r["label"]) == ("CM-2", "context-keyed",
                                                    "context-keyed (CM)")
    assert r["keys"] == ["width"]
    # The same far corners failed: mixed, CM-6.
    cm, r = chain(at(plain, "P4A/Kstr/16e-3", [0, 1, 2], d_ood=3.0, complete=False))
    assert (cm["counts"]["mixed"], cm["cm"], r["reading"]) == (3, "CM-6", "mixed")
    # Near certified failures at 6 corners (two conditions, 3 corners each, so neither has
    # C <= 0.5): fragile, CM-1.
    cond = at(at(plain, "P4A/Kstr/1e-3", [0, 1, 2], complete=False), "P4A/Kstr/2e-3",
              [3, 4, 5], complete=False)
    cm, r = chain(cond)
    assert (cm["counts"]["fragile"], cm["cm"], r["label"]) == (6, "CM-1", "fragile (CM)")
    # Nothing at or above the failing d anywhere: every cell unmatched, CM not testable.
    cm, r = chain(plain)
    assert (cm["n"], cm["cm"], r["reading"]) == (0, "not testable", "undecided")


# ------------------------------------------------------------------------- tiers A and G


def test_tier_table_follows_pair_values(b2):
    kt = A._to_table([
        dict(cell=0, direction=1, tiers="A", approach="A2", R_seg=20.0, no_cap=False,
             braking=True, key_status="verified", multi_basin=False),
        dict(cell=1, direction=-1, tiers="A", approach="A2", R_seg=20.0, no_cap=False,
             braking=True, key_status="verified", multi_basin=False),
        dict(cell=2, direction=1, tiers="A", approach="A2", R_seg=9.0, no_cap=False,
             braking=True, key_status="verified", multi_basin=False),
        dict(cell=3, direction=1, tiers="A", approach="A2", R_seg=30.0, no_cap=True,
             braking=True, key_status="verified", multi_basin=False),
        dict(cell=4, direction=1, tiers="A", approach="A2", R_seg=30.0, no_cap=False,
             braking=True, key_status="unsolved", multi_basin=False),
        dict(cell=5, direction=1, tiers="A", approach="A2", R_seg=30.0, no_cap=False,
             braking=False, key_status="verified", multi_basin=True),
        dict(cell=6, direction=0, tiers="G", approach="G", R_seg=float("nan"), no_cap=False,
             braking=False, key_status="verified", multi_basin=False),
    ], ("cell", "direction", "tiers", "approach", "R_seg", "no_cap", "braking", "key_status",
        "multi_basin"))
    pair_of = {0: 10, 1: 10, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15}
    rows = []
    for c, done, crash in ((0, 4, 0), (1, 3, 1), (2, 0, 4), (3, 0, 4), (4, 0, 4), (5, 4, 0),
                           (6, 4, 0)):
        for k in range(4):
            rows.append(dict(agent="S4", timesteps=1, mode="det", cell=c, phase=k,
                             completed=k < done, crashed=k < crash))
    rt = A._to_table(rows, ("agent", "timesteps", "mode", "cell", "phase", "completed",
                            "crashed"))
    t = b2.tier_table(rt, kt, pair_of)
    by = {int(p): i for i, p in enumerate(t["pair"])}
    assert t["n_runs"][by[10]] == 8 and t["det_complete"][by[10]] == 7
    assert t["det_crashes"][by[10]] == 1
    assert t["excluded"][by[13]] == "unsolved" and t["excluded"][by[14]] == "multi_basin"
    assert t["excluded"][by[10]] == ""
    tier_a = b2.pio.select(t, A.has_tier(t, "A"))
    # Pair 10 counts (7 of 8); 11 is below 10.7 m, 12 no-cap, 13 unsolved (EXCL_ALL); 14 is
    # multi-basin, which EXCL_ALL keeps (4 of 4).
    assert b2.tier_a_completion(tier_a) == pytest.approx(11 / 12)
    tier_g = b2.pio.select(t, A.has_tier(t, "G"))
    assert A._n(tier_g) == 1 and int(tier_g["det_crashes"][0]) == 0


# ------------------------------------------------------------------------- manifest helpers


def test_check_report_problems(b2):
    report = ("round p4b2: scripts/p4b2_runs.py: 5 runs, 5 seeds\nruns: done 4, failed 1\n"
              "FAILED: 2 problem(s)\n"
              "  hockenheim-specialist-mlp-200M-s6: status 'failed' in runs/p4b2-manifest-0of1"
              ".json\n  hockenheim-specialist-mlp-200M-s6: runs/x/config.json is not here\n")
    probs = b2.check_problems(report)
    assert len(probs) == 2
    assert b2.unexplained(probs, {"hockenheim-specialist-mlp-200M-s6"}) == []
    assert b2.unexplained(probs, set()) == probs
    assert b2.check_problems("tag 'p4-b2-train-v1' not found") is None
    assert b2.unexplained(None, {"x"}) == ["round_manifest.py check did not finish"]


@pytest.mark.skipif(not (S2_RUN / "agent.pkl").exists(), reason="S2's run directory is local")
def test_manifest_entry_on_s2(b2):
    """A finished specialist run gives a full entry (S2 stands in for S4-S8)."""
    run = {"name": S2_RUN.name, "seed": 2}
    entry, probs = b2.manifest_entry("S2", run, {"status": "done"})
    assert probs == []
    assert entry["final"]["timesteps"] == b2.FINAL_TIMESTEPS
    assert len(entry["checkpoints"]) == 16 and entry["final"]["path"].startswith("runs/")
    assert entry["eval"]["completion"] == 1.0 and entry["eval"]["k4_gate"] is True
    assert str(ROOT) not in json.dumps(entry)
    _, probs = b2.manifest_entry("S2", {"name": S2_RUN.name, "seed": 5}, {})
    assert any("seed" in p for p in probs)


# ------------------------------------------------------------------------- freeze, lockbox


def test_frozen_pins_hold(b2):
    assert b2.frozen_problems() == []
    wrong = {"scripts/psychophysics.py": {"blob": "0" * 40, "sha256": "0" * 64}}
    assert len(b2.frozen_problems(wrong)) == 2


def test_lockbox_refusal(b2):
    for name in (R_TEST[0], f"{R_TEST[3]}_mirrored", f"{R_TEST[-1]}-reversed"):
        with pytest.raises(PermissionError):
            b2.hockenheim(name)
    other = next(n for n in R_DEV if n != "Hockenheim")
    with pytest.raises(ValueError, match="Hockenheim only"):
        b2.hockenheim(other)


def test_lockbox_run_list_refused(b2, rm, monkeypatch):
    monkeypatch.chdir(ROOT)
    run = dict(b2.p4b2_runs.runs()[0])
    run["cmd"] = [*run["cmd"][:2], R_TEST[0], *run["cmd"][3:]]
    assert any("lockbox" in p for p in rm.run_list_problems([run]))


def test_write_new_runs_once(b2, tmp_path):
    p = tmp_path / "decisions.json"
    b2.write_new(p, {"a": 1})
    with pytest.raises(SystemExit, match="runs once"):
        b2.write_new(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 1}


# ------------------------------------------------------------------------- the tag gate


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


class ScratchRepo:
    """A bare ``origin`` and a clone in tmp_path with real git and no user config, holding a
    stand-in script, a file under outputs/p4b2/pre/ (force-added) and the record committed
    alone. ``b2.ROOT`` points at the clone and ``b2.BOUND`` at those three paths."""

    def __init__(self, tmp_path: Path, monkeypatch, b2):
        cfg = tmp_path / "gitconfig"
        cfg.write_text("")
        for k, v in {"GIT_CONFIG_GLOBAL": str(cfg), "GIT_CONFIG_NOSYSTEM": "1",
                     "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                     "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}.items():
            monkeypatch.setenv(k, v)
        self.origin, self.clone = tmp_path / "origin.git", tmp_path / "clone"
        subprocess.run(["git", "init", "-q", "--bare", str(self.origin)], check=True)
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.clone)], check=True,
                       capture_output=True)
        (self.clone / ".gitignore").write_text("outputs/\n")
        (self.clone / "scripts").mkdir()
        (self.clone / "scripts" / "p4_b2.py").write_text("# the script\n")
        pre = self.clone / "outputs" / "p4b2" / "pre"
        pre.mkdir(parents=True)
        (pre / "manifest.json").write_text("{}\n")
        self.git("add", ".gitignore", "scripts/p4_b2.py")
        self.git("add", "-f", "outputs/p4b2/pre/manifest.json")
        self.git("commit", "-qm", "code and manifest")
        (self.clone / b2.RECORD_REL).write_text("{}\n")
        self.git("add", b2.RECORD_REL)
        self.git("commit", "-qm", "the record, alone")
        self.git("push", "-q", "origin", "HEAD:refs/heads/main")
        monkeypatch.setattr(b2, "ROOT", self.clone)
        monkeypatch.setattr(b2, "BOUND", ("scripts/p4_b2.py", b2.RECORD_REL, b2.PRE_REL))

    def git(self, *args) -> str:
        return subprocess.run(["git", *args], cwd=self.clone, check=True, capture_output=True,
                              text=True).stdout.strip()

    def tag(self, annotated: bool = True) -> None:
        self.git("tag", *(["-a", "-m", "P4-B2"] if annotated else []), "p4-b2-prereg-v1")
        self.git("push", "-q", "origin", "refs/tags/p4-b2-prereg-v1")


@needs_git
def test_tag_gate_passes_on_the_record_commit(b2, tmp_path, monkeypatch):
    repo = ScratchRepo(tmp_path, monkeypatch, b2)
    with pytest.raises(PermissionError, match="not on origin"):
        b2.tag_gate()
    assert not b2.tag_exists()
    repo.tag()
    got = b2.tag_gate()
    assert got["commit"] == repo.git("rev-parse", "HEAD") == got["head"]
    assert got["tag_object"] != got["commit"]
    assert b2.tag_exists()
    with pytest.raises(SystemExit, match="after the tag"):
        b2.refuse_after_tag("build")
    # A later commit that leaves the bound paths alone is fine.
    (repo.clone / "notes.txt").write_text("later\n")
    repo.git("add", "notes.txt")
    repo.git("commit", "-qm", "later")
    assert b2.tag_gate()["commit"] != b2.tag_gate()["head"]


@needs_git
@pytest.mark.parametrize("change", ["lightweight", "not_record_commit", "script_changed",
                                    "pre_extra_file", "pre_changed", "assume_unchanged",
                                    "skip_worktree", "pre_skip_worktree", "only_v10_on_origin",
                                    "local_tag_differs", "head_before_tag"])
def test_tag_gate_refuses(b2, tmp_path, monkeypatch, change):
    repo = ScratchRepo(tmp_path, monkeypatch, b2)
    if change == "not_record_commit":
        (repo.clone / "notes.txt").write_text("after the record\n")
        repo.git("add", "notes.txt")
        repo.git("commit", "-qm", "not the record")
        repo.git("push", "-q", "origin", "HEAD:refs/heads/main")
    if change == "only_v10_on_origin":       # the tag's name is matched in full, not as a prefix
        repo.git("tag", "-a", "-m", "P4-B2", "p4-b2-prereg-v10")
        repo.git("push", "-q", "origin", "refs/tags/p4-b2-prereg-v10")
        with pytest.raises(PermissionError, match="not on origin"):
            b2.tag_gate()
        return
    repo.tag(annotated=change != "lightweight")
    pre = repo.clone / "outputs" / "p4b2" / "pre"
    if change == "script_changed":
        (repo.clone / "scripts" / "p4_b2.py").write_text("# changed\n")
    if change == "pre_extra_file":
        (pre / "extra.npz").write_bytes(b"x")
    if change == "pre_changed":
        (pre / "manifest.json").write_text('{"a": 1}\n')
    if change in ("assume_unchanged", "skip_worktree"):   # git diff trusts the index for these
        repo.git("update-index", "--" + change.replace("_", "-"), "scripts/p4_b2.py")
        (repo.clone / "scripts" / "p4_b2.py").write_text("# changed\n")
        assert repo.git("diff", "p4-b2-prereg-v1", "--", "scripts/p4_b2.py") == ""
    if change == "pre_skip_worktree":
        repo.git("update-index", "--skip-worktree", "outputs/p4b2/pre/manifest.json")
        (pre / "manifest.json").write_text('{"a": 1}\n')
    if change == "local_tag_differs":
        repo.git("tag", "-f", "-a", "-m", "another tag object", "p4-b2-prereg-v1")
    if change == "head_before_tag":
        repo.git("checkout", "-q", "--detach", "HEAD~1")
    with pytest.raises(PermissionError):
        b2.tag_gate()


@needs_git
def test_frozen_tree_pin_is_byte_level(b2, tmp_path, monkeypatch):
    """A frozen tree's files are compared byte for byte with the pinned tree's blobs, so a file
    marked assume-unchanged or skip-worktree cannot hide an edit."""
    repo = ScratchRepo(tmp_path, monkeypatch, b2)
    (repo.clone / "frozen").mkdir()
    (repo.clone / "frozen" / "code.py").write_text("x = 1\n")
    repo.git("add", "frozen/code.py")
    repo.git("commit", "-qm", "frozen code")
    pins = {"frozen": {"tree": repo.git("rev-parse", "HEAD:frozen")}}
    assert b2.frozen_problems(pins) == []
    for flag in ("--assume-unchanged", "--skip-worktree"):
        repo.git("update-index", flag, "frozen/code.py")
        (repo.clone / "frozen" / "code.py").write_text("x = 2\n")
        assert repo.git("status", "--porcelain", "--", "frozen") == ""
        assert any("byte for byte" in p for p in b2.frozen_problems(pins))
        (repo.clone / "frozen" / "code.py").write_text("x = 1\n")
        repo.git("update-index", flag.replace("--", "--no-"), "frozen/code.py")
    assert b2.frozen_problems(pins) == []


@needs_git
@pytest.mark.parametrize("stage", ["manifest", "buffers", "buffers --check", "starts",
                                   "starts --check", "build", "build --owners check", "reproduce",
                                   "cm-dryrun", "record"])
def test_pre_tag_stages_refuse_after_the_tag(b2, tmp_path, monkeypatch, stage):
    """Section 3: every pre-tag stage, its S1-S3 check variants included, is refused once the
    tag exists (before it reads or writes anything)."""
    repo = ScratchRepo(tmp_path, monkeypatch, b2)
    repo.tag()
    monkeypatch.setattr(b2, "require_frozen", lambda: None)
    name, *rest = stage.split()
    args = _args(check="--check" in rest, owners="check" if "check" in rest[1:] else "new")
    with pytest.raises(SystemExit, match="after the tag"):
        b2.PRE_TAG[name](args)


@pytest.mark.skipif(not (ROOT / "data" / "tracks" / "tumftm" / "Hockenheim.csv").exists(),
                    reason="needs the TUMFTM track data (not in the repo)")
def test_certified_strip_of_corner_14(b2):
    """One certified strip rebuilt by option A's builder matches its frozen per-corner sha256
    (the build stage checks all 90)."""
    cert = b2.certified()
    track = b2.hockenheim()
    dc = b2.ST.hockenheim_corners(track)[14]
    prof = b2.P4A.extension_conditions(dc, track)[0]["P4A/Kstr/16e-3"]
    sha = b2.pio.sha256_arrays(prof.kappa, prof.w_left, prof.w_right)
    assert sha == cert["P4A/Kstr/16e-3"]["strips"]["14"]


@needs_git
def test_tracked_clean_sees_ignored_files_under_pre(b2, tmp_path, monkeypatch):
    repo = ScratchRepo(tmp_path, monkeypatch, b2)
    assert b2.tracked_clean(["scripts/p4_b2.py", b2.PRE_REL]) == []
    (repo.clone / "outputs" / "p4b2" / "pre" / "knn_S4.npz").write_bytes(b"x")
    assert any("files not added" in p for p in b2.tracked_clean([b2.PRE_REL]))
    assert b2.tracked_clean(["scripts/missing.py"]) == ["scripts/missing.py: not tracked"]


# ------------------------------------------------------------------------- rehearsals


def _args(**kw):
    import argparse

    base = dict(force=False, train_tag="p4-b2-train-v1", workers=1, check=False, owners="new")
    return argparse.Namespace(**{**base, **kw})


@pytest.mark.skipif(not (S2_RUN / "agent.pkl").exists(), reason="S2's run directory is local")
def test_manifest_stage_rehearsal(b2, tmp_path, monkeypatch):
    """The manifest stage on S2's run standing in for S4, with S5 failing twice in the queue
    (round_manifest.py check replaced by its report)."""
    s5 = "hockenheim-specialist-mlp-200M-s5"
    monkeypatch.setattr(b2, "NEW", ("S4", "S5"))
    monkeypatch.setattr(b2, "RUNS", {"S4": {"name": S2_RUN.name, "seed": 2},
                                     "S5": {"name": s5, "seed": 5}})
    monkeypatch.setattr(b2, "PRE", tmp_path / "pre")
    report = (f"runs: done 1, failed 1\nFAILED: 1 problem(s)\n  {s5}: status 'failed' in "
              "runs/p4b2-manifest-0of1.json")
    monkeypatch.setattr(b2, "round_check", lambda tag: (1, report))
    box = {"shard": "0/1", "head": "a" * 40, "box_type": "massedcompute_A6000_plus",
           "runs": [{"name": S2_RUN.name, "status": "done"}, {"name": s5, "status": "failed"}]}
    monkeypatch.setattr(b2, "shard_manifests", lambda: [(tmp_path / "m.json", box)])
    (tmp_path / "m.json").write_text(json.dumps(box))
    # The box's sha256 list (Section 8, step 2), in sha256sum's format.
    files = [S2_RUN / "agent.pkl", *sorted((S2_RUN / "ckpt").glob("*.pkl"))]
    listing = tmp_path / "p4b2-box-sha256-0of1.txt"
    listing.write_text("".join(f"{b2.sha256_file(p)}  {p.relative_to(ROOT)}\n" for p in files))
    monkeypatch.setattr(b2, "box_sha256_files", list)
    with pytest.raises(SystemExit, match="sha256sum"):
        b2.stage_manifest(_args())
    monkeypatch.setattr(b2, "box_sha256_files", lambda: [listing])
    b2.stage_manifest(_args())
    man = json.loads((tmp_path / "pre" / "manifest.json").read_text())
    assert list(man["networks"]) == ["S4"] and list(man["missing"]) == ["S5"]
    assert man["networks"]["S4"]["final"]["timesteps"] == b2.FINAL_TIMESTEPS
    assert man["k4_gate"]["per_network"]["S4"]["passes"] is True
    assert man["box_sha256"]["files_equal"] == 17
    assert man["load"]["decided"] is False           # the planted box records no lanes
    assert b2.new_finals(man) == {"S4": (man["networks"]["S4"]["final"]["path"],
                                         man["networks"]["S4"]["final"]["sha256"])}
    with pytest.raises(SystemExit, match="exists"):
        b2.stage_manifest(_args())
    # A pulled file that is not the box's stops the stage.
    lines = listing.read_text().splitlines()
    listing.write_text("\n".join(["0" * 64 + lines[0][64:], *lines[1:]]) + "\n")
    with pytest.raises(SystemExit, match="not the box's"):
        b2.stage_manifest(_args(force=True))
    # Any problem other than a run that failed twice stops the stage.
    monkeypatch.setattr(b2, "round_check", lambda tag: (1, report + "\n  the boxes' GPU "
                                                        "models differ"))
    with pytest.raises(SystemExit, match="other than"):
        b2.stage_manifest(_args(force=True))


def test_read_sha256sum(b2):
    text = f"{'a' * 64}  runs/x/agent.pkl\n{'b' * 64} *runs/x/ckpt/1.pkl\nsha256sum: y: missing\n"
    assert b2.read_sha256sum(text) == {"runs/x/agent.pkl": "a" * 64, "runs/x/ckpt/1.pkl": "b" * 64}


def test_load_summary(b2):
    """Section 2's load (all five at once on GPU 0) holds at GPUS=1 LANES_PER_GPU=5 only."""
    names = [r["name"] for r in b2.p4b2_runs.runs()]

    def recs(gpus, lanes):
        n = gpus * lanes
        return {name: {"gpu": (j % n) % gpus, "lane": j % n} for j, name in enumerate(names)}

    def box(gpus, lanes):
        return [(Path("m.json"), {"shard": "0/1", "gpus": gpus, "lanes_per_gpu": lanes,
                                  "gpu_count": gpus})]

    assert b2.load_summary(box(1, 5), recs(1, 5))["decided"] is True
    for gpus, lanes in ((1, 4), (4, 4)):             # S8 after S4, or beside it on GPU 0
        got = b2.load_summary(box(gpus, lanes), recs(gpus, lanes))
        assert got["decided"] is False and got["deviation"]


HAVE_CHECK = (ROOT / "outputs" / "p4b2" / "check" / "rollouts" / "GA" / "results.json").exists()


@pytest.mark.skipif(not HAVE_CHECK, reason="needs S1-S3's check rollouts (reproduce, local)")
def test_classify_stage_rehearsal(b2, tmp_path, monkeypatch):
    """The post-tag classify stage, end to end, on S1-S3's check rollouts standing in for
    S4-S8, with S4 missing (failed twice) and S5 class 0 by Section 9; the gate is replaced."""
    check = ROOT / "outputs" / "p4b2" / "check"
    build, roll = tmp_path / "build", tmp_path / "rollouts"
    build.mkdir()
    roll.mkdir()
    meta = json.loads((check / "build" / "cells_meta.json").read_text())
    meta["excluded"] = {"S5": "class 0: the real lap ends before a braking corner's start"}
    (build / "cells_meta.json").write_text(json.dumps(meta))
    shutil.copy(check / "build" / "keys.npz", build / "keys.npz")
    for d in ("D", "GA"):
        (roll / d).symlink_to(check / "rollouts" / d)
    (roll / "rollouts.json").write_text("{}")
    man = {"networks": {}, "missing": {"S4": {"reason": "failed twice in the queue"}},
           "k4_gate": {"per_network": {a: dict(completion=1.0, gap=0.05, passes=True)
                                       for a in ("S1", "S2", "S3")}}}
    monkeypatch.setattr(b2, "NEW", ("S1", "S2", "S3", "S4", "S5"))
    monkeypatch.setattr(b2, "BUILD", build)
    monkeypatch.setattr(b2, "ROLL", roll)
    monkeypatch.setattr(b2, "DECISIONS", tmp_path / "decisions.json")
    buffers = {f"outputs/psychophysics/controls/knn_{a}.npz" for a in ("S1", "S2", "S3")}
    rec = {"design_hash": json.loads((check / "build" / "build.json").read_text())["design_hash"],
           "inputs_sha256": {p: b2.sha256_file(ROOT / p) for p in buffers}}
    monkeypatch.setattr(b2, "read_manifest", lambda: man)
    monkeypatch.setattr(b2, "committed_ever", lambda path: False)
    # The rollouts must be of the recorded design, each against its recorded buffer.
    monkeypatch.setattr(b2, "_post_tag", lambda: ({"tag": "rehearsal"},
                                                  {**rec, "design_hash": "0" * 64}))
    with pytest.raises(SystemExit, match="not the recorded design"):
        b2.stage_classify(_args())
    monkeypatch.setattr(b2, "_post_tag", lambda: ({"tag": "rehearsal"},
                                                  {**rec, "inputs_sha256": {}}))
    with pytest.raises(SystemExit, match="recorded buffer"):
        b2.stage_classify(_args())
    assert not (tmp_path / "decisions.json").exists()
    monkeypatch.setattr(b2, "_post_tag", lambda: ({"tag": "rehearsal"}, rec))
    b2.stage_classify(_args())
    d = json.loads((tmp_path / "decisions.json").read_text())
    labels = {a: r["label"] for a, r in d["readings"].items()}
    assert labels == {"S1": "mixed", "S2": "mixed (CM)", "S3": "fragile (CM)", "S4": "missing",
                      "S5": "not competent"}
    assert d["recipe"]["claimed"] is None and d["recipe"]["wording"]["key"] == "none"
    assert d["recipe_registered"]["classes"] == {"S1": 6, "S2": 2, "S3": 2, "S4": None, "S5": 0}
    assert d["specialists"]["S2"]["cm_decides"] is True
    assert d["specialists"]["S2"]["tier_g_rule"] and not d["specialists"]["S1"]["tier_g_rule"]
    assert d["expected"]["EB6"]["holds"] is True and d["expected"]["EB1"]["holds"] is False
    with pytest.raises(SystemExit, match="runs once"):
        b2.stage_classify(_args())
