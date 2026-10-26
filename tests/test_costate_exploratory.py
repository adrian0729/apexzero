"""scripts/costate_exploratory.py's scaffold: the items against the committed plan, the plan's
append-only rule, the row guard (decision #8), the input checks, the write-once stamped
outputs, the output roots, and the unlock after the checks. Fast: synthetic repositories,
tables and files, plus read-only checks of the committed plan, the record and the freeze."""

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
PREREG = REPO / "src" / "apexzero" / "costate" / "prereg.json"


@pytest.fixture(scope="module")
def cx():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("costate_exploratory")
    finally:
        sys.path.remove(str(SCRIPTS))


def _args(**kw):
    base = dict(check=False, run=None, out=None, network=None, limit=None, workers=4,
                threads=2, nice=0)
    base.update(kw)
    return SimpleNamespace(**base)


def _git(root: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=root, check=True,
                       capture_output=True, text=True, env=env)
    return r.stdout.strip()


def _plan_table(text: str) -> dict[str, list[str]]:
    """{item id: its row's cells} from the plan's Section 1 table."""
    sec = text.split("## 1. Items", 1)[1].split("## 2.", 1)[0]
    out = {}
    for ln in sec.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and (cells[0].startswith("CS-") or cells[0].startswith("BC-")):
            out[cells[0]] = cells
    return out


# ------------------------------------------------------------------------- the items and the plan


def test_items_match_the_plan(cx):
    text = (REPO / cx.PLAN_REL).read_text()
    table = _plan_table(text)
    ids = [it.id for it in cx.ITEMS]
    assert ids == list(table), "the items are the plan's Section 1 table, in its order"
    assert len(ids) == 16 and ids[-1] == "BC-a0"
    for it in cx.ITEMS:
        assert f"### {it.id}. " in text, f"{it.id} has no section in the plan"
        cells = table[it.id]
        assert it.trained == cells[2].startswith("yes"), it.id
        assert it.solves == (cells[3] != "–"), it.id
        if "pooled" in cells[4]:
            assert it.pooled, f"{it.id}'s cost is pooled in the plan"
        for g in it.reads:
            assert g in cx.CHECKED_GROUPS or g in cx.ITEM_GROUPS or g.startswith("gates:"), g
        assert set(it.needs) <= set(ids)
    assert set(cx.POOLED_ITEMS) <= set(cx.BY_CMD)


def test_parser_has_one_subcommand_per_item(cx):
    ap = cx.build_parser()
    for it in cx.ITEMS:
        a = ap.parse_args([it.cmd, "--check"])
        assert a.cmd == it.cmd and a.check
        assert (a.workers, a.threads, a.nice) == (cx.DEFAULT_WORKERS, cx.DEFAULT_THREADS,
                                                  cx.DEFAULT_NICE) == (4, 2, 10)
    assert ap.parse_args(["verify"]).cmd == "verify"
    assert ap.parse_args(["status"]).cmd == "status"
    with pytest.raises(SystemExit):
        ap.parse_args(["cs-16"])


def test_command_line_scan(cx):
    assert cx._command(["x", "cs-4", "--threads", "3"]) == "cs-4"
    assert cx._command(["x", "--threads", "3", "bc-a0", "--workers", "2"]) == "bc-a0"
    assert cx._command(["x", "--check"]) is None
    assert cx._flag(["x", "cs-4", "--threads", "3"], "--threads") == "3"
    assert cx._flag(["x", "cs-4", "--threads=5"], "--threads") == "5"
    assert cx._flag(["x", "cs-4"], "--threads") is None


def test_computes_registry(cx, monkeypatch):
    monkeypatch.setattr(cx, "COMPUTE", {})

    @cx.computes("CS-2")
    def fn(run):
        return run

    assert cx.COMPUTE == {"CS-2": fn}
    with pytest.raises(KeyError):
        cx.computes("CS-99")


def test_no_lockbox_reference(cx):
    src = (SCRIPTS / "costate_exploratory.py").read_text()
    assert "R_TEST" not in src and "tracks.corners" not in src


# ------------------------------------------------------------------------- the plan's rule


def _plan_repo(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    _git(root, "init", "-q")
    rel = "docs/plan.md"
    (root / rel).write_text("# Plan\n\nItems.\n\n## 7. Amendments\n\nAppended below.\n")
    _git(root, "add", rel)
    _git(root, "commit", "-q", "-m", "plan")
    return root, rel, _git(root, "rev-parse", "HEAD")


def _commit(root: Path, rel: str, text: str) -> None:
    (root / rel).write_text(text)
    _git(root, "add", rel)
    _git(root, "commit", "-q", "-m", "change")


def test_plan_append_only(cx, tmp_path):
    root, rel, sha = _plan_repo(tmp_path)
    first = (root / rel).read_text()
    assert cx.plan_problems(root, sha, rel) == []
    assert cx.plan_problems(root, "0" * 40, rel)            # another first commit
    # an amendment: refused until committed, then accepted
    good = first + "\n### Amendment 1 (2026-10-26)\n\nWhat changed, and why.\n"
    (root / rel).write_text(good)
    assert any("differs from HEAD" in p for p in cx.plan_problems(root, sha, rel))
    _git(root, "add", rel)
    _git(root, "commit", "-q", "-m", "amendment 1")
    assert cx.plan_problems(root, sha, rel) == []
    assert cx.plan_identity(root, rel)["amendments"] == 1
    assert cx.plan_identity(root, rel)["first_commit"] == sha
    # a skipped number and a heading that is not an amendment
    _commit(root, rel, good + "\n### Amendment 3 (2026-10-27)\n\nx\n## A new section\n")
    probs = cx.plan_problems(root, sha, rel)
    assert any("numbered on" in p for p in probs)
    assert any("A new section" in p for p in probs)


def test_plan_edit_in_place_refused(cx, tmp_path):
    root, rel, sha = _plan_repo(tmp_path)
    first = (root / rel).read_text()
    _commit(root, rel, first.replace("Items.", "Other items."))
    probs = cx.plan_problems(root, sha, rel)
    assert any("edited in place" in p for p in probs)
    assert any("append-only" in p for p in probs)


def test_plan_needs_its_amendments_section(cx, tmp_path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "docs/plan.md").write_text("# Plan\n")
    _git(root, "add", "docs/plan.md")
    _git(root, "commit", "-q", "-m", "plan")
    sha = _git(root, "rev-parse", "HEAD")
    assert any("Amendments" in p for p in cx.plan_problems(root, sha, "docs/plan.md"))
    assert cx.plan_problems(root, sha, "docs/missing.md") == ["the plan docs/missing.md is missing"]


def test_amendment_headings(cx):
    n, bad = cx._amendment_problems("\n### Amendment 1 (2026-10-26)\ntext\n", 0)
    assert (n, bad) == (1, [])
    _, bad = cx._amendment_problems("### Amendment 2 (2026-13-01)\n", 1)
    assert any("not a date" in b for b in bad)
    _, bad = cx._amendment_problems("### Amendment 1 (2026-10-26)\n", 1)
    assert any("numbered on" in b for b in bad)
    assert cx._amendments_in("x\n## 7. Amendments\n\n### Amendment 1 (2026-10-26)\n") == 1


# ------------------------------------------------------------------------- rows (decision #8)


def _tab():
    ph = np.array(["approach", "braking", "corner", "exit", "approach", "exit", "approach"])
    return {"phase": ph, "multi_basin": np.zeros(len(ph), bool)}


def test_guard_refuses_braking_and_corner_rows(cx):
    excl = json.loads(PREREG.read_text())["frozen"]["k2b_excluded"]
    tab = _tab()
    ok = np.array([1, 0, 0, 1, 1, 1, 0], bool)
    assert cx.guard_rows(tab, ok).sum() == 4                  # a target outside K_P (Pi, sigma)
    for i in (1, 2):                                           # braking, corner
        rows = np.zeros(7, bool)
        rows[i] = True
        with pytest.raises(cx.RowsRefused, match="decision #8"):
            cx.guard_rows(tab, rows)
        with pytest.raises(cx.RowsRefused):
            cx.guard_rows(tab, rows, "s", excl)
    with pytest.raises(ValueError):
        cx.guard_rows(tab, np.ones(3, bool))


def test_guard_keeps_k2bs_frozen_decision_rows(cx):
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    excl = json.loads(PREREG.read_text())["frozen"]["k2b_excluded"]
    assert excl["braking"] == excl["corner"] == list(TG.K_P)
    tab = _tab()
    approach, exit_ = tab["phase"] == "approach", tab["phase"] == "exit"
    assert cx.guard_rows(tab, approach | exit_, "s", excl).sum() == 5     # s: approach, exit
    assert cx.guard_rows(tab, approach, "xi", excl).sum() == 3           # xi: approach only
    with pytest.raises(cx.RowsRefused, match="K2b excludes"):
        cx.guard_rows(tab, exit_, "xi", excl)
    for c in ("n", "vx"):                                                # no decision row
        with pytest.raises(cx.RowsRefused):
            cx.guard_rows(tab, approach, c, excl)
    with pytest.raises(cx.RowsRefused, match="frozen exclusions"):
        cx.guard_rows(tab, approach, "s", None)
    # the guard admits exactly the frozen decision mask on the scored phases
    valid = ST.decision_mask(tab, None, excl)
    scored = np.isin(tab["phase"], cx.SCORED_PHASES)
    for k, c in enumerate(TG.K_P):
        rows = valid[:, k] & scored
        assert cx.guard_rows(tab, rows, c, excl, valid[:, k]).sum() == rows.sum()
    mask = valid[:, 0].copy()
    mask[0] = False
    with pytest.raises(cx.RowsRefused, match="decision mask"):
        cx.guard_rows(tab, approach, "s", excl, mask)
    mb = dict(tab, multi_basin=np.eye(7, dtype=bool)[0])
    with pytest.raises(cx.RowsRefused, match="multi-basin"):
        cx.guard_rows(mb, approach, "s", excl)


def test_mask_hash_is_the_confirmatory_formula(cx):
    valid = np.array([[1, 0, 1, 0], [0, 0, 1, 0]], bool)
    sha = hashlib.sha256(np.ascontiguousarray(valid).tobytes()).hexdigest()
    assert cx.mask_sha256(valid) == sha
    dec = {"networks": {"runs/a": {"decision_mask_sha256": sha}}}
    assert cx.mask_problems(valid, "runs/a", dec) == []
    assert cx.mask_problems(~valid, "runs/a", dec)
    assert cx.mask_problems(~valid, "runs/other", dec) == []


# ------------------------------------------------------------------------- inputs


def test_bound_hashes_merge_and_conflict(cx):
    rec = {"outputs": {"inputs/world.npz": {"sha256": "a"}, "nulls/x.npz": {"sha256": "n"}}}
    dec = {"inputs": {"nulls/x.npz": "n", "post/own/x.npz": "o"}}
    assert cx.bound_hashes(rec, dec) == {"inputs/world.npz": "a", "nulls/x.npz": "n",
                                         "post/own/x.npz": "o"}
    with pytest.raises(SystemExit, match="different sha256"):
        cx.bound_hashes(rec, {"inputs": {"nulls/x.npz": "m"}})


def test_group_inputs(cx, tmp_path):
    bound = {f"post/own/{f}.npz": "1" for f in ("a", "b")}
    bound |= {f"post/critic/{f}.npz": "2" for f in ("a", "b", "c")}
    bound |= {"post/activations/a.npz": "3", "post/activations/b.npz": "3"}
    assert cx.decision_files(bound) == ["a", "b"]
    assert [r for r, _ in cx.group_inputs("critic:decision", bound, tmp_path)] == [
        "post/critic/a.npz", "post/critic/a.k3.json", "post/critic/b.npz",
        "post/critic/b.k3.json"]
    assert len(cx.group_inputs("critic:all", bound, tmp_path)) == 3
    assert cx.group_inputs("gates:k2b", bound, tmp_path) == [("gates/k2b.json", "bound")]
    d = tmp_path / "post" / "activations"
    d.mkdir(parents=True)
    for f in ("a", "b", "c"):
        (d / f"{f}.npz").write_bytes(b"x")
    assert cx.group_inputs("activations:other", bound, tmp_path) == [
        ("post/activations/c.npz", "stamped")]
    with pytest.raises(KeyError):
        cx.group_inputs("step2", bound, tmp_path)


def test_read_checks_and_records_sha256(cx, tmp_path):
    item = cx.BY_CMD["cs-2"]
    costate = tmp_path / "outputs" / "costate"
    (costate / "inputs").mkdir(parents=True)
    f = costate / "inputs" / "world.npz"
    f.write_bytes(b"world")
    sha = hashlib.sha256(b"world").hexdigest()
    run = cx.Run(item, _args(out=str(tmp_path / "scratch")), tmp_path, {}, {},
                 {"inputs/world.npz": sha, "inputs/summary.json": "0" * 64})
    assert run.read("inputs/world.npz") == f
    assert run.inputs == {"outputs/costate/inputs/world.npz": sha}
    with pytest.raises(SystemExit, match="missing"):
        run.read("inputs/summary.json")
    (costate / "inputs" / "summary.json").write_text("{}")
    with pytest.raises(SystemExit, match="not the file"):
        run.read("inputs/summary.json")
    (costate / "gates").mkdir()
    (costate / "gates" / "k9.json").write_text("{}")
    with pytest.raises(SystemExit, match="not bound"):
        run.read("gates/k9.json")


def test_unbound_files_need_a_clean_stamp(cx, tmp_path):
    p = tmp_path / "x.npz"
    p.write_bytes(b"arrays")
    tag = {"object": "o" * 40, "commit": "c" * 40}
    rec = {"code_hash": "h" * 64, "environment": None}
    stamp = {"files": {p.name: hashlib.sha256(b"arrays").hexdigest()}, "code_hash": "h" * 64,
             "partial": [], "deviation": None, "tag": tag}
    side = tmp_path / "x.npz.provenance.json"
    side.write_text(json.dumps(stamp))
    assert cx.unbound_problems(p, rec, tag) == []
    side.write_text(json.dumps({**stamp, "partial": ["--limit"]}))
    assert any("partial" in b for b in cx.unbound_problems(p, rec, tag))
    side.write_text(json.dumps({**stamp, "code_hash": "g" * 64}))
    assert any("other code" in b for b in cx.unbound_problems(p, rec, tag))
    side.write_text(json.dumps(stamp))
    p.write_bytes(b"changed")
    assert any("changed" in b for b in cx.unbound_problems(p, rec, tag))
    side.unlink()
    assert any("no provenance stamp" in b for b in cx.unbound_problems(p, rec, tag))


def test_sealed_paths_refused(cx, tmp_path):
    sealed = tmp_path / "outputs" / "sealed"
    sealed.mkdir(parents=True)
    with pytest.raises(SystemExit, match="sealed"):
        cx.refuse_sealed(sealed / "a.pkl")
    (tmp_path / "link").symlink_to(sealed)
    with pytest.raises(SystemExit, match="sealed"):
        cx.refuse_sealed(tmp_path / "link" / "a.pkl")
    assert cx.refuse_sealed(tmp_path / "outputs" / "costate" / "x") == (
        tmp_path / "outputs" / "costate" / "x")


# ------------------------------------------------------------------------- outputs


def test_output_roots(cx, tmp_path):
    item = cx.BY_CMD["cs-2"]
    run = cx.Run(item, _args(), REPO)
    assert run.out == REPO / cx.EXP_REL / "cs-2" and run.partial == []
    assert cx.Run(item, _args(run="b1"), REPO).out == REPO / cx.EXP_REL / "cs-2" / "b1"
    with pytest.raises(SystemExit, match="plain name"):
        cx.Run(item, _args(run="../x"), REPO)
    for kw in (dict(network="G1"), dict(limit=3)):
        with pytest.raises(SystemExit, match="partial run"):
            cx.Run(item, _args(**kw), REPO)
    for bad in (REPO / cx.OUT_REL, REPO / cx.OUT_REL / "scratch", REPO / cx.EXP_REL / "x"):
        with pytest.raises(SystemExit, match="outside"):
            cx.Run(item, _args(out=str(bad)), REPO)
    with pytest.raises(SystemExit, match="sealed"):
        cx.Run(item, _args(out=str(tmp_path / "outputs" / "sealed" / "x")), REPO)
    run = cx.Run(item, _args(out=str(tmp_path), network="G1"), REPO)
    assert run.out == tmp_path / "cs-2" and run.partial == ["--out", "--network"]


def test_safe_argv(cx, tmp_path):
    got = cx.safe_argv([str(SCRIPTS / "costate_exploratory.py"), "cs-1", "--out",
                        str(REPO / "scratch" / "x"), f"--out={tmp_path / 'y'}", "--run", "b"],
                       REPO)
    assert got == [cx.SCRIPT_REL, "cs-1", "--out", "scratch/x", "--out=<outside>/y", "--run",
                   "b"]
    assert not any(a.startswith("/") for a in got)


def test_savez_bytes_is_deterministic(cx):
    import io

    arrays = {"a": np.arange(5.0), "b": np.eye(2, dtype=bool)}
    b1, b2 = cx.savez_bytes(arrays), cx.savez_bytes(dict(arrays))
    assert b1 == b2
    with np.load(io.BytesIO(b1)) as d:
        assert np.array_equal(d["a"], arrays["a"]) and np.array_equal(d["b"], arrays["b"])


def test_write_once_stamped_outputs(cx, tmp_path, monkeypatch):
    from apexzero.fields import prereg as FP

    monkeypatch.setattr(cx, "import_problems", lambda *a, **k: [])
    run = cx.Run(cx.BY_CMD["cs-2"], _args(out=str(tmp_path)), REPO, {}, {}, {})
    run.note_input("outputs/costate/inputs/world.npz", "a" * 64)
    p = run.write_json("summary", {"x": 1.5, "rows": [1, 2]})
    d = json.loads(p.read_text())
    st = d["provenance"]
    assert d["label"] == st["label"] == "exploratory" and st["item"] == "CS-2"
    assert st["plan"]["path"] == cx.PLAN_REL
    assert st["plan"]["first_commit"] == cx.PLAN_COMMIT
    assert st["head"] == _git(REPO, "rev-parse", "HEAD")
    assert st["inputs"] == {"outputs/costate/inputs/world.npz": "a" * 64}
    assert st["partial"] == ["--out"] and st["decisions_sha256"] == cx.DECISIONS_SHA256
    assert st["content_sha256"] == cx.F.content_sha256({k: v for k, v in d.items()
                                                        if k != "provenance"})
    assert not any(str(a).startswith("/") for a in st["argv"])
    with pytest.raises(SystemExit, match="write-once"):
        run.write_json("summary", {"x": 2})
    with pytest.raises(ValueError):
        run.write_json("other", {"provenance": {}})
    q = run.write_npz("arrays", {"a": np.arange(3)})
    sc = FP.read_stamp(q)
    assert sc["label"] == "exploratory"
    assert sc["files"] == {q.name: hashlib.sha256(q.read_bytes()).hexdigest()}
    with pytest.raises(SystemExit, match="write-once"):
        run.write_npz("arrays", {"a": np.arange(3)})


def _double(state, unit):
    return state * unit


def test_pool_records_its_workers(cx, tmp_path):
    run = cx.Run(cx.BY_CMD["cs-4"], _args(out=str(tmp_path), workers=1), REPO)
    assert run.pool(_double, [1, 2, 3], state=2) == [2, 4, 6]
    assert run.workers == 1
    assert cx.BY_CMD["cs-4"].pooled and not cx.BY_CMD["cs-1"].pooled


def test_status_writes_nothing(cx, tmp_path):
    rows = cx.status(tmp_path)
    assert [r["item"] for r in rows] == [it.id for it in cx.ITEMS]
    assert all(r["outputs"] == [] for r in rows)
    assert not (tmp_path / cx.EXP_REL).exists()


# ------------------------------------------------------------------------- the unlock


@pytest.fixture
def locked(monkeypatch):
    from apexzero.costate import critic as CR
    from apexzero.costate import probes as PB
    from apexzero.fields import network as NW

    mods = (CR, PB, NW)
    for m in mods:
        monkeypatch.setattr(m, "_TRAINED_OK", False)
    return lambda: [bool(m._TRAINED_OK) for m in mods]


def test_unlock_only_after_the_checks(cx, monkeypatch, tmp_path, locked):
    item = cx.BY_CMD["cs-1"]
    monkeypatch.setattr(cx, "preflight", lambda root=None: ["the plan is missing"])
    with pytest.raises(SystemExit, match="refusing to run CS-1"):
        cx.run_item(item, _args(out=str(tmp_path)))
    assert locked() == [False, False, False]
    monkeypatch.setattr(cx, "preflight", lambda root=None: [])
    monkeypatch.setattr(cx, "load_context", lambda root=None: ({}, {}, {}))
    monkeypatch.setattr(cx, "check_inputs", lambda run: [])
    monkeypatch.setattr(cx, "COMPUTE", {})
    with pytest.raises(SystemExit, match="not implemented yet"):
        cx.run_item(item, _args(out=str(tmp_path)))
    assert locked() == [False, False, False]
    seen = []
    monkeypatch.setitem(cx.COMPUTE, "CS-1", lambda run: seen.append(locked()))
    cx.run_item(item, _args(out=str(tmp_path), check=True))
    assert seen == [] and locked() == [False, False, False]
    cx.run_item(item, _args(out=str(tmp_path)))
    assert seen == [[True, True, True]]


def test_untrained_items_stay_locked(cx, monkeypatch, tmp_path, locked):
    monkeypatch.setattr(cx, "preflight", lambda root=None: [])
    monkeypatch.setattr(cx, "load_context", lambda root=None: ({}, {}, {}))
    monkeypatch.setattr(cx, "check_inputs", lambda run: [])
    seen = []
    monkeypatch.setattr(cx, "COMPUTE", {"CS-2": lambda run: seen.append(locked())})
    cx.run_item(cx.BY_CMD["cs-2"], _args(out=str(tmp_path)))
    assert seen == [[False, False, False]]


def test_import_rule(cx):
    def mod(rel):
        return SimpleNamespace(__file__=str(REPO / rel))

    ok = {"a": mod("src/apexzero/costate/analysis.py"), "b": mod("scripts/costate.py"),
          "c": mod("scripts/fields.py"), "d": mod("scripts/costate_exploratory.py"),
          "e": SimpleNamespace(__file__="/elsewhere/x.py"), "f": SimpleNamespace()}
    assert cx.import_problems(REPO, ok) == []
    bad = cx.import_problems(REPO, {"g": mod("scripts/h4r.py"), "h": mod("tests/test_x.py")})
    assert len(bad) == 2 and all("outside the frozen code" in b for b in bad)


# ------------------------------------------------------------------------- the real tree


def test_the_committed_plan_and_the_freeze(cx):
    """On this tree: the committed plan passes its rule, and, while the frozen code is the
    tag's, the freeze passes (a tree whose src/ moved on is refused by design)."""
    if cx.F._tag_object(cx.TAG, REPO) is None:
        pytest.skip("no costate-prereg-v1 tag here")
    assert cx.plan_problems(REPO) == []
    if cx._changed(REPO, cx.TAG, cx.FROZEN_PATHS + (cx.CS.DESIGN_DOC,)):
        assert cx.freeze_problems(REPO), "a tree whose frozen code moved must be refused"
        pytest.skip("the frozen code here is not the tag's")
    if not (REPO / cx.DECISIONS_REL).exists():
        pytest.skip("no decisions.json here")
    assert cx.freeze_problems(REPO) == []
    rec, dec, bound = cx.load_context(REPO)
    assert len(dec["inputs"]) == 113 and len(rec["outputs"]) == 23
    assert len(bound) == 127
    assert len(cx.decision_files(bound)) == 9


# ------------------------------------------------------------------------- CS-1: the xi tail


def test_cs1_is_registered_as_a_trained_item(cx):
    it = cx.BY_CMD["cs-1"]
    assert "CS-1" in cx.COMPUTE and it.trained and not it.pooled
    assert "checkpoints" in it.reads and "gates:k4" in it.reads
    assert set(cx.CS1_FINDINGS) == set(cx.CS1_ORDER) and len(cx.CS1_ORDER) == 9
    assert all(len(v) == len(cx.CS1_FINDINGS_FIELDS) for v in cx.CS1_FINDINGS.values())
    assert [cx.CS1_FINDINGS[k][1] for k in cx.CS1_ORDER] == [286, 247, 259, 283, 315, 229,
                                                             240, 336, 294]


def test_cs1_channel_groups(cx):
    from apexzero.env import racing
    from apexzero.fields import network as NW

    assert len(cx.EDGE) == 160 and len(set(cx.EDGE)) == 160
    assert not set(cx.EDGE) & set(NW.KAPPA_CHANNELS.tolist())
    assert set(cx.EDGE) | set(NW.KAPPA_CHANNELS.tolist()) == set(range(racing.OBS_PREVIEW))
    assert max(cx.EDGE_NEAR) < min(cx.EDGE_FAR) and len(cx.EDGE_NEAR) == len(cx.EDGE_FAR)
    assert cx.LATERAL_XI == (racing.OBS_PREVIEW + 4, racing.OBS_PREVIEW + 5)
    assert cx.TYRE_UTIL == (racing.OBS_PREVIEW + 12, racing.OBS_PREVIEW + 13)
    assert abs(6.0 * cx.KINK_VX["kink_vx25"] - 150.0) < 1e-12
    assert abs(6.0 * cx.KINK_VX["kink_vx83"] - 500.0) < 1e-12


def test_xi_tail_is_the_calibrations_leverage_cut(cx):
    from apexzero.costate import analysis as AN

    rng = np.random.default_rng(0)
    n = 300
    d = np.where(rng.random(n) < 0.5, -1, 1)
    g = rng.standard_normal(n) + 2.0 * d
    g[:12] += 40.0 * rng.choice([-1, 1], 12)
    w = rng.random(n) + 0.1
    tail = cx.xi_tail(g, w, d)
    assert np.array_equal(tail, AN.cal_weights(g, w, d, True) < w)
    # by hand: centred on the weighted median within direction, then > 3 robust SDs from the
    # centred values' weighted median
    c = g.copy()
    for s in (-1, 1):
        m = d == s
        c[m] -= AN.wmedian(g[m], w[m])
    med = AN.wmedian(c, w)
    sd = AN.wmedian(np.abs(c - med), w) / AN.MAD_NORMAL
    assert np.array_equal(tail, np.abs(c - med) > 3.0 * sd)
    assert tail[:12].all() and tail.sum() < 40


def test_flag_and_category_tables(cx):
    tail = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    flag = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    w = np.ones(12)
    err = np.array([3.0, 1.0, 0.0, 0.0] + [1.0] * 8)
    t = cx.flag_table(flag, tail, w, err)
    assert (t["tail_rows"], t["bulk_rows"]) == (2, 1)
    assert t["tail_share"] == 0.5 and t["bulk_share"] == 0.125
    assert t["enrichment_rows"] == 4.0 and t["enrichment_weight"] == 4.0
    assert t["tail_sqerr_share"] == 1.0
    none = cx.flag_table(np.zeros(12, bool), tail, w, err)
    assert np.isnan(none["enrichment_rows"]) and none["tail_sqerr_share"] == 0.0
    only = cx.flag_table(tail, tail, w, err)
    assert only["enrichment_rows"] == float("inf")
    cat = np.array([0, 0, 1, 2, 0, 1, 1, 1, 2, 2, 2, 2])
    tab = cx.category_table(cat, tail, w, err, values=np.arange(4))
    assert tab["value"] == [0, 1, 2, 3]
    assert tab["tail_rows"] == [2, 1, 1, 0] and tab["bulk_rows"] == [1, 3, 4, 0]
    assert tab["tail_sqerr_share"] == [1.0, 0.0, 0.0, 0.0]
    assert np.isnan(tab["enrichment_rows"][3])
    assert cx.half_set([0, 1, 2, 3], [0.1, 0.45, 0.05, 0.4]) == [1, 3]
    assert cx.half_set([7, 8], [0.5, 0.5]) == [7]
    assert cx.half_set([7, 8], [0.0, 0.0]) == []


def test_huber_groups_is_huber_fit_with_one_direction(cx):
    from apexzero.costate import analysis as AN

    rng = np.random.default_rng(1)
    for slope in (0.7, -0.4):
        x = rng.standard_normal(200)
        y = 1.5 + slope * x + rng.standard_t(2, 200)
        w = rng.random(200) + 0.2
        h = AN.huber_fit(x, y, w, nonneg=True)
        a, b, s, clamped, _ = cx.huber_groups(x, y, w, None)
        assert abs(a[0] - h.a) <= 1e-12 * (1 + abs(h.a)) and np.isnan(a[1])
        assert abs(b - h.b) <= 1e-12 * (1 + abs(h.b)) and abs(s - h.scale) <= 1e-12 * h.scale
        assert clamped == h.clamped == (slope < 0)
        a1, b1, *_ = cx.huber_groups(x, y, w, np.ones(200, int))     # one side, sigma > 0
        assert np.isnan(a1[0]) and a1[1] == a[0] and b1 == b


def test_huber_groups_direction_intercepts(cx):
    rng = np.random.default_rng(2)
    n = 400
    d = np.where(np.arange(n) % 2 == 0, -1, 1)
    x = rng.standard_normal(n)
    y = np.where(d > 0, -3.0, 2.0) + 0.5 * x + 0.01 * rng.standard_normal(n)
    y[:6] += 50.0                                                    # outliers in y
    a, b, _, clamped, _ = cx.huber_groups(x, y, None, d)
    assert abs(b - 0.5) < 0.01 and abs(a[0] - 2.0) < 0.01 and abs(a[1] + 3.0) < 0.01
    assert not clamped
    a, b, _, clamped, _ = cx.huber_groups(-x, y, None, d)            # anti-aligned: clamped
    assert b == 0.0 and clamped and abs(a[0] - 2.0) < 0.05
    # the calibration: per fold, the odd column by direction, clipped to the bounds
    outer = np.arange(n) % 3
    lam = np.stack([y, y], 1)
    gg = np.stack([x, x], 1)
    ok = np.ones((n, 2), bool)
    out = cx.calibrate_huber(gg, lam, outer, np.ones(n), ok[:, 0], ok, d, [False, True],
                             [[-4.0, 4.0], [-4.0, 4.0]], n_folds=3)
    assert out["pred"].shape == (n, 2) and np.isfinite(out["pred"]).all()
    assert out["pred"].max() <= 4.0 and out["pred"].min() >= -4.0
    assert np.all(out["a"][:, 0, 0] == out["a"][:, 0, 1])           # even: one intercept
    assert np.all(np.abs(out["a"][:, 1, 1] + 3.0) < 0.05)


def test_rank_map_and_calibration(cx):
    from apexzero.costate import states as ST

    g = np.array([0.0, 1.0, 2.0, 3.0])
    lam = np.array([10.0, 20.0, 30.0, 40.0])
    w = np.ones(4)
    # each training g maps to its own weighted mid-rank: lambda's quantile there is lambda
    assert np.allclose(cx.rank_map(g, lam, w, g), lam)
    got = cx.rank_map(g, lam, w, np.array([-5.0, 1.5, 9.0]))
    assert np.allclose(got, ST.wquantile(lam, w, np.array([0.0, 0.5, 1.0])))
    assert np.all(np.diff(cx.rank_map(g, lam, w, np.linspace(-1, 4, 50))) >= 0)
    # a constant g (zero and direction-only plants) has no Spearman: the direction's mean
    n = 60
    d = np.where(np.arange(n) % 2 == 0, -1, 1)
    rng = np.random.default_rng(3)
    lam2 = np.where(d > 0, 1.0, -1.0) + rng.exponential(1.0, n)
    outer = np.arange(n) % 3
    ok = np.ones((n, 1), bool)
    wv = rng.random(n) + 0.5
    for gg in (np.zeros((n, 1)), d[:, None].astype(float), -lam2[:, None]):
        out = cx.calibrate_rank(gg, lam2[:, None], outer, wv, ok[:, 0], ok, d, [[-9, 9]],
                                n_folds=3)
        assert out["fallback"].all()
        for f in range(3):
            for s in (-1, 1):
                tr, held = (outer != f) & (d == s), (outer == f) & (d == s)
                assert np.allclose(out["pred"][held, 0], wv[tr] @ lam2[tr] / wv[tr].sum())
    out = cx.calibrate_rank(lam2[:, None] + 0.0, lam2[:, None], outer, wv, ok[:, 0], ok, d,
                            [[-9, 9]], n_folds=3)
    assert not out["fallback"].any() and np.all(out["spearman"] > 0.99)


def test_group_shares_and_sizes(cx):
    rng = np.random.default_rng(4)
    n = 50
    c = rng.standard_normal((n, 5))
    g = c.sum(1)
    lam = rng.standard_normal(n)
    w = rng.random(n) + 0.1
    sh = cx.group_shares(c, g, lam, w)
    assert abs(sum(sh["share_signed"]) - 1.0) < 1e-12
    assert abs(sum(sh["share_unsigned"]) - 1.0) < 1e-12
    assert all(0 <= v <= 1 for v in sh["sign_agree"] + sh["opposite_lambda"])
    dy = rng.standard_normal((n, 222))
    J = rng.standard_normal((n, 222, 8))
    J[:, 210:, 2] = 0.0
    clipped = rng.random((n, 222)) < 0.05
    sd = rng.random(222) + 0.5
    m, cs, chan = cx.map_and_critic_sizes(dy, clipped, J, sd, 2.0)
    cos = chan / (cs * m)
    assert np.all(np.abs(cos) <= 1.0 + 1e-12)
    contrib = cx.xi_contributions(dy, clipped, J, sd, 2.0, np.zeros((n, 222, 2)),
                                  np.zeros((n, 2, 8)), np.zeros(n, bool))
    assert np.allclose(contrib[:, :4].sum(1), chan) and np.all(contrib[:, 4] == 0.0)


def test_cs1_expected_and_rule(cx):
    def net(far_tail, crit, r2_huber, plants=True):
        fl = {k: {"enrichment_rows": v} for k, v in (
            ("kink_vx83", 2.0), ("clip_any", 1.0), ("clip_xi_channels", float("nan")),
            ("clip_other_channels", 1.2), ("kappa_clipped", 1.4), ("tyre_kink_xi", float("nan")))}
        def r2(v):
            return {"r2c": v, "ci": [v - 0.1, v + 0.1]}

        return {"where": {"half_set_size": 3, "geometry38_in_half_set": True, "flags": fl,
                          "vx": {"median_tail": 60.0, "median_bulk": 55.0}},
                "critic_vs_map": {"tail": {"share_signed": [0.2, far_tail, 0.1, 0, 0]},
                                  "bulk": {"share_signed": [0.6, 0.3, 0.1, 0, 0]},
                                  "ratio": {"critic": crit, "map": 1.0}},
                "calibrations": {"frozen": {"all": {"s": r2(0.0), "xi": r2(-1.0)}},
                                 "huber": {"plants_pass": plants,
                                           "all": {"s": r2(0.01), "xi": r2(r2_huber)}},
                                 "rank": {"plants_pass": False,
                                          "all": {"s": r2(0.5), "xi": r2(0.0)}}}}

    nets = {f"N{i}": net(0.7, 3.0, -0.02) for i in range(6)}
    nets |= {f"M{i}": net(0.3, 1.0, -0.3) for i in range(3)}
    e = cx.cs1_expected(nets)
    assert e["E1.1"]["holds"] and e["E1.2"]["holds"]
    assert e["E1.3"]["far_over_half_tail"]["count"] == 6 and e["E1.3"]["holds"]
    assert e["E1.4"]["huber"]["xi_at_least_-0.05"]["count"] == 6
    assert not e["E1.4"]["huber"]["holds"] and not e["E1.4"]["rank"]["holds"]
    r = cx.cs1_rule(nets)
    assert r["huber"]["count"] == 9 and r["huber"]["fires"] and r["co_report"] == ["huber"]
    assert r["rank"]["count"] == 0 and not r["rank"]["fires"]
    nets = {k: net(0.7, 3.0, -0.95, plants=k != "N0") for k in nets}
    assert cx.cs1_rule(nets)["huber"]["count"] == 0          # a gain of 0.05 does not fire


def test_plain_makes_numpy_values_json_safe(cx):
    from apexzero.interp import io as IO

    obj = {"a": np.float64("nan"), "b": [np.float64(1.5), np.inf], "c": np.arange(2),
           "d": np.bool_(True)}
    got = json.loads(IO.json_bytes(cx.plain(obj)))
    assert got == {"a": None, "b": [1.5, "inf"], "c": [0, 1], "d": True}


_CS1_X64 = r"""
import sys
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
sys.path.insert(0, sys.argv[1])
import costate_exploratory as cx
from apexzero.agents.ppo import RunningMeanStd
from apexzero.costate import critic as CR

rng = np.random.default_rng(5)
n = 9
rms = RunningMeanStd(rng.standard_normal(222), rng.random(222) + 0.3, np.float64(1.0))
W = rng.standard_normal(222)
net = CR.linear_critic(W, 0.3, rms, sigma=3.0)
reset = np.arange(n) % 3 == 0
ob = rng.standard_normal((n, 222)) * 2.0
ob[0, 7] = 1e3                                    # a clipped channel the critic reads
oj = CR.ObsJac(ob, rng.standard_normal((n, 222, 8)), rng.standard_normal((n, 222, 2)),
               rng.standard_normal((n, 222, 2)), rng.standard_normal((n, 2, 8)),
               rng.standard_normal((n, 2, 8)), rng.standard_normal((n, 2, 2)),
               np.zeros((n, 8)), reset)
inp = CR.Inputs(np.zeros((n, 8)), np.zeros((n, 8)), np.zeros((n, 2)), reset)
pb = CR.pullback(net, None, inp, oj)
z, cl, sd = oj.z(net.obs_rms)
assert cl[0, 7]
f = CR._grad_jit(net.fn)
_, dy = CR._batched(lambda o_: f(net.params, o_), n, np.clip(z, -10, 10))
assert np.allclose(dy[:, 0], W, rtol=0, atol=1e-14)
c = cx.xi_contributions(dy[:, 0], cl, oj.J, sd, net.sigma, oj.J_acc, oj.Mx, oj.reset)
assert np.max(np.abs(c.sum(1) - pb.g[:, 0, 2]) / np.abs(pb.g[:, 0, 2])) < 1e-12
assert np.all(c[reset, 4] == 0.0) and np.all(c[~reset, 4] != 0.0)
x = np.column_stack([rng.uniform(0, 5e3, 20), rng.uniform(-3, 3, 20), rng.uniform(-0.3, 0.3, 20),
                     rng.uniform(0.5, 90, 20), rng.uniform(-3, 3, 20), rng.uniform(-1, 1, 20),
                     rng.uniform(-0.2, 0.2, 20), rng.uniform(-1, 1, 20)])
assert not cx.tyre_kink_xi(x).any()
reg = cx.tyre_region(x)
assert np.array_equal(reg[:, 2], x[:, 3] <= 1.0)
print("ok")
"""


def test_cs1_groups_sum_to_the_frozen_pullback():
    """The channel groups and the accel term sum to critic.pullback's g_xi (a planted linear
    critic, a clipped channel, reset rows), the fresh VJP route gives dV/do, and the tyre
    channels' region never moves with xi (a float64 subprocess, as the costate tests run)."""
    r = subprocess.run([sys.executable, "-c", _CS1_X64, str(SCRIPTS)], capture_output=True,
                       text=True, cwd=REPO, timeout=600, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.strip().endswith("ok")


# ------------------------------------------------------------------------- CS-2 to CS-5


def test_cs2_to_cs5_are_registered_untrained(cx):
    for cmd in ("cs-2", "cs-3", "cs-4", "cs-5"):
        it = cx.BY_CMD[cmd]
        assert it.id in cx.COMPUTE and not it.trained and not it.solves, cmd
        assert "checkpoints" in it.reads and "world" in it.reads, cmd
    assert cx.BY_CMD["cs-4"].pooled and cx.BY_CMD["cs-5"].pooled
    assert not cx.BY_CMD["cs-2"].pooled and not cx.BY_CMD["cs-3"].pooled
    assert {"step1", "step2", "critic:all"} <= set(cx.BY_CMD["cs-3"].reads)
    assert cx.SX == ("s", "xi") and cx.K_SX == (0, 2)


def _c3_int_pairs():
    from apexzero.costate import analysis as AN

    return AN.C3_INT_PAIRS


def test_cs3_measures_are_c3_ints(cx):
    assert set(cx.CS3_MEASURES) == {m for _, m, _ in _c3_int_pairs()}
    assert set(cx.STEP1_MEASURES) | set(cx.STEP2_MEASURES) == set(cx.CS3_MEASURES)


def test_untrained_cs_items_stay_locked(cx, monkeypatch, tmp_path, locked):
    monkeypatch.setattr(cx, "preflight", lambda root=None: [])
    monkeypatch.setattr(cx, "load_context", lambda root=None: ({}, {}, {}))
    monkeypatch.setattr(cx, "check_inputs", lambda run: [])
    for cmd in ("cs-2", "cs-3", "cs-4", "cs-5"):
        seen = []
        it = cx.BY_CMD[cmd]
        monkeypatch.setitem(cx.COMPUTE, it.id, lambda run, seen=seen: seen.append(locked()))
        cx.run_item(it, _args(out=str(tmp_path)))
        assert seen == [[False, False, False]], cmd


def test_direction_floor_and_fold_r2(cx):
    from apexzero.costate import analysis as AN

    n = 60
    outer = np.arange(n) % 3
    d = np.where(np.arange(n) % 2 == 0, -1, 1)
    rng = np.random.default_rng(10)
    lam = np.where(d > 0, 2.0, -1.0) + rng.standard_normal(n)
    w = rng.random(n) + 0.5
    fit = np.ones(n, bool)
    fit[:6] = False
    fl = cx.direction_floor(lam, outer, w, fit, d, n_folds=3)
    for f in range(3):
        for s in (-1, 1):
            tr = (outer != f) & fit & (d == s)
            assert np.allclose(fl[(outer == f) & (d == s)], w[tr] @ lam[tr] / w[tr].sum())
    one_side = cx.direction_floor(lam, outer, w, fit & (d > 0), d, n_folds=3)
    assert np.isnan(one_side[d < 0]).all() and np.isfinite(one_side[d > 0]).all()
    cl = np.arange(n) % 6
    pred = lam + 0.5 * rng.standard_normal(n)
    fr2 = cx.per_fold_r2(lam, pred, w, cl, 6, outer, d, True, n_folds=3)
    for f in range(3):
        m = outer == f
        want = AN.boot_r2(lam[m], [pred[m]], w[m], cl[m], 6, np.ones((1, 6)), d[m], True)[0][0]
        assert fr2[f] == want
    assert np.isnan(cx.per_fold_r2(lam, pred, w, cl, 6, np.full(n, 7), n_folds=3)).all()


def test_wcorr_and_within_direction_reliability(cx):
    from apexzero.costate import critic as CR

    rng = np.random.default_rng(11)
    n = 200
    x = rng.standard_normal(n)
    y = 0.6 * x + rng.standard_normal(n)
    w = rng.random(n) + 0.1
    assert abs(cx.wcorr(x, y, np.ones(n)) - np.corrcoef(x, y)[0, 1]) < 1e-12
    d = np.where(np.arange(n) < 80, -1, 1)
    shifted = x + 10.0 * d                                     # a direction offset in x only
    a, b = d > 0, d <= 0
    want = (w[a].sum() * cx.wcorr(x[a], y[a], w[a]) + w[b].sum() * cx.wcorr(x[b], y[b], w[b])) \
        / w.sum()
    assert abs(cx.wcorr(shifted, y, w, d) - want) < 1e-12
    assert abs(cx.wcorr(shifted, y, w)) < abs(cx.wcorr(shifted, y, w, d)) + 1.0
    assert np.isnan(cx.wcorr(np.ones(5), np.arange(5.0), np.ones(5)))
    q = x[:, None] + 0.1 * rng.standard_normal((n, 5))
    assert abs(cx.reliability_within(q, w, np.ones(n)) - CR.head_reliability(q, w)) < 1e-12
    qs = q + 10.0 * d[:, None]
    assert CR.head_reliability(qs, w) > cx.reliability_within(qs, w, d)


def test_accel_share_and_record(cx):
    g = np.array([1.0, 2.0, 0.0, -1.0])
    gh = np.array([1.0, 1.0, 0.0, 1.0])
    assert np.allclose(cx.accel_share(g, gh), [0.0, 0.5, 0.0, 2.0 / 3.0])
    util = np.array([0.1, 0.6, 1.0, 2.0])
    sens = np.array([1.0, 2.0, 3.0, 4.0])
    rec = cx.accel_record(g, gh, sens, util, np.ones(4, bool), np.full(4, 0.25))
    assert rec["rows"] == 4 and abs(rec["share_mean"] - (0.5 + 2 / 3) / 4) < 1e-12
    assert abs(rec["share_aggregate"] - 3.0 / 6.0) < 1e-12
    assert abs(rec["sens_util_spearman"] - 1.0) < 1e-12
    assert [b["rows"] for b in rec["util_bins"]] == [1, 0, 1, 0, 2]   # 1.0 and 2.0: last bin
    assert cx.accel_record(g, gh, sens, util, np.zeros(4, bool), np.ones(4)) == {"rows": 0}


def test_geometry_tables_and_blocks(cx):
    from apexzero.costate import states as ST

    geo = np.array([0, 0, 1, 1, 2])
    v = np.array([1.0, 3.0, 5.0, np.nan, 7.0])
    w = np.array([1.0, 3.0, 1.0, 1.0, 1.0])
    rows = np.array([1, 1, 1, 1, 0], bool)
    got = cx.geometry_means(v, w, geo, rows, 4)
    assert got[0] == 2.5 and got[1] == 5.0 and np.isnan(got[2:]).all()
    med = cx.geometry_medians({"a": 1.0, "b": 3.0, "c": np.nan, "d": 9.0, "z": 4.0},
                              {"a": 0, "b": 0, "c": 1, "d": 2}, 4)
    assert med[0] == 2.0 and np.isnan(med[1]) and med[2] == 9.0 and np.isnan(med[3])
    labels = np.array([[6.0, 90.0], [6.0, 180.0], [9.4, 90.0], [20.0, 45.0], [40.0, 90.0],
                       [44.7, 180.0]])
    b = cx.geometry_blocks(labels)
    k = len(ST.THETA_CLASSES) + 1
    assert np.array_equal(b // k, ST.r_tercile(labels[:, 0], labels[:, 1]))
    assert np.array_equal(b % k, ST.theta_class(labels[:, 1]))


def test_holm(cx):
    got = cx.holm([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(got, [0.04, 0.09, 0.09, 0.2])
    got = cx.holm([0.5, float("nan"), 0.001])
    assert np.allclose(got[0], 0.5) and np.isnan(got[1]) and np.allclose(got[2], 0.002)
    assert cx.holm([0.6, 0.7]) == [1.0, 1.0]


def test_c3_pairs_test(cx):
    from apexzero.costate import analysis as AN

    rng = np.random.default_rng(12)
    N, G = 8, 12
    res = rng.standard_normal((N, G))
    linked = res + 0.05 * rng.standard_normal((N, G))       # the same network x geometry pattern
    noise = rng.standard_normal((N, G))
    noise[0, :3] = np.nan                                    # missing cells
    beh = {"linked": linked, "noise": noise}
    blocks = np.arange(G) % 3
    idx = np.arange(N)
    pairs = [("a", "c", "linked", idx), ("b", "c", "noise", idx[:6])]
    out = cx.c3_pairs_test({"c": res}, beh, pairs, blocks, 199, np.random.default_rng(0))
    a, b = out["pairs"]["a"], out["pairs"]["b"]
    want = AN._corr_present(AN.double_centre(res).ravel(), AN.double_centre(linked).ravel())
    assert a["r"] == want and a["r"] > 0.9 and a["p"] == 1.0 / 200
    want_b = AN._corr_present(AN.double_centre(res[:6]).ravel(),
                              AN.double_centre(noise[:6]).ravel())
    assert b["r"] == want_b and 0.0 < b["p"] <= 1.0 and b["networks"] == 6
    assert b["cells"] == 6 * G - 3 and out["null"].shape == (199, 2)
    again = cx.c3_pairs_test({"c": res}, beh, pairs, blocks, 199, np.random.default_rng(0))
    assert np.array_equal(again["null"], out["null"])
    # the permutation keeps each network's geometries within their blocks: a behaviour table
    # constant within blocks is unchanged by it, so its null is its observed value
    const = np.stack([rng.standard_normal(3)[blocks] for _ in range(N)])
    t = cx.c3_pairs_test({"c": res}, {"k": const}, [("k", "c", "k", idx)], blocks, 20,
                         np.random.default_rng(1))
    assert np.allclose(t["null"][:, 0], t["pairs"]["k"]["r"])


def test_control_gain(cx):
    rng = np.random.default_rng(13)
    B, J = 50, 20
    base = rng.random(1 + B) * 0.1
    full = base + 0.2
    ctl = base[None] + rng.normal(0.05, 0.01, (J, 1 + B))
    pool = rng.integers(0, J, (B, J))
    g = cx.control_gain(full, ctl, base, pool)
    assert abs(g["gain"] - 0.2) < 1e-12
    assert abs(g["stat"] - (0.2 - np.median(ctl[:, 0] - base[0]))) < 1e-12
    assert g["rank"] == 1 and g["of"] == J + 1
    two = 0.2 - np.median((ctl - base[None])[:, 1:][pool, np.arange(B)[:, None]], axis=1)
    lo, hi = np.quantile(two, [0.025, 0.975])
    assert np.allclose(g["ci"], [lo, hi])
    geo = 0.2 - np.median((ctl - base[None])[:, 1:], axis=0)
    assert np.allclose(g["ci_geometry"], np.quantile(geo, [0.025, 0.975]))


def test_c1_wiring_reads_s_and_xi(cx):
    a = np.arange(6 * 4 * 2, dtype=float).reshape(6, 4, 2)
    b = np.arange(24, dtype=float).reshape(6, 4)
    c1 = SimpleNamespace(r2c=np.array([0.1, np.nan, 0.2, np.nan]), stat=0.15, ci=(-0.1, 0.3),
                         calibrated=SimpleNamespace(a=a, b=b))
    want = {"r2c": [0.1, None, 0.2, None], "stat": 0.15, "ci": [-0.1, 0.3],
            "calibration": {"a": a.tolist(), "b": b.tolist()}}
    assert all(v == 0.0 for v in cx.c1_wiring(c1, want).values())
    want["calibration"]["b"] = (b + np.array([1e-6, 9.0, 0.0, 9.0])).tolist()
    got = cx.c1_wiring(c1, want)
    assert abs(got["b"] - 1e-6) < 1e-12 and got["a"] == 0.0


def test_cs5_rows_are_approach_and_exit_interior(cx):
    ph = np.array(["approach", "braking", "corner", "exit", "approach", "exit"])
    inner = np.array([1, 1, 1, 1, 0, 1], bool)
    tab = {"phase": ph, "interior": inner, "multi_basin": np.zeros(6, bool)}
    assert cx.cs5_rows(tab).tolist() == [True, False, False, True, False, True]


def test_all_networks_order(cx):
    from apexzero.costate import analysis as AN

    rec = json.loads(PREREG.read_text())
    nets = cx.all_networks(rec)
    assert len(nets) == 34 and nets[:31] == list(AN.GENERALISTS)
    assert nets[31:] == ["S1", "S2", "S3"]
    assert [cx.network_label(AN.network_of(lb)) for lb in cx.CS1_ORDER] == list(cx.CS1_ORDER)
    assert cx.network_label("S1") == "S1"


# ------------------------------------------------------------------------- CS-10 and CS-11


def _have_keys() -> bool:
    from apexzero.costate import data_dirs

    keys, stim = data_dirs()
    return (keys / "nominal").exists() and (stim / "cells.json").exists()


def test_cs10_cs11_are_registered_trained_pooled(cx):
    for cmd in ("cs-10", "cs-11"):
        it = cx.BY_CMD[cmd]
        assert it.id in cx.COMPUTE and it.trained and it.pooled and not it.solves, cmd
        assert {"world", "critic:decision", "checkpoints"} <= set(it.reads), cmd
    assert "own" in cx.BY_CMD["cs-10"].reads
    assert cx.OWN_STEPS == 800 and cx.OWN_QUARTERS == 4
    # amendment 3's tolerances, with amendment 4's for the two cross-program checks
    assert cx.J_TOL == 1e-6 and cx.JVP_TOL == 1e-6 and cx.DV_TOL == 1e-9    # amendments 5, 6
    from apexzero.costate import critic as CR

    assert cx.J_TOL == CR.FD_TOL
    assert cx.PULLBACK_TOL == 1e-10


def test_own_states_follow_t_end(cx):
    lm = SimpleNamespace(s_c0=100.0, s_apex=150.0, s_c1=200.0)
    x = np.zeros((801, 8))
    x[:, 0] = 3.0 * np.arange(801)                  # s_k = 3 k: s_c1 + 50 at k = 84
    ks, te = cx.own_states(x, lm, 800)
    assert te == 84 and ks.tolist() == list(range(0, 85, 5))
    ks, te = cx.own_states(x, lm, 60)               # H before t_end: states below H, counted
    assert te == -1 and ks.tolist() == list(range(0, 60, 5))
    ks, _ = cx.own_states(x, lm, 80)
    assert ks.max() < 80


def test_own_interior_rule(cx):
    from apexzero.costate import states as ST

    n = 40
    clear = np.full((n, 2), 1.0)
    slip = np.full(n, 0.05)
    clear[20, 1] = 0.2                              # below CLEARANCE_MIN, no contact
    slip[30] = 5e-5                                 # slip contact
    inner, contact, ok = cx.own_interior(clear, slip)
    assert contact.tolist() == [i == 30 for i in range(n)]
    assert ok.tolist() == [i not in (20, 30) for i in range(n)]
    want = ST._window_all(ok, ST.INTERIOR_K)
    assert np.array_equal(inner, want)
    assert inner[9] and not inner[10] and not inner[20] and not inner[21] and not inner[30]
    assert inner[31] and inner[n - 1]


def test_run_inputs_reset_and_previous(cx):
    x = np.arange(30.0).reshape(10, 3) @ np.ones((3, 8)) / 3.0
    a = np.linspace(-2, 2, 20).reshape(10, 2)
    inp = cx.run_inputs(x, a, [0, 5])
    assert inp.reset.tolist() == [True, False]
    assert np.array_equal(inp.x_prev[0], x[0]) and np.array_equal(inp.x_prev[1], x[4])
    assert np.array_equal(inp.a_prev[0], [0.0, 0.0]) and np.array_equal(inp.a_prev[1], a[4])
    assert np.array_equal(inp.prev_action[1], np.clip(a[4], -1, 1))


def test_drop_reason_and_inside_field(cx):
    from apexzero.fields import perturb as PT

    assert cx.drop_reason(True, True, "x") == ""
    assert cx.drop_reason(False, False, "") == "not alive through the horizon"
    assert cx.drop_reason(False, True, "crashed before the window end") == \
        "crashed before the window end"
    assert cx.drop_reason(False, True, "") == "B <= 0"
    f = PT.Samples(np.zeros(3), np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    assert cx.inside_field(f, 1).sum() == 6.0 and cx.inside_field(f, -1).sum() == 15.0


def test_c1_own_wiring(cx):
    a = np.arange(6 * 4 * 2, dtype=float).reshape(6, 4, 2)
    b = np.arange(24, dtype=float).reshape(6, 4)
    rec = {"r2c": [0.1, None, 0.2, None], "stat": 0.15, "ci": [-0.1, 0.3],
           "calibration": {"a": a.tolist(), "b": b.tolist()}}
    assert all(v == 0.0 for v in cx.c1_own_wiring(rec, json.loads(json.dumps(rec))).values())
    other = json.loads(json.dumps(rec))
    other["ci"] = [-0.1, 0.3 + 1e-6]
    other["calibration"]["b"] = (b + np.array([0.0, 9.0, 0.0, 9.0])).tolist()
    got = cx.c1_own_wiring(rec, other)
    assert abs(got["ci"] - 1e-6) < 1e-12 and got["b"] == 0.0


def test_cs10_expected(cx):
    def net(dec, rows, ci, sp_lo):
        return {"c1_own": {"decision": dec, "n_rows": rows, "ci": ci,
                           "spearman_ci": [[0.0, 0.1], [sp_lo, 0.5]]},
                "confirmatory": {"n_rows": [162, 161], "ci": [-0.5, 0.1]}}

    nets = {f"G{i}": net("refute", [1700, 1650], [-0.1, 0.1], 0.1) for i in range(9)}
    e = cx.cs10_expected(nets)
    assert e["E10.1"]["holds"] and e["E10.2"]["holds"]
    assert abs(e["ci_width_ratio"]["G0"] - 0.2 / 0.6) < 1e-12
    nets["G0"] = net("refute", [1700, 1600], [-0.1, 0.1], -0.1)   # xi rows below 10 x 161
    nets["G1"] = net("inconclusive", [1700, 1650], [-0.1, 0.25], -0.1)
    nets["G2"] = net("refute", [1700, 1650], [-0.5, 0.15], -0.1)    # CI not half as wide
    nets["G3"] = net("refute", [1700, 1650], [-0.1, 0.1], -0.2)
    e = cx.cs10_expected(nets)
    assert not e["E10.1"]["holds"]
    assert e["E10.1"]["rows_x10"]["count"] == 8 and e["E10.1"]["refute"]["count"] == 8
    assert e["E10.1"]["ci_half_as_wide"]["count"] == 7
    assert e["E10.2"]["count"] == 5 and not e["E10.2"]["holds"]


def test_cs11_expected(cx):
    def net(r_co, r_copt, b_gl, b_hi, r_oo):
        return {"common": {"critic_own": {"r2c": r_co}, "critic_optimum": {"r2c": r_copt},
                           "own_optimum": {"r2c": r_oo, "b_gl": b_gl, "b_gl_ci": [0.0, b_hi]}}}

    nets = {f"G{i}": net(0.3, 0.1, 0.5, 0.8, 0.05) for i in range(6)}
    nets.update({f"H{i}": net(0.0, 0.1, 0.9, 1.2, 0.3) for i in range(3)})
    e = cx.cs11_expected(nets)
    assert all(e[k]["holds"] and e[k]["count"] == 6 for k in ("E11.1", "E11.2", "E11.3"))
    nets["G0"] = net(0.3, 0.1, float("nan"), 0.8, float("nan"))
    e = cx.cs11_expected(nets)
    assert e["E11.2"]["count"] == 5 and e["E11.3"]["count"] == 5 and not e["E11.2"]["holds"]


def _own_frame(cx, drop=False):
    rec = json.loads(PREREG.read_text())
    geos = rec["frozen"]["geometries"]
    keys, recs = [], []
    tab = {c: [] for c in ("key_id", "stim_id", "pair_id", "direction", "R", "theta", "W",
                           "width", "s_apex", "s_c0", "s_c1", "s_b")}
    for i, (R, th) in enumerate(geos):
        for c, v in (("key_id", f"k{i}"), ("stim_id", i), ("pair_id", i), ("direction", 1),
                     ("R", R), ("theta", th), ("W", 400), ("width", 12.0), ("s_apex", 150.0),
                     ("s_c0", 100.0), ("s_c1", 200.0), ("s_b", 120.0)):
            tab[c].append(v)
        keys.append({"row": i, "lm": (100.0, 150.0, 200.0), "s_b": 120.0})
        if drop and i == 3:
            recs.append({"kept": False, "ks": np.array([0]), "t_end": -1,
                         "x": np.zeros((1, 8))})
            continue
        ks = np.arange(0, 30, 5)
        x = np.zeros((len(ks), 8))
        x[:, 0] = 90.0 + 25.0 * np.arange(len(ks))       # every phase, approach to exit
        recs.append({"kept": True, "ks": ks, "t_end": 25, "x": x,
                     "lam": np.ones((len(ks), 8)), "lam_held": np.ones((len(ks), 8)),
                     "g": np.ones((len(ks), 8)), "g_held": np.ones((len(ks), 8)),
                     "clearance": np.ones((len(ks), 2)), "value": np.ones(len(ks)),
                     "slip_margin": np.ones(len(ks)), "interior": np.ones(len(ks), bool),
                     "contact": np.zeros(len(ks), bool), "mapped": ks > 0,
                     "s_unobservable": np.zeros(len(ks), bool)})
    tab = {k: np.asarray(v) for k, v in tab.items()}
    return rec, keys, recs, tab


def test_own_table_and_its_frame(cx):
    rec, keys, recs, tab = _own_frame(cx, drop=True)
    t, arr = cx.own_table(keys, recs, tab)
    n = len(t["k"])
    assert n == 6 * 47 + 1 and len(arr["lam"]) == n
    i3 = np.flatnonzero(t["key_id"] == "k3")
    assert len(i3) == 1 and not t["kept"][i3[0]] and not t["interior"][i3[0]]
    assert np.isnan(arr["lam"][i3]).all() and np.isnan(arr["g"][i3]).all()
    ph = t["phase"][t["key_id"] == "k0"].tolist()
    assert ph == ["approach", "approach", "braking", "corner", "corner", "exit"]
    assert cx.own_frame_problems(t, rec["frozen"]) == []
    sub = {k: v[t["key_id"] != "k5"] for k, v in t.items()}
    assert cx.own_frame_problems(sub, rec["frozen"]) == [
        "the own-run table's geometries are not the frozen ones"]


def test_own_c1_records_refuse_braking_rows(cx):
    rec, keys, recs, tab = _own_frame(cx)
    t, arr = cx.own_table(keys, recs, tab)
    from apexzero.costate import states as ST

    valid = ST.decision_mask(t, None, rec["frozen"]["k2b_excluded"])
    rows = t["kept"] & t["interior"]                # includes braking and corner rows
    with pytest.raises(cx.RowsRefused):
        cx.own_c1_records(t, arr, rows, valid, rec["frozen"], None, True, True)
    ok = rows & np.isin(t["phase"], cx.SCORED_PHASES)
    assert not valid[ok & (t["phase"] == "exit"), 2].any()    # xi: approach only (K2b)


_OWN_UNITS = r"""
import json, sys
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
sys.path.insert(0, sys.argv[1])
import costate_exploratory as cx
from apexzero.costate import critic as CR, inputs as IN, own as OW
from apexzero.fields import network as NW, outcomes as O

bank, cells = IN.design()
st = NW.cell_start(cells[344], 0, bank)
lm = bank.stimuli[st.stim_id].landmarks
kd = {"key_id": "cell344", "row": 0, "x0": np.asarray(st.x0, float), "stim_id": int(st.stim_id),
      "direction": int(st.sigma), "W": 400, "s_b": float(lm.s_c0),
      "lm": (lm.s_c0, lm.s_apex, lm.s_c1)}
rng = np.random.default_rng(3)
crit = CR.linear_critic(rng.standard_normal(222) * 1e-2, 0.1, None, sigma=2.0)
W, b = NW.linear_driver(kd["direction"])
W = W.copy()
W[:, CR.ACCEL] += 0.2 * np.asarray(OW.ACCEL_WEIGHTS, np.float32)   # it reads accel
net = NW.linear_net(W, b)
cx.own_nets = lambda agent: (net, crit)             # untrained stand-ins (no unlock needed)
rd = NW.road(bank, kd["stim_id"])
inp = CR.Inputs(kd["x0"][None], kd["x0"][None], np.zeros((1, 2)), np.array([True]))
dv = float(IN.width_derivative(crit, rd, inp, kd["direction"])[0])
state, unit = {"keys": [kd]}, {"label": "t", "agent": None, "keys": [0]}
r10 = cx.cs10_unit(state, unit)
try:                                                # a wrong stored dV_dwin stops the unit
    cx.cs11_unit(state, {**unit, "dv": {"0": dv + 1.0}, "dv_scale": 1.0})
    stopped = False
except RuntimeError:
    stopped = True
r11 = cx.cs11_unit(state, {**unit, "dv": {"0": dv}, "dv_scale": abs(dv)})
k10, k11 = r10["keys"][0], r11["keys"][0]
oc = OW.own_costate(net, rd.base, cx.own_start(kd["x0"]), 800, k10["H"], reset=True)
clear, slip = cx.run_margins(rd.base, oc.x)
side = 0 if kd["direction"] > 0 else 1
pb0 = CR.pullback(crit, None, inp, CR.obs_jacobians(rd.base, inp))
print(json.dumps({
  "kept": k10["kept"], "H": k10["H"], "restart": r10["restart"]["error"],
  "ks0": int(k10["ks"][0]), "mapped0": bool(k10["mapped"][0]), "mapped1": bool(k10["mapped"][1]),
  "held_differs": float(np.max(np.abs(k10["lam"][1:] - k10["lam_held"][1:]))),
  "g0_gap": float(np.max(np.abs(k10["g"][0] - pb0.g[0, 0]))),
  "clear_gap": float(np.max(np.abs(clear[:, side] - O.clearance_np(oc.x, rd.base, kd["direction"])))),
  "t_end": k10["t_end"], "n_states": len(k10["ks"]),
  "kept_W": k11["kept_W"], "kept_H": k11["kept_H"], "j_gap": k11["j_gap"],
  "jvp": r11["checks"]["error"], "fd": r11["fd"]["error_J_env"], "B_W": k11["B_W"],
  "pi_W": k11["pi_own_W"], "field_sum": float(np.sum(k11["field_W"], dtype=np.float64)),
  "dv_gap": abs(k11["dV_dwin"] - dv), "unit_dv_gap": k11["dv_gap"], "stopped": stopped}))
"""


@pytest.mark.skipif(not _have_keys(), reason="step 1's keys or stimuli are missing")
def test_cs10_cs11_units_on_a_planted_driver():
    """CS-10's and CS-11's units on step 2's linear driver with accel weights (untrained) and
    a planted linear critic, from cell 344's start: the run is kept, the restarts reproduce
    the on-manifold sweep (which differs from the held one), state 0 is a reset row whose
    pullback is the critic's at D, the clearances are outcomes' c(s), the width VJP equals
    the JVP and Richardson differences, its J is own_return's, and dV_dwin is
    inputs.width_derivative's (a float64 subprocess)."""
    r = subprocess.run([sys.executable, "-c", _OWN_UNITS, str(SCRIPTS)], capture_output=True,
                       text=True, cwd=REPO, timeout=900, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["kept"] and out["H"] == 800 and out["n_states"] >= 10 and out["t_end"] > 0
    assert out["restart"] <= 1e-9
    assert out["ks0"] == 0 and not out["mapped0"] and out["mapped1"]
    assert out["held_differs"] > 0.0
    assert out["g0_gap"] <= 1e-12 and out["clear_gap"] <= 1e-12
    assert out["kept_W"] and out["kept_H"] and out["B_W"] > 0
    assert out["j_gap"] <= 1e-12 and out["jvp"] <= 1e-9 and out["fd"] <= 1e-6
    assert abs(out["field_sum"] - out["pi_W"]) <= 1e-6 * max(abs(out["pi_W"]), 1e-9)
    assert out["dv_gap"] <= 1e-12 and out["unit_dv_gap"] <= 1e-12 and out["stopped"]


# ------------------------------------------------------------------------- CS-7 and CS-8


def test_cs7_cs8_are_registered_untrained_pooled(cx):
    for cmd in ("cs-7", "cs-8"):
        it = cx.BY_CMD[cmd]
        assert it.id in cx.COMPUTE and not it.trained and not it.solves and it.pooled, cmd
        assert {"world", "activations:decision", "checkpoints"} <= set(it.reads), cmd
    assert "gates:c2_null" in cx.BY_CMD["cs-8"].reads


def test_cs7_cs8_stay_locked(cx, monkeypatch, tmp_path, locked):
    monkeypatch.setattr(cx, "preflight", lambda root=None: [])
    monkeypatch.setattr(cx, "load_context", lambda root=None: ({}, {}, {}))
    monkeypatch.setattr(cx, "check_inputs", lambda run: [])
    for cmd in ("cs-7", "cs-8"):
        seen = []
        it = cx.BY_CMD[cmd]
        monkeypatch.setitem(cx.COMPUTE, it.id, lambda run, seen=seen: seen.append(locked()))
        cx.run_item(it, _args(out=str(tmp_path)))
        assert seen == [[False, False, False]], cmd


def test_cs7_chunks_cover_every_fit(cx):
    from apexzero.costate import critic as CR
    from apexzero.costate import probes as PB

    assert len(CR.RANDOM_SEEDS) == 20 and tuple(PB.POOL_SEEDS) == tuple(range(20))
    chunks = cx.cs7_chunks()
    assert len(chunks) == 4
    keys = []
    for ch in chunks:
        for j in ch:
            if j[0] == "base":
                keys.append("base")
            elif j[0] == "critic":
                keys.append(f"critic_{j[1]}")
            elif j[0] == "actor":
                keys.append("actor")
            elif j[0] == "pool":
                keys.append(f"pool{j[1]}")
            else:
                keys += [f"random{j[1]}_{L}" for L in cx.CS7_LAYERS]
    assert sorted(keys) == sorted(cx.cs7_keys()) and len(keys) == len(set(keys)) == 106
    # similar cost: about 9 s per 256-unit fit and 12 s per 320-unit fit (amendment 7)
    cost = []
    for ch in chunks:
        c = 0.0
        for j in ch:
            c += {"base": 0.5, "actor": 9, "pool": 9}.get(j[0], 0.0)
            c += (12 if j[1] == "heads" else 9) if j[0] == "critic" else 0.0
            c += 39.0 if j[0] == "random" else 0.0
        cost.append(c)
    assert max(cost) / min(cost) < 1.25
    a = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)
    assert np.array_equal(cx.heads_block(a), np.concatenate([a[0], a[1]], axis=1))


def test_control_draws_match_control_gain(cx):
    from apexzero.costate import analysis as AN

    rng = np.random.default_rng(21)
    B, J = 60, 20
    base = rng.random(1 + B) * 0.1
    full = base + 0.1 + 0.01 * rng.standard_normal(1 + B)
    ctl = base[None] + rng.normal(0.05, 0.02, (J, 1 + B))
    pool = rng.integers(0, J, (B, J))
    g = cx.control_gain(full, ctl, base, pool)
    stat, two, geo = cx.control_draws(full, ctl, base, pool)
    assert stat == g["stat"]
    assert list(AN._ci(two)) == g["ci"] and list(AN._ci(geo)) == g["ci_geometry"]
    other = cx.control_draws(full + 0.02, ctl, base, pool)
    m = cx.mean_of_draws([(stat, two, geo), other])
    assert abs(m["stat"] - (stat + other[0]) / 2) < 1e-15
    assert m["ci"] == list(AN._ci((two + other[1]) / 2))


def test_cs7_component_scores_layers_against_their_controls(cx):
    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB

    rng = np.random.default_rng(22)
    n, G = 120, 12
    cl = np.arange(n) % G
    d = np.where(np.arange(n) % 2 == 0, -1, 1)
    y = np.stack([rng.standard_normal(n) + d, rng.standard_normal(n)], 1)
    w = np.full(n, 1.0 / n)
    preds = {}
    for k in cx.cs7_keys():
        noise = 0.4 if k.startswith("critic_") else (0.9 if k == "base" else 0.8)
        preds[k] = y + noise * rng.standard_normal((n, 2))
    counts = PB.cluster_counts(G)
    pool = PB.pool_draws(20)
    rec, draws = cx.cs7_component(preds, y, w, cl, G, counts, pool, d, True)
    assert set(rec) == set(cx.CS7_TARGETS)
    keys = cx.cs7_keys()
    pt, dr = AN.boot_r2(y[:, 1], [preds[k][:, 1] for k in keys], w, cl, G, counts, d, True)
    allr = np.vstack([pt[None], dr])
    col = {k: allr[:, i] for i, k in enumerate(keys)}
    ctl = np.stack([col[f"random{s}_h2"] for s in range(20)])
    want = cx.control_gain(col["critic_h2"], ctl, col["base"], pool)
    got = rec["res"]["layers"]["h2"]
    assert got["stat"] == want["stat"] and got["ci"] == want["ci"] and got["rank"] == 1
    act = np.stack([col[f"pool{s}"] for s in range(20)])
    assert rec["res"]["layers"]["actor_h3"]["stat"] == cx.control_gain(
        col["actor"], act, col["base"], pool)["stat"]
    assert draws[("res", "h2")][0] == want["stat"]
    bad = dict(preds)
    bad.pop("pool7")
    with pytest.raises(ValueError, match="missing"):
        cx.cs7_component(bad, y, w, cl, G, counts, pool, d, True)


def _cs7_net(stat_h3=0.01, ci_lo=-0.01, heads=0.0, actor=0.0):
    def g(stat, lo=-0.01):
        return {"stat": stat, "ci": [lo, stat + 0.02], "ci_geometry": [lo, stat + 0.01]}

    layers = {"h1": g(0.0), "h2": g(0.0), "h3": g(stat_h3, ci_lo), "heads": g(heads),
              "actor_h3": g(actor)}
    return {"components": {c: {t: {"layers": layers} for t in ("lam", "res")}
                           for c in ("s", "xi")}}


def test_cs7_expected(cx):
    nets = {f"N{i}": _cs7_net() for i in range(9)}
    e = cx.cs7_expected(nets)
    assert e["E7.1"]["holds"] and e["E7.2"]["holds"] and e["E7.3"]["holds"]
    nets["N0"] = _cs7_net(stat_h3=0.08, ci_lo=0.01)
    e = cx.cs7_expected(nets)
    assert not e["E7.1"]["holds"]
    assert e["E7.1"]["above_0.05_with_ci_above_0"] == [f"N0/{c}/{t}/h3" for c in ("s", "xi")
                                                        for t in ("lam", "res")]
    nets = {f"N{i}": _cs7_net(stat_h3=0.0, actor=0.02 if i < 4 else -0.01, heads=0.03)
            for i in range(9)}
    e = cx.cs7_expected(nets)
    assert e["E7.2"]["lam:s"]["count"] == 5 and e["E7.2"]["holds"] is False
    assert e["E7.3"]["res:xi"]["count"] == 0 and not e["E7.3"]["holds"]


def test_trim_logo(cx):
    rec = {"stat": 0.1, "logo": {
        "kind": "rule", "geometry": [0, 6, 7, 9], "decision": ["refute", "inconclusive",
                                                              "refute", "refute"],
        "most_influential": {"geometry": 6, "move": 0.2},
        "shares": {"lam_s": {"weight": [0.3, 0.1, 0.4, 0.2],
                             "sse_base": [0.1, 0.7, float("nan"), 0.2]}}}}
    out = cx.trim_logo(rec)
    assert out["stat"] == 0.1 and "geometry" not in out["logo"]
    lg = out["logo"]
    assert lg["n_sets"] == 4 and lg["decisions"] == {"refute": 3, "inconclusive": 1}
    assert lg["shares"]["lam_s"]["geometry6"] == {"weight": 0.1, "sse_base": 0.7}
    assert [r["geometry"] for r in lg["shares"]["lam_s"]["top_sse_base"]] == [6, 9, 0]
    assert cx.trim_logo({"decision": "not testable"}) == {"decision": "not testable"}


def test_cs8_rows_are_approach_and_exit_switching(cx):
    ph = np.array(["approach", "braking", "corner", "exit", "approach", "exit"])
    sw = np.array([1, 1, 1, 1, 0, 1], bool)
    tab = {"phase": ph, "pedal_switching": sw, "multi_basin": np.zeros(6, bool)}
    assert cx.cs8_rows(tab).tolist() == [True, False, False, True, False, True]


def test_cs8_jobs_and_chunks(cx):
    from apexzero.costate import gates as GT

    full, bare = cx.cs8_jobs(True), cx.cs8_jobs(False)
    assert len(full) == 64 and len(bare) == 44 and set(bare) < set(full)
    assert {j[1] for j in full if j[0] == "host"} == set(GT.C2_NULL_SEEDS)
    assert all(j[-1] == "sigma" for j in full if j[0] == "host")
    ch = cx.chunked(full, 4)
    flat = [j for c in ch for j in c]
    assert len(ch) == 4 and sorted(flat, key=str) == sorted(full, key=str)
    assert max(map(len, ch)) - min(map(len, ch)) <= 1
    assert cx.chunked([1, 2], 4) == [[1], [2]]


def _same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def _synthetic_folds(n: int, G: int, rng):
    from apexzero.costate import probes as PB

    cl = rng.permutation(np.arange(n) % G)
    outer = cl % 6
    inner = np.stack([np.where(outer == f, -1, (cl // 6) % 5) for f in range(6)])
    fo = PB.Folds(outer, inner, cl, 0)
    fo.check()
    return fo


def test_sw_record_equals_c2_sw_network(cx):
    """The probes fitted once and assembled by ``sw_record`` give analysis.c2_sw_network's
    record exactly (amendment 7 item 3: the base and the pool's probes are shared by every
    host of a scaling)."""
    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB

    rng = np.random.default_rng(23)
    n, G, J = 240, 12, 4
    fo = _synthetic_folds(n, G, rng)
    z = rng.standard_normal((n, 5))
    sigma = np.abs(z[:, 0] + 0.5 * rng.standard_normal(n))
    h3 = np.tanh(z @ rng.standard_normal((5, 4)) + 0.3 * sigma[:, None])
    rnd = [np.tanh(z @ rng.standard_normal((5, 4))) for _ in range(J)]
    m_pedal = 1.0 + 0.1 * rng.standard_normal(n)
    rows = rng.random(n) < 0.85
    ws = rng.random(n) + 0.5
    ws = ws / ws[rows].sum()
    labels = np.stack([np.arange(G) + 5.0, np.full(G, 90.0)], 1)
    draws = PB.pool_draws(J)
    seeds = tuple(range(J))
    want = AN.c2_sw_network(h3, rnd, z, sigma, m_pedal, fo, ws[rows], rows, pool=draws,
                            controls=seeds, logo="rule", labels=labels)
    sub = fo.subset(rows)
    names = ("sigma_pedal",)
    base = PB.ridge_cv(z[rows], sigma[rows], sub, ws[rows], None, names)
    full = PB.ridge_cv(z[rows], sigma[rows], sub, ws[rows], h3[rows], names)
    ctl = [PB.ridge_cv(z[rows], sigma[rows], sub, ws[rows], r[rows], names) for r in rnd]
    got = cx.sw_record(base, full, ctl, ws[rows], sub, draws, labels, m_pedal[rows],
                       sigma[rows], seeds)
    assert _same(got, want)
    e = cx.null_entry(26, got)
    assert e["seed"] == 26 and e["stat"] == got["stat"]
    assert e["decision_geometry_ci"] == PB.decide(got["stat"], *got["ci_geometry"])
    assert cx.share_at_or_above([0.1, None, 0.3, float("nan"), -0.2], 0.1) == 2 / 3


def _d1(n: int, rng, thetas=(90.0, 180.0, 45.0)):
    th = rng.choice(thetas, n)
    d = rng.choice([-1.0, 1.0], n)
    return np.stack([rng.uniform(-150, 50, n), rng.uniform(20, 80, n),
                     np.log(rng.choice([8.4, 9.4, 10.5, 30.0, 60.0], n)), th,
                     rng.uniform(8, 12, n), d], 1)


def test_cross_fit_with_reproduces_states_cross_fit(cx):
    from apexzero.costate import states as ST

    rng = np.random.default_rng(24)
    n = 300
    X = _d1(n, rng)
    Y = np.stack([np.sin(X[:, 0] / 40.0) + 0.01 * X[:, 1], X[:, 5] * X[:, 2]], 1)
    fold = rng.integers(0, 6, n)
    w = rng.random(n) + 0.2
    got, info = cx.cross_fit_with(X, Y, fold, w, cx.knn_fitpred(8))
    want = ST.cross_fit(X, Y, fold, w, 8).by_fold
    assert np.array_equal(got, want)
    assert info["fits"] == 36 and info["fallback_rows"] == 0


def test_group_knn_searches_its_theta_and_direction(cx):
    rng = np.random.default_rng(25)
    n = 400
    X = _d1(n, rng)
    X[:, 0] = np.round(X[:, 0], -1)                  # many cross-group near neighbours
    grp = 10.0 * np.searchsorted([45.0, 90.0, 180.0], X[:, 3]) + X[:, 5]
    Y = grp[:, None] + 0.01 * rng.standard_normal((n, 1))
    w = np.ones(n)
    tr, te = np.arange(n) < 300, np.arange(n) >= 300
    p, info = cx.group_knn_fitpred(8)(X[tr], Y[tr], w[tr], np.zeros(300, int), X[te])
    assert info["fallback_rows"] == 0
    assert np.max(np.abs(p[:, 0] - grp[te])) < 0.05     # only its own group's values
    p0, _ = cx.knn_fitpred(8)(X[tr], Y[tr], w[tr], None, X[te])
    assert np.max(np.abs(p0[:, 0] - grp[te])) > 1.0      # f_D's own search mixes groups
    # a theta with no training row falls back to its turn direction, and is counted
    Xte = X[te].copy()
    Xte[:5, 3] = 60.0
    p2, info2 = cx.group_knn_fitpred(8)(X[tr], Y[tr], w[tr], None, Xte)
    assert info2["fallback_rows"] == 5 and np.isfinite(p2).all()
    # log R x 3 changes which rows are neighbours
    p3, _ = cx.group_knn_fitpred(8, 3.0)(X[tr], X[tr][:, [2]], w[tr], None, X[te])
    p1, _ = cx.group_knn_fitpred(8, 1.0)(X[tr], X[tr][:, [2]], w[tr], None, X[te])
    err3 = np.mean(np.abs(p3[:, 0] - X[te][:, 2]))
    err1 = np.mean(np.abs(p1[:, 0] - X[te][:, 2]))
    assert err3 < err1


def test_spline_ridge_pieces(cx):
    from apexzero.costate import probes as PB

    u = np.linspace(-1.0, 2.0, 50)
    B = cx.bspline_basis(u, 0.0, 1.0, [0.2, 0.4, 0.6, 0.8, 0.9])
    assert B.shape == (50, 9) and np.allclose(B.sum(1), 1.0)
    assert np.array_equal(B[u <= 0], np.tile(B[np.argmin(np.abs(u))], ((u <= 0).sum(), 1)))
    rng = np.random.default_rng(26)
    X = _d1(80, rng)
    spec = cx.spline_spec(X, np.ones(80))
    A = cx.spline_design(X, spec)
    assert A.shape == (80, 486)
    assert spec[0][0] == X[:, 0].min() and spec[1][1] == X[:, 1].max()
    # the ridge path at a vanishing penalty is weighted least squares
    Xr = rng.standard_normal((60, 4))
    yr = Xr @ np.array([1.0, -2.0, 0.5, 0.0]) + 3.0 + 0.1 * rng.standard_normal(60)
    wr = rng.random(60) + 0.5
    P = cx.wridge_path(Xr, yr[:, None], wr, Xr[:5], [1e-12, 1e4])
    D = np.hstack([np.ones((60, 1)), Xr])
    beta = np.linalg.lstsq(D * np.sqrt(wr)[:, None], yr * np.sqrt(wr), rcond=None)[0]
    assert np.allclose(P[0, :, 0], D[:5] @ beta, atol=1e-8)
    assert np.allclose(P[1, :, 0], np.average(yr, weights=wr), atol=1e-2)
    assert cx.choose_penalty(np.array([[3.0, 2.0], [1.0, 2.0], [1.0, 5.0]])).tolist() == [2, 1]
    assert len(PB.GRID) == 17


def test_spline_ridge_fits_a_smooth_decoy(cx):
    rng = np.random.default_rng(27)
    n = 700
    X = _d1(n, rng)
    # in the model class: B(ds, vx) times each of (1, log R, theta class, w, sigma)
    f = (np.sin(X[:, 0] / 50.0) * (1 + 0.02 * X[:, 1]) * (1 + 0.3 * X[:, 5])
         + 0.4 * X[:, 2] * np.cos(X[:, 1] / 30.0))
    Y = np.stack([f, -f], 1) + 0.05 * rng.standard_normal((n, 2))
    ftr = rng.integers(0, 5, n)
    tr, te = np.arange(n) < 600, np.arange(n) >= 600
    p, info = cx.spline_fitpred(X[tr], Y[tr], np.ones(600), ftr[tr], X[te])
    r2 = 1 - np.sum((Y[te] - p) ** 2, 0) / np.sum((Y[te] - Y[te].mean(0)) ** 2, 0)
    assert (r2 > 0.9).all() and len(info["penalties"]) == 2


def test_geometry_shares_and_the_rule(cx):
    v = np.array([1.0, -1.0, 3.0, -3.0])
    w = np.ones(4)
    geo = np.array([6, 1, 6, 1])
    assert cx.geometry_share(v, w, geo) == 10.0 / 20.0
    d = np.array([1, 1, -1, -1])
    # within direction: means 0 and 0, so the same here; shifting one direction changes
    # the pooled share but not the within-direction one
    vs = v + np.where(d > 0, 10.0, 0.0)
    assert cx.geometry_share(vs, w, geo, d, True) == 0.5
    assert cx.geometry_share(vs, w, geo) != 0.5
    good = {("lam", "s"): 0.11780, ("lam", "xi"): 0.0900, ("res", "s"): 0.7045,
            ("res", "xi"): 0.6466}
    assert cx.design_g6_problems(good) == []
    bad = {**good, ("lam", "s"): 0.1134}
    assert len(cx.design_g6_problems(bad)) == 1
    assert not cx.passes_g6(good)
    assert cx.passes_g6({**good, ("res", "s"): 0.2356, ("res", "xi"): 0.18})
    assert not cx.passes_g6({**good, ("res", "s"): 0.2356, ("res", "xi"): 0.1801})


def _sw_rec(stat, lo, rank):
    return {"stat": stat, "ci": [lo, stat + 0.02], "pool": {"rank": rank}}


def test_cs8_expected(cx):
    labels = ("G1", "G1s1", "G1s2", "G2", "G2s1", "G2s2", "G3", "G3s1", "G3s2")
    sw = {lb: {"sigma": _sw_rec(0.03, 0.01, 1), "res": _sw_rec(0.01, -0.01, 3)}
          for lb in labels}
    null = {"summary": {"mean": -0.002, "sd": 0.015, "max": 0.025}}
    e = cx.cs8_expected(sw, null, {}, None)
    assert e["E8a.1"]["holds"] and e["E8b.1"]["holds"]
    assert e["E8b.1"]["count_at_most_half"] == 9 and e["E8b.1"]["count_rank1_on_res"] == 0
    sw["G2"]["sigma"]["stat"] = 0.02
    e = cx.cs8_expected(sw, null, {}, None)
    assert not e["E8a.1"]["holds"] and not e["E8a.1"]["confirmatory_rank1_above_every_host"]["G2"]
    e = cx.cs8_expected(sw, {"summary": {"mean": 0.006, "sd": 0.015, "max": 0.0}}, {}, None)
    assert not e["E8a.1"]["mean_part"] and e["E8a.1"]["sd_part"]
    sw["G3"]["sigma"]["stat"] = -0.01
    sw["G3s1"]["res"] = _sw_rec(0.03, 0.01, 1)
    e = cx.cs8_expected(sw, null, {}, None)
    assert e["E8b.1"]["sigma_stat_not_positive"] == ["G3"]
    assert e["E8b.1"]["count_at_most_half"] == 7 and e["E8b.1"]["rank1_with_ci_above_0"]["G3s1"]

    def c2net(share, width, stat):
        sh = {f"lam_{c}": {"geometry6": {"sse_base": share}} for c in ("s", "xi")}
        return {"chosen": {"stat": stat, "ci": [0.0, width], "logo": {"shares": sh}},
                "frozen": {"stat": 0.0, "ci": [0.0, 0.3]}}

    c2 = {"networks": {lb: c2net(0.2, 0.15, 0.01) for lb in labels}}
    e = cx.cs8_expected(sw, null, {}, c2)
    assert e["E8c.1"]["holds"]
    c2["networks"]["G1"] = c2net(0.2, 0.15, 0.06)
    assert not cx.cs8_expected(sw, null, {}, c2)["E8c.1"]["holds"]
    c2 = {"networks": {lb: c2net(0.5 if i < 4 else 0.2, 0.25, 0.0) for i, lb in
                       enumerate(labels)}}
    e = cx.cs8_expected(sw, null, {}, c2)["E8c.1"]
    assert e["count_geometry6"] == 5 and e["count_narrower"] == 0 and not e["holds"]
    e = cx.cs8_expected(sw, null, {}, {"reason": "no candidate"})
    assert e["E8c.1"]["holds"] is None
