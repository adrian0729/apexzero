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
