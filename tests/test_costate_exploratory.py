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
