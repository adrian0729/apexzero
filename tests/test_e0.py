"""scripts/e0_runs.py and scripts/e0_gate.py (E0, docs/plan/e0.md): the run list against the
plan's Section 1 and the run-list contract, the final-checkpoint constant against the real
checkpoint schedule, the R-dev sha256 table against docs/findings/gw4.md, the config check
against round_manifest.py's own, the gate's count and median on planted eval_rdev.json files
(exactly 10 passes, 9 fails, a failed or evaluation-less seed counts 0), the run-once rules, the
manifest on a planted round, and the tag gate on fakes and on real scratch repositories. The
code-path constant is checked against what train_dr.py actually imports.

No network is touched (the scratch repositories' origin is a bare directory), nothing is trained,
no R_TEST circuit is read and nothing under outputs/sealed/ is opened: every run is planted in
tmp_path."""

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import e0_gate as E
import e0_runs as ER
import gw3_runs
import round_manifest as RM
import vg1_runs

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
TRAIN_DR = (REPO / "scripts" / "train_dr.py").read_text()


# ------------------------------------------------------------------------- the run list


def test_runs_are_the_plans():
    runs = ER.runs()
    assert [r["name"] for r in runs] == [f"e0-s{s}" for s in range(50, 55)]
    assert [r["seed"] for r in runs] == [50, 51, 52, 53, 54]
    for r in runs:
        s = r["seed"]
        assert r["cmd"] == ["scripts/train_dr.py", "--steps", "1e9", "--seed", str(s), "--name",
                            f"e0-s{s}", "--real-widths"]
        assert r["config"] == f"runs/e0-s{s}/config.json"
    assert ER.runs("main") == runs
    with pytest.raises(ValueError, match="one phase"):
        ER.runs("pilot")


def test_runs_are_disjoint_from_gw3_and_vg1():
    mine = ER.runs()
    others = (gw3_runs.runs() + vg1_runs.pilot_runs() + vg1_runs.pilot2_runs()
              + [{"name": f"vg1-{arm}-s{s}", "seed": s}
                 for arm in vg1_runs.ARMS for s in vg1_runs.MAIN_SEEDS])
    assert {r["seed"] for r in mine}.isdisjoint({r["seed"] for r in others})
    assert {r["name"] for r in mine}.isdisjoint({r["name"] for r in others})


def test_runs_pass_the_run_list_contract_and_the_queue_deals_one_lane_each(monkeypatch):
    monkeypatch.chdir(REPO)
    assert RM.run_list_problems(ER.runs()) == []
    sched = RM.schedule(ER.runs(), 1, 5)
    assert [(lane, gpu, pos) for lane, gpu, pos, _ in sched] == [(k, 0, 0) for k in range(5)]


def test_the_script_prints_one_json_line_per_run():
    for extra in ([], ["--phase", "main"]):
        out = subprocess.run([sys.executable, str(SCRIPTS / "e0_runs.py"), *extra], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
        assert [json.loads(ln) for ln in out.splitlines()] == ER.runs()


def test_the_cmds_parse_with_train_dr_and_record_the_plans_flags():
    parser = RM.trainer_parser(TRAIN_DR)
    defaults = E.trainer_defaults(TRAIN_DR)
    for r in ER.runs():
        args = vars(parser.parse_args(r["cmd"][1:]))
        want = E.expected_args(r["seed"], defaults)
        assert {k: args[k] for k in want if args[k] is not RM.COMPUTED} == {
            k: v for k, v in want.items() if args[k] is not RM.COMPUTED}


# ------------------------------------------------------------------------- the constants


def test_final_timesteps_is_the_last_checkpoint_of_the_1e9_schedule():
    from apexzero.agents.checkpoints import checkpoint_schedule

    d = E.trainer_defaults(TRAIN_DR)
    batch = d["num_envs"] * d["num_steps"]
    assert batch == 131_072 and int(float(ER.STEPS)) == 1_000_000_000
    sched = checkpoint_schedule(1e9, batch)
    assert max(sched.values()) == E.FINAL_TIMESTEPS == 999_948_288
    assert E.FINAL_TIMESTEPS == (1_000_000_000 // batch) * batch == 7629 * batch
    assert len(sched) == 16 and 9_900_000 < min(sched.values()) <= 10_000_000
    assert max(checkpoint_schedule(200e6, batch).values()) == 199_884_800      # VG-1's, same rule


def test_the_computed_defaults_are_ppos():
    from apexzero.agents.ppo import ValueGradient

    assert (E.COMPUTED_DEFAULTS["vg_lambda"], E.COMPUTED_DEFAULTS["vg_winsor"]) == (
        ValueGradient().lam, ValueGradient().winsor)


def gw4_table() -> dict[str, str]:
    """The sha256 table of docs/findings/gw4.md Section 3, parsed."""
    text = (REPO / "docs" / "findings" / "gw4.md").read_text()
    sec = re.search(r"^## 3\. .*?(?=^## )", text, re.DOTALL | re.MULTILINE)[0]
    return dict(re.findall(r"^\| (\w+) \| `([0-9a-f]{64})` \|$", sec, re.MULTILINE))


def test_the_rdev_hashes_are_gw4s_table():
    table = gw4_table()
    assert len(table) == 12 and E.RDEV_SHA256 == table
    from apexzero.tracks.corners import R_DEV

    assert list(E.RDEV_SHA256) == R_DEV == E.r_dev()


def test_the_hashed_files_are_the_rdev_files_on_this_machine():
    """Where the (untracked) centrelines are here, they are the table's; read by name only."""
    have = {n: REPO / E.TRACKS_DIR / f"{n}.csv" for n in E.RDEV_SHA256}
    if not all(p.is_file() for p in have.values()):
        pytest.skip("the R-dev centrelines are not on this machine")
    assert E.rdev_problems() == []


def test_the_code_paths_cover_train_drs_import_closure():
    """train_dr.py --help imports what the trainer loads, and its end-of-run evaluation unpickles
    outputs/ocp/<circuit>.pkl, which loads apexzero.ocp (BrandsHatch's, by name, where it is
    here); every src/apexzero file that came in must be under E0_CODE_PATHS, so that a new import
    cannot slip outside the gate."""
    code = (
        "import json, os, pickle, runpy, sys\n"
        "sys.path.insert(0, 'scripts')\n"
        "sys.argv = ['train_dr.py', '--help']\n"
        "try:\n"
        "    runpy.run_path('scripts/train_dr.py', run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "ocp = 'outputs/ocp/BrandsHatch.pkl'\n"
        "done = os.path.isfile(ocp)\n"
        "if done:\n"
        "    with open(ocp, 'rb') as fh:\n"
        "        pickle.load(fh)\n"
        "print(json.dumps([done, sorted({m.__file__ for m in sys.modules.values()\n"
        "                                if getattr(m, '__file__', None)})]))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                         check=True).stdout
    unpickled, names = json.loads(out.splitlines()[-1])
    files = [Path(f).resolve() for f in names]
    src = (REPO / "src" / "apexzero").resolve()
    loaded = [f.relative_to(REPO.resolve()).as_posix() for f in files
              if f.is_relative_to(src) and f.suffix == ".py"]
    assert "src/apexzero/agents/ppo.py" in loaded
    if unpickled:
        assert "src/apexzero/ocp/min_time.py" in loaded
    outside = [f for f in loaded
               if not any(f == p or f.startswith(p + "/") for p in E.E0_CODE_PATHS)]
    assert outside == []
    for p in E.E0_CODE_PATHS:
        assert (REPO / p).exists(), p
    assert {"src/apexzero/ocp", "uv.lock", "pyproject.toml"} <= set(E.E0_CODE_PATHS)
    assert set(E.UPSTREAM_CODE_PATHS) == set(E.E0_CODE_PATHS) - set(E.E0_SCRIPTS)


def test_the_pinned_upstream_commit_is_the_tags():
    out = subprocess.run(["git", "rev-parse", "generator-v1^{commit}", "sim-v1^{commit}"],
                         cwd=REPO, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        pytest.skip("the generator-v1 and sim-v1 tags are not here")
    assert out.stdout.split() == [E.GENERATOR_COMMIT] * 2


def table_after(text: str, heading: str) -> dict[str, str]:
    """The markdown table of ``| <Circuit> | `<sha256>` |`` rows right under ``heading``."""
    rows = {}
    for line in text.split(heading, 1)[1].splitlines():
        m = re.fullmatch(r"\| (\w+) \| `([0-9a-f]{64})` \|", line)
        if m:
            rows[m[1]] = m[2]
        elif rows and not line.startswith("|"):
            break
    return rows


def test_the_ocp_hashes_are_e0s_table():
    table = table_after((REPO / "docs" / "plan" / "e0.md").read_text(), "**OCP reference files.**")
    assert len(table) == 12 and E.OCP_SHA256 == table
    assert list(E.OCP_SHA256) == list(E.RDEV_SHA256) == E.r_dev()


def test_the_hashed_ocp_files_are_the_rdev_ones_on_this_machine():
    have = {n: REPO / E.OCP_DIR / f"{n}.pkl" for n in E.OCP_SHA256}
    if not all(p.is_file() for p in have.values()):
        pytest.skip("the R-dev OCP files are not on this machine")
    assert {n: E.sha256_file(p) for n, p in have.items()} == E.OCP_SHA256


def test_the_box_command_names_every_file_and_globs_only_the_runs(capsys):
    cmd = E.box_command()
    assert capsys.readouterr().out == cmd + "\n"
    names = list(E.RDEV_SHA256)
    assert cmd == " ".join(
        ["sha256sum", "runs/e0-*/ckpt/*.pkl", "runs/e0-*/eval_rdev.json",
         *[f"data/tracks/tumftm/{n}.csv" for n in names], *[f"outputs/ocp/{n}.pkl" for n in names]])
    assert [w for w in cmd.split() if "*" in w] == ["runs/e0-*/ckpt/*.pkl",
                                                    "runs/e0-*/eval_rdev.json"]
    assert len(cmd.split()) == 1 + 2 + 24 and "Austin" not in cmd


def test_e0_logic_imports_nothing_from_another_studys_script():
    mods = set()
    for f in ("e0_gate.py", "e0_runs.py"):
        for node in ast.walk(ast.parse((SCRIPTS / f).read_text())):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
    assert not {m for m in mods if m.startswith(("vg1", "gw3", "h4r"))}
    assert {"e0_runs", "round_manifest"} <= mods


def test_the_copied_helpers_are_vg1_analysis_s_bodies():
    def defs(path):
        return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef | ast.ClassDef)}

    mine, theirs = defs(SCRIPTS / "e0_gate.py"), defs(SCRIPTS / "vg1_analysis.py")
    for name in ("CheckFailed", "sha256_file", "clean", "num", "box_sums", "read_list"):
        assert mine[name] == theirs[name], name


# ------------------------------------------------------------------------- the config check


def good_config(seed: int = 50) -> dict:
    d = E.trainer_defaults(TRAIN_DR)
    args = json.loads(json.dumps(E.expected_args(seed, d)))        # as config.json holds it
    return {"ppo": {}, "args": args, "env": {}}


def test_config_check_accepts_a_correct_config_and_agrees_with_the_queues():
    d = E.trainer_defaults(TRAIN_DR)
    for run in ER.runs():
        cfg = good_config(run["seed"])
        assert E.config_problems(cfg, run["seed"], d) == []
        problems, _ = RM.compare_config(run, cfg, RM.trainer_parser(TRAIN_DR))
        assert problems == []
    assert cfg["args"]["steps"] == 1e9 and cfg["args"]["real_widths"] is True


@pytest.mark.parametrize("key, value", [
    ("real_widths", False), ("real_widths_from", "BrandsHatch,Sochi"), ("stat_widths", True),
    ("tilt_fraction", 0.5), ("tilt_alpha", 2.0), ("vg_coef", 0.1), ("vg_lambda", 0.9),
    ("steps", 200e6), ("seed", 51), ("name", "e0-s51"), ("num_envs", 1024), ("bank", 2048),
    ("network", "simba"), ("fixed_car", True), ("reexpress", True), ("skip_eval_rdev", True),
    ("stat_widths_from", "Sochi"), ("ensemble_heads", 3), ("real_widths", 1),
])
def test_config_check_rejects_each_wrong_flag(key, value):
    d = E.trainer_defaults(TRAIN_DR)
    cfg = good_config()
    cfg["args"][key] = value
    got = E.config_problems(cfg, 50, d)
    assert len(got) == 1 and f"--{key}" in got[0]


def test_config_check_rejects_a_missing_extra_or_absent_args():
    d = E.trainer_defaults(TRAIN_DR)
    cfg = good_config()
    del cfg["args"]["vg_coef"]
    assert E.config_problems(cfg, 50, d) == ["config does not record --vg_coef"]
    cfg = good_config()
    cfg["args"]["new_flag"] = 1
    assert "which train_dr.py does not define" in E.config_problems(cfg, 50, d)[0]
    assert E.config_problems({}, 50, d) == ['config has no "args" object']


# ------------------------------------------------------------------------- the gate's count


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A planted repo root: R-dev's source, runs/ and outputs/e0/."""
    (tmp_path / "src" / "apexzero" / "tracks").mkdir(parents=True)
    shutil.copy(REPO / "src" / "apexzero" / "tracks" / "corners.py",
                tmp_path / "src" / "apexzero" / "tracks" / "corners.py")
    monkeypatch.setattr(E, "ROOT", tmp_path)
    return tmp_path


def plant_eval(root: Path, name: str, completed: int, gap: float = 0.05, at: dict | None = None):
    """eval_rdev.json as train_dr.py writes it: completion 1.0 on the first ``completed``
    circuits, 0.0 (lap time NaN) on the rest; ``at`` sets chosen circuits' completion."""
    nan = float("nan")
    rows = {}
    for k, n in enumerate(E.r_dev()):
        c = (at or {}).get(n, 1.0 if k < completed else 0.0)
        rows[n] = ({"completion": c, "median_lap_s": 90.0, "ocp_lap_s": 90.0 / (1 + gap),
                    "gap": gap} if c is not None and c > 0.5 else
                   {"completion": c, "median_lap_s": nan, "ocp_lap_s": 85.0, "gap": nan})
    d = root / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval_rdev.json").write_text(json.dumps(rows, indent=1))


def planted_manifest(root: Path, counts, failed=(), no_eval=(), raw=None, **kw) -> dict:
    """E0's manifest as the gate reads it, over planted eval_rdev.json files (``counts`` per
    seed; the seeds in ``failed`` are listed as failed twice, those in ``no_eval`` have no file)."""
    runs = []
    for run, c in zip(ER.runs(), counts, strict=True):
        if run["seed"] not in no_eval:
            plant_eval(root, run["name"], c, **kw)
            if raw and run["seed"] in raw:             # a file of the test's own making
                (root / "runs" / run["name"] / "eval_rdev.json").write_text(raw[run["seed"]])
        else:
            (root / "runs" / run["name"] / "eval_rdev.json").unlink(missing_ok=True)
        p = root / "runs" / run["name"] / "eval_rdev.json"
        runs.append({"name": run["name"], "seed": run["seed"],
                     "failure": {"status": "failed", "reason": "box died twice"} if run["seed"] in failed else None,
                     "config_sha256": None, "metrics_sha256": None,
                     "eval_rdev_sha256": E.sha256_file(p) if p.is_file() else None})
    return {"record": "e0-manifest", "tag": E.TAG, "commit": COMMIT, "runs": runs,
            "provenance": {"inputs": {}}}


def counts_of(root, counts, **kw):
    res = E.compute_gate(planted_manifest(root, counts, **kw), E.r_dev())
    return res


def test_median_of_exactly_10_passes_and_9_fails(root):
    res = counts_of(root, [12, 11, 10, 8, 2])
    assert res["counts"] == [12, 11, 10, 8, 2] and res["median"] == 10 and res["pass"] is True
    assert res["seeds_at_least_pass_median"] == 3 and res["label"] == "in-sample for widths"
    res = E.compute_gate(planted_manifest(root, [12, 11, 9, 8, 2]), E.r_dev())
    assert res["median"] == 9 and res["pass"] is False and res["seeds_at_least_pass_median"] == 2


def test_completion_must_exceed_one_half(root):
    n = E.r_dev()
    res = counts_of(root, [10, 10, 10, 10, 10], at={n[0]: 0.5})        # 0.5 is not completed
    assert res["counts"] == [9] * 5 and res["pass"] is False
    res = counts_of(root, [9, 9, 9, 9, 9], at={n[11]: 0.51})
    assert res["counts"] == [10] * 5 and res["pass"] is True


def test_a_failed_seed_counts_zero_whatever_its_file_says(root):
    m = planted_manifest(root, [12, 12, 12, 9, 9], failed=(50, 51))
    assert (root / "runs" / "e0-s50" / "eval_rdev.json").is_file()
    res = E.compute_gate(m, E.r_dev())
    assert res["counts"] == [0, 0, 12, 9, 9] and res["median"] == 9 and res["pass"] is False
    s = res["per_seed"]["e0-s50"]
    assert s["counted_as_zero"].startswith("failed twice") and s["completion_by_circuit"] is None
    assert E.compute_gate(planted_manifest(root, [12, 12, 12, 9, 9]), E.r_dev())["pass"] is True


def test_a_seed_with_no_eval_rdev_counts_zero(root):
    res = counts_of(root, [12, 12, 12, 9, 9], no_eval=(52,))
    assert res["counts"] == [12, 12, 0, 9, 9] and res["median"] == 9 and res["pass"] is False
    assert res["per_seed"]["e0-s52"]["counted_as_zero"] == "no eval_rdev.json"
    res = counts_of(root, [12, 12, 12, 9, 9], no_eval=(50, 51))
    assert res["counts"] == [0, 0, 12, 9, 9] and res["median"] == 9


def test_per_circuit_completion_and_gap_on_completed_circuits_only(root):
    res = counts_of(root, [3, 12, 12, 12, 12], gap=0.07)
    s = res["per_seed"]["e0-s50"]
    n = E.r_dev()
    assert s["count"] == 3 and list(s["completion_by_circuit"]) == n
    assert [c for c, v in s["completion_by_circuit"].items() if v > 0.5] == n[:3]
    assert list(s["gap_by_circuit_completed"]) == n[:3]
    assert all(v == pytest.approx(0.07) for v in s["gap_by_circuit_completed"].values())


@pytest.mark.parametrize("value", [None, float("nan")])
def test_null_and_nan_completion_are_not_completed(root, value):
    n = E.r_dev()
    res = counts_of(root, [10, 10, 10, 10, 10], at={n[0]: value, n[9]: value})
    assert res["counts"] == [8] * 5 and res["pass"] is False
    s = res["per_seed"]["e0-s50"]
    assert all(s["completion_by_circuit"][c] != s["completion_by_circuit"][c] for c in (n[0], n[9]))
    assert list(s["gap_by_circuit_completed"]) == n[1:9]
    res = counts_of(root, [10, 10, 10, 10, 10], at={n[11]: value})       # the 11th is not a 10th
    assert res["counts"] == [10] * 5 and res["pass"] is True


def good_rows(completed: int = 12) -> dict:
    return {n: {"completion": 1.0 if k < completed else 0.0, "median_lap_s": 90.0,
                "ocp_lap_s": 85.0, "gap": 0.05} for k, n in enumerate(E.r_dev())}


def malformed(how: str) -> str:
    rows = good_rows()
    n = E.r_dev()
    if how == "reversed":
        rows = dict(reversed(list(rows.items())))
    elif how == "missing circuit":
        rows.pop(n[-1])
    elif how == "extra circuit":
        rows["Austin"] = rows[n[0]]
    elif how == "row not an object":
        rows[n[3]] = "completed"
    elif how == "string completion":
        rows[n[3]]["completion"] = "1.0"
    elif how == "bool completion":
        rows[n[3]]["completion"] = True
    elif how == "no completion":
        del rows[n[3]]["completion"]
    elif how == "string gap":
        rows[n[3]]["gap"] = "0.05"
    elif how == "list":
        return json.dumps(list(rows.values()))
    return json.dumps(rows)


MALFORMED = ["reversed", "missing circuit", "extra circuit", "row not an object",
             "string completion", "bool completion", "no completion", "string gap", "list"]


def test_eval_problems_accepts_train_drs_files_with_null_and_nan():
    names = E.r_dev()
    assert E.eval_problems(good_rows(), names) == []
    rows = good_rows()
    rows[names[0]]["completion"], rows[names[1]]["completion"] = None, float("nan")
    rows[names[1]]["gap"] = float("nan")
    del rows[names[2]]["gap"]
    assert E.eval_problems(rows, names) == []


@pytest.mark.parametrize("how", MALFORMED)
def test_eval_problems_names_each_malformed_file(how, root):
    assert E.eval_problems(json.loads(malformed(how)), E.r_dev())


@pytest.mark.parametrize("how", MALFORMED)
def test_a_malformed_eval_is_refused_before_the_lock(root, monkeypatch, how):
    """The gate runs once: a file that cannot be read is refused in the input check, with the
    lock untaken and nothing written."""
    monkeypatch.setattr(E, "git_run", FakeGit())
    monkeypatch.setattr(E, "tag_gate", lambda *a, **k: GATE)
    plant_manifest_file(root, planted_manifest(root, [12] * 5, raw={52: malformed(how)}))
    with pytest.raises(E.CheckFailed, match=r"e0-s52: runs/e0-s52/eval_rdev.json: "):
        E.cmd_gate(argparse.Namespace())
    assert not (root / E.LOCK).exists() and not (root / E.GATE).exists()


def test_null_and_nan_completion_pass_the_input_check_and_the_gate_runs(root, monkeypatch):
    n = E.r_dev()
    rows = good_rows(10)
    rows[n[0]]["completion"], rows[n[1]]["completion"] = None, float("nan")
    monkeypatch.setattr(E, "git_run", FakeGit())
    monkeypatch.setattr(E, "tag_gate", lambda *a, **k: GATE)
    plant_manifest_file(root, planted_manifest(root, [12] * 5, raw={50: json.dumps(rows)}))
    E.cmd_gate(argparse.Namespace())
    rec = json.loads((root / E.GATE).read_text())
    assert rec["counts"] == [8, 12, 12, 12, 12] and rec["pass"] is True
    assert rec["per_seed"]["e0-s50"]["completion_by_circuit"][n[0]] is None


def test_eval_of_other_circuits_is_refused(root):
    m = planted_manifest(root, [12] * 5)
    p = root / "runs" / "e0-s50" / "eval_rdev.json"
    rows = json.loads(p.read_text())
    rows.pop(E.r_dev()[-1])
    p.write_text(json.dumps(rows))
    with pytest.raises(E.CheckFailed, match="not R-dev's"):
        E.compute_gate(m, E.r_dev())


# ------------------------------------------------------------------------- run once


OBJ, COMMIT, HEAD, GEN = "a" * 40, "b" * 40, "c" * 40, E.GENERATOR_COMMIT
GATE = {"tag": E.TAG, "tag_object": OBJ, "commit": COMMIT, "head": HEAD,
        "upstream": {"generator-v1": GEN, "sim-v1": GEN}}
BOTH = {"generator-v1": GEN, "sim-v1": GEN}


class FakeGit:
    """``git_run`` with an annotated e0-train-v1 on origin and locally, HEAD descending from it,
    generator-v1 and sim-v1 on GENERATOR_COMMIT (``up_origin`` and ``up_local``: where each tag
    peels, there and here) below it, E0's code unchanged and everything committed, unless a
    field says otherwise."""

    def __init__(self, **kw):
        self.remote = (f"{OBJ}\trefs/tags/e0-train-v1\n{COMMIT}\trefs/tags/e0-train-v1^{{}}\n")
        self.remote_rc, self.local, self.kind, self.commit = 0, OBJ, "tag", COMMIT
        self.ancestor, self.gen_ok = True, True
        self.up_origin, self.up_local = dict(BOTH), dict(BOTH)
        self.diff_clean, self.untracked, self.log, self.absent = True, "", "", False
        self.gen_tree = ""                  # generator-v1's tree listing (the tag's is empty)
        self.__dict__.update(kw)

    def __call__(self, *args):
        def cp(rc, out=""):
            return subprocess.CompletedProcess(["git", *args], rc, out, "")

        cmd = args[0]
        if cmd == "ls-remote":
            if "refs/tags/e0-train-v1" not in args:           # the upstream tags
                text = "".join(f"{'7' * 40}\trefs/tags/{t}\n{c}\trefs/tags/{t}^{{}}\n"
                               for t, c in self.up_origin.items())
                return cp(self.remote_rc, text if self.remote_rc == 0 else "")
            return cp(self.remote_rc, self.remote if self.remote_rc == 0 else "")
        if cmd == "rev-parse":
            if args[-1] == "refs/tags/e0-train-v1":
                return cp(0 if self.local else 1, self.local or "")
            if args[-1] in ("refs/tags/generator-v1^{commit}", "refs/tags/sim-v1^{commit}"):
                tag = args[-1].split("/")[-1].split("^")[0]
                return cp(0 if self.up_local[tag] else 1, self.up_local[tag])
            if args[-1].endswith("^{commit}"):
                return cp(0, self.commit)
            return cp(0, HEAD)
        if cmd == "cat-file":
            return cp(1 if self.absent else 0) if args[1] == "-e" else cp(0, self.kind)
        if cmd == "merge-base":
            return cp(0 if (self.ancestor if args[-1] == "HEAD" else self.gen_ok) else 1)
        if cmd == "diff":
            return cp(0 if self.diff_clean else 1)
        if cmd == "ls-files":
            return cp(0, self.untracked if "--others" in args else "")
        if cmd == "ls-tree":
            return cp(0, self.gen_tree if args[2] == GEN else "")
        if cmd == "hash-object":
            return cp(0, "")
        if cmd == "log":
            return cp(0, self.log)
        if cmd in ("status", "show"):
            return cp(0, "")
        raise AssertionError(f"unexpected git call {args}")


def plant_manifest_file(root: Path, m: dict) -> None:
    (root / "outputs" / "e0").mkdir(parents=True, exist_ok=True)
    (root / E.MANIFEST).write_text(json.dumps(m))


def test_gate_end_to_end_writes_once_and_takes_the_lock(root, monkeypatch, capsys):
    monkeypatch.setattr(E, "git_run", FakeGit())
    monkeypatch.setattr(E, "tag_gate", lambda *a, **k: GATE)
    plant_manifest_file(root, planted_manifest(root, [12, 11, 10, 8, 2]))
    E.cmd_gate(argparse.Namespace())
    rec = json.loads((root / E.GATE).read_text())
    assert rec["record"] == "e0-gate" and rec["median"] == 10 and rec["pass"] is True
    assert rec["label"] == "in-sample for widths" and rec["counts"] == [12, 11, 10, 8, 2]
    assert rec["manifest_sha256"] == E.sha256_file(root / E.MANIFEST)
    assert rec["provenance"]["inputs"][E.MANIFEST] == rec["manifest_sha256"]
    assert "runs/e0-s50/eval_rdev.json" in rec["provenance"]["inputs"]
    assert (root / E.LOCK).is_file() and "PASS" in capsys.readouterr().out
    with pytest.raises(E.CheckFailed, match="runs once"):           # now both files exist
        E.cmd_gate(argparse.Namespace())


@pytest.mark.parametrize("on_disk, log", [((E.GATE,), ""), ((E.LOCK,), ""), ((), "f" * 40)])
def test_gate_refuses_a_second_run(root, monkeypatch, on_disk, log):
    monkeypatch.setattr(E, "git_run", FakeGit(log=log))
    monkeypatch.setattr(E, "tag_gate", lambda *a, **k: GATE)
    plant_manifest_file(root, planted_manifest(root, [12] * 5))
    for p in on_disk:
        (root / p).write_text("x")
    with pytest.raises(E.CheckFailed, match="runs once"):
        E.cmd_gate(argparse.Namespace())
    assert not (root / E.GATE).exists() or on_disk == (E.GATE,)
    assert (root / E.LOCK).exists() == (E.LOCK in on_disk)


def test_lock_is_taken_exclusively(root, monkeypatch):
    """A lock that appears between the check and the open stops the gate (O_EXCL)."""
    monkeypatch.setattr(E, "git_run", FakeGit())
    monkeypatch.setattr(E, "tag_gate", lambda *a, **k: GATE)
    plant_manifest_file(root, planted_manifest(root, [12] * 5))
    real = E.load_manifest

    def racing(*a, **k):
        m = real(*a, **k)
        (root / E.LOCK).write_text("other")
        return m

    monkeypatch.setattr(E, "load_manifest", racing)
    with pytest.raises(FileExistsError):
        E.cmd_gate(argparse.Namespace())
    assert not (root / E.GATE).exists()


@pytest.mark.parametrize("plant, git, msg", [
    (lambda r: (r / "runs" / "e0-s52" / "eval_rdev.json").write_text("{}"), {},
     "is not the manifest's"),
    (lambda r: None, {"diff_clean": False}, "must be committed"),
    (lambda r: (r / E.MANIFEST).write_text(json.dumps({"record": "other"})), {}, "not E0's"),
])
def test_gate_checks_the_manifest_and_inputs_before_taking_the_lock(root, monkeypatch, plant,
                                                                     git, msg):
    monkeypatch.setattr(E, "git_run", FakeGit(**git))
    monkeypatch.setattr(E, "tag_gate", lambda *a, **k: GATE)
    plant_manifest_file(root, planted_manifest(root, [12] * 5))
    plant(root)
    with pytest.raises(E.CheckFailed, match=msg):
        E.cmd_gate(argparse.Namespace())
    assert not (root / E.LOCK).exists() and not (root / E.GATE).exists()


def test_gate_refuses_a_manifest_of_another_commit_or_run_list(root, monkeypatch):
    monkeypatch.setattr(E, "git_run", FakeGit())
    monkeypatch.setattr(E, "tag_gate", lambda *a, **k: GATE)
    m = planted_manifest(root, [12] * 5)
    plant_manifest_file(root, {**m, "commit": "9" * 40})
    with pytest.raises(E.CheckFailed, match="not e0-train-v1"):
        E.cmd_gate(argparse.Namespace())
    plant_manifest_file(root, {**m, "runs": m["runs"][:4]})
    (root / E.MANIFEST).write_text(json.dumps({**m, "runs": m["runs"][:4]}))
    with pytest.raises(E.CheckFailed, match="not the run list's"):
        E.cmd_gate(argparse.Namespace())
    assert not (root / E.LOCK).exists()


def test_gate_refuses_a_changed_failures_file(root, monkeypatch):
    monkeypatch.setattr(E, "git_run", FakeGit())
    monkeypatch.setattr(E, "tag_gate", lambda *a, **k: GATE)
    plant_manifest_file(root, planted_manifest(root, [12] * 5))       # no failures at the time
    (root / E.FAILURES).write_text(json.dumps(
        {"record": "e0-failures", "runs": {"e0-s50": {"status": "failed", "reason": "late"}}}))
    with pytest.raises(E.CheckFailed, match="not as it was"):
        E.cmd_gate(argparse.Namespace())
    assert not (root / E.LOCK).exists()


def test_gate_refuses_without_the_tag_before_anything(root, monkeypatch):
    monkeypatch.setattr(E, "git_run", FakeGit(remote=""))
    monkeypatch.setattr(E, "rdev_problems", list)
    for cmd in (E.cmd_gate, E.cmd_manifest):
        with pytest.raises(PermissionError, match="not on origin"):
            cmd(argparse.Namespace(tag=E.TAG, box_sums=[]))
    assert not (root / "outputs").exists()


# ------------------------------------------------------------------------- the manifest


CHECK = {"returncode": 0, "tolerated": [], "report_sha256": "0", "report": "OK"}


def plant_round(root: Path) -> tuple[list, Path]:
    runs = ER.runs()
    recs = []
    for k, r in enumerate(runs):
        d = root / "runs" / r["name"]
        (d / "ckpt").mkdir(parents=True)
        for ts in (10_027_008, E.FINAL_TIMESTEPS):
            (d / "ckpt" / f"{ts:012d}.pkl").write_bytes(f"{r['name']}{ts}".encode())
        (d / "config.json").write_text(json.dumps(good_config(r["seed"])))
        (d / "metrics.csv").write_text("explained_var\n0.93\n")
        plant_eval(root, r["name"], 12)
        (root / "runs" / f"{r['name']}.log").write_text("log\n")
        recs.append({"name": r["name"], "seed": r["seed"], "status": "done", "lane": k, "gpu": 0,
                     "start": "2026-10-29T10:00:00Z", "end": "2026-10-29T12:30:00Z"})
    (root / "runs" / "e0-manifest-0of1.json").write_text(json.dumps(
        {"runs": recs, "gpu_model": "NVIDIA RTX A6000", "gpu_count": 1,
         "box_type": "massedcompute_A6000_plus"}))
    lines = [f"{E.sha256_file(p)}  {p.relative_to(root).as_posix()}"
             for p in sorted((root / "runs").glob("e0-*/ckpt/*.pkl"))
             + sorted((root / "runs").glob("e0-*/eval_rdev.json"))]
    lines += [f"{h}  {E.TRACKS_DIR}/{n}.csv" for n, h in E.RDEV_SHA256.items()]
    lines += [f"{h}  {E.OCP_DIR}/{n}.pkl" for n, h in E.OCP_SHA256.items()]
    sums = root / "sums.txt"
    sums.write_text("\n".join(lines) + "\n")
    return runs, sums


def build(root, runs, sums, failures=None):
    return E.build_manifest(E.box_sums([str(sums)]), runs, failures or {}, CHECK,
                            E.trainer_defaults(TRAIN_DR), GATE)


def test_manifest_records_every_file_and_the_box(root):
    runs, sums = plant_round(root)
    m = build(root, runs, sums)
    assert m["record"] == "e0-manifest" and m["commit"] == COMMIT and len(m["runs"]) == 5
    e = m["runs"][0]
    assert e["final"]["timesteps"] == E.FINAL_TIMESTEPS and len(e["checkpoints"]) == 2
    assert e["final"]["sha256"] == E.sha256_file(root / e["final"]["path"])
    assert e["eval_rdev_sha256"] == E.sha256_file(root / "runs" / "e0-s50" / "eval_rdev.json")
    assert e["config_sha256"] and e["metrics_sha256"] and list(e["logs_sha256"]) == [
        "runs/e0-s50.log"]
    assert e["box"]["gpu_model"] == "NVIDIA RTX A6000" and e["box"]["lane"] == 0
    assert set(m["round_files"]) == {"runs/e0-manifest-0of1.json"}
    assert m["box_data_sha256"] == {
        **{f"data/tracks/tumftm/{n}.csv": h for n, h in E.RDEV_SHA256.items()},
        **{f"outputs/ocp/{n}.pkl": h for n, h in E.OCP_SHA256.items()}}


def edit_sums(sums: Path, path: str, new: str | None) -> None:
    """Drop ``path``'s line from the box's sums, or give it the sha256 ``new``."""
    out = []
    for ln in sums.read_text().splitlines():
        if ln.endswith("  " + path):
            if new is None:
                continue
            ln = f"{new}  {path}"
        out.append(ln)
    sums.write_text("\n".join(out) + "\n")


@pytest.mark.parametrize("path", ["data/tracks/tumftm/Sochi.csv", "outputs/ocp/Zandvoort.pkl"])
def test_manifest_needs_the_boxs_data_files_and_their_pinned_hashes(root, path):
    runs, sums = plant_round(root)
    edit_sums(sums, path, "0" * 64)
    with pytest.raises(E.CheckFailed, match=f"{path} on the box is not the pinned file"):
        build(root, runs, sums)
    edit_sums(sums, path, None)
    with pytest.raises(E.CheckFailed, match=f"{path} has no box sha256"):
        build(root, runs, sums)


def test_manifest_reads_no_data_file_off_the_box_sums(root):
    """The box's data files are checked from its sums alone: none of them need exist here."""
    runs, sums = plant_round(root)
    assert not (root / "data").exists() and not (root / "outputs" / "ocp").exists()
    assert build(root, runs, sums)["box_data_sha256"]


@pytest.mark.parametrize("plant, msg", [
    (lambda r: (r / "runs" / "e0-s51" / "ckpt" / f"{E.FINAL_TIMESTEPS:012d}.pkl").write_bytes(
        b"x"), "is not the box's file"),
    (lambda r: (r / "runs" / "e0-s51" / "ckpt" / f"{E.FINAL_TIMESTEPS:012d}.pkl").unlink(),
     "no final checkpoint"),
    (lambda r: (r / "runs" / "e0-s51" / "ckpt" / "000199884800.pkl").write_bytes(b"y"),
     "no box sha256"),
    (lambda r: (r / "runs" / "e0-s52" / "eval_rdev.json").write_text("{}"), "is not the box's"),
    (lambda r: (r / "runs" / "e0-s53" / "metrics.csv").unlink(), "no metrics.csv"),
    (lambda r: (r / "runs" / "e0-s53" / "config.json").unlink(), "no config.json"),
    (lambda r: (r / "runs" / "e0-s54" / "config.json").write_text(json.dumps(
        {**good_config(54), "args": {**good_config(54)["args"], "real_widths": False}})),
     "--real_widths is False"),
    (lambda r: (r / "runs" / "e0-s54" / "config.json").write_text(json.dumps(
        {**good_config(54), "args": {**good_config(54)["args"], "seed": 53}})),
     "--seed is 53"),
])
def test_manifest_refuses_mismatches(root, plant, msg):
    runs, sums = plant_round(root)
    plant(root)
    with pytest.raises(E.CheckFailed, match=msg):
        build(root, runs, sums)


def test_manifest_tolerates_only_a_listed_failure(root):
    runs, sums = plant_round(root)
    box = root / "runs" / "e0-manifest-0of1.json"
    m = json.loads(box.read_text())
    m["runs"][2]["status"] = "failed"
    box.write_text(json.dumps(m))
    with pytest.raises(E.CheckFailed, match="e0-s52: status 'failed' and not in failures.json"):
        build(root, runs, sums)
    got = build(root, runs, sums, {"e0-s52": {"status": "failed", "reason": "OOM twice"}})
    assert got["runs"][2]["failure"] == {"status": "failed", "reason": "OOM twice"}
    with pytest.raises(E.CheckFailed, match="outside the round"):
        build(root, runs, sums, {"e0-s52": {"status": "failed", "reason": "x"}, "e0-s99": {"status": "failed", "reason": "x"}})
    with pytest.raises(E.CheckFailed, match="listed in failures.json but its status"):
        build(root, runs, sums, {"e0-s52": {"status": "failed", "reason": "x"}, "e0-s50": {"status": "failed", "reason": "x"}})


def test_round_check_tolerates_only_listed_runs(monkeypatch):
    report = "round e0\nFAILED: 2 problem(s)\n  e0-s52: status 'failed' in x\n  e0-s52: runs/e0-s52 is not here\n"
    monkeypatch.setattr(E, "check_report", lambda: (1, report))
    got = E.round_check({"e0-s52": {"status": "failed", "reason": "x"}})
    assert len(got["tolerated"]) == 2 and got["returncode"] == 1
    with pytest.raises(E.CheckFailed, match="e0-s52: status"):
        E.round_check({})
    monkeypatch.setattr(E, "check_report", lambda: (1, "tag 'x' not found"))
    with pytest.raises(E.CheckFailed, match="did not run"):
        E.round_check({"e0-s52": {"status": "failed", "reason": "x"}})
    monkeypatch.setattr(E, "check_report", lambda: (0, "OK"))
    assert E.round_check({})["tolerated"] == []


def test_read_failures_needs_a_committed_record_with_reasons(root, monkeypatch):
    assert E.read_failures({}) == {}
    (root / E.FAILURES).parent.mkdir(parents=True)
    (root / E.FAILURES).write_text(json.dumps(
        {"record": "e0-failures", "runs": {"e0-s50": {"status": "failed", "reason": "twice"}}}))
    monkeypatch.setattr(E, "git_run", FakeGit(diff_clean=False))
    with pytest.raises(E.CheckFailed, match="must be committed"):
        E.read_failures({})
    monkeypatch.setattr(E, "git_run", FakeGit())
    inputs: dict = {}
    assert list(E.read_failures(inputs)) == ["e0-s50"] and E.FAILURES in inputs
    (root / E.FAILURES).write_text(json.dumps({"record": "e0-failures",
                                               "runs": {"e0-s50": {}}}))
    with pytest.raises(E.CheckFailed, match="need status 'failed'"):
        E.read_failures({})
    for entry in ({"reason": "no status"}, {"status": "missing", "reason": "box died"},
                  {"status": "retried", "reason": "x"}, {"status": "failed"},
                  {"status": "failed", "reason": ""}, None):
        (root / E.FAILURES).write_text(json.dumps({"record": "e0-failures",
                                                   "runs": {"e0-s50": entry}}))
        with pytest.raises(E.CheckFailed, match="need status 'failed'"):
            E.read_failures({})


def test_manifest_is_write_once(root, monkeypatch):
    monkeypatch.setattr(E, "git_run", FakeGit())
    monkeypatch.setattr(E, "tag_gate", lambda *a, **k: GATE)
    (root / "outputs" / "e0").mkdir(parents=True)
    (root / E.MANIFEST).write_text("{}")
    with pytest.raises(E.CheckFailed, match="write-once"):
        E.cmd_manifest(argparse.Namespace(tag=E.TAG, box_sums=[]))
    with pytest.raises(E.CheckFailed, match="E0's tag is"):
        E.cmd_manifest(argparse.Namespace(tag="other", box_sums=[]))
    with pytest.raises(E.CheckFailed, match="write-once"):
        E.write_new(root / E.MANIFEST, {})


# ------------------------------------------------------------------------- the tag gate


def rdev_ok(monkeypatch):
    monkeypatch.setattr(E, "rdev_problems", list)


def test_tag_gate_passes_on_the_full_tag(monkeypatch):
    monkeypatch.setattr(E, "git_run", FakeGit())
    rdev_ok(monkeypatch)
    assert E.tag_gate() == {"tag": "e0-train-v1", "tag_object": OBJ, "commit": COMMIT,
                            "head": HEAD, "upstream": BOTH}


@pytest.mark.parametrize("change", [
    {"remote": ""}, {"remote_rc": 128}, {"remote": f"{COMMIT}\trefs/tags/e0-train-v1\n"},
    {"local": ""}, {"local": "f" * 40}, {"kind": "commit"}, {"commit": "9" * 40},
    {"ancestor": False}, {"absent": True}, {"diff_clean": False},
    {"untracked": "src/apexzero/agents/new.py\n"},
    {"gen_ok": False}, {"up_origin": {"generator-v1": GEN, "sim-v1": "9" * 40}},
    {"up_origin": {"generator-v1": "9" * 40, "sim-v1": GEN}},
    {"up_local": {"generator-v1": GEN, "sim-v1": "9" * 40}},
    {"up_local": {"generator-v1": "", "sim-v1": GEN}},
    {"gen_tree": "100644 blob " + "1" * 40 + "\tscripts/train_dr.py"},
])
def test_tag_gate_refuses(monkeypatch, change):
    monkeypatch.setattr(E, "git_run", FakeGit(**change))
    rdev_ok(monkeypatch)
    with pytest.raises(PermissionError):
        E.tag_gate()


def test_tag_gate_refuses_changed_rdev_centrelines(monkeypatch):
    monkeypatch.setattr(E, "git_run", FakeGit())
    monkeypatch.setattr(E, "rdev_problems", lambda: ["data/tracks/tumftm/Sochi.csv is missing"])
    with pytest.raises(PermissionError, match="Sochi.csv is missing"):
        E.tag_gate()


class ScratchRepo:
    """A bare ``origin`` and a clone of it, with real git and no user config, holding a stand-in
    for E0's code: generator-v1 and sim-v1 on the first commit, e0-train-v1 (annotated, pushed)
    on the second, which adds the two E0 scripts (and, with ``change_upstream``, edits ppo.py).
    ``E.ROOT`` points at the clone, whose R-dev centrelines are planted files."""

    CODE = ("src/apexzero/__init__.py", "src/apexzero/agents/ppo.py",
            "src/apexzero/env/racing.py", "src/apexzero/eval/laps.py",
            "src/apexzero/oracle/o.py", "src/apexzero/vehicle/v.py", "scripts/train_dr.py",
            "scripts/_bootstrap.py", "scripts/gpu_round.sh", "scripts/round_manifest.py",
            "src/apexzero/ocp/min_time.py", "uv.lock", "pyproject.toml")

    def __init__(self, tmp_path: Path, monkeypatch, change_upstream: bool = False):
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
        corners = self.clone / "src" / "apexzero" / "tracks" / "corners.py"
        corners.parent.mkdir(parents=True)
        shutil.copy(REPO / "src" / "apexzero" / "tracks" / "corners.py", corners)
        for rel in self.CODE:
            self.write(rel, f"# {rel}\nALPHA = 0.05\n")
        self.git("add", "-A")
        self.git("commit", "-qm", "generator-v1")
        self.git("tag", "-a", "-m", "gen", "generator-v1")
        self.git("tag", "-a", "-m", "sim", "sim-v1")
        self.generator = self.git("rev-parse", "HEAD")
        monkeypatch.setattr(E, "GENERATOR_COMMIT", self.generator)
        for rel in E.E0_SCRIPTS:
            self.write(rel, f"# {rel}\n")
        if change_upstream:
            self.write("src/apexzero/agents/ppo.py", "# changed\n")
        self.git("add", "-A")
        self.git("commit", "-qm", "e0")
        self.git("tag", "-a", "-m", "E0", "e0-train-v1")
        self.git("push", "-q", "origin", "HEAD:refs/heads/main", "refs/tags/e0-train-v1",
                 "refs/tags/generator-v1", "refs/tags/sim-v1")
        monkeypatch.setattr(E, "ROOT", self.clone)
        # the (untracked) centrelines: planted bytes, and the table of their hashes
        table = {}
        for n in E.r_dev():
            self.write(f"{E.TRACKS_DIR}/{n}.csv", f"x,y\n{n}\n")
            table[n] = E.sha256_file(self.clone / E.TRACKS_DIR / f"{n}.csv")
        monkeypatch.setattr(E, "RDEV_SHA256", table)

    def write(self, rel: str, text: str) -> None:
        p = self.clone / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def git(self, *args) -> str:
        return subprocess.run(["git", *args], cwd=self.clone, check=True, capture_output=True,
                              text=True).stdout.strip()


@needs_git
def test_tag_gate_passes_on_a_real_tag(tmp_path, monkeypatch):
    repo = ScratchRepo(tmp_path, monkeypatch)
    got = E.tag_gate()
    head = repo.git("rev-parse", "HEAD")
    assert got["commit"] == head and got["tag_object"] != got["commit"]
    assert got["upstream"] == dict.fromkeys(("generator-v1", "sim-v1"), repo.generator)
    assert repo.generator != head
    assert E.code_byte_problems(head) == []


@needs_git
def test_tag_gate_ignores_files_outside_e0s_code(tmp_path, monkeypatch):
    """Other studies' files under src/apexzero (maze, costate) are not E0's code: an untracked
    one, and a commit that changes one after the tag, leave the gate open."""
    repo = ScratchRepo(tmp_path, monkeypatch)
    repo.write("src/apexzero/maze/sfl.py", "x = 1\n")
    E.tag_gate()
    repo.git("add", "src/apexzero/maze/sfl.py")
    repo.git("commit", "-qm", "maze")
    repo.write("src/apexzero/maze/sfl.py", "x = 2\n")
    assert E.tag_gate()["head"] == repo.git("rev-parse", "HEAD")


@needs_git
@pytest.mark.parametrize("rel", ["src/apexzero/agents/ppo.py", "src/apexzero/__init__.py",
                                 "scripts/train_dr.py", "scripts/gpu_round.sh",
                                 "scripts/round_manifest.py", "scripts/e0_gate.py",
                                 "scripts/e0_runs.py"])
def test_tag_gate_refuses_a_dirty_code_path(tmp_path, monkeypatch, rel):
    repo = ScratchRepo(tmp_path, monkeypatch)
    repo.write(rel, "# edited\n")
    with pytest.raises(PermissionError, match="differs from e0-train-v1"):
        E.tag_gate()


@needs_git
def test_tag_gate_refuses_a_committed_change_to_a_code_path(tmp_path, monkeypatch):
    repo = ScratchRepo(tmp_path, monkeypatch)
    repo.write("src/apexzero/env/racing.py", "# later\n")
    repo.git("commit", "-qam", "later")
    with pytest.raises(PermissionError, match="differs from e0-train-v1"):
        E.tag_gate()


@needs_git
def test_tag_gate_refuses_an_untracked_file_in_a_code_path(tmp_path, monkeypatch):
    repo = ScratchRepo(tmp_path, monkeypatch)
    repo.write("src/apexzero/tracks/extra.py", "x = 1\n")
    with pytest.raises(PermissionError, match="untracked files under E0's code"):
        E.tag_gate()


@needs_git
@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_tag_gate_refuses_an_edit_git_diff_does_not_show(tmp_path, monkeypatch, flag):
    """git diff and git status skip a file marked assume-unchanged or skip-worktree; the gate
    compares the bytes on disk with the tag's."""
    repo = ScratchRepo(tmp_path, monkeypatch)
    rel = "scripts/train_dr.py"
    repo.git("update-index", flag, rel)
    repo.write(rel, "# edited\n")
    assert repo.git("diff", "--name-only", "e0-train-v1") == ""
    assert repo.git("status", "--porcelain", "--untracked-files=no") == ""
    with pytest.raises(PermissionError, match="train_dr.py on disk is not"):
        E.tag_gate()


@needs_git
def test_code_byte_problems_names_a_missing_and_an_added_file(tmp_path, monkeypatch):
    repo = ScratchRepo(tmp_path, monkeypatch)
    tag = repo.git("rev-parse", "HEAD")
    repo.git("update-index", "--skip-worktree", "scripts/train_dr.py")
    (repo.clone / "scripts" / "train_dr.py").unlink()
    repo.write("src/apexzero/agents/new.py", "x = 1\n")
    repo.git("add", "src/apexzero/agents/new.py")
    assert E.code_byte_problems(tag) == [
        f"src/apexzero/agents/new.py is tracked here but not in {tag[:12]}",
        "scripts/train_dr.py is missing on disk"]


@needs_git
def test_tag_gate_refuses_code_that_moved_since_generator_v1(tmp_path, monkeypatch):
    ScratchRepo(tmp_path, monkeypatch, change_upstream=True)
    with pytest.raises(PermissionError, match="differs between generator-v1 and e0-train-v1"):
        E.tag_gate()
    assert E.tree_problems("generator-v1", "e0-train-v1", ("scripts/e0_gate.py",)) == [
        "scripts/e0_gate.py is in only one"]


@needs_git
def test_tag_gate_checks_the_rdev_files_by_name(tmp_path, monkeypatch):
    repo = ScratchRepo(tmp_path, monkeypatch)
    repo.write(f"{E.TRACKS_DIR}/Austin.csv", "lockbox\n")           # a lockbox circuit is there
    opened = []
    real = E.sha256_file
    monkeypatch.setattr(E, "sha256_file", lambda p: opened.append(Path(p).name) or real(p))
    E.tag_gate()
    assert sorted(o for o in opened if o.endswith(".csv")) == sorted(
        f"{n}.csv" for n in E.RDEV_SHA256)
    repo.write(f"{E.TRACKS_DIR}/Sochi.csv", "changed\n")
    with pytest.raises(PermissionError, match="Sochi.csv is not gw4.md's"):
        E.tag_gate()
    (repo.clone / E.TRACKS_DIR / "Sochi.csv").unlink()
    with pytest.raises(PermissionError, match="Sochi.csv is missing"):
        E.tag_gate()


@needs_git
def test_tag_gate_refuses_an_upstream_tag_that_moved(tmp_path, monkeypatch):
    """generator-v1 and sim-v1 must peel to GENERATOR_COMMIT here and on origin."""
    repo = ScratchRepo(tmp_path, monkeypatch)
    repo.git("tag", "-f", "-a", "-m", "moved", "sim-v1", "HEAD")             # here only
    with pytest.raises(PermissionError, match="the local sim-v1 is missing or is not"):
        E.tag_gate()
    repo.git("push", "-q", "-f", "origin", "refs/tags/sim-v1")                # and on origin
    with pytest.raises(PermissionError, match="sim-v1 on origin is"):
        E.tag_gate()
    repo.git("tag", "-d", "generator-v1")
    with pytest.raises(PermissionError, match="the local generator-v1 is missing"):
        E.tag_gate()


@needs_git
def test_tag_gate_compares_the_tree_with_the_pinned_commit_not_the_tags_word(tmp_path,
                                                                             monkeypatch):
    repo = ScratchRepo(tmp_path, monkeypatch)
    monkeypatch.setattr(E, "GENERATOR_COMMIT", repo.git("rev-parse", "HEAD"))    # another commit
    for t in E.UPSTREAM_TAGS:
        repo.git("tag", "-f", "-a", "-m", "moved", t, "HEAD")
        repo.git("push", "-q", "-f", "origin", f"refs/tags/{t}")
    assert E.tag_gate()["upstream"] == dict.fromkeys(E.UPSTREAM_TAGS, repo.git("rev-parse", "HEAD"))
    monkeypatch.setattr(E, "GENERATOR_COMMIT", repo.generator)
    with pytest.raises(PermissionError, match="generator-v1 on origin is"):
        E.tag_gate()
