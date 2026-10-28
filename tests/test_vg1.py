"""VG-1's run lists, C1 script and analysis (docs/prereg/vg1-value-gradient.md), on planted data.

The run lists give the draft's names and flags in the pre-registered order, pass the run-list
contract, and deal the main run's arms evenly over GPUs, lanes and boxes; the main list reads
beta* only from the second pilot's committed, eligible record (Amendment 1: the first pilot
found no beta eligible). scripts/vg1_c1.py's copied functions are
the pinned costate-exploratory blob's byte for byte, and (where the worktree and the costate
data are on this machine) its per-network path reproduces a decision network's C1 from
decisions.json. scripts/vg1_analysis.py's exact rank-sum and sign-flip tests give known p
values, the failed-network rule puts the worst value in each test's direction, the pilot rule
takes each branch (no beta eligible included), the manifest refuses every planted leak or
mismatch, and the tag gate and run-once rules refuse what they must."""

import argparse
import importlib.util
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def VR():
    return _load("vg1_runs_t", "vg1_runs.py")


@pytest.fixture(scope="module")
def VA():
    return _load("vg1_analysis_t", "vg1_analysis.py")


@pytest.fixture(scope="module")
def VC():
    return _load("vg1_c1_t", "vg1_c1.py")


@pytest.fixture(scope="module")
def RM():
    return _load("round_manifest_t", "round_manifest.py")


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                       cwd=root, capture_output=True, text=True, check=True)
    return r.stdout.strip()


def pilot_record(beta_star=0.3, eligible=None) -> dict:
    eligible = eligible or {"0.1": True, "0.3": True, "1.0": True}
    return {"record": "vg1-pilot", "beta_star": beta_star, "eligible": eligible}


def pilot2_record(beta_star=0.03, eligible=None) -> dict:
    eligible = eligible or {"0.01": True, "0.03": True}
    return {"record": "vg1-pilot", "phase": "pilot2", "beta_star": beta_star,
            "eligible": eligible}


NO_BETA = pilot_record(None, {"0.1": False, "0.3": False, "1.0": False})   # the first pilot's


def pilot_repo(tmp_path: Path, rec: dict | None, commit: bool = True,
               first: dict | None = NO_BETA, first_commit: bool = True) -> Path:
    """A repo with the second pilot's record (``rec``, outputs/vg1/pilot2.json) and the first
    pilot's (``first``, outputs/vg1/pilot.json: by default no beta eligible)."""
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "README").write_text("x\n")
    _git(root, "add", "README")
    _git(root, "commit", "-qm", "init")
    for name, r, c in (("pilot", first, first_commit), ("pilot2", rec, commit)):
        if r is None:
            continue
        p = root / "outputs" / "vg1" / f"{name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r))
        if c:
            _git(root, "add", "-f", f"outputs/vg1/{name}.json")
            _git(root, "commit", "-qm", name)
    return root


# ------------------------------------------------------------------------- the run lists


def test_pilot_runs_are_the_drafts(VR, RM, monkeypatch):
    runs = VR.runs("pilot")
    assert [r["name"] for r in runs] == ["vg1p-b0.1-s15", "vg1p-b0.3-s15", "vg1p-b1.0-s15",
                                         "vg1p-b0.1-s16", "vg1p-b0.3-s16", "vg1p-b1.0-s16"]
    for r in runs:
        b = r["name"].split("-")[1][1:]
        assert r["cmd"] == ["scripts/train_dr.py", "--steps", "200e6", "--seed", str(r["seed"]),
                            "--name", r["name"], "--vg-coef", b, "--skip-eval-rdev"]
        assert r["config"] == f"runs/{r['name']}/config.json" and r["beta"] == float(b)
    monkeypatch.chdir(REPO)
    assert RM.run_list_problems(runs) == []
    # one beta per lane at 1 GPU x 3 lanes; at 2 GPUs x 3 lanes each GPU runs every beta once
    lanes = {}
    for lane, _gpu, _pos, run in RM.schedule(runs, 1, 3):
        lanes.setdefault(lane, set()).add(run["beta"])
    assert sorted(map(len, lanes.values())) == [1, 1, 1]
    per_gpu = {}
    for _lane, gpu, pos, run in RM.schedule(runs, 2, 3):
        assert pos == 0
        per_gpu.setdefault(gpu, []).append(run["beta"])
    assert all(sorted(v) == [0.1, 0.3, 1.0] for v in per_gpu.values())


def test_pilot_refuses_a_run_without_the_skip_flag(VR, monkeypatch):
    monkeypatch.setattr(VR, "SKIP", "--no-such-flag")
    real = VR._run

    def drop(name, seed, extra, **k):
        return real(name, seed, [a for a in extra if a != "--no-such-flag"], **k)

    monkeypatch.setattr(VR, "_run", drop)
    with pytest.raises(SystemExit, match="do not pass"):
        VR.pilot_runs()


def test_pilot2_runs_are_the_amendments(VR, RM, monkeypatch):
    runs = VR.runs("pilot2")
    assert [r["name"] for r in runs] == ["vg1p2-b0.01-s15", "vg1p2-b0.03-s15",
                                         "vg1p2-b0.01-s16", "vg1p2-b0.03-s16"]
    for r in runs:
        b = r["name"].split("-")[1][1:]
        assert r["cmd"] == ["scripts/train_dr.py", "--steps", "200e6", "--seed", str(r["seed"]),
                            "--name", r["name"], "--vg-coef", b, "--skip-eval-rdev"]
        assert r["config"] == f"runs/{r['name']}/config.json" and r["beta"] == float(b)
        assert r["arm"] == "pilot2"
    monkeypatch.chdir(REPO)
    assert RM.run_list_problems(runs) == []
    # GPUS=1 LANES_PER_GPU=4: all four at once on one GPU, one run per lane
    sched = RM.schedule(runs, 1, 4)
    assert sorted(lane for lane, *_ in sched) == [0, 1, 2, 3]
    assert all(gpu == 0 and pos == 0 for _lane, gpu, pos, _run in sched)
    assert [run["name"] for _l, _g, _p, run in sorted(sched, key=lambda t: t[0])] == [
        r["name"] for r in runs]


def test_pilot2_refuses_a_run_without_the_skip_flag(VR, monkeypatch):
    monkeypatch.setattr(VR, "SKIP", "--no-such-flag")
    real = VR._run

    def drop(name, seed, extra, **k):
        return real(name, seed, [a for a in extra if a != "--no-such-flag"], **k)

    monkeypatch.setattr(VR, "_run", drop)
    with pytest.raises(SystemExit, match="do not pass"):
        VR.pilot2_runs()


def test_runs_parse_with_train_dr_and_record_the_flags(VR, RM):
    """Each cmd parses with train_dr.py's own parser (rebuilt as round_manifest.py check does):
    --skip-eval-rdev on in the pilot, off in the main run, --vg-coef only where it belongs."""
    parser = RM.trainer_parser((SCRIPTS / "train_dr.py").read_text())
    for r in VR.runs("pilot") + VR.runs("pilot2"):
        args = vars(parser.parse_args(r["cmd"][1:]))
        assert args["skip_eval_rdev"] is True and args["vg_coef"] == r["beta"]
        assert args["steps"] == 200e6 and args["seed"] == r["seed"] and args["name"] == r["name"]
        assert RM.compare_train_dr({"args": dict(args)}, args)[0] == []
    args = vars(parser.parse_args(["--steps", "200e6", "--seed", "30", "--name", "vg1-base-s30"]))
    assert args["skip_eval_rdev"] is False and args["vg_coef"] == 0.0


@needs_git
def test_main_runs_order_names_flags_and_dealing(VR, RM, tmp_path, monkeypatch):
    root = pilot_repo(tmp_path, pilot2_record(0.03))
    runs = VR.runs("main", root)
    assert len(runs) == 40 and len({r["name"] for r in runs}) == 40
    order = [r["name"] for r in runs]
    assert order[:8] == [f"vg1-base-s{s}" for s in range(30, 34)] + [
        f"vg1-vgl-s{s}" for s in range(30, 34)]
    assert order[-8:] == [f"vg1-base-s{s}" for s in range(46, 50)] + [
        f"vg1-vgl-s{s}" for s in range(46, 50)]
    for r in runs:
        base = ["scripts/train_dr.py", "--steps", "200e6", "--seed", str(r["seed"]), "--name",
                r["name"]]
        assert r["cmd"] == (base if r["arm"] == "base" else base + ["--vg-coef", "0.03"])
    assert sorted({r["seed"] for r in runs}) == list(range(30, 50))
    monkeypatch.chdir(REPO)
    assert RM.run_list_problems(runs) == []

    def check(jobs, gpus, lanes):
        """Per GPU: both arms of each seed, 2 base and 2 vgl lanes, never an arm alone."""
        sched = RM.schedule(jobs, gpus, lanes)
        for g in range(gpus):
            mine = [(lane, run) for lane, gpu, _p, run in sched if gpu == g]
            seeds = {}
            for _lane, run in mine:
                seeds.setdefault(run["seed"], set()).add(run["arm"])
            assert all(v == {"base", "vgl"} for v in seeds.values())
            arm_of = {}
            for lane, run in mine:
                arm_of.setdefault(lane, set()).add(run["arm"])
            assert all(len(v) == 1 for v in arm_of.values())
            arms = [next(iter(v)) for v in arm_of.values()]
            assert sorted(arms) == ["base", "base", "vgl", "vgl"]

    check(runs, 4, 4)                               # one 4-GPU box at 4 lanes per GPU
    sched = RM.schedule(runs, 4, 5)                 # at 5 lanes: one run of each arm per lane
    by_lane, gpu_of = {}, {}
    for lane, gpu, _p, run in sched:
        by_lane.setdefault(lane, []).append(run["arm"])
        gpu_of.setdefault(run["seed"], set()).add(gpu)
    assert len(by_lane) == 20 and all(sorted(v) == ["base", "vgl"] for v in by_lane.values())
    assert all(len(v) == 1 for v in gpu_of.values())
    for g in range(4):
        arms = [run["arm"] for _l, gpu, _p, run in sched if gpu == g]
        assert arms.count("base") == 5 and arms.count("vgl") == 5
    for k in (2, 4, 5, 10):                         # K single-GPU boxes, whole seeds
        shards = [RM.in_shard(runs, i, k) for i in range(k)]
        assert sorted(r["name"] for s in shards for r in s) == sorted(order)
        for jobs in shards:
            check(jobs, 1, 4)
    for k in (1, 3):            # another K: both runs of a seed still share the box's GPU
        for i in range(k):
            sched = RM.schedule(RM.in_shard(runs, i, k), 1, 4)
            seeds = {}
            for _lane, _gpu, _p, run in sched:
                seeds.setdefault(run["seed"], set()).add(run["arm"])
            assert all(v == {"base", "vgl"} for v in seeds.values())


@needs_git
@pytest.mark.parametrize("rec, commit, msg", [
    (None, True, "not found"),
    (pilot2_record(0.03), False, "not committed"),
    (pilot2_record(None), True, "no beta is eligible"),
    (pilot2_record(0.3), True, "not in the grid"),          # the first grid's beta
    (pilot2_record(0.1), True, "not in the grid"),
    (pilot2_record(0.05), True, "not in the grid"),
    (pilot2_record(0.03, {"0.01": True, "0.03": False}), True, "not marked eligible"),
    ({**pilot2_record(0.03), "record": "other"}, True, "not VG-1's pilot record"),
    ({k: v for k, v in pilot2_record(0.03).items() if k != "phase"}, True, "second pilot's"),
    ({**pilot2_record(0.03), "phase": "pilot"}, True, "second pilot's"),
])
def test_main_runs_refuse_without_a_committed_eligible_beta(VR, tmp_path, rec, commit, msg):
    root = pilot_repo(tmp_path, rec, commit)
    with pytest.raises(SystemExit, match=msg):
        VR.runs("main", root)


@needs_git
@pytest.mark.parametrize("first, first_commit, msg", [
    (None, True, "pilot.json not found"),
    (NO_BETA, False, "pilot.json is not committed"),
    (pilot_record(0.3), True, "not null"),                  # the first pilot named a beta
    ({**NO_BETA, "record": "other"}, True, "not VG-1's pilot record"),
])
def test_main_runs_refuse_when_the_first_pilot_is_not_a_null_record(VR, tmp_path, first,
                                                                    first_commit, msg):
    root = pilot_repo(tmp_path, pilot2_record(0.03), first=first, first_commit=first_commit)
    with pytest.raises(SystemExit, match=msg):
        VR.runs("main", root)


@needs_git
def test_main_runs_refuse_a_pilot_record_changed_since_commit(VR, tmp_path):
    root = pilot_repo(tmp_path, pilot2_record(0.03))
    (root / "outputs" / "vg1" / "pilot2.json").write_text(json.dumps(pilot2_record(0.01)))
    with pytest.raises(SystemExit, match="differs from HEAD"):
        VR.runs("main", root)
    root = pilot_repo(tmp_path / "b", pilot2_record(0.03))
    (root / "outputs" / "vg1" / "pilot.json").write_text(json.dumps(pilot_record(0.3)))
    with pytest.raises(SystemExit, match="differs from HEAD"):
        VR.runs("main", root)


@needs_git
def test_beta_star_reads_the_second_pilot_by_default_and_either_record_alone(VR, tmp_path):
    root = pilot_repo(tmp_path, pilot2_record(0.01))
    assert VR.beta_star(root) == "0.01"
    # one record alone (first=None): the first pilot's own grid, as the old signature read it
    root = pilot_repo(tmp_path / "b", None, first=pilot_record(0.3))
    assert VR.beta_star(root, VR.PILOT_RECORD, first=None, grid=VR.PILOT_BETAS) == "0.3"


# ------------------------------------------------------------------------- vg1_c1.py's pin


def _pinned(VC) -> str:
    text = VC.pinned_text()
    if text is None:
        pytest.skip("the pinned costate-exploratory commit is not in this clone")
    return text


def test_copied_functions_are_the_pinned_source(VC):
    """Every function in COPIED equals scripts/costate_exploratory.py's at the pinned commit,
    byte for byte, and the commit's blobs are the recorded ones."""
    text = _pinned(VC)
    a, b = VC.function_sources(text), VC.function_sources((SCRIPTS / "vg1_c1.py").read_text())
    for k in VC.COPIED:
        assert a[k] is not None and a[k] == b[k], k
    assert VC.copy_problems(text, (SCRIPTS / "vg1_c1.py").read_text()) == []
    assert VC.pinned_problems() == []
    assert {"freeze_problems", "bound_hashes", "check_inputs", "verify"} <= set(VC.COPIED)


def test_copy_problems_names_an_edited_copy(VC):
    text = _pinned(VC)
    here = (SCRIPTS / "vg1_c1.py").read_text()
    edited = here.replace("    return out + [f\"no record {CS.PREREG_REL} at {TAG}\"]",
                          "    return out")
    edited = edited.replace('        out.append(f"{TAG} is not an annotated tag")',
                            '        pass', 1)
    assert edited != here
    assert VC.copy_problems(text, edited) == ["freeze_problems: differs from the pinned source"]
    gone = here.replace("def bound_hashes(", "def bound_hashes_renamed(", 1)
    assert VC.copy_problems(text, gone) == ["bound_hashes: not copied here"]


def test_vg1_c1_imports_nothing_from_apexzero_until_it_enters_the_worktree(VC):
    """Its module scope is the standard library: CS and F are bound only by enter_worktree."""
    assert VC.CS is None and VC.F is None
    code = ("import sys; sys.path.insert(0, 'scripts'); import vg1_c1; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('apexzero', 'jax', 'numpy', 'costate', 'fields')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "[]"


WT = REPO.parent / "RL-racing-vg1-c1"
DECISIONS = REPO / "outputs" / "costate" / "analyse" / "decisions.json"


@pytest.mark.skipif(not (WT / "scripts" / "costate.py").exists() or not DECISIONS.exists(),
                    reason="needs the vg1_c1 worktree and the costate data on this machine")
def test_measure_reproduces_a_decision_networks_c1(VC, tmp_path):
    """The wiring gate passes, and the per-network path (pin, K3 at 5 heads, pullback, frozen C1)
    gives G1s1's decisions.json C1 (a partial run through a planted manifest)."""
    show = "costate-prereg-v1:src/apexzero/costate/prereg.json"
    rec = json.loads(subprocess.run(["git", "show", show], cwd=REPO, capture_output=True,
                                    text=True, check=True).stdout)
    pin = rec["networks"]["runs/dr-pilot-200M-s1"]
    if not (REPO / pin["path"]).exists():
        pytest.skip("G1s1's checkpoint is not on this machine")
    m = {"record": "vg1-manifest", "phase": "pilot",
         "runs": [{"name": "dr-pilot-200M-s1", "seed": 1, "final": pin, "checkpoints": [pin]}]}
    (tmp_path / "m.json").write_text(json.dumps(m))
    env = {**os.environ, "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false "
                                      "intra_op_parallelism_threads=2"}
    subprocess.run([sys.executable, "scripts/vg1_c1.py", "measure", "--phase", "pilot",
                    "--manifest", str(tmp_path / "m.json"), "--out", str(tmp_path / "c1.json")],
                   cwd=REPO, check=True, env=env, capture_output=True)
    d = json.loads((tmp_path / "c1.json").read_text())
    assert d["gate"]["pass"] and d["partial"]
    assert d["gate"]["checks"]["pullback_bit_for_bit"]
    n = d["networks"]["dr-pilot-200M-s1"]
    want = json.loads(DECISIONS.read_text())["networks"]["runs/dr-pilot-200M-s1"]["C1"]
    assert abs(n["c1"]["stat"] - want["stat"]) <= 1e-12
    assert n["c1"]["n_rows"] == {"s": want["n_rows"][0], "xi": want["n_rows"][2]}
    assert abs(n["c1"]["spearman"]["xi"] - want["spearman"][2]) <= 1e-12
    assert n["k3"]["pass"] and n["k3"]["expected_heads"] == 5 and n["params_finite"]


def test_vg1_c1_refuses_outputs_under_costate_and_sealed(VC):
    for bad in (REPO / "outputs" / "costate" / "x.json", REPO / "outputs" / "sealed" / "x.json"):
        with pytest.raises(SystemExit, match="refusing"):
            VC.refuse_out(bad)
    assert VC.refuse_out(REPO / "outputs" / "vg1" / "c1" / "pilot.json")


# ------------------------------------------------------------------------- the exact tests


def brute_ranksum(x, y, VA):
    r = VA.midranks(np.concatenate([x, y]))
    w = r[:len(x)].sum()
    sums = [r[list(c)].sum() for c in itertools.combinations(range(len(r)), len(x))]
    return (np.mean([s >= w - 1e-9 for s in sums]), np.mean([s <= w + 1e-9 for s in sums]))


def test_midranks_ties_and_infinities(VA):
    r = VA.midranks([3.0, -math.inf, 1.0, 1.0, math.inf, -math.inf])
    assert r.tolist() == [5.0, 1.5, 3.5, 3.5, 6.0, 1.5]
    with pytest.raises(ValueError, match="NaN"):
        VA.midranks([1.0, math.nan])


def test_ranksum_known_p_values(VA):
    t = VA.ranksum_exact([4, 5, 6], [1, 2, 3])
    assert t["w"] == 15 and t["p_greater"] == 1 / 20 and t["p_less"] == 1.0
    assert t["two_sided"] == 0.1
    t = VA.ranksum_exact(np.arange(20, 40), np.arange(20))       # 20 against 20, separated
    assert t["p_greater"] == 1 / math.comb(40, 20) and t["p_less"] == 1.0
    assert t["w"] == sum(range(21, 41)) and t["expected_w"] == 410.0


def test_ranksum_matches_scipy_exact_without_ties(VA):
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(5)
    for shift in (0.0, 0.4, 1.0):
        x, y = rng.normal(shift, 1, 20), rng.normal(0, 1, 20)
        t = VA.ranksum_exact(x, y)
        for alt, key in (("greater", "p_greater"), ("less", "p_less")):
            want = stats.mannwhitneyu(x, y, alternative=alt, method="exact").pvalue
            assert abs(t[key] - want) < 1e-12, (shift, alt)


def test_ranksum_with_ties_and_infinities_matches_enumeration(VA):
    x = [-math.inf, 0.2, 0.2, 1.0, math.inf]
    y = [-math.inf, -math.inf, 0.2, 0.5, 0.1, 3.0]
    t = VA.ranksum_exact(x, y)
    pg, pl = brute_ranksum(np.asarray(x), np.asarray(y), VA)
    assert abs(t["p_greater"] - pg) < 1e-12 and abs(t["p_less"] - pl) < 1e-12
    t = VA.ranksum_exact([1.0, 1.0], [1.0, 1.0])                 # all tied: every p is 1
    assert t["p_greater"] == t["p_less"] == 1.0


def test_hodges_lehmann_and_its_bootstrap(VA):
    assert VA.hodges_lehmann([1, 2, 3], [0]) == 2.0
    assert VA.hodges_lehmann([1, 2], [0, 0]) == 1.5
    # -inf - -inf counts as 0: differences -inf, 0, 1, inf; median (0 + 1) / 2
    assert VA.hodges_lehmann([-math.inf, 1.0], [-math.inf, 0.0]) == 0.5
    assert VA.hodges_lehmann([-math.inf, -math.inf, 0.0], [5.0]) == -math.inf
    a = VA.bootstrap_ci(np.arange(10.0), np.arange(10.0) + 1, "hl", n_boot=500, seed=1)
    b = VA.bootstrap_ci(np.arange(10.0), np.arange(10.0) + 1, "hl", n_boot=500, seed=1)
    assert a == b and a["lo"] <= -1.0 <= a["hi"]
    c = VA.bootstrap_ci([-math.inf] * 3 + [1.0], [0.0, 0.0], "hl", n_boot=200, seed=2)
    assert c["lo"] == -math.inf                 # nearest rank keeps the infinite tail usable
    assert VA.nearest_rank([3.0, 1.0, 2.0, 4.0], 0.5) == 2.0
    assert VA.nearest_rank([-math.inf, 1.0, math.inf], 0.975) == math.inf
    m = VA.bootstrap_ci([1.0, 1.0], [0.0, 0.0], "mean", n_boot=100, seed=3)
    assert m["lo"] == m["hi"] == 1.0


def test_signflip_exact_known_and_enumerated(VA):
    t = VA.signflip_exact([1.0, 2.0, 3.0])
    assert t["flips"] == 8 and t["p_greater"] == 1 / 8 and t["p_less"] == 1.0
    t = VA.signflip_exact([1.0, -1.0])                    # sums 0, 2, -2, 0: ties count
    assert t["p_greater"] == 3 / 4 and t["p_less"] == 3 / 4
    d = np.random.default_rng(3).normal(0.3, 1, 10)
    sums = [np.dot(s, d) for s in itertools.product((1, -1), repeat=10)]
    assert VA.signflip_exact(d)["p_greater"] == np.mean([s >= d.sum() - 1e-12 for s in sums])
    with pytest.raises(ValueError, match="finite"):
        VA.signflip_exact([1.0, -math.inf])


def test_reading_maps_every_cell_to_one_row(VA):
    want = {("refute", True): 1, ("refute", False): 1, ("confirm", True): 2,
            ("confirm", False): 3, ("inconclusive", True): 4, ("not testable", True): 4,
            ("inconclusive", False): 5, ("not testable", False): 5}
    for (p, s3), row in want.items():
        assert VA.reading(p, s3)["row"] == row


def test_primary_is_not_testable_below_15_per_arm(VA):
    p = VA.primary([0.1] * 14, [0.0] * 20)
    assert p["result"] == "not testable"
    p = VA.primary(list(np.linspace(0.5, 1.0, 20)), list(np.linspace(-1.0, 0.4, 20)))
    assert p["result"] == "confirm" and p["p"] == 1 / math.comb(40, 20)
    assert p["hodges_lehmann"] > 0 and p["hodges_lehmann_ci"]["n_boot"] == 20_000
    p = VA.primary(list(np.linspace(-1.0, 0.4, 20)), list(np.linspace(0.5, 1.0, 20)))
    assert p["result"] == "refute"


# ------------------------------------------------------------------------- failed networks


@pytest.fixture
def root(VA, tmp_path, monkeypatch):
    """A planted repo root: R_DEV's source, runs/ and outputs/vg1/."""
    (tmp_path / "src" / "apexzero" / "tracks").mkdir(parents=True)
    shutil.copy(REPO / "src" / "apexzero" / "tracks" / "corners.py",
                tmp_path / "src" / "apexzero" / "tracks" / "corners.py")
    monkeypatch.setattr(VA, "ROOT", tmp_path)
    return tmp_path


def plant_run(root: Path, name: str, ev=0.93, last_nan=False, completion=None, gaps=None):
    d = root / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    rows = ["explained_var,value_loss,timesteps", "0.5,1.0,131072",
            f"{ev},{'nan' if last_nan else 0.01},199884800"]
    (d / "metrics.csv").write_text("\n".join(rows) + "\n")
    if completion is not None:
        (d / "eval_rdev.json").write_text(json.dumps(
            {n: {"completion": c, "median_lap_s": 80.0, "gap": (gaps or {}).get(n, 0.05)}
             for n, c in completion.items()}))


def rdev():
    return ["BrandsHatch", "Hockenheim", "MoscowRaceway", "Norisring", "Nuerburgring",
            "Oschersleben", "Sepang", "Sochi", "YasMarina", "Melbourne", "Catalunya", "Zandvoort"]


def ent(name, seed, arm, failure=None, metrics=True, evals=True):
    return {"name": name, "seed": seed, "arm": arm, "failure": failure,
            "metrics_sha256": "x" if metrics else None, "eval_rdev_sha256": "y" if evals else None,
            "final": {"path": f"runs/{name}/ckpt/000199884800.pkl", "sha256": "z"}}


def c1rec(stat, rho=0.1, finite=True, k3=True, failed=()):
    return {"c1": {"stat": stat, "spearman_mean": rho}, "params_finite": finite,
            "k3": {"pass": k3, "failed": list(failed)}}


def m0rec(m0=8.0, steer=4.0, pedal=6.0):
    return {"normalised_residual_median": m0,
            "action_projected_residual": {"steer": steer, "pedal": pedal}}


def test_the_failed_network_rule(VA, root):
    names = rdev()
    full = dict.fromkeys(names, 1.0)
    plant_run(root, "vg1-vgl-s30", completion={**full, "Sochi": 0.4})
    v = VA.network_values(ent("vg1-vgl-s30", 30, "vgl"), c1rec(0.2), m0rec(), names)
    assert not v["failed"] and v["value"] == {"r2c": 0.2, "rho": 0.1, "m0": 8.0, "s4_steer": 4.0,
                                              "s4_pedal": 6.0, "ev": 0.93, "completion": 11.0}
    # non-finite R2_c, M0 and S4: failed, each at its worst; finite ones as measured
    plant_run(root, "vg1-vgl-s31", completion=full)
    v = VA.network_values(ent("vg1-vgl-s31", 31, "vgl"), c1rec(None, rho=None),
                          m0rec(math.nan, math.inf, 5.0), names)
    assert v["failed"] and "a non-finite R2_c" in v["failed_because"]
    assert v["value"]["r2c"] == v["value"]["rho"] == -math.inf
    assert v["value"]["m0"] == v["value"]["s4_steer"] == math.inf and v["value"]["s4_pedal"] == 5.0
    assert v["value"]["ev"] == 0.93 and v["value"]["completion"] == 12.0
    # a non-finite last metrics row: failed, EV at -inf when it is the non-finite one
    plant_run(root, "vg1-base-s32", ev="nan", completion=full)
    v = VA.network_values(ent("vg1-base-s32", 32, "base"), c1rec(-0.3), m0rec(), names)
    assert v["failed"] and v["value"]["ev"] == -math.inf and v["value"]["r2c"] == -0.3
    plant_run(root, "vg1-base-s33", last_nan=True, completion=full)
    v = VA.network_values(ent("vg1-base-s33", 33, "base"), c1rec(-0.3), m0rec(), names)
    assert v["failed"] and v["value"]["ev"] == 0.93                  # finite: as measured
    # a numerical K3 failure and a non-finite parameter: failed, finite numbers as measured
    plant_run(root, "vg1-base-s34", completion=full)
    v = VA.network_values(ent("vg1-base-s34", 34, "base"),
                          c1rec(0.05, finite=False, k3=False, failed=["float32"]), m0rec(), names)
    assert v["failed"] and len(v["failed_because"]) == 2 and v["value"]["r2c"] == 0.05
    # raised on non-finite numbers (failed twice, classified 'failed'): worst everywhere, C = 0
    v = VA.network_values(ent("vg1-vgl-s35", 35, "vgl", {"class": "failed", "reason": "NaN"},
                              metrics=False, evals=False), None, None, names)
    assert v["failed"] and v["value"] == {**VA.WORST, "completion": 0.0}
    # missing: dropped
    v = VA.network_values(ent("vg1-vgl-s36", 36, "vgl", {"class": "missing", "reason": "OOM"},
                              metrics=False, evals=False), None, None, names)
    assert v["missing"] and "value" not in v


def _vals(VA, root, r2c_vgl, r2c_base):
    names = rdev()
    out = []
    for arm, rs in (("vgl", r2c_vgl), ("base", r2c_base)):
        for i, r in enumerate(rs):
            name = f"vg1-{arm}-s{30 + i}"
            plant_run(root, name, completion=dict.fromkeys(names, 1.0 if i % 2 else 0.0))
            out.append(VA.network_values(ent(name, 30 + i, arm), c1rec(r), m0rec(), names))
    return out


def test_failed_networks_rank_lowest_and_s6_uses_the_floor(VA, root):
    vgl = [0.3 + 0.01 * i for i in range(20)]
    base = [-0.5 + 0.01 * i for i in range(20)]
    vals = _vals(VA, root, [None] + vgl[1:], base)       # vgl s30 failed: R2_c -inf
    t = VA.analysis_tests(vals)
    finite = VA.analysis_tests(_vals(VA, root, [-2.0] + vgl[1:], base))
    # -inf ranks lowest, exactly as any value below all 40 would
    assert t["P"]["w"] == finite["P"]["w"] and t["P"]["p"] == finite["P"]["p"]
    s6 = t["S6"]
    assert s6["substituted_seeds"] == [30] and s6["floor"] == -0.5 and s6["n"] == 20
    assert s6["mean"] == pytest.approx(np.mean([b - a for a, b in zip(
        base, [-0.5] + vgl[1:], strict=True)]))
    assert t["S1"]["result"] == "no difference shown" and t["S1"]["mean_difference"] == 0.0
    assert t["reading"]["row"] in (2, 3)


# ------------------------------------------------------------------------- the pilot rule


def pilot(VA, **over):
    per = {}
    for b in VA.BETAS:
        for s in (15, 16):
            name = f"vg1p-b{b}-s{s}"
            per[name] = {"name": name, "beta": b, "seed": s, "finished": True,
                         "m1": {"0.1": -0.5, "0.3": -0.2, "1.0": -0.4}[b] + (s - 15) * 0.02,
                         "m0": 9.0, "ev": 0.93, "metrics_finite": True, "params_finite": True,
                         "gpu_minutes": {"0.1": 20.0, "0.3": 22.0, "1.0": 24.0}[b]}
    for k, v in over.items():
        per[k].update(v)
    return per


def pilot2(VA, **over):
    """The second pilot's planted per-run values: both betas eligible, 0.03 the better M1."""
    per = {}
    for b in VA.BETAS2:
        for s in (15, 16):
            name = f"vg1p2-b{b}-s{s}"
            per[name] = {"name": name, "beta": b, "seed": s, "finished": True,
                         "m1": {"0.01": -0.6, "0.03": -0.3}[b] + (s - 15) * 0.02,
                         "m0": 9.0, "ev": 0.93, "metrics_finite": True, "params_finite": True,
                         "gpu_minutes": {"0.01": 10.0, "0.03": 12.0}[b]}
    for k, v in over.items():
        per[k].update(v)
    return per


def test_pilot_rule_picks_the_best_eligible_beta(VA):
    r = VA.pilot_rule(pilot(VA))
    assert r["eligible"] == {"0.1": True, "0.3": True, "1.0": True}
    assert r["beta_star"] == 0.3 and r["beta_star_text"] == "0.3"
    assert r["median_m1"]["0.3"] == pytest.approx(-0.19)


def test_pilot_rule_ties_go_to_the_smaller_beta(VA):
    per = pilot(VA, **{"vg1p-b1.0-s15": {"m1": -0.2}, "vg1p-b1.0-s16": {"m1": -0.18}})
    assert VA.pilot_rule(per)["beta_star_text"] == "0.3"


@pytest.mark.parametrize("over, why", [
    ({"ev": 0.8999}, "EV 0.8999 < 0.9"),
    ({"m0": 13.99}, "M0 13.990 >= 13.99"),
    ({"m1": math.nan}, "non-finite metric"),
    ({"m0": None}, "non-finite metric"),
    ({"finished": False}, "did not finish"),
    ({"metrics_finite": False}, "in its last metrics.csv row"),     # EV finite, value_loss NaN
    ({"params_finite": False}, "non-finite parameter"),
    ({"params_finite": None}, "non-finite parameter"),              # not recorded: not finite
])
def test_pilot_rule_eligibility_branches(VA, over, why):
    r = VA.pilot_rule(pilot(VA, **{"vg1p-b0.3-s16": over}))
    assert r["eligible"]["0.3"] is False and any(why in w for w in r["ineligible_because"]["0.3"])
    assert r["beta_star_text"] == "1.0"                # the next best eligible beta


def test_pilot_rule_with_no_beta_eligible(VA):
    per = pilot(VA, **{f"vg1p-b{b}-s15": {"ev": 0.5} for b in VA.BETAS})
    r = VA.pilot_rule(per)
    assert r["beta_star"] is None and not any(r["eligible"].values())
    c = VA.cost_check(r["beta_star_text"], per, 2.0, 4)
    assert c["applies"] is False and "does not start" in c["reason"]


def test_pilot2_rule_picks_over_the_second_grid_only(VA):
    r = VA.pilot_rule(pilot2(VA), VA.BETAS2)
    assert r["eligible"] == {"0.01": True, "0.03": True}
    assert r["beta_star"] == 0.03 and r["beta_star_text"] == "0.03"
    assert r["median_m1"]["0.03"] == pytest.approx(-0.29)
    assert set(r["median_m1"]) == {"0.01", "0.03"}


@pytest.mark.parametrize("over, why", [
    ({"ev": 0.8999}, "EV 0.8999 < 0.9"),
    ({"ev": 0.90}, None),                                   # exactly the bar: eligible
    ({"m0": 13.99}, "M0 13.990 >= 13.99"),
    ({"m0": 13.98}, None),
    ({"m1": math.nan}, "non-finite metric"),
    ({"m0": math.inf}, "non-finite metric"),
    ({"ev": None}, "non-finite metric"),
    ({"finished": False}, "did not finish"),
    ({"metrics_finite": False}, "in its last metrics.csv row"),
    ({"params_finite": False}, "non-finite parameter"),
])
def test_pilot2_rule_eligibility_by_ev_m0_and_non_finite_values(VA, over, why):
    r = VA.pilot_rule(pilot2(VA, **{"vg1p2-b0.03-s16": over}), VA.BETAS2)
    if why is None:
        assert r["eligible"]["0.03"] is True and r["beta_star_text"] == "0.03"
        return
    assert r["eligible"]["0.03"] is False
    assert any(why in w for w in r["ineligible_because"]["0.03"])
    assert r["eligible"]["0.01"] is True and r["beta_star_text"] == "0.01"   # the other beta


def test_pilot2_rule_ties_go_to_the_smaller_beta(VA):
    per = pilot2(VA, **{"vg1p2-b0.01-s15": {"m1": -0.3}, "vg1p2-b0.01-s16": {"m1": -0.3 + 0.02}})
    assert VA.pilot_rule(per, VA.BETAS2)["beta_star_text"] == "0.01"


def test_pilot2_rule_with_none_eligible(VA):
    per = pilot2(VA, **{f"vg1p2-b{b}-s15": {"ev": 0.5} for b in VA.BETAS2})
    r = VA.pilot_rule(per, VA.BETAS2)
    assert r["beta_star"] is None and r["beta_star_text"] is None
    assert not any(r["eligible"].values())
    c = VA.cost_check(None, per, 1.0, 4, VA.FIRST_PILOT_GPU_HOURS)
    assert c["applies"] is False and "does not start" in c["reason"]
    # a run missing from the grid: not 2 runs
    r = VA.pilot_rule({k: v for k, v in pilot2(VA).items() if k != "vg1p2-b0.01-s16"},
                      VA.BETAS2)
    assert r["eligible"] == {"0.01": False, "0.03": True} and r["beta_star_text"] == "0.03"


def test_pilot2_cost_check_adds_the_first_pilots_hours_and_the_threshold(VA):
    assert VA.FIRST_PILOT_GPU_HOURS == 1.50
    per = pilot2(VA)
    c = VA.cost_check("0.03", per, 1.2, 4, VA.FIRST_PILOT_GPU_HOURS)
    main = (20 * 6.5 + 20 * 12.0 + 15 * 4) / 60
    assert c["first_pilot_gpu_hours"] == 1.50 and c["this_pilot_gpu_hours"] == 1.2
    assert c["pilot_gpu_hours"] == pytest.approx(2.7)
    assert c["main_projected_gpu_hours"] == pytest.approx(main)
    assert c["total"] == pytest.approx(2.7 + main) and c["to_the_user"] is False
    # without the first pilot's term (the first pilot's own call) the total is 1.5 lower
    assert VA.cost_check("0.03", per, 1.2, 4)["total"] == pytest.approx(1.2 + main)
    # the threshold: just under and just over 14 GPU-hours
    under = 14.0 - main - 1.50
    assert VA.cost_check("0.03", per, under - 1e-6, 4, 1.50)["to_the_user"] is False
    assert VA.cost_check("0.03", per, under + 1e-6, 4, 1.50)["to_the_user"] is True
    assert VA.cost_check("0.03", per, 1.2, 4, 1.50)["limit"] == 14.0
    # a run without times is not computable: to the user
    per = pilot2(VA, **{"vg1p2-b0.03-s15": {"gpu_minutes": math.nan}})
    assert VA.cost_check("0.03", per, 1.2, 4, 1.50)["to_the_user"] is True


def test_cost_check_in_gpu_hours(VA):
    per = pilot(VA)
    c = VA.cost_check("0.3", per, 2.5, 4)
    main = (20 * 6.5 + 20 * 22.0 + 15 * 4) / 60
    assert c["main_projected_gpu_hours"] == pytest.approx(main)
    assert c["total"] == pytest.approx(2.5 + main) and c["to_the_user"] is False
    per = pilot(VA, **{f"vg1p-b0.3-s{s}": {"gpu_minutes": 33.0} for s in (15, 16)})
    c = VA.cost_check("0.3", per, 2.5, 4)
    assert c["total"] > 14 and c["to_the_user"] is True and c["stop_rule"] == 16.8
    # the r of the draft's Section 8 table: a single-GPU pilot box, the main run on the x4 box
    for r, total in ((2.0, 9.05), (3.6, 13.557), (4.1, 14.965)):      # about 9, 13.6, 15
        per = pilot(VA, **{f"vg1p-b0.3-s{s}": {"gpu_minutes": 6.5 * r} for s in (15, 16)})
        box = 6 * 6.5 * r / 60 + 0.25
        assert VA.cost_check("0.3", per, box, 4)["total"] == pytest.approx(total, abs=5e-4)
    # a run without times: the projection is not computable and goes to the user
    per = pilot(VA, **{"vg1p-b0.3-s15": {"gpu_minutes": math.nan}})
    c = VA.cost_check("0.3", per, 2.5, 4)
    assert not math.isfinite(c["total"]) and c["to_the_user"] is True


@needs_git
def test_the_pilot_record_feeds_the_main_run_list(VA, VR, tmp_path):
    """The rule's beta* and eligibility, as pilot2.json holds them, are what vg1_runs reads,
    given a first pilot's record with no beta eligible."""
    first = json.loads(json.dumps(VA.clean({"record": "vg1-pilot", **VA.pilot_rule(
        pilot(VA, **{f"vg1p-b{b}-s15": {"ev": 0.5} for b in VA.BETAS}))})))
    assert first["beta_star"] is None
    r = VA.pilot_rule(pilot2(VA), VA.BETAS2)
    rec = json.loads(json.dumps(VA.clean({"record": "vg1-pilot", "phase": "pilot2", **r})))
    root = pilot_repo(tmp_path, rec, first=first)
    vgl = [x for x in VR.runs("main", root) if x["arm"] == "vgl"]
    assert len(vgl) == 20 and all(x["cmd"][-2:] == ["--vg-coef", "0.03"] for x in vgl)
    # a first pilot that had named a beta leaves no room for the second
    root = pilot_repo(tmp_path / "b", rec, first={**first, "beta_star": 0.3})
    with pytest.raises(SystemExit, match="not null"):
        VR.runs("main", root)


def test_gpu_minutes_share_each_gpu(VA):
    def e(name, gpu, start, end, box="vg1p-manifest-0of1.json"):
        return {"name": name, "box": {"box": box, "gpu": gpu, "start": f"2026-10-28T{start}Z",
                                      "end": f"2026-10-28T{end}Z", "gpu_count": 1}}

    ents = [e("a", 0, "10:00:00", "11:00:00"), e("b", 0, "10:00:00", "11:00:00"),
            e("c", 0, "10:00:00", "11:00:00"), e("d", 0, "11:00:00", "11:30:00"),
            e("x", 0, "10:00:00", "10:30:00", box="other.json")]
    gm = VA.gpu_minutes(ents)
    assert gm["a"] == gm["b"] == gm["c"] == pytest.approx(20.0)
    assert gm["d"] == pytest.approx(30.0) and gm["x"] == pytest.approx(30.0)
    ents = [e("a", 0, "10:00:00", "11:00:00"), e("b", 0, "10:30:00", "11:00:00")]
    gm = VA.gpu_minutes(ents)
    assert gm["a"] == pytest.approx(45.0) and gm["b"] == pytest.approx(15.0)
    assert VA.queue_gpu_hours(ents) == pytest.approx(1.0)


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset")
def test_gpu_minutes_read_the_manifests_times_as_utc(VA, monkeypatch):
    """round_manifest.py writes UTC. Read as local time (mktime), a span across New York's
    fall-back (1 Nov 2026, 02:00 local) gained an hour: 00:30 to 02:30 read as 3 h, not 2."""
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        span = [{"name": "a", "box": {"box": "m", "gpu": 0, "start": "2026-11-01T00:30:00Z",
                                      "end": "2026-11-01T02:30:00Z", "gpu_count": 1}}]
        assert VA.gpu_minutes(span) == {"a": pytest.approx(120.0)}
        assert VA.queue_gpu_hours(span) == pytest.approx(2.0)
        assert VA._t("2026-11-01T00:00:00Z") == 1793491200.0
    finally:
        monkeypatch.undo()
        time.tzset()


# ------------------------------------------------------------------------- the manifest


def plant_pilot_round(VA, VR, root: Path, phase: str = "pilot") -> tuple[list, Path]:
    runs = VR.runs(phase)
    rnd = VA.ROUNDS[phase]
    recs = []
    for k, r in enumerate(runs):
        d = root / "runs" / r["name"]
        (d / "ckpt").mkdir(parents=True)
        for ts in (10_000_000, 199_884_800):
            (d / "ckpt" / f"{ts:012d}.pkl").write_bytes(os.urandom(64))
        (d / "config.json").write_text(json.dumps({
            "args": {"seed": r["seed"], "skip_eval_rdev": True, "vg_coef": r["beta"],
                     "vg_lambda": 0.95, "vg_winsor": 0.005},
            "vg": {"coef": r["beta"], "lam": 0.95, "winsor": 0.005}}))
        (d / "metrics.csv").write_text("explained_var\n0.93\n")
        (root / "runs" / f"{r['name']}.log").write_text("--skip-eval-rdev: no R-dev evaluation\n")
        recs.append({"name": r["name"], "seed": r["seed"], "status": "done", "lane": k % 3,
                     "gpu": 0, "start": "2026-10-28T10:00:00Z", "end": "2026-10-28T11:00:00Z"})
    (root / "runs" / f"{rnd}-manifest-0of1.json").write_text(json.dumps(
        {"runs": recs, "gpu_model": "NVIDIA RTX A6000", "gpu_count": 1,
         "box_type": "massedcompute_A6000_plus"}))
    sums = root / "sums.txt"
    lines = [f"{VA.sha256_file(p)}  {p.relative_to(root).as_posix()}"
             for p in sorted((root / "runs").glob(f"{rnd}-*/ckpt/*.pkl"))]
    sums.write_text("\n".join(lines) + "\n")
    return runs, sums


class StubGit:
    def __call__(self, *args):
        out = "c" * 40 if args[0] == "rev-parse" else ""
        return subprocess.CompletedProcess(["git", *args], 0, out, "")


CHECK = {"returncode": 0, "tolerated": [], "report_sha256": "0", "report": "OK"}


def test_manifest_records_every_file_and_the_box(VA, VR, root, monkeypatch):
    monkeypatch.setattr(VA, "git_run", StubGit())
    runs, sums = plant_pilot_round(VA, VR, root)
    m = VA.build_manifest("pilot", "c" * 40, VA.box_sums([str(sums)]), runs, {}, CHECK, None)
    assert m["record"] == "vg1-manifest" and m["commit"] == "c" * 40 and len(m["runs"]) == 6
    e = m["runs"][0]
    assert e["final"]["timesteps"] == VA.FINAL_TIMESTEPS and len(e["checkpoints"]) == 2
    assert e["final"]["sha256"] == VA.sha256_file(root / e["final"]["path"])
    assert e["eval_rdev_sha256"] is None and e["box"]["gpu_model"] == "NVIDIA RTX A6000"
    assert set(m["round_files"]) == {"runs/vg1p-manifest-0of1.json"}
    gm = VA.gpu_minutes(m["runs"])
    assert all(v == pytest.approx(10.0) for v in gm.values())         # 6 at once on one GPU


@pytest.mark.parametrize("plant, msg", [
    (lambda r: (r / "runs" / "vg1p-b0.3-s15" / "eval_rdev.json").write_text("{}"),
     "eval_rdev.json exists"),
    (lambda r: (r / "runs" / "vg1p-b0.3-s15.log").write_text("Hockenheim {\"completion\": 1}\n"),
     r"names \['Hockenheim'"),
    (lambda r: (r / "runs" / "vg1p-b1.0-s16" / "ckpt" / "000199884800.pkl").write_bytes(b"x"),
     "is not the box's file"),
    (lambda r: (r / "runs" / "vg1p-b0.1-s16" / "config.json").write_text(json.dumps(
        {"args": {"seed": 16, "skip_eval_rdev": False, "vg_lambda": 0.95, "vg_winsor": 0.005},
         "vg": {"coef": 0.1, "lam": 0.95, "winsor": 0.005}})),
     "--skip-eval-rdev is not in its config"),
    (lambda r: (r / "runs" / "vg1p-b0.1-s16" / "config.json").write_text(json.dumps(
        {"args": {"seed": 16, "skip_eval_rdev": True, "vg_lambda": 0.95, "vg_winsor": 0.005},
         "vg": {"coef": 0.3, "lam": 0.95, "winsor": 0.005}})),
     r"config vg \{"),
    (lambda r: (r / "runs" / "vg1p-b0.1-s16" / "config.json").write_text(json.dumps(
        {"args": {"seed": 16, "skip_eval_rdev": True, "vg_lambda": 0.9, "vg_winsor": 0.005},
         "vg": {"coef": 0.1, "lam": 0.95, "winsor": 0.005}})),
     "config vg_lambda 0.9"),
    (lambda r: (r / "runs" / "vg1p-b0.1-s16" / "ckpt" / "000199884800.pkl").unlink(),
     "no final checkpoint"),
])
def test_manifest_refuses_leaks_and_mismatches(VA, VR, root, monkeypatch, plant, msg):
    monkeypatch.setattr(VA, "git_run", StubGit())
    runs, sums = plant_pilot_round(VA, VR, root)
    plant(root)
    with pytest.raises(VA.CheckFailed, match=msg):
        VA.build_manifest("pilot", "c" * 40, VA.box_sums([str(sums)]), runs, {}, CHECK, None)


def test_manifest_pilot2_records_the_second_round(VA, VR, root, monkeypatch):
    monkeypatch.setattr(VA, "git_run", StubGit())
    runs, sums = plant_pilot_round(VA, VR, root, "pilot2")
    m = VA.build_manifest("pilot2", "c" * 40, VA.box_sums([str(sums)]), runs, {}, CHECK, None)
    assert m["phase"] == "pilot2" and m["round"] == "vg1p2" and len(m["runs"]) == 4
    assert [e["name"] for e in m["runs"]] == [r["name"] for r in runs]
    assert all(e["eval_rdev_sha256"] is None for e in m["runs"])
    assert set(m["round_files"]) == {"runs/vg1p2-manifest-0of1.json"}
    assert all(v == pytest.approx(15.0) for v in VA.gpu_minutes(m["runs"]).values())   # 4 at once


@pytest.mark.parametrize("plant, msg", [
    (lambda r: (r / "runs" / "vg1p2-b0.03-s15" / "eval_rdev.json").write_text("{}"),
     "eval_rdev.json exists"),
    (lambda r: (r / "runs" / "vg1p2-b0.01-s16.log").write_text("Sochi {\"completion\": 1}\n"),
     r"names \['Sochi'"),
    (lambda r: (r / "runs" / "vg1p2-b0.01-s16.log").write_text("the completion is 1\n"),
     r"names \['completion'\]"),
    (lambda r: (r / "runs" / "vg1p2-b0.01-s16" / "config.json").write_text(json.dumps(
        {"args": {"seed": 16, "skip_eval_rdev": False, "vg_lambda": 0.95, "vg_winsor": 0.005},
         "vg": {"coef": 0.01, "lam": 0.95, "winsor": 0.005}})),
     "--skip-eval-rdev is not in its config"),
    (lambda r: (r / "runs" / "vg1p2-b0.01-s16" / "config.json").write_text(json.dumps(
        {"args": {"seed": 16, "skip_eval_rdev": True, "vg_lambda": 0.95, "vg_winsor": 0.005},
         "vg": {"coef": 0.03, "lam": 0.95, "winsor": 0.005}})),
     r"config vg \{"),
    (lambda r: (r / "runs" / "vg1p2-b0.03-s16" / "ckpt" / "000199884800.pkl").write_bytes(b"x"),
     "is not the box's file"),
    (lambda r: (r / "runs" / "vg1p2-b0.03-s16" / "ckpt" / "000199884800.pkl").unlink(),
     "no final checkpoint"),
])
def test_manifest_pilot2_refuses_leaks_and_mismatches(VA, VR, root, monkeypatch, plant, msg):
    monkeypatch.setattr(VA, "git_run", StubGit())
    runs, sums = plant_pilot_round(VA, VR, root, "pilot2")
    plant(root)
    with pytest.raises(VA.CheckFailed, match=msg):
        VA.build_manifest("pilot2", "c" * 40, VA.box_sums([str(sums)]), runs, {}, CHECK, None)


def test_manifest_needs_a_classification_for_a_failed_run(VA, VR, root, monkeypatch):
    monkeypatch.setattr(VA, "git_run", StubGit())
    runs, sums = plant_pilot_round(VA, VR, root)
    p = root / "runs" / "vg1p-manifest-0of1.json"
    m = json.loads(p.read_text())
    m["runs"][2]["status"] = "failed"
    p.write_text(json.dumps(m))
    shutil.rmtree(root / "runs" / "vg1p-b1.0-s15" / "ckpt")
    sums.write_text("\n".join(ln for ln in sums.read_text().splitlines()
                              if "vg1p-b1.0-s15" not in ln) + "\n")
    with pytest.raises(VA.CheckFailed, match="not in failures.json"):
        VA.build_manifest("pilot", "c" * 40, VA.box_sums([str(sums)]), runs, {}, CHECK, None)
    fails = {"vg1p-b1.0-s15": {"class": "missing", "reason": "the box ran out of memory"}}
    out = VA.build_manifest("pilot", "c" * 40, VA.box_sums([str(sums)]), runs, fails, CHECK,
                            None)
    e = next(x for x in out["runs"] if x["name"] == "vg1p-b1.0-s15")
    assert e["failure"]["class"] == "missing" and e["final"] is None


def test_main_configs_carry_the_arms_flags(VA):
    run = {"name": "vg1-vgl-s30", "seed": 30, "arm": "vgl"}
    vg_args = {"vg_lambda": 0.95, "vg_winsor": 0.005}
    good = {"args": {"seed": 30, "skip_eval_rdev": False, "vg_coef": 0.3, **vg_args},
            "vg": {"coef": 0.3, "lam": 0.95, "winsor": 0.005}}
    assert VA.config_problems(run, good, "main", "0.3") == []
    assert VA.config_problems(run, {**good, "vg": {**good["vg"], "coef": 1.0}}, "main", "0.3")
    assert VA.config_problems(run, {**good, "vg": {**good["vg"], "lam": 0.9}}, "main", "0.3")
    base = {"name": "vg1-base-s30", "seed": 30, "arm": "base"}
    ok = {"args": {"seed": 30, "skip_eval_rdev": False, "vg_coef": 0.0, **vg_args}}
    assert VA.config_problems(base, ok, "main", "0.3") == []
    assert VA.config_problems(base, {**ok, "vg": good["vg"]}, "main", "0.3")
    assert VA.config_problems(base, {"args": {**ok["args"], "skip_eval_rdev": True}}, "main",
                              "0.3")
    # round_manifest.py check compares --vg-lambda and --vg-winsor only where a cmd sets them
    # (computed defaults), and no VG-1 cmd does: the manifest checks the recorded arguments
    for key, v in (("vg_lambda", 0.9), ("vg_winsor", 0.01)):
        for r, c in ((run, good), (base, ok)):
            got = VA.config_problems(r, {**c, "args": {**c["args"], key: v}}, "main", "0.3")
            assert any(key in p for p in got), (r["name"], key)
    pilot_run = {"name": "vg1p-b0.1-s15", "seed": 15, "arm": "pilot", "beta": 0.1}
    pc = {"args": {"seed": 15, "skip_eval_rdev": True, "vg_coef": 0.1, **vg_args},
          "vg": {"coef": 0.1, "lam": 0.95, "winsor": 0.005}}
    assert VA.config_problems(pilot_run, pc, "pilot", None) == []
    assert VA.config_problems(pilot_run, {**pc, "args": {**pc["args"], "vg_lambda": 0.5}},
                              "pilot", None)
    p2 = {"name": "vg1p2-b0.01-s15", "seed": 15, "arm": "pilot2", "beta": 0.01}
    c2 = {"args": {"seed": 15, "skip_eval_rdev": True, "vg_lambda": 0.95, "vg_winsor": 0.005},
          "vg": {"coef": 0.01, "lam": 0.95, "winsor": 0.005}}
    assert VA.config_problems(p2, c2, "pilot2", None) == []
    assert VA.config_problems(p2, {**c2, "args": {**c2["args"], "skip_eval_rdev": False}},
                              "pilot2", None)


def test_round_check_tolerates_only_classified_runs(VA, monkeypatch):
    report = ("round vg1: ...\nruns: done 39, failed 1\nFAILED: 2 problem(s)\n"
              "  vg1-vgl-s33: status 'failed' in runs/vg1-manifest-0of1.json\n"
              "  vg1-vgl-s33: runs/vg1-vgl-s33 is not here\n")
    monkeypatch.setattr(VA, "check_report", lambda phase, tag: (1, report))
    got = VA.round_check("main", "vg1-prereg-v1", {"vg1-vgl-s33": {"class": "failed"}})
    assert len(got["tolerated"]) == 2
    with pytest.raises(VA.CheckFailed, match="vg1-vgl-s33: status"):
        VA.round_check("main", "vg1-prereg-v1", {})
    worse = report + "  runs/vg1-manifest-0of1.json: the box's tracked tree was not clean\n"
    monkeypatch.setattr(VA, "check_report", lambda phase, tag: (1, worse))
    with pytest.raises(VA.CheckFailed, match="not clean"):
        VA.round_check("main", "vg1-prereg-v1", {"vg1-vgl-s33": {"class": "failed"}})
    monkeypatch.setattr(VA, "check_report", lambda phase, tag: (1, "tag 'x' not found"))
    with pytest.raises(VA.CheckFailed, match="did not run"):
        VA.round_check("main", "vg1-prereg-v1", {})


def test_m0_and_c1_records_are_checked_against_the_manifest(VA, monkeypatch):
    monkeypatch.setattr(VA, "code_at", lambda head, commit, paths=VA.CODE_PATHS: head == "good")
    e = {"name": "vg1p-b0.3-s15", "final": {"path": "runs/vg1p-b0.3-s15/ckpt/000199884800.pkl",
                                            "sha256": "s"}}
    m0 = {"run": "vg1p-b0.3-s15", "checkpoint": "000199884800.pkl", "checkpoint_sha256": "s",
          "sizes": {**VA.M0_SIZES, "refresh": 16}, "vg": dict(VA.M0_VG),
          "environment": {"head": "good", "src_or_scripts_dirty": False}}
    side = {"vg1_code_clean": True, "m0_sha256": "h", "head": "good", "commit": "c"}
    assert VA.check_m0(m0, e, "c", side, "h") == []
    assert VA.check_m0({**m0, "checkpoint_sha256": "t"}, e, "c", side, "h")
    assert VA.check_m0({**m0, "sizes": {**VA.M0_SIZES, "seed": 1}}, e, "c", side, "h")
    assert VA.check_m0({**m0, "environment": {"head": "bad", "src_or_scripts_dirty": False}},
                       e, "c", {**side, "head": "bad"}, "h")
    # vgl.py's own clean flag trusts the index (assume-unchanged, skip-worktree): only m0's
    # byte-level sidecar vouches, and other studies' untracked files (dirty here) do not count
    assert VA.check_m0(m0, e, "c")
    dirty = {**m0, "environment": {"head": "good", "src_or_scripts_dirty": True}}
    assert VA.check_m0(dirty, e, "c", side, "h") == []
    assert VA.check_m0(dirty, e, "c", side, "other")              # not this record's sidecar
    assert VA.check_m0(dirty, e, "c", {**side, "head": "bad"}, "h")
    assert VA.check_m0(dirty, e, "c", {**side, "vg1_code_clean": False}, "h")
    assert VA.check_m0(dirty, e, "c", {**side, "commit": "d"}, "h")      # checked elsewhere
    man = {"_sha256": "m", "runs": [e]}
    monkeypatch.setattr(VA, "blob_sha256", lambda rev, rel: {"c": "v"}.get(rev))
    c1 = {"manifest_sha256": "m", "partial": False, "gate": {"pass": True},
          "provenance": {"head": "good", "changed": [" M docs/x.md"], "script_sha256": "v"},
          "networks": {"vg1p-b0.3-s15": {"checkpoint": {"sha256": "s"}}}}
    assert VA.check_c1(c1, man, "c") == []
    assert VA.check_c1({**c1, "partial": True}, man, "c")
    assert VA.check_c1({**c1, "gate": {"pass": False}}, man, "c")
    assert VA.check_c1({**c1, "provenance": {**c1["provenance"],
                                             "changed": [" M scripts/vgl.py"]}}, man, "c")
    assert VA.check_c1({**c1, "networks": {}}, man, "c")
    # vg1_c1.py's own bytes: an edit the index hides still changes its recorded sha256
    assert any("vg1_c1.py is not" in p for p in VA.check_c1(
        {**c1, "provenance": {**c1["provenance"], "script_sha256": "w"}}, man, "c"))
    assert VA.check_c1(c1, man, "d")


# ------------------------------------------------------------------------- the tag gate


OBJ, COMMIT, HEAD = "a" * 40, "b" * 40, "c" * 40


class FakeGit:
    """``git_run`` with an annotated vg1-prereg-v1 on origin and locally, HEAD descending from
    it and VG-1's code unchanged, unless a field says otherwise."""

    def __init__(self, **kw):
        self.remote = (f"{OBJ}\trefs/tags/vg1-prereg-v1\n"
                       f"{COMMIT}\trefs/tags/vg1-prereg-v1^{{}}\n")
        self.remote_rc, self.local, self.kind, self.commit = 0, OBJ, "tag", COMMIT
        self.ancestor, self.diff_clean, self.untracked, self.log = True, True, "", ""
        self.__dict__.update(kw)

    def __call__(self, *args):
        def cp(rc, out=""):
            return subprocess.CompletedProcess(["git", *args], rc, out, "")

        cmd = args[0]
        if cmd == "ls-remote":
            return cp(self.remote_rc, self.remote if self.remote_rc == 0 else "")
        if cmd == "rev-parse":
            if args[-1] == "refs/tags/vg1-prereg-v1":
                return cp(0 if self.local else 1, self.local or "")
            if args[-1].endswith("^{commit}"):
                return cp(0, self.commit)
            return cp(0, HEAD)
        if cmd == "cat-file":
            return cp(0, self.kind)
        if cmd == "merge-base":
            return cp(0 if self.ancestor else 1)
        if cmd == "diff":
            return cp(0 if self.diff_clean else 1)
        if cmd == "ls-files":
            return cp(0, self.untracked if "--others" in args else "")
        if cmd in ("ls-tree", "hash-object"):           # no files: code_byte_problems passes
            return cp(0, "")
        if cmd == "log":
            return cp(0, self.log)
        if cmd == "status":
            return cp(0, "")
        raise AssertionError(f"unexpected git call {args}")


def test_tag_gate_passes_on_the_full_tag(VA, monkeypatch):
    monkeypatch.setattr(VA, "git_run", FakeGit())
    assert VA.tag_gate() == {"tag": "vg1-prereg-v1", "tag_object": OBJ, "commit": COMMIT,
                             "head": HEAD}


@pytest.mark.parametrize("change", [
    {"remote": ""}, {"remote_rc": 128}, {"remote": f"{COMMIT}\trefs/tags/vg1-prereg-v1\n"},
    {"local": ""}, {"local": "d" * 40}, {"kind": "commit"}, {"commit": "e" * 40},
    {"ancestor": False}, {"diff_clean": False}, {"untracked": "scripts/vg1_new.py\n"},
])
def test_tag_gate_refuses(VA, monkeypatch, change):
    monkeypatch.setattr(VA, "git_run", FakeGit(**change))
    with pytest.raises(PermissionError):
        VA.tag_gate()


class ScratchRepo:
    """A bare ``origin`` and a clone of it, with real git and no user config, holding a stand-in
    for VG-1's code (``CODE``); ``VA.ROOT`` points at the clone."""

    CODE = ("scripts/vg1_analysis.py", "src/apexzero/agents/ppo.py")

    def __init__(self, tmp_path: Path, monkeypatch, VA):
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
        for rel in self.CODE:
            (self.clone / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.clone / rel).write_text(f"# {rel}\nALPHA = 0.05\n")
        self.git("add", "-A")
        self.git("commit", "-qm", "first")
        self.git("tag", "-a", "-m", "VG-1", "vg1-prereg-v1")
        self.git("push", "-q", "origin", "HEAD:refs/heads/main", "refs/tags/vg1-prereg-v1")
        monkeypatch.setattr(VA, "ROOT", self.clone)

    def git(self, *args) -> str:
        return subprocess.run(["git", *args], cwd=self.clone, check=True, capture_output=True,
                              text=True).stdout.strip()


@needs_git
def test_tag_gate_passes_on_a_real_tag(VA, tmp_path, monkeypatch):
    repo = ScratchRepo(tmp_path, monkeypatch, VA)
    got = VA.tag_gate(paths=("scripts", "src/apexzero"))
    assert got["commit"] == repo.git("rev-parse", "HEAD") and got["tag_object"] != got["commit"]
    assert VA.code_byte_problems(got["commit"], ("scripts", "src/apexzero")) == []


@needs_git
@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_tag_gate_refuses_an_edit_git_diff_does_not_show(VA, tmp_path, monkeypatch, flag):
    """git diff and git status skip a file marked assume-unchanged or skip-worktree; the gate
    and m0's check compare the bytes on disk with the tag's."""
    repo = ScratchRepo(tmp_path, monkeypatch, VA)
    rel = "src/apexzero/agents/ppo.py"
    repo.git("update-index", flag, rel)
    (repo.clone / rel).write_text(f"# {rel}\nALPHA = 0.10\n")
    assert repo.git("diff", "--name-only", "vg1-prereg-v1") == ""
    assert repo.git("status", "--porcelain", "--untracked-files=no") == ""
    with pytest.raises(PermissionError, match="ppo.py on disk is not"):
        VA.tag_gate(paths=("scripts", "src/apexzero"))
    assert VA.code_byte_problems(repo.git("rev-parse", "HEAD"), ("src/apexzero",)) == [
        f"{rel} on disk is not {repo.git('rev-parse', 'HEAD')[:12]}'s"]


@needs_git
def test_code_byte_problems_names_a_missing_and_an_added_file(VA, tmp_path, monkeypatch):
    repo = ScratchRepo(tmp_path, monkeypatch, VA)
    tag = repo.git("rev-parse", "HEAD")
    repo.git("update-index", "--skip-worktree", "scripts/vg1_analysis.py")
    (repo.clone / "scripts" / "vg1_analysis.py").unlink()
    (repo.clone / "src" / "apexzero" / "new.py").write_text("x = 1\n")
    repo.git("add", "src/apexzero/new.py")
    got = VA.code_byte_problems(tag, ("scripts", "src/apexzero"))
    assert got == [f"src/apexzero/new.py is tracked here but not in {tag[:12]}",
                   "scripts/vg1_analysis.py is missing on disk"]


def test_analyse_runs_once(VA, root, monkeypatch):
    monkeypatch.setattr(VA, "git_run", FakeGit())
    (root / "outputs" / "vg1").mkdir(parents=True)
    (root / "outputs" / "vg1" / "analysis.lock").write_text("x")
    with pytest.raises(VA.CheckFailed, match="runs once"):
        VA.cmd_analyse(None)
    (root / "outputs" / "vg1" / "analysis.lock").unlink()
    monkeypatch.setattr(VA, "git_run", FakeGit(log="f" * 40))     # analysis.json was committed
    with pytest.raises(VA.CheckFailed, match="runs once"):
        VA.cmd_analyse(None)
    monkeypatch.setattr(VA, "git_run", FakeGit(remote=""))         # no tag: refused before inputs
    with pytest.raises(PermissionError, match="not on origin"):
        VA.cmd_analyse(None)
    assert not (root / "outputs" / "vg1" / "analysis.lock").exists()


def test_pilot_rule_runs_once(VA, root, monkeypatch):
    monkeypatch.setattr(VA, "git_run", FakeGit(log="f" * 40))
    with pytest.raises(VA.CheckFailed, match="runs once"):
        VA.cmd_pilot(argparse.Namespace(phase="pilot"))


def test_pilot2_json_is_write_once(VA, root, monkeypatch):
    (root / "outputs" / "vg1").mkdir(parents=True)
    (root / "outputs" / "vg1" / "pilot2.json").write_text("{}")      # on disk
    monkeypatch.setattr(VA, "git_run", FakeGit())
    with pytest.raises(VA.CheckFailed, match="pilot2.json exists or was committed"):
        VA.cmd_pilot(argparse.Namespace(phase="pilot2"))
    (root / "outputs" / "vg1" / "pilot2.json").unlink()
    monkeypatch.setattr(VA, "git_run", FakeGit(log="f" * 40))        # ever committed
    with pytest.raises(VA.CheckFailed, match="pilot2.json exists or was committed"):
        VA.cmd_pilot(argparse.Namespace(phase="pilot2"))
    # a first pilot that is not committed: refused before any run is read
    monkeypatch.setattr(VA, "git_run", FakeGit(diff_clean=False))
    with pytest.raises(VA.CheckFailed, match="pilot.json is not committed"):
        VA.cmd_pilot(argparse.Namespace(phase="pilot2", box_gpu_hours=1.0, main_gpus=4))
    assert not (root / "outputs" / "vg1" / "pilot2.json").exists()


def test_pilot2_binds_the_first_pilot_and_needs_its_beta_null(VA, root, monkeypatch):
    out = root / "outputs" / "vg1"
    out.mkdir(parents=True)
    monkeypatch.setattr(VA, "git_run", FakeGit())
    a = argparse.Namespace(phase="pilot2", box_gpu_hours=1.0, main_gpus=4)
    with pytest.raises(VA.CheckFailed, match="pilot.json not found"):
        VA.cmd_pilot(a)
    (out / "pilot.json").write_text(json.dumps(pilot_record(0.3)))
    with pytest.raises(VA.CheckFailed, match="not null"):
        VA.cmd_pilot(a)
    (out / "pilot.json").write_text(json.dumps(pilot_record("0.1")))
    with pytest.raises(VA.CheckFailed, match=r"beta_star is '0.1', not null"):
        VA.cmd_pilot(a)
    (out / "pilot.json").write_text(json.dumps({**NO_BETA, "record": "other"}))
    with pytest.raises(VA.CheckFailed, match="is not VG-1's pilot record"):
        VA.cmd_pilot(a)
    # with a null first record it goes on to the second pilot's own manifest
    (out / "pilot.json").write_text(json.dumps(NO_BETA))
    with pytest.raises(VA.CheckFailed, match="manifest-pilot2.json not found"):
        VA.cmd_pilot(a)


def plant_pilot2_inputs(VA, root: Path, monkeypatch, vals: dict) -> None:
    """The second pilot's cmd_pilot inputs in a real repo at ``root`` (pilot.json committed,
    beta_star null): its manifest, C1 record, M0 records with sidecars and metrics.csv rows.
    ``vals`` is {run: (m1, m0, ev)}. The code-at-commit checks (git trees) are stubbed."""
    monkeypatch.setattr(VA, "ROOT", root)
    monkeypatch.setattr(VA, "code_at", lambda head, commit, paths=VA.CODE_PATHS: head == "good")
    monkeypatch.setattr(VA, "blob_sha256", lambda rev, rel: "v")
    commit = "c" * 40
    out = root / "outputs" / "vg1"
    (out / "m0").mkdir(parents=True, exist_ok=True)
    (out / "c1").mkdir(parents=True, exist_ok=True)
    ents, nets = [], {}
    for k, (name, (m1, m0, ev)) in enumerate(vals.items()):
        plant_run(root, name, ev=ev)
        final = {"path": f"runs/{name}/ckpt/000199884800.pkl", "sha256": f"s-{name}",
                 "timesteps": VA.FINAL_TIMESTEPS}
        ents.append({"name": name, "seed": int(name[-2:]), "arm": "pilot2",
                     "beta": float(name.split("-")[1][1:]), "status": "done", "failure": None,
                     "final": final, "checkpoints": [final], "config_sha256": None,
                     "metrics_sha256": VA.sha256_file(root / "runs" / name / "metrics.csv"),
                     "eval_rdev_sha256": None,
                     "box": {"box": "vg1p2-manifest-0of1.json", "lane": k, "gpu": 0,
                             "start": "2026-10-28T10:00:00Z", "end": "2026-10-28T10:15:00Z",
                             "gpu_count": 1}})
        nets[name] = {**c1rec(m1), "checkpoint": {"sha256": final["sha256"]}}
        rec = {**m0rec(m0), "run": name, "checkpoint": "000199884800.pkl",
               "checkpoint_sha256": final["sha256"], "sizes": dict(VA.M0_SIZES),
               "vg": dict(VA.M0_VG), "environment": {"head": "good"}}
        (out / "m0" / f"{name}.json").write_text(json.dumps(rec))
        (out / "m0" / f"{name}.check.json").write_text(json.dumps(
            {"vg1_code_clean": True, "m0_sha256": VA.sha256_file(out / "m0" / f"{name}.json"),
             "head": "good", "commit": commit}))
    (out / "manifest-pilot2.json").write_text(json.dumps(
        {"record": "vg1-manifest", "phase": "pilot2", "round": "vg1p2", "commit": commit,
         "runs": ents}))
    (out / "c1" / "pilot2.json").write_text(json.dumps(
        {"record": "vg1-c1", "phase": "pilot2", "partial": False, "gate": {"pass": True},
         "manifest_sha256": VA.sha256_file(out / "manifest-pilot2.json"),
         "provenance": {"head": "good", "changed": [], "script_sha256": "v"}, "networks": nets}))


def pilot2_vals(ev=0.93, m0=0.7) -> dict:
    m1 = {"0.01": -0.6, "0.03": -0.3}
    return {f"vg1p2-b{b}-s{s}": (m1[b] + (s - 15) * 0.02, m0, ev)
            for s in (15, 16) for b in ("0.01", "0.03")}


@needs_git
def test_cmd_pilot2_writes_the_record_and_main_reads_it_back(VA, VR, tmp_path, monkeypatch):
    root = pilot_repo(tmp_path, None)                       # pilot.json committed, beta null
    plant_pilot2_inputs(VA, root, monkeypatch, pilot2_vals())
    VA.cmd_pilot(argparse.Namespace(phase="pilot2", box_gpu_hours=0.30, main_gpus=4))
    path = root / "outputs" / "vg1" / "pilot2.json"
    rec = json.loads(path.read_text())
    assert rec["record"] == "vg1-pilot" and rec["phase"] == "pilot2"
    assert rec["beta_star"] == 0.03 and rec["beta_star_text"] == "0.03"
    assert rec["eligible"] == {"0.01": True, "0.03": True}
    assert rec["rule"]["betas"] == ["0.01", "0.03"] and rec["pilot_commit"] == "c" * 40
    assert set(rec["runs"]) == set(pilot2_vals())
    assert rec["median_m1"]["0.03"] == pytest.approx(-0.29)
    cost = rec["cost_check"]
    assert cost["first_pilot_gpu_hours"] == 1.50 and cost["this_pilot_gpu_hours"] == 0.30
    assert cost["pilot_gpu_hours"] == pytest.approx(1.80)
    assert cost["beta_star_gpu_minutes"] == pytest.approx(3.75)     # 4 runs at once for 15 min
    main = (20 * 6.5 + 20 * 3.75 + 15 * 4) / 60
    assert cost["total"] == pytest.approx(1.80 + main) and cost["to_the_user"] is False
    # the first pilot's record is bound in the inputs
    assert "outputs/vg1/pilot.json" in rec["provenance"]["inputs"]
    # write-once: a second run refuses, and so does one after the record is committed
    with pytest.raises(VA.CheckFailed, match="runs once"):
        VA.cmd_pilot(argparse.Namespace(phase="pilot2", box_gpu_hours=0.30, main_gpus=4))
    _git(root, "add", "-f", "outputs/vg1/pilot2.json")
    _git(root, "commit", "-qm", "pilot2")
    path.unlink()
    with pytest.raises(VA.CheckFailed, match="runs once"):
        VA.cmd_pilot(argparse.Namespace(phase="pilot2", box_gpu_hours=0.30, main_gpus=4))
    _git(root, "checkout", "--", "outputs/vg1/pilot2.json")
    assert VR.beta_star(root) == "0.03"
    assert next(x["cmd"][-2:] for x in VR.runs("main", root) if x["arm"] == "vgl") == [
        "--vg-coef", "0.03"]


@needs_git
def test_cmd_pilot2_with_every_run_below_the_ev_bar_has_no_beta(VA, VR, tmp_path, monkeypatch):
    root = pilot_repo(tmp_path, None)
    plant_pilot2_inputs(VA, root, monkeypatch, pilot2_vals(ev=0.85))
    VA.cmd_pilot(argparse.Namespace(phase="pilot2", box_gpu_hours=0.30, main_gpus=4))
    rec = json.loads((root / "outputs" / "vg1" / "pilot2.json").read_text())
    assert rec["phase"] == "pilot2" and rec["beta_star"] is None
    assert rec["eligible"] == {"0.01": False, "0.03": False}
    assert all("EV 0.8500" in w for ws in rec["ineligible_because"].values() for w in ws[:1])
    assert rec["cost_check"]["applies"] is False
    _git(root, "add", "-f", "outputs/vg1/pilot2.json")
    _git(root, "commit", "-qm", "pilot2")
    with pytest.raises(SystemExit, match="no beta is eligible"):
        VR.beta_star(root)
    with pytest.raises(SystemExit, match="no beta is eligible"):
        VR.runs("main", root)


def test_main_paths_need_both_pilot_records_committed(VA, root, monkeypatch):
    out = root / "outputs" / "vg1"
    out.mkdir(parents=True)
    (out / "pilot.json").write_text(json.dumps(NO_BETA))
    (out / "pilot2.json").write_text(json.dumps(pilot2_record(0.03)))
    monkeypatch.setattr(VA, "git_run", FakeGit(diff_clean=False))               # not tracked
    with pytest.raises(VA.CheckFailed, match="is not committed"):
        VA.main_pilot_records({})
    monkeypatch.setattr(VA, "git_run", FakeGit())
    inputs = {}
    first, second = VA.main_pilot_records(inputs)
    assert first["beta_star"] is None and second["beta_star"] == 0.03
    assert set(inputs) == {"outputs/vg1/pilot.json", "outputs/vg1/pilot2.json"}
