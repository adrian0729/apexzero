"""VG-1's run lists, C1 script and analysis (docs/prereg/vg1-value-gradient.md), on planted data.

The run lists give the draft's names and flags in the pre-registered order, pass the run-list
contract, and deal the main run's arms evenly over GPUs, lanes and boxes; the main list reads
beta* only from a committed, eligible pilot record. scripts/vg1_c1.py's copied functions are
the pinned costate-exploratory blob's byte for byte, and (where the worktree and the costate
data are on this machine) its per-network path reproduces a decision network's C1 from
decisions.json. scripts/vg1_analysis.py's exact rank-sum and sign-flip tests give known p
values, the failed-network rule puts the worst value in each test's direction, the pilot rule
takes each branch (no beta eligible included), the manifest refuses every planted leak or
mismatch, and the tag gate and run-once rules refuse what they must."""

import importlib.util
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
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


def pilot_repo(tmp_path: Path, rec: dict | None, commit: bool = True) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "README").write_text("x\n")
    _git(root, "add", "README")
    _git(root, "commit", "-qm", "init")
    if rec is not None:
        p = root / "outputs" / "vg1" / "pilot.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(rec))
        if commit:
            _git(root, "add", "-f", "outputs/vg1/pilot.json")
            _git(root, "commit", "-qm", "pilot")
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


def test_runs_parse_with_train_dr_and_record_the_flags(VR, RM):
    """Each cmd parses with train_dr.py's own parser (rebuilt as round_manifest.py check does):
    --skip-eval-rdev on in the pilot, off in the main run, --vg-coef only where it belongs."""
    parser = RM.trainer_parser((SCRIPTS / "train_dr.py").read_text())
    for r in VR.runs("pilot"):
        args = vars(parser.parse_args(r["cmd"][1:]))
        assert args["skip_eval_rdev"] is True and args["vg_coef"] == r["beta"]
        assert args["steps"] == 200e6 and args["seed"] == r["seed"] and args["name"] == r["name"]
        assert RM.compare_train_dr({"args": dict(args)}, args)[0] == []
    args = vars(parser.parse_args(["--steps", "200e6", "--seed", "30", "--name", "vg1-base-s30"]))
    assert args["skip_eval_rdev"] is False and args["vg_coef"] == 0.0


@needs_git
def test_main_runs_order_names_flags_and_dealing(VR, RM, tmp_path, monkeypatch):
    root = pilot_repo(tmp_path, pilot_record(0.3))
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
        assert r["cmd"] == (base if r["arm"] == "base" else base + ["--vg-coef", "0.3"])
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

    check(runs, 4, 4)                               # one 4-GPU box
    for k in (2, 4, 5):                             # K single-GPU boxes, whole seeds
        shards = [RM.in_shard(runs, i, k) for i in range(k)]
        assert sum(map(len, shards)) == 40
        for jobs in shards:
            check(jobs, 1, 4)


@needs_git
@pytest.mark.parametrize("rec, commit, msg", [
    (None, True, "not found"),
    (pilot_record(0.3), False, "not committed"),
    (pilot_record(None), True, "no beta is eligible"),
    (pilot_record(0.5), True, "not in the grid"),
    (pilot_record(0.3, {"0.1": True, "0.3": False, "1.0": True}), True, "not marked eligible"),
    ({**pilot_record(0.3), "record": "other"}, True, "not VG-1's pilot record"),
])
def test_main_runs_refuse_without_a_committed_eligible_beta(VR, tmp_path, rec, commit, msg):
    root = pilot_repo(tmp_path, rec, commit)
    with pytest.raises(SystemExit, match=msg):
        VR.runs("main", root)


@needs_git
def test_main_runs_refuse_a_pilot_record_changed_since_commit(VR, tmp_path):
    root = pilot_repo(tmp_path, pilot_record(0.3))
    (root / "outputs" / "vg1" / "pilot.json").write_text(json.dumps(pilot_record(1.0)))
    with pytest.raises(SystemExit, match="differs from HEAD"):
        VR.runs("main", root)


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
                         "m0": 9.0, "ev": 0.93, "gpu_minutes": {"0.1": 20.0, "0.3": 22.0,
                                                                 "1.0": 24.0}[b]}
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


def test_cost_check_in_gpu_hours(VA):
    per = pilot(VA)
    c = VA.cost_check("0.3", per, 2.5, 4)
    main = (20 * 6.5 + 20 * 22.0 + 15 * 4) / 60
    assert c["main_projected_gpu_hours"] == pytest.approx(main)
    assert c["total"] == pytest.approx(2.5 + main) and c["to_the_user"] is False
    per = pilot(VA, **{f"vg1p-b0.3-s{s}": {"gpu_minutes": 33.0} for s in (15, 16)})
    c = VA.cost_check("0.3", per, 2.5, 4)
    assert c["total"] > 14 and c["to_the_user"] is True and c["stop_rule"] == 16.8


@needs_git
def test_the_pilot_record_feeds_the_main_run_list(VA, VR, tmp_path):
    """The rule's beta* and eligibility, as pilot.json holds them, are what vg1_runs reads."""
    r = VA.pilot_rule(pilot(VA))
    root = pilot_repo(tmp_path, json.loads(json.dumps(VA.clean({"record": "vg1-pilot", **r}))))
    vgl = [x for x in VR.runs("main", root) if x["arm"] == "vgl"]
    assert len(vgl) == 20 and all(x["cmd"][-2:] == ["--vg-coef", "0.3"] for x in vgl)


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


# ------------------------------------------------------------------------- the manifest


def plant_pilot_round(VA, VR, root: Path) -> tuple[list, Path]:
    runs = VR.runs("pilot")
    recs = []
    for k, r in enumerate(runs):
        d = root / "runs" / r["name"]
        (d / "ckpt").mkdir(parents=True)
        for ts in (10_000_000, 199_884_800):
            (d / "ckpt" / f"{ts:012d}.pkl").write_bytes(os.urandom(64))
        (d / "config.json").write_text(json.dumps({
            "args": {"seed": r["seed"], "skip_eval_rdev": True, "vg_coef": r["beta"]},
            "vg": {"coef": r["beta"], "lam": 0.95, "winsor": 0.005}}))
        (d / "metrics.csv").write_text("explained_var\n0.93\n")
        (root / "runs" / f"{r['name']}.log").write_text("--skip-eval-rdev: no R-dev evaluation\n")
        recs.append({"name": r["name"], "seed": r["seed"], "status": "done", "lane": k % 3,
                     "gpu": 0, "start": "2026-10-28T10:00:00Z", "end": "2026-10-28T11:00:00Z"})
    (root / "runs" / "vg1p-manifest-0of1.json").write_text(json.dumps(
        {"runs": recs, "gpu_model": "NVIDIA RTX A6000", "gpu_count": 1,
         "box_type": "massedcompute_A6000_plus"}))
    sums = root / "sums.txt"
    lines = [f"{VA.sha256_file(p)}  {p.relative_to(root).as_posix()}"
             for p in sorted((root / "runs").glob("vg1p-*/ckpt/*.pkl"))]
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
        {"args": {"seed": 16, "skip_eval_rdev": False}, "vg": {"coef": 0.1, "lam": 0.95,
                                                               "winsor": 0.005}})),
     "--skip-eval-rdev is not in its config"),
    (lambda r: (r / "runs" / "vg1p-b0.1-s16" / "config.json").write_text(json.dumps(
        {"args": {"seed": 16, "skip_eval_rdev": True}, "vg": {"coef": 0.3, "lam": 0.95,
                                                              "winsor": 0.005}})),
     "config vg"),
    (lambda r: (r / "runs" / "vg1p-b0.1-s16" / "ckpt" / "000199884800.pkl").unlink(),
     "no final checkpoint"),
])
def test_manifest_refuses_leaks_and_mismatches(VA, VR, root, monkeypatch, plant, msg):
    monkeypatch.setattr(VA, "git_run", StubGit())
    runs, sums = plant_pilot_round(VA, VR, root)
    plant(root)
    with pytest.raises(VA.CheckFailed, match=msg):
        VA.build_manifest("pilot", "c" * 40, VA.box_sums([str(sums)]), runs, {}, CHECK, None)


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
    good = {"args": {"seed": 30, "skip_eval_rdev": False, "vg_coef": 0.3},
            "vg": {"coef": 0.3, "lam": 0.95, "winsor": 0.005}}
    assert VA.config_problems(run, good, "main", "0.3") == []
    assert VA.config_problems(run, {**good, "vg": {**good["vg"], "coef": 1.0}}, "main", "0.3")
    assert VA.config_problems(run, {**good, "vg": {**good["vg"], "lam": 0.9}}, "main", "0.3")
    base = {"name": "vg1-base-s30", "seed": 30, "arm": "base"}
    ok = {"args": {"seed": 30, "skip_eval_rdev": False, "vg_coef": 0.0}}
    assert VA.config_problems(base, ok, "main", "0.3") == []
    assert VA.config_problems(base, {**ok, "vg": good["vg"]}, "main", "0.3")
    assert VA.config_problems(base, {"args": {**ok["args"], "skip_eval_rdev": True}}, "main",
                              "0.3")


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
    assert VA.check_m0(m0, e, "c") == []
    assert VA.check_m0({**m0, "checkpoint_sha256": "t"}, e, "c")
    assert VA.check_m0({**m0, "sizes": {**VA.M0_SIZES, "seed": 1}}, e, "c")
    assert VA.check_m0({**m0, "environment": {"head": "good", "src_or_scripts_dirty": True}},
                       e, "c")
    assert VA.check_m0({**m0, "environment": {"head": "bad", "src_or_scripts_dirty": False}},
                       e, "c")
    # other studies' untracked files mark vgl.py's record dirty: m0's sidecar vouches instead
    dirty = {**m0, "environment": {"head": "good", "src_or_scripts_dirty": True}}
    side = {"vg1_code_clean": True, "m0_sha256": "h", "head": "good"}
    assert VA.check_m0(dirty, e, "c", side, "h") == []
    assert VA.check_m0(dirty, e, "c", side, "other")              # not this record's sidecar
    assert VA.check_m0(dirty, e, "c", {**side, "head": "bad"}, "h")
    assert VA.check_m0(dirty, e, "c", {**side, "vg1_code_clean": False}, "h")
    man = {"_sha256": "m", "runs": [e]}
    c1 = {"manifest_sha256": "m", "partial": False, "gate": {"pass": True},
          "provenance": {"head": "good", "changed": [" M docs/x.md"]},
          "networks": {"vg1p-b0.3-s15": {"checkpoint": {"sha256": "s"}}}}
    assert VA.check_c1(c1, man, "c") == []
    assert VA.check_c1({**c1, "partial": True}, man, "c")
    assert VA.check_c1({**c1, "gate": {"pass": False}}, man, "c")
    assert VA.check_c1({**c1, "provenance": {"head": "good", "changed": [" M scripts/vgl.py"]}},
                       man, "c")
    assert VA.check_c1({**c1, "networks": {}}, man, "c")


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
        VA.cmd_pilot(None)
