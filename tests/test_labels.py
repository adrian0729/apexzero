"""Paper labeller (PA-1): levels as the consumers draw them, the three starts, the retry
policy, verification, resumable records and the two exports, on a 1 s window (the solves
are the TO-OCP's, tested in test_window_ocp.py)."""

import dataclasses
import importlib.util
import json
import pickle
import shutil
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.ocp import labels as L
from apexzero.ocp.replay import replay_window
from apexzero.ocp.window import min_time_guess
from apexzero.tracks import levels as paper_levels
from apexzero.ued import pool

ROOT = Path(__file__).resolve().parents[1]
TUMFTM = ROOT / "data" / "tracks" / "tumftm"
SCRIPTS = ROOT / "scripts"
pytestmark = pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")

# A 1 s window (20 steps) in two stages: the labeller's whole path at a fraction of the cost.
TINY = L.LabelPolicy(seconds=1.0, stages_s=(0.5,), retry_stages_s=(0.25, 0.5, 0.75), agent_windows=2)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    """Labels are float64 (guesses, solves, replays); switched on for this module only, since
    other modules' float32 guards need it off."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def grammar():
    return L.r_dev_grammar()


@pytest.fixture(scope="module")
def level(grammar):
    return L.make_paper_levels("a_dr", 2026, [0], grammar)[0]


@pytest.fixture(scope="module")
def solvers():
    return L.SolverCache(compile=False)


@pytest.fixture(scope="module")
def socp_pickle(level, tmp_path_factory):
    """The level's s-OCP for TINY, solved once and seeded into each test's work cache."""
    path = tmp_path_factory.mktemp("socp") / "socp.pkl"
    path.write_bytes(pickle.dumps(L.solve_socp(level, TINY)))
    return path


def work_dir(tmp_path, socp_pickle, lid):
    work = tmp_path / "work" / lid
    work.mkdir(parents=True)
    shutil.copy(socp_pickle, work / "socp.pkl")
    return work


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    """A checkpoint of an untrained actor in the ``agents.checkpoints`` format."""
    from apexzero.agents.ppo import Actor, PPOConfig, RunningMeanStd
    from apexzero.env import racing

    cfg = PPOConfig()
    with jax.enable_x64(False):
        params = Actor(cfg, racing.ACTION_DIM, False).init(jax.random.PRNGKey(0), jnp.zeros(racing.OBS_DIM))
        rms = RunningMeanStd.create((racing.OBS_DIM,))
    d = {"actor": jax.tree.map(np.asarray, params), "critic": None,
         "obs_rms": {k: np.asarray(v) for k, v in rms._asdict().items()}, "ret_rms": None,
         "config": dataclasses.asdict(cfg), "timesteps": 0}
    path = tmp_path_factory.mktemp("agent") / "000000000000.pkl"
    path.write_bytes(pickle.dumps(d))
    return path


# ----------------------------------------------------------------------------- levels


def test_paper_levels_are_the_consumers_float32_draws(level, grammar):
    """train_ued.py and pilot_audit.py draw their levels with x64 off; a label must be made on
    those draws even when the labeller runs with x64 on (x64 draws another car)."""
    from apexzero.tracks.generator import to_track
    from apexzero.tracks.levels import build, make_levels

    with jax.enable_x64(False):
        lv = make_levels(jax.random.PRNGKey(2026), "a_dr", n=1)
        tr, _, x0 = build(lv, grammar)
        ref = to_track(jax.tree.map(lambda a: a[0], tr))
        ctx32, mu32, keys = np.asarray(lv.ctx[0]), np.asarray(lv.hidden_mu[0]), np.asarray(lv.keys[0])
        x032 = np.asarray(x0[0])
    assert ctx32.dtype == np.float32
    assert np.array_equal(level.keys, keys)
    assert np.array_equal(level.ctx, ctx32.astype(np.float64)) and level.hidden_mu == float(mu32)
    assert np.array_equal(level.x0, x032.astype(np.float64))
    assert np.array_equal(level.track.kappa, np.asarray(ref.kappa, np.float64))
    assert np.array_equal(level.track.w_left, np.asarray(ref.w_left, np.float64))
    assert level.track.kappa.dtype == np.float64
    x64 = make_levels(jax.random.PRNGKey(2026), "a_dr", n=1)
    assert not np.allclose(np.asarray(x64.ctx[0]), ctx32, rtol=1e-6)   # why x64 is switched off
    enc = level.encoding()
    assert enc["id"] == "a_dr-r2026-00000" and enc["strip_sha256"] == L.strip_sha256(level.track)


def test_a_level_does_not_depend_on_which_others_are_built(level, grammar):
    two = L.make_paper_levels("a_dr", 2026, [2, 0], grammar)
    assert [lv.index for lv in two] == [0, 2]
    assert two[0].encoding() == level.encoding()
    with pytest.raises(ValueError):
        L.make_paper_levels("a_dr", 2026, [-1], grammar)


def test_shards_are_disjoint_and_cover_every_index():
    idx = list(range(17))
    parts = [L.select_shard(idx, (i, 4)) for i in range(4)]
    assert sorted(i for part in parts for i in part) == idx
    assert all(not set(a) & set(b) for a in parts for b in parts if a is not b)
    with pytest.raises(ValueError):
        L.select_shard(idx, (4, 4))


def test_policy_schedules():
    p = L.LabelPolicy()
    assert p.horizons() == [200, 400, 800, 1200]
    assert p.horizons(retry=True) == [200, 400, 600, 800, 1000, 1200]
    assert TINY.horizons() == [10, 20] and TINY.horizons(retry=True) == [5, 10, 15, 20]
    json.dumps(p.to_json())


# ----------------------------------------------------------------------------- guesses


def test_qss_line_is_a_guess_inside_the_corridor(level):
    margin = 1.03
    line = L.qss_line(level.track, level.x0, level.p, margin)
    n = line.z[:, 0]
    assert abs(n[0] - level.x0[1]) <= 1e-3
    assert np.all(n <= level.track.w_left - margin + 1e-3) and np.all(n >= -(level.track.w_right - margin) - 1e-3)
    assert line.z[0, 2] == pytest.approx(level.x0[3])
    assert np.all(np.diff(line.t) > 0) and np.all(np.abs(line.u) <= 1.0)
    assert np.array_equal(line.u[:, 0] * level.p.delta_max, line.z[:, 5])
    xs, acts = min_time_guess(line, level.track, level.x0, 1200)
    assert np.array_equal(xs[0], level.x0) and np.all(np.diff(xs[:, 0]) > 0)
    assert xs.shape == (1201, 8) and acts.shape == (1200, 2)


def test_jitter_is_seeded_and_keeps_x0_and_the_action_box(level):
    xs = np.tile(level.x0, (21, 1))
    acts = np.full((20, 2), 0.99)
    a = L.jitter_guess((xs, acts), (1, 2, 3), (0.05, 0.005, 0.02))
    b = L.jitter_guess((xs, acts), (1, 2, 3), (0.05, 0.005, 0.02))
    c = L.jitter_guess((xs, acts), (1, 2, 4), (0.05, 0.005, 0.02))
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    assert not np.array_equal(a[1], c[1])
    assert np.array_equal(a[0][0], level.x0) and np.all(np.abs(a[1]) <= 1.0)


def test_splice_keeps_the_prefix_and_extends_from_its_last_state():
    xs = np.arange(11 * 8, dtype=float).reshape(11, 8)
    acts = np.arange(20, dtype=float).reshape(10, 2)

    def extend(state, n):
        return np.tile(state, (n + 1, 1)) + 0.5, np.full((n, 2), -1.0)

    sx, sa = L.splice(xs, acts, 4, 10, extend)
    assert np.array_equal(sx[:5], xs[:5]) and np.array_equal(sa[:4], acts[:4])
    assert np.array_equal(sx[5], xs[4] + 0.5) and np.all(sa[4:] == -1.0)
    assert sx.shape == (11, 8) and sa.shape == (10, 2)
    full = L.splice(xs, acts, 10, 10, extend)
    assert np.array_equal(full[0], xs) and np.array_equal(full[1], acts)


def test_code_identity_stamps_the_commit_and_says_sim_v1_is_missing():
    from apexzero.tracks.corners import R_DEV, R_TEST

    code = L.code_identity()
    assert len(code["sha256"]) == 64
    assert "src/apexzero/ocp/labels.py" in code["files"] and "src/apexzero/ocp/window.py" in code["files"]
    assert all(not Path(f).is_absolute() for f in [*code["files"], *code["data"]])
    assert sorted(Path(f).stem for f in code["data"]) == sorted(R_DEV)
    assert not {Path(f).stem for f in code["data"]} & set(R_TEST)
    if code["sim_v1"] is None:
        assert "sim-v1 is not tagged" in code["sim_v1_note"]
    assert code["git_commit"] is None or len(code["git_commit"]) == 40


# ----------------------------------------------------------------------------- a level


def test_a_level_is_labelled_from_three_verified_starts(level, solvers, agent, socp_pickle, tmp_path):
    work = work_dir(tmp_path, socp_pickle, level.id)
    ag = L.AgentStart(agent, windows=TINY.agent_windows)
    lab = L.label_level(level, TINY, solvers, ag, work=work, code={"sha256": "x"})
    rec = lab.record
    json.dumps(rec)
    assert rec["schema"] == L.SCHEMA and rec["level"] == json.loads(json.dumps(level.encoding()))
    assert [o["name"] for o in rec["starts"]] == ["socp", "qss", "agent"]
    assert {o["round"] for o in rec["starts"]} == {"primary"}
    lab_ = rec["label"]
    assert lab_["status"] == "verified" and not lab_["retried"]
    verified = [o for o in rec["starts"] if o["verified"]]
    assert lab_["j_star"] == max(o["value"] for o in verified)
    assert lab_["n_verified_primary"] == len(verified) >= 2
    # The stored label is the solve's optimum, and its controls replay to J*.
    ret, alive, _ = replay_window(level.track, level.x0, lab.arrays["label_a"], ctx=level.ctx,
                                  hidden_mu=level.hidden_mu)
    assert alive and abs(ret - lab_["j_star"]) <= TINY.replay_rtol * abs(lab_["j_star"])
    assert lab.arrays["label_x"].shape == (21, 8) and lab.arrays["label_a"].shape == (20, 2)
    assert rec["agent"]["sha256"] and len(rec["agent"]["rollouts"]["returns"]) == TINY.agent_windows + 1
    cpu = rec["cpu"]
    assert cpu["level_s"] == pytest.approx(cpu["socp_s"] + cpu["guesses_s"] + cpu["solves_s"] + cpu["replays_s"])
    assert cpu["cached"] == ["socp.pkl"]
    # Every finished start is cached: a second call (a resumed run) solves nothing again.
    again = L.label_level(level, TINY, solvers, ag, work=work, code={"sha256": "x"})
    assert again.record["label"]["j_star"] == lab_["j_star"]
    assert sorted(again.record["cpu"]["cached"]) == sorted(
        ["socp.pkl"] + [f"start-primary-{n}.pkl" for n in L.STARTS])
    assert again.record["cpu"]["level_s"] == pytest.approx(cpu["level_s"], rel=0.5)


def test_retry_runs_only_when_no_start_verifies_and_stops_at_the_first_that_does(level, solvers, socp_pickle,
                                                                                  tmp_path):
    policy = dataclasses.replace(TINY, max_iter=1)          # every primary start fails
    lab = L.label_level(level, policy, solvers, None, starts=("socp", "qss"),
                        work=work_dir(tmp_path, socp_pickle, level.id))
    rec = lab.record
    assert [(o["round"], o["name"], o["verified"]) for o in rec["starts"]] == [
        ("primary", "socp", False), ("primary", "qss", False), ("retry", "socp", True)]
    assert rec["label"]["status"] == "verified" and rec["label"]["retried"]
    assert rec["label"]["start"] == "retry-socp" and rec["label"]["n_verified_primary"] == 0
    assert [s["seconds"] for s in rec["starts"][2]["stages"]] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert rec["cpu"]["retry_s"] > 0


def test_a_level_no_start_verifies_is_unsolved_and_cannot_be_exported(level, solvers, socp_pickle, tmp_path):
    policy = dataclasses.replace(TINY, max_iter=1, retry_max_iter=1)
    lab = L.label_level(level, policy, solvers, None, starts=("socp",),
                        work=work_dir(tmp_path, socp_pickle, level.id))
    assert lab.record["label"]["status"] == "unsolved" and lab.record["label"]["j_star"] is None
    assert "label_a" not in lab.arrays
    L.write_record(tmp_path, lab)
    assert L.is_done(L.load_record(tmp_path, level.id))
    assert not (tmp_path / "work" / level.id).exists()            # an outcome clears the cache
    with pytest.raises(L.LabelError, match="unsolved"):
        L.export_j_opt(tmp_path, tmp_path / "j.npy", 1)


def test_a_start_past_its_cpu_budget_stops_and_counts_as_failed(level, solvers, socp_pickle, tmp_path):
    policy = dataclasses.replace(TINY, start_cpu_budget_s=1e-9, retry=False)
    lab = L.label_level(level, policy, solvers, None, starts=("qss",),
                        work=work_dir(tmp_path, socp_pickle, level.id))
    (o,) = lab.record["starts"]
    assert o["capped"] and not o["verified"] and len(o["stages"]) == 1
    assert lab.record["label"]["status"] == "unsolved"


def test_an_error_keeps_the_cache_and_is_labelled_again(level, socp_pickle, tmp_path):
    work = work_dir(tmp_path, socp_pickle, level.id)
    err = L.error_record(level, RuntimeError("boom"), TINY, None, 1.0)
    L.write_record(tmp_path, err)
    rec = L.load_record(tmp_path, level.id)
    assert rec["label"]["status"] == "error" and rec["error"]["message"] == "boom"
    assert not L.is_done(rec) and (work / "socp.pkl").exists()
    assert L.summarise([rec])["error"] == 1


def test_label_level_refuses_float32_and_a_missing_checkpoint(level, solvers):
    with pytest.raises(ValueError, match="checkpoint"):
        L.label_level(level, TINY, solvers, None, starts=("agent",))
    with pytest.raises(ValueError, match="unknown starts"):
        L.label_level(level, TINY, solvers, None, starts=("policy",))
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="x64"):
            L.label_level(level, TINY, solvers, None, starts=("qss",))
    finally:
        jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------------- exports


def fake_records(out: Path, stream: str, root: int, n: int, code: str = "c0", skip=(), js=None):
    """Verified records for levels 0..n-1 of a stream (encodings as the labeller stores them)."""
    lv = L.float32_levels(stream, root, n)
    keys, ctx, mu = (np.asarray(getattr(lv, k)) for k in ("keys", "ctx", "hidden_mu"))
    js = np.linspace(60.0, 70.0, n) if js is None else js
    for i in range(n):
        if i in skip:
            continue
        rec = {"schema": L.SCHEMA, "level": {"id": L.level_id(stream, root, i), "stream": stream, "root": root,
                                             "index": i, "keys": [int(k) for k in keys[i]],
                                             "ctx": [float(v) for v in ctx[i]], "hidden_mu": float(mu[i])},
               "policy": TINY.to_json(), "label": {"status": "verified", "j_star": float(js[i]), "retried": False},
               "code": {"sha256": code, "git_commit": "0" * 40, "sim_v1": None}, "agent": None}
        L.write_record(out, L.LevelLabel(record=rec, arrays={}))
    return lv, js


def test_export_j_opt_is_what_pilot_audit_reads(tmp_path):
    _, js = fake_records(tmp_path, "a_dr", 2026, 3)
    out = L.export_j_opt(tmp_path, tmp_path / "j_opt.npy", 3)
    j = np.load(tmp_path / "j_opt.npy")                  # pilot_audit: np.load, shape (--levels,)
    assert j.shape == (3,) and j.dtype == np.float64 and np.array_equal(j, js) and np.array_equal(out, js)
    prov = json.loads((tmp_path / "j_opt.npy.json").read_text())
    assert prov["levels"] == 3 and prov["code_sha256"] == ["c0"] and prov["consumer"] == "pilot_audit --j-opt"
    with pytest.raises(L.LabelError, match="no record"):
        L.export_j_opt(tmp_path, tmp_path / "j4.npy", 4)


def test_exports_refuse_another_level_and_mixed_code(tmp_path):
    fake_records(tmp_path, "a_dr", 2026, 2)
    rec = L.load_record(tmp_path, "a_dr-r2026-00001")
    rec["level"]["hidden_mu"] += 1e-3
    L.write_record(tmp_path, L.LevelLabel(record=rec, arrays={}))
    with pytest.raises(L.LabelError, match="differs"):
        L.export_j_opt(tmp_path, tmp_path / "j.npy", 2)
    fake_records(tmp_path, "a_dr", 2026, 2)
    rec = L.load_record(tmp_path, "a_dr-r2026-00001")
    rec["code"]["sha256"] = "c1"
    L.write_record(tmp_path, L.LevelLabel(record=rec, arrays={}))
    with pytest.raises(L.LabelError, match="code versions"):
        L.export_j_opt(tmp_path, tmp_path / "j.npy", 2)
    assert L.export_j_opt(tmp_path, tmp_path / "j.npy", 2, allow_mixed_code=True).shape == (2,)


def test_export_oracle_is_what_train_ued_loads(tmp_path):
    with jax.enable_x64(False):
        lv = paper_levels.make_levels(jax.random.PRNGKey(5), "pool", n=3)
        paper_levels.save(lv, tmp_path / "pool.npz")
    _, js = fake_records(tmp_path, "pool", 5, 3)
    L.export_oracle(tmp_path, tmp_path / "pool_jstar.npz", paper_levels.load(tmp_path / "pool.npz"), root=5)
    with jax.enable_x64(False):                          # train_ued: load the pool, then the labels
        got = pool.load_oracle_labels(tmp_path / "pool_jstar.npz", paper_levels.load(tmp_path / "pool.npz"))
    assert np.array_equal(got, js)
    with pytest.raises(L.LabelError, match="differs"):    # the labels are pool root 5's, not root 6's
        L.export_oracle(tmp_path, tmp_path / "x.npz", L.float32_levels("pool", 6, 3), root=5)


# ----------------------------------------------------------------------------- the script


@pytest.fixture(scope="module")
def script():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    prev = jax.config.jax_enable_x64
    spec = importlib.util.spec_from_file_location("label_levels", SCRIPTS / "label_levels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    jax.config.update("jax_enable_x64", prev)           # the module fixture above restores it
    return mod


def test_script_labels_resumes_pins_its_set_and_exports(script, tmp_path, capsys):
    out = tmp_path / "labels"
    run = ["run", "--levels", "0", "--out", str(out), "--seconds", "1", "--stages", "0.5",
           "--starts", "socp", "qss", "--no-compile"]
    script.main(run)
    rec = L.load_record(out, "a_dr-r2026-00000")
    assert rec["label"]["status"] == "verified" and rec["code"]["sha256"] == L.code_identity()["sha256"]
    assert not (out / "work" / "a_dr-r2026-00000").exists()
    pinned = json.loads((out / "run.json").read_text())
    assert pinned["stream"] == "a_dr" and pinned["root"] == 2026 and pinned["policy"]["seconds"] == 1.0
    capsys.readouterr()
    script.main(run)                                     # done: nothing to label
    assert "1 done, 0 to label" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="policy"):
        script.main([*run[:-1], "--max-iter", "50", "--no-compile"])
    script.main(["status", "--out", str(out)])
    assert "a_dr-r2026-00000 | verified" in capsys.readouterr().out
    script.main(["export-j-opt", "--out", str(out), "--to", str(tmp_path / "j.npy"), "--n", "1"])
    assert np.load(tmp_path / "j.npy")[0] == rec["label"]["j_star"]
    with pytest.raises(SystemExit, match="not exported"):
        script.main(["export-j-opt", "--out", str(out), "--to", str(tmp_path / "j2.npy"), "--n", "2"])
