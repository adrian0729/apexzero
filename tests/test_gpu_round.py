"""The GPU round queue and its manifests (scripts/gpu_round.sh, scripts/round_manifest.py and
gpu_sync.sh's pull-round), on the Mac: no GPU, no box and no real trainer.

Every end-to-end test runs in temporary git repositories holding copies of the three scripts, a
planted run list (scripts/demo_runs.py) and stand-in trainers at scripts/train_dr.py and
scripts/train.py that write their configs in the real trainers' formats in a few milliseconds.
"Boxes" are clones of the tagged repository; ssh and rsync are replaced by local stand-ins, and
nvidia-smi by one listing FAKE_GPUS A6000s where a test needs it (otherwise it is absent).

- DRY_RUN's schedule for GPUS 1, 2 and 4 at 4 lanes per GPU: every run once, job j on lane
  j mod L and lane l on GPU l mod GPUS, in run-list order.
- SHARD for K = 1, 2, 5 and 10: every run in exactly one shard, each seed's runs together,
  shard i the seeds whose first-appearance index is i mod K.
- The environment's refusals (SHARD, GPUS, LANES_PER_GPU, more GPUS than nvidia-smi lists), the
  run list's (names, seeds, --name, the trainer, config paths, lockbox circuits) and a dirty
  tracked tree's (the queue refuses it at launch; DRY_RUN notes it).
- A real queue on the Mac with GPUS=1 and no nvidia-smi: each run's CUDA_VISIBLE_DEVICES and
  memory fraction, lanes in order, a run that fails once retried from scratch (first log
  kept), one that fails twice recorded as failed, the manifest, and the refusal to start over
  existing runs.
- Two shards on two boxes, pulled back with pull-round and checked: the check passes, then
  fails on each planted fault (a missing run, a duplicate, the wrong HEAD, a dirty tree,
  arguments that differ from the cmd for each trainer format, a config that is not the box's,
  a stray run directory, a changed runs script, differing GPU models, a missing run directory)
  and on an unknown trainer's config.
- The rebuilt argument parsers of the real train_dr.py and train.py, and the lockbox rule
  against ``viz.export.is_lockbox``.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TAG = "demo-v1"
SEEDS = [27, 20, 25, 21, 29, 22, 24, 23, 28, 26]  # first-appearance order is not sorted order

RUNS_SCRIPT = '''"""Planted run list for tests/test_gpu_round.py."""

import argparse
import json

SEEDS = [27, 20, 25, 21, 29, 22, 24, 23, 28, 26]
ARMS = {"base": [], "realw": ["--real-widths"], "tilt": ["--tilt-fraction", "0.5"]}


def runs(seeds=10, fail_once="", fail_always="", specialist=False, track="Hockenheim",
         other=False):
    out = []
    for arm, extra in ARMS.items():
        for seed in SEEDS[:seeds]:
            name = f"demo-{arm}-s{seed}"
            cmd = ["scripts/train_dr.py", "--steps", "1e3", "--seed", str(seed), "--name", name]
            cmd += extra + ["--fail-once"] * (name == fail_once)
            cmd += ["--fail-always"] * (name == fail_always)
            out.append({"name": name, "seed": seed, "cmd": cmd,
                        "config": f"runs/{name}/config.json"})
    for seed in SEEDS[:seeds] if specialist else []:
        name = f"demo-spec-s{seed}"
        out.append({"name": name, "seed": seed, "config": f"runs/{name}/config.json",
                    "cmd": ["scripts/train.py", "--tracks", track, "--steps", "2e3", "--seed",
                            str(seed), "--name", name]})
    if other:
        name = "demo-other-s27"
        out.append({"name": name, "seed": 27, "config": f"runs/{name}/config.json",
                    "cmd": ["scripts/other.py", "--seed", "27", "--name", name]})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--fail-once", default="")
    ap.add_argument("--fail-always", default="")
    ap.add_argument("--specialist", action="store_true")
    ap.add_argument("--track", default="Hockenheim")
    ap.add_argument("--other", action="store_true")
    for run in runs(**vars(ap.parse_args())):
        print(json.dumps(run))
'''

TRAIN_DR = '''"""Stand-in for train_dr.py: its config format (vars(args) as "args"), no training."""

import argparse
import json
import os
import sys
from pathlib import Path

LAM = 0.5 * 1.0

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=float, default=200e6)
ap.add_argument("--name", default="dr-pilot")
ap.add_argument("--network", default="mlp", choices=["mlp", "simba"])
ap.add_argument("--real-widths", action="store_true")
ap.add_argument("--tilt-fraction", type=float, default=0.0)
ap.add_argument("--vg-lambda", type=float, default=LAM)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fail-once", action="store_true")
ap.add_argument("--fail-always", action="store_true")
args = ap.parse_args()
out = Path("runs") / args.name
if (out / "partial").exists():
    sys.exit("stale run directory")
out.mkdir(parents=True, exist_ok=True)
(out / "partial").write_text("")
if args.fail_always or (args.fail_once and not Path(f"runs/{args.name}.log.1").exists()):
    sys.exit("planted failure")
(out / "config.json").write_text(json.dumps({"ppo": {}, "args": vars(args)}, indent=1))
keys = ("CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION")
(out / "env.json").write_text(json.dumps({k: os.environ.get(k) for k in keys}))
(out / "partial").unlink()
print("trained", args.name)
'''

TRAIN = '''"""Stand-in for train.py: its config format (tracks, seed, PPOConfig), no training."""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tracks", nargs="+", default=["Hockenheim"])
ap.add_argument("--steps", type=float, default=50e6)
ap.add_argument("--name", default=None)
ap.add_argument("--network", default="mlp", choices=["mlp", "simba"])
ap.add_argument("--num-envs", type=int, default=2048)
ap.add_argument("--num-steps", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--log-every", type=int, default=10)
args = ap.parse_args()
out = Path("runs") / args.name
out.mkdir(parents=True, exist_ok=True)
ppo = {"num_envs": args.num_envs, "num_steps": args.num_steps, "lr": args.lr,
       "total_timesteps": int(args.steps), "network": args.network}
(out / "config.json").write_text(json.dumps({"ppo": ppo, "tracks": args.tracks, "env": {},
                                             "seed": args.seed}, indent=1))
'''

OTHER = '''import json, sys
from pathlib import Path
name = sys.argv[sys.argv.index("--name") + 1]
Path("runs", name).mkdir(parents=True, exist_ok=True)
Path("runs", name, "config.json").write_text(json.dumps({"argv": sys.argv[1:]}))
'''

FAKE_SSH = '''#!/bin/sh
# ssh <box> <command>: run it in the box's stand-in home
box="$1"; shift
cd "$FAKE_BOXES/$box" || exit 255
exec /bin/sh -c "$*"
'''

FAKE_RSYNC = '''#!/bin/sh
# rsync -az <box>:<path> <dir>: a local copy from the box's stand-in home
src="$2"; dest="$3"
exec cp -R "$FAKE_BOXES/${src%%:*}/${src#*:}" "$dest"
'''

FAKE_SMI = '''#!/bin/sh
i=0
while [ "$i" -lt "${FAKE_GPUS:-2}" ]; do echo "NVIDIA RTX A6000"; i=$((i + 1)); done
'''

CLEAR = ("BREV_TYPE", "GPUS", "LANES_PER_GPU", "SHARD", "DRY_RUN", "PYTHON", "FAKE_GPUS",
         "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION")
FIXTURE_ARGS = ["--seeds", "4", "--fail-once", "demo-realw-s25", "--specialist"]


def _load():
    """scripts/round_manifest.py as a module (nothing runs on import)."""
    spec = importlib.util.spec_from_file_location("round_manifest", ROOT / "scripts" /
                                                  "round_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RM = _load()


def _g(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                        "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _put(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _env(tmp: Path, gpus: int | None = None, **extra) -> dict:
    """The test environment: ssh and rsync stand-ins, nvidia-smi listing ``gpus`` if given."""
    env = {k: v for k, v in os.environ.items() if k not in CLEAR}
    path = [str(tmp / "bin")] + ([str(tmp / "smi")] if gpus is not None else [])
    env.update(PATH=os.pathsep.join([*path, env["PATH"]]), PYTHON=sys.executable,
               FAKE_BOXES=str(tmp / "boxes"), GIT_CONFIG_NOSYSTEM="1",
               GIT_CONFIG_GLOBAL=os.devnull, **({"FAKE_GPUS": str(gpus)} if gpus else {}))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _sh(*cmd: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, env=env, capture_output=True, text=True, timeout=60)


def _round(repo: Path, env: dict, *runs_args: str) -> subprocess.CompletedProcess:
    return _sh("bash", str(repo / "scripts/gpu_round.sh"), "demo", "scripts/demo_runs.py",
               *runs_args, env=env)


def _pull(repo: Path, box: str, env: dict) -> subprocess.CompletedProcess:
    return _sh("bash", str(repo / "scripts/gpu_sync.sh"), box, "pull-round", "demo",
               "scripts/demo_runs.py", *FIXTURE_ARGS, env=env)


def _manifest(repo: Path, cmd: str, *args: str, env: dict) -> subprocess.CompletedProcess:
    return _sh(sys.executable, str(repo / "scripts/round_manifest.py"), cmd, "--round", "demo",
               "--runs-script", "scripts/demo_runs.py", *args, env=env)


def _check(repo: Path, env: dict, tag: str = TAG, runs_args=FIXTURE_ARGS):
    return _manifest(repo, "check", "--tag", tag, "--runs-args", *runs_args, env=env)


def _runs(*runs_args: str) -> list[dict]:
    """The planted run list, computed here independently of round_manifest.py."""
    proc = subprocess.run([sys.executable, "-c", RUNS_SCRIPT, *runs_args], capture_output=True,
                          text=True, check=True)
    return [json.loads(line) for line in proc.stdout.splitlines()]


def _tsv(proc: subprocess.CompletedProcess) -> list[list[str]]:
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0] == "lane\tgpu\tposition\tname"
    return [line.split("\t") for line in lines[1:]]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The tagged origin repository, the stand-in tools, two boxes that ran shards 0/2 and
    1/2 (two A6000s each, 2 lanes per GPU) and a Mac clone that pulled both back."""
    tmp = tmp_path_factory.mktemp("gpu_round")
    origin = tmp / "origin"
    origin.mkdir()
    _g(origin, "init", "-q", "-b", "main")
    for rel in ("scripts/gpu_round.sh", "scripts/round_manifest.py", "scripts/gpu_sync.sh",
                "src/apexzero/tracks/corners.py"):
        _put(origin, rel, (ROOT / rel).read_text())
    for rel, text in [(".gitignore", "runs/\n"), ("README.md", "demo\n"),
                      ("scripts/demo_runs.py", RUNS_SCRIPT), ("scripts/train_dr.py", TRAIN_DR),
                      ("scripts/train.py", TRAIN), ("scripts/other.py", OTHER)]:
        _put(origin, rel, text)
    _g(origin, "add", ".")
    _g(origin, "commit", "-q", "-m", "demo")
    _g(origin, "tag", TAG)
    for d, name, text in [("bin", "ssh", FAKE_SSH), ("bin", "rsync", FAKE_RSYNC),
                          ("smi", "nvidia-smi", FAKE_SMI)]:
        _put(tmp, f"{d}/{name}", text).chmod(0o755)
    (tmp / "boxes").mkdir()
    for i in (0, 1):
        box = tmp / "boxes" / f"box{i}" / "apexzero"
        _g(tmp, "clone", "-q", str(origin), str(box))
        proc = _round(box, _env(tmp, gpus=2, SHARD=f"{i}/2", LANES_PER_GPU=2,
                                BREV_TYPE="massedcompute_A6000_plusx2"), *FIXTURE_ARGS)
        assert proc.returncode == 0, (proc.stdout, proc.stderr,
                                      (box / f"runs/demo-queue-{i}of2.log").read_text())
    mac = tmp / "mac"
    _g(tmp, "clone", "-q", str(origin), str(mac))
    for i in (0, 1):
        proc = _pull(mac, f"box{i}", _env(tmp, SHARD=f"{i}/2"))
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
    return {"tmp": tmp, "origin": origin, "mac": mac}


# --------------------------------------------------------------------------- the schedule


@pytest.mark.parametrize("gpus", [1, 2, 4])
def test_dry_run_schedule(world, gpus):
    runs = _runs("--specialist")
    rows = _tsv(_round(world["origin"], _env(world["tmp"], GPUS=gpus, LANES_PER_GPU=4,
                                             DRY_RUN=1), "--specialist"))
    lanes = gpus * 4
    assert [r[3] for r in rows] == [r["name"] for r in runs]       # every run once, in order
    for j, (lane, gpu, position, _) in enumerate(rows):
        assert (int(lane), int(gpu), int(position)) == (j % lanes, (j % lanes) % gpus, j // lanes)
    assert not (world["origin"] / "runs").exists()                  # nothing ran or was written


@pytest.mark.parametrize("k", [1, 2, 5, 10])
def test_shards_take_whole_seeds(world, k):
    runs = _runs("--specialist")
    shards = [[r[3] for r in _tsv(_round(world["origin"], _env(world["tmp"], GPUS=1, DRY_RUN=1,
                                                               SHARD=f"{i}/{k}"), "--specialist"))]
              for i in range(k)]
    every = [n for shard in shards for n in shard]
    assert sorted(every) == sorted(r["name"] for r in runs)          # each run in one shard
    seed = {r["name"]: r["seed"] for r in runs}
    for i, shard in enumerate(shards):
        assert {seed[n] for n in shard} == {s for n, s in enumerate(SEEDS) if n % k == i}
        assert shard == [r["name"] for r in runs if SEEDS.index(r["seed"]) % k == i]


@pytest.mark.parametrize("env, message", [
    (dict(GPUS=1, SHARD="2/2"), "SHARD must be i/K"),
    (dict(GPUS=1, SHARD="01/2"), "SHARD must be i/K"),
    (dict(GPUS=0), "GPUS must be a positive integer"),
    (dict(GPUS=1, LANES_PER_GPU="x"), "LANES_PER_GPU must be a positive integer"),
    (dict(), "nvidia-smi not found: set GPUS"),
    (dict(smi=2, GPUS=4), "GPUS=4, but nvidia-smi lists 2 GPU(s)"),
    (dict(GPUS=1, SHARD="9/10", runs=["--seeds", "4"]), "shard 9/10 has no runs"),
    (dict(GPUS=1, runs=["--specialist", "--track", "Monza"]), "names a lockbox (R_TEST) circuit"),
])
def test_refusals(world, env, message):
    smi, runs_args = env.pop("smi", None), env.pop("runs", [])
    proc = _round(world["origin"], _env(world["tmp"], gpus=smi, DRY_RUN=1, **env), *runs_args)
    assert proc.returncode == 1 and message in proc.stderr, proc.stderr
    assert proc.stdout == ""


def test_gpu_count_from_nvidia_smi(world):
    rows = _tsv(_round(world["origin"], _env(world["tmp"], gpus=2, DRY_RUN=1), "--seeds", "4"))
    assert {r[1] for r in rows} == {"0", "1"} and {r[0] for r in rows} == set(map(str, range(8)))


def test_run_list_problems(monkeypatch):
    monkeypatch.chdir(ROOT)
    base = {"name": "gw3-v1-base-s20", "seed": 20, "config": "runs/gw3-v1-base-s20/config.json",
            "cmd": ["scripts/train_dr.py", "--steps", "200e6", "--seed", "20", "--name",
                    "gw3-v1-base-s20"]}
    assert RM.run_list_problems([base]) == []
    assert RM.run_list_problems([]) == ["the run list is empty"]

    def problems(**change):
        return RM.run_list_problems([{**base, **change}])

    def with_cmd(*extra):
        return problems(cmd=base["cmd"] + list(extra))

    assert "listed twice" in RM.run_list_problems([base, base])[0]
    assert "not a plain directory name" in problems(name="../x")[0]
    assert "not a plain directory name" in problems(name="a b")[0]
    assert "is not an int" in problems(seed=True)[0]
    assert "is not an int" in problems(seed="20")[0]
    assert "non-empty list" in problems(cmd=[])[0]
    assert "is not a script in the repo" in problems(cmd=["scripts/nope.py"] + base["cmd"][1:])[0]
    assert "exactly once" in problems(cmd=base["cmd"][:-2])[0]
    assert "exactly once" in with_cmd("--name=gw3-v1-base-s20")[0]
    assert "not a file inside runs/gw3-v1-base-s20/" in problems(config="runs/other/config.json")[0]
    assert "not a file inside" in problems(config="runs/gw3-v1-base-s20/../x/config.json")[0]
    assert "NUL" in with_cmd("a\0b")[0]
    for arg in ("Monza", "spa-reversed", "data/tracks/tumftm/Suzuka.csv", "--tracks=Austin",
                "Hockenheim,Silverstone"):
        assert "names a lockbox (R_TEST) circuit" in with_cmd("--tracks", arg)[0], arg
    assert "outputs/sealed" in with_cmd("--from", "outputs/sealed/x")[0]
    assert with_cmd("--tracks", "Hockenheim,Sochi", "--real-widths") == []   # R-dev only: fine


def test_lockbox_rule_matches_viz_export(monkeypatch):
    from apexzero.tracks.corners import R_DEV, R_TEST
    from apexzero.viz.export import is_lockbox
    monkeypatch.chdir(ROOT)
    prefixes = RM.lockbox_prefixes()
    names = [*R_DEV, *R_TEST, "Monza-mirrored", "Mexico City", "Spa_reversed", "./Monza",
             "data/tracks/tumftm/Austin.csv", "SpaFrancorchamps", "gw3-v1-base-s20", "mlp", "200e6"]
    assert [RM.names_lockbox(n, prefixes) for n in names] == [is_lockbox(n) for n in names]


# --------------------------------------------------------------------------- a queue on the Mac


def test_queue_on_the_mac(world):
    tmp = world["tmp"]
    box = tmp / "boxes" / "mac-box" / "apexzero"
    _g(tmp, "clone", "-q", str(world["origin"]), str(box))
    args = ["--seeds", "3", "--fail-once", "demo-realw-s25", "--fail-always", "demo-tilt-s20",
            "--other"]
    proc = _round(box, _env(tmp, GPUS=1), *args)               # no nvidia-smi: GPUS is forced
    assert proc.returncode == 1                                   # demo-tilt-s20 failed twice
    assert proc.stdout == "queue log: runs/demo-queue-0of1.log\n"
    runs = _runs(*args)
    qlog = (box / "runs/demo-queue-0of1.log").read_text()
    assert "retry demo-realw-s25" in qlog and "FAILED demo-tilt-s20 (twice" in qlog
    man = json.loads((box / "runs/demo-manifest-0of1.json").read_text())
    status = {r["name"]: r["status"] for r in man["runs"]}
    assert [r["name"] for r in man["runs"]] == [r["name"] for r in runs]
    assert status.pop("demo-realw-s25") == "retried" and status.pop("demo-tilt-s20") == "failed"
    assert set(status.values()) == {"done"}
    assert man["head"] == _g(box, "rev-parse", "HEAD") and man["tree_clean"] is True
    assert man["tags"] == [TAG] and man["shard"] == "0/1" and man["box_type"] == "unknown"
    assert (man["gpu_model"], man["gpu_count"], man["gpus"], man["lanes_per_gpu"]) == (None, None,
                                                                                       1, 4)
    assert man["xla_mem_fraction"] == "0.23" and man["status_unlisted"] == []
    for rec in man["runs"]:
        assert rec["start"] <= rec["end"] and rec["gpu"] == 0 and rec["lane"] == runs.index(
            next(r for r in runs if r["name"] == rec["name"])) % 4
        config = box / rec["config"]
        if rec["status"] != "failed":
            assert rec["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
        if rec["name"].startswith(("demo-base", "demo-realw", "demo-tilt")) and config.exists():
            env = json.loads((config.parent / "env.json").read_text())
            assert env == {"CUDA_VISIBLE_DEVICES": "0", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.23"}
    # the retry started from scratch and kept the first attempt's log
    assert "planted failure" in (box / "runs/demo-realw-s25.log.1").read_text()
    assert not (box / "runs/demo-realw-s25/partial").exists()
    assert (box / "runs/demo-realw-s25/config.json").exists()
    assert "planted failure" in (box / "runs/demo-tilt-s20.log").read_text()
    # each lane ran its jobs in order
    starts = [line.split("\t") for line in (box / "runs/demo-status-0of1.tsv").read_text()
              .splitlines() if line.split("\t")[1] == "start"]
    for lane in range(4):
        assert [f[0] for f in starts if f[3] == str(lane)] == [r["name"] for j, r in
                                                              enumerate(runs) if j % 4 == lane]
    # the queue does not start over what is there; the refusal is in the queue log
    again = _round(box, _env(tmp, GPUS=1), *args)
    assert again.returncode == 1 and "refusing to start" in again.stderr
    assert "ERROR no schedule: refusing to start" in (box / "runs/demo-queue-0of1.log").read_text()
    # the check reports the failed run, the missing GPU and the unknown trainer
    out = _check(box, _env(tmp), runs_args=args).stdout
    assert "demo-tilt-s20: status 'failed'" in out
    assert "GPU model None" in out
    assert "demo-other-s27: scripts/other.py's config format is unknown" in out
    assert "FAILED" in out and "OK" not in out


def test_queue_refuses_a_dirty_tree(world):
    """round_manifest.py check refuses a round whose tracked tree was not clean, so the queue
    does not start on one (the refusal in the queue log, nothing trained); DRY_RUN only notes
    it. Once the tree is clean the same shard starts: the refusal left nothing in its way."""
    tmp = world["tmp"]
    box = tmp / "boxes" / "dirty-box" / "apexzero"
    _g(tmp, "clone", "-q", str(world["origin"]), str(box))
    (box / "README.md").write_text("edited on the box\n")
    dry = _round(box, _env(tmp, GPUS=1, DRY_RUN=1), "--seeds", "2")
    assert dry.returncode == 0 and "the tracked tree is not clean" in dry.stderr
    assert len(_tsv(dry)) == 6
    proc = _round(box, _env(tmp, GPUS=1), "--seeds", "2")
    assert proc.returncode == 1 and proc.stdout == ""
    assert "the tracked tree is not clean" in proc.stderr and "README.md" in proc.stderr
    assert "ERROR the tracked tree is not clean" in (box / "runs/demo-queue-0of1.log").read_text()
    assert sorted(p.name for p in (box / "runs").iterdir()) == ["demo-queue-0of1.log"]
    _g(box, "checkout", "--", "README.md")
    again = _round(box, _env(tmp, GPUS=1), "--seeds", "2")
    assert again.returncode == 0, again.stderr
    man = json.loads((box / "runs/demo-manifest-0of1.json").read_text())
    assert man["tree_clean"] is True and {r["status"] for r in man["runs"]} == {"done"}


# --------------------------------------------------------------------------- write and check


def test_pulled_shards_pass_the_check(world):
    mac = world["mac"]
    proc = _check(mac, _env(world["tmp"]))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "runs/demo-manifest-0of2.json: shard 0/2, 8 runs, massedcompute_A6000_plusx2" in out
    assert "2 x NVIDIA RTX A6000 (2 used, 2 lanes per GPU)" in out
    assert "runs: 15 done, 1 retried; 16 configs compared" in out
    assert "--vg-lambda has a computed default" in out and "--log-every is not recorded" in out
    assert out.rstrip().endswith("with the arguments of its cmd")
    assert (mac / "runs/demo-realw-s25.log.1").exists()            # pull-round took .log.1
    for f in ("demo-queue-0of2.log", "demo-queue-1of2.log", "demo-status-1of2.tsv"):
        assert (mac / "runs" / f).exists()


def test_pull_round_reports_what_is_missing(world):
    tmp = world["tmp"]
    mac = tmp / "mac-partial"
    _g(tmp, "clone", "-q", str(world["origin"]), str(mac))
    proc = _pull(mac, "box0", _env(tmp))
    assert proc.returncode == 1                       # SHARD 0/1 expects box1's runs here too
    assert "missing (16): runs/demo-base-s20 runs/demo-base-s20.log" in proc.stderr
    assert (mac / "runs/demo-base-s27/config.json").exists()       # box0's were pulled


def _edit_json(path: Path, fn) -> None:
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data, indent=1))


def _set_config(repo: Path, name: str, fn) -> None:
    """Edit a run's config and record its new sha256 in the manifest that lists it."""
    _edit_json(repo / f"runs/{name}/config.json", fn)
    sha = hashlib.sha256((repo / f"runs/{name}/config.json").read_bytes()).hexdigest()
    for man in repo.glob("runs/demo-manifest-*.json"):
        _edit_json(man, lambda m: [r.update(config_sha256=sha) for r in m["runs"]
                                   if r["name"] == name])


def _first_record(repo):
    return json.loads((repo / "runs/demo-manifest-1of2.json").read_text())["runs"][0]


FAULTS = {
    "missing": (lambda r: _edit_json(r / "runs/demo-manifest-0of2.json",
                                     lambda m: m["runs"].pop(0)),
                ["demo-base-s27: in no manifest"]),
    "duplicate": (lambda r: _edit_json(r / "runs/demo-manifest-0of2.json",
                                       lambda m: m["runs"].append(_first_record(r))),
                  ["demo-base-s20: in 2 manifests", "demo-base-s20 is not in shard 0/2"]),
    "dirty": (lambda r: _edit_json(r / "runs/demo-manifest-1of2.json",
                                   lambda m: m.update(tree_clean=False)),
              ["runs/demo-manifest-1of2.json: the box's tracked tree was not clean"]),
    "train_dr_args": (lambda r: _set_config(r, "demo-tilt-s27",
                                            lambda c: c["args"].update(tilt_fraction=0.3)),
                      ["demo-tilt-s27: --tilt-fraction is 0.3 in the config, 0.5 by the cmd"]),
    "train_dr_extra": (lambda r: _set_config(r, "demo-base-s20",
                                             lambda c: c["args"].update(stat_widths=True)),
                       [("demo-base-s20: config records --stat-widths, which the trainer "
                         "does not define")]),
    "train_args": (lambda r: _set_config(r, "demo-spec-s21",
                                         lambda c: c["ppo"].update(total_timesteps=3000)),
                   ["demo-spec-s21: --steps is 3000 in the config, 2000 by the cmd"]),
    "seed": (lambda r: _set_config(r, "demo-spec-s21", lambda c: c.update(seed=22)),
             ["demo-spec-s21: --seed is 22 in the config, 21 by the cmd"]),
    "not_the_box_config": (lambda r: _edit_json(r / "runs/demo-base-s25/config.json",
                                                lambda c: c.update(note="edited")),
                           ["demo-base-s25: runs/demo-base-s25/config.json is not the box's"]),
    "stray": (lambda r: (r / "runs/demo-stray-s99").mkdir(),
              ["runs/demo-stray-s99: has the round's prefix but is not in the run list"]),
    "gpu_model": (lambda r: _edit_json(r / "runs/demo-manifest-1of2.json",
                                       lambda m: m.update(gpu_model="NVIDIA L40S")),
                  ["the boxes' GPU models differ"]),
    "runs_script": (lambda r: _put(r, "scripts/demo_runs.py",
                                   (r / "scripts/demo_runs.py").read_text() + "# edited\n"),
                    ["scripts/demo_runs.py differs from its version at demo-v1"]),
    "run_dir": (lambda r: shutil.rmtree(r / "runs/demo-base-s21"),
                ["demo-base-s21: runs/demo-base-s21 is not here",
                 "demo-base-s21: runs/demo-base-s21/config.json is not here"]),
}


@pytest.mark.parametrize("fault", list(FAULTS))
def test_check_reports_each_fault(world, tmp_path, fault):
    repo = tmp_path / "mac"
    shutil.copytree(world["mac"], repo, symlinks=True)
    plant, expected = FAULTS[fault]
    plant(repo)
    proc = _check(repo, _env(world["tmp"]))
    assert proc.returncode == 1, proc.stdout
    problems = proc.stdout.split("FAILED: ", 1)[1].splitlines()[1:]
    for e in expected:
        assert any(e in p for p in problems), (e, problems)
    assert len(problems) == len(expected), problems


def test_check_wants_the_tagged_head(world, tmp_path):
    repo = tmp_path / "mac"
    shutil.copytree(world["mac"], repo, symlinks=True)
    _put(repo, "README.md", "demo, after the round\n")
    _g(repo, "commit", "-q", "-am", "later")
    _g(repo, "tag", "demo-v2")
    out = _check(repo, _env(world["tmp"]), tag="demo-v2").stdout
    head = _g(repo, "rev-parse", "HEAD")[:12]
    assert f"runs/demo-manifest-0of2.json: HEAD {_g(repo, 'rev-parse', TAG)[:12]} is not " \
           f"demo-v2 ({head})" in out
    assert _check(repo, _env(world["tmp"]), tag="nope").returncode == 1


def test_write_records_the_tree_and_the_box(world, tmp_path):
    box = tmp_path / "box"
    shutil.copytree(world["tmp"] / "boxes/box0/apexzero", box, symlinks=True)
    _put(box, "README.md", "edited on the box\n")
    proc = _manifest(box, "write", "--shard", "0/2", "--runs-args", *FIXTURE_ARGS,
                     env=_env(world["tmp"], gpus=2, BREV_TYPE="massedcompute_A6000_plus",
                              LANES_PER_GPU=3))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("wrote runs/demo-manifest-0of2.json: 8 runs, 7 done, 1 retried")
    man = json.loads((box / "runs/demo-manifest-0of2.json").read_text())
    assert man["tree_clean"] is False and man["box_type"] == "massedcompute_A6000_plus"
    assert (man["gpu_model"], man["gpu_count"], man["lanes_per_gpu"], man["gpus"]) == (
        "NVIDIA RTX A6000", 2, 3, None)
    assert man["run_list_sha256"] == RM.run_list_sha256(_runs(*FIXTURE_ARGS))


# --------------------------------------------------------------------------- the real trainers


def test_real_trainers_parsers():
    dr = RM.trainer_parser((ROOT / "scripts/train_dr.py").read_text())
    cmd = ["--steps", "200e6", "--seed", "20", "--name", "gw3-v1-base-s20", "--tilt-fraction",
           "0.5", "--tilt-alpha", "2", "--real-widths"]
    args = vars(dr.parse_args(cmd))
    assert {k for k, v in args.items() if v is RM.COMPUTED} == {"vg_lambda", "vg_winsor"}
    assert (args["steps"], args["seed"], args["tilt_alpha"], args["real_widths"],
            args["num_envs"], args["network"], args["vg_coef"]) == (2e8, 20, 2.0, True, 2048,
                                                                     "mlp", 0.0)
    recorded = {"args": {**args, "vg_lambda": 0.9, "vg_winsor": 0.99}}
    assert RM.compare_train_dr(recorded, args)[0] == []
    with pytest.raises(ValueError, match="unrecognized"):
        dr.parse_args(["--tracks", "Hockenheim"])
    tr = RM.trainer_parser((ROOT / "scripts/train.py").read_text())
    args = vars(tr.parse_args(["--tracks", "Hockenheim", "Sochi", "--steps", "50e6", "--seed",
                               "3", "--name", "p4b2-x"]))
    assert RM.COMPUTED not in args.values()
    config = {"ppo": {"num_envs": 2048, "num_steps": 64, "total_timesteps": 50_000_000,
                      "lr": 3e-4, "network": "mlp"}, "tracks": ["Hockenheim", "Sochi"], "seed": 3}
    problems, notes = RM.compare_train(config, args)
    assert problems == []
    assert notes == ["--log-every is not recorded in train.py's config: not compared"]
    config["tracks"] = ["Hockenheim"]
    assert RM.compare_train(config, args)[0] == [
        "--tracks is ['Hockenheim'] in the config, ['Hockenheim', 'Sochi'] by the cmd"]
