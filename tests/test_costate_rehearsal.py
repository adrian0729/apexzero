"""The post-tag rehearsal (design v2.1 Section 7: after the tag the post-tag stages run once,
and a crash voids the tag): every stage of scripts/costate.py through its ``main()``, in
order, on random-init stand-ins for the trained networks, before the tag.

Opt-in (about 83 minutes on 2 threads; K4 takes 32 of them on the reduced T, PR-6b and PR-6c
most of it, the probes stage 27, decoy-k and c2-null 6): ``COSTATE_REHEARSAL=1 uv run --no-sync pytest
tests/test_costate_rehearsal.py -p no:cacheprovider``, or ``uv run --no-sync python
tests/test_costate_rehearsal.py --dir DIR`` (keeps DIR; ``--from STEP`` resumes there).

- **A scratch clone** of this repository's HEAD (its branch 'main'; step 1's, step 2's and
  H4R's tags with it) whose origin is a local bare repository, so the freeze check's tag,
  push, ancestor, pin and committed-file rules run for real. Its outputs/outputs links to
  step 1's keys and stimuli and step 2's outputs (read only); its runs/ is a directory of
  its own.
- **Stand-in runs** (``stand_in``): each network's real config.json, eval_rdev.json and
  metrics.csv copied, and its checkpoint rewritten in the same pickle format with the real
  config, obs_rms, ret_rms and timesteps but random-init actor and critic params of exactly
  the real trees (``network.random_actor_params``, ``critic.random_critic_params`` with the
  real PPOConfig; checked leaf by leaf on names, shapes and dtypes). No trained parameter
  enters any computation, and the clone cannot reach the real runs/.
- **Reduced lists, not partial options** (``_DRIVER``, in every stage's process): the A2
  keys (``A2``: 10 geometries, a horizon twin pair so T holds W = 800 keys, a geometry with
  5 keys, the kappa set, braking and non-braking keys, one multi-basin key), N's anchored
  finals and K1's sample (``N``, ``K1``), and the networks (``NETWORKS``: one decision
  network, two other generalists, one of them without step 2's overlap, and one specialist
  with 0 heads). Every output is therefore stamped as a full run and prereg binds it. K2
  draws its insurance solves from the full A2 list (30 of them, as in the real run). The
  power floor and K2b's exclusions on the reduced T are replaced by the real T's
  (``POWER_FLOOR``, ``K2B_EXCLUDED``; the reduced T's own are kept beside them as
  'power_floor_reduced' and 'excluded_reduced'), K4's, decoy-k's and K5's verdicts are
  forced to pass (``FORCE_PASS``; 'pass_computed' beside them: decoy-k cannot reproduce the
  full-size table on the reduced T), K5's descriptive MLP ceilings stop at
  ``K5_MLP_MAXITER`` iterations, decoy-k runs one candidate on one validation seed
  (``DECOY_K_REDUCED``) and c2-null two null actors (``C2_NULL_REDUCED``).
- **The sequence** (``STEPS``): inputs, k2, inputs, k2b, k0, k1, k3, k3b, k4, decoy-k,
  c2-null, k5, k6, nulls, prereg; the record committed, annotated-tagged and pushed; a pre-tag stage refused;
  activations, tag.json committed, critic, own, probes, a non-confirmatory analyse
  (coverage only), analyse --confirmatory, its decisions committed; a second analyse
  --confirmatory refused; a non-confirmatory analyse unlocked by the committed decisions.

The report (``report``; ``--dir``'s report.json) lists each step's time, every contained
error and every decision stated on a non-finite CI; the test fails on either.

What it cannot exercise: trained weights. A random-init policy crashes within about 20 steps,
so no own run is valid: C1-own, its held form and value accuracy record 'not testable' or
'fewer than 3 rows', and K3b's trained analogue fails (tests/test_costate_stages.py scores
them on planted own arrays). Every stand-in passes K3's after-the-tag checks (the failing
network is planted there too). Nor the full T (9,887 rows), the 34 networks and the 9
decision networks (R18's variants see one seed), their run time and memory, C1-dev's 16
snapshots per network (one here), or K4's, decoy-k's and K5's real verdicts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:          # run as a script
    sys.path.insert(0, str(REPO / "src"))

from apexzero.costate import data_dirs

KEYS, STIM = data_dirs()
TAG = "costate-prereg-v1"
# The reduced A2 list (key ids), with why each is there.
A2 = (
    "352b4526cd14ce24ab1ea154",   # R13/T180 nominal W = 400: a horizon twin (K2) and K2b's pool
    "6e8647f9cf83158e92edc1ce",   # its W = 800 twin (in T: W = 800 rows)
    "8462466b41cadeca372d8d97",   # R13/T180 phases 1-3: a geometry with 5 keys (drop_small)
    "7b1e8008b6d9271a44f77ac1",
    "c8c699565066c5fcbf37fb56",
    "0005ec83ccf94435ead117b7",   # R8.4/T90: the kappa set (R <= 8.94)
    "34ccb8024e28b43334578b23",   # R22.9/T45 nominal: a non-braking key, K2b's pool
    "125279c3981f47365f220d1a",   # one phase key on each of 7 more geometries
    "18d48f23589210d214a1cc2e",
    "08a68ba6739ee47769c371bb",
    "06df14402af6b9063ae89370",
    "0205da2fbbbfdc8dfab273e5",
    "0d4a845670701aae0c2ca77f",
    "051c06f9954f9a9e8130af21",
    "6077a27814dde7460f26ec3a",   # multi-basin: left out of T and listed
)
N = {"G1": ("0016821f8af1f6e163150a04",), "S1": ("0008733b7d9b371327de21b9",)}
K1 = ("0016821f8af1f6e163150a04", "0005ec83ccf94435ead117b7")   # an anchored key (pi) and an A2
NETWORKS = {
    "generalists": ("runs/dr-pilot-200M",            # G1: the decision network
                    "runs/dr-pilot-200M-s3",         # base-v1 with G1: C3-w's ranks
                    "runs/dr-pilot-v2-tilt30-s3"),   # base-v2: no step 2 overlap
    "decision": ("G1",),
    "specialists": ("S1",),
}
# The reduced T's power floor and K2b exclusions are replaced by the real T's (the smoke run
# of the pre-tag stages on every key): the post-tag stages then meet the components and
# phases the real run decides on (n and vx have no decision row at all). The floor is the
# interior one at f_D's k = 8 (the third full-size check, 428f51f): n's share 0.091 < 0.10.
POWER_FLOOR = {"s": True, "n": False, "xi": True, "vx": False}
K2B_EXCLUDED = {"approach": ["n", "vx"], "braking": ["s", "n", "xi", "vx"],
                "corner": ["s", "n", "xi", "vx"], "exit": ["n", "xi", "vx"]}
# K4's planted recoveries fail on so small a T and K5's wiring check can (on the real T both
# pass at 428f51f, a pre-tag matter), and decoy-k cannot reproduce the full-size validation
# table on it: their verdicts are forced to pass so that prereg binds them, the computed one
# kept beside it ('pass_computed').
FORCE_PASS = ("k4", "decoy-k", "k5")
DECOY_K_REDUCED = {"candidates": [8], "seeds": [0]}   # f_D's k re-derived: one k, one seed
C2_NULL_REDUCED = (5, 6)       # C2's random-only null on two of its 20 actors
K5_MLP_MAXITER = 25            # K5's MLP ceilings (descriptive) at 25 L-BFGS iterations, not 500:
                               # 19 minutes on the reduced T, about one with the cap
STAND_IN_SEED = 101            # + index: never one of the random controls' seeds (0-19)
COPIED = ("config.json", "eval_rdev.json", "metrics.csv")
STEPS = ("inputs", "k2", "inputs-cut", "k2b", "k0", "k1", "k3", "k3b", "k4", "decoy-k",
         "c2-null", "k5", "k6", "nulls", "prereg", "tag", "refuse-pre-tag", "activations", "pin", "critic", "own",
         "probes", "analyse-coverage", "analyse", "commit-decisions", "analyse-again",
         "analyse-dry")
POST = ("activations", "critic", "own", "probes", "analyse")

# Runs in the clone, one process per stage (as the real stages do): the reduced lists
# patched in, then the clone's scripts/costate.py main() with the stage's own argv.
_DRIVER = r"""
import json, sys
from pathlib import Path

repo, conf = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
threads, argv = sys.argv[3], sys.argv[4:]
sys.argv = ["scripts/costate.py", "--threads", threads, *argv]   # what every stamp records
sys.path[:0] = [str(repo / "scripts"), str(repo / "src")]
import costate as cs                                   # the clone's (XLA flags, src/ first)
from apexzero.costate import analysis as AN, gates as GT, keys_root, prereg as PR
from apexzero.costate import states as ST
from apexzero.fields import keys as K

AN.GENERALISTS = tuple(conf["generalists"])
AN.DECISION_NETWORKS = tuple(conf["decision"])
AN.SPECIALISTS = tuple(conf["specialists"])
_a2, a2 = ST.a2_refs, set(conf["a2"])


def a2_refs(root=None):
    return tuple(r for r in _a2(root) if r.key_id in a2)


def n_refs(root=None):
    got = {}
    for r in K.key_index("anchored", keys_root(root)):
        for a, ids in conf["n"].items():
            if r.key_id in ids:
                got.setdefault(a, []).append(r)
    return {a: tuple(got[a]) for a in conf["n"] if a in got}


def k1_sample(*args, **kwargs):
    pool = {r.key_id: r for r in (*_a2(), *(r for v in n_refs().values() for r in v))}
    return [pool[i] for i in conf["k1"]]


if argv[0] != "k2":            # K2's 30 insurance solves draw from the full A2 list
    ST.a2_refs = a2_refs
ST.n_refs = n_refs
GT.k1_sample = k1_sample
if argv[0] == "inputs":
    _fv = PR.frozen_values

    def frozen_values(tab, t_H, partitions=None):
        out = _fv(tab, t_H, partitions)
        out["power_floor_reduced"] = out["power_floor"]
        out["power_floor"] = dict(conf["power_floor"])
        return out

    PR.frozen_values = frozen_values
if argv[0] == "decoy-k":
    _dk = GT.decoy_k

    def decoy_k(pw, *args, **kwargs):
        kwargs.update(candidates=tuple(conf["decoy_k"]["candidates"]),
                      seeds=tuple(conf["decoy_k"]["seeds"]))
        return _dk(pw, *args, **kwargs)

    GT.decoy_k = decoy_k
if argv[0] == "c2-null":
    _cn = GT.c2_null

    def c2_null(pw, cfg, *args, **kwargs):
        kwargs["seeds"] = tuple(conf["c2_null_seeds"])
        return _cn(pw, cfg, *args, **kwargs)

    GT.c2_null = c2_null
if argv[0] in conf["force_pass"]:
    _gate = getattr(GT, argv[0].replace("-", "_"))

    def forced(*args, **kwargs):
        out = _gate(*args, **kwargs)
        out["pass_computed"] = out["pass"]
        out["pass"] = True
        return out

    setattr(GT, argv[0].replace("-", "_"), forced)
if argv[0] == "k5":
    from apexzero.costate import probes as PB

    _fit = PB.mlp_fit

    def mlp_fit(X, y, w, alpha, rng, hidden=PB.MLP_HIDDEN, maxiter=PB.MLP_MAXITER, sc=None):
        return _fit(X, y, w, alpha, rng, hidden, min(int(maxiter), conf["k5_mlp_maxiter"]), sc)

    PB.mlp_fit = mlp_fit
if argv[0] == "k2b":
    _rule = GT.k2b_rule

    def k2b_rule(keys, *args, **kwargs):
        out = _rule(keys, *args, **kwargs)
        out["excluded_reduced"] = out["excluded"]
        out["excluded"] = {p: list(v) for p, v in conf["k2b_excluded"].items()}
        return out

    GT.k2b_rule = k2b_rule
cs.main(sys.argv[1:])
"""


def _env() -> dict:
    return dict(os.environ, GIT_AUTHOR_NAME="rehearsal", GIT_AUTHOR_EMAIL="rehearsal@invalid",
                GIT_COMMITTER_NAME="rehearsal", GIT_COMMITTER_EMAIL="rehearsal@invalid")


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
                       cwd=root, capture_output=True, text=True, env=_env(), check=False)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[-500:]}")
    return r.stdout.strip()


def _layout(tree) -> list:
    import jax
    import numpy as np

    return [(jax.tree_util.keystr(p), np.shape(v), str(np.asarray(v).dtype))
            for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]]


def stand_in(src: Path, dst: Path, seed: int) -> dict:
    """``src``'s checkpoint as a random-init stand-in at ``dst`` (module docstring): the
    pickle's every item but the actor's and critic's params kept, those built by the
    project's own init with the real PPOConfig and checked against the real trees' layout
    (names, shapes, dtypes; no value read)."""
    from apexzero.agents.ppo import PPOConfig
    from apexzero.costate import critic as CR
    from apexzero.fields import network as NW

    with open(src, "rb") as fh:
        d = pickle.load(fh)
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = PPOConfig(**{k: v for k, v in d["config"].items() if k in known})
    params = {"actor": NW.random_actor_params(cfg, seed),
              "critic": CR.random_critic_params(cfg, seed)}
    for k, p in params.items():
        if _layout(p) != _layout(d[k]):
            raise ValueError(f"{src}: the random-init {k} is not the checkpoint's tree")
    out = {**{k: v for k, v in d.items() if k not in params}, **params}
    del d
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as fh:
        pickle.dump(out, fh)
    return {"path": str(dst), "seed": seed, "heads": int(cfg.ensemble_heads),
            "timesteps": out.get("timesteps")}


def build(scratch: Path, rev: str = "HEAD") -> Path:
    """The scratch clone, its bare origin and its stand-in runs (module docstring)."""
    from apexzero.costate import prereg as PR
    from apexzero.fields import stages as S

    scratch.mkdir(parents=True, exist_ok=True)
    repo, origin = scratch / "repo", scratch / "origin.git"
    sha = _git(REPO, "rev-parse", rev)
    _git(scratch, "clone", "-q", "--no-checkout", "--no-hardlinks", str(REPO), str(repo))
    _git(repo, "checkout", "-q", "-B", "main", sha)
    _git(repo, "remote", "remove", "origin")
    _git(scratch, "init", "-q", "--bare", str(origin))
    _git(repo, "remote", "add", "origin", f"file://{origin}")
    _git(repo, "push", "-q", "origin", "main", "--tags")
    (repo / "outputs").mkdir(exist_ok=True)
    (repo / "outputs" / "outputs").symlink_to(KEYS.resolve().parents[1], target_is_directory=True)
    names = [*NETWORKS["generalists"], *NETWORKS["specialists"]]
    made = {}
    for i, name in enumerate(names):
        src = S.checkpoint_path(name)
        if src is None:
            raise FileNotFoundError(f"{name}: no finished run")
        rel = PR._repo_rel(src, REPO)
        made[name] = stand_in(src, repo / rel, STAND_IN_SEED + i)
        run = Path(rel).parts[:2]
        for f in COPIED:
            p = REPO.joinpath(*run, f)
            if p.exists():
                shutil.copy2(p, repo.joinpath(*run, f))
    conf = {"a2": list(A2), "n": {a: list(v) for a, v in N.items()}, "k1": list(K1),
            **{k: list(v) for k, v in NETWORKS.items()}, "power_floor": POWER_FLOOR,
            "k2b_excluded": K2B_EXCLUDED, "force_pass": list(FORCE_PASS),
            "k5_mlp_maxiter": K5_MLP_MAXITER, "decoy_k": DECOY_K_REDUCED,
            "c2_null_seeds": list(C2_NULL_REDUCED)}
    (scratch / "rehearsal.json").write_text(json.dumps(conf, indent=1) + "\n")
    (scratch / "stand_ins.json").write_text(json.dumps(made, indent=1) + "\n")
    return repo


class Rehearsal:
    """The steps (``STEPS``) on a built scratch directory, each logged with its time."""

    def __init__(self, scratch: Path, threads: int = 2, deviation: str | None = None,
                 echo=print):
        self.scratch = Path(scratch)
        self.repo = self.scratch / "repo"
        self.logs = self.scratch / "logs"
        self.logs.mkdir(exist_ok=True)
        self.threads, self.deviation, self.echo = threads, deviation, echo
        self.state_path = self.scratch / "state.json"
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.exists()
                      else {"head_before": {}, "times": {}})

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1) + "\n")

    def stage(self, step: str, *argv: str, ok: bool = True) -> subprocess.CompletedProcess:
        """``scripts/costate.py <argv>`` in the clone through ``_DRIVER``; ``ok``: it must
        exit 0, else it must refuse (exit non-zero)."""
        args = list(argv)
        if self.deviation and argv[0] in POST:
            args += ["--post-tag-deviation", self.deviation]
        r = subprocess.run([sys.executable, "-c", _DRIVER, str(self.repo),
                            str(self.scratch / "rehearsal.json"), str(self.threads), *args],
                           cwd=self.repo, capture_output=True, text=True, check=False,
                           env={k: v for k, v in os.environ.items()
                                if k not in ("PYTHONPATH", "XLA_FLAGS")})
        (self.logs / f"{step}.log").write_text(f"$ {' '.join(args)}\n[exit {r.returncode}]\n"
                                               f"{r.stdout}\n--- stderr ---\n{r.stderr}")
        if ok and r.returncode != 0:
            raise RuntimeError(f"{step} crashed (exit {r.returncode}):\n{r.stderr[-4000:]}")
        if not ok and r.returncode == 0:
            raise RuntimeError(f"{step} ran where it must refuse")
        return r

    def rewind(self, step: str) -> None:
        """Resume at ``step``: HEAD back to where it was before it (the working tree kept),
        and the post-tag files written from there on removed."""
        later = STEPS[STEPS.index(step):]
        head = next((self.state["head_before"][s] for s in later
                     if s in self.state["head_before"]), None)
        if head is None:
            raise SystemExit(f"no recorded HEAD at or after {step}")
        _git(self.repo, "reset", "-q", "--mixed", head)
        out = self.repo / "outputs" / "costate"
        if "tag" in later and self.state.get("tag_commit"):
            _git(self.repo, "tag", "-d", TAG)
            _git(self.repo, "push", "-q", "origin", f":refs/tags/{TAG}")
            self.state.pop("tag_commit")
        if "activations" in later:
            (out / "tag.json").unlink(missing_ok=True)
        if "analyse" in later:
            (out / "analyse" / "decisions.json").unlink(missing_ok=True)

    def step(self, step: str) -> None:
        r, out = self.repo, self.repo / "outputs" / "costate"
        if step == "inputs-cut":
            self.stage(step, "inputs")
        elif step == "tag":
            _git(r, "add", "src/apexzero/costate/prereg.json")
            _git(r, "commit", "-q", "-m", "Idea 2: the pre-registration record (rehearsal)")
            _git(r, "tag", "-a", TAG, "-m", "costate pre-registration (rehearsal)")
            _git(r, "push", "-q", "origin", "main", TAG)
            self.state["tag_commit"] = _git(r, "rev-parse", "HEAD")
        elif step == "refuse-pre-tag":
            p = self.stage(step, "k6", ok=False)
            if "after the tag" not in p.stderr:
                raise RuntimeError(f"k6 refused for another reason: {p.stderr[-800:]}")
        elif step == "pin":
            if not (out / "tag.json").exists():
                raise RuntimeError("the first post-tag stage wrote no tag.json")
            _git(r, "add", "-f", "outputs/costate/tag.json")
            _git(r, "commit", "-q", "-m", "Idea 2: pin the tag (rehearsal)")
        elif step == "analyse-coverage":
            self.stage(step, "analyse")
            d = json.loads((out / "analyse" / "decisions_dry.json").read_text())
            if d.get("decided") or "coverage" not in d:
                raise RuntimeError("a non-confirmatory analyse before the decisions are "
                                   "committed computed decisions")
        elif step == "analyse":
            self.stage(step, "analyse", "--confirmatory")
        elif step == "commit-decisions":
            _git(r, "add", "-f", "outputs/costate/analyse/decisions.json")
            _git(r, "commit", "-q", "-m", "Idea 2: the confirmatory decisions (rehearsal)")
        elif step == "analyse-again":
            p = self.stage(step, "analyse", "--confirmatory", ok=False)
            if "again" not in p.stderr:
                raise RuntimeError(f"the second analyse refused for another reason: "
                                   f"{p.stderr[-800:]}")
        elif step == "analyse-dry":
            self.stage(step, "analyse")
        else:
            self.stage(step, step)

    def run(self, start: str | None = None) -> dict:
        steps = STEPS[STEPS.index(start):] if start else STEPS
        if start and any(s in self.state["head_before"] for s in steps):
            self.rewind(start)
        for s in steps:
            self.state["head_before"][s] = _git(self.repo, "rev-parse", "HEAD")
            self._save()
            t0 = time.time()
            self.echo(f"[{time.strftime('%H:%M:%S')}] {s}")
            self.step(s)
            self.state["times"][s] = round(time.time() - t0, 1)
            self._save()
        return report(self.repo, self.state)


def _errors(d, path: str = "") -> list[str]:
    """Every contained error ({'error': ...}) in a decisions file, by path."""
    out = []
    if isinstance(d, dict):
        if set(d) == {"error"} or ("error" in d and isinstance(d["error"], str)):
            out.append(f"{path}: {d['error']}")
        for k, v in d.items():
            out += _errors(v, f"{path}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out += _errors(v, f"{path}[{i}]")
    return out


DECIDED = ("confirm", "refute", "inconclusive", "not confirmed",
           "rules out a strong association")


def _nan_decided(d, path: str = "") -> list[str]:
    """Every record that states a decision on a CI that is not finite (a NaN read as a
    verdict: the rehearsal's C1-pattern and C1-limit findings), by path."""
    out = []
    if isinstance(d, dict):
        ci = d.get("ci")
        if d.get("decision") in DECIDED and isinstance(ci, list) and not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in ci):
            out.append(f"{path}: {d['decision']} on ci {ci}")
        for k, v in d.items():
            out += _nan_decided(v, f"{path}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out += _nan_decided(v, f"{path}[{i}]")
    return out


def report(repo: Path, state: dict) -> dict:
    """What the rehearsal produced: step times, the decisions file's shape and its
    secondaries' verdicts, every contained error, every decision on a non-finite CI, the
    strata rows that are not testable and the forced gates' computed verdicts."""
    out = repo / "outputs" / "costate"
    dec = json.loads((out / "analyse" / "decisions.json").read_text())
    dry = json.loads((out / "analyse" / "decisions_dry.json").read_text())
    nets = dec["networks"]
    strata = {}
    for n, rec in nets.items():
        for h, table in (rec.get("by_stratum") or {}).items():
            for s, row in (table or {}).items():
                if isinstance(row, dict) and row.get("band") in ("not testable", "error"):
                    strata[f"{n}/{h}/{s}"] = row.get("reason") or row.get("error")
    secondaries = {n: {k: (v.get("decision") if isinstance(v, dict) else None)
                       for k, v in rec.items() if k.startswith(("C1", "C2", "W1"))}
                   for n, rec in nets.items()}
    forced = {g: json.loads((out / "gates" / f"{g.replace('-', '_')}.json").read_text()).get(
        "pass_computed") for g in FORCE_PASS}
    return {"times": state["times"], "total_s": round(sum(state["times"].values()), 1),
            "networks": {n: sorted(v) for n, v in nets.items()},
            "descriptive_networks": sorted(dec["descriptive_networks"]),
            "per_network": dec["per_network"], "secondaries": secondaries,
            "C3-w": dec["C3-w"]["decision"], "C3-R2c": dec["C3-R2c"]["decision"],
            "k3_trained": dec["k3_trained"]["failed"], "errors": _errors(dec),
            "nan_decided": _nan_decided(dec), "not_testable_strata": strata,
            "forced_gates_computed": forced, "dry_decided": dry.get("decided"),
            "dry_unlocked_by": dry.get("unlocked_by")}


def _have_inputs() -> bool:
    from apexzero.fields import stages as S

    try:
        return (KEYS / "nominal").exists() and all(
            S.checkpoint_path(n) is not None
            for n in (*NETWORKS["generalists"], *NETWORKS["specialists"]))
    except (KeyError, OSError):
        return False


@pytest.mark.skipif(not os.environ.get("COSTATE_REHEARSAL"),
                    reason="the full post-tag rehearsal (about 83 min): COSTATE_REHEARSAL=1")
@pytest.mark.skipif(not _have_inputs(), reason="step 1's keys or the runs are missing")
def test_every_stage_through_main_on_random_init_stand_ins(tmp_path):
    build(tmp_path)
    rep = Rehearsal(tmp_path).run()
    assert not rep["errors"], rep["errors"]
    assert not rep["nan_decided"], rep["nan_decided"]
    assert rep["networks"] and all("C1-own" in v for v in rep["networks"].values())
    assert rep["descriptive_networks"] == sorted(
        [*NETWORKS["generalists"][1:], *NETWORKS["specialists"]])
    assert rep["dry_decided"] is True and rep["dry_unlocked_by"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, required=True, help="the scratch directory (kept)")
    ap.add_argument("--from", dest="start", choices=STEPS, default=None,
                    help="resume an earlier rehearsal in --dir at this step")
    ap.add_argument("--rev", default="HEAD", help="the commit to rehearse (default HEAD)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--deviation", default=None,
                    help="--post-tag-deviation for the post-tag stages (code changed in the "
                         "clone after the tag: a resumed rehearsal only)")
    a = ap.parse_args(argv)
    if not (a.dir / "repo").exists():
        if a.start is not None:
            raise SystemExit(f"no rehearsal in {a.dir} to resume")
        build(a.dir, a.rev)
    elif a.start is None:
        raise SystemExit(f"{a.dir} holds a rehearsal: give --from STEP, or an empty --dir")
    rep = Rehearsal(a.dir, a.threads, a.deviation).run(a.start)
    (a.dir / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps(rep, indent=1, default=str))
    return 1 if rep["errors"] or rep["nan_decided"] else 0


if __name__ == "__main__":
    sys.exit(main())
