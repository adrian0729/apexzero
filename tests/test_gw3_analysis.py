"""scripts/gw3_runs.py and scripts/gw3_analysis.py (GW-3, docs/prereg/gw3-width-round.md): the
run list against the draft's Section 2 and its lane balance, the exact permutation counts
against listing every relabelling, Section 3's statistics on planted networks with known
answers (null and strong effects, missing runs, networks that complete nothing), Section 5's
GW-4 rule and its check of the width report, the tag gate on real scratch repositories, and
the manifest and analysis stages end to end on planted runs, with the real
scripts/round_manifest.py writing and checking the box's manifest.

No network is trained, loaded or rolled out, no R_TEST circuit is read and nothing under
outputs/sealed/ is opened: every run is planted in tmp_path."""

import hashlib
import importlib
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from apexzero.tracks.corners import R_DEV

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DRAFT = (ROOT / "docs" / "prereg" / "gw3-width-round.md").read_text()
REPORT = ROOT / "outputs" / "width-report" / "report.json"
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
NAN = float("nan")


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield {m: importlib.import_module(m)
               for m in ("gw3_analysis", "gw3_runs", "round_manifest", "gw3_power", "h4r")}
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture(scope="module")
def ga(mods):
    return mods["gw3_analysis"]


@pytest.fixture(scope="module")
def rl(mods):
    return mods["gw3_runs"]


def section(n: int) -> str:
    return DRAFT.split(f"\n## {n}. ")[1].split("\n## ")[0]


# ------------------------------------------------------------------------- the run list


def test_run_list_is_section_2s(rl, mods, monkeypatch):
    """120 runs, seed-major, each seed's cells Section 2's list rotated left by s - 20, every
    command Section 2's template with its variant's and arm's flags as the draft writes them."""
    s2 = section(2)
    folds = {f: tuple(v.split(",")) for f, v in re.findall(r"`<fold (f[12])>` = `([^`]+)`", s2)}
    assert folds == rl.FOLDS
    assert tuple(re.search(r"`(v1-base, [^`]+)`", s2)[1].split(", ")) == rl.CELLS
    assert tuple(re.search(r"v3: `([^`]+)`", s2)[1].split()) == rl.VARIANT_FLAGS["v3"]
    assert rl.VARIANT_FLAGS["v1"] == () and "v1: the base grammar" in s2
    arms = {}
    for line in s2.splitlines():
        if line.startswith("| `"):
            first, flags = [c.strip() for c in line.strip("|").split("|")][:2]
            text = re.findall(r"`([^`]+)`", flags)
            for a in re.findall(r"`([a-z0-9-]+)`", first.split(" (")[0]):
                fold = ",".join(rl.FOLDS[a[-2:]]) if a[-3:-1] == "-f" else None
                arms[a] = tuple(text[0].replace("<fold f>", fold or "?").split()) if text else ()
    assert arms == rl.ARM_FLAGS
    assert "**Seeds:** 20 to 29" in s2 and rl.SEEDS == tuple(range(20, 30))
    template = ("`scripts/train_dr.py --steps 200e6 --seed <s> --name <name> <variant flags> "
                "<arm flags>`")
    assert template in s2 and "`gw3-<variant>-<arm>-s<seed>`" in s2
    runs = rl.runs()
    assert len(runs) == 120 and len({r["name"] for r in runs}) == 120
    for j, r in enumerate(runs):
        i, p = divmod(j, 12)
        v, arm = rl.CELLS[(p + i) % 12].split("-", 1)
        s = 20 + i
        name = f"gw3-{v}-{arm}-s{s}"
        assert (r["name"], r["seed"], r["variant"], r["arm"]) == (name, s, v, arm)
        assert r["cmd"] == ["scripts/train_dr.py", "--steps", "200e6", "--seed", str(s),
                            "--name", name, *rl.VARIANT_FLAGS[v], *rl.ARM_FLAGS[arm]]
        assert r["config"] == f"runs/{name}/config.json"
    assert Counter(r["seed"] for r in runs) == {s: 12 for s in rl.SEEDS}
    assert Counter((r["variant"], r["arm"]) for r in runs) == {
        (v, a): 10 for v in rl.VARIANT_FLAGS for a in rl.ARM_FLAGS}
    assert not [a for r in runs for a in r["cmd"] if "reexpress" in a or "vg-" in a]
    monkeypatch.chdir(ROOT)      # round_manifest.py reads paths from the repo root
    assert mods["round_manifest"].run_list_problems(runs) == []


def test_run_list_script_prints_the_list(rl):
    out = subprocess.run([sys.executable, str(SCRIPTS / "gw3_runs.py")], capture_output=True,
                         text=True, check=True).stdout
    assert [json.loads(ln) for ln in out.splitlines()] == rl.runs()
    same = subprocess.run([sys.executable, str(SCRIPTS / "gw3_runs.py"), "--phase", "main"],
                          capture_output=True, text=True, check=True).stdout
    assert same == out
    bad = subprocess.run([sys.executable, str(SCRIPTS / "gw3_runs.py"), "--phase", "pilot"],
                         capture_output=True, text=True, check=False)
    assert bad.returncode == 2


def test_folds_are_the_committed_leave_out_folds(ga, rl):
    assert list(rl.FOLDS["f1"]) == R_DEV[0::2] and list(rl.FOLDS["f2"]) == R_DEV[1::2]
    assert ga.HELD_OUT == {"f1": rl.FOLDS["f2"], "f2": rl.FOLDS["f1"]}


def test_lane_balance_is_as_section_2_states(rl, mods):
    """Section 2's balance claims for one box with GPUS 1, 2 or 4 at 4 lanes per GPU, and for K
    single-GPU boxes taking whole seeds (K = 2, 5, 10)."""
    rm = mods["round_manifest"]
    runs = rl.runs()

    def cell(r):
        return f"{r['variant']}-{r['arm']}"

    for gpus, k in ((1, 1), (2, 1), (4, 1), (1, 2), (1, 5), (1, 10)):
        lanes = 4 * gpus
        for i in range(k):
            jobs = rm.in_shard(runs, i, k)
            assert len(jobs) == 120 // k and len({r["seed"] for r in jobs}) == 10 // k
            plan = rm.schedule(jobs, gpus, 4)
            per_gpu = {g: Counter(cell(r) for _, g_, _, r in plan if g_ == g) for g in range(gpus)}
            per_lane = {ln: [cell(r) for ln_, _, _, r in plan if ln_ == ln] for ln in range(lanes)}
            for c in per_gpu.values():
                assert len(c) == 12
                if gpus == 1:
                    assert set(c.values()) == {10 // k}           # every cell 10/K times a box
                elif gpus == 2:
                    assert set(c.values()) == {5}
                else:
                    assert set(c.values()) == {2, 3} and sum(c.values()) == 30
            if (gpus, k) == (1, 1):
                assert all(len(x) == 30 and set(Counter(x).values()) <= {2, 3}
                           and len(set(x)) == 12 for x in per_lane.values())
            if gpus == 2:
                assert all(len(x) == 15 and len(set(x)) == 11 for x in per_lane.values())
            if gpus == 4:
                assert sorted(len(x) for x in per_lane.values()) == [7] * 8 + [8] * 8
                assert all(len(set(x)) == len(x) for x in per_lane.values())
            for w in range(0, len(plan), lanes):
                wave = [r for _, _, _, r in plan[w:w + lanes]]
                assert {r["variant"] for r in wave} == {"v1", "v3"}
                assert len({r["arm"] for r in wave}) >= 2


def test_expected_args_are_the_trainers_parse_of_each_cmd(ga, rl, mods):
    """expected_args is what train_dr.py's own argument parser (rebuilt from its source, as
    round_manifest.py check does) makes of each run's cmd; the two computed VG defaults are
    ppo.ValueGradient's."""
    from apexzero.agents.ppo import ValueGradient

    rm, h4r = mods["round_manifest"], mods["h4r"]
    source = (SCRIPTS / "train_dr.py").read_text()
    parser, defaults = rm.trainer_parser(source), ga.trainer_defaults(source)
    for r in rl.runs():
        got = vars(parser.parse_args(r["cmd"][1:]))
        design = ga.design_args(r["variant"], r["arm"], r["seed"])
        assert design.keys() <= got.keys()
        assert h4r._diff("args", {k: got[k] for k in design}, design) == []
        want = ga.expected_args(r["variant"], r["arm"], r["seed"], defaults)
        assert got.keys() == want.keys()
        assert got["vg_lambda"] is rm.COMPUTED and got["vg_winsor"] is rm.COMPUTED
        got |= {"vg_lambda": ga.VG_LAMBDA, "vg_winsor": ga.VG_WINSOR}
        assert h4r._diff("args", got, want) == []
    assert (ValueGradient().lam, ValueGradient().winsor) == (ga.VG_LAMBDA, ga.VG_WINSOR)


def test_ppo_block_is_train_drs_default(ga):
    from dataclasses import asdict

    from apexzero.agents.ppo import PPOConfig

    cfg = PPOConfig(num_envs=2048, num_steps=64, total_timesteps=int(200e6), network="mlp",
                    gamma=0.995, gae_lambda=0.95, actor_width=256, critic_width=256,
                    mlp_layers=3, activation="tanh", ensemble_heads=5)
    assert asdict(cfg) == ga.PPO
    assert ga.FINAL_TIMESTEPS == 199_884_800 and ga.FINAL_CKPT == "000199884800.pkl"


def defaults(ga):
    return ga.trainer_defaults((SCRIPTS / "train_dr.py").read_text())


def test_config_problems_name_a_wrong_flag(ga):
    d = defaults(ga)
    cfg = {"ppo": dict(ga.PPO), "args": ga.expected_args("v3", "sl-f2", 24, d), "env": {}}
    assert ga.config_problems(cfg, "v3", "sl-f2", 24, d) == []
    bad = json.loads(json.dumps(cfg))
    bad["args"]["reexpress"] = True
    bad["args"]["stat_widths_from"] = ",".join(R_DEV)
    bad["vg"] = {"coef": 0.1}
    probs = ga.config_problems(bad, "v3", "sl-f2", 24, d)
    assert any("reexpress" in p for p in probs) and any("stat_widths_from" in p for p in probs)
    assert any("config blocks" in p for p in probs)
    assert ga.config_problems(cfg, "v1", "sl-f2", 24, d)      # v3's tilt is not v1's
    extra = json.loads(json.dumps(cfg))
    extra["args"]["an_option_added_later"] = False                # not in the tag's trainer
    assert ga.config_problems(extra, "v3", "sl-f2", 24, d) == [
        "args.an_option_added_later = False is not in design Section 2"]


# ------------------------------------------------------------------------- the exact counts


def listed(groups: dict) -> tuple[int, int, int]:
    """(T* >= T, T* <= T, all) by listing every joint relabelling, in exact fractions."""
    obs = sum(Fraction(sum(t), len(t)) - Fraction(sum(b), len(b)) for t, b in groups.values())
    per = []
    for t, b in groups.values():
        v, k = list(t) + list(b), len(t)
        tot = sum(v)
        per.append([Fraction(sum(v[i] for i in c), k)
                    - Fraction(tot - sum(v[i] for i in c), len(v) - k)
                    for c in itertools.combinations(range(len(v)), k)])
    ts = [sum(x) for x in itertools.product(*per)]
    return sum(x >= obs for x in ts), sum(x <= obs for x in ts), len(ts)


def test_subset_sum_counts_against_listing(ga):
    rng = random.Random(3)
    for _ in range(50):
        vals = [rng.randint(0, 12) for _ in range(rng.randint(1, 9))]
        k = rng.randint(0, len(vals))
        want = Counter(sum(c) for c in itertools.combinations(vals, k))
        assert ga.subset_sum_counts(vals, k) == dict(want)


def test_exact_counts_equal_full_enumeration(ga):
    """Counting gives exactly the counts that listing every relabelling gives, for one or two
    variants and up to 5 units per arm, arms of unequal size included (missing runs)."""
    rng = random.Random(1)
    cases = 0
    for _ in range(150):
        variants = ("v1", "v3")[:rng.randint(1, 2)]
        top = 5 if rng.random() < 0.1 else 4
        groups = {v: ([rng.randint(0, 12) for _ in range(rng.randint(3, top))],
                      [rng.randint(0, 12) for _ in range(rng.randint(3, top))])
                  for v in variants}
        c = ga.exact_counts(groups)
        assert (c["ge"], c["le"], c["total"]) == listed(groups), groups
        cases += 1
    # ties: every unit equal, so every relabelling ties the observed T
    c = ga.exact_counts({"v1": ([4, 4, 4], [4, 4, 4]), "v3": ([0, 0, 0], [0, 0, 0, 0])})
    assert c["ge"] == c["le"] == c["total"] == 20 * 35 and c["T"] == 0
    assert cases == 150


def test_exact_counts_agree_with_the_power_script_at_10_against_10(ga, mods):
    import numpy as np

    rng = random.Random(7)
    for _ in range(20):
        groups = {v: ([rng.randint(0, 12) for _ in range(10)],
                      [rng.randint(0, 12) for _ in range(10)]) for v in ("v1", "v3")}
        c = ga.exact_counts(groups)
        p, q, n = mods["gw3_power"].exact_p({v: (np.array(t), np.array(b))
                                             for v, (t, b) in groups.items()})
        assert n == c["total"] == math.comb(20, 10) ** 2
        assert (c["ge"] / c["total"], c["le"] / c["total"]) == (p, q)


def test_known_p_values_and_the_boundary(ga):
    def test(t, b, variants=("v1",)):
        units = {"x": {v: dict(enumerate(t)) for v in variants},
                 "y": {v: dict(enumerate(b)) for v in variants}}
        return ga.permutation_test(units, "x", "y", variants)

    r = test([3, 4, 5], [0, 1, 2])          # the largest of C(6, 3) = 20 sums: p = 1/20
    assert r["p"] == Fraction(1, 20) and r["outcome"] == "confirm"   # p <= 0.05 confirms
    assert test([0, 1, 2], [3, 4, 5])["outcome"] == "refute"
    r = test([5, 5, 5], [5, 5, 5])
    assert (r["p"], r["p_opposite"], r["outcome"]) == (1, 1, "inconclusive")
    r = test([3, 4, 5], [0, 1, 2], ("v1", "v3"))
    assert r["p"] == Fraction(1, 400) and r["relabellings"] == 400
    r = test([3, 4], [0, 1, 2])
    assert r["outcome"] == "not testable" and r["p"] is None
    with pytest.raises(ValueError, match="integers"):
        test([3, 4, 13], [0, 1, 2])


# ------------------------------------------------------------------------- planted networks


def net(v, arm, s, done, gap):
    """A planted network: completion 0.6 on ``done`` and exactly 0.5 elsewhere (not completed:
    the rule is completion > 0.5), its gap (points) from ``gap(circuit)`` where completed."""
    return {"variant": v, "arm": arm, "seed": s,
            "completion": {c: 0.6 if c in done else 0.5 for c in R_DEV},
            "gap": {c: gap(c) / 100 if c in done else NAN for c in R_DEV}}


def planted(rl, value, gap=None, drop=()):
    """Every GW-3 network with chosen units: ``value(variant, unit, seed)`` for unit base, sa,
    sl or cl. base and sa networks complete the first n R-dev circuits; a leave-out pair's unit
    u is min(u, 6) held-out circuits of its f1 network and the rest of its f2 network's. Every
    fold network also completes its own fold's 6 circuits, in sample, which no unit counts."""
    gap = gap or (lambda v, arm, s, c: 7.0)
    out = {}
    for v in rl.VARIANT_FLAGS:
        for s in rl.SEEDS:
            for arm in rl.ARM_FLAGS:
                name = rl.run_name(v, arm, s)
                if name in drop:
                    continue
                if arm in ("base", "sa"):
                    done = set(R_DEV[:value(v, arm, s)])
                else:
                    x, f = arm.split("-")
                    u = value(v, x, s)
                    k = min(u, 6) if f == "f1" else u - min(u, 6)
                    held = rl.FOLDS["f2" if f == "f1" else "f1"]
                    done = set(held[:k]) | set(rl.FOLDS[f])
                out[name] = net(v, arm, s, done, lambda c, v=v, arm=arm, s=s: gap(v, arm, s, c))
    return out


def by_unit(**units):
    """value() from {unit: n}, one value for every variant and seed (default 5)."""
    return lambda v, u, s: units.get(u, 5)


def report():
    return REPORT.read_bytes()


def test_null_effect(ga, rl):
    res = ga.analyse_runs(planted(rl, by_unit()), {}, report())
    for k in ("P1", "S1", "S2"):
        t = res["primary" if k == "P1" else "secondary"][k]
        assert (t["T"], t["p"], t["p_opposite"], t["outcome"]) == (0, 1, 1, "inconclusive")
        assert t["relabellings"] == math.comb(20, 10) ** 2
    assert res["secondary"]["S1-v1"]["relabellings"] == math.comb(20, 10)
    assert res["runs"] == {"expected": 120, "present": 120, "missing": {}}


def test_strong_effect_and_in_sample_circuits_do_not_count(ga, rl):
    """sl's units 12, base's 2: only the observed labelling reaches T in each variant. cl's
    networks complete their own fold's 6 circuits, but their held-out unit is 2, like base."""
    res = ga.analyse_runs(planted(rl, by_unit(sl=12, base=2, cl=2, sa=2)), {}, report())
    p1 = res["primary"]["P1"]
    assert p1["outcome"] == "confirm" and p1["count_ge"] == 1
    assert p1["p"] == Fraction(1, math.comb(20, 10) ** 2) and p1["T"] == 20
    assert res["secondary"]["S1"]["outcome"] == "inconclusive"
    assert res["secondary"]["S1"]["p"] == 1
    counts = res["descriptive"]["counts"]["v1/cl-f1"]
    assert counts["held_out"]["mean"] == 2 and counts["in_sample_fold"]["mean"] == 6
    assert counts["all_12"]["mean"] == 8
    refute = ga.analyse_runs(planted(rl, by_unit(sl=0, base=6)), {}, report())
    assert refute["primary"]["P1"]["outcome"] == "refute"


def test_planted_units_match_an_independent_count(ga, rl, mods):
    """Random units: P1, S1-v1 and S2 equal gw3_power.exact_p on the same planted units."""
    import numpy as np

    rng = random.Random(11)
    table = {(v, u, s): rng.randint(0, 12) for v in ("v1", "v3") for u in ("base", "sl", "cl",
                                                                            "sa")
             for s in rl.SEEDS}
    res = ga.analyse_runs(planted(rl, lambda v, u, s: table[v, u, s]), {}, report())
    for name, (t, variants) in {"P1": ("sl", ("v1", "v3")), "S1-v1": ("cl", ("v1",)),
                                "S2": ("sa", ("v1", "v3"))}.items():
        groups = {v: (np.array([table[v, t, s] for s in rl.SEEDS]),
                      np.array([table[v, "base", s] for s in rl.SEEDS])) for v in variants}
        p, q, _ = mods["gw3_power"].exact_p(groups)
        got = res["primary"].get(name) or res["secondary"][name]
        assert (float(got["p"]), float(got["p_opposite"])) == (p, q)


def test_missing_runs_remove_their_units(ga, rl):
    """A missing half removes the pair's unit, and p counts every relabelling of what is left;
    fewer than 3 units in an arm of a variant is not testable (not confirmed)."""
    nets = planted(rl, by_unit(sl=9, base=3))
    gone = "gw3-v1-sl-f1-s20"
    res = ga.analyse_runs({k: v for k, v in nets.items() if k != gone}, {gone: "failed twice"},
                          report())
    p1 = res["primary"]["P1"]
    assert p1["n_units"]["v1"] == {"treated": 9, "baseline": 10}
    assert p1["relabellings"] == math.comb(19, 9) * math.comb(20, 10)
    assert p1["outcome"] == "confirm"
    assert res["descriptive"]["units"]["sl"]["v1"].keys() == set(range(21, 30))
    assert res["descriptive"]["counts"]["v1/sl-f2"]["held_out"]["n"] == 10   # its other half
    assert res["runs"]["missing"] == {gone: "failed twice"} and res["runs"]["present"] == 119
    drop = {rl.run_name("v3", "base", s) for s in range(20, 28)}
    res = ga.analyse_runs({k: v for k, v in nets.items() if k not in drop}, {}, report())
    for k in ("S1", "S1-v3", "S2"):
        assert res["secondary"][k]["outcome"] == "not testable"
    assert res["primary"]["P1"]["outcome"] == "not testable"
    assert res["secondary"]["S1-v1"]["outcome"] != "not testable"
    assert res["gw4"]["criteria"]["1"]["holds"] is False


def test_zero_completion_networks(ga, rl):
    """A network that completes nothing counts 0 in its unit and has no median gap: it is left
    out of Delta-gap and counted. If every sa network completes nothing, Delta-gap is undefined
    and criterion 3 does not hold."""
    nets = planted(rl, by_unit(sa=0))
    res = ga.analyse_runs(nets, {}, report())
    lap = res["lap_time"]
    assert lap["delta_gap"] is None and lap["pooled"]["sa"]["n_left_out"] == 20
    assert lap["pooled"]["sa"]["left_out_per_variant"] == {"v1": 10, "v3": 10}
    assert res["gw4"]["criteria"]["3"]["holds"] is False
    assert res["secondary"]["S2"]["n_units"]["v1"] == {"treated": 10, "baseline": 10}
    assert res["secondary"]["S2"]["outcome"] == "refute"          # 0 against 5 everywhere
    one = planted(rl, lambda v, u, s: 0 if (v, u, s) == ("v3", "base", 23) else 5)
    lap = ga.analyse_runs(one, {}, report())["lap_time"]
    assert lap["pooled"]["base"] == {**lap["pooled"]["base"], "n": 20, "n_left_out": 1}
    assert lap["pooled"]["base"]["left_out_per_variant"] == {"v1": 0, "v3": 1}
    assert lap["delta_gap"] == 0.0


def test_lap_time_statistics(ga, rl):
    """sa completes 8 circuits and base 4, at the same per-circuit gaps: the per-network medians
    differ by the circuit mix alone (Delta-gap 1.0, which criterion 3 does not pass), while on
    the circuits both complete the pairs differ by 0."""
    def gap(v, arm, s, c):
        return 6.0 + R_DEV.index(c) / 2

    res = ga.analyse_runs(planted(rl, by_unit(sa=8, base=4), gap), {}, report())
    lap = res["lap_time"]
    assert lap["delta_gap"] == pytest.approx(1.0)
    assert lap["pooled"]["sa"]["mean"] == pytest.approx(7.75)
    assert lap["pooled"]["base"]["mean"] == pytest.approx(6.75)
    assert res["gw4"]["criteria"]["3"]["holds"] is False
    assert lap["paired_by_seed"]["mean_over_pairs"]["diff_of_medians"]["pooled"] == 0.0
    assert lap["paired_by_seed"]["mean_over_pairs"]["mean_of_diffs"]["v1"] == 0.0
    assert len(lap["paired_by_seed"]["pairs"]) == 20
    rows = lap["per_circuit"]["v1"]
    assert rows["n_circuits"] == 4 and rows["median_delta"] == 0.0
    shifted = ga.analyse_runs(planted(rl, by_unit(sa=6, base=6),
                                      lambda v, arm, s, c: 7.5 if arm == "sa" else 7.0), {},
                              report())["lap_time"]
    assert shifted["delta_gap"] == pytest.approx(0.5)
    assert shifted["per_variant"]["v3"]["delta"] == pytest.approx(0.5)
    assert shifted["per_circuit"]["v3"]["n_positive"] == 6


def test_s3_and_d2(ga, rl):
    """sl-f1 6 and sl-f2 2 held-out circuits against cl-f1 4 and cl-f2 0: each cell's difference
    2 with no spread, D2 2. The boundary D2 = -0.5 holds."""
    s3 = ga.analyse_runs(planted(rl, by_unit(sl=8, cl=4)), {}, report())["S3"]
    assert s3["D2"] == 2
    for c in s3["cells"].values():
        assert c["diff"] == 2 and c["ci95"] == [2.0, 2.0] and c["draws_used"] == ga.BOOT_N
    edge = ga.analyse_runs(planted(rl, by_unit(sl=5, cl=6)), {}, report())
    assert edge["S3"]["D2"] == Fraction(-1, 2) and edge["gw4"]["criteria"]["2"]["holds"]
    worse = ga.analyse_runs(planted(rl, by_unit(sl=2, cl=12)), {}, report())
    assert worse["S3"]["D2"] == -5 and not worse["gw4"]["criteria"]["2"]["holds"]
    spread = ga.analyse_runs(planted(rl, lambda v, u, s: (s - 20) % 7 if u == "sl" else 3), {},
                             report())["S3"]["cells"]["v1/f1"]
    lo, hi = spread["ci95"]
    assert lo < float(spread["diff"]) < hi


def test_descriptive_tables(ga, rl):
    res = ga.analyse_runs(planted(rl, by_unit(base=10, sa=11, sl=12)), {}, report())
    d = res["descriptive"]
    assert d["c_at_least_10_of_12"]["label"].startswith("not the gate")
    assert d["c_at_least_10_of_12"]["per_cell"]["v1/base"] == {"n": 10, "c_at_least_10": 10}
    assert d["c_at_least_10_of_12"]["per_cell"]["v1/cl-f1"]["c_at_least_10"] == 10  # 5 + 6
    assert d["c_at_least_10_of_12"]["per_cell"]["v1/cl-f2"]["c_at_least_10"] == 0   # 0 + 6
    shares = d["per_circuit_share"]["shares"]
    assert shares["v3/base"]["Norisring"]["share"] == 1.0          # 4th of R_DEV, within 10
    assert shares["v3/base"]["Zandvoort"]["share"] == 0.0
    assert shares["v1/sl-f1"]["Norisring"]["held_out"] is True
    assert d["per_circuit_share"]["named"] == ["Norisring", "YasMarina", "Sochi", "Melbourne"]
    assert d["per_seed_pairs_sl_base"]["v1"][25] == {"sl": 12, "base": 10, "diff": 2}
    assert res["gate_80pct"]["read_here"] is False


# ------------------------------------------------------------------------- GW-4's rule


def decisions(res):
    g = res["gw4"]
    return g["if_criterion_5_holds"], g["if_criterion_5_fails"], g["in_sample_label"]


def test_gw4_adopts_when_criteria_1_to_4_hold(ga, rl):
    res = ga.analyse_runs(planted(rl, by_unit(sl=12, base=2, cl=2, sa=12)), {}, report())
    g = res["gw4"]
    assert g["criteria_1_to_4_hold"] is True and g["s1_v1"] == "inconclusive"
    assert decisions(res) == ("adopt", "default", {"if_criterion_5_holds": None,
                                                   "if_criterion_5_fails": None})
    assert g["criteria"]["5"]["holds"] is None and len(g["criteria"]["5"]["tests"]) == 9
    both = ga.analyse_runs(planted(rl, by_unit(sl=12, base=2, cl=12)), {}, report())
    assert decisions(both)[:2] == ("adopt", "copied")
    assert decisions(both)[2]["if_criterion_5_fails"] == "in-sample for widths"


def test_gw4_copied_fallback_and_its_label(ga, rl):
    res = ga.analyse_runs(planted(rl, by_unit(sl=2, base=2, cl=12)), {}, report())
    assert res["primary"]["P1"]["outcome"] == "inconclusive"
    assert res["secondary"]["S1-v1"]["outcome"] == "confirm"
    holds, fails, label = decisions(res)
    assert (holds, fails) == ("copied", "copied")
    assert label == {"if_criterion_5_holds": "in-sample for widths",
                     "if_criterion_5_fails": "in-sample for widths"}
    assert res["gw4"]["in_sample_label_covers"]


def test_gw4_default_widths(ga, rl):
    res = ga.analyse_runs(planted(rl, by_unit()), {}, report())
    assert decisions(res)[:2] == ("default", "default")
    # copied widths transfer in v3 only: S1 pooled may confirm, S1-v1 does not, so no copying
    v3 = ga.analyse_runs(planted(rl, lambda v, u, s: 12 if (v, u) == ("v3", "cl") else 2), {},
                         report())
    assert v3["secondary"]["S1-v3"]["outcome"] == "confirm"
    assert v3["secondary"]["S1"]["outcome"] == "confirm"
    assert decisions(v3)[:2] == ("default", "default")


def test_gw4_falls_back_when_a_criterion_fails(ga, rl):
    """P1 confirms, but the lap gap rises by more than a point (criterion 3) or the stat model
    is far below the copied one (criterion 2): (2a) fails the rule."""
    slow = ga.analyse_runs(planted(rl, by_unit(sl=12, base=2, cl=2, sa=6),
                                   lambda v, arm, s, c: 8.5 if arm == "sa" else 7.0), {},
                           report())
    assert slow["primary"]["P1"]["outcome"] == "confirm"
    assert slow["lap_time"]["delta_gap"] == pytest.approx(1.5)
    assert decisions(slow)[:2] == ("default", "default")
    far = ga.analyse_runs(planted(rl, by_unit(sl=9, base=2, cl=12)), {}, report())
    assert far["primary"]["P1"]["outcome"] == "confirm" and far["S3"]["D2"] < -0.5
    assert decisions(far)[:2] == ("copied", "copied")


# ------------------------------------------------------------------------- the width report


def test_committed_width_report_meets_criterion_4(ga):
    raw = report()
    assert hashlib.sha256(raw).hexdigest() == ga.WIDTH_REPORT_SHA256
    c4 = ga.criterion_4(raw)
    assert c4["holds"] is True and c4["n_checks"] == 90
    assert sorted(c4["failed"]) == sorted(ga.C4_V3_ALLOWED)
    assert ga.WIDTH_REPORT_SHA256 in section(5)


def test_width_report_sha256_mismatch_stops_gw4(ga, rl):
    for bad in (report() + b"\n", None):
        c4 = ga.criterion_4(bad)
        assert c4["holds"] is None and "stops for an amendment" in c4["stopped"]
        res = ga.analyse_runs(planted(rl, by_unit(sl=12, base=2)), {}, bad)
        g = res["gw4"]
        assert g["stopped"] and g["if_criterion_5_holds"] is None
        assert g["if_criterion_5_fails"] is None
        assert res["primary"]["P1"]["outcome"] == "confirm"     # Section 3 is still computed


def edited_report(edit) -> bytes:
    r = json.loads(report())
    edit(r)
    return json.dumps(r).encode()


def _fail(r, key):
    fold, v, stat = key.split("/")
    if stat == "nocopy":
        r["models"]["2a"][fold][v]["nocopy"]["ok"] = False
    else:
        next(c for c in r["models"]["2a"][fold][v]["checks"] if c["stat"] == stat)["ok"] = False
    r["verdict"]["2a"]["failed"].append(key)


@pytest.mark.parametrize("key", ["f1/v1/share_1m", "f2/v1/apex_dip:<25", "f2/v3/spectrum:8-16",
                                 "f1/v3/w1_all", "all/v3/nocopy", "f2/v1/nocopy"])
def test_criterion_4_fails_outside_the_restated_misses(ga, key):
    raw = edited_report(lambda r: _fail(r, key))
    c4 = ga.criterion_4(raw, pin=hashlib.sha256(raw).hexdigest())
    assert c4["holds"] is False and key in c4["failed"]


def test_criterion_4_refuses_a_report_that_contradicts_its_verdict(ga):
    def edit(r):
        r["models"]["2a"]["f1"]["v1"]["checks"][0]["ok"] = False       # verdict not updated

    raw = edited_report(edit)
    with pytest.raises(ga.CheckFailed, match="verdict"):
        ga.criterion_4(raw, pin=hashlib.sha256(raw).hexdigest())


def test_criterion_4_holds_with_fewer_misses(ga):
    def edit(r):
        next(c for c in r["models"]["2a"]["f1"]["v3"]["checks"]
             if c["stat"] == "share_1m")["ok"] = True
        r["verdict"]["2a"]["failed"].remove("f1/v3/share_1m")

    raw = edited_report(edit)
    c4 = ga.criterion_4(raw, pin=hashlib.sha256(raw).hexdigest())
    assert c4["holds"] is True and len(c4["failed"]) == 2


# ------------------------------------------------------------------------- scratch repositories

GATE_FILES = ("scripts/gw3_analysis.py", "scripts/gw3_runs.py", "scripts/h4r.py",
              "scripts/fields.py", "scripts/_bootstrap.py", "scripts/round_manifest.py",
              "scripts/train_dr.py", "docs/prereg/gw3-width-round.md",
              "outputs/width-report/report.json", "src/apexzero/tracks/corners.py")


class Scratch:
    """A bare ``origin`` and a clone of it in tmp_path, with real git and no user config,
    holding copies of the files the gate names (and round_manifest.py and train_dr.py, which
    the shared layer reads). ``ga.ROOT`` points at the clone."""

    def __init__(self, tmp_path: Path, monkeypatch, ga):
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
        for rel in GATE_FILES:
            (self.clone / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, self.clone / rel)
        self.git("add", "-A")
        self.git("commit", "-qm", "first")
        self.git("tag", "-a", "gw3-prereg-v10", "-m", "a different tag")
        self.git("push", "-q", "origin", "HEAD:refs/heads/main", "--tags")
        monkeypatch.setattr(ga, "ROOT", self.clone)

    def git(self, *args) -> str:
        return subprocess.run(["git", *args], cwd=self.clone, check=True, capture_output=True,
                              text=True).stdout.strip()

    def tag(self, annotated: bool = True) -> None:
        self.git("tag", *(["-a", "-m", "GW-3"] if annotated else []), "gw3-prereg-v1")
        self.git("push", "-q", "origin", "refs/tags/gw3-prereg-v1")


@needs_git
def test_tag_gate_refuses_without_the_tag(ga, tmp_path, monkeypatch):
    Scratch(tmp_path, monkeypatch, ga)
    with pytest.raises(PermissionError, match="not on origin"):    # only gw3-prereg-v10 exists
        ga.tag_gate()


@needs_git
def test_tag_gate_refuses_a_lightweight_tag(ga, tmp_path, monkeypatch):
    repo = Scratch(tmp_path, monkeypatch, ga)
    repo.tag(annotated=False)
    with pytest.raises(PermissionError, match="not an annotated tag"):
        ga.tag_gate()


@needs_git
def test_tag_gate_passes_on_the_tag_and_refuses_a_changed_file(ga, tmp_path, monkeypatch):
    repo = Scratch(tmp_path, monkeypatch, ga)
    repo.tag()
    got = ga.tag_gate()
    assert got["tag_object"] == repo.git("rev-parse", "refs/tags/gw3-prereg-v1")
    assert got["commit"] == repo.git("rev-parse", "HEAD") == got["head"]
    (repo.clone / "notes.txt").write_text("later work\n")         # outside the gated files
    repo.git("add", "notes.txt")
    repo.git("commit", "-qm", "later")
    assert ga.tag_gate()["commit"] == got["commit"]
    runs = repo.clone / "scripts" / "gw3_runs.py"
    runs.write_text(runs.read_text() + "# changed\n")
    with pytest.raises(PermissionError, match="differ.*gw3_runs.py"):
        ga.tag_gate()
    repo.git("checkout", "--", "scripts/gw3_runs.py")
    repo.git("rm", "-q", "--cached", "docs/prereg/gw3-width-round.md")
    with pytest.raises(PermissionError, match="differ|untracked"):
        ga.tag_gate()


@needs_git
def test_tag_gate_refuses_a_local_tag_that_is_not_origins(ga, tmp_path, monkeypatch):
    repo = Scratch(tmp_path, monkeypatch, ga)
    repo.tag()
    repo.git("tag", "-d", "gw3-prereg-v1")
    repo.git("tag", "-a", "-m", "another GW-3", "gw3-prereg-v1", "HEAD")
    with pytest.raises(PermissionError, match="not origin's tag object"):
        ga.tag_gate()


@needs_git
def test_tag_gate_refuses_a_head_off_the_tag(ga, tmp_path, monkeypatch):
    repo = Scratch(tmp_path, monkeypatch, ga)
    repo.tag()
    repo.git("checkout", "-q", "--orphan", "elsewhere")
    repo.git("commit", "-qm", "unrelated")
    with pytest.raises(PermissionError, match="does not descend"):
        ga.tag_gate()


# ------------------------------------------------------------------------- end to end

OCP = {c: 60.0 + i for i, c in enumerate(R_DEV)}
CKPTS = [f"{t:012d}.pkl" for t in (10_000_000, 12_210_553, 14_909_760, 18_205_642, 22_230_095,
                                    27_144_176, 33_144_540, 40_471_316, 49_417_715, 60_341_763,
                                    73_680_629, 89_968_123, 109_856_054, 134_140_317,
                                    163_792_745, 199_884_800)]


def plant_runs(ga, rl, clone: Path, nets: dict, status: dict | None = None,
               skip: tuple = ()) -> None:
    """runs/ in ``clone`` as four GPUs at 4 lanes would leave it: each run's config, evaluation
    and 16 checkpoints, the status file, the box's sha256 list, a queue log, and the box's
    manifest from the real ``round_manifest.py write`` with a stand-in nvidia-smi listing four
    A6000s. ``status`` overrides a run's events ('retried', 'failed' or 'not run')."""
    status = status or {}
    runs_dir = clone / "runs"
    runs_dir.mkdir(exist_ok=True)
    lines, sums, d_args = [], [], defaults(ga)
    for j, r in enumerate(rl.runs()):
        name, lane = r["name"], j % 16
        st = status.get(name, "done")
        if st == "not run":
            continue
        t0 = f"2026-10-28T{j // 16:02d}:00:00Z"
        t1 = f"2026-10-28T{j // 16:02d}:26:00Z"
        events = {"done": ["start", "done"], "retried": ["start", "retry", "retried"],
                  "failed": ["start", "retry", "failed"]}[st]
        lines += [f"{name}\t{e}\t{t0 if e == 'start' else t1}\t{lane}\t{lane % 4}"
                  for e in events]
        d = runs_dir / name
        d.mkdir()
        cfg = {"ppo": ga.PPO, "args": ga.expected_args(r["variant"], r["arm"], r["seed"], d_args),
               "env": {"dt": 0.01, "grip": 1.0}}
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        (runs_dir / f"{name}.log").write_text("trained\n")
        if st == "retried":
            (runs_dir / f"{name}.log.1").write_text("first attempt\n")
        if st == "failed":
            continue
        n = nets[name]
        ev = {}
        for c in R_DEV:
            done = n["completion"][c] > 0.5
            lap = OCP[c] * (1 + n["gap"][c]) if done else NAN
            ev[c] = {"completion": n["completion"][c], "median_lap_s": lap, "ocp_lap_s": OCP[c],
                     "gap": lap / OCP[c] - 1.0 if done else NAN}
        (d / "eval_rdev.json").write_text(json.dumps(ev, indent=1))
        (d / "ckpt").mkdir()
        for k in CKPTS:
            (d / "ckpt" / k).write_bytes(f"{name} {k}".encode())
        for rel in ("config.json", "eval_rdev.json", "ckpt/000199884800.pkl"):
            if f"{name}/{rel}" not in skip:
                sha = hashlib.sha256((d / rel).read_bytes()).hexdigest()
                sums.append(f"{sha}  {name}/{rel}")
    (runs_dir / "gw3-status-0of1.tsv").write_text("\n".join(lines) + "\n")
    (runs_dir / "gw3-sums-0of1.txt").write_text("\n".join(sums) + "\n")
    (runs_dir / "gw3-queue-0of1.log").write_text("queue\n")
    bin_dir = clone.parent / "bin"
    bin_dir.mkdir(exist_ok=True)
    smi = bin_dir / "nvidia-smi"
    smi.write_text("#!/bin/sh\nfor i in 1 2 3 4; do echo 'NVIDIA RTX A6000'; done\n")
    smi.chmod(0o755)
    env = {**os.environ, "BREV_TYPE": "massedcompute_A6000_plusx4",
           "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    subprocess.run([sys.executable, "scripts/round_manifest.py", "write", "--round", "gw3",
                    "--runs-script", "scripts/gw3_runs.py", "--shard", "0/1", "--gpus", "4",
                    "--lanes-per-gpu", "4"], cwd=clone, env=env, check=True, capture_output=True)


@pytest.fixture
def scratch(ga, tmp_path, monkeypatch, mods):
    repo = Scratch(tmp_path, monkeypatch, ga)
    repo.tag()
    monkeypatch.setattr(mods["h4r"], "_pickle_timesteps", lambda path, name: int(Path(path).stem))
    monkeypatch.setattr(ga, "ocp_laps", lambda: (OCP, {f"outputs/ocp/{c}.pkl": "0" * 64
                                                       for c in R_DEV}))
    return repo


@needs_git
def test_manifest_and_analysis_end_to_end(ga, rl, scratch):
    """The real round_manifest.py writes the box's manifest and checks it; the manifest stage
    records every run; once committed, the analysis runs once and refuses a second time, and
    after its output is committed and removed."""
    repo = scratch
    nets = planted(rl, by_unit(sl=12, base=2, cl=2, sa=12))
    plant_runs(ga, rl, repo.clone, nets, status={"gw3-v1-base-s21": "retried"})
    with pytest.raises(ga.CheckFailed, match="committed"):
        ga.main(["analyse"])                                    # no manifest yet
    ga.main(["manifest"])
    man = json.loads((repo.clone / ga.MANIFEST_REL).read_text())
    assert len(man["present"]) == 120 and man["missing"] == {} and man["check"]["returncode"] == 0
    assert man["retries"] == {"gw3-v1-base-s21": {"first_attempt_log": True}}
    assert man["launch_commit"] == [repo.git("rev-parse", "gw3-prereg-v1^{commit}")]
    box = man["boxes"][0]
    assert (box["box_type"], box["gpu_model"], box["gpu_count"], box["lanes_per_gpu"]) == (
        "massedcompute_A6000_plusx4", "NVIDIA RTX A6000", 4, 4)
    ent = man["runs"]["gw3-v3-sl-f2-s29"]
    assert ent["timesteps"] == 199_884_800 and ent["n_checkpoints"] == 16
    assert ent["checkpoint_path"] == "runs/gw3-v3-sl-f2-s29/ckpt/000199884800.pkl"
    assert man["provenance"]["check_script_is_tagged"] is True
    with pytest.raises(ga.CheckFailed, match="must be committed"):
        ga.main(["analyse"])                                    # not committed yet
    repo.git("add", "-f", ga.MANIFEST_REL)
    repo.git("commit", "-qm", "GW-3: run manifest")
    ga.main(["analyse", "--check"])
    assert not (repo.clone / ga.ANALYSIS_REL).exists()
    ga.main(["analyse"])
    res = json.loads((repo.clone / ga.ANALYSIS_REL).read_text())
    assert res["primary"]["P1"]["outcome"] == "confirm"
    assert res["gw4"]["if_criterion_5_holds"] == "adopt"
    prov = res["provenance"]
    assert prov["tag"] == "gw3-prereg-v1" and prov["inputs_sha256"][ga.MANIFEST_REL]
    assert len([k for k in prov["inputs_sha256"] if k.endswith("eval_rdev.json")]) == 120
    with pytest.raises(ga.CheckFailed, match="runs once"):
        ga.main(["analyse"])
    repo.git("add", "-f", ga.ANALYSIS_REL)
    repo.git("commit", "-qm", "GW-3: analysis")
    (repo.clone / ga.ANALYSIS_REL).unlink()
    with pytest.raises(ga.CheckFailed, match="runs once"):
        ga.main(["analyse"])
    with pytest.raises(ga.CheckFailed, match="not rebuilt"):
        ga.main(["manifest"])


@needs_git
def test_manifest_counts_a_run_that_failed_twice_as_missing(ga, rl, scratch):
    repo = scratch
    gone = "gw3-v3-sl-f1-s24"
    plant_runs(ga, rl, repo.clone, planted(rl, by_unit(sl=9, base=3)), status={gone: "failed"})
    man, problems = ga.build_manifest(ga.tag_gate())
    assert problems == []
    assert man["missing"] == {gone: "failed twice in the queue (runs/gw3-status-0of1.tsv)"}
    assert man["failed_twice"] == [gone] and man["check"]["returncode"] == 1
    tolerated = f"{gone}: status 'failed' in runs/gw3-manifest-0of1.json"
    assert man["check"]["tolerated"] == [tolerated]
    ga.main(["manifest"])
    repo.git("add", "-f", ga.MANIFEST_REL)
    repo.git("commit", "-qm", "GW-3: run manifest")
    ga.main(["analyse"])
    res = json.loads((repo.clone / ga.ANALYSIS_REL).read_text())
    assert res["runs"]["present"] == 119 and list(res["runs"]["missing"]) == [gone]
    assert res["primary"]["P1"]["n_units"]["v3"] == {"treated": 9, "baseline": 10}
    assert res["primary"]["P1"]["relabellings"] == math.comb(19, 9) * math.comb(20, 10)


@needs_git
def test_manifest_refuses_what_it_cannot_explain(ga, rl, scratch):
    """A run the queue never started is refused unless it is accepted as missing (a deviation);
    a config with another flag, and a file whose sha256 is not the box's, are refused."""
    repo = scratch
    nets = planted(rl, by_unit())
    lost = "gw3-v1-sa-s27"
    plant_runs(ga, rl, repo.clone, nets, status={lost: "not run"})
    gate = ga.tag_gate()
    man, problems = ga.build_manifest(gate)
    assert any(p.startswith(f"{lost}: status 'not run'") and "--accept-missing" in p
               for p in problems)
    man, problems = ga.build_manifest(gate, {lost: "the box was lost"})
    assert problems == [] and man["accepted_missing"] == {lost: "the box was lost"}
    with pytest.raises(SystemExit, match="not a GW-3 run"):
        ga.parse_accept(["gw3-v1-sa-s30=gone"])
    shutil.rmtree(repo.clone / "runs")
    plant_runs(ga, rl, repo.clone, nets, skip=("gw3-v3-cl-f2-s20/ckpt/000199884800.pkl",))
    cfg_path = repo.clone / "runs" / "gw3-v1-base-s22" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["args"]["real_widths"] = True
    cfg_path.write_text(json.dumps(cfg))
    _, problems = ga.build_manifest(gate)
    assert any(p.startswith("gw3-v1-base-s22: config: args.real_widths") for p in problems)
    assert any("round_manifest.py check: gw3-v1-base-s22" in p for p in problems)
    assert any(p.startswith("gw3-v3-cl-f2-s20: ckpt/000199884800.pkl: sha256") and
               "none on the box" in p for p in problems)
