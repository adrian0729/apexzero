"""GW-3: power of the primary test P1, by simulation (docs/prereg/gw3-width-round.md, Section 3).

    nice -n 10 uv run python scripts/gw3_power.py

The draft of 26 Oct 2026 computed its power table by simulation and committed no script, so
its method is re-implemented here:
- **Units.** Each unit (a network's, or a same-seed network pair's, count of completed
  circuits of 12) is drawn independently as round(N(mean + gain, SD)), clipped to 0..12.
  The means and SDs are H4R's baselines (v1 5.0 +- 1.58, v3 7.2 +- 1.10; the per-seed counts
  in docs/findings/h4r.md, Section 5.2). Baseline units take gain 0, the stat-model
  leave-out units the row's gain, with the same SD.
- **The test is P1 exactly.** T = sum over variants of (mean leave-out unit - mean baseline
  unit); one-sided exact permutation, labels exchanged within each variant; ties count toward
  p (T* >= T); P1 confirms at p <= 0.05.
- **The exact p is counted, not listed.** Per variant, a dynamic programme over the integer
  units counts the relabellings that give each treated-arm sum. The two variants' counts are
  then combined over every pair of sums, with T compared in integer arithmetic (T times the
  least common multiple of n_t * n_b over variants). This equals full enumeration, which is
  checked below on 200 planted cases with 3 to 5 units per arm, and stays exact at 10 against
  10, where the C(20, 10)^2 = 3.4e10 joint relabellings cannot be listed.

It first reproduces the draft's 5-seed column (0.03, 0.25, 0.37, 0.70 at 1,500 draws; with
--draws-5 20000 it gives the draft's 20,000-draw figures, 0.034, 0.272, 0.353, 0.707), then
computes the same rows at 10 seeds per cell (20,000 draws by default). It also reports
S1-v1's power at 10 seeds (the same test within v1 alone, C(20, 10) = 184,756 relabellings;
the #12 condition of the draft's Section 5). Prints JSON. Runs in about 25 s; reads and
writes no file.
"""

import argparse
import itertools
import json
import math

import numpy as np

MEAN = {"v1": 5.0, "v3": 7.2}
SD = {"v1": 1.58, "v3": 1.10}
ROWS = [(0.0, 0.0), (1.3, 0.3), (1.3, 0.6), (2.6, 0.6)]    # true gain (v1, v3), circuits of 12
DRAFT_5 = [0.03, 0.25, 0.37, 0.70]
S1V1_GAINS = [0.0, 1.3, 2.6]
ALPHA = 0.05
MAX_UNIT = 12


def subset_sum_counts(values, k: int) -> np.ndarray:
    """counts[s] = the number of k-element subsets of ``values`` (integers 0..12) summing to s."""
    vals = [int(v) for v in values]
    top = sum(vals)
    n = np.zeros((k + 1, top + 1), dtype=np.int64)
    n[0, 0] = 1
    for v in vals:
        shifted = np.zeros_like(n)
        shifted[1:, v:] = n[:-1, :top + 1 - v]
        n = n + shifted
    return n[k]


def exact_p(groups: dict) -> tuple[float, float, int]:
    """(p, opposite p, number of relabellings) of the within-variant exact permutation test.

    ``groups`` maps a variant to (treated units, baseline units), both integers. T is the sum
    over variants of mean(treated) - mean(baseline); p counts relabellings with T* >= T, the
    opposite p those with T* <= T, the observed labelling included in both."""
    scale = 1
    for t, b in groups.values():
        scale = math.lcm(scale, len(t) * len(b))
    dist = {0: 1}
    obs = 0
    for t, b in groups.values():
        nt, nb = len(t), len(b)
        weight = scale * (nt + nb) // (nt * nb)        # scale * (1/nt + 1/nb)
        offset = scale // nb * (int(sum(t)) + int(sum(b)))
        obs += int(sum(t)) * weight - offset
        counts = subset_sum_counts(list(t) + list(b), nt)
        sums = np.nonzero(counts)[0]
        new: dict[int, int] = {}
        for a, ca in dist.items():
            for s in sums:
                key = a + int(s) * weight - offset
                new[key] = new.get(key, 0) + ca * int(counts[s])
        dist = new
    total = sum(dist.values())
    ge = sum(c for key, c in dist.items() if key >= obs)
    le = sum(c for key, c in dist.items() if key <= obs)
    return ge / total, le / total, total


def enumerated_p(groups: dict) -> tuple[float, float, int]:
    """The same test by listing every joint relabelling (as apexzero.fields.analysis does)."""
    obs, dist = 0.0, np.zeros(1)
    for t, b in groups.values():
        v = np.concatenate([np.asarray(t, float), np.asarray(b, float)])
        n, k, tot = len(v), len(t), v.sum()
        d = np.array([v[list(c)].sum() / k - (tot - v[list(c)].sum()) / (n - k)
                      for c in itertools.combinations(range(n), k)])
        obs += float(np.mean(t) - np.mean(b))
        dist = (dist[:, None] + d[None, :]).ravel()
    eps = 1e-12 * max(1.0, float(np.max(np.abs(dist))))
    return float(np.mean(dist >= obs - eps)), float(np.mean(dist <= obs + eps)), len(dist)


def check_against_enumeration(n_cases: int = 200, seed: int = 1) -> int:
    """Planted integer data, 3-5 units per arm and variant: counting equals listing."""
    rng = np.random.default_rng(seed)
    for _ in range(n_cases):
        groups = {v: (rng.integers(0, MAX_UNIT + 1, rng.integers(3, 6)),
                      rng.integers(0, MAX_UNIT + 1, rng.integers(3, 6))) for v in MEAN}
        a, b = exact_p(groups), enumerated_p(groups)
        if a[2] != b[2] or abs(a[0] - b[0]) > 1e-12 or abs(a[1] - b[1]) > 1e-12:
            raise AssertionError(f"counting and enumeration differ: {groups}, {a}, {b}")
    return n_cases


def units(rng, mean: float, sd: float, n: int) -> np.ndarray:
    return np.clip(np.rint(rng.normal(mean, sd, n)), 0, MAX_UNIT).astype(int)


def power(n_seeds: int, gains: tuple, variants: tuple, draws: int, seed: int) -> float:
    """P(the test confirms) for true gains ``gains`` (one per variant in ``variants``)."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(draws):
        groups = {}
        for v, g in zip(variants, gains):
            base = units(rng, MEAN[v], SD[v], n_seeds)
            treated = units(rng, MEAN[v] + g, SD[v], n_seeds)
            groups[v] = (treated, base)
        hits += exact_p(groups)[0] <= ALPHA
    return hits / draws


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draws-5", type=int, default=1500, help="draws for the 5-seed column")
    ap.add_argument("--draws-10", type=int, default=20000, help="draws for the 10-seed column")
    ap.add_argument("--seed-5", type=int, default=0)
    ap.add_argument("--seed-10", type=int, default=20261027)
    args = ap.parse_args()

    checked = check_against_enumeration()

    def se(p: float, n: int) -> float:
        return math.sqrt(p * (1 - p) / n)

    rows = []
    for i, (g1, g3) in enumerate(ROWS):
        p5 = power(5, (g1, g3), ("v1", "v3"), args.draws_5, args.seed_5 + i)
        p10 = power(10, (g1, g3), ("v1", "v3"), args.draws_10, args.seed_10 + i)
        rows.append({"gain_v1": g1, "gain_v3": g3, "draft_5_seeds": DRAFT_5[i],
                     "reproduced_5_seeds": round(p5, 4), "se_5": round(se(p5, args.draws_5), 4),
                     "p1_10_seeds": round(p10, 4), "se_10": round(se(p10, args.draws_10), 4)})
    s1v1 = []
    for j, g in enumerate(S1V1_GAINS):
        p = power(10, (g,), ("v1",), args.draws_10, args.seed_10 + 100 + j)
        s1v1.append({"gain_v1": g, "s1v1_10_seeds": round(p, 4),
                     "se": round(se(p, args.draws_10), 4)})
    print(json.dumps({
        "method": "units round(N(mean + gain, SD)) clipped to 0..12; exact within-variant "
                  "permutation test, one-sided, ties toward p, alpha 0.05",
        "baseline": {"mean": MEAN, "sd": SD},
        "exact_count_checked_against_enumeration": checked,
        "draws": {"5_seeds": args.draws_5, "10_seeds": args.draws_10},
        "rng": {"5_seeds": f"numpy default_rng(seed {args.seed_5} + row index)",
                "10_seeds": f"numpy default_rng(seed {args.seed_10} + row index)",
                "s1v1": f"numpy default_rng(seed {args.seed_10} + 100 + row index)"},
        "relabellings": {"p1_5_seeds": math.comb(10, 5) ** 2,
                         "p1_10_seeds": math.comb(20, 10) ** 2,
                         "per_variant_10_seeds": math.comb(20, 10)},
        "p1": rows, "s1_v1": s1v1}, indent=2))


if __name__ == "__main__":
    main()
