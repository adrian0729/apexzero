"""DV-B0: power of the drivers split-half gate, simulated from the recorded noise budgets.

    nice -n 15 uv run python scripts/drivers_power.py            # simulate, then report
    nice -n 15 uv run python scripts/drivers_power.py --report   # report from the cache

Plan, committed before the first run: docs/findings/drivers-power-plan.md (model, sweep
and the go/no-go rule for DV-B). Writes outputs/drivers-lite/power/power.json and
docs/findings/drivers-power.md with its figure. EXPLORATORY.

Inputs are all on disk: the injection-recovery and persistent-error runs and the four
pilot lap tables (outputs/drivers-lite/, from drivers_pilot.py), the registration tables,
and the team rosters (drivers.json.gz) of the sessions in scope. No laps, positions or
timing are read from any session outside the pilot. Runs in about a minute.
"""

import argparse
import gzip
import json
import pickle
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.stats import rankdata

from apexzero.drivers import deltas as dl
from apexzero.drivers.circuits import CIRCUITS
from apexzero.drivers.segments import PHASES

OUT = Path("outputs/drivers-lite")
POWER = OUT / "power"
ROOT = Path("data/raw/openf1")
DOCS = Path("docs/findings")
PILOT = ("Suzuka", "Montreal", "Spa", "Monza")    # TUMFTM names, in drivers_pilot.PILOT_KEYS order
TYPES = ("slow", "medium", "fast")
CELLS3 = [(t, p) for t in TYPES for p in PHASES]
GATE = 0.5
TAUS_MS = np.arange(0.0, 41.0, 5.0)
CAL_TAUS_MS = np.arange(0.0, 41.0, 1.0)
KS = (1.0, 1.36)
N_SIMS = 2000
CHUNK = 500
SEED = 20261025
# Phase schemes: which recorded phase each 3-phase cell merges into.
SCHEMES = {"3": dict(entry="entry", apex="apex", exit="exit"),
           "2a": dict(entry="entry+apex", apex="entry+apex", exit="exit"),
           "2b": dict(entry="entry", apex="apex+exit", exit="apex+exit")}
SCOPES = {"q": "35 qualifying sessions, 2023-25",
          "q+sq": "plus 2023-25 sprint qualifying at the same circuits",
          "q+sq+2026": "plus 2026 qualifying at current-layout circuits"}
# Categorical slots 1-3 of the reference palette (validated, light surface).
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def merge_matrix(scheme: str) -> tuple[np.ndarray, list]:
    """[9, m] 0/1 matrix summing 3-phase cells into the scheme's cells, and their names."""
    names = []
    for t in TYPES:
        for p in PHASES:
            if (t, SCHEMES[scheme][p]) not in names:
                names.append((t, SCHEMES[scheme][p]))
    m = np.zeros((len(CELLS3), len(names)))
    for i, (t, p) in enumerate(CELLS3):
        m[i, names.index((t, SCHEMES[scheme][p]))] = 1.0
    return m, names


def lap_sums(per_phase: np.ndarray, labels) -> np.ndarray:
    """[..., P] per-phase values -> [..., 9] sums over the lap per corner type x phase."""
    out = np.zeros(per_phase.shape[:-1] + (len(CELLS3),))
    for j, (_, ph, kind) in enumerate(labels):
        out[..., CELLS3.index((kind, ph))] += per_phase[..., j]
    return out


# ----------------------------------------------------------------------------- inputs


def load_inputs() -> dict:
    with open(OUT / "injection.pkl", "rb") as f:
        inj = pickle.load(f)
    with open(OUT / "persistent.pkl", "rb") as f:
        pers = pickle.load(f)
    libs, tabs = [], []
    for name in PILOT:
        with open(OUT / f"pilot_{name}.pkl", "rb") as f:
            tab = pickle.load(f).table
        tabs.append(tab)
        rec, both = inj["pilot"][name], pers["runs"]["both"][name]
        if rec.labels != tab.labels:
            raise ValueError(f"{name}: injection and pilot segmentations differ")
        if not np.array_equal(rec.pair_true, both.pair_true):
            raise ValueError(f"{name}: persistent run differs in more than the persistent terms")
        n_type = {t: len({cx for cx, _, k in tab.labels if k == t}) for t in TYPES}
        if min(n_type.values()) == 0:
            raise ValueError(f"{name}: a corner type is missing; cells would need masking")
        shares = dl.time_shares(tab)
        libs.append({
            "name": name,
            "n_type": np.array([n_type[t] for t, _ in CELLS3], float),
            "shares": np.array([shares[c] for c in CELLS3]),
            "noise": lap_sums(rec.pair_est - rec.injected[None, :], rec.labels),
            "persist": lap_sums(both.pair_est - rec.pair_est, rec.labels),
        })
    pairs = [(i, d) for i, tab in enumerate(tabs) for d in dl.teammate_deltas(tab)]
    n_car_sessions = sum(len(set(tab.team)) for tab in tabs)
    return {"libs": libs, "tabs": tabs, "pilot_pairs": pairs,
            "var_factors": np.array([d.var_factor for _, d in pairs]),
            "yield": len(pairs) / n_car_sessions, "n_car_sessions": n_car_sessions}


def gap_model(inp: dict, k: float) -> dict:
    """sd of the pair's constant gap (g) and per-session gap (eta) from the pilot's lap
    deltas, by moments: lap-delta noise is k^2 * var_factor * the circuit's recorded
    lap-sum noise variance. The sign of a delta is arbitrary (B minus A by number), so
    second moments are taken about zero."""
    lap_var = [float(np.mean(lib["noise"].sum(axis=1) ** 2)) for lib in inp["libs"]]
    rows = [(d.a, d.b, d.lap_delta, k**2 * d.var_factor * lap_var[i])
            for i, d in inp["pilot_pairs"]]
    by_pair: dict = {}
    for a, b, lap, nv in rows:
        by_pair.setdefault((a, b), []).append((lap, nv))
    ss, dof, nv_within = 0.0, 0, []
    for v in by_pair.values():
        if len(v) >= 2:
            x = np.array([lap for lap, _ in v])
            ss += float(((x - x.mean()) ** 2).sum())
            dof += len(v) - 1
            nv_within.extend(nv for _, nv in v)
    eta2 = max(ss / dof - float(np.mean(nv_within)), 0.0) if dof else 0.0
    total = float(np.mean([lap**2 for _, _, lap, _ in rows]))
    noise = float(np.mean([nv for *_, nv in rows]))
    g2 = max(total - eta2 - noise, 0.0)
    return {"sd_g": float(np.sqrt(g2)), "sd_eta": float(np.sqrt(eta2)), "within_ms2": ss / dof,
            "total_ms2": total, "noise_ms2": noise, "pairs_repeated": int(sum(
                len(v) >= 2 for v in by_pair.values())), "pair_sessions": len(rows)}


def _session_meta(d: Path) -> dict:
    meta = json.loads((d / "session.json").read_text())
    with gzip.open(d / "drivers.json.gz", "rt") as f:
        drivers = json.load(f)
    teams: dict = {}
    for r in drivers:
        teams.setdefault(r["team_name"], set()).add(int(r["driver_number"]))
    meta["pairs"] = sorted(tuple(sorted(v)) for v in teams.values() if len(v) == 2)
    return meta


def season_design(scope: str) -> dict:
    """Sessions in scope with their rosters (team rosters only), rounds in date order."""
    base = []
    for year in (2023, 2024, 2025):
        name = "registration.json" if year == 2025 else f"registration_{year}.json"
        for r in json.loads((OUT / name).read_text()):
            if r.get("tumftm") and r["layout_matches"] and r["accepted"]:
                base.append(_session_meta(Path(r["dir"])))
    sessions = list(base)
    if scope != "q":
        have = {(m["year"], m["circuit"]) for m in base}
        for p in sorted(ROOT.glob("202[345]/*/session.json")):
            meta = json.loads(p.read_text())
            if meta["name"] == "Sprint Qualifying" and (meta["year"], meta["circuit"]) in have:
                sessions.append(_session_meta(p.parent))
    if scope == "q+sq+2026":
        for p in sorted(ROOT.glob("2026/*/session.json")):
            meta = json.loads(p.read_text())
            c = CIRCUITS.get(meta["circuit"])
            if meta["name"] == "Qualifying" and c is not None and c.usable:
                sessions.append(_session_meta(p.parent))
    meetings = sorted({m["meeting_key"]: m["date_start"] for m in sessions}.items(),
                      key=lambda kv: kv[1])
    half_of = {mk: i % 2 for i, (mk, _) in enumerate(meetings)}
    sessions.sort(key=lambda m: m["date_start"])
    pair_ids: dict = {}
    q_pair, q_session, q_half = [], [], []
    for si, m in enumerate(sessions):
        for ab in m["pairs"]:
            q_pair.append(pair_ids.setdefault(ab, len(pair_ids)))
            q_session.append(si)
            q_half.append(half_of[m["meeting_key"]])
    return {"scope": scope, "n_sessions": len(sessions), "n_meetings": len(meetings),
            "n_pairs": len(pair_ids), "pair": np.array(q_pair), "session": np.array(q_session),
            "half": np.array(q_half), "circuit": None, "var_factor": None, "mask": None,
            "names": [f"{m['year']} {m['circuit']} {m['name']}" for m in sessions]}


def pilot_design(inp: dict) -> dict:
    """The pilot's exact pair-sessions: circuits, lap counts and halves as in the report
    (alternate sessions: Suzuka + Spa against Montreal + Monza)."""
    pair_ids: dict = {}
    q_pair, q_session = [], []
    for i, d in inp["pilot_pairs"]:
        q_pair.append(pair_ids.setdefault((d.a, d.b), len(pair_ids)))
        q_session.append(i)
    q_session = np.array(q_session)
    return {"scope": "pilot", "n_sessions": len(PILOT), "n_pairs": len(pair_ids),
            "pair": np.array(q_pair), "session": q_session, "half": q_session % 2,
            "circuit": np.arange(len(PILOT)), "var_factor": inp["var_factors"],
            "mask": np.ones(len(q_pair), bool)}


# ----------------------------------------------------------------------------- simulation


def spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman rho per row over the entries finite in both (NaN with fewer than 3)."""
    ok = np.isfinite(x) & np.isfinite(y)
    rx = np.where(ok, rankdata(np.where(ok, x, np.nan), axis=1, nan_policy="omit"), 0.0)
    ry = np.where(ok, rankdata(np.where(ok, y, np.nan), axis=1, nan_policy="omit"), 0.0)
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.where(ok, rx - rx.sum(axis=1, keepdims=True) / n[:, None], 0.0)
        cy = np.where(ok, ry - ry.sum(axis=1, keepdims=True) / n[:, None], 0.0)
        rho = (cx * cy).sum(axis=1) / np.sqrt((cx**2).sum(axis=1) * (cy**2).sum(axis=1))
    return np.where(n >= 3, rho, np.nan)


def draws(design: dict, inp: dict, n: int, rng: np.random.Generator) -> dict:
    """Every random input of n simulated seasons, as standard draws; configurations
    scale and combine them (common random numbers across the grid)."""
    libs = inp["libs"]
    nq, npair, ns = len(design["pair"]), design["n_pairs"], design["n_sessions"]
    circ = (np.broadcast_to(design["circuit"], (n, ns)) if design["circuit"] is not None
            else rng.integers(len(libs), size=(n, ns)))
    cq = circ[:, design["session"]]                                    # [n, Q]
    mask = (np.broadcast_to(design["mask"], (n, nq)) if design["mask"] is not None
            else rng.random((n, nq)) < inp["yield"])
    vf = (np.broadcast_to(design["var_factor"], (n, nq)) if design["var_factor"] is not None
          else rng.choice(inp["var_factors"], size=(n, nq)))
    noise = np.stack([lib["noise"] for lib in libs])                   # [C, 20, 9]
    pers = np.stack([lib["persist"] for lib in libs])
    eps = noise[cq, rng.integers(noise.shape[1], size=(n, nq))]
    eps *= rng.choice((-1.0, 1.0), size=(n, nq, 1)) * np.sqrt(vf)[..., None]
    pc = rng.integers(len(libs), size=(n, npair))
    pi = pers[pc, rng.integers(pers.shape[1], size=(n, npair))]
    pi *= rng.choice((-1.0, 1.0), size=(n, npair, 1))
    n_type = np.stack([lib["n_type"] for lib in libs])[cq]             # [n, Q, 9]
    shares = np.stack([lib["shares"] for lib in libs])[cq]
    z_theta = rng.standard_normal((n, npair, len(CELLS3)))
    return {"mask": mask, "eps": eps, "pi": pi[:, design["pair"]], "shares": shares,
            "signal": n_type * z_theta[:, design["pair"]],
            "z_g": rng.standard_normal((n, npair))[:, design["pair"]],
            "z_eta": rng.standard_normal((n, nq))}


def rhos(design: dict, dr: dict, scheme: str, taus_s, k: float, gap: dict, persist: bool,
         stats=("raw", "profile")) -> dict:
    """{stat: [len(taus), n] split-half rho} for one configuration."""
    m, _ = merge_matrix(scheme)
    a = [np.zeros((len(design["pair"]), design["n_pairs"])) for _ in range(2)]
    for h in range(2):
        q = np.flatnonzero(design["half"] == h)
        a[h][q, design["pair"][q]] = 1.0
    mask = dr["mask"].astype(float)
    cnt = [mask @ a[h] for h in range(2)]                               # [n, P]
    valid = (cnt[0] > 0) & (cnt[1] > 0)
    shares = dr["shares"] @ m
    gap_t = dr["shares"] * (gap["sd_g"] * dr["z_g"] + gap["sd_eta"] * dr["z_eta"])[..., None]
    fixed = (k * dr["eps"] + gap_t + (dr["pi"] if persist else 0.0)) @ m
    sig = dr["signal"] @ m
    out = {s: [] for s in stats}
    for tau in taus_s:
        d = tau * sig + fixed                                           # [n, Q, cells]
        for s in stats:
            x = d - shares * d.sum(axis=-1, keepdims=True) if s == "profile" else d
            x = x * mask[..., None]
            with np.errstate(invalid="ignore", divide="ignore"):
                h = [np.einsum("nqc,qp->npc", x, a[i]) / cnt[i][..., None] for i in range(2)]
            h = [np.where(valid[..., None], hi, np.nan).reshape(len(d), -1) for hi in h]
            out[s].append(spearman_rows(h[0], h[1]))
    return {s: np.array(v) for s, v in out.items()}


# ----------------------------------------------------------------------------- stages


def calibrate(inp: dict, gaps: dict) -> dict:
    """tau_hat and tau_80 from the pilot's own across-session split-half (3 phases)."""
    tabs = inp["tabs"]
    observed = {"raw": dl.split_half_across(tabs).rho,
                "profile": dl.split_half_across(tabs, profile=True).rho}
    design = pilot_design(inp)
    out = {"observed": observed, "pairs_in_both_halves": dl.split_half_across(tabs).n_pairs,
           "by_k": {}}
    for k in KS:
        rng = np.random.default_rng(SEED + 1)
        sims = {s: [] for s in observed}
        for _ in range(N_SIMS // CHUNK):
            r = rhos(design, draws(design, inp, CHUNK, rng), "3", 1e-3 * CAL_TAUS_MS, k,
                     gaps[k], persist=False)
            for s in observed:
                sims[s].append(r[s])
        res = {}
        for s, obs in observed.items():
            r = np.concatenate(sims[s], axis=1)                         # [taus, sims]
            med, q20 = np.nanmedian(r, axis=1), np.nanpercentile(r, 20, axis=1)
            res[s] = {"median": med.tolist(), "q20": q20.tolist(),
                      "tau_hat_ms": _first_cross(CAL_TAUS_MS, med, obs),
                      "tau_80_ms": _first_cross(CAL_TAUS_MS, q20, obs)}
        out["by_k"][str(k)] = res
    return out


def pilot_persistent(inp: dict, gaps: dict) -> dict:
    """POST HOC (added after the first run): would the pilot have seen the persistent-error
    scenario if it repeated across sessions? The pilot's exact design at tau = 0 with pi
    on: the median simulated rho, and how often it is at or below the observed one."""
    tabs = inp["tabs"]
    observed = {"raw": dl.split_half_across(tabs).rho,
                "profile": dl.split_half_across(tabs, profile=True).rho}
    design = pilot_design(inp)
    out = {}
    for k in KS:
        rng = np.random.default_rng(SEED + 2)
        sims = {s: [] for s in observed}
        for _ in range(N_SIMS // CHUNK):
            r = rhos(design, draws(design, inp, CHUNK, rng), "3", [0.0], k, gaps[k],
                     persist=True)
            for s in observed:
                sims[s].append(r[s][0])
        out[str(k)] = {s: {"median": float(np.nanmedian(np.concatenate(v))),
                           "p_at_or_below_observed": float(np.nanmean(
                               np.concatenate(v) <= observed[s]))} for s, v in sims.items()}
    return out


def _first_cross(x, y, level) -> float | None:
    """Smallest x where y reaches level (linear between grid points); 0 if y[0] already
    does; None if it never does on the grid."""
    y = np.asarray(y)
    if y[0] >= level:
        return 0.0
    for i in range(1, len(x)):
        if y[i] >= level:
            return float(x[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1]))
    return None


def sweep(inp: dict, gaps: dict) -> dict:
    out = {"designs": {}, "p_pass": {}}
    for scope, label in SCOPES.items():
        t0 = time.process_time()
        design = season_design(scope)
        rng = np.random.default_rng(SEED)
        acc: dict = {}
        both, sessions_per_pair = [], []
        for _ in range(N_SIMS // CHUNK):
            dr = draws(design, inp, CHUNK, rng)
            for scheme in SCHEMES:
                for k in KS:
                    for persist in (False, True):
                        r = rhos(design, dr, scheme, 1e-3 * TAUS_MS, k, gaps[k], persist)
                        for s, v in r.items():
                            acc.setdefault((scheme, s, k, persist), []).append(v >= GATE)
            per = np.zeros((CHUNK, design["n_pairs"], 2))
            for h in range(2):
                q = design["half"] == h
                np.add.at(per[:, :, h], (slice(None), design["pair"][q]), dr["mask"][:, q])
            both.append(((per > 0).all(axis=2)).sum(axis=1))
            sessions_per_pair.append(per.sum(axis=2))
        spp = np.concatenate(sessions_per_pair)
        out["designs"][scope] = {
            "label": label, "sessions": design["n_sessions"],
            "meetings": design["n_meetings"], "pairs": design["n_pairs"],
            "pair_sessions": len(design["pair"]),
            "mean_pairs_in_both_halves": float(np.mean(np.concatenate(both))),
            "mean_usable_sessions_top2_pairs": float(np.mean(np.sort(spp, axis=1)[:, -2:])),
            "median_usable_sessions_per_pair": float(np.median(spp)),
        }
        out["p_pass"][scope] = {f"{sch}|{s}|{k}|{int(p)}": np.concatenate(v, axis=1).mean(
            axis=1).tolist() for (sch, s, k, p), v in acc.items()}
        print(f"{scope}: {design['n_sessions']} sessions, {design['n_pairs']} pairs "
              f"[{time.process_time() - t0:.0f} s cpu]", flush=True)
    return out


def decide(res: dict) -> dict:
    """The plan's rule: profile, k = 1.36, pi off, P(pass) >= 0.5 at the profile tau_80;
    3 phases, or 2 phases if both merges reach it; any scope."""
    k = "1.36"
    tau80 = res["calibration"]["by_k"][k]["profile"]["tau_80_ms"]
    checks = []
    for scope in SCOPES:
        pp = res["sweep"]["p_pass"][scope]
        at = {sch: (float(np.interp(tau80, TAUS_MS, pp[f"{sch}|profile|{k}|0"]))
                    if tau80 is not None else None) for sch in SCHEMES}
        two = None if at["2a"] is None else min(at["2a"], at["2b"])
        checks.append({"scope": scope, "p3": at["3"], "p2a": at["2a"], "p2b": at["2b"],
                       "pass3": at["3"] is not None and at["3"] >= 0.5,
                       "pass2": two is not None and two >= 0.5})
    go = tau80 is not None and any(c["pass3"] or c["pass2"] for c in checks)
    if tau80 is None:   # the pilot does not bound tau below 40 ms: the rule cannot be read
        go = None
    return {"tau_80_ms": tau80, "checks": checks, "go": go}


def run() -> dict:
    t0 = time.process_time()
    inp = load_inputs()
    gaps = {k: gap_model(inp, k) for k in KS}
    res = {"inputs": {"yield": inp["yield"], "car_sessions": inp["n_car_sessions"],
                      "pilot_pairs": len(inp["pilot_pairs"]),
                      "var_factors": inp["var_factors"].tolist(),
                      "noise_sd_ms": {lib["name"]: (1e3 * lib["noise"].std(axis=0)).tolist()
                                      for lib in inp["libs"]},
                      "n_type": {lib["name"]: lib["n_type"].tolist() for lib in inp["libs"]}},
           "gap": {str(k): v for k, v in gaps.items()},
           "taus_ms": TAUS_MS.tolist(), "cal_taus_ms": CAL_TAUS_MS.tolist(), "ks": list(KS),
           "n_sims": N_SIMS, "seed": SEED}
    res["calibration"] = calibrate(inp, gaps)
    res["calibration"]["persistent_tau0"] = pilot_persistent(inp, gaps)
    print("calibration:", json.dumps({k: {s: (v["tau_hat_ms"], v["tau_80_ms"])
                                          for s, v in r.items()}
                                      for k, r in res["calibration"]["by_k"].items()}),
          res["calibration"]["observed"], flush=True)
    res["sweep"] = sweep(inp, gaps)
    res["decision"] = decide(res)
    res["cpu_s"] = time.process_time() - t0
    POWER.mkdir(parents=True, exist_ok=True)
    (POWER / "power.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="write the report from the cache")
    args = ap.parse_args()
    res = json.loads((POWER / "power.json").read_text()) if args.report else run()
    from drivers_power_report import write_report

    write_report(res)
