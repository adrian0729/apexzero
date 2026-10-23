"""Idea 2's decision rules and their statistics (design v2.1, Section 6 with Section 12's
decisions; the descriptives of Section 6 'Secondary and descriptive').

Everything here works on arrays that the stages computed (targets, pulled-back gradients,
hidden units, probe predictions, per-network scalars); nothing evaluates a network. The
primary family is **C1, C2 and C3-w** (Section 12 decision 6), each with its own bands and
no correction across them (Section 6 'Multiplicity'). C2-sw is a named secondary;
C1-pattern, C1-heads, C1-own (``c1_own``, on T's interior restarts), C1-limit, C1-braking and
W1 (``calibrated_agreement``) are pre-registered secondaries; the rest is descriptive. C3-int
(``c3_int``), C1-dev (``c1_dev``), Delta C1 (``delta_c1``), C2-pairs (``c2_pairs``) and C1-own
on its own-run and N states are deferred (Section 13 'Deferred', ``prereg.DEFERRED``): not
part of costate-prereg-v1, exploratory if run later; their code is kept.

**C1** (``c1``; Section 6 'C1'). Per component k of K_P = (s, n, xi, vx)
(``targets.K_P``), a per-fold affine calibration of lambda_k on g_k (``calibrate``) on the
training folds' rows of the fit stratum (the interior, Section 4), with the slope clamped at
>= 0. **Deviation from Section 6's literal 'Huber affine fit', fixed in code before the tag
and to be approved in the design (review findings on the calibration's floor and leverage):**

- *within direction for the mirror-odd components* (n, xi, vy, r, delta): direction-specific
  training-fold intercepts with a shared slope (lambda = a_dir + b g), so the calibration
  matches R2_wd's SST (a pooled intercept cannot fit the two directions' means, and gave a
  zero gradient R2_c = -0.49 on xi);
- *the slope* is the weighted least-squares slope (``CAL_SLOPE`` = 'wls'; 'huber' keeps the
  Huber 1.345 slope as a sensitivity) of the direction-centred lambda on the direction-centred
  g, with the metric's weights times Mallows-type leverage weights v = min(1, ``K_LEV`` s_g /
  |g - med_g|)^2 (``leverage_weights``: s_g the weighted MAD / 0.6745 of g on the training
  rows; squared, so a row's least-squares influence v g^2 stays below K_LEV^2 s_g^2: with the
  first power four rows of one key with a 100 SD gradient spike still crushed every other
  fold's slope from 0.9 to 0.28; v depends on g only, so a correct linear gradient keeps an
  unbiased slope), clamped at >= 0 (the targets are already winsorised; the Huber slope
  attenuated the skewed lambda_s);
- *the intercept* is the weighted training-fold mean of lambda - b g (within direction for
  the odd components; for b > 0 under the same leverage-weighted weights, so a spike in g
  cannot shift it), never the Huber location: any b = 0 fit (a zero, pure-noise or
  clamped anti-aligned gradient) predicts the training-fold mean and scores R2_c of about 0
  (the Huber location of the skewed lambda_s gave -0.13);
- *clipping*: the calibrated predictions are clipped to the frozen winsorisation bounds of
  their target where they are scored (``bounds``; the targets are winsorised to the same
  bounds, so clipping never increases a squared error), and the clipped share is recorded.

R2_c is the R2_wd (``probes.r2``: SST within direction for the mirror-odd
components) of the calibrated g on the scored rows, averaged over K_P; components carry
their own row masks (``valid``: s is dropped where it is structurally unobservable, the
tau_s rule, ``states.observability``; K2b's per-phase exclusions; multi-basin keys never
enter T), and a row is scored only where its fold has a calibration (a fold with fewer than
3 training rows, e.g. every fold of a stratum inside one geometry, has none: its rows are
dropped and counted, never scored against a filled-in 0; a component left without rows drops
out of the mean, and with none left C1 is 'not testable' with its reason). Nulls go
through the same calibration on the same scored rows; the ones fitted on or built from
targets (``FOLD_NULLS``: the value fits and the far-geometry lambda) are [F, n, m] per
held-out fold, as C2's lambda_res (``nulls`` module docstring 'Per held-out fold'), and
fold f's calibration of such a null is fitted on slice f's training rows and predicts fold
f's rows from slice f, so no target of fold f reaches fold f's prediction. They are compared
by a paired cluster bootstrap over geometries (the same resamples, ``probes.cluster_counts``
over T's geometries): value-fit ridge and MLP, QSS, clearance barrier, phase-matched
far-geometry lambda (``nulls``) and the random-init critics, whose maximum over the 20 seeds
(of the K_P mean, per bootstrap draw) is the sanity floor. The signed Spearman
rho(g_k, lambda_k) is weighted (``probes.wspearman``'s mid-ranks) and, for the mirror-odd
components, the direction-size weighted mean of the within-direction rho (R2_wd's
convention: a raw rho would reward the turn direction). Confirm: R2_c's CI lower end >
``C1_CONFIRM`` = 0.6, every paired R2_c - null CI lower end > 0, and every K_P
component's Spearman CI lower end > 0. Refute: R2_c's CI upper end < ``C1_REFUTE`` = 0.2.
Otherwise inconclusive. The label "costate gradient at scale" (``C1.label``): a confirm
plus, for each K_P component, the Frisch bracket of the pooled scale inside ``SCALE_BAND``
= [0.5, 2] (Section 12 decision 2: it gates the label, not the decision). g is in reward
units already (sigma_R applied by ``critic.pullback``). With g = c lambda + noise the slope
of lambda on g is about R2 / c, so it confounds reliability with scale; the slope of g on
lambda (b_gl) recovers c under classical error in g, 1 / b_lg under error in lambda, and c
lies between them under errors in either or both (their ratio is the fit's R2). The label
needs every component's b_gl CI lower end >= 0.5 and 1 / b_lg's CI upper end <= 2 (a
conservative rule: under classical error it accepts c in [0.5, 2 R2]); both are pooled
(not cross-fitted) WLS slopes with the leverage weights of their regressor, within direction
for the odd components, bootstrapped on the same cluster resamples (computed only for a
confirm; the points always). The design's 'pooled sigma_R-converted slope' is ambiguous
between the two: the lead is told.

**C2 and C2-sw** (``c2_network``, ``c2_sw_network``): ``probes.c2`` and ``probes.c2_sw`` on
the stratum's rows (``PROBE_FIT`` = 'stratum': the probes are fitted and scored on the
decision stratum, with the folds dealt on all of T and subset, ``probes.Folds.subset``), with
C2's power floor from the pre-tag residual shares (``states.power_floor``; a component below
it is descriptive, and with none passing C2 is 'not testable', Section 12 decision 5). C2-sw
co-reports the weighted Spearman rho(m_pedal - 1, sigma^ped).

**Per network and per variant** (``network_decisions``, ``variant_decisions``). C1, C2 and
C2-sw are decided per network on the 9 step 1 generalist finals (``DECISION_NETWORKS``,
interp/registry.py:110-114); a variant claim needs the same decision in 2 of its 3 seeds:
step 1's rule R18 (``interp.analysis.variant_verdict``, analysis.py:3976-3993), used as is.

**C3-w** (``c3_w``; Section 6 'C3'): the size-weighted mean over the 7 recipe strata
(``C3_STRATA``) of the within-stratum Spearman rho(x, y) (average ranks for ties; a stratum
with a constant x or y contributes rho = 0), a one-sided stratified permutation test (y
permuted within strata, ``C3_PERMUTATIONS`` = 100,000 draws on stream ``costate_c3``;
p = (1 + #{null >= observed}) / (1 + draws)); confirm at p <= 0.05. The 'rules out a strong
association' bound is the shift inversion of the same test: the 95% upper bound is
rho_obs - q_0.05(null), and it fires when that is below ``C3_STRONG`` = 0.3 (with the null SD
of about 0.20 it fires about 40% of the time under an exact null, as Section 6 states).
There is no reversed-sign refute. The covariates (competence, overlap) are reported as
stratified partial associations: within each stratum the ranks of x and y are residualised on
the covariate's ranks, with the same permutation scheme (descriptive). A stratum holding a
network without the covariate is left out of that partial whole, and the size weights are
renormalised over the strata scored (the networks without it are listed: step 2 has no
fields-a2 rows for runs/dr-pilot-v2-tilt30-s3 and -s4, so the overlap partial scores 26 of
the 31 generalists, base-v2 left out).

**C3-int** (``c3_int``; deferred, so not pre-registered, though the record keeps its
constants): per network x geometry, the signed calibrated critic residual of each K_P
component (sigma-folded for the mirror-odd ones, so a geometry's two directions add)
and the behavioural deviations, each double-centred over networks and geometries
(``double_centre``: alternating row and column centring, exact on a complete table); the
statistic is the mean over the pre-signed pairs (``C3_INT_PAIRS``) of sign x the Pearson
correlation over the cells both hold; one-sided, against permutations of the geometry labels
within each network, blocked on R tercile x theta class (``states.r_tercile``,
``states.theta_class``), ``C3_INT_PERMUTATIONS`` draws on stream ``costate_c3`` sub-stream 1.
Confirm at p <= 0.05. The pairs: lambda_n underpricing the inside wall (sigma (g_n - lambda_n)
> 0) goes with G < 1 and with a smaller buffer B (step 2's G - 1 and B, sign -1); a too-negative
lambda_vx (g_vx - lambda_vx < 0) goes with earlier braking (the brake-onset residual
s_b,net - s_b,opt < 0) and with a lower entry speed (the entry-speed ratio - 1 < 0), sign +1.

**Value accuracy** (``value_accuracy``; Section 3.3 'Value accuracy', Section 12 decision 7,
Section 13 'Built before the tag'): per network, the out-of-fold R2 of sigma_R V_c against
the deterministic policy's own discounted return (``VALUE_TARGET`` = 'v_own_det', the own
stage's ``own.own_return``: J over up to 1,000 steps or to the strip's end, the gamma^H tail
reported) from the states the own stage restarts from (``VALUE_ROWS`` = 'own': T's interior
restarts every 20th step and D, every key's t = 0 state; a run that crashes is recorded and
dropped), with the frozen folds and C1's per-fold affine calibration (slope >= 0); it passes
at R2 >= ``VALUE_ACCURACY_R2`` = 0.8 with every requested row scored, and is a precondition
for C1-own's validity: a network that fails it (or K3b's trained analogue) has a descriptive
C1-own (``c1_own_status``: 'descriptive (value accuracy)'). C1-own is the critic against its
own policy, so the target is that policy's return, not the optimum's. The critic itself
regresses on its stochastic training policy's bootstrapped returns (Section 3.3), so even a
perfect fit differs from this deterministic return by the policy noise, grip jitter and
crash risk. Reported beside it, descriptively: the same score against the optimum's
horizon-corrected window value V*inf_t (``VALUE_TARGET_VSTAR``) on T's interior (Section 9's
'V^pi rather than V*' reading); it never decides C1-own.

**Strata tables** (``stratum_rows``, ``stratum_record``; ``STRATA``): per network and
hypothesis, the descriptive per-stratum tables of Sections 4, 6, 12 and 13 (C1 and C2:
interior k = 10 primary, k = 20, near-limit, contact, the corner-phase band and its interior
part, the interior's phases, its braking and non-braking keys, its left and right turns, its
W 400 and 800 keys, its pedal-saturated and its kappa-clipped states; C2-sw: the
pedal-switching steps, their four phase parts, the raw pedal-saturated flag, their braking
and non-braking keys, left and right, W 400 and 800 and kappa-clipped), each the
hypothesis' statistic recomputed on the stratum (C1 with its nulls and the slope CIs per
component), bootstrapped over the geometries present there, with the band it falls in
reported as 'band', never a decision. Kappa clipping is per network (``inputs.flags`` under
its own obs_rms), a column added to T's for the tables (``strata_tab``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from apexzero.costate import probes as PB
from apexzero.costate import states as ST
from apexzero.costate import stream_rng
from apexzero.costate.targets import K_P, SECONDARY
from apexzero.interp import registry

HUBER_C = 1.345               # Section 6 C1: the Huber tuning constant (MAD scale)
CAL_SLOPE = "wls"             # C1's calibration slope: 'wls' (primary) or 'huber' (module
                              # docstring: a deviation from Section 6, approved before the tag)
K_LEV = 3.0                   # Mallows leverage weights of the calibration's regressor
MAD_NORMAL = 0.6744897501960817   # Phi^-1(0.75): MAD -> SD of a normal
HUBER_ITERS = 100
HUBER_TOL = 1e-10
C1_CONFIRM = 0.6              # Section 6 C1: R2_c's CI lower end above this confirms ...
C1_REFUTE = 0.2               # ... and its CI upper end below this refutes
SCALE_BAND = (0.5, 2.0)       # Section 12 decision 2: the "at scale" label only
C2_BAND = PB.C2_BAND          # Section 6 C2 and C2-sw: 0.05
PROBE_FIT = "stratum"         # probes fitted and scored on the decision stratum's rows
C3_PERMUTATIONS = 100_000     # Section 6 C3-w
C3_ALPHA = 0.05
C3_STRONG = 0.3               # 'rules out a strong association' below this upper bound
C3_INT_PERMUTATIONS = 10_000  # C3-int's permutations (stream costate_c3, sub-stream 1)
VALUE_ACCURACY_R2 = 0.8       # Section 12 decision 7: out-of-fold R2 of sigma_R V_c
VALUE_TARGET = "v_own_det"    # ... against the deterministic policy's own discounted return
                              # (Section 3.3 'Value accuracy'; the own stage, ``own_of``)
VALUE_ROWS = "own"            # ... from the own stage's states (T's interior restarts and D),
                              # per network, the frozen folds; crashed runs dropped
VALUE_TARGET_VSTAR = "v_star_inf"   # descriptive: against the optimum's V*inf_t ...
VALUE_ROWS_VSTAR = "interior"       # ... on T's interior (Section 9's 'V^pi, not V*' reading)
DOUBLE_CENTRE_ITERS = 200
DOUBLE_CENTRE_TOL = 1e-12
NULL_NAMES = ("value_ridge", "value_mlp", "value_ridge_window", "qss", "barrier",
              "far_geometry", "random_critics")
# the nulls fitted on or built from targets: [F, n, m] per held-out fold (``calibrate``;
# ``nulls`` module docstring 'Per held-out fold'), on partition 0's folds only
FOLD_NULLS = ("value_ridge", "value_mlp", "value_ridge_window", "far_geometry")
C3W_ROWS = "interior"         # C3-w's x is fitted and scored on T's interior stratum (frozen;
                              # the full-T and near-limit x are pre-registered descriptives)
W1_ROWS = "interior"          # W1's stratum (frozen with C3-w's)

# The 31 generalists in C3-w's 7 recipe strata (Section 6 'C3-w'), by run directory.
C3_STRATA = {
    **{(f"runs/dr-pilot-200M-s{s}" if s else "runs/dr-pilot-200M"): "base-v1" for s in range(5)},
    **{(f"runs/dr-pilot-v2-tilt30-s{s}" if s else "runs/dr-pilot-v2-tilt30"): "base-v2"
       for s in range(5)},
    **{(f"runs/dr-pilot-v3-tilt50a2-s{s}" if s else "runs/dr-pilot-v3-tilt50a2"): "base-v3"
       for s in range(5)},
    **{f"runs/dr-realw-v1-s{s}": "real-v1" for s in range(4)},
    **{f"runs/dr-realw-v3-s{s}": "real-v3" for s in range(4)},
    **{f"runs/dr-realw-loo-f1-s{s}": "loo-f1" for s in range(4)},
    **{f"runs/dr-realw-loo-f2-s{s}": "loo-f2" for s in range(4)},
}
GENERALISTS = tuple(C3_STRATA)
DECISION_NETWORKS = registry.generalists()      # G1..G3 and R18's six replicates
SPECIALISTS = registry.specialists()            # descriptive throughout
# C3-int's pre-signed pairs (component, behavioural deviation, sign); fixed at the tag.
C3_INT_PAIRS = (("n", "G_minus_1", -1), ("n", "B", -1),
                ("vx", "brake_onset_residual", +1), ("vx", "entry_speed_ratio_minus_1", +1))


def network_of(agent: str) -> str:
    """A registry agent's run directory (G1 -> runs/dr-pilot-200M); a run path itself."""
    spec = registry.AGENTS.get(agent)
    return spec.run if spec is not None else agent


# ------------------------------------------------------------------------- calibration


def wmedian(y: np.ndarray, w: np.ndarray) -> float:
    return float(ST.wquantile(y, w, 0.5))


@dataclass(frozen=True)
class Huber:
    """A weighted Huber affine fit y ~ a + b x (``huber_fit``)."""

    a: float
    b: float
    scale: float
    clamped: bool
    iterations: int


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    sw = w.sum()
    mx, my = w @ x / sw, w @ y / sw
    sxx = w @ ((x - mx) ** 2)
    b = float(w @ ((x - mx) * (y - my)) / sxx) if sxx > 0 else 0.0
    return float(my - b * mx), b


def _irls(x: np.ndarray, y: np.ndarray, w: np.ndarray, c: float, slope: bool
          ) -> tuple[float, float, float, int]:
    a, b = _wls(x, y, w) if slope else (float(w @ y / w.sum()), 0.0)
    s = 0.0
    for it in range(1, HUBER_ITERS + 1):
        r = y - a - b * x
        s = wmedian(np.abs(r - wmedian(r, w)), w) / MAD_NORMAL
        if not s > 0:
            return a, b, 0.0, it
        u = np.abs(r) / s
        wt = w * np.minimum(1.0, c / np.maximum(u, 1e-300))
        a1, b1 = _wls(x, y, wt) if slope else (float(wt @ y / wt.sum()), 0.0)
        done = abs(a1 - a) + abs(b1 - b) <= HUBER_TOL * (1.0 + abs(a) + abs(b))
        a, b = a1, b1
        if done:
            return a, b, s, it
    return a, b, s, HUBER_ITERS


def huber_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, c: float = HUBER_C,
              nonneg: bool = True) -> Huber:
    """Weighted Huber regression of y on x by IRLS, the scale the MAD of the residuals about
    their median / 0.6745 (re-estimated every iteration); ``nonneg``: the slope clamped at
    >= 0, i.e. the location-only fit when the free slope is negative (module docstring)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    if not len(x):
        return Huber(np.nan, np.nan, np.nan, False, 0)
    a, b, s, it = _irls(x, y, w, c, True)
    if nonneg and b < 0.0:
        a, _, s, it2 = _irls(x, y, w, c, False)
        return Huber(a, 0.0, s, True, it + it2)
    return Huber(a, b, s, False, it)


def leverage_weights(x: np.ndarray, w: np.ndarray, k: float = K_LEV) -> np.ndarray:
    """Leverage weights min(1, k s / |x - med|)^2 of a regressor (module docstring 'C1'):
    med and s the weighted median and MAD / 0.6745 of x; all 1 when s = 0."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    if not len(x) or not np.isfinite(k):
        return np.ones(len(x))
    med = wmedian(x, w)
    dev = np.abs(x - med)
    s = wmedian(dev, w) / MAD_NORMAL
    if not s > 0:
        return np.ones(len(x))
    return np.minimum(1.0, k * s / np.maximum(dev, 1e-300)) ** 2


def _groups(direction: np.ndarray | None, odd: bool, n: int) -> list[np.ndarray]:
    if direction is None or not odd:
        return [np.ones(n, bool)]
    d = np.asarray(direction) > 0
    return [m for m in (~d, d) if m.any()]


def _centre(v: np.ndarray, w: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    out = np.asarray(v, float).copy()
    for m in groups:
        out[m] = out[m] - w[m] @ out[m] / w[m].sum()
    return out


def _robust_centre(v: np.ndarray, w: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    out = np.asarray(v, float).copy()
    for m in groups:
        out[m] = out[m] - wmedian(out[m], w[m])
    return out


def cal_weights(x: np.ndarray, w: np.ndarray, direction: np.ndarray | None = None,
                odd: bool = False) -> np.ndarray:
    """The calibration's row weights: the metric's times the leverage weights of the
    regressor centred on its weighted median (within direction for an odd component; a
    robust centre, so a spike cannot shift the other rows' leverage)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    grp = _groups(direction, odd, len(x))
    return w * leverage_weights(_robust_centre(x, w, grp), w)


def cal_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray, direction: np.ndarray | None = None,
              odd: bool = False, method: str | None = None) -> float:
    """The calibration's unclamped slope of y on x (module docstring 'C1'): x and y centred
    within direction for an odd component, both centrings and the fit under the metric's
    weights times the regressor's leverage weights (``cal_weights``); WLS through the
    centres or (``method`` 'huber') the Huber slope."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    grp = _groups(direction, odd, len(x))
    ww = cal_weights(x, w, direction, odd)
    xc, yc = _centre(x, ww, grp), _centre(y, ww, grp)
    if (method or CAL_SLOPE) == "huber":
        return float(huber_fit(xc, yc, ww, nonneg=False).b)
    sxx = float(ww @ (xc * xc))
    return float(ww @ (xc * yc) / sxx) if sxx > 0 else 0.0


@dataclass(frozen=True)
class Calibrated:
    """Out-of-fold calibrated predictions [n, m] of a gradient (every row, by its outer
    fold's fit; clipped to ``bounds`` where given) and the per-fold coefficients: intercepts
    a [F, m, 2] (per direction, index 1 = sigma > 0; equal for the even components), slopes
    b [F, m] (nan where a fold had no fit rows), and the share of rows whose prediction
    the bounds clipped ([m])."""

    pred: np.ndarray
    a: np.ndarray
    b: np.ndarray
    clamped: np.ndarray
    clip_share: np.ndarray | None = None
    bounds: np.ndarray | None = None

    def apply(self, g: np.ndarray, outer: np.ndarray, direction: np.ndarray | None = None,
              clip: bool = False) -> np.ndarray:
        """The fitted calibration on other rows ``g`` [n', m] with their outer folds (and
        directions; ``clip`` to the bounds)."""
        g = np.asarray(g, float)
        outer = np.asarray(outer, int)
        di = (np.zeros(len(outer), int) if direction is None
              else (np.asarray(direction) > 0).astype(int))
        A = self.a[outer]                                            # [n', m, 2]
        pred = A[np.arange(len(outer)), :, di] + self.b[outer] * g
        if clip and self.bounds is not None:
            pred = np.clip(pred, self.bounds[:, 0], self.bounds[:, 1])
        return pred


def fold_slices(g: np.ndarray, n: int, n_folds: int = ST.N_FOLDS) -> np.ndarray:
    """[F, n, m] a regressor per held-out fold: a 3-D ``g`` as given (slice f: what every row
    carries while fold f is held out, ``FOLD_NULLS``), a 1-D or 2-D one the same in every
    slice."""
    g = np.asarray(g, float)
    if g.ndim == 3:
        if g.shape[:2] != (n_folds, n):
            raise ValueError(f"a per-fold regressor {g.shape}: expected [{n_folds}, {n}, m]")
        return g
    g = np.atleast_2d(g.T).T.reshape(n, -1)
    return np.broadcast_to(g, (n_folds, *g.shape))


def calibrate(g: np.ndarray, lam: np.ndarray, outer: np.ndarray, w: np.ndarray,
              fit_rows: np.ndarray | None = None, valid: np.ndarray | None = None,
              n_folds: int = ST.N_FOLDS, direction: np.ndarray | None = None,
              odd=None, bounds=None) -> Calibrated:
    """C1's per-fold calibration (module docstring 'C1') of each column of ``lam`` on the
    same column of ``g``, fitted on the training folds' ``fit_rows`` that are ``valid`` for
    the component; ``odd`` [m] the mirror-odd columns (intercepts per ``direction``);
    ``bounds`` [m, 2] the targets' winsorisation bounds the predictions are clipped to.
    ``g`` [n, m], or [F, n, m] per held-out fold (a fold-dependent null, ``FOLD_NULLS``):
    fold f is fitted on slice f's training rows and predicts fold f's rows from slice f, so
    nothing of fold f enters its own calibration through the regressor."""
    lam = np.atleast_2d(np.asarray(lam, float).T).T
    n, m = lam.shape
    gs = fold_slices(g, n, n_folds)
    w = np.asarray(w, float)
    outer = np.asarray(outer, int)
    fit = np.ones(n, bool) if fit_rows is None else np.asarray(fit_rows, bool)
    val = np.ones((n, m), bool) if valid is None else np.asarray(valid, bool).reshape(n, m)
    odd = np.zeros(m, bool) if odd is None else np.asarray(odd, bool).reshape(m)
    d = None if direction is None else np.asarray(direction)
    a = np.full((n_folds, m, 2), np.nan)
    b = np.full((n_folds, m), np.nan)
    cl = np.zeros((n_folds, m), bool)
    for f in range(n_folds):
        gf = gs[f]
        for k in range(m):
            tr = (outer != f) & fit & val[:, k] & np.isfinite(gf[:, k]) & np.isfinite(lam[:, k])
            if tr.sum() < 3:
                continue
            x, y, w0 = gf[tr, k], lam[tr, k], w[tr]
            dk = None if d is None else d[tr]
            bk = cal_slope(x, y, w0, dk, bool(odd[k]))
            cl[f, k] = bk < 0.0
            bk = max(bk, 0.0)
            # the intercept: the same weights (a b = 0 fit does not read g: plain weights)
            ww = cal_weights(x, w0, dk, bool(odd[k])) if bk > 0.0 else w0
            r = y - bk * x
            a[f, k, :] = ww @ r / ww.sum()
            if odd[k] and dk is not None:
                for side in (0, 1):
                    msk = (dk > 0) == bool(side)
                    if msk.any():
                        a[f, k, side] = ww[msk] @ r[msk] / ww[msk].sum()
            b[f, k] = bk
    bnd = None if bounds is None else np.asarray(bounds, float).reshape(m, 2)
    cal = Calibrated(np.zeros((n, m)), a, b, cl, None, bnd)
    raw = cal.apply(gs[outer, np.arange(n)], outer, d)     # each row from its own fold's slice
    share = None
    if bnd is not None:
        with np.errstate(invalid="ignore"):
            out_ = (raw < bnd[:, 0]) | (raw > bnd[:, 1])
        fin = np.isfinite(raw)
        share = np.array([float(out_[fin[:, k], k].mean()) if fin[:, k].any() else 0.0
                          for k in range(m)])
        raw = np.clip(raw, bnd[:, 0], bnd[:, 1])
    return Calibrated(raw, a, b, cl, share, bnd)


# ------------------------------------------------------------------------- bootstrap helpers


def _boot_sums(y: np.ndarray, P: np.ndarray, w: np.ndarray, cl: np.ndarray, G: int,
               direction: np.ndarray | None, odd: bool) -> PB.BootSums:
    """``probes.boot_sums`` for one component with a fixed cluster count G (clusters absent
    from these rows get zero sums, so every component uses the same resamples)."""
    y = np.asarray(y, float)[:, None]
    P = np.asarray(P, float)
    P = P[None, :, None] if P.ndim == 1 else P[:, :, None]
    if direction is None or not odd:
        d = np.zeros(len(y), int)
    else:
        d = (np.asarray(direction) > 0).astype(int)
    D = 2
    idx = np.asarray(cl, int) * D + d
    e2 = (y[None] - P) ** 2
    return PB.BootSums(PB._acc(w, idx, G, D), PB._acc(w[:, None] * y, idx, G, D),
                       PB._acc(w[:, None] * y * y, idx, G, D),
                       PB._acc(np.moveaxis(w[None, :, None] * e2, 1, 0), idx, G, D),
                       np.array([bool(odd and direction is not None)]))


def boot_r2(y: np.ndarray, preds: Sequence[np.ndarray], w: np.ndarray, cl: np.ndarray, G: int,
            counts: np.ndarray, direction: np.ndarray | None = None, odd: bool = False
            ) -> tuple[np.ndarray, np.ndarray]:
    """(point [K], draws [B, K]) R2 (R2_wd when ``odd``) of K prediction vectors of one
    component on the given rows, with the cluster bootstrap ``counts`` [B, G]."""
    bs = _boot_sums(y, np.stack(preds), np.asarray(w, float), cl, G, direction, odd)
    return PB.boot_r2(bs, np.ones((1, G)))[0, :, 0], PB.boot_r2(bs, counts)[:, :, 0]


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), 0-based."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1]
    ends = np.r_[starts[1:], len(xs)]
    r = np.empty(len(x))
    for s0, e0 in zip(starts, ends, strict=True):
        r[order[s0:e0]] = 0.5 * (s0 + e0 - 1)
    return r


def _wrank_boot(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """[B, n] weighted mid-ranks (``probes.wrank``) of x under each weight row of W."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1]
    grp = np.zeros(len(x), int)
    grp[order] = np.cumsum(np.isin(np.arange(len(x)), starts)) - 1
    tot = np.add.reduceat(W[:, order], starts, axis=1)               # [B, U]
    rank_g = np.cumsum(tot, axis=1) - 0.5 * tot
    return rank_g[:, grp]


def _wpearson_rows(rx: np.ndarray, ry: np.ndarray, W: np.ndarray) -> np.ndarray:
    sw = W.sum(1, keepdims=True)
    mx, my = (W * rx).sum(1, keepdims=True) / sw, (W * ry).sum(1, keepdims=True) / sw
    cx, cy = rx - mx, ry - my
    num = (W * cx * cy).sum(1)
    den = np.sqrt((W * cx * cx).sum(1) * (W * cy * cy).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


def spearman_boot(x: np.ndarray, y: np.ndarray, w: np.ndarray, cl: np.ndarray,
                  counts: np.ndarray, direction: np.ndarray | None = None, odd: bool = False
                  ) -> tuple[float, np.ndarray]:
    """(point, draws [B]) of the weighted Spearman rho; for a mirror-odd component with
    ``direction``, the direction-weight-weighted mean of the within-direction rho."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    cl = np.asarray(cl, int)
    Wall = np.vstack([w[None], w[None] * np.asarray(counts, float)[:, cl]])   # [1 + B, n]
    groups = [np.ones(len(x), bool)] if (direction is None or not odd) else \
        [np.asarray(direction) > 0, np.asarray(direction) <= 0]
    num, den = np.zeros(len(Wall)), np.zeros(len(Wall))
    for gmask in groups:
        if gmask.sum() < 2:
            continue
        Wg = Wall[:, gmask]
        rho = _wpearson_rows(_wrank_boot(x[gmask], Wg), _wrank_boot(y[gmask], Wg), Wg)
        tw = Wg.sum(1)
        num += tw * rho
        den += tw
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return float(out[0]), out[1:]


def _ci(draws: np.ndarray, level: float = PB.CI_LEVEL) -> tuple[float, float]:
    lo, hi = PB.ci(np.asarray(draws, float), level)
    return float(lo), float(hi)


def geometry_clusters(tab) -> tuple[np.ndarray, int]:
    """(cluster index [n], number of clusters): the geometry, the bootstrap unit."""
    g = ST.geometry_index(tab)
    return g, int(g.max()) + 1


# ------------------------------------------------------------------------- C1


def _names_odd(names) -> np.ndarray:
    return np.array([PB.is_odd(f"lam_{nm}") for nm in names])


@dataclass(frozen=True)
class C1:
    """C1's statistic, its nulls and its decision (module docstring)."""

    names: tuple
    r2c: np.ndarray              # [m] per component
    stat: float                  # mean over the components
    ci: tuple[float, float]
    null_r2c: dict               # name -> [m] (random_critics: the max over seeds, per comp)
    null_diff: dict              # name -> (point, lo, hi) of stat - null
    spearman: np.ndarray         # [m]
    spearman_ci: np.ndarray      # [m, 2]
    slope: np.ndarray            # [m] pooled slope of lambda on g, b_lg (reward units)
    slope_ci: np.ndarray         # [m, 2]
    n_rows: np.ndarray           # [m] scored rows per component
    decision: str
    label: bool
    calibrated: Calibrated
    slope_gl: np.ndarray | None = None       # [m] pooled slope of g on lambda, b_gl
    slope_gl_ci: np.ndarray | None = None    # [m, 2]
    inv_slope_ci: np.ndarray | None = None   # [m, 2] CI of 1 / b_lg
    n_stratum: int | None = None             # the stratum's rows (``rows``), any component
    n_uncalibrated: np.ndarray | None = None  # [m] valid rows dropped: no out-of-fold fit
    reason: str | None = None                # why 'not testable'

    def record(self) -> dict:
        def lst(v):
            return None if v is None else np.asarray(v).tolist()

        extra = {} if self.reason is None else {"reason": self.reason}
        return {**extra, "names": list(self.names), "r2c": self.r2c.tolist(), "stat": self.stat,
                "ci": list(self.ci), "null_r2c": {k: np.asarray(v).tolist()
                                                  for k, v in self.null_r2c.items()},
                "null_diff": {k: list(v) for k, v in self.null_diff.items()},
                "spearman": self.spearman.tolist(), "spearman_ci": self.spearman_ci.tolist(),
                "slope": self.slope.tolist(), "slope_ci": self.slope_ci.tolist(),
                "slope_gl": lst(self.slope_gl), "slope_gl_ci": lst(self.slope_gl_ci),
                "inv_slope_lg_ci": lst(self.inv_slope_ci),
                "n_rows": self.n_rows.tolist(), "n_stratum": self.n_stratum,
                "n_uncalibrated": lst(self.n_uncalibrated), "decision": self.decision,
                "label_at_scale": self.label,
                "calibration": {"a": self.calibrated.a.tolist(), "b": self.calibrated.b.tolist(),
                                "clamped": self.calibrated.clamped.tolist(),
                                "clip_share": lst(self.calibrated.clip_share),
                                "slope": CAL_SLOPE, "k_lev": K_LEV}}


def c1_decide(lo: float, hi: float, null_lo: Sequence[float], spearman_lo: Sequence[float],
              confirm: float = C1_CONFIRM, refute: float = C1_REFUTE) -> str:
    """C1's bands (Section 6)."""
    if lo > confirm and all(v > 0.0 for v in null_lo) and all(v > 0.0 for v in spearman_lo):
        return "confirm"
    if hi < refute:
        return "refute"
    return "inconclusive"


def _slope_draws(x: np.ndarray, y: np.ndarray, W: np.ndarray, groups: list[np.ndarray]
                 ) -> np.ndarray:
    """[B] WLS slopes of y on x under each weight row of W [B, n], centred within each group
    under that row's weights."""
    num, den = np.zeros(len(W)), np.zeros(len(W))
    for m in groups:
        Wg = W[:, m]
        sw = Wg.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = (Wg * x[m]).sum(1, keepdims=True) / sw
            my = (Wg * y[m]).sum(1, keepdims=True) / sw
        cx = np.where(np.isfinite(mx), x[m] - mx, 0.0)
        cy = np.where(np.isfinite(my), y[m] - my, 0.0)
        num += (Wg * cx * cy).sum(1)
        den += (Wg * cx * cx).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def pooled_slopes(g: np.ndarray, lam: np.ndarray, w: np.ndarray, cl: np.ndarray,
                  counts: np.ndarray | None, direction: np.ndarray | None = None,
                  odd: bool = False) -> dict:
    """The label's pooled (not cross-fitted) slopes (module docstring 'C1'): b_lg of lambda on
    g and b_gl of g on lambda, WLS with the metric's weights times the leverage weights of the
    regressor (within direction for an odd component), their points and, with ``counts``,
    their bootstrap draws on the cluster resamples (and those of 1 / b_lg)."""
    g, lam, w = np.asarray(g, float), np.asarray(lam, float), np.asarray(w, float)
    grp = _groups(direction, odd, len(g))
    v_g = cal_weights(g, w, direction, odd) / np.where(w > 0, w, 1.0)
    v_l = cal_weights(lam, w, direction, odd) / np.where(w > 0, w, 1.0)
    W0 = w[None]
    out = {"lg": float(_slope_draws(g, lam, W0 * v_g, grp)[0]),
           "gl": float(_slope_draws(lam, g, W0 * v_l, grp)[0])}
    if counts is not None:
        cw = w[None] * np.asarray(counts, float)[:, np.asarray(cl, int)]
        out["lg_draws"] = _slope_draws(g, lam, cw * v_g, grp)
        out["gl_draws"] = _slope_draws(lam, g, cw * v_l, grp)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["inv_lg_draws"] = np.where(out["lg_draws"] > 0, 1.0 / out["lg_draws"], np.inf)
    return out


def _pooled_slope_boot(g: np.ndarray, lam: np.ndarray, w: np.ndarray, cl: np.ndarray,
                       counts: np.ndarray) -> tuple[float, np.ndarray]:
    """The pooled slope of lambda on g (``pooled_slopes``' b_lg) and its bootstrap draws."""
    ps = pooled_slopes(g, lam, w, cl, counts)
    return ps["lg"], ps["lg_draws"]


def _ci_inf(draws: np.ndarray, level: float = PB.CI_LEVEL) -> tuple[float, float]:
    """A percentile CI that keeps infinite draws (1 / b_lg with b_lg <= 0) as infinite."""
    d = np.asarray(draws, float)
    if not len(d) or np.all(np.isnan(d)):
        return float("nan"), float("nan")
    a = (1.0 - level) / 2.0
    fin = d[np.isfinite(d)]
    if not np.isinf(d).any():
        return float(np.quantile(d, a)), float(np.quantile(d, 1.0 - a))
    if not len(fin):
        return tuple(float(np.sign(np.nanmedian(d)) * np.inf) for _ in range(2))
    # infinite draws as huge finite ones (numpy interpolates inf - inf to NaN); a quantile
    # beyond every finite draw involves an infinite one, so it is infinite
    big = np.where(np.isinf(d), np.sign(d) * 1e300, d)
    q = [float(np.quantile(big, p)) for p in (a, 1.0 - a)]
    return tuple(np.inf if v > fin.max() else (-np.inf if v < fin.min() else v) for v in q)


def _null_array(x, n: int) -> np.ndarray:
    """A null's gradient as [n, m], or [F, n, m] per held-out fold as given."""
    x = np.asarray(x, float)
    return x if x.ndim == 3 else x.reshape(n, -1)


def c1(g: np.ndarray, lam: np.ndarray, outer: np.ndarray, w: np.ndarray, cluster: np.ndarray,
       n_clusters: int, direction: np.ndarray, rows: np.ndarray,
       nulls: dict[str, np.ndarray | list[np.ndarray]] | None = None,
       valid: np.ndarray | None = None, names: Sequence[str] = K_P,
       counts: np.ndarray | None = None, fit_rows: np.ndarray | None = None,
       slope_boot: bool | str = True, bounds=None) -> C1:
    """C1 on ``rows`` (the interior stratum) of g [n, m] against lambda [n, m] (winsorised
    targets, columns ``names``). ``nulls``: name -> a gradient [n, m], or [F, n, m] per
    held-out fold (``FOLD_NULLS``: ``calibrate`` fits and predicts fold f from slice f), put
    through the same calibration, or a list of them (the random critics: the per-draw
    maximum). ``fit_rows`` defaults to ``rows``; ``valid`` [n, m] the per-component row mask
    (observability, K2b); ``bounds`` [m, 2] the targets' winsorisation bounds the calibrated
    predictions are clipped to (the critic's, every null's and every random critic's alike).
    A component is scored on its valid rows that have an out-of-fold calibration (a fold
    with fewer than 3 training rows has none: its rows are dropped and counted,
    'n_uncalibrated', never scored against a filled-in prediction); the nulls are scored on
    the same rows (a null without a prediction on one of them predicts 0 there). A
    component left without a row drops out of the mean, and with none left C1 is 'not
    testable' ('reason'). The pooled slopes' CIs are bootstrapped for a confirm
    (``slope_boot`` True: the label's), always ('always': Section 6's 'slope CIs per
    component and stratum') or never (False)."""
    names = tuple(names)
    n = len(w)
    g = np.asarray(g, float).reshape(n, -1)
    lam = np.asarray(lam, float).reshape(n, -1)
    m = lam.shape[1]
    rows = np.asarray(rows, bool)
    outer = np.asarray(outer, int)
    direction = np.asarray(direction)
    fit_rows = rows if fit_rows is None else np.asarray(fit_rows, bool)
    val = np.ones((n, m), bool) if valid is None else np.asarray(valid, bool)
    val = val & np.isfinite(lam) & np.isfinite(g)
    counts = PB.cluster_counts(n_clusters) if counts is None else np.asarray(counts)
    odd = _names_odd(names)
    cal = calibrate(g, lam, outer, w, fit_rows, val, direction=direction, odd=odd,
                    bounds=bounds)
    nulls = dict(nulls or {})
    null_cal = {}
    for k_, v in nulls.items():
        vs = v if isinstance(v, list) else [v]
        null_cal[k_] = [calibrate(_null_array(x, n), lam, outer, w, fit_rows, val,
                                  direction=direction, odd=odd, bounds=bounds).pred for x in vs]
    B = len(counts)
    r2c, draws = np.zeros(m), np.zeros((B, m))
    null_pt = {k_: np.zeros((len(v), m)) for k_, v in null_cal.items()}
    null_dr = {k_: np.zeros((B, len(v), m)) for k_, v in null_cal.items()}
    sp, sp_ci = np.zeros(m), np.zeros((m, 2))
    slope, slope_ci = np.full(m, np.nan), np.full((m, 2), np.nan)
    slope_gl, gl_ci = np.full(m, np.nan), np.full((m, 2), np.nan)
    inv_ci = np.full((m, 2), np.nan)
    n_rows, n_uncal = np.zeros(m, int), np.zeros(m, int)
    scored = rows[:, None] & val & np.isfinite(cal.pred)      # [n, m]: the scored rows
    for k in range(m):
        sel = scored[:, k]
        n_rows[k] = int(sel.sum())
        n_uncal[k] = int((rows & val[:, k]).sum()) - n_rows[k]
        if not sel.any():
            r2c[k], draws[:, k] = np.nan, np.nan
            continue
        preds = [cal.pred[sel, k]]
        spans = []
        for k_, ps in null_cal.items():
            spans.append((k_, len(preds), len(ps)))
            preds += [np.where(np.isfinite(p[sel, k]), p[sel, k], 0.0) for p in ps]
        pt, dr = boot_r2(lam[sel, k], preds, w[sel], cluster[sel], n_clusters, counts,
                         direction[sel], bool(odd[k]))
        r2c[k], draws[:, k] = pt[0], dr[:, 0]
        for k_, i0, cnt in spans:
            null_pt[k_][:, k] = pt[i0:i0 + cnt]
            null_dr[k_][:, :, k] = dr[:, i0:i0 + cnt]
        s0, sd = spearman_boot(g[sel, k], lam[sel, k], w[sel], cluster[sel], counts,
                               direction[sel], bool(odd[k]))
        sp[k], sp_ci[k] = s0, _ci(sd)
        ps = pooled_slopes(g[sel, k], lam[sel, k], w[sel], cluster[sel], None, direction[sel],
                           bool(odd[k]))
        slope[k], slope_gl[k] = ps["lg"], ps["gl"]
    ok = n_rows > 0                     # a component with no row left drops out of the mean
    if not ok.any():
        n_f = len(np.unique(outer[rows]))
        reason = ("no row" if not rows.any() else
                  f"the rows span {n_f} outer fold: no out-of-fold calibration" if n_f < 2 else
                  "no component keeps a valid row with an out-of-fold calibration")
        return C1(names, r2c, float("nan"), (float("nan"), float("nan")), {}, {}, sp, sp_ci,
                  slope, slope_ci, n_rows, "not testable", False, cal, slope_gl, gl_ci, inv_ci,
                  int(rows.sum()), n_uncal, reason)
    stat = float(r2c[ok].mean())
    st_draws = draws[:, ok].mean(1)
    lo, hi = _ci(st_draws)
    null_diff, null_r2c = {}, {}
    for k_ in null_cal:
        # a null's R2_c is the mean over the components; the random critics' is the maximum
        # of that mean over the seeds, per bootstrap draw (Section 6: 'the maximum R2_c')
        pt_k = null_pt[k_][:, ok].mean(1).max()
        dr_k = null_dr[k_][:, :, ok].mean(2).max(1)
        dl, dh = _ci(st_draws - dr_k)
        null_diff[k_] = (float(stat - pt_k), dl, dh)
        null_r2c[k_] = null_pt[k_][int(np.argmax(null_pt[k_][:, ok].mean(1)))]
    decision = c1_decide(lo, hi, [v[1] for v in null_diff.values()], sp_ci[ok, 0])
    if slope_boot == "always" or (slope_boot and decision == "confirm"):
        # the label's slope CIs (Section 12 decision 2), or every stratum's (descriptive)
        for k in np.flatnonzero(ok):
            sel = scored[:, k]
            ps = pooled_slopes(g[sel, k], lam[sel, k], w[sel], cluster[sel], counts,
                               direction[sel], bool(odd[k]))
            slope_ci[k] = _ci(ps["lg_draws"])
            gl_ci[k] = _ci(ps["gl_draws"])
            inv_ci[k] = _ci_inf(ps["inv_lg_draws"])
    label = bool(decision == "confirm" and np.all(np.isfinite(gl_ci[ok]))
                 and np.all(gl_ci[ok, 0] >= SCALE_BAND[0])
                 and np.all(inv_ci[ok, 1] <= SCALE_BAND[1]))
    return C1(names, r2c, stat, (lo, hi), null_r2c, null_diff, sp, sp_ci, slope, slope_ci,
              n_rows, decision, label, cal, slope_gl, gl_ci, inv_ci, int(rows.sum()), n_uncal)


# ------------------------------------------------------------------------- C1 secondaries


def c1_pattern(g: np.ndarray, lam: np.ndarray, outer: np.ndarray, w: np.ndarray,
               cluster: np.ndarray, n_clusters: int, direction: np.ndarray, rows: np.ndarray,
               valid: np.ndarray, names: Sequence[str], primary: Sequence[str] = K_P,
               secondary: Sequence[str] = SECONDARY, counts: np.ndarray | None = None,
               bounds=None) -> dict:
    """C1-pattern: R2_c over K_P minus R2_c over {vy, r, delta, pedal} (``valid`` restricts
    delta and pedal to bound-inactive states), paired bootstrap; confirm when the CI's lower
    end is > 0. C1's calibration (within direction for the odd components, the training-fold
    mean intercept), so a zero or pure-noise gradient scores about 0 on every component and
    cannot confirm."""
    names = tuple(names)
    counts = PB.cluster_counts(n_clusters) if counts is None else counts
    odd = _names_odd(names)
    cal = calibrate(g, lam, outer, w, rows, valid, direction=np.asarray(direction), odd=odd,
                    bounds=bounds)
    pts, drs = {}, {}
    for k, nm in enumerate(names):
        sel = rows & valid[:, k] & np.isfinite(lam[:, k])
        pt, dr = boot_r2(lam[sel, k], [cal.pred[sel, k]], w[sel], cluster[sel], n_clusters,
                         counts, direction[sel], bool(odd[k]))
        pts[nm], drs[nm] = pt[0], dr[:, 0]
    a = np.mean([pts[n] for n in primary]) - np.mean([pts[n] for n in secondary])
    d = np.mean([drs[n] for n in primary], 0) - np.mean([drs[n] for n in secondary], 0)
    lo, hi = _ci(d)
    return {"r2c": {k: float(v) for k, v in pts.items()}, "diff": float(a), "ci": [lo, hi],
            "decision": "confirm" if lo > 0 else "not confirmed"}


def head_disagreement_boot(q: np.ndarray, w: np.ndarray, cl: np.ndarray, counts: np.ndarray
                           ) -> tuple[float, np.ndarray]:
    """(D_q, draws [B]) of ``critic.head_disagreement``: mean_t var_k(q^k_t) /
    var_t(mean_k q^k_t), weighted over states (heads' variance with ddof 1)."""
    q = np.asarray(q, float)
    within = q.var(axis=1, ddof=1)
    mean = q.mean(axis=1)
    Wall = np.vstack([w[None], w[None] * np.asarray(counts, float)[:, cl]])
    sw = Wall.sum(1)
    num = (Wall * within).sum(1) / sw
    mu = (Wall * mean).sum(1) / sw
    den = (Wall * (mean[None] - mu[:, None]) ** 2).sum(1) / sw
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return float(out[0]), out[1:]


def c1_heads(quantities: dict[str, np.ndarray], w: np.ndarray, cluster: np.ndarray,
             n_clusters: int, rows: np.ndarray, counts: np.ndarray | None = None) -> dict:
    """C1-heads (Section 6): D_q per quantity (``critic.head_quantities``' keys) and the two
    pre-registered paired comparisons D_grad(K_P) - D_V > 0 and D_grad({vy, r, delta,
    pedal}) - D_grad(K_P) > 0 (confirm when the CI's lower end is > 0)."""
    counts = PB.cluster_counts(n_clusters) if counts is None else counts
    pts, drs = {}, {}
    for name, q in quantities.items():
        pts[name], drs[name] = head_disagreement_boot(np.asarray(q)[rows], w[rows],
                                                      cluster[rows], counts)
    kp = [f"g_{c}" for c in K_P]
    sec = [f"g_{c}" for c in SECONDARY]
    out = {"D": pts}
    for label, a, b in (("grad_KP_minus_V", kp, ["V"]), ("grad_secondary_minus_KP", sec, kp)):
        pa = np.mean([pts[n] for n in a]) - np.mean([pts[n] for n in b])
        da = np.mean([drs[n] for n in a], 0) - np.mean([drs[n] for n in b], 0)
        lo, hi = _ci(da)
        out[label] = {"diff": float(pa), "ci": [lo, hi],
                      "decision": "confirm" if lo > 0 else "not confirmed"}
    return out


def c1_limit(cal: Calibrated, g: np.ndarray, lam: np.ndarray, outer: np.ndarray, w: np.ndarray,
             cluster: np.ndarray, n_clusters: int, wall_side: np.ndarray, sd: np.ndarray,
             names: Sequence[str], counts: np.ndarray | None = None,
             components: Sequence[str] = ("n", "xi"), direction: np.ndarray | None = None,
             valid: np.ndarray | None = None) -> dict:
    """C1-limit (Section 6): at states whose own corridor row is strictly active
    (``wall_side`` +1 left wall, -1 right wall, 0 none), the interior-fitted calibration of
    g_n and g_xi applied out of fold, and the mean signed excess d (g_cal - lambda) / SD_T,
    d = -wall_side (away from the binding wall: decreasing n at the left wall); confirm when
    the CI's lower end of the mean over the components is > 0. Slip contacts have no wall
    direction and are left out."""
    names = list(names)
    counts = PB.cluster_counts(n_clusters) if counts is None else counts
    sel = np.asarray(wall_side) != 0
    if not sel.any():
        return {"n": 0, "decision": "not testable"}
    pred = cal.apply(g, outer, direction)              # unclipped: a signed excess
    d = -np.asarray(wall_side, float)
    ok = np.ones((len(w), len(names)), bool) if valid is None else np.asarray(valid, bool)
    per = {}
    tot, used = None, 0
    for nm in components:
        k = names.index(nm)
        sk = sel & ok[:, k]
        if not sk.any():
            per[nm] = {"n": 0, "excess": None, "ci": None}
            continue
        cw = np.vstack([w[None], w[None] * np.asarray(counts, float)[:, cluster]])[:, sk]
        e = d[sk] * (pred[sk, k] - lam[sk, k]) / sd[k]
        e = np.where(np.isfinite(e), e, 0.0)
        val = (cw * e).sum(1) / cw.sum(1)
        per[nm] = {"n": int(sk.sum()), "excess": float(val[0]), "ci": list(_ci(val[1:]))}
        tot = val if tot is None else tot + val
        used += 1
    if not used:
        return {"n": int(sel.sum()), "per_component": per, "decision": "not testable"}
    tot = tot / used
    lo, hi = _ci(tot[1:])
    return {"n": int(sel.sum()), "per_component": per, "excess": float(tot[0]), "ci": [lo, hi],
            "decision": "confirm" if lo > 0 else "not confirmed"}


def _passed(ok) -> bool:
    """A precondition's pass: a scalar boolean (Python or NumPy) that is True. None (not
    computed), NaN, numbers, strings and arrays of more than one value count as failed."""
    if ok is None:
        return False
    a = np.asarray(ok)
    return bool(a.shape == () and a.dtype == np.bool_ and bool(a))


def c1_own_status(k3b_trained_pass: bool | None, value_accuracy_pass: bool | None) -> str:
    """C1-own's standing per network (Section 12 decision 3, Section 13 'Built before the
    tag'): 'secondary' only where the network passed K3b's trained analogue and its critic's
    value accuracy (``value_accuracy``); otherwise 'descriptive (<the failed
    preconditions>)', e.g. 'descriptive (value accuracy)'. A precondition not computed
    (None) counts as failed; NumPy booleans count as booleans (``_passed``)."""
    failed = [name for name, ok in (("K3b", k3b_trained_pass),
                                    ("value accuracy", value_accuracy_pass)) if not _passed(ok)]
    return "secondary" if not failed else f"descriptive ({', '.join(failed)})"


def c1_own(g: np.ndarray, lam_own: np.ndarray, outer: np.ndarray, w: np.ndarray,
           cluster: np.ndarray, n_clusters: int, direction: np.ndarray, rows: np.ndarray,
           k3b_trained_pass: bool, names: Sequence[str] = K_P,
           counts: np.ndarray | None = None, valid: np.ndarray | None = None,
           value_accuracy_pass: bool | None = None) -> dict:
    """C1-own (Section 6, Section 12 decision 3): C1's bands and sign rules with the policy's
    own costate lambda^own (``own.own_costate``) in place of lambda*, on the rows where it was
    computed on a valid run; a pre-registered secondary only where the network passed K3b's
    trained analogue and its critic's value accuracy (Section 13: a precondition for
    C1-own's validity), otherwise descriptive (``c1_own_status``). No nulls: it is the first
    link of the decomposition (critic against its own policy)."""
    status = c1_own_status(k3b_trained_pass, value_accuracy_pass)
    validity = {"k3b_trained": None if k3b_trained_pass is None else _passed(k3b_trained_pass),
                "value_accuracy": None if value_accuracy_pass is None
                else _passed(value_accuracy_pass)}
    ok = np.asarray(rows, bool) & np.all(np.isfinite(lam_own), axis=1)
    if ok.sum() < 10:
        return {"decision": "not testable", "n": int(ok.sum()), "status": status,
                "validity": validity}
    c = c1(g, lam_own, outer, w, cluster, n_clusters, direction, ok, None, valid, names, counts)
    rec = c.record()
    rec["status"] = status
    rec["validity"] = validity
    return rec


def c1_dev(r2c_by_snapshot: dict[str, Sequence[tuple[int, float]]]) -> dict:
    """C1-dev (Section 6): per network, the Spearman rho of R2_c across its snapshots with log
    timesteps; per variant, 'confirm' when rho > 0 in at least 2 of its 3 seeds."""
    rho = {}
    for net, pairs in r2c_by_snapshot.items():
        ts, r = (np.asarray(v, float) for v in zip(*pairs, strict=True)) if pairs else ([], [])
        if len(ts) < 3:
            rho[net] = float("nan")
            continue
        rx, ry = _avg_ranks(np.log(ts)), _avg_ranks(r)
        rho[net] = float(np.corrcoef(rx, ry)[0, 1]) if ry.std() > 0 else 0.0
    out = {"rho": rho, "variants": {}}
    for v in registry.variants():
        seeds = registry.variant_seeds(v)
        pos = [s for s in seeds if np.isfinite(rho.get(s, np.nan)) and rho[s] > 0]
        have = [s for s in seeds if np.isfinite(rho.get(s, np.nan))]
        out["variants"][v] = ("confirm" if len(pos) >= 2 else
                              ("not confirmed" if len(have) == len(seeds) else "not testable"))
    return out


def delta_c1(z: np.ndarray, g: np.ndarray, lam: np.ndarray, fo: PB.Folds, w: np.ndarray,
             direction: np.ndarray, names: Sequence[str]) -> dict:
    """Delta C1 = R2(z + g) - R2(z) (ridge probes, conditional probing): does the gradient
    add to the input?"""
    cols = [f"lam_{n}" for n in names]
    base = PB.ridge_cv(z, lam, fo, w, None, cols)
    full = PB.ridge_cv(z, lam, fo, w, g, cols)
    r0, r1 = PB.probe_r2(base, w, direction), PB.probe_r2(full, w, direction)
    return {"r2_z": r0.tolist(), "r2_z_g": r1.tolist(), "delta": (r1 - r0).tolist(),
            "mean": float(np.mean(r1 - r0))}


# ------------------------------------------------------------------------- C2, C2-sw


def stratum_folds(fo: PB.Folds, rows: np.ndarray) -> PB.Folds:
    return fo.subset(np.asarray(rows, bool))


def c2_network(h3: np.ndarray, random_h3: Sequence[np.ndarray], z: np.ndarray, m: np.ndarray,
               lam_res: np.ndarray, fo: PB.Folds, w_all, direction: np.ndarray,
               names: Sequence[str], passing: dict, rows: np.ndarray, weights=None,
               counts: np.ndarray | None = None, grid=PB.GRID, h_grid=PB.H_GRID,
               mask: np.ndarray | None = None) -> dict:
    """C2 on the stratum ``rows`` (``PROBE_FIT``): ``lam_res`` [F, n, m] (or [n, m]) over all
    of T, the folds dealt on T. ``w_all`` is a callable giving the metric's weights of a row
    subset (``states.weights`` of the subset table) or the weights of the stratum's rows.
    ``mask`` [n, m]: component k is scored only on rows & mask[:, k] (fitted on the whole
    stratum; ``states.decision_mask``). Returns the record, with 'not testable' when no
    component passes the power floor (or none passing keeps a scored row)."""
    rows = np.asarray(rows, bool)
    cols = [f"lam_{n}" for n in names]
    pas = PB._passing(passing, tuple(cols))
    if not pas.any():
        return {"decision": "not testable", "reason": "no component passes the power floor"}
    sub = stratum_folds(fo, rows)
    w = np.asarray(w_all, float) if weights is None else np.asarray(weights, float)
    Y = np.asarray(lam_res, float)
    Y = Y[:, rows] if Y.ndim == 3 else Y[rows]
    mk = None if mask is None else np.asarray(mask, bool)[rows]
    gn = PB.c2(h3[rows], [r[rows] for r in random_h3], z[rows], m[rows], Y, sub, w,
               np.asarray(direction)[rows], cols, passing, None, counts, grid, h_grid, mask=mk)
    rec = gn.record()
    rec["n_stratum"] = int(rows.sum())
    if mk is not None:
        rec["scored_rows"] = mk.sum(0).tolist()
    return rec


def c2_sw_network(h3: np.ndarray, random_h3: Sequence[np.ndarray], z: np.ndarray,
                  sigma_pedal: np.ndarray, m_pedal: np.ndarray, fo: PB.Folds, w: np.ndarray,
                  rows: np.ndarray, counts: np.ndarray | None = None, grid=PB.GRID,
                  h_grid=PB.H_GRID) -> dict:
    """C2-sw on its stratum ``rows`` (``states.SW_STRATUM``: the folded pedal-command
    multiplier active, where sigma^ped is nonzero; the raw pedal-saturated flag is a
    descriptive stratum), with the co-reported Spearman rho(m_pedal - 1, sigma^ped)
    (weighted)."""
    rows = np.asarray(rows, bool)
    sub = stratum_folds(fo, rows)
    gn = PB.c2_sw(h3[rows], [r[rows] for r in random_h3], z[rows], sigma_pedal[rows], sub, w,
                  None, counts, grid, h_grid)
    rec = gn.record()
    rec["n_stratum"] = int(rows.sum())
    rec["spearman_margin"] = PB.wspearman(np.asarray(m_pedal)[rows] - 1.0,
                                          np.asarray(sigma_pedal)[rows], w)
    return rec


def c2_pairs(tab, pred: np.ndarray, lam: np.ndarray, w: np.ndarray, names: Sequence[str],
             ds_tol: float = 2.5, dv_tol: float = 1.0, min_dlam: float = 0.0,
             counts: np.ndarray | None = None) -> dict:
    """C2-pairs (Section 6): pairs of rows of one geometry with matched decoys (|delta (s -
    s_apex)| < 2.5 m, |delta vx| < 1 m/s, the same direction) from different keys; per
    component, the weighted Spearman of the probe's predicted delta lambda against the true
    delta lambda over the pairs (pair weight: the product of the rows' weights), with a
    geometry bootstrap."""
    g, G = geometry_clusters(tab)
    ds = np.asarray(tab["ds_apex"], float)
    vx = np.asarray(tab["vx"], float)
    key = np.asarray(tab["key_id"])
    dirn = np.asarray(tab["direction"])
    I, J = [], []
    for gi in np.unique(g):
        idx = np.flatnonzero(g == gi)
        for a_ in range(len(idx)):
            i = idx[a_]
            jj = idx[a_ + 1:]
            ok = ((np.abs(ds[jj] - ds[i]) < ds_tol) & (np.abs(vx[jj] - vx[i]) < dv_tol)
                  & (key[jj] != key[i]) & (dirn[jj] == dirn[i]))
            I += [i] * int(ok.sum())
            J += jj[ok].tolist()
    I, J = np.asarray(I, int), np.asarray(J, int)
    if not len(I):
        return {"n_pairs": 0}
    counts = PB.cluster_counts(G) if counts is None else counts
    out = {"n_pairs": len(I), "components": {}}
    pw = w[I] * w[J]
    for k, nm in enumerate(names):
        dl = lam[I, k] - lam[J, k]
        keep = np.abs(dl) > min_dlam
        if keep.sum() < 3:
            continue
        pt, dr = spearman_boot(pred[I, k][keep] - pred[J, k][keep], dl[keep], pw[keep],
                               g[I][keep], counts)
        out["components"][nm] = {"rho": pt, "ci": list(_ci(dr)), "n": int(keep.sum())}
    return out


# ------------------------------------------------------------------------- per network, variant


def network_decisions(c1_rec: dict | None, c2_rec: dict | None, c2sw_rec: dict | None) -> dict:
    """{C1, C2, C2-sw: decision} of one network ('missing' without a record)."""
    def dec(r):
        return "missing" if r is None else str(r.get("decision", "missing"))

    return {"C1": dec(c1_rec), "C2": dec(c2_rec), "C2-sw": dec(c2sw_rec)}


def variant_decisions(per_network: dict[str, dict]) -> dict:
    """R18's rule (step 1's ``variant_verdict``) per variant and hypothesis over the variant's
    3 seeds (registry ids G1, G1s1, G1s2, ...; 'missing' where a network has no decision)."""
    from apexzero.interp.analysis import variant_verdict

    out = {}
    for v in registry.variants():
        seeds = registry.variant_seeds(v)
        out[v] = {}
        for h in ("C1", "C2", "C2-sw"):
            ver = {s: per_network.get(s, {}).get(h, "missing") for s in seeds}
            verdict, n = variant_verdict(ver)
            out[v][h] = {"verdict": verdict, "seeds_agreeing": int(n), "seeds": ver}
    return out


# ------------------------------------------------------------------------- C3-w


def _strata_index(nets: Sequence[str], strata: dict[str, str]) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {}
    for i, n in enumerate(nets):
        out.setdefault(strata[n], []).append(i)
    return {k: np.asarray(v, int) for k, v in sorted(out.items())}


def _centred_ranks(v: np.ndarray) -> np.ndarray:
    r = _avg_ranks(v)
    return r - r.mean()


def c3_w(x: dict[str, float], y: dict[str, float], strata: dict[str, str] | None = None,
         n_perm: int = C3_PERMUTATIONS, rng: np.random.Generator | None = None,
         covariates: dict[str, dict[str, float]] | None = None, alpha: float = C3_ALPHA,
         strong: float = C3_STRONG) -> dict:
    """C3-w (module docstring). ``x``, ``y``: per network (C3_STRATA's keys); networks
    missing from either are left out and listed."""
    strata = C3_STRATA if strata is None else strata
    nets = [n for n in strata if n in x and n in y and np.isfinite(x[n]) and np.isfinite(y[n])]
    missing = [n for n in strata if n not in nets]
    rng = stream_rng("costate_c3") if rng is None else rng
    idx = _strata_index(nets, strata)
    X = np.array([x[n] for n in nets], float)
    Y = np.array([y[n] for n in nets], float)
    sizes = {s: len(i) for s, i in idx.items()}
    N = sum(sizes.values())
    obs = 0.0
    null = np.zeros(int(n_perm))
    per = {}
    for s, i in idx.items():
        rx, ry = _centred_ranks(X[i]), _centred_ranks(Y[i])
        den = np.sqrt((rx @ rx) * (ry @ ry))
        rho = float(rx @ ry / den) if den > 0 else 0.0
        per[s] = {"n": len(i), "rho": rho}
        obs += len(i) / N * rho
        if den > 0:
            perm = rng.permuted(np.tile(ry, (int(n_perm), 1)), axis=1)
            null += len(i) / N * (perm @ rx) / den
    p = float((1 + np.sum(null >= obs - 1e-12)) / (1 + len(null)))
    ub = float(obs - np.quantile(null, alpha))
    decision = ("confirm" if p <= alpha else
                ("rules out a strong association" if ub < strong else "inconclusive"))
    out = {"n": N, "missing": missing, "strata": per, "rho": float(obs), "p": p,
           "upper_bound": ub, "null_sd": float(null.std()),
           "critical": float(np.quantile(null, 1.0 - alpha)), "decision": decision}
    if covariates:
        out["partial"] = {name: _partial_c3(X, Y, np.array([cov.get(n, np.nan) for n in nets],
                                                           float), idx, nets)
                          for name, cov in covariates.items()}
    return out


def _partial_c3(X, Y, C, idx, nets: Sequence[str]) -> dict:
    """Stratified partial Spearman: within each stratum the centred ranks of x and y
    residualised on the centred ranks of the covariate, size-weighted over the strata it
    scores (descriptive). A stratum with a network lacking the covariate (step 2 has no
    fields-a2 rows for two base-v2 seeds, so no overlap) is skipped whole, the weights
    renormalised over the scored strata, and the networks without it listed ('missing')."""
    tot, n_scored, skipped = 0.0, 0, []
    for s, i in idx.items():
        if not np.all(np.isfinite(C[i])):
            skipped.append(s)
            continue
        rx, ry, rc = _centred_ranks(X[i]), _centred_ranks(Y[i]), _centred_ranks(C[i])
        cc = rc @ rc
        if cc > 0:
            rx = rx - (rx @ rc) / cc * rc
            ry = ry - (ry @ rc) / cc * rc
        den = np.sqrt((rx @ rx) * (ry @ ry))
        tot += len(i) * (float(rx @ ry / den) if den > 0 else 0.0)
        n_scored += len(i)
    return {"rho": float(tot / n_scored) if n_scored else float("nan"),
            "complete": not skipped, "n": n_scored, "strata_skipped": skipped,
            "missing": [n for n, c in zip(nets, C, strict=True) if not np.isfinite(c)]}


# ------------------------------------------------------------------------- C3-int


def double_centre(A: np.ndarray, iters: int = DOUBLE_CENTRE_ITERS,
                  tol: float = DOUBLE_CENTRE_TOL) -> np.ndarray:
    """A [networks, geometries] (nan = missing) minus row and column effects: alternating
    row and column centring of the present cells until both means vanish (one pass on a
    complete table)."""
    A = np.array(A, float)
    have = np.isfinite(A)
    for _ in range(int(iters)):
        with np.errstate(invalid="ignore"):
            r = np.nanmean(np.where(have, A, np.nan), axis=1)
        A = A - np.where(np.isfinite(r), r, 0.0)[:, None]
        with np.errstate(invalid="ignore"):
            c = np.nanmean(np.where(have, A, np.nan), axis=0)
        A = A - np.where(np.isfinite(c), c, 0.0)[None, :]
        if np.nanmax(np.abs(np.where(np.isfinite(c), c, 0.0)), initial=0.0) <= tol:
            break
    return np.where(have, A, np.nan)


def _corr_present(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    den = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / den) if den > 0 else 0.0


def _c3_int_stat(res: dict, beh: dict, pairs) -> tuple[float, dict]:
    per = {}
    for comp, b, sign in pairs:
        if comp in res and b in beh:
            per[f"{comp}~{b}"] = sign * _corr_present(res[comp].ravel(), beh[b].ravel())
    vals = [v for v in per.values() if np.isfinite(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


def c3_int(residuals: dict[str, np.ndarray], behaviour: dict[str, np.ndarray],
           blocks: np.ndarray, pairs=C3_INT_PAIRS, n_perm: int = C3_INT_PERMUTATIONS,
           rng: np.random.Generator | None = None, alpha: float = C3_ALPHA) -> dict:
    """C3-int (module docstring). ``residuals``: component -> [networks, geometries] signed
    calibrated critic residual (sigma-folded); ``behaviour``: name -> [networks, geometries];
    ``blocks`` [geometries] the R tercile x theta class label of each geometry."""
    rng = stream_rng("costate_c3", 1) if rng is None else rng
    res = {k: double_centre(v) for k, v in residuals.items()}
    beh_raw = {k: np.asarray(v, float) for k, v in behaviour.items()}
    beh = {k: double_centre(v) for k, v in beh_raw.items()}
    stat, per = _c3_int_stat(res, beh, pairs)
    if not np.isfinite(stat):
        return {"decision": "not testable", "pairs": per}
    blocks = np.asarray(blocks)
    n_net, n_geo = next(iter(beh_raw.values())).shape
    groups = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    null = np.empty(int(n_perm))
    for d in range(int(n_perm)):
        perm = np.empty((n_net, n_geo), int)
        for i in range(n_net):
            p = np.arange(n_geo)
            for gidx in groups:
                p[gidx] = gidx[rng.permutation(len(gidx))]
            perm[i] = p
        rows = np.arange(n_net)[:, None]
        bp = {k: double_centre(v[rows, perm]) for k, v in beh_raw.items()}
        null[d] = _c3_int_stat(res, bp, pairs)[0]
    null = null[np.isfinite(null)]
    p = float((1 + np.sum(null >= stat - 1e-12)) / (1 + len(null)))
    return {"stat": stat, "pairs": per, "p": p, "n_perm": len(null),
            "decision": "confirm" if p <= alpha else "not confirmed"}


# ------------------------------------------------------------------------- W1, descriptives


def calibrated_agreement(x: np.ndarray, target: np.ndarray, outer: np.ndarray, w: np.ndarray,
                         cluster: np.ndarray, n_clusters: int, rows: np.ndarray,
                         counts: np.ndarray | None = None, bounds=None) -> dict:
    """The calibrated R2 (C1's calibration, out of fold; ``bounds`` the target's own
    winsorisation bounds where given) of one signal against one target on ``rows``, its CI,
    the weighted Spearman and the pooled slope: C3-w's x (the critic's uniform
    inside-widening derivative against Pi^in) and W1's comparisons."""
    counts = PB.cluster_counts(n_clusters) if counts is None else counts
    x, t = np.asarray(x, float)[:, None], np.asarray(target, float)[:, None]
    cal = calibrate(x, t, outer, w, rows, bounds=bounds)
    sel = rows & np.isfinite(t[:, 0]) & np.isfinite(x[:, 0])
    pt, dr = boot_r2(t[sel, 0], [np.nan_to_num(cal.pred[sel, 0])], w[sel], cluster[sel],
                     n_clusters, counts)
    rho, rd = spearman_boot(x[sel, 0], t[sel, 0], w[sel], cluster[sel], counts)
    s0, sd = _pooled_slope_boot(x[sel, 0], t[sel, 0], w[sel], cluster[sel], counts)
    return {"r2c": float(pt[0]), "ci": list(_ci(dr[:, 0])), "spearman": rho,
            "spearman_ci": list(_ci(rd)), "slope": s0, "slope_ci": list(_ci(sd)),
            "n": int(sel.sum())}


def value_accuracy(v_critic: np.ndarray, v_target: np.ndarray | None, outer: np.ndarray,
                   w: np.ndarray, rows: np.ndarray | None = None,
                   cluster: np.ndarray | None = None, n_clusters: int | None = None,
                   counts: np.ndarray | None = None, threshold: float = VALUE_ACCURACY_R2,
                   n_folds: int = ST.N_FOLDS, target_name: str = VALUE_TARGET,
                   rows_name: str = VALUE_ROWS) -> dict:
    """Value accuracy (Section 3.3 'Value accuracy', Section 12 decision 7; Section 13 'Built
    before the tag': a precondition for C1-own's validity): the out-of-fold R2 of the critic's
    value sigma_R V_c (reward units, ``critic.Pullback.value``) against ``v_target`` on
    ``rows``, with the frozen outer folds ``outer``. The decision's target is the
    deterministic policy's own discounted return from those states (``VALUE_TARGET``,
    ``own.own_return``, on the own stage's rows, ``VALUE_ROWS``); the optimum's
    horizon-corrected window value V*inf_t on T's interior (``VALUE_TARGET_VSTAR``) is scored
    the same way as a descriptive. (The critic regresses on neither: it is fitted to its
    stochastic training policy's bootstrapped returns.) Per held-out fold, C1's affine
    calibration (``calibrate``: slope >= 0, the training-fold mean intercept) of the target on
    sigma_R V_c is fitted on the other folds' rows and predicts the fold's rows; R2 = 1 -
    SSE/SST pooled over the folds, under the metric's weights ``w``. Passes at R2 >=
    ``threshold`` (0.8) with every requested row (``rows`` with w > 0) scored: a requested row
    without a finite critic value or target ('n_missing', 'n_missing_target') or without an
    out-of-fold prediction (a fold that cannot be calibrated) fails it. An anti-aligned or
    uninformative value scores R2 <= about 0. Reported beside it (descriptive): the raw R2
    (sigma_R V_c read as the target itself, no calibration), the weighted Spearman, the
    per-fold slopes and intercepts and, with ``cluster``, ``n_clusters`` and ``counts``, the
    cluster-bootstrap CI (the rule reads the point)."""
    v = np.asarray(v_critic, float).ravel()
    w = np.asarray(w, float)
    out = {"threshold": float(threshold), "target": target_name, "rows": rows_name}
    if v_target is None:
        return {**out, "r2": float("nan"), "pass": False, "n": 0,
                "reason": f"no {target_name} column"}
    t = np.asarray(v_target, float).ravel()
    rows = np.ones(len(t), bool) if rows is None else np.asarray(rows, bool)
    req = rows & (w > 0)
    miss = int((req & ~np.isfinite(v)).sum())
    miss_t = int((req & ~np.isfinite(t)).sum())
    ok = req & np.isfinite(v) & np.isfinite(t)
    out.update(n=int(ok.sum()), n_requested=int(req.sum()), n_missing=miss,
               n_missing_target=miss_t)
    if ok.sum() < 3:
        return {**out, "r2": float("nan"), "pass": False, "reason": "fewer than 3 rows"}
    cal = calibrate(v[:, None], t[:, None], outer, w, ok, ok[:, None], n_folds)
    pred = cal.pred[:, 0]
    sel = ok & np.isfinite(pred)
    r = float(PB.r2(t[sel], pred[sel], w[sel])[0])
    out.update(r2=r, n_scored=int(sel.sum()),
               raw_r2=float(PB.r2(t[ok], v[ok], w[ok])[0]),
               spearman=PB.wspearman(v[ok], t[ok], w[ok]),
               slope=cal.b[:, 0].tolist(), intercept=cal.a[:, 0, 0].tolist(),
               clamped=cal.clamped[:, 0].tolist())
    out["pass"] = bool(np.isfinite(r) and r >= threshold and miss == 0 and miss_t == 0
                       and sel.sum() == ok.sum())
    why = [f"{miss} of {int(req.sum())} rows lack a finite critic value" if miss else "",
           f"{miss_t} of {int(req.sum())} rows lack a finite target" if miss_t else "",
           f"{int(ok.sum() - sel.sum())} rows without an out-of-fold prediction"
           if sel.sum() != ok.sum() else ""]
    if any(why):
        out["reason"] = "; ".join(x for x in why if x)
    if cluster is not None and n_clusters is not None and counts is not None:
        _, dr = boot_r2(t[sel], [pred[sel]], w[sel], np.asarray(cluster)[sel], int(n_clusters),
                        counts)
        out["ci"] = list(_ci(dr[:, 0]))
    return out


# ------------------------------------------------------------------------- strata tables
#
# Sections 4/6/12/13's descriptive per-stratum tables, per network and hypothesis: each
# hypothesis' statistic recomputed on each stratum with the same statistics (Section 12
# decision 4: 'reported with the same statistics, descriptively'), never a decision. A
# stratum is '&'-joined terms over T's columns (``stratum_rows``): a boolean column
# (``states.key_rows``' strata, or a per-network column, ``PER_NETWORK_COLUMNS``), '~' for
# its complement, 'phase:<p>' (``states.PHASES``), 'direction:<+1|-1>' or 'W:<400|800>'.
# 'interior' is k = 10 (``states.INTERIOR_K``, the primary), 'interior20' k = 20;
# 'corner_band' is Section 12's corner-phase stratum s_c0 <= s < s_c1 (all of T's states
# there, and its interior part); the braking and non-braking keys are the interior's
# sub-scores (Section 4); left and right ('direction:'), W 400 and 800, pedal-saturated and
# kappa-clipped are Section 4's 'Strata' (kappa clipping under the network's own obs_rms,
# ``inputs.flags``). C2-sw's tables start from its decision stratum (the pedal-switching
# steps, Section 13) and report it per phase (Section 6: approach and exit separately). Its
# full-throttle rows also fall in 'corner' (the apex to s_c1) and in 'braking' (from s_c0 on
# a key without a sustained brake, ``states.phases``), so all four phase rows are listed and
# they split the stratum; 'exit' means s >= s_c1 in every table. The raw pedal-saturated flag
# is descriptive (Section 13). Each list is frozen in its order: the C1 rows' bootstrap
# streams are ``STRATUM_SEED`` + the row's index.

PER_NETWORK_COLUMNS = ("kappa_clipped",)   # strata columns from each network's own flags
C1_STRATA = ("interior", "interior20", "near_limit", "contact", "corner_band",
             "interior&corner_band", "interior&phase:approach", "interior&phase:braking",
             "interior&phase:corner", "interior&phase:exit", "interior&braking_key",
             "interior&~braking_key", "interior&direction:1", "interior&direction:-1",
             "interior&W:400", "interior&W:800", "interior&pedal_saturated",
             "interior&kappa_clipped")
C2_STRATA = ("interior", "interior20", "near_limit", "contact", "corner_band",
             "interior&corner_band", "interior&braking_key", "interior&~braking_key",
             "interior&phase:approach", "interior&phase:braking", "interior&phase:corner",
             "interior&phase:exit", "interior&direction:1", "interior&direction:-1",
             "interior&W:400", "interior&W:800", "interior&pedal_saturated",
             "interior&kappa_clipped")
C2SW_STRATA = (ST.SW_STRATUM, f"{ST.SW_STRATUM}&phase:approach",
               f"{ST.SW_STRATUM}&phase:braking", f"{ST.SW_STRATUM}&phase:corner",
               f"{ST.SW_STRATUM}&phase:exit", "pedal_saturated",
               f"{ST.SW_STRATUM}&braking_key", f"{ST.SW_STRATUM}&~braking_key",
               f"{ST.SW_STRATUM}&direction:1", f"{ST.SW_STRATUM}&direction:-1",
               f"{ST.SW_STRATUM}&W:400", f"{ST.SW_STRATUM}&W:800",
               f"{ST.SW_STRATUM}&kappa_clipped")
STRATA = {"C1": C1_STRATA, "C2": C2_STRATA, "C2-sw": C2SW_STRATA}
PRIMARY_STRATUM = {"C1": "interior", "C2": "interior", "C2-sw": ST.SW_STRATUM}
STRATUM_SEED = 50             # _subclusters' bootstrap seeds of the strata tables (50 + index)


def strata_tab(tab, flags: dict | None = None) -> dict:
    """T's columns plus the per-network strata columns (``PER_NETWORK_COLUMNS``) from one
    network's ``inputs.flags`` (absent ones stay absent: their strata are 'not testable')."""
    out = dict(tab)
    for c in PER_NETWORK_COLUMNS:
        if flags is not None and c in flags:
            out[c] = np.asarray(flags[c], bool)
    return out


def _terms(expr: str) -> list[tuple[bool, str]]:
    return [(t.startswith("~"), t.removeprefix("~")) for t in str(expr).split("&")]


def stratum_rows(tab, expr: str) -> np.ndarray:
    """[n] bool: the rows of one stratum expression (the section comment above)."""
    n = len(np.asarray(tab["t"]))
    m = np.ones(n, bool)
    for neg, term in _terms(expr):
        if term.startswith("phase:"):
            v = np.asarray(tab["phase"]).astype(str) == term[len("phase:"):]
        elif term.startswith("direction:"):
            v = np.asarray(tab["direction"]) == int(term[len("direction:"):])
        elif term.startswith("W:"):
            v = np.asarray(tab["W"]) == int(term[len("W:"):])
        elif term in tab:
            v = np.asarray(tab[term], bool)
        else:
            raise KeyError(f"stratum {expr!r}: no column {term!r}")
        m &= ~v if neg else v
    return m


def missing_columns(tab, expr: str) -> list[str]:
    """The per-network columns a stratum needs that ``tab`` lacks (``strata_tab``)."""
    return [t for _, t in _terms(expr) if t in PER_NETWORK_COLUMNS and t not in tab]


def strata_rows(tab, names: Sequence[str]) -> dict[str, np.ndarray | None]:
    """Each stratum's rows; None where a per-network column is absent (``missing_columns``).
    Any other unknown term raises (``stratum_rows``)."""
    return {nm: None if missing_columns(tab, nm) else stratum_rows(tab, nm) for nm in names}


def strata_counts(tab, names: Sequence[str]) -> dict[str, int | None]:
    """Each stratum's row count (None without its per-network column): coverage only."""
    return {nm: None if r is None else int(r.sum()) for nm, r in strata_rows(tab, names).items()}


_C1_FIELDS = ("names", "r2c", "stat", "ci", "spearman", "spearman_ci", "slope", "slope_ci",
              "slope_gl", "slope_gl_ci", "inv_slope_lg_ci", "n_rows", "n_uncalibrated",
              "null_diff", "reason")
_C2_FIELDS = ("names", "passing", "r2_base", "r2_full", "gain", "raw", "partial", "stat", "ci",
              "scored_rows", "spearman_margin", "reason")


def stratum_record(hypothesis: str, rec: dict, rows: np.ndarray | None, tab,
                   primary: bool = False) -> dict:
    """One row of a hypothesis' strata table from its record on that stratum (``c1``'s
    ``record()``, ``c2_network``, ``c2_sw_network``): the statistic and its CI, per
    component (R2_c, Spearman and the pooled slopes with their CIs for C1; the gain and the
    R2s for C2 and C2-sw), the nulls' paired differences (C1), the rows, keys and
    geometries, and the band the statistic falls in there ('band': descriptive, never a
    decision; 'primary' marks the decision stratum's own row). ``rows`` None: a stratum whose
    per-network column is absent ('not testable'). A record computed on other rows than the
    stratum's (its 'n_stratum') raises: a row is never written under the wrong stratum."""
    if rows is None:
        return {"primary": False, "descriptive": True, "n": None, "keys": None,
                "geometries": None, "band": "not testable",
                "reason": str(rec.get("reason", "a per-network column is absent"))}
    rows = np.asarray(rows, bool)
    if rec.get("n_stratum") is not None and int(rec["n_stratum"]) != int(rows.sum()):
        raise ValueError(f"{hypothesis}: a record of {rec['n_stratum']} rows written under a "
                         f"stratum of {int(rows.sum())}")
    fields = _C1_FIELDS if hypothesis == "C1" else _C2_FIELDS
    out = {"primary": bool(primary), "descriptive": not primary, "n": int(rows.sum()),
           "keys": len(np.unique(np.asarray(tab["key_id"])[rows])),
           "geometries": len(np.unique(ST.geometry_index(tab)[rows])) if rows.any() else 0}
    out.update({k: rec[k] for k in fields if k in rec})
    out["band"] = str(rec.get("decision", "not testable"))
    return out


def ratio_to_ceiling(r2c: float, r2_mlp: float) -> float:
    """R2_c / R2_MLP-on-z: normalised fidelity (descriptive)."""
    return float(r2c / r2_mlp) if r2_mlp and np.isfinite(r2_mlp) else float("nan")
