"""The policy's own costate: the gradient of the deterministic policy's discounted H-step
return with respect to the state and the previous action (design v2.1, Section 3.3 'The
policy's own costate'; K3b in Section 7; C1-own in Section 6).

**The loop** (``own_loop``). Step 2's frozen ``network.closed_loop`` (network.py:631-661)
with s_stop = +inf, rebuilt here so that a zero perturbation delta_k is added to each state
x_k before the policy reads it and the env steps it (``delta`` [steps, 8]); x_{k+1} is the
env's step from x_k + delta_k (``racing.step_no_reset``, racing.py:144-191). A second zero
perturbation eps_k [steps, 2] is added to the accel the policy observes at step k (the env's
dynamics never read it). Two returns are
accumulated over the first ``count`` steps (a traced cut, so one compile serves every H):

- J = sum_k gamma^k r_k rebuilt from the states and commands, r_k = (s_{k+1} - s~_k)/10
  - 0.02 ||clip(a_k) - prev_k||^2 (racing.py:174-175), s~_k = s_k + delta_k[0] and prev_k the
  clipped previous command (a_{-1} for k = 0): the quantity differentiated. The perturbation
  of x_k enters r_k, never r_{k-1}, so the VJP with respect to delta_k is gamma^k dV_k/dx_k
  with V_k the tail from k (and -e_s/10 included, the costate's convention, targets.py);
- J_env = sum_k gamma^k (the env's own reward of step k): the independent reference K3b's
  finite differences use (no termination occurs on a counted run, so it carries no penalty).

``own_costate`` returns, from one reverse sweep (``jax.vjp``), the accel-held gradients
lambda^own_held,k = gamma^-k dJ/ddelta_k, pi^own_held = dJ/da_{-1} and the policy's own accel
sensitivity gamma^-k dJ/deps_k, and **the on-manifold costate** (the comparison C1-own makes
with the critic's primary, on-manifold pullback, Section 3.2's M2 argument applied to this
side too; a design change from Section 3.3's 'accel held fixed', owned by the design's
author): on the data manifold accel = A(x_t, Phi^-1(x_t; a_{t-1})) is a function of (x_t,
a_{t-1}), so V^pi_det is Markov there like V*, and

    lambda^own_k = lambda^own_held,k + M_x(k)' gamma^-k dJ/deps_k,
    pi^own = pi^own_held + M_a(0)' dJ/deps_0,

with M_x and M_a the maps of ``critic.obs_jacobians`` at (x_k, x_{k-1}, clip(a_{k-1})) (for a
T restart exactly the key row's ObsJac the critic used, ``maps0``; otherwise x_{-1}
recovered by ``critic.prev_state``). A reset start (accel = prev_action = 0, as an A2 key's
t = 0) keeps the held form, as the critic's reset rows do. A run counts only if it is valid
on all ``count`` steps (no termination, no truncation; Section 3.3).

**K3b** (``k3b_start``; Section 7). For one start:
- the delta = 0 loop reproduces ``closed_loop``'s Trace (states, commands, validity) to
  round-off, and J equals J_env;
- the AD gradient of J along each x_0 component and a_{-1} against Richardson central
  differences (h, h/2) of J_env, K1's steps (``targets.FD_H``) and ``critic.PI_H`` for
  a_{-1}; a comparison counts only where the run's discrete structure is the same at -h,
  -h/2, 0, h/2 and h (``_signature``: the commands' saturation set, the obs clip mask, the
  preview's d_max regime and the tyre min(util, 2) per step, vx < 1, the pedal relu knots,
  every pose lookup's 2 m segment (the car's and the 40 preview points'), and the run's
  length) and FD(h) and FD(h/2) agree to O(h^2) (step 2's Gate 2 rule, network.py:76-80);
  the error |AD - FD| / |FD| with Gate 1's zero rule (kkt.py:163) at the difference's
  resolution (``_err``: 100 x the round-off floor 64 eps |J| / h; a 90-step float64 return
  of about 30 resolves dJ above about 4e-5 at h = 1e-6);
- per-state VJPs against restarts from those states (x_k, clip(a_{k-1}), the run's accel
  over step k-1, count - k steps): equal to round-off;
- planted bugs must fail (``OWN_PLANTS``): the action clip dropped from the rebuilt reward,
  a wrong gamma power (gamma^(k+1)), an off-by-one prev_action (the command before the
  previous one);
- the on-manifold check (``k3b_manifold``) at a non-reset restart from the run's own state
  x_k (k = ``MANIFOLD_AT`` x count): x_k (or a_{k-1}) moved, x_{k-1} recovered by Newton on the
  control step at fixed a_{k-1} (``critic.prev_state``), accel recomputed, J_env re-run;
  Richardson differences against the on-manifold lambda^own and pi^own, with the same
  counting rule. On a control that reads the accel channels (``accel_driver``: step 2's
  linear driver with weights on channels 208-209) the held gradients must fail it; on the
  linear driver (dJ/daccel = 0) both coincide.
``k3b_decision`` pools the starts: per component median relative error < ``K3B_MEDIAN`` =
1e-3 and p95 < ``K3B_P95`` = 1e-2, the planted bugs failing that rule. The controls that
survive are step 2's linear driver and its hybrid small-gain random-init actor
(network.py:388-460; Section 7: plain random-init actors leave the road).

**Pre-registration.** ``network.NetPolicy``'s trained mark is honoured: a trained policy's
loop waits for the tag (``network.unlock_trained``, network.py:282-294) exactly as step 2's
fields do. A float64 process (``network.require_x64``).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.costate import critic as CR
from apexzero.costate.targets import FD_H, STATE_NAMES
from apexzero.env import racing
from apexzero.fields import network as NW
from apexzero.fields.network import LoopIn, NetPolicy
from apexzero.interp.constants import OBS_CLIP
from apexzero.vehicle.sim import TrackArrays

GAMMA = 0.995
OWN_PLANTS = ("no_clip", "gamma_power", "prev_off_by_one")
K3B_MEDIAN = 1e-3
K3B_P95 = 1e-2
ZERO_ATOL = 1e-6              # Gate 1's zero-field rule (kkt.py:163)
RICH_RTOL = NW.RICH_RTOL
ROUNDOFF = 64.0               # x eps |J| / h: the round-off floor of FD(h) - FD(h/2)
RESOLVE = 100.0               # a difference resolves |dJ| above this x the round-off floor
TRACE_TOL = 1e-12             # the delta = 0 loop against closed_loop (round-off)
RESTART_RTOL = 1e-9           # per-state VJP against a restart's gradient
FD_STEPS = (-1.0, -0.5, 0.0, 0.5, 1.0)
MANIFOLD_AT = 0.5             # the on-manifold check's restart, as a share of count
ACCEL_WEIGHTS = ((0.0, 0.3), (0.2, 0.0))   # accel_driver: (steer, pedal) x (ax, ay) / 50
DIRECTIONS = (*STATE_NAMES, "prev_steer", "prev_pedal")
STEP_OF = {**FD_H, "prev_steer": CR.PI_H, "prev_pedal": CR.PI_H}


def _keep(alive, new, old):
    return jax.tree.map(lambda a, b: jnp.where(alive, a, b), new, old)


def own_loop(fn, params, obs_rms, tr: TrackArrays, li: LoopIn, steps: int, delta, count,
             gamma: float = GAMMA, plant: str | None = None, eps=None):
    """The loop of the module docstring: (J, J_env, outputs) with outputs x [steps+1, 8],
    valid [steps+1], mean, a [steps, 2], ob [steps, 222], z [steps, 222]. ``eps`` [steps, 2]
    perturbs the accel the policy observes (default 0). Traceable."""
    if plant is not None and plant not in OWN_PLANTS:
        raise ValueError(f"unknown plant {plant!r}")
    st0 = NW.env_state(li.x0, li.prev_action, li.accel, li.ctx)
    aux = jnp.zeros((), jnp.int32)
    lg = jnp.log(jnp.asarray(gamma, li.x0.dtype))

    if eps is None:
        eps = jnp.zeros((steps, 2), li.x0.dtype)

    def body(carry, k):
        st, alive, prev2, J, Je = carry
        safe = _keep(alive, st, st0)
        cur = safe._replace(x=safe.x + delta[k], accel=safe.accel + eps[k])
        ob = NW.observe(cur, tr, li.ctx_obs)
        o, z = NW.policy_input(ob, obs_rms)
        mean = fn(params, o, cur, tr, aux)[0]
        a = jnp.where(k < li.n_forced, li.a_forced[k], mean)
        _, nxt, rew, term, trunc, _ = racing.step_no_reset(cur, a, tr, NW.ENV)
        ac = jnp.clip(a, -1.0, 1.0)
        prev = cur.prev_action
        a_r = a if plant == "no_clip" else ac
        p_r = prev2 if plant == "prev_off_by_one" else prev
        r = (nxt.x[0] - cur.x[0]) / racing.EnvParams().progress_scale \
            - racing.EnvParams().action_cost * jnp.sum((a_r - p_r) ** 2)
        power = k + 1 if plant == "gamma_power" else k
        use = alive & (k < count)
        J = J + jnp.where(use, jnp.exp(lg * power) * r, 0.0)
        Je = Je + jnp.where(use, jnp.exp(lg * k) * rew, 0.0)
        new = _keep(alive, nxt, st)
        out = dict(x=new.x, valid=alive, mean=mean, a=a, ob=ob, z=z)
        return (new, alive & ~term & ~trunc, prev, J, Je), out

    zero = jnp.zeros((), li.x0.dtype)
    (_, _, _, J, Je), outs = jax.lax.scan(
        body, (st0, jnp.array(True), li.prev_action, zero, zero), jnp.arange(steps))
    outs["x"] = jnp.concatenate([li.x0[None], outs["x"]])
    outs["valid"] = jnp.concatenate([jnp.array([True]), outs["valid"]])
    return J, Je, outs


@functools.lru_cache(maxsize=32)
def _vjp_jit(fn, steps: int, plant: str | None = None):
    """jit((params, obs_rms, tr, li, count) -> (J, J_env, outs, dJ/dx0, dJ/da_{-1},
    dJ/ddelta [steps, 8], dJ/deps [steps, 2]))."""
    def run(params, obs_rms, tr, li, count):
        def f(x0, pa, delta, eps):
            J, Je, outs = own_loop(fn, params, obs_rms, tr, li._replace(x0=x0, prev_action=pa),
                                   steps, delta, count, plant=plant, eps=eps)
            return J, (Je, outs)

        d0 = jnp.zeros((steps, 8), li.x0.dtype)
        e0 = jnp.zeros((steps, 2), li.x0.dtype)
        J, vjp, (Je, outs) = jax.vjp(f, li.x0, li.prev_action, d0, e0, has_aux=True)
        gx, gp, gd, ge = vjp(jnp.ones_like(J))
        return J, Je, outs, gx, gp, gd, ge

    return jax.jit(run)


@functools.lru_cache(maxsize=32)
def _forward_jit(fn, steps: int):
    """jit(vmap over (x0, prev_action, accel) of (J, J_env, outs))."""
    def run(params, obs_rms, tr, li, count, x0s, pas, accs):
        def one(x0, pa, acc):
            d0 = jnp.zeros((steps, 8), li.x0.dtype)
            return own_loop(fn, params, obs_rms, tr,
                            li._replace(x0=x0, prev_action=pa, accel=acc), steps, d0, count)

        return jax.vmap(one)(x0s, pas, accs)

    return jax.jit(run)


def _refuse(net: NetPolicy) -> None:
    NW._refuse_trained(net, "the own-costate loop")
    NW.require_x64()


@dataclass(frozen=True)
class OwnCostate:
    """One run's own costate (module docstring): ``lam`` and ``pi`` on the manifold where
    ``mapped`` (held elsewhere and on a reset start), the held forms beside them."""

    J: float
    J_env: float
    lam: np.ndarray           # [count, 8] on-manifold lambda^own_k (held where not mapped)
    pi: np.ndarray            # [2] on-manifold pi^own (held on a reset start)
    grad_x0: np.ndarray       # [8] dJ/dx_0 at held accel (= lam_held[0])
    x: np.ndarray             # [steps+1, 8]
    a: np.ndarray             # [steps, 2]
    valid_all: bool           # valid on every counted step
    count: int
    lam_held: np.ndarray | None = None   # [count, 8] gamma^-k dJ/ddelta_k
    pi_held: np.ndarray | None = None    # [2] dJ/da_{-1}
    daccel: np.ndarray | None = None     # [count, 2] gamma^-k dJ/deps_k
    mapped: np.ndarray | None = None     # [count] bool: on-manifold rows


def inputs_inf(li: LoopIn) -> LoopIn:
    """The loop inputs with s_stop = +inf (Section 3.3)."""
    return li._replace(s_stop=np.float64(np.inf))


def own_costate(net: NetPolicy, tr: TrackArrays, li: LoopIn, steps: int, count: int,
                gamma: float = GAMMA, plant: str | None = None, reset: bool = False,
                maps0: tuple[np.ndarray, np.ndarray] | None = None,
                x_prev: np.ndarray | None = None, all_states: bool = False) -> OwnCostate:
    """lambda^own and pi^own of one run (module docstring). The start's maps: ``maps0`` =
    (M_x [2, 8], M_a [2, 2]) (a T restart: the key row's ObsJac), or from ``x_prev`` (x_{-1}),
    or recovered by Newton; ``reset`` keeps the held form at the start. ``all_states`` maps
    every counted step k >= 1 too (own-run sweeps) from the run's own (x_k, x_{k-1},
    clip(a_{k-1}))."""
    _refuse(net)
    li = inputs_inf(li)
    J, Je, outs, gx, gp, gd, ge = _vjp_jit(net.fn, int(steps), plant)(
        net.params, net.obs_rms, tr, li, int(count))
    k = np.arange(count)
    inv = gamma ** (-k.astype(float))
    lam_h = np.asarray(gd)[:count] * inv[:, None]
    dacc = np.asarray(ge)[:count] * inv[:, None]
    pi_h = np.asarray(gp)
    x, a = np.asarray(outs["x"]), np.asarray(outs["a"])
    lam, pi = lam_h.copy(), pi_h.copy()
    mapped = np.zeros(count, bool)
    if not reset:
        a_prev0 = np.clip(np.asarray(li.prev_action, float), -1.0, 1.0)
        if maps0 is not None:
            Mx0, Ma0 = (np.asarray(v, float) for v in maps0)
        else:
            xp = (CR.prev_state(tr, np.asarray(li.x0, float)[None], a_prev0[None])[0][0]
                  if x_prev is None else np.asarray(x_prev, float))
            oj = CR.obs_jacobians(tr, CR.Inputs(np.asarray(li.x0, float)[None], xp[None],
                                                a_prev0[None], np.zeros(1, bool)))
            Mx0, Ma0 = oj.Mx[0], oj.Ma[0]
        lam[0] = lam_h[0] + Mx0.T @ dacc[0]
        pi = pi_h + Ma0.T @ dacc[0]
        mapped[0] = True
    if all_states and count > 1:
        idx = np.arange(1, count)
        oj = CR.obs_jacobians(tr, CR.Inputs(x[idx], x[idx - 1], np.clip(a[idx - 1], -1.0, 1.0),
                                            np.zeros(len(idx), bool)))
        lam[idx] = lam_h[idx] + np.einsum("ned,ne->nd", oj.Mx, dacc[idx])
        mapped[idx] = True
    valid = np.asarray(outs["valid"])
    return OwnCostate(float(J), float(Je), lam, pi, np.asarray(gx), x, a,
                      bool(valid[:count + 1].all()), int(count), lam_h, pi_h, dacc, mapped)


# ------------------------------------------------------------------------- K3b


def _signature(outs: dict, count: int, raw: bool, ds: float) -> np.ndarray:
    """The discrete structure of one run (module docstring) as a flat int vector."""
    x = np.asarray(outs["x"])[:count + 1]
    mean = np.asarray(outs["mean"])[:count]
    ob = np.asarray(outs["ob"])[:count]
    parts = [np.sign(np.round(np.abs(mean) - 1.0, 14)).astype(int).ravel()]
    if not raw:
        parts.append((np.abs(np.asarray(outs["z"])[:count]) >= OBS_CLIP).astype(int).ravel())
    vx = x[:-1, 3]
    parts.append(np.digitize(6.0 * vx, [150.0, 500.0]).astype(int))
    util = ob[:, CR.EGO["util_f"]:CR.EGO["util_r"] + 1] + 1.0
    parts.append((util >= 2.0).astype(int).ravel())
    parts.append((vx < 1.0).astype(int))
    parts.append(np.digitize(x[:, 7], CR.PEDAL_KNOTS).astype(int))
    pts = x[:-1, :1] + np.asarray(racing.preview_offsets(np.maximum(vx, 0.0)[:, None]))
    seg = np.floor(np.concatenate([x[:, :1], pts], axis=None) / ds).astype(int)
    parts.append(seg)
    parts.append(np.array([int(np.asarray(outs["valid"])[:count + 1].sum())]))
    return np.concatenate(parts)


def _fd_points(x0: np.ndarray, pa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs, ps = [], []
    for d in DIRECTIONS:
        h = STEP_OF[d]
        for c in FD_STEPS:
            x, p = x0.copy(), pa.copy()
            if d in STATE_NAMES:
                x[STATE_NAMES.index(d)] += c * h
            else:
                p[0 if d == "prev_steer" else 1] += c * h
            xs.append(x)
            ps.append(p)
    return np.asarray(xs), np.asarray(ps)


def _err(ad: float, fd: float, floor: float = 0.0) -> float:
    """|AD - FD| / |FD| with Gate 1's zero rule, both at the difference's resolution
    max(ZERO_ATOL, ``RESOLVE`` x the round-off floor of J / h): a gradient the difference
    cannot resolve to 1% is a zero field."""
    z = max(ZERO_ATOL, RESOLVE * floor)
    if abs(ad) <= z and abs(fd) <= z:
        return 0.0
    return abs(ad - fd) / max(abs(fd), z)


@dataclass(frozen=True)
class K3bStart:
    """One start's K3b record (module docstring)."""

    name: str
    count: int
    valid: bool
    trace_err: float
    j_gap: float
    fd: np.ndarray            # [10] Richardson FD of J_env per direction
    ad: np.ndarray            # [10] dJ/d(x_0, a_{-1})
    counted: np.ndarray       # [10] bool
    err: np.ndarray           # [10]
    plants: dict              # plant -> [10] errors (AD of the planted J against the same FD)
    restart_err: float
    manifold: dict | None = None   # ``k3b_manifold``'s record

    def record(self) -> dict:
        return {"name": self.name, "count": self.count, "valid": self.valid,
                "trace_err": self.trace_err, "j_gap": self.j_gap, "fd": self.fd.tolist(),
                "ad": self.ad.tolist(), "counted": self.counted.tolist(),
                "err": self.err.tolist(), "plants": {k: v.tolist() for k, v in self.plants.items()},
                "restart_err": self.restart_err, "directions": list(DIRECTIONS),
                "manifold": self.manifold}


def accel_driver(sigma: int) -> tuple[np.ndarray, np.ndarray]:
    """Step 2's linear driver (``network.linear_driver``) with weights on the accel channels
    (208-209, ``ACCEL_WEIGHTS``): a control whose return depends on the observed accel, so
    the held and on-manifold own costates differ (the K3b plant the linear driver cannot
    be)."""
    W, b = NW.linear_driver(sigma)
    W = W.copy()
    W[:, CR.ACCEL] += np.asarray(ACCEL_WEIGHTS, np.float32)
    return W, b


def _fd_rule(Jes: np.ndarray, sigs: list, valid: bool, J_ref: float, skip=()) -> tuple:
    """Richardson differences per direction from the 5-point runs, the counting rule
    (unchanged structure, O(h^2) consistency, a valid run, ``skip`` directions) and the
    round-off floors."""
    fd, counted, floors = np.zeros(10), np.zeros(10, bool), np.zeros(10)
    for i, d in enumerate(DIRECTIONS):
        h = STEP_OF[d]
        Jm1, Jmh, J0, Jph, Jp1 = Jes[i]
        f1, f2 = (Jp1 - Jm1) / (2 * h), (Jph - Jmh) / h
        rich = (4.0 * f2 - f1) / 3.0
        same = all(np.array_equal(sigs[i][j], sigs[i][2]) for j in range(5))
        floor = ROUNDOFF * np.finfo(float).eps * max(abs(J0), 1.0) / h
        cons = abs(f1 - f2) <= RICH_RTOL * abs(rich) + floor
        fd[i], counted[i] = rich, bool(same and cons and valid and d not in skip)
        floors[i] = floor
    return fd, counted, floors


def k3b_manifold(net: NetPolicy, tr: TrackArrays, li: LoopIn, steps: int, count: int,
                 oc: OwnCostate) -> dict:
    """The on-manifold check (module docstring) at the run's own state x_k, k =
    ``MANIFOLD_AT`` x count: AD (on-manifold and held) against Richardson differences of
    J_env with x_{k-1} recovered by Newton at fixed a_{k-1} and accel recomputed."""
    k = max(int(MANIFOLD_AT * count), 1)
    if k >= count:
        return {"k": k, "count": 0}
    xk, xkm1 = np.asarray(oc.x[k], float), np.asarray(oc.x[k - 1], float)
    ap = np.clip(np.asarray(oc.a[k - 1], float), -1.0, 1.0)
    lk = li._replace(x0=xk, prev_action=ap, accel=np.asarray(CR.accel_of(xk, xkm1)))
    cnt = count - k
    rk = own_costate(net, tr, lk, steps, cnt, x_prev=xkm1)
    ad_man, ad_held = np.r_[rk.lam[0], rk.pi], np.r_[rk.lam_held[0], rk.pi_held]
    xs, ps = _fd_points(xk, ap)
    xp, res = CR.prev_state(tr, xs, ps, guess=np.repeat(xkm1[None], len(xs), 0))
    accs = np.asarray(CR.accel_of(xs, xp))
    fwd = _forward_jit(net.fn, int(steps))
    _, Jes, outs = fwd(net.params, net.obs_rms, tr, lk, int(cnt), xs, ps, accs)
    Jes = np.asarray(Jes).reshape(len(DIRECTIONS), len(FD_STEPS))
    ds = float(np.asarray(tr.ds))
    raw = net.obs_rms is None
    sigs = [_signature({kk: np.asarray(v)[i] for kk, v in outs.items()}, cnt, raw, ds)
            for i in range(len(xs))]
    sigs = [sigs[i * len(FD_STEPS):(i + 1) * len(FD_STEPS)] for i in range(len(DIRECTIONS))]
    # a previous command within h of its clip: the difference straddles the kink
    skip = tuple(d for j, d in enumerate(("prev_steer", "prev_pedal"))
                 if abs(ap[j]) >= 1.0 - STEP_OF[d])
    fd, counted, floors = _fd_rule(Jes, sigs, rk.valid_all, rk.J_env, skip)
    err_man = np.array([_err(ad_man[i], fd[i], floors[i]) for i in range(10)])
    err_held = np.array([_err(ad_held[i], fd[i], floors[i]) for i in range(10)])
    return {"k": k, "count": int(cnt), "ad_manifold": ad_man.tolist(),
            "ad_held": ad_held.tolist(), "fd": fd.tolist(), "counted": counted.tolist(),
            "err_manifold": err_man.tolist(), "err_held": err_held.tolist(),
            "newton": float(np.max(res)), "accel_sens": float(np.max(np.abs(rk.daccel[0])))}


def k3b_start(net: NetPolicy, tr: TrackArrays, li: LoopIn, steps: int, count: int | None = None,
              margin: int = 10, restarts=(0.25, 0.6), name: str = "",
              manifold: bool = True) -> K3bStart:
    """K3b on one start (module docstring). ``count`` defaults to the base run's valid
    length minus ``margin`` steps (at most ``steps``); ``manifold`` adds the on-manifold
    check (``k3b_manifold``)."""
    _refuse(net)
    li = inputs_inf(li)
    raw = net.obs_rms is None
    fwd = _forward_jit(net.fn, int(steps))
    acc0 = np.asarray(li.accel, float)

    def one(x0s, pas, c):
        return fwd(net.params, net.obs_rms, tr, li, int(c), x0s, pas,
                   np.repeat(acc0[None], len(x0s), 0))

    x0, pa = np.asarray(li.x0, float), np.asarray(li.prev_action, float)
    _, _, base = one(x0[None], pa[None], steps)
    valid = np.asarray(base["valid"])[0]
    alive = int(np.argmin(valid)) - 1 if not valid.all() else steps
    if count is None:
        count = max(min(steps, alive - margin), 1)
    # the start's own check is at held accel (a cell start is a reset row)
    oc = own_costate(net, tr, li, steps, count, reset=True)
    ref = jax.jit(NW.closed_loop, static_argnums=(0, 5))(
        net.fn, net.params, net.obs_rms, jax.tree.map(jnp.asarray, tr),
        jax.tree.map(jnp.asarray, li), steps)
    trace_err = float(max(np.max(np.abs(np.asarray(ref.x) - oc.x)),
                          np.max(np.abs(np.asarray(ref.a) - oc.a)),
                          float(np.any(np.asarray(ref.valid) != valid))))
    xs, ps = _fd_points(x0, pa)
    _, Jes, outs = one(xs, ps, count)
    Jes = np.asarray(Jes).reshape(len(DIRECTIONS), len(FD_STEPS))
    ds = float(np.asarray(tr.ds))
    sigs = [_signature({k: np.asarray(v)[i] for k, v in outs.items()}, count, raw, ds)
            for i in range(len(xs))]
    sigs = [sigs[i * len(FD_STEPS):(i + 1) * len(FD_STEPS)] for i in range(len(DIRECTIONS))]
    ad = np.r_[oc.grad_x0, oc.pi_held]
    fd, counted, floors = _fd_rule(Jes, sigs, oc.valid_all, oc.J_env)
    err = np.array([_err(ad[i], fd[i], floors[i]) for i in range(10)])
    plants = {}
    for p in OWN_PLANTS:
        op = own_costate(net, tr, li, steps, count, plant=p, reset=True)
        adp = np.r_[op.grad_x0, op.pi_held]
        plants[p] = np.array([_err(adp[i], fd[i], floors[i]) for i in range(10)])
    rerr = 0.0
    xk_all = oc.x
    for q in restarts:
        k = int(q * count)
        if k < 1:
            continue
        acc = CR.accel_of(xk_all[k], xk_all[k - 1])
        lk = li._replace(x0=np.asarray(xk_all[k]), prev_action=np.clip(oc.a[k - 1], -1.0, 1.0),
                         accel=np.asarray(acc))
        rk = own_costate(net, tr, lk, steps, count - k, reset=True)
        scale = max(np.max(np.abs(oc.lam_held[k])), 1e-12)
        rerr = max(rerr, float(np.max(np.abs(rk.grad_x0 - oc.lam_held[k])) / scale))
    man = k3b_manifold(net, tr, li, steps, count, oc) if manifold else None
    return K3bStart(name, int(count), oc.valid_all, trace_err, abs(oc.J - oc.J_env)
                    / max(abs(oc.J_env), 1e-12), fd, ad, counted, err, plants, rerr, man)


def _rule(errs: np.ndarray, counted: np.ndarray) -> dict:
    out = {}
    for i, d in enumerate(DIRECTIONS):
        e = errs[:, i][counted[:, i]]
        out[d] = {"n": len(e),
                  "median": float(np.median(e)) if len(e) else None,
                  "p95": float(np.quantile(e, 0.95)) if len(e) else None,
                  "pass": bool(len(e) == 0 or (np.median(e) < K3B_MEDIAN
                                               and np.quantile(e, 0.95) < K3B_P95))}
    return out


def k3b_decision(starts: list[K3bStart]) -> dict:
    """K3b's pass over the starts (module docstring)."""
    if not starts:
        return {"pass": False, "reason": "no start"}
    errs = np.stack([s.err for s in starts])
    counted = np.stack([s.counted for s in starts])
    comp = _rule(errs, counted)
    plants = {}
    for p in OWN_PLANTS:
        pe = np.stack([s.plants[p] for s in starts])
        r = _rule(pe, counted)
        plants[p] = {"fails": bool(not all(v["pass"] for v in r.values())), "rule": r}
    ok = (all(v["pass"] for v in comp.values()) and all(v["fails"] for v in plants.values())
          and all(s.trace_err <= TRACE_TOL * 1e3 for s in starts)
          and all(s.j_gap <= 1e-10 for s in starts)
          and all(s.restart_err <= RESTART_RTOL for s in starts)
          and all(s.valid for s in starts))
    out = {"components": comp, "plants": plants, "share_counted": float(counted.mean()),
           "trace_err": max(s.trace_err for s in starts),
           "j_gap": max(s.j_gap for s in starts),
           "restart_err": max(s.restart_err for s in starts)}
    man = [s for s in starts if s.manifold and s.manifold.get("count")]
    if man:
        cm = np.stack([np.asarray(s.manifold["counted"], bool) for s in man])
        em = np.stack([np.asarray(s.manifold["err_manifold"], float) for s in man])
        rule = _rule(em, cm)
        out["manifold"] = {"components": rule, "share_counted": float(cm.mean()),
                           "newton": max(s.manifold["newton"] for s in man),
                           "pass": bool(all(v["pass"] for v in rule.values()))}
        ok = ok and out["manifold"]["pass"] and out["manifold"]["newton"] <= 1e-9
        acc = [s for s in man if s.name.startswith("accel")]
        if acc:
            ca = np.stack([np.asarray(s.manifold["counted"], bool) for s in acc])
            eh = np.stack([np.asarray(s.manifold["err_held"], float) for s in acc])
            rh = _rule(eh, ca)
            out["manifold"]["held_on_accel_plant"] = {
                "fails": bool(not all(v["pass"] for v in rh.values())), "rule": rh}
            ok = ok and out["manifold"]["held_on_accel_plant"]["fails"]
    out["pass"] = bool(ok)
    return out
