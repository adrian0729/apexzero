"""The input side of idea 2: T's rows with their targets, the tail values, the
network-independent observation maps and the per-network input flags (design v2.1, Section 1
'Input-side quantities ... need no network and are computed before the tag'; Sections 3.2,
4 and 7 K6).

- ``key_data``: one A2 key re-polished with its targets (``targets.key_targets``), its T
  rows (``states.key_rows``), the wall side of each row's own corridor contact (+1 left,
  -1 right, 0 none; C1-limit), whether delta's box binds (C1-pattern), the optimum's a*_t and
  a*_{t-1} (PR-2, D2), its tail values V*_t (``nulls.tail_values``) and the
  horizon-corrected V*inf_t (``nulls.horizon_values``, the value fits' target) and, in a float64
  process, its rows' ``critic.Inputs`` and ``critic.ObsJac`` on the stimulus's T64 road
  (``network.road``, network.py:766-769; the lockbox guard runs there).
- ``World``: T's table over a set of keys (``states.concat``), lambda [n, 8], V* [n], the
  per-key network inputs (``nulls.KeyInput``), saved and loaded as one npz (``save_world``,
  ``load_world``; the ObsJac arrays included).
- Per network (its obs_rms only, read from its snapshot: ``obs_rms_of``): z and J_z at the
  rows, the observability rule (``states.observability``, tau_s), the kappa-clipping flag of
  each state (step 2's rule, network.py:110-117: a preview kappa channel at the obs clip,
  |z| >= 10), the valid mask per K_P component for C1 (s dropped where unobservable).
- ``width_derivative`` (Section 3.4 'Road fields', C3-w's x): sigma_R dV_c/dDelta of a
  critic for a uniform widening of the inside edge (every w_in sample + Delta,
  ``inside_widening``) of the road it observes, through step 2's strip_arrays tangent
  (``perturb.perturbed_arrays``, perturb.py:303-317) and ``network.observe`` with the context
  channels (network.py:616-619), by ``jax.jvp``; the state, prev_action and accel are held.
  It is the critic's width field summed over the preview's support, the counterpart of
  Pi^in_t (``targets.Prices.inside``). Like ``critic.pullback`` it refuses a trained critic
  before the tag; a float64 process.

Nothing else here evaluates a network: ``obs_rms_of`` reads a snapshot's stored normaliser.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import RunningMeanStd
from apexzero.costate import critic as CR
from apexzero.costate import nulls as NL
from apexzero.costate import states as ST
from apexzero.costate import stimuli_dir
from apexzero.costate import targets as TG
from apexzero.fields import keys as K
from apexzero.fields import network as NW
from apexzero.fields import perturb as PT
from apexzero.interp import io

OBSJAC_FIELDS = ("ob", "J", "J_prev", "J_acc", "Ax", "Mx", "Ma", "defect", "reset")


def design():
    """(bank, cells) of step 1's stimulus design, read from its outputs."""
    return K.design(stimuli_dir())


@dataclass
class KeyData:
    """One key's input-side data (module docstring)."""

    ref: K.KeyRef
    kt: TG.KeyTargets
    rows: io.Table
    v_star: np.ndarray        # [W+1]
    inp: CR.Inputs | None = None
    oj: CR.ObsJac | None = None


def wall_side(cs: TG.Costate, t: np.ndarray) -> np.ndarray:
    """+1 where x_t's own left corridor row is active, -1 the right, 0 neither."""
    oc = cs.own_corridor[np.asarray(t, int)]
    return np.where(oc[:, 0], 1, np.where(oc[:, 1], -1, 0)).astype(int)


def key_data(ref: K.KeyRef, bank=None, with_obs: bool = True, t_H: int | None = None,
             kt: TG.KeyTargets | None = None) -> KeyData:
    """The input-side data of one A2 key (module docstring); ``with_obs`` needs x64."""
    if bank is None:
        bank, _ = design()
    kt = TG.key_targets(ref, bank=bank) if kt is None else kt
    cs, pr = kt.jac, kt.prices
    lm = bank.stimuli[ref.stim_id].landmarks
    rows = ST.key_rows(cs, pr, ref, ST.strata(ref), lm, t_H=t_H)
    t = np.asarray(rows["t"], int)
    rows["wall_side"] = wall_side(cs, t)
    rows["delta_bound"] = cs.own_box[t, TG.DELTA]
    v_star = NL.tail_values(cs.x, cs.a, float(kt.problem.spec.gamma))
    rows["v_star"] = v_star[t]
    rows["v_star_inf"] = NL.horizon_values(cs.x, cs.a, float(kt.problem.spec.gamma))[t]
    rows["a_steer"], rows["a_pedal"] = cs.a[t, 0], cs.a[t, 1]
    prev = np.where((t > 0)[:, None], cs.a[np.maximum(t - 1, 0)], 0.0)
    rows["a_prev_steer"], rows["a_prev_pedal"] = prev[:, 0], prev[:, 1]
    kd = KeyData(ref, kt, rows, v_star)
    if with_obs:
        kd.inp = CR.key_inputs(cs.x, cs.a, t, anchored=ref.kind == "anchored")
        kd.oj = CR.obs_jacobians(NW.road(bank, ref.stim_id).base, kd.inp)
    return kd


@dataclass
class World:
    """T's rows over a set of keys with the network-side inputs (module docstring)."""

    tab: io.Table
    keys: list[NL.KeyInput]

    @property
    def n(self) -> int:
        return io.check_table(self.tab)

    @property
    def lam(self) -> np.ndarray:
        return np.stack([np.asarray(self.tab[f"lam_{c}"], float) for c in TG.STATE_NAMES], 1)

    @property
    def v_star(self) -> np.ndarray:
        return np.asarray(self.tab["v_star"], float)

    @property
    def v_star_inf(self) -> np.ndarray:
        return np.asarray(self.tab["v_star_inf"], float)


def build_world(kds: list[KeyData]) -> World:
    """T's table (``states.concat``) and the per-key inputs, in key order."""
    tab = ST.concat([k.rows for k in kds])
    keys, at = [], 0
    for k in kds:
        n = io.check_table(k.rows)
        if k.oj is not None:
            keys.append(NL.KeyInput(k.ref.key_id, np.arange(at, at + n), k.inp, k.oj))
        at += n
    return World(tab, keys)


def save_world(path: Path, world: World) -> Path:
    """One npz: the table's columns ('tab/<col>') and every key's rows, inputs and ObsJac
    ('key/<i>/<field>'), with the key order in 'keys.json'."""
    arrays = {f"tab/{k}": np.asarray(v) for k, v in world.tab.items()}
    meta = []
    for i, k in enumerate(world.keys):
        meta.append(k.key_id)
        arrays[f"key/{i}/rows"] = np.asarray(k.rows)
        for f, v in zip(CR.Inputs._fields, k.inp, strict=True):
            arrays[f"key/{i}/inp/{f}"] = np.asarray(v)
        for f in OBSJAC_FIELDS:
            arrays[f"key/{i}/oj/{f}"] = np.asarray(getattr(k.oj, f))
    arrays["keys.json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_world(path: Path) -> World:
    with np.load(Path(path), allow_pickle=False) as d:
        tab = {k[4:]: d[k] for k in d.files if k.startswith("tab/")}
        meta = json.loads(bytes(d["keys.json"]).decode())
        keys = []
        for i, kid in enumerate(meta):
            inp = CR.Inputs(*(d[f"key/{i}/inp/{f}"] for f in CR.Inputs._fields))
            oj = CR.ObsJac(**{f: d[f"key/{i}/oj/{f}"] for f in OBSJAC_FIELDS})
            keys.append(NL.KeyInput(kid, d[f"key/{i}/rows"], inp, oj))
    return World(tab, keys)


# ------------------------------------------------------------------------- per network


def obs_rms_of(name: str) -> tuple[RunningMeanStd, object]:
    """(obs_rms float64 upcast, PPOConfig) of a network's final snapshot (a registry id or a
    run directory), read from the pickle; no parameter is evaluated."""
    from apexzero.fields import stages as S
    from apexzero.interp import agents

    path = S.checkpoint_path(name)
    if path is None:
        raise FileNotFoundError(f"{name}: no finished run with a checkpoint")
    ag = agents.load_agent(path, S.network_agent_id(name))
    return NW.upcast32(ag.obs_rms), ag.cfg


def z_and_jz(world: World, obs_rms: RunningMeanStd) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(z unclipped [n, 222], clipped [n, 222], J_z [n, 222, 8]) of every row."""
    n = world.n
    z = np.full((n, CR.OBS_DIM), np.nan)
    cl = np.zeros((n, CR.OBS_DIM), bool)
    jz = np.full((n, CR.OBS_DIM, 8), np.nan)
    for k in world.keys:
        zz, cc, _ = k.oj.z(obs_rms)
        jj, _ = k.oj.jz(obs_rms)
        z[k.rows], cl[k.rows], jz[k.rows] = zz, cc, jj
    return z, cl, jz


def kappa_clipped(clipped: np.ndarray) -> np.ndarray:
    """[n] a preview kappa channel at the obs clip (step 2's rule at one state)."""
    return np.asarray(clipped, bool)[:, NW.KAPPA_CHANNELS].any(axis=1)


def flags(world: World, obs_rms: RunningMeanStd) -> dict:
    """Per row under one obs_rms: the observability verdicts (``states.observability``),
    kappa clipping, and C1's valid mask over K_P (s only where observable)."""
    _, cl, jz = z_and_jz(world, obs_rms)
    ob = ST.observability(jz, cl)
    valid = np.ones((world.n, len(TG.K_P)), bool)
    valid[:, 0] = ~ob.s_unobservable
    return {"observability": ob, "kappa_clipped": kappa_clipped(cl), "valid_kp": valid,
            "clipped_any": cl.any(axis=1)}


# ------------------------------------------------------------------------- width derivative


def inside_widening(n: int, sigma: int) -> PT.Samples:
    """A uniform widening of the inside edge by 1 m per unit (w_left on a left corner)."""
    one, zero = np.ones(n), np.zeros(n)
    return PT.Samples(zero, one if sigma > 0 else zero, one if sigma < 0 else zero)


@functools.lru_cache(maxsize=8)
def _width_jit(fn):
    def one(params, mean, var, base, samples, delta, x, pa, acc, ctx, ctx_obs):
        def V(e):
            tr = PT.perturbed_arrays(base, samples, PT.Samples(*(e * a for a in delta)))
            ob = NW.observe(NW.env_state(x, pa, acc, ctx), tr, ctx_obs)
            o = jnp.clip((ob - mean) / jnp.sqrt(var + CR.RMS_EPS), -10.0, 10.0)
            return fn(params, o)[0]

        return jax.jvp(V, (jnp.zeros((), x.dtype),), (jnp.ones((), x.dtype),))

    return jax.jit(jax.vmap(one, in_axes=(None,) * 6 + (0, 0, 0, None, None)))


def width_values(net: CR.CriticNet, rd: NW.Road, inp: CR.Inputs, sigma: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(V_c [n], dV_c/dDelta [n]) in the critic's units (module docstring)."""
    CR._refuse_trained(net, "a width derivative")
    NW.require_x64()
    delta = inside_widening(rd.n, int(sigma))
    mean, var = (jnp.asarray(a, jnp.float64) for a in (net.obs_rms.mean, net.obs_rms.var))
    ctx, ctx_obs = jnp.asarray(NW.ctx64()), jnp.asarray(NW.context_channels())
    v, dv = _width_jit(net.fn)(net.params, mean, var, rd.base, rd.samples, delta,
                               jnp.asarray(inp.x), jnp.asarray(inp.prev_action),
                               jnp.asarray(inp.accel), ctx, ctx_obs)
    return np.asarray(v), np.asarray(dv)


def width_derivative(net: CR.CriticNet, rd: NW.Road, inp: CR.Inputs, sigma: int) -> np.ndarray:
    """[n] sigma_R dV_c/dDelta in reward units per m (module docstring)."""
    return net.sigma * width_values(net, rd, inp, sigma)[1]
