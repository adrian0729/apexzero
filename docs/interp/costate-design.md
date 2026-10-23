# Idea 2: does the network encode the costate? (design v2.2, not frozen)

Status: 22 Oct 2026.
- **v1** was reviewed on four lenses (math, probing, feasibility, science): 42 findings were
  confirmed and 2 rejected.
- **v2** applies the confirmed findings; Section 11 maps each one to its change.
- **v2.1** records the lead's decisions on the open questions (Section 12). They were taken before
  step 2's confirmatory decisions existed.
- **v2.2** records the implementation in `src/apexzero/costate/` (merged 22 Oct) and the decisions
  it required (Section 13). The implementation was adversarially reviewed: 25 findings were
  confirmed and fixed.

Nothing here has been computed on a trained network: no activation, gradient or probe. The
numbers come from step 1's key files, run metadata (`runs/*/config.json`, `runs/*/eval_rdev.json`,
`runs/*/ckpt` listings) and the reviewers' scratch checks on key files. Input-side quantities
(z at the plan states under each network's obs_rms, clip flags, ridge or MLP on z, the decoy and
value-fit models) need no network and are computed before the tag.

The rules of steps 1 and 2 carry over:
- design, review and gates on controls with known answers come first. Then a tag,
  `costate-prereg-v1`, is placed before any trained network's activation or gradient is computed.
- R_TEST stays sealed. Every new strip (K2b) is synthetic and passes the lockbox guard.
- Step 1's and step 2's code stays frozen. Idea 2's code lives in `src/apexzero/costate/` and
  `scripts/costate.py`, and imports `fields` and `interp` read-only.
- Idea 2 runs after step 2's confirmatory decisions (C3 needs them, Section 6).

## 1. Question

The TO-OCP's window problem has an optimal value V*_W: W steps of the agent's own reward, no
terminal value, the OCP's margins (3 cm inside the env's edge, 10⁻⁴ rad inside the slip limit),
nominal grip and the full known road. Along its optimum, the costate λ_t is the gradient of the
remaining value with respect to the state: the shadow price of each state component, what one
more metre per second or one more metre of lateral position is worth. The defect multipliers
give it exactly, in discrete form, at every state of every answer key (Section 3.1).

Three questions follow:
1. **Is the critic's gradient the costate?** If the critic were the window problem's V*, its
   input gradient, pulled back to the car's state, would equal λ. Even the training MDP's V*
   differs from that (Section 3.3), and the critic estimates V^π. Value regression fits values,
   not gradients, and PPO critics fit the true value poorly (Ilyas et al. 2020).
2. **Do the hidden activations (actor and critic) carry the costate** in a linearly accessible
   form, beyond the input, the output and smooth position-and-speed variables?
3. **Does costate fidelity go with good driving,** across networks and, within a network,
   across geometries?

**What is new, narrowly.** Adaptive-critic designs train a critic to output λ = ∂J/∂x
directly: DHP and GDHP (Werbos; Prokhorov & Wunsch 1997), value-gradient learning (Fairbank &
Alonso 2012), and Sobolev training (Czarnecki et al. 2017). Probes of RL agents usually target
human concepts or environment variables (McGrath et al. 2022). Here an emergent value-regression
critic in deep on-policy RL is checked against the exact per-state costate of the same reward, on
a constrained nonlinear problem, with targets whose sign is verified and checked by finite
differences (Section 7).

**Link to step 2.** The optimum's inside-corridor multiplier prices the inside wall, and G* = 1
because that contact is active. λ_n already carries this price at the level of the state
(Section 3.2); W1 is its road-coordinate counterpart (Section 3.4). A policy that keeps a buffer
(H2′) has an inactive inside corridor of its own, so even a perfect critic of that policy prices
the inside wall near 0. A value-level deficit is therefore judged against the policy's own width
price, not against the optimum's (Section 3.4).

## 2. What exists

**A2 keys** (`keys.a2_refs`, keys.py:7-15). There are 481 verified keys, with states identical for
every network:
- 142 nominal and 339 phase keys; 363 at W = 400 and 118 at W = 800;
- 136 cells and 68 mirror pairs, over 48 geometries (R, θ) with 2–24 keys each. R takes 23 values
  from 6 to 300 m, and θ ∈ {45, 60, 90, 135, 180}°;
- tier A has 276 keys (W 12 m), tier C 163 (W 9, 12 and 15 m) and tier V 42 (the horizon twins);
- 4 keys are flagged `multi_basin`;
- the decision is a*_0 at x_0, with a_{−1} = 0 (keys.py:273-276).

**Anchored finals** (keys.py:16-22). There are 8,136 agent-specific keys, all fast approaches,
for step 1's 12 agents only (`interp/agents.py` AGENTS; 393–766 per agent). Each is solved from
x_{L−1} with a_0 fixed (keys.py:414-416), so the decision is a*_1 at x_1. Every step 1 NLP has
a_{−1} = 0, anchored ones included (window.py:479-490; answer_key.py:613-616 pins only a_0).
Naming hazard: the key JSON's `anchor['a_prev']` is a_{L−1}, the fixed a_0, not a_{−1}; a_{L−2}
is not in the key.

**Stored arrays** (keys.py:3-5): x [W+1, 8], a [W, 2], lam_g [12W] and lam_x [10W + 8]. Step 2's
re-polish kept only the movement, so idea 2 re-polishes with `keys.repolish` (keys.py:545-561).

**The layout of lam_g** (window.py:498-501). Step 1's keys are unscaled (kkt.py:679-680).

| Rows | Constraint |
|---|---|
| 8t + i, t = 0…W−1 | Defect x_{t+1} − Φ(x_t, a_t) = 0 |
| 8W + 2t + k | Corridor at x_{t+1}. Left (k = 0): n − (w_l − 1.03) ≤ 0. Right (k = 1): −(w_r − 1.03) − n ≤ 0 (window.py:331-334) |
| 10W + 2t + k | Slip α_f/α_max and α_r/α_max at x_{t+1}, within ±(1 − 10⁻⁴/α_max) (window.py:695-697) |

lam_x holds x_t at 8t + i (t = 0…W) and a_t at 8(W+1) + 2t + j (j = 0 steer, 1 pedal).

**The sign.** The convention is f = −J and L = f + lam_gᵀg + lam_xᵀw (keys.py:34-37, 464-469;
kkt.py:8). A bound shift c then gives dJ*/dc = +lam. Step 2's Gate 0 pins this convention, and the
planted flip_lam_g fails Gate 1 with median error 0.77 (receptive-fields design, Section 14).

x_0 is fixed by its bounds (keys.py:411). IPOPT's default `fixed_variable_treatment =
make_parameter` fills a fixed variable's multiplier from the Lagrangian gradient, so
lam_x[0:8] = Φ_x(x_0, a*_0)ᵀ lam_g[0:8] − e_s/10 holds by construction. That λ_0 means
dJ*_W/dx_0 rests on K1's x_0-bound differences and on the chain rule from lam_g[0:8], which K1's
defect-row shifts test.

**One key, read from its file** (illustrative; no network): key `03b6076863e3adfe335e6757`
(`syn/R13.1/T180/W12/typical/L`, W 400).
- At x_1 (55.5 m/s): λ_1 = (−0.0057, −0.053, −0.72, +0.028, …) over (s, n, ξ, vx, …). No
  constraint acts on x_1, so the two forms of Section 3.1 agree there.
- λ_vx is −0.077 at t = 20 (braking, 37.9 m/s) and +0.134 at t = 60 (14.2 m/s, λ_ξ = 5.7).
- λ_vx changes sign twice. The first is between t = 6 and 7, one step after the pedal command
  crosses zero at the brake point. The second is between t = 52 and 53, at the end of braking:
  about 2 m before the geometric apex, 1–2 steps before the inside contact, and 5–6 steps before
  v_min (t = 58). It leads the throttle reapplication at t = 58.
- Priced constraints: the inside corridor on x_54 and x_55 (rows 53–54: 0.030, 0.117); the
  outside corridor on x_101 (row 100: 0.070); the rear slip limit on x_44–x_48 (rows 43–47); and
  the pedal-state and pedal-command bounds on 303 and 305 steps. The corridor and slip rows are the
  only priced road and grip constraints.
- At the inside contact x_55, λ_n = +0.102 after the jump (the target) and −0.052 before it.
- λ_s ≈ 0 from t ≈ 85, once no s-dependent constraint lies ahead (10⁻¹² at t = 100 and 200).

**Horizon twins** (tier V, from the reviewers' check). On T's grid, the largest
|λ^400 − λ^800| is at most 0.54% of the pooled SD on the 6 A2 twins (p90 0.14% for λ_s, 0.27%
for λ_vx), and at most 2.8% on the 2 imposed twins. λ_vx at t = 1 on R13/T180 is 0.0274 at both
horizons. Divergence starts at t ≈ W − 132 and is vx-only. At W = 400 the plan ends near top speed
(vx_W 86.9–87.2 m/s).

**Networks.** Every run has an MLP actor and an MLP critic, each with 3 tanh layers of 256. All
use γ 0.995, GAE λ 0.95, 2,048 envs × 64-step rollouts, observation and reward normalisation, and
obs_noise 0 (`runs/*/config.json`; ppo.py:147-157, 183-217). Each run keeps 16 snapshots, from
10M to 200M timesteps (`runs/*/ckpt`). The runs are step 1's 12 finals (G1–G3, their 6
replicates, S1–S3; interp/registry.py:82-97), H4's real-width v1/v3 (seeds 0–3) and baselines
v1/v3 (seeds 0–4), and H4-transfer's LOO f1/f2 (seeds 0–3). With the v2 baselines s3 and s4 that
makes **31 generalists**, each with a final checkpoint and an `eval_rdev.json` (12 R-dev circuits).

| Network | Activations | Notes |
|---|---|---|
| Actor | z (222) → h1, h2, h3 (256 tanh) → mean m (2, pre-clip) | The env clips to [−1, 1] (racing.py:150); log_std is state-independent (ppo.py:194-195) |
| Critic | z → h1, h2, h3 (256 tanh) → V | V is in units of reward/σ_R, with σ_R = √(ret_rms.var + 10⁻⁸) (ppo.py:425-429; audit/rollouts.py:30-34) |
| Heads | 5 × [Dense 64, ReLU, Dense 1] on stop_gradient(h3) | Generalists only; S1–S3 have none (ppo.py:207-217) |

`Critic.apply` returns V, or V and the head outputs, never hidden units (ppo.py:207-217);
Section 4 gives idea 2's extractor. The input is z = clip((o − mean)/√(var + 10⁻⁸), ±10)
(ppo.py:282-283). A snapshot holds the actor, critic, obs_rms and ret_rms (audit/rollouts.py:26-34,
53-57). Every network evaluation (T, D, N, every specialist) uses `fields.network.observe` with
`ctx_obs = context_channels()` (network.py:234-252, 616-619), as step 2's loop does; network.py:20-25
explains why. Idea 2 adds the critic with step 2's T64 upcast.

## 3. Targets

### 3.1 The costate at a state

The reward is r_t = (s_{t+1} − s_t)/10 − 0.02‖a_t − a_{t−1}‖² (racing.py:174-175;
window.py:756-760), and J_W = Σ_{t<W} γ^t r_t. The value from x_t with previous action a_{t−1} is
the tail V_t = Σ_{τ≥t} γ^{τ−t} r_τ, over W − t steps.

**The target is the post-jump costate**: the gradient of the value of the future from x_t, without
x_t's own path and box rows. The tail V_t and the critic's V(o_t) do not contain x_t's
constraint: the env decides termination on the post-step state (racing.py:159-167), and its edge
sits 3 cm outside the OCP corridor. For every 0 ≤ t < W:

  λ_t = γ^{−t} Φ_x(x_t, a*_t)ᵀ lam_g′[8t : 8t+8] − e_s/10
  π_t = ∂V_t/∂a_{t−1} = 0.04 (a*_t − a*_{t−1}),   a*_{−1} = 0

lam_g′ is lam_g after the pedal fold below, which changes only the pedal component.
- **A2 decision (t = 0):** λ_0 = lam_x[0:8], already this form, except that the fold adds
  α·c_0 to its pedal entry; π_0 = 0.04 a*_0.
- **Anchored decision (t = 1):** λ_1 = γ^{−1} Φ_x(x_1, a*_1)ᵀ lam_g′[8:16] − e_s/10;
  π_1 = 0.04 (a*_1 − a_0). x_1 is pinned by the fixed x_0 and a_0, so lam_g[0:8] is not unique
  whenever a constraint on x_1 is active (pedal-bound multipliers of about 10⁻⁴ at x_1 on 226 of
  396 sampled anchored keys). lam_g[8:16] is unique.
- **Cost.** One `jac_g` evaluation per key gives every Φ_x block (keys.py:464-469; 2.7 s per key
  with the rebuild).
- **Cheap equivalent, no jac_g:** λ_t = λ⁻_t + γ^{−t}(J_path(x_t)ᵀν_{t−1} + lam_x[8t:8t+8]), with
  the pre-jump form λ⁻_t = γ^{−t}(lam_g[8(t−1):8t] − γ^{t−1} e_s/10). For n it adds
  γ^{−t}(ν^L_{t−1} − ν^R_{t−1}); for s, a −w′(s)ν term where the width varies; the slip rows enter
  vx, vy, r and δ through the path Jacobian; in this data the box enters only pedal. K0 requires
  the two routes to agree to 10⁻¹⁰.

λ⁻_t is the costate just before the jump: the defect multiplier prices a shift injected before
x_t's own constraint. It is used only as λ(t⁻) inside the switching function of a_{t−1}
(Section 6). Where x_t's own constraint is active the target jumps within one step, and the
critic's gradient may follow either side, so every result is also reported by whether x_t's own
constraint is active (the contact stratum, Section 4).

**The pedal fold.** Wherever full throttle holds, the pedal-state bound is active and LICQ fails:
lam_g's pedal component is split arbitrarily between the defect, the pedal-state bound and the
command bound. The pedal-state box is redundant, because the env clips the action
(racing.py:150) and pedal_{t+1} = α pedal_t + (1 − α) a_t, with α = 0.905¹⁰, is a convex
combination. Idea 2 uses the multipliers of the TO-OCP without that box. They are unique and
follow from the key files in O(W):
- c_{W−1} = lam_x[8W+7]; c_{t−1} = lam_x[8t+7] + α c_t;
- lam_g′[8(t−1)+7] = lam_g[8(t−1)+7] + c_{t−1};
- the pedal-command multiplier at t gains (1 − α) c_t; the pedal-state multipliers become 0.

Stationarity holds by construction, because pedal's defect row is linear and involves only pedal
and a_pedal. λ_pedal is then the one-sided dV*/dpedal from below, which is what an env without a
state bound defines. On all 481 keys the folded command multipliers keep their sign, and the fold
cuts the R40/T90 twin disagreement from 0.234 to 0.018 SD (the other twins stay at or below 0.005
SD). The flag lam_x[8t+7] ≠ 0 is kept for reporting. If the fold is rejected, the fallback
excludes λ_pedal and π_pedal where |lam_x[8t+7]| > tol, where 1 − |pedal_t| < tol, or where the
command bound is active at t − 1 or t.

The corrections:
- **e_s/10.** r_{t−1} contains s_t, but V(o_t) counts only rewards from t on.
- **γ^{−t}.** J_W is discounted to t = 0.
- **Truncation.** The tail has W − t steps and no terminal value; the critic's horizon is infinite
  and bootstrapped (ppo.py:478-495). The bias of λ_t is γ^{W−t}(∂x_W/∂x_t)ᵀ∇V_∞(x_W), not the
  missing tail weight γ^{W−t}. On these strips the road after the corner is straight and x_W is
  near top speed, so the bias is small (Section 2: at most 0.54% SD on the A2 twins). A lead that
  persists to the window end is worth exactly 0, because the reward is a progress increment, so
  λ_s is the least exposed component. What the window does add is road beyond the critic's
  preview; K2b measures that part.

Values are in reward units, where 1 is 10 m of progress. The costate is per m for λ_s and λ_n;
per rad for λ_ξ and λ_δ; per rad/s for λ_r; per m/s for λ_vx and λ_vy; and per unit for λ_pedal
and π.

### 3.2 The critic's estimate, and the map from state to observation

The network sees o = observe(x, prev_action, accel, track, ctx) (racing.py:91-120), not x.

| State | Observation channels it moves |
|---|---|
| s | Every preview point (edges via pose and s + d_k; κ(s + d_k)), and lateral via the widths at s. All are invariant to sliding along a homogeneous road |
| n, ξ | The pose of all 160 edge channels; lateral (204); ξ (205) |
| vx | 200; the preview spacing d_max = clip(6vx, 150, 500) m for 25 < vx < 83 m/s (racing.py:85-88); sideslip; tyre use and slip (212-215) |
| vy, r, δ, pedal | Their own channels (201, 202, 206, 207), sideslip, and the tyre channels |
| With the state | accel (208-209), through the inverse one-step flow (racing.py:155-156) |
| Held | prev_action (210-211), ctx (216-221) |

**The pullback.** accel is a function of the state: ax = (vx_t − vx_{t−1})/dt − vy_t r_t and
ay = (vy_t − vy_{t−1})/dt + vx_t r_t, with x_{t−1} = Φ⁻¹(x_t; a_{t−1}) on the data manifold. A
critic equal to V* on the manifold has its derivatives along the manifold pinned, and its
derivative in the two accel directions normal to it unconstrained by any data.
- **Rows with t ≥ 1** (T at t ≥ 5, and N): the primary pullback is the on-manifold total
  derivative at hidden_mu = 1,
    g_t = σ_R [J_z + J_acc M_x]ᵀ ∂V_c/∂z,   M_x = ∂A/∂x_t + ∂A/∂x_{t−1} Φ_x(x_{t−1}, a_{t−1})⁻¹,
    π̂_t = σ_R [J_prev + J_acc M_a]ᵀ ∂V_c/∂z,   M_a = −∂A/∂x_{t−1} Φ_x⁻¹ Φ_a.
  J_z = ∂z/∂x at fixed accel and prev_action; J_acc and J_prev are the accel and prev-action
  columns. x_{t−1} and a_{t−1} come from the key (from the agent's run for N); Φ_x and Φ_a come
  from JAX `jacfwd` of `sim.control_step`, with no Newton step. Only the body block of M_x is
  non-zero, so g_s, g_n and g_ξ are unchanged; in K_P only vx moves, plus the secondary components
  and π.
- **Rows with t = 0** (D, and T's t = 0 rows) have accel = prev_action = 0, like training resets
  (racing.py:136-139), and keep the accel-held derivative.
- The clip zeroes rows where |z| = 10.
- **What is claimed.** If V_c∘z = V* on the manifold of consistent observations (reset rows:
  accel ≡ 0), the on-manifold pullback equals λ.
- **Sensitivities.** The accel-held pullback is pre-registered as a sensitivity analysis. The
  x_{t−1}-held version is descriptive at most, because its accel term is about 36 times the vx
  channel in z units. The share of g_vx that comes from the accel term is reported per state.

**The accel channels.** In this env accel is a deterministic function of the observed body
state, prev_action, ctx and the hidden μ (racing.py:152-156; the body dynamics do not depend on
the track, dynamics.py:116-153). Apart from the reset flag, the only information it adds is μ
(adapters.py:143). ∂V_c/∂accel is reported as **accel-channel sensitivity**: largely
unidentified (redundant with (x_t, a_{t−1}); only a ±2% grip direction is constrained by data),
compatible with μ inference or with redundant re-encoding of the state, and not evidence of
non-Markov use. Descriptively, |∂V_c/∂accel_y| is shown against lateral tyre load and
utilisation, beside ∂V_c/∂util_{f,r} (212-213), since lateral force scales with μ at every slip
angle (dynamics.py:103).

**Probe for λ, or for its pullback?** For λ, in state coordinates.
- The pullback is unique: J_zᵀ maps the critic's 222-d gradient to 8 numbers.
- The reverse is not. λ has no unique image in observation space: J_zᵀμ = λ has a 214-dimensional
  family of solutions, and the minimum-norm one depends on the metric (z or o).
- So probes predict λ_k, and the critic's gradient is compared after pullback. Only the road
  targets (Section 3.4) are native to input space.

| Component | Optimum side | Network side | Role |
|---|---|---|---|
| s | Set by the s-dependent constraints ahead: 0 once none lies ahead (t ≈ 85 in the example). Least sensitive to truncation | Observable on T before the exit, because the corner is always in the preview (margin ≥ 15 m at t = 0). Structurally unobservable on the exit straight, where it is excluded | Primary |
| n, ξ | Jump at x_t's own contacts; the target is the post-jump value. λ_n integrates the corridor multipliers backward through the adjoint, plus a curvature progress term, so it carries the wall price | Edge pose; own channels | Primary |
| vx | Defined; the most exposed to road beyond the preview (K2b) | Own channel; preview spacing; tyres | Primary |
| vy, r, δ, pedal | Fast; decay within about 1 s. δ and pedal are degenerate at bound-active states, where a box multiplier that no observation channel moves smoothly sets them | Slaved to (vx, κ, a) on visited states, so regression barely constrains the gradient | Secondary |
| π | Analytic | Channels 210-211, with M_a | Secondary |

So **K_P = {s, n, ξ, vx}**. λ_s on T: 86% of its winsorised SST lies in corner states, and its twin
truncation error is at most 0.54% SD (p90 0.14%, below λ_vx's 0.27%). It is even under mirroring,
so it is not multiplied by σ. Its right tail is heavy (p99.5 ≈ 1.0 against a median ≈ 0), so
winsorisation matters. On D it is descriptive only, since it hardly varies there.

**Exclusions and flags.** All are computable for every T, D and N state from key files and z:
- **multi-basin key:** excluded;
- **unobservable:** per component, the squared residual of J_z's column k after least-squares
  projection onto the other columns (QR or lstsq; no explicit inverse), below a per-component
  threshold set in the dry run. First, s is dropped and marked structurally unobservable when
  ‖J_z e_s‖ / median_k ‖J_z e_k‖ < τ_s = 10⁻⁴. τ_s sits above the float32 rounding floor (about
  5·10⁻⁶ relative on upcast arrays) and below the transitional states (2·10⁻² to 4·10⁻⁴ within 10 m
  past s_c1). The other components are then computed without the s column. Every A2 strip is
  exactly straight and of constant width beyond s_c1, so this fires on the exit straight, where
  J_zᵀJ_z would otherwise be singular;
- **contact** (x_t's own corridor, slip or box row active): a reporting stratum, not an exclusion;
- **pedal-saturated** (lam_x[8t+7] ≠ 0): a stratum for pedal;
- **kappa clipping** and **beyond-preview exposure:** Section 4 and K2b.

K1's kink rule decides only K1's own pass; it is not an exclusion rule.

The components n, ξ, vy, r and δ, and steer, are odd under mirroring; s, vx and pedal are even.
Odd components are multiplied by σ (toward the turn) for reporting only. Probes use the raw
targets, with mirror partners in one fold.

### 3.3 Why even an exact critic would not equal λ

Four gaps separate even the training MDP's V*(o) from the window problem's V*:
- the hidden grip jitter μ ~ U(0.98, 1.02), which the policy does not observe (adapters.py:123,
  143);
- the preview limit d_max (racing.py:85-88), while the window problem knows the road beyond;
- the termination penalty 5 + 5(v/50)² (racing.py:177);
- termination at the edge margin of 1.0 m and at α_max (racing.py:54, 160-163), against the OCP's
  further 3 cm and 10⁻⁴ rad (window.py:90-91).

The critic estimates V^π of the stochastic training policy (σ 0.04–0.09), including crash risk,
and conditioned on what accel and the tyre-utilisation channels (212-213, computed with the hidden
μ, racing.py:116) reveal about μ.

Consequences:
- **Near the limit.** About 18% of T sits at the optimum's wall or slip limit, where a correct
  critic of a stochastic policy that almost never goes there must diverge from λ*. C1 and C2 are
  decided on an interior stratum (Section 4). The near-limit stratum gets a signed secondary
  prediction (C1-limit).
- **Calibration.** Comparisons are scale-free in shape but fixed in sign: a robust per-fold
  affine calibration with the slope clamped at ≥ 0 (Section 6). The σ_R-converted slope carries a
  label band.
- **The policy's own costate** splits the gap into two links.

**The policy's own costate** λ^own: the gradient of the deterministic policy's discounted
H-step return with respect to x and a_{−1}, at μ = 1, with accel held fixed.
- **Computation.** The frozen `network.closed_loop` (network.py:631) with steps = H and a `LoopIn`
  with s_stop = +∞. J = Σ_{k<H} γ^k r_k is rebuilt from `Trace.x`, clip(`Trace.a`, ±1) and the
  previous action, as at racing.py:174-175. A run counts only if `Trace.valid` holds on all H
  steps (no termination or truncation). For every state of a run at once, idea 2's own loop in
  `costate/` adds a zero perturbation δ_k to each state; the VJP with respect to δ_k gives
  γ^k ∇_{x_k}V_k in one reverse sweep. That loop must reproduce `closed_loop`'s Trace at δ = 0
  (K3b).
- **Horizon.** H = W − t of the key, for the comparison with λ*. H = 800 for every D key, for the
  comparison with the critic (γ⁸⁰⁰ ≈ 0.018), checking that the run stays below S_TRUNC (plans
  reach at most 4,209 m of 4,598 m).
- **States.** (i) T's interior states, restarting from x_t with prev_action a*_{t−1}, every 20th
  step (about 1,000 per network), never near-limit states, where a restart may crash. (ii) Every
  state of each network's own run from each D key, stratified by T's phases. (iii) N, for step 1's
  12 finals: λ* was solved at the network's own x_1, so the critic's gradient, λ^own and λ* exist
  at the same states.
- **What it is.** The critic's gradient against its deterministic policy's H-step value gradient,
  not "critic error". It still mixes in the stochastic policy, grip jitter, crash risk and the
  terminal penalty, the bootstrapped horizon against H, and survivor selection.
- **Value accuracy.** σ_R·V_c against the deterministic policy's discounted return from the same
  states (T interior, D, N), with loops of up to 1,000 steps (γ¹⁰⁰⁰ ≈ 0.007) or to the end of the
  strip, and the γ^H tail reported. Section 9's "values fitted, gradients not" reading is allowed
  only where this check passes (threshold fixed at the tag). The training explained variance
  (0.917–0.945 in metrics.csv) comes from training rollouts and does not transfer to these states.

### 3.4 Shadow prices of the walls and the road (secondary)

**Wall and limit prices.** ν^in_t and ν^out_t ≥ 0 are the corridor multipliers, recoded to inside
and outside. Widths enter the TO-OCP only through the corridor rows, so by the envelope theorem:
- Π^in_t = γ^{−t} Σ_{τ≥t} ν^in_τ is dV_t/dΔ for a uniform inside widening ahead. The sum starts at
  the corridor row of x_{t+1}, the post-jump convention of Section 3.1. Π^out_t is the same for
  the outside.
- Π^slip_t = γ^{−t} Σ_{τ≥t} (|ν^f_τ| + |ν^r_τ|) is dV_t/d ln α_max: the price of the
  loss-of-control slip limit, with the 10⁻⁴ rad margin held fixed. It is not a price of grip.
  α_max = 10° lies past the Pacejka force peak (about 7.1°; the force at α_max is 97% of peak),
  α_max does not enter Φ (dynamics.py:61-112), and the network never observes it (it sees α·5,
  racing.py:117).
- Π^μ_t = γ^{−t} Σ_{τ≥t} lam_g[defect τ]ᵀ ∂Φ_τ/∂μ is the grip price: the mu_scale column of step
  2's F_p (kkt.py:673), restricted to the defect rows τ ≥ t. The path rows add nothing, since slip
  angles do not depend on μ. It is the only route by which grip is priced, because the friction
  ellipse is enforced inside Φ (dynamics.py:103-108) and has no multiplier. The keys use
  hidden_mu = 1 (keys.py:27), while the network observes only the nominal ctx[4].
- d^in_t, the ν-weighted mean of s_{τ+1} − s_t, locates the binding contact.
- On stretches whose road ahead in the window is straight and of constant width,
  λ_n = Π^R_t − Π^L_t (a K0 gate).

**Road fields** (sums over τ ≥ t only).
- Width: ∂V*_t/∂w_in(s_j) = γ^{−t} Σ_{τ≥t} ν^in_τ ∂w_in(s_{τ+1})/∂w_j, through step 2's map from
  samples to coefficients (kkt.py:191, 652).
- Curvature: through the g rows of step 2's F_p (kkt.py:673), rows τ ≥ t.
- Critic: ∂V_c/∂(samples) through `observe`, via the strip_arrays tangent (perturb.py:282-317).
  It is native to input space and compared on the preview's support. The optimum's mass beyond
  d_max is reported as "unseen".

**The policy's own width price.** ∂V^{π_det}_W/∂w_in(s_j), and Π^{in,own} for a uniform widening,
as a VJP of the closed-loop return through `closed_loop` (network.py:631) along the
`perturbed_arrays`/`strip_tangent` path (perturb.py:282-317). It is computed on D and N only, for
runs alive through the window with B > 0. W1 then reports three comparisons: the critic against
its own policy, its own policy against the optimum, and the critic against the optimum. π_det
leaves out the noise and grip-jitter crash risk, so a critic price above π_det's is not by itself
an error (the stochastic V^π width price may be estimated from rollouts, descriptively).
Magnitudes are reported as the σ_R-scaled slope of the critic's field against Π^{in,own} and
against Π^in, because R² and Spearman are scale-free.

**Locality.** ν, Π^in and both G's are derivatives at a fixed active set, valid for millimetre to
centimetre changes. Any link to step 1's crashes, which involve width changes of metres, goes
through step 2's finite-amplitude dose-response (receptive-fields-design.md:174-183).

## 4. Probes

**State sets.**
- **T (primary).** The A2 plan states every 5th step from t = 0 to t_end, the first t with
  s_t ≥ s_c1 + 50 m, included (median 102, range 70–199), within K2's cut. That is at most 9,967
  states before the cut (9,170 at t_H = 100); K6 gives the final count.
  - Observations use prev_action a*_{t−1} and the accel of x_{t−1} → x_t (racing.py:155-156);
    both are 0 at t = 0, as with `Cell.initial_aux` (network.py:52-55).
  - **Interior stratum** (decisions for C1, C2): from t to t + k, with k = 20 steps (1 s), no
    corridor or slip row is active and the clearance to the termination boundary (not the OCP
    bound) and the slip margin stay above thresholds fixed at the tag. Expected share 25–50% of T
    for k = 10–20; the dry run reports it by phase.
  - **Near-limit stratum:** the rest of T, reported, with C1-limit.
  - **Contact stratum** (x_t's own constraint active): reported separately throughout.
- **D (secondary).** The 481 decision states, i.e. T at t = 0.
- **N (secondary; step 1's 12 finals only).** Each agent's anchored-final x_1, with its own
  a_{L−1} and accel: 8,136 states. Its 9 generalists are exactly the networks on which C1 and C2
  are decided, so per-agent replication is possible wherever a decision is made. C3 does not use
  N.

**Kappa clipping** (input side, before the tag). For each network, z is computed at T, D and N
under its own obs_rms, and each state is flagged with step 2's rule (network.py:110-117,
993-1008); "kappa clipping" is a K6 reason. One sensitivity set, shared by every network, drops
the geometries with R ≤ 8.94 m (the largest R_clip_max over the 31 generalists): R ∈ {6.0, 7.5,
8.4}, 6 geometries and 80 keys. C1, C2 and C3 are reported on it beside the primary results. Step
1's support regime (support.py:599-611) is a descriptive stratum; all A2 radii from 6.0 to 9.4 m
lie below every step 1 generalist's q1.

**Features and extraction.** z; the actor's h1, h2, h3 and m; the critic's h1, h2 and h3; and the
heads' hidden units (descriptive). One float64 extractor over the parameter tree serves the actor,
the critic trunk and the heads: h_k = tanh(h_{k−1}W + b) from `MLPEncoder_0/Dense_k`, and head
i's hidden units u_i = relu(h3·Dense_{2i+2} + b), i = 0…4, paired by construction order
(ppo.py:213), not by shape or name. K3 checks it against `Actor.apply` and `Critic.apply`. Head
gradients are taken only through this extractor; through `Critic.apply` they are exactly 0
(stop_gradient, ppo.py:211). C1 uses only the main V.

**Preprocessing and penalties** (fixed at the tag).
- Every feature is z-scored on the outer training folds, and units with a training-fold SD below
  10⁻⁶ are dropped (saturated or dead tanh units would otherwise be inflated). z is treated the
  same way in every z-only probe, so R²_z does not depend on each run's obs_rms.
- Two penalties: one for [z, m] (and decoys, where present) and one for h3, from a 2-D grid
  10⁻⁴…10⁴ in half-decade steps, chosen per component by inner grouped CV on stream
  `costate_inner`. The full model nests the baseline (h3's penalty → ∞).
- A two-stage check (Frisch–Waugh–Lovell) is reported: within the training folds, residualise
  λ_k and every h3 unit on [z, m], then ridge the residualised target on the residualised h3.
- Fits use the metric's weights.

**Probe families.**
- **Ridge (primary).**
- **One-hidden-layer MLP of 64 tanh units (secondary).** A decodability ceiling only, never used in
  a decision.
- **Conditional probing (Hewitt et al. 2021).** Fit [h_l, baseline] against the baseline alone.
  Any function of z holds no more information than z, so "encoding" means accessibility to a fixed
  probe family, beyond a stated baseline.

**Decoys.**
- **D₁** = (s − s_apex, vx, log R, θ, w, σ), with s_apex the geometric apex. f̂_D is kNN (k = 15)
  on D₁ standardised on the outer training fold, fitted on that fold only. λ_res = λ − f̂_D,
  cross-fitted, is C2's primary target.
- **D₂** = D₁ plus a*_t, 1/vx, the distance and time to the optimum's brake onset
  (measures.py:1444), preview κ at the apex, the QSS speed margin and the clearance to each edge.
  It serves the descriptive Δ2_D and C1's context comparators.
- **Before the tag**, from key files: each K_P component's residual variance share on T and on
  the interior stratum (reviewers' estimates beyond D₁: n 0.20, ξ 0.09, vx 0.11; λ_s not yet
  computed), and how stable f̂_D is from fold to fold. A component whose residual share is below
  0.10 is descriptive in C2. Split-half reliability is not used: T is deterministic and mirror
  partners are exact copies up to sign.

**Splits.**
- **Outer: 6 folds over the 48 geometries, 8 per fold.** Within each stratum R tercile × θ class
  {90°, 180°, other}, geometries are permuted on stream `costate_folds`, sorted by (θ, R) and dealt
  to the folds in serpentine order. Each geometry sits whole in one fold: both mirror directions,
  every phase, every tier C width, both twins and every plan state. No held-out geometry enters a
  fit or a penalty choice. θ = 60° (1 geometry), 45° (3) and 135° (4) appear in at most 1, 3 and 4
  folds. R and θ are confounded (R ≥ 69.8 m only at 60° or 90°; 45° and 135° only at 9.4–44.7 m),
  so θ strata are descriptive. Partition 0 is frozen at the tag; 10 further partitions are
  reported as robustness.
- **Inner:** 5 grouped folds over the training geometries.
- **Descriptive:** family transfer (train on W 12, test on W 9 and W 15 of held-out geometries),
  and radius interpolation (hold out 30 ≤ R ≤ 100 m).

**Weights.** Each key weighs equally, and states weigh equally within a key, in the fits, the
inner CV and the metrics; the bootstrap resamples geometries. Sensitivities: (i) geometries
weighted equally (mirror pairs equally within a geometry, states equally within a pair); (ii)
states weighted equally; (iii) the 11 geometries with 3 keys or fewer dropped.

**Metrics.**
- **Per component:** out-of-fold R² = 1 − SSE/SST, pooled over folds.
- **Direction-aware R²_wd.** For the mirror-odd components (n, ξ, vy, r, δ), SST is taken about the
  weighted mean within each turn direction. This equals folding target and prediction by σ,
  without relying on exact mirror symmetry. For s, vx and pedal, R²_wd is the raw R². Every
  threshold and decision uses R²_wd, so the K_P mean has one weighting. Raw R² is reported beside
  it, with the direction-only floor (Section 5: ξ ≈ 0.36 and vy 0.46 under geometry weights,
  recomputed under the final weights before the tag).
- **Winsorising.** Targets (post-jump, folded) are winsorised at the 0.5th and 99.5th percentiles
  of T, with bounds fixed at the tag. Spearman ρ and the unwinsorised R² are reported alongside.
- **Summary:** the mean over K_P (for C2, over the components that pass the power floor).
- **CIs:** a cluster bootstrap over geometries (2,000 resamples) of the out-of-fold predictions,
  plus the spread over folds and over partitions.
- **Strata:** approach, braking, corner and exit (from the optimum's brake onset and apex,
  measures.py:1444); left and right; W 400 and 800; interior, near-limit and contact;
  pedal-saturated; kappa-clipped.
- **Sub-scores** for every decision statistic: braking keys (λ_vx has 2 sign changes; 459 keys, 38
  geometries) and non-braking keys (22 keys, 10 geometries, including R = 69.8–300 m at 90°). The
  pooled score stays primary: it keeps the real λ_vx level differences between geometries and all
  48 clusters.

## 5. Controls and nulls

Random networks: actors from `fields.network.random_actor_params(cfg, seed)` (network.py:306-313)
and critics from a new `random_critic_params(cfg, seed)` in `costate/`:
`Critic(cfg).init(PRNGKey(seed), zeros(OBS_DIM, float32))` under `jax.enable_x64(False)`, upcast
with T64, V only. cfg is always the network's own PPOConfig, never `PPOConfig()` (SimBa,
critic_width 384). Seeds: actors 0–4, critics 0–19. Each trained network's random controls use its
own obs_rms (and, for C2, its own m). The gates K3–K5 use G1's obs_rms, as step 2 did.

| Control | What it answers |
|---|---|
| Input baseline: ridge on z | R²_z, a descriptive comparator for C1 (Δ1); C2's conditioning set |
| Input ceiling: MLP on z | Whether λ is decodable at all from what the network sees; R²_c/R²_MLP is reported as normalised fidelity |
| Direction-only | The per-direction training-fold mean: a floor for raw R² |
| Decoy models | f̂_D on D₁ (C2's residualisation; its R² is reported beside R²_c as context, not as a bar); D₂; and [pedal, a*_{t−1}, vx], which gives R² 0.70 for λ_vx |
| Phase template | Per held-out geometry, λ interpolated from the training geometries only (same direction; n, ξ, vy, r and δ rescaled by σ; aligned on phase or time to apex; weighted by nearest (R, θ)). R²_template is a descriptive baseline |
| Random-init actors (5) and critics (20) | C2's and C2-sw's offset; C1's sanity floor |
| Value-gradient comparators | C1's nulls (Section 6) |
| Control task (Hewitt & Liang 2019) | Descriptive only; matching in K5. It measures probe expressivity and is confounded with the random-feature function class |
| Permuted targets | A wiring check (K5), not a null for any hypothesis |

## 6. Hypotheses with decision bands

Decisions use 95% cluster-bootstrap CIs over geometries, with bands fixed at the tag. C1 and C2
are decided on T's interior stratum, C2-sw on T's pedal-saturated steps, and C3 as stated below. C1, C2 and C2-sw are decided per network on the 9 step 1 generalist finals; a variant
claim needs 2 of 3 seeds (R18). The other 22 generalists are descriptive for those and enter C3.
The specialists are descriptive throughout.

### Primary family

**C1: the critic's gradient is the costate, in sign and shape.**
- **Calibration.** Per component, a per-fold Huber affine fit of λ_k on g_k (tuning constant 1.345
  on the MAD scale), with the slope clamped at ≥ 0: if the fitted slope is negative, the
  out-of-fold prediction is the intercept. An anti-aligned or uninformative gradient then scores
  R²_c ≤ 0.
- **Statistic:** R²_c, the R²_wd of the calibrated g_t, averaged over K_P.
- **Nulls,** each put through the same pullback and calibration, compared by paired cluster
  bootstrap:
  - **value-fit gradients:** ridge and the 64-unit MLP fitted on z to the optimum's tail values
    V*_t, on the training folds, with their z-gradients pulled back. They ask whether fitting
    values with a smooth function of z already yields costate-like gradients through J_z;
  - **a QSS value** (the discounted progress of the preview's quasi-steady speed profile, the law
    step 1's Qp reads) and **a clearance-barrier value** (a log-barrier in the current and
    previewed lateral clearances), both coded before the tag. They reproduce the vx and n sign
    structure;
  - **phase-matched far-geometry λ:** λ from a far geometry at the same phase, as in step 2's Δρ
    (receptive-fields-design.md:438-440);
  - **random-init critics:** the maximum R²_c over 20 seeds (a sanity floor).
- **Confirm** if R²_c's CI lower end is > 0.6, the paired CI lower end of R²_c minus each null
  is > 0, and the signed Spearman ρ(g_k, λ_k) has CI lower end > 0 on every component of K_P.
- **Refute** if R²_c's CI upper end is < 0.2.
- **Inconclusive** otherwise.
- **Label "costate gradient at scale":** confirm, plus a pooled σ_R-converted slope CI within
  [0.5, 2] for each K_P component. Truncation moves the targets by at most a few percent of SD, so
  it cannot widen the band. Slope CIs are reported per component and stratum.
- **Direction from theory.**
  - If V_c∘z equals the window problem's V* on the manifold of consistent observations, g_t = λ_t;
    Section 3.3 lists why even the MDP's V* differs.
  - The critic is fitted on about 2·10⁸ transitions of the same objective, with the same γ.
  - λ_vx's sign follows the pedal state and a*_{t−1} (92–96% agreement in the keys), so on T it is
    largely readable from z (ridge on z: R² ≈ 0.79 averaged over {n, ξ, vx}). The input baseline
    is therefore descriptive, not the comparator.
  - Against this: regression constrains values, not gradients; the critic estimates V^π; and PPO
    critics fit the true value poorly (Ilyas et al. 2020).

**C2: the actor's hidden layers carry costate information beyond the input, the output and
position-and-speed.**
- **Statistic:** Δ2 = [R²(h3 ⊕ z ⊕ m) − R²(z ⊕ m)] on λ_res, trained minus the median over 5
  random-init actors (each on the trained network's obs_rms and given its m). It is averaged over
  the K_P components that pass the power floor. h3 is pre-specified; h1 and h2 are descriptive.
- **Why condition on m:** m is linear in h3. Without it, costate components that merely co-vary
  with the action would confirm C2.
- **Why residualise on D₁:** on T, about 85% of each K_P target is a smooth function of
  position, speed and geometry. Any layer that holds time-to-corner and speed more linearly than z
  would otherwise confirm without holding anything specific to the costate.
- **Confirm** if the CI lower end is > 0 and the point estimate is ≥ 0.05. **Refute** if the CI
  upper end is < 0.05. **Inconclusive** otherwise. The 0.05 band is checked against the dry run's
  headroom (1 − R²_b) before the tag.
- **Direction from theory.**
  - The TO-OCP is discrete-time, so only stationarity holds; no maximum principle over a is
    assumed. With H_t = r_t + γ(λ⁻_{t+1}ᵀΦ + π_{t+1}ᵀa_t) and
    ∂r/∂a_t = Φ_aᵀe_s/10 − 0.04(a_t − a_{t−1}), ∂H/∂a_t is the switching function
      σ_t = γ^{−t} lam_x(a_t) = Φ_aᵀ(e_s/10 + γλ⁻_{t+1}) + γπ_{t+1} − 0.04(a_t − a_{t−1}),
    zero off the bounds and the bound multiplier at saturated steps.
  - A saturated control leaves its bound when σ_t reaches 0. On interior stretches σ_t ≡ 0, and
    Φ_aᵀ(e_s/10 + γλ_{t+1}) balances the rate terms. So a C2 confirm means "costate information
    beyond z, m and D₁", not "the switching function"; C2-sw tests that.
  - PPO's signal carries ∇V^π and its prev-action component π (C3, path (a)).
  - The prior is weak. A refute (a reflexive map from observation to action) is also informative.

**C2-sw: the actor carries the switching function on saturated stretches.**
- **States:** the pedal-saturated steps of T (about 25% of T, all full throttle), with approach and
  exit reported separately. N replicates.
- **Target:** σ^ped_t = γ^{−t} × the folded pedal-command multiplier, which counts down to 0 at
  brake onset.
- **Statistic:** Δsw = [R²(h3 ⊕ z) − R²(z)], trained minus the median over 5 random-init actors.
  It is not conditioned on m, because m is the candidate readout. Co-reported: Spearman
  ρ(m_pedal − 1, σ^ped_t), the pre-clip margin.
- **Bands** as C2.

**C3: costate fidelity goes with driving quality.** C3-w and C3-int are primary; C3-R²c is
secondary. All three need step 2's decisions.

*C3-w, between networks, on the width axis.* Steps 1 and 2 place R-dev failures at width changes,
so x lies on that axis.
- **Networks:** the 31 generalists in 7 recipe strata: the v1, v2 and v3 baselines (5 each), and
  real-width v1, real-width v3, LOO f1 and LOO f2 (4 each).
- **x:** the critic's inside-wall price agreement, the calibrated R² (as in C1) of its uniform
  inside-widening derivative (its width field summed over the preview) against Π^in_t on T.
- **y:** R-dev completion, the mean over 12 circuits of `eval_rdev.json`'s completion. The gap on
  completed circuits would mix in which circuits were completed, and any new behavioural measure
  would need new R-dev rollouts after the tag.
- **Test:** the size-weighted mean of within-stratum Spearman ρ, with a one-sided stratified
  permutation (y permuted within strata), 100,000 draws on stream `costate_c3`. With the known y
  and its ties: the null SD is about 0.20 and the critical weighted ρ 0.337, so 80% power needs a
  latent ρ of about 0.6.
- **Confirm** if p ≤ 0.05 for a positive association. **"Rules out a strong association"** if the
  95% upper bound on the weighted ρ is < 0.3 (this fires about 40% of the time under an exact
  null). **Inconclusive** otherwise. There is no reversed-sign refute.
- **Covariates,** reported with and without, as stratified partial associations: competence (final
  training episode_return from metrics.csv) and overlap (the network's distance from its own line
  to the optimum's, step 2's B on the A2 cells). Kappa clipping is nearly constant within each
  recipe stratum (R_clip ranges of 0.1 m or less), so the strata absorb it; only v2 at R = 7.5 m is
  marginal.

*C3-int, network × geometry.*
- For each of the roughly 29 networks with step 2 fields, and each geometry (grouped as in the
  folds): the signed calibrated critic residual per K_P component, averaged over the geometry's
  interior T states; and the behavioural deviation: step 2's G − 1 and B, plus step 1's entry-speed
  ratio and brake-onset residual where they exist.
- Both are double-centred over networks and over geometries, which removes corner difficulty and
  network-level competence.
- **Pre-signed pairs,** fixed at the tag: λ_n underpricing the inside wall goes with G < 1 or a
  smaller B; a too-negative λ_vx goes with earlier braking or a lower entry speed.
- **Test:** the mean signed correlation over the pairs, one-sided, against a permutation of
  geometry labels within each network, blocked on R tercile × θ class (a crossed random-effects
  model as a check). Confirm if p ≤ 0.05.

*C3-R²c (secondary).* v1's x = C1's R²_c against completion, with C3-w's test, reported as an
effect size. The step 1/2 width account predicts a null here.

**Theory for C3: positive, associational.** Two paths predict it, and the tests cannot separate
them (Section 9):
- (a) **critic → actor.** PPO never differentiates the critic with respect to its input. Under
  GAE(0.95), a critic error ε enters the expected actor signal as
  Σ_k γ(1−λ)(γλ)^k (∂x_{t+k+1}/∂a_t)ᵀ ∇ε(x_{t+k+1}), plus the prev-action channel, cut at the
  64-step rollout end (ppo.py:478-495). The weight at x_{t+1} is about 0.05, the weights sum to
  about 0.91, and the mean delay is about 18 steps (0.9 s). In the one-step form B projects mainly
  onto δ and pedal (0.05 s lags, dynamics.py:131-132). So a K_P gradient error biases the actor
  only along action-reachable directions over about 1 s, and fidelity matters along the closed
  loop, not only at the next state.
- (b) **policy → value:** a better policy has V^π nearer V*.
- **Prospective in x only.** Completion is already known; x is not.

**Multiplicity.** The primary family is C1, C2 and C3-w (Section 12). C1 and C2 are decided per
network, then per variant, and C3-w once. They answer different questions and are reported
separately, with no correction across them (as in step 2). C2-sw and C3-int are secondary
(Section 12).

### Secondary (named, pre-registered) and descriptive

- **C1-pattern.** R²_c over K_P exceeds R²_c over {vy, r, δ, pedal} (paired bootstrap, CI lower end
  > 0), with δ and pedal restricted to states where their bounds are inactive; the saturated
  stratum is reported separately. In theory, regression constrains the gradient only along
  directions in which the visited states vary.
- **C1-heads.** For each generalist, each head's gradient is pulled back like g_t. For
  q ∈ {V, g_s, g_n, g_ξ, g_vx, g_vy, g_r, g_δ, g_pedal, π}, the between-head disagreement
  D_q = mean_t var_k(q^k_t) / var_t(mean_k q^k_t) (equivalently 1 − ICC(1)). Two pre-registered
  comparisons by paired cluster bootstrap: D_grad(K_P) − D_V > 0 (values pinned more tightly than
  gradients), and D_grad({vy, r, δ, pedal}) − D_grad(K_P) > 0 (the target-free counterpart of
  C1-pattern). Per-head R²_c and the inter-head gradient reliability ρ_k are reported, with √ρ_k
  as a ceiling on the heads' |corr(g^k, λ_k)| and a descriptive reference for the main critic.
  D is "head-level indeterminacy given the trunk": the shared stop-gradient trunk hides
  indeterminacy, and the ReLU heads with their own bootstrapped targets (ppo.py:478-495) may
  overstate it.
- **C1-own.** C1's bands and sign rules with λ^own in place of λ*, on T's interior, own-run and N
  states; per network, valid only if that network passes K3b's trained analogue (otherwise
  descriptive). It is the first link of the decomposition: critic against own policy, then own
  policy against optimum.
- **C1-limit.** At strictly active contacts, the calibrated (interior-fitted) g_n and g_ξ exceed
  λ* in magnitude and point away from the binding wall or slip direction: the CI lower end of the
  mean signed excess > 0.
- **C1-braking.** C1's rule on the braking keys, since the theory behind C1 is about sign flips.
- **C1-dev.** Across the 16 snapshots, the Spearman ρ of R²_c with log timesteps is > 0 in at least
  2 of 3 seeds per variant. The 10M snapshot is a partly trained critic that anchors this trend,
  not a null.
- **ΔC1** = R²(z ⊕ g_t) − R²(z): does the gradient add to the input?
- **C2-pairs.** Pairs of states with matched decoys (same geometry, |Δ(s − s_apex)| < 2.5 m,
  |Δvx| < 1 m/s) but different λ: phase keys, tier C widths at equal s − s_apex, the W 400/800
  twins, and N's agent states against plan states in the same cell. Spearman of the probe's
  predicted Δλ against the true Δλ over pairs, with a geometry bootstrap.
- **W1.** Probe R² and Spearman for Π^in, Π^out, Π^slip and Π^μ; the critic's width field against
  the optimum's and the policy's own (three comparisons and calibrated magnitudes, Section 3.4).
- **Descriptive only:**
  - Δ1 = R²_c − R²_z; R²_c/R²_MLP-on-z; R²_z with and without channels 207 and 210-211, and on D;
    R²_{z⊕D₂}; the decoy-only R²; R²_template;
  - Δ2_D = [R²(h3 ⊕ z ⊕ m ⊕ D₂) − R²(z ⊕ m ⊕ D₂)]; Δ2 without residualisation; Δ2 conditioned on
    z only; conditioning on a*_{t+k} (k = 10, 20, 40), which is co-determined with λ and can null
    the test by construction; λ probed with channels 207 and 210-211 residualised;
  - the switching function γ^{−t}lam_x(a_t) on saturated steps against m; λ(t⁻) − λ(t⁺) per
    state; the action-projected critic signal
    q_c(t) = Σ_k w_k (Φ_x(t+k)…Φ_x(t+1) B_t)ᵀ g_{t+k+1} + ∂r/∂a (GAE weights, 64-step cut, the
    key's own Jacobians) against ∂Q*/∂a_t = ∂r/∂a + γB_tᵀλ⁻_{t+1}, which the GAE mixture of V*
    equals at the optimum for every λ; and q_c's association with completion;
  - the D and N sets; h1, h2, the critic's layers and the heads' units; MLP probes and the
    descriptive splits; accel-channel sensitivity and the share of g_vx from the accel term; the
    calibration slopes; the unseen-road share (K2b); context prices (Section 10); the specialists;
    the left/right asymmetry index; pooled Spearman for C3; C2's statistic against completion;
    after step 2, the critic-against-own width-price agreement against G (the Π^in agreement
    against G is expected even for perfect critics).

## 7. Gates and validation (before any trained network)

**K0, targets from files** (every key, no solves).
- **x_0 identity, a convention and implementation check, exact by construction.**
  lam_x[0:8] = Φ_x(x_0, a*_0)ᵀlam_g[0:8] − e_s/10 to 10⁻¹² on all 481 A2 keys (all rebuild here
  with defects below 10⁻¹⁰). It checks the evaluation point, sign, e_s/10 and block index, and that
  the file carries filled duals. It is not evidence that λ_0 is a sensitivity. The code asserts
  that the solver options do not set `fixed_variable_treatment = make_parameter_nodual`.
- **Anchored a_0 companion:** lam_x at a_0 = Φ_a(x_0, a_0)ᵀlam_g[0:8] − 0.04 a_0 +
  0.04γ(a*_1 − a_0), with a_{−1} = 0. This pins the sign of the γπ_1 term before any re-solve.
- **Post-jump λ:** the jac_g route and the cheap route agree to 10⁻¹⁰ on all T rows; λ⁻ − λ⁺ is
  reported per component.
- **Pedal fold:** the stationarity residual with the pedal-state box removed is at most 10⁻¹⁰
  (Gate 0's precision), and no folded command multiplier violates its sign.
- **Action rows:** σ_t = γ^{−t}lam_x(a_t) against Φ_aᵀ(e_s/10 + γλ⁻_{t+1}) + γπ_{t+1} −
  0.04(a_t − a_{t−1}), with π_W = 0 (1.2·10⁻¹¹ on key 03b6…; Gate 0's action rows rearranged).
  Planted bugs must fail: no e_s/10; no π; no γ^{−t}.
- **λ_n = Π^R − Π^L** on stretches whose road ahead is straight and of constant width.
- Every λ_t and π_t is finite. The difference between re-polished and stored multipliers is
  reported.

**K1, targets against re-solves.**
- **Keys:** 50, stratified as in step 2's Gate 1. Each is checked at its decision and at 3 plan
  states: approach, braking and corner. **Coverage rule:** the checked states include
  corridor-contact, slip-active and pedal-bound plan states, sampled explicitly (at about 5%
  prevalence, the phase picks rarely land on one).
- **Shifts.** A2 decisions: the x_0 bounds. Plan states (t ≥ 1): pin the prefix with lbx = ubx on
  x_0…x_{t−1} and a_0…a_{t−1}; relax x_t's own corridor and slip rows and its box (not s's window
  bound); shift defect row t − 1 by c·e_i. Then dJ_W/dc = γ^t λ_{t,i} + γ^{t−1}[i = s]/10. The
  prefix must be pinned: relaxing an active row with the prefix free moves the base optimum.
  Anchored decisions: the same, with J_tail. A tail re-solve from x_t (prev a*_{t−1}, W − t steps,
  x_t's bounds shifted) is an optional cross-check.
- **Contacts and bounds.** Where a shifted x_t pushes past a tangential contact at x_{t+1}, use a
  one-sided difference away from the contact, or a smaller c (one side hit max_iter at t = 101 in
  the reviewers' check). At pedal-saturated states, use a one-sided (c < 0) pedal difference
  against the folded target, because a +c shift is infeasible.
- **Steps:** 10⁻⁴ for s, n, vx and vy; 10⁻⁶ for ξ, r and δ; 10⁻⁵ for pedal. Richardson over
  (c, c/2); re-solves at tol 10⁻¹⁰ (`fd_solver`, `resolve_from`; kkt.py:980, 1004).
- **Pass:** per component, median relative error < 10⁻³ and p95 < 10⁻², with Gate 1's zero-field
  rule.
- **Kinks:** if the forward and backward differences disagree by more than 10× the Richardson
  discrepancy, the comparison does not count toward the pass. A component with more than 30% kinks
  is descriptive.
- **Planted bugs must fail:** no e_s/10; no γ^{−t}; a sign flip; an off-by-one defect block; J_W in
  place of J_tail; the pre-jump form in place of the post-jump form (on the contact states).
- **π on 10 anchored keys:** set lbx = ubx = a_0 + c on a_0 and
  lbg = ubg = Φ(x_0, a_0) − Φ(x_0, a_0 + c) on defect row 0, so x_1 is exactly unchanged. Φ uses
  the key's own t = 0 parameters (Pk[:,0], J0[0], q, ctx). J_tail = (J_W − r_0)/γ, with the
  window's r_0 = (s_1 − s_0)/10 − 0.02‖a_0 + c‖² taken from the re-solve's own w (a_{−1} = 0;
  window.py:485, 758), never the agent's a_{L−2}. Equivalently, J_tail = Σ_{t≥1} γ^{t−1} r_t. The
  Richardson central difference of J_tail must equal 0.04(a*_1 − a_0). Planted bugs must fail: the
  shift +Φ_a·c; r_0 with `anchor['a_prev']`; r_0 with the agent's a_{L−2}. The λ shifts cannot
  detect the last two, because r_0's action term is constant under them.

**K2, horizon (a confirmation).**
- **Data:** the 6 tier V A2 twins and the 2 imposed twins (Section 2), plus 30 new W = 800 solves
  of W = 400 A2 keys (stream `costate_horizon`, stratified by R), by 400 → 800 continuation from the
  stored key (`solve_continuation`, window.py:837). They are cheap insurance on radii, tier C and
  phase keys the twins do not cover. A different basin is flagged.
- **Expectation, stated now:** t_H ≈ W − 130, beyond T's largest t (120 at W = 400, 199 at
  W = 800). The retention rule should drop nothing.
- **Retention:** keep t ≤ t_H, the largest t at which the p90 of |λ^400_t − λ^800_t| / SD_T(λ_k) is
  ≤ 0.1 for every component of K_P. The cut is on remaining horizon: W = 400 keys keep t ≤ t_H and
  W = 800 keys keep t ≤ 400 + t_H.
- **Fallback:** if any W = 400 key's plan has not reached s_c1 + 50 m by t_H, the affected W = 400
  A2 keys are re-solved at W = 800 by continuation from the stored optimum. K0 and K1 run on them,
  basin changes are flagged, and they become T's targets, with Section 4's folds and the bands
  unchanged. Cost: 10–25 CPU-s per key (A2 W = 800 keys: median 10 s, p90 39 s), at most about
  2 CPU-h.
- This supersedes the horizon remark in step 2's design Section 12, which stays as tagged.

**K2b, road beyond the preview.**
- 30 A2 keys stratified by R and θ (synthetic strips only, never R_TEST; behind the lockbox guard,
  perturb.py:333-337). For each, a variant strip with a second corner starting just beyond the
  maximum over its T states of s_t + d_max(vx_t), at two severities (a fast and a slow corner),
  built with step 2's `perturbed_track`/`strip_arrays` or the stimulus generator.
- Re-solved by warm continuation from the stored key.
- Reported: |Δλ_k|/SD_T per K_P component and per phase stratum (exit and λ_vx are expected to move
  most).
- **Rule:** if the p90 of |Δλ_k|/SD_T exceeds 0.1 in a phase stratum, component k is excluded
  from decisions in that stratum. Otherwise Δλ is reported as the unseen-road share of λ. The
  critic's gradient cannot carry that share, so it bounds the attainable R²_c, beside the input
  ceiling.

**K3, pullback and wiring.**
- **J_z:** AD against float64 central differences on 200 T states, < 10⁻⁶ relative. Only states
  where the clip mask, the tyre min(·, 2) and the d_max clip are unchanged within ±h count.
- **On-manifold pullback:** perturb x_t, recover x_{t−1} by Newton on the control step at fixed
  a_{t−1}, recompute accel; AD and differences agree to 10⁻⁶.
- **Accel-normal plant:** V_c = V(x, a) + c·φ(z_acc − A(x, a)), for arbitrary c and φ. The primary
  pullback must return ∇V and the right π to 10⁻¹⁰; the accel-held pullback must fail.
- **Planted linear critic:** V = wᵀz must give the pullback of w to 10⁻¹², on the right channels
  (208-211).
- **Random-init critics:** AD must match finite differences to 10⁻⁶. Head gradients likewise on
  200 states, excluding states where a head's ReLU pre-activation changes sign within ±h.
- **Observability:** on an exit-straight state (ds > 25 m past s_c1), a transitional state
  (0 < ds ≤ 10 m) and an apex state, s is flagged on the exit straight, ξ's and vx's statistics
  with and without the s column agree to 10⁻⁶ relative there, and the result is the same on
  float32-upcast and float64 key arrays.
- **Extractor self-consistency,** in float64 to 10⁻¹⁰ (random-init networks before the tag; trained
  networks after it, on D): h3·Dense_0 + b reproduces `Actor.apply`'s mean and `Critic.apply`'s V;
  u_i·Dense_{2i+1} + b reproduces each head output of `Critic.apply(..., with_heads=True)`; every
  generalist has exactly 5 heads, S1–S3 none.
- **After the tag:** each trained critic's float64 V on D must match float32 to 10⁻⁴.

**K3b, the policy's own costate loop** (before the tag).
- AD against Richardson central differences (h, h/2) of J along each x_0 component and a_{−1},
  step sizes as in K1. Only comparisons whose discrete structure (saturation set, obs clip, d_max
  and tyre kinks, end of run) is unchanged count, by step 2's rule with a kink radius.
- **Pass:** per component, median relative error < 10⁻³ and p95 < 10⁻².
- **Controls that survive:** step 2's linear driver and its hybrid small-gain random-init actor
  (network.py:388, 416-460). Plain random-init actors leave the road and would count almost
  nothing.
- The δ-perturbation loop reproduces `closed_loop`'s Trace at δ = 0 to round-off, and its
  per-state VJPs match restarts from those states.
- **Planted bugs must fail:** the action clip dropped from the reward; a wrong γ power; an
  off-by-one prev_action.
- The share of comparisons that count is reported. After the tag, each network runs a trained
  analogue; a network that fails has descriptive C1-own.

**K4, planted representations** (the whole probe pipeline). λ̃ is the per-component standardised,
winsorised target; where R²_wd applies, SNR is defined against the within-direction variance.

| Plant | Construction | Pass |
|---|---|---|
| PR-1a | Random-init h3 plus one unit per target, the targets independent of z and of each other (each component permuted across geometries with lengths matched, or smooth synthetic series along each trajectory); SNR 0.25, 1 and 4 | Held-out R² and Δ2 within 0.03 of SNR/(1 + SNR) |
| PR-1b | Real λ, one component at a time: u_k = r_k + η_k, where r_k is λ_res,k residualised out-of-fold on the plant's own baseline (random h3 ⊕ z ⊕ m), Var η = Var r_k/SNR, scale α = the median h3 unit SD | Partial R² (ΔR²/(1 − R²_b)) within 0.03 of SNR/(1 + SNR); Δ2 within 0.03 of (1 − R²_b)·SNR/(1 + SNR); R²_b reported |
| PR-2 | Random-init h3 plus m; a variant whose output m is a*_t | Δ2 within 0.01 of 0 |
| PR-3 | Random-init h3 plus one-hot geometry and pair codes | Grouped split: gain ≤ 0.01. A deliberately mis-grouped split (by cell or key, mirror partners separated) must show the leak |
| PR-4 | Trained-architecture actor and critic trunk, through the shared extractor, whose h2 holds tanh of a linear readout of z fitted to λ on all geometries | The readout's R² recovered from h2 and h3 within 0.02 |
| PR-5 | λ̃ plus noise of known SNR as the critic's gradient | R²_c within 0.03 of expected; at SNR 1–4, C1 says confirm or inconclusive, never refute |
| PR-5b, c, d | −λ̃; one component sign-flipped; 10·λ̃ | b: R²_c ≤ 0 within its CI, no confirm, no label. c: no confirm. d: confirm, but outside the label band |
| PR-6 | Random-init h3 plus about 32 tanh units of random linear mixes of (s − s_apex, vx, σ, log R), scaled so a probe on them alone matches the decoy kNN's R² | Primary Δ2 within 0.01 of 0, while PR-1b is still recovered on the same pipeline |
| PR-7 | Random-init h3 scaled ×0.1 and ×10; z channels scaled by random factors in [0.2, 5] | Δ2 and R²_z unchanged to 10⁻⁶ |
| PR-8 | Random-init h3 plus a unit holding σ^ped_t + noise of known SNR | C2-sw's gain within 0.03 of expected |
| PR-9 | Synthetic heads sharing an exact V on visited states, differing by c_k·u(z), with u = 0 on T and a known nonzero gradient there | D_V ≈ 0 and the planted D_grad recovered |

**K5, nulls and controls on random-init networks.**
- **Permuted targets (wiring):** the training targets are permuted state by state within each outer
  training fold (stream `costate_perm`, all 6 folds) and scored against the unpermuted held-out λ.
  The R² CI must lie at or below 0.
- **Control task (descriptive):** for each k ∈ K_P, f_k is a fixed random 2-layer tanh function of
  standardised z (stream `costate_control`, stated gain), standardised on T. R²*_k is λ_k's
  MLP-on-z ceiling on the real pipeline. y_ctrl,k = f_k(z) + η, with η i.i.d. Gaussian, drawn once
  per state, of variance Var_T(f_k)(1 − R²*_k)/R²*_k; the noise lies outside z, like λ's
  beyond-preview part. The control's ceiling must land within 0.02 of R²*_k, or σ_k is adjusted by
  bisection for a fixed number of steps. Targets are standardised the same way as λ.

**K6, coverage.** Exclusions and strata are counted per component and per reason (unobservable,
with s by ds band; multi-basin; contact; pedal-saturated; kappa clipping; beyond-preview), with
T's size after K2 and the interior share by phase.

**Freeze.** As in `fields/prereg.py` (prereg.py:208), the tag binds:
- the target code (the post-jump form and the pedal fold) and the state lists (T with its t-range,
  D and N), the strata rules (k and thresholds), τ_s and the observability thresholds;
- D₁, D₂, f̂_D, the residual shares and the power floor, the weights, fold partition 0, the 10
  robustness partitions and the inner stream, the penalty grid, the Huber constant and the
  winsorisation bounds on λ;
- the null definitions (value-fit, QSS, barrier, far-geometry), the random seeds, the C3
  covariates, its power table and the pre-signed C3-int pairs, the kappa sensitivity set, the
  value-accuracy threshold, and the bands;
- the networks with their checkpoint sha256s (which cover the head subtrees Dense_1–Dense_10), and
  the code hash. The analysis runs once.

## 8. Budget

| Item | Estimate |
|---|---|
| Re-polish and Φ_x | About 1 min for the 481 A2 keys and 12 min for the anchored finals, on 4 workers. One jac_g per A2 key, about 20 CPU-min; anchored keys use the cheap route |
| Targets, fold, strata, decoys, value-fit nulls, kappa flags | Minutes (key files and input side) |
| K1 | About 6,400 warm re-solves, plus the coverage states; about 1 CPU-h |
| K2 | 30 continuations at 10–40 s each: 10–20 CPU-min (fallback at most about 2 CPU-h) |
| K2b | 60 variant re-solves by warm continuation: about 1 CPU-h |
| Network side | Per network: one forward pass, one main-V VJP and 5 head VJPs on about 10k states. Random controls: 25 weight sets through all 34 networks' obs_rms (850 forward passes, 680 critic VJPs), seconds each. C1-dev: 16 snapshots × 9 critics. Tens of minutes |
| Policy's own costate | D at H = W and H = 800, one own-run sweep per D run, about 1,000 T restarts per network, and N: 0.1–0.3 s per VJP; about 4 CPU-h for 34 networks. K3b: about 0.5 CPU-h |
| Own width price, value accuracy | About 1 CPU-h each |
| Probes | Ridge with 2-D penalties over 11 partitions, C2's 5 random fits per network, bootstrap, permutations: 1–2 CPU-h. MLP probes: 1–3 CPU-h |

In total, about 10–15 CPU-h on this machine after step 2's run. No rented box.

## 9. What would be learned

**C1.**
- **Confirm:** a deep on-policy critic trained only on values has an input gradient that matches
  the window problem's costate in sign and shape, beyond value-fit, QSS, barrier and
  phase-template gradients ("at scale" if the slope band holds). Critic gradients would then be a
  cheap costate estimator, for example of where speed is worth something.
- **Refute** (R²_c below 0.2), read with C1-own, the value-accuracy check and C1-heads:
  - values accurate and D_grad high: the regression does not determine the gradient;
  - D_grad low: the critic has a determined gradient, but of V^π rather than V* (suboptimality or
    the gaps of Section 3.3), not Sobolev-style indeterminacy.

  The actor's GAE signal then carries the projected error over about 1 s. The matching
  intervention, testable like H4, applies DHP/GDHP/VGL: a value-gradient term in the critic loss,
  aimed at the action-projected gradient error rather than at per-component R² over K_P.
- **Inconclusive:** most likely wide CIs over 48 geometries, or too small an interior stratum.

**C2 and C2-sw.**
- **C2 confirm:** the actor holds costate information beyond its input, its output and smooth
  position-and-speed variables. "An anticipatory or switching-function variable" is claimed only
  if C2-pairs also confirms, and the switching function only with C2-sw. The causal follow-up is
  to steer activations along the probe direction and measure brake onset.
- **C2-sw confirm:** the actor holds a countdown to brake onset on saturated stretches, beyond the
  input.
- **Refutes:** the actor maps observations to actions with no linearly accessible costate stage
  beyond position and speed. If the costate is anywhere, it is in the critic.

**C3.**
- **C3-w confirm:** networks whose critic prices the inside wall like the optimum complete more
  R-dev circuits. We cannot tell critic → policy from policy → V^π ≈ V*; a critic-gradient
  intervention would be the causal test.
- **C3-int confirm:** within a network, the geometries where the critic misprices in the
  pre-signed direction are the ones where its driving deviates. This is the most direct
  behavioural link available without an intervention.
- **"Rules out a strong association" or inconclusive:** C3-R²c's null is what steps 1 and 2 predict
  (width perception rather than value); effect sizes are reported.

**With step 2.** A W1 gap against ν^in combined with an H2′ deficit is what a perfect critic of a
low-G, buffer-keeping policy would show. "The networks do not value the inside wall the way the
optimum does" is claimed only if the critic's width price differs from its own policy's (CI
excluding 0) in the direction that would discourage tracking.

## 10. Open questions

**v1's questions, resolved by the review.**
1. T (interior stratum) is primary; D is descriptive; N replicates per agent.
2. λ_s is in K_P: on T it is observable before the exit and insensitive to truncation.
3. The policy's own costate is a pre-registered secondary (C1-own), gated by K3b. The reason "D
   only" did not hold (own-run sweeps, T restarts, N).
4. R²_wd on winsorised targets, with a robust sign-constrained calibration and a signed Spearman in
   C1's confirm rule.
5. C2 conditions on the decoys by residualising on D₁; D₂ conditioning is descriptive.
6. Context prices: the 31 generalists only (all have fixed_car false and context var about 1/3).
   The specialists are excluded even descriptively: their context var is about 5·10⁻¹³, so
   z_ctx = 10⁴·Δo and the clip binds at |Δo| > 10⁻³. The network-side target is the total
   derivative d/dctx of V_c(observe(x, ctx)) at fixed x: the ctx_obs channel path (through
   2/(hi − lo)) plus the `params_from_context` path into δ/δ_max and the tyre channels 212-215
   (racing.py:110-119). With `fields.network.observe`, the gradient is also taken with respect to
   ctx_obs, because `.at[CTX].set` cuts the st.ctx path. The optimum side is Π^μ and the other
   ctx columns of F_p, restricted to τ ≥ t.
7. N covers step 1's 12 finals; per-agent replication of C1 and C2 is possible there.
8. The switching function is a primary actor target (C2-sw).
9. C3's y is completion.
10. Idea 2's tag waits for step 2's decisions, so that C3-w and C3-int can be pre-registered.

**For the lead.**
1. **Weighting unit.** Keys weigh equally (probing review); the geometry-equal scheme the
   feasibility review proposed is kept as a sensitivity. Confirm.
2. **Scale in C1.** The slope band [0.5, 2] gates only the "at scale" label; the science review
   wanted it in C1's confirm rule. Confirm or move it.
3. **C1-own.** A pre-registered secondary with C1's bands (science review) rather than descriptive
   (feasibility review). Confirm.
4. **Interior stratum.** With k = 20, C1 and C2 lose most braking-into-apex and corner states (on
   the example key, t = 24–55 and 81–101 are near-limit). k = 10 keeps more. Choose k, and say
   whether the corner phase should get its own decision.
5. **Power floor.** ξ's residual share (about 0.09) is below 0.10, so C2 may be decided on
   {s, n, vx} or fewer. Accept, or lower the floor?
6. **Primary family size.** C1, C2, C2-sw, C3-w and C3-int are all primary. Keep, or demote C3-int
   (it needs step 2's per-geometry G and B) to secondary?
7. **Thresholds still to set:** the value-accuracy threshold (proposal: out-of-fold R² ≥ 0.8), the
   interior clearance and slip-margin thresholds, the QSS and barrier value definitions, and
   C3's competence covariate (proposal: final training episode_return).
8. **K2's 30 W = 800 solves:** kept as insurance (science review) rather than dropped (math
   review). They cost 10–20 CPU-min.

## 11. What changed from v1, and why

Findings are labelled by lens: M (math), P (probing), F (feasibility), S (science).

| Finding | Change | Sections |
|---|---|---|
| M1, P4, F1: for t ≥ 1 the target was the pre-jump costate wherever x_t's own constraint binds | Post-jump λ_t from the next defect block for every t, including anchored x_1; cheap route and two-route K0 check; K1 pins the prefix and relaxes x_t's rows; planted pre-form bug and K1 coverage of contact states; contact stratum; Π and field sums over τ ≥ t; example quotes λ⁺ | 2, 3.1, 3.4, 4, 7 |
| M2: the accel-held pullback is not the derivative for which V_c = V* implies g = λ | On-manifold pullback (M_x, M_a) for t ≥ 1 rows; accel-held at t = 0 and as a sensitivity; accel-normal plant and Newton FD check in K3 | 3.2, 7 |
| M3: λ_pedal unidentified at full throttle | Pedal fold to unique multipliers, K0 residual, one-sided K1 pedal difference, saturation flag and stratum, fallback exclusion; Section 2 lists the pedal bounds | 2, 3.1, 3.2, 6, 7 |
| M4, S7: truncation framing wrong for s and overstated for vx | Bias formula; s least exposed; twin numbers; K2 a confirmation with a stated expectation; K2b beyond-preview control | 2, 3.1, 3.2, 7 |
| M5, S12: switching function and the PPO signal misstated | Stationarity only; σ_t with e_s/10, π and rate terms; K0 action-row identity with planted bugs; path (a) in GAE terms; q_c descriptive | 6, 7, 9 |
| M6, F12: K1 π shift sign; a_{−1} = 0 | Exact defect shift; r_0 with a_{−1} = 0; three planted bugs; K0 anchored a_0 companion; `anchor['a_prev']` warning | 2, 7 |
| M7: Π^slip is not a grip price | Renamed dV/d ln α_max; Π^μ added | 3.4 |
| P1: C2 cannot separate the costate from decoys | C2 on cross-fitted λ_res; power floor; PR-6; C2-pairs; split-half reliability dropped; C1 unchanged in this respect | 4, 6, 7 |
| P2, S1: C1's comparator and refute reading | Absolute bands; value-gradient nulls; random critics a sanity floor; Δ1 descriptive; ΔC1; C1-dev; the "no linear readout" claim deleted; Section 9 rewritten | 6, 9 |
| P3: affine calibration blind to sign and scale | Slope clamped ≥ 0; signed Spearman in confirm; scale band on the label; PR-5b–d | 6, 7 |
| P5: probe scaling and penalties unspecified | z-scoring with dead-unit drop; two penalties; FWL check; weights; PR-7 | 4, 7 |
| P6, F9: K4 pass criteria | PR-1a and PR-1b with correct expectations; PR-2 a*_t variant; PR-3 mis-grouped split | 7 |
| P7: raw R² rewards turn direction | R²_wd for every threshold; direction-only floor | 4, 5 |
| P8, F10: weighting, folds, T size | Key weights with three sensitivities; braking and non-braking sub-scores; C1-braking; exact fold algorithm; 10 more partitions; T ≤ 9,967 | 4, 6 |
| P9: C3 underpowered, refute unreachable | Power table; equivalence bound instead of refute; y settled; no within-run checkpoints for power | 6 |
| P10, F8: control task and shuffle | Control task descriptive with noise matching; permuted targets a wiring check; D₂; phase template | 5, 7 |
| F2: observability singular on the exit | Column-residual rule via QR; τ_s for s; K3 three-state check; K6 by ds band | 3.2, 7 |
| F3, S6: the policy's own costate | closed_loop with s_stop = ∞ and the reward rebuilt; δ-sweep loop; K3b; T interior, own runs and N; value-accuracy check; C1-own pre-registered | 3.3, 6, 7, 8 |
| F4: K0's x_0 identity cannot fail | Relabelled as an implementation check; solver-option assert | 2, 7 |
| F5: K2 fallback collapsed the geometry design | Conditional W = 800 re-solve keeping all 48 geometries; budget corrected | 7, 8 |
| F6: no random critic; counts off | `random_critic_params` spec; own obs_rms; counts and budget | 5, 8 |
| F7: hidden-unit extraction | Shared extractor, head pairing by construction order, self-consistency gate, PR-4 on the critic trunk | 4, 7 |
| F11: kappa clipping unhandled | Input-side flags, shared sensitivity set, support regime, C3 note | 4, 6 |
| F13: N and context prices | N = step 1's 12 finals; context prices for generalists, as a total derivative; `observe` with ctx_obs everywhere | 2, 4, 10 |
| S2: near-limit states | Interior stratum for decisions; C1-limit; robust calibration; overlap covariate in C3 | 3.3, 4, 6 |
| S3: C3 measured the wrong axis | C3-w (width axis) and C3-int primary; C3-R²c secondary with its expected null | 6, 9 |
| S4: the H2′ story fits a perfect critic | The policy's own width price; three comparisons; claims only against the own price; locality caveat; λ_n carries the wall price | 1, 3.2, 3.4, 9 |
| S5: C2's theory and the switching function | C2-sw primary; C2's meaning restated; D₂ variant; a*_{t+k} descriptive; example timings corrected | 2, 6 |
| S8: λ_s demoted for wrong reasons | λ_s in K_P, with its safeguards | 3.2, 10 |
| S9: novelty and "an exact critic's gradient would equal it" | V* defined as the window problem's; DHP, VGL, Sobolev and Ilyas cited; 3.3 lists the MDP gaps; the intervention framed as DHP/VGL | 1, 3.3, 9 |
| S10: unused ensemble heads | C1-heads; head-gradient FD in K3; PR-9 | 6, 7 |
| S11: ∂V_c/∂accel is not non-Markov use | "Accel-channel sensitivity" wording; grip conditioning in 3.3; tyre-load check | 3.2, 3.3 |
| Not applied | S2's cite fix window.py:90-91 → 89-90: line 89 is `edge_margin`, and lines 90-91 hold the 3 cm and 10⁻⁴ rad margins, so the cite stays | 3.3 |

## 12. Lead decisions on the open questions (v2.1)

These were taken on 22 Oct, before step 2's confirmatory decisions existed. C3 depends on step 2's
outputs, so fixing it before seeing them keeps it prospective.

1. **Weighting.** Each key is weighted equally (primary). Equal weight per geometry is a
   pre-registered sensitivity analysis.
2. **Scale band.** [0.5, 2] gates only the "costate gradient at scale" label, not C1's decision.
   C1 asks about sign and shape. The slopes are reported per component and stratum.
3. **C1-own** is a pre-registered secondary, valid per network only after K3b's trained analogue
   passes.
4. **Interior stratum.**
   - The primary horizon is k = 10 steps (0.5 s) from the nearest wall or slip contact. It keeps
     more of the braking and corner states that the question is about.
   - k = 20 and a corner-phase stratum (s_c0 to s_c1) are reported with the same statistics,
     descriptively.
5. **Power floor.** It stays at 0.10 of the target's variance in λ_res. A K_P component below it
   is descriptive for C2, and C2 is decided on the components that pass. If none passes, C2 is
   not testable. The floor is not lowered to rescue a component.
6. **Primary family: C1, C2 and C3-w.** C2-sw and C3-int are named secondaries.
   - C3-int depends on step 2's per-geometry G and B values.
   - C2-sw tests a narrower mechanism.
   - Five primaries would dilute the family.
7. **Thresholds fixed at the tag:**
   - value accuracy: out-of-fold R² ≥ 0.8;
   - C3's competence covariate: final training episode_return;
   - the interior clearance and slip margins, and the definitions of the QSS and clearance-barrier
     values, are fixed in code and checked in the dry run before the tag.
8. **K2's 30 insurance solves** are kept (about 10–20 CPU-min).



## 13. Implementation decisions (v2.2; before the tag, no trained network computed)

**Code.** `src/apexzero/costate/` (targets, states, critic, probes, gates, nulls, analysis,
prereg) and `scripts/costate.py`, with 117 tests. It was built in stages and reviewed on three
lenses (math, probing and analysis, integrity and tests). 25 findings were confirmed, all fixed,
including three critical ones:
- C1's calibration gave uninformative gradients a negative floor;
- the confirmatory analysis would have found no nulls;
- a dry analysis computed every decision.

**C1's calibration (Section 6).**
- Odd components get direction-specific intercepts.
- The slope is a weighted least-squares slope, clamped at ≥ 0. Its leverage weights are squared,
  centred on the median, with K_LEV = 3; first-power weights still let a few spiked rows crush the
  slope.
- The intercept is the training fold's weighted mean of λ − b·g (plain weights when b = 0).
- Out-of-fold predictions are clipped to the winsor bounds, and the clip share is recorded.
- Huber is kept as a coded option, unused.
- The planted control PR-5 must give SNR/(1 + SNR) within 0.02. Uninformative, zero, noise,
  direction-only and anti-aligned gradients must score in [−0.02, 0.005] (gate PR-5e).

**The "at scale" label.** A Frisch bracket: the CI lower end of the slope of λ on g must be at
least 0.5, and the CI upper end of the inverse slope of g on λ at most 2.

**K1's pedal step at saturated states.** A one-sided ladder of 1e-3, 5e-4 and 2.5e-4, with a
noise-signature fallback, not 1e-5. Smaller steps measure re-solve noise.

**C1-own.** λ^own is the on-manifold derivative, with accel moving with the state; the
accel-held form is reported beside it. Section 3.3's "accel held fixed" applies to t = 0 only.

**C2-sw's stratum.** The pedal-switching steps (σ^ped > ACTIVE_TOL), where its target is nonzero.
The raw pedal-saturated flag is descriptive.

**Value-fit nulls.** They fit the horizon-corrected V*_∞. The window-target fit is kept as an
extra null.

**Rows and weights.**
- C3-w's x is scored on the interior stratum, as C1 is ("on T" in Section 6 means T's interior).
- The power floor (Section 12, item 5) is computed on the interior shares; the T-wide value is
  reported.

**Exclusions.**
- Multi-basin keys are dropped at the source, with their row counts recorded (4 keys and 80 rows
  at the time of writing).
- K2b's per-phase exclusions are frozen in the record.
- The kappa-clipping set is a sensitivity variant.

**Gate rules.**
- K0's verdict now requires the action-row identity (1e-9), the straight-road identity (1e-12),
  the end gap (1e-10), and every action-row plant failing by more than 1e-4.
- K3's observability sub-checks (the with/without-s agreement, and the transitional rows) are
  reported, not gated.
- **K3's pass rule, fixed after its full-size pre-tag run** (random and planted networks only, 22 Oct;
  before the tag and before any trained network is computed).
  - On 221 states, 19 of K3's 20 checks passed on the first run. Since then, K3 has 22 checks and
    passes 22 of 22 (see "Built before the tag").
  - The finite-difference pullback check passes for the held and xprev forms, at errors of
    4–9e-8 against 1e-6.
  - The on-manifold form's worst error is 1.3e-6, at the one row where the stencil crosses an
    action clip inside the manifold's Newton recovery. 98.7% of its comparisons are counted.
  - The rule: the held and xprev forms keep FD_TOL = 1e-6. The on-manifold form uses 1e-5,
    because its reference passes through a Newton solve over clipped actions.
  - Every planted wrong pullback (for example the held form on the accel-reading plant) must still
    exceed 1e-4, ten times the on-manifold tolerance, or K3 fails.

**Built before the tag** (implemented, reviewed on two lenses with 14 findings fixed, 123 tests).
- **K3.**
  - Its final form has 22 checks.
  - Every planted wrong pullback must exceed 1e-4 (24 cases; the smallest measured is 0.030).
  - The plants' own forms must match their finite differences.
  - The accel-reading plant's wrong held forms (g, π, g_xprev) must each exceed 1e-4.
  - At full size: held and xprev forms at most 9e-8 (limit 1e-6), on-manifold at most 1.3e-6
    (limit 1e-5).
- **Value accuracy** (Section 12, item 7), as Section 3.3 defines it.
  - It is the out-of-fold R² of the calibrated σ_R·V_c against the network's own deterministic
    discounted return.
  - The return is computed by the network's own closed loop, up to 1,000 steps or the strip's
    end, from T's interior restarts and from D.
  - It uses the frozen folds and C1's calibration. The threshold is 0.8, and every requested row
    must be scored.
  - Crashed runs are dropped, and their share is reported.
  - The same score against the optimum's V*_∞ is reported descriptively and never decides
    anything.
  - C1-own is a secondary only when both K3b's trained analogue and value accuracy pass;
    otherwise it is descriptive, with the reason.
  - Pre-tag check: the best of 20 random-init critics scores R² 0.23, and none passes.
- **By-stratum reports.** These are descriptive tables, never decisions.
  - **C1 and C2:** interior (k = 10, primary), k = 20, near-limit, contact, corner band, and
    interior with each of: the four phases, braking or non-braking keys, turn direction ±1,
    W 400 or 800, pedal-saturated, and κ-clipped. κ-clipped is a per-network column.
  - **C2-sw:** pedal-switching (primary), its approach and exit parts, all four phases, and
    pedal-saturated.
  - A descriptive C1-pattern on the pedal-saturated stratum.
  - Sensitivity variants carry the label scope "descriptive".

**Deferred.** These are not part of `costate-prereg-v1`. If run later, they are exploratory.
- C3-int, C1-dev, ΔC1 and C2-pairs.
- C1-own on its other two state sets (Section 3.3 (ii) and (iii)): every state of each network's
  own run from each D key, and N. They would need critic pullbacks at own-run states. C1-own
  is decided on T's interior restarts only.

**Order before the tag.**
1. Stage inputs (with no t_H yet), then K2, then inputs again (binding t_H).
2. K2b, then K0 on every A2 key and every anchored final, with the stricter verdict.
3. K1, then K3 at full size, then K4.
4. The nulls.
5. A final pre-tag review, then prereg and the tag.

**Post-tag cost.** C2 runs 17 sensitivity variants per network. The rough, unmeasured estimate is
20–30 minutes per network on the full T set.

