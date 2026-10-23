# Idea 2: does the network encode the costate? (design v2.5, not frozen)

Status: 23 Oct 2026.
- **v1** was reviewed on four lenses (math, probing, feasibility, science): 42 findings were
  confirmed and 2 rejected.
- **v2** applies the confirmed findings; Section 11 maps each one to its change.
- **v2.1** records the lead's decisions on the open questions (Section 12). They were taken before
  step 2's confirmatory decisions existed.
- **v2.2** records the implementation in `src/apexzero/costate/` (merged 22 Oct) and the decisions
  it required (Section 13). The implementation was adversarially reviewed: 25 findings were
  confirmed and fixed.
- **v2.3** records the decisions of a final pre-tag review of that code (Section 14): 21
  findings were confirmed, and each is resolved by a code change, a rule stated there or a
  deferral.
- **v2.4** records the gate rules after the full-size pre-tag run (Section 15): K1's rule
  written down, K5's wiring check, K4's PR-1b, PR-6, PR-6b and PR-6c, f̂_D's one k = 8, and
  K4's no-gain plants as paired increments. A third full-size check passes every gate. The
  rules are final from this version.
- **v2.5** records a review of v2.4 (Section 15.14): 10 findings were confirmed. C2's
  random-only null is measured and frozen in the record (its SD on C2's decision rows is about
  0.03, not the 0.007 v2.4 stated). Two pre-tag stages are added: decoy-k re-derives f̂_D's k
  and its validation table on the canonical world, and c2-null computes the null. The inputs
  that K4 and the new stages read are bound at prereg, K4's band notes stand beside C2 only,
  and Section 15's record of the code and the rehearsal is brought up to date. No gate rule,
  tolerance, seed or k changes.

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

*Reading guide.* Sections 13–15 record how these gates are applied as coded: K3's pass rule
(13), the pre-tag review's rules (14), and K1, K4, K5 and f̂_D's k after the full-size pre-tag
run (15).

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
- Section 14 (v2.3) extends this list. The record's 'deferred' list holds all of it.

**Order before the tag.**
1. Stage inputs (with no t_H yet), then K2, then inputs again (binding t_H).
2. K2b, then K0 on every A2 key and every anchored final, with the stricter verdict.
3. K1, then K3 at full size, then K4.
4. The nulls.
5. A final pre-tag review, then prereg and the tag.

**Post-tag cost.** C2 runs 17 sensitivity variants per network. The rough, unmeasured estimate is
20–30 minutes per network on the full T set.

## 14. Pre-tag review decisions (v2.3; before the tag, no trained network computed)

22 Oct 2026. A final review of the v2.2 code, before the tag, on four lenses (design
conformance, post-tag crash, lockbox and pre-registration, statistics) confirmed 21 findings
(#0–#20: one blocker, six major, fourteen minor) and refuted one. The fixes are in
`src/apexzero/costate/` and `scripts/costate.py`, with 146 tests (the opt-in rehearsal among
them). After the tag the post-tag stages run once, and a crash voids the tag, so the post-tag
path's correctness and crash safety came first. This section states each rule as the code
applies it; the last table maps every finding to its resolution. The numbers come from the input
side (the full-size smoke run of the pre-tag stages at v2.2: its inputs/world.npz and gate
outputs, under each network's stored obs_rms) and from random-init stand-ins.

**Labels (Sections 6 and 12, decision 6).**
- The primary family is unchanged: C1, C2 and C3-w.
- The named secondaries are C2-sw and C3-R²c. C3-int is no longer one: it is deferred.
- The pre-registered secondaries are C1-pattern, C1-heads, C1-own (on T's interior restarts),
  C1-limit, C1-braking and W1. C1-dev, ΔC1 and C2-pairs are deferred, so no longer
  pre-registered.
- W1 as run has no decision band. It is the calibrated agreement (R²_c with its CI, the Spearman
  ρ and the pooled slope) of the critic's uniform inside-widening derivative with Π^in on T's
  interior.
- Section 7 'Freeze' lists "the pre-signed C3-int pairs" among the frozen items. Read this as
  kept in the record, not pre-registered ("The record and the freeze" below).

**What decides on T.** The counts below come from the smoke run's input side. The re-run before
prereg binds them again, from code that did not change.
- K2b's frozen exclusions are: approach n and vx; braking and corner s, n, ξ and vx; exit n, ξ
  and vx. They leave n and vx no interior decision row. s keeps 1,643 and ξ 1,581, the same under
  each of the 9 decision networks' obs_rms.
- So C1, C1-own and C1-braking decide on {s, ξ}, and the strata tables are scored on them. C2's
  power floor passes s, n and ξ (vx fails), but n has no scored row, so C2 is scored on {s, ξ}.
  This is the decision set.
- *Since v2.4* (Section 15.10): at f̂_D's k = 8, λ_n's interior residual share is 0.091, so the
  power floor passes s and ξ only. The decision set is unchanged; the floor now decides it.
- **C1-pattern.** As in C1, a component without a decision row drops out of its side's mean, so
  K_P's side is {s, ξ}. C1-pattern is 'not testable' if either side has no component left.
- **C1-limit.** T has 482 wall-contact rows: approach 2, braking 174, corner 149 and exit 157.
  n keeps none, and ξ keeps the 2 approach rows, both in one geometry. That geometry is absent
  from 687 of the 2,000 bootstrap resamples. A component with a draw that holds none of its rows
  has no CI and drops out, so C1-limit is 'not testable' on T for every network. Its rows are
  not revised.
- **The accel-held pullback's** R²_c cannot differ from C1's here. The accel term reaches only
  vx, vy, r, δ and pedal (Section 3.2: only the body block of M_x is non-zero). So g_held and g
  agree on s, n and ξ, to 10⁻¹⁴ in the rehearsal.
- **C3-w and C3-R²c** are 'not testable' when no recipe stratum varies in both x and y, because
  the permutation null is then the point 0. Networks whose x calibration slope clamps at 0 share
  the same x, which is a tie within a stratum.

**C1's nulls per held-out fold (#20).**
- The nulls fitted on or built from targets are value_ridge, value_mlp, value_ridge_window and
  far_geometry (`analysis.FOLD_NULLS`). They are computed on partition 0's folds as one slice
  per held-out outer fold: an [F = 6, n, 8] array.
- **Value fits.** Slice f pulls every T row back through the value fit made without fold f. The
  fit is not refitted per row, so a training row's slice-f gradient is that fit's in-sample
  gradient.
- *Since v2.4* (Section 15.9): the value fits are cross-fitted in full. A training row of fold
  g carries the fit without folds f and g, so the bullet above is superseded, and the value
  fits' "per held-out fold" numbers below are Section 15.9's "before".
- **Far geometry.** Slice f is, for every row, the median over far geometries whose outer fold
  is neither f nor the row's own fold. For a held-out row of fold f this is the v2.2 predictor.
- Fold f's calibration of such a null (slope and direction intercepts) is fitted on slice f's
  training rows (outer ≠ f) and predicts fold f's rows from slice f. No target of fold f (λ,
  V*_∞ or V*) reaches fold f's prediction.
- QSS, the barrier and the 20 random critics read no target and stay single [n, 8] arrays. On
  robustness partitions 1–10 the four fold-dependent nulls are left out, as before.
- **Why.** v2.2 gave each row one predictor, built without that row's own fold. The training
  rows that fit fold f's calibration then carried fold f's targets: far geometries borrowed
  from fold f, and value fits fitted on fold f's V*. A value fit was also calibrated on a mix of
  five folds' fits.
- **Measured effect** (the smoke run's world, input side only; the interior K_P-mean null R²_c,
  v2.2 then per held-out fold): far_geometry 0.5846 then 0.5908; G1's value_ridge 0.2112 then
  0.2546; G1's value_ridge_window 0.3282 then 0.3344. The v2.2 nulls were weaker, which made
  C1's paired "R²_c minus null" rule easier to pass.
- **Binding before the tag.** nulls/summary.json passes prereg only if every decision network
  lists every C1 null except the random critics, with its shape. value_ridge, value_mlp,
  value_ridge_window and far_geometry must be [6, n, 8], qss and barrier [n, 8], all with the
  same n. Every null except far_geometry must be finite in every slice. far_geometry's
  non-finite (slice, row) count and out-of-fold row count are recorded, not required. On the
  smoke run's world both are 0.

**Rows without an out-of-fold calibration (#10).**
- This applies to every C1 computation: the decision, its variants, the strata table and C1-own.
  A component is scored only on the stratum's rows that are valid for it and whose outer fold
  has a calibration (at least 3 training fit rows for that component). Other rows are dropped
  and counted ('n_uncalibrated'), never scored against a filled-in prediction.
- R²_c, every null's paired R²_c, the Spearman ρ and the pooled slopes use the same rows. A null
  without a prediction on a scored row predicts 0 there.
- A component with no scored row leaves the K_P mean. C1's record gives its null R²_c and
  Spearman as NaN, not 0.
- With no component left, C1 is 'not testable' with a reason: "the rows span 1 outer fold: no
  out-of-fold calibration" for a stratum inside one outer fold (every one-geometry stratum), and
  otherwise "no component keeps a valid row with an out-of-fold calibration" (or "no row").
- Consequence: G2's, G2s1's and G2s2's descriptive 'interior&kappa_clipped' rows (23–24 rows in
  one geometry) are 'not testable', not 'refute'. No decision stratum is affected.
- C1-pattern and C1-limit follow the same rule.
- *Since v2.4* (Section 15.9): so do W1's calibrated agreement and C3-w's x, which zero-filled
  such rows; C1-pattern counts the rows it drops.

**C1's Spearman rule (#1).** For the mirror-odd components of K_P (n and ξ), the Spearman
ρ(g_k, λ_k) in C1's confirm rule is the mean of the within-direction ρ, weighted by each
direction's total weight. This is R²_wd's convention: a raw ρ would reward the turn direction.
For s and vx it is the weighted Spearman over the scored rows.

**Secondaries as coded (#3).**
- C1-limit is scored on wall contacts only (wall_side ≠ 0). Slip contacts have no n or ξ
  direction.
- C1-own uses the unwinsorised λ^own, and its calibrated predictions are not clipped: λ*'s
  winsorisation bounds do not apply to the policy's own costate.
- C1-pattern's pedal mask follows Section 3.2's pedal-saturated flag, unchanged: pedal is scored
  where the pedal state is not saturated, δ where its box is inactive.

**Observability (#5).** No per-component observability threshold is applied, and only τ_s
excludes rows (s, where ‖J_z e_s‖ / median_k ‖J_z e_k‖ < 10⁻⁴). The thresholds that Section 3.2
and Section 7 'Freeze' leave to the dry run are therefore none. The evidence, from the smoke
run's world (T's 9,887 rows), with J_z under each of the 9 decision networks' obs_rms and
`states.observability`:
- The smallest relative squared residual of each non-s column (its squared residual after
  projection on the other columns, divided by its own squared norm), over every row and network,
  is: vx 0.0106, ξ 0.052, vy 0.111, r 0.134, pedal 0.153, δ 0.224 and n 0.412. Every non-s column
  keeps at least a tenth of its norm outside the others' span.
- The smallest absolute squared residuals are: vx 2.8·10⁻³, vy 1.16, n 1.24, pedal 3.9, r 4.3,
  ξ 98.8 and δ 259. vx's column is the smallest (squared norm down to 2.8·10⁻³). Under G1, its
  smallest relative residual is at a t = 0 row, where the s and vx columns nearly share a
  direction (cosine 0.995).
- Structural unobservability looks like s on the exit straight, whose column norm falls below
  10⁻⁴ of the median (at most 5.1·10⁻⁵ there, under G1). No non-s column comes near it: the
  smallest relative residual is about 10⁻². vx and n have no decision row on T in any case
  (K2b).

**C2 and C2-sw 'not testable' (#6; the record of Section 12, decision 5).**
- The primary C2 and C2-sw are 'not testable' only on `probes.NotTestable`: no component passes
  the power floor, or no component passing it has a scored row.
- Two more 'not testable' cases are returned before any fit. Either the stratum has no row, or
  for some held-out fold the stratum's training rows lie in fewer than 2 inner folds, so the
  probe penalties cannot be inner-cross-validated (`analysis.probe_cv_problem`). C1's
  counterpart is a fold without an out-of-fold calibration. The decision strata span T's
  geometries, so in practice this hits descriptive strata only, such as the 2-geometry κ-clipped
  ones.
- Any other error on a primary entry is a failed run, never a verdict. That includes a
  ValueError, a LinAlgError, a broadcasting error, and the "not fitted on the same targets" and
  bootstrap-count checks.

**The accel-held pullback (#0; Section 3.2's pre-registered sensitivity).**
- 'pullback:held' is C1's statistic with the critic's accel-held main-V pullback g_held in place
  of the on-manifold g. It uses T's interior, each network's decision mask, C1's calibration
  with the frozen winsorisation bounds and clipping, the geometry bootstrap and the slope CIs
  per component.
- It has no nulls: 'nulls_omitted' lists every one, the random critics included, so C1's null
  rule is vacuous for it. Its record is marked 'descriptive', with label scope descriptive. It
  reports its band ('band', as a strata row does) and has no 'decision' or 'label_at_scale'
  field.
- 'C1-own_held', beside C1-own, is the critic's accel-held pullback against λ^own's accel-held
  form. It uses C1-own's rows (valid own runs with a finite held λ^own) and C1-own's rules, with
  no nulls. It is descriptive, with the status "descriptive (the accel-held form)".

**C3-w and C3-R²c (#2, #7, #16).**
- **C3-w's sensitivities.** C3-w gets only the drop_small and κ-set variants. Section 4's
  weight, partition and sub-score sensitivities are not run for C3-w.
- **Covariates** (descriptive). In each covariate's stratified partial Spearman, a recipe
  stratum that holds any network without that covariate is skipped whole. The partial ρ is the
  size-weighted mean over the scored strata, with weights n_s / Σ(scored n_s). The output lists
  the networks without the covariate ('missing'), the skipped strata, the networks scored ('n')
  and 'complete'.
- Step 2 has no fields-a2 rows for runs/dr-pilot-v2-tilt30-s3 and -s4. So the overlap partial
  scores 26 of the 31 generalists, with base-v2 skipped. The competence partial scores all 31.
- **C3-R²c** (Section 6; a named secondary).
  - x is each of the 31 generalists' C1 R²_c without nulls: T's interior, the K_P mean, C1's
    calibration and bounds, and the network's own decision mask. For the 9 decision networks
    this is their C1 R²_c, since nulls never enter R²_c.
  - y is the mean R-dev completion from the record's C3-w inputs.
  - The test is C3-w's: the 7 recipe strata, the size-weighted within-stratum Spearman, and a
    one-sided stratified permutation with y permuted within strata and 100,000 draws. It
    confirms at p ≤ 0.05 and "rules out a strong association" when the shift-inverted 95% upper
    bound is < 0.3.
  - It runs on its own seed stream, costate_c3 sub-stream 2 (C3-w uses the stream itself, C3-int
    sub-stream 1).
  - Competence and overlap are reported as descriptive stratified partials, with C3-w's rule for
    a stratum missing the covariate. The weighted ρ is reported as the effect size.

**Networks outside the decision set (#2; Section 6, "descriptive for those").**
- For each of the 22 other generalists, the analysis reports C1 without nulls (descriptive,
  label scope descriptive) and C1-heads (D_q and the two paired comparisons).
- For each of the 3 specialists, it reports C1 without nulls (descriptive).
- The 9 decision networks' C1-heads are in their own per-network records, so C1-heads is
  reported for all 31 generalists. C2 and C2-sw beyond the decision networks are deferred.

**Containment (new rule).** The analysis runs once after the tag.
- Every descriptive computation is wrapped (`analysis.guarded`). If it raises an Exception, its
  result is replaced by {'error': '<ExceptionType>: <message>'}. That is never a verdict and
  never a decision.
- A strata-table row whose statistic raised keeps its n, keys and geometries and gets band
  'error'.
- **Guarded:**
  - every C1 and C2/C2-sw sensitivity and robustness variant except C1-braking;
  - robustness_C1 and robustness_C2;
  - every strata table and row of C1, C2 and C2-sw;
  - the accel-held pullback sensitivity and C1-own's accel-held form;
  - C1-heads (D_q and its two comparisons) for every generalist;
  - W1;
  - the value accuracy's per-head scores, horizons, tails and horizon-corrected score, and the
    V*_∞ value score;
  - C1-pattern on the pedal-saturated states;
  - the specialists' descriptive C1;
  - C3-w's two sensitivity variants (small geometries and κ set dropped);
  - each covariate partial (competence, overlap) of C3-w and of C3-R²c.
- **Not guarded,** so an error stops the run:
  - C1, the primary C2 and C2-sw, and C3-w's test;
  - C3-R²c's test, including the 22 other generalists' C1 without nulls that gives its x;
  - the value accuracy's main-V score (C1-own's precondition);
  - C1-pattern, C1-own, C1-limit, and C1-braking (C1 on the braking keys).

  The primary decisions and the named secondaries' decision rules raise.
- One more case: a network that failed K3's after-the-tag checks decides nothing, so its whole
  analysis or description is contained (next paragraph). Its value accuracy is contained with
  it.
- The record's constants carry the rule, the K3-failed case included
  (constants()['analysis']['containment'], and constants()['critic']['k3_trained']['fail']).

**K3 after the tag (#18; Section 7 K3, "trained networks after it, on D" and "After the tag").**
- **Place.** K3 runs in the critic stage after the tag, after the freeze check and the unlock.
  It covers every pinned network (the 31 generalists and S1–S3), before that network's critic
  arrays.
- **Output.** post/critic/<net>.k3.json, with the verdict repeated as 'k3_trained_pass' in
  post/critic/<net>.npz. The code is `k3_trained` in scripts/costate.py, and `gates.k3_trained`
  and `gates.k3_trained_judge`.
- **D in the code** is T's t = 0 rows. That is 477 states, not 481, because the 4 multi-basin
  keys are left out of T. Each state is the bound world's raw observation, normalised and
  clipped under the network's own obs_rms, as every pullback reads it.
- **Float64.** The work runs in the critic stage's own process, not a separate subprocess. That
  process is already an x64 process: every post-tag stage switches x64 on in `_post_tag`. The
  float32 V is computed in the same process, with x64 switched off around it
  (jax.enable_x64(False), as `random_critic_params` draws its weights). On random-init critics
  this is bit for bit what a float32 process computes (tested).
- **Extractor check.** On D, in float64, max |extractor − Flax| ≤ 10⁻¹⁰ for:
  - the actor's mean (Actor.apply);
  - the critic's V (Critic.apply);
  - each head output (Critic.apply(..., with_heads=True)).

  Both extractors run it: `critic.extract`, which the pullbacks use (`critic.net_apply_gap`),
  and `probes.hidden`, which the activations and C2 use (`probes.extractor_gap`). Every result's
  dtype must be float64. The pre-tag check's input-gradient comparisons are not repeated after
  the tag.
- **Head rule.** Four counts must all equal 5 for each of the 31 generalists and 0 for S1–S3:
  the config's ensemble_heads, the tree's head count, Flax's with_heads output count and the
  probe extractor's count (`gates.expected_heads`, `probes.N_HEADS_GENERALIST`). The tree's
  count equals the config's by construction there: `critic.trained_critic` refuses a tree
  whose head count is not the config's, before any measurement ("Containment" below). So the
  judge can fail on the config's count against the expected one, on Flax's, on the probe
  extractor's, or on a measurement that raised, but never because a tree disagrees with its
  config.
- **Float32 check.**
  - V32 is RunningMeanStd.normalize and Critic.apply in float32, with x64 off, on the float32
    params and the raw D observation rounded to float32.
  - V64 is the float64 V the pullback reads: the critic's output 0 on the float64 normalised and
    clipped observation.
  - Pass: max over D of |V64 − V32| ≤ 10⁻⁴ absolute, with V32 computed in float32. The units are
    the critic's own: Critic.apply's output, reward/σ_R, before σ_R.
  - This reads "match float32 to 10⁻⁴" as an absolute tolerance, like step 2's Gate 3 MEAN_TOL.
    The largest relative difference and the largest |V| are reported, not judged
    (`critic.F32_TOL`, `critic.float32_gap`).
- **Containment.** Before the tag the checks are refused (the trained marks), and that refusal
  is never contained. After the tag, each of the three measurements (both extractor gaps and
  the float32 gap) that raises is recorded as {'error': ...} and fails its check, with the
  error as the reason.
  - The network's setup is not contained. A non-MLP config, or a tree whose head count is not
    the config's, is a refusal in `critic.trained_critic` and stops the critic stage. The
    network's critic arrays (`critic_of`, unguarded right after) would stop it the same way,
    and the design provides no way to skip that network's arrays.
  - These are pre-tag matters: `load_agent` pins each checkpoint by sha256, so the trees after
    the tag are the ones pinned before it.
- **A fail** (new rule, as H4R's gate2-trained rule treats a failing network). A network that
  fails any of the three checks (extractor, heads, float32) keeps every statistic.
  - These records are marked descriptive: C1, C1-heads, C1-own and its accel-held form,
    C1-pattern, C1-limit, C1-braking, W1, C2 and C2-sw, and its strata tables' primary rows.
    Each gets 'descriptive': true and 'descriptive_reason': "K3 trained check failed: <check>
    (<why>); ...". The records, not the rows, also get label_scope 'descriptive'. C1-own's
    status becomes "descriptive (<reason>)".
  - It is treated as missing for every decision it enters. Its per-network C1, C2 and C2-sw are
    'missing' (with 'excluded': the reason), so R18's variant verdict counts it as a missing
    seed. C3-w, both of C3-w's sensitivity variants and C3-R²c leave it out, and list it under
    'missing' and 'k3_trained_excluded'.
  - Its own analysis, or description, is contained, not allowed to raise.
  - The analysis lists every network's verdict ('k3_trained': networks, failed, not_run). The
    record's constants pin the rule (constants()['critic']['k3_trained']).
- **Run once.** The confirmatory analysis refuses unless all 34 networks have a k3.json with a
  boolean decision, stamped by this code under this tag. Each critic's k3_trained_pass must also
  equal that decision. A non-confirmatory analysis without these files treats the networks as
  'not run', not excluded.

**Confirmatory inputs (#11, #14; Section 7, "the analysis runs once").** The confirmatory
analysis refuses to start unless these exist:
- for every decision network, post/own/<net>.npz with lam_own, valid, v_rows, v_own_det,
  v_own_valid, v_own_terminated, v_own_strip_end and k3b_pass;
- for every decision network, post/own/<net>.k3b.json with a boolean decision.pass equal to that
  k3b_pass (both files are stamp-checked like the other post-tag inputs);
- every specialist's critic file;
- every generalist's critic with g, value and dV_dwin (a decision network's also with
  random_g);
- the 34 K3 files above.

A non-confirmatory analysis without the own stage's output records C1-own and its held form as
'not testable', descriptive, with the reason "no own stage output".

**The record and the freeze (Section 7 'Freeze'; #4, #9, #13, #15, #17, #19).**
- **design_hash (#9).** This is step 1's `stimuli.design_hash` over `stimuli.load_design` of
  step 1's stimuli directory: the bank, and the cells as a list in cells.json order. It is the
  value step 2's record holds (e7846c92b1e4…). Every post-tag freeze check recomputes it and
  compares. v2.2 handed it `keys.design`'s dict of cells, a TypeError in prereg and in every
  post-tag freeze check.
- **Step tags (#19).** The record registers step 1's and step 2's tag objects:
  - steps.step1.object is f68d0430cccf…, the pinned STEP1_TAG_OBJECT;
  - steps.step2.object is the local receptive-fields-prereg-v1 object at prereg.

  Every post-tag stage refuses, with a non-waivable 'frozen' problem, when a local step tag is
  not the recorded object, or when the record registers none. In this repository, prereg and
  every freeze check also require the local psychophysics-prereg-v2 to be its pinned object. The
  reason: scripts/fields.py is imported by every stage but is outside the code hash, and only
  step 2's tag freezes it.
- **Deferred (#4).** The record carries a 'deferred' list (`prereg.DEFERRED`, below). Some items
  stay in the record but do not make their analyses pre-registered:
  - C3-int's frozen permutation count and pre-signed pairs (constants.analysis.c3_int_perm and
    c3_int_pairs);
  - C1-dev's pinned snapshots ('snapshots').

  The record labels them as kept, not pre-registered. Any later deferral is added to both
  `prereg.DEFERRED` and the design's deferred list.
- **Environment (#17).** prereg refuses when any bound pre-tag output or array was stamped under
  a different environment from the one prereg runs in. That means a different Python version, or
  different versions of the compared packages: jax, jaxlib, numpy, scipy, casadi, flax and optax.
  It also refuses a stamp that records no environment. The record captures prereg's
  environment, and every post-tag stage is held to it.
- **The nulls under a deviation (#17).** The confirmatory analysis checks the pre-tag nulls
  arrays against the record's code hash and the sha256 the record binds, not against the
  current run's hash. The pre-tag stages cannot rewrite them after the tag, so a logged post-tag
  code deviation does not block the confirmatory analysis.
- **Partial runs (#15).** A partial pre-tag run cannot be bound. These options make a run
  partial: --out, any --limit, --n or --n-extend given (0 included: a K2b on --n 0 passes on no
  key), and --network (on the nulls stage, --network only when it does not cover the decision
  networks). So do these:
  - k0 without N's anchored finals (--kinds without n; the default is now a2 and n);
  - k3b with --steps other than 160 (`own.K3B_STEPS`, the record's constants own.k3b_steps; K3b's
    trained analogue after the tag uses the same count);
  - k4 with --which naming plants;
  - k5 with --ceiling ridge;
  - nulls with --mlp-alphas other than (10⁻⁴, 10⁻³, 10⁻²), or --mlp-maxiter other than 500
    (`probes.MLP_ALPHAS`, `MLP_MAXITER`);
  - inputs with --partitions other than all 11.
- **K3's subprocesses (#13).**
  - Each subprocess's report (gates/k3/plants and random, .json and .npz; extractor_5.json and
    extractor_0.json) is deleted right before that subprocess runs.
  - A critic dry run must exit 0 and write its report.
  - An extractor check must rewrite its report. It must exit 0 exactly when the report passes,
    and 1 exactly when it fails.
  - Anything else stops K3 without writing k3.json. k3.json records every subprocess's exit
    status ('returncodes').

**Deferred** (this extends Section 13's list). These are not part of `costate-prereg-v1`. If run
later, they are exploratory. The record's 'deferred' list (`prereg.DEFERRED`) holds exactly these
nine:
1. C3-int. The record keeps its permutation count and pre-signed pairs.
2. C1-dev. The record keeps its snapshot pins.
3. ΔC1.
4. C2-pairs.
5. C1-own on its own-run and N states (Section 3.3 (ii) and (iii)). C1-own is decided on T's
   interior restarts only.
6. N's per-agent replication of C1, C2 and C2-sw ("N replicates", Sections 3.2 and 6). N's
   states are never built: the inputs stage builds T only.
7. W1's probe R² and Spearman for Π^in, Π^out, Π^slip and Π^μ.
8. The policy's own width price (Section 3.4), and W1's three comparisons of the critic's width
   field with the optimum's and the policy's own, with their calibrated magnitudes.
9. The descriptive C2 and C2-sw of the 22 other generalists and of the specialists. The probes
   stage runs on the 9 decision networks only.

**Not in the confirmatory run (#8).** The analysis does not compute these descriptives from
Sections 3.2, 4 and 6. They are not part of the confirmatory run. They may follow as exploratory
output after the decisions are committed:
- the unwinsorised R² beside R²_c, and raw R² with the direction-only floor;
- R²_z, the MLP-on-z ceiling and R²_c/R²_MLP (`analysis.ratio_to_ceiling` has no caller);
- Section 4's two-stage (FWL) check (`probes.fwl`);
- probes of h1 and h2; Δ2_D; Δ2 without residualisation; Δ2 conditioned on z only;
- the spread over folds;
- C1 and C2 on the D set by itself (D's rows are T's t = 0 rows; as a set, D enters only value
  accuracy and K3's after-the-tag checks);
- the share of g_vx from the accel term, and the accel-channel sensitivity (`accel_sens` is
  stored in the critic files, not reported);
- C1-heads' per-head R²_c and the inter-head reliability ρ_k;
- from Section 6's 'Descriptive only' list: Δ1; R²_{z⊕D₂}, the decoy-only R² and R²_template;
  conditioning on a*_{t+k}; λ probed with channels 207 and 210–211 residualised; the switching
  function against m, λ(t⁻) − λ(t⁺) and the action-projected critic signal q_c with its
  association with completion; the critic's layers, the heads' units and MLP probes of trained
  networks; context prices; the left/right asymmetry index; a pooled Spearman for C3; and C2's
  statistic against completion.

This list is not exhaustive. Any descriptive in Sections 3–6 that `analyse_network`,
`describe_network`, `probes_of`, `c3w_of` or `c3r2c_of` (scripts/costate.py) does not compute is
outside the confirmatory run, and exploratory if run later. Some of Section 6's items are
covered elsewhere: N by deferred item 6, the policy's own width price by deferred item 8.
C3-R²c, like C3-w, gets none of Section 4's weight, partition or sub-score sensitivities. It is
computed once, on partition 0, with no variants.

**Where it runs, and the order before the tag (#12).** This replaces the costate-dev procedure
in the v2.2 code, and step 5 of Section 13's "Order before the tag".
- The pre-tag stages and prereg run in the main checkout.
- prereg.json is committed on main, and costate-prereg-v1 is an annotated tag on that commit. It
  is pushed with `git push origin main costate-prereg-v1`.
- Every post-tag stage runs in the main checkout, where tag.json and decisions.json are
  committed. There is no tag on costate-dev, and no merge after the tag.
- The fixes change the code hash, and prereg binds only outputs stamped by the code it runs
  with. So, with this code on main, every pre-tag stage is re-run before prereg, in this order:
  inputs, K2, inputs again (binding t_H), K2b, K0, K1, K3, K3b, K4, K5, K6, the nulls.
- The v2.2 smoke run is evidence only. Nothing it wrote is bound.
- *Since v2.4* (Section 15.11): what this re-run must reproduce, gate by gate.
- *Since v2.5* (Sections 15.11 and 15.14): the stages decoy-k and c2-null run after K4 (both
  need inputs and K2b), and prereg binds their outputs.

**The rehearsal** (tests/test_costate_rehearsal.py). It runs every stage of scripts/costate.py
through its `main()`, in order, on random-init stand-ins for the trained networks, before the
tag.
- **What it exercises.**
  - A scratch clone of HEAD whose origin is a local bare repository, so the freeze check's tag,
    push, ancestor, pin and committed-file rules run for real.
  - Stand-in runs: each network's real config.json, eval_rdev.json and metrics.csv, and a
    checkpoint in the same pickle format with the real config, obs_rms, ret_rms and timesteps.
    Its params are random-init, on exactly the real trees.
  - Reduced lists patched into every stage's process, so every output is stamped as a full run
    and prereg binds it. The keys are 15 A2 keys on 10 geometries: a horizon twin pair, a 5-key
    geometry, the κ set, braking and non-braking keys and one multi-basin key. The networks are
    G1, two other generalists (one without step 2's overlap) and S1.
  - The real T's power floor and K2b exclusions replace the reduced T's. K4's and K5's verdicts
    are forced to pass, with the computed ones kept beside them.
- **The sequence:**
  1. inputs, K2, inputs, K2b, K0, K1, K3, K3b, K4, K5, K6, the nulls and prereg;
  2. the record committed, annotated-tagged and pushed, and a pre-tag stage refused;
  3. activations, tag.json committed, then critic, own and probes;
  4. a non-confirmatory analyse (coverage only), then analyse --confirmatory, with its decisions
     committed;
  5. a second confirmatory analyse, which must refuse, and a non-confirmatory analyse unlocked
     by the committed decisions.

  It fails on any contained error, and on any decision stated on a non-finite CI.
- **Result.** It took 47 minutes on 2 threads beside the smoke run (probes 24, the pre-tag
  stages 20). Every stage exited 0, with no contained error and no decision on a non-finite CI.
- **What it found.** Its first pass found the wrong-data paths recorded above: C1-pattern
  averaging a NaN component, C1-limit's zero-filled excess and NaN CI, C1's null R²_c of 0 for a
  dropped component, and C3-w's point-0 null. It also found an absolute local path in the
  record's overlap file. All are fixed.
- **What it cannot exercise.** Trained weights. A random-init policy crashes within about 20
  steps, so no own run is valid: C1-own, its held form and value accuracy are 'not testable', and
  K3b's trained analogue fails. tests/test_costate_stages.py scores them on planted own arrays.
  Nor the full T, the 34 networks and the 9 decision networks (R18's variants see one seed),
  their run time and memory, or K4's and K5's real verdicts.
- *Since v2.4* (Section 15.11): K4's and K5's real verdicts at full size are recorded there. The
  rehearsal was not re-run for v2.4.
- *Since v2.5* (Section 15.11): the rehearsal was re-run at 6d901a4 and again at v2.5's code.
  It still forces K4's verdict (and now decoy-k's) and K5's, so it shows crash safety only.
- **Re-run.** `COSTATE_REHEARSAL=1 uv run --no-sync pytest tests/test_costate_rehearsal.py -p
  no:cacheprovider`. Or run `uv run --no-sync python tests/test_costate_rehearsal.py --dir DIR`,
  which keeps DIR; `--from STEP` resumes there. Its report (report.json) lists each step's time.

**Cost** (informational).
- **The nulls** (before the tag) take about 327 s per network on the shared Mac with 2 threads,
  so about 50 minutes for the 9 decision networks. value_mlp's fits take 320 s; value_ridge
  0.8 s; value_ridge_window 0.8 s; QSS 4.4 s; the barrier 0.3 s; far_geometry 1.1 s. The per-fold
  pullbacks add about 3 s per network.
- *Since v2.4* (Section 15.9): the value fits' nested fits add about a fifth to the nulls stage.
- **K3 after the tag** was measured on random-init stand-ins at the real D (477 states). It took
  1.2 s for the first network (JIT compile), then 0.04–0.06 s per network: about 3–4 s for all 34
  in the critic stage. On those stand-ins both extractors matched Flax to at most 2·10⁻¹⁵, and
  the float32 gap was 7.4·10⁻⁷ with |V| up to 1.09.

**Findings and resolutions.** "Code" names the change, "rule" a rule in this section, and
"deferred" an entry in the deferred list.

| # | Finding | Resolution |
|---|---|---|
| 0 | The accel-held pullback, a pre-registered sensitivity, was never computed | Code: `analyse_network`'s 'pullback:held' and 'C1-own_held' (scripts/costate.py). Rule: "The accel-held pullback" |
| 1 | C1's Spearman rule is the within-direction mean for n and ξ, not recorded | Rule: "C1's Spearman rule" |
| 2 | Named secondaries missing and not deferred (C3-R²c, C1-heads beyond 9 networks, W1, N, the other networks) | Code: `analysis.c3_r2c`, `c3r2c_of`, `describe_network` (scripts/costate.py). Deferred: items 6–9. Rule: "C3-w and C3-R²c", "Networks outside the decision set" |
| 3 | C1-limit's slip contacts, C1-own's winsorisation, C1-pattern's pedal mask | Rule: "Secondaries as coded" (C1-pattern unchanged) |
| 4 | Deferred analyses still pinned and labelled pre-registered | Code: `prereg.DEFERRED` and the record's 'deferred'; the `analysis` docstring. Rule: "Labels", "The record and the freeze" |
| 5 | No per-component observability thresholds | Rule: "Observability" (none, with the residuals) |
| 6 | Any ValueError made a primary C2 or C2-sw 'not testable' | Code: `probes.NotTestable`, `probes_of` (scripts/costate.py), `analysis.probe_cv_problem`. Rule: "C2 and C2-sw 'not testable'" |
| 7 | C3-w lacks Section 4's weight, partition and sub-score sensitivities | Rule: "C3-w and C3-R²c" |
| 8 | Descriptives the design says are reported are not computed | Rule: "Not in the confirmatory run" |
| 9 | `stimulus_design_hash` crashed in prereg and every post-tag freeze check | Code: `prereg.stimulus_design_hash` (`stimuli.load_design`), with a test on step 1's stimuli |
| 10 | One-geometry strata scored against zero-filled, uncalibrated predictions | Code: `analysis.c1` ('n_uncalibrated', 'reason'), `c1_pattern`, `c1_limit`. Rule: "Rows without an out-of-fold calibration" |
| 11 | The confirmatory analysis ran without the own stage | Code: `confirmatory_input_problems`, `analyse_network` (scripts/costate.py). Rule: "Confirmatory inputs" |
| 12 | The documented tag and post-tag procedure was on costate-dev | Code: scripts/costate.py's docstring and prereg's message. Rule: "Where it runs" |
| 13 | K3 accepted a crashed subprocess and stale reports | Code: `stage_k3`, `_subprocess` (scripts/costate.py) |
| 14 | C1-own and value accuracy could go undecided for good | As #11 |
| 15 | `_partial` missed options that change a bound output | Code: `_partial`, k0's default (scripts/costate.py), `own.K3B_STEPS`. Rule: "Partial runs" |
| 16 | The overlap partial's weights summed to 26/31 | Code: `analysis._partial_c3`. Rule: "C3-w and C3-R²c" |
| 17 | The nulls were stamp-checked against the current run; environments were checked only at the end | Code: `confirmatory_input_problems`, `bound_env_problems` (scripts/costate.py). Rule: "The record and the freeze" |
| 18 | K3's after-the-tag checks were not implemented | Code: `gates.k3_trained`, `gates.k3_trained_judge`, `critic.net_apply_gap`, `critic.float32_gap`, `probes.extractor_gap`, `analysis.k3_trained_failed`, `k3_trained` and `stage_critic` (scripts/costate.py). Rule: "K3 after the tag" |
| 19 | scripts/fields.py is outside the code hash, and the step tags were not checked against the record | Code: `step_tags_problems`, `frozen_steps_problems` (scripts/costate.py). Rule: "The record and the freeze" |
| 20 | The fold-dependent C1 nulls let the held-out fold's targets into the calibration | Code: `nulls.far_lambda`, `nulls.fold_gradients`, `analysis.calibrate` and `FOLD_NULLS`, `prereg._null_shapes_ok`, `stage_nulls`. Rule: "C1's nulls per held-out fold" |

## 15. Gate rules after the full-size pre-tag run (v2.4; before the tag, no trained network computed)

23 Oct 2026. The full-size smoke run of the pre-tag stages at v2.2 (22 Oct, code 185c637;
Section 14's evidence) failed K4 (PR-1b and PR-6) and K5. K1 passed only on a rule that Section
7 does not state. K0, K2, K2b, K3 and K3b passed, and K6 counted. Each failure was diagnosed,
and each diagnosis was verified adversarially on the same data (four items: K1, K4's PR-1b,
K4's PR-6 and K5). Eight commits on costate-dev (d1ff055 to 428f51f) apply the results and the
lead's later decisions. Three further full-size checks of the gates followed, and the third, at
428f51f, passes every gate.

This section states each rule as the code at 428f51f applies it. It also gives the history in
rounds, the third check's numbers and what the canonical re-run must reproduce. The code's
docstrings call these the v2.3 decisions before the tag; this section is their design record.
Two later changes touch the code. 6d901a4 caps the over-smoothed fault's k at the cross-fit's
smallest training set (15.6), which does not bind on the full T. v2.5 (15.14) adds the
pre-tag stages decoy-k and c2-null, binds the inputs K4 reads, restricts K4's band notes to C2
and corrects 15.10's noise floor. Neither changes a gate rule or a number of the third check.
**The rules are final from this version.** A gate that fails the canonical re-run is reported
and blocks the tag. No rule, seed, grid, candidate or tolerance is chosen again.

**Setting of every number below**, unless stated otherwise.
- The smoke run's world: T has 9,887 states, 477 keys and 48 geometries, and the interior
  3,951 states. Partition 0, key weights, G1's obs_rms and config.
- Random-init actors of seeds 0–5. The plants' host is seed 0. K4's controls are seeds 1–3
  (`gates.K4_N_CONTROLS`); C2's primary statistic inside K4 uses seeds 1–5 as its 5 controls.
- C2's decision rows are the interior rows valid under K2b's exclusions and G1's
  observability: s 1,643 and ξ 1,581, none for n or vx.
- Only key files, the input side and random-init or planted networks enter. Component lists
  are in the order (s, n, ξ, vx).

### 15.1 The rounds

**Round 0: the smoke run** (22 Oct; v2.2).
- K1 passed with its plants, on the coded rule: zero comparisons pooled as 0, a floor ε_t, and
  a noise term in the kink test. Section 7 states none of these.
- K4 failed on PR-1b (partial R² 0.698–0.751 against 0.80 ± 0.03) and PR-6 (Δ2 −0.0118
  against "within 0.01 of 0"). PR-1a, PR-2, PR-3, PR-4, PR-5 and PR-7 to PR-9 passed.
- K5's wiring check failed: λ_s's and λ_n's CI upper ends were 0.0004 and 0.0032, and λ_ξ
  scored −0.70.

**Round 1: diagnosis, verification and the first fixes** (23 Oct; d1ff055 to 3e2242f).
- One agent diagnosed each failure and an adversarial verifier checked the diagnosis.
  - K1: the verdict held. The verifier refuted the diagnosis's "the literal rule fails δ":
    Section 7's literal kink test makes every component descriptive (15.2). It also found the
    floor's development history (15.2, "Disclosure").
  - PR-1b: held (15.4).
  - PR-6: did not hold. The verifier found that f̂_D at k = 15 leaks in a kNN-shaped
    known-truth world. Gating on the smooth world alone would have kept the world that passes
    (15.5–15.7).
  - K5: held, with corrections: the null calibration's scheme, a false-fail rate of about 1–2%
    per run, two blind spots and a new planted fault (15.3).
- Commits: K1 (d1ff055); K5 (6fde947); f̂_D's k by nested grouped CV, the verifier's
  target-side proposal and the lead's first decision on f̂_D (bac63d0); K4's PR-1b, PR-6,
  PR-6b and PR-6c (778c06e); the value-fit nulls' full cross-fit and the zero-fill rule
  (3e2242f; 15.9).
- Measured after them: K1, K5 and PR-1b pass, re-scored offline or at full size. bac63d0
  moved λ_n's interior residual share from 0.107 to 0.064, below the power floor.

**Round 2: the first full-size check** (3e2242f; K4 also at 778c06e, with the same numbers).
- K1, K2, K5 and every K4 plant but one passed. PR-6b failed on gate seed 0 at 0.0165 (CI
  0.0039–0.0261), with 0.0095 and 0.0026 on seeds 1 and 2. The nested-CV rule took k = 15–30
  for s and 30 for ξ in every fold on that smooth y.
- A descriptive diagnostic on the same draws was known when the lead decided: a fixed k = 15
  gives 0.0032, 0.0040 and 0.0059, and k = 5 gives 0.0010, −0.0013 and 0.0060.
- **The lead's second decision**, on f̂_D (55b84c4). Grounds: what leaks into C2 is f̂_D's
  bias, which a network holding the decoy function recovers; its variance only costs power. An
  out-of-geometry SSE trades the two, so it is the wrong criterion. The decision: one fixed k,
  chosen by a rule declared before any result, on 6 validation seeds disjoint from K4's gate
  seeds (15.7). The rule chose k = 8.
- π under the λ components' zero rule (b8001d6): the lead's decision, for consistency. The
  K1 re-score had already shown π passing under either reading (15.2).

**Round 3: the second full-size check** (55b84c4).
- Every gate passed except K4's PR-3: its grouped Δ2 was 0.0108 against 0.01.
- Traced by a diagnostic on that run's world: the random host alone scores +0.0069 against its
  3 controls, and the codes add +0.0038. At k = 15 the same host scores −0.0001 and host +
  codes −0.0062. The failure was the host's chance offset, which the change of f̂_D's k had
  re-drawn.
- **The lead's third decision** (428f51f): every K4 plant whose rule is a no-gain tolerance is
  scored by its paired increment (15.8). Grounds: a plant's effect is the change it causes in
  the same network. Tolerances, seeds and every-seed rules are unchanged. The k rule was
  re-applied mechanically under the paired statistic and still gives k = 8.
- Disclosure: this rule came after the PR-3 failure was seen. The verified write-ups cover
  round 1's four items only. The decisions of rounds 2 and 3 rest on the grounds stated here
  and on their tests.

**Round 4: the third full-size check** (428f51f). Every gate passes (15.11). Only K4 differs
from the second check: K1, K2, K2b, K5, the nulls' npz and every frozen value are identical
apart from provenance.

**After round 4.** 6d901a4 capped the over-smoothed fault's k, after the rehearsal crashed on
it (15.6, 15.11). A review of v2.4 confirmed 10 findings, one a blocker: 15.10's noise floor
was about 4 times too small in SD (0.028 measured). v2.5 resolves them (15.14), with a fourth
full-size run of the stages its code changes touch (15.11).

**Before and after each decision.**

| Decision | Before | After |
|---|---|---|
| K1's rule written down (d1ff055) | δ with zeros pooled: n 235, median 4.8·10⁻⁶, p95 2.5·10⁻³ | n 184 (51 zero), median 2.2·10⁻⁵, p95 2.8·10⁻³ |
| π's zero rule (b8001d6) | n 20, median 1.35·10⁻⁷, p95 6.5·10⁻⁶ | n 12 (8 zero), median 8.9·10⁻⁷, p95 7.0·10⁻⁶ |
| K5's scheme and rule (6fde947) | upper ends s +0.0004, n +0.0032; ξ −0.70 | points −0.0017, −0.0028, −0.0013, −0.0012; passes |
| PR-1b on one array (778c06e) | partial 0.698–0.751 | 0.795–0.812 (nested-CV k), 0.786–0.803 (k = 8) |
| PR-6 one-sided (778c06e) | −0.0118, two-sided rule | −0.0097 (nested-CV k), −0.0013 (k = 8, unpaired), −0.0083 (paired) |
| Nested-CV k (bac63d0) | k = 15: λ_n's interior share 0.107 | 0.064; PR-6b fails at 0.0165 |
| One fixed k = 8 (55b84c4) | PR-6b 0.0165, 0.0095, 0.0026 | 0.0028, 0.0007, 0.0052 (unpaired) |
| Paired increments (428f51f) | PR-3 grouped 0.0108 | 0.0037 |

### 15.2 K1 (with π)

**Smoke run.** K1 passed with its plants (50 keys, 2,008 comparisons, 20 π comparisons) on a
rule that Section 7 does not state; Section 7's 'Pass' and 'Kinks' are its only statement.
Read with Gate 1's 10⁻⁶ zero threshold, δ fails: median 5.4·10⁻⁵ and p95 0.037, zero
comparisons pooled as 0.

**Cause.**
- At δ's step h = 10⁻⁶, a δ costate of 10⁻⁶ is a J difference of about 10⁻¹², which K1's warm
  re-solves cannot resolve. On a full-throttle straight λ_δ is that small, and the coverage
  rule puts 50 pedal-saturated states among δ's comparisons.
- The 24 δ comparisons above 10⁻² are on 24 keys, 18 of them at pedal-saturated states with
  the fold active. Their |λ_δ| runs from 9·10⁻⁷ to 2.7·10⁻⁵ and their absolute errors from
  5·10⁻⁸ to 3.5·10⁻⁶, all below the coded floor. No comparison with |λ_δ| ≥ 3·10⁻⁵ exceeds
  9·10⁻³.
- At the worst one (key cb10d183, t = 70), J(c) is smooth with a biased slope. K1's difference
  is 0.71–1.06·10⁻⁶ at c = 10⁻⁷ to 10⁻⁴, against 4.29·10⁻⁶ from a base re-converged at tol
  10⁻¹² (4.22–4.27·10⁻⁶ at three steps). The warm re-solves settle along nearly flat
  directions, by about 5·10⁻¹² in J(c) − J(−c).
- Section 7's kink test cannot be passed under re-solve scatter. The one-sided estimates
  contain the unshifted J(0) with weight 6/h, and the central pair does not.

**The rule** (`gates.k1_error`, `k1_decision`, `targets.FD`, `fd_costate_k1`, `fd_pi`; the
record's constants targets.k1_rule, kink_j_noise, shrink_rtol, max_shrink and pi_h).
- Steps: 10⁻⁴ for s, n, vx and vy; 10⁻⁶ for ξ, r and δ; 10⁻⁵ for pedal, with the one-sided
  ladder (10⁻³, 5·10⁻⁴, 2.5·10⁻⁴) at pedal-saturated states (Section 13). Richardson over (c,
  c/2), re-solves at tol 10⁻¹⁰.
- The difference's noise term is 6ε_J/(h·γ^t), with ε_J = `KINK_J_NOISE` = 10⁻¹¹, J's typical
  spread between warm re-solves (not a bound). On the pedal ladder, ε_J is the rungs' own
  spread where it exceeds 10⁻¹¹.
- The floor is ε_t = max(10⁻⁶, 6ε_J/(h·γ^t)): 10⁻⁶ to 1.6·10⁻⁶ at the 10⁻⁴ steps, 6·10⁻⁶/γ^t
  at pedal's, and 6·10⁻⁵/γ^t at the 10⁻⁶ steps (6.5·10⁻⁵ to 1.2·10⁻⁴ on K1's states, at most
  1.6·10⁻⁴ on T). It is twice the largest error that J errors of size ε_J at the four stencil
  points can give the central value (3ε_J/(h·γ^t)). Under independent scatter of that size the
  error's p95 is about 3.7ε_J/(h·γ^t).
- Error: err = |λ − FD| / max(|FD|, 3·disc, ε_t).
- **Zero comparison:** |λ| and |FD| both below ε_t. It is counted ('n_zero', per component,
  per plant and per reading) and never pooled (Gate 1's zero-field rule).
- **Kink:** |forward − backward| > 10·max(disc, 6ε_J/(h·γ^t)). A kink does not count.
- **Smaller c:** while the comparison is not a kink and its Richardson pair disagrees by more
  than max(10⁻³·|FD|, 6ε_J/(h·γ^t)), c is divided by 10, at most twice. A comparison still
  unresolved does not count.
- **Pass per component:** over the non-zero comparisons that count, median < 10⁻³ and p95 <
  10⁻². A component is descriptive when kinks and unresolved comparisons exceed 30% of its
  converged ones, or when no non-zero comparison counts. K1 needs at least one decided
  component, and every decided one must pass.
- **π** (10 anchored keys, j = 0 and 1): `fd_pi` at h = `PI_H` = 10⁻⁵. It is the same central
  stencil of four re-solves, without J(0) and without a kink test. J errors reach J_tail
  divided by γ, so π's floor is max(10⁻⁶, 6ε_J/(h·γ)) = 6.03·10⁻⁶. The same zero rule applies
  to π and to each π plant's comparison, and π needs a non-zero comparison.
- **Plants** must fail the same rule on the same re-solves, none vacuously. Each required plant
  must be scored on non-zero counted comparisons of a decided component and fail there; each π
  plant must fail on its non-zero comparisons. The record binds K1 on `pass_with_plants`.
- **Reported, binding nothing:** 'design_rule' (Gate 1's 10⁻⁶ with zero comparisons counted,
  unresolved comparisons scored and the coded kink flag); 'design_rule_literal_kink' (the same
  with Section 7's 10·disc kink test, from the one-sided estimates every row now stores); and
  π's 'design_rule'.

**Disclosure.** The floor was coded at the first implementation (c7d2285, 22 Oct 16:44),
before K1's full-size run (23:13–23:31). It was coded after δ's comparison on the first sampled
key (0005ec83, t = 72: target −9.7·10⁻⁷, FD −1.16·10⁻⁶, err 0.165 on the relative rule) had
been seen to fail in development. That comparison is the floor's unit test
(`test_k1_rules_with_known_answers`). The K1 sample is deterministic, so the floor was coded
after a development row from this very sample had been seen. It rests here only on grounds
that do not depend on K1's outcome:
- ε_J is a property of the instrument (the J spread between warm re-solves), not of the
  targets. At cb10d183, t = 70, the J(0) offsets are 2–8·10⁻¹² and the smooth-fit residuals up
  to 6·10⁻¹¹.
- The verdict is not knife-edge in ε_J. δ passes for any ε_J ≥ 10⁻¹² (p95 8.0·10⁻³ there) and
  fails at 3·10⁻¹³ or less; r, ξ, n and vy pass for every ε_J down to 0.
- The kink term only removes flags, and counting zero comparisons instead of pooling them
  tightens the rule: δ's median rises from 4.8·10⁻⁶ to 2.2·10⁻⁵.
- Every planted bug still fails (below).

**Known truth.**
- A synthetic J(c) = γ^t(λc + qc²) + noise through K1's stencil (20,000 draws, h = 10⁻⁶, γ^t =
  0.7): Section 7's 10·disc flags 30.8–31.0% of smooth comparisons for independent scatter of
  10⁻¹⁴ to 10⁻¹¹, and 83.9–99.9% when J(0) carries its own offset of 10⁻¹³ to 10⁻¹¹. The coded
  test flags none of them. It still flags a planted kink whose one-sided slopes differ by
  2·10⁻³, but not one at 6·10⁻⁴.
- At cb10d183, t = 70, the correct target (confirmed by re-converging) scores 4.2 on the literal
  rule. Under this rule it is a zero comparison.
- On the real re-solves the literal kink test flags 40–49% of every component's comparisons
  (third check: s 0.482, n 0.458, ξ 0.486, vx 0.478, vy 0.442, r 0.398, δ 0.466, pedal 0.442).
  Under Section 7's literal reading every component is therefore descriptive, and K1 cannot
  pass with any targets. (The smoke run's rows stored no one-sided estimates; the verifier
  recomputed 1,423 of them, and the third check scores all 2,008.)

**Planted faults that still fail** (third check; medians over the non-zero comparisons that
count).
- flip_sign 2.0 on all 8 components; no_discount 0.161–0.384; off_by_one on all 8 (n 0.0088 up
  to δ 1.94); j_w on the 30 anchored decisions, 0.005 = 1 − γ, except s (4.60); no_es on s
  (3.64); pre_jump on n (0.44), vx, vy, r and δ; unfolded on pedal (0.015 over 63 fold-active
  rows).
- π's plants, each scored on all 20 comparisons: j_w 1.0, plus_shift 0.934, r0_prev 1.0.
- On K1's own comparisons (on the smoke run's rows, the recorded difference as truth): a gain
  error (1 + ε)·λ_k at ε = 1.0–1.1·10⁻³ on every component, and an offset of 1.6·10⁻⁶ on λ_δ.
- Passing by construction: no_es on every component but s, since it differs from λ only by
  e_s/10 (if s were descriptive, no_es could not fail and K1 would fail); pre_jump on s, ξ and
  pedal, which lie outside the active rows' gradients.

**What the floor can hide.**
- ε_t is at most 7.6·10⁻⁴ SD_T(λ_δ), 1.2·10⁻³ SD_T(λ_r), 2.8·10⁻⁴ SD_T(λ_pedal), and 3·10⁻⁵
  SD_T or less for s, n, ξ, vx and vy.
- Below it lie at most 0.2% of each K_P component's interior rows, 0.1% of vy's, 4.6% of
  pedal's, 14% of r's and 43% of δ's. On those rows λ is checked in absolute terms only, to
  about 3.5·10⁻⁶.
- The targets carry the same inexactness as the differences. Re-converging 16 of K1's keys
  moves T's targets by at most 1.3·10⁻⁵ (ξ), at most 5·10⁻⁶ SD_T for every component. K1
  verifies the targets to about 10⁻⁵ absolute.

**Consequence.** On K_P both readings give the same verdict and nearly the same numbers (p95
≤ 3.5·10⁻³ under both), so no C1, C2 or C2-sw decision depends on the choice. C1-pattern's
non-K_P side includes r and δ. Their rows below the floor are checked to about 3.5·10⁻⁶
(at most 1.2·10⁻³ SD_T), which can move an R² on standardised targets by about 10⁻⁶ at most.

### 15.3 K5's wiring check

**Smoke run.** It failed. R²_wd was −0.0014, +0.0017, −0.698 and −0.0023, with CI upper ends
+0.0004, +0.0032, −0.602 and −0.0004, against "the R² CI must lie at or below 0".

**Cause.** There was no leak, and nothing wrong in the probe pipeline that C1 and C2 use. An
independent loop over the folds reproduces `ridge_cv`'s predictions exactly, and no held-out
row enters a scaling, a penalty choice or a fit. The check had three problems.
- **The rule.** "CI at or below 0" is an equivalence test with zero margin. Under correct
  wiring the expected R²_wd is that of the unweighted leave-fold-out training mean, −0.002 to
  −0.001 (the permutation moves targets between rows of unequal weight). One draw adds a
  chance term that the geometry bootstrap holds fixed, and the CI half-widths are about 0.002
  (0.018 for ξ). The rule failed 30 of 30 scratch draws of the old scheme and 39 of 40 single
  draws of the new one; the exact no-information predictor cleared it by 3·10⁻⁵ on λ_n.
- **Permuting within each training fold** kept each fold's mean, which carries an eighth of
  each of its geometries' own mean: real signal that transfers. For λ_s it added +0.002 over a
  pooled permutation (3.9 SE; the upper end was above 0 in 30 of 30 draws, against 13 of 30
  pooled).
- **Permuting across turn directions** held λ_ξ's R²_wd at −between/within = −0.41/0.59 ≈
  −0.70 whatever the wiring, so a fault in ξ alone still scored −0.31.

**The rule** (`probes.permuted_targets`, `wiring_null`; the record's probes.wiring = [10, 0.01]
and probes.wiring_rule).
- While outer fold f is held out, the training targets (every row outside f) are permuted
  state by state across all training folds, never within one. λ_n and λ_ξ are permuted within
  each turn direction (stream `costate_perm`, sub-stream (draw, f, 1 + j)); λ_s and λ_vx across
  both (sub-stream (draw, f, 0)). Held-out rows keep their own targets. The direction is
  required.
- `WIRING_DRAWS` = 10 draws. Their out-of-fold predictions are averaged and scored by R²_wd
  against the unpermuted held-out λ (raw λ, not λ_res), with the cluster-bootstrap percentile
  CI on K5's own counts (k4_rng(12)).
- Every K_P component needs a CI lower end ≤ 0 and a point ≤ `WIRING_TOL` = 0.01, K4's zero
  tolerance. The per-draw points are reported. The control task stays descriptive.
- The scheme, the 10 draws and the tolerance on the point were chosen after the smoke run's
  failure, from null and planted runs, and before the corrected run on the real stream. They
  are frozen. A tolerance on the upper end is infeasible for ξ, whose averaged upper end is
  about +0.014 from geometry heterogeneity, not from the wiring.

**Known truth and the false-fail rate.**
- The 40 null draws behind the calibration permuted every column within direction. The
  adopted scheme pools s and vx across directions, which roughly doubles vx's single-draw null
  SD (from about 0.0018 to 0.0035).
- Two fresh 10-draw replicates of the adopted scheme both passed. The closest was λ_n, with a
  point of +0.0004 and a lower end of −0.0005. A Gaussian estimate from 30–60 single draws puts
  **the false-fail rate at about 1–2% per run**, from the n and vx lower ends. A single draw can
  break both bounds with no leak (one draw gave n's point 0.0098 with lower end +0.006), so the
  average is required.
- The probes test world (3,840 rows, base z): without a leak the points are −0.0101 to
  −0.0009 and the check passes. Its s and vx upper ends are +0.0003, so the old rule would have
  failed there.

**Planted faults that still fail** (the diagnosis's and verifier's runs on the cached
full-size K5 world; most under a scheme that permuted every column within direction, the
first re-run with the adopted code):
- one outer fold's model fitted on unpermuted targets: points 0.198–0.286, lower ends ≥ 0.039;
- one training fold left unpermuted in every model: 0.17–0.25; the held-out rows in the fit:
  0.21–0.29;
- 5% of keys unpermuted: all four components fail; 2%: n, ξ and vx; 1% (one draw): n and ξ;
- one held-out geometry per fold in the fit: n (lower end 0.0069), ξ (point 0.018) and vx
  (lower end 0.0023);
- ξ's column unpermuted for one model: 0.204, lower end 0.050;
- the decoy fit outside the permutation: 0.84–0.97;
- test-set tuning (penalties chosen by held-out SSE against the real targets): points 0.0126,
  0.0138, 0.0236 and 0.0146, lower ends 0.0073, 0.0109, −0.0088 and 0.0114, so every component
  fails;
- in the unit tests, on the exact code: fold 0's model unpermuted, λ_n unpermuted for one model
  (n alone fails), and test-set tuning.

**Not seen** (stated limitations). A leak confined to the intercept or the fold means passes
(averaged points ≤ 0.0002). It is bounded by the between-fold share of the target's variance,
at most 0.002 for λ_s and about 0 for the others. A single leaked geometry sits near the floor
(λ_n's lower end +0.0004 in one single-draw run). `Folds.check`, `ridge_cv`'s training masks
and PR-3 guard both.

**Consequence.** None for C1, C2 or C2-sw: only K5 uses `permuted_targets`. C2 scores the
cross-fitted λ_res, whose D₁ contains σ, and C1's calibration has direction intercepts.

### 15.4 K4's PR-1b

**Smoke run.** It failed. The partial R² was 0.751, 0.744, 0.698 and 0.713 against 0.80 ±
0.03, and Δ2 was 0.502, 0.638, 0.641 and 0.603, between 0.046 and 0.083 short of (1 − R²_b)·0.8.

**Cause.** The plant and the probe used different target arrays. The plant's r came from each
row's own-fold λ_res. While fold f is held out, C2's probe trains on λ − f̂_D^(−f,−g), which
changes with f. One planted column cannot follow it, so on training rows it carried d =
f̂_D^(−g) − f̂_D^(−f,−g), the held-out fold's share of the decoy fit. Var d/Var r was 0.24–0.48
pooled and up to 1.4 in single folds, and the per-fold shortfall tracked it (Spearman 0.70 over
24 fold × component cells). Every other candidate explains about 0.03 at most: ridge
shrinkage, the penalty grid, one component at a time, the scale α, winsorisation and
finite-sample loss. Removing h3 recovers up to 0.039 on s and n, but through d.

**The rule** (`gates.k4`; the record's gates.k4_pr1b_rule).
- One target array for the plant's baseline, the plant, the probe and the random controls:
  each row's own-fold λ_res (`Residuals.oof`). SNR 4. The unit is u = α(r̃ + η), with Var η =
  1/SNR and α the median h3 unit SD. All four K_P components are planted at once; one at a time
  differs by at most about 0.008 (at k = 15).
- The pass rule is unchanged: per component, the partial R² within 0.03 of 0.8, and Δ2 within
  0.03 of (1 − R²_b)·0.8.
- A limitation. That array's f̂_D for a training row was fitted with the held-out fold's
  geometries, so PR-1b checks the ridge machinery (both penalties, inner CV, scaling), not the
  cross-fit. At k = 15 it over-recovered by at most 0.022 over three plant seeds. The cross-fit
  is covered instead by a `ridge_cv` unit test (it trains and scores fold f on slice f of
  by-fold targets) and by the fold-consistent plant on C2's own by-fold targets
  (`pr1b_by_fold`), reported beside PR-1b as descriptive.

**Known truth.** With C2's by-fold targets kept and the plant made to follow them on training
rows, recovery is 0.78–0.80 at k = 15, so the shortfall came from the plant's construction. A
unit independent of the decoy variables is recovered at 1.01–1.21 of SNR/(1 + SNR) under the
by-fold targets, as with one array or with no residualisation.

**Planted faults that still fail** (at k = 15, not re-measured at k = 8): a hidden penalty
floored at 1 (partial 0.46–0.55) or at 0.1 (0.76–0.79, with Δ2 on s 0.546 against 0.589); the
network's rows permuted within geometry (−0.14 to −0.07). Not detected: a single shared penalty
over [z, m, h3, u] (0.802–0.813), which is not a recovery fault; penalties chosen on the
held-out fold fail on ξ only.

**Third check** (k = 8). Partial 0.786, 0.791, 0.788 and 0.803. Δ2 0.718, 0.770, 0.826 and
0.748 against 0.738, 0.781, 0.842 and 0.738, the largest gap 0.020 on s. R²_b 0.077, 0.024,
−0.052 and 0.078. The by-fold plant lies within the tolerance (partial 0.780–0.799). PR-1b
fails at k = 3 and 5 (λ_s's Δ2 is short by 0.037–0.038); that played no part in the choice of
k (15.7).

**Consequence.** C2 keeps the by-fold targets. A real h3 is a function of z and cannot carry
d. A layer that encodes the decoy variables is not neutral (15.5).

### 15.5 K4's PR-6

**Smoke run.** It failed: Δ2 was −0.0118 against "within 0.01 of 0". The bisection ran to gain
4.47, where the units' probe-alone R² was 0.364 against the decoy kNN's 0.798.

**Cause.** The probe pipeline is not at fault.
- Units that carry nothing beyond z, m and h3 lower the out-of-fold R², because coefficients
  fitted on 40 training geometries do not transfer. On T, 32 noise units cost 0.004, PR-6's
  units 0.011 and 32 random geometry codes 0.026. On a known-truth null (y = G(z) + an AR(1)
  ε + a per-geometry effect), PR-6's units cost −0.006 to −0.007 and biased decoy units −0.005
  to −0.016. The null expectation is negative, not 0.
- Without a bias every unit is an odd function of the centred decoys, and a ridge on them
  peaks at R² 0.36–0.37. With biases N(0, 1), 32 units peak at 0.50 (gain 2). The kNN's 0.80 is
  above every reachable value, so any search runs to the grid's edge.
- On λ the units are redundant with z, m and a random h3: with the residualisation off they
  add only 0.005 (0.008 with biases). PR-6 alone has no power to fail a broken residualiser.

**The rule** (`gates.pr6`, `probes.decoy_units`; the record's gates.k4_pr6 and k4_pr6_rule).
- 32 units tanh(gain·(D̃A + b)) of the standardised (s − s_apex, vx, σ, log R), with A ~ N(0,
  0.5²) and b ~ N(0, 1) (`K4_PR6_BIAS` = 1), appended to the host's h3.
- The gain is the one from the fixed grid (0.25, 0.5, 1, 2, 4, 8) whose probe-alone R² (mean
  over K_P) lies closest to the decoy kNN's. 'calibrated' (|difference| ≤ 0.02) is
  descriptive.
- Pass: the paired Δ2 (15.8) ≤ 0.01, one-sided, on C2's by-fold λ_res, on all of T and K_P,
  with K4's 3 controls; and PR-1b passes. The negative part is reported as the plant's
  over-fitting cost ('overfit_cost'), and the unresidualised Δ2 beside it.
- Grounds for one-sided: the gate exists to catch a spurious positive gain, PR-3 was already
  one-sided, and the known-truth null is negative.
- PR-6 stays on the by-fold λ_res. One array would hide the C2 behaviour PR-6 measures: at k
  = 15 the units scored −0.012 by fold, −0.007 on one array and about −0.001 with their rows
  permuted.

**Third check.** Paired Δ2 −0.0083 (CI −0.0328 to 0.0114); by component −0.0079, −0.0156,
−0.0129 and 0.0031; cost 0.0083; unpaired −0.0013. Gain 2: the decoy units' R² 0.501 against
the kNN's 0.794 (0.419–0.501 over the grid), so calibrated = False. With the residualisation
off: paired 0.00999 (CI 0.0068–0.0152), unpaired 0.0080.

**Planted faults.** On real λ, PR-6 cannot fail a broken residualiser. The residualisation off
(paired 0.00999, just under 0.01), D₁ without s − s_apex (−0.003 at k = 15) and a partial
residualisation (c = 0.8 or 0.9: −0.010) would all pass. The decoy-leak sensitivity therefore
rests on PR-6b and PR-6c. Units holding a kNN of λ fitted on every other geometry (+0.240)
carry λ information from held-out geometries; they are not a decoy fault and are not cited as
one.

**Consequence for C2.** A layer whose decoy-structured units its random controls lack loses
about 0.01 of Δ2 (0.004–0.017 over seeds on T, 0.011–0.018 on the interior). C2 is therefore
conservative for a confirm, and a refute near the band is slightly easier; hence the band note
(15.10).

### 15.6 K4's PR-6b and PR-6c (known truth)

**Smoke run.** Neither existed. Both come from the PR-6 verification: two known-truth worlds,
both required, so that the gate is not judged on the world that passes.

**Construction** (`gates.pr6_world`, `pr6b_f`, `pr6c_f`, `hold_units`; the record's
gates.k4_pr6).
- F per K_P component. PR-6b's F is smooth: the weighted ridge (α = 10⁻², weights normalised,
  intercept unpenalised) of λ on 256 biased tanh features of (s − s_apex, vx, σ, log R) (k4
  sub-stream 13), fitted on every row. PR-6c's F matches the decoy kNN (Section 7's wording):
  each geometry's rows are predicted by the k = 15 kNN of λ on D₁ fitted on every other
  geometry (`K4_PR6C_K`, the world's own k, whatever f̂_D's). λ only shapes F.
- y = std(F) + 0.5ε. ε is an AR(1) series (ρ = 0.9) along each key in t order, independent
  across keys and components and weighted-standardised, so it is 20% of Var y. The gate seeds
  are `K4_PR6_SEEDS` = (0, 1, 2) on k4 sub-stream (14, seed), one draw shared by both worlds.
- 32 units hold F: tanh(F̃A + b), with A and b ~ N(0, 0.5²) (sub-stream (15, j)), appended to
  the host's h3.
- y is residualised by the pipeline's own `states.cross_fit` at f̂_D's k (8).
- The statistic is C2's primary one (`c2_primary`, through `analysis.c2_network`): C2's
  decision stratum (the interior, 3,951 rows), each component on its decision rows (s 1,643,
  ξ 1,581), the frozen power floor's components ({s, ξ}), C2's 5 random-init controls and key
  weights renormalised on the stratum. It is the units' paired increment over the host
  (15.8).
- **Pass per world:** on every seed Δ2 ≤ 0.01; on every seed Δ2 > 0.01 with the
  residualisation turned off (y itself); and for PR-6b, on every seed, the units' probe-alone
  R² at least the kNN's (means over K_P, on T). K4 needs both worlds.
- Descriptive: the T-wide statistic (every row and component), the per-component gains, and two
  planted faults on the first seed: the kNN at k = 300, and D₁ without s − s_apex at f̂_D's k.
- The over-smoothed fault's k is min(300, the cross-fit's smallest training set: the rows
  outside two outer folds), `gates.oversmooth_k` (6d901a4). Each fault records its k. On the
  full T the smallest training set is 5,614 rows (partition 0), so the cap does not bind and the
  fault is the k = 300 kNN. It binds on the rehearsal's reduced T (283 training rows), where
  k = 300 crashed the cross-fit.

**History.**
- Under the smoke run's k = 15, the verifier measured the kNN-shaped world at 0.0093–0.0155 on
  T over 7 runs (6 above 0.01; λ_s's gain 0.036–0.055), and at 0.002, 0.018 and 0.017 on the
  interior with 5 controls (λ_s's gain up to 0.083).
- A leave-one-geometry-out residualiser still leaked 0.0118, so the leak is f̂_D's smoothing
  bias, not interpolation across held-out geometries. k = 5 removed it (0.0010 and 0.0016 on
  T, −0.002 on the interior).
- Under the nested-CV k, PR-6b failed at full size (15.1). At k = 8 both worlds pass.

**Third check** (k = 8, paired).
- PR-6b: 0.0033 (CI 0.0001–0.0065), 0.0016 and 0.0032. Leak bound 0.0033 (unpaired 0.0052);
  largest gains s 0.0063, ξ 0.0032. Off: 0.090, 0.097 and 0.070. Probe-alone R² 0.751–0.755
  against the kNN's 0.631–0.645. T-wide −0.0015 to −0.0009.
- PR-6c: 0.0069 (CI −0.0075 to 0.0217), 0.0069 (CI −0.0027 to 0.0171) and 0.0045. Leak bound
  0.0069 (unpaired 0.0070); largest gains s 0.0135, ξ 0.0004. Off: 0.177, 0.217 and 0.215.
  Probe-alone R² 0.764–0.767 against 0.678–0.685. T-wide −0.0024 to −0.0009.
- The host's own offset on these rows is −0.0010 to 0.0020 for PR-6b and below 0.0002 in
  absolute value for PR-6c.

**Planted faults.**
- Detected: the residualisation turned off, on every seed of both worlds (0.070–0.217); the kNN
  at k = 300 (PR-6b 0.032, PR-6c 0.184); skipping the residualisation (a unit test).
- **Not detected: D₁ without s − s_apex** (PR-6b 0.0014, PR-6c −0.0059 on seed 0). On the
  validation seeds at k = 8 (unpaired) it was detected on 3 of 6 PR-6b seeds and on none of
  PR-6c's. C2's decision rows score s on approach and exit interior rows and ξ on approach rows
  only, and the T-wide +0.017 and +0.020 that the verifier saw at k = 15 did not reproduce
  there. This is a stated limitation: K4 does not test that D₁ holds s − s_apex. A states test
  pins D₁'s columns on a stored key (D₁[:, 0] = s − s_apex).

**Margins.** PR-6c's leak bound is 0.0069, and its largest validation value at k = 8 is 0.0098,
both against 0.01.

### 15.7 f̂_D's k

**Smoke run.** f̂_D was the k = 15 kNN (Section 4), and no gate tested k directly. The PR-6
verification found that k = 15 over-smooths λ_s. It leaked 0.009–0.016 on T in the kNN-shaped
world (15.6). On real λ, a training-fold kNN predicts about 19–25% of λ_res,s for a held-out
geometry: the cross-fitted kNN's R² for λ_s is 0.772 at k = 5 against 0.720 at k = 15 on T,
and 0.788 against 0.717 on the interior.

**Round 1: a nested-CV k** (bac63d0; the verifier's target-side proposal).
- k per (partition, outer fold, K_P component) from (3, 5, 8, 15, 30), by the weighted
  out-of-geometry SSE over the frozen inner folds of the fold's training geometries.
- On real λ it chose 3–15 for s by fold, 5 for n in every fold, 15–30 for ξ and 15 for vx.
  λ_n's interior residual share fell from 0.107 to 0.064, below the floor.
- In the known-truth worlds it left the kNN-shaped leak at 0.0085–0.0139 on T (6 of 7 above
  0.01) and raised the smooth world to 0.0125. The SSE-optimal k trades the kNN's bias against
  noise, and a plant that holds F exactly gains on the bias. At full size PR-6b failed (15.1).

**The rule** (55b84c4, re-applied at 428f51f; `gates.decoy_k_choice`, `decoy_k_leaks`,
`at_decoy_k`, `states.KNN_K` = 8; the record's states.knn_k, knn_k_candidates, knn_k_rule and
knn_k_validation).
- Grounds (the lead's): in C2 the residualiser serves validity. What leaks into C2 is f̂_D's
  bias, which a network holding the decoy function recovers. Its variance (noise borrowed from
  other geometries) no network can recover; it only costs power, in the conservative
  direction. An out-of-geometry SSE trades the two, so it is the wrong criterion.
- One k for every fit, outer fold, partition and K_P component. `cross_fit` refuses anything
  but one positive integer, and the nested-CV code is removed.
- The rule, declared before any result: among (3, 5, 8, 15), the largest k for which every one
  of 6 validation seeds of both PR-6b and PR-6c gives C2's primary Δ2 ≤ 0.01, with y
  residualised at k and C2's power floor recomputed at k (`frozen_values`' rule, on partition
  0's own-fold residuals). If none qualifies: k = 3, as a blocker.
- **Validation seeds.** Stream `costate_decoy_k` (index 311), sub-stream v = 0…5, one noise draw
  shared by both worlds, and otherwise exactly K4's construction (`pr6_world`: the same F,
  units, decision rows, controls and statistic). K4's gate seeds are on stream `costate_k4`
  (310), sub-stream (14, s), s = 0, 1, 2, and none was computed at any candidate k for the
  choice. Before it they had been scored under the nested-CV residualiser (the first check)
  and, for PR-6b only, in that check's descriptive diagnostic at fixed k = 15 and 5 (both
  passing there). When the rule was re-applied (428f51f), both worlds' gate seeds had also
  been scored at k = 8, unpaired, in the second check (55b84c4: leak bounds 0.0052 and
  0.0070), and PR-3 at k = 15 in that check's diagnostic. The re-application read only the
  validation seeds on stream 311 and kept k = 8.
- The statistic read is the units' paired increment over the host (15.8). The rule was first
  applied to the unpaired statistic (55b84c4) and re-applied mechanically when the no-gain
  plants became paired (428f51f). Both give k = 8.
- The record freezes decoy_k = {k: 8, scope: every partition, outer fold and K_P component},
  and the post-tag C2 (`probes_of`) cross-fits every partition at the frozen k.
- **Re-derived before the tag** (v2.5; the pre-tag stage decoy-k, `gates.decoy_k`, writing
  gates/decoy_k.json, which prereg binds). It runs `decoy_k_leaks` at every candidate on the
  canonical world and applies the rule. It passes when the choice is `states.KNN_K` (not a
  blocker) and every per-seed value, paired and unpaired, is within 10⁻⁶ of
  `DECOY_K_VALIDATION` and `DECOY_K_VALIDATION_UNPAIRED` (`decoy_k_reproduction`). Without it
  the table was design-time evidence that no stage computed. With it, a canonical world that
  differs (PR-6c's margin is 0.0002) blocks prereg instead of leaving a table that describes
  another world.

**Validation table.** Δ2 per world as max / mean / seeds ≤ 0.01 of 6. The per-seed values are
`gates.DECOY_K_VALIDATION` (paired) and `DECOY_K_VALIDATION_UNPAIRED`.

| k | PR-6b, paired (decides) | PR-6c, paired (decides) | PR-6b, unpaired | PR-6c, unpaired | Qualifies |
|---|---|---|---|---|---|
| 3 | 0.0035 / 0.0018 / 6 | 0.0130 / 0.0045 / 5 | 0.0061 / 0.0025 / 6 | 0.0130 / 0.0045 / 5 | no |
| 5 | 0.0074 / 0.0018 / 6 | 0.0165 / 0.0083 / 3 | 0.0105 / 0.0032 / 5 | 0.0162 / 0.0081 / 3 | no |
| 8 | 0.0035 / 0.0000 / 6 | 0.0098 / 0.0054 / 6 | 0.0053 / 0.0012 / 6 | 0.0096 / 0.0054 / 6 | yes |
| 15 | 0.0113 / 0.0051 / 5 | 0.0114 / −0.0099 / 5 | 0.0114 / 0.0049 / 5 | 0.0089 / −0.0103 / 6 | no |

The host's own offset on these rows (unpaired minus paired) is at most 0.0067 in absolute
value over every k and seed, and 0.0042 at k = 8.

**Descriptive, per k** (unpaired, computed after the first choice, not read by it).

| k | Interior shares (s, n, ξ, vx) | Interior floor | D₁ without s − s_apex detected (PR-6b / PR-6c) | PR-1b | PR-6 Δ2 |
|---|---|---|---|---|---|
| 3 | 0.176, 0.063, 0.251, 0.070 | {s, ξ} | 1 / 0 of 6 | fails | −0.0089 |
| 5 | 0.187, 0.064, 0.249, 0.070 | {s, ξ} | 3 / 0 of 6 | fails | −0.0060 |
| 8 | 0.224, 0.091, 0.248, 0.076 | {s, ξ} | 3 / 0 of 6 | passes | −0.0013 |
| 15 | 0.289, 0.107, 0.213, 0.070 | {s, n, ξ} | 4 / 0 of 6 | passes | −0.0020 |

At every k the residualisation off and the k = 300 kNN are detected on 6 of 6 seeds (smallest
0.074 and 0.044 on PR-6b, 0.186 and 0.178 on PR-6c), PR-6b's probe-alone condition holds on
6 of 6, and the T-wide floor passes s, n and ξ.

**Consequence.**
- At k = 8 λ_n's interior share is 0.091 and vx's 0.076, both below the 0.10 floor (Section 12,
  decision 5, unchanged). k = 15 would have kept n, but it fails the rule under both
  statistics.
- Section 4's k = 15 is superseded.
- C2-sw's σ^ped is not residualised, and K5 scores raw λ, so neither depends on k.
- The margin on PR-6c is thin (0.0098 on the validation seeds). A failing gate seed in the
  canonical re-run is a finding, not a reason to choose again.

### 15.8 K4's no-gain plants as paired increments (PR-2, PR-3, PR-6, PR-6b, PR-6c)

**Smoke run.** At k = 15, unpaired, PR-2 passed (Δ2 −0.0002 for m, 0.0010 for a*_t) and PR-3
passed (grouped −0.0060; by key 0.028, with the base's rise 0.074).

**Second check** (k = 8). PR-3's grouped Δ2 was 0.0108 against 0.01, K4's one failure. PR-2's
m variant was 0.0070, inside its tolerance.

**Cause.** The unpaired statistic compares one random host plus the plant with other random
networks, so it carries the host's own chance offset against its controls. On that run's world
the host alone scored +0.0069 against its 3 controls, and the codes added +0.0038. At k = 15
the same host scored −0.0001, and host + codes −0.0062. The offset is not a property of the
plant, and any pipeline change re-draws it (here f̂_D's k).

**The rule** (428f51f; `probes.c2_paired` and `paired`, `analysis.c2_network(host=...)`,
`gates.c2_primary(host=...)`; the record's gates.k4_no_gain_rule).
- A no-gain plant is scored by Δ2(host + plant) − Δ2(host). There is one fit of the base [z, m]
  and of the random controls, and both gains use the same rows, components, mask and bootstrap
  counts. The controls' median cancels, so per component the statistic is R²(host + plant) −
  R²(host). The CI comes from the paired bootstrap draws.
- Scored this way: PR-2 in both variants (|Δ2| ≤ 0.01); PR-3's grouped split (Δ2 ≤ 0.01);
  PR-6 (≤ 0.01; 15.5); PR-6b and PR-6c (every seed ≤ 0.01), with their residualisation-off
  checks (> 0.01) and their planted faults.
- Unchanged: tolerances, seeds and every-seed rules; the recovery plants (PR-1a, PR-1b, PR-5
  and PR-8); and PR-3's mis-grouped leak check. A leak is shown there when the codes' unpaired
  gain by key or the base R²'s rise exceeds 0.01, and the paired leak ('by_key_paired') is
  reported. A leak that raises the host's fit as much as the host + plant's cancels in a paired
  increment, which is why that check stays unpaired.
- Reported, descriptive: each plant's unpaired value, and the host's own offset per plant and
  per seed (`host_offsets`; 'host_offset' in gates/k4.json), with every random network's own
  gain and their SD.
- PR-4 is read one-sided: h2's and h3's R² must each be at least the readout's minus 0.02,
  since the probe also sees the layer's other units. It has been coded so since the first
  implementation (c7d2285); 428f51f states it (the record's gates.k4_pr4_rule).
- Tests. A host whose own offset exceeds 0.01 fails no no-gain plant (PR-2, PR-3, PR-6 and
  PR-6b), while a planted gain still does. In the probes test world a plant that adds nothing
  scores −0.00005 paired against 0.0438 unpaired, and a unit of ξ's residual at SNR 1 scores
  0.109 paired, with a CI lower end above 0. The paired increment equals the difference of
  c2's two gains exactly.

**Third check.**
- PR-2: m −0.0001 (CI −0.0007 to 0.0002), unpaired 0.0070; a*_t −0.0001 (CI −0.0005 to
  0.0001), unpaired 0.0041.
- PR-3: grouped 0.0037 (CI −0.0077 to 0.0292); unpaired 0.0108, the second check's failure
  reproduced exactly. By key: unpaired 0.0222 and the base's rise 0.0823, so the leak is shown;
  paired 0.0173.
- The host's own offsets on T with K4's 3 controls: 0.0071 for the λ_res fit that PR-2's m
  variant, PR-3's grouped split and PR-6 share (the four random networks' own gains have SD
  0.0058 there), 0.0042 with a*_t in the base, and 0.0050 on the split by key.

**Consequence.** K4 only. The real C2 and C2-sw on trained networks are unchanged: trained Δ2
minus the median over 5 random-init actors, unpaired (`probes_of` never passes a host). The
offsets are K4's: on T or on the plants' rows, against K4's 3 controls or C2's 5 with a
synthetic target, the random host's own m in the base. They are not C2's noise floor on its
decision rows, which is measured separately (15.10).

### 15.9 The value-fit nulls' full cross-fit, and rows without a calibration

**Value fits** (3e2242f; `nulls.FoldNets`, `fold_gradients`; the record's nulls.per_fold_rule
and nulls.value_mlp_nested_seed). This replaces Section 14's "Value fits" bullet (#20).
- Before, slice f pulled every row back through the fit without fold f. A training row's
  slice-f gradient was therefore in-sample: fold f's calibration was fitted on in-sample
  predictors and applied to out-of-sample ones.
- Now each slice is cross-fitted as `states.cross_fit` cross-fits C2's decoys. In slice f, fold
  f's rows carry the fit without fold f, and a training row of fold g carries the fit without
  folds f and g.
- Every nested fit serving slice f uses fold f's penalty (not chosen again per pair) and is
  z-scored on its own training rows. One fit per unordered pair of folds and penalty serves both
  slices: 15 extra fits when the penalties agree (both ridge fits) and 20 for G1's MLP.
- The MLP's nested fit without folds a < b is seeded mlp_rng(partition, a, 99, N_INNER + 1 +
  b), which no inner-CV or outer fit uses. Far geometry is unchanged (neither fold f nor the
  row's own fold), and so are the shapes ([6, n, 8]; QSS and the barrier [n, 8]).
- Remaining dependency: fold f's penalty was chosen by inner CV over rows that include fold
  g's, so fold g's targets reach slice f's fold-g rows through that one scalar. The ridge
  penalties are 0.01 in every fold.
- **Measured** on the smoke run's world under G1's obs_rms, as the interior K_P-mean null R²_c,
  before → after:
  - every K_P component scored: value_ridge 0.2546 → 0.2172 (n 0.370 → 0.224; its in-sample
    calibration slopes were 9.6–120 by fold, cross-fitted 2.4–9.5); value_ridge_window 0.3344 →
    0.3328; value_mlp 0.4070 → 0.3978. On all of K_P the cross-fit weakens these nulls
    slightly, the opposite direction to #20's change;
  - G1's decision rows (s 1,643, ξ 1,581): both ridge fits' slopes clamp at 0 for s and ξ in
    every fold either way (−0.0128); value_mlp −0.0278 → −0.0164 (s −0.045 → −0.022).
- Cost: 283 → 345 s per network, single-threaded. In the third check G1's nulls stage took
  413 s on 2 threads beside other stages.
- Planted faults that fail the unit tests (ridge and MLP): the nested lookup returning the fit
  without f (the previous layout), or the fit without g.

**Rows without a calibration** (3e2242f; `analysis.calibrated_agreement`, `c3w_of`; the
record's nulls.uncalibrated_rows). Section 14's rule (#10) now covers W1's calibrated agreement
and C3-w's x.
- They zero-filled (np.nan_to_num) the rows whose outer fold had no calibration. Now such rows
  are dropped and counted ('n_uncalibrated'), and R²_c, the Spearman and the slope read the same
  rows. With no row left, the record is 'not testable' with its reason, and C3-w leaves that
  network out as missing ('x_not_testable').
- C1-pattern already dropped such rows (since 58fe333); it now counts them per component.
- Measured effect: none can be measured before the tag, since both read a trained critic's
  width derivative. In the unit test (two geometries, one fold with 2 training rows), the
  zero-filled score differs from the dropped one, and the old zero-fill, planted, fails.

### 15.10 Consequences for C1, C2 and C2-sw

**C1.**
- K1's rule: no decision depends on it (15.2).
- K4's and K5's changes and f̂_D's k do not enter C1. PR-5 is unchanged and passes (15.11).
- C1's nulls: the value fits are cross-fitted in full (15.9). On the decision set the ridge
  nulls stay at the clamp floor (−0.0128 for G1), and value_mlp moves from −0.0278 to −0.0164.
  C1's rule (each null's paired CI lower end of R²_c minus the null > 0) is unchanged.
- W1 (descriptive) and C3-w's x drop rows without a calibration (15.9).

**C2.**
- **Decided on s and ξ only, through the power floor.** At k = 8 the interior residual shares
  are s 0.224, n 0.091, ξ 0.248 and vx 0.076, so the floor passes s and ξ. Under Section 14 the
  floor passed n too, which dropped out only for want of a scored row (K2b). The decision set
  is the same; the reason is now the floor. The T-wide floor (descriptive) passes s, n and ξ.
  The floor's rule (Section 12, decision 5) is unchanged.
- **The statistic** is unchanged: trained Δ2 minus the median over 5 random-init actors,
  unpaired, at the frozen k = 8 and floor.
- **The noise floor** (v2.5; it replaces v2.4's "about 0.007", which was wrong). What C2 gives
  a network that holds nothing beyond a random one is now measured directly: C2's random-only
  null (the pre-tag stage c2-null, `gates.c2_null`, writing gates/c2_null.json, which prereg
  binds). Each of 20 random-init actors of G1's architecture (seeds 5–24, disjoint from C2's
  controls) is scored as the network, exactly as `probes_of` scores a trained one: its own m
  in the base [z, m], the random-init actors 0–4 as controls, z under G1's obs_rms, the real
  λ_res at k = 8 on partition 0, C2's 3,951 interior rows with key weights renormalised there,
  the decision mask (s 1,643 rows, ξ 1,581) and the floor {s, ξ}.
- **The null** (the fourth run, 15.11). Mean −0.0192, SD 0.0282, median −0.0112, range
  −0.0901 to +0.0314, 5th and 95th percentiles −0.0636 and +0.0168. 7 of the 20 refute and 13
  are inconclusive; none confirms. 4 CIs lie wholly below 0 and none wholly above it. By
  component, the gain's SD over the 20 is 0.047 for both λ_s and λ_ξ, and the random networks'
  own gains over the base run from 0.008 to 0.183 for λ_s. (The review's 12 points, seeds
  5–16, are these to the digits it gave.)
- **What v2.4's 0.007 was.** K4's host offsets: on T with K4's 3 controls, and on PR-6b's and
  PR-6c's synthetic targets on the decision rows. Neither is C2's statistic on the real λ_res
  at its decision rows, which v2.4 said it had not measured. Under the design's own definition
  the floor is about 4 times larger in SD, and its mean is negative.
- **Consequences.** C2's CI is the cluster bootstrap over geometries (`probes.gain`), with the
  network and its controls fixed. It does not carry the between-network variance above, so a
  CI that excludes 0 says less about a network's distance from a random one than it seems: 4
  of the 20 random networks have CIs wholly below 0. The confirm rule (point ≥ 0.05 and CI lower
  end > 0) is unchanged and is not broken: the null's largest point is 0.031 and no null
  network confirms. A refute comes easily, since the null's mean is negative and 7 of 20 random
  networks refute. Adding a condition, such as a point above the null's 95th percentile, is the
  lead's decision before the tag. v2.5 does not make it.
- **In the record and the analysis.** prereg copies the null into frozen['c2_null']: the
  summary, each seed's statistic, CI and decision, and its scope. The analysis prints it beside
  each network's primary C2 with the share of null statistics at or above the network's
  (`analysis.c2_null_note`, contained and descriptive). No decision reads it. Its scope is G1's
  obs_rms; each trained network's z is under its own.
- K4's host offsets stay in gates/k4.json, descriptive, and nothing in the analysis reads them.
  The gates docstring's claim that the analysis cites them as C2's noise floor is removed.
- **Band notes** (`gates.band_notes`, copied at prereg into frozen['k4_band_notes'];
  `analysis.band_notes`, contained and descriptive; no decision changes). Beside any C2 refute
  (the primary, a variant or a strata row) whose CI upper end is ≥ 0.03: PR-6's paired
  over-fitting cost (third check: 0.0083, with Δ2's CI −0.0328 to 0.0114). Beside any C2
  confirm whose point is ≤ 0.07: PR-6c's paired leak bound (0.0069, with λ_s's largest gain
  0.0135 and ξ's 0.0004). The canonical re-run's K4 binds the values. Both are measured on C2's
  setup, so v2.5 no longer prints them beside C2-sw (below).
- **Limitations.** PR-1b checks the ridge machinery, not the cross-fit (15.4). PR-6's plant
  reaches R² 0.50 against the kNN's 0.79 and its residualisation-off check is marginal
  (0.00999), so the decoy-leak sensitivity rests on PR-6b and PR-6c. K4 does not detect a D₁
  without s − s_apex (15.6). A decoy-encoding layer that its controls lack loses about 0.01 of
  Δ2 (15.5).
- The PR-6 diagnosis's "no evidence of upward bias" is replaced by the measured PR-6c leak: up
  to 0.0069 on the gate seeds and 0.0098 on the validation seeds at k = 8.

**C2-sw.** σ^ped is not residualised, so f̂_D's k does not enter. PR-8 is unchanged and passes
(gain 0.0815 against 0.086, n 2,911). Its statistic is unpaired, like C2's. K4's band notes do
not carry over (v2.5). PR-6's cost and PR-6c's leak bound were measured on C2's setup: λ_res
through f̂_D, the base [z, m], K_P's components. C2-sw's base is [z] alone on σ^ped, which is
not residualised, so there is no residualiser bias for a leak bound to bound, and units that
hold the decoys could raise its gain by their whole predictive share. C2-sw's between-network
noise floor was not measured: C2's random-only null is not C2-sw's.

### 15.11 The third full-size check, and the canonical re-run

428f51f, code hash a1d336d254dc…, stamped partial (--out; the nulls also --network). Every
stage exited 0, on the default 2 threads, in 96.8 minutes of wall time with K2b, K5 and the
nulls run in parallel.

| Gate | Third check (428f51f) | Pass |
|---|---|---|
| K2 | t_H 267; drops_T False; extend_ok False (3 of 30 insurance solves: 2 in a different basin, 1 at 800 iterations), as in the smoke run | yes |
| K2b | 26 keys; excluded: approach {n, vx}, braking and corner {s, n, ξ, vx}, exit {n, ξ, vx} | yes |
| K1 | 2,008 rows, 0 failed solves. Non-zero n / zero / median / p95: s 203/32/1.7·10⁻⁷/3.8·10⁻⁶; n 212/23/1.5·10⁻⁷/2.3·10⁻⁵; ξ 236/0/3.5·10⁻⁷/3.5·10⁻³; vx 236/0/1.1·10⁻⁷/2.9·10⁻⁶; vy 236/0/4.0·10⁻⁷/1.7·10⁻³; r 188/47/1.65·10⁻⁶/8.8·10⁻⁵; δ 184/51/2.2·10⁻⁵/2.8·10⁻³; pedal 233/0/1.0·10⁻⁵/8.4·10⁻⁴. Kink shares 0.060–0.064; none descriptive; every plant fails, none missing. 'design_rule' fails on δ (p95 0.042); the literal-kink reading leaves all 8 descriptive | yes |
| K1, π | 12 non-zero and 8 zero comparisons; median 8.9·10⁻⁷, p95 7.0·10⁻⁶; every plant fails on 20 rows | yes |
| K4 PR-1a | partial / Δ2 at SNR 0.25, 1 and 4: 0.196 / 0.197, 0.498 / 0.498, 0.794 / 0.794 | yes |
| K4 PR-1b | 15.4 | yes |
| K4 PR-2, PR-3 | 15.8 | yes |
| K4 PR-4 | readout 0.916; h2 0.979 and h3 0.978 (actor and critic) | yes |
| K4 PR-5, PR-5e | 3,951 interior rows: SNR 0.25 0.198, 0.5 0.334 and 1 0.503 (inconclusive), 4 0.803 (confirm); −λ, zero, noise and direction-only −0.0033 to −0.0027 (refute); one component flipped 0.750; ×10 1.0; floors −0.0056 to −0.0012 | yes |
| K4 PR-6 | 15.5 | yes |
| K4 PR-6b, PR-6c | 15.6 | yes |
| K4 PR-7, PR-8, PR-9 | scale changes 0; PR-8 0.0815 against 0.086; D_V 0, largest error 0 | yes |
| K5 wiring | point (lower, upper): s −0.0017 (−0.0069, −0.0000); n −0.0028 (−0.0041, −0.0015); ξ −0.0013 (−0.0256, 0.0144); vx −0.0012 (−0.0034, −0.0005). Per-draw points −0.0168 to +0.0057 | yes |
| K5 control task | descriptive: r* clipped at 0.95 for all four, ceilings 0.946–0.947; selectivity −0.045 to 0.041 | — |
| Nulls (G1) | value_ridge, value_mlp, value_ridge_window and far_geometry [6, 9887, 8]; QSS and barrier [9887, 8]; all finite; ridge α 0.01 (training R² 0.938–0.951); MLP α 10⁻⁴, 10⁻³ in one fold (0.999); window 0.752–0.813 | yes |
| Frozen values | decoy_k 8; interior shares 0.2238, 0.0911, 0.2483, 0.0760; floor {s, ξ}; T-wide shares 0.2473, 0.2034, 0.3116, 0.0626 | — |

Stage times: inputs 103 s, K2 248 s, inputs again 96 s, K2b 3,174 s, K1 1,116 s, K5 2,692 s,
K4 2,179 s (8 minutes at the smoke run), and G1's nulls 413 s.

**Not covered by the third check.** K0, K3, K3b and K6 were not re-run in this round, and the
nulls ran for G1 only.

**The fourth run** (v2.5; a fresh root, the default 2 threads per stage). Inputs, K2, inputs
again and K2b ran at 8392864; K4, decoy-k and c2-null then ran in parallel at e14865d.
- Inputs, K2 and K2b match the third check exactly: the world's sha256 (b5130c418642…), every
  frozen value, K2's record and K2b's record.
- K4 at v2.5's code passes, and its record equals the third check's in every value (0
  differences, compared leaf by leaf apart from provenance, the new 'inputs' and each fault's
  new 'k', which is 300 and 8 as before). The band notes are unchanged: pr6_cost 0.0083,
  pr6c_leak 0.0069.
- decoy-k passes. The rule's choice is k = 8 (qualifying: 8 only), and every per-seed value
  reproduces the table: the largest differences are 4.9·10⁻⁷ paired and 4.9·10⁻⁷ unpaired,
  within the table's rounding. Its per-k interior shares and floors are 15.7's (at k = 8: s
  0.2238, n 0.0911, ξ 0.2483, vx 0.0760; floor {s, ξ}), and the host's offset is at most
  0.0067 in absolute value.
- c2-null: the null of 15.10, every statistic finite. A pre-check of `gates.c2_null` on the
  cached world of the earlier checks gave the same 20 statistics and CIs to the last digit.
- prereg's cross check (`prereg.probe_world_problems`) finds every output's recorded inputs
  equal to the bound ones.
- Stage times: K2b 3,170 s; K4 2,483 s, decoy-k 6,777 s and c2-null 1,206 s, in parallel with
  each other and the rehearsal.

**The canonical re-run** (Section 14, "Where it runs"). Every pre-tag stage runs in the main
checkout on this code, in Section 14's order with v2.5's two stages after K4: inputs, K2,
inputs again (binding t_H), K2b, K0, K1, K3, K3b, K4, decoy-k, c2-null, K5, K6 and the nulls
for all 9 decision networks; then prereg. It must reproduce:
- the frozen decoy_k = 8, the interior shares and the floor {s, ξ} above, t_H = 267 and K2b's
  exclusions;
- K1's, K4's and K5's numbers above, every plant's verdict, and K4's decoy_k = 8;
- the band notes: pr6_cost 0.0083 and pr6c_leak 0.0069;
- decoy-k's choice k = 8 and its validation table (its pass requires the table to 10⁻⁶);
- C2's random-only null as the fourth run gives it (15.10);
- G1's null arrays: their shapes and finiteness (they were byte-identical in the second and
  third checks).

The stages are deterministic on the same code, keys and seeds. K1, K2, K2b, K5, the frozen
values and G1's nulls were identical in the second and third checks apart from provenance, and
K1's fresh re-solves reproduced the smoke run's stored errors. So a difference beyond round-off
is a finding to report. A failing gate blocks the tag and is reported; nothing is chosen again.
The code hash covers src, scripts/costate.py and scripts/_bootstrap.py, not this document.
6d901a4 and v2.5 changed src and scripts/costate.py after 428f51f, so the canonical re-run's
hash is the tagged code's, not the third check's a1d336d254dc…. Its numbers must still match
the third check's: 6d901a4's cap does not bind on the full T (15.6), and v2.5 adds records
(each output's 'inputs', each fault's 'k') and two stages but changes no computation of an
existing gate, as the fourth run shows for K4.
- Expected cost on 2 threads: K2b about 53 minutes, K5 45, K4 36, K1 19, decoy-k 113 (one
  process, about 28 per candidate; it can run beside K5 and the nulls) and c2-null 20; the nulls about 7 minutes per network.
- **The rehearsal** (tests/test_costate_rehearsal.py). At bb96c61 it crashed in K4's PR-6b
  fault ("283 training states < k = 300" on the reduced T); 6d901a4 capped the fault's k
  (15.6). The re-run at 6d901a4 completed in 4,421 s (about 74 minutes: K4 1,922 s, the probes
  stage 1,441 s), every stage exiting 0, with the stand-in floor {s, ξ}, the real interior
  floor at k = 8 (bb96c61). At v2.5's code (e14865d) it ran again with decoy-k (one candidate,
  one validation seed) and c2-null (two of its actors). It completed in 4,999 s (about 83
  minutes: K4 1,922 s, the probes stage 1,622 s, decoy-k 270 s, c2-null 113 s), every stage
  exiting 0, with no contained error and no decision on a non-finite CI; prereg bound the new
  outputs and their recorded inputs, and the analysis placed the null's note beside G1's
  stand-in's primary C2. It still forces K4's, decoy-k's and K5's verdicts to pass. Their
  computed verdicts on the reduced T are False, False and True; decoy-k cannot reproduce the
  full-size table there. It remains evidence of the post-tag path's crash safety only.
- **Tests:** 174 test functions in tests/test_costate_*.py (141 at ed4d743, 170 at 6d901a4).

### 15.12 Earlier text read through this section

| Where | Earlier text | Read as |
|---|---|---|
| Section 4, 'Decoys' | f̂_D is kNN (k = 15) | one k = 8 everywhere (15.7) |
| Section 4, 'Decoys' | a component whose residual share is below 0.10 is descriptive in C2 | at k = 8: n (0.091) and vx (0.076) (15.10) |
| Section 5 | the random controls | K4's plants use 3 of C2's 5; PR-6b and PR-6c use all 5 (15.6) |
| Section 7, K1 | 'Pass', 'Kinks' and π | 15.2 |
| Section 7, K4 PR-1b | one component at a time, r_k on the plant's own baseline | one target array, all four at once (15.4) |
| Section 7, K4 PR-2 and PR-3 | Δ2 within 0.01 of 0; grouped gain ≤ 0.01 | the paired increment (15.8) |
| Section 7, K4 PR-4 | recovered within 0.02 | one-sided (15.8) |
| Section 7, K4 PR-6 | Δ2 within 0.01 of 0; units scaled to match the kNN | one-sided and paired, a fixed gain grid; PR-6b and PR-6c added (15.5, 15.6) |
| Section 7, K5 | permuted within each training fold; the CI at or below 0 | 15.3 |
| Section 7, 'Freeze' | f̂_D | the frozen decoy_k, its validation tables (re-derived by decoy-k, 15.7), K4's band notes and C2's random-only null (15.10) |
| Section 12, decision 5 | the power floor | its consequence at k = 8 (15.10) |
| Section 14, 'What decides on T' | the floor passes s, n and ξ | s and ξ (15.10) |
| Section 14, #20 'Value fits' | slice f through the fit without f | the full cross-fit (15.9) |
| Section 14, #10 | C1, C1-pattern and C1-limit | also W1 and C3-w's x (15.9) |
| Section 14, 'Where it runs' | the pre-tag order | decoy-k and c2-null after K4 (15.11) |
| Section 14, 'The rehearsal' | K4's and K5's verdicts forced | still forced, with decoy-k's; the computed verdicts beside them. K4's and K5's real verdicts are the full-size checks' (15.11) |

### 15.13 Smoke-run results and their resolutions

| Smoke-run result (v2.2, 185c637) | Resolution | Commit |
|---|---|---|
| K1 passed with its plants, on a rule Section 7 does not state | The rule written down (15.2): zero comparisons counted, not pooled; a decided component needs a non-zero comparison; one-sided estimates stored and the literal reading reported; constants and rule in the record | d1ff055 |
| K1's π: 8 of 20 zero comparisons pooled as 0 | π under the same zero rule, at its floor 6.03·10⁻⁶ | b8001d6 |
| K4 PR-1a passed | Unchanged | — |
| K4 PR-1b failed: partial 0.70–0.75, Δ2 0.046–0.083 short | One target array; all four components at once; the by-fold plant descriptive | 778c06e |
| K4 PR-2 passed (unpaired) | Paired increment | 428f51f |
| K4 PR-3 passed (unpaired; failed the second check at 0.0108) | Grouped split paired; mis-grouped leak check unchanged | 428f51f |
| K4 PR-4 passed (one-sided as coded) | One-sided reading stated | 428f51f (coded at c7d2285) |
| K4 PR-5, PR-7, PR-8 and PR-9 passed | Unchanged | — |
| K4 PR-6 failed: Δ2 −0.0118; units' R² 0.36 against the kNN's 0.80 | Biased units, a fixed gain grid, one-sided; then paired | 778c06e, 428f51f |
| None (found in PR-6's verification): f̂_D at k = 15 leaks in a kNN-shaped world | PR-6b and PR-6c; a nested-CV k, which failed PR-6b at full size; one fixed k = 8 by the validation rule, re-applied on the paired statistic | 778c06e, bac63d0, 55b84c4, 428f51f |
| K5 failed: s's and n's CI upper ends +0.0004 and +0.0032; ξ −0.70 | Permuted across training folds, odd components within direction; 10 draws averaged; lower end ≤ 0 and point ≤ 0.01 | 6fde947 |
| K0, K2, K2b, K3 and K3b passed; K6 counted | Unchanged; K2 and K2b re-run in the checks with the same results | — |
| No gate: C1's value-fit nulls were in-sample on training rows | Full cross-fit | 3e2242f |
| No gate: W1 and C3-w's x zero-filled uncalibrated rows | Dropped and counted | 3e2242f |

### 15.14 The review of v2.4, and v2.5's resolutions

23 Oct 2026, before the tag. A review of v2.4 on three lenses (forking paths, statistics,
integrity) confirmed 10 findings: one blocker and nine minor, three of which repeat a
finding from another lens. No finding showed a defect in a gate rule, tolerance, seed or the choice of k, and
none is changed. v2.5's code is e14865d. Its full-size run (the fourth) is in 15.11.

| # | Finding (lens; severity) | Resolution |
|---|---|---|
| 1 | C2's noise floor on its decision statistic is about 0.03 SD, not the 0.007 that 15.10 stated; C2's CI does not carry it (forking paths; blocker) | Measured and frozen: the pre-tag stage c2-null (`gates.c2_null`; 20 random-init actors: mean −0.0192, SD 0.0282, largest +0.0314), the record's 'c2_null', `analysis.c2_null_note` beside each primary C2 (descriptive). 15.10 rewritten. No confirm condition is added; that is the lead's decision |
| 2 | 15.11 stale at HEAD: 6d901a4, the rehearsal's crash and re-run, its 74 minutes and floor {s, ξ}, the test count; 15.12's rehearsal row (forking paths; minor) | 15.6, 15.11 and 15.12 updated |
| 3 | The gates docstring says the analysis cites the host offsets as C2's noise floor; nothing does (forking paths; minor) | Clause replaced: the offsets are descriptive and read by nothing; C2's noise floor is `c2_null`'s |
| 4 | 15.7's disclosure leaves out that both worlds' gate seeds had been scored at k = 8 before the rule was re-applied (forking paths; minor) | Sentence added to 15.7 |
| 5 | K4's band notes beside C2-sw, where no K4 measurement supports them (statistics; minor) | `analysis.band_notes` on C2's records only; 15.10's C2-sw paragraph rewritten; the record's constants state the scope |
| 6 | Section 15 describes 428f51f, but 6d901a4 changed gates.py; the code-hash expectation is stale (statistics; minor) | As #2. 15.11 states that the canonical re-run's hash is the tagged code's, and why its numbers still match the third check's |
| 7 | f̂_D's validation table is a hand-copied constant that no pre-tag stage computes or checks (integrity; minor) | The pre-tag stage decoy-k (`gates.decoy_k`), bound by prereg: it passes only when the rule's choice is 8 and every per-seed value reproduces the table to 10⁻⁶ (15.7). In the fourth run it passes (largest difference 4.9·10⁻⁷) |
| 8 | K4 reads the power floor and C2's mask from inputs/summary.json and gates/k2b.json, and prereg never checks k4.json against them (integrity; minor) | K4, decoy-k, c2-null and K5 record their 'inputs'; `prereg.probe_world_problems` refuses an output computed on other inputs than the bound ones ('rerun k4') |
| 9 | As #6 (integrity; minor) | As #2 and #6; the record's constants also state the cap rule ('oversmooth_k_rule') |
| 10 | As #3 (integrity; minor) | As #3 |

Also corrected: the gates docstring's host offset at k = 8 (0.0042, as 15.7, not 0.0043).
