# Level-score audit harness: implementation spec (`audit-v1` scorer, racing PLR⊥, racing SFL)

Draft for sign-off, 2026-10-18.

This spec covers the six headline estimators, the two samplers, a table of deviations from the reference code, and unit tests.
- Formulas marked **REF** come from pinned source code.
- Anything in our version that differs from the reference is tagged **[DEV-n]** and listed in §3.
- I re-read every code line quoted here at the commits below.
- I computed the expected values in §4 by running line-for-line float64 ports of three functions: JaxUED `accumulate_rollout_stats` and `compute_gae`, and MNA `compute_clipped_gae`.

---

## 0. Sources, notation, shared machinery

### 0.1 Pinned sources

| Tag | Repo @ commit | Files |
|---|---|---|
| JaxUED | `DramaCow/jaxued` @ `0f8f1284677375b889e4f13a32c9617cd009f8c4` | `src/jaxued/utils.py`, `src/jaxued/level_sampler.py`, `examples/maze_plr.py`, `examples/craftax/craftax_plr.py` |
| minimax | `facebookresearch/minimax` @ `2ae9e04d37f97d7c14308f5a26237dcfca63470f` | `src/minimax/util/rl/ued_scores.py`, `util/rl/rollout_storage.py`, `agents/ppo.py`, `models/common.py` |
| MNA | `HarryMJMead/Dynamic-Environment-Generation-for-UED` @ `12340c7fe443278f4033bab977054af1e40b8188` | `minigrid/maze_plr.py`, `src/jaxued/utils.py`, `minigrid/config/main_plr.yaml` |
| SFL | `amacrutherford/sampling-for-learnability` @ `8b14e2ec07b38dc33a9df476b84faa2f3ca5fd3b` | `sfl/train/minigrid_sfl.py`, `sfl/train/config/{minigrid-sfl,jaxnav-sfl}.yaml`, `config/learning/{ippo-minigrid,ippo-jaxnav}.yaml` |
| Ours | `adrian0729/apexzero` @ `99ee5a0` | `src/apexzero/agents/ppo.py`, `src/apexzero/env/racing.py`, `src/apexzero/audit/rollouts.py` |

Papers:
- Robust PLR, arXiv 2110.02439, §4.2.
- SFL ("No Regrets"), arXiv 2408.15099v3: §4.1, §4.3, Alg. 1–2.
- MNA, arXiv 2601.14957: §4–4.1, Eqs. 8–13.

### 0.2 Notation

- **Slot j** is the env index; **t** is the step.
- **Episode e** is one window. It runs over steps t₀(e)…t_end(e), and T_e = t_end − t₀ + 1. T_e counts the terminal transition, as JaxUED does (`step_count += 1` before the done test, `utils.py` L14).
- **r_t** is the raw reward (`raw_reward`): progress/10 m − 0.02‖Δa‖², minus 5 + 5(v/50)² on a crash (`racing.py` L175–178).
- **V̂_t** is the main critic output on the (noisy, normalised) observation. It is trained on scaled rewards `reward / sqrt(ret_rms.var + 1e-8)` (`ppo.py` L385–386).
- **V_t := σ·V̂_t**, with σ = sqrt(ret_rms.var + 1e-8). **[DEV-6]**
  - In the fresh audit, σ comes from the checkpoint's `ret_rms`.
  - In training, σ is the agent's value at the start of the chunk.
- **V^k_t, k = 1…5** are the ensemble heads, scaled by σ in the same way.
- **term_t** marks a crash. **trunc_t** marks the window end (t = 1200) or the end of the strip (`racing.py` L167, L172). d_t = term_t ∨ trunc_t.
- **V^f_t** is the value of the pre-reset final observation on truncation (`Transition.final_value`, `rollouts.py` `final_value`).
- **κ_t = 1** if t is the last step of a 128-step chunk and d_t = 0. The bootstrap value there is b_t = V(s_{t+1}), evaluated with the parameters that produced V_t (`last_value`, `ppo.py` L404).
  - In the fresh audit, κ_t = 1 when t ≡ φ_e − 1 (mod 128). φ_e ~ U{0..127} is drawn per window from the audit seed, and b_t = V_{t+1} (frozen critic). Rationale: in training, window starts are uniformly phased against chunk boundaries.
- **γ = 0.995**. **λ_PPO = 0.95** (`PPOConfig`). The stored λ variants are {0.9, 0.95, 0.98, 1}; λ_PPO is the headline.
- **G_e** = Σ_{t∈e} r_t (undiscounted).
- **G^γ_e** = Σ_{τ=0}^{T_e−1} γ^τ r_{t₀+τ}, with no bootstrap. This equals `WindowRollouts.returns(γ)`.
- **V̄_e** = (1/T_e) Σ_{t∈e} V_t.

### 0.3 Scoring GAE

REF. JaxUED `examples/maze_plr.py` L63–77. The same recursion appears in minimax `rollout_storage.py` L172–199 and MNA `maze_plr.py` L69–83.

```python
delta = reward + gamma * next_value * (1 - done) - value          # L66
gae = delta + gamma * lambd * (1 - done) * gae                    # L67
```

- `next_value` is V_{t+1}, or `last_value` at the end of the rollout.
- `done` includes the time limit (maze `env.py` L167–170). The time limit is therefore **terminal**, with no bootstrap.
- The score uses the PPO's own advantage tensor and the PPO's λ: the maze default is `--gae_lambda 0.98` (L871).

OURS. This is a separate scoring pass in raw units. It does not reuse the PPO's tensor, which is in scaled units and bootstraps truncations. **[DEV-6]**

```
nxt_t = 0                if term_t
      = 0 | V^f_t        if trunc_t        # mode F (headline, = REF) | mode B (variant, = our PPO, ppo.py L407-412)
      = b_t              if κ_t
      = V_{t+1}          otherwise
δ_t = r_t + γ·nxt_t − V_t
A_t = δ_t + γλ·(1 − d_t)·(1 − κ_t)·A_{t+1}
```

Notes on mode F (window end treated as terminal):
- **Expected artefact.** The critic is time-unaware: the observation has no time-in-window. At every window end, δ ≈ r − V ≈ −(value scale), roughly −60 raw at racing speed (horizon 1/(1−γ) = 200 × ~0.3).
- **Where it lands.** A_t is pushed strongly negative over the last ~1/(1−γλ) ≈ 18 steps.
  - L1 gains about |V_end|·18/T_e (≈ 0.9 per step at T_e = 1200).
  - Mean(A) is biased negative, so PVL/L1 < 0.5 even with an accurate critic.
- **Consequence for E2.** The plan's prediction "PVL/L1 → 0.5" can only hold in mode B. Log both modes, plus the tail term (the Σ|A| of the last 64 steps of each completed window).

### 0.4 Chunk accumulator (one implementation, used by every estimator)

Each slot j carries the following across chunks:
- n;
- ΣV, V_first;
- Σr;
- Σγ^τ r, together with the running γ^τ;
- for each λ variant: Σmax(A,0), Σ|A| and ΣA;
- ΣĜ (MNA);
- Σ std_k(V^k);
- distance;
- the level assignment.

```
for each chunk:
  1. roll out 128 steps; store (r, V̂, V̂^k, term, trunc, V̂^f); compute b at the chunk end with the same params
  2. reverse scans over the chunk: GAE (one per λ, mode F and mode B) and MNA clipped GAE (§1.6), with κ at the last step
  3. forward scan: add the per-step terms to the carry; when d_t = 1, emit EpisodeRecord(e) and reset the carry
     a carry with d = 0 at the chunk end is NOT emitted; it continues into the next chunk          [DEV-4]
```

EpisodeRecord holds:
- slot, level_id, kind ∈ {explore, replay, dr, audit};
- t_end (global step), T_e, crashed, distance;
- G_e, G^γ_e, V_first, V̄_e;
- for each λ: PVL_e, L1_e, mean A_e;
- MNA sum and T_e;
- disagreement_e.

Precedent for this design is JaxUED Craftax (`craftax_plr.py` L224–291):
- it runs `outer_rollout_steps` chunks of `num_steps`, with PPO updates between them;
- GAE is computed per chunk, bootstrapped with that chunk's `last_value`;
- scores are computed on the concatenation (L655–656, L693–694).

So an episode can see up to 11 parameter snapshots: 1,200 steps cover 10 or 11 chunks. **[DEV-2] [DEV-3]**

**Equivalence requirement (a unit test, §4).** The emitted per-episode values must equal JaxUED `accumulate_rollout_stats(time_average=True)` applied to the concatenated per-chunk advantages, for every episode that completes.

---

## 1. The six headline estimators

### Summary: how the reference handles incomplete episodes, and what we do

| Estimator | Reference function | Trailing incomplete episode (REF) | No completed episode (REF) | Ours |
|---|---|---|---|---|
| L1 | minimax `compute_l1_value_loss` | dropped | 0 (sum/max(n,1), `ued_scores.py` L89–96), or `ignore_val` (L164–170); MNA-utils `l1_value_loss` gives −inf | never scored (carried to the next chunk); in-flight episodes at the end of a run are discarded **[DEV-4]** |
| PVL | JaxUED `positive_value_loss` | dropped | −inf | same as above |
| MaxMC | JaxUED `max_mc` + `compute_max_returns` | dropped, from both R_max and the score | −inf (and R_max = 0) | same as above |
| Learnability | SFL `_calc_outcomes_by_agent` | dropped (`end_idx != max_steps`); at most 10 episodes counted | NaN, which `argsort` sorts last and top-K then **selects** | assert that no window is incomplete or missing |
| MNA | MNA `compute_clipped_gae` + `negative_value_loss_solved_sum` | **included** (bootstrapped with `last_value`) | Ĉ = 0 on new levels (R_max = 0), so the score is 0 | carried; only completed episodes are scored **[DEV-4]** |
| Disagreement | minimax `compute_value_disagreement` | dropped | 0 (minimax); −inf (MNA-utils) | same as L1 |

A port note: JaxUED/minimax accumulators require bool or uint `dones`. Float dones break the `uint32` scan carry, and the MNA clipped GAE needs bool dones.

### 1.1 L1 value loss

**REF** (minimax `util/rl/ued_scores.py` L182–186; the advantages are PPO GAE from `rollout_storage.py` L145–199):
```python
def compute_l1_value_loss(batch):
    mean_scores, max_scores = compute_episodic_stats(jnp.abs(batch.advantages), batch.dones, time_average=True)
```

The score is the mean over the episodes that complete in the rollout:

  S_L1 = (1/E) Σ_{e∈𝓔} (1/T_e) Σ_{t∈e} |A_t|

The mean is episode-weighted. With an eval axis, minimax pools the episodes of all eval copies (`ued_scores.py` L89–96).

**Ours.**
- Per episode: L1_e = Σ|A_t| / T_e.
- Level score: the mean of L1_e over the counted episodes (see §1.8).
- Headline uses λ_PPO and mode F. Variants: the other λ values, mode B, and the discount-weighted form Σγ^τ m_τ / Σγ^τ.

### 1.2 Positive value loss (PVL)

**REF** (JaxUED `src/jaxued/utils.py` L55–57; called from `maze_plr.py` L389–395, L545–548 and L603–606):
```python
def positive_value_loss(dones, advantages, incomplete_value=-jnp.inf):
    mean_scores, _, episode_count = accumulate_rollout_stats(dones, jnp.maximum(advantages, 0), time_average=True)
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)
```

  S_PVL = (1/E) Σ_e (1/T_e) Σ_{t∈e} max(A_t, 0)

This matches the Robust PLR §4.2 formula with a per-episode time average and an equal-weight mean across episodes.

**E2 identity.** Per episode, PVL_e = ½·L1_e + ½·mean_e(A) exactly, because max(a,0) = (|a| + a)/2.

**Ours.** As for L1, with max(A,0) in place of |A|.

### 1.3 MaxMC, as implemented

**REF** (JaxUED `utils.py` L44–53):
```python
def compute_max_returns(dones, rewards):
    _, max_returns, _ = accumulate_rollout_stats(dones, rewards, time_average=False)
def max_mc(dones, values, max_returns, incomplete_value=-jnp.inf):
    mean_scores, _, episode_count = accumulate_rollout_stats(dones, max_returns[None, :] - values, time_average=True)
```

- `max_val` starts at **zeros** (`utils.py` L33, L36). Therefore R_max = max(0, max over completed episodes of G_e). It is **undiscounted** and floored at 0.
- **New and mutated levels:** R_max comes from this rollout only (`maze_plr.py` L545–548).
- **Replay:** `max_returns = jnp.maximum(stored["max_return"], compute_max_returns(dones, rewards))` (L604), and the updated value is used in the score and stored (L605–606).
- **Duplicate-check hit on insertion:** `level_sampler.py` L170–176 calls `update`, which **overwrites** `max_return` (L237–256).
- R_max is computed over the whole rollout before scoring, so every episode is scored against the rollout-level maximum:

  S_MaxMC = (1/E) Σ_e (1/T_e) Σ_{t∈e} (R_max − V_t) = R_max − (1/E) Σ_e V̄_e

**Units.** R_max is undiscounted (up to ~1,200 steps of reward) while V is discounted (≤ 1/(1−γ) = 200 steps of reward). We keep this mismatch because this is the as-implemented score.

**Ours.**
- Each record keeps G_e and V̄_e. At write time:
  - R_max := max(0, stored R_max [replay only], max of G_e over the write group);
  - S = R_max − mean(V̄_e).
- **Fresh audit:** R_max is taken over windows 1–4. This is exactly JaxUED's new-level branch on a rollout with 4 completed episodes. **[DEV-23]**
- **In buffer:** the rules above, including the overwrite on a duplicate hit.

### 1.4 MaxMC, unit-consistent variant (ours; no reference) [DEV-22]

  S_MaxMC-unit = (1/E) Σ_e (R^γ_max − V_{t₀(e)})

- R^γ_max = max of G^γ_e over the same episode set and stored history as R_max.
- It is initialised to **−inf**, with no zero floor: returns can be negative early in training, and R^γ_max is meant to estimate J*.
- The score is **not** time-averaged. V_{t₀} is the critic at the first observation of the window.
- R^γ_max is stored in `levels_extra["max_return_disc"]` and follows the same update and overwrite rules as R_max.

**E2 identity.** S_unit − Reg = (R^γ_max − J*_20) + (Ĵ^π − mean V_{t₀}):
- the first term is the ceiling gap (≤ 0 if J* is the true optimum);
- the second is the critic bias;
- Ĵ^π comes from the target windows.

### 1.5 Learnability p(1−p)

**REF** (SFL `sfl/train/minigrid_sfl.py`):
```python
action = pi.sample(seed=_rng).squeeze()                                              # L185 (stochastic)
mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)              # L225
goal_r = (returns > 0) * 1.0                                                         # L227
success = jnp.sum(goal_r * mask)                                                     # L228
done_idxs = jnp.argwhere(dones, size=10, fill_value=max_steps).squeeze()            # L234
"success_rate": success.mean(where=mask_done)                                        # L240
learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=0)           # L280
```

- Each level is played in one L-step stream with AutoReplay to the same level (`AutoReplayWrapper`, L89).
- p̂ = mean over the first ≤10 completed episodes of success_e. Here success_e is the **count of steps with r > 0** in the episode; that is an indicator only because maze rewards are sparse. JaxNav uses `info["GoalR"]` instead.
- ℓ = p̂(1−p̂). Paper §4.1: "we follow Tzannetos et al. [10] and define learnability to be p·(1−p)."

**Ours.**
- success_e is a per-episode **indicator**. **[DEV-9]** A literal port would count every dense-reward step and give p > 1.
- **S1, window completion:** ¬crashed, i.e. the window ended by truncation at t = 1200. Assert that `end_of_strip` never fires.
- **S2, non-saturating:** S1 ∧ distance_e ≥ 0.97·d_QSS(θ), where d_QSS is the distance the QSS oracle covers from x0 in 60 s.
- p̂ = mean of success_e over the counted windows, which are independent windows rather than one AutoReplay stream. **[DEV-10]**
  - Each window contributes exactly one episode, so this pooled mean equals the analysis notebook's p̄ over workers.
  - ℓ = p̂(1−p̂).

### 1.6 MNA (Maximised Negative Advantage)

**REF** (MNA `minigrid/maze_plr.py` L85–143, called with `use_max_value=True` at L674, L736 and L805):
```python
deltas = rewards + gamma * extended_values[1:] * (1 - dones) - extended_values[:-1]
td_total_array = td_total_array + gamma**i * delta
clipped_td_total_array = jnp.minimum(td_total_array, 0)
if use_max_value: td_total_array = clipped_td_total_array
gae_array = gae_array + (lambd**(i)) * clipped_td_total_array * (1 - is_done_array) + ((lambd**(i+1))/(1 - lambd)) * clipped_td_total_array * done * (1 - is_done_array)
advantages = (1 - lambd)*(gae_array + ((lambd**(i))/(1 - lambd)) * clipped_td_total_array * (1 - is_done_array))
```

Scoring (`maze_plr.py` L496–498 and `src/jaxued/utils.py` L68–70):
```python
solved = max_returns > 0
sum_scores = -jnp.minimum(advantages, 0).sum(axis=0) * solved
```

Closed form (I derived and checked it):
- Let A_s^(n) = Σ_{i<n} γ^i δ_{s+i}, using the same next-value rule as GAE.
- Let M_n = max(0, A^(1), …, A^(n)).
- The clipped carry equals A^(n) − M_n. Hence:

  Ĝ_s^(n) = M_n − A_s^(n)  (paper Eq. 10)
  Ĝ_s = Σ_{n=1}^{N_s−1} (1−λ)λ^{n−1} Ĝ_s^(n) + λ^{N_s−1} Ĝ_s^(N_s)

- N_s is the number of steps from s up to and including the episode's done step, or up to the end of the rollout.
- The paper's Eq. 11 indexes from n = 0. We follow the code.
- REF score = Ĉ · Σ_{s=0}^{T−1} Ĝ_s, summed over **all** T = 512 rollout steps, including several episodes and the incomplete tail.
  - Ĉ = 1[R_max > 0], with JaxUED's zero-floored undiscounted R_max; on replay it is combined with the stored value.
  - Defaults (`main_plr.yaml`): γ = 0.995, λ = 0.95, num_steps 512.

**Ours.**
- Ĝ_t is computed per chunk, with κ playing the role of the rollout end and bootstrapping from b. Mode F is the headline.
- Episode sum ΣĜ_e. Level score = Ĉ · (Σ_e ΣĜ_e) / (Σ_e T_e): steps are pooled, as the reference pools steps, and the result is normalised by the step count because our T_e varies (the reference T is fixed). **[DEV-11]**
- Ĉ = 1[any counted window, or any window since admission, satisfied S1]. **[DEV-12]** Also log Ĉ_lit = 1[R_max > 0], which is ≈ 1 always under dense reward.
- Implement with the closed-form weights. The code's `1/(1−λ)` form divides by zero at λ = 1; at λ = 1 the closed form gives Ĝ_s = Ĝ_s^(N_s).
- Cost is O(128²) per slot per chunk.

### 1.7 Ensemble disagreement (5 stop-gradient heads)

**REF** (minimax `ued_scores.py` L212–217; MNA-utils `value_disagreement` is identical except for −inf):
```python
mean_scores, max_scores = compute_episodic_stats(batch.values.std(-1), batch.dones, time_average=True)
```

- `std` is the population standard deviation (ddof = 0).
- The ensemble is `EnsembleValueHead`: `nn.vmap` with `split_rngs={"params": True}` (`models/common.py` L225–255) on a **shared trunk that receives gradients from every head**.
- Each head is trained on **its own** GAE target. `rollout_storage.py` L186–199 broadcasts reward and done over the ensemble axis, so target_k = A^k + V^k.
- The policy uses head 0's GAE (`agents/ppo.py` L233–234).

  S_dis = (1/E) Σ_e (1/T_e) Σ_{t∈e} std_k(V^k_t)

**Ours.**
- 5 extra heads read `stop_gradient(trunk)` (`ppo.py` L207–216, commit f419547). The main critic is excluded from the std.
- Heads are 64-unit ReLU (minimax: 1 hidden layer of 256 tanh). **[DEV-13]**
- **Target mismatch:** the current code trains every head on the main critic's target (`head_loss = ((heads - tgt[:, None]) ** 2).mean()`, `ppo.py` L452–454). This is the AutoDIME choice, not minimax's.
  - Per the "follow the reference" rule: switch to per-head GAE(λ_PPO) targets computed from each head's own values. That needs per-head V^k(final_obs) at truncation.
  - If the switch is not made, the shared target becomes **[DEV-14]**.

### 1.8 Aggregation over windows and writes

| Context | Episodes used | L1, PVL, MaxMC, MaxMC-unit, disagreement | Learnability | MNA | R_max, R^γ_max, Ĉ |
|---|---|---|---|---|---|
| E1 operational | windows 1–4 of (level, checkpoint) | episode-weighted mean (JaxUED L40) | pooled p̂ | step-pooled, Ĉ-gated | from windows 1–4 |
| E1 high-budget | odd windows (score) vs even windows (target) | same | same | same | from the score windows |
| Training write (racing PLR⊥) | write group: the episodes of one slot on one assignment that end in the same chunk (normally 1) | same | — | — | stored ⊕ group (rules §1.3) |

E1 runs offline on the `WindowRollouts` arrays:
- T_e = Σ`alive`. `alive` includes the terminal step (`rollouts.py` L82–85).
- term at t_end iff `terminated`.
- V^f = `final_value`.
- σ is read from the checkpoint dict key `"ret_rms"` (written at `ppo.py` L497–499). `audit.rollouts.Snapshot` currently drops it.

---

## 2. Samplers

### 2.1 PLR⊥: the reference algorithm (JaxUED)

```
LevelSampler state (level_sampler.py L67-94): levels[C], scores[C] = -inf, timestamps[C] = 0, size = 0, c = 0
                                              (c is "episode_count"), levels_extra[C] (init {"max_return": -inf}, maze_plr.py L499)

level_weights():                                                        # L276-341
  mask_i = i < size
  ord   = argsort(-where(mask, scores, -inf))                           # stable: ties go to the lower index
  rank_i = position of i in ord, + 1
  w_s   = mask * (1/rank)^(1/β);  w_s /= Σ w_s                          # L312-316
  w_c   = mask * (c - timestamps)                                       # L338-339
  w_c   = w_c/Σw_c if Σw_c > 0 else mask/size                           # L340
  return (1-ρ)·w_s + ρ·w_c                                              # L290

replay_decision():  (size/C >= min_fill) & (U < replay_prob)            # L108 — ONE decision per update (maze_plr.py L708)

each update (num_envs envs, num_steps steps, every env reset to its level with a fresh RNN carry; AutoReplay on done):
  if not replay:                                                        # on_new_levels
     levels ← generator; roll out; A ← GAE(γ, λ_PPO); R_max ← compute_max_returns
     S ← score(...)            (-inf if no episode completed)
     for each env in order: insert(level, S, {max_return: R_max})       # insert_batch, L180-193
     PPO update computed but NOT applied (update_grad=False, L563; L267-268)          # the "⊥"
  else:                                                                 # on_replay_levels
     for each env in order: i ~ Categorical(level_weights()); c += 1; timestamps[i] ← c   # L122-127, a sequential scan
     roll out; R_max ← max(stored[i], rollout R_max); S ← score(...)
     for each env in order: scores[i] ← S; extras[i] ← R_max            # update_batch, last write wins (L258-274)
     PPO update applied (L621)

insert(level, S, extra):                                                # L145-178, L375-404
  if duplicate_check and level == levels[i] for some i < size:
     scores[i] ← S; extras[i] ← extra; timestamps[i] ← c+1; c += 1; return
  i ← size if size < C else argmin(level_weights())                    # lowest COMBINED weight, not lowest score
  if scores[i] < S:  levels[i], scores[i], extras[i] ← level, S, extra; timestamps[i] ← c+1; size ← min(size+1, C)
  c += 1                                                                # every attempt, accepted or not
```

So P(l_i) = (1−ρ)·h_i^{1/β}/Σ_j h_j^{1/β} + ρ·(c − C_i)/Σ_j (c − C_j), with h = 1/rank. A −inf score is never admitted, because −inf < −inf is False.

| Parameter | `LevelSampler` class default | `maze_plr.py` default (L860–891) | Robust PLR paper (MiniGrid) | **Ours (curriculum C)** |
|---|---|---|---|---|
| capacity | required | 4000 | 4000 | **1000** [DEV-20] |
| replay_prob | 0.95 | 0.8 | 0.5 | **0.8** |
| staleness ρ | 0.5 | 0.3 | 0.3 | **0.3** |
| prioritisation / β | rank / 1.0 | rank / 0.3 | rank / 0.3 | **rank / 0.3** |
| minimum_fill_ratio | 1.0 | 0.5 | – | **0.5** (taken from the same defaults; confirm) |
| duplicate_check | False | True | – | **True, matched by `level_id`** |
| exploratory_grad_updates | – | False (PLR⊥) | – | **False** |
| score | – | MaxMC (choices MaxMC, pvl) | MaxMC | arms: PVL, MaxMC (as implemented), oracle-latest |
| envs × steps | – | 32 × 256 | 32 × 256 | N_envs × 128-step chunks; 1,200-step windows |
| γ / λ | – | 0.995 / 0.98 | 0.995 / 0.95 | 0.995 / 0.95 |

With β = 0.3 and a 1,000-level buffer, the top-ranked level takes 0.872 of the score weight (β = 1: 0.134; β = 0.1: 0.999). After the ρ mix it receives ≥ 0.61 of all replays. This is reference behaviour, and C replicates it.

### 2.2 PLR⊥: our adaptation (asynchronous windows over 128-step chunks)

```
state: the JaxUED buffer (+ level_id, + extras {R_max, R^γ_max, ever_S1}), c
per slot j: cur_j = (kind, level_id, buf_idx), next_j (pre-drawn), accumulator acc_j

for each chunk k:
  A. pre-draw, from the buffer state at the chunk start (weights frozen for the chunk)             [DEV-17]
       w = level_weights()
       for each j: replay_j = (size/C ≥ 0.5) & (U_j < 0.8)                                        [DEV-15]
                   next_j = (replay, i ~ Cat(w), level_id[i]) if replay_j else (explore, θ ~ U(pool of 4,000))
  B. roll out 128 steps. When d_t = 1 on slot j: emit a record tagged cur_j; set cur_j ← next_j; reset to next_j's level.
       If slot j finishes again within the same chunk, it replays next_j (AutoReplay semantics).
  C. scoring pass (§0.4), then the PPO update with loss_mask = 1[cur.kind == replay] on the actor, critic and head
     losses; obs/return normalisers are updated from masked-in transitions only; minibatch advantage
     normalisation runs over masked-in samples                                                    [DEV-16]
  D. process the events of the chunk in (t, j) order:
       first use of a pre-drawn replay assignment: c += 1; timestamps[i] ← c   (unused pre-draws are discarded)
       finished write group, explore: R_max = max(0, max G); S = score(group); insert(level, S, extras)
                                      (duplicate hit by level_id ⇒ overwrite score and extras, refresh timestamp, c += 1)
       finished write group, replay:  if level_id[i] ≠ group.level_id: drop (level evicted mid-window)  [DEV-18]
                                      else R_max ← max(extras[i].R_max, group R_max) (same for R^γ_max);
                                           scores[i] ← S; extras[i] ← ...; store write_time for score age
                                      (each write reads the latest stored R_max)                  [DEV-19]
```

Properties of this protocol:
- A new level gets one full exploratory window before it can enter the buffer.
- −inf scores cannot arise; assert this.
- An oracle-latest arm plugs into step D with S = J*_20 − mean of G^γ_e over the write group. This is jaxgmg `oracle-actor`: `oracle_max_return - average_return`. Ours never has a trailing partial episode, whereas jaxgmg's `compute_average_return` counts one.

### 2.3 SFL: the reference algorithm

Paper (§4.3, Alg. 1–2):
- "D ← collect_learnable_levels(π)"; then, for t = 1…T: "ρ·N_L levels sampled uniformly from D", together with "(1−ρ)·N_L randomly generated levels".
- Algorithm 2 returns the top K of N random levels, ranked by p(1−p) from L-step rollouts.
- Stated defaults: "T=50, ρ=0.5, NL=256, L=2000, N=5000 and K=1000".

Code (`minigrid_sfl.py`):
```
every EVAL_FREQ updates (L678-680):
  for NUM_BATCHES × BATCH_SIZE fresh random levels: roll out ROLLOUT_STEPS with the stochastic policy and AutoReplay
  p, ℓ as in §1.5;  D ← argsort(ℓ)[-NUM_TO_SAVE:]       (L289; ascending stable sort ⇒ ties go to the HIGHER index; NaN ⇒ selected)
  each PPO update: slots [0, N_gen) get new random levels; slots [N_gen, N) get randint(0, K) draws from D   (L622-641)
                   all envs reset; NUM_STEPS rollout with AutoReplay; PPO update on everything
  assert NUM_ENVS_FROM_SAMPLED + NUM_ENVS_TO_GENERATE == NUM_ENVS (L83)
```

D is fully replaced at each refresh. There is no staleness term and no re-scoring between refreshes.

| | Paper §4.3 | MiniGrid code | JaxNav code | **Ours (proposed; not fixed in the plan)** |
|---|---|---|---|---|
| N | 5000 | 5000 × 5 = 25000 | 1000 × 5 = 5000 | **4000 = whole pool, shuffled per refresh** [DEV-21] |
| L / trials | 2000 steps | 2000 steps, ≤10 episodes | 1000 steps, ≤10 episodes | **M = 4 windows** (as for the operational score) |
| K | 1000 | 1000 | 100 | **1000** |
| T | 50 | 100 (`EVAL_FREQ`) | 50 | **50 PPO updates (chunks)** |
| ρ | 0.5 | 128/256 | 128/256 | **0.5** |
| N_L / NUM_STEPS | 256 / – | 256 / 256 | 256 / 512 | N_envs / 128 |
| γ / λ | – | 0.99 / 0.95 | 0.99 / 0.95 | 0.995 / 0.95 |

Cost per refresh:
- Ours: 4000 × 4 × 1200 = 19.2M scoring steps, against 50 × 2048 × 128 = 13.1M training steps (1.5×).
- References: MiniGrid 7.6×; JaxNav 0.76×.

### 2.4 SFL: our adaptation

```
every T = 50 chunks (at a chunk boundary; params frozen while scoring):
  perm ← shuffle(pool)                                  # restores the reference's random tie-breaking
  for θ in perm: M = 4 independent windows (audit.rollouts.rollout_windows: stochastic, 1% obs noise)
  p̂_θ = mean S1 (S2 stored); ℓ = p̂(1−p̂); assert no NaN
  D ← perm[argsort(ℓ, stable)[-K:]]
between refreshes, at every episode reset of slot j:
  j ≥ (1−ρ)·N_envs → uniform-with-replacement draw from D; otherwise uniform draw from the pool
  every transition is trained on (SFL has no ⊥ masking); the in-buffer audit reads ℓ stored at the refresh,
  with score age = chunks since the refresh
```

---

## 3. Deviations from the reference code

| ID | Item | Reference | Ours | Reason |
|---|---|---|---|---|
| DEV-1 | Episode and rollout structure | each update resets envs to their levels and runs `num_steps` with AutoReplay; scores cover the episodes completed in that rollout | asynchronous 1,200-step windows across 128-step chunks; per-slot carries; write at episode end | 1,200-step windows exceed the 128-step chunk |
| DEV-2 | λ-sum inside a scored episode | cut only at the rollout end, which affects only the dropped tail | cut and bootstrap at every chunk end; E1 emulates this with a phase offset φ_e; an unchunked variant is stored | the plan's protocol; Craftax precedent |
| DEV-3 | Parameter snapshots per episode | 1 (maze) | up to 11 | the policy updates between chunks; Craftax precedent |
| DEV-4 | Incomplete episodes | dropped (MNA includes them); −inf or 0 or NaN when none complete | carried until done; never scored at a chunk end | windows span chunks |
| DEV-5 | Episodes per score | all completed in the rollout (≥1 at maze defaults) | 1 per write in training (grouped within a chunk); 4 in E1 | window length |
| DEV-6 | Units | no reward normalisation; scores reuse the PPO's advantage tensor | V := σ·V̂; separate raw-unit scoring GAE | our PPO scales rewards (`ppo.py` L385–386) |
| — | Window end (mode F) | time limit terminal | **same** (headline); mode B (bootstrap V^f) stored as a variant | follows the reference |
| DEV-8 | λ | PPO λ (maze 0.98) | PPO λ = 0.95 (+ variants) | same rule, different parameter |
| DEV-9 | Success | count of r > 0 steps, or GoalR | per-window indicator, S1 or S2 | dense reward |
| DEV-10 | Learnability trials | one AutoReplay stream, ≤10 episodes | M independent windows | 1,200-step episodes |
| DEV-11 | MNA normalisation | Σ over a fixed T = 512 steps | step-pooled mean (paper Eq. 12) | T_e varies; identical ranking when T is fixed |
| DEV-12 | MNA Ĉ | 1[R_max > 0] | 1[ever S1]; Ĉ_lit logged | R_max > 0 is almost always true under dense reward |
| DEV-13 | Ensemble architecture | heads are the critic, with a shared trunk getting gradients; head 0 drives GAE | 5 extra heads on a stop-gradient trunk; main critic excluded; 64-unit ReLU | heads must not change the audited critic |
| DEV-14 | Ensemble targets (only if not fixed) | own GAE target per head | main critic's target (`ppo.py` L452–454) | current code; fix recommended |
| DEV-15 | Replay decision | one Bernoulli per update for the whole batch | one per slot per chunk | asynchronous resets |
| DEV-16 | ⊥ mechanism | whole update discarded; no gradient step on DR updates | per-transition loss mask; a gradient step every chunk on ~80% of the data; normalisers masked | asynchronous mix inside each batch |
| DEV-17 | Weights during draws | re-evaluated per draw (sequential scan, 32 draws) | frozen per chunk (~220 draws); commits in (t, j) order | buffer operations at chunk granularity |
| DEV-18 | Stale replay write | cannot happen (synchronous) | dropped if the slot's level was evicted | asynchrony |
| DEV-19 | Concurrent R_max | stale read, last write wins (`update_batch`) | sequential read of the latest value | asynchrony; the maximum is preserved |
| DEV-20 | Buffer and level source | 4000; generator; exact-field duplicate check | 1000; 4,000-level pool; duplicate check by level_id | plan C |
| DEV-21 | SFL | N fresh levels; L steps; per-update reassignment; NaN not handled | N = shuffled pool; M windows; per-reset draw; T in chunks; NaN asserted | finite pool; windows |
| DEV-22 | MaxMC-unit | — | new estimator: discounted, −inf init, V(o₀), no time average | the plan's variant |
| DEV-23 | E1 R_max history | buffer history | windows 1–4 (= the new-level branch) | fresh audit |

---

## 4. Unit tests (all float64; dones are bool)

### 4.1 Estimators

| ID | Inputs | Expected |
|---|---|---|
| E-1 PVL/L1/MaxMC, one chunk | γ = 0.5, λ = 0.5; r = [1, 0, 2, 5], V = [1, 2, 1, 3], term = [0, 0, 1, 0]; chunk ends after t = 3 with b = 4 | A = [0.6875, −1.25, 1, 4]. PVL = 0.5625; L1 = 0.979167; mean A_e = 0.145833 (PVL = ½L1 + ½mean A holds). MaxMC: R_max = 3, S = 1.666667. MaxMC-unit: G^γ = 1.5, S = 0.5. The step-3 episode is not emitted. A JaxUED port on the same data gives the same values, plus −inf for envs with no completed episode. |
| E-2 chunk cut | E-1 with a cut after t = 1, b₁ = V₂ = 1 | A = [0.625, −1.5, 1, 4]; PVL = 0.541667; L1 = 1.041667; MaxMC unchanged |
| E-3 window end | γ = λ = 0.5; r = [1, 1, 1], V = [2, 2, 2], trunc at t = 2, V^f = 2 | Mode F: A = [−0.0625, −0.25, −1]; PVL = 0; L1 = 0.4375. Mode B: A = 0; PVL = L1 = 0 |
| E-4 units | σ = 2, V̂ = [0.5, 1, 0.5, 1.5], b̂ = 2, otherwise E-1 | identical to E-1 |
| E-5 MaxMC, multi-episode | episodes (r; V): ([−1,−1]; [1,1]), ([3]; [2]), ([0,1,0]; [0,0.5,1]), ([0,0]; [1,1]) | R_max = 3; S = 3 − 1.125 = 1.875 |
| E-6 MaxMC floor | ([−1,−1]; [1,1]), ([−3]; [2]) | R_max = 0; S = −1.5 |
| E-7 R_max history | stored 5; replay group G = 3, V̄ = 1. Then an exploratory duplicate hit on the same level with G = 2, V̄ = 1 | replay: R_max = 5, S = 4, stored value 5. Duplicate hit: stored value overwritten to 2, S = 1 |
| E-8 MaxMC-unit | G^γ = [1.5, −2, 0.7], V₀ = [1, 1.2, 0.8]; then G^γ = [−2, −3], V₀ = [0, 0] | 0.5; then −2 (no floor) |
| E-9 learnability | S1 = [1, 0, 1, 1]; distances [4000, 3900, 4100, 3950], d_QSS = 4100 | S1: p = 0.75, ℓ = 0.1875. S2 = [1, 0, 1, 0]: ℓ = 0.25. All successes: 0. A literal port on 3 steps of r = 0.3 gives p = 3 and ℓ = −6, which must be rejected |
| E-10 MNA | γ = 1, λ = 0.5; r = [0, 2, −1], V = [1, 0, 0], term at t = 2 | Ĝ = [0.75, 0.5, 1]; REF sum 2.25; ours 0.75; Ĉ_lit = 1 (G = 1) |
| E-11 MNA chunk cut | E-10 with a cut after t = 1, b = 0 | Ĝ = [0.5, 0, 1]; ours 0.5 |
| E-12 MNA incomplete tail | γ = 1, λ = 0.5; V = [1, 1, 1], r = 0, term = [1, 0, 0], last value 0 | REF Ĝ = [1, 0.5, 1], raw sum 2.5; R_max = 0, so Ĉ_lit = 0 and the REF score is 0. Ours emits only step 0: m = 1 |
| E-13 MNA mode B | E-10 with trunc (not term) at t = 2, V^f = 2 | Ĝ = [0.5, 0, 0]; mean 0.166667 |
| E-14 MNA λ = 1 | E-10 at λ = 1, closed form | Ĝ = [1, 1, 1] (the code form returns NaN at λ = 1, so it must not be used) |
| E-15 MNA port equality | random data, λ ∈ {0.9, 0.95, 0.98} | closed form equals the `compute_clipped_gae(use_max_value=True)` port to 1e−12 |
| E-16 disagreement | heads: step 0 [1,2,3,4,5], step 1 [2,2,2,2,2] (term at step 1); trailing step [0,10,0,10,0] | per-step std = [1.414214, 0, 4.898979]; S = 0.707107. With ddof = 1 it would be 0.790569, which must fail |
| E-17 chunk property test | random trajectories, random cuts, ≥1 completed episode | per-episode values equal JaxUED `accumulate_rollout_stats(time_average=True)` on the concatenated per-chunk A |
| E-18 online vs offline | the same trajectory through the chunk accumulator and the E1 per-window scorer, with identical κ | identical outputs |
| E-19 E2 identity | random data | PVL_e − (½L1_e + ½mean A_e) = 0 to 1e−12 |

### 4.2 Samplers

| ID | Inputs | Expected |
|---|---|---|
| P-1 weights | scores [0.5, 2.0, 1.0], timestamps [2, 9, 5], c = 10, β = 0.3, ρ = 0.3 | ranks [3, 1, 2]; w_s = [0.022829, 0.888974, 0.088197]; w_c = [0.571429, 0.071429, 0.357143]; P = [0.187409, 0.643710, 0.168881] |
| P-2 admission (full, C = 3) | P-1 state; new S = 0.6, then S = 1.5 | candidate = argmin P = index 2. S = 0.6 is rejected (1.0 < 0.6 is False), although index 0 holds 0.5; c → 11. S = 1.5 replaces index 2, with timestamp 11 and c = 11 |
| P-3 not full | size 2, C = 3, S = −inf | slot 2 chosen; not inserted; c += 1 (our scorer must never emit −inf) |
| P-4 ties | scores [1, 1, 0] | ranks [1, 2, 3] |
| P-5 staleness fallback | all timestamps = c | w_c = mask/size |
| P-6 duplicate | level_id already at index 1 | score and extras overwritten; timestamp = c+1; c+1; size unchanged |
| P-7 fill gate | size/C < 0.5 | replay decision always False |
| P-8 eviction | slot replays index 7 (level 123); index 7 is replaced by level 456 before the window ends | write dropped |
| P-9 rank mass | 1,000 levels | top score-weight 0.8716 (β = 0.3), 0.1336 (β = 1), 0.9990 (β = 0.1) |
| S-1 top-K | ℓ = [0.25, 0, 0.1875, 0, 0.24], K = 2 | {4, 0} |
| S-2 ties | ℓ = [0, 0, 0], K = 1 | REF {2} (highest index); ours shuffles first |
| S-3 NaN | ℓ = [0.25, NaN, 0.1], K = 1 | REF selects index 1; ours raises |
| S-4 slot split | N_envs = 8, ρ = 0.5 | slots 0–3 draw new levels; slots 4–7 draw from D |

---

## 5. Where the extraction reports disagree, and the resolution

1. **Window end.** The JaxUED report says time-outs are terminal. The SFL report says "bootstrapped at truncation". The recent-scores report says MNA treats `done` as terminal.
   - The SFL report's "truncation" is the notebook's `last_val` at the rollout end, which is our chunk end. In all three codebases the time limit is terminal.
   - **Resolution: mode F is the headline; mode B is stored.**
2. **R_max on replay.** JaxUED feeds the stored running max into the score (`maze_plr.py` L604–605). The SFL report says SFL's own `jaxnav_plr.py` feeds only the current rollout; I did not re-check that.
   - **Resolution: follow JaxUED.**
3. **λ for PVL.** JaxUED maze uses 0.98, the Robust PLR paper 0.95, and MNA 0.95.
   - The rule in all of them is "use the PPO's λ". **Ours: 0.95.**
4. **MNA indexing and normalisation.** The paper's Eq. 11 starts at n = 0 and Eq. 13 uses 1/T. The code starts at n = 1 and sums over the rollout.
   - **Resolution: code weights; step-pooled mean (DEV-11).**
   - I confirmed the running-max form applies because every call site passes `use_max_value=True`.
5. **Learnability averaging.** The analysis notebook computes p̄ over workers; the training code uses a single stream. These are equivalent in our setup (one episode per window).
6. **Ensemble targets.** minimax uses per-head GAE; AutoDIME, and our current code, use a shared target.
   - **Resolution: minimax (DEV-14 if not changed).**
7. **SFL parameters.** The paper and code differ: JaxNav L = 1000 and ρ = 0.5 in the code; XLand T = 1.
   - **Resolution: paper defaults where the 4,000-level pool permits.**
8. **Rollout length.** The JaxUED report suggests `num_steps ≥ 1200`. The plan specifies 128-step chunks.
   - **Resolution: chunked, following the plan and the Craftax precedent; the unchunked variant is stored for E1.**

---

## 6. Open items needing sign-off

- **Ensemble targets:** switch `ppo.py` L452–454 to per-head GAE targets (recommended), or accept DEV-14.
- **σ in the audit:** the scorer must read `ret_rms` from the checkpoint, because `audit.rollouts.Snapshot` lacks it.
- **Chunk length:** `PPOConfig.num_steps` defaults to 64; the plan's chunk length is 128.
- **S2 definition:** define d_QSS precisely (the QSS distance in 60 s from x0, capped at top speed as in commit a86ba32).
- **SFL parameters:** N, M and T are proposals.
- **Curriculum C arms:** confirm that they use mode F (the reference). Mode B would add a deviation.
- **E2 wording:** restate the "PVL/L1 → 0.5" prediction as applying to mode B only (§0.3).