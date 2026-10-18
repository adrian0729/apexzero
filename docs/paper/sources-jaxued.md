# JaxUED audit: how PVL and MaxMC scores, the PLR level buffer and the Maze environment work

I read everything below from source. The commit is `DramaCow/jaxued@main`, SHA `0f8f1284677375b889e4f13a32c9617cd009f8c4` (2025‑09‑09), package version `0.0.2`.

Permalink base: `https://github.com/DramaCow/jaxued/blob/0f8f1284677375b889e4f13a32c9617cd009f8c4/`. Line numbers refer to that commit. `utils.py` and `level_sampler.py` are unchanged in substance since 2024‑02; the last commit only swapped `jax.tree_map` for `jax.tree_util.tree_map`.

---

## 1. Level scores: `src/jaxued/utils.py`

The file header (L4‑6) says it is "a modified version of" minimax `src/minimax/util/rl/ued_scores.py` @ `2ae9e04`. I checked minimax's `compute_episodic_stats`, and the core logic is the same, including the zero‑initialised max.

### 1.1 Core accumulator (L8‑42)

```python
def accumulate_rollout_stats(dones, metrics, *, time_average):
    def iter(carry, input):
        sum_val, max_val, accum_val, step_count, episode_count = carry
        done, step_val = input
        accum_val = jax.tree_util.tree_map(lambda x, y: x + y, accum_val, step_val)   # L13
        step_count += 1                                                                # L14
        if time_average:
            val = jax.tree_util.tree_map(lambda x: x / step_count, accum_val)         # L18
        else:
            val = accum_val
        sum_val = jax.tree_util.tree_map(lambda x, y: x + done * y, sum_val, val)     # L22
        max_val = jax.tree_util.tree_map(lambda x, y: (1 - done) * x + done * jnp.maximum(x, y), max_val, val)  # L23
        episode_count += done                                                          # L25
        accum_val = jax.tree_util.tree_map(lambda x: (1 - done) * x, accum_val)       # L27
        step_count = (1 - done) * step_count                                           # L28
        ...
    zeros = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x[0]), metrics)           # L33
    (sum_val, max_val, _, _, episode_count), _ = jax.lax.scan(iter,
        (zeros, zeros, zeros, jnp.zeros(batch_size, dtype=jnp.uint32), jnp.zeros(batch_size, dtype=jnp.uint32)),
        (dones, metrics))                                                              # L34-38
    mean_val = jax.tree_util.tree_map(lambda x: x / jnp.maximum(episode_count, 1), sum_val)  # L40
    return mean_val, max_val, episode_count
```

Inputs have shape `(num_steps, num_envs)`. `dones[t]` is the done flag returned by the transition at step `t`.

- **Per‑episode value.** For each episode that completes inside the rollout, the value is either the sum of the metric over its steps (`time_average=False`) or the mean over its steps (`time_average=True`). The step count includes the terminal transition.
- **Averaging over episodes.** Completed episodes are averaged with equal weight per episode, not per step (L40).
- **Trailing incomplete episode.** The episode still running when the rollout ends is discarded completely. `sum_val`, `max_val` and `episode_count` change only when `done=1`.
- **`max_val` starts at 0 (L33/L36), not at `-inf`.** Any max computed here is therefore `max(0, best completed episode)`.

### 1.2 Score functions (L44‑57)

```python
def compute_max_returns(dones, rewards):                                  # L44
    _, max_returns, _ = accumulate_rollout_stats(dones, rewards, time_average=False)
    return max_returns

def max_mc(dones, values, max_returns, incomplete_value=-jnp.inf):         # L51
    mean_scores, _, episode_count = accumulate_rollout_stats(dones, max_returns[None, :] - values, time_average=True)
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)

def positive_value_loss(dones, advantages, incomplete_value=-jnp.inf):     # L55
    mean_scores, _, episode_count = accumulate_rollout_stats(dones, jnp.maximum(advantages, 0), time_average=True)
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)
```

**The exact formulas.** For env (level) i, let 𝓔ᵢ be the episodes that complete within the `num_steps` rollout, let E = |𝓔ᵢ|, and let Tₑ be the number of transitions in episode e, including the terminal one.

- **PVL:** `S_PVL = (1/E) Σ_e (1/T_e) Σ_{t∈e} max(Â_t, 0)`. If E = 0, the score is `-inf`.
- **MaxMC:** `S_MaxMC = (1/E) Σ_e (1/T_e) Σ_{t∈e} (R_max − V(s_t))`. If E = 0, the score is `-inf`.
- **R_max:** `R_max = max(0, max_{e∈𝓔} Σ_{t∈e} r_t)`. This is the **undiscounted** episodic return, clamped below at 0. On replay it is also combined with the stored value (see 1.4).
- **The two papers' versions.** Robust PLR (Jiang et al. 2021, arXiv 2110.02439, §4.2) writes `1/T Σ_{t=0}^{T} max(Σ_{k=t}^{T} (γλ)^{k−t} δ_k, 0)` for PVL and `1/T Σ_{t=0}^{T} (R_max − V(s_t))` for MaxMC, with R_max = "highest return achieved" on the level. JaxUED implements these as a per‑episode time average followed by an equal‑weight mean across the completed episodes in the rollout.

### 1.3 Advantages and discounting: `examples/maze_plr.py` L42‑77

```python
def compute_gae_at_timestep(carry, x):
    gae, next_value = carry
    value, reward, done = x
    delta = reward + gamma * next_value * (1 - done) - value          # L66
    gae = delta + gamma * lambd * (1 - done) * gae                    # L67
    return (gae, value), gae
_, advantages = jax.lax.scan(compute_gae_at_timestep, (jnp.zeros_like(last_value), last_value),
                             (values, rewards, dones), reverse=True, unroll=16)   # L70-76
return advantages, advantages + values                                # L77
```

- `Â_t = Σ_k (γλ)^{k−t} δ_k`, truncated at the episode's `done`, with `δ_t = r_t + γ V(s_{t+1})(1−d_t) − V(s_t)`.
- PVL uses the **same λ as PPO**: `gae_lambda = 0.98` in the maze defaults (L871). There is no separate scoring λ.
- `last_value` (L142‑145) bootstraps only the trailing incomplete episode, and PVL discards that episode anyway.
- **Timeouts count as terminal.** `done` covers both goal and time limit (`env.py` L167‑170). No episode is ever bootstrapped across a truncation.
- The value convention is `V(s_t) = E[Σ_{k≥0} γ^k r_{t+k}]`, with γ = 0.995 by default (L868). MaxMC therefore subtracts a **discounted** V from an **undiscounted** R_max.

### 1.4 How R_max is stored and reused (`maze_plr.py`)

- **Init (L499):** `sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})`. This adds a per‑slot `levels_extra["max_return"]`.
- **New levels, DR branch (L545‑548):**
  ```python
  advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
  max_returns = compute_max_returns(dones, rewards)
  scores = compute_score(config, dones, values, max_returns, advantages)
  sampler, _ = level_sampler.insert_batch(sampler, new_levels, scores, {"max_return": max_returns})
  ```
  R_max here comes from this rollout only, so it is ≥ 0.
- **Replay (L603‑606):**
  ```python
  max_returns = jnp.maximum(level_sampler.get_levels_extra(sampler, level_inds)["max_return"], compute_max_returns(dones, rewards))
  scores = compute_score(config, dones, values, max_returns, advantages)
  sampler = level_sampler.update_batch(sampler, level_inds, scores, {"max_return": max_returns})
  ```
  This gives the **all‑time max since the level entered the buffer**. The updated R_max, including this rollout, is used for the score.
- **ACCEL mutate (L665‑668):** same as the new‑level branch, with fresh R_max.
- **Lost history 1 (duplicate check).** When `duplicate_check=True` and a re‑generated level is already in the buffer, `insert` calls `self.update(...)` (`level_sampler.py` L172‑173). That **overwrites** `max_return` with the new rollout's value instead of taking the max.
- **Lost history 2 (same level twice in a batch).** If one replay batch samples the same slot twice, `update_batch` scans in order and the last write wins (L271‑274).
- `compute_score` (L389‑395) dispatches between `"MaxMC"` → `max_mc(dones, values, max_returns)` and `"pvl"` → `positive_value_loss(dones, advantages)`.

### 1.5 Incomplete episodes and `-inf`

- Each branch resets every env to its level at the start of the rollout (`env.reset_to_level`, L529/L588/L648) with a fresh LSTM carry, so **episodes never span updates**. `AutoReplayWrapper` (`wrappers/autoreplay.py` L34‑41) restarts the **same** level on `done`, so one `num_steps` rollout can contain several episodes of that level.
- If no episode finishes, the score is `-inf` (`incomplete_value`).
- **New or mutated levels:** `_insert_new` admits only if `sampler["scores"][idx] < score` (`level_sampler.py` L377). Empty slots are initialised to `-inf` (L87), and `-inf < -inf` is False. A `-inf` level is therefore **never inserted, even into a non‑full buffer**, but it still increments `episode_count` (L392).
- **Replay:** `update` sets the score unconditionally (L252), so a replayed level with no completed episode drops to `-inf` and gets the lowest rank.
- In the maze defaults, `num_steps=256 > max_steps_in_episode=250`, so at least one episode always completes and `-inf` never occurs in practice.

---

## 2. PLR / Robust PLR level sampler: `src/jaxued/level_sampler.py`

### 2.1 Constructor defaults (L41‑65)

`LevelSampler(capacity, replay_prob=0.95, staleness_coeff=0.5, minimum_fill_ratio=1.0, prioritization="rank", prioritization_params=None, duplicate_check=False)`. With `rank` and no parameters, the temperature defaults to 1.0 (L59‑63). These library defaults are **not** what the examples use (see 2.7).

The state (L85‑93) holds `levels`, `scores` (initialised to `-inf`), `timestamps` (initialised to 0), `size`, `episode_count`, and optionally `levels_extra`.

### 2.2 Sampling weights (L276‑341)

```python
return (1 - self.staleness_coeff) * w_s + self.staleness_coeff * w_c                    # L290
# rank:
ord = (-jnp.where(mask, sampler["scores"], -jnp.inf)).argsort()                         # L312
ranks = jnp.empty_like(ord).at[ord].set(jnp.arange(len(ord)) + 1)                       # L313
w_s = jnp.where(mask, 1 / ranks, 0) ** (1 / temperature)                                # L315
w_s = w_s / w_s.sum()                                                                   # L316
# staleness:
staleness = sampler["episode_count"] - sampler["timestamps"]                            # L338
w_c = jnp.where(mask, staleness, 0)                                                     # L339
w_c = jax.lax.select(w_c.sum() > 0, w_c / w_c.sum(), mask / sampler["size"])            # L340
```

- **Formula:** `P(l_i) = (1−ρ)·h(S_i)^{1/β} / Σ_j h(S_j)^{1/β} + ρ·(c − C_i) / Σ_j (c − C_j)`, with `h = 1/rank` and rank 1 = highest score. This matches Robust PLR's Appendix B.
- **Ties:** broken by buffer index, because `argsort` is stable by default in current JAX.
- There is also `topk`: a softmax over the top‑k scores (L317‑321).

**How staleness is counted.** `c = episode_count`. It increases by 1 for:
- each replay level sampled (L123‑127, which also sets that slot's timestamp to the new count);
- each insertion attempt, accepted or not (L392);
- each duplicate hit (L174‑175).

A successful insertion sets the timestamp to `episode_count+1` (L384). With 32 envs, `c` therefore grows by 32 per update. It counts level‑episodes in this sense, not environment steps.

### 2.3 Replay decision (L96‑108)

`(size/capacity >= minimum_fill_ratio) & (uniform() < replay_prob)`. Replay levels are drawn **with replacement**, one `jax.random.choice(rng, capacity, p=weights)` per env (L121‑122, L143).

### 2.4 Admission rule (L375‑404)

```python
idx = self._get_next_idx(sampler)
replace_cond = sampler["scores"][idx] < score                                           # L377
...  # on replace: set level, score, timestamp=episode_count+1, size=min(size+1, capacity), levels_extra
new_sampler["episode_count"] += 1                                                       # L392
def _get_next_idx(self, sampler):
    return jax.lax.select(sampler["size"] < self.capacity, sampler["size"],
                          self.level_weights(sampler).argmin())                         # L399-404
```

- **Buffer not full:** use the next empty slot. Admission requires `score > -inf`.
- **Buffer full:** the candidate slot is the one with the **lowest combined replay weight** (score rank plus staleness), not the lowest score. It is replaced only if its stored score is strictly below the new score. This matches the paper's "minimal replay support … if S(θ_min) < S_θ".
- **With `duplicate_check`:** if the level already exists (exact match on every Level field, L195‑209), its score and extras are overwritten and its timestamp refreshed. Nothing new is added.

### 2.5 Updating scores on replay

**Scores are updated on replay:** `update_batch` overwrites score and `max_return` for every replayed slot (L237‑274, called at `maze_plr.py` L606). Replay sampling also refreshes the timestamp (L126).

### 2.6 Training loop and branching (`maze_plr.py` L699‑718)

```python
if config["use_accel"]:
    s = train_state.update_state
    branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s
else:
    branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)
return jax.lax.switch(branch, [on_new_levels, on_replay_levels, on_mutate_levels], rng, train_state)
```

- **Robust PLR (PLR⊥), the default:** `exploratory_grad_updates=False`. New‑level and mutation rollouts run PPO with `update_grad=False` (L563, L683), so they only score. Only replay updates the policy (L621).
- **PLR:** add `--exploratory_grad_updates`.
- **ACCEL:** add `--use_accel`. After every replay step comes a mutate step. It edits the batch just replayed (`replay_last_level_batch`, L646‑647) using `make_level_mutator_minimax(100)` with `num_edits=5`. Each edit is uniform over {NO_OP, FLIP_WALL, MOVE_GOAL} (`util.py` L127‑198). The children are scored and admitted with `insert_batch` under the same rule.

### 2.7 Default hyperparameters

The maze argparse defaults are in `maze_plr.py` L860‑891 and match `docs/maze_plr.md`. ACCEL runs from the same file; there is no separate config.

| Param | `maze_plr.py` (PLR / PLR⊥ / ACCEL) | LevelSampler class default | Robust PLR paper, MiniGrid (Table 4) | ACCEL paper, MiniGrid (Table 11) |
|---|---|---|---|---|
| score_function | **MaxMC** (choices: MaxMC, pvl) | – | MaxMC | positive value loss |
| level_buffer_capacity | 4000 | (required) | 4000 | 4000 |
| replay_prob | 0.8 | 0.95 | 0.5 | 0.8 |
| staleness_coeff ρ | 0.3 | 0.5 | 0.3 | 0.3 |
| prioritization / temperature β | rank / 0.3 | rank / 1.0 | rank / 0.3 | rank / 0.3 |
| topk_k | 4 | 1 | – | – |
| minimum_fill_ratio | 0.5 | 1.0 | – | – |
| buffer_duplicate_check | True | False | – | – |
| exploratory_grad_updates | False (i.e. PLR⊥) | – | – | – |
| use_accel / num_edits | False / 5 | – | – | – / 5 |
| gamma | 0.995 | – | 0.995 | 0.995 |
| gae_lambda | **0.98** | – | 0.95 | 0.95 |
| num_steps × num_train_envs | 256 × 32 | – | 256 × 32 | 256 × 32 |
| PPO epochs / minibatches | 5 / 1 | – | – | 5 / 1 |
| clip / entropy / critic coef | 0.2 / 1e‑3 / 0.5 | – | – | – / 0.0 / – |
| lr / grad clip / num_updates | 1e‑4 (linear decay) / 0.5 / 30000 | – | – | 1e‑4 / – / – |
| agent_view_size / n_walls | 5 / 25 | – | – | – |

The two paper columns come from ar5iv renderings of the papers, read through WebFetch's summariser rather than from the PDFs. In particular, "–" in the ACCEL column means not reported, not a confirmed blank.

Other example scripts, for comparison:
- **Gymnax** (`examples/gymnax/gymnax_plr.py` L1004‑1038, which includes continuous Pendulum with `MultivariateNormalDiag`): num_steps 512, lr 2.5e‑4, num_minibatches 4, num_updates 3500. The PLR settings are the same as maze.
- **Craftax** (`examples/craftax/craftax_plr.py` L922‑940): num_steps 64 × outer_rollout_steps 64, 1024 envs, gae_lambda 0.9, replay_prob 0.5, temperature 1.0. The rest is the same.

---

## 3. Maze environment: `src/jaxued/environments/maze/`

- **Grid:** `Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)` (`maze_plr.py` L456).
  - Generated levels are always 13×13 (`make_level_generator(13, 13, n_walls=25)`).
  - Wall cells are drawn **with replacement** (`util.py` L26‑28), so there are at most 25 distinct walls.
  - Agent position, direction and goal are uniform over free cells (L32‑41).
  - Solvability is **not** guaranteed.
  - There are no border walls inside the grid. Moves are clipped to the grid (`env.py` L220‑223), and observation padding uses walls (L273‑284).
- **Observation:** egocentric `k×k×3` (object, colour, state) with k = `agent_view_size = 5`, divided by 10. Also includes `agent_dir`. Implemented in `env.py` L184‑216; the network is a 3×3 conv(16) + LSTM 256 (`maze_plr.py` L301‑310).
- **Actions:** `Discrete(7)`, of which only left, right and forward have any effect (`env.py` L11‑18, L144‑146).
- **Max steps:** `EnvParams.max_steps_in_episode = 250` (`env.py` L76‑77). Termination is `time >= max_steps OR terminal` (L167‑170).
- **Reward (`env.py` L235‑238):**
  ```python
  reward = (1.0 - 0.9*((state.time+1)/params.max_steps_in_episode))*fwd_pos_has_goal
  ```
  - The reward is paid on the action that moves forward into the goal. The goal cell blocks movement, and `terminal=fwd_pos_has_goal`.
  - `state.time` is the pre‑step time, so reaching the goal on the n‑th action pays `1 − 0.9·n/250`.
  - `penalize_time=False` would give a reward of 1 instead.
- **Eval:** solve rate is `cum_rewards > 0` over 10 attempts on the 8 prefab mazes (`maze_plr.py` L757, L848‑857; prefabs in `level.py` L145‑271).

### 3.1 Exact optimum: `env_solved.py` (`MazeSolved`, not used by any example)

```python
def _precompute_min_steps_to_goal(self, level):                                          # L56
    wall_values = jnp.repeat(jnp.where(level.wall_map, jnp.inf, -jnp.inf)[None, ...], 4, axis=0)
    def compute_next(values):
        fwd_values = jnp.array([ jnp.roll(values[0], -1, axis=1).astype(float).at[:,-1].set(jnp.inf),
                                 jnp.roll(values[1], -1, axis=0).astype(float).at[-1,:].set(jnp.inf),
                                 jnp.roll(values[2], 1, axis=1).astype(float).at[:,0].set(jnp.inf),
                                 jnp.roll(values[3], 1, axis=0).astype(float).at[0,:].set(jnp.inf) ])
        for i in range(4):
            new_values = new_values.at[i].set(jnp.min(jnp.array([values[i], values[i-1] + 1, values[(i+1)%4] + 1, fwd_values[i] + 1]), axis=0))
        return jnp.maximum(new_values, wall_values)
    values = jnp.full((4, self.max_height, self.max_width), jnp.inf).at[:, level.goal_pos[1], level.goal_pos[0]].set(0)
    return jax.lax.while_loop(cond_fn, body_fn, (values, compute_next(values)))[0]       # L81-82

def optimal_value(self, state, gamma, params):                                           # L44-51
    n = self.min_steps_to_goal(state)          # state.min_steps[agent_dir, y, x]
    if self.penalize_time:
        N = state.time + n
        value = (1.0 - 0.9*((N+1)/params.max_steps_in_episode)) * gamma ** n
    ...
    return jnp.where(n != jnp.inf, value, 0)
```

**What the DP computes.** It is unit‑cost Bellman–Ford (BFS) on the 4 × 13 × 13 (direction, y, x) state space, where turning costs 1 and moving forward costs 1. `n` is the exact minimum number of actions, counting the final forward into the goal. The JaxUED paper describes this as a JIT shortest‑path algorithm that runs in O(N²).

**Possible off‑by‑one in `optimal_value` (my derivation; I did not run it).** Under the env's reward timing and the GAE value convention (§1.3), the reward lands at action index n−1. The correct values are:
- discounted: `V*(s) = γ^{n−1}·(1 − 0.9·(t+n)/T)`;
- undiscounted: `R* = 1 − 0.9·(t+n)/T`.

`optimal_value` instead returns `γ^n·(1 − 0.9·(t+n+1)/T)`, which is one step off in both the time penalty and the discount. It also never checks that `t + n ≤ T`; when that fails, the true optimum is 0.

My recommendation is to reuse `min_steps` but compute the value yourself:
- exact regret: `Regret(θ) = V*(s₀) − V^π(s₀)`;
- undiscounted variant, matching how R_max is computed: `R*(θ) − E_π[G]`;
- unsolvable levels have `n = inf`, so V* = 0.

---

## 4. Consequences for the racing audit

Each of these is a reference behaviour to replicate exactly, or to document wherever you deviate.

1. **Rollout length.** With the maze default `num_steps=256` and 1,200‑step windows, no episode completes, so every score is `-inf`, no level is ever admitted, and replayed levels fall to `-inf`. You need `num_steps ≥ 1200` per scoring rollout. Alternatively, copy Craftax's pattern (`craftax_plr.py` L224‑291): `outer_rollout_steps` chunks of `num_steps`, with PPO updates between chunks and GAE bootstrapped at chunk ends. Scores are then computed on the concatenated trajectory, so V and Â come from several policy snapshots.
2. **R_max is clamped at 0** (zero‑initialised `max_val`). This also affects Pendulum, whose rewards are always ≤ 0. If racing returns can be negative, the faithful reference MaxMC uses `max(0, G)`. A "fixed" variant would initialise to `-inf`; document which one you report.
3. **Scale mismatch in MaxMC.** R_max is undiscounted while V is discounted. With γ = 0.995 (effective horizon 200, γ^1200 ≈ 0.0024), dense‑reward MaxMC will be dominated by roughly `G_undisc − V_disc`.
4. **Window ends are terminal.** The 1,200‑step window end becomes `done=1` with no bootstrap in GAE (timeouts are terminal), and PVL inherits that.
5. **Averaging.** Each completed episode is time‑averaged, then episodes are averaged with equal weight. The trailing incomplete episode is ignored.
6. **Buffer rules.** Eviction targets the lowest combined weight, not the lowest score. Staleness ticks once per level‑episode sampled or attempted. With duplicate‑check on, R_max is overwritten.
7. **DCD is different.** The `facebookresearch/dcd` `level_replay/level_sampler.py` behaves differently (partial‑episode merging across rollouts, `after_update` flushes, the alpha EMA). For example, MaxMC there is `grounded_signed_value_loss` with `grounded_value = rewards[start_t:t].sum(0)` (L358‑393, L551‑571). Don't treat JaxUED as interchangeable with it; it needs its own audit.

**Files:**
- `src/jaxued/utils.py`
- `src/jaxued/level_sampler.py`
- `examples/maze_plr.py`
- `examples/gymnax/gymnax_plr.py`
- `examples/craftax/craftax_plr.py`
- `src/jaxued/environments/maze/{env.py, env_solved.py, level.py, util.py}`
- `src/jaxued/wrappers/autoreplay.py`
- `docs/maze_plr.md`

**Papers:**
- https://arxiv.org/abs/2403.13091 (JaxUED)
- https://arxiv.org/abs/2110.02439 (Robust PLR)
- https://arxiv.org/abs/2203.01302 (ACCEL)