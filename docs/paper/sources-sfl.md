# SFL / "No Regrets" (Rutherford et al. 2024): how the learnability score, the SFL sampler and the score-vs-success analysis work in the paper and the code

## Sources

**Paper.** arXiv 2408.15099, version v3 (29 Oct 2024, NeurIPS 2024, licensed CC BY 4.0).
- Abstract: https://arxiv.org/abs/2408.15099
- HTML: https://arxiv.org/html/2408.15099v3
- The Figure 2/12/13 annotations below were read from the text extracted from the PDF.

**Code.** https://github.com/amacrutherford/sampling-for-learnability, pinned to HEAD `8b14e2ec07b38dc33a9df476b84faa2f3ca5fd3b` (2025-10-21). Below, `R/` stands for `https://github.com/amacrutherford/sampling-for-learnability/blob/8b14e2ec07b38dc33a9df476b84faa2f3ca5fd3b/`. Raw files are at `https://raw.githubusercontent.com/amacrutherford/sampling-for-learnability/8b14e2ec07b38dc33a9df476b84faa2f3ca5fd3b/<path>`.

**Dependencies.**
- JaxNav env from JaxMARL. `requirements.txt` asks for the latest version; I read HEAD `976aeb1` from 2026-09, so it may differ from the 2024 code. Path: `jaxmarl/environments/jaxnav/jaxnav_env.py`.
- JaxUED Maze env (`main`): `src/jaxued/environments/maze/env.py` and `wrappers/autoreplay.py`.

---

## 1. The learnability score p(1−p)

### 1.1 What the paper says

- **Definition (Sec 4.1).** If p is the agent's success rate on a level, learnability is **ℓ(θ) = p·(1−p)**. This follows Tzannetos et al. (ProCuRL).
  - It is justified as (probability of improving) × (room left to improve), i.e. the variance of a Bernoulli(p).
  - Its maximum is 0.25, at p = 0.5.
- **Multi-agent (Sec 7.2).** **ℓ(θ) = Σᵢ₌₁ⁿ pᵢ(1−pᵢ)**. This is a sum, not a mean, so its maximum is n/4.
- **Algorithm 2 (Collect learnable levels).**
  1. B ← N random levels.
  2. Roll out π for L steps on every θ in B.
  3. Set p to the success rate of each rollout.
  4. Set Learnability ← p·(1−p).
  5. Return the top K levels in B ranked by learnability.
- **Scope (Sec 9).** SFL only applies to deterministic, binary-outcome domains. The authors suggest that the continuous analogue would be the variance of the rewards obtained by playing the same level several times.

### 1.2 How the code estimates p in JaxNav (single- and multi-agent)

File: `R/sfl/train/jaxnav_sfl.py`

```python
# L196  stochastic policy sample (Gaussian, state-independent log_std; network.py L79-80)
action = pi.sample(seed=_rng)
# L206-208  auto-reset to the SAME level (start_state) on done
obsv, env_state, reward, done, info = jax.vmap(
    env.step, in_axes=(0, 0, 0, 0)
)(rng_step, env_state, env_act, start_state)
# L232-255
def _calc_outcomes_by_agent(max_steps: int, dones, returns, info):
    idxs = jnp.arange(max_steps)
    def __ep_outcomes(start_idx, end_idx):
        mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
        ...
        success = jnp.sum(info["GoalR"] * mask)
    done_idxs = jnp.argwhere(dones, size=10, fill_value=max_steps).squeeze()
    mask_done = jnp.where(done_idxs == max_steps, 0, 1)
    ...
    return {..., "success_rate": success.mean(where=mask_done), ...}
# L283-286
o = _calc_outcomes_by_agent(config["ROLLOUT_STEPS"], traj_batch.done, traj_batch.reward, info_by_actor)
success_by_env = o["success_rate"].reshape((env.num_agents, config["BATCH_SIZE"]))
learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=0)
```

The estimator, for agent i on level θ, is:

  p̂ᵢ,θ = (1/|Eᵢ,θ|) · Σ over e in Eᵢ,θ of 1[GoalR fired in episode e],  and  ℓ(θ) = Σᵢ p̂ᵢ,θ(1−p̂ᵢ,θ)

Details:

- **Which episodes count.** Eᵢ,θ is the set of episodes that finish (goal, collision or timeout) inside one L-step rollout.
  - At most 10 are counted, because of `argwhere(..., size=10)`.
  - The last episode, cut off by the end of the window, is dropped (`end_idx != max_steps`).
- **Number of rollouts per level.** There is one rollout stream per level, not parallel copies. Repeated trials come from auto-resetting to the same `start_state`.
  - JaxMARL `MultiAgentEnv.step(..., reset_state)` swaps in `reset_state` when the episode ends.
  - With the repo's L = 1000 and `max_steps` = 500 (`R/sfl/train/config/env/jaxnav.yaml` L13), each level gets between 2 and 10 counted episodes.
- **Hidden state.** The GRU state is reset at every episode boundary (`R/sfl/train/common/network.py` L30-34 resets on `last_done`), so the episodes are independent trials.
  - The environment is deterministic, so the only randomness in p̂ comes from sampling actions from the training policy.
  - The mean action is never used.
- **What "success" means** (JaxMARL `jaxnav_env.py`):
  - L292-295: `goal_reached = _check_goal_reached(new_pos, goal)`, with `_check_goal_reached` at L511-513 testing `||pos−goal|| <= goal_radius`. The goal radius is 0.3 m.
  - L368: `goal_r = goal_reached * (1 - old_goal_reached)`.
  - Timeout happens at `step >= max_steps` (L296).
  - Because agents do not evaporate, the episode ends when every agent has reached its goal, collided or timed out (L306-309).
- **Edge case.** If a level has no finished episode, `mean(where=all False)` gives NaN, and `jnp.argsort` sorts NaN to the end. The top-K slice would then pick that level as "most learnable".
  - This never happens in JaxNav or MiniGrid because L ≥ 2 × max_steps.
  - It does matter in our setting if L is no longer than the 1,200-step window.

### 1.3 MiniGrid and XLand

**MiniGrid** (`R/sfl/train/minigrid_sfl.py`):
- The environment is JaxUED `Maze` wrapped in `AutoReplayWrapper` (L85-89). The wrapper resets to `state.level` (`autoreplay.py` L35).
- Actions are sampled with `pi.sample` (L185).
- Success is defined as `goal_r = (returns > 0) * 1.0` (L227-228). In Maze, reward is positive only when the agent reaches the goal: `reward = (1.0 - 0.9*((state.time+1)/max_steps_in_episode))*fwd_pos_has_goal`, with `max_steps_in_episode` = 250.
- At most 10 episodes are counted (`size=10`, L234), and learnability is computed at L279-280.
- With L = 2000, each level gets between 8 and 10 counted episodes.

**XLand-MiniGrid** (`R/xland/training/utils.py` L162-204, `rollout_nsteps`):

```python
success=stats.success + jnp.where(timestep.discount == 0.0, 1, 0),   # L189
episodes=stats.episodes + timestep.last(),                            # L192
```

and in `R/xland/training/train_meta_task_sfl.py`:

```python
env_params.max_steps * config.sfl_rollout_factor,     # L199  (factor = 10, L69)
sucess_rate = rollout_stats.success / rollout_stats.episodes   # L206
learnability = sucess_rate * (1 - sucess_rate)                  # L207
```

- Success is termination with discount 0, i.e. the goal was reached. The episode count also includes truncated episodes.
- The hidden state is **not** reset between episodes, because this is meta-RL. So p here includes in-context adaptation over roughly 10 or more consecutive trials.
- There is no cap on the episode count.

---

## 2. The SFL sampler: N, K, ρ, T and defaults

### 2.1 Algorithm (paper Sec 4.3, Alg. 1 and 2)

Repeat until training converges:
1. Run Alg. 2 to fill the buffer: D ← top-K of N random levels, ranked by p̂(1−p̂) from L-step rollouts.
2. Then do T times:
   - D_t ← ρ·N_L levels drawn **uniformly** from D, plus (1−ρ)·N_L freshly generated random levels.
   - Collect PPO trajectories on D_t and update the policy.

### 2.2 How the code implements it (JaxNav; MiniGrid is the same)

```python
# R/sfl/train/jaxnav_sfl.py
# L290-300  N = BATCH_SIZE * NUM_BATCHES candidates; deterministic top-K
rngs = jax.random.split(rng, config["NUM_BATCHES"])
_, (learnability, env_instances) = jax.lax.scan(_batch_step, None, rngs, config["NUM_BATCHES"])
learnability = learnability.flatten()
top_1000 = jnp.argsort(learnability)[-config["NUM_TO_SAVE"]:]
# L617-627  every PPO update: fresh random levels + uniform-with-replacement draws from the buffer
reset_rng = jax.random.split(_rng, t_config["NUM_ENVS_TO_GENERATE"])
obsv_gen, env_state_gen = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
sampled_env_instances_idxs = jax.random.randint(_rng, (t_config["NUM_ENVS_FROM_SAMPLED"],), 0, num_env_instances)
# L655-662  refresh period T = EVAL_FREQ PPO updates
learnabilty_scores, instances = get_learnability_set(learnability_rng, runner_state[0].params)
runner_state_instances, metrics = jax.lax.scan(train_step, runner_state_instances, None, t_config["EVAL_FREQ"])
```

**How levels move through the buffer:**
- The buffer is fully replaced at every refresh. There is no staleness term, no persistence between refreshes and no re-scoring during the T updates.
- Top-K is a hard `argsort`, not sampling in proportion to score or rank. Draws from the buffer are uniform with replacement.

**What each PPO update looks like:**
- Every update starts on new levels; each level stays fixed for the `NUM_STEPS` = 512-step PPO rollout and auto-resets to its start.
- The buffer is scored with the parameters at the start of each block of T updates.
- `assert NUM_ENVS_FROM_SAMPLED + NUM_ENVS_TO_GENERATE == NUM_ENVS` (L77), so ρ = `NUM_ENVS_FROM_SAMPLED / NUM_ENVS`.

**XLand differences** (`train_meta_task_sfl.py`):
- Top-K is taken at L213 and the batch is mixed at L234-240.
- T counts meta-steps (`sfl_buffer_refresh_freq`, L73 and L94). One meta-step is 8192 envs × 4096 steps, which is 128 PPO updates of 32 steps each.

### 2.3 Hyperparameters: paper defaults, paper Table 4 and repo configs

| | Paper default (Sec 4.3) | Table 4: JaxNav single-agent | Table 4: JaxNav multi-agent | Table 4: MiniGrid | Table 4: XLand |
|---|---|---|---|---|---|
| N (levels scored per refresh) | 5000 | 5000 | (same) | 25000 | 40000 |
| L (rollout steps per level) | 2000 | 2000 | (same) | (2000) | 5070 |
| K (buffer size) | 1000 | 100 | 100 | 1000 | 8192 |
| T (refresh period) | 50 | 50 | 50 | 100 | 4 meta-steps |
| ρ (share of batch from buffer) | 0.5 | **1.0** | **1.0** | 0.5 | 1.0 |
| N_L (levels per batch) | 256 | 256 | 256 | 256 | 8192 |

What the released code actually uses:

| Setting | `jaxnav-sfl.yaml` L17-27 + `ippo-jaxnav.yaml` | `minigrid-sfl.yaml` L13-21 + `ippo-minigrid.yaml` | `train_meta_task_sfl.py` L48, L68-74 |
|---|---|---|---|
| N | `BATCH_SIZE` 1000 × `NUM_BATCHES` 5 = **5000** | 5000 × 5 = **25000** | `sfl_batch_size` 40000 × `sfl_num_batches` 1 = **40000** |
| L | `ROLLOUT_STEPS` **1000** (paper: 2000) | **2000** | 10 × `max_steps` (paper: 5070) |
| K | `NUM_TO_SAVE` **100** | **1000** | `sfl_buffer_size` **8192** |
| T | `EVAL_FREQ` **50** PPO updates | **100** | `sfl_buffer_refresh_freq` **1** (paper: 4) |
| ρ | 128/256 = **0.5** (paper: 1.0) | 128/256 = **0.5** | 8192/8192 = **1.0** |
| PPO steps per update | 512; γ = 0.99, λ = 0.95 | 256 | 32 |

**Discrepancies between paper and code:**
- The JaxNav config (single- and multi-agent, launched by the sweep files without overrides) uses L = 1000 and ρ = 0.5. Table 4 says L = 2000 and ρ = 1.0.
- The XLand default T is 1, while the paper says 4.
- MiniGrid was corrected to match the paper in commit `09946071de` on 2025-03-06 (`BATCH_SIZE` changed from 1000 to 5000). JaxNav was never changed.
- The ablation figures 28 and 29 say their "default" is N = 5000, T = 50/100, ρ = 0.5, K = 1000, L = 2000.

**Ablations (App. I.1, JaxNav):**
- Larger N helps.
- L = 1000 is slightly worse than 2000; 4000 gives no extra gain.
- A smaller K is better.
- A larger T hurts.
- A higher ρ is better.
- Sampling the buffer in decreasing order of learnability beats uniform sampling.
- In App. I.3, a peak at p = 0.6 does best; Linear(0) (closest to regret) matches SFL on JaxNav but struggles on XLand.

**Cost per refresh:**
- JaxNav (repo config): 5000 × 1000 = 5M rollout steps, against 50 × 256 × 512 ≈ 6.6M training steps between refreshes.
- MiniGrid: 25000 × 2000 = 50M rollout steps.
- App. H reports compute time. SFL ran fewer PPO updates on single-agent JaxNav and XLand to match compute.

---

## 3. How No Regrets checked PVL/MaxMC against success rate

### 3.1 The protocol (Sec 4.2, with the code that produced Figure 2)

**Policy and checkpoint.**
- One policy: the top-performing seed of **PLR-MaxMC on single-agent JaxNav**, wandb run `alex-plus/multi_robot_ued/5yvzkn8p`.
- One checkpoint: the **final** one, artifact `{run.name}-checkpoint:latest`, loaded in `R/sfl/deploy/sample_uniform_map_set.py` L121 and L130 and in the notebook `R/sfl/deploy/deploy_on_sampled_and_calc_regret.ipynb` cell 6 L2 and L14.
- No other checkpoints, seeds, methods or environments were used.

**Level set** (`sample_uniform_map_set.py`):
- Parameters: `NUM_ENVS_TO_SAMPLE=1000`, `N_BINS=10`, `N_ENVS_PER_BIN=250` (L112-118), and `NUM_STEPS = env.max_steps * N_BINS`, which is 5000 steps (L140).
- Up to 50 episodes per level are counted (`size=50`, L255).
- Levels are binned by mean success into (−∞, 0.1], (0.1, 0.2], …, (0.9, 1.0] and taken in order of appearance until each bin holds 250 (L276-290).
- The result is 2500 levels **stratified to be uniform in success rate**, not drawn from the training distribution. The binning used the same policy that was later scored.
- No solvability filter is applied, so the bottom bin may include unsolvable levels.
- The set is released as `R/sfl/data/test_sets/sampled_tc_uniform2500e_1a.pkl`.

**Rollouts** (notebook cells 4, 8 and 10):
- `N_PARRALEL = 10`, `ROLLOUT_LEN = 512`.
- Each level is copied to 10 parallel workers, and each worker runs 512 steps with a stochastic policy and auto-reset.
- GAE uses the run's γ = 0.99 and λ = 0.95, bootstraps `last_val` at step 512, and cuts at `done["__all__"]` (cell 10 L95-119).
- Success uses the same `_calc_outcomes_by_agent` (cell 10 L122-152, `size=10`).

**Scores** (cells 11 and 13, using `R/sfl/util/jaxued/jaxued_utils.py`, a modified copy of minimax `ued_scores.py`):

```python
# jaxued_utils.py L33-36 (max_val initialised to ZEROS), L23, L40, L44-57
max_val = jax.tree_map(lambda x, y: (1 - done) * x + done * jnp.maximum(x, y), max_val, val)
def compute_max_returns(dones, rewards):
    _, max_returns, _ = accumulate_rollout_stats(dones, rewards, time_average=False)
def max_mc(dones, values, max_returns, incomplete_value=-jnp.inf):
    mean_scores, _, episode_count = accumulate_rollout_stats(dones, max_returns[None, :] - values, time_average=True)
def positive_value_loss(dones, advantages, incomplete_value=-jnp.inf):
    mean_scores, _, episode_count = accumulate_rollout_stats(dones, jnp.maximum(advantages, 0), time_average=True)
```

```python
# notebook cell 11 L3-6 ; cell 13 L1-7
mean_success_by_actor = success_by_actor.reshape((env.num_agents, num_envs, N_PARRALEL)).mean(axis=-1)
learn_prob_by_level = (mean_success_by_actor * (1 - mean_success_by_actor)).sum(axis=0)
max_mc_by_actor = max_mc_all.reshape((env.num_agents, num_envs, N_PARRALEL)).mean(axis=-1)
pvl_by_actor = pvl_by_all.reshape((env.num_agents, num_envs, N_PARRALEL)).mean(axis=-1)
```

In formulas, per worker, with completed episodes k = 1..M, each of length |k|:

- **Maximum return.** R_max = max(0, maxₖ Σ_{t∈k} r_t). It is **undiscounted**, floored at 0 because the accumulator starts at zero, and computed **only from this worker's 512-step rollout**.
- **MaxMC** = (1/M) Σₖ (1/|k|) Σ_{t∈k} (R_max − V(o_t)).
- **PVL** = (1/M) Σₖ (1/|k|) Σ_{t∈k} max(Â_t^GAE, 0).
- In both, the truncated last episode is ignored, and the score is −∞ when M = 0.
- **Level score** = the mean over the 10 workers.
- **Learnability** = p̄(1−p̄), where p̄ is the mean success over the 10 workers. It is not the mean of each worker's p(1−p).

The same functions are used during PLR training (`R/sfl/train/jaxnav_plr.py` L606-610 and L645-651). When a level is replayed, the score still uses the max return from the current rollout. The buffer's stored `max_return` is updated with a running max but never fed into `compute_score` (L647 versus L648).

**The statistic** (notebook cell 16 L66-90):
- `r, p = pearsonr(x, y)` between **mean success rate** (x) and the score (y), over the 2500 levels.
- The plot overlays a linear fit (`np.polyfit` with degree 1) and a quadratic fit (`Polynomial.fit` with degree 2).
- **No statistic is ever computed between MaxMC/PVL and learnability or regret.** The claim that "MaxMC does not correlate with learnability" rests only on looking at the scatter plots.

**Reported values.** These come from the PDF figure annotations and match the notebook outputs in cells 18-20.

| Figure | Score | Against | Pearson r | p-value |
|---|---|---|---|---|
| Fig 2a | MaxMC | mean success | 0.61 | 1e-251 |
| Fig 2b | PVL | mean success | 0.51 | 2.9e-165 |
| Fig 12 | MaxMC | mean return | 0.61 | 6.2e-250 |
| Fig 12 | PVL | mean return | 0.42 | 3.2e-107 |
| Fig 13 | L1 value loss | success / return | 0.07 / 0.07 | 2e-4 / 6e-4 |

The text calls the MaxMC correlation "slight", but the reported r is 0.61.

### 3.2 Other related analyses in the paper

- **G.3, Table 5.** Mean and median learnability and success of PLR/ACCEL/SFL buffer levels, averaged over training, mean (std) over 3 seeds.
  - PLR-MaxMC buffer: learnability 0.02, success 0.84.
  - SFL-selected levels: learnability 0.22, success 0.59.
  - The paper says there is "no statistically significant correlation" between learnability and PVL/MaxMC, but names no statistic, and the public repo has no code for it.
- **"Perfect Regret (Oracle)" (Sec 7.1).** This is SFL with score 1 − p(success), sampling only solvable levels. It is a training baseline, not a ground truth for checking scores, and it measures regret in success-rate units.
  - Its training code is not public; only evaluation CSVs exist under `rsim_regret_single_v6` (`R/sfl/deploy/eval_jaxnav_single_agent_2_analyse.py` L46).
- **Unused RRT\* "true regret" code.** `R/sfl/train/ued/plr_manager.py` contains an RRT\*-based oracle:
  - `calculate_rrt_return` (L161-195) sums the shaped reward along an RRT\* path, undiscounted and kinematic only.
  - `true_regret` (L279-305) computes `ep_true_regret = true_reward - reward`.
  - It imports `sfl.env.maps`, which does not exist, is not imported by any training or analysis script, and is not mentioned in the paper.

### 3.3 What No Regrets did not do, which is what our audit adds

1. **No true regret.** The reference was success rate or return (Pearson) plus a visual comparison with p(1−p). There was never an optimal-control U(π\*_θ) − U(π_θ), and learnability itself was never checked against regret.
2. **One snapshot.** One checkpoint (the final one), one seed, one method (PLR-MaxMC) and one environment (single-agent JaxNav). Nothing over the course of training or across seeds.
3. **Non-representative level set.** The 2500 levels were stratified by success rate using the same policy, possibly including unsolvable ones. They were not the training generator's distribution nor the curriculum buffer.
4. **Pearson r only.** No rank statistics (Spearman/Kendall), no top-K or selection-overlap metric (PLR selects by rank or top-k), and no confidence intervals.
5. **Binary outcome with shaped reward and γ = 0.99.** No dense-reward, continuous-action control task with long horizons.

### 3.4 Deviations to document for our setting (1,200-step windows, γ = 0.995, dense reward)

- **p is undefined without a binary success.** We need either a declared success event, or the paper's own Sec 9 analogue (variance of returns over repeated plays of the same level).
- **Number of trials.** The reference gets repeated trials by auto-resetting inside L steps. With 1,200-step episodes we need M parallel repeats; M = 10 matches the analysis notebook. We must also handle the case of no finished episode (the NaN/−∞ issue).
- **Stochastic actions.** Roll out with sampled actions, as the reference does, not the mean action.
- **MaxMC as coded:**
  - R_max is undiscounted and floored at 0, while V is discounted. That gap grows at γ = 0.995 over 1,200 steps.
  - R_max comes from the current rollout only, then is time-averaged per episode, averaged over completed episodes, and finally averaged over workers.
- **PVL.** Per-episode time average of the positive part of GAE (γ, λ from the run config), bootstrapped at truncation.