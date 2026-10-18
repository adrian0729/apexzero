# Recent UED level scores: exact definitions for the regret audit

Code was read at pinned commits and is linked as permalinks. Every equation below is quoted from the paper's own arXiv HTML. All symbols are as the paper uses them.

---

## (a) MNA: Maximised Negative Advantage (Mead, Lacerda, Foerster, Hawes)

- **Paper:** "Improving Regret Approximation for Unsupervised Dynamic Environment Generation", arXiv:2601.14957 v1 (21 Jan 2026), https://arxiv.org/abs/2601.14957. MNA is §4 and §4.1. The venue is not listed on the arXiv page.
- **Code:** https://github.com/HarryMJMead/Dynamic-Environment-Generation-for-UED at commit `12340c7fe443278f4033bab977054af1e40b8188`.

### Paper formulas (verbatim)

- **Eq. 8:**
  `V^max_n(s_t) = max( V(s_t), …, γ^n V(s_{t+n}) + Σ_{k=t}^{t+n-1} γ^{k-t} r(s_k,a_k) )`
- **Eq. 9:**
  `Regret ≥ −V(s_0) + V^max_T(s_0)`
  This is a lower bound only if V is the true value function and the environment is deterministic. With the learned V̂, the paper says "the inequality in Equation 7 does not necessarily hold".
- **Eq. 10:**
  `Ĝ_t^(n) = −( γ^n V(s_{t+n}) + Σ_{k=t}^{t+n-1} γ^{k-t} r(s_k,a_k) ) + V̂^max_n(s_t)`
- **Eq. 11:**
  `Ĝ_t^λ = (1−λ) Σ_{n=0}^∞ λ^n Ĝ_t^(n)`
- **Eq. 12:**
  `(1/T) Σ_{t=0}^T Ĝ_t^λ`
  The paper says this was "a more effective metric" than scoring the first state alone.
- **Eq. 13:**
  `MNA = ( (1/T) Σ_{t=0}^T Ĝ_t^λ ) · Ĉ`
  Here "Ĉ is 0 if the level is approximately unsolvable and 1 otherwise". A level is approximately unsolvable "if it has never been solved, e.g. has a success rate of 0%".

### Code: PLR/ACCEL path (`minigrid/maze_plr.py`)

[`compute_clipped_gae`, L85–143](https://github.com/HarryMJMead/Dynamic-Environment-Generation-for-UED/blob/12340c7fe443278f4033bab977054af1e40b8188/minigrid/maze_plr.py#L85-L143):

```python
deltas = rewards + gamma * extended_values[1:] * (1 - dones) - extended_values[:-1]
...
td_total_array = td_total_array + gamma**i * delta
clipped_td_total_array = jnp.minimum(td_total_array, 0)
if use_max_value:
    td_total_array = clipped_td_total_array
gae_array = gae_array + (lambd**(i)) * clipped_td_total_array * (1 - is_done_array) + ((lambd**(i+1))/(1 - lambd)) * clipped_td_total_array * done * (1 - is_done_array)
...
advantages = (1 - lambd)*(gae_array + ((lambd**(i))/(1 - lambd)) * clipped_td_total_array * (1 - is_done_array))
```

[`compute_score`, L491–500](https://github.com/HarryMJMead/Dynamic-Environment-Generation-for-UED/blob/12340c7fe443278f4033bab977054af1e40b8188/minigrid/maze_plr.py#L491-L500):

```python
elif config['score_function'] == "mna":
    solved = max_returns > 0
    return negative_value_loss_solved_sum(advantages, solved)
```

[`src/jaxued/utils.py` L68–70](https://github.com/HarryMJMead/Dynamic-Environment-Generation-for-UED/blob/12340c7fe443278f4033bab977054af1e40b8188/src/jaxued/utils.py#L68-L70):

```python
def negative_value_loss_solved_sum(advantages, solved):
    sum_scores = -jnp.minimum(advantages, 0).sum(axis=0) * solved
```

**Call sites:**
- New levels: L672–676, with `max_returns = compute_max_returns(dones, rewards)` from this rollout only.
- Replay: L734–739, with `max_returns = jnp.maximum(stored max_return, compute_max_returns(...))`.

The default config `minigrid/config/main_plr.yaml` sets `score_function: mna`, `gamma: 0.995`, `gae_lambda: 0.95`, `num_steps: 512`. These match our γ and λ.

### What the code actually computes

I derived this from the code; it is not stated in the paper.

- **The max term.** The recursion `c ← min(c + γ^i δ, 0)` gives c^(n) = A^(n) − max(0, A^(1), …, A^(n)), where A^(n) is the n-step advantage. That is exactly −Ĝ^(n) from Eq. 10, so the clipping implements the V^max operator.
- **λ-weighting.** Ĝ^(n) gets weight (1−λ)λ^{n−1} for n ≥ 1, the same indexing as standard GAE. The leftover weight λ^N/(1−λ) goes to the last available n-step term, either at episode end (`done`) or at rollout end (bootstrapped with `last_value`). The paper's Eq. 11 starts at n=0, which is an indexing difference.
- **Sum, not mean.** The score is Σ_t Ĝ_t^λ over all T=512 steps of the rollout, across every episode in it. The 1/T in Eq. 13 is dropped. Since T is fixed this does not change ranks.
- **Ĉ.** In code, Ĉ = 1[max completed-episode undiscounted return > 0]. On replay it is the maximum over history. The shortest-path `unsolvable` flag is only logged and never used in the score.
- **DEGen variant.** In [`minigrid/maze_obs_gen.py` L968–988](https://github.com/HarryMJMead/Dynamic-Environment-Generation-for-UED/blob/12340c7fe443278f4033bab977054af1e40b8188/minigrid/maze_obs_gen.py#L968-L988), Ĉ is computed differently: `agent_failed = pro_mean_returns == 0`, and the score is `optimisation_metric = -clipped_advantages_reward` per step.

### Inputs, sign, and adaptation

- **Rollouts needed:** only the student's ordinary PPO rollout, using values, rewards and dones. It also needs a max-return record for Ĉ.
- **Sign:** every term is ≥ 0, and larger means more regret.
- **Deviations to document for our setting:**
  - `max_returns > 0` assumes a sparse +1 success reward. Our dense reward needs an explicit success definition.
  - `done` is treated as terminal with no bootstrap. If our 1,200-step window boundary is a truncation, this changes Ĝ^(n).

---

## (b1) TRACED (Cho et al., ICLR 2026)

- **Paper:** arXiv:2506.19997 v5, https://arxiv.org/html/2506.19997v5, §3.1–3.2.
- **Code:** https://github.com/Cho-Geonwoo/TRACED at commit `3a5e8ed43b2074f0ac82c50416b6a6963ec5565a`. It is DCD-based (PyTorch) and extends ACCEL.

### Paper formulas (verbatim)

- **Eq. 3:** `ATPL(τ) = (1/T) Σ_{t=0}^T L_trans(s_t,a_t)`
  L_trans is a one-step loss between the true s_{t+1} and the LSTM model's prediction ŝ_{t+1} = f_φ(s_t,a_t).
- **Per-step loss ℓ_t (App. H.3):**
  - MiniGrid: ‖ŝ_{t+1}−s_{t+1}‖₁
  - BipedalWalker: ‖ŝ_{t+1}−s_{t+1}‖²₂
- **Eq. 4:** `Regret^(τ) = PVL(τ) + α ATPL(τ)`
- **Eq. 5:** TaskDifficulty(i,k) is the most recent value in the task difficulty buffer (TDB) for task i, or 0 if task i was never sampled.
- **Eq. 6:** `CoLearnability_i(k) = (1/|T_{k+1}|) Σ_{j∈T_{k+1}} [TaskDifficulty(j,k) − TaskDifficulty(j,k+1)]`
- **Eq. 7:** `TaskPriority(i,t) = Rank( TaskDifficulty(i,t) + β CoLearnability(i,t) )`, with sampling p ∝ 1/rank.
- **Table 10 hyperparameters:**
  - MiniGrid: α=1.0, β=1.0, γ=0.995, λ=0.95
  - BipedalWalker: α=1.5, β=0.6, γ=0.99, λ=0.9

### Code (`level_replay/level_sampler.py`)

PVL, [L585–594](https://github.com/Cho-Geonwoo/TRACED/blob/3a5e8ed43b2074f0ac82c50416b6a6963ec5565a/level_replay/level_sampler.py#L585-L594). Here `returns` is GAE + V, from [`algos/storage.py` L299–307](https://github.com/Cho-Geonwoo/TRACED/blob/3a5e8ed43b2074f0ac82c50416b6a6963ec5565a/algos/storage.py#L299-L307):

```python
clipped_advantages = (returns - value_preds).clamp(0)
mean_score = clipped_advantages.mean().item()
```

ATPL, [L225–261](https://github.com/Cho-Geonwoo/TRACED/blob/3a5e8ed43b2074f0ac82c50416b6a6963ec5565a/level_replay/level_sampler.py#L225-L261). The loss is `F.l1_loss` for image observations and `F.mse_loss` for flat ones (L140/L147). The model is trained online: one AdamW step (lr 1e-4) per scored episode, after the score is read.

```python
pred_next_obs, _ = self.transition_prob_model(observations, actions)
loss_raw = self.transition_prediction_loss(pred_next_obs, next_observations, reduction="none")
loss_per_step = loss_raw.mean(dim=-1)
mean_score = loss_per_step.mean().item()
...
score += self.transition_prob_weight * mean_prob_score
```

Co-learnability, [L334–397](https://github.com/Cho-Geonwoo/TRACED/blob/3a5e8ed43b2074f0ac82c50416b6a6963ec5565a/level_replay/level_sampler.py#L334-L397):

```python
self.regret_diffs[changed_seed_idx].append(self.previous_regret_score[changed_seed_idx] - self.current_regret_score[changed_seed_idx])
...
mean_regret_diff = np.mean(updated_regret_diffs)
for previously_updated_idx in self.previously_updated_indices:
    self.seed_scores[previously_updated_idx] += (self.colearnability_weight * mean_regret_diff)
```

### Rollouts, sign, and discrepancies

- **Rollouts needed:** no extra rollouts. Scores are computed per completed episode from the PPO training rollouts. The method also needs a learned recurrent transition model.
- **Sign:**
  - PVL ≥ 0 and ATPL ≥ 0; larger means harder.
  - A positive CoLearnability means the other tasks' difficulty went down.
  - The CL bonus is one batch-level mean, added to every level replayed in the previous cycle.
- **Discrepancy:** both shipped configs set `colearnability_weight: 1.0` and `transition_prob_weight: 1.0`. That includes [`bipedal_traced.json`](https://github.com/Cho-Geonwoo/TRACED/blob/3a5e8ed43b2074f0ac82c50416b6a6963ec5565a/train_scripts/grid_configs/bipedal/bipedal_traced.json), even though Table 10 gives α=1.5 and β=0.6 for BipedalWalker. Configs also set `handle_timelimits: true`, which bootstraps through truncations.
- **For our continuous state:** the MSE/flat-model branch applies.

---

## (b2) PACE (Yuan et al., ICML 2026)

- **Paper:** arXiv:2605.01358 v1 (2 May 2026), https://arxiv.org/html/2605.01358v1, §3.1–3.2. Poster page: https://icml.cc/virtual/2026/poster/64347.
- **Code: not found.** There is no link in the paper, on the ICML page or on GitHub. The OpenReview forum (OVlAgNFMfS) returned a challenge page, so I could not check its supplementary material.

### Formulas (verbatim)

- **Eq. 1:** `ΔJ(l) = J(θ_new,l) − J(θ_old,l)`
- **Eq. 5:** `ΔJ(l) ≈ (1/α)‖θ_new − θ_old‖²₂`
  This assumes one plain gradient-ascent step θ_new = θ_old + α∇J (Eq. 3).
- **Eq. 6:** `S(l) = (1/α)‖θ_new − θ_old‖²₂`
- **Eq. 7:** rank prioritisation, `P(l_i) ∝ (1/rank(S(l_i)))^{1/β}`, plus staleness.
- **Table 6, MiniGrid:** γ=0.995, λ_GAE=0.98, PPO epochs 5, 1 minibatch, buffer 1000, p=0.7, β=0.8, staleness 0.6. Implemented in JaxUED.

### Rollouts, sign, and what is unspecified

- **Rollouts needed:** one rollout on level l, then a provisional PPO update for scoring. That update is "without writing back to the policy".
- **Sign:** S ≥ 0, and higher means more valuable. It measures learning progress; it is not a regret estimate.
- **Unspecified in the paper:**
  - Whether θ includes the critic and shared torso or only the actor.
  - What the "provisional update" consists of: the full multi-epoch PPO update or a single step.
  - How α is interpreted under Adam. It is a constant, so it does not affect ranking.

---

## (c) Oracle-latest (Abdel Sadek et al., RLC 2025)

- **Paper:** "Mitigating Goal Misgeneralization via Minimax Regret", arXiv:2507.03068, §6.3, Eqs. 3–4 and App. F.
- **Code:** https://github.com/matomatical/jaxgmg, **`experiments` branch**. The README says this branch holds the paper code. Commit `6538664ef1c797bc74fe43b5280ddcbd09b20829`.

### Formulas (verbatim)

- **Eq. 3 (max-latest):** `Ĝ^R_max-latest(π;θ) = V̂^R_max(θ) − V̂^R_latest(π;θ)`
  V̂_max is "the highest empirical return ever achieved for this level throughout training".
- **Eq. 4 (oracle-latest):** `Ĝ^R_oracle-latest(π;θ) = max_{π'} V^R(π';θ) − V̂^R_latest(π;θ)`
  V̂_latest is "the empirical average return achieved on θ in the latest batch of rollouts with the current policy π" (wording from Eq. 16).
- **Eq. 13 (oracle, cheese env):** `max_π V^R(π;θ) = γ^{d−1}`, where d is the BFS/Floyd–Warshall shortest-path length.
- **Discounting:** returns are discounted, with γ=0.999 in the paper's hyperparameter table.

### Code

The code calls these methods `oracle-actor` and `maxmc-actor`. I matched them to the paper's names by their formulas and docstrings.

[`baselines/autocurricula/scores.py` L461–597](https://github.com/matomatical/jaxgmg/blob/6538664ef1c797bc74fe43b5280ddcbd09b20829/jaxgmg/baselines/autocurricula/scores.py#L461-L597):

```python
oracle_max_return = discount_rate ** goal_dist
...
average_return = experience.compute_average_return(rewards=rewards, dones=dones, discount_rate=discount_rate)
return oracle_max_return - average_return
```

`maxmc-actor` (L437–457) is `max_ever_return - average_return`. The max is carried across visits in [`plr.py` L234–249](https://github.com/matomatical/jaxgmg/blob/6538664ef1c797bc74fe43b5280ddcbd09b20829/jaxgmg/baselines/autocurricula/plr.py#L234-L249).

`compute_average_return`, [`experience.py` L449–498](https://github.com/matomatical/jaxgmg/blob/6538664ef1c797bc74fe43b5280ddcbd09b20829/jaxgmg/baselines/experience.py#L449-L498), computes the discounted return-to-go at each episode start and averages them. A trailing incomplete episode counts as an episode, with its partial return and no bootstrap.

### Rollouts, sign, and discrepancies

- **Rollouts needed:** the latest PPO rollout batch on the level, plus an exact per-level solver.
- **Sign:** oracle minus achieved, so ≥ 0 nominally.
- **Discrepancies:**
  - The code uses γ^{goal_dist}, but paper Eq. 13 says γ^{d−1}. The two differ by a factor of γ, so a perfect policy could score slightly negative.
  - The docstring says the oracle "assumes there is no linear time penalty and no time limit".

---

## (d) Ensemble value-disagreement scores

### 1. minimax `VALUE_DISAGREEMENT` (PLR/PAIRED score option)

[`util/rl/ued_scores.py` L212–217](https://github.com/facebookresearch/minimax/blob/2ae9e04d37f97d7c14308f5a26237dcfca63470f/src/minimax/util/rl/ued_scores.py#L212-L217):

```python
def compute_value_disagreement(batch):
    mean_scores, max_scores = compute_episodic_stats(batch.values.std(-1), batch.dones, time_average=True)
```

- **Formula:** for each episode, the mean over t of std_k V_k(s_t); then the mean over completed episodes.
- **Ensemble:** `--value_ensemble_size` ([`arguments.py` L444–448](https://github.com/facebookresearch/minimax/blob/2ae9e04d37f97d7c14308f5a26237dcfca63470f/src/minimax/arguments.py#L444-L448)) builds an `EnsembleValueHead` of vmapped heads with independent init ([`models/common.py` L225–255](https://github.com/facebookresearch/minimax/blob/2ae9e04d37f97d7c14308f5a26237dcfca63470f/src/minimax/models/common.py#L225-L255)).
- **Policy:** the policy gradient uses head 0's GAE ([`agents/ppo.py` L233–234](https://github.com/facebookresearch/minimax/blob/2ae9e04d37f97d7c14308f5a26237dcfca63470f/src/minimax/agents/ppo.py#L233-L234)).
- **Associated paper:** I found none that evaluates this option. The MNA repo copies the function (`utils.py` L76–78).

### 2. VDS (Zhang, Abbeel, Pinto, NeurIPS 2020)

- **Paper:** arXiv:2006.09641. **Code:** https://github.com/zzyunzhi/vds.
- **Formula:** δ^π(g) = std of {Q_1(g),…,Q_K(g)}.
- **Eq. 2:** `Ĉ^π(g) = f(δ^π(g)) / Σ_n f(δ^π(g^(n)))`, with f = identity and K=3.
- **In code** ([`baselines/her/goal_sampler.py` L5–47](https://github.com/zzyunzhi/vds/blob/a29fc5a6ace9cc4c625b07064b7ba96611cf7281/baselines/her/goal_sampler.py#L5-L47)), Q is evaluated at the initial observation with a = π(s_0,g):

```python
input_u = None if not value_ensemble.use_Q else policy.get_actions(o=input_o, ag=input_ag, g=candidates)
vals = value_ensemble.get_values(o=input_o, ag=input_ag, g=candidates, u=input_u)
disagreement = disagreement_fun(vals)   # 'std': np.std(vals, axis=0)
disagreement /= sum_disagreement
```

- **Rollouts needed:** none per candidate. It needs an ensemble trained off-policy with Bellman targets.

### 3. AutoDIME (Kanitscheider et al., arXiv:2203.02481)

- **Teacher reward:** `r = Σ_t r_t`, with `r_t = std_i V_i(s_t)`.
- **Ensemble:** size 2, so r_t = ½|V_1(s_t) − V_2(s_t)|. The heads are independently initialised and "trained with the same value target" (the GAE target).
- **Code:** none linked in the paper.

**Sign for all three:** ≥ 0, larger means more epistemic uncertainty. None of them is a regret estimate.

---

## ReMiDi (Beukman et al., ICML 2024; arXiv:2402.12284)

Beukman et al. show that once an agent reaches the minimax-regret bound on all levels, a regret-maximising adversary samples only levels whose regret cannot be reduced further, so learning stagnates on other levels where improvement is still possible; they propose the Bayesian level-perfect MMR refinement (BLP) and the ReMiDi algorithm, which converges to it.

---

## Not found

- **PACE:** no code anywhere I could reach. The OpenReview supplementary material was blocked by a challenge page.
- **AutoDIME:** no code link.
- **MNA:** no venue is stated on arXiv.
- **minimax `value_disagreement`:** no published paper that evaluates it.