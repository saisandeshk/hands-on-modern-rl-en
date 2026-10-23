---
title: 12.4 Summary and Exercises
---

# 12.4 Summary and Exercises

## What You Should Now Be Able to Do

- Explain, with numbers, why undirected exploration fails on sparse-reward tasks, and when to reach for count-based bonuses, curiosity (ICM/RND), or episodic memory (NGU) instead.
- Write down the UCB rule, the count-based bonus, the ICM and RND intrinsic rewards, and the NGU two-timescale product, and state what makes each reward signal decay with familiarity.
- Distinguish sparse, delayed, and deceptive rewards, and match each to the right tool: novelty bonuses, multi-step return propagation or hierarchy, and stronger exploration pressure.
- Distinguish an exploration bonus from potential-based reward shaping, and explain why a non-decaying bonus can change the optimal policy.
- Explain non-stationarity in multi-agent learning, draw the CTDE information flow (who sees what, in training vs. execution), and write the MADDPG and MAPPO update rules.
- Compare MADDPG, MAPPO, and value decomposition (VDN/QMIX) on Critic inputs, on/off-policy data use, and the cooperative vs. mixed settings each suits.
- Define options, FeUdal's Manager–Worker losses, HIRO's goal-conditioned hierarchy with goal transition, and HER relabeling, and explain how each shortens the effective decision horizon.
- Spot privileged information in a training setup (for example a centralized Critic's global state) and judge whether an experimental comparison controls for it.

## Key Equations

- UCB action selection: $U_t(a) = \hat{\mu}_t(a) + c\sqrt{\ln t / N_t(a)}$ — act as if each arm were as good as its uncertainty allows.
- Total training reward: $r^{\text{total}}_t = r^{\text{ext}}_t + \beta \cdot r^{\text{int}}_t$ — extrinsic score plus weighted curiosity.
- Count-based bonus: $r^{\text{int}}_t = 1 / \sqrt{N(s_t)}$, scaled by $\beta$ in the total reward above — novelty that is used up by revisits.
- Potential-based shaping: $F(s, s') = \gamma\Phi(s') - \Phi(s)$ — the only additive shaping that provably preserves the optimal policy.
- ICM intrinsic reward: $r^{\text{int}}_t = \frac{1}{2}\|\Phi(s_{t+1}) - \hat{\Phi}(s_{t+1})\|^2$ with $\hat{\Phi}(s_{t+1}) = f_\psi(\Phi(s_t), a_t)$ — reward for what the forward model cannot yet predict, in action-relevant features.
- RND intrinsic reward: $r^{\text{int}}(s) = \|f_\psi(s) - \hat{f}(s)\|^2$ — reward for what the predictor has not yet memorized about a fixed random target.
- NGU two timescales: $r^{\text{int}}_t(s) = r^{\text{episodic}}_t(s) \cdot r^{\text{life-long}}_t(s)$ — novel only if unseen both in this episode and across training.
- MADDPG actor gradient: $\nabla_{\theta_i} J = \mathbb{E}[\nabla_{\theta_i}\mu_{\theta_i}(o_i) \cdot \nabla_{a_i} Q_i(o_1, a_1, \ldots, o_n, a_n)|_{a_i = \mu_{\theta_i}(o_i)}]$ — the centralized Critic's evaluation flows back through only agent $i$'s action.
- PPO-clip (per-agent in MAPPO): $L^{\text{CLIP}} = \mathbb{E}[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)]$ — each agent's own ratio is clipped, so no single agent destabilizes the joint policy.
- Value decomposition: $Q_{\text{tot}} = \sum_i Q_i(o_i, a_i)$ (VDN); $\partial Q_{\text{tot}} / \partial Q_i \ge 0$ (QMIX) — monotonicity lets decentralized $\arg\max$ recover the joint $\arg\max$.
- Option SMDP value: $Q^\mu(s, \omega) = \mathbb{E}[\sum_{t=0}^{T-1}\gamma^t r_t + \gamma^T \max_{\omega'} Q^\mu(s_T, \omega')]$ — one "step" lasts $T$ primitive steps.
- FeUdal manager loss: $\mathcal{L}_M = -\langle g_t, \hat{z}_{t+c} - \hat{z}_t\rangle$ — reward latent directions the worker actually follows.
- HIRO low-level reward: $r^l_t = -\|s_{t+1} - (s_t + g_t)\|$ — get closer to the commanded displacement.
- HIRO goal transition: $g' \approx \arg\max_{g \in \text{candidates}} \sum_{k=0}^{c-1} \log \pi^{\text{low}}_{\phi'}(a_{t+k} \mid s_{t+k}, g)$ — relabel old goals to ones the current low level ($\phi'$) would obey.
- Goal-conditioned binary reward: $r_g(s) = 0$ if $g$ achieved, $-1$ otherwise — honest but brutal, which is why HER relabels failures in hindsight.

## Common Pitfalls

- **Treating $\epsilon$-greedy as a solution to sparse exploration.** Undirected noise diffuses around visited states; covering a 50-step precise sequence needs directed novelty-seeking (counts, curiosity, episodic memory), not a larger $\epsilon$.
- **Keeping the intrinsic bonus at full strength forever.** A non-decaying bonus redefines "optimal": the agent farms novelty instead of solving the task. Check that bonuses are exhaustible or annealed, and prefer potential-based shaping when the true objective must be preserved exactly.
- **Diagnosing sparse tasks with training return alone.** Return stays at zero long before and after a breakthrough. Track state-action visitation and success coverage (which states/goals were ever reached) to see whether exploration is working.
- **Comparing MARL or exploration methods with unequal privileged information.** A centralized Critic with the true global state, extra resets, or oracle features can explain a "win" by itself. Hold Critic inputs and observation preprocessing fixed across baselines.
- **Letting hierarchy collapse into a single level.** If the low level ignores the goal (it can earn reward anyway) or the high level commands unreachable subgoals, the levels disconnect and all hierarchy overhead buys nothing. Verify the low level actually conditions on the goal and that commanded subgoals are achieved at a nontrivial rate.
- **Reusing off-policy hierarchical data without relabeling.** Old high-level transitions were collected under an older low level; training on raw $(s_t, g_t)$ pairs teaches the high level to command goals the current low level cannot execute. That is the bug HIRO's goal transition exists to fix.

## Exercises

### Exercise 1 — How hopeless is blind search? (numeric)

Montezuma's Revenge needs at least 50 consecutive correct actions from 6 options each. Compute (a) the success probability of uniform random search, (b) how many attempts one success takes on average, and (c) how that compares with $10^{17}$ (seconds since the Big Bang).

::: details Solution
(a) $(1/6)^{50} \approx 1.2 \times 10^{-39}$, about $10^{-39}$.

(b) The mean of a geometric distribution with success probability $p$ is $1/p$, so about $8 \times 10^{38} \approx 10^{39}$ attempts per success.

(c) $10^{39} / 10^{17} = 10^{22}$: even one attempt per second since the Big Bang falls short by a factor of ten sextillion. Random search is not a plan.
:::

### Exercise 2 — Use up a count bonus; farm a bad bonus (numeric + conceptual)

(a) With $r^{\text{int}} = 1/\sqrt{N(s)}$ and $\beta = 2$ (so the training bonus is $\beta \cdot r^{\text{int}}$), compute the bonus on the 1st, 4th, and 100th visit to the same state.
(b) An engineer adds $+1$ every time the agent _enters_ a particular "interesting" state $s^\star$, forever. With $\gamma = 0.9$, what extra return can the agent collect by cycling $s_0 \to s^\star \to s_0 \to \cdots$ forever instead of staying put? Explain why this differs from potential-based shaping.

::: details Solution
(a) $2/\sqrt{1} = 2$; $2/\sqrt{4} = 1$; $2/\sqrt{100} = 0.2$. Novelty decays exactly as required by exhaustibility.

(b) Each two-step cycle pays $+1$ on entering $s^\star$ — at $t = 1, 3, 5, \dots$, since the agent starts in $s_0$ and the first bonus is discounted once: $\gamma + \gamma^3 + \gamma^5 + \cdots = \gamma/(1-\gamma^2) = 0.9/0.19 \approx 4.74$ extra return — a perpetual-motion machine the optimal policy will exploit, possibly instead of the real task. Potential-based shaping $F = \gamma\Phi(s') - \Phi(s)$ telescopes over any trajectory (Exercise 4), so cycles sum to zero and cannot be farmed. That is the difference between a bonus (changes the objective) and shaping (preserves it).
:::

### Exercise 3 — NGU episodic novelty by hand (numeric)

The simplified episodic term is $r^{\text{episodic}} = 1/(\sqrt{k} + c\sum_{i=1}^{k} 1/\sqrt{d_i})$ with $k = 1$, $c = 1$. Compute it for nearest-neighbor distances $d = 9$ and $d = 0.04$, and explain in one sentence what the drop means.

::: details Solution
For $d = 9$: $1/\sqrt{9} = 1/3$, so $r = 1/(1 + 1/3) = 0.75$. For $d = 0.04$: $1/\sqrt{0.04} = 5$, so $r = 1/(1+5) = 1/6 \approx 0.167$. Revisiting a state seen earlier in the same episode cuts its novelty to about one-fifth — short-term novelty is used up within the episode, which is what stops the agent farming the spawn room.
:::

### Exercise 4 — Shaping telescopes (short derivation)

For $F(s_t, a_t, s_{t+1}) = \gamma\Phi(s_{t+1}) - \Phi(s_t)$ with terminal potential $\Phi(s_T) = 0$, show the shaped return $G' = \sum_{t=0}^{T-1}\gamma^t(r_t + F_t)$ equals $G - \Phi(s_0)$, and conclude the optimal policy is unchanged.

::: details Solution
$$\sum_{t=0}^{T-1}\gamma^t F_t = \sum_{t=0}^{T-1}(\gamma^{t+1}\Phi_{t+1} - \gamma^t\Phi_t) = \gamma^T\Phi_T - \Phi_0 = -\Phi(s_0),$$
a telescoping sum where every interior term cancels. Hence $G' = G - \Phi(s_0)$: shaping shifts every trajectory's return from the same start state by the same constant, so the ranking of policies — and the $\arg\max$ — is unchanged.
:::

### Exercise 5 — MAPPO clipping by hand (numeric)

With $\epsilon = 0.2$: (a) ratio $r = 1.5$, advantage $A = +2$; (b) ratio $r = 1.5$, advantage $A = -1$. Compute the per-sample clipped objective $\min(rA, \text{clip}(r)A)$ in each case and say which term the $\min$ picks.

::: details Solution
(a) Unclipped: $1.5 \times 2 = 3$. Clipped: $\text{clip}(1.5) = 1.2$, $1.2 \times 2 = 2.4$. $\min(3, 2.4) = 2.4$ — the clip binds, stopping the overshooting update. (b) Unclipped: $-1.5$. Clipped: $1.2 \times (-1) = -1.2$. $\min(-1.5, -1.2) = -1.5$ — the $\min$ picks the unclipped term, i.e. PPO does not clip this recovery move back toward the trust interval (for $A < 0$ the pessimistic bound is the more negative one; same-direction moves with $r < 1-\epsilon$ are still clipped).
:::

### Exercise 6 — Why CTDE, and why not one super-agent? (conceptual)

(a) With 6 actions per agent and 10 agents, how many joint actions does a centralized "super-agent" face?
(b) A skeptic says independent Q-learning is enough because "the environment transition given my action is fixed". Explain the flaw using $P(s' \mid s, a_i)$ vs. $P(s' \mid s, a_i, a_{-i})$, and what MADDPG's centralized Critic does (and does not) stabilize about the learning target while keeping execution decentralized.

::: details Solution
(a) $6^{10} = 60{,}466{,}176 \approx 60$ million joint actions — infeasible to enumerate, let alone learn, and it must be relearned if one agent is added.

(b) The environment's physics is fixed given the _joint_ action, $P(s' \mid s, a_i, a_{-i})$, but agent $i$'s marginal transition $P(s' \mid s, a_i) = \sum_{a_{-i}} P(s'\mid s,a_i,a_{-i})\,\pi_{-i}(a_{-i}\mid\cdot)$ shifts as teammates' policies $\pi_{-i}$ change — the same $(s, a_i)$ yields different outcomes in episode 1 vs. 100, so independent targets chase a moving goal. MADDPG's Critic $Q_i(o_1, a_1, \ldots, o_n, a_n)$ conditions on the joint action, which makes the transition model stationary given the joint action; the target value still drifts as the other agents' policies change, which is why MARL uses replay of recent joint behavior and off-policy critics to stay stable. Execution stays decentralized because each Actor $\mu_{\theta_i}(o_i)$ sees only its own observation.
:::

### Exercise 7 — Coding: add a count bonus to the chapter's MARL gridworld

`code/chapter12_exploration_marl_hierarchical/multi_agent_marl.py` compares independent Q-learning (`train_independent`) with a shared policy (`train_shared`). Extend the independent learners with the count bonus from Section 12.1: maintain per-agent visit counts $N_i(s)$ over the learner's state key and train on $r_i + \beta/\sqrt{N_i(s)}$ with $\beta = 1.0$. Run both variants for 200 episodes (`max_steps=80`, seed 1) and report each variant's mean team return over the last 50 episodes.

::: details Solution

```python
import sys
sys.path.insert(0, 'code/chapter12_exploration_marl_hierarchical')
import numpy as np
from collections import defaultdict
import multi_agent_marl as m

def train_independent_count_bonus(n_episodes=200, max_steps=80, beta=1.0, seed=1):
    np.random.seed(seed)
    env = m.MultiAgentGridWorld()
    agents = [m.QLearningAgent(agent_id=i) for i in range(env.n_agents)]
    visits = defaultdict(int)  # (agent_id, state_key) -> count
    team_returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        sk = [agents[i].get_state_key(obs[i], env) for i in range(env.n_agents)]
        total = 0.0
        for _ in range(max_steps):
            acts = [agents[i].select_action(sk[i]) for i in range(env.n_agents)]
            nxt, rews, done, _ = env.step(acts)
            nsk = [agents[i].get_state_key(nxt[i], env) for i in range(env.n_agents)]
            for i in range(env.n_agents):
                visits[(i, sk[i])] += 1
                bonus = beta / np.sqrt(visits[(i, sk[i])])
                agents[i].update(sk[i], acts[i], rews[i] + bonus, nsk[i], done)
            total += sum(rews)  # log raw team return, bonus excluded
            sk = nsk
            if done:
                break
        for a in agents:
            a.decay_epsilon()
        team_returns.append(total)
    return team_returns

bonus_returns = train_independent_count_bonus()
np.random.seed(1)
env = m.MultiAgentGridWorld()
plain = m.train_independent(env, n_episodes=200, max_steps=80, verbose=False)['episode_rewards']
print('plain IQL last-50 mean:', round(float(np.mean(plain[-50:])), 2))
print('IQL + count bonus last-50 mean:', round(float(np.mean(bonus_returns[-50:])), 2))
```

Notes: counts increment _before_ computing the bonus so the first visit pays $\beta/\sqrt{1} = 1.0$; the logged return excludes the bonus so the two variants are comparable (comparing bonus-inflated returns would be an unfair-comparison bug of exactly the kind Section 12.2 warns about). On seed 1 the two variants score comparably over the last 50 episodes (plain $\approx -39.0$, bonus $\approx -42.4$ mean team return per episode): on this small dense-reward gridworld undirected $\epsilon$-greedy already covers the space, so the bonus buys little. The exercise's point is the mechanism and the fair-comparison methodology — rerun on larger grids or sparser resources, where $\epsilon$-greedy coverage collapses, to see the bonus pull ahead.
:::

### Exercise 8 — Coding: HIRO-style goal selection in numpy

A 1-D low-level policy is Gaussian, $\pi(a \mid s, g) = \mathcal{N}(g - s,\, 1)$. From $s_0 = 0$ it executed $a_0 = 2.1 \to s_1 = 2.1$, then $a_1 = 1.9 \to s_2 = 4.0$. (a) By hand, compute $\sum_{k=0}^{1}\log\pi(a_k \mid s_k, g)$ up to the shared Gaussian constant for candidates $g \in \{8.0,\, 4.0,\, 0.0\}$. (b) Write a numpy snippet that selects the best candidate and confirm the achieved displacement $s_2 - s_0$ wins.

::: details Solution
(a) Dropping the shared $-\log\sqrt{2\pi}$ constant, $\log\pi(a\mid s,g) = -\frac{1}{2}(a - (g - s))^2$:

- $g = 8.0$: $-\frac{1}{2}[(2.1-8)^2 + (1.9-5.9)^2] = -\frac{1}{2}(34.81 + 16) = -25.405$.
- $g = 4.0$: $-\frac{1}{2}[(2.1-4)^2 + (1.9-1.9)^2] = -\frac{1}{2}(3.61 + 0) = -1.805$.
- $g = 0.0$: $-\frac{1}{2}[(2.1-0)^2 + (1.9+2.1)^2] = -\frac{1}{2}(4.41 + 16) = -10.205$.

(b)

```python
import numpy as np
states = [0.0, 2.1]
actions = [2.1, 1.9]
candidates = [8.0, 4.0, 0.0]
scores = [sum(-0.5 * (a - (g - s)) ** 2 for s, a in zip(states, actions))
          for g in candidates]
best = candidates[int(np.argmax(scores))]
print(scores)  # [-25.405, -1.805, -10.205]
print(best)    # 4.0 == s_2 - s_0
```

The achieved displacement $4.0$ wins because the observed actions look nearly optimal for it and far-fetched under the other candidates — the maximizing selection, not the raw displacement formula, is what makes relabeled data trustworthy.
:::
