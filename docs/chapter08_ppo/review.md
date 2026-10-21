---
title: 8.7 Summary and Exercises
---

# 8.7 Summary and Exercises

This chapter turned the Actor-Critic of Chapter 7 into a stable, reusable algorithm: reuse old data through importance sampling, cap each update with clipping, estimate credit with GAE, and — in the LLM setting — manufacture rewards with a reward model. This page collects what to retain, the formulas that do the work, the mistakes that recur in practice, and exercises with full solutions.

## What You Should Now Be Able to Do

1. **Read a PPO training run**: look at episode reward, policy entropy, clip fraction, and approximate KL together and say whether training is healthy, too aggressive, or prematurely converged ([8.1](./ppo-bipedal-walker)).
2. **Explain why the vanilla policy gradient cannot reuse data**: state the on-policy expectation assumption and show how a stale batch biases the gradient ([8.2](./trust-region-clipping)).
3. **Derive the PPO-Clip objective from scratch**: importance-sampled surrogate, KL-constrained TRPO form, then the clipped first-order approximation, including which advantage sign activates which bound ([8.2](./trust-region-clipping), [8.4](./ppo-math)).
4. **Compute GAE by hand on a short trajectory**: TD errors, the backward recursion $\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}$, and the $\lambda = 0$ / $\lambda = 1$ endpoints ([8.3](./gae-reward-model)).
5. **Implement the PPO losses**: policy ratio from stored versus recomputed log-probabilities, `min` of the unclipped and clipped terms, value loss with clipping, entropy bonus, and advantage normalization ([8.4](./ppo-math)).
6. **Choose hyperparameters for continuous control**: rollout length, minibatch size, epochs, clip range, and entropy coefficient, and what breaks when each is off ([8.1](./ppo-bipedal-walker)).
7. **Describe the LLM-alignment extension**: the Bradley-Terry reward model, why sparse terminal rewards make credit assignment hard, why GAE helps, and why four models must fit in memory ([8.3](./gae-reward-model)).

## Key Equations

- $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t}\gamma^t r_t\right]$ — the policy's score is its average discounted return.
- $\nabla_\theta J(\theta) = \mathbb{E}_t\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat{A}_t\right]$ — shift probability toward actions that beat the baseline.
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ — the one-step surprise; GAE's building block.
- $\hat{A}_t^{\mathrm{GAE}} = \sum_{k=0}^{\infty}(\gamma\lambda)^k\,\delta_{t+k}$, computed backward as $\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}$ — propagate credit back with exponential decay.
- $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\mathrm{old}}(a_t \mid s_t)$ — how much the new policy changed its mind about a sampled action.
- $L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t A_t,\ \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\,A_t\right)\right]$ — improve the surrogate, but only inside the trust interval.
- $L^{\mathrm{PPO}} = L^{\mathrm{CLIP}} + c_1 L^{\mathrm{VF}} - c_2 H[\pi_\theta]$ — actor direction, critic accuracy, and exploration in one backward pass.
- $P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$ — turn a pairwise human preference into a score gap the reward model can fit.

## Common Pitfalls

1. **Normalizing the wrong thing, at the wrong granularity.** Standardize advantages once per rollout batch, never per minibatch, and never standardize the Critic's value targets — the Critic must learn returns on their true scale.
2. **Letting `old_logprobs` go stale.** The stored log-probabilities must be frozen at sampling time. Recomputing them after a parameter update (or mixing buffers from two policy versions) makes the ratio measure the wrong change.
3. **Pushing too hard per batch.** Large $\varepsilon$ with many epochs and no KL or clip-fraction monitoring is the standard recipe for policy collapse: fast early gains, then a reward dive that never recovers.
4. **Reading raw episode rewards.** In environments like BipedalWalker, adjacent episodes can differ by 200 points on identical policies. Judge progress by the moving average, and never by a single seed.
5. **Mistaking entropy collapse for convergence.** Entropy falling fast with reward still low means the policy quit exploring (for example, standing still to avoid falling), not that it solved the task. Raise `ent_coef` before raising the learning rate.
6. **Trusting the reward model as ground truth.** A rising RM score with flat or falling human-judged quality is reward hacking, not progress. Keep the KL penalty to the reference model and re-check with held-out human evaluation.

## Exercises

### Exercise 1 — Which bound is active?

For $\varepsilon = 0.2$, state for each case (a) which clipping bound can activate, (b) what the gradient does once the ratio crosses it, and (c) what `min` picks if the ratio overshoots in the opposite direction.

1. $A_t > 0$. 2. $A_t < 0$.

::: details Solution

1. $A_t > 0$ wants the ratio up, so only the upper bound $1 + \varepsilon = 1.2$ can bind. Past it, the clipped term is the constant $1.2\,A_t < r_t A_t$, `min` picks it, and the gradient is zero: stop pushing up. If instead $r_t < 0.8$ (overshoot in the wrong direction), $r_t A_t < 0.8\,A_t$, so `min` picks the unclipped term and its nonzero gradient pulls the ratio back up.
2. $A_t < 0$ wants the ratio down, so only the lower bound $1 - \varepsilon = 0.8$ can bind. Below it, the clipped term $0.8\,A_t$ is the more negative (smaller) value, `min` picks it, gradient zero: stop pushing down. If instead $r_t > 1.2$, $r_t A_t < 1.2\,A_t$ (multiplying by a negative flips the order), `min` picks the unclipped term and pulls the ratio back down.
   :::

### Exercise 2 — Clip arithmetic, good action

Let $A_t = 2$ and $\varepsilon = 0.2$. For $r_t \in \{0.7, 1.0, 1.3\}$, compute the unclipped term, the clipped term, the value of $\min$, and which term `min` selects.

::: details Solution
The clip interval is $[0.8, 1.2]$.

- $r_t = 0.7$: unclipped $0.7 \times 2 = 1.4$; clipped $0.8 \times 2 = 1.6$; $\min = 1.4$, picks the unclipped term (wrong-direction overshoot, recover normally).
- $r_t = 1.0$: unclipped $2.0$; clipped $2.0$; $\min = 2.0$, tie (policies agree).
- $r_t = 1.3$: unclipped $1.3 \times 2 = 2.6$; clipped $1.2 \times 2 = 2.4$; $\min = 2.4$, picks the clipped term (same-direction overshoot, gradient zero).
  :::

### Exercise 3 — Clip arithmetic, bad action

Let $A_t = -2$ and $\varepsilon = 0.2$. For $r_t \in \{0.7, 2.0\}$, compute both terms and the value of $\min$. In each case say whether the gradient is zero or nonzero, and why.

::: details Solution

- $r_t = 0.7$: unclipped $0.7 \times (-2) = -1.4$; clipped $0.8 \times (-2) = -1.6$; $\min(-1.4, -1.6) = -1.6$, picks the clipped term. Gradient is zero: the bad action's probability was already cut enough (same-direction stop).
- $r_t = 2.0$: unclipped $2.0 \times (-2) = -4.0$; clipped $1.2 \times (-2) = -2.4$; $\min(-4.0, -2.4) = -4.0$, picks the unclipped term. Gradient is nonzero and points toward decreasing $r_t$: the ratio overshot in the wrong direction (a bad action became more likely), so the honest unclipped signal pulls it back.
  :::

### Exercise 4 — KL check on the running example

In Section 8.2, $\pi_{\mathrm{old}}(a_1) = 0.6$ and an update pushes $\pi_\theta(a_1) = 0.99$ (two actions). Compute $D_{\mathrm{KL}}(\pi_{\mathrm{old}} \| \pi_\theta)$ and say whether TRPO with $\delta = 0.01$ accepts the update.

::: details Solution
$$D_{\mathrm{KL}} = 0.6 \ln\frac{0.6}{0.99} + 0.4 \ln\frac{0.4}{0.01}.$$
$\ln(0.6/0.99) = \ln(0.6061) \approx -0.5008$, so the first term is $0.6 \times (-0.5008) \approx -0.3005$. $\ln(40) \approx 3.6889$, so the second term is $0.4 \times 3.6889 \approx 1.4756$. Total $\approx 1.175 \approx 1.18 \gg 0.01$: TRPO rejects the update outright.
:::

### Exercise 5 — GAE telescopes to Monte Carlo (derivation)

Show that with $\lambda = 1$, the GAE sum $\sum_{k=0}^{\infty}\gamma^k \delta_{t+k}$ equals $G_t - V(s_t)$ for a finite episode (with $V = 0$ past termination).

::: details Solution
Write the partial sum to horizon $N$ and expand $\delta_{t+k} = r_{t+k} + \gamma V_{t+k+1} - V_{t+k}$:
$$S_N = \sum_{k=0}^{N}\gamma^k r_{t+k} + \sum_{k=0}^{N}\gamma^{k+1}V_{t+k+1} - \sum_{k=0}^{N}\gamma^k V_{t+k}.$$
Shift the middle sum's index ($j = k+1$): it becomes $\sum_{j=1}^{N+1}\gamma^j V_{t+j}$, which cancels every term of the last sum except $-V_t$ and the leftover $+\gamma^{N+1}V_{t+N+1}$. Hence $S_N = \sum_{k=0}^{N}\gamma^k r_{t+k} - V_t + \gamma^{N+1}V_{t+N+1}$. For a finished episode take $N$ past the end: rewards are zero and $V = 0$ at and past termination, so $S_\infty = G_t - V(s_t)$.
:::

### Exercise 6 — Advantage from samples

Near some state, action $a_L$ was tried 3 times with returns $80, 100, 70$, and action $a_R$ 3 times with returns $120, 140, 130$. The current policy picks each with probability $0.5$. Estimate $Q(s, a_L)$, $Q(s, a_R)$, $V(s)$, and both advantages. Which action should the gradient reinforce?

::: details Solution
$Q(s, a_L) \approx (80 + 100 + 70)/3 = 250/3 \approx 83.3$. $Q(s, a_R) \approx (120 + 140 + 130)/3 = 390/3 = 130$. $V(s) \approx 0.5 \times 83.3 + 0.5 \times 130 \approx 106.7$. Advantages: $A(s, a_L) \approx 250/3 - 320/3 = -70/3 \approx -23.3$; $A(s, a_R) \approx 130 - 106.7 = 23.3$. The gradient should reinforce $a_R$ (positive advantage) and suppress $a_L$.
:::

### Exercise 7 — Coding: clipped loss on a fixed trajectory

Implement the PPO policy loss from log-probabilities and check it on two fixed two-sample batches ($\varepsilon = 0.2$). Both samples were taken with probability $0.5$ (`old_logp = log 0.5`), advantages are $[+2, -2]$.

- Batch 1: new probabilities $[0.6, 0.4]$.
- Batch 2: new probabilities $[0.9, 0.1]$.

Hand-compute the expected loss for each batch first, then verify with code.

::: details Solution
Ratios are new/old probabilities. Batch 1: $r = [1.2, 0.8]$, both on the clip boundary. Unclipped: $[1.2 \times 2,\ 0.8 \times (-2)] = [2.4, -1.6]$; clipped terms are identical. Loss $= -\mathrm{mean}([2.4, -1.6]) = -0.4$. Batch 2: $r = [1.8, 0.2]$. Unclipped: $[3.6, -0.4]$; clipped: $[1.2 \times 2,\ 0.8 \times (-2)] = [2.4, -1.6]$. $\min$ picks $[2.4, -1.6]$ (caps the good action's gain above $1+\varepsilon$; for the bad action at $r = 0.2 < 1-\varepsilon$ the clipped term wins, so that sample's gradient is zero and its probability stops falling further). Loss $= -\mathrm{mean}([2.4, -1.6]) = -0.4$: clipping visibly threw away $3.6 - 2.4 = 1.2$ of inflated objective on the first sample.

```python
import math
import torch

def ppo_policy_loss(new_logp, old_logp, adv, eps=0.2):
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    return -torch.min(surr1, surr2).mean()

old = torch.log(torch.tensor([0.5, 0.5]))
adv = torch.tensor([2.0, -2.0])
for new_p, expected in [([0.6, 0.4], -0.4), ([0.9, 0.1], -0.4)]:
    new = torch.log(torch.tensor(new_p))
    loss = ppo_policy_loss(new, old, adv)
    assert abs(loss.item() - expected) < 1e-6, (new_p, loss.item())
    print(new_p, "-> loss", round(loss.item(), 4))
```

:::

### Exercise 8 — Coding: GAE on a fixed trajectory plus multi-seed evaluation

1. Implement `compute_gae` (backward recursion) and test it on the chapter's 5-step table ($r = [0,0,0,0,1]$, $V = [0.1,0.2,0.3,0.5,0.8]$, terminal at $t = 4$, $\gamma = 1$): with $\lambda = 0$ the result must equal the TD errors $[0.1, 0.1, 0.2, 0.3, 0.2]$, and with $\lambda = 1$ it must equal the Monte Carlo advantages $[0.9, 0.8, 0.7, 0.5, 0.2]$.
2. Sketch (no need to run 2M steps) an evaluation harness that loads PPO checkpoints from several training seeds and reports mean $\pm$ std return over 20 episodes each. Why is one seed insufficient here?

::: details Solution
Part 1:

```python
def compute_gae(rewards, values, dones, gamma=1.0, lam=0.95):
    gae, out = 0.0, [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        nxt = 0.0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        out[t] = gae
    return out

rewards = [0.0, 0.0, 0.0, 0.0, 1.0]
values = [0.1, 0.2, 0.3, 0.5, 0.8]
dones = [0, 0, 0, 0, 1]
td = compute_gae(rewards, values, dones, gamma=1.0, lam=0.0)
mc = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
assert all(abs(a - b) < 1e-9 for a, b in zip(td, [0.1, 0.1, 0.2, 0.3, 0.2])), td
assert all(abs(a - b) < 1e-9 for a, b in zip(mc, [0.9, 0.8, 0.7, 0.5, 0.2])), mc
print("lambda=0:", [round(x, 2) for x in td])
print("lambda=1:", [round(x, 2) for x in mc])
```

Part 2 sketch:

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

def eval_checkpoint(path, n_episodes=20):
    model = PPO.load(path)
    env = gym.make("BipedalWalker-v3")
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))

for seed in [0, 1, 2]:
    # Prerequisite: the chapter script saves a single seed by default
    # (output/ppo_bipedal_walker.zip plus the _100k/_500k checkpoints).
    # Re-run training once per seed with
    # model.save(f"output/ppo_bipedal_walker_seed{seed}") to produce these files first.
    mean, std = eval_checkpoint(f"output/ppo_bipedal_walker_seed{seed}.zip")
    print(f"seed {seed}: {mean:.1f} +- {std:.1f}")
```

One seed is insufficient because adjacent BipedalWalker episodes on the same policy can differ by 200+ points (terrain randomization plus bimodal fall-or-walk behavior, e.g. $-65.2 \pm 73.1$ mid-training): a single seed confounds luck with policy quality, while mean $\pm$ std over episodes and seeds separates the two.
:::
