---
title: 6.7 Summary and Exercises
---

# 6.7 Summary and Exercises

## What You Should Now Be Able to Do

1. Write down the policy objective $J(\theta)$ and explain in words what it averages over.
2. State the policy gradient theorem and translate it into the one-sentence reading: reinforce actions that led to good outcomes.
3. Implement REINFORCE from scratch: roll out an episode, compute discounted returns backward, and form the loss `-log_prob * G_t`.
4. Replace full-episode returns with reward-to-go and explain why past rewards only add noise.
5. Prove that subtracting a state-dependent baseline leaves the gradient unbiased, and explain why $V(s)$ is the default choice.
6. Diagnose high variance from training curves and fix the two most common causes: missing baseline and a learning rate that is far too large.
7. Add entropy regularization to a policy-gradient loss and predict how the coefficient $\beta$ changes exploration.
8. Read a CartPole training script and point to exactly which line is the policy update, which is the return computation, and which data must be freshly collected because the method is on-policy.

## Key Equations

- **Policy objective.** $J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$ — the average discounted return of the policy; the single number we want to maximize.

- **Gradient ascent.** $\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)$ — walk uphill on $J$; the plus sign marks maximization, unlike supervised-learning minimization.

- **Policy gradient theorem.** $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$ — turn the gradient of an expectation into an expectation of gradients, which samples can estimate.

- **Score-function identity.** $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)] = 0$ — probability-weighted gradient directions cancel; the fact that makes baselines legal.

- **REINFORCE loss.** $\text{loss} = -\sum_t \log \pi_\theta(a_t|s_t) \cdot G_t$ — minimize this with gradient descent and the two minus signs cancel into gradient ascent on $J$.

- **Reward-to-go.** $G_t = r_t + \gamma G_{t+1}$ — each action is weighted only by rewards that came after it, since it cannot influence the past.

- **Baseline estimator.** $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$ — same expectation as REINFORCE whenever $b$ ignores the action, but much less noisy.

- **Advantage estimate.** $\hat{A}_t = G_t - V(s_t)$ — "how much better than this state's normal outcome"; positive reinforces, negative suppresses.

- **Entropy-regularized objective.** $J_\beta(\theta) = J(\theta) + \beta \, \mathbb{E}[H(\pi_\theta(\cdot|s))]$ — pay the policy a bonus $\beta$ per nat of randomness so it keeps exploring.

## Common Pitfalls

1. **Weighting every step by the full-episode return.** Using $G_0$ for the action at step $t$ credits it for rewards earned before it acted. Always use the reward-to-go $G_t$.
2. **Subtracting something that depends on the action.** The unbiasedness proof needs $b(s_t)$ fixed given the history. A "baseline" containing $a_t$ (for example the sampled return of a competing action in the same state without proper correction) silently biases the gradient.
3. **Reusing data after a policy update.** REINFORCE's expectation is over trajectories of the _current_ $\pi_\theta$. Training twice on one episode, or mixing in a replay buffer, breaks that match; the chapter scripts collect fresh data every episode for this reason.
4. **Letting the policy loss train the value network.** Advantages must be computed under `torch.no_grad()` (as in `reinforce_with_baseline.py`). Without detaching, policy gradients flow into $V(s)$ and value gradients into $\pi$, destabilizing both.
5. **Reading single episodes instead of moving averages.** One lucky CartPole episode proves nothing under REINFORCE variance. Compare last-50 averages and, when possible, repeat with a second seed before concluding anything.
6. **Killing exploration with step size, then blaming the algorithm.** A learning rate that is far too large collapses the policy to near-deterministic after a few lucky episodes; entropy then drops to zero and learning stalls. Tune $\alpha$ first, add an entropy bonus second.

## Exercises

### Exercise 1 — Which Route, and Why?

For each task, say whether a value-based method (DQN-style) or a policy-based method is the natural fit, and give the one decisive reason: (a) CartPole with two discrete pushes; (b) a 6-joint robot arm with continuous torques; (c) LLM next-token generation where diverse outputs matter.

::: details Solution
(a) Either works; DQN is the simpler fit because the action set is tiny and $\arg\max$ over two $Q$ values is trivial. (b) Policy-based: the action space $[-10, 10]^6$ is continuous, so $\arg\max_a Q(s,a)$ is not computable, while a Gaussian policy head outputs $\mu, \sigma$ in one forward pass. (c) Policy-based: sampling from $\pi_\theta$ produces the diversity greedy $\arg\max$ decoding destroys.
:::

### Exercise 2 — Compute Returns by Hand

Rewards $[1, 0, 2]$ with $\gamma = 0.9$. Compute $G_2$, $G_1$, $G_0$.

::: details Solution
$G_2 = r_2 = 2$. $G_1 = r_1 + \gamma r_2 = 0 + 0.9 \times 2 = 1.8$. $G_0 = r_0 + \gamma r_1 + \gamma^2 r_2 = 1 + 0 + 0.81 \times 2 = 1 + 1.62 = 2.62$.
:::

### Exercise 3 — One REINFORCE Step by Hand

A bandit policy picks arm B with probability $0.25$ and wins $G_0 = 2.0$. Write the REINFORCE loss for this episode and say whether the update raises or lowers $\pi(B)$.

::: details Solution
$\log \pi(B) = \ln 0.25 \approx -1.386$. Loss $= -\log \pi \cdot G_0 \approx 1.386 \times 2.0 \approx 2.773$. Gradient descent on this loss moves $\log \pi(B)$ upward (the loss has a minus sign and $G_0 > 0$), so $\pi(B)$ rises: a rewarded action becomes more likely.
:::

### Exercise 4 — Entropy Numbers

Compute the entropy in nats of (a) a uniform two-action policy and (b) the policy $[0.8, 0.2]$. With $\beta = 0.01$, what is each policy's per-step entropy bonus?

::: details Solution
(a) $H = -2 \times 0.5 \ln 0.5 = \ln 2 \approx 0.693$ nats; bonus $0.01 \times 0.693 \approx 0.0069$. (b) $H = -(0.8 \ln 0.8 + 0.2 \ln 0.2) = -(-0.1785 - 0.3219) \approx 0.500$ nats; bonus $\approx 0.0050$. The bonus gap is only $\approx 0.0019$ per step: a gentle pull toward exploration, not a takeover.
:::

### Exercise 5 — Prove the Score-Function Identity

For a discrete action set, prove $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)] = 0$ starting from $\sum_a \pi_\theta(a|s) = 1$.

::: details Solution
Differentiate both sides of $\sum_a \pi_\theta(a|s) = 1$ with respect to $\theta$: $\sum_a \nabla_\theta \pi_\theta(a|s) = 0$. Use $\nabla_\theta \log \pi = \nabla_\theta \pi / \pi$, i.e. $\nabla_\theta \pi = \pi \cdot \nabla_\theta \log \pi$. Substituting gives $\sum_a \pi_\theta(a|s) \, \nabla_\theta \log \pi_\theta(a|s) = 0$, which is exactly the stated expectation.
:::

### Exercise 6 — Reward-to-Go Removes Past Rewards (Up to Discount)

Show that using the full-episode return $G_0$ in the step-$t$ gradient term, instead of the reward-to-go $G_t$, changes that term's expectation by a factor $\gamma^t$: the past-reward part has expectation zero, so the direction is unchanged while the past-reward noise is removed.

::: details Solution
Split $G_0 = \sum_{k<t} \gamma^k r_k + \gamma^t G_t$. The past-reward part contributes $\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \sum_{k<t} \gamma^k r_k]$. Condition on the history up to $s_t$: past rewards and $s_t$ are then fixed numbers, and the inner expectation over $a_t \sim \pi_\theta(\cdot|s_t)$ of the score function is zero by Exercise 5. So the past-reward term has expectation zero, and $\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) G_0] = \gamma^t \, \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) G_t]$: the same direction, rescaled by $\gamma^t$ (for example $0.81$ at $t = 2$, $\gamma = 0.9$). Equality of the two expectations would require absolute discounting ($\sum_{k \ge t} \gamma^k r_k$), not the chapter's $G_t$.
:::

### Exercise 7 — Coding: Constant Baseline on the Bandit

Extend the Section 6.3 bandit script (`policy_gradient_bandit.py` pattern: `loss = -log_prob * reward`) with a constant baseline $b$ equal to the running mean of past rewards, i.e. `loss = -log_prob * (reward - b)`. Run 300 episodes with and without it. Report final $P(B)$ and the standard deviation of $P(B)$ over the last 50 episodes for each version.

::: details Solution
Keep a running mean `b`: after each episode, `b += (reward - b) / count`. Replace the loss with `loss = -log_prob * (reward - b)` where `b` is a plain Python float (never a tensor requiring grad, so nothing extra flows into autograd). What to measure: run both versions and compare final $P(B)$ and the standard deviation of $P(B)$ over the last 50 episodes. The expected pattern is equal-or-better final $P(B)$ with lower late-training fluctuation, because with $b \approx 0.5$ mid-training, a lucky $1$ on arm A contributes weight $1 - 0.5 = 0.5$ instead of $1$. Exact numbers vary by seed, so report your measured numbers rather than taking any quoted band as a result.
:::

### Exercise 8 — Coding: Entropy Bonus on CartPole

Starting from `code/chapter06_policy_gradient/reinforce_cartpole.py`, add an entropy bonus to `train_one_episode`: compute the per-step entropy $H_t = -\sum_a p_a \log p_a$ from `probs`, and change the loss to `loss = -(log_probs * returns_tensor).mean() - beta * entropy.mean()` with `beta = 0.01`. Train 500 episodes and compare the last-50-episode average against the unmodified script at the same seed. Then repeat with `beta = 0.1` and explain the difference.

::: details Solution
The modified loss reads (insert after `log_probs` is computed):

```python
entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
loss = -(log_probs * returns_tensor).mean() - beta * entropy.mean()
```

`entropy` needs no `no_grad`: the bonus must differentiate through `probs` so the policy is rewarded for staying stochastic. Scale check before running: a typical mean return weight on CartPole is $G_t \approx 37$, so the per-step return term $|G_t| \cdot |\log \pi| \approx 25$ dwarfs the entropy bonus ($\le 0.693$ nats, times $\beta$). At $\beta = 0.01$ the bonus is at most $0.0069$ and at $\beta = 0.1$ at most $0.069$ — about $0.03\%$ and $0.3\%$ of the return term — so expect no measurable difference from vanilla REINFORCE at either setting in this raw-return loss. That null result is the point: the $10^{-3}$–$10^{-2}$ $\beta$ range only matters for return-normalized losses (PPO/RLHF); here $\beta$ must be $O(10)$ (try $\beta \approx 50$) before the bonus can compete with returns and keep the policy near-uniform. As with all single-seed CartPole comparisons, compare moving averages, not single episodes.
:::
