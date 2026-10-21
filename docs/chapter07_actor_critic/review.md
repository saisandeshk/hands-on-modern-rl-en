---
title: 7.R Summary and Exercises
---

# 7.R Summary and Exercises

This chapter built the Actor-Critic architecture: a Critic that learns state values and an Actor that climbs the policy gradient scaled by the advantage. The chapter then stress-tested the idea from CartPole through Pendulum and BipedalWalker up to AlphaStar-scale systems.

## What you should now be able to do

1. Define the advantage $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ and explain in plain terms why it gives a lower-variance policy-gradient signal than the raw return $G_t$.
2. Derive the one-step TD advantage $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ from the definition of $Q^\pi$, and compute it by hand on a short trajectory.
3. Train a Critic with DP, MC, and TD targets, and state for each whether it bootstraps, whether it is biased, and whether it must wait for the episode to end.
4. Write down the one-step Actor-Critic update (Actor loss $-\log \pi(a|s)\,\delta$, Critic loss $\delta^2$) and explain why $\delta$ is detached in the Actor loss but differentiated in the Critic loss.
5. Compute $n$-step advantages $A^{(n)}_t = \sum_{k=0}^{n-1} \gamma^k \delta_{t+k}$ and GAE $A^{\text{GAE}}_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$, and describe how $\gamma$ sets the horizon and $\lambda$ trades bias against variance.
6. Explain why a continuous policy outputs distribution parameters (Gaussian mean and standard deviation) rather than action probabilities, including the roles of $\tanh$ squashing and entropy regularization.
7. Diagnose the characteristic failure of unconstrained Actor-Critic on hard tasks (large policy step $\to$ stale Critic $\to$ bad advantages $\to$ worse policy) and say what constraint PPO adds in Chapter 8.

## Key equations

- **Advantage:** $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ — "how much better is this action than this state's average?"
- **Policy gradient with advantage:** $\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t)\,\hat{A}_t$ — the Actor climbs toward actions with positive estimated advantage.
- **TD advantage:** $\hat{A}_t = \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ — a one-step, low-variance, slightly biased advantage estimate.
- **$n$-step advantage:** $A^{(n)}_t = \sum_{k=0}^{n-1} \gamma^k \delta_{t+k}$ — looking $n$ real rewards ahead interpolates between TD ($n=1$) and MC ($n \to \infty$).
- **GAE:** $A^{\text{GAE}}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$ — $\lambda = 0$ is pure TD, $\lambda = 1$ is pure MC; $\lambda \approx 0.95$ is the usual compromise.
- **Actor loss:** $L_{\text{actor}} = -\log \pi(a|s)\,\delta.\text{detach}()$ — gradient descent on this loss equals policy-gradient ascent.
- **Critic loss:** $L_{\text{critic}} = \delta^2$ — squared TD error pulls $V(s)$ toward the one-step target $r + \gamma V(s')$.
- **Maximum-entropy objective (SAC):** $J(\pi) = \mathbb{E}[\sum_t \gamma^t (r_t + \alpha \mathcal{H}(\pi(\cdot|s_t)))]$ — reward plus a bonus for staying stochastic.

## Common pitfalls

1. **Letting gradients flow through $\delta$ in the Actor loss.** The advantage must be treated as a constant weight (`.detach()`); otherwise the Actor update also tries to change the Critic's prediction, corrupting both networks.
2. **Forgetting that one-step Actor-Critic is biased.** $\delta$ bootstraps from $V(s')$, so early in training the "advantage" mostly reflects the Critic's ignorance. Large policy steps on top of a bad Critic are exactly the instability BipedalWalker exhibits.
3. **Setting $\gamma$ or $\lambda$ by superstition.** $\gamma$ too small makes the agent myopic (it will not invest torque now for balance later); $\lambda = 1$ on long noisy episodes reintroduces the MC variance you adopted Actor-Critic to escape.
4. **Squashing Gaussian actions without correcting the log-probability.** $a = 2\tanh(u)$ changes the density; the $\tanh$ Jacobian term belongs in $\log \pi(a|s)$ or the Actor gradient is wrong near the torque limits.
5. **Reading Actor/Critic losses like supervised losses.** The loss can rise while the policy improves (the data distribution shifts every update). Judge training by episode return first, loss divergence second, entropy collapse third.

## Exercises

### 1. Read a TD error (easy, conceptual)

A CartPole Critic estimates $V(s) = 1.0$. The agent acts, receives $r = -1.0$, and lands in $s'$ with $V(s') = 2.0$. Take $\gamma = 0.99$. Compute $\delta$ and state whether the Actor should increase or decrease this action's probability.

::: details Solution
$$\delta = r + \gamma V(s') - V(s) = -1.0 + 0.99 \times 2.0 - 1.0 = -1.0 + 1.98 - 1.0 = -0.02.$$
$\delta < 0$: the outcome was very slightly worse than expected, so the Actor gradient step slightly decreases this action's probability. The magnitude is tiny ($0.02$), so the update is correspondingly small.
:::

### 2. One-step versus $n$-step advantages (easy, numeric)

A 3-step episode with $\gamma = 0.9$ gives rewards $(1.0, 2.0, 0.0)$ and Critic values $V = (1.0, 2.0, 1.0)$ with $V(s_3) = 0$ (terminal). Compute $\delta_0, \delta_1, \delta_2$, then $A^{(1)}_0$, $A^{(2)}_0$, $A^{(3)}_0$. Confirm that $A^{(3)}_0 = G_0 - V(s_0)$.

::: details Solution
$$\delta_0 = 1.0 + 0.9 \times 2.0 - 1.0 = 1.8,$$
$$\delta_1 = 2.0 + 0.9 \times 1.0 - 2.0 = 0.9,$$
$$\delta_2 = 0.0 + 0.9 \times 0 - 1.0 = -1.0.$$
Then:
$$A^{(1)}_0 = \delta_0 = 1.8,$$
$$A^{(2)}_0 = \delta_0 + 0.9\,\delta_1 = 1.8 + 0.81 = 2.61,$$
$$A^{(3)}_0 = \delta_0 + 0.9\,\delta_1 + 0.9^2 \delta_2 = 1.8 + 0.81 - 0.81 = 1.80.$$
Check against MC: $G_0 = 1.0 + 0.9 \times 2.0 + 0.81 \times 0.0 = 2.8$, so $G_0 - V(s_0) = 2.8 - 1.0 = 1.8 = A^{(3)}_0$. The 3-step advantage telescopes exactly to the MC advantage, as claimed in Section 7.3.
:::

### 3. GAE at three values of $\lambda$ (medium, numeric)

Suppose $\delta = (1.0, -0.5, 0.25)$ for steps $t = 0, 1, 2$ (episode ends after step 2) with $\gamma = 0.9$. Compute $A^{\text{GAE}}_0$ for $\lambda \in \{0, 0.5, 1\}$ and explain the ordering of the three answers.

::: details Solution
$$A^{\text{GAE}}_0 = \delta_0 + (\gamma\lambda)\,\delta_1 + (\gamma\lambda)^2 \delta_2.$$

- $\lambda = 0$: $A = \delta_0 = 1.0$ (pure TD; the later $\delta$ values are ignored).
- $\lambda = 1$: $A = 1.0 + 0.9(-0.5) + 0.81(0.25) = 1.0 - 0.45 + 0.2025 = 0.7525$.
- $\lambda = 0.5$ ($\gamma\lambda = 0.45$): $A = 1.0 + 0.45(-0.5) + 0.45^2(0.25) = 1.0 - 0.225 + 0.050625 = 0.825625 \approx 0.826$.
  Ordering $1.0 > 0.826 > 0.7525$: the negative $\delta_1$ drags the estimate down, but higher $\lambda$ weights it (and the partially offsetting $\delta_2$) more heavily. $\lambda = 0$ trusts only the first surprise; $\lambda = 1$ averages all of them with pure discounting.
  :::

### 4. Baselines do not bias the gradient (medium, derivation)

Show that subtracting any state-dependent baseline $b(s)$ from the return leaves the expected policy gradient unchanged:
$$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)\,b(s)] = 0.$$

::: details Solution
Fix $s$ and pull the baseline (which does not depend on $a$) out of the action expectation:
$$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)\,b(s)] = b(s)\,\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)].$$
Now use the score-function identity. Since $\nabla_\theta \log \pi = (\nabla_\theta \pi)/\pi$:
$$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)] = \sum_a \pi_\theta(a|s)\,\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} = \sum_a \nabla_\theta \pi_\theta(a|s) = \nabla_\theta \sum_a \pi_\theta(a|s) = \nabla_\theta 1 = 0.$$
(The same holds with an integral for continuous actions.) Hence the baseline term contributes zero in expectation: it recenters the update without changing its mean — the variance-reduction mechanism of Section 7.1.
:::

### 5. GAE with $\lambda = 1$ is the MC advantage (medium, derivation)

Prove that with $\lambda = 1$ and an episode ending at step $T$ (so $V(s_T) = 0$), the GAE sum telescopes:
$$\sum_{l=0}^{T-t-1} \gamma^l \delta_{t+l} = G_t - V(s_t).$$

::: details Solution
Write out the first two terms with $\delta_{t+k} = r_{t+k+1} + \gamma V(s_{t+k+1}) - V(s_{t+k})$:
$$\sum_{l=0}^{T-t-1} \gamma^l \delta_{t+l} = \big[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\big] + \gamma\big[r_{t+2} + \gamma V(s_{t+2}) - V(s_{t+1})\big] + \cdots$$
The $\gamma V(s_{t+1})$ from the first bracket cancels the $-\gamma V(s_{t+1})$ from the second; likewise every intermediate $\gamma^{k}V(s_{t+k})$ cancels between consecutive brackets. Only $-V(s_t)$ and the reward terms survive, plus the final $\gamma^{T-t}V(s_T) = 0$:
$$= -V(s_t) + \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1} = G_t - V(s_t).$$
So $\lambda = 1$ recovers exactly the unbiased MC advantage, while $\lambda = 0$ keeps only $\delta_t$.
:::

### 6. Effective horizons (medium, numeric)

The rule of thumb is that rewards $k$ steps away are weighted by $\gamma^k$, giving an effective horizon of about $1/(1-\gamma)$. Compute the horizon for $\gamma \in \{0.9, 0.99, 0.999\}$, evaluate $\gamma^k$ at $k$ equal to each horizon, and say which $\gamma$ you would start with for BipedalWalker (episodes up to 1600 steps) and why.

::: details Solution
Horizons: $1/(1-0.9) = 10$; $1/(1-0.99) = 100$; $1/(1-0.999) = 1000$.
Decay at the horizon: $0.9^{10} \approx 0.349$; $0.99^{100} \approx 0.366$; $0.999^{1000} \approx 0.368$ (all near $1/e \approx 0.368$, as expected).
For BipedalWalker, start with $\gamma = 0.99$ (the chapter scripts' choice): a horizon of $\sim 100$ steps covers several gait cycles so the Critic credits actions that set up future steps, while $\gamma = 0.9$ (horizon 10) would be too myopic to value sustained walking and $\gamma = 0.999$ (horizon 1000) would drag a full episode of noise into every update.
:::

### 7. Implement GAE with a reverse loop (harder, coding)

In `code/chapter07_actor_critic/`, both training scripts pass `gae_lambda=0.95` to SB3 but never show what it computes. Write a standalone `compute_gae(rewards, values, dones, gamma, lam)` using a single backward pass, then test it on the Exercise 2 data (`rewards = [1.0, 2.0, 0.0]`, `values = [1.0, 2.0, 1.0, 0.0]`, episode ends at the last step). It should reproduce $\delta = (1.8, 0.9, -1.0)$ at $\lambda = 0$ and $A^{\text{GAE}}_0 = 1.8$ at $\lambda = 1$.

::: details Solution

```python
import numpy as np

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    adv = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        v_next = 0.0 if dones[t] else values[t + 1]
        delta = rewards[t] + gamma * v_next - values[t]
        gae = delta + gamma * lam * (0.0 if dones[t] else gae)
        adv[t] = gae
    return adv

rewards = [1.0, 2.0, 0.0]
values = [1.0, 2.0, 1.0, 0.0]
dones = [False, False, True]
print(compute_gae(rewards, values, dones, gamma=0.9, lam=0.0))  # [1.8 0.9 -1. ]
print(compute_gae(rewards, values, dones, gamma=0.9, lam=1.0)[0])  # 1.8
print(compute_gae(rewards, values, dones, gamma=0.9, lam=0.5))  # [2.0025 0.45 -1. ]
```

Why it works: the recurrence `gae = delta + gamma*lam*gae_next` unrolls to $\delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$, exactly Equation (7.6); `dones` resets both the bootstrap (`v_next = 0`) and the accumulation at episode boundaries. At $\lambda = 0$ each entry equals its own $\delta$; at $\lambda = 1$ entry 0 equals $G_0 - V(s_0) = 1.8$, matching Exercise 2.
:::

### 8. One gradient step moves the policy the right way (harder, coding)

Build a minimal two-action Actor (logits $[0, 0]$) and scalar Critic ($V = 1.0$) in PyTorch. Take action $1$ with reward $1.0$, next value $2.0$, $\gamma = 0.99$, and perform one SGD step ($\alpha = 0.1$) on $L = -\log \pi(1)\,\delta.\text{detach}() + \delta^2$. Report the action-1 probability and $V$ before and after, and confirm both moved in the expected directions.

::: details Solution

```python
import torch

torch.manual_seed(0)
logits = torch.tensor([0.0, 0.0], requires_grad=True)
V = torch.tensor(1.0, requires_grad=True)
opt = torch.optim.SGD([logits, V], lr=0.1)

logp = torch.log_softmax(logits, dim=-1)[1]          # action 1
delta = 1.0 + 0.99 * 2.0 - V                          # = 1.98
loss = -logp * delta.detach() + delta.pow(2)
opt.zero_grad(); loss.backward(); opt.step()

print(torch.softmax(logits, dim=-1).detach())  # [0.4507, 0.5493]
print(V.item())                                # 1.396
```

Before: $P(a=1) = 0.5$, $V = 1.0$, $\delta = 1.0 + 1.98 - 1.0 = 1.98 > 0$. After: $P(a=1) \approx 0.549$ (increased, as required for positive advantage) and $V = 1.0 + 0.1 \times 2 \times 1.98 = 1.396$ (moved toward the TD target $2.98$). Removing `.detach()` would additionally route the Actor loss gradient into $V$ (via $\delta$), letting the Actor "improve" its loss by making the Critic pessimistic — the bug Pitfall 1 warns about.
:::
