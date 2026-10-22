---
title: 6.2 Variance Reduction and Baselines
---

# 6.2 Policy Gradient Improvements

The CartPole experiment in Section 6.4 will show the most direct symptom of high variance in REINFORCE: the learning curve shakes violently, and the policy gets dragged around by luck. This section answers a key question:

**Can we reduce the variance of $G_t$ without changing the direction of the gradient in expectation?**

Yes. The policy gradient theorem has an important property: in the gradient estimator, we are allowed to subtract a _baseline_ that does not depend on the action.

## A Baseline Does Not Change the Expectation

Recall the policy gradient theorem:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

Now replace $G_t$ with $G_t - b(s_t)$, where $b(s_t)$ is any function that depends only on the state and not on the action:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \left(G_t - b(s_t)\right) \right]$$

Why does this not change the expectation? Because the baseline term contributes zero:

$$\mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot b(s_t) \right] = \sum_t b(s_t) \cdot \underbrace{\mathbb{E}_{a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \right]}_{= \, 0} = 0$$

The last step uses a key identity: the expectation of the score function (the gradient of the log-probability) is zero. Intuitively, $\nabla_\theta \log \pi_\theta(a|s)$ measures “how should we adjust parameters to increase the probability of a particular action.” If we take a probability-weighted average over all actions, the increases and decreases cancel exactly. (Rigorously, condition on the history up to $s_t$: given $s_t$, the baseline $b(s_t)$ is a fixed number, and the inner expectation over $a_t \sim \pi_\theta(\cdot|s_t)$ vanishes.)

:::details Proof: $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)] = 0$

The normalization condition of a probability distribution is $\sum_a \pi_\theta(a|s) = 1$. Taking the gradient with respect to $\theta$ on both sides gives:

$$\sum_a \nabla_\theta \pi_\theta(a|s) = 0$$

Using $\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$, rewrite $\nabla_\theta \pi$ as $\pi \cdot \nabla_\theta \log \pi$:

$$\sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s) = 0$$

The left-hand side is exactly $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)]$.

:::

So a baseline does not change the expectation (and therefore the expected direction) of the gradient. What it changes in practice is the **variance** of the gradient estimator.

## Intuition: Why a Baseline Reduces Variance

After subtracting a baseline, the update signal changes from “how many points did this rollout get” to “how much better was this rollout than what we expected.”

Consider an example in CartPole. Suppose the current policy is already reasonably good: starting from state $s$, it lasts about 100 steps on average ($V(s) \approx 100$):

| Case                          | $G_t$ | $G_t - V(s)$ | Update Direction (No Baseline) | Update Direction (With Baseline) |
| ----------------------------- | ----- | ------------ | ------------------------------ | -------------------------------- |
| Good luck, lasted 150 steps   | 150   | +50          | Strongly reinforce             | Moderately reinforce             |
| Typical, lasted 100 steps     | 100   | 0            | Moderately reinforce           | No update                        |
| Bad luck, lasted only 50 step | 50    | -50          | Slightly reinforce             | Decrease probability             |

Without a baseline, all three cases produce a positive $G_t$, so the policy gets reinforced even when that particular outcome is _worse than average_ (the “bad luck” case). With a baseline, the typical case produces no update, and the bad-luck case is correctly penalized.

What the baseline does is build a per-state “passing line”: if the outcome is above the line, reinforce; if it is below the line, suppress. The line is not constant: different states have different $V(s)$, because what counts as “normal performance” depends on where you are in the episode.

## The Standard Baseline Is $V(s)$

The baseline can be any function that does not depend on the action. The simplest choice is a constant (for example, the average return across episodes). A constant baseline is already useful in stateless bandits, but it cannot distinguish between different states.

A better choice is a state-dependent baseline $b(s)$. A natural and effective choice is $b(s) = V^\pi(s)$: theory shows it typically removes a large share of the variance, although it is not exactly the minimum-variance baseline in general [^greensmith2004]. Look at it from another angle: $V^\pi(s)$ answers exactly the question "starting from this state, and following the current policy, how many points do we get on average." Using it as a baseline turns the update signal into "how much better was the actual outcome than the average."

We call $G_t - V(s_t)$ the **advantage**:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

In REINFORCE, $G_t$ is a Monte Carlo estimate of $Q^\pi(s_t,a_t)$, so the advantage estimate takes the form:

$$\hat{A}_t = G_t - V(s_t)$$

- $\hat{A}_t > 0$: this action is better than the average at this state; increase its probability
- $\hat{A}_t < 0$: this action is worse than the average; decrease its probability
- $\hat{A}_t \approx 0$: about as expected; no strong update

## What the Advantage Function Means

The advantage function $A^\pi(s,a)$ is one of the most important ideas in policy gradient methods. It does not ask “how good is this action,” but rather “how much better is this action than average.” This _relative_ signal is far more stable than the _absolute return_ signal $G_t$.

We will use the advantage function repeatedly in later chapters:

- **Chapter 7 Actor-Critic**: use a critic network to estimate $V(s)$ directly, enabling per-step updates (no need to wait for the episode to end)
- **Chapter 8 PPO**: use GAE (Generalized Advantage Estimation) to trade off bias and variance
- **Chapter 13 RLHF**: a learned reward model supplies the token-level reward signal; advantages are then estimated over token positions within each response

## Entropy Regularization: Pay the Policy to Stay Uncertain

Baselines reduce variance without changing the expected gradient (so the stationary point is unchanged in expectation; with a learned, moving baseline the finite-sample update path still changes). Entropy regularization changes the objective itself, on purpose: it rewards the policy for staying stochastic so that exploration does not collapse too early.

The entropy of a policy at state $s$ is

$$H(\pi_\theta(\cdot|s)) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s),$$

large when the distribution is close to uniform, near zero when one action dominates. The regularized objective adds it with a coefficient $\beta > 0$:

$$J_\beta(\theta) = J(\theta) + \beta \, \mathbb{E}_{\pi_\theta}[H(\pi_\theta(\cdot|s))].$$

Its gradient, $\beta \, \nabla_\theta H$, pushes probability mass back toward uniformity and directly counteracts premature convergence to a deterministic policy.

A concrete comparison shows both the direction and the scale. A two-action policy at $[0.9, 0.1]$ has entropy $-(0.9 \ln 0.9 + 0.1 \ln 0.1) = -(-0.0948 - 0.2303) \approx 0.325$ nats. A uniform $[0.5, 0.5]$ policy has entropy $\ln 2 \approx 0.693$ nats. With $\beta = 0.01$, the entropy bonus differs by only $0.01 \times (0.693 - 0.325) \approx 0.0037$ per step. Compare that with the return term in this chapter's own raw-return loss, `-(log_probs * returns_tensor).mean()`: a typical per-step return weight is $G_t \approx 37$ on a 100-step CartPole episode with $\gamma = 0.99$, so the return term per step is $|G_t| \cdot |\log \pi| \approx 37 \times 0.69 \approx 25$ — three orders of magnitude larger than the $0.0069$-nat entropy bonus at $\beta = 0.01$. In this unnormalized loss the entropy term only becomes comparable when $\beta = O(10)$; the often-quoted $\beta = 10^{-3}$ to $10^{-2}$ range refers to return-normalized losses such as PPO or RLHF, not to the raw-return loss of Section 6.4/6.6. So here a small $\beta$ is a gentle pull that matters most when returns are ambiguous and fades into the background once one action is clearly better. In practice keep $\beta$ small or anneal it toward zero late in training; too large a $\beta$ stops the policy from ever committing, too small lets it lock in after a few lucky episodes. Chapter 8 reuses this mechanism inside PPO, where the entropy bonus is a standard diagnostic alongside the clipped surrogate loss.

## Implementation: Adding a Value Network

In practice, we estimate $V(s)$ with an additional neural network (a value network):

```python
# The value network learns V(s)
values = value_net(states_t)
value_loss = nn.MSELoss()(values, returns_t)  # Use G_t as the training target

# Update the policy using the advantage
with torch.no_grad():
    values_pred = value_net(states_t)
advantages = returns_t - values_pred  # Â_t = G_t - V(s_t)
policy_loss = -(log_probs * advantages).mean()
```

The value network is trained so that $V(s_t)$ is as close to $G_t$ as possible. What this step means is that it is learning “starting from this state, what score do we get on average.” The policy network no longer uses $G_t$ directly, but uses the advantage $\hat{A}_t = G_t - V(s_t)$.

This is **REINFORCE with a Value Baseline**. It is still REINFORCE (you still have to wait until the episode ends and use Monte Carlo returns), but the update signal changes from $G_t$ to $\hat{A}_t$.

In Section 6.6, we will compare vanilla REINFORCE and REINFORCE + Value Baseline on CartPole: [Hands-on: CartPole Comparison Experiment](./cartpole-baseline).

---

[^greensmith2004]: Greensmith, E., Bartlett, P. L., & Baxter, J. (2004). Variance reduction techniques for gradient estimates in reinforcement learning. _Journal of Machine Learning Research_, 5, 1471-1530.
