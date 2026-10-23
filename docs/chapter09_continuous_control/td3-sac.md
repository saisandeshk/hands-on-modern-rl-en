# 9.2 TD3 and SAC

## Reading Guide

**Key Ideas**

- Understand how TD3 repairs DDPG's overestimation and instability with three targeted engineering techniques: twin Q-networks, delayed policy updates, and target policy smoothing.
- Understand the core idea of maximum-entropy RL: maximize not only the reward but also the policy entropy, which encourages exploration, improves robustness, and smooths the Q-function.
- Master how SAC turns the maximum-entropy objective into a practical algorithm: a stochastic policy, the reparameterization trick, and automatic temperature tuning.
- Compare DDPG, TD3, and SAC, and know which algorithm to choose first in practice.

In the previous section we saw that DDPG pioneered off-policy learning for continuous control, but it has three serious weaknesses: Q-value overestimation, sensitivity to hyperparameters, and training that collapses easily. In this section we look at two repair strategies. One is "patching": **TD3** uses three simple but effective engineering techniques, each aimed at a specific problem of DDPG. The other is "redesign": **SAC** changes the objective function at its root and uses maximum-entropy RL to make training stable by construction.

Let's start with the simpler one: how does TD3 patch DDPG?

## TD3: Three Stability Patches for DDPG

The name Twin Delayed Deep Deterministic Policy Gradient (Fujimoto et al. 2018) is a mouthful, but it is just a combination of three techniques, each solving one specific problem of DDPG. We take them one at a time.

### Patch 1: Twin Q-Networks, Against Overestimation

The first problem of DDPG is Q-value overestimation. Recall how Double DQN solves the overestimation of DQN: it uses two networks, one to select the action and one to evaluate its Q-value, which avoids the overestimation caused by "choosing and grading by the same network".

TD3 uses a more direct method: **train two independent Critic networks $Q_{\phi_1}$ and $Q_{\phi_2}$, and take the smaller of the two when computing the target**:

$$
y = r + \gamma \cdot \min(Q_{\phi_1'}, Q_{\phi_2'})(s', \mu_{\theta'}(s'))
$$

Why does this work? The intuition is simple: if one Q-network happens to overestimate the value of some action because of noise, an independently trained network is unlikely to overestimate the same action. Taking the minimum systematically pushes the overestimation down.

For example, suppose the true Q-value of some $(s', a')$ is 10. Because network estimates are noisy:

- $Q_1$ might estimate 12 (an overestimate of 2)
- $Q_2$ might estimate 9 (an underestimate of 1)

Taking the min gives 9. It is slightly below the true value but far better than using 12: **underestimation is conservative and does not let wrong, inflated values lead the policy astray**. Overestimation causes a positive-feedback collapse, while underestimation may slow learning a little but at least does not diverge.

```python
class TD3Critic:
    def __init__(self, state_dim, action_dim):
        self.Q1 = QNetwork(state_dim, action_dim)
        self.Q2 = QNetwork(state_dim, action_dim)  # Independent initialization

    def forward(self, s, a):
        return self.Q1(s, a), self.Q2(s, a)

    def target_min(self, s, a):
        return torch.min(self.Q1(s, a), self.Q2(s, a))
```

The two Critic networks have exactly the same structure, with independently initialized parameters. During training, both networks are updated toward the same target y, and each computes its own MSE loss, with the two losses added together.

### Patch 2: Delayed Policy Updates, Letting the Critic Learn First

The second problem of DDPG is training instability. When the Critic has not yet learned well, the gradient it gives the Actor is noise. After the Actor is updated with this noisy gradient, it generates even worse data for the Critic, and a vicious cycle forms.

TD3 observes that **the Critic is much harder to learn than the Actor**. The Critic must fit a function of two arguments, $Q(s,a)$, and estimate the long-term value of every state-action pair accurately. The Actor only needs to learn a function of one argument, $\mu(s)$: in each state, pick an action that maximizes Q. Since the Critic learns slowly, let the Actor wait: update the Actor only after the Critic has had several more updates and become more accurate.

Concretely: **the Critic is updated at every step, and the Actor only once every $d$ steps** (the paper uses $d=2$). The target networks are updated in sync with the Actor.

```python
for step in range(total_steps):
    # Update the critic at every step
    update_critic()

    # Update the actor + target networks only every d=2 steps
    if step_count % policy_delay == 0:
        update_actor()
        soft_update_targets()
```

This technique looks simple, but its effect is significant. It prevents the Actor from being "led astray" by the gradient of an inaccurate Critic, and gives the Actor a direction only once the Critic is more stable.

### Patch 3: Target Policy Smoothing, Against Overfitting to Q-Function Spikes

The third problem of DDPG is more subtle. The deterministic policy $\mu_{\theta'}(s')$ gives a single point when the target is computed, but the Q-function may have sharp peaks at certain points. Those peaks may be only noise from function approximation and not truly high-value regions. If the Actor keeps aiming at these spikes, the policy becomes unstable.

TD3's solution is to **add a little clipped noise to the target action, which performs local smoothing in the action space**:

$$
a' = \text{clip}\left(\mu_{\theta'}(s') + \epsilon,\ a_{\text{low}},\ a_{\text{high}}\right), \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)
$$

This expression says: add Gaussian noise to the target action $\mu_{\theta'}(s')$, clip the noise to the range $[-c, c]$, and finally clip the whole action to the legal range. A common configuration is $\sigma=0.2$, $c=0.5$.

What does this amount to? We no longer evaluate the Q-value at one exact point $a'$. We average over a small region around $a'$. If the Q-function has a very narrow spike at $a'$, this average flattens it, and the policy avoids chasing spurious peaks. **The Q-function becomes smoother, and so do the policy updates.**

### The Combined Effect of the Three Patches

All three patches are simple, with no complicated mathematics, but their combination is remarkably effective. On MuJoCo, TD3 **significantly stabilizes DDPG** and even surpasses the early version of SAC from the same period. To this day TD3 remains a strong baseline for continuous control.

Let's look at the complete TD3 update code:

```python
class TD3:
    def update(self, batch_size=256):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # === Critic update (twin Q + target smoothing) ===
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # Target policy smoothing: add clipped noise to the target action
            noise = (torch.randn_like(next_actions) * 0.2).clamp(-0.5, 0.5)
            next_actions = (next_actions + noise).clamp(-self.action_max, self.action_max)
            # Twin Q, take the min
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optim.zero_grad(); critic_loss.backward()
        self.critic_optim.step()

        # === Actor update (delayed) ===
        if self.step_count % self.policy_delay == 0:
            # Note: using Q1 alone is enough to update the Actor
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optim.zero_grad(); actor_loss.backward()
            self.actor_optim.step()

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
```

Note one detail: when updating the Actor, only $Q_1$ is used and $Q_2$ is not. Why? The Actor's goal is to maximize the Q-value, and any Q will do as long as it is already a conservative estimate. One is enough, and using both may make them interfere with each other.

TD3 works well, but in essence it is still "patching the DDPG framework": the policy is still deterministic, exploration still relies on added noise, and although there are fewer hyperparameters than in DDPG, the noise scale still has to be tuned. Is there a way to change the objective function at its root, so that exploration and stability are built into the algorithm rather than added as patches?

That is what SAC does.

## SAC: Maximum-Entropy Reinforcement Learning

Soft Actor-Critic (Haarnoja et al. 2018) approaches the problem from a completely different angle. It says: so far we have defined the goal as "maximize the expected cumulative reward", and this goal is too "greedy". It wants only the highest score and does not care whether the policy has put all its eggs in one basket. What if we change the goal so that we want not only high reward but also **high policy entropy**, that is, the policy keeps some randomness in every state and does not converge too early to a single deterministic action?

### The Maximum-Entropy Objective

The SAC objective function is:

$$
J(\pi) = \mathbb{E}_{(s_t, a_t) \sim \pi}\left[\sum_t \gamma^t \left(r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t))\right)\right]
$$

Let's take it apart. The original objective was $\sum \gamma^t r_t$, which adds up rewards only. Now there is one more term: $\alpha \mathcal{H}(\pi(\cdot \mid s))$.

- $\mathcal{H}(\pi(\cdot \mid s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$ is the **entropy** of the policy in state $s$. What is entropy? Put simply, it is "randomness". If the policy picks one action with certainty (probability 1) in a state, the entropy is 0. If the policy spreads probability over many actions, the entropy is large.
- $\alpha$ is the **temperature coefficient**, which controls the weight of the entropy bonus. The larger $\alpha$ is, the more exploration is encouraged. With $\alpha = 0$ the objective reduces to ordinary RL.

You may ask: "Why add an entropy bonus? What good does encouraging randomness do?"

**First benefit: automatic exploration.** DDPG and TD3 both require us to add Gaussian noise by hand to explore, and the noise scale must be tuned. SAC's policy is itself stochastic: it outputs a distribution and samples actions from it. The entropy bonus prevents the policy from collapsing too early into a delta function, so exploration comes for free and no noise needs to be added manually.

**Second benefit: robustness.** If several actions in a state are about equally good, the maximum-entropy policy assigns some probability to each of them rather than picking only one. When the environment is noisy or the model has errors, the policy then does not collapse because of a small perturbation.

**Third benefit: a smoother Q-function.** The entropy bonus encourages the policy to "spread out" among actions with similar Q-values. This makes the learning signal of the Q-function smoother and reduces the problem of overfitting to spikes. It is akin to TD3's target smoothing, but here it arises naturally from the objective rather than being an added patch.

Let's get a feel for it with a simple example. Suppose a state has three actions:

- Action A: Q = 10 (the best)
- Action B: Q = 9.9 (almost as good)
- Action C: Q = 1 (very poor)

Ordinary RL (DDPG/TD3) converges to choosing A with probability 100%, which has entropy 0. What does maximum-entropy RL do? It trades off between "choosing high-Q actions" and "keeping entropy". With a suitable $\alpha$, it might assign 60% probability to A, 39% to B, and 1% to C. The entropy is fairly large and the expected Q is still high. If it later turns out that A has a problem (for instance, the environment changed), there is a good chance the policy is already trying B and can switch quickly.

### The Soft Bellman Equation

When the objective changes, the Bellman equation must change with it. The original Bellman equation is:

$$
Q(s,a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q(s', a')\right]
$$

Under maximum entropy we no longer take $\max_{a'}$. We take a "soft maximum" that considers the Q-value of every action and also accounts for entropy:

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi}\left[Q^\pi(s, a)\right] + \alpha \mathcal{H}(\pi(\cdot \mid s))
$$

$$
Q^\pi(s, a) = \mathbb{E}_{s'}\left[r + \gamma V^\pi(s')\right]
$$

Here $V(s)$ is no longer $\max_a Q(s,a)$ but the expectation of Q plus the entropy bonus. For reference, the optimal soft value has a closed form in continuous action spaces:

$$
V^*(s) = \alpha \log \int \exp\left(\frac{Q^*(s, a)}{\alpha}\right) da
$$

This integral looks a bit frightening, but you do not need to memorize it. SAC uses a clever technique (the reparameterization trick) in its implementation to avoid computing this integral directly.

### The Stochastic Policy and the Reparameterization Trick

The DDPG/TD3 policy is deterministic, $a = \mu_\theta(s)$. The SAC policy is stochastic: it outputs the mean and standard deviation of a Gaussian distribution:

$$
a \sim \mathcal{N}\left(\mu_\theta(s),\ \sigma_\theta^2(s)\right)
$$

This is the simplified unsquashed form used in the snippets below so the reparameterization idea stays visible. Real bounded actuators need the squashed policy of the Bounded Actions subsection ($a = a_{\max}\tanh(u)$ with the corrected $\log \pi$); every $\log \pi$ there replaces the plain Gaussian log-density used here.

But there is a problem. If we sample $a$ directly from the Gaussian and then compute the Q-value, how does the gradient flow back to $\mu$ and $\sigma$? The sampling operation is not differentiable: you cannot tell whether a small change in the sampled $a$ is due to a change in $\mu$ or a change in the sampled noise.

The solution is the **reparameterization trick**. We separate the randomness from the deterministic part: first sample noise $\epsilon$ from a fixed standard normal distribution, then use $\mu$ and $\sigma$ to "transform" the noise into an action:

$$
a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Here $\odot$ is element-wise multiplication. Now $a$ is still a sample from a Gaussian distribution, but the randomness comes only from $\epsilon$. Since $\epsilon$ is fixed (it has no gradient), all gradients can flow back to the policy network through $\mu_\theta(s)$ and $\sigma_\theta(s)$.

With reparameterization, the Actor loss can be written as:

$$
\mathcal{L}_{\text{actor}} = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}}\left[\alpha \log \pi_\theta(a \mid s) - Q_\phi(s, a)\right]
$$

Here $a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon$ is the reparameterized action. Minimizing this loss is equivalent to maximizing $Q(s,a)$ while subtracting $\alpha \log \pi(a|s)$, and the latter is the contribution of the entropy term.

```python
def actor_loss(self, states):
    # Simplified: no tanh squashing, so this matches the Gaussian above.
    # For bounded actions use the squashed policy of the Bounded Actions
    # subsection and its corrected log_prob instead.
    mu, sigma = self.actor(states)
    dist = Normal(mu, sigma)
    # Use rsample() rather than sample(): rsample supports reparameterized gradients
    actions = dist.rsample()
    log_probs = dist.log_prob(actions).sum(-1, keepdim=True)

    # Minimize: alpha * log_prob - Q(s,a)
    q1, q2 = self.critic(states, actions)
    min_q = torch.min(q1, q2)
    return (self.alpha * log_probs - min_q).mean()
```

Note two details:

1. The code uses `rsample()` rather than `sample()`. The former keeps the gradient path (reparameterization) and the latter does not.
2. As in TD3, SAC also uses twin Q-networks and takes the min to suppress overestimation.

### Automatic Temperature Tuning

SAC has one key hyperparameter, $\alpha$, the temperature coefficient, which controls the balance between exploration and exploitation. If $\alpha$ is too large, the policy is too random and only explores without converging. If $\alpha$ is too small, the policy becomes deterministic too fast and we are back to DDPG's problem. Tuning $\alpha$ by hand is tedious.

A very practical engineering innovation of SAC is **automatic temperature tuning**: treat $\alpha$ as a learnable parameter and adjust it automatically during training.

What is the objective? We want the policy entropy to stay near a target value $\mathcal{H}_0$. If the actual entropy is larger than the target (too random), reduce $\alpha$ so the reward term matters more. If the actual entropy is smaller than the target (too deterministic), increase $\alpha$ to encourage more exploration.

Formally, the optimization objective for $\alpha$ is:

$$
\alpha^* = \arg\min_\alpha \mathbb{E}\left[-\alpha \log \pi(a \mid s) - \alpha \mathcal{H}_0\right]
$$

In practice the target entropy $\mathcal{H}_0$ is usually set to $-|\mathcal{A}|$, the negative of the action dimension. For example, if the action is a 6-dimensional continuous vector, the target entropy is set to -6. This is a heuristic, but it works well on most tasks and needs no manual tuning.

```python
def update_alpha(self, states):
    # Simplified: same unsquashed log_prob as the actor snippet above.
    # With bounded actions, substitute the squashed corrected log_prob here too.
    mu, sigma = self.actor(states)
    dist = Normal(mu, sigma)
    actions = dist.rsample()
    log_pi = dist.log_prob(actions).sum(-1)

    # alpha loss: push the entropy toward target_entropy
    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
    self.alpha_optim.zero_grad()
    alpha_loss.backward()
    self.alpha_optim.step()
    self.alpha = self.log_alpha.exp().item()
```

Note that the code optimizes `log_alpha` rather than $\alpha$ directly. This guarantees that $\alpha$ is always positive (since $\exp(\log \alpha) > 0$).

### Bounded Actions: Tanh Squashing

There is one practical problem we have glossed over: a Gaussian has unbounded support, but real actuators do not. A HalfCheetah joint accepts torques only in a bounded range such as $[-1, 1]$, and a steering command saturates at its mechanical limit. Clipping the sampled action afterwards is tempting but wrong for SAC, because clipping piles probability mass onto the boundary points while the stored $\log \pi(a \mid s)$ still describes the unclipped Gaussian, so the entropy term and the temperature update are computed for a distribution the policy does not actually execute.

The standard fix is **squashing**: sample an unbounded pre-action $u$ with the reparameterization trick, then pass it through $\tanh$ and rescale:

$$
u \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s)), \qquad a = a_{\max} \cdot \tanh(u).
$$

Because $\tanh$ maps $\mathbb{R}$ onto $(-1, 1)$, the executed action always respects the control constraint, and the policy stays differentiable. The price is a change-of-variables correction to the log-probability: probability mass gets compressed where $\tanh$ is flat, so

$$
\log \pi(a \mid s) = \log \mu(u \mid s) - \sum_i \log a_{\max} - \sum_i \log\left(1 - \tanh^2(u_i)\right),
$$

where $\mu(u \mid s)$ is the Gaussian density of the pre-action. (For the usual MuJoCo bounds $[-1, 1]$, $a_{\max} = 1$ and the middle term vanishes.) Every $\log \pi$ in the actor loss and the temperature loss must use this corrected value, which corresponds to the entropy of the squashed distribution the environment actually sees.

```python
mu, sigma = self.actor(states)
dist = Normal(mu, sigma)
u = dist.rsample()                          # unbounded pre-action
actions = self.action_max * torch.tanh(u)   # squash into [-action_max, action_max]
log_probs = dist.log_prob(u).sum(-1, keepdim=True) \
    - torch.log(torch.as_tensor(self.action_max)) * u.shape[-1] \
    - torch.log(1 - torch.tanh(u).pow(2) + 1e-6).sum(-1, keepdim=True)
# Note: with action bounds [-1, 1] the log action_max term is zero.
```

A quick check in one dimension with $a_{\max} = 1$, $\mu = 0$, $\sigma = 0.5$, and a sampled $u = 0.5$: the executed action is $\tanh(0.5) \approx 0.462$, safely inside the bound. The Gaussian log-density at $u = 0.5$ is about $-0.726$, and the squashing correction is $-\log(1 - \tanh^2(0.5)) \approx +0.240$, giving $\log \pi(a \mid s) \approx -0.486$. Near saturation ($|u|$ large) the correction grows, correctly reporting that the squashed policy has lower entropy there than the raw Gaussian suggests.

### The Advantages of SAC

SAC puts these pieces together:

- **Stochastic policy + entropy bonus**: automatic exploration, with no manual noise
- **Twin Q-networks**: suppress overestimation (as in TD3)
- **Reparameterization trick**: makes the gradient of a stochastic policy computable
- **Automatic temperature tuning**: no manual tuning of exploration strength

How well does it work? On the MuJoCo continuous-control benchmarks, SAC has **long dominated the leaderboards**, for these reasons:

1. **High sample efficiency**: it is off-policy and inherits the strengths of DDPG
2. **Stable training**: maximum entropy plus twin Q-networks make it insensitive to hyperparameters
3. **Out of the box**: with automatic temperature tuning, one set of hyperparameters works in most environments
4. **Strong final performance**: in published 1M-step HalfCheetah results it reaches scores around 12000, matching or surpassing TD3 and DDPG, with higher scores possible given longer training

### Comparing DDPG, TD3, and SAC

Let's compare the three algorithms side by side:

| Dimension                  | DDPG          | TD3           | SAC                         |
| -------------------------- | ------------- | ------------- | --------------------------- |
| Policy type                | Deterministic | Deterministic | Stochastic (Gaussian)       |
| Q-networks                 | 1             | 2 (twin)      | 2 (twin)                    |
| Exploration                | Added noise   | Added noise   | Entropy bonus (built in)    |
| Stability                  | Poor          | Moderate      | Strong                      |
| Hyperparameter sensitivity | High          | Moderate      | Low (automatic temperature) |
| Recommended first choice   | ❌            | ⚠️            | ✅                          |

**Practical recommendation**: **choose SAC first** for continuous-control tasks. Choose TD3 only when you explicitly need a deterministic policy (for example, when you want no randomness at all at deployment time). DDPG now has mainly teaching value and is rarely used in practice.

## Comparing Training Curves

Training for 1 million steps on the MuJoCo HalfCheetah-v4 environment, the typical behavior of the three algorithms looks like this:

```
Return
12000 │                    ╭─────── SAC (stable convergence)
10000 │                  ╭─╯
 8000 │                ╭─╯  ╭─────── TD3 (stable but slightly slower)
 6000 │              ╭─╯   ╱
 4000 │            ╭─╯    ╱
 2000 │          ╭─╯     ╱  ╭───── DDPG (diverges, occasionally recovers)
     0 │─────────╯──────╱──╯
       └───────────────────────────────
        0    200K  400K  600K  800K  1M steps
```

Three features stand out:

- **SAC** converges fastest and most stably. The exploration that comes with maximum entropy speeds up early learning, and automatic temperature tuning keeps convergence stable later.
- **TD3** is slightly slower than SAC but ends with similar performance. The three stability patches finally make DDPG usable.
- **DDPG** diverges most of the time. It trains successfully only under some random seeds and is very unreliable.

::: details Bonus: Details of SAC's Soft Q Update
For readers who want to go deeper into the mathematics of SAC, here is the full form of the Critic update. Because the policy is stochastic, the target y is not $r + \gamma Q(s', a')$. It must take the expectation over actions $a' \sim \pi(\cdot | s')$ and include the entropy term:

$$
y = r + \gamma \left( \min(Q_{\phi_1'}, Q_{\phi_2'})(s', \tilde{a}') - \alpha \log \pi_\theta(\tilde{a}' \mid s') \right)
$$

Here $\tilde{a}'$ is an action sampled with reparameterization from the target policy. Note that subtracting $\alpha \log \pi$ amounts to adding the entropy term (since $\mathcal{H} = -\mathbb{E}[\log \pi]$). With bounded actions, $\log \pi_\theta(\tilde{a}' \mid s')$ must be the squashed corrected value from the Bounded Actions subsection, not the plain Gaussian density.

Taking the min over twin Q-networks serves, as in TD3, to suppress overestimation. In the actual code, the computation of this target is structured like TD3's, except for the extra log_prob term and the use of the sampled action in place of the deterministic action.
:::

## Section Summary

DDPG → TD3 → SAC is a three-step evolution of off-policy algorithms for continuous control:

1. **DDPG** extends the DQN idea to continuous actions with a deterministic policy, Actor-Critic, experience replay, and target networks, but it is unstable and collapses easily.
2. **TD3** stabilizes DDPG with three engineering techniques: twin Q-networks with a min to suppress overestimation, delayed policy updates to let the Critic learn first, and target policy smoothing to avoid chasing Q-function spikes.
3. **SAC** reformulates the objective at its root. Maximum-entropy RL adds an entropy bonus on top of the reward, which builds exploration into the algorithm and stabilizes training. Together with twin Q-networks, the reparameterization trick, and automatic temperature tuning, it has become the first choice for continuous control.

In practice, SAC is the first choice, TD3 is the alternative when a deterministic policy is needed, and DDPG is no longer recommended.

The next section, [9.3 Model-Based RL](./model-based), turns to another direction. When sampling the real environment is very expensive (a physical robot, for instance), reusing historical data is not enough. We learn an environment model that generates "imagined" data, which improves sample efficiency by another factor of 10-100.
