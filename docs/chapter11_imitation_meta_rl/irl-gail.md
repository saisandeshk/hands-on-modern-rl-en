# 11.2 Inverse Reinforcement Learning and GAIL

## Reading Guide

**Key Ideas**

- Understand the core idea of inverse reinforcement learning (IRL): instead of imitating actions directly, infer from expert trajectories "what is good", that is, the reward function.
- See a fundamental problem of IRL: **the reward is not unique**. There are infinitely many reward functions under which the expert is optimal, and the maximum-entropy principle helps us pick a reasonable one.
- Understand the training objective of maximum-entropy IRL, and see its fatal bottleneck: the partition function $Z$ is hard to compute and requires repeatedly running an inner RL loop.
- Master how GAIL uses the adversarial training idea of GANs to bypass the partition function, extending imitation learning to high-dimensional problems.
- Compare the strengths and weaknesses of the three routes BC, MaxEnt IRL, and GAIL, and understand why GAIL is a forerunner of LLM alignment methods such as DPO.

In the previous section we saw that behavior cloning, although simple, suffers from distribution shift, and that DAgger can solve distribution shift but needs the expert to label repeatedly online, which is simply unrealistic in many settings. You may ask: "The expert does not need to tell me what to do in every state. He has already 'told me' what is good through his own behavior. Can I infer the reward function from his trajectories myself, and then run RL on this reward by myself?"

This is the idea of **inverse reinforcement learning** (IRL). Forward RL is "given a reward, find the optimal policy". Inverse RL goes the other way: "given an expert policy (or trajectories), find the reward function that explains this behavior".

This section first explains why a reward cannot be determined uniquely from demonstrations, then uses the maximum-entropy principle to choose a learnable reward, then introduces how GAIL bypasses the partition function with a discriminator, and finally compares the costs and applicable conditions of the three imitation learning routes.

## 1. Why Infer Rewards from Demonstrations

Let's first build intuition for why "inferring the reward" may be better than "imitating actions directly".

Imagine watching an experienced driver. He drives steadily in the middle of the lane, slows down at a red light, and gives way to pedestrians. If you only do behavior cloning, what you learn is "brake when you see a red light" and "steer when you see a pedestrian", but you do not know _why_. If you meet a new scene never seen in training (such as a strange obstacle), you may not know what to do.

But if you can figure out "what reward he is maximizing", for example "keep a safe distance", "reach the destination as soon as possible", and "ride comfortably", then even in a new state you can judge for yourself what to do based on this reward. This is the effect that IRL hopes to achieve: to learn an "intention" that generalizes, rather than rote-memorizing an action mapping.

### 1.1 The Basic Setting of Inverse RL

Inverse RL assumes that expert behavior comes from some reward function that has not been observed. Training first infers this reward from trajectories and then solves for the corresponding policy with ordinary RL.

Given expert trajectories $\mathcal{D}_{\text{expert}}=\{\tau_1,\ldots,\tau_M\}$, each $\tau=(s_0,a_0,\ldots,s_T)$, we want to find a reward function $r_\psi(s,a)$ such that the expert trajectories are more likely to occur under this reward than other trajectories:

$$\text{the expert policy is optimal under } r_\psi$$

Wait, there is a big problem here. You may ask: "Does requiring only that 'the expert is optimal' determine the reward?"

The answer is: **no**. Let's think of the simplest example: what if the reward function is $r(s,a) = c$ (the same constant $c$ for every state-action pair)? The total return of every trajectory is $c \times T$, and whatever policy you adopt, the return is the same! Under this reward, any policy (including the expert policy) can be called "optimal". This is a trivial solution and is completely useless.

Consider another example, a maze task. The expert walked a path from the start to the goal. Can you be sure the reward is "+1 for reaching the goal"? Not necessarily. The reward could also be "+0.1 for every step closer to the goal and +1 at the goal", or "+10 for taking the shortest path and -1 for detours"... All of these rewards make the expert's policy optimal.

These two examples show that requiring only "the expert is optimal" cannot select the correct reward from among infinitely many candidates. We need an extra principle. Maximum-entropy IRL provides it: among all the rewards that explain the expert's behavior, choose the one with the **maximum entropy**, that is, the one that is the least peculiar and the least arbitrary.

## 2. Determining the Reward with the Maximum-Entropy Principle

Ziebart et al. 2008 proposed maximum-entropy inverse RL. It requires the trajectory distribution to match the expert's feature expectations while retaining as high an entropy as possible among the trajectories that satisfy the constraints. In the maximum-entropy model, the probability of a trajectory is proportional to the exponential of its cumulative reward:

$$p(\tau \mid r_\psi) = \frac{1}{Z(r_\psi)} \exp\left(\sum_t r_\psi(s_t, a_t)\right)$$

Let's take this formula apart and look at each part:

- $\tau = (s_0,a_0,s_1,a_1,\ldots,s_T)$: a complete trajectory
- $\sum_t r_\psi(s_t, a_t)$: the total reward of this trajectory. The higher the reward, the larger the probability that this trajectory occurs should be.
- $\exp(\cdot)$: the exponential function. The total reward increases linearly while the probability grows exponentially, which is why it is called a "softmax" distribution.
- $Z(r_\psi)$: the **partition function**. It adds up the unnormalized scores $\exp(\sum r)$ of all possible trajectories to normalize, so that the probabilities of all trajectories sum to 1. It is a constant (for a fixed $r_\psi$), but also the most troublesome part.

Let's compute $Z$ with numbers. Suppose the environment is tiny, with only three possible trajectories in total, and the current reward function gives them the following total scores:

| Trajectory                        | Total reward $\sum_t r_\psi$ | Unnormalized weight   | Normalized probability |
| --------------------------------- | ---------------------------- | --------------------- | ---------------------- |
| $\tau_1$: the expert's short path | 10                           | $e^{10}\approx 22026$ | 0.993                  |
| $\tau_2$: a somewhat longer path  | 5                            | $e^{5}\approx 148$    | 0.0067                 |
| $\tau_3$: the path into a wall    | 0                            | $e^{0}=1$             | 0.00005                |

$Z = 22026 + 148 + 1 \approx 22176$. Adding up the whole third column normalizes the probabilities. The exponential amplifies the reward gap between "10 points and 5 points" into a probability gap of about 148 times ($e^5 \approx 148.4$): trajectories with high total reward are strongly preferred, while trajectories with close scores still keep a chance.

Applying the same exponential rule to the action choice at each step gives the softmax policy:

$$\pi(a \mid s) \propto \exp\left(Q^{\text{soft}}_{r_\psi}(s, a)\right)$$

Actions with high Q-values have large probability, and actions with similar Q-values all retain some probability. This is of the same lineage as the maximum-entropy objective of SAC in Chapter 9.

Taking logarithms over the $M=|\mathcal D_{\text{expert}}|$ expert trajectories, the training objective is:

$$\max_\psi \; \mathcal{L}(\psi) = \sum_{\tau \in \mathcal{D}_{\text{expert}}} \left[\sum_t r_\psi(s_t, a_t)\right] - |\mathcal{D}_{\text{expert}}| \log Z(r_\psi)$$

This objective looks a bit complicated, but the intuition is clear:

- **First term**: raise the cumulative reward of the expert trajectories. What the expert does should score high.
- **Second term**: subtract $|\mathcal{D}| \log Z$, which prevents the reward function from giving high scores to all states without limit (then all trajectories would have equal probability and it would be useless).

Taking the gradient with respect to the parameters $\psi$ gives a more intuitive form:

$$\nabla_\psi \mathcal{L} = \mathbb{E}_{\tau \sim \text{expert}}\left[\sum_t \nabla_\psi r_\psi(s_t, a_t)\right] - \mathbb{E}_{\tau \sim p(\cdot \mid r_\psi)}\left[\sum_t \nabla_\psi r_\psi(s_t, a_t)\right]$$

This is the per-trajectory (mean-over-demonstrations) form: the full-batch gradient of the objective above carries an overall positive factor $M = \lvert\mathcal{D}_{\text{expert}}\rvert$ on both terms, which only scales the step size and is absorbed into the learning rate.

Let's "translate" this gradient into words:

- First term: **the reward gradient on expert trajectories**. For state-action pairs that the expert often visits, their rewards should be raised.
- Second term: **the reward gradient on trajectories induced by the current policy**. For places that the current policy often visits but the expert does not, their rewards should be lowered.

Put in the driving scenario: the expert almost always drives in states of "middle of the lane, steady speed", while the current policy often drifts into states of "crossing the lane line, speed fluctuating". This round of gradient raises the reward of the former kind of state-action pairs and pushes down the latter. When the policy then does RL on the new reward, it will return to the middle of the lane.

When the feature statistics on both sides are close, that is, when the state-action distribution visited by the policy agrees with the expert's, the update tends to stop. This is very similar in structure to many algorithms we know well (such as GANs, and even policy gradients).

### 2.1 Why the Partition Function Is Difficult to Compute

It looks beautiful, doesn't it? But there is a fatal problem: $\log Z(r_\psi)$ **cannot be computed analytically** in continuous state-action spaces. For the three trajectories just now, adding them directly gives $Z$. In a continuous space, $Z$ is an integral over all (infinitely many) trajectories, with no analytic form, and it cannot be enumerated either.

Three mainstream approximation schemes:

1. **Model-based**: use a learned environment model to estimate $Z$ with forward rollouts
2. **Sampling-based soft Q iteration**: approximate with soft Bellman backups (Guided Cost Learning, Finn et al. 2016)
3. **Adversarial (GAIL)**: express $r_\psi$ implicitly with a discriminator. This is the focus of the next part.

```python
def maxent_irl_step(reward_net, expert_states_actions, env_sampler, soft_q_planner):
    # 1. Run soft Q planning under the current reward to get a sampling distribution
    current_rewards = reward_net(states_actions_tensor)
    sampled_trajectories = soft_q_planner.rollout(reward_net)

    # 2. Compute the difference in feature expectations
    expert_feat = feature_expectation(expert_states_actions, reward_net)
    sampled_feat = feature_expectation(sampled_trajectories, reward_net)

    # 3. Gradient ascent to update the reward
    grad = expert_feat - sampled_feat
    reward_net.update(grad)
```

MaxEnt IRL is expensive: every outer update requires solving a complete soft Q problem in an inner loop. This makes it hard to scale to high-dimensional problems (such as visual inputs). Every time the reward network changes a little, you must solve an RL problem again, which is too slow for complex environments.

**GAIL** cleverly avoids computing $Z$ explicitly with adversarial training.

## 3. Matching Occupancy Distributions Directly with GAIL

Generative Adversarial Imitation Learning (Ho & Ermon 2016) borrows the idea of GANs and writes inverse RL as a game between a discriminator $D_\phi$ and a policy $\pi_\theta$.

You may already be familiar with GANs: a generator creates fake samples, a discriminator tells real from fake, and the two train against each other until the generator's samples are indistinguishable from real ones. GAIL applies this idea to imitation learning:

- The **generator** is our policy $\pi_\theta$, which generates "state-action pairs"
- The **discriminator** $D_\phi$ distinguishes "was this (s,a) produced by the expert or generated by the policy"
- The two are trained alternately: the discriminator tries to tell the expert from the policy, and the policy tries to "fool" the discriminator, that is, to make its own state-action distribution identical to the expert's

### 3.1 How the Discriminator and the Policy Are Trained Alternately

The discriminator is a binary classification network, and its training objective is:

$$\max_\phi \; \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{expert}}}\left[\log D_\phi(s, a)\right] + \mathbb{E}_{(s,a) \sim \pi_\theta}\left[\log (1 - D_\phi(s, a))\right]$$

Let's take it apart:

- Expert data: $D_\phi(s,a)$ should be close to 1 (judged "real"), so $\log D$ is maximized
- Policy data: $D_\phi(s,a)$ should be close to 0 (judged "fake"), so $\log(1-D)$ is maximized

This is the standard binary cross-entropy loss.

What is the policy's objective? The policy needs to bring its state-action distribution closer to the expert's, that is, make the discriminator unable to tell whether the data it generates is real or fake. With the convention that $D_\phi(s,a)$ denotes the probability that "the sample comes from the expert", a commonly used policy objective is to minimize:

$$\min_\theta \; \mathbb{E}_{(s,a) \sim \pi_\theta}\left[\log (1-D_\phi(s, a))\right] - \lambda \mathcal{H}(\pi_\theta)$$

- First term: make the (s,a) generated by the policy "look like the expert's" to the discriminator. Minimizing $\log(1-D)$ is equivalent to maximizing $D$, that is, fooling the discriminator.
- Second term: entropy regularization with coefficient $\lambda$, which prevents the policy from collapsing too early to outputting only a few actions (the same as the entropy regularization in SAC)

In implementation, an equivalent variant of $-\log(1-D_\phi(s,a))$ or $\log D_\phi(s,a)$ is often used as the **implicit reward**. The exact sign depends on whether the discriminator labels the expert as 1 or 0, and the code and the formula must use the same convention.

```python
class GAIL:
    def __init__(self, expert_data, policy, discriminator):
        self.expert_buffer = expert_data   # Expert (s, a) pairs
        self.policy = policy               # Any RL algorithm (PPO/TRPO/SAC)
        self.disc = discriminator          # Binary classification network

    def update(self, n_policy_steps=5, n_disc_steps=1):
        # === 1. Train the discriminator ===
        for _ in range(n_disc_steps):
            # Sample policy data
            policy_states, policy_actions = self.policy.sample_rollout()
            # Binary cross-entropy (logits in, targets created to match shape)
            expert_states, expert_actions = self.expert_buffer.sample()
            expert_logits = self.disc(expert_states, expert_actions)
            policy_logits = self.disc(policy_states, policy_actions)
            ones = torch.ones_like(expert_logits)
            zeros = torch.zeros_like(policy_logits)
            d_loss = (
                F.binary_cross_entropy_with_logits(expert_logits, ones) +
                F.binary_cross_entropy_with_logits(policy_logits, zeros)
            )
            self.disc_optim.zero_grad(); d_loss.backward(); self.disc_optim.step()

        # === 2. Train the policy: use -log(1-D) as the reward ===
        for _ in range(n_policy_steps):
            states, actions, next_states, _ = self.policy.rollout()
            with torch.no_grad():
                # disc outputs logits l with D = sigmoid(l) the "expert" probability,
                # so the reward -log(1-D) equals -logsigmoid(-l)
                rewards = -F.logsigmoid(-self.disc(states, actions))
            # Feed into any RL algorithm (PPO is assumed here)
            self.policy.ppo_update(states, actions, rewards, next_states)
```

The cleverest part is that we do not need to learn a reward function explicitly! The discriminator itself provides the reward signal, $-\log(1-D(s,a))$, which is high when (s,a) looks like the expert's and low when it does not. We can feed this signal directly into any RL algorithm (PPO, TRPO, and SAC all work) to update the policy.

### 3.2 The Connection Between GAIL and Maximum-Entropy IRL

You may ask: "Why does GAIL work this way? What is its relation to the maximum-entropy IRL discussed before?"

With the policy fixed, the optimal solution of the binary discriminator can be written as the ratio of two occupancy distributions:

$$D_\phi^*(s, a) = \frac{p_{\text{expert}}(s, a)}{p_{\text{expert}}(s, a) + p_{\pi_\theta}(s, a)}$$

Here each $p(s, a)$ is a discounted state-action occupancy measure, $p(s,a) \equiv \rho_\pi(s,a) = (1-\gamma)\sum_{t \geq 0}\gamma^t \Pr(s_t=s, a_t=a \mid \pi)$, so matching occupancies means matching the discounted frequency with which each state-action pair is visited rather than matching single-step actions. GAIL therefore performs occupancy matching: it minimizes a divergence between $\rho_{\pi_\theta}$ and $\rho_{\text{expert}}$ without ever writing down a reward.

Substituting this $D^*$ into the log ratio gives:

$$\log D^* - \log(1-D^*) = \log\frac{p_{\text{expert}}(s,a)}{p_{\pi_\theta}(s,a)}$$

When the policy's occupancy distribution approaches the expert's, with $p_\pi = p_{\text{expert}}$, this ratio equals 1 and its logarithm equals 0. That is, the optimal discriminator is in fact telling us: "how much worse the current policy is than the expert at this (s,a)".

Let's see with numbers how this discriminator gives a reward. Count the occurrences of two kinds of state-action pairs in the expert data and the policy data:

| $(s, a)$                                 | Occurs in expert | Occurs in policy | $D^*$ | Implicit reward $-\log(1-D^*)$ |
| ---------------------------------------- | ---------------- | ---------------- | ----- | ------------------------------ |
| (middle of the lane, steady steering)    | 800 times        | 200 times        | 0.8   | $-\log 0.2 \approx 1.61$       |
| (driving over the lane line, sharp turn) | 0 times          | 500 times        | 0     | $-\log 1 = 0$                  |

Where the expert often goes and the policy rarely goes, $D^*$ is high and so is the implicit reward. Where the policy often goes and the expert does not, the discriminator sees through it at a glance and the reward is close to 0. The `rewards = -log(1-D)` in the code of the previous part uses exactly this signal: as the policy does RL toward higher reward, it goes more and more often to "expert-like" state-action pairs.

GAIL estimates this distribution difference with the discriminator, so it does not need to enumerate all trajectories to compute $Z$. This is why it is far more efficient than MaxEnt IRL and can scale to high-dimensional tasks.

::: details Bonus: GAIL as Jensen–Shannon Divergence Minimization
From a more theoretical perspective, GAIL is in fact minimizing the Jensen-Shannon divergence between the expert distribution and the policy distribution (which is exactly the divergence that GANs minimize). Later work such as DAC (Discriminator-Actor-Critic) and WGAIL (Wasserstein GAIL) explored replacing it with other divergences or distances (such as the Wasserstein distance from optimal transport theory) to improve training stability.
:::

## 4. Comparing the Three Imitation Learning Routes

Let's compare BC, MaxEnt IRL, and GAIL systematically:

| Dimension                  | BC   | MaxEnt IRL                     | GAIL                             |
| -------------------------- | ---- | ------------------------------ | -------------------------------- |
| Solves distribution shift  | ❌   | ✅                             | ✅                               |
| Needs an environment model | ❌   | ✅ (or a soft Q approximation) | ❌                               |
| Explicit reward function   | —    | ✅ (interpretable)             | ❌ (implicit)                    |
| Computational cost         | Low  | High (inner-loop RL)           | Medium (adversarial training)    |
| Scaling to high dimensions | Easy | Hard                           | Medium                           |
| Counterpart in LLMs        | SFT  | —                              | DPO, implicitly (see Chapter 14) |

### 4.1 Training Stability of GAIL

But GAIL is not perfect either. It inherits the common ailment of GANs: unstable training. When the discriminator is too strong, the generator's gradient vanishes, and when the discriminator is too weak, the generator gets no signal. In practice a few tricks are commonly used:

- Lipschitz regularization for Wasserstein GAIL variants (original WGAN-style weight clipping, superseded by the WGAN-GP-style gradient penalty)
- Adjusting the update frequencies of the discriminator and the policy (for example, update the discriminator once for every 5 policy updates, to keep the discriminator from becoming too strong)
- Tuning the entropy regularization coefficient $\lambda$ to 0.1-1.0 to prevent the policy from collapsing to a deterministic policy

GAIL can approach expert level on MuJoCo, but it needs millions of environment interaction steps, so **sample efficiency remains a bottleneck**. This has driven research on **offline imitation learning** (such as DemoDICE and Discriminator-Weighted Behavioral Cloning (DWBC)), which combines expert data with suboptimal data and needs no online interaction.

## Section Summary

Inverse RL (IRL) changes the problem from "learn directly what action to take" to "first learn what is good": infer the reward function from expert behavior, and then train a policy with this reward. MaxEnt IRL uses the maximum-entropy principle to resolve the ill-posedness of IRL (infinitely many rewards explain the expert's behavior), but every update needs an inner RL solve, so the computational cost is too high and it is hard to scale.

GAIL borrows the idea of GANs and uses adversarial training between a discriminator and a policy to bypass explicit reward inference and the computation of the partition function, greatly improving the scalability of imitation learning. The discriminator itself provides an implicit reward, and the policy only has to try to "fool" the discriminator. This is of the same origin in spirit as preference optimization methods such as DPO in later LLMs.

But GAIL still assumes that training and deployment face the same task. The next section, [11.3 Meta-RL: MAML, RL², PEARL, and In-Context RL](./meta-rl), turns to a more ambitious question: **when the environment keeps changing, how can an agent adapt quickly to a new task from a small amount of experience?** This is meta-reinforcement learning.
