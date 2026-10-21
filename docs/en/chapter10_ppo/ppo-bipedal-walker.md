---
title: '8.1 Hands-On: BipedalWalker Continuous Control'
---

# 8.1 Hands-On: PPO on BipedalWalker

## Why We Need PPO

In the previous chapter we built the [Actor-Critic architecture](../chapter09_actor_critic/actor-critic). The Actor chooses actions, the Critic judges how good they are, and the two cooperate through the [advantage function](../chapter09_actor_critic/advantage-function) $A(s,a)$. In simple environments Actor-Critic performs quite well. But move the same architecture to a more complex environment (such as the bipedal walker BipedalWalker) or to a larger model (such as a language model with billions of parameters), and a serious problem surfaces: **training is unstable**.

[Policy gradient methods](../chapter08_policy_gradient/reinforce) have a notorious weakness: a single update that is too large causes "policy collapse". Imagine learning to ride a bicycle. If you shift your center of gravity too far at once, you do not ride better, you simply fall. The [TD error](../chapter09_actor_critic/critic-training) signal of Actor-Critic lowers the variance, but it does not solve this problem at the root. We need a mechanism that **constrains the size of every update**, so that the policy takes small, quick steps instead of one giant leap.

**PPO (Proximal Policy Optimization)** was created to solve exactly this problem. It is a stabilized improvement of Actor-Critic that keeps the division of labor between Actor and Critic. It only adds a more stable set of rules to the step "how to update the Actor": by clipping the objective function, it makes sure the new policy does not stray too far from the old one. Because it is simple to implement and trains stably, PPO has become one of the most widely used algorithms in deep reinforcement learning (and later in LLM alignment).

::: tip Prerequisites Review
This chapter uses the following concepts frequently:

- [Policy gradient $\nabla_\theta J$](../chapter08_policy_gradient/reinforce): PPO adds constraints on top of the policy gradient
- [The high-variance problem of REINFORCE](../chapter08_policy_gradient/cartpole): why all the improvements are needed
- [Advantage function $A(s,a)$](../chapter09_actor_critic/advantage-function): the PPO policy update depends on the advantage signal
- [TD error and Critic training](../chapter09_actor_critic/critic-training): how the Critic is trained in PPO
- [Actor-Critic architecture](../chapter09_actor_critic/actor-critic): PPO is a practical variant of Actor-Critic
:::

This chapter follows the path "**practice → constraint → estimation → extension**":

| Section                                                          | Questions you will answer                                                                                                            |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **8.1 Hands-On: PPO on BipedalWalker**                           | What does PPO training look like? How do you read Reward, Entropy, Clip Fraction, and KL? How does it handle a continuous action space? |
| [8.2 Trust Regions and PPO-Clip](./trust-region-clipping)        | Why does an update that is too large collapse? How do the KL constraint of TRPO and the clipping of PPO each work?                   |
| [8.3 GAE Advantage Estimation](./gae-reward-model)               | How does GAE interpolate between bias and variance? How many models must run at once for PPO in LLM alignment?                       |

**Bonus material** (optional): [PPO Mathematical Derivation](./ppo-math) · [PPO Game Projects](./ppo-game-benchmark) · [RL Exploration in Long-Horizon Tasks](./rl-long-horizon-planning)

---

> **Section goal**: Train PPO to control a bipedal robot walking on randomized terrain, and understand how policy learning in a continuous action space differs in essence from the discrete case.

> **Learning path**: **8.1 PPO on BipedalWalker** → [8.2 Trust Regions and PPO-Clip](./trust-region-clipping) → [8.3 GAE Advantage Estimation](./gae-reward-model)

> **Code and resources for this section**: [ppo_bipedal_walker.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter07_ppo/ppo_bipedal_walker.py) · [render_bipedal_walker.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter07_ppo/render_bipedal_walker.py) · [requirements.txt](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter07_ppo/requirements.txt)

In the previous chapters we looked at discrete-action tasks with CartPole and LunarLander, where the policy only has to pick one action out of a few. Real control problems such as robot joint torques, a car's throttle and brake, or drone rotor speeds usually have a **continuous action space**. One of the core strengths of PPO is that it handles continuous actions natively. The policy network directly outputs the mean and standard deviation of a Gaussian distribution, and a continuous action is sampled from it, so the action space does not have to be discretized. BipedalWalker-v3 is exactly this kind of continuous control task.

## 8.1.1 Running BipedalWalker Training

The task in BipedalWalker is to control a bipedal robot walking on randomly generated terrain. The state has 24 dimensions (including lidar range readings, joint angles, and velocities), and the action is a 4-dimensional continuous vector (the hip and knee torques of the two legs). It suits the main experiment of this chapter better than LunarLander, because the agent no longer chooses among a few discrete actions and must learn a continuous control signal directly.

![The bipedal robot in BipedalWalker must walk stably on randomized terrain](../../chapter10_ppo/images/bipedalwalker_demo.gif)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 8.1-1: The goal of BipedalWalker is to teach a bipedal robot to walk over the terrain instead of falling.</em>
</div>

Install the dependencies:

```bash
pip install -r code/chapter07_ppo/requirements.txt
```

Run training:

```bash
python code/chapter07_ppo/ppo_bipedal_walker.py \
  --total-timesteps 2000000
```

BipedalWalker trains much more slowly than the discrete-action environments we have seen. A task like LunarLander shows a clear learning trend within 200k steps, while BipedalWalker usually needs more than 2 million steps before it walks stably. On an ordinary CPU, 2 million steps take about 60-90 minutes. If you only want to check that the pipeline runs, first test quickly with `--total-timesteps 100000`.

The PPO hyperparameter configuration for BipedalWalker:

```python
model = PPO(
    policy="MlpPolicy",       # multilayer perceptron policy
    env=vec_env,              # 8 parallel environments
    learning_rate=3e-4,       # learning rate
    n_steps=2048,             # rollout steps per update
    batch_size=256,           # a larger minibatch stabilizes gradient estimates in continuous control
    n_epochs=10,              # update epochs per batch
    clip_range=0.2,           # PPO clipping range
    ent_coef=0.005,           # entropy coefficient (continuous spaces explore more, so a slightly lower value is enough)
    gamma=0.99,               # discount factor
    gae_lambda=0.95,          # GAE lambda
)
```

`batch_size` is set to 256 because policy updates have higher variance in a continuous action space, and a larger batch helps stabilize the gradient estimate. `ent_coef` is set to 0.005 because a Gaussian policy already explores continuously (every sample carries randomness), so not much extra entropy regularization is needed. We use 8 parallel environments because BipedalWalker episodes are longer (up to 1600 steps), and more parallel environments keep the sampling throughput up.

## 8.1.2 Reading the Training Curves

This experiment uses the PPO implementation of **Stable-Baselines3 (SB3)**. SB3 is one of the most mainstream RL libraries today, and PPO is its most widely used algorithm. Our training configuration is 8 parallel `DummyVecEnv` environments, `MlpPolicy` (a multilayer perceptron), learning rate `3e-4`, `batch_size=256`, `clip_range=0.2`, and 2M total training steps. The training script generates 4 separate training-metric plots under the `output/` directory.

PPO training has 4 key metrics, and each reflects the learning state of the policy from a different angle. We analyze them one by one.

### Episode Reward

Episode reward is the most intuitive metric: the cumulative return at the end of each episode. The reward in BipedalWalker ranges from about -110 (falling quickly) to +340 (walking efficiently).

![PPO BipedalWalker-v3 episode reward curve: rising from -110 to around 250, light blue is the raw value and dark blue is the 50-episode moving average](../../chapter10_ppo/images/ppo_bipedal_walker_reward.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 8.1-2: Episode reward curve. Light blue is the raw value of each episode, and dark blue is the 50-episode moving average. The green dashed line is the solved threshold (300 points).</em>
</div>

The overall trend of the curve has three phases:

- **First 300 episodes (~500k steps)**: the reward rises slowly from -110 to around 0. The policy is learning "not to fall". The robot goes from falling in every episode to running through all 1600 steps without falling. In this phase PPO accumulates experience and gradually finds movement patterns that keep balance.
- **Episodes 300-800 (500k-1.3M steps)**: the curve climbs quickly, from 0 to above 230. The policy starts to move forward effectively, crossing from "not falling" to "walking". The reward fluctuates strongly in this phase: some episodes go very well (100+ points) and some still end in a fall (-100 points).
- **After episode 800 (after 1.3M steps)**: the reward fluctuates between 200 and 260, and the policy has formed a stable gait. It enters a gradual optimization phase, and most episodes walk stably.

Note that the raw light-blue values fluctuate wildly: the rewards of two adjacent episodes can differ by more than 200 points. This is a feature of BipedalWalker: different initial perturbations and terrains make the same policy perform very differently. When reading the training curve, therefore, watch the trend of the moving average (the dark blue line) instead of the fluctuation of individual points.

### Policy Entropy

Policy entropy measures how "random" the action distribution output by the policy is. The higher the entropy, the more the policy treats all actions alike and is still exploring everywhere. The lower the entropy, the more the policy has formed a clear action preference.

![PPO BipedalWalker-v3 policy entropy curve: the Y axis is the negative entropy recorded by SB3, rising from -5.8 to -3.5, which corresponds to the actual entropy falling from 5.8 to 3.5](../../chapter10_ppo/images/ppo_bipedal_walker_entropy.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 8.1-3: Policy entropy (the negative value recorded by SB3) over training steps. Curve rising = actual entropy falling = the policy becoming more deterministic.</em>
</div>

Look at the Y axis: the values are **negative**, dropping from 0 to -5.8 and then slowly recovering to -3.5. This is the recording convention of Stable-Baselines3, which stores `entropy_loss = -H(π)` internally, the negative entropy. To read the true policy entropy, flip the sign of the Y-axis values.

Translated into actual meaning:

- **At the start of training** (after the first update): entropy ≈ 5.8. The standard deviation $\sigma$ of the Gaussian output by the policy is large, the sampling range of every joint torque is wide, and the actions look like random twitching.
- **After 2M training steps**: entropy ≈ 3.5. The standard deviation $\sigma$ has shrunk, the actions are more and more deterministic, and the policy "knows" which torque to output.
- **Overall trend**: entropy falls slowly from 5.8 to 3.5, which is normal. The policy keeps enough exploration while gradually focusing on effective actions.

If the entropy falls too fast (say from 5.8 straight down to 1.0), the policy has given up exploring too early and may be stuck at a local optimum, for example having learned that "standing still means not falling" while never trying to step. The `ent_coef` parameter of PPO guards against this: it adds an entropy regularization term to the loss, so that even if the policy wants to converge, it is forced to keep some randomness.

### Clip Fraction

Clip fraction is the most central monitoring metric of PPO. It is the fraction of action probability ratios $r_t(\theta) = \pi_{new}(a|s) / \pi_{old}(a|s)$ that fall outside the range $[1-\varepsilon, 1+\varepsilon]$ in one policy update (here $\varepsilon = 0.2$).

![PPO BipedalWalker-v3 clip fraction curve: fluctuating early in training and gradually converging to around 0.1](../../chapter10_ppo/images/ppo_bipedal_walker_clip.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 8.1-4: Clip fraction over training steps. The gray dashed line is the clip_range=0.2 reference line.</em>
</div>

Clip fraction directly reflects **how aggressive the policy updates are**:

- **0.05-0.15**: the normal range, with the policy improving steadily inside the safe interval.
- **> 0.2**: the update is too aggressive, the old and new policies differ too much, and there is a risk of policy collapse. This usually corresponds to a reward dive.
- **Close to 0**: the update is too conservative and the policy barely changes. If the reward is still low at this point, the learning rate may be too small.

The figure shows that the clip fraction fluctuates strongly early in training (the policy has just started and each update changes a lot) and gradually settles around 0.1 in the middle and later stages. The occasional small spikes (for example above 0.15) correspond to small policy adjustments and are no cause for concern as long as the value does not stay high.

If the clip fraction stays at 0, there are two possibilities: the learning rate is so small that the policy barely updates, or the policy has fully converged. To tell them apart, check whether the reward is still growing.

### Approximate KL Divergence

The KL divergence $D_{KL}(\pi_{old} \| \pi_{new})$ measures the difference between the old and new policies. The clipped objective of PPO limits the change of the probability ratio, which indirectly reduces the policy shift caused by one update.

![PPO BipedalWalker-v3 approximate KL divergence curve: below 0.016 most of the time and stable overall](../../chapter10_ppo/images/ppo_bipedal_walker_kl.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 8.1-5: Approximate KL divergence over training steps. Smaller KL = old and new policies closer = safer update.</em>
</div>

In the figure the KL divergence stays below 0.016 throughout, with no large spikes during the whole training run. This shows that the clipping mechanism of PPO works well on BipedalWalker: every policy update keeps the difference between the old and new policies within a safe range.

Putting KL divergence and clip fraction side by side, you will find that the two trends are highly correlated: when the clip fraction is high, the KL divergence is also high. This is natural, since more clipped actions means a larger overall policy shift. They are two views of the same phenomenon: the clip fraction looks at "how many actions were restricted", and the KL divergence looks at "how far the policy distribution as a whole moved".

If the KL divergence suddenly spikes above 0.05, some update pushed the policy too far, which is exactly the situation PPO wants to avoid. It did not occur in this experiment.

### How the Four Metrics Relate

The four metrics are not independent, and there are clear causal relations among them:

| Phenomenon            | Reward          | Policy entropy   | Clip fraction        | KL divergence        |
| --------------------- | --------------- | ---------------- | -------------------- | -------------------- |
| Normal training       | Gradually rising | Slowly falling  | 0.05-0.15            | 0.01-0.03            |
| Update too aggressive | Sudden dive     | Sharp change     | Spikes above 0.2     | Spikes above 0.05    |
| Premature convergence | Stagnant        | Close to 0       | Close to 0           | Close to 0           |
| Late training         | Stable and high | Stable and low   | Stable and low       | Stable and low       |

A reward dive almost always comes with a spike in the clip fraction and a sudden jump in the KL divergence. When all three metrics are abnormal, the policy update is too large, and you need to reduce the learning rate or increase `n_steps` (so that each rollout collects more data to stabilize the gradient estimate).

## 8.1.3 Criteria for Successful Walking

The reward of BipedalWalker-v3 has several components:

- **Forward reward**: each step gives a positive reward according to the distance moved to the right, higher for walking faster.
- **Joint efficiency penalty**: using excessive joint torque costs points, which encourages efficient movement.
- **Fall penalty**: if the robot falls (its head touches the ground), 100 points are deducted and the episode ends.

The environment defines "solved" as an average reward $\geq$ 300 over 100 consecutive episodes. In actual training, the score of a single episode can be read as follows:

- **$\geq$ 300**: high-quality walking, fast, steady, and efficient.
- **200-300**: it can walk but not steadily, and the efficiency or speed of the gait still has problems.
- **100-200**: it barely walks for a stretch and often falls.
- **$<$ 100**: it has basically not learned, and most episodes end in a fall.

Look first at the baseline of a random policy:

```python
import gymnasium as gym
import numpy as np

env = gym.make("BipedalWalker-v3")
rng = np.random.default_rng(0)

returns = []
for ep in range(50):
    obs, _ = env.reset(seed=ep)
    total_reward = 0.0
    for step in range(1600):
        action = rng.uniform(-1, 1, size=4)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    returns.append(total_reward)

print(f"Random policy mean return: {np.mean(returns):.1f}")
print(f"Best episode: {np.max(returns):.1f}")
print(f"Worst episode: {np.min(returns):.1f}")
```

The mean return of the random policy is roughly between -100 and -50, and almost every episode ends in a fall (100 points deducted). If the evaluation return after PPO training is still in this range, the policy has not learned any effective behavior.

The result of one run:

```text
Random policy mean return: -103.7
Standard deviation: 12.6
Best episode: -77.8
Worst episode: -124.7
```

## 8.1.4 Replaying the Three Training Stages

To get a feel for how PPO learns, we compare the policy at three different stages under the same training configuration. The three models use exactly the same hyperparameters and differ only in the number of training steps.

After training, a rendering script can generate replay GIFs:

```bash
python code/chapter07_ppo/render_bipedal_walker.py \
  --model output/ppo_bipedal_walker.zip \
  --output-dir output/bipedalwalker_episodes \
  --episodes 10 --seeds 0 1 2 3 4 5 6 7 8 9
```

### Early Stage (100k Steps, Return -35.8)

At 100k steps the policy has learned "not to fall". The robot can run all 1600 steps without falling but barely moves forward, and its limb movements look like keeping balance in place. The reward of -35.8 comes from the joint-torque penalty and the lack of forward distance.

![BipedalWalker at 100k steps: stands steadily but cannot walk, return -35.8](../../chapter10_ppo/images/bipedalwalker_ep1_100k.gif)

### Middle Stage (500k Steps, Return 109.3)

At 500k steps the policy is in the transition stage of learning to walk. The same model behaves very differently across episodes: in a lucky episode it earns over 100 points, and in an unlucky one it still falls and scores -100. The episode shown here is a successful walk. The robot moves forward, but the gait is not well coordinated and the speed is slow.

![BipedalWalker at 500k steps: starting to walk but very unsteady, return 109.3](../../chapter10_ppo/images/bipedalwalker_ep2_500k.gif)

### Late Stage (2M Steps, Return 295.1)

At 2M steps the policy has formed a stable and efficient gait. The joints are coordinated smoothly, and the walk finishes in 1118 steps (the 100k and 500k stages both need all 1600 steps).

![BipedalWalker at 2M steps: stable and efficient walking, return 295.1](../../chapter10_ppo/images/bipedalwalker_ep3_2m.gif)

Evaluation comparison of the three stages (average over 20 episodes):

| Training steps | Mean reward | Std dev | Behavior                                                          |
| -------------- | ----------- | ------- | ----------------------------------------------------------------- |
| 100k           | -34.1       | 3.3     | Stands steadily but cannot walk, runs all 1600 steps every episode |
| 500k           | -65.2       | 73.1    | Transition: about 15% of episodes walk, the rest still fall       |
| 2M             | 282.5       | 59.7    | Stable and efficient walking, the vast majority of episodes 290+  |

The learning trajectory of PPO on a continuous control task: first it learns "not to fall" (100k), then goes through the unstable "can walk occasionally" (500k), and finally forms "stable walking" (2M). The bimodal behavior at 500k steps deserves particular attention. The same model may get a reward of 100+ or -100 in different episodes, which shows that the policy is at the critical stage of crossing a behavioral threshold. This process is slower than in discrete control tasks, but the stage boundaries are clearer, because the policy space of a continuous action space is larger and each stage needs more data to break through.

## 8.1.5 State, Action, and the Continuous Policy

The 24-dimensional state of BipedalWalker can be divided into several groups:

| State component              | Dimensions | Meaning                                    |
| ---------------------------- | ---------- | ------------------------------------------ |
| `hull_angle`                 | 1          | Tilt angle of the hull                     |
| `hull_angular_velocity`      | 1          | Angular velocity of the hull               |
| `vx, vy`                     | 2          | Horizontal / vertical velocity of the hull |
| `hip1, hip2`                 | 2          | Hip joint angles of the two legs           |
| `knee1, knee2`               | 2          | Knee joint angles of the two legs          |
| `leg1_contact, leg2_contact` | 2          | Whether each foot touches the ground       |
| `lidar[0..9]`                | 10         | Lidar range readings (sensing the terrain ahead) |
| `hip_speed1, hip_speed2`     | 2          | Hip joint angular velocities               |
| `knee_speed1, knee_speed2`   | 2          | Knee joint angular velocities              |

The action is a 4-dimensional continuous vector, with every component in $[-1, 1]$:

| Action component | Meaning              |
| ---------------- | -------------------- |
| `action[0]`      | Leg 1 hip torque     |
| `action[1]`      | Leg 1 knee torque    |
| `action[2]`      | Leg 2 hip torque     |
| `action[3]`      | Leg 2 knee torque    |

PPO handles continuous actions differently from discrete ones. In a discrete action space, the policy network outputs the probability of each action and samples from it. In a continuous action space, the policy network outputs a Gaussian distribution, with mean $\mu(s)$ and standard deviation $\sigma(s)$, and the action is sampled from $\mathcal{N}(\mu, \sigma^2)$:

$$a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$$

This means PPO does not need to discretize continuous actions into a finite set of choices. For a quantity that needs fine control, such as a joint torque, discretization loses resolution: if only the three torque levels -1, 0, +1 were allowed, the robot would be very clumsy. A continuous policy can output fine values such as 0.37 or -0.82 directly, with far higher control precision than a discretized scheme.

The learning process of BipedalWalker usually goes through three stages:

1. **Standing (0-500k steps)**: the policy first learns not to fall. In this stage the reward rises slowly from -110 to around 0, and the robot goes from "falling right after the start" to "running all 1600 steps without falling". It still cannot advance, though, and only twists in place while standing.
2. **Shuffling (500k-1M steps)**: the policy starts to step tentatively but very unsteadily. Some episodes earn 100+ points and some still fall for -100. The standard deviation as high as 73 reflects the policy flipping back and forth between "can walk" and "cannot walk".
3. **Walking (after 1M steps)**: the gait gradually takes shape and walking becomes steadier. After 2M steps most episodes earn 290-299 points, with an occasional fall (about 1-2 out of 20 episodes).

These three stages are not strictly separated, and the boundaries shift between training runs. The overall trend holds: first learn "not to fall", then "to step", and finally "to walk efficiently".

## 8.1.6 Common Failures and Tuning

BipedalWalker fails to train more easily than the common discrete-action environments. If the result is unsatisfactory, check the following in order.

First, confirm that the number of training steps is sufficient. One million steps is a starting point, not an end point. If the curve is still rising but the slope is too small, training can continue:

```python
from stable_baselines3 import PPO
model = PPO.load("output/ppo_bipedal_walker.zip")
model.learn(total_timesteps=2_000_000, reset_num_timesteps=False)
```

Second, confirm that batch_size is large enough. The gradient variance of a continuous action space is higher than that of a discrete one, and `batch_size=64` may not be stable enough. This section uses 256, and if the training curve still oscillates sharply you can try 512.

Third, check the policy entropy. If the entropy drops to near zero early in training, the policy has converged prematurely to a suboptimal behavior (such as standing still all the time). You can raise `ent_coef` to 0.01.

Fourth, consider the network capacity. The default `MlpPolicy` uses a network with two layers of 64 neurons. For a 24-dimensional state this capacity may be insufficient. The network can be enlarged through `policy_kwargs`:

```python
model = PPO(
    policy="MlpPolicy",
    policy_kwargs=dict(net_arch=[128, 128]),
    ...
)
```

Reference for common hyperparameters:

| Parameter       | Setting in this section | What happens if it is unsuitable                                              |
| --------------- | ----------------------- | ----------------------------------------------------------------------------- |
| `learning_rate` | `3e-4`                  | Too large makes the joint torque output oscillate sharply, too small learns very slowly |
| `batch_size`    | `256`                   | Too small gives unstable gradient estimates, too large makes each update too slow |
| `n_steps`       | `2048`                  | Too small gives too little data per rollout, too large delays the policy update frequency |
| `ent_coef`      | `0.005`                 | Too small makes the policy converge early to standing still, too large never forms a stable gait |
| `clip_range`    | `0.2`                   | Too large makes the gait change abruptly and fall, too small stalls training  |
| `gamma`         | `0.99`                  | Too low focuses on short-term not falling and ignores long-term walking efficiency |

## 8.1.7 Why BipedalWalker

BipedalWalker has several key advantages for teaching:

- **Continuous action space.** BipedalWalker controls the joints with 4-dimensional continuous torques. This is exactly where PPO's core advantage over DQN lies: DQN cannot handle continuous actions directly and must discretize them first, while PPO supports them natively through a Gaussian policy.
- **A richer state space.** The 24-dimensional state contains 10 lidar range signals, a simplified simulation of the sensors of a real robot (lidar, tactile sensors). The agent has to infer terrain changes from this sensor data instead of simply reading position coordinates.
- **More complex learning dynamics.** BipedalWalker requires discovering a full gait cycle, which involves multi-joint coordination, weight transfer, and terrain adaptation. The policy does not simply switch among a few actions. It forms a stable movement pattern in a continuous space.

From CartPole (Chapter 1) to BipedalWalker, the difficulty and realism of the experiments clearly increase. The core mechanism of PPO is unchanged, though: clipping constrains the update size, the same batch of data is reused over several epochs, and the Actor-Critic optimizes the policy and value together. The added complexity comes from the environment and the task, not from the algorithm itself.

After running the experiment, you may wonder why PPO is so much more stable than the A2C of the previous chapter. In the next section we take apart the core mechanism of PPO, the trust-region constraint and the clipped objective: [8.2 Trust Regions and PPO-Clip](./trust-region-clipping).

## Section Summary

- `BipedalWalker-v3` shows PPO directly in a continuous action space: 4-dimensional continuous torques, a 24-dimensional state, and random terrain.
- PPO supports continuous actions natively through a Gaussian policy (outputting a mean and a standard deviation and sampling from them), with no need for discretization.
- Learning in BipedalWalker usually goes through the three stages "standing → shuffling → walking", and the number of training steps required is far higher than in common discrete-action tasks.
- This section uses the PPO implementation of Stable-Baselines3. The training entry point is `code/chapter07_ppo/ppo_bipedal_walker.py`, and the replay GIFs are produced with `render_bipedal_walker.py`.
- The environment defines "solved" as an average reward $\geq$ 300 over 100 episodes. The 2M-step training in this section reaches 282.5 ± 59.7, with most episodes steady in the 290-299 range.

## References

[^1]: Raffin, A., et al. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. _Journal of Machine Learning Research_, 22(268), 1-8.

[^2]: Schulman, J., et al. (2017). Proximal policy optimization algorithms. _arXiv preprint arXiv:1707.06347_.
