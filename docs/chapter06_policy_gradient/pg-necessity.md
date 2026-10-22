---
title: 'Supplement: Why Policy Gradients'
---

# Supplement: Why Policy Gradients

## Reading Guide

**Key Ideas**

- Recall the core idea of DQN in Chapter 5: learn $Q(s,a)$, then choose actions with $\arg\max$.
- Understand why the action-enumeration mechanism of classical DQN has difficulty handling continuous action spaces directly.
- See why policy-based methods learn $\pi_\theta(a|s)$ directly, and what the essential differences between the two routes are in action space, exploration mechanism, and data use.

## What DQN Got Right

DQN in Chapter 5 follows a clear route: first approximate $Q(s,a)$ with a neural network to score each action, then pick the highest-scoring one with $\arg\max_a Q(s,a)$. The underlying logic of this route is: instead of learning "what to do" directly, first learn "how many points each action is worth," and then pick the best from them. The policy is implicit: it is hidden in the $\arg\max$ over the table of $Q$ values.

Let's look at this process at one concrete moment in CartPole. Suppose the cart's current state is $s = [0.05,\; 0.1,\; -0.02,\; 0.3]$ (position, velocity, pole angle, angular velocity). The DQN network does one forward pass on this state and outputs two $Q$ values:

| Action     | $Q(s,a)$ |
| ---------- | -------- |
| Push left  | $0.8$    |
| Push right | $1.2$    |

$\arg\max$ compares them one by one and takes the action that corresponds to the maximum:

$$
a^* = \arg\max_a Q(s,a) = \arg\max\{0.8,\; 1.2\} = \text{push right}.
$$

The key premise of this process is that the action set is finite and not large, so the $Q$ value of every action can be computed and then compared. CartPole has only 2 actions, so 2 numbers are compared; LunarLander has 4 actions, so 4 numbers are compared. If we expand the actions to 10, 100, or even 1000, $\arg\max$ is still feasible: a few more $Q$ values have to be computed and more comparisons made, but the computation grows linearly and there is no difficulty in principle.

| Number of actions | How many $Q$ values to compute | $\arg\max$ comparisons | Feasibility    |
| ----------------- | ------------------------------ | ---------------------- | -------------- |
| 2                 | 2                              | 1                      | Easy           |
| 4                 | 4                              | 3                      | Easy           |
| 1000              | 1000                           | 999                    | Feasible       |
| $10^6$            | $10^6$                         | $10^6 - 1$             | Feasible, slow |
| $\infty$          | $\infty$                       | $\infty$               | Impossible     |

The last row is where the problem lies. When the action space is continuous, the number of actions is infinite, so it is impossible to compute a $Q$ value for every action, let alone find the maximum among infinitely many numbers.

## Where $\arg\max$ Breaks Down

$\arg\max$ requires comparing the $Q$ values of all actions. As long as the number of actions is finite, this poses no problem. But the action spaces of many practical tasks are continuous, with infinitely many actions.

### Dimension Explosion

Controlling a robotic arm is a typical example. The shoulder, elbow, and wrist joints each have several degrees of freedom, and each degree of freedom can apply a continuous torque $\tau \in [-10, 10]$. Suppose there are 6 joints; the action space is $[-10, 10]^6$, infinitely many points in a six-dimensional continuous space. It is impossible to compute a $Q$ value for every point, let alone find the $\arg\max$ among infinitely many points.

A natural idea is to discretize the continuous space and then use $\arg\max$. For example, take only 100 torque values for each joint, approximating the continuous space by a finite grid. With 100 values for each of 6 joints, the total number of actions is:

$$
N = 100^6 = 10^{12}.
$$

For $10^{12}$ actions, a $Q$ value has to be computed for each one. Suppose one forward pass of the neural network takes $1\mu\text{s}$ ($10^{-6}$ seconds); then the computation time for **one action selection** is:

$$
T = 10^{12} \times 10^{-6}\text{s} = 10^6\text{s} \approx 11.6 \text{ days}.
$$

It takes 11.6 days to take a single action. By contrast, a policy network needs only one forward pass: feed the state into the network and output the action vector directly, which takes about $1\text{ms}$:

| Method               | Computation for one action selection                 | Time                 |
| -------------------- | ---------------------------------------------------- | -------------------- |
| DQN + discretization | $10^{12}$ forward passes                             | $\approx 11.6$ days  |
| Policy network       | 1 forward pass, directly outputs $\mu = f_\theta(s)$ | $\approx 1\text{ms}$ |

And this is only 6 joints, each discretized to just 100 values. When a real robotic arm has more joints and higher precision requirements, the number of discretized actions grows exponentially. This is the **curse of dimensionality**: every additional joint multiplies the total number of actions by a factor.

### Large Model Generation and Why a Distribution Beats Greedy

Text generation by large models faces a similar problem. At each step one token has to be chosen from tens of thousands. Suppose the vocabulary size is 50,000; $\arg\max$ itself is not difficult, since it only needs to compare 50,000 numbers, and the computation is entirely acceptable. The problem lies not in computational feasibility but in generation quality.

Suppose we are about to generate the next word, and the network outputs the probabilities of some tokens (here Chinese tokens, shown in pinyin with a gloss):

| token                | $P(\text{token} \mid \text{context})$ |
| -------------------- | ------------------------------------- |
| "shi" (is)           | $0.40$                                |
| "wei" (be, as)       | $0.25$                                |
| "nai" (be, literary) | $0.15$                                |
| "xi" (be, literary)  | $0.10$                                |
| ...                  | ...                                   |

$\arg\max$ (greedy decoding) picks the most probable "shi" every time. If the probability distributions at the next several positions are similar, greedy decoding will output the same character over and over. Sampling from the probability distribution, on the other hand, picks "wei" with 25% probability and "nai" with 15% probability; this randomness is exactly what fluent text generation requires. A policy network naturally outputs a probability distribution $\pi_\theta(a|s)$, and sampling is the generation process itself.

## Learning the Policy Directly

Since "score first, then pick" does not work, let's take another road: **skip the $Q$ values and learn the policy $\pi_\theta(a|s)$ directly**. Instead of asking "how many points is each action worth," we learn directly "what to do in what situation."

This is exactly the core idea of the [policy objective $J(\theta)$](./reinforce) introduced in Chapter 2 and derived in Section 6.1: define a policy objective function $J(\theta)$, and then directly optimize the parameters $\theta$ so as to maximize $J(\theta)$.

The difference between the two routes can be made clear with an analogy: a value-based method is like a food critic, who scores every dish and then picks the one with the highest score; a policy-based method is like an experienced chef, who does not need to score anything and directly knows what dish to cook given which ingredients and which occasion.

### What the Policy Network Outputs

The output of the policy network $\pi_\theta(a|s)$ is not an action score but a probability distribution. Take CartPole as an example: feed in the state $s = [0.05, 0.1, -0.02, 0.3]$, the network does a forward pass, and a final Softmax layer outputs the probability of each action:

$$
\pi_\theta(\text{left} \mid s) = 0.3, \quad \pi_\theta(\text{right} \mid s) = 0.7.
$$

| Symbol                        | Meaning                                                              |
| ----------------------------- | -------------------------------------------------------------------- |
| $\pi_\theta$                  | The policy network with parameters $\theta$                          |
| $\pi_\theta(a \mid s)$        | The probability of choosing action $a$ in state $s$                  |
| $s = [0.05, 0.1, -0.02, 0.3]$ | The current state (position, velocity, pole angle, angular velocity) |
| $[0.3, 0.7]$                  | The action-probability vector output by the network                  |

Actions are chosen by **sampling** and not by comparison. To sample from the distribution $[0.3, 0.7]$: generate a uniform random number $u$ on $[0,1)$; if $u < 0.3$, push left, otherwise push right. For example, with $u = 0.65$, since $0.65 > 0.3$, we choose "push right."

### Compared with DQN

In the same state, the two methods take completely different paths:

**DQN path:** the network outputs $Q$ values $\to$ $\arg\max$ $\to$ deterministic action

$$
[0.8,\; 1.2] \;\xrightarrow{\arg\max}\; \text{push right} \quad (\text{always this one}).
$$

**Policy network path:** the network outputs probabilities $\to$ sampling $\to$ stochastic action

$$
[0.3,\; 0.7] \;\xrightarrow{\text{sampling}}\; \begin{cases} \text{push left} & \text{with probability } 0.3 \\ \text{push right} & \text{with probability } 0.7 \end{cases}
$$

The most critical difference between the two is that, once trained, DQN always outputs the same action for the same state (a deterministic policy), while the policy network may output different actions for the same state (a stochastic policy). This randomness is a feature and not a defect: it naturally includes exploration, with no need for an extra $\varepsilon$-greedy.

For continuous action spaces, the policy network's output takes the form of the parameters of a Gaussian distribution. For example, if a robotic arm needs to output the torques of 6 joints, the policy network outputs a mean vector $\mu_\theta(s) \in \mathbb{R}^6$ and a standard deviation vector $\sigma_\theta(s) \in \mathbb{R}^6$, and then samples the action from $\mathcal{N}(\mu_\theta(s),\; \text{diag}(\sigma_\theta^2(s)))$. There is no need for discretization or $\arg\max$; one forward pass is enough.

## Differences Between the Two Routes

|                           | Value-Based (DQN)                                 | Policy-Based (Policy Gradient)                                        |
| ------------------------- | ------------------------------------------------- | --------------------------------------------------------------------- |
| What it learns            | $Q(s,a)$: how many points each action is worth    | $\pi_\theta(a\|s)$: with what probability each action should be taken |
| How it picks actions      | $\arg\max_a Q(s,a)$ (take the highest score)      | Sample from $\pi_\theta(\cdot\|s)$                                    |
| Policy form               | Deterministic (always picks the highest score)    | Stochastic (outputs a probability distribution)                       |
| Action space              | Discrete only                                     | Discrete + continuous                                                 |
| Exploration mechanism     | Added externally ($\varepsilon$-greedy)           | Built in (a probability distribution naturally includes exploration)  |
| Data use                  | Off-policy (the replay buffer can reuse old data) | On-policy (must use fresh data from the current policy)               |
| Variance                  | Low (TD targets are relatively stable)            | High (Monte Carlo returns fluctuate a lot)                            |
| Representative algorithms | DQN (Chapter 5)                                   | REINFORCE (this chapter) → PPO (Chapter 8)                            |

Let's explain the key differences row by row.

**Action space**: this is the main basis for choosing a route. DQN's $\arg\max$ simply cannot be computed in a continuous space. Policy gradients output a probability distribution directly: Softmax for discrete actions and a Gaussian distribution for continuous actions, and changing the output layer is all it takes.

**Exploration mechanism**: DQN's policy is deterministic (it always picks the $\arg\max$), and exploration relies on $\varepsilon$-greedy (recall: [the three components of DQN](../chapter05_dqn/dqn-components)). $\varepsilon$ has to be scheduled by hand: too large wastes samples, too small does not explore enough. Policy gradients naturally output a probability distribution, so exploration is built in: if the network believes an action has a 30% probability of being worth trying, it will try it with 30% probability.

**Data use**: this is the most practical engineering difference between the two routes. DQN is off-policy: the replay buffer stores old data, which can be used for training again and again. Policy gradient is on-policy: the $\mathbb{E}_{\pi_\theta}$ in the gradient estimate requires data generated by the current policy. As soon as the policy is updated, the old data becomes invalid. Its data efficiency is naturally lower than DQN's, which is the biggest engineering shortcoming of policy gradients.

### A Numerical Comparison of the Two Routes in the Same Scenario

Let's walk through the whole process of the two routes with a concrete scenario. The setup: 3 states $\{s_1, s_2, s_3\}$, 2 actions $\{a_1, a_2\}$, and discount factor $\gamma = 0.9$.

**DQN route: learn $Q$ values, pick actions with $\arg\max$**

Suppose that after training, DQN has learned the following $Q$ table:

| State | $Q(s, a_1)$ | $Q(s, a_2)$ |
| ----- | ----------- | ----------- |
| $s_1$ | $1.5$       | $2.3$       |
| $s_2$ | $0.8$       | $-0.4$      |
| $s_3$ | $3.1$       | $2.9$       |

Apply $\arg\max$ in each state:

$$
\pi(s_1) = \arg\max\{1.5,\; 2.3\} = a_2,
$$

$$
\pi(s_2) = \arg\max\{0.8,\; -0.4\} = a_1,
$$

$$
\pi(s_3) = \arg\max\{3.1,\; 2.9\} = a_1.
$$

The result is a deterministic policy table: each state always picks the same action. If exploration is needed, $\varepsilon$-greedy must be layered on top; for example, with $\varepsilon = 0.1$, an action is chosen at random with 10% probability:

| State | Probability of choosing $a_1$ | Probability of choosing $a_2$ |
| ----- | ----------------------------- | ----------------------------- |
| $s_1$ | $0.1 \times 0.5 = 0.05$       | $0.9 + 0.1 \times 0.5 = 0.95$ |
| $s_2$ | $0.9 + 0.1 \times 0.5 = 0.95$ | $0.1 \times 0.5 = 0.05$       |
| $s_3$ | $0.9 + 0.1 \times 0.5 = 0.95$ | $0.1 \times 0.5 = 0.05$       |

The exploration of $\varepsilon$-greedy is uniform: the 10% random exploration is split evenly between $a_1$ and $a_2$. Even though the gap between $Q(s_3, a_2) = 2.9$ and $Q(s_3, a_1) = 3.1$ is tiny (the two actions are nearly equally good), the exploration probability is allocated in exactly the same way as in $s_1$, where the gap is large.

**Policy-gradient route: learn $\pi_\theta(a|s)$, pick actions by sampling**

Suppose the policy network has learned the following probability distributions:

| State | $\pi(a_1 \mid s)$ | $\pi(a_2 \mid s)$ |
| ----- | ----------------- | ----------------- |
| $s_1$ | $0.2$             | $0.8$             |
| $s_2$ | $0.9$             | $0.1$             |
| $s_3$ | $0.55$            | $0.45$            |

In $s_3$, the policy network considers the two actions nearly equally good ($0.55$ vs. $0.45$), so the exploration proportion is naturally high; in $s_1$ and $s_2$, one action is clearly better than the other, so the exploration proportion is naturally low. There is no need to tune $\varepsilon$ by hand, because the probability distribution itself encodes "how much to explore."

Putting the key numbers of the two routes side by side:

| Comparison dimension                   | DQN at $s_3$                                                  | Policy gradient at $s_3$                                         |
| -------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------- |
| Network output                         | $Q(s_3, a_1) = 3.1$, $Q(s_3, a_2) = 2.9$                      | $\pi(a_1 \mid s_3) = 0.55$, $\pi(a_2 \mid s_3) = 0.45$           |
| How the action is chosen               | $\arg\max\{3.1, 2.9\} = a_1$                                  | Sampling: $a_1$ with 55% probability, $a_2$ with 45% probability |
| Exploration                            | Add $\varepsilon$-greedy (uniform random)                     | Built in (the probability distribution adapts)                   |
| Applicable to continuous action spaces | No (requires discretizing and computing a grid of $Q$ values) | Yes (directly outputs Gaussian parameters)                       |

The core difference is in the last row: DQN's $\arg\max$ confines the action space to a finite discrete set; policy gradients skip the step of "scoring every action" and directly output a probability distribution over "how to choose actions," so continuous action spaces are no longer an obstacle.

## The Two Routes Are Not Opposed

Each of the two routes has its strengths and weaknesses, but they are not irreconcilable. The Actor-Critic of Chapter 7 will merge the two routes: use a policy network to make decisions and a value network to reduce variance. Before that, however, we first need to lay a solid mathematical foundation for the policy-based route.

The next section starts from the policy objective function, derives the policy gradient theorem, and introduces the REINFORCE algorithm: [REINFORCE Algorithm](./reinforce).
