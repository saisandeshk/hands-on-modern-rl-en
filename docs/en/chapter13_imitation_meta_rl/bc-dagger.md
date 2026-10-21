# 11.1 Behavior Cloning and Interactive Imitation Learning

## Reading Guide

**Key Ideas**

- Understand how behavior cloning (BC) turns expert demonstrations into a supervised learning problem. It is the simplest and most commonly used imitation learning method.
- See the fatal flaw of behavior cloning, **distribution shift**: during training it sees expert states, and at deployment, once it makes a mistake, it can never recover.
- Understand how DAgger solves distribution shift through the interactive procedure of "let the policy run, let the expert label".
- Compare the data requirements and applicable settings of the three routes BC, DAgger, and GAIL, and understand why we need to evolve from "imitating actions" to "inferring rewards".

In the earlier chapters we learned many reinforcement learning algorithms: DQN, PPO, SAC, and so on. They share a common premise: **the environment provides a reward function**. At every step the environment tells the agent how many points that step earned, and the algorithm tries to maximize the total.

But let's stop and think. In reality, for many tasks you can find expert demonstrations but cannot write down a reward function. For example:

- Autonomous driving: human drivers have driven millions of kilometers and there is a great deal of video, but you cannot define "what good driving is" in a few lines of code
- Robot manipulation: workers assemble parts skillfully, but it is hard to write down precisely how every joint should move
- Large language models: high-quality question-answer pairs are everywhere, but the reward function for a "good answer" simply cannot be written by hand

These scenarios share one feature: **the expert can do it, but you cannot say why the expert does it that way**. If we do not design a reward and instead let the agent "copy the homework", watching what action the expert takes in which state and learning to do the same, would that work?

This is the problem that **imitation learning** addresses. Chapter 11 unfolds in three steps. First we learn expert actions directly with behavior cloning and DAgger. Then we infer rewards from demonstrations with inverse reinforcement learning and GAIL. Finally we enter MAML, RL², PEARL, and in-context reinforcement learning, which study how a policy can adapt quickly to new tasks. This section first answers four questions: how behavior cloning is trained, why errors accumulate, how DAgger collects erroneous states, and what data each of the routes needs.

## 1. Writing Expert Demonstrations as Supervised Learning

Recall the policy-gradient methods of Chapter 6, which all assume that the environment provides a reward. But in many real tasks we have only **expert demonstrations**: trajectories of human drivers, operation records of skilled workers, high-quality question-answer pairs. **Imitation learning** learns a policy directly from demonstrations and skips the design of a reward function.

You may ask: "How can we learn without a reward?" The answer is simple: turn it into supervised learning.

### 1.1 The Behavior-Cloning Objective

The most direct method is to write the expert data as supervised learning samples: the state $s$ is the input and the expert action $a$ is the label. The higher the probability the policy gives the expert action, the smaller the loss:

$$\mathcal{L}_{BC}(\theta) = -\mathbb{E}_{(s, a) \sim \mathcal{D}_{\text{expert}}}\left[\log \pi_\theta(a \mid s)\right]$$

Let's take this expression apart:

- $\mathcal{D}_{\text{expert}}=\{(s_i,a_i)\}_{i=1}^N$: the expert demonstration dataset, containing $N$ "state-action" pairs, meaning that in state $s_i$ the expert took action $a_i$
- $\pi_\theta(a\mid s)$: our policy network, the probability that it chooses action $a$ in state $s$
- $\log \pi_\theta(a \mid s)$: the log probability of the expert action. The larger this value, the more the policy "agrees" with the expert's choice.
- The minus sign: turns "raise the probability of the expert action" into a minimization problem. Minimizing the negative log-likelihood is equivalent to maximizing the probability of the expert action.

Let's compute this loss with concrete numbers. The expert data contains one sample: the car is exactly in the center of the lane (state $s$), and the expert's action is "steer gently to the left" (action $a$). The outputs of two policies on this sample are:

- Policy one gives this action probability $\pi_1(a \mid s) = 0.7$, with loss $-\log 0.7 \approx 0.36$
- Policy two gives this action probability $\pi_2(a \mid s) = 0.95$, with loss $-\log 0.95 \approx 0.05$

Policy two "agrees" more with the expert, so its loss is smaller. Behavior cloning training minimizes this kind of loss over all samples in the dataset, which is the same operation as cross-entropy training in handwritten-digit classification.

Discrete actions usually use the cross-entropy loss (a classification problem), and continuous actions can use the mean squared error or the negative log-likelihood of a Gaussian distribution. Supervised fine-tuning (SFT) of LLMs also uses the same conditional-likelihood objective, except that the "action" becomes the next token. It is essentially behavior cloning.

```python
def behavior_cloning_step(policy_net, expert_batch):
    states, actions = expert_batch
    log_probs = policy_net.log_prob(states, actions)
    loss = -log_probs.mean()  # Negative log-likelihood
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

This looks too simple: collect expert data, do supervised learning, and that's it? If it were really this simple, reinforcement learning would long ago have been replaced by supervised learning. Let's stop and think: under what conditions does behavior cloning go wrong?

## 2. Why Behavior-Cloning Errors Accumulate

During training, BC sees the state distribution visited by the expert, $d_{\text{expert}}(s)$, but at deployment it visits $d_{\pi_\theta}(s)$, generated by the current policy. One small mistake can carry the agent into a state absent from the training set, and the subsequent actions are then more likely to keep going wrong.

Let's get a feel for this with a concrete example. Suppose you are learning to drive, and the instructor teaches you only in "correct driving states": the car in the middle of the lane, a suitable speed, a normal distance to the car ahead. You learn well in these states, with steering, throttle, and braking all right.

But if you turn the steering wheel a bit too much at some moment and the car drifts toward the edge of the lane, you enter a state that you **have never seen during training**. You do not know what to do, and may panic and turn the wheel again, drifting even further. The next state is even more unfamiliar, and the mistakes get more outrageous... This is **error accumulation**, also called **distribution shift**.

Let's compute with numbers. Suppose the policy's error rate on states visited by the expert is only $\epsilon=0.01$, agreeing with the expert 99% of the time. Looking at each step alone, this error rate is negligibly low. But the task runs for $T$ consecutive steps, and the probability of making no error throughout is $(1-\epsilon)^T$, which decays sharply with the number of steps:

| Task length $T$                  | 10 steps | 50 steps | 100 steps | 1000 steps |
| -------------------------------- | -------- | -------- | --------- | ---------- |
| Probability of no error overall  | 0.90     | 0.61     | 0.37      | 0.00004    |

In a 100-step task, in two out of three episodes at least one mistake is made along the way. And this is still an optimistic estimate: it assumes that the error rate stays at 1% after deviating from the expert trajectory. In reality, once the agent deviates, the states are no longer in the training set and the error rate rises sharply. The DAgger paper (Ross et al. 2011) writes the cumulative task cost on the order of $O(T^2\epsilon)$:

$$\mathbb{E}\left[\sum_{t=0}^T \mathbb{1}[\pi_\theta(s_t) \neq \pi^*(s_t)]\right] \leq O(T^2 \epsilon)$$

What does this expression tell us?

- $T$ is the task length (number of steps) and $\epsilon$ is the supervised learning error
- $T^2$ means that errors **snowball**: an early error not only affects the current step but also changes the states seen in many later steps
- The longer the task, the more costly it is to train only on expert states. For a 1000-step task the cost may be a million times that of a 1-step task.

This is why behavior cloning can work on short, simple tasks (such as easy levels of Atari games), yet often fails on long-horizon decision tasks (autonomous driving, robot manipulation).

::: details Bonus: The DAgger Theoretical Bound of Ross et al.
The error bound of BC is $O(T^2\epsilon)$, which comes from the accumulation of "covariate shift": each time the policy meets an out-of-distribution state, it may make a new mistake, and these mistakes push later states even further away. Ross et al. proved that if the expert can continually label the states visited by the current policy during training, this bound can be improved to $O(T\epsilon)$, linear instead of quadratic.
:::

## 3. Using DAgger to Collect the States the Policy Actually Visits

Since the problem is that "states in which the policy itself makes mistakes were never seen during training", the solution is direct: **let the policy run until it reaches states where it makes mistakes, and then ask the expert to tell it what to do in those states**. This is the core idea of DAgger (Dataset Aggregation).

Dataset Aggregation directly supplements the states that the policy will visit and that the expert data does not cover. The current policy first executes the task, and the expert then provides the correct actions for these states. Here is the code:

```python
def dagger(env, expert, policy_net, n_iterations=20, n_traj_per_iter=50):
    dataset = []
    for it in range(n_iterations):
        # 1. Roll out with the current policy (note: not with the expert!)
        trajectories = []
        for _ in range(n_traj_per_iter):
            s = env.reset()
            traj = []
            done = False
            while not done:
                # β mixing: use the expert more early on for safety, the policy more later
                beta = max(0.0, 1.0 - it / 10)
                if np.random.rand() < beta:
                    a = expert(s)
                else:
                    a = policy_net.act(s)
                s_next, r, done, _ = env.step(a)
                traj.append((s, a))
                s = s_next
            trajectories.append(traj)

        # 2. Key: ask the expert to relabel the states the policy visited (including failure states)
        for traj in trajectories:
            for s, _ in traj:
                a_expert = expert(s)
                dataset.append((s, a_expert))

        # 3. Retrain the policy on the enlarged dataset
        train_bc(policy_net, dataset)
```

Let's take apart the three key steps of DAgger:

**Step one: run the policy.** Let the current policy (the still imperfect one) interact with the environment and collect the trajectories it actually visits. Note that the code has a $\beta$ mixing strategy: in early iterations it uses the expert more (to avoid crashing dangerously at the start), and later it gradually uses the policy more (so that it exposes its own mistakes).

**Step two: the expert labels.** This is the core and most "expensive" step of DAgger: whether the actions in a trajectory were chosen by the expert or the policy, for **every state $s$ that appears in the trajectory**, ask the expert for "what the correct action in this state should be". Even if the policy did something foolish in this state, the expert must tell it the correct answer.

**Step three: retrain.** Add these newly labeled "states where the policy goes wrong" to the dataset and redo behavior cloning. The dataset then contains both the states the expert normally visits and the states where the policy tends to make mistakes.

Let's follow the training through. In the first few rounds, $\beta$ is close to 1, the car is basically driven by the expert, and everything collected is "middle of the lane, normal speed", so labeling is effortless. After a few rounds, $\beta$ drops to one half, the car is controlled by the not-yet-skilled policy half the time, and states such as "crossing the lane line" and "going too fast" start to appear. These states never occur in the expert demonstrations, and now the expert gives the correct action for them, so the dataset fills in these accident-prone positions. In the late stage of training $\beta$ goes to zero, the policy drives on its own and still makes mistakes, but every erroneous state is labeled and stored, and in the next round the policy learns its lesson at these positions. In this way the dataset grows round by round and gradually covers exactly the policy's own distribution $d_{\pi_\theta}$.

Under conditions such as no-regret online learning, the cumulative cost can be improved from BC's $O(T^2\epsilon)$ to the order of $O(T\epsilon)$, from quadratic to linear!

But you may already have thought of a problem: the price is that the expert must be called again and again during training.

## 4. Comparing BC, DAgger, and GAIL

Let's put the three methods side by side:

| Method | Source of training data                  | Solves distribution shift | Needs online expert labeling     |
| ------ | ---------------------------------------- | ------------------------- | -------------------------------- |
| BC     | Offline expert data only                 | ❌                        | ❌                               |
| DAgger | Expert + states visited by the policy    | ✅                        | ✅ (the key limitation)          |
| GAIL   | Expert + policy rollouts                 | ✅ (implicitly)           | ❌ (only state-action pairs needed) |

DAgger's engineering bottleneck is that it **needs online interaction with the expert**. It is hard for a human driver to label the correct action in real time for the abnormal states the policy visits. You cannot have an instructor sit beside you and tell you immediately, every time the car veers off, how many degrees the wheel should turn, and you certainly cannot have the instructor label thousands upon thousands of failed trajectories.

This pushes us to think about the next question: can we avoid having the expert label in real time? Can we, from the expert's existing trajectories alone, learn not only "what action to take" but also infer "why it was done" in reverse, that is, the reward function? This is the problem that the next section, on inverse reinforcement learning and GAIL, addresses.

## Section Summary

Behavior cloning (BC) is the most basic imitation learning: it trains a policy with expert trajectories as supervised data. It is simple, cheap to compute, and essentially the same thing as SFT for large language models. But it has the problem of **distribution shift**: during training it learns only on the expert state distribution, and at deployment, once it deviates, it can never recover, with errors accumulating as $O(T^2\epsilon)$. DAgger solves this problem by having the expert correct the agent's actual trajectories, improving the error bound to $O(T\epsilon)$, at the price of requiring the expert to label repeatedly online.

The next section, [11.2 Inverse Reinforcement Learning and GAIL](./irl-gail), no longer imitates actions directly but **infers the reward function from expert behavior in reverse**. This is inverse reinforcement learning (IRL).
