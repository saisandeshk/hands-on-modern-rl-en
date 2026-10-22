# 9.3 Model-Based Reinforcement Learning

## Reading Guide

**Key Ideas**

- Understand the basic difference between model-based and model-free RL, and why "learn an environment model first, then train the policy" can improve sample efficiency by a factor of 10-100.
- Master the Dyna framework: how to treat the environment model as an "extra data generator" and perform N simulated updates after every real interaction.
- Understand the problem of compounding model error: why long rollouts make prediction error explode, and how the Lipschitz constant characterizes the problem.
- See how PETS uses probabilistic ensembles to quantify two kinds of model uncertainty, and combines them with CEM planning for robust control.
- Master the short-horizon rollout idea behind MBPO: why predicting only 5 steps with the model and then resetting to a real state avoids error accumulation while still improving sample efficiency substantially.

In the previous section, [9.2](./td3-sac), we dealt with the stability of model-free continuous-control algorithms. TD3 used three engineering patches and SAC reformulated the objective through maximum entropy, so that DDPG-style methods finally became dependable on MuJoCo and could deliver quite good policies after a million training steps.

Let's pause and think about that number, though. A million steps in a simulator such as MuJoCo may take only a few hours. On a real robot the picture is different. The motors of a robot arm wear down, battery life is limited, and a collision in the real world can damage the equipment, so **every real-environment sample has a cost**. A million steps on a real robot can mean weeks or even months of uninterrupted operation, which is simply unaffordable in practice.

This leads to the central question of model-based RL: **can an agent learn a good policy while interacting with the real environment very little?** Model-based RL answers by learning an environment model, a predictor that says "if I take action $a$ in state $s$, what will the next state be and how much reward will I receive?", and then training the policy mainly inside this model. The number of real interactions drops from a million steps to the order of ten thousand.

In this section we take apart the three mainstream routes of model-based RL: plain data augmentation (Dyna), probabilistic ensembles with trajectory sampling (PETS), and short-horizon rollouts (MBPO).

## The Fundamental Difference Between Model-Based and Model-Free RL

All the earlier algorithms (DDPG, TD3, SAC) are **model-free**. You may ask: "What does model-free mean? What is it free of?" Model-free does not mean "there is no model". It means that **the agent makes no attempt to understand or model how the environment works**. It behaves like a learner who is told: you give me a state, I output an action, you give me a reward, and I adjust my policy based on the reward signal alone. Why the environment returns that reward, and how the world changes after my action, is something I do not care about at all.

**Model-based RL** takes the opposite route. Before learning a policy, or while learning it, the agent first learns an environment model:

$$
\hat{P}(s' \mid s, a), \quad \hat{R}(s, a)
$$

Let's look at these two symbols separately:

- $\hat{P}(s' \mid s, a)$ is the **dynamics model**. Given the current state $s$ and action $a$, it predicts the probability distribution of the next state $s'$. The hat $\hat{}$ indicates that this is a learned approximation, not the true environment $P$.
- $\hat{R}(s, a)$ is the **reward model**. Given state $s$ and action $a$, it predicts the immediate reward.

With these two models the agent can "imagine in its head": what would happen if I took this action now, and how much reward would I get? It can then plan, or train a policy, on these imagined trajectories.

### Why Use a Model?

The central motivation is **sample efficiency**.

Let's get a feel for the gap with concrete numbers. A model-free algorithm such as SAC usually needs 1 million steps (1M steps) on MuJoCo to converge to a good policy. A model-based algorithm? MBPO needs only 10K-100K steps to reach the same performance, an **improvement in sample efficiency of 10-100 times**.

That is not a small number. If each real step takes 1 second (a physical robot needs time to move), 1 million steps means about 11.5 days of continuous operation, while 100,000 steps take about 1 day. For more expensive settings, such as autonomous driving, industrial robots, and chemical-reaction control, this gap can be the difference between "usable" and "unusable".

Why can model-based methods save so many real samples? Once the model is learned, you can **"imagine" samples inside the model as many times as you like**, and these imagined samples cost almost nothing. It is like a person learning chess: you do not need to play every game to the end. You can work through it in your head: "if I make this move, how will my opponent respond, and then how will I respond?" This "mental rehearsal" is model-based planning.

But, as you may already have wondered, what if the model is wrong? Will a policy trained on a wrong model also be wrong in the real environment? This is the core tension of model-based RL: **model bias**. As we will see, Dyna, PETS, and MBPO are essentially three different ways of coping with this tension.

### Overview of the Three Paradigms

Before going into details, let's list the three mainstream paradigms of model-based RL to give you the full picture:

| Paradigm | Core idea                                    | Representative algorithm | Applicable setting                                      |
| -------- | -------------------------------------------- | ------------------------ | ------------------------------------------------------- |
| **Dyna** | Model as data augmentation                   | Dyna-Q                   | Discrete actions, fast training                         |
| **PETS** | Probabilistic ensemble + trajectory sampling | PETS                     | High-precision control, where model uncertainty matters |
| **MBPO** | Short-horizon rollouts                       | MBPO                     | General continuous control                              |

Let's begin with the simplest, Dyna.

## The Model as Data Augmentation

The Dyna framework, proposed by Sutton in 1990, is the most classic and most straightforward idea in model-based RL. Its premise is surprisingly simple: **the model is just an extra data generator**. Each time you interact once with the real environment and obtain a real experience $(s, a, r, s')$, you use that experience to do several things at once:

1. Use it for one model-free update as usual (for example, Q-learning);
2. Use it to train your environment model;
3. Then, and this is the key step, "imagine" N fake experiences from the model and use them for N further model-free updates.

Let's read the code to understand the flow:

```python
for step in range(total_steps):
    # 1. Real interaction
    a = policy.select(s)
    s_prime, r = env.step(a)
    replay_buffer.add(s, a, r, s_prime)

    # 2. Update the model-free algorithm (e.g., Q-Learning) with real data
    q_learning_update(replay_buffer.sample())

    # 3. Train the environment model with real data
    model.train(s, a, r, s_prime)

    # 4. Use the model to generate "fake" data and do N more Q-Learning updates
    for _ in range(N):  # N = 10-100
        s_sim, a_sim = replay_buffer.sample_state_action()
        s_sim_next, r_sim = model.predict(s_sim, a_sim)
        q_learning_update(s_sim, a_sim, r_sim, s_sim_next)
```

Do you see it? After every real interaction we perform N extra "simulated" Q-learning updates. These updates use no real environment; they use data predicted by the model. With N=10, each real interaction is accompanied by 11 Q-learning updates, and in theory the sample efficiency improves by about a factor of N.

That idea looks wonderful, doesn't it? But wait, there is a problem.

### The Key Limitation of Dyna

Let's stop and think. The code above uses a deterministic model that directly predicts the next state $s'$. Even if you use a probabilistic model, one fundamental problem cannot be avoided: **in a continuous physical environment (such as MuJoCo), if you repeatedly feed the model's predictions back in as the next input, the error snowballs and grows larger and larger**.

Why does this happen? Let's do a mathematical analysis. Suppose the model's error at each step is at most $\epsilon$, that is, the distance between the predicted state and the true state satisfies $\|\hat{s}' - s'\| \leq \epsilon$. Suppose further that the true dynamics function is Lipschitz continuous in the state, with Lipschitz constant $L$.

You may ask: "What is a Lipschitz constant?" Put simply, the Lipschitz constant $L$ measures "how much the next state changes when the state changes a little". If $L=1$, the next state differs by as much as the state did. If $L>1$, errors are amplified. If $L<1$, errors shrink on their own. For physical environments such as MuJoCo, $L$ is usually close to 1 and sometimes slightly above 1.

Now, under the same action sequence, let's derive a recursion for the state error at each step, $e_t = \|s_t^{\text{predicted}} - s_t^{\text{true}}\|$:

$$
e_{t+1} \leq L e_t + \epsilon
$$

Let's check that the recursion makes sense. At step $t$ we already carry an error $e_t$. Passing through the true dynamics, this error is amplified by at most a factor $L$, becoming $L e_t$. Then our own model adds at most $\epsilon$ of fresh one-step prediction error. The upper bound on the total error at step $t+1$ is the sum of the two.

How large does the error become after $T$ steps? Unrolling the recursion gives:

$$
e_T \leq \epsilon \sum_{i=0}^{T-1} L^i
$$

Let's compute with concrete numbers to build intuition:

- **Case 1: $L = 0.9$ (errors shrink on their own).** This is a geometric series, and for large $T$ the sum converges to $\epsilon/(1-0.9) = 10\epsilon$. The error does not grow without bound and is eventually held within $10\epsilon$. This is the ideal case.
- **Case 2: $L = 1$ (errors neither grow nor shrink).** Each step adds one $\epsilon$, so after $T$ steps the error is $T\epsilon$, which is **linear growth**. After a rollout of 100 steps the error is 100 times the one-step error.
- **Case 3: $L = 1.1$ (errors are amplified).** The error grows geometrically. At $T=50$ the sum is $\epsilon \times (1.1^{50} - 1)/(1.1 - 1) \approx 1164\epsilon$. If the one-step error is 0.01, then after 50 steps the error exceeds 11. For robot control, where the state dimension is typically in the tens, the prediction is already completely inaccurate.

This is why plain Dyna works well in simple discrete environments such as grid worlds but runs into trouble in complex continuous-control environments: **the longer the rollout, the more severely the model error accumulates, until the trajectory predicted by the model and the real trajectory have nothing to do with each other**. A policy trained on such fake data is likely to be led astray.

What can we do? The follow-up works (PETS and MBPO) address one and the same core problem: "how do we quantify and cope with model error?"

- PETS's idea: **I know the model is uncertain, so I will model that uncertainty explicitly.** A probabilistic ensemble (several models predicting together) expresses "how confident I am in this prediction". Planning takes this uncertainty into account, so the policy stays robust to the possibility that the model is wrong.
- MBPO's idea: **if long rollouts make the error explode, I will not run long rollouts.** It makes only very short model predictions (for example, 5 steps) and immediately resets to a real state, so the error has no chance to accumulate much.

Let's look at PETS first.

## Probabilistic Ensembles with Trajectory Sampling

Probabilistic Ensembles with Trajectory Sampling (Chua et al. 2018), abbreviated PETS, starts from a key observation: model uncertainty comes in two kinds, with different sources and different remedies.

You may ask: "Isn't a model simply inaccurate? Does uncertainty have kinds?" Yes, it has two, and the distinction matters a great deal:

- **Epistemic uncertainty**: this uncertainty **comes from insufficient data**. Because you have not seen enough situations similar to $(s,a)$, the model itself does not know what will happen. Think of a novice driver on snowy roads for the first time: he is unsure what will happen, not because snow is inherently random, but because he has never experienced it. This kind of uncertainty **can be eliminated with more data**.
- **Aleatoric uncertainty**: this uncertainty **comes from the randomness of the environment itself**. Even with unlimited data you cannot predict perfectly. Rolling a die is the standard example: however many times you observe it, the next outcome is still random. This kind of uncertainty **cannot be eliminated by more data**, and all you can do is model it.

PETS handles the two kinds of uncertainty with two different mechanisms:

- An **ensemble** $M_1, \ldots, M_K$, that is, $K$ independently trained models, expresses **epistemic uncertainty**. If the predictions of the $K$ models agree, the model is confident in this region. If they differ widely, the models have not seen this region and are uncertain.
- A **probabilistic output** $p(s' \mid s, a)$ expresses **aleatoric uncertainty**. Instead of predicting a single deterministic $s'$, each model predicts the mean and variance of a Gaussian distribution. Where the variance is large, the environment itself is random.

### Model Architecture

The PETS model is an ensemble of $K$ probabilistic neural networks, and $K=5$ is usually enough. Here is the code structure:

```python
class PEModel:
    def __init__(self, n_models=5):
        self.models = [ProbabilisticNN() for _ in range(n_models)]

    def predict(self, s, a):
        # Each model outputs (mean, var)
        means, vars = [], []
        for m in self.models:
            mu, sigma = m(s, a)
            means.append(mu); vars.append(sigma)
        return means, vars  # Ensemble disagreement = epistemic uncertainty
```

Each ProbabilisticNN does not output the next state directly. It outputs the mean $\mu$ and standard deviation $\sigma$ of a Gaussian distribution, that is, it assumes $s' \sim \mathcal{N}(\mu(s,a), \sigma^2(s,a))$.

During training, each model is initialized independently and trained independently, all on the same real interaction data, but with different mini-batch orderings and different initializations so that the models end up slightly different.

### Trajectory Sampling Strategy

Given a probabilistic ensemble model, how do we plan with it? PETS uses the idea of **MPC (model predictive control)**: at every step it replans the actions for the next H steps but **executes only the first one**, then moves to the next state and replans. This "rolling optimization" lets the controller keep correcting its plan from the latest real state, so it does not drift far under model error.

The planning algorithm is **CEM (Cross-Entropy Method)**, a simple but powerful sampling-based optimizer that is particularly well suited to planning in continuous action spaces. Here is the code:

```python
def cem_planning(model, s, horizon=10, n_samples=500, n_iters=5):
    # Initialize the action distribution
    action_mean = zeros(horizon, action_dim)
    action_var = ones(horizon, action_dim)

    for it in range(n_iters):
        # 1. Sample N action sequences
        action_seqs = sample_normal(action_mean, action_var, n_samples)

        # 2. Roll out with the model; each sequence uses one randomly chosen ensemble member
        rewards = []
        for seq in action_seqs:
            model_id = random_int(0, K)
            s_pred = s
            total_r = 0
            for a in seq:
                s_pred, r = model[model_id].predict(s_pred, a)
                total_r += r
            rewards.append(total_r)

        # 3. Pick the top 20% of sequences and update the distribution
        elite = top_k_indices(rewards, k=0.2 * n_samples)
        action_mean = action_seqs[elite].mean(0)
        action_var = action_seqs[elite].var(0)

    return action_mean[0]  # Execute only the first action (the MPC idea)
```

There is a key detail here. In step 2, **each trajectory uses the same randomly chosen ensemble member from start to finish**, rather than switching models at every step. This is called TS1 (Trajectory Sampling with 1 model). Why do it this way? If the model were switched at random at every step, you would effectively be planning on the dynamics of an "averaged model". But the real world is one world, not the average of five models. Rolling out a whole trajectory with a single fixed model better captures the multimodal futures and makes planning more robust to model uncertainty.

The way CEM works is also intuitive:

- At the beginning we know nothing about what a good action sequence looks like, so we sample from a very wide Gaussian (variance 1).
- We sample 500 candidate sequences and use the model to roll each out, to see which sequences have high expected total reward.
- We select the top 20% by reward (the elite sequences) and use their mean and variance as the sampling distribution for the next iteration.
- After repeating this 5 times, the distribution gradually converges to the region of action sequences that "look the best".

This is much simpler than gradient-based optimization, and it is less likely to get stuck in a local optimum, because every iteration explores with a large number of samples.

CEM has an instructive degenerate case worth knowing by name: **random shooting**. Sample $N$ action sequences once from a fixed distribution (often uniform over the action bounds), roll each out in the model, and execute the first action of the best sequence, with no refitting iterations at all:

$$
a_{0:H-1}^{(i)} \sim \text{Uniform}(\mathcal{A}^H), \quad i = 1, \ldots, N, \qquad \text{execute first action of } \arg\max_i \hat{R}(\tau^{(i)}).
$$

Random shooting is CEM with a single iteration and no distribution update. It is a weak planner, but precisely for that reason it is the honest **simple-controller baseline** every model-based paper should beat: if a learned model plus random shooting cannot outperform model-free training or even random actions on the task, the model itself is not pulling its weight, and there is no point blaming the planner yet.

### Experimental Results for PETS

On MuJoCo, PETS was the first model-based algorithm to match the performance of model-free algorithms such as SAC, while using **10-50 times fewer** samples. This is a milestone result: it showed for the first time that model-based methods are not "toys" on continuous-control benchmarks and can truly be used.

But, as you can probably guess, PETS has a price: **planning is very expensive**. To choose one action it runs 5 CEM iterations, each sampling 500 sequences, each rolled out for 10 steps. That is 5 × 500 × 10 = 25,000 model predictions just to pick one action. On a real robot, where each control step may allow only 20 ms, this planning overhead is a heavy burden at deployment time.

Is there a way to enjoy the sample efficiency of model-based methods without doing expensive planning at every step, as PETS does? That is the problem MBPO sets out to solve.

## Model-Based Policy Optimization

Model-Based Policy Optimization (Janner et al. 2019), abbreviated MBPO, rests on an innovation that can really be said in one sentence: **since long rollouts make error accumulate and explode, run only very short model rollouts**.

Wait, doesn't that sound too simple? Dyna can also do short rollouts, so what is MBPO's contribution? The point is that MBPO found a very counterintuitive result: **the rollout length does not need to be long. One to five steps are enough to capture almost all of the sample-efficiency gain that model-based methods bring, while keeping the effect of model error within an acceptable range.**

### Short-Horizon Rollouts

Let's return to the error-accumulation formula from before. Suppose $L=1.01$ (very close to 1, slightly amplifying) and the one-step error is $\epsilon=0.01$:

- Rollout of 1 step: error $\epsilon = 0.01$
- Rollout of 5 steps: error about $5.1\epsilon = 0.051$ (still small)
- Rollout of 20 steps: error about $22\epsilon = 0.22$ (starting to grow)
- Rollout of 100 steps: error about $170\epsilon = 1.7$ (about the scale of the state itself, so the prediction is no longer accurate)

Do you see it? In the first few steps the error actually grows slowly, and only after the rollout reaches a certain length does it start to explode exponentially. MBPO's strategy is to **use only the "safe zone" of the first few steps, where the error is tiny, and "hit the brakes" before the error has time to accumulate: reset to a real state and then start the next model rollout**.

Let's read the code to understand the flow:

```python
# Short-horizon rollouts keep model error under control.
# Each rollout starts from a real visited state and runs only K_short steps,
# so at most K_short steps of error can accumulate before we reset.
for _ in range(n_model_rollouts):
    s_sim = replay_buffer.sample_state()  # a real state the agent visited
    for _ in range(K_short):  # K_short = 5
        a = policy(s_sim)
        s_sim_next, r = model.predict(s_sim, a)
        replay_buffer.add(s_sim, a, r, s_sim_next)
        s_sim = s_sim_next
    # Rollout ends here: the next rollout restarts from another real state.
```

Note the logic here. We do not roll out from the initial state all the way to the end. We start from **some state in the real data** (a state sampled from the replay buffer, one the agent has actually visited), roll out only 5 steps, stop, and then switch to another real state. Every model trajectory therefore starts from an "accurate" point (taken from the real environment) and accumulates at most 5 steps of error, so the error is held strictly within a small range.

### MBPO Training Process

Putting these pieces together, the MBPO training process is a three-step loop, which we can see clearly in a diagram:

```
┌────────────────────────────────────────────────────┐
│ 1. Train model M with real data                    │
│    M.predict(s, a) → s', r                         │
├────────────────────────────────────────────────────┤
│ 2. Use M to generate short rollouts (5 steps)      │
│    Start: some state s from the real data          │
│    Each step: a = policy(s), s' = M(s, a)          │
│    Result: (s, a, r, s') × 5 added to replay       │
├────────────────────────────────────────────────────┤
│ 3. SAC updates on the replay buffer (real + fake)  │
└────────────────────────────────────────────────────┘
```

This process is quite clever. Notice that it does not do MPC planning as PETS does. Like Dyna, it uses the model to generate data and then trains with a standard model-free algorithm (SAC) on the replay buffer. The differences are:

1. The data generated by the model comes from 5-step short rollouts rather than a single step;
2. Every rollout starts from real data;
3. The underlying policy optimization uses a stable off-policy algorithm such as SAC.

Better still, the model rollout length $k$ in MBPO is not fixed. It can be **adjusted dynamically as training proceeds**. At the start the model is very inaccurate, so $k$ is set to 1 (only one-step predictions). As the model becomes more accurate, $k$ can slowly increase to 5 or even more. This is like a person learning something: at first you have no grip on how the world works and can only reason one step ahead, and as understanding deepens you can reason several steps ahead.

The experimental results of MBPO are striking: on MuJoCo it matches or even exceeds the performance of model-free SAC, with **10-100 times fewer samples**. And because what it ultimately trains is a SAC policy, deployment does not need the expensive CEM planning at every step that PETS requires. You simply run the policy network, at the same deployment cost as a model-free method.

### Pessimism, Decision-Relevant Error, and Distribution Shift

Short rollouts control how far error can compound, but three related problems remain, and they motivate most modern variants of these algorithms.

**Model exploitation.** A planner or policy trained inside the model is an optimizer, and optimizers find loopholes. If the model wrongly predicts a huge reward in some region it has never seen, the policy will steer exactly there: it "hacks" the model rather than solving the task. This is the model-based analogue of overestimation in DDPG, except the adversary is the planner instead of the actor. Short rollouts limit the damage but do not remove the incentive.

**Pessimistic planning.** One direct remedy is to charge the planner for uncertainty. Train an ensemble as in PETS, measure disagreement $u(s, a)$ (for example the standard deviation of the members' predicted next states), and penalize the imagined reward:

$$
\tilde{r}(s, a) = \hat{r}(s, a) - \lambda \, u(s, a).
$$

Planning against $\tilde{r}$ steers away from regions where the model is guessing, an idea used explicitly by algorithms such as MOPO. PETS achieves a similar effect implicitly: trajectories that pass through high-uncertainty regions vary wildly across ensemble members, so few of them survive elite selection.

**Decision-relevant error.** Average one-step prediction error is the wrong scoreboard. A model can have tiny pixel-level error yet miss the one feature the decision depends on (the speed of an oncoming car), or large pixel error that is irrelevant (the exact texture of clouds). MuZero's implicit model takes this seriously: it never predicts observations at all and trains the hidden dynamics only to predict rewards, values, and policies, that is, exactly the quantities the planner consumes. When you evaluate a dynamics model, report multi-step reward and value-prediction error on the planner's own state distribution, not just one-step state MSE on held-out data.

**Distribution shift and data aggregation.** Every one of these methods has a feedback loop: the policy changes, so the states it visits change, so yesterday's training data no longer covers today's queries, so the model degrades exactly where the new policy needs it. The standard fix is **data aggregation**: keep interleaving real collection with the current policy and retrain (or fine-tune) the model on the growing dataset, as the MBPO loop already does. A model trained once on random-exploration data and then frozen will almost always be exploited, because the optimized policy drives the system into regions the frozen model never saw.

### Comparing the Three Model-Based RL Algorithms

Let's put Dyna, PETS, and MBPO, three landmark algorithms, side by side to help you choose in practice:

| Algorithm | Model type             | Planning method  | Sample efficiency | Compute cost |
| --------- | ---------------------- | ---------------- | ----------------- | ------------ |
| Dyna      | Deterministic          | 1-step fake data | ~10×              | Low          |
| PETS      | Probabilistic ensemble | CEM MPC          | ~50×              | High         |
| MBPO      | Deterministic          | Short rollouts   | ~100×             | Medium       |

Practical recommendations:

- **Quick experiments and testing ideas**: choose Dyna. It is simple to implement, behaves stably, and needs no probabilistic models or planning, which makes it a good baseline.
- **High-precision control and robot manipulation**: choose PETS. The probabilistic ensemble handles model uncertainty well and closed-loop MPC control is very robust, which suits safety-critical, high-precision settings, but you must accept the high planning cost.
- **General continuous-control benchmarks**: choose MBPO. It has the highest sample efficiency, deploys as fast as a model-free method, works across the whole MuJoCo suite, and is currently a strong baseline for model-based continuous control.

### How to Evaluate a Model-Based Method Fairly

Model-based papers invite a specific kind of self-deception: the model does the learning, the planner does the work, and the policy takes the credit. Three controls keep the accounting honest.

1. **Match on real interactions, and report imagined samples and compute separately.** The fair x-axis is the number of real environment steps. Imagined rollouts are nearly free but not free: log how many model predictions each method used per real step (recall PETS spends about 25,000 per action) alongside wall-clock time, so a "10 times fewer samples" claim cannot hide "100 times more compute".
2. **Beat the model-free baseline and the no-model baselines.** Compare against SAC or TD3 trained on the same number of real steps, against random shooting inside your own learned model, and, when a simulator is available, against MPC with the ground-truth dynamics (the known-dynamics oracle). If the oracle fails, the planner is the bottleneck; if the oracle works but the learned model does not, the model is.
3. **Attribute failures before tuning.** When performance drops, test the four suspects separately: planner (does CEM with more samples fix it?), model (does one-step and multi-step prediction error spike on the current policy's states?), reward (does the predicted reward match the true reward along imagined trajectories?), and policy (does the policy underperform the planner's own action choice?). Most "model failures" turn out to be one identifiable link, and the fix depends on which one it is.

::: details Bonus: Why Not Use Long Rollouts? The Theory Behind MBPO
Janner et al. give, in their paper, a theoretical bound on how model bias affects policy performance. The rough conclusion is this. If the model's prediction error, measured in total variation (TV) distance, is $\epsilon_m$, then the performance gap between the policy $\hat{\pi}$ learned in the model and the optimal policy $\pi^*$ in the real environment satisfies:

$$
|J(\hat{\pi}) - J(\pi^*)| \leq \mathcal{O}\left(\frac{\epsilon_m}{(1 - \gamma)^2}\right)
$$

Note the $(1 - \gamma)^2$ in the denominator. When $\gamma$ is close to 1 (we usually use 0.99), the denominator is very small, which means that even a small model error $\epsilon_m$ can be amplified many times in the performance gap.

But if we use only short rollouts, we effectively shorten the horizon, which is equivalent to using a smaller effective $\gamma$. The denominator is then not so small, and the influence of model error is kept under control. This is the theoretical basis for MBPO's short rollouts.
:::

## Section Summary

Model-based RL greatly improves sample efficiency by **learning an environment model**. This section covered the evolution through three generations of algorithms:

1. **Dyna** treats the model as an "extra data generator". After every real interaction it generates N fake samples with the model for extra updates, improving sample efficiency by about a factor of N. In continuous control, however, it runs into error accumulation, and long rollouts can make predictions completely useless.
2. **PETS** confronts model uncertainty. A probabilistic ensemble models epistemic uncertainty (ensemble disagreement) and aleatoric uncertainty (variance of the probabilistic output) separately, and CEM sampling drives MPC planning. It was the first to bring model-based RL to model-free-level performance on MuJoCo, at the price of expensive planning at every step.
3. **MBPO** sidesteps error accumulation with short-horizon rollouts (usually 5 steps). Every model prediction starts from a real state and is reset after a few steps, before the error can explode. It still trains the policy with SAC, so deployment adds no extra computation, and its sample efficiency reaches 10-100 times that of model-free methods.

We call the line Dyna → PETS → MBPO the "data augmentation route": the role of the model is to generate more data, and the final policy is still trained in the model-free way. But model-based RL has another, entirely different flagship route: **explicit search + neural-network value estimation**. This route runs from AlphaGo through AlphaZero and MuZero to Dreamer V3. It represents the theoretical ceiling of model-based RL and also directly inspired process reward model search in the LLM era. That is the topic of the next section.

In the next section, [9.4 Search and World Models](./search-world-models), we will see what performance ceiling can be reached when you not only have a model but can also perform lookahead search inside it.
