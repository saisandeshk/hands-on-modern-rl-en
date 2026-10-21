# 12.3 Hierarchical Reinforcement Learning and World Models

## Reading Guide

**Key Ideas**

- Understand why long-horizon tasks are hard: why a single-level policy can barely be trained on a task of thousands of steps. The reward signal travels too far and gets "diluted".
- Master the core idea of hierarchical RL: a high level chooses subgoals or options, a low level executes primitive actions, and a long horizon is cut into short horizons.
- Compare three classic hierarchical methods: the formal definition of the Options framework, the Manager-Worker architecture of FeUdal Networks, and HIRO's off-policy learning of continuous subgoals.
- See how generative world models (Dreamer, Genie) turn the environment itself into an object of learning: learning a game engine from internet video and training policies in imagination.
- Think about how exploration, multi-agent, and hierarchical methods recombine inside the generated environments that world models provide.

In the previous section, [12.2](./marl), we solved the non-stationarity problem that arises when several agents learn at the same time, using the CTDE paradigm to let multiple agents train cooperatively. In this section we return to a single agent but stretch the time scale of the task. Imagine a household robot that has to "clean the whole house". It must first fetch the vacuum cleaner from the living room, then vacuum the bedroom, then take out the trash, and finally mop the floor. This may take thousands of consecutive primitive actions.

You may ask: haven't we already learned PPO, SAC, and RND? Can't we just train with those?

Let's stop and think about the discount factor. Suppose the discount at each step is $\gamma = 0.99$ (a common value). How much is a reward 100 steps away worth today? $0.99^{100} \approx 0.37$, so 37% remains. And 1000 steps away? $0.99^{1000} \approx 0.00004$, only four parts in a hundred thousand! The final reward signal is almost completely "diluted", and the earlier steps get no useful learning signal at all. This is why single-level policies are hard to train on long-horizon tasks.

What can we do? Hierarchical reinforcement learning offers a natural answer, modeled on how people work: do not think "what do I ultimately want to do?" at every step. Think of a "small goal" first, concentrate on completing that goal, and then think of the next one.

## 1. Shortening the Decision Span of Long Tasks with Hierarchy

Let's use Montezuma's Revenge again. The agent must first pick up a key, then open a door, and finally enter the next level. This is a long-horizon task. If we train it directly with PPO, the final "level cleared" reward is hard to assign accurately to an early action such as "reach out and take the key", which is several hundred steps earlier.

**Hierarchical reinforcement learning** (HRL) decomposes decision-making into two or more levels:

- **High-level policy**: it does not decide at every step. Only once every $c$ steps it outputs a decision, either choosing a "subgoal" or choosing an "option" (a temporally extended action).
- **Low-level policy**: guided by the subgoal given by the high level, it outputs a primitive action at every step until the subgoal is completed or time runs out.

What is the benefit? The high level does not need to care about details such as "how to walk to the key". It only has to decide "go and get the key now" and hand this goal to the low level, which figures out how to walk there. The high level only makes decisions at the sparse "subgoal completed" nodes, which cuts a horizon of hundreds or thousands of steps into many short horizons of only $c$ steps. Gradient signals can propagate effectively within each sub-level: the low level only cares about "reaching the subgoal within these $c$ steps", and the high level only cares about "which subgoal maximizes the team's return".

It is like writing code. You do not write assembly instructions directly. You first think "I need to write a sorting function", and only inside that function do you consider which loops and which comparisons to use. The high level is the function call, and the low level is the concrete implementation.

## 2. Comparing Three Hierarchical Reinforcement Learning Methods

The idea of hierarchy is intuitive, but how is it implemented? We look at three classic works: from the earliest formal framework, Options, to the end-to-end learnable FeUdal Networks, and then to the modern off-policy method HIRO. Along the way we fix one shared piece of vocabulary — goal-conditioned policies and hindsight relabeling — right before HIRO, which is the method that needs it.

### 2.1 Options: Encapsulating a Policy Segment as a Temporally Extended Action

The **options** framework, proposed by Sutton, Precup & Singh in 1999, is the formal foundation of hierarchical RL. You can think of an option as a "macro action that can run for many consecutive steps". For example, "go to the door" is an option. It is not a single-step action but a whole segment of policy that terminates only when you reach the door.

Formally, an option $\omega$ consists of three parts:

$$\omega = (\mathcal{I}_\omega, \pi_\omega, \beta_\omega)$$

Let's look at them one by one:

- **Initiation set** $\mathcal{I}_\omega$: in which states can this option be started? For the "go to the door" option, you must be inside the house to start it. If you are already outside, you do not need it.
- **Intra-option policy** $\pi_\omega$: the policy followed while the option runs, that is, the concrete policy for "how to walk to the door".
- **Termination function** $\beta_\omega(s)$: after reaching state $s$, with what probability does the option terminate? At the door the termination probability is close to 1, and before reaching it the probability is close to 0.

You may ask: since an option may run for $T$ consecutive steps, how should the Q-function be written when the high level learns to choose among options? Ordinary Q-learning looks at only one step at a time, but now one step corresponds to $T$ steps inside an option.

Exactly. So the option's Q-function has to add up the rewards of all $T$ steps and then attach the value of the termination state:

$$Q^\mu(s, \omega) = \mathbb{E}\left[\sum_{t=0}^{T-1}\gamma^t r_t + \gamma^T \max_{\omega'} Q^\mu(s_T, \omega')\right]$$

Let's take this expression apart:

- $s$: the state when the option is started
- $\omega$: the option currently chosen
- $T$: the number of steps the option actually runs (possibly random, depending on the termination function)
- $s_T$: the state when the option terminates
- The first term, the sum $\sum_{t=0}^{T-1} \gamma^t r_t$: the discounted sum of all rewards received while the option runs
- The second term, $\gamma^T \max_{\omega'} Q^\mu(s_T, \omega')$: after the option ends, the future value obtainable by continuing to choose the best option from the termination state

The high level chooses options within a framework called an **SMDP** (Semi-Markov Decision Process). "Semi" means that the duration of each decision is no longer a fixed 1 but may last $T$ steps. The low level executes the intra-option policies.

The Options framework is elegant, but it has a practical problem: options must be defined in advance. You have to tell the system by hand which options exist, when to start them, and how to terminate them. This is hard to do in complex environments. Can the system learn the options by itself?

### 2.2 FeUdal Networks: The Manager Sets a Direction and the Worker Acts

FeUdal Networks (Vezhnevets et al. 2017) take this idea one step further: both the "high level" and the "low level" are end-to-end learnable neural networks, and no subgoals have to be defined in advance.

The architecture resembles a feudal hierarchy (hence the name FeUdal):

- **Manager (the lord, the high level)** $M_\theta$: it sits higher up and does not need to manage how each step is taken. Every $c$ steps (for example $c=16$) it outputs a **direction vector in a latent space**, $g_t \in \mathbb{R}^k$. Note that it does not output a subgoal directly but a "direction", such as "move in that direction in latent space".
- **Worker (the laborer, the low level)** $W_\phi$: it does the actual work. Within a window of $c$ steps, at every step it outputs a primitive action $a_t \sim \pi_\phi(a \mid s; g_t)$. Its policy is modulated by the direction $g_t$ from the Manager, that is, "walk in the direction the lord points".

You may ask: the Manager outputs a direction in latent space, so how do we know whether the Worker actually moved in that direction? It is a latent space, and we cannot see it.

FeUdal uses a clever auxiliary loss to train the Manager. A shared encoder maps each state $s_t$ to a latent representation $\hat{z}_t$, and we then check whether the change of the latent representation over the next $c$ steps points in the same direction as the Manager's direction:

$$\mathcal{L}_M = -\langle g_t,\ \hat{z}_{t+c} - \hat{z}_t\rangle$$

This expression looks a bit abstract, so let's explain it:

- $\hat{z}_t$: the latent state representation at step $t$ (when the Manager makes its decision)
- $\hat{z}_{t+c}$: the latent state representation after $c$ steps (at the Manager's next decision)
- $\hat{z}_{t+c} - \hat{z}_t$: the direction vector in which the latent state actually moved during these $c$ steps
- $\langle \cdot, \cdot \rangle$: the inner product of two vectors. If the two directions are roughly the same, the inner product is positive and larger the more aligned they are. If the directions are opposite, it is negative.
- The minus sign in front: we want the inner product to be as large as possible (the actual direction agrees with the Manager's direction), so minimizing the negative inner product is the same as maximizing the inner product.

Let's compute with two-dimensional numbers. Let the Manager's direction be $g_t = (1, 0)$ ("pointing right" in latent space). If over the next $c$ steps the latent representation actually moves by $\hat{z}_{t+c} - \hat{z}_t = (0.9, 0.2)$, the inner product is $1 \times 0.9 + 0 \times 0.2 = 0.9$ and the loss is $-0.9$. The directions basically agree, so the loss is small. If it actually moves by $(-0.8, 0)$, the inner product is $-0.8$ and the loss is $0.8$. The Worker went the opposite way, and this direction is strongly penalized.

What does this auxiliary objective do? It trains the Manager to give directions that "the Worker can actually carry out". The Manager cannot point in arbitrary directions, because a direction the Worker cannot follow is useless. The external reward of the task still decides "which directions are worth choosing", but this auxiliary loss guarantees that the "language" between the levels is shared.

FeUdal showed the potential of hierarchical structure on long-horizon tasks such as Montezuma's Revenge, but it also exposed a problem: jointly training the Manager and Worker end to end is sensitive to hyperparameters and often unstable. The Manager does not know how to point, and the Worker cannot understand what the Manager is saying.

### 2.3 Goal-Conditioned Policies and Hindsight Relabeling

Every hierarchical method below commands its low level with some notion of "goal", so let's fix what that means. A **goal-conditioned policy** $\pi(a \mid s, g)$ takes, alongside the state, a goal $g$ describing the outcome to achieve — a target position, a desired object configuration, a latent direction. Its value function, a universal value function approximator (UVFA), is likewise goal-indexed: $V^\pi(s, g)$ is the expected return of pursuing goal $g$ from state $s$. One network thus represents many skills at once: change $g$ and the same weights reach for a different target.

Goals pair naturally with sparse rewards. A common choice is the binary goal-achievement reward

$$r_g(s) = \begin{cases} 0 & \text{if goal } g \text{ is achieved in } s, \\ -1 & \text{otherwise}, \end{cases}$$

so the agent gets $0$ only on success and $-1$ everywhere else. This is honest — no hand-shaped shaping — but brutal: a failed episode is a flat string of $-1$s with nothing to learn from.

**Hindsight Experience Replay** (HER, Andrychowicz et al. 2017) rescues the failure. Its observation: an episode that missed goal $g$ still _achieved_ something — it ended in some state $s_T$. So relabel the episode in hindsight: pretend the goal had been $g' = m(s_T)$ (the achieved outcome) all along, recompute the rewards under $r_{g'}$, and store both the original and the relabeled transition. Under $r_{g'}$ the final transition earns $0$ instead of $-1$, so every episode, success or failure, contributes a positive example of "how to reach the state you actually reached".

Walk through a robot arm pushing a box. Commanded goal: push the box to $(1, 0)$. The arm pushes it to $(0.3, 0.5)$ instead — reward $-1$ at every step, apparently useless. HER relabels: "suppose the goal had been $(0.3, 0.5)$". Under that goal the same trajectory earns $0$ at the end, teaching the goal-conditioned policy a real skill (reaching $(0.3, 0.5)$) from a failed rollout. Across many episodes the agent accumulates reaching skills for wherever it actually ends up, and the commanded goals gradually become reachable. HER needs off-policy learning (relabeled transitions were not generated by the current goal-conditioned policy), which is why it is standardly paired with DDPG-style or SAC-style updates.

The lesson for hierarchy: a low level trained this way already "understands" goal commands before the high level is any good at issuing them — which is exactly the property HIRO's off-policy machinery exploits next.

### 2.4 HIRO: Learning Continuous Subgoals from Off-Policy Data

Data-Efficient Hierarchical Reinforcement Learning (HIRO, Nachum et al. 2018) is a modernized improvement on FeUdal that solves FeUdal's stability problem. It has two key innovations: **off-policy training** and **goal transition**.

HIRO's design is more direct:

- Every $c$ steps the high level outputs a **continuous subgoal** $g_t \in \mathbb{R}^d$. This time it is no longer a direction in latent space but a **displacement** directly in state space: "I want the state, $c$ steps from now, to be displaced by $g_t$ relative to the present."
- The low level's **intrinsic reward** is defined directly as "how far from the target point": $r^l_t = -\|s_{t+1} - (s_t + g_t)\|$. The target point is the position you should reach after displacing by $g_t$ from the current position. The closer you are to it, the higher the reward, and if you drift off the reward is low.
- The high level is trained with an off-policy algorithm (such as TD3), which can use old data in the replay buffer, so the sample efficiency is higher.

Let's walk through it with coordinates. The state is the robot's position on the ground, and at this moment the high level gives the subgoal $g_t = (10, 0)$, "move 10 meters east". The current position is $(2, 0)$, so the target point is $(2,0) + (10,0) = (12, 0)$. After one step the low level is at $(3, 0)$, still 9 meters from the target point, with intrinsic reward $-9$. Walking on to $(11, 0)$ leaves only 1 meter, with reward $-1$. The low level does not need to understand an external task like "clean the house". Its only job is to get closer to the target point at every step, while the external reward is all credited to the high level.

The code skeleton of HIRO looks quite clear:

```python
# Skeleton of the HIRO main loop
for step in range(total_steps):
    if step % c == 0:
        # The high level samples a subgoal every c steps
        goal = high_level_policy(state)
    # Low-level conditioned policy: given the current state and goal, output an action
    a = low_level_policy(state, goal)
    s_next, r_ext, done = env.step(a)
    # Low-level intrinsic reward: how far the actual displacement is from the goal displacement
    r_int = -np.linalg.norm(s_next - (state + goal))
    low_buffer.add(state, a, r_int, s_next)
    if step % c == 0:
        # The high-level reward is the external reward accumulated over these c steps
        high_buffer.add(state, goal, ext_reward_sum, s_next_c)
    update(low_level_policy, low_buffer)
    update(high_level_policy, high_buffer, goal_transition=transition_fn)
```

Wait, you may have noticed a problem. The high level is trained off-policy and draws old data from the replay buffer. But the subgoal $g$ in the old data was the goal that the low-level policy of that time was trying to reach, and the low-level policy has been updated since! Even if the low level stands in the same state and sees the same subgoal $g$, its behavior now may differ from what it was then. This is the **off-policy bias** problem.

HIRO's most important technical innovation is **goal transition**, which solves this problem. When we take an old transition $(s_t, g_t, \sum r, s_{t+c})$ from the replay buffer, we do not use the old $g_t$ directly. We remap it to a "new subgoal $g'_t$" with the following property: if the **current low-level policy** starts from $s_t$ with goal $g'_t$, the endpoint it reaches is still roughly $s_{t+c}$.

::: details Bonus: How Goal Transition Works
The concrete procedure of goal transition is as follows. For a stored high-level transition, HIRO does not blindly reuse the old goal. Instead it samples a small set of **candidate goals** around the actually achieved displacement $s_{t+c} - s_t$ (typically including the originally commanded goal $g_t$) and selects the candidate $g'$ that maximizes the log-probability the **current** low-level policy assigns to the action sequence actually executed in the buffer:

$$g' \approx \arg\max_{g \in \text{candidates}} \sum_{k=0}^{c-1} \log \pi^{\text{low}}_{\phi'}(a_{t+k} \mid s_{t+k}, g),$$

where $\phi'$ denotes the current low-level parameters.

In practice the achieved displacement $s_{t+c} - s_t$ is one of the candidates and often wins the maximization, which is why summaries sometimes shorten the rule to "$g' = s_{t+c} - s_t$". But the maximizing selection is the load-bearing part: it is what guarantees the relabeled transition looks like something the current low-level policy would actually do, so old data stays compatible with the updated low level instead of going stale.
:::

### 2.5 Comparing Options, FeUdal, and HIRO

Let's put the three hierarchical methods side by side:

| Algorithm | High-level output    | Low-level objective | Training method       | Main problem               |
| --------- | -------------------- | ------------------- | --------------------- | -------------------------- |
| Options   | option id            | Fixed sub-policy    | SMDP-Q                | Options must be predefined |
| FeUdal    | Latent direction $g$ | Worker intrinsic    | On-policy, end-to-end | Unstable training          |
| HIRO      | State displacement   | State matching      | Off-policy            | Design of goal transition  |

### 2.6 Practical Difficulties of Hierarchical RL

After listing so many benefits, you may wonder: if hierarchy works so well, why has it not become the default choice for every RL task?

The hierarchical structure itself introduces extra assumptions and difficulties:

- **The level-partition problem**: how many levels are appropriate? What should $c$ be? How is the subgoal space defined? All of this has to be designed by hand or tuned.
- **The level-disconnect problem**: if the low level cannot execute the subgoals the high level outputs, or the low level simply learns to "ignore" the high level's instructions (it can earn the external reward directly anyway), the hierarchy is built for nothing. The Manager outputs meaningless directions and the Worker pretends not to see them.
- **The credit assignment problem**: when the task fails, was it a wrong decision at the high level or a wrong execution at the low level? This blame assignment problem is even more severe in a hierarchical structure.

You may have noticed that LLM agents today are often written explicitly as a "plan first, then execute" process, for example having the model write a plan first (ReAct, Plan-and-Solve) and then carry it out step by step. Why not build the hierarchy inside the neural network and instead write it as an explicit interaction flow? One reason is that explicit plans and execution records are easier to inspect and debug. The high level's "subgoals" are plans written in natural language, which we can read and easily correct.

But whether the hierarchy sits inside the network or in the interaction flow, the core principle is the same: verify whether the subgoals really shorten the decision span of the original task and really make reward propagation easier.

## 3. Using a Generative World Model as the Training Environment

The hierarchical methods above all assume that the environment already exists, and all we have to do is learn a policy inside it. Can we go one step further and let a model learn the environment itself?

This is the idea of the **generative world model**. First train a model on real data so that it learns "in state s, taking action a, what is the next state", that is, it learns the transition dynamics of the environment. The policy can then be trained or planned directly inside the "imagined trajectories" generated by this model, without interacting with the real environment every time.

### 3.1 From Dreamer to Genie

In [Chapter 9, Dreamer V3](../chapter09_continuous_control/search-world-models#dreamer-v3-and-a-new-generation-of-world-models) we already met one successful world model. The Dreamer series first trains an RSSM (Recurrent State Space Model) world model on real interaction data, and then trains an actor-critic inside the "imagined" trajectories generated by the model. Dreamer showed that "training policies inside a world model" is feasible, and on many tasks it is even more sample-efficient than model-free methods.

But Dreamer's world model is still **task-specific**. It learns the environment of whichever Atari game you train on, and for a different game you must train a new world model.

Genie (Bruce et al. 2024) pushed world models to a new stage: **generative, cross-task** world models. Given an internet video or an image, it can learn an interactive "game engine": you input an action, and the model generates the next frame. What does this mean?

Several revolutionary changes follow:

1. **Environment data can come from internet video**, no longer depending on official game engines or physics simulation. YouTube holds countless videos of people playing games, and all of it can serve as training data.
2. **One model can generate many environments**, generalizing across tasks. You no longer need to train a separate model for every game.
3. **RL training can take place entirely in generated environments**, with no real physics engine. Imagine training an agent in a model-generated Super Mario and then putting it into the real game and having it work.

Genie (Bruce et al. 2024) already introduced **latent action** learning as part of this package: without any human-labeled action data, the model automatically discovers from video the "latent control variables that cause the next frame to change". Formally:

$$z_t = \text{LatentAction}(x_t, x_{t+1}),\quad x_{t+1} = \text{Decoder}(x_t, z_t)$$

Let's explain these two expressions:

- $x_t$: the video frame at step $t$
- $\text{LatentAction}$: a model that looks at two consecutive frames and automatically infers "what latent action $z_t$ caused the change from $x_t$ to $x_{t+1}$", even though we never gave it action labels
- $\text{Decoder}$: given the current frame $x_t$ and a latent action $z_t$, it predicts what the next frame $x_{t+1}$ will look like
- $z_t$: the "latent action" the model has learned, which represents the hidden control factor in the video that causes the picture to change

Let's walk through a recording of Super Mario. The model sees Mario standing in front of a pipe in frame 100 and Mario having jumped over the pipe in frame 101. The latent action $z_t$ that explains this change ends up corresponding roughly to "press jump". The clouds in the background also drift between the two frames, but no action can explain the change in the clouds' position, so the latent action model learns to ignore it. The decoder works the other way around: given "Mario standing in front of the pipe" plus "press jump", it generates the next frame in which Mario has jumped over the pipe.

If these latent actions $z_t$ can be controlled stably (they are not noise jumping around), we can treat $z_t$ as the agent's action space and let the decoder play the "environment". The agent outputs a latent action, the decoder returns the next frame, and the reward function is defined separately. In this way we can even learn a fully interactive environment from "unlabeled video"!

Of course, one important limitation remains unresolved: do the learned latent actions correspond to real controls that humans can understand and execute? For example, would the model also learn "camera shake" into the actions? This has to be verified in concrete environments.

## 4. How Exploration, Cooperation, and Hierarchy Use Generated Environments

Once the world model itself becomes a generative, interactive training environment, the three families of methods in this chapter (exploration, multi-agent, and hierarchy) can act on different layers of the environment and recombine:

1. **Exploration**: intrinsic rewards no longer need to act in pixel space and can act directly on the world model's **latent space**. Did you notice that ICM's forward-prediction error has a structure very close to the world model's own state-prediction loss? A world model is already "predicting the next state", so its prediction error is naturally an exploration signal.

2. **Multi-agent**: generative models such as Genie can generate complex environments that contain NPCs (non-player characters), and multiple agents can perform **self-play** in the generated environment, as described in [Chapter 26 on self-play](../chapter26_selfplay_frontiers/self-play-outlook/). Teammates and opponents are all inside the generated environment, and no human player data is needed.

3. **Hierarchy**: the high-level policy no longer has to output displacements in state space as subgoals. It can output **latent subgoals** directly and let the world model decode them into changes of the environment state. This amounts to an implicit version of the options framework, with subgoals defined in the world model's latent space.

Together, these directions advance the world model from "an auxiliary model that predicts the next state" to "infrastructure that provides interactive training trajectories". We still have to stay alert, of course: we need to check whether the generated environment obeys the rules of the real task, and whether the policy is "cheating" by exploiting loopholes of the generative model. For example, if the model generates a buggy picture in one region, the agent may get stuck there and farm points. The later Agent chapters will keep discussing how to use tool environments and executable feedback in place of purely textual or purely generated simulation.

## Chapter Summary

In this chapter we discussed three important extensions beyond the classic single-agent, dense-reward setting:

1. **Sparse rewards → intrinsic rewards**:
   - ICM uses the prediction error of a forward model as curiosity, but needs an inverse model to filter out irrelevant noise, and still suffers from the noisy-TV problem
   - RND uses random network distillation, which is simpler and more robust, does not depend on actions, and is naturally resistant to the noisy TV
   - NGU combines within-episode (episodic) novelty with cross-episode long-term novelty from RND
   - Agent57 adaptively switches the exploration-exploitation intensity and was the first algorithm to reach superhuman performance on all 57 Atari games

2. **Multi-agent non-stationarity → CTDE**:
   - The core difficulty is environment non-stationarity: other agents' learning changes the transition probabilities
   - CTDE is the most practical paradigm: during training the Critic acts as a coach with global information, and during execution the Actors decide in a decentralized manner
   - MADDPG gives each agent a centralized Critic and can handle mixed cooperative-competitive settings, but inherits DDPG's instability
   - MAPPO uses a shared Critic plus PPO's clipped updates. It is simple to implement, trains stably, and is a commonly used strong baseline for cooperative tasks such as SMAC and Hanabi

3. **Long horizon → hierarchy**:
   - The core idea is to split a long task into several levels: the high level chooses subgoals and the low level executes primitive actions, which shortens the decision span
   - The Options framework provides the formal foundation for temporally extended actions; goal-conditioned policies with hindsight relabeling (HER) turn failed episodes into training signal for the low level
   - FeUdal Networks propose an end-to-end Manager-Worker architecture but train unstably
   - HIRO uses off-policy training and the goal-transition trick to learn continuous subgoals
   - The practical difficulties of hierarchical methods are level design and the level-disconnect problem

These three lines also provide tools and inspiration for the later LLM reinforcement learning chapters:

- Tool calls can be seen as a kind of temporally extended action (similar to options)
- Multi-agent cooperation has to handle the balance between joint training and decentralized execution
- World models can generate interactive training environments, reducing dependence on real interaction

In the next chapter, [Chapter 13: The RLHF Training Pipeline](../chapter13_rlhf/base-model-to-assistant), we enter the main line of large-model alignment: how to turn human preferences into a reward signal and then use RL to train a base model into a useful assistant. The ideas of exploration, cooperation, and hierarchical decision-making from this chapter will reappear in the later Agent chapters, combined at a larger scale and in more interesting ways.

## Further Reading

- [Pathak et al. 2017 "Curiosity-driven Exploration by Self-Supervised Prediction" (ICM)](https://arxiv.org/abs/1705.05363)
- [Burda et al. 2018 "Exploration by Random Network Distillation" (RND)](https://arxiv.org/abs/1810.12894)
- [Badia et al. 2020 "Agent57: Outperforming the Atari Human Benchmark"](https://arxiv.org/abs/2003.13350)
- [Lowe et al. 2017 "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (MADDPG)](https://arxiv.org/abs/1706.02275)
- [Yu et al. 2022 "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (MAPPO)](https://arxiv.org/abs/2103.01955)
- [Andrychowicz et al. 2017 "Hindsight Experience Replay" (HER)](https://arxiv.org/abs/1707.01495)
- [Vezhnevets et al. 2017 "FeUdal Networks for Hierarchical Reinforcement Learning"](https://arxiv.org/abs/1703.01161)
- [Nachum et al. 2018 "Data-Efficient Hierarchical Reinforcement Learning" (HIRO)](https://arxiv.org/abs/1805.08296)
- [Bruce et al. 2024 "Genie: Generative Interactive Environments"](https://arxiv.org/abs/2402.15391)
