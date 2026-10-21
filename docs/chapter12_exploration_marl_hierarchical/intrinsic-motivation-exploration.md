# 12.1 Intrinsic Motivation and Exploration

## Reading Guide

**Key Ideas**

- Understand why random exploration almost cannot succeed in sparse-reward tasks. Take Montezuma's Revenge as the example: the probability that $\epsilon$-greedy stumbles on the first reward is only $10^{-18}$. How small is that?
- Master the central idea of intrinsic rewards: before any external reward arrives, use "curiosity" to give the agent a learning signal.
- Understand how ICM (the Intrinsic Curiosity Module) builds an intrinsic reward from prediction error, and why it needs an inverse model to filter out irrelevant noise.
- Understand why RND (Random Network Distillation) is simpler yet more robust: it needs no actions and no inverse model, and it also solves the "noisy TV" problem.
- See how NGU and Agent57 manage short-term (within an episode) and long-term (across episodes) exploration at the same time, becoming the first algorithm to surpass human performance on all 57 Atari games.

<OnlineTraining studios="minigrid" compact />

[Chapter 9: Continuous Control](../chapter09_continuous_control/ddpg) studies how a single agent can learn stably in a continuous action space. [Chapter 10: Offline RL](../chapter10_offline_rl/offline-data-distribution-shift) studies how to use a fixed dataset when online interaction is no longer possible. [Chapter 11: Imitation Learning and Meta-RL](../chapter11_imitation_meta_rl/bc-dagger) studies learning methods for the case where there is no explicit reward or where the agent must adapt quickly to new tasks. In this chapter we keep relaxing the classic MDP setting and prepare for three more realistic challenges: a reward may appear only once in a very long while (sparse-reward exploration), the environment may contain other agents that are learning at the same time (multi-agent), and a task may be so long that a single-level policy cannot obtain an effective training signal at all (hierarchical reinforcement learning).

Let's start with the first challenge, sparse rewards.

You may ask: haven't we already learned exploration methods such as $\epsilon$-greedy and UCB? Why do we need a whole section on exploration?

Not so fast. Let's stop and think about one concrete number.

## 1. Why Sparse Rewards Need Intrinsic Motivation

[The exploration-exploitation problem in Chapter 2](../chapter02_rl_basics/bandit) already introduced the exploration-exploitation trade-off in the stateless bandit setting. The expected return of each arm is unknown, and the agent must allocate its sampling budget between "pulling the arm that currently looks best" (exploitation) and "pulling an arm that is still uncertain" (exploration). The UCB rule introduced there uses an upper confidence bound:

$$U_t(a) = \hat{\mu}_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}}$$

Let's take this expression apart:

- $\hat{\mu}_t(a)$: the estimate of the average reward of arm $a$ up to step $t$. This is the "exploitation" part and tells us which arm looks best right now.
- $c\sqrt{\ln t / N_t(a)}$: the uncertainty bonus. The fewer times the arm has been pulled, $N_t(a)$, the larger this term, which encourages us to try arms that have hardly been pulled.
- $c$: the exploration coefficient, which controls how willing we are to try something for the sake of uncertainty.

In a bandit problem this formula works well. But when we enter the high-dimensional visual environments of deep RL, the problem is magnified thousands upon thousands of times.

### That Despairing Number: $10^{-18}$

In Atari games such as Montezuma's Revenge or Pitfall!, to obtain the first reward from the initial state the agent must execute a chain of **precise** actions: jump over a trap, climb a ladder, pick up a key, and then open a door. That takes dozens of completely correct steps.

Let's do the arithmetic. Suppose the agent uses the simplest $\epsilon$-greedy exploration, with probability $\epsilon = 0.1$ of choosing a random action at each step (already a fairly high exploration rate), with 6 available actions at every step, and obtaining the first reward requires at least 50 consecutive correct choices.

What is the probability that it stumbles on the first reward by wandering blindly?

About $10^{-18}$.

How small is that number? Let's make it concrete:

- The age of the universe is about $10^{17}$ seconds
- The number of grains of sand on Earth is about $10^{18}$
- In other words, an agent exploring once per second since the Big Bang might not yet have met a single reward today!

This is why DQN scored zero for a long time on these **hard-exploration** games. The algorithm was not bad. Against rewards this sparse, random exploration is almost impossible to succeed probabilistically.

Where does the difficulty come from? From the extreme sparsity of the reward signal. Let's formalize it: suppose the agent starts at the initial state $s_0$ and needs at least $H^\star$ steps to reach the first rewarding state $s^\star$. Any update that depends only on the environment reward $r_t$ must wait until the first successful trajectory appears before learning can start, and the density of successful trajectories falls exponentially as $H^\star$ grows.

This leads to the core idea of the **intrinsic reward**: before reaching the external reward, we manufacture some "curiosity signals" for the agent ourselves, encouraging it to visit novel or hard-to-predict states, just as a human infant actively explores its surroundings. The infant does not do this to get a reward right away, but purely because "I have never seen this thing and I want to take a look."

During training we add the environment reward and the exploration reward together:

$$r^{\text{total}}_t = r^{\text{ext}}_t + \beta \cdot r^{\text{int}}_t$$

Let's explain every symbol in this expression:

- $r_t^{\text{ext}}$: the **extrinsic reward** given by the environment at step $t$, that is, the score defined by the game, which may be 0 most of the time
- $r_t^{\text{int}}$: the **intrinsic reward** that the agent computes itself from the novelty of the state. This is the "curiosity" we construct artificially.
- $\beta$: the exploration coefficient, which decides how much weight the intrinsic reward signal carries
- $r^{\text{total}}_t$: the total reward finally used for training

A concrete numerical example: if the extrinsic reward is 0 (no points yet), the intrinsic reward is computed as 0.4, and $\beta=0.1$, then the total reward is $0 + 0.1 \times 0.4 = 0.04$. Although it is only 0.04, it is far better than all zeros, because the agent at least knows "that place a moment ago was somewhat interesting."

But an intrinsic reward cannot be given carelessly. It has to satisfy two key conditions:

1. **Computable**: it depends only on data that has already been observed and needs no additional external supervision. We cannot label "is this state novel?" by hand during training.
2. **Exhaustible**: after a state has been visited enough times, its intrinsic reward should decay to zero. Otherwise the agent will find some "novel wall", stand there forever "farming points", and never complete the real task again.

In the next two sections we look at the two mainstream routes for constructing intrinsic rewards: ICM, based on prediction error, and RND, based on random network distillation.

## 2. Driving Exploration with ICM's Prediction Error

### 2.1 Constructing an Intrinsic Reward from State Prediction Error

The central idea of the Intrinsic Curiosity Module (Pathak et al. 2017) is very intuitive: if I cannot predict "what will happen next after I take this action", then this region is "unfamiliar" and worth more exploration. The larger the prediction error, the less I know about this place, and the higher the intrinsic reward.

Let's stop and think: the idea is natural, but can we simply predict in pixel space? For instance, predict the RGB pixel values of the next frame directly?

The answer is no. The next frame has far too many pixel details: leaves are swaying, the water surface is rippling, there may be a small animation flickering in a corner of the screen... All of this is high-frequency noise that **the agent cannot control and that is irrelevant to the task**. If we compute the prediction error directly in pixel space, this irrelevant noise dominates the error, and the agent will be drawn to things that are "never predicted accurately but utterly meaningless".

ICM solves this problem cleverly. It first uses an **inverse model** $g_\phi$ to learn a feature space $\Phi(s)$. What does this inverse model do? Its input is two consecutive frames $(s_t, s_{t+1})$, and it predicts which action $a_t$ the agent executed in between.

You may ask: why predict the action? What does it have to do with exploration?

This is the crux: only the "changes that actions can influence" are kept in the feature $\Phi(s)$. How the background leaves sway and how the screen flickers have nothing to do with the agent's action, so the inverse model has no need to care about them, and they are naturally filtered out. The features that remain contain only the parts "the agent can control and influence".

It is clearer to treat this filtering mechanism as a puzzle. Give the inverse model two consecutive frames and ask it to guess "which key the player pressed". Mario moved half a tile to the right, which directly reveals "pressed right". At the same time the leaves changed their swaying angle and the score bar in the upper right flashed, but none of these changes reveals any key press. If the encoder stuffs leaves and the score bar into the features, it does not help the inverse model answer at all and only adds a burden. So in the trained $\Phi(s)$, what remains is information that actions can affect: Mario's position, the terrain, and the positions of enemies.

With clean features $\Phi(s_t)$ in hand, we then train a **forward model** $f_\psi$ that predicts the features of the next state from the current features and the executed action:

$$\hat{\Phi}(s_{t+1}) = f_\psi(\Phi(s_t), a_t)$$

Symbol explanation:

- $\Phi(s_t)$: the features extracted by the encoder from the current state $s_t$
- $a_t$: the action executed in state $s_t$
- $f_\psi$: the forward dynamics model, with parameters $\psi$
- $\hat{\Phi}(s_{t+1})$: the next-state features predicted by the forward model

Then the distance between the true features and the predicted features is the intrinsic reward we want:

$$r^{\text{int}}_t = \frac{1}{2}\|\Phi(s_{t+1}) - \hat{\Phi}(s_{t+1})\|^2$$

This is the squared prediction error. The $1/2$ in front is there only to cancel the 2 when differentiating, and does not change the relative size of the reward.

Why does this mechanism match our intuition?

- When the agent first enters an unfamiliar region, the forward model has not seen these states, so its predictions are inaccurate. This distance is large and the intrinsic reward high, which encourages the agent to keep exploring.
- After similar states have appeared repeatedly, the forward model gradually learns to predict them, the error shrinks, and the reward falls. The agent then goes elsewhere to find novelty.

During training we add the policy loss, the inverse-model loss, and the forward-model loss together:

$$\mathcal{L} = \mathcal{L}_{\text{policy}}(\theta) + \lambda_{\text{inv}}\,\mathcal{L}_{\text{inv}}(\phi) + \lambda_{\text{fwd}}\,\mathcal{L}_{\text{fwd}}(\psi)$$

- $\lambda_{\text{inv}}$ and $\lambda_{\text{fwd}}$: control the weights of the two auxiliary tasks
- The inverse model helps the feature extractor $\Phi$ keep only action-relevant information
- The forward model provides the novelty signal
- The policy is updated with the total reward that includes the intrinsic reward

```python
class ICM(nn.Module):
    def __init__(self, feat_dim=256, action_dim=6):
        self.encoder = CNNtoMLP(out=feat_dim)              # Φ(s)
        self.inverse = nn.Linear(feat_dim * 2, action_dim) # g_φ
        self.forward_net = MLP(feat_dim + action_dim, feat_dim)

    def intrinsic_reward(self, s, a, s_next):
        phi, phi_next = self.encoder(s), self.encoder(s_next)
        phi_pred = self.forward_net(torch.cat([phi, a], -1))
        return 0.5 * (phi_next - phi_pred).pow(2).sum(-1)

    def forward_loss(self, s, a, s_next):
        phi, phi_next = self.encoder(s), self.encoder(s_next)
        phi_pred = self.forward_net(torch.cat([phi, a], -1))
        return F.mse_loss(phi_pred, phi_next.detach()) + \
               F.cross_entropy(self.inverse(torch.cat([phi, phi_next], -1)), a)
```

ICM performs well on tasks such as Super Mario Bros. with continuous control plus visual input: with no external reward at all, the agent can cross the whole map purely on curiosity. But, as you may have guessed, it has a fatal weakness called the **noisy TV problem**.

Imagine there is a television in the environment that always plays random static. Whatever the agent does, the picture is completely random and unpredictable. The forward model can never learn to predict the next frame, so the intrinsic reward stays high forever. What is the result? The agent will plant itself in front of the TV without moving, watching the static forever to "farm points", and never complete the real task again.

Is there a way to construct an intrinsic reward that is more robust and simpler, and that also solves the noisy TV problem? That is what RND does.

## 3. Identifying Unvisited States with RND

Random Network Distillation (Burda et al. 2018) takes a simpler and cleverer path. It does not predict the next frame at all, and does not even need to know what the action is!

RND's method is surprisingly simple:

1. First randomly initialize a **target network** $\hat{f}(s)$, then freeze it after initialization and **never update it**
2. Then train a **predictor network** $f_\psi(s)$ to fit the target network's output for the same input

The training loss is:

$$\mathcal{L}_{\text{RND}}(\psi) = \mathbb{E}_s\bigl[\|f_\psi(s) - \hat{f}(s)\|^2\bigr]$$

Then comes the cleverest part: we use the **same prediction error** directly as the intrinsic reward:

$$r^{\text{int}}(s) = \|f_\psi(s) - \hat{f}(s)\|^2$$

Wait, you may ask, what magic is this? Why can the error in predicting the output of a randomly initialized fixed network represent novelty?

Let's stop and think about how this mechanism works:

- For states that have been **visited many times**: the predictor network $f_\psi$ has seen them repeatedly in training and has had ample opportunity to fit the outputs for these inputs, so the error is usually small.
- For a new state seen for the **first time**: the predictor network has never seen it in training, so naturally it does not fit well, and the error is usually large.
- The target network $\hat{f}$ is a fixed random network. It does not need to represent any task semantics. Its only role is "to provide a fixed, random target value for each input", like issuing every state a unique "ID card".

It is like memorizing vocabulary: words you see often you remember well (small error), and a new word you see for the first time you certainly cannot answer (large error). RND is essentially "remembering" which states have been seen and which are new.

RND has several big advantages over ICM:

- **No inverse model is needed**, which saves half of the computation
- **It does not depend on actions**, so it can be stacked directly onto any model-free algorithm (PPO, A2C, and DQN all work)
- **Natural resistance to the noisy TV**: the complexity of the random target network is finite (it is just a fixed-size neural network), so even against random noise the prediction error has an upper bound and is not pushed up without limit. The agent watches the TV for a while, finds that the predictor network can slowly fit the random output, the intrinsic reward drops, and it naturally walks away.

```python
class RND(nn.Module):
    def __init__(self, obs_shape, feat_dim=512):
        # Target network: frozen, never updated
        self.target = CNN(obs_shape, feat_dim)
        for p in self.target.parameters():
            p.requires_grad = False
        # Predictor network: trained
        self.predictor = CNN(obs_shape, feat_dim)

    def intrinsic_reward(self, s):
        with torch.no_grad():
            target = self.target(s)
        pred = self.predictor(s)
        return (pred - target).pow(2).sum(-1)  # one scalar per state
```

Burda et al. found a surprising result in large-scale experiments: with the RND intrinsic reward alone (no external reward at all), a PPO agent can discover complex behavior on several Atari games. More importantly, on Montezuma's Revenge, where the external reward is extremely sparse, RND for the first time let the agent break through the zero score.

### 3.1 Comparing ICM and RND

Let's put the two methods side by side to help you understand where each applies:

| Dimension                    | ICM                                     | RND                       |
| ---------------------------- | --------------------------------------- | ------------------------- |
| Depends on actions           | Yes (the forward model needs $a$)       | No                        |
| Submodules to train          | Encoder + inverse + forward             | Predictor only            |
| Robustness to noisy TV       | Weak                                    | Strong                    |
| Computational cost           | High                                    | Medium                    |
| Representative applications  | Visual exploration (Mario, DMLab)       | Atari hard-exploration    |

ICM and RND each solve part of the problem, but they share a blind spot. Can you think of what it is?

Here is a hint: a state may be seen for the first time **in the current episode** (short-term novelty), yet from the point of view of **the whole training process** you have already seen it thousands of times in past episodes. Can the neural-network fitting error of ICM and RND distinguish these two kinds of novelty?

## 4. Managing Short-Term and Long-Term Exploration Together

The answer is no. From the prediction error of a neural network alone, we cannot tell "first time in this episode" from "first time since training began".

For example, suppose you are playing a game in which you spawn in the same room at the start of every episode. You have entered this room hundreds of times, so in the long run it is not novel at all. But at the beginning of each episode you really have "just entered" this room. If the intrinsic reward only gives "prediction error", you would receive a high reward at the start of every episode, which is a waste, because you already know this room and do not need to explore it again.

Never Give Up (NGU, Badia et al. 2020) and its successor Agent57 (Badia et al. 2020) model exploration at **short-term** and **long-term** time scales together, and thereby became the **first algorithm to surpass human performance** on the full set of 57 Atari games.

### 4.1 NGU's Two-Timescale Intrinsic Reward

NGU's central idea is direct: split the intrinsic reward into two parts and multiply them:

$$r^{\text{int}}_t(s) = r^{\text{episodic}}_t(s) \cdot r^{\text{life-long}}_t(s)$$

Let's look at the two parts separately:

**The short-term (episodic, within-episode) part** $r^{\text{episodic}}$: it maintains a fixed-capacity table that records the features of states visited **within the current episode**. For a new state, if its distance to all the states in the table is large (a large k-nearest-neighbor distance), it has hardly been visited in this episode and its novelty is high. If it has already been visited frequently in this episode, the novelty decays quickly.

A simplified form is:

$$r^{\text{episodic}}_t = \frac{1}{\sqrt{k} + c \sum_{i=1}^{k} \frac{1}{\sqrt{d(s_t, s_i)}}}$$

Let's explain the parts of this expression:

- $k$: the number of neighbors used in the comparison (the k in kNN)
- $s_i$: the $k$ historical states in the table closest to the current state $s_t$
- $d(s_t, s_i)$: the distance from the current state to the neighbor $s_i$. If this episode has just been here, the table has a nearly coincident neighbor and $d$ is close to 0.
- $c$: controls how fast repeated visits cause decay
- Intuitively: the closer the neighbors, the larger $1/\sqrt{d}$, the larger the denominator, and the lower the within-episode novelty

Let's compute with numbers (taking $k=1$, $c=1$). If the nearest-neighbor distance of the current state in the table is $d=4$, the novelty is $1/(1 + 1/\sqrt{4}) = 1/1.5 \approx 0.67$, fairly fresh. If this episode has already been here, the nearest-neighbor distance shrinks to $d=0.01$, and the novelty falls to $1/(1 + 10) \approx 0.09$. The second occurrence of the same place within an episode leaves only about one-seventh of the original intrinsic reward. This is how short-term novelty is used up quickly within an episode.

**The long-term (life-long, across-episode) part** $r^{\text{life-long}}$: this part uses RND directly, recording across episodes which states are truly new over the whole training process. It is responsible for recognizing states that were "not visited in the current episode but have been visited repeatedly in past episodes", such as the spawn room mentioned above.

What happens when the two are multiplied? Only states that are **both rarely seen in the current episode and rare in the whole training history** receive a high intrinsic reward. This avoids the problem of "farming points" at the spawn location at the start of each episode.

### 4.2 Propagating Rewards with Retrace and Distributed Actors

The two-timescale reward alone is not enough, because the horizon of hard-exploration games is still very long and the reward signal has difficulty propagating back. NGU adopts the distributed architecture of R2D2: many actors sample in parallel, an LSTM handles partial observability, and the Retrace($\lambda$) algorithm estimates off-policy Q-values stably, propagating the intrinsic reward signal over very long decision sequences.

::: details Bonus: Details of the Retrace Algorithm
Retrace($\lambda$) is an off-policy return estimation method that balances bias and variance through truncated importance sampling, and is more stable than ordinary Q($\lambda$) in off-policy settings. For tasks with very long horizons (for example, needing hundreds of steps to obtain a reward), Retrace passes the reward signal backward more effectively. NGU combines R2D2's experience replay and prioritized sampling with LSTM memory in order to succeed on games such as Montezuma's Revenge that need long-range planning. The whole system is extremely expensive to train, requiring billions of game frames and hundreds of TPU cores, but it showed for the first time that Atari hard-exploration games can be conquered by end-to-end RL.
:::

### 4.3 How Agent57 Adaptively Chooses the Exploration Intensity

NGU leaves one problem behind: the weight $\beta$ of the intrinsic reward is fixed. What trouble does this cause?

- On simple games (Pong, Space Invaders), if $\beta$ is set too high, the agent behaves like a curious child looking all around and does not make good use of the optimal policy it has already learned to score.
- On hard-exploration games, if $\beta$ is set too low, the agent is too "utilitarian", explores too little, and never reaches the first reward.

What can we do? We surely cannot tune $\beta$ separately for each of the 57 games. **Agent57** solves this with a clever method, an **adaptive policy scheduler**:

- It maintains **a family of policies** $\pi_i$, each with different exploration parameters $(\beta_i, \gamma_i, c_i)$, with the parameters covering a range from "pure exploitation, no exploration" to "pure exploration, ignore the score"
- A meta-controller estimates the relative return of each policy online and preferentially samples well-performing policies to collect data
- During training, all policies share the same replay buffer and Q-network, so the data is common to all

In this way we no longer need to fix a $\beta$ by hand for each game. The algorithm decides for itself when to explore more and when to exploit more. On simple games it automatically samples the "exploitation" policies more, and on hard-exploration games it gives the "exploration" policies more chances.

Agent57 was the first algorithm to exceed the human baseline score on all games in the Atari 57 suite, which shows that one system can cover both the tasks that need strong exploration and those that need strong exploitation.

## Section Summary

Intrinsic motivation supplements sparse-reward tasks with a sustainable training signal, letting the agent explore the environment "curiously" before the external reward arrives. Let's review the evolution of this line of methods:

- ICM uses the prediction error of a forward model and filters out irrelevant noise through an inverse model, but is limited by the noisy TV problem
- RND uses random network distillation, which is simpler and more robust, does not depend on actions, and is naturally noise-resistant
- NGU combines short-term within-episode novelty with long-term across-episode novelty, solving the problem of repeatedly exploring the spawn point
- Agent57 goes further and adaptively selects exploration policies of different intensities according to task performance, finally surpassing humans on all 57 Atari games

This line of methods significantly improved performance on hard-exploration games such as Montezuma's Revenge, and it also provides basic ideas for exploration in later LLM agents.

But in this section we have assumed all along that there is only one agent in the environment. What if several agents are learning in the environment at the same time, as when you play a multiplayer game in which teammates and opponents are all updating their policies? For a single agent, the environment becomes "non-stationary": even if you stand in the same state and take the same action, the outcome may differ because the other agents have changed. This is the problem the next section addresses.

The next section, [12.2 Multi-Agent RL: CTDE, MADDPG, MAPPO](./marl), turns to the challenge of multiple agents: when several agents learn in the environment at once, non-stationarity breaks the familiar MDP assumption.
