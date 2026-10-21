# 10.2 Offline Reinforcement Learning through Sequence Modeling

## Reading Guide

**Key Ideas**

- Understand the core idea of Decision Transformer (DT): why RL can drop the Bellman equation entirely and become a conditional sequence-generation problem of "given a target return, predict the action".
- Understand what return-to-go (RTG) is, why it can serve as a condition that controls how good the generation is, and how RTG is used during training and during inference.
- See how Trajectory Transformer discretizes a whole trajectory into tokens and plans with beam search.
- See how Diffuser generates complete trajectories with a diffusion model, and the strengths, weaknesses, and suitable settings of the three sequence-modeling methods.

In the previous section, [10.1](./offline-data-distribution-shift), we studied BCQ, CQL, IQL, and TD3+BC. Each has its own tricks, but at heart they all follow the traditional RL approach: estimate a value function, perform policy improvement, and try every way to keep extrapolation error from blowing up in the Bellman backup.

But let's stop and think: do we really need a Q-function? Do we really need to iterate the Bellman equation over and over?

The 2021 proposal of Decision Transformer gave a surprising answer to this question: no. It steps entirely outside the Bellman framework and redefines offline reinforcement learning as a **conditional sequence-generation problem**. States, actions, and returns are arranged in a sequence, and the model directly learns "when I want to obtain this much return, what action should be taken in the current state?" This idea is exactly the same as how large language models predict the next token!

Does this sound a bit counterintuitive? Without computing Q-values or running value iteration, just by predicting actions, can a good policy really be learned? In this section we resolve the doubt in four steps. First we build Decision Transformer with return-to-go, then we explain how Trajectory Transformer searches complete trajectories, then we introduce the conditional generation of Diffuser, and finally we compare which tasks each of the three methods suits.

## 1. Training Decision Transformer with a Target Return

All the methods in the previous section stabilize the Bellman update by constraining actions, pushing down out-of-dataset Q-values, or adding a BC regularizer. Decision Transformer (Chen et al. 2021, DT for short below) takes a revolutionary route: it no longer learns a Q-function and no longer performs Bellman updates, but rewrites offline trajectories directly as a conditional sequence-generation problem.

### 1.1 Using Return-to-Go as a Condition

Let's first recall what a trajectory looks like: $\tau=(s_1,a_1,r_1,s_2,a_2,r_2,\ldots,s_T,a_T,r_T)$, triples of state, action, and reward arranged in time order.

Traditional RL looks backward from the future: at time t, you estimate how much reward you will receive in the future (the Q-value) and then pick the action that maximizes it. DT reverses this logic. It looks forward: at step t, in addition to the current state, the model is directly told "**from now until the end of the episode, how much total return I want to obtain**", and it is then asked to predict which action was actually taken in the historical data in that situation.

This "sum of the remaining rewards from the current time to the end of the trajectory" has a dedicated name, **return-to-go (RTG)**:

$$\hat{R}_t = \sum_{t'=t}^{T} r_{t'}$$

Let's understand it with a concrete small example. Suppose the rewards obtained in the last three steps of a trajectory are 2, 1, and 3. What are the RTGs at these three positions?

- Step one (before receiving 2): $\hat{R}_1 = 2 + 1 + 3 = 6$ (6 points can still be obtained)
- Step two (after receiving 2, before receiving 1): $\hat{R}_2 = 1 + 3 = 4$ (4 points remain)
- Step three (after receiving 1, before receiving 3): $\hat{R}_3 = 3$ (only the last 3 points remain)

You see, as time goes on, the reward already obtained is deducted from the target RTG, and $\hat{R}_t$ always means "how much return is still to be obtained". Given $\hat{R}_t$ and $s_t$, the training task becomes remarkably simple: make the model predict the action $a_t$ actually taken in the data in this situation.

How do we feed this information into a Transformer? DT reorganizes the trajectory into a sequence with the three elements interleaved:

$$\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots, \hat{R}_T, s_T, a_T$$

Each time step contributes three tokens in turn: RTG, state, and action. A familiar GPT-style causal attention mask then ensures that when predicting $a_t$ the model can see only the current and earlier information and cannot peek at the future:

$$\pi_\theta(a_t \mid \hat{R}_t, s_t, a_{t-1}, \ldots) = \text{Transformer}(\hat{R}_{1:t}, s_{1:t}, a_{1:t-1})$$

Let's interpret this expression:

- Left side $\pi_\theta(a_t \mid \ldots)$: the probability that the model chooses action $a_t$ in the current context (historical RTGs, states, and actions, plus the current RTG and current state)
- Right side $\text{Transformer}(\ldots)$: the Transformer reads all the RTGs and states up to time t and the actions up to time t-1, and then outputs the distribution of $a_t$

Note the most crucial difference: **the whole training process never invokes a Bellman update!** It is purely supervised learning, with the goal of making the predicted action as close as possible to the action actually taken in the training data. What does this mean? It means data loading, optimizers, distributed training, and all the mature toolchain you use for training GPT/BERT can be used directly!

Looking at the code, you will find it is a standard GPT with slightly modified inputs:

```python
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim, n_heads, n_layers,
                 max_ep_len=4096, K=20):
        super().__init__()
        self.K = K  # context length: how many recent timesteps to condition on
        self.act_dim = act_dim
        # Three embedding layers: RTG, state, action each mapped to hidden_dim
        self.embed_rtg  = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(act_dim, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        # Positional encoding: timestep embedding
        self.pos_emb = nn.Embedding(max_ep_len, hidden_dim)
        # The GPT backbone
        self.transformer = GPT(
            d_model=hidden_dim, n_heads=n_heads, n_layers=n_layers,
            # Key: each timestep takes 3 tokens, so the attention mask must match
            attn_pdrop=0.1, resid_pdrop=0.1
        )
        # Action prediction head (regression, assuming continuous actions)
        self.action_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, rtg, states, actions, timesteps):
        B, T, _ = states.shape
        # Embed and interleave: (R1, s1, a1, R2, s2, a2, ...)
        rtg_emb   = self.embed_rtg(rtg)
        state_emb = self.embed_state(states) + self.pos_emb(timesteps)
        action_emb = self.embed_action(actions)

        # Stack the three into (B, 3T, H), in the order RTG, state, action
        stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=1)
        stacked = stacked.permute(0, 2, 1, 3).reshape(B, 3 * T, -1)
        stacked = self.embed_ln(stacked)

        # Causal attention: each token can only see the past
        h = self.transformer(stacked)
        # Take the outputs at the state positions to predict the corresponding action
        h_states = h[:, 1::3, :]  # indices 1, 4, 7, ...
        return self.action_head(h_states)  # Regress to continuous actions

    @torch.no_grad()
    def act(self, state, target_rtg, history):
        # At inference, use the target RTG as a "prompt" and generate actions autoregressively.
        # history stores past (rtg, state, action) rows; append the current
        # (target_rtg, state) plus a zero placeholder action so the three
        # sequences share one length T, then add the batch dimension back.
        rtg = torch.cat([history.rtg, target_rtg.view(1, 1)], dim=0)[-self.K:]
        states = torch.cat([history.states, state.view(1, -1)], dim=0)[-self.K:]
        actions = torch.cat([history.actions,
                             torch.zeros(1, self.act_dim)], dim=0)[-self.K:]
        timesteps = torch.arange(states.shape[0]).unsqueeze(0)
        pred_a = self.forward(rtg.unsqueeze(0), states.unsqueeze(0),
                              actions.unsqueeze(0), timesteps)
        return pred_a[0, -1]  # Take the prediction of the last timestep
```

Note an implementation detail: each time step takes 3 tokens (R, s, a), so the position indices of the state tokens are 1, 4, 7, ..., that is, starting from 1 and taking every third one, and the line `h[:, 1::3, :]` does exactly this.

### 1.2 Predicting Actions with Supervised Learning

DT's training loss is outrageously simple: it is just the MSE for continuous-action regression (or cross-entropy for discrete actions), with nothing fancy:

$$\mathcal{L} = \mathbb{E}_{\tau \sim \mathcal{D}}\left[\sum_t \|\hat{a}_t - a_t\|^2\right]$$

Here:

- $\hat{a}_t$: the action predicted by the model at time t
- $a_t$: the action actually taken at time t in the training data
- Use mean squared error for continuous actions, and replace MSE with cross-entropy loss for discrete actions

The training process is exactly like training any sequence model: cut a history window of length K (like GPT's context window), feed it to the Transformer, let it predict the action at every position, then compute the loss against the true actions and backpropagate. There is no target network, no soft update, and no conservative regularizer. It is pure supervised learning, as simple as it gets.

### 1.3 Using the Target Return at Inference Time

Training is this simple, so how is it used at deployment time? You will find it even simpler. DT does not need to take an argmax of Q over the action space. You only need to **specify a target RTG** (for example, check roughly what score expert trajectories in your dataset get and set the target to that number), and DT then generates actions autoregressively, like GPT generating text, trying to make the cumulative return of the whole trajectory approach the target you set.

Here is the inference pseudocode:

```python
target_return = 9000  # Roughly the expert-level score in the HalfCheetah environment
state = env.reset()
history = TrajectoryBuffer()  # Stores the most recent K steps of history
for t in range(max_steps):
    action = model.act(state, target_return, history, t)
    next_state, reward, done, _ = env.step(action)
    history.append(state, action, reward)
    state = next_state
    # Key: subtract the reward actually received from the RTG at each step, as the "remaining target"
    target_return -= reward
```

The key line here is `target_return -= reward`: after every step executed in the environment and the actual reward received, that reward is subtracted from the remaining target. The RTG fed to the model at the next step then accurately represents "how much return is still needed until the end of the episode".

For a concrete example, suppose you set a target total return of 6000 points.

- The first step goes very well and earns a reward of 100, so target_return becomes 5900. This tells the model "good, 5900 points remain, keep it up."
- The second step goes badly and earns a reward of -50 (hit a wall), so target_return becomes 5950. This tells the model "you messed up just now, and in the rest of the road you must find a way to earn 50 more points to make up for it."

That is how plain the mechanism is! But you may ask: "What if I set the target RTG very high, higher than any trajectory in the dataset?" The answer is that the model may generate strange, incoherent actions never seen in the training data, because it has never seen the condition "get such a high score" during training and naturally cannot generalize to it. So do not set the target RTG carelessly. Choose it with reference to the return range in your dataset.

### 1.4 Why Decision Transformer Can Work

At this point you surely have a question: this is such a simple method, just a sequence model predicting actions, neither estimating Q nor doing a Bellman backup. How can it work? And the paper says it can match or even beat CQL on many D4RL benchmarks?

Whether Decision Transformer works depends on whether your dataset contains a stable correspondence among states, actions, and final returns. If your dataset contains **trajectories of different quality**, with expert trajectories (high RTG), medium trajectories (medium RTG), and random trajectories (low RTG), then the RTG condition can help the model tell them apart: in similar states, how did those past good trajectories behave to obtain a high return?

Specifically:

- Given a very high target RTG, the conditional distribution $p(a \mid \hat{R}_{\text{high}}, s)$ learned by the Transformer naturally leans toward actions that often appear in high-return trajectories
- Given a very low target RTG, it generates actions like a random policy
- You can understand it intuitively as: **imitating "those trajectories that once reached a similar RTG in similar states"**

Written formally, the policy learned by DT can be expressed as:

$$\pi_\theta(a \mid s, \hat{R}) \propto \exp\left(-\frac{1}{2\sigma^2}\|a - f_\theta(s, \hat{R})\|^2\right)$$

This is a Gaussian distribution whose mean is the Transformer's regression output $f_\theta(s, \hat{R})$ and whose variance is $\sigma^2$. When $\sigma \to 0$ it degenerates into a deterministic policy $a = f_\theta(s, \hat{R})$.

What is the relation between this distribution and the behavior policy $\pi_\beta$ that generated the data? A simple one:

$$\pi_\theta(a \mid s, \hat{R}) \approx \pi_\beta(a \mid s, \text{return} \approx \hat{R})$$

In other words, what DT learns is nothing other than the **conditional distribution of the behavior policy under a specified return**. If you give it a high RTG target, it reproduces the action patterns of the trajectories in the data that obtained high returns. If you give it a low RTG, it reproduces the actions of poor trajectories.

Of course, this also means it has limits: for state-action combinations not covered by the data it still cannot predict reliably, and whether it can "stitch" good segments from different trajectories together depends on the dataset and the ability of the Transformer.

Even so, DT's contribution is pioneering. It spawned a whole line of research on "RL via supervised learning": RvS (offline RL cast directly as goal-conditioned supervised learning), in-context RL (Algorithm Distillation), StARformer, and Trajectory Transformer–style planning work can all trace their source to DT.

### 1.5 Limitations of Decision Transformer

Although DT is simple and elegant, it is not a silver bullet, and there are a few clear limitations you need to know:

1. **It can only learn the best policy that exists in the data.** This is the most fundamental limit. If your dataset has no expert trajectories at all (for example only random data), then no matter how high you set the target RTG, it cannot produce expert-level behavior from nothing, because it has never seen it.
2. **Weak stitching (trajectory splicing) ability.** Traditional offline RL (such as CQL/IQL) can in theory do subtrajectory stitching: "sew together" the good parts of two suboptimal trajectories to obtain a policy better than any trajectory in the data. But DT is pure supervised learning and cannot achieve this kind of strong compositional generalization. At most it reproduces the average level of the trajectories in the data that are close to the target RTG.
3. **Sensitivity to the choice of RTG.** A target RTG set too high produces incoherent, reckless actions, and one set too low is too conservative and cannot reach high scores. This hyperparameter needs to be tuned carefully for each environment.

## 2. Searching Trajectories with Trajectory Transformer

DT proved that the path of "RL as sequence modeling" can work, and afterward this line developed quickly and produced many works. Two of the most representative follow-ups are **Trajectory Transformer (TT)**, which does not merely predict actions but models the entire trajectory as a token sequence and plans with beam search, and **Diffuser**, which is more radical and generates the whole trajectory at once with a diffusion model.

### 2.1 Discretizing Trajectories and Using Beam Search

The idea of Trajectory Transformer, proposed by Janner et al. 2021, is more thorough than DT's. DT still predicts actions autoregressively by time step, while TT discretizes the RTG, state, action, and reward all into individual tokens and then trains a standard Transformer to predict the next token, modeling the joint distribution of the whole trajectory:

$$p_\theta(\tau) = \prod_{t=1}^{T} p_\theta(s_t, a_t, r_t \mid s_{<t}, a_{<t}, r_{<t})$$

What does this mean? TT does not distinguish what is a state and what is an action. It treats the whole trajectory (the s, a, r of all time steps) as one long sequence of tokens and predicts the next token one by one like a language model.

At inference, it does not simply sample one action as DT does. It uses **beam search**, familiar from machine translation, to maximize the probability of the whole trajectory. And you can even add reward constraints explicitly during the search (for example "I want a final return ≥ X"), which amounts to implicit model-based planning!

What are TT's characteristics? Let's summarize:

- **Advantage**: strong planning ability. During beam search you can look several steps into the future and explicitly account for long-term return, which is very different from DT's single-step prediction.
- **Drawback one**: continuous quantities must be discretized. Every dimension of the state has to be discretized into tokens separately. For example, if the state has 20 dimensions with each dimension discretized into 100 bins, one state takes 20 tokens, the sequence length explodes, and the computation is heavy.
- **Drawback two**: beam-search inference is slow. It has to expand several candidate trajectories at once, and cannot produce an action in a single step as DT does.

## 3. Generating Complete Trajectories with Diffuser

Diffuser, proposed by Janner et al. 2022, takes yet another approach: since what we ultimately want is a good trajectory, why generate it autoregressively one token at a time? Can we use a diffusion model to generate a whole trajectory directly at once?

The idea is very much like diffusion models in image generation: rather than painting pixels from left to right, start from pure noise, denoise step by step, and end up with a clear image. Diffuser applies this idea to trajectory generation.

If the state dimension is $d_s$, the action dimension is $d_a$, and the trajectory length is $T$, a trajectory can be represented as a matrix of shape $T \times (d_s + d_a)$. You can think of it as a "trajectory image" whose horizontal axis is time and whose vertical axis is each component of the state/action.

The training procedure is exactly the same as for an image diffusion model: first add Gaussian noise to a real trajectory step by step until it becomes pure noise, and then have a network predict what noise was added at each step:

$$\min_\theta \; \mathbb{E}_{\tau, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(\tau_t, t)\|^2\right]$$

Let's take this training objective apart:

- $\tau$: a real trajectory sampled from the dataset
- $t$: the diffusion step (from 0 to $K$, where $t=0$ is the clean trajectory and $t=K$ is pure noise)
- $\epsilon \sim \mathcal{N}(0, I)$: the Gaussian noise we actually add
- $\tau_t$: the noisy trajectory after t steps of noise
- $\epsilon_\theta(\tau_t, t)$: the network's prediction of "what noise was added at this step"

The training objective is to make the predicted noise as close as possible to the noise actually added (an MSE loss). This is standard diffusion-model training, without any special modification.

At inference we go in reverse: start from pure random noise and iterate the reverse process over its $K$ diffusion steps. At each step the network predicts the noise, and this noise is subtracted from the current noisy trajectory, and in the end we obtain a clean trajectory that looks like those in the training data.

But wait! What we want is not just trajectories that "look like real data" but **high-return trajectories**. How do we control whether the generated trajectory is good or bad? Diffuser uses **classifier-free guidance**, a well-established technique from image diffusion. During training the conditioning information (such as reward or the starting state) is randomly dropped, so the model learns both conditional and unconditional generation. At inference the predictions of the two are combined by a weighted extrapolation:

$$\tilde{\epsilon}_\theta = (1 + w) \cdot \epsilon_\theta(\tau_t, t, c) - w \cdot \epsilon_\theta(\tau_t, t)$$

Here:

- $c$: the conditioning information (such as "starting from the initial state s0, the final return must be ≥ a high score")
- $\epsilon_\theta(\tau_t, t, c)$: the noise predicted with the condition (denoising toward trajectories that satisfy condition c)
- $\epsilon_\theta(\tau_t, t)$: the noise predicted without the condition (it suffices to generate trajectories that look like real data)
- $w$: the guidance scale. The larger w is, the more strongly the condition c is emphasized, and the higher the return of the generated trajectory, but diversity may drop and the trajectory may look unnatural.

Through this guidance mechanism, the reward condition changes the denoising direction so that high-return trajectories get a higher generation probability. Optimization thereby changes from "explicitly choosing the action that maximizes Q", as in traditional RL, into "sampling from a reward-guided trajectory distribution", yet another completely different idea.

## 4. Comparing the Three Sequence-Modeling Methods

Let's put the three sequence-modeling routes, Decision Transformer, Trajectory Transformer, and Diffuser, side by side so that you can choose according to the setting:

| Dimension                                 | Decision Transformer                 | Trajectory Transformer                       | Diffuser                                 |
| ----------------------------------------- | ------------------------------------ | -------------------------------------------- | ---------------------------------------- |
| What is modeled                           | A conditional policy given RTG       | The joint distribution of a whole trajectory | A diffusion model of a whole trajectory  |
| Discretization                            | No                                   | Yes (every state dimension is discretized)   | No                                       |
| Inference method                          | Autoregressive single-step sampling  | Beam search over multiple steps              | Iterative denoising (dozens of steps)    |
| Planning ability                          | Weak (implicit, single-step)         | Strong (explicit multi-step lookahead)       | Strong (conditional generation)          |
| Stitching ability                         | Weak                                 | Medium                                       | Strong                                   |
| Inference speed                           | Fast                                 | Slow                                         | Medium (needs dozens of denoising steps) |
| Compatibility with the LLM training stack | Strong (architecture closest to GPT) | Strong                                       | Weak (diffusion-model architecture)      |

How do you choose? Here is a simple practical guide:

- If you just want to quickly build a strong baseline that is fast, simple to implement, and compatible with the existing Transformer training stack, choose DT
- If you need to do explicit long-horizon planning and are willing to spend more compute for better planning, try TT
- If you need strong stitching ability, want to combine the good parts of different trajectories, or need complex conditional generation (for example constraining the starting state, intermediate milestones, and the final return at once), Diffuser is the better choice

## Section Summary

In this section we looked at an offline RL route that steps entirely outside the Bellman framework: redefining reinforcement learning as a sequence-modeling problem.

Decision Transformer is the pioneer of this route. It introduces return-to-go as a condition, arranges trajectories as sequences of (R, s, a) triples, and performs pure supervised learning with a GPT-style causal Transformer: given a target RTG and the current state, predict the action directly. Its implementation and training are extremely simple, and it reuses the mature toolchain of large language models completely.

Trajectory Transformer then goes further: it discretizes the whole trajectory into tokens, models the joint distribution with a standard language model, and plans explicitly with beam search at inference. Diffuser introduces diffusion models, iteratively denoising from noise to generate complete trajectories, and uses classifier-free guidance to steer generation toward high-return trajectories, with stronger stitching and planning ability.

But have you noticed that all of this "learning from a fixed dataset" is useful not only in traditional RL settings such as robot control? Doesn't it look a bit like preference alignment of large language models today? When DPO trains, doesn't it also have to use an already-collected preference dataset and cannot freely interact with the environment or annotators? Could there be a deeper connection here?

In the next section, [10.3 Offline Reinforcement Learning and Preference Data](./experiments), we answer this question: bring the offline RL perspective to LLM preference optimization, see why DPO is in fact a form of implicit offline Q-learning, and see which post-training phenomena the offline RL way of thinking can help us understand.
