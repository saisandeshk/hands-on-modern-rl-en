# 9.4 Search and World Models

## Reading Guide

**Key Ideas**

- Understand how AlphaZero combines Monte Carlo tree search (MCTS) with neural networks, where the policy network narrows the width of the search and the value network shortens its depth, and how self-play reaches superhuman strength.
- Take apart the PUCT formula and see how its three terms (exploitation, prior, exploration) balance each other, and why this is more effective than plain UCB.
- Understand MuZero's central innovation, implicit model learning: without knowing the rules of the game, it learns dynamics and plans in a hidden space, which makes "general-purpose reinforcement learning" real.
- Master the recurrent state-space model (RSSM) of Dreamer V3: how it models deterministic trajectories and stochasticity together, and trains an actor-critic inside "imagination".
- Compare the strengths and weaknesses of model-based and model-free methods, know when to use which, and see how these ideas inspired PRM search and world models in the LLM era.

In the previous section, [9.3](./model-based), we covered the "data augmentation" route of model-based RL. Dyna, PETS, and MBPO all use a model to generate extra data, and the policy is still trained in the model-free way. Model-based RL has another route that is more radical and more powerful: **explicit search + neural-network value estimation**.

Let's stop and think about how a person plays chess. You do not output a move directly. You look ahead a few steps: "if I play here, the opponent will probably answer there, and then I play here...". You rehearse future possibilities in your head, compare the outcomes of different moves, and pick the one that looks best. This "thinking" process is **search**.

From AlphaGo (2016) defeating Lee Sedol, to AlphaZero (2017) surpassing every human Go program starting from scratch, to MuZero (2019) learning without being told the rules, to Dreamer V3 (2023) working "out of the box" on 150+ tasks with a world model, this line of work represents the theoretical ceiling of model-based RL, and it directly inspired Process Reward Model reasoning search in today's LLM era. In this section we walk through that line of evolution.

## AlphaZero and the Full Realization of Search with Learning

The line AlphaGo (2016) → AlphaGo Zero (2017) → AlphaZero (2017) → MuZero (2019) represents another philosophy of model-based RL: **instead of merely learning a model to generate data, embed search directly in the decision process. Fast neural-network value estimates guide the search, and the search results in turn train the neural network.**

How powerful is this combination of "neural network + search"? Starting from scratch, with no human game records at all and learning only through self-play, AlphaZero defeated Stockfish, the world-champion chess program, in 4 hours, defeated Elmo, the world-champion shogi program, in 24 hours, and surpassed every human Go program in 72 hours. And, most strikingly, **the same algorithm with the same hyperparameters reached superhuman level in three completely different games: Go, chess, and shogi**.

### The Core AlphaZero Loop

The core of AlphaZero is a combination of MCTS (Monte Carlo tree search) and neural networks. Let's look at the pseudocode of the search function first, and then take each part apart:

```python
def alphazero_search(state, neural_net, n_simulations=800):
    root = MCTSNode(state)
    for _ in range(n_simulations):
        # 1. Selection: pick the best child by PUCT
        node = root
        while not node.is_leaf():
            node = node.select_child()

        # 2. Expansion: evaluate the leaf with the neural network
        policy, value = neural_net(node.state)
        node.expand(policy)

        # 3. Backup: propagate the value back to the root
        node.backup(value)

    # Return the visit-count distribution at the root as the action probabilities
    return root.compute_action_distribution()
```

This search loop runs 800 simulations: before actually making a move, the agent "plays" 800 games in its head. Each simulation has three steps: selection, expansion/evaluation, and backup.

But wait, you may ask: "Isn't that different from traditional MCTS? Traditional MCTS has to roll out to the end of the game to get a value. Why can we simply use the neural network's value output here?"

This is exactly the two big innovations of AlphaZero over traditional MCTS. Two networks address the two pain points of search:

- **The policy network** $p_\theta(a \mid s)$ outputs a prior probability that says "how good each action looks in this state". Its role is to **narrow the width of the search**. Traditional MCTS must consider all legal actions at every node on an equal footing, but with a policy network we only need to search the actions with high prior probability.
- **The value network** $v_\theta(s)$ outputs an estimate of "the probability of eventually winning from this state". Its role is to **shorten the depth of the search**. Traditional MCTS must roll out to the end of the game (the terminal state) to obtain a reward, but with a value network the leaf node is evaluated directly by $v_\theta$ and there is no need to search deeper.

These two changes alone improved search efficiency by several orders of magnitude. Traditional MCTS needs millions of simulations to play good Go, while AlphaZero needs only 800.

### The PUCT Formula

The heart of the search is this question: at each node of the tree, which child should we expand first? AlphaZero scores each candidate action with the PUCT (Predictor + Upper Confidence Bound) formula:

$$
\text{PUCT}(a) = Q(s, a) + c_{\text{prior}} \cdot p_\theta(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}
$$

The formula looks a little complicated, so let's split it into three terms, each with a clear meaning:

1. **First term: $Q(s, a)$**. This is the **exploitation term**. It is the current value estimate of action $a$, that is, the average value this path has obtained in the simulations so far. The higher $Q$ is, the better this action has performed historically, and the more we should keep considering it.

2. **Second term: $c_{\text{prior}} \cdot p_\theta(a \mid s)$**. This is the **prior term**. $p_\theta(a|s)$ is the policy network's "probability that this action deserves consideration". An action that the policy network likes gets an initial priority even if it has hardly been explored. $c_{\text{prior}}$ is a constant that controls the weight of the prior.

3. **Third term: $\sqrt{N(s)} / (1 + N(s, a))$**. This is the **exploration term**, which comes from the classic UCB (Upper Confidence Bound) algorithm. $N(s)$ is the total number of visits to the current node and $N(s, a)$ is the number of times action $a$ has been selected. This term says:
   - The more total visits, the larger the exploration bonus (we already have enough data to exploit, so we can afford to explore more);
   - The less often an action has been selected, the larger its exploration bonus (this encourages trying actions that have hardly been tried).

Let's get a feel for it with concrete numbers. Suppose the root node has been visited 100 times ($N(s)=100$) and has two actions:

- Action A: $Q=0.8$ (looks very good), $p=0.3$ (the policy network also thinks it is decent), $N(s,a)=60$ (already selected 60 times)
- Action B: $Q=0.2$ (looks mediocre so far), $p=0.4$ (the policy network thinks it is actually quite promising), $N(s,a)=5$ (tried only 5 times)

Taking $c_{\text{prior}}=1$, let's compute the PUCT scores:

- PUCT(A) = 0.8 + 1 × 0.3 × √100 / (1 + 60) = 0.8 + 0.3 × 10 / 61 ≈ 0.8 + 0.049 = 0.849
- PUCT(B) = 0.2 + 1 × 0.4 × √100 / (1 + 5) = 0.2 + 0.4 × 10 / 6 ≈ 0.2 + 0.667 = 0.867

Interesting, isn't it? Although action A has a much higher Q value at the moment, action B has hardly been explored and the policy network considers it promising, so its PUCT score is actually higher, and the next simulation will select B to explore it. This is the subtlety of the exploitation-exploration balance: choose what looks good now, and also give a chance to actions that have "potential but have not been tried".

### Self-Play Training

You may ask: "How are the policy network and the value network trained? At the start they are random, so isn't the search random too?"

The answer is **self-play**. The AI plays against itself, uses the search results as labels to train itself, then searches with the stronger network, plays again, and repeats, getting stronger each round. It is a process of "fighting oneself".

Concretely, each self-play game goes like this:

```python
def self_play_training(network, n_games=10000):
    for game in range(n_games):
        # 1. Self-play
        trajectory = []
        state = initial_state()
        while not state.is_terminal():
            policy = alphazero_search(state, network)
            action = sample_from(policy)
            trajectory.append((state, policy, action))
            state = state.next(action)

        # 2. Label the outcome
        winner = state.winner()
        for s, p, a in trajectory:
            value = +1 if winner == s.current_player else -1
            train_network(s, p, value)
```

Let's go step by step:

1. **Self-play stage**: starting from the initial state, at every step we run MCTS with the current network to obtain a "better action distribution" (the visit counts computed by MCTS), sample an action from this distribution (adding some exploration), and store $(\text{state}, \text{MCTS policy distribution}, \text{action})$.
2. **Labeling stage**: when the game ends, there is a winner. For each step, if the player who moved eventually won, the value label of this step is +1. If that player lost, it is -1.
3. **Training stage**: use this data to train the two networks in a supervised way:
   - Policy network: make its output $p_\theta(a|s)$ as close as possible to the action distribution found by MCTS. MCTS search is stronger than choosing actions directly with the network, so this amounts to "using the search result as a better policy label".
   - Value network: make its output $v_\theta(s)$ as close as possible to the final outcome (+1/-1).

The beauty of this process is that **the labels do not need to be provided by humans; the search generates them itself**. At the beginning the network is random and the search is weak. But even weak search is a little better than a purely random network. Training the network on the results of weak search makes the network a little stronger, and a stronger network makes the search a little stronger, and so on. In this positive feedback loop the network and the search evolve together and eventually reach superhuman level.

Most striking of all, **the whole process needs no human game records**. AlphaZero starts from scratch. It knows no joseki, no openings, none of the Go knowledge that humans have accumulated. It knows only the rules, plays against itself, and after 72 hours it surpasses every human Go program.

::: details Bonus: The Evolution from AlphaGo to AlphaGo Zero to AlphaZero
You may be curious about the differences among these names, so here is a brief overview:

- **AlphaGo Fan (2015)**: the first version to defeat a human professional player. It used a large number of human game records for supervised pretraining and then applied policy-gradient reinforcement learning.
- **AlphaGo Lee (2016)**: the version that defeated Lee Sedol. It still used supervised pretraining on human game records, but with a stronger value network, and its search used 40 TPUs.
- **AlphaGo Master (2017)**: defeated the top human players 60-0 on online platforms. It used a single neural network that outputs both policy and value, and still started from human game records.
- **AlphaGo Zero (2017)**: **abandoned human game records entirely**. It started from scratch with self-play and surpassed AlphaGo Lee in only 40 days, using a single residual network that is simpler and more efficient.
- **AlphaZero (2017)**: generalized the AlphaGo Zero algorithm to chess, shogi, and other perfect-information games. One set of hyperparameters handles all three games, with no domain knowledge required.
  :::

AlphaZero is very powerful, but it has a precondition: you must know the **rules of the game**, that is, what the state transition is, which actions are legal, and what the terminal conditions are. MCTS needs to know "what the next state is after taking action $a$". But what if we do not know the rules? What if the dynamics of the environment are unknown? Atari games come in as pixels and you have no access to the game code, and in the real world the physical rules are too complex to write down.

In that case, as you might guess, we need to learn a model. This model is not used to generate data, though. It is used to plan in a hidden space. That is what MuZero does.

## MuZero and Implicit Model Learning

AlphaZero needs to know the rules of the game (state transitions and legal actions) in order to search. In the real world, most of the time we have no precise environment model at all. How would you write "how will the object move if the robot arm pushes forward 5 centimeters" as an analytical formula? The key innovation of MuZero (Schrittwieser et al. 2019) is this: **without knowing the true rules, learn an implicit model. Encode the state into a hidden space, predict state transitions, rewards, policies, and values in the hidden space, and then run MCTS directly in the hidden space.**

You may ask: "What does 'searching in a hidden space' mean?" The idea goes like this:

- I do not need to predict the pixels or physical state of the next state. Those details are too numerous, and many of them are irrelevant to decisions.
- I only need to learn a **representation network** that encodes the real state $s$ (whether a board or pixels) into a hidden representation $x = h(s)$. The hidden representation only needs to contain "all the information needed to make decisions".
- Then I learn a **dynamics network** that predicts in the hidden space: given the current hidden state $x$ and action $a$, it predicts the next hidden state $x'$ and the immediate reward $r$.
- Finally I learn a **prediction network** that outputs the policy and value directly from the hidden state.

The whole process is a chain of predictions:

$$
s \xrightarrow{h} x_0 \xrightarrow{g} x_1 \xrightarrow{g} x_2 \to \ldots
$$

The real state $s$ is encoded into $x_0$, and the dynamics network $g$ then unrolls step by step in the hidden space. At every step the prediction network $f$ provides the policy and value. The neat part is that **the dynamics of the whole hidden space need not correspond one-to-one with the pixels or physical state of the real world. They only need to correctly predict what matters for decisions (reward, value, policy).**

### MuZero's Three Networks

Concretely, MuZero has three core networks with clearly divided roles:

- **Representation Network** $h(s) \to x$: encodes the real state $s$ (such as a Go board or an Atari pixel frame) into the hidden representation $x$. This is done only once, at the root of the search, "translating" the current real state into a hidden language the model understands.

- **Dynamics Network** $g(x, a) \to x', r$: this is the "implicit world model" itself. Given the current hidden state $x$ and action $a$, it predicts two things: the next hidden state $x'$ and the **immediate reward $r$ obtained by taking this action**. With it we can "look ahead" in the hidden space.

- **Prediction Network** $f(x) \to p, v$: this plays the same role as AlphaZero's policy and value networks. Given a hidden state $x$, it outputs the policy $p$ (the action prior) and the value $v$ (how good this state is). During search it evaluates leaf nodes and provides priors for expanded children.

Let's read the code to see how the three networks cooperate in planning:

```python
class MuZero:
    def plan(self, state, n_simulations):
        # 1. Encode the real state into the hidden space (done only once)
        root_hidden = self.representation(state)
        root_policy, root_value = self.prediction(root_hidden)

        # 2. MCTS searches in the hidden space
        for _ in range(n_simulations):
            self._mcts_iteration(root_hidden)

        # 3. Return the action distribution at the root (same as AlphaZero)
        return root.action_distribution()

    def _mcts_iteration(self, root):
        # Select, expand, and back up in the hidden space
        path = self._select_path(root)
        # Advance one step in the hidden space with the dynamics network
        next_hidden, reward = self.dynamics(path[-1].hidden, path[-1].action)
        # Evaluate the leaf with the prediction network
        policy, value = self.prediction(next_hidden)
        path[-1].expand(policy, reward)
        for node in path:
            node.update(value, reward)
```

Compare this with AlphaZero's search code. Do you see the difference? The whole MCTS procedure is **almost identical** (PUCT selection, expansion, backup), and the only differences are:

- AlphaZero starts from the real state and uses the true game rules for state transitions;
- MuZero also starts from the real state, but **encodes the real state only once, at the root**. After that, all state transitions are predicted in the hidden space by the dynamics network $g$.

The training process is also similar to AlphaZero's: self-play, with the MCTS result as the policy label and the final outcome as the value label. There are just a few more loss terms: they make the rewards predicted by the dynamics network accurate and keep the whole prediction chain consistent over a rollout.

### The Significance of MuZero

What is the real significance of MuZero? It **can learn to plan without knowing the rules of the environment**. It learns the rules itself. They are rules of the hidden space, but as long as they predict reward, value, and policy correctly, they can be used for high-quality search.

For the first time, this let model-based RL generalize to:

- **Atari games**: no access to the emulator code is needed. It learns directly from pixel input and reached model-free SOTA performance on 57 Atari games.
- **Board games**: Go, chess, and shogi, as strong as AlphaZero but without writing the rules in advance.
- **Partially observable environments**: for example imperfect-information games such as poker (MuZero itself was designed for perfect information, but the idea can be extended).
- **Any MDP**: in theory, as long as you can encode the state as a vector, MuZero can learn a model and search on it.

MuZero can be seen as an exploration of a "unified architecture" for model-based RL: one algorithmic idea and one network structure that span visual input (Atari pixels), vector input (boards), discrete actions, and even (in later extensions) continuous actions. But MuZero still uses MCTS, and at every step of the search it still needs many network forward passes, so the computational cost is not low. MuZero was also validated mainly on games and has not reached the level of "general, out of the box".

What pushed world models toward generality and out-of-the-box use is the Dreamer series, and Dreamer V3 in particular.

## Dreamer V3 and a New Generation of World Models

The Dreamer series (Hafner et al. 2020-2023) is the modern flagship of model-based RL. If AlphaZero/MuZero are the "search + model" route, which does a lot of search at planning time, then Dreamer is the "pure model" route: **learn a sufficiently accurate recurrent latent-variable world model, then "dream" inside this model to train an actor-critic. Training involves zero interaction with the real environment, which is used only when collecting data.**

You may ask: "How is this different from MBPO?" MBPO also uses a model to generate data, but it returns to real data after a 5-step model rollout and trains on the mixture. MBPO's model is also a simple feedforward network that does not account for partial observability or long-term memory. Dreamer uses a **Recurrent State-Space Model (RSSM)**, a more powerful world-model architecture that models the determinism and stochasticity of the environment together, and can even handle partially observable environments (POMDPs).

### The Recurrent State-Space Model

Why do we need recurrence? Why latent variables? Let's stop and think. The real world is partially observable: the single frame you see right now does not tell you everything. You cannot see the velocity of an object, or what is behind a door. To make accurate multi-step predictions you need **memory**, which integrates information from past observations. The hidden state $h_t$ of an RNN can do exactly this.

But a deterministic RNN hidden state alone is not enough, because the environment itself has randomness (remember the aleatoric uncertainty discussed with PETS?). So the RSSM models two parts together:

- **The deterministic part**: the RNN hidden state $h_t$, which is updated deterministically over time, integrating all historical information to form "memory".
- **The stochastic part**: the latent variable $z_t$, which represents the random factors in the environment that cannot be fully predicted from history. The RSSM learns two distributions:
  - **Posterior distribution** $q(z_t \mid h_t, o_t)$: after seeing the current observation $o_t$, it infers what the current stochastic latent variable is. This is "knowing after looking".
  - **Prior distribution** $p(\hat{z}_t \mid h_t)$: without looking at the current observation, it predicts from historical memory alone what $z_t$ should be. This is "the prediction before looking".

When training the world model, we want the prior prediction $\hat{z}_t$ to be as close as possible to the posterior inference $z_t$. If the two align well, the model can already predict accurately what will happen next, and can "imagine" a plausible future without seeing real observations.

Here is the code structure of the RSSM:

```python
class RSSM:
    def forward(self, obs_seq, action_seq):
        h = zeros(batch, hidden_dim)
        posterior_zs = []
        prior_zs = []

        for t in range(T):
            # Prior: look only at h_t (historical memory) and predict z_t
            prior_mean, prior_std = self.prior(h)
            prior_zs.append((prior_mean, prior_std))

            # Posterior: look at h_t and the current observation obs_t, and infer z_t
            posterior_mean, posterior_std = self.posterior(h, encoder(obs_seq[t]))
            z = reparameterize(posterior_mean, posterior_std)
            posterior_zs.append((posterior_mean, posterior_std))

            # Use z and the action a_t to update the RNN deterministic hidden state h_t → h_{t+1}
            h = self.rnn(h, z, action_seq[t])

        return prior_zs, posterior_zs
```

This flow deserves a careful look:

1. At every step we have two $z$'s: the prior (predicted from memory alone) and the posterior (corrected after seeing the observation). During training the two are pulled together (with a KL divergence loss).
2. The RNN state $h$ is updated with the **posterior $z$**, because during training we have real observations, so we refresh the memory with the more accurate posterior.
3. But when we run an imagined rollout, that is, when we "dream" inside the world model, we **have no real observations**. We can only sample $z$ from the prior, use it to update the RNN, and continue predicting forward.

If the prior is accurate enough, meaning it can predict a reasonable $z$ without seeing observations, then the imagined rollouts will resemble trajectories in the real environment, and we can train policies on them with full confidence.

### Actor-Critic in Imagination

Given a trained world model, how do we train the policy? Dreamer's approach is very direct: **train the actor-critic entirely inside the world model, with no interaction with the real environment at all**.

The procedure is as follows:

1. Take earlier real interaction data (stored in a dataset) and encode it with the RSSM into initial hidden states (including the deterministic $h$ and the stochastic $z$);
2. Starting from this initial state, choose actions with the current actor and roll the world model out for H steps (usually H=15), entirely in imagination space with no real environment;
3. Store the imagined trajectories and compute the value loss and policy loss on them as in an ordinary actor-critic, updating the actor and critic.

Here is the pseudocode:

```python
# "Dream" inside the world model to train
h = world_model.encode(real_observation_sequence)
for t in range(H):  # H = 15 imagination horizon
    a = actor(h)
    h, r = world_model.predict(h, a)
    imagined_trajectory.append((h, a, r))

# Train the actor-critic on imagined trajectories
for (h, a, r) in imagined_trajectory:
    critic_loss = ...
    actor_loss = ...
```

Wait, you may ask, how is this different from Dyna? Dyna also generates data inside a model and then trains. The difference is that Dreamer's world model is far stronger:

- The RSSM has RNN memory, can handle partially observable environments, and makes long-horizon predictions more accurately;
- The latent variables model the randomness of the environment, rather than making simple deterministic predictions;
- Dreamer trains the actor-critic directly in **latent space**. The actor's input is the world model's hidden state $h$ rather than the raw observation, which makes policy learning easier (the hidden state is already a high-level feature);
- Imagined rollouts can run 15 steps or longer, because the world model is accurate enough that the error does not explode as it does with a simple feedforward model.

The whole training loop is: **collect a small amount of data with the real environment → train the world model on the data → run a large amount of imagined training inside the world model → collect more data in the real environment with the trained policy → repeat**. Real data is collected so that the world model sees more scenarios and becomes more accurate. The policy-gradient updates, which need a large number of samples, are done entirely in imagination at almost no cost.

### The Generality of Dreamer V3

Dreamer V1 (2020) already worked well on many continuous-control tasks, but hyperparameters still had to be tuned for each domain. Dreamer V2 (2021) introduced discrete latent variables and reached model-free SOTA on Atari. **Dreamer V3 (Hafner et al. 2023)** achieved a true milestone: **with a single hyperparameter setting, it matched or surpassed model-free SOTA on tasks across 150+ different domains**.

These tasks include:

- **Atari**: discrete actions + visual pixel input (26 games)
- **MuJoCo**: continuous actions + vector state input (the standard continuous-control benchmark)
- **Crafter**: an open-world survival game (visual input, sparse reward, complex task chains)
- **DMLab**: first-person 3D navigation (visual input, first-person view)
- **BSuite**: a suite of cognitive tasks designed to test the core capabilities of RL algorithms

What does this mean? Before this, no RL algorithm, model-free or model-based, could use one set of hyperparameters across all these domains. To do well on Atari you had to use Atari-specific hyperparameters, and to do well on MuJoCo you had to use MuJoCo-specific ones. Dreamer V3 showed for the first time that the world-model route can be **general**.

So what did Dreamer V3 get right that made it suddenly so general? The paper summarizes three key engineering innovations. They are all, in the end, engineering details, but together the effect is striking.

### Three Key Engineering Innovations

1. **Discrete latent variables (Discrete Latents)**. Dreamer V1/V2 already used discrete latents, but Dreamer V3 does it properly: it changes the continuous Gaussian latent variable $z$ into a categorical distribution, in the spirit of VQ-VAE. Why? Discrete latents train more stably, avoid the variance-collapse problem of Gaussians, and model multimodal futures more easily (there may be several plausible futures from the same state).

2. **Symlog loss**. Reward scales differ enormously across tasks: rewards in MuJoCo may be 0-10, scores in Atari may be 0-1000, and rewards in Crafter are very sparse. If the value function fits the raw scale directly, the gradients from large-reward tasks drown out those from small-reward tasks. Dreamer V3 uses a simple but extremely effective transformation, symlog:

$$
\text{symlog}(x) = \text{sign}(x) \log(|x| + 1)
$$

   What does this function look like? For positive numbers it is $\log(x+1)$, for negative numbers it is $-\log(-x+1)$, it is approximately linear near 0, and it grows logarithmically at large absolute values. It automatically compresses the range of large values while keeping the sign. The value network predicts in symlog space, and the output is mapped back with symexp (the inverse of symlog), so learning stays stable however large the reward scale is.

3. **No KL annealing (No KL Annealing)**. When training latent-variable models one usually uses KL annealing: the KL term has a small weight at the start so the model first learns to reconstruct observations, and the weight is then gradually increased to align prior and posterior. Dreamer V3 found this entirely unnecessary. It fixes the KL weight at 1 and directly maximizes the ELBO (evidence lower bound), letting the posterior match the prior naturally. Removing the KL-annealing hyperparameter greatly improves robustness.

With just these three ingredients (discrete latents, symlog loss, and a fixed KL weight), Dreamer V3 works "out of the box" on 150+ tasks without per-task tuning. This is a first in the history of RL.

::: details Bonus: The World-Model Loss of Dreamer V3
When Dreamer V3 trains the world model, it maximizes the ELBO, and the loss function can be split into three terms:

$$
\mathcal{L}_{\text{world model}} = \mathcal{L}_{\text{rec}} + \beta_{\text{KL}} \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{reward}}
$$

- $\mathcal{L}_{\text{rec}}$: the reconstruction loss, which decodes the latent variable back to the observation (image or state) so that the latent variable contains enough information to reconstruct the true input. It uses MSE or a discrete entropy loss for visual input, and MSE for vector input.
- $\mathcal{L}_{\text{KL}}$: the KL divergence, which pulls the posterior $q(z_t|h_t,o_t)$ toward the prior $p(z_t|h_t)$, so that the $z$ sampled from the prior during imagination is reasonable. In Dreamer V3, $\beta_{\text{KL}}=1$, fixed and untuned.
- $\mathcal{L}_{\text{reward}}$: the reward prediction loss, which aligns the model's predicted immediate reward with the true reward.

Training the three terms together end to end teaches the whole world model to "know what is going on after looking, and still predict the future when not looking".
:::

The success of Dreamer V3 convinced many people that **world models are a viable path toward general reinforcement learning**. You learn a sufficiently good world model and then train the policy inside it, like a person who first builds an understanding of the world in their head, rehearses and practices there, and only then acts in the real world.

Having covered the three lines AlphaZero → MuZero → Dreamer V3, we can now step back and answer a broader question: **what are the strengths and weaknesses of model-based and model-free methods, and which should I choose in practice?**

## Model-Based vs. Model-Free, and When to Use Which

Let's put model-free and model-based side by side in one table and look at their trade-offs along several dimensions:

| Dimension                | Model-Free                  | Model-Based                                   |
| ------------------------ | --------------------------- | --------------------------------------------- |
| **Sample efficiency**    | Low (millions of steps)     | High (tens of thousands of steps)             |
| **Asymptotic performance** | High                      | Limited by model error                        |
| **Computational cost**   | Low (uses data directly)    | High (model training + search/planning)       |
| **Interpretability**     | Black box                   | The model can be analyzed                     |
| **Transferability**      | Weak                        | The model can transfer to downstream tasks    |
| **Hyperparameter sensitivity** | Medium                | High (model quality determines everything)    |

Let's explain these dimensions one by one:

- **Sample efficiency**: this is the biggest advantage of model-based methods. As we computed earlier, in settings where real samples are expensive, model-based methods can cut the number of interactions by 10-100 times.
- **Asymptotic performance**: with unlimited samples, model-free methods usually reach higher final performance because they carry no model bias. If the model is imperfect (and in practice it never is perfect), a model-based method eventually hits a ceiling set by model error, while a model-free method can keep optimizing toward the optimal policy given enough samples.
- **Computational cost**: this is the price of model-based methods. You must train the model in addition to the policy, and with the AlphaZero/MuZero search route, deployment also requires a lot of computation at every step. After training, a model-free policy needs one network forward pass at deployment, which is very fast.
- **Interpretability**: the model itself is an understanding of the environment dynamics. You can inspect what the model predicts, where its predictions are inaccurate, and where it is uncertain. The policy network and Q network of a model-free method are basically black boxes, and it is hard to explain why they choose a given action.
- **Transferability**: if you have already learned a world model for one environment, then for a new task (a different reward function) you do not need to relearn the dynamics. You can train the policy directly in the existing model under the new reward. A model-free method has to retrain from scratch for each new task.
- **Hyperparameter sensitivity**: model-based methods are more sensitive to hyperparameters, especially model-related ones such as model capacity, model training frequency, and rollout length. If the model is learned badly, everything is wasted. Model-free methods also have hyperparameters but are comparatively more robust.

So how do you choose in practice?

**When to choose model-free:**

- **The simulator is cheap**: in Atari, MuJoCo, StarCraft and the like, thousands of steps run per second and samples are nearly free. There is no need to save samples, and SAC/PPO are simple, direct, and effective.
- **Only the final performance matters**: with no limit on training time or sample count, and the goal of the strongest possible final policy, the asymptotic performance of model-free methods is usually higher.
- **Deployment compute is limited**: on robots, phones, and edge devices there is no extra compute to run a model and search. A model-free policy network produces an action in one forward pass, with the lowest latency.

**When to choose model-based:**

- **Real-environment samples are expensive**: this is the most typical scenario. Real robots, autonomous driving, chemical-reaction control, and industrial process optimization all carry time, money, and safety costs for every real interaction, and the 10-100 times sample efficiency of model-based methods is the difference between "usable" and "unusable".
- **Fast adaptation to new tasks is needed**: in meta-RL, online learning, and continual learning, you already have a pretrained world model, so a new task can be planned or trained inside the model and adapted to in seconds, without another million interactions with the environment.
- **Interpretability and safety are needed**: in safety-critical settings (autonomous driving, medicine), you want to know "why" the agent made a decision, predict what it will do next, and verify before deployment that the policy will not fail. The model provides internal representations that can be analyzed.

This is not an either-or choice. In practice you can also mix them: pretrain the policy inside a world model and then fine-tune it model-free with a small number of real samples, or use model-based methods for high-level planning and model-free methods for low-level control, just as the human brain has a world model for high-level thinking while much low-level motor control is reflexive (model-free).

## Connections to RL for LLMs

If you have been working with LLMs, some of this may feel familiar. Many core ideas in LLM training and inference today can be traced back to model-based RL. Once you understand the concepts in this section (search, world models, and model bias), you can see why some recent directions in the LLM field look so natural.

In LLM training and inference:

- **The model-free route**: this is the familiar RLHF and GRPO. A reward model (RM) scores the whole answer directly, and a policy-gradient algorithm such as PPO/GRPO optimizes the language model. This is purely model-free: the language model itself is the policy, the reward model is the reward signal, and there is no "world model" in between.

- **The model-based / search route**: the **Process Reward Model (PRM)** and inference-time search (such as MCTS, beam search, best-of-n) are a direct embodiment of the model-based idea. A PRM is like AlphaZero's value/policy network and scores **every step** of a reasoning chain. A search algorithm (such as MCTS) then searches the "space of possible reasoning steps" for a better reasoning chain, rather than outputting the answer in one go. This is exactly the same logic as in AlphaZero, where "MCTS uses the network to guide search, and the search result is stronger than the network itself". We will go deeper into this in [Chapter 17: PRMs and Search](../chapter20_prm_search/inference-time-search).

- **The world-model route**: the recent **Code World Model** and the **world-model components in SWE-Agent** ([Chapter 20](../chapter23_rl_based_swe/world-model-and-deep-swe)) rebuild the ideas of MuZero/Dreamer in the LLM field. They learn a "code world model" that predicts "what will happen if I make this edit and execute this code, and will the tests pass", then search and plan patches inside this world model, improving the sample efficiency of solving code problems.

You can see that reasoning models such as Tongyi DeepResearch and OpenAI o1/o3 use PRMs to guide search, and code agents such as SWE-Agent use world models to improve efficiency. None of these ideas appeared out of thin air. They are natural extensions of core model-based RL ideas into the LLM era. Once you understand the fundamental trade-offs between model-based and model-free methods (sample efficiency vs. asymptotic performance, computational cost vs. decision quality, model bias vs. unbiased but inefficient), you can understand more deeply why the technical routes of LLM post-training and inference look the way they do today.

## Chapter Summary

Continuous control and model-based RL are the two major advanced directions of classical deep reinforcement learning. In this chapter we walked through three main lines of technical evolution:

1. **DDPG → TD3 → SAC**: the evolution of deterministic policy gradients, which solves the problem of off-policy learning with continuous actions. From the pioneering work of DDPG, to TD3 stabilizing training with three engineering patches (twin Q, delayed updates, target smoothing), to SAC fundamentally reformulating the objective with maximum-entropy RL. With a stochastic policy that has exploration built in, automatic temperature tuning, and out-of-the-box usability, SAC has become the preferred continuous-control algorithm today.

2. **Dyna → PETS → MBPO**: the evolution of the model-based data-augmentation route. From Dyna's simple "model as an extra data generator", to PETS modeling two kinds of uncertainty with probabilistic ensembles plus CEM MPC planning, to MBPO sidestepping error accumulation with short-horizon rollouts. This line uses the model as a "data amplifier" and still trains the policy in the model-free way, but improves sample efficiency by 10-100 times.

3. **AlphaZero → MuZero → Dreamer V3**: the flagship route of explicit search + learned models. AlphaZero combines MCTS with neural networks and reaches superhuman strength through self-play with zero human knowledge. MuZero pushes this into latent space, learning to plan without knowing the rules of the environment. Dreamer V3 uses a recurrent latent-variable world model and, for the first time, works across 150+ tasks with a single hyperparameter setting. This represents the ceiling of model-based RL and directly inspired PRM search and world-model work in the LLM era.

After this chapter you should have a complete picture of "how to do continuous control" and "how to use a model to improve sample efficiency". But we have been carrying an implicit assumption: **the agent can interact freely with the environment and collect as much data as it likes**.

Is that true in reality? No. In many settings, such as recommender systems, medicine, and LLM post-training, you simply cannot "explore by interacting" with the environment online. You have a pile of historical data but cannot deploy a policy to trial and error, or the cost of trial and error is extremely high. What then? This is the central problem of the next chapter: **offline reinforcement learning (Offline RL)**, which asks how to learn a good policy when the agent cannot interact and can only use previously collected historical data. This is the core problem in real settings such as LLM post-training (the offline stages of DPO and RLHF), recommender systems, and medical AI.

The next chapter, [Chapter 10: Offline Reinforcement Learning](../chapter12_offline_rl/offline-data-distribution-shift), answers this question.

## Further Reading

- [Silver et al. 2018 "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)
- [Schrittwieser et al. 2020 "Mastering Atari, Go, chess and shogi by planning with a learned model" (MuZero)](https://arxiv.org/abs/1911.08265)
- [Hafner et al. 2023 "Mastering Diverse Domains through World Models" (Dreamer V3)](https://arxiv.org/abs/2301.04104)
- [Janner et al. 2019 "When to Trust Your Model: Model-Based Policy Optimization" (MBPO)](https://arxiv.org/abs/1906.08253)
