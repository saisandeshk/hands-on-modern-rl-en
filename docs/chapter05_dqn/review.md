---
title: 5.6 Summary and Exercises
---

# 5.6 Summary and Exercises

This chapter turned tabular Q-Learning into a deep reinforcement learning algorithm that works on continuous states and pixel input. The core substitution is one line — a neural network $Q(s,a;\theta)$ replaces the Q table — and everything else in the chapter exists to make that substitution train stably: experience replay breaks sample correlation, and a target network freezes the moving TD target. LunarLander showed the full loop on a low-dimensional control task, and the DQN family (Double, Dueling, prioritized replay, multi-step, distributional, Noisy Nets, Rainbow) showed how each structural weakness of the original DQN gets its own fix. The visual-games project then moved the same algorithm to pixels, where the TD target stays the same but state representation and training conditions become the hard part.

## What You Should Now Be Able to Do

- Explain why a Q table cannot scale to continuous or pixel states, and state the function-approximation replacement $Q(s,a;\theta) \approx Q^*(s,a)$.
- Write the DQN loss from scratch, including the termination flag and the target network, and map each symbol to a step of the `update()` code.
- Explain the deadly triad (function approximation, bootstrapping, off-policy data) and reproduce the mechanism by which shared parameters can amplify error.
- Justify each DQN component: what breaks if you remove experience replay, and what breaks if the target network tracks the online network exactly.
- Run and evaluate the LunarLander DQN: read a multi-episode evaluation curve against a random-policy baseline, and diagnose failures with replays and checkpoints.
- Compute Double DQN and Dueling DQN targets by hand, and explain which bias or structural problem each one fixes.
- Compute prioritized-replay sampling probabilities and importance weights from TD errors, and explain what $\alpha$ and $\beta$ control.
- Choose suitable DQN tasks (discrete actions, observable state, timely rewards, explorable with $\varepsilon$-greedy) and say why a task such as full-game Pokemon or Minecraft exceeds vanilla DQN.

## Key Equations

$$
Q(s,a;\theta) \approx Q^*(s,a)
$$

The network replaces the table: one shared parameter vector outputs all action values.

$$
y = r + \gamma(1-d)\max_{a'}Q(s',a';\theta^-)
$$

The TD target: immediate reward plus the discounted best next-state value from the frozen target network ($d=1$ at true termination zeroes the future term).

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\left[\left(y - Q(s,a;\theta)\right)^2\right]
$$

The DQN loss: mean squared TD error over batches sampled from the replay buffer, optimized with the semi-gradient (no gradient flows through $y$).

$$
\mathbb{E}\left[\max_a \hat{Q}(s,a)\right] \geq \max_a \mathbb{E}\left[\hat{Q}(s,a)\right]
$$

Why vanilla DQN overestimates: the maximum of noisy estimates is biased upward.

$$
y_{\text{Double}} = r + \gamma(1-d)\,Q(s',a^*;\theta^-), \quad a^* = \arg\max_{a'}Q(s',a';\theta)
$$

Double DQN: the online network selects, the target network evaluates.

$$
Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')
$$

Dueling DQN: state value plus mean-centered advantage, so $V$ learns "how good is this state" separately.

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \qquad w_i = \left(\frac{1}{NP(i)}\right)^\beta, \qquad p_i = |\delta_i| + \epsilon
$$

(Here $\epsilon$ is the small PER priority floor, not the exploration rate $\varepsilon$; $\alpha$ is the PER exponent, not the learning rate.)

Prioritized replay: sample by TD-error priority, then correct the bias with importance weights.

$$
\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-
$$

Polyak soft update: the continuous alternative to periodic hard copies of the target network.

## Common Pitfalls

- **Evaluating with exploration left on.** Random actions during evaluation contaminate returns; always evaluate the greedy policy ($\varepsilon = 0$) and report means over at least 5 episodes.
- **Reading a single episode as the verdict.** RL curves fluctuate; one lucky landing or one good Pong game proves little. Judge the evaluation mean and variance across episodes and checkpoints.
- **Chasing the loss instead of the return.** A falling MSE can mean the predictions caught up with a stale target. Track mean Q, target scale, and absolute TD error on a fixed reference batch, plus the actual evaluation return.
- **Updating the target network too eagerly or never.** Copying every step removes the stabilization the target network exists for; copying almost never leaves the Q-network chasing outdated values. The same applies to Polyak $\tau$ near 1 or near 0.
- **Letting the replay buffer go stale or tiny.** A buffer that is too small holds only the latest failures; stored transitions that predate a reward-function fix carry the old semantics forever. Version the buffer when the environment changes.
- **Applying vanilla DQN where its assumptions fail.** Continuous actions, heavily partial observability, or rewards thousands of steps away need different algorithms or task decomposition — not a bigger Q-network.

## Exercises

1. **Table size.** A robot arm has 3 joints, each discretized into 10 angles, and 5 discrete gripper commands. How many state-action entries does a Q table need? If each joint is instead discretized into 20 angles, by what factor does the table grow?
2. **TD target by hand.** With $r = 1.0$, $\gamma = 0.99$, $d = 0$, target-network next-state values $[2.0, 8.0]$, and current $Q(s,a;\theta) = 5.0$: compute the TD target, the TD error, and the squared error.
3. **Semi-gradient.** Write the gradient of a single-sample squared TD loss with respect to $\theta$, first as a full gradient (differentiating through the target) and then as the semi-gradient DQN actually uses. Which terms differ, and why does DQN drop them?
4. **Baird-style divergence trace.** In the two-state sketch of Section 5.2 ($V(s_1)=w$, $V(s_2)=2w$, $\gamma=0.9$, $r=0$, repeated $s_1\to s_2$ transitions, $\alpha=0.1$, $w_0=1$), compute $w_1, w_2, w_3$ and state the growth factor per step.
5. **Double DQN by hand.** Next-state online values are $[3.0, 1.0]$ and target-network values are $[2.0, 2.5]$, with $r = 0$, $\gamma = 1$, $d = 0$. Compute the vanilla DQN target and the Double DQN target. Which action does each select, and which target is lower?
6. **Dueling arithmetic.** Given $V(s) = 5$ and advantages $[2, 0, 4, -2]$, compute the four Q values with mean subtraction, and verify the resulting advantages have mean zero.
7. **PER arithmetic.** Four transitions have $|\delta| = [0.1, 0.5, 1.0, 2.0]$, $\epsilon \approx 0$, $\alpha = 1$, $\beta = 1$, $N = 4$. Compute the sampling probabilities, the raw weights $w_i$, and the weights after dividing by the batch maximum.
8. **Code: fixed-batch differential test.** Using `code/chapter05_dqn/dqn_cartpole.py` (or your Section 5.2 implementation; see also `code/chapter05_dqn/double_dqn_cartpole.py` for a runnable Double DQN reference), sample one fixed batch, clone the network, and apply 50 updates with the vanilla max target versus the Double target. Report the mean predicted Q and mean absolute TD error of each clone on a held-out batch (stating which target each error is scored against), and explain the difference.
9. **Code: LunarLander ablation.** Train `code/chapter05_dqn/dqn_gym_sb3.py` on `LunarLander-v3` for 100k steps with the default settings, then repeat with `target_update_interval` set 10x larger. Compare the evaluation CSVs (means with standard deviations across evaluation episodes): which run's curve rises earlier, which fluctuates more, and how do the final-window means compare? Report the observed direction rather than assuming one in advance.
10. **Design: is DQN suitable?** For each task — (a) CartPole, (b) Pong from pixels, (c) a robot arm with continuous torque commands, (d) completing Pokemon Red end-to-end — state whether vanilla DQN is suitable and name the specific assumption (discrete actions, observable state, timely rewards, explorable) that holds or fails.

::: details Solution

1. States: $10^3 = 1000$; times 5 actions $= 5000$ entries. At 20 angles per joint: $20^3 \times 5 = 8000 \times 5 = 40000$, a factor of $40000/5000 = 8 = 2^3$.
2. Target: $1.0 + 0.99 \times 8.0 = 8.92$. TD error: $8.92 - 5.0 = 3.92$. Squared error: $3.92^2 = 15.3664$.
3. Single-sample loss $L = (y - Q)^2$ with $y = r + \gamma\max_{a'}Q(s',a';\theta^-)$ sharing parameters in the one-network case. Full gradient: $\nabla L = -2(y-Q)(\nabla Q - \gamma\nabla\max Q')$. Semi-gradient: $\nabla L \approx -2(y-Q)\nabla Q$, treating $y$ as constant (in code, `torch.no_grad()` on the target branch). DQN drops the $\gamma\nabla\max Q'$ term because differentiating through the bootstrapped target couples the prediction to its own reference answer and destabilizes training; the target network makes the "constant target" view approximately true between syncs.
4. Update: $w_{k+1} = w_k + 0.1 \times 0.8\,w_k = 1.08\,w_k$. So $w_1 = 1.08$, $w_2 = 1.1664$, $w_3 = 1.259712$; growth factor $1.08$ per step, diverging from the true fixed point $w = 0$.
5. Vanilla: $\max = 2.5$, target $= 2.5$ (selects action 2 via the target net). Double: online argmax is action 1 ($3.0 > 1.0$), target evaluates it at $2.0$, target $= 2.0$. The Double target is lower; it refuses to score the online network's favorite action with the most flattering noise.
6. Mean advantage: $(2 + 0 + 4 - 2)/4 = 1$. Q values: $[5+2-1, 5+0-1, 5+4-1, 5-2-1] = [6, 4, 8, 2]$. Centered advantages: $[1, -1, 3, -3]$, mean $0$.
7. Priorities $\approx [0.1, 0.5, 1.0, 2.0]$, sum $3.6$. $P = [0.0278, 0.1389, 0.2778, 0.5556]$. Raw $w_i = 1/(4P_i) = [9.0, 1.8, 0.9, 0.45]$. Dividing by the max $9.0$: $[1.0, 0.2, 0.1, 0.05]$. The highest-error sample is drawn most often and keeps the largest rescaled weight ($1.0$); low-error samples are drawn rarely and keep small weights, so each individual update stays dominated by high-error transitions while the weights cap the overall step size.
8. Expected outcome: the Double clone ends with a lower mean Q on the held-out batch, because the vanilla clone's max-target updates keep selecting each batch element's most overestimated action. The mean absolute TD error gap is not a fixed prediction — scored against the vanilla max target, the Double clone can show a larger error precisely because it refuses the inflated maximum, while scored with each variant's own target the comparison is close to a wash. Exact numbers depend on the seed; the graded point is the protocol (fixed batch, cloned start, held-out evaluation) and the direction of the Q gap, not any particular value. See also `code/chapter05_dqn/double_dqn_cartpole.py` for a runnable Double DQN reference.
9. Task: run both settings with the script defaults (`--total-timesteps 100000`, `--eval-freq 10000`, `--eval-episodes 10`) and report what you observe: plot both evaluation CSVs with means and standard deviations, state which curve rises earlier, which fluctuates more, and how the final-window means compare. A common outcome is that the 10x-lag run learns from staler targets and its curve lags behind the default, but the direction and size of the gap vary by seed (note that Mnih et al. (2015) themselves used a 10,000-step target interval, so a larger interval is not automatically worse) — the graded point is the comparison protocol and the reported numbers, not a pre-declared winner. Exact numbers vary by seed — report both CSVs and compare means with standard deviations, not single episodes.
10. (a) Suitable: discrete actions, fully observed 4-vector, reward every step, $\varepsilon$-greedy explores it easily. (b) Suitable with CNN + frame stacking: discrete actions and short feedback, but needs the full wrapper/replay/target pipeline. (c) Not suitable: continuous torques cannot be an argmax over a Q-vector output — use DDPG/TD3/SAC (Chapter 9). (d) Not suitable end-to-end: sparse story rewards thousands of steps out and hidden flags/menus break observability and explorability — decompose into short subtasks first (Section 5.5 appendices).
    :::

## References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. _Nature_, 518(7540), 529-533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. _AAAI_.
- Wang, Z., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. _ICML_.
- Schaul, T., et al. (2016). Prioritized Experience Replay. _ICLR_.
- Hessel, M., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. _AAAI_.
- Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_, 2nd ed. MIT Press. Chapters 9-11 (function approximation, the deadly triad, Baird's counterexample).
- Tsitsiklis, J. N., & Van Roy, B. (1997). An analysis of temporal-difference learning with function approximation. _IEEE Transactions on Automatic Control_, 42(5), 674-690.
- Baird, L. (1995). Residual algorithms: Reinforcement learning with function approximation. _ICML_.
