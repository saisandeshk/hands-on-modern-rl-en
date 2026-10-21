---
title: 2.R Summary and Exercises
---

# 2.R Summary and Exercises

This chapter built the modeling language of reinforcement learning. Section 2.1 used the multi-armed bandit to isolate trial-and-error learning and the exploration-exploitation trade-off. Section 2.2 introduced the Markov decision process tuple, the discounted return, and policies. Section 2.3 turned policies into trajectories, trajectories into returns, and returns into value functions that evaluate states and actions before the future unfolds.

## What You Should Now Be Able to Do

1. Compute the expected reward of a bandit arm and of a random policy from payout probabilities, using linearity of expectation.
2. Explain the exploration-exploitation trade-off and estimate both of its costs: opportunities spent exploring and misjudgment from too few samples.
3. Implement epsilon-greedy, UCB, and Beta-Bernoulli Thompson sampling for a bandit, and explain which actions each method directs exploration toward.
4. Write down the MDP 5-tuple for a new task (bandit, CartPole, token generation) and state what each component is in that task.
5. Compute discounted returns by hand for short reward sequences and explain how the discount factor sets the effective horizon.
6. Distinguish termination (absorbing outcome, zero future value) from truncation (external cutoff, future value remains) and say how each is treated when estimating values.
7. Relate policies, returns, and values: define $V^\pi(s)$, $Q^\pi(s,a)$, and $A^\pi(s,a)$, and derive $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s,a)$.
8. Spot reward-design traps: hacking (maximizing the letter of the reward while violating its intent) and shaping that changes the optimal policy.

## Key Equations

- Expected reward: $\mathbb{E}[R_a] = \sum_r r\, P(R_a = r)$. Reading: the long-run average payoff per pull of arm $a$.
- Sample mean: $\hat{\mu}_a = \frac{1}{N_a}\sum_{i=1}^{N_a} r_i$. Reading: estimate an arm's average reward from the samples seen so far.
- Epsilon-greedy: act greedily with probability $1 - \epsilon$, uniformly at random with probability $\epsilon$. Reading: exploit most of the time, keep a fixed exploration budget.
- UCB: $A_t = \arg\max_a \left[\hat{\mu}_a + c\sqrt{\ln t / N_a}\right]$. Reading: prefer arms that look good or have been tried too rarely to judge.
- Regret: $\mathrm{Regret}(T) = T\mu^* - \mathbb{E}[\sum_{t=1}^T R_t]$. Reading: how much was lost by not pulling the best arm every round.
- MDP tuple: $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$. Reading: the situation set, the action set, the dynamics, the scoring rule, and the patience parameter.
- Discounted return: $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma G_{t+1}$. Reading: future rewards summed with shrinking weights; the tail is summarized by the next return.
- Return bound: $G_t \le R_{\max}/(1-\gamma)$. Reading: discounting keeps infinite-horizon returns finite, with effective horizon $H = 1/(1-\gamma)$.
- Optimal policy: $\pi^* = \arg\max_\pi \mathbb{E}_\pi[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)]$. Reading: the decision rule with the largest expected long-term return.
- State and action values: $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$, $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$. Reading: the average future return from a state, and from a state plus a fixed first action.
- Value-action link: $V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s,a)$. Reading: a state's value is the policy-weighted average of its action values.
- Advantage: $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$. Reading: how much better this action is than the policy's average choice.

## Common Pitfalls

1. **Trusting a short sample.** Ten pulls each at 60% versus 40% still misjudge about 12.8% of the time. Small-sample estimates are noisy; neither commit too early nor pay for more exploration than the remaining horizon justifies.
2. **Confusing observed and true rates.** The observed fraction $\hat{p}$ fluctuates around the true parameter $p$. A worse arm can look better in one run without anything about the environment changing.
3. **Mixing up termination and truncation.** Treating a time-limit cutoff as a terminal state with zero future value biases value estimates downward. Truncation keeps the continuation value; only true termination sets it to zero.
4. **Shaping the optimum away.** Adding a "helpful" bonus changes which behavior is optimal. Always ask what policy maximizes the reward as written, not as intended.
5. **Keeping fixed exploration forever.** A constant $\epsilon$ keeps costing reward after the estimates have stabilized (about 15 regret per 1000 rounds at $\epsilon = 0.1$ on a 0.8/0.5 bandit). Decay exploration or direct it at uncertain actions instead.
6. **Reading $V(s)$ as a property of the state alone.** Values carry a policy superscript for a reason: $V^\pi(s)$ changes when the follow-up policy changes. Comparing values across policies without holding the policy fixed is meaningless.

## Exercises

### Exercise 1: Expected Rewards in the Chapter Code

The script `code/chapter02_rl_basics/two_armed_bandit.py` uses 0/1 rewards with win probabilities 0.6 (arm A) and 0.4 (arm B). Compute the expected reward of each arm and the expected per-step reward of the uniform random policy. What cumulative average reward should the random baseline converge to?

::: details Solution
For 0/1 rewards, $\mathbb{E}[R_a] = 1 \cdot p_a + 0 \cdot (1 - p_a) = p_a$. So $\mathbb{E}[R_A] = 0.6$ and $\mathbb{E}[R_B] = 0.4$. The random policy picks each arm with probability 0.5, so by the law of total expectation its per-step expectation is $0.5 \times 0.6 + 0.5 \times 0.4 = 0.5$. Its cumulative average reward converges to 0.5, matching the gray dotted baseline in the script's plot.
:::

### Exercise 2: The $\pm 1$ Bandit by Hand

Use the text's $\pm 1$ reward version (arm A pays out with probability 0.6, arm B with 0.4, +1 for a win and $-1$ for a loss) over 100 rounds. Compute (a) the per-round expectation of each arm, (b) the 100-round expectation of uniform random play, and (c) the 100-round expectation of always playing A.

::: details Solution
(a) $\mathbb{E}[R_A] = 0.6 \times 1 + 0.4 \times (-1) = 0.2$. $\mathbb{E}[R_B] = 0.4 \times 1 + 0.6 \times (-1) = -0.2$.
(b) Half the rounds earn 0.2 on average and half lose 0.2: $0.5 \times 0.2 + 0.5 \times (-0.2) = 0$ per round, so $100 \times 0 = 0$ total.
(c) $100 \times 0.2 = 20$. This is the unattainable upper bound, since it assumes knowing A is better without exploration.
:::

### Exercise 3: Trace the Incremental Mean

Start with estimate $q = 0$ and count $n = 0$. Apply the update $q \leftarrow q + (r - q)/n$ (after incrementing $n$) to the reward sequence $1, 0, 1$. Show each step and confirm the final estimate equals the sample mean.

::: details Solution
After $r_1 = 1$: $n = 1$, $q = 0 + (1 - 0)/1 = 1$.
After $r_2 = 0$: $n = 2$, $q = 1 + (0 - 1)/2 = 0.5$.
After $r_3 = 1$: $n = 3$, $q = 0.5 + (1 - 0.5)/3 = 0.5 + 0.1667 = 0.6667 = 2/3$.
The sample mean is $(1 + 0 + 1)/3 = 2/3$. They match: the incremental rule maintains the exact running average without storing history.
:::

### Exercise 4: UCB Picks the Uncertain Arm

At round $t = 100$ with $c = 1$, arm 1 has estimate 0.65 from 50 pulls and arm 2 has estimate 0.55 from 5 pulls. Compute both UCB scores and say which arm is chosen. Then explain in one sentence why the lower-estimate arm wins.

::: details Solution
$\ln 100 \approx 4.605$. Arm 1: $0.65 + \sqrt{4.605/50} = 0.65 + \sqrt{0.0921} \approx 0.65 + 0.304 = 0.954$. Arm 2: $0.55 + \sqrt{4.605/5} = 0.55 + \sqrt{0.921} \approx 0.55 + 0.960 = 1.510$. Arm 2 is chosen ($1.51 > 0.95$). It wins because its uncertainty bonus ($\approx 0.96$) dwarfs arm 1's ($\approx 0.30$): five samples are too few to rule it out.
:::

### Exercise 5: Recompute the Cost of Misjudgment

Arm A has true payout 0.6 and arm B 0.4, each tried 10 times.

(a) Compute $P(N_A = 4)$ from the binomial formula.
(b) Given a misjudgment probability of about 12.8%, an explore-then-commit policy (20 exploration rounds, 80 commit rounds, $\pm 1$ rewards) earns $+16$ when correct and $-16$ when wrong. Compute its full expected return and the two layers of loss versus the upper bound 20.

::: details Solution
(a) $P(N_A = 4) = \binom{10}{4} 0.6^4 0.4^6 = 210 \times 0.1296 \times 0.004096$. Since $0.1296 \times 0.004096 = 0.0005308416$, the product is $210 \times 0.0005308416 \approx 0.1115$, about 11.1%: roughly one run in nine.
(b) Full expectation: $0.872 \times 16 + 0.128 \times (-16) = 13.952 - 2.048 = 11.904 \approx 11.9$. Layer 1 (exploration cost): $20 - 16 = 4$ points spent sampling arm B. Layer 2 (misjudgment): $16 - 11.9 \approx 4.1$ points from committing to the wrong arm in about one run in eight.
:::

### Exercise 6: Derive the $V$–$Q$ Link and Check Advantages

(a) Starting from the definitions of $V^\pi(s)$ and $Q^\pi(s,a)$, derive $V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s,a)$.
(b) With $Q^\pi(s, \text{left}) = 70$, $Q^\pi(s, \text{right}) = 110$, and policy probabilities $(0.25, 0.75)$, compute $V^\pi(s)$ and both advantages, and verify the policy-weighted advantages sum to zero.

::: details Solution
(a) $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$. The first action $a_t$ is drawn from $\pi(\cdot \mid s)$, so group the expectation by its value using the law of total expectation: $\mathbb{E}_\pi[G_t \mid s_t = s] = \sum_a \pi(a \mid s)\, \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$. Each inner expectation is $Q^\pi(s,a)$ by definition, giving $V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s,a)$.
(b) $V^\pi(s) = 0.25 \times 70 + 0.75 \times 110 = 17.5 + 82.5 = 100$. Advantages: $A(\text{left}) = 70 - 100 = -30$, $A(\text{right}) = 110 - 100 = +10$. Weighted sum: $0.25 \times (-30) + 0.75 \times 10 = -7.5 + 7.5 = 0$, as must hold since $V$ is the weighted average of $Q$.
:::

### Exercise 7: CartPole Returns, Horizons, and Endings

CartPole gives +1 per step.

(a) Compute $G_0$ for an episode lasting $T = 200$ steps with $\gamma = 0.99$.
(b) State the effective horizon for $\gamma \in \{0.9, 0.99, 0.999\}$ and explain what it does and does not mean.
(c) Classify each ending as termination or truncation and say whether the final state's continuation value is zero: (i) the pole falls at step 87; (ii) the episode hits the 500-step cap while balanced; (iii) a training loop cuts a continuing patrol task into 128-step rollout slices.

::: details Solution
(a) $G_0 = \sum_{k=0}^{199} 0.99^k = (1 - 0.99^{200})/(1 - 0.99)$. Since $0.99^{200} = e^{200 \ln 0.99} \approx e^{-2.010} \approx 0.1340$, $G_0 \approx (1 - 0.1340)/0.01 = 86.60$.
(b) $H = 1/(1-\gamma)$: 10, 100, and 1000 steps. It is the sum of the discount weights, a scale for how fast reward weights decay (about 63% of the total weight lies within $H$ steps as $\gamma \to 1$). It is not a hard cutoff: rewards beyond $H$ still contribute with smaller weights.
(c) (i) Termination: the pole fell, no future, continuation value zero. (ii) Truncation: the task could have continued, continuation value is the estimated value of the balanced state, not zero. (iii) Truncation: each slice boundary keeps its bootstrap value; only genuine task failure would be termination.
:::

### Exercise 8 (coding): Add a Decaying-Epsilon Strategy to the Chapter Script

Extend `code/chapter02_rl_basics/two_armed_bandit.py` with a `strategy_epsilon_decay` function (start $\epsilon = 1.0$, multiply by 0.995 each step, floor at 0.01), run it alongside the existing four strategies for 1000 steps averaged over 200 runs, and compare total regret. Explain why it ranks where it does.

::: details Solution

```python
def strategy_epsilon_decay(bandit, n_steps, eps_start=1.0, eps_end=0.01, decay=0.995):
    rewards = []
    Q = np.zeros(2)
    counts = np.zeros(2)
    eps = eps_start
    for _ in range(n_steps):
        if np.random.random() < eps:
            arm = np.random.choice([0, 1])
        else:
            arm = np.argmax(Q)
        eps = max(eps_end, eps * decay)
        reward = bandit.pull(arm)
        rewards.append(reward)
        counts[arm] += 1
        Q[arm] += (reward - Q[arm]) / counts[arm]
    return np.array(rewards)
```

Reference result (200 runs, 1000 steps, seeds 0–199): fixed $\epsilon = 0.1$ totals about 588 (regret $\approx 12$), decaying $\epsilon$ about 579 (regret $\approx 21$), against the optimal 600. The decaying schedule starts at full exploration ($\epsilon \approx 0.61$ still at step 100), so it pays heavy early exploration cost inside a 1000-step horizon; its advantage (vanishing late exploration) only dominates over longer horizons. Your exact numbers will vary with seeds, but the ordering fixed < decay regret here is robust for this horizon and decay rate. Trying a faster decay (e.g. 0.99) or a longer horizon flips the comparison, which is the point: the schedule must match the horizon.
:::

### Exercise 9 (coding): Thompson Sampling From Scratch

Implement Beta-Bernoulli Thompson sampling for the chapter's 0.6/0.4 bandit without looking at the text's pseudocode first, then compare: run it, epsilon-greedy ($\epsilon = 0.1$), and UCB ($c = 2$) for 1000 steps over 200 seeds and report mean totals. Explain the mechanism behind the ranking.

::: details Solution

```python
def strategy_thompson(bandit, n_steps):
    rewards = []
    alpha = [1, 1]
    beta = [1, 1]
    for _ in range(n_steps):
        samples = [np.random.beta(alpha[a], beta[a]) for a in range(2)]
        arm = int(np.argmax(samples))
        reward = bandit.pull(arm)
        rewards.append(reward)
        alpha[arm] += reward
        beta[arm] += 1 - reward
    return np.array(rewards)
```

`Beta(1, 1)` is uniform, expressing total initial ignorance. Each 1 reward increments `alpha`, each 0 increments `beta`, so the posterior mean `alpha / (alpha + beta)` tracks the observed win rate while its width shrinks with evidence, concentrating future pulls on the likely-best arm.

Reference result (200 runs, 1000 steps): Thompson sampling totals about 594–595 (regret $\approx$ 5–6), ahead of epsilon-greedy ($\approx$ 588, regret $\approx$ 12). UCB ($c = 2$) is competitive with Thompson sampling on this easy two-arm gap; the clear gap is between both directed methods and undirected epsilon exploration. Exact totals shift with seeds, but directed exploration (UCB, Thompson) should beat fixed-epsilon randomness, because it spends trials on arms proportional to their chance of being optimal rather than at a flat rate.
:::
