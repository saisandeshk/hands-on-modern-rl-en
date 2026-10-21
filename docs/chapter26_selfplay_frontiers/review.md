---
title: 26.5 Summary and Exercises
---

# 26.5 Summary and Exercises

This chapter followed one question: once a policy can generate its own data, where should the next unit of compute and supervision go? Section 26.1 built self-play and generator-judge loops, Section 26.2 compared training-time and test-time compute with outcome and process rewards, Section 26.3 extended the problem to several language agents that must share credit, Section 26.4 showed how the same closed loop discovers programs when a trustworthy evaluator exists, and the offline-RL supplement handled the case where no new interaction is allowed at all.

## What You Should Now Be Able to Do

1. Distinguish game-theoretic self-play (a changing opponent distribution) from generator-judge and debate loops (a changing data distribution), and name what could collapse in each.
2. Explain why a historical opponent pool or population prevents cycling, using the mixture form of the opponent distribution.
3. Decide, for a fixed total budget, whether extra compute belongs in training (stronger single samples) or at test time (more candidates or search), and set up the comparison fairly.
4. Compare outcome and process supervision (ORM vs. PRM) by signal density, annotation cost, and failure mode, including automated step labels by Monte Carlo rollouts.
5. Write down the multi-agent credit-assignment problem for a planner-coder-reviewer team and apply the hybrid outcome-plus-process reward with a justified weight.
6. Use freeze-rotate training and group-relative team comparison (M-GRPO) to stabilize heterogeneous language agents.
7. State when an AlphaEvolve-style discovery loop applies (code-like candidates plus a cheap, trustworthy evaluator) and audit a claimed discovery with four checks.
8. Choose among behavior cloning, TD3+BC, CQL, IQL, and Decision Transformer for a fixed offline dataset, and explain each choice as a different answer to out-of-distribution actions.

## Key Equations

- Nash check (symmetric teaching form): $V(\pi^*, \pi^*) \ge V(\pi, \pi^*)$ for all $\pi$. Reading: no unilateral deviation beats the equilibrium profile.
- Opponent mixture: $\pi_{\text{opponent}} = \sum_{k=1}^{K} w_k \pi_k$, $\sum_k w_k = 1$. Reading: sampling from a history pool trains against a distribution, not just the latest self.
- Difficulty-matched sampling: $\mathcal{P}^*(d) \propto \mathcal{P}_0(d) \cdot (1 - p(d))$. Reading: spend samples where the pass rate $p(d)$ is low.
- Best-of-$N$ under independent tries: $1 - (1-p)^N$. Reading: test-time compute converts per-sample accuracy $p$ into at-least-one-success probability.
- Team-relative advantage (M-GRPO): $\text{Advantage}_i = (R_i - \mathrm{mean}(R_{1..G})) / \mathrm{std}(R_{1..G})$. Reading: a team's update is its reward measured against the other sampled teams.
- Hybrid multi-agent reward: $R_i = \alpha \cdot R^{\text{outcome}} + (1-\alpha) \cdot R_i^{\text{process}}$, $\alpha \approx 0.5$--$0.7$. Reading: shared outcome plus role-specific process signal, weighted toward the measurable outcome.
- CQL (discrete teaching form): $\mathcal{L}_{\text{CQL}} = \mathcal{L}_{\text{TD}} + \alpha(\mathbb{E}_s[\log\sum_a \exp Q(s,a)] - \mathbb{E}_{(s,a)\sim\mathcal{D}}[Q(s,a)])$. Reading: push down Q everywhere, spare the actions actually seen.
- IQL expectile value: $\mathcal{L}_V = \mathbb{E}_{\mathcal{D}}[L_2^{\tau}(Q - V)]$, $L_2^{\tau}(u) = |\tau - \mathbf{1}(u < 0)|u^2$. Reading: with $\tau > 0.5$, $V$ tracks the good tail of data actions without querying unseen ones.
- IQL policy extraction: $\mathcal{L}_{\pi} = -\mathbb{E}_{\mathcal{D}}[\exp(\beta A)\log\pi(a \mid s)]$, $A = Q - V$. Reading: imitate data actions in proportion to how much better than average they are.
- Decision Transformer: $\pi_{\theta}(a_t \mid \hat{R}_t, s_{\le t}, a_{< t})$ with $\hat{R}_t = \sum_{t'=t}^{T} r_{t'}$. Reading: condition generation on the return you want, then imitate trajectory pieces that achieved it.
- DPO: $\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta \log\frac{\pi_{\theta}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi_{\theta}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)})]$. Reading: prefer the winning response while the reference model anchors the policy near the data.
- World model (teaching abstraction): $\hat{P}(s_{t+1} \mid s_t, a_t)$ or $\hat{P}(o_{t+1} \mid o_{\le t}, a_t)$. Reading: plan by rolling a learned transition forward; model error misleads the whole plan.

## Common Pitfalls

1. Calling every generator-judge loop "self-play" and importing game-theoretic guarantees that do not hold. Generator-judge loops can echo a shared blind spot; only checkable rewards, held-out sets, or independent judges break the echo.
2. Training only against the latest checkpoint. Policies cycle (A beats B beats A) unless a history pool keeps old strategies in the training distribution.
3. Comparing training-time and test-time compute without fixing the other axis. A rising curve with both changing at once does not tell you where the next unit of compute belongs.
4. Treating a world model's visual plausibility as physical correctness. Genie-style environments are for prototyping and controlled evaluation until sim-to-real gaps are measured.
5. Letting one shared team reward carry all the credit. Without process rewards or role-level advantage separation, free-riding roles are reinforced and reviewers stop adding information.
6. Maximizing an offline Q-function as if it were online. The $\max_{a'}$ backup actively seeks the least trustworthy values; use constraints (TD3+BC), conservatism (CQL), expectiles (IQL), or conditioning (DT) instead.

## Exercises

### 1. Name the loop

For each setup, say whether it is (a) game-theoretic self-play, (b) a generator-judge data loop, or (c) neither, and name the collapse mode to watch for: (i) a chess engine training against its own past checkpoints; (ii) one language model writing answers and scoring them with its own judge prompt, retrained with DPO; (iii) supervised fine-tuning once on a fixed human dataset.

::: details Solution
(i) is (a): the opponent distribution moves with the learner; watch for cycling/policy collapse without a history pool. (ii) is (b): the data distribution moves but there is no opponent; watch for self-loop degradation (generator and judge share a style bias and amplify it). (iii) is (c): the data distribution is fixed, so there is no loop to collapse, only ordinary overfitting or distribution shift at deployment.
:::

### 2. Compute best-of-5 by hand

A reasoning model answers a problem correctly with probability $p = 0.30$ per independent sample. A verifier reliably picks a correct answer when one exists. Compute the probability that best-of-5 succeeds, showing each step.

::: details Solution
Failure probability per sample is $1 - 0.30 = 0.70$. All five fail with probability $0.70^5 = 0.70^2 \times 0.70^2 \times 0.70 = 0.49 \times 0.49 \times 0.70 = 0.2401 \times 0.70 = 0.16807$. At least one succeeds with probability $1 - 0.16807 = 0.83193 \approx 0.83$. Weights are unchanged; the gain comes purely from test-time sampling plus verification.
:::

### 3. Verify a mixed Nash profile by hand

Consider a symmetric zero-sum game where the row player's payoffs are $\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$. Show that both players randomizing $(0.5, 0.5)$ is a Nash equilibrium, and compute its value.

::: details Solution
Fix the column player at $(0.5, 0.5)$. The row player's expected payoff for row 1 is $0.5(1) + 0.5(-1) = 0$; for row 2 it is $0.5(-1) + 0.5(1) = 0$. Any mixture of the two rows also yields $0$, so no unilateral deviation improves on $0$. By symmetry the same holds for the column player. Hence $(0.5, 0.5)$ against itself satisfies $V(\pi^*, \pi^*) \ge V(\pi, \pi^*)$ with equality, and the game value is $0$.
:::

### 4. Score three teams with M-GRPO normalization

Three coder-reviewer-tester teams earn total rewards $R = [0.85, 0.60, 0.90]$. Using population standard deviation, compute each team's group-relative advantage and verify the advantages sum to zero.

::: details Solution
Mean: $(0.85 + 0.60 + 0.90)/3 = 2.35/3 \approx 0.78333$. Deviations: $0.06667$, $-0.18333$, $0.11667$. Squared deviations: $0.004444$, $0.033611$, $0.013611$; sum $\approx 0.051667$. Population variance $= 0.051667/3 \approx 0.017222$, so std $\approx \sqrt{0.017222} \approx 0.13123$. Advantages: $0.06667/0.13123 \approx 0.508$, $-0.18333/0.13123 \approx -1.397$, $0.11667/0.13123 \approx 0.889$. Sum: $0.508 - 1.397 + 0.889 = 0.0$ up to rounding. Team 3 is reinforced most, team 2 is pushed down.
:::

### 5. One expectile step by hand

IQL's value loss uses $L_2^{\tau}(u) = |\tau - \mathbf{1}(u < 0)|u^2$ with $\tau = 0.8$. Data actions at one state have $Q$-values $\{2, 5, 9\}$ and the current $V = 4$. Compute each term's weight and the mean loss. Explain what raising $\tau$ toward 1 does.

::: details Solution
Residuals $u = Q - V$: $-2$, $1$, $5$. Weights: $u < 0 \to 1 - \tau = 0.2$; $u \ge 0 \to \tau = 0.8$. Terms: $0.2 \times (-2)^2 = 0.8$, $0.8 \times 1^2 = 0.8$, $0.8 \times 5^2 = 20.0$. Mean loss $= (0.8 + 0.8 + 20.0)/3 = 21.6/3 = 7.2$. Raising $\tau$ weights positive residuals more (weight $\tau$) and negative residuals less (weight $1 - \tau$), so the positive-to-negative weight ratio $\tau:(1-\tau)$ grows and $V$ moves further into the upper tail of the data-action values (toward the best actions seen, not toward unseen actions).
:::

### 6. Read the sign of the CQL penalty

At one state, the mean data-action Q is $1.0$ and three uniformly sampled actions have Q-values $[0.5, 2.0, 1.0]$. Compute $\log\sum\exp$ over the three samples and the resulting conservative penalty $\text{logsumexp} - Q_{\text{data}}$. Does this penalty increase or decrease the loss for $\alpha > 0$?

::: details Solution
Exponentials: $e^{0.5} \approx 1.6487$, $e^{2.0} \approx 7.3891$, $e^{1.0} \approx 2.7183$; sum $\approx 11.7561$. $\log(11.7561) \approx 2.4644$. Penalty $= 2.4644 - 1.0 = 1.4644 > 0$. For $\alpha > 0$ this positive term increases the loss, so gradient descent pushes sampled-action Q-values down relative to data-action Q-values -- exactly the intended conservatism. (With more samples the logsumexp better approximates the max; the sign logic is unchanged.)
:::

### 7. Coding: best-of-$N$ vs. majority vote

Simulate $p = 0.3$ per-sample correctness over 20,000 problems with $N = 5$ tries each. Compare (a) best-of-$N$ with a perfect verifier and (b) majority vote without a verifier. Report both success rates and explain the gap.

::: details Solution

```python
import random

random.seed(0)
p, N, trials = 0.3, 5, 20_000
best_ok = maj_ok = 0
for _ in range(trials):
    draws = [random.random() < p for _ in range(N)]
    best_ok += any(draws)          # perfect verifier picks a correct one if present
    maj_ok += sum(draws) > N / 2   # majority needs at least 3 of 5 correct
print(f"best-of-N: {best_ok / trials:.4f} (theory ~0.8319)")
print(f"majority:  {maj_ok / trials:.4f} (theory ~0.1631)")
```

Typical output: `best-of-N: 0.8337`, `majority: 0.1632`. Theory for majority: $\sum_{k=3}^{5}\binom{5}{k}0.3^k 0.7^{5-k} = 10(0.027)(0.49) + 5(0.0081)(0.7) + 0.00243 = 0.1323 + 0.02835 + 0.00243 = 0.16308$. The gap is the verifier: best-of-$N$ needs only one correct draw, while majority needs most draws correct, which is unlikely when $p < 0.5$.
:::

### 8. Coding: fictitious-play opponent sampling

Implement a minimal loop for the game in Exercise 3 that (a) always best-responds to the latest opponent and (b) best-responds to the average of a history pool (fictitious play). Show that (a) cycles while the pool average in (b) settles near $(0.5, 0.5)$.

::: details Solution

```python
PAYOFF = [[1, -1], [-1, 1]]  # row player's payoffs


def row_br(col):
    v1 = col[0] * 1 + col[1] * -1
    if v1 > 0:
        return [1.0, 0.0]
    if v1 < 0:
        return [0.0, 1.0]
    return [0.5, 0.5]


def col_br(row):
    # Column player minimizes the row player's payoff.
    if row[0] < 0.5:
        return [1.0, 0.0]
    if row[0] > 0.5:
        return [0.0, 1.0]
    return [0.5, 0.5]


print("latest-only (both sides best-respond to the newest opponent):")
row, col = [0.9, 0.1], [0.9, 0.1]
for it in range(6):
    row = row_br(col)
    col = col_br(row)
    print(f"  iter {it}: row={row} col={col}")

print("fictitious play (respond to the pool average):")
pool = [[0.9, 0.1]]
row = [0.9, 0.1]
for it in range(12):
    avg = [sum(m[i] for m in pool) / len(pool) for i in range(2)]
    row = row_br(avg)
    pool.append(list(col_br(row)))
    print(f"  iter {it:2d}: avg={[round(x, 3) for x in avg]} row={row}")
```

Typical output: latest-only alternates forever between `row=[1, 0], col=[0, 1]` and `row=[0, 1], col=[1, 0]` -- a pure cycle that never plays the equilibrium. With fictitious play the pool average moves `0.9 -> 0.45 -> 0.633 -> 0.475 -> ... -> 0.492`, converging toward `[0.5, 0.5]`, at which point the best response is the equilibrium mixture itself. Note this is a one-sided teaching sketch: the row player best-responds to the empirical average of the column's past plays (the pool), while the column here best-responds to the row's current strategy -- textbook fictitious play averages on both sides. In this toy loop "training" is an exact best response for clarity; real PPO takes a small step toward it, which damps but does not remove the cycle -- the pool is still what guarantees coverage of old strategies.
:::
