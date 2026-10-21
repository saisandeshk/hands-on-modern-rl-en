---
title: 3.R Summary and Exercises
---

# 3.R Summary and Exercises

## What You Should Now Be Able to Do

1. Distinguish one-step reward $r$, trajectory return $G_t$, and expected value $V^\pi(s)$ or $Q^\pi(s,a)$, and compute each in a small numeric example.
2. Write down the Bellman expectation equation for $V^\pi$ and $Q^\pi$ and explain the two averages it contains (over actions via $\pi$, over next states via $P$).
3. Write down the Bellman optimality equations for $V^*$ and $Q^*$, and explain why averaging becomes $\max$ when behavior is optimal from now on.
4. Convert between $V^\pi$ and $Q^\pi$ in both directions, and compute advantages $A^\pi = Q^\pi - V^\pi$ with the zero-mean check.
5. Read the finite-state matrix form $\boldsymbol{v}_\pi = \boldsymbol{r}_\pi + \gamma \boldsymbol{P}_\pi \boldsymbol{v}_\pi$, state when $(\boldsymbol{I} - \gamma \boldsymbol{P}_\pi)^{-1}$ exists, and explain why the closed form does not scale.
6. State what the Bellman operators $T^\pi$ and $T^*$ are, why they are $\gamma$-contractions, and what their fixed points are.
7. Run the chapter scripts, read $V^*$, the greedy policy, and the training curves, and describe how synchronous value iteration propagates reward one layer per sweep while Q-learning updates one visited pair per step.

## Key Equations

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]
$$

One-line reading: the average long-term return from $s$ when all future actions follow $\pi$.

$$
Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]
$$

One-line reading: the same average, except the first action is fixed to $a$.

$$
V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s,a)
$$

One-line reading: state value is the policy-weighted average of action values in that state.

$$
V^\pi(s) = \sum_a \pi(a \mid s)\left[R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, V^\pi(s')\right]
$$

One-line reading: current value equals expected immediate reward plus discounted expected next-state value.

$$
V^*(s) = \max_a\left[R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, V^*(s')\right]
$$

One-line reading: the optimal value assumes the best first action and optimal behavior afterward.

$$
Q^*(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s',a') \mid s,a\right]
$$

One-line reading: the optimal action value is consistent with acting greedily from the next state on.

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s), \qquad \sum_a \pi(a \mid s)\, A^\pi(s,a) = 0
$$

One-line reading: advantage measures gain over the policy average, so its policy-weighted mean is zero.

$$
\delta = r + \gamma V(s') - V(s), \qquad V(s) \leftarrow V(s) + \alpha \delta
$$

One-line reading: the TD error is the gap between the one-sample target and the current estimate.

$$
\boldsymbol{v}_\pi = \boldsymbol{r}_\pi + \gamma \boldsymbol{P}_\pi \boldsymbol{v}_\pi, \qquad \boldsymbol{v}_\pi = (\boldsymbol{I} - \gamma \boldsymbol{P}_\pi)^{-1}\boldsymbol{r}_\pi
$$

One-line reading: all per-state Bellman equations stacked into one linear system with a closed-form solution for $\gamma < 1$.

$$
\lVert T\boldsymbol{u} - T\boldsymbol{v}\rVert_\infty \le \gamma\, \lVert \boldsymbol{u} - \boldsymbol{v}\rVert_\infty
$$

One-line reading: each Bellman backup shrinks the largest table error by $\gamma$, hence iteration converges to a unique fixed point.

## Common Pitfalls

1. **Mixing up $r$, $G$, and $V$.** The reward $r$ is one step, the return $G$ is one realized trajectory, and the value $V$ is an average over many trajectories. Averaging two $G$ samples estimates $V$; a single $G$ does not equal $V$.
2. **Dropping the outer expectation over the next state.** $\mathbb{E}_\pi[G_{t+1} \mid s_t = s]$ is a number, while $V^\pi(s_{t+1})$ is still random from today's viewpoint. The correct step is $\mathbb{E}_\pi[G_{t+1} \mid s_t] = \mathbb{E}_\pi[V^\pi(s_{t+1}) \mid s_t]$, not a direct substitution.
3. **Counting the terminal reward twice.** The reward is received on the transition _into_ the goal or trap, and the terminal state's own value is $0$. Bootstrapping with $V(\text{terminal})$ must contribute $0$, otherwise the entry reward gets an extra discounted copy.
4. **Expecting every TD update to help.** A single sample target $r + \gamma V(s')$ is noisy: in the Section 3.1 example, one update moves $V(s)$ from $3$ to $3.25$ (toward the exact target $3.55$) while the other moves it to $2.6$ (away from it). Only the probability-weighted average update points toward the Bellman target.
5. **Treating `argmax` ties as meaningful.** Several actions can share the maximal $Q$. Different tie-breaking rules (lowest index, random choice) give different but equally optimal arrow patterns; a single arrow differing from a figure does not imply a bug.
6. **Assuming the matrix inverse is a practical algorithm.** $(\boldsymbol{I} - \gamma \boldsymbol{P}_\pi)^{-1}\boldsymbol{r}_\pi$ is an $O(N^3)$ exact solution for tiny problems. It needs the full model and fails for large or continuous spaces, which is why iteration and sampling exist.

## Exercises

### 1. One-step reward versus return versus value

State in one sentence what $r_t$, $G_t$, and $V^\pi(s)$ each describe, then use the Section 3.1 example with $\gamma = 0.9$ and reward sequences $[2, 4, 6]$ and $[2, 1, -3]$ (truncated after three steps) to compute $G_t^{(1)}$, $G_t^{(2)}$, and the two-sample estimate of $V^\pi(s)$.

::: details Solution
$r_t$ is the immediate reward of the current step; $G_t$ is the discounted total along one realized trajectory; $V^\pi(s)$ is the average of $G_t$ over many trajectories from $s$ under $\pi$.

$$
G_t^{(1)} = 2 + 0.9 \times 4 + 0.9^2 \times 6 = 2 + 3.6 + 4.86 = 10.46.
$$

$$
G_t^{(2)} = 2 + 0.9 \times 1 + 0.9^2 \times (-3) = 2 + 0.9 - 2.43 = 0.47.
$$

Two-sample value estimate:

$$
V^\pi(s) \approx \frac{10.46 + 0.47}{2} = 5.465.
$$

:::

### 2. Exact Bellman target versus two TD updates

In state $s$ with $\gamma = 0.9$ and current estimate $V(s) = 3$, the environment goes to $s_1$ with probability $0.7$ (reward $+1$, $V(s_1) = 5$) and to $s_2$ with probability $0.3$ (reward $-1$, $V(s_2) = 0$). Compute (a) the exact one-step Bellman target, (b) the TD target, TD error, and updated $V(s)$ with $\alpha = 0.1$ for each of the two outcomes, and (c) the probability-weighted average of the two updated values.

::: details Solution
(a) Exact target averages both branches:

$$
0.7 \times (1 + 0.9 \times 5) + 0.3 \times (-1 + 0.9 \times 0) = 0.7 \times 5.5 - 0.3 = 3.85 - 0.30 = 3.55.
$$

(b) Reaching $s_1$: TD target $1 + 0.9 \times 5 = 5.5$, error $5.5 - 3 = 2.5$, updated $3 + 0.1 \times 2.5 = 3.25$. Reaching $s_2$: TD target $-1 + 0 = -1$, error $-1 - 3 = -4$, updated $3 + 0.1 \times (-4) = 2.6$.

(c) Average updated value:

$$
0.7 \times 3.25 + 0.3 \times 2.6 = 2.275 + 0.78 = 3.055,
$$

which lies between $3$ and $3.55$. Equivalently, the mean TD error is $0.7 \times 2.5 + 0.3 \times (-4) = 0.55 > 0$, so on average the table moves upward toward $3.55$.
:::

### 3. Advantages average to zero

In some state $s$, $Q^\pi(s, \text{right}) = 10$ and $Q^\pi(s, \text{left}) = 4$ with $\pi(\text{right} \mid s) = 0.7$. Compute $V^\pi(s)$, both advantages, and verify the zero-mean property. Does replacing $Q^\pi$ by $A^\pi$ change the greedy action?

::: details Solution

$$
V^\pi(s) = 0.7 \times 10 + 0.3 \times 4 = 7 + 1.2 = 8.2.
$$

$$
A^\pi(s, \text{right}) = 10 - 8.2 = 1.8, \qquad A^\pi(s, \text{left}) = 4 - 8.2 = -4.2.
$$

Zero-mean check:

$$
0.7 \times 1.8 + 0.3 \times (-4.2) = 1.26 - 1.26 = 0.
$$

No: subtracting the shared $V^\pi(s)$ preserves the ordering, so $\arg\max_a A^\pi(s,a) = \arg\max_a Q^\pi(s,a) = \text{right}$.
:::

### 4. Two synchronous sweeps on a three-state corridor

Consider the deterministic corridor $S \to M \to G$ with reward $-1$ on each move, terminal value $V(G) = 0$, discount $\gamma = 0.9$, and initial table $V_0 \equiv 0$. Using the synchronous update $V_{k+1}(s) = -1 + 0.9\, V_k(s')$, compute $V_1(S)$, $V_1(M)$, $V_2(S)$, and $V_2(M)$ by hand.

::: details Solution
First sweep reads only $V_0 = 0$:

$$
V_1(M) = -1 + 0.9 \times V_0(G) = -1, \qquad V_1(S) = -1 + 0.9 \times V_0(M) = -1.
$$

Second sweep reads $V_1$:

$$
V_2(M) = -1 + 0.9 \times V_1(G) = -1 + 0 = -1,
$$

$$
V_2(S) = -1 + 0.9 \times V_1(M) = -1 + 0.9 \times (-1) = -1.9.
$$

So $M$ is already correct after one sweep ($-1$), while $S$ needs the second sweep to see past $M$ ($-1.9$). Each synchronous sweep propagates nonzero values one layer backward.
:::

### 5. One-line contraction for the policy operator

For fixed $\pi$ and any two tables $\boldsymbol{u}, \boldsymbol{v}$, show that $\lVert T^\pi \boldsymbol{u} - T^\pi \boldsymbol{v} \rVert_\infty \le \gamma\, \lVert \boldsymbol{u} - \boldsymbol{v} \rVert_\infty$.

::: details Solution
For each state $s$:

$$
(T^\pi \boldsymbol{u})(s) - (T^\pi \boldsymbol{v})(s) = \gamma \sum_{s'} P_\pi(s' \mid s)\, (u(s') - v(s')).
$$

Take absolute values and use that the weights sum to $1$:

$$
\begin{aligned}
|(T^\pi \boldsymbol{u})(s) - (T^\pi \boldsymbol{v})(s)|
&\le \gamma \sum_{s'} P_\pi(s' \mid s)\, |u(s') - v(s')| \\
&\le \gamma \max_{s'} |u(s') - v(s')| \\
&= \gamma\, \lVert \boldsymbol{u} - \boldsymbol{v} \rVert_\infty.
\end{aligned}
$$

Since this holds for every $s$, taking the maximum over $s$ on the left gives the claimed contraction. The same argument with $\max_a$ in place of the policy average gives the result for $T^*$.
:::

### 6. Shortest-path return in the 4×4 GridWorld snippet

In the Section 3.2 snippet, every step gives $-1$ (including the step into the goal) and $\gamma = 0.9$. A shortest path takes $6$ steps. Compute its return and explain why the learned start-state values near $-4.7$ for "right"/"down" support the claim that the policy is near-optimal.

::: details Solution
The discounted return of six rewards of $-1$ is

$$
-(1 + 0.9 + 0.9^2 + 0.9^3 + 0.9^4 + 0.9^5) = -\frac{1 - 0.9^6}{1 - 0.9} \approx -4.686.
$$

Since $0.9^6 = 0.531441$, the sum is $(1 - 0.531441)/0.1 = 4.68559$. Learned values around $-4.7$ for the two moves that stay inside the grid are within a few percent of the optimal $-4.69$, while wall-bumping moves are worse (about $-4.8$ to $-4.9$), so the greedy choice "right or down" recovers a shortest path.
:::

### 7. Coding: reproduce the value-iteration experiment

Run the chapter script from the repository root:

```bash
.timeout 300 .venv/bin/python code/chapter03_value_functions/gridworld_q_learning.py --output-dir /tmp/opencode/value-experiment
```

Read the printed lines and `results.json`. Report (a) the number of converged sweeps, (b) $V^*(0,0)$, and (c) the greedy evaluation steps for each of the three $\varepsilon$ schedules. Verify $V^*(0,0)$ by hand from a six-step path with rewards $[-0.01 \times 5, +1]$ and $\gamma = 0.95$.

::: details Solution
Expected output (curve decimals may differ in the last digits across seeds/library versions, but these three numbers are asserted by the script):

- Converged sweeps: $7$.
- $V^*(0,0) \approx 0.728537$.
- Greedy evaluation steps: $6.0$ for all three schedules (success rate $100\%$).

Hand check with $\gamma = 0.95$: $0.95^2 = 0.9025$, $0.95^3 = 0.857375$, $0.95^4 = 0.81450625$, $0.95^5 = 0.77378094$. Then

$$
V^*(0,0) = -0.01(1 + 0.95 + 0.9025 + 0.857375 + 0.81450625) + 0.77378094 \approx -0.04524381 + 0.77378094 = 0.72853713,
$$

matching the program's $0.728537$.
:::

### 8. Coding: terminal convention and tie-breaking

Using the same `code/chapter03_value_functions/` directory: (a) run the regression tests with `.venv/bin/python -m pytest code/chapter03_value_functions/test_gridworld_q_learning.py -v` and report the result; (b) explain in one paragraph why `action_value` must return the bare reward when `done` is true, and why Q-learning code breaks ties among equal maxima instead of always taking action $0$.

::: details Solution
(a) All $5$ tests ($8$ subtests) pass.

(b) The terminal reward is already collected on entry (goal $+1$, trap $-1$), and terminal states are absorbing with value $0$. If `action_value` added $\gamma V(\text{terminal})$ with a nonzero terminal entry, the same reward would be counted twice: once as $r$ and once inside the bootstrap. The test sets $V(\text{GOAL}) = 999$ and still requires `action_value` to return exactly $1.0$. For ties: many grid states have two equally optimal moves (for example, "right" and "down" from the start state). Always taking action $0$ would hide this, bias exploration, and make the printed policy look falsely unique; sampling or `argmax`-with-ties preserves all shortest paths and matches the multi-arrow optimal policy in the figures.
:::
