---
title: 4.R Summary and Exercises
---

# 4.R Summary and Exercises

This chapter built classical tabular RL from one shared idea: maintain a value table and repeatedly move each entry toward a target built from rewards plus future value. Dynamic programming (DP) computes the target from a known model, Monte Carlo (MC) from complete sampled returns, and temporal-difference (TD) learning from one-step rewards plus the current next-state estimate. Control extends the same loop to action values with SARSA, Expected SARSA, and Q-learning; data-source distinctions (on/off-policy, online/offline) with importance sampling decide which data each update may legally use; and reward design decides what all of this machinery optimizes.

## What You Should Now Be Able to Do

1. Write the DP policy-evaluation backup from the Bellman expectation equation and run it by hand on a small MDP.
2. Improve a policy greedily from a value table, and explain how policy iteration, value iteration, and generalized policy iteration (GPI) interleave evaluation with improvement.
3. Compute MC first-visit and every-visit targets from a sampled episode and update a table with the incremental-mean form.
4. Compute TD(0) targets and TD errors step by step, and explain the bias-variance trade-off against MC.
5. Write $n$-step returns and the TD($\lambda$) return, and trace eligibility traces through a short trajectory.
6. Write the SARSA, Expected SARSA, and Q-learning updates from memory, state which is on-policy and which is off-policy, and predict how they differ on risky tasks such as cliff-walk.
7. Apply ordinary and weighted importance sampling to correct off-policy returns, and check support coverage and effective sample size before trusting the result.
8. Distinguish true termination (bootstrap zero) from truncation by a time limit (keep bootstrapping), and spot reward-design failure modes: sparse signals, unsafe dense shaping, weight and scale mismatch, and proxy-versus-true-objective gaps.

## Key Equations

$$
V_{k+1}(s)=\sum_a\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_k(s')\right]
$$

Iterative policy evaluation: sweep the Bellman expectation over every action and next state the model provides.

$$
\pi'(s)=\arg\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right]
$$

Policy improvement: act greedily with respect to the values of the current policy.

$$
V_{k+1}(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_k(s')\right]
$$

Value iteration: one greedy backup per sweep converges directly to the optimal values.

$$
V(s)\leftarrow V(s)+\alpha\left[G_t-V(s)\right]
$$

MC update: nudge the table toward the realized full return (unbiased, higher variance, waits for termination).

$$
\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t),\qquad V(s)\leftarrow V(s)+\alpha\,\delta_t
$$

TD(0): the TD error is the gap between the one-step bootstrapped target and the old estimate.

$$
G_{t:t+n}=R_{t+1}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^nV(S_{t+n}),\qquad G_t^\lambda=(1-\lambda)\sum_{n\ge1}\lambda^{n-1}G_{t:t+n}
$$

$n$-step returns interpolate between TD(0) and MC; TD($\lambda$) averages over all horizons with decaying weights.

$$
Q\!\leftarrow\! Q+\alpha[R+\gamma Q(S',A')-Q],\quad Q\!\leftarrow\! Q+\alpha[R+\gamma\max_{a'}Q(S',a')-Q],\quad Q\!\leftarrow\! Q+\alpha[R+\gamma\textstyle\sum_{a'}\pi(a'\mid S')Q(S',a')-Q]
$$

SARSA (next action actually taken), Q-learning (greedy next action), Expected SARSA (average over next actions).

$$
\rho=\frac{\pi(A\mid S)}{\mu(A\mid S)},\qquad \hat{V}_{\text{ord}}=\frac{1}{n}\sum_i\rho_iG_i,\qquad \hat{V}_{\text{w}}=\frac{\sum_i\rho_iG_i}{\sum_i\rho_i}
$$

Importance sampling: ordinary IS is unbiased but noisy; weighted (self-normalized) IS trades a shrinking bias for lower variance.

$$
F(s,s')=\gamma\Phi(s')-\Phi(s)
$$

Potential-based shaping: telescoping extra rewards leave the optimal policy unchanged.

## Common Pitfalls

1. **Treating DP, MC, and TD as unrelated algorithms.** They edit the same table with targets built under different information constraints (model known, full return observed, one-step plus estimate). Memorizing three recipes without the shared Bellman template makes every later algorithm look new.
2. **Bootstrapping through (or from) the wrong endpoint.** Writing $V(G)\ne0$ for a terminal state, updating terminal entries, or zeroing the bootstrap at a time limit (truncation) instead of only at true termination silently biases every predecessor.
3. **Running Q-learning with the exploration baked into the target.** The behavior policy must stay exploratory ($\epsilon$-greedy) while the target uses the greedy action; sampling the target action $\epsilon$-greedily turns the update into SARSA, and maximizing the behavior action removes exploration.
4. **Trusting off-policy averages without checking coverage.** If $\mu(a\mid s)=0$ anywhere $\pi(a\mid s)>0$, no importance weight can repair the gap; if one trajectory carries nearly all the weight (effective sample size near $1$), the estimate is one sample wearing a disguise.
5. **Reading a high proxy reward as task success.** Step costs, shaping bonuses, multi-term weights, and learned reward models all change the optimum unless constrained (PBRS form, constraints instead of penalties, KL limits). A rising curve next to a broken task is Goodhart's law, not progress.
6. **Stopping value iteration too late or policy iteration too early in the wrong way.** Values converge geometrically (contraction by $\gamma$ per sweep), and the greedy policy usually locks in before values settle, so monitor the max sweep change $\max_s|V_{k+1}-V_k|<\theta$ and stop improving once the greedy actions stop changing.

## Exercises

### Exercise 1 (conceptual): Name the target

For each situation, state whether the natural update is DP, MC, or TD, and what its target is: (a) a known gridworld model and a full sweep with no environment interaction; (b) a robot episode that just terminated, updating from the recorded return; (c) a continuing task where the agent updates after every step using its current next-state estimate.

::: details Solution
(a) DP with $\text{target}_{\mathrm{DP}}(s)=\sum_a\pi(a\mid s)[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_k(s')]$. No sampling is needed because the model supplies all branches. (b) MC with $\text{target}_{\mathrm{MC}}=G_t$, the realized return from the visit to termination. Updates happen only after the episode ends. (c) TD with $\text{target}_{\mathrm{TD}}=R_{t+1}+\gamma V(S_{t+1})$, built from one real step plus bootstrapping, so it updates online in continuing tasks.
:::

### Exercise 2 (by hand): One DP sweep

Use the chapter corridor ($S,M$ nonterminal, $G$ terminal with $V(G)=0$; right with probability $0.8$ for $-1$, left with probability $0.2$ for $-2$; $\gamma=1$). Starting from $V_0(S)=V_0(M)=0$, compute $V_1(S)$ and $V_1(M)$ by hand.

::: details Solution
Expand both action branches and weight by the policy:

$$
V_1(S)=0.8[-1+V_0(M)]+0.2[-2+V_0(S)]=0.8(-1)+0.2(-2)=-0.8-0.4=-1.2,
$$

$$
V_1(M)=0.8[-1+V_0(G)]+0.2[-2+V_0(S)]=0.8(-1)+0.2(-2)=-1.2.
$$

The first sweep records only average immediate cost, since the old table is zero.
:::

### Exercise 3 (by hand): Solve the corridor exactly

Write the two Bellman expectation equations for $V^\pi(S)$ and $V^\pi(M)$ under the same policy and solve for the fixed point. Confirm the chapter values.

::: details Solution
With $\gamma=1$ and $V(G)=0$:

$$
V(M)=0.8(-1)+0.2(-2+V(S))=-1.2+0.2V(S),
$$

$$
V(S)=0.8(-1+V(M))+0.2(-2+V(S))=-1.2+0.8V(M)+0.2V(S).
$$

From the second equation, $0.8V(S)=-1.2+0.8V(M)$, so $V(S)=-1.5+V(M)$. Substituting into the first:

$$
V(M)=-1.2+0.2(-1.5+V(M))=-1.5+0.2V(M),
$$

hence $0.8V(M)=-1.5$, $V(M)=-1.875$, and $V(S)=-1.5-1.875=-3.375$. These match the converged DP table.
:::

### Exercise 4 (by hand): MC and TD on one short episode

Take the short episode $S\xrightarrow{-1}M\xrightarrow{-1}G$ with $\gamma=1$, table initialized to zero, $\alpha=0.5$. (a) List the every-visit MC targets and the table after this episode. (b) List the TD(0) targets step by step and the table after this episode.

::: details Solution
(a) MC waits for termination. Returns: from $M$, $G=-1$; from $S$, $G=-1+(-1)=-2$. Updates: $V(M)=0+0.5(-1-0)=-0.5$, $V(S)=0+0.5(-2-0)=-1$. Final: $V(S)=-1$, $V(M)=-0.5$.

(b) TD updates online. Step 1, $S\xrightarrow{-1}M$: target $-1+V(M)=-1+0=-1$, so $V(S)=0+0.5(-1)=-0.5$. Step 2, $M\xrightarrow{-1}G$: target $-1+V(G)=-1+0=-1$, so $V(M)=0+0.5(-1)=-0.5$. Final: $V(S)=-0.5$, $V(M)=-0.5$. MC already propagated the full $-2$ to $S$; TD gave $S$ only the one-step target because $V(M)$ was still $0$ when $S$ was updated.
:::

### Exercise 5 (derivation): One-step importance sampling is unbiased

Show that $\mathbb{E}_\mu[\rho R]=\mathbb{E}_\pi[R]$ for a fixed state, where $\rho=\pi(A\mid S)/\mu(A\mid S)$, assuming $\mu(a\mid s)>0$ whenever $\pi(a\mid s)>0$. Then give the effective-sample-size formula and evaluate it for weights $(1.6, 0.4, 1.0, 1.0)$.

::: details Solution
Condition on $S=s$ and sum over actions:

$$
\mathbb{E}_\mu[\rho R\mid s]=\sum_a\mu(a\mid s)\frac{\pi(a\mid s)}{\mu(a\mid s)}\mathbb{E}[R\mid s,a]=\sum_a\pi(a\mid s)\mathbb{E}[R\mid s,a]=\mathbb{E}_\pi[R\mid s].
$$

Coverage guarantees no division by zero. Effective sample size:

$$
\mathrm{ESS}=\frac{(\sum_i\rho_i)^2}{\sum_i\rho_i^2}=\frac{(1.6+0.4+1.0+1.0)^2}{1.6^2+0.4^2+1.0^2+1.0^2}=\frac{16}{2.56+0.16+1+1}=\frac{16}{4.72}\approx3.39,
$$

so four nominal samples carry the information of about $3.4$ even samples.
:::

### Exercise 6 (derivation): PBRS leaves the optimal policy unchanged

For a trajectory $s_0\to\cdots\to s_T$ with shaping $F(s,s')=\gamma\Phi(s')-\Phi(s)$ and terminal potential $\Phi(s_T)=0$, show the total shaped addition depends only on $s_0$, and explain why policy ordering is preserved.

::: details Solution
Sum the discounted shaping terms:

$$
\sum_{t=0}^{T-1}\gamma^tF(s_t,s_{t+1})=\sum_{t=0}^{T-1}\gamma^t[\gamma\Phi(s_{t+1})-\Phi(s_t)].
$$

Consecutive terms telescope: $\gamma\Phi(s_1)$ cancels $-\gamma\Phi(s_1)$, $\gamma^2\Phi(s_2)$ cancels $-\gamma^2\Phi(s_2)$, and so on, leaving $-\Phi(s_0)+\gamma^T\Phi(s_T)=-\Phi(s_0)$ since $\Phi(s_T)=0$. Every trajectory from $s_0$ receives the same constant shift, so return differences between policies are unchanged and the optimal policy is unchanged.
:::

### Exercise 7 (coding): Iterative policy evaluation

Implement synchronous iterative policy evaluation for the chapter corridor ($\gamma=1$, $\theta=10^{-6}$, $V(G)=0$ pinned) and report the converged values and sweep count.

::: details Solution

```python
V = {"S": 0.0, "M": 0.0, "G": 0.0}
sweeps = 0
while True:
    old = V.copy()
    V["S"] = 0.8 * (-1 + old["M"]) + 0.2 * (-2 + old["S"])
    V["M"] = 0.8 * (-1 + old["G"]) + 0.2 * (-2 + old["S"])
    sweeps += 1
    if max(abs(V[s] - old[s]) for s in ("S", "M")) < 1e-6:
        break
print(sweeps, V)
# Expected: 23 sweeps, V['S'] ~= -3.375, V['M'] ~= -1.875
```

The loop contracts geometrically (dominant error eigenvalue about $0.51$ per sweep), so the residual decays to the fixed point $V(S)=-3.375$, $V(M)=-1.875$ of Exercise 3 within 23 sweeps at $\theta=10^{-6}$. Pinning $V(G)=0$ is essential; updating it would leak a spurious degree of freedom into both backups.
:::

### Exercise 8 (coding): Q-learning finds the corridor optimum

Implement tabular Q-learning on the corridor with $\epsilon$-greedy behavior ($\epsilon=0.2$, $\alpha=0.1$, $\gamma=1$, $Q$ initialized to zero, $Q(G,\cdot)=0$ fixed, episodes capped at $100$ steps). Train for $20{,}000$ episodes from $S$ and report the greedy policy and $V(S)=\max_aQ(S,a)$, $V(M)=\max_aQ(M,a)$. They should be near the optimal $(-2,-1)$, clearly better than the fixed-policy values $(-3.375,-1.875)$.

::: details Solution

```python
import random

def step(state, action):
    if state == "S":
        return ("M", -1) if action == "R" else ("S", -2)
    if state == "M":
        return ("G", -1) if action == "R" else ("S", -2)
    return ("G", 0)

random.seed(0)
Q = {"S": {"R": 0.0, "L": 0.0}, "M": {"R": 0.0, "L": 0.0}}
eps, alpha = 0.2, 0.1
for _ in range(20000):
    s = "S"
    for _ in range(100):
        a = random.choice(["R", "L"]) if random.random() < eps else max(Q[s], key=Q[s].get)
        ns, r = step(s, a)
        best_next = 0.0 if ns == "G" else max(Q[ns].values())
        target = r + best_next  # gamma = 1; terminal bootstraps zero
        Q[s][a] += alpha * (target - Q[s][a])
        s = ns
        if s == "G":
            break
greedy = {s: max(Q[s], key=Q[s].get) for s in ("S", "M")}
print(greedy, {s: max(Q[s].values()) for s in ("S", "M")})
# Expected: greedy {'S': 'R', 'M': 'R'}, values near {'S': -2.0, 'M': -1.0}
```

Why this works: behavior stays exploratory so both actions keep coverage, while the $\max$ target learns the greedy (optimal) values directly. The optimal policy walks right everywhere for returns $-2$ from $S$ and $-1$ from $M$, beating evaluation of the stochastic $(0.8, 0.2)$ policy. A SARSA agent with the same $\epsilon=0.2$ behavior would instead converge near the value of the $\epsilon$-greedy policy itself (worse than $(-2,-1)$), illustrating off-policy versus on-policy targets.
:::
