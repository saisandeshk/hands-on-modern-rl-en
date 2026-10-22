---
title: 4.L Lab: Tabular Methods Against Exact Oracles
---

# 4.L Lab: Tabular Methods Against Exact Oracles

## What You Will Learn

- How to read the exact transition model out of a gymnasium toy environment and turn it into a dynamic-programming oracle.
- Why iterative policy evaluation, policy iteration, and value iteration agree with each other, and what the residual stopping rule buys you.
- How Monte Carlo control, SARSA, Expected SARSA, and Q-learning differ in their update targets, and how to check sampling-based methods against the exact oracle.
- Why SARSA learns the safer CliffWalking path while Q-learning learns the optimal cliff-edge path, and why "better final policy" does not mean "better behaviour during training".
- How ordinary and weighted importance sampling trade bias against variance in off-policy evaluation.

## Background

This lab is the hands-on companion to the three theory sections of Chapter 4:

- [4.1 Dynamic Programming, Monte Carlo and TD](./dp-mc-td) — the three ways of editing a value table, and the corridor worked example. The lab replays the same story on FrozenLake: DP expands the model, MC waits for full returns, TD bootstraps after one step.
- [4.2 Policy Sampling and Data Sources](./algorithm-taxonomy) — behaviour policy vs. target policy, on-policy vs. off-policy. SARSA is on-policy (it evaluates the policy it executes, wobble included); Q-learning is off-policy (it aims at the greedy policy while behaving epsilon-greedily); the importance-sampling experiment makes the behaviour/target split quantitative.
- [4.3 Reward Function Design](./reward-design) — CliffWalking's cliff penalty of $-100$ versus $-1$ per step is a small case study in how reward magnitudes shape the learned path.

Notation matches the chapter: states $s$, actions $a$, rewards $r$, discount $\gamma$, values $V^\pi, Q^\pi$, returns $G_t$, importance ratio $\rho$.

## Setup and How to Run

You need the project virtualenv (Python 3.12, numpy, gymnasium, matplotlib, pytest). Everything runs offline on a laptop CPU. All commands below run from the repository root.

Smoke test (under ~60 s, verifies the whole pipeline):

```bash
.venv/bin/pytest code/chapter04_classical_methods -q
```

Full lab (about ten minutes on CPU):

```bash
.venv/bin/python code/chapter04_classical_methods/dp_oracle.py
.venv/bin/python code/chapter04_classical_methods/mc_td_control.py
.venv/bin/python code/chapter04_classical_methods/cliffwalking.py
.venv/bin/python code/chapter04_classical_methods/ope_is.py
```

Each script accepts `--seed`, `--device {auto,cpu,cuda}`, `--smoke`, and `--outdir` (default `runs/chapter04_classical_methods/`, gitignored). Every script prints a summary table to stdout and writes metrics (JSON/CSV) plus PNG plots under the output directory. The lab code is NumPy-only and implemented from scratch — no RL libraries:

- [common.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter04_classical_methods/common.py) — model loading from `env.unwrapped.P`, epsilon-greedy action selection, greedy evaluation, plotting helpers.
- [dp_oracle.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter04_classical_methods/dp_oracle.py) — the three DP algorithms.
- [mc_td_control.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter04_classical_methods/mc_td_control.py) — MC, SARSA, Expected SARSA, Q-learning.
- [cliffwalking.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter04_classical_methods/cliffwalking.py) — the SARSA/Q-learning cliff comparison.
- [ope_is.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter04_classical_methods/ope_is.py) — ordinary and weighted importance sampling.
- [test_smoke.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter04_classical_methods/test_smoke.py) — unit tests plus per-script smoke tests.

## Walkthrough of the Key Code

**Reading the exact model.** Both environments expose their dynamics as nested dictionaries, so the oracle needs no sampling at all:

```python
raw_P = env.unwrapped.P  # P[s][a] = list of (prob, next_state, reward, done)
```

**One backup, three algorithms.** Policy evaluation averages over the policy, value iteration maximises over actions — compare the inner loops of `dp_oracle.py`:

```python
# iterative policy evaluation: weight branches by pi(a|s)
new += policy[s, a] * action_value
# value iteration: assume the greedy action from now on
V_new = Q.max(axis=1)
```

Both stop on the same residual rule: halt when the largest single-state change `max|V_new - V|` drops below `theta`.

**One line separates SARSA from Q-learning.** In `mc_td_control.py` the updates differ only in the bootstrap target:

```python
target = reward + gamma * Q[ns, na]        # SARSA: next behaviour action
target = reward + gamma * float(Q[ns].max())  # Q-learning: greedy next action
```

Expected SARSA sits between them: it takes the exact expectation of the next value under the epsilon-greedy policy instead of sampling one next action. First-visit MC, by contrast, waits for the episode to end and updates toward the full return `G` with an incremental mean `Q += (G - Q) / N(s, a)`.

**Importance weights are exact here.** Because the behaviour policy is epsilon-greedy with a known epsilon, each step ratio `pi(a|s) / mu(a|s)` is computed exactly, and the trajectory weight is their product:

```python
rho *= pi / mu  # ordinary IS: mean(rho * G); weighted IS: sum(rho * G) / sum(rho)
```

## Results

All numbers below come from the full run (`--seed 0`, CPU). Re-run the scripts to reproduce them; the plots land in `runs/chapter04_classical_methods/`.

### Experiment 1: the three DP methods agree

On slippery 4x4 FrozenLake with $\gamma = 1$ and stopping threshold $\theta = 10^{-6}$:

| Method | Sweeps / work | Bellman residual | $V^\star(s_0)$ |
|---|---|---|---|
| Policy evaluation (uniform $\pi$) | 40 sweeps | — | — |
| Policy iteration | 617 eval sweeps, 2 improvements | $6.56 \times 10^{-7}$ | 0.823506 |
| Value iteration | 431 sweeps | $9.61 \times 10^{-7}$ | 0.823506 |

Policy iteration and value iteration agree to $1.12 \times 10^{-5}$ in max norm. The start-state value $0.8235$ equals the known textbook answer $14/17$: the optimal policy reaches the goal with probability $14/17$. Its $Q^\star$ table is saved as the exact oracle for the next experiments.

### Experiment 2: MC and TD control against the oracle

Four methods, epsilon-greedy behaviour ($\epsilon = 0.1$), $\alpha = 0.1$, 10 seeds $\times$ 20,000 episodes on slippery FrozenLake:

| Method | Final greedy return | Final max $\|Q - Q^\star\|$ | Online return (last 10%) |
|---|---|---|---|
| MC first-visit | RESULT | RESULT | RESULT |
| SARSA | RESULT | RESULT | RESULT |
| Expected SARSA | RESULT | RESULT | RESULT |
| Q-learning | RESULT | RESULT | RESULT |

<!-- RESULTS-TODO: fill from control_summary.json -->

The learning-curve plot shows mean $\pm$ std over the 10 seeds: greedy-policy return rises toward the optimal $0.82$ success rate while the max-norm oracle error falls on a log scale.

### Experiment 3: SARSA takes the safe path, Q-learning the optimal one

On CliffWalking ($\alpha = 0.5$, $\epsilon = 0.1$, 10 seeds $\times$ 500 episodes):

| Method | Online return (behaviour) | Greedy-policy return | Falls per episode |
|---|---|---|---|
| SARSA | $-35.90 \pm 67.94$ | $-53.60 \pm 73.20$ | 0.102 |
| Q-learning | $-56.95 \pm 90.65$ | $-13.00 \pm 0.00$ | 0.365 |

Q-learning reaches the optimal greedy return of $-13$ (13 steps along the cliff edge) on all 10 seeds, but pays for it during training with roughly $3.5\times$ more cliff falls and a worse online return. SARSA's values include the cost of its own exploration wobble, so it prefers the safer path away from the edge — better behaviour, worse final greedy policy. One seed's greedy maps:

```text
SARSA greedy path:      Q-learning greedy path:
>>>>>>>>>>vv            >>>>>>>>vv>v
>^^>>^^^^^>v            >>>>>>>>vv>v
^^^<^^^^^^>v            >>>>>>>>>>>v
SCCCCCCCCCCG            SCCCCCCCCCCG
```

The top row of the SARSA map and the bottom row of the Q-learning map tell the story: SARSA travels high, Q-learning hugs the cliff.

### Experiment 4: ordinary vs. weighted importance sampling

Target policy: the DP-optimal greedy policy on FrozenLake (exact $V^\pi(s_0) = 0.823529$). Behaviour policy: $\epsilon$-greedy around $Q^\star$ with $\epsilon = 0.1$. RESULT trials $\times$ RESULT behaviour episodes each:

| Estimator | Mean | Std | Bias | RMSE |
|---|---|---|---|---|
| Ordinary IS | RESULT | RESULT | RESULT | RESULT |
| Weighted IS | RESULT | RESULT | RESULT | RESULT |

<!-- RESULTS-TODO: fill from ope_summary.json -->

## What to Look For and Why It Happens

- **DP agreement is exactness, not luck.** All three DP methods solve the same Bellman equations from the same model; the residual rule just decides when the digits stop changing. Policy iteration needs only 2 improvements here because FrozenLake's optimal policy is found almost immediately — most of its sweeps go into evaluating intermediate policies, which is why its total sweep count exceeds value iteration's.
- **The oracle error separates "still learning" from "unlucky".** A flat greedy-return curve could mean the policy stopped improving or that evaluation is noisy. The max-norm error to $Q^\star$ has no sampling noise on one side of the comparison, so a shrinking error means the table itself is converging.
- **On-policy vs. off-policy is visible in the cliff paths.** SARSA's backup uses the next behaviour action, which near the cliff is sometimes a random step into $-100$. Its action values therefore price in the wobble, and the greedy policy extracted from them keeps its distance. Q-learning's backup uses the greedy next action, so its values describe the optimal policy even while its behaviour keeps falling in — better destination, bumpier road.
- **Importance weights collapse over long horizons.** The behaviour policy matches the deterministic target at each step with probability around $0.925$, so a 30-step episode matches fully with probability $0.925^{30} \approx 0.1$. Most episodes get weight exactly zero; a few get huge weights. That is why ordinary IS is unbiased but jumps around (variance), while self-normalising (weighted IS) calms the estimates down at the price of finite-sample bias. Watch the mean effective-sample-size fraction: only a few percent of each batch effectively counts.

## Extension Exercises

1. **Stopping threshold vs. sweep count.** Re-run `dp_oracle.py` with `--theta 1e-2` down to `--theta 1e-10` and plot sweeps against the final Bellman residual. *Hint: value iteration's error shrinks roughly geometrically; the iteration count grows only logarithmically in $1/\theta$.*
2. **Step-size sensitivity.** Train Q-learning on FrozenLake with `--alpha 0.01, 0.1, 0.5` and compare the oracle-error curves. *Hint: small $\alpha$ learns slowly but steadily; large $\alpha$ bounces around the optimum — the max-norm error makes the bouncing visible.*
3. **First-visit vs. every-visit MC.** Modify `train_mc` to update every visit instead of the first visit and compare. *Hint: the two estimate slightly different quantities on continuing revisits; on this episodic task they should nearly coincide — check whether they do.*
4. **Break the OPE demo.** Raise `--eps-behavior` to 0.3 and 0.5 and watch the ESS fraction and the RMSE of both estimators. *Hint: at some point no episode in a batch matches the target at all, and weighted IS returns exactly 0 — the guard in `weighted_is` exists for this case.*
5. **Deterministic FrozenLake.** Gymnasium's `FrozenLake-v1` is slippery; create it with `is_slippery=False`, re-run DP and Q-learning, and compare iteration counts and learning speed. *Hint: the model becomes deterministic, so DP converges in a handful of sweeps and TD needs far fewer samples.*
6. **Evaluate a uniform policy by OPE.** Change the target to the uniform-random policy (exact value from `iterative_policy_evaluation`) with behaviour $\epsilon$-greedy around it, and repeat Experiment 4. *Hint: behaviour and target now overlap heavily, so ESS stays high — the bias/variance gap between the estimators should nearly vanish.*
