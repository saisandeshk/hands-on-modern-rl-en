# Chapter 4 Lab: Tabular Methods Against Exact Oracles

Hands-on lab for Chapter 4 (classical RL: DP, MC, TD). The lab uses two
tabular environments — slippery 4x4 `FrozenLake-v1` and `CliffWalking-v1`
(`-v0` on older gymnasium) — and reads the exact transition model from
`env.unwrapped.P`, so dynamic programming gives an **exact oracle** that
every sampling-based method is measured against.

What the lab shows:

1. **DP oracle** (`dp_oracle.py`): iterative policy evaluation, policy
   iteration, and value iteration with a max-norm residual stopping rule.
   All three agree; the optimal values (`V*(s0) = 14/17`) become the `Q*`
   oracle for the rest of the lab.
2. **MC/TD control** (`mc_td_control.py`): first-visit MC control,
   SARSA, Expected SARSA, and Q-learning with epsilon-greedy behaviour.
   Plots show learning curves (mean over 10 seeds, shaded std band) and
   the max-norm error `||Q - Q*||_inf` to the oracle.
3. **CliffWalking** (`cliffwalking.py`): reproduces the classic
   Sutton–Barto result — SARSA learns the safer path (better online
   return, fewer falls) while Q-learning learns the optimal cliff-edge
   path (greedy return `-13` on all seeds, worse online return).
4. **Off-policy evaluation** (`ope_is.py`): ordinary vs. weighted
   importance sampling of the optimal greedy policy from epsilon-greedy
   behaviour episodes, compared against the exact DP value (bias,
   variance, RMSE, effective sample size).

Shared helpers live in `tabular_common.py`; `test_ch04_lab.py` holds unit tests
(Bellman residual ~0, hand-computed SARSA/Q-learning/Expected-SARSA
updates, IS estimators on a two-state toy with known answers) plus a
smoke test that runs every script with `--smoke`.

## How to run

From the repository root:

```bash
# quick check: unit tests + every script in smoke mode (< ~60 s on CPU)
.venv/bin/pytest code/chapter04_classical_methods -q

# full lab (about ten minutes on CPU; each script prints a summary table)
.venv/bin/python code/chapter04_classical_methods/dp_oracle.py
.venv/bin/python code/chapter04_classical_methods/mc_td_control.py
.venv/bin/python code/chapter04_classical_methods/cliffwalking.py
.venv/bin/python code/chapter04_classical_methods/ope_is.py
```

Common flags: `--seed`, `--device {auto,cpu,cuda}`, `--smoke` (tiny
config for a <60 s CPU check), `--outdir` (default
`runs/chapter04_classical_methods/`, gitignored). Script-specific flags
are listed with `--help` (e.g. `--n-episodes`, `--n-seeds`,
`--alpha`, `--epsilon`, `--theta`, `--algos`).

## Expected runtime

| Script             | Full config                      | Approx. CPU time (measured, `--seed 0`) |
| ------------------ | -------------------------------- | --------------------------------------- |
| `dp_oracle.py`     | 16-state model, theta=1e-6       | < 1 s                                   |
| `mc_td_control.py` | 4 algos x 10 seeds x 20,000 eps  | ~90 s                                   |
| `cliffwalking.py`  | 2 algos x 10 seeds x 500 eps     | ~5 s                                    |
| `ope_is.py`        | 100 trials x 2,000 behaviour eps | ~35 s                                   |

## Expected results (full run, `--seed 0`, CPU)

DP on slippery FrozenLake (gamma=1, theta=1e-6): policy evaluation of the
uniform policy takes 40 sweeps; policy iteration takes 617 eval sweeps
with 2 improvements (residual 6.6e-7); value iteration takes 431 sweeps
(residual 9.6e-7). PI and VI agree to ~1e-5; `V*(s0) = 0.8235 = 14/17`.

MC/TD control (10 seeds x 20,000 episodes, eps=0.1; alpha=0.1 for the
three TD methods, MC uses the incremental 1/N mean instead of a step
size). Measured full-run numbers (`--seed 0`, CPU):

| Method         | Final greedy return | Final max oracle error | Online (last 10%) |
| -------------- | ------------------- | ---------------------- | ----------------- |
| MC first-visit | 0.1400 +/- 0.118    | 0.8516                 | 0.0734            |
| SARSA          | 0.3800 +/- 0.384    | 0.7957                 | 0.1932            |
| Expected SARSA | 0.3800 +/- 0.384    | 0.8486                 | 0.1960            |
| Q-learning     | 0.2300 +/- 0.353    | 0.7071                 | 0.1337            |

CliffWalking (10 seeds x 500 episodes, alpha=0.5, eps=0.1): SARSA online
return -38.56 (0.108 falls/ep) with greedy return -35.50 (safer path);
Q-learning online return -56.62 (0.362 falls/ep) with greedy return
-13.00 on all 10 seeds (optimal cliff-edge path).

OPE of the optimal greedy policy (exact V(s0)=0.823529) from
eps=0.1 behaviour data (100 trials x 2,000 episodes). Measured full-run
numbers (`--seed 0`, CPU):

| Estimator   | Mean   | Std    | Bias    | RMSE   |
| ----------- | ------ | ------ | ------- | ------ |
| Ordinary IS | 0.7722 | 0.2437 | -0.0513 | 0.2491 |
| Weighted IS | 0.7949 | 0.1458 | -0.0286 | 0.1486 |

Mean effective-sample-size fraction: 0.0226.

Ordinary IS is unbiased for full episodes but shows a small downward
bias here (-0.05) because gymnasium truncates episodes at 100 steps
without bootstrapping the missing tail; weighted IS has smaller
variance at the cost of finite-sample bias (most trajectory weights
are exactly zero; only a few percent of each batch effectively counts).
