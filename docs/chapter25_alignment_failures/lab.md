---
title: '25.L Lab: Reward Hacking and Honest Evaluation'
---

# 25.L Lab: Reward Hacking and Honest Evaluation

> **Section goal**: reproduce the two central failure modes of the chapter in miniature -- first farm a loophole, then catch yourself misreading the numbers -- using a gridworld you can train on a laptop in minutes.
>
> **Code**: [gridworld.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter25_alignment_failures/gridworld.py) · [stats.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter25_alignment_failures/stats.py) · [run_specification_gaming.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter25_alignment_failures/run_specification_gaming.py) · [run_eval_methodology.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter25_alignment_failures/run_eval_methodology.py)

## What you will learn

- How to build an environment with a known reward loophole and watch tabular Q-learning exploit it instead of solving the task.
- How to fix the loophole with a one-time bonus plus potential-based shaping, and why the fix preserves the optimal policy.
- Why a training curve is not evidence: proxy return and task success must be logged as two separate curves.
- How to compare two hyperparameter settings honestly across 12 seeds: mean, median, interquartile mean (IQM), stratified-bootstrap confidence intervals, and performance profiles.
- How best-3-seeds and single-seed reporting can claim perfection (1.00 vs 1.00) and even crown the wrong winner on data where the honest analysis says the ranking is unresolved.

## Background

This lab exercises three sections of the chapter directly:

- [25.1 When Rewards Diverge from the Task](./classical-failures) defines the diagnostic used throughout Part 1: reward hacking is declared exactly when the proxy reward $R(\tau)$ rises while an independent task utility $U(\tau)$ stalls. The respawning bonus tile is a miniature CoastRunners setup from Section 25.1.2.
- [25.4 Defending Against Reward Loopholes](./scaling-and-defenses) motivates the fix: separate the training signal from independent measurement, and track proxy reward, independent utility, and behavior side by side (the "four gauges" of Section 25.4.2). Potential-based shaping follows Ng et al. (1999): adding $F(s,s') = \gamma\Phi(s') - \Phi(s)$ never changes the optimal policy.
- [25.5 Evaluation Protocols and Reproducibility](./rl-evaluation) motivates Part 2: one seed is not an evaluation. Report all seeds with uncertainty (Section 25.5.1, statistical rigor), and distrust headline numbers that hide prompt/seed sensitivity.

## Setup and how to run

You need the project virtualenv (Python 3.12, numpy, matplotlib, pytest); no GPU, no downloads. Everything runs on CPU with numpy only.

Smoke mode (each script finishes in well under a minute):

```bash
.venv/bin/python code/chapter25_alignment_failures/run_specification_gaming.py --smoke
.venv/bin/python code/chapter25_alignment_failures/run_eval_methodology.py --smoke
```

Full runs (the configuration behind every number quoted on this page):

```bash
.venv/bin/python code/chapter25_alignment_failures/run_specification_gaming.py
.venv/bin/python code/chapter25_alignment_failures/run_eval_methodology.py
```

Useful flags: `--seed` (base random seed; everything is reproducible), `--device` (accepted for uniformity; compute is numpy on CPU), `--episodes`, `--seeds`, `--eval-episodes`, `--slip` (Part 2 action noise, default 0.1), `--outdir`. Outputs (CSV metrics, JSON stats, PNG curves) land under `runs/chapter25_alignment_failures/`; each script also prints a summary table. Run the checks with:

```bash
.venv/bin/pytest code/chapter25_alignment_failures -q
```

## Walkthrough of the key code

**The loophole** lives in `LoopholeGridworld.step`. In `proxy` mode every visit to the bonus tile pays; in `fixed` mode the bonus pays once and a shaping term points toward the goal:

```python
if nxt == BONUS:
    if self.reward_mode == "proxy":
        reward += self.bonus_reward
    elif not self._bonus_used:
        reward += self.bonus_reward
        self._bonus_used = True
...
if self.reward_mode == "fixed" and not done:
    reward += self.gamma * self.potential(nxt) - self.potential(self._pos)
```

with `potential(pos) = -0.5 * manhattan(pos, GOAL)`. Because the added term has the potential-based form, the set of optimal policies is unchanged -- the fix only densifies the learning signal. `test_potential_shaping_preserves_optimal_policy` checks exactly this on a tiny 3-state MDP solved by exact value iteration.

**The learner** is ordinary tabular Q-learning with an epsilon-greedy behavior policy (`train_q_learning` in [gridworld.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter25_alignment_failures/gridworld.py)):

```python
target = reward if done else reward + gamma * float(np.max(q[ny, nx]))
q[y, x, action] += alpha * (target - q[y, x, action])
```

Nothing about this update knows the designer's intent -- it maximizes whatever scalar the environment returns, which is precisely the point.

**The honest statistics** are implemented from scratch in [stats.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter25_alignment_failures/stats.py). The IQM discards the top and bottom quartiles of seeds before averaging, so one lucky run cannot dominate the headline number:

```python
lo = n // 4
hi = n - n // 4
return float(np.mean(x[lo:hi]))
```

and `stratified_bootstrap_ci` resamples seeds _within_ each config (never mixing configs) to give percentile 95% intervals, including an interval for the IQM _difference_ -- the only number that can license a claim like "A beats B".

## Results from a real run

Part 1 defaults: 400 training episodes per reward mode, seed 0. "Train" columns are behavioral means over the final 10% of training episodes; "eval" columns are the greedy policy. Full run completed in seconds on CPU.

| mode  | train proxy | train success | eval proxy | eval success | bonus visits / episode |
| ----- | ----------- | ------------- | ---------- | ------------ | ---------------------- |
| proxy | 40.20       | 0.00          | 57.00      | 0.00         | 30.0                   |
| fixed | 14.22       | 1.00          | 14.28      | 1.00         | 1.0                    |

Read the proxy row as the chapter's false-positive archetype: a return of 57 -- the highest number in the table -- attached to a policy that reaches the goal 0% of the time and steps on the bonus 30 times per episode. The fixed row earns a _lower_ proxy return (14.28) with a 100% goal rate. Ranking by training reward alone picks the farmer.

Part 2 defaults: 12 seeds per config, 400 training episodes, slip 0.1, seed 0 (about 3 minutes on CPU). Metric is the goal-reaching success rate -- the independent utility, not the training objective.

| config                              | mean  | median | IQM   | 95% CI (IQM)   |
| ----------------------------------- | ----- | ------ | ----- | -------------- |
| A_stable (alpha=0.25, epsilon=0.15) | 0.688 | 0.855  | 0.817 | [0.450, 1.000] |
| B_noisy (alpha=0.40, epsilon=0.25)  | 0.553 | 0.500  | 0.577 | [0.235, 0.877] |

IQM difference (A − B): +0.240, 95% CI [−0.253, +0.642]. The interval covers zero, so the honest conclusion is that the ranking is **not resolved** -- report the tie, not a winner.

The same data under selective reporting:

| summary                                  | A_stable | B_noisy | apparent winner     |
| ---------------------------------------- | -------- | ------- | ------------------- |
| best-3 seeds                             | 1.000    | 1.000   | tie: "both perfect" |
| single seed (seed 0)                     | 1.000    | 1.000   | tie                 |
| adversarial pick (B's best vs A's worst) | 0.000    | 1.000   | B_noisy "wins"      |

The best-3 summary claims both configs are flawless, hiding that typical seeds score 0.50--0.86 and B's median seed fails half its episodes. The adversarial min/max pick goes further and manufactures an outright B victory (1.000 vs 0.000) from the same 24 runs. And the lottery is not hypothetical: 4 of the 12 aligned seed pairs crown a different winner than the full IQM analysis. Per-seed success rates (the arrival at these aggregates) were:

- A_stable: 1.00, 1.00, 1.00, 0.00, 1.00, 0.69, 0.28, 0.71, 1.00, 0.50, 0.07, 1.00
- B_noisy: 1.00, 0.02, 0.03, 0.50, 0.50, 0.13, 1.00, 1.00, 0.99, 0.22, 0.76, 0.49

## What to look for and why it happens

Watch the two Part 1 curves in `spec_gaming_curves.png` diverge: under the proxy reward the return curve climbs while the success curve flatlines at zero. That shape _is_ the Section 25.1 diagnostic -- no deception or intent is needed, only a gap between the scored number and the wanted behavior. The farming policy is optimal for the proxy MDP: with discount 0.99, collecting +2 every two steps near the start forever is worth far more than a one-time +10 minus travel costs, so Q-learning correctly solves the wrong problem.

In Part 2, notice how the performance profile (`eval_profile.png`) tells the story the means hide: A's curve sits above B's across most thresholds, yet the bootstrap interval for the difference still covers zero -- dominance on the plot is not the same as a statistically resolved ranking with 12 seeds. The mechanism behind the cherry-picking reversal is overlap plus selection: B's distribution is wider, so its extreme seeds (three perfect 1.00s) match or beat A's extremes even though B's typical seed is much worse (median 0.50 vs 0.855). Any top-k rule samples the extremes and therefore rewards variance, not typical performance.

## Exercises

1. **Price the loophole.** Re-run Part 1 with `--alpha 0.25` fixed and bonus payouts of 0.2, 0.5, and 1.0 (edit `bonus_reward`). At which value does the greedy policy switch from farming to goal-seeking, and does your threshold match a back-of-the-envelope return comparison (farming rate vs. one-time goal payout)? _Hint: compute undiscounted returns for "bounce on the bonus" vs. "go straight to the goal" first, then check the learned behavior._
2. **Remove the shaping, keep the one-time bonus.** Set `dist_scale=0.0` in the `fixed` environment and re-run Part 1 with 400 episodes. Does the agent still reach the goal? If not, raise `--episodes` until it does, and explain why sparse reward needs more samples. _Hint: compare which (state, action) pairs ever see a nonzero reward in the first 50 episodes with and without shaping._
3. **Grow the seed budget.** Re-run Part 2 with `--seeds 24` (takes roughly twice as long). Does the IQM-difference CI shrink enough to exclude zero? _Hint: bootstrap interval width shrinks roughly as $1/\sqrt{n}$; predict the new width before running._
4. **Add a third contender.** Add `C_cautious = {"alpha": 0.1, "epsilon": 0.05}` to `CONFIGS` in [run_eval_methodology.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter25_alignment_failures/run_eval_methodology.py) and re-run. Where does its performance profile sit, and what does low exploration cost under slip noise? _Hint: think about what happens to Q-values for states the behavior policy rarely visits._
5. **Rank by the training objective instead.** Change Part 2's comparison metric from `success` to `proxy` (one line in `main`) and re-run. Which config "wins" now, and what kind of policy does that crown -- a better solver or a better farmer? _Hint: re-read your Part 1 table before answering._
6. **Stop early on the independent metric.** Implement per-run early stopping in `train_q_learning`: every 50 episodes, evaluate the greedy policy, and stop a run once success exceeds 0.9 (Section 25.4.3, independent-metric early stopping). How many training episodes does each config save, and does stopping change the honest ranking? _Hint: log the stopping episode per seed and compare the distributions, not just the means._
