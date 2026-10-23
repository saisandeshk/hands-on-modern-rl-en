# Chapter 25 Lab: Reward Hacking and Honest Evaluation

Two hands-on parts for Chapter 25 (_Reward hacking and RL evaluation_):

1. **Specification gaming** (`run_specification_gaming.py`): a small gridworld
   with a respawning bonus tile next to the start (the CoastRunners-style
   loophole of Section 25.1). Tabular Q-learning under the proxy reward
   farms the bonus and never reaches the goal; after the fix (one-time bonus
   - potential-based shaping) the same agent reaches the goal reliably.
2. **Honest evaluation** (`run_eval_methodology.py`): the same tabular agent
   trained with 12 seeds under two hyperparameter settings. Reports mean,
   median, IQM, stratified-bootstrap 95% confidence intervals and a
   performance profile (all implemented from scratch in `stats.py`), and
   shows how best-3-seeds and single-seed summaries reverse the honest
   conclusion (Section 25.5).

## What the lab shows

- Proxy return and task success can diverge completely: the hacking policy
  scores ~57 per episode with a 0% goal rate, while the repaired policy
  scores ~14 with a 100% goal rate.
- With 12 seeds per config, the IQM gap (+0.24 for `A_stable`) has a
  bootstrap 95% CI of [-0.25, +0.64]: the ranking is _not_ resolved. The
  best-3 summary instead claims both configs are perfect (1.00 vs 1.00),
  and 4 of 12 single-seed comparisons crown the wrong winner.

## How to run it

Smoke mode (each script finishes in well under 60 s on CPU):

```bash
.venv/bin/python code/chapter25_alignment_failures/run_specification_gaming.py --smoke
.venv/bin/python code/chapter25_alignment_failures/run_eval_methodology.py --smoke
```

Full runs (the configuration behind the numbers in the docs page):

```bash
.venv/bin/python code/chapter25_alignment_failures/run_specification_gaming.py
.venv/bin/python code/chapter25_alignment_failures/run_eval_methodology.py
```

Common options: `--seed` (base random seed), `--device` (accepted for
uniformity; all compute here is numpy on CPU), `--episodes`, `--seeds`,
`--eval-episodes`, `--slip` (Part 2 action noise), `--outdir`.

Outputs (CSV metrics, JSON stats, PNG curves) go to
`runs/chapter25_alignment_failures/` (gitignored); a short summary table is
printed to stdout. `matplotlib` uses the non-interactive `Agg` backend, so
no display is needed.

## Expected runtime and results

On a laptop CPU: Part 1 full run ~10 s; Part 2 full run (24 trainings of
400 episodes + 2000 bootstrap resamples) ~2-4 min. Expected full-run
results (seed 0, deterministic):

| Part            | Result                                                                                   |
| --------------- | ---------------------------------------------------------------------------------------- |
| 1 proxy         | train proxy ~40, success 0.00; greedy proxy 57.0, success 0.00, ~30 bonus visits/episode |
| 1 fixed         | train proxy ~14, success 1.00; greedy proxy 14.28, success 1.00                          |
| 2 honest        | IQM A 0.817 [0.450, 1.000] vs B 0.577 [0.235, 0.877]; diff CI includes zero              |
| 2 cherry-picked | best-3: 1.000 vs 1.000; 4/12 seed pairs disagree with IQM                                |

## Tests

```bash
.venv/bin/pytest code/chapter25_alignment_failures -q
```

`test_smoke.py` runs both scripts in smoke mode as subprocesses (with a
timeout) and checks their artefacts, plus unit tests: IQM against a
hand-computed trim-then-average, bootstrap CIs on fixed arrays (degenerate
constant array, coverage of the sample mean), the performance profile
against exact exceedance fractions, bonus respawn/fix semantics, goal
termination, and potential-based shaping preserving the optimal policy on a
tiny 3-state MDP solved exactly by value iteration.

## Files

| File                          | Purpose                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------- |
| `gridworld.py`                | Loophole gridworld (`proxy` / `fixed` rewards), tabular Q-learning, greedy evaluation       |
| `stats.py`                    | Mean, median, IQM, stratified-bootstrap CIs, performance profiles from scratch (numpy only) |
| `run_specification_gaming.py` | Part 1: train under both rewards, compare proxy vs. success, save curves                    |
| `run_eval_methodology.py`     | Part 2: 12-seed comparison of two settings with robust statistics                           |
| `test_smoke.py`               | Subprocess smoke tests + exact unit tests                                                   |
