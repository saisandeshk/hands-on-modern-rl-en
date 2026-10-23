# Chapter 26 lab: self-play dynamics (RPS, Kuhn poker, tic-tac-toe)

Hands-on companion to Chapter 26 ("Self-play, scaling and research
frontiers"), Section 26.1. Three scripts, one idea: when the opponent is
another copy of the learner, myopic updates cycle, while averaging over
history (fictitious play, opponent pools, CFR's average strategy)
converges toward Nash equilibrium.

## What each script shows

| Script                   | What it does                                                                                                | What to look for                                                                                                                                              |
| ------------------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rps_self_play.py`       | Best-response, projected gradient, and fictitious play self-play in standard and biased rock-paper-scissors | Naive updates cycle (current exploitability stays high); time averages converge to (1/3, 1/3, 1/3), resp. (1/4, 1/2, 1/4)                                     |
| `kuhn_cfr.py`            | Vanilla CFR from scratch in Kuhn poker, exact exploitability by best-response enumeration                   | Exploitability of the average strategy falls toward 0; learned frequencies match the equilibrium family (K value-bets 3x the J bluff; Q-calls at alpha + 1/3) |
| `tictactoe_self_play.py` | Tabular Q-learning X vs a mix of random and self-play O, evaluated vs random and minimax                    | Random is beaten fast (~95%+ wins); holding the optimal minimax player to a draw takes longer self-play                                                       |

## How to run

Smoke mode (each script finishes in seconds, for a first check):

```bash
.venv/bin/python code/chapter26_selfplay_frontiers/rps_self_play.py --smoke
.venv/bin/python code/chapter26_selfplay_frontiers/kuhn_cfr.py --smoke
.venv/bin/python code/chapter26_selfplay_frontiers/tictactoe_self_play.py --smoke
```

Full lab (everything, a few minutes on a laptop CPU):

```bash
.venv/bin/python code/chapter26_selfplay_frontiers/rps_self_play.py --variant standard --dynamics all --seed 0
.venv/bin/python code/chapter26_selfplay_frontiers/rps_self_play.py --variant biased --dynamics all --seed 0
.venv/bin/python code/chapter26_selfplay_frontiers/kuhn_cfr.py --iterations 20000 --seed 0
.venv/bin/python code/chapter26_selfplay_frontiers/tictactoe_self_play.py --episodes 30000 --seed 0
```

Tests (unit checks plus every script in smoke mode):

```bash
.venv/bin/pytest code/chapter26_selfplay_frontiers -q
```

Common flags: `--seed` (reproducibility), `--device {auto,cpu,cuda}`
(torch device used for a metric cross-check; compute itself is tiny),
`--smoke` (tiny configuration), `--outdir` (default
`runs/chapter26_selfplay_frontiers/<part>/`, gitignored). All plots use
matplotlib's `Agg` backend; each script prints a summary table and writes
CSV metrics, a JSON summary, and PNG curves.

## Expected runtime and results (seed 0, CPU)

| Script (full config)                    | Runtime (laptop CPU) | Headline result                                                                                                                          |
| --------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| RPS, standard, 20000 iters x 3 dynamics | ~5 s                 | current expl: BR 2.00, gradient 0.89 (both cycle); average expl: ~0.0001 (BR/gradient), 0.013 (FP)                                       |
| RPS, biased, 20000 iters x 3 dynamics   | ~5 s                 | averages converge near (0.25, 0.50, 0.25); gradient current expl 0.88, average expl 0.0003                                               |
| Kuhn CFR, 20000 iterations              | ~5 s                 | final exploitability 0.0016, game value -0.0556 (true -1/18); P1 bets J 0.22 / K 0.66 (ratio 2.99); P1 calls Q 0.56 (alpha + 1/3 = 0.55) |
| Tic-tac-toe, 30000 episodes             | ~5 s                 | X wins ~97.5% vs random; holds minimax to a draw from ~18000 episodes on                                                                 |

## Requirements

See `requirements.txt` (numpy, matplotlib, torch, pytest); all are part of
the project virtualenv, so nothing needs installing.
