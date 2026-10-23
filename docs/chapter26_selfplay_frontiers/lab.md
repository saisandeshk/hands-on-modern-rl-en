---
title: '26.L Lab: Self-Play Dynamics from RPS to Kuhn Poker'
---

# 26.L Lab: Self-Play Dynamics from RPS to Kuhn Poker

## What you will learn

- Why naive self-play cycles: best responses and gradient steps chase the
  current opponent, so the current strategy keeps moving even though the
  time average settles on the Nash equilibrium.
- Why averaging fixes it: fictitious play and CFR's average strategy are the
  tabular ancestors of the historical opponent pools in
  [Section 26.1](/chapter26_selfplay_frontiers/self-play-outlook/).
- How to measure "distance to equilibrium" exactly: exploitability via
  best-response enumeration, in games small enough to solve by hand.
- How CFR learns to bluff and bluff-catch in Kuhn poker, and how the learned
  frequencies match the known equilibrium family.
- What self-play can and cannot do in tic-tac-toe: a weak opponent is beaten
  fast, but holding an optimal opponent to a draw takes a longer curriculum.

## Background

Section 26.1 defines a Nash equilibrium as a profile no one wants to
deviate from unilaterally:

$$
V(\pi^*, \pi^*) \ge V(\pi, \pi^*) \quad \forall \pi.
$$

It also warns that playing only the latest checkpoint can cycle: version
$N+1$ counters version $N$ and forgets how to handle version $N-1$. The
industrial fix is a pool of historical opponents, which is fictitious play
in spirit: respond to the average of past behavior, not to the last move.
This lab makes that contrast executable in three games of increasing
difficulty, and connects to the regret-minimization methods surveyed in
Section 26.1. The multi-agent credit-assignment view of the same ideas is
developed in [Section 26.3](/chapter26_selfplay_frontiers/llm-multi-agent-rl/);
the offline question of what fixed historical data can teach appears in the
[offline RL supplement](/chapter26_selfplay_frontiers/offline-rl/).

## Setup and how to run

The lab lives in `code/chapter26_selfplay_frontiers/` and needs only the
project virtualenv (numpy, matplotlib, torch, pytest). Every script takes
`--seed`, `--device {auto,cpu,cuda}`, and `--smoke` (a seconds-long
configuration), writes CSV/JSON metrics plus PNG curves under
`runs/chapter26_selfplay_frontiers/` (gitignored), and prints a summary
table. First check:

```bash
.venv/bin/python code/chapter26_selfplay_frontiers/rps_self_play.py --smoke
.venv/bin/python code/chapter26_selfplay_frontiers/kuhn_cfr.py --smoke
.venv/bin/python code/chapter26_selfplay_frontiers/tictactoe_self_play.py --smoke
.venv/bin/pytest code/chapter26_selfplay_frontiers -q
```

Full lab (about 20 seconds on a laptop CPU at seed 0):

```bash
.venv/bin/python code/chapter26_selfplay_frontiers/rps_self_play.py --variant standard --dynamics all --seed 0
.venv/bin/python code/chapter26_selfplay_frontiers/rps_self_play.py --variant biased --dynamics all --seed 0
.venv/bin/python code/chapter26_selfplay_frontiers/kuhn_cfr.py --iterations 20000 --seed 0
.venv/bin/python code/chapter26_selfplay_frontiers/tictactoe_self_play.py --episodes 30000 --seed 0
```

## Walkthrough of the key code

**RPS dynamics**
([rps_self_play.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter26_selfplay_frontiers/rps_self_play.py)).
Each rule is a few lines. Fictitious play keeps empirical counts and
best-responds to the opponent's average:

```python
pi[best_response_index(utility @ avg2[t - 1], rng)] = 1.0
```

while projected gradient ascent steps both strategies simultaneously:

```python
pi = project_simplex(pi + lr * (utility @ sigma))
sigma = project_simplex(sigma - lr * (utility.T @ pi))
```

Exploitability is the Nash gap,
`max_a (U sigma)_a - min_b (pi^T U)_b`, zero exactly at equilibrium.

**Kuhn CFR**
([kuhn_cfr.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter26_selfplay_frontiers/kuhn_cfr.py)).
Twelve information sets, full-tree traversal, regret matching for the
current strategy, reach-weighted averaging. The update is the whole
algorithm:

```python
advantage = action_utils - node_util
if player == 1:
    advantage = -advantage  # player 2 minimizes player 1's payoff
self.regret[infoset] += opp_reach * advantage
```

Exploitability is exact: enumerate all 64 pure strategies per player,
take the best response values, and halve their difference. No sampling,
no neural networks, nothing to tune.

**Tic-tac-toe TD**
([tictactoe_self_play.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter26_selfplay_frontiers/tictactoe_self_play.py)).
Tabular Q-learning for X (the Q-table is one torch tensor indexed by the
base-3 board code) against a mix of random O and self-play O that
minimizes X's own values. A memoized minimax solver provides both the
optimal yardstick and the "draw is the ceiling" certificate.

## The equilibria this lab checks

Standard RPS has value 0 and equilibrium (1/3, 1/3, 1/3). The biased
variant pays +2 for a Rock win; indifference at a symmetric profile
requires $-p_P + 2p_S = p_R - p_S = -2p_R + p_P$, giving
(1/4, 1/2, 1/4), still with value 0. Rock's extra power is cancelled by
playing extra Paper.

Kuhn poker's equilibrium family (bluff parameter $\alpha \in [0, 1/3]$)
is pinned down by five indifference conditions. Writing $x_J, x_K$ for
player 1's bet frequencies with J/K, $y_Q$ for player 1's call frequency
with Q facing a bet, $z_J$ for player 2's bluff frequency with J after a
check, and $w_Q$ for player 2's call frequency with Q facing a bet:

- Player 1's Jack is indifferent between bluffing and checking only if
  $w_Q = 1/3$ (bluffing wins $1 - 3w_Q$ against Q and $-2$ against K,
  versus $-1$ for checking).
- Player 1's King is indifferent only if $w_Q = z_J$ (betting earns
  $1 + w_Q/2$, checking earns $1 + z_J/2$).
- Player 1's Queen is indifferent between calling and folding only if
  $z_K = 3z_J$ (the posterior after a bet must be 1/4 Jack, 3/4 King).
- Player 2's Queen is indifferent only if $x_K = 3x_J$ (same 1-to-3
  posterior logic from the other side).
- Player 2's Jack is indifferent only if $x_K = 3y_Q - 1$: conditional on
  player 1 having checked, player 1 is Queen-heavy (Kings often bet
  first), so the bluff-catching point shifts up from 1/3.

Together: $x_J = \alpha$, $x_K = 3\alpha$, $y_Q = \alpha + 1/3$,
$z_J = w_Q = 1/3$, $z_K = 1$, with everything else pure (fold the
never-winning Jacks, never open-bet the Queen). The game value is
$-1/18$ for every $\alpha$. The two relations the lab checks are
$x_K = 3x_J$ and $y_Q = x_J + 1/3$.

## Results from a run you can reproduce

Seed 0, full configuration (see commands above).

**RPS, standard game.** Naive rules cycle forever; their time averages
converge:

| Dynamics                    | Current exploitability | Average exploitability | L1 of average to (1/3, 1/3, 1/3) |
| --------------------------- | ---------------------- | ---------------------- | -------------------------------- |
| Best response               | 2.0000                 | 0.0001                 | 0.0001                           |
| Projected gradient (lr 0.1) | 0.8947                 | 0.0001                 | 0.0000                           |
| Fictitious play             | 2.0000                 | 0.0130                 | ~0.008                           |

**RPS, biased game** (equilibrium (1/4, 1/2, 1/4)). Gradient and
fictitious-play averages converge (average exploitability 0.0003 and
0.0146). Pure best-response cycling instead averages to (1/3, 1/3, 1/3)
in both variants, which is exact Nash only in the symmetric game: in the
biased game its average stays 0.6667-exploitable. Myopic play does not
even average correctly once symmetry breaks.

**Kuhn CFR, 20000 iterations.** Exploitability of the average strategy
falls from 0.0119 (iteration 200) to 0.0016 (iteration 20000, NashConv
0.0032); the game value settles at $-0.055555$ (true $-1/18$). Learned
key frequencies: player 1 bets J 0.2211 and K 0.6605 (ratio 2.987,
theory 3.0); player 1 calls with Q 0.5613 (theory $\alpha + 1/3 = 0.5545$);
player 2 bluffs J 0.3325 and calls with Q 0.3411 (theory 1/3 each).
Per-checkpoint exploitability can wiggle (0.0015 at iteration 10000,
0.0024 at 10200), but the trend is monotone on a log scale.

**Tic-tac-toe, 30000 episodes.** X beats random play quickly and holds
the optimal player late:

| Episodes | Win vs random | Draw vs random | Loss vs random | Vs minimax |
| -------- | ------------- | -------------- | -------------- | ---------- |
| 1500     | 0.895         | 0.040          | 0.065          | O wins     |
| 6000     | 0.960         | 0.020          | 0.020          | draw       |
| 15000    | 0.975         | 0.015          | 0.010          | O wins     |
| 30000    | 0.975         | 0.015          | 0.010          | draw       |

The random opponent is essentially solved after a few thousand episodes;
a draw against minimax (the ceiling for X) arrives around episode 6000
and becomes stable from about episode 18000, with occasional
single-checkpoint relapses while epsilon-greedy exploration is still
active.

## What to look for and why it happens

- **Last iterate vs time average.** In every RPS plot the "current
  profile" curve stays high while the "time average" curve falls. The
  learner visits good strategies without staying at any of them. This is
  exactly the Section 26.1 argument for opponent pools: deploy the
  average (the pool), not the last checkpoint.
- **Symmetry breaking.** Biased RPS shows that naive cycling can average
  to a non-equilibrium (uniform play stays exploitable). Only dynamics
  that track history (fictitious play) or average it (CFR-style
  averaging) adapt to the shifted equilibrium.
- **Regret follows the money.** In Kuhn poker the Q-call coordinate moves
  slowly because it sits on a near-indifference ridge: deviating gains
  almost nothing per visit, so regret accumulates slowly there while
  exploitability still collapses. Judge CFR by exploitability, not by any
  single frequency.
- **Weak opponents teach fast, strong opponents certify.** Tic-tac-toe
  against random play looks solved early, yet minimax still wins some
  checkpoints. Fixed weak opponents measure progress; only a strong or
  adaptive opponent measures robustness.

## Exercises with hints

1. **Sweep the bias.** Change the Rock-win payoff from +2 to +1.5 and
   +3, re-derive the symmetric equilibrium by hand, and check fictitious
   play finds it. Hint: redo the two indifference equations from the
   equilibria section; only the $p_R$ equation changes.
2. **Break the gradient cycle (or fail trying).** Try learning rates 0.01
   to 1.0 and decaying step sizes in `run_gradient`. Does the last
   iterate ever converge? Hint: track the orbit radius (distance of the
   current profile to Nash) rather than exploitability; compare with the
   time average at each setting.
3. **CFR scaling law.** Run 1000 to 100000 iterations, plot final
   exploitability versus iterations on a log-log axis, and fit the
   slope. Hint: vanilla CFR theory predicts roughly $1/\sqrt{T}$ for the
   average profile; check whether the measured slope matches and where it
   bends.
4. **How weak an opponent suffices?** Train with `--opp-random 1.0`
   (pure random opponent) and compare against the default mix: episodes
   until the first draw against minimax, and the final loss rate against
   random. Hint: a random opponent plus epsilon-greedy exploration covers
   this tiny state space, so both curricula reach a draw; explain why,
   then describe a larger game where training only against random play
   would leave whole regions of the state space unvisited.
5. **Alternating vs simultaneous updates.** Change `run_best_response` so
   players move alternately (player 1 responds to the new player 2 each
   step) and compare the cycle length and the average in both RPS
   variants. Hint: write down the 2-step transition on pure profiles
   before running.
6. **Sample the chance nodes.** Replace full-tree traversal in
   `kuhn_cfr.py` with sampling one of the six deals per iteration
   (chance-sampled CFR) and compare wall-clock time to reach
   exploitability 0.01. Hint: keep the average-strategy weighting by the
   player's own sampled reach probability; average over seeds because
   sampling adds variance.
