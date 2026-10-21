# Report: ch26-lab (self-play dynamics lab)

## Summary

Built the Chapter 26 hands-on lab: `code/chapter26_selfplay_frontiers/`
with three scripts (`rps_self_play.py`, `kuhn_cfr.py`,
`tictactoe_self_play.py`), `test_smoke.py` (10 tests), `README.md`,
`requirements.txt`, plus the docs page
`docs/chapter26_selfplay_frontiers/lab.md`. Part 1 contrasts cycling
(best response, projected gradient) vs converging time-averages
(fictitious play) in standard and biased RPS. Part 2 implements vanilla
CFR from scratch in Kuhn poker with exact exploitability. Part 3 is
tabular self-play Q-learning on tic-tac-toe evaluated vs random and
minimax opponents. All tests pass; `npx vitepress build docs` succeeds.

## Technical fixes

No existing files were edited (scope forbids it), but two real bugs in
new code were found by testing and fixed, plus one wrong theory claim:

1. **CFR regret sign for player 2** (`kuhn_cfr.py`, `_traverse`). The
   traversal returns player-1-centric utilities, so the first version
   added `opp_reach * (u(a) - u(node))` for both players, i.e. player 2
   maximized player 1's payoff. Smoke run showed exploitability stuck at
   0.83 with game value +1.33. Fix: negate the advantage for player 1's
   opponent. Verified: exploitability 0.0016 and value -0.055555
   (-1/18) at 20000 iterations.
2. **History-string collision** (`kuhn_cfr.py`): call action `"k"`
   produced terminal histories (`cbk`) mismatching `payoff`/`is_terminal`
   (`cbc`). Fix: call is `"c"` (position disambiguates from check),
   matching standard Kuhn notation. Verified by payoff unit tests.
3. **Wrong equilibrium claim, caught by my own test.** I first wrote
   P1's Q-call frequency as 1/3; the exact-exploitability test returned
   0.033 instead of ~0. Re-derivation from scratch showed the correct
   family has `y_Q = alpha + 1/3` (the posterior after a check is
   Queen-heavy since Kings often bet first: indifference needs
   `x_K = 3*y_Q - 1`, and `x_K = 3*x_J` forces `y_Q = x_J + 1/3`).
   CFR independently confirmed it (learned 0.5613 vs predicted 0.5545).
   Fixed `equilibrium_profile`, the test now asserts `expl < 1e-9` and
   passes, and the lab page carries the full five-condition derivation.
4. **Exercise premise falsified.** Draft exercise 4 claimed training vs
   pure random leaves lines minimax exploits; a control run
   (`--opp-random 1.0`) also draws minimax. Rewrote the exercise to ask
   for a speed comparison plus why a tiny state space forgives weak
   opponents.

## Other changes

- `rps_self_play.py`: gradient dynamics starts far from equilibrium so
  the orbit is visible (near-uniform starts cycle with tiny amplitude).
- `test_smoke.py`: subprocess smoke tests pass `--device cpu` per the
  "tests run on CPU" requirement.
- Prettier applied to the new markdown files (repo prettier has no
  Python parser; `.py` files follow the existing textbook style).

## Needs external verification

None. All game-theoretic claims were re-derived by hand and confirmed
numerically (Nash profiles to exploitability ~0, CFR value to -1/18,
ratio relations). No papers, benchmarks, or URLs were invented; code
links use the standard repo path pattern.

## Sidebar additions

- Title: `26.L Lab: Self-Play Dynamics from RPS to Kuhn Poker`
- Link: `/chapter26_selfplay_frontiers/lab`
- (Could not edit `docs/.vitepress/config.mjs` per scope rules.)

## Commands run and results

- `.venv/bin/pytest code/chapter26_selfplay_frontiers -q` → **10 passed
  in ~6 s** (4 Kuhn unit tests, 3 RPS unit tests, 1 tic-tac-toe unit
  test... precisely: Kuhn payoffs, equilibrium exploitability,
  CFR convergence, RPS Nash, FP convergence, BR cycling, minimax
  optimality, plus 3 script smoke subprocess runs).
- Full runs (seed 0): RPS standard/biased 20000 iters x 3 dynamics
  (~5 s each on CPU); Kuhn 20000 iterations (~5 s, expl 0.001608, value
  -0.055555, P1 bets J 0.2211 / K 0.6605 ratio 2.987, P1 calls Q 0.5613);
  tic-tac-toe 30000 episodes (~5 s CPU, X 97.5% wins vs random, draws
  minimax from ~18000 episodes).
- `npx prettier --write` on new `.md`/`.txt` files → clean.
- `npx vitepress build docs` → **succeeds** (~17 s), no dead links.

## Open issues

- CFR's average Q-call coordinate converges slowly along a
  near-indifference ridge (0.56 at 20k and 100k iterations vs limit
  ~0.55 for its alpha); exploitability is the reliable metric. The lab
  page documents this.
- The lab page is unreachable from the sidebar until the coordinator
  adds the entry above.
- `runs/` outputs are gitignored as required; PNG curves exist locally
  but are intentionally not embedded in the docs page.
