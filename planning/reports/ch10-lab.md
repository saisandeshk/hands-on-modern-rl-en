# Report: ch10-lab — offline-RL hands-on lab

## Summary

Built the missing hands-on lab for Chapter 10 from scratch. New directory
`code/chapter10_offline_rl/` with `common.py`, `collect_datasets.py`,
`train_offline.py`, `ope.py`, `test_smoke.py`, `requirements.txt`,
`README.md`; new docs page `docs/chapter10_offline_rl/lab.md`. The lab runs
CartPole-v1 end to end: behaviour datasets of four quality levels from one
online DQN, offline BC / naive DQN / discrete CQL comparison, and OPE
(PDIS, WIS, FQE) of the expert target against online ground truth. Full
configuration runs in ~4 min on CPU and is bit-reproducible for a fixed
`--seed`. All 6 tests pass; smoke mode finishes in ~15 s on CPU.

## Technical fixes

No existing files were edited (scope forbids touching chapter pages), so
there are no corrections to published claims. Two bugs were found and fixed
in the new lab code itself:

1. **Unseeded `action_space.sample()` broke reproducibility**
   (`code/chapter10_offline_rl/collect_datasets.py`). Epsilon-greedy
   exploration draws from `env.action_space.sample()`, whose RNG is
   independent of `np.random`/`torch` seeds. Two identical CPU runs gave
   expert returns of 500.0 vs 430.6. Verified by: running the 150-episode
   config twice (197.9 vs 179.4 before the fix; 165.7 vs 165.7 after).
   Fix: `env.action_space.seed(...)` / `observation_space.seed(...)` on
   every created env, plus single-threaded deterministic torch in
   `common.seed_everything`. Re-ran collect/train/ope to confirm
   bit-identical outputs across runs.
2. **FQE underestimated by ~10x (9.3 vs truth ~93)** (initial `ope.py`).
   Root cause worked out by hand: with a target-net sync every 500 steps
   over 5000 total steps, the Bellman backup propagates only ~10 steps
   back, and $(1-\gamma^{10})/(1-\gamma) \approx 9.6$ — exactly the observed
   value. Verified by: a 20k-step/sync-100 experiment reaching 86.5.
   Fix: default `--fqe-steps 20000`, sync every 100 steps. FQE now lands at
   85.7–92.2 vs truth 93.03 on all four datasets.

## Other changes

- `train_offline.py` records final mean max-Q per method in
  `summary.json` (`dqn_max_q_final`, `cql_max_q_final`); Q-curve plot uses
  per-panel autoscale instead of a 0–500 axis that hid the expert-panel
  divergence (34.7 naive DQN vs 26.9 CQL); OPE legend moved outside the axes
  (it covered the random-dataset bar).
- Online DQN keeps the best checkpoint seen during training (evaluated
  every 25 episodes) so the expert behaviour is robust to seed luck; without
  it one seed-0 run produced an 83-return "expert".
- `test_smoke.py` forces `--device cpu` and writes to an isolated
  `runs/chapter10_offline_rl_smoke_test/` directory (an early pytest run
  with the default outdir overwrote the full-run artifacts, which were then
  regenerated; final artifacts are from the fixed code).

## Needs external verification

- None on factual claims: the lab cites only the papers already listed in
  Chapter 10 (Fujimoto et al. 2019, Kumar et al. 2020, Kostrikov et al. 2022) and otherwise reports locally measured numbers. The CQL formula and
  the $\epsilon_{\text{ood}}/(1-\gamma)$ amplification match Section 10.1.
- Behaviour-policy quality (e.g. medium 203.2, expert 490.9) and all table
  numbers are single-seed (`--seed 0`) measurements, stated as such in the
  lab page and README.

## Sidebar additions

- Title: `10.L Lab: Offline RL from Fixed Datasets`, link:
  `/chapter10_offline_rl/lab` (to be placed after `10.3` in the Chapter 10
  sidebar group). I did not edit `docs/.vitepress/config.mjs` per the work
  rules.

## Commands run and results

Full pipeline, `--seed 0 --device cpu` (repo root):

- `collect_datasets.py` — 49 s. Behaviour returns: random 21.9 (2286
  episodes), medium 203.2 (244), expert 490.9 (101), mixed 233.3 (214);
  online DQN greedy 500.0 ± 0.0; best checkpoint 500.0.
- `train_offline.py` — ~120 s. random: BC 9.1/DQN 332.9/CQL 241.6; medium:
  500.0/479.5/500.0; expert: 500.0/9.3/500.0; mixed: 500.0/500.0/500.0.
  Re-ran: bit-identical table (determinism check passed).
- `ope.py` — ~44 s. Truth 93.03 ± 19.31. random: PDIS 27.71 / WIS 18.37 /
  FQE 92.23 / ESS 0.0%; medium: 35.74/31.89/86.46/4.9%; expert:
  78.84/90.91/86.34/5.1%; mixed: 47.24/90.17/85.69/2.4%. Re-ran:
  bit-identical (determinism check passed).
- `.venv/bin/pytest code/chapter10_offline_rl -q` — 6 passed (~15 s on CPU).
- `npx prettier --write` on the new/changed md files — clean.
- `npx vitepress build docs` — success (16 s; only a pre-existing `jinja2`
  highlighting warning unrelated to this lab).

## Open issues

- Lab page links assume the sidebar entry `/chapter10_offline_rl/lab` will
  be added by the coordinator; until then the page builds but is reachable
  only by direct URL.
- Single-seed results: the lab discloses this, and extension exercises
  invite multi-seed checks, but CI-style seed sweeps were out of scope.
- `bc_as_q` wraps BC logits as Q-values purely for shared evaluation
  plumbing; it is not a value function, which the code comment states.
