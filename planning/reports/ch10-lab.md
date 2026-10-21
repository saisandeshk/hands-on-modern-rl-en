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

## Review response

Reference numbers after the fix (`--seed 0 --device cpu`; train ~141 s, OPE ~54 s): offline — random BC 42.1±8.8 / DQN 500.0±0.0 / CQL 111.8±3.1; medium 500.0 / 99.1±3.5 / 500.0; expert 500.0 / 9.3±0.6 / 500.0; mixed 500.0 / 238.4±14.5 / 500.0. Expert max-Q finals: naive ≈2.2×10⁶ vs CQL 81.1. OPE truth 93.03±19.31; PDIS/WIS unchanged; FQE now 86.21/85.08/86.13/84.35 (gaps 6.8/8.0/6.9/8.7).

- Finding 1 (target sync masks overestimation): accepted — verified 2.2M vs 81.1 on expert with sync 100; default `--target-sync-every` is now 100 (smoke: 50); page/README rewritten around the new gap.
- Finding 2 (CQL "matches or beats ... on all four"): accepted — now "never catastrophic: 500 on medium/expert/mixed, 111.8 vs 500.0 on random"; same fix in README.
- Finding 3 ("~7 points everywhere" / "86–92"): accepted — now per-dataset gaps 6.8/8.0/6.9/8.7 and range 84–86 in page and README.
- Finding 4 ("held-out states"): accepted — code comments, plot labels/titles, page and README now say "first 2000 dataset states".
- Finding 5 (BC random 9.1 "faithfully copying noise"): accepted — re-measured (entropy 0.6929 ≈ log 2, agreement 0.502); page now states BC copies the distribution and argmax denoises it into a deterministic policy (now 42.1 vs behaviour 21.9 by luck, still far from solving).
- Finding 6 (FQE greedy backup vs smoothed truth): accepted — backup and V(s0) now use the smoothed-target expectation matching `target_probs_for`/`true_target_value`; new FQE 84–86 uniformly ~7–9 low from bootstrapping bias.
- Finding 7 (exercise 4 gamma/ESS hint): accepted — hint rewritten (estimand scale changes, ESS has no discount factor; check relative error).
- Finding 8 (exercise 3 zero-mass hint): accepted — hint rewritten (ratios 0/2, all trajectory weights 0 → WIS 0/0 = NaN, ESS 0, PDIS finite via prefix products).
- Finding 9 (exercise 5 not executable): accepted — `train_offline.py` now saves `offline/<dataset>_<method>_q.pt`; exercise rewritten to fine-tune `offline/random_cql_q.pt`.
- Finding 10 (exercise 2 destructive/small-data): accepted — rewritten with `--outdir runs/ch10_small` and curve-based prediction (full-data collapse already total).
- Finding 11 (sidebar / "10.L" numbering): rejected — scope forbids editing `docs/.vitepress/config.mjs`, and the task spec mandates `10.L Lab` front matter/H1; kept `10.L`, sidebar entry stays listed under "Sidebar additions" for the coordinator.
- Finding 12 (order-dependent max-Q quote): accepted — per-(dataset, method) derived seeds (`run_seed`) plus `torch.manual_seed` at each trainer entry; verified identical inits across orders; page quotes the stable values.
- Doubt 1 (PDIS/WIS "collapse onto behaviour"): accepted — now WIS-near-behaviour with PDIS noted as a high-variance realisation 44% above it.
- Doubt 2 ("plain behaviour average"): accepted — now "discounted behaviour average".
- Doubt 3 (FQE "initial states"): accepted — now "dataset episodes' initial states".
- Doubt 4 (best-checkpoint optimism): accepted — one-line disclosure added to Step 1.
- Doubt 5 ("never bad"): accepted — now "never catastrophic ... not free".
- Doubt 6 (smoke never syncs target): accepted — smoke sets `--target-sync-every 50` (8 syncs per 400 steps); default full-run sync is 100.
- Doubt 7 (FQE 10x diagnosis vs DQN trainers): accepted — same sync-100 fix applied to `train_q_offline`; report addendum above records the new max-Q endpoints.

## Open issues

- Lab page links assume the sidebar entry `/chapter10_offline_rl/lab` will
  be added by the coordinator; until then the page builds but is reachable
  only by direct URL.
- Single-seed results: the lab discloses this, and extension exercises
  invite multi-seed checks, but CI-style seed sweeps were out of scope.
- `bc_as_q` wraps BC logits as Q-values purely for shared evaluation
  plumbing; it is not a value function, which the code comment states.
