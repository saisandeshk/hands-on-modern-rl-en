# Report: ch25-lab — Reward Hacking and Honest Evaluation Lab

## Summary

Built the Chapter 25 hands-on lab from scratch: new directory
`code/chapter25_alignment_failures/` (gridworld env + tabular Q-learning,
from-scratch robust statistics, two CLI experiment scripts, pytest suite,
README) and new docs page `docs/chapter25_alignment_failures/lab.md`.
Part 1 shows specification gaming (respawning bonus farmed, 0% goal rate at
proxy return 57.0) and its fix (one-time bonus + potential-based shaping,
100% goal rate). Part 2 compares two hyperparameter settings over 12 seeds
with mean/median/IQM, stratified-bootstrap 95% CIs, and a performance
profile: IQM gap +0.240 with CI [-0.253, +0.642] (ranking unresolved),
while best-3 claims both configs are perfect (1.00 vs 1.00) and 4/12
single-seed pairs crown the wrong winner. All 9 tests pass; full runs take
~12 s on CPU; `npx vitepress build docs` succeeds.

## Technical fixes

No existing files were edited (another worker owns the chapter pages), so
there are no corrections to existing text. Design decisions verified by
actually running the code, not by assertion:

1. **File: `code/chapter25_alignment_failures/gridworld.py` — greedy-policy
   collapse under long proxy training.** At 1500 training episodes the
   greedy policy fell into a 2-state loitering loop (eval proxy -1.0) while
   the behavioral policy still farmed (train proxy ~25). Verified by
   tracing the greedy rollout and Q-values. Fix: default 400 episodes, and
   the script now reports _both_ behavioral tail statistics and greedy eval
   so the hacking claim rests on the training curve (proxy 40.2 at 0.00
   success), matching the Section 25.1 diagnostic rather than a fragile
   greedy snapshot.
2. **File: `code/chapter25_alignment_failures/run_eval_methodology.py` —
   degenerate all-zero Part 2 metric.** First version compared configs on
   success rate under proxy training: all 24 seeds scored 0.00 (no
   variance, nothing to aggregate). Verified by running. Fix: Part 2 trains
   under the repaired (fixed) reward with slip noise 0.1, giving genuinely
   overlapping score distributions (A median 0.855, B median 0.500, both
   with perfect and failed seeds).
3. **File: `code/chapter25_alignment_failures/run_eval_methodology.py` —
   best-3 can never flip a shared ceiling.** Verified by grid search: when
   one config reliably hits success 1.0, both configs' top-3 tie at 1.0.
   Resolution: the lab demonstrates the reversal three honest ways instead
   of forcing one — (a) best-3 claims "both perfect" while typical seeds
   score 0.50–0.86, (b) 4/12 aligned seed pairs disagree with the IQM
   winner, (c) an explicitly labeled adversarial min/max pick manufactures
   a B victory (1.000 vs 0.000) from the same data.
4. **File: `code/chapter25_alignment_failures/test_smoke.py` — shaping term
   in bonus test.** The fixed-mode first-visit bonus test ignored the
   potential-based shaping contribution (+0.525). Verified by hand
   computation; test now derives the expectation from `env.potential`.

## Other changes

- `code/chapter25_alignment_failures/stats.py`: mean/median/IQM,
  percentile bootstrap CI, stratified bootstrap per group, bootstrap CI of
  the IQM difference, performance profiles, one-row summary — numpy only.
- `code/chapter25_alignment_failures/requirements.txt`: numpy, matplotlib,
  pytest (all already in the project venv).
- `code/chapter25_alignment_failures/README.md`: what the lab shows, smoke
  vs full commands, runtime and expected results, test instructions, file
  overview.
- `docs/chapter25_alignment_failures/lab.md`: full lab page (learn goals,
  background with links to 25.1/25.4/25.5 sections, setup, code
  walkthrough with short excerpts, real run numbers in tables, mechanism
  discussion, 6 extension exercises with hints). No image files added, per
  spec.
- Smoke tests write to pytest `tmp_path` scratch dirs so they never
  clobber the full-run artefacts under `runs/`.

## Needs external verification

None. The only cited external result is Ng et al. (1999) on potential-based
shaping (well-established; the lab additionally unit-tests the invariance
on a tiny MDP solved exactly). No benchmark numbers, papers, or URLs were
invented — GitHub code links follow the mandated pattern and point to files
created in this task.

## Sidebar additions

- Title: `25.L Lab: Reward Hacking and Honest Evaluation`, link:
  `/chapter25_alignment_failures/lab` (page exists at
  `docs/chapter25_alignment_failures/lab.md`; not added to
  `docs/.vitepress/config.mjs` per instructions).

## Commands run and results

- `.venv/bin/python code/chapter25_alignment_failures/run_specification_gaming.py --smoke`
  → pass (proxy: return 57.00, success 0.00, ~30 bonus visits; fixed:
  return 14.27, success 1.00).
- `.venv/bin/python code/chapter25_alignment_failures/run_specification_gaming.py`
  (full, seed 0, 400 episodes) → proxy train 40.20/success 0.00, eval
  57.00/success 0.00; fixed train 14.22/success 1.00, eval 14.28/1.00.
- `.venv/bin/python code/chapter25_alignment_failures/run_eval_methodology.py`
  (full, 12 seeds, 400 episodes, slip 0.1, seed 0) → A: mean 0.688, median
  0.855, IQM 0.817 [0.450, 1.000]; B: mean 0.553, median 0.500, IQM 0.577
  [0.235, 0.877]; diff +0.240 [-0.253, +0.642]; best-3 1.000 vs 1.000;
  4/12 seed pairs disagree with IQM.
- `.venv/bin/pytest code/chapter25_alignment_failures -q` → 9 passed.
- `npx prettier --write` on README, lab page, and this report → clean.
- `npx vitepress build docs` → success, no dead links.
- Full-run wall time ≈ 12 s total on CPU (both scripts), far under the
  15-minute budget; smoke modes each finish in seconds.

## Open issues

- `lab.md` is not in the sidebar (coordinator to add; see above).
- The docs page quotes the seed-0 full-run numbers; re-running with a
  different `--seed` will change them (Part 2 CIs especially). The page
  states the seed; reproducibility is exact given the same seed.
- `runs/chapter25_alignment_failures/` artefacts are gitignored build
  outputs, intentionally left in the working tree for the coordinator to
  inspect.
