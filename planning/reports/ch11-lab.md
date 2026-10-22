# Report: ch11-lab — imitation lab (BC vs DAgger)

## Summary

Built a hands-on lab for Chapter 11 (previously no runnable code) under
`code/chapter11_imitation_meta_rl/` plus a docs page
`docs/chapter11_imitation_meta_rl/lab.md`. Main experiment
(`bc_dagger_cartpole.py`): a deterministic hand-written PD expert
(500/500 every seed), a BC sweep over N in {1, 2, 5, 10, 25} trajectories
showing the compounding-error effect (per-step agreement vs return, expert
states vs own states), and DAgger from a 1-trajectory start closing the
gap in two iterations. Companion script
(`gail_discriminator_cartpole.py`): the supervised GAIL discriminator half
only, reporting implicit rewards 3.451 (expert pairs) vs 0.502 (learner
pairs). Full main run: 472 s on CPU; all 8 pytest tests pass; vitepress
build succeeds.

## Technical fixes

No pre-existing files were corrected (the lab directory is new; existing
chapter pages were left untouched per instructions). Design claims in the
new material were each verified empirically:

- `bc_dagger_cartpole.py` — claim: the hand-written PD controller is an
  adequate expert. Verified: 500.0 mean return over 10 episodes and
  identical returns across repeated seeded runs (determinism unit test).
- `bc_dagger_cartpole.py` — claim: few-demo BC exhibits compounding
  error. Verified in the full run: N=1 gives 0.823 agreement on expert
  states but only 377.7 mean return (seed 1: 133.1 at 0.775 agreement);
  own-state agreement (0.766) trails expert-state agreement, confirming
  the distribution-shift mechanism of Section 11.1.2.
- `bc_dagger_cartpole.py` — claim: DAgger repairs a failing policy with
  on-policy corrections. Verified: seed 1 goes 133.1 → 451.4 → 500.0 over
  iterations 0–2 using one full expert trajectory plus corrections.
- `gail_discriminator_cartpole.py` — initial version trained the
  discriminator for 300 minibatch steps and failed to separate
  (accuracies 0.618/0.165, rewards 0.770 vs 0.717). Diagnosed by sweeping
  step budgets (300 → 2000 steps: expert/learner separation 0.83/0.95)
  and fixed by raising the full config to 2000 updates; final full run
  gives 0.739/0.819 and 3.451 vs 0.502.
- Full adversarial GAIL (REINFORCE on the learned reward from scratch)
  was prototyped twice and rejected: proper per-episode discounted
  returns still stalled below return 100 after 40 iterations. This matches
  the chapter's Section 11.2.4.1 (instability, millions of steps needed),
  so the lab ships the discriminator half only and the README explains
  the omission.

## Other changes

- `code/chapter11_imitation_meta_rl/README.md`: what the lab shows, run
  commands (full/smoke/tests), runtime table, reference results, and the
  rationale for leaving out full GAIL.
- `code/chapter11_imitation_meta_rl/requirements.txt`: gymnasium, torch,
  numpy, matplotlib (all already in the project env).
- `code/chapter11_imitation_meta_rl/test_smoke.py`: subprocess smoke runs
  of both scripts with timeouts and output-file assertions, plus unit
  tests for expert determinism/optimality, BC loss vs hand computation,
  expert relabelling, DAgger aggregation growth with loss improvement,
  agreement bounds, and evaluation bounds.
- `docs/chapter11_imitation_meta_rl/lab.md`: learn/background/setup/code
  walkthrough/real-results/what-to-look-for/6 exercises; links back to
  Sections 11.1 and 11.2; code links use the required GitHub URLs; no
  image files added (numbers in markdown tables only).

## Needs external verification

None. All paper/year claims cited in the lab page point at the chapter's
own Further Reading list (Ross et al. 2011, Ho & Ermon 2016); no new
papers, numbers, or URLs were introduced. All reported numbers come from
the runs performed in this worktree.

## Sidebar additions

- Title: `11.L Lab: Behaviour Cloning versus DAgger`, link:
  `/chapter11_imitation_meta_rl/lab`. (Config not edited per worker
  rules; coordinator to add.)

## Commands run and results

- `.venv/bin/pytest code/chapter11_imitation_meta_rl -q` → 8 passed
  (~8 s; subprocess smoke runs on CPU plus 6 unit tests).
- `.venv/bin/python
code/chapter11_imitation_meta_rl/bc_dagger_cartpole.py --device cpu`
  (full) → exit 0, runtime 471.7 s; wrote `metrics.json`, `metrics.csv`,
  `bc_dagger_results.png` under `runs/chapter11_imitation_meta_rl/`.
- `.venv/bin/python
code/chapter11_imitation_meta_rl/gail_discriminator_cartpole.py
--device cpu` (full) → exit 0, runtime 5.6 s; wrote
  `gail_metrics.json`, `gail_discriminator_reward.png`.
- `npx prettier --write` on the README and lab page → clean (`.py`
  files have no prettier parser; left as written).
- `npx vitepress build docs` → success (first attempt failed on
  unquoted `title:` colon in front matter; fixed with quotes, rebuilt
  clean).

## Open issues

- An earlier full run of `bc_dagger_cartpole.py` printed the complete
  summary and wrote all outputs, then the process died with SIGSEGV
  during interpreter teardown. The rerun exited 0 with identical
  structure. Treated as a flaky teardown (torch CUDA-build shutdown on
  CPU); outputs were validated (JSON parses, PNG exists). If the
  coordinator sees it again, running with `KMP_AFFINITY=disabled` or
  `MKL_THREADING_LAYER=GNU` is worth trying.
- CartPole is easy enough that BC saturates at N=2 trajectories, so the
  BC-vs-DAgger label-efficiency gap is modest in absolute labels; the
  lab frames the win as expert _trajectories_ (1 vs 2) plus the
  repair of the failing seed-1 policy. A harder env (LunarLander) is
  left as extension exercise 5.
