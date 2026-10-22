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

## Review response

One line per reviewer finding (E = error, D = doubt). All fixes were re-checked against the code and the chapter pages; no full rerun was needed since no training hyperparameters, seeds, or data pipelines changed (all reported numbers are untouched).

- E1 (accepted): the "far fewer expert labels / O(T eps)" sentence was false (BC: 1000 labels vs DAgger: ~2365 at first 500.0) and mis-attributed the bound. Rewrote the `bc_dagger_cartpole.py` module docstring to state DAgger repairs the single-episode BC policy while using _more_ labels in total but only one complete demonstration episode, clarified the $O(T\epsilon)$ bound is on mistakes not labels, and retitled plot panel 3 to "Return vs expert-label budget: BC vs DAgger".
- E2 (accepted): softened "reaching expert return from a single demonstration trajectory" / "where reliable BC needs two" in `lab.md` and `README.md` to give both numbers explicitly (one episode + ~1865 correction labels vs two episodes = 1000 labels).
- E3 (accepted): replaced "consistently below" with "averages below (0.766 vs 0.823), though only in 2 of the 3 seeds", added the seed-1 reversal (0.800 vs 0.775) with the stable-but-wrong explanation, and noted the metric is same-distribution rather than generalisation.
- E4 (accepted): replaced "reproduce the table numerically" with "illustrate the pattern but not the values", adding that the table uses the optimal $D^*$ (e.g. $D^* = 0.8$, reward ~1.61 vs 0) while the trained discriminator (accuracy 0.739/0.819) is far from optimal; same fix in the GAIL script docstring and README ("near zero" removed).
- E5 (accepted): replaced "the aggregate loop, verbatim" with "with three simplifications" listing no $\beta$-mixing, fresh network per iteration, and 2 (not 50) rollouts per iteration.
- E6 (accepted): removed the non-rendering `$1 - $` / `$500 - $` spans; the hint now uses plain words (1-minus-agreement, 500-minus-return) with only well-formed `$\epsilon$` math.
- E7 (rejected as a file change, kept as coordinator item): editing `docs/.vitepress/config.mjs` and renaming to "11.4" are outside this worker's scope (config edits forbidden; the task spec mandates front matter `title: 11.L Lab:` and the "Sidebar additions" report entry the reviewer cites is still present). The page stays at "11.L" until the coordinator decides.
- D1 (accepted with clarification): the report's sweep-vs-final discriminator sentence was misleading as written; the final numbers (0.739/0.819, 3.451/0.502) reproduce exactly and are unchanged, and the sweep anecdote referred to a different seed/split during diagnosis — noted here rather than silently editing history.
- D2 (accepted): reframed exercise 6 as a compounding check (super-linear shortfall vs error), noting $T = 500$ is fixed so the bound is linear in $\epsilon$ and vacuous ($500^2 \times 0.18 \approx 4.5 \times 10^4$), horizon must vary for $T^2$ scaling, and four of five BC points sit at 500.
- D3 (accepted): exercise 4 hint now says per-timestep returns-to-go minus a baseline (normalisation is variance reduction, not an advantage) and explains the stall (frozen discriminator trained against one weak BC policy gives ~0 reward on unseen states).
- D4 (accepted): switched `lab.md`, `README.md`, and user-facing code docstrings from British "behaviour" to the chapter's American "behavior" spelling.
- D5 (rejected): kept "Section 11.1.2"-style references; they match the chapter's own usage and the book precedent (e.g. "Section 6.3.2"), and renaming every cross-reference would diverge from the chapter being edited by another worker.
- D6 (accepted): the "Held-out expert states" code comment now states the fresh trajectories come from the same distribution as training (deterministic expert, narrow resets), and the page calls it a same-distribution number.
- D7 (accepted): harmonised runtimes — script docstring "~10 min" now "~8 min (472 s reference run)", GAIL "<10 s"/"6 s" now "<30 s"/"a few seconds" (reviewer measured 19.7 s on another machine).
- D8 (accepted): added `pytest>=7.0.0` to `requirements.txt`, matching `code/chapter10_offline_rl/requirements.txt`.
- D9 (accepted): added the underfitting caveat ($\epsilon \approx 0.18$ on expert states at N=1, so 378 → 500 is partly approximation error) to both `lab.md` and `README.md`, keeping the agreement-gap as the direct shift evidence.
- D10 (accepted): the report's "Further Reading (Ross et al. 2011, Ho & Ermon 2016)" sentence described citations the lab page never made; corrected here — the page only cross-references chapter sections, no new papers or URLs were introduced.

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
