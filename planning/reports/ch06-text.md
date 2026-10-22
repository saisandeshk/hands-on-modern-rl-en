# Report: ch06-text

## Summary

Chapter 6 text pass. No partial work was present in the worktree (clean tree, no prior branch commits), so the full pass was done from scratch. Read all seven pages in scope plus the chapter's code (read-only) and sidebar config. Fixed six wrong/backward cross-references, one wrong output path, duplicate footnote definitions, two overstated technical claims, and two placeholder section summaries. Added the two missing in-scope core treatments (reward-to-go/causality in `reinforce.md`, entropy regularization in `pg-improvements.md`). Created `review.md` (8 outcomes, 9 key equations, 6 pitfalls, 8 exercises with fully worked, numerically verified solutions). `prettier` clean and `vitepress build` succeeds.

## Technical fixes

1. **`reinforce.md`: "Chapter 3 followed one route: learn $Q(s,a)$…" → Chapter 5.** The value route (learn $Q$, act with $\arg\max$) is Deep Q-Networks, Chapter 5 of this book's numbering table. Verified against sidebar (`5. Policy Gradient Methods` vs DQN chapter dir `chapter05_dqn`).
2. **`reinforce.md`: "policy objective $J(\theta)$ from Chapter 3" → Chapter 2.** The linked page is `chapter02_rl_basics/policy-value`; the objective is introduced in Chapter 2. Same fix applied to "Chapter 3 introduced the policy objective" in §Policy Objective.
3. **`pg-necessity.md`: "policy objective $J(\theta)$ in Chapter 2: [link](./reinforce)" → "introduced in Chapter 2 and derived in Section 6.1".** The old wording pointed a "Chapter 2" label at a Chapter 6 page. Verified both targets exist.
4. **`pg-improvements.md`: "$b(s)=V^\pi(s)$, the variance reduction is close to optimal" → softened.** Greensmith–Bartlett–Baxter (2004) derive a minimum-variance baseline that is generally _not_ $V^\pi$ and explicitly note the average-return baseline is not always optimal — a claim the same chapter's `baseline-experiment.md` already states correctly. Changed to "typically removes a large share of the variance, although it is not exactly the minimum-variance baseline in general". Verified from knowledge of the paper's results.
5. **`pg-improvements.md`: "the signal produced by a reward model is, in essence, also a kind of advantage estimate" (Chapter 13 bullet) → "a learned reward model supplies the reward signal from which advantage-like comparisons between responses are computed".** A reward model outputs rewards, not advantages; the old sentence conflated the two. Verified conceptually.
6. **`baseline-experiment.md`: render script output path `docs/chapter08_policy_gradient/images/` → `docs/chapter06_policy_gradient/images/`.** Chapter 8 is PPO; all referenced GIFs live in the chapter06 images directory (confirmed by listing it). This was a copy-paste error.
7. **`baseline-experiment.md`: removed unused duplicate footnote definitions `[^1]`, `[^2]`, `[^3]`.** Body uses only the named footnotes (`[^williams1992]`, `[^sutton1999]`, `[^greensmith2004]`); the numbered defs duplicated Williams/Sutton and added an unreferenced Gymnasium entry. Verified with grep that `[^1]`/`[^2]`/`[^3]` have no in-body references.
8. **Recomputed every numeric worked example in `reinforce.md`:** $1+1.8+2.43=5.23$ ✓; $\ln0.7\approx-0.357$, $\ln0.3\approx-1.204$ ✓; gradient sum $[0.929, 1.957]$ ✓; loss $6.282$ ✓; pg-necessity discretization $10^{12}\times10^{-6}\text{s}\approx11.6$ days ✓. No corrections needed.
9. **New reward-to-go subsection in `reinforce.md`:** $[0.1,0.2]\times5.23=[0.523,1.046]$ vs $[0.1,0.2]\times3=[0.3,0.6]$ — verified by execution-adjacent arithmetic and re-checked with `.venv/bin/python`.
10. **New entropy numbers in `pg-improvements.md` and `review.md` exercises:** $H([0.8,0.2])\approx0.500$, $H([0.5,0.5])=\ln2\approx0.693$, $\ln0.25\approx-1.386$, $G_0=2.62$ for $[1,0,2]$ — all verified with `.venv/bin/python`.

## Other changes

- `cartpole.md` and `cartpole-baseline.md`: replaced identical placeholder section summaries ("This lab connects the runnable experiment…") with real summaries of each page's takeaways.
- `reinforce.md`: added "Reward-to-Go: Past Rewards Are Not Your Doing" subsection (causality identity + numeric contrast using the chapter's running 3-step example + pointer to `compute_returns`).
- `pg-improvements.md`: added "Entropy Regularization" subsection (objective, gradient reading, worked entropy comparison, $\beta$ guidance, forward pointer to PPO); added one conditioning clause to the baseline-unbiasedness proof.
- Created `docs/chapter06_policy_gradient/review.md` (new page, needs sidebar addition).
- Ran `prettier --write` on all touched/created files (only `reinforce.md` and `review.md` needed reformatting).

## Needs external verification

- Williams (1992) and Sutton et al. (1999/2000) citations: standard, high-confidence; DOI/JMLR links retained as-is (not fetched — no internet).
- Reported experiment numbers in 6.5/6.6 (last-50 averages `95.1`/`493.0`, variances `100.41`/`38.27`): consistent between the two pages and plausible for the shipped scripts at the stated seeds, but not re-run here (a full 2×500-episode CartPole comparison exceeds the time budget; code text matches the prose: lr `1e-3`, hidden 128, seeds 0/100).
- `render_cartpole_baseline.py` CLI flags (`--episodes 500 --seed 0`) quoted in 6.5/6.6: not executed; flag correctness should be confirmed by the code owner if the render script changes.

## Sidebar additions

- `6.R Summary and Exercises` → `/chapter06_policy_gradient/review` (new file `docs/chapter06_policy_gradient/review.md` with front matter `title: 6.R Summary and Exercises`). No other additions; did not touch `config.mjs`.

## Commands run and results

- `npx prettier --write` on all 8 files: clean (2 reformatted).
- `npx vitepress build docs`: success, all pages render, no dead links.
- `.venv/bin/python -c` arithmetic checks for review/pg-improvements numbers: all match.
- Allowed-shell note: `code/` was read but never edited (another worker owns it); no tests were run since the task names none and the chapter scripts require long training runs.

## Open issues

- **Sections 6.5 (`baseline-experiment.md`) and 6.6 (`cartpole-baseline.md`) are near-duplicates**: same experiment, same scripts, same reported numbers (`95.1`/`493.0`, `100.41`/`38.27`), same figures, near-identical prose. Both are individually correct, so both were left intact, but the coordinator should consider merging them into one page (and updating the sidebar) rather than shipping the same comparison twice. The 6.6 learning-path line (6.4 → 6.5 → 6.6) currently asks the reader to run the same experiment twice.
- **RL-4 checklist coverage** (in-scope items only): covered — trajectory objectives/expected return; likelihood-ratio/score-function identities; policy-gradient theorem; REINFORCE/Monte Carlo PG; causality/reward-to-go (added); constant and state-dependent baselines; baseline unbiasedness/variance reduction; learned value baselines (critic targets properly deferred to Ch. 7); entropy regularization (added); categorical/Gaussian action distributions. Correctly deferred to Ch. 7/8/9: actor-critic dependencies, n-step/GAE, $\gamma$/$\lambda$ tradeoffs, importance ratios, PPO clipped surrogate, approx-KL/clip-fraction/value-loss diagnostics, advantage centering/minibatch leakage, multi-epoch drift, tanh-squash corrections, vectorized rollouts, fixed-trajectory gradient tests, collapse/stale-log-prob diagnosis, multi-seed evaluation protocol; trust-region/monotonic-improvement proofs (Depth/Later).
- Minor: a full-width minus sign in the `reinforce.md` loss derivation (line ~354) could not be byte-matched for replacement; harmless for rendering/build, left as-is.
