# appD-text report

## Summary

Checked every definition, derivation, and numeric example in `docs/appendix_d_math/` (27 pages across modules D.1--D.4) for mathematical correctness by recomputing all numbers. Fixed 8 real errors: three wrong discount-return values, a wrong Q-learning update target, a wrong advantage formula, missing `+` signs in the multivariate Taylor expansion, a misleading "Q does not depend on theta" claim, an off-by-10x convergence-bound entry, plus roadmap link-text mismatches and small clarifications. Added the new page `docs/appendix_d_math/notation.md` (front matter `title: D.N Notation`, H1 `# D.N Notation`) with grouped one-line symbol definitions and a conflicts section. No summary/exercises page was created or modified beyond the fixes above.

## Technical fixes

1. `docs/appendix_d_math/probability-value.md` — discount-return table was wrong. Wrong claim: rows `3.19`, `5.97`, `6.92` for rewards `[2,1,3,0,1]`. Correction: `2+0.5+0.75+0+0.0625=3.3125` (now `3.31`); `2+0.9+2.43+0+0.6561=5.9861` (now `5.99`); `2+0.99+2.9403+0+0.9606=6.8909` (now `6.89`). Verified with `.venv/bin/python` arithmetic.
2. `docs/appendix_d_math/probability-sampling.md` — Q-learning skeleton row was wrong. Wrong claim: `Q(s,a) <- Q(s,a)+alpha[r+gamma max Q-V(s)]`. Correction: `...-Q(s,a)]` with target `r+gamma max_{a'}Q(s',a')`. Verified against the standard Q-learning update (the subtrahend must be the current Q estimate, not V).
3. `docs/appendix_d_math/calculus-policy-gradient.md` — advantage summary formula was wrong. Wrong claim: `A(s,a)=R-V(s)` (immediate reward minus value). Correction: `A(s,a)=Q(s,a)-V(s)`. Verified: advantage compares action value (or return), not one-step reward, against the state baseline; consistent with `A=G-V` used two paragraphs above.
4. `docs/appendix_d_math/calculus-derivations.md` — multivariate Taylor expansion had the wrong signs. Wrong claim: `f(theta) - nabla f(theta)^T Delta - 1/2 Delta^T H Delta` (both correction terms subtracted). Correction: `f(theta)+nabla f^T Delta+1/2 Delta^T H Delta`. Verified against the scalar expansion directly above it.
5. `docs/appendix_d_math/calculus-derivations.md` — overstated independence claim. Wrong claim: "`Q^pi(s,a)` does not depend on `theta`". Correction: qualified as a simplified one-state derivation holding Q fixed, noting the full policy gradient theorem shows state-distribution gradient terms cancel. Verified from standard policy-gradient-theorem reasoning (Q does depend on the policy in general).
6. `docs/appendix_d_math/calculus-advanced-formulas.md` — missing DQN assumption. Added: the target `y_t` is treated as fixed when differentiating (not differentiated w.r.t. `theta`). Verified against semi-gradient TD/DQN convention.
7. `docs/appendix_d_math/linear-algebra-advanced.md` — convergence-bound table entry off by 10x. Wrong claim: `gamma=0.9, k=50` error `0.005`. Correction: `10*0.9^50=0.0515`, now `0.05`. Verified with `.venv/bin/python` (`0.9**50*10 = 0.0515`); all other table cells rechecked and correct.
8. `docs/appendix_d_math/probability-statistics.md` — roadmap link texts did not match page titles (`D.2.3 Variance, Monte Carlo, and Sample Averages`, `D.2.5 Bellman Expectation Equation`, `D.2.6 Summary, Formulas, and Exercises`). Correction: aligned to actual titles (`D.2.3 Monte Carlo, Incremental Averages, and Importance Sampling`, `D.2.5 ... and Action Values`, `D.2.6 Probability and Statistics Formula Reference and Exercises`). The old D.2.3 label was additionally misleading since variance is covered in D.2.2.
9. `docs/appendix_d_math/probability-bellman-advanced.md` — covariance example now labeled "(population form, averaging over the two samples)" to resolve the divide-by-N vs N-1 ambiguity. Value `2.25` itself verified correct.
10. New `docs/appendix_d_math/notation.md` — created per task (title/H1 `D.N Notation`); symbols grouped by topic (environment/trajectories; values/advantages; policies/ratios/data; optimization; LLM alignment; information theory) with one-line meanings, based on sampling Chapters 3, 5, 8, 10, 11, 13, 14, 20, 22, 25, 26.

Other numerics spot-checked and found correct (no change): `E[R]=7`; `p_pi=0.62`; `E[G|S]=6`; `v(s)=4.8`; MC mean `8`; incremental `7.8`; trajectory probability `0.024`; TD error `0.5`; PPO clip `min(6,4.8)=4.8`; Bellman solution `(3.33,2.67)`; `v=6.8, A=(-1.8,1.2)`; `rho=4`, `4*5=20`, `3^10=59049`; eigenvalues `5,2`; gradient-clip norm `15.56` and rescaled vector; weighted norm `5`; softmax `0.73/0.27`; entropies `1` bit / `0.47` bit / `0.693` nat; GRPO `mu=5, sigma=sqrt(5), A_4=1.34`; Taylor `9.6/9.61`; DQN gradient form; Bellman matrix identities.

## Other changes

- Prettier reformatting only (no content change) in touched files.
- No cross-reference/numbering changes; all internal links left intact.

## Needs external verification

- None. All fixes were arithmetic, standard-formula, or internal-consistency corrections verifiable from the text itself. No paper/year/venue/benchmark claims were added or altered.

## Sidebar additions

- New page: "D.N Notation" linking to `/appendix_d_math/notation` (file `docs/appendix_d_math/notation.md`). Coordinator: please add it to the sidebar in `docs/.vitepress/config.mjs` (suggested placement: end of the Appendix D section, after the D.4.6 formulas page).

## Commands run and results

- `.venv/bin/python -c ...` — recomputed discount returns (`3.3125`, `5.9861`, `6.8909`) and convergence bounds (`0.9^50*10=0.0515`); confirmed the reported errors.
- `npx prettier --write <21 touched/created files>` — all formatted successfully.
- `npx vitepress build docs` — succeeded (build complete, no dead links).

## Open issues

- Notation conflicts documented in `notation.md` under "Known conflicting usages": `V^pi` vs `v_pi`; `pi_old` vs `pi_{theta_old}`; `rho_t` vs `r_t(theta)`; `R_{t+1}` vs `r_t` in TD errors; overloaded `beta`, `delta`, `sigma`, `D`/`mathcal{D}`. Harmonizing these across the book is a larger cross-chapter task left to the coordinator.

## Review response

- Error 1 (notation.md KL row broken by unescaped `|`): accepted. Replaced with `$D_{\mathrm{KL}}(P \Vert Q)$`, kept two cells, restored two-cell separator row.
- Error 2 (notation.md not in sidebar, `D.N` placeholder): rejected as a worker fix. Worker rules forbid editing `docs/.vitepress/config.mjs`; the original task required front matter `title: D.N Notation` and H1 `# D.N Notation`. Kept `D.N`; sidebar addition remains listed under Sidebar additions for the coordinator.
- Error 3 (reward-index convention backwards): accepted. Re-checked: `r_t + \gamma V(s_{t+1}) - V(s_t)` dominates in the book; fixed the symbol-table row to list `$r_t$` first and rewrote the conflict note (dominant convention is `r_t`; D.2.4 uses `r_t`, D.3.5 uses `R_{t+1}`).
- Error 4 (ratio attribution): partially accepted, partially rejected. Accepted that the note was too narrow: rewrote it to cover both usages — `rho_t` in D.2.5 and the Chapter 10 review/experiments pages, `r_t(theta)` in PPO chapters and D.3, with product forms `rho_{0:T}` (D.2.5) and `rho_{1:t}` (Chapter 10). Rejected the claims that Chapter 10 never uses `rho` and that `rho_{1:t}` exists nowhere: `docs/chapter10_offline_rl/review.md:30,81` and `experiments.md:152,172` use `\rho_{1:t}` explicitly.
- Error 5 (D.1/D.3/D.4 roadmap link texts mismatch): accepted. Aligned all roadmap rows to the sidebar/H1 titles (e.g. `D.3.2 Policy Gradients and Advantage Functions`, `D.3.5 Complete Formulas for PG, DQN, GAE, PPO, and GRPO`, `D.4.3 Information Theory in PPO, RLHF, and DPO`), and fixed the matching `Next` pointers.
- Error 6 (15 front-matter `title:` vs H1 mismatches): accepted. Set each front matter to the sidebar/H1 title (D.3.1–D.3.6, D.4.1–D.4.6, D.1.1, D.2.1–D.2.2).
- Error 7 (factual fixes not recorded in `planning/content-review-notes.md`): rejected as a worker fix. That file is outside the task scope (scope lists only Appendix D files, `notation.md`, and this report), and worker rules forbid editing files outside scope. The fixes remain recorded here.
- Doubt 1 (state-distribution gradient wording): accepted. Reworded to cancellation against the `Q^\pi` dependence rather than terms summing to zero by themselves.
- Doubt 2 (advantage body vs summary): accepted. Body now marks the numbers as one-sample estimates (`A(s,a) \approx G - V(s)`) and states the definition `A(s,a)=Q(s,a)-V(s)` explicitly.
- Doubt 3 (GRPO `/(sigma)` vs `/(sigma+epsilon)`): accepted. Standardized D.3.4 and D.3.6 to `/(sigma+epsilon)` with a stability note, matching D.3.5 and `chapter15_grpo/grpo-practice-and-mechanism.md:526` (`group_std + eps`); noted the variant on the notation page.
- Doubt 4 (`Chapter 2 defined J(theta)`): accepted. Changed to the chapter-neutral `The policy-gradient chapters define a policy objective J(theta)` since only Chapter 6 defines it.
- Doubt 5 (report misdescribed the Taylor defect): accepted. Fixed item 4 above to say the two terms were subtracted instead of added.
- Doubt 6 (ratio expansion missing evaluation point): accepted. Added `\big|_{\theta=\theta_{old}}` to both gradient and Hessian in the `r_t(theta)` expansion.
