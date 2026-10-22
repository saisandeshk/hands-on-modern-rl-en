# appD-text report

## Summary

Checked every definition, derivation, and numeric example in `docs/appendix_d_math/` (27 pages across modules D.1--D.4) for mathematical correctness by recomputing all numbers. Fixed 8 real errors: three wrong discount-return values, a wrong Q-learning update target, a wrong advantage formula, missing `+` signs in the multivariate Taylor expansion, a misleading "Q does not depend on theta" claim, an off-by-10x convergence-bound entry, plus roadmap link-text mismatches and small clarifications. Added the new page `docs/appendix_d_math/notation.md` (front matter `title: D.N Notation`, H1 `# D.N Notation`) with grouped one-line symbol definitions and a conflicts section. No summary/exercises page was created or modified beyond the fixes above.

## Technical fixes

1. `docs/appendix_d_math/probability-value.md` — discount-return table was wrong. Wrong claim: rows `3.19`, `5.97`, `6.92` for rewards `[2,1,3,0,1]`. Correction: `2+0.5+0.75+0+0.0625=3.3125` (now `3.31`); `2+0.9+2.43+0+0.6561=5.9861` (now `5.99`); `2+0.99+2.9403+0+0.9606=6.8909` (now `6.89`). Verified with `.venv/bin/python` arithmetic.
2. `docs/appendix_d_math/probability-sampling.md` — Q-learning skeleton row was wrong. Wrong claim: `Q(s,a) <- Q(s,a)+alpha[r+gamma max Q-V(s)]`. Correction: `...-Q(s,a)]` with target `r+gamma max_{a'}Q(s',a')`. Verified against the standard Q-learning update (the subtrahend must be the current Q estimate, not V).
3. `docs/appendix_d_math/calculus-policy-gradient.md` — advantage summary formula was wrong. Wrong claim: `A(s,a)=R-V(s)` (immediate reward minus value). Correction: `A(s,a)=Q(s,a)-V(s)`. Verified: advantage compares action value (or return), not one-step reward, against the state baseline; consistent with `A=G-V` used two paragraphs above.
4. `docs/appendix_d_math/calculus-derivations.md` — multivariate Taylor expansion was malformed. Wrong claim: `f+e...` with missing operators (`f(theta) nabla f^T Delta 1/2 Delta^T H Delta`). Correction: `f(theta)+nabla f^T Delta+1/2 Delta^T H Delta`. Verified against the scalar expansion directly above it.
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
- `npx prettier --write <9 touched/created files>` — all formatted successfully.
- `npx vitepress build docs` — succeeded (build complete, no dead links).

## Open issues

- Notation conflicts documented in `notation.md` under "Known conflicting usages": `V^pi` vs `v_pi`; `pi_old` vs `pi_{theta_old}`; `rho_t` vs `r_t(theta)`; `R_{t+1}` vs `r_t` in TD errors; overloaded `beta`, `delta`, `sigma`, `D`/`mathcal{D}`. Harmonizing these across the book is a larger cross-chapter task left to the coordinator.
