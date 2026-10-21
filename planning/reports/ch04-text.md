# Report: ch04-text

## Summary

Text pass over Chapter 4 (`docs/chapter04_classical_methods/`). Read all three pages fully, recomputed every numeric example, and executed verification code for the DP/MC/TD corridor, the GRPO numbers, and the new exercise solutions. Fixed one real math typo (missing `+` in a DP target equation), several citation/cross-reference errors, and two front-matter title mismatches. Added the chapter's missing core coverage — policy iteration, value iteration, GPI, stopping criteria, first/every-visit MC with incremental averages, n-step returns, TD(λ)/eligibility traces, SARSA, Expected SARSA, Q-learning, ordinary vs weighted importance sampling, and terminal vs truncated bootstrapping — as tight textbook sections. Created `review.md` (summary + 8 exercises with worked solutions, all numerics verified). Build passes; prettier clean.

## Technical fixes

1. **`dp-mc-td.md` — DP target equation missing `+`.** The "Relationship" section expansion read `R(s,a)` followed directly by `γ Σ P V` with no operator. **Correction:** inserted the missing `+` so it matches the (correct) expansion earlier in the same page and the Bellman expectation form. **Verified:** by inspection against the page's own correct formula and the standard Bellman backup.
2. **`reward-design.md` — HER author garbled.** Footnote read "Tobin, J., Pieter Abbeel, O., & Zaremba, W." **Correction:** "Tobin, J., Abbeel, P., & Zaremba, W." **Verified:** well-known author list of Andrychowicz et al. (2017) Hindsight Experience Replay.
3. **`reward-design.md` — Gao et al. year inconsistent.** Body/figure caption cited Gao et al. (2023); footnote said 2022, ICML. **Correction:** footnote changed to "(2023). Scaling laws for reward model overoptimization. ICML." **Verified:** the paper is arXiv Oct 2022, published ICML 2023 — the in-text (2023) date is the consistent one.
4. **`reward-design.md` — PopArt footnote wrong.** It listed wrong authors and the title "When using function approximation, a simpler target often yields better results" with a workshop venue. **Correction:** "van Hasselt, H., et al. (2016). Learning values across many orders of magnitude (PopArt). NeurIPS." **Verified:** well-known PopArt paper title/venue; used "et al." to avoid asserting an uncertain full author list (see Needs external verification).
5. **`reward-design.md` — wrong cross-reference.** "In Section 3.1 we defined … the discounted return" — return $G_t$ is defined in Chapter 2 (`chapter02_rl_basics/mdp.md`, `policy-value.md`), not Section 3.1. **Correction:** "In Section 2.2 (Chapter 2)…". **Verified:** by grep of Chapters 2–3.
6. **`reward-design.md` — stale prev-section link text** ("Where Does the Data Come From"). **Correction:** "Policy Sampling and Data Sources", matching the sidebar/H1 of 4.2. **Verified:** against sidebar.
7. **`algorithm-taxonomy.md` / `dp-mc-td.md` — front-matter titles mismatched H1/sidebar** (`4.2 Data Sources…` vs `4.2 Policy Sampling and Data Sources`; `4.1 DP, MC, and TD` vs full title). **Correction:** front matter aligned to H1/sidebar titles. **Verified:** against sidebar.
8. **`algorithm-taxonomy.md` — unverifiable 44% throughput claim** ("FP8-RL in veRL … improving rollout throughput by 44% while matching the BF16 baseline") and named-method claims (MinPRO, R3). **Correction:** softened to qualitative statements (higher-throughput low-precision training "while staying close to the higher-precision baseline"; "reported variants… per-prefix statistics"; "Routing consistency" instead of "R3"). **Verified:** cannot verify offline — no internet; listed below.
9. **`review.md` Exercise 7 — sweep-count estimate.** Draft said "about 60–70 sweeps". **Correction:** "23 sweeps" after executing the exact loop (final $V(S)=-3.37499918$, $V(M)=-1.87499968$ vs $-3.375/-1.875$). The dominant error eigenvalue is ~0.51/sweep, not 0.8.
10. **Recomputed chapter numerics (all confirmed, no change needed):** DP fixed point $V(S)=-3.375$, $V(M)=-1.875$ (closed-form solve); sweeps $V_1=(-1.2,-1.2)$, $V_2=(-2.4,-1.44)$, $V_3=(-2.832,-1.68)$; MC every-visit targets $(-7,-5,-4,-2,-1)$ and post-episode table $(-3.125,-1.5)$; TD step-by-step table ending $(-1.75,-1.25)$; PBRS numbers $1.3/-0.5$; GRPO advantages $1.29/-0.77$ (std $0.484$); CartPole/KL/PBRS-telescoping arithmetic; Q-learning corridor run converging to greedy (R,R), $(-2,-1)$. **Verified:** hand derivation plus `.venv/bin/python` execution.

Known suspected issues in this chapter: (none recorded).

## Other changes

- **`dp-mc-td.md`**: new `### First-Visit, Every-Visit, and Incremental Averages` under MC (both estimators, incremental-mean derivation, $1/N$ vs constant-$\alpha$); new `## From Evaluation to Control` (residual stopping $\max|V_{k+1}-V_k|<\theta$, contraction intuition, policy-vs-value convergence, policy improvement with corridor $Q$ numbers, policy iteration, value iteration with 2-sweep worked example, GPI, sync/async, tie handling, terminal/unreachable conventions); new `## n-Step Returns and TD(λ)` (return boundaries, bias-variance knob, $\lambda$-return, accumulating-trace backward view); new `## Tabular Control: SARSA, Expected SARSA, and Q-Learning` (three updates, cliff-walk risk difference, $\epsilon$-greedy/GLIE + Robbins-Monro schedules, terminal-vs-truncated code sketch); summary extended (bullets 6–8) and stale "next section moves to Q" pointer fixed.
- **`algorithm-taxonomy.md`**: new `### Correcting the Mismatch: Importance Sampling` (one-step ratio with $0.8/0.5=1.6$ worked check, trajectory and per-decision IS, ordinary vs weighted estimators, support/coverage, ESS with $3.39$ worked example, exploratory-behavior rationale).
- **`review.md`** (new): outcomes, key equations, pitfalls, 8 ordered exercises (conceptual, by-hand, derivations, 2 coding) each with a `::: details Solution` block; all numeric answers executed/checked.
- No sidebar, image, component, or `code/` edits.

## Coverage table (in-scope items)

| Item                                                                                         | Status                                          | Where                                         |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------- |
| Iterative policy evaluation                                                                  | covered (was covered)                           | 4.1 DP section + stopping rule added          |
| Policy iteration                                                                             | covered (was partial, now full)                 | 4.1 new control section, corridor $Q$ numbers |
| Value iteration                                                                              | covered (was missing)                           | 4.1, 2-sweep worked example                   |
| Generalized policy iteration                                                                 | covered (was missing)                           | 4.1                                           |
| Stopping criteria (residual, policy vs value)                                                | covered (was missing)                           | 4.1                                           |
| Monte Carlo, first/every-visit, incremental means, constant step size                        | covered (was partial)                           | 4.1 new MC subsection                         |
| TD(0), TD error                                                                              | covered (was covered)                           | 4.1, unchanged                                |
| n-step returns and return-target boundaries                                                  | covered (was missing)                           | 4.1                                           |
| Eligibility traces / TD(λ), forward-backward view                                            | covered at working depth (was missing)          | 4.1                                           |
| SARSA control                                                                                | covered (was missing)                           | 4.1                                           |
| Expected SARSA control                                                                       | covered (was missing)                           | 4.1                                           |
| Q-learning control                                                                           | covered (was mentioned only; update now taught) | 4.1 (+4.2 mention, Ch.5 link)                 |
| On- vs off-policy learning, behavior/target policies                                         | covered (was covered)                           | 4.2, unchanged + IS section                   |
| Importance sampling, ordinary vs weighted                                                    | covered (was missing)                           | 4.2 new subsection                            |
| Terminal vs truncated bootstrapping                                                          | covered (was missing)                           | 4.1 + review Ex. 8                            |
| Epsilon-greedy schedules, learning-rate/visitation conditions                                | covered (supporting detail)                     | 4.1 control section                           |
| Support/coverage/ESS                                                                         | covered (supporting detail)                     | 4.2 IS subsection                             |
| Risky-policy SARSA vs Q-learning difference                                                  | covered (supporting detail)                     | 4.1 (cliff-walk)                              |
| Bellman-operator contraction intuition, sync/async, ties, terminal conventions               | covered (supporting detail)                     | 4.1 control section                           |
| Reward shaping/PBRS, multi-objective weights, Goodhart/proxy rewards, RLHF over-optimization | covered (was covered)                           | 4.3, unchanged in substance                   |

Depth/Later/Optional items (belief-state POMDP planning, large-space approximate planning, average-reward theory, off-policy convergence proofs) intentionally not added.

## Needs external verification

- `algorithm-taxonomy.md` appendix paper titles/IDs (e.g. arXiv 2509.24203, 2510.01161, 2508.11408, 2511.10843, 2509.05735; GRPO/DeepSeek-2024, TOP-ERL ICLR 2025 Spotlight, OBAC): titles, IDs, venues, and claimed results (M2PO on 1.7B–32B, 31-environment comparison) could not be checked offline; left intact, flagged for a worker with internet.
- Training-inference-mismatch specifics (vLLM/SGLang/FSDP stacks, truncated-IS/MinPRO-style variants, long-tail pruning, MoE routing replay): general mechanism kept, vendor/framework specifics softened; verify before re-adding names or numbers.
- `reward-design.md` MuJoCo/Gymnasium reward coefficients (HalfCheetah 0.1, Ant 0.5/0.0005, BipedalWalker 130/5.0/0.00035) and BipedalWalker PBRS-guarantee remark: plausible but not checked against source.
- PopArt full author list (footnote now uses "et al.").
- Pathak/Burda (curiosity/TV), Amin et al. (2024 survey), Nair et al. (2020), Wen et al. (2024) venues/claims: not checked offline.
- CoastRunners/Minecraft anecdotes: standard lore, kept as-is.

## Sidebar additions

- Title: `4.R Summary and Exercises`, link: `/chapter04_classical_methods/review` (new page `docs/chapter04_classical_methods/review.md`; coordinator to insert after 4.3).

## Commands run and results

- `npx prettier --write` on the four chapter files: three unchanged, `review.md` reformatted; clean.
- `npx vitepress build docs`: success (client + SSR + pages rendered, ~16 s). No dead links.
- `.venv/bin/python` checks (single `-c` commands, no chaining): DP closed form $(-3.375,-1.875)$; sweep-1 $(-1.2,-1.2)$; 1000-sweep convergence exact; $\theta=10^{-6}$ loop stops at 23 sweeps; GRPO std $0.4841$, advantages $1.291/-0.775$; Q-learning 20k-episode run → greedy (R,R), values $(-2.0,-1.0)$, $Q(\cdot,L)=-4.0$ as predicted; ESS $3.39$. All match text.
- `git status --short` / `git diff --stat`: 3 modified + 1 new file, all inside scope.

## Open issues

- `review.md` is orphaned until the coordinator adds it to `docs/.vitepress/config.mjs` (I did not touch the sidebar per instructions).
- The 4.2 appendix ("Reading Terminology from Paper Titles") and the training-inference-mismatch subsection are LLM-engineering material sitting in a classical-methods chapter; I softened unverifiable specifics but left placement for the coordinator to decide (possible move to Chapter 18).
- No commits made; changes left in the working tree.
