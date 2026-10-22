# Report: ch09-text (Chapter 9 text pass + review response)

## Summary

Chapter 9 text pass over `docs/chapter09_continuous_control/` (`ddpg.md`, `td3-sac.md`, `model-based.md`, `search-world-models.md`, new `review.md`). The previous worker commit fixed the two flagged compounding-error sums, rewrote the broken MBPO reset loop, corrected the HalfCheetah 15000+ claim to ~12000, fixed the temperature-objective sign ($\arg\max\to\arg\min$) and the soft-value closed form ($V^\pi\to V^*$), and added tanh-squashing, pessimism/decision-relevant-error, fair-evaluation, random-shooting-baseline, and review/exercise material. This follow-up adjudicated all 12 reviewer findings (7 errors, 5 doubts), accepted 8 fully or in substance, rejected 2 on scope/task-rule grounds, and partially accepted 2. All in-scope fixes below were applied, prettier is clean, and `npx vitepress build docs` succeeds.

Note: the reviewer correctly observed that `planning/reports/ch09-text.md` did not exist on the branch (only the review file had been committed). This file recreates the required report, including the original fixes reconstructed from the diff plus the review response.

## Technical fixes

### Original pass (reconstructed from diff `124c5b0^..124c5b0`, verified by re-derivation)

- `model-based.md` (compounding sums): Case 3 $L=1.1,T=50$ corrected $1174\epsilon\to1164\epsilon$. Verified: $(1.1^{50}-1)/0.1 = 1163.909$. Mark `[x]`.
- `model-based.md` (MBPO short-rollout numbers): 5-step $5\epsilon\to5.1\epsilon$ ($5.101\epsilon$), 100-step $270\epsilon\to170\epsilon$ ($170.481\epsilon$ for $L=1.01,T=100$); 20-step $22\epsilon$ confirmed ($22.019\epsilon$). Mark `[x]`.
- `model-based.md` (MBPO reset loop): removed the never-true `if rollout_step % K_short == 0` reset (loop ran `range(K_short)`, so the condition only held at step 0 with the already-real state) and rewrote it as sample-a-real-state → roll `K_short=5` steps → stop; next rollout restarts from another real state. Mark `[x]` (reset-condition half of the notes item).
- `td3-sac.md` (HalfCheetah claim): "above 15000" → "around 12000 after a million steps", matching the ASCII sketch (SAC plateau at the 12000 row) and published 1M-step SAC curves (~11k–12.5k). Mark `[x]`.
- `td3-sac.md` (temperature objective): $\alpha^*=\arg\max\to\arg\min$ of $\mathbb{E}[-\alpha\log\pi-\alpha\mathcal{H}_0]$ (Haarnoja et al. 2018, Eq. 17). Verified sign: minimizing decreases $\alpha$ when entropy is above target.
- `td3-sac.md` (soft value): $V^\pi(s)\to V^*(s)$ in the log-partition closed form; the integral form holds for the optimal soft value, not an arbitrary $\pi$.
- `td3-sac.md`: new "Bounded Actions: Tanh Squashing" subsection (formula, code, $u=0.5$ numeric check: $\tanh=0.4621$, Gaussian log-density $-0.7258$, correction $+0.2402$, total $-0.4856$ — recomputed, all correct for $a_{\max}=1$).
- `model-based.md`: new "Pessimism, Decision-Relevant Error, and Distribution Shift" and "How to Evaluate a Model-Based Method Fairly" subsections; new random-shooting degenerate-case block (CEM baseline).
- `review.md`: created `9.R Summary and Exercises` with outcomes, key equations, pitfalls, 8 exercises with worked solutions (all numerics re-derived: Ex. 3: 1163.9/23.3/2.0/25→0.98/26→1.09; Ex. 4: PUCT(A)=0.7684 vs PUCT(B)=0.7714; Ex. 6: $-0.7616/-1.4189/+0.8676/-0.5514$ — recomputed, correct).

### Review-response fixes (this follow-up)

- `td3-sac.md` + `review.md` (review Error 1, accepted): squashed log-probability now includes the action-scale Jacobian, $\log\pi(a\mid s)=\log\mu(u\mid s)-\sum_i\log a_{\max}-\sum_i\log(1-\tanh^2(u_i))$, with the note that the term vanishes for $[-1,1]$ bounds; the code snippet subtracts $\log(\text{action\_max})$ per dimension. Verified by change-of-variables ($da/du=a_{\max}(1-\tanh^2u)$).
- `td3-sac.md` (review Error 2, accepted in substance): earlier stochastic-policy/actor-loss/temperature snippets are now explicitly labeled "Simplified: no tanh squashing" with cross-references to the Bounded Actions subsection, and the soft-Q bonus box states the target must use the squashed corrected $\log\pi$. Chose labeling over a full rewrite to keep the pedagogical ramp (unsquashed first, squashing as the correction) intact.
- `model-based.md` (review Error 3, accepted in substance): replaced the wrong "uncertain trajectories vary wildly across members so few survive elite selection" mechanism (TS1 uses one member per trajectory, so there is no within-trajectory cross-member filtering) with the correct account: one-member-per-trajectory coherent futures plus MPC replanning; explicit uncertainty penalties attributed to successors (MOPO). Rejected the reviewer's suggested $V(s)=\mathbb{E}_iV_i-\beta\,\mathrm{std}_iV_i$ wording because PETS has no horizon-end value function — that would have introduced a new fabrication.
- `model-based.md` (review Doubt 5, accepted): MuZero attribution tightened to reward (dynamics function) vs policy/value (prediction function).
- `review.md` (review Error 6, accepted): Exercise 7 counts fixed — 50,000 gradient updates (one per env step), $50{,}000\times256\approx12.8$M samples, each transition seen ~256x on average — replacing "hundreds of thousands of times".
- `review.md` (review Error 7, accepted): 1M-step sketch now reported per algorithm (SAC ~12000, TD3 ~8000, DDPG ~4000), matching the ASCII rows.
- `review.md` (review Doubt 1, accepted): Exercise 8 softened to a hypothesis-to-test over seeds (fixed-$0.2$ SAC can still do well on Pendulum) and the sketch now records `train/entropy_coef` over time with a callback instead of printing one final value.
- `review.md` (review Doubt 2, accepted): Exercise 7 now qualifies HalfCheetah-v4 vs Pendulum-v1 fallback (PPO comparatively competitive on Pendulum) and seed variance.
- `review.md` (review Doubt 4, accepted): Exercise 3 closing reworded to "the quantitative reason to keep model rollouts short (MBPO uses 5)".
- `td3-sac.md` (review Doubt 3, accepted as qualifier): performance claim now reads "in published 1M-step HalfCheetah results … around 12000", since the repo's own script trains only 100k steps.

## Other changes

- Cross-references checked against the numbering table: Chapter 6 (policy gradient), `chapter05_dqn`, `chapter07_actor_critic`, `chapter08_ppo`, Chapters 10/17/20 links in `search-world-models.md` — all point at the right topics; no stale-outline numbers found in scope. No garbled-text or leftover-editorial cleanup was needed beyond the above.

## Coverage against the curriculum checklist (RL-5 in-scope + RL-7)

| Item                                                                                                                                   | Status       | Where                                             |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------------------------------------------------- |
| Continuous action spaces and control constraints                                                                                       | covered      | §9.2 squashing, §9.1                              |
| Deterministic policy gradients and DDPG vocabulary                                                                                     | covered      | §9.1                                              |
| Twin critics, delayed updates and TD3 vocabulary                                                                                       | covered      | §9.2                                              |
| Maximum-entropy RL and Soft Actor-Critic                                                                                               | covered      | §9.2                                              |
| Automatic entropy-temperature tuning                                                                                                   | covered      | §9.2 + review Ex. 8                               |
| Known vs learned transition/reward models; system ID; data collection                                                                  | covered      | §9.3 intro/Dyna/MBPO loop                         |
| One-step and multi-step prediction objectives                                                                                          | covered      | §9.3 pessimism subsection                         |
| Open-loop rollout and closed-loop replanning                                                                                           | covered      | §9.3 (MPC)                                        |
| Model Predictive Control                                                                                                               | covered      | §9.3 (PETS/CEM)                                   |
| Random shooting and CEM planning                                                                                                       | covered      | §9.3 (incl. new random-shooting block)            |
| Dyna-style real and imagined experience                                                                                                | covered      | §9.3                                              |
| Model ensembles and epistemic uncertainty                                                                                              | covered      | §9.3                                              |
| Aleatoric uncertainty and stochastic dynamics                                                                                          | covered      | §9.3                                              |
| Uncertainty penalties and pessimistic planning                                                                                         | covered      | §9.3 (MOPO penalty)                               |
| Compounding model error over horizon                                                                                                   | covered      | §9.3 (Lipschitz recursion + numbers)              |
| Decision-relevant vs average prediction error                                                                                          | covered      | §9.3 (MuZero paragraph)                           |
| Policy-induced distribution shift                                                                                                      | covered      | §9.3 (data aggregation)                           |
| Model exploitation and adversarial plans                                                                                               | covered      | §9.3                                              |
| Rollout horizon and model-bias trade-offs                                                                                              | covered      | §9.3 (MBPO + theory bonus)                        |
| Model-Based Policy Optimization                                                                                                        | covered      | §9.3                                              |
| Latent-state dynamics and representation learning                                                                                      | covered      | §9.4 (world models)                               |
| World-model policy learning vocabulary                                                                                                 | covered      | §9.4                                              |
| Dreamer-style latent imagination (Depth)                                                                                               | covered      | §9.4                                              |
| MCTS (Optional)                                                                                                                        | covered      | §9.4                                              |
| Planner, model, reward and policy error attribution                                                                                    | covered      | §9.3 fair-evaluation subsection                   |
| Known-dynamics oracle and simple-controller baseline                                                                                   | covered      | §9.3 (random shooting + oracle)                   |
| Model-free baseline under matched real interactions                                                                                    | covered      | §9.3 + review Ex. 7                               |
| Imagined vs real sample and compute accounting                                                                                         | covered      | §9.3 (25k calls/action)                           |
| One-step, multi-step, uncertainty and control evaluation                                                                               | covered      | §9.3                                              |
| Data aggregation after policy distribution changes                                                                                     | covered      | §9.3                                              |
| Fallback and replanning under uncertainty                                                                                              | covered      | §9.3 (MPC replanning)                             |
| Exploration/empirical-method items (RL-5 rest: $\epsilon$, optimism, counts, curiosity, HER, seeds, IQM, budgets, tuning, sim-to-real) | out of scope | belong to Chapters 12 / Appendix A per task brief |
| High-dimensional generative world-model research (Later)                                                                               | out of scope | correctly deferred                                |

## Needs external verification

- Exact published HalfCheetah 1M-step SAC numbers vary by codebase/seed (commonly ~11k–12.5k); "around 12000" is deliberately imprecise — no specific paper table is cited, so nothing to verify, and none is claimed.
- PETS TS1 description ("one randomly chosen member per trajectory") follows the standard reading of Chua et al. 2018; fine-grained step-vs-trajectory sampling variants exist across reimplementations, but the corrected sentence no longer depends on that distinction.

## Sidebar additions

- `9.5 Summary and Exercises` → `/chapter09_continuous_control/review` (file created with front matter/H1 `9.R Summary and Exercises` per the task brief; coordinator to renumber to 9.5 and add the sidebar entry — I am not permitted to edit `docs/.vitepress/config.mjs`).

## Commands run and results

- `npx prettier --write` on the three changed docs files: clean (unchanged, already formatted).
- `npx vitepress build docs`: succeeds (build complete, no dead-link errors).
- No code in `code/` was touched (out of scope); numeric checks were done with `.venv/bin/python` one-liners (compounding sums, tanh/log-prob, PUCT) — all match the text.

## Open issues

- Review Errors 4–5 ask for edits outside my scope (`docs/.vitepress/config.mjs` sidebar entry + `9.R→9.5` rename; `planning/content-review-notes.md` checkbox marking). Both are rejected as scope violations (worker rules + task brief), not as technical disagreements; the sidebar addition is listed above and the notes entries (model-based sums/reset, td3-sac 15000→12000, plus unrecorded $\arg\min$ and $V^*$ fixes) are ready for the coordinator to mark `[x]`.
- Stray/foreign files noted by the reviewer (`MUJOCO_LOG.TXT` at repo root, re-touched `sac-real-robot.gif` and `backgammon.jpg` binaries) are outside my scope; leaving them for the coordinator to drop or revert.
- Exercise solutions that depend on MuJoCo/SB3 runs (Ex. 7–8) state expected patterns, not committed run logs; the text now flags seed/environment variance explicitly.

## Review response

- Error 1 (tanh log-prob drops $\log a_{\max}$; review.md repeats it): accepted — added $-\sum_i\log a_{\max}$ to the equation, snippet, key-equation entry, and Ex. 6 note.
- Error 2 (unsquashed snippets contradict new squashing section): accepted — labeled earlier snippets "Simplified (no squashing)" with cross-refs and fixed the soft-Q bonus note.
- Error 3 (PETS elite-selection mechanism backwards): accepted in substance — replaced with TS1-coherent-futures + MPC account; rejected the suggested $V(s)$ formula as a PETS fabrication.
- Error 4 (review.md orphan / `9.R` label / sidebar): rejected on scope grounds — task brief mandates `9.R` and forbids config edits; sidebar addition listed above for the coordinator.
- Error 5 (`content-review-notes.md` unchecked): verified the underlying fixes are correct but rejected the file edit as out of scope; exact `[x]` text provided above for the coordinator.
- Error 6 (Ex. 7 "hundreds of thousands" counts): accepted — now 50,000 updates / ~12.8M samples / ~256x reuse per the chapter script.
- Error 7 (Ex. 7 "SAC/TD3 near 12000"): accepted — now SAC ~12000, TD3 ~8000, DDPG ~4000 per the sketch.
- Doubt 1 (Ex. 8 fixed-0.2 mechanism + single-value logging): accepted — softened to testable hypothesis, added time-series callback.
- Doubt 2 (Ex. 7 outcome not backed / Pendulum fallback): accepted — qualified by environment and seed variance.
- Doubt 3 ("around 12000" more precise than repo supports): accepted as qualifier — now "in published 1M-step results".
- Doubt 4 (Ex. 3 causal leap from T=26 to K=5): accepted — reworded to "reason to keep rollouts short".
- Doubt 5 (MuZero dynamics/prediction attribution loose): accepted — split into dynamics-function (reward) vs prediction-function (policy/value) targets.
