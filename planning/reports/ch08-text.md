# Report: ch08-text (Chapter 8 text pass)

## Summary

Read all six Chapter 8 pages end to end, re-derived the equations, and recomputed every numeric table by hand and by execution. Fixed one duplicated block, three wrong numbers/outputs, one wrong clip-table (advantage-sign case), one misattributed method (PPG), two wrong expansions/names (DAC, DreamerV3 date), one loose theory claim (DAgger regret), and one dangling list. Added three tight coverage treatments (reward-to-go, advantage normalization, categorical-vs-Gaussian + tanh-squash note) and created `docs/chapter08_ppo/review.md` with outcomes, key equations, pitfalls, and 8 fully solved exercises. Build passes.

Suspected-issue disposition (from the task brief):

- [x] `ppo-math.md` duplicated implementation-overview block — **confirmed, removed** the second copy.
- [x] `ppo-bipedal-walker.md` random-baseline 3-lines-vs-4-lines — **confirmed, fixed** by adding the missing `std` print.
- [x] `rl-long-horizon-planning.md` "DAC (Deep Adversarial Control)" — **confirmed, fixed** to Discriminator-Actor-Critic.
- [x] `rl-long-horizon-planning.md` DreamerV3 "Nature 2023" — **confirmed, fixed** to Nature 2025 (preprint arXiv 2023).
- [x] `rl-long-horizon-planning.md` DAgger regret "O(1/T) vs O(T)" + lead-in-less list — **confirmed, both fixed**.
- [x] Cross-chapter numbers checked against the sidebar table (details in Technical fixes).

## Technical fixes

1. **File:** `docs/chapter08_ppo/ppo-math.md` — **wrong claim:** for $A_t = -2$, rows $r_t = 1.3$ and $r_t = 2.0$ said `torch.min` picks the clipped term (`surr2`). **Correction:** $\min(-2.6, -2.4) = -2.6$ and $\min(-4.0, -2.4) = -4.0$: `min` picks the _unclipped_ term with nonzero recovery gradient; also rewrote the over-general summary sentence. **Verified by:** direct arithmetic, cross-checked against the (correct) four-case analysis in `trust-region-clipping.md` lines 256–263.
2. **File:** `docs/chapter08_ppo/ppo-math.md` — **wrong claim:** advantage worked example gave $A_L \approx -22.4$, $A_R \approx 22.3$. **Correction:** $83.3 - 106.7 = -23.3$ and $130 - 106.7 = 23.3$. **Verified by:** recomputation ((80+100+70)/3 = 83.33, (120+140+130)/3 = 130, $V = 106.67$).
3. **File:** `docs/chapter08_ppo/gae-reward-model.md` — **wrong claim:** printed `compute_gae` output `[0.6203 0.4951 0.4159 0.3222 0.2]` / returns `[0.7203 ...]`. **Correction:** `[0.7611 0.7051 0.6465 0.4801 0.2]` / `[0.8611 0.9051 0.9465 0.9801 1.0]`, plus "0.62 at $t=0$" → "0.76", plus $\lambda=0$ column made $\gamma$-exact ($[0.098, 0.097, 0.195, 0.292, 0.2]$). **Verified by:** running the page's exact function in `.venv/bin/python`.
4. **File:** `docs/chapter08_ppo/ppo-bipedal-walker.md` — **wrong claim (mismatch):** code printed 3 lines, sample output showed 4 (extra `Standard deviation: 12.6`). **Correction:** added `print(f"Standard deviation: {np.std(returns):.1f}")`. **Verified by:** reading code vs output.
5. **File:** `docs/chapter08_ppo/trust-region-clipping.md` — **wrong claim:** "PPO-PPG (Phasic Policy Gradient) suggests using a larger $\varepsilon$ early…". **Correction:** adaptive-$\varepsilon$ is a generic heuristic; PPG is about separating policy/value training into phases, not adaptive clipping. **Verified by:** confident knowledge of Cobbe et al. 2020 (PPG = phasic separate training + auxiliary epoch, no adaptive $\varepsilon$).
6. **File:** `docs/chapter08_ppo/rl-long-horizon-planning.md` — **wrong claim:** "DAC (Deep Adversarial Control)". **Correction:** DAC = Discriminator-Actor-Critic (Kostrikov et al. 2018).
7. **File:** `docs/chapter08_ppo/rl-long-horizon-planning.md` — **wrong claim:** "Dreamer-V3 (published in Nature in 2023)". **Correction:** Nature 2025 (preprint arXiv 2301.04104, 2023).
8. **File:** `docs/chapter08_ppo/rl-long-horizon-planning.md` — **wrong claim:** DAgger "cumulative regret decreases at $O(1/T)$, far better than $O(T)$ of BC". **Correction:** no-regret reduction; error compounding drops from BC's quadratic-in-$T$ to near-linear-in-$T$.
9. **File:** `docs/chapter08_ppo/ppo-math.md` — **wrong cross-ref:** "The RLHF training in Chapter 8 records both `clip_fraction` and `approx_kl`" (RLHF is Ch. 13). **Correction:** "The SB3 PPO run in Section 8.1 logs both…". Other in-scope refs verified correct against the sidebar: Ch. 4 (DP/MC/TD, reward design), §7.1 advantage, Ch. 6 REINFORCE, Ch. 14 DPO, Ch. 15 RLVR, §8.2/§8.3 internal links.

## Other changes

- Removed the verbatim-duplicated six-question implementation overview in `ppo-math.md` (kept one copy).
- Added a causality/reward-to-go paragraph in `ppo-math.md` App. A.3 (~110 words).
- Added an "Advantage Normalization" subsection in `ppo-math.md` (centering/scaling, once-per-batch, never normalize value targets; ~170 words).
- Added categorical-vs-Gaussian contrast + tanh-squash log-prob correction note in `ppo-bipedal-walker.md` §8.1.5 (~150 words).
- Added a lead-in sentence for the orphaned three-item world-model list in `rl-long-horizon-planning.md` Stage 3.
- Softened "In 2016, Schulman again proposed GAE" to "(arXiv 2015, ICML 2016)".
- Created `docs/chapter08_ppo/review.md`: 7 outcomes, 8 key equations with one-line readings, 6 pitfalls, 8 exercises (conceptual, derivations, hand numerics, 2 coding with fixed-trajectory asserts) with complete `details`-block solutions; all numerics verified (clip cases, KL $\approx 1.18$, GAE endpoints, advantage $\pm 23.3$, loss $-0.4$ via executed torch code).

### Coverage against RL-4 checklist

| Item                                                       | Status                                                             |
| ---------------------------------------------------------- | ------------------------------------------------------------------ |
| Trajectory objectives and expected return                  | covered                                                            |
| Likelihood-ratio and score-function identities             | covered                                                            |
| Policy-gradient theorem at working depth                   | covered                                                            |
| REINFORCE and Monte Carlo policy gradients                 | covered                                                            |
| Causality and reward-to-go                                 | partial → covered (added A.3 paragraph)                            |
| Constant and state-dependent baselines                     | covered                                                            |
| Baseline unbiasedness and variance reduction               | covered                                                            |
| Learned value baselines and critic targets                 | covered                                                            |
| Actor-critic data and update dependencies                  | covered                                                            |
| One-step and n-step advantage estimates                    | covered (TD/MC/GAE span)                                           |
| Generalized Advantage Estimation                           | covered in depth                                                   |
| $\gamma$ and $\lambda$ bias-variance behavior              | covered                                                            |
| Entropy regularization and exploration effects             | covered                                                            |
| Old and new policy log probabilities                       | covered                                                            |
| Importance probability ratios and direction                | covered                                                            |
| PPO clipped surrogate with advantage-sign cases            | covered (bug fixed)                                                |
| Approximate KL as diagnostic and stop signal               | covered                                                            |
| Clip fraction, ratio and policy-update diagnostics         | covered                                                            |
| Value loss, value clipping and loss coefficients           | covered                                                            |
| Advantage centering, scaling and minibatch leakage         | missing → covered (added subsection)                               |
| Multiple epochs and induced off-policy drift               | covered                                                            |
| Categorical and Gaussian action distributions              | partial → covered (added contrast)                                 |
| Tanh-squashed actions and log-prob correction              | missing → covered (added note)                                     |
| Vectorized rollouts and policy-version identity            | covered                                                            |
| Fixed-trajectory loss, ratio, GAE and gradient tests       | missing in text → covered via review Exercises 7–8                 |
| Policy collapse, critic failure, stale-log-prob diagnosis  | covered                                                            |
| Multi-seed return/entropy/KL, sample-efficiency evaluation | partial (single-seed curves; Exercise 8.2 adds multi-seed harness) |
| Trust-region optimization and natural gradients (Depth)    | covered at working depth                                           |
| Full monotonic-improvement proofs (Later)                  | out of scope, not added                                            |

## Needs external verification

- `ppo-game-benchmark.md` benchmark numbers and links were left untouched (no internet): Snake 5×5/10×10 scores, 2048 win rates, Tetris 3829, Breakout 398±33, Procgen CoinRun 8.31, CarRacing stages, Mario 31/32, Retro 4692 vs 7438, Crafter 4.6%/15.6%/50.5%, MiniHack NetHackScore ~700 — and the `PokeRL paper https://arxiv.org/abs/2604.10812` URL (April-2026-dated ID could not be checked offline).
- `rl-long-horizon-planning.md` MBPO bound "$O(\varepsilon \cdot \gamma^k/(1-\gamma))$" reads suspicious (error shrinking in $k$) but was left as-is; needs the Janner et al. 2019 reference checked by someone with access.
- BipedalWalker "solved = 300 over 100 episodes", "60–90 min on CPU", InstructGPT "~40 annotators": plausible, left unchanged.

## Sidebar additions

- Title `8.R Summary and Exercises`, link `/chapter08_ppo/review` — place after `8.6 Hands-on: Long-Horizon Tasks and Planning` (coordinator to renumber). No `config.mjs` edits made.

## Commands run and results

- `npx prettier --write` on the 6 touched/created files: clean (2 files reformatted, 4 unchanged).
- `npx vitepress build docs`: success, 16.9s, no dead links.
- `.venv/bin/python -c` recomputation of the chapter's `compute_gae` (yielded the corrected outputs) and of the review Exercise 7 torch loss ($-0.4$ both batches, within float32 tolerance of the 1e-6 assert).
- No repo test suites are named for this text task and `code/` was not edited.

## Review response

| #   | Verdict                                       | Action                                                                                                                                                                                                              |
| --- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Accepted                                      | Rewrote `ppo-math.md` summary sentence: `min` picks clipped term only above $1+\varepsilon$ for $A_t>0$ (unclipped below $1-\varepsilon$); $A_t<0$ is the mirror image. Removed "asymmetric" framing.               |
| 2   | Accepted                                      | Qualified all three residual "gradient vanishes" sentences (`ppo-math.md` gradient-control bullet, Step-3 summary, `trust-region-clipping.md` viz caption) to same-direction overshoot; pointed at four-case table. |
| 3   | Accepted                                      | Fixed MBPO bound in `rl-long-horizon-planning.md` to $O(\epsilon(1-\gamma^k)/(1-\gamma))$ — grows with $k$ and saturates, consistent with $k\approx5$.                                                              |
| 4   | Accepted                                      | Fixed Dreamer lead-in: world model (items 1–2, Stage 1) vs policy model (item 3, trained Stage 2 in imagination).                                                                                                   |
| 5   | Accepted                                      | Fixed arithmetic display in `ppo-math.md` and `review.md` Ex.6 to exact-fraction form $250/3-320/3=-70/3\approx-23.3$.                                                                                              |
| 6   | Accepted                                      | Fixed `review.md` Ex.7 parenthetical: $r=0.2<1-\varepsilon$ with $A_t<0$ picks clipped term with zero gradient (same-direction stop), not a recovery signal. Numbers/loss unchanged.                                |
| 7   | Accepted in principle, no edit (out of scope) | `config.mjs` edits are forbidden for this worker; sidebar entry stays listed under "Sidebar additions" for the coordinator, and the `8.R` label was the task-brief-mandated title (`8.R Summary and Exercises`).    |
| 8   | Accepted                                      | Added explicit prerequisite comment to `review.md` Ex.8 harness: re-run/save one model per seed to produce the `seed{seed}` paths.                                                                                  |
| 9   | Accepted                                      | Rephrased reward-to-go paragraph: full-trajectory return $G_{0:T}$ carries the past-reward noise; $G_t$ above already is the causally trimmed reward-to-go.                                                         |
| 10  | Accepted                                      | Fixed entropy wording in `ppo-bipedal-walker.md`: values start near $-5.8$ and rise to $-3.5$ (no "dropping from 0").                                                                                               |
| 11  | Accepted in principle, no edit (out of scope) | `planning/content-review-notes.md` is outside this task's editable scope; coordinator please mark the five Chapter 8 items `[x]` with the corrected statements.                                                     |
| D1  | Accepted (softened)                           | Advantage-normalization now says "a common practice (this chapter's code, CleanRL, TRL/veRL)" with note that SB3 defaults `normalize_advantage=False` and baselines PPO2 used per-minibatch.                        |
| D2  | Accepted                                      | DAC/AIRL sentence reworded: AIRL addresses reward ambiguity, DAC reuses discriminator signal off-policy — "built on GAIL's discriminator", not "stability fixes".                                                   |
| D3  | Accepted                                      | DAgger sentence now names the quantity (expected imitation cost $\approx\epsilon T^2/2\to O(T)$).                                                                                                                   |
| D4  | Accepted                                      | Gaussian notation reconciled: $\sigma_\theta$ state-independent, SB3 default `use_expln=False` noted.                                                                                                               |
| D5  | No change                                     | Std-dev line kept; four numbers mutually consistent (range $\approx3.7\sigma$ over 50 episodes), presented as one run's output.                                                                                     |

## Open issues

- Out-of-scope stale chapter numbers I could not edit — suggested fixes for the owning workers: `preface/introduction.md:234` "Chapters 8–9" for DPO → Chapters 13–14; `preface/introduction.md:248` "Chapter 8" for Agentic RL → Chapters 19–22; `preface/env-setup.md:61,221` "Chapters 7–10" for LLM fine-tuning → Chapters 13–18; `chapter12_exploration_marl_hierarchical/marl.md:188` "PPO … since Chapter 7" → Chapter 8.
- Multi-seed evaluation remains a text gap (only addressed in a review exercise); a future pass could add error bars to the §8.1 curves if the code worker logs them.
