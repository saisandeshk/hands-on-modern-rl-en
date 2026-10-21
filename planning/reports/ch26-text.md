# Report: ch26-text

## Summary

Text pass over all five in-scope Chapter 26 pages plus a new `review.md`. Fixed cross-reference numbers to match the book's numbering table, fixed title/H1 mismatches against the sidebar, fixed one wrong snippet run-path, one inconsistent reward-weighting equation, two pseudocode bugs, several wrong chapter attributions, and section-numbering gaps. Softened two unverifiable quantitative claims (PRM table, "first framework" language). Created `docs/chapter26_selfplay_frontiers/review.md` (8 outcomes, 12 key equations, 6 pitfalls, 8 exercises with worked solutions, all numerics executed). `npx vitepress build docs` succeeds with no KaTeX errors; `npx prettier --write` applied to all touched files.

## Technical fixes

1. `docs/chapter26_selfplay_frontiers/rl-scaling-outlook.md` — opening best-of-5 arithmetic. Recomputed: $1-(1-0.30)^5 = 1-0.7^5 = 1-0.16807 = 0.83193 \approx 0.83$. Page's ~0.83 is correct; no change needed. Verified by hand.
2. `docs/chapter26_selfplay_frontiers/llm-multi-agent-rl/index.md` — hybrid reward weight inconsistency. First equation (line ~107) defined $R_i = \alpha R^{\text{outcome}} + (1-\alpha)R_i^{\text{process}}$ with $\alpha \approx 0.5$–$0.7$; second (line ~180) had $\alpha$ and $(1-\alpha)$ swapped. Correction: unified both on outcome weight $\alpha$, matching the trainer code (`0.6 * outcome + 0.4 * mean(process)`). Verified by reading both equations and the `FreezeRotateTrainer` code.
3. `docs/chapter26_selfplay_frontiers/llm-multi-agent-rl/index.md` — ORM/PRM attribution "Chapter 19". Outcome vs. process rewards are Chapter 17 (Section 17.1); Chapter 19 Section 19.3 applies them to trajectories. Correction: line ~109 and connections list now cite Section 17.1 (keeping the Section 19.3 application note). Verified against sidebar topics.
4. `docs/chapter26_selfplay_frontiers/llm-multi-agent-rl/index.md` — "KL constraint from the reference model in PPO (Chapter 8)". Reference-model KL is RLHF (Chapter 13); PPO's stabilizer is clipping. Correction: cite Chapter 13. Verified against chapter table.
5. `docs/chapter26_selfplay_frontiers/llm-multi-agent-rl/index.md` — "Bespoke Labs KL=0.001 (Section 19.6)". Section 19.6 is Code Interpreter RL, and the vendor constant is unverifiable offline. Correction: replaced with generic "KL / reference-model constraints (Chapters 8 and 13)". Listed under Needs external verification if the coordinator wants the claim restored.
6. `docs/chapter26_selfplay_frontiers/llm-multi-agent-rl/index.md` — "reward design (Section 19.6)". Wrong section. Correction: "(Chapters 4 and 13)".
7. `docs/chapter26_selfplay_frontiers/llm-multi-agent-rl/index.md` — ChatDev "jointly proposed by Tsinghua University and Tsinghua-affiliated startup ModelBest". Overly specific affiliation claim I could not verify offline. Correction: softened to "ChatDev is a representative work of this architecture".
8. `docs/chapter26_selfplay_frontiers/llm-multi-agent-rl/index.md` — "FlexMARL is the first end-to-end framework…". "First" is unverifiable offline. Correction: "is an end-to-end framework…".
9. `docs/chapter26_selfplay_frontiers/offline-rl/index.md` — snippet run path `docs/chapter32_selfplay/offline-rl/snippets/...`. Wrong directory (stale outline). Correction: `docs/chapter26_selfplay_frontiers/offline-rl/snippets/minimal_offline_rl_contextual_bandit.py`. Verified: file exists there; GitHub blob link on line ~113 already uses the correct path.
10. `docs/chapter26_selfplay_frontiers/offline-rl/index.md` — "PPO's KL constraint (Chapter 8)" in connections table. Same issue as #4. Correction: "PPO's trust-region / clipping stabilizer (Chapter 8)".
11. `docs/chapter26_selfplay_frontiers/offline-rl/index.md` — CQL section's TD loss written with $V_{\bar\theta}(s')$ with no comment, inconsistent with the earlier $\max_{a'}Q$ backup. This is a legitimate continuous-action simplification, not an error, so I added the qualifier "written here in a simplified form with a target value" rather than changing math. Verified IQL/CQL/DT equations against standard forms (expectile loss, advantage-weighted BC, return-to-go conditioning); all correct.
12. `docs/chapter26_selfplay_frontiers/offline-rl/index.md` — snippet output numbers. Ran the snippet twice: got `naive_q_actor return=-0.1164 support_gap=0.4857, cql_actor -0.0429/0.0096, td3bc -0.0260/0.0520` vs. page's `-0.1137/0.4819, -0.0420/0.0112, -0.0260/0.0520`. Differences are run-to-run stochasticity (unseeded minibatch sampling); ordering and magnitudes match. No change; page already says "Typical output".
13. `docs/chapter26_selfplay_frontiers/rl-scaling-outlook.md` — RLMT "Concept Source" bullets cited Ch8/Ch7 for RLVR, RLHF, GRPO, R1-Zero. Correction: RLVR long CoT → Chapter 15; RLHF preference reward → Chapter 13; GRPO → Chapter 15; R1-Zero → Chapters 15–16. Also "In Chapter 8, we completed…" → "In Chapters 8 and 13–15…". Verified against chapter table.
14. `docs/chapter26_selfplay_frontiers/rl-scaling-outlook.md` — PRM-vs-ORM accuracy table (82/85/88/90% GSM8K, 40/45/50/55% MATH) presented as exact results. These round numbers are inconsistent with the cited paper's actual reported figures and read as fabricated benchmarks. Correction: kept the table but labeled it schematic/setup-dependent, and softened "5 percentage points" to "several percentage points … in the cited setup". Flagged under Needs external verification.
15. `docs/chapter26_selfplay_frontiers/self-play-outlook/index.md` — `self_play_training_loop` used undefined `save_interval` and unimported `np`/`random`. Correction: added imports and `save_interval=10` parameter.
16. `docs/chapter26_selfplay_frontiers/self-play-outlook/index.md` — debate-training sketch overwrote `loss_a`/`loss_b` in a loop with no accumulation or optimizer step (misleading as "policy gradient update"). Correction: accumulate with `sum(-lp * reward …)` and note the optimizer step is omitted.
17. `docs/chapter26_selfplay_frontiers/self-play-outlook/index.md` — "in DeepSeek-R1's RL training, maintaining diverse historical checkpoints is likewise key". R1's math/coding RL is single-agent training against verifiable rewards, not opponent-pool training. Correction: contrast clause instead.
18. `docs/chapter26_selfplay_frontiers/self-play-outlook/index.md` — "AlphaGo self-play (Chapter 7)". Per the sidebar, AlphaGo is the Section 7.5 hands-on. Correction: "(Section 7.5)".
19. `docs/chapter26_selfplay_frontiers/alphaevolve/index.md` — "same article as Genie 3…" (wrong unit) → "same section". Strassen/48-mults/56-year/Borg-0.7%/kernel-23%/FlashAttention-32.5%/kissing-593 claims checked against solid knowledge of the AlphaEvolve report; all consistent, no change.

## Other changes

- Titles/H1s unified with the sidebar (front matter and H1 now identical on every page): 26.1 → "26.1 How Models Generate Training Data"; 26.2 → "26.2 Where Additional Compute Should Go"; 26.3 → "26.3 How Multiple LLMs Learn Together"; 26.4 → added missing front matter "26.4 How LLMs Search for New Algorithms" with matching H1; offline supplement front matter/H1 unified as "Supplement: Offline Reinforcement Learning from Historical Data".
- Section numbering gaps closed: 26.3's recipe section 26.3.5→26.3.4, infrastructure→26.3.5, MBRL 26.3.7→26.3.6; 26.2's unnumbered RLMT/RL-scaling/test-time/PRM-vs-ORM sections numbered 26.2.2–26.2.5 (26.2.1 and 26.2.6 unchanged).
- "Previous Chapter —" labels on cross-chapter connection bullets (llm-multi-agent-rl) changed to "Connection —".
- New `docs/chapter26_selfplay_frontiers/review.md`: 8 learning outcomes; 12 key equations with one-line readings; 6 pitfalls; 8 exercises (conceptual, 4 hand-computed numerics, 2 coding) each with a `::: details Solution` block. All numerics executed: best-of-5 0.83193; M-GRPO advantages [0.508, −1.397, 0.889] (population std ≈ 0.13123, sum 0); expectile mean loss 7.2; CQL penalty ≈ 1.4644; best-of-N simulation 0.8337 vs theory 0.8319 and majority 0.1632 vs 0.1631; fictitious-play trace cycles latest-only and converges pool-average 0.492→0.5. An early draft of Exercise 8's code converged to the wrong mixture; rewrote it as a two-variant (latest-only vs pool-average) demo and re-verified the exact pasted code.

## Needs external verification

- PRM-vs-ORM exact numbers (rl-scaling-outlook.md table): confirm or replace with numbers from Lightman et al. "Let's Verify Step by Step" (2023) under a stated base model and sampling budget.
- Bespoke Labs KL=0.001 claim (removed from llm-multi-agent-rl/index.md): restore only with a source.
- Recent-paper facts I kept on good-faith readings but could not verify offline: M-GRPO/Multi-Agent Deep Research (arXiv 2511.13288), MAPoRL (2502.18439), SAGE (2603.15255), MARTI (ICLR 2026), FlexMARL (2602.09578), KD-MARL (2604.06691), RLMT (2509.20357) result magnitudes, Genie 3 specs (720p, 20–24 FPS). Recommend a literature check when network access is available.
- No recorded suspected issues existed for this chapter ("none recorded"), so there was nothing to mark [x].

## Sidebar additions

- Title: `26.R Summary and Exercises`, link: `/chapter26_selfplay_frontiers/review` (one addition; coordinator to renumber/insert).
- Note: `docs/chapter26_selfplay_frontiers/offline-rl/index.md` (Supplement: Offline Reinforcement Learning from Historical Data) is not in the sidebar; linked from llm-multi-agent-rl and offline-rl pages. Coordinator decision needed: add it to the sidebar (e.g. as 26.S) or leave as an unlisted supplement.

## Commands run and results

- `npx prettier --write` on all six files: clean (review.md reformatted once, re-checked).
- `npx vitepress build docs`: succeeds. First build surfaced 3 KaTeX errors from my review.md (`^{\*}` invalid in this KaTeX setup); fixed to `^*`; final build has zero KaTeX errors.
- `.venv/bin/python docs/chapter26_selfplay_frontiers/offline-rl/snippets/minimal_offline_rl_contextual_bandit.py` (twice): outputs match the page's "Typical output" within stochasticity.
- `.venv/bin/python -c` numeric checks: M-GRPO advantages, logsumexp penalty (2.4644), best-of-N/majority theory + 20k-trial simulation, fictitious-play trace: all match solutions in review.md.
- `node scripts/check-code-links.mjs`: could not run — the sandbox rejected the `node` invocation (permission rejection). `git status`/`git diff` used for change inventory instead. Dead-link risk is low: no links were added except the review page (no outbound links) and all edited links resolve to existing in-scope targets.

## Open issues

- Sidebar decisions for the coordinator: (1) insert review page; (2) whether to list the offline-RL supplement page.
- The offline-RL supplement overlaps Chapters 10 (Offline RL) and 14 (DPO). I left its content intact as the hands-on complement; a coordinator may consider a dedup pointer to Chapter 10.
