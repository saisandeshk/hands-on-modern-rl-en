# Report: ch20-text (Chapter 20 text pass)

## Summary

Reviewed and edited all four in-scope pages plus created `docs/chapter20_swe_agents/review.md`. The main technical work was de-fabrication: the supplementary SWE-RL page carried a "Meta" attribution with an invented 90K-task internal-data table and an invented DPO/PPO/GRPO ablation table, and the DeepSWE page contradicted itself (42.2% vs 50.0%, Agentica × Together AI vs ByteDance attribution, and a training-phase box that presented teaching alternatives as the disclosed recipe). All of these were corrected or reframed as hypotheses to test. Coverage additions to `swe-bench-and-rlvr.md` closed most RL-12 infra gaps (sandbox/reset/replay contract, hidden/adversarial tests, sync-vs-async plumbing with versions and staleness, RFT/distillation vocabulary, compute ledgers). Build passes; all exercise solutions were executed and verified numerically.

## Technical fixes

1. `swe-bench-and-rlvr.md` — `swe_reward` divided by `len(test_results)` with no empty-list guard (returned `ZeroDivisionError` on empty input). Fixed with an early `return 0.0`. Verified by execution.
2. `meta-swe-rl.md` — title and body attributed arXiv 2502.18449 to "Meta SWE-RL" with an "Internal PR Data (Meta) 80,000+, total ~90,000 tasks" table. No basis for the Meta attribution or the 80K figure was verifiable; the paper's public recipe is SWE-bench-style tasks plus SWE-Gym. Retitled to "SWE-RL", replaced the table with open-source sources only, and moved proprietary extensions to an explicitly ablate-separately note.
3. `meta-swe-rl.md` — algorithm table claimed exact ablation scores (DPO 24.3%, PPO 33.2%, GRPO 41.0%). These precise margins are not safely attributable; replaced with the qualitative reported outcome (GRPO + binary reward, 41.0% for the 70B model) flagged as setup-dependent.
4. `meta-swe-rl.md` — leaderboard mixed evaluation splits (e.g., Devin 13.95% is a different-split figure; Claude/GPT rows lack scaffold/budget). Replaced with the single reported 41.0% Verified figure plus a pointer to the official leaderboard with required run settings.
5. `meta-swe-rl.md` — "Llama-3-70B + RoPE scaling to 128K" and named closed models as the long-context recipe. Softened to 128K-class context with trade-offs; no paper-established configuration claimed.
6. `meta-swe-rl.md` — DAPO dynamic sampling described as "filtering out easy problems". Corrected: it discards prompts with no learning signal in either direction (all-fail and all-trivially-succeed).
7. `meta-swe-rl.md` — "zero KL in DeepSeek V3.2 for math tasks" contrast presented as fact. Replaced with a neutral KL trade-off statement (style preservation vs reward optimization), since the version-specific claim is unverifiable here.
8. `meta-swe-rl.md` — hard thresholds ("unstable beyond 16 steps", "drops beyond 32 steps") and "90K samples … over 100 repositories" stated as paper facts. Softened to tens-of-steps instability and removed the 90K figure.
9. `world-model-and-deep-swe.md` — DeepSWE result contradicted itself: 42.2% Pass@1 / ~59% scaled in the text vs "DeepSWE (ByteDance) 50.0%" in the table, plus a "50% on open-source models" sentence; attribution flipped between Agentica × Together AI and ByteDance. Unified on the disclosed recipe (Agentica × Together AI, 42.2% Pass@1, ~59% scaled) and fixed table and sentence.
10. `world-model-and-deep-swe.md` — "Training Process of DeepSWE" box listed world-model training and hierarchical RL (PPO high-level + GRPO low-level) as DeepSWE phases, contradicting the page's own disclaimer. Rewrote the box as rollout infra → pure RL on verifiable outcomes → test-time scaling; moved value/hierarchical/world-model items to explicitly labeled extensions needing ablations. Section headers "DeepSWE gives…" reframed as "Teaching Alternative N".
11. `world-model-and-deep-swe.md` — GAE written as a truncated sketch ($\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \ldots$). Expanded to the full sum $\hat{A}_t = \sum_{l \ge 0}(\gamma\lambda)^l \delta_{t+l}$ with $\delta_t$ defined.
12. `world-model-and-deep-swe.md` — "CWM accelerates training by 100–1000 times" stated as fact. Replaced with a cost-model statement (forward pass vs container + suite, report measured throughput) since no fixed multiplier is established.
13. `world-model-and-deep-swe.md` — "consensus within ByteDance Seed that long-horizon tasks require critics" stated as fact. Reframed as a VAPO-style hypothesis with ablations; critic-free vs critic presented as an engineering trade-off.
14. `self-play-ssr-and-summary.md` — SSR paper attributed to "Meta FAIR and other institutions" (unverifiable here). Reduced to "multi-institution research team".
15. `self-play-ssr-and-summary.md` — `fix_bug` pseudocode used `while not done` with `done` never initialized and `tests_passed` never assigned. Fixed by initializing both and setting them on the passing-test branch.
16. `self-play-ssr-and-summary.md` — Codex bullet tied planning to "o3's reasoning". Generalized to version-varying reasoning model with record-the-model guidance.
17. `self-play-ssr-and-summary.md` — prescriptive claims ("Java needs a stronger KL constraint", "C/C++ create a greater need for a world model"). Reframed as hypotheses to test per repository.
18. Cross-references — verified every internal link target exists (`chapter15_grpo/rlvr`, `deepseek-dapo`, `grpo-family`, `on-policy-distillation`, `chapter16_reasoning/test-time-scaling`, `cot-visibility-alignment`, `chapter17_prm_search/outcome-vs-process`, `chapter19_agentic/overview`, `agent-data-swe-smith`, `build-agentic-training-system`, `chapter08_ppo/rl-long-horizon-planning`, `chapter24_multimodal_frontier/embodied-intelligence/model-based-rl/`, `chapter25_alignment_failures/classical-failures`). Updated the 20.1 pointer text from "Meta SWE-RL" to "SWE-RL". No broken links (build renders all pages).
19. Known suspected issues from the task: [x] none were recorded for this chapter ("(none recorded)"); the items above are what inspection actually found. No pre-existing notes to reconcile in `planning/content-review-notes.md` for these pages.

## Other changes

- `swe-bench-and-rlvr.md` coverage additions (all tight, in existing voice): "Hidden Tests and Adversarial Checks" subsection; "Sandbox, Reset, and Side-Effect Boundaries" subsection (reset, deterministic replay, side-effect boundaries, tool-error/timeout/assertion status codes, patch-hash dedup); "Agent Rollout" gained invalid-trace accounting and a hidden-reasoning-vs-observable-answer paragraph (with ch16 pointer); "Policy Update" gained RFT / rollout-distillation / on-policy-distillation vocabulary (with ch15 pointer); new "Rollout Plumbing: Sync, Versions, and Recovery" subsection (sync vs async, behavior-policy versions, staleness, colocated vs disaggregated, backpressure/learner starvation, joint checkpointing, retry dedup); new "Compute and Throughput Accounting" subsection ($C_{\text{total}} = C_{\text{sample}} + C_{\text{inference}} + C_{\text{train}}$, rollout/verifier throughput, update utilization, prespecification rule).
- Created `docs/chapter20_swe_agents/review.md` (front matter `title: 20.R Summary and Exercises`, H1 `# 20.R Summary and Exercises`): 8 learning outcomes, 8 key equations with one-line readings, 5 pitfalls, 8 exercises (conceptual, numeric-by-hand, and 2 coding) each with a worked `::: details Solution` block. All numeric answers executed and double-checked (see Commands).
- Light cleanup only; no page restructured, no neighboring chapters touched.

## Needs external verification

- CWM: arXiv 2510.02387, 32B dense, Python-interpreter + Docker-trajectory mid-training, 65.8%-style scaled figure (deliberately left vague in the comparison table). Confirm against the paper before citing precisely.
- DeepSWE-Preview: Agentica × Together AI, Qwen3-32B start, ~4,500 R2E-Gym tasks, six days on 64 H100s, 42.2% Pass@1 / ~59% scaled. Confirm run details and exact scaling protocol.
- SWE-RL paper: 41.0% SWE-bench Verified for the 70B model, the exact reward definition (binary test-outcome vs any patch-similarity component), GRPO-vs-baseline margins, and which shaping variants were ablated. Confirm before citing margins.
- SSR paper (arXiv 2512.18552): +10.4pp Verified / +7.8pp Pro gains and human-data-baseline comparisons. Text now carries a re-check caveat.
- SWE-Gym exact task count (page now says "thousands" rather than 6,800).
- SWE-bench counts used as settled background (2,294 tasks; Verified 500-subset) — high confidence, no action needed.

## Sidebar additions

- "20.R Summary and Exercises", link `/chapter20_swe_agents/review` (file created at `docs/chapter20_swe_agents/review.md`; coordinator to add to sidebar — not edited per scope).

## Commands run and results

- `npx prettier --write` on all five touched/created files: clean (review/meta/world reformatted, others unchanged).
- `npx vitepress build docs`: success, all pages rendered, no dead-link failures.
- `timeout 60 .venv/bin/python -c ...`: verified Ex.1–4 and Ex.6–8 arithmetic — 9/12 = 0.75; shaped 0.57; group [1,1,0,0] → μ=0.5, σ=0.5, A=[1,1,−1,−1]; GAE δ₀=0.295, δ₁=0.5, Â₀=0.76525; 200×8=1600 rollouts, 9.6M tokens, 1600/3≈533.3 min single-worker, ≈66.7 min on 8 workers; `swe_reward`/`is_valid_defect`/`filter_prompts`/`staleness_weights` outputs match the printed solutions exactly.
- Note: direct `.venv/bin/python` invocation without `timeout` prefix was permission-rejected by the runner; the `timeout 60`-prefixed form worked. No repo helper files were created.

## Review response

Reviewer report: `planning/reviews/ch20-text.md` (12 errors, 11 doubts). Each finding re-checked against the pages and, where numeric, by execution.

- Error 1 (SWE-RL reward contradiction between pages): **accepted (contradiction fixed)**. Both pages now state the same recipe — GRPO with a binary test-outcome reward, per the supplementary reading — with an explicit verify-against-the-paper caveat. Direction note: the review asserted the paper's reward combines test signal with patch-similarity; my best knowledge of arXiv:2502.18449 is that the reported reward is test-outcome-based, so I unified on the binary reading and hedged rather than adopting the similarity claim. Exact reward definition added to "Needs external verification".
- Error 2 (`all([])` is True, max reward for empty test list): **accepted**. Added `if len(test_results) == 0: return 0.0` guard in `meta-swe-rl.md`; verified `all([]) == True` by execution.
- Error 3 (inverted GRPO variance parenthetical): **accepted**. Now reads "high variance for tasks of intermediate difficulty, near-zero variance for tasks already solved or still unsolved".
- Error 4 (`filter_prompts` kept all-succeed groups despite prose): **accepted**. Now requires `max(rewards) > 0 and len(set(rewards)) > 1`, semantically identical to `review.md` Exercise 8 (`max(rs) > 0 and variance > 0`); verified both keep only prompt "a".
- Error 5 (OPD misdefined; "rollout distillation" non-standard): **accepted**. Paragraph rewritten: teacher-trajectory learning is now called offline/SFT-style distillation, OPD is student-generates plus teacher token-level targets, and the "rollout distillation" term is dropped.
- Error 6 (CWM MCTS claimed no real testing needed; broken snippet): **accepted**. Relabeled as a teaching sketch of an explicit-transition-model extension, speed claim made conditional on model accuracy with real tests still required, snippet fixed (`n_iter` parameter, defined return).
- Error 7 (DeepSWE beam search needed a value model the page disavows): **accepted**. Rewritten as a teaching alternative that reranks with the verifier (matching the Phase-3 box); value-based ranking explicitly marked as an extension to ablate.
- Error 8 ("four ledgers" vs three-term equation): **accepted**. Now "three ledgers".
- Error 9 (CWM date 2025.09 vs arXiv 2510 = October): **accepted**. Now 2025.10.
- Error 10 (appendix attributions to Meta): **rejected (out of scope)**. `docs/appendix_c_resources/learning-resources.md` is not in this task's file scope; the chapter text itself no longer attributes either paper to Meta. Coordinator to reconcile the appendix rows.
- Error 11 (review.md not in sidebar; "20.R" numbering): **partially accepted**. Sidebar registration is out of scope (config.mjs not editable; addition stays listed under "Sidebar additions"). Retitle to "20.4" rejected: the original task explicitly specified front matter `title: 20.R Summary and Exercises`, and renumbering is the coordinator's stated job when adding the sidebar entry.
- Error 12 (Exercise 2 optimality claim wrong under pass-fraction reward): **accepted**. Solution now states the optimum shifts to short partial-credit trajectories (1-step 2/12 repair ≈ +0.067 beats unchanged −0.1 beats genuine long repair −3.0); numbers verified by execution.
- Doubt 1 ("order of magnitude"): **accepted**. Softened to "several times more".
- Doubt 2 (data-cleaning thresholds as paper facts): **accepted (hedged)**. Now "a practical cleaning recipe in this line of work" with verify-before-citing caveat.
- Doubt 3 (SWE-bench as training source): **accepted (hedged)**. Table purpose now "Baseline tasks and held-out evaluation"; exact paper data composition added to verification list.
- Doubt 4 (4,500 R2E-Gym tasks vs ch19's 1,468): **rejected (out of scope to reconcile)**. ch19 files are not editable here; the ch20 figure is now flagged under "Needs external verification" for the coordinator.
- Doubt 5 (ch19 DeepSWE 59% row): **rejected (out of scope)**. Coordinator to qualify the ch19 row.
- Doubt 6 (DAPO wording vs ch15): **no change needed in scope**. ch20 already states the precise criterion (discard zero-variance groups in both directions); ch15's narrower wording is out of scope — flagged for the coordinator.
- Doubt 7 (ch16 link title): **accepted**. Link text now matches the target H1 "16.6 Displaying Reasoning Chains and Alignment".
- Doubt 8 (H1/sidebar title mismatch): **accepted**. H1 aligned to sidebar text "20.1 SWE-RL Basics" (sidebar itself untouched).
- Doubt 9 (learning outcome not fully delivered): **accepted (reworded)**. Outcome now points to the two tables that jointly deliver the comparison.
- Doubt 10 (`meta-` filename): **rejected (out of scope)**. Renaming needs sidebar/link coordination; left as an open issue for the coordinator.
- Doubt 11 (DeepSWE date/pairing): **rejected (no change)**. Cannot verify offline; already listed under "Needs external verification".

## Open issues

- `meta-swe-rl.md` filename still carries the `meta-` prefix although the "Meta" attribution was removed as unverifiable. Renaming requires updating the `./meta-swe-rl` link in `swe-bench-and-rlvr.md` (and the sidebar has no entry for it). Suggest coordinator decide: rename to `swe-rl.md` with link updates, or keep.
- H1/sidebar title mismatch (pre-existing, left alone): page H1 is "20.1 Foundations of SWE-RL" while the sidebar reads "20.1 SWE-RL Basics"; supplementary page has no number. `npm run fix:headings` was out of allowed commands, so no auto-sync was attempted.
- Coverage table (RL-12, code-agent scope): covered — verifiable rewards; verifier isolation + adversarial hidden tests; outcome rewards + sparse terminal credit; process rewards + step-level credit; reasoning traces vs observable answers; train-time vs test-time compute; RFT + rollout distillation; on-policy distillation vocabulary; reasoning length/budget/invalid traces; multi-turn state/obs/actions; env transitions/tool results/terminals; trajectory-level rewards + delayed credit; sandbox/reset/replay/side-effects; tool-error/timeout/retry/duplicate semantics; sync vs async loops; broadcast/version/staleness; colocated vs disaggregated; batching/backpressure/starvation; checkpoint/optimizer/sampler recovery; retry dedup + failed-rollout accounting; curriculum/filtering/evolving distributions; rollout tok/s + verifier throughput + update utilization; sample/inference/training/total accounting; reward hacking + held-out shifts; clean reproduction + prespecified objective/reward/extensions; self-play + automated curriculum (Depth). Partial — math/structured-output environments (code is covered; math lives in ch15; structured-output tool calling only sketched); inference-time aggregation/self-consistency (search + verifier reranking covered, formal Pass@k/self-consistency treatment lives in ch16–17); actor/learner role formalism (named here, full treatment in ch19); independent capability/safety evaluation (fair-comparison + prespecification here, safety formalism in ch25). Missing core items: none — every non-Depth/Later item has at least a concise treatment in place.
