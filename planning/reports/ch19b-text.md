# Report: ch19b-text — Chapter 19 text pass

## Summary

Reviewed all eight in-scope files completely and made targeted correctness fixes, two code-snippet bug fixes, one compact coverage addition (~380 words), and a new `review.md` summary/exercises page. The chapter's core technical content (POMDP formalization, ORM/PRM credit assignment, trajectory synthesis methods, tool-use RL, rLLM labs) is sound; most claims check out and the numeric examples I recomputed are correct. The largest structural finding is heavy cross-page duplication between `tool-use-and-trajectory.md` (19.4) and `tool-use-agents.md` (19.5), which I left intact per minimal-change rules but flag below for the coordinator.

Known suspected issues in the task brief: [x] (none recorded).

## Technical fixes

1. **Wrong chapter number, RLVR connection** — `tool-use-and-trajectory.md` and `trajectory-synthesis.md`: section heading "How Trajectory Synthesis Connects to Chapter 9" → "Chapter 15". The section discusses RLVR verifiers, which live in Chapter 15 (RLVR/GRPO), not Chapter 9 (continuous control / model-based RL). Verified by checking `docs/chapter15_grpo/rlvr.md` exists and the section body itself cites "Chapter 15".
2. **Wrong chapter number, test-time search** — same two files: TSR paragraph said test-time search "will be discussed in Chapter 12". Per the numbering table, Chapter 12 is exploration/multi-agent/hierarchical RL; test-time compute is Chapter 16. Changed to "from Chapter 16". (Left the HER/Section 8.4 reference alone — it points at carried-over robotics material I could not verify against the current outline, listed below.)
3. **Wrong chapter number in summary** — `tool-use-agents.md`: "core takeaways from Chapter 9" → "Chapter 19". The summarized points (multi-turn RL, credit assignment, RLVR, environment bottleneck) are this chapter's content.
4. **"RAGEN et al." misattribution** — `overview.md`: RAGEN is a system/project name, not an author list. Changed to "The RAGEN project reminds us…".
5. **Self-reference error** — `overview.md` (Section 19.1, the chapter's first section): "The previous sections provided the formal framework…" → "The formulas above set up the formal framework…".
6. **Misleading link text** — `overview.md`: "[PPO and Reward Models](../chapter08_ppo/ppo-bipedal-walker)" → "[PPO](…)" since the target is the PPO Bipedal Walker page (verified the file exists); reward models live in Chapter 13.
7. **Stale redirect links** — `projects.md` and `rllm-deepcoder-lab.md` pointed at the legacy split-stub `./multi-turn-rl` for the ORM-vs-PRM discussion; retargeted both to `./credit-assignment` (Section 19.3, verified title "19.3 Credit Assignment along Trajectories").
8. **Inaccurate GRPO-RoC mechanism** — `tool-use-agents.md` and `tool-use-and-trajectory.md`: the text claimed GRPO-RoC "continues to explore from [a] successful trajectory to see whether a better path can be found". The method's documented idea is resampling _among_ correct trajectories (preferring concise, tool-error-free successes) for a finer within-group signal, not continued rollout from successes. Softened both passages accordingly. Listed under verification since I could not check the paper offline.
9. **Undefined variable in reward sketch** — `compute_agent_reward` in both `tool-use-agents.md` and `tool-use-and-trajectory.md` referenced `has_format_error` with no such parameter (NameError if executed). Added `has_format_error=False` parameter; no other change.
10. **Prompt tokens leaked into the RL loss** — `projects.md` multi-tool rollout built the masked stream as a full copy of the prompt (`masked_ids = list(prompt_ids)`), so prompt tokens would receive policy-gradient loss. Changed to `[PAD_ID] * len(prompt_ids)` with an explanatory comment, matching the section's own claim that "only model decision tokens participate in the loss".
11. **Return-arity bug in lab sketch** — `projects.md` `ResearchEnvironment.step` returned a bare string for `search`/`read` actions but the caller unpacks three values (`state, answer, done`). Changed both branches to return `(new_state, "", False)`.
12. **Undefined `ground_truth` in lab sketch** — `projects.md` `rollout_one` used `ground_truth` without a parameter; added the parameter and updated the `grpo_train_step` call site to pass `gt`.
13. **Verified-correct, left unchanged**: GRPO advantage example in `rllm-deepcoder-lab.md` (rewards [0,1,0,1] → mean 0.5, population std 0.5, advantages [−1,+1,−1,+1]; recomputed); FinQA worked example (2140/42100 ≈ 5.083%, 1820/39400 ≈ 4.619%, delta ≈ 0.464pp; recomputed); discounted-return semantics in `multi_turn_rl.py`; REINFORCE softmax gradient in `tool_use_agent.py`; POMDP/throughput formulas (heuristic but correctly labeled).

## Other changes

- **Coverage addition** (`tool-use-and-trajectory.md`): new subsection "Staleness, Retries, and Recovery: Making Async Rollouts Sound" (~380 words) after the async-concurrency section, covering policy-version tracking and staleness bounds, idempotency keys and duplicate-action semantics, timeout/backoff, failed-rollout accounting by cause, checkpointing of rollout cursor/optimizer/sampler state, and utilization accounting (rollout tok/s, verifier throughput, update-utilization equation). This closes the previously missing items 22, 23, 25 and upgrades 16, 19, 21 from partial to covered (see table).
- **New page** `docs/chapter19_agentic/review.md` (title `19.R Summary and Exercises`): 8 learning outcomes, 8 key equations with one-line readings, 6 pitfalls, 8 exercises (conceptual → derivations → hand numerics → 2 coding exercises against `code/chapter19_agentic/`). Both coding solutions were executed and pass (discounted returns [0.81, 0.9, 1.0] / [0.25, 0.5, 1.0]; REINFORCE one-step p(search) 33.3% → 34.45%).
- Deliberately **not** changed: the ~600-line overlap between 19.4 and 19.5 and the legacy duplicate bodies in `projects.md`/`trajectory-synthesis.md` (merging would be a rewrite; flagged under Open issues); volatile citation counts (819/131/247) and benchmark percentages (kept, flagged for verification); arXiv-dated 2025–2026 paper claims (kept, flagged).

## Needs external verification

- rStar2-Agent: 14B, 510 RL steps on 64× MI300X, 80.6% AIME24, surpassing 671B DeepSeek-R1; GRPO-RoC resampling detail (fix #8 softens to the defensible core).
- DeepCoder-14B 60.6% LiveCodeBench ≈ o3-mini; DeepScaleR-1.5B AIME24 43.1% > O1-Preview; DeepSWE-32B 59% SWE-bench Verified; rLLM-FinQA-4B 59.7% vs Qwen3-235B 51.4% vs Gemini 2.5 Pro 60.6% (all vendor-blog figures in the rLLM labs).
- Framework claims: AReaL 2.77× speedup; slime MoE/FP8 details; Seer (arXiv:2511.14617), Relax (arXiv:2604.11554) entries; OpenResearcher 97K trajectories; Web-Shepherd 10.9pp gain at 1/10 judge cost; Magnet-14B > teacher on BFCL-v3; LoopTool 8B > 32B generator; HardGen 4B > closed-source models; TSR +15%; Memento 87.88% GAIA Pass@3.
- Paper metadata that may postdate the cutoff: ToolRL/VERLTOOL venues, ZeroTIR/Astra/HardGen/TSR/OpenResearcher/LongWriter-Zero/Writer-R1/RL-Struct 2025–2026 venues, AWM "ICML 2025", ToolDecathlon "ICLR 2026".
- HER in "Section 8.4" (referenced by trajectory-synthesis pages; could not confirm against current Chapter 8 outline — left as is).

## Coverage against the RL-12 checklist (scope: agentic RL)

| Item                                                   | Status                   | Where                                                                                         |
| ------------------------------------------------------ | ------------------------ | --------------------------------------------------------------------------------------------- |
| Verifiable rewards (RLVR)                              | Covered                  | 19.4 RLVR connection, 19.8 sandbox evaluator                                                  |
| Math/code/structured-output envs                       | Covered                  | 19.8 code env, 19.9 SQL env, 19.5 RL-Struct                                                   |
| Verifier isolation + adversarial hidden tests          | Partial                  | Sandbox + hidden tests covered; _adversarial_ test design only via hardcoding-test discussion |
| Outcome rewards, sparse terminal credit                | Covered                  | ORM sections, `multi_turn_rl.py`                                                              |
| Process rewards, step-level credit                     | Covered                  | 19.3, Web-Shepherd, ATR, CaRR, STeCa                                                          |
| Reasoning traces, hidden vs observable answers         | Partial                  | SearchR1 tags + action masking; no explicit hidden-reasoning discipline                       |
| Inference-time search, self-consistency, aggregation   | Partial                  | TSR (beam/best-of-N); Ch.16/17 own the topic                                                  |
| Train-time vs test-time compute                        | Partial                  | TSR paragraph; trade-off not quantified                                                       |
| Rejection fine-tuning, rollout distillation            | Partial                  | Rejection sampling + STeCa/Magnet; distillation vocabulary implicit                           |
| On-policy teacher-student distillation vocab           | Partial                  | LoopTool student>teacher; terms not formalized                                                |
| Reasoning length, budget control, invalid traces       | Covered                  | Efficiency + format penalties, max_turns                                                      |
| Multi-turn state, observations, tool-call actions      | Covered                  | 19.1–19.2 POMDP tables                                                                        |
| Env transitions, tool results, terminal conditions     | Covered                  | 19.2 formulation, rollout loops                                                               |
| Trajectory rewards, delayed credit                     | Covered                  | J(θ), G_t, ORM                                                                                |
| Sandbox, reset, deterministic replay, side effects     | Partial→Covered-leaning  | Sandbox/reset/replay-expiry covered; side-effect boundaries now explicit (idempotency)        |
| Tool-error/timeout/retry/duplicate-action semantics    | Covered (after addition) | New 19.4 subsection (was partial)                                                             |
| Actor/rollout-worker/verifier/learner roles            | Covered                  | AReaL, framework table, rLLM cookbooks                                                        |
| Sync vs async rollout-update loops                     | Covered                  | 19.1/19.4, AReaL                                                                              |
| Policy broadcast, rollout version, staleness           | Covered (after addition) | New 19.4 subsection (was partial)                                                             |
| Colocated vs disaggregated generation/training         | Partial                  | verl shared-GPU-pool row only; design trade-off unexplained                                   |
| Generation batching, backpressure, learner starvation  | Covered (after addition) | Utilization equation + starvation (was partial)                                               |
| Checkpoint/rollout/optimizer/sampler recovery          | Covered (after addition) | New 19.4 subsection (was missing)                                                             |
| Retry dedup, failed-rollout accounting                 | Covered (after addition) | New 19.4 subsection (was missing)                                                             |
| Prompt curriculum, filtering, evolving distributions   | Covered                  | HardGen, LoopTool GCP, RLER rubrics                                                           |
| Rollout tok/s, verifier throughput, update utilization | Covered (after addition) | New 19.4 subsection (was missing)                                                             |
| Sample/inference/training/total compute accounting     | Partial                  | 510-step/64-GPU datapoint; no accounting framework                                            |
| Reasoning/capability/safety/agent evaluation           | Covered                  | Benchmark tables, sandbox safety                                                              |
| Reward hacking, spec gaming, held-out shifts           | Covered                  | Hacking sections, held-out splits, adversarial tests                                          |
| Clean reproduction, prespecified objectives            | Partial                  | Independent splits, smoke runs; prespecification discipline implicit                          |
| Multi-agent self-play, auto curriculum (Depth)         | Covered                  | Fathom/DUETQA self-play                                                                       |
| Multimodal/embodied/production agents (Later)          | Partial                  | ReLook vision reward; embodied/production minimal                                             |

## Sidebar additions

- `19.R Summary and Exercises` → `/chapter19_agentic/review` (new file `docs/chapter19_agentic/review.md`, front matter `title: 19.R Summary and Exercises`). Coordinator to insert after 19.10 and renumber if needed. No other new pages.

## Commands run and results

- `.venv/bin/python -c …` (arithmetic verification): GRPO advantages [−1,+1,−1,+1] ✓; discounted returns [0.81,0.9,1.0] (γ=0.9), [0.25,0.5,1.0] (γ=0.5) ✓; REINFORCE one-step p=0.3445 ✓; FinQA ratios 5.083%/4.619%/0.464pp ✓; both review.md coding solutions executed and passing.
- `npx prettier --write` on all 8 touched files: 6 unchanged, `review.md` + report formatted.
- `npx vitepress build docs`: first pass succeeded but flagged a KaTeX parse error (extra closing brace) in the new update-utilization equation; fixed the brace, rebuilt clean with no KaTeX errors.

## Open issues

1. **19.4 vs 19.5 duplication**: `tool-use-and-trajectory.md` and `tool-use-agents.md` share most of their tool-use-RL and engineering sections nearly verbatim (reward design, web/code agents, SearchR1, training loop, sandboxing, frameworks, async, reward-design methodology, chapter summary). Recommend the coordinator decide: either trim 19.5 to search-only material (its sidebar title is "Search-Augmented RL") with cross-links, or re-scope the sidebar. Deliberately left alone as beyond minimal-change scope.
2. **Legacy stub pages** (`multi-turn-rl.md`, `trajectory-synthesis.md`, `projects.md` headers) duplicate live content by design; fine to keep, but `projects.md` still carries ~1600 lines of legacy labs that now overlap the rLLM cookbooks — candidate for future archival.
3. The `Seer`/`Relax` rows put arXiv IDs in a "Repository" column — cosmetic; left for the owner of that table.
