# Report: ch21-text (Chapter 21 text pass)

## Summary

Reviewed both Chapter 21 pages completely and edited for technical correctness, cross-references, and curriculum coverage; created `docs/chapter21_deep_research/review.md` (21.R Summary and Exercises). No code, image, component, or config files touched. `npx vitepress build docs` succeeds; prettier applied to all changed/created files.

## Technical fixes

1. **BrowseComp attribution was wrong** (`deep-research-eval.md`): heading and body said "BrowseComp (Meta, 2025)" / "released by Meta". BrowseComp is an OpenAI benchmark (2025), consistent with the page's own `[^browsecomp]` footnote pointing at OpenAI. Fixed heading and body to OpenAI. Verified from solid knowledge of the benchmark's provenance.
2. **BrowseComp worked example was factually wrong** (`deep-research-eval.md`): claimed Batistuta scored Argentina's goal in the 1998 World Cup quarter-final. Argentina's scorer in the 1998 QF (Netherlands 2–1 Argentina) was Claudio López, not Batistuta. Replaced the example with a generic two-hop template that asserts no specific player/club, so it cannot be wrong.
3. **Fabricated SOTA table removed** (`deep-research-eval.md`): table of exact scores "as of June 2026" (GPT-5 38.2%, Claude Opus 4.6 35.7%, Kimi K2.5 Swarm 72.1%, Tongyi 51.4%, Human 87.5%) plus claims of "+30pp lead" and "15× more expensive". These precise numbers are unverifiable and perishable. Replaced with a qualitative statement (multi-agent systems score higher at substantially higher token cost) plus a rule to compare only on identical benchmark version/budget/attempts.
4. **GAIA attribution/year fixed** (`deep-research-eval.md`): said "(Meta + HuggingFace, 2024)". GAIA paper is Mialon et al., arXiv 2311.12950 (Nov 2023), from Meta, Hugging Face, and AutoGPT. Fixed to "(2023)" with correct institutions.
5. **"Reverse verification with random-string answers" softened** (`deep-research-eval.md`): BrowseComp has no such mechanism as described; replaced with the accurate property (hard-to-guess, precisely specified answers requiring multi-step navigation).
6. **GRPO pseudocode corrected** (`browser-rl-harness.md`, Step 3): the old snippet sorted trajectories and did a best-vs-worst pairwise `grpo_update(chosen, rejected)` — that is a DPO-style update, not GRPO. Rewrote to compute group-relative advantages $A_i=(r_i-\mathrm{mean})/\mathrm{std}$ over all group members and pass them with retrieved-token masking to the clipped update, matching Chapter 15.
7. **`parallel_rollout` inconsistencies fixed** (`browser-rl-harness.md`): `await env.reset/step` on the synchronous `BrowserEnv` defined earlier in the same page, and undefined `MAX_STEPS`. Changed to sync env calls, added `max_steps=30` parameter, and break on `done or truncated`.
8. **Duplicate figure removed** (`browser-rl-harness.md`): `tongyi_iter_research.webp` was embedded twice with two captions; removed the second occurrence.
9. **Over-precise unverifiable numbers softened** (`browser-rl-harness.md`): Tongyi "30.5B total / 3.3B active / surpasses o3 and DeepSeek-V3.1 (671B)" → qualitative parameter-efficiency claim; Memento "87.88% Pass@3" → "strong results on GAIA validation"; LLM-Explorer "37.27% improvement" → qualitative; "Search-R1++ (arXiv 2602.19526)" citation → generic "independent reproductions" note (the variance warning itself is valuable and kept).
10. **Added missing front matter** (`deep-research-eval.md`): page had no `title:` front matter though the convention (and 21.1) requires it; added `title: 21.2 Evaluation Benchmarks and Open-Source Projects`. (Also fixes [x] no known-issue list was recorded for this chapter.)

Math/code verification performed: EM/coverage/citation/redundancy/budget formulas are correct; Stage-2/3 and report reward weights each sum to 1.0; GRPO advantage example $[0,0,1,1]\to[-1,-1,1,1]$ recomputed; $0.99^{29}\approx0.747$, $0.9^{29}\approx0.047$ recomputed via `.venv/bin/python`; Ex7 redundancy code run ($3/8=0.375$); Ex8 idempotent-retry env executed (backend hit once, typed timeout obs, correct step/history counts).

## Other changes

- **Coverage additions** (`browser-rl-harness.md`, new `## Rollout Scheduling, Versioning, and Recovery`, ~550 words, three subsections): sync vs async rollout–update loops with staleness $v_\ell-v_r$; actor/rollout-worker/verifier/learner roles; policy broadcast and versioning; colocated vs disaggregated generation/training; generation batching, backpressure, learner starvation; checkpoint/recovery (policy, optimizer, sampler, rollout cursor); retry deduplication and failed-rollout accounting; tool-error/timeout/retry/duplicate-action semantics with idempotency keys; sandbox, reset, deterministic replay, side-effect boundaries; throughput accounting (rollout tokens/s, verifier throughput, update utilization) with a hand-checkable numeric example ($256/90\approx2.8$ traj/s vs $16/2=8$ traj/s verifier capacity); sample/inference/training/verifier compute split; prompt curriculum/filtering/evolving distributions.
- **New page** `docs/chapter21_deep_research/review.md`: 8 learning outcomes, 9 key equations with one-line readings, 5 common pitfalls, 8 exercises (conceptual, derivation, hand computation, 2 coding) each with a worked `::: details Solution`; all numeric answers recomputed and code solutions executed.
- Light cleanup: fixed curly-quote/whitespace drift around the de-duplicated figure; prettier reformatted one table.

## Needs external verification

No internet access; the following pre-existing paper/project claims could not be checked and are flagged, not asserted as new facts. Numbers already softened where they looked fabricated; the rest is kept with attribution but should be checked by someone online:

- Tongyi DeepResearch technical report contents (agentic mid-training + post-training, context-management paradigm, benchmark SOTA claims) [^tongyi_dr].
- DeepResearcher end-to-end real-web RL + emergent behaviors [^deepresearcher]; SFR-DeepResearch single-agent route [^sfr_dr]; rStar2-Agent [^rstar2].
- PokeeResearch-7B existence/scale; OpenResearcher 97K trajectories, `search/open/find` primitives [^openresearcher]; O-Researcher 72B models, 2.92K SFT / 10K RL examples [^oresearcher]; Fathom-DeepResearch DUETQA [^fathom_dr].
- CaRR rubric-ratio reward [^carr_dr]; Atom-Searcher ATR [^atom_searcher]; DR Tulu RLER evolving rubrics [^dr_tulu]/[^rler_dr] (note: `[^rler_dr]` duplicates `[^dr_tulu]` — same paper; coordinator may want to merge); Web-Shepherd +10.9pp claim [^web_shepherd].
- LongWriter-Zero triple reward [^longwriter]; Writer-R1 [^writerr1]; RL-Struct hierarchy [^rlstruct]; Memento memory approach [^memento].
- Search-R1 paper results (Qwen2.5-7B +41%, 3B +20% on 7 QA benchmarks) and repo contents [PeterGriffinJin/Search-R1].
- xbench-DeepSearch attribution "(Tsinghua University and HKU, 2025)" and its task-type percentages.
- STORM repo/package names, GPT Researcher usage snippet, Search-R1 install commands/versions (vLLM 0.6.3, torch 2.4.0, faiss 1.8.0) — plausible but stale-sensitive; kept as-is.

## Sidebar additions

- Title: `21.R Summary and Exercises`, link: `/chapter21_deep_research/review` (new file `docs/chapter21_deep_research/review.md` with front matter `title: 21.R Summary and Exercises` and H1 `# 21.R Summary and Exercises`).

## Commands run and results

- `npx prettier --write docs/chapter21_deep_research/browser-rl-harness.md docs/chapter21_deep_research/deep-research-eval.md docs/chapter21_deep_research/review.md` — all formatted (harness reformatted, others unchanged).
- `npx vitepress build docs` — **succeeds** (client + SSR bundles built, all pages rendered; only pre-existing chunk-size/`jinja2`-highlight warnings).
- `.venv/bin/python -c ...` — verified GRPO advantages ([0,0,1,1] → mean 0.5, std 0.5, adv [-1,-1,1,1]), discount returns ($0.99^{29}\approx0.747$, $0.9^{29}\approx0.047$), Ex7 redundancy (3/8 = 0.375), Ex8 retry env behavior. (Note: bare `grep`/piped shell commands are auto-rejected in this environment, so inspection used file tools and `.venv/bin/python`.)
- No `pytest`/repo test suites apply (no `code/` edits permitted; chapter has no named tests).

## Open issues

- `[^rler_dr]` and `[^dr_tulu]` footnotes cite the same paper; suggest merging in a follow-up.
- The `::: info Expected Outcomes` box and raw `<details>` block in `browser-rl-harness.md` build fine but are stylistically inconsistent with the `::: details` containers used elsewhere; left untouched per minimal-change rule.
- The mid-page survey material in `browser-rl-harness.md` (Deep Research Agent overview through report generation) sits between harness sections; restructuring would exceed a text-pass scope — noted for the coordinator.
- Curriculum checklist table (in-scope RL-12 items): covered — verifiable rewards, math/code/structured-output envs, verifier isolation/adversarial tests (via Ch15/21.1), outcome + sparse terminal rewards, process rewards, reasoning/hidden traces, inference-time search/aggregation (21.1 data-synthesis + 21.2 baselines), train-vs-test compute (new accounting section), rejection fine-tuning/rollout distillation (21.1 O-Researcher), on-policy distillation vocab (Ch15 link), reasoning length/budget/invalid traces (21.1 max-turns + report length control), multi-turn state/obs/tool actions, env transitions/terminal conditions, trajectory rewards/delayed credit, sandbox/reset/replay/side-effects (new section), tool-error/timeout/retry/duplicate semantics (new section + Ex8), actor/rollout/verifier/learner roles (new section), sync vs async loops (new section), broadcast/version/staleness (new section), colocated vs disaggregated (new section), batching/backpressure/starvation (new section), checkpoint/recovery (new section), dedup/failed-rollout accounting (new section), prompt curriculum/filtering/evolving distributions (new section), tokens-per-sec/verifier throughput/update utilization (new section), sample/inference/training/total accounting (new section), independent capability/safety/agent eval (21.2 four layers), reward hacking/held-out shifts (Steps 4–5 + 21.2), clean reproduction/prespecified objective (21.2 protocol + smoke-test rule). Partial — multimodal/embodied training (Later, out of scope by design). Depth — multi-agent self-play covered via Fathom/O-Researcher; automated curriculum generation partially via evolving rubrics + data flywheel.

## Coverage checklist (RL-12)

| Item                                                                      | Status                                                                             |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Reinforcement learning with verifiable rewards                            | covered (RLVR outcome rewards, CaRR, Search-R1)                                    |
| Math, code and structured-output task environments                        | covered (QA/multi-hop/citation/report tasks; code via Ch15 link)                   |
| Executable verifier isolation and adversarial hidden tests                | covered (verifier module, adversarial/trap questions, held-out sets)               |
| Outcome rewards and sparse terminal credit                                | covered (sparse $r_T$, Search-R1 0/1 reward, Ex4)                                  |
| Process rewards and step-level credit assignment                          | covered (ATR, Web-Shepherd PRM, staged reward sketches)                            |
| Reasoning traces, hidden reasoning and observable answers                 | covered (thinkpad/info/answer masking, report generation)                          |
| Inference-time search, self-consistency and answer aggregation            | covered (baselines A–C, STORM/GPT-Researcher, Memento retrieval)                   |
| Train-time learning versus test-time compute                              | covered (new throughput/accounting section; 21.2 budget success)                   |
| Rejection fine-tuning and rollout distillation                            | covered (O-Researcher SFT→RL, trajectory filtering)                                |
| On-policy teacher-student distillation vocabulary                         | covered (via Chapter 15 on-policy distillation cross-ref)                          |
| Reasoning length, budget control and invalid traces                       | covered (max-turns budgets, length rewards, truncation)                            |
| Multi-turn state, observations and tool-call actions                      | covered (21.1 observations/actions)                                                |
| Environment transitions, tool results and terminal conditions             | covered (transition $P$, termination rules)                                        |
| Trajectory-level rewards and delayed credit for agents                    | covered (trajectory record, sparse reward, Ex4)                                    |
| Sandbox, reset, deterministic replay and side-effect boundaries           | covered (new fault-tolerance section; 21.2 freeze rules)                           |
| Tool-error, timeout, retry and duplicate-action semantics                 | covered (new section + review Ex8)                                                 |
| Actor, rollout worker, reward or verifier and learner roles               | covered (new section)                                                              |
| Synchronous versus asynchronous rollout-update loops                      | covered (new section)                                                              |
| Policy broadcast, rollout policy version and staleness                    | covered (new section)                                                              |
| Colocated versus disaggregated generation and training                    | covered (new section)                                                              |
| Generation batching, backpressure and learner starvation                  | covered (new section)                                                              |
| Checkpoint, rollout, optimizer and sampler recovery                       | covered (new section)                                                              |
| Retry deduplication and failed-rollout accounting                         | covered (new section)                                                              |
| Prompt curriculum, filtering and evolving data distributions              | covered (new section + evolving rubrics)                                           |
| Rollout tokens per second, verifier throughput and update utilization     | covered (new section, worked example)                                              |
| Sample, inference, training and total compute accounting                  | covered (new section)                                                              |
| Independent reasoning, capability, safety and agent evaluation            | covered (21.2 four layers + failure atlas)                                         |
| Reward hacking, specification gaming and held-out shifts                  | covered (Steps 4–5, evolving rubrics, 21.2)                                        |
| Clean reproduction and prespecified objective, reward or system extension | covered (21.2 protocol, smoke-test rule, Ex7)                                      |
| Multi-agent self-play and automated curriculum generation (Depth)         | partial (Fathom self-play + data flywheel; no full curriculum-generator treatment) |
| Multimodal, embodied and production agent training (Later)                | missing by design (deferred; Ch22/23 own it)                                       |
