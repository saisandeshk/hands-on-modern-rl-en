# Report: ch19a-text — Chapter 19 text pass

## Summary

Reviewed and corrected the eight in-scope files in `docs/chapter19_agentic/` (plus context reads of `overview.md`, `tool-use-and-trajectory.md`, `tool-use-agents.md` and others, which were not edited). Fixed verified math/code/cross-reference errors, removed two fabricated quantitative claims, added four tight coverage insertions (sandbox retry semantics; versions/staleness/backpressure/recovery/accounting; test-time vs train-time compute; held-out shifts and hidden tests), and recorded all unverifiable 2025–2026 paper claims for external checking. No new pages were created. `npx vitepress build docs` passes.

There were no pre-recorded suspected issues in the task (`(none recorded)`), so there are no `[x]` items to mark.

## Technical fixes

1. **`credit-assignment.md` — wrong PRM return numbers in the Mini Agent Loop (recomputed).**
   The printed table claimed PRM credits `[0.656, 0.376, 0.170, -0.500]` for the failed episode. With the code's own parameters (`immediate = [0.3, 0.3, 0.3, -0.5]`, `gamma = 0.95`, backward accumulation), the correct discounted returns are `[0.427, 0.134, -0.175, -0.500]`. Verified with `.venv/bin/python`. Fixed the table.
2. **`credit-assignment.md` — removed fabricated "19x discriminability / 30pp success-rate" claims.**
   The text asserted PRM discriminability of 0.856 vs ORM 0.045 ("19 times") and a ~30-point success-rate gain. Neither number is computed anywhere in the page or cited. Removed both; replaced with an honest interpretation, including the caveat that the toy PRM rewards tool-execution success rather than semantic correctness (the buggy `π = 3` calculation still earns +0.3 immediate reward), which the old text concealed.
3. **`credit-assignment.md` — undefined `task` / `ground_truth` in the lab snippet.**
   `run_agent_loop(env, task, good_plan, ground_truth)` referenced names never defined. Added `task = ...` and `ground_truth = "40030"` (verified: `2·π·6371 ≈ 40030.17`).
4. **`credit-assignment.md` — incoherent turn-discounting equation.**
   The page defined `R = Σ γᵗ·rₜ` and claimed `γᵗ` "is not a discount for future steps," contradicting the code directly below it (standard backward accumulation). Replaced with `Gₜ = Σ_{k≥t} γ^{k-t}·rₖ` and prose matching the implementation. The numeric example (`[0.531, …, 1.000]` for `γ = 0.9`) was recomputed and confirmed correct.
5. **`credit-assignment.md` — malformed LaTeX `A_k^(s_t^{(i)})`.**
   Fixed to `A_k^H(s_t^{(i)})`.
6. **`credit-assignment.md` — GiGPO example inconsistent with its formula.**
   The formula normalizes by `F_norm` but the worked example used raw differences. Defined `F_norm` as the group standard deviation and labeled the example as showing unnormalized differences. Arithmetic (`mean = 0.58`; `+0.42/+0.15/−0.58`) verified.
7. **`credit-assignment.md` — undefined `STO-RL` term.**
   The selection guide cited "STO-RL two-stage" which appears nowhere else in the chapter. Replaced with a generic "two-stage pipeline (offline warm-start, then online refinement)."
8. **`build-agentic-training-system.md` — `code_reward` crash on `None` observation.**
   Finish-turn interactions store `"observation": None`; `obs.splitlines()` then raises `AttributeError` whenever the agent actually answers. Fixed with `or ""`.
9. **`build-agentic-training-system.md` — misleading clipping narrative.**
   `old_logprobs = new_logprobs.detach()` forces the ratio to 1, so clipping can never activate; the code is a single on-policy REINFORCE-style step with a GRPO baseline, not multi-epoch PPO. Added a design note stating this and pointing to Extension Exercise 1.
10. **`build-agentic-training-system.md` — wrong cross-reference.**
    "In Sections 19.3 through 19.5, we analyzed the architectures of frameworks such as OpenRLHF, veRL, and Relax" — but 19.3 is credit assignment and 19.5 is search-augmented RL; the framework survey lives in 19.1. Fixed.
11. **`industrial-evaluation.md` + `industrial-practice.md` — Gamma-decay formula cannot incentivize efficiency.**
    Both files gave `rᵢ = r·γ^{T−i}`, under which the final step (`i = T`) always receives `r` regardless of trajectory length, so shorter trajectories are not rewarded. Replaced with `R = r·γ^{T}` (verified: `0.95⁵ ≈ 0.77`, `0.95²⁰ ≈ 0.36`) plus a pointer to the discounted-return decomposition in 19.3.
12. **`industrial-evaluation.md` — invalid Python in `frontend_process_stats`.**
    `s.content contains "plan" or ...` is a syntax error. Fixed to valid membership tests.
13. **`multi-agent-swarm.md` — wrong Debate attribution.**
    "AI Safety via Debate by Anthropic (Irving et al., 2018)": the Debate proposal (Irving, Christiano, Amodei 2018) predates Anthropic and was not an Anthropic paper. Reworded to attribute the proposal without the wrong venue.
14. **`multi-agent-swarm.md` — wrong protocol attribution.**
    "(e.g., Anthropic A2A, …)": A2A is Google's open agent-to-agent protocol, not Anthropic's. Fixed.
15. **`multi-agent-swarm.md` — wrong OpenRLHF attribution.**
    Listed OpenRLHF as "(ByteDance)"; it is a community open-source project (verl is the ByteDance one). Fixed in the RL-training framework list (the inference-frameworks table in `overview.md`, out of scope, already lists it as "Open-source community" and was left alone).
16. **`formulation.md` — stale section name in link text.**
    "[19.1 Overview](./overview)" while the page is titled "19.1 Foundations of Agentic RL". Fixed link text.

## Other changes

- **`build-agentic-training-system.md`: new subsection "Sandbox Determinism, Side Effects, and Retry Semantics" (~250 words)** after the `SandboxEnv` design notes — deterministic replay and cache keys, per-rollout side-effect scoping and `reset()`, tool-error vs timeout vs truncation semantics, duplicate-call caching, idempotency keys, and failure-class accounting for failed rollouts.
- **`build-agentic-training-system.md`: new subsection "What the Minimal Loop Omits" (~250 words)** after the production-gaps table — policy version tags and staleness budgets, bounded queues with backpressure and learner-starvation avoidance, joint checkpointing (weights + optimizer + sampler cursor + rollout log) with retry deduplication, and throughput/compute accounting (rollout tokens/s, verifier throughput, update utilization; sample/inference/training/total split).
- **`build-agentic-training-system.md`: curriculum/filtering bullet** in the Trainer design notes (DAPO-style dynamic sampling, evolving prompt distributions).
- **`credit-assignment.md`: "Train-time learning versus test-time compute" tip box (~150 words)** — Best-of-N/beam search/self-consistency at inference vs folding search into gradient updates, with the practical distill-after-search recipe.
- **`industrial-evaluation.md`: held-out shifts + adversarial hidden tests paragraph (~130 words)** after the benchmark-exploitation paragraph — held-out environment slices and hidden/mutated verification tests, tied to the F2P/P2F split.
- `agent-data-swe-smith.md` and `extended-readings.md`: read fully; no confident errors found, so no edits (see verification section).

## Needs external verification

No internet access was available, so every specific 2025–2026 paper/result claim below is recorded as-is for a coordinator with access to verify; nothing here was changed on suspicion alone:

- `credit-assignment.md`: SALT (Li et al., EACL 2026), GiGPO (`2505.10978`, NeurIPS 2025), HGPO (`2602.22817`, ICLR 2026), Group-Graph PO (`2606.22995`), SPA-RL (`2505.20732`), AgentPRM (`2502.10325`) three-phase loop and hypothesis, PAIR (`2605.17877`), Web-Shepherd (`2505.15277`, "NeurIPS 2025 Spotlight", "+10.9% on GPT-4o-mini at 1/10 cost"), ARPO entropy-spike claim (`2507.19849`), IGPO/DR-Venus (`2604.19859`), AEM (`2605.00425`), RefGRPO reflection-gap claim (`2606.14211`), StepPO (`2604.18401`), Turn-PPO (`2512.17008`), AT²PO (`2601.04767`), survey `2604.09459`. The page's framing ("four families developed in 2025–2026") and all paper-level result claims need a reference sweep.
- `multi-agent-swarm.md`: Anthropic internal figures (90.2% speedup, +18–32% on SWE-bench Verified; 2.7x error-chain rate); Kimi K2.5 all specifics (2026.01, `2602.02276`, 32 agents, 12M trajectories, 68.3%/72.1%, 280K vs 18K tokens); Step 3.7 Flash Advisor Mode figures (62.4%, 52K tokens); "Anthropic's Debate system enforces a devil's-advocate agent"; OpenResearcher entry.
- `industrial-practice.md` / `industrial-evaluation.md`: Amazon 72-sample AppWorld figures (39.2% → 72%, vs Claude Sonnet 3.7/4.0); Bespoke Labs composite-reward/gibberish and KL=0.001 + ref-refresh-every-100-steps claims; Alibaba WebShaper/AgentFounder/rLLM descriptions; LinkedIn Attention-Sink backward-pass story; Salesforce "250x memory efficiency, 120B MoE on 16 H200s"; Kimi context-manager ">50 turns" ablation.
- `agent-data-swe-smith.md`: `swesmith` package API names (`swesmith.build_repo`, `bug_gen.lm.generate_bug_lm`, `bug_gen.procedural.generate_bug_procedural`, `validate.validate_task_instances`, `issue_gen.generate_issue`), the four-strategy yield table (1.93%, ~30%, ~20%, 47.04%, ~25%, ~15%, ~35%, 13.18% vs 2.46%), the 50,000+/128-repo/128 GB figures, and "SWE-agent-LM-32B 40.2% Pass@1".
- `formulation.md`: Agent-R1 sliding-window-vs-append-only on GSM8K claim (odd pairing of a context-management result with a single-turn math benchmark); Search-R1 `state_masking=true` detail; Echo/PaW follow-ups.
- `extended-readings.md`: all ~120 entries with 2025–2026 arXiv IDs need a link/venue sweep; it also contains two different works abbreviated SAGE (multi-agent vs dialogue), which is confusing but possibly legitimate.
- `build-agentic-training-system.md`: DeepSeekMath equations (3)/(4) citation for the per-token ratio/KL estimators; Qwen2.5-0.5B runnable-on-CPU premise.

## Sidebar additions

None (no pages added).

## Coverage against the curriculum checklist (RL-12, scope-relevant)

| Item | Verdict | Where |
| --- | --- | --- |
| RL with verifiable rewards | covered | swe-smith RLVR loop; `code_reward`; ORM/RLVR discussion |
| Math, code, structured-output envs | covered | code tasks; BFCL/ACEBench; sympy/SWE-smith |
| Verifier isolation, adversarial hidden tests | partial → improved | sandbox + Docker rows; F2P/P2F; new hidden-tests paragraph (adversarial mutation still thin) |
| Outcome rewards, sparse terminal credit | covered | ORM section; lab |
| Process rewards, step-level credit | covered | PRM + GiGPO/HGPO/SPA/AgentPRM/IGPO/StepPO/Turn-PPO |
| Reasoning traces / hidden reasoning / observable answers | covered | Search-R1 four-tag scheme; action mask |
| Inference-time search, self-consistency, aggregation | partial → improved | Best-of-N; new test-time-vs-train-time box (self-consistency voting not detailed) |
| Train-time vs test-time compute | covered (added) | new tip box in 19.3 |
| Rejection fine-tuning, rollout distillation | partial | rejection sampling lives in 19.4 (other worker's scope); RFT term unused in scope files |
| On-policy teacher-student distillation vocabulary | partial | distillation idea appears only in 19.4 LoopTool (out of scope) |
| Reasoning length, budget control, invalid traces | covered | gamma-decay (fixed); truncation filtering; `max_turns` |
| Multi-turn state, observations, tool-call actions | covered | formulation (POMDP, action mask) |
| Env transitions, tool results, terminal conditions | covered | formulation + `SandboxEnv` done/observation |
| Trajectory-level rewards, delayed credit | covered | credit assignment; turn discounting (fixed) |
| Sandbox, reset, deterministic replay, side-effect boundaries | covered (added) | new sandbox subsection (live-env replay inherently partial) |
| Tool-error, timeout, retry, duplicate-action semantics | covered (added) | new sandbox subsection |
| Actor, rollout worker, reward/verifier, learner roles | covered | four-component build (Policy/Env/Worker/Trainer) |
| Sync vs async rollout-update loops | covered | overview + build system + rLLM |
| Policy broadcast, version, staleness | covered (added) | new versions/staleness paragraph |
| Colocated vs dislocated generation/training | partial | framework tables + async discussion; no dedicated cost analysis |
| Generation batching, backpressure, learner starvation | covered (added) | new backpressure paragraph + gaps table |
| Checkpoint, rollout, optimizer, sampler recovery | covered (added) | new recovery paragraph |
| Retry dedup, failed-rollout accounting | covered (added) | failure-class logging + dedup paragraph |
| Prompt curriculum, filtering, evolving distributions | covered (added) | trainer bullet (+ HardGen/LoopTool in 19.4) |
| Rollout tok/s, verifier throughput, update utilization | covered (added) | new accounting paragraph |
| Sample/inference/training/total compute accounting | covered (added) | same paragraph |
| Independent reasoning/capability/safety/agent eval | partial | capability eval thorough; safety eval only via hallucination penalties |
| Reward hacking, spec gaming, held-out shifts | covered (added) | hacking sections + benchmark-exploit + new held-out paragraph |
| Clean reproduction, prespecified objective/reward/extension | partial | deterministic envs, decontamination, versioned rubrics; no explicit prespecification protocol |
| Multi-agent self-play, auto curriculum (Depth) | covered | swarm + Ch26 pointer; AgentFounder |
| Multimodal, embodied, production agents (Later) | partial | extended-readings index only, as expected |

## Commands run and results

- `.venv/bin/python -c ...` (arithmetic verification): PRM returns `[0.4270625, 0.13375, -0.175, -0.5]`; ORM-negative first step `-0.857`; `0.95⁵ ≈ 0.774`, `0.95²⁰ ≈ 0.358`, `0.9⁶ = 0.531`; GiGPO mean `0.577`; `2π·6371 ≈ 40030.17`. All match the corrected text.
- `npx prettier --write` on the six edited files: clean (two files reformatted).
- `npx vitepress build docs`: success, no dead-link errors.
- `git status --short`: only the six intended scope files modified.

## Open issues

1. `industrial-practice.md` (19.6) and `industrial-evaluation.md` (Supplement) overlap heavily — the latter states it merged the former plus benchmarks, so most scenario content now exists twice. Recommend the coordinator pick one canonical home (or an explicit "read this, not that" pointer). I fixed the shared Gamma-decay bug in both but did not deduplicate.
2. `rllm-deepcoder-lab.md:863` (out of scope) links "Section 19.3" to `./multi-turn-rl`; the ORM-vs-PRM discussion is in `./credit-assignment`. Flagged for the owning worker.
3. The 2026-dated paper ecosystem in `credit-assignment.md` and `extended-readings.md` needs an online reference sweep (listed above) — I corrected only what arithmetic and first-principles reasoning could settle.
