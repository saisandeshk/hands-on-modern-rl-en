# Report: ch16-text — Chapter 16 text pass

## Summary

Reviewed all six pages in `docs/chapter16_reasoning/` end to end, re-derived every equation and recomputed every numeric example by execution. Fixed both suspected items from `planning/content-review-notes.md` (one confirmed wrong number removed, one off-by-one fixed), plus two further verified errors (an overstated sampling probability and a nonstandard `:::output Output` marker). Added two tight content boxes (distillation vocabulary in 16.4; hidden-test/sandbox hygiene in 16.2) to cover checklist gaps, and created the new `review.md` summary-and-exercises page (8 exercises, all numeric solutions verified by running code). `npx vitepress build docs` passes.

## Technical fixes

1. **File:** `docs/chapter16_reasoning/emergence-and-o1.md` (AIME table + §"o1 Turns Training into a Product Interface" paragraph)
   - **Wrong claim:** GPT-4o consensus@64 on AIME 2024 stated as "about 56%".
   - **Correction:** Removed the specific figure; the cell now reads "rises with sampling but stays well below o1's single attempt", and the paragraph no longer cites 56%, adding that the exact consensus figure depends on the aggregation rule (majority vote vs best-of-N vs reranking) so only same-protocol numbers are comparable.
   - **Verification:** The 56% figure coincides with o1-preview's widely reported single-sample AIME score (56.7%) and is implausibly high for GPT-4o majority-vote consensus; it could not be settled offline, so per standards the overly specific detail was softened rather than replaced with another number. The retained ordering (o1 single attempt ≈74% ≫ GPT-4o even with sampling) is well established. **[x]**

2. **File:** `docs/chapter16_reasoning/test-time-scaling.md` (§"When to Stop Adding Compute")
   - **Wrong claim:** "the marginal gain of the 16th candidate is down to only 0.6 percentage points".
   - **Correction:** Changed "16th" to "17th".
   - **Verification:** The section's own definition (and table) is that the column shows $P(N{+}1)-P(N) = p(1-p)^N$; at $p=0.2$, $0.2\times0.8^{16}\approx0.0056\approx0.6\%$ is the gain of adding the 17th candidate. Arithmetic rechecked by execution. **[x]**

3. **File:** `docs/chapter16_reasoning/r1-zero-pure-rl-reasoning.md` (§"A Sampled Group Needs Both Right and Wrong Answers")
   - **Wrong claim:** with 6/100 base-model success and $G=8$, "the probability that a group contains both right and wrong answers is very high".
   - **Correction:** Replaced with the computed value: about $1-0.94^8\approx39\%$ of groups mix right and wrong — enough for a steady signal, but not "very high".
   - **Verification:** Computed by execution ($1-0.94^8 = 0.3904$; the $0.06^8$ term is negligible). **[x]** (new finding, not in the notes file)

4. **File:** `docs/chapter16_reasoning/r1-zero-pure-rl-reasoning.md` (Python verification block)
   - **Wrong claim (markup):** nonstandard `:::output Output` container.
   - **Correction:** Changed to `:::output` to match the convention used in `chapter03_value_functions/value-bellman.md:983`.
   - **Verification:** Grep showed `:::output` is the repo convention; the page was the only other user of the marker. Build passes.

5. **Other numeric claims rechecked and found correct (no change):** coverage table in 16.3 ($0.8^2,0.8^4,0.8^8,0.8^{16}$ rows and all marginal gains); $1-(1-0.12)^{64}\approx99.97\%$ in 16.1; weighted-success example $0.05\times0.01+0.1\times0.4\approx0.04\to0.14$ in 16.1; GRPO Examples 1–2 ($\mathrm{std}=\sqrt{0.1875}\approx0.433$, advantages $(0.577,-1.732)$) and all-signal-vanishes table ($p=0.9/0.7/0.5/0.3/0.1$, $G=8$) in 16.2; $\lambda=0.0002$ cost-penalty table in 16.5; Kimi $\lambda_i$ worked example ($0.5/0.17/-0.5$) in 16.4.

## Other changes

- **`hybrid-thinking.md`**: added subsection "Distillation Vocabulary: Teacher, Student, and On-Policy Data" (~250 words) after "Model weight merging": teacher/student definitions, off-policy vs on-policy (rollout) distillation, the sequence-level cross-entropy distillation loss with one-line reading, and a 100-trace filtering example motivating rejection fine-tuning before distillation. Covers the checklist's "rollout distillation" and "on-policy teacher-student distillation vocabulary" items.
- **`r1-zero-pure-rl-reasoning.md`**: added a short paragraph on hidden-test holdouts and sandboxed execution (time/memory/network limits) to the verifier section, with pointers to `../chapter15_grpo/verl-code-sandbox` and Chapter 19. Covers the checklist's "executable verifier isolation and adversarial hidden tests" at concept level without duplicating Chapter 15.
- **New file `docs/chapter16_reasoning/review.md`**: front matter + H1 `16.R Summary and Exercises`; 7 learning outcomes; 7 key equations with one-line readings; 6 common pitfalls; 8 exercises (conceptual, derivations, compute-by-hand, 2 coding) each with a full solution in a `details` block. Every numeric answer was executed: Ex4 ($83.2\%$, $5.0\%$), Ex5 ($N=19$; closed form $18.43$), Ex6 (NumPy advantages match page), Ex8 ($\lambda_i = 0.5/0.167/-0.5$).
- **Cross-references:** all "Chapter N / Section N.k" references checked against the numbering table and link targets verified to exist (`../chapter19_agentic/overview`, `../chapter15_grpo/grpo-practice-and-mechanism`, `../chapter15_grpo/deepseek-dapo`, Section 17.5 → `chapter17_prm_search/inference-time-search.md` whose H1 is `# 17.5 Inference-Time Search`). No stale-outline numbers found in scope. No garbled text or leftover editorial notes found.

## Coverage against the RL-12 checklist (in-scope items)

| Item                                                                                                                                                                                                                                   | Status                                 | Where                                                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Reinforcement learning with verifiable rewards                                                                                                                                                                                         | covered                                | 16.2 (correctness/format reward, R1-Zero loop)                                                               |
| Math, code and structured-output task environments                                                                                                                                                                                     | covered                                | 16.1 (Codeforces), 16.2 (math/code/logic + verifier)                                                         |
| Executable verifier isolation and adversarial hidden tests                                                                                                                                                                             | partial → now covered at concept level | 16.2 new paragraph; tooling in Ch15/Ch19                                                                     |
| Outcome rewards and sparse terminal credit                                                                                                                                                                                             | covered                                | 16.2 (binary outcome reward; indirect credit discussion)                                                     |
| Process rewards and step-level credit assignment                                                                                                                                                                                       | partial (pointer; Ch17 owns it)        | 16.2 format/parse boundary, 16.5–16.6 PRM pointers                                                           |
| Reasoning traces, hidden reasoning and observable answers                                                                                                                                                                              | covered                                | 16.1 special-token separation, 16.6 three layers                                                             |
| Inference-time search, self-consistency and answer aggregation                                                                                                                                                                         | covered                                | 16.3 (best-of-N, majority-vote box, tree search → 17.5)                                                      |
| Train-time learning versus test-time compute                                                                                                                                                                                           | covered                                | 16.1 division of labor, 16.3 Snell et al.                                                                    |
| Rejection fine-tuning and rollout distillation                                                                                                                                                                                         | covered                                | 16.2 (R1 rejection sampling), 16.4 shortest-rejection + new distillation box                                 |
| On-policy teacher-student distillation vocabulary                                                                                                                                                                                      | partial → now covered                  | 16.4 new distillation box                                                                                    |
| Reasoning length, budget control and invalid traces                                                                                                                                                                                    | covered                                | 16.4 (budget, wrap-up, truncation), 16.5 (adaptive + hard caps)                                              |
| Multi-turn state, tool actions, trajectory rewards, sandboxing, rollout infra (actor/rollout-worker/learner, batching, backpressure, staleness, colocated vs disaggregated, checkpoint/retry accounting, curriculum, token accounting) | out of scope for Ch16                  | owned by Ch18/Ch19 (industrial training, agentic loops); Ch16 touches only the reasoning-budget slice (16.5) |
| Independent reasoning/capability/safety/agent evaluation; reward hacking, spec gaming, held-out shifts                                                                                                                                 | partial                                | reasoning-eval hygiene in 16.3/16.6; full treatment in Ch17/Ch19/Ch25                                        |
| Multi-agent self-play, automated curriculum (Depth); multimodal/embodied/agent training (Later)                                                                                                                                        | out of scope                           | correctly deferred                                                                                           |

## Needs external verification

- **Claude adaptive-thinking API surface** (`adaptive-thinking.md`): `thinking: {"type": "adaptive"}`, `output_config: {"effort": ...}`, the `budget_tokens`→adaptive deprecation timeline, "Claude 4.6/4.7" versioning, and thinking-block `signature` semantics. Signature/redaction fields exist in the real Claude API, but the `adaptive` type and `effort` levels could not be confirmed offline (no internet). Left intact; recommend the owner verify against current Anthropic docs.
- **2026 Claude Constitution** reference (`adaptive-thinking.md`): could not confirm offline.
- **OpenAI Instruction Hierarchy year (2025)** (`adaptive-thinking.md`): the hierarchy work is real but the year could not be confirmed offline.
- **arXiv:2511.18397 faithfulness study** (`cot-visibility-alignment.md`): identifier could not be confirmed offline.
- **Kimi k1.5 "60.8 Pass@1 on AIME 2024 with 3,272 tokens"** (`hybrid-thinking.md`): attributed to the report; exact operating point could not be confirmed offline.
- **DeepSeek V3.1 / Qwen3 / o3 / o4-mini / Deep Think / o1-benchmark figures** (16.1, 16.4, 16.3): all match solid public knowledge (endpoints, thinking budget, long2short families, IMO gold) but post-cutoff details (e.g., o3 release month) should get a final check at review time.

## Sidebar additions

- **16.R Summary and Exercises** — link `/chapter16_reasoning/review` (new file `docs/chapter16_reasoning/review.md`; coordinator to renumber/insert after 16.6).

## Commands run and results

- `.venv/bin/python -c ...` — verified $1-0.94^8\approx0.3904$, coverage/marginal numbers ($83.2\%$, $5.0\%$), $N=19$ threshold ($94.6\%$ vs $95.4\%$), $\lambda_i=0.1667$, GRPO advantages $(1.732,-0.577\times3)$. All pass.
- `npx prettier --write <5 files>` — review.md reformatted; the four edited pages unchanged (already conformant).
- `npx vitepress build docs` — **success** (client + SSR bundles, all pages rendered, no dead-link errors).

## Open issues

- None blocking. The `review.md` page is intentionally absent from `docs/.vitepress/config.mjs` (workers may not edit it); the coordinator must add the sidebar entry. The "Chapter 16 Review" recap at the end of `cot-visibility-alignment.md` (16.6) overlaps slightly with the new review page by design (in-page recap vs standalone exercises) — no action needed unless the coordinator wants it trimmed.

## Review response

- **Error 1 (hidden tests causal claim) — accepted.** Rewrote the sentence in `r1-zero-pure-rl-reasoning.md` to state that the hidden set does not change the training reward but makes evaluation an unbiased estimate exposing overfit/gaming of the visible tests; also corrected the sandbox parenthetical (subprocess-isolated reward pipeline vs real sandbox, full setup in Ch19).
- **Error 2 (on-policy distillation redefinition) — accepted (most serious).** Rewrote the box in `hybrid-thinking.md` as "Distillation Vocabulary: Teacher, Student, and Rejection Fine-Tuning": verifier-filtered cross-entropy training is now labeled off-policy SFT/SeqKD on filtered data (RFT), and OPD is defined consistently with §15.7 (student rollouts scored token-by-token by a teacher, reverse-KL) with a pointer to `chapter15_grpo/on-policy-distillation`. Updated `review.md` outcome 7 and the distillation-loss reading to match.
- **Error 3 ("teacher solves 100 / 30 wrong") — accepted.** Changed to "the teacher generates 100 traces for 100 problems, of which 30 end in a wrong answer."
- **Error 4 (sandbox attribution to Ch15) — accepted.** Reworded to say the subprocess-isolated reward pipeline is wired up in Ch15 (which explicitly disclaims it as a security sandbox) and a real sandbox plus the industrial agent setup are in Ch19.
- **Error 5 (Llama-3-8B base model) — accepted.** Changed to Llama-3.1-8B-Instruct per Snell et al. (arXiv:2408.03314); the Path A vs Path B claim is unchanged.
- **Error 6 (sidebar entry) — rejected (out of scope).** Adding `{ text: '16.7 Summary and Exercises', link: '/chapter16_reasoning/review' }` to `docs/.vitepress/config.mjs` requires editing a file my task scope explicitly forbids; the entry remains listed under "Sidebar additions" for the coordinator.
- **Error 7 (16.R numbering) — rejected (task requirement).** The original task explicitly mandates front matter/H1 `16.R Summary and Exercises` and states the coordinator will renumber on sidebar insertion; renaming to 16.7 unilaterally would violate that instruction.
- **Error 8 (content-review-notes.md) — rejected (out of scope, noted for coordinator).** The fix log belongs in `planning/content-review-notes.md` (mark both Ch16 bullets `[x]`; AIME cell removed, 17th-candidate index with `0.2·0.8^16 ≈ 0.6%`), but that file is outside my editable scope, so I leave the edit to the coordinator rather than violating scope.
- **Doubt 1 (GPT-4o ~12% vs 9.3%/13.3%) — partially accepted.** Cannot settle the exact source figure offline, so per standards I did not substitute another specific number; softened the table cell to "on the order of 10% (single-digit to low-teens depending on the harness)" with a matching paragraph note. The `p=0.12 → 99.97%` illustration is kept as an explicitly hypothetical example (verified `1-0.88^64 ≈ 0.9997`).
- **Doubt 2 (o1 Codeforces 1673 vs 1891) — rejected (no change).** Cannot verify the cited release figure offline; 1673 at 89th percentile is retained and flagged under "Needs external verification" for a pre-publication check.
- **Doubt 3 (Instruction Hierarchy year) — accepted.** Changed 2025 to 2024 (arXiv:2404.13273, April 2024; well-established).
- **Doubt 4 (shortest rejection sampling "off-policy") — accepted (wording tightened).** The box now states the behavior policy is the frozen long-model snapshot and the data is fixed before student training, i.e. off-policy/offline with respect to the student being trained, avoiding the previous non-sequitur justification.
- **Doubt 5 ("much larger" format weight) — accepted (softened without new numbers).** Changed to "usually larger", dropping the overstated quantifier without asserting the unverified 1.0-vs-0.5 ratio.
- **Doubt 6 (2026 Claude Constitution) — rejected (no change).** Cannot confirm the document/URL offline; already listed under "Needs external verification".
- **Doubt 7 (Kimi 60.8 Pass@1 / 3,272 tokens) — rejected (no change).** Plausible but unconfirmable offline (arXiv:2501.12599 check needed); already listed under "Needs external verification".
