# Report: ch17-text (Chapter 17 text pass)

## Summary

- Read all six chapter pages end to end and checked derivations, numeric examples, pseudocode, and paper/number claims.
- Fixed the confirmed Pass@8192 arithmetic error in `formal-prm.md` and standardized the running $\sqrt{2}$ error example (`4k^2 = 2q^2` vs. faulty `4k = q^2`) across `outcome-vs-process.md` and `generative-prm.md`.
- Recomputed every in-text numeric example (discounted weights, cross-entropy losses, aggregation, UCB, majority vote, Pass@k inversion) with `.venv/bin/python`; all other numbers check out.
- Added three tight coverage inserts: calibration/shortcuts/shift/active-collection (discriminative), soundness/completeness/verifier-tests/sandbox (formal), judge biases plus rejection fine-tuning/distillation/rollout roles (parallel).
- Created `docs/chapter17_prm_search/review.md` (summary, equations, pitfalls, 8 exercises with worked solutions, 2 coding).
- `npx prettier --write` on all touched files and `npx vitepress build docs` both succeed.

## Technical fixes

1. **File:** `docs/chapter17_prm_search/formal-prm.md` (Prover-V2 Pass@k bullets)
   - **Wrong claim:** `(1-0.0003)^8192 ≈ 0.111`, implying $p \approx 0.03\%$ explains Pass@8192 = 88.9%.
   - **Correction:** $(1-0.0003)^{8192} \approx 0.0856$, so $1 - 0.0856 \approx 0.914$. The value consistent with 88.9% is $p \approx 0.027\%$ since $(1-0.000268)^{8192} \approx 0.111$ and $1-0.111 = 0.889$. Bullets now read $p=5.3\%$ with $(1-0.053)^{32}\approx0.175$ and $p=0.027\%$ with $(1-0.000268)^{8192}\approx0.111$.
   - **Verification:** `.venv/bin/python` computation: `(1-0.0003)**8192 = 0.08560867018597403`; `1-(1-0.0002683)**8192` complement `0.11100183981925713`; inverted $p = 1-0.111^{1/8192} \approx 0.0002683$. Suspected-item line marked **[x]**.
2. **Files:** `docs/chapter17_prm_search/generative-prm.md`, `docs/chapter17_prm_search/outcome-vs-process.md` (worked $\sqrt{2}$ error example)
   - **Suspected issue:** mixing of "$4k^2 = 2q^2$", "divide by 2", and "$4k = q^2$".
   - **Finding:** the underlying math was consistent (correct chain $p=2k \to 4k^2=2q^2 \to 2k^2=q^2$ vs. faulty $4k=q^2$) but the outcome page's Step 6 line and the generative page's two retellings described different halves of the error. Standardized all three tellings to: correct substitution $4k^2 = 2q^2$ (i.e. $2k^2 = q^2$ after dividing by 2); faulty step writes $4k = q^2$, dropping the square and skipping the division. Suspected-item line marked **[x]** (explained + tightened, no arithmetic change).
   - **Verification:** derivation re-traced from $p^2 = 2q^2$, $p = 2k$ by hand; no code involved.
3. **Checked and left unchanged (all recomputed):** $0.9^{8,4,3,1} \approx 0.43/0.66/0.73/0.90$; $-\log(0.95/0.70/0.10) \approx 0.05/0.36/2.30$ with gradient $-1/p$ discussion; aggregation A mean 0.70/min 0.2/product 0.1296 vs. B 0.60/0.6/0.1296; UCB terms $\approx 0.43/0.55/1.36$ giving 1.23/1.25/1.96; majority vote $N=5$, $p=0.6 \to 0.68256$ and $N=11 \to 0.7535$; Pass@32 = 82.4% $\to p \approx 5.28\%$. Verified with `.venv/bin/python` one-liners.

## Other changes

- `discriminative-prm.md`: new subsection "Calibration, Shortcuts, Shift, and Active Collection" (~330 words): held-out ranking accuracy + calibration/margins/disagreement slices; verbosity/format/length shortcuts with mitigations; distribution shift/overoptimization with version lineage and rollback rule; active collection targeting uncertain comparisons with stop rules.
- `formal-prm.md`: new subsection "Soundness, Completeness, and Verifier Tests" (~300 words): soundness vs. completeness under budgets, parser/elaboration boundaries with three-artifact versioning, false-positive/negative verifier tests, hidden tests, isolated sandbox with reset and deterministic replay.
- `parallel-reasoning-and-summary.md`: judge-bias paragraph (position, identity/self-preference, correlated errors + mitigations) and new subsection "From Search Traces to Training Data" (~280 words): RFT, on-policy vs. off-policy rollout distillation vocabulary, actor/rollout-worker/verifier/learner roles, sync vs. async and staleness, colocated vs. disaggregated pointer to Chapter 18, compute accounting and clean-reproduction note.
- Cross-references: verified Chapter 8 (PPO/GAE), Chapter 15 (GRPO), Chapter 16 (test-time scaling, Section 16.3 coverage formula), Chapter 18/19/25 forward links, and internal 17.1–17.6 numbering — all point at the correct topics; no changes needed. Section 2.1 $\epsilon$-greedy reference left as is (outside scope).
- Light cleanup: clarified Step 6 bullet labels in the discounted-return example; no page restyles.

## Needs external verification

- DeepSeek-Prover-V2 numbers (671B, 82.4% Pass@32 / 88.9% Pass@8192 MiniF2F-test, 47/658 PutnamBench) — reputable but unverifiable offline; arithmetic around them is now correct.
- ThinkPRM deltas (same-base outperformance on ProcessBench/MATH-500/AIME 2024; +8pp GPQA-Diamond and +4.5pp LiveCodeBench subsets; +7.2pp over LLM-as-Judge; ~8K vs. 712K labels) — taken from the paper as stated; subsets and budgets as caveated in text.
- PaCoRe figures (StepFun 2026 report; `[32,4]` 93.7% AIME 2025 / 94.5% HMMT 2025 at ~1.87M/~1.80M tokens vs. RLVR-8B 84.1%/75.4% at ~50K) — no independent source available offline; text already frames them as the paper's reported curves.
- "Let's Verify Step by Step" 78% MATH-subset Best-of-N figure and ToT 24-Game 7.3/4.0/9.0/74% figures — well-known published numbers, kept with the existing in-text caveats against overgeneralization.
- Lean4 snippet idioms and AlphaProof/AlphaGeometry 2024 IMO medal narrative — standard public accounts; code sketch is illustrative, not executed.

## Sidebar additions

- `17.R Summary and Exercises` → `docs/chapter17_prm_search/review.md` (front matter `title: 17.R Summary and Exercises`, H1 `# 17.R Summary and Exercises`). Coordinator to add to sidebar and renumber as needed.

## Commands run and results

- `.venv/bin/python -c` arithmetic checks (Pass@k, logs, UCB, binomial, aggregation): all pass, values listed above.
- `.venv/bin/python -c` importing `code/chapter17_prm_search/tree_of_thought.py` (`ThoughtNode`, `evaluate_node`, `generate_children`, `apply_op` division-by-zero guard): import succeeds; 4-number node scores 0.38, yields 48 children, `1/0` returns None. Grounds the two coding exercises.
- `npx prettier --write` on all seven chapter files: success (two files reformatted, rest unchanged).
- `npx vitepress build docs`: success (build complete, no dead links).

## Open issues

- `code/` was out of scope and untouched; the coding exercises only import from the existing `tree_of_thought.py` and use stdlib otherwise.
- Checklist items marked Later/Depth (sensitive human-subject data, frontier-scale learned process supervision, multimodal training) intentionally receive only pointer-level coverage.
- No commits made; all changes are in the working tree for coordinator review.

## Coverage table (RL-9 / RL-12 items touching this chapter's scope)

| Item                                                                               | Status                                                                     |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Outcome reward models                                                              | Covered (17.1)                                                             |
| Process reward models and step boundaries                                          | Covered (17.1, 17.2)                                                       |
| Generative reward models and judge rationales                                      | Covered (17.3, 17.6)                                                       |
| Executable, rule-based and formal verifiers                                        | Covered (17.4, AlphaCodium tests)                                          |
| Verifier completeness, soundness, parser boundaries                                | Now covered (new 17.4 subsection)                                          |
| False-positive / false-negative verifier tests                                     | Now covered (new 17.4 subsection)                                          |
| Format, verbosity, style, length shortcuts                                         | Now covered (new 17.2 subsection)                                          |
| Reward distribution shift as policy changes                                        | Now covered (new 17.2 subsection)                                          |
| Reward overoptimization and proxy regression                                       | Now covered (new 17.2 subsection)                                          |
| Active feedback collection, uncertain comparisons                                  | Now covered (new 17.2 subsection)                                          |
| AI feedback, constitutions, rubric-conditioned labels                              | Covered (RLAIF / LLM-as-Judge, 17.6)                                       |
| Judge position/identity/self-preference/correlated-error bias                      | Now covered (17.6 addition)                                                |
| Multiple reward components and scalarization                                       | Covered ($\alpha$/$\beta$ mix, 17.2)                                       |
| Preference/reward/verifier version lineage                                         | Now covered (17.2 rollback rule; 17.6 repro note)                          |
| Train vs. independent task/human/safety evaluation                                 | Covered + strengthened (17.2, 17.6)                                        |
| Adversarial completions and hidden verifier tests                                  | Now covered (17.2 adversarial; 17.4 hidden tests)                          |
| Reward governance, rollback, collection stop rules                                 | Now covered briefly (17.2)                                                 |
| Calibration, margins, disagreement slices                                          | Now covered (17.2)                                                         |
| Pairwise triples / listwise comparisons; Bradley-Terry                             | Covered elsewhere (Ch. 13/14); GenRM A-vs-B noted in 17.6                  |
| Demonstrations/comparisons/rankings/ratings/critiques taxonomy                     | Partial (step ratings + GenRM critiques; full taxonomy lives in Ch. 13/14) |
| RLVR; math/code/structured-output environments                                     | Covered (17.1, 17.4, 17.5)                                                 |
| Executable verifier isolation / hidden tests                                       | Now covered (17.4 sandbox)                                                 |
| Outcome sparse credit; process step-level credit                                   | Covered (17.1)                                                             |
| Reasoning traces, hidden reasoning, observable answers                             | Partial (touched via 17.3/17.6; depth in Ch. 16)                           |
| Inference-time search, self-consistency, aggregation                               | Covered (17.5, 17.6)                                                       |
| Train-time vs. test-time compute                                                   | Covered (17.5, 17.6)                                                       |
| Rejection fine-tuning and rollout distillation                                     | Now covered (17.6)                                                         |
| On-policy teacher-student distillation vocabulary                                  | Now covered briefly (17.6)                                                 |
| Reasoning length, budget control, invalid traces                                   | Covered (token budgets, 17.5/17.6)                                         |
| Multi-turn state, tool-call actions, transitions, terminal conditions              | Partial (17.1 agents, 17.5 AlphaCodium loop)                               |
| Trajectory-level rewards / delayed agent credit                                    | Covered (17.1–17.2)                                                        |
| Sandbox, reset, deterministic replay, side effects                                 | Now covered (17.4)                                                         |
| Tool-error/timeout/retry/duplicate-action semantics                                | Partial (retry-dedup accounting noted in 17.6; full semantics in Ch. 19)   |
| Actor/rollout/verifier/learner roles; sync vs. async; staleness                    | Now covered briefly (17.6; depth in Ch. 18)                                |
| Colocated vs. disaggregated; batching/backpressure/starvation; checkpoint recovery | Pointer to Ch. 18 (added in 17.6)                                          |
| Prompt curriculum / filtering / evolving distributions                             | Partial (filtering + active collection; depth in Ch. 18)                   |
| Rollout tokens/s, verifier throughput, update utilization                          | Now covered as accounting checklist (17.6)                                 |
| Independent capability/safety/agent evaluation                                     | Covered via pointers (Ch. 18/19/25)                                        |
| Reward hacking, spec gaming, held-out shifts                                       | Covered via overoptimization + Ch. 25 pointer                              |
| Clean reproduction / prespecified objective                                        | Now covered briefly (17.6)                                                 |
| Self-play / auto-curriculum (Depth); multimodal (Later)                            | Pointer-level only, as scoped                                              |
