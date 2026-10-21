# Report: ch15-text (Chapter 15 text pass)

## Summary

Read all eight pages in `docs/chapter15_grpo/` end to end, recomputed every numeric example, and checked derivations, code sketches, paper claims, and cross-references against the chapter-numbering table. Fixed one confirmed fabrication (the "Modified Hamiltonian Monte Carlo sampler"), removed or softened six unverifiable venue/score/attribution claims, repaired three wrong cross-references, and added four tight textbook inserts for checklist items that were missing or partial (KL estimators, within- vs batch normalization, rollout reuse/staleness/collapse signals, failed-rollout accounting and recovery, matched-compute eval controls). Created `docs/chapter15_grpo/review.md` (15.R Summary and Exercises) with 6 outcomes, 6 equations, 6 pitfalls, and 8 fully worked exercises. `npx vitepress build docs` succeeds.

## Technical fixes

1. **`grpo-family.md` — removed invented "Modified Hamiltonian Monte Carlo (mHC) sampler" (suspected item [x]).**
   Wrong claim: DeepSeek V3.2 optimized "the numerical stability of the Modified Hamiltonian Monte Carlo (mHC) sampler during long CoT training." No such sampler exists in any verifiable GRPO/V3.2 source; the cited `arXiv:2512.02556` is not verifiable offline.
   Fix: deleted the bullet and rewrote the V3.2 subsection as a cautious paragraph (mean-only advantages, weakened KL for math) with an explicit note that attached benchmark numbers should not be quoted as settled.
   Verified by: attempted source check — no mHC reference in the chapter's code, no verifiable citation; the surrounding dated claims (Dec-2025 arXiv, AIME-2025-vs-GPT-5) were likewise unverifiable, so all were cut together.
2. **`grpo-family.md` — fixed self-contradictory VAPO comparison.**
   Wrong claim: "VAPO achieves 60.4 on AIME 2024, surpassing all contemporary GRPO variants (DAPO at 50, R1-Zero at 71 …)" — 60.4 does not surpass 71.
   Fix: rewrote as efficiency/setup comparison (VAPO above DAPO in its setup; R1-Zero higher with far more steps).
   Verified by: arithmetic ordering.
3. **`grpo-family.md` — softened unverifiable CISPO engineering specifics.**
   Wrong/risky claim: "trains on 512 H800 cards, achieving training speed twice as fast as DAPO" plus a causal story about lightning-attention error accumulation.
   Fix: kept the mechanism (scale IS weights instead of dropping tokens), replaced hardware/speedup numbers with "reported to be more robust in that setting … not independently verifiable."
4. **`grpo-family.md` — marked RPT loss as schematic.**
   The equation `L_RPT = −E[log π] + β·RL loss` is a pedagogical invention, not a published loss. Rewrote with `≈` and "schematically" wording.
5. **`deepseek-dapo.md` — removed unverifiable venue claim.**
   "Accepted as a NeurIPS 2025 poster" → "Proposed in 2025." Venue not verifiable offline.
6. **`rlvr.md` — softened two unverifiable attributions.**
   "ICLR 2025 research showed 1-shot RLVR" → "a widely discussed 2025 study reported…"; "raised by a NeurIPS 2025 oral paper" → "raised by a 2025 critique of RLVR."
7. **`rl-environments.md` — fixed three wrong cross-references.**
   "asynchronous RL, see 22.6" (no Section 22.6 exists; Ch.22 has only 22.1–22.2) → "see Section 18.4" (distributed RL training). "The entire Chapter 23 revolves around these two numbers" → "The engineering chapters return to these two numbers" (Ch.23 is VLM RL, unrelated). Fixed nonstandard gradient notation `log π(a_t|s_<t,a_<t)` (missing parenthesis, nonstandard conditioning) to the standard history-conditioned form with a definition of `s_t`.
8. **`rl-environments.md` — softened two unverifiable claims.**
   "Karpathy proposed … 5–6 agents" → unattributed practitioner pattern (also fixed: "select the first one to complete" misdescribed best-of-N; now selects max verifier score). "Standard technique used by Sonnet 3.5 / Claude 4 / GPT-5 … 2–3×" → qualitative statement conditioned on verifier quality. AReaL table numbers (45%→92%, 320 s→115 s, 2.77×) reframed as "the paper's reported figures, not independently reproduced" in both body and summary.
9. **`verl-code-sandbox.md` — fixed dangling pointer.**
   "prioritize reading 10.5" (no Section 10.5 exists) → "start with Section 19.8" (the DeepCoder lab this page already contrasts itself with).
10. **Numeric examples recomputed and confirmed correct (no change needed).**
    Apple-example mean 0.625 and advantages ±; token-ratio product 1.1×0.9×1.5 = 1.485 with clip to [1.1, 0.9, 1.2]; `grpo_group_normalize` example mean 0.9375 / std 0.6343 / advantages [0.887, −1.478, …] (recomputed with `numpy`/`torch`); OPD logits→softmax→logps table (0.67/0.24/0.09, −0.41/−1.41/−2.41); `selected[prompt_len−1:]` slicing in `rlvr.md` correctly captures the first response token. Review-page answers cross-checked by execution (Ex. 2: ±1.4142 sum 0; Ex. 3: mean 0.5333; Ex. 4: k3 ≈ 0.0052; Ex. 5: 1.485^{1/3} ≈ 1.141).

## Other changes

- Added front matter `title: 15.4 Improved Methods for GRPO` to `grpo-family.md` (was the only page missing it besides pages that already build without it; behavior unchanged).
- `grpo-practice-and-mechanism.md`: new "Which KL estimator? k1 versus k3" box with worked numbers (Δ=0.1 → k1 −0.1, k3 ≈ 0.0052); new paragraph on within-prompt vs batch normalization trade-off (why GRPO discards cross-prompt difficulty; curricula decide prompt selection).
- `grpo-family.md`: new "Operating envelopes" subsection (zero-signal/invalid/verifier-failure groups; one-vs-several updates with clipped-fraction freshness rule; entropy-collapse/KL-spike/verbosity diagnostics).
- `rl-environments.md`: new "Failed rollouts, retries, and recovery" subsection (success/invalid/infra_error accounting, retry deduplication, checkpoint four-tuple, backpressure/learner-starvation monitoring).
- `financial-tool-calling-grpo.md`: new matched-compute-controls paragraph (same generated calls/verifier executions/updates across baselines; KL/entropy/length/validity traces).
- Created `docs/chapter15_grpo/review.md` (see Sidebar additions).

## Needs external verification

- GSPO sequence-ratio formula and Qwen3-adoption claims (`grpo-family.md`): formula matches my knowledge of the GSPO paper (`arXiv:2507.18071`) but could not be verified offline.
- CISPO reversed-ratio equation and MiniMax M1 paper (`arXiv:2506.13585`): mechanism description kept as-is; confirm the equation against the paper when online.
- VAPO (`arXiv:2504.05118`), Dr.GRPO (`arXiv:2503.20783`), RPT (`arXiv:2506.08007`): paper identities match my knowledge; the specific AIME numbers (VAPO 60.4, DAPO 50) and "Qwen internal training" remarks need source checks.
- DAPO "half the training steps" / AIME-50 claim; SimpleRL-Zoo (`arXiv:2503.18892`); Eurus-2-RL-Data structure and VolcEngine PPO numbers in `verl-code-sandbox.md` (0.82→0.83, 0.59→0.59, 0.50→0.53); AWS tool-calling gains quoted in `financial-tool-calling-grpo.md` (~0.62→0.96 etc.).
- All 2026-dated OPD citations (`on-policy-distillation.md`: arXiv 2604.x/2608.x, MOPD/Open-MOPD scores 0.9373/0.8818, KL 0.04 vs 0.19, token-share 49.7%/0.99%): internally consistent but unverifiable offline; spot-check the most load-bearing numbers (MOPD vs Mix-RL gap, capability-imbalance diagnosis) when online. No changes made there beyond this flag.
- "Pash 2025, Evals = RL Environments" blog (already marked offline in the page): confirm attribution when online.

## Sidebar additions

- `15.R Summary and Exercises` → `/chapter15_grpo/review` (new file `docs/chapter15_grpo/review.md`, front matter `title: 15.R Summary and Exercises`, H1 `# 15.R Summary and Exercises`). Coordinator to renumber/insert after 15.8.

## Commands run and results

- `.venv/bin/python -c` (numpy/torch recomputation of group-normalization example, ratio product, review exercise answers): all matched the text; see item 10 above.
- `npx prettier --write <all 8 touched/created files>`: clean (2 files reformatted, rest unchanged).
- `npx vitepress build docs`: success (client + SSR bundles, all pages rendered, no dead-link errors).
- `git status --short`: attempted; clean tree before edits (a piped variant was auto-rejected per the no-pipe rule; unpiped rerun worked).
- No code tests named in this task and `code/` is out of scope (not edited); none run.

## Open issues

- `grpo-family.md` has no per-claim citations for GSPO/CISPO/VAPO/RPT beyond arXiv links; if the coordinator wants a uniform "reported, not reproduced" caveat policy for benchmark numbers across workers, this chapter now follows it for the worst cases only.
- Checklist items marked Depth/Later (GSPO details kept brief, frontier-scale convergence, multimodal/embodied, multi-agent self-play) were intentionally left to Chapters 16/18/26; RL.11–RL.12 core items are otherwise covered (table below used working definitions: covered = taught with equation/example; partial = mentioned without mechanism).

Coverage (RL.11): on-policy sampling, sequence PG, sparse terminal reward, token-space REINFORCE, critic/PPO-RLHF contrast, grouped sampling, group advantages, GRPO objective/trade-offs, all-equal-group behavior, old/current logprobs, per-token ratios, clipping + aggregation, masks/truncation, length/verbosity control, group-size cost, curricula, DAPO/GSPO variants — covered. RLOO, within-vs-batch normalization, KL estimators, multi-step drift/freshness, collapse signals, invalid/verifier-failure groups, matched-compute controls, full trace logging — were partial/missing, now covered by the inserts and review page. Frontier-scale convergence — Later, skipped.
Coverage (RL.12): RLVR, math/code/structured-output envs, sandbox isolation, outcome rewards, rejection fine-tuning/distillation, teacher-student vocabulary, length/budget control, multi-turn state/actions, transitions/terminal conditions, trajectory credit, roles, sync/async, broadcast/staleness, curriculum/filtering, benchmark suites, hacking defenses — covered. Process rewards — intentionally Ch.17 territory (referenced). Retry dedup, failed-rollout accounting, recovery, batching/backpressure, throughput accounting — were missing, now covered by the new subsection. Self-play/multimodal — Depth/Later, skipped.

## Review response

One line per reviewer finding (E = error, D = doubt). Fixes verified by re-reading the edited passages; numeric changes rechecked with `.venv/bin/python` (sample-std advantages, ln 1.485 intermediates).

- E1 (sidebar missing review page / 15.R label): rejected — `docs/.vitepress/config.mjs` is outside my scope and the task explicitly orders front matter `title: 15.R Summary and Exercises` with H1 `# 15.R Summary and Exercises`; the page is listed under "Sidebar additions" for the coordinator, as the task requires.
- E2 (Clip-Higher described backwards): accepted — rewrote the `deepseek-dapo.md` paragraph around the upper bound (0.01 → 0.012 under [0.8, 1.2], → 0.0128 under 1.28) and fixed the table cell.
- E3 (token-level loss mischaracterized as per-token credit): accepted — rewrote the paragraph/table cell as a reduction change (batch-token average vs per-response average removing the long-answer penalty); advantage stays sequence-level.
- E4 (population vs sample std contradiction): accepted — declared sample std (Bessel, `ddof=1`) once in 15.1, fixed the NumPy snippet to `rewards.std(ddof=1)` with corrected outputs (std 0.6781, advantages [0.8295, −1.3825, …]), and updated Exercises 2 (answer ±1.2247) and 6 to the same convention.
- E5 (AReaL 2.77× contradicts Ch.18): accepted — dropped the "Llama-3-8B on agentic tasks" attribution in 15.5; the paragraph now reports the paper's headline figure and defers to Section 18.4 for the setup. Ch.18 itself is outside my scope.
- E6 (policy-gradient objective linked to 15.4): accepted — link now points to `./grpo-practice-and-mechanism` (Section 15.1).
- E7 (k3 labelled "squared"): accepted — now "**k3 (exponential)**"; derivation untouched (verified correct).
- E8 (ln 1.485 intermediate 0.3958): accepted — now 0.3954 / 0.1318 / exp ≈ 1.1409; final 1.141 unchanged.
- E9 (`grpo_group_normalize` not in `code/chapter15_grpo/grpo_mechanism.py`): accepted within scope — Exercise 6 now cites the Section 15.1 snippet instead of the code file; `code/` is outside my scope so the file itself (and its stale "Chapter 9" docstring) is left for its owner, noted below.
- E10 (`isinstance(x, type(gold_value))` too strict): accepted — `schema_ok` now treats int/float as one numeric class (120 vs 120.0 both pass) and excludes bool explicitly.
- E11 (four vs five directions): accepted — now "five directions" including pre-training integration.
- E12 (fixes not recorded in `planning/content-review-notes.md`): rejected (out of scope) — that file is not in my editable scope; the four fixed items (mHC sampler, VAPO-vs-R1-Zero ordering, DAPO NeurIPS venue, 1-shot-RLVR/RLVR-critique venues) are recorded in this report's Technical fixes; coordinator to copy them over.
- E13 (front-matter title vs sidebar label): rejected — the slight heading/sidebar wording difference follows this chapter's existing pattern (15.1, 15.3 do the same) and the sidebar is coordinator-owned; renaming risks churn for no factual gain.
- D1 (mHC = Manifold-Constrained Hyper-Connections): rejected pending verification — no network access to check the V3.2 report; reintroducing a differently-expanded claim would trade one unverifiable statement for another. Flagged under Needs external verification.
- D2 (VAPO expansion "Value-based Augmented PPO"): accepted (softened) — expansion dropped, paper cited by title/arXiv id only.
- D3 ("mainstream in 2024"): accepted — now "around 2025".
- D4 (GSPO 10,000-card / "all Qwen3" claims): accepted (softened) — replaced with a mechanism-only statement marked not independently verified.
- D5 (AReaL model/task attribution probably wrong): accepted — attribution removed per E5; exact paper table still needs an online check.
- D6 (Exercise 8 KL-flat explanation): accepted — solution now notes per-token KL can stay flat under gradual drift and gates the diagnosis on held-out accuracy, not KL alone.
- D7 (KL orientation vs TRL): accepted — added the Δ-orientation sentence and the opposite-sign warning.
- D8 ("roughly a third" clipped-fraction cut-off): accepted — now "a rising clipped fraction" with no invented threshold.
- D9 (retry-dedup wording vs GRPO groups): accepted — now "one judged trajectory per (prompt, attempt id); re-sample missing group members".
- D10 ("trading format stability for lucky guesses"): accepted — sentence rewritten so falling validity reads as its own red flag.

Follow-up for coordinator/owners (not actionable in my scope): add `15.R Summary and Exercises → /chapter15_grpo/review` (or renumbered `15.9`) to the sidebar; reconcile the AReaL 2.77× setup wording with Chapter 18.4; copy the four factual-fix records into `planning/content-review-notes.md`; consider adding `grpo_group_normalize` with a zero-std guard to `code/chapter15_grpo/grpo_mechanism.py` and fixing its stale "Chapter 9" docstring.

## Commands run (review round)

- `.venv/bin/python -c` (sample-std advantages for both reward vectors; ln(1.485)/3/exp check): `[0.8295, −1.3825, …, 0.0922, …, −0.6452]`, mean 0.9375 sample-std 0.6781; Ex.2 sample-std 0.8165 → ±1.2247; ln 1.485 ≈ 0.395415 → /3 ≈ 0.131805 → exp ≈ 1.140886.
- `npx prettier --write` on all touched files (see below); `npx vitepress build docs` — success (build complete, all pages rendered).
