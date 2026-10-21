# Report — ch25-text (Chapter 25 text pass)

## Summary

Read all five chapter pages in full, checked derivations, recomputed every numeric example, traced code snippets, and verified paper/benchmark claims against solid knowledge. Made targeted fixes (no rewrites): repaired three fragile code sketches, softened unverifiable system-card numbers, and added four tight coverage insertions (IQM/seeds/stratified bootstrap; judge biases; verifier completeness/soundness with adversarial tests; reward governance/lineage with stop rules). Created `review.md` with outcomes, equations, pitfalls, and 9 exercises with worked solutions. Build passes.

## Technical fixes

1. `rl-evaluation.md` §25.5.1 (`stratified_eval` snippet) — the wrong claim: none about results, but the sketch called undefined `np`, `classify_difficulty`, `verify` and divided by possibly-empty lists. Correction: added `import numpy as np`, stub `verify`/`classify_difficulty` with a comment to replace them, and guarded empty groups (`if v else 0.0`). Verified by reading the snippet; no behavior change to prose.
2. `rl-evaluation.md` §25.5.2 (`ngram_contamination`) — the wrong claim: snippet called undefined `get_ngrams` and divided by zero on short tests. Correction: defined `_ngrams`/`get_ngrams` on whitespace tokens and returned `0.0` for empty test sets. Verified by reading; consistent with the surrounding GPT-3 13-gram discussion.
3. `rl-evaluation.md` §25.5.6 (`long_horizon_eval`) — same class of issue: bare `np.mean`/`np.std` with no import. Correction: added `import numpy as np` and `float(...)` casts. Verified by reading.
4. `modern-incidents.md` §25.2.3 (`solve` lookup table) — the text said the program "fails immediately" on hidden input `4`, but `answers[x]` raises `KeyError` rather than returning a wrong value. Correction: kept the code, added comment `# raises KeyError on any hidden input such as 4`. Verified by Python semantics (no execution needed; `dict[x]` on a missing key raises).
5. `rl-evaluation.md` §25.5.7 (suspected item [x]) — the wrong claim (unverifiable as stated): precise figures `34×`, `20.96`, `65.83%`, `0.612` presented as stable facts, and `34×` described as spanning "the three tasks LLM Training, Text-RL, and Quadruped-RL". Correction: clarified `34×` covers only the LLM-training-optimization task, reworded the four numbers as approximate values "quoted from that card at the time of writing" with an instruction to check the current card, and renamed the task labels to neutral wording. Verified only structurally (no internet): the fix removes the over-specific reading without asserting new numbers. **The underlying figures themselves still need external verification (see below).**
6. Numeric spot-checks that required no change (verified by hand arithmetic): §25.4.1 search budget ($1-0.99^{10}\approx9.6\%$, $1-0.99^{100}\approx63.4\%$); §25.5.1 Wilson-demo interval ($0.60\pm1.96\times0.049\to50.4\%$–$69.6\%$); §25.3.4 safety-rate example ($990/1000=99\%$). All correct.

## Other changes

- `rl-evaluation.md` §25.5.1: new `### Multiple Seeds, IQM, and Stratified Bootstrap` (~300 words) with the IQM formula, a worked 8-seed example (mean 0.5375-style computation done with fresh numbers: mean 0.5125 vs IQM 0.475 — recomputed in review exercises), and stratified-bootstrap guidance. Cites Agarwal et al. 2108.13264 (already the book's standard reference for this).
- `rl-evaluation.md` §25.5.5: new `### Judge Biases in Model-Based Evaluation` (~250 words): position, identity/style, self-preference, correlated-error bias, each with a concrete check/mitigation and a reporting rule.
- `modern-incidents.md` §25.2.3: new `### Completeness, Soundness, and False-Positive Tests` (~250 words): soundness/completeness definitions, adversarial completions, known-good variants, shortcut channels (length, tone, formatting), parser-boundary versioning.
- `scaling-and-defenses.md` §25.4.8: new `### Reward Governance: Versioning, Rollback, and Stop Rules` (~200 words): lineage (data hash, checkpoints, verifier code, scalarization weights), rollback, collection stop rules.
- New file `docs/chapter25_alignment_failures/review.md` (front matter `title: 25.R Summary and Exercises`, H1 `# 25.R Summary and Exercises`): 7 outcomes, 11 key equations with one-line readings, 6 pitfalls, 9 exercises (conceptual, derivations, 4 hand computations, 2 coding) each with a `::: details Solution` block. Every numeric solution was recomputed: Ex2 gaps 0.37/0.34; Ex3 $1-0.98^{50}\approx63.6\%$ ($\ln0.98\approx-0.020203$, $e^{-1.0101}\approx0.364$); Ex4 SE 0.049, interval 50.4–69.6%; Ex5 mean 0.5125, IQM 0.475 ($4.10/8$; $1.90/4$); Ex6 $p_{\text{clean}}=0.004$, $p_{\text{trigger}}=0.70$, $\Delta=0.696$, overall $501/510\approx98.2\%$; Ex7 $R'=1.95$.
- Cross-references: checked all `Chapter N`/`Section 25.k`/appendix links against the numbering table — `Chapter 22/20` computer-use/SWE links, `Appendix A.2 → rl-infrastructure`, and `Section 25.2/25.4` internal links all resolve to the correct topics. No changes needed. Image references to `../chapter04_*`/`../chapter02_*`/`../chapter26_*` images were left untouched (another worker's scope; build passes).

## Needs external verification

- §25.5.7 system-card figures (`34×`, `20.96`, `65.83%`, `0.612`, task names "Novel Compiler", "Internal suite 2", Claude Opus 4.6 card existence/content, and the "unfixed CPUs rerun" claim): no internet access, so the numbers could not be checked against the cited card. I softened the wording to attribute-and-verify; the coordinator should confirm each figure against the current system card or drop them.
- `modern-incidents.md` arXiv 2511.18397 ("Natural Emergent Misalignment from Reward Hacking in Production RL") and 2507.10532 ("Reasoning or Memorization?"): titles/claims read as plausible and internally consistent, but publication status and exact attributions (e.g. Qwen2.5 vs Qwen3 footnote) could not be web-checked.
- Alignment-faking rates (12–14% reasoning, 78% post-RL): kept as stated with the existing hedging; worth confirming against the paper's abstract on a connected machine.

## Sidebar additions

- Title: `25.R Summary and Exercises`; link: `/chapter25_alignment_failures/review`. (Coordinator to add; files themselves were not edited for the sidebar.)

## Commands run and results

- `npx prettier --write` on all five chapter files plus `review.md`: chapter files unchanged, `review.md` formatted.
- `npx vitepress build docs`: success (client + SSR bundles built, all pages rendered, ~18 s).

## Review response (2026-10-21)

1. IQM closed form wrong for M not divisible by 4 — accepted: divisor corrected to `M - 2k` with `k = floor(M/4)`, kept M=8 worked example; added note that `rliable` uses interpolated quantiles.
2. Saturated-task domination mechanism wrong — accepted: rewrote as high-variance runs/lucky seeds moving the mean, IQM trimming both tails.
3. Stratification justification (majority domination) and 80/20 vs 60/20 inconsistency — accepted: rewrote as mixture-fluctuation variance with fixed 60/20 composition matching Exercise 9.
4. Empty-group guard returning 0.0 and single-band stub — accepted: empty bands now return `(None, count)` and the stub cycles through all three bands.
5. Review page not in sidebar config — accepted but no change made: editing `docs/.vitepress/config.mjs` is outside this task's scope; entry `25.R Summary and Exercises → /chapter25_alignment_failures/review` remains listed under Sidebar additions for the coordinator.
6. `25.R` numbering vs `25.6` and heading-tool regex — rejected: the original task explicitly requires front matter/H1 `25.R Summary and Exercises` with coordinator renumbering, so the label was kept.
7. Exercise 9 `easy_a[ie]` fails on Python lists — accepted: added `map(np.asarray, ...)` conversion; verified with list inputs.
8. Section 25.5.7 high-precision system-card figures with "around" hedges — accepted: removed the unverifiable `34x / 20.96 / 65.83% / 0.612` figures, kept per-task attribution (speedup vs locomotion vs pass rate) with an instruction to check the current card.
9. Section 25.4.6 debate-paper title mismatching arXiv 2407.04622 — accepted: relinked text to "Debating with More Persuasive LLMs Leads to More Truthful Answers".
10. Section 25.2.1 RLVR link pointing at DeepSeek-R1 (2501.12948) — accepted: replaced with the internal book link to `/chapter15_grpo/rlvr`; DeepSeek-R1 keeps its existing citation in Section 25.4.4.

- Doubt 1 (non-canonical truncated IQM) — accepted as a clarification: added one sentence noting the `rliable` interpolated-quantile implementation differs slightly at the boundaries.
- Doubt 2 (empty reasoning must score 0 contradicts outcome RLVR) — accepted: qualified so only hard-coded answers and deleted/skipped tests must score 0, with empty-reasoning penalties described as task policy, not outcome-soundness.
- Doubt 3 (unverifiable system-card claims) — accepted: handled via finding 8; figures removed and flagged under Needs external verification.
- Doubt 4 (sixth subsection under "Five Standards", missing Agarwal Further Reading) — accepted: demoted the seeds/IQM block to `####` under Statistical Rigor and added Agarwal et al. 2021 to Further Reading.
- Doubt 5 (whitespace word-grams vs token 13-grams) — accepted: added a code comment stating the sketch is a word-gram approximation.

Review-round commands: `npx prettier --write` on the four touched chapter files plus the report; bootstrap list-input snippet executed successfully via `.venv/bin/python`; `npx vitepress build docs` — success (see below).

## Open issues

- No front matter on the five existing chapter pages (they start directly with `# 25.k`); left as-is since the build passes and front matter is outside this task's scope — flagging in case the coordinator wants a uniform pass.
- Images referenced from other chapters' `images/` dirs were not verified to exist; not in scope and the build does not fail on them.
- Checklist items belonging primarily to other chapters (Bradley-Terry likelihood, BT architecture/loss, PRM step boundaries, DPO/GRPO specifics) are covered in Chapters 13–17 and were deliberately not duplicated here; the coverage table below reflects only this chapter's scope.

### Coverage table (in-scope checklist items)

| Item                                                                                                                 | Status              | Where                                                                                                                                             |
| -------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reward distribution shift as policy changes                                                                          | covered             | 25.1 Goodhart, 25.4.2 overoptimization                                                                                                            |
| Reward overoptimization and proxy regression                                                                         | covered             | 25.4.2 (Gao et al. proxy/gold, KL)                                                                                                                |
| Judge position/identity/self-preference/correlated-error bias                                                        | covered (added)     | 25.5.5 Judge Biases                                                                                                                               |
| Multiple reward components and scalarization policy                                                                  | covered             | 25.4.3–25.4.4 templates + 25.4.8 lineage note                                                                                                     |
| Verifier completeness, soundness, parser boundaries                                                                  | covered (added)     | 25.2.3 new subsection                                                                                                                             |
| False-positive / false-negative verifier tests                                                                       | covered (added)     | 25.2.3 adversarial + known-good tests                                                                                                             |
| Format/verbosity/style/length shortcuts                                                                              | covered (extended)  | 25.2.3 + 25.4.8 stop rules                                                                                                                        |
| Train reward vs independent task/human/safety eval                                                                   | covered             | 25.2.7 record, 25.4.2 four gauges, 25.5                                                                                                           |
| Adversarial completions and hidden verifier tests                                                                    | covered (added)     | 25.2.3–25.2.4, 25.4.3 gate set                                                                                                                    |
| Reward governance, rollback, collection stop rules                                                                   | covered (added)     | 25.4.8 new subsection                                                                                                                             |
| Preference/reward/verifier version lineage                                                                           | covered (added)     | 25.4.8 new subsection                                                                                                                             |
| Multiple seeds, CIs, IQM / stratified bootstrap (working depth)                                                      | covered (added)     | 25.5.1 new subsection                                                                                                                             |
| Held-out shifts (transfer/one-shot gate sets)                                                                        | covered             | 25.2.3, 25.4.3, 25.5                                                                                                                              |
| Demonstrations/comparisons/rankings/ratings/critiques; BT likelihood; PRM step boundaries; AI feedback/constitutions | partial (by design) | primary coverage in Ch. 13/14/17; Ch. 25 touches constitutions (25.4.7), AI prefs, and process-vs-outcome scoring without duplicating derivations |
