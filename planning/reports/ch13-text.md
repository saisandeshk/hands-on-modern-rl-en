# Report: ch13-text (Chapter 13 text pass)

## Summary

Reviewed all 9 pages in `docs/chapter13_rlhf/` end to end. Fixed 13 load-bearing technical errors (wrong win-rate math, wrong hand-computed reward totals, a padding-unsafe reward-model snippet, broken pseudocode, a broken KaTeX command, a broken mermaid style, two wrong chapter cross-references, and two missing front-matter blocks, one of which broke the VitePress build). Aligned all front-matter titles and H1s with the sidebar. Added tight coverage inserts (~150-300 words each) for checklist gaps: bandit/token-MDP/environment-MDP framings, sequence-score aggregates, mask boundaries plus log-probability oracle, BT identifiability/scale, rollout records plus version binding, verifier soundness/completeness, inference-policy vs learned-change distinction, outcome tags, KL estimators plus sampling identity, token vs sequence uncertainty, and compute accounting. Created `docs/chapter13_rlhf/review.md` (13.R) with outcomes, key equations, pitfalls, and 9 exercises with fully worked solutions (all numerics recomputed). Build passes.

## Technical fixes

- `docs/chapter13_rlhf/evaluation.md` — wrong claim: preference table showed 68.2% and 83.7% win rates for (58W/27L/15T) and (72W/14L/14T). Correction: 65.5% and 79.0% per the page's own ties-count-half formula, with the arithmetic shown inline. Verified by direct arithmetic: $(58+7.5)/100=0.655$, $(72+7)/100=0.79$; the old numbers corresponded to dropping ties, contradicting the stated formula.
- `docs/chapter13_rlhf/evaluation.md` — wrong claim: flawed-reward worked example totals 0.32 vs 2.35. Correction: 1.16 vs 3.40 (116 chars → 1.16; 150 chars → 1.50, plus 1.0 format, plus $3\times0.3=0.9$ politeness since all three phrases match). Verified with `.venv/bin/python` char counts.
- `docs/chapter13_rlhf/extended-practice.md` — wrong claim: same flawed-reward table showed 0.32 vs 2.35 with politeness 0.6. Correction: 1.23 vs 3.55 (123 chars → 1.23; 165 chars → 1.65, plus 1.0 format, plus $3\times0.3=0.9$ — the text matches "I'm happy to help", "Here are some", and "Hope this helps"). Verified with `.venv/bin/python`.
- `docs/chapter13_rlhf/imitation-learning-pipeline.md` — wrong claim: reversed-ordering loss $-\log 0.182 \approx 1.704$. Correction: $\approx 1.701$. Verified with `.venv/bin/python` ($-\ln(0.18243)=1.70141$).
- `docs/chapter13_rlhf/standard-rlhf-pipeline.md` — wrong code: `RewardModel.forward` scored `last_hidden_state[:, -1, :]`, which is a padding vector under padded batches. Correction: index the last non-padded token via `attention_mask` and added the missing `import torch` (snippet used `torch.arange` with only `torch.nn` imported). Verified by reading the snippet.
- `docs/chapter13_rlhf/hhh-practice.md` — broken pseudocode: `s(f"prompt": x, "response": y_star)` (invalid Python, never appends). Correction: `sft_pairs.append({"prompt": x, "response": y_star})`. Verified by reading.
- `docs/chapter13_rlhf/hhh-practice.md` — broken math: `\mathbb{0}` in the SL-CAI loss. Correction: `\mathbb{E}`. Verified by reading.
- `docs/chapter13_rlhf/hhh-practice.md` — broken mermaid: `style RL fill:#e8f5e9,style:#2e7d32`. Correction: `stroke:#2e7d32`. Verified by reading.
- `docs/chapter13_rlhf/hhh-practice.md` — wrong cross-reference: "(as proved in Chapter 17, [DPO Theory and Family]…)". Correction: Chapter 14. Verified against sidebar (`/chapter14_dpo/dpo-theory-and-family` is 14.3).
- `docs/chapter13_rlhf/scaling-to-large-models.md` — wrong cross-reference: "before moving to Chapter 9". Correction: Chapter 14 (Chapter 9 is continuous control; the next chapter after 13 is 14 DPO). Verified against the numbering table.
- `docs/chapter13_rlhf/hhh-practice.md`, `docs/chapter13_rlhf/verl-ppo-gsm8k.md` — missing front-matter `title:` blocks. Added; the verl title contains a colon so it is quoted (`'13.8 Hands-on: veRL PPO on GSM8K'`) — the unquoted form broke the VitePress build (confirmed by build failure, then success after quoting).
- Title/H1/sidebar alignment: `imitation-learning-pipeline.md` front matter said "13.2 SFT Instruction Tuning" vs sidebar/H1 "13.2 Supervised Fine-Tuning" (fixed to the latter); `standard-rlhf-pipeline.md` front matter said "13.4 The RLHF Pipeline" vs sidebar/H1 "13.4 Reinforcement Learning Fine-Tuning" (fixed); `evaluation.md` front matter said "13.6 Evaluation" vs sidebar/H1 "13.6 Alignment Evaluation" (fixed); `scaling-to-large-models.md` H1 had an extra suffix beyond the sidebar title (trimmed); base roadmap row for 13.2 misdescribed the section (fixed) and now lists all pages including 13.9.
- `docs/chapter13_rlhf/hhh-practice.md` — softened unverifiable specifics without deleting the substance: Claude 2/3 exact principle counts, model sizes, 70/30 split, 50%/30% effect sizes, 1,000-respondent/20-language claims; replaced the misattributed "(Lee et al., 2023, 'Star')" 10-20% claim with an unsourced qualitative statement; hedged the Claude 4 2026 document description as the chapter's synthesis; added a caveat that the `Honest = 1 - D_KL` formalization is schematic (KL can exceed 1).

## Other changes

- Coverage inserts (all short, placed where they fit best): three-framings table plus sequence-score aggregates in `base-model-to-assistant.md`; feedback-type taxonomy, BT identifiability/scale limits, and mask-boundary/oracle paragraph in `imitation-learning-pipeline.md`; rollout-record immutability plus version binding and verifier soundness/completeness in `standard-rlhf-pipeline.md`; compute-accounting paragraph in `scaling-to-large-models.md`; inference-policy vs learned-change plus outcome tags in `evaluation.md`; KL direction/estimators/limits plus token-entropy vs sequence-diversity in `ppo-rlhf-loop.md`.
- New page `docs/chapter13_rlhf/review.md`: 6 learning outcomes, 6 key equations with one-line readings, 6 pitfalls, 9 exercises (conceptual, derivation, numeric, 2 coding) each with a worked solution in `::: details Solution` blocks.
- Light cleanup: rewrote the `verl-ppo-gsm8k.md` closing summary, which read like an editing note, into a substantive paragraph scoping the GSM8K gain to its setup.

## Needs external verification

- Self-Rewarding LMs three-iteration AlpacaEval/RewardBench numbers in `hhh-practice.md` (kept as-is; could not verify offline).
- RLAIF vs RLHF cost arithmetic ($2/pair, $0.002/1k tokens) — arithmetic is consistent but input prices need a source.
- veRL claims (10k+ stars, HybridFlow at EuroSys 2025), GSM8K split sizes (7,500/1,319), Volcano Engine 42.08% → 54.89% baseline, and all Claude training specifics — plausible but unverifiable without internet; the most load-bearing numbers were hedged, the rest flagged here.
- InstructGPT/Self-Instruct/WizardLM/Evol-Instruct/DAgger/IRL paper attributions — standard and kept, spot-checked from knowledge only.

## Sidebar additions

- Title: `13.R Summary and Exercises`, link: `/chapter13_rlhf/review` (one entry; numbering left to the coordinator). No other new pages.

## Commands run and results

- `npx prettier --write <all 10 touched/created chapter files>` — passed (2 files reformatted, rest unchanged).
- `npx vitepress build docs` — failed once on the unquoted-colon front matter in `verl-ppo-gsm8k.md`; passed cleanly after quoting (16.6s, only pre-existing warnings about `jinja2` highlighting and chunk size).
- Numeric checks via `.venv/bin/python` (BT sigmoid/loss, KL penalty, win rates, char-length reward totals, stress-test scores) — all match the corrected text. (Note: bare `python3 -c` is rejected in this environment, so verification used the allowed `.venv/bin/python`.)
- No test suite is named in this task and `code/` belongs to another worker, so no code tests were run.

## Coverage table

| Checklist item                                                                          | Verdict                                                                  |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| RL-8: prompt distribution, context/task instance as initial state                       | covered                                                                  |
| RL-8: tokens as actions, autoregressive factorization                                   | covered                                                                  |
| RL-8: completion / conversation / tool episode as trajectory choices                    | partial (completion covered; multi-turn tool episodes pointer to Ch. 19) |
| RL-8: contextual-bandit vs token-MDP vs environment-MDP                                 | covered (new)                                                            |
| RL-8: hidden user / evaluator / environment state limits                                | covered (new)                                                            |
| RL-8: prompt / completion / padding / EOS / truncation masks                            | covered (new)                                                            |
| RL-8: per-token and sequence log-probabilities                                          | covered                                                                  |
| RL-8: summed / mean / length-normalized sequence scores                                 | covered (new)                                                            |
| RL-8: sequence / terminal / token / step-level rewards                                  | covered                                                                  |
| RL-8: reward placement and return propagation                                           | covered                                                                  |
| RL-8: policy / reference / critic / RM / verifier roles                                 | covered                                                                  |
| RL-8: SFT as behavior initialization / cloning                                          | covered                                                                  |
| RL-8: base / SFT / reference / trainable policy identity                                | covered                                                                  |
| RL-8: on-policy / off-policy / replayed trajectories                                    | partial (PPO-epoch reuse covered; replay buffers are Ch. 18 territory)   |
| RL-8: temperature / top-p / top-k / generation policy identity                          | covered (new)                                                            |
| RL-8: greedy / sampling / best-of-N / rejection / search as distinct inference policies | covered (new)                                                            |
| RL-8: policy / tokenizer / template / reward / verifier / RNG version binding           | covered (new)                                                            |
| RL-8: immutable rollout records, old log-probs                                          | covered (new)                                                            |
| RL-8: valid / invalid / refused / timed-out / truncated outcomes                        | covered (new)                                                            |
| RL-8: token entropy / sequence diversity                                                | covered (new)                                                            |
| RL-8: forward / reverse KL direction vocabulary                                         | covered (new)                                                            |
| RL-8: reference-policy KL estimators, approximation limits                              | covered (new)                                                            |
| RL-8: fixed-sequence log-probability and mask oracles                                   | covered (new)                                                            |
| RL-8: sampling / training / evaluation prompt separation                                | covered                                                                  |
| RL-8: rollout tokens / verifier calls / latency / compute accounting                    | covered (new)                                                            |
| RL-8: learned policy change vs inference-time selection                                 | covered (new)                                                            |
| RL-8: multi-turn tool environment, delayed credit (Depth)                               | partial (Ch. 19 owns)                                                    |
| RL-8: transformer architecture / SFT mechanics (Later)                                  | missing (owned by LLM tracks)                                            |
| RL-9: demonstrations / comparisons / rankings / ratings / critiques                     | covered (new)                                                            |
| RL-9: prompt sampling, policy-dependent collection                                      | covered (new)                                                            |
| RL-9: annotator population / rubric / target-preference definition                      | partial (rubrics covered; population sampling light)                     |
| RL-9: pairwise triples, listwise comparisons                                            | covered                                                                  |
| RL-9: Bradley-Terry likelihood                                                          | covered                                                                  |
| RL-9: reward differences, identifiability, scale limits                                 | covered (new)                                                            |
| RL-9: pairwise RM architecture and training loss                                        | covered                                                                  |
| RL-9: held-out ranking accuracy                                                         | covered                                                                  |
| RL-9: calibration, margins, disagreement slices                                         | covered                                                                  |
| RL-9: outcome reward models                                                             | covered                                                                  |
| RL-9: PRMs and step boundaries                                                          | partial (Ch. 17 owns detail)                                             |
| RL-9: generative RMs, judge rationales                                                  | covered                                                                  |
| RL-9: executable / rule-based / formal verifiers                                        | covered                                                                  |
| RL-9: verifier completeness / soundness / parser boundaries                             | covered (new)                                                            |
| RL-9: false-positive / false-negative verifier tests                                    | covered (new)                                                            |
| RL-9: format / verbosity / style / length shortcuts                                     | covered                                                                  |
| RL-9: reward distribution shift as policy changes                                       | covered                                                                  |
| RL-9: overoptimization, proxy regression                                                | covered                                                                  |
| RL-9: active feedback, uncertain comparisons                                            | covered                                                                  |
| RL-9: AI feedback, constitutions, rubric-conditioned labels                             | covered                                                                  |
| RL-9: judge position / identity / self-preference / correlated error                    | covered                                                                  |
| RL-9: multiple reward components, scalarization                                         | covered                                                                  |
| RL-9: preference / reward / version lineage                                             | covered (new)                                                            |
| RL-9: train vs independent task / human / safety eval                                   | covered                                                                  |
| RL-9: adversarial completions, hidden verifier tests                                    | covered                                                                  |
| RL-9: governance, rollback, collection stop rules                                       | covered                                                                  |
| RL-9: sensitive human-subject data (Later)                                              | missing (out of scope)                                                   |
| RL-9: learned process supervision at frontier scale (Depth)                             | missing (Ch. 17 owns)                                                    |

## Open issues

- Could not append to `planning/content-review-notes.md` (outside task scope) — coordinator should port the Technical fixes list there per repo guidelines.
- `code/chapter13_rlhf/*.py` headers say "Chapter 8" (stale after renumbering) — flagged for the code worker, not edited.
- The sidebar orders 13.8 (veRL hands-on) before 13.9 (PPO loop supplement) although the PPO loop is conceptually prerequisite; left as-is for the coordinator.
- Evaluation and extended-practice both contain a `flawed_reward` experiment by design (evaluation points to extended practice for the full version); numbers are now mutually consistent (different example strings, both correct).
- Checklist items marked Depth/Later were intentionally left to their owning chapters (multi-turn tool credit → Ch. 19, step boundaries/process supervision → Ch. 17, transformer/SFT mechanics → LLM tracks).

## Review response

- Finding 1 (sidebar registration + "13.R" label): partially accepted. The sidebar entry and any retitling are out of scope (`docs/.vitepress/config.mjs` is not editable on this branch, and the task mandates the `13.R` title/H1 with the coordinator to renumber). Accepted the in-scope part: the 13.1 Roadmap table in `base-model-to-assistant.md` now uses the exact sidebar titles (13.1/13.4/13.6) and adds a `[13.R Summary and Exercises](./review)` row; the "Sidebar additions" entry for `/chapter13_rlhf/review` is retained for the coordinator.
- Finding 2 (BT "scale is arbitrary"): accepted. Reworded in `imitation-learning-pipeline.md`, `review.md`, and the "Reward Scale" paragraph in `standard-rlhf-pipeline.md`: only the offset is unidentified; the scale is pinned by the sigmoid's unit temperature but must still be calibrated before PPO.
- Finding 3 (`R_bad` formula): accepted. Now `$R_{bad}=0.01|y|+0.5\cdot\mathbb{1}[\text{list/bullet}]+0.5\cdot\mathbb{1}[\text{bold}]+0.3\sum_k\mathbb{1}[\text{phrase}_k]$`, matching the code and the worked table.
- Finding 4 (`R_safe` weights): accepted. Formula now `0.4/0.35/0.15/-0.05/-0.05`, identical to `safer_reward`.
- Finding 5 ("cannot be hacked"): accepted. Now reads deterministic against semantic manipulation but still exploitable via parser tricks; table row changed to "Deterministic, but parser tricks still apply".
- Finding 6 ("Boundary with Chapter 9"): accepted. Retitled "Boundary with Chapters 14-16".
- Finding 7 ("classification network in Chapter 5"): accepted. Now points at the softmax policy network from Chapter 6.
- Finding 8 (trajectory notation in Chapter 3): accepted. Now cites Chapter 2 (Section 2.2, MDPs); verified `s_0,a_0,r_0,…` appears in `docs/chapter02_rl_basics/mdp.md` and `s_0` has no match under `docs/chapter03_value_functions/`.
- Finding 9 (reward-design in Chapter 3): accepted. Now cites Chapter 4 (Section 4.3, Reward Design); verified the "defines 'what is worth pursuing'" sentence lives there.
- Finding 10 (BT rounding 0.182/1.701): accepted. Both pages now print $\sigma(-1.5)\approx0.1824$, loss $\approx1.702$ (verified: $\sigma(-1.5)=0.18243$, $-\log=1.70141$; $-\log(0.1824)=1.70156\to1.702$).
- Doubt 1 (self-rewarding numbers): kept numbers, added hedge pointing at Yuan et al. 2024 Tables 1-2 as indicative; stays in Needs external verification.
- Doubt 2 (Claude 4 Constitution 2026): accepted in part. Dropped the year from the Further Reading entry (now "Anthropic 'Claude's Constitution' document", URL kept as-is); body already frames the description as the chapter's synthesis.
- Doubt 3 (κ ≈ 0.4-0.6 / 0.7-0.9): accepted. Both cells now marked "illustrative range".
- Doubt 4 (Section 13.2 split reference): accepted. Now points at the "Data Splitting: By Prompt, Not By Pair" subsection in the same page.
- Doubt 5 (Exercise 9 `rlhf_ppo_train.py` pointer): accepted. Reworded to note the script has its own rule-based scorer rather than implying it consumes `flawed_reward`.
- Doubt 6 (Roadmap naming): accepted. Rows 13.1/13.4/13.6 fixed plus review row added (see Finding 1).
- Doubt 7 (Instruct as "pretraining baseline", 7,500): accepted. Now "pre-RL baseline (instruction-tuned)", "7,473 training examples", and "pre-RL starting checkpoint".
- Doubt 8 (right-padding assumption): accepted. Comment "(assumes right padding)" added to the RM snippet.
- Doubt 9 (truncation "must mask"): accepted. Reframed as one common policy with alternatives (drop example / train on visible prefix), applied consistently.
