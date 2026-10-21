# Report: ch14-text (Chapter 14 text pass)

## Summary

Reviewed and edited all four Chapter 14 pages (`dpo-objective-derivation.md`, `metrics.md`, `dpo-theory-and-family.md`, `dpo-hands-on.md`) for technical correctness, cross-references, and curriculum coverage, and created `review.md` (summary + 8 exercises with worked solutions). No suspected issues were pre-recorded for this chapter, so there is nothing to mark `[x]`. All derivations were re-checked by hand; the loss/margin/accuracy numeric examples in the logs were verified mutually consistent. `npx vitepress build docs` succeeds.

## Technical fixes

1. **`dpo-objective-derivation.md` — implicit-reward statement was imprecise.**
   Wrong claim: `$r(x,y)\propto\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}$` ("proportional to").
   Correction: `$r(x,y) = \beta \ln\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \mathrm{const}(x)$`, noting the constant is prompt-dependent and cancels in comparisons.
   Verified by: re-deriving from the Exercise 4 Lagrangian closed form ($\pi^* \propto \pi_{\text{ref}}\exp(r/\beta)$ implies $r = \beta\log(\pi^*/\pi_{\text{ref}}) + \beta\log Z(x)$).
2. **`metrics.md` — duplicated "Reward Accuracy" section.**
   Wrong claim: none factual, but two consecutive `## Reward Accuracy` sections repeated the same definition (lines ~96-119).
   Correction: merged into one section (coarse binary signal vs fine continuous margin, healthy 0.8-0.95 range).
   Verified by: reading the full file; no content lost (both wordings preserved in merged form).
3. **`metrics.md` — misplaced "Chapter Summary".**
   Wrong claim (structural): a 14.2 page summarized the whole chapter including hands-on results taught later in 14.4.
   Correction: renamed to "Section Summary", trimmed to metrics-only takeaways; 14.3 pointer kept (correct next-section order: 14.1 -> 14.2 -> 14.3 -> 14.4 per sidebar).
   Verified by: sidebar order in `docs/.vitepress/config.mjs` (lines 1026-1046).
4. **`dpo-theory-and-family.md` — front matter / H1 / sidebar mismatch.**
   Wrong: front matter `title: 14.3 DPO Theory, Math, and Method Selection` vs H1 `# 14.3 DPO Improvements` vs sidebar `14.3 DPO Improvements`.
   Correction: front matter set to `14.3 DPO Improvements`.
   Verified by: direct comparison of the three strings.
5. **`dpo-theory-and-family.md` — wrong GRPO pointer.**
   Wrong claim: mermaid node `GRPO (2025) ... covered in Section 14.1`.
   Correction: `GRPO (2024) ... covered in Chapter 15` (GRPO is Chapter 15 per the book table; GRPO originates from the 2024 DeepSeekMath paper, arXiv:2402.03300).
   Verified by: chapter-numbering table (GRPO = Ch. 15) and sidebar Ch. 15 entries.
6. **`dpo-theory-and-family.md` — implicit-reward sketch used unmasked mean log-probs.**
   Wrong claim (code sketch): `reward = beta * (policy_log_probs.mean() - ref_log_probs.mean())` over the full concatenated prompt+response.
   Correction: rewrote as completion-only masked sum (`sequence_logprob_sum` with prompt mask), matching `sequence_logprob` in `snippets/dpo-code-map.py` (labels `-100`, `.sum(dim=-1)`) and the completion-mask requirement.
   Verified by: reading `snippets/dpo-code-map.py` lines 13-28.
7. **`dpo-theory-and-family.md` — IPO empirical thresholds stated without source.**
   Wrong claim: "fewer than 1,000 examples ... exceeds 10,000 examples" stability crossover.
   Correction: softened to qualitative guidance (capped target matters most in scarce-data regimes; gap narrows with abundant clean data) and added the missing units clarification (IPO target $1/(2\beta)$ is on the raw, un-$\beta$-scaled log-ratio gap, unlike the DPO margin).
   Verified by: IPO loss formula itself (no $\beta$ multiplies the log-ratio difference inside the square).
8. **`dpo-theory-and-family.md` — SimPO loss missing target margin.**
   Wrong claim (omission): SimPO presented as pure length-normalized gap.
   Correction: full form $\mathcal{L} = -\log\sigma(\beta(\bar\ell_w - \bar\ell_l) - \gamma)$ with $\gamma > 0$ target margin.
   Verified by: standard SimPO formulation (Meng et al. 2024); the $\gamma$ term is load-bearing for the "must exceed, not merely exceed zero" semantics.
9. **`dpo-hands-on.md` — missing front matter broke the build pattern.**
   Wrong: page had no `--- title ---` block (every sibling page has one).
   Correction: added `title: '14.4 Hands-on: A DPO Alignment Experiment'` (quoted: unquoted colon failed YAML parsing and broke `vitepress build`; confirmed by the build error, then fixed).
   Verified by: successful build after quoting.
10. **`dpo-hands-on.md` — leftover translator note.**
    Wrong (editorial): "We translate the surrounding explanation, and only translate clearly-explanatory print strings..."
    Correction: removed; kept the one substantive sentence.
11. **Training-log numbers re-verified (no change needed).** Margin column equals chosen-minus-rejected in all four rows (e.g. $0.9201-(-0.6431)=1.5632$); losses are consistent with $-\ln\sigma(\text{margin})$ up to batch-averaging Jensen gap (e.g. margin 1.5632 gives $0.190$ vs listed $0.2103$ — plausible for mean-of-losses, and the values are labeled illustrative). Initial $\ln 2 \approx 0.693$ claims correct.

## Other changes

- `dpo-theory-and-family.md`: added "DPO Gradient: Easy, Hard, and Mislabeled Pairs Get Different Weights" ($\nabla\ell = -\sigma(-u)\beta\nabla(\Delta_w-\Delta_l)$ with worked weights $u=2 \to 0.119$, $u=0 \to 0.5$, $u=-2 \to 0.88$); "Masks, Lengths, and Label Conventions" (completion-only masks/prompt cancellation, summed vs length-normalized with SimPO $\gamma$, order/sign/label invariants + fixed-batch tests); new "Label Noise, Unpaired Feedback, and No-Reference Losses" subsection (cDPO loss, KTO value-function sketch with loss aversion $\lambda>1$, ORPO loss with odds ratio + SFT-on-chosen baseline); one paragraph on support mismatch and iterative/online DPO.
- `metrics.md`: added "Beyond the Five Columns: KL, Entropy, Length, and Validity" diagnostics plus fixed-batch recompute/swap tests and the identical-tokenizer/template/truncation rule.
- `dpo-hands-on.md`: added "14.4.5 Controls, Held-Out Slices, and What Counts as Evidence" (held-out prompts/slices, compute-matched SFT and rejection-sampling baselines, external task/safety/style checks, overoptimization-as-early-stopping-signal).
- `dpo-objective-derivation.md`: only the item-1 fix; derivation itself verified correct (Lagrangian steps, KL expansion, $Z(x)$ cancellation all check out).
- `review.md`: new page per task spec (outcomes, key equations, pitfalls, 8 ordered exercises each with `::: details Solution`, two coding exercises building on `code/chapter14_dpo/` data format and the `dpo-code-map.py` snippet; all numerics hand-verified: Ex. 2 batch loss $0.650$/margin $0.09$/acc $2/3$; Ex. 3 weights $0.119/0.5/0.818$; Ex. 6 sign flip $+0.05$ vs $-0.015$; Ex. 8 $\beta$-sweep losses $0.6872/0.6349/0.2633$, cDPO $0.5067$ at $u=5$).

## Curriculum checklist (RL-10)

| Item                                                         | Status                                        |
| ------------------------------------------------------------ | --------------------------------------------- |
| KL-regularized reward-maximization objective                 | covered                                       |
| Optimal policy form under a reference policy                 | covered                                       |
| Implicit reward reparameterization from policy ratios        | covered                                       |
| Bradley-Terry substitution and DPO derivation                | covered                                       |
| Chosen and rejected sequence log probabilities               | covered                                       |
| Policy and frozen reference log-ratio margins                | covered                                       |
| DPO logistic loss and $\beta$                                | covered                                       |
| Gradient weighting for easy, hard and mislabeled pairs       | covered (added)                               |
| Completion-only masks and prompt cancellation                | covered (fixed + added)                       |
| Summed versus length-normalized log probabilities            | covered (added)                               |
| Chosen-rejected order, sign and label invariants             | covered (added)                               |
| Reference-policy identity and drift                          | covered                                       |
| Preference-dataset policy and support mismatch               | covered (added)                               |
| Offline objective versus online RL distinction               | covered                                       |
| SFT, preferred-fine-tuning and no-reference baselines        | covered (added: SFT-on-chosen, ORPO)          |
| IPO and preference-model misspecification                    | covered (added misspecification + units)      |
| Label smoothing and conservative DPO vocabulary              | covered (added cDPO)                          |
| KTO, ORPO and unpaired-feedback vocabulary (Optional)        | covered (added; optional but trivial)         |
| Online and iterative DPO variants (Depth)                    | partial (one paragraph; acceptable for Depth) |
| Margin, reward, KL, entropy, length and validity diagnostics | covered (added KL/entropy/length/validity)    |
| Fixed-batch loss, gradient and invariance tests              | covered (added)                               |
| Tokenizer, chat-template and truncation consistency          | covered (added)                               |
| Preference accuracy versus downstream behavior               | covered                                       |
| Overoptimization as $\beta$, epochs or data reuse increase   | covered (added early-stopping divergence)     |
| Held-out prompt and annotator slices                         | covered (added)                               |
| Compute-matched rejection and SFT controls                   | covered (added)                               |
| External task, safety and style evaluation                   | covered (added)                               |
| Exhaustive catalog of direct-alignment variants (Later)      | missing (out of scope by design)              |

## Needs external verification

- SimPO $\gamma$ default values and KTO loss-aversion coefficient $\lambda$ magnitudes: forms are standard but no numeric defaults are stated in the chapter, so nothing to verify numerically.
- `dpo-theory-and-family.md` line on TRL internals ("checked 2026-05-01: `DataCollatorForPreference`, `ref_chosen_logps`, `PPOTrainer` init signature") describes a moving upstream API; I did not change it, but the coordinator may want a freshness check since TRL evolves.
- Hands-on log excerpts and model outputs are labeled illustrative; not verified by execution (no GPU/model download in this environment).

## Sidebar additions

- Title: `14.R Summary and Exercises`, link: `/chapter14_dpo/review` (to be placed after `14.4 Hands-on: A DPO Alignment Experiment`).

## Commands run and results

- `npx prettier --write <all 5 chapter files>`: pass (only `review.md` needed reformatting; others unchanged).
- `npx vitepress build docs`: initially failed on unquoted `:` in the new `dpo-hands-on.md` front matter (YAML parse error); after quoting, **build succeeds** (client + SSR + rendering complete).
- Shell-based test execution was unavailable in this environment (a chained `&&` command was rejected by the tool policy; single commands for prettier/build worked). Numeric exercise answers were therefore verified by hand derivation (steps recorded above) rather than by running scripts. No repo test suite is named for this chapter's text scope.

## Open issues

- `code/`, images, and components were not touched per scope; the Chinese-language strings in `code/chapter14_dpo/dpo_hands_on.py` and `dpo_math_reward.py` vs the English-only book policy belong to the code worker.
- `review.md` references `output/preference_data.json` from `1-generate_data.py` with an inline fallback, so its coding exercise works with or without running the pipeline.
- No `docs/.vitepress/config.mjs` edits made (coordinator to add the `review.md` sidebar entry).
