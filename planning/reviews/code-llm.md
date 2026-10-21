# Review of code-llm

Scope: `git diff english-only...w/code-llm` (code only: `code/chapter13_rlhf`, `code/chapter14_dpo`,
`code/chapter15_grpo`, `code/chapter17_prm_search`, `code/chapter19_agentic`, `code/chapter23_vlm`,
plus `planning/reports/code-llm.md`). No `docs/` page and no `review.md` was added by this branch,
so the "exercise solutions in a new review.md" check is not applicable.

Independent checks I ran: extracted the branch tree with `git show` and ran the branch's own suite
(`.venv/bin/python -c "pytest ..."` → **62 passed**, matching the report); re-derived every
hand-computed number in the new tests (−1.4 / −1.6, 0.52, 48 children, 0.25/0.5/1.0 discounted
returns, ORM `[0,0,1]`, PRM `[0.865,0.215,0.881]`, 24/24 query classification, DPO loss sign);
re-ran the CJK scan over the six directories (clean, as claimed); and checked the chapter
numbering in `docs/.vitepress/config.mjs` (13.8 / 15.8 confirmed).

## Errors

1. `code/chapter14_dpo/dpo_math_reward.py` — `"return int(match[-1])"` — the English rewrite changed the
   capture groups from `(\d+)` to `([+-]?\d+\.?\d*)` but kept the integer conversion, so any decimal
   answer raises. Verified: `extract_number("The result is 3.5")` → `ValueError: invalid literal for
   int() with base 10: '3.5'`; same for `"Answer: 12.0"`, `"equals 80.5"`, `"the answer is 3."`
   (the trailing-dot form also fails `int("3.")`). `evaluate_math()` calls this on **every** generated
   response, so one decimal output (e.g. "56 / 8 = 7.0") aborts the whole before/after evaluation with an
   unhandled `ValueError`. The old Chinese patterns (`(\d+)`) could not produce this crash, so it is a
   regression introduced by the rewrite, and `test_extract_number_english` only exercises integers so it
   does not catch it. — Suggested fix: parse with `float(...)` and compare with a tolerance (and/or
   restrict the capture to `(\d+)`), e.g. `return float(match[-1])` with `isinstance` checks, plus a
   decimal case in the test.

2. `code/chapter15_grpo/grpo_math_reasoning.py` — "1. Sample a group of G completions from the old
   (sampling) policy … `ratio = exp(log pi_theta - log pi_old)`" — the two are not the same distribution.
   `generate_responses()` samples with `do_sample=True, temperature=0.7, top_p=0.9`, i.e. from the
   temperature/top-p-modified distribution `pi_temp`, but `grpo_policy_update()` builds `old_logp` with
   the *unmodified* model (`_token_log_probs(model, batch_ids)`), so the denominator is `pi_theta`, not
   the distribution the samples actually came from. Consequences: (a) at the first inner epoch the ratio
   is not 1 but `pi_theta/pi_temp`, so the PPO trust region is violated before any update; (b) the
   clipping saturates immediately for tokens whose probability was boosted by the low temperature;
   (c) the "old log-probs" are not the log-probs of the sampled trajectory, so the estimator is not the
   importance ratio the docstring claims. DeepSeekMath samples its GRPO rollouts at temperature 1.0 for
   exactly this reason. — Suggested fix: sample with `temperature=1.0, top_p=1.0` (or compute the
   denominator under the same modified distribution), and state the sampling temperature in the
   "old (sampling) policy" claim.

3. `code/chapter15_grpo/grpo_math_reasoning.py` — "Format bonus: +0.1 if step-by-step reasoning markers
   are present … The final reward is capped at 1.0" — the bonus can never fire. `compute_reward()` sets
   `reward = 1.0` (correct) or `0.0` (wrong) and nothing else, so the branch
   `reward = min(1.0, reward + 0.1)` always evaluates to `min(1.0, 1.1) = 1.0`. The documented
   three-level reward {0, 1.0, 1.1} does not exist; the reward is binary, and the downstream
   `accuracy = mean(1.0 if r >= 1.0 else 0.0)` is only correct *because* of the cap. The rewrite
   re-worded this docstring without noticing that the two statements cancel each other. — Suggested fix:
   either raise the cap to 1.1 (and keep the accuracy test at `r > 0`), or delete the bonus code and
   state that the reward is binary 0/1.

4. `code/chapter15_grpo/grpo_math_reasoning.py` — "How to run (full mode needs a GPU; ~2-4 GB VRAM in
   fp16)" and `--model` help "0.5B fits a 6 GB GPU in fp16" — the arithmetic does not support this for
   the default (non-LoRA) path. Qwen2.5-0.5B is 0.49 B parameters, so in fp16: trainable weights 0.98 GB
   + frozen `copy.deepcopy(model)` reference 0.98 GB + fp16 gradients 0.98 GB + two AdamW moments
   (torch keeps `exp_avg`/`exp_avg_sq` in the parameter dtype, verified: fp16) 1.96 GB ≈ **4.9 GB of
   states before any activation**, plus logits/log-softmax for the group of 4 sequences
   (4 × 230 tokens × 151 936 vocab × 2 B ≈ 0.28 GB per copy, and `grpo_loss` keeps a second copy for the
   reference model) and the backward graph. The "~2-4 GB" figure is only reachable in `--lora` mode
   (no base-model gradients/optimizer state), and even 6 GB is marginal for the default path. — Suggested
   fix: state the measured peak for the default path (or drop the number and recommend `--lora` /
   bfloat16 / an 8-12 GB card).

5. `code/chapter15_grpo/grpo_math_reasoning.py` — `k3_kl_estimate` docstring: "penalizes large deviations
   more steeply than the naive k1 estimator (d itself)". With `d = log pi_ref - log pi_theta`, the k1
   (single-sample) estimator of `KL(pi_theta || pi_ref)` is `log(pi_theta/pi_ref) = -d`; `d` itself
   estimates **-KL**, not KL. The implemented formula `exp(d) - d - 1` is correct (it is Schulman's k3
   and matches DeepSeekMath Eq. (4), and also the book's own 15.1 page), so only the explanation is
   wrong. — Suggested fix: write "the naive k1 estimator (`-d`)".

6. `code/chapter15_grpo/grpo_math_reasoning.py` — `compute_group_advantages` docstring: "The population
   std (unbiased=False) matches the original GRPO definition." DeepSeekMath does not specify whether the
   denominator is `G` or `G-1`; the book's own page `docs/chapter15_grpo/grpo-practice-and-mechanism.md`
   says exactly that ("DeepSeekMath writes `std` without specifying … the example follows the default
   definition of PyTorch `torch.std` and uses Bessel's correction") and its code map therefore uses the
   *unbiased* std. So the docstring states more precisely than anyone can know, and the script silently
   uses a different convention (advantage scale larger by `sqrt(G/(G-1))`) from the chapter it belongs
   to. — Suggested fix: soften to "population std (we use unbiased=False; the paper does not specify)"
   or align with the chapter and use `unbiased=True`.

7. `code/chapter13_rlhf/rlhf_ppo_train.py` — the relevance block now contains the same statement twice:
   ```
               overlap = len(prompt_keywords & response_words)
               overlap = len(prompt_keywords & response_words)
   ```
   The duplicate was introduced by the English rewrite (a leftover from the edit) and is dead code.
   — Suggested fix: delete one line.

8. `docs/chapter23_vlm/vlm-grpo-hands-on.md` — "# Recall Chapter 16: group-relative advantage =
   (R_i - mean) / std" — group-relative advantage is introduced in Chapter 15 (GRPO); Chapter 16 is
   reasoning. This is the page that documents the script the writer rewrote, and the writer stated that
   no docs change was needed. — Suggested fix: "Recall Chapter 15".

9. `docs/chapter23_vlm/qwen3-vl-reflection.md` — "**GRPO's within-group normalization (Chapter 18)**"
   (should be Chapter 15) and "**Multi-dimensional rewards in RLHF (Chapter 15)**" (should be Chapter 13
   — RLHF is Chapter 13, GRPO is Chapter 15; the two look swapped). — Suggested fix: correct both
   chapter numbers.

10. `docs/chapter19_agentic/tool-use-agents.md` — "Let us review the core takeaways from Chapter 9:" in
    the Chapter 19 summary; and `docs/chapter19_agentic/tool-use-and-trajectory.md` /
    `docs/chapter19_agentic/trajectory-synthesis.md` — "How Trajectory Synthesis Connects to Chapter 9"
    (the body of both paragraphs is about RLVR in Chapter 15 and Agentic RL in Chapter 19). Chapter 9 is
    continuous control. — Suggested fix: replace with Chapter 19 (or 15 as the body text says).

11. `docs/chapter13_rlhf/base-model-to-assistant.md` — "## Boundary with Chapter 9" — the section
    immediately below says "Chapters 14-16 depart from this pipeline" and lists DPO/GRPO/RLVR/DAPO;
    Chapter 9 is continuous control. — Suggested fix: "## Boundary with Chapters 14-16".

    (Items 8-11 are pre-existing in `english-only` — this branch edits no `docs/` file — but they sit in
    the six chapters the writer owned, and the report explicitly concludes "No docs changes needed", so
    they were missed.)

## Doubts

1. `code/chapter15_grpo/grpo_math_reasoning.py` — the "before/after" comparison uses
   `test_indices = [0, 7, 14]`, which are three of the 20 *training* problems, while
   `docs/chapter15_grpo/rlvr.md` states "The second script requires model dependencies and compute; its
   result must be evaluated on held-out problems." The rewrite kept the in-sample evaluation (and the
   report did not mention it), so the printed "Before/After training comparison" is not a generalization
   measurement. Pre-existing behavior, but worth fixing now that the script is presented as the
   chapter's GRPO implementation.

2. `code/chapter15_grpo/grpo_math_reasoning.py` — default `--kl-coef 0.01`, while DeepSeekMath uses
   β = 0.04 (the book's 15.1 page states "DeepSeekMath used β = 0.04"). The module header claims "the
   GRPO objective from DeepSeekMath"; a reader may expect the paper's β. Not wrong (it is a CLI flag),
   but the default could match the paper.

3. `code/chapter15_grpo/grpo_math_reasoning.py` — the prompt and each completion are tokenized
   separately (`tokenizer.encode(prompt_text)` and `tokenizer.encode(resp)`) and then concatenated, so
   the BPE boundary between prompt and completion is not the one seen at sampling time; the first
   completion token's log-prob is therefore computed under a slightly different context. Standard
   simplification in teaching code, but it is another reason the "old log-probs" are not exactly the
   log-probs of the sampled trajectory (see Error 2).

4. `planning/reports/code-llm.md` — "Needs external verification … (b) `vlm_grpo_train.py` docstring line
   'similar to Chapter 8, where we run PPO with Stable Baselines3' left as-is". That line does not exist
   in `code/chapter23_vlm/vlm_grpo_train.py` (its docstring only says "Chapter 23"); the line is in
   `docs/chapter13_rlhf/verl-ppo-gsm8k.md` line 11. The verification item is therefore attached to the
   wrong file, and the underlying doc sentence ("Chapter 8, where we run PPO with Stable Baselines3")
   was never actually checked: Chapter 8's own code (`code/chapter08_ppo/ppo_bipedal_walker.py`) is
   hand-written PPO, not Stable Baselines3.

5. `planning/reports/code-llm.md` — "Default 256px behavior unchanged" for the geometry generator is not
   accurate: only the *size* range is unchanged (still 15-30 px); the placement bounds changed from
   `[size+10, img-size-10]` to `[size+5, img-size-5]`, so images generated with the same seed differ
   from the previous version (and the saved `output/geometry_dataset` of the old version no longer
   matches the generator).

6. `code/chapter15_grpo/rule_based_reward.py` — the new pattern `r'[Ss]o[,.]?\s*(?:the answer is\s*)?([+-]?\d+\.?\d*)'`
   in `check_answer_correctness` grabs the *first* number after "So", which is often an intermediate
   value: for "Step 1: 15 + 27 = 42.\nSo 15 + 27 = 42 apples." it extracts `15` (verified), whereas the
   last-number fallback (Method 4) would have produced the correct `42`. It is only reached when no
   "the answer is / final answer is / the result is" marker is present, so the exposure is limited, but
   the pattern is strictly worse than the fallback it shadows.

7. `code/chapter15_grpo/grpo_math_reasoning.py` — full mode trains with `torch.float16` weights and
   `AdamW` without a gradient scaler (`clip_grad_norm_` only). fp16 master weights plus fp16
   `exp_avg_sq` (verified dtype above) risk gradient/moment underflow for a loss whose advantages are
   O(1) but whose token log-prob differences are small; the book's own TRL example uses bfloat16. Worth
   switching the CUDA dtype to bfloat16 (which also removes the need for a GradScaler).

8. `code/chapter19_agentic/multi_turn_rl.py` — `tool_search` now tests `"RL" in query` (raw, case-sensitive
   substring). It hits the intended "RL policy gradients", but it would also fire on any future query
   containing the letters "RL" (e.g. "GERMANY", "CONTROL"). A word-boundary or lowercase check would be
   safer.

9. `code/chapter23_vlm/multi_modal_reward.py` — the rewritten format heuristics are loose substring
   tests on the lower-cased response: `r'total\s*\d*'` matches the word "total" anywhere (e.g. "not
   total"), and `r'in all\s*\d*'` matches "in all". The 0.05 bonus is small, so the effect is minor, but
   the format reward is weaker than the Chinese version it replaced (which anchored on "总共/合计/答:").
   The visual-grounding keyword list has the same property ("top" matching "topic", "left" matching
   "cleft", etc.).

10. `code/chapter19_agentic/agentic_system/run.py` — `main()` is defined *above* the module-level
    `code_reward` and `prompts` definitions it uses. This works when the file is executed as a script
    (the names are resolved at call time, after the module body has run), but `main()` raises `NameError`
    if it is called from an import, and the file is no longer a linear top-to-bottom script. Moving the
    `TASK_EXPECTED_OUTPUTS` / `code_reward` / `prompts` block above `main()` would remove the trap.

## Verdict

This is a careful, mostly correct revision: the English-only conversion is complete and verified
(no CJK in the six directories), the chapter-number relabeling is right (13/14/15/17/19/23, README
sections 13.8 and 15.8 verified against `docs/.vitepress/config.mjs`), the GRPO rewrite is a real
DeepSeekMath-style objective (token-wise ratio, PPO clipping, k3 KL, completion masking,
sequence-mean aggregation — the k3 estimator, the clipping algebra and the −1.4/−1.6 aggregation
values all re-derive correctly), and the new test suite genuinely passes (62 passed, re-run
independently) with meaningful hand-computed checks. The most important problems are localized:
one crash-level regression in `dpo_math_reward.extract_number` (decimals now break `int()`), a
mis-specified "old (sampling) policy" in the GRPO script that makes the PPO ratio and trust region
technically wrong under temperature/top-p sampling, a format-bonus that can never fire, a VRAM
figure that is roughly half of what the default path needs, and two docstrings that over-claim
about the paper (k1 sign, GRPO std convention). The writer also left several wrong chapter
cross-references in the six chapters they owned (Chapter 9/16/18 where 13/15/19 belong) while
concluding that no docs change was needed, and the report misattributes one of its verification
items to a file that does not contain the line in question. None of these undermine the overall
structure of the revision, but the GRPO script should not be presented as "the full DeepSeekMath
GRPO objective" until the sampling distribution, the format bonus and the memory figure are
corrected.
