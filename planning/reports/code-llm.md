# Report: `code-llm`

## Summary

Made the LLM-RL code (`chapter13_rlhf`, `chapter14_dpo`, `chapter15_grpo`,
`chapter17_prm_search`, `chapter19_agentic`, `chapter23_vlm`) English-only,
correct, and tested:

- **English only.** All Chinese training data, prompts, few-shot examples,
  keyword/regex reward logic, and comments replaced with natural English
  equivalents. Verified with a scripted scan: no file in scope contains
  CJK Unified Ideographs (U+4E00–U+9FFF, plus extensions) or CJK/fullwidth
  punctuation. Result: `CLEAN`.
- **A real GRPO.** `code/chapter15_grpo/grpo_math_reasoning.py` was rewritten
  from simplified group-REINFORCE (no ratio, no clipping, no KL — as its own
  docstring admitted) into the full DeepSeekMath-style GRPO objective: group
  sampling, verifiable reward, group-normalized advantages, per-token old-policy
  ratio with PPO clipping, k3 KL to a frozen reference, completion-only masks,
  and a documented sequence-mean vs token-mean aggregation choice. Full mode
  defaults to `Qwen/Qwen2.5-0.5B-Instruct` (fp16 on CUDA, optional `--lora`);
  `--smoke` runs fully offline on a tiny local GPT-2 + character tokenizer.
- **Smoke tests.** One `test_smoke.py` per directory (plus `__init__.py`
  markers so the six same-named test modules collect cleanly). Final gate:
  `62 passed`.
- **Algorithm review.** Fixed real bugs found while testing (details below);
  verified BT loss, DPO formula delegation to TRL, PPO masks/KL handling, and
  PRM/ORM utilities by hand.
- **Stale labels.** All `Chapter N` headers, requirements comments, and the two
  named README sections updated to the current numbering (13/14/15/17/19/23;
  READMEs `8.7`→`13.8`, `16.8`→`15.8`, plus a stale `16.8` in a launch script
  and a stale `docs/chapter10_agentic_rl/...` run path).

No `docs/` pages edited (the chapter snippets show TRL `GRPOTrainer` code, not
this file's code — see below). No sidebar changes needed. Build passes.

## Technical fixes

1. **Real GRPO** — `code/chapter15_grpo/grpo_math_reasoning.py` (full rewrite).
   - Wrong claim (old docstring): "it is not the full DeepSeekMath GRPO
     objective: there is no old-policy ratio, PPO clipping, or
     reference-policy KL term here."
   - Correction: implemented `compute_group_advantages`
     (`(r-mean)/(std+eps)`, population std, all-equal group → zeros),
     `k3_kl_estimate` (`exp(d)-d-1`, `d=log pi_ref-log pi_theta`),
     `grpo_loss` (per-token `min(ratio*A, clip(ratio)*A) + kl_coef*KL` with
     completion mask and `sequence-mean`/`token-mean` aggregation), and
     `grpo_policy_update` (frozen old log-probs + frozen reference, PPO-style
     multi-epoch updates). Training loop keeps sampling→reward→normalize→update
     with a frozen `deepcopy` reference model.
   - Verified: 16 unit tests with fixed tensors (advantage values incl.
     all-equal/singleton groups, clipping for +/− advantages with hand-computed
     losses, k3 non-negativity and zero-on-match, masking, both aggregations
     with hand-computed −1.4/−1.6), plus an offline `--smoke` end-to-end run.

2. **Case-sensitive tone scorer missed capitalized words** —
   `code/chapter13_rlhf/rlhf_ppo_train.py` (`SimpleRewardModel._rule_based_score`).
   - Wrong: `"Please ..."` scored the same as neutral text because
     `"please" in response` is case-sensitive.
   - Correction: match politeness/rudeness word lists against
     `response.lower()`.
   - Verified: `test_rule_based_reward_english_keywords` failed before, passes now.

3. **Substring keyword classifier misrouted queries** —
   `code/chapter19_agentic/tool_use_agent.py` (`ToolPolicy.get_query_type`).
   - Wrong: `"What is the capital of China?"` → `run_code` because `"capital"`
     contains `"api"`; English translation exposed this.
   - Correction: match whole word tokens (`[a-z]+` set) for single-word
     keywords, substring only for phrases (`how much`, `times table`,
     `to the power`); keyword lists converted from Chinese fragments to full
     English words.
   - Verified: `test_query_type_classifier_covers_dataset` checks all 24
     queries classify to their labeled tool.

4. **Case-blind reward extraction returned 0 for valid answers** —
   `code/chapter23_vlm/multi_modal_reward.py` (`reward_correctness`).
   - Wrong: lowercase patterns run against the original mixed-case response, so
     `"Triangles 3, ..."` extracted nothing → 0.0.
   - Correction: `re.findall(pattern, response_lower)`.
   - Verified: `test_reward_correctness_english` failed before, passes now.

5. **Geometry generator crashed on small canvases** —
   `code/chapter23_vlm/geometry_counting_dataset.py` (`generate_single_image`).
   - Wrong: `random.randint(size+10, img-size-10)` with fixed size 15–30 raises
     `ValueError: empty range` for test-size canvases (e.g. 64px).
   - Correction: size adapts to canvas
     (`max(6, min(30, (w-20)//2, (h-20)//2))`); placement bounds clamped with a
     no-margin fallback. Default 256px size range unchanged (15–30px); placement
     margins narrowed from `size+10` to `size+5` (required so small canvases work).
   - Verified: `test_dataset_generation_offline` (64px, tmp dir) passes.

6. **Module-level execution made scripts unimportable** — `dpo_hands_on.py`,
   `dpo_math_reward.py` (wrapped tails in `def main()` + `__main__` guard),
   `2-test_before.py` / `3-train_dpo.py` / `4-test_after.py` (rewrote with
   `main()`; `3-train_dpo` also gained import-safe `load_preference_dataset` /
   `build_trainer` helpers), `tool_use_agent.py` / `multi_turn_rl.py`
   (reconstructed with `main()`), `agentic_system/run.py` (same),
   `generate_synthetic_data.py` (lazy `openai` import + client construction).
   - Verified: `test_heavy_scripts_import_safe` and the full suite import every
     module with no network/model access.
   - Note: one automation slip-up during wrapping corrupted the tails of
     `tool_use_agent.py`/`multi_turn_rl.py` (a `python -c` in-place edit bug);
     both files were reconstructed in full from verified content and re-tested.
     No other files were touched by that script (it aborted before writing).

7. **Stale labels fixed.** Headers/prints/requirements: ch13 `Chapter 8`→`13`
   (3 scripts + requirements); ch14 `Chapter 2`/`7`→`14` (requirements ×2,
   both DPO demos, `0-download_model.py`); ch15 `Chapter 9`→`15` (3 scripts +
   requirements); ch17 `Chapter 13`→`17` (+requirements `Future Trends`→`17`);
   ch19 `Chapter 12`→`19` (2 scripts + requirements); ch23 `Chapter 11`→`23`
   (3 scripts + requirements). READMEs: `verl_gsm8k` `8.7`→`13.8`,
   `verl_code_rlvr` `16.8`→`15.8` (header + section link; also a `16.8` comment
   in `run_qwen_coder_ppo_single_gpu.sh`). Stale run path in
   `agentic_system/deep_research_rl_benchmark.py`
   (`docs/chapter10_agentic_rl/...` → `code/chapter19_agentic/...`).

## Other changes

- English rewrites preserving teaching intent: SFT pairs (15), RM pairs (20),
  PPO prompts/pools, DPO politeness pairs (10), DPO math pairs (10+5,
  `extract_number` now matches `Answer:`/`The answer is`/`The result is`),
  GRPO math set (20), `rule_based_reward` patterns/keywords/cases (English
  step markers, connectives, divide-by-zero checks), ToT/agentic/VLM prompts,
  VLM shape keys renamed `三角形/圆形/正方形` → `triangle/circle/square`
  across all three ch23 scripts (regexes, keyword lists, templates, metadata).
- `SimpleRewardModel` relevance check now uses English word tokens instead of
  Chinese-punctuation splitting.
- `grpo_math_reasoning.py`: `CharTokenizer` + `build_smoke_model_and_tokenizer`
  (2-layer GPT-2, CPU in smoke mode), `--model/--group-size/--epochs/--lr/
--clip-eps/--kl-coef/--aggregation/--update-epochs/--lora/--smoke` CLI,
  generation helpers pass `attention_mask` through.
- Six `test_smoke.py` + six empty `__init__.py` (needed: identical test-module
  basenames collide at collection without packages).
- `planning/reports/code-llm.md` (this file).

## Needs external verification

- None blocking. Full-model GRPO/DPO/PPO training runs were not executed
  (need model download + GPU; covered by offline tests instead).
- Note (review correction): the "Chapter 8, where we run PPO with Stable
  Baselines3" sentence lives in `docs/chapter13_rlhf/verl-ppo-gsm8k.md`
  (line 11), not in `code/chapter23_vlm/vlm_grpo_train.py` as previously
  written here. Whether Chapter 8's `code/chapter08_ppo/ppo_bipedal_walker.py`
  (hand-written PPO, not Stable Baselines3) matches that doc sentence is left
  for the owner of those chapters/docs.

## Sidebar additions

None (no pages added; no `docs/` edits at all).

Docs-snippet check for the GRPO rewrite: `docs/chapter15_grpo/rlvr.md` only
links the file and runs `python grpo_math_reasoning.py` (still valid; new
`--smoke` flag is additive). `docs/chapter15_grpo/grpo-practice-and-mechanism.md`
snippets show TRL `GRPOTrainer` configuration, not this file's code — and the
rewritten script now matches the full-GRPO semantics those pages describe
(ratio + clip, token-wise KL, sequence-mean aggregation). No docs changes needed.

## Commands run and results

- CJK scan (U+4E00–U+9FFF + extensions + CJK/fullwidth punctuation, `.py/.md/
.txt/.sh` in all six dirs): `CLEAN`.
- `.venv/bin/pytest code/chapter13_rlhf code/chapter14_dpo code/chapter15_grpo
code/chapter17_prm_search code/chapter19_agentic code/chapter23_vlm -q`:
  **62 passed** (10 + 8 + 16 + 6 + 13 + 9).
- `npx prettier --write` on the two edited READMEs: clean (Python files are
  outside prettier's parsers; report formatted on write).
- `npx vitepress build docs`: success (`build complete in 20.17s`), no dead links.

## Open issues

- `agentic_system/{trainer,rollout_worker}.py` were reviewed but not
  restructured; `Policy.train_step_with_advantage` uses detached current
  log-probs as the old policy (single-update on-policy simplification) —
  documented in its comment, acceptable for the demo.
- The shared 6 GB GPU was not used for verification (per task: CPU); full-mode
  VRAM fit (0.5B fp16 + frozen ref + optional LoRA) is by construction, not
  measured — worth one GPU run by whoever has the card free.
- `output/` directories are created at import time by several demo scripts
  (`os.makedirs("output", exist_ok=True)`); harmless and left as-is.

## Review response (follow-up to `planning/reviews/code-llm.md`)

All 11 Errors re-checked; 7 fixed in scope, 4 confirmed but out of scope
(docs prose — this task may only touch docs lines showing a command, flag or
path, so they are forwarded to the coordinator). Doubts: 9 accepted/partially
accepted and fixed or documented, 1 rejected with reason.

Errors:

1. `dpo_math_reward.extract_number` decimals crash — accepted and fixed:
   parse with `float()` via a `_to_number` helper (int when integral, else
   float) for both the marker and fallback paths; fallback regex widened to
   `[+-]?\d+\.?\d*`. Verified `extract_number("The result is 3.5") == 3.5`,
   `"Answer: 12.0" == 12`, plus new decimal asserts in
   `test_extract_number_english`. Full suite still passes (float `==` int
   comparison keeps `evaluate_math` working).
2. GRPO sampling distribution vs old-policy ratio — accepted and fixed:
   `generate_responses` now samples at `temperature=1.0, top_p=1.0` with a
   docstring stating the sampling distribution equals the old-policy
   distribution used in the ratio.
3. Format bonus that can never fire — accepted and fixed: removed the dead
   `+0.1`/cap block; `compute_reward` is now documented as binary 0/1, which
   the `r >= 1.0` accuracy statistic already assumes. Existing reward tests
   pass unchanged.
4. VRAM figure "~2-4 GB fp16" — accepted and fixed: module header and
   `--model` help now state full fine-tune needs ~5 GB+ before activations
   (0.5B weights + frozen reference + gradients/Adam states) and recommend
   `--lora` on 6 GB cards.
5. k3 docstring "k1 estimator (d itself)" — accepted and fixed: now reads
   "the naive k1 estimator (−d)".
6. `compute_group_advantages` std-convention over-claim — accepted and fixed:
   docstring now says population std is a convention choice and the paper
   does not specify G vs G-1.
7. Duplicate `overlap = ...` line in `rlhf_ppo_train.py` — accepted and fixed:
   deleted one line.
8. `docs/chapter23_vlm/vlm-grpo-hands-on.md` "Recall Chapter 16" — confirmed
   (line 222) but NOT changed: docs prose is outside this task's scope
   (only command/flag/path lines may be edited); should read Chapter 15.
   Forwarded to the coordinator.
9. `docs/chapter23_vlm/qwen3-vl-reflection.md` swapped numbers — confirmed
   (lines 222/224) but NOT changed (same scope reason); GRPO normalization
   should point to Chapter 15, multi-dimensional RLHF rewards to Chapter 13.
   Forwarded to the coordinator.
10. `docs/chapter19_agentic/*` "Chapter 9" references — confirmed (3 hits) but
    NOT changed (same scope reason); body text points at RLVR/Chapter 15 and
    Agentic RL/Chapter 19. Forwarded to the coordinator.
11. `docs/chapter13_rlhf/base-model-to-assistant.md` "Boundary with Chapter 9"
    — confirmed (line 141; next line already says "Chapters 14-16") but NOT
    changed (same scope reason); should read "Chapters 14-16". Forwarded to
    the coordinator.

Doubts:

1. In-sample before/after evaluation — partially accepted: kept the same
   3 problems (no held-out set exists in the script) but relabeled prints and
   added a comment stating it is an in-sample demo, not a generalization test.
2. Default `--kl-coef 0.01` vs paper β=0.04 — kept (CLI flag, not wrong), but
   the `--kl-coef` help text now notes DeepSeekMath used 0.04.
3. Separate prompt/completion tokenization (BPE boundary) — accepted as a
   documented limitation: noted in the `grpo_policy_update` docstring.
4. Report misattribution of the Stable-Baselines3 line — accepted and fixed:
   "Needs external verification" now points at
   `docs/chapter13_rlhf/verl-ppo-gsm8k.md` line 11 instead of
   `vlm_grpo_train.py`, and leaves the Chapter 8 check to that chapter's owner.
5. "Default 256px behavior unchanged" — accepted as inaccurate wording: fixed
   above (size range unchanged; `size+10`→`size+5` margins changed
   intentionally so small canvases work). No code change.
6. Over-broad `[Ss]o ...` pattern shadowing the fallback — accepted and fixed:
   removed the pattern (with an explanatory comment); verified
   `"Step 1: 15 + 27 = 42.\nSo 15 + 27 = 42 apples."` now extracts `42` via
   the last-number fallback instead of `15`.
7. fp16 master weights/optimizer states — accepted and fixed: full mode now
   loads with `torch.bfloat16` on CUDA (also removes any GradScaler need).
8. `"RL" in query` substring — accepted and fixed: `re.search(r"\bRL\b",
query)` (verified "PEARL" no longer triggers, "RL policy gradients"
   still does).
9. Loose format/visual heuristics — partially accepted and tightened:
   summary patterns now require `in total` / `total`+digits / `altogether` /
   `in all` / `answer:` (bare "not total" no longer matches), and both the
   reasoning-keyword and visual-keyword lists match on word boundaries
   ("top" no longer matches "topic"). Comparative smoke tests still pass.
10. `agentic_system/run.py` use-before-definition layout — accepted and fixed:
    moved `TASK_EXPECTED_OUTPUTS` / `code_reward` / `prompts` above `main()`.

Follow-up verification: `.venv/bin/pytest code/chapter13_rlhf
code/chapter14_dpo code/chapter15_grpo code/chapter17_prm_search
code/chapter19_agentic code/chapter23_vlm -q` → **62 passed**; CJK scan →
`CLEAN`; `npx prettier --write` on changed files (Python files have no
prettier parser, as before; report formatted); `npx vitepress build docs`
re-run below.

- `agentic_system/{trainer,rollout_worker}.py` were reviewed but not
  restructured; `Policy.train_step_with_advantage` uses detached current
  log-probs as the old policy (single-update on-policy simplification) —
  documented in its comment, acceptable for the demo.
- The shared 6 GB GPU was not used for verification (per task: CPU); full-mode
  VRAM fit (0.5B fp16 + frozen ref + optional LoRA) is by construction, not
  measured — worth one GPU run by whoever has the card free.
- `output/` directories are created at import time by several demo scripts
  (`os.makedirs("output", exist_ok=True)`); harmless and left as-is.
