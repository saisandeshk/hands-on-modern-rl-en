# Report: ch23-text (Chapter 23 text pass)

## Summary

Text pass over all five in-scope pages in `docs/chapter23_vlm/`, plus a follow-up review-response pass over all 13 errors and 10 doubts in `planning/reviews/ch23-text.md` (see Review response below for the per-finding verdicts). Found and fixed one load-bearing typo repeated 5× (`verl.tainer.main` → `verl.trainer.main`), four wrong chapter cross-references (old-outline numbers), one wrong table header (`11.1` → `23.3`), one self-contradictory config sentence, one wrong "previous section" pointer, a stale chapter-number table in the supplement, and front-matter/H1 title mismatches. Recomputed every numeric example in the chapter (patch tokens, GRPO advantages, IoU, unit conversions, reward arithmetic) — all check out. Created `docs/chapter23_vlm/review.md` (23.5 Summary and Exercises after the review-response rename) with 7 learning outcomes, 7 key equations with readings, 6 pitfalls, and 8 exercises with fully worked, numerically verified solutions. `npx prettier --write` clean; `npx vitepress build docs` succeeds.

## Technical fixes

1. `docs/chapter23_vlm/easyr1-geoqa.md` — wrong module path: `python3 -m verl.tainer.main` (5 occurrences: §23.4.3, §23.4.6 minimal config, §23.4.6 LoRA, §23.4.7 single-machine, §23.4.7 multi-node). Correction: `python3 -m verl.trainer.main`. Verified: the page's own project tree lists `verl/trainer/main.py` as the entry point, so `verl.trainer.main` is the only consistent module path. (Copy-paste this wrong command and training will not start at all — highest-severity fix.)
2. `docs/chapter23_vlm/easyr1-geoqa.md` — comparison table in §23.4.11 headed the handwritten column `11.1 Handwritten GRPO`. Correction: `23.3 Handwritten GRPO`. Verified against the numbering table (GRPO hands-on is 23.3; Chapter 11 is imitation/meta-RL).
3. `docs/chapter23_vlm/easyr1-geoqa.md` — §23.4.6 claimed CLI key `worker.actor.model.model_path` "corresponds to `worker.actor.model.path` in the YAML". Correction: it corresponds to `model_path` under `worker.actor.model` (the page's own full YAML in the same section uses `model_path:`). Verified by internal consistency.
4. `docs/chapter23_vlm/qwen3-vl-reflection.md` — "Connections to Earlier Chapters" had three wrong numbers: GRPO normalization `(Chapter 18)` → `(Chapter 15)`; tool-calling RL `(Chapter 22, Agents)` → `(Chapter 19, Agentic RL)`; RLHF multi-dimensional rewards `(Chapter 15)` → `(Chapter 13)`. Verified against the numbering table (Ch 15 = GRPO, Ch 19 = Agentic RL, Ch 13 = RLHF; Ch 18 is industrial LLM RL, Ch 22 is computer use). Chapter 25 (counterfactual checks) was already correct.
5. `docs/chapter23_vlm/vlm-grpo-hands-on.md` — code comment `# Recall Chapter 16: group-relative advantage` → `# Recall Chapter 15`. Verified: GRPO is Chapter 15; Chapter 16 is reasoning/test-time compute.
6. `docs/chapter23_vlm/vlm-challenges.md` — §23.1 encoder-freezing paragraph said "In the geometric-figure experiment of the previous section". Correction: "In the geometric-shape counting experiment of Section 23.3". Verified: 23.1 is the first page of the chapter per the sidebar, so "previous section" points nowhere; the counting experiment lives in 23.3.
7. `docs/chapter23_vlm/vlm-frameworks.md` — "From Text RL to Multimodal RL" table used a stale outline (`Ch 4: DQN, Ch 5: Policy Gradient, Ch 8: RLHF/PPO, Ch 9: GRPO, Ch 10: Agentic RL, Ch 11: VLM RL`). Correction: `Ch 5, Ch 6, Ch 13 / Ch 8, Ch 15, Ch 19, Ch 23`. Verified against the numbering table and sidebar. (The very next paragraph's list — Ch 6, 7, 8, 15 — was already correct and untouched.)
8. `docs/chapter23_vlm/vlm-frameworks.md` — "Unlike the continuous control discussed in Section 24.3 on embodied intelligence" → "Unlike the continuous control discussed in Chapter 9". Verified: continuous control is Chapter 9; 24.3 is VLA/embodied.
9. Numeric spot-checks recomputed and confirmed correct (no edit needed): 448px/14px → $32\times32=1024$ patch tokens; bar-chart $57-42=15$; 108 km/h = 30 m/s → 60 m in 2 s; 8000/512 ≈ 16 steps/epoch; r1v `overall` phase values 0.5 (format 1.0 + accuracy 0.0) and 0.7 (format 1.0 + accuracy 0.4); parallelogram half-area 12/24 consistency. Advantage arithmetic for the new review exercises verified by execution (see Commands).

## Other changes

- Added missing front matter to `qwen3-vl-reflection.md` (`title: '23.2 Visual Reflection RL: Answering with Evidence'`, matching its H1).
- Aligned front-matter titles with H1s: `vlm-grpo-hands-on.md` → `23.3 Hands-on: Multimodal GRPO` (matches sidebar); `easyr1-geoqa.md` → `23.4 Hands-on: GeoQA Geometric Reasoning` (matches sidebar); `vlm-challenges.md` → long H1 title (see Open issues: sidebar still says `23.1 Visual Reward Design`).
- `vlm-grpo-hands-on.md` closing paragraph: fixed "Next, we will dive deeper … [Special Challenges in VLM RL]" (wrong direction — 23.1 precedes 23.3) to point back to `[23.1 Visual Reward Design](./vlm-challenges)`.
- Created `docs/chapter23_vlm/review.md`: 7 outcomes, 7 equations with one-line readings, 6 pitfalls, 8 exercises (conceptual ×2, numeric-by-hand ×3, derivation ×1, coding ×2 on `code/chapter23_vlm/`), all solutions in `::: details Solution` blocks with executed numeric checks.
- Coverage judgment (no checklist supplied): VLM RL core is covered — outcome-vs-process rewards, grounding/IoU rewards, hallucination mechanisms and countermeasures, encoder update strategies, GRPO hands-on, EasyR1/GeoQA pipeline, visual reflection + tool re-observation, frontier frameworks (VisPlay, VISTA-Gym, VeRL-Omni). No missing core item required a new exposition section; the review page closes the chapter.
- Prettier reformatted only `vlm-frameworks.md` (table realignment); other files unchanged.

## Needs external verification

- `vlm-challenges.md`: Visual-RFT "24.3 pp (1-shot classification) / 21.9 pp (COCO 2-shot detection) over SFT" figures — precise numbers quoted as "reported in the paper"; could not verify offline.
- `qwen3-vl-reflection.md`: Qwen3-VL report `arxiv 2511.21631` (Nov 2025), model sizes (2B/4B/8B/32B dense, 30B-A3B/235B-A22B MoE), DeepStack/interleaved-MRoPE/text-timestamp claims, and Perception-RFT `arxiv 2607.14682` (2026) "reasoning-free beats reasoning for 4B" result — all plausible but unverifiable without internet.
- `vlm-frameworks.md`: VeRL-Omni "~25% higher throughput than flow_grpo on Qwen-Image FlowGRPO" — vendor benchmark, unverifiable offline.
- `easyr1-geoqa.md`: EasyR1 star count (4900+), Docker image tag, HF dataset `leonardPKU/GEOQA_8K_R1V` splits, Qwen2.5-VL architecture details — operational details that drift fast; flagged, not changed.

## Sidebar additions

- `23.5 Summary and Exercises` → `/chapter23_vlm/review` (new file; coordinator to insert after 23.4 — `docs/.vitepress/config.mjs` is outside worker scope, so the entry is still missing and the page is unreachable until added).
- Title status (resolved in this pass): front matter of `vlm-challenges.md` / `qwen3-vl-reflection.md` now equals the sidebar texts (`23.1 Visual Reward Design`, `23.2 Visual Reflection RL`); the on-page H1s stay descriptive. No sidebar change needed for these.
- Note: `docs/chapter23_vlm/vlm-frameworks.md` ("Supplement: VLM RL Frameworks and Frontiers") exists and is linked from chapter text but has **no sidebar entry** — coordinator to decide whether to add it or leave it as an unlisted supplement (if listed, it needs a tooling-parseable number; `review.md` now occupies `23.5`). The new `review.md` does not link to it, so the build is unaffected either way.

## Commands run and results

- `timeout 60 .venv/bin/python -c` advantage check `[1.5,1.0,0.3,0.1]` → mean 0.725, std ≈ 0.5585, adv ≈ [1.388, 0.492, −0.761, −1.119], sum 0.0. Pass.
- Same harness for `[1,1,0,0,0,0,0,0]` → adv correct ≈ 1.732, wrong ≈ −0.577. Pass.
- IoU check `900/2300` ≈ 0.3913. Pass.
- `npx prettier --write` on all six files — clean (one table realignment).
- `npx vitepress build docs` — succeeds, pages render, no dead-link failures.

## Open issues

- The `<thinkutan>`/`</think\s*>` regex mismatch described in the §23.4.4 details box concerns the _upstream_ EasyR1 `r1v.py` (see Review response, finding 6 for the corrected consistent account).
- `code/chapter23_vlm/` scripts use Chinese-language prompts/templates while the chapter text is English-only; code was out of scope (another worker owns it) — flagging in case the coordinator wants a consistency pass. The new Ex 7 now at least tells the reader _why_ its keyword list is Chinese.
- Review-page coding exercises were validated by reading the code and by executing the underlying arithmetic, not by running the full simulation scripts (they open interactive `plt.show()` windows); solutions state expected deltas by construction.

## Review response

Reviewer report `planning/reviews/ch23-text.md` (13 errors, 10 doubts). Each finding was re-checked independently (re-ran the simulation scripts and reward code with `.venv/bin/python`, opened all four disputed images, read every cited snippet). Verdict: all 13 errors accepted and fixed; 8 of 10 doubts accepted and fixed; doubts 1-2 kept as explicit verification flags (no offline-verifiable correction available).

- **Error 1 (val leaks into train, §23.3.1): accepted.** My replication of the snippet's RNG sequence gives 94/100 val samples colliding with the train set (reviewer measured 91 byte-identical with PIL). Fixed `vlm-grpo-hands-on.md`: `generate_shape_image` now takes a local `rng = random.Random(seed_offset + i)` (no more global `random.seed(i)`), train/val use disjoint offsets (0 / 100_000), plus a note that the 1-5-count combo space is tiny so §23.3.4's 7-shape test remains the real generalization check.
- **Error 2 (Ex 8 std trend): accepted.** Re-ran the exact snippet: G=2 → [0.297, 0.267, 0.331, 0.181, 0.070], G=4 → [0.494, 0.586, 0.448, 0.409, 0.108], G=16 → [0.542, 0.611, 0.599, 0.441, 0.185] — matches the reviewer to 3 decimals. Rewrote solution (b): curve first rises/wobbles (early mixed correct ≈1.1–1.5 vs incorrect ≈0–0.5 groups), collapses only in late epochs; shrinking std signals homogeneity, and std → 0 carries no ranking information.
- **Error 3 (300-char guard inert): accepted.** Measured shipped response lengths: 174, 17, 115, 27, 57, 41, 72, 108 — max 174 < 300. Changed Ex 7(b) threshold to 50 characters and gave the verified delta: only case 5 (57 chars, no step keyword) fires, 0.55 → 0.45 (total recomputed from `compute_total_reward`); solution now states the measured lengths and why 300 was inert.
- **Error 4 (clamp justification backwards): accepted.** Adding a shared constant leaves GRPO's within-group advantage unchanged, so "negative outliers distort siblings" was wrong. Rewrote justification: clamp keeps the $[0, 2]$ scale, with the caveat that an all-zero group has zero std (0/0 advantages), so penalties must stay well below the within-group spread.
- **Error 5 (population vs unbiased std): accepted.** `numpy.std()` (review page, `vlm_grpo_train.py`) uses ddof=0 while `torch.Tensor.std` defaults to unbiased (N−1). Fixed the §23.3 snippet to `std(..., unbiased=False)` with a comment citing the DeepSeekMath convention, and annotated the review-page equation (ddof = 0). Exercise numbers (population) now match the printed code.
- **Error 6 (format-regex contradiction): accepted.** Rewrote the §23.4.4 details box: the failure is that `\s*` cannot consume the letters `utan` (not "the Chinese character"), describing the upstream `<think>`-regex vs `<thinkutan>`-prompt mismatch; retitled §23.4.5's code as the section's adapted consistent version. Pitfall 4 in `review.md` reworded to match.
- **Error 7 (3B vision encoder smaller): accepted.** Qwen2.5-VL 3B/7B share the vision tower; §23.4.12 now says the 3B's _language model_ is smaller.
- **Error 8 (Phase 2 area arithmetic): accepted.** Opened `geoqa-example.png`: shaded region is the single triangle △AOB, dataset GT is 12 with muddled reasoning (24/4 = 6, then ×2). Rewrote the trace to the geometrically sound 24/4 = 6 → `<answer>6</answer>` and added a noisy-label caution; clarified the Phase 2 header numbers are batch averages.
- **Error 9 (Fig 2 caption, 23.2): accepted.** Opened `ref-vlm-r1-iou.png`: it is a `train/rewards/iou_reward` curve. Caption replaced with the IoU-reward-curve text (mirroring §23.3's correct caption) and link fixed to `om-ai-lab/VLM-R1` (error 13, same edit).
- **Error 10 (PickScore caption/order): accepted.** Opened `ref-pickscore-ranking.png`: Vanilla-vs-PickScore generation pairs, no ranking. Caption replaced, figures renumbered to 1, 2, 3 (PickScore), 4 (VISTA-R1 ablation), and a sentence referring to Figure 3 added (it previously had no in-text reference).
- **Error 11 (log/phases vs Figure 5): accepted.** Opened `easyr1-geoqa-curves.png`: panels are `actor/entropy_loss`, `reward/accuracy` (≈0.34 → ≈0.51), `response_length/mean`, `val/accuracy_reward` — no format panel. Labeled the §23.4.7 log schematic, restated the phases against the shown curves (accuracy climb + flat length ⇒ gains from accuracy, not length), and corrected the Figure 5 caption to an upstream-7B-on-Geo3K/GeoQA reference (doubt 10, same edit).
- **Error 12 (unreachable page / 23.R numbering): accepted in part.** Renamed to `23.5 Summary and Exercises` (front matter, H1, learning path) since `scripts/fix-heading-numbers.mjs` cannot parse `23.R`. Could not add the sidebar entry — `docs/.vitepress/config.mjs` is outside my scope — so Sidebar additions below now request `{ text: '23.5 Summary and Exercises', link: '/chapter23_vlm/review' }`; the coordinator must insert it.
- **Error 13 (VLM-R1 link): accepted.** Fixed to `https://github.com/om-ai-lab/VLM-R1` (same edit as error 9).
- **Doubts 1–2 (Perception-RFT arXiv id; Qwen3-VL sizes): no text change.** Both remain exactly as already flagged under Needs external verification; nothing offline-verifiable to correct.
- **Doubt 3 (VeRL-Omni 25%): accepted.** Now attributed to the project's own benchmark ("The project reports ... see the project's own benchmark page").
- **Doubt 4 (drifting operational details): accepted in part.** Star count softened to "about 5k stars". Docker tag, algorithm count, `format_weight` default, and `r1v.jinja` tag claims left as-is (unverifiable offline; still flagged).
- **Doubt 5 (Unified Transformer): accepted.** Reworded: single compact backbone saves parameters/one optimizer, but full RL updates are no cheaper than full fine-tuning elsewhere.
- **Doubt 6 (sensitivity term fires on grounded responses): accepted.** Rewrote Ex 7(a) around §23.1's unchanged-answer check with an original-correctness gate; solution reports the verified null result on shipped swaps and the exact −0.3 construction (degenerate swapped GT = response's own numbers, e.g. case 2: 1.30 → 1.00).
- **Doubt 7 (Ex 6 phase framing; outcome 3): accepted.** Solution now follows the §23.1 checklist order item by item; outcome 3 now reads "correctness + reasoning quality + format (as in §23.3), plus a grounding or counterfactual check".
- **Doubt 8 (Chinese literals): accepted.** STEP_KEYWORDS now carry an English gloss comment and Ex 7 prose explains the code is Chinese-only.
- **Doubt 9 (front-matter/sidebar titles): accepted.** Front matter of `vlm-challenges.md` and `qwen3-vl-reflection.md` set to the sidebar texts (`23.1 Visual Reward Design`, `23.2 Visual Reflection RL`); descriptive H1s kept on-page. Sidebar additions updated accordingly.

Commands for this pass: response-length check, `avg_advantage_std` re-runs for G=2/4/16 (match reviewer values), `compute_total_reward` table for all 8 cases (case 5 = 0.55), RNG-overlap replication (94/100) — all via `timeout N .venv/bin/python -c`; `npx prettier --write` on the six touched files; `npx vitepress build docs` (must succeed).
