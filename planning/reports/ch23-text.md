# Report: ch23-text (Chapter 23 text pass)

## Summary

Text pass over all five in-scope pages in `docs/chapter23_vlm/`. Found and fixed one load-bearing typo repeated 5× (`verl.tainer.main` → `verl.trainer.main`), four wrong chapter cross-references (old-outline numbers), one wrong table header (`11.1` → `23.3`), one self-contradictory config sentence, one wrong "previous section" pointer, a stale chapter-number table in the supplement, and front-matter/H1 title mismatches. Recomputed every numeric example in the chapter (patch tokens, GRPO advantages, IoU, unit conversions, reward arithmetic) — all check out. Created `docs/chapter23_vlm/review.md` (23.R Summary and Exercises) with 7 learning outcomes, 7 key equations with readings, 6 pitfalls, and 8 exercises with fully worked, numerically verified solutions. `npx prettier --write` clean; `npx vitepress build docs` succeeds.

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

- `23.R Summary and Exercises` → `/chapter23_vlm/review` (new file; coordinator to insert after 23.4).
- Note: `docs/chapter23_vlm/vlm-frameworks.md` ("Supplement: VLM RL Frameworks and Frontiers") exists and is linked from chapter text but has **no sidebar entry** — coordinator to decide whether to add it (e.g. as `23.S Supplement: VLM RL Frameworks and Frontiers` → `/chapter23_vlm/vlm-frameworks`) or leave it as an unlisted supplement. The new `review.md` does not link to it, so the build is unaffected either way.
- Title mismatch for coordinator: sidebar says `23.1 Visual Reward Design` while the page H1 (and now front matter) is `23.1 Why a VLM Can Guess Right Without Looking: Visual Rewards and Hallucination`; `fix:headings` may want to reconcile. 23.2 has a milder variant (sidebar `23.2 Visual Reflection RL` vs H1 `23.2 Visual Reflection RL: Answering with Evidence`).

## Commands run and results

- `timeout 60 .venv/bin/python -c` advantage check `[1.5,1.0,0.3,0.1]` → mean 0.725, std ≈ 0.5585, adv ≈ [1.388, 0.492, −0.761, −1.119], sum 0.0. Pass.
- Same harness for `[1,1,0,0,0,0,0,0]` → adv correct ≈ 1.732, wrong ≈ −0.577. Pass.
- IoU check `900/2300` ≈ 0.3913. Pass.
- `npx prettier --write` on all six files — clean (one table realignment).
- `npx vitepress build docs` — succeeds, pages render, no dead-link failures.

## Open issues

- The `<thinkutan>`/`</think\s*>` regex mismatch described in the §23.4.4 details box concerns the *upstream* EasyR1 `r1v.py`; the code printed in §23.4.5 already uses a consistent regex, so no text change was needed — but if upstream fixes their regex, the details box becomes stale.
- `code/chapter23_vlm/` scripts use Chinese-language prompts/templates while the chapter text is English-only; code was out of scope (another worker owns it) — flagging in case the coordinator wants a consistency pass.
- Review-page coding exercises were validated by reading the code and by executing the underlying arithmetic, not by running the full simulation scripts (they open interactive `plt.show()` windows); solutions state expected deltas by construction.
