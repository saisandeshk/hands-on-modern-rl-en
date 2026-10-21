# Report: appC-text — Text pass over `docs/appendix_c_resources/`

## Summary

Reviewed and fixed all four Appendix C pages (`learning-resources.md`, `gpu-hours-estimation.md`,
`metrics-glossary.md`, `industrial-exercises.md`) for duplicated entries, well-known-work
accuracy, internal table consistency, and chapter cross-reference numbering. All 16 in-repo
chapter/appendix cross-reference targets were verified to exist on disk; all links use the
current Part I–VII / chapter 1–26 numbering. Fixed one arithmetically impossible table entry
(Llama 3 8B GPU hours), one inconsistent cost breakdown (DeepSeek-V3 $5.576M), a duplicated
exercise, a duplicated paper entry, wrong team attribution (DAPO), and several smaller
consistency errors. `npx vitepress build docs` passes. No pages added, no code touched.

## Technical fixes

1. **`learning-resources.md` — duplicated DQN (Nature, 2015) entry (C.1.1 vs C.1.2).**
   The same Mnih et al. 2015 paper appeared as a full entry in both tables. Converted the
   C.1.2 row into an explicit "Already listed in C.1.1" pointer, matching the convention this
   file already uses for DAPO/GSPO/CISPO/Kimi K2 in C.1.5. Verified by comparing titles, years,
   and contribution text of both rows.
2. **`learning-resources.md` — DAPO attributed to "the Qwen team" (two places: C.1.3 table and
   C.1.5.1 pointer).** DAPO (arXiv:2503.14476, Yu et al.) is a ByteDance Seed-team work
   (repo `BytedTsinghua-SIA/DAPO`, also referenced as such later in the same file). Changed both
   to "ByteDance Seed team". Verified from well-known authorship plus the file's own DAPO repo link.
3. **`learning-resources.md` — PETS contribution "cognitive and epistemic uncertainty".**
   "Cognitive" is a garble; the standard distinction is aleatoric vs epistemic uncertainty
   (Chua et al., PETS, arXiv:1805.12114, ensemble of probabilistic dynamics models). Fixed.
4. **`learning-resources.md` — "100+ papers" claim.** Counted table rows after the fix:
   12 (C.1.1) + 17 (C.1.2) + 15 (C.1.3) + 14 (C.1.4) + 25 (C.1.5) = 83 rows, of which 5 are
   intentional "already listed" pointers, i.e. ~78 unique papers. Changed "The 100+ papers" to
   "The 80+ entries". Verified by direct row count.
5. **`learning-resources.md` — safety reading path cites "Sutton & Barto, Chapter 15 on
   exploration and exploitation".** Wrong: in Sutton & Barto (2nd ed.) Chapter 15 is
   Neuroscience; exploration/exploitation is Chapter 2 (multi-armed bandits). Fixed the reference.
6. **`learning-resources.md` — correspondence box maps "Chapter 8 PPO → C.1.1 Schulman PPO".**
   The PPO paper (Schulman et al. 2017) is listed in C.1.2, not C.1.1 (C.1.1 holds TRPO).
   Fixed to "C.1.2 Schulman PPO".
7. **`learning-resources.md` — broken DDPO link `https://github.com/jannerm/dd/F`.**
   Truncated URL (the `/F` is a fragment of a table). Correct is `https://github.com/jannerm/ddpo`
   (Jan Peters-style DDPO reference implementation; consistent with the adjacent Diffusers-DDPO
   and `ddpo-pytorch` rows). Fixed.
8. **`learning-resources.md` — "Zhang Zihua's textbook".** Author name order garbled; the book is
   by Zhihua Zhang (Peking University). Fixed to "Zhihua Zhang's", matching the section heading.
9. **`learning-resources.md` — "Sutton's university's RL introductory course".** Garbled possessive.
   Fixed to "The University of Alberta's introductory RL course" (CMPUT 365, Marlos Machado —
   per the entry itself).
10. **`learning-resources.md` — missing section number IV.** Resource sections ran
    I, II, III, then V ("Practical Tutorials"), with the Lambert/Ryu/DeepLearning.AI/HF LLM-alignment
    courses orphaned under III. Inserted "**IV. LLM Alignment and Post-Training Courses**" header
    before the Lambert entry; no content moved.
11. **`learning-resources.md` — MIT 6.7920 described as "exploitation" vs "exploration".**
    Misleading labels for a foundations-vs-frontiers split. Reworded to "established
    foundations … frontier topics", keeping the lecture counts.
12. **`learning-resources.md` — "C.1.4 (first five chapters)".** C.1.4 has papers, not chapters.
    Fixed to "first five papers".
13. **`gpu-hours-estimation.md` — Llama 3 8B row arithmetically impossible.**
    15T tokens × 8B params at 130K H100 hours implies ~1500 TFLOPS sustained, above the H100
    BF16 peak (~989 TFLOPS). Recomputed via the page's own formula 6ND/(MFU·FLOPS):
    6·8e9·15e12/(0.35·989e12) ≈ 1.8e9 s ≈ 500K hours. Changed 130,000 → ~500,000 and marked the
    hours as estimated from 6ND. Cross-checked neighboring rows the same way (Llama 2 7B/13B/70B,
    Llama 3 70B/405B all imply MFU 28–41%, i.e. plausible — only the 8B row was impossible).
14. **`gpu-hours-estimation.md` — DeepSeek-V3 $5.576M breakdown did not add up.**
    2.664M × $2 = $5.328M, plus "12K" and "8K" hours ($0.04M) = $5.368M ≠ $5.576M. The reported
    total corresponds to 2.788M hours at $2/h, so post-training/validation/ablation must be the
    remaining ~124K hours (~$0.248M). Rewrote the box accordingly; totals now consistent.
15. **`gpu-hours-estimation.md` — "fourfold increase in GPU hours" (Llama 2 70B → Llama 3 70B).**
    Numerically ~3.8× in raw hours but compares A100 hours to H100 hours. Added the caveat that
    the GPU generation changed, so the FLOP increase is larger.
16. **`gpu-hours-estimation.md` — "Kimi K5 tech report" for the Kimi K2.5 row.** Internal name
    mismatch; fixed to "Kimi K2.5 tech report".
17. **`gpu-hours-estimation.md` — typos:** `B**F16` → `BF16`; broken list `...larger models. 2.**\n
Early Stopping**` → proper `2. **Early Stopping**` item (renumbered Shared Checkpoint to 3).
18. **`gpu-hours-estimation.md` — H1 "C.2 GPU Hour Estimation Table" vs sidebar "C.2 GPU Hours
    Estimation Table".** Fixed H1 to "GPU Hours"; added missing `title:` front matter to this page
    and to `learning-resources.md` per repo convention.
19. **`metrics-glossary.md` — front matter `title: C.3 Training Metrics Glossary` vs H1/sidebar
    "C.3 Metrics Glossary".** Fixed front matter to match.
20. **`metrics-glossary.md` — MFU "typically 50%–60%" contradicts C.2.1 "MFU ranges 30%–55%".**
    Aligned §10 to "typically 40%–55%" (inside C.2.1's envelope). Verified no other MFU figures conflict.
21. **`metrics-glossary.md` — KL metric named `actor/ppo_kl` in §11/§12 while §2.4 defines
    `actor/approx_kl`.** Standardized all three spots to `actor/approx_kl`.
22. **`industrial-exercises.md` — duplicated "### Exercise 2: Minimal Domain Post-Training Loop".**
    Two Exercise 2 headers back-to-back; the first block's tasks ("Collect 20 job posts…") duplicated
    Exercise 1's content under the wrong title, making nine headers under "Eight Industrial
    Exercises". Deleted the erroneous block; exercises are now 1–8 with the correct domain
    post-training Exercise 2 intact.
23. **Cross-references verified (all exist):** `chapter08_ppo/ppo-bipedal-walker`,
    `chapter08_ppo/gae-reward-model`, `chapter13_rlhf/base-model-to-assistant`,
    `chapter13_rlhf/hhh-practice`, `chapter14_dpo/dpo-objective-derivation`,
    `chapter15_grpo/grpo-practice-and-mechanism`, `chapter09_continuous_control/ddpg`,
    `chapter25_alignment_failures/classical-failures`,
    `chapter18_llm_rl_industrial/single-machine-to-industrial`,
    `chapter20_swe_agents/swe-bench-and-rlvr`, `chapter22_computer_use/training`,
    `chapter24_multimodal_frontier/audio-reward-design`,
    `chapter17_prm_search/outcome-vs-process`, `appendix_a_engineering/training-debugging`,
    `appendix_b_algorithms/sft-kl`, `./gpu-hours-estimation`, `/preface/env-setup`.
    Chapter numbers in prose (8, 9, 13, 14, 15, 17, 18, 20, 22, 24, 25) match the numbering table.

## Other changes

- Added `title:` front matter to `learning-resources.md` (was missing; other pages have it).
- Spot-checked well-known paper metadata (Sutton & Barto 2018, Watkins & Dayan 1992, TD-Gammon
  1995, Sutton 1988, REINFORCE 1992, TRPO 2015, A3C 2016, PPO 2017, DDPG 2015, TD3 2018,
  SAC 2018, Rainbow 2017, AlphaZero 2018, MuZero 2019, DreamerV3 2023, C51 2017, PER 2015,
  Dueling 2015, Double-DQN 2015, MBPO 2019, InstructGPT 2022, Constitutional AI 2022, DPO 2023,
  DeepSeekMath 2024, R1 2025, ReFT 2024, Self-Rewarding LMs 2024, Sleeper Agents 2024, Alignment
  Faking 2024, In-context Scheming 2024, Sycophancy 2023, Sycophancy-to-Subterfuge 2024,
  RM-overoptimization scaling laws 2022, mesa-optimization 2019, Instruction Hierarchy 2024,
  rStar-Math 2025, SWE-RL 2025, Qwen3 2025, DeepSeek-V3 2024, Titans 2025, AlphaEvolve 2025,
  Genie 3 2025): titles/authors/years/venues all check out against solid knowledge; no changes needed.
- Cost-table arithmetic spot-checked (price × hours products, e.g. 405B 30M×$2.5=$75M): consistent.
- No entries deleted outright; only the erroneous duplicate exercise block was removed (restored
  numbering 1–8) and the duplicate DQN row converted to the file's own "already listed" pointer style.

## Needs external verification

Cannot open URLs (no internet); the following could not be settled from solid knowledge and should
be checked by someone with access before print:

1. **David Silver 2015 course vs "DeepMind x UCL RL Lecture Series (2021)" share the identical
   YouTube playlist URL** (`PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb`). One of them is very likely wrong;
   flagged, not changed.
2. **2025–2026-dated preprints/tech reports** (kept as-is; plausible but unverifiable offline):
   GSPO (2507.18071), CISPO/MiniMax-M1 (2506.13585), ThinkPRM (2504.16828), DeepSeek-Prover-V2
   (2504.21801, incl. the "88.9% on MiniF2F" figure), DAPO details, VAPO (2504.05118),
   Free Process Rewards (2412.01981), PRM Rewarding Progress (2410.08146), Tajwar et al.
   (2404.14367), Anthropic emergent-misalignment report (2511.18397), School of Reward Hacks
   (2508.17511), METR Frontier Reward Hacking (2024, no arXiv given), Kimi K2/CWM/Wei et al.
   self-play SWE-RL/Step-Audio-R1/Tongyi DeepResearch/Titans/M-GRPO/Kimi K2.5 entries, Seedance 1.0,
   DanceGRPO, LongCat-Video, Ishibashi et al. (2410.15639). In particular the **Kimi K2 ↔ GSPO
   linkage** (K2 report validating GSPO at scale) and **future-dated items** (Wei et al. 2512.18552,
   Kimi K2.5 2602.02276) deserve a check.
3. **GPU-hour estimates marked "~" or "Estimate"** (Qwen2.5/3 rows, Kimi K2 ~9M, Mistral/Mixtral,
   R1 RL-phase 128K/80K and their attribution to the R1 tech report, Step-2, GLM-4.6): kept with
   estimate labeling; the Kimi K2 figure implies unusually low MFU (~9% on H800-class hardware
   if taken literally) and should be sanity-checked against the tech report.
4. **Footnote career-posting URLs** in `industrial-exercises.md` (job IDs, repost links): plausible
   but unverifiable offline; role descriptions are generic enough that link rot is the main risk.
5. **Minor numeric claims left as-is:** DeepSeek-Prover-V2 "88.9% on MiniF2F"; Kimi K2 "1T MoE
   (32B active)"; R1 "7B close to GPT-4o"; Qwen3 "36T tokens"; DeepSeek-V3 "14.8T tokens".

## Sidebar additions

None (no new pages; all fixes are in-place).

## Commands run and results

- `npx prettier --write docs/appendix_c_resources/{learning-resources,gpu-hours-estimation,metrics-glossary,industrial-exercises}.md` — 3 files reformatted (table alignment), 1 unchanged.
- `npx vitepress build docs` — **success** (17.6 s; only pre-existing chunk-size/`jinja2` warnings, no dead links).
- Cross-reference targets verified with file globs (chapter/appendix slugs all exist); table
  arithmetic (Llama rows, V3 cost, cost-table products) recomputed by hand from the 6ND formula.
- Per worker rules, `git status/diff` (allowed) used to confirm only the 4 in-scope files changed;
  no code tests apply to this scope (no `code/` files touched, no test files named in task).

## Open issues

- The identical-playlist issue (#1 above) is the only suspected-wrong resource link; everything else
  flagged is "unverifiable offline", not "believed wrong".
- Coordinator decision needed: none for build; optionally confirm the "80+ entries" wording and the
  ByteDance-Seed attribution for DAPO if a second source is available.
