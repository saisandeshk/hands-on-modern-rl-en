# Report: ch24-text

## Summary

Text pass over Chapter 24 (audio, embodied/VLA, model-based RL, image and video generation). Fixed the confirmed 50-steps × 80-frames arithmetic error, removed the leftover editorial sentence about a previous LongCat description, corrected SFT/RLHF loss-objective signs, fixed stale chapter-number cross-references in two Connections tables, fixed a stale snippet path and a self-link in the MBRL supplement, and added the new `review.md` summary and exercises page. VitePress build passes.

Suspected issue from the task:

- [x] `chapter24_multimodal_frontier/video-generation-modern.md` — "50 steps × 80 frames ≈ four thousand forward passes" is wrong: each denoising step is one forward pass over all frames. Fixed in §24.5.2 and the Summary; the leftover editorial sentence ("the previous version of this page…") was removed.

## Technical fixes

1. `docs/chapter24_multimodal_frontier/video-generation-modern.md` (§24.5.2)
   - Wrong claim: one 5-second video "corresponds to about four thousand model forward computations".
   - Correction: each of the ~50 denoising steps updates all 80 frames in a single joint forward pass, so the cost is ~50 forward passes; $50 \times 80 = 4{,}000$ counts frame-level latent updates, not forward passes. Same correction applied to the chapter Summary ("four thousand steps" → "fifty denoising steps (roughly four thousand frame-level latent updates)").
   - Verified by: reasoning about DiT joint attention over video tokens (one pass processes all frames); arithmetic $50 \times 80 = 4{,}000$ recomputed.

2. `docs/chapter24_multimodal_frontier/video-generation-modern.md` (§24.5.5)
   - Wrong claim (editorial leftover): "The previous version of this page described LongCat as fixed five-second chunks, overlap averaging, a story-LLM reward, and two-level diffusion…".
   - Correction: deleted the sentence; kept the valid institutional distinction (LongCat-Video/Meituan vs. Seedance/ByteDance).
   - Verified by: inspection; it is a meta-commentary about page history, not textbook content.

3. `docs/chapter24_multimodal_frontier/video-generation-modern.md` (reward table)
   - Checked, no change: candidate totals $0.5875 \to 0.59$ and $0.60$ recomputed by hand and with Python; $1429.5/116.5 \approx 12.27 \to 12.3\times$ verified.

4. `docs/chapter24_multimodal_frontier/audio-reward-design.md` (MGRD SFT loss)
   - Wrong claim: $\mathcal{L}_{\mathrm{SFT}}^{(t)} = +\mathbb{E}[\log\pi] + \mathbb{E}[\log\pi]$ (a loss to be minimized cannot equal positive log-likelihood).
   - Correction: added the negative signs (negative log-likelihood).
   - Verified by: standard MLE loss convention, consistent with the chapter's own critic/MSE losses.

5. `docs/chapter24_multimodal_frontier/audio-reward-design.md` (mid-training loss)
   - Wrong claim: same sign error in $\mathcal{L}_{\mathrm{mid}}$.
   - Correction: negated both expectations.
   - Verified by: same convention check as above.

6. `docs/chapter24_multimodal_frontier/audio-reward-design.md` (RLHF objective)
   - Wrong claim: symbol $\mathcal{L}_{\mathrm{RLHF}}(\theta) = \mathbb{E}[\min(\dots)] - \beta\mathrm{KL}$ presented as a loss.
   - Correction: renamed to $\mathcal{J}_{\mathrm{RLHF}}(\theta)$ (objective to maximize), matching the surrounding text ("PPO-style objective"). The PPO clip and KL terms themselves are the standard maximizing form.
   - Verified by: comparison with the standard PPO objective and the page's own wording.

7. `docs/chapter24_multimodal_frontier/embodied-intelligence/index.md` (Connections table)
   - Wrong claims: PPO (Ch. 7), Actor-Critic (Ch. 6), VLM RL (Ch. 11), policy gradient (Ch. 5), RLHF (Ch. 8).
   - Correction: PPO Ch. 8, Actor-Critic Ch. 7, VLM RL Ch. 23, policy gradient Ch. 6, RLHF Ch. 13, per the current numbering table.
   - Verified by: the task's book-layout table.

8. `docs/chapter24_multimodal_frontier/embodied-intelligence/model-based-rl/index.md` (Connections table)
   - Wrong claim: "Policy gradients and actor-critic (Chapters 5-6)".
   - Correction: "Chapters 6-7".
   - Verified by: the numbering table (Ch. 6 policy gradient, Ch. 7 actor-critic).

9. `docs/chapter24_multimodal_frontier/embodied-intelligence/model-based-rl/index.md` (snippet command)
   - Wrong claim: `python docs/chapter28_vla/embodied-intelligence/...` (stale outline path).
   - Correction: `python docs/chapter24_multimodal_frontier/embodied-intelligence/model-based-rl/snippets/minimal_mbrl_point_mass.py`.
   - Verified by: `ls`/glob confirms the file exists at the corrected path; the old `chapter28_vla` directory does not exist.

10. `docs/chapter24_multimodal_frontier/embodied-intelligence/model-based-rl/index.md` (return link)
    - Wrong claim: "return to the main embodied intelligence article" linked to `./` (self).
    - Correction: link changed to `../`.
    - Verified by: directory structure (`model-based-rl/index.md` → parent `embodied-intelligence/`).

11. `docs/chapter24_multimodal_frontier/embodied-intelligence/model-based-rl/snippets/minimal_mbrl_point_mass.py` (numbers quoted in text)
    - Checked, no change: ran the snippet; output `one_step_model_mse=0.013716`, `random_policy_return=-2246.07, final_state=[8.50, -0.25]`, `mbrl_mpc_return=-18.76, final_state=[0.17, 0.07]` exactly matches the page.

## Other changes

- `embodied-intelligence/index.md`: "In the previous eight chapters" → "In earlier chapters" (the count did not match the current outline).
- New page `docs/chapter24_multimodal_frontier/review.md` (`24.R Summary and Exercises`): 8 learning outcomes, key equations with one-line readings, 6 pitfalls, 8 exercises (conceptual, derivations, hand computations, 2 coding) each with a worked `::: details Solution`.
- Verified hand computations in the new exercises with Python: GRPO advantages for $r=(1,2,3)$ → $(-1.2247, 0, +1.2247)$; video totals $0.5875/0.60$; token counts $75\times8=600$/s, $6{,}000$/10 s, $80/120$ frames; speedup $1429.5/116.5\approx12.27$.

## Coverage against the curriculum checklist

Checklist: (none).

| Item                          | Status                                                                                                                                                                                                                 |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (no checklist items provided) | n/a — judged coverage on my own; audio, embodied/VLA, MBRL, image-RL, and video sections each already have intuition, key equations, and worked examples, so no added core sections were needed beyond the review page |

## Needs external verification

- Step-Audio-R1/R1.5 numbers (MMAU 77.7 vs 76.5, reasoning lengths 2,800→1,500, Big Bench Audio 98.7, Realtime latency 0.92 s vs GPT/Gemini figures, eight-benchmark average 77.97 vs 72.50) and arXiv IDs (2511.15848, 2604.25719, 2507.16632, 2510.09592) are taken from the pages as paper-reported; no internet access, so I could not re-check them against the reports. They are hedged as paper-configuration values in the text.
- Codec historical details (SoundStream/EnCodec/SpeechTokenizer/WavTokenizer/Mimi years, rates, layer counts), VLA product claims (RT-2, Gemini Robotics), simulator claims, and Seedance/LongCat/Wan report details are likewise carried as cited-report claims, not independently verified.
- DanceGRPO reproduction GPU counts (8/16 H800 etc.) are cited to the official repo; not verified.

## Sidebar additions

- Title: `24.R Summary and Exercises`, link: `/chapter24_multimodal_frontier/review` (file `docs/chapter24_multimodal_frontier/review.md`; coordinator to renumber/title as needed).

## Commands run and results

- `timeout 120 .venv/bin/python docs/chapter24_multimodal_frontier/embodied-intelligence/model-based-rl/snippets/minimal_mbrl_point_mass.py` — ran successfully; outputs match the page exactly (see item 11 above).
- `npx prettier --write <all 7 files>` — success (2 files reformatted slightly, rest unchanged).
- `npx vitepress build docs` — success, build complete.

## Open issues

- None blocking. The MBRL supplement (`model-based-rl/index.md`) is still not listed in the sidebar; it is reachable only via the tip link from the VLA page — flagging in case the coordinator wants it added. No `code/` changes were made (another worker owns it).
