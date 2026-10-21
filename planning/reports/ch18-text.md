# Report: ch18-text — Chapter 18 text pass

## Summary

Worked through all five Chapter 18 pages in full, plus created the new summary page. Changes are deliberately tight:

- **Technical fixes (3):** corrected the z-score normalization range claim (18.1), corrected the 70B memory-accounting table baseline and added the missing GPU-count assumption (18.4, with matching summary line), and softened the ZeRO-1/2 saving factors to large-$N$ limits (18.4). All verified arithmetically (details below).
- **Coverage additions (4, all in-scope systems items):** throughput/compute-accounting subsection in 18.1 (§3.3); colocated-vs-disaggregated, queue-health (batching/backpressure/starvation/recovery/dedup), and multi-epoch reuse-drift treatments in 18.4; loss-mask paragraph (truncation, tool tokens, reward placement) in 18.5.
- **New page** `docs/chapter18_llm_rl_industrial/review.md` (18.R Summary and Exercises): 7 outcomes, 7 key equations, 6 pitfalls, 8 exercises with fully worked, numerically verified solutions.
- **Conventions:** added missing `title` front matter to all five pages (they had bare H1s; the rest of the book uses front matter).
- Cross-references checked: all 18.x numbers and ch16/ch19 links resolve to existing pages.
- Known suspected issues: (none recorded) — nothing to mark `[x]`.
- Build: `npx vitepress build docs` succeeds; `npx prettier --write` applied to all touched files.

## Technical fixes

1. **File:** `docs/chapter18_llm_rl_industrial/single-machine-to-industrial.md` (§2.4 Reward Scaling Alignment). **Wrong claim:** "After normalization, all rewards are in the scale of $[-3, 3]$, making them safe to add together." **Correction:** z-score normalization does not bound values; outliers can exceed any fixed range. Reworded to "typically concentrated on a comparable scale (roughly $[-3,3]$ for most samples)" plus a note that extreme rewards still need clipping. **Verified by:** definition of $\tilde r=(r-\mu)/\sigma$ (unbounded for heavy-tailed rewards).
2. **File:** `docs/chapter18_llm_rl_industrial/distributed-sync.md` (§Distributing Memory Across GPUs). **Wrong claim:** "Full parameters + Adam, no sharding | ~940 GB" with per-GPU rows and no GPU-count assumption ("ZeRO-3 … ~118 GB … Still no", "ZeRO-3 + checkpointing … ~30 GB … Yes"). **Correction:** the chapter's own 16-bytes/param accounting gives $70{\times}10^9 \times 16 = 1.12$ TB, not 940 GB; and gradient checkpointing only shrinks activations, so it cannot take a sharded-state-OOM configuration to "Yes" without stating $N$. Rewrote the block around 32× 80GB H100s ($1.12\text{TB}/32 \approx 35$ GB sharded state per GPU + activations), and updated the section Summary bullet consistently ("1.12TB … about 35GB plus activations"). **Verified by:** $70{\times}10^9 \times 16\,\text{B} = 1.12{\times}10^{12}$ B; $1120/32 = 35$; checkpointing reduces the $O(L)\to O(\sqrt L)$ activation term only.
3. **File:** same file (ZeRO table). **Wrong claim:** ZeRO-1 "~4×", ZeRO-2 "~8×" per-GPU saving stated unconditionally. **Correction:** "up to ~4× / ~8× (large $N$)", since e.g. ZeRO-1 on 8 GPUs gives $(4 + 12/8)^{-1}\times16 \approx 2.9\times$. **Verified by:** ZeRO sharding arithmetic (optimizer 12 B/param sharded, weights+grads 4 B/param replicated at ZeRO-1).
4. **Numerics re-checked and left unchanged (all correct):** GPU-hours example ($6\cdot7\mathrm{e}9\cdot1\mathrm{e}9/(989\mathrm{e}12\cdot0.4\cdot3600)\approx29.5$ h); KL/entropy two-action examples (0.19/0.50 → 0.49/0.20 — recomputed: $0.8\ln1.6+0.2\ln0.4=0.1927$; $0.95\ln1.9+0.05\ln0.1=0.4947$); stale-ratio example ($e^{-0.19}\approx0.827$, ~17% deviation); GLM-5.2 waste arithmetic ($8{\times}870+8{\times}780+12{\times}600=20{,}400$ GPU-s).

## Other changes

- Front matter `title: 18.x …` added to all five in-scope pages (18.5 uses its H1 title "18.5 Large-Scale RL Data Engineering", matching the file; the sidebar wording differs slightly — see Open issues).
- 18.1: new §3.3 "Measuring Throughput" (rollout tok/s, verifier throughput, update utilization; $C_{\text{total}}=C_{\text{sample}}+C_{\text{inference}}+C_{\text{training}}$ with a worked saving argument); old §3.3 renumbered to §3.4 (no dangling references — grepped).
- 18.4: new "Colocated versus Disaggregated Deployment" subsection; new "Keeping the Queue Healthy: Batching, Backpressure, and Recovery" subsection (length bucketing, bounded queues + oldest-version metric, ready-buffers against learner starvation, four-state checkpointing, retry dedup by (prompt, version, attempt) ID, failed-rollout budgets); short paragraph on multi-epoch reuse drift (why GRPO takes one step per batch).
- 18.5: new paragraph on loss masks (truncated completions, tool/environment returns, reward placement; store mask with tokens) with pointer to the 18.2 TIR recipe.
- No edits to `code/`, images, components, sidebar config, or neighboring chapters.

## Coverage table (scope = systems for large-scale LLM RL)

Legend: ✅ covered in ch18 · ➕ covered by this pass's additions · 🟡 partial · ➖ out of ch18 scope (belongs to earlier/later chapters) · ❌ missing in ch18 (only fixed where in-scope and core).

**RL-11 (Online Policy Optimization):** on-policy sampling/generation ✅ · sequence PG for autoregressive trajectories ➖ (Ch. 6/15) · sparse terminal reward + per-token loss assignment 🟡 (overlong shaping + new mask paragraph) · REINFORCE in token space ➖ (Ch. 6/15) · critic + PPO-style RLHF ✅ · LOO/RLOO ➖ (Ch. 15) · grouped sampling + group-relative advantages ✅ · GRPO objective + critic-free trade-offs ✅ · all-equal/all-correct/all-wrong groups ✅ (+ review Ex. 5) · within-prompt vs batch normalization ✅ · reward scaling/whitening/cross-prompt comparability ✅ (fix #1) · old vs current token log-probs ✅ · per-token ratios + sequence dependence ✅ · clipping + token/sequence/group aggregation ✅ (+ review Ex. 7) · reference-policy KL penalties ✅ · reward placement/masks/truncation 🟡➕ (new 18.5 paragraph; loss-aggregation theory stays in Ch. 15) · length normalization/verbosity/boundaries ✅ · multiple gradient steps + off-policy drift ➕ (new 18.4 paragraph) · freshness/lag/importance correction ✅ · critic memory vs group-sampling cost 🟡 (critic-removal memory win stated; no head-to-head variance/cost numbers) · entropy collapse/KL spikes/mode contraction ✅ · invalid output/verifier failure/zero-signal groups ✅ · prompt balancing/difficulty curricula/sparsity ✅ · fixed-rollout estimator/ratio/clipping/gradient tests 🟡 (review Ex. 7–8 are the test harness; no separate section) · SFT/DPO/RS/best-of-N controls ✅ · matched tokens/verifier calls/updates/wall time ➕ (new 18.1 §3.3) · reward/quality/KL/entropy/length/validity traces ✅ · DAPO/GSPO variants (Depth) ✅ · frontier convergence claims (Later) ➖.

**RL-12 (Reasoning, Agents, Scalable Training):** RLVR ✅ · math/code/structured-output envs ✅ · verifier isolation + hidden tests ✅ · outcome rewards/sparse credit ✅ · process rewards/step-level credit 🟡 (turn-weighting + intermediate-reward mentions; full PRM theory is Ch. 17) · reasoning traces/hidden reasoning ➖ (Ch. 16) · inference-time search/aggregation ➖ (Ch. 17) · train- vs test-time compute ➖ (Ch. 16) · rejection fine-tuning + rollout distillation ✅ · on-policy distillation vocabulary ✅ · length/budget/invalid traces ✅ · multi-turn state/observations/tool actions ✅ · env transitions/tool results/terminal conditions ✅ · trajectory rewards/delayed credit ✅ · sandbox/reset/replay/side-effects ✅ · tool-error/timeout/retry/duplicate semantics 🟡➕ (retry dedup + failure labels added; duplicate-action suppression policies not detailed) · actor/rollout/reward/learner roles ✅ · sync vs async loops ✅ · broadcast/version/staleness ✅ · colocated vs disaggregated 🟡➕ (new subsection) · batching/backpressure/starvation ❌➕ (new subsection) · checkpoint/rollout/optimizer/sampler recovery ❌➕ (new subsection) · retry dedup/failed-rollout accounting ❌➕ (new subsection + Ex. 8) · curriculum/filtering/evolving distributions ✅ · tok/s + verifier throughput + update utilization 🟡➕ (new 18.1 §3.3) · sample/inference/training/total accounting 🟡➕ (same) · independent capability/safety/agent eval ✅ · hacking/spec-gaming/held-out shifts ✅ · clean reproduction + prespecified extensions 🟡 (minimal-reproduction recipes + versioning; no formal preregistration template) · self-play/auto-curriculum (Depth) ➖ (Ch. 19/26) · multimodal/embodied/production agents (Later) ➖.

## Needs external verification

No internet access; paper/report-specific claims below were **not** altered (beyond noting them) and should be checked by someone with sources. Most carry 2025–2026 dates past my knowledge cutoff, including future-dated arXiv IDs:

- 18.1: AReaL/LlamaRL arXiv IDs (2505.24298, 2505.24034), AgentRL (2510.04206), ROLL, Seed-Thinking criteria attribution, MiniCPM-5/Kimi-K3/Qwen-AgentWorld seed counts (30K prompts, 5K→30K tasks), GenRM arXiv 2410.12832, DAPO arXiv 2503.14476, LlamaFactory arXiv 2403.13372.
- 18.2 (largest exposure — dozens of vendor claims): MiniCPM5/Kimi-K3/Qwen-AgentWorld/GLM-5.2 figures (e.g. GLM-5.2 SAO arXiv 2607.07508; MiniCPM5 arXiv 2601.04962; Kimi K3 2504.12593; Qwen-AgentWorld 2506.07340; SLIME 2602.02779; WebExplorer 40K QA/5 evolution rounds; Qwen3 3,995 query-verifier pairs; DeepSeekMath 144K problems / 64 outputs; Hunyuan-T1 "96.7% of post-training compute"; Kimi K2 "1 loss spike per 10T tokens, +15% speed"; Skywork-OR1 accuracy deltas 57.8→72.8 / 43.6→57.5; MiMo 130K problems; all GPT-5.x/Claude/Gemini system-card characterizations).
- 18.3: GLM-4.5 (355B total / 32B active, July 2025), Llama 4 variants (Scout 109B/17B/10M ctx, Maverick 400B/17B, Behemoth 2T/288B) and the LMArena-version account, Kimi K2 MuonClip/QK-clip effects, VAPO hyperparameter values.
- 18.4: LlamaRL "10.4×", AReaL "2.77× for 671B MoE" + clip range [0.8, 1.2], AgentRL publication details, DeepSeek-V3 utilization 70%→95% via packing, GSPO sequence-level-ratio claim, MFU bands (35–45% / 50–60% / 40–50% & 70–80%).
- 18.5: MiniCPM5/Kimi-K3/Qwen-AgentWorld data figures, SimpleRL-Zoo / SLIME / DeepSeek-V3 references.

## Sidebar additions

- Title: `18.R Summary and Exercises`, link: `/chapter18_llm_rl_industrial/review` (one entry; coordinator to place after 18.5 and renumber as needed).

## Commands run and results

- `npx prettier --write` on all six files — clean (two files reformatted, no errors).
- `npx vitepress build docs` — **success** (client + SSR bundles built, all pages rendered, no dead-link failures).
- `timeout 60 .venv/bin/python -c …` — verified review-exercise numbers: 59 GPU-h; $e^{-0.19}\approx0.827$; GRPO group $\mu=0.375,\sigma\approx0.484$, advantages $+1.291/-0.775$; 13B state 208 GB → 26 GB/GPU over 8 with ZeRO-3.
- `git diff --stat` — 5 modified files, +71/−15 (review.md is new/untracked).
- Note: chained/piped shell forms (e.g. `git log --oneline -5; … | head`) are rejected in this sandbox; used single allowed commands only. No `code/` tests apply (no code touched; prose-only task).

## Open issues

1. **Coordinator decision:** add the review page to the sidebar; also note the 18.5 title mismatch (sidebar: "Data Engineering for Large-Scale RL" vs file H1/front matter: "18.5 Large-Scale RL Data Engineering") — I matched the file's own H1; rename either side if uniformity is wanted.
2. `industrial-post-training.md` is ~1300 lines and reads as a vendor-case library rather than a narrative section; any split/reorganization is left to the coordinator (out of scope for a text pass).
3. Residual 🟡 items above (critic-vs-group cost numbers, formal preregistration template, duplicate-action suppression) are deliberately left: they need either external numbers or chapters that own them (15/17/19).

## Review response

1. **Accepted.** §3.3 saving claim was a tie under its own numbers ($100+90\times3=370$ vs $100+300\times0.9=370$, both 7.5%). Rewrote as: same ~7.5% step saving, but ~39% more useful signal per compute ($90/370\approx0.243$ vs $70/400=0.175$). Verified with `.venv/bin/python`.
2. **Accepted.** Three-bucket equation double-counted discarded rollouts and omitted the reference forward. Redefined: $C_{\text{sample}}$ = generation of thrown-away rollouts; $C_{\text{inference}}$ = retained-rollout generation + judge scoring; $C_{\text{training}}$ = actor forward-backward + reference forward; noted $C_{\text{total}}=C_{\text{RL-step}}+C_{\text{sample}}$.
3. **Accepted.** ZeRO-3 row claimed "Yes, tight" while the page's own ~100 GB activation figure exceeds 80 GB H100 memory. Changed to "~35 GB + activations (≈100 GB at the batch size above) | No without gradient checkpointing" and added "+ activations" to the LoRA row.
4. **Accepted.** Colocation paragraph wrongly claimed nothing crosses the network and every GPU works in each phase. Rewrote: no separate pool or inter-pool broadcast, but rollout and update phases cannot overlap (step = rollout + update time plus switch cost); rollouts/returns/rewards still travel between processes.
5. **Accepted.** Rollout ID (prompt + version + attempt) was self-contradictory and inconsistent with Exercise 8. Changed the dedup key to (prompt ID, policy version) with attempt as a separate field in 18.4, and updated Exercise 8's prompt/code/trace to match (dedup by `(id, version)`; reran the snippet — batch `[a,d,e]`, failed `[c]`, stale `1`).
6. **Accepted.** Fixed the pointer to name the "Locating Performance Bottlenecks" table's "Stragglers" row, and noted continuous batching refills slots but batch completion still waits for the longest sequence.
7. **Rejected (task/scope constraint).** The task explicitly requires `title: 18.R Summary and Exercises` / `# 18.R Summary and Exercises` with the coordinator to renumber at sidebar-insertion time, and forbids editing `docs/.vitepress/config.mjs`. Renaming to 18.6 and touching the sidebar would violate both rules, so no change was made; sidebar registration remains a coordinator decision (see Open issues).
8. **Accepted.** 200-step run failing at step 312 was contradictory (also quoted as "the first 200 steps"). Changed both to 400 steps, keeping step 312.
9. **Accepted.** Exercise 1 solution sent new weights to "(and reference) GPUs" — the frozen reference needs no sync. Fixed to broadcast to rollout GPUs only, noting the reference needs no sync.
10. **Accepted.** Exercise 7(b) stopped at the second term without the final loss. Added: terms $[1.0,-0.75]$, mean $0.125$, loss $-0.125$ (matches the reference implementation).

Doubts: (1) MuonClip 15% / loss-spike figures — left in text, already listed under Needs external verification (no source access). (2) GLM-4.6 October vs late-September 2025 — left unchanged, added to Needs external verification below. (3) LlamaRL/AReaL speedup and clip-range numbers — left unchanged, already listed under Needs external verification. (4) 18.2/18.5 vendor parameters and arXiv IDs — left unchanged, already listed. (5) **Accepted in part.** §3.2 "50%–60% of total computation" now reads "50%–60% of wall-clock time" with a note that the raw-FLOP share is smaller (autoregressive decoding runs at far lower MFU). (6) **Accepted.** Mask sentence reworded: rescoring changes rewards, not masked positions; the stored mask keeps the two consistent. (7) **Accepted.** Generation-batching bullet now states why bucketing still helps alongside continuous batching. (8) Noted in the Exercise 8 solution: entries beyond `max_size` are left untaken and a production monitor should count them as capacity-dropped; return signature unchanged to match the exercise spec. (9) `planning/content-review-notes.md` is outside this task's editable scope, so the three factual fixes (z-score bound, 70B 1.12 TB baseline, ZeRO large-$N$ limits) plus the new factual corrections in this round are recorded here and in Technical fixes instead; coordinator to merge into content-review-notes.

Needs external verification (added this round): GLM-4.6 release month (October 2025 vs late September 2025).
