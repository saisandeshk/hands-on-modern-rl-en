# Content review notes

Suspected technical errors in the course text, collected while re-translating the English pages from the
latest Chinese originals (October 2026). The translations keep the source wording; nothing below has been fixed yet.
Paths use the current English-only layout. When you fix an item, mark it `[x]` and note the fix.

Status legend: **confirmed** = checked by arithmetic or against the cited paper during translation review;
**suspected** = flagged by a translator, needs checking against the source.

## Part II · Deep RL

- [ ] `chapter08_ppo/ppo-math.md` — the "implementation structure overview" block (six questions + `<PpoCodeFocus focus="overview"/>`) appears twice almost verbatim (under "PPO training framework" and under its own heading). **confirmed (duplication)**
- [ ] `chapter08_ppo/ppo-bipedal-walker.md` — the random-baseline code prints three lines, but the sample output shows four (an extra standard-deviation line). **suspected**
- [ ] `chapter08_ppo/rl-long-horizon-planning.md` — "DAC (Deep Adversarial Control)" should be Discriminator-Actor-Critic. **confirmed**
- [ ] `chapter08_ppo/rl-long-horizon-planning.md` — DreamerV3 is dated "Nature 2023"; the Nature paper appeared in 2025 (arXiv 2023). **confirmed**
- [ ] `chapter08_ppo/rl-long-horizon-planning.md` — DAgger regret "O(1/T) vs O(T)" is stated loosely; under "stage three" a three-item list has no lead-in. **suspected**
- [ ] `chapter09_continuous_control/model-based.md` — compounding-error sums: for L=1.1, T=50 the sum is ≈1164ε (text: 1174ε); for L=1.01, T=100 it is ≈170ε (text: 270ε). The MBPO reset condition `rollout_step % K_short == 0` looks wrong. **sums confirmed by arithmetic ((L^T−1)/(L−1)); reset condition suspected**
- [ ] `chapter09_continuous_control/td3-sac.md` — text claims HalfCheetah reaches 15000+, but the ASCII curve tops out near 12000. **suspected**

## Part III · Advanced RL

- [ ] `chapter10_offline_rl/offline-data-distribution-shift.md` — the CQL code snippet uses undefined variables (`logsumexp_q2`, `q_curr2`) and has a concat/dim mismatch; CQL is described as a pointwise lower bound for all (s, a), but the guarantee holds in expectation under the policy. **suspected**
- [ ] `chapter10_offline_rl/sequence-modeling.md` — RvS is called online RL; it is an offline (supervised) method. **suspected**
- [ ] `chapter11_imitation_meta_rl/irl-gail.md` — Wasserstein GAIL "gradient clipping" wording is garbled; the optimal-transport box heading contradicts its body (which talks about Jensen-Shannon divergence). **suspected**
- [ ] `chapter11_imitation_meta_rl/meta-rl.md` — §3.2 says Algorithm Distillation in-context RL and GPT-3 in-context learning are "almost the same thing", then hedges; reads as contradictory. **suspected**
- [ ] `chapter12_exploration_marl_hierarchical/intrinsic-motivation-exploration.md` — the 10⁻¹⁸ probability (ε=0.1, 6 actions, 50 steps) cannot be derived from the stated numbers; ICM on Mario is described as continuous control, but Mario has discrete actions. **suspected**
- [ ] `chapter12_exploration_marl_hierarchical/hierarchical.md` — HIRO's off-policy correction is described as g' = s\_{t+c} − s_t; HIRO actually picks g' (among candidates) to maximize the low-level policy's action log-likelihood. Latent actions are credited to Genie 3; they come from Genie (2024). **suspected (likely correct critique)**

## Part IV · LLM post-training

- [ ] `chapter16_reasoning/emergence-and-o1.md` — AIME table: "GPT-4o consensus@64 ≈ 56%" looks like o1-preview's single-sample score (56.7%). **suspected**
- [ ] `chapter16_reasoning/test-time-scaling.md` — the text attributes the 0.6% gain to the 16th candidate, but the table value is the gain from adding the 17th. **suspected**
- [ ] `chapter17_prm_search/formal-prm.md` — Pass@8192: (1 − 0.0003)^8192 ≈ 0.086, so 1 − (1 − 0.0003)^8192 ≈ 0.914; the stated 0.111 matches neither (a per-sample rate ≈ 0.027% gives 0.89). **confirmed (arithmetic)**
- [ ] `chapter17_prm_search/generative-prm.md` — the worked error-explanation example mixes "4k² = 2q²", "divide by 2" and "4k = q²" inconsistently. **suspected**
- [ ] `chapter15_grpo/grpo-family.md` line ~32 — "Modified Hamiltonian Monte Carlo (mHC) sampler" looks invented; check against the cited paper. **suspected**

## Part VI · Multimodal

- [ ] `chapter24_multimodal_frontier/video-generation-modern.md` — "50 steps × 80 frames ≈ four thousand forward passes" is wrong: each denoising step is one forward pass over all frames. The page also contains a leftover editorial sentence ("the previous version of this page…"). **suspected (arithmetic confirmed)**

## Part VII · Safety and evaluation

- [ ] `chapter25_alignment_failures/rl-evaluation.md` §25.5.7 — system-card numbers (34×, 20.96, 65.83%, 0.612) need checking against the cited source. **suspected**

## Cross-references and numbering

- [ ] Several pages cite chapter numbers from older outlines, e.g. `preface/introduction.md` ("Chapters 8–9" for DPO, "Chapter 8" for Agentic RL), `chapter12_exploration_marl_hierarchical/marl.md` ("PPO since Chapter 7"), `chapter08_ppo/trust-region-clipping.md` ("Chapter 6" for REINFORCE is correct), `chapter08_ppo/gae-reward-model.md` (Chapters 4, 7.1, 14, 15). Check each against the sidebar numbering.
