# Report: appB-text — Appendix B text pass

## Summary

Text and correctness pass over all eight Appendix B pages (`docs/appendix_b_algorithms/`). Checked every code block line by line (loss signs, masks, detach, reductions, shapes, KL estimators, advantage normalization) against the surrounding prose and math, and fixed the discrepancies found. Also fixed one front-matter/sidebar title mismatch and verified the examples numerically. `npx vitepress build docs` passes. No new pages were added, so no sidebar changes are needed.

## Technical fixes

1. `docs/appendix_b_algorithms/dpo-family.md` — front matter said `title: B.3 DPO Family` while the H1 and the sidebar (`docs/.vitepress/config.mjs` line 1434, `B.3 DPO Methods`) say "DPO Methods". Correction: changed front matter to `title: B.3 DPO Methods`. Verified by reading the sidebar config and the H1.
2. `docs/appendix_b_algorithms/grpo-rlvr.md` — PyTorch GRPO used `rewards.std(dim=1, keepdim=True)` (default unbiased/Bessel correction, divides by G−1) while the NumPy version uses `np.std` (divides by G). For small group sizes the two disagree. Correction: added `unbiased=False` with a comment. Verified by code reading and by running both normalizers (group `[1,2,3]` gives z-scores ±1.2247 with the fixed version, matching NumPy).
3. `docs/appendix_b_algorithms/grpo-rlvr.md` — "RLVR ... so there is no reward hacking risk" is overstated; verifiers can still be gamed by spurious solutions that pass the checks. Correction: softened to "greatly reduces reward hacking from a learned proxy — though gaming the verifier ... is still possible". Verified from general knowledge of RLVR failure modes; flagged below since no external source was checked.
4. `docs/appendix_b_algorithms/dapo.md` — total-loss sketch used `rewards.mean(dim=G)` / `rewards.std(dim=G)` where `G` is the group _size_ (e.g. 8), not a dimension index. Correction: changed to `dim=1` with `unbiased=False`, noted the `[B, G] → [B, T]` broadcast, guarded the masked division with `.clamp_min(1)`, and labeled the KL term optional. Verified by reading the sketch and by executing the masked aggregation pattern.
5. `docs/appendix_b_algorithms/dapo.md` — the unmasked `dapo_policy_loss` snippets (NumPy and PyTorch) silently give wrong gradients on padded batches. Correction: added notes that they assume no padding and pointed to the masked `token_level_loss()`; added `.clamp_min(1)` to the masked divisor. Verified by reading.
6. `docs/appendix_b_algorithms/dapo.md` — `dynamic_sampling_filter` used Bessel-corrected `std`, inconsistent with the z-score normalizer. Correction: `unbiased=False`, plus a `G > 1` note. Verified by execution (`[1,2,3]` kept, `[0.5,0.5,0.5]` filtered).
7. `docs/appendix_b_algorithms/ppo-gae.md` — value-loss snippet had no `import torch`, so it is not standalone like the other snippets. Correction: added the import and a detach note for `old_values`. Verified by reading.
8. `docs/appendix_b_algorithms/ppo-gae.md` — PyTorch GAE used `1 - dones[:, t]` (fails on bool tensors) and `torch.zeros(B)` (wrong device/dtype on GPU or float64). Correction: `1.0 - dones[:, t].float()` and `torch.zeros(B, device=..., dtype=...)`, plus a `dones` docstring note. Verified by executing with a bool `dones` tensor.
9. `docs/appendix_b_algorithms/softmax-ce.md` — `manual_cross_entropy` gathered with raw `targets` containing `ignore_index=-100`; negative ids wrap (or error when `C < 100`) before the mask is applied. Correction: `safe_targets = targets.clamp_min(0)` before `gather`. Verified by execution: masked result (0.41703) exactly equals `F.cross_entropy(..., ignore_index=-100)`.
10. `docs/appendix_b_algorithms/top-k-top-p.md` — PyTorch `top_p_filtering` used `logits.scatter(1, ...)`, which crashes on 1-D `[V]` inputs that the NumPy twin and the docstring pattern accept. Correction: `scatter(-1, ...)` with a docstring covering both shapes. Verified by executing both 1-D and 2-D inputs.
11. `docs/appendix_b_algorithms/top-k-top-p.md` — `generate_sample`/`top_k_sample`/`top_p_sample` divided by `temperature` unconditionally, so `T=0` divides by zero (the temperature-only sampler already had an argmax branch). Correction: added the same `T < 1e-8 → argmax(keepdim=True)` branch to all three and aligned `sample_with_temperature` to `keepdim=True` so shapes match `multinomial` (`[..., 1]`). Verified by reading.
12. `docs/appendix_b_algorithms/attention-mha.md` — NumPy snippet uses an additive mask (`0/-inf`, added to scores) while the PyTorch snippet uses a boolean mask (`1/0` with `masked_fill`), with no comment; a reader copying one convention into the other gets silent errors. Correction: added a paragraph and docstring note naming both conventions. Verified by reading both snippets.
13. `docs/appendix_b_algorithms/attention-mha.md` — comparison table said GQA is used by "LLaMA 2/3"; small LLaMA 2 models use MHA. Correction: "LLaMA 2 (larger variants), LLaMA 3, Mistral". Listed under external verification since it was fixed from general knowledge.
14. `docs/appendix_b_algorithms/dpo-family.md` — DPO snippet gave no detach guidance though the prose requires detaching the reference log-probs. Correction: docstring now says `ref_*` must be computed under `torch.no_grad()`. Verified by reading.
15. `docs/appendix_b_algorithms/dpo-family.md` — KTO prose did not say which formulation the `-log_sigmoid` code follows (the original paper writes a $\lambda-\sigma$ value form; the code is the TRL-style logistic implementation). Correction: added one sentence stating the snippet is the TRL-style logistic form with the same optimum direction, and clarified `z_ref` is a detached scalar. No formula or code changed.
16. `docs/appendix_b_algorithms/dpo-family.md` — SimPO divides by response length with no guard. Correction: comment that each length must be ≥ 1. Verified by reading.
17. `docs/appendix_b_algorithms/sft-kl.md` — `kl_penalty` averaged over `[B, seq_len]` with no padding or `no_grad` guidance. Correction: docstring notes the reference log-probs go under `torch.no_grad()` and padding must be excluded beforehand. Verified by reading.

Non-issues checked and left alone: DPO/IPO formulas and signs, KTO masking math (`cumsum - prob > p` keeps ≥ 1 token), SimPO margin subtraction, GAE backward recursion and `(1-done)` masking, PPO clip/`min`/value-`max`/entropy-sign conventions, k1/k3 KL direction and nonnegativity argument, DAPO clip-higher bounds (0.2/0.28) and the inactive-bound note, DAPO soft-overlong linear schedule and cap, top-k tie behavior (`>=` threshold keeps ties in both NumPy and PyTorch versions), MHA/GQA `view→transpose`/`repeat_interleave` patterns, and the attention complexity table.

## Other changes

- Prettier formatting applied to the eight touched files (only `attention-mha.md` had formatting drift; the rest were unchanged by prettier).
- No structural rewrites, no new pages, no config edits.

## Needs external verification

- DAPO paper details (from solid secondary knowledge, not checked against the paper): `eps_low=0.2/eps_high=0.28` defaults, the soft-overlong linear schedule shape, and whether the reference DAPO objective keeps the GRPO-style KL term (the sketch now labels KL "optional", which is safe either way).
- KTO original-paper vs TRL-implementation loss forms (the added sentence describes the page's code as the TRL-style logistic form; confirm wording against TRL docs if strict attribution is needed).
- LLaMA 2 GQA coverage by model size (fixed from general knowledge: larger variants use GQA).
- RLVR verifier-gaming claim (softened from general knowledge).

## Sidebar additions

None (no new pages; existing B.1–B.8 links and titles already match after the B.3 front-matter fix).

## Commands run and results

- `ls /home/saisandeshk/Study/RL/wt/appB-text/.venv/bin/python` — venv present.
- Inline `.venv/bin/python -c` checks (all passed): GRPO z-scores/row-means/filter behavior; bool-`dones` cast; `clamp_min(0)` cross-entropy equality with `F.cross_entropy(..., ignore_index=-100)`; `scatter(-1, ...)` for 1-D and 2-D top-p; DAPO masked token aggregation; DPO/KTO/SimPO smoke values.
- `npx prettier --write` on the eight files — clean.
- `npx vitepress build docs` — success (client + server bundles, all pages rendered).
- `git status --short` — only the eight in-scope files modified (plus this report).

## Open issues

- None blocking. If the coordinator wants, a follow-up could add explicit `attention_mask` plumbing to the GRPO/DAPO PyTorch snippets (currently sequence-level means), but that would expand the snippets beyond one-page cheatsheet scope, so I left it as prose guidance in `sft-kl.md`.

## Review response

Reviewer report: `planning/reviews/appB-text.md` (5 errors, 6 doubts). Re-checked each claim against the page code and, where possible, by execution; fixed in the in-scope files.

- Error 1 (DAPO overlong paragraph contradicts its own code): accepted. Recomputed from `soft_overlong_penalty` (`L=1500 → 0.0`, `L=1800 → −0.6`, `L=2000 → −1.0` with `max_len=2000, buffer_len=500`): the deduction ramps inside the buffer zone, not above it. Rewrote the paragraph in `dapo.md` (unpenalized only up to `max_len − buffer_len`; linear ramp to `−penalty_factor` at `max_len`, flat after).
- Error 2 (DAPO sketch `[B, G] → [B, T]` broadcast): accepted. `[B, G]` cannot broadcast against `[B, T]`; rewrote the sketch in `dapo.md` with explicit `[B, G, T]` log-probs and `advantages.unsqueeze(-1)` (`[B, G, 1]`, constant over tokens), plus a `mask: [B, G, T]` shape note.
- Error 3 (implicit reward "divided by β"): accepted. The DPO implicit reward is $r = \beta \log(\pi_\theta/\pi_{ref})$ (cf. Chapter 14 `dpo-objective-derivation.md` Step 3); fixed `dpo-family.md` to say the log-ratio times $\beta$ is the implicit reward, including the pseudocode comments.
- Error 4 (KTO "same optimum" for logistic vs value form): accepted. The gradients differ ($-\sigma(-z)$ vs $-\sigma(z)(1-\sigma(z))$) and the value form saturates at $\lambda_U$ while the logistic version keeps a logarithmic penalty; rewrote the KTO paragraph in `dpo-family.md` as a logistic-style implementation with that saturation difference, sharing only the push direction.
- Error 5 (top-k "keep exactly k"): accepted. Both implementations keep ties (`>= threshold` / `masked_fill(logits < threshold, …)`), so a saturated logit vector keeps more than k; fixed the comparison table, one-line memory, and threshold bullet in `top-k-top-p.md`.
- Doubt 1 (DAPO KL term): accepted in the soft sense. Without the paper to hand I did not assert a new fact beyond the reviewer's verl-based recollection plus the in-repo "Zero KL for Math Tasks" trend (`grpo-family.md`); relabeled sketch step 5 in `dapo.md` as "dropped in DAPO; shown here only for comparison with GRPO" and changed the final line to `loss = policy_loss`. Flagged under Needs external verification.
- Doubt 2 (TRL KTO attribution): accepted in the soft sense. No TRL source is vendored, so removed the unverifiable "TRL-style" attribution in `dpo-family.md` (now "a logistic-style implementation"); the `z_ref`-includes-β code convention note is kept as-is since only the loss-form attribution was disputed.
- Doubt 3 (population vs Bessel std): accepted. Added the missing convention note to the pitfalls table in `grpo-rlvr.md` (page uses population std to match NumPy; some production recipes use the Bessel-corrected default, rescaling by $\sqrt{G/(G-1)}$).
- Doubt 4 (dynamic-sampling mismatch with Chapter 15): accepted within scope. Chapter 15 (`deepseek-dapo.md` lines 105–129) is out of scope so left untouched; added a pointer in the `dapo.md` sketch to `[§15.2](/chapter15_grpo/deepseek-dapo)` noting it illustrates the same idea with an accuracy-threshold sketch while the operative DAPO filter is zero within-group variance.
- Doubt 5 (SimPO `gamma` "learnable-style"): accepted. SimPO's $\gamma$ is a fixed hyperparameter; changed `dpo-family.md` to "a fixed hyperparameter (not trained)".
- Doubt 6 (acronym capitalization): accepted. Normalized `dapo.md` to "Decoupled Clip and Dynamic Sampling Policy Optimization" to match Chapter 15.

Verification: `npx prettier --write` on the four changed files (clean); `npx vitepress build docs` succeeds; overlong schedule spot-checked (L=1800 → −0.6). No sidebar changes.
