# Report: appA-text — Appendix A text pass

## Summary

Text pass over all seven in-scope Appendix A files. Added the missing empirical-method core (seeds as unit of uncertainty, CIs, IQM + stratified bootstrap per Agarwal et al. 2021, matched compute/step budgets, tuning-vs-final separation, deterministic vs stochastic eval policies) split across A.1 (checklist + pointer) and A.4 (eval-side mechanics + report template), with a new `[^rliable]` citation in both. Strengthened core diagnostics with formulas (approx-KL estimators, clip-fraction definition, explained variance, TD-error/Q-scale tracking, grad-norm pre/post-clip logging). Fixed one wrong-direction advice bug in each of two legacy-table cells, one memory-arithmetic error, one wrong script output path, and softened unverifiable hardware-specific numbers in three legacy/system pages. `vitepress build` passes; prettier applied.

## Technical fixes

1. `docs/appendix_a_engineering/monitoring.md` (legacy stub, quick-lookup table): the old fix for "Reward crashes / policy collapse" said "increase the clip epsilon" — widening epsilon makes PPO updates _more_ aggressive, the opposite of the correct response. Corrected to "reduce the clip epsilon or PPO epochs". The old fix for "KL spikes" said "Decrease batch size" — smaller batches increase gradient noise and worsen KL spikes. Corrected to "Increase batch size or increase the KL penalty coefficient". Verified by PPO trust-region reasoning (clip bounds $[1-\epsilon,1+\epsilon]$; KL/stepsize monotonicities) consistent with the corrected guidance already in A.1.
2. Same file: self-link text "See Appendix A …" pointing at `training-debugging` corrected to "See [A.1](./training-debugging)…". Also converted to relative link to match book convention.
3. `docs/appendix_a_engineering/parallelism.md` (legacy stub): "Four of them is roughly 56 GB" for four 7B FP16 models understated reality — one 7B training replica alone needs ~14 GB params + ~14 GB grads + ~56 GB Adam FP32 moments ≈ 84 GB before activations. Rewrote paragraph with correct arithmetic; fixed the DP-limitation sentence the same way. Verified by parameter-count arithmetic (7e9 × 2 bytes = 14 GB; Adam m+v at 4 bytes each = 56 GB).
4. Same file: replaced fabricated-precision savings "~25% / ~50% / ~40%" with qualitative statements, and qualified "PagedAttention … 2-4×" as workload-dependent. No source in repo supports those exact figures.
5. `docs/appendix_a_engineering/async-training.md` (legacy stub): replaced the hardware-specific GRPO timing table (3/11/45 min, 14/56 min/3.7 h) and the "GPUs idle 99%" blanket claim with a structural statement (autoregressive generation vs parallel training) plus guidance to measure on own hardware. Removed exact transfer timings (100–500 ms, ~20 ms, <1 ms) from the weight-sync table, keeping data volumes; kept "~50 MB" adapter size (derivable from LoRA rank math, order-of-magnitude). Softened "20 ms" sync example, "LoRA sync under 1 ms", and "best practice" overclaim. Updated the footnote that claimed "core numbers on this page are derived from" the survey.
6. `docs/appendix_a_engineering/evaluation-badcase.md`: fixed radar-script bug `OUT = Path("docs/appendix_industrial_training/images")` — that directory does not exist; corrected to `docs/appendix_a_engineering/images` where the referenced PNGs actually live. Verified: `radar-llm-core-regression.png` and `radar-code-agent-tool-bench.png` exist in that dir and are referenced by the page.
7. `docs/appendix_a_engineering/rl-infrastructure.md`: qualified the MuJoCo×64 ~10K fps vs Isaac Gym×4096 ~1M fps comparison as illustrative/workload-dependent. The Sample Factory "100000 FPS" figure was left as-is (matches the cited paper's own title/claim).
8. `docs/appendix_a_engineering/agentic-rl-infra.md`: "DCS … taking no extra time" corrected to "to hide most of the transfer cost" (overlap hides cost; it cannot be literally zero). "2.77x speedup" (AReaL) and "+74–97%" (Seer) now explicitly marked as reported figures.
9. `docs/appendix_a_engineering/training-debugging.md`: expanded "Watch Three Metrics" with approx-KL estimators ($r$ log-ratio; Schulman $k_3 = e^{-r} - 1 + r$), clip-fraction definition ($\pi_{\text{new}}/\pi_{\text{old}}$ hitting $[1-\epsilon,1+\epsilon]$, sustained >0.3–0.4 = too aggressive), entropy; added explained-variance formula ($1 - \mathrm{Var}(y-\hat V)/\mathrm{Var}(y)$) and TD-error/Q-scale tracking ($\delta = r + \gamma V(s') - V(s)$, done-mask caution); grad-norm guidance now says log pre- and post-clip norms with typical 0.5–1.0 range. Verified against standard PPO/actor-critic definitions and C.3 glossary field names.

## Other changes

- New section "Report Results with Uncertainty, Not Single Seeds" (`training-debugging.md`, before Summary): five habits — multiple seeds + CIs, IQM + stratified bootstrap, matched budgets, tuning/final separation, deterministic-vs-stochastic eval — with cross-links to A.4 and C.3. New `[^rliable]` footnote (Agarwal et al., NeurIPS 2021).
- New subsection "Comparing Checkpoints Honestly: Seeds, Budgets, and Uncertainty" (`evaluation-badcase.md`, Step Three after "Number of Samples"): eval-side mechanics + minimal honest-report template. New `[^rliable]` footnote there too.
- "multi-seed results" checklist bullet in A.1 now points at the new section and A.4.
- No pages added, removed, or renamed; chapter/section numbering untouched.

## Needs external verification

- `evaluation-badcase.md` cites future-dated 2026 sources I could not verify offline: Claw-Eval-Live (`arXiv:2604.28139`), ClawWork (GitHub `HKUDS/ClawWork`), Rollout Cards (`arXiv:2605.12131`), MAJ-EVAL (`arXiv:2507.21028`), `tau2-bench` repo, and SGLang/vLLM/OpenRLHF/slime/Miles docs dated 2025–2026. They were pre-existing, not added by me; spot-checking them against the live web is recommended.
- Paper-reported engineering numbers kept as reported (Seer +74–97%, AReaL 2.77×, Sample Factory 100K fps) should ideally be cross-checked against the cited papers.
- The `[^rliable]` arXiv ID `2105.05246` is from confident knowledge (Agarwal et al., NeurIPS 2021) but was not verifiable offline; worth a link check by the coordinator.

## Sidebar additions

None (no new pages; requirement was to avoid creating summary/exercises pages).

## Commands run and results

- `npx prettier --write` on all seven edited files: clean (3 reformatted, 4 unchanged).
- `npx vitepress build docs`: success, all pages rendered, no dead-link errors.
- `git status`/`git diff --stat`: only the 7 in-scope files modified, +95/−56 lines.

## Open issues

- A.4's benchmark inventory (MMLU-Pro, BFCL, tau-bench naming, Claw-family entries) may drift as benchmarks evolve; suggest a periodic link/rot check, ideally via `npm run verify:code-links`.
- The three legacy stubs (`monitoring.md`, `parallelism.md`, `async-training.md`) still duplicate A.2 content by design (legacy-link entry points); a future worker may want to shrink them to pure redirects, but that is outside this task's scope.
- No code tests apply to this task (no `code/` files in scope; no test names given).
