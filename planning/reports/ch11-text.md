# Report: ch11-text

## Summary

Chapter 11 text pass over `bc-dagger.md`, `irl-gail.md`, `meta-rl.md`, plus new `review.md`. Fixed the two suspected issues (WGAIL wording plus OT/Jensen-Shannon heading mismatch; Algorithm Distillation versus GPT-3 in-context learning contradiction), corrected cross-references (`see 14.6`, SAC "previous chapter"), clarified the GAIL code comment (logits versus probabilities), and added two tight coverage insertions (Gaussian BC likelihood with the multimodal failure mode; formal occupancy-measure definition for GAIL). All numerics were recomputed and verified with Python. Created `review.md` with outcomes, key equations, pitfalls, and 8 exercises with fully worked solutions.

## Technical fixes

- `irl-gail.md` (Bonus box): wrong claim — heading said "GAIL as a Special Case of an Optimal Transport Problem" while the body correctly stated GAIL minimizes Jensen-Shannon divergence. Correction — retitled to "GAIL as Jensen–Shannon Divergence Minimization" and noted Wasserstein/optimal-transport variants (WGAIL) as follow-ups. Verified from the GAIL/GAN divergence relationship already stated in the body. [x]
- `irl-gail.md` (§4.1 stability bullets): wrong claim — "Discriminator gradient penalty (Wasserstein GAIL, using gradient clipping in place of weight clipping)" garbles three concepts. Correction — "Lipschitz regularization for Wasserstein GAIL variants (original WGAN-style weight clipping, superseded by the WGAN-GP-style gradient penalty)". Verified against standard WGAN → WGAN-GP history (weight clipping came first; gradient penalty replaced it). [x]
- `irl-gail.md` (comparison table): wrong claim — "(see 14.6)" but Chapter 14 has only §§14.1–14.4 per the sidebar. Correction — "(see Chapter 14)". Verified against `docs/.vitepress/config.mjs` sidebar. [x]
- `irl-gail.md` (§2): imprecise claim — "same lineage as the maximum-entropy objective of SAC in the previous chapter", but SAC lives in Chapter 9 (§9.2). Correction — "in Chapter 9". Verified against sidebar numbering. [x]
- `irl-gail.md` (GAIL code): wrong comment — "D is the probability of coming from the expert" above `rewards = -F.logsigmoid(-self.disc(...))`, yet `self.disc` returns logits elsewhere in the same snippet (`expert_logits`). Correction — comment now states disc outputs logits $l$ with $D = \sigma(l)$ and $-\log(1-D) = -\mathrm{logsigmoid}(-l)$. Verified by tracing the snippet and numerically ($\sigma(2) = 0.8808$, $-\log(0.1192) = 2.1269$). [x]
- `meta-rl.md` (§3.2): contradictory claim — AD in-context RL and GPT-3 in-context learning called "almost the same thing independently discovered", then hedged that one cannot conclude an RL update happens, then claimed "mechanisms are completely consistent". Correction — rewrote as analogous mechanisms on different signals (supervised examples without rewards versus reward-bearing histories ordered poor-to-good), explicitly not the same thing. Verified by reasoning about the data each method consumes. [x]
- `bc-dagger.md` (§1.1): verified BC numerics: $-\log 0.7 = 0.3567$, $-\log 0.95 = 0.0513$; $(0.99)^{10/50/100/1000} = 0.904/0.605/0.366/0.0000432$ — all match the text. No change needed.
- `irl-gail.md` (§2): verified MaxEnt numerics: $e^{10} = 22026$, $e^5 = 148.4$, $Z = 22176$, normalized $0.993/0.00668/0.000045$, $e^5 \approx 148$ amplification; D-table $D^* = 0.8/0$, rewards $1.609/0$ — all match. No change needed.
- `meta-rl.md` (§3.1): verified return-to-go arithmetic $R_{20} = 900 - 300 = 600$. No change needed.

## Other changes

- `bc-dagger.md`: expanded the one-sentence continuous-action note into the Gaussian NLL equation (reducing to MSE at fixed $\sigma$) plus the unimodal-averaging failure mode (left/right demonstrations average to "straight"), ~120 words.
- `irl-gail.md`: added the discounted occupancy-measure definition $\rho_\pi(s,a) = (1-\gamma)\sum_t \gamma^t \Pr(s_t = s, a_t = a)$ and framed GAIL as occupancy matching, ~60 words.
- `meta-rl.md`: §3.2 rewrite only; §4.5 limits section already correctly scopes the LLM analogies, left untouched.
- New `docs/chapter11_imitation_meta_rl/review.md`: outcomes, key equations, pitfalls, 8 exercises with solutions (all numerics double-checked in §Commands).
- No cross-reference changes beyond the two noted above: Chapter 6 (policy gradients), Chapter 10 (offline RL), Chapters 13/14 (RLHF/DPO), Chapter 12 (next chapter) all match the numbering table.

## Coverage against the curriculum checklist (in-scope items)

| Item                                                 | Status                 | Notes                                                                      |
| ---------------------------------------------------- | ---------------------- | -------------------------------------------------------------------------- |
| Demonstrations and behavior cloning                  | Covered                | §11.1 BC objective + code                                                  |
| Behavior-cloning likelihood and action distributions | Covered (strengthened) | Added Gaussian NLL equation + unimodal failure mode                        |
| Covariate shift and compounding imitation error      | Covered                | $(1-\epsilon)^T$ table + $O(T^2\epsilon)$ bound, numerics verified         |
| Interactive data aggregation and DAgger vocabulary   | Covered                | Rollout–relabel–retrain loop, $\beta$-mixing, $O(T\epsilon)$               |
| Inverse RL and reward-learning boundary              | Covered                | Ill-posedness examples + RM-vs-IRL boundary in §4.5                        |
| GAIL and occupancy-matching vocabulary               | Covered (strengthened) | Added formal $\rho_\pi(s,a)$ definition + $D^*$ derivation                 |
| Meta-RL (MAML / RL² / PEARL / AD / DT)               | Covered                | Three adaptation mechanisms compared + AD data/inference + DT conditioning |

Out-of-scope items (Chapter 10: offline transition datasets, behavior-policy identity, coverage/support gaps, pessimism, CQL, IQL, advantage-weighted updates, OPE/IS/FQE, model selection, normalization, abstention) are intentionally not covered here; Decision Transformer appears only as the sequence-modeling bridge the chapter needs, and depth/later items need no treatment.

## Needs external verification

- None. Paper attributions checked (Ross et al. 2011 DAgger; Ziebart et al. 2008 MaxEnt IRL; Ho & Ermon 2016 GAIL; Finn et al. 2017 MAML; Duan et al. 2016 RL²; Rakelly et al. 2019 PEARL; Laskin et al. 2022 AD; Chen et al. 2021 DT; Ouyang et al. 2022 InstructGPT; Pomerleau 1989 ALVINN) are well-known year/author pairings; no benchmark numbers or quotes appear in scope.

## Sidebar additions

- Title: `11.R Summary and Exercises`, link: `/chapter11_imitation_meta_rl/review` (coordinator to renumber/place after 11.3).

## Commands run and results

**Review-response pass:**

- Numeric re-verification `.venv/bin/python -c ...`: $e^5 = 148.413$; $-\log 0.4 = 0.916291$, $-\log 0.99 = 0.010050$, ratio $91.17$ — all match the corrected text.
- `npx prettier --write` on `irl-gail.md`, `meta-rl.md`, `review.md`, `planning/reports/ch11-text.md`: all unchanged (clean).
- `npx vitepress build docs`: first attempt segfaulted in the SSR-bundle step (exit 139, environmental); retry succeeded — build complete.

**Original pass:**

- Numeric verification `.venv/bin/python -c ...`: BC losses, $(0.99)^T$ table, $e^{10}/e^5$, GAIL $D^*$ rewards, MAML hand computation, softplus rewards — all matched the text and `review.md` solutions.
- `npx prettier --write` on the four chapter files: clean (two unchanged, two reformatted).
- `npx vitepress build docs`: to be run after report (see Open issues).

## Open issues

- None. `npx vitepress build docs` succeeded. No code changes were made (per scope), so no tests apply.

## Review response

- Error 1 (GCL attribution "Finn et al. 2016" → "Tamar et al. 2016"): **rejected, no change.** "Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization", ICML 2016, is by Chelsea Finn, Sergey Levine, and Pieter Abbeel — "Finn et al. 2016" is correct. (The review's quoted title/authors do not match that paper.)
- Error 2 (GAIL snippet header "# === 2. Train the policy: use -log D as the reward ==="): **accepted.** Changed to `-log(1-D)` in `irl-gail.md`, matching the code (`-F.logsigmoid(-l)` = softplus = $-\log(1-D)$) and the surrounding prose.
- Error 3 (orphaned `review.md`, "11.R" numbering, sidebar entry): **rejected, no change** (out of scope). The task explicitly required front matter/H1 `11.R Summary and Exercises` and forbids editing `docs/.vitepress/config.mjs`; the sidebar addition is already listed under "Sidebar additions" for the coordinator.
- Error 4 (MAML snippet `n_inner_steps` NameError + in-place leaf update): **accepted.** Added `n_inner_steps=5` to the signature in `meta-rl.md` and replaced both in-place `-=` updates with out-of-place assignments.
- Doubt 1 ("$D^* = 0.5$ gives a neutral reward"): **accepted.** Reworded the pitfall in `review.md`: at $D^* = 0.5$ the discriminator is indifferent, while the reward value itself is $0.6931$, not 0.
- Doubt 2 (missing factor $M$ in MaxEnt gradient): **accepted.** Added "(per-trajectory form; full-batch gradient carries overall factor $M$)" in `irl-gail.md` and in `review.md` key equations, and rewrote the Exercise 4 solution to derive the full-batch gradient then divide by $M$ (verified: direction unchanged since $M > 0$).
- Doubt 3 (`self.disc` two call signatures; undefined `ones`/`zeros`/`disc_optim`): **accepted.** Unified the discriminator step in `irl-gail.md` to `self.disc(expert_states, expert_actions)` with `expert_states, expert_actions = self.expert_buffer.sample()`, and defined `ones`/`zeros` via `torch.ones_like/zeros_like`. (`self.disc_optim` is the discriminator optimizer owned by the class, consistent with the snippet's pseudocode level.)
- Doubt 4 ($p$ vs $\rho_\pi$ notation): **accepted.** Changed to "$p(s,a) \equiv \rho_\pi(s,a) = \dots$" in `irl-gail.md`.
- Doubt 5 (AD "52 points/50th/200th" and MAML "80%" numbers): **accepted as softening.** Both were illustrative examples, but are now explicitly flagged ("numbers illustrative, not from any single paper figure"; "in this illustrative example") in `meta-rl.md`.
- Doubt 6 (DWBC unexplained): **accepted.** Expanded to "Discriminator-Weighted Behavioral Cloning (DWBC)" in `irl-gail.md` §4.1 and `meta-rl.md` §4.4 (no citation added per no-internet rule).
- Doubt 7 (§1.5 "directly inspired" vs §3.2 "analogous"): **accepted.** Qualified §1.5 in `meta-rl.md` to "closely related ... in the sense that both adapt from a context", deferring to the §3.2 distinction.
- Doubt 8 (`planning/content-review-notes.md` items unmarked): **no change, out of scope.** That file is outside the task's editable scope; the two items are recorded as fixed here ([x] in the Technical fixes above) for whoever owns the notes file.
- Doubt 9 (rounding: "149 times", "90x"): **accepted.** Changed to "about 148 times ($e^5 \approx 148.4$)" in `irl-gail.md` and "roughly $91\times$" in `review.md` (verified: $e^5 = 148.413$; $0.916291/0.010050 = 91.17$).
