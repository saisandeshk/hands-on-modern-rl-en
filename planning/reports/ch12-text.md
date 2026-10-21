# Report: ch12-text — Chapter 12 text pass

## Summary

Read all three Chapter 12 pages fully and made targeted correctness fixes plus tight coverage additions. Verified every numeric claim by recomputation and every cross-reference against the sidebar. Fixed the four suspected technical errors (all confirmed real), corrected one wrong chapter cross-reference in scope, added missing checklist treatments (count-based bonuses, bonus-vs-shaping, sparse/delayed/deceptive taxonomy, HER/goal-conditioning, privileged-information note), fixed two code-sketch bugs in the text, and created `review.md` with outcomes, equations, pitfalls, and 8 fully worked exercises (solutions verified by execution). `npx vitepress build docs` passes.

## Technical fixes

- [x] `intrinsic-motivation-exploration.md` — the $10^{-18}$ probability cannot be derived from $(\epsilon{=}0.1, 6 \text{ actions}, 50 \text{ steps})$. **Correction:** uniform random search gives $(1/6)^{50} \approx 1.2\times10^{-39}$; $\epsilon$-greedy with $\epsilon{=}0.1$ gives at most $(0.1/6)^{50} \approx 10^{-89}$ (only the random branch can discover an unknown sequence). Rewrote the "despairing number" section and the Reading-Guide bullet around $10^{-39}$ with corrected analogies ($10^{39}/10^{17} = 10^{22}$ short of one-per-second-since-Big-Bang). **Verified** with `.venv/bin/python` computation.
- [x] `intrinsic-motivation-exploration.md` — ICM on Mario described as "continuous control plus visual input"; Mario has discrete actions (ICM, Pathak et al. 2017, used discrete-action visual navigation: Mario/VizDoom). **Correction:** "discrete actions and visual input". Also fixed the ICM code sketch, which concatenated raw action indices into the forward model (`torch.cat([phi, a], -1)` with `MLP(feat_dim + action_dim, ...)`): now one-hot encodes `a` via `F.one_hot` in a shared `_features` helper. **Verified** by reading the snippet logic (shape mismatch otherwise).
- [x] `hierarchical.md` — HIRO's off-policy correction described as directly setting $g' = s_{t+c} - s_t$. HIRO (Nachum et al. 2018) actually samples candidate goals (original goal plus goals around the achieved displacement) and selects the one maximizing the current low-level policy's log-likelihood of the executed actions. **Correction:** rewrote the "How Goal Transition Works" details block with the argmax-over-candidates formula and an explicit note that "$g' = s_{t+c}-s_t$" is only a frequent-winner shorthand. **Verified** against knowledge of the HIRO paper (Sec. 3.3 / relabeling procedure).
- [x] `hierarchical.md` — latent-action learning credited to "Genie 3". Latent actions (LAM) come from Genie itself (Bruce et al. 2024); Genie 3 is the later real-time world model. **Correction:** "Genie went a step further and introduced latent action learning…".
- [x] `marl.md` — "the strong baseline we have used since Chapter 7" for PPO. PPO is Chapter 8 per sidebar. **Correction:** "since Chapter 8".
- [x] Out-of-scope chapter-number cites checked against the sidebar (no edits made, not in scope): `preface/introduction.md` lines 234 ("Chapters 8–9" for the DPO pipeline — should be ~Chapters 13–14) and 248 ("Chapter 8" for Agentic RL — should be ~Chapter 19) look stale; `chapter08_ppo/trust-region-clipping.md` ("Chapter 6" for REINFORCE) is **correct**; `chapter08_ppo/gae-reward-model.md` cites (Ch 4 reward-design, Sec 7.1 advantage, Ch 14 DPO, Ch 15 RLVR) are all **correct**.
- `marl.md` MAPPO code sketch — wrong reduction order: `s1` was already a mean before `torch.min(s1, s2)`, i.e. min-of-means instead of PPO's elementwise min-then-mean. **Correction:** `s1 = ratio * adv[i]`; `policy_loss = -torch.min(s1, s2).mean()`. Sign convention on the entropy term was checked and is correct (minimizing `-mean(logp)` increases entropy).
- Discount arithmetic in `hierarchical.md` checked: $0.99^{100} \approx 0.37$, $0.99^{1000} \approx 4\times10^{-5}$ — **correct, unchanged**. NGU episodic-novelty example ($0.67 \to 0.09$, "one-seventh") recomputed — **correct, unchanged**. Joint-action count $6^{10} \approx 60$M — **correct, unchanged**. Dreamer-V3 anchor `#dreamer-v3-and-a-new-generation-of-world-models` and Chapter 26 `self-play-outlook/` links — targets exist, **unchanged**.

## Other changes

- `intrinsic-motivation-exploration.md`: new subsection "Counts, Pseudo-Counts, and the Price of Undirected Exploration" (~450 words: entropy-regularized objective, optimism→MDP generalization, $\beta/\sqrt{N}$ bonus with numeric check, pseudo-counts via density models, PBRS vs. bonus with exhaustibility, visitation/success-coverage vocabulary); new sparse/delayed/deceptive taxonomy paragraph in §1.
- `hierarchical.md`: new §2.3 "Goal-Conditioned Policies and Hindsight Relabeling" (~330 words: UVFA, binary goal reward, HER relabeling with robot-arm example, off-policy requirement); old §§2.3–2.5 renumbered to 2.4–2.6; Chapter Summary + Further Reading (HER arXiv:1707.01495) updated.
- `marl.md`: new §4.3 "A Note on Privileged Information and Fair Comparisons" (~150 words).
- Created `review.md` (front matter `title: 12.R Summary and Exercises`): 8 learning outcomes, 16 key equations with one-line readings, 6 pitfalls, 8 ordered exercises (numeric, derivation, conceptual, 2 coding) with worked `details` solutions. Exercise 7's claimed outcome was corrected after execution showed bonus ≈ −42.4 vs plain ≈ −39.0 on seed 1 (comparable; note now says so and explains why).

## Coverage against checklist RL-5 (in-scope items)

| Item                                             | Status                | Where                                                                |
| ------------------------------------------------ | --------------------- | -------------------------------------------------------------------- |
| Exploration/exploitation as data collection      | covered               | 12.1 §1 + Ch 2 link                                                  |
| Undirected $\epsilon$ and entropy-based          | partial → covered     | new counts subsection (entropy objective)                            |
| Optimism and UCB vocabulary                      | covered               | UCB formula + optimism/MBIE                                          |
| Count, pseudocount, intrinsic-reward vocabulary  | partial → covered     | new counts subsection                                                |
| Curiosity, representation-dependent novelty      | covered               | ICM/RND (unchanged)                                                  |
| Sparse, delayed, deceptive rewards               | partial → covered     | new taxonomy paragraph                                               |
| State-action visitation, success coverage        | partial → covered     | new coverage paragraph                                               |
| Bonuses vs reward shaping                        | missing → covered     | PBRS + exhaustibility                                                |
| Privileged info, unfair comparisons              | missing → covered     | marl §4.3                                                            |
| Goal-conditioned policies + relabeling, HER      | missing → covered     | hierarchical §2.3                                                    |
| Partial observability / recurrent policy vocab   | partial               | NGU LSTM/Retrace mention; full RNN treatment lives in Ch 10/11 scope |
| Continuous-action/DDPG/TD3/SAC/temperature items | out of scope (Ch 9)   | 9.1, 9.2 sequence-modeling — not duplicated                          |
| Empirical-method items (seeds, IQM, budgets, …)  | out of scope (App. A) | noted, not duplicated                                                |
| Multi-agent + hierarchical exploration (Depth)   | covered               | 12.2, 12.3                                                           |
| Sim-to-real (Later)                              | skipped               | Ch 24 territory                                                      |

## Needs external verification

- "First algorithm to surpass human performance on all 57 Atari games" (Agent57, Badia et al. 2020): standard characterization; the concurrent MuZero/Replay comparison nuances were left as-is.
- NGU/Retrace distributed-training cost claims ("hundreds of TPU cores", "billions of frames"): plausible but not re-verified; left as-is.
- HIRO candidate-set details in my rewrite (original goal + achieved-displacement-centered candidates): faithful to the paper's procedure at the level of a textbook summary; exact candidate count (10) deliberately omitted.
- Preface stale chapter numbers (lines 234, 248) need another worker's fix; I did not touch them.

## Sidebar additions

- Title: `12.R Summary and Exercises`, link: `/chapter12_exploration_marl_hierarchical/review` (file created; coordinator to add to sidebar).

## Commands run and results

- `.venv/bin/python -c` probability/discount/NGU/HIRO-arithmetic checks — all match values in the text.
- `.venv/bin/python` import + 10-episode smoke run of `code/chapter12_exploration_marl_hierarchical/multi_agent_marl.py` — runs.
- `.venv/bin/python` full Exercise 7 solution (200-episode IQL vs IQL+bonus) — runs; plain −39.0, bonus −42.4 (note corrected accordingly).
- `.venv/bin/python` Exercise 8 numpy check — scores [−25.405, −1.805, −10.205], best = 4.0.
- `npx prettier --write` on the 4 touched files — clean.
- `npx vitepress build docs` — success (18 s), no dead links.
- `git status --short` (via allowed command set; `|`-piped and `sed` variants were rejected by the sandbox, used `.venv/bin/python` for byte inspection instead).

## Open issues

- Existing ch12 pages lack the `--- title: ---` front-matter block used elsewhere; I added it only to the new `review.md` per instructions and left existing pages untouched to keep the diff scoped.
- `marl.md` MADDPG sketch ends with a `...` stub and `value normalization: as in R2D2` assumes R2D2 context from §12.1; acceptable as sketches but a future pass could complete the optimizer lines.
