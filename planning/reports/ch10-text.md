# Report: ch10-text — Chapter 10 text pass

## Summary

Read all three in-scope pages fully and made targeted correctness fixes plus tight coverage additions. Fixed: the broken CQL code snippet (undefined variables, concat on flattened tensors), the overstated CQL lower-bound claim, an inverted TD3+BC adaptive-weight formula, a wrong $\gamma$-decay number, a backwards DPO-$\beta$ explanation, the "RvS in online RL" mislabel, and a broken DT inference snippet (undefined `self.K`, wrong tensor shapes). Added: a data-audit subsection (quality/coverage/normalization) and a fine-tuning protocol note in 10.1, a full off-policy-evaluation section in 10.3 (previously entirely missing), and the new `review.md` summary-and-exercises page. All cross-references verified against the numbering table; link targets confirmed to exist.

## Technical fixes

1. **File:** `docs/chapter10_offline_rl/offline-data-distribution-shift.md` (CQL snippet)
   **Wrong claim:** code referenced undefined `logsumexp_q2`, `q_curr2` while the note said Q2 was "omitted", and `torch.cat([1-D tensors], dim=1)` raises a dimension error.
   **Correction:** rewrote the regularizer for Q1 only (`[B,1]` tensors concatenated on `dim=1` to `[B,3]`, `logsumexp(dim=1)`), using `torch.empty_like(a).uniform_(-1, 1)` for OOD stand-ins; note now matches the code.
   **Verification:** traced shapes by hand (`[B,1]`→`cat dim=1`→`[B,3]`→`logsumexp dim=1`→`[B]`→`.mean()` scalar); matches the CQL paper's sampled-logsumexp form. — suspected-issue 1, code part `[x]`
2. **File:** same (CQL guarantee)
   **Wrong claim:** "$\hat{Q}(s,a) \le Q^\pi(s,a)$ for all (s,a)" pointwise lower bound.
   **Correction:** softened to the actual theorem: $\hat{V}^\pi(s) = \mathbb{E}_{a\sim\pi}[\hat{Q}(s,a)]$ lower-bounds $V^\pi(s)$ on covered states — an expectation-under-$\pi$ guarantee, plus an $\alpha$-growing penalty gap on OOD actions.
   **Verification:** from solid knowledge of Kumar et al. 2020 Thm. 3.2 (bound is on the policy value, i.e. expectation over $\pi$-actions). — suspected-issue 1, theory part `[x]`
3. **File:** same (TD3+BC weight)
   **Wrong claim:** $\mathcal{L} = -\mathbb{E}[Q] + \lambda\,\mathbb{E}[(\mu-a)^2]$ with $\lambda = \alpha/\mathrm{mean}|Q(\cdot,\mu_{\rm old})|$, and "large Q → weaker BC".
   **Correction:** $\mathcal{L} = -\lambda\,\mathbb{E}[Q] + \mathbb{E}[(\mu-a)^2]$ with $\lambda = \alpha/\mathrm{mean}_{(s_i,a_i)}|Q(s_i,a_i)|$; $\lambda$ scales the Q-term down when returns are large so both forces stay balanced. Also fixed the §3.4 table row.
   **Verification:** matches Fujimoto & Gu 2021 (policy maximizes $\lambda Q - \mathrm{BC}$, $\lambda = \alpha/\mathrm{mean}|Q|$, $\alpha=2.5$); scale-invariance proof worked out in new review.md Exercise 7.
4. **File:** same (decay arithmetic)
   **Wrong claim:** "with γ=0.99, after 100 iterations less than 1% remains".
   **Correction:** $0.99^{100} \approx 0.366$ (≈37% remains); below 1% takes ≈460 iterations.
   **Verification:** recomputed: $100\ln0.99 \approx -1.005 \to e^{-1.005} \approx 0.366$; $\ln0.01/\ln0.99 \approx 458$.
5. **File:** same (notation)
   **Wrong claim:** bullet called $\|Q_T - Q^\pi\|_\infty$ the gap to "the true Q\*".
   **Correction:** "the true $Q^\pi$".
   **Verification:** follows from the displayed equation on the same page.
6. **File:** `docs/chapter10_offline_rl/sequence-modeling.md` (RvS line)
   **Wrong claim:** "RL via supervised learning (RvS) in online RL" and "Star-Vector".
   **Correction:** "RvS (offline RL cast directly as goal-conditioned supervised learning)" and "StARformer".
   **Verification:** RvS (Emmons et al. 2021/22) is an offline supervised method by definition; StARformer (Shang et al., ICML 2022) is the well-known DT follow-up whose name best explains the garbled "Star-Vector". — suspected-issue 2 `[x]`
7. **File:** same (DT `act` snippet)
   **Wrong claim:** used undefined `self.K`, passed unbatched 1-D sequences to a batched `forward`, and sliced `history.actions[-K-1:-1]`, dropping the most recent action.
   **Correction:** added `K=20`/`self.act_dim` to `__init__`; `act` appends current `(target_rtg, state)` plus a zero placeholder action, keeps last-K rows, and restores the batch dim before calling `forward`.
   **Verification:** traced shapes: `(T,1)/(T,D)/(T,A)` → `unsqueeze(0)` → `(1,T,·)` matching `forward`'s `(B,T,·)` contract and the `h[:, 1::3, :]` state indexing.
8. **File:** same (diffusion steps)
   **Wrong claim:** diffusion step "from 0 to T" and "iterate for T steps", reusing the trajectory-length symbol.
   **Correction:** diffusion horizon renamed to $K$ in both places.
   **Verification:** textual consistency check against §3's own definition of $T$ as trajectory length.
9. **File:** `docs/chapter10_offline_rl/experiments.md` (DPO $\beta$)
   **Wrong claim:** "β too large → steps too big → OOD; β too small → too conservative".
   **Correction:** swapped — larger $\beta$ = stronger KL anchor (conservative, preferences unseparated); smaller $\beta$ = weaker constraint (drift, OOD gibberish).
   **Verification:** DPO optimizes reward minus $\beta\mathrm{KL}(\pi\|\pi_{\rm ref})$; larger $\beta$ pins $\pi$ closer to $\pi_{\rm ref}$ by construction.

## Other changes

- 10.1 §2.5 (new, ~300 words): pre-algorithm data audit — behavior-policy quality/diversity, coverage sweeps over data fractions/quality splits, action gaps, and normalization consistency (freeze dataset statistics; reward scale and its interaction with $\alpha$/$\lambda$).
- 10.1 §3.2: +3-sentence offline-to-online fine-tuning protocol (keep conservative critic, mix buffers, small actor LR, watch KL drift, anneal conservatism on disagreement).
- 10.3 §5 (new, ~850 words): off-policy evaluation — OPE target and logging policy, trajectory IS with worked 3-trajectory example ($\hat{J}_{\rm IS}=40/3$, all weight on one trajectory), per-decision IS, clipping, WIS ($=10$ on the example), ESS ($=1$), direct/DR estimators with equation readings, FQE vocabulary (no-max backup), and a selection protocol (held-out split, two disagreeing estimators, abstain/fallback, reserve online rollouts for estimator audit). Reading-guide bullet and chapter-summary point 6 added to match.
- 10.1 minor display fix: none needed beyond above; Ch11 link display text ("Imitation Learning, Inverse RL, and Meta-RL") matches that chapter's topic, left as is.
- New `docs/chapter10_offline_rl/review.md`: 7 outcomes, 12 key equations with one-line readings, 5 pitfalls, 9 exercises (2 easy, 1 conceptual, 2 derivations, 2 numeric hand computations, 2 coding) each with a worked `::: details Solution` block; all numbers derived by hand (RTG $6,6,6,1$; error bound $20$ vs $2$; IS $40/3\approx13.3$ with ESS $1$; WIS $10$; bandit ESS $\approx25$ at 50% overlap, $\approx2$–$3$ at 5%).

## Coverage against checklist RL-6 (in-scope verdicts)

| Item                                                         | Verdict                                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| Demonstrations and behavior cloning                          | Partial (BC-as-regularizer covered; full BC is Ch11's topic)                    |
| Behavior-cloning likelihood and action distributions         | Out of scope (Ch11) — mentioned via AWR losses                                  |
| Covariate shift and compounding imitation error              | Out of scope (Ch11)                                                             |
| Interactive data aggregation and DAgger vocabulary           | Out of scope (Ch11)                                                             |
| Inverse RL and reward-learning boundary                      | Out of scope (Ch11)                                                             |
| GAIL and occupancy-matching vocabulary                       | Out of scope (Ch11)                                                             |
| Offline transition datasets and behavior-policy identity     | Covered (10.1 §1)                                                               |
| Dataset coverage, support and action gaps                    | Covered (10.1 §§1–2 + new §2.5)                                                 |
| Offline distribution shift and extrapolation error           | Covered (10.1 §1)                                                               |
| Pessimism and conservative value estimation                  | Covered (10.1 §2)                                                               |
| Conservative Q-Learning vocabulary                           | Covered (10.1 §2.2)                                                             |
| Implicit Q-Learning expectile and advantage weighting        | Covered (10.1 §2.3)                                                             |
| Advantage-weighted actor update vocabulary                   | Covered (10.1 §§2.3, 3.2–3.3)                                                   |
| Decision Transformer and sequence-modeling view (Optional)   | Covered (10.2)                                                                  |
| Online fine-tuning after offline initialization              | Partial → extended (10.1 §3.2 protocol note)                                    |
| Off-policy evaluation target and logging policies            | Was missing → added (10.3 §5)                                                   |
| Trajectory and per-decision IS for evaluation                | Was missing → added with worked example                                         |
| Weight clipping, normalization and ESS                       | Was missing → added                                                             |
| Direct, model-based and doubly robust estimator vocabulary   | Was missing → added                                                             |
| Fitted Q Evaluation vocabulary                               | Was missing → added                                                             |
| Unsupported actions and impossible counterfactuals           | Partial (unsupported/OOD covered; "impossible counterfactuals" now named in §5) |
| Offline model selection without online feedback              | Was missing → added (§5.4)                                                      |
| Dataset quality, diversity and coverage sweeps               | Was partial → added (§2.5)                                                      |
| Action and observation normalization consistency             | Was missing → added (§2.5)                                                      |
| Policy abstention, fallback and data-collection triggers     | Was missing → added (§5.4)                                                      |
| Online ground truth reserved for estimator audit             | Was missing → added (§5.4)                                                      |
| Full offline-RL theory and minimax guarantees (Depth)        | Skipped per instructions                                                        |
| High-stakes deployment from observational logs alone (Later) | Skipped per instructions                                                        |

## Needs external verification

- "StARformer" fix for "Star-Vector": high-confidence inference (only well-known DT-era paper matching the garbled name), but could not check the cited source offline.
- Paper years/venues/arXiv IDs in Further Reading and TT/Diffuser attributions were checked against solid knowledge and look right, but not against the papers themselves (no internet).
- The `target_return = 9000` HalfCheetah-expert comment is hedged ("roughly") and plausible; exact DT-paper target values not verified. (Addressed in this revision: replaced with a neutral example value plus "pick from the dataset's return range".)

## Review response

One line per reviewer finding (Errors 1–7, Doubts 1–10 from `planning/reviews/ch10-text.md`):

- Error 1 (DT `act` call-site mismatch): **accepted** — inference pseudocode now calls `model.act(state, target_return, history)` (3 args, matching the definition) and documents the buffer's `(rtg, state, action)` column contract.
- Error 2 (TD3+BC $\lambda$ denominator): **accepted** — restored $\lambda = \alpha / \mathrm{mean}_i|Q(s_i, \mu_\theta(s_i))|$ with "current policy's actions" wording (matches Fujimoto & Gu 2021 and the official implementation); my earlier claim that the dataset-action form "matches Fujimoto & Gu 2021" was wrong, and I retract it. Scale-invariance argument is unaffected.
- Error 3 (DPO $\beta$ "step size" paragraph): **accepted** — lead-in now reads "strength of the KL anchor to the reference policy (equivalently, the temperature of the implicit reward)" and both bullets rewritten without step-size language, consistent with Chapter 14.
- Error 4 (AWAC $\max_{a'}\pi(a'|s')$): **accepted** — now $\max\_{a'} Q(s', a')`.
- Error 5 (Exercise 9 division by zero): **accepted** — solution guards `den == 0` (returns NaN/0.0, dropped before averaging) and states ~8% of seeds at $p = 0.05$ carry zero information; verified by execution (18/200 degenerate, kept ESS mean 2.76 / median 3.0, WIS variance below IS).
- Error 6 (sidebar `config.mjs`): **no change — out of scope** (worker rules forbid editing `docs/.vitepress/config.mjs`). Coordinator action needed: add `{ text: '10.4 Summary and Exercises', link: '/chapter10_offline_rl/review' }` to the chapter-10 group. "Sidebar additions" entry retained for this.
- Error 7 (`content-review-notes.md` checkboxes): **no change — out of scope** (that file is outside my editable set). Coordinator action needed: mark both chapter-10 items `[x]`; fixes were the CQL snippet rewrite + expectation-under-$\pi$ guarantee correction, and the RvS/StARformer relabel.
- Doubt 1 (CQL `q_next` evaluated at wrong state): **accepted** — regularizer now uses a dedicated `a_pi = self.actor(s)` / `q_pi` at the same state $s$; `a_next` remains only for the Bellman target.
- Doubt 2 (CQL guarantee missing slack/assumptions): **accepted** — added "up to sampling and function-approximation error, and under coverage of the data" qualifier.
- Doubt 3 (IQL "guessing" in action gaps): **accepted** — sentence restricted to CQL-style logsumexp methods, with an explicit note that IQL never evaluates OOD actions.
- Doubt 4 ($K$ double meaning): **accepted** — diffusion horizon renamed to $N_{\text{diff}}$ in both places; §1 keeps $K$ for DT context length.
- Doubt 5 (buffer contract + `target_return = 9000`): **accepted** — pseudocode documents the `rtg [T,1] / states [T,D] / actions [T,A]` contract and appends `(target_return, state, action)`; value replaced with a neutral example plus dataset-range advice.
- Doubt 6 (OPE example same-state assumption + undefined $T$): **accepted** — added "assume all three episodes start from the same state" and defined $T$ as the episode length; same assumption noted in `review.md` Exercise 6.
- Doubt 7 (DR cumulative vs per-decision weight): **accepted with note** — formula kept (cumulative form is still doubly robust via the telescoping identity) with a parenthetical citing Jiang & Li 2016's per-decision form.
- Doubt 8 (roadmap paragraph missing §5): **accepted** — roadmap now names the OPE grading step.
- Doubt 9 (BCQ $n = 100$ vs $n = 10$): **accepted** — now "say n=10, as in BCQ Algorithm 1".
- Doubt 10 (`10.R` numbering): **rejected — per task spec**: the task explicitly requires front matter `title: 10.R Summary and Exercises` and H1 `# 10.R Summary and Exercises`, and forbids editing the sidebar; renumbering is the coordinator's job at merge time.

## Sidebar additions

- Title: `10.R Summary and Exercises`, link: `/chapter10_offline_rl/review` (new file `docs/chapter10_offline_rl/review.md`, front matter `title: 10.R Summary and Exercises`, H1 `# 10.R Summary and Exercises`).

## Commands run and results

- `npx prettier --write` on the four touched chapter files plus the report: chapter files already conformant ("unchanged"); report reformatted.
- Exercise 9 OPE snippet executed via `.venv/bin/python`: 200 seeds at 50% overlap give IS mean 0.978 / var 0.056, WIS mean 0.980 / var 0.036, mean ESS 24.9; at 5% overlap 18/200 seeds degenerate (zero arm-1 pulls), kept ESS mean 2.76 / median 3.0 with WIS variance (0.497) below IS (0.803) — matches the reviewer's numbers and the solution's stated expected behavior.
- `npx vitepress build docs`: **succeeds** (build complete, pages rendered).

## Open issues

1. **Build/format verified in this revision** — `npx prettier --write` clean and `npx vitepress build docs` succeeds. Two required follow-ups are outside my scope and left for the coordinator: adding `review.md` to the VitePress sidebar (`config.mjs`) and marking the two chapter-10 items `[x]` in `planning/content-review-notes.md`.
2. New OPE section lives in 10.3 (`experiments.md`), whose title is "Offline Reinforcement Learning and Preference Data" — the section fits ("selecting without deploying" for both robots and LLMs) but the coordinator may prefer retitling the page or relocating it.
3. Code in `code/` untouched per scope; the corrected CQL/DT snippets in the text were shape-traced by hand, not executed.
