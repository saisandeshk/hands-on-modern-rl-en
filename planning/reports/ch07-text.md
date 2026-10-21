# Report: ch07-text (Chapter 7 text pass)

## Summary

Reviewed all seven Chapter 7 pages end to end, recomputed every numeric example, and fixed all errors found. Main work: corrected two wrong equations in `ac-frontier.md` (V-trace double-counted $\gamma$; SAC target misplaced $\gamma$), fixed a misleading variance explanation in `advantage-function.md`, fixed equation tags/figure numbers/a bad asset path, added the missing in-scope coverage (new $n$-step/GAE/$\gamma$-$\lambda$ section in `critic-training.md`, tanh log-prob correction and loss-coefficient clarification in `pendulum.md`), cleaned two stale editorial notes in `alphago.md`, and created `review.md` (summary + exercises with verified solutions). Build passes.

## Technical fixes

1. **V-trace formula double-counted the discount** (`ac-frontier.md`): the old display multiplied by $\gamma^{t-s}$ outside **and** $\prod \gamma c_k$ inside, giving $\gamma^{2(t-s)}$. Corrected to $v_s = V(s) + \sum \gamma^{t-s}(\prod c_k)\rho_t\,\delta_t$, matching Espeholt et al. (IMPALA, 2018). Verified against the standard V-trace definition from knowledge of the paper.
2. **SAC critic target misplaced $\gamma$** (`ac-frontier.md`): old target $r + \gamma\min Q - \alpha\log\pi$ left the entropy term undiscounted. Corrected to $r + \gamma(\min_j Q_j(s',a') - \alpha\log\pi(a'|s'))$, the standard SAC target (Haarnoja et al., 2018). Verified from knowledge of the SAC backup.
3. **Baseline-variance paragraph was misleading** (`advantage-function.md`): claimed the centered signal has lower variance on the shown 4-sample slice, but both $G_t$ and $A$ slices have variance exactly $32.5$ (recomputed: means 11 vs 1, variances both $(49+16+16+49)/4$). Rewrote to state that a constant shift cannot change that slice's variance and that the real reduction comes from averaging over states via the score-function identity $\mathbb{E}[\nabla\log\pi]=0$. Verified by recomputation.
4. **Wrong equation tags** (`critic-training.md`): $\tag{6.3}\to\tag{7.3}$, $\tag{6.4}\to\tag{7.4}$.
5. **DP corridor converged values rechecked** (`critic-training.md`): solved $V(S)=-1.5+V(M)$, $V(M)=-1.2+0.2V(S)$ giving $V(S)=-3.375$, $V(M)=-1.875$ — matches page. Rounds 1–3 arithmetic ($-1.2$; $-2.4,-1.44$; $-2.832,-1.68$) all recomputed correct. MC targets ($-7,-5,-4,-2,-1$) and sequential $\alpha=0.5$ updates ($-3.5,-4.25,-2,-3.125,-1.5$) recomputed correct. TD steps ($\delta=-2,0,-3,-1.5,+0.5$; $V$: $-1,-1,-1.5,-1.75,-1.25$) recomputed correct, including $\partial L/\partial V = -2\delta = 6$. No change needed; recording verification.
6. **Actor-Critic numerics rechecked** (`actor-critic.md`): $G_t=4.9010$ (recomputed $1+0.99+0.9801+0.970299+0.96059601=4.9010$), $\log0.6\approx-0.5108$, $\delta=1.97$, single-step $\delta=1.48$, $L_{\text{actor}}=1.7819$, $L_{\text{critic}}=2.1904$, total $3.9723$, trace values ($-0.9163$, $1.6310$, $3.1684$, $4.7994$) all recomputed correct. No change needed.
7. **Advantage worked examples rechecked** (`advantage-function.md`): $G_0=5.51$, $G_1=3.9$ correct; TD examples ($+0.6$, $-2.4$, $0$) correct; 3-step $\delta$ values ($+4.6,-2.1,+1.0$) and MC advantages ($+3.52,-1.2,+1.0$) recomputed correct. No change needed.
8. **Title/sidebar mismatch** (`actor-critic.md`): front matter said "7.2 Actor-Critic Architecture" while sidebar and H1 say "Updates". Changed front matter to `7.2 Actor-Critic Updates`.
9. **Over-specific unverifiable numbers softened** (`ac-frontier.md`): "roughly 200M params / 44 days / hundreds of millions of games / 12,000 TPU-years" replaced with qualitative wording ("well over a hundred million parameters", "several weeks", "a very large number of games"); SAC robot-table intro softened to "SAC-style maximum-entropy methods … representative examples" since per-task SAC attribution (e.g. Shadow-Hand pen rotation) could not be settled offline. All listed under Needs external verification.
10. **No recorded suspected issues**: the task listed "(none recorded)" — nothing to check off.

## Other changes

- **New in-scope coverage — $n$-step advantages, GAE, $\gamma$/$\lambda$** (`critic-training.md`): added section "From One Step to $n$ Steps: Preview of GAE" with Equations (7.5)–(7.6), a worked example reusing Section 7.1 numbers ($A^{(1)}_0=4.6$, $A^{(2)}_0=2.71$, $A^{(3)}_0=3.52=G_0-V$; GAE at $\lambda=0.95\approx3.54$ — all recomputed and cross-checked in Python), plus $\gamma$-horizon and $\lambda$ bias–variance rules and the `gamma=0.99, gae_lambda=0.95` link to this chapter's scripts. Points forward to Chapter 8 for full depth.
- **Pendulum** (`pendulum.md`): fixed figure labels `6.4-x` → `7.4-x` (4×); fixed copy path `docs/chapter09_actor_critic/images/` → `docs/chapter07_actor_critic/images/`; clarified illustrative loss coefficients vs SB3 config (`vf_coef=0.5`, `ent_coef=0.0`); added tanh-squash log-prob correction tip; added `gae_lambda=0.95` row linking to Section 7.3.
- **BipedalWalker** (`bipedalwalker.md`): fixed figure labels `6.6-x` → `7.6-x` (3×).
- **Advantage page** (`advantage-function.md`): removed self-link in roadmap row and a duplicated intro sentence.
- **AlphaGo** (`alphago.md`): removed stale "prompts are kept in Chinese" note (code is English); replaced generic leftover Section Summary with a real chapter-mapping summary; added warning that the demo evaluates the value net at terminal nodes instead of backing up the true outcome.
- **New page** (`review.md`): 7 learning outcomes, 8 key equations with one-line readings, 5 pitfalls, 8 ordered exercises (conceptual, 2 derivations, 3 numeric hand computations, 2 coding) each with a verified worked solution (all numerics recomputed; Python snippets executed — see Commands).

## Needs external verification

- AlphaStar specifics softened in this pass: exact parameter count, league training duration, total self-play games, TPU-time, and "top 0.2%" claim against Vinyals et al., Nature 2019.
- Per-task attribution in the SAC robot table (Shadow Hand pen rotation, Sawyer grasping, ANYmal locomotion, Franka door-handle) to SAC vs adjacent max-entropy/dexterous-manipulation work.
- `discrete 21-bin → 21^4 = 194,481` arithmetic verified correct; the claim that this discretization "barely runs" is qualitative and fine.
- JGDB/KGS/CWI dataset sizes in `alphago.md` were not independently verifiable offline; left untouched.

## Sidebar additions

- Title: **7.R Summary and Exercises**; link: `/chapter07_actor_critic/review` (coordinator to renumber/place after 7.7).

## Commands run and results

- Recomputation checks: `.venv/bin/python -c` snippets for GAE/$n$-step/exercise numerics (Ex2: 1.8/2.61/1.80; Ex3: 1.0/0.825625/0.7525; horizons 10/100/1000; decays 0.366/0.349; Ex1 −0.02) — all match the text.
- Executed torch one-step-update snippet (prob $0.5\to0.5493$, $V$ $1.0\to1.396$) and `compute_gae` reverse-loop implementation — outputs match review.md solutions.
- `npx prettier --write` on all 8 touched files — clean.
- `npx vitepress build docs` — **success** (16.45 s), no dead links.
- No `code/` changes (out of scope); no code test suite named in task, so no pytest run. Did not run the 300k/3M-step training scripts.

## Open issues

- `actor-critic.md` roadmap table references `[Chapter 8 GAE](../chapter08_ppo/gae-reward-model)` whose page title is "8.3 Advantage Estimation and Reward Modeling" — link target exists and build passes, but Chapter 8's numbering may deserve a coordinator check.
- Section ordering vs numbering: sidebar order is 7.2 Actor-Critic Updates **before** 7.3 Critic Training, yet 7.2's intro says "in the previous two sections we met the advantage and Critic training" — mild forward-reference tension; flagging for coordinator rather than reordering pages unilaterally.

## Review response

- Error 1 (ac-frontier self-contradiction on AlphaStar scale): accepted. Softened league paragraph ("many distinct policies", "millions of self-play games over weeks of league training") and comparison-table cell ("millions of self-play games over weeks of training"); removed surviving "hundreds of millions of games, 44 days".
- Error 2 (review.md Exercise 8 direction): accepted. The actor-term gradient is $\log\pi = -0.6931 < 0$, so removing `.detach()` pushes $V$ up (optimistic), not down. Changed "pessimistic" to "optimistic (it inflates $V(s)$, shrinking $\delta$ …)".
- Error 3 (pendulum tanh tip describes wrong squashing): accepted. Rewrote tip: the page's head squashes the mean and clips the sample; SB3 A2C samples unsquashed and clips (no Jacobian needed); the correction applies only when the sample itself is squashed (SAC-style).
- Error 4 (pendulum Jacobian drops $\log 2$ factor): accepted. Equation now reads $\log\pi(a|s) = \log\mathcal{N}(u;\mu,\sigma) - \sum_i \log(2(1-\tanh^2 u_i)+\epsilon)$.
- Error 5 (review.md not in sidebar / "7.R" numbering): rejected. Task mandates `review.md` with title `7.R` and forbids editing `docs/.vitepress/config.mjs`; sidebar registration was already listed as a coordinator action. No file change.
- Error 6 (7.2/7.3 "previous section" contradiction): accepted. Reworded both intros to cite explicit section numbers (7.1 for advantage, 7.3 for Critic details) instead of implying advantage → critic-training → actor-critic order.
- Error 7 ("from Chapter 3" TD reference): accepted. Changed to "from Chapter 4", matching the linked `dp-mc-td` page.
- Error 8 (orphan "## References" blocks): accepted. Added `[⁠^2]` marker at the REINFORCE gradient in actor-critic.md, `[⁠^1]` marker at the TD-loss intro in critic-training.md (removed uncited A3C `[⁠^2]` there), and `[⁠^1]`/`[⁠^2]` `[⁠^3]` markers at the Lee Sedol and AlphaZero sentences in alphago.md, so no page ends with an empty references block.
- Doubt 1 (AlphaStar "well over a hundred million" still too precise): accepted. Reworded to "roughly two orders of magnitude above a small MLP baseline" in both places.
- Doubt 2 (SAC robot-table attribution): accepted in part. Added an explicit "illustrative, not single-paper claims" qualifier, retitled Figure 4 to "SAC-style … methods", and de-quantified the Sawyer row ("learns grasping … without demonstrations"); full per-task sourcing left under Needs external verification.
- Doubt 3 ("hundreds of distinct policies"): accepted. Changed to "many distinct policies" (folded into Error 1 fix).
- Doubt 4 (Isaac Lab table over-precision): accepted. Retitled the column to "Typical order of magnitude (varies by task, environment count, and hardware)" with softened rows.
- Doubt 5 (Pendulum missing +1 alive bonus): rejected. Gymnasium `Pendulum-v1` reward is exactly $-(\theta^2 + 0.1\dot\theta^2 + 0.001a^2)$ with maximum 0; there is no alive bonus, so "close to 0" and the negative return scale stand. No change.
- Doubt 6 (pendulum front-matter/H1 title mismatch): accepted. Changed front matter to `7.4 Hands-on: Pendulum Continuous Control` to match H1 and sidebar.
- Doubt 7 (alphago warning parenthetical): accepted. Rewrote to distinguish game-over-by-passes from full-board-no-legal-moves; the code checks neither and always backs up the value estimate.
- Doubt 8 ($\nabla_\theta$ for Critic path): accepted. Changed to $\nabla_\phi$ with a shared-trunk note.
- Doubt 9 (Ch7 $r_{t+1}$ vs Ch8 $r_t$ indexing): rejected. Both conventions are internally consistent and standard; renaming either chapter is out-of-scope churn. No change.
- Doubt 10 (`code/` scripts print "Chapter 6"): rejected as out of scope. `code/` edits are forbidden for this task; flagged for the code owner/coordinator. No change.
