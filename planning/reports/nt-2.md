# nt-2 report — notation consistency, Chapters 6–12

## Summary

Audited all 40 pages in `docs/chapter06_policy_gradient`, `chapter07_actor_critic`,
`chapter08_ppo`, `chapter09_continuous_control`, `chapter10_offline_rl`,
`chapter11_imitation_meta_rl`, `chapter12_exploration_marl_hierarchical` against
`docs/appendix_d_math/notation.md` (D.5). Made 41 small edits in 22 files:
no derivation rewritten, no number changed, no code touched. Chapter 6 was
already consistent (2 spots fixed). The main fix aligns Chapter 7's per-step
reward index to the book-dominant $r_t$ convention (it mixed $r_{t+1}$,
$R_{t+1}$, and 1-based $r_1$ across pages, contradicting its own §7.2
trajectory table). Chapter 8's old-policy/KL/ratio symbols were unified to
$\pi_{\text{old}}$, $D_{\mathrm{KL}}$, $p_0$; overloaded symbols
($\tau$, OU $\mu$/$\sigma$, $\xi$, $\beta$, $c$, $A_t$/$\hat{A}_t$, bare $D$)
got one-line disambiguations where introduced. `npx vitepress build docs`
succeeds.

## Technical fixes

Notation-change table (page, old → new, reason):

| Page                                       | Old notation                                                                                   | New notation                                                                                                                     | Reason                                                                                                                                                                        |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| §6.5 `baseline-experiment.md` (2×)         | $A_t = G_t - V(s_t)$                                                                           | $\hat{A}_t = G_t - V(s_t)$                                                                                                       | Same chapter (§6.3, review) writes the MC advantage estimate with the hat; bare $A_t$ for the same object violates within-chapter rule. Matches notation page $\hat{A}_t$.    |
| §7.1 `advantage-function.md`               | $G_0 = r_1 + \gamma r_2 + \gamma^2 r_3$ (and $G_1, G_2$)                                       | $G_0 = r_0 + \gamma r_1 + \gamma^2 r_2$ (etc.)                                                                                   | 1-based reward index used nowhere else in the book; align to dominant $r_t$ (notation D.5, Ch. 6/8). Values unchanged ($5.51, 3.9, 1$).                                       |
| §7.1 `advantage-function.md`               | $Q^\pi(s,a) = \mathbb{E}[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s, A_t = a]$               | $\mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) \mid s_t = s, a_t = a]$                                                                  | $R_{t+1}$ + uppercase $S_t/A_t$ appear only here; rest of book uses lowercase $r_t, s_t, a_t$.                                                                                |
| §7.2 `actor-critic.md`                     | $G_t = r_{t+1} + \cdots + \gamma^4 r_{t+5}$                                                    | $G_t = r_t + \cdots + \gamma^4 r_{t+4}$                                                                                          | Old form contradicted the page's own trajectory table (rewards listed at rows $t$–$t{+}4$); new form matches table and Ch. 6/8. Value $4.9010$ unchanged.                     |
| §7.2 `actor-critic.md` (3× prose/table/δ)  | $r_{t+1} = 1.0$, $\delta = r_{t+1} + \dots$                                                    | $r_t$, $\delta = r_t + \dots$                                                                                                    | Same $r_t$ alignment; $\delta = 1.97$ unchanged. Symbol-table row $r_{t+k}$ needed no change (already correct under $r_t$).                                                   |
| §7.2 `actor-critic.md`                     | $\alpha$ first used bare in Critic update                                                      | gloss "(here $\alpha$ is the step size, as in Chapters 4 and 6)"                                                                 | Rule 3: step-size $\alpha$ had no in-chapter definition (matches notation page $\alpha$ = learning rate).                                                                     |
| §7.3 `critic-training.md` (3×)             | $G_{t:t+n} = r_{t+1} + \dots$, TD target $r_{t+1} + \dots$, $\delta_{t+k} = r_{t+k+1} + \dots$ | $r_t$-based forms                                                                                                                | Same $r_t$ alignment; §7.4 (Pendulum, $y_t = r_t + \dots$) and V-trace ($r_t$) already use $r_t$, so chapter is now uniform.                                                  |
| §7 review `review.md` (5×)                 | $\delta_t = r_{t+1} + \dots$, proof expansion with $r_{t+k+1}$                                 | $r_t$-based forms                                                                                                                | Same alignment; telescoping to $G_t - V(s_t)$ rechecked term-by-term.                                                                                                         |
| §8.1 `ppo-bipedal-walker.md`               | $r_t(\theta) = \pi_{new}/\pi_{old}$                                                            | $\pi_\theta / \pi_{\text{old}}$                                                                                                  | $\pi_{new}$ appears nowhere else; $\pi_{\text{old}}$ is the §8.2/§8.4 form and a notation-page variant.                                                                       |
| §8.1 `ppo-bipedal-walker.md`               | $D*{KL}(\pi*{old} \\                                                                           | \pi\_{new})$                                                                                                                     | $D*{\mathrm{KL}}(\pi*{\text{old}} \\                                                                                                                                          | \pi\_\theta)$                                                                                                                                                                                      | Notation page prefers $D_{\mathrm{KL}}$; old-policy subscript aligned. |
| §8.2 `trust-region-clipping.md` (7×)       | $D_{\text{KL}}$                                                                                | $D_{\mathrm{KL}}$                                                                                                                | Unlisted $\text{KL}$ variant → notation-preferred $\mathrm{KL}$.                                                                                                              |
| §8.2 `trust-region-clipping.md`            | $A_t$ unexplained at first PPO use                                                             | added "$A_t$ is the estimated advantage $\hat{A}_t$ (GAE, §8.3); PPO objectives drop the hat"                                    | Same object written $\hat{A}_t$ in theorem statements on the same track ( Knights §8.4 L12 vs L189); note resolves without touching formulas.                                 |
| §8.4 `ppo-math.md` (3×)                    | $\rho_0(s_0)$, "initial distribution $\rho_0$" (2×), $\bar{D}_{\text{KL}}$ (2×)                | $p_0(s_0)$, $p_0$, $\bar{D}_{\mathrm{KL}}$                                                                                       | $\rho_0$ for the initial distribution collides with ratio $\rho_t$ (notation page: $p_0(s)$); prose "initial distribution $p_0$" now defines it at use. KL subscript aligned. |
| §8.4 `ppo-math.md`                         | "Advantage function $A_t$" bullet                                                              | gloss "(the estimated advantage $\hat{A}_t$; hat dropped in PPO objectives)"                                                     | Same $A_t$/$\hat{A}_t$ disambiguation at thefactors juxtaposition (L12 vs L13).                                                                                               |
| §8 review `review.md` (2×)                 | $\pi_{\mathrm{old}}$                                                                           | $\pi_{\text{old}}$                                                                                                               | Align to §8.2/§8.4.                                                                                                                                                           |
| §8 review `review.md`                      | $L^{\mathrm{CLIP}}$, $\mathrm{clip}$, $L^{\mathrm{PPO}}$, $L^{\mathrm{VF}}$                    | $L^{\text{CLIP}}$, $\text{clip}$, $L^{\text{PPO}}$, $L^{\text{VF}}$ (+ "$A_t$ is $\hat{A}_t$" gloss)                             | Main text uses $\text{}$ throughout; review even mixed $\mathrm{clip}$ (L26) with $\text{clip}$ (L28) on one page.                                                            |
| §8.3 `gae-reward-model.md`                 | $\hat{A}_t^{\text{GAE}}$ vs $A_t^{\text{TD}}$, $A_t^{\text{MC}}$, PPO $A_t$                    | added "(same estimate written without the hat as $A_t$ …)"                                                                       | Documents the hat/no-hat convention instead of renaming ~20 occurrences.                                                                                                      |
| §9.1 `ddpg.md`                             | $\tau$ (soft update) undefined vs notation $\tau$ = trajectory                                 | added "(Here $\tau$ is the soft-update (Polyak) coefficient, not a trajectory)"                                                  | Rule 2 note at introduction.                                                                                                                                                  |
| §9.1 `ddpg.md` (OU box)                    | $x_{t+1} = x_t + \vartheta(\mu - x_t) + \sigma\epsilon_t$                                      | added "(Here $\mu$, $\sigma$ are the OU process's own parameters — unrelated to policy mean $\mu_\theta$ / std $\sigma_\theta$)" | Same chapter reuses $\mu_\theta, \sigma_\theta$ for policy heads; OU $\mu/\sigma$ are different objects.                                                                      |
| §9.2 `td3-sac.md`                          | $\mu_0 = 0$ (Gaussian mean value)                                                              | $\mu = 0$                                                                                                                        | Subscript $0$ undefined and misleading (not time 0); same fix in §9 review exercise. Meaning (mean is 0) unchanged.                                                           |
| §9 review `review.md`                      | SAC bullet with $\mu_\theta(s)$ mean head                                                      | gloss "(here $\mu_\theta$ is the Gaussian mean head, not the §9.1 deterministic policy)"                                         | Same symbol, related-but-distinct objects on one page; main text (§9.2 L195) already bridges them, review did not.                                                            |
| §10 review `review.md`                     | constrained objective with bare $D(\pi\_\theta \\                                              | \pi\_\beta)$                                                                                                                     | gloss "($D$ is any divergence, e.g. KL or MMD; see §10.1)"                                                                                                                    | Bare $D$ is not on the notation page ($D_{\mathrm{KL}}$ / $\mathcal{D}$ are); §10.1 defines it as generic, review did not. Deliberately not $D_{\mathrm{KL}}$ (would narrow meaning: MMD allowed). |
| §10.1 `offline-data-distribution-shift.md` | second $\xi$ (Lagrangian target gap ≈ 5.0)                                                     | gloss "(a different $\xi$ from the BCQ perturbation bound in §2.1)"                                                              | Same page uses $\xi$ for two objects (§2.1 perturbation limit vs §2.2 gap target).                                                                                            |
| §10 lab `lab.md`                           | ratio $\rho = \pi_e/\pi_b$                                                                     | gloss "($\pi_b$ is the behaviour policy $\pi_\beta$ of §10.1; $\pi_e$ the evaluation target)"                                    | Bridges lab's $\pi_b$ to §10.1/§10.3's $\pi_\beta$ / $\mu$ (the latter equated explicitly in `experiments.md` L136).                                                          |
| §10.3 `experiments.md`                     | DR estimator with $r_t, \hat{V}(s_{t+1}), \hat{Q}(s_t,a_t)$ inside $\sum_i$                    | $r_{i,t}, \hat{V}(s_{i,t+1}), \hat{Q}(s_{i,t},a_{i,t})$                                                                          | PDIS equation directly above uses $r_{i,t}$; DR dropped the trajectory index for the same object.                                                                             |
| §10.2 `sequence-modeling.md`               | 1-based $\tau = (s_1, a_1, r_1, \dots)$                                                        | added "(this chapter numbers steps from 1, so $s_1$ is the initial state)"                                                       | Whole chapter (§10.2–10.3, review Ex. 1, DR $\hat{V}(s_1)$, $\rho_{1:t}$) is self-consistently 1-based vs the book's 0-based; note instead of ~15 renumbering edits.          |
| §11.3–4 `meta-rl.md`                       | $D_{\text{KL}}$ (PEARL)                                                                        | $D_{\mathrm{KL}}$                                                                                                                | Aligns with §11 review (already $\mathrm{KL}$) and notation page.                                                                                                             |
| §11.3–4 `meta-rl.md` + §11 review          | PEARL KL weight $\beta$ vs DAgger mixing $\beta$ (§11.1)                                       | gloss "(unrelated to the DAgger expert-mixing $\beta$)" in both                                                                  | Same chapter, two meanings; notation-page $\beta$ list does not cover mixing. Code variable `beta` untouched.                                                                 |
| §12.3 `hierarchical.md`                    | $\mu$ in $Q^\mu(s,\omega)$ never defined                                                       | added bullet "$\mu$: the policy over options"                                                                                    | Rule 3: only symbol in the Options block ($s, \omega, T, s_T$ all glossed) with no definition. Standard Sutton meaning.                                                       |
| §12.3 `hierarchical.md`                    | $\beta_\omega(s)$ termination                                                                  | gloss "(unrelated to the exploration weight $\beta$ in §12.1)"                                                                   | Second $\beta$ meaning in the chapter.                                                                                                                                        |
| §12.3 `hierarchical.md`                    | high-level "every $c$ steps"                                                                   | gloss "(here $c$ is the high-level decision interval, not the gradient-clipping threshold)"                                      | Bare $c$ vs notation-page $c$; first in-chapter $c$ is §12.1's UCB constant (also locally defined).                                                                           |
| §12.2 `marl.md`                            | $r_t(\theta) = \pi_\theta / \pi_{\theta_{\text{old}}}$                                         | $\pi_{\text{old}}$ (+ "$A_t$ is the estimated advantage (see §8.3)")                                                             | Aligns MAPPO ratio with Ch. 8; $A_t$ had no in-chapter definition.                                                                                                            |

How verified: re-derived the §7.2 worked example ($G_t = 4.9010$, $\delta = 1.97$),
§7.1 examples ($5.51$, $3.9$; $5.52$, $2.8$), review telescoping proof, and
$n$-step/GAE endpoint ($A^{(3)}_0 = G_0 - V$) by hand plus
`.venv/bin/python` recomputation (`5.51 3.9 4.90099501 1.97 5.52` — matches
text); re-grepped each chapter for leftover variants ($R_{t+1}$, $S_t$,
$\pi_{old}$, $D_{\text{KL}}$, $\rho_0$, $\mu_0$, $\pi_{\theta_{\text{old}}}$ —
zero hits); `npx prettier --write` on all 22 files; `npx vitepress build docs`
succeeds (no dead links).

## Other changes

- Prettier reflow only (no content change) in `actor-critic.md` comparison
  table (column widths after the $r_t$ substitution); all other files
  prettier-unchanged.
- No sidebar, cross-reference, number, or code changes.

## Needs external verification

- None. All edits are internal-consistency alignments or disambiguations of
  locally defined symbols; no paper/year/venue/benchmark claims touched. Two
  judgments rest on standard-knowledge (not external sources) and are flagged
  for the coordinator: (a) $Q^\mu$'s $\mu$ = "policy over options" (Sutton
  options framework); (b) $\mu_0 = 0$ read as "Gaussian mean is 0" rather than
  a time-0 or dimension-0 subscript (no temporal/dimensional context supports
  the latter).

## Sidebar additions

- None (no pages added).

## Commands run and results

- `npx prettier --write <22 changed files>` — all pass (1 table reflow, rest unchanged).
- `npx vitepress build docs` — succeeds in ~18 s, no dead links.
- `.venv/bin/python -c` recomputation of §7.1/§7.2 numerics — all match text.
- Note: `git status`/`git log` via shell are rejected in this environment
  (only bare commands allowed, no pipes/chaining), so change review was done
  through file reads and targeted greps instead. `node`/pytest tasks: none
  apply (docs-only change; code untouched).

## Open issues

1. Cross-chapter return-to-go: Ch. 10 uses $\hat{R}_t$ (undiscounted,
   1-based), Ch. 11 §11.3 uses $R_t$ (and $R^*$ target) for the same object.
   Both locally defined; notation page D.5 lists neither. Suggest coordinator
   add $\hat{R}_t$ return-to-go to `notation.md` (out of scope for this task)
   and unify Ch. 11 to it.
2. Clipping bound $\varepsilon$ (Ch. 8, PPO) vs $\epsilon$ (Ch. 12 MAPPO
   recap): both are listed notation-page variants; left as-is.
3. Total-reward superscript order differs across chapters ($r_t^{\text{total}}$
   in §8.6 vs $r^{\text{total}}_t$ in §12.1/§12.4); each chapter internally
   consistent; left as-is.
4. §12 review Ex. 5(b) wording ("PPO does not clip updates that move a bad
   action's probability down") is imprecise as a general claim — true of the
   worked $r = 1.5$ recovery case, but same-direction $r < 1-\varepsilon$
   moves _are_ clipped (gradient zero). Left untouched as beyond notation
   scope; suggest a wording fix ("does not clip recovery moves back toward
   the trust interval").
5. Accepted local overloads (defined at every use, no change): $\mu/\sigma$
   for the §9.3 ensemble dynamics Gaussian and $\mu(u\mid s)$ Gaussian density
   in §9.2 vs policy heads; $u(s,a)$ uncertainty (§9.3) vs $u$ pre-action
   sample (§9.2); $\alpha$ as CQL weight / TD3+BC scalar vs learning rate
   (both chapter-defined); $\lambda$ as GAIL entropy weight / pessimistic-reward
   weight / TD3+BC adaptive weight vs GAE decay (all chapter-defined);
   $L_2^\tau$ expectile level vs trajectory $\tau$; $G_i$ trajectory return vs
   $G_t$; $\tau_1$ trajectory index; $d_{\text{expert}}$, $d_{\pi_\theta}$
   lowercase state distributions (notation-page $d^\pi$ analogues).
