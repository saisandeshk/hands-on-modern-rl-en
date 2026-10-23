# Review of nt-1

## Errors

1. `docs/chapter04_classical_methods/lab.md` — "Final max $\|Q - Q^\star\|$" — The star-to-asterisk cleanup is incomplete: the Experiment 2 table header and the sentence below it still use `Q^\star`, while the same page and chapter now use `Q^*`. The reported bulk replacement is incomplete; two occurrences remain. — Suggested fix: replace the remaining `Q^\star` with `Q^*`.

2. `docs/chapter05_dqn/from-q-to-dqn.md` — "shorthand for the per-step reward $r_t$ used in Section 5.2" — Section 5.2 is `dqn-components.md`, and it writes the reward as bare $r$ throughout (the TD target table and the loss). `$r_t$` does not occur there; the `$r_t$` notation appears later in the n-step target of Section 5.4. — Suggested fix: remove the cross-reference or point to the n-step return in Section 5.4.

3. `docs/chapter03_value_functions/value-q.md` — `print(\"\\nOne greedy policy learned:\")` — The backslashes are literal inside the fenced code block, so the snippet is invalid Python and raises `SyntaxError: unexpected character after line continuation character`. Copying the displayed code cannot reproduce the shown output. After replacing it with `print("\nOne greedy policy learned:")`, the code runs and its Q-values match the page. — Suggested fix: remove the backslashes.

4. `docs/chapter03_value_functions/value-q.md` — "Chapter 4 will tackle this exact issue ... The answer is Deep Q-Networks (DQN)" — DQN is Chapter 5; Chapter 4 is classical methods (DP, MC, TD). — Suggested fix: "Chapter 5".

5. `docs/chapter03_value_functions/value-q.md` — "the shortest path along the top row, one cell above the cliff" — The 13-step safe path is one step up, eleven steps across, and one step down, i.e. along the row directly above the cliff (the third row), not the top row. The lab's own CliffWalking map also places Q-learning's path on row 2. — Suggested fix: "along the row directly above the cliff".

6. `docs/chapter04_classical_methods/dp-mc-td.md` — "stand in for the unknown true $v_\pi$" — The revision changed the earlier occurrence to $V^\pi$ but left this one, so the same page now uses $v_\pi$ and $V^\pi$ for the same object, violating the chapter's one-symbol rule. — Suggested fix: use $V^\pi$ here too.

7. `docs/chapter02_rl_basics/policy-value.md` — "This page writes the reward after step $t$ as $R_{t+1}$" — The Section Summary on the same page writes the return as $\sum \gamma^k r$ (lowercase), and the concrete trajectory uses $r_1,r_2,r_3$; if uppercase $R$ is the page's convention for the random reward, the summary is inconsistent with it. The defining note at line 153 also comes after the first use in the formula at line 139. — Suggested fix: use $R$ consistently in the summary and move the convention note above line 139, or state explicitly that lowercase marks realized values.

8. `docs/chapter01_cartpole/review.md` — "single-digit value losses late in training (for example 0.00014 at iteration 40)" — $0.00014 = 1.4\times10^{-4}$, which is not a single-digit number; it is four orders of magnitude below one. — Suggested fix: "very small value losses" or "value losses of order $10^{-4}$".

9. `docs/chapter01_cartpole/review.md` — "each episode now spans the full 2048-step rollout, $2048 / 500 \approx 4$" — A completed episode is 500 steps; four such episodes occupy about 2000 of the rollout's 2048 steps. The episode does not span the rollout; the rollout contains about four episodes. — Suggested fix: "the rollout contains about four full 500-step episodes".

10. `docs/chapter02_rl_basics/bandit.md` — "Here $\delta$ and $\epsilon$ are the PAC accuracy parameters" — In the standard PAC statement "with probability at least $1-\delta$, return an action within $\epsilon$ of the optimum", $\epsilon$ is the accuracy tolerance and $\delta$ is the confidence/failure probability. Calling both "accuracy parameters" is incorrect. — Suggested fix: "$\delta$ is the confidence parameter and $\epsilon$ the accuracy tolerance".

## Doubts

1. `docs/chapter02_rl_basics/review.md` — "Optimal policy: $\pi^* = \arg\max_\pi \mathbb{E}_\pi[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)]$" — The revision changed the same formula in §2.3 to $R_{t+1}$, but the chapter summary still uses $R(s_t,a_t)$. Both forms are valid when $R(s,a)$ is the expected reward, but the chapter now uses two forms, contradicting the stated one-symbol-per-object goal. — Suggested fix: align this formula with §2.3, or revert §2.3.

2. `docs/chapter05_dqn/dqn-family.md` — "This PER $\beta$ is distinct from the KL-penalty $\beta$ in later LLM chapters" — The same page also uses $\beta$ for the intrinsic-reward weight (line 338: "where $\beta$ controls the weight of the intrinsic reward"), which the new note does not mention; the page still overloads $\beta$. — Suggested fix: extend the note to cover the intrinsic-reward $\beta$ as well.

3. `docs/chapter04_classical_methods/lab.md` — "the two estimate slightly different quantities on continuing revisits" — First-visit and every-visit MC both estimate $V^\pi$ in an episodic task; they differ in which returns are averaged, not in the estimand (every-visit is biased only for continuing tasks). — Suggested fix: say they weight revisits differently and that every-visit uses more samples.

4. `docs/chapter01_cartpole/review.md` — "and $r_i$ in the next line likewise abbreviates $r_i(\theta)$" — The next line is the value-loss bullet; $r_i$ appears in the clip-fraction bullet two lines below. — Suggested fix: "in the clip-fraction line below".

5. `docs/preface/brief-history/index.md` — "maximizes the expected long-term discounted cumulative reward: $G_t = \sum...$" — $G_t$ is a realized return, not its expectation, so the displayed equation is not the expectation the sentence describes. — Suggested fix: write $\max_\pi \mathbb{E}_\pi[G_0]$ (or $\mathbb{E}[\sum_k \gamma^k r_{t+k+1}]$) in the display.

## Verdict

The revision is mostly a careful notation pass, and the numerical material I checked survives scrutiny: I reproduced the lab's DP oracle exactly (40 uniform-policy sweeps, 617 policy-iteration sweeps with 2 improvements, 431 value-iteration sweeps, $V^*(s_0)=0.823506/0.823512$, and the $1.12\times10^{-5}$ max-norm gap), and the bandit probabilities, decay values, UCB scores and regret arithmetic also check out. The most important problems are the false Section 5.2 cross-reference in `from-q-to-dqn.md`, the non-runnable Q-learning snippet in `value-q.md`, and the two factual errors in that page's CliffWalking/DQN paragraph (Chapter 4 instead of Chapter 5; "top row" instead of the row above the cliff). The cleanup also left `Q^\star` in `lab.md` and $v_\pi$ in `dp-mc-td.md`, and the policy-value page's note and its summary/trajectory still use different reward notation. These are local fixes, but they show that the reported verification was not complete.
