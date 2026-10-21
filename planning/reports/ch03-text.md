# Report: ch03-text

## Summary

Reviewed all three Chapter 3 pages end to end, recomputed every numeric example, and ran the chapter code. Fixed one wrong printed number, two missing discount assumptions, one wrong CliffWalking step count, and one stale Q-learning run-output block that contradicted both the repo script and the page's own SVG figure. Added the two missing in-scope core treatments (advantages; Bellman operators, contraction, and fixed points) plus small tie-handling/terminal-convention and stopping-criterion notes. Created `docs/chapter03_value_functions/review.md` (3.R Summary and Exercises) with 7 outcomes, 10 key equations, 6 pitfalls, and 8 exercises with fully worked, numerically checked solutions. Build passes; chapter tests pass.

## Technical fixes

- [x] (none recorded — no pre-listed suspected issues for this chapter.)
- `value-bellman.md`: 50-iteration slot-machine output `1.998698` was wrong. Wrong claim: after 50 updates of `V = 0.2 + 0.9V` the value is `1.998698`. Correction: `1.989692`. Verified by running the exact snippet (`.venv/bin/python -c` loop): final `V = 1.9896924495853598`, i.e. `2*(1-0.9^50)`.
- `value-bellman.md` treasure-corridor box: rewards `-1` per step with values `[-4,-3,-2,-1,0]` implicitly assumed no discounting but never said so. Correction: added "For simplicity this example uses $\gamma=1$". Verified arithmetically: with $\gamma=1$ each step is $-1 + V(s')$; with $\gamma<1$ the table would differ.
- `value-bellman.md` two-step $V^\pi$/$V^*$ example ($1+5=6$, $0+2=2$, $1+7=8$, $0+4=4$, $V^\pi=3.6$, $V^*=8$): formulas in the chapter carry $\gamma$ but the arithmetic dropped it. Correction: stated $\gamma=1$ for the example. Verified: with $\gamma=1$ all six equalities hold exactly.
- `value-q.md` CliffWalking: "shortest path ... takes 11 steps" was wrong. Correction: shortest safe path is 13 steps (up 1, across 11, down 1); 11 is the Manhattan distance of the直-though-the-cliff route. Verified from grid geometry (4x12, cliff on bottom row between S and G) and Sutton & Barto Example 6.6.
- `value-q.md` 4x4 Q-learning "Run output" block was stale: Q-values `{-5.1,-4.6,-4.6,-5.2}` and policy grid with `↑` arrows in the left column contradicted (a) the page's own `q-learning-learned-policy.svg` (which shows `→ ↓ → ↓ / ↓ → → ↓ / → → → ↓ / → → → G`) and (b) a fresh run of the printed snippet (`[-4.797,-4.686,-4.686,-4.919]`, optimal arrows, 6-step return $-(1-0.9^6)/0.1 \approx -4.686$). Correction: replaced numbers with the verified run rounded to one decimal (`-4.8/-4.7/-4.7/-4.9`), replaced the grid with the SVG-matching optimal pattern, and noted decimals vary by NumPy version while ordering/ties are stable. Also noted the optimal wall-bump value $\approx -5.22$ for context.
- `value-q.md` Exercise 2 ("+10 on goal, 0 elsewhere — still prefer shortest?") was underspecified: answer depends on $\gamma$. Correction: fixed $\gamma=0.9$ for the question and added a $\gamma=1$ follow-up.
- `value-bellman.md` front matter `title: 3.1 Value Functions and Bellman Equations` did not match the H1/sidebar `3.1 State Values and Bellman Expectation`. Correction: aligned front matter to H1/sidebar.
- `value-q.md` had no `title` front matter (only `outline`). Correction: added `title: 3.2 Action Values and Bellman Optimality`.
- `value-experiment.md` had no front matter at all. Correction: added quoted `title: "3.3 Hands-on: Value Iteration and Q-Learning"` (quotes required because of the colon; unquoted broke the VitePress build).
- `value-bellman.md` matrix solution stated invertibility without conditions. Correction: added that $(\boldsymbol{I}-\gamma\boldsymbol{P}_\pi)$ is invertible when $\gamma<1$ (stochastic-matrix eigenvalue argument).
- `value-experiment.md` convergence sentence omitted the tolerance used by the script. Correction: noted max change below the $10^{-12}$ tolerance at sweep 7.

## Other changes

- Added `### Bellman Operators, Contraction, and Fixed Points` to `value-bellman.md` (~250 words): definitions of $T^\pi$, $T^*$, $\gamma$-contraction in max norm, Banach fixed-point consequence for iterative policy evaluation/value iteration, residual bound $\theta/(1-\gamma)$, policy-vs-value convergence note, terminal $V=0$ convention.
- Added `## Advantage: How Much Better Than Average?` to `value-q.md` (~200 words): $A^\pi=Q^\pi-V^\pi$, zero-mean proof sketch, numeric check on the $10/4$ example, greedy-invariance, pointer to Chapter 7.
- Added one sentence on `argmax` ties and unreachable states in `value-q.md`, and one short paragraph tying the new Q numbers to optimal values and tie-breaking.
- Created `docs/chapter03_value_functions/review.md` per spec (outcomes, equations, pitfalls, 8 exercises with `::: details Solution` blocks; solutions hand-checked, see below).
- Verified `value-experiment.md` table ($V^*(0,0)=0.728537$, 7 sweeps, greedy 6.0 steps/100% success) against a fresh script run; no change needed.

## Needs external verification

None. Paper attributions touched (Watkins & Dayan 1992; Sutton & Barto Fig. 4.1/Example 6.6; Bellman 1957) are well-known facts and were left as written. No new papers, numbers, URLs, or quotes were introduced.

## Sidebar additions

- Title: `3.R Summary and Exercises`, link: `/chapter03_value_functions/review` (new file `docs/chapter03_value_functions/review.md`; coordinator to renumber/place).

## Commands run and results

- `.venv/bin/python -m pytest code/chapter03_value_functions/test_gridworld_q_learning.py -v` — 5 passed, 8 subtests passed.
- `.venv/bin/python code/chapter03_value_functions/gridworld_q_learning.py --output-dir /tmp/opencode/value-test` — converged after 7 sweeps, $V^*(0{,}0)=0.728537$, all three schedules greedy 6.0 steps/100% success; matches text table.
- `.venv/bin/python -c` recomputations — slot-machine 50-iter `1.989692`; 3-state MDP solution `[8.714, 6.818, 9.214]` (code-only sanity check); value-q snippet rerun `[-4.797,-4.686,-4.686,-4.919]` with optimal arrow grid matching the SVG.
- `npx prettier --write` on the four touched docs — clean.
- `npx vitepress build docs` — first attempt failed on unquoted YAML colon in the new experiment title; after quoting, build completed successfully.

## Review response

- Error 1 (value-q wall-bump "matches" claim): accepted. Recomputed $V^*(0,0)=-4.68559$ and $Q^*=-5.21703$ for wall bumps; rewrote the paragraph to say the learned ordering is right/down above up/left, the two inside-grid moves match the exact optimum, and the wall-bump entries ($-4.80$/$-4.92$) are over-estimates of $-5.22$ from incomplete convergence. Also clarified review.md Exercise 6 to distinguish learned versus exact wall-bump values.
- Error 2 (tie-break toward lower index): accepted. Rewrote to say `argmax` returns the first maximum but entries equal to $\sim 10^{-15}$ flip on floating-point noise, so the arrow pattern at tied cells is not a fixed rule.
- Error 3 (run-output decimals / NumPy 2 repr): accepted. Replaced one-decimal values with the three-decimal `.round(3)` output (`-4.797`/`-4.686`/`-4.686`/`-4.919`) and noted NumPy 2 prints the same digits wrapped in `np.float64(...)`.
- Error 4 (Advantage H2 orphans the $Q^\pi$ definition): accepted. Moved the whole Advantage section to after the "Mathematical definition" block, just before "The Bellman Optimality Equation".
- Error 5 (stochastic-matrix eigenvalues "inside the unit disk"): accepted. Changed to $\rho(\boldsymbol{P}_\pi)=1$ so $\rho(\gamma\boldsymbol{P}_\pi)\le\gamma<1$ for $\gamma<1$; verified the conclusion (invertibility) is unchanged.
- Error 6 (review.md missing from sidebar / "3.R" number): rejected as out of scope for this worker. The task spec mandates `title: 3.R Summary and Exercises` and states the coordinator will renumber/place the page, and worker rules forbid editing `docs/.vitepress/config.mjs`. Sidebar placement and renumbering are left to the coordinator (recommended: `3.4 Summary and Exercises`, link `/chapter03_value_functions/review`).
- Doubt 1 (policy-iteration early-stopping sentence): accepted. Dropped the "which is why policy iteration can stop updating values early" clause; now says the greedy policy usually settles a few sweeps before the values do.
- Doubt 2 ("beside the cliff"): accepted. Changed to "along the top row, one cell above the cliff".
- Doubt 3 (3.3 sidebar text vs page H1): no page change (page H1 is the accurate description of the experiment); coordinator decision recorded: either update the sidebar entry to `3.3 Hands-on: Value Iteration and Q-Learning` to match the page, or confirm the generic sidebar wording is intentional.
- Doubt 4 (`planning/content-review-notes.md` not updated): no change — that file is outside this task's edit scope (worker rules: only edit files in scope). Coordinator may copy the three fixed factual errors there (slot-machine `1.998698`→`1.989692`; CliffWalking 11→13 steps; stale 4x4 Q-learning run-output block).
- Doubt 5 (Exercise 8(b) terminal-reward wording): accepted. Added "the terminal state has no successor state, so there is nothing to bootstrap" to rule out reading the reward as belonging to the terminal state.

## Open issues

- Images were out of scope and left untouched. The stale-text problem is resolved in the other direction: the text grid now matches `q-learning-learned-policy.svg`.
- `value-experiment.md` H1 (`3.3 Hands-on: Value Iteration and Q-Learning`) is more specific than the sidebar entry (`3.3 Hands-on: Value Function Experiments`); left as is for the coordinator to reconcile.
- Full algorithm treatments (policy iteration, MC/TD estimators, reward shaping) remain in Chapter 4 by design; the coverage table marks them accordingly.

### Coverage against the RL-1 checklist (Chapter 3 perspective)

| Item                                                                           | Status                | Note                                                                                                         |
| ------------------------------------------------------------------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------ |
| Sequential decisions vs supervised prediction and contextual bandits           | Covered (Ch.2)        | Deferred; Ch.3 assumes it                                                                                    |
| Agent, environment and interaction boundary                                    | Covered (Ch.2)        | Deferred                                                                                                     |
| State, observation and the Markov property                                     | Covered (Ch.2)        | Used in the §3.1 Markov-step proof                                                                           |
| Action spaces, transition kernels and reward functions                         | Covered (Ch.2)        | Used, not redefined                                                                                          |
| Trajectories, episodes and continuing tasks                                    | Covered (Ch.2)        | Used                                                                                                         |
| Finite and infinite horizons                                                   | Partial               | Infinite-horizon discounting used throughout; finite-horizon backward induction deferred to Ch.4/planning    |
| Discounting, termination and truncation as separate semantics                  | Partial               | Discounting + termination ($V=0$) covered incl. terminal-entry convention; truncation-as-time-limit deferred |
| Deterministic and stochastic policies                                          | Covered (Ch.2)        | Used in expectation vs optimality contrast                                                                   |
| Returns and reward-to-go                                                       | Covered               | §3.1 G/V numeric example                                                                                     |
| State values, action values and advantages                                     | Covered (after fix)   | V/Q were covered; advantage added to `value-q.md`                                                            |
| Bellman expectation equations for policies                                     | Covered               | V and Q forms derived                                                                                        |
| Bellman optimality equations                                                   | Covered               | V and Q forms derived                                                                                        |
| Matrix form of finite policy evaluation                                        | Covered (after fix)   | Plus $\gamma<1$ invertibility note                                                                           |
| Exact policy evaluation and iterative policy evaluation                        | Covered (concept)     | Closed form + Bellman backup iteration in §3.1; full iterative-PE algorithm is Ch.4                          |
| Policy improvement and greedy policy construction                              | Covered               | Greedy from Q + tie note                                                                                     |
| Policy iteration                                                               | Deferred (Ch.4)       | Mentioned only as context                                                                                    |
| Value iteration                                                                | Covered (hands-on)    | §3.3 experiment; theory via $T^*$                                                                            |
| Finite-horizon backward induction                                              | Deferred (Ch.4)       | Not in Ch.3 scope                                                                                            |
| Generalized policy iteration                                                   | Deferred (Ch.4)       | Mentioned implicitly only                                                                                    |
| Bellman operators, contraction intuition and fixed points                      | Covered (after fix)   | New subsection                                                                                               |
| Residual-aware stopping criteria and policy vs value convergence               | Covered (after fix)   | Residual bound + early-policy note                                                                           |
| Synchronous and asynchronous update order                                      | Partial               | Synchronous covered in §3.3; asynchronous ordering deferred to Ch.4                                          |
| Tie handling, unreachable states and terminal-state conventions                | Covered (after fix)   | Tie + unreachable note; terminal $V=0$/on-entry convention                                                   |
| Reward timing, reward shaping and unintended optimal policies                  | Partial               | Timing covered in §3.3; shaping is Ch.4 `reward-design`                                                      |
| MDP formulation with state sufficiency, action authority and evaluation policy | Deferred (Ch.2/Ch.4)  | Assumed                                                                                                      |
| Belief states and finite POMDP planning (Depth)                                | Missing (intentional) | Depth topic; not added                                                                                       |
| Large state-space approximate planning (Later)                                 | Missing (intentional) | Later topic; DQN pointer only                                                                                |
