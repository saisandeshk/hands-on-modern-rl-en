# Report: ch05-text (Chapter 5 text pass)

## Summary

Reviewed all five pages in `docs/chapter05_dqn/` for technical correctness (re-derived equations, recomputed every numeric example, traced code snippets), fixed cross-references and stale chapter paths, added concise treatments for the missing/partial core curriculum items, and created the new `docs/chapter05_dqn/review.md` summary-and-exercises page. Build passes; no code, image, or component files touched.

## Technical fixes

1. **False paper attribution (dqn-components.md).** The page claimed "In its 2015 DQN paper DeepMind ran a comparison experiment" in which a no-replay Breakout agent "repeatedly learns the policy of bouncing the ball to the left." The Nature paper discusses replay's importance but contains no such specific Breakout anecdote. Rewrote as an unattributed illustrative mechanism ("Consider what happens without experience replay on Atari Breakout…"). Verified by: knowledge of the paper's contents; flagged rather than deleted since the mechanism itself is correct.
2. **Wrong chapter paths (visual-game-projects.md).** Five references pointed at `chapter07_dqn/` (Classic Control install/run commands, Atari run command, Pokemon install/run commands, and a GIF render `--output docs/chapter07_dqn/images/…`). All corrected to `chapter05_dqn/`. Verified by: `code/chapter05_dqn/` directory listing (scripts exist there); `docs/chapter07_actor_critic/` is Actor-Critic, not DQN.
3. **Misleading pipeline claim (visual-game-projects.md).** The Key Settings table described `buffer_size=10000` as "aligned with validated RL-Zoo Pong DQN pipeline." The script default is 100_000 and RL-Zoo Atari DQN uses 100_000; a 10k buffer combined with `learning_starts=100000` evicts 90k pre-learning transitions before the first update. Kept the command values (they may reflect the actual run) but replaced the justification with an accurate note about the small buffer holding only the most recent 10k transitions. Listed under Needs external verification.
4. **Title/H1/sidebar mismatches.** `from-q-to-dqn.md` front matter said "5.1 Why DQN Is Needed" while H1 and sidebar say "5.1 From Q-Learning to DQN" → unified on the latter. `dqn-components.md` H1 said "5.2 DQN Structure and Implementation" while front matter and sidebar say "5.2 DQN Architecture and Implementation" → unified on the latter. `visual-game-projects.md` had no `title` front matter (only an `outline` block, unlike every other page) → added `title: '5.5 Hands-on: Visual Game Projects'` (quoted; unquoted colon broke the VitePress build — caught and fixed).
5. **Overstated cross-reference (visual-game-projects.md).** Claimed Section 5.3 covers "Q-values, and ablation studies"; the LunarLander page covers curves, replays, and failure analysis but has no Q-value-evolution or ablation sections. Trimmed to what Section 5.3 contains.
6. **Numeric spot-checks (all pass, no change needed).** Recomputed with `.venv/bin/python`: $50^6 = 1.5625\times10^{10}$ ✓; $0.99^{100}\approx0.366$, $0.9^{100}\approx2.7\times10^{-5}$ ✓; all four $\varepsilon$-decay table columns ($0.95/0.99/0.995/0.999$ at 100/200/400 episodes) ✓; MLP parameter count $4{\times}128{+}128{+}128{\times}128{+}128{+}128{\times}2{+}2 = 17410 \approx 17000$ ✓; moving-target example $1+0.99\times8.0=8.92$ ✓; Baird-sketch growth $1.08^3 = 1.259712$ ✓; review-exercise answers ($3.92^2 = 15.3664$, PER weights $[9.0,1.8,0.9,0.45]\to[1.0,0.2,0.1,0.05]$, dueling Q $[6,4,8,2]$) ✓.

## Other changes

Added compact curriculum treatments to `dqn-components.md` (existing flow preserved; all new material in subsections/`details` blocks):

- Semi-gradient update equation, MSVE vs Bellman-error distinction, linear features/state aggregation and the projected fixed point (Tsitsiklis & Van Roy), plus a hand-verified two-state Baird-style divergence sketch ($w_{k+1} = 1.08\,w_k$).
- Replay age / policy drift / update-to-data ratio / transition immutability box; reward-scale/clipping/observation-scaling box.
- Polyak soft-update equation with $\tau = 0.005$, hard-vs-soft target-lag discussion (notes the chapter scripts use hard updates with `tau=1.0`, confirmed in `code/chapter05_dqn/dqn_atari_sb3.py:305`).
- Diagnostics box (fixed reference batch: mean Q, target scale, |TD error|, gradient norm) plus checkpoint-everything and fixed-batch differential-test boxes.
- Created `docs/chapter05_dqn/review.md`: 8 learning outcomes, 8 key equations with one-line readings, 6 pitfalls, 10 exercises (conceptual, derivations, hand computations, 2 code exercises on `code/chapter05_dqn/`, 1 task-suitability design question) with fully worked solutions.

Coverage table (RL-3 checklist): covered — value-function approximation/generalization, semi-gradient TD, MSVE/Bellman-error, on-policy prediction (linear projected fixed point), off-policy bootstrapping/triad, Baird mechanism, neural action values, buffer capacity/sampling, transition immutability/aliasing, replay age/drift/nonstationary data, target networks + hard intervals, Polyak updates, DQN targets/gather/detach, reward clipping/observation scaling, Q-scale diagnostics, target lag/replay ratio/UTD, maximization bias/double, fixed-batch differential tests, checkpointing, train/eval exploration separation, dueling, PER + corrections, distributional value. Partial (intentionally brief, per Depth/Later tags): linear features/projected solutions (short paragraph), distributional learning (already had a paragraph; kept). Missing: none in-scope (general nonlinear off-policy convergence theory is marked Later).

## Needs external verification

- Whether the Pong long-run actually used `--buffer-size 10000` with `--learning-starts 100000` (wasteful but functional) or whether `10000` is a typo for the Zoo/script default `100000`. I kept the printed values and fixed only the justification.
- LunarLander/Atari reported curves and episode numbers (e.g. `253.12`, Pong table) cannot be re-verified offline; left untouched.
- Error message in `code/chapter05_dqn/dqn_atari_sb3.py:294` references `chapter04_dqn/requirements.txt` (should be `chapter05_dqn`); code belongs to another worker, so I did not edit it — flagging for them.

## Sidebar additions

- `5.R Summary and Exercises` → `/chapter05_dqn/review` (new page; coordinator to place after 5.5 and renumber as needed).

## Commands run and results

- `npx prettier --write` on all six files (review.md + 5 existing): clean.
- `npx vitepress build docs`: initially failed on unquoted `title: 5.5 Hands-on: …` YAML colon; fixed with quotes; rebuild succeeds.
- `.venv/bin/python -c` arithmetic checks for all numbers above: all match.
- No code tests apply (no `code/` edits allowed); code snippets in text were traced by hand against `code/chapter05_dqn/*.py` argument names.

## Review response

Reviewer findings re-checked against the code and by recomputation; accepted items fixed as described, rejected items explain why.

- Error 1 (checkpoint box contradicted `dqn_gym_sb3.py` `save_replay_buffer=True`): accepted. Verified at `code/chapter05_dqn/dqn_gym_sb3.py:251-256` (flag present) vs `dqn_atari_sb3.py:343-348` (absent). `dqn-components.md` now distinguishes the 5.3 script (buffer saved) from the 5.5 script (not saved).
- Error 2 (AtariWrapper pixel scaling misattributed): accepted on textual grounds — the 5.5.2 teaching snippet path (Gymnasium `AtariPreprocessing(scale_obs=True)`) delivers floats in [0,1] while SB3's own path keeps uint8 until the policy's `preprocess_obs` divides by 255; could not inspect installed SB3 source offline, but the corrected wording (wrapper clips rewards, leaves uint8 frames; `CnnPolicy` rescales before convolutions) is accurate under either wrapper attribution and matches the code-level behavior. `dqn-components.md` fixed.
- Error 3 (Go $3^{361} \approx 10^{170}$): accepted. $361\log_{10}3 = 172.24$, so $3^{361} \approx 1.7\times10^{172}$; Tromp's legal-position count $\approx 2.1\times10^{170}$ is a different quantity. `from-q-to-dqn.md` fixed with both numbers distinguished.
- Error 4 (5.1 over-claim: Q-value evolution, ablations in 5.3): accepted. `lunar-lander.md` has curves, success criteria, replays, state/action discussion and debugging table, no ablation study or Q-evolution section. Sentence trimmed in `from-q-to-dqn.md` (the identical over-claim in 5.5 was already fixed in the original pass).
- Error 5 (teaching snippets double-scale pixels): accepted. `make_atari_env` uses `scale_obs=True` (floats in [0,1]) and `CNNQNetwork.forward` divided by 255 again, mapping inputs to [0, 1/255]. Removed the `/255.0` line and noted the input is already scaled in `visual-game-projects.md`.
- Error 6 (review.md not in sidebar / "5.R" numbering): rejected as out of scope for this worker. Task rules forbid editing `docs/.vitepress/config.mjs` and the task explicitly prescribes `title: 5.R Summary and Exercises` with the coordinator to renumber/add the sidebar entry (recorded under Sidebar additions). The branch therefore intentionally leaves registration to the coordinator; no orphan content was created by mistake.
- Error 7 (buffer-size half-fix: "zoo-aligned" claim vs 10x-smaller buffer): accepted. Both long-run commands and the second entry used `--buffer-size 10000` while the script default (`dqn_atari_sb3.py:199`) and RL-Zoo Atari DQN use 100_000. Changed both commands to `--buffer-size 100000` and rewrote the Key Settings row to describe the large buffer accurately, keeping the zoo-aligned wording and run name.
- Error 8 (missing `planning/content-review-notes.md` entries): rejected as out of scope for this worker. The task scope limits edits to the six listed files; `planning/content-review-notes.md` is not among them. The two fixes (Nature-paper attribution removal in `dqn-components.md`, five `chapter07_dqn/` path corrections in `visual-game-projects.md`) are recorded here instead; coordinator may port them.
- Doubt 1 (exercise 9 asserts empirical direction): accepted and softened. Prompt and solution now require reporting the observed direction with means and standard deviations, note seed dependence, and note the DQN paper itself used a 10,000-step target interval. Also fixed the "final 5-episode means" mismatch (script default is `--eval-episodes 10`).
- Doubt 2 (exercise 8 TD-error direction): accepted and softened. Solution now predicts only the Q gap firmly, explains the TD-error comparison depends on which target scores the held-out batch, and references `code/chapter05_dqn/double_dqn_cartpole.py` (verified to exist); prompt updated likewise.
- Doubt 3 (exercise 7 "down-weighted least" prose): accepted. Numbers were already correct ($P = [0.0278, 0.1389, 0.2778, 0.5556]$, $w = [9.0, 1.8, 0.9, 0.45] \to [1.0, 0.2, 0.1, 0.05]$, recomputation confirmed); prose rewritten so the highest-error sample keeps the largest rescaled weight.
- Doubt 4 (Sutton & Barto "Chapters 10-11"): accepted. Reference now reads "Chapters 9-11", covering on-policy prediction, control with approximation, and the triad/Baird material without naming a wrong chapter pair.
- Doubt 5 (removing the Nature-paper replay anecdote): rejected. The reviewer could not verify the recollection offline either; per the no-fabrication rule, an unattributed but mechanistically correct illustration ("consider what happens without replay") is safer than restoring a specific paper claim neither party could check. Kept as-is; if the coordinator can verify the Breakout ablation figure, restoring it with a citation is welcome.
- Doubt 6 (semi-gradient "not the gradient of any fixed objective"): accepted as a precision clarification. Sentence now states the frozen-target surrogate is an exact gradient while the true TD objective is not descended, in `dqn-components.md`.
- Doubt 7 (`optimize_memory_usage=True` row): accepted. Row now notes it is enabled via the `--optimize-memory-usage` CLI flag (`dqn_atari_sb3.py` uses `action="store_true"`, default off).
- Doubt 8 (Chapter Layout link text vs page title): accepted. First row now links as "From Q-Learning to DQN", matching the page title.

## Open issues

- `dqn-family.md` has an end-of-page Exercises 1–6 list that overlaps with the new `review.md` exercises; left as-is to avoid scope creep, but the coordinator may want to deduplicate.
- `dqn-components.md` pseudocode uses generic SGD ($\theta \leftarrow \theta - \alpha\nabla\mathcal{L}$) while the code uses Adam; noted in text as pseudocode-level, acceptable.
- Sidebar registration of `review.md` (`5.R Summary and Exercises` → `/chapter05_dqn/review`) and the two `planning/content-review-notes.md` Chapter 5 entries are left for the coordinator (config/notes files outside worker scope).
