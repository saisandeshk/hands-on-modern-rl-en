# Review of ch05-text

Scope reviewed: `docs/chapter05_dqn/dqn-components.md`, `docs/chapter05_dqn/from-q-to-dqn.md`,
`docs/chapter05_dqn/visual-game-projects.md`, new `docs/chapter05_dqn/review.md`, and
`planning/reports/ch05-text.md`. All equations in the new material were re-derived, all numbers
recomputed (`.venv/bin/python`), all code claims checked against `code/chapter05_dqn/*.py` and
against the installed `stable_baselines3` 2.9.0 source, and every exercise in `review.md` worked
out independently.

## Errors

1. `docs/chapter05_dqn/dqn-components.md` — "The SB3 scripts in Sections 5.3 and 5.5 save model and
   optimizer state in `final_model.zip` and periodic checkpoints, which covers weights and optimizer
   but not the buffer or RNG streams" — false for Section 5.3.
   `code/chapter05_dqn/dqn_gym_sb3.py:251-256` builds its checkpoint callback as
   `CheckpointCallback(save_freq=…, save_path=…, name_prefix=…, save_replay_buffer=True)`, and in SB3
   2.9.0 (`common/callbacks.py:307-312`) `save_replay_buffer=True` makes the callback also write
   `replay_buffer_<steps>_steps.pkl` next to every model checkpoint. So the 5.3 script _does_
   persist the buffer with its periodic checkpoints; only the Atari script
   (`dqn_atari_sb3.py:343-348`) omits the flag. The new sentence therefore contradicts the code of
   the very page it cites.
   — Suggested fix: "…periodic checkpoints, which cover weights and optimizer (the 5.3 script also
   saves the replay buffer alongside each checkpoint; the 5.5 script does not) but not the RNG
   streams or the ε-schedule position — exact resumption needs those too."

2. `docs/chapter05_dqn/dqn-components.md` — "the SB3 `AtariWrapper` used in Section 5.5 therefore
   clips rewards to ±1 and scales pixels to $[0,1]$" — the clipping half is right, the scaling half is
   misattributed. In SB3 2.9.0 (`common/atari_wrappers.py`), `AtariWrapper` → `WarpFrame` keeps the
   frames as uint8 in [0,255] (`observation_space = spaces.Box(low=0, high=255, …)`, and
   `WarpFrame.observation` performs no division by 255). The /255 normalization happens later, in the
   policy: `common/preprocessing.py:118-119` (`preprocess_obs` → `obs.float() / 255.0` when
   `normalize_images=True`, the default), invoked from `common/policies.py:130`. The AtariWrapper
   therefore scales nothing; the [0,1] scaling in the 5.5.2 teaching snippet comes from a different
   wrapper (Gymnasium's `AtariPreprocessing(scale_obs=True)`).
   — Suggested fix: "…clips rewards to ±1 and leaves the frames as uint8 in [0,255]; SB3's `CnnPolicy`
   then rescales pixels to [0,1] before the convolutional layers."

3. `docs/chapter05_dqn/from-q-to-dqn.md` — "Go has about $3^{361} \approx 10^{170}$ positions" —
   arithmetically wrong. $3^{361} = 1.74\times10^{172}$ (since $\log_{10}3^{361} = 361\log_{10}3 =
   172.24$), so the two numbers differ by two orders of magnitude. The $\approx2.1\times10^{170}$ figure
   is Tromp's count of _legal_ Go positions, which is a different quantity from the $3^{361}$ board
   configurations; writing them as equal is the error.
   — Suggested fix: "Go has about $3^{361} \approx 1.7\times10^{172}$ board configurations (Tromp's
   count of legal positions is $\approx 2.1\times10^{170}$)."

4. `docs/chapter05_dqn/from-q-to-dqn.md` — "we move to the LunarLander hands-on section, where one
   and the same task lets us observe training curves, the evolution of Q values, ablation experiments,
   and replay evaluation" — Section 5.3 contains neither Q-value evolution nor ablations.
   `docs/chapter05_dqn/lunar-lander.md` contains 5.3.1 the training run, 5.3.2 curves, 5.3.3 success
   criteria + random baseline, 5.3.4 three replays, 5.3.5 a state/action discussion that only mentions
   Q values qualitatively ("Early in training, the network's Q-values for the 4 actions have no reliable
   meaning"), 5.3.6 a debugging table, and 5.3.7 a wrap-up; there is no ablation study anywhere on the
   page. The writer identified precisely this over-claim in Section 5.5 ("Section 5.3 placed these
   components into a LunarLander experiment: observing real training, evaluation returns, replay
   animations, and training curves") and fixed it there, but left the identical false statement in
   Section 5.1.
   — Suggested fix: delete "the evolution of Q values, ablation experiments" so that the sentence
   reads "…training curves, replay evaluation, and failure analysis".

5. `docs/chapter05_dqn/visual-game-projects.md` — the two consecutive teaching snippets double-scale
   the pixels. The env builder (lines 89-99) uses
   `gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)`, so the
   observations are already floats in [0,1]; the next snippet's `CNNQNetwork.forward` (line 128) then
   executes `x = x / 255.0`, so every pixel entering the CNN lies in [0, 1/255] ≈ [0, 0.0039] — the
   network sees an essentially constant image. (SB3's own path is correct: `WarpFrame` keeps uint8 and
   the policy divides by 255 exactly once.) This is a pre-existing bug on a page the writer owned, and
   it is invisible unless you run the two snippets together, which is exactly how the page presents them.
   — Suggested fix: remove `x = x / 255.0` from the snippet (or set `scale_obs=False` in the wrapper).

6. `docs/chapter05_dqn/review.md` — the new page is not registered in `docs/.vitepress/config.mjs`
   (the branch touches no config file), so it does not appear in the sidebar and is unreachable in the
   built site; its section number "5.R" also has no counterpart in the sidebar, where every page is
   numbered `X.Y` and no chapter has a separate summary page (see the full sidebar listing: chapters
   run 1.1 … 26.4 with no `X.R` entry). AGENTS.md is explicit: "Every navigable page must be listed
   there", and "New pages … must be added to the sidebar in `docs/.vitepress/config.mjs`"; the heading
   numbering also has to follow the sidebar (`npm run fix:headings` only knows `X.Y` entries, so it
   cannot repair this page). The writer's own report defers this to a coordinator, which leaves the
   branch in a state where a whole chapter-summary page is orphaned.
   — Suggested fix: add `{ text: '5.6 Summary and Exercises', link: '/chapter05_dqn/review' }` to the
   Chapter 5 sidebar block, set the front-matter title to `5.6 Summary and Exercises` and the H1 to
   `# 5.6 Summary and Exercises`, then run `npm run fix:headings`.

7. `docs/chapter05_dqn/visual-game-projects.md` — the buffer-size correction was left half-done, so
   the page now contradicts itself. The table row was correctly rewritten ("it holds only the most
   recent 10k transitions, so with `learning_starts=100000` most pre-learning frames are evicted
   before the first update" — true: at the first update the buffer holds steps 90k–100k), but the
   surrounding text still says the experiment is "configured along the validated pipeline" /
   "to stay consistent with the validated Pong DQN pipeline in SB3 / RL-Zoo" (lines 213, 226, 270, 323)
   and the run is still named `PongNoFrameskip-v4_dqn_seed0_10m_zoo_aligned`, while the command passes
   `--buffer-size 10000` (lines 230, 394) and RL-Zoo's Atari DQN — like the script's own default
   (`dqn_atari_sb3.py:199`, `buffer_size = 100_000`) — uses 100000. A 10× difference in replay-buffer
   size is not "aligned" with the Zoo pipeline; the removed justification was wrong but the surrounding
   claim is now unsupported as well.
   — Suggested fix: change both commands to `--buffer-size 100000` (keeping the zoo-aligned wording and
   the run name), or delete the "validated / zoo-aligned pipeline" wording and rename the run.

8. `planning/reports/ch05-text.md` + `docs/chapter05_dqn/*` — the revision fixed at least two factual
   errors (the false "In its 2015 DQN paper DeepMind ran a comparison experiment…" attribution and the
   five `chapter07_dqn/` path references) but did not record either in
   `planning/content-review-notes.md`, which the repository guidelines require ("When fixing a factual
   error, record it (and the fix) in `planning/content-review-notes.md`"). That file currently has
   entries for Chapters 8, 9, 10, 11, 12, 15, 16, 17, 24 and 25 and none for Chapter 5.
   — Suggested fix: add a Chapter 5 block with the two fixes, marked `[x]`.

## Doubts

1. `docs/chapter05_dqn/review.md` (exercise 9 solution) — "the default run's evaluation mean rises
   faster and fluctuates less; the 10x-lag run learns from staler targets, so its curve lags and its
   final 5-episode mean is typically lower (often by tens of points on LunarLander)". The direction is
   asserted, not established: (a) a staler target moves in bigger but rarer jumps, so the _stale_ run
   may well be the smoother one, contradicting "the default run … fluctuates less"; (b) the DQN paper
   itself (Mnih et al., 2015) trained with a target update every 10,000 steps, i.e. a _larger_ target
   interval than the 1,000-step default used here, so "10× larger ⇒ worse" is not the literature's
   verdict; (c) "final 5-episode means" does not match the script default `--eval-episodes 10` (only
   the 5.3 command sets 5), and with `--total-timesteps 100000` and `--eval-freq 10000` only 10
   evaluation points exist. Suggest the answer specify the protocol and require the student to report
   the observed direction rather than assert it.

2. `docs/chapter05_dqn/review.md` (exercise 8 solution) — "the Double clone ends with a lower mean Q
   and usually a slightly lower absolute TD error on the held-out batch". The Q-gap part is well
   founded (Double DQN removes the maximization bias). The TD-error part depends entirely on which
   target scores the held-out batch: scored against the _vanilla_ max target, the Double clone should
   show a _larger_ error, since it deliberately refuses the inflated maximum; scored with each
   variant's own target, the comparison is close to a wash. Also note the code directory already
   contains `code/chapter05_dqn/double_dqn_cartpole.py`, the natural reference for this exercise.

3. `docs/chapter05_dqn/review.md` (exercise 7 solution) — "The highest-error sample is drawn most
   often but down-weighted least after normalization" reads backwards as written: with
   $w_i = (1/(NP(i)))^\beta = [9.0, 1.8, 0.9, 0.45]$ normalized by the max, the highest-error sample
   ends with the _largest_ weight (1.0), i.e. it is up-weighted most. The numbers in the answer are
   right (I recomputed $P = [0.0278, 0.1389, 0.2778, 0.5556]$, $w = [9.0, 1.8, 0.9, 0.45]$,
   normalized $[1.0, 0.2, 0.1, 0.05]$); only the prose is muddled, and the following clause ("low-error
   samples keep small weights so they barely move the parameters") states the correct idea.

4. `docs/chapter05_dqn/review.md` (References) — "Sutton, R. S., & Barto, A. G. (2018). Reinforcement
   Learning: An Introduction, 2nd ed. MIT Press. Chapters 10-11 (function approximation, the deadly
   triad, Baird's counterexample)". In the 2nd edition the chapter layout is Ch. 9 "Value-Function
   Approximation with Linear Function Approximation", Ch. 10 "Controlling the Approximation Error"
   (Baird's counterexample §10.4, the deadly triad §10.5), Ch. 11 "Approximate Policy Optimality" — so
   "Chapters 10-11" does not cover "function approximation" and includes an irrelevant chapter.
   Bibliographic imprecision only; the page's other citation (Tsitsiklis & Van Roy 1997, IEEE TAC
   42(5):674-690, for the projected Bellman fixed point) is correct, as is the Baird 1995 ICML entry.

5. `docs/chapter05_dqn/dqn-components.md` — the rewrite of the replay anecdote removed the paper
   attribution on the stated ground that "the Nature paper … contains no such specific Breakout
   anecdote". To my recollection the opposite holds: Mnih et al. show, in the replay ablation for
   Breakout, an agent trained without replay adopting the degenerate "keep the ball bouncing on one
   side" policy (the figure appears in the 2013 arXiv version and in the 2015 Nature version's
   supplementary). I could not verify this offline, so I flag it as a doubt: if the recollection is
   right, the revision replaced a correct, citable statement with an unattributed one, which is a
   regression in rigour. Suggested fix: restore the citation with hedging, e.g. "the replay ablation
   in DeepMind's DQN paper shows the agent in Breakout learning to keep the ball bouncing on one side
   when replay is removed".

6. `docs/chapter05_dqn/dqn-components.md` — "It is not the gradient of any fixed objective, which is
   one reason DQN can oscillate even when the loss looks well-behaved". More precisely: with the target
   frozen at a given sync, the update _is_ the exact gradient of the surrogate
   $\mathcal{L}(\theta)=\mathbb{E}[(y-Q(\theta))^2]$ with $y$ held constant; what it is not is a descent
   step on the true TD objective whose target depends on $\theta$ — which is why the fixed point is a
   projected fixed point rather than an MSVE minimizer. As written the sentence denies that the
   surrogate is a well-defined objective at all. (Sutton's own phrasing is close to this, so this is a
   precision question rather than a wrong claim.)

7. `docs/chapter05_dqn/visual-game-projects.md` — the "Key Settings" table lists
   `optimize_memory_usage=True` as a DQN parameter, but in `code/chapter05_dqn/dqn_atari_sb3.py` it is
   a `store_true` CLI flag (default off); it is true only for the two commands that pass
   `--optimize-memory-usage`. The row should say "enabled via `--optimize-memory-usage`".

8. `docs/chapter05_dqn/from-q-to-dqn.md` — the Chapter Layout table still links the first row as
   "[Why We Need Deep Q-Networks](./from-q-to-dqn)" although the page title was changed to
   "5.1 From Q-Learning to DQN"; the link text now disagrees with the page it points to.

## Verdict

The revision is technically strong where it matters most: the semi-gradient / MSVE / Bellman-error
subsection is correct and well derived (I re-derived the gradient
$\nabla_\theta\mathcal{L} = -2\mathbb{E}[(y-Q)\nabla_\theta Q]$, confirmed the Tsitsiklis–Van Roy
projected-fixed-point claim and the Baird citation, and re-checked the two-state sketch
$w_{k+1}=1.08\,w_k$, $1.0, 1.08, 1.1664, 1.259712$), the Polyak paragraph matches
`dqn_atari_sb3.py:305` (`tau=1.0`), the replay-ratio claim matches `train_freq=4`/`gradient_steps=1`,
and every one of the ten `review.md` exercise solutions that involve arithmetic is correct
($8.92$, $15.3664$, $1.08$ growth, $[6,4,8,2]$, $P=[0.0278,\dots]$, $w=[9.0,1.8,0.9,0.45]\to[1.0,0.2,0.1,0.05]$,
$2.5$ vs $2.0$, $40000/5000=8$). The most important problems are, however, concrete: the new
checkpoint box contradicts `dqn_gym_sb3.py` (`save_replay_buffer=True`), the new reward-scaling box
misattributes the /255 normalization to SB3's `AtariWrapper` (it is done in the policy's
`preprocess_obs`), the "5.1" page still claims Section 5.3 covers Q-value evolution and ablations
(which the writer had already corrected in Section 5.5), the Go count $3^{361}\approx10^{170}$ is off by
two orders of magnitude, the teaching snippets on the 5.5 page double-scale the pixels, the brand-new
`review.md` is not in the sidebar (so it is unreachable and its "5.R" number has no home), and the
half-applied buffer-size fix leaves the page claiming "zoo-aligned" while running a 10× smaller
buffer than RL-Zoo. Two exercise answers assert empirical outcomes (exercise 9's "typically lower by
tens of points", exercise 8's "slightly lower absolute TD error") that the cited literature does not
support, and the removal of the Nature-paper replay anecdote may have discarded a genuine citation.
Fix the seven factual items, register the new page in the sidebar, and soften the two empirical
predictions; the rest of the chapter is in good shape.
