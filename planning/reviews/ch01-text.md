# Review of ch01-text

Scope reviewed: `docs/chapter01_cartpole/metrics.md`, `docs/chapter01_cartpole/training.md`, the new
`docs/chapter01_cartpole/review.md`, and `planning/reports/ch01-text.md`. `docs/chapter01_cartpole/principles.md`,
`code/chapter01_cartpole/*.py` and `code/chapter01_cartpole/output/training_metrics_seed42.csv` were read as
cross-checks. Every equation in the new material was re-derived, every numeric claim was recomputed
(`.venv/bin/python`), all eight exercise solutions in `review.md` were worked out independently, and the
figure numbering and the sidebar were checked against `docs/.vitepress/config.mjs`.

## Errors

1. `docs/chapter01_cartpole/metrics.md:246` — "transitions, obs = collect_rollout(model, env, obs, device, num_steps=2048)"
   — the sketch still does not match the function it calls. `collect_rollout` in
   `code/chapter01_cartpole/2-pytorch_ppo.py` returns a **six**-tuple
   `(transitions, obs, completed_rewards, completed_lengths, episode_reward, episode_length)`, so unpacking it
   into two names raises `ValueError: too many values to unpack`. The comment added in the same edit ("…and
   returns the completed-episode statistics alongside the transitions") states the very fact that the code line
   contradicts. Two lines further down, `ppo_update(model, optimizer, transitions, advantages, returns)` also
   drops the required positional argument `device` (real signature: `ppo_update(model, optimizer, transitions,
   advantages, returns, device, clip_eps=0.2, epochs=10, batch_size=64)`), so that call fails as well. The
   writer's report claims this sketch was corrected to match the real signature ("hides the multi-value
   return"); the call form was fixed, the unpacking was not.
   — Suggested fix: `transitions, obs, ep_rewards, ep_lengths, *_ = collect_rollout(model, env, obs, device, num_steps=2048)`
   and pass `device` in the `ppo_update` call (or extend the "simplified" comment to say so explicitly).

2. `docs/chapter01_cartpole/training.md:163` — "The per-iteration mean reward in the saved CSV is 21.35 in
   iteration 1, 500.0 in iteration 10, and falls back to 460.4 in iteration 11." The numbers are right for the
   *archived* CSV, but §1.3.2 (line 42) still instructs the reader to regenerate **that same file**
   (`--log-csv output/training_metrics_seed42.csv`), i.e. to overwrite the archived copy with the current
   script's schema, in which `mean_episode_reward` is the 100-episode rolling average. After that regeneration
   the quoted numbers no longer hold: the rolling mean at iteration 10 is ≈148, not 500.0. Calculation from
   the CSV's own per-iteration totals: the last 100 completed episodes at the end of iteration 10 are the
   4+6+9+10+10+15+29 = 83 episodes of iterations 4–10 (2000+1908+1923+2243+1894+2001+2012 = 13981 steps)
   plus 17 episodes of iteration 3 (mean 49.17, ≈836 steps) → ≈14817/100 ≈ **148.2**; the same window at
   iteration 5 is ≈65 (not 133.4). `plot_curves.py` reads exactly that column, so the regenerated figure would
   be a smoothed curve that no longer matches Figure 1-5. The revision documents the schema drift in prose but
   leaves the reproduction flow aimed at the archived file.
   — Suggested fix: point the reader at a different output path (`--log-csv output/training_metrics_rerun.csv`),
   or add one sentence to §1.3.4: "a freshly generated CSV stores the rolling 100-episode average in
   `mean_episode_reward` (the archived file behind the figures stores the per-iteration average), so the
   numbers quoted here describe the archived run".

3. `docs/chapter01_cartpole/training.md:196` — "The training script saves the model parameters and the
   unsmoothed CSV metrics." — this contradicts the revision's own new text (metrics.md:307: "`mean_episode_reward`
   … is the rolling average over the most recent 100 complete episodes") and the code: in the current script
   `mean_episode_reward` is a 100-episode rolling, i.e. smoothed, average; only `mean_episode_reward_rollout`
   and the other columns are per-iteration.
   — Suggested fix: "The training script saves the model parameters and the per-iteration CSV metrics, plus the
   rolling 100-episode reward average."

4. `docs/chapter01_cartpole/review.md` — the new page is not registered in `docs/.vitepress/config.mjs` (the
   branch touches no config file). The Chapter 1 sidebar block still lists only 1.1/1.2/1.3, so
   `/chapter01_cartpole/review` is unreachable from the navigation and from the "next/previous page" footer of
   the built site, and the title "1.R Summary and Exercises" has no counterpart in the sidebar, where every page
   is numbered `X.Y` (the full sidebar runs 1.1 … 26.4 with no `X.R` entry anywhere). AGENTS.md is explicit:
   "Every navigable page must be listed there" and "New pages … must be added to the sidebar in
   `docs/.vitepress/config.mjs`". `npm run fix:headings` cannot repair this page either, because its sidebar
   regex only matches numeric minor numbers (`scripts/fix-heading-numbers.mjs`:
   `pairRe = /text:\s*'(\d+)\.(\d+)\s([^']*)'/`), so "1.R" is invisible to it. The writer's report defers this
   to a coordinator, which leaves the branch with an orphaned chapter-summary page.
   — Suggested fix: add `{ text: '1.4 Summary and Exercises', link: '/chapter01_cartpole/review' }` to the
   Chapter 1 sidebar block, set the front-matter title and the H1 to `1.4 Summary and Exercises`, then run
   `npm run fix:headings`.

5. `planning/content-review-notes.md` — the revision fixed several factual statements (the entropy wording at
   metrics.md:354, the learning-rate description at metrics.md:388, the CSV-schema description at
   training.md:54) but recorded none of them in `planning/content-review-notes.md`, which the repository
   guidelines require ("When fixing a factual error, record it (and the fix) in
   `planning/content-review-notes.md`"). That file currently carries blocks for Chapters 8–12, 15–17, 24 and 25
   and none for Chapter 1; the writer's own report (`planning/reports/ch01-text.md`) is not a substitute for it.
   — Suggested fix: add a "Part I · Chapter 1" block listing the three fixes, marked `[x]`.

## Doubts

1. `docs/chapter01_cartpole/metrics.md:369` — "In this run the value loss fluctuates between 41.8 and 64.1 early
   on and finally falls to 0.00014." The band 41.8–64.1 holds only for iterations 1–5 (51.38, 41.82, 57.38,
   64.06, 64.14); from iteration 6 on the loss is already below it (39.17, 31.73, 31.51 at iterations 6, 7, 9)
   and reaches 8.56 by iteration 12. The writer's report lists "value-loss range 41.8–64.1" as verified, but as
   written the sentence overstates the range. Suggested fix: "In the first five iterations the value loss stays
   between 41.8 and 64.1 before it falls to 0.00014."

2. `docs/chapter01_cartpole/metrics.md:369` — "The rise in the middle of training means the policy reached a new
   state distribution, and the Critic needed to readjust its predictions." The scale of the regression targets
   also grows during training: a 21-step episode has a discounted return of ≈19 while a full 500-step episode has
   ≈99 (the new review page computes exactly this in Exercise 6), so part of the early rise in the squared error
   follows from the growing magnitude of the targets, not only from distribution shift. The two pages of the same
   revision therefore give two different readings of the same curve.

3. `docs/chapter01_cartpole/training.md:30` — "The CartPole network is very small and trains on the CPU by
   default, so no discrete GPU is required." The script's `--device` default is `auto`, and
   `resolve_torch_device` prefers CUDA, then MPS, then CPU, so on a machine with a GPU the run trains on the
   GPU. The point the sentence wants to make (no discrete GPU needed) is right; the mechanism is not.

4. `docs/chapter01_cartpole/review.md:145` (Exercise 8) — "the diagnostic pair moves in opposite directions, which
   is exactly the trade-off the clipped objective encodes". This holds only because the clip-fraction *threshold*
   moves with `clip_eps` (the metric is `mean(|r-1| > clip_eps)`, so widening the band mechanically shrinks the
   count). Chapter 8 states the opposite relation for a fixed threshold: "when the clip fraction is high, the KL
   divergence is also high" (`docs/chapter08_ppo/ppo-bipedal-walker.md:165`). A reader who carries this
   conclusion over to a normal run will contradict Chapter 8; one clarifying clause in the solution would prevent
   that.

5. `docs/chapter01_cartpole/review.md` — the page carries no links at all: no "Learning path" line, no "Code and
   data" line to `2-pytorch_ppo.py` / `plot_curves.py` / the archived CSV (every other Chapter 1 page has one),
   and Exercise 7 refers to "Section 1.2" without a link. Not an error, but the page is orphaned apart from the
   sidebar entry it does not have.

6. `docs/chapter01_cartpole/review.md` — Exercise 7 says the numbers "may differ from the book's saved file if the
   installed library versions differ", but the run is not pinned: metrics.md:273 claims "Python 3.12.13, PyTorch
   2.13.0, Gymnasium 1.3.0, NumPy 2.5.2" while `code/chapter01_cartpole/requirements.txt` only lower-bounds the
   libraries (`torch>=2.0.0`, `gymnasium[classic-control]>=0.29.0`). A reader cannot reproduce the quoted CSV from
   the repository alone. Related, and worth a note in the exercise: both ablation commands of Exercise 8 write
   their CSV into `output/`, and the training script derives the model path from `--log-csv`
   (`model_path = os.path.join(os.path.dirname(os.path.abspath(args.log_csv)), "pytorch_ppo_cartpole.pth")`), so
   each ablation silently overwrites `output/pytorch_ppo_cartpole.pth`, the saved model the frames of §1.3.5
   depend on.

## Verdict

The revision's central correction is right and important: the archived
`training_metrics_seed42.csv` really does store the per-iteration average in its single `mean_episode_reward`
column, not the 100-episode rolling average (a rolling window at iteration 5 would be ≈65, while the file stores
133.4, and the plotted figure matches the per-iteration values), and the current script really does write both
columns. The figure renumbering is now unique and sequential across the chapter (1-1/1-2 in principles.md, 1-3/1-4
in metrics.md, 1-5/1-6 in training.md), the entropy wording ("concentrates on one action") is the right
correction, and the learning-rate sentence is confirmed by the code (`frac = 1 - iteration/total_iterations`,
`lr = 3e-4 * frac` → 7.5e-6 at iteration 40, as stored in the CSV). All eight exercise solutions are numerically
correct: H(0.3,0.7) = 0.6109, δ₀ = 1.79, A₀ = 1.3197, clip fraction 3/5 = 60 %, Σ₀⁴⁹⁹0.99ᵗ = 99.343,
2048/21.35 = 95.9 and 91 × 21.35 = 1943 (105 steps left over). The problems are in the integration, not the
mathematics: the edited pseudocode still unpacks a six-value function into two names and drops `device` from
`ppo_update`; the reproduction instructions overwrite the archived CSV with the new schema, after which the
quoted per-iteration numbers (500.0 at iteration 10) become a rolling mean of ≈148 and the regenerated figure no
longer matches the book; the Section Summary still calls the CSV metrics "unsmoothed", which the revision's own
new paragraph contradicts; the brand-new `review.md` is absent from the sidebar and carries a non-numeric "1.R"
label that `fix:headings` cannot manage; and none of the factual fixes were recorded in
`planning/content-review-notes.md` as the repository rules require. Fix those five items, soften the value-loss
reading and the Exercise 8 claim to stay consistent with Chapter 8, and the revision is in good shape.
