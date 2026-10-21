# Report: ch01-text

## Summary

Technical pass over Chapter 1 (`docs/chapter01_cartpole/`). Read all three in-scope pages fully and checked every derivation, code snippet, numeric claim, and paper reference against the chapter code (`code/chapter01_cartpole/2-pytorch_ppo.py`, `plot_curves.py`) and the saved data (`output/training_metrics_seed42.csv`). Found no wrong equations or fabricated references; the main defects were prose inaccuracies about which reward average is plotted/stored, duplicated figure numbers, and two imprecise wordings. Fixed all of them, created `docs/chapter01_cartpole/review.md` (summary + 8 exercises with worked solutions, all numerics recomputed), and verified with `prettier` and a full `vitepress build`.

## Technical fixes

All verified by reading the code/data and recomputing by hand (see "Commands run and results"). Suspected-issues list from the task was empty; items below are what I found myself.

1. **File: `docs/chapter01_cartpole/metrics.md` (reward-averaging paragraph).** Wrong claim: the text said the console, the CSV's `mean_episode_reward`, and SwanLab's `rollout/ep_rew_mean` all use the rolling 100-episode average. Correction: that describes only the _current_ script; the CSV saved in the repo was written by an older single-column version whose `mean_episode_reward` holds the _per-iteration_ average (it has no `*_rollout` columns — confirmed by reading the CSV header). Rewrote the paragraph to distinguish the two schemas and state that every point in the saved file corresponds to the current `mean_episode_reward_rollout`. Also updated the Figure 1-3 caption to name the actually plotted column (`mean_episode_reward`, which `plot_curves.py` line 61 reads — the script does not read any rollout column).
2. **File: `docs/chapter01_cartpole/training.md` (CSV column description).** Wrong claim: said the saved `training_metrics_seed42.csv` contains both `mean_episode_reward` and `mean_episode_reward_rollout`. Correction: noted the current script writes both, while the saved copy uses the older single-column schema with per-iteration averages. Also reworded the iteration 1/10/11 numbers as per-iteration means of the saved file.
3. **File: `docs/chapter01_cartpole/training.md` (figure numbers).** Wrong: reward figure and diagnostics figure were both labeled Figure 1-4 (metrics.md already uses 1-3/1-4), duplicating numbers across pages. Correction: renumbered to Figure 1-5 (reward) and Figure 1-6 (frames).
4. **File: `docs/chapter01_cartpole/metrics.md` (entropy wording).** Wrong/confusing claim: "the action probabilities gradually spread apart" as entropy falls. Correction: entropy falling means the distribution _concentrates_ on one action; fixed.
5. **File: `docs/chapter01_cartpole/metrics.md` (learning-rate decay).** Imprecise claim: "the learning rate gradually decays to 0". Correction: it decays linearly toward zero, reaching $7.5\times10^{-6}$ at iteration 40 (from `frac = 1 - iteration/total_iterations`, `lr = 3e-4 * frac`).
6. **File: `docs/chapter01_cartpole/metrics.md` (rollout pseudocode).** The sketch `collect_rollout(model, env, obs, 2048)` passes `2048` into the `device` positional slot of the real signature and hides the multi-value return. Correction: changed to keyword form `collect_rollout(model, env, obs, device, num_steps=2048)` with a comment noting the simplification.
7. **Verified as correct (no change):** observation-space bounds vs. termination thresholds ($\pm4.8$/$\pm24^\circ$ declared vs. $\pm2.4$/$\pm12^\circ$ enforced); $+1$/step reward and 500-step cap; Actor-Critic 4→64→64→2/Tanh architecture and gain-0.01 output init; TD/GAE formulas and the `(1 - episode_end)` cutoff; `terminated`→0 vs. `truncated`→bootstrap-$V(s')$; clip range $[0.8,1.2]$; all six log-excerpt numbers (KL/clip%/episodes/rewards match CSV rows 1, 5, 10, 11, 20, 40); $40\times2048=81{,}920$ steps and 20,480 steps at iteration 10; entropy 0.685→0.421 vs. $\ln2\approx0.693$; value-loss range 41.8–64.1 → 0.00014; max KL 0.00871; clip fraction 14.48% → 0% in last three; references (GAE 2016, PPO 2017 arXiv, Sutton & Barto 2nd ed.).

## Other changes

- Created `docs/chapter01_cartpole/review.md` (`title: 1.R Summary and Exercises`): 7 learning outcomes, 7 key equations with one-line readings, 6 common pitfalls, 8 exercises (conceptual Q1–Q2, by-hand numerics Q3–Q6, coding Q7–Q8 using `2-pytorch_ppo.py` flags, `plot_curves.py`, and a `clip_eps` ablation). Every solution worked out; numeric answers recomputed independently (see commands).
- No checklist applies to Chapter 1 (task states "none"), so no coverage table was needed and no extra topical sections were added.
- Cross-references checked: all "Section 1.x" links resolve to the right topics (1.1 run, 1.2 principles, 1.3 visualization); the "next chapter starts from the multi-armed bandit" pointer matches Chapter 2's topic. No stale-outline numbers found.

## Needs external verification

None. All paper citations are well-known works cited without disputed specifics (no benchmark numbers, quotes, or URLs that need checking). All numeric claims were verified against the repo's own CSV/code.

## Sidebar additions

- Title: `1.R Summary and Exercises`, link: `/chapter01_cartpole/review` (new file `docs/chapter01_cartpole/review.md`; coordinator to insert and renumber as needed).

## Commands run and results

- `.venv/bin/python -c` (arithmetic check): entropy(0.3,0.7)≈0.611, ln2≈0.693, $\sum_{t<500}0.99^t$≈99.34, $\delta_0$=1.79, GAE $A_0$≈1.32, SB3 KL(1.1)≈0.0047, clipfrac example 60% — all match the exercise solutions.
- `npx prettier --write` on the three files: clean (review.md reformatted, others unchanged).
- `npx vitepress build docs`: success, no dead links.
- No code tests apply (no `code/` edits allowed; chapter has no unit tests named in the task).
- Review-response pass: `npx prettier --write` on the four touched files (clean) and `npx vitepress build docs` (success, no dead links).

## Open issues

- **Swapped filenames (coordinator decision):** `docs/chapter01_cartpole/principles.md` holds Section 1.1 ("Run CartPole") while `metrics.md` holds Section 1.2 ("CartPole Principles"). Titles, H1s, and all `./`-relative links are internally consistent, so nothing is broken, but the names are misleading. Renaming requires sidebar/config updates outside my scope — flagging for the coordinator.
- **Latent code/docs drift:** `plot_curves.py` plots `mean_episode_reward`, which is the rolling average in _new_ CSVs but the per-iteration average in the _saved_ CSV. The docs now describe this accurately, but if the CSV is ever regenerated with the current script, the book's figures will silently become smoothed curves. Suggest the code owner either pin the plot to `mean_episode_reward_rollout` or regenerate the figure.
- **Content-review notes (coordinator action):** the reviewer asked for a "Part I · Chapter 1" block in `planning/content-review-notes.md` listing the entropy, learning-rate, and CSV-schema wording fixes as `[x]`. That file is outside this task's editable scope, so I did not touch it; suggested block text: entropy "concentrates on one action" (metrics.md), learning-rate "decays linearly toward zero, 7.5e-6 at iteration 40" (metrics.md), saved-CSV single-column per-iteration schema vs. current two-column script (metrics.md/training.md).

## Review response

- Error 1 (rollout/ppo\*update sketch): accepted — fixed unpacking to `transitions, obs, ep_rewards, ep_lengths, \** = collect_rollout(...)`and added the missing`device`argument to`ppo_update`, verified against `collect_rollout`(6-tuple return) and`ppo_update`(required`device`) in `code/chapter01_cartpole/2-pytorch_ppo.py`.
- Error 2 (rerun overwrites archived CSV): accepted — reproduction commands in `training.md` and `metrics.md` now write to `output/training_metrics_rerun.csv`, and `training.md` §1.3.4 plus the `metrics.md` excerpt note state that fresh CSVs store the rolling 100-episode average in `mean_episode_reward` while the quoted numbers describe the archived per-iteration file.
- Error 3 ("unsmoothed CSV metrics"): accepted — `training.md` Section Summary now reads "per-iteration CSV metrics, plus the rolling 100-episode reward average", consistent with the code's `mean_episode_reward` (rolling) and `mean_episode_reward_rollout` (per-iteration) columns.
- Error 4 (review.md absent from sidebar / "1.R" label): rejected — the task scope forbids editing `docs/.vitepress/config.mjs` and explicitly prescribes the `1.R` title/H1 with the coordinator to renumber; the sidebar addition was already listed under "Sidebar additions" above, so this needs coordinator action, not a worker edit.
- Error 5 (no entry in planning/content-review-notes.md): rejected as a worker edit — that file is outside this task's editable scope ("Only edit files inside the scope"), and the report is the prescribed record; accepted in substance, with suggested block text under "Open issues" for the coordinator to paste.
- Doubt 1 (value-loss band 41.8–64.1): accepted — rechecked the CSV (iterations 1–5: 51.38/41.82/57.38/64.06/64.14; iteration 6: 39.17, iteration 8: 53.93) and narrowed the sentence to "over the first five iterations".
- Doubt 2 (value-loss rise only distribution shift): accepted — softened to name both causes: new state distribution plus the growing discounted-return scale (21-step episode ≈19 vs. full 500-step episode ≈99, as Exercise 6 computes).
- Doubt 3 (CPU-by-default wording): accepted — replaced with "no discrete GPU required; `--device` default is `auto`, which prefers CUDA then MPS when available", verified against `parse_args`/`resolve_torch_device` in the training script.
- Doubt 4 (Exercise 8 vs. Chapter 8 clip/KL relation): accepted — added the clarifying clause that the opposite movement here comes from the threshold itself moving with `clip_eps`, while for a fixed threshold larger updates raise both clip fraction and KL as Chapter 8 states.
- Doubt 5 (review.md has no links): accepted — added "Learning path" and "Code and data" header lines matching the other Chapter 1 pages, and linked "Section 1.2" to `./metrics` in Exercise 7.
- Doubt 6 (unpinned versions; ablations overwrite model): accepted — Exercise 7 now notes `requirements.txt` only lower-bounds versions, and Exercises 7–8 use fresh `--log-csv` subdirectories (`output/rerun_check/metrics.csv`, `output/ablation_clip*/metrics.csv`) with an explicit warning that the script derives `pytorch_ppo_cartpole.pth` from the CSV directory, verified against `train()` lines 316–319.
