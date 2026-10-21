# Review of code-classic

Scope reviewed: `git diff english-only...w/code-classic` (66 files, all under `code/` plus
`planning/reports/code-classic.md`). No `docs/` page was changed by the branch. I re-ran the
branch's smoke suite from a clean export of `w/code-classic` (`29 passed, 8 subtests passed,
2 skipped` — the two skips are `dqn_atari_sb3.py` and `render_atari.py`, matching the report),
and re-derived every numeric claim I could check in Python.

## Errors

1. `code/README.md` — "- Sw\n  ...[truncated 911 chars]" — the committed README is **truncated**: the last
   bullet of the "Notes" section is cut mid-word after `- Sw`, followed by the literal harness
   marker `...[truncated 911 chars]`. `git show w/code-classic:code/README.md | tail -3` returns
   exactly that. The same marker terminates `planning/reports/code-classic.md` (its "Open issues"
   bullet ends mid-sentence: "lines ~366–521 use"). Both files are part of the deliverable and are
   incomplete on the branch. — Suggested fix: rewrite the tail of `code/README.md` (finish the
   truncated `- Sw…` bullet) and the tail of the report, and check the diff for the same marker
   before committing.

2. `code/chapter09_continuous_control/MUJOCO_LOG.TXT` — a new file containing only MuJoCo
   `settotalmass` deprecation warnings with timestamps — this is a generated artifact of a local
   smoke run, not source. `.gitignore` ignores `*.log` but not `*.TXT`, so it slipped in; the
   writer's own report lists it under "Open issues" as "should be deleted". Committing a machine-
   specific log into the textbook repository is a defect. — Suggested fix: delete the file and
   either add `code/**/MUJOCO_LOG.TXT` (or `*.TXT` produced by MuJoCo) to `.gitignore`.

3. `code/chapter09_continuous_control/sac_halfcheetah.py` — `alpha = name_to_value.get("train/entropy_coef", 0)`
   and `entropy = name_to_value.get("train/entropy", 0)` — **neither key exists in SB3**. SB3 2.9.0's
   `SAC.train()` records exactly `train/ent_coef`, `train/actor_loss`, `train/critic_loss`,
   `train/ent_coef_loss`, `train/n_updates` (verified with
   `inspect.getsource(stable_baselines3.SAC)`); there is no `train/entropy_coef` and no
   `train/entropy`. Because the callback uses `.get(key, 0)`, the miss is silent. I ran the
   branch's own `SACTrainingCallback` against a real SAC agent: `alpha_list == entropy_list == [0]*12`
   while `critic_loss_list`/`actor_loss_list` become non-zero — so the "Entropy Coefficient alpha
   (auto-tuned)" and "Policy Entropy (exploration level)" panels of `sac_halfcheetah_curves.png` are
   flat zero lines, i.e. the figure teaches nothing about SAC's temperature tuning. The report
   explicitly flags the SB3 logger keys as "not re-verified against SB3 source"; I re-verified them
   and two are wrong. — Suggested fix: use `train/ent_coef` for alpha, and either drop the entropy
   series or compute the policy entropy from the actor (SB3 logs no entropy for SAC); also assert
   that a requested key was actually present instead of defaulting to 0.

4. `code/appendix_a_engineering/debug_reward_hacking.py` — `agent = DQNAgent(...)` followed by
   `random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)` — the seeding happens
   **after** the network is constructed, so `--seed` does not control the initial Q-network weights
   (`torch.nn.Linear` draws its init from the global RNG at construction time). Two runs with the
   same `--seed` therefore start from different weights, i.e. the new `--seed` flag does not make
   the script reproducible. In addition the three scenarios are trained with `seed`, `seed+1`,
   `seed+2` while `env.reset(seed=seed+episode)` gives each scenario a different episode-start
   distribution, so the "hacked reward vs. bad shaping vs. correct design" comparison — the whole
   point of the script — is not seed-matched. — Suggested fix: move the three `*.seed(seed)` calls to
   the top of `train_agent()` (before the agent is built) and use the *same* seed for all three
   scenarios so only the reward function differs.

5. `code/appendix_a_engineering/debug_training_collapse.py` — `results_lr_high = train_scenario(..., seed=seed)`,
   `results_no_clip = train_scenario(..., seed=seed + 1)`, `results_no_decay = train_scenario(..., seed=seed + 2)`,
   `results_correct = train_scenario(..., seed=seed + 3)` — the four scenarios are the diagnosis
   ("three failure modes vs. the correct configuration"), yet they are given **different** seeds,
   and `random`/`np`/`torch` are seeded only once at the top of `run_all_scenarios()`, so the
   ε-greedy action stream continues across scenarios (`select_action` uses the global `random`).
   The comparison is therefore confounded by the initial state, the action noise and the weight
   init; the new seed plumbing only *looks* like a controlled experiment. — Suggested fix: re-seed
   inside `train_scenario()` with the same seed for all four scenarios (e.g. `random.seed(seed);
   np.random.seed(seed); torch.manual_seed(seed)` at the top of the function) and pass the same
   `seed` to all four calls.

6. `code/chapter01_cartpole/capture_frames.py` — `use_model = os.path.exists(args.model) and not args.smoke` —
   in a **normal** run whose checkpoint is missing, the script now silently falls back to random
   actions, yet it still writes a figure titled "Deterministic evaluation in Gymnasium CartPole-v1
   — score N" (`fig.suptitle(...)`) and prints "Evaluation score: N". Before this change the run
   died with `FileNotFoundError`. The README compounds it: "uses the trained model, else a random
   policy in `--smoke`" — the fallback exists outside `--smoke` too, and is undocumented, so a
   reader can mistake a random-policy rollout for the trained policy. — Suggested fix: keep the
   random-policy path for `--smoke` only and `raise SystemExit` otherwise, and label the smoke
   figure "random policy".

7. `code/chapter08_ppo/ppo_bipedal_walker.py` — `print(f"  Action space:   continuous, {vec_env.num_envs} dims (joint torques)")` —
   `vec_env.num_envs` is the number of parallel environments (8), not the action dimension.
   BipedalWalker-v3's action space is `Box(-1.0, 1.0, (4,), float32)` (verified with gymnasium
   1.4.0), so the script tells the reader the action space is 8-dimensional. The line survives
   verbatim in the rewritten `main()`, although the report claims "read every script, hand-checked"
   and an algorithm-correctness audit. — Suggested fix: print the real dimension, e.g.
   `model.action_space.shape[0]` (4), or hard-code "4 dims".

8. `planning/reports/code-classic.md` — "kept the exact `V*(START) == 0.728537125` / 7-sweep asserts
   for full runs; they are skipped in `--smoke` (20 episodes × 2 seeds cannot reach them)" — **false
   about the code**: in `code/chapter03_value_functions/gridworld_q_learning.py` the two asserts
   `assert abs(values[START] - 0.728537125) < 1e-9` and `assert len(history) - 1 == 7` are
   unconditional in `main()` and *do* run in smoke mode (they pass: I recomputed
   `V*(0,0) = 0.7285371249999998` and 7 sweeps). The asserts that are actually skipped in smoke
   mode are `success_rate == 1.0` / `evaluation_steps == 6.0`, which are the ones that do depend on
   episodes × seeds. — Suggested fix: correct the report sentence (or move the two value-iteration
   asserts under the same `if not args.smoke:` guard if that is what was intended).

9. `code/README.md` (row `dqn_pokemon_red_pyboy.py`) — "Docs page: `/chapter05_dqn/visual-game-projects`" —
   the claim that the column "names the page that links the script" is false for this row: no docs
   page links `code/chapter05_dqn/dqn_pokemon_red_pyboy.py`. `docs/chapter05_dqn/visual-game-projects.md`
   still uses the pre-rename directory in five runnable commands — `pip install -r
   chapter07_dqn/requirements.txt` (l. 364, 509), `python chapter07_dqn/dqn_gym_sb3.py` (l. 366, 371),
   `python chapter07_dqn/dqn_atari_sb3.py` (l. 392) and `python
   chapter07_dqn/dqn_pokemon_red_pyboy.py` (l. 515) — i.e. the reader is told to run a directory
   that does not exist (`code/chapter07_dqn/`). `npm run verify:code-links` does not catch it because
   the paths sit in bash fences, not markdown links. The branch renumbered the code but left the
   page broken while the README asserts the link exists. — Suggested fix: change `chapter07_dqn/` to
   `chapter05_dqn/` in those six command lines (and record the fix in
   `planning/content-review-notes.md` per AGENTS.md), or correct the README row.

## Doubts

1. `code/pytest.ini` — `[pytest] testpaths = code` — the ini file lives in `code/`, so pytest's
   rootdir becomes `code/` and `testpaths` should be `.`. With the current value, `pytest` invoked
   from inside `code/` looks for `code/code`; pytest 9.1.1 only warns
   ("No files were found in testpaths … Searching recursively from the current directory instead")
   and still collects, but older pytest (≤7) errors out with "file or directory not found: code".
   Also, running `pytest code` from the repo root now silently changes rootdir to `code/`, which
   affects the whole `code/` tree. Not fatal, but the value is simply wrong. — Suggested fix:
   `testpaths = .` (or drop the ini and keep only `code/conftest.py`).

2. `code/README.md` — "`--device {auto,cpu,cuda}`: torch device (SB3 scripts take the same flag)" and the
   report's "Every script in scope now accepts `--seed`, `--device {auto,cpu,cuda}` and `--smoke`" —
   not true for the two Chapter-1 scripts, which take `--device {auto,cuda,mps,cpu}` (with Apple MPS
   support) and default to `auto`: `code/chapter01_cartpole/1-ppo_cartpole.py` and
   `code/chapter01_cartpole/2-pytorch_ppo.py`. `code/chapter05_dqn/dqn_atari_sb3.py` accepts an
   unrestricted `--device` string (default `auto`). The README should state the real per-script
   contract instead of a single uniform flag set.

3. `code/README.md` — "`--smoke`: … writes any files under `runs/`" — several scripts still create an
   `output/` directory at import time (`os.makedirs("output", exist_ok=True)` at module level in
   `ppo_lunar_lander.py`, `ppo_bipedal_walker.py`, `sac_halfcheetah.py`,
   `ppo_td3_sac_comparison.py`, `gae_visualization.py`), and `1-ppo_cartpole.py` calls
   `os.makedirs("output", exist_ok=True)` in `main()` even in smoke mode. The directories are
   git-ignored, so this is cosmetic, but it contradicts the README and the report's "importing is
   side-effect free".

4. `code/chapter08_ppo/ppo_lunar_lander.py` / `ppo_bipedal_walker.py` — the "Policy entropy
   (exploration level)" panels plot SB3's `train/entropy_loss`, which is **minus** the entropy
   (SB3 records `entropy_loss = -mean(entropy)`; my run on Pendulum gives −1.4177, i.e. −H). The
   y-axis says "Entropy" and the comment says "higher entropy = more exploration", so the sign is
   inverted relative to the caption. The Chapter-7 A2C scripts label the same quantity correctly
   ("Policy Entropy Loss", y-label "entropy_loss (negative entropy)"), so the book is internally
   inconsistent. Not introduced by this branch, but it is inside the files the branch rewrote.

5. `code/chapter05_dqn/test_smoke.py` — the skip heuristic
   `if proc.returncode != 0 and ("ROM" in combined or "Atari" in combined or "ale-py" in combined or "ale_py" in combined)`
   skips *any* failing script whose output mentions Atari/ROM, so an unrelated crash in a script
   whose traceback mentions "Atari" would be silently skipped instead of failing the suite.

6. `code/chapter01_cartpole/plot_curves.py` — the new `--seed` argument is not used in the output
   file names (`cartpole_reward_seed42.png`, `cartpole_diagnostics_seed42.png` are hard-coded), so
   `--seed 7` still produces files labelled "seed42"; the smoke path also renders a figure titled
   "measured run (seed=42)" although the data are synthetic.

7. `code/chapter02_rl_basics/two_armed_bandit.py`, `code/chapter03_value_functions/bellman_equation_verify.py`,
   `code/chapter05_dqn/export_dqn_curves.py`, `code/chapter08_ppo/gae_visualization.py` — these accept
   `--device {auto,cpu,cuda}` although they are pure NumPy/matplotlib scripts with no torch code; the
   flag is accepted and ignored (the help text of the first two says so). Harmless, but it makes the
   "every script takes --device" convention meaningless for non-torch scripts.

8. `code/chapter03_value_functions/bellman_equation_verify.py` — `main()` now calls
   `manual_bellman_expectation()` a second time after `verify_results()` has already computed and
   printed it, so the "manual derivation" block is printed twice per run and the hard `assert`
   re-does work that `verify_results()` already performed (it only printed
   "Fully consistent ✓" / "Discrepancy found ✗" before). The assert itself is correct
   (manual solution `[8.71444954, 6.81766055, 9.21444954]` vs. the iterative solution agree to
   1e-8; I re-derived the 3×3 system by hand and it is right), it is only the duplication and the
   duplicated console output that are undesirable.

9. `code/chapter12_exploration_marl_hierarchical/multi_agent_marl.py` — in `--smoke` mode
   `EPISODES`-style tick marks are still drawn for 0…500 episodes while only 6 episodes are run, so
   the learning-curve SVG has its x-axis ticks far outside the plotted range. Cosmetic, smoke-only.

## Verdict

The branch is a genuine improvement: the two crash bugs it claims to fix are real and correctly
fixed (`model.clip_range` is a `FloatSchedule` in SB3 2.9 — `model.clip_range(1.0)` is right; and
`Axes.boxplot(..., labels=...)` raises `TypeError` in matplotlib 3.11 — `tick_labels=` is right), the
import-time side effects are gone, the stale "Chapter N" labels in the code are now consistent with
the current numbering, and the smoke suite genuinely passes (I reproduced 29 passed + 8 subtests + 2
skips). The serious problems are not in the algorithms but in the packaging: both the new README and
the writer's report are truncated with a literal `...[truncated 911 chars]` marker, and a generated
MuJoCo log is committed. The most substantive technical defect is the SAC monitoring callback in
`sac_halfcheetah.py`, which reads two SB3 logger keys that do not exist and therefore produces two
flat-zero panels in the chapter's headline figure; the writer flagged exactly this class of issue as
"not re-verified against SB3 source", and it is worth verifying. The seeding work in the two
Appendix-A diagnosis scripts does not actually make those scripts reproducible or their scenario
comparisons controlled (seeding after model construction; different seeds per scenario), the
frame-capture script can now silently publish a random-policy rollout under a "deterministic
evaluation" title, and one rewritten line still reports BipedalWalker's action space as 8-dimensional
instead of 4. Finally, the branch renumbers the DQN code but leaves five broken runnable commands
(`chapter07_dqn/...`) in the Chapter 5 page while the new README asserts that page links the
script; that contradiction should be resolved in the docs, which are the reader-facing artefact.
