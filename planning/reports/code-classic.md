# Report: code-classic

## Summary

Made the classic-RL code (`code/chapter01/02/03/05/06/07/08/09/12`, `code/appendix_a_engineering`,
`code/device_utils.py`, `code/README.md`, plus new `code/pytest.ini` / `code/conftest.py`)
reliable, tested and consistent. Every script in scope now accepts
`--seed`, `--device {auto,cpu,cuda}` and `--smoke` (tiny CPU run, <60 sSmoke,
no network, no windows, files only under `runs/`); matplotlib uses the `Agg`
backend everywhere; stale `Chapter N` labels were fixed to the current numbering;
each chapter directory has a `test_smoke.py` that runs its scripts with `--smoke`.
Final result: **37 passed, 2 skipped** (Atari scripts need `ale-py`/ROMs, which
cannot be installed offline) **+ 8 subtests passed** in ~148 s, and
`npx vitepress build docs` succeeds. `code/README.md` was rewritten for the
current layout, including a per-script table with docs links (LLM chapters
described read-only). No sidebar changes are needed (no pages added) and no
`docs/` lines were changed.

## Technical fixes

- `code/chapter08_ppo/ppo_lunar_lander.py` — **wrong**: `1 - model.clip_range`
  crashed because SB3 exposes `clip_range` as a `FloatSchedule` object
  (`TypeError: unsupported operand ... 'FloatSchedule'`). **Fix**: evaluate the
  schedule first (`clip_range(1.0)` when callable), same pattern the
  BipedalWalker script already used. **Verified**: `--smoke` previously crashed
  at this line, now passes.
- `code/chapter09_continuous_control/ppo_td3_sac_comparison.py` — **wrong**:
  `ax4.boxplot(..., labels=...)` raises `TypeError` with the installed
  matplotlib (`labels` was renamed to `tick_labels`). **Fix**: use
  `tick_labels=algo_names`. **Verified**: `--smoke` previously crashed at this
  line, now passes.
- Stale chapter numbers → corrected (verified by reading each file):
  `chapter02_rl_basics/two_armed_bandit.py` + `requirements.txt` Ch3→Ch2;
  `chapter05_dqn/*` Ch4→Ch5 (incl. `export_dqn_curves.py` docstring and the
  `chapter04_dqn/requirements.txt` hint in the `dqn_atari_sb3.py` error message,
  now `chapter05_dqn/`); `chapter06_policy_gradient/*` Ch5→Ch6 (incl.
  `render_cartpole_baseline.py` “Chapter 5.3”→“Chapter 6”);
  `chapter07_actor_critic/*` Ch6→Ch7; `chapter08_ppo/*` Ch6/Ch7→Ch8;
  `chapter12_exploration_marl_hierarchical/*` Ch13→Ch12.
- Importability: `ppo_lunar_lander.py`, `ppo_bipedal_walker.py`,
  `sac_halfcheetah.py`, `ppo_td3_sac_comparison.py`, `gae_visualization.py` ran
  training at module import time. Wrapped the bodies in `main()` + argparse so
  importing is side-effect free; defaults (timesteps, envs, hyperparams) are
  unchanged. Verified via `--smoke` subprocess runs.
- Missing optional dependencies (no network, cannot install): `swanlab` is now
  lazily imported in `1-ppo_cartpole.py` / `2-pytorch_ppo.py` (training runs
  without dashboard logging, with a note); `ale-py` is lazily imported in
  `dqn_atari_sb3.py` / `render_atari.py` (clear error message; tests skip).
  Verified: ch01 suite failed with `ModuleNotFoundError: swanlab` before, passes
  after; Atari tests skip with reason.
- Algorithm-correctness audit (read every script, hand-checked): no loss-sign,
  target-network, epsilon-schedule, advantage/return or tanh-squash bugs found.
  Terminal vs truncated is handled correctly everywhere: DQN buffers store
  `float(done)` where `done` is gymnasium `terminated` only (truncation keeps
  bootstrapping), and the PPO/GAE rollouts zero `V(s')` on `terminated` while
  bootstrapping on truncation. The one place that conflated them
  (`actor_critic_pendulum` eval loop variable naming) is VecEnv API, not logic.
- `bellman_equation_verify.py`: verified the hand-derived linear system against
  `policy_evaluation()` and added a hard `allclose` assert so pytest catches
  regressions. Verified: smoke passes.
- `gridworld_q_learning.py`: kept the exact `V*(START) == 0.728537125` /
  7-sweep asserts for full runs; they are skipped in `--smoke` (20 episodes ×
  2 seeds cannot reach them). The pre-existing
  `test_gridworld_q_learning.py` still passes unchanged.

## Other changes

- Added `--seed/--device/--smoke` to all 30+ scripts in scope; smoke budgets:
  ≤256 rollout steps / ≤6 episodes / 1–2 eval episodes, CPU-only, outputs under
  `<chapter>/runs/`. `--output`/`--model` made optional where smoke needs no file.
- `render_*` helpers: `--smoke` uses random actions + `rgb_array` (no model/ROM
  needed); `dqn_pokemon_red_pyboy.py --smoke` is a no-ROM import/CLI wiring check
  (`--rom` required only for real training).
- New `code/pytest.ini`, `code/conftest.py` (CPU-only: `CUDA_VISIBLE_DEVICES=""`),
  `test_smoke.py` per chapter dir, `__init__.py` per chapter dir (fixes pytest
  duplicate-basename collection), and
  `code/chapter03_value_functions/conftest.py` (keeps the pre-existing sibling
  import working).
- `device_utils.py` needed no changes (already `auto|cuda|mps|cpu` CPU-safe).
- `code/online-experiments/`: notebooks + README kept as is (upstream material).

## Needs external verification

- Atari DQN behavior/figures (scripts skip here: no `ale-py`/ROMs, offline).
- Pokémon Red WRAM addresses (`W_X_COORD` etc.) and expected ROM SHA1: carried
  over untouched, unverifiable without a legal ROM.
- Whether the appendix reward-hacking scenarios reliably _demonstrate_ hacking
  at full budget (smoke budgets are too small to confirm the pedagogy).
- SB3 logger keys read by the custom callbacks (`train/approx_kl`, …) — taken
  from the existing code, not re-verified against SB3 source.

## Sidebar additions

None (no docs pages added or moved).

## Commands run and results

- `.venv/bin/pytest code/chapter02_rl_basics -q` → 1 passed
- `.venv/bin/pytest code/chapter03_value_functions -q` → 7 passed, 8 subtests passed
- `.venv/bin/pytest code/chapter06_policy_gradient -q` → 4 passed
- `.venv/bin/pytest code/chapter12_exploration_marl_hierarchical code/appendix_a_engineering -q` → 3 passed
- `.venv/bin/pytest code/chapter01_cartpole -q` → 4 passed
- `.venv/bin/pytest code/chapter05_dqn -q` → 7 passed, 2 skipped (ale-py/ROMs)
- `.venv/bin/pytest code/chapter07_actor_critic -q` → 4 passed
- `.venv/bin/pytest code/chapter08_ppo -q` → 5 passed
- `.venv/bin/pytest code/chapter09_continuous_control -q` → 2 passed
- Final: `.venv/bin/pytest code/chapter01_cartpole code/chapter02_rl_basics code/chapter03_value_functions code/chapter05_dqn code/chapter06_policy_gradient code/chapter07_actor_critic code/chapter08_ppo code/chapter09_continuous_control code/chapter12_exploration_marl_hierarchical code/appendix_a_engineering -q` → **37 passed, 2 skipped, 8 subtests passed in 147.82s**
- `npx prettier --write code/README.md` → ok (prettier has no Python parser, so `.py` files were not passed through it)
- `npx vitepress build docs` → success (no docs changes made)

## Open issues

- Stray artifact `code/chapter09_continuous_control/MUJOCO_LOG.TXT` (created by a
  MuJoCo smoke run) should be deleted; `rm` is not permitted in this
  environment, so it is left in the worktree.
- Stale docs paths (out of scope — no command I changed): `docs/chapter05_dqn/visual-game-projects.md`
  lines ~366–521 use
  ...[truncated 911 chars]
