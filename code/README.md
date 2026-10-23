# Code Index

Companion code for _Hands-on Modern RL_ (English). This repository is a fork of
[walkinglabs/hands-on-modern-rl](https://github.com/walkinglabs/hands-on-modern-rl),
used under CC BY-NC-SA 4.0; attribution to WalkingLabs is kept wherever it appears.

`code/` has one directory per book chapter (`code/chapter05_dqn` belongs to Chapter 5),
plus `code/device_utils.py` (shared torch-device helpers), `code/online-experiments/`
(upstream ModelScope companion notebooks, kept as is), `code/pytest.ini` and
`code/conftest.py` (test configuration). Each chapter directory has its own
`requirements.txt` with that chapter's minimal dependencies, and each
hands-on lab chapter (4, 10, 11, 16, 18, 20, 21, 22, 24, 25, 26) has its own
`README.md` with run instructions. The pinned full environment is
`code/requirements.lock.txt`.

## Environment setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Full dependency set, pinned for reproducibility (recommended)
uv pip install -r code/requirements.lock.txt

# ... or only the dependencies for one chapter
uv pip install -r code/chapter05_dqn/requirements.txt
```

The project virtualenv is `.venv` (Python 3.12, torch,
gymnasium, stable-baselines3, transformers, trl, pytest).
`code/requirements.lock.txt` pins the full reproducible set; do not edit
it by hand. Scripts that need an unavailable package (Atari ROMs via
`ale-py`, `pyboy`, `swanlab`) import it lazily, skip the corresponding
test with a clear reason, and keep working without it.

## Running the code

Every training script follows the same CLI conventions (the code stays
readable — this is a textbook):

- `--seed`: seeds Python, NumPy, torch and the environment.
- `--device`: torch device. Most scripts take `{auto,cpu,cuda}`; the two
  Chapter 1 scripts (`1-ppo_cartpole.py`, `2-pytorch_ppo.py`) also accept
  `mps` (Apple Silicon) and default to `auto`, and `dqn_atari_sb3.py` takes
  an unrestricted `--device` string defaulting to `auto`. Pure
  NumPy/matplotlib scripts (e.g. `two_armed_bandit.py`,
  `bellman_equation_verify.py`, `export_dqn_curves.py`,
  `gae_visualization.py`) accept `--device` for CLI consistency but ignore
  it.
- `--smoke`: a tiny configuration (few steps, small nets) that finishes in well
  under ~60 s on CPU, needs no network, opens no windows (never
  `render_mode="human"`), and writes its figures under `runs/` (gitignored).
  (Some scripts still create an empty `output/` directory at import time via
  a module-level `os.makedirs`; that directory is gitignored and holds no
  smoke outputs.)

Matplotlib always uses the non-interactive `Agg` backend when saving figures.

Run a single script, e.g.:

```bash
cd code/chapter05_dqn
../../.venv/bin/python dqn_cartpole.py --smoke
```

Run the smoke tests (from the repository root):

```bash
.venv/bin/pytest code -q
```

There is one smoke-test file per chapter directory (`test_smoke.py` in
the classic chapters, `test_chNN_lab.py` alongside the newer hands-on
labs): each runs every script of that chapter with `--smoke` in a
subprocess and asserts it exits 0 (and, where cheap, that it produced
its output). The lab test files additionally hold unit tests that check
the lab's computations against hand-worked values. Render /
frame-capture helpers are also covered: they are importable and work in
smoke mode with `rgb_array` rendering. Scripts that cannot run here
(Atari ROMs, `pyboy`) are skipped with an explicit reason.

## Chapter code overview

Docs pages are root-relative links without the `.md` extension, e.g.
`/chapter05_dqn/from-q-to-dqn`. The “Docs page” column names the page that
covers or runs the script (via a runnable command or an explicit
reference); “—” means the script is exercised through its chapter's
smoke tests and chapter pages without a single dedicated reference.

| Ch. | Directory                                  | Scripts                         | What each does                                                                                     | Docs page                                                                                        |
| --- | ------------------------------------------ | ------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1   | `chapter01_cartpole/`                      | `1-ppo_cartpole.py`             | SB3 PPO trains CartPole, logs SwanLab curves (skipped if `swanlab` is missing)                     | `/chapter01_cartpole/principles`                                                                 |
|     |                                            | `2-pytorch_ppo.py`              | Pure PyTorch PPO: Actor-Critic, GAE, clipping                                                      | `/chapter01_cartpole/principles`, `/chapter01_cartpole/metrics`, `/chapter01_cartpole/training`  |
|     |                                            | `plot_curves.py`                | Plots reviewable metric curves from the exported CSV                                               | `/chapter01_cartpole/metrics`, `/chapter01_cartpole/training`                                    |
|     |                                            | `capture_frames.py`             | Captures real Gymnasium rendered frames (trained model; `--smoke` uses a random policy)            | `/chapter01_cartpole/training`                                                                   |
| 2   | `chapter02_rl_basics/`                     | `two_armed_bandit.py`           | Two-armed bandit: random vs greedy vs ε-greedy vs UCB                                              | `/chapter02_rl_basics/review`                                                                    |
| 3   | `chapter03_value_functions/`               | `bellman_equation_verify.py`    | Numerical check of the Bellman expectation/optimality equations                                    | —                                                                                                |
|     |                                            | `gridworld_q_learning.py`       | 4×4 GridWorld value iteration + Q-learning, exports SVGs/CSV/JSON                                  | `/chapter03_value_functions/value-experiment`                                                    |
| 5   | `chapter05_dqn/`                           | `dqn_cartpole.py`               | From-scratch DQN on CartPole                                                                       | `/chapter05_dqn/review`                                                                          |
|     |                                            | `double_dqn_cartpole.py`        | DQN vs Double DQN comparison                                                                       | `/chapter05_dqn/review`                                                                          |
|     |                                            | `dqn_gym_sb3.py`                | SB3 DQN on discrete Gym tasks (CartPole, MountainCar, LunarLander)                                 | `/chapter05_dqn/lunar-lander`                                                                    |
|     |                                            | `dqn_atari_sb3.py`              | SB3 DQN on Atari (needs `ale-py`/ROMs; skipped without them)                                       | `/chapter05_dqn/visual-game-projects`                                                            |
|     |                                            | `dqn_pokemon_red_pyboy.py`      | SB3 DQN on early Pokémon Red exploration (needs a legal ROM; `--smoke` is a no-ROM wiring check)   | `/chapter05_dqn/visual-game-projects`                                                            |
|     |                                            | `export_dqn_curves.py`          | Exports DQN eval curves as docs figures                                                            | `/chapter05_dqn/lunar-lander`, `/chapter05_dqn/visual-game-projects`                             |
|     |                                            | `render_atari.py`               | Renders a trained Atari DQN model as GIF (needs ROMs; skipped without them)                        | `/chapter05_dqn/visual-game-projects`                                                            |
|     |                                            | `render_lunarlander.py`         | Renders a trained LunarLander DQN as GIF (the split-episode variant below is the one the docs use) | —                                                                                                |
|     |                                            | `render_lunarlander_split.py`   | Renders one GIF per LunarLander episode                                                            | `/chapter05_dqn/lunar-lander`                                                                    |
| 6   | `chapter06_policy_gradient/`               | `reinforce_cartpole.py`         | From-scratch REINFORCE on CartPole                                                                 | `/chapter06_policy_gradient/reinforce`, `/chapter06_policy_gradient/cartpole`                    |
|     |                                            | `reinforce_with_baseline.py`    | REINFORCE vs REINFORCE + value baseline                                                            | `/chapter06_policy_gradient/baseline-experiment`, `/chapter06_policy_gradient/cartpole-baseline` |
|     |                                            | `actor_critic_cartpole.py`      | Online one-step Actor-Critic on CartPole                                                           | —                                                                                                |
|     |                                            | `render_cartpole_baseline.py`   | Trains both REINFORCE variants and renders GIFs                                                    | `/chapter06_policy_gradient/baseline-experiment`, `/chapter06_policy_gradient/cartpole-baseline` |
| 7   | `chapter07_actor_critic/`                  | `actor_critic_pendulum.py`      | A2C on Pendulum-v1 (Gaussian policy + VecNormalize)                                                | `/chapter07_actor_critic/pendulum`                                                               |
|     |                                            | `actor_critic_bipedalwalker.py` | A2C on BipedalWalker-v3                                                                            | `/chapter07_actor_critic/bipedalwalker`                                                          |
|     |                                            | `render_pendulum.py`            | Renders a trained Pendulum A2C replay as GIF                                                       | `/chapter07_actor_critic/pendulum`                                                               |
|     |                                            | `render_bipedalwalker.py`       | Renders trained BipedalWalker A2C replays as GIFs                                                  | `/chapter07_actor_critic/bipedalwalker`                                                          |
| 8   | `chapter08_ppo/`                           | `ppo_from_scratch.py`           | Pure-PyTorch PPO on CartPole                                                                       | —                                                                                                |
|     |                                            | `ppo_lunar_lander.py`           | SB3 PPO on LunarLander-v3                                                                          | —                                                                                                |
|     |                                            | `ppo_bipedal_walker.py`         | SB3 PPO on BipedalWalker-v3                                                                        | `/chapter08_ppo/ppo-bipedal-walker`                                                              |
|     |                                            | `gae_visualization.py`          | GAE λ/γ bias–variance demo                                                                         | —                                                                                                |
|     |                                            | `render_bipedal_walker.py`      | Renders trained BipedalWalker PPO replays as GIFs                                                  | `/chapter08_ppo/ppo-bipedal-walker`                                                              |
| 9   | `chapter09_continuous_control/`            | `sac_halfcheetah.py`            | SB3 SAC on HalfCheetah-v4                                                                          | —                                                                                                |
|     |                                            | `ppo_td3_sac_comparison.py`     | PPO vs TD3 vs SAC comparison                                                                       | `/chapter09_continuous_control/review`                                                           |
| 12  | `chapter12_exploration_marl_hierarchical/` | `multi_agent_marl.py`           | Independent Q-learning vs shared-policy GridWorld collection                                       | `/chapter12_exploration_marl_hierarchical/review`                                                |
| A   | `appendix_a_engineering/`                  | `debug_reward_hacking.py`       | Reward-hacking reproduction + diagnosis                                                            | —                                                                                                |
|     |                                            | `debug_training_collapse.py`    | Training-collapse diagnosis + 4-step debugging                                                     | —                                                                                                |

### LLM, agentic and frontier chapters

| Ch. | Directory                                  | Scripts                                                                                              | What each does                                                             | Docs area                |
| --- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------ |
| 13  | `chapter13_rlhf/` (+ `verl_gsm8k/`)        | `sft_pipeline.py`                                                                                    | SFT (supervised fine-tuning) pipeline                                      | `/chapter13_rlhf/`       |
|     |                                            | `reward_model_training.py`                                                                           | Reward model training                                                      | `/chapter13_rlhf/`       |
|     |                                            | `rlhf_ppo_train.py`                                                                                  | Simplified PPO-RLHF training loop                                          | `/chapter13_rlhf/`       |
| 14  | `chapter14_dpo/`                           | `0-download_model.py`, `1-generate_data.py`, `2-test_before.py`, `3-train_dpo.py`, `4-test_after.py` | Numbered end-to-end DPO flow: download → data → test → train → test        | `/chapter14_dpo/`        |
|     |                                            | `dpo_hands_on.py`, `dpo_math_reward.py`                                                              | DPO alignment/beta comparison; math-preference DPO with rule-based signals | `/chapter14_dpo/`        |
| 15  | `chapter15_grpo/` (+ `verl_code_rlvr/`)    | `grpo_mechanism.py`                                                                                  | GRPO core-mechanism demo                                                   | `/chapter15_grpo/`       |
|     |                                            | `grpo_math_reasoning.py`, `rule_based_reward.py`                                                     | Small GRPO math experiment; verifiable rule-based reward                   | `/chapter15_grpo/`       |
| 17  | `chapter17_prm_search/`                    | `tree_of_thought.py`                                                                                 | Tree-of-Thought search demo                                                | `/chapter17_prm_search/` |
| 19  | `chapter19_agentic/` (+ `agentic_system/`) | `tool_use_agent.py`, `multi_turn_rl.py`                                                              | Tool-calling policy training; multi-turn credit assignment (ORM vs PRM)    | `/chapter19_agentic/`    |
|     |                                            | `generate_synthetic_data.py`, `mini_deep_research_grpo.py`                                           | Synthetic trajectory data; mini deep-research GRPO example                 | `/chapter19_agentic/`    |
| 23  | `chapter23_vlm/`                           | `geometry_counting_dataset.py`, `multi_modal_reward.py`, `vlm_grpo_train.py`                         | Geometry-counting dataset; multi-modal reward; VLM GRPO training example   | `/chapter23_vlm/`        |

### Lab chapters (hands-on labs with unit tests)

Each of these directories pairs a docs `lab.md` page with runnable scripts
and a `test_chNN_lab.py` file that checks the lab's computations against
hand-worked values. Each directory also has its own `README.md` with run
instructions.

| Ch. | Directory                        | Scripts                                                                                            | What each does                                                                                   | Lab page                             |
| --- | -------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------ |
| 4   | `chapter04_classical_methods/`   | `dp_oracle.py`, `mc_td_control.py`, `cliffwalking.py`, `ope_is.py` (+ `tabular_common.py` helpers) | Exact DP oracles, MC/TD control, the Sutton–Barto CliffWalking comparison, off-policy evaluation | `/chapter04_classical_methods/lab`   |
| 10  | `chapter10_offline_rl/`          | `collect_datasets.py`, `train_offline.py`, `ope.py` (+ `offline_common.py` helpers)                | Dataset collection, offline training, off-policy evaluation                                      | `/chapter10_offline_rl/lab`          |
| 11  | `chapter11_imitation_meta_rl/`   | `bc_dagger_cartpole.py`, `gail_discriminator_cartpole.py`                                          | Behavior cloning vs DAgger; GAIL discriminator                                                   | `/chapter11_imitation_meta_rl/lab`   |
| 16  | `chapter16_reasoning/`           | `arith_cot.py`, `run_lab.py`                                                                       | Test-time compute on tiny arithmetic traces                                                      | `/chapter16_reasoning/lab`           |
| 18  | `chapter18_llm_rl_industrial/`   | `rollout_system.py`                                                                                | Rollout workers, learner updates, policy staleness                                               | `/chapter18_llm_rl_industrial/lab`   |
| 20  | `chapter20_swe_agents/`          | `sandbox.py`, `swe_policy.py`, `tasks.py`, `train.py`                                              | Toy code-agent sandbox, policy, tasks, training                                                  | `/chapter20_swe_agents/lab`          |
| 21  | `chapter21_deep_research/`       | `policies.py`, `train_research_agent.py`, `web_env.py`                                             | Multi-turn research with delayed reward                                                          | `/chapter21_deep_research/lab`       |
| 22  | `chapter22_computer_use/`        | `gui_env.py`, `train_gui_agent.py`                                                                 | Form-filling GUI agent                                                                           | `/chapter22_computer_use/lab`        |
| 24  | `chapter24_multimodal_frontier/` | `ddpo_toy.py`                                                                                      | DDPO fine-tuning of a tiny diffusion model                                                       | `/chapter24_multimodal_frontier/lab` |
| 25  | `chapter25_alignment_failures/`  | `gridworld.py`, `run_specification_gaming.py`, `run_eval_methodology.py` (+ `stats.py` helpers)    | Reward hacking and honest evaluation                                                             | `/chapter25_alignment_failures/lab`  |
| 26  | `chapter26_selfplay_frontiers/`  | `rps_self_play.py`, `kuhn_cfr.py`, `tictactoe_self_play.py`                                        | Self-play dynamics from RPS to Kuhn poker                                                        | `/chapter26_selfplay_frontiers/lab`  |

### Online experiments (upstream ModelScope material, kept as is)

`code/online-experiments/` holds the WalkingLabs × Hands-on Modern RL ModelScope
Studio notebooks (CartPole PPO, Gymnasium playground, ViZDoom, Atari, board
self-play, multi-agent games, MiniGrid, JAX games, ManiSkill, MineStudio, Unity
ML-Agents, AI2-THOR); see `code/online-experiments/README.md`. They install
their own runtimes and are not part of the pytest smoke suite.

## Notes

- Each chapter directory's `requirements.txt` holds that chapter's minimal
  dependencies; `code/requirements.txt` is the merged summary and
  `code/requirements.lock.txt` pins the full reproducible set installed by
  `uv pip install -r code/requirements.lock.txt`.
- LLM chapters use small models by default, but a GPU environment is still
  recommended for them.
- Smoke runs write under `<chapter>/runs/` (gitignored). Full training runs
  write under `<chapter>/output/`; `code/chapter01_cartpole/output/` keeps the
  checked-in reference CSV used by the Chapter 1 pages.
- `capture_frames.py` without `--smoke` requires a trained checkpoint and
  exits with a message if it is missing; it never silently substitutes a
  random policy for a trained one.
- SB3 logger keys read by the custom callbacks were verified against the
  installed SB3 source: PPO `train/entropy_loss` is the _negative_ policy
  entropy (negated before plotting), and SAC temperature uses
  `train/ent_coef` / `train/ent_coef_loss` (SB3 logs no policy entropy
  for SAC).
- Seeding in the Appendix A diagnosis scripts is seed-matched across
  scenarios: the same `--seed` is used for every scenario so that only the
  reward function or training configuration differs.
- Atari DQN behavior/figures need `ale-py`/ROMs (unavailable offline);
  those tests skip with a reason. Pokémon Red paths/addresses are carried
  over untouched and are unverifiable without a legal ROM.
