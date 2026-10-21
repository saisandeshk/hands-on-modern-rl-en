# Code Index

Companion code for _Hands-on Modern RL_ (English). This repository is a fork of
[walkinglabs/hands-on-modern-rl](https://github.com/walkinglabs/hands-on-modern-rl),
used under CC BY-NC-SA 4.0; attribution to WalkingLabs is kept wherever it appears.

`code/` has one directory per book chapter (`code/chapter05_dqn` belongs to Chapter 5),
plus `code/device_utils.py` (shared torch-device helpers), `code/online-experiments/`
(upstream ModelScope companion notebooks, kept as is), `code/pytest.ini` and
`code/conftest.py` (test configuration). Each chapter directory has its own
`requirements.txt` with that chapter's minimal dependencies.

## Environment setup

```bash
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Full dependency set, pinned for reproducibility (recommended)
uv pip install -r code/requirements.lock.txt

# ... or only the dependencies for one chapter
uv pip install -r code/chapter05_dqn/requirements.txt
```

The project virtualenv used by CI/reviewers is `.venv` (Python 3.12, torch,
gymnasium, stable-baselines3, transformers, trl, pytest). Do not edit
`code/requirements.lock.txt`. You cannot install packages at review time: scripts
that need an unavailable package (Atari ROMs via `ale-py`, `pyboy`, `swanlab`)
import it lazily, skip the corresponding test with a clear reason, and keep
working without it.

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

There is one `test_smoke.py` per chapter directory: it runs each script of that
chapter with `--smoke` in a subprocess and asserts it exits 0 (and, where cheap,
that it produced its output). Render / frame-capture helpers are also covered:
they are importable and work in smoke mode with `rgb_array` rendering. Scripts
that cannot run here (Atari ROMs, `pyboy`) are skipped with an explicit reason.

## Chapter code overview

Docs pages are root-relative links without the `.md` extension, e.g.
`/chapter05_dqn/from-q-to-dqn`. The “Docs page” column names the page that
covers or runs the script (via a runnable command or an explicit
reference); scripts marked “(unreferenced — suggested)” are not covered
from any docs page yet, with a suggestion of where each belongs.

| Ch. | Directory                                  | Scripts                         | What each does                                                                                                                                              | Docs page                                                                                        |
| --- | ------------------------------------------ | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1   | `chapter01_cartpole/`                      | `1-ppo_cartpole.py`             | SB3 PPO trains CartPole, logs SwanLab curves (skipped if `swanlab` is missing)                                                                              | `/chapter01_cartpole/principles`                                                                 |
|     |                                            | `2-pytorch_ppo.py`              | Pure PyTorch PPO: Actor-Critic, GAE, clipping                                                                                                               | `/chapter01_cartpole/principles`, `/chapter01_cartpole/metrics`, `/chapter01_cartpole/training`  |
|     |                                            | `plot_curves.py`                | Plots reviewable metric curves from the exported CSV                                                                                                        | `/chapter01_cartpole/metrics`, `/chapter01_cartpole/training`                                    |
|     |                                            | `capture_frames.py`             | Captures real Gymnasium rendered frames (trained model; `--smoke` uses a random policy)                                                                     | `/chapter01_cartpole/training`                                                                   |
| 2   | `chapter02_rl_basics/`                     | `two_armed_bandit.py`           | Two-armed bandit: random vs greedy vs ε-greedy vs UCB (unreferenced — suggested: `/chapter02_rl_basics/bandit`)                                             | —                                                                                                |
| 3   | `chapter03_value_functions/`               | `bellman_equation_verify.py`    | Numerical check of the Bellman expectation/optimality equations (unreferenced — suggested: `/chapter03_value_functions/value-bellman`)                      | —                                                                                                |
|     |                                            | `gridworld_q_learning.py`       | 4×4 GridWorld value iteration + Q-learning, exports SVGs/CSV/JSON                                                                                           | `/chapter03_value_functions/value-experiment`                                                    |
| 5   | `chapter05_dqn/`                           | `dqn_cartpole.py`               | From-scratch DQN on CartPole (unreferenced — suggested: `/chapter05_dqn/from-q-to-dqn` or `/chapter05_dqn/dqn-components`)                                  | —                                                                                                |
|     |                                            | `double_dqn_cartpole.py`        | DQN vs Double DQN comparison (unreferenced — suggested: `/chapter05_dqn/dqn-family`)                                                                        | —                                                                                                |
|     |                                            | `dqn_gym_sb3.py`                | SB3 DQN on discrete Gym tasks (CartPole, MountainCar, LunarLander)                                                                                          | `/chapter05_dqn/lunar-lander`                                                                    |
|     |                                            | `dqn_atari_sb3.py`              | SB3 DQN on Atari (needs `ale-py`/ROMs; skipped without them)                                                                                                | `/chapter05_dqn/visual-game-projects`                                                            |
|     |                                            | `dqn_pokemon_red_pyboy.py`      | SB3 DQN on early Pokémon Red exploration (needs a legal ROM; `--smoke` is a no-ROM wiring check)                                                            | `/chapter05_dqn/visual-game-projects`                                                            |
|     |                                            | `export_dqn_curves.py`          | Exports DQN eval curves as docs figures                                                                                                                     | `/chapter05_dqn/lunar-lander`, `/chapter05_dqn/visual-game-projects`                             |
|     |                                            | `render_atari.py`               | Renders a trained Atari DQN model as GIF (needs ROMs; skipped without them)                                                                                 | `/chapter05_dqn/visual-game-projects`                                                            |
|     |                                            | `render_lunarlander.py`         | Renders a trained LunarLander DQN as GIF (unreferenced — the split-episode variant below is the one the docs use; suggested: `/chapter05_dqn/lunar-lander`) | —                                                                                                |
|     |                                            | `render_lunarlander_split.py`   | Renders one GIF per LunarLander episode                                                                                                                     | `/chapter05_dqn/lunar-lander`                                                                    |
| 6   | `chapter06_policy_gradient/`               | `reinforce_cartpole.py`         | From-scratch REINFORCE on CartPole                                                                                                                          | `/chapter06_policy_gradient/reinforce`, `/chapter06_policy_gradient/cartpole`                    |
|     |                                            | `reinforce_with_baseline.py`    | REINFORCE vs REINFORCE + value baseline                                                                                                                     | `/chapter06_policy_gradient/baseline-experiment`, `/chapter06_policy_gradient/cartpole-baseline` |
|     |                                            | `actor_critic_cartpole.py`      | Online one-step Actor-Critic on CartPole (unreferenced — suggested: `/chapter06_policy_gradient/cartpole` or `/chapter07_actor_critic/actor-critic`)        | —                                                                                                |
|     |                                            | `render_cartpole_baseline.py`   | Trains both REINFORCE variants and renders GIFs                                                                                                             | `/chapter06_policy_gradient/baseline-experiment`, `/chapter06_policy_gradient/cartpole-baseline` |
| 7   | `chapter07_actor_critic/`                  | `actor_critic_pendulum.py`      | A2C on Pendulum-v1 (Gaussian policy + VecNormalize)                                                                                                         | `/chapter07_actor_critic/pendulum`                                                               |
|     |                                            | `actor_critic_bipedalwalker.py` | A2C on BipedalWalker-v3                                                                                                                                     | `/chapter07_actor_critic/bipedalwalker`                                                          |
|     |                                            | `render_pendulum.py`            | Renders a trained Pendulum A2C replay as GIF                                                                                                                | `/chapter07_actor_critic/pendulum`                                                               |
|     |                                            | `render_bipedalwalker.py`       | Renders trained BipedalWalker A2C replays as GIFs                                                                                                           | `/chapter07_actor_critic/bipedalwalker`                                                          |
| 8   | `chapter08_ppo/`                           | `ppo_from_scratch.py`           | Pure-PyTorch PPO on CartPole (unreferenced — suggested: `/chapter08_ppo/trust-region-clipping`)                                                             | —                                                                                                |
|     |                                            | `ppo_lunar_lander.py`           | SB3 PPO on LunarLander-v3 (unreferenced — suggested: a PPO section such as `/chapter08_ppo/ppo-bipedal-walker` or a LunarLander PPO page)                   | —                                                                                                |
|     |                                            | `ppo_bipedal_walker.py`         | SB3 PPO on BipedalWalker-v3                                                                                                                                 | `/chapter08_ppo/ppo-bipedal-walker`                                                              |
|     |                                            | `gae_visualization.py`          | GAE λ/γ bias–variance demo (unreferenced — suggested: `/chapter08_ppo/gae-reward-model`)                                                                    | —                                                                                                |
|     |                                            | `render_bipedal_walker.py`      | Renders trained BipedalWalker PPO replays as GIFs                                                                                                           | `/chapter08_ppo/ppo-bipedal-walker`                                                              |
| 9   | `chapter09_continuous_control/`            | `sac_halfcheetah.py`            | SB3 SAC on HalfCheetah-v4 (unreferenced — suggested: `/chapter09_continuous_control/ddpg`)                                                                  | —                                                                                                |
|     |                                            | `ppo_td3_sac_comparison.py`     | PPO vs TD3 vs SAC comparison (unreferenced — suggested: `/chapter09_continuous_control/ddpg`)                                                               | —                                                                                                |
| 12  | `chapter12_exploration_marl_hierarchical/` | `multi_agent_marl.py`           | Independent Q-learning vs shared-policy GridWorld collection (unreferenced — suggested: `/chapter12_exploration_marl_hierarchical/marl`)                    | —                                                                                                |
| A   | `appendix_a_engineering/`                  | `debug_reward_hacking.py`       | Reward-hacking reproduction + diagnosis (unreferenced — suggested: appendix A reward/evaluation pages)                                                      | —                                                                                                |
|     |                                            | `debug_training_collapse.py`    | Training-collapse diagnosis + 4-step debugging (unreferenced — suggested: `/appendix_a_engineering/training-debugging`)                                     | —                                                                                                |

### LLM chapters (maintained by another worker — described here, not edited)

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

### Online experiments (upstream ModelScope material, kept as is)

`code/online-experiments/` holds the WalkingLabs × Hands-on Modern RL ModelScope
Studio notebooks (CartPole PPO, Gymnasium playground, ViZDoom, Atari, board
self-play, multi-agent games, MiniGrid, JAX games, ManiSkill, MineStudio, Unity
ML-Agents, AI2-THOR); see `code/online-experiments/README.md`. They install
their own runtimes and are not part of the pytest smoke suite.

## Notes

- Each chapter directory's `requirements.txt` holds that chapter's minimal
  dependencies; `code/requirements.txt` is the merged summary.
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
