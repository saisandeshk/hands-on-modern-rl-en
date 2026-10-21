# Hands-on Modern RL (English study edition)

A practice-first reinforcement learning textbook: from CartPole, MDPs and Bellman equations through DQN, policy gradients and PPO, to RLHF, DPO, GRPO, reasoning models, agentic RL and multimodal RL. Each chapter pairs the mathematics with runnable code.

This repository is an English-only, single-source adaptation of [**Hands-On Modern RL**](https://github.com/walkinglabs/hands-on-modern-rl) by WalkingLabs, maintained for personal study. Compared with upstream:

- English is the only language and the site root (the Chinese originals and the `/en/` locale were removed after the English pages were re-synchronized with the latest Chinese text).
- Directory numbers match the displayed chapter numbers in both `docs/` and `code/` (e.g. Chapter 5 is `docs/chapter05_dqn/` and `code/chapter05_dqn/`).
- Links point at this fork: `https://github.com/saisandeshk/hands-on-modern-rl-en`.

> Upstream notes that the course was written with AI assistance and is not fully reviewed. Treat claims about specific papers and numbers with care; `planning/content-review-notes.md` tracks suspected errors found so far.

## Read the book locally

Requirements: Node.js 18+ and npm.

```bash
npm install
npm run dev:fast   # start the site without the asset optimization step (fast)
npm run dev        # optimize images first, then start the site
```

Open the URL VitePress prints (usually `http://localhost:5173`). Full-text search is built in (press `/`). Set `LOCAL_SEARCH=0` to disable it.

## Book structure

| Part                      | Ch. | Topic                                          | Pages                                                                                                                         | Code                                                                                            |
| :------------------------ | :-: | :--------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- |
| Preface                   |  –  | Introduction, history of RL, environment setup | [`docs/preface/`](docs/preface/introduction.md)                                                                               | –                                                                                               |
| I. Fundamentals           |  1  | Starting with CartPole                         | [`chapter01_cartpole`](docs/chapter01_cartpole/principles.md)                                                                 | [`code/chapter01_cartpole`](code/chapter01_cartpole/)                                           |
|                           |  2  | RL problems and basic definitions              | [`chapter02_rl_basics`](docs/chapter02_rl_basics/bandit.md)                                                                   | [`code/chapter02_rl_basics`](code/chapter02_rl_basics/)                                         |
|                           |  3  | Value functions and Bellman equations          | [`chapter03_value_functions`](docs/chapter03_value_functions/value-bellman.md)                                                | [`code/chapter03_value_functions`](code/chapter03_value_functions/)                             |
|                           |  4  | Classical RL methods (DP, MC, TD)              | [`chapter04_classical_methods`](docs/chapter04_classical_methods/dp-mc-td.md)                                                 | –                                                                                               |
| II. Deep RL               |  5  | Deep Q-networks                                | [`chapter05_dqn`](docs/chapter05_dqn/from-q-to-dqn.md)                                                                        | [`code/chapter05_dqn`](code/chapter05_dqn/)                                                     |
|                           |  6  | Policy gradient methods                        | [`chapter06_policy_gradient`](docs/chapter06_policy_gradient/reinforce.md)                                                    | [`code/chapter06_policy_gradient`](code/chapter06_policy_gradient/)                             |
|                           |  7  | Actor-critic methods                           | [`chapter07_actor_critic`](docs/chapter07_actor_critic/advantage-function.md)                                                 | [`code/chapter07_actor_critic`](code/chapter07_actor_critic/)                                   |
|                           |  8  | TRPO and PPO                                   | [`chapter08_ppo`](docs/chapter08_ppo/ppo-bipedal-walker.md)                                                                   | [`code/chapter08_ppo`](code/chapter08_ppo/)                                                     |
|                           |  9  | Continuous control and model-based RL          | [`chapter09_continuous_control`](docs/chapter09_continuous_control/ddpg.md)                                                   | [`code/chapter09_continuous_control`](code/chapter09_continuous_control/)                       |
| III. Advanced RL          | 10  | Offline RL                                     | [`chapter10_offline_rl`](docs/chapter10_offline_rl/offline-data-distribution-shift.md)                                        | –                                                                                               |
|                           | 11  | Imitation, inverse and meta-RL                 | [`chapter11_imitation_meta_rl`](docs/chapter11_imitation_meta_rl/bc-dagger.md)                                                | –                                                                                               |
|                           | 12  | Exploration, multi-agent and hierarchical RL   | [`chapter12_exploration_marl_hierarchical`](docs/chapter12_exploration_marl_hierarchical/intrinsic-motivation-exploration.md) | [`code/chapter12_exploration_marl_hierarchical`](code/chapter12_exploration_marl_hierarchical/) |
| IV. LLM post-training     | 13  | The RLHF pipeline                              | [`chapter13_rlhf`](docs/chapter13_rlhf/base-model-to-assistant.md)                                                            | [`code/chapter13_rlhf`](code/chapter13_rlhf/)                                                   |
|                           | 14  | Preference optimization and DPO                | [`chapter14_dpo`](docs/chapter14_dpo/dpo-objective-derivation.md)                                                             | [`code/chapter14_dpo`](code/chapter14_dpo/)                                                     |
|                           | 15  | GRPO and verifiable rewards                    | [`chapter15_grpo`](docs/chapter15_grpo/grpo-practice-and-mechanism.md)                                                        | [`code/chapter15_grpo`](code/chapter15_grpo/)                                                   |
|                           | 16  | Reasoning models and test-time compute         | [`chapter16_reasoning`](docs/chapter16_reasoning/emergence-and-o1.md)                                                         | –                                                                                               |
|                           | 17  | Process rewards and inference-time search      | [`chapter17_prm_search`](docs/chapter17_prm_search/outcome-vs-process.md)                                                     | [`code/chapter17_prm_search`](code/chapter17_prm_search/)                                       |
|                           | 18  | Industrial LLM RL engineering                  | [`chapter18_llm_rl_industrial`](docs/chapter18_llm_rl_industrial/single-machine-to-industrial.md)                             | –                                                                                               |
| V. Agentic RL             | 19  | Agentic RL systems                             | [`chapter19_agentic`](docs/chapter19_agentic/overview.md)                                                                     | [`code/chapter19_agentic`](code/chapter19_agentic/)                                             |
|                           | 20  | RL for code agents                             | [`chapter20_swe_agents`](docs/chapter20_swe_agents/swe-bench-and-rlvr.md)                                                     | –                                                                                               |
|                           | 21  | Deep research and browser agents               | [`chapter21_deep_research`](docs/chapter21_deep_research/browser-rl-harness.md)                                               | –                                                                                               |
|                           | 22  | Computer use and GUI agents                    | [`chapter22_computer_use`](docs/chapter22_computer_use/training.md)                                                           | –                                                                                               |
| VI. Multimodal RL         | 23  | Vision-language model RL                       | [`chapter23_vlm`](docs/chapter23_vlm/vlm-challenges.md)                                                                       | [`code/chapter23_vlm`](code/chapter23_vlm/)                                                     |
|                           | 24  | Audio, embodied (VLA) and visual generation    | [`chapter24_multimodal_frontier`](docs/chapter24_multimodal_frontier/audio-reward-design.md)                                  | –                                                                                               |
| VII. Safety and frontiers | 25  | Reward hacking and RL evaluation               | [`chapter25_alignment_failures`](docs/chapter25_alignment_failures/classical-failures.md)                                     | –                                                                                               |
|                           | 26  | Self-play, scaling and research frontiers      | [`chapter26_selfplay_frontiers`](docs/chapter26_selfplay_frontiers/self-play-outlook/index.md)                                | –                                                                                               |
| Appendices                |  A  | Training debugging and engineering             | [`appendix_a_engineering`](docs/appendix_a_engineering/training-debugging.md)                                                 | [`code/appendix_a_engineering`](code/appendix_a_engineering/)                                   |
|                           |  B  | Core algorithm implementations                 | [`appendix_b_algorithms`](docs/appendix_b_algorithms/sft-kl.md)                                                               | –                                                                                               |
|                           |  C  | Learning resources and reference tables        | [`appendix_c_resources`](docs/appendix_c_resources/learning-resources.md)                                                     | –                                                                                               |
|                           |  D  | Mathematical foundations                       | [`appendix_d_math`](docs/appendix_d_math/linear-algebra-basics.md)                                                            | –                                                                                               |

Suggested route: read Chapters 1–9 in order. For LLM post-training, Chapters 6–8 (policy gradients, actor-critic, PPO) are the prerequisites for Chapters 13–18. Chapters 25–26 apply to every method in the book. Use the appendices as references.

## Run the code

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r chapter01_cartpole/requirements.txt   # or code/requirements.txt for everything
python chapter01_cartpole/1-ppo_cartpole.py
```

Each `code/chapterNN_*` directory has its own `requirements.txt`. Chapters 1–9 run on a laptop CPU. The LLM chapters (13–15, 19, 23) use Qwen2.5-0.5B-class models and want one GPU. The veRL scripts need a datacenter GPU (one of them needs 8). See [`code/README.md`](code/README.md) for the full index.

`code/online-experiments/` and `modelscope-space/` contain the upstream online training notebooks and Gradio apps (built for ModelScope). They are kept as is.

## Repository layout

```text
docs/                  VitePress book (English, site root)
  .vitepress/          config.mjs (nav + sidebar), theme
  chapterNN_*/         one directory per chapter; images/ live next to the pages
  appendix_[a-d]_*/    appendices
code/                  runnable experiments, numbered like the chapters
scripts/               maintenance: checks, sitemap, asset optimization, PDF/EPUB builds
planning/              upstream outline notes and this fork's review notes
```

## Maintenance commands

```bash
npm run verify             # format check, lint, English checks, code-link check, build
npm run verify:english     # CJK / punctuation / known-bad-phrase checks on docs/
npm run verify:code-links  # every link into code/ or docs/ on GitHub points at a file that exists
npm run fix:headings       # sync page heading numbers with the sidebar in config.mjs
npm run build && npm run preview
```

## License and attribution

The course content is © WalkingLabs and contributors, licensed under [CC BY-NC-SA 4.0](LICENSE). This adaptation is distributed under the same license, for non-commercial use, with changes as described above. If you cite the material, cite the original:

```bibtex
@misc{hands_on_modern_rl,
  title        = {Hands-On Modern RL: Practice-first reinforcement learning from CartPole to LLM post-training and agentic systems},
  author       = {WalkingLabs},
  year         = {2026},
  howpublished = {\url{https://github.com/walkinglabs/hands-on-modern-rl}},
  note         = {Open courseware repository}
}
```
