---
title: 1.1 Run CartPole
---

# 1.1 Run CartPole

> **Code for this section**: [SB3 version](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter01_cartpole/1-ppo_cartpole.py) · [Pure PyTorch version](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter01_cartpole/2-pytorch_ppo.py)

In this section we complete our first experiment: run one PPO training session on CartPole and watch how the reward curve changes.

## The First Training Run

CartPole is the classic introductory task in reinforcement learning. A pole is attached to a cart by a joint, and at each step the controller can only choose to push the cart left or right. The goal is to keep the pole upright for as long as possible.

The task looks simple, but balancing is not easy. Once the pole starts to tilt, gravity makes it fall faster and faster. The controller has to reverse its force continuously, ahead of the pole's fall, and each time it can only push fully left or fully right, with no intermediate strength to choose from.

Training it needs no special equipment. An ordinary laptop CPU finishes in about 30 seconds.

![The CartPole inverted-pendulum environment: the cart moves left and right to keep the pole balanced upright](./images/cartpole.gif)

<div class="figure-caption">Figure 1-1: The CartPole-v1 environment. The agent moves the cart left and right to keep the pole upright. Source: <a href="https://gymnasium.farama.org/environments/classic_control/cart_pole/" target="_blank" rel="noopener noreferrer">Gymnasium</a></div>

<OnlineTraining studios="cartpole" compact />

If you want to view the code cell by cell, or run the complete script in a terminal, you can use the following two options:

<NavGrid>
  <NavCard href="https://modelscope.cn/my/mynotebook" title="Online development environment" description="Start a CPU environment, pull the course repository, then open notebooks/cartpole-ppo.ipynb and run it cell by cell." />
  <NavCard href="https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole/file/view/master/train.py" title="Local or cloud terminal" description="Run python train.py --timesteps 30000. For local environment installation and the running workflow, see Section 1.3." />
</NavGrid>

At the start of training, the policy in the program is still a "blank sheet": it pushes left and right with almost equal probability, the pole falls after about 20 steps on average, and the reward curve oscillates around 20.

As training proceeds, the curve climbs step by step and finally settles near 500. This is the step limit of an episode, meaning the cart can hold on for the entire episode.

![Measured frames of the trained policy in Gymnasium CartPole-v1](./images/cartpole_frames_seed42.png)

<div class="figure-caption">Figure 1-2: One evaluation episode of the trained policy in CartPole-v1. The five frames come from steps 0, 125, 250, 375, and 500 of the same episode. The episode reached the 500-step limit, showing that the policy can keep controlling the cart.</div>

At this point the training runs, and the learning phenomenon can be observed. The mechanism behind it, namely what information the environment gives the agent, how the agent makes decisions, and how PPO uses a batch of interaction data to improve the policy, comes in the next section.
