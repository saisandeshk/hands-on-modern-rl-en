---
title: '24.L Lab: DDPO Fine-Tuning of a Tiny Diffusion Model'
---

# 24.L Lab: DDPO Fine-Tuning of a Tiny Diffusion Model

In [24.4](./visual-generation-dancegrpo) a denoising chain became a reinforcement-learning
trajectory: the state is the current latent plus the timestep, the action is the next
sampled latent, and the reward model scores only the finished image. This lab shrinks that
idea until it fits on a laptop CPU. You will pretrain a tiny diffusion model on a 2D toy
distribution, then fine-tune its sampler with a DDPO-style policy gradient, with and
without a reference-model KL penalty, and watch reward, diversity, and KL move against
each other.

## What you will learn

- How a diffusion sampler maps onto states, actions, trajectories, and a terminal reward,
  so that REINFORCE applies to a generative model.
- How one scalar reward on the final sample is broadcast back to every denoising step
  through per-step Gaussian log-probabilities.
- Why a KL penalty toward the pretrained model trades reward against diversity, and what
  reward hacking looks like when the penalty is removed.
- How to report a generation-RL experiment honestly: reward, diversity, and KL together,
  never reward alone.

## Background

Read [24.4.2](./visual-generation-dancegrpo#_24-4-2-how-an-image-is-generated-from-noise)
first: it defines the state $s_t = (x_t, t, c)$, the denoising policy
$p_\theta(x_{t-1} \mid x_t, t, c)$, the trajectory $\tau = (x_T, \ldots, x_0)$, and the
terminal reward $R(\tau, c)$. Our lab drops the text condition $c$ (an unconditional 2D
model) but keeps everything else. The update follows
[24.4.3](./visual-generation-dancegrpo#_24-4-3-ddpo-passing-the-terminal-score-back-through-the-whole-denoising-trajectory):
advantages from a baseline, then the PPO-clipped loss and the KL term from
[24.4.4](./visual-generation-dancegrpo#_24-4-4-from-policy-gradients-to-a-stable-training-loss).
The group-relative advantage normalization is the miniature version of the DanceGRPO
advantage in
[24.4.5](./visual-generation-dancegrpo#_24-4-5-dancegrpo-why-generate-a-group-of-results-for-the-same-prompt).
The no-KL run demonstrates the lesson of
[24.5](./video-generation-modern): a single reward faithfully optimized is also
faithfully overfitted.

## Setup and how to run

You need the project virtualenv (Python 3.12 with torch, numpy, matplotlib, pytest).
All commands run from the repository root. Nothing is downloaded; the dataset is sampled
on the fly from a Gaussian mixture.

Smoke run (tiny config, well under 60 seconds on CPU):

```bash
.venv/bin/python code/chapter24_multimodal_frontier/ddpo_toy.py --smoke
```

Full lab run (about 10–11 minutes on CPU):

```bash
.venv/bin/python code/chapter24_multimodal_frontier/ddpo_toy.py --seed 0 --device cpu
```

Useful flags: `--device {auto,cpu,cuda}`, `--seed`, `--beta` (KL weight of the penalized
run; the comparison run always uses 0), `--rl-iters`, `--batch-rl`,
`--diffusion-steps`, `--skip-nokl`, `--outdir`. Outputs (scatter plots, training curves,
`metrics.csv`, `summary.json`) go to `runs/chapter24_multimodal_frontier/`, and a short
summary table is printed to stdout. Run the test suite with:

```bash
.venv/bin/pytest code/chapter24_multimodal_frontier -q
```

## Walkthrough of the key code

The full script is
[ddpo_toy.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter24_multimodal_frontier/ddpo_toy.py).
Three pieces carry the RL content; the rest is a small DDPM and plotting.

**1. Each denoising step is a Gaussian action.** The reverse step samples
$x_{t-1} \sim \mathcal{N}(\mu_\theta(x_t, t), \beta_t I)$ with a fixed variance, so its
log-probability is analytic:

```python
logps.append(gaussian_log_prob(x_next, mu, var))
```

`gaussian_log_prob`
([code](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter24_multimodal_frontier/ddpo_toy.py))
evaluates $\log p_\theta(x_{t-1} \mid x_t, t)$, the per-step policy term whose gradient
carries the reward signal. Because the variance is fixed and shared with the reference
model, the per-step KL also has a closed form,
$\lVert \mu_\theta - \mu_{\mathrm{ref}} \rVert^2 / (2\beta_t)$.

**2. The terminal reward is broadcast to all steps.** The rollout scores only the final
sample with a Gaussian bump around a low-density target region (our stand-in for an
aesthetic or alignment reward), then one update turns it into advantages shared by the
whole chain:

```python
advantages = compute_advantages(batch["rewards"])  # (r - mean) / std over the batch
loss = pg_loss(new_logps, old, advantages, clip_eps, beta, kl)
```

The whole rollout batch is the comparison group, exactly the group-relative idea from
DanceGRPO. `pg_loss`
([code](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter24_multimodal_frontier/ddpo_toy.py))
is the PPO-clipped objective $-\min(\rho_t \hat A, \mathrm{clip}(\rho_t) \hat A)$ averaged
over steps and trajectories, plus $\beta$ times the mean per-step KL. The reward model is
never differentiated: only its scalar scores enter through $\hat A$.

**3. Two fine-tuning runs isolate the KL trade-off.** `finetune` trains one copy of the
pretrained model with $\beta = 0.05$ and a second copy with $\beta = 0$ on the same
reward. `evaluate` reports mean reward, fraction of samples in the target region,
diversity (mean pairwise distance), and the mean per-step KL to the frozen pretrained
model — the four numbers every generation-RL report needs.

## Results from a real run

Full configuration, seed 0, CPU (631 s). Pretraining: 4000 steps, batch 512, $T = 30$
diffusion steps. RL: 60 iterations, 256 trajectories each, clip $\epsilon = 0.2$,
2 gradient epochs per batch, learning rate $3 \times 10^{-4}$.

| checkpoint               | mean reward | frac in target | diversity | KL/step to pretrained |
| ------------------------ | ----------- | -------------- | --------- | --------------------- |
| pretrained               | 0.022       | 0.001          | 3.23      | 0.000                 |
| RL + KL ($\beta = 0.05$) | 0.648       | 0.756          | 1.98      | 0.222                 |
| RL, no KL ($\beta = 0$)  | 0.845       | 0.965          | 0.73      | 6.290                 |

The reward curves rise steadily in both runs (at iteration 20: 0.25 with KL vs 0.31
without; at iteration 40: 0.45 vs 0.67). Diversity tells the runs apart: it stays above
3.0 through iteration 40 with the KL penalty and ends at 1.98, while the unpenalized run
falls to 2.21 by iteration 30 and 0.73 by iteration 60. The KL estimate mirrors this:
0.23 per step with the penalty versus 6.29 without — a 27-fold gap. The scatter plots
(saved as `samples_before_after.png`) show the same story visually: the pretrained model
covers all three mixture modes, the KL run shifts most mass to the reward peak while
keeping a remnant of the original modes, and the no-KL run piles nearly every sample
onto the peak.

## What to look for and why it happens

- **Reward rises in both runs because the signal is dense in trajectory space, not in
  time.** Every step shares the same advantage, so all 30 denoising means get pushed in
  the direction of chains that ended near the target. The reward model can be a black box
  precisely because the gradient flows through the sampler's log-probabilities, never
  through the reward.
- **The KL run keeps diversity because the penalty prices every step's drift.**
  $\beta \times \mathrm{KL}$ is added per denoising step, so moving the whole chain far
  from the reference costs $T$ times the per-step price. The policy settles where the
  marginal reward gain equals that price — most mass near the target, some mass left on
  the old modes.
- **The no-KL run collapses because nothing prices drift.** Once sampling near the peak
  earns above-average reward, the clipped update keeps raising those chains' probability
  until the sampler forgets the other modes. This is the same mechanism as the
  single-reward failures in the chapter (oily images from HPS-only training, static videos
  from quality-only rewards): the optimizer is not misbehaving, it is executing the
  objective it was given.
- **The baseline matters for variance, not for direction.** Subtracting the batch mean
  leaves the expected gradient unchanged (the probabilities of all trajectories sum to
  1. while shrinking update noise; dividing by the batch std keeps step sizes stable as
     rewards shift upward during training.

## Extension exercises

1. **Sweep the KL weight.** Run with `--beta 0.0 --skip-nokl`, `0.02`, `0.1`, `0.3` and
   plot final reward versus diversity. Hint: you are tracing a Pareto frontier; expect
   reward to saturate while diversity keeps falling as $\beta \to 0$.
2. **Move the reward peak.** Change `TARGET_CENTER` to the valley between the modes
   (e.g. `(0.0, -0.3)`) and rerun. Hint: the KL cost of reaching it is smaller now, so
   the two runs should differ less — a case where alignment and the prior agree.
3. **Break the reward on purpose.** Replace the bump with a narrow spike
   (`REWARD_WIDTH = 0.15`) and compare. Hint: a sharper peak is harder to find by chance,
   so early advantages are noisier; watch whether the KL run still finds it.
4. **Ablate the PPO clip.** Try `--clip-eps 1.0` (effectively REINFORCE) versus `0.05`.
   Hint: with only 256 trajectories per batch, large ratios make a few lucky chains
   dominate; the KL and diversity curves reveal the instability first.
5. **Ablate advantage normalization.** Standardize with the mean only (edit
   `compute_advantages` to skip the std division). Hint: as mean reward climbs from 0.02
   to 0.6, the raw advantage scale drifts; check whether a fixed learning rate still
   behaves late in training.
6. **Rerank instead of fine-tuning.** Sample 32 chains from the frozen pretrained model,
   keep the highest-reward sample, and compare its reward/diversity against the fine-tuned
   policy's single samples. Hint: this is inference-time reranking from
   [24.4.7](./visual-generation-dancegrpo#_24-4-7-a-reward-can-be-used-at-training-time-or-at-inference-time);
   it costs 32 samples per image and never collapses — when is that the better deal?
