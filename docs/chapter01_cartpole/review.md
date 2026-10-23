---
title: 1.4 Summary and Exercises
outline: [2, 3]
---

# 1.4 Summary and Exercises

> **Learning path**: [1.1 Run CartPole](./principles) → [1.2 CartPole Principles](./metrics) → [1.3 PPO Training Visualization](./training) → **1.R Summary and Exercises**

> **Code and data for this section**: [Pure PyTorch PPO](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter01_cartpole/2-pytorch_ppo.py) · [Plotting script](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter01_cartpole/plot_curves.py) · [Raw training metrics CSV](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter01_cartpole/output/training_metrics_seed42.csv)

This chapter ran a first end-to-end reinforcement learning experiment: PPO on CartPole-v1, from raw interaction data to a trained policy that balances the pole for the full 500-step episode. This page collects what to take away and gives exercises to check that the mechanism is really understood.

## What You Should Now Be Able to Do

1. Describe the CartPole-v1 task precisely: the four-dimensional observation, the two discrete actions, the $+1$-per-step reward, and the three ways an episode ends (pole fell, cart out of bounds, 500-step time limit).
2. Explain what the Actor and the Critic each output, and why PPO needs both: the Actor decides action probabilities, the Critic supplies the baseline $V(s_t)$ that turns raw rewards into advantages.
3. Trace one PPO iteration in order: collect 2048 steps with the current policy, compute TD errors and GAE advantages, then run the clipped update over mini-batches.
4. Read a training curve correctly: distinguish stochastic training reward from deterministic independent evaluation, and per-iteration means from the rolling 100-episode average.
5. Diagnose a run with the four auxiliary metrics: policy entropy, value loss, approximate KL, and clip fraction, and say which training stage each one monitors.
6. Handle episode boundaries correctly in value computation: use $0$ after `terminated`, bootstrap with $V(s')$ after `truncated`, cut the GAE recursion at every reset, and carry unfinished episodes across rollout boundaries.
7. Reproduce the chapter's run yourself: train with `2-pytorch_ppo.py`, regenerate the curves with `plot_curves.py`, and render evaluation frames with `capture_frames.py`.

## Key Equations

- Policy: $\pi(a \mid s) = P(A_t = a \mid S_t = s)$ — the probability of each action in a given state.
- TD error: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ — how much better the step turned out than the Critic expected.
- GAE advantage: $A_t = \delta_t + \gamma\lambda\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$ — a smoothed credit signal blending the current and future TD errors.
- Probability ratio: $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)$ — how far the new policy has moved from the policy that collected the data.
- Clipped objective: $L = -\min(r_t A_t,\ \mathrm{clip}(r_t, 0.8, 1.2)\, A_t)$ — raise probabilities of above-baseline actions, but stop pushing once the ratio leaves $[0.8, 1.2]$. (Here $r_t$ abbreviates the ratio $r_t(\theta)$ above, and $r_i$ in the clip-fraction line below likewise abbreviates $r_i(\theta)$.)
- Value loss: $L_V = \frac{1}{|B|}\sum_{i \in B}(V(s_i) - \hat{G}_i)^2$ — the Critic's mean squared error against its return targets on a mini-batch.
- Clip fraction: $\mathrm{clipfrac} = \frac{1}{|B|}\sum_{i \in B}\mathbf{1}[|r_i - 1| > 0.2]$ — the share of samples hitting the clipping boundary.

## Common Pitfalls

1. **Confusing training reward with policy quality.** Training samples actions stochastically, so the training curve wiggles even for a good policy. Judge the final policy with deterministic independent evaluation (20 episodes here), and report the two numbers separately.
2. **Mixing up the two reward averages.** The rolling 100-episode average smooths across iterations; the per-iteration average reacts immediately but is noisy when few episodes complete (iteration 11 averages only 5 episodes). Know which column is plotted before concluding anything.
3. **Treating auxiliary metrics as objectives.** Falling entropy or a tiny value loss does not mean the task is solved: a policy can be confidently wrong, and a Critic can fit the wrong targets precisely. The episode reward and the independent evaluation decide; the four metrics only explain.
4. **Mishandling `truncated` as `terminated`.** Setting the bootstrap value to $0$ at the 500-step time limit throws away the value of a still-balanced state and biases every advantage near the boundary. Use $V(s')$ after truncation and cut the GAE recursion at the reset.
5. **Normalizing the wrong target.** Advantage normalization (zero mean, unit variance) is for the Actor's policy loss; the Critic must train against the unnormalized returns. Swapping them destabilizes the value fit.
6. **Reading too much into one seed.** Seed 42 reaches a mean reward of 500 by iteration 10, but that says nothing about other seeds. Compare algorithms with multiple seeds, identical training budgets, and identical evaluation protocols.

## Exercises

### 1. Observation bounds vs. termination thresholds

CartPole-v1 declares position bounds of $\pm 4.8$ and angle bounds of about $\pm 24^\circ$ in `observation_space`, yet an episode ends when the cart passes $\pm 2.4$ or the pole passes about $\pm 12^\circ$. Why are these two pairs of numbers different, and which pair actually terminates an episode?

::: details Solution
They serve different purposes. The `observation_space` bounds declare the range of values the environment may ever report; Gymnasium uses them to validate observations, not to end episodes. Termination is decided inside the `step` logic by the thresholds `x_threshold = 2.4` and `theta_threshold_radians` ($\approx 0.2095$ rad $\approx 12^\circ$). An episode ends when the second, tighter pair is crossed. In practice a run therefore never observes positions near $\pm 4.8$, because the episode stops at $\pm 2.4$ first.
:::

### 2. Episode reward arithmetic

An episode of CartPole-v1 survives 37 steps. What is its episode reward? If a rollout of 2048 steps contains only complete episodes averaging 21.35 steps each, about how many episodes ended in that rollout?

::: details Solution
The reward is $+1$ per step, so 37 steps earn exactly 37 points: episode reward equals episode length. For the rollout, $2048 / 21.35 \approx 95.9$, so about 96 episodes if every step belonged to a completed episode. (Iteration 1 of the saved run reports 91 completed episodes averaging 21.35 steps: $91 \times 21.35 = 1943$ steps, leaving $2048 - 1943 = 105$ steps in an episode still running when sampling stopped.)
:::

### 3. Policy entropy by hand

At the start of training the two actions each have probability about 0.5, and the policy entropy is close to $\ln 2 \approx 0.693$. Later, in some state the policy outputs $\pi(0 \mid s) = 0.3$, $\pi(1 \mid s) = 0.7$. Compute the entropy of this distribution and explain in one sentence what the drop from 0.693 means.

::: details Solution
For a discrete distribution $H = -\sum_a \pi(a \mid s) \ln \pi(a \mid s)$:

$$
H = -(0.3 \ln 0.3 + 0.7 \ln 0.7) \approx -(-0.361 - 0.250) \approx 0.611.
$$

The drop from 0.693 to 0.611 means the policy has become more decisive in this state, putting more mass on one action; entropy is maximized at the uniform distribution and falls as the distribution concentrates.
:::

### 4. TD error and GAE by hand

With $\gamma = 0.99$ and $\lambda = 0.95$, the Critic predicts $V(s_0) = 20.0$ and $V(s_1) = 21.0$, and the observed reward is $r_0 = 1$. The next step has TD error $\delta_1 = -0.5$ and the episode continues. Compute $\delta_0$ and the two-step GAE advantage $A_0 = \delta_0 + \gamma\lambda\delta_1$. Is action $a_0$ judged above or below baseline?

::: details Solution

$$
\delta_0 = r_0 + \gamma V(s_1) - V(s_0) = 1 + 0.99 \times 21.0 - 20.0 = 1 + 20.79 - 20.0 = 1.79.
$$

$$
A_0 = 1.79 + 0.99 \times 0.95 \times (-0.5) = 1.79 - 0.47025 \approx 1.32.
$$

$A_0 > 0$, so action $a_0$ is judged above baseline: PPO will raise its probability (subject to clipping).
:::

### 5. Clipped objective and clip fraction by hand

Take $\epsilon = 0.2$, advantage $A = 2.0$, and probability ratio $r = 1.3$. Compute the unclipped term, the clipped term, and the resulting policy-loss contribution. Then compute the clip fraction of a mini-batch whose ratios are $[0.7, 0.9, 1.0, 1.25, 1.5]$.

::: details Solution
Unclipped: $rA = 1.3 \times 2.0 = 2.6$. Clipped: $\mathrm{clip}(1.3, 0.8, 1.2) \times 2.0 = 1.2 \times 2.0 = 2.4$. The loss contribution is $-\min(2.6, 2.4) = -2.4$: clipping binds and stops this sample from pushing the policy further. For the batch, $|r - 1| > 0.2$ holds for $0.7$, $1.25$, and $1.5$ (three of five), so the clip fraction is $3/5 = 60\%$.
:::

### 6. Discounted value of a perfect episode

A trained policy scores the maximum 500 on every episode. With the chapter's $\gamma = 0.99$, compute the discounted return $G_0 = \sum_{t=0}^{499} \gamma^t$ of such an episode. Why does this number (rather than 500) set the scale the Critic's predictions must match?

::: details Solution
This is a geometric series with ratio $\gamma = 0.99$:

$$
G_0 = \frac{1 - 0.99^{500}}{1 - 0.99} \approx \frac{1 - 0.00657}{0.01} \approx 99.34.
$$

($0.99^{500} = e^{500 \ln 0.99} \approx e^{-5.025} \approx 0.00657$.) The Critic is trained against discounted returns, so a perfect episode looks like $\approx 99$, not 500, to the value network. That is why very small value losses late in training (for example $1.4\times 10^{-4}$ at iteration 40) are consistent with values in the tens, and why fitting "500" would be a scale error.
:::

### 7. Coding: reproduce the run and check the numbers

Run the chapter's training script with the book's settings ([Section 1.2](./metrics)), then open the saved CSV and verify three facts from Section 1.2: iteration 1 ends about 91 episodes near a mean reward of 21, iteration 10 reaches 500, and entropy falls from about 0.68 to about 0.42.

```bash
cd code/chapter01_cartpole
python 2-pytorch_ppo.py \
  --seed 42 \
  --iterations 40 \
  --steps-per-rollout 2048 \
  --swanlab-mode disabled \
  --log-csv output/rerun_check/metrics.csv
```

::: details Solution
After the run finishes, inspect the CSV (any spreadsheet tool or `python -c` with `csv` works). Expected pattern for seed 42: row 1 shows `completed_episodes` near 91 with `mean_episode_reward_rollout` near 21 (per-iteration average over short episodes from the near-random initial policy); by iteration 10 the per-iteration mean reaches 500.0 with only 4 completed episodes (the rollout contains about four full 500-step episodes, $2048 / 500 \approx 4$); the `entropy` column declines from $\approx 0.68$ toward $\approx 0.42$. The repository's `requirements.txt` only lower-bounds the library versions rather than pinning them, so exact digits may differ from the book's saved file if the installed versions differ, but the three qualitative facts (many short episodes early, 500 by about iteration 10, entropy decline) must hold. Write the check to a new directory (for example `output/rerun_check/metrics.csv` as above) rather than overwriting the archived `training_metrics_seed42.csv`; the script derives the model path from the CSV directory, so running in `output/` directly would also overwrite the saved `pytorch_ppo_cartpole.pth`. If iteration 1 shows far fewer episodes or entropy starts near 0, the Actor's output layer was probably mis-initialized (check the gain-0.01 orthogonal initialization).
:::

### 8. Coding: clipping range ablation

Run two short ablations that change only the clipping range, keeping seed 42 and everything else fixed. The script exposes the range through the update code (`clip_eps=0.2` in `ppo_update`); temporarily try `clip_eps=0.05` and `clip_eps=0.5`, training for 10 iterations each with separate `--log-csv` files, then compare the `clip_fraction` and `approx_kl` columns against the baseline.

::: details Solution
Suggested commands (edit `clip_eps` default in `ppo_update` or pass it through a small local edit, keeping the change to that one number):

```bash
cd code/chapter01_cartpole
python 2-pytorch_ppo.py --seed 42 --iterations 10 \
  --steps-per-rollout 2048 --swanlab-mode disabled \
  --log-csv output/ablation_clip005/metrics.csv   # with clip_eps=0.05
python 2-pytorch_ppo.py --seed 42 --iterations 10 \
  --steps-per-rollout 2048 --swanlab-mode disabled \
  --log-csv output/ablation_clip050/metrics.csv   # with clip_eps=0.5
```

What to look for: with `clip_eps=0.05` the allowed band is $[0.95, 1.05]$, so many more ratios fall outside it and the reported `clip_fraction` is much larger than the baseline's $\approx 14.5\%$ at iteration 1, while `approx_kl` per iteration shrinks because each update is constrained more tightly. With `clip_eps=0.5` (band $[0.5, 1.5]$) the reverse holds: clip fraction drops toward zero and `approx_kl` grows, since almost nothing gets clipped. Either way the diagnostic pair moves in opposite directions here, which is exactly the trade-off the clipped objective encodes: a tighter band means more clipping and smaller policy steps. (The pairing moves this way because the clip-fraction threshold itself moves with `clip_eps` — the metric counts `|r - 1| > clip_eps`. For a fixed threshold, a larger update normally raises both the clip fraction and the KL divergence, as in Chapter 8.) Remember to restore `clip_eps=0.2` afterwards. Back up `output/pytorch_ppo_cartpole.pth` first (or point `--log-csv` into a fresh directory): the script derives the model path from the CSV directory, so each ablation run in `output/` silently overwrites that saved model.
:::
