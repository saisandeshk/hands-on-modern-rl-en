---
title: 9.5 Summary and Exercises
---

# 9.5 Summary and Exercises

This page closes Chapter 9. It collects what the chapter taught, the formulas worth memorizing, the mistakes practitioners actually make, and a set of exercises with full solutions.

## What You Should Now Be Able to Do

After working through Sections 9.1–9.4, you should be able to:

1. Explain why DQN's $\arg\max_a Q(s, a)$ fails in continuous action spaces, and state how a deterministic policy plus the Deterministic Policy Gradient theorem removes the need for that maximization.
2. Implement the DDPG update from scratch: a TD target built with target actor and target critic, an actor loss that maximizes $Q(s, \mu(s))$, and soft target updates, and diagnose its three classic failure modes (overestimation, hyperparameter sensitivity, actor-critic feedback collapse).
3. Apply TD3's three stabilization patches (clipped double-Q with a minimum, delayed policy updates, target policy smoothing) and explain which DDPG failure each one targets.
4. Write down the maximum-entropy RL objective, derive how it changes the Bellman backup, and implement SAC's reparameterized actor loss, twin-critic target with the entropy term, and automatic temperature tuning, including the $\tanh$ squashing correction for bounded actions.
5. Quantify compounding model error with the Lipschitz recursion $e_{t+1} \le L e_t + \epsilon$, and explain why Dyna works for short horizons, why PETS models epistemic and aleatoric uncertainty separately, and why MBPO restricts itself to short rollouts from real states.
6. Compare model-free and model-based methods on matched real-environment budgets: sample efficiency, asymptotic performance, compute cost, and the role of baselines such as random shooting and the known-dynamics oracle.
7. Trace the AlphaZero → MuZero → Dreamer V3 line: what MCTS plus a learned value buys, what a latent dynamics model buys, and what a recurrent world model buys, including the RSSM prior/posterior structure and the purpose of symlog scaling.

## Key Equations

- **Deterministic Policy Gradient** (§9.1): $\nabla_\theta J = \mathbb{E}_{s \sim d^\mu}[\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q^\mu(s, a)|_{a=\mu_\theta(s)}]$. _Reading: move the action in the direction the critic says is uphill, then move the parameters to produce that action._
- **DDPG actor loss** (§9.1): $L_{\text{actor}} = -\mathbb{E}_s[Q_\phi(s, \mu_\theta(s))]$. _Reading: the actor is trained to output actions the current critic scores highly._
- **Soft target update** (§9.1): $\theta' \leftarrow \tau \theta + (1-\tau)\theta'$, typically $\tau = 0.005$. _Reading: drag the target network a fraction of a percent toward the online network each step so TD targets move slowly._
- **TD3 clipped double-Q target** (§9.2): $y = r + \gamma \min_{i=1,2} Q_{\phi_i'}(s', \mu_{\theta'}(s') + \text{clipped noise})$. _Reading: two critics cross-check each other and the pessimistic one wins, so overestimates cannot compound._
- **Maximum-entropy objective** (§9.2): $J(\pi) = \mathbb{E}[\sum_t \gamma^t (r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t)))]$. _Reading: maximize reward while staying stochastic; $\alpha$ prices exploration._
- **SAC reparameterized actor loss** (§9.2): $\mathcal{L}_{\text{actor}} = \mathbb{E}[\alpha \log \pi_\theta(a \mid s) - Q_\phi(s, a)]$ with $a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon$ (here $\mu_\theta$ is the Gaussian mean head, not the §9.1 deterministic policy). _Reading: push actions toward high Q and high entropy at once; noise is external so gradients flow._
- **Temperature loss** (§9.2): minimize $\mathbb{E}[-\alpha(\log \pi(a \mid s) + \mathcal{H}_0)]$ with target $\mathcal{H}_0 = -|\mathcal{A}|$. _Reading: raise $\alpha$ when the policy is too deterministic, lower it when it is too random._
- **Squashed log-probability** (§9.2): $\log \pi(a \mid s) = \log \mu(u \mid s) - \sum_i \log a_{\max} - \sum_i \log(1 - \tanh^2(u_i))$ for $a = a_{\max}\tanh(u)$ (the $\log a_{\max}$ term vanishes for $[-1, 1]$ bounds). _Reading: squashing compresses probability mass near the bounds, and the log-probability must account for the compression._
- **Compounding error** (§9.3): $e_{t+1} \le L e_t + \epsilon$ unrolls to $e_T \le \epsilon \sum_{i=0}^{T-1} L^i$ — about $1164\epsilon$ for $L = 1.1, T = 50$ and $170\epsilon$ for $L = 1.01, T = 100$. _Reading: slightly expansive dynamics turn small one-step errors into useless long rollouts._
- **Pessimistic imagined reward** (§9.3): $\tilde{r}(s, a) = \hat{r}(s, a) - \lambda u(s, a)$. _Reading: charge the planner for ensemble disagreement so it avoids regions the model is guessing about._
- **PUCT selection** (§9.4): $\text{PUCT}(a) = Q(s,a) + c_{\text{prior}} p_\theta(a \mid s)\sqrt{N(s)}/(1+N(s,a))$. _Reading: exploit high values, trust the prior, and bonus rarely tried actions._
- **Symlog** (§9.4): $\text{symlog}(x) = \text{sign}(x)\log(|x|+1)$. _Reading: near-linear at zero, logarithmic for large magnitudes, so one value head survives reward scales from $0$–$10$ to $0$–$1000$._

## Common Pitfalls

1. **Clipping SAC actions without correcting the log-probability.** Clipping a Gaussian sample to $[-a_{\max}, a_{\max}]$ piles mass on the boundaries while $\log \pi$ still describes the unclipped density. Use $\tanh$ squashing with the change-of-variables correction everywhere $\log \pi$ appears.
2. **Updating the actor as often as the critic in TD3-style methods.** The delayed update ($d = 2$) exists because gradients from an unconverged critic actively mislead the actor. If training oscillates, check the update ratio before touching the learning rate.
3. **Rolling the model out from its own predictions for too long.** Error grows geometrically when $L > 1$; the MBPO lesson is to restart every few steps from a real visited state. A "looks good for 3 steps" model is not a "good for 50 steps" model.
4. **Reporting model MSE instead of decision-relevant error.** A dynamics model with low one-step state error can still produce wrong rewards, wrong values, and exploited plans. Evaluate multi-step reward/value error on the current policy's state distribution.
5. **Comparing sample efficiency on different x-axes.** The fair budget is real environment steps; imagined samples and planner compute (e.g. PETS's ~25,000 model calls per action) must be reported separately, alongside a matched model-free baseline and a random-shooting control.
6. **Freezing the model while the policy keeps changing.** The new policy visits states the frozen model never saw, and the planner will exploit exactly those regions. Keep aggregating fresh on-policy data into model training.

## Exercises

### Exercise 1 — Why DQN Cannot Do Continuous Control

CartPole has 2 actions; a 7-joint arm has $a \in [-1, 1]^7$. Explain in two or three sentences why DQN's decision rule $a^* = \arg\max_a Q(s, a)$ breaks down for the arm, and name the two ingredients DDPG uses to avoid the $\arg\max$ entirely.

::: details Solution
DQN's rule requires evaluating $Q(s, a)$ for every candidate action and picking the best. With 2 discrete actions that is two forward passes; with $a \in [-1, 1]^7$ there are infinitely many candidates, so enumeration is impossible and even fine grid search would need exponentially many evaluations in 7 dimensions. DDPG avoids the $\arg\max$ with (1) a deterministic actor $\mu_\theta(s)$ that outputs one action directly instead of maximizing over actions, and (2) the Deterministic Policy Gradient, which trains that actor by backpropagating the critic's action-gradient $\nabla_a Q$ through $\mu_\theta$ rather than by searching over $a$.
:::

### Exercise 2 — Twin Critics by Hand

The true value of $(s', a')$ is $10$. Critic 1 estimates $12$, critic 2 estimates $9$. (a) What TD target does single-critic DDPG-style backup use, and what does TD3's clipped double-Q use (ignoring $r + \gamma$ scaling, i.e. comparing the $Q$ terms)? (b) Why is the TD3 choice safer for training stability?

::: details Solution
(a) A single critic uses its own estimate, here $12$ — an overestimate of $2$. TD3 uses $\min(12, 9) = 9$ — an underestimate of $1$. (b) Overestimation is dangerous because the actor actively seeks actions the critic scores highly: an overestimated action attracts the policy, the policy collects data there, and the critic's error is reinforced — a positive-feedback collapse. Underestimation is conservative: the policy merely undervalues a good action slightly, which may slow learning but does not let inflated values steer the policy off a cliff. Taking the minimum across two independently initialized critics makes systematic overestimation unlikely, since both critics would have to overestimate the same action.
:::

### Exercise 3 — Compounding Error by Hand

Assume one-step model error $\epsilon = 0.02$ and the bound $e_T \le \epsilon\sum_{i=0}^{T-1} L^i$. (a) Compute the bound for $L = 1.1$, $T = 50$. (b) Compute it for $L = 1$, $T = 100$. (c) With $\epsilon = 0.01$ and $L = 1.1$, after how many steps does the bound first exceed $1.0$?

::: details Solution
(a) $(1.1^{50} - 1)/0.1 \approx 1163.9$, so $e_{50} \le 0.02 \times 1163.9 \approx 23.3$. A 50-step imagined rollout is pure noise. (b) For $L = 1$ the sum is $T$, so $e_{100} \le 0.02 \times 100 = 2.0$ — linear growth. (c) Solve $0.01 \times (1.1^T - 1)/0.1 > 1$, i.e. $1.1^T > 11$, i.e. $T > \ln 11 / \ln 1.1 \approx 2.3979/0.0953 \approx 25.2$. Check integers: at $T = 25$ the bound is $0.01 \times 98.35 \approx 0.98 < 1$, at $T = 26$ it is $0.01 \times 109.18 \approx 1.09 > 1$. So errors exceed unit scale after just 26 steps — the quantitative reason to keep model rollouts short (MBPO uses 5).
:::

### Exercise 4 — PUCT by Hand

A search node was visited $N(s) = 50$ times, with $c_{\text{prior}} = 1$. Action A: $Q = 0.6$, prior $p = 0.5$, $N(s, A) = 20$. Action B: $Q = 0.3$, prior $p = 0.2$, $N(s, B) = 2$. Compute both PUCT scores and say which action the next simulation explores.

::: details Solution
$\sqrt{50} \approx 7.071$. PUCT(A) $= 0.6 + 0.5 \times 7.071/21 \approx 0.6 + 0.168 = 0.768$. PUCT(B) $= 0.3 + 0.2 \times 7.071/3 \approx 0.3 + 0.471 = 0.771$. Action B wins ($0.771 > 0.768$) despite a much lower $Q$: it has been tried only twice and carries a decent prior, so its exploration bonus outweighs A's value advantage. This mirrors the worked example in §9.4 — PUCT spends simulations on actions that are promising but under-explored.
:::

### Exercise 5 — Derive the Geometric Error Sum

Starting from $e_{t+1} \le L e_t + \epsilon$ with $e_0 = 0$, prove by unrolling that $e_T \le \epsilon \sum_{i=0}^{T-1} L^i$, and evaluate the closed form for $L < 1$, $L = 1$, and $L > 1$.

::: details Solution
Unroll once: $e_1 \le \epsilon$. Then $e_2 \le L\epsilon + \epsilon$, $e_3 \le L(L\epsilon + \epsilon) + \epsilon = L^2\epsilon + L\epsilon + \epsilon$. By induction, $e_T \le \epsilon(1 + L + \dots + L^{T-1}) = \epsilon\sum_{i=0}^{T-1} L^i$. The geometric sum gives: for $L \ne 1$, $e_T \le \epsilon(L^T - 1)/(L - 1)$; for $L = 1$, $e_T \le \epsilon T$. For $L < 1$, letting $T \to \infty$ yields $e_\infty \le \epsilon/(1 - L)$ (bounded — contracting dynamics forgive model error, e.g. $10\epsilon$ for $L = 0.9$). For $L = 1$ growth is linear; for $L > 1$ it is exponential in $T$. The derivation assumes the same action sequence in model and reality and a uniform one-step bound $\epsilon$ — optimistic assumptions, so real compounding is typically worse.
:::

### Exercise 6 — Squashed Log-Probability by Hand

A 1-D SAC policy outputs mean $\mu_\theta(s) = 0$, $\sigma_\theta(s) = 1.0$, samples pre-action $u = -1.0$, and squashes with $a = \tanh(u)$ ($a_{\max} = 1$). Compute (a) the executed action, (b) the Gaussian log-density $\log \mathcal{N}(u; \mu, \sigma^2)$, (c) the squashing correction, and (d) the final $\log \pi(a \mid s)$.

::: details Solution
(a) $a = \tanh(-1.0) \approx -0.762$, inside $(-1, 1)$. (b) $\log \mathcal{N}(-1.0; 0, 1) = -\tfrac{1}{2}(-1.0)^2 - \log\sqrt{2\pi} \approx -0.5 - 0.9189 = -1.4189$. (c) $-\log(1 - \tanh^2(-1.0)) = -\log(1 - 0.5803) = -\log(0.4197) \approx +0.8676$. (d) $\log \pi(a \mid s) \approx -1.4189 + 0.8676 = -0.551$. (Here $a_{\max} = 1$, so the $-\log a_{\max}$ term is zero.) Note the correction is positive and large here: near saturation $\tanh$ compresses a wide range of $u$ into a narrow range of $a$, so the squashed action is far more likely than the raw Gaussian density suggests. Dropping the correction would overstate the policy's entropy and corrupt both the actor loss and temperature tuning.
:::

### Exercise 7 — Coding: PPO vs TD3 vs SAC on the Same Budget

Run the chapter comparison script from the repository root:

```bash
.venv/bin/python code/chapter09_continuous_control/ppo_td3_sac_comparison.py
```

It trains PPO, TD3, and SAC for 50,000 steps each (HalfCheetah-v4 when MuJoCo is installed, Pendulum-v1 otherwise) with matched seeds and network sizes, then prints a summary table and saves `output/ppo_td3_sac_comparison.png`. Report the final mean rewards and answer: (a) which algorithm leads at 50k steps and why does that ranking make sense given on-policy versus off-policy data use? (b) Why does the script warn that 50k steps is demonstration-only?

::: details Solution
On HalfCheetah-v4 (MuJoCo installed), the expected pattern at 50k steps is that the off-policy methods (SAC, TD3) clearly lead PPO — for example a run giving roughly SAC ≈ TD3 > PPO by a wide margin, with SAC usually marginally ahead of TD3. On the Pendulum-v1 fallback the gaps are smaller and PPO can be comparatively competitive, so always report which environment you ran. (a) PPO is on-policy: each of its 50k samples is used for a few epochs and then discarded, while TD3 and SAC store every transition in a replay buffer ($100{,}000$ capacity here) and re-use it: with one gradient update per environment step, the 50,000-step run performs 50,000 updates, each drawing a 256-sample batch — about 12.8M samples in total, so every stored transition is seen roughly 256 times on average. At a small fixed budget, reuse beats fresh-but-thin data; SAC additionally explores via its entropy bonus rather than fixed action noise. Exact numbers vary with seed and hardware — the robust finding is the gap between off-policy and on-policy, not SAC-versus-TD3 ordering at 50k. (b) 50k steps is ~5% of the ~1M steps these benchmarks use: learning curves have not separated asymptotically, rankings can flip with longer training (the chapter's 1M-step sketch has SAC near 12000, TD3 near 8000 and DDPG near 4000), and a single seed cannot support a serious claim. For conclusions, rerun with several seeds and the full budget.
:::

### Exercise 8 — Coding: Fixed versus Automatic Temperature

SAC's `ent_coef` can be fixed or automatic. On Pendulum-v1 (fast on CPU), train two SAC agents for 20,000 steps each with seed 0 — one with `ent_coef="auto"` and one with `ent_coef=0.2` fixed — keeping everything else identical, and record `train/entropy_coef` over time with a callback. Which run adapts its exploration, and what goes wrong with a fixed coefficient that is too large?

::: details Solution
Sketch (Stable Baselines 3):

```python
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

class EntropyCoefLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.coefs = []
        self.timesteps = []
    def _on_step(self) -> bool:
        self.coefs.append(self.model.logger.name_to_value.get("train/entropy_coef"))
        self.timesteps.append(self.num_timesteps)
        return True

for coef in ["auto", 0.2]:
    model = SAC("MlpPolicy", "Pendulum-v1", ent_coef=coef,
                learning_rate=3e-4, buffer_size=50_000, seed=0, verbose=0)
    logger = EntropyCoefLogger()
    model.learn(total_timesteps=20_000, callback=logger)
    print(coef, logger.coefs[::len(logger.coefs) // 5])
```

Expected result, treated as a hypothesis to test over several seeds rather than a guaranteed gap: with `ent_coef="auto"`, the logged coefficient starts near $1.0$ and drifts down as the policy sharpens, holding entropy near the target $-|\mathcal{A}| = -1$; the run typically converges to a stabilizing swing-up. With a fixed $0.2$, the entropy bonus stays on through late training, so the policy tends to stay more diffuse and often plateaus below the auto-tuned run on the same budget — but on Pendulum the per-step rewards are large and SB3's fixed-$0.2$ SAC can still reach near-optimal returns, so compare the two return curves over several seeds rather than assuming a large gap. The general lesson matches §9.2: a fixed $\alpha$ must be retuned per environment and per training phase, while automatic tuning tracks the target entropy throughout. (If fixed $0.2$ happens to suit Pendulum, repeat with $1.0$ fixed — the policy then barely exploits at all — to see the failure mode unambiguously.)
:::
