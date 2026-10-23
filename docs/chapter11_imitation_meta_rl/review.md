---
title: 11.4 Summary and Exercises
---

# 11.4 Summary and Exercises

## What You Should Now Be Able to Do

- Train behavior cloning (BC) as maximum likelihood on expert state-action pairs, and choose the right likelihood (cross-entropy, Gaussian, multimodal) for the action space.
- Explain covariate shift: why a small per-step error $\epsilon$ compounds to $O(T^2\epsilon)$ cost over horizon $T$, and compute $(1-\epsilon)^T$ survival probabilities by hand.
- Describe how DAgger fixes the state distribution by rolling out the current policy and asking the expert to relabel visited states, including the role of the $\beta$-mixing schedule.
- State why inverse reinforcement learning (IRL) is ill-posed (many rewards make the expert optimal) and how maximum-entropy IRL selects one reward via trajectory softmax plus a partition function.
- Derive the MaxEnt IRL gradient as expert feature expectation minus model feature expectation, and explain why the partition function $Z$ forces an inner RL loop.
- Explain GAIL as occupancy matching with a discriminator: write the discriminator and generator objectives, the optimal discriminator $D^*$, and the implicit reward $-\log(1-D)$.
- Compare MAML, RL², and PEARL by where adaptation lives (parameters, hidden state, task posterior) and write the MAML inner/outer updates.
- Map the chapter onto LLM post-training precisely where the equivalence holds (SFT $=$ BC) and loosely where it is only an analogy (reward model $\approx$ IRL, DPO $\approx$ implicit GAIL, few-shot prompting $\approx$ in-context meta-RL).

## Key Equations

- **BC negative log-likelihood:** $\mathcal{L}_{BC}(\theta) = -\mathbb{E}_{(s,a)\sim\mathcal{D}_{\text{expert}}}[\log\pi_\theta(a\mid s)]$ — raise the probability of the expert's action. (Classification for discrete actions; Gaussian NLL or MSE for continuous actions.)
- **Compounding error:** $P(\text{no mistake}) = (1-\epsilon)^T$; BC cost scales as $O(T^2\epsilon)$, DAgger improves it to $O(T\epsilon)$ — early mistakes change all later states, so cost snowballs quadratically without on-policy data.
- **MaxEnt IRL distribution:** $p(\tau\mid r_\psi) = Z^{-1}\exp(\sum_t r_\psi(s_t,a_t))$ — trajectories with higher total reward are exponentially more probable; $Z$ sums over all trajectories.
- **MaxEnt gradient:** $\nabla_\psi\mathcal{L} = \mathbb{E}_{\tau\sim\text{expert}}[\sum_t\nabla r_\psi] - \mathbb{E}_{\tau\sim p(\cdot\mid r_\psi)}[\sum_t\nabla r_\psi]$ (per-trajectory form; the full-batch gradient carries an overall factor $M = \lvert\mathcal{D}\rvert$) — push expert state-actions up, push current-policy state-actions down.
- **GAIL discriminator:** $\max_\phi\,\mathbb{E}_{\text{expert}}[\log D_\phi] + \mathbb{E}_{\pi_\theta}[\log(1-D_\phi)]$ — binary classifier for expert versus policy state-action pairs.
- **GAIL policy and optimal discriminator:** $\min_\theta\,\mathbb{E}_{\pi_\theta}[\log(1-D_\phi)] - \lambda\mathcal{H}(\pi_\theta)$ with $D^*(s,a) = p_{\text{expert}}/(p_{\text{expert}}+p_{\pi})$ — the implicit reward $-\log(1-D)$ is high where the expert goes and the policy does not.
- **MAML:** inner $\theta_i' = \theta - \alpha\nabla\mathcal{L}_{T_i}(\theta)$; outer $\min_\theta \mathbb{E}_{T_i}[\mathcal{L}_{T_i}(\theta_i')]$ — optimize the initialization for post-adaptation loss, backpropagating through the inner step.
- **PEARL:** $\mathcal{L} = -\mathbb{E}_{z\sim q_\phi}[\sum_t r(s_t,a_t,z)] + \beta D_{\mathrm{KL}}(q_\phi(z\mid\tau)\|p(z))$ — infer a task variable from context, regularized toward the prior.
- **SFT $=$ BC:** $\mathcal{L}_{\text{SFT}} = -\sum_t \log\pi_\theta(y_t\mid x, y_{<t})$ — next-token likelihood is BC with tokens as actions.

## Common Pitfalls

- **Treating BC error like i.i.d. test error.** A 1% per-step error is not a 1% episode failure rate: $(0.99)^{100} \approx 0.37$ episodes are error-free, so one third survive untouched and two thirds hit at least one mistake — and post-mistake states are out of distribution, where the error rate is higher.
- **Fitting a unimodal policy to a multimodal expert.** Averaging "turn left" and "turn right" demonstrations with MSE yields "drive straight into the obstacle". Check whether the same state admits several valid actions before choosing a Gaussian or deterministic head.
- **Mixing up GAIL signs and conventions.** Whether the implicit reward is $-\log(1-D)$, $\log D$, or $-\log D$ depends on whether the discriminator labels the expert as 1 and whether it outputs probabilities or logits. Always verify $D^* = 0.5$ means the discriminator is indifferent (it assigns equal probability to expert-like and policy-like pairs) and expert-like pairs give high reward under your convention. Note the reward value at $D^* = 0.5$ is not zero (e.g. $-\log(1-0.5) = 0.6931$); what is neutral there is the discriminator, not the reward value.
- **Calling every prompt an RL algorithm.** Few-shot prompting carries input-output examples without rewards; Algorithm Distillation carries reward-bearing histories ordered from poor to good. Only the latter even has the signal needed to implement policy improvement in context, and even then it must be verified with adaptation curves.
- **Expecting GAIL to work with a frozen discriminator or no environment interaction.** GAIL still needs on-policy rollouts against the learned reward signal and careful balancing of discriminator and policy updates; an overtrained discriminator gives vanishing gradients, an undertrained one gives noise.
- **Overclaiming the LLM correspondences.** SFT and BC share the exact conditional-likelihood objective, but reward models use pairwise preferences (not full trajectories like MaxEnt IRL) and DPO is preference optimization (not occupancy matching like GAIL). Keep "exact equivalence" and "conceptual analogy" separate.

## Exercises

### Exercise 1 — BC loss by hand

One expert sample has $\pi_1(a\mid s) = 0.7$ and $\pi_2(a\mid s) = 0.95$. Compute each loss $-\log \pi(a\mid s)$ (natural log). Then compute the mean NLL over three samples with probabilities $0.7$, $0.5$, $0.9$.

::: details Solution
$-\log 0.7 = 0.3567$, since $\log 0.7 = -0.3567$. $-\log 0.95 = 0.0513$, since $\log 0.95 = -0.0513$. Policy 2 agrees more with the expert, so its loss is smaller.

For the three samples: $-\log 0.5 = 0.6931$ and $-\log 0.9 = 0.1054$. The mean is $(0.3567 + 0.6931 + 0.1054)/3 = 1.1552/3 = 0.3851$.
:::

### Exercise 2 — Compounding error by hand

Suppose the per-step mistake rate on expert states is $\epsilon = 0.02$ and the horizon is $T = 50$. Compute the probability of zero mistakes, the probability of at least one mistake, and the orders $O(T\epsilon)$ and $O(T^2\epsilon)$ (drop constants, just evaluate $T\epsilon$ and $T^2\epsilon$).

::: details Solution
$P(\text{no mistake}) = (1-\epsilon)^T = 0.98^{50}$. Since $\log 0.98 = -0.020203$, $50 \times (-0.020203) = -1.0101$, and $e^{-1.0101} = 0.364$. So about 36% of episodes are clean and $1 - 0.364 = 0.636$ (about 64%) contain at least one mistake.

$T\epsilon = 50 \times 0.02 = 1.0$ (the DAgger-scale linear term) versus $T^2\epsilon = 2500 \times 0.02 = 50$ (the BC-scale quadratic term). The factor-of-50 gap is the snowball effect: after the first mistake the agent leaves the training distribution.
:::

### Exercise 3 — Why DAgger rolls out the student

DAgger rolls out the current (imperfect) policy with a $\beta$-mixing schedule, then asks the expert to label every visited state. Why not simply collect more expert-only trajectories instead? What goes wrong if $\beta = 1$ for all iterations?

::: details Solution
More expert-only data only thickens coverage of the expert state distribution $d_{\text{expert}}$. The failure states the policy actually visits, $d_{\pi_\theta}$, remain unlabeled, so covariate shift persists no matter how many expert demonstrations are added.

If $\beta = 1$ always, every rollout action comes from the expert, the visited states are again drawn from $d_{\text{expert}}$, and DAgger degenerates to plain BC on a growing expert dataset. The $\beta \to 0$ annealing is what exposes the policy's own mistakes so the expert can correct exactly those states.
:::

### Exercise 4 — Derive the MaxEnt IRL gradient

Starting from $\mathcal{L}(\psi) = \sum_{\tau\in\mathcal{D}}[\sum_t r_\psi(s_t,a_t)] - M\log Z(r_\psi)$ with $M = |\mathcal{D}|$ and $Z = \sum_{\tau}\exp(\sum_t r_\psi)$, derive $\nabla_\psi\mathcal{L}$ and explain the two expectations in words.

::: details Solution
Differentiate term by term. The first term gives $\sum_{\tau\in\mathcal{D}}\sum_t \nabla_\psi r_\psi(s_t,a_t) = M\,\mathbb{E}_{\tau\sim\text{expert}}[\sum_t\nabla r_\psi]$.

For the second term, $\nabla_\psi \log Z = Z^{-1}\nabla_\psi Z$ and $\nabla_\psi Z = \sum_{\tau}\exp(\sum_t r_\psi)\sum_t\nabla_\psi r_\psi$. Dividing by $Z$ turns $\exp(\cdot)/Z$ into $p(\tau\mid r_\psi)$, so $\nabla_\psi\log Z = \mathbb{E}_{\tau\sim p(\cdot\mid r_\psi)}[\sum_t\nabla_\psi r_\psi]$. Multiplying by $M$ and subtracting gives the full-batch gradient $M(\mathbb{E}_{\text{expert}}[\cdot] - \mathbb{E}_{\text{model}}[\cdot])$; dividing through by the positive constant $M$ gives the per-trajectory form stated in §11.2 (same update direction, absorbed into the step size). In words: raise rewards where the expert goes, lower rewards where the current reward-induced policy goes, and stop when the two visit the same state-actions.
:::

### Exercise 5 — Optimal discriminator and implicit reward

Two state-action pairs have visit counts:

- Pair A: expert $600$, policy $400$.
- Pair B: expert $10$, policy $990$.

Compute $D^*$ and the implicit reward $-\log(1-D^*)$ for each. Which pair should the policy move toward?

::: details Solution
$D^* = N_{\text{expert}}/(N_{\text{expert}}+N_{\text{policy}})$.

Pair A: $D^* = 600/1000 = 0.6$. Reward $= -\log(1-0.6) = -\log 0.4 = 0.9163$.

Pair B: $D^* = 10/1000 = 0.01$. Reward $= -\log(1-0.01) = -\log 0.99 = 0.0101$.

Pair A gives roughly $91\times$ more reward, so policy-gradient steps toward higher implicit reward shift visits from B-like pairs to A-like pairs, which is exactly occupancy matching.
:::

### Exercise 6 — MAML initialization by hand

Two 1D tasks have losses $\mathcal{L}_1(\theta) = (\theta-2)^2$ and $\mathcal{L}_2(\theta) = (\theta+2)^2$. Inner learning rate $\alpha = 0.1$, one inner step. Compute the post-adaptation losses from meta-initialization $\theta = 0$ and from $\theta = 2$. Which initialization is better under the MAML outer objective (mean post-adaptation loss)?

::: details Solution
Gradient: $\nabla\mathcal{L}_i(\theta) = 2(\theta - c_i)$ with $c_1 = 2$, $c_2 = -2$.

From $\theta = 0$: $\theta_1' = 0 - 0.1\times2(0-2) = 0.4$; $\theta_2' = 0 - 0.1\times2(0+2) = -0.4$. Post losses: $(0.4-2)^2 = 2.56$ and $(-0.4+2)^2 = 2.56$. Mean $= 2.56$.

From $\theta = 2$: $\theta_1' = 2 - 0 = 2$; $\theta_2' = 2 - 0.1\times2(2+2) = 2 - 0.8 = 1.2$. Post losses: $0$ and $(1.2+2)^2 = 3.2^2 = 10.24$. Mean $= 5.12$.

The MAML objective picks $\theta = 0$: it is worse on task 1 before adaptation but far better on average after one gradient step, which is precisely "an initialization that is easy to fine-tune".
:::

### Exercise 7 — Coding: Gaussian BC and its failure mode (coding)

Fit a Gaussian BC head on 1D expert actions $\{1.0, -1.0, 1.0, -1.0\}$ in the same state (a left-or-right expert). Compute the MLE mean and the MSE-optimal prediction, then explain what the policy does at test time.

::: details Solution

```python
import numpy as np

actions = np.array([1.0, -1.0, 1.0, -1.0])
mu_mle = actions.mean()          # 0.0
mse = ((actions - mu_mle) ** 2).mean()  # 1.0
print(mu_mle, mse)
```

The MLE mean is $0.0$ with variance $1.0$. An MSE-trained deterministic policy outputs exactly $0.0$ — going straight — even though the expert never chose $0.0$. The average of two good modes is a bad action. This is the unimodal-Gaussian pitfall: multimodal demonstration data needs a mixture or diffusion head, and the symptom to watch for is high BC validation loss concentrated on states with divergent expert actions.
:::

### Exercise 8 — Coding: GAIL implicit reward from logits (coding)

A discriminator outputs logits $l$ with $D = \sigma(l)$. Implement the implicit reward $r = -\log(1-D) = -\mathrm{logsigmoid}(-l)$ in NumPy and evaluate it at $l \in \{-2, 0, 2\}$. Confirm that expert-like logits give high reward and policy-like logits give low reward.

::: details Solution

```python
import numpy as np

def gail_reward(logits):
    # r = -log(1 - sigmoid(l)) = softplus(l) = log(1 + exp(l))
    return np.logaddexp(0, logits)

for l in (-2.0, 0.0, 2.0):
    print(l, float(gail_reward(l)))
```

Output: $l=-2 \to 0.1269$, $l=0 \to 0.6931$, $l=2 \to 2.1269$. Check: $\sigma(2) = 0.8808$, $-\log(1-0.8808) = -\log(0.1192) = 2.1269$; $\sigma(0) = 0.5$, $-\log(0.5) = 0.6931$; $\sigma(-2) = 0.1192$, $-\log(0.8808) = 0.1269$. High logits (confident "expert") earn high reward, low logits earn near-zero reward, and $l = 0$ (indistinguishable) is the neutral midpoint, as required for the generator update.
:::
