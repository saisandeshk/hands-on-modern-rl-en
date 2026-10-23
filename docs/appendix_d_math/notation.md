---
title: D.5 Notation
---

# D.5 Notation

This page collects the standard symbols used throughout the book. Each entry gives the meaning in one line. Where the book uses more than one form for the same object, the preferred form comes first and variants follow in parentheses.

## Environment and trajectories

| Symbol                                | Meaning                                                                            |
| ------------------------------------- | ---------------------------------------------------------------------------------- |
| $s$                                   | Current state                                                                      |
| $a$                                   | Action taken in state $s$                                                          |
| $s'$                                  | Next state after taking action $a$                                                 |
| $r$                                   | Immediate scalar reward for one step                                               |
| $R_{t+1}$ ($r_t$, $r_{t+1}$)          | Reward received after the action at step $t$ (book's dominant convention is $r_t$) |
| $\gamma$                              | Discount factor in $[0, 1)$; weights near-term reward more heavily                 |
| $G_t$                                 | Discounted return from step $t$, $G_t = R_{t+1} + \gamma R_{t+2} + \cdots$         |
| $\tau$                                | Trajectory $(s_0, a_0, s_1, a_1, \dots)$                                           |
| $T$                                   | Horizon (length) of an episode or trajectory window                                |
| $\mathcal{S}$                         | Set of all states                                                                  |
| $\mathcal{A}$                         | Set of all actions                                                                 |
| $P(s' \mid s, a)$ ($p(s' \mid s, a)$) | Transition probability of reaching $s'$ from $(s, a)$                              |
| $R(s, a)$                             | Expected immediate reward for $(s, a)$                                             |
| $d^\pi(s)$                            | State visitation frequency under policy $\pi$                                      |
| $p_0(s)$ ($p(s_0)$)                   | Distribution of the initial state                                                  |

## Values, advantages, and TD errors

| Symbol                             | Meaning                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| $V^\pi(s)$ ($v_\pi(s)$)            | State value: expected return from $s$ under $\pi$                               |
| $Q^\pi(s, a)$ ($q_\pi(s, a)$)      | Action value: expected return after taking $a$ in $s$ under $\pi$               |
| $V^*$ ($v_*$)                      | Optimal state value, $\max_\pi V^\pi(s)$                                        |
| $Q^*(s, a)$ ($q_*$)                | Optimal action value                                                            |
| $A^\pi(s, a)$ ($A_\pi$, $A(s, a)$) | Advantage, $Q^\pi(s, a) - V^\pi(s)$; positive means better than average         |
| $\hat{A}_t$                        | Estimated advantage at step $t$ used in a gradient update                       |
| $\delta_t$ ($\delta$)              | One-step TD error, $r_t + \gamma V(s_{t+1}) - V(s_t)$ ($R_{t+1}$ in some pages) |
| $\lambda$                          | GAE decay in $[0, 1]$; trades TD bias against Monte Carlo variance              |
| $\hat{v}(s)$, $\hat{V}(s)$         | Estimated (approximate) state value                                             |
| $y_t$                              | TD or DQN target that the predicted value is regressed toward                   |

## Policies, ratios, and data sources

| Symbol                                                                 | Meaning                                                                                                                 |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| $\pi(a \mid s)$                                                        | Policy: probability of action $a$ in state $s$                                                                          |
| $\pi_\theta$ ($\pi_\theta(a \mid s)$)                                  | Parametric policy with parameters $\theta$                                                                              |
| $\pi_{\text{old}}$ ($\pi_{\theta_{\text{old}}}$, $\pi_{\theta_{old}}$) | Policy that collected the current batch (PPO denominator)                                                               |
| $\pi_{\text{ref}}$                                                     | Reference model, usually the SFT checkpoint, that anchors RLHF/DPO                                                      |
| $b$ ($\mu$, $\pi_\beta$, $\pi_b$)                                      | Behavior (logging) policy that generated off-policy data                                                                |
| $\rho_t$ ($r_t(\theta)$)                                               | Importance ratio $\pi(a_t \mid s_t) / b(a_t \mid s_t)$; PPO chapters write the same new-over-old ratio as $r_t(\theta)$ |
| $r_t(\theta)$                                                          | PPO probability ratio $\pi_\theta(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$                                       |
| $\rho_{0:T}$ ($\rho_{1:t}$)                                            | Product of per-step ratios ($\rho_{0:T}$ in D.2.5; $\rho_{1:t}$ in Chapter 10)                                          |
| $\mathcal{D}$                                                          | Dataset of prompts, trajectories, or preference pairs                                                                   |
| $\epsilon$ ($\varepsilon$)                                             | PPO clipping half-width (commonly $0.1$--$0.2$)                                                                         |

## Optimization and function approximation

| Symbol                          | Meaning                                                                                            |
| ------------------------------- | -------------------------------------------------------------------------------------------------- |
| $\theta$                        | Vector of policy (or model) parameters being optimized                                             |
| $J(\theta)$                     | Policy objective (expected return, possibly with bonuses/penalties)                                |
| $L(\theta)$ ($\mathcal{L}$)     | Loss minimized during training                                                                     |
| $\alpha$                        | Learning rate (step size)                                                                          |
| $\nabla$                        | Gradient operator collecting all partial derivatives                                               |
| $\mathbf{v}$, $\mathbf{r}$, $P$ | Value vector, reward vector, and transition matrix for all states at once                          |
| $I$                             | Identity matrix in $(I - \gamma P)^{-1}$                                                           |
| $\mathbf{x}(s)$, $\phi(s, a)$   | Feature vector of a state or state-action pair                                                     |
| $\mathbf{w}$                    | Linear weight vector, $\hat{v}(s) = \mathbf{w}^\top \mathbf{x}(s)$                                 |
| $F$                             | Fisher information matrix defining the trust-region metric                                         |
| $\delta$ (trust region)         | Allowed size $\Delta\theta^\top F\, \Delta\theta \le \delta$ (context disambiguates from TD error) |
| $c$ ($\text{max\_grad\_norm}$)  | Gradient-clipping threshold on the L2 norm                                                         |

## LLM alignment (RLHF, DPO, GRPO)

| Symbol                     | Meaning                                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| $x$                        | Prompt (input context)                                                                                                          |
| $y$                        | Response (generated completion)                                                                                                 |
| $y_w$, $y_l$               | Winner and loser responses in a preference pair                                                                                 |
| $r_\phi(x, y)$ ($r(x, y)$) | Reward model score for response $y$ to prompt $x$                                                                               |
| $\beta$                    | KL penalty coefficient (also entropy-bonus coefficient in exploration)                                                          |
| $\sigma$                   | Sigmoid function in the DPO loss (context disambiguates from standard deviation)                                                |
| $\mu$, $\sigma$ (GRPO)     | Group mean and standard deviation used to normalize rewards (implementations use $\sigma + \epsilon$ to avoid division by zero) |
| $n$ ($G$)                  | Number of sampled responses per prompt in a GRPO group                                                                          |

## Information theory

| Symbol                                  | Meaning                                                             |
| --------------------------------------- | ------------------------------------------------------------------- |
| $H(P)$ ($H(\pi)$)                       | Entropy: average surprise (randomness) of a distribution or policy  |
| $H(P, Q)$                               | Cross-entropy: cost of encoding $P$ with the wrong distribution $Q$ |
| $D_{\mathrm{KL}}(P \Vert Q)$ ($D_{KL}$) | KL divergence: directed gap from $P$ to $Q$                         |
| $I(X; Y)$                               | Mutual information: uncertainty in $X$ removed by knowing $Y$       |
| $I(x)$                                  | Self-information of a single event, $-\log p(x)$                    |
| $\mathbb{E}[\cdot]$                     | Expectation (probability-weighted average)                          |
| $\mathrm{Var}$ ($\mathbb{V}$)           | Variance of a random variable                                       |

## Known conflicting usages in this book

- **Value case.** Main chapters write $V^\pi$, $Q^\pi$, $A^\pi$ (uppercase); several Appendix D pages write $v_\pi$, $q_\pi$, $A_\pi$ (lowercase). Both mean the same objects.
- **Old policy.** PPO chapters use $\pi_{\text{old}}$ (sometimes $\pi_{old}$); Appendix D sometimes writes $\pi_{\theta_{\text{old}}}$ or $\pi_{\theta_{old}}$. All denote the sampling policy in the ratio denominator.
- **Ratio.** Both $\rho_t$ and $r_t(\theta)$ denote a new-over-old probability ratio. $\rho_t$ is used in D.2.5 and in the Chapter 10 review/experiments pages (product forms $\rho_{0:T}$ in D.2.5 and $\rho_{1:t}$ in Chapter 10); PPO chapters and Appendix D.3 use $r_t(\theta)$.
- **Reward index.** TD errors appear as both $r_t + \gamma V(s_{t+1}) - V(s_t)$ and $R_{t+1} + \gamma V(s_{t+1}) - V(s_t)$. The book's dominant convention is $r_t$ (D.2.4 uses $r_t$; D.3.5 uses $R_{t+1}$).
- **Beta.** $\beta$ is the KL penalty in RLHF/DPO, the entropy-bonus coefficient in exploration, and (as a subscript, $\pi_\beta$) the behavior policy in offline RL. Read it from context.
- **Delta.** $\delta_t$ is the TD error; bare $\delta$ is the trust-region radius in TRPO. Read it from context.
- **Sigma.** $\sigma$ is the sigmoid in DPO and the standard deviation in GRPO statistics (implementations write the GRPO advantage with $\sigma + \epsilon$ to avoid division by zero). Read it from context.
- **Calligraphic D.** $\mathcal{D}$ is a dataset; $D_{\mathrm{KL}}$ is KL divergence. They are unrelated despite the shared letter.
