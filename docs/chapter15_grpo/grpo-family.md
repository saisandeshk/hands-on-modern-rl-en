---
title: 15.4 Improved Methods for GRPO
---

# 15.4 Improved Methods for GRPO

In the previous section, we examined DeepSeek-R1-Zero and DAPO — they respectively demonstrated that "pure RL can replace SFT cold start" and "engineering improvements can allow GRPO to reach the level of R1-Zero with half the steps." However, DAPO is just **one member** of the family of GRPO improvements in 2025. From the publication of the R1 paper (2025.01) to early 2026, the open-source community and industrial laboratories proposed at least five influential GRPO variants within less than a year. These variants are not mutually exclusive but rather address different defects of GRPO from different angles.

This section reorganizes the improvements by **improvement direction** — placing the five approaches of Dr.GRPO, GSPO, CISPO, VAPO, and RPT side by side for comparison, so you can clearly understand "when to use which variant."

## Improvement Route One: Removing Normalization Bias

### Discovery of Dr.GRPO

The original form of GRPO in the R1 paper applied two steps of normalization to the group's reward (the within-group mean and spread, written $\bar r_j$ and $s_j$ in Section 15.1):

$$\tilde{r}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

Dividing by the standard deviation seems natural — it standardizes the advantage values. However, Liu et al. discovered in their 2025 study ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)) that this seemingly harmless normalization introduced two types of bias:

- **Length Bias**: When a prompt's reward variance is large (some answers are correct, some are not), dividing by the standard deviation compresses the advantage. When the variance is small (all answers are correct or all are incorrect), the standard deviation approaches zero, and the advantage is amplified to unreasonable magnitudes. The model thus learns that "producing diverse outputs is more important than being correct."
- **A breeding ground for reward hacking**: Dividing by the standard deviation is equivalent to encouraging the model to increase the variance of the group's reward. The simplest way to increase variance is to make some answers **longer** (more tokens, more chances to be correct). This is one of the direct causes of the explosive answer length in the later stages of R1-Zero training.

Dr.GRPO's correction is extremely simple — **only subtract the mean, not divide by the standard deviation**:

$$\tilde{r}_i^{\text{Dr.GRPO}} = r_i - \text{mean}(r_1, \ldots, r_G)$$

Experiments show that this change significantly alleviates the problem of answer length inflation in the later stages of training and reduces reward hacking behavior. The Qwen series adopted a similar correction in their internal training.

### Further engineering in the DeepSeek-V3.2 line

DeepSeek's late-2025 V3.2 technical report describes pushing the Dr.GRPO idea further for mathematical reasoning tasks. Two directions are reported: using mean-only advantages without standard-deviation normalization, and weakening or disabling the KL penalty during the pure mathematical RL phase, on the argument that a strong verifiable reward already constrains the policy and the KL term can suppress exploration of new solution paths. Specific benchmark numbers attached to these reports (including AIME scores and comparisons with other labs' models) come from rapidly changing internal evaluations and should not be quoted as settled results.

## Improvement Route Two and Sequence-Level Importance Sampling

### GSPO (Qwen3's Choice)

GRPO and PPO both use **token-level importance sampling ratios**:

$$\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

Each token has its own ratio. GRPO clips those ratios independently and then averages the token objectives; it does not multiply all token ratios together. This is the original definition in [DeepSeekMath Equation (3)](https://arxiv.org/html/2402.03300#S3.SS1). The GSPO paper argues that these token-wise importance weights introduce high-variance noise when the reward belongs to the whole response, and that routing changes in MoE models amplify the fluctuations.

GSPO (Group Sequence Policy Optimization) elevates the ratio from the token level to the **sequence level**:

$$
s_i(\theta)
=
\left(
\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\text{old}}}(y_i\mid x)}
\right)^{1/|y_i|}
=
\exp\!\left(
\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}
\right).
$$

Equivalently, this is the geometric mean of the token ratios:

$$
s_i(\theta)=\left(\prod_{t=1}^{|y_i|}\rho_{i,t}\right)^{1/|y_i|}.
$$

The entire response uses a single ratio for clipping. The clipping target is also adjusted to the sequence level:

$$\mathcal{J}^{\text{GSPO}} = \mathbb{E}\left[\min\left(s_i(\theta)\hat A_i, \; \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon)\hat A_i\right)\right].$$

The $1/|y_i|$ length normalization is essential. Using the raw product $\prod_t\rho_{i,t}$ makes the ratio fluctuate with response length and would require different clipping ranges for responses of different lengths. For this reason, the former GRPO example—which summed all token log-probability differences and exponentiated without length normalization—was not a correct GSPO implementation either. [Equations (5) and (7) in the original GSPO paper](https://arxiv.org/html/2507.18071#S4.SS1)

This change may seem simple, but it has a significant impact on the training stability of MoE models. The variance of the sequence-level ratio is much smaller than that of the token-level product, which is reported to make large-scale MoE training more feasible (specific cluster sizes and model-line attributions are not independently verified here).

The cost of GSPO: the sequence-level ratio **couple all token updates**, and its ability to perform fine-grained credit assignment at the token level is weaker than token-level methods. Therefore, GSPO performs significantly well on long CoT tasks (reasoning, mathematics), but is less effective than DAPO on code generation tasks that require token-level rewards.

## Improvement Route Three and Rewriting of the Pruning Object

### CISPO (Innovation from MiniMax)

GRPO and DAPO both prune the "product of the policy ratio and advantage" — the pruning occurs at the gradient update level. In the M1 model ([arXiv:2506.13585](https://arxiv.org/abs/2506.13585)), MiniMax proposes CISPO (Clipped IS-weight Policy Optimization), which changes the pruning object from "token update" to "**importance sampling weight**":

$$\hat{r}_{t}(\theta) = \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, 1-\epsilon^{IS}_{low}, 1+\epsilon^{IS}_{high}\right)$$

Note the ratio keeps the standard direction — $\pi_\theta / \pi_{\theta_{\text{old}}}$ — but the clipped weight is used differently: it enters the loss as a **detached (stop-gradient) coefficient** multiplying the log-probability gradient, $\hat{r}_{t}(\theta)\,\hat{A}_{t}\,\log \pi_\theta(a_t|s_t)$, rather than as a PPO-style min-clip on the update itself. Because the gradient flows through $\log \pi_\theta$ for every token, **all tokens keep contributing gradients** even when their weight is clipped.

Intuitively: traditional pruning is "if a token's policy deviates too much, directly remove it from the gradient." CISPO is "if the deviation is too large, reduce its contribution weight, but keep the gradient direction unchanged." The latter avoids the problem of "some tokens completely stop updating in the later stage of training," which can cause the policy to get stuck.

CISPO also has an engineering motivation — it is designed to work with MiniMax's own (lightning) attention implementation, where per-token ratio computation in low precision can accumulate floating-point error. Scaling the importance weight instead of dropping tokens is reported to be more robust in that setting. The paper reports that combining hybrid attention with CISPO let the full M1 RL run complete on 512 H800 GPUs in three weeks (at a stated rental cost of about \$534,700); treat exact cost figures as setup-specific rather than transferable.

## Improvement Route Four: Against the Value-based Counter-trend

### VAPO (Byte Seed's Counter-trend)

At this point, you might have the impression that **the Critic network has been eliminated by GRPO**. However, the paper VAPO ([arXiv:2504.05118](https://arxiv.org/abs/2504.05118)) published by ByteDance Seed in April 2025 demonstrates — at least in long CoT reasoning tasks — that **the value model has once again defeated GRPO**.

The core argument of VAPO is that GRPO replaces the Critic network with the group mean, essentially using the "relative ranking among multiple rollouts of the same prompt" to estimate advantage. This is sufficient for short answer tasks (e.g., function calling, simple math problems). However, in long CoT tasks:

- Within a rollout, there are hundreds of tokens, and the **true advantage signal is at the token level** — one step of reasoning is good, the next is bad.
- The group mean treats the entire rollout as a single unit, losing the token-level signal.
- As training proceeds, the model tends to learn to "answer some rollouts by chance" rather than "being correct at every step."

VAPO reintroduces the value model $V_\phi(s)$, using GAE to estimate token-level advantage:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

where $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ is the TD error. Then, it applies PPO-style clipping on this token-level advantage.

VAPO reports a score of 60.4 on AIME 2024 with a Qwen-32B base model, outperforming DeepSeek-R1-Zero-Qwen-32B and DAPO by more than 10 points under identical experimental settings. (The much higher ~71 figure sometimes quoted for R1-Zero belongs to the full-scale DeepSeek-R1-Zero, a different model scale, not the Qwen-32B comparison.) So the honest reading is about efficiency and setup rather than outright supremacy: reintroducing a value model can pay off on long chain-of-thought tasks where token-level credit matters, at the cost of the extra model.

The cost of VAPO: it requires training an independent value model, doubling the memory usage, and increasing engineering complexity. This is why GRPO became the mainstream approach around 2025 — **critic-free is an engineering compromise, not an algorithmic necessity**.

## Improvement Route Five: Integrating RL into Pre-training

### RPT (Reinforcement Pre-training)

The first four improvement routes assumed that RL occurred in the **post-training phase** — the model had already been pre-trained, and RL was merely fine-tuning. However, Dong et al.'s June-2025 paper on Reinforcement Pre-training ([arXiv:2506.08007](https://arxiv.org/abs/2506.08007)) challenges this binary distinction.

The core idea of RPT is to **reframe the next-token prediction task as a reasoning task**. The traditional pre-training loss is:

$$\mathcal{L}_{\text{LM}} = -\mathbb{E}\left[\log \pi_\theta(a_t | s_{<t})\right]$$

In this setup, each token is treated as an equal teacher-forcing target. RPT transforms this by having the model first generate a reasoning about the next token ("Given the context, the next word might be X because..."), and then use the reasoning result to predict the next token, rewarding the model when it is correct:

$$\mathcal{L}_{\text{RPT}} \approx -\mathbb{E}\left[\log \pi_\theta(a_t \mid s_{<t}, \text{reasoning}_t)\right] + \text{RL term rewarding correct prediction}$$

Schematically: the model first generates a short reasoning trace about the next token and is rewarded when the resulting prediction is correct. This change is conceptual — **RL can be performed during the pre-training phase** — and RPT's scaling properties are reported to be comparable to traditional pre-training. This suggests that the future may no longer have a clear distinction between "pre-training" and "post-training," with RL being integrated throughout the entire training process.

RPT is still in its early stages and has not yet been widely adopted in industry practice. However, its conceptual impact is significant enough for this book to dedicate a separate improvement route to it.

## Operating Envelopes: Zero-Signal Groups, Stale Rollouts, and Collapse Signals

The variants above change the update rule. A separate set of problems comes from how rollouts are reused and what the training traces look like while they are reused.

**Zero-signal groups.** When every answer in a group is correct or every answer is wrong, the within-group advantage is exactly zero and the group contributes no gradient. DAPO's dynamic sampling handles this by filtering out solved prompts and oversampling the rest, so each batch keeps a usable fraction of groups with nonzero variance. The same logic applies to invalid outputs: a response that cannot be parsed or executed should receive the group's minimum reward (typically 0) rather than being dropped silently, because dropping it biases the group mean upward. Verifier failures need the same treatment — if the verifier itself errors, mark the rollout as failed and exclude the whole prompt from the update instead of recording a 0, or the policy learns to avoid whatever triggered the verifier bug.

**One update per rollout versus several.** The original GRPO analysis uses each rollout batch once (`num_iterations=1`): the sampling policy and the policy being updated coincide, so the importance ratio starts at 1 and clipping rarely binds. Taking several gradient steps on the same batch (as PPO commonly does) improves sample efficiency but makes the data off-policy: after the first step, $\pi_\theta$ has moved away from $\pi_{\theta_{\text{old}}}$, the ratios drift from 1, and clipping starts discarding token updates. A practical rule is to watch the clipped-token fraction and the mean absolute ratio error $|\rho - 1|$: a rising clipped fraction is the signal that the rollout batch is going stale and should be replaced rather than re-used. Asynchronous systems (see Section 15.5) face the same trade-off continuously and typically cap policy lag at a few versions, downweighting or discarding older rollouts.

**Collapse signals.** Three traces warn that the policy is contracting instead of improving: entropy of the next-token distribution falls toward zero while reward plateaus (entropy collapse — the model repeats one template), KL against the reference model spikes suddenly (one update moved too far), and mean response length drifts up while external quality is flat (verbosity incentive — longer answers harvest format or partial-credit rewards without solving more problems). Length normalization in the loss (averaging inside each response, as `loss_type="grpo"` does) removes the direct reward for verbosity, but the indirect incentive remains whenever longer answers have more chances to hit a sparse reward. The fix is diagnostic rather than algorithmic: track reward together with KL, entropy, mean length, and the fraction of valid outputs, and stop or regularize when reward rises alone.

## Decision Tree for Selection

The following table summarizes the core differences, applicable scenarios, and typical users of the five variants:

| Algorithm   | Core Innovation                        | Pain Points Solved                   | Typical Use Cases                  | Representative Users |
| ----------- | -------------------------------------- | ------------------------------------ | ---------------------------------- | -------------------- |
| **GRPO**    | Replacing Critic with Intra-group Mean | Critic memory overhead               | General RLHF / RLVR                | DeepSeek-R1          |
| **Dr.GRPO** | Removing std normalization             | Length explosion, reward hacking     | Mathematical reasoning             | Qwen Internal        |
| **GSPO**    | Sequence-level IS                      | MoE training instability             | RL for MoE models                  | Qwen3 Full Line      |
| **CISPO**   | Pruned IS weights                      | Token loss, precision alignment      | Lightning attention, low precision | MiniMax M1           |
| **VAPO**    | Reintroducing value model              | Long CoT credit assignment           | Training of reasoning models       | Byte Seed            |
| **DAPO**    | Four engineering improvements          | Training efficiency, length control  | Mathematical / code RL             | Byte + Tsinghua      |
| **RPT**     | Pre-training RL integration            | Pre-training to fine-tuning boundary | Next-generation base models        | Microsoft Research   |

The selection logic in practical industrial applications is roughly as follows:

```text
Task Type?
├── Mathematical/Code Reasoning (Long CoT)
│   ├── MoE Architecture → GSPO + Dr.GRPO Ideas
│   ├── Dense Architecture → VAPO or DAPO
│   └── Extreme Stability Requirements → CISPO
├── General Dialogue Alignment
│   └── GRPO / PPO (Basic Enough)
├── Multi-turn Tool Calls
│   └── DAPO + Token-level Loss
└── Next-generation Base Models
    └── RPT (Experimental)
```

This decision tree is not absolute — within ByteDance Seed, there is frequent mixing (for example, DAPO's engineering techniques + VAPO's value model). But it provides a checklist of what to consider first after seeing a task.

## Summary

The rapid evolution of the GRPO improvement family reflects a fact: **RL in the era of large models is no longer a single choice of "just using PPO is enough."** Each lab has chosen different improvement directions based on their training infrastructure (MoE vs Dense, lightning attention vs standard attention, memory budget) and task characteristics (inference vs dialogue, long CoT vs short answer).

The true value of this section is not in remembering the specific formulas of each algorithm — it is in building a sense of judgment: **when seeing a new GRPO variant, being able to immediately ask, "Which of the five directions — normalization, sequence-level, clipping, value model, or pre-training integration — is this variant addressing?"** This sense of judgment is a crucial step from reading papers to being able to improve algorithms hands-on.
