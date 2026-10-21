---
title: 13.10 Summary and Exercises
---

# 13.10 Summary and Exercises

This page closes Chapter 13: the classic three-stage RLHF pipeline (SFT, reward model, PPO), its evaluation guardrails, and the controlled reward-hacking experiment. Chapter 14 replaces the explicit reward model with DPO; Chapters 15-17 replace more pieces with verifiable and process rewards.

## What you should now be able to do

1. Describe LLM generation as a sequential decision problem: name the state, action, policy, reward, and episode for a prompt-completion pair, and say when the contextual-bandit, token-MDP, and environment-MDP framings each apply.
2. Explain what each RLHF stage contributes: SFT as behavior cloning of demonstrations, the reward model as a learned preference ordering, and PPO as KL-constrained optimization of that ordering.
3. Write down the Bradley-Terry preference likelihood, compute its loss by hand for a given score pair, and explain what the margin measures beyond raw accuracy.
4. Read a PPO-RLHF training log (reward, KL, response length, entropy, value loss, judge win rate) and distinguish healthy training from reward hacking, mode collapse, and reference drift.
5. Diagnose the three canonical failure signatures: reward rising while human quality drops, length-reward correlation spiking, and diversity collapsing into templates.
6. Map a small TRL experiment (Actor, Reference, Reward Model, Critic) onto its large-scale counterparts (rollout workers, reward service, distributed trainer, evaluator) and name the four-model VRAM cost.

## Key equations

$$
\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}_{SFT}}[\log \pi_\theta(y \mid x)]
$$

SFT imitates high-quality demonstrations; it teaches format, not preference boundaries.

$$
\mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{pref}}[\log \sigma(r_\phi(x,y_w) - r_\phi(x,y_l))]
$$

The reward model turns "chosen beats rejected" into a trainable score difference; only differences are identified, so the reward offset is arbitrary (the scale is pinned by the sigmoid but must still be calibrated before PPO).

$$
\max_\theta\ \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[r_\phi(x,y) - \beta D_{KL}(\pi_\theta(\cdot\mid x)\,\|\,\pi_{ref}(\cdot\mid x))]
$$

PPO-RLHF improves the preference reward while the frozen reference anchors the policy; $\beta$ sets how far the actor may drift.

$$
r_t = -\beta(\log \pi_\theta(y_t\mid s_t) - \log \pi_{ref}(y_t\mid s_t)) + \mathbf{1}[t = T]\, r_{RM}(x, y)
$$

Each token pays a KL cost for drifting; only the final position collects the sequence-level judge score, which the critic then distributes backward as advantages.

$$
\mathcal{L}_{clip}(\theta) = -\mathbb{E}_t[\min(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)], \quad \rho_t = \frac{\pi_\theta(y_t\mid s_t)}{\pi_{\theta_{old}}(y_t\mid s_t)}
$$

Clipping stops any single token update from moving too far on one rollout batch.

$$
\text{win rate} = \frac{N_{win} + 0.5\, N_{tie}}{N_{win} + N_{lose} + N_{tie}}
$$

Preference win rate counts ties as half a win and is only comparable under identical prompts, judges, and decoding settings.

## Common pitfalls

1. **Reading reward curves alone.** A rising RM score with rising length and falling distinct-$n$ is the standard reward-hacking signature, not progress. Always read reward next to KL, length, diversity, and manual spot-checks.
2. **Training on the wrong tokens.** Forgetting the assistant-only loss mask teaches the model to parrot prompts, tool outputs, and chat templates. Padding, post-EOS tokens, and truncated fragments must be masked too.
3. **Trusting RM accuracy without margins and slices.** 80% pairwise accuracy with near-zero margins is a weak PPO signal, and accuracy averaged over all domains hides the weakest one. Check margins, reward-length correlation, and per-domain breakdowns.
4. **Letting evaluation prompts leak into training.** Reusing eval prompts (or their near-duplicates) in SFT, RM, or PPO splits inflates every metric downstream. Split preference data by prompt, not by pair, and freeze a held-out eval set.
5. **Confusing inference-time selection with learning.** Best-of-$N$, rejection sampling, and lower temperatures can raise win rates without changing the policy at all. Name the decoding settings on both sides of every comparison.
6. **Scaling up before the small loop is healthy.** If SFT outputs are weak, the RM margin is flat, or PPO KL oscillates at 0.5B, moving to 7B multiplies debugging cost without fixing any of it.

## Exercises

### 1. Name the MDP objects (easy, conceptual)

For the prompt "Explain the KL penalty in one sentence" and a 40-token response, state what $s_t$, $a_t$, $\pi_\theta$, the transition, the reward, and the episode are. Then say which framing (contextual bandit, token MDP, environment MDP) fits single-turn RLHF best, and why.

::: details Solution
$s_t = (x, y_{<t})$: the prompt plus tokens generated so far. $a_t = y_t$: the next vocabulary token. $\pi_\theta(a_t \mid s_t)$: the LM's next-token distribution. The transition is deterministic concatenation: $s_{t+1} = s_t + a_t$. The reward is the sequence-level RM score $r_\phi(x, y)$ delivered at EOS (with per-token KL costs in PPO-RLHF). The episode is one full response from prompt to EOS or the length limit. The contextual bandit fits best here because feedback arrives once per response and there are no intermediate observations; the token MDP is the implementation view PPO uses for credit assignment, and the environment MDP only becomes necessary once tool outputs or verifier verdicts arrive mid-episode.
:::

### 2. Why SFT is not enough (easy, conceptual)

Give two reasons SFT alone cannot align a model, one appealing to distribution shift and one to the expressiveness of demonstrations versus comparisons.

::: details Solution
Distribution shift: SFT trains only on states from human demonstrations. At deployment the model generates its own prefix, and the first small deviation lands it in states the demonstrations never covered, where errors compound (the BC failure mode from Section 13.2). RL closes this by scoring the model's own rollouts. Expressiveness: one prompt admits many acceptable answers differing in helpfulness, tone, or honesty. A demonstration says "this answer is worth imitating" but never "this answer beats that one," so SFT cannot represent graded quality boundaries; pairwise preferences can.
:::

### 3. Hand-compute a Bradley-Terry loss (medium, numeric)

A reward model scores the chosen response $2.0$ and the rejected response $0.5$. Compute the preference probability and the RM loss. Then compute both for the reversed (wrong) ordering.

::: details Solution
$\Delta r = 2.0 - 0.5 = 1.5$. Preference probability $\sigma(1.5) = 1/(1+e^{-1.5}) \approx 1/1.2231 \approx 0.818$. Loss $-\log(0.818) \approx 0.201$. Reversed: $\Delta r = -1.5$, $\sigma(-1.5) \approx 0.1824$, loss $-\log(0.1824) \approx 1.702$. The confident correct ranking costs little; the confident wrong ranking costs about $1.5$ nats more, and its gradient pushes the chosen score up and the rejected score down.
:::

### 4. Hand-compute a token KL penalty (medium, numeric)

At some position the actor log-probability of the generated token is $-2.0$ and the reference log-probability is $-2.4$. With $\beta = 0.1$, what is this token's KL penalty term in the total reward? Does the actor favor this token more or less than the reference?

::: details Solution
$\log \pi_\theta - \log \pi_{ref} = -2.0 - (-2.4) = 0.4$. The penalty term is $-\beta \times 0.4 = -0.04$. The actor favors this token more ($-2.0 > -2.4$ means higher probability), so the positive gap is taxed: drifting toward tokens the reference dislikes costs reward.
:::

### 5. Compute a win rate (medium, numeric)

Ten pairwise comparisons give 6 wins, 3 losses, and 1 tie. Compute the win rate with ties as half-wins. Then explain why reporting "60% excluding ties" would be misleading.

::: details Solution
$(6 + 0.5 \times 1)/10 = 6.5/10 = 65\%$. Dropping the tie reports $6/9 \approx 66.7\%$ on a different denominator and discards the information that one case was indistinguishable; worse, if ties are dropped selectively across experiments, denominators differ and the numbers are no longer comparable. Always report the sample size, list ties separately, and keep the denominator fixed.
:::

### 6. Derive the Bradley-Terry gradient (hard, derivation)

Starting from $\mathcal{L}_{RM} = -\log \sigma(z)$ with $z = r_\phi(x, y_w) - r_\phi(x, y_l)$, show that $\partial \mathcal{L}_{RM} / \partial z = -(1 - \sigma(z))$, and state the gradients with respect to the chosen and rejected scores. Evaluate them at $z = 1.5$ and interpret the signs.

::: details Solution
$\sigma'(z) = \sigma(z)(1-\sigma(z))$, so by the chain rule $\partial \mathcal{L}/\partial z = -\sigma'(z)/\sigma(z) = -(1-\sigma(z))$. Hence $\partial \mathcal{L}/\partial r_w = -(1-\sigma(z))$ and $\partial \mathcal{L}/\partial r_l = +(1-\sigma(z))$. At $z = 1.5$, $\sigma \approx 0.818$, so the gradients are $\mp 0.182$: gradient descent increases the chosen score and decreases the rejected score, each proportional to the residual error $1 - \sigma(z)$. Confident correct pairs ($z \gg 0$) contribute almost no gradient; confidently wrong pairs ($z \ll 0$) contribute gradients near $\pm 1$.
:::

### 7. Work the PPO clip in both directions (hard, numeric)

Let $\epsilon = 0.2$. (a) Old probability $0.10$, new probability $0.13$, advantage $A = 2$: compute $\rho$, the clipped and unclipped objectives, and the value PPO optimizes. (b) Same probabilities but $A = -3$: which term does $\min$ select, and what does the update do?

::: details Solution
(a) $\rho = 0.13/0.10 = 1.3$. Unclipped: $1.3 \times 2 = 2.6$. Clipped: $\mathrm{clip}(1.3, 0.8, 1.2) \times 2 = 2.4$. PPO optimizes $\min(2.6, 2.4) = 2.4$: the token is good but already moved enough this step. (b) Unclipped: $1.3 \times (-3) = -3.9$. Clipped: $1.2 \times (-3) = -3.6$. $\min(-3.9, -3.6) = -3.9$, the unclipped term: because the update already overshot upward on a bad token, the full negative gradient applies to push the probability back down.
:::

### 8. Mask the loss correctly (coding)

Implement `build_labels` from Section 13.2 for the toy sequence `input_ids = [101, 202, 303, 404, 505]` with `assistant_mask = [0, 0, 0, 1, 1]` and `ignore_index = -100`. State the output labels and how many tokens contribute to the loss. Then describe one concrete failure caused by setting the mask to all ones, and how you would catch it in the metrics from Section 13.2 (`train_loss`, `eval_loss`, format pass rate, human sampling).

::: details Solution
`labels = input_ids.clone(); labels[assistant_mask == 0] = -100`, giving `[-100, -100, -100, 404, 505]`; exactly 2 tokens contribute to the loss. With an all-ones mask the model is trained to predict the system and user tokens too, so it learns to parrot prompts and role markers. `train_loss` can still look excellent (prompts are easy to memorize), which is the trap: the failure shows up as a rising `eval_loss` gap, degraded format pass rate on held-out prompts, and human samples that echo the question instead of answering it. A minimal test: after a short SFT run in `code/chapter13_rlhf/sft_pipeline.py`, generate on unseen prompts and check that outputs contain no copied user text or role markers.
:::

### 9. Stress-test a flawed reward (coding)

Using the `flawed_reward` from Section 13.7 (length$/100$ + list/bold bonuses $0.5$ each + $0.3$ per politeness phrase in `["I'm happy to help", "Hope this helps", "Please note", "Here are some"]`), hand-compute and then programmatically verify the scores of: (a) `""`, (b) `"OK."`, (c) `"Please note: "` followed by 200 `x` characters, (d) `"The answer is 42."`. Which response wins, and what does that tell you?

::: details Solution
(a) Length $0 \rightarrow 0.0$; no format or politeness matches; total **0.0**. (b) Length $3 \rightarrow 0.03$; no other matches; total **0.03**. (c) Length $13 + 200 = 213 \rightarrow 2.13$; matches `"Please note"` ($+0.3$); no list/bold markers; total **2.43**. (d) Length $17 \rightarrow 0.17$; no other matches; total **0.17**. The vacuous 213-character filler (c) outscores the correct concise answer (d) by more than $14\times$. That is the lesson: the reward function encodes "longer plus polite phrasing is better," so a policy optimizing it will learn padding, not correctness. Gate reward functions with such unit tests before any PPO run (compare with the training loop in `code/chapter13_rlhf/rlhf_ppo_train.py`, which scores rollouts with its own rule-based scorer).
:::
