---
title: 15.R Summary and Exercises
---

# 15.R Summary and Exercises

This chapter replaced the two learned models of RLHF — the Critic and the reward model — with two cheaper substitutes: within-group comparison and verifiable rewards. GRPO (Sections 15.1–15.2, 15.4) samples several answers per prompt and turns "better than the group average" into an advantage; RLVR (Sections 15.3, 15.5–15.6, 15.8) scores those answers with rules, tests, and sandboxes instead of a reward model; on-policy distillation (Section 15.7) fills the remaining gap with dense token-level teacher feedback on the student's own trajectories.

## What You Should Now Be Able to Do

1. Explain why the group mean can replace a Critic, and state exactly what information is lost when it does (cross-prompt difficulty signal).
2. Compute group-relative advantages by hand, including the all-equal edge case where the gradient vanishes.
3. Write down the token-wise GRPO objective — ratio, clipping, per-response averaging, k3 KL penalty — and evaluate it on small numbers.
4. Distinguish `loss_type="grpo"` (per-response averaging) from batch-global averaging, and predict how each one biases the policy toward short or long responses.
5. Choose between GRPO variants (Dr.GRPO, GSPO, DAPO, VAPO) from a task description, and justify the choice from the failure mode each variant addresses.
6. Design a minimal RLVR verifier (math, tool call, or code pass rate), name its gaming surface, and list the training traces — KL, entropy, length, validity — that catch reward hacking.

## Key Equations

Group-relative advantage (Section 15.1) — how much better this answer is than its same-prompt siblings:

$$\hat A_{j,i} = \frac{r_{j,i} - \bar r_j}{s_j + \epsilon}$$

Leave-one-out baseline (the RLOO variant) — the same idea without letting a sample grade itself:

$$\hat A^{\text{RLOO}}_{j,i} = r_{j,i} - \frac{1}{G-1}\sum_{k \ne i} r_{j,k}$$

Token-wise clipped objective (DeepSeekMath Eq. 3) — PPO-style trust region applied per token, then averaged per response:

$$\mathcal{J}_{\text{clip}} = \mathbb{E}\left[\frac{1}{G}\sum_i \frac{1}{T_{j,i}}\sum_t \min\left(\rho_{j,i,t}\hat A_{j,i},\, \text{clip}(\rho_{j,i,t}, 1-\epsilon, 1+\epsilon)\hat A_{j,i}\right)\right]$$

k3 KL estimator (DeepSeekMath Eq. 4) — the non-negative per-token penalty toward the reference policy:

$$\widehat D_{j,i,t} = \exp(\Delta_{j,i,t}) - \Delta_{j,i,t} - 1, \qquad \Delta_{j,i,t} = \ell^{\text{ref}}_{j,i,t} - \ell^{\theta}_{j,i,t}$$

GSPO sequence ratio — one geometric-mean ratio per response instead of one ratio per token:

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}\right)^{1/|y_i|} = \left(\prod_{t=1}^{|y_i|} \rho_{i,t}\right)^{1/|y_i|}$$

On-policy distillation signal (Section 15.7) — the teacher's approval of each token the student actually sampled:

$$r_t = \log q(y_t \mid c_t) - \log \pi_\theta(y_t \mid c_t)$$

## Common Pitfalls

1. **Treating zero advantage as a bug.** When all answers in a group score the same, advantages are exactly zero and the group contributes no gradient. That is the design working as intended — filter such prompts (dynamic sampling) rather than "fixing" the normalization.
2. **Switching KL estimators without rescaling $\beta$.** The k1 log-ratio estimator can be negative per token; the k3 estimator never is. They imply different effective penalty scales, so a $\beta$ tuned for one is wrong for the other.
3. **Averaging over the wrong denominator.** Per-response averaging (`loss_type="grpo"`) weights every answer equally; batch-global averaging weights every token equally and silently upweights long responses. Picking the wrong one reintroduces the verbosity incentive the loss was supposed to remove.
4. **Watching reward alone.** Reward can rise while KL spikes, entropy collapses, length explodes, or validity falls. Log all five traces; a reward gain unaccompanied by the others holding steady is a hacking suspect, not a capability gain.
5. **Scoring infrastructure failures as zeros.** Sandbox timeouts, tool 500s, and verifier crashes are `infra_error`, not reward 0. Mixing them into group statistics teaches the policy to avoid whatever preceded the outage.
6. **Reusing a rollout batch until it goes stale.** After the first gradient step the data is off-policy. If the clipped-token fraction climbs past roughly a third, generate fresh rollouts instead of squeezing more steps out of old ones.

## Exercises

### Exercise 1 — Why is the group mean a legitimate baseline?

State in one paragraph what job the Critic's baseline does in policy gradients, and why replacing $V(s)$ with the group mean $\bar r_j$ preserves that job. Then name one thing the group mean cannot do that a trained Critic can.

::: details Solution
The baseline's job is variance reduction: subtracting a reward predictor that does not depend on the sampled action leaves the expected gradient unchanged (in the exact-score limit) while shrinking its sampling noise, because constant shifts across actions cancel in expectation. The group mean $\bar r_j$ plays exactly this role — it is computed from the same prompt's other samples, so it tracks prompt difficulty without needing a second network, and advantages $r - \bar r$ keep only the within-prompt comparison the update needs. What it cannot do: generalize across prompts. A trained Critic predicts values for unseen states and carries cross-prompt difficulty information (this prompt is hard, that one is easy); the group mean discards it, so two prompts with uniformly high and uniformly low rewards both yield zero gradient.
:::

### Exercise 2 — Compute group advantages by hand

Four answers to one prompt receive rewards $[2.0, 1.0, 1.0, 0.0]$. Using population standard deviation (divide by $G$), compute the mean, the standard deviation, and the four advantages. Verify that the advantages sum to zero.

::: details Solution
Mean: $\bar r = (2 + 1 + 1 + 0)/4 = 1.0$. Deviations: $[1, 0, 0, -1]$. Population variance: $(1 + 0 + 0 + 1)/4 = 0.5$, so $s = \sqrt{0.5} \approx 0.7071$. Advantages: $[1/0.7071, 0, 0, -1/0.7071] \approx [+1.4142, 0, 0, -1.4142]$. Sum: $1.4142 + 0 + 0 - 1.4142 = 0$, as required — the positive and negative pressures inside one group always balance.
:::

### Exercise 3 — Clipped token objective on paper

Token ratios are $[1.1, 0.9, 1.5]$, the shared response advantage is $\hat A = +0.5$, and $\epsilon = 0.2$. Compute the clipped ratios, the per-token values $\min(\rho \hat A, \text{clip}(\rho)\hat A)$, and their mean. Which token's update is constrained by clipping?

::: details Solution
Clipped ratios with $[0.8, 1.2]$ bounds: $[1.1, 0.9, 1.2]$ (only the third binds). Unclipped values: $[0.55, 0.45, 0.75]$. Clipped values: $[0.55, 0.45, 0.60]$. Per-token minima: $[0.55, 0.45, 0.60]$. Mean: $(0.55 + 0.45 + 0.60)/3 = 1.60/3 \approx 0.5333$. The third token is constrained: its ratio $1.5$ is cut to $1.2$, capping its contribution at $0.60$ instead of $0.75$. Note the contrast with a sequence-level ratio: the raw product $1.485$ would clip the whole response at once, punishing all three tokens for one large move.
:::

### Exercise 4 — k1 versus k3 on one token

At some token, $\ell^{\text{ref}} = -2.0$ and $\ell^{\theta} = -2.1$. Compute $\Delta$, the k1 estimate $-\Delta$, and the k3 estimate $\exp(\Delta) - \Delta - 1$. Explain in one sentence why the k3 value is better behaved as a per-token penalty.

::: details Solution
$\Delta = -2.0 - (-2.1) = 0.1$. k1: $-0.1$. k3: $e^{0.1} - 0.1 - 1 \approx 1.10517 - 1.1 \approx 0.0052$. The k1 estimate is negative even though the policies genuinely differ at this token, so per-token k1 penalties and logs are hard to interpret; k3 is always non-negative (since $e^x \ge 1 + x$) and zero only when the policies agree, making it a well-behaved per-token regularizer.
:::

### Exercise 5 — GSPO sequence ratio on paper

Using the same token ratios $[1.1, 0.9, 1.5]$ with response length $|y| = 3$, compute the GSPO sequence ratio $s = (\prod_t \rho_t)^{1/3}$. With $\epsilon = 0.2$, does clipping bind for the sequence ratio? Contrast with the raw product.

::: details Solution
Product: $1.1 \times 0.9 \times 1.5 = 1.485$. Sequence ratio: $1.485^{1/3} \approx 1.141$ (check: $\ln 1.485 \approx 0.3958$, divided by $3$ is $0.1319$, $\exp(0.1319) \approx 1.141$). With $\epsilon = 0.2$ the allowed band is $[0.8, 1.2]$: $1.141$ lies inside, so clipping does not bind. The raw product $1.485$ would bind at $1.2$ — without the $1/|y|$ normalization, longer responses produce more extreme ratios and need different clipping ranges per length, which is exactly the length dependence GSPO removes.
:::

### Exercise 6 — Code the advantage function with a zero-std guard

Write `grpo_group_normalize(rewards)` following `code/chapter15_grpo/grpo_mechanism.py`: return zeros when the standard deviation is below a small threshold, else the standardized rewards. Test it on `[1.5, 1.5, 1.5, 1.5]` and on `[1.5, 0.0, 1.5, 0.0, 1.0, 1.5, 0.5, 1.5]`, and confirm the second case reproduces mean $0.9375$ and advantages starting $[0.887, -1.478, \dots]$.

::: details Solution

```python
import numpy as np

def grpo_group_normalize(rewards, eps=1e-8):
    rewards = np.asarray(rewards, dtype=float)
    std = rewards.std()  # population std, matches the chapter's convention
    if std < eps:
        return np.zeros_like(rewards)
    return (rewards - rewards.mean()) / std

print(grpo_group_normalize([1.5, 1.5, 1.5, 1.5]))
# [0. 0. 0. 0.]
adv = grpo_group_normalize([1.5, 0.0, 1.5, 0.0, 1.0, 1.5, 0.5, 1.5])
print(adv)  # [0.8868 -1.4780 0.8868 -1.4780 0.0985 0.8868 -0.6897 0.8868]
```

The uniform group has std $0 < \epsilon$, so it returns zeros (no gradient signal — the "graduated" case). The mixed group has mean $7.5/8 = 0.9375$ and std $\approx 0.6343$; $(1.5 - 0.9375)/0.6343 \approx 0.8868$ and $(0 - 0.9375)/0.6343 \approx -1.4780$, matching the chapter's worked example.
:::

### Exercise 7 — Harden a math verifier

Extend the `grade_answer` sketch from Section 15.3 so that (a) `"63"`, `"63.0"`, and `" 63 "` all match ground truth `"63"`, (b) `"22/7"` matches `"3.142857"` within tolerance, and (c) an empty prediction never matches. Write three assertions per case and run them.

::: details Solution

```python
def grade_answer(predicted: str, ground_truth: str) -> bool:
    predicted = predicted.strip().replace(" ", "")
    ground_truth = ground_truth.strip().replace(" ", "")
    if not predicted:
        return False
    if predicted == ground_truth:
        return True
    try:
        return abs(float(predicted) - float(ground_truth)) < 1e-6
    except ValueError:
        pass
    # fractional form, e.g. "22/7"
    try:
        num, den = predicted.split("/")
        return abs(float(num) / float(den) - float(ground_truth)) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False

assert grade_answer("63.0", "63") and grade_answer(" 63 ", "63")
assert grade_answer("63", "63")
assert grade_answer("22/7", "3.142857")  # |3.1428571-3.142857| < 1e-6
assert not grade_answer("", "63") and not grade_answer("64", "63")
assert not grade_answer("", "")
```

`"63.0"` passes via the float branch ($|63.0 - 63| = 0$); `"22/7"` gives $3.14285714\ldots$, within $10^{-6}$ of $3.142857$; the empty-string guard returns `False` before any comparison. A production verifier would add `sympy` equivalence on top, but these three branches already cover the most common grading escapes.
:::

### Exercise 8 — Diagnose a sick training run

A GRPO run on math problems shows: mean reward flat at $0.85$, entropy falling $4.2 \to 1.1$, mean length rising $400 \to 1400$ tokens, KL against the reference flat. Name the most likely failure, explain the mechanism in two sentences, and give two concrete fixes drawn from this chapter.

::: details Solution
Most likely failure: verbosity-driven entropy collapse — the policy has converged onto one long template that reliably harvests format/partial credit without solving more problems (reward flat, entropy collapsing, length tripling, KL flat because the drift is gradual, not a single large step). Mechanism: with sparse terminal reward, longer answers get more chances to contain a correct-looking span, so selection pressure favors the longest reproducible template and the distribution sharpens around it. Fixes from this chapter: (1) enforce per-response length normalization (`loss_type="grpo"` averaging) plus an overlong shaping penalty in the DAPO style so extra tokens cost rather than pay; (2) filter the data with dynamic sampling so batches keep discriminative groups, and gate on external quality (held-out accuracy with an independent parser) rather than training reward — if length rises while held-out accuracy is flat, stop and regularize.
:::
