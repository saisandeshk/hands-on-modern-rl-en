---
title: 16.8 Summary and Exercises
---

# 16.8 Summary and Exercises

## What You Should Now Be Able to Do

After working through Sections 16.1–16.6, you should be able to:

1. Tell **knowledge failures** apart from **sampling failures** for a wrong answer, and decide whether the fix is more training compute or more test-time compute.
2. Compute **coverage** $1-(1-p)^N$ and the **marginal gain** of one more candidate, and use them to justify a stopping point for parallel sampling.
3. Compute **GRPO within-group advantages** by hand for a small group, and explain why groups that are all-correct or all-wrong provide no learning signal.
4. Explain how an **outcome-only reward** can indirectly reinforce checking, verification, and method-switching without ever rewarding those behaviors directly.
5. Choose between **one long chain** and **several short answers** under a fixed token budget, and set a thinking budget that reserves room for the final answer.
6. Keep **internal reasoning**, **execution trace**, and **user-facing explanation** apart, and state what process alignment must check beyond final-answer correctness.
7. Use the distillation vocabulary correctly: **teacher vs student**, **off-policy distillation and rejection fine-tuning**, and how they differ from **on-policy distillation** (student rollouts scored token-by-token by a teacher; see Section 15.7).

## Key Equations

- **Coverage of parallel sampling**: $P(\text{at least one correct}) = 1-(1-p)^N$. _Reading: with single-attempt accuracy $p$, this is the ceiling any verifier can reach from $N$ independent tries._
- **Marginal gain of the next candidate**: $P(N+1)-P(N) = p\,(1-p)^N$. _Reading: each extra sample only helps when all previous ones failed, so the gain decays geometrically._
- **GRPO relative advantage**: $\hat A_i = \dfrac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}{\operatorname{std}(r_1,\ldots,r_G)+\epsilon}$. _Reading: how much better answer $i$ is than the average answer to the same problem, in units of the group's spread._
- **R1-Zero total reward**: $r_i = r_{\text{acc}}(x,y_i) + \lambda_{\text{fmt}}\cdot r_{\text{fmt}}(x,y_i)$. _Reading: correctness sets the direction, formatting only secures the parsing boundary._
- **Cost-sensitive adaptive reward**: $r_{\text{total}} = r_{\text{task}} - \lambda C$. _Reading: one unit of cost $C$ trades against $\lambda$ units of task reward; $\lambda$ sets the easy/hard dividing line._
- **Within-group length reward (Kimi k1.5)**: $\lambda_i = 0.5 - \dfrac{\operatorname{len}(i)-l_{\min}}{l_{\max}-l_{\min}}$. _Reading: among answers to the same problem, the shortest gets $+0.5$ and the longest $-0.5$ — compress length only relative to same-problem peers._
- **Off-policy distillation / rejection fine-tuning loss**: $\mathcal{L}_{\text{distill}}(\theta)=-\mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{kept}}}\sum_{t=1}^{|y|}\log\pi_\theta(y_t\mid x,y_{<t})$. _Reading: maximize the student's likelihood of each kept (verified, possibly short) teacher trace, token by token. This is the SFT/SeqKD objective on filtered data — distinct from the token-level teacher-scored OPD objective in Section 15.7._

## Common Pitfalls

1. **Comparing scores across sampling protocols.** A single-attempt score and a consensus-of-64 score differ in inference compute, not just in model quality. Always report the sampling and aggregation rule with the number.
2. **Mistaking coverage for accuracy.** $1-(1-p)^N$ is the probability that a correct answer _appears_; the system's accuracy is lower whenever the verifier or the vote picks wrong. No verifier, no best-of-N gain.
3. **Training on problems with no within-group contrast.** Batches where every answer is right (too easy) or every answer is wrong (too hard) give near-zero GRPO advantages. Spend sampling on problems the current policy sometimes solves.
4. **Confusing length with quality.** Extra thinking tokens only help while they fix errors the model can detect. Truncating a derivation mid-stream with a hard budget — without reserving tokens for the final answer — turns a good trace into an unusable one.
5. **Trusting visible reasoning as faithful, or hidden reasoning as safe.** Generated chains of thought can rationalize after the fact, and hidden traces can still drive unsafe tool actions. Check consistency between text, tool logs, and final behavior instead.
6. **Distilling unfiltered traces.** Training a student on teacher outputs that were never verified teaches it to reproduce the teacher's mistakes too. Verify first (rejection fine-tuning), then distill.

## Exercises

### Exercise 1 — Why Uniform Groups Teach Nothing

A GRPO group of $G=6$ answers to one problem receives rewards $(0,0,0,0,0,0)$. Another group receives $(1,1,1,1,1,1)$. What are the standardized advantages in each group, and why does neither group move the policy?

::: details Solution
For $(0,0,0,0,0,0)$ the mean is $0$ and every deviation $r_i-\text{mean}$ is $0$, so every $\hat A_i = 0$ (the $\epsilon$ only avoids division by zero). For $(1,1,1,1,1,1)$ the mean is $1$ and again every deviation is $0$, so every $\hat A_i = 0$.

The policy gradient scales with the advantage: zero advantage means zero update from this group. Intuitively there is nothing to contrast — no answer is better or worse than its peers — so the update cannot tell the policy what to do more or less of. This is why training must concentrate on problems of middling difficulty, where a group mixes right and wrong answers.
:::

### Exercise 2 — GRPO Advantages by Hand

For rewards $(1,0,0,0)$ compute the mean, the population standard deviation, and the four standardized advantages. Check that the advantages sum to zero.

::: details Solution
Mean: $(1+0+0+0)/4 = 0.25$.

Population variance: $\frac{(1-0.25)^2 + 3(0-0.25)^2}{4} = \frac{0.5625 + 3 \times 0.0625}{4} = \frac{0.75}{4} = 0.1875$.

Standard deviation: $\sqrt{0.1875} \approx 0.4330$.

Advantages: $\frac{0.75}{0.4330} \approx 1.732$ for the correct answer, and $\frac{-0.25}{0.4330} \approx -0.577$ for each wrong one, so $\hat A \approx (1.732, -0.577, -0.577, -0.577)$.

Zero-sum check: $1.732 + 3 \times (-0.577) = 1.732 - 1.732 = 0$ (up to rounding). Standardized advantages always sum to zero because they are deviations from the mean divided by a common scale.
:::

### Exercise 3 — Derive the Marginal-Gain Formula

Starting from $P(N) = 1-(1-p)^N$, derive $P(N+1)-P(N) = p(1-p)^N$ in two lines, and explain in words why the gain shrinks as $N$ grows.

::: details Solution
$P(N+1)-P(N) = [1-(1-p)^{N+1}] - [1-(1-p)^N] = (1-p)^N - (1-p)^{N+1} = (1-p)^N\,[1-(1-p)] = p(1-p)^N$.

In words: the $(N+1)$-th candidate adds value only in the event that all $N$ previous candidates failed, which has probability $(1-p)^N$, times the chance $p$ that the new one succeeds. Each extra candidate multiplies the remaining failure mass by another $(1-p) < 1$, hence geometric decay.
:::

### Exercise 4 — Coverage Arithmetic

A model has single-attempt accuracy $p=0.3$. Compute the coverage for $N=5$ and the marginal gain of the 6th candidate. Give both as percentages to one decimal place.

::: details Solution
$0.7^2 = 0.49$, $0.7^4 = 0.49^2 = 0.2401$, $0.7^5 = 0.2401 \times 0.7 = 0.16807$.

Coverage: $1 - 0.16807 = 0.83193 \approx 83.2\%$.

Marginal gain of the 6th candidate: $p(1-p)^5 = 0.3 \times 0.16807 = 0.050421 \approx 5.0\%$.
:::

### Exercise 5 — How Many Candidates for 95%? (Coding)

Write a short program that, given single-attempt accuracy $p=0.15$, finds the smallest $N$ with coverage at least $95\%$. Verify that $N-1$ falls short, and cross-check against the closed-form $N \geq \ln(0.05)/\ln(1-p)$.

::: details Solution

```python
import math

p = 0.15
target = 0.95
N = 1
while 1 - (1 - p) ** N < target:
    N += 1

print("smallest N:", N)                       # 19
print("coverage at N-1:", 1 - 0.85 ** (N - 1))  # ~0.9464 < 0.95
print("coverage at N:", 1 - 0.85 ** N)          # ~0.9544 >= 0.95
print("closed form:", math.log(0.05) / math.log(0.85))  # ~18.43 -> ceil 19
```

Working: $0.85^{18} = e^{18\ln 0.85} = e^{-2.9253} \approx 0.0536$, so coverage at $N=18$ is $1-0.0536 = 94.6\% < 95\%$; $0.85^{19} = 0.85 \times 0.0536 \approx 0.0456$, so coverage at $N=19$ is $95.4\% \geq 95\%$. The closed form gives $\ln(0.05)/\ln(0.85) \approx 18.43$, whose ceiling is $19$. All three agree.
:::

### Exercise 6 — Implement and Test `grpo_advantage` (Coding)

Implement `grpo_advantage(rewards, eps)` with NumPy using the population standard deviation, then assert it reproduces the three worked examples in Section 16.2 — $(1,1,0,0) \to (1,1,-1,-1)$, $(1,1,1,0) \to (0.577,0.577,0.577,-1.732)$, all-equal $\to$ zeros — and that every output sums to zero.

::: details Solution

```python
import numpy as np

def grpo_advantage(rewards, eps=1e-6):
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)

a1 = grpo_advantage([1, 1, 0, 0])
a2 = grpo_advantage([1, 1, 1, 0])
a3 = grpo_advantage([1, 1, 1, 1])

assert np.allclose(a1, [1, 1, -1, -1]), a1
assert np.allclose(a2, [0.57735, 0.57735, 0.57735, -1.73205], atol=1e-4), a2
assert np.allclose(a3, [0, 0, 0, 0]), a3
for a in (a1, a2, a3):
    assert abs(a.sum()) < 1e-9, a
print("all checks passed")
```

Notes: `np.std` defaults to the population standard deviation (`ddof=0`), matching the hand computation $\sqrt{0.25} = 0.5$ for Example 1 and $\sqrt{0.1875} \approx 0.433$ for Example 2. For the all-equal group the numerator is exactly zero, so the output is zero regardless of `eps`. The zero-sum property holds because the advantage is a mean-centered vector divided by a scalar.
:::

### Exercise 7 — Knowledge Failure or Sampling Failure?

Classify each case and name the cheaper fix:

(a) The model asked to prove convergence of $a_n = (1+1/n)^n$ writes a full derivation but flips a sign in the last algebra step.
(b) The model asked about a just-released API invents function names that do not exist, identically across 20 independent samples.
(c) A code model passes 9 of 10 hidden tests; the failing test needs only a one-line boundary fix the model produces when asked to revise.

::: details Solution
(a) **Sampling failure.** The method and knowledge are present; one unlucky generation step went wrong. Cheaper fix: test-time compute — resample, majority vote, or one round of sequential revision targeting the last step.

(b) **Knowledge failure.** The model never saw the new API, and repeated sampling reproduces the same hallucination (errors are correlated, not independent). Cheaper fix: training-side — fresh data, retrieval/tools, or a larger model; more sampling cannot invent unseen facts.

(c) **Sampling failure at the boundary.** The model is one local edit from passing. Cheaper fix: sequential revision reusing the 90%-correct solution, or best-of-N with the test suite as verifier — regenerating from scratch wastes the correct prefix.
:::

### Exercise 8 — Within-Group Length Rewards

Three answers to one problem have lengths $1000$, $1500$, and $2500$ tokens. Compute each $\lambda_i$ from $\lambda_i = 0.5 - (\operatorname{len}(i)-l_{\min})/(l_{\max}-l_{\min})$. Then consider two cases: (i) all three are correct; (ii) only the longest is correct. In which case does the shortest answer get the highest total reward, and why must correctness outweigh length in the other case?

::: details Solution
$l_{\min} = 1000$, $l_{\max} = 2500$, range $1500$.

- $1000$ tokens: $0.5 - 0/1500 = 0.5$.
- $1500$ tokens: $0.5 - 500/1500 = 0.5 - 1/3 \approx 0.17$.
- $2500$ tokens: $0.5 - 1500/1500 = -0.5$.

Case (i), all correct: the correctness term ties, so the length term decides — the shortest answer ($+0.5$) gets the highest total reward. This is the intended steering: among equally correct answers, prefer the concise one.

Case (ii), only the longest correct: the longest answer must still win overall despite its $-0.5$ length penalty, otherwise training would punish the only correct behavior and teach the model to be briefly wrong rather than verbosely right. Hence the correctness reward gap (here $1$ vs $0$) must be scaled to outweigh the length spread — exactly the "get it right first, then be concise" priority stated in Section 16.4.
:::
