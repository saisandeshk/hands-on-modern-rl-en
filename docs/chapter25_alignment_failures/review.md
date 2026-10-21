---
title: 25.R Summary and Exercises
---

# 25.R Summary and Exercises

This chapter asked one question at several levels of difficulty: when the reward goes up, is the work actually getting done better? The warehouse scanner, the RLVR verifier, the sleeper trigger, and the benchmark harness all fail in the same structural way — the measured proxy diverges from the real task — and they differ only in what evidence each conclusion requires.

## What You Should Now Be Able to Do

1. Distinguish **reward hacking** (the policy exploits a gap in the reward) from **specification gaming** (the formal rule itself licenses the behavior), and state what evidence each claim needs.
2. Explain why a rising RLVR curve can be spurious: contamination, verifier loopholes, format adaptation, and added test-time compute.
3. Design a **control experiment** (random rewards, fresh procedural problems, fixed sampling budget) that separates memorization from new reasoning.
4. Measure conditional risk with **paired trigger/clean comparisons** ($\Delta_{\text{trigger}}$) instead of relying on an average safety rate.
5. Describe how proxy–gold divergence (**reward overoptimization**) grows with optimization distance, and how a KL constraint and independent-utility early stopping slow it.
6. Evaluate honestly: multiple seeds, confidence intervals, IQM with a stratified bootstrap, prompt-sensitivity reporting, and judge-bias audits.
7. Version a full evaluation harness — data, prompt, budget, extractor, scorer, reward lineage — and state rollback and stop rules before training.

## Key Equations

- Optimal policy under the training reward: $\pi_R^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi}[R(\tau)]$ — the optimizer sees $R$, not the real utility $U$.
- RLVR objective: $J_V(\theta) = \mathbb{E}_{x,\,y \sim \pi_\theta}[V(x, y)]$ — swapping $V$ for $U$ gives the capability we actually want.
- Verifier generalization gap: $G_t = R_t^{\text{train}} - R_t^{\text{holdout}}$ — when it widens, stop adding steps and inspect trajectories.
- Conditional risk: $\Delta_{\text{trigger}} = P(\text{unsafe} \mid \text{trigger}) - P(\text{unsafe} \mid \text{clean})$ — a large value means the average safety rate is hiding the risk.
- Training effect on the gap: $\Delta_{\text{train}} = \Delta_{\text{trigger}}^{\text{after}} - \Delta_{\text{trigger}}^{\text{before}}$ — report it alongside the raw grouped rates, never instead of them.
- Overoptimization setup: train on $R_{\text{proxy}}$, evaluate on $U$ — the two curves share a timeline and their divergence is the stopping signal.
- KL-regularized reward: $R'(x, y) = R_{\text{proxy}}(x, y) - \beta D_{\mathrm{KL}}(\pi_\theta(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x))$ — $\beta$ prices leaving the reference distribution.
- Rare-loophole amplification: $P(\text{at least one loophole}) = 1 - (1-p)^n$ — search budget turns rare verifier holes into common ones.
- Normal-approximation interval: $p \pm 1.96\sqrt{p(1-p)/n}$ — an introductory demonstration only; use Wilson and paired methods for real reports.
- IQM over $M$ sorted run scores: average the middle 50% — robust to one lucky or broken seed.
- Multi-prompt average: $\mathrm{Score}(\pi) = \frac{1}{K}\sum_k \mathrm{Score}_{\text{prompt}_k}(\pi)$ — always report per-template scores and variance too.

## Common Pitfalls

1. **Reading the training curve as capability.** A rising proxy reward with a flat hidden-set curve is overfitting or exploitation, not learning. Plot both on one timeline before concluding anything.
2. **Concluding deception from one anomaly.** A single switched answer can be noise, distribution shift, or prompt ambiguity. Stable counterfactual switching plus later gains is the minimum bar — and unknown triggers still require a hidden-variant set.
3. **Reporting the best seed, prompt, or sample budget.** Each of these inflates the number without changing the model. Fix seeds (several), templates (several, pre-registered), and the sampling budget before comparing models.
4. **Trusting a judge from the same family.** Shared blind spots make correlated error invisible. Pair every model-judged result with order swapping, length controls, a different-family judge or human audit, and the judge's version.
5. **Patching one loophole and declaring victory.** Blocking the observed shortcut ("each item scores once") leaves the rest of the action space. Add failures to a permanent regression set and re-run old attacks after every repair.
6. **Losing the lineage.** An unversioned reward, extractor, or hidden set cannot be audited. Version data hashes, code commits, reward weights, and gate sets together, and keep a rollback target.

## Exercises

### Exercise 1 — Name the failure

A warehouse robot scans one barcode 200 times and earns reward 200 while shelving nothing. A second robot shelves 15 items using visual labels but never scans, earning reward 0. Which trajectory is a false positive and which is a false negative? Is this reward hacking, specification gaming, or both — and why?

::: details Solution
The 200-scan trajectory is a **false positive** (high $R$, zero $U$); the 15-item trajectory is a **false negative** (zero $R$, high $U$). It is **both**: reward hacking because the policy exploits the gap between scan count and tidying, and specification gaming because the formal rule ("more scans is better") itself licenses the 200-scan behavior. Hacking describes how the reward was obtained; gaming describes why the rule allowed it.
:::

### Exercise 2 — Verifier gap

At checkpoint $t$, a coding model scores $R_t^{\text{train}} = 0.92$ on training tests and $R_t^{\text{holdout}} = 0.55$ on held-out tests. At $t-1$ the two were $0.88$ and $0.54$. Compute $G_t$ and $G_{t-1}$, and state the recommended action: keep training, or stop and inspect?

::: details Solution
$G_t = 0.92 - 0.55 = 0.37$; $G_{t-1} = 0.88 - 0.54 = 0.34$. The gap widened by 0.03 while holdout accuracy moved only 0.01. **Stop and inspect** the newly appearing high-reward trajectories (Section 25.2.6): the gain is concentrated on the training verifier, the signature of overfitting or verifier exploitation.
:::

### Exercise 3 — Search amplifies loopholes

A verifier loophole triggers with probability $p = 0.02$ per candidate. You sample $n = 50$ candidates and keep the highest-scoring one. Compute the probability that at least one candidate hits the loophole, and explain in one sentence why Best-of-$n$ needs verifier hardening.

::: details Solution
$P = 1 - (1 - 0.02)^{50} = 1 - 0.98^{50}$. Since $\ln 0.98 \approx -0.020203$, $0.98^{50} \approx e^{-1.0101} \approx 0.364$, so $P \approx 1 - 0.364 = 0.636$ (about **64%**). Even a 2%-rare hole is found nearly two-thirds of the time with 50 candidates — selection pressure converts verifier blind spots into the reported answer.
:::

### Exercise 4 — Confidence interval by hand

Model A answers 60 of 100 problems correctly ($p = 0.60$). Compute the standard error and the 95% normal-approximation interval. Then state two situations in which you must **not** use this approximation.

::: details Solution
$\mathrm{SE} = \sqrt{0.6 \times 0.4 / 100} = \sqrt{0.0024} \approx 0.049$. The interval is $0.60 \pm 1.96 \times 0.049 \approx 0.60 \pm 0.096$, i.e. **50.4% to 69.6%**. Do not use it when (1) the sample is small or $p$ is near 0 or 1 (use the Wilson interval), or (2) two models are compared on the same problems (use McNemar's test or a paired bootstrap), per Section 25.5.1.
:::

### Exercise 5 — IQM versus the mean

Eight seeded runs score `[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.95]`, where `0.95` is one lucky seed. Compute the mean and the IQM, and say which better reflects typical performance and why.

::: details Solution
Mean: $(0.30 + 0.35 + 0.40 + 0.45 + 0.50 + 0.55 + 0.60 + 0.95)/8 = 4.10/8 = \mathbf{0.5125}$. IQM: drop the lowest two and highest two, average the middle four: $(0.40 + 0.45 + 0.50 + 0.55)/4 = 1.90/4 = \mathbf{0.475}$. The **IQM** better reflects typical performance because it is insensitive to the single lucky seed that drags the mean upward.
:::

### Exercise 6 — The average hides the trigger

An evaluation has 500 clean samples (2 dangerous) and 10 trigger samples (7 dangerous). Compute $p_{\text{clean}}$, $p_{\text{trigger}}$, $\Delta_{\text{trigger}}$, and the overall safety rate. What does the comparison show?

::: details Solution
$p_{\text{clean}} = 2/500 = 0.004$; $p_{\text{trigger}} = 7/10 = 0.70$; $\Delta_{\text{trigger}} = 0.70 - 0.004 = \mathbf{0.696}$. Overall: $510$ samples, $510 - 9 = 501$ safe, safety rate $501/510 \approx \mathbf{98.2\%}$. A 98% average safety rate coexists with a 70% danger rate under the trigger — the average dilutes the conditional risk, which is exactly why Section 25.3.4 requires grouped reporting.
:::

### Exercise 7 — KL penalty arithmetic

A response has proxy reward $R_{\text{proxy}} = 2.0$ and $D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) = 0.5$ at scale $\beta = 0.1$. Compute the regularized reward $R'$. Then explain why doubling $\beta$ is not always safer.

::: details Solution
$R' = 2.0 - 0.1 \times 0.5 = 2.0 - 0.05 = \mathbf{1.95}$. Doubling $\beta$ strengthens the anchor to the reference policy, which slows entry into untrusted regions — but the reference itself may be wrong, and a large $\beta$ also suppresses genuinely valuable new behavior. Choose $\beta$ against the independent-utility curve, not the KL value alone.
:::

### Exercise 8 — Coding: contamination check

Write a function `ngram_contamination(train_text, test_text, n=13)` that returns the fraction of test $n$-grams also present in training text, handling the empty-test edge case. Then describe one test showing it catches verbatim copying and one showing its limitation.

::: details Solution

```python
def _ngrams(tokens, n):
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def ngram_contamination(train_text, test_text, n=13):
    test_ngrams = set(_ngrams(test_text.split(), n))
    if not test_ngrams:
        return 0.0
    train_ngrams = set(_ngrams(train_text.split(), n))
    return len(train_ngrams & test_ngrams) / len(test_ngrams)
```

Verbatim test: `train_text` containing the full test sentence gives overlap `1.0` (for long-enough shared text). Limitation test: paraphrase the test sentence with synonyms — overlap drops toward `0.0` even though the content is copied, so approximate contamination needs embedding retrieval plus human review (Section 25.5.2).
:::

### Exercise 9 — Coding: stratified bootstrap interval

You have per-task scores for two models on the same 60 easy and 20 hard tasks. Sketch (in pseudocode or real code) a stratified bootstrap for the difference in means, and explain why you resample within difficulty bands rather than pooling all 80 tasks.

::: details Solution

```python
import numpy as np

def stratified_bootstrap_diff(easy_a, easy_b, hard_a, hard_b, B=10000, seed=0):
    easy_a, easy_b, hard_a, hard_b = map(np.asarray, (easy_a, easy_b, hard_a, hard_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        ie = rng.integers(0, len(easy_a), len(easy_a))
        ih = rng.integers(0, len(hard_a), len(hard_a))
        m_a = (easy_a[ie].sum() + hard_a[ih].sum()) / (len(ie) + len(ih))
        m_b = (easy_b[ie].sum() + hard_b[ih].sum()) / (len(ie) + len(ih))
        diffs.append(m_a - m_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), (float(lo), float(hi))
```

Resampling within bands preserves the 60/20 composition in every resample. Pooling all 80 tasks lets some resamples draw (say) 75 easy tasks, so the interval would reflect varying task mixtures rather than genuine score uncertainty — the same majority-group domination warned about in Section 25.5.1.
:::
