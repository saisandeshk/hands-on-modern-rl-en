---
title: 14.5 Summary and Exercises
---

# 14.5 Summary and Exercises

This chapter replaced the RLHF loop (train a reward model, then run PPO) with a single contrastive objective trained on fixed preference pairs. Section 14.1 derived the DPO loss from the KL-regularized objective, Section 14.2 taught you to read its training logs, Section 14.3 derived it again from first principles and surveyed the offline-preference family (KTO, SimPO, IPO, ORPO, conservative DPO), and Section 14.4 ran two alignment experiments on a 0.5B instruct model.

## What You Should Now Be Able to Do

1. Write down the KL-regularized RLHF objective and derive the closed-form optimal policy $\pi^*(y\mid x) \propto \pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$ with a Lagrange multiplier.
2. Reparameterize the reward as $\beta\log(\pi_\theta/\pi_{\text{ref}})$ (up to a prompt-only constant), substitute it into the Bradley-Terry model, and arrive at the DPO loss.
3. Compute sequence log-probabilities with completion-only masks, form chosen/rejected log-ratios, and explain why the prompt cancels.
4. Read a DPO training log (loss, margin, accuracy, chosen/rejected rewards) and distinguish healthy learning from overfitting, drift, and length hacking.
5. Predict how $\beta$ reshapes training: margin scale, KL drift, and the gradient weight $w = \sigma(-u)$ on easy versus hard versus mislabeled pairs.
6. Choose between DPO, IPO, KTO, SimPO, and ORPO from data constraints (pairs versus thumbs-up/down, memory budget, small data, label noise) and defend the choice.
7. Run the chapter's DPO experiment loop (prepare pairs, test before, train, test after) and evaluate it with held-out prompts, compute-matched baselines, and external checks.

## Key Equations

$$\max_{\pi_\theta}\ \mathbb{E}_{x,y\sim\pi_\theta}[r(x,y)] - \beta\,D_{\mathrm{KL}}(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x))$$

The KL-regularized RLHF objective: earn reward without straying from the reference model; $\beta$ sets the leash length (larger $\beta$ means a shorter leash).

$$\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\left(\frac{r(x,y)}{\beta}\right)$$

Closed-form optimal policy: reweight the reference by exponentiated reward; $Z(x)$ only normalizes.

$$r(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \mathrm{const}(x)$$

Implicit reward: the policy ratio _is_ the reward signal (up to a prompt constant that cancels in comparisons), so no separate reward model is needed.

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]$$

The DPO loss: negative log-likelihood of the Bradley-Terry preference event, expressed purely in policy probabilities.

$$\nabla_\theta \ell = -\sigma(-u)\,\beta\,\nabla_\theta(\Delta_w - \Delta_l), \qquad u = \beta(\Delta_w - \Delta_l)$$

Per-sample gradient: the weight $\sigma(-u)$ concentrates updates on hard pairs and nearly ignores already-separated ones (and, dangerously, amplifies mislabeled ones).

$$\mathcal{L}_{\text{IPO}} = \mathbb{E}\left[\left(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)} - \frac{1}{2\beta}\right)^2\right]$$

IPO: stop at a fixed raw log-ratio gap of $1/(2\beta)$ instead of pushing the margin to infinity.

## Common Pitfalls

1. **Reading accuracy as quality.** Reward accuracy near 1.0 only says the model ranks training pairs correctly. Without held-out prompts and external task/safety checks, it is often memorization.
2. **Forgetting the reference is frozen.** If $\pi_{\text{ref}}$ accidentally trains (or is reloaded with a different tokenizer/template), every log-ratio is garbage. Catch a trainable reference by recomputing reference log-probs after an optimizer step -- they must be unchanged. (The fixed-batch recompute test in Section 14.2 catches nondeterminism and mask bugs, not a trainable reference.)
3. **Letting length decide the winner.** Summed log-probabilities penalize long completions. If chosen answers are systematically longer or shorter, check the length-normalized margin before celebrating.
4. **Setting $\beta$ blindly.** Tiny $\beta$ lets the policy drift into fluent nonsense; huge $\beta$ nails it to the reference so nothing learns. Both chosen and rejected rewards stuck near zero means the leash is too short.
5. **Trusting noisy labels.** DPO up-weights exactly the pairs it ranks backwards, so flipped labels dominate the gradient. With annotator disagreement, reach for conservative DPO/label smoothing, not more epochs.
6. **Skipping the SFT-on-chosen baseline.** If supervised fine-tuning on chosen completions alone (same token budget) matches DPO downstream, the contrastive term earned nothing and the pairs were just demonstrations.

## Exercises

### 1. Chosen-only DPO

Suppose you delete every rejected response and train with $\mathcal{L} = -\log\sigma(\beta\log(\pi_\theta(y_w\mid x)/\pi_{\text{ref}}(y_w\mid x)))$ on chosen completions only. What behavior does this objective encourage, and what capability from Section 14.1 is lost?

::: details Solution
With no rejected term, the "margin" is just the single log-ratio $\beta\Delta_w$, and the loss $-\log\sigma(\beta\Delta_w)$ decreases monotonically as $\Delta_w \to +\infty$: it pushes $\pi_\theta(y_w\mid x)$ up relative to $\pi_{\text{ref}}$ without bound, with weight $\sigma(-\beta\Delta_w)$ that never sees a negative example. There is no pressure to lower the probability of anything, so the model never learns what to avoid -- the contrastive structure of Section 14.1 (raise chosen _relative to_ rejected) is gone. In the limit this is preferred fine-tuning (SFT on chosen) with a KL-flavored scaling, inheriting SFT's weakness: sycophantic or unsafe completions are never explicitly penalized, so they can survive or even rise alongside the chosen ones.
:::

### 2. Compute a DPO batch by hand

With $\beta = 0.1$, a three-pair batch gives these sequence log-probabilities:

| pair | $\log\pi_\theta(y_w)$ | $\log\pi_{\text{ref}}(y_w)$ | $\log\pi_\theta(y_l)$ | $\log\pi_{\text{ref}}(y_l)$ |
| ---- | --------------------- | --------------------------- | --------------------- | --------------------------- |
| A    | $-12.0$               | $-12.5$                     | $-14.0$               | $-13.0$                     |
| B    | $-9.0$                | $-9.2$                      | $-8.5$                | $-9.0$                      |
| C    | $-10.0$               | $-11.0$                     | $-13.5$               | $-13.0$                     |

Compute each pair's margin $u$, loss $-\ln\sigma(u)$, the batch mean loss, mean margin, and reward accuracy.

::: details Solution
Pair A: $\Delta_w = -12.0 - (-12.5) = 0.5$; $\Delta_l = -14.0 - (-13.0) = -1.0$. Margin $u_A = 0.1(0.5 - (-1.0)) = 0.15$. $\sigma(0.15) = 1/(1+e^{-0.15}) = 1/1.8607 = 0.5374$, so $\ell_A = -\ln(0.5374) = 0.621$. Ranked correctly ($u_A > 0$).

Pair B: $\Delta_w = -9.0 - (-9.2) = 0.2$; $\Delta_l = -8.5 - (-9.0) = 0.5$. Margin $u_B = 0.1(0.2 - 0.5) = -0.03$. $\sigma(-0.03) = 1/(1+e^{0.03}) = 1/2.0305 = 0.4925$, so $\ell_B = -\ln(0.4925) = 0.708$. Ranked incorrectly.

Pair C: $\Delta_w = -10.0 - (-11.0) = 1.0$; $\Delta_l = -13.5 - (-13.0) = -0.5$. Margin $u_C = 0.1(1.0 - (-0.5)) = 0.15$, hence $\ell_C = 0.621$ and correctly ranked.

Batch mean loss $= (0.621 + 0.708 + 0.621)/3 = 0.650$; mean margin $= (0.15 - 0.03 + 0.15)/3 = 0.09$; accuracy $= 2/3 \approx 0.667$. Note the initial-loss sanity check: all-zero margins would give $\ln 2 \approx 0.693$, and pair B's negative margin correctly pulls the batch loss above the other pairs' values.
:::

### 3. Who dominates the gradient?

For per-sample margins $u \in \{2, 0, -1.5\}$, compute the DPO gradient weight $w = \sigma(-u)$ for each. If the $u = -1.5$ pair is mislabeled (the "rejected" answer is actually better), what happens if you keep training on it?

::: details Solution
$w(2) = \sigma(-2) = 1/(1+e^2) = 1/8.389 = 0.119$. $w(0) = \sigma(0) = 0.5$. $w(-1.5) = \sigma(1.5) = 1/(1+e^{-1.5}) = 1/1.2231 = 0.818$. The backwards pair carries roughly $0.818/0.119 \approx 6.9$ times the weight of the easy pair. If it is mislabeled, every step pushes the model toward the wrong ranking with the largest weight in the batch, and the weight never decays because fitting the wrong label keeps the margin negative. The fix is to cap such pairs' influence: filter or relabel them, or use conservative DPO so that already-separated pairs keep a non-vanishing loss floor instead of dropping out of the objective entirely.
:::

### 4. Derive the optimal policy

Starting from $\max_q \sum_y q(y)r(x,y) - \beta\sum_y q(y)\log\frac{q(y)}{\pi_{\text{ref}}(y\mid x)}$ subject to $\sum_y q(y) = 1$, derive $\pi^*(y\mid x)$ with a Lagrange multiplier.

::: details Solution
Form the Lagrangian $\mathcal{L} = \sum_y q(y)r(x,y) - \beta\sum_y q(y)\log\frac{q(y)}{\pi_{\text{ref}}(y\mid x)} + \lambda(\sum_y q(y) - 1)$. Differentiate with respect to one $q(y)$ (treating other values as constant): $\frac{\partial\mathcal{L}}{\partial q(y)} = r(x,y) - \beta(\log\frac{q(y)}{\pi_{\text{ref}}(y\mid x)} + 1) + \lambda = 0$, using $\frac{d}{dq}[q\log(q/c)] = \log(q/c) + 1$. Rearranging, $\log\frac{q(y)}{\pi_{\text{ref}}(y\mid x)} = \frac{r(x,y)}{\beta} + \frac{\lambda}{\beta} - 1$, so $q(y) = \pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)\exp(\lambda/\beta - 1)$. The factor $\exp(\lambda/\beta - 1)$ is $y$-independent; absorbing it into the normalizer $Z(x) = \sum_y \pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$ gives $\pi^*(y\mid x) = Z(x)^{-1}\pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$.
:::

### 5. The normalizer cancels; labels flip the sign

(a) From $r(x,y) = \beta\log(\pi^*(y\mid x)/\pi_{\text{ref}}(y\mid x)) + \beta\log Z(x)$, prove the Bradley-Terry probability $P(y_w \succ y_l\mid x) = \sigma(r(x,y_w) - r(x,y_l))$ does not involve $Z(x)$. (b) Show swapping $y_w, y_l$ maps margin $u \mapsto -u$, loss $\ell(u) = -\log\sigma(u)$ to $\ell(-u)$, and batch accuracy $a \mapsto 1 - a$.

::: details Solution
(a) Subtracting, $r(x,y_w) - r(x,y_l) = \beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)} + \beta\log Z(x) - \beta\log Z(x)$. The two $\beta\log Z(x)$ terms are identical (same prompt $x$) and cancel, leaving only policy ratios inside $\sigma$.

(b) The margin $u = \beta(\Delta_w - \Delta_l)$ is antisymmetric in $(w, l)$, so swapping gives $-u$. The loss becomes $-\log\sigma(-u) = \ell(-u)$, and since $\sigma(-u) = 1 - \sigma(u)$, a pair ranked correctly ($u > 0$) becomes ranked incorrectly and vice versa; each pair's 0/1 correctness flips, so the batch mean maps $a \mapsto 1 - a$. This is the fixed-batch swap test from Section 14.2.
:::

### 6. Summed versus length-normalized: a sign flip

Policy and reference per-token log-probabilities are constant within each completion. Chosen (10 tokens): policy $-0.40$, reference $-0.50$. Rejected (2 tokens): policy $-0.25$, reference $-0.50$. With $\beta = 0.1$, compute the DPO margin under (a) summed log-probabilities and (b) length-normalized (mean) log-probabilities. Which completion wins under each convention, and what is the lesson?

::: details Solution
(a) Summed: $\Delta_w = 10(-0.40) - 10(-0.50) = -4.0 + 5.0 = 1.0$; $\Delta_l = 2(-0.25) - 2(-0.50) = -0.5 + 1.0 = 0.5$. Gap $= 0.5$, margin $u = 0.1 \times 0.5 = 0.05 > 0$: chosen wins.

(b) Length-normalized: $\bar\Delta_w = -0.40 - (-0.50) = 0.10$; $\bar\Delta_l = -0.25 - (-0.50) = 0.25$. Gap $= -0.15$, margin $u = -0.015 < 0$: rejected wins.

Same data, opposite verdicts: the short rejected completion looks better per-token, while the long chosen completion accumulates a larger total gain. The lesson is Section 14.3's rule -- when completion lengths differ systematically between chosen and rejected, the summation convention itself picks winners, so always inspect length statistics and compare against the length-normalized margin.
:::

### 7. Coding: fixed-batch loss, gradient, and invariance tests

Using only `torch` (no model download), rebuild the core of `docs/chapter14_dpo/snippets/dpo-code-map.py`: write `dpo_forward(chosen_logps, ref_chosen_logps, rejected_logps, ref_rejected_logps, beta)` returning loss, margin, and accuracy from raw per-sample log-probability tensors. Then write tests showing (i) two recomputes give bit-identical results, (ii) swapping chosen/rejected maps margin $m \mapsto -m$ and accuracy $a \mapsto 1-a$, and (iii) the analytic gradient weight $\sigma(-u)$ matches `torch.autograd.grad` of the loss with respect to the margin. Use the batch from Exercise 2 as the fixture.

::: details Solution

```python
import torch
import torch.nn.functional as F

def dpo_forward(chosen_logps, ref_chosen, rejected_logps, ref_rejected, beta=0.1):
    chosen_logps = torch.as_tensor(chosen_logps, dtype=torch.float64)
    ref_chosen = torch.as_tensor(ref_chosen, dtype=torch.float64)
    rejected_logps = torch.as_tensor(rejected_logps, dtype=torch.float64)
    ref_rejected = torch.as_tensor(ref_rejected, dtype=torch.float64)
    margin = beta * ((chosen_logps - ref_chosen) - (rejected_logps - ref_rejected))
    loss = -F.logsigmoid(margin).mean()
    acc = (margin > 0).double().mean()
    return loss, margin, acc

# Fixture from Exercise 2 (per-sample sequence log-probs)
chosen  = [-12.0, -9.0, -10.0]
ref_c   = [-12.5, -9.2, -11.0]
rejected = [-14.0, -8.5, -13.5]
ref_r   = [-13.0, -9.0, -13.0]

loss1, m1, a1 = dpo_forward(chosen, ref_c, rejected, ref_r, beta=0.1)
loss2, m2, a2 = dpo_forward(chosen, ref_c, rejected, ref_r, beta=0.1)
assert loss1.item() == loss2.item() and torch.equal(m1, m2)  # (i) determinism
print(f"loss={loss1.item():.4f} margin_mean={m1.mean().item():.4f} acc={a1.item():.4f}")
# expect loss=0.6501 margin_mean=0.0900 acc=0.6667

# (ii) swap invariance (exact in float64; with float32 logits use a ~1e-6 tolerance)
_, m_swap, a_swap = dpo_forward(rejected, ref_r, chosen, ref_c, beta=0.1)
assert torch.allclose(m_swap, -m1) and abs(a_swap.item() - (1 - a1.item())) < 1e-12

# (iii) gradient weight: d/du[-logsig(u)] = -sig(-u)
u = torch.tensor([2.0, 0.0, -1.5], dtype=torch.float64, requires_grad=True)
l = -F.logsigmoid(u).sum()
g, = torch.autograd.grad(l, u)
assert torch.allclose(g, -torch.sigmoid(-u))
print("weights:", (-g).tolist())  # [0.1192, 0.5, 0.8176]
print("all invariance tests passed")
```

The printed loss $0.6501$ reproduces the hand computation in Exercise 2 ($0.650$), closing the loop between the math and the code map's `dpo_forward`.
:::

### 8. Coding: $\beta$ sweep and conservative DPO on the chapter's data format

Load the preference pairs produced by `code/chapter14_dpo/1-generate_data.py` (`output/preference_data.json` with `prompt`/`chosen`/`rejected` keys; fall back to two inline toy pairs if the file is absent). Assign each pair synthetic log-ratios $\Delta_w = +0.8$, $\Delta_l = -0.4$ (as if the policy already favors chosen). For $\beta \in \{0.01, 0.1, 1.0\}$ compute the mean DPO margin, the mean loss, and the accuracy; then implement the conservative DPO loss $\mathcal{L}_{\text{cDPO}} = -(1-\varepsilon)\log\sigma(u) - \varepsilon\log\sigma(-u)$ with $\varepsilon = 0.1$ and show it floors the loss on a confidently-correct pair ($u = 5$) while barely changing an uncertain one ($u = 0$). Pure Python/`math` only.

::: details Solution

```python
import json, math

def sig(z):
    return 1.0 / (1.0 + math.exp(-z))

try:
    with open("output/preference_data.json", encoding="utf-8") as f:
        pairs = json.load(f)
except FileNotFoundError:
    pairs = [
        {"prompt": "p1", "chosen": "polite answer", "rejected": "sarcastic answer"},
        {"prompt": "p2", "chosen": "corrects gently", "rejected": "blindly agrees"},
    ]
print(f"loaded {len(pairs)} pairs (schema: prompt/chosen/rejected)")

Dw, Dl = 0.8, -0.4  # synthetic log-ratios, identical for every pair here
for beta in (0.01, 0.1, 1.0):
    u = beta * (Dw - Dl)          # 1.2 * beta
    loss = -math.log(sig(u))
    acc = 1.0 if u > 0 else 0.0
    print(f"beta={beta:<5} margin={u:.4f} loss={loss:.4f} acc={acc:.1f}")
# beta=0.01 margin=0.0120 loss=0.6872 -- margin scales linearly with beta,
# beta=0.10 margin=0.1200 loss=0.6349 -- accuracy is 1.0 throughout because
# beta=1.00 margin=1.2000 loss=0.2633 -- sign(u) never changes for beta > 0.

def cdpo(u, eps=0.1):
    return -(1 - eps) * math.log(sig(u)) - eps * math.log(sig(-u))

for u in (5.0, 0.0):
    print(f"u={u}: DPO={-math.log(sig(u)):.4f} cDPO={cdpo(u):.4f}")
# u=5: DPO=0.0067 cDPO=0.5067 -- the confident pair can no longer collapse to
#      zero loss; the eps-weighted wrong-direction term floors it near
#      eps * 5 = 0.5. u=0: DPO=0.6931 cDPO=0.6931 -- identical, since both
#      terms equal ln 2 and (1-eps) + eps = 1.
```

Takeaway: $\beta$ rescales the margin but (for fixed log-ratios) cannot flip accuracy, which is why Section 14.4's $\beta$ table speaks of drift speed and stability rather than ranking direction; and cDPO's floor of roughly $\varepsilon|u|$ on extreme margins keeps already-separated pairs in the objective instead of letting their loss saturate at zero while all of the gradient comes from the hardest (possibly mislabeled) pairs.
:::
