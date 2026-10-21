---
title: 18.R Summary and Exercises
---

# 18.R Summary and Exercises

This chapter moved from single-machine RL scripts to the systems that train production language models: role orchestration, reward plumbing, stability diagnosis, distributed execution, and data engineering. This page collects what to retain and gives you problems to check that you retained it.

## What You Should Now Be Able to Do

1. Decompose one industrial RL iteration into **generate → score → update → synchronize**, and say which device group performs each step.
2. Decide whether a job needs a single-machine stack (LlamaFactory, TRL), a distributed orchestrator (veRL, OpenRLHF, slime), or async agent infrastructure (AReaL, LlamaRL) — and justify the choice from model size and task-duration spread.
3. Combine verifiable and preference rewards safely: align scales by domain normalization, filter prompts to the learnable band, and mask truncated and tool-generated tokens out of the loss.
4. Estimate training cost with the $6ND$ FLOPs rule, split it into sampling, inference, and training buckets, and read rollout tokens/s, verifier throughput, and update utilization to find the bottleneck.
5. Diagnose a sick training run in causal order — data/rewards, policy change (KL, entropy), numerical updates, training system — and pick a fix at the right layer.
6. Explain where training-inference probability mismatch comes from (precision, kernels, MoE routing, stale versions) and how to correct it (align, log the true behavior policy, clip extreme ratios).
7. Compare colocated versus disaggregated deployment and synchronous versus asynchronous scheduling, including the staleness correction $\pi_\theta / \pi_{\theta_{\text{gen}}}$ and queue-health controls (batching, backpressure, recovery).

## Key Equations

- **GPU-hours estimate:** $\text{GPU-hours} \approx 6 N_{\text{active}} N_{\text{tokens}} / (\text{GPU\_FLOPS} \cdot \text{MFU} \cdot 3600)$ — total FLOPs divided by realized throughput.
- **PPO clipped objective:** $\mathcal{L}^{\text{CLIP}} = \mathbb{E}[\min(r_t \hat A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A_t)]$ with $r_t = \pi_\theta / \pi_{\text{old}}$ — limit how far one update moves the policy.
- **Stale-policy correction ratio:** $\rho_t^{\text{stale}} = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{gen}}}(a_t \mid s_t)$ — down-weight experience from old policy versions.
- **GRPO group advantage:** $\hat A_i = (r_i - \mu_{\text{group}}) / \sigma_{\text{group}}$ — a response is good only relative to its siblings; all-equal groups give zero signal.
- **Domain reward normalization:** $\tilde r = (r - \mu_{\text{domain}}) / \sigma_{\text{domain}}$ — put heterogeneous rewards on one scale before mixing.
- **KL and entropy:** $D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \sum_a \pi_\theta(a) \ln(\pi_\theta(a)/\pi_{\text{ref}}(a))$, $H = -\sum_a \pi_\theta(a)\ln\pi_\theta(a)$ — read together, they say whether the policy is moving too fast while losing exploration.
- **Full Adam training state:** $\approx 16$ bytes per parameter (2 weights + 2 gradients + 4 master + 4 + 4 moments) — the number that forces sharding past single-GPU memory.

## Common Pitfalls

1. **Reading only the reward curve.** A rising reward with flat held-out accuracy and growing length is reward gaming, not learning — check independent evaluation first.
2. **Fixing at the wrong layer.** Gradient clipping cannot repair a wrong reward, and data cleaning cannot repair a weight-sync bug; diagnose in causal order.
3. **Treating recomputed log-probabilities as behavior probabilities.** If rollout used FP8 vLLM and training recomputes in BF16, the PPO denominator is wrong before the update starts — log the rollout-time values.
4. **Training on unmasked trajectories.** Truncated completions and tool/environment returns in the loss teach the model to generate observations and fear cutoffs.
5. **Unbounded async queues.** Queue depth without a version bound is silent off-policy drift; monitor the oldest queued policy version, not just throughput.
6. **Restoring weights without sampler and queue state.** Recovery that forgets which prompts were served or which versions generated the queue corrupts the curriculum and provenance — checkpoint all four states together.

## Exercises

### Exercise 1: The Four Steps

Name the four steps of one industrial RL training iteration, and for a 70B Actor with a separate reward model, say which device group performs each step and what data crosses the network between steps.

::: details Solution
**Generate → score → update → synchronize parameters.** (1) Rollout GPUs generate responses from the current policy. Responses (tokens plus rollout-time log-probabilities and policy version) move to scoring. (2) Reward processes — verifier sandboxes or reward-model GPUs — consume responses and return scalar rewards and component scores. (3) Training GPUs compute advantages and gradients and update the Actor. (4) New weights broadcast back to rollout (and reference) GPUs, typically over NCCL; a LoRA adapter may be synced instead of full weights to shrink the transfer. The reward model and reference model may live on their own pools (disaggregated) or share the training pool (colocated).
:::

### Exercise 2: GPU-Hours by Hand

A 7B dense model trains on 2 billion tokens. Each GPU peaks at 989 TFLOPS with 40% MFU. Estimate the GPU-hours, showing each step.

::: details Solution
Numerator: $6 \cdot 7{\times}10^9 \cdot 2{\times}10^9 = 8.4{\times}10^{19}$ FLOPs. Realized throughput per GPU: $989{\times}10^{12} \cdot 0.4 = 3.956{\times}10^{14}$ FLOP/s. Seconds: $8.4{\times}10^{19} / 3.956{\times}10^{14} \approx 212{,}335$ s. Hours: $212{,}335 / 3600 \approx 59$ GPU-hours. This excludes communication, data loading, checkpointing, and rollout/verifier overhead.
:::

### Exercise 3: Memory by Hand

A 13B model trains in BF16 with Adam (FP32 master weights and moments). (a) How large is the full training state in GB? (b) With ZeRO-3 over 8 GPUs, roughly how much sharded state sits on each GPU, before activations?

::: details Solution
(a) Per parameter: 2 (weights) + 2 (gradients) + 4 (master) + 4 ($m$) + 4 ($v$) = 16 bytes. $13{\times}10^9 \cdot 16 = 2.08{\times}10^{11}$ bytes $\approx 208$ GB. (b) ZeRO-3 shards everything: $208 / 8 \approx 26$ GB per GPU, plus transient gathered parameters and activations. That fits an 80GB GPU with room for a reasonable micro-batch; without sharding, one GPU would need the full 208GB.
:::

### Exercise 4: Stale Ratio and Clipping

A token was generated with log-probability $-1.32$ under $\pi_{\theta_{\text{gen}}}$; the current policy assigns it $-1.51$. (a) Compute the correction ratio. (b) AReaL-style clipping bounds it to $[0.8, 1.2]$ — does clipping change this token's weight? (c) In one sentence, what goes wrong if the trainer instead uses a recomputed denominator that differs from the true behavior probability by this amount on every token?

::: details Solution
(a) $\rho = e^{-1.51 - (-1.32)} = e^{-0.19} \approx 0.827$. (b) No: $0.827$ lies inside $[0.8, 1.2]$, so the weight passes through unchanged; only extreme deviations get truncated. (c) Every gradient step then contains a systematic $\sim$17%-per-token importance error that clipping cannot remove, because the ratio is wrong before any update begins — the policy drifts as if off-policy data were on-policy.
:::

### Exercise 5: Group Advantage by Hand

A GRPO group of 8 responses earns rewards $[1, 1, 1, 0, 0, 0, 0, 0]$. Compute the group mean, standard deviation (population), and the advantage of a correct and an incorrect response. What happens to the advantages if all 8 responses are correct?

::: details Solution
Mean $\mu = 3/8 = 0.375$. Population variance: $(3 \cdot 0.625^2 + 5 \cdot 0.375^2)/8 = (1.171875 + 0.703125)/8 = 0.234375$, so $\sigma \approx 0.484$. Correct-response advantage: $(1 - 0.375)/0.484 \approx +1.291$. Incorrect-response advantage: $(0 - 0.375)/0.484 \approx -0.775$. If all 8 are correct, $\sigma = 0$, every advantage is $0/0$ (treated as zero) — the group contributes no gradient and its rollout compute is wasted, which is why dynamic sampling filters all-correct and all-wrong groups.
:::

### Exercise 6: Reading KL and Entropy

A two-action policy moves from $[0.80, 0.20]$ to $[0.95, 0.05]$ against a uniform reference $[0.5, 0.5]$. Verify that KL rises from $\approx 0.19$ to $\approx 0.49$ while entropy falls from $\approx 0.50$ to $\approx 0.20$, and explain why neither curve alone justifies intervening.

::: details Solution
Start: KL $= 0.8\ln(0.8/0.5) + 0.2\ln(0.2/0.5) = 0.8(0.470) + 0.2(-0.916) \approx 0.376 - 0.183 = 0.193$. Entropy $= -(0.8\ln0.8 + 0.2\ln0.2) = -(-0.178 - 0.322) \approx 0.500$. End: KL $= 0.95\ln1.9 + 0.05\ln0.1 = 0.95(0.642) + 0.05(-2.303) \approx 0.610 - 0.115 = 0.495$. Entropy $= -(0.95\ln0.95 + 0.05\ln0.05) = -(-0.0487 - 0.1498) \approx 0.199$. KL alone may just be healthy movement away from the SFT initialization, and falling entropy alone may mean a genuinely better solution was found; together — fast divergence plus collapsing exploration — they say the policy is locking in before it can still correct mistakes. (Caveat: uniformly easy data also collapses entropy, so check within-group difficulty too.)
:::

### Exercise 7: Coding — Masked Clipped Surrogate Loss

Implement `clipped_loss(old_logp, new_logp, adv, mask, eps)` in pure Python (lists of floats): per-token ratio $r = \exp(\text{new} - \text{old})$, per-token term $\min(r \cdot A, \text{clip}(r, 1-\epsilon, 1+\epsilon) \cdot A)$, mean over entries with `mask == 1`, return the negative mean (a loss to minimize). Then verify by hand: (a) `old = new = [ln 0.5, ln 0.25]`, `adv = [1.0, -0.5]`, `mask = [1, 1]`, `eps = 0.2`; (b) same but the second token has ratio $1.5$ with `adv = -0.5`.

::: details Solution

```python
import math

def clipped_loss(old_logp, new_logp, adv, mask, eps=0.2):
    terms = []
    for o, n, a, m in zip(old_logp, new_logp, adv, mask):
        if m != 1:
            continue
        r = math.exp(n - o)
        rc = min(max(r, 1.0 - eps), 1.0 + eps)
        terms.append(min(r * a, rc * a))
    return -sum(terms) / len(terms)
```

(a) Ratios are $[1, 1]$, both inside $[0.8, 1.2]$, so terms are $[1.0 \cdot 1.0, 1.0 \cdot (-0.5)] = [1.0, -0.5]$. Mean $= 0.25$; loss $= -0.25$. (b) Second token: $r = 1.5$ clips to $1.2$; with negative advantage, $\min(1.5 \cdot (-0.5), 1.2 \cdot (-0.5)) = \min(-0.75, -0.60) = -0.75$. Note the asymmetry: for $A < 0$, `min` keeps the _unclipped_ (more negative) term when the ratio exceeds $1 + \epsilon$ — clipping bounds the update the policy wants to make, and the loss takes the pessimistic side on each token.
:::

### Exercise 8: Coding — Bounded Queue with Retry Deduplication

Simulate one async step in pure Python: the rollout queue holds dicts with keys `id`, `version`, and `ok`. The trainer (at `cur_version`, with `max_staleness = 3` and `max_size = 4`) must: drop entries with `cur_version - version > max_staleness`, drop `ok == False` entries into a `failed` list (counted, not trained on), deduplicate retries by `id` (keep first), and take at most `max_size` entries. Implement `drain(queue, cur_version, max_staleness, max_size)` returning `(batch, failed, dropped_stale)`. Test it on: `ids/versions/ok = [(a,7,T),(b,2,T),(c,6,F),(a,7,T),(d,6,T),(e,5,T)]` with `cur_version = 8`.

::: details Solution

```python
def drain(queue, cur_version, max_staleness, max_size):
    batch, failed, dropped_stale, seen = [], [], 0, set()
    for e in queue:
        if cur_version - e["version"] > max_staleness:
            dropped_stale += 1
            continue
        if not e["ok"]:
            failed.append(e)
            continue
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        if len(batch) < max_size:
            batch.append(e)
    return batch, failed, dropped_stale
```

Hand trace with `cur_version = 8`, `max_staleness = 3`: (a,7): staleness 1, train. (b,2): staleness 6 > 3, stale-drop. (c,6): failed list. (a,7) again: duplicate, skipped. (d,6): train. (e,5): staleness 3, allowed, train. Batch ids $= [a, d, e]$ (3 $\le$ 4), failed $= [c]$, dropped_stale $= 1$. The trace shows all four mechanisms: staleness bound, failure accounting without training, retry dedup, and the size cap as backpressure.
:::
