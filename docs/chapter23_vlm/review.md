---
title: '23.R Summary and Exercises'
---

# 23.R Summary and Exercises

> **Learning path**: [23.1 Visual Reward Design](./vlm-challenges) → [23.2 Visual Reflection RL](./qwen3-vl-reflection) → [23.3 Hands-on: Multimodal GRPO](./vlm-grpo-hands-on) → [23.4 Hands-on: GeoQA Geometric Reasoning](./easyr1-geoqa) → **23.R Summary and Exercises**

This chapter kept the algorithm fixed and changed the input. The policy optimizer is still GRPO: sample a group of responses, normalize rewards within the group, and update toward the relatively better ones. What changes when an image enters the prompt is the reward design problem. A scalar outcome reward can no longer tell whether the vision encoder misread the image or the language model miscalculated, it cannot tell a genuine observation from a lucky guess, and it cannot see whether the model looked at the image at all. Every technique in this chapter -- process rewards, grounding checks, counterfactual controls, differentiated learning rates, visual reflection, tool-augmented re-observation -- is an answer to that one attribution problem.

## What You Should Now Be Able To Do

1. Explain why a correct VLM answer is weaker evidence of learning than a correct text-only answer, using the "guess without looking" scenario.
2. Write the GRPO within-group advantage formula and compute it by hand for a small reward group.
3. Design a three-part rule reward (accuracy + format + grounding) for a visual counting or detection task, with justified weights.
4. Decide whether to freeze, fully update, or differentially update the vision encoder for a given task, and predict the failure mode of each choice.
5. Implement an image-sensitivity (counterfactual) check that detects answers produced from language priors rather than pixels.
6. Read a VLM RL training curve (accuracy, format reward, grounding reward, no-image control, KL) and diagnose hallucination, encoder degradation, or length hacking.
7. Describe how visual reflection (observe, reason, verify) is trained: cold start, distillation, reasoning RL, and tool-integrated RL.
8. Run the shape-counting GRPO simulation in `code/chapter23_vlm/` and explain which parts transfer to a real VLM run and which do not.

## Key Equations

Group-relative advantage (the GRPO core used in Sections 23.3 and 23.4):

$$
A_i = \frac{R_i - \mathrm{mean}(R_1, \dots, R_G)}{\mathrm{std}(R_1, \dots, R_G)}
$$

Reading: each response is judged only against its siblings for the same prompt, so no Critic network is needed.

Simplified multimodal policy-gradient loss (Section 23.1):

$$
\mathcal{L}(\theta) = -\sum_{t \in \text{response}} A_t \, \log \pi_\theta\left(y_t \mid y_{<t}, x_{\text{text}}, x_{\text{image}}\right)
$$

Reading: gradients flow only through generated text tokens, but reach the vision encoder through attention -- so freezing versus updating the encoder is an optimizer decision, not a reward decision.

GeoQA composite reward (Section 23.4, `r1v.py`):

$$
R_{\text{overall}} = (1 - w) \, R_{\text{accuracy}} + w \, R_{\text{format}}, \quad w = 0.5
$$

Reading: half the reward pays for looking-and-reasoning in the right format, half for the right answer; either alone earns only half marks.

Tool-augmented visual reward with cost (Section 23.2):

$$
R = R_{\text{answer}} + \lambda_f R_{\text{format}} + \lambda_t R_{\text{tool}} - \lambda_c C_{\text{tool}}
$$

Reading: zoom and search tools earn their keep through valid, useful calls, but gratuitous tool use is taxed.

Image-sensitivity check (Section 23.1):

$$
\text{penalize if } \mathrm{answer}(x_{\text{image}}) = \mathrm{answer}(x_{\text{swapped}}) \text{ while correct on the original}
$$

Reading: an answer that does not change when the image changes was never grounded in the image.

Intersection-over-Union grounding reward (Visual-RFT, VLM-R1):

$$
\mathrm{IoU} = \frac{|\mathrm{pred} \cap \mathrm{gt}|}{|\mathrm{pred} \cup \mathrm{gt}|}
$$

Reading: localization quality becomes a rule-verifiable scalar the policy gradient can climb.

Constrained safety objective (autonomous driving, Section 23.1):

$$
\max_\pi \; \mathbb{E}\left[\sum_t r_t\right] \quad \text{s.t.} \quad \mathbb{E}\left[\sum_t c_t\right] \le d
$$

Reading: safety is a budget constraint on cumulative cost, not just another reward term to outvote.

## Common Pitfalls

1. **Trusting outcome accuracy alone.** A rising task curve with a flat or rising no-image control curve means the model learned to guess, not to see. Always report both.
2. **Unfreezing the vision encoder with a text-sized learning rate.** Full-rate RL gradients can destroy pretrained visual features; the model keeps talking but goes effectively blind. Use a ~10:1 text-to-vision ratio or freeze first.
3. **Rewarding reasoning length instead of evidence use.** If the reward model prefers detailed answers, the policy adds steps and adjectives. Track tokens per correct answer and evidence-hit rate alongside accuracy.
4. **Letting format reward dominate or die silently.** Too much weight and the model writes beautiful empty reasoning; a mismatched tag regex (as in the original `<thinkutan>` versus `</think\s*>` mismatch) zeroes the signal without any error. Log each reward component separately.
5. **Evaluating tools by call success instead of answer revision.** A valid zoom call that never updates the conclusion is theater. Save call arguments, returned observations, and the final answer together.
6. **Skipping the general visual probe.** Task accuracy can climb while general perception degrades. Keep a held-out general-VQA set and a blank-image control from before training starts.

## Exercises

### Exercise 1 (conceptual): Two failures, one reward

Section 23.1 describes answers B (visual misread, faithful report) and C (correct perception, reasoning slip), both earning reward 0. Explain in two to three sentences why the scalar outcome reward gives the optimizer no way to fix B and C differently, and name one concrete observable you would add to separate them.

::: details Solution
The outcome reward maps two different internal trajectories (mis-see versus mis-reason) onto the same scalar 0, so both receive the same negative group-relative advantage and the same blended gradient. The optimizer cannot assign the error to the vision encoder in case B and to the text decoder in case C. One separating observable is a visual grounding check: ask the model to report what it saw (counts, boxes, or a description) before reasoning. If the report is wrong, the error is visual (case B); if the report is right but the conclusion is wrong, the error is textual (case C). Alternatives: an IoU reward on predicted boxes, or attention/region overlap with the relevant image area.
:::

### Exercise 2 (numeric): GRPO advantage by hand

For one prompt, a group of $G = 4$ responses earns rewards $R = [1.5, 1.0, 0.3, 0.1]$. Compute the mean, the population standard deviation, and the four advantages. Show that the advantages sum to zero.

::: details Solution
Mean:

$$
m = \frac{1.5 + 1.0 + 0.3 + 0.1}{4} = \frac{2.9}{4} = 0.725.
$$

Deviations: $0.775, 0.275, -0.425, -0.625$. Squared: $0.600625, 0.075625, 0.180625, 0.390625$, summing to $1.2475$. Population variance $= 1.2475 / 4 = 0.311875$, so $\sigma = \sqrt{0.311875} \approx 0.5585$.

$$
A \approx \left[\frac{0.775}{0.5585}, \frac{0.275}{0.5585}, \frac{-0.425}{0.5585}, \frac{-0.625}{0.5585}\right] \approx [1.388, 0.492, -0.761, -1.119].
$$

Sum: $1.388 + 0.492 - 0.761 - 1.119 = 0.000$ (up to rounding). The zero sum is exact in general because $\sum_i (R_i - m) = 0$; dividing by the shared $\sigma$ preserves it. Responses 1 and 2 are reinforced, 3 and 4 suppressed, purely by within-group rank.
:::

### Exercise 3 (numeric): How guessing gets reinforced

A 4-option visual multiple-choice item is answered without looking at the image. In a GRPO group of 8, exactly 2 responses guess correctly (reward 1) and 6 guess wrong (reward 0). Compute the advantage of a correct guess and of a wrong guess, and explain what the policy learns.

::: details Solution
Mean $m = 2/8 = 0.25$. Population variance $= (2 \times 0.75^2 + 6 \times 0.25^2)/8 = (1.125 + 0.375)/8 = 0.1875$, so $\sigma = \sqrt{0.1875} \approx 0.4330$.

$$
A_{\text{correct}} = \frac{1 - 0.25}{0.4330} \approx 1.732, \qquad A_{\text{wrong}} = \frac{0 - 0.25}{0.4330} \approx -0.577.
$$

The two lucky guesses get a large positive advantage ($+\sqrt{3}$) and are reinforced, even though neither used the image. Repeated over many items, the policy learns "guess the prior" instead of "look at the image" -- the amplification mechanism of Section 23.1. This is why the chapter adds image-sensitivity checks and grounding rewards.
:::

### Exercise 4 (numeric): Visual tokens and IoU

(a) A ViT with $14 \times 14$ patches encodes a $448 \times 448$ image. How many patch tokens result, before any merging or special tokens? (b) A detector predicts box $[10, 10, 50, 50]$ (xyxy, pixels) and the ground truth is $[20, 20, 60, 60]$. Compute the IoU.

::: details Solution
(a) $448 / 14 = 32$ patches per side, so $32 \times 32 = 1024$ patch tokens. As Section 23.1 notes, merging, special tokens, and dynamic resolution change the realized count; 1024 is the order-of-magnitude estimate.

(b) Intersection: $[\max(10,20), \max(10,20), \min(50,60), \min(50,60)] = [20, 20, 50, 50]$, area $30 \times 30 = 900$. Each box area is $40 \times 40 = 1600$. Union $= 1600 + 1600 - 900 = 2300$. IoU $= 900 / 2300 \approx 0.391$. An IoU-type reward (Visual-RFT, VLM-R1) would score this prediction $\approx 0.39$ on the localization component.
:::

### Exercise 5 (derivation): Freezing blocks the visual gradient

Starting from the Section 23.1 loss, use the chain rule to show that freezing the vision encoder parameters $\phi$ (while updating language parameters $\theta$) zeroes the encoder update but leaves the text update intact, and explain why the reward function alone cannot achieve this separation.

::: details Solution
Write the vision encoder output as visual tokens $v = f_\phi(x_{\text{image}})$ and the loss as $\mathcal{L}(\theta, \phi)$ through the language model $\pi_{\theta}(\cdot \mid v, x_{\text{text}})$. By the chain rule,

$$
\nabla_\phi \mathcal{L} = \frac{\partial \mathcal{L}}{\partial v} \frac{\partial v}{\partial \phi}, \qquad \nabla_\theta \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \theta}.
$$

Freezing sets $\phi$ outside the optimizer's parameter groups, so no update $\phi \leftarrow \phi - \eta \nabla_\phi \mathcal{L}$ is applied (equivalently $\eta_\phi = 0$), while $\theta$ still steps along $\nabla_\theta \mathcal{L}$. The reward function only produces the scalar $R$ that scales the advantage; it has no addressing mechanism to route credit to $\phi$ versus $\theta$. Routing is done by the parameter-group partition (and differentiated rates $\eta_\phi \ll \eta_\theta$), exactly as the `setup_optimizer_with_lr_decay` sketch in Section 23.1 shows.
:::

### Exercise 6 (conceptual): Diagnose the curve

During GeoQA training you observe: `reward/format` rises to 0.9 by step 30 and stays there, `reward/accuracy` stays near 0.1, KL grows steadily, and a no-image control scores nearly the same as the with-image validation. Give the diagnosis in the order of the Section 23.1 checklist, and name two interventions.

::: details Solution
Diagnosis: Phase 1 (format learned) completed but Phase 2 (content) never started, and the no-image control matching validation means the policy scores from language priors without using the image -- a visual shortcut, possibly early hallucination reinforcement. KL growth without accuracy gain means the policy is drifting (exploring phrasing) without finding visually grounded solutions.

Interventions (any two well-justified): (1) add an image-sensitivity penalty or counterfactual control to the reward so prior-guessing is taxed; (2) add a grounding/process reward (describe-then-answer, IoU or evidence-hit term) so looking is paid; (3) lower the vision learning rate or briefly freeze the encoder to check whether RL gradients damaged perception (compare the general visual probe before/after); (4) audit the format/accuracy weight -- with $w = 0.5$ the model may satisfice on format; try $w = 0.1$--$0.2$ so accuracy dominates once format is learned. Also verify the format regex actually matches the template tags, or the format signal may be silently dead.
:::

### Exercise 7 (coding): Harden the multimodal reward

Using `code/chapter23_vlm/multi_modal_reward.py` as a base: (a) add an `image_sensitivity` term that subtracts 0.3 when the same response text would score as correct on a swapped image's ground truth (simulating prior-guessing), and (b) add a length guard that subtracts 0.1 when the response exceeds 300 characters without containing any step keyword. Run the built-in test cases and report which cases change score and by how much. The theoretical maximum stays 2.0; penalties can push totals below 0 -- clamp at 0 and justify the clamp in one sentence.

::: details Solution
Sketch (adapting the file's style):

```python
STEP_KEYWORDS = ['首先', '然后', '接着', '最后', '分析', '观察', '因为', '所以']

def reward_image_sensitivity(response, swapped_ground_truth):
    from multi_modal_reward import reward_correctness
    if reward_correctness(response, swapped_ground_truth) == 1.0:
        return -0.3  # correct on swapped labels: likely a prior guess
    return 0.0

def reward_length_guard(response):
    if len(response) > 300 and not any(kw in response for kw in STEP_KEYWORDS):
        return -0.1
    return 0.0

total = compute_total_reward(...)['total'] + reward_image_sensitivity(...) + reward_length_guard(...)
total = max(0.0, min(total, 2.0))
```

Expected behavior on the file's 8 cases: the perfect, short-correct, and reasoning-wrong cases are long-with-keywords or short, so the length guard does not fire; the sensitivity term fires only when you deliberately pass a swapped ground truth under which the response still extracts as correct (construct one by swapping two shapes' counts to match the response's numbers -- then the total drops by exactly 0.3). Clamping at 0 is justified because GRPO normalizes within the group, so negative outliers would distort sibling advantages; the penalty's job is ranking, not unbounded punishment. Report the per-case delta table from your run; the key check is that deltas are exactly $-0.3$ / $-0.1$ / $0.0$ as designed.
:::

### Exercise 8 (coding): Group size and advantage stability

Using `code/chapter23_vlm/vlm_grpo_train.py`: run `simulate_grpo_training` with `group_size` 2, 4, and 16 (same seed), and plot or tabulate `avg_advantage_std` across epochs for each setting. Explain (a) why `group_size=2` gives noisy, extreme advantages, (b) why the within-group std shrinks as the simulated policy converges, and (c) what this implies for choosing `rollout.n` in the Section 23.4 EasyR1 config.

::: details Solution
Run e.g.:

```python
from vlm_grpo_train import create_training_samples, simulate_grpo_training
samples = create_training_samples(num_samples=20, seed=42)
for g in (2, 4, 16):
    h = simulate_grpo_training(samples, group_size=g, num_epochs=5, seed=42)
    print(g, [round(v, 3) for v in h['avg_advantage_std']])
```

(a) With $G = 2$ the std is estimated from two points, so any reward gap becomes an advantage of $\pm 1$ regardless of its size ($|(r_1-r_2)/2| / (|r_1-r_2|/2) = 1$); magnitude information is destroyed and noise dominates. Larger $G$ estimates spread better and preserves ranking gradations. (b) As quality rises, group members are mostly correct responses with similar rewards (correctness 1.0 plus similar bonuses), so the within-group spread -- and hence the std -- falls; the annotation in `plot_training_curves` says the same. Late-training advantages shrink, which is why real runs raise the group size or sharpen rewards to keep learning signal. (c) This is the trade-off behind `rollout.n=5` in Section 23.4: larger $n$ stabilizes the advantage estimate but costs linear rollout memory and time; 5 is the starter value, 8--16 when affordable. Note the script simulates quality improvement rather than real gradient updates, so the trend demonstrates the statistics, not VLM learning.
:::
