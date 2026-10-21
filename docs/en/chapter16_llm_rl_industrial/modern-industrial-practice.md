# 18.3 Training Stability

> **Goal of this section**: Master the four-layer order for diagnosing training stability (data and rewards, policy change, numerical updates, training system), learn to read KL and entropy and loss and gradient norm, and be able to use public cases to judge which layer a stability tool acts on.

[18.2](./industrial-post-training) assembled the post-training loop: data → SFT → RL → evaluation → data feedback. Once the loop is running and you open the monitoring dashboard, the first thing you usually look at is the reward curve.

Consider a real scenario in mathematical RL training. A 7B model runs GRPO on the MATH dataset for 200 steps. The training reward rises steadily from $0.30$ to $0.80$ along a smooth curve. But the accuracy on the independent evaluation set MATH-500 stays at $42\%$ throughout, and the AIME pass rate even drops from $18\%$ to $15\%$. The average response length grows from $400$ tokens to $1{,}200$ tokens, a threefold increase. The team first suspected that the learning rate was too large and lowered it from $1\mathrm{e}{-6}$ to $3\mathrm{e}{-7}$. The reward curve kept rising and the evaluation still did not move. At step $312$ the loss suddenly became NaN, the gradient norm shot up to $10^4$, and all the parameters afterward were ruined. Until the NaN appeared, the reward curve had been climbing, and from the monitoring it looked as if training was still making progress.

The same rising reward has at least four completely different explanations behind it. The model may really have learned to solve problems. It may have learned only a reward loophole of the kind "writing longer makes a high score easier." The optimizer may already have pushed the parameters off and a NaN may be spreading. Or the generation side may sample with the old model while the training side computes probabilities with the new model, so the two versions do not match and the recorded log probabilities are not the probabilities at the time that trajectory was generated.

Looking at the reward curve alone, it is impossible to tell these explanations apart. During training, loss, gradient norm, KL, entropy, and independent evaluation have to be compared on the same timeline. This section checks layer by layer in the causal order of data and rewards, policy change, numerical updates, and the training system. At the end, four public cases (GLM, Llama 4, Seed-Thinking, Kimi K2) are matched against the typical problems of the four layers.

::: info Key Concept
Diagnosing training stability follows a four-layer causal order. Data and rewards decide what the model learns. Policy metrics describe how the model changes. Numerical metrics reflect whether the parameters can update normally. The training system decides whether the probabilities in the logs really come from the policy that generated the responses. Optimizers and clipping act only on the third layer (parameter updates). They cannot repair a wrong objective in the first layer, and they cannot replace version alignment in the fourth layer. Using a tool at the wrong layer trains a "stable bias" to be even more stable.
:::

---

## The Four-Layer Diagnostic Framework

With four curves laid out together, which one should be looked at first?

The answer lies in the causal order. Data and rewards decide what the model learns, policy metrics describe how the model changes, numerical metrics reflect whether the parameters can update normally, and the training system decides whether the probabilities in the logs really come from the policy that generated the responses. When an upstream layer goes wrong, the downstream curves are sure to be disturbed as well. Conversely, a downstream tool cannot fix an upstream problem. Gradient clipping can only limit the magnitude of a single update. If the reward function measures the wrong target, every clipped step still heads toward the wrong target.

In this order, the signals and first checks of the four layers are:

| Layer | Main signals | Check first | Corresponding tools |
| ---------- | ------------------------ | ---------------------------------- | ------------------- |
| Data and rewards | Reward rises, independent evaluation does not move | Reward rules, data duplication, evaluation contamination, length | Rewrite the reward, clean the data |
| Policy change | KL rises fast, entropy falls fast | Update size, KL constraint, sampling difficulty, version | Lower the learning rate, tighten KL |
| Numerical updates | Loss / gradient blow-up, NaN | Learning rate, precision, anomalous batches, clipping | Gradient clipping, optimizer |
| Training system | The two sides' probabilities disagree, versions are misaligned | Weight synchronization, tokens, precision, MoE routing | Align versions and recompute probabilities |

The diagnosis should go through this table from top to bottom: first confirm that the target the model is learning is correct, then see whether the policy has moved too fast, then check the single-step update, and finally verify the distributed system. This avoids two common mistaken remedies: using numerical tools to fix a reward problem, or using data cleaning to cover up a weight-synchronization error. Both remedies make training "look recovered" when in fact the fault has only been hidden in a deeper layer.

The next four sections expand each layer in turn: first what the signals look like, then a concrete numerical example, and finally the way to check this layer.

## Data and Rewards

The first layer asks only one question: is what the training reward measures the same thing as the capability we really want?

Return to the mathematical RL example at the start. Put the three monitoring numbers of the first 200 steps together: the training reward rises from $0.30$ to $0.80$, the MATH-500 accuracy stays at $42\%$, and the average response length grows from $400$ tokens to $1{,}200$ tokens. The reward more than doubled while the independent evaluation barely moved. This shows that what the model learned is an output form that earns reward more easily, namely writing longer, covering more edge cases, and being more likely to get a positive verdict from the parser. The target problem-solving ability did not improve.

Besides length as a confounding factor, this layer has two other common sources. One is that the model memorized the training tasks themselves: the problems appear repeatedly in the data, or the training set and the evaluation set overlap heavily (evaluation contamination), so the reward rises from memorization instead of generalization, and the independent evaluation naturally does not move. The other is that the reward was gamed. The answer parser accepts only a fixed format (for example, it must output `\boxed{}`), and the model can earn partial credit with a correct format even when the answer is wrong. Or the test harness has loopholes that can be bypassed (for example, it checks only the newly added tests and not whether the original tests were broken), and the model learns to delete the old tests to pass verification.

The check at this layer proceeds along this line: first check whether the training samples, the answer parser, the test environment, and the direction of the reward still correspond to the real objective, and then check training-set duplication and evaluation contamination. It must be done first, because once the target signal is wrong, the more stable the later optimization is, the more stably the model drifts away from the real objective. The optimizer faithfully amplifies the signal that training gives it and does not judge whether the signal is right.

## Policy Change

After confirming that the target the model is learning is correct, we look at how fast the model itself is moving. The two metrics at this layer are KL and entropy. KL measures how far the output distribution of the current model is from that of the reference model, and entropy tells how dispersed the output distribution is (that is, how much exploration the model still retains).

Start with an intuitive two-action example. Take the reference policy to be the uniform distribution $[0.5, 0.5]$, and suppose that early in training the current policy is $[0.80, 0.20]$. The model already leans slightly toward action A but still keeps a good deal of randomness. Substituting into the KL formula:

$$
D_{\mathrm{KL}} = 0.80\ln\frac{0.80}{0.50} + 0.20\ln\frac{0.20}{0.50}
= 0.80\times 0.47 + 0.20\times(-0.92) \approx 0.19.
$$

The entropy of the same distribution is $-(0.80\ln 0.80 + 0.20\ln 0.20)\approx 0.50$.

As training continues, suppose that after some steps the distribution concentrates to $[0.95, 0.05]$. Substituting into the formulas again, the KL becomes about $0.49$ and the entropy about $0.20$.

KL more than doubled: the model is getting further and further from the reference model. Entropy fell by more than half: the exploration space is getting narrower. Reading the two curves together gives "the policy moves too fast" a quantitative meaning: the increase of KL per unit of training step is too large, and at the same time entropy is sliding down at an accelerating rate. Neither signal alone is enough to call it an anomaly. A growing KL may just mean the model is moving away from the SFT initial distribution normally, and falling entropy may mean the model really found a better solution. Only when the two appear together does it show that the model, while moving away from the reference distribution, has also given up room for correction, and it is then almost impossible to get out of a wrong pattern later.

In real training the action is the whole vocabulary (tens of thousands to hundreds of thousands of tokens), and the two-action example still applies. The KL and entropy values computed over a vocabulary differ, but the curves are read in exactly the same way, and the key is the slope. When the KL curve steepens, the model is being pushed away from the reference model. When the entropy curve slides faster, exploration is collapsing.

When an anomaly appears at this layer, compare the old policy at sampling time with the new policy after the update, and check the learning rate, the clipping range, the KL constraint coefficient, and whether the rollout experience is already stale (if the policy versions are too far apart, the old data can no longer be used to update the new policy). There is one more easily overlooked factor: when the task is too simple, the model may also "reasonably" concentrate quickly on a few high-reward answers. If all the problems are too easy, every group is all correct, the within-group variance is zero, the model quickly settles on the single solution, and entropy naturally falls. In this case entropy falls because the task is too easy, and the update itself was not too fast. Data difficulty therefore has to be read together with the policy metrics, which is exactly the problem the Seed-Thinking case below deals with.

Finally, here are the full definitions of the two metrics, which will be used repeatedly in the following sections:

$$
D_{\mathrm{KL}}\bigl(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\bigr)
= \sum_a \pi_\theta(a)\ln\frac{\pi_\theta(a)}{\pi_{\mathrm{ref}}(a)},
\qquad
H(\pi_\theta) = -\sum_a \pi_\theta(a)\ln \pi_\theta(a)
$$

In the KL formula, $\pi_\theta$ is the output distribution of the current policy, and $\pi_{\mathrm{ref}}$ is the reference policy, usually the SFT model before RL starts. $\ln\frac{\pi_\theta(a)}{\pi_{\mathrm{ref}}(a)}$ is the log probability ratio of the same action under the two policies. $D_{\mathrm{KL}}$ is a sum weighted by the current policy's probabilities. It is 0 when the two distributions are identical and grows as the deviation grows.

The entropy $H(\pi_\theta)$ is the uncertainty of the output distribution: the more uniform the distribution, with action probabilities close to each other, the larger the entropy, and the more concentrated on a few actions, the smaller the entropy. A rapid fall in entropy shows that the model is stopping exploration too early and concentrating probability on a few response patterns.

## Numerical Updates

The first two layers both describe the model's behavior: what the model is learning and in which direction the policy is moving. The third layer looks at whether the parameters themselves can update normally. The loss reflects the change in the current training objective, and the gradient norm reflects how far this step is about to push the parameters. In normal training both fluctuate within a small range. A sudden blow-up, or a direct NaN, shows that a numerical problem has occurred in a single forward pass, backward pass, or parameter update.

NaN is the most dangerous signal at this layer, because it propagates: once one parameter becomes NaN, one matrix multiplication turns it into a whole patch of NaN, and within a few steps the whole model is ruined. What makes it worse is that a "warning period" usually precedes the NaN, in which the gradient norm keeps growing and the loss shows spikes. If nobody notices during the warning period, it is too late to roll back by the time the NaN appears.

There is a classic trick for locating a NaN: fix one batch, turn off data shuffling, and repeat the same forward and backward pass. If the NaN reproduces stably, the problem lies in the data of this batch or in some deterministic computation path (for example, a sample triggers a division by zero, a negative number appears inside a logarithm, or some attention score overflows). If it appears randomly, the problem is more likely in the precision (overflow under FP8/BF16), the operator implementation, or the optimizer state. After that, check in turn whether the learning rate is too large, whether low-precision computation overflows, whether the gradient clipping threshold is suitable, and whether there are anomalous samples (for example, samples with extreme length or too many repeated tokens).

AdamW, Muon, and the various clipping methods all act at this layer: they turn gradients into more controllable parameter updates. AdamW scales the gradient by its second moment so that the update step is more uniform across parameter directions. Muon orthogonalizes, flattening the singular values of the update matrix to prevent certain directions from being amplified again and again. Clipping limits the norm of a single update to prevent an anomalous batch from pushing the parameters far from the current solution. The boundaries of these tools should be clear: an optimizer cannot repair a wrong reward, cannot repair corrupted data, and cannot make the generation side and the training side automatically use the same model version. This gives a practical rule of thumb. When you see a loss spike, you can check the optimizer and clipping. When you see the reward and evaluation diverge and the length blow up, changing the optimizer further is pointless, since that is a first-layer problem.

## The Training System

When the first three layers are all normal and training still seems off, the last step is to verify that the rollout engine and the training engine are consistent. The trouble at this layer is that everything in the logs looks normal: the loss, KL, and entropy are all in reasonable ranges and the reward is rising too. The problem lies in the "record" itself: the probability used in training is not the probability at the time that trajectory was generated.

A typical asymmetry: the generation side uses a slightly earlier model version, FP8 precision, and a set of inference operators optimized for throughput (such as the speculative decoding and quantization kernels of vLLM and SGLang), while the training side uses the updated weights, BF16/FP32, and a different set of training operators. Even if the two sides read the same parameter file, the log probability of the same token can still deviate: precision truncation, kernel implementation differences, and different attention computation paths all bring small differences in log probabilities. MoE models are also affected by differences in expert routing: the routing implementations of the inference engine and the training engine differ slightly, the same token may activate different experts, and the forward results are fundamentally inconsistent.

Let us use concrete numbers to see how this deviation gets mixed into the gradient. Suppose the generation side samples a trajectory with the model of version $v_t$ and records the log probability of one token as $-1.32$. The training side has already updated to $v_{t+1}$ and recomputes the same token to get $-1.51$. The importance ratio in the policy gradient formula is

$$
\frac{\pi_{v_{t+1}}(a)}{\pi_{v_t}(a)} = e^{-1.51-(-1.32)} = e^{-0.19} \approx 0.83.
$$

This data has no sampling randomness at all, yet the ratio deviates from 1 by about 17%. This deviation comes from the version gap and the precision gap, but the training log treats it as part of the policy gradient and quietly mixes it into every update. Accumulated over thousands of steps, it is enough to push the policy off. This layer requires aligning, item by item, the model version, the tokenization, the log probability computation, the numerical precision, and the expert routing, to confirm that the old policy described by the training log is the policy that actually generated the trajectory. [18.4](./distributed-sync) goes on to expand on weight synchronization and training-inference consistency.

::: details Bonus: A Quick Diagnostic Checklist, How to Read the Four Curves

When you get a training monitoring chart, look in the following order to locate the layer quickly:

1. **First look at reward vs independent evaluation**: If the training reward rises steadily but the independent evaluation does not move (or drops), and the average length is growing → first layer, a data and reward problem. Check for reward loopholes, length confounding, and evaluation contamination.

2. **Then look at KL and entropy**: If KL doubles within a few hundred steps and entropy slides faster (for example from 2.0 down to 0.5) → second layer, the policy moves too fast. Check the learning rate, the KL coefficient, and the clipping range. If the data is too easy the same thing happens, so look at the right-versus-wrong ratio within groups.

3. **Look at loss and gradient norm**: If the loss shows a spike (for example jumping from 1.0 to 10+), the gradient norm blows up, or a NaN appears → third layer, a numerical update problem. Check anomalous batches, precision, gradient clipping, and the optimizer state.

4. **If the first three layers are all normal but the result is still abnormal**: Check whether the log probabilities of the generation side and the training side agree, and by how much the two sides differ on the same token → fourth layer, a training system problem. Check weight synchronization, precision alignment, and MoE routing.

A common mistake is "fixing from the bottom up": adding gradient clipping when you see a NaN, or lowering the learning rate when you see a large KL. When the real problem lies in the reward design, these remedies only make the model optimize more smoothly toward a flawed target.

:::

## Public Cases

With the framework built, let us see how the public reports of four companies fall onto the four layers. The four cases correspond to the typical problems of the four layers. When reading, first find which layer's instability signal it addresses, and then look at the specific method.

- **GLM-4.5 / GLM-4.6**: MoE routing, multi-stage training, and mode switching. The instability signals are expert load imbalance and capability regression between stages.
- **Llama 4**: Multimodality, long context, and evaluation versions. The instability signal is inconsistency between training scores and real-task performance.
- **Seed-Thinking**: Data difficulty, curriculum learning, and self-verification. The instability signals are overly sparse rewards and within-group advantages close to zero.
- **Kimi K2**: Optimizer updates and attention scores. The instability signals are loss spikes and anomalous gradient or attention values.

### GLM and Multi-Stage Training

GLM-4.5 (Zhipu AI, released in July 2025) and GLM-4.6 (released in October 2025) both use multi-stage training. This case is good for observing first-layer and fourth-layer problems: the routing-stability constraints that the MoE architecture places on the training system, and how multi-stage training prevents capability regression between stages.

First look at the constraints the architecture puts on post-training. GLM-4.5 is an MoE model with 355B parameters in total, of which only 32B are activated per forward pass. In RL updates, beyond the regular metrics (reward, KL, entropy), expert load and routing stability also need attention: some experts may be overused or completely idle, and local expert imbalance can still occur while the total loss looks normal. The same model also has to support both Thinking and Non-Thinking modes, learning when to expand its reasoning and when to answer directly. The training data therefore has to cover code generation, tool calling, and multi-step execution. The team published the weights, the training method, and part of the data, which provides an entry point for reproduction.

The post-training of GLM-4.5 has five stages:

```text
Phase 1: Base pretraining (MoE architecture)
  - 15T tokens of high-quality data
  - MoE: 355B total / 32B active
  - RoPE scaling supports long context

Phase 2: General SFT
  - Multilingual dialogue data
  - Tool-calling format training

Phase 3: Reasoning RL
  - Math, code, and reasoning tasks
  - GRPO + rule-based rewards
  - Self-validation integration

Phase 4: General RLHF
  - Dialogue quality and safety
  - Dual objectives of helpfulness / harmlessness

Phase 5: Unified Thinking / Non-Thinking
  - SFT on mixed data
  - Teach the model to switch modes
```

The order of the five stages has a clear dependency. SFT first establishes the basic dialogue and tool-calling behavior, and with a usable initial policy, reasoning RL can get meaningful reward differences on verifiable tasks. After accuracy has improved, general RLHF corrects dialogue quality and safety. Finally, mixed data unifies the two response modes in a single model. Multi-stage training brings a problem specific to the first layer: capability regression at stage transitions. Math accuracy goes up in the reasoning RL stage, but if too much chat preference data is mixed in during the general RLHF stage, the math ability may be washed away. Cross-stage distillation or data replay is then needed to preserve the old ability, and the on-policy cross-stage distillation published with GLM-5 deals with exactly this problem.

GLM-4.6 continues in three directions: reasoning length (supporting long reasoning trajectories of 100K+ tokens), tool range (covering more agent tools such as search, code execution, and file operations), and mode control (offering a finer Thinking Budget so users can control the reasoning budget), and it works together with the GLM-4.5V vision model for multimodal collaboration. It improves over GLM-4.5 on benchmarks such as AIME 2025, MATH-500, LiveCodeBench, and GPQA Diamond. When reading these results, the specific evaluation settings still have to be taken into account, and a single average score cannot be used to judge whether training is stable.

From the four-layer perspective, the GLM case offers three lessons. First, MoE and reasoning RL have to be debugged jointly: beyond reward and KL, expert load, routing distribution, and cross-card communication must be monitored, because local expert imbalance can still occur while the total loss looks normal (fourth layer). Second, the dual modes need separate evaluation: the accuracy, length, and cost of the Thinking mode cannot stand in for the response quality of the Non-Thinking mode, and they must be looked at separately during evaluation (first layer). Third, code and tool tasks need real execution: static answer scores cannot cover environment state, tool returns, and long-trajectory failures (first layer).

### Llama 4 and Evaluation Consistency

Llama 4 (Meta, released in April 2025) puts MoE, native multimodality, and long context into one series, and shows how changes in model architecture propagate into post-training data and evaluation methods. This case falls on the first layer: when the evaluation itself is contaminated, training scores keep rising while the real user experience does not improve with them.

The series contains three variants: Scout (an MoE with 109B total parameters / 17B active, 10M context), Maverick (400B total / 17B active, 1M context), and the unreleased Behemoth (2T total / 288B active). Every architectural change propagates to the training side. Native multimodality lets text and image tokens be processed together from the pretraining stage (Early Fusion), so post-training has to prepare text tasks, multimodal tasks, and cross-modal consistency evaluations at the same time. A multimodal reward must judge whether the response really used the image evidence, and a response that merely repeats the text of the question cannot score. The whole series uses MoE, in which every token passes through only some of the experts, so the training system has to handle expert routing and communication in addition. Scout's 10M-token context relies on iRoPE and sparse attention, and long-context evaluation has to check evidence retrieval, answer accuracy, and inference cost together, since fitting the input is only the minimum requirement.

The most important contribution of Llama 4 to the discussion of training stability is the public demonstration of how a "training score" can come apart from the "real experience." Maverick scored high on several benchmarks, but in actual use users generally felt it was not as good as Claude 3.5 / GPT-4o of the same period, and Meta later admitted that there was a gap between benchmark evaluation and the real experience. Further investigation found that the Maverick run on LM Arena was a specially optimized version, using chat template adjustments and prompt engineering, and the open-sourced Maverick is not the same as the arena version, so the two scores were in fact comparing two sets of evaluation configurations.

This falls squarely on the "evaluation contamination" check item of the first layer: when comparing models, the weight version, the chat template, the system prompt, and the sampling parameters must be recorded, otherwise the score difference may reflect only a configuration difference and have nothing to do with model capability. It belongs to the same class of problem as reward loopholes in training: when the evaluation configuration differs from the deployment configuration, the evaluation score is a flawed reward function, and an improvement on this score does not represent a real improvement in capability.

From the four-layer perspective, the engineering lessons of Llama 4 are threefold. MoE needs independent routing monitoring, since local expert load imbalance can still occur while the total loss looks normal (fourth layer). Early Fusion changes the shape of training samples, and text and images must form a checkable correspondence within the same task (first layer). Long context adds evaluation dimensions: beyond whether the input fits, one has to check whether the model can find the evidence and control the generation cost (first layer).

### Seed-Thinking and Data Difficulty

Seed1.5-Thinking (ByteDance Seed, April 2025) combines data curation, policy optimization, self-verification, and curriculum learning into one reasoning training pipeline. The instability signals it addresses lie between the first and second layers: rewards that are too sparse and within-group advantages close to zero. When a whole group is correct or a whole group is wrong, GRPO produces no gradient at all, and the compute for that batch buys no parameter update.

Returning to the conclusion of [16.2](../chapter19_reasoning/r1-zero-pure-rl-reasoning), the learning signal of GRPO comes from the reward differences among the responses in the same group. If a group of 16 responses is all correct (rewards all 1.0) or all wrong (all 0), the within-group standard deviation is zero, this problem produces no gradient in this batch, and the generation cost of the 16 rollouts is entirely wasted. Early in training, when the model is weak, an entirely wrong group is the norm, and when the data is too simple, an entirely correct group is a waste. Either case burns a lot of compute on samples with no learning signal.

Seed-Thinking handles this problem in four directions. First, data curation: the math data covers high-quality competition problems and automatically generated problems, graded by the pass rate of the base model, and the code data covers Codeforces, SWE-bench, and function-generation tasks. Grading by pass rate directly serves "some right and some wrong within a group." Problems that are too hard or too easy both fail to provide gradients, and only boundary problems (which the current model gets right about 30%-70% of the time) have the highest learning value.

Next come three improvements in policy optimization. Dynamic KL constrains strongly early in training and weakens later, letting the model first stand firm near the reference model and then loosening exploration, which prevents the policy from rushing off before it has learned an effective pattern. Adaptive clip adjusts the clipping range according to training progress. Group size scheduling uses a large group early on (for example 32 rollouts per problem, which makes a mix of right and wrong more likely) and a small group later (saving compute). All three act on the second layer and control the pace of policy movement.

The third item is Self-Verification, which is also the one most worth expanding. It has the model verify itself after generating an answer, and the reward scores two steps of behavior: a correct answer that passes verification earns $1.0$, a wrong answer whose error the model can identify earns $0.5$, and a wrong answer with no awareness earns $0$. Compare with the within-group advantage of GRPO. Without an intermediate level, the within-group rewards are $(1,0,0)$, and "wrong but noticed the error" and "wrong and unaware" are suppressed equally. With the 0.5 level added, the "noticed the error" response moves from the negative-advantage region to the middle, and the checking behavior itself is partially rewarded. This level of reward covers both the "answering" and the "checking" steps, and it also makes a gradient appear on hard problems that might otherwise have an all-zero group: some models cannot get the answer right but can notice that they got it wrong, and such responses are more valuable than "wrong yet confident" ones.

Finally comes curriculum learning, which handles the other end of the same problem. Early in training the model is weak and easily produces an entire group of $(0,0,\ldots,0)$. Curriculum learning sorts the training data by difficulty, lets the current model first earn effective rewards on easier tasks, and then raises the difficulty gradually, which reduces the cases where the advantage signal is close to zero. The basis for ordering difficulty is the pass rate of the current model, so the curriculum itself has to be updated as the model's capability changes: a boundary problem of the last stage may have become an all-correct problem in the next stage.

From the four-layer perspective, Seed-Thinking touches both the first and the second layers. Data difficulty grading and curriculum learning solve the "sparse reward" problem of the first layer (letting more samples produce effective gradients). Dynamic KL, adaptive clipping, and Self-Verification solve the policy-control problem of the second layer (giving exploration and exploitation a suitable rhythm). It achieves high results on AIME 2024, MATH-500, GPQA Diamond, and Codeforces. Product deployment still requires checking response length, inference latency, and general-capability regression, which goes back to the check items of the first layer.

### Kimi K2 and Limiting Anomalous Updates

The instability signals that Kimi K2 (Moonshot, July 2025) addresses lie in the third layer: loss spikes and anomalous gradient or attention values. It uses two tools at the same time, MuonClip acting on parameter updates and QK-clip acting on attention scores. Both tools are in the third layer, but they hold at different positions.

First look at the underlying Muon (proposed in February 2025). It is a momentum-orthogonalized optimizer that does two things: it accumulates directional information across consecutive updates (Momentum), and it orthogonalizes the update matrix (Orthogonalization). Plain SGD or Adam updates along the gradient direction, but the singular value distribution of the gradient matrix is often very uneven, with the components of a few directions far larger than the rest, and accumulated over thousands of consecutive steps the parameters are pushed too far along these directions. Orthogonalization readjusts the singular values of the update matrix, flattens the update magnitude across directions, and limits certain directions from being over-amplified.

MuonClip adds one more restriction on top of this: after orthogonalization it computes the norm of the update matrix, and if it exceeds a threshold, it scales the matrix proportionally, preventing an anomalous batch from causing an abrupt change in the parameters. It governs "how far this step pushes." The source of the anomalous batch (bad data, bad rewards) still has to be handled by first-layer checks. When the data contains a bad sample that makes the gradient explode, MuonClip only scales the exploded update back, and it does not point out which sample is responsible. It therefore still needs to be used together with learning-rate scheduling, gradient monitoring, and data checks.

QK-clip handles an anomaly at another position: the attention scores. In long contexts, the attention scores $QK^\top$ may keep growing. When the norms of the query and key grow, the dot product can grow to tens or even hundreds, and after entering the Softmax it concentrates probability excessively on a few tokens, makes gradients extremely small or large, and amplifies numerical errors in low-precision computation. QK-clip directly limits the range of $QK^\top$ in attention, clamping it to $[-clip\_value, clip\_value]$ before it enters the Softmax, and reduces outliers at the level of the attention computation. This is not a change at the policy level. It is purely a numerical safety measure.

Public results show that this combination lowered loss spikes from about once per 1T tokens to once per 10T tokens, and improved training speed by about 15% relative to Adam. The methodological value of this case is that it demonstrates "a stability tool must correspond to a specific fault layer": MuonClip limits anomalous update norms at the parameter-update layer, and QK-clip limits extreme attention scores at the attention-computation layer. The training log has to record the update norm and the attention statistics separately in order to tell which layer is taking effect. If QK-clip is applied to a first-layer problem (a loophole in the reward), it does not help at all: however normal the attention scores are, the model is still stably exploiting the loophole.

### Sources

- [GLM-4.5 Technical Report](https://arxiv.org/abs/2508.06471)
- [GLM-5 Technical Report](https://arxiv.org/html/2602.15763v1)
- [Llama 4 Technical Report](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Seed1.5-Thinking Technical Report](https://arxiv.org/abs/2504.13914)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://seed.bytedance.com/en/public_papers/dapo-an-open-source-llm-reinforcement-learning-system-at-scale)
- [Kimi K2 Technical Report](https://arxiv.org/abs/2507.20534)
- [The Muon Optimizer](https://arxiv.org/abs/2502.16982)

---

Having seen the four cases, we now put the "stable training methods" published by various companies back into the four-layer framework. They do not all act at the same position, so when reading a paper, first confirm the layer and then judge whether the methods can be combined.

## The Layers at Which Stability Methods Act

The methods published by various companies have many names (GRPO, GSPO, DAPO, CISPO, VAPO, MuonClip), and the problems they solve are not the same. Classified by the four layers:

- **Layer 1 (data and rewards)**: DAPO's Dynamic Sampling (filtering all-correct and all-wrong groups) and Overlong Reward Shaping (handling truncated samples). VAPO's Length-Adaptive GAE (normalizing reward scales across long and short responses). Seed-Thinking's Self-Verification (partial rewards increase gradient density). Skywork-OR1's entropy monitoring (preventing premature collapse). MiMo's test-difficulty-driven reward (fine-grained rewards by test difficulty).
- **Layer 2 (policy change)**: The within-group relative advantage and KL constraint of GRPO/GSPO. DAPO's Clip-Higher (asymmetric clipping that leaves room for exploration) and Token-Level Policy Gradient (preventing signal dilution over long sequences). MiniMax-M1's CISPO (clipping the importance sampling weights instead of the token updates). Hunyuan-T1's curriculum learning and policy reset. DeepSeek-R1's multi-stage, domain-by-domain RL.
- **Layer 3 (numerical updates)**: MuonClip (clipping the norm of the update matrix). QK-clip (clipping attention scores). Gradient clipping. Precision switching (mixed-precision training). Optimizer-state monitoring.
- **Layer 4 (training system)**: GLM-5/SAO's asynchronous RL with version tags. LongCat DORA's fully asynchronous streaming training. Training-inference log probability alignment. MoE routing consistency. Weight synchronization and checkpoint management.

When comparing schemes, first confirm the layer a scheme addresses and then judge whether it can be combined with another. Two schemes that both act on the policy layer (for example two different clipping strategies) may conflict, while a scheme on the policy layer plus a scheme on the numerical layer are often complementary. A common misconception is "the more tools, the more stable." If the reward itself has a loophole, piling on more optimizers and clipping only makes the model exploit the loophole more stably.

::: details Bonus: How to Quickly Tell Which Layer a Stability Method Acts On

When you get a new "stable training trick," you do not need to read the formulas first. Three questions locate the layer:

1. **Does it change the reward or the data?** For example, it changes the reward function, adds data filtering, or changes the sampling method → first or second layer.
2. **Does it change the gradient or the way parameters are updated?** For example, a new optimizer, gradient clipping, or attention truncation → third layer.
3. **Does it change distributed or training-inference consistency?** For example, a new synchronization method, version checking, or weight alignment → fourth layer.

If a method changes several layers at once (for example, DAPO changes both data sampling and clipping), look at which layer each component belongs to separately, and do not be bound by one algorithm name.

:::

## Additional Checks for Scaled-Up Training

The four-layer framework covers fault localization within a single training run. As the scale grows further, each layer sprouts new check items.

### Very Large Models and Long Context

As the total parameters and the active parameters per step keep growing, MoE routing and cross-card communication themselves become part of the stability metrics. Expert load imbalance is more than an efficiency problem: persistent skewed routing also leaves some experts undertrained and others overfit, which shows up as a sudden capability regression on specific kinds of tasks. A context of 10M+ tokens requires training to control GPU memory, attention numerics, and long-trajectory cost at the same time. Attention scores overflow more easily at extreme lengths, and the precision management of the KV cache also has to be monitored separately. Optimizer clipping still handles only parameter updates, and the other problems have to be solved together with data, reward, and system monitoring.

### Native Multimodal RL

Text, images, and environment actions enter the same training trajectory, and the reward also has to check cross-modal evidence: did the response really use the information in the image, or is it only guessing from text priors? Llama 4's Early Fusion shows one scheme for unifying modalities from the pretraining stage. Multimodal RL also needs replayable image, video, and interactive environments, otherwise a failed trajectory cannot be traced to a specific modality: did it misread the image, reason wrongly, or call a tool wrongly? The fixes for these three are completely different.

### Industrializing Agentic RL

Training tasks expand from software engineering to customer service, research, and computer operation. Agent trajectories need to record tool arguments, environment returns, file-system snapshots, and intermediate state, and trajectories of uneven length add the cost of environment scheduling, failure recovery, and asynchronous training. Most of the check items in this part fall in the fourth layer: version and state consistency between the environment and the training engine. After sandboxes expand from Docker containers to browsers, mobile devices, and desktop GUIs, the failure rate of the environment itself rises (dependencies that cannot be installed, page-load timeouts, tool crashes), and the rollout failure rate itself becomes a stability metric that must be monitored.

### Trade-offs Between Training Cost and Efficiency

Training cost consists of pretraining, data generation, rollout, model updates, and evaluation. Higher generation throughput (such as PD disaggregation and speculative decoding), smaller active parameters (MoE), and more accurate data filtering (concentrating compute on boundary problems instead of all-correct or all-wrong ones) can reduce wasted computation. Asynchronous training (such as GLM SAO and LongCat DORA) reduces GPU idling by decoupling generation from training, but it introduces version inconsistency and requires extra data-version management. For a small team, first reproducing the full four-layer monitoring on small verifiable tasks and only then deciding whether to scale up the model and cluster is a more reliable path than piling on compute directly.

## Section Summary

Training stability requires watching data and rewards, policy change, numerical updates, and the training system together, and the diagnosis also follows this causal order: first confirm the training objective is trustworthy, then see whether KL and entropy are abnormal, then check the loss, gradients, and optimizer, and finally verify the model version and probabilities of the generation side and the training side. The key points of this section:

1. **The four-layer causal chain**: data and rewards → policy change → numerical updates → training system. A reward that separates from independent evaluation together with a doubled length is the most typical first-layer instability signal. NaN and gradient explosion belong to the third layer. Inconsistent log probabilities between the two sides belong to the fourth layer.
2. **Reading the policy metrics**: when the distribution concentrates from $[0.80,0.20]$ to $[0.95,0.05]$, KL rises from 0.19 to 0.49 and entropy falls from 0.50 to 0.20, and "moving too fast" is just these two slopes. Data that is too easy also makes entropy fall, so it has to be read together with difficulty.
3. **The boundary of optimizers**: AdamW, Muon, and the various clipping methods only turn gradients into more controllable updates. Loss spikes and numerical anomalies are their business, and the separation of reward from evaluation is not. Changing the optimizer cannot fix a reward loophole.
4. **Version consistency**: when the log probabilities of the two sides differ by 0.19, even with no randomness in the data the importance ratio deviates from 1 by about 17%. This is a system-layer deviation and not a policy gradient. Training-inference precision, operators, and MoE routing all have to be aligned.
5. **Layer attribution of cases and methods**: GLM shows MoE routing and multi-stage regression (first and fourth layers), Llama 4 shows evaluation-version contamination (first layer), Seed shows data difficulty and the self-verification reward (first and second layers), and Kimi K2 shows update norms and attention scores (third layer). The methods published by various companies fall on different layers, so when comparing them, confirm the layer first and then talk about combination.

[18.4 Distributed RL Training](./distributed-sync) continues with the last layer of problems: when generation, reward, and training are spread over many GPUs, how does the system guarantee that data and model versions flow correctly?
