# 16.4 Hybrid Thinking and Budget Control

The previous section covered Test-Time Scaling: with parameters fixed, success rates can be raised through more sampling, more revision, and more search. But this brings a direct engineering problem: not every request can be given the full budget.

Imagine two requests arriving at an inference service at the same time:

1. "Translate hello into Chinese," which needs no expanded reasoning and can be answered directly.
2. "Prove that the sequence $a_n = (1+\frac{1}{n})^n$ converges," which needs several steps of derivation and also verification.

If both enter long-thinking mode, the translation request wastes a few seconds of latency and thousands of tokens of cost for nothing. If both are answered directly, the proof problem will most likely go wrong in an intermediate step.

This is the problem **Hybrid Thinking** solves: one model supports two modes at once, answering simple questions directly and expanding long reasoning for complex ones, while keeping the budget controllable. This section looks at four questions in turn: how the two modes coexist in one model, how the budget is set, whether the same budget should go to one long chain or to several short answers, and how redundant thinking steps are compressed once a model has learned long thinking.

<img src="../../chapter19_reasoning/images/hybrid-thinking-routing.svg" alt="A request is routed between direct answering and extended reasoning under a budget constraint">

## One Model Supporting Two Thinking Modes

An idea that comes easily is to maintain two models: a "fast model" for simple question answering and a "thinking model" for complex reasoning. But this has three costs:

- Two sets of weights must be deployed and maintained separately.
- Knowledge updates fall out of sync: what the fast model learns, the thinking model may not yet know, and vice versa.
- GPU memory use doubles, and so does cost.

The approach of Hybrid Thinking is to **share a single set of parameters, and use training samples or control markers to specify which mode the model should output in**.

### DeepSeek V3.1: A Dual-Interface Design

[DeepSeek V3.1](https://api-docs.deepseek.com/news/news250821), released in August 2025, is one example of making this dual mode a public product interface. On a single model version it exposes two API endpoints:

- `deepseek-chat`: Non-Think mode, answers directly
- `deepseek-reasoner`: Think mode, expands long reasoning

Three things can be confirmed from the public description:

1. **It really is the same model**, not two independent models.
2. **The mode is switched at request time**, with routing deciding which path to take.
3. **Knowledge and tool capabilities are shared**, and the two modes are updated in sync.

Public material does not disclose the details of each kind of training data and loss, so the specific training procedure cannot be inferred backward from the result of "dual mode." What can be determined is that the training data must cover both long-reasoning answers and direct answers, or one mode will crowd out the other. If the model is trained only on long text, its ability to give short answers degrades.

### Qwen3: A Systematic Dual-Mode Scheme

The earlier [Qwen3 technical report](https://arxiv.org/abs/2505.09388) (May 2025) proposed a more systematic scheme. The whole Qwen3 family (from 0.6B to 235B) supports both modes, and the training method is described more clearly.

Its core approach is **mixed training data plus template control**:

- The training data mixes two kinds of samples:
  - **thinking samples**: a full long reasoning chain plus the final answer
  - **non-thinking samples**: a short answer given directly, with no intermediate reasoning
- At inference time, the chat template or user settings tell the model whether it should currently generate long reasoning or answer directly.

The parameters are the same set, and the control markers activate different behavior modes: on seeing the thinking marker the model expands reasoning step by step, and on seeing the direct-answer marker it outputs the conclusion.

Sharing parameters avoids the cost of maintaining two models, but one problem remains unsolved: **who decides which mode to use right now?** Relying entirely on the user to choose manually gives a poor experience, and letting the model judge for itself may lead to wrong judgments. This question is left for Section 16.5 on adaptive thinking. In this section we first assume that the choice of mode has already been made by the user or a router, and look at how the budget is controlled.

### Thinking Budget: Controlling the Length of Thought

Even in thinking mode, the model cannot be allowed to derive without limit. Qwen3 proposed the **thinking budget**: at deployment, the reasoning process is allotted a cap on tokens, so that one model can form different "quality versus latency" operating points.

Different services may use different parameter names and implementations. The meaning can be shown with pseudocode:

```python
# Illustration of a thinking budget interface
response = client.chat.completions.create(
    model="qwen3-235b-a22b",
    messages=[{"role": "user", "content": "Prove that the square root of 2 is irrational"}],
    extra_body={"thinking_budget": 2000}  # the thinking part uses at most 2000 tokens
)
```

This cap directly affects three deployment metrics:

- **Small budget** (for example 500 tokens): frequent simple requests are not slowed by long thinking, and p99 latency stays under control.
- **Cost upper bound**: a single call has a ceiling on its worst-case cost.
- **Large budget** (for example 8000 tokens): competition and proof problems have enough room to expand the derivation, check for errors, and go back to revise.

One model with different budgets amounts to picking different operating points on the same "accuracy versus latency" curve: translation and summarization use a small budget and return quickly, while competitive programming and mathematical proofs use a large budget to secure accuracy.

There is one easily misused point here, though: **a budget cap does not mean the model will wrap up correctly**. If a proof is cut off midway, the model may not have time to write the final answer: it has thought for 3,000 tokens and is cut off before the conclusion is output. Training and deployment must therefore be handled separately:

- At training time, let the model learn to wrap up sensibly under different budget conditions.
- At deployment time, either reserve tokens for the final answer or trigger a wrap-up as the budget approaches.

## One Long Chain vs. Several Short Answers

Once the principle is settled that hard problems get more budget and easy ones less, there is still a choice to make: after a task enters the "needs more compute" category, how should the budget be spent?

For the same total budget of 4,000 tokens, there are two extreme ways to spend it:

1. **Thinking mode**: generate one long reasoning chain of about 4,000 tokens, going deep along a single path.
2. **NoThinking mode**: skip explicit long reasoning, generate eight short answers of about 500 tokens each, and vote for the best.

The April 2025 paper by Ma et al., [Reasoning Models Can Be Effective Without Thinking](https://arxiv.org/abs/2504.09858), systematically compared the two allocations.

### Comparison Under the Same Budget

The paper's experimental design is to **control the total token budget (or wall-clock latency) and compare the effect of the two allocations**. They used the DeepSeek-R1-Distill-Qwen family of models and compared across seven kinds of tasks, including math, formal proving, and code.

A concrete reading of "the same budget": with a total budget of 4,000 tokens,

- Thinking mode: 1 × 4000 tokens, keeping the complete, continuous derivation process.
- NoThinking mode: 8 × 500 tokens, eight independent attempts, with the answer chosen by voting or a verifier.

The experimental conclusions concentrate in the **low-budget range**, and they cannot be extrapolated to "NoThinking is better on all tasks."

### When Several Short Answers Are Better

One long chain concentrates the whole budget on a single solution method. The risk is that if the early direction is chosen wrongly, the following thousands of tokens all continue deriving on a wrong prefix, drifting further and further.

Several short answers spread the budget over different starting points and cover a richer set of solution methods. This depends on two conditions:

1. A short answer must itself have some single-attempt success rate ($p$ cannot be too low).
2. The system must be able to judge which answer is better (the verifier is reliable, or the voting mechanism is effective).

When both conditions hold, as in knowledge question answering and simple math problems, voting over several short answers may be more cost-effective than one long chain.

Conversely, if the task requires **continuous state retention**, a long chain that can keep intermediate state is more valuable. When writing a complex algorithm, functions defined earlier are called later. In a multi-step proof, each step depends on the conclusion of the previous one. A long proof cannot be split into eight independent short proofs and voted on, because the later derivation depends on the earlier conclusions.

So this is a question of choosing a compute allocation according to task type:

- **Does it need continuous derivation dependencies?**
  - Better for Thinking (one long chain): yes (proofs, code, multi-step planning)
  - Better for NoThinking (several short answers): no (question answering, multiple choice, simple calculation)
- **Single-attempt success rate of a short answer?**
  - Better for Thinking (one long chain): low
  - Better for NoThinking (several short answers): medium to high
- **Is there a reliable verifier?**
  - Better for Thinking (one long chain): works even without one
  - Better for NoThinking (several short answers): preferably yes

## Long2Short: Compressing Long Reasoning Chains

Long-thinking training has a side effect: to raise accuracy, the model may learn to check repeatedly, repeat itself, and even go around in circles. Solving the length problem with a hard cutoff at deployment is risky: if the cutoff falls before the conclusion, the model has no time to output the answer. What is needed is to **let the model learn at the training stage to keep necessary steps and delete redundant ones, without losing accuracy**. This is the long2short problem.

### Why Reasoning Chains Get Longer

In reasoning RL training, answers growing longer is a common phenomenon, with three direct causes:

1. **The reward for correct answers encourages long trajectories**: a longer reasoning chain means more chances to check, a higher probability of a correct answer, and the model tends to think longer.
2. **Reflection and verification behaviors get reinforced**: checking really can find errors, so it is selected indirectly by the outcome reward.
3. **There is no explicit length constraint**: when the reward contains only "correct" and nothing for "saving tokens," the model has no incentive to compress its answers.

Increased length is valuable only when it really improves accuracy. Repetition and ineffective checking only add deployment cost. Long2short aims to transfer the planning and correction abilities learned in long reasoning into shorter answers.

The Kimi k1.5 technical report (2025) compared four methods, each working at a different level:

- **Model weight merging**
  - Level of action: parameters
  - Core idea: directly average the weights of a long-CoT model and a short-CoT model
- **Shortest Rejection Sampling**
  - Level of action: supervised data (SFT)
  - Core idea: from several correct answers, pick the shortest one for supervised fine-tuning
- **DPO**
  - Level of action: preference data
  - Core idea: treat "short and correct" as chosen and "longer or wrong" as rejected
- **long2short RL**
  - Level of action: RL reward
  - Core idea: add a within-group length reward in RL and shorten the rollout cap at the same time

### Supervised Compression

The first three methods all start from "already having a long model that can solve the problem correctly but outputs very long answers," and arrive at a "short model that outputs shorter answers and can still solve the problem correctly."

**Shortest Rejection Sampling** is the most direct: for the same problem, have the long model generate several answers, filter out the correct ones, and pick the **shortest** among the correct answers as SFT data. What the short model learns are examples that are "both correct and as concise as possible."

**DPO** (Direct Preference Optimization) takes the preference-learning route: for the same problem, the "short and correct" answer serves as the preferred sample (chosen), and the "longer but also correct" or "wrong" answers serve as the rejected samples. The preference the model learns is: when all can solve it correctly, prefer the short one.

**Model weight merging** directly averages the weights of a long-CoT model and a short-CoT model. It does not generate new compressed text, and it does not guarantee that each ability of the merged model lies between those of the two, so a full evaluation is required after use.

### Compressing with Reinforcement Learning

The finest-grained method is long2short RL: continue training from a model that has already reached a good balance between accuracy and length, shortening the permitted rollout length cap while adding a length term to the reward.

Kimi k1.5's length reward makes a relative comparison within the group of answers to the same problem, **avoiding setting one fixed target length for all answers** (for example, requiring every problem to be answered within 1,000 tokens):

1. Sample $G$ answers for the same problem and first mark which are correct and which are wrong.
2. Compare lengths among the correct answers: the shorter the answer, the higher its length reward.
3. A wrong answer cannot earn a positive reward merely for being "shorter": get the answer right first, then talk about conciseness.

The length term coefficient is

$$
\lambda_i=0.5-\frac{\operatorname{len}(i)-l_{\min}}{l_{\max}-l_{\min}},
$$

where $\operatorname{len}(i)$ is the token length of the $i$-th answer, and $l_{\min}$ and $l_{\max}$ are the shortest and longest lengths in this group. The shortest correct answer gets $+0.5$ and the longest gets $-0.5$. If all answers in a group have the same length, the length term is set to 0.

Let's walk through this formula with three answers to the same problem. Suppose their lengths are 800, 1200, and 2000 tokens, so $l_{\min}=800$ and $l_{\max}=2000$:

- **Answer A** is 800 tokens long. Its normalized length is $0$, so $\lambda_i=0.5$.
- **Answer B** is 1200 tokens long. Its normalized length is $400/1200\approx0.33$, so $\lambda_i\approx0.17$.
- **Answer C** is 2000 tokens long. Its normalized length is $1$, so $\lambda_i=-0.5$.

Consider the final reward in two cases:

- **Case 1: A and B are correct, C is wrong.** The length term and the correctness term point the same way: A, the shortest and correct, gets the highest reward, and C, the longest and wrong, suffers both the length penalty and the error penalty. This is the ideal case.
- **Case 2: C is the only correct one.** Here the correctness term must outweigh the length term: although C is long, it is the only correct answer and should still receive a high reward. The length reward and the correctness reward together achieve this: first make sure the answer is correct, and the length reward prefers shorter answers only among those that are all correct.

::: details Why use relative length within the group and not absolute length
Why not directly reward "all answers stay under 1,000 tokens"? Because the length different problems need is inherently different: a simple multiple-choice question may be explained in 200 tokens, while a competition proof may need 5,000. If a fixed absolute length target is set, the model may omit necessary steps just to stay short.

Within-group relative comparison solves this: lengths are compared among the several answers to the same problem and are not compared directly across different problems. What the model learns is "for this problem, be as short as possible provided the answer is right," and not "every problem must be short."
:::

The Kimi k1.5 report says its long2short RL model reaches a Pass@1 of 60.8 on AIME 2024 using only 3,272 tokens on average. This number is only one quality-versus-length operating point in the experiments. When choosing a model to deploy, one should plot the complete "length versus accuracy" curve, and not compare the score at a single point.

### Combining Training-Time Compression with Inference Budgets

Long2short and the thinking budget are two complementary layers of control, acting at different stages:

- **Long2short is compression at the training stage**: it changes the model's own output distribution so that the model tends to generate more compact reasoning chains, moving the usual operating point toward shorter lengths.
- **The thinking budget is control at the inference stage**: it does not change the model and only sets a hard cap on the server side, truncating the worst-case tail length.

The two can be combined: first train the model to be more efficient with long2short RL, then set a safety cap with a thinking budget at deployment. The former changes the distribution and the latter cuts the tail.

## Modes and Budgets at Deployment

After training the dual-mode ability and compressing the long chain more efficiently, the details of the service interface come last. Once a request enters the service, three decisions must be made in turn:

1. **Should it enter thinking mode?**
2. **After entering thinking, how much budget at most?**
3. **When the budget is nearly used up, how do we ensure a complete answer is output?**

Once the mode switch is settled, three more engineering problems must be handled well.

### Who Decides Whether to Think

There are three ways to implement mode selection:

- **Explicit user choice**: pass a flag in the API parameters, and let the caller decide.
- **Automatic router judgment**: train a small router from task type, historical success rate, and latency target.
- **The model judges for itself**: let the model read the input once and decide for itself whether to expand its thinking.

Whichever approach is adopted, the decision should be recorded, and the cost of the two kinds of error should be evaluated separately:

- **Wrongly turning on long thinking** (a simple question enters thinking mode): what is wasted is latency and cost.
- **Wrongly turning off thinking** (a hard question gets a direct answer): what is lost is the task success rate.

The costs of the two errors are quite different, and the routing thresholds should differ accordingly: better to let some simple questions spend a little extra thinking than to misjudge a hard question as simple and answer it wrongly.

### Answers in the Two Modes Must Be Consistent

For one deterministic question, such as "What is 1+1?", if Think mode answers 2 and Non-Think mode answers 3, then at least one of the modes is unreliable.

At training time, shared data or a consistency objective can be added to reduce such conflicts, but at deployment we cannot assume that "shared parameters naturally guarantee consistent answers." The outputs of the two modes must be evaluated separately, with consistency checks on key factual questions.

For open-ended questions (such as writing a passage of prose), it is normal for the two modes to produce different outputs, and the evaluation should focus on factual correctness and task completion, not word-for-word agreement.

### How to Wrap Up When the Budget Is Nearly Used Up

The problem with a hard cutoff is the greatest: the answer stops in the middle of "next we need to prove...", the conclusion is never output, and the user gets incomplete text.

A more robust interface design does three things:

1. **Reserve tokens for the final answer**: with a thinking budget of 4,000, thinking uses at most 3,500, leaving 500 for the final answer.
2. **Trigger a wrap-up as the budget approaches**: monitor tokens used, and when close to the cap, have the model begin to conclude rather than continue expanding new steps.
3. **Cover different budget conditions in the training data**: let the model see tight-budget situations during training and learn to summarize within a limited length.

If the budget runs out and there is still no reliable conclusion, the system should return an unfinished status, or fall back to a simpler answer, and **must not package an arbitrary guess from an intermediate step as a definite answer** returned to the user.

## Section Summary

Hybrid Thinking solves a practical deployment problem: do not let easy questions waste thinking cost, and do not let hard questions be answered wrongly for lack of budget.

- **One model, two modes**: DeepSeek V3.1 exposes Think/Non-Think interfaces on the same model, and Qwen3 lets the two outputs coexist through mixed training data and template markers. Sharing parameters avoids the cost of maintaining two models, but mode selection still needs a router or a user's judgment.
- **Thinking Budget**: set a token cap on thinking, so one model forms different quality-versus-latency operating points. Note the problem with hard truncation: space must be left for the final answer.
- **Trade-off between depth and breadth**: the NoThinking experiments show that in the low-budget range, voting over several short answers can compete with one long chain, provided the short answers have enough single-attempt success and the aggregation is reliable. Tasks that need continuous derivation still suit a long chain better.
- **Long2short compression**: four methods compress reasoning chains at four levels: parameters, supervision, preference, and RL. What Kimi k1.5's within-group length reward $\lambda_i$ learns is "prefer the shorter one among equally correct answers," and not forcing every problem to be short.
- **Two layers of control working together**: at the training stage, long2short changes the length distribution, and at the inference stage the budget sets a hard cap. The two are complementary.
- **Three things at deployment**: for mode judgment, evaluate the cost of the two kinds of misjudgment separately; for answers in the two modes, evaluate consistency separately; when the budget runs out, reserve answer tokens or honestly return an unfinished status.

The four cases each cover a level: **DeepSeek V3.1** offers two interfaces from one model at the product level; **Qwen3** forms a complete scheme at the training level from mixed data, template control, and thinking_budget; **Kimi k1.5** actively compresses reasoning chain length with long2short RL at the efficiency level; the **NoThinking study** compares the trade-off between a long chain and parallel short answers under the same budget at the strategy level.

Whether it is a fixed mode switch or a fixed budget cap, a user or router has to judge the task difficulty in advance. What if the judgment is wrong, and a problem that looks very short is actually very hard, or one that looks very long has a very direct answer? The next section's **adaptive thinking** hands this judgment to the model: it decides how long to think dynamically from the input and its current progress.
