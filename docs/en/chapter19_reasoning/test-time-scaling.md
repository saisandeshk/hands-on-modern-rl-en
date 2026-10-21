# 16.3 Test-Time Scaling

In the previous two sections we saw why reasoning models expand their thinking before giving an answer (16.1), and how R1-Zero elicited reasoning ability using only an outcome reward and GRPO (16.2). The trained model now thinks on its own initiative, but deploying it in a service immediately raises a practical problem: the same model faces two kinds of request, "Translate hello into Chinese" and an AIME competition problem. If both are made to generate ten thousand tokens, the former wastes compute for nothing, while the thinking time for the latter may still not be enough.

This is the problem that **Test-Time Compute Scaling** addresses: the model parameters have already been trained and are fixed, so can the system still adapt to tasks of different difficulty by adjusting the amount of compute at inference time? With parameters fixed, there are three basic ways to add inference compute: **generate more candidates in parallel**, **revise one answer repeatedly**, and **run tree search across several intermediate steps**. This section first answers "why might extra compute work," then looks at how each of the three uses works, and finally discusses when the gains saturate.

<img src="../../chapter19_reasoning/images/test-time-compute-options.svg" alt="Three ways to use an extra inference budget after the model parameters are fixed">

## What Test-Time Compute Scaling Can Solve

Start with the most basic scenario: ask the model to solve the quadratic equation $x^2+5x+6=0$. The first time it computes $x=-2$ and $x=-3$ (correct), but the second time it makes a sign error and writes $x=2$ and $x=3$. The model has the method: it knows the quadratic formula and also knows how to factor, but in this generation some intermediate step went wrong. Asking it to do it again or to check once more often yields the correct answer.

But not every failure can be remedied this way. Two failures of different nature must be told apart:

- **Knowledge failure**
  - Cause: the model never mastered the knowledge or method needed to solve the problem
  - Remedy: can only be solved by further training (more data, a larger model)
- **Sampling failure**
  - Cause: the model has the method, but in this generation it chose the wrong route, miscalculated a number, or did not check
  - Remedy: can be mended by adding inference compute (more sampling, more revision, more search)

Training compute and inference compute occur at different stages and solve different problems:

- **Training compute**: updates the model parameters so the model learns new knowledge and strategies, raising the ceiling of the model's ability.
- **Inference compute**: leaves the parameters completely untouched and only adds sampling, checking, revision, or search on the current problem, bringing the model closer to its own ceiling.

Early language models "generated one short answer at a time," so inference compute was roughly a fixed cost, with about the same number of tokens spent on any problem. Reasoning models changed this: on the same problem they can generate longer thinking trajectories or more candidate answers, and inference compute turned from a fixed overhead into a budget that can be allocated flexibly. The question then becomes: once the model's ability is fixed, how should the inference budget be allocated to raise the success rate on the current task most effectively?

### Snell et al.'s Study of Budget Allocation

In 2024, Snell et al. answered this question systematically in the paper [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314). They ran a clean set of controlled experiments:

**Setup**: fix one base model (Llama-3-8B-Instruct), and compare two ways of improving on math problems of different difficulty:

- **Path A**: keep the model but increase inference compute: have the model generate $N$ candidate solutions and then use a verifier to pick the best (best-of-N).
- **Path B**: add no inference compute, but switch to a model with more parameters and stronger ability.

The experiments gave three conclusions that still guide the design of reasoning systems:

1. **Allocating inference compute properly can let a small model beat a larger model whose compute is "allocated badly."** Under the experimental conditions given in the paper, a small model with enough inference budget and a good verifier can outperform a larger baseline whose inference budget is set rigidly.
2. **If the base model cannot produce correct candidates at all, the return on adding a limited budget falls quickly.** If the single-attempt accuracy $p$ is itself near 0, no amount of extra sampling will cover the correct answer.
3. **There is no universal allocation strategy.** The budget and allocation suited to easy, medium, and hard problems all differ, and the more reliable the verifier, the more it is worth spending the budget on parallel sampling.

This set of experiments showed quantitatively for the first time that training compute and inference compute are complementary: training determines the ceiling of the model's ability, and inference compute determines how close the model gets to that ceiling.

## Three Ways to Use Inference Compute

Once an inference budget is fixed, there are three basic ways to spend it. We continue to use math problem solving to understand them.

### Parallel Sampling: Best-of-N

The most direct idea is this: one attempt may go wrong, so generate $N$ answers independently and use a verifier to pick out the correct one. This is the idea of **best-of-N**.

Suppose the probability that the model independently generates a correct answer in a single attempt is $p$. The probability that all $N$ candidates are wrong is $(1-p)^N$, so the probability that at least one correct candidate appears is

$$
P(\text{at least one correct})=1-(1-p)^N.
$$

This quantity is usually called the **coverage**, and it gives the theoretical upper bound of parallel sampling. Take $p=0.2$ (single-attempt accuracy of 20%). How the coverage changes with $N$:

- **$1$**
  - Probability of at least one correct: $20.0\%$
  - Marginal gain from one more candidate: $+16.0\%$
- **$2$**
  - Probability of at least one correct: $36.0\%$
  - Marginal gain from one more candidate: $+12.8\%$
- **$4$**
  - Probability of at least one correct: $59.0\%$
  - Marginal gain from one more candidate: $+8.2\%$
- **$8$**
  - Probability of at least one correct: $83.2\%$
  - Marginal gain from one more candidate: $+3.4\%$
- **$16$**
  - Probability of at least one correct: $97.2\%$
  - Marginal gain from one more candidate: $+0.6\%$

The marginal gain in the last column has a closed form. The $(N+1)$-th candidate brings new benefit only when all the first $N$ are wrong, so its contribution is

$$
P(N+1)-P(N) = p\,(1-p)^N.
$$

Here $p$ is the probability that a single candidate is correct, and $(1-p)^N$ is the probability that the first $N$ are all wrong. Plugging in $p=0.2$: the marginal gain of the 2nd candidate is $0.2\times0.8=16\%$, consistent with the table, and by the 17th candidate only $0.2\times0.8^{16}\approx 0.6\%$ remains. Each additional candidate multiplies the marginal gain by $(1-p)$, a geometric decay, while the cost of each candidate is fixed. This is the concrete meaning of "diminishing marginal returns," and it also gives a quantitative basis for stopping sampling: when $p(1-p)^N$ falls below the cost of one candidate, further sampling no longer pays.

The pseudocode for parallel sampling is very simple:

```python
# Illustration of parallel sampling
candidates = [model.generate(prompt) for _ in range(N)]  # generate N independently
scores = [verifier.score(prompt, c) for c in candidates] # score each one
best = candidates[argmax(scores)]                        # pick the highest score
```

There is a key precondition here: a **reliable verifier** is required. Coverage only guarantees that "a correct answer appears among the candidates," and whether it can be picked out depends on the verifier. If the verifier treats wrong answers as right, no value of $N$ helps. This is exactly the problem that the process reward model (PRM) of Chapter 17 sets out to solve.

::: details An alternative when there is no verifier: majority voting
When there is no good verifier, **majority voting** (self-consistency) can serve as a substitute: generate several answers, normalize the final answers (unify them as decimals or fractions in lowest terms), count which answer appears most often, and choose it.

For example, the model gives 5 answers to the same problem:

```text
x = -3,  x = -2,  x = -3,  x = 3,  x = -3
```

After normalization and counting: $-3$ appears 3 times, $-2$ once, and $3$ once, so the majority vote goes to $-3$.

Majority voting carries an implicit assumption: when different reasoning paths make errors independently, the correct answer is more likely to be reached by several paths together. It works for problems with several solution methods that all lead to the same answer, but it is powerless against systematic errors: when all paths make the same mistake at the same place, the majority vote still selects the wrong answer.
:::

### Sequential Revision: Correcting Errors Repeatedly

Parallel sampling restarts from the problem every time and throws away the parts that were already right. Suppose the model's first four algebraic manipulations are right and it errs only in the last step when combining like terms. Parallel sampling throws away the four correct steps and regenerates everything.

Sequential revision takes a different approach: first generate an initial solution, have the model (or another critic model) find the problems in it, then produce the next version targeting those problems, and iterate for $K$ rounds:

```python
# Illustration of sequential revision
solution = model.generate(prompt)           # generate the first version
for _ in range(K):
    feedback = model.critique(prompt, solution)  # critique: find the problems
    solution = model.revise(prompt, solution, feedback)  # revise: fix the problems
```

Sequential revision **reuses the work from the previous version that was already right** and modifies only the problematic parts. It suits tasks where local errors are easy to locate and a local edit is cheaper than regenerating from scratch: when code has one bug, only that line needs fixing, and the whole file need not be rewritten.

The two methods also have different cost structures. Let the generation cost of one complete solution be $L$ tokens. Parallel sampling to generate $N$ candidates costs $N\cdot L$ in total, but the $N$ can be spread across different GPUs and run at the same time. The cost of sequential revision is $L$ plus $K$ rounds of critique and revision, each round targeting only a local problem and usually far shorter than $L$, but the $K$ rounds can only wait in sequence. The former trades total compute for parallelism, and the latter trades serial latency for compute reuse.

Its drawbacks are equally clear:

1. **Latency accumulates**: each round depends on the result of the previous one, so $K$ rounds of revision mean the serial waiting time becomes $K$ times as long.
2. **Path dependence**: if the first step has the wrong direction, later revisions patch a wrong foundation and may drift further the more they fix.
3. **The critique may be wrong**: if the critic model misjudges and "fixes" something that was right, all later revisions go off course.

The trade-off between the two methods at different error locations:

- **Sign error in the last step**
  - Parallel sampling: wasteful, regenerates even the correct steps
  - Sequential revision: efficient, changes only the last step
- **Wrong method chosen in the first step**
  - Parallel sampling: efficient, switching to another route may get it right
  - Sequential revision: inefficient, patches a wrong prefix and drifts further

### Tree Search: Combining Breadth and Depth

Parallel sampling and sequential revision each have a blind spot: the former wastes prefixes, and the latter is stuck on a single path. **Tree search** combines the two: it expands the reasoning process into a tree where each node is an intermediate step, tries several subsequent branches from the same prefix, scores the nodes with a verifier, and concentrates more budget on the high-scoring paths.

The specific algorithms (Beam Search, Tree of Thoughts, MCTS, and others) are developed in Section 17.5. For now, remember the core problem it solves: **how to reuse intermediate results without being stuck on a single path**.

## Deep Think Turns Parallel Thinking into a Product

The principles of parallel sampling and sequential revision are both straightforward, but turning them into a product raises a number of engineering problems: how do multiple paths share GPUs? When should sampling stop? Who integrates the conclusions of the multiple paths at the end?

Google's [Gemini Deep Think](https://blog.google/products-and-platforms/products/gemini/gemini-2-5-deep-think/), released in 2025, is a typical case of productizing parallel reasoning.

### How Deep Think Works

According to Google's public description, the Deep Think system **considers several hypotheses at the same time**, allowing candidate paths to be revised or combined before the final answer. Public material does not disclose the full algorithmic details, so we cannot tell whether the internals use best-of-N, tree search, or some cross-path mechanism, but three core system tasks can be abstracted from the interface behavior:

1. **Generate several independent reasoning paths at the same time.**
2. **Compare or combine the information in different paths** (not simply choosing one, but possibly including cross-verification or information integration).
3. **Decide within the budget when to stop and produce the final answer.**

If these $N$ paths are assigned to independent compute resources, they can be generated in parallel. But note a common misconception:

> Parallelism shortens the **wall-clock waiting time**, and does not eliminate the **total compute**. Not a bit of the needed compute is saved, it is merely spread across several GPUs and executed at the same time.

The more paths there are, the more engineering problems the system must handle: GPU scheduling, candidate deduplication, evaluator throughput, and stopping-condition judgment.

### Reading Product Scores Correctly

Google reported that a dedicated Deep Think version reached the gold-medal range at the 2025 IMO (International Mathematical Olympiad), and later versions also published results on benchmarks such as HLE, ARC-AGI-2, and Codeforces. These results show that parallel thinking is an effective inference-budget scheme, but note:

> These scores alone cannot prove that the gains come from "how many paths" or "which aggregation algorithm."

The evaluation may have used code execution, external search, and different budget settings at the same time. When comparing two systems, these conditions must be recorded together, or the score comparison is meaningless.

### When to Stop Sampling

The simplest implementation generates a fixed $N$ paths and runs all $N$ regardless of how easy the problem is. Even easy problems then waste budget: when 5 paths have already reached the same verifiable answer (solved $x=2$ and substituted back to confirm it is correct), generating a 6th, 7th, and 8th path is pointless.

A dynamic stopping policy checks three things after each batch of candidates:

1. **Has a verifiable correct answer appeared?** For a math problem, the answer passes substitution verification; for code, all tests pass.
2. **Do the high-scoring candidates already agree?** When the top three candidate answers are all the same, the probability that further sampling produces a new answer is very low.
3. **Can more candidates still bring new solutions?** When the newly generated candidates only repeat old ideas, sampling can stop.

When the answer has passed an external check and the candidates have converged, stopping early is fine. If the candidates still conflict with each other and the verifier cannot tell right from wrong, the value of further sampling is usually low as well, and what is needed more is tools, external search, or human rules to supplement feedback.

The stopping logic must be designed together with the **task verifier**, the **maximum token limit**, the **wall-clock timeout**, and the **failure fallback strategy**.

## When to Stop Adding Compute

The number of candidates, the number of revisions, and the number of search nodes can all keep increasing, but accuracy will not keep rising at the same rate forever. When deploying a system, what must be compared is how much further the success rate can rise with the next portion of compute, and how much latency and cost it adds.

Extra compute increases generated tokens, verifier calls, and hardware occupancy, but **latency does not necessarily grow in proportion to total compute**: parallel sampling can trade more GPUs for a shorter wall-clock wait, though the total token cost still grows, while sequential revision adds serial latency directly with the number of rounds, which adding GPUs cannot fix.

At deployment, five metrics should be monitored together:

- Task success rate
- Wall-clock latency (the user's waiting time)
- Total token consumption
- Number of verifier calls
- Cost per request

### Marginal Returns Will Necessarily Diminish

Whichever allocation method is adopted, the marginal return of extra compute eventually falls:

- **Easy problems**: the answer is already right after a little compute, and further generation only repeats an existing correct path without raising the success rate.
- **Medium problems**: more candidates or one round of revision may correct an accidental error, and the return first rises and then falls.
- **Hard problems beyond the base model's ability**: the model cannot produce a correct path, extra sampling only produces more similar errors, and the return is very low from the start.

The coverage table above has already given a quantitative example for parallel sampling: at $p=0.2$, the marginal gain of the 16th candidate is down to only 0.6 percentage points. Sequential revision and tree search follow the same pattern: the more rounds of revision, the fewer fixable errors remain, and the deeper the search, the lower the expected value of the unexplored branches.

Inference-time compute can therefore never substitute for training compute without limit. The relationship between the two is:

> - **Training compute** determines the ceiling of ability: the most the model can do.
> - **Inference compute** determines how close we get to the ceiling: how much of its ability the model actually delivers this time.

When the base model has not mastered the knowledge or operations needed to solve a problem ($p \approx 0$), no amount of sampling and revision can easily produce a correct path. Inference compute must build on abilities the base model already has.

::: details How task type affects budget settings
Budget settings cannot rest on a surface classification such as "is this a math problem or a translation problem." Three factors must also be considered:

1. **Is the verifier reliable?** With a reliable verifier (code can be run and tests passed), best-of-N is very effective; without one, majority voting is the fallback.
2. **What is the single-attempt success rate?** When $p$ is too low, raising $p$ first (by training or changing the model) is more cost-effective than increasing $N$.
3. **What is the cost of task failure?** In settings such as medicine, law, and production code, failure is costly and more budget can be given; for casual chat and simple question answering, quickly giving an acceptable answer matters more.
:::

## Section Summary

Test-time compute scaling provides three adjustable inference resources: the number of parallel candidates, the number of revisions of a single reasoning chain, and the size of the search tree. All of them give the model more chances to perform when "the model already knows how, but did not play it well this time."

- **Keep the two kinds of failure apart**: failures from missing knowledge can only be solved by training; sampling failures such as choosing the wrong route or miscalculating a number can be remedied by inference compute.
- **The coverage formula** $1-(1-p)^N$ gives the theoretical upper bound of parallel sampling, and the actual gain also depends on whether the verifier can pick out the correct answer; the marginal gain diminishes with $N$.
- **Three allocations of compute**: parallel sampling buys breadth and suits early steps where the method may be chosen wrongly; sequential revision buys depth and suits settings where local errors are easy to locate; tree search reuses intermediate results and combines the two.
- **Parallelism does not save total compute**: a system such as Deep Think trades more GPUs for a shorter wait, and the total compute does not disappear.
- **Stopping conditions are designed**: at deployment, monitor success rate, latency, tokens, and cost together, and stop where the marginal gain falls below the marginal cost.

The harder the task, and the higher the model's single-attempt success rate while still below 100%, the more likely extra inference resources are to bring gains. When the base model's ability is insufficient or the task is already very simple, the gains fall off quickly.

The next section applies this pattern to real deployment: how one model switches between "answer directly" and "think deeply," and how a thinking budget is used to limit latency and cost. This is the problem Hybrid Thinking addresses.
