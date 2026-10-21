# Supplementary Reading: SWE-RL and Open-Source SOTA

The [SWE-RL paper](https://arxiv.org/abs/2502.18449) (2025.02) is a representative work of open-source SWE-RL. Its core contributions are:

- Training using open-source software-evolution data (SWE-bench style tasks plus SWE-Gym)
- Utilizing GRPO with a simple test-based reward
- Reporting 41.0% on SWE-bench Verified with a 70B model (a strong open-source result at the time)

This section takes a close look at SWE-RL's data, algorithm, and engineering details. Numbers below are the paper's reported figures; verify current standings against the [official SWE-bench leaderboard](https://www.swebench.com/) with matching model, scaffold, and sampling budget.

## 20.1.1 Data Scale and Composition

The public SWE-RL recipe builds on two open-source task sources:

| Data Source             | Scale                         | Purpose               |
| ----------------------- | ----------------------------- | --------------------- |
| SWE-bench (open-source) | 2,294 tasks                   | High-quality baseline |
| SWE-Gym (open-source)   | Thousands of additional tasks | Extended training     |

Together these provide an order of magnitude more training tasks than pure SWE-bench. Some labs additionally mix in internal pull-request histories at much larger scale; those extensions are proprietary, vary by codebase, and their contribution should be ablated separately rather than folded into the open-source number.

### Data Preprocessing

The paper reports data cleaning along three axes (repository filtering, pull-request filtering, and test filtering):

**Step 1: Repository Filtering**

- Exclude: repositories with test coverage < 50% (unreliable for verification)
- Exclude: inactive repositories (last commit > 6 months ago)
- Exclude: repositories with Python < 3.8 (incompatible with latest dependencies)

**Step 2: PR Filtering**

- Exclude: PRs that modify more than 10 files (too complex for early RL training)
- Exclude: PRs that are pure dependency updates (not real "bug fixes")
- Exclude: PRs that delete features (not aligned with "fix" semantics)

**Step 3: Test Filtering**

- Keep: PRs that include new tests (clear verification criteria)
- Exclude: PRs with tests that cannot be run independently
- Exclude: PRs with tests that depend on external services (e.g., databases, API keys)

These filters significantly improved the final data quality. The reported lesson is that filtering for reproducible, well-tested tasks mattered more than raw task count — **data quality > data quantity**.

## 20.1.2 Algorithm: GRPO + Simple Reward

The SWE-RL algorithm choice is deliberately minimal — **GRPO + binary test reward**.

### Why Use GRPO?

The paper compares GRPO against DPO-style and PPO-style baselines in its setup and reports GRPO ahead (culminating in the 41.0% SWE-bench Verified figure for the 70B model). Treat the exact margins as setup-dependent rather than universal, but the qualitative argument travels:

Advantages of GRPO:

- **No Critic Required**: saves GPU memory, suitable for large-scale training
- **Group-wise Normalization**: naturally handles tasks of varying difficulty (high variance in easy tasks, low variance in hard tasks)
- **Simple and Stable**: easier to implement in engineering than PPO

This is consistent with the findings of [DeepSeek-R1](../chapter15_grpo/deepseek-dapo) — **GRPO is the default choice for SWE-RL**.

### Reward Function

The reward function in SWE-RL is extremely simple:

```python
def swe_rl_reward(test_results):
    """Binary reward for test pass"""
    if all(test_results):
        return 1.0  # All tests passed
    else:
        return 0.0  # Any test failed
```

The paper reports that shaped variants tried in its setup underperformed the simple binary reward:

- **Continuous reward (passed/total)**: the model can chase partial passes instead of full fixes.
- **Length penalty**: an over-strong penalty can teach early abort rather than efficient repair.
- **Complex shaping**: each added term opens a new gaming surface.

This matches the R1-Zero lesson discussed in [Chapter 15](../chapter15_grpo/deepseek-dapo): start from a **simple reward + large-scale RL**, and add shaping terms only with ablations. (The exact ablation margins are setup-dependent; treat them as reported observations, not universal constants.)

## 20.1.3 Engineering Details: Context Management

Context management in SWE-RL is an engineering challenge. A complete SWE task may involve:

- The entire repository (10K–1M lines of code)
- Issue description (hundreds of tokens)
- Tool calls (read file, edit, run test)
- Model reasoning (CoT)

These can add up to more than 100K tokens — exceeding the context window of most models. How to handle this?

### Tool Invocation as an Alternative to Reading the Entire Codebase

Instead of making the model read through the entire codebase at once, we let it **actively invoke tools to read only the necessary parts**:

```text
<issue> Django filter(field__in=[]) should short-circuit </issue>

<thought>
I need to find the implementation of this filter.
</thought>

<tool_call> grep -r "def filter" django/db/models/sql/query.py </tool_call>

<tool_result> def filter(self, *args, **kwargs): ... </tool_result>

<thought>
Found it. Let me check the as_sql method.
</thought>

<tool_call> read django/db/models/sql/query.py lines 100-150 </tool_call>

<tool_result> def as_sql(self, compiler, connection):
    ...
</tool_result>

<thought>
I see there is no short-circuit logic here. Let me add it.
</thought>

<tool_call> edit django/db/models/sql/query.py
+ if not self.bloom_metadata and not value:
+     return '', []
</tool_call>
```

This agentic approach allows the model to explore the codebase itself, **reading only the necessary parts** — significantly reducing context usage.

### Long-context Model

Use models with large context windows (128K-class) to fit more of the repository directly. Longer context brings trade-offs:

- Increased training cost (attention is O(n²))
- Slower inference speed
- Position bias (the model is less sensitive to the middle parts of long context)

### RAG (Retrieval-Augmented Generation)

Pre-build an embedding index for the repository, retrieve relevant files based on the issue description, and include only the relevant files in the context.

```python
def build_context(issue, repo):
    # 1. Use embedding to retrieve relevant files
    relevant_files = retrieve(issue, repo, top_k=5)

    # 2. Concatenate into context
    context = ""
    for file in relevant_files:
        context += f"### {file.path}\n{file.content}\n\n"

    return context
```

RAG is the most commonly used method in industry—simple, efficient, and compatible with existing models.

A common production recipe combines retrieval for the base context with tool calls for follow-up exploration.

## 20.1.4 Techniques for Training Stability

SWE-RL training is more challenging than mathematical RL — because:

- Trajectory length is long (16–100+ steps)
- Reward is extremely sparse (only the final test pass yields a reward)
- Most trajectories are failures (reward = 0)

The paper reports several techniques for training stability:

### Success Rate Filtering

In RL training, **only keep prompts that have at least one successful rollout**. If a prompt's N rollouts all fail (reward is 0 for all), its intra-group variance is also 0, and it cannot provide training signals.

```python
def filter_prompts(prompts, model, num_rollouts=8):
    useful_prompts = []
    for prompt in prompts:
        rollouts = [model.generate(prompt) for _ in range(num_rollouts)]
        rewards = [compute_reward(r) for r in rollouts]
        if max(rewards) > 0:  # At least one success
            useful_prompts.append(prompt)
    return useful_prompts
```

This aligns with the idea of [DAPO's Dynamic Sampling](../chapter15_grpo/deepseek-dapo) — discarding prompts whose rollouts carry no learning signal (all fail, or all succeed trivially).

### Curriculum Learning

Sort prompts by difficulty, first training on simple ones (small PRs, single-file, clear issues), then moving to complex ones (multi-file, ambiguous issues).

```python
def curriculum_order(prompts):
    # Sort by the number of files changed
    prompts.sort(key=lambda p: p.num_files_changed)
    return prompts
```

### KL Constraint

During the later stages of SWE-RL training, it is common for the model to overfit to passing tests at the expense of code style. A standard guard is a KL constraint against the pre-RL checkpoint:

$$\mathcal{L} = \mathcal{L}_{\text{RL}} + \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})$$

Here, $\pi_{\text{ref}}$ is the model before RL (the version after SFT), and $\beta$ is the constraint strength. The KL weight trades reward optimization against staying close to readable, conventional code: math-reasoning setups sometimes run with little or no KL, while code setups often keep a small KL term because style and interface conventions matter to human reviewers. Tune $\beta$ by watching patch readability alongside the resolve rate.

## 20.1.5 SWE-bench Verified 41.0%

The SWE-RL paper reports 41.0% on SWE-bench Verified for its 70B model — a strong open-source result at the time of publication. SWE-bench Verified is the human-checked 500-problem subset, so this number already controls for some of the noise in the full set. Even so, it is comparable only to results with the same scaffold, tool permissions, step budget, and sampling budget (Pass@1 versus scaled inference). For current standings, check the [official SWE-bench leaderboard](https://www.swebench.com/) and record the model version, the scaffold, the evaluation set, and the run settings together.

## 20.1.6 Limitations of SWE-RL

However, SWE-RL as a public recipe also has several limitations:

### Primarily Python

The public training data and evaluation are concentrated in Python. Extending the same loop to other languages means rebuilding the sandbox, the test harness, and the defect distribution for each ecosystem.

### Relies on Test Suites

Repositories without tests cannot be trained. This is a significant issue in industrial practice — many companies' code lacks complete unit tests.

### Unstable Training for Long Horizons

Training on trajectories of tens of steps is unstable — credit assignment in RL is difficult, and performance degrades as trajectories lengthen. Success-rate filtering and curricula (Sections 20.1.4) mitigate this but do not remove it.

### Data Diversity

The open-source tasks come from GitHub pull requests — the distribution is biased toward the open-source ecosystem. The characteristics of industrial code (such as enterprise internal systems in other languages) are not covered.

## Summary

SWE-RL is a representative work of open-source SWE-RL. Its core contributions are:

- **Data**: open-source software-evolution tasks (SWE-bench style tasks plus SWE-Gym), with quality filtering doing most of the work
- **Algorithm**: GRPO + simple binary reward, sharing the same origin as R1-Zero
- **Engineering**: context management, success-rate filtering, curricula, and KL regularization
- **Results**: a reported 41.0% on SWE-bench Verified for the 70B model

SWE-RL demonstrates the feasibility of RLVR in the domain of software engineering. Its limitations — concentration in Python, instability for long horizons, and dependency on tests — lead to [20.2 Code World Model and DeepSWE](./world-model-and-deep-swe): how to model software execution and scale long-horizon training with reliable environments.
