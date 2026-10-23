---
search: false
---

# 19.5 Search-Augmented RL

General tool-use RL — why RL is needed for tool calls, the main training frameworks, reward design, web and code agents, and the engineering required to run training at scale — lives in [Section 19.4](./tool-use-and-trajectory). This section focuses on the single most demanding tool setting: **search**. Search results are open-ended and unstructured, the model's parametric knowledge may be stale, and deciding when to search, what to query, and how to use the results is a policy problem that prompting alone cannot solve.

## Background in Brief: What Section 19.4 Already Covers

The topics below duplicate Section 19.4, so each is summarized here in a sentence or two with a link to the full treatment. This keeps the page focused on search.

- **Why RL is needed for tool use.** SFT can teach the JSON syntax of a tool call, but when to call, which tool to call, and how to combine tools are strategic decisions the model must discover through trial and error; see [Why Is RL Essential for Tool Use?](./tool-use-and-trajectory#why-is-rl-essential-for-tool-use).
- **Core methods.** ReTool lets the model invoke tools freely mid-reasoning, VERL-TOOL unifies tools from many domains into one RL action space, MCP-RL adds a standard protocol layer so the same loop can discover and call tools across servers, and ToolRL treats tool calls as first-class RL actions; see [Core Methods](./tool-use-and-trajectory#core-methods).
- **Reward design.** Tool-use rewards combine a verifiable task outcome with efficiency and format penalties, with the exact mix set by the scenario; see [Reward Design](./tool-use-and-trajectory#reward-design-the-scenario-determines-the-reward).
- **Web agents.** Browser agents operate on screenshots plus DOM state with click-and-type actions, and step-level rewards such as Web-Shepherd's densify the sparse task-completion signal; see [Web Agent RL](./tool-use-and-trajectory#web-agent-rl-teaching-a-model-to-use-the-web).
- **Code agents.** Code agents iterate write-execute-fix against unit-test pass rates, and results such as rStar2-Agent show a 14B tool-using model surpassing a 671B model; see [Code Agent RL](./tool-use-and-trajectory#code-agent-rl-write-code-debug-and-iterate).
- **Training flow and RLVR.** The standard recipe is SFT for format, then RL for strategy, then iterative data improvement — agentic RL as multi-turn RLVR with verifiable rewards; see [Training Flow](./tool-use-and-trajectory#training-flow-for-tool-use-strategy) and [Connection to RLVR](./tool-use-and-trajectory#connection-to-rlvr).
- **Engineering practice.** Sandboxing, nondeterministic tool results, and GPU-idle latency turn agentic training into a distributed-systems problem addressed with asynchronous rollouts; see [Agentic RL Engineering Practice](./tool-use-and-trajectory#agentic-rl-engineering-practice).
- **Reward systems and evaluation.** Production reward design moves from simple success checks to multidimensional rubrics, evolving standards, and LLM judges; see [Agent Reward Design and Evaluation Systems](./tool-use-and-trajectory#agent-reward-design-and-evaluation-systems).

## Search-Tool RL: SearchR1 and Search-Augmented Reasoning

The web-agent and code-agent settings covered in [Section 19.4](./tool-use-and-trajectory) each have their own focus, but one tool scenario is especially important and especially challenging: **search engines**. Search differs from calculators and databases. Search results are open-ended and unstructured, and a good search strategy depends heavily on context. If the question is "What is the difference between GRPO and PPO?", the model may not need search. But if the question is "Who won the 2025 Nobel Prize in Physics?", the model must search, because its internal knowledge may be out of date.

In 2025, SearchR1[^searchr1] (Jin et al.) pioneered the use of RL for search-tool training, allowing models to **learn autonomously when to search, what to search for, and how to use search results**. Later work such as ReSearch[^research] and ToRL[^torl] advanced this direction from different angles.

### Why Prompting Is Not Enough

Before SearchR1, the mainstream approach was to use prompting to teach the model that "you may call a search engine during reasoning". ReAct[^react], Self-RAG[^selfrag], and related methods follow this route. But prompting has three fundamental limits.

**The timing of search cannot be exhaustively specified.** You can write in the prompt, "search when knowledge is uncertain", but what counts as "uncertain"? The model may be very confident about outdated information, not knowing that it does not know; or it may over-search obvious common knowledge.

**Search-query strategy cannot be imitated mechanically.** For a task such as "compare the latest performance data for three quantum-computing platforms", the search strategy must adapt dynamically to previous results. The first query, "quantum computing benchmark 2025", may be too broad, so the second query becomes "IBM quantum advantage vs Google Sycamore 2025". This kind of **adaptive query generation** is a policy-learning problem, not just a language-modeling problem.

**Long-horizon optimization over multiple searches.** A complex task may require several rounds of search. Stopping too early means insufficient information; stopping too late wastes resources. This trade-off is exactly where RL is useful.

### SearchR1's MDP Formulation

SearchR1 formulates search-augmented reasoning as a special MDP:

- **State $s_t$**: the current reasoning context (generated text + previous search results)
- **Action $a_t$**: two types: (1) continue generating tokens, or (2) generate a search query and trigger search using `<search>...</search>` tags
- **Transition**: a search action triggers the search engine, and the result is appended to the context
- **Reward**: final-answer correctness (RLVR) + search-efficiency penalty

```
Reasoning + search interaction process:

User: "Who received the 2025 Turing Award?"

Model reasoning: "This question concerns recent 2025 information, so I need to search."
Model action: <search>2025 Turing Award winner</search>
Search result: "The 2025 ACM Turing Award was given to..."
Model reasoning: "Now I have enough information."
Final answer: [complete answer]

Reward: answer correctness - λ × number of searches
```

Training uses GRPO group sampling plus within-group comparison. The key design is that the text returned by search is **masked out** from the loss. The model should not be reinforced simply because the search engine returned good content. Search behavior **emerges spontaneously** during RL training: even if SFT never taught search, the model gradually learns to trigger search at the right time.

### Key Findings from SearchR1

- **Search behavior emerges from RL.** RL can do more than optimize known strategies; it can discover new ones.
- **Scaling law for search frequency.** As training steps increase, the model's search frequency rises on questions that require search and falls on questions that do not. The model learns to distinguish the two settings.
- **Generalization to unseen search scenarios.** Search strategies trained on math problems can generalize to history and science questions.

### Technical Lineage After SearchR1

| Work                       | Core innovation                                                                                         | Citations / venue |
| -------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------- |
| **SearchR1 **[^searchr1]   | RL trains the model to search autonomously, GRPO + RLVR                                                 | 819               |
| **ReSearch **[^research]   | Deep integration of reasoning and search; each reasoning step can include reflection on search strategy | -                 |
| **ToRL **[^torl]           | Extends to computational tools (code executors), finding scaling laws for tool use                      | 131               |
| **ReTool **[^retool]       | Distinguishes reasoning tasks from computational tasks; RL lets the model choose tools strategically    | 247               |
| **ZeroTIR **[^zeroscaling] | Models spontaneously learn code execution without supervised examples; power-law scaling law            | NeurIPS 2025      |

The 32B model trained by ToRL[^torl] outperformed a 70B model without tools on mathematical reasoning, showing that "small model + tools > large model with pure reasoning". ReTool[^retool] further teaches models strategic tool selection: not every problem needs a tool; the model should decide dynamically based on the problem's characteristics.

```python
def search_reward(answer, ground_truth, num_searches, max_searches=5):
    """Reward function for search RL."""
    correctness = 1.0 if verify_answer(answer, ground_truth) else 0.0
    efficiency_penalty = -0.05 * (min(num_searches, max_searches) / max_searches)  # Bounded search-cost penalty.
    return correctness + efficiency_penalty
```

## Section Summary

Search-augmented RL trains the model to decide for itself when to search, what to query, and how to use the results, with SearchR1 showing that this behavior can emerge from RLVR-style rewards plus an efficiency penalty. For the broader picture — trajectory synthesis, general tool-use methods, and the engineering stack — see the [Chapter Summary in Section 19.4](./tool-use-and-trajectory#chapter-summary).

In the next section, we turn to the problems code agents hit in real training — instability, excessive trajectory length, and reward hacking: [Code Interpreter RL](./industrial-practice).

## References

[^searchr1]: Jin B, et al. "[Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)." COLM 2025. First work to formulate search-tool use as an RL problem. [GitHub](https://github.com/PeterGriffinJin/Search-R1)

[^research]: Chen M, et al. "[ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470)." arXiv:2503.19470, 2025. A framework for deep integration of reasoning and search.

[^torl]: Li X, et al. "[ToRL: Scaling Tool-Integrated RL](https://arxiv.org/abs/2503.23383)." arXiv:2503.23383, 2025. Extends tool-use RL to computational tools and discovers scaling laws for tool use.

[^retool]: Feng J, et al. "[ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)." arXiv:2504.11536, 2025. Uses RL to optimize tool-use strategies during reasoning.

[^zeroscaling]: Mai X, Xu H, Wang X, et al. "[Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving](https://arxiv.org/abs/2505.07773)." NeurIPS 2025. ZeroTIR: spontaneous code execution without supervised examples, revealing a power-law relationship between training steps and performance.

[^react]: Yao S, et al. "[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)." ICLR 2023. A classic prompting method for combining reasoning and action.

[^selfrag]: Asai A, et al. "[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)." ICLR 2024. Retrieval-augmented generation through self-reflection.
