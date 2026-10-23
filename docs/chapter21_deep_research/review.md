---
title: 21.3 Summary and Exercises
---

# 21.3 Summary and Exercises

## What you should now be able to do

1. Cast a browser research task as a partially observable Markov decision process: name the state, observation, action, transition, and reward, and explain why the reward is sparse and terminal.
2. Choose an action-space abstraction (search API, Playwright browser, set-of-mark) for a given research task and justify the trade-off between coverage and training cost.
3. Build the five modules of a browser RL harness — environment wrapper, action parser, reward verifier, progress tracker, parallel rollout engine — and explain what breaks when each one is missing.
4. Train a search agent with outcome rewards (Search-R1 style) using GRPO with retrieved-token masking, and explain why environment-returned tokens must not receive gradients.
5. Decompose evaluation into answer, citation, process, and cost layers, and compute exact match, coverage, citation support, redundancy, and budget-success scores from a run log.
6. Diagnose the standard failure modes — hallucinated citations, shortcut-taking, reward hacking, cost contamination — and pick the matching fix (citation-aware rewards, evolving rubrics, held-out evaluation, fixed-budget comparison).
7. Design a reproducible evaluation protocol: frozen task versions, fixed environment inventory and inference budget, complete artifacts, and layered scoring before any aggregation.
8. Reason about the training system behind the algorithm: synchronous versus asynchronous rollout–update loops, policy staleness, fault tolerance with exactly-once accounting, and throughput/compute bookkeeping.

## Key equations

- Browser MDP and sparse terminal reward:
  $$\mathcal{M}_{\text{browser}}=(\mathcal{S},\mathcal{A},P,R,\gamma,T),\qquad r_T=\mathbb{1}[\text{answer correct}],\; r_{t<T}=0.$$
  Reading: the agent plans over 20–100 tool steps but learns only from the final verdict.
- Trajectory record:
  $$\tau=(o_0,a_0,o_1,a_1,\ldots,o_T,a_T,R).$$
  Reading: keep observations, actions, and the score in order, or no failure can be attributed.
- Exact match for short answers:
  $$\operatorname{EM}(\hat y,y)=\mathbb{1}[\operatorname{norm}(\hat y)=\operatorname{norm}(y)].$$
  Reading: correct after normalization; aliases need an accepted-answer set.
- Fact coverage for open reports:
  $$S_{\text{coverage}}=\frac{\text{correctly covered facts}}{\text{required facts}}.$$
  Reading: split the report into checkable facts and count them.
- Citation support:
  $$S_{\text{citation}}=\frac{1}{n}\sum_{j=1}^n c_j,\qquad c_j=1\text{ if citation }j\text{ supports its claim}.$$
  Reading: reachability is not support — the passage must back the specific claim.
- Process redundancy:
  $$S_{\text{redundancy}}=\frac{\text{repeated queries}+\text{repeated visits}}{\text{all tool calls}}.$$
  Reading: a diagnostic for wasted effort, never a replacement for answer and citation scores.
- Budget success at fixed budget $B$:
  $$S_{\text{budget}}(B)=\frac{\text{tasks completed within }B}{\text{all tasks}}.$$
  Reading: compare systems under identical resources, not accuracy per token.
- GRPO group-relative advantage for rewards $r_1,\dots,r_G$:
  $$A_i=\frac{r_i-\operatorname{mean}(r)}{\operatorname{std}(r)}.$$
  Reading: the group's own spread decides credit — no separate critic needed.
- Bradley–Terry preference loss for the search reward model:
  $$\mathcal{L}=-\log\sigma\big(r_\phi(\text{chosen})-r_\phi(\text{rejected})\big).$$
  Reading: push the preferred search result's score above the worse one's.

## Common pitfalls

1. **Training on the environment's words.** Forgetting to mask `<information>` tokens lets retrieval quality leak into the policy gradient — the model gets punished for a bad search engine. Mask every environment-returned token out of the loss.
2. **Collapsing four layers into one number.** A single aggregate hides "correct answer, broken citation" and "good evidence, over budget." Always report answer, citation, process, and cost separately before any weighted ranking.
3. **Letting the verifier be gamed.** Static rubrics invite citation stuffing, keyword stuffing, and length inflation. Track an independent held-out set and tighten criteria (evolving rubrics) when training scores rise but held-out quality stalls.
4. **Comparing systems at different budgets.** A 200-step swarm will beat a 10-step agent on accuracy and lose on cost. Freeze tool steps, tokens, and wall time first; then compare.
5. **Counting environment failures as policy failures.** A search-API 500 error scored as reward zero teaches the policy to avoid a blameless situation. Label tool errors separately, retry idempotently, and exclude corrupt trajectories from the update.

## Exercises

### Exercise 1 — Name the layer

For each symptom below, state which evaluation layer (answer, citation, process, cost) detects it and which score from this chapter quantifies it:

(a) The final university name is right, but the cited page belongs to a different person with the same name.
(b) The agent issues the same query six times before opening any page.
(c) Two systems both reach 80% exact match, but one uses five times the tokens.
(d) The report covers three of the five facts the rubric requires.

::: details Solution

(a) Citation layer: $S_{\text{citation}}$ drops because the passage does not support the claim about the right person, even though EM is 1.
(b) Process layer: $S_{\text{redundancy}}$ is high (repeated queries over all tool calls). The fix is a stopping/query-diversity issue, not an answer issue.
(c) Cost layer: $S_{\text{budget}}(B)$ separates them — set one fixed budget $B$ and count completions within it.
(d) Answer layer for open reports: $S_{\text{coverage}} = 3/5 = 0.6$.

The general rule: answer first, then whether evidence supports it, then how wasteful the path was, then what it cost.

:::

### Exercise 2 — Compute redundancy and budget success by hand

A run log contains 8 tool calls: 5 searches with queries `[q1, q2, q1, q3, q2]` and 3 page visits `[u1, u2, u1]`. Repeated means "already seen earlier in this trajectory."

(a) Compute $S_{\text{redundancy}}$.
(b) In a 20-task evaluation, 14 tasks finish within the step/token budget. Compute $S_{\text{budget}}(B)$.
(c) The agent then adds cross-checking visits so the log grows to 12 tool calls with the same 3 repeats. Recompute redundancy and explain why chasing this number down can be harmful.

::: details Solution

(a) Repeated queries: `q1` appears twice (1 repeat), `q2` appears twice (1 repeat), so 2 repeated queries. Repeated visits: `u1` appears twice, so 1 repeated visit. Total tool calls = 8.
$$S_{\text{redundancy}}=\frac{2+1}{8}=\frac{3}{8}=0.375.$$
(b) $S_{\text{budget}}(B)=14/20=0.7$.
(c) New redundancy $= 3/12 = 0.25$. The number fell, but only because the denominator grew — the agent did not remove any waste. More importantly, legitimate cross-checking (revisiting a source to verify a conflicting fact) also counts as "repeated," so minimizing redundancy directly would penalize verification. Use it to explain behavior, not as a reward.

:::

### Exercise 3 — GRPO advantages by hand

One query yields a group of $G=4$ trajectories with rewards $[0, 0, 1, 1]$.

(a) Compute the group mean and (population) standard deviation.
(b) Compute each trajectory's advantage $A_i=(r_i-\text{mean})/\text{std}$.
(c) Verify the advantages sum to zero, and explain why trajectories 1–2 get a negative update even though "the answer was just wrong, not harmful."

::: details Solution

(a) Mean $= (0+0+1+1)/4 = 0.5$. Variance $= ((0.5)^2+(0.5)^2+(0.5)^2+(0.5)^2)/4 = 1/4 = 0.25$, so std $= 0.5$.
(b) $A = ([0,0,1,1]-0.5)/0.5 = [-1,-1,+1,+1]$.
(c) Sum $= -1-1+1+1 = 0$ (mean-centering guarantees this up to rounding). Trajectories 1–2 are pushed down not because they were catastrophic but because they were below the group's own average — GRPO credit is relative, so a mediocre majority still generates a learning signal toward the better minority.

:::

### Exercise 4 — Why $\gamma = 1$ for browser agents

Consider a 30-step trajectory with reward 1 only at the final step and 0 elsewhere.

(a) With $\gamma = 1$, what is the return $G_0$?
(b) With $\gamma = 0.99$, compute $G_0 = 0.99^{29}$ (give the value to two decimals).
(c) With $\gamma = 0.9$, compute $G_0 = 0.9^{29}$ (order of magnitude suffices) and explain the design choice in [Section 21.1](./browser-rl-harness).

::: details Solution

(a) $G_0 = 1$: every step is credited with the full final outcome.
(b) $0.99^{29} \approx 0.75$. The opening query — often the most consequential decision — keeps three-quarters of its credit.
(c) $0.9^{29} \approx 0.047$: the early steps that chose the research direction receive almost no credit, so the agent cannot learn query strategy. That dilution is why browser agents use $\gamma = 1$ (finite horizon, no discounting) and handle credit assignment structurally instead — via process rewards, advantage normalization, or trajectory filtering.

:::

### Exercise 5 — Derive the masking rule

Write the REINFORCE-style policy gradient for one trajectory as a sum over token positions, then split the sum into model-generated positions $\mathcal{M}$ and environment-returned positions $\mathcal{E}$. Show what goes wrong if the $\mathcal{E}$ terms are kept, and state the implementation fix used in Search-R1.

::: details Solution

For trajectory tokens $x_1,\dots,x_T$ with return $G$ and baseline $b$:
$$\nabla_\theta J \approx (G-b)\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(x_t\mid x_{<t}) = (G-b)\left(\sum_{t\in\mathcal{M}}\nabla_\theta\log\pi_\theta(\cdot)+\sum_{t\in\mathcal{E}}\nabla_\theta\log\pi_\theta(\cdot)\right).$$
The $\mathcal{E}$ tokens (search results inside `<information>`) were sampled from the search engine, not from $\pi_\theta$ — but the gradient treats them as policy choices. A low-quality snippet then pushes the policy away from a context it did not choose, adding variance at best and a wrong attribution at worst. The fix is retrieved-token masking: multiply the per-token loss by $m_t = 1$ for $t \in \mathcal{M}$ and $m_t = 0$ for $t \in \mathcal{E}$, so only model-generated tokens carry gradients.

:::

### Exercise 6 — Pick the action space

For each task, choose among (i) search-API abstraction, (ii) full Playwright browser, (iii) set-of-mark clicking, and justify in one or two sentences:

(a) Answering factual questions from Wikipedia-style pages at minimum training cost.
(b) Comparing prices across JavaScript-heavy store sites that require login, scrolling, and pagination.
(c) Training a vision-language agent that must ground "the checkout button" on rendered pages.

::: details Solution

(a) Search-API abstraction: clean Markdown observations, tiny action space, cheap and cacheable — JavaScript rendering buys nothing on static text pages.
(b) Full Playwright browser: only a real browser preserves sessions, executes scripts, and exposes scroll/pagination; accept the latency and selector fragility, and record per-tool cost so the comparison with (a) is fair.
(c) Set-of-mark: the agent selects a labeled element id instead of regressing pixels or writing CSS selectors, which keeps actions compact while still training visual grounding; the harness must log labeling errors separately since a wrong box is an environment fault, not a policy fault.

:::

### Exercise 7 — Coding: extend the evaluation function

Extend `evaluate_run` from [Section 21.2](./deep-research-eval) so it also returns the redundancy score and a duplicate-action count. Assume `run["tool_calls"]` is a list of `{"kind": ..., "key": ...}` dicts where two calls with equal `(kind, key)` are duplicates.

```python
def evaluate_run_extended(task, run):
    ...
```

::: details Solution

```python
def evaluate_run_extended(task, run):
    calls = run["tool_calls"]
    seen, repeats = set(), 0
    for c in calls:
        key = (c["kind"], c["key"])
        if key in seen:
            repeats += 1
        seen.add(key)
    redundancy = repeats / len(calls) if calls else 0.0
    base = evaluate_run(task, run)  # answer, citation, within_budget, status
    base["redundancy"] = redundancy
    base["duplicate_actions"] = repeats
    return base
```

Check on Exercise 2's log: 8 calls, repeats $= 2 + 1 = 3$, redundancy $= 3/8 = 0.375$, matching the hand computation. Note the function still keeps components separate — the caller decides how to weight them.

:::

### Exercise 8 — Coding: idempotent retries in the harness

Extend `TinyResearchEnv` from [Section 21.1](./browser-rl-harness) so that:

1. Every tool action may carry an `idempotency_key`; a repeated key returns the cached observation with reward 0.0 and `done=False` instead of re-executing the tool.
2. A simulated tool failure (action name `"timeout_demo"`) returns the observation `{"error": "timeout", "retryable": True}` with reward 0.0 and `done=False`, and is recorded in the history like any other step.

::: details Solution

```python
class TinyResearchEnvRetry(TinyResearchEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}

    def step(self, action):
        key = action.get("idempotency_key")
        if key is not None and key in self._cache:
            # Duplicate: no backend call, but the turn still costs budget.
            self.steps += 1
            observation = self._cache[key][0]
            self.history.append((action, observation))
            return observation, 0.0, False, self.steps >= self.max_steps
        if action["name"] == "timeout_demo":
            self.steps += 1
            observation = {"error": "timeout", "retryable": True}
            self.history.append((action, observation))
            return observation, 0.0, False, self.steps >= self.max_steps
        result = super().step(action)
        if key is not None:
            self._cache[key] = result
        return result
```

Behavior check: calling `search` twice with the same key hits the backend once (verified by counting backend calls in a test harness); a retried timeout returns a typed, retryable observation instead of raising, so the policy can learn a backoff-and-retry strategy rather than crashing the rollout.

:::
