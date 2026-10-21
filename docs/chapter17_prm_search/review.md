---
title: 17.7 Summary and Exercises
---

# 17.7 Summary and Exercises

## What you should now be able to do

- Explain why a terminal outcome score underdetermines credit assignment on long trajectories, using the faulty $\sqrt{2}$ proof as the running example.
- Write down the mathematical forms of an outcome reward and a process reward, and describe how step scores enter RL training (trajectory-level vs. token-level).
- Describe the discriminative PRM data pipeline (step splitting, good/bad/neutral labels, cross-entropy training, score aggregation) and its cost drivers.
- Describe how a generative PRM (ThinkPRM) reuses human labels as a filter rather than as direct classifier targets, and when its inference cost is worth paying.
- State what a formal verifier guarantees (soundness by kernel check) and what it does not (statement faithfulness, parser boundaries, completeness under budgets).
- Choose among Best-of-N, Beam Search, Tree of Thoughts, MCTS, and parallel aggregation for a new task, and justify the choice from verifier reliability and compute budget.
- Run and extend the Game-of-24 search code in `code/chapter17_prm_search/` to compare search breadth against success rate.

## Key equations

- **Discounted return:** $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots$ — passes one terminal score backward with geometrically decaying weight.
- **GAE advantage:** $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, $A_t^{\text{GAE}} = \sum_{l} (\gamma\lambda)^l \delta_{t+l}$ — smooths the backward pass but still starts from the terminal score.
- **ORM / PRM signatures:** $r_{\text{ORM}} = \text{ORM}(q, o)$ scores a whole response; $r_t = \text{PRM}(q, o, \text{step}(t))$ scores the step each token belongs to.
- **Discriminative PRM loss:** $\mathcal{L}_{\text{PRM}} = -\sum_i \sum_{c} y_{i,c} \log p_\theta(c \mid q, o_{\leq i}, s_i)$ — cross-entropy against good/bad/neutral labels; gradients concentrate on serious misjudgments.
- **Aggregation (mean / min / product):** $\text{mean} = \frac{1}{T}\sum_i p_i$, $\min = \min_i p_i$, $\text{product} = \prod_i p_i$ — three different answers to "how much should one bad step count?"
- **Coverage formula:** $\text{Pass}@k = 1-(1-p)^k$ — the probability that at least one of $k$ independent attempts succeeds given single-attempt rate $p$.
- **UCB selection:** $\text{UCB}(n) = Q(n) + c\sqrt{\ln N(p)/N(n)}$ — average reward plus an exploration bonus for rarely visited nodes.
- **Incremental mean update:** $Q(n) \leftarrow Q(n) + (r - Q(n))/N(n)$ — running average of rollout returns without storing history.
- **GenRM score:** $\text{GenRM}(q, o) = P(\text{"good"} \mid q, o, \text{prompt})$ — the judgment-token probability read out as a scalar reward.
- **Majority-vote reliability:** $P(\text{majority correct}) = \sum_{k>N/2} \binom{N}{k} p^k (1-p)^{N-k}$ — rises with $N$ only if errors are independent and answers are comparable.

## Common pitfalls

- **Reading Best-of-N or Pass@$k$ numbers as single-attempt accuracy.** A PRM reranking result such as 78% with many candidates says nothing about greedy decoding; always report the candidate budget alongside the number.
- **Using min-aggregation outside tight logical chains.** Min works for math proofs where one wrong equation ruins the proof, but it is overly harsh for open-ended writing; validate the aggregation on the target task.
- **Assuming a formal pass means the original problem is solved.** The kernel checks the formal statement, not the translation; a mistranslated quantifier or missing side condition makes every downstream proof meaningless.
- **Letting the generator and the judge share blind spots.** Self-grading and same-family judges produce fluent agreement on wrong answers; keep an independent backstop (executable checks, a different judge, human spot checks) and report disagreement.
- **Optimizing the proxy past its validity.** As the policy drifts off the reward model's training distribution, the PRM score can rise while true task success falls; track an independent metric, version the reward and verifier, and roll back when they diverge.
- **Paying for search without a reliable intermediate signal.** Tree search and parallel breadth multiply token cost by an order of magnitude or more; with a noisy PRM the extra compute amplifies the evaluator's bias instead of correcting errors.

## Exercises

### Exercise 1: When is an ORM enough?

A team wants to train a model for (a) single-sentence sentiment classification, (b) 5,000-token tool-calling trajectories that book travel, and (c) math contest proofs. For each, say whether an ORM alone suffices or a PRM is worth its cost, using the three conditions from Section 17.1 (segmentable steps, error propagation, evaluator reliability).

::: details Solution
(a) ORM suffices: the trajectory is one short decision, there is no intermediate error to locate, and resampling is cheap.

(b) PRM (or at least step-level checks) is worth it: tool calls are natural step boundaries, one early wrong call (wrong date, wrong city) poisons thousands of later tokens, and execution results plus receipts give an evaluator more reliable than the generator.

(c) PRM or formal verification is worth it: proof steps segment naturally, one wrong equation invalidates everything after it, and mathematical rules let the evaluator (human experts, stronger models, or Lean) judge steps more reliably than the generator produces them. For short contest answers with cheap automatic checking, an ORM plus sampling can still be the starting point.
:::

### Exercise 2: Invert a Pass@k number

A prover reports Pass@32 = 82.4% on MiniF2F-test. Assuming independent attempts with single-attempt rate $p$, solve for $p$ and show the steps.

::: details Solution
$\text{Pass}@k = 1-(1-p)^k$, so $(1-p)^{32} = 1 - 0.824 = 0.176$.

Take logs: $32 \ln(1-p) = \ln 0.176 \approx -1.737$.

Hence $\ln(1-p) \approx -0.0543$, $1-p \approx e^{-0.0543} \approx 0.947$, and $p \approx 0.053$, i.e. about **5.3%** single-attempt success. The headline 82.4% needs 32 attempts to materialize.
:::

### Exercise 3: Discounted-return weights by hand

For the nine-step proof with only $r_9 = 1$ and $\gamma = 0.9$, compute the return weights $\gamma^{T-t}$ for steps 1, 5, 6, and 8, and explain in one sentence why this weighting is backwards for proofs.

::: details Solution
Step 1 is 8 steps from the end: $0.9^8 \approx 0.430$. Step 5 is 4 steps away: $0.9^4 = 0.656$. Step 6 is 3 steps away: $0.9^3 = 0.729$. Step 8 is 1 step away: $0.9^1 = 0.900$.

The founding assumption of the proof by contradiction (step 1) gets the smallest weight while the steps nearest the conclusion get the largest, even though in a proof early definitions determine everything downstream — responsibility runs forward, not backward.
:::

### Exercise 4: Aggregation picks different winners

Candidates A and B have step scores A = $[0.9, 0.8, 0.2, 0.9]$ and B = $[0.6, 0.6, 0.6, 0.6]$. Compute mean, min, and product for each, state which candidate each rule selects, and say which rule fits mathematical proofs.

::: details Solution
A: mean $= (0.9+0.8+0.2+0.9)/4 = 2.8/4 = 0.70$; min $= 0.2$; product $= 0.9 \times 0.8 \times 0.2 \times 0.9 = 0.1296$.

B: mean $= 0.60$; min $= 0.6$; product $= 0.6^4 = 0.1296$.

Mean picks A ($0.70 > 0.60$), min picks B ($0.6 > 0.2$), product ties ($\approx 0.130$ each). For math proofs, min (or product) fits better: a single wrong equation with score 0.2 should sink the solution even if the other steps look good.
:::

### Exercise 5: UCB by hand

A parent node has $N(p) = 100$ visits ($\ln 100 \approx 4.605$) and $c = \sqrt{2} \approx 1.414$. Children: A with $Q = 0.8$, $N = 50$; B with $Q = 0.7$, $N = 30$; C with $Q = 0.6$, $N = 5$. Compute each UCB and say which node is expanded next.

::: details Solution
Exploration term is $1.414\sqrt{4.605/N}$.

A: $1.414\sqrt{4.605/50} = 1.414 \times 0.3035 \approx 0.429$; UCB $\approx 0.8 + 0.429 = 1.229$.

B: $1.414\sqrt{4.605/30} = 1.414 \times 0.3918 \approx 0.554$; UCB $\approx 0.7 + 0.554 = 1.254$.

C: $1.414\sqrt{4.605/5} = 1.414 \times 0.9597 \approx 1.357$; UCB $\approx 0.6 + 1.357 = 1.957$.

Node C is expanded next despite the lowest average, because its exploration bonus dominates. If repeated visits keep its average near 0.6, the bonus decays and the budget returns to A and B.
:::

### Exercise 6: Majority vote gain and its limits

Each of 5 paths is correct independently with $p = 0.6$. Compute the majority-vote accuracy, the gain over single-attempt accuracy, and name the two assumptions that break on hard problems.

::: details Solution
$P = \binom{5}{3}0.6^3 0.4^2 + \binom{5}{4}0.6^4 0.4 + \binom{5}{5}0.6^5 = 10(0.216)(0.16) + 5(0.1296)(0.4) + 0.07776 \approx 0.3456 + 0.2592 + 0.0778 = 0.6826$, about **0.68**.

Gain: $0.683 - 0.600 \approx 0.083$, under 9 percentage points, at 5 times the generation cost (plus normalization and voting).

Breaking assumptions: (1) errors are independent — systematic mistakes make all paths fail together and even produce unanimous wrong answers; (2) answers are uniquely comparable — without normalization and semantic judgment, equivalent answers (e.g. $\frac{1}{2}$ vs. $0.5$) split the vote.
:::

### Exercise 7 (coding): Min-aggregation reranker on the chapter's example

Using only the Python standard library, write a function `select(candidates, rule)` where each candidate maps to a list of step scores and `rule` is one of `"mean"`, `"min"`, or `"product"`. Test it on A = $[0.9, 0.8, 0.2, 0.9]$ and B = $[0.6, 0.6, 0.6, 0.6]$, and reproduce the Exercise 4 result. Then apply your function to a small reranking set built from `code/chapter17_prm_search/tree_of_thought.py` step scores of your choice.

::: details Solution

```python
import math

def aggregate(scores, rule):
    if rule == "mean":
        return sum(scores) / len(scores)
    if rule == "min":
        return min(scores)
    if rule == "product":
        return math.prod(scores)
    raise ValueError(rule)

def select(candidates, rule):
    scored = {name: aggregate(s, rule) for name, s in candidates.items()}
    best = max(scored, key=lambda k: scored[k])
    return best, scored

cands = {"A": [0.9, 0.8, 0.2, 0.9], "B": [0.6, 0.6, 0.6, 0.6]}
for rule in ("mean", "min", "product"):
    print(rule, select(cands, rule))
```

Expected output (verified): `mean` picks A with `{'A': 0.7, 'B': 0.6}`; `min` picks B with `{'A': 0.2, 'B': 0.6}`; `product` is effectively a tie (`A ≈ 0.1296`, `B = 0.1296`; in floating point `math.prod` gives `0.12960000000000002` vs `0.1296`, so the `max` call returns `A`). For the second part, score any `ThoughtNode` path with `evaluate_node` from `tree_of_thought.py`, collect one score per depth level, and confirm that `min` rejects paths containing a near-zero step while `mean` can still select them.
:::

### Exercise 8 (coding): UCB picker and running-average update

Implement `ucb(q, n, n_parent, c)` and `update(q, n, r)` for the incremental mean $Q \leftarrow Q + (r - Q)/N$ used in the MCTS backpropagation step. Verify on the Exercise 5 numbers that C wins, then simulate 10 rewards of $0.6$ arriving at C and show its exploration bonus decaying.

::: details Solution

```python
import math

def ucb(q, n, n_parent, c=math.sqrt(2)):
    return q + c * math.sqrt(math.log(n_parent) / n)

def update(q, n_before, r):
    n = n_before + 1
    return q + (r - q) / n, n

nodes = {"A": (0.8, 50), "B": (0.7, 30), "C": (0.6, 5)}
for name, (q, n) in nodes.items():
    print(name, round(ucb(q, n, 100), 3))
# A 1.229, B 1.254, C 1.957 -> expand C

q, n = 0.6, 5
for _ in range(10):
    q, n = update(q, n, 0.6)
print("after 10 more visits: Q =", round(q, 3), "N =", n,
      "UCB =", round(ucb(q, n, 100 + 10), 3))
```

Expected output (verified): `A 1.229, B 1.254, C 1.957`. After 10 further visits with reward $0.6$ each, $Q$ stays $0.6$ and $N = 15$; with the parent at 110 visits ($\ln 110 \approx 4.700$), the bonus falls to about $1.414\sqrt{4.700/15} \approx 0.792$ and UCB to about $1.392$ — still competitive here, but every additional visit without improvement shrinks the bonus until A and B retake the lead. This is the exploration term being "used up."
:::
