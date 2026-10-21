---
title: 19.11 Summary and Exercises
---

# 19.11 Summary and Exercises

This chapter moved the training objective from a single response to a complete interaction trajectory: the model acts, the environment responds, and learning happens over the whole loop. The pages below distill what that change requires — formalization, credit assignment, trajectory data, tool-use policy learning, and the rollout infrastructure that makes training run.

## What You Should Now Be Able to Do

1. Formalize a multi-turn agent task as a POMDP: name the joint state, the structured action space (text plus tool calls), the partial observations, and the terminal conditions.
2. Explain why the training objective becomes the expected cumulative return over trajectories, and why early actions carry disproportionate influence over later observations.
3. Decide between outcome rewards (ORM) and process rewards (PRM) for a new task, and describe what each choice costs in signal density versus annotation effort.
4. Trace how a trajectory-level reward is converted into per-step learning signal: discounted returns, baselines, and group-relative advantages.
5. Design a minimal verifiable reward for a tool-use task, including efficiency and format penalties, and justify each term.
6. Diagnose the three engineering bottlenecks of agentic RL — sandbox safety, nondeterministic environments, and tool latency — and explain how asynchronous rollouts, action masking, and retry accounting address each one.
7. Read a training curve together with rollout-health metrics (reward variance, tool-error rate, policy-version lag) and distinguish a learning failure from an infrastructure failure.
8. Run the chapter's rLLM labs end to end: prepare data, evaluate a baseline, apply GRPO, and check that the test split stayed independent and the reward was not hacked.

## Key Equations

$$ \tau = (s_0, a_0, o_1, a_1, o_2, \ldots, a_T) $$

A trajectory unifies model tokens, tool calls, and environment observations into the single object that RL optimizes.

$$ \langle S*{\text{agent}},\ A*{\text{agent}},\ P*{\text{agent}},\ R*{\text{agent}},\ \gamma,\ O \rangle $$

The POMDP six-tuple: the agent never sees the full environment state, only observations and interaction history.

$$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right] $$

The multi-turn objective: maximize expected discounted return over complete trajectories, not over single responses.

$$ G*t = r_t + \gamma G*{t+1} $$

Discounted return: each step's credit is its own reward plus discounted downstream rewards, so $\gamma$ controls how far terminal signal propagates backward.

$$ A_t = R(\tau) - \bar{R}(s_t) $$

Step-level advantage in its simplest form: compare this trajectory's return against the baseline return from the same state.

$$ A_i = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})} $$

GRPO group-relative advantage: within one group of rollouts for the same task, above-average trajectories are reinforced and below-average ones suppressed, with no critic network.

$$ R*{\text{total}} = R*{\text{task}} - \lambda*{\text{efficiency}} \cdot T - \lambda*{\text{format}} \cdot \mathbb{1}(\text{format error}) $$

Agentic reward with shaping: task completion minus penalties for trajectory length $T$ and malformed tool calls.

$$ \text{update utilization} = \frac{T*{\text{update}}}{T*{\text{rollout}} + T*{\text{verify}} + T*{\text{update}}} $$

Infrastructure health: the fraction of a training step spent optimizing versus generating rollouts and verifying them; low values mean the learner is starved.

## Common Pitfalls

1. **Training on environment tokens.** Forgetting the action mask lets the optimizer reinforce tool outputs and retrieved text as if the policy had chosen them. Mask observations (and prompts) so the loss sees only model decision tokens.
2. **Using sparse outcome reward on long horizons without support.** A binary terminal reward over a 50-turn research trajectory gives almost no per-step gradient. Either add process rewards, shorten the effective horizon with milestones, or use group comparison so that at least relative signal exists.
3. **Rewarding the shortcut instead of the skill.** Citation counts reward citation stuffing; test pass rates reward hardcoded answers. Verify with an independent check — held-out tests, evidence-connectivity checks, adversarial prompts — before trusting a rising curve.
4. **Treating the environment as deterministic.** Search results drift, sandboxes flake, and latencies spike. Old trajectories expire, retries need idempotency keys, and failed rollouts must be counted by cause rather than silently dropped.
5. **Confusing throughput with learning.** Raising GPU utilization with async rollouts also raises policy-version lag. If staleness grows unchecked, the learner chases behavior it already outgrew; bound the version lag and monitor it alongside reward.
6. **Evaluating on the training distribution.** Training tasks leaking into the test set, or a judge model scoring its own policy's outputs, inflates Pass@1 without improving the agent. Keep an independent split and re-check it after every reward change.

## Exercises

### Exercise 1 — Map an agent task onto the POMDP tuple

Take the flight-booking example from Section 19.1: search flights, filter by time and price, check seat inventory, then call the booking API. Write down what belongs in each slot of $\langle S, A, P, R, \gamma, O \rangle$, and state one thing the agent observes versus one thing about the true state it cannot observe.

::: details Solution
One valid mapping:

- $S$: the environment state space — airline database contents, live seat inventory, prices — plus the agent's context and conversation history so far.
- $A$: text replies plus structured tool calls such as `search_flights(origin, dest, date)`, `check_seat(flight_id)`, and `order(flight_id, passenger)`.
- $P$: how actions change the world and produce observations — a search returns matching rows, an order decrements inventory and returns a confirmation or an error.
- $R$: task reward, for example 1 if the correct ticket is issued and 0 otherwise, possibly with an efficiency penalty per turn.
- $\gamma$: discount factor weighting early decisions against later ones, typically close to 1 for short booking episodes.
- $O$: what the policy actually sees — rendered search results, tool return strings, and its own history; not the full airline database.

Observable: the list of flights returned for the issued query. Unobservable: whether a displayed price will change before booking, or seats held by other buyers. The policy must therefore act on history and partial observations, which is exactly why the POMDP — rather than a fully observed MDP — is the right formalization.
:::

### Exercise 2 — ORM or PRM?

For each task, say whether an outcome reward alone is a reasonable starting point or a process reward is needed from the start, and give the mechanism behind your answer: (a) code generation graded by hidden unit tests; (b) web navigation over 30-page browsing sessions; (c) multi-hop research answers requiring cited evidence.

::: details Solution
(a) ORM is a reasonable start. Unit-test pass/fail is verifiable, noise-free, and decomposable (pass rate gives partial credit), so the sparse terminal signal still correlates tightly with each step's contribution. This matches the finding discussed in Section 19.5 that single-step reward can guide multi-turn code generation.

(b) PRM from the start. Thirty turns of clicks with one terminal bit means almost every intermediate decision gets no gradient, and a lucky final click reinforces 29 wasted ones. Step-level scoring (as in Web-Shepherd) supplies the dense signal that makes credit assignment tractable.

(c) A hybrid. Answer correctness (ORM) is necessary but cannot detect fabricated citations, so add a process-level citation check: decompose the answer into atomic claims and verify each against fetched sources (the CaRR pattern). The mechanism is that verifiability differs per dimension — outcome for correctness, process for grounding — so the reward should follow the verifiability, not a single default.
:::

### Exercise 3 — Why mask tool observations?

A SearchR1-style rollout contains `<search>query</search>` tokens from the model and `<information>documents</information>` tokens from the retriever. Suppose you forget the action mask and train on every token. Describe concretely how the gradient update goes wrong, using the example of a trajectory where the search engine returned an excellent document but the model's query was poor.

::: details Solution
Without the mask, the policy-gradient loss includes log-probabilities of the `<information>` tokens, so the update increases the probability of generating that excellent document text. But the policy did not choose those tokens — the environment did. The model gets reinforced for output it never controlled, and worse, the genuinely bad decision (the poor query) receives relatively less corrective pressure because the trajectory's total reward is shared across all tokens. Over many steps the policy learns to imitate retriever language instead of learning when and what to search. Masking restricts the loss to model decision tokens — the query, the reasoning, the final answer — so credit flows only to choices the policy actually made.
:::

### Exercise 4 — Derive the masked multi-turn policy gradient

Starting from the trajectory objective $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ (take $\gamma = 1$ and a terminal reward for simplicity), derive the REINFORCE gradient and show how the action mask enters: which tokens contribute to $\nabla_\theta \log \pi_\theta(\tau)$ and why environment tokens must be excluded.

::: details Solution
Write the trajectory probability under the policy as a product over token positions:

$$ \pi*\theta(\tau) = \prod*{t=0}^{T} p*\theta(x_t \mid x*{<t}) $$

where $x_t$ ranges over every token in the trajectory, including tool observations. The REINFORCE gradient is:

$$ \nabla*\theta J(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}\left[ R(\tau) \nabla*\theta \log \pi*\theta(\tau) \right] = \mathbb{E}*{\tau \sim \pi*\theta}\left[ R(\tau) \sum*{t=0}^{T} \nabla*\theta \log p*\theta(x*t \mid x*{<t}) \right]. $$

Now split positions into model-generated tokens $a(\tau)$ and environment-produced tokens. Environment tokens carry no factor in the trajectory log-probability, so they contribute no term to the gradient; if they are nevertheless fed through the policy network as if self-generated, the implemented loss includes spurious pseudo-gradients that push the policy toward reproducing observations. The action mask $m_t \in \{0, 1\}$ (1 for model tokens, 0 for observations and prompts) corrects the estimator:

$$ \nabla*\theta J(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}\left[ R(\tau) \sum*{t \in a(\tau)} \nabla*\theta \log p*\theta(x*t \mid x*{<t}) \right], $$

so only decisions the policy controlled are reinforced. This is the same masking used in the action-mask formalism of Section 19.2 and in the rollout code of the chapter's project pages.
:::

### Exercise 5 — Compute discounted returns by hand

A three-turn agent trajectory earns immediate rewards $[0, 0, 1]$ (failure, failure, final success).

1. Compute the discounted return $G_t$ at each step for $\gamma = 0.9$.
2. Repeat for $\gamma = 0.5$.
3. Explain in one paragraph what the comparison shows about credit propagation to the first turn.

::: details Solution
Work backward from $G_t = r_t + \gamma G_{t+1}$ with $G_3 = 0$.

For $\gamma = 0.9$:

- $G_2 = 1 + 0.9 \times 0 = 1.0$
- $G_1 = 0 + 0.9 \times 1.0 = 0.9$
- $G_0 = 0 + 0.9 \times 0.9 = 0.81$

So $\mathbf{G} = [0.81, 0.9, 1.0]$.

For $\gamma = 0.5$:

- $G_2 = 1.0$
- $G_1 = 0.5$
- $G_0 = 0.25$

So $\mathbf{G} = [0.25, 0.5, 1.0]$.

Comparison: with $\gamma = 0.9$ the first turn still receives 81% of the terminal reward, so the opening decision gets a meaningful learning signal; with $\gamma = 0.5$ it receives only 25%, and the signal decays geometrically with distance from the reward. This is why long-horizon agents need $\gamma$ close to 1 (or dense process rewards): heavy discounting strands early decisions — which Section 19.1 showed are often the decisions that determine the whole trajectory — with near-zero gradient.
:::

### Exercise 6 — Group-relative advantage and Pass@1

In the DeepCoder lab, one GRPO group of 4 rollouts for the same problem earns rewards $[0, 1, 0, 1]$.

1. Compute the group mean, the population standard deviation, and the advantage of each rollout.
2. State which rollouts are reinforced and which are suppressed.
3. The baseline evaluation solves 6 of 20 problems and the post-training evaluation scores about 20 of 50 (39.0% as printed by the lab). Compute both Pass@1 values and the percentage-point gain.

::: details Solution

1. Mean: $(0 + 1 + 0 + 1)/4 = 0.5$. Squared deviations are each $(0 - 0.5)^2 = 0.25$, so variance $= 0.25$ and population std $= 0.5$. Advantages $A_i = (r_i - 0.5)/0.5$:

$$ \mathbf{A} = [-1.0,\ +1.0,\ -1.0,\ +1.0]. $$

2. The two successful rollouts (advantage $+1$) are reinforced — the update raises the probability of the decisions that produced them. The two failed rollouts (advantage $-1$) are suppressed. Note the signal exists only because the group is mixed: had all four succeeded or all four failed, the std would be 0 and no relative advantage could be computed, which is why GRPO needs enough samples per group.

3. Baseline Pass@1 $= 6/20 = 30.0\%$. Post-training Pass@1 $= 39.0\%$ as reported by the lab (about 20 of 50). Gain $\approx 9$ percentage points (consistent with the lab's reported +8–10 point range).
   :::

### Exercise 7 — Coding: verify discounted returns with the chapter code

Using `compute_discounted_returns` from `code/chapter19_agentic/multi_turn_rl.py`, write a short script that checks the hand computation from Exercise 5: for immediate rewards `[0, 0, 1]`, assert the function returns `[0.81, 0.9, 1.0]` at $\gamma = 0.9$ and `[0.25, 0.5, 1.0]` at $\gamma = 0.5$ (within floating-point tolerance). Also assert the ORM property used in the chapter: for any reward vector with a single terminal reward of 1.0, the last return equals 1.0 regardless of $\gamma$.

::: details Solution

```python
import importlib.util
import math

spec = importlib.util.spec_from_file_location(
    "mtrl", "code/chapter19_agentic/multi_turn_rl.py")
# NOTE: importing executes the module's demo simulation and plots;
# in practice, copy compute_discounted_returns into your script instead.
```

Because the chapter script runs its full simulation and plotting on import, the robust approach is to copy the six-line function (it has no dependencies beyond plain Python):

```python
def compute_discounted_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def check(rewards, gamma, expected):
    got = compute_discounted_returns(rewards, gamma=gamma)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert math.isclose(g, e, rel_tol=1e-9), (g, e)

check([0, 0, 1], 0.9, [0.81, 0.9, 1.0])
check([0, 0, 1], 0.5, [0.25, 0.5, 1.0])

# ORM property: terminal reward 1.0 always yields final return 1.0.
for gamma in (0.0, 0.5, 0.9, 0.99):
    assert compute_discounted_returns([0, 0, 0, 1], gamma=gamma)[-1] == 1.0

print("All return checks passed.")
```

Why the last assertion holds: the loop starts with $G = 0$ and processes the final reward first, so $G \leftarrow 1 + \gamma \times 0 = 1$ before any discounting applies. Earlier positions then decay geometrically, which is exactly the credit-propagation behavior computed by hand in Exercise 5.
:::

### Exercise 8 — Coding: one REINFORCE step for tool selection

`code/chapter19_agentic/tool_use_agent.py` implements a softmax tool policy updated with REINFORCE. Suppose one query type has logits $[0, 0, 0]$ over tools `[search, calculate, run_code]`, the sampled action is index 0 (`search`), the reward is $+1.0$, and the learning rate is $0.05$.

1. Compute by hand the new logits and the new probability of choosing `search`.
2. Write a script that performs the same update (softmax + `grad = one_hot − probs`) and asserts your hand-computed probability.

::: details Solution

1. Initial probabilities are uniform: $p = [1/3, 1/3, 1/3]$. The REINFORCE gradient of the log-probability for a softmax policy is $e_0 - p = [2/3, -1/3, -1/3]$. The update is $\Delta = 0.05 \times 1.0 \times [2/3, -1/3, -1/3] = [0.03333, -0.01667, -0.01667]$, so the new logits are $[0.03333, -0.01667, -0.01667]$. Softmax: $\exp(0.03333) \approx 1.0339$, $\exp(-0.01667) \approx 0.9835$ (twice), sum $\approx 3.0009$, giving:

$$ p(\text{search}) \approx \frac{1.0339}{3.0009} \approx 0.3445. $$

One rewarded trial moves the correct-tool probability from 33.3% to about 34.5% — small steps like this accumulate over the 50 training episodes in the chapter script.

2. Verification script:

```python
import math

logits = [0.0, 0.0, 0.0]
lr, reward, action = 0.05, 1.0, 0

m = max(logits)
exps = [math.exp(x - m) for x in logits]
probs = [e / sum(exps) for e in exps]

grad = [-p for p in probs]
grad[action] += 1.0
logits = [l + lr * reward * g for l, g in zip(logits, grad)]

m = max(logits)
exps = [math.exp(x - m) for x in logits]
new_probs = [e / sum(exps) for e in exps]

assert math.isclose(new_probs[0], 0.3445, rel_tol=1e-3), new_probs
assert new_probs[0] > 1 / 3 > new_probs[1]
print(f"Updated probs: {[round(p, 4) for p in new_probs]}")
```

The assertions check both the absolute value from the hand derivation and the qualitative direction: the rewarded tool gains probability while the alternatives lose it, which is the mechanism that drives the policy from the uniform 33% baseline toward reliable tool selection.
:::
