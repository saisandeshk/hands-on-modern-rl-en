---
title: 20.R Summary and Exercises
---

# 20.R Summary and Exercises

This chapter treated repository repair as a trainable trajectory: an agent reads an issue, explores a codebase, edits, runs tests, and RL raises the probability of trajectories that pass held-out verification. Section 20.1 built the SWE-bench task format, the RLVR test reward, and the rollout pipeline; the supplementary reading showed the SWE-RL recipe (GRPO plus a binary test reward); Section 20.2 separated environment-trajectory modeling (CWM) from long-horizon pure RL with test-time scaling (DeepSWE); Section 20.3 closed the data loop with self-play defect injection and repair (SSR).

## What You Should Now Be Able to Do

- Formalize a repair session as a trajectory $\tau = (s_0, a_0, s_1, \ldots, s_T)$ and explain which parts of it the reward observes.
- Convert executable test outcomes into pass-fraction and binary-resolved rewards, and state when each one trains better.
- Design a verifier split (visible, held-out, adversarial hidden tests) that blocks hardcoding, deleted tests, and entry-point tampering.
- Specify a sandbox contract: reset, deterministic replay, side-effect boundaries, and distinct status codes for tool errors, timeouts, and assertion failures.
- Diagnose a training run with rollout plumbing concepts: synchronous versus asynchronous loops, policy versions, staleness, backpressure, and checkpoint recovery.
- Compare CWM-style environment modeling, DeepSWE-style pure RL with test-time scaling, and SSR-style self-play on data cost, reward density, and transfer risk.
- Account compute in three ledgers (sampling, inference, training) and report verifier throughput and update utilization alongside resolve rate.
- Detect specification gaming and flywheel failure modes (generator–fixer imbalance, domain drift, invalid defects) before they corrupt the reward signal.

## Key Equations

- **Repair trajectory.** $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$. Reading: later states depend on earlier tool actions, so RL must credit the whole search-and-edit sequence, not just the final diff.
- **Test pass fraction.** $R_{\text{test}}(\tau) = \frac{1}{m}\sum_{i=1}^{m} z_i$ with $z_i \in \{0, 1\}$. Reading: the share of passing tests; dense feedback for partial repair.
- **Resolved reward.** $R_{\text{resolved}}(\tau) = \mathbb{1}[\sum_i z_i = m]$. Reading: 1 only when every test passes; matches the deployment criterion exactly but is sparse.
- **Shaped teaching reward.** $R(\tau) = R_{\text{test}} - \lambda_{\text{step}} C_{\text{step}} + \lambda_{\text{patch}} Q_{\text{patch}} + \lambda_{\text{ctx}} E_{\text{ctx}}$. Reading: each shaping term buys efficiency or style at the price of a new gaming surface — ablate term by term.
- **Group-relative advantage (GRPO).** $A_i = (r_i - \mu)/\sigma$ over a group of rollouts of the same task. Reading: rollouts compete within their task, which normalizes away task-difficulty differences.
- **TD error and GAE.** $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$, $\hat{A}_t = \sum_{l \ge 0}(\gamma\lambda)^l \delta_{t+l}$. Reading: the value model converts one sparse terminal reward into per-step credit.
- **KL-regularized objective.** $\mathcal{L} = \mathcal{L}_{\text{RL}} + \beta\,\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$. Reading: $\beta$ trades reward optimization against staying close to readable, conventional code.
- **Compute ledger.** $C_{\text{total}} = C_{\text{sample}} + C_{\text{inference}} + C_{\text{train}}$. Reading: sampling, test-time, and gradient costs answer different questions — never merge them into one number.

## Common Pitfalls

- **Grading on visible tests.** If the agent can read or rewrite the tests that decide the reward, the policy learns test manipulation (hardcoded answers, deleted assertions) instead of repair. Keep the graded set read-only and hidden.
- **Counting infrastructure failures as model failures.** A failed dependency install or container timeout is not evidence the patch was bad. Use separate status codes, or the learner optimizes against noise.
- **Confusing Pass@1 with scaled accuracy.** A 59%-style figure that spends search or repeated sampling per task cannot be compared with a single-sample Pass@1. Always report the test-time budget beside the score.
- **Shaping before the binary reward works.** Length penalties and patch-quality bonuses change the optimum of the objective. Verify the plain test reward first, then add one shaping term at a time with ablations.
- **Letting the flywheel grade its own homework.** A self-play generator that proposes defects and a fixer that solves them can collude on trivial or invalid bugs (for example, syntax errors). Enforce the validity triple — test passes on original code, fails after injection, passes after repair — and check transfer to human-written issues.

## Exercises

### Exercise 1 — Pass Fraction versus Binary Reward

A task has $m = 12$ tests. A rollout passes 9 of them. Compute $R_{\text{test}}$ and $R_{\text{resolved}}$. Then state one training situation where the pass fraction helps and one where the binary reward is the right objective.

::: details Solution
$R_{\text{test}} = 9/12 = 0.75$. $R_{\text{resolved}} = 0$ because not all 12 pass.

Use the pass fraction early in training or on hard multi-part tasks: a patch that fixes 9 of 12 tests earns 0.75 and gives the optimizer a gradient direction, whereas the binary reward gives 0 and no information. Use the binary reward when the deployment criterion is all-or-nothing (the release either passes the suite or it does not), and always report the binary resolve rate even when training on the fraction — otherwise partial fixes can look like solved tasks.
:::

### Exercise 2 — Shaped Reward by Hand

A rollout passes 9 of 12 tests and takes 18 tool steps. The shaped reward is $R = R_{\text{test}} - 0.01 \times \text{len}(\tau)$. Compute $R$. Then explain what goes wrong if the step coefficient is raised to $0.1$ on tasks that genuinely need about 40 steps.

::: details Solution
$R_{\text{test}} = 9/12 = 0.75$. Step penalty $= 0.01 \times 18 = 0.18$. So $R = 0.75 - 0.18 = 0.57$.

With coefficient $0.1$, a necessary 40-step repair pays $0.1 \times 40 = 4.0$ in penalty — far larger than the maximum test reward of $1.0$. The optimal policy then prefers immediate submission of an unchanged patch ($R \approx 0 - 0.1 \times 1 = -0.1$, or 0 with no penalty on abort) over a genuine repair ($1.0 - 4.0 = -3.0$). The agent learns to abort, and both the resolve rate and the average step count must be watched together to catch this.
:::

### Exercise 3 — Group-Relative Advantage by Hand

Four rollouts of the same task earn rewards $[1, 1, 0, 0]$. Compute the group mean $\mu$, the population standard deviation $\sigma$, and each rollout's advantage $A_i = (r_i - \mu)/\sigma$. What happens to this computation if all four rollouts fail?

::: details Solution
$\mu = (1 + 1 + 0 + 0)/4 = 0.5$. Squared deviations: $(0.5)^2 + (0.5)^2 + (-0.5)^2 + (-0.5)^2 = 0.25 \times 4 = 1.0$. Variance $= 1.0/4 = 0.25$, so $\sigma = 0.5$.

Advantages: for each $r_i = 1$, $A_i = (1 - 0.5)/0.5 = 1.0$; for each $r_i = 0$, $A_i = (0 - 0.5)/0.5 = -1.0$. So $A = [1.0, 1.0, -1.0, -1.0]$, which sums to zero as expected within a group.

If all four fail, rewards are $[0,0,0,0]$, so $\mu = 0$ and $\sigma = 0$ — the advantage is undefined (division by zero) and the group carries no learning signal. This is exactly why success-rate filtering discards prompts with zero intra-group variance (and, symmetrically, prompts where every rollout succeeds trivially).
:::

### Exercise 4 — GAE Credit Assignment by Hand

A two-step repair trace has rewards $r_0 = 0$, $r_1 = 1$ and value estimates $V_0 = 0.2$, $V_1 = 0.5$, $V_2 = 0$ (terminal). With $\gamma = 0.99$ and $\lambda = 0.95$, compute $\delta_0$, $\delta_1$, $\hat{A}_1$, and $\hat{A}_0$. Explain in one sentence what $\hat{A}_0$ tells the policy.

::: details Solution
$\delta_1 = r_1 + \gamma V_2 - V_1 = 1 + 0.99 \times 0 - 0.5 = 0.5$.

$\delta_0 = r_0 + \gamma V_1 - V_0 = 0 + 0.99 \times 0.5 - 0.2 = 0.495 - 0.2 = 0.295$.

Terminal step: $\hat{A}_1 = \delta_1 = 0.5$.

$\hat{A}_0 = \delta_0 + \gamma\lambda\,\delta_1 = 0.295 + (0.99 \times 0.95) \times 0.5 = 0.295 + 0.9405 \times 0.5 = 0.295 + 0.47025 = 0.76525$.

$\hat{A}_0 > 0$ tells the policy to raise the probability of the first action: even though no reward arrived at step 0, that action led to a state ($V_1 = 0.5$) from which the final reward became reachable. This is precisely how the value model converts a sparse terminal signal into per-step credit.
:::

### Exercise 5 — Design a Verifier Split

An agent may run tests during its rollout, and a final score decides the reward. Design the visible / held-out / adversarial-hidden split for a Python repair task, and describe two concrete attacks the split defeats.

::: details Solution
One workable split: (a) **visible tests** — 2–3 example tests shipped with the issue that the agent may execute any number of times for debugging; (b) **held-out regression tests** — the repository's remaining suite, run only by the verifier from a read-only copy; (c) **adversarial hidden tests** — extra cases added by the task author covering boundary inputs (empty collections, `None`, off-by-one indices) plus a collection-integrity check that fails if any test file was modified, skipped, or never collected.

Attack 1, hardcoding (`if input == [1, 2]: return 42`): passes the visible examples but fails the hidden boundary cases. Attack 2, deleting or skipping the failing test: the integrity check and the held-out run mark the trajectory as an environment violation rather than a pass. A gap between the visible pass rate and the held-out pass rate quantifies how much of the reward was gaming.
:::

### Exercise 6 — Throughput and Cost Arithmetic

A training job covers 200 tasks with 8 rollouts each. Each rollout averages 6,000 generated tokens and one test-suite run. The verifier completes 3 suites per minute on a single worker, and 8 verifier workers run in parallel. Compute (a) total rollouts, (b) total generated tokens, (c) single-worker verifier hours, and (d) wall-clock verifier time with 8 workers. Which ledger does each quantity belong to?

::: details Solution
(a) Total rollouts $= 200 \times 8 = 1{,}600$ trajectories.

(b) Total generated tokens $= 1{,}600 \times 6{,}000 = 9{,}600{,}000$ (9.6M tokens).

(c) Verifier load $= 1{,}600$ suites at 3 per minute $= 1{,}600 / 3 \approx 533.3$ minutes $\approx 8.9$ hours single-worker.

(d) With 8 parallel workers, $533.3 / 8 \approx 66.7$ minutes wall-clock (ignoring queueing overhead).

Ledger mapping: (b) is rollout/inference token cost ($C_{\text{sample}}$ during training, $C_{\text{inference}}$ at evaluation); (c)–(d) are verifier throughput, usually the binding constraint and the first thing to scale. None of these is $C_{\text{train}}$ (gradient-update FLOPs), which must be reported separately.
:::

### Exercise 7 — Coding: Test Reward and Bug Validity

Implement two functions: `swe_reward(test_results)`, returning the pass fraction (0.0 for an empty list), and `is_valid_defect(passes_original, fails_injected, passes_repaired)`, returning `True` only when the test passes on the original code, fails after injection, and passes again after repair. Then evaluate `is_valid_defect` on the triples `(True, True, True)`, `(True, True, False)`, and `(False, True, True)`, and explain which flywheel failure each rejecting case prevents.

::: details Solution

```python
def swe_reward(test_results):
    if len(test_results) == 0:
        return 0.0
    return sum(test_results) / len(test_results)


def is_valid_defect(passes_original, fails_injected, passes_repaired):
    return bool(passes_original and fails_injected and passes_repaired)


print(swe_reward([True] * 9 + [False] * 3))  # 0.75
print(swe_reward([]))                        # 0.0
print(is_valid_defect(True, True, True))     # True
print(is_valid_defect(True, True, False))    # False
print(is_valid_defect(False, True, True))    # False
```

Case `(True, True, False)` — fails after repair — rejects unfixable or mis-specified injections; training on them teaches the fixer that effort never pays. Case `(False, True, True)` — fails even on the original code — rejects broken test patches; without this check the generator is rewarded for writing tests that fail regardless of the injected bug. The most common degenerate case in practice is a syntax-error injection, which the `(True, True, True)` triple alone does not catch — pair the triple with a check that the injected diff parses and resembles a plausible edit.
:::

### Exercise 8 — Coding: Success-Rate Filtering and Staleness

Implement `filter_prompts(prompt_rewards)` that keeps only prompts with at least one success ($r > 0$) and nonzero variance, and `staleness_weights(versions, current, max_lag)` that returns weight 1.0 for trajectories within `max_lag` versions of the current policy and 0.0 otherwise. Run both on `rewards = {"a": [1, 0, 0, 0], "b": [0, 0, 0, 0], "c": [1, 1, 1, 1]}` with versions `{"a": 9, "b": 9, "c": 4}`, `current = 10`, `max_lag = 2`, and state which prompts survive both filters.

::: details Solution

```python
def _variance(xs):
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


def filter_prompts(prompt_rewards):
    kept = []
    for prompt, rs in prompt_rewards.items():
        if max(rs) > 0 and _variance(rs) > 0:
            kept.append(prompt)
    return kept


def staleness_weights(versions, current, max_lag):
    return {
        p: (1.0 if current - v <= max_lag else 0.0)
        for p, v in versions.items()
    }


rewards = {"a": [1, 0, 0, 0], "b": [0, 0, 0, 0], "c": [1, 1, 1, 1]}
versions = {"a": 9, "b": 9, "c": 4}

print(filter_prompts(rewards))  # ['a']
print(staleness_weights(versions, current=10, max_lag=2))
# {'a': 1.0, 'b': 1.0, 'c': 0.0}
```

`filter_prompts` keeps only `"a"`: `"b"` has no success (all-zero group, zero variance, no signal) and `"c"` has zero variance (all succeed trivially, no contrast to learn from). The staleness filter keeps `"a"` (lag $10 - 9 = 1 \le 2$) and drops `"c"` (lag $6 > 2$). Only prompt `"a"` survives both filters — it is learnable (mixed outcomes) and fresh (on-policy enough for a clipped update).
:::
