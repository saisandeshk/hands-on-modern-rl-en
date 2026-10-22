---
title: '20.L Lab: Reward Hacking in a Toy Code Agent'
---

# 20.L Lab: Reward Hacking in a Toy Code Agent

Chapter 20 argues that executable tests turn software repair into a reinforcement-learning problem -- and warns that the reward is only as faithful as the verifier behind it. This lab lets you watch both halves of that claim happen in a few minutes on a laptop: a policy trained with verifiable rewards learns to solve small coding tasks, and the same policy trained on an incomplete verifier learns to game it.

## 20.L.1 What you will learn

- How a test suite becomes an RL reward: a candidate implementation is executed in an isolated sandbox, and the pass fraction $R_{\text{test}}$ is the reward signal.
- How REINFORCE with a per-task baseline raises the probability of high-reward candidates in a discrete action space (one softmax per task).
- What reward hacking looks like quantitatively: train reward 1.000 alongside a hidden-test pass rate of 0.491, with 0.537 of the policy mass sitting on memorizing shortcuts.
- Why held-out tests and randomized test inputs fix the hack: both restore the correct implementations as the unique optimum.

## 20.L.2 Background

Read [20.1 Foundations of SWE-RL](/chapter20_swe_agents/swe-bench-and-rlvr) first, especially the sections on test rewards and the limits of the test signal. Our lab is a direct miniature of that section:

- Each toy task has **visible tests** (the analogue of the repository tests the agent may run) and **hidden tests** (the analogue of the held-out evaluation tests). The chapter's split between "tests the agent can call" and "tests that decide the result" is implemented literally as two lists per task.
- The training reward is the chapter's pass fraction $R_{\text{test}}(\tau) = \frac{1}{m}\sum_i z_i$, and the headline evaluation metric is its binary sibling $R_{\text{resolved}}$, reported here as the fraction of tasks whose greedy pick passes every hidden test.
- The shortcut candidates implement exactly the failure modes the chapter lists: hardcoding answers to the public examples, plus one shallow heuristic per tricky task (first/last-character match for palindromes, digital root for digit sums) that passes the visible tests while missing the real requirement.
- The fixes mirror the chapter's prescriptions: evaluate on tests the agent cannot modify (the `hidden` regime), and broaden test coverage with freshly randomized inputs (the `randomized` regime).

For context on how the same test-reward idea scales to real repositories, see the [supplementary reading on Meta SWE-RL](/chapter20_swe_agents/meta-swe-rl) (GRPO with a binary test reward) and [20.3 Self-Play SWE-RL](/chapter20_swe_agents/self-play-ssr-and-summary) (what happens when even the tasks must be generated). The multi-step agent loop that this lab deliberately compresses into a single decision is covered in [19.1 Agentic RL Basics](/chapter19_agentic/overview).

## 20.L.3 Setup and how to run

No installation is needed beyond the project virtualenv (Python 3.12, torch, numpy, matplotlib -- all used offline). From the repository root:

```bash
# Smoke run: 4 tasks, 12 epochs, all three reward regimes (< 60 s on CPU)
.venv/bin/python code/chapter20_swe_agents/train.py --smoke --seed 0 --device cpu

# Full run: 14 tasks, 300 epochs, all three reward regimes (~8 min on CPU)
.venv/bin/python code/chapter20_swe_agents/train.py --seed 0 --device cpu
```

Useful options: `--reward {all,visible,hidden,randomized}` trains one regime only, `--epochs` and `--lr` control the REINFORCE loop, `--n-random` sets how many fresh inputs each task draws per epoch in the randomized regime, `--timeout` sets the sandbox wall-clock limit per test batch, and `--max-tasks` truncates the task list. Everything is seeded (`--seed`); reruns are reproducible.

Artifacts land in `runs/chapter20_swe_agents/` (gitignored): `metrics_<mode>.csv` per-epoch histories, `summary.json` with the config and final numbers, `train_curves.png` and `visible_vs_hidden.png` plots, plus a summary table printed to stdout. The test suite also runs the smoke configuration end to end:

```bash
.venv/bin/pytest code/chapter20_swe_agents -q
```

## 20.L.4 Walkthrough of the key code

**The verifier** ([sandbox.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter20_swe_agents/sandbox.py)) writes the candidate source and the test cases into a fresh temporary directory and executes a small driver there as a subprocess with a timeout. Timeouts and child crashes are flagged separately from test failures, following the chapter's rule that infrastructure errors must not be counted as ability failures:

```python
proc = subprocess.run(
    [sys.executable, "driver.py"],
    cwd=tmpdir,          # the sandbox: relative paths land here, never in the repo
    capture_output=True,
    text=True,
    timeout=timeout,     # whole batch shares one wall-clock budget
)
```

**The policy and update** ([policy.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter20_swe_agents/policy.py)) keep one logit row per task, $\pi(a \mid t) = \mathrm{softmax}(\theta[t,:])_a$, and apply REINFORCE with a per-task running baseline $b[t]$ that depends on the task but never on the sampled action:

```python
advantages = rewards.to(log_probs.device) - baselines.to(log_probs.device)
loss = -(advantages.detach() * log_probs).mean()
```

**The three reward regimes** ([train.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter20_swe_agents/train.py)) differ only in which verifier answers the reward query. Deterministic suites are executed once and cached (the candidates are fixed, so caching changes no value); randomized inputs are executed live every epoch because the inputs change:

```python
if mode == "visible":
    r = rates[(task.task_id, action, "visible")]
elif mode == "hidden":
    r = rates[(task.task_id, action, "hidden")]
else:
    r = randomized_rate(task, action, rng, n_random, timeout)
```

The task suite itself ([tasks.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter20_swe_agents/tasks.py)) holds 14 single-function tasks (`fib`, `factorial`, `reverse_words`, `is_palindrome`, `gcd`, `sum_squares`, `count_vowels`, `fizzbuzz`, `clamp`, `unique_sorted`, `binary_search`, `second_largest`, `collatz`, `digit_sum`), each with visible tests, disjoint hidden tests, four candidates (correct / two memorizers-or-heuristics / one plausible-but-wrong), plus a reference oracle and a random-input generator per task. A static audit plus a contract test guarantee the design: every correct candidate scores 1.0 on both suites, every shortcut scores 1.0 on visible and strictly below 1.0 on hidden.

## 20.L.5 Results from a real run

Full configuration, seed 0 on CPU (461 s wall time). `E[.]` denotes the exact expectation under the learned stochastic policy; `resolved` is the fraction of tasks whose greedy pick passes all hidden tests; `P(hack)` is the mean policy mass on memorizing/heuristic shortcuts.

| mode       | train reward | E[visible] | E[hidden] | greedy hidden | resolved | P(hack) |
| ---------- | ------------ | ---------- | --------- | ------------- | -------- | ------- |
| visible    | 1.000        | 0.999      | 0.491     | 0.554         | 0.500    | 0.537   |
| hidden     | 0.982        | 0.945      | 0.980     | 0.982         | 0.929    | 0.002   |
| randomized | 1.000        | 0.976      | 0.998     | 1.000         | 1.000    | 0.002   |

Learning dynamics (train reward / E[hidden] / P(hack) at selected epochs):

| epoch | visible               | hidden                | randomized            |
| ----- | --------------------- | --------------------- | --------------------- |
| 0     | 0.881 / 0.395 / 0.500 | 0.321 / 0.414 / 0.488 | 0.357 / 0.405 / 0.512 |
| 9     | 0.940 / 0.398 / 0.536 | 0.589 / 0.684 / 0.251 | 0.548 / 0.575 / 0.376 |
| 49    | 0.976 / 0.452 / 0.578 | 0.964 / 0.950 / 0.013 | 0.988 / 0.887 / 0.109 |
| 149   | 1.000 / 0.472 / 0.557 | 0.964 / 0.960 / 0.005 | 1.000 / 0.992 / 0.015 |
| 299   | 1.000 / 0.491 / 0.537 | 0.982 / 0.980 / 0.002 | 1.000 / 0.998 / 0.002 |

The visible-only regime reaches a perfect train reward by epoch 149 while its hidden pass rate never rises above 0.5: the policy has learned the verifier, not the tasks. Both the hidden-test regime and the randomized-input regime drive hack mass to ~0.00 and resolve nearly every task. Randomized inputs need more epochs than hidden tests (compare epoch 49: 0.887 vs 0.950) because the reward is noisier, but converge to the same optimum.

## 20.L.6 What to look for and why it happens

- **Train reward saturates in all three regimes, but means different things.** In the visible regime the reward stops distinguishing candidates once the policy concentrates on the three visible-optimal arms (correct, memorizer, heuristic) -- the gradient vanishes because every sampled reward equals the baseline. Saturation is a property of the verifier's coverage, not of task mastery.
- **The hacked policy is a near-even mix, not a corner solution.** P(hack) settles around 0.54 rather than 1.0 because REINFORCE has no reason to prefer one optimal arm over another; early sampling noise decides each task. Expect small seed-to-seed movement in the visible row (hidden ~0.4-0.5), which is itself the lesson: under-specified rewards leave the final behavior to chance.
- **Wrong candidates provide the early dense signal.** Plausible-but-wrong arms earn partial visible credit (e.g. the halving heuristic scores 2/3 on the Collatz visible tests), which is why train reward climbs steeply in the first 10 epochs in every regime before the policy sorts optimal arms from merely decent ones.
- **Hidden and randomized rewards agree because they measure the same thing.** Both make the correct candidate the unique optimum -- one by withholding the grading tests, the other by generating inputs memorization cannot anticipate. Their final rows match (E[hidden] 0.980 vs 0.998), which is the toy version of the chapter's claim that coverage, not reward cleverness, is what defeats gaming.

## 20.L.7 Extension exercises

1. **Add a task.** Add a 15th task (e.g. anagrams, or merging two sorted lists) with visible tests, disjoint hidden tests, and four candidates following the file's contract. Hint: pick hidden inputs that break the most natural memorizer fallback, and let `test_candidate_reward_contract` verify your design -- it runs the sandbox for you.
2. **Binary resolve reward.** Change the train reward to the chapter's $R_{\text{resolved}}$ (1.0 only if every test in the suite passes). Hint: partial-credit arms now all score 0, so the signal is sparser; try raising `--epochs` and compare the epoch-9 row against the table above.
3. **Shaping-term ablation.** Add the chapter's trajectory-cost term, e.g. $R = R_{\text{test}} - \lambda \cdot (\text{lines of code} / 20)$, and sweep $\lambda$. Hint: memorizers are shorter than correct solutions, so a large $\lambda$ makes hacks optimal even under hidden rewards -- find the $\lambda$ where the hidden regime flips, and explain why this is a verifier-design bug rather than an RL bug.
4. **Baseline comparison.** Replace the running baseline with the batch mean, then with no baseline at all, logging the variance of the advantages across epochs. Hint: keep the learning rate fixed; the no-baseline run should show noisier early updates and slower ARM sorting while converging to the same visible-regime fixed point.
5. **A tiny world model.** Following [20.2](/chapter20_swe_agents/world-model-and-deep-swe), fit a predictor of sandbox pass rates from cheap features (candidate length, keyword flags) on the cached grid, train the policy against predicted rewards for 300 epochs, then evaluate against the real sandbox. Hint: report sandbox calls saved versus the final E[hidden] gap -- the gap is the world-model deviation the chapter warns about.
6. **Two-step composition.** Replace "pick one candidate" with two sequential picks (e.g. pick a helper, then a main function) where the reward is granted only for the pair. Hint: the action space becomes the product set; watch the epoch-9 train reward -- credit assignment over the pair is harder, which is the toy analogue of the long-horizon difficulty in [20.2](/chapter20_swe_agents/world-model-and-deep-swe).
