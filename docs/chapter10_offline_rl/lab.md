---
title: 10.4 Lab: Offline RL from Fixed Datasets
---

# 10.4 Lab: Offline RL from Fixed Datasets

In [10.1](./offline-data-distribution-shift) we diagnosed why offline RL is hard: the Bellman `max` queries actions the dataset never saw, the network extrapolates arbitrarily there, and the error is amplified by $1/(1-\gamma)$. In this lab you reproduce that failure with your own hands, fix it with the discrete version of the CQL penalty from Section 10.1, and then evaluate a policy you are not allowed to run, using the off-policy evaluation tools every offline project needs.

All code is linked below as `code/chapter10_offline_rl/<file>` on GitHub; run it from the repository root with the project virtualenv.

## What you will learn

- How to build fixed offline datasets of controlled quality (random, medium, expert, mixed) from a single online DQN on CartPole-v1, and what "coverage" looks like in practice.
- Why behaviour cloning copies the behaviour policy (including denoising it), and why it cannot exceed the data.
- How naive offline DQN can look healthy on broad random data yet collapse as coverage narrows (medium, mixed, expert), and how the mean $\max_a Q(s,a)$ curve on dataset states exposes the overestimation.
- How discrete CQL's penalty $\alpha(\log\sum_a e^{Q(s,a)} - Q(s,a_{\text{data}}))$ keeps Q-values bounded and recovers expert performance.
- How per-decision IS, weighted (self-normalized) IS, and FQE estimate a target policy's value from behaviour data only, and how poor coverage shows up as a collapsed effective sample size.

## Background

Read [10.1](./offline-data-distribution-shift) first, especially Sections 1 (where extrapolation error comes from), 2.2 (the CQL regularizer), and the comparison of BC-style constraints in Section 3. The lab implements the discrete-action special case of the CQL loss:

$$\mathcal{L}_{\text{CQL}}(Q) = \alpha \left(\mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_a \exp(Q(s, a))\right] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[Q(s, a)]\right) + \mathcal{L}_{\text{Bellman}}(Q)$$

with $\alpha = 1.0$ by default. Because CartPole has only two actions, the $\log\sum_a\exp$ term is exact rather than sampled, which makes this lab a clean small-scale check of the idea before you meet sampled variants in continuous control. The OPE step previews a theme that returns in later chapters: before deploying a policy learned from fixed data, estimate its value from that same data, and distrust the estimate when coverage is poor.

## Setup and how to run

You need the project virtualenv (Python 3.12, torch, gymnasium, matplotlib) and about 5 minutes on a laptop CPU. Outputs (CSV/JSON metrics and PNG plots) go to `runs/chapter10_offline_rl/`, which is gitignored.

```bash
# Smoke run: tiny configuration, under ~60 s on CPU. Start here.
.venv/bin/python code/chapter10_offline_rl/collect_datasets.py --smoke --seed 0 --device cpu
.venv/bin/python code/chapter10_offline_rl/train_offline.py --smoke --seed 0 --device cpu
.venv/bin/python code/chapter10_offline_rl/ope.py --smoke --seed 0 --device cpu

# Full run: the configuration behind every number reported below.
.venv/bin/python code/chapter10_offline_rl/collect_datasets.py --seed 0 --device cpu
.venv/bin/python code/chapter10_offline_rl/train_offline.py --seed 0 --device cpu
.venv/bin/python code/chapter10_offline_rl/ope.py --seed 0 --device cpu

# Tests: hand-computed unit checks plus all three smoke runs on CPU.
.venv/bin/pytest code/chapter10_offline_rl -q
```

Useful options: `--dataset {random,medium,expert,mixed,all}` selects which dataset to train on; `--behaviour {...}` selects which dataset to evaluate from; `--cql-alpha` sets the conservatism; `--device {auto,cpu,cuda}` picks the torch device. Every script prints a summary table to stdout, and every run is bit-reproducible on CPU for a fixed `--seed` (single-threaded deterministic execution, all environment action spaces seeded).

## Walkthrough of the key code

**Step 1 — datasets** ([collect_datasets.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter10_offline_rl/collect_datasets.py)). A small online DQN trains for 500 episodes with epsilon-greedy exploration; the best checkpoint seen during training is kept (online evaluation during online training is legitimate — the offline ban starts at collection). Note the selection takes the maximum over a handful of noisy 5-episode evaluations, so the kept "expert" is optimistically biased by construction; treat its 500.0 as a selected checkpoint, not an unbiased estimate. That frozen Q-network then defines three behaviour policies: uniform random, eps=0.3 (medium), and eps=0.02 (expert); a fourth `mixed` dataset round-robins episodes across the three. Each transition stores the behaviour action probabilities alongside $(s, a, r, s')$, because OPE needs the ratio $\rho = \pi_e/\pi_b$ later.

**Step 2 — offline learners** ([train_offline.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter10_offline_rl/train_offline.py)). All three methods train on fixed batches with no environment interaction. Behaviour cloning is plain cross-entropy against dataset actions. Naive offline DQN uses the familiar target $y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$ — the same `max` that Section 10.1 blames for extrapolation error. The target network syncs every 100 gradient steps (`--target-sync-every`), frequent enough that Bellman backups propagate across the 500-step horizon instead of stalling after ~30 steps. CQL adds one function:

```python
logsumexp = torch.logsumexp(q_values, dim=1)          # soft max over ALL actions
q_data = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
penalty = (alpha * (logsumexp - q_data)).mean()       # push OOD down, data up
```

The `logsumexp` term can only shrink by pushing down Q on every action, while the $-Q(s, a_{\text{data}})$ term protects actions seen in the data. The gap between them is exactly the conservative margin from Section 10.1. Each (dataset, method) pair trains from its own derived seed, so re-running with `--dataset expert` alone reproduces the expert row of the full `--dataset all` run. Trained weights are saved as `offline/<dataset>_<method>_q.pt` (for example `offline/random_cql_q.pt`), so extension exercises can load and fine-tune them.

**Step 3 — OPE** ([ope.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter10_offline_rl/ope.py)). The target is the frozen expert (smoothed with 5% uniform so all ratios stay finite). Per-decision IS reweights each reward only by ratios seen so far, $\sum_t \gamma^t r_t \prod_{k \le t} \rho_k$; weighted IS self-normalizes full-trajectory weights, $\sum_i w_i G_i / \sum_i w_i$; FQE regresses $y = r + \gamma \mathbb{E}_{a' \sim \pi_e}[Q_{\bar{}}(s', a')]$ — the expectation under the _smoothed_ target, matching the policy whose value the IS columns estimate — with no `max` and no ratios, then averages over the dataset episodes' initial states (every CartPole episode starts from `env.reset()`, so this matches the reset distribution). The script also reports the effective sample size $(\sum w)^2/\sum w^2$ — the single number that tells you whether to trust the IS columns at all.

## Results from the reference run

Full configuration, `--seed 0 --device cpu` (collect ~50 s, train ~2 min, OPE ~45 s). Dataset returns observed during collection: random 21.9, medium 203.2, expert 490.9, mixed 233.3 (CartPole-v1 caps at 500).

**Offline learning** (each cell: mean ± std over 20 online eval episodes; training itself never touched the environment):

| dataset | behaviour | BC          | naive DQN    | CQL         |
| ------- | --------- | ----------- | ------------ | ----------- |
| random  | 21.9      | 42.1 ± 8.8  | 500.0 ± 0.0  | 111.8 ± 3.1 |
| medium  | 203.2     | 500.0 ± 0.0 | 99.1 ± 3.5   | 500.0 ± 0.0 |
| expert  | 490.9     | 500.0 ± 0.0 | 9.3 ± 0.6    | 500.0 ± 0.0 |
| mixed   | 233.3     | 500.0 ± 0.0 | 238.4 ± 14.5 | 500.0 ± 0.0 |

**Off-policy evaluation** of the expert target (true discounted value from 50 online rollouts: 93.03; $\gamma = 0.99$; dataset MC is the discounted behaviour average):

| behaviour | PDIS  | WIS   | FQE   | ESS (% of episodes) | dataset MC |
| --------- | ----- | ----- | ----- | ------------------- | ---------- |
| random    | 27.71 | 18.37 | 86.21 | 0.0%                | 19.24      |
| medium    | 35.74 | 31.89 | 85.08 | 4.9%                | 61.17      |
| expert    | 78.84 | 90.91 | 86.13 | 5.1%                | 98.03      |
| mixed     | 47.24 | 90.17 | 84.35 | 2.4%                | 58.23      |

## What to look for and why it happens

**BC copies the behaviour distribution, then argmax denoises it.** On medium data BC scores 500 while the behaviour averages 203: cloning with argmax at test time strips the eps=0.3 exploration noise and recovers the underlying greedy policy. But BC can never exceed what the data's best actions show. On random data the cloned softmax is essentially uniform (mean entropy 0.6929 nats vs the maximum $\log 2 = 0.6931$; agreement with the dataset actions 0.502), so BC did reproduce the behaviour distribution faithfully. What changes the score is the evaluation: `evaluate_greedy` takes the argmax of the logits, turning the copied stochastic policy into a deterministic one. That deterministic policy scores 42.1 — above the behaviour's own 21.9 by luck of the draw, but nowhere near solving the task. This is the ceiling Section 10.1 attributes to pure imitation: copying a bad policy cannot create a good one.

**Naive DQN collapses as coverage narrows.** On the expert dataset it scores 9.3 — worse than random play — while CQL on identical data scores 500. The gradient across datasets tells the story: random (broadest coverage) 500.0, mixed 238.4, medium 99.1, expert (narrowest) 9.3. The expert data visits a thin slice of state space (long episodes of balanced-pole states); every Bellman `max` then queries actions in states the network barely trained on, and the error compounds through bootstrapping. The `q_curves.png` plot makes this visible: on the expert dataset the naive DQN's mean $\max_a Q$ on the first 2000 dataset states explodes to about $2.2 \times 10^6$ while CQL stays at 81.1, near the true discounted scale ($\sum_{t=0}^{499} 0.99^t = 99.34$ for a 500-step episode). Coverage, not dataset "quality", decides whether the `max` is safe — the central claim of Section 10.1.

**CQL is the only method that is never catastrophic.** It matches or beats both baselines on medium/expert/mixed (500 everywhere), and loses only on random data (111.8 vs naive DQN's 500.0). The penalty costs something on broad data and buys total rescue on narrow data. That asymmetry — small price on broad data, rescue on narrow data — is why conservatism is the default in real offline projects. (It is not free: 111.8 is the second-worst cell in the table, so "never catastrophic" should not be read as "always best".)

**IS estimates die with coverage; FQE survives.** From the random dataset, WIS (18.37) sits near the discounted behaviour average (19.24) with an ESS of 0.0% of episodes — a handful of huge weights dominate, so the estimate is the dataset, not the target. (PDIS at 27.71 is 44% above the behaviour average: it is unbiased, so read that gap as one high-variance realisation, not a collapse onto the behaviour.) From expert and mixed data, WIS reaches 90–91, within 3 points of truth. FQE lands at 84–86 from _every_ dataset: no ratios means no variance explosion, at the price of function-approximation bias (it undershoots truth by 6.8/8.0/6.9/8.7 points on random/medium/expert/mixed, the cost of bootstrapping on finite data). When your ESS rounds to zero, believe FQE over IS — and ideally collect better coverage before deploying.

## Extension exercises

1. **Sweep the conservatism.** Re-run training with `--cql-alpha 0.1` and `--cql-alpha 5.0`. At which value does CQL start behaving like naive DQN (collapse on expert) or like BC (degraded on random)? _Hint: watch the `cql_max_q` curve and the penalty term, not just the final return._
2. **Shrink the expert dataset.** Collect into a separate directory with `--num-transitions 5000 --outdir runs/ch10_small` (a fresh `--outdir` keeps the full 50k datasets intact for comparison), then train naive DQN on the small expert data and compare its max-Q curve against the full-data run. _Hint: thinner coverage means a weaker Bellman signal, so watch how fast the naive-DQN max-Q curve separates from CQL's; note the full-data collapse is already total (9.3/500), so the testable prediction is in the curves, not the final return._
3. **Break WIS on purpose.** In `ope.py`, evaluate a fully greedy target (`TARGET_EPS = 0.0`) from the random dataset and explain the NaNs or zeros. _Hint: with a fully greedy target the ratios are 0 or 2; since each episode contains at least one zero-ratio step, every trajectory weight is 0, WIS becomes 0/0 = NaN and the ESS collapses to 0, while PDIS stays finite because it only needs a nonzero prefix product._
4. **Horizon stress test.** Change `GAMMA` in `ope.py` to 0.9 and to 0.999 and re-run. Which estimator moves most, and why? _Hint: the estimand's scale changes with $\gamma$ (about 10x smaller at 0.9, 4x larger at 0.999), so all three estimators move together; the ESS column is unaffected because the weights contain no discount factor — check which estimator's relative error grows with the effective horizon._
5. **Close the loop offline-to-online.** `train_offline.py` saves each trained network as `offline/<dataset>_<method>_q.pt`. Load `offline/random_cql_q.pt` and fine-tune it with 20 online DQN episodes (à la AWAC's offline-to-online regime in Section 10.1), and compare against 20 episodes from scratch. _Hint: initialise the online Q-network from the saved CQL weights and keep epsilon low so the comparison measures fine-tuning, not relearning._
6. **Try expectile thinking.** Replace the CQL penalty with a one-line change that only trains on dataset actions but weights the Bellman loss by whether the TD error is positive (a crude expectile tilt toward good actions, in the spirit of IQL from Section 10.1). Does it beat naive DQN on expert data? _Hint: start from `train_q_offline` with `alpha=0` and scale each sample's squared error by `tau` or `1-tau`._
