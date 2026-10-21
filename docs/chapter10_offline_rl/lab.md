---
title: '10.L Lab: Offline RL from Fixed Datasets'
---

# 10.L Lab: Offline RL from Fixed Datasets

In [10.1](./offline-data-distribution-shift) we diagnosed why offline RL is hard: the Bellman `max` queries actions the dataset never saw, the network extrapolates arbitrarily there, and the error is amplified by $1/(1-\gamma)$. In this lab you reproduce that failure with your own hands, fix it with the discrete version of the CQL penalty from Section 10.1, and then evaluate a policy you are not allowed to run, using the off-policy evaluation tools every offline project needs.

All code is linked below as `code/chapter10_offline_rl/<file>` on GitHub; run it from the repository root with the project virtualenv.

## What you will learn

- How to build fixed offline datasets of controlled quality (random, medium, expert, mixed) from a single online DQN on CartPole-v1, and what "coverage" looks like in practice.
- Why behaviour cloning copies the behaviour policy (including denoising it), and why it cannot exceed the data.
- How naive offline DQN can look healthy on broad datasets yet collapse on a narrow expert dataset, and how the mean $\max_a Q(s,a)$ curve exposes the overestimation.
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

**Step 1 — datasets** ([collect_datasets.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter10_offline_rl/collect_datasets.py)). A small online DQN trains for 500 episodes with epsilon-greedy exploration; the best checkpoint seen during training is kept (online evaluation during online training is legitimate — the offline ban starts at collection). That frozen Q-network then defines three behaviour policies: uniform random, eps=0.3 (medium), and eps=0.02 (expert); a fourth `mixed` dataset round-robins episodes across the three. Each transition stores the behaviour action probabilities alongside $(s, a, r, s')$, because OPE needs the ratio $\rho = \pi_e/\pi_b$ later.

**Step 2 — offline learners** ([train_offline.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter10_offline_rl/train_offline.py)). All three methods train on fixed batches with no environment interaction. Behaviour cloning is plain cross-entropy against dataset actions. Naive offline DQN uses the familiar target $y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$ — the same `max` that Section 10.1 blames for extrapolation error. CQL adds one function:

```python
logsumexp = torch.logsumexp(q_values, dim=1)          # soft max over ALL actions
q_data = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
penalty = (alpha * (logsumexp - q_data)).mean()       # push OOD down, data up
```

The `logsumexp` term can only shrink by pushing down Q on every action, while the $-Q(s, a_{\text{data}})$ term protects actions seen in the data. The gap between them is exactly the conservative margin from Section 10.1.

**Step 3 — OPE** ([ope.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter10_offline_rl/ope.py)). The target is the frozen expert (smoothed with 5% uniform so all ratios stay finite). Per-decision IS reweights each reward only by ratios seen so far, $\sum_t \gamma^t r_t \prod_{k \le t} \rho_k$; weighted IS self-normalizes full-trajectory weights, $\sum_i w_i G_i / \sum_i w_i$; FQE regresses $y = r + \gamma Q_{\bar{}}(s', \pi_e(s'))$ with no `max` and no ratios, then averages over initial states. The script also reports the effective sample size $(\sum w)^2/\sum w^2$ — the single number that tells you whether to trust the IS columns at all.

## Results from the reference run

Full configuration, `--seed 0 --device cpu` (collect ~50 s, train ~2 min, OPE ~45 s). Dataset returns observed during collection: random 21.9, medium 203.2, expert 490.9, mixed 233.3 (CartPole-v1 caps at 500).

**Offline learning** (each cell: mean ± std over 20 online eval episodes; training itself never touched the environment):

| dataset | behaviour | BC          | naive DQN    | CQL          |
| ------- | --------- | ----------- | ------------ | ------------ |
| random  | 21.9      | 9.1 ± 0.8   | 332.9 ± 76.8 | 241.6 ± 69.4 |
| medium  | 203.2     | 500.0 ± 0.0 | 479.5 ± 46.3 | 500.0 ± 0.0  |
| expert  | 490.9     | 500.0 ± 0.0 | 9.3 ± 0.6    | 500.0 ± 0.0  |
| mixed   | 233.3     | 500.0 ± 0.0 | 500.0 ± 0.0  | 500.0 ± 0.0  |

**Off-policy evaluation** of the expert target (true discounted value from 50 online rollouts: 93.03; $\gamma = 0.99$; dataset MC is the plain behaviour average):

| behaviour | PDIS  | WIS   | FQE   | ESS (% of episodes) | dataset MC |
| --------- | ----- | ----- | ----- | ------------------- | ---------- |
| random    | 27.71 | 18.37 | 92.23 | 0.0%                | 19.24      |
| medium    | 35.74 | 31.89 | 86.46 | 4.9%                | 61.17      |
| expert    | 78.84 | 90.91 | 86.34 | 5.1%                | 98.03      |
| mixed     | 47.24 | 90.17 | 85.69 | 2.4%                | 58.23      |

## What to look for and why it happens

**BC denoises the behaviour policy.** On medium data BC scores 500 while the behaviour averages 203: cloning with argmax at test time strips the eps=0.3 exploration noise and recovers the underlying greedy policy. But BC can never exceed what the data's best actions show — on random data it scores 9.1, faithfully copying noise. This is the ceiling Section 10.1 attributes to pure imitation.

**Naive DQN collapses exactly where coverage is narrowest.** On the expert dataset it scores 9.3 — worse than random play — while CQL on identical data scores 500. The expert data visits a thin slice of state space (101 long episodes of balanced-pole states); every Bellman `max` then queries actions in states the network barely trained on, and the error compounds through bootstrapping. On broad datasets (random, mixed) the same algorithm reaches 333–500, because the `max` usually lands on actions the data actually covers. The `q_curves.png` plot makes this visible: on the expert dataset the naive DQN's mean $\max_a Q$ on held-out states ends at 34.7 versus CQL's 26.9, drifting upward late in training while CQL stays flat. Coverage, not dataset "quality", decides whether the `max` is safe — the central claim of Section 10.1.

**CQL is the only method that is never bad.** It matches or beats both baselines on all four datasets (242–500). The penalty costs a little on random data (241.6 vs naive DQN's luckier 332.9) and buys total immunity on expert data. That asymmetry — small price on broad data, rescue on narrow data — is why conservatism is the default in real offline projects.

**IS estimates die with coverage; FQE survives.** From the random dataset, both PDIS (27.71) and WIS (18.37) collapse onto the behaviour value (19.24) with an ESS of 0.0% of episodes — a handful of huge weights dominate, so the estimate is the dataset, not the target. From expert and mixed data, WIS reaches 90–91, within 3 points of truth. FQE lands at 86–92 from _every_ dataset: no ratios means no variance explosion, at the price of function-approximation bias (it undershoots truth by ~7 points everywhere, the cost of bootstrapping on finite data). When your ESS rounds to zero, believe FQE over IS — and ideally collect better coverage before deploying.

## Extension exercises

1. **Sweep the conservatism.** Re-run training with `--cql-alpha 0.1` and `--cql-alpha 5.0`. At which value does CQL start behaving like naive DQN (collapse on expert) or like BC (degraded on random)? _Hint: watch the `cql_max_q` curve and the penalty term, not just the final return._
2. **Shrink the expert dataset.** Collect with `--num-transitions 5000` and compare naive DQN on the small vs full expert data. _Hint: fewer transitions means thinner coverage, so the collapse should arrive earlier and the max-Q curve should separate from CQL's sooner._
3. **Break WIS on purpose.** In `ope.py`, evaluate a fully greedy target (`TARGET_EPS = 0.0`) from the random dataset and explain the NaNs or zeros. _Hint: what happens to $\rho = \pi_e/\pi_b$ when $\pi_b$ puts zero mass where $\pi_e$ acts?_
4. **Horizon stress test.** Change `GAMMA` in `ope.py` to 0.9 and to 0.999 and re-run. Which estimator moves most, and why? _Hint: longer horizons multiply more ratios into each weight; check the ESS column._
5. **Close the loop offline-to-online.** Fine-tune the CQL-on-random policy with 20 online DQN episodes (à la AWAC's offline-to-online regime in Section 10.1) and compare against 20 episodes from scratch. _Hint: reuse `expert_q.pt` as the Q initialization and keep epsilon low._
6. **Try expectile thinking.** Replace the CQL penalty with a one-line change that only trains on dataset actions but weights the Bellman loss by whether the TD error is positive (a crude expectile tilt toward good actions, in the spirit of IQL from Section 10.1). Does it beat naive DQN on expert data? _Hint: start from `train_q_offline` with `alpha=0` and scale each sample's squared error by `tau` or `1-tau`._
