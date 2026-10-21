# Chapter 10 lab: offline RL and OPE from fixed CartPole datasets

Hands-on companion to Chapter 10 (offline RL). Everything runs on
`CartPole-v1` (discrete actions) with small MLPs implemented from scratch in
PyTorch — no RL libraries. The lab has three steps:

1. **`collect_datasets.py`** — trains one small online DQN (the only place
   that may interact with the environment to _improve_ a policy), freezes it,
   and rolls out behaviour policies of different quality:
   `random`, `medium` (trained Q, eps=0.3), `expert` (trained Q, eps=0.02)
   and `mixed`. Each fixed dataset is stored as `.npz` plus a `_meta.json`
   sidecar, including the per-step behaviour probabilities needed for
   importance sampling.
2. **`train_offline.py`** — from each fixed dataset, with _zero_ environment
   interaction: behaviour cloning (supervised cross-entropy), naive offline
   DQN (fixed-data Bellman MSE with `max` over actions; target sync every 100
   steps via `--target-sync-every`), and discrete CQL
   (naive DQN plus `alpha * (logsumexp_a Q(s,a) − Q(s,a_data))`, Section 10.1).
   Policies are evaluated with real rollouts only for measurement. Each
   (dataset, method) pair uses its own derived seed, so per-dataset runs
   reproduce the full-run rows; trained weights are saved as
   `offline/<dataset>_<method>_q.pt`.
3. **`ope.py`** — off-policy evaluation of the frozen expert target policy
   (5%-smoothed) from each behaviour dataset: per-decision IS (unbiased, high
   variance), self-normalized weighted IS (biased, lower variance) and fitted Q
   evaluation (Bellman regression against the smoothed-target expectation,
   no `max`, no ratios; FQE averages over the dataset episodes' initial
   states), compared against the true online value. Reports effective sample
   size (ESS) so poor coverage is visible.

Shared plumbing (seeding, device selection, MLPs, dataset save/load, online
evaluation) lives in `common.py`. Unit tests plus subprocess smoke runs of
every script live in `test_smoke.py`.

## How to run

From the repository root, using the project virtualenv:

```bash
# Smoke run: tiny configuration, finishes in under ~60 s on CPU.
.venv/bin/python code/chapter10_offline_rl/collect_datasets.py --smoke --seed 0 --device cpu
.venv/bin/python code/chapter10_offline_rl/train_offline.py --smoke --seed 0 --device cpu
.venv/bin/python code/chapter10_offline_rl/ope.py --smoke --seed 0 --device cpu

# Full run: the configuration behind the numbers in the docs lab page.
.venv/bin/python code/chapter10_offline_rl/collect_datasets.py --seed 0 --device cpu
.venv/bin/python code/chapter10_offline_rl/train_offline.py --seed 0 --device cpu
.venv/bin/python code/chapter10_offline_rl/ope.py --seed 0 --device cpu

# Tests (unit checks + all three smoke runs on CPU).
.venv/bin/pytest code/chapter10_offline_rl -q
```

Useful flags: `--dataset {random,medium,expert,mixed,all}` (train step),
`--behaviour {...}` (OPE step), `--cql-alpha`, `--fqe-steps`,
`--device {auto,cpu,cuda}`, `--outdir` (default
`runs/chapter10_offline_rl/`). All scripts print a summary table to stdout
and write CSV/JSON metrics plus PNG plots (matplotlib `Agg` backend) under
the output directory. Every script accepts `--seed`; runs are
bit-reproducible on CPU (single-threaded deterministic execution, all
environment action spaces seeded).

## Expected runtime (CPU)

Measured on a laptop-class CPU, `--seed 0 --device cpu`:

| Step                                                              | Full config | Smoke config |
| ----------------------------------------------------------------- | ----------- | ------------ |
| `collect_datasets.py` (500 online episodes + 4 × 50k transitions) | ~50 s       | < 5 s        |
| `train_offline.py` (4 datasets × 3 methods × 15k steps)           | ~2 min      | < 30 s       |
| `ope.py` (FQE 20k steps × 4 + 50 online episodes)                 | ~45 s       | < 15 s       |

Each script finishes well within 15 minutes on CPU; GPU (`--device cuda`)
is faster but unnecessary.

## Expected results (`--seed 0`, CPU)

Dataset returns observed during collection: random 21.9, medium 203.2,
expert 490.9, mixed 233.3 (CartPole-v1 caps at 500).

Offline training (20 eval episodes each):

| dataset | behaviour | BC          | naive DQN    | CQL (alpha=1) |
| ------- | --------- | ----------- | ------------ | ------------- |
| random  | 21.9      | 42.1 ± 8.8  | 500.0 ± 0.0  | 111.8 ± 3.1   |
| medium  | 203.2     | 500.0 ± 0.0 | 99.1 ± 3.5   | 500.0 ± 0.0   |
| expert  | 490.9     | 500.0 ± 0.0 | 9.3 ± 0.6    | 500.0 ± 0.0   |
| mixed   | 233.3     | 500.0 ± 0.0 | 238.4 ± 14.5 | 500.0 ± 0.0   |

Read it as: BC copies (and argmax denoises) the behaviour policy but cannot exceed
it on random data (the cloned softmax is near-uniform; the 42.1 is the luck of the
deterministic argmax policy, not a solution); naive offline DQN degrades as coverage
narrows (500.0 → 238.4 → 99.1 → 9.3 from random to expert), collapsing on the narrow
expert dataset (9.3), where the `max` keeps querying poorly covered actions — mean
max-Q on the first 2000 dataset states explodes to ~2.2M vs CQL's 81.1; CQL is never
catastrophic (500 on medium/expert/mixed) at the price of some conservatism on random
data (111.8).

OPE of the expert target (true discounted value 93.03, gamma=0.99):

| behaviour | PDIS  | WIS   | FQE   | ESS (% of episodes) |
| --------- | ----- | ----- | ----- | ------------------- |
| random    | 27.71 | 18.37 | 86.21 | 0.0%                |
| medium    | 35.74 | 31.89 | 85.08 | 4.9%                |
| expert    | 78.84 | 90.91 | 86.13 | 5.1%                |
| mixed     | 47.24 | 90.17 | 84.35 | 2.4%                |

Read it as: IS estimates are accurate only under good coverage (WIS ≈ 91 on
expert/mixed) and degenerate otherwise (ESS ≈ 0% on random; WIS sits near the
discounted behaviour average while PDIS shows a high-variance realisation
above it); FQE stays within ~7-9 points of truth on _all four_ datasets
(undershoots of 6.8/8.0/6.9/8.7) because it never uses importance ratios.
