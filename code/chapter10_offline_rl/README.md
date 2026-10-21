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
   DQN (fixed-data Bellman MSE with `max` over actions), and discrete CQL
   (naive DQN plus `alpha * (logsumexp_a Q(s,a) − Q(s,a_data))`, Section 10.1).
   Policies are evaluated with real rollouts only for measurement.
3. **`ope.py`** — off-policy evaluation of the frozen expert target policy
   from each behaviour dataset: per-decision IS (unbiased, high variance),
   self-normalized weighted IS (biased, lower variance) and fitted Q
   evaluation (Bellman regression with no `max` and no ratios), compared
   against the true online value. Reports effective sample size (ESS) so poor
   coverage is visible.

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
| random  | 21.9      | 9.1 ± 0.8   | 332.9 ± 76.8 | 241.6 ± 69.4  |
| medium  | 203.2     | 500.0 ± 0.0 | 479.5 ± 46.3 | 500.0 ± 0.0   |
| expert  | 490.9     | 500.0 ± 0.0 | 9.3 ± 0.6    | 500.0 ± 0.0   |
| mixed   | 233.3     | 500.0 ± 0.0 | 500.0 ± 0.0  | 500.0 ± 0.0   |

Read it as: BC copies (and denoises) the behaviour policy but cannot exceed
it on random data; naive offline DQN is fine under broad coverage
(random/medium/mixed) yet collapses on the narrow expert dataset (9.3),
where the `max` keeps querying poorly covered actions — mean max-Q on
held-out states ends at 34.7 vs CQL's 26.9; CQL matches or beats every
baseline on every dataset.

OPE of the expert target (true discounted value 93.03, gamma=0.99):

| behaviour | PDIS  | WIS   | FQE   | ESS (% of episodes) |
| --------- | ----- | ----- | ----- | ------------------- |
| random    | 27.71 | 18.37 | 92.23 | 0.0%                |
| medium    | 35.74 | 31.89 | 86.46 | 4.9%                |
| expert    | 78.84 | 90.91 | 86.34 | 5.1%                |
| mixed     | 47.24 | 90.17 | 85.69 | 2.4%                |

Read it as: IS estimates are accurate only under good coverage (WIS ≈ 91 on
expert/mixed) and degenerate otherwise (ESS ≈ 0% on random, both IS variants
collapse toward the behaviour value); FQE stays within ~7 points of truth on
_all four_ datasets because it never uses importance ratios.
