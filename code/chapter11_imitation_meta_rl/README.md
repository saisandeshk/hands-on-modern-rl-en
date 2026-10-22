# Chapter 11 lab: imitation learning — behaviour cloning versus DAgger

Hands-on companion to Chapter 11 (Imitation, inverse and meta-RL), Sections
11.1–11.2. Environment: `CartPole-v1`. Everything is implemented from
scratch with numpy/torch (no RL libraries); the only learned components are
small MLPs trained with supervised losses.

## What the lab shows

1. **Expert.** A hand-written deterministic PD controller
   (`expert_act` in `bc_dagger_cartpole.py`: push toward the lean,
   damp the angular velocity, centre the cart) scores **500/500** on every
   seed. Because it is a function of state, it can also label any state
   DAgger visits — the "interactive expert" of Section 11.1.3.
2. **Behaviour cloning and compounding error.** BC minimises
   `-E[log pi(a|s)]` over N expert trajectories. With N=1 the learner
   agrees with the expert on ~82% of expert states yet scores only ~378
   (and just 133 on one seed): a small per-step error snowballs once the
   learner leaves the expert state distribution (Section 11.1.2,
   `O(T^2 eps)`).
3. **DAgger closes the gap.** Starting from the same 1-trajectory BC
   policy, DAgger rolls out the learner, labels the visited states with
   the expert, aggregates and retrains. Mean return goes
   378 → 484 → 500 in two iterations while using only **one** full expert
   trajectory (plus on-policy corrections), instead of the two full
   trajectories BC needs for reliable 500s.
4. **GAIL-style discriminator reward (small demo).**
   `gail_discriminator_cartpole.py` trains D(s, a) to tell expert pairs
   from a weak learner's pairs and reports the implicit reward
   `-log(1-D)` (Section 11.2.3). Only the supervised discriminator half
   is implemented; the adversarial policy-optimisation half is left out
   (see "Why no full GAIL" below).

## How to run

From the repository root (project virtualenv, everything offline, CPU fine):

```bash
# Full lab (~8 min on a laptop CPU)
.venv/bin/python code/chapter11_imitation_meta_rl/bc_dagger_cartpole.py --device cpu
.venv/bin/python code/chapter11_imitation_meta_rl/gail_discriminator_cartpole.py --device cpu

# Tiny smoke versions (<60 s each, for a quick check)
.venv/bin/python code/chapter11_imitation_meta_rl/bc_dagger_cartpole.py --smoke
.venv/bin/python code/chapter11_imitation_meta_rl/gail_discriminator_cartpole.py --smoke

# Tests (smoke runs of both scripts + unit tests, all on CPU)
.venv/bin/pytest code/chapter11_imitation_meta_rl -q
```

Useful flags: `--seed N` (base seed; seeds N, N+1, N+2 are used),
`--device {auto,cpu,cuda}`, `--outdir <dir>`.

## Expected runtime

| Script (full config)                                                     | CPU time |
| ------------------------------------------------------------------------ | -------- |
| `bc_dagger_cartpole.py` (3 seeds; BC N in {1,2,5,10,25}; DAgger 5 iters) | ~8 min   |
| `gail_discriminator_cartpole.py`                                         | <10 s    |
| `pytest` (smoke + unit tests)                                            | ~1 min   |

## Expected results

`bc_dagger_cartpole.py` writes `metrics.json`, `metrics.csv` and
`bc_dagger_results.png` under `runs/chapter11_imitation_meta_rl/`
(gitignored) and prints a summary table. Reference run
(`--seed 0`, CPU, 472 s):

| Expert trajs (BC)          | 1     | 2     | 5     | 10    | 25    |
| -------------------------- | ----- | ----- | ----- | ----- | ----- |
| Mean return                | 377.7 | 500.0 | 500.0 | 500.0 | 500.0 |
| Agreement on expert states | 0.823 | 0.935 | 0.988 | 0.991 | 0.996 |
| Agreement on own states    | 0.766 | 0.780 | 0.966 | 0.989 | 0.994 |

DAgger from a 1-trajectory start (mean labels, mean return):
iter0 (500, 377.7) → iter1 (1365, 483.8) → iter2+ (2365+, 500.0).
Notably, seed 1 goes 133.1 → 451.4 → 500.0: a failing BC policy is
repaired by two rounds of on-policy corrections.

`gail_discriminator_cartpole.py` writes `gail_metrics.json` and
`gail_discriminator_reward.png`. Reference run: discriminator accuracy
0.739 (expert pairs) / 0.819 (learner pairs); mean implicit reward
`-log(1-D)` 3.451 on expert pairs versus 0.502 on learner pairs —
expert-visited pairs score high, learner-only pairs score near zero,
as in Section 11.2.3.2.

## Why no full GAIL

A complete GAIL loop (REINFORCE/PPO against the learned discriminator
reward from scratch) was prototyped and **left out**: on this CPU budget
it is unreliable — a from-scratch REINFORCE policy on the discriminator
reward stalls below return 100 after 40 iterations, consistent with the
known GAN-style instability and poor sample efficiency of GAIL
(Section 11.2.4.1; the literature uses millions of environment steps).
The discriminator-reward demo above trains reliably in seconds and shows
the core idea (occupancy matching via a classifier) without the unstable
inner RL loop.
