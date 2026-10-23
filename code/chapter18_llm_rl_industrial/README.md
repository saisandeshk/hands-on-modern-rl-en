# Chapter 18 Lab: Rollout/Learner Systems and Policy Staleness

A small actor-learner system on `CartPole-v1` that mirrors, at textbook scale,
the industrial rollout/training split from Chapter 18
([18.1](../../docs/chapter18_llm_rl_industrial/single-machine-to-industrial.md),
[18.4](../../docs/chapter18_llm_rl_industrial/distributed-sync.md)):

- **K rollout workers** hold a copy of the policy, collect trajectories, and
  send them through a queue. The `throughput` experiment uses real Python
  `multiprocessing` workers; the `staleness`/`sweep` experiments use a
  deterministic single-process simulation of the same queue system (same
  `(version, data)` pairs a real queue would deliver, without process
  scheduling noise, so curves are reproducible).
- **A learner** updates the policy (PPO-style clipped objective from scratch,
  no RL libraries) and broadcasts new weights every M updates
  (`--broadcast-every`), with an optional broadcast delay (`--delay`).

What the lab shows:

1. **Throughput vs workers** — env steps/second and learner utilization for
   K = 1, 2, 4, 8 (startup/spawn cost reported separately from steady state).
2. **Policy lag** — every batch records the behaviour-policy version that
   generated it; the learner reports the mean/max lag in versions.
3. **Corrections under staleness** — sync on-policy baseline vs async with no
   correction vs async with truncated importance weights (PPO-style ratio
   clipping `--mode clip`, or V-trace-style two-level truncation
   `--mode vtrace`, cf. AReaL's clipped stale weights in 18.4, "Comparing
   Three Asynchronous Frameworks").

## How to run

From the repository root (outputs go to `runs/chapter18_llm_rl_industrial/`,
which is gitignored):

```bash
# Smoke test: tiny config, finishes in well under 60 s on CPU
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment all --smoke

# Full lab run (~4-15 min on a laptop CPU depending on core count; the
# reference timing below took 231 s wall-clock)
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment all --seed 0

# Individual experiments / options
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment throughput --workers 1,2,4,8
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment staleness --modes sync,none,clip,vtrace --delay 4
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment sweep --delays 0,1,2,4,8
# LunarLander needs the Box2D extra, included in this chapter's requirements.txt
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment staleness --env LunarLander-v3
```

Useful flags: `--seed`, `--device {auto,cpu,cuda}`, `--updates`,
`--steps-per-update`, `--broadcast-every`, `--delay`, `--clip-eps`,
`--c-bar`, `--rho-bar`, `--eval-episodes`. `python rollout_system.py --help`
lists everything.

Expected runtime: smoke < 60 s; full run a few minutes on a laptop CPU
(up to ~15 min on machines with fewer cores; throughput < 1 min,
staleness curves a few minutes, sweep a few minutes).

## Expected results

- Throughput rises sublinearly with K: the steady-state window excludes
  spawn/startup (reported separately as `startup_s`), and no weight
  broadcast happens inside the throughput run, so the sublinear scaling
  comes from per-worker CPU contention — in the reference run the
  per-worker rate fell from ~6240 steps/s (K=1) to ~1761 steps/s (K=8),
  buying only ~2.26x total throughput (~6240 to ~14092 steps/s) for 8x
  the workers. Learner utilization (`busy / steady_s`) stays low in
  absolute terms because the synthetic update workload is tiny and each
  worker sends exactly one shard, so updates-per-collected-step never
  changes with K — expect a roughly flat, non-monotone utilization
  curve, not a saturation knee. This is the
  colocated-vs-disaggregated trade-off of 18.4 in miniature.
- With a broadcast delay of a few versions, all async variants learn far slower
  than the sync baseline (reference run, delay 4: sync 118.67 vs async 9–22
  greedy return). PPO-style clipping and V-trace-style truncation help where
  ratios actually spread — e.g. sweep delay 8: clip 46.00 vs none 12.33 — but
  at small delays single-seed noise dominates the corrected-vs-uncorrected
  ordering, so compare across delays rather than over-interpreting one row.
  Mean importance ratio stays near 1 for sync and drifts for stale batches.
- The sweep does not show a monotone degrade-with-delay curve at this
  budget: most points sit near the random-policy level and the ordering
  is not monotone in delay (in the reference run, `clip` scores its best,
  46.00, at the largest delay, 8). The short sweep budget
  (25 updates × 1024 steps) keeps absolute returns low, which is itself a
  lesson about training budget vs staleness.

## Tests

```bash
.venv/bin/pytest code/chapter18_llm_rl_industrial -q
```

Unit tests check the truncated importance weights against hand-computed
values, the version-lag bookkeeping, determinism, and that a 1-worker
zero-delay run equals the synchronous update on the same data. The smoke
tests run every experiment (`throughput`, `staleness`, `sweep`) in `--smoke`
mode as subprocesses and assert the CSV/PNG/JSON outputs are written.

## Files

- `rollout_system.py` — the whole lab: policy net, rollout collection,
  learner updates (sync / none / clip / vtrace), version-lag bookkeeping,
  multiprocessing throughput benchmark, deterministic async simulation, CLI.
- `test_smoke.py` — unit + subprocess smoke tests.
- `requirements.txt` — minimal deps (gymnasium, torch, numpy, matplotlib).
