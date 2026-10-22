# Chapter 18 Lab: Rollout/Learner Systems and Policy Staleness

A small actor-learner system on `CartPole-v1` that mirrors, at textbook scale,
the industrial rollout/training split from Chapter 18
([18.1](../docs/chapter18_llm_rl_industrial/single-machine-to-industrial.md),
[18.4](../docs/chapter18_llm_rl_industrial/distributed-sync.md)):

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
   `--mode vtrace`, cf. AReaL's clipped stale weights in 18.4.4).

## How to run

From the repository root (outputs go to `runs/chapter18_llm_rl_industrial/`,
which is gitignored):

```bash
# Smoke test: tiny config, finishes in well under 60 s on CPU
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment all --smoke

# Full lab run (~5-10 min on a laptop CPU)
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment all --seed 0

# Individual experiments / options
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment throughput --workers 1,2,4,8
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment staleness --modes sync,none,clip,vtrace --delay 4
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment sweep --delays 0,1,2,4,8
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment staleness --env LunarLander-v3
```

Useful flags: `--seed`, `--device {auto,cpu,cuda}`, `--updates`,
`--steps-per-update`, `--broadcast-every`, `--delay`, `--clip-eps`,
`--c-bar`, `--rho-bar`, `--eval-episodes`. `python rollout_system.py --help`
lists everything.

Expected runtime: smoke < 60 s; full run finishes in at most ~15 min on CPU
(throughput < 1 min, staleness curves a few minutes, sweep a few minutes).

## Expected results

- Throughput rises sublinearly with K (spawn/broadcast overhead is reported as
  `startup_s`, ~2.4 s at 1 worker growing to ~4.1 s at 8 in the reference run;
  steady-state collection went from ~6240 to ~14092 steps/s); learner
  utilization is low in absolute terms because the synthetic update workload is
  tiny — read the trend across K, not the level. This is the
  colocated-vs-disaggregated trade-off of 18.4 in miniature.
- With a broadcast delay of a few versions, all async variants learn far slower
  than the sync baseline (reference run, delay 4: sync 118.67 vs async 9–22
  greedy return). PPO-style clipping and V-trace-style truncation help where
  ratios actually spread — e.g. sweep delay 8: clip 46.00 vs none 12.33 — but
  at small delays single-seed noise dominates the corrected-vs-uncorrected
  ordering, so compare across delays rather than over-interpreting one row.
  Mean importance ratio stays near 1 for sync and drifts for stale batches.
- The sweep shows final return degrading with delay; the short sweep budget
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
