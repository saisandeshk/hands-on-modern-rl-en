---
title: '18.6 Lab: Rollout Workers, Learner Updates, and Policy Staleness'
---

# 18.6 Lab: Rollout Workers, Learner Updates, and Policy Staleness

> **Section goal**: Build a small actor-learner system with rollout workers and a learner, measure how throughput scales with the number of workers, and see how stale behaviour policies distort learning unless importance weights are truncated.

> **Code**: [rollout_system.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter18_llm_rl_industrial/rollout_system.py) · [test_smoke.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter18_llm_rl_industrial/test_smoke.py) · [README](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter18_llm_rl_industrial/README.md)

## What you will learn

- How a rollout/learner split works: K workers collect trajectories with a copy of the policy, send them through a queue, and a learner updates the policy and broadcasts new weights every M updates.
- How to measure **throughput** (environment steps per second) and **learner utilization**, and why both change as workers are added.
- What **policy lag** is: each batch records the behaviour-policy version that generated it, so you can report exactly how many versions stale every update is.
- Why stale data without correction learns worse, and where two truncated importance weights help: PPO-style ratio clipping and V-trace-style two-level truncation.
- How this maps to the chapter's systems language: colocated vs disaggregated deployment, synchronous vs asynchronous scheduling, and the throughput/staleness trade-off.

## Background

Two chapter sections carry this lab:

- [18.1 From Single-Machine Experiments to Industrial Training](/chapter18_llm_rl_industrial/single-machine-to-industrial) introduces synchronous vs asynchronous training, the staleness ratio $\rho_t^{\text{stale}} = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{gen}}}(a_t \mid s_t)$, and why long tool-calling trajectories force asynchronous designs.
- [18.4 Distributed RL Training](/chapter18_llm_rl_industrial/distributed-sync) covers training-inference mismatch (record the true behaviour log-probs, clip extreme ratios), veRL's ResourcePool/Worker/Driver placement, generation-throughput techniques, and the LlamaRL / AReaL / AgentRL comparison — including AReaL's token-level weights clipped to $[0.8, 1.2]$.

The lab shrinks that stack to one CPU: CartPole-v1 replaces LLM rollouts, a two-layer MLP replaces the Actor, and Python multiprocessing replaces the GPU pools. The systems questions stay the same: who generates, who updates, how versions move, and what stale data costs.

Notation matches the chapter: states $s$, actions $a$, policy $\pi_\theta$, behaviour policy $\pi_{\theta_{\text{gen}}}$, staleness ratio $\rho_t^{\text{stale}}$, advantage $A_t$, return $G_t$.

## Setup and how to run

Install the chapter requirements and run from the repository root:

```bash
pip install -r code/chapter18_llm_rl_industrial/requirements.txt
# Smoke test: all three experiments, finishes in well under 60 s on CPU
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment all --smoke
# Full lab run (a few minutes on a laptop CPU; the reference timing below
# took 231 s wall-clock, up to ~15 min on machines with fewer cores)
.venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment all --seed 0
```

Useful flags: `--experiment {throughput,staleness,sweep,all}`, `--workers 1,2,4,8`, `--updates`, `--steps-per-update`, `--modes sync,none,clip,vtrace`, `--delay`, `--broadcast-every`, `--clip-eps`, `--c-bar`, `--rho-bar`, `--env CartPole-v1`, `--seed`, `--device {auto,cpu,cuda}`, `--smoke`. Run with `--help` for the full list.

All outputs (CSV metrics, `summary.json`, PNG learning curves) go to `runs/chapter18_llm_rl_industrial/`, which is gitignored. Each run prints a short summary table to stdout.

Tests:

```bash
.venv/bin/pytest code/chapter18_llm_rl_industrial -q
```

## Walkthrough of the key code

**1. Rollouts record the behaviour log-probs.** `collect_batch` steps the environment with the workers' policy copy and stores $\log \pi_{\theta_{\text{gen}}}(a_t \mid s_t)$ next to every transition — the rollout-time record the training-inference mismatch discussion in 18.4 insists on (never substitute a trainer-side recomputation for the true behaviour probability):

```python
a, lp, _ = policy.act(obs)   # sampled from the behaviour copy
...
beh_lps.append(lp)           # stored with the transition
```

**2. The staleness ratio and its two truncations.** The learner recomputes $\log \pi_\theta$ for the same $(s, a)$ pairs and forms $\rho = \exp(\text{new\_lp} - \text{beh\_lp})$. PPO-style clipping bounds it to $[1-\epsilon, 1+\epsilon]$; V-trace-style truncation applies two caps, `c_bar` on the policy term and `rho_bar` on the value target:

```python
def vtrace_truncated_weights(new_logprobs, beh_logprobs, c_bar=1.0, rho_bar=1.0):
    rho = torch.exp(new_logprobs - beh_logprobs)
    c = torch.minimum(torch.full_like(rho, c_bar), rho)
    rho_tr = torch.minimum(torch.full_like(rho, rho_bar), rho)
    return c, rho_tr
```

`--mode none` ignores the ratio entirely (plain policy gradient on stale data); `--mode clip` uses the PPO-clip loss against the stored behaviour log-probs; `--mode vtrace` weights the policy term by `c` and corrects the value target with `rho_tr`. Note this is a simplified one-step Monte-Carlo analogue of V-trace: the full algorithm applies the truncation to bootstrapped TD residuals, while here the target is `V(s) + rho_tr * (G - V(s))` with Monte-Carlo returns.

**3. Version bookkeeping.** Weights are broadcast every M updates and arrive D updates late, so at learner update `t` the workers hold:

```python
def behaviour_version_at(update, broadcast_every, delay):
    t = update - delay
    if t < 0:
        return 0
    return (t // broadcast_every) * broadcast_every
```

Every batch carries its `beh_version`; `LagTracker` records `learner_version - beh_version` per update and reports mean/max lag — the same per-trajectory version metadata AReaL attaches in the asynchronous-scheduling comparison in 18.4.

**4. Real multiprocessing for throughput.** In `run_throughput`, each worker process loads the broadcast weights once, collects a step budget, and puts a single shard on the queue; the parent (learner) drains the queue and performs a real gradient update per shard. Startup (spawn → all workers ready) is timed separately from the steady-state window, so `steps_per_sec` excludes spawn cost by construction. No weight broadcast happens inside the throughput run — the scaling number isolates collection parallelism (and CPU contention between worker processes). The staleness curves use a deterministic single-process simulation of the identical `(version, data)` queue discipline, so they are exactly reproducible per seed.

## Results from a full run

The tables below come from an actual run of the full (non-smoke) configuration on
a laptop CPU (`--experiment all --seed 0`, CartPole-v1, 60 updates of 2048 steps
for the staleness curves, 25 updates of 1024 steps per sweep point):

```bash
timeout 900 .venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment all --seed 0
```

A timed repeat of this exact command took 231 s wall-clock on a laptop CPU.
Throughput timings vary from run to run (multiprocessing scheduling); the
staleness and sweep learning curves are deterministic per seed and reproduced
exactly across repeats on the same machine. Reference environment: torch
2.14.1, gymnasium 1.4.0, numpy 2.5.4 — the action-sampling stream
(`torch.distributions.Categorical.sample`) is not guaranteed stable across
torch releases, so exact episode returns may differ on other versions even
at the same seed.

**Throughput (steady-state; startup excluded).** Collection used 8000
environment steps per worker; startup is the spawn-to-all-ready time:

| workers | total_steps | startup_s | steady_s | steps_per_sec | learner_util | mean_ep_return |
| ------- | ----------- | --------- | -------- | ------------- | ------------ | -------------- |
| 1       | 8000        | 2.369     | 1.282    | 6240          | 0.188        | 21.85          |
| 2       | 16000       | 2.226     | 2.720    | 5883          | 0.074        | 21.62          |
| 4       | 32000       | 3.582     | 4.876    | 6562          | 0.085        | 21.87          |
| 8       | 64000       | 4.064     | 4.542    | 14092         | 0.164        | 21.41          |

Read it as: steady-state collection throughput roughly doubles from 1 to 8
workers (6240 to 14092 steps/s), but the steady-state window excludes
spawn/startup by construction (`steps_per_sec` is timed from the moment the
last worker reports ready), so startup cost cannot explain the shortfall —
the cause is per-worker CPU contention. The per-worker rate falls from 6240
steps/s at K=1 to 14092/8 ≈ 1761 steps/s at K=8: 8× the workers buy only
~2.26× the throughput. Learner utilization (`busy / steady_s`) stays low
here (0.07–0.19) because the synthetic update workload is deliberately tiny
next to collection; with exactly one shard (and hence one learner update)
per worker, updates-per-collected-step is constant in K, so there is no
saturation knee to find in this design — utilization is roughly flat and
non-monotone (highest at K=1), tracking `t_update / t_worker` rather than
rising with K.

**Staleness curves (60 updates × 2048 steps, delay 4, broadcast every update).**
`final_return` is the mean greedy return over 3 evaluation episodes:

| mode   | delay | final_return | mean_lag | max_lag | mean_rho |
| ------ | ----- | ------------ | -------- | ------- | -------- |
| sync   | 0     | 118.67       | 0.000    | 0       | 0.999    |
| none   | 4     | 22.00        | 3.833    | 4       | 1.009    |
| clip   | 4     | 13.67        | 3.833    | 4       | 1.001    |
| vtrace | 4     | 9.33         | 3.833    | 4       | 1.056    |

The honest reading: at delay 4, _all_ async variants collapse far below the
sync baseline (118.67 vs 9–22), and on this single seed the truncated variants
do not beat uncorrected replay. With ratios hovering near 1.0 (`mean_rho`),
clipping rarely binds, so the gap is dominated by single-seed CartPole noise
and the short training budget rather than by the correction doing harm — do not
read the clip-vs-none ordering at one delay as a general ranking.

**Staleness sweep (25 updates × 1024 steps per point; shorter budget, so lower
absolute returns).** Clipping helps at some delays and ties at others:

| mode | delay | final_return | mean_lag |
| ---- | ----- | ------------ | -------- |
| none | 0     | 9.33         | 0.00     |
| clip | 0     | 19.00        | 0.00     |
| none | 1     | 9.67         | 0.96     |
| clip | 1     | 26.67        | 0.96     |
| none | 2     | 9.67         | 1.88     |
| clip | 2     | 9.33         | 1.88     |
| none | 4     | 9.33         | 3.60     |
| clip | 4     | 9.33         | 3.60     |
| none | 8     | 12.33        | 6.56     |
| clip | 8     | 46.00        | 6.56     |

Note the delay-8 row: with very stale data, PPO-style clipping (46.00) clearly
beats uncorrected replay (12.33) — the one place in this budget where the
truncation has room to matter, because ratios spread wider and the bound binds.
The `mean_lag` column confirms the bookkeeping: lag grows exactly as
`broadcast_every` and `delay` predict (about 3.8 versions at delay 4 over 60
updates, 6.56 at delay 8 over 25 updates).

## What to look for and why

- **Throughput rises sublinearly — because workers contend for cores, not because of spawn cost.** Each extra worker adds collection capacity, but the steady-state window excludes startup by construction and no broadcast happens inside the throughput run, so the sublinear number isolates parallel-collection efficiency: per-worker rate falls with K while total throughput still climbs. This is the small-scale shadow of 18.4's colocated-vs-disaggregated choice: separate rollout capacity helps until core contention and queue handling become the bottleneck.
- **Learner utilization does not rise with K in this design.** Because each worker sends exactly one shard, the learner performs exactly K updates over K×8000 collected steps for every K — updates-per-collected-step is constant, the queue never backs up, and utilization is `t_update / t_worker`: roughly flat and non-monotone (highest at K=1 in the reference run), falling when workers contend for cores. In the reference run the synthetic update workload is deliberately tiny, so utilization stays low in absolute terms (0.07–0.19) — read the flat shape, not the level. A real saturation knee would need streaming workers (many shards each) so the queue can stay full; here there is none to find, which is itself the lesson about what has to be true for utilization to saturate.
- **Uncorrected staleness biases the policy toward the past.** `--mode none` replays old behaviour as if it were fresh, so improvements the learner already made get overwritten by gradients from data generated before they existed. The learning curve lags the sync baseline and plateaus lower — the failure mode 18.1 warns about for long agent trajectories.
- **Truncation helps when ratios actually spread.** In the full run above, ratios
  stay near 1.0 at delay 4, so clipping rarely binds and all async variants
  score far below sync — the correction has nothing to bite on. It is at delay
  8 in the sweep (clip 46.00 vs none 12.33) that truncation visibly helps: very
  stale batches spread the ratios wider, the bound binds, and off-policy steps
  are muted while on-policy-like steps still contribute. Compare `mean_rho`
  across runs: near 1.0 for sync, drifting for stale batches.
- **Lag does not compound into a monotone return curve at this budget.** In the sweep, each extra version of delay shifts the behaviour distribution further from the learner and the measured `mean_lag` grows exactly as `broadcast_every` and `delay` predict (about 3.8 versions at delay 4 over 60 updates, 6.56 at delay 8 over 25 updates) — the bookkeeping is exact. But final returns do not degrade monotonically with delay: most sweep points sit near the random-policy level and `clip` scores its best value (46.00) at the largest delay. The throughput/staleness trade-off is still the right lens (delaying broadcasts buys rollout throughput at the price of training on older data), but at 25 updates × 1024 steps the data say budget, not delay, dominates the level — compare shapes across seeds before reading a monotone story into one row.

## Exercises

1. **Vary the broadcast interval.** Run `--broadcast-every 4 --delay 0` against `--broadcast-every 1 --delay 3`. Both produce stale batches — do they produce the _same_ lag distribution and the same learning curve? _Hint: compare the per-update `lag` column in `staleness_curves.csv`, not just the mean._
2. **Tighten and loosen the clip.** Re-run the delay-4 `clip` curve with `--clip-eps 0.1` and `--clip-eps 0.4`. Which degrades faster, and why does too-tight clipping also hurt? _Hint: track the per-update `clip_frac` column in `staleness_curves.csv` (and `mean_clip_frac` in the stdout table) — a bound that always binds discards signal along with staleness._
3. **Split the V-trace caps.** Set `--c-bar 1.0 --rho-bar 0.5` vs `--c-bar 0.5 --rho-bar 1.0` at delay 8. Which cap matters more for the actor, which for the critic? _Hint: read the `vtrace` branch of `learner_update` and predict before running._
4. **Why there is no saturation knee here.** Run throughput with `--workers 1` and `--workers 8` and compare `learner_util` against `steps_per_sec`. Why doesn't utilization rise with K in this design, and what would have to change (think: how many shards each worker sends) for the queue to stay full and utilization to saturate? _Hint: compute updates-per-collected-step for each K from the `throughput.csv` row — it is constant by construction._
5. **Break the behaviour-logprob record.** Deliberately replace the stored `beh_logprobs` with recomputed log-probs from a perturbed policy copy, then re-run `clip` at delay 4. How does the curve change? _Hint: this reproduces the 18.4 training-inference mismatch — the ratio is wrong before any update begins, and clipping cannot repair it._
6. **Try LunarLander-v3.** Run `--env LunarLander-v3 --smoke` first (needs the Box2D extra, already in this chapter's `requirements.txt`), then a short full comparison. Does staleness hurt more or less than on CartPole, and what property of the environment explains it? _Hint: think about episode length and reward sparsity._
