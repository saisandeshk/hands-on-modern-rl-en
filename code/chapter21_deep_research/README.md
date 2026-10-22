# Chapter 21 Lab: Multi-Turn Research Agent with Delayed Reward

A hands-on lab for the browser-RL harness (Section 21.1) and research-agent
evaluation (Section 21.2): an offline toy "web" of synthetic documents
related by shared names, labs, and titles, multi-hop questions, and a
tool-calling agent trained with REINFORCE.

## What the lab shows

- A repeatable harness: `search(term)` / `open(doc)` / `answer(candidate)`
  episodes with a step budget and a verifiable terminal reward.
- A learned policy (small MLP + value baseline, REINFORCE from scratch)
  compared against a random policy and a scripted heuristic.
- Credit assignment: sparse trajectory-level reward versus a dense shaped
  reward for opening relevant documents — and how a loose shaping proxy
  (bonus for opening _any_ document) gets gamed.
- Evaluation split by question difficulty (2-document chains with or without
  a same-name cue), reporting success rate, average steps, and
  search/open call counts.

## Files

- `web_env.py` — synthetic corpus generator and `TinyResearchEnv`.
- `policies.py` — feature encoding, `ActorCriticMLP`, `RandomPolicy`,
  `HeuristicPolicy`, discounted-return helper.
- `train_research_agent.py` — behavior-cloning warm-start, REINFORCE
  fine-tuning under three reward regimes plus a from-scratch control,
  evaluation, plots, and the summary table.
- `test_smoke.py` — unit tests and the smoke-run test.

## How to run

Smoke mode (tiny corpus, ~15 s on CPU):

```bash
.venv/bin/python code/chapter21_deep_research/train_research_agent.py --smoke --seed 0 --device cpu
```

Full run (~3-7 minutes on CPU, all output under `runs/chapter21_deep_research/`):

```bash
.venv/bin/python code/chapter21_deep_research/train_research_agent.py --seed 0 --device cpu
```

Useful options: `--episodes`, `--hidden`, `--lr`, `--gamma`, `--step-cost`,
`--rel-bonus`, `--any-bonus`, `--max-steps`, `--batch`, `--bc-epochs`,
`--device {auto,cpu,cuda,mps}`, `--outdir`.

Tests (CPU only):

```bash
.venv/bin/pytest code/chapter21_deep_research -q
```

## Expected results

Full run, seed 0 (representative; exact numbers vary with seed):

- Scripted heuristic: ~100% success, ~5 steps (2 searches, 2 opens).
- REINFORCE from a BC warm-start: high success (roughly 0.9-1.0) under the
  outcome-only and relevant-shaping regimes.
- REINFORCE from scratch (outcome reward only): at the chance guessing rate
  (1/8 ≈ 0.12), answering almost immediately — sparse terminal reward cannot
  discover a 5-step tool chain unaided.
- Loose `any`-document shaping (bonus 0.4): trajectories get longer (more
  searches/opens) while success stalls or drops — the policy harvests
  open bonuses instead of answering. Tight relevant shaping keeps
  trajectories short and success high.

See `docs/chapter21_deep_research/lab.md` for the walkthrough of one full run
with its real numbers.
