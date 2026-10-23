# Chapter 20 Lab: RL for a Toy Code Agent with Executable Verifiers

A hands-on miniature of SWE-RL training (Chapter 20): a softmax policy picks
among fixed candidate implementations for each of 14 small Python function
tasks, and the reward is the test pass rate measured by actually executing
the candidate in a subprocess sandbox (temporary working directory, wall-clock
timeout, per-test pass/fail).

What the lab shows:

1. **Executable verifiers as rewards.** `sandbox.py` runs each candidate
   against a test suite in a subprocess and returns the pass fraction --
   the toy analogue of the SWE-bench evaluator / RLVR test reward.
2. **REINFORCE with a baseline.** `policy.py` implements a per-task softmax
   policy and the update `loss = -mean((R - b[t]) * log pi(a|t))` with a
   running per-task baseline `b[t]`.
3. **Reward hacking.** Under visible-tests-only rewards the policy converges
   to shortcut candidates that memorize the visible tests: train pass rate
   1.000 but hidden pass rate ~0.49, with ~0.54 probability mass on hacks.
4. **The fix.** Training on held-out hidden tests -- or on freshly randomized
   test inputs with oracle-computed expected outputs -- recovers the correct
   candidates (hidden pass rate ~0.98-1.00, hack mass ~0.00).

## How to run

Smoke (under ~60 s on CPU; 4 tasks, 12 epochs, all three reward regimes):

```bash
.venv/bin/python code/chapter20_swe_agents/train.py --smoke --seed 0 --device cpu
```

Full run (~8 min on CPU; 14 tasks, 300 epochs):

```bash
.venv/bin/python code/chapter20_swe_agents/train.py --seed 0 --device cpu
```

Useful options: `--reward {all,visible,hidden,randomized}` selects which
regime(s) to train (`all` runs all three), `--epochs`, `--lr`, `--n-random` (fresh random inputs per task/epoch),
`--timeout` (sandbox seconds per test batch), `--max-tasks`, `--outdir`.

Outputs go to `runs/chapter20_swe_agents/` (gitignored): `metrics_<mode>.csv`
per-epoch histories, `summary.json`, `train_curves.png` and
`visible_vs_hidden.png` plots, plus a summary table printed to stdout.

## Tests

```bash
.venv/bin/pytest code/chapter20_swe_agents -q
```

This checks the sandboxed runner (pass/fail, exception isolation, timeout,
temp-dir working directory), the REINFORCE loss against a hand-computed
value, the candidate-design contract (correct = 1.0 on both suites, hacks =
1.0 visible / < 1.0 hidden, verified by real sandbox runs), and an
end-to-end smoke training run in a subprocess.

## Expected results (full run, seed 0, CPU, 461 s wall time)

| mode       | train_R | E[visible] | E[hidden] | greedy hidden | resolved | P(hack) |
| ---------- | ------- | ---------- | --------- | ------------- | -------- | ------- |
| visible    | 1.000   | 0.999      | 0.491     | 0.554         | 0.500    | 0.537   |
| hidden     | 0.982   | 0.945      | 0.980     | 0.982         | 0.929    | 0.002   |
| randomized | 1.000   | 0.976      | 0.998     | 1.000         | 1.000    | 0.002   |

`E[...]` is the exact expectation under the learned stochastic policy;
`resolved` is the fraction of tasks whose greedy pick passes every hidden
test; `P(hack)` is the mean policy mass on memorizing/heuristic shortcuts.
Expect small seed-to-seed variation in the visible row (any mix of the three
visible-optimal arms is a fixed point), but the gap -- train ~1.0 with hidden
~0.4-0.5 versus hidden/randomized ~1.0 -- is robust.

## Files

- `tasks.py` -- 14 tasks (visible/hidden tests, candidates, oracles, random
  input generators).
- `sandbox.py` -- subprocess test runner with timeout and cwd isolation.
- `policy.py` -- softmax bandit policy + REINFORCE-with-baseline update.
- `train.py` -- experiment CLI (REINFORCE under three reward regimes).
- `test_smoke.py` -- pytest suite described above.
