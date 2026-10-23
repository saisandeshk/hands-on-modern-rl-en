# Lab: Test-Time Compute on a Tiny Reasoner (Chapter 16)

A 73k-parameter character-level transformer learns two-digit addition written
as an explicit carry trace (for example `23+58=` →
`3+8=11>1c1;2+5+1=8>8c0;=81`), and we spend extra compute **at inference
time** — more samples, voting, verifier selection — instead of training a
bigger model. This is the Chapter 16 story in miniature: sampling failures
(§16.1) are fixed with inference compute, a rule-based verifier supplies the
reward signal (§16.2), and parallel sampling follows the coverage formula
$1-(1-p)^N$ (§16.3) until a noisy verifier breaks the selection step.

## What the lab shows

1. Single-sample accuracy falls as sampling temperature rises (greedy is
   best for one shot; temperature buys the diversity that voting needs).
2. Test-time scaling at fixed temperature: the pass@k estimator, majority
   vote (self-consistency) and best-of-N with an exact verifier, N = 1…64.
   Majority vote saturates well below 100% while verifier best-of-N keeps
   climbing toward full coverage — the §16.3 prediction.
3. A noisy verifier (false positives at a fixed rate) makes best-of-N
   plateau: more candidates stop helping once false accepts dominate.
4. One rejection-sampling round (fine-tune on self-generated,
   verifier-accepted traces mixed with teacher traces) lifts single-sample
   accuracy.

## How to run

From the repository root, using the project virtualenv (everything runs
offline; no downloads):

```bash
# Smoke test: full pipeline on a tiny config (<60 s on CPU)
.venv/bin/python code/chapter16_reasoning/run_lab.py --smoke --device cpu

# Full lab (~30 s on CPU; GPU optional via --device cuda)
.venv/bin/python code/chapter16_reasoning/run_lab.py --device cpu

# Unit tests + smoke subprocess test
.venv/bin/pytest code/chapter16_reasoning -q
```

Useful flags: `--seed`, `--device {auto,cpu,cuda}`, `--epochs`,
`--n-train`, `--n-test`, `--n-samples`, `--temps 0.0,0.7`,
`--fp-rates 0.1,0.3`, `--rft-samples`, `--no-rft` (skip stage 4),
`--outdir`. Outputs (CSV/JSON metrics and PNG plots with the `Agg`
backend) go to `runs/chapter16_reasoning/` (gitignored); a summary table
is printed to stdout.

## Expected runtime

| Configuration | What it does                                      | CPU time      |
| ------------- | ------------------------------------------------- | ------------- |
| `--smoke`     | 256 train / 2 epochs, 24 test, 8 samples          | a few seconds |
| full default  | 2000 train / 30 epochs, 256 test, 64 samples, RFT | ~30 s         |

## Expected results (seed 0, CPU)

Single-sample accuracy drops with temperature: 0.727 (greedy) → 0.664
(T=0.3) → 0.512 (T=0.7) → 0.320 (T=1.0) → 0.113 (T=1.3). At T=0.7 the
N-scaling looks like this (256 test problems, pool rate p≈0.52):

| N   | pass@k | majority | best-of-N exact | best-of-N FP=0.1 | best-of-N FP=0.3 |
| --- | ------ | -------- | --------------- | ---------------- | ---------------- |
| 1   | 0.47   | 0.52     | 0.52            | 0.52             | 0.52             |
| 4   | 0.83   | 0.54     | 0.85            | 0.79             | 0.72             |
| 16  | 0.98   | 0.66     | 0.98            | 0.85             | 0.75             |
| 64  | 1.00   | 0.71     | 1.00            | 0.86             | 0.75             |

Rejection sampling (4000 kept self-traces + 2000 teacher traces, 2 epochs
at 0.2× lr): greedy 0.727 → 0.754, T=0.7 accuracy 0.512 → 0.605.
Training the kept traces _without_ the teacher mix and at full lr hurts
instead (0.727 → 0.594 in our trial) — pure self-training overfits the
already-solved problems and forgets the rest.

## Files

- `arith_cot.py` — trace format, tokenizer, splits, tiny transformer,
  teacher-forced training, and the §16.3 metrics (unbiased pass@k,
  majority vote, exact/noisy best-of-N), all from scratch in torch/numpy.
- `run_lab.py` — the experiment driver (argparse CLI, CSV/JSON/PNG
  outputs, stdout summary table).
- `test_ch16_lab.py` — unit tests (pass@k vs brute-force enumeration,
  tie-breaking, parser, splits, a training overfit check) plus a
  subprocess smoke run of the whole pipeline.
