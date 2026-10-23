---
title: '16.7 Lab: Test-Time Compute on Tiny Traces'
---

# 16.7 Lab: Test-Time Compute on Tiny Traces

Train a 73k-parameter character-level transformer to add two-digit numbers
through an explicit carry trace, then fix its mistakes without touching its
weights: draw more samples, vote, and let a verifier pick. Everything in
this lab runs on a laptop CPU in under a minute.

## What you will learn

- Why a model that is right barely half the time on one attempt can still
  approach perfect accuracy when you sample from it 64 times — and what
  exactly "approach" costs at each step.
- How the three parallel-sampling rules of
  [Section 16.3](/chapter16_reasoning/test-time-scaling) behave on real
  model outputs: the pass@k estimator, majority vote (self-consistency),
  and best-of-N with a verifier.
- Why majority vote saturates while verifier best-of-N keeps improving,
  and why a noisy verifier breaks that improvement.
- How one round of rejection-sampling fine-tuning (the stage
  [Section 16.2](/chapter16_reasoning/r1-zero-pure-rl-reasoning) adds on top
  of pure RL) shifts single-sample accuracy — and how it can backfire when
  the teacher data is left out.

## Background

Chapter 16 separates two kinds of failure. A **knowledge failure** means the
model never learned the skill; only training helps. A **sampling failure**
means the skill is there but this particular generation went wrong; extra
inference compute — more candidates, revision, search — can fix it
([16.3](/chapter16_reasoning/test-time-scaling)). This lab constructs a
clean sampling-failure regime on purpose: the base model from
[16.1](/chapter16_reasoning/emergence-and-o1) solves about half its test
problems on a single attempt at intermediate temperature, so almost every
problem sits in the "sometimes right, sometimes wrong" band where
within-group comparison ([16.2](/chapter16_reasoning/r1-zero-pure-rl-reasoning))
and test-time scaling both have signal.

The task is two-digit addition written as a short chain of thought that
makes every carry explicit:

```text
23+58=  →  3+8=11>1c1;2+5+1=8>8c0;=81
```

Read it left to right: add the units, write down `1`, carry `1`; add the
tens plus the carry, write down `8`, carry `0`; the answer follows `;=`.
A rule-based checker that compares the parsed answer with the true sum
plays the role of §16.2's verifier $v(x, y)$: cheap, exact, and immune to
reward hacking. The lab then replays the §16.3 parallel-sampling analysis —
coverage $1-(1-p)^N$, majority voting as the verifier-free fallback, and
the plateau a fallible verifier imposes — on the model's own samples.

## Setup and how to run

The lab needs only the project virtualenv (torch, numpy, matplotlib; no
downloads, no internet). From the repository root:

```bash
# Smoke test: the whole pipeline on a tiny config (<60 s on CPU)
.venv/bin/python code/chapter16_reasoning/run_lab.py --smoke --device cpu

# Full lab (~30 s on CPU; add --device cuda if you have a GPU)
.venv/bin/python code/chapter16_reasoning/run_lab.py --device cpu

# Unit tests, including a subprocess smoke run
.venv/bin/pytest code/chapter16_reasoning -q
```

Outputs land in `runs/chapter16_reasoning/` (gitignored): `metrics.json`
with every number, `scaling.csv` / `temperature.csv`, the plots
`scaling_curves.png` / `temperature.png`, and a summary table printed to
stdout. Useful flags: `--seed` (everything is seeded), `--epochs`,
`--n-train`, `--n-test`, `--n-samples`, `--temps`, `--fp-rates`,
`--rft-samples`, and `--no-rft` to skip the fine-tuning round.

The implementation lives in three files:

- [`arith_cot.py`](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter16_reasoning/arith_cot.py) —
  trace format, tokenizer, data splits, the tiny transformer, training,
  and the §16.3 metrics, all from scratch in torch/numpy.
- [`run_lab.py`](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter16_reasoning/run_lab.py) —
  the experiment driver (CLI, outputs, plots, summary table).
- [`test_ch16_lab.py`](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter16_reasoning/test_ch16_lab.py) —
  unit tests plus the end-to-end smoke test.

## Walkthrough of the key code

**The trace is the chain of thought.** `render` in `arith_cot.py` turns an
operand pair into a prompt, a column-by-column derivation, and the answer,
so supervision never has to say _how_ to carry — the format demonstrates
it:

```python
prompt, completion, answer = render(23, 58)
# prompt     = "23+58="
# completion = "3+8=11>1c1;2+5+1=8>8c0;=81"
```

**Parsing and verifying stay separate.** `parse_answer` reads whatever the
model claimed after `;=` (returning `None` for malformed traces); the
caller decides whether the claim equals the true sum. That split mirrors
§16.2: extraction is a format problem, correctness is a verifier problem.

```python
parse_answer("3+8=11>1c1;2+5+1=8>8c0;=81")  # 81
parse_answer("3+8=11>1c1 no answer")        # None
```

**Pass@k is estimated, not just counted.** With $n$ samples of which $c$
are correct, the lab uses the unbiased Chen et al. estimator
$1 - \binom{n-c}{k}/\binom{n}{k}$ rather than the raw fraction of pools
containing a hit, and the test suite checks it against brute-force subset
enumeration:

```python
def pass_at_k(n, c, k):
    ...
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
```

**Voting needs a tie-break.** `majority_answer` lets unparseable samples
abstain and breaks ties toward the smallest answer, so repeated runs agree
exactly. `best_of_n_noisy` implements verifier selection as "first
accepted sample, else fall back to sample 0", with an acceptance matrix
fixed per false-positive rate so the N-curves stay comparable. An exact
verifier accepts precisely the correct samples, which is why its curve
equals the empirical coverage.

**Rejection sampling mixes, not replaces.** The fine-tuning round trains
on verified self-generated traces _plus_ an equal number of original
teacher traces at a reduced learning rate — pure self-training skews
toward already-solved problems, as the results below show.

## Results from an actual run

Full configuration, seed 0, CPU, 25.7 s total: 2000 training pairs, 30
epochs (73,344 parameters), 256 disjoint test pairs, 64 samples per
problem at T=0.7.

Single-sample accuracy falls monotonically with temperature — greedy
decoding is the best one-shot strategy, and heat buys diversity at the
price of correctness:

| T        | 0.0   | 0.3   | 0.7   | 1.0   | 1.3   |
| -------- | ----- | ----- | ----- | ----- | ----- |
| accuracy | 0.727 | 0.664 | 0.512 | 0.320 | 0.113 |

At T=0.7 (pool rate $p \approx 0.52$) the N-scaling reproduces the §16.3
picture quantitatively:

| N   | pass@k | majority | best-of-N exact | best-of-N FP=0.1 | best-of-N FP=0.3 |
| --- | ------ | -------- | --------------- | ---------------- | ---------------- |
| 1   | 0.47   | 0.52     | 0.52            | 0.52             | 0.52             |
| 2   | 0.67   | 0.46     | 0.68            | 0.66             | 0.65             |
| 4   | 0.83   | 0.54     | 0.85            | 0.79             | 0.72             |
| 8   | 0.93   | 0.66     | 0.95            | 0.85             | 0.75             |
| 16  | 0.98   | 0.66     | 0.98            | 0.85             | 0.75             |
| 32  | 1.00   | 0.69     | 1.00            | 0.86             | 0.75             |
| 64  | 1.00   | 0.71     | 1.00            | 0.86             | 0.75             |

The rejection-sampling round kept 4000 verified self-traces (keep rate
0.25 at T=0.8), mixed them 1:1 with teacher traces, and fine-tuned 2
epochs at 0.2× learning rate: greedy accuracy rose 0.727 → 0.754 and
T=0.7 accuracy rose 0.512 → 0.605. The guardrails matter — an early trial
that fine-tuned on kept traces alone at full learning rate _dropped_
greedy accuracy 0.727 → 0.594.

## What to look for and why it happens

**Majority vote saturates; coverage does not.** As N grows, the majority
converges to the model's modal answer per problem — right on problems
where the model is usually right, permanently wrong where its most common
output is a systematic error (here, mostly mishandled carries). The vote
therefore plateaus near 0.71. Best-of-N with an exact verifier only needs
_one_ correct sample in the pool, so it follows the coverage curve
$1-(1-p)^N$ all the way to 1.00. This is the §16.3 distinction between
"the correct answer appears" and "the system picks it": voting approximates
picking by popularity, verification picks by checking.

**A noisy verifier turns selection into dilution.** With false-positive
rate $f$, each wrong sample is accepted with probability $f$, so the first
accepted sample is correct with probability roughly $p/(p + (1-p)f)$ no
matter how large N grows. That asymptote is 0.86 at $f=0.1$ and 0.75 at
$f=0.3$ — exactly the plateaus in the table. Past N=8, extra candidates
only add more false accepts for the verifier to trip over. When §16.3 says
"the more reliable the verifier, the more it is worth spending the budget
on parallel sampling", this plateau is the quantitative form of that
advice.

**Watch the N=2 dip in the majority column.** With two samples, a 1–1
split is broken toward the smaller answer, which is worse than a coin
flip here — majority voting needs odd N or a better tie-break to be
monotone. The unit test pins the tie behavior down so this artifact is
reproducible rather than mysterious.

**Rejection sampling amplifies what the filter keeps.** The verifier keeps
traces the current model already gets right, so pure self-training
overfits the easy slice (no-carry additions) and forgets the hard slice
(double carries) — hence the 0.594 collapse without the teacher mix. The
mix anchors the original distribution while the verified traces raise the
weight of correct reasoning, the same reason full R1-style pipelines pair
rejection sampling with other data stages.

## Exercises

1. **Find the diversity sweet spot.** Re-run the pool stage at T=1.0 and
   T=0.3 (`--temps` only affects stage 2; edit `T_SCALE` or add a flag for
   the pool). At which temperature does majority vote peak, and why does
   the exact-verifier curve prefer a higher temperature than the vote
   does? _Hint: voting needs the mode to be right; verification needs just
   one hit — relate both to the per-answer distribution as T varies._
2. **Map the verifier-reliability threshold.** Sweep `--fp-rates` over
   `0.0,0.05,0.2,0.5` and find where best-of-N at N=64 stops beating
   majority vote. _Hint: compare the asymptote $p/(p+(1-p)f)$ with the
   majority plateau from the table._
3. **Reproduce the RFT collapse.** Fine-tune on kept traces only at full
   lr (set the mix to zero and `--rft-lr` to the training lr). Check which
   problem slice loses most — group test accuracy by carry count.
   _Hint: count carries in the true trace with a one-line helper over
   `render`._
4. **Add sequential revision.** Implement §16.3's second strategy: when a
   sample's parsed answer is wrong, re-prompt the model with its own trace
   plus a "check the carries" suffix and one more attempt. Does one
   revision round beat drawing one extra parallel sample for the same token
   budget? _Hint: reuse `generate` with the trace as prefix; keep the
   verifier fixed so the comparison is fair._
5. **Harden the task.** Switch the data to 2-digit × 1-digit products
   (e.g. `23×7=` with partial-product traces) and re-tune epochs until
   greedy accuracy lands near 0.5. Which scaling curve changes shape most,
   and why? _Hint: multiplication facts are less regular than addition
   facts — think about what that does to the error correlation across
   samples._
6. **Stop early.** Implement adaptive stopping: after each batch of 4
   samples, stop when all accepted/voted answers agree, and record samples
   saved versus accuracy lost against fixed N=64. _Hint: this is the
   §16.5 idea in miniature — the stopping rule is a tiny adaptive budget._
