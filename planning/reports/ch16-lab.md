# Report: ch16-lab (Chapter 16 hands-on lab)

## Summary

Built the missing Chapter 16 lab: a 73k-parameter character-level
transformer learns two-digit addition as explicit carry traces
(`23+58=` → `3+8=11>1c1;2+5+1=8>8c0;=81`), then test-time compute is
scaled without touching weights. New directory `code/chapter16_reasoning/`
(`arith_cot.py`, `run_lab.py`, `test_smoke.py`, `requirements.txt`,
`README.md`) plus docs page `docs/chapter16_reasoning/lab.md`. Full run
(seed 0, CPU): 25.7 s; temperature sweep greedy 0.727 → 0.113 at T=1.3;
at T=0.7, pass@k and exact-verifier best-of-N reach 1.00 by N=64 while
majority vote saturates at 0.71 and noisy-verifier best-of-N plateaus at
0.86 (FP=0.1) / 0.75 (FP=0.3); rejection-sampling fine-tuning lifts greedy
0.727 → 0.754. All 11 pytest tests pass; `npx vitepress build docs`
succeeds; code-links check passes.

## Technical fixes

Bugs below were found by actually running the code (loss curves, decoded
traces, metric tables), not by inspection.

1. **Label-mask length bug (`arith_cot.py:train_reasoner`).** The
   ignore-mask was built as `[-100] * (plen + prompt_len) + ...`, counting
   the prompt twice, so labels were one prompt-length longer than inputs
   and `F.cross_entropy` raised a batch-size mismatch (9984 vs 8448).
   Fixed to `[-100] * plen`. Verified by: the crash reproduced on the
   first smoke run and disappeared after the one-line fix.
2. **No end-of-trace token (`arith_cot.py` + `run_lab.py`).** With
   fixed-length generation and PAD-only termination, a 60-epoch model
   produced arithmetically perfect traces but never stopped (`;=1383`,
   `;=6666`), so every parse failed. Added an `<eos>` token: appended to
   every training completion, per-row stopping in `generate`, `decode`
   cuts at first `<eos>`. Verified by: probe script printing greedy
   completions for 8 held-out problems — 8/8 exact traces with parses
   after the fix vs 0/8 before.
3. **RFT collapse without teacher mix (`run_lab.py` stage 4).** First RFT
   design (kept self-traces only, full lr) dropped greedy accuracy
   0.727 → 0.594: the verifier keeps what the model already solves, so
   pure self-training overfits no-carry problems and forgets double
   carries. Changed to a 1:1 self/teacher mix at 0.2× lr (standard RFT
   practice, matching §16.2's pipeline pairing rejection sampling with
   other data). Verified by: rerun gives greedy 0.727 → 0.754 and T=0.7
   accuracy 0.512 → 0.605.
4. **Calibration to the sampling-failure regime.** 14 epochs gave ~0%
   (facts unlearned); 60 epochs gave ~100% (curves flat). 30 epochs /
   2000 pairs lands greedy at 0.73 and pool rate p≈0.52 — the
   "sometimes right" band where scaling has signal. Verified by: the
   measured temperature table (monotone 0.727 → 0.113).

## Other changes

- `code/chapter16_reasoning/arith_cot.py`: trace format, char tokenizer,
  disjoint splits, `TinyReasoner` (2-layer transformer from scratch),
  teacher-forced training, unbiased `pass_at_k`, `majority_answer`
  (smallest-answer tie-break), `coverage_at_n`, `best_of_n_noisy`.
- `code/chapter16_reasoning/run_lab.py`: argparse CLI (`--seed`,
  `--device`, `--smoke`, data/model/RFT knobs, `--no-rft`), five stages,
  CSV/JSON/PNG outputs under `runs/chapter16_reasoning/`, stdout summary.
- `code/chapter16_reasoning/test_smoke.py`: 11 tests — hand-computed
  `render` traces, parser cases, pass@k vs brute-force subset enumeration
  (20 random cases) plus edge cases, tie-breaks, noisy-verifier cases,
  disjoint splits, EOS/tokenizer behavior, a CPU overfit check (loss
  halves, ≥6/8 greedy reproductions), and a subprocess `--smoke` run
  asserting outputs and pass@k monotonicity.
- `docs/chapter16_reasoning/lab.md`: full lab page (learn/background with
  links to §§16.1–16.3, setup, code walkthrough with short excerpts, the
  real seed-0 tables above, why-it-happens analysis, 6 exercises with
  hints). No images, per spec.
- `code/chapter16_reasoning/README.md`, `requirements.txt` (torch, numpy,
  matplotlib).

## Needs external verification

- Exact table numbers are seed-0 specific; reruns with other seeds will
  differ by sampling noise (n=256 test keeps it to a few points). The
  qualitative pattern (saturation vs continued growth, noisy plateaus) is
  robust by construction, but I ran only seed 0 end-to-end.
- The pass@k estimator attribution (Chen et al., HumanEval 2021) is
  standard knowledge; no paper claims beyond the chapter's own text were
  added, so no external fact-checking backlog.

## Sidebar additions

- `16.L Lab: Test-Time Compute on Tiny Traces` → `/chapter16_reasoning/lab`
  (new page; coordinator to place after 16.6).

## Commands run and results

- `.venv/bin/python code/chapter16_reasoning/run_lab.py --smoke --device cpu`
  → exit 0, ~1.4 s.
- `.venv/bin/python code/chapter16_reasoning/run_lab.py --device cpu --seed 0`
  (full: 2000 train / 30 epochs / 256 test / 64 samples / RFT) → exit 0,
  25.7 s; outputs in `runs/chapter16_reasoning/`.
- `.venv/bin/pytest code/chapter16_reasoning -q` → 11 passed in ~5 s.
- `npx prettier --write` on the md files → clean (`--check` passes);
  `.py` files are outside prettier's parsers (explicit paths error with
  "no parser inferred", same as all existing `code/**/*.py`).
- `node scripts/check-code-links.mjs` → "Repository links OK (90 checked)".
- `npx vitepress build docs` → success in ~19 s.

## Open issues

- **Stray file needs coordinator deletion:
  `code/chapter16_reasoning/_probe_tmp.py`.** It was a temporary
  train-and-print probe. `rm`/`mv` are permission-blocked in this
  environment (both auto-rejected; per rules I did not work around it).
  Please delete it; nothing imports it and pytest does not collect it.
- RFT's T=0.7 jump (0.512 → 0.605) is much larger than its greedy bump
  (0.727 → 0.754). Plausible (fine-tuning sharpens the distribution, so
  sampling concentrates near the greedy answer), but worth a second look
  if the coordinator reruns with other seeds.
- The even-N majority dip (N=2: 0.46 < N=1: 0.52) is a deterministic
  tie-break artifact, documented on the lab page and pinned by a unit
  test — not a bug.
