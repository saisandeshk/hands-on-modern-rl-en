# Review of ch16-lab

Scope: `docs/chapter16_reasoning/lab.md`, `code/chapter16_reasoning/*` (new files on
`w/ch16-lab`), `planning/reports/ch16-lab.md`. Everything below was checked by
re-running the shipped code on CPU (seed 0) and re-deriving every number; the
reported tables themselves reproduce exactly, so the problems are in the
*explanations* attached to the numbers, not in the tables.

## Errors

1. `docs/chapter16_reasoning/lab.md` (line 187) — "the first accepted sample is correct with
   probability roughly $p/(p + (1-p)f)$ no matter how large N grows. That asymptote is 0.86 at
   $f=0.1$ and 0.75 at $f=0.3$ — exactly the plateaus in the table." — The arithmetic does not
   follow from the formula the page states. With the page's own pool rate $p \approx 0.52$:
   $0.52/(0.52+0.48\cdot0.1)=0.915$ and $0.52/(0.52+0.48\cdot0.3)=0.783$, i.e. 0.92 and 0.78 —
   not 0.86 and 0.75. With the *pooled* rate $p=0.474$ (item 3) the formula gives 0.90 and 0.75,
   so only the $f=0.3$ number matches any value of the formula, and no single $p$ reproduces the
   pair (0.86 at $f=0.1$ would require $p\approx0.85$, which makes the $f=0.3$ value 0.95). The
   quoted numbers are in fact the *empirical* table plateaus (0.859 / 0.746), so the page
   attributes a measured plateau to a formula that does not produce it. Worse, the closed
   form is not the asymptote of the implemented rule ("first accepted sample, else fall back to
   sample 0") for a pool with heterogeneous per-problem difficulty: using the measured per-problem
   correct-sample counts $c_i$ of the seed-0 pool, the exact asymptote is
   $\mathbb{E}_i[c_i/(c_i+w_i)]$ with $w_i\sim\mathrm{Bin}(64-c_i,f)$, which I evaluated as
   0.852 at $f=0.1$ and 0.700 at $f=0.3$ — again neither 0.86/0.75 nor the formula's 0.92/0.78.
   — Suggested fix: either report the formula's own value (0.92 / 0.78) and explain why the
   measured plateau is lower (per-problem $p$ varies from $q_i=0.03$ to $0.95$, median 0.47), or
   drop the closed form and describe the plateau as measured, e.g. "a fallible verifier caps
   best-of-N at 0.86 (FP=0.1) / 0.75 (FP=0.3) on this pool".

2. `docs/chapter16_reasoning/lab.md` (line 179) — "Best-of-N with an exact verifier only needs
   _one_ correct sample in the pool, so it follows the coverage curve $1-(1-p)^N$ all the way to
   1.00." — Contradicted by the lab's own output. The code computes a `coverage_theory` column
   (printed in the stdout summary and plotted) that the page omits. Measured seed-0 values,
   best-of-N exact vs $1-(1-p)^N$ with $p=0.516$: N=1: 0.516/0.516; N=2: 0.684/0.765; N=4:
   0.848/0.945; N=8: 0.953/0.997; N=16: 0.984/1.000; N=64: 1.000/1.000. The gap is 8 points at
   N=2 and 10 points at N=4; the two curves meet only at N=64. Reason: $1-(1-p)^N$ assumes every
   problem has the same $p$, whereas here $q_i$ ranges over 0.03…0.95 (20% of the 256 problems
   below 0.25), so the empirical coverage is $\mathbb{E}_i[1-(1-q_i)^N]=0.825$ at N=4, far below
   the i.i.d. value 0.924 (Jensen). — Suggested fix: add the `coverage_theory` column to the
   table and say that best-of-N equals the *empirical* coverage of the pool, which is below
   $1-(1-p)^N$ whenever per-problem difficulty is heterogeneous.

3. `docs/chapter16_reasoning/lab.md` (line 152) — "At T=0.7 (pool rate $p \approx 0.52$)" — the
   quoted "pool rate" is only the *first* sample of each pool (`p_pool = mean(f[0])` = 0.516, SE
   ≈ 0.031 for 256 problems). The pool's true single-sample rate over all 256×64 = 16384 draws is
   0.474, which is exactly the page's own pass@1 row (0.47); the unbiased estimator with $k=1$ is
   $c_i/64$ per problem, averaged. So the page simultaneously reports a pool rate of 0.52 and a
   pass@1 of 0.47 for the same pool, and then feeds $p=0.52$ into $1-(1-p)^N$. — Suggested fix:
   state $p \approx 0.474$ (measured on 16384 samples) or explain that `p_pool` is a first-sample
   estimate, and make the coverage discussion consistent with the pass@k column.

4. `docs/chapter16_reasoning/lab.md` (lines 134, 166) — "mixed them 1:1 with teacher traces" /
   "trains on verified self-generated traces _plus_ an equal number of original teacher traces" —
   the code mixes 4000 self-traces with `min(len(kept), len(train_examples))` = 2000 teacher
   traces, i.e. **2:1**, not 1:1. The run log prints "finetune on 6000 traces (4000 self + 2000
   teacher)". (The README states the counts correctly: "4000 kept self-traces + 2000 teacher
   traces"; the page and the report say 1:1.) — Suggested fix: say "4000 self + 2000 teacher
   traces (2:1, limited by the 2000 available teacher examples)".

5. `docs/chapter16_reasoning/lab.md` (lines 165-166) — "kept 4000 verified self-traces (keep rate
   0.25 at T=0.8)" — the 0.25 is not a measurement but an artifact of the cap: `kept` is truncated
   at `--rft-cap` = 4000 and `keep_rate = len(kept)/(n_train*rft_samples) = 4000/16000 = 0.25`
   exactly, by construction. Re-running the RFT sampling and counting every parse myself gives
   7601/16000 = **0.475** true keep rate. — Suggested fix: report 0.475 (or count the rate before
   applying the cap and report the cap separately).

6. `docs/chapter16_reasoning/lab.md` (lines 170, 202-203) — "an early trial that fine-tuned on
   kept traces alone at full learning rate _dropped_ greedy accuracy 0.727 → 0.594" and "pure
   self-training overfits the easy slice (no-carry additions) and forgets the hard slice (double
   carries) — hence the 0.594 collapse without the teacher mix" — two problems.
   (a) The number does not reproduce with the shipped code path: running the described variant
   (teacher mix removed, RFT lr = training lr = 1e-3, kept traces only) on seed 0 gives greedy
   0.727 → **0.645**, not 0.594.
   (b) The stated mechanism is contradicted by the measurement, and the experiment confounds two
   changes (mix removed *and* lr raised 5×). Grouping the 256 test problems by the number of
   carries in the true trace (0/1/2 carries: 60/126/70 problems) and decoding greedily:
   base model = 0.783 / 0.706 / 0.714; after the no-mix full-lr RFT = **0.583** / 0.627 /
   **0.729**. The *easy* (no-carry) slice is where the model loses the most and the double-carry
   slice does not degrade at all — the opposite of the page's claim. (For reference, the shipped
   recipe — teacher mix, 0.2× lr — gives 0.800 / 0.722 / 0.771, an improvement in every slice.)
   — Suggested fix: report the reproduced 0.645, do not attribute the drop to a carry-slice
   story that the shipped code does not produce, and separate the two changed knobs (mix vs lr)
   before drawing a causal conclusion. Note that Exercise 3 then asks the reader to measure
   exactly this breakdown, which the page has already (incorrectly) answered.

7. `docs/chapter16_reasoning/lab.md` + `docs/.vitepress/config.mjs` — the new page is not
   registered in the sidebar: the Chapter 16 block still lists only 16.1–16.6, so
   `/chapter16_reasoning/lab` is unreachable from the site navigation (it is still built and
   still picked up by the sitemap generator, which scans `docs/**/*.md`, but no reader can
   reach it from the table of contents). AGENTS.md requires every navigable page to be listed
   in `docs/.vitepress/config.mjs`; the writer's report defers this to the coordinator ("Sidebar
   additions … coordinator to place after 16.6"), so the branch as it stands cannot be merged.
   — Suggested fix: add
   `{ text: '16.7 Lab: Test-Time Compute on Tiny Traces', link: '/chapter16_reasoning/lab' }`
   to the Chapter 16 sidebar block.

8. `docs/chapter16_reasoning/lab.md` (front-matter and H1) — "16.L Lab: Test-Time Compute on
   Tiny Traces" — "16.L" does not fit the book's numbering (Chapter 16 uses 16.1–16.6; hands-on
   pages are numbered as ordinary sections, e.g. "19.8 Hands-on: Training a DeepCoder Agent with
   rLLM"), and `scripts/fix-heading-numbers.mjs` only recognises numeric prefixes
   (`/text:\s*'((?:[A-Z]\.)?\d+(?:\.\d+)*)\s+([^']+)'/`), so an "L" section can never be synced
   with the sidebar by `npm run fix:headings`. — Suggested fix: renumber to 16.7 in the H1 and the
   front-matter title, and add it to the sidebar as 16.7 (see item 7).

9. `code/chapter16_reasoning/run_lab.py` — `print(f"[lab] train=... gen_len={max_new} vocab=16")`
   — the vocabulary has **17** tokens: `VOCAB = ['<pad>', '<eos>', '0'…'9', '+', '=', '>', 'c',
   ';']` = 2 + 10 + 5. The model head is `Linear(64, 17)`, and the 73 344 parameter count quoted
   on the page depends on that 17 (I verified `sum(p.numel()) == 73344`). The log line is printed
   on every run and is wrong. — Suggested fix: print `len(VOCAB)`.

10. `code/chapter16_reasoning/run_lab.py` (stage 4) — the kept-trace loop appends only while
    `len(kept) < args.rft_cap`, so with the cap binding (4000 of 16000 draws) the RFT set is
    drawn only from the **first ~1039 of the 2000 training problems** (987 distinct problems
    appear in the kept set; the scan stops after the 1039th training problem, 0-based index
    1038). Half of the training distribution therefore contributes no self-traces at all, and
    the "rejection-sampling" dataset is not a filtered sample of the training distribution. The
    page's explanation of stage 4 ("the verifier keeps traces the current model already solves,
    so pure self-training overfits the easy slice") does not mention this truncation.
    — Suggested fix: shuffle the training pairs before scanning (or keep scanning and subsample
    afterwards) so the kept set covers the whole training distribution, and report how many
    distinct problems it covers.

11. `code/chapter16_reasoning/_probe_tmp.py` — a leftover temporary probe script (60-epoch,
    lr=3e-3 trace printer) is committed in the branch; the writer's own report asks the
    coordinator to delete it. It is not part of the lab, is not imported anywhere, and sits in
    `code/`, which the repository convention reserves for runnable chapter experiments.
    — Suggested fix: delete the file before merging.

12. `docs/chapter16_reasoning/lab.md` (line 35) — "the base model from
    [16.1](/chapter16_reasoning/emergence-and-o1) solves about half its test problems on a single
    attempt at intermediate temperature" — §16.1 is about o1 / R1-Zero and their base models; the
    link makes the sentence read as a claim about those models, while the subject is the lab's
    own 73k-parameter addition model. — Suggested fix: "the lab's own base model (in the spirit
    of §16.1's base-model sampling analysis) solves about half its test problems …".

## Doubts

1. `docs/chapter16_reasoning/lab.md` — "T=0.7 accuracy rose 0.512 → 0.605" — the two numbers come
   from different RNG streams (the "before" value is stage 2, sampling seed 102; the "after" value
   uses seed 556), so this is not a paired comparison. Measuring the same model at T=0.7 with four
   seeds gives before = {0.512, 0.492, 0.484, 0.480} and after = {0.590, 0.590, 0.605, 0.594} for
   seeds {102, 555, 556, 777}, so the improvement is real (+8 to +12 points paired) but the quoted
   pair happens to use the best "before" seed. The greedy pair (0.727 → 0.754) is properly paired
   (same seed 555) and reproduces exactly.
2. `docs/chapter16_reasoning/lab.md` — Exercise 3 asks the reader to "set the mix to zero and
   `--rft-lr` to the training lr", but no CLI flag exposes the teacher mix (the mix is hard-coded
   in `run_lab.run`), so the exercise needs a source edit; and the shipped code has no
   per-carry-count breakdown, so the slice claim the page asserts is never measured by the lab
   itself.
3. `code/chapter16_reasoning/arith_cot.py` — `TinyReasoner.generate(self, prompts, prompt_lens,
   max_new, ...)` takes a `prompt_lens` argument that is never used in the body (the prompt is
   left-padded to a common width instead); the docstring advertises it, so a reader following
   Exercise 4's hint ("reuse `generate` with the trace as prefix") may rely on a parameter that
   does nothing. Also note `max_len = 64` bounds any such re-prompt.
4. `docs/chapter16_reasoning/lab.md` — "Past N=8, extra candidates only add more false accepts for
   the verifier to trip over." — consistent with the data (0.848 at N=8 → 0.859 at N=64), but the
   plateau also depends on the "fall back to sample 0" rule and on the fact that acceptance is
   drawn once per pool rather than per N, so the closed-form asymptote in item 1 should not be
   presented as exact.
5. `planning/reports/ch16-lab.md` / `docs/chapter16_reasoning/lab.md` — every number is seed-0
   specific; the report acknowledges this, but the page states the patterns as if they were
   properties of the method ("the qualitative pattern … is robust by construction" is asserted,
   not measured — only seed 0 was run). A second seed would strengthen the table.

## Verdict

The engineering is solid: the shipped code reproduces every table on the page bit-for-bit on
seed 0 (temperature table, the whole N-scaling table, greedy 0.727 → 0.754 after RFT), the 11
pytest tests pass, the parameter count (73 344) and the trace format check out, and the tie-break
artefact the page apologises for is real (smallest-answer tie-break 0.457 vs 0.500 under a random
tie-break). The problems are in the interpretation layer. Two central quantitative claims are
wrong: the noisy-verifier asymptote (the stated formula evaluates to 0.92/0.78, not the quoted
0.86/0.75, and the formula is not the asymptote of the implemented rule on a heterogeneous pool),
and the claim that best-of-N "follows the coverage curve $1-(1-p)^N$", which the lab's own
`coverage_theory` column contradicts by up to 10 points. The RFT paragraph is the weakest part:
the "keep rate 0.25" is an artifact of the truncation cap (true rate 0.475), the self/teacher mix
is 2:1 and not 1:1, the headline collapse "0.727 → 0.594" does not reproduce (0.645 with the
shipped knobs), and the asserted causal story (the page says the easy no-carry slice is over-fit
while the double-carry slice is forgotten; the per-carry-count measurement shows the opposite —
the easy slice loses 20 points and the double-carry slice does not degrade) is contradicted by a
measurement of the very experiment the page describes —
which, ironically, is what Exercise 3 asks the reader to measure. Finally, the page is not wired
into the sidebar and carries a non-conforming "16.L" section number, so the branch cannot be
merged as it stands.
