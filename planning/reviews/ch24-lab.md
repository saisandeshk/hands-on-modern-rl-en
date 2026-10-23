# Review of ch24-lab

Scope: `docs/chapter24_multimodal_frontier/lab.md`, `code/chapter24_multimodal_frontier/*` and
`planning/reports/ch24-lab.md` (all new on `w/ch24-lab`). I re-ran the shipped code (full run on CPU
seed 0, full run on CUDA seed 1, plus smoke runs), re-derived every formula and every unit-test hand
value, and compared the page's table with the runs. The table is honest: my CPU seed-0 run reproduces
every published number to the printed precision (pretrained 0.0217 / 0.0010 / 3.2314; RL+KL 0.6476 /
0.7563 / 1.9786 / 0.22228; RL no-KL 0.8445 / 0.9653 / 0.7308 / 6.29024), and my CUDA seed-1 run
reproduces the report's replication exactly (0.7079 / 1.3974 / 0.30321 vs 0.8578 / 0.7279 / 5.40573).
The problems below are in the reporting, the navigation and the explanation layer, not in the learning
code.

## Errors

1. `docs/.vitepress/config.mjs` + `docs/chapter24_multimodal_frontier/lab.md` — the new page is not
   registered in the sidebar, and its section number "24.L" is not a number this book uses.
   AGENTS.md requires every navigable page to be listed in `docs/.vitepress/config.mjs`. The Chapter
   24 block on the branch still lists only 24.1–24.5, so `/chapter24_multimodal_frontier/lab` is
   unreachable from the navigation; the book builds also read the page list from the sidebar
   (`scripts/build-epub.mjs`: "Found N pages in sidebar"), so the lab is dropped from the PDF/EPUB
   as well. Every other entry is `N.M` (appendices `A.1`, `B.1`, …) and no "L" section exists
   anywhere in the book. `scripts/fix-heading-numbers.mjs` matches entries with
   `text:\s*'((?:[A-Z]\.)?\d+(?:\.\d+)*)\s+([^']+)'`, so it can never sync a "24.L" heading; once a
   numeric entry is added, the script rewrites the body H1 to `# 24.6 …` while leaving the quoted
   front-matter title `title: '24.L Lab: …'` stale, so nav label and page title disagree (the same
   defect was flagged for `10.L` and `16.L`). Note also that `english-only` already ends Chapter 24
   with "24.6 Summary and Exercises" (`docs/chapter24_multimodal_frontier/review.md`), so the lab
   has to take a number and that page has to move.
   — Suggested fix: add
   `{ text: '24.6 Lab: DDPO Fine-Tuning of a Tiny Diffusion Model', link: '/chapter24_multimodal_frontier/lab' }`
   to the Chapter 24 block, rename the H1 and the front-matter title to the same string, and
   renumber the summary page to 24.7 (or make the lab 24.7).

2. `code/chapter24_multimodal_frontier/ddpo_toy.py` (`main`, `--skip-nokl`) +
   `docs/chapter24_multimodal_frontier/lab.md` (Exercise 1) — with `--skip-nokl` the script reports
   the *pretrained* checkpoint as the "no KL" run.
   `main` executes `model_nokl, hist_nokl, eval_nokl = None, [], eval_pre`, so `eval_nokl` **is** the
   pretrained evaluation. `print_summary_table` then prints a row labelled "rl no-kl" whose numbers
   are a byte-for-byte copy of the "pretrained" row, `summary.json` stores the pretrained metrics
   under the key `"rl_nokl"`, and the third panel of `samples_before_after.png` is titled
   "RL, no KL (beta=0)" while showing the pretrained samples. Reproduced
   (`--smoke --skip-nokl --device cuda`): pretrained 0.0433 0.0059 3.1066 0.00000 / rl+kl 0.0458
   0.0195 3.0502 0.00983 / **rl no-kl 0.0433 0.0059 3.1066 0.00000**. Exercise 1 of the page
   explicitly instructs `--beta 0.0 --skip-nokl`, so the exercise's own workflow produces a table in
   which the "no KL" row is the pretrained checkpoint.
   — Suggested fix: when `--skip-nokl` is given, omit the "rl no-kl" row (or print "skipped"), set
   `summary["rl_nokl"] = null`, and drop the third scatter panel instead of labelling the pretrained
   samples as "RL, no KL".

3. `docs/chapter24_multimodal_frontier/lab.md` — "The KL run keeps diversity because the penalty
   prices every step's drift. β × KL is added per denoising step, so moving the whole chain far from
   the reference costs T times the per-step price."
   The implementation does not charge T times the per-step price. `pg_loss` adds `beta * kl.mean()`
   and `kl` has shape (batch, steps), so the penalty per trajectory is β·mean_t KL_t, not
   β·Σ_t KL_t. The page states the correct formula itself a few paragraphs earlier ("plus β times
   the mean per-step KL"), so the two statements contradict each other, and the mechanism sentence
   built on it ("the policy settles where the marginal reward gain equals that price") uses the
   wrong price. What is true is only this: a chain that drifts at all T steps pays T times what a
   chain that drifts in one step pays, because the penalty is the *average* of the per-step KLs.
   — Suggested fix: "the penalty is β times the average per-step KL, so a chain that drifts at every
   step pays T times what a chain that drifts in one step pays, but the price of a chain is
   β·mean_t KL_t, not β·Σ_t KL_t".

4. `planning/reports/ch24-lab.md` — "`.venv/bin/python code/chapter24_multimodal_frontier/ddpo_toy.py
   --smoke --seed 0 --device cpu` → OK, 1.2 s" (and `code/chapter24_multimodal_frontier/README.md`:
   "Smoke: a few seconds on CPU").
   The report contradicts itself: the same report says "`.venv/bin/pytest code/chapter24_multimodal_frontier
   -q` → 10 passed in ~63 s", and that suite runs exactly that smoke command as a subprocess plus
   nine unit tests that take < 0.1 s in total (measured: `test_smoke_run_end_to_end` = 42.7 s of the
   45.1 s suite). If the smoke command took 1.2 s, the suite could not take 63 s — on the reference
   machine the smoke run itself is ~60 s.
   Measured here: the smoke run takes 38–50 s on CPU ("Done in 38.0 s" for seed 1, 39.4 s for seed
   0) while the full CPU run takes 441 s, i.e. a runtime ratio of 9–12× (the smoke configuration is
   dominated by fixed per-call overhead, not by arithmetic: counting sample-forwards the full
   configuration is ≈50× the smoke one — pretraining 4000×512 vs 300×256 = 26.7×, RL 120 iterations
   × 256×30×(1 + 2×3) vs 8 × 64×8×(1+3) ≈ 394×, evaluations 12×2048×30×3 vs 8×512×8×3 ≈ 20× — so the
   observed 9–12× is a lower bound on the work ratio). Applying that ratio to the report's 631 s full
   run gives a smoke run of 55–70 s, exactly the number implied by its own 63 s pytest line. The
   page's "well under 60 seconds on CPU" is therefore marginal even on the reference machine, and the
   README's "a few seconds" is simply wrong.
   — Suggested fix: quote one machine's timings consistently in the page, the README and the report;
   drop the 1.2 s line or replace it with the ~60 s implied by the pytest number, and say on the page
   that the smoke run is ≈1 min on CPU rather than "well under 60 seconds".

5. `docs/chapter24_multimodal_frontier/lab.md` — "The KL estimate mirrors this: 0.23 per step with
   the penalty versus 6.29 without — a 27-fold gap."
   The table on the same page gives 0.222 for the KL run, and 6.290/0.222 = 28.3, not 27. The "27"
   comes from the prose's 0.23, which is a *different* measurement of the same checkpoint: 0.23054
   is the last in-training evaluation (iteration 60 of the curve) while 0.22228 is the separate
   final `evaluate()` call in `main` with a fresh 2048-sample rollout — my CPU seed-0 run reproduces
   both exactly. The same mixing appears in the diversity sentence ("falls to 2.21 by iteration 30 and
   0.73 by iteration 60"): the curve's last point is 0.783, and 0.73 is the final evaluation.
   — Suggested fix: quote one evaluation per checkpoint ("0.222 per step with the penalty versus 6.29
   without — a 28-fold gap"), and state that the table rows come from a final evaluation with a
   fresh 2048-sample rollout while the curves end at the last in-training evaluation (0.2305 / 0.783).

6. `docs/chapter24_multimodal_frontier/lab.md` — "Subtracting the batch mean leaves the expected
   gradient unchanged (the probabilities of all trajectories sum to 1. while shrinking update noise;
   dividing by the batch std keeps step sizes stable as rewards shift upward during training."
   The sentence is broken as printed: the parenthesis opened after "unchanged" is never closed and
   "sum to 1. while shrinking" reads as a truncated "1.", so the reader cannot tell where the
   parenthetical ends or which clause it governs.
   — Suggested fix: "Subtracting the batch mean leaves the expected gradient unchanged (the
   probabilities of all trajectories sum to 1, so differentiating them gives 0) and shrinks update
   noise; dividing by the batch std keeps step sizes stable as rewards shift upward during training."

7. `docs/chapter24_multimodal_frontier/lab.md` (Exercise 4) — "Try `--clip-eps 1.0` (effectively
   REINFORCE) versus `0.05`."
   With ε = 1.0 the ratio is still clipped to [0, 2]: for a positive advantage the objective is
   A·min(ρ, 2), so a lucky chain can at most double its probability ratio per update and the
   gradient remains trust-region-limited. The run is PPO with a very loose region, not REINFORCE;
   unclipped REINFORCE needs a much larger clip (e.g. 10) or no clipping at all.
   — Suggested fix: "Try `--clip-eps 1.0` (the ratio is capped at 2, so this is PPO with a very loose
   trust region) versus `0.05`", or use `--clip-eps 10` for "effectively REINFORCE".

8. `docs/chapter24_multimodal_frontier/lab.md` (Exercise 2 hint) — "the KL cost of reaching it is
   smaller now, so the two runs should differ less — a case where alignment and the prior agree."
   As written this is backwards: (0, −0.3) is the *valley* between the three modes, i.e. exactly
   where the pretrained density is lowest, so the reward and the prior still disagree there. What
   actually changes is the travel distance: the valley lies 2.30 from the nearest mode ((0, 2)),
   2.77 from (−2.5, −1.5) and 2.60 from (2.5, −1), against 3.04 for the original peak (3, 2) from
   its nearest mode (2.5, −1), so the KL price of reaching the new peak is smaller.
   — Suggested fix: "the KL cost of reaching it is smaller now (the valley is 2.3 from the nearest
   mode, against 3.0 for the original peak), so the two runs should differ less — a case where the
   reward sits closer to the pretrained distribution, not where the two agree."

## Doubts

1. `docs/chapter24_multimodal_frontier/lab.md` — "Subtracting the batch mean leaves the expected
   gradient unchanged (the probabilities of all trajectories sum to 1 …)" — the "sum to 1" argument
   is §24.4.3's derivation for a baseline that does not depend on the sampled trajectory (the
   chapter's b(c)); the lab's baseline is the mean over the *same* batch, and there the identity is
   only approximate. With b = (1/N)Σ_j R_j and independent rollouts,
   E[(1/N)Σ_i (R_i − b)∇log p_i] = E[R∇log p] − (1/N)E[R∇log p] = (1 − 1/N)·E[R∇log p]
   (because E[b∇log p_i] = (1/N)E[R_i∇log p_i], the j≠i terms vanishing via E[∇log p_i] = 0). With
   N = 256 that is a 0.4 % shrinkage — harmless in practice, and the direction is unchanged, but the
   page's derivation is not the one that applies, and dividing by the batch std adds a further
   batch-dependent scale on top.
2. `docs/chapter24_multimodal_frontier/lab.md` — "the mean per-step KL to the frozen pretrained
   model" — this is the per-step *conditional* KL at the policy's own states (a k1-style estimator),
   not the trajectory-level D_KL(p_θ‖p_ref) that §24.4.4 adds to the loss. In the training loss the
   KL is moreover evaluated at the *stale* rollout states (one or two updates old), so the penalty is
   an approximation of the distribution-level KL the chapter's formula suggests. The page is
   accurate about the metric it reports, but the reader may carry the stronger reading across.
3. `docs/chapter24_multimodal_frontier/lab.md` (Exercise 6) — "compare its reward/diversity against
   the fine-tuned policy's single samples" — diversity here is the mean *pairwise* distance, which is
   undefined for a single sample; the reranked arm has to be compared on reward (best-of-32 vs the
   fine-tuned policy's mean reward) and on the diversity of the 32-sample pool, not on a single
   sample's diversity.
4. `docs/chapter24_multimodal_frontier/lab.md` ("What you will learn") and the module docstring —
   "what reward hacking looks like when the penalty is removed" / "reward hacking / mode collapse in
   miniature" — the no-KL run does not exploit a loophole: the analytic bump has no judge to fool, so
   the policy simply concentrates on the unique maximiser of a peaked reward. The chapter's oily
   (HPS-v2.1 only) and static (HPSv3 only) images are genuine over-optimization of a proxy; here
   the same mechanism is plain mode-seeking. The page's own text ("the optimizer is not
   misbehaving, it is executing the objective it was given") is the accurate description, so the
   bullet overstates what the lab demonstrates.
5. `docs/chapter24_multimodal_frontier/video-generation-modern.md` §24.5.2 (a page the lab links to
   for its "lesson") still says "Multiplying roughly, one video corresponds to about four thousand
   model forward computations", and the summary repeats "a decision chain on the order of four
   thousand steps". The chapter's own summary page lists exactly this as the classic mistake
   (`english-only:docs/chapter24_multimodal_frontier/review.md`, "Common pitfalls": 50×80 = 4 000 is
   the number of frame-level latent updates, but the DiT processes all frames jointly, so the cost is
   about fifty forward passes). Pre-existing and owned by another worker, but the lab's "lesson of
   24.5" rests on that section.
6. Every number on the page is a single-seed measurement (seed 0, CPU) and the report's replication
   is a different seed on a different device. The pattern held in my runs (seed 0 CPU and seed 1
   CUDA), but the page states the table as if it were a property of the method; one extra seed and a
   note that the table is one seed would be enough.

## Verdict

The engineering is good and the science is honest: the shipped script reproduces every number on
the page to the printed precision (I re-ran the whole lab on CPU seed 0 and CUDA seed 1), the
unit tests' hand values all check out (−log 2π, −0.5(2 + 2 log π), exp(−2), the ±1.2247
advantages, −2.4 / +3.0 for the clipped loss, 0.5 and 2.0 for the Gaussian KL), the DDPM reverse
mean and the fixed-variance Gaussian log-probability are correct, and every anchor and cross-reference
on the page points at a heading that exists (the slugs themselves could not be checked, because
node_modules is not installed in this worktree, so the site cannot be built here). The defects are in
the packaging and in a handful of explanation sentences:
the page is not in the sidebar and carries a section number ("24.L") that the book's numbering and
`scripts/fix-heading-numbers.mjs` do not support, so it is unreachable and missing from the
compiled book; `--skip-nokl` (the flag Exercise 1 tells the reader to use) silently reports the
pretrained checkpoint as the "RL, no KL" run in the table, in `summary.json` and in the third
scatter panel; the KL-penalty explanation says a drifting chain costs "T times the per-step price"
when the code charges β·mean_t KL_t; the report's 1.2 s smoke timing contradicts its own 63 s
pytest timing and the workload ratio (and the README's "a few seconds" is wrong); the "27-fold" KL
gap is 28-fold against the page's own table because two different evaluations of the same checkpoint
are mixed; one bullet on the baseline ends in an unclosed parenthesis; and two exercise hints are
technically off (ε = 1.0 still clips the ratio at 2, so it is not REINFORCE; the reward peak moved
into the valley is where the prior agrees *least*). Fix the sidebar entry and the numbering, the
`--skip-nokl` labelling, the explanation sentences, the timings, and the two hints, and the lab is
ready to merge.
