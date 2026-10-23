# Report: ch24-lab — DDPO toy lab for Chapter 24

## Summary

Built a hands-on DDPO-style lab for Chapter 24 (which had no runnable code). New files,
all inside the allowed scope:

- `code/chapter24_multimodal_frontier/ddpo_toy.py` — main lab script. Pretrains a tiny
  MLP denoiser on a 2D three-mode Gaussian mixture with denoising score matching, then
  RL-fine-tunes its sampler with a PPO-clipped policy gradient over the denoising chain
  (terminal Gaussian-bump reward on the final sample, group-relative advantages,
  per-step analytic Gaussian log-probs, DPOK-style KL penalty to the frozen pretrained
  model), twice: with `beta = 0.05` and with `beta = 0`. Saves scatter plots, reward /
  diversity / KL curves, `metrics.csv`, `summary.json` under
  `runs/chapter24_multimodal_frontier/` and prints a summary table. CLI has `--seed`,
  `--device {auto,cpu,cuda}`, `--smoke`, plus lab options.
- `code/chapter24_multimodal_frontier/test_smoke.py` — 10 pytest tests: subprocess smoke
  run plus unit tests of the Gaussian log-prob, reward, and clipped per-step PG loss
  against hand-computed values.
- `code/chapter24_multimodal_frontier/README.md`, `requirements.txt`.
- `docs/chapter24_multimodal_frontier/lab.md` — lab page (`24.L Lab: DDPO Fine-Tuning
of a Tiny Diffusion Model`) with real numbers from the full run, no images.

Full run (seed 0, CPU, 631 s): pretrained reward 0.022/diversity 3.23 → RL+KL
0.648/1.98 (KL/step 0.22) vs RL no-KL 0.845/0.73 (KL/step 6.29). A second full run
(seed 1, CUDA, 37.5 s) replicates the pattern (0.708/1.40/KL 0.30 vs 0.858/0.73/KL 5.41).

## Technical fixes

No pre-existing files were touched (another worker owns the chapter pages), so there
are no corrections to existing claims. Verification performed on the new content:

- **Gaussian log-prob** (`gaussian_log_prob`): unit-tested against hand values —
  `N(0,I)` at the mean gives `-log(2π)`; `x=(1,0)`, var 0.5 gives `-0.5·(2+2·log π)`.
- **Reward** (`target_reward`): 1.0 at the peak, `exp(-2)` two widths away, range
  [0, 1] on a random batch.
- **Advantages** (`compute_advantages`): rewards `[0.3, 0.5, 0.7]` give
  `[-1.2247, 0, +1.2247]` (population std, matching the worked example style of
  Section 24.4.5).
- **Clipped PG loss** (`pg_loss`): ratio 1 → `-mean(A)`; ratio 3 with A=+2/clip 0.2 →
  -2.4 (clip binds); ratio 3 with A=-1 → +3.0 (clip correctly does not bind for
  negative advantage); `beta>0` without `kl` raises `ValueError`.
- **KL** (`gaussian_kl`): unit-norm mean shift at var 1 → 0.5, at var 0.25 → 2.0.
- **End-to-end learning**: both full runs show reward rising monotonically in both
  conditions with diversity/KL diverging as predicted by Sections 24.4.4/24.5
  (KL run keeps ~60% of pretrained diversity at ~77% of the no-KL reward).
- **Notation** matches the chapter: $s_t=(x_t,t)$, $a_t=x_{t-1}$,
  $\tau=(x_T,\dots,x_0)$, trajectory advantage $\hat A$ broadcast to all steps,
  ratio $\rho_t$, clip $\epsilon$, KL weight $\beta$.

## Other changes

- Fixed a crash found during development: `reverse_mean` was decorated with
  `@torch.no_grad()`, which detached the sampler means and broke `loss.backward()`
  in `ddpo_update` (callers that need no-grad — `rollout`, `evaluate` — are already
  decorated themselves).
- Relaxed four unit-test assertions from exact float equality to `math.isclose`
  (float32 rounding, ~3e-8 relative).
- Quoted the docs front-matter title (`title: '24.L Lab: ...'`) — the unquoted colon
  broke the VitePress build (YAML parse error).

## Needs external verification

None. No papers, authors, venues, benchmark numbers, quotes, or URLs were invented:
the lab cites only chapter sections (all anchor links verified against the built HTML),
and all reported numbers come from executed runs. One deliberate simplification to be
aware of: the "aesthetic-like" reward is an analytic Gaussian bump, stated as such in
both the code docstring and the docs page — it stands in for a learned reward model.

## Sidebar additions

- Title: `24.L Lab: DDPO Fine-Tuning of a Tiny Diffusion Model`, link:
  `/chapter24_multimodal_frontier/lab` (file: `docs/chapter24_multimodal_frontier/lab.md`).
  `docs/.vitepress/config.mjs` was not edited per the worker rules.

## Commands run and results

- `.venv/bin/python code/chapter24_multimodal_frontier/ddpo_toy.py --smoke --seed 0 --device cpu` → OK, 1.9 s (re-measured during review fixes).
- `.venv/bin/python code/chapter24_multimodal_frontier/ddpo_toy.py --seed 0 --device cpu` (full) → OK, 630.9 s, results in table above.
- Full run seed 1 on CUDA (background, robustness check) → OK, 37.5 s, same pattern.
- `.venv/bin/pytest code/chapter24_multimodal_frontier -q` → 11 passed in 4.4 s
  (re-measured during review fixes on the reference machine; includes the subprocess
  smoke run on CPU plus the new `--skip-nokl` plotting test — the previously reported
  "~63 s" came from a much slower machine and contradicted the 1–2 s smoke timing).
- `.venv/bin/python code/chapter24_multimodal_frontier/ddpo_toy.py --smoke --seed 2 --device cpu --skip-nokl` → OK, 1.4 s; table prints only
  pretrained + rl+kl rows, `summary.json` records `"rl_nokl": null`, scatter figure
  has 2 panels.
- `npx prettier --write` on the two new Markdown files + `npx prettier --check` on the
  new code dir and lab page → clean. (Prettier has no Python parser, so the two `.py`
  files cannot be prettier-formatted; they follow the surrounding code style.)
- `npx vitepress build docs` → success (49.6 s). Dead-link safe; all five
  `#_24-4-N-...` anchors verified present in the built `visual-generation-dancegrpo.html`.
- `node scripts/check-code-links.mjs` → "Repository links OK (90 checked)".
- `node scripts/check-english-quality.mjs` → passed, 0 issues.

## Open issues

- Coordinator must add the lab page to the sidebar in `docs/.vitepress/config.mjs`.
- `runs/` output (plots, CSV, JSON, seed-1 replication) is gitignored and stays in the
  working tree only.
- Full-run CPU time (631 s) is within the 15-minute budget but leaves modest headroom;
  the dominant cost is the two 60-iteration RL loops (KL + no-KL) with full-size evals.
  If CI time becomes a concern, `--skip-nokl` or fewer `--rl-iters` degrade gracefully.

## Review response

One line per finding in `planning/reviews/ch24-lab.md` (errors 1–8, then doubts 1–6).

- Error 1 (sidebar / "24.L" number): rejected — worker rules forbid editing
  `docs/.vitepress/config.mjs`, the task spec mandates the `24.L Lab:` title, and the
  24.6 slot is owned by another worker's page; the sidebar addition stays listed under
  "Sidebar additions" for the coordinator.
- Error 2 (`--skip-nokl` mislabels pretrained as "rl no-kl"): accepted — `main` now
  sets `eval_nokl = None` / `summary["rl_nokl"] = None`, prints only two table rows plus
  a "(no-KL run skipped)" note, and `make_plots` draws two scatter panels when the run
  is skipped; verified with a `--skip-nokl` smoke run and a new plotting unit test.
- Error 3 ("T times the per-step price"): accepted — page now says the loss adds
  β times the _mean_ per-step KL, so an all-steps drifting chain pays T times a
  one-step drifting chain, not T times a per-step price.
- Error 4 (smoke timing "1.2 s" vs "~63 s" suite): accepted as self-contradiction —
  re-measured on one machine: smoke 1.9 s, suite 11 passed in 4.4 s; page/README now
  say "under a minute on CPU" with the measured numbers (the reviewer's ~40–60 s
  reflects a slower machine; both satisfy the <60 s smoke budget).
- Error 5 ("27-fold" KL gap): accepted — table now reads 28-fold (6.290/0.222), and a
  note distinguishes the final 2048-sample evaluation from the in-training curve
  endpoints (KL 0.2305, no-KL diversity 0.783 at iter 60), all cross-checked against
  `runs/chapter24_multimodal_frontier/metrics.csv`.
- Error 6 (unclosed parenthesis in baseline bullet): accepted — sentence repaired to
  "the probabilities of all trajectories sum to 1, so differentiating them gives 0".
- Error 7 (`--clip-eps 1.0` "effectively REINFORCE"): accepted — hint now says the
  ratio is still capped at 2, i.e. PPO with a very loose trust region.
- Error 8 (valley hint "prior agrees"): accepted — hint now says the valley is where
  prior density is lowest but sits closer to the modes (2.3 vs 3.04), so the KL price
  of reaching it is smaller (distances hand-checked: valley–(0,2) = 2.3,
  peak–(2.5,−1) = 3.04).
- Doubt 1 (batch-mean baseline unbiasedness): accepted in part — bullet now notes the
  identity is exact for a fixed baseline and holds up to a negligible 1/N shrinkage
  for the in-batch mean (N = 256).
- Doubt 2 (conditional per-step KL vs trajectory KL): accepted in part — `evaluate`
  description now states the metric is a conditional per-step average along sampled
  chains, an approximation of the trajectory-level KL.
- Doubt 3 (Exercise 6 diversity of a single sample): accepted — exercise now compares
  best-of-32 reward vs fine-tuned mean reward and pool diversity vs policy diversity.
- Doubt 4 ("reward hacking" wording): accepted — page bullet, README, and code
  docstring now say "single-reward over-optimization (mode collapse)".
- Doubt 5 (video-generation §24.5.2 "four thousand forwards"): rejected (out of
  scope) — that page is owned by another worker; the lab no longer leans on it beyond
  the pre-existing cross-reference.
- Doubt 6 (single-seed table): accepted — results section now states all numbers are
  the single seed-0 run (plus the seed-1 CUDA replication noted in this report).
