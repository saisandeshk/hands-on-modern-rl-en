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

- `.venv/bin/python code/chapter24_multimodal_frontier/ddpo_toy.py --smoke --seed 0 --device cpu` → OK, 1.2 s.
- `.venv/bin/python code/chapter24_multimodal_frontier/ddpo_toy.py --seed 0 --device cpu` (full) → OK, 630.9 s, results in table above.
- Full run seed 1 on CUDA (background, robustness check) → OK, 37.5 s, same pattern.
- `.venv/bin/pytest code/chapter24_multimodal_frontier -q` → 10 passed in ~63 s
  (includes the subprocess smoke run on CPU).
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
