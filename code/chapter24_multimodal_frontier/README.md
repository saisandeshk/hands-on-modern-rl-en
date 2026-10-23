# Chapter 24 Lab: DDPO Fine-Tuning of a Tiny Diffusion Model

RL fine-tuning of a small generative model, DDPO-style, on a laptop CPU.

## What the lab shows

A 2D diffusion sampler is treated as a Markov decision process: each denoising
step draws the next latent from a Gaussian, i.e. it is an action with an
analytic log-probability. A reward scored **only on the final sample**
(a Gaussian bump over a low-density target region, standing in for an
aesthetic/alignment reward) is optimized with a PPO-clipped policy gradient
that broadcasts the terminal reward to every denoising step. The script:

1. **Pretrains** a tiny MLP denoiser (2 → 64 → 64 → 2 with a sinusoidal time
   embedding) on a three-mode Gaussian mixture with denoising score matching.
2. **Fine-tunes with a KL penalty** (`--beta 0.05`, DPOK-style): reward rises
   while the sampler stays near the pretrained distribution.
3. **Fine-tunes without the KL penalty** (`beta = 0`): higher reward, but the
   sampler collapses onto the reward peak and forgets the other modes —
   single-reward over-optimization (mode collapse) in miniature.

It saves scatter plots (before/after samples), reward/diversity/KL curves,
`metrics.csv`, `summary.json`, and prints a summary table.

Main script: `ddpo_toy.py`. Tests: `test_ch24_lab.py` (unit tests for the
Gaussian log-prob, the reward, and the clipped per-step PG loss against
hand-computed values, plus an end-to-end smoke run).

## How to run (from the repository root)

Smoke test (tiny config, under a minute on CPU):

```bash
.venv/bin/python code/chapter24_multimodal_frontier/ddpo_toy.py --smoke
```

Full lab run:

```bash
.venv/bin/python code/chapter24_multimodal_frontier/ddpo_toy.py --seed 0 --device cpu
```

Useful options: `--device {auto,cpu,cuda}`, `--seed`, `--beta` (KL weight of
the KL run), `--rl-iters`, `--batch-rl`, `--diffusion-steps`,
`--skip-nokl` (skip the beta=0 comparison), `--outdir`.
Run with `--help` for the full list.

Run the tests:

```bash
.venv/bin/pytest code/chapter24_multimodal_frontier -q
```

## Expected runtime

- Smoke: under a minute on CPU (1.9 s measured on the reference machine; the full
  pytest suite, which includes one smoke run plus the unit tests, takes ~5 s).
- Full run: about 10–11 minutes on CPU (631 s measured on the reference run),
  well under the 15-minute budget. GPU is faster but not needed.

## Expected results (full run, seed 0, CPU)

| checkpoint          | mean reward | frac in target | diversity (mean pairwise dist) | KL/step to pretrained |
| ------------------- | ----------- | -------------- | ------------------------------ | --------------------- |
| pretrained          | 0.022       | 0.001          | 3.23                           | 0.0                   |
| RL + KL (beta=0.05) | 0.65        | 0.76           | 1.98                           | 0.22                  |
| RL, no KL (beta=0)  | 0.84        | 0.97           | 0.73                           | 6.29                  |

Reading: both RL runs learn to hit the target region (reward 0.02 → 0.65/0.84),
but the unpenalized run collapses onto the reward peak (diversity 3.23 → 0.73,
KL/step 6.3) while the KL-penalized run keeps most of the pretrained spread
(diversity 1.98, KL/step 0.22) at a moderate reward cost. Outputs land in
`runs/chapter24_multimodal_frontier/` (gitignored): `samples_before_after.png`,
`training_curves.png`, `metrics.csv`, `summary.json`.
