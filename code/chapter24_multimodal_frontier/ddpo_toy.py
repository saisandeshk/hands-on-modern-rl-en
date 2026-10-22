#!/usr/bin/env python3
"""DDPO-style RL fine-tuning of a tiny 2D diffusion model (Chapter 24 lab).

What this lab shows
-------------------
A diffusion sampler is a sequential decision process: starting from noise
``x_T``, each denoising step draws the next latent ``x_{t-1}`` from a Gaussian
whose mean is predicted by the model. That is exactly a policy
``pi_theta(a_t | s_t)`` with state ``s_t = (x_t, t)`` and action
``a_t = x_{t-1}``, and one full denoising chain is a trajectory
``tau = (x_T, ..., x_0)``. A reward scored on the final sample ``x_0`` can
therefore be optimized with a policy-gradient method (DDPO): the terminal
reward is broadcast back to every denoising step through the sum of
per-step log-probabilities.

The lab has three phases:

1. **Pretrain.** A tiny MLP denoiser learns a 2D mixture-of-Gaussians
   distribution with standard denoising score matching.
2. **RL fine-tune with a KL penalty (DPOK-style).** A copy of the pretrained
   model is fine-tuned with a PPO-clipped policy gradient so its samples move
   into a high-reward target region, while a per-step KL penalty toward the
   frozen pretrained (reference) model keeps it close to the original
   distribution.
3. **RL fine-tune without the KL penalty.** A second copy trains on the same
   reward with ``beta = 0``. It reaches a higher reward but collapses onto the
   reward peak and forgets the pretrained distribution (single-reward
   over-optimization / mode collapse in miniature).

Everything is implemented from scratch with torch (no RL libraries).

How to run (from the repository root)::

    .venv/bin/python code/chapter24_multimodal_frontier/ddpo_toy.py --smoke
    .venv/bin/python code/chapter24_multimodal_frontier/ddpo_toy.py

Outputs (metrics CSV/JSON and PNG plots) go to
``runs/chapter24_multimodal_frontier/``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Fixed lab setup: toy data distribution and reward.
# ---------------------------------------------------------------------------

# Three Gaussian modes forming a triangle in the plane.
MODE_CENTERS = ((-2.5, -1.5), (0.0, 2.0), (2.5, -1.0))
MODE_STD = 0.6

# Target region for the RL reward: a low-density corner of the plane, so the
# policy must visibly move mass away from the pretrained modes to earn reward.
TARGET_CENTER = (3.0, 2.0)
REWARD_WIDTH = 1.0
# A sample counts as "in the target region" when its reward exceeds this.
TARGET_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Reproducibility and device helpers.
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, numpy and torch (CPU and CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str) -> torch.device:
    """Resolve ``--device {auto,cpu,cuda}`` to a torch device."""
    choice = choice.lower()
    if choice not in ("auto", "cpu", "cuda"):
        raise ValueError(f"--device must be auto, cpu or cuda, got {choice!r}")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(choice)


# ---------------------------------------------------------------------------
# Core RL pieces (also covered by unit tests in test_smoke.py).
# ---------------------------------------------------------------------------


def gaussian_log_prob(
    x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor | float
) -> torch.Tensor:
    """Log-probability of ``x`` under a diagonal Gaussian ``N(mean, var * I)``.

    Sums over the last dimension, so for 2D samples the result has one value
    per sample: ``log p(x) = -0.5 * ||x - mean||^2 / var
    - (d / 2) * log(2 * pi * var)`` with ``d = 2``.
    """
    if not torch.is_tensor(var):
        var = torch.as_tensor(var, dtype=x.dtype, device=x.device)
    sq = ((x - mean) ** 2 / var).sum(dim=-1)
    dim = x.shape[-1]
    norm = dim * torch.log(2.0 * torch.as_tensor(math.pi, dtype=x.dtype) * var)
    return -0.5 * (sq + norm)


def target_reward(x: torch.Tensor) -> torch.Tensor:
    """Final-sample reward: a Gaussian bump around ``TARGET_CENTER``.

    This plays the role of the aesthetic / alignment reward in the chapter: a
    black-box scalar scored only on the finished sample ``x_0``. It equals 1
    at the target center and decays with squared distance (width
    ``REWARD_WIDTH``).
    """
    center = torch.as_tensor(
        TARGET_CENTER, dtype=x.dtype, device=x.device
    )
    sq_dist = ((x - center) ** 2).sum(dim=-1)
    return torch.exp(-sq_dist / (2.0 * REWARD_WIDTH**2))


def compute_advantages(
    rewards: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Group-relative advantages: ``(r - mean) / (std + eps)``.

    The whole rollout batch is the comparison group (the DanceGRPO idea from
    Section 24.4.5 in miniature): subtracting the group mean removes the
    prompt-independent reward level, and dividing by the group std puts the
    update on a stable scale. Uses the population std so the values match the
    hand computation in the unit test.
    """
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def gaussian_kl(mean_new: torch.Tensor, mean_old: torch.Tensor, var: float) -> torch.Tensor:
    """Per-sample KL ``D_KL(N(new, var I) || N(old, var I))``.

    For two Gaussians sharing the same fixed variance this is
    ``||mean_new - mean_old||^2 / (2 * var)``, summed over dimensions.
    """
    return ((mean_new - mean_old) ** 2).sum(dim=-1) / (2.0 * var)


def pg_loss(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.0,
    kl: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-step PPO-clipped policy-gradient loss on a fixed batch.

    Args:
        new_logps: ``(batch, steps)`` log-probs under the current policy.
        old_logps: ``(batch, steps)`` log-probs stored at rollout time.
        advantages: ``(batch,)`` trajectory-level advantages, broadcast to
            every denoising step of their trajectory (the terminal reward is
            shared by all steps, as derived in Section 24.4.3).
        clip_eps: PPO clip range for the probability ratio.
        beta: weight of the reference-model KL penalty (DPOK-style).
        kl: ``(batch, steps)`` per-step KL estimates (required if beta > 0).
    """
    ratio = torch.exp(new_logps - old_logps)
    adv = advantages.unsqueeze(1)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    loss = -torch.minimum(ratio * adv, clipped * adv).mean()
    if beta > 0.0:
        if kl is None:
            raise ValueError("beta > 0 requires per-step kl estimates")
        loss = loss + beta * kl.mean()
    return loss


def mean_pairwise_distance(x: torch.Tensor, max_n: int = 256) -> float:
    """Diversity metric: mean Euclidean distance between pairs of samples."""
    n = min(len(x), max_n)
    sub = x[:n]
    dists = torch.cdist(sub, sub, p=2)
    idx = torch.triu_indices(n, n, offset=1)
    if idx.shape[1] == 0:
        return 0.0
    return dists[idx[0], idx[1]].mean().item()


# ---------------------------------------------------------------------------
# Tiny diffusion model: schedule, denoiser, pretraining, sampling.
# ---------------------------------------------------------------------------


class DiffusionSchedule:
    """Linear-beta DDPM forward process in 2D.

    Forward: ``x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps``.
    Reverse steps sample ``x_{t-1} ~ N(mu_theta(x_t, t), beta_t * I)`` with a
    *fixed* variance, so every denoising step has an analytic Gaussian
    log-probability for the policy gradient.
    """

    def __init__(self, num_steps: int, beta_start: float, beta_end: float):
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas = alphas
        self.abars = torch.cumprod(alphas, dim=0)

    def to(self, device: torch.device) -> "DiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.abars = self.abars.to(device)
        return self


class TimeEmbedding(nn.Module):
    """Fixed sinusoidal embedding of the integer timestep ``t``."""

    def __init__(self, dim: int = 16):
        super().__init__()
        freq = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("freq", freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        ang = t.float().unsqueeze(1) * self.freq.unsqueeze(0)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class DenoiserMLP(nn.Module):
    """Tiny noise predictor ``eps_theta(x_t, t)``: an MLP with time input."""

    def __init__(self, hidden: int = 64, time_dim: int = 16):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_emb(t)
        return self.net(torch.cat([x, emb], dim=1))


def sample_data(n: int, device: torch.device) -> torch.Tensor:
    """Draw ``n`` samples from the 2D mixture-of-Gaussians dataset."""
    comp = torch.randint(0, len(MODE_CENTERS), (n,), device=device)
    centers = torch.as_tensor(MODE_CENTERS, dtype=torch.float32, device=device)
    return centers[comp] + MODE_STD * torch.randn(n, 2, device=device)


def reverse_mean(
    model: DenoiserMLP, x: torch.Tensor, t: int, sched: DiffusionSchedule
) -> torch.Tensor:
    """DDPM reverse mean ``mu_theta(x_t, t)`` from the predicted noise."""
    idx = t - 1  # timesteps are 1-indexed; schedule tensors are 0-indexed
    t_batch = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
    eps_hat = model(x, t_batch)
    beta = sched.betas[idx]
    alpha = sched.alphas[idx]
    abar = sched.abars[idx]
    return (x - beta / torch.sqrt(1.0 - abar) * eps_hat) / torch.sqrt(alpha)


def pretrain(
    model: DenoiserMLP,
    sched: DiffusionSchedule,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int = 500,
) -> list[float]:
    """Denoising score matching on the toy mixture (the 'pretrained model').

    Each step: draw ``x_0`` from the data, a random timestep ``t`` and noise
    ``eps``, form the noisy ``x_t``, and regress ``eps`` with MSE.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for step in range(steps):
        x0 = sample_data(batch_size, device)
        t = torch.randint(1, sched.num_steps + 1, (batch_size,), device=device)
        abar = sched.abars[t - 1].unsqueeze(1)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(abar) * x0 + torch.sqrt(1.0 - abar) * eps
        pred = model(xt, t)
        loss = ((pred - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (step + 1) % log_every == 0 or step == 0:
            print(f"    pretrain step {step + 1}/{steps}  mse={loss.item():.4f}")
    model.eval()
    return losses


@torch.no_grad()
def rollout(
    model: DenoiserMLP, sched: DiffusionSchedule, n: int, device: torch.device
) -> dict:
    """Sample ``n`` full denoising chains; one chain = one RL trajectory.

    Returns the chain ``xs`` of shape ``(T + 1, n, 2)`` from ``x_T`` to
    ``x_0``, the per-step old log-probs ``(T, n)``, the final samples and
    their rewards.
    """
    x = torch.randn(n, 2, device=device)
    xs = [x]
    logps = []
    for t in range(sched.num_steps, 0, -1):
        mu = reverse_mean(model, x, t, sched)
        var = float(sched.betas[t - 1])
        x_next = mu + math.sqrt(var) * torch.randn_like(mu)
        logps.append(gaussian_log_prob(x_next, mu, var))
        x = x_next
        xs.append(x)
    xs_t = torch.stack(xs)  # (T+1, n, 2)
    old_logps = torch.stack(logps)  # (T, n)
    x0 = xs_t[-1]
    rewards = target_reward(x0)
    return {"xs": xs_t, "old_logps": old_logps, "x0": x0, "rewards": rewards}


def batch_logps(
    model: DenoiserMLP, sched: DiffusionSchedule, xs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute per-step log-probs and means for stored chains.

    Args:
        xs: ``(T + 1, B, 2)`` chains. Returns ``(T, B)`` log-probs of each
            stored transition under ``model`` plus the ``(T, B, 2)`` means
            (needed for the reference-model KL).
    """
    t_total = xs.shape[0] - 1
    logps: list[torch.Tensor] = []
    mus: list[torch.Tensor] = []
    for k in range(t_total):
        t = sched.num_steps - k
        x_t = xs[k]
        x_prev = xs[k + 1]
        mu = reverse_mean(model, x_t, t, sched)
        var = float(sched.betas[t - 1])
        logps.append(gaussian_log_prob(x_prev, mu, var))
        mus.append(mu)
    return torch.stack(logps), torch.stack(mus)


def ddpo_update(
    model: DenoiserMLP,
    ref_model: DenoiserMLP,
    opt: torch.optim.Optimizer,
    batch: dict,
    sched: DiffusionSchedule,
    clip_eps: float,
    beta: float,
    epochs: int,
) -> dict:
    """One DDPO update: recompute log-probs, build ratios, clipped PG loss.

    The trajectory advantage (from the terminal reward only) is broadcast to
    all ``T`` denoising steps. Gradients flow only through the sampler means;
    the reward model is a black box (never differentiated).
    """
    xs, old_logps = batch["xs"], batch["old_logps"]
    advantages = compute_advantages(batch["rewards"])
    stats: dict = {}
    for _ in range(epochs):
        new_logps_tb, mus = batch_logps(model, sched, xs)  # (T, B)
        new_logps = new_logps_tb.T  # (B, T)
        old = old_logps.T.detach()  # (B, T)
        kl = None
        if beta > 0.0:
            with torch.no_grad():
                _, ref_mus = batch_logps(ref_model, sched, xs)
            vars_tb = sched.betas.unsqueeze(1).unsqueeze(2)
            kl_tb = ((mus - ref_mus) ** 2).sum(dim=-1) / (2.0 * vars_tb.squeeze(-1))
            kl = kl_tb.T  # (B, T)
        loss = pg_loss(new_logps, old, advantages, clip_eps, beta, kl)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        stats["loss"] = loss.item()
        if kl is not None:
            stats["kl"] = kl.mean().item()
    return stats


@torch.no_grad()
def evaluate(
    model: DenoiserMLP,
    ref_model: DenoiserMLP,
    sched: DiffusionSchedule,
    n: int,
    device: torch.device,
) -> dict:
    """Report reward, diversity and KL estimate for the current policy."""
    out = rollout(model, sched, n, device)
    rewards = out["rewards"]
    diversity = mean_pairwise_distance(out["x0"])
    # KL estimate: mean per-step KL between policy and reference along the
    # sampled chains (a scalar summary of how far the policy has drifted).
    _, mus = batch_logps(model, sched, out["xs"])
    _, ref_mus = batch_logps(ref_model, sched, out["xs"])
    vars_tb = sched.betas.unsqueeze(1).unsqueeze(2)
    kl = (((mus - ref_mus) ** 2).sum(dim=-1) / (2.0 * vars_tb.squeeze(-1))).mean()
    return {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "frac_target": (rewards > TARGET_THRESHOLD).float().mean().item(),
        "diversity": diversity,
        "kl": kl.item(),
        "x0": out["x0"].detach().cpu(),
    }


# ---------------------------------------------------------------------------
# Plotting and reporting.
# ---------------------------------------------------------------------------


def make_plots(
    eval_pre: dict,
    eval_kl: dict,
    eval_nokl: dict | None,
    hist_kl: list,
    hist_nokl: list,
    outdir: str,
) -> None:
    """Scatter plots before/after RL plus reward / diversity / KL curves.

    When the no-KL comparison run is skipped (``eval_nokl is None``), only
    the pretrained and KL panels are drawn so the figure never labels
    pretrained samples as an RL run.
    """
    tx, ty = TARGET_CENTER
    panels = [(eval_pre, "Pretrained"), (eval_kl, "RL + KL penalty")]
    if eval_nokl is not None:
        panels.append((eval_nokl, "RL, no KL (beta=0)"))
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.6),
                             sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, (ev, title) in zip(axes, panels):
        pts = ev["x0"].numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.35)
        ax.scatter([tx], [ty], marker="*", s=160, c="red", label="reward peak")
        circ = plt.Circle((tx, ty), REWARD_WIDTH, color="red", fill=False, ls="--")
        ax.add_patch(circ)
        ax.set_title(f"{title}\nmean r={ev['reward_mean']:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    axes[0].set_ylabel("$x_2$")
    for ax in axes:
        ax.set_xlabel("$x_1$")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Samples before and after DDPO fine-tuning (red star = reward peak)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "samples_before_after.png"), dpi=150)
    plt.close(fig)

    iters_kl = [h["iter"] for h in hist_kl]
    iters_nokl = [h["iter"] for h in hist_nokl]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(iters_kl, [h["reward_mean"] for h in hist_kl], label="KL run")
    if hist_nokl:
        axes[0].plot(iters_nokl, [h["reward_mean"] for h in hist_nokl], label="no-KL run")
    axes[0].set_title("Mean reward")
    axes[0].set_xlabel("RL iter")
    axes[0].legend(fontsize=8)
    axes[1].plot(iters_kl, [h["diversity"] for h in hist_kl], label="KL run")
    if hist_nokl:
        axes[1].plot(iters_nokl, [h["diversity"] for h in hist_nokl], label="no-KL run")
    axes[1].set_title("Diversity (mean pairwise dist)")
    axes[1].set_xlabel("RL iter")
    axes[1].legend(fontsize=8)
    axes[2].plot(iters_kl, [h["kl"] for h in hist_kl], label="KL run")
    if hist_nokl:
        axes[2].plot(iters_nokl, [h["kl"] for h in hist_nokl], label="no-KL run")
    axes[2].set_title("KL to pretrained (per step)")
    axes[2].set_xlabel("RL iter")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("DDPO training curves")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "training_curves.png"), dpi=150)
    plt.close(fig)


def print_summary_table(rows: list[dict]) -> None:
    """Print a short summary table of the three checkpoints to stdout."""
    print("")
    print(f"  {'checkpoint':<16s} {'mean reward':>11s} {'frac target':>11s} "
          f"{'diversity':>9s} {'KL/step':>8s}")
    print(f"  {'----------':<16s} {'-----------':>11s} {'-----------':>11s} "
          f"{'---------':>9s} {'--------':>8s}")
    for r in rows:
        print(f"  {r['name']:<16s} {r['reward_mean']:>11.4f} {r['frac_target']:>11.4f} "
              f"{r['diversity']:>9.4f} {r['kl']:>8.5f}")


# ---------------------------------------------------------------------------
# Training driver.
# ---------------------------------------------------------------------------


def finetune(
    base: DenoiserMLP,
    sched: DiffusionSchedule,
    args: argparse.Namespace,
    device: torch.device,
    beta: float,
    tag: str,
) -> tuple[DenoiserMLP, list[dict]]:
    """DDPO fine-tuning of a fresh copy of ``base``; returns model + history."""
    model = copy.deepcopy(base)
    ref = copy.deepcopy(base)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_rl)
    history: list[dict] = []
    eval_every = max(1, args.rl_iters // 6)
    for it in range(1, args.rl_iters + 1):
        batch = rollout(model, sched, args.batch_rl, device)
        stats = ddpo_update(
            model, ref, opt, batch, sched, args.clip_eps, beta, args.epochs
        )
        if it % eval_every == 0 or it == args.rl_iters:
            ev = evaluate(model, ref, sched, args.eval_n, device)
            ev["iter"] = it
            ev["loss"] = stats["loss"]
            history.append({k: v for k, v in ev.items() if k != "x0"})
            print(f"    [{tag}] iter {it}/{args.rl_iters}  "
                  f"reward={ev['reward_mean']:.4f} frac_target={ev['frac_target']:.3f} "
                  f"diversity={ev['diversity']:.3f} kl={ev['kl']:.5f}")
    return model, history


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDPO toy lab for Chapter 24.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--smoke", action="store_true",
                   help="tiny config finishing in <60 s on CPU")
    p.add_argument("--pretrain-steps", type=int, default=4000)
    p.add_argument("--batch-pretrain", type=int, default=512)
    p.add_argument("--rl-iters", type=int, default=60)
    p.add_argument("--batch-rl", type=int, default=256)
    p.add_argument("--diffusion-steps", type=int, default=30)
    p.add_argument("--beta-start", type=float, default=0.001)
    p.add_argument("--beta-end", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.05,
                   help="KL penalty weight for the KL run (no-KL run uses 0)")
    p.add_argument("--lr-pretrain", type=float, default=2e-3)
    p.add_argument("--lr-rl", type=float, default=3e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=2,
                   help="gradient epochs per rollout batch")
    p.add_argument("--eval-n", type=int, default=2048)
    p.add_argument("--outdir", type=str, default="runs/chapter24_multimodal_frontier")
    p.add_argument("--skip-nokl", action="store_true",
                   help="skip the beta=0 comparison run")
    args = p.parse_args(argv)
    if args.smoke:
        args.pretrain_steps = 300
        args.batch_pretrain = 256
        args.rl_iters = 4
        args.batch_rl = 64
        args.diffusion_steps = 8
        args.beta_start = 0.01
        args.beta_end = 0.35
        args.epochs = 1
        args.eval_n = 512
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    t_start = time.time()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"DDPO toy lab (Chapter 24): seed={args.seed} device={device} "
          f"smoke={args.smoke}")
    print(f"  pretrain_steps={args.pretrain_steps} rl_iters={args.rl_iters} "
          f"batch_rl={args.batch_rl} T={args.diffusion_steps} beta={args.beta}")

    outdir = args.outdir
    if not os.path.isabs(outdir):
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        outdir = os.path.join(root, outdir)
    os.makedirs(outdir, exist_ok=True)

    sched = DiffusionSchedule(
        args.diffusion_steps, args.beta_start, args.beta_end
    ).to(device)

    print("  [1/4] pretraining tiny diffusion model...")
    base = DenoiserMLP().to(device)
    pretrain(base, sched, args.pretrain_steps, args.batch_pretrain,
             args.lr_pretrain, device)

    print("  [2/4] evaluating pretrained model...")
    ref = copy.deepcopy(base)
    eval_pre = evaluate(base, ref, sched, args.eval_n, device)
    print(f"    pretrained: reward={eval_pre['reward_mean']:.4f} "
          f"diversity={eval_pre['diversity']:.3f}")

    print(f"  [3/4] DDPO fine-tune with KL penalty (beta={args.beta})...")
    model_kl, hist_kl = finetune(base, sched, args, device, args.beta, "kl")
    eval_kl = evaluate(model_kl, ref, sched, args.eval_n, device)

    if args.skip_nokl:
        model_nokl, hist_nokl, eval_nokl = None, [], None
    else:
        print("  [4/4] DDPO fine-tune without KL penalty (beta=0)...")
        model_nokl, hist_nokl = finetune(base, sched, args, device, 0.0, "no-kl")
        eval_nokl = evaluate(model_nokl, ref, sched, args.eval_n, device)

    # Save metrics.
    with open(os.path.join(outdir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "iter", "reward_mean",
                                          "reward_std", "frac_target",
                                          "diversity", "kl", "loss"])
        w.writeheader()
        w.writerow({"run": "pretrained", "iter": 0,
                    "reward_mean": eval_pre["reward_mean"],
                    "reward_std": eval_pre["reward_std"],
                    "frac_target": eval_pre["frac_target"],
                    "diversity": eval_pre["diversity"], "kl": 0.0, "loss": ""})
        for h in hist_kl:
            w.writerow({"run": "kl", **h})
        for h in hist_nokl:
            w.writerow({"run": "nokl", **h})

    make_plots(eval_pre, eval_kl, eval_nokl, hist_kl, hist_nokl, outdir)

    elapsed = time.time() - t_start
    summary = {
        "config": {k: v for k, v in vars(args).items()},
        "elapsed_s": elapsed,
        "pretrained": {k: v for k, v in eval_pre.items() if k != "x0"},
        "rl_kl": {k: v for k, v in eval_kl.items() if k != "x0"},
        "rl_nokl": (
            {k: v for k, v in eval_nokl.items() if k != "x0"}
            if eval_nokl is not None
            else None
        ),
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    rows = [
        {"name": "pretrained", **{k: v for k, v in eval_pre.items() if k != "x0"}},
        {"name": "rl+kl", **{k: v for k, v in eval_kl.items() if k != "x0"}},
    ]
    if eval_nokl is not None:
        rows.append(
            {"name": "rl no-kl", **{k: v for k, v in eval_nokl.items() if k != "x0"}}
        )
    else:
        print("  (no-KL run skipped via --skip-nokl)")
    print_summary_table(rows)
    print(f"\nDone in {elapsed:.1f} s. Outputs in {outdir}")
    return summary


if __name__ == "__main__":
    main()
