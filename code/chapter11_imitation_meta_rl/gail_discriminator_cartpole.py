"""
Chapter 11 lab (companion): a minimal GAIL-style discriminator reward on CartPole-v1.

Section 11.2.3 trains a discriminator D(s, a) to tell expert state-action
pairs from the learner's own, then uses  -log(1 - D(s, a))  as an implicit
reward for the policy. This script demonstrates the *discriminator half* of
that loop, which is supervised and trains reliably in seconds:

  1. Collect expert (s, a) pairs (the PD controller from
     ``bc_dagger_cartpole.py``) and learner (s, a) pairs (a weak BC policy
     trained on a single trajectory).
  2. Train D(s, a) with binary cross-entropy (expert = 1, learner = 0).
  3. Report held-out accuracy and the mean implicit reward -log(1-D) on
     expert versus learner pairs -- the same *pattern* as the table in
     Section 11.2.3.2 (pairs the expert visits often get high reward,
     learner-only pairs get low reward), but not its numbers: the table
     uses the optimal discriminator D*, while the trained discriminator
     here is far from optimal (accuracy ~0.74/0.82).

The *policy-optimization half* (REINFORCE/PPO against the learned reward)
is deliberately left out: on a CPU time budget it is unstable (GAN-style
dynamics, Section 11.2.4.1) -- a from-scratch REINFORCE policy on the
discriminator reward stalls below return 100 after dozens of iterations,
while GAIL in the literature needs millions of environment steps. See the
README for details.

How to run (from the repository root):
    .venv/bin/python code/chapter11_imitation_meta_rl/gail_discriminator_cartpole.py
    .venv/bin/python code/chapter11_imitation_meta_rl/gail_discriminator_cartpole.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_OUTDIR = REPO_ROOT / "runs" / "chapter11_imitation_meta_rl"

sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(THIS_DIR))
from device_utils import resolve_torch_device  # noqa: E402

from bc_dagger_cartpole import (  # noqa: E402
    PolicyNet,
    collect_expert_trajectories,
    evaluate,
    rollout_states,
    seed_all,
    train_bc,
)


class Discriminator(nn.Module):
    """Binary classifier D(s, a): one logit, expert = 1, learner = 0."""

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, sa: torch.Tensor) -> torch.Tensor:
        return self.net(sa).squeeze(-1)


def encode(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Concatenate state with one-hot action: the discriminator input."""
    oh = np.zeros((len(actions), 2), dtype=np.float32)
    oh[np.arange(len(actions)), actions] = 1.0
    return np.concatenate([states, oh], axis=1)


def implicit_reward(disc: Discriminator, states: np.ndarray, actions: np.ndarray, device) -> np.ndarray:
    """GAIL implicit reward r = -log(1 - D(s, a)) used to train the policy."""
    disc.eval()
    with torch.no_grad():
        sa = torch.as_tensor(encode(states, actions), dtype=torch.float32, device=device)
        d = torch.sigmoid(disc(sa)).clamp(1e-6, 1 - 1e-6)
        return (-torch.log(1 - d)).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ch11 lab: GAIL-style discriminator reward")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)

    n_expert, n_steps = (4, 150) if args.smoke else (10, 2000)
    t0 = time.time()

    # Learner data: a deliberately weak BC policy (1 trajectory) as in the main lab.
    s_bc, a_bc, _ = collect_expert_trajectories(1, start_seed=1000 + args.seed)
    learner = PolicyNet().to(device)
    train_bc(learner, s_bc, a_bc, device, epochs=5 if args.smoke else 200)
    print(f"weak-learner return: {np.mean(evaluate(learner, 3, 20000, device)):.1f}", flush=True)

    s_exp, a_exp, _ = collect_expert_trajectories(n_expert, start_seed=3000 + args.seed)
    s_pol, a_pol, _ = rollout_states(learner, 6 if args.smoke else 10, start_seed=4000 + args.seed, device=device)
    print(f"expert pairs: {len(s_exp)}, learner pairs: {len(s_pol)}", flush=True)

    # Held-out split for both sources.
    def split(s, a, frac=0.8):
        n = int(len(s) * frac)
        return (s[:n], a[:n]), (s[n:], a[n:])

    (tr_es, tr_ea), (te_es, te_ea) = split(s_exp, a_exp)
    (tr_ps, tr_pa), (te_ps, te_pa) = split(s_pol, a_pol)

    disc = Discriminator().to(device)
    opt = torch.optim.Adam(disc.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    xe = torch.as_tensor(encode(tr_es, tr_ea), dtype=torch.float32, device=device)
    xp = torch.as_tensor(encode(tr_ps, tr_pa), dtype=torch.float32, device=device)
    ye = torch.ones(len(xe), device=device)
    yp = torch.zeros(len(xp), device=device)
    for _ in range(n_steps):
        ie = torch.randperm(len(xe), device=device)[:512]
        ip = torch.randperm(len(xp), device=device)[:512]
        loss = bce(disc(xe[ie]), ye[ie]) + bce(disc(xp[ip]), yp[ip])
        opt.zero_grad()
        loss.backward()
        opt.step()

    disc.eval()
    with torch.no_grad():
        def acc(s, a, target):
            sa = torch.as_tensor(encode(s, a), dtype=torch.float32, device=device)
            pred = (torch.sigmoid(disc(sa)) > 0.5).float().cpu().numpy()
            return float((pred == target).mean())

        acc_exp = acc(te_es, te_ea, 1.0)
        acc_pol = acc(te_ps, te_pa, 0.0)
    r_exp = implicit_reward(disc, te_es, te_ea, device)
    r_pol = implicit_reward(disc, te_ps, te_pa, device)
    print(f"discriminator accuracy: expert {acc_exp:.3f}, learner {acc_pol:.3f}", flush=True)
    print(f"mean implicit reward -log(1-D): expert {r_exp.mean():.3f}, learner {r_pol.mean():.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["expert pairs", "learner pairs"], [r_exp.mean(), r_pol.mean()], color=["#2E7D32", "#C62828"])
    ax.set_ylabel("mean implicit reward -log(1-D)")
    ax.set_title("GAIL discriminator reward separates expert from learner")
    fig.tight_layout()
    fig.savefig(outdir / "gail_discriminator_reward.png", dpi=150)
    plt.close(fig)

    summary = {
        "acc_expert": acc_exp,
        "acc_learner": acc_pol,
        "mean_reward_expert": float(r_exp.mean()),
        "mean_reward_learner": float(r_pol.mean()),
        "runtime_s": time.time() - t0,
    }
    with open(outdir / "gail_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {outdir / 'gail_metrics.json'}, {outdir / 'gail_discriminator_reward.png'}", flush=True)
    print(f"runtime: {summary['runtime_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
