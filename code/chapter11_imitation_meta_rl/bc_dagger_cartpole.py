"""
Chapter 11 lab: behaviour cloning versus DAgger on CartPole-v1.

What this lab shows (see docs ``11.1 Behavior Cloning and Interactive
Imitation Learning``):
  1. A hand-written expert controller solves CartPole-v1 (return 500).
  2. Behaviour cloning (BC) turns the expert demonstrations into supervised
     learning: minimise  -E[log pi(a|s)]  over expert (s, a) pairs. With few
     demonstrations the learner agrees with the expert on most *expert*
     states yet scores far below it -- the compounding-error /
     distribution-shift effect of Section 11.1.2 (error O(T^2 eps)).
  3. DAgger (Section 11.1.3) rolls out the learner, asks the expert to label
     the visited states, aggregates and retrains. It reaches expert return
     with far fewer expert labels than BC, i.e. the O(T eps) behaviour.

How to run (from the repository root):
    .venv/bin/python code/chapter11_imitation_meta_rl/bc_dagger_cartpole.py
    .venv/bin/python code/chapter11_imitation_meta_rl/bc_dagger_cartpole.py --smoke

Outputs (metrics JSON/CSV and PNG plots) go to
``runs/chapter11_imitation_meta_rl/`` (gitignored); a summary table is
printed to stdout. Full run takes ~10 min on a laptop CPU; ``--smoke``
finishes in under ~60 s.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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
from device_utils import resolve_torch_device  # noqa: E402


# ---------------------------------------------------------------------------
# Part 1: the expert -- a small deterministic PD controller.
# ---------------------------------------------------------------------------
def expert_act(obs: np.ndarray) -> int:
    """Hand-written CartPole expert: push in the direction of the lean.

    obs = [cart position x, cart velocity, pole angle, pole angular
    velocity]. The control law ``u = th + 0.3*thd + 0.05*x + 0.05*xd``
    pushes the cart toward the side the pole leans to (proportional term
    on the angle), damps the angular velocity (derivative term), and keeps
    the cart near the centre (small terms on x). It is deterministic, needs
    no training, and scores 500/500 on every seed -- so it can stand in for
    the "expert" of Section 11.1 and can label any state DAgger visits.
    """
    x, xd, th, thd = (float(v) for v in obs)
    u = 1.0 * th + 0.3 * thd + 0.05 * x + 0.05 * xd
    return 1 if u > 0 else 0


# ---------------------------------------------------------------------------
# Part 2: the learner -- a small MLP policy trained from scratch with torch.
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    """MLP policy: obs (4) -> 64 -> 64 -> action logits (2)."""

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: torch.device) -> int:
        logits = self.forward(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        return int(logits.argmax(dim=1).item())


def bc_loss_from_logits(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Behaviour-cloning loss: cross-entropy = -mean(log pi(a|s)).

    This is exactly the objective L_BC of Section 11.1.1: the negative
    log-likelihood of the expert action under the learner policy.
    """
    return nn.functional.cross_entropy(logits, actions)


def train_bc(
    policy: PolicyNet,
    states: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> float:
    """Supervised training of the policy on (state, expert-action) pairs."""
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    s_all = torch.as_tensor(states, dtype=torch.float32, device=device)
    a_all = torch.as_tensor(actions, dtype=torch.long, device=device)
    n = len(s_all)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            loss = bc_loss_from_logits(policy(s_all[idx]), a_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    policy.eval()
    with torch.no_grad():
        final = bc_loss_from_logits(policy(s_all), a_all).item()
    return final


def label_with_expert(states: np.ndarray) -> np.ndarray:
    """Ask the expert for the correct action in each visited state.

    This is step 2 of DAgger: relabel *learner-visited* states (including
    failure states) with the expert action.
    """
    return np.array([expert_act(s) for s in states], dtype=np.int64)


# ---------------------------------------------------------------------------
# Part 3: environment interaction helpers (gymnasium, no RL libraries).
# ---------------------------------------------------------------------------
def _make_env():
    import gymnasium as gym

    return gym.make("CartPole-v1")


def collect_expert_trajectories(n_trajs: int, start_seed: int):
    """Roll out the expert; return (states, actions, returns, n_labels)."""
    states, actions, returns = [], [], []
    for k in range(n_trajs):
        env = _make_env()
        obs, _ = env.reset(seed=start_seed + k)
        done, total = False, 0.0
        while not done:
            a = expert_act(np.asarray(obs, dtype=np.float32))
            states.append(np.asarray(obs, dtype=np.float32))
            actions.append(a)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        returns.append(total)
        env.close()
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), returns


def rollout_states(policy: PolicyNet, n_trajs: int, start_seed: int, device: torch.device):
    """Roll out the *learner* (DAgger step 1); return visited states/actions."""
    states, actions, returns = [], [], []
    for k in range(n_trajs):
        env = _make_env()
        obs, _ = env.reset(seed=start_seed + k)
        done, total = False, 0.0
        while not done:
            obs_a = np.asarray(obs, dtype=np.float32)
            a = policy.act(obs_a, device)
            states.append(obs_a)
            actions.append(a)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        returns.append(total)
        env.close()
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), returns


def evaluate(policy: PolicyNet, n_eps: int, start_seed: int, device: torch.device):
    """Greedy (argmax) evaluation of the learner; returns list of returns."""
    returns = []
    for k in range(n_eps):
        env = _make_env()
        obs, _ = env.reset(seed=start_seed + k)
        done, total = False, 0.0
        while not done:
            a = policy.act(np.asarray(obs, dtype=np.float32), device)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        returns.append(total)
        env.close()
    return returns


@torch.no_grad()
def agreement(policy: PolicyNet, states: np.ndarray, expert_actions: np.ndarray, device: torch.device) -> float:
    """Fraction of states where the learner's argmax action matches the expert."""
    logits = policy(torch.as_tensor(states, dtype=torch.float32, device=device))
    pred = logits.argmax(dim=1).cpu().numpy()
    return float((pred == expert_actions).mean()) if len(states) else float("nan")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Part 4: experiments.
# ---------------------------------------------------------------------------
def run_bc_sweep(seeds, traj_counts, epochs, eval_eps, device):
    """BC for each demo count; reports return + agreement on expert vs own states."""
    rows = []
    for seed in seeds:
        # Held-out expert states for the "agreement on expert states" metric.
        test_s, test_a, _ = collect_expert_trajectories(5, start_seed=50_000 + seed)
        for n in traj_counts:
            seed_all(seed)
            s, a, _ = collect_expert_trajectories(n, start_seed=1_000 * seed)
            policy = PolicyNet().to(device)
            train_bc(policy, s, a, device, epochs=epochs)
            rets = evaluate(policy, eval_eps, start_seed=20_000 + seed, device=device)
            agree_exp = agreement(policy, test_s, test_a, device)
            own_s, _, _ = rollout_states(policy, 5, start_seed=30_000 + seed, device=device)
            agree_own = agreement(policy, own_s, label_with_expert(own_s), device)
            rows.append(
                {
                    "method": "bc",
                    "seed": seed,
                    "n_trajs": n,
                    "n_labels": len(s),
                    "return_mean": float(np.mean(rets)),
                    "return_std": float(np.std(rets)),
                    "agree_expert_states": agree_exp,
                    "agree_own_states": agree_own,
                }
            )
            print(
                f"  [BC] seed={seed} trajs={n:3d} labels={len(s):5d} "
                f"return={np.mean(rets):6.1f}+/-{np.std(rets):5.1f} "
                f"agree(exp)={agree_exp:.3f} agree(own)={agree_own:.3f}",
                flush=True,
            )
    return rows


def run_dagger(seeds, n_init_trajs, n_rollouts_per_iter, n_iters, epochs, eval_eps, device):
    """DAgger from a BC start: iterate rollout -> expert relabel -> aggregate -> retrain."""
    rows = []
    for seed in seeds:
        seed_all(seed)
        s_agg, a_agg, _ = collect_expert_trajectories(n_init_trajs, start_seed=1_000 * seed)
        for it in range(n_iters + 1):
            policy = PolicyNet().to(device)  # retrain from scratch on the aggregate
            train_bc(policy, s_agg, a_agg, device, epochs=epochs)
            rets = evaluate(policy, eval_eps, start_seed=20_000 + seed, device=device)
            rows.append(
                {
                    "method": "dagger",
                    "seed": seed,
                    "iter": it,
                    "n_labels": len(s_agg),
                    "return_mean": float(np.mean(rets)),
                    "return_std": float(np.std(rets)),
                }
            )
            print(
                f"  [DAgger] seed={seed} iter={it} labels={len(s_agg):5d} "
                f"return={np.mean(rets):6.1f}+/-{np.std(rets):5.1f}",
                flush=True,
            )
            if it == n_iters:
                break
            # DAgger steps 1+2: learner rollouts relabelled by the expert, then aggregated.
            own_s, _, _ = rollout_states(policy, n_rollouts_per_iter, start_seed=40_000 + seed * 997 + it * 131, device=device)
            s_agg = np.concatenate([s_agg, own_s])
            a_agg = np.concatenate([a_agg, label_with_expert(own_s)])
    return rows


def save_plots(bc_rows, dagger_rows, outdir: Path) -> None:
    bc = [r for r in bc_rows]
    dg = [r for r in dagger_rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: return vs number of expert trajectories (BC compounding curve).
    ax = axes[0]
    ns = sorted({r["n_trajs"] for r in bc})
    means = [np.mean([r["return_mean"] for r in bc if r["n_trajs"] == n]) for n in ns]
    stds = [np.std([r["return_mean"] for r in bc if r["n_trajs"] == n]) for n in ns]
    ax.errorbar(ns, means, yerr=stds, marker="o", capsize=3)
    ax.axhline(500, color="gray", linestyle="--", alpha=0.6, label="expert")
    ax.set_xlabel("expert trajectories (BC)")
    ax.set_ylabel("mean return")
    ax.set_title("BC: return vs demonstrations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: per-step agreement on expert states vs on the learner's own states.
    ax = axes[1]
    ae = [np.mean([r["agree_expert_states"] for r in bc if r["n_trajs"] == n]) for n in ns]
    ao = [np.mean([r["agree_own_states"] for r in bc if r["n_trajs"] == n]) for n in ns]
    ax.plot(ns, ae, marker="o", label="on expert states")
    ax.plot(ns, ao, marker="s", label="on learner's own states")
    ax.set_xlabel("expert trajectories (BC)")
    ax.set_ylabel("action agreement")
    ax.set_title("Compounding error: agreement gap")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: return vs expert-label budget, BC against DAgger.
    # (DAgger label totals vary by seed because learner episodes have
    # different lengths, so DAgger points are grouped by iteration with
    # x = mean label count across seeds.)
    ax = axes[2]
    bc_lab = sorted({r["n_labels"] for r in bc})
    ax.errorbar(
        bc_lab,
        [np.mean([r["return_mean"] for r in bc if r["n_labels"] == L]) for L in bc_lab],
        yerr=[np.std([r["return_mean"] for r in bc if r["n_labels"] == L]) for L in bc_lab],
        marker="o",
        capsize=3,
        label="BC",
    )
    dg_iters = sorted({r["iter"] for r in dg})
    dg_lab = [np.mean([r["n_labels"] for r in dg if r["iter"] == it]) for it in dg_iters]
    ax.errorbar(
        dg_lab,
        [np.mean([r["return_mean"] for r in dg if r["iter"] == it]) for it in dg_iters],
        yerr=[np.std([r["return_mean"] for r in dg if r["iter"] == it]) for it in dg_iters],
        marker="s",
        capsize=3,
        label="DAgger",
    )
    ax.axhline(500, color="gray", linestyle="--", alpha=0.6, label="expert")
    ax.set_xlabel("expert labels (state-action pairs)")
    ax.set_ylabel("mean return")
    ax.set_title("Label efficiency: BC vs DAgger")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Imitation on CartPole-v1: BC compounding error and DAgger fix")
    fig.tight_layout()
    fig.savefig(outdir / "bc_dagger_results.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ch11 lab: BC vs DAgger on CartPole-v1")
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="tiny config finishing in <60 s on CPU")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds = [args.seed]
        traj_counts = [1, 2]
        bc_epochs, dagger_iters, eval_eps, dagger_epochs = 5, 2, 2, 5
    else:
        seeds = [args.seed, args.seed + 1, args.seed + 2]
        traj_counts = [1, 2, 5, 10, 25]
        bc_epochs, dagger_iters, eval_eps, dagger_epochs = 200, 4, 10, 200

    t0 = time.time()
    print(f"device: {device}  seeds: {seeds}  smoke: {args.smoke}", flush=True)

    # Expert reference return (Section 11.1: the demonstrations to imitate).
    seed_all(args.seed)
    expert_policy_returns = []
    for k in range(10):
        _, _, r = collect_expert_trajectories(1, start_seed=9000 + k)
        expert_policy_returns.append(r[0])
    print(f"expert mean return over 10 episodes: {np.mean(expert_policy_returns):.1f}", flush=True)

    print("BC sweep ...", flush=True)
    bc_rows = run_bc_sweep(seeds, traj_counts, bc_epochs, eval_eps, device)
    print("DAgger (from a 1-trajectory BC start) ...", flush=True)
    dagger_rows = run_dagger(seeds, 1, 2, dagger_iters, dagger_epochs, eval_eps, device)

    with open(outdir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in bc_rows + dagger_rows for k in r}))
        writer.writeheader()
        writer.writerows(bc_rows + dagger_rows)
    summary = {
        "expert_mean_return": float(np.mean(expert_policy_returns)),
        "seeds": seeds,
        "bc_traj_counts": traj_counts,
        "dagger_iters": dagger_iters,
        "bc": [
            {
                "n_trajs": n,
                "return_mean": float(np.mean([r["return_mean"] for r in bc_rows if r["n_trajs"] == n])),
                "agree_expert": float(np.mean([r["agree_expert_states"] for r in bc_rows if r["n_trajs"] == n])),
                "agree_own": float(np.mean([r["agree_own_states"] for r in bc_rows if r["n_trajs"] == n])),
            }
            for n in traj_counts
        ],
        "dagger": [
            {
                "iter": it,
                "n_labels": int(np.mean([r["n_labels"] for r in dagger_rows if r["iter"] == it])),
                "return_mean": float(np.mean([r["return_mean"] for r in dagger_rows if r["iter"] == it])),
            }
            for it in sorted({r["iter"] for r in dagger_rows})
        ],
        "runtime_s": time.time() - t0,
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    save_plots(bc_rows, dagger_rows, outdir)

    print("-" * 64)
    print(f"{'trajs':>6} {'BC return':>10} {'agree(exp)':>11} {'agree(own)':>11}")
    for b in summary["bc"]:
        print(f"{b['n_trajs']:>6d} {b['return_mean']:>10.1f} {b['agree_expert']:>11.3f} {b['agree_own']:>11.3f}")
    print("DAgger: iter -> (labels, return)")
    for d in summary["dagger"]:
        print(f"  iter {d['iter']}: labels={d['n_labels']}, return={d['return_mean']:.1f}")
    print(f"wrote {outdir / 'metrics.json'}, {outdir / 'metrics.csv'}, {outdir / 'bc_dagger_results.png'}")
    print(f"runtime: {summary['runtime_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
