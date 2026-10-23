"""Step 2 of the Chapter 10 lab: offline learning from a fixed dataset.

For one fixed dataset (or all of them) this script trains, with *zero*
environment interaction:

1. **Behaviour cloning (BC)** -- supervised cross-entropy on (s, a) pairs.
   Copies the average behaviour; cannot exceed the dataset.
2. **Naive offline DQN** -- standard DQN loss
   ``(Q(s,a) - [r + gamma * max_a' Q_target(s',a')])^2`` computed only on
   fixed batches. The ``max`` may query actions never seen in the data, so
   Q-values can blow up on out-of-distribution (OOD) actions.
3. **Discrete CQL** -- naive DQN plus the conservative penalty from
   Section 10.1 (Kumar et al. 2020, adapted to a discrete action space):

   .. math:: alpha * (E_s[logsumexp_a Q(s,a)] - E_(s,a)~D[Q(s,a_data)])

   The ``logsumexp`` softly maximises over *all* actions (including OOD
   ones) and pushes them down, while dataset actions are pushed up.

Each trained policy is then evaluated with real online rollouts purely for
*measurement* (the training itself never touches the environment).

Outputs (under ``runs/chapter10_offline_rl/``)
----------------------------------------------
- ``offline/summary.json`` / ``offline/summary.csv``: return of every
  (dataset, method) pair.
- ``offline/<dataset>_<method>_q.pt``: trained weights (e.g.
  ``offline/random_cql_q.pt``), so extension exercises can fine-tune them.
- ``offline/q_curves_<dataset>.png``: mean max-Q on dataset states during
  training (watch naive DQN inflate vs CQL staying bounded).
- ``offline/returns.png``: bar chart of return vs dataset quality.

Example:
    python train_offline.py --dataset all --seed 0 --device cpu
    python train_offline.py --dataset random --smoke --seed 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from offline_common import (
    QMLP,
    PolicyMLP,
    default_outdir,
    evaluate_greedy,
    load_dataset,
    resolve_device,
    seed_everything,
)

METHODS = ("bc", "dqn", "cql")


def run_seed(base_seed: int, *parts: str) -> int:
    """Stable per-(dataset, method) seed so init order cannot change results.

    Network initialisation draws from the global torch RNG, which earlier
    code left shared across datasets and methods: running ``--dataset expert``
    alone gave a different expert Q-network than running ``--dataset all``.
    Deriving an independent seed per (dataset, method) from ``base_seed``
    makes every run reproducible regardless of processing order.
    """
    h = hashlib.sha256(("|".join([str(base_seed), *parts])).encode()).digest()
    return int.from_bytes(h[:4], "little") % (2**31)


# ---------------------------------------------------------------------------
# CQL penalty (unit-tested in test_ch10_lab.py)
# ---------------------------------------------------------------------------


def cql_penalty(q_values: torch.Tensor, actions: torch.Tensor, alpha: float) -> torch.Tensor:
    """Discrete CQL regularizer for one batch.

    ``mean(logsumexp_a Q(s,a) - Q(s, a_data))`` scaled by ``alpha``.
    Positive when OOD actions score above the actions seen in the data.
    """
    logsumexp = torch.logsumexp(q_values, dim=1)
    q_data = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    return (alpha * (logsumexp - q_data)).mean()


# ---------------------------------------------------------------------------
# Offline trainers (no environment interaction)
# ---------------------------------------------------------------------------


def _batches(arrays: dict, batch_size: int, steps: int, rng: np.random.Generator):
    n = len(arrays["states"])
    for _ in range(steps):
        yield rng.integers(0, n, size=batch_size)


def train_bc(
    arrays: dict, steps: int, batch_size: int, seed: int, device: torch.device,
    hidden: int = 64, lr: float = 3e-3,
) -> tuple[PolicyMLP, list[float]]:
    """Supervised cloning of the behaviour policy (cross-entropy loss)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    state_dim = arrays["states"].shape[1]
    action_dim = int(arrays["actions"].max()) + 1
    net = PolicyMLP(state_dim, action_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses: list[float] = []
    for idx in _batches(arrays, batch_size, steps, rng):
        s = torch.as_tensor(arrays["states"][idx], device=device)
        a = torch.as_tensor(arrays["actions"][idx], dtype=torch.int64, device=device)
        loss = nn.functional.cross_entropy(net(s), a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return net, losses


def train_q_offline(
    arrays: dict,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    alpha: float = 0.0,
    hidden: int = 64,
    lr: float = 1e-3,
    gamma: float = 0.99,
    target_sync_every: int = 100,
    eval_states: torch.Tensor | None = None,
) -> tuple[QMLP, dict]:
    """Naive offline DQN (``alpha=0``) or discrete CQL (``alpha>0``).

    Both minimise the fixed-data Bellman MSE; CQL adds
    :func:`cql_penalty`. Returns the Q-network plus training curves.
    """
    rng = np.random.default_rng(seed + 7919)
    torch.manual_seed(seed)
    state_dim = arrays["states"].shape[1]
    action_dim = int(arrays["actions"].max()) + 1
    q_net = QMLP(state_dim, action_dim, hidden=hidden).to(device)
    target_net = QMLP(state_dim, action_dim, hidden=hidden).to(device)
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=lr)

    hist = {"bellman": [], "cql": [], "max_q": []}
    for step, idx in enumerate(_batches(arrays, batch_size, steps, rng)):
        s = torch.as_tensor(arrays["states"][idx], device=device)
        a = torch.as_tensor(arrays["actions"][idx], dtype=torch.int64, device=device)
        r = torch.as_tensor(arrays["rewards"][idx], device=device)
        ns = torch.as_tensor(arrays["next_states"][idx], device=device)
        d = torch.as_tensor(arrays["dones"][idx], device=device)
        q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            y = r + gamma * (1.0 - d) * target_net(ns).max(dim=1)[0]
        bellman = nn.functional.mse_loss(q_sa, y)
        cql = cql_penalty(q_net(s), a, alpha)
        loss = bellman + cql
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        opt.step()
        if (step + 1) % target_sync_every == 0:
            target_net.load_state_dict(q_net.state_dict())
        if (step + 1) % 200 == 0 or step == 0:
            hist["bellman"].append(float(bellman.item()))
            hist["cql"].append(float(cql.item()))
            with torch.no_grad():
                ref = s if eval_states is None else eval_states
                hist["max_q"].append(float(q_net(ref).max(dim=1)[0].mean().item()))
    return q_net, hist


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline BC / DQN / CQL on fixed datasets.")
    p.add_argument("--dataset", default="all",
                   choices=["all", "random", "medium", "expert", "mixed"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--smoke", action="store_true", help="tiny config finishing in <60 s on CPU")
    p.add_argument("--steps", type=int, default=15000, help="gradient steps per method")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--cql-alpha", type=float, default=1.0)
    p.add_argument("--target-sync-every", type=int, default=100,
                   help="target-network sync interval in gradient steps")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def bc_as_q(policy: PolicyMLP) -> QMLP:
    """Wrap a BC policy so it can be evaluated by :func:`evaluate_greedy`.

    Builds a Q-network whose greedy action always equals the BC action by
    mapping logits directly to Q-values (argmax-preserving).
    """
    state_dim = policy.net[0].in_features
    action_dim = policy.net[-1].out_features

    class _Wrap(QMLP):
        def __init__(self, inner: PolicyMLP):
            super().__init__(state_dim, action_dim)
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.inner(x)

    return _Wrap(policy)


def run_one_dataset(
    name: str, outdir: str, args: argparse.Namespace, device: torch.device
) -> tuple[dict, dict]:
    arrays, meta = load_dataset(os.path.join(outdir, "datasets", f"{name}.npz"))
    n_eval_states = min(2000, len(arrays["states"]))
    # NOTE: these are the first 2000 *training* states, not held-out states.
    eval_states = torch.as_tensor(
        arrays["states"][:n_eval_states], dtype=torch.float32, device=device
    )
    results: dict = {"behaviour_return": meta.get("mean_episode_return")}
    curves: dict = {}
    offline_dir = os.path.join(outdir, "offline")

    # 1. Behaviour cloning.
    bc_seed = run_seed(args.seed, name, "bc")
    bc_net, bc_losses = train_bc(arrays, args.steps, args.batch_size,
                                 bc_seed, device)
    bc_eval = evaluate_greedy(bc_as_q(bc_net), n_episodes=args.eval_episodes,
                              base_seed=50_000 + args.seed, device=str(device))
    results["bc"] = bc_eval.as_dict()
    curves["bc_loss"] = bc_losses[:: max(1, len(bc_losses) // 60)]
    torch.save(bc_net.state_dict(), os.path.join(offline_dir, f"{name}_bc.pt"))

    # 2. Naive offline DQN vs 3. discrete CQL.
    for method, alpha in (("dqn", 0.0), ("cql", args.cql_alpha)):
        q_seed = run_seed(args.seed, name, method)
        q_net, hist = train_q_offline(
            arrays, args.steps, args.batch_size, q_seed, device,
            alpha=alpha, eval_states=eval_states,
            target_sync_every=args.target_sync_every,
        )
        ev = evaluate_greedy(q_net, n_episodes=args.eval_episodes,
                             base_seed=60_000 + args.seed, device=str(device))
        results[method] = ev.as_dict()
        curves[f"{method}_max_q"] = hist["max_q"]
        curves[f"{method}_bellman"] = hist["bellman"]
        torch.save(q_net.state_dict(), os.path.join(offline_dir, f"{name}_{method}_q.pt"))
        if method == "cql":
            curves["cql_penalty"] = hist["cql"]

    print(f"[offline] dataset={name:6s} behaviour={results['behaviour_return']:.1f} | "
          + " | ".join(f"{m}={results[m]['mean']:.1f}+/-{results[m]['std']:.1f}" for m in METHODS))
    return results, curves


def plot_q_curves(curves: dict, datasets: list[str], path: str) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, name in zip(axes, datasets):
        c = curves[name]
        xs = np.arange(len(c["dqn_max_q"]))
        ax.plot(xs, c["dqn_max_q"], label="naive DQN max-Q")
        ax.plot(xs, c["cql_max_q"], label="CQL max-Q")
        ax.set_title(f"dataset: {name}")
        ax.set_xlabel("checkpoint (x200 steps)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mean max_a Q(s,a) on dataset states")
    fig.suptitle("Offline Q-values on dataset states (divergence = OOD overestimation)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_returns(summary: dict, datasets: list[str], path: str) -> None:
    x = np.arange(len(datasets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(["behaviour"] + list(METHODS)):
        vals = [summary[d][m]["mean"] if m != "behaviour" else summary[d]["behaviour_return"]
                for d in datasets]
        ax.bar(x + (i - 1.5) * width, vals, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("episode return")
    ax.set_title("Offline return vs dataset quality (CartPole-v1, 500 max)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = 400
        args.eval_episodes = 4
        args.batch_size = 128
        args.target_sync_every = 50
    seed_everything(args.seed)
    device = resolve_device(args.device)
    outdir = args.outdir or default_outdir()
    offline_dir = os.path.join(outdir, "offline")
    os.makedirs(offline_dir, exist_ok=True)
    t0 = time.time()

    datasets = ["random", "medium", "expert", "mixed"] if args.dataset == "all" else [args.dataset]
    summary: dict = {}
    all_curves: dict = {}
    for name in datasets:
        results, curves = run_one_dataset(name, outdir, args, device)
        for key in ("dqn_max_q", "cql_max_q"):
            if curves.get(key):
                results[key + "_final"] = curves[key][-1]
        summary[name] = results
        all_curves[name] = curves

    with open(os.path.join(offline_dir, "summary.json"), "w") as f:
        json.dump({"seed": args.seed, "steps": args.steps,
                   "cql_alpha": args.cql_alpha, "results": summary}, f, indent=2)
    with open(os.path.join(offline_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "behaviour_return", "bc_mean", "bc_std",
                    "dqn_mean", "dqn_std", "cql_mean", "cql_std"])
        for d in datasets:
            r = summary[d]
            w.writerow([d, f"{r['behaviour_return']:.1f}",
                        f"{r['bc']['mean']:.1f}", f"{r['bc']['std']:.1f}",
                        f"{r['dqn']['mean']:.1f}", f"{r['dqn']['std']:.1f}",
                        f"{r['cql']['mean']:.1f}", f"{r['cql']['std']:.1f}"])

    plot_q_curves(all_curves, datasets, os.path.join(offline_dir, "q_curves.png"))
    plot_returns(summary, datasets, os.path.join(offline_dir, "returns.png"))

    print(f"\n[offline] summary table (return, {args.eval_episodes} eval episodes each):")
    print(f"{'dataset':8s} {'behav':>7s} {'BC':>12s} {'naive DQN':>12s} {'CQL':>12s}")
    for d in datasets:
        r = summary[d]
        print(f"{d:8s} {r['behaviour_return']:7.1f} "
              f"{r['bc']['mean']:5.1f}+/-{r['bc']['std']:<4.1f} "
              f"{r['dqn']['mean']:5.1f}+/-{r['dqn']['std']:<4.1f} "
              f"{r['cql']['mean']:5.1f}+/-{r['cql']['std']:<4.1f}")
    print(f"[offline] done in {time.time() - t0:.1f}s. Files under {offline_dir}/")


if __name__ == "__main__":
    main()
