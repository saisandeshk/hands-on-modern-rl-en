"""Chapter 26 lab, part 1: self-play dynamics in rock-paper-scissors.

Covers two games and three learning rules:

* Standard RPS (Nash equilibrium (1/3, 1/3, 1/3)) and a biased variant
  where a Rock win pays 2 instead of 1 (Nash equilibrium
  (1/4, 1/2, 1/4); derived in the module docstring of ``payoff_matrix``).
* Naive self-play by best response (cycles through pure strategies),
  simultaneous projected gradient ascent (orbits the equilibrium without
  converging), and fictitious play (both players best-respond to the
  opponent's empirical average, whose time average converges to Nash).

The RL idea: when the opponent is another copy of the learner, the data
distribution moves with the policy. Myopic updates chase the current
opponent and cycle; averaging over history (the opponent pool in
Section 26.1) stabilizes learning.

How to run:
    .venv/bin/python code/chapter26_selfplay_frontiers/rps_self_play.py --smoke
    .venv/bin/python code/chapter26_selfplay_frontiers/rps_self_play.py \
        --variant biased --dynamics all --iterations 20000 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from device_utils import resolve_torch_device

ACTION_NAMES = ("Rock", "Paper", "Scissors")


def payoff_matrix(variant: str) -> np.ndarray:
    """Row-player payoff matrix U (zero-sum, so column player gets ``-U``).

    Standard RPS: win = +1, loss = -1. The unique symmetric Nash
    equilibrium is (1/3, 1/3, 1/3) with game value 0.

    Biased RPS: a Rock win pays +2 (row 0, column 2). Indifference at a
    symmetric profile p requires ``-p_P + 2 p_S = p_R - p_S`` and
    ``p_R - p_S = -2 p_R + p_P`` with ``p_R + p_P + p_S = 1``, which
    solves to (1/4, 1/2, 1/4), again with game value 0. Intuition: Rock
    is a more potent weapon, so the equilibrium plays more Paper to
    keep Rock in check.
    """
    if variant == "standard":
        return np.array(
            [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]
        )
    if variant == "biased":
        return np.array(
            [[0.0, -1.0, 2.0], [1.0, 0.0, -1.0], [-2.0, 1.0, 0.0]]
        )
    raise ValueError(f"unknown variant {variant!r}")


def nash_equilibrium(variant: str) -> np.ndarray:
    if variant == "standard":
        return np.full(3, 1.0 / 3.0)
    if variant == "biased":
        return np.array([0.25, 0.5, 0.25])
    raise ValueError(f"unknown variant {variant!r}")


def project_simplex(vector: np.ndarray) -> np.ndarray:
    """Euclidean projection of a vector onto the probability simplex."""
    n = vector.shape[0]
    order = np.sort(vector)[::-1]
    cumulative = np.cumsum(order)
    eligible = order - (cumulative - 1.0) / np.arange(1, n + 1) > 0
    rho = int(np.nonzero(eligible)[0].max()) + 1
    threshold = (cumulative[rho - 1] - 1.0) / rho
    return np.clip(vector - threshold, 0.0, None)


def best_response_index(payoffs: np.ndarray, rng: np.random.Generator) -> int:
    """Pure best response with seeded random tie-breaking."""
    best = float(np.max(payoffs))
    candidates = np.nonzero(payoffs >= best - 1e-12)[0]
    return int(rng.choice(candidates))


def exploitability(pi: np.ndarray, sigma: np.ndarray, utility: np.ndarray) -> float:
    """Nash gap of the profile (pi, sigma): max response value minus min.

    ``max_a (U sigma)_a - min_b (pi^T U)_b``. Zero exactly at a Nash
    equilibrium; positive otherwise. For these symmetric games the game
    value is 0, so this is also the sum of both players' incentives to
    deviate.
    """
    return float(np.max(utility @ sigma) - np.min(pi @ utility))


def torch_exploitability(
    pi: np.ndarray, sigma: np.ndarray, utility: np.ndarray, device: str
) -> float:
    """Recompute exploitability with torch tensors (cross-check on device)."""
    import torch

    dev = resolve_torch_device(device)
    t_pi = torch.as_tensor(pi, dtype=torch.float64, device=dev)
    t_sigma = torch.as_tensor(sigma, dtype=torch.float64, device=dev)
    t_u = torch.as_tensor(utility, dtype=torch.float64, device=dev)
    gap = torch.max(t_u @ t_sigma) - torch.min(t_pi @ t_u)
    return float(gap.item())


def run_best_response(
    utility: np.ndarray, iterations: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Naive self-play: each side plays a pure best response to the last move."""
    pi = np.zeros(3)
    sigma = np.zeros(3)
    pi[best_response_index(utility @ np.full(3, 1.0 / 3.0), rng)] = 1.0
    sigma[best_response_index(-utility.T @ np.full(3, 1.0 / 3.0), rng)] = 1.0
    cur1 = np.zeros((iterations + 1, 3))
    cur2 = np.zeros((iterations + 1, 3))
    cur1[0] = pi
    cur2[0] = sigma
    for t in range(1, iterations + 1):
        pi = np.zeros(3)
        pi[best_response_index(utility @ sigma, rng)] = 1.0
        sigma = np.zeros(3)
        sigma[best_response_index(-utility.T @ cur1[t - 1], rng)] = 1.0
        cur1[t] = pi
        cur2[t] = sigma
    avg1 = np.cumsum(cur1, axis=0) / np.arange(1, iterations + 2)[:, None]
    avg2 = np.cumsum(cur2, axis=0) / np.arange(1, iterations + 2)[:, None]
    return {"current1": cur1, "current2": cur2, "average1": avg1, "average2": avg2}


def run_gradient(
    utility: np.ndarray, iterations: int, lr: float, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Simultaneous projected gradient ascent on each player's payoff.

    Row player ascends ``pi^T U sigma`` in pi, column player ascends
    ``-pi^T U sigma`` in sigma. This is the textbook example of a
    no-regret-in-time-average but cycling-in-last-iterate dynamic.
    Both players start far from equilibrium so the orbit is visible.
    """
    pi = project_simplex(np.array([0.6, 0.3, 0.1]) + 0.01 * rng.standard_normal(3))
    sigma = project_simplex(np.array([0.1, 0.6, 0.3]) + 0.01 * rng.standard_normal(3))
    cur1 = np.zeros((iterations + 1, 3))
    cur2 = np.zeros((iterations + 1, 3))
    cur1[0] = pi
    cur2[0] = sigma
    for t in range(1, iterations + 1):
        pi = project_simplex(pi + lr * (utility @ sigma))
        sigma = project_simplex(sigma - lr * (utility.T @ pi))
        cur1[t] = pi
        cur2[t] = sigma
    avg1 = np.cumsum(cur1, axis=0) / np.arange(1, iterations + 2)[:, None]
    avg2 = np.cumsum(cur2, axis=0) / np.arange(1, iterations + 2)[:, None]
    return {"current1": cur1, "current2": cur2, "average1": avg1, "average2": avg2}


def run_fictitious(
    utility: np.ndarray, iterations: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Fictitious play: best-respond to the opponent's empirical frequency.

    Counts start at one pseudocount of each action (Laplace smoothing) so
    the first average is uniform. Only the averages are meaningful here;
    the current strategies stay pure by construction.
    """
    counts1 = np.ones(3)
    counts2 = np.ones(3)
    cur1 = np.zeros((iterations + 1, 3))
    cur2 = np.zeros((iterations + 1, 3))
    avg1 = np.zeros((iterations + 1, 3))
    avg2 = np.zeros((iterations + 1, 3))
    avg1[0] = counts1 / counts1.sum()
    avg2[0] = counts2 / counts2.sum()
    for t in range(1, iterations + 1):
        pi = np.zeros(3)
        pi[best_response_index(utility @ avg2[t - 1], rng)] = 1.0
        sigma = np.zeros(3)
        sigma[best_response_index(-utility.T @ avg1[t - 1], rng)] = 1.0
        cur1[t] = pi
        cur2[t] = sigma
        counts1 += pi
        counts2 += sigma
        avg1[t] = counts1 / counts1.sum()
        avg2[t] = counts2 / counts2.sum()
    return {"current1": cur1, "current2": cur2, "average1": avg1, "average2": avg2}


def summarize(
    trajectory: dict[str, np.ndarray], utility: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    avg1, avg2 = trajectory["average1"][-1], trajectory["average2"][-1]
    cur1, cur2 = trajectory["current1"][-1], trajectory["current2"][-1]
    return {
        "expl_current": exploitability(cur1, cur2, utility),
        "expl_average": exploitability(avg1, avg2, utility),
        "l1_to_nash_avg1": float(np.abs(avg1 - target).sum()),
        "l1_to_nash_avg2": float(np.abs(avg2 - target).sum()),
    }


def plot_trajectory(
    trajectory: dict[str, np.ndarray],
    utility: np.ndarray,
    dynamics: str,
    variant: str,
    path: Path,
) -> None:
    iters = np.arange(trajectory["current1"].shape[0])
    expl_cur = np.array(
        [
            exploitability(trajectory["current1"][t], trajectory["current2"][t], utility)
            for t in range(0, len(iters), max(1, len(iters) // 2000))
        ]
    )
    expl_avg = np.array(
        [
            exploitability(trajectory["average1"][t], trajectory["average2"][t], utility)
            for t in range(0, len(iters), max(1, len(iters) // 2000))
        ]
    )
    sample = np.arange(0, len(iters), max(1, len(iters) // 2000))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    shown = trajectory["average1"] if dynamics == "fictitious" else trajectory["current1"]
    for k, name in enumerate(ACTION_NAMES):
        axes[0].plot(iters[:: max(1, len(iters) // 2000)], shown[:: max(1, len(iters) // 2000), k], label=name)
    axes[0].set_title(f"{dynamics}: player 1 strategy")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("probability")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(sample, np.maximum(expl_cur, 1e-12), label="current profile")
    axes[1].plot(sample, np.maximum(expl_avg, 1e-12), label="time-average profile")
    axes[1].set_yscale("log")
    axes[1].set_title("Exploitability (Nash gap)")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("gap (log scale)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"RPS self-play: {dynamics} ({variant})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RPS self-play dynamics lab")
    parser.add_argument("--variant", choices=["standard", "biased"], default="standard")
    parser.add_argument(
        "--dynamics",
        choices=["all", "best-response", "gradient", "fictitious"],
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args(argv)


def default_outdir() -> Path:
    return Path(__file__).resolve().parents[2] / "runs" / "chapter26_selfplay_frontiers" / "rps"


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.smoke:
        args.iterations = 500
    rng = np.random.default_rng(args.seed)
    import torch

    torch.manual_seed(args.seed)
    device = resolve_torch_device(args.device)
    utility = payoff_matrix(args.variant)
    target = nash_equilibrium(args.variant)
    dynamics = (
        ["best-response", "gradient", "fictitious"]
        if args.dynamics == "all"
        else [args.dynamics]
    )
    outdir = Path(args.outdir) if args.outdir else default_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"RPS self-play ({args.variant}): Nash target {target}, device {device}")
    print(f"  {'dynamics':<14}{'expl_current':>13}{'expl_average':>13}"
          f"{'L1(avg1-NE)':>13}{'L1(avg2-NE)':>13}")
    results: dict = {"variant": args.variant, "seed": args.seed, "dynamics": {}}
    for name in dynamics:
        runner = {
            "best-response": lambda: run_best_response(utility, args.iterations, rng),
            "gradient": lambda: run_gradient(utility, args.iterations, args.lr, rng),
            "fictitious": lambda: run_fictitious(utility, args.iterations, rng),
        }[name]
        traj = runner()
        stats = summarize(traj, utility, target)
        check = torch_exploitability(
            traj["average1"][-1], traj["average2"][-1], utility, str(device)
        )
        assert abs(check - stats["expl_average"]) < 1e-9, "numpy/torch gap mismatch"
        plot_trajectory(traj, utility, name, args.variant, outdir / f"{name}_{args.variant}.png")
        with open(outdir / f"metrics_{name}_{args.variant}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["iter", "pi1_R", "pi1_P", "pi1_S", "pi2_R", "pi2_P", "pi2_S",
                 "expl_current", "expl_average"]
            )
            step = max(1, args.iterations // 500)
            for t in range(0, args.iterations + 1, step):
                writer.writerow(
                    [t, *traj["current1"][t], *traj["current2"][t],
                     exploitability(traj["current1"][t], traj["current2"][t], utility),
                     exploitability(traj["average1"][t], traj["average2"][t], utility)]
                )
        print(f"  {name:<14}{stats['expl_current']:>13.4f}{stats['expl_average']:>13.4f}"
              f"{stats['l1_to_nash_avg1']:>13.4f}{stats['l1_to_nash_avg2']:>13.4f}")
        results["dynamics"][name] = {
            **stats,
            "avg1": traj["average1"][-1].tolist(),
            "avg2": traj["average2"][-1].tolist(),
        }
    with open(outdir / f"summary_{args.variant}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"outputs written to {outdir}")
    return results


if __name__ == "__main__":
    main()
