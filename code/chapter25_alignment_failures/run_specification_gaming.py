"""Part 1: specification gaming in the loophole gridworld.

Trains tabular Q-learning twice from the same starting conditions -- once
under the loophole (``proxy``) reward where a bonus tile near the start
respawns on every visit, once under the ``fixed`` reward where the bonus
pays only once and potential-based shaping points toward the goal -- and
compares what training optimised (proxy return) against what the designer
wanted (goal-reaching success rate).

This is the Chapter 25, Section 25.1 diagnostic in miniature: reward hacking
is declared exactly when the proxy curve rises while the independent task
metric stalls or drops.

Outputs (under ``runs/chapter25_alignment_failures/``):
    spec_gaming_metrics.csv   per-episode proxy return / success for both modes
    spec_gaming_curves.png     proxy-return and success-rate training curves

Usage:
    .venv/bin/python code/chapter25_alignment_failures/run_specification_gaming.py --smoke
    .venv/bin/python code/chapter25_alignment_failures/run_specification_gaming.py --episodes 1500
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld import (
    LoopholeGridworld,
    count_bonus_visits,
    evaluate_greedy,
    train_q_learning,
)

DEFAULT_OUTDIR = os.path.join("runs", "chapter25_alignment_failures")


def moving_average(x: list[float], window: int) -> np.ndarray:
    """Trailing moving average for smoothing training curves."""
    arr = np.asarray(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Specification-gaming gridworld lab (Ch. 25, part 1).")
    parser.add_argument("--episodes", type=int, default=400, help="Training episodes per reward mode.")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Greedy evaluation episodes.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Q-learning step size.")
    parser.add_argument("--epsilon", type=float, default=0.15, help="Epsilon-greedy exploration rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Accepted for CLI uniformity; compute is numpy-only (CPU).")
    parser.add_argument("--smoke", action="store_true", help="Tiny configuration finishing in under ~60 s on CPU.")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory for CSV/PNG artefacts.")
    return parser.parse_args(argv)


def run(episodes: int, eval_episodes: int, alpha: float, epsilon: float, gamma: float, seed: int) -> dict:
    """Train under both reward modes and evaluate proxy vs. true success."""
    results: dict = {"modes": {}}
    for mode in ("proxy", "fixed"):
        env = LoopholeGridworld(reward_mode=mode, gamma=gamma)
        q, history = train_q_learning(
            env, episodes=episodes, alpha=alpha, epsilon=epsilon, gamma=gamma, seed=seed
        )
        eval_stats = evaluate_greedy(q, reward_mode=mode, eval_episodes=eval_episodes)
        farming = count_bonus_visits(q, reward_mode="proxy")
        tail = max(1, episodes // 10)
        results["modes"][mode] = {
            "q": q,
            "history": history,
            "eval": eval_stats,
            "bonus_visits": farming,
            # Behavioural (exploratory) statistics over the final stretch of
            # training: this is the "training curve" view of Section 25.1,
            # where reward hacking shows up as high proxy + zero task success.
            "train_proxy_tail": float(np.mean(history["proxy_return"][-tail:])),
            "train_success_tail": float(np.mean(history["success"][-tail:])),
        }
    return results


def save_metrics_csv(results: dict, episodes: int, path: str) -> None:
    """Write per-episode training curves for both reward modes."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "mode", "proxy_return", "success", "length"])
        for mode in ("proxy", "fixed"):
            hist = results["modes"][mode]["history"]
            for ep in range(episodes):
                writer.writerow(
                    [ep, mode, hist["proxy_return"][ep], hist["success"][ep], hist["length"][ep]]
                )


def save_curves_png(results: dict, episodes: int, path: str) -> None:
    """Plot smoothed proxy-return and success-rate curves for both modes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    window = max(5, episodes // 30)
    for mode, color in (("proxy", "#c62828"), ("fixed", "#2e7d32")):
        hist = results["modes"][mode]["history"]
        axes[0].plot(
            moving_average(hist["proxy_return"], window), color=color, label=f"{mode} proxy return"
        )
        axes[1].plot(
            moving_average(hist["success"], window), color=color, label=f"{mode} success rate"
        )
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Proxy return (smoothed)")
    axes[0].set_title("What training optimised: proxy return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Goal-reaching rate (smoothed)")
    axes[1].set_title("What the designer wanted: task success")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Specification gaming: proxy reward rises while task success stalls")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(results: dict) -> None:
    """Print the proxy-vs-success comparison table to stdout."""
    print("=" * 78)
    print("  Part 1 -- Specification gaming: proxy reward vs. true task success")
    print("=" * 78)
    print(f"  {'mode':<8s} {'train proxy':>11s} {'train succ':>10s} {'eval proxy':>10s} {'eval succ':>9s} {'bonus/ep':>8s}")
    print("-" * 78)
    for mode in ("proxy", "fixed"):
        info = results["modes"][mode]
        print(
            f"  {mode:<8s} {info['train_proxy_tail']:>11.2f} {info['train_success_tail']:>10.2f}"
            f" {info['eval']['mean_proxy_return']:>10.2f} {info['eval']['success_rate']:>9.2f}"
            f" {info['bonus_visits']:>8.2f}"
        )
    print("-" * 78)
    proxy = results["modes"]["proxy"]
    fixed = results["modes"]["fixed"]
    print(f"  'train ...' = behavioural mean over the final 10% of training episodes;")
    print(f"  'eval ...'  = greedy policy without exploration.")
    print(f"  Proxy training: success {proxy['train_success_tail']:.2f} (train) / "
          f"{proxy['eval']['success_rate']:.2f} (eval) at proxy return "
          f"{proxy['train_proxy_tail']:.2f} (train) / {proxy['eval']['mean_proxy_return']:.2f} (eval),")
    print(f"  visiting the bonus {proxy['bonus_visits']:.1f} times per greedy episode.")
    print(f"  Fixed training: success {fixed['train_success_tail']:.2f} (train) / "
          f"{fixed['eval']['success_rate']:.2f} (eval).")
    print("  Takeaway: under the loophole reward the agent scores highly while never")
    print("  reaching the goal -- never rank policies by training reward alone.")
    print("=" * 78)


def main(argv: list[str] | None = None) -> dict:
    """Entry point: train, evaluate, save artefacts, print the summary."""
    args = parse_args(argv)
    if args.smoke:
        args.episodes = min(args.episodes, 60)
        args.eval_episodes = min(args.eval_episodes, 30)
    os.makedirs(args.outdir, exist_ok=True)

    results = run(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        alpha=args.alpha,
        epsilon=args.epsilon,
        gamma=args.gamma,
        seed=args.seed,
    )
    save_metrics_csv(results, args.episodes, os.path.join(args.outdir, "spec_gaming_metrics.csv"))
    save_curves_png(results, args.episodes, os.path.join(args.outdir, "spec_gaming_curves.png"))
    print_summary(results)
    return results


if __name__ == "__main__":
    main()
