"""Part 2: evaluation methodology -- robust statistics over many seeds.

Trains the tabular Q-learning agent of Part 1 under the repaired (``fixed``)
reward with 10+ seeds for each of two hyperparameter settings, then compares
the settings on the *independent* metric (goal-reaching success rate) using
mean, median, IQM, stratified-bootstrap 95% confidence intervals and a
performance profile -- all implemented from scratch in ``stats.py``.

The script also prints the two dishonest summaries from Section 25.5: the
single-seed comparison and the best-3-seeds comparison, so the reader can see
directly whether cherry-picking reverses the conclusion that the full,
uncertainty-aware analysis supports.

Outputs (under ``runs/chapter25_alignment_failures/``):
    eval_scores.csv           per-seed proxy return and success rate per config
    eval_stats.json           mean / median / IQM / bootstrap CIs / cherry-pick check
    eval_profile.png          performance profiles over success rate thresholds

Usage:
    .venv/bin/python code/chapter25_alignment_failures/run_eval_methodology.py --smoke
    .venv/bin/python code/chapter25_alignment_failures/run_eval_methodology.py --seeds 12
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld import LoopholeGridworld, evaluate_greedy, train_q_learning
from stats import bootstrap_diff_ci, performance_profile, summarize

DEFAULT_OUTDIR = os.path.join("runs", "chapter25_alignment_failures")

# Two hyperparameter settings: a stable learner vs. a noisier one.
# Config B explores more (epsilon=0.25 vs 0.15) and updates faster
# (alpha=0.4 vs 0.25), so its per-seed outcomes spread more widely. The two
# distributions overlap: A is better on average, yet B owns several of the
# very best individual seeds -- exactly the situation in which reporting
# only the top seeds reverses the honest conclusion.
CONFIGS = {
    "A_stable": {"alpha": 0.25, "epsilon": 0.15},
    "B_noisy": {"alpha": 0.40, "epsilon": 0.25},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Honest-evaluation methodology lab (Ch. 25, part 2).")
    parser.add_argument("--seeds", type=int, default=12, help="Seeds per hyperparameter setting (>= 10 for the full run).")
    parser.add_argument("--episodes", type=int, default=400, help="Training episodes per seed.")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Greedy evaluation episodes per seed.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--slip", type=float, default=0.1, help="Action-noise probability (slippery floor); makes scores continuous.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; run seed s uses base + s.")
    parser.add_argument("--device", type=str, default="auto", help="Accepted for CLI uniformity; compute is numpy-only (CPU).")
    parser.add_argument("--smoke", action="store_true", help="Tiny configuration finishing in under ~60 s on CPU.")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory for CSV/JSON/PNG artefacts.")
    return parser.parse_args(argv)


def run_all(seeds: int, episodes: int, eval_episodes: int, gamma: float, slip: float, base_seed: int) -> dict[str, dict[str, list[float]]]:
    """Train every config on every seed; collect proxy and success scores."""
    scores: dict[str, dict[str, list[float]]] = {
        name: {"proxy": [], "success": []} for name in CONFIGS
    }
    for name, hp in CONFIGS.items():
        for s in range(seeds):
            seed = base_seed + s
            env = LoopholeGridworld(reward_mode="fixed", gamma=gamma, slip=slip, slip_seed=seed)
            q, _ = train_q_learning(
                env,
                episodes=episodes,
                alpha=hp["alpha"],
                epsilon=hp["epsilon"],
                gamma=gamma,
                seed=seed,
            )
            stats = evaluate_greedy(
                q, reward_mode="fixed", eval_episodes=eval_episodes, gamma=gamma, slip=slip
            )
            scores[name]["proxy"].append(stats["mean_proxy_return"])
            scores[name]["success"].append(stats["success_rate"])
    return scores


def save_scores_csv(scores: dict, path: str) -> None:
    """Write one row per (config, seed) with both metrics."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "seed", "proxy_return", "success_rate"])
        for name, vals in scores.items():
            for s, (p, u) in enumerate(zip(vals["proxy"], vals["success"])):
                writer.writerow([name, s, f"{p:.4f}", f"{u:.4f}"])


def save_profile_png(groups: dict[str, np.ndarray], path: str) -> None:
    """Plot performance profiles: P(success rate > tau) per config."""
    taus = np.linspace(0.0, 1.0, 51)
    profiles = performance_profile(groups, taus)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color in (("A_stable", "#1976d2"), ("B_noisy", "#e65100")):
        ax.plot(taus, profiles[name], color=color, linewidth=2, label=name)
    ax.set_xlabel("Success-rate threshold tau")
    ax.set_ylabel("Fraction of seeds with success rate > tau")
    ax.set_title("Performance profile (higher curve = more robust)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cherry_pick_report(groups: dict[str, np.ndarray]) -> dict:
    """Compare honest (IQM) vs. cherry-picked (best-3, single-seed) winners."""
    from stats import iqm_score

    honest = "A_stable" if iqm_score(groups["A_stable"]) >= iqm_score(groups["B_noisy"]) else "B_noisy"
    top3 = {
        name: float(np.mean(np.sort(np.asarray(x, dtype=float))[-3:]))
        for name, x in groups.items()
    }
    best3_winner = max(top3, key=lambda k: top3[k])
    single_winner = (
        "A_stable"
        if float(np.asarray(groups["A_stable"])[0]) >= float(np.asarray(groups["B_noisy"])[0])
        else "B_noisy"
    )
    # Aligned seed pairs: for how many of the 12 seed indices would a
    # single-seed paper crown a different winner than the full IQM analysis?
    # A high fraction proves the single-seed conclusion is a lottery.
    a_arr = np.asarray(groups["A_stable"], dtype=float)
    b_arr = np.asarray(groups["B_noisy"], dtype=float)
    n_pairs = min(a_arr.shape[0], b_arr.shape[0])
    pair_disagree = sum(
        1 for i in range(n_pairs) if (a_arr[i] >= b_arr[i]) != (honest == "A_stable")
    )
    # The adversarial extreme: the best seed of one config against the worst
    # seed of the other. Any desired winner can be manufactured this way,
    # which is precisely why min/max anecdotes prove nothing.
    adv_winner = "B_noisy" if float(np.max(b_arr)) >= float(np.min(a_arr)) else "A_stable"
    return {
        "honest_iqm_winner": honest,
        "best3_winner": best3_winner,
        "best3_A": top3["A_stable"],
        "best3_B": top3["B_noisy"],
        "single_seed_winner": single_winner,
        "seed_pair_disagreements": pair_disagree,
        "seed_pair_total": n_pairs,
        "adversarial_B_best_vs_A_worst": [float(np.max(b_arr)), float(np.min(a_arr))],
        "adversarial_winner": adv_winner,
        "flipped_best3": honest != best3_winner,
        "flipped_single": honest != single_winner,
    }


def print_summary(groups: dict[str, np.ndarray], summary: dict, report: dict, diff_ci: tuple) -> None:
    """Print the robust-statistics table and the cherry-picking warning."""
    print("=" * 78)
    print("  Part 2 -- Honest evaluation: 12 seeds x 2 hyperparameter settings")
    print("  Metric: goal-reaching success rate (independent of training reward)")
    print("=" * 78)
    print(f"  {'config':<10s} {'mean':>6s} {'median':>7s} {'IQM':>6s} {'95% CI (IQM)':>18s}")
    print("-" * 78)
    for name in ("A_stable", "B_noisy"):
        row = summary[name]
        print(
            f"  {name:<10s} {row['mean']:>6.3f} {row['median']:>7.3f}"
            f" {row['iqm']:>6.3f} [{row['iqm_lo']:>6.3f}, {row['iqm_hi']:>6.3f}]"
        )
    print("-" * 78)
    print(f"  IQM difference (A - B): {diff_ci[0]:+.3f}  95% CI [{diff_ci[1]:+.3f}, {diff_ci[2]:+.3f}]")
    if diff_ci[1] > 0 or diff_ci[2] < 0:
        print("  The CI excludes zero: the ranking survives seed noise.")
    else:
        print("  The CI includes zero: the ranking is NOT resolved -- report the tie,")
        print("  not a winner. (This non-result is itself the honest conclusion.)")
    print()
    print(f"  Cherry-picked best-3 means: A={report['best3_A']:.3f}  B={report['best3_B']:.3f}"
          f" -> winner: {report['best3_winner']}")
    print(f"  Single seed (seed 0):       A={float(np.asarray(groups['A_stable'])[0]):.3f}  "
          f"B={float(np.asarray(groups['B_noisy'])[0]):.3f} -> winner: {report['single_seed_winner']}")
    print(f"  Honest IQM winner: {report['honest_iqm_winner']}. "
          f"Best-3 flips it: {report['flipped_best3']}. Single-seed flips it: {report['flipped_single']}.")
    print(f"  Aligned seed pairs disagreeing with IQM: {report['seed_pair_disagreements']}/{report['seed_pair_total']} --")
    print("  a single-seed paper would be a lottery over which seed it drew.")
    blo, awo = report["adversarial_B_best_vs_A_worst"]
    print(f"  Adversarial pick (B's best {blo:.3f} vs A's worst {awo:.3f}): '{report['adversarial_winner']}' wins --")
    print("  any winner can be manufactured from min/max anecdotes.")
    print("  Takeaway: with 10+ seeds and bootstrap intervals, selective reporting")
    print("  is exposed. Always publish all seeds, the IQM/median, and the CI.")
    print("=" * 78)


def main(argv: list[str] | None = None) -> dict:
    """Entry point: run all seeds, analyse, save artefacts, print summary."""
    args = parse_args(argv)
    n_boot = 200 if args.smoke else 2000
    if args.smoke:
        args.seeds = min(args.seeds, 4)
        args.episodes = min(args.episodes, 60)
        args.eval_episodes = min(args.eval_episodes, 20)
    os.makedirs(args.outdir, exist_ok=True)

    scores = run_all(
        seeds=args.seeds,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        gamma=args.gamma,
        slip=args.slip,
        base_seed=args.seed,
    )
    groups = {name: np.asarray(vals["success"]) for name, vals in scores.items()}
    summary = summarize(groups, n_resamples=n_boot, seed=args.seed)
    diff_ci = bootstrap_diff_ci(groups["A_stable"], groups["B_noisy"], n_resamples=n_boot, seed=args.seed)
    report = cherry_pick_report(groups)

    save_scores_csv(scores, os.path.join(args.outdir, "eval_scores.csv"))
    save_profile_png(groups, os.path.join(args.outdir, "eval_profile.png"))
    payload = {
        "configs": CONFIGS,
        "n_seeds": args.seeds,
        "episodes": args.episodes,
        "summary": summary,
        "iqm_diff_A_minus_B": {"estimate": diff_ci[0], "lo": diff_ci[1], "hi": diff_ci[2]},
        "cherry_pick": {k: (bool(v) if isinstance(v, bool) else v) for k, v in report.items()},
    }
    with open(os.path.join(args.outdir, "eval_stats.json"), "w") as f:
        json.dump(payload, f, indent=2)

    print_summary(groups, summary, report, diff_ci)
    return payload


if __name__ == "__main__":
    main()
