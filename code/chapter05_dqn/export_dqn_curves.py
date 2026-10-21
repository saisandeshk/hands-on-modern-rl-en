"""Export Chapter 5 DQN evaluation curves for the VitePress docs.

The training scripts write `eval/eval_metrics.csv` and `summary.json` under
their run directories. This helper turns those CSV files into stable image
assets under `docs/chapter05_dqn/images`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RUNS = {
    "lunarlander": {
        "csv": Path("output/dqn_gym_runs/LunarLander-v3/eval/eval_metrics.csv"),
        "output": Path("docs/chapter05_dqn/images/dqn-lunarlander-100k-eval-curve.png"),
        "title": "LunarLander-v3 DQN evaluation",
        "color": "#2563EB",
        "fill": "#93C5FD",
    },
    "pong": {
        "csv": Path("output/dqn_atari_long/PongNoFrameskip-v4_dqn_seed0_10m_zoo_aligned/eval/eval_metrics.csv"),
        "output": Path("docs/chapter05_dqn/images/dqn-atari-pong-training-curve.png"),
        "title": "PongNoFrameskip-v4 DQN evaluation",
        "color": "#DC2626",
        "fill": "#FCA5A5",
    },
}


def read_eval_csv(path: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    timesteps = np.array([float(row["timesteps"]) for row in rows])
    mean_rewards = np.array([float(row["mean_reward"]) for row in rows])
    std_rewards = np.array([float(row["std_reward"]) for row in rows])
    return timesteps, mean_rewards, std_rewards


def export_curve(csv_path: Path, output_path: Path, title: str, color: str, fill: str) -> None:
    timesteps, mean_rewards, std_rewards = read_eval_csv(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(timesteps, mean_rewards, color=color, linewidth=2.4, marker="o", markersize=4)
    ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, color=fill, alpha=0.28)
    ax.set_title(title)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation mean reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Chapter 5 DQN evaluation curves.")
    parser.add_argument(
        "--run",
        choices=["all", *DEFAULT_RUNS.keys()],
        default="all",
        help="Which default run to export.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (unused, kept for CLI consistency)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test mode: export a tiny synthetic curve under runs/",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Override the input CSV")
    parser.add_argument("--output", type=Path, default=None, help="Override the output image")
    return parser.parse_args()


def write_tiny_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timesteps", "mean_reward", "std_reward", "mean_ep_length", "std_ep_length"])
        for step in (100, 200):
            writer.writerow([step, 20.0 + step / 100.0, 1.0, 100.0, 5.0])


def main() -> None:
    args = parse_args()
    if args.smoke:
        tiny_csv = Path("runs/smoke_export_dqn/eval_metrics.csv")
        write_tiny_csv(tiny_csv)
        export_curve(tiny_csv, Path("runs/smoke_export_dqn/eval_curve.png"),
                     "smoke DQN evaluation", "#2563EB", "#93C5FD")
        return
    if args.csv is not None or args.output is not None:
        if args.csv is None or args.output is None:
            raise SystemExit("--csv and --output must be given together")
        export_curve(args.csv, args.output, "DQN evaluation", "#2563EB", "#93C5FD")
        return
    items = DEFAULT_RUNS.items() if args.run == "all" else [(args.run, DEFAULT_RUNS[args.run])]
    for _, cfg in items:
        export_curve(cfg["csv"], cfg["output"], cfg["title"], cfg["color"], cfg["fill"])


if __name__ == "__main__":
    main()
