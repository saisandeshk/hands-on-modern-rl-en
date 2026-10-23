"""Train a softmax policy over candidate fixes with REINFORCE + baseline.

The loop is a tabular miniature of SWE-RL training from Chapter 20:

    sample task -> agent picks a candidate implementation (action)
      -> sandboxed tests run in a subprocess (verifier)
      -> pass rate becomes the reward -> REINFORCE update with baseline.

Three reward regimes are compared (selected with ``--reward``):

- ``visible``: train reward = pass rate on the visible tests only. Shortcut
  candidates that memorize the visible tests are optimal here, so the policy
  learns to hack: high train reward, low hidden-test pass rate.
- ``hidden``: train reward = pass rate on the held-out tests (an oracle
  skyline: this is what the reward would be if the verifier covered the real
  requirement). Only the genuinely correct candidates are optimal.
- ``randomized``: train reward = pass rate on freshly sampled test inputs
  whose expected outputs come from a reference oracle. Memorized answers do
  not transfer to unseen inputs, so this also recovers the correct
  candidates -- the toy analogue of "broaden test coverage".

Outputs (under ``runs/chapter20_swe_agents/``): per-mode CSV histories,
``summary.json``, and two PNG plots. A summary table is printed to stdout.

Examples:
    .venv/bin/python code/chapter20_swe_agents/train.py --smoke --seed 0
    .venv/bin/python code/chapter20_swe_agents/train.py --epochs 300 --seed 0
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swe_policy import SoftmaxPolicy, reinforce_step  # noqa: E402
from sandbox import run_candidate  # noqa: E402
from tasks import ORACLES, TASKS, Task, random_inputs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO_ROOT / "runs" / "chapter20_swe_agents"
MODES = ("visible", "hidden", "randomized")
HACK_KINDS = {"hack", "heuristic"}


def resolve_device(choice: str) -> torch.device:
    choice = choice.lower()
    if choice not in ("auto", "cpu", "cuda"):
        raise ValueError(f"--device must be one of auto/cpu/cuda, got {choice!r}")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if choice == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def precompute_rates(
    tasks: list[Task], timeout: float
) -> dict[tuple[str, int, str], float]:
    """Run every (task, candidate) once on visible + hidden suites.

    Candidates are fixed snippets, so these sandbox results are deterministic;
    caching them keeps training fast without changing any value the policy
    sees. Keys are (task_id, candidate_idx, suite).
    """
    rates: dict[tuple[str, int, str], float] = {}
    for task in tasks:
        for i, cand in enumerate(task.candidates):
            for suite in ("visible", "hidden"):
                cases = getattr(task, suite)
                res = run_candidate(cand.source, cases, timeout=timeout)
                rates[(task.task_id, i, suite)] = res.pass_rate
    return rates


def randomized_rate(task: Task, cand_idx: int, rng: random.Random, n: int, timeout: float) -> float:
    """Live sandbox pass rate on freshly sampled inputs (never cached)."""
    oracle = ORACLES[task.oracle_name]
    argsets = random_inputs(task.task_id, rng, n)
    cases = [(tuple(a), oracle(*a)) for a in argsets]
    return run_candidate(task.candidates[cand_idx].source, cases, timeout=timeout).pass_rate


def policy_metrics(
    policy: SoftmaxPolicy, tasks: list[Task], rates: dict[tuple[str, int, str], float]
) -> dict[str, float]:
    """Exact expectations under the current stochastic policy (no sampling noise)."""
    with torch.no_grad():
        probs = policy.probs()
    exp_vis, exp_hid, hack_mass, correct_mass, ent = 0.0, 0.0, 0.0, 0.0, 0.0
    greedy_vis, greedy_hid, resolved = 0.0, 0.0, 0.0
    for t, task in enumerate(tasks):
        p = probs[t, : task.n_candidates]
        for i, cand in enumerate(task.candidates):
            w = float(p[i].item())
            exp_vis += w * rates[(task.task_id, i, "visible")]
            exp_hid += w * rates[(task.task_id, i, "hidden")]
            if cand.kind in HACK_KINDS:
                hack_mass += w
            if cand.kind == "correct":
                correct_mass += w
        ent += float(torch.distributions.Categorical(probs=p).entropy().item())
        best = int(torch.argmax(p).item())
        greedy_vis += rates[(task.task_id, best, "visible")]
        gh = rates[(task.task_id, best, "hidden")]
        greedy_hid += gh
        resolved += 1.0 if gh == 1.0 else 0.0
    n = len(tasks)
    return {
        "exp_visible": exp_vis / n,
        "exp_hidden": exp_hid / n,
        "greedy_visible": greedy_vis / n,
        "greedy_hidden": greedy_hid / n,
        "hidden_resolved": resolved / n,
        "hack_mass": hack_mass / n,
        "correct_mass": correct_mass / n,
        "entropy": ent / n,
    }


def train_mode(
    mode: str,
    tasks: list[Task],
    rates: dict[tuple[str, int, str], float],
    epochs: int,
    lr: float,
    entropy_coef: float,
    baseline_alpha: float,
    n_random: int,
    timeout: float,
    seed: int,
    device: torch.device,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    """Run REINFORCE with a per-task running baseline under one reward regime."""
    counts = [t.n_candidates for t in tasks]
    policy = SoftmaxPolicy(counts, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    baselines = torch.zeros(len(tasks))
    history: list[dict] = []

    for epoch in range(epochs):
        log_probs, rewards, base, entropies = [], [], [], []
        train_r = 0.0
        for t, task in enumerate(tasks):
            action, logp, ent = policy.sample(t)
            if mode == "visible":
                r = rates[(task.task_id, action, "visible")]
            elif mode == "hidden":
                r = rates[(task.task_id, action, "hidden")]
            else:
                r = randomized_rate(task, action, rng, n_random, timeout)
            train_r += r
            log_probs.append(logp)
            rewards.append(r)
            base.append(baselines[t].item())
            entropies.append(ent)
        stats = reinforce_step(
            policy,
            optimizer,
            torch.stack(log_probs),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(base, dtype=torch.float32),
            entropy=torch.stack(entropies),
            entropy_coef=entropy_coef,
        )
        with torch.no_grad():
            for t, r in enumerate(rewards):
                baselines[t] += baseline_alpha * (r - baselines[t].item())
        row = {"epoch": epoch, "mode": mode, "train_reward": train_r / len(tasks)}
        row.update(policy_metrics(policy, tasks, rates))
        row["loss"] = stats["loss"]
        history.append(row)
    final = {"mode": mode, **{k: history[-1][k] for k in history[-1] if k != "epoch"}}
    return history, final


def write_csv(path: Path, history: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def make_plots(outdir: Path, all_histories: dict[str, list[dict]]) -> None:
    modes = list(all_histories.keys())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode in modes:
        xs = [r["epoch"] for r in all_histories[mode]]
        ys = [r["train_reward"] for r in all_histories[mode]]
        ax.plot(xs, ys, label=f"{mode} (train reward)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean train pass rate")
    ax.set_title("REINFORCE train reward under three verifier regimes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "train_curves.png", dpi=150)

    labels, vis, hid = [], [], []
    for mode in modes:
        labels.append(mode)
        vis.append(all_histories[mode][-1]["exp_visible"])
        hid.append(all_histories[mode][-1]["exp_hidden"])
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - 0.2, vis, 0.4, label="E[visible pass rate]")
    ax.bar(x + 0.2, hid, 0.4, label="E[hidden pass rate]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Final policy: visible vs hidden pass rates (expectation)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "visible_vs_hidden.png", dpi=150)
    plt.close("all")


def print_table(finals: dict[str, dict]) -> None:
    print("\nFinal results (expectation under the learned stochastic policy):")
    header = f"  {'mode':<12}{'train_R':>9}{'E[vis]':>9}{'E[hid]':>9}{'greedy_hid':>12}{'resolved':>10}{'P(hack)':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mode, f in finals.items():
        print(
            f"  {mode:<12}{f['train_reward']:>9.3f}{f['exp_visible']:>9.3f}"
            f"{f['exp_hidden']:>9.3f}{f['greedy_hidden']:>12.3f}"
            f"{f['hidden_resolved']:>10.3f}{f['hack_mass']:>9.3f}"
        )
    print("\nResponse to 'did it really fix the bug?': compare E[hid] and 'resolved' "
          "of visible-only training against the hidden/randomized regimes.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toy SWE-agent REINFORCE lab (Chapter 20).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--smoke", action="store_true",
                   help="tiny config finishing in <60 s on CPU")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--reward", default="all",
                   choices=["all", "visible", "hidden", "randomized"])
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--baseline-alpha", type=float, default=0.1)
    p.add_argument("--n-random", type=int, default=6,
                   help="fresh random test inputs per task per epoch")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="sandbox seconds per test batch")
    p.add_argument("--max-tasks", type=int, default=-1,
                   help="use only the first K tasks (-1 = all)")
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    if args.smoke:
        epochs, task_list = 12, TASKS[:4]
    else:
        epochs, task_list = args.epochs, TASKS
    if args.max_tasks > 0:
        task_list = TASKS[: args.max_tasks]
    modes = list(MODES) if args.reward == "all" else [args.reward]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Chapter 20 lab: {len(task_list)} tasks, {epochs} epochs, "
          f"modes={modes}, seed={args.seed}, device={device}")

    t0 = time.time()
    print("Precomputing sandbox rewards on visible/hidden suites...")
    rates = precompute_rates(task_list, timeout=args.timeout)
    rng = random.Random(args.seed)

    all_histories, finals = {}, {}
    for mode in modes:
        print(f"Training with {mode!r} rewards...")
        history, final = train_mode(
            mode, task_list, rates, epochs, args.lr, args.entropy_coef,
            args.baseline_alpha, args.n_random, args.timeout, args.seed,
            device, rng,
        )
        all_histories[mode] = history
        finals[mode] = final
        write_csv(outdir / f"metrics_{mode}.csv", history)

    make_plots(outdir, all_histories)
    summary = {
        "config": {
            "seed": args.seed, "epochs": epochs, "lr": args.lr,
            "entropy_coef": args.entropy_coef, "baseline_alpha": args.baseline_alpha,
            "n_random": args.n_random, "timeout": args.timeout,
            "tasks": [t.task_id for t in task_list], "modes": modes,
            "device": str(device),
        },
        "final": finals,
        "wall_seconds": time.time() - t0,
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done in {summary['wall_seconds']:.1f} s. Artifacts in {outdir}")
    print_table(finals)


if __name__ == "__main__":
    main()
