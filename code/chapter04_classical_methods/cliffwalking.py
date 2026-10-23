"""Experiment 3: the classic SARSA vs Q-learning cliff result.

On CliffWalking, both methods act epsilon-greedily (``eps=0.1``), but their
targets differ: SARSA bootstraps from the next *behaviour* action, so its
values account for the exploration wobble near the cliff and it settles on
the safer path; Q-learning bootstraps from the *greedy* next action, so it
aims at the optimal cliff-edge path while paying for more falls during
training (Sutton & Barto, Example 6.6).

We report both views: the online (behaviour) return during learning and
the greedy-policy return evaluated alongside. Q-learning should win the
second and lose the first.

How to run (from the repository root)::

    .venv/bin/python code/chapter04_classical_methods/cliffwalking.py
    .venv/bin/python code/chapter04_classical_methods/cliffwalking.py --smoke
"""

import argparse
import csv
import os
import time

import numpy as np

from tabular_common import (
    add_common_args,
    ensure_outdir,
    epsilon_greedy_action,
    evaluate_greedy,
    finalize_plot,
    make_env,
    plot_mean_band,
    resolve_device,
    save_json,
    seed_all,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Grid geometry of CliffWalking: 4 rows x 12 columns, start at (3, 0).
N_ROWS, N_COLS = 4, 12
START = (3, 0)
GOAL = (3, 11)
CLIFF = {(3, c) for c in range(1, 11)}
ARROWS = {0: "^", 1: ">", 2: "v", 3: "<"}  # gymnasium CliffWalking action order


def run_cliff(algo, n_episodes, alpha, epsilon, gamma, seed,
              eval_every, n_eval, max_steps=200):
    """Train one seed of SARSA or Q-learning on CliffWalking.

    Returns per-episode online returns, cliff-fall counts, and periodic
    greedy-policy evaluation returns.

    CliffWalking has no time limit, so episodes are capped at ``max_steps``
    (hitting the cap is treated as truncation: the update still
    bootstraps, the rollout just stops). The cap almost never binds once
    the agent moves purposefully; it only guards against degenerate
    always-up-style loops early in training.
    """
    from mc_td_control import qlearning_update, sarsa_update

    rng = seed_all(seed)
    env, env_id = make_env("CliffWalking-v1")
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    online = np.zeros(n_episodes)
    falls = np.zeros(n_episodes, dtype=int)
    checkpoints, greedy_means = [], []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed * 1000003 + ep)
        action = epsilon_greedy_action(Q, int(state), epsilon, rng)
        total, fell = 0.0, 0
        done = False
        steps = 0
        while not done:
            ns, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if float(reward) <= -100:
                fell += 1
            steps += 1
            capped = steps >= max_steps
            step_done = terminated or truncated or capped
            # Termination ends the episode for real; truncation (gym signal
            # or our step cap) only cuts the rollout short, so the update
            # must still bootstrap from the next state. Sample the next
            # behaviour action whenever the update bootstraps; it is unused
            # on true termination but required on truncation steps for
            # SARSA to stay on-policy.
            bootstrapped_done = terminated and not truncated and not capped
            next_action = (
                epsilon_greedy_action(Q, int(ns), epsilon, rng)
                if not bootstrapped_done
                else 0
            )
            if algo == "sarsa":
                sarsa_update(Q, int(state), action, float(reward), int(ns),
                             next_action, bootstrapped_done, alpha, gamma)
            else:
                qlearning_update(Q, int(state), action, float(reward), int(ns),
                                 bootstrapped_done, alpha, gamma)
            state, action = ns, next_action
            done = step_done
        online[ep] = total
        falls[ep] = fell
        if (ep + 1) % eval_every == 0 or ep + 1 == n_episodes:
            checkpoints.append(ep + 1)
            greedy_means.append(
                evaluate_greedy(env_id, Q, n_eval, seed * 7919 + ep,
                                gamma, max_steps)["mean_return"]
            )
    env.close()
    return {
        "online": online,
        "falls": falls,
        "checkpoints": np.array(checkpoints),
        "greedy_means": np.array(greedy_means),
        "final_Q": Q,
    }


def render_path(Q) -> str:
    """ASCII map of the greedy policy: S start, G goal, C cliff, arrows else."""
    lines = []
    for r in range(N_ROWS):
        row = []
        for c in range(N_COLS):
            if (r, c) == START:
                row.append("S")
            elif (r, c) == GOAL:
                row.append("G")
            elif (r, c) in CLIFF:
                row.append("C")
            else:
                row.append(ARROWS[int(np.argmax(Q[r * N_COLS + c]))])
        lines.append("".join(row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="SARSA vs Q-learning on CliffWalking")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--n-eval", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    add_common_args(parser)
    args = parser.parse_args()

    if args.smoke:
        args.n_episodes = min(args.n_episodes, 40)
        args.n_seeds = min(args.n_seeds, 2)
        args.eval_every = min(args.eval_every, 10)
        args.n_eval = min(args.n_eval, 3)

    device = resolve_device(args.device)
    outdir = ensure_outdir(args.outdir)
    print(f"device={device} seed={args.seed} smoke={args.smoke}")

    seeds = [args.seed * 1000 + i for i in range(args.n_seeds)]
    t0 = time.time()
    results = {}
    for algo in ("sarsa", "qlearning"):
        results[algo] = [
            run_cliff(algo, args.n_episodes, args.alpha, args.epsilon,
                      args.gamma, sd, args.eval_every, args.n_eval,
                      args.max_steps)
            for sd in seeds
        ]
        print(f"  {algo:>9s}: done {args.n_seeds} seeds x {args.n_episodes} "
              f"episodes ({time.time() - t0:.1f}s elapsed)")

    checkpoints = results["sarsa"][0]["checkpoints"].astype(float)

    csv_path = os.path.join(outdir, "cliff_curves.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algo", "seed", "episode", "online_return", "falls"])
        for algo in ("sarsa", "qlearning"):
            for sd, out in zip(seeds, results[algo]):
                for ep, (ret, fell) in enumerate(zip(out["online"], out["falls"])):
                    writer.writerow([algo, sd, ep + 1, f"{ret:.1f}", fell])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for algo, color in (("sarsa", "#2980b9"), ("qlearning", "#8e44ad")):
        on = np.array([o["online"] for o in results[algo]])
        plot_mean_band(axes[0],
                       np.arange(1, args.n_episodes + 1, dtype=float),
                       on.mean(0), on.std(0), algo.upper(), color)
    axes[0].set_xlabel("episodes")
    axes[0].set_ylabel("online return per episode")
    axes[0].set_title("Behaviour return during learning (higher = fewer falls)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for algo, color in (("sarsa", "#2980b9"), ("qlearning", "#8e44ad")):
        gm = np.array([o["greedy_means"] for o in results[algo]])
        plot_mean_band(axes[1], checkpoints, gm.mean(0), gm.std(0),
                       algo.upper() + " greedy", color)
    axes[1].set_xlabel("episodes")
    axes[1].set_ylabel("greedy-policy mean return")
    axes[1].set_title("Greedy-policy return (optimal = -13)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    finalize_plot(fig, os.path.join(outdir, "cliff_comparison.png"))

    print("=" * 72)
    print(f"  CliffWalking: SARSA vs Q-learning "
          f"({args.n_seeds} seeds x {args.n_episodes} episodes)")
    print(f"  {'algo':<10s} {'online(ret)':>16s} {'greedy(ret)':>16s} "
          f"{'falls/ep':>10s}")
    summary = {}
    for algo in ("sarsa", "qlearning"):
        on = np.array([o["online"] for o in results[algo]])
        gm = np.array([o["greedy_means"] for o in results[algo]])
        fa = np.array([o["falls"].sum() / args.n_episodes for o in results[algo]])
        summary[algo] = {
            "online_mean": float(on.mean()),
            "online_std": float(on.std()),
            "final_greedy_mean": float(gm[:, -1].mean()),
            "final_greedy_std": float(gm[:, -1].std()),
            "falls_per_episode": float(fa.mean()),
        }
        print(f"  {algo.upper():<10s} {on.mean():>10.2f}+/-{on.std():<5.2f} "
              f"{gm[:, -1].mean():>10.2f}+/-{gm[:, -1].std():<5.2f} "
              f"{fa.mean():>10.3f}")
    print("=" * 72)
    # Show one seed's greedy path per method so the safer/optimal split is visible.
    for algo in ("sarsa", "qlearning"):
        print(f"  {algo.upper()} greedy path (seed {seeds[0]}):")
        for line in render_path(results[algo][0]["final_Q"]).splitlines():
            print(f"    {line}")
        summary[algo]["path_seed0"] = render_path(results[algo][0]["final_Q"])
    summary["config"] = {
        "gamma": args.gamma, "alpha": args.alpha, "epsilon": args.epsilon,
        "n_episodes": args.n_episodes, "n_seeds": args.n_seeds,
        "device": device, "seed": args.seed, "seconds": time.time() - t0,
    }
    save_json(summary, os.path.join(outdir, "cliff_summary.json"))
    print(f"  metrics saved to {outdir}/cliff_curves.csv, cliff_summary.json")


if __name__ == "__main__":
    main()
