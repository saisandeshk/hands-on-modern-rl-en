"""Experiment 2: Monte Carlo and TD control on slippery FrozenLake.

Compares four tabular control methods, all with epsilon-greedy behaviour:

1. first-visit Monte Carlo control (incremental sample-mean updates),
2. SARSA (on-policy TD: bootstraps from the next behaviour action),
3. Expected SARSA (on-policy TD: bootstraps from the expected next value),
4. Q-learning (off-policy TD: bootstraps from the greedy next value).

The exact ``Q*`` from :mod:`dp_oracle` is the oracle: besides the usual
learning curves (mean over seeds with a shaded std band) we plot the
max-norm error ``||Q - Q*||_inf``. For Q-learning this error measures
genuine convergence (its target is ``Q*``); the on-policy methods target
``Q`` of their epsilon-greedy behaviour policy instead, so their curves
bottom out at a positive floor (see the lab page). The comparison still
tells sampling noise apart from learning: a shrinking error means the
table itself is moving.

How to run (from the repository root)::

    .venv/bin/python code/chapter04_classical_methods/mc_td_control.py
    .venv/bin/python code/chapter04_classical_methods/mc_td_control.py --smoke
"""

import argparse
import csv
import os
import time

import numpy as np

from common import (
    FROZENLAKE_ID,
    add_common_args,
    ensure_outdir,
    epsilon_greedy_action,
    evaluate_greedy,
    finalize_plot,
    load_model,
    make_env,
    plot_mean_band,
    resolve_device,
    save_json,
    seed_all,
)
from dp_oracle import value_iteration

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGOS = ("mc", "sarsa", "expected_sarsa", "qlearning")
ALGO_LABELS = {
    "mc": "MC first-visit",
    "sarsa": "SARSA",
    "expected_sarsa": "Expected SARSA",
    "qlearning": "Q-learning",
}
COLORS = {
    "mc": "#e74c3c",
    "sarsa": "#2980b9",
    "expected_sarsa": "#27ae60",
    "qlearning": "#8e44ad",
}


# --------------------------------------------------------------------------
# Single-step TD updates (also used directly by the unit tests).
# --------------------------------------------------------------------------
def sarsa_update(Q, s, a, reward, ns, na, done, alpha, gamma):
    """On-policy one-step update towards ``r + gamma Q(s',a')``."""
    target = reward + (0.0 if done else gamma * Q[ns, na])
    td_error = target - Q[s, a]
    Q[s, a] += alpha * td_error
    return float(td_error)


def qlearning_update(Q, s, a, reward, ns, done, alpha, gamma):
    """Off-policy one-step update towards ``r + gamma max_a' Q(s',a')``."""
    target = reward + (0.0 if done else gamma * float(Q[ns].max()))
    td_error = target - Q[s, a]
    Q[s, a] += alpha * td_error
    return float(td_error)


def expected_sarsa_update(Q, s, a, reward, ns, done, alpha, gamma, epsilon):
    """On-policy update towards the expected next value under epsilon-greedy.

    The expectation is exact (no sampling of the next action): with
    probability ``1 - eps`` the greedy value and ``eps`` a uniform action.
    """
    if done:
        target = reward
    else:
        n_actions = Q.shape[1]
        best = Q[ns].max()
        target = reward + gamma * (
            (1.0 - epsilon) * best + epsilon * float(Q[ns].mean())
        )
    td_error = target - Q[s, a]
    Q[s, a] += alpha * td_error
    return float(td_error)


# --------------------------------------------------------------------------
# Full training runs (one seed each).
# --------------------------------------------------------------------------
def run_episode_td(env, Q, update_fn, epsilon, gamma, rng, seed):
    """Run one epsilon-greedy episode with per-step TD updates.

    ``update_fn`` implements the target of each method: SARSA uses the
    sampled next behaviour action, while Expected SARSA and Q-learning
    use the expectation / maximum over next actions instead. Returns the
    undiscounted online return. Truncation bootstraps (the episode did not
    really end); termination does not. The next behaviour action is always
    sampled when the update bootstraps (including on truncation steps),
    so SARSA stays on-policy; it is only skipped when the update target
    uses no bootstrap at all (true termination).
    """
    state, _ = env.reset(seed=seed)
    action = epsilon_greedy_action(Q, int(state), epsilon, rng)
    total = 0.0
    done = False
    while not done:
        ns, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        step_done = terminated or truncated
        # A truncated episode is cut short by the time limit, not by the
        # MDP, so the bootstrap target must still look past it. Sample the
        # next behaviour action whenever the update bootstraps; the value
        # is unused on true termination (target = r), but on a truncation
        # step SARSA needs the actual behaviour action to stay on-policy.
        bootstrapped_done = terminated and not truncated
        next_action = (
            epsilon_greedy_action(Q, int(ns), epsilon, rng)
            if not bootstrapped_done
            else 0
        )
        update_fn(Q, int(state), action, float(reward), int(ns),
                  next_action, bootstrapped_done)
        state, action = ns, next_action
        done = step_done
    return total


def train_td(algo, env_id, n_episodes, alpha, epsilon, gamma, seed,
             eval_every, n_eval, oracle_Q):
    """Train one seed of a TD method; record online, eval and oracle curves."""
    rng = seed_all(seed)
    env, _ = make_env(env_id)
    n_actions = env.action_space.n
    n_states = env.observation_space.n
    Q = np.zeros((n_states, n_actions))

    if algo == "sarsa":
        def update_fn(Q, s, a, r, ns, na, done):
            return sarsa_update(Q, s, a, r, ns, na, done, alpha, gamma)
    elif algo == "expected_sarsa":
        def update_fn(Q, s, a, r, ns, na, done):
            return expected_sarsa_update(Q, s, a, r, ns, done,
                                         alpha, gamma, epsilon)
    elif algo == "qlearning":
        def update_fn(Q, s, a, r, ns, na, done):
            return qlearning_update(Q, s, a, r, ns, done, alpha, gamma)
    else:  # pragma: no cover
        raise ValueError(algo)

    online = np.zeros(n_episodes)
    checkpoints, greedy_means, oracle_errors = [], [], []
    for ep in range(n_episodes):
        online[ep] = run_episode_td(env, Q, update_fn, epsilon, gamma, rng,
                                    seed * 1000003 + ep)
        if (ep + 1) % eval_every == 0 or ep + 1 == n_episodes:
            checkpoints.append(ep + 1)
            greedy_means.append(
                evaluate_greedy(env_id, Q, n_eval, seed * 7919 + ep)["mean_return"]
            )
            oracle_errors.append(float(np.abs(Q - oracle_Q).max()))
    env.close()
    return {
        "online": online,
        "checkpoints": np.array(checkpoints),
        "greedy_means": np.array(greedy_means),
        "oracle_errors": np.array(oracle_errors),
        "final_Q": Q,
    }


def train_mc(env_id, n_episodes, epsilon, gamma, seed,
             eval_every, n_eval, oracle_Q):
    """First-visit MC control with epsilon-greedy behaviour policies.

    Each first visit ``(s,a)`` is updated towards its full return ``G``
    with an incremental sample mean ``Q += (G - Q) / N(s,a)``. No step
    size is needed: ``1 / N`` is the learning rate.
    """
    rng = seed_all(seed)
    env, _ = make_env(env_id)
    n_actions = env.action_space.n
    n_states = env.observation_space.n
    Q = np.zeros((n_states, n_actions))
    counts = np.zeros((n_states, n_actions))

    online = np.zeros(n_episodes)
    checkpoints, greedy_means, oracle_errors = [], [], []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed * 1000003 + ep)
        trajectory = []
        done = False
        total = 0.0
        while not done:
            action = epsilon_greedy_action(Q, int(state), epsilon, rng)
            ns, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((int(state), action, float(reward)))
            total += float(reward)
            state = ns
            done = terminated or truncated
        online[ep] = total
        # Backward pass: accumulate returns, update first visits only.
        G = 0.0
        visited = set()
        for s, a, r in reversed(trajectory):
            G = r + gamma * G
            if (s, a) not in visited:
                visited.add((s, a))
                counts[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / counts[s, a]
        if (ep + 1) % eval_every == 0 or ep + 1 == n_episodes:
            checkpoints.append(ep + 1)
            greedy_means.append(
                evaluate_greedy(env_id, Q, n_eval, seed * 7919 + ep)["mean_return"]
            )
            oracle_errors.append(float(np.abs(Q - oracle_Q).max()))
    env.close()
    return {
        "online": online,
        "checkpoints": np.array(checkpoints),
        "greedy_means": np.array(greedy_means),
        "oracle_errors": np.array(oracle_errors),
        "final_Q": Q,
    }


def block_average(curve: np.ndarray, block: int) -> np.ndarray:
    """Average a per-episode curve in non-overlapping blocks."""
    n = len(curve) // block
    return curve[: n * block].reshape(n, block).mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="MC/TD control on FrozenLake")
    parser.add_argument("--env-id", type=str, default=FROZENLAKE_ID)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--n-episodes", type=int, default=20000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--algos", type=str, default="all",
                        help="comma-separated subset of mc,sarsa,expected_sarsa,qlearning")
    add_common_args(parser)
    args = parser.parse_args()

    if args.smoke:
        args.n_episodes = min(args.n_episodes, 300)
        args.n_seeds = min(args.n_seeds, 2)
        args.eval_every = min(args.eval_every, 100)
        args.n_eval = min(args.n_eval, 5)

    if args.algos == "all":
        algos = list(ALGOS)
    else:
        algos = [a.strip() for a in args.algos.split(",") if a.strip()]
        assert all(a in ALGOS for a in algos), f"unknown algo in {algos}"

    device = resolve_device(args.device)
    outdir = ensure_outdir(args.outdir)
    print(f"device={device} seed={args.seed} smoke={args.smoke} algos={algos}")

    # Exact oracle Q* (same gamma), computed once by value iteration.
    P, _, _, _, _ = load_model(args.env_id)
    _, oracle_Q, _ = value_iteration(P, args.gamma, theta=1e-8)
    print(f"  oracle Q*(start) max = {oracle_Q[0].max():.6f}")

    seeds = [args.seed * 1000 + i for i in range(args.n_seeds)]
    t0 = time.time()
    results = {}
    for algo in algos:
        per_seed = []
        for sd in seeds:
            if algo == "mc":
                out = train_mc(args.env_id, args.n_episodes, args.epsilon,
                               args.gamma, sd, args.eval_every, args.n_eval,
                               oracle_Q)
            else:
                out = train_td(algo, args.env_id, args.n_episodes, args.alpha,
                               args.epsilon, args.gamma, sd, args.eval_every,
                               args.n_eval, oracle_Q)
            per_seed.append(out)
        results[algo] = per_seed
        print(f"  {algo:>14s}: done {args.n_seeds} seeds x "
              f"{args.n_episodes} episodes ({time.time() - t0:.1f}s elapsed)")

    checkpoints = results[algos[0]][0]["checkpoints"]

    # ---- save checkpoint curves (one row per algo/seed/checkpoint) ----
    curve_path = os.path.join(outdir, "control_curves.csv")
    with open(curve_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algo", "seed", "episode",
                         "greedy_mean_return", "oracle_q_error"])
        for algo in algos:
            for sd, out in zip(seeds, results[algo]):
                for ep, g, e in zip(out["checkpoints"], out["greedy_means"],
                                    out["oracle_errors"]):
                    writer.writerow([algo, sd, ep, f"{g:.6f}", f"{e:.6f}"])
    online_path = os.path.join(outdir, "control_online.csv")
    block = max(1, args.n_episodes // 200)
    with open(online_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algo", "seed", "block", "mean_online_return"])
        for algo in algos:
            for sd, out in zip(seeds, results[algo]):
                for b, v in enumerate(block_average(out["online"], block)):
                    writer.writerow([algo, sd, b, f"{v:.6f}"])

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    xs = checkpoints.astype(float)
    for algo in algos:
        gm = np.array([o["greedy_means"] for o in results[algo]])
        plot_mean_band(axes[0], xs, gm.mean(0), gm.std(0),
                       ALGO_LABELS[algo], COLORS[algo])
    axes[0].set_xlabel("episodes")
    axes[0].set_ylabel("greedy-policy mean return")
    axes[0].set_title("Learning curves (greedy evaluation)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for algo in algos:
        oe = np.array([o["oracle_errors"] for o in results[algo]])
        plot_mean_band(axes[1], xs, oe.mean(0), oe.std(0),
                       ALGO_LABELS[algo], COLORS[algo])
    axes[1].set_xlabel("episodes")
    axes[1].set_ylabel("max |Q - Q*|")
    axes[1].set_title("Max-norm error to the DP oracle Q*")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, which="both")

    xblocks = (np.arange(len(block_average(results[algos[0]][0]["online"], block))) + 1) * block
    for algo in algos:
        ob = np.array([block_average(o["online"], block) for o in results[algo]])
        plot_mean_band(axes[2], xblocks.astype(float), ob.mean(0), ob.std(0),
                       ALGO_LABELS[algo], COLORS[algo])
    axes[2].set_xlabel("episodes")
    axes[2].set_ylabel("online return (block mean)")
    axes[2].set_title("Behaviour (epsilon-greedy) return during learning")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    finalize_plot(fig, os.path.join(outdir, "control_frozenlake.png"))

    # ---- summary table ----
    summary = {}
    print("=" * 72)
    print(f"  MC/TD control on {args.env_id} "
          f"({args.n_seeds} seeds x {args.n_episodes} episodes)")
    print(f"  {'algo':<16s} {'greedy return':>14s} {'|Q-Q*|_inf':>12s} "
          f"{'online last10%':>15s}")
    for algo in algos:
        gm = np.array([o["greedy_means"] for o in results[algo]])
        oe = np.array([o["oracle_errors"] for o in results[algo]])
        tail = np.array([o["online"][-max(1, args.n_episodes // 10):].mean()
                         for o in results[algo]])
        summary[algo] = {
            "final_greedy_mean": float(gm[:, -1].mean()),
            "final_greedy_std": float(gm[:, -1].std()),
            "final_oracle_error_mean": float(oe[:, -1].mean()),
            "final_oracle_error_std": float(oe[:, -1].std()),
            "online_tail_mean": float(tail.mean()),
        }
        print(f"  {ALGO_LABELS[algo]:<16s} "
              f"{gm[:, -1].mean():>10.4f}+/-{gm[:, -1].std():<5.3f} "
              f"{oe[:, -1].mean():>12.4f} {tail.mean():>15.4f}")
    print("=" * 72)
    summary["config"] = {
        "env_id": args.env_id, "gamma": args.gamma, "alpha": args.alpha,
        "epsilon": args.epsilon, "n_episodes": args.n_episodes,
        "n_seeds": args.n_seeds, "eval_every": args.eval_every,
        "n_eval": args.n_eval, "device": device, "seed": args.seed,
        "seconds": time.time() - t0,
    }
    save_json(summary, os.path.join(outdir, "control_summary.json"))
    print(f"  metrics saved to {outdir}/control_curves.csv, "
          f"control_online.csv, control_summary.json")


if __name__ == "__main__":
    main()
