"""Experiment 4: off-policy evaluation with importance sampling.

A fixed deterministic target policy (the DP-optimal greedy policy on
slippery FrozenLake) is evaluated from episodes collected by a different
behaviour policy (epsilon-greedy around ``Q*``), using two estimators:

* ordinary importance sampling (OIS): ``mean(rho * G)`` — unbiased for
  full episodes, but the product weight ``rho`` over a whole trajectory
  has huge variance (and gymnasium's 100-step time limit truncates a few
  percent of episodes without bootstrapping, which adds a small downward
  bias — see the results section);
* weighted (self-normalized) importance sampling (WIS):
  ``sum(rho * G) / sum(rho)`` — biased, but much lower variance.

The exact start-state value ``V^pi(s0)`` from DP policy evaluation is the
ground truth both estimators are compared against (bias, variance, RMSE
over repeated datasets).

How to run (from the repository root)::

    .venv/bin/python code/chapter04_classical_methods/ope_is.py
    .venv/bin/python code/chapter04_classical_methods/ope_is.py --smoke
"""

import argparse
import os
import time

import numpy as np

from tabular_common import (
    FROZENLAKE_ID,
    add_common_args,
    ensure_outdir,
    epsilon_greedy_action,
    finalize_plot,
    load_model,
    make_env,
    resolve_device,
    save_json,
    seed_all,
)
from dp_oracle import iterative_policy_evaluation, value_iteration

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Estimators (also used directly by the unit tests).
# --------------------------------------------------------------------------
def ordinary_is(returns: np.ndarray, rhos: np.ndarray) -> float:
    """Ordinary IS: plain mean of the weight-corrected returns."""
    returns = np.asarray(returns, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    return float((rhos * returns).mean())


def weighted_is(returns: np.ndarray, rhos: np.ndarray) -> float:
    """Weighted (self-normalized) IS: ``sum(rho G) / sum(rho)``."""
    returns = np.asarray(returns, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    total = rhos.sum()
    if total == 0.0:
        return 0.0
    return float((rhos * returns).sum() / total)


def effective_sample_size(rhos: np.ndarray) -> float:
    """ Kish's ESS ``(sum w)^2 / sum w^2``: how many unweighted samples the
    weighted batch is worth. Tiny ESS fractions diagnose weight collapse."""
    rhos = np.asarray(rhos, dtype=float)
    denom = float((rhos**2).sum())
    if denom == 0.0:
        return 0.0
    return float(rhos.sum() ** 2 / denom)


# --------------------------------------------------------------------------
# Data collection.
# --------------------------------------------------------------------------
def collect_dataset(env_id: str, behavior_Q: np.ndarray, epsilon: float,
                    gamma: float, n_episodes: int, seed: int,
                    target_policy: np.ndarray):
    """Roll out the behaviour policy; record returns and full-trajectory weights.

    ``target_policy[s, a]`` is the target action probability. The behaviour
    action probabilities are exact (epsilon-greedy), so each step weight
    ``pi(a|s) / mu(a|s)`` is exact and the cumulative product ``rho`` is the
    correct trajectory correction.
    """
    rng = seed_all(seed)
    env, _ = make_env(env_id)
    n_actions = behavior_Q.shape[1]
    returns = np.zeros(n_episodes)
    rhos = np.zeros(n_episodes)
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed * 100003 + ep)
        G, rho, t = 0.0, 1.0, 0
        done = False
        while not done:
            action = epsilon_greedy_action(behavior_Q, int(state), epsilon, rng)
            greedy = int(np.argmax(behavior_Q[int(state)]))
            mu = (1.0 - epsilon) * (1.0 if action == greedy else 0.0) + epsilon / n_actions
            pi = float(target_policy[int(state), action])
            rho *= pi / mu
            ns, reward, terminated, truncated, _ = env.step(action)
            G += (gamma**t) * float(reward)
            t += 1
            state = ns
            done = terminated or truncated
        returns[ep] = G
        rhos[ep] = rho
    env.close()
    return returns, rhos


def main() -> None:
    parser = argparse.ArgumentParser(description="OPE with OIS/WIS on FrozenLake")
    parser.add_argument("--env-id", type=str, default=FROZENLAKE_ID)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eps-behavior", type=float, default=0.1,
                        help="epsilon of the eps-greedy(Q*) behaviour policy; "
                             "small enough that whole trajectories still match "
                             "the deterministic target with non-trivial probability")
    parser.add_argument("--n-episodes", type=int, default=2000,
                        help="behaviour episodes per dataset trial")
    parser.add_argument("--n-trials", type=int, default=100)
    add_common_args(parser)
    args = parser.parse_args()

    if args.smoke:
        args.n_episodes = min(args.n_episodes, 50)
        args.n_trials = min(args.n_trials, 3)

    device = resolve_device(args.device)
    outdir = ensure_outdir(args.outdir)
    print(f"device={device} seed={args.seed} smoke={args.smoke}")

    P, n_states, n_actions, _, resolved = load_model(args.env_id)
    _, oracle_Q, _ = value_iteration(P, args.gamma, theta=1e-8)

    # Fixed target policy: deterministic greedy policy w.r.t. Q*.
    target_policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        target_policy[s, int(np.argmax(oracle_Q[s]))] = 1.0
    exact_V, _ = iterative_policy_evaluation(P, target_policy, args.gamma, theta=1e-10)
    exact_start = float(exact_V[0])
    print(f"  target = DP-optimal greedy policy; exact V^pi(s0) = {exact_start:.6f}")
    print(f"  behaviour = eps-greedy(Q*, eps={args.eps_behavior})")

    t0 = time.time()
    ois_estimates, wis_estimates, ess_frac = [], [], []
    for trial in range(args.n_trials):
        returns, rhos = collect_dataset(
            resolved, oracle_Q, args.eps_behavior, args.gamma,
            args.n_episodes, args.seed * 10000 + trial, target_policy,
        )
        ois_estimates.append(ordinary_is(returns, rhos))
        wis_estimates.append(weighted_is(returns, rhos))
        ess_frac.append(effective_sample_size(rhos) / args.n_episodes)
        if (trial + 1) % max(1, args.n_trials // 5) == 0:
            print(f"  trial {trial + 1}/{args.n_trials} "
                  f"({time.time() - t0:.1f}s elapsed)")
    ois_estimates = np.array(ois_estimates)
    wis_estimates = np.array(wis_estimates)
    ess_frac = np.array(ess_frac)

    def stats(est: np.ndarray) -> dict:
        bias = float(est.mean() - exact_start)
        var = float(est.var())
        return {"mean": float(est.mean()), "std": float(est.std()),
                "bias": bias, "variance": var,
                "rmse": float(np.sqrt(bias**2 + var))}

    summary = {
        "env_id": resolved,
        "exact_V_start": exact_start,
        "ois": stats(ois_estimates),
        "wis": stats(wis_estimates),
        "mean_ess_fraction": float(ess_frac.mean()),
        "config": {
            "gamma": args.gamma, "eps_behavior": args.eps_behavior,
            "n_episodes": args.n_episodes, "n_trials": args.n_trials,
            "device": device, "seed": args.seed,
            "seconds": time.time() - t0,
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = np.arange(1, args.n_trials + 1)
    axes[0].scatter(xs, ois_estimates, s=18, color="#e74c3c", alpha=0.7,
                    label="ordinary IS")
    axes[0].scatter(xs, wis_estimates, s=18, color="#2980b9", alpha=0.7,
                    label="weighted IS")
    axes[0].axhline(exact_start, color="black", linestyle="--",
                    label=f"exact V(s0)={exact_start:.4f}")
    axes[0].set_xlabel("dataset trial")
    axes[0].set_ylabel("estimated V(s0)")
    axes[0].set_title("Per-trial OPE estimates (same behaviour policy)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    labels = ["bias$^2$", "variance"]
    ois_vals = [summary["ois"]["bias"] ** 2, summary["ois"]["variance"]]
    wis_vals = [summary["wis"]["bias"] ** 2, summary["wis"]["variance"]]
    x = np.arange(len(labels))
    axes[1].bar(x - 0.2, ois_vals, width=0.4, color="#e74c3c", label="ordinary IS")
    axes[1].bar(x + 0.2, wis_vals, width=0.4, color="#2980b9", label="weighted IS")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("squared error contribution")
    axes[1].set_title(f"MSE decomposition (RMSE: OIS={summary['ois']['rmse']:.4f}, "
                      f"WIS={summary['wis']['rmse']:.4f})")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")
    finalize_plot(fig, os.path.join(outdir, "ope_is.png"))

    print("=" * 72)
    print(f"  OPE of the optimal greedy policy from eps={args.eps_behavior} "
          f"behaviour data ({args.n_trials} trials x {args.n_episodes} episodes)")
    print(f"  exact V^pi(s0) = {exact_start:.6f}")
    print(f"  {'estimator':<14s} {'mean':>9s} {'std':>9s} "
          f"{'bias':>9s} {'rmse':>9s}")
    for name in ("ois", "wis"):
        s = summary[name]
        print(f"  {name:<14s} {s['mean']:>9.4f} {s['std']:>9.4f} "
              f"{s['bias']:>+9.4f} {s['rmse']:>9.4f}")
    print(f"  mean ESS fraction: {summary['mean_ess_fraction']:.4f}")
    print("=" * 72)
    save_json(summary, os.path.join(outdir, "ope_summary.json"))
    print(f"  metrics saved to {outdir}/ope_summary.json")


if __name__ == "__main__":
    main()
