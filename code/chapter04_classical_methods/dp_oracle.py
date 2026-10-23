"""Experiment 1: dynamic programming against the exact model (the lab oracle).

Reads the transition model ``P`` from ``env.unwrapped.P`` and runs three
classical DP algorithms on it:

1. iterative policy evaluation (fixed uniform-random policy),
2. policy iteration (evaluation + greedy improvement to optimality),
3. value iteration (Bellman optimality backup to optimality).

All three share one residual stopping rule: stop sweeping when the largest
single-state change (max norm) drops below ``--theta``. The optimal value
function ``V*`` / ``Q*`` produced here is the exact oracle that the
sampling-based experiments (MC, SARSA, Expected SARSA, Q-learning, IS)
are measured against.

How to run (from the repository root)::

    .venv/bin/python code/chapter04_classical_methods/dp_oracle.py
    .venv/bin/python code/chapter04_classical_methods/dp_oracle.py --smoke
"""

import argparse
import os
import time

import numpy as np

from common import (
    FROZENLAKE_ID,
    add_common_args,
    ensure_outdir,
    load_model,
    resolve_device,
    save_json,
)


def q_from_v(P, V: np.ndarray, gamma: float) -> np.ndarray:
    """One-step lookahead: ``Q(s,a) = sum_{s'} P(s'|s,a) [r + gamma V(s')]``.

    Transitions flagged ``done`` lead out of the episode, so they
    contribute no bootstrapped value (``V`` of a terminal state is 0).
    """
    n_states = len(P)
    n_actions = len(P[0])
    Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            total = 0.0
            for prob, ns, reward, done in P[s][a]:
                total += prob * (reward + (0.0 if done else gamma * V[ns]))
            Q[s, a] = total
    return Q


def bellman_optimality_residual(P, V: np.ndarray, gamma: float) -> float:
    """Max-norm Bellman optimality residual ``||T*V - V||_inf``.

    This is the quantity the unit test checks: a true ``V*`` makes it ~0.
    """
    Q = q_from_v(P, V, gamma)
    return float(np.abs(Q.max(axis=1) - V).max())


def iterative_policy_evaluation(P, policy: np.ndarray, gamma: float, theta: float):
    """Evaluate a fixed stochastic policy by sweeping the Bellman expectation backup.

    ``V_{k+1}(s) = sum_a pi(a|s) [R(s,a) + gamma sum_{s'} P(s'|s,a) V_k(s')]``
    until the max-norm change (residual) falls below ``theta``.
    Returns ``(V, n_sweeps)``.
    """
    n_states = len(P)
    V = np.zeros(n_states)
    n_sweeps = 0
    while True:
        n_sweeps += 1
        delta = 0.0
        for s in range(n_states):
            old = V[s]
            new = 0.0
            for a in range(len(P[s])):
                action_value = 0.0
                for prob, ns, reward, done in P[s][a]:
                    action_value += prob * (reward + (0.0 if done else gamma * V[ns]))
                new += policy[s, a] * action_value
            V[s] = new
            delta = max(delta, abs(old - new))
        if delta < theta:
            break
    return V, n_sweeps


def policy_iteration(P, gamma: float, theta: float):
    """Alternate full policy evaluation with greedy policy improvement.

    Starts from the uniform-random policy. Returns
    ``(policy, V, Q, n_improvements, total_eval_sweeps)``.
    """
    n_states = len(P)
    n_actions = len(P[0])
    policy = np.ones((n_states, n_actions)) / n_actions
    total_eval_sweeps = 0
    n_improvements = 0
    while True:
        V, n_sweeps = iterative_policy_evaluation(P, policy, gamma, theta)
        total_eval_sweeps += n_sweeps
        Q = q_from_v(P, V, gamma)
        stable = True
        for s in range(n_states):
            old_action = int(np.argmax(policy[s]))
            best_action = int(np.argmax(Q[s]))
            if old_action != best_action:
                stable = False
            row = np.zeros(n_actions)
            row[best_action] = 1.0
            policy[s] = row
        if stable:
            break
        n_improvements += 1
    Q = q_from_v(P, V, gamma)
    return policy, V, Q, n_improvements, total_eval_sweeps


def value_iteration(P, gamma: float, theta: float):
    """Sweep the Bellman optimality backup ``V <- max_a Q`` until convergence.

    Returns ``(V, Q, n_sweeps)``.
    """
    n_states = len(P)
    V = np.zeros(n_states)
    n_sweeps = 0
    while True:
        n_sweeps += 1
        Q = q_from_v(P, V, gamma)
        V_new = Q.max(axis=1)
        delta = float(np.abs(V_new - V).max())
        V = V_new
        if delta < theta:
            break
    return V, Q, n_sweeps


def run_dp(env_id: str, gamma: float, theta: float) -> dict:
    """Run all three DP algorithms; check that they agree."""
    P, n_states, n_actions, terminals, resolved_id = load_model(env_id)

    uniform = np.ones((n_states, n_actions)) / n_actions
    V_rand, eval_sweeps = iterative_policy_evaluation(P, uniform, gamma, theta)

    t0 = time.time()
    pi_pi, V_pi, Q_pi, n_improvements, pi_eval_sweeps = policy_iteration(
        P, gamma, theta
    )
    t_pi = time.time() - t0

    t0 = time.time()
    V_vi, Q_vi, vi_sweeps = value_iteration(P, gamma, theta)
    t_vi = time.time() - t0

    agreement = float(np.abs(V_pi - V_vi).max())
    residual_pi = bellman_optimality_residual(P, V_pi, gamma)
    residual_vi = bellman_optimality_residual(P, V_vi, gamma)

    return {
        "env_id": resolved_id,
        "gamma": gamma,
        "theta": theta,
        "n_states": n_states,
        "n_actions": n_actions,
        "terminals": sorted(terminals),
        "policy_evaluation_uniform": {
            "sweeps": eval_sweeps,
            "V": V_rand.tolist(),
        },
        "policy_iteration": {
            "improvements": n_improvements,
            "total_eval_sweeps": pi_eval_sweeps,
            "seconds": t_pi,
            "V_star": V_pi.tolist(),
            "greedy_policy": [int(np.argmax(row)) for row in pi_pi],
            "residual": residual_pi,
        },
        "value_iteration": {
            "sweeps": vi_sweeps,
            "seconds": t_vi,
            "V_star": V_vi.tolist(),
            "greedy_policy": [int(np.argmax(row)) for row in Q_vi],
            "residual": residual_vi,
        },
        "Q_star": Q_vi.tolist(),
        "V_star_agreement_max_abs": agreement,
    }


def print_summary(results: dict) -> None:
    pi = results["policy_iteration"]
    vi = results["value_iteration"]
    print("=" * 64)
    print(f"  DP oracle on {results['env_id']} "
          f"(gamma={results['gamma']}, theta={results['theta']})")
    print("=" * 64)
    print(f"  {'method':<28s} {'sweeps':>8s} {'residual':>12s}")
    print(f"  {'policy eval (uniform pi)':<28s} "
          f"{results['policy_evaluation_uniform']['sweeps']:>8d} {'--':>12s}")
    print(f"  {'policy iteration':<28s} "
          f"{pi['total_eval_sweeps']:>8d} {pi['residual']:>12.2e}")
    print(f"    (policy improvements: {pi['improvements']})")
    print(f"  {'value iteration':<28s} "
          f"{vi['sweeps']:>8d} {vi['residual']:>12.2e}")
    print(f"  PI vs VI max |V diff|: {results['V_star_agreement_max_abs']:.2e}")
    print(f"  V*(start state 0): {vi['V_star'][0]:.6f}")
    print(f"  greedy policy: {vi['greedy_policy']}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="DP oracle for the Chapter 4 lab")
    parser.add_argument("--env-id", type=str, default=FROZENLAKE_ID)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=1e-6)
    add_common_args(parser)
    args = parser.parse_args()

    if args.smoke:
        args.theta = max(args.theta, 1e-4)

    device = resolve_device(args.device)
    seed = args.seed  # DP is deterministic; recorded for provenance only.
    outdir = ensure_outdir(args.outdir)
    print(f"device={device} seed={seed} smoke={args.smoke}")

    results = run_dp(args.env_id, args.gamma, args.theta)
    results["device"] = device
    results["seed"] = seed
    save_json(results, os.path.join(outdir, "dp_results.json"))
    print_summary(results)
    print(f"  metrics saved to {outdir}/dp_results.json")


if __name__ == "__main__":
    main()
