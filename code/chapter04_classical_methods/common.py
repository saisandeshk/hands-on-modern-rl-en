"""Shared helpers for the Chapter 4 classical-methods lab.

All experiments in this lab are tabular: the state and action spaces are
small and discrete, so policies and value functions are plain numpy arrays.
The helpers here load the exact transition model ``P`` from a gymnasium
toy-text environment (``env.unwrapped.P``), which is what makes dynamic
programming an exact oracle for everything else in the lab.
"""

import argparse
import json
import os
import random
from typing import Callable

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# CliffWalking was registered as v0 in older gymnasium releases and as v1 in
# newer ones. Try v1 first, fall back to v0 so the lab works with either.
CLIFFWALKING_IDS = ("CliffWalking-v1", "CliffWalking-v0")
FROZENLAKE_ID = "FrozenLake-v1"

OUTPUT_SUBDIR = os.path.join("runs", "chapter04_classical_methods")


def resolve_device(name: str) -> str:
    """Resolve ``--device {auto,cpu,cuda}`` to a concrete device string.

    This lab is pure numpy, so the device only affects what we report; we
    still accept the flag so every lab script shares the same CLI surface.
    """
    if name == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return name


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the CLI flags every lab script must accept."""
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="compute device (lab is numpy-only; recorded for provenance)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny configuration that finishes in under ~60 s on CPU",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=OUTPUT_SUBDIR,
        help="directory for metrics (JSON/CSV) and plots (PNG)",
    )
    return parser


def seed_all(seed: int) -> np.random.Generator:
    """Seed ``random`` and ``numpy``; return a dedicated Generator."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return np.random.default_rng(seed)


def make_env(env_id: str):
    """Create a gymnasium environment, tolerating the CliffWalking rename."""
    if env_id in ("CliffWalking-v1", "CliffWalking-v0", "CliffWalking"):
        last_error = None
        for candidate in CLIFFWALKING_IDS:
            try:
                return gym.make(candidate), candidate
            except gym.error.NameNotFound as exc:  # pragma: no cover
                last_error = exc
        raise last_error  # pragma: no cover
    return gym.make(env_id), env_id


def load_model(env_id: str):
    """Read the exact transition model ``P`` from a toy-text environment.

    Returns ``(P, n_states, n_actions, terminals)`` where
    ``P[s][a]`` is a list of ``(prob, next_state, reward, done)`` tuples,
    exactly as stored in ``env.unwrapped.P``. Terminal states are those
    from which every action ends the episode (holes and goals).
    """
    env, resolved_id = make_env(env_id)
    raw_P = env.unwrapped.P
    n_states = len(raw_P)
    n_actions = len(raw_P[0])
    P = [[list(raw_P[s][a]) for a in range(n_actions)] for s in range(n_states)]
    terminals = set()
    for s in range(n_states):
        if all(done for a in range(n_actions) for (_, _, _, done) in P[s][a]):
            terminals.add(s)
    env.close()
    return P, n_states, n_actions, terminals, resolved_id


def epsilon_greedy_action(
    Q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator
) -> int:
    """Sample an action epsilon-greedily from Q (ties -> lowest index)."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[state]))


def greedy_action(Q: np.ndarray, state: int) -> int:
    """Greedy action with deterministic tie-breaking (lowest index)."""
    return int(np.argmax(Q[state]))


def evaluate_greedy(
    env_id: str, Q: np.ndarray, n_episodes: int, base_seed: int, gamma: float = 1.0,
    max_steps: int | None = None,
) -> dict:
    """Roll out the greedy policy; report mean undiscounted return and success.

    "Success" is environment specific: reaching the goal in FrozenLake
    (return > 0) and we also report the plain mean return, which is the
    right metric for CliffWalking (reward -1 per step, -100 for the cliff).

    ``max_steps`` caps each rollout (needed for environments without a
    time limit, where a degenerate greedy policy could otherwise loop
    forever); hitting the cap just stops the rollout.
    """
    env, _ = make_env(env_id)
    returns = np.zeros(n_episodes)
    for k in range(n_episodes):
        state, _ = env.reset(seed=base_seed + k)
        done = False
        total = 0.0
        steps = 0
        while not done:
            action = greedy_action(Q, int(state))
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated or (max_steps is not None and steps >= max_steps)
        returns[k] = total
    env.close()
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "success_rate": float((returns > 0).mean()),
    }


def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save_json(payload: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_mean_band(
    ax,
    xs: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    label: str,
    color: str,
) -> None:
    """Plot a mean curve with a shaded +/- one-standard-deviation band."""
    ax.plot(xs, means, label=label, color=color, linewidth=2)
    ax.fill_between(
        xs, means - stds, means + stds, color=color, alpha=0.2, linewidth=0
    )


def finalize_plot(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  plot saved to {path}")


PolicyFn = Callable[[int], int]
