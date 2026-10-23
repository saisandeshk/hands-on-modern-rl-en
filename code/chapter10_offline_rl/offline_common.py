"""Shared helpers for the Chapter 10 offline-RL lab (CartPole-v1).

This module contains only plumbing: seeding, device selection, small MLP
networks, fixed-dataset save/load, and online evaluation. The RL ideas
(BC, naive offline DQN, discrete CQL, IS/WIS/FQE) live in the scripts that
import this module.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

VALID_DEVICES = ("auto", "cpu", "cuda")


def resolve_device(choice: str = "auto") -> torch.device:
    """Pick a torch device. ``auto`` prefers CUDA, then falls back to CPU."""
    choice = choice.lower()
    if choice not in VALID_DEVICES:
        raise ValueError(f"--device must be one of {VALID_DEVICES}, got {choice!r}")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`--device cuda` requested but CUDA is not available")
    return torch.device(choice)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) for reproducibility.

    Also forces single-threaded CPU execution: multithreaded reductions in
    Linear/Adam backprop can otherwise reorder floating-point sums run to
    run, which visibly changes DQN training curves on CartPole.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):  # very old torch: keyword missing
        torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class PolicyMLP(nn.Module):
    """State -> action logits (used by behaviour cloning and the BC policy)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QMLP(nn.Module):
    """State -> Q(s, a) for every discrete action (used by DQN / CQL / FQE)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Fixed-dataset IO
# ---------------------------------------------------------------------------

DATASET_FIELDS = ("states", "actions", "rewards", "next_states", "dones", "behaviour_probs")


def save_dataset(path: str, arrays: dict, meta: dict) -> None:
    """Save transitions as ``.npz`` plus a sidecar ``_meta.json`` file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    with open(os.path.splitext(path)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_dataset(path: str) -> tuple[dict, dict]:
    """Load a dataset saved by :func:`save_dataset`.

    Returns ``(arrays, meta)`` where ``arrays`` maps each field name to a
    NumPy array and ``meta`` is the JSON sidecar (possibly empty).
    """
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    meta: dict = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    for field in DATASET_FIELDS:
        if field not in arrays:
            raise ValueError(f"dataset {path} is missing required field {field!r}")
    if not (len(arrays["states"]) == len(arrays["actions"]) == len(arrays["rewards"])):
        raise ValueError(f"dataset {path} has inconsistent array lengths")
    return arrays, meta


# ---------------------------------------------------------------------------
# Online evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    mean: float
    std: float
    returns: list

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_greedy(
    q_net: QMLP,
    env_name: str = "CartPole-v1",
    n_episodes: int = 20,
    base_seed: int = 0,
    device: str = "cpu",
) -> EvalResult:
    """Roll out the greedy policy of ``q_net`` and report episode returns."""
    import gymnasium as gym

    dev = torch.device(device)
    q_net = q_net.to(dev).eval()
    returns: list[float] = []
    with torch.no_grad():
        for ep in range(n_episodes):
            env = gym.make(env_name)
            obs, _ = env.reset(seed=base_seed + ep)
            total = 0.0
            done = False
            while not done:
                s = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
                action = int(q_net(s).argmax(dim=1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = bool(terminated or truncated)
            returns.append(total)
            env.close()
    return EvalResult(mean=float(np.mean(returns)), std=float(np.std(returns)), returns=returns)


def evaluate_stochastic(
    probs_fn: Callable[[np.ndarray], np.ndarray],
    env_name: str = "CartPole-v1",
    n_episodes: int = 20,
    base_seed: int = 0,
) -> EvalResult:
    """Roll out a stochastic policy given as ``obs -> action-prob vector``."""
    import gymnasium as gym

    returns: list[float] = []
    for ep in range(n_episodes):
        env = gym.make(env_name)
        obs, _ = env.reset(seed=base_seed + ep)
        total = 0.0
        done = False
        while not done:
            probs = np.asarray(probs_fn(obs), dtype=np.float64)
            action = int(np.random.choice(len(probs), p=probs / probs.sum()))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        returns.append(total)
        env.close()
    return EvalResult(mean=float(np.mean(returns)), std=float(np.std(returns)), returns=returns)


def default_outdir() -> str:
    """Repository ``runs/chapter10_offline_rl`` directory (gitignored)."""
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(repo_root, "runs", "chapter10_offline_rl")
