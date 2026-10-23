"""Softmax policy and the REINFORCE update for the toy code agent.

Setup: a contextual bandit with one context per task. For task ``t`` the
agent holds logits ``theta[t, a]`` over that task's candidate implementations
and samples

    pi(a | t) = softmax(theta[t, :])_a.

This is the tabular analogue of the LLM policy ``pi_theta`` in Chapter 20:
sampling a candidate patch, then observing an executable test reward. The
update is REINFORCE with a per-task running baseline ``b[t]``:

    advantage = R - b[t],   loss = -mean(advantage * log pi(a | t)),

which subtracts the task's average reward so that only better-than-average
candidates are reinforced. The baseline depends only on the task (the
context), never on the sampled action, so the gradient stays unbiased while
its variance drops -- the same variance-reduction argument as the value
baseline in policy-gradient methods.
"""

from __future__ import annotations

import torch
from torch import nn


class SoftmaxPolicy(nn.Module):
    """Table of logits, one row per task, masked softmax over candidates."""

    def __init__(self, counts: list[int], device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.counts = list(counts)
        self.n_tasks = len(counts)
        self.max_arms = max(counts)
        self.logits = nn.Parameter(torch.zeros(self.n_tasks, self.max_arms))
        mask = torch.zeros(self.n_tasks, self.max_arms, dtype=torch.bool)
        for t, c in enumerate(counts):
            mask[t, c:] = True
        self.register_buffer("pad_mask", mask)
        self.to(device)

    def probs(self) -> torch.Tensor:
        """Action probabilities per task,shape [T, K]; padded arms get 0."""
        masked = self.logits.masked_fill(self.pad_mask, float("-inf"))
        return torch.softmax(masked, dim=-1)

    def sample(self, task_idx: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Sample one candidate for a task; returns (action, log_prob, entropy)."""
        p = self.probs()[task_idx]
        dist = torch.distributions.Categorical(probs=p)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action), dist.entropy()


def reinforce_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    baselines: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """REINFORCE loss on a fixed batch with given baselines.

    Args:
        log_probs: ``log pi(a_t | t)`` for each sampled task, shape [B].
        rewards: sandbox test-pass rewards, shape [B].
        baselines: task-dependent (action-independent) baselines, shape [B].

    Returns:
        ``(loss, advantages)`` with ``loss = -mean((R - b) * log pi)``.
    """
    advantages = rewards.to(log_probs.device) - baselines.to(log_probs.device)
    loss = -(advantages.detach() * log_probs).mean()
    return loss, advantages


def reinforce_step(
    policy: SoftmaxPolicy,
    optimizer: torch.optim.Optimizer,
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    baselines: torch.Tensor,
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
) -> dict[str, float]:
    """One gradient step; returns scalars for logging."""
    loss, advantages = reinforce_loss(log_probs, rewards, baselines)
    if entropy is not None and entropy_coef > 0.0:
        loss = loss - entropy_coef * entropy.to(loss.device).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item()), "adv_mean": float(advantages.mean().item())}
