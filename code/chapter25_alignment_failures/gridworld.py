"""Loophole gridworld for Chapter 25: proxy reward vs. true task utility.

The environment is a small deterministic grid inspired by the CoastRunners
boat-racing example from Section 25.1: a bonus tile sits right next to the
start state and pays a reward every time the agent steps on it, while the
real task is to reach a distant goal. A proxy reward that counts bonus
visits therefore has a loophole: endlessly circling the bonus earns more
proxy reward than finishing the task.

Layout (W=7, H=5), ``S`` = start, ``+`` = respawning bonus, ``G`` = goal::

    . . . . . . .
    . . . . . . .
    S + . . . . G
    . . . . . . .
    . . . . . . .

Two reward modes are provided:

- ``proxy``: every visit to the bonus tile pays ``bonus_reward`` (the
  loophole), reaching the goal pays ``goal_reward`` and ends the episode.
- ``fixed``: the bonus pays only on its first visit per episode (the
  loophole is removed) and a potential-based shaping term
  ``F(s, s') = gamma * Phi(s') - Phi(s)`` with
  ``Phi(s) = -dist_scale * manhattan(s, goal)`` is added. By Ng et al.
  (1999), potential-based shaping does not change the optimal policy; it
  only densifies the learning signal toward the goal.

In both modes the *true task utility* is tracked separately: 1.0 when the
agent reaches the goal within the step limit, 0.0 otherwise. Comparing the
proxy return against this independent success metric is the honest-evaluation
diagnostic of Sections 25.1 and 25.4: reward hacking is exactly the regime
where the proxy curve rises while the independent metric stalls or drops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Grid layout constants
# ---------------------------------------------------------------------------
WIDTH = 7
HEIGHT = 5
START = (0, 2)
BONUS = (1, 2)
GOAL = (6, 2)

# Action set: up, down, left, right as (dx, dy) steps.
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
N_ACTIONS = len(ACTIONS)


@dataclass
class LoopholeGridworld:
    """Deterministic gridworld with a respawning bonus tile.

    Args:
        reward_mode: ``"proxy"`` (bonus pays on every visit) or ``"fixed"``
            (bonus pays once per episode plus potential-based shaping).
        max_steps: episode truncation horizon.
        step_cost: small per-step penalty so shorter paths are preferred.
        bonus_reward: proxy payout for stepping on the bonus tile.
        goal_reward: payout for reaching the goal (ends the episode).
        dist_scale: scale of the shaping potential ``Phi(s)`` in fixed mode.
        gamma: discount factor, used only for the shaping term
            ``F(s, s') = gamma * Phi(s') - Phi(s)``.
    """

    reward_mode: str = "proxy"
    max_steps: int = 60
    step_cost: float = 0.05
    bonus_reward: float = 2.0
    goal_reward: float = 10.0
    dist_scale: float = 0.5
    gamma: float = 0.99
    slip: float = 0.0
    slip_seed: int = 0

    _pos: tuple[int, int] = field(default=START, init=False, repr=False)
    _steps: int = field(default=0, init=False, repr=False)
    _bonus_used: bool = field(default=False, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.reward_mode not in ("proxy", "fixed"):
            raise ValueError(f"reward_mode must be 'proxy' or 'fixed', got {self.reward_mode!r}")
        if not 0.0 <= self.slip < 1.0:
            raise ValueError(f"slip must be in [0, 1), got {self.slip!r}")
        self._rng = np.random.default_rng(self.slip_seed)
        self.reset()

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        """Manhattan distance between two grid cells."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def potential(self, pos: tuple[int, int]) -> float:
        """Shaping potential: higher (less negative) near the goal."""
        return -self.dist_scale * self.manhattan(pos, GOAL)

    # -- environment API -------------------------------------------------
    def reset(self) -> tuple[int, int]:
        """Start a new episode; return the start state."""
        self._pos = START
        self._steps = 0
        self._bonus_used = False
        return self._pos

    @property
    def pos(self) -> tuple[int, int]:
        """Current agent position."""
        return self._pos

    def step(self, action: int) -> tuple[tuple[int, int], float, float, bool]:
        """Apply one action.

        Returns:
            A ``(next_state, proxy_reward, true_utility, done)`` tuple.
            ``true_utility`` is 1.0 exactly when the goal is reached and 0.0
            otherwise, so it never counts bonus farming as progress.

        With ``slip > 0`` the requested action is replaced by a uniformly
        random one with probability ``slip`` (a slippery warehouse floor).
        Slip noise makes evaluation success rates continuous rather than
        all-or-nothing, which is what Part 2 needs to demonstrate robust
        statistics: with a deterministic environment every seed would score
        exactly 0 or 1 and there would be nothing to aggregate.
        """
        if self.slip > 0.0 and self._rng.random() < self.slip:
            action = int(self._rng.integers(N_ACTIONS))
        dx, dy = ACTIONS[action]
        x = min(max(self._pos[0] + dx, 0), WIDTH - 1)
        y = min(max(self._pos[1] + dy, 0), HEIGHT - 1)
        nxt = (x, y)
        self._steps += 1

        reward = -self.step_cost
        true_utility = 0.0
        done = False

        if nxt == BONUS:
            if self.reward_mode == "proxy":
                reward += self.bonus_reward
            elif not self._bonus_used:
                # Fixed mode: the bonus is a one-time novelty payment.
                reward += self.bonus_reward
                self._bonus_used = True
        if nxt == GOAL:
            reward += self.goal_reward
            true_utility = 1.0
            done = True

        if self.reward_mode == "fixed" and not done:
            # Potential-based shaping: provably optimal-policy-preserving.
            reward += self.gamma * self.potential(nxt) - self.potential(self._pos)

        if self._steps >= self.max_steps:
            done = True

        self._pos = nxt
        return nxt, reward, true_utility, done


# ---------------------------------------------------------------------------
# Tabular Q-learning
# ---------------------------------------------------------------------------
def train_q_learning(
    env: LoopholeGridworld,
    episodes: int,
    alpha: float = 0.25,
    epsilon: float = 0.15,
    gamma: float = 0.99,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Run tabular Q-learning with an epsilon-greedy behaviour policy.

    Args:
        env: the gridworld to train in (its reward mode is used as-is).
        episodes: number of training episodes.
        alpha: Q-learning step size.
        epsilon: fixed exploration rate of the behaviour policy.
        gamma: discount factor of the Q-learning target.
        seed: random seed for reproducible exploration.

    Returns:
        ``(Q, history)`` where ``Q`` has shape ``(HEIGHT, WIDTH, N_ACTIONS)``
        and ``history`` records per-episode proxy return, success flag and
        episode length for later plotting.
    """
    rng = np.random.default_rng(seed)
    q = np.zeros((HEIGHT, WIDTH, N_ACTIONS))
    history: dict[str, list[float]] = {"proxy_return": [], "success": [], "length": []}

    for _ in range(episodes):
        state = env.reset()
        ep_return = 0.0
        ep_success = 0.0
        length = 0
        done = False
        while not done:
            x, y = state
            if rng.random() < epsilon:
                action = int(rng.integers(N_ACTIONS))
            else:
                action = int(np.argmax(q[y, x]))
            nxt, reward, utility, done = env.step(action)
            nx, ny = nxt
            # Q-learning target with no bootstrapping past termination.
            target = reward if done else reward + gamma * float(np.max(q[ny, nx]))
            q[y, x, action] += alpha * (target - q[y, x, action])
            ep_return += reward
            ep_success = max(ep_success, utility)
            length += 1
            state = nxt
        history["proxy_return"].append(ep_return)
        history["success"].append(ep_success)
        history["length"].append(float(length))
    return q, history


def evaluate_greedy(
    q: np.ndarray,
    reward_mode: str,
    eval_episodes: int = 200,
    max_steps: int = 60,
    **env_kwargs,
) -> dict[str, float]:
    """Evaluate the greedy policy: proxy return vs. true task success.

    The greedy policy derived from ``q`` is rolled out without exploration.
    Both the proxy return (what training optimised) and the goal-reaching
    success rate (the independent task metric) are reported, mirroring the
    proxy-vs-utility diagnostic of Sections 25.1 and 25.4.
    """
    env = LoopholeGridworld(reward_mode=reward_mode, max_steps=max_steps, **env_kwargs)
    returns: list[float] = []
    successes: list[float] = []
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        ep_success = 0.0
        while not done:
            x, y = state
            action = int(np.argmax(q[y, x]))
            state, reward, utility, done = env.step(action)
            ep_return += reward
            ep_success = max(ep_success, utility)
        returns.append(ep_return)
        successes.append(ep_success)
    return {
        "mean_proxy_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
    }


def count_bonus_visits(
    q: np.ndarray, reward_mode: str = "proxy", max_steps: int = 60
) -> float:
    """Mean number of bonus-tile visits per greedy rollout (farming meter)."""
    env = LoopholeGridworld(reward_mode=reward_mode, max_steps=max_steps)
    counts: list[float] = []
    for _ in range(50):
        state = env.reset()
        done = False
        visits = 0
        while not done:
            x, y = state
            action = int(np.argmax(q[y, x]))
            state, _, _, done = env.step(action)
            if state == BONUS:
                visits += 1
        counts.append(float(visits))
    return float(np.mean(counts))
