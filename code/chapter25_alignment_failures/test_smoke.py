"""Tests for the Chapter 25 lab: smoke runs plus exact unit checks.

Run with:
    .venv/bin/pytest code/chapter25_alignment_failures -q

Every script is executed in smoke mode in a subprocess (with a timeout), and
the core estimators are checked against hand-computed values or an exact
dynamic-programming oracle -- no learned behaviour is asserted, only
mathematics that must hold by construction.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

sys.path.insert(0, HERE)

from gridworld import BONUS, START, LoopholeGridworld  # noqa: E402
from stats import bootstrap_ci, iqm_score, performance_profile, stratified_bootstrap_ci  # noqa: E402


def run_script(name: str, extra: list[str] | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a lab script in smoke mode; fail loudly on nonzero exit."""
    cmd = [sys.executable, os.path.join(HERE, name), "--smoke"] + (extra or [])
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# Smoke tests: each script must finish quickly and write its artefacts
# ---------------------------------------------------------------------------
def test_specification_gaming_smoke(tmp_path):
    """Part 1 finishes in smoke mode and writes CSV + PNG (into a scratch dir)."""
    outdir = str(tmp_path / "p1")
    proc = run_script("run_specification_gaming.py", ["--outdir", outdir])
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert os.path.exists(os.path.join(outdir, "spec_gaming_metrics.csv"))
    assert os.path.exists(os.path.join(outdir, "spec_gaming_curves.png"))
    assert "success" in proc.stdout


def test_eval_methodology_smoke(tmp_path):
    """Part 2 finishes in smoke mode and writes CSV + JSON + PNG (scratch dir)."""
    outdir = str(tmp_path / "p2")
    proc = run_script("run_eval_methodology.py", ["--outdir", outdir])
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert os.path.exists(os.path.join(outdir, "eval_scores.csv"))
    assert os.path.exists(os.path.join(outdir, "eval_stats.json"))
    assert os.path.exists(os.path.join(outdir, "eval_profile.png"))
    assert "IQM" in proc.stdout or "iqm" in proc.stdout.lower()


# ---------------------------------------------------------------------------
# Unit tests: estimators against direct computation
# ---------------------------------------------------------------------------
def test_iqm_matches_direct_computation():
    """IQM on a fixed array equals trim-then-average done by hand."""
    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])
    # Sorted already; drop lowest 2 and highest 2, average the middle 4.
    expected = (0.2 + 0.3 + 0.4 + 0.5) / 4.0
    assert iqm_score(x) == pytest.approx(expected)
    # Median cross-check against numpy on the same fixed array.
    assert float(np.median(x)) == pytest.approx(0.35)


def test_bootstrap_ci_on_constant_array_is_degenerate():
    """Resampling a constant array must yield a point interval at that value."""
    x = np.full(10, 0.7)
    est, lo, hi = bootstrap_ci(x, n_resamples=200, seed=0)
    assert est == pytest.approx(0.7)
    assert lo == pytest.approx(0.7)
    assert hi == pytest.approx(0.7)


def test_stratified_bootstrap_covers_sample_mean():
    """On a fixed non-degenerate array the 95% CI should cover the mean."""
    x = np.array([0.1, 0.4, 0.5, 0.6, 0.9])
    from stats import mean_score

    est, lo, hi = bootstrap_ci(x, statistic=mean_score, n_resamples=2000, seed=1)
    assert est == pytest.approx(float(np.mean(x)))
    assert lo <= float(np.mean(x)) <= hi
    assert lo < hi
    # Stratified variant returns one interval per group with the same property.
    out = stratified_bootstrap_ci({"g": x}, statistic=mean_score, n_resamples=500, seed=1)
    _, glo, ghi = out["g"]
    assert glo <= float(np.mean(x)) <= ghi


def test_performance_profile_counts_thresholds():
    """Profile values are exact exceedance fractions on a fixed array."""
    groups = {"g": np.array([0.0, 0.5, 1.0])}
    taus = np.array([0.25, 0.75, 1.5])
    prof = performance_profile(groups, taus)
    assert prof["g"] == pytest.approx([2 / 3, 1 / 3, 0.0])


# ---------------------------------------------------------------------------
# Unit tests: environment semantics and shaping invariance
# ---------------------------------------------------------------------------
def test_bonus_respawns_under_proxy_but_not_fixed():
    """The loophole exists in proxy mode and is removed in fixed mode."""
    # Step right onto the bonus tile from the start, step back, repeat.
    # In fixed mode the first visit also carries the shaping term
    # F = gamma * Phi(bonus) - Phi(start); the *second* visit must not pay
    # the bonus again, which is the actual loophole removal.
    for mode in ("proxy", "fixed"):
        env = LoopholeGridworld(reward_mode=mode)
        env.reset()
        shaping = env.gamma * env.potential(BONUS) - env.potential(START)
        _, r1, _, _ = env.step(3)  # right -> BONUS
        if mode == "proxy":
            assert r1 == pytest.approx(env.bonus_reward - env.step_cost)
        else:
            assert r1 == pytest.approx(env.bonus_reward - env.step_cost + shaping)
        env.step(2)  # left -> START
        _, r2, _, _ = env.step(3)  # right -> BONUS again
        if mode == "proxy":
            assert r2 == pytest.approx(env.bonus_reward - env.step_cost)
        else:
            # No second bonus payment (only step cost, possibly shaping).
            assert r2 < env.bonus_reward - env.step_cost


def test_goal_reaches_utility_one_and_terminates():
    """Reaching the goal yields true utility 1 and ends the episode."""
    env = LoopholeGridworld(reward_mode="proxy")
    env.reset()
    done = False
    # Walk right along the middle row until termination.
    for _ in range(20):
        _, _, u, done = env.step(3)
        if done:
            break
    assert done
    assert u == 1.0
    assert env.pos == (6, 2)


def value_iteration(probs, rewards, gamma=0.9, theta=1e-10):
    """Exact solver for a tiny finite MDP: (S, A) tables -> (V, policy)."""
    n_states, n_actions = rewards.shape
    v = np.zeros(n_states)
    while True:
        delta = 0.0
        for s in range(n_states):
            q_s = rewards[s] + gamma * (probs[s] @ v)
            best = float(np.max(q_s))
            delta = max(delta, abs(best - v[s]))
            v[s] = best
        if delta < theta:
            break
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        policy[s] = int(np.argmax(rewards[s] + gamma * (probs[s] @ v)))
    return v, policy


def test_potential_shaping_preserves_optimal_policy():
    """PBRS shaping keeps the optimal policy on a tiny MDP solved exactly.

    Chain of 3 states, 2 actions (0 = step right, 1 = stay). State 2 is an
    absorbing goal: entering it pays +1. The shaped reward adds
    F(s, s') = gamma * Phi(s') - Phi(s) with Phi(s) = -manhattan-to-goal.
    Ng et al. (1999) prove the optimal policy is unchanged; the exact
    value-iteration oracle checks the argmax policy state by state.
    """
    n_states, n_actions = 3, 2
    probs = np.zeros((n_states, n_actions, n_states))
    # Action 0: move right (clamped); action 1: stay.
    for s in range(n_states):
        probs[s, 0, min(s + 1, 2)] = 1.0
        probs[s, 1, s] = 1.0
    # Absorbing goal: all transitions from state 2 stay in state 2, no reward.
    probs[2, :, :] = 0.0
    probs[2, :, 2] = 1.0
    rewards = np.zeros((n_states, n_actions))
    rewards[0, 0] = 1.0  # stepping 0 -> 1 ... (reward on entry structure)
    rewards[1, 0] = 1.0  # stepping 1 -> 2 (goal entry)
    # Make staying costly so "go right" is uniquely optimal from 0 and 1.
    rewards[0, 1] = -0.1
    rewards[1, 1] = -0.1

    gamma = 0.9
    _, policy_plain = value_iteration(probs, rewards, gamma=gamma)
    assert list(policy_plain[:2]) == [0, 0]

    phi = np.array([-2.0, -1.0, 0.0])  # -distance to goal
    shaped = rewards.copy()
    for s in range(n_states):
        for a in range(n_actions):
            for ns in range(n_states):
                shaped[s, a] += probs[s, a, ns] * (gamma * phi[ns] - phi[s])
    _, policy_shaped = value_iteration(probs, shaped, gamma=gamma)
    assert list(policy_shaped) == list(policy_plain)
