"""Smoke + unit tests for the Chapter 4 classical-methods lab.

Run from the repository root::

    .venv/bin/pytest code/chapter04_classical_methods -q

The file has two parts:

1. unit tests of the core pieces against hand-computed values or the exact
   DP oracle (fast, no subprocess);
2. smoke tests that run every lab script in ``--smoke`` mode in a
   subprocess with a timeout and check the expected output files appear.
"""

import os
import subprocess
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUTDIR = os.path.join(REPO_ROOT, "runs", "chapter04_classical_methods")
PYTHON = os.path.join(REPO_ROOT, ".venv", "bin", "python")

sys.path.insert(0, HERE)

from tabular_common import load_model  # noqa: E402
from dp_oracle import (  # noqa: E402
    bellman_optimality_residual,
    policy_iteration,
    q_from_v,
    value_iteration,
)
from mc_td_control import (  # noqa: E402
    expected_sarsa_update,
    qlearning_update,
    sarsa_update,
)
from ope_is import effective_sample_size, ordinary_is, weighted_is  # noqa: E402


# --------------------------------------------------------------------------
# Part 1: unit tests.
# --------------------------------------------------------------------------
class BellmanResidualTest(unittest.TestCase):
    """The DP solution must (nearly) satisfy the Bellman optimality equation."""

    def test_value_iteration_residual_near_zero(self):
        P, _, _, _, _ = load_model("FrozenLake-v1")
        V, _, _ = value_iteration(P, gamma=1.0, theta=1e-8)
        self.assertLess(bellman_optimality_residual(P, V, 1.0), 1e-6)

    def test_policy_iteration_agrees_with_value_iteration(self):
        P, _, _, _, _ = load_model("FrozenLake-v1")
        _, V_pi, _, _, _ = policy_iteration(P, gamma=1.0, theta=1e-6)
        V_vi, _, _ = value_iteration(P, gamma=1.0, theta=1e-6)
        self.assertLess(float(np.abs(V_pi - V_vi).max()), 1e-4)

    def test_optimal_start_value_is_known_answer(self):
        # Slippery 4x4 FrozenLake: optimal success probability from the
        # start state is 14/17 (standard textbook answer).
        P, _, _, _, _ = load_model("FrozenLake-v1")
        V, _, _ = value_iteration(P, gamma=1.0, theta=1e-10)
        self.assertAlmostEqual(V[0], 14 / 17, places=5)


class SingleStepUpdateTest(unittest.TestCase):
    """One SARSA / Q-learning update on a fixed transition, by hand."""

    def setUp(self):
        self.Q = np.zeros((3, 2))
        self.Q[0, 0] = 2.0  # Q(s, a) before the update
        self.Q[1, 0] = 1.0
        self.Q[1, 1] = 4.0  # greedy next value is 4
        self.alpha = 0.5
        self.gamma = 0.9

    def test_sarsa_update(self):
        # Next behaviour action a' = 0 with Q(s',a') = 1:
        # target = 1 + 0.9 * 1 = 1.9; new Q = 2 + 0.5 * (1.9 - 2) = 1.95.
        sarsa_update(self.Q, 0, 0, 1.0, 1, 0, False, self.alpha, self.gamma)
        self.assertAlmostEqual(self.Q[0, 0], 1.95, places=10)

    def test_qlearning_update(self):
        # Greedy next value max Q(s') = 4:
        # target = 1 + 0.9 * 4 = 4.6; new Q = 2 + 0.5 * (4.6 - 2) = 3.3.
        qlearning_update(self.Q, 0, 0, 1.0, 1, False, self.alpha, self.gamma)
        self.assertAlmostEqual(self.Q[0, 0], 3.3, places=10)

    def test_expected_sarsa_update(self):
        # eps = 0.5: expectation = 0.5 * 4 + 0.5 * mean(1, 4) = 3.25;
        # target = 1 + 0.9 * 3.25 = 3.925;
        # new Q = 2 + 0.5 * (3.925 - 2) = 2.9625.
        expected_sarsa_update(self.Q, 0, 0, 1.0, 1, False,
                              self.alpha, self.gamma, epsilon=0.5)
        self.assertAlmostEqual(self.Q[0, 0], 2.9625, places=10)

    def test_terminal_transition_has_no_bootstrap(self):
        # done=True: target = r regardless of Q(s'); 2 + 0.5*(1-2) = 1.5.
        sarsa_update(self.Q, 0, 0, 1.0, 1, 1, True, self.alpha, self.gamma)
        self.assertAlmostEqual(self.Q[0, 0], 1.5, places=10)
        self.Q[0, 0] = 2.0
        qlearning_update(self.Q, 0, 0, 1.0, 1, True, self.alpha, self.gamma)
        self.assertAlmostEqual(self.Q[0, 0], 1.5, places=10)


class ImportanceSamplingTest(unittest.TestCase):
    """OIS/WIS on a tiny two-episode dataset with a hand-computed answer."""

    def test_estimators_on_known_answer(self):
        # returns = [2, 0], rhos = [1, 3]:
        # OIS = (1*2 + 3*0) / 2 = 1.0; WIS = (1*2 + 3*0) / (1+3) = 0.5.
        returns = np.array([2.0, 0.0])
        rhos = np.array([1.0, 3.0])
        self.assertAlmostEqual(ordinary_is(returns, rhos), 1.0, places=12)
        self.assertAlmostEqual(weighted_is(returns, rhos), 0.5, places=12)

    def test_ois_recovers_target_value_for_matching_weights(self):
        # Two-state toy: target always takes a0 (return 1), behaviour picks
        # uniformly, so rho = 2 on a0 episodes and 0 on a1 episodes.
        # OIS = (2*1 + 0*0 + 2*1 + 0*0) / 4 = 1.0 = V^pi, exactly.
        returns = np.array([1.0, 0.0, 1.0, 0.0])
        rhos = np.array([2.0, 0.0, 2.0, 0.0])
        self.assertAlmostEqual(ordinary_is(returns, rhos), 1.0, places=12)
        self.assertAlmostEqual(weighted_is(returns, rhos), 1.0, places=12)

    def test_effective_sample_size(self):
        # ESS = (1+3)^2 / (1+9) = 1.6; degenerate all-zero weights give 0.
        self.assertAlmostEqual(
            effective_sample_size(np.array([1.0, 3.0])), 1.6, places=12
        )
        self.assertEqual(effective_sample_size(np.array([0.0, 0.0])), 0.0)
        self.assertEqual(weighted_is(np.array([1.0]), np.array([0.0])), 0.0)


class QFromVTest(unittest.TestCase):
    """One-step lookahead agrees with the Bellman optimality equation."""

    def test_q_from_v_matches_vi(self):
        P, _, _, _, _ = load_model("FrozenLake-v1")
        V, Q_vi, _ = value_iteration(P, gamma=1.0, theta=1e-8)
        Q = q_from_v(P, V, 1.0)
        self.assertLess(float(np.abs(Q - Q_vi).max()), 1e-8)


# --------------------------------------------------------------------------
# Part 2: smoke tests (each script in --smoke mode, with a timeout).
# --------------------------------------------------------------------------
SMOKE_CASES = [
    ("dp_oracle.py", ["--smoke"], ["dp_results.json"], 180),
    ("mc_td_control.py", ["--smoke"], ["control_curves.csv", "control_online.csv",
                                       "control_summary.json",
                                       "control_frozenlake.png"], 300),
    ("cliffwalking.py", ["--smoke"], ["cliff_curves.csv", "cliff_summary.json",
                                      "cliff_comparison.png"], 300),
    ("ope_is.py", ["--smoke"], ["ope_summary.json", "ope_is.png"], 300),
]


class SmokeTest(unittest.TestCase):
    def test_scripts_run_in_smoke_mode(self):
        for script, extra_args, expected_files, timeout in SMOKE_CASES:
            with self.subTest(script=script):
                cmd = [PYTHON, os.path.join(HERE, script), "--seed", "0",
                       "--device", "cpu"] + extra_args
                proc = subprocess.run(
                    cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                    timeout=timeout,
                )
                self.assertEqual(
                    proc.returncode, 0,
                    msg=f"{script} failed:\nSTDOUT:\n{proc.stdout[-3000:]}"
                        f"\nSTDERR:\n{proc.stderr[-3000:]}",
                )
                for name in expected_files:
                    path = os.path.join(OUTDIR, name)
                    self.assertTrue(os.path.isfile(path), msg=f"missing {path}")


if __name__ == "__main__":
    unittest.main()
