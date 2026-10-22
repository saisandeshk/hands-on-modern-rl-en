"""Smoke + unit tests for the Chapter 22 computer-use lab.

Run from the repository root::

    .venv/bin/pytest code/chapter22_computer_use -q

Covers: element hit-testing, state updates, the task success checker,
deterministic layouts from a seed, the REINFORCE math against hand-computed
values, and one subprocess run of the training script in ``--smoke`` mode.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import gui_env
from gui_env import (MAX_STEPS, N_ACTIONS, N_ELEMENTS, N_TOKENS, OBS_DIM,
                     MiniFormEnv, action_name, check_success, count_satisfied,
                     element_at, make_layout, make_task)
from train_gui_agent import compute_returns, reinforce_losses

CHAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = CHAPTER_DIR.parent.parent


def submit_id(env: MiniFormEnv) -> int:
    return next(e.eid for e in env.elements if e.kind == "submit")


def field_id_for(env: MiniFormEnv, label: str) -> int:
    return next(e.eid for e in env.elements if e.label == label)


class HitTestingTest(unittest.TestCase):
    def test_centre_hit(self):
        els = make_layout(0)
        for el in els:
            with self.subTest(eid=el.eid):
                self.assertIs(element_at(els, el.x, el.y), el)

    def test_gap_miss(self):
        els = make_layout(1)
        self.assertIsNone(element_at(els, -1.0, -1.0))
        self.assertIsNone(element_at(els, 100.0, 100.0))

    def test_half_open_edges(self):
        els = make_layout(2)
        el = els[0]
        # Left/bottom edges belong to the rect, right/top edges do not, so
        # two adjacent rects can never both claim a boundary point.
        self.assertIs(element_at(els, el.x - el.w / 2, el.y), el)
        self.assertIsNot(element_at(els, el.x + el.w / 2, el.y), el)
        self.assertIsNot(element_at(els, el.x, el.y + el.h / 2), el)


class StateUpdateTest(unittest.TestCase):
    def test_type_sets_field(self):
        env = MiniFormEnv()
        env.reset(seed=3)
        label, _ = env.task.field_targets[0]
        eid = field_id_for(env, label)
        tok_idx = 1
        action = N_ELEMENTS + eid * N_TOKENS + tok_idx
        env.step(action)
        self.assertEqual(env.field_values[label], gui_env.VOCAB[tok_idx])

    def test_type_on_non_field_is_safe_noop(self):
        env = MiniFormEnv()
        env.reset(seed=3)
        box_eid = next(e.eid for e in env.elements if e.kind == "box")
        before = dict(env.box_states)
        env.step(N_ELEMENTS + box_eid * N_TOKENS + 0)
        self.assertEqual(env.box_states, before)

    def test_click_toggles_box(self):
        env = MiniFormEnv()
        env.reset(seed=4)
        label = next(e.label for e in env.elements if e.kind == "box")
        eid = field_id_for(env, label)
        env.step(eid)
        self.assertTrue(env.box_states[label])
        env.step(eid)
        self.assertFalse(env.box_states[label])

    def test_click_distractor_is_noop(self):
        env = MiniFormEnv()
        env.reset(seed=4)
        eid = next(e.eid for e in env.elements if e.kind == "distract")
        before_f, before_b = dict(env.field_values), dict(env.box_states)
        _, reward, done, _ = env.step(eid)
        self.assertEqual((before_f, before_b),
                         (env.field_values, env.box_states))
        self.assertFalse(done)
        self.assertEqual(reward, 0.0)

    def test_invalid_action_rejected(self):
        env = MiniFormEnv()
        env.reset(seed=0)
        with self.assertRaises(AssertionError):
            env.step(N_ACTIONS)


class SuccessCheckerTest(unittest.TestCase):
    def _solve(self, env: MiniFormEnv) -> None:
        for label, token in env.task.field_targets:
            env.field_values[label] = token
        for label, wanted in env.task.box_targets:
            env.box_states[label] = wanted

    def test_correct_submit(self):
        env = MiniFormEnv(reward_mode="sparse", level=3)
        env.reset(seed=5)
        self._solve(env)
        _, reward, done, info = env.step(submit_id(env))
        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertEqual(reward, 1.0)

    def test_wrong_submit_sparse_retries(self):
        env = MiniFormEnv(reward_mode="sparse", level=3)
        env.reset(seed=5)
        # Leave the state untouched: cannot be correct. A wrong submit is
        # a form error, not terminal: the episode continues with reward 0.
        _, reward, done, info = env.step(submit_id(env))
        self.assertFalse(done)
        self.assertFalse(info["success"])
        self.assertEqual(reward, 0.0)

    def test_retry_after_wrong_submit_can_succeed(self):
        env = MiniFormEnv(reward_mode="sparse", level=0)
        env.reset(seed=5)
        env.step(submit_id(env))  # wrong: episode goes on
        self._solve(env)
        _, reward, done, info = env.step(submit_id(env))
        self.assertTrue(done and info["success"])
        self.assertEqual(reward, 1.0)

    def test_wrong_submit_hackable_pays(self):
        env = MiniFormEnv(reward_mode="shaped_hackable", level=3)
        env.reset(seed=5)
        _, reward, done, info = env.step(submit_id(env))
        self.assertFalse(done or info["success"])
        self.assertAlmostEqual(reward, gui_env.SUBMIT_BONUS - gui_env.STEP_PENALTY)
        # The bonus is farmable: every *failed* submit click pays, so
        # submit-spam out-earns solving the task (the lab's reward-hacking
        # demo).
        _, reward2, _, _ = env.step(submit_id(env))
        self.assertAlmostEqual(reward2, gui_env.SUBMIT_BONUS - gui_env.STEP_PENALTY)

    def test_correct_submit_hackable_pays_no_bonus(self):
        env = MiniFormEnv(reward_mode="shaped_hackable", level=3)
        env.reset(seed=5)
        for label, token in env.task.field_targets:
            env.field_values[label] = token
        for label, wanted in env.task.box_targets:
            env.box_states[label] = wanted
        _, reward, done, info = env.step(submit_id(env))
        self.assertTrue(done and info["success"])
        # A correct submit pays only the terminal reward (plus the
        # per-step penalty) -- no SUBMIT_BONUS on top.
        self.assertAlmostEqual(
            reward, gui_env.SUBMIT_REWARD - gui_env.STEP_PENALTY)

    def test_timeout_ends_episode(self):
        env = MiniFormEnv(level=0)
        env.reset(seed=6)
        eid = next(e.eid for e in env.elements if e.kind == "distract")
        done = False
        for _ in range(MAX_STEPS):
            _, _, done, info = env.step(eid)
        self.assertTrue(done)
        self.assertFalse(info["success"])

    def test_empty_field_never_matches(self):
        task = make_task(make_layout(7), 1, 7)
        fields = {lab: None for lab, _ in task.field_targets}
        self.assertFalse(check_success(fields, {}, task))
        self.assertEqual(count_satisfied(fields, {}, task), 0)


class DeterminismTest(unittest.TestCase):
    def test_same_seed_same_screen(self):
        a, b = make_layout(42), make_layout(42)
        self.assertEqual([(e.eid, e.kind, e.label, e.x, e.y) for e in a],
                         [(e.eid, e.kind, e.label, e.x, e.y) for e in b])
        ta = make_task(a, 3, 42)
        tb = make_task(b, 3, 42)
        self.assertEqual((ta.field_targets, ta.box_targets),
                         (tb.field_targets, tb.box_targets))

    def test_obs_shape_matches_const(self):
        env = MiniFormEnv()
        obs = env.reset(seed=8)
        self.assertEqual(obs.shape, (OBS_DIM,))

    def test_heldout_seeds_unseen(self):
        train = {(e.kind, e.label, e.x, e.y) for e in make_layout(1_000_000)}
        held = {(e.kind, e.label, e.x, e.y)
                for e in make_layout(gui_env.HELDOUT_SEED_BASE)}
        self.assertNotEqual(train, held)

    def test_action_names(self):
        self.assertEqual(action_name(0), "click(0)")
        self.assertEqual(action_name(N_ELEMENTS), "type(0, 'a')")


class ReturnsTest(unittest.TestCase):
    def test_hand_computed_returns(self):
        # G_2 = 1; G_1 = 0 + .5*1 = .5; G_0 = 1 + .5*.5 = 1.25
        self.assertEqual(compute_returns([1.0, 0.0, 1.0], 0.5), [1.25, 0.5, 1.0])
        self.assertEqual(compute_returns([0.0, 0.0, 1.0], 1.0), [1.0, 1.0, 1.0])
        self.assertEqual(compute_returns([], 0.99), [])


class ReinforceLossTest(unittest.TestCase):
    def test_exact_oracle(self):
        # Two steps, equal log-probs, returns [1, 0], baseline V = .5 both.
        # Raw advantages [.5, -.5] standardise to [+1, -1], so with zero
        # entropy bonus the policy loss is exactly 0 and the value loss is
        # mean((.5-1)^2, (.5-0)^2) = .25.
        log_probs = torch.log(torch.tensor([0.5, 0.5]))
        returns = torch.tensor([1.0, 0.0])
        values = torch.tensor([0.5, 0.5])
        entropy = torch.tensor([0.6931, 0.6931])
        p_loss, v_loss, adv = reinforce_losses(log_probs, returns, values,
                                               entropy, 0.0)
        self.assertTrue(torch.allclose(adv, torch.tensor([1.0, -1.0])))
        self.assertAlmostEqual(p_loss.item(), 0.0, places=5)
        self.assertAlmostEqual(v_loss.item(), 0.25, places=5)

    def test_advantage_standardisation(self):
        torch.manual_seed(0)
        logp = torch.randn(32)
        ret = torch.randn(32) * 5 + 100  # large offset must not matter
        val = torch.zeros(32)
        _, _, adv = reinforce_losses(logp, ret, val, torch.zeros(32), 0.0)
        self.assertAlmostEqual(adv.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(adv.std(unbiased=False).item(), 1.0, places=5)


class SmokeRunTest(unittest.TestCase):
    def test_train_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, str(CHAPTER_DIR / "train_gui_agent.py"),
                   "--smoke", "--device", "cpu", "--outdir", tmp]
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                                  text=True, timeout=300)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr[-3000:])
            for mode in ("sparse", "curriculum", "shaped", "hackable"):
                self.assertTrue((Path(tmp) / f"metrics_{mode}.csv").exists())
                self.assertTrue((Path(tmp) / f"eval_{mode}.csv").exists())
            summary = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual(set(summary["summary"]),
                             {"sparse", "curriculum", "shaped", "hackable"})
            self.assertIn("regime", proc.stdout)


if __name__ == "__main__":
    unittest.main()
