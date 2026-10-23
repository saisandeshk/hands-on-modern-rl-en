"""Tests for the Chapter 20 toy SWE-agent lab.

Run with:  .venv/bin/pytest code/chapter20_swe_agents -q

Covers the two unit tests required by the lab spec -- the sandboxed runner
(timeout, exception, pass/fail) and the REINFORCE update on a fixed batch --
plus the candidate-design contract (correct = 1.0 everywhere, hacks = 1.0 on
visible but < 1.0 on hidden) and an end-to-end smoke run of ``train.py``.
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys

import torch

import swe_policy as policy
import sandbox
import tasks
from tasks import FORBIDDEN_TOKENS, TASKS

TRAIN_PY = os.path.join(os.path.dirname(__file__), "train.py")


# ---------------------------------------------------------------------------
# Sandbox runner
# ---------------------------------------------------------------------------


def test_sandbox_pass_fail():
    src = "def solve(n):\n    return n * n\n"
    res = sandbox.run_candidate(src, [((3,), 9), ((4,), 999)], timeout=10.0)
    assert (res.passed, res.total) == (1, 2)
    assert res.ok_per_case == [True, False]
    assert res.pass_rate == 0.5
    assert not res.timed_out and not res.crashed


def test_sandbox_exception_is_test_failure_not_crash():
    src = "def solve(n):\n    raise ValueError('boom')\n"
    res = sandbox.run_candidate(src, [((1,), 1)], timeout=10.0)
    assert res.passed == 0
    assert not res.timed_out and not res.crashed
    assert res.errors[0] is not None and "ValueError" in res.errors[0]


def test_sandbox_timeout():
    src = "def solve(n):\n    while True:\n        pass\n"
    res = sandbox.run_candidate(src, [((1,), 1)], timeout=2.0)
    assert res.passed == 0
    assert res.timed_out and not res.crashed


def test_sandbox_runs_in_temp_cwd(tmp_path):
    # A candidate that writes to a relative path must land in the sandbox's
    # temp dir, never in the caller's working directory.
    marker = "evil_probe_ch20.txt"
    src = f"def solve(n):\n    open({marker!r}, 'w').write('x')\n    return n\n"
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        res = sandbox.run_candidate(src, [((1,), 1)], timeout=10.0)
        assert res.passed == 1
        assert not os.path.exists(os.path.join(tmp_path, marker))
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Candidate-design contract
# ---------------------------------------------------------------------------


def test_no_forbidden_tokens_in_candidates():
    for task in TASKS:
        for cand in task.candidates:
            for tok in FORBIDDEN_TOKENS:
                assert tok not in cand.source, (task.task_id, cand.name, tok)


def test_visible_hidden_inputs_disjoint():
    for task in TASKS:
        vis = {repr(a) for a, _ in task.visible}
        hid = {repr(a) for a, _ in task.hidden}
        # repr comparison: equivalent inputs would let a memorizer pass hidden.
        assert not (vis & hid), task.task_id


def test_candidate_reward_contract():
    # correct -> 1.0 on both suites and the unique maximizer on hidden;
    # hack/heuristic -> 1.0 visible, < 1.0 hidden; wrong -> strictly below
    # 1.0 on visible (partial credit). The hidden uniqueness check is the
    # lab's core design requirement: held-out tests must break every
    # shortcut *and* every plausible-but-wrong fallback.
    assert 10 <= len(TASKS) <= 20
    for task in TASKS:
        assert len(task.candidates) >= 3
        kinds = {c.kind for c in task.candidates}
        assert "correct" in kinds and "wrong" in kinds
        assert kinds & {"hack", "heuristic"}
        hidden_rates: dict[str, float] = {}
        for i, cand in enumerate(task.candidates):
            vis = sandbox.pass_rate(cand.source, task.visible, timeout=10.0)
            hid = sandbox.pass_rate(cand.source, task.hidden, timeout=10.0)
            hidden_rates[cand.name] = hid
            if cand.kind == "correct":
                assert vis == 1.0, (task.task_id, cand.name)
                assert hid == 1.0, (task.task_id, cand.name)
            elif cand.kind in ("hack", "heuristic"):
                assert vis == 1.0, (task.task_id, cand.name)
                assert hid < 1.0, (task.task_id, cand.name)
            else:
                assert vis < 1.0, (task.task_id, cand.name)
        correct_hid = max(
            hid for c, hid in zip(task.candidates, hidden_rates.values()) if c.kind == "correct"
        )
        assert correct_hid == 1.0, task.task_id
        for cand in task.candidates:
            if cand.kind != "correct":
                assert hidden_rates[cand.name] < correct_hid, (task.task_id, cand.name)


def test_correct_is_unique_maximizer_on_randomized_inputs():
    # Fresh inputs with oracle-expected outputs must also separate the
    # correct candidate from every shortcut and every wrong fallback
    # (fixed seed, enough samples that near-miss arms cannot tie by luck).
    import random

    rng = random.Random(0)
    for task in TASKS:
        oracle = tasks.ORACLES[task.oracle_name]
        argsets = tasks.random_inputs(task.task_id, rng, 50)
        cases = [(tuple(a), oracle(*a)) for a in argsets]
        rates = {
            cand.name: sandbox.pass_rate(cand.source, cases, timeout=10.0)
            for cand in task.candidates
        }
        correct = next(c for c in task.candidates if c.kind == "correct")
        assert rates[correct.name] == 1.0, task.task_id
        for cand in task.candidates:
            if cand.kind != "correct":
                assert rates[cand.name] < 1.0, (task.task_id, cand.name)


def test_random_oracle_agrees_with_correct_candidate():
    import random

    rng = random.Random(0)
    for task in TASKS:
        oracle = tasks.ORACLES[task.oracle_name]
        correct = next(c for c in task.candidates if c.kind == "correct")
        argsets = tasks.random_inputs(task.task_id, rng, 4)
        cases = [(tuple(a), oracle(*a)) for a in argsets]
        assert sandbox.pass_rate(correct.source, cases, timeout=10.0) == 1.0


# ---------------------------------------------------------------------------
# REINFORCE update on a fixed batch
# ---------------------------------------------------------------------------


def test_reinforce_loss_matches_hand_computation():
    # Fixed batch: logits [1, 0] -> p = softmax = [e/(e+1), 1/(e+1)].
    logits = torch.tensor([1.0, 0.0])
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs)  # actions [0, 1]
    rewards = torch.tensor([1.0, 0.0])
    baselines = torch.tensor([0.2, 0.2])
    loss, adv = policy.reinforce_loss(log_probs, rewards, baselines)

    e = math.e
    logp0 = 1.0 - math.log(e + 1.0)
    logp1 = -math.log(e + 1.0)
    expected = -((0.8 * logp0 + -0.2 * logp1) / 2.0)
    assert all(abs(a - b) < 1e-6 for a, b in zip(adv.tolist(), [0.8, -0.2]))
    assert abs(loss.item() - expected) < 1e-6


def test_reinforce_step_reinforces_positive_advantage():
    torch.manual_seed(0)
    pol = policy.SoftmaxPolicy([2])
    opt = torch.optim.SGD(pol.parameters(), lr=1.0)
    logp = torch.log(pol.probs()[0])  # [log 0.5, log 0.5], differentiable
    before = pol.probs()[0, 0].item()
    policy.reinforce_step(
        pol,
        opt,
        logp,
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.5, 0.5]),
    )
    after = pol.probs()[0, 0].item()
    assert after > before > 0.49


def test_masked_arms_get_zero_probability():
    pol = policy.SoftmaxPolicy([2, 3])
    p = pol.probs()
    assert p[0, 2].item() == 0.0
    assert abs(p[0, :2].sum().item() - 1.0) < 1e-6
    assert abs(p[1].sum().item() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# End-to-end smoke run
# ---------------------------------------------------------------------------


def test_train_smoke(tmp_path):
    outdir = tmp_path / "runs"
    proc = subprocess.run(
        [sys.executable, TRAIN_PY, "--smoke", "--seed", "0",
         "--device", "cpu", "--outdir", str(outdir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    for mode in ("visible", "hidden", "randomized"):
        csv_path = outdir / f"metrics_{mode}.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 12  # smoke epochs
        for col in ("train_reward", "exp_visible", "exp_hidden", "hidden_resolved"):
            assert col in rows[0]
            float(rows[-1][col])
    assert (outdir / "summary.json").exists()
    assert (outdir / "train_curves.png").exists()
    assert (outdir / "visible_vs_hidden.png").exists()
