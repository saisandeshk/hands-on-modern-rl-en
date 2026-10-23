"""Smoke + unit tests for the Chapter 11 imitation lab (run on CPU).

Run from the repository root:
    .venv/bin/pytest code/chapter11_imitation_meta_rl -q
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = THIS_DIR / "bc_dagger_cartpole.py"
GAIL_SCRIPT = THIS_DIR / "gail_discriminator_cartpole.py"

from bc_dagger_cartpole import (  # noqa: E402
    PolicyNet,
    agreement,
    bc_loss_from_logits,
    collect_expert_trajectories,
    evaluate,
    expert_act,
    label_with_expert,
    rollout_states,
    seed_all,
    train_bc,
)

DEVICE = torch.device("cpu")


def run_smoke(script: Path, tmp_path: Path, timeout_s: int = 240) -> subprocess.CompletedProcess:
    outdir = tmp_path / script.stem
    proc = subprocess.run(
        [sys.executable, str(script), "--smoke", "--device", "cpu", "--outdir", str(outdir)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    assert proc.returncode == 0, f"{script.name} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc


# ---------------------------------------------------------------------------
# End-to-end smoke runs (each must finish in well under ~60 s on CPU).
# ---------------------------------------------------------------------------
def test_main_script_smoke(tmp_path):
    proc = run_smoke(MAIN_SCRIPT, tmp_path)
    outdir = tmp_path / MAIN_SCRIPT.stem
    assert "expert mean return" in proc.stdout
    assert (outdir / "metrics.json").exists()
    assert (outdir / "metrics.csv").exists()
    assert (outdir / "bc_dagger_results.png").exists()
    summary = json.loads((outdir / "metrics.json").read_text())
    assert summary["expert_mean_return"] == pytest.approx(500.0)
    assert len(summary["bc"]) == 2 and len(summary["dagger"]) == 3


def test_gail_script_smoke(tmp_path):
    proc = run_smoke(GAIL_SCRIPT, tmp_path)
    outdir = tmp_path / GAIL_SCRIPT.stem
    assert "discriminator accuracy" in proc.stdout
    assert (outdir / "gail_metrics.json").exists()
    assert (outdir / "gail_discriminator_reward.png").exists()
    summary = json.loads((outdir / "gail_metrics.json").read_text())
    # The discriminator must tell expert pairs from weak-learner pairs.
    assert summary["acc_expert"] > 0.7
    assert summary["mean_reward_expert"] > summary["mean_reward_learner"]


# ---------------------------------------------------------------------------
# Unit tests of the core pieces.
# ---------------------------------------------------------------------------
def test_expert_is_deterministic_and_optimal():
    obs = np.array([0.05, -0.1, 0.02, 0.15], dtype=np.float32)
    assert expert_act(obs) == expert_act(obs.copy())
    # Same seed twice -> identical episode return (deterministic expert + env).
    _, _, r1 = collect_expert_trajectories(2, start_seed=7)
    _, _, r2 = collect_expert_trajectories(2, start_seed=7)
    assert r1 == r2 == [500.0, 500.0]


def test_bc_loss_matches_hand_computation():
    torch.manual_seed(0)
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.5], [-1.0, 3.0]])
    actions = torch.tensor([0, 1, 1])
    got = bc_loss_from_logits(logits, actions).item()
    # Hand computation: -mean of the log-softmax at the taken actions.
    logp = torch.log_softmax(logits, dim=1)
    expected = -(logp[0, 0] + logp[1, 1] + logp[2, 1]).item() / 3
    assert got == pytest.approx(expected, rel=1e-6)
    # Sanity: confident-correct < uncertain, i.e. loss orders policies correctly.
    confident = bc_loss_from_logits(torch.tensor([[10.0, -10.0]]), torch.tensor([0])).item()
    unsure = bc_loss_from_logits(torch.tensor([[0.0, 0.0]]), torch.tensor([0])).item()
    assert confident < unsure


def test_label_with_expert_matches_controller():
    rng = np.random.RandomState(0)
    states = rng.uniform(low=-0.5, high=0.5, size=(20, 4)).astype(np.float32)
    labels = label_with_expert(states)
    assert labels.shape == (20,)
    assert set(np.unique(labels)) <= {0, 1}
    assert (labels == np.array([expert_act(s) for s in states])).all()


def test_dagger_aggregation_grows_dataset_and_improves_fit():
    """Dataset aggregation logic: new labelled states are appended, and a
    policy retrained on the aggregate fits the aggregate better."""
    seed_all(0)
    s_exp, a_exp, _ = collect_expert_trajectories(1, start_seed=123)
    policy = PolicyNet()
    train_bc(policy, s_exp, a_exp, DEVICE, epochs=5)
    own_s, _, _ = rollout_states(policy, 1, start_seed=999, device=DEVICE)
    assert len(own_s) > 0
    # Aggregate: the DAgger dataset is the union of expert + labelled learner states.
    s_agg = np.concatenate([s_exp, own_s])
    a_agg = np.concatenate([a_exp, label_with_expert(own_s)])
    assert len(s_agg) == len(s_exp) + len(own_s)
    assert len(a_agg) == len(s_agg)
    # Retraining on the aggregate must reduce (or match) the aggregate loss.
    before = bc_loss_from_logits(policy(torch.tensor(s_agg)), torch.tensor(a_agg)).item()
    train_bc(policy, s_agg, a_agg, DEVICE, epochs=10)
    after = bc_loss_from_logits(policy(torch.tensor(s_agg)), torch.tensor(a_agg)).item()
    assert after <= before


def test_agreement_is_fraction_on_fixed_states():
    seed_all(0)
    policy = PolicyNet()
    s, a, _ = collect_expert_trajectories(1, start_seed=321)
    agree = agreement(policy, s, a, DEVICE)
    assert 0.0 <= agree <= 1.0
    # A policy trained on exactly these states must agree more afterwards.
    train_bc(policy, s, a, DEVICE, epochs=30)
    assert agreement(policy, s, a, DEVICE) >= agree


def test_evaluate_returns_bounded():
    seed_all(0)
    policy = PolicyNet()
    rets = evaluate(policy, 2, start_seed=11, device=DEVICE)
    assert len(rets) == 2
    assert all(1.0 <= r <= 500.0 for r in rets)
