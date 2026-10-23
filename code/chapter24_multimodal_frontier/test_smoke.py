"""Tests for the Chapter 24 DDPO toy lab.

Run with::

    .venv/bin/pytest code/chapter24_multimodal_frontier -q

Covers the three unit-test targets from the lab spec (Gaussian log-prob of a
denoising step, the reward function, the per-step policy-gradient loss on a
fixed batch) plus the KL/diversity helpers, rollout shapes, and an end-to-end
smoke run of the training script in a subprocess.
"""

import math
import os
import subprocess
import sys

import torch

import ddpo_toy

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SCRIPT = os.path.join(
    REPO_ROOT, "code", "chapter24_multimodal_frontier", "ddpo_toy.py"
)


def test_gaussian_log_prob_matches_hand_value():
    # N(0, I) in 2D evaluated at the mean: -0.5 * 0 - log(2 * pi).
    x = torch.zeros(1, 2)
    got = ddpo_toy.gaussian_log_prob(x, torch.zeros(1, 2), 1.0)
    assert math.isclose(got.item(), math.log(1.0 / (2.0 * math.pi)), rel_tol=1e-6)

    # x = (1, 0), var = 0.5: -0.5 * (2 + 2 * log(pi)).
    x = torch.tensor([[1.0, 0.0]])
    got = ddpo_toy.gaussian_log_prob(x, torch.zeros(1, 2), 0.5)
    assert math.isclose(got.item(), -0.5 * (2.0 + 2.0 * math.log(math.pi)),
                        rel_tol=1e-6)

    # Batch + broadcasting over the mean.
    xs = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    got = ddpo_toy.gaussian_log_prob(xs, torch.zeros(2), 1.0)
    assert got.shape == (2,)
    assert got[0].item() > got[1].item()


def test_target_reward_shape_and_hand_values():
    center = torch.as_tensor(ddpo_toy.TARGET_CENTER).unsqueeze(0)
    assert math.isclose(ddpo_toy.target_reward(center).item(), 1.0, rel_tol=1e-6)

    # One point two widths away along x1: exp(-(2w)^2 / 2w^2) = exp(-2).
    off = center + torch.tensor([[2.0 * ddpo_toy.REWARD_WIDTH, 0.0]])
    assert math.isclose(ddpo_toy.target_reward(off).item(), math.exp(-2.0),
                        rel_tol=1e-6)

    # Rewards always lie in [0, 1].
    batch = torch.randn(64, 2) * 4.0
    r = ddpo_toy.target_reward(batch)
    assert r.shape == (64,)
    assert bool(((r >= 0.0) & (r <= 1.0)).all())


def test_advantages_match_hand_computation():
    # Rewards 0.3, 0.5, 0.7 (the worked example from Section 24.4.5):
    # mean 0.5, population std sqrt(0.08 / 3), advantages -1.2247, 0, +1.2247.
    r = torch.tensor([0.3, 0.5, 0.7])
    adv = ddpo_toy.compute_advantages(r, eps=0.0)
    assert math.isclose(adv[0].item(), -(0.2 / math.sqrt(0.08 / 3.0)), rel_tol=1e-6)
    assert abs(adv[1].item()) < 1e-6
    assert math.isclose(adv[2].item(), -adv[0].item(), rel_tol=1e-6)
    assert abs(adv.mean().item()) < 1e-6


def test_pg_loss_no_clipping_binding():
    # ratio == 1 everywhere: loss = -mean(advantage) = -(1 - 1) / 2 = 0.
    new = torch.zeros(2, 3)
    old = torch.zeros(2, 3)
    adv = torch.tensor([1.0, -1.0])
    assert abs(ddpo_toy.pg_loss(new, old, adv, clip_eps=0.2).item()) < 1e-9


def test_pg_loss_clips_positive_advantage():
    # ratio 3 > 1 + 0.2 with advantage +2: min(3*2, 1.2*2) = 2.4 -> loss -2.4.
    new = torch.full((1, 2), math.log(3.0))
    old = torch.zeros(1, 2)
    adv = torch.tensor([2.0])
    loss = ddpo_toy.pg_loss(new, old, adv, clip_eps=0.2)
    assert math.isclose(loss.item(), -2.4, rel_tol=1e-6)


def test_pg_loss_clipping_does_not_bind_for_negative_advantage():
    # ratio 3 with advantage -1: min(-3, -1.2) = -3 -> loss +3. PPO only
    # clips moves that would improve the objective too far in one step.
    new = torch.full((1, 2), math.log(3.0))
    old = torch.zeros(1, 2)
    adv = torch.tensor([-1.0])
    loss = ddpo_toy.pg_loss(new, old, adv, clip_eps=0.2)
    assert math.isclose(loss.item(), 3.0, rel_tol=1e-6)


def test_pg_loss_adds_kl_penalty():
    new = torch.zeros(2, 3)
    old = torch.zeros(2, 3)
    adv = torch.tensor([1.0, -1.0])
    kl = torch.ones(2, 3)
    loss = ddpo_toy.pg_loss(new, old, adv, clip_eps=0.2, beta=0.5, kl=kl)
    assert math.isclose(loss.item(), 0.5, rel_tol=1e-6)
    try:
        ddpo_toy.pg_loss(new, old, adv, beta=0.5, kl=None)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when kl is missing")


def test_gaussian_kl_matches_hand_value():
    new = torch.tensor([[1.0, 0.0]])
    old = torch.tensor([[0.0, 0.0]])
    assert math.isclose(ddpo_toy.gaussian_kl(new, old, 1.0).item(), 0.5, rel_tol=1e-6)
    assert math.isclose(ddpo_toy.gaussian_kl(new, old, 0.25).item(), 2.0, rel_tol=1e-6)
    assert ddpo_toy.gaussian_kl(new, new, 1.0).item() == 0.0


def test_rollout_shapes_and_reward_range():
    device = torch.device("cpu")
    ddpo_toy.set_seed(0)
    sched = ddpo_toy.DiffusionSchedule(4, 0.01, 0.35).to(device)
    model = ddpo_toy.DenoiserMLP(hidden=16).to(device)
    out = ddpo_toy.rollout(model, sched, 8, device)
    assert out["xs"].shape == (5, 8, 2)
    assert out["old_logps"].shape == (4, 8)
    assert out["x0"].shape == (8, 2)
    assert bool(((out["rewards"] >= 0.0) & (out["rewards"] <= 1.0)).all())
    assert torch.isfinite(out["old_logps"]).all()


def test_make_plots_handles_skipped_nokl(tmp_path):
    device = torch.device("cpu")
    ddpo_toy.set_seed(0)
    sched = ddpo_toy.DiffusionSchedule(4, 0.01, 0.35).to(device)
    model = ddpo_toy.DenoiserMLP(hidden=16).to(device)
    ref = ddpo_toy.DenoiserMLP(hidden=16).to(device)
    ev = ddpo_toy.evaluate(model, ref, sched, 32, device)
    outdir = str(tmp_path / "nokl-skipped")
    os.makedirs(outdir, exist_ok=True)
    hist = [
        {"iter": 1, "reward_mean": 0.1, "diversity": 3.0, "kl": 0.01},
    ]
    ddpo_toy.make_plots(ev, ev, None, hist, [], outdir)
    assert os.path.isfile(os.path.join(outdir, "samples_before_after.png"))
    assert os.path.isfile(os.path.join(outdir, "training_curves.png"))


def test_smoke_run_end_to_end(tmp_path):
    outdir = str(tmp_path / "smoke-out")
    cmd = [
        sys.executable, SCRIPT, "--smoke", "--device", "cpu", "--seed", "1",
        "--outdir", outdir,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "checkpoint" in proc.stdout
    for name in ("metrics.csv", "summary.json", "samples_before_after.png",
                 "training_curves.png"):
        assert os.path.isfile(os.path.join(outdir, name)), name
    with open(os.path.join(outdir, "metrics.csv")) as f:
        header = f.readline().strip().split(",")
    assert header == ["run", "iter", "reward_mean", "reward_std",
                      "frac_target", "diversity", "kl", "loss"]
