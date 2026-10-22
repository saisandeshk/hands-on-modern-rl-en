"""Smoke + unit tests for the Chapter 18 rollout/learner lab.

Run:  .venv/bin/pytest code/chapter18_llm_rl_industrial -q
"""

import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from rollout_system import (
    ActorCritic,
    Batch,
    LagTracker,
    behaviour_version_at,
    collect_batch,
    importance_ratios,
    learner_update,
    ppo_clipped_weights,
    simulate_training,
    vtrace_truncated_weights,
)

DEVICE = torch.device("cpu")
CHAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(CHAPTER_DIR, "rollout_system.py")


# ---------------------------------------------------------------------------
# Unit tests: staleness math checked against hand-computed values
# ---------------------------------------------------------------------------

def test_importance_ratio_hand_values():
    # pi = 0.6, pi_beh = 0.4  ->  rho = 1.5
    new_lp = torch.tensor([math.log(0.6)])
    beh_lp = torch.tensor([math.log(0.4)])
    rho = importance_ratios(new_lp, beh_lp)
    assert rho.item() == pytest.approx(1.5)
    # identical policies -> ratio exactly 1
    assert importance_ratios(new_lp, new_lp).item() == pytest.approx(1.0)


def test_ppo_clip_hand_values():
    rho = torch.tensor([1.5, 0.5, 1.0])
    clipped = ppo_clipped_weights(rho, eps=0.2)
    # 1.5 -> 1.2 (upper bound), 0.5 -> 0.8 (lower bound), 1.0 unchanged
    assert clipped.tolist() == pytest.approx([1.2, 0.8, 1.0])


def test_vtrace_truncation_hand_values():
    # rho = 1.5 with c_bar = rho_bar = 1.0 -> both truncate to 1.0
    new_lp = torch.tensor([math.log(0.6)])
    beh_lp = torch.tensor([math.log(0.4)])
    c, rho_tr = vtrace_truncated_weights(new_lp, beh_lp, 1.0, 1.0)
    assert c.item() == pytest.approx(1.0)
    assert rho_tr.item() == pytest.approx(1.0)
    # rho = 0.5 passes through untouched
    new_lp2 = torch.tensor([math.log(0.2)])
    beh_lp2 = torch.tensor([math.log(0.4)])
    c2, rho_tr2 = vtrace_truncated_weights(new_lp2, beh_lp2, 1.0, 1.0)
    assert c2.item() == pytest.approx(0.5)
    assert rho_tr2.item() == pytest.approx(0.5)


def test_behaviour_version_bookkeeping():
    # delay 0, broadcast every update: workers always hold the current version
    assert [behaviour_version_at(t, 1, 0) for t in range(4)] == [0, 1, 2, 3]
    # delay 2: workers lag two versions behind (floored at 0)
    assert [behaviour_version_at(t, 1, 2) for t in range(5)] == [0, 0, 0, 1, 2]
    # broadcast every 2 updates, no delay: versions refresh at t = 0, 2, 4
    assert [behaviour_version_at(t, 2, 0) for t in range(5)] == [0, 0, 2, 2, 4]


def test_lag_tracker_summary():
    tr = LagTracker()
    tr.record(learner_version=3, beh_version=1)  # lag 2
    tr.record(learner_version=4, beh_version=4)  # lag 0
    s = tr.summary()
    assert s["mean_lag"] == pytest.approx(1.0)
    assert s["max_lag"] == 2
    assert s["n"] == 2


def _fixed_batch() -> Batch:
    rng = np.random.RandomState(0)
    n = 64
    return Batch(states=rng.randn(n, 4).astype(np.float32),
                 actions=rng.randint(0, 2, size=n),
                 returns=rng.randn(n),
                 beh_logprobs=np.full(n, math.log(0.5), dtype=np.float32))


def test_one_worker_equals_sync_update():
    """A 1-worker run with zero delay must equal the synchronous update.

    When the behaviour policy is the current policy, the stored behaviour
    log-probs match fresh log-probs, every ratio is 1, and the async-clip
    update is bit-for-bit the sync update on the same data.
    """
    batch = _fixed_batch()
    torch.manual_seed(7)
    net_sync = ActorCritic(4, 2)
    torch.manual_seed(7)
    net_async = ActorCritic(4, 2)
    assert net_sync is not net_async
    # Refresh the stored behaviour log-probs from the live policy (delay 0:
    # behaviour == learner), exactly what collect_batch would have recorded.
    with torch.no_grad():
        fresh_lp, _, _ = net_async.evaluate(
            torch.as_tensor(batch.states), torch.as_tensor(batch.actions))
    batch.beh_logprobs = fresh_lp.numpy().astype(np.float32)

    opt_s = torch.optim.Adam(net_sync.parameters(), lr=3e-3)
    opt_a = torch.optim.Adam(net_async.parameters(), lr=3e-3)
    learner_update(net_sync, opt_s, batch, "sync", DEVICE, epochs=2)
    learner_update(net_async, opt_a, batch, "clip", DEVICE, epochs=2)
    for p_s, p_a in zip(net_sync.parameters(), net_async.parameters()):
        assert torch.equal(p_s, p_a)


def test_simulation_deterministic():
    kw = dict(env_name="CartPole-v1", num_updates=3, steps_per_update=256,
              mode="clip", broadcast_every=1, delay=2, device=DEVICE,
              eval_episodes=1)
    r1 = simulate_training(seed=0, **kw)
    r2 = simulate_training(seed=0, **kw)
    assert r1["eval_curve"] == pytest.approx(r2["eval_curve"])
    assert r1["lags"] == r2["lags"] == [0, 1, 2]  # learner pulls ahead of delay-2 snapshot


def test_collect_batch_records_behaviour_logprobs():
    torch.manual_seed(0)
    policy = ActorCritic(4, 2)
    batch = collect_batch(policy, "CartPole-v1", 128, seed=0, beh_version=3)
    assert batch.beh_version == 3
    assert len(batch.states) == len(batch.actions) == 128
    assert np.all(np.isfinite(batch.beh_logprobs))
    assert np.all(np.isfinite(batch.returns))


# ---------------------------------------------------------------------------
# Smoke tests: every script runs end-to-end in <60 s on CPU
# ---------------------------------------------------------------------------

def _run_smoke(experiment: str, tmp_path, tag: str) -> dict:
    outdir = str(tmp_path / tag)
    cmd = [sys.executable, SCRIPT, "--experiment", experiment, "--smoke",
           "--outdir", outdir, "--seed", "0"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                       cwd=CHAPTER_DIR)
    assert r.returncode == 0, f"smoke {experiment} failed:\n{r.stderr[-3000:]}"
    return {"outdir": outdir, "stdout": r.stdout}


@pytest.mark.parametrize("experiment,expected", [
    ("throughput", ["throughput.csv"]),
    ("staleness", ["staleness_curves.csv", "staleness_curves.png"]),
    ("sweep", ["staleness_sweep.csv", "staleness_sweep.png"]),
])
def test_smoke_experiment(tmp_path, experiment, expected):
    res = _run_smoke(experiment, tmp_path, experiment)
    for fname in expected + ["summary.json"]:
        assert os.path.exists(os.path.join(res["outdir"], fname)), \
            f"missing {fname} after smoke {experiment}"
    assert "Wrote CSV/JSON metrics and PNG plots" in res["stdout"]
