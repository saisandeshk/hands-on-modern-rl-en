"""Tests for the Chapter 10 offline-RL lab.

Run with:  .venv/bin/pytest code/chapter10_offline_rl -q

Covers:
- hand-computed unit checks (CQL penalty, WIS, dataset loader shapes);
- smoke runs of every script in a subprocess (each finishes in <60 s CPU).
"""

import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUTDIR = os.path.join(REPO_ROOT, "runs", "chapter10_offline_rl")
# Smoke runs use an isolated outdir so they never clobber full-run results.
SMOKE_OUTDIR = os.path.join(REPO_ROOT, "runs", "chapter10_offline_rl_smoke_test")

sys.path.insert(0, HERE)

TIMEOUT = 600


def run_script(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Unit tests with hand-computed answers
# ---------------------------------------------------------------------------


def test_cql_penalty_matches_hand_computation():
    """CQL penalty on a hand-made Q table checked against NumPy by hand."""
    from train_offline import cql_penalty

    q = torch.tensor([[1.0, 2.0], [0.5, 0.5]])
    a = torch.tensor([1, 0])
    got = cql_penalty(q, a, alpha=1.0).item()
    # Row 0: logsumexp = log(e^1 + e^2) = 2 + log(1 + e^-1) ~= 2.313262
    #   gap = 2.313262 - 2.0 = 0.313262
    # Row 1: logsumexp = log(2*e^0.5) = 0.5 + log 2 ~= 1.193147
    #   gap = 1.193147 - 0.5 = 0.693147
    # mean = (0.313262 + 0.693147) / 2 = 0.503204
    lse = np.log(np.exp([1.0, 2.0]).sum()), 0.5 + np.log(2.0)
    expected = ((lse[0] - 2.0) + (lse[1] - 0.5)) / 2.0
    assert abs(got - expected) < 1e-5, f"{got} vs {expected}"
    # alpha scaling is linear.
    assert abs(cql_penalty(q, a, alpha=2.0).item() - 2.0 * expected) < 1e-5


def test_wis_matches_hand_computation():
    """WIS on a toy with a known answer: sum(w*G)/sum(w)."""
    from ope import wis_estimate, pdis_estimate

    returns = np.array([10.0, 0.0, 5.0])
    weights = np.array([1.0, 1.0, 2.0])
    assert abs(wis_estimate(returns, weights) - (10.0 + 0.0 + 10.0) / 4.0) < 1e-9

    # PDIS with all ratios = 1 reduces to the plain discounted return mean.
    rew = [np.array([1.0, 1.0]), np.array([2.0])]
    rat = [np.array([1.0, 1.0]), np.array([1.0])]
    gamma = 0.99
    expected = ((1.0 + 0.99) + 2.0) / 2.0
    assert abs(pdis_estimate(rew, rat, gamma) - expected) < 1e-9

    # PDIS down-weights a late reward when the early ratio is small.
    rew = [np.array([0.0, 10.0])]
    rat = [np.array([0.5, 1.0])]
    expected = (gamma**1) * 0.5 * 10.0
    assert abs(pdis_estimate(rew, rat, gamma) - expected) < 1e-9


def test_dataset_loader_shapes(tmp_path):
    """Loader accepts a well-formed dataset and rejects inconsistent ones."""
    from common import load_dataset, save_dataset

    n, sd, ad = 64, 4, 2
    arrays = {
        "states": np.zeros((n, sd), dtype=np.float32),
        "actions": np.zeros(n, dtype=np.int64),
        "rewards": np.ones(n, dtype=np.float32),
        "next_states": np.zeros((n, sd), dtype=np.float32),
        "dones": np.zeros(n, dtype=np.float32),
        "behaviour_probs": np.full((n, ad), 0.5, dtype=np.float32),
    }
    path = str(tmp_path / "toy.npz")
    save_dataset(path, arrays, {"behaviour": "toy"})
    got, meta = load_dataset(path)
    assert got["states"].shape == (n, sd)
    assert got["behaviour_probs"].shape == (n, ad)
    assert meta["behaviour"] == "toy"

    bad = dict(arrays)
    bad["rewards"] = np.ones(n + 1, dtype=np.float32)
    bad_path = str(tmp_path / "bad.npz")
    np.savez_compressed(bad_path, **bad)
    try:
        load_dataset(bad_path)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for inconsistent lengths")


# ---------------------------------------------------------------------------
# Smoke runs of every script
# ---------------------------------------------------------------------------


def test_smoke_collect():
    r = run_script(["collect_datasets.py", "--smoke", "--seed", "0", "--device", "cpu",
                    "--outdir", SMOKE_OUTDIR])
    assert r.returncode == 0, r.stderr[-3000:]
    for name in ("random", "medium", "expert", "mixed"):
        assert os.path.exists(os.path.join(SMOKE_OUTDIR, "datasets", f"{name}.npz"))
    assert os.path.exists(os.path.join(SMOKE_OUTDIR, "expert_q.pt"))


def test_smoke_train_offline():
    r = run_script(["train_offline.py", "--smoke", "--dataset", "all", "--seed", "0",
                    "--device", "cpu", "--outdir", SMOKE_OUTDIR])
    assert r.returncode == 0, r.stderr[-3000:]
    assert os.path.exists(os.path.join(SMOKE_OUTDIR, "offline", "summary.json"))
    with open(os.path.join(SMOKE_OUTDIR, "offline", "summary.json")) as f:
        summary = json.load(f)
    assert set(summary["results"]) == {"random", "medium", "expert", "mixed"}
    for ds, res in summary["results"].items():
        for m in ("bc", "dqn", "cql"):
            assert "mean" in res[m], (ds, m)


def test_smoke_ope():
    r = run_script(["ope.py", "--smoke", "--seed", "0", "--device", "cpu",
                    "--outdir", SMOKE_OUTDIR])
    assert r.returncode == 0, r.stderr[-3000:]
    assert os.path.exists(os.path.join(SMOKE_OUTDIR, "ope", "ope_summary.json"))
    with open(os.path.join(SMOKE_OUTDIR, "ope", "ope_summary.json")) as f:
        summary = json.load(f)
    assert "truth" in summary and len(summary["rows"]) == 4
