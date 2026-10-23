"""Tests for the Chapter 16 test-time-compute lab.

Run from the repository root with::

    .venv/bin/pytest code/chapter16_reasoning -q

The file mixes exact unit tests (estimator, voting, parser) with one
end-to-end subprocess test that runs ``run_lab.py --smoke`` on CPU.
"""

import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

import arith_cot
from arith_cot import (
    TinyReasoner,
    best_of_n_noisy,
    build_examples,
    coverage_at_n,
    decode,
    encode,
    make_splits,
    majority_answer,
    parse_answer,
    pass_at_k,
    render,
    set_seed,
    train_reasoner,
)

LAB_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = LAB_DIR / "run_lab.py"


# ---------------------------------------------------------------------------
# Trace format and parser
# ---------------------------------------------------------------------------

def test_render_hand_computed_traces():
    """render() must show units first, then tens-plus-carry, then the sum."""
    assert render(23, 58) == ("23+58=", "3+8=11>1c1;2+5+1=8>8c0;=81", 81)
    assert render(10, 10) == ("10+10=", "0+0=0>0c0;1+1+0=2>2c0;=20", 20)
    # Carry out of the tens column (99 + 99 = 198).
    assert render(99, 99) == (
        "99+99=", "9+9=18>8c1;9+9+1=19>9c1;=198", 198,
    )


def test_parse_answer_valid_and_broken():
    assert parse_answer("3+8=11>1c1;2+5+1=8>8c0;=81") == 81
    assert parse_answer("...;=198") == 198
    assert parse_answer(";=81 ") == 81  # trailing whitespace tolerated
    assert parse_answer("3+8=11>1c1 no answer here") is None
    assert parse_answer("") is None
    assert parse_answer(";=abc") is None


def test_tokenizer_roundtrip_and_eos_cutoff():
    ids = encode("3+8=11>1c1;=81")
    assert decode(ids) == "3+8=11>1c1;=81"
    # Decoding stops at the first <eos>; trailing babble never parses.
    assert decode(ids + [arith_cot.EOS_ID] + ids) == "3+8=11>1c1;=81"
    # Padding ids are skipped.
    assert decode([arith_cot.PAD_ID] + ids) == "3+8=11>1c1;=81"


def test_splits_are_disjoint():
    train, test = make_splits(500, 100, seed=7)
    assert len(train) == 500 and len(test) == 100
    assert set(train).isdisjoint(set(test))


# ---------------------------------------------------------------------------
# pass@k unbiased estimator
# ---------------------------------------------------------------------------

def brute_force_pass_at_k(flags: list[bool], k: int) -> float:
    """Fraction of k-subsets (enumerated explicitly) holding a success."""
    n = len(flags)
    if k >= n:
        return 1.0 if any(flags) else 0.0
    subsets = list(itertools.combinations(range(n), k))
    hits = sum(1 for s in subsets if any(flags[i] for i in s))
    return hits / len(subsets)


def test_pass_at_k_matches_brute_force():
    rng = __import__("random").Random(123)
    for trial in range(20):
        n = rng.randint(2, 8)
        flags = [rng.random() < 0.4 for _ in range(n)]
        c = sum(flags)
        for k in (1, 2, 3, 5):
            assert pass_at_k(n, c, k) == pytest.approx(
                brute_force_pass_at_k(flags, k)
            ), f"flags={flags} k={k}"


def test_pass_at_k_edge_cases():
    assert pass_at_k(10, 0, 3) == 0.0  # nothing correct: no coverage
    assert pass_at_k(10, 10, 3) == 1.0  # everything correct
    assert pass_at_k(8, 3, 1) == pytest.approx(3 / 8)  # k=1 is the mean
    assert pass_at_k(5, 2, 5) == 1.0  # k >= n with a hit: covered
    assert pass_at_k(5, 0, 5) == 0.0
    # Hand check: n=4, c=1, k=2 -> 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5.
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Majority vote and verifier rules
# ---------------------------------------------------------------------------

def test_majority_vote_basic_and_ties():
    assert majority_answer([81, 80, 81, 79, 81]) == 81
    assert majority_answer([None, 81, None, 81]) == 81  # Nones abstain
    assert majority_answer([None, None]) is None  # all abstain: no vote
    # Ties break deterministically toward the smallest answer.
    assert majority_answer([80, 81]) == 80
    assert majority_answer([81, 80]) == 80
    assert majority_answer([82, 80, 81, 80, 81, 82]) == 80


def test_best_of_n_noisy_reduces_to_coverage_when_exact():
    flags = [False, True, False, False, True]
    accepted = list(flags)  # exact verifier: accepts iff truly correct
    assert best_of_n_noisy(flags, accepted, 1) is False
    assert best_of_n_noisy(flags, accepted, 2) is True
    assert coverage_at_n(flags, 1) == 0.0
    assert coverage_at_n(flags, 2) == 1.0


def test_best_of_n_noisy_false_positive_and_fallback():
    # A false accept on sample 0 makes best-of-1 wrong even though a
    # correct answer exists later in the pool.
    assert best_of_n_noisy([False, True], [True, True], 2) is False
    # Total rejection falls back to sample 0.
    assert best_of_n_noisy([True, False], [False, False], 2) is True
    assert best_of_n_noisy([False, True], [False, False], 2) is False


# ---------------------------------------------------------------------------
# Tiny end-to-end learning check (CPU, seconds)
# ---------------------------------------------------------------------------

def test_training_overfits_a_handful_of_problems():
    """Teacher forcing must drive loss down on a tiny fixed set."""
    set_seed(11)
    device = torch.device("cpu")
    pairs = [(23, 58), (10, 10), (99, 99), (46, 92),
             (39, 27), (64, 95), (84, 85), (53, 94)]
    examples = build_examples(pairs)
    model = TinyReasoner().to(device)
    losses = train_reasoner(
        model, examples, epochs=60, batch_size=8, lr=3e-3,
        device=device, seed=11,
    )
    assert losses[-1] < 0.5 * losses[0]
    # Greedy decoding reproduces most of the training traces exactly.
    model.eval()
    hits = 0
    for (a, b) in pairs:
        prompt, _, ans = render(a, b)
        ids = torch.tensor([encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            full = model.generate(ids, torch.tensor([6]), 29, temperature=0.0)
        if parse_answer(decode(full[0, 6:])) == ans:
            hits += 1
    assert hits >= 6, f"greedy reproduced only {hits}/8 training traces"


# ---------------------------------------------------------------------------
# Smoke run of the full pipeline in a subprocess
# ---------------------------------------------------------------------------

def test_smoke_run_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, str(RUN_SCRIPT), "--smoke", "--device", "cpu",
            "--seed", "0", "--outdir", tmp,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, f"smoke run failed:\n{proc.stderr[-4000:]}"
        for name in ("metrics.json", "scaling.csv", "temperature.csv",
                     "scaling_curves.png", "temperature.png"):
            assert (Path(tmp) / name).exists(), f"missing output {name}"
        metrics = json.loads((Path(tmp) / "metrics.json").read_text())
        assert metrics["config"]["n_samples"] == 8
        ns = [row["N"] for row in metrics["scaling"]]
        assert ns == [1, 2, 4, 8]
        for row in metrics["scaling"]:
            for key, val in row.items():
                if key != "N":
                    assert 0.0 <= val <= 1.0, f"{key}={val} out of range"
        # pass@k is monotone non-decreasing in N by construction.
        paks = [row["pass_at_k"] for row in metrics["scaling"]]
        assert all(b >= a for a, b in zip(paks, paks[1:]))
        assert "0.0" in proc.stdout  # summary table reached stdout
