"""Smoke tests for Chapter 15 (GRPO + RLVR).

Offline: uses fixed tensors and a tiny local GPT-2. Needs no network.
Run: .venv/bin/pytest code/chapter15_grpo -q
"""

import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grpo_math_reasoning import (
    build_completion_mask,
    build_smoke_model_and_tokenizer,
    compute_group_advantages,
    compute_reward,
    extract_answer_from_response,
    grpo_loss,
    grpo_policy_update,
    k3_kl_estimate,
    parse_args,
)
from rule_based_reward import (
    check_answer_correctness,
    check_format,
    compute_total_reward,
)
from grpo_mechanism import compute_grpo_advantages as mech_advantages


def test_advantages_basic():
    adv = compute_group_advantages(torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(adv, torch.tensor([1.0, -1.0, 1.0, -1.0]))
    assert abs(adv.mean().item()) < 1e-6


def test_advantages_all_equal_group():
    # All-equal rewards carry no preference signal: advantages must be ~0, not NaN.
    adv = compute_group_advantages(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    assert torch.all(torch.isfinite(adv))
    assert torch.allclose(adv, torch.zeros(4))


def test_advantages_single_group_member():
    adv = compute_group_advantages(torch.tensor([1.0]))
    assert torch.all(torch.isfinite(adv))
    assert torch.allclose(adv, torch.zeros(1))


def test_mechanism_demo_advantages_match():
    import numpy as np

    rewards = np.array([0.35, 0.52, 0.68, 0.41, 0.89, 0.73, 0.28, 0.61])
    adv = mech_advantages(rewards)
    assert abs(adv.mean()) < 1e-6
    assert adv[np.argmax(rewards)] > 0
    assert adv[np.argmin(rewards)] < 0


def test_ratio_clipping_positive_advantage():
    # Positive advantage: unclipped ratio above 1+eps must be clipped.
    new_logp = torch.log(torch.tensor([[0.9, 0.9]]))
    old_logp = torch.log(torch.tensor([[0.5, 0.5]]))
    ref_logp = old_logp.clone()
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 2)
    loss = grpo_loss(new_logp, old_logp, ref_logp, adv, mask,
                     clip_eps=0.2, kl_coef=0.0)
    # ratio = 1.8 -> clipped to 1.2; loss = -1.2 per token, mean = -1.2
    assert abs(loss.item() - (-1.2)) < 1e-5


def test_ratio_clipping_negative_advantage():
    # Negative advantage: ratio below 1-eps must be clipped from below.
    new_logp = torch.log(torch.tensor([[0.1, 0.1]]))
    old_logp = torch.log(torch.tensor([[0.5, 0.5]]))
    ref_logp = old_logp.clone()
    adv = torch.tensor([-2.0])
    mask = torch.ones(1, 2)
    loss = grpo_loss(new_logp, old_logp, ref_logp, adv, mask,
                     clip_eps=0.2, kl_coef=0.0)
    # ratio = 0.2 -> clipped to 0.8; loss = -min(0.2*-2, 0.8*-2) = +1.6
    assert abs(loss.item() - 1.6) < 1e-5


def test_ratio_inside_trust_region_unclipped():
    new_logp = torch.log(torch.tensor([[0.55]]))
    old_logp = torch.log(torch.tensor([[0.5]]))
    ref_logp = old_logp.clone()
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 1)
    loss = grpo_loss(new_logp, old_logp, ref_logp, adv, mask,
                     clip_eps=0.2, kl_coef=0.0)
    assert abs(loss.item() - (-1.1)) < 1e-5


def test_k3_non_negative_and_zero_on_match():
    torch.manual_seed(0)
    new_logp = torch.randn(2, 5)
    ref_logp = torch.randn(2, 5)
    kl = k3_kl_estimate(new_logp, ref_logp)
    assert bool((kl >= 0).all())
    kl_same = k3_kl_estimate(new_logp, new_logp.clone())
    assert torch.allclose(kl_same, torch.zeros_like(kl_same))


def test_masking_excludes_prompt_and_padding():
    new_logp = torch.zeros(1, 4)
    old_logp = torch.zeros(1, 4)
    ref_logp = torch.zeros(1, 4)
    adv = torch.tensor([1.0])
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    loss_masked = grpo_loss(new_logp, old_logp, ref_logp, adv, mask,
                            clip_eps=0.2, kl_coef=0.0)
    # Masked positions hold different (but valid) log-probs; they must not
    # change the loss.
    new_junk = torch.tensor([[-50.0, -80.0, 0.0, 0.0]])
    loss_junk = grpo_loss(new_junk, old_logp, ref_logp, adv, mask,
                          clip_eps=0.2, kl_coef=0.0)
    assert abs(loss_masked.item() - loss_junk.item()) < 1e-5


def test_build_completion_mask():
    ids = torch.tensor([[7, 8, 9, 0], [7, 8, 0, 0]])
    mask = build_completion_mask([2, 1], ids, pad_id=0)
    assert mask.tolist() == [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]


def test_aggregation_modes():
    # Two completions: 1 token vs 3 tokens, same per-token loss of 1.0.
    new_logp = torch.zeros(2, 3)
    old_logp = torch.zeros(2, 3)
    ref_logp = torch.zeros(2, 3)
    adv = torch.tensor([-1.0, -1.0])
    mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    seq = grpo_loss(new_logp, old_logp, ref_logp, adv, mask,
                    clip_eps=0.2, kl_coef=0.0, aggregation="sequence-mean")
    tok = grpo_loss(new_logp, old_logp, ref_logp, adv, mask,
                    clip_eps=0.2, kl_coef=0.0, aggregation="token-mean")
    # ratio=1, adv=-1 -> per-token loss +1 everywhere unmasked.
    assert abs(seq.item() - 1.0) < 1e-6
    assert abs(tok.item() - 1.0) < 1e-6
    # With differing per-token losses the two modes must disagree.
    new3 = torch.log(torch.tensor([[0.5, 0.5, 0.5], [0.9, 0.9, 0.9]]))
    old3 = torch.log(torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]))
    seq3 = grpo_loss(new3, old3, old3, torch.tensor([1.0, 1.0]), mask,
                     clip_eps=10.0, kl_coef=0.0, aggregation="sequence-mean")
    tok3 = grpo_loss(new3, old3, old3, torch.tensor([1.0, 1.0]), mask,
                     clip_eps=10.0, kl_coef=0.0, aggregation="token-mean")
    # seq-mean: -(1.0 + 1.8)/2 = -1.4; token-mean: -(1 + 3*1.8)/4 = -1.6
    assert abs(seq3.item() - (-1.4)) < 1e-5
    assert abs(tok3.item() - (-1.6)) < 1e-5


def test_answer_extraction_english():
    assert extract_answer_from_response(r"work ... \boxed{36} done") == "36"
    assert extract_answer_from_response("The answer is 42") == "42"
    assert extract_answer_from_response("Answer: 7") == "7"
    assert extract_answer_from_response("no numbers here!") is None


def test_compute_reward_english():
    assert compute_reward("Step 1: 15 + 27 = 42. The answer is 42", "42") == 1.0
    assert compute_reward("The answer is 35", "42") == 0.0
    assert compute_reward("nothing", "42") == 0.0


def test_rule_based_reward_components():
    good = (
        "First, compute 4 x 9.\n"
        "Step 1: 4 x 9 = 36\n"
        "Therefore the answer is \\boxed{36}"
    )
    res = check_answer_correctness(good, "36")
    assert res["correct"] and res["score"] == 1.0
    fmt = check_format(good)
    assert fmt["score"] > 0
    total = compute_total_reward(good, "36")
    bad = compute_total_reward("I do not know", "36")
    assert total["total_reward"] > bad["total_reward"]


def test_grpo_update_step_runs_offline():
    torch.manual_seed(0)
    model, tokenizer = build_smoke_model_and_tokenizer(device="cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    loss = grpo_policy_update(
        model, None, tokenizer, optimizer,
        "2 + 3 = ?",
        ["The answer is 5", "The answer is 6"],
        torch.tensor([1.0, -1.0]),
        clip_eps=0.2, kl_coef=0.01,
    )
    assert loss != 0.0
    assert any(not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters()))


def test_smoke_train_short():
    # End-to-end offline run with a tiny local model (CPU, few tokens).
    from grpo_math_reasoning import train

    args = parse_args(["--smoke", "--max-new-tokens", "8"])
    train(args)
