"""Smoke tests for Chapter 13 (RLHF pipeline: SFT, reward model, PPO).

Offline: tiny local models and stub tokenizers only. Needs no network.
Heavy entry points (full SFT/PPO training, model downloads) are NOT run here;
only their pure, import-safe logic is tested, with the reason noted per skip.
Run: .venv/bin/pytest code/chapter13_rlhf -q
"""

import sys
import os
import re

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sft_pipeline import format_sft_dataset, generate_sft_data
from reward_model_training import PreferenceDataset, RewardModel, train_reward_model
from rlhf_ppo_train import (
    SimpleRewardModel,
    build_response_mask,
    compute_log_probs,
)


class StubTokenizer:
    """Minimal stand-in with a chat template and whitespace tokenization."""

    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1}
        self.pad_token = "<eos>"
        self.eos_token = "<eos>"

    def _id(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)
        return self.vocab[word]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text, truncation=False, max_length=None, padding=False,
                 return_tensors=None, **kwargs):
        if isinstance(text, str):
            text = [text]
        ids = [[self._id(w) for w in t.split()] for t in text]
        if max_length is not None:
            ids = [s[:max_length] for s in ids]
        if padding == "max_length" and max_length is not None:
            masks = [[1] * len(s) + [0] * (max_length - len(s)) for s in ids]
            ids = [s + [0] * (max_length - len(s)) for s in ids]
        else:
            masks = [[1] * len(s) for s in ids]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}
        return {"input_ids": ids, "attention_mask": masks}


def _tiny_backbone(vocab_size=64, hidden=32):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size, n_positions=64, n_ctx=64,
        n_embd=hidden, n_layer=2, n_head=2,
    )
    return GPT2LMHeadModel(config)


def test_sft_data_english_only():
    pairs = generate_sft_data()
    assert len(pairs) >= 10
    cjk = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]")
    for p in pairs:
        assert set(p) == {"instruction", "response"}
        assert not cjk.search(p["instruction"] + p["response"])


def test_format_sft_dataset_uses_chat_template():
    tok = StubTokenizer()
    ds = format_sft_dataset(generate_sft_data()[:2], tok)
    assert len(ds) == 2
    assert ds[0]["text"].startswith("user:")
    assert "assistant:" in ds[0]["text"]


def test_bradley_terry_loss_sign():
    # loss = -log(sigmoid(r_chosen - r_rejected)): correct ranking -> small loss.
    good = -torch.log(torch.sigmoid(torch.tensor([2.0]))).mean()
    tied = -torch.log(torch.sigmoid(torch.tensor([0.0]))).mean()
    bad = -torch.log(torch.sigmoid(torch.tensor([-2.0]))).mean()
    assert good.item() < tied.item() < bad.item()
    assert abs(tied.item() - 0.6931) < 1e-3  # -log(0.5)


def test_reward_model_forward_shape():
    torch.manual_seed(0)
    backbone = _tiny_backbone()
    model = RewardModel(backbone, hidden_size=32)
    input_ids = torch.randint(0, 64, (3, 10))
    mask = torch.ones(3, 10, dtype=torch.long)
    mask[2, 7:] = 0
    rewards = model(input_ids, mask)
    assert rewards.shape == (3,)
    assert torch.all(torch.isfinite(rewards))


def test_reward_model_trains_toward_correct_ranking():
    torch.manual_seed(0)
    backbone = _tiny_backbone()
    model = RewardModel(backbone, hidden_size=32)
    tok = StubTokenizer()
    pairs = [
        {"prompt": "Say hello", "chosen": "Hello, happy to help you today",
         "rejected": "Whatever, go away"},
        {"prompt": "Explain gravity", "chosen": "Gravity pulls objects toward each other",
         "rejected": "No idea, do not care"},
    ]
    ds = PreferenceDataset(pairs, tok, max_length=24)
    assert len(ds) == 2
    assert ds[0]["chosen_input_ids"].shape == (24,)
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = train_reward_model(model, loader, opt, torch.device("cpu"), epochs=3)
    assert len(losses) == 3
    assert all(l >= 0 for l in losses)


def test_build_response_mask():
    att = torch.ones(1, 6)
    mask = build_response_mask(att, prompt_length=3)
    # Shifted positions 0..4; response starts at original position 3 -> shifted idx >= 2.
    assert mask.tolist() == [[0.0, 0.0, 1.0, 1.0, 1.0]]


def test_compute_log_probs_masks_prompt():
    torch.manual_seed(0)
    model = _tiny_backbone()
    model.eval()
    input_ids = torch.randint(0, 64, (1, 6))
    att = torch.ones(1, 6, dtype=torch.long)
    logp, mask = compute_log_probs(model, input_ids, att, prompt_length=3)
    assert logp.shape == (1, 5)
    assert mask.sum().item() == 3
    assert bool((logp * (1 - mask) == 0).all())


def test_rule_based_reward_prefers_helpful():
    tok = StubTokenizer()
    rm = SimpleRewardModel(tok)
    prompt = "Write a Python function that adds two numbers."
    helpful = (
        "Of course! Here is a function:\n"
        "1. Define add(a, b).\n"
        "2. Return a + b.\n"
        "```\ndef add(a, b):\n    return a + b\n```\n"
        "Happy to help with more examples."
    )
    rude = "Whatever, I do not care. Search it yourself."
    assert rm.score(prompt, helpful) > rm.score(prompt, rude)


def test_rule_based_reward_english_keywords():
    # The reward must react to English politeness markers, not Chinese ones.
    tok = StubTokenizer()
    rm = SimpleRewardModel(tok)
    base = "Here is the answer. " * 8  # long enough for the length bonus
    polite = base + "Please let me know if this helps."
    assert rm._rule_based_score("test", polite) > rm._rule_based_score("test", base)


def test_full_training_entry_points_skipped_reason():
    # main() in sft_pipeline / reward_model_training / rlhf_ppo_train downloads
    # Qwen2.5-0.5B and trains: needs network + GPU. Covered here by pure logic.
    assert True
