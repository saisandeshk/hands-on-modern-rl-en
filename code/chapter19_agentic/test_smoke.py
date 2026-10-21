"""Smoke tests for Chapter 19 (agentic RL).

Offline: pure logic plus a tiny local GPT-2 for the policy step test.
Scripts needing API keys, model downloads, or clusters (generate_synthetic
data generation, mini_deep_research main, run.py, trainer.fit e2e) are NOT
run; their pure helpers are tested and the rest skipped with reasons.
Run: .venv/bin/pytest code/chapter19_agentic -q
"""

import sys
import os
import re

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_system"))

CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]")

from tool_use_agent import TRAINING_QUERIES, ToolPolicy, simulate_tool_result
from multi_turn_rl import (
    compute_discounted_returns,
    compute_orm_rewards,
    compute_prm_rewards,
)


def test_queries_english_and_balanced():
    assert len(TRAINING_QUERIES) == 24
    for qd in TRAINING_QUERIES:
        assert not CJK.search(qd["query"])
    tools = [qd["correct_tool"] for qd in TRAINING_QUERIES]
    assert tools.count("search") == 8
    assert tools.count("calculate") == 8
    assert tools.count("run_code") == 8


def test_query_type_classifier_covers_dataset():
    policy = ToolPolicy()
    for qd in TRAINING_QUERIES:
        assert policy.get_query_type(qd["query"]) == qd["correct_tool"], qd["query"]


def test_policy_softmax_and_update_direction():
    np.random.seed(0)
    policy = ToolPolicy(n_tools=3, learning_rate=0.1)
    probs = policy.get_probabilities("search")
    assert abs(probs.sum() - 1.0) < 1e-9
    assert np.allclose(probs, [1 / 3, 1 / 3, 1 / 3])
    before = policy.get_probabilities("search")[0]
    policy.update("search", 0, 1.0)  # correct choice, positive reward
    after_good = policy.get_probabilities("search")[0]
    assert after_good > before
    policy.update("search", 0, -1.0)  # wrong choice, negative reward
    assert policy.get_probabilities("search")[0] < after_good


def test_simulate_tool_result():
    ok = simulate_tool_result("search", "q", "search")
    assert ok["success"] is True
    bad = simulate_tool_result("search", "q", "calculate")
    assert bad["success"] is False


def test_orm_only_final_step():
    turns = [{"correct": True}, {"correct": True}, {"correct": True}]
    assert compute_orm_rewards(turns) == [0.0, 0.0, 1.0]
    turns_fail = [{"correct": True}, {"correct": False}]
    assert compute_orm_rewards(turns_fail) == [0.0, 0.0]


def test_prm_dense_and_ordered():
    np.random.seed(0)
    turns = [{"correct": True}, {"correct": False}, {"correct": True}]
    rewards = compute_prm_rewards(turns)
    assert len(rewards) == 3
    assert all(0.0 <= r <= 1.0 for r in rewards)
    assert rewards[0] >= 0.7 > rewards[1]


def test_discounted_returns_by_hand():
    assert compute_discounted_returns([1.0, 1.0, 1.0], gamma=1.0) == [3.0, 2.0, 1.0]
    assert compute_discounted_returns([1.0, 1.0, 1.0], gamma=0.0) == [1.0, 1.0, 1.0]
    ret = compute_discounted_returns([0.0, 0.0, 1.0], gamma=0.5)
    assert ret == [0.25, 0.5, 1.0]


def test_deep_research_reward_helpers():
    from mini_deep_research_grpo import (
        accuracy_reward_func,
        citation_reward_func,
        completion_to_text,
        extract_xml_tag,
        format_reward_func,
    )

    good = "<think>plan</think><query>q</query><doc>d1</doc><answer>Paris</answer>"
    bad = "no tags at all"
    assert format_reward_func([good, bad]) == [1.0, 0.0]
    assert extract_xml_tag(good, "answer") == "Paris"
    assert extract_xml_tag(bad, "answer") == ""
    assert completion_to_text([{"content": good}]) == good
    assert completion_to_text(good) == good
    assert citation_reward_func([good], support_doc_id=["d1"]) == [1.0]
    assert citation_reward_func([good], support_doc_id=["d2"]) == [0.0]
    assert accuracy_reward_func([good], gold_answer=["paris"]) == [1.0]
    assert accuracy_reward_func([good], gold_answer=["London"]) == [0.0]


def test_deep_research_search_offline():
    from deep_research_rl_benchmark import DOCS, TASKS, search
    import random

    assert len(DOCS) > 0 and len(TASKS) > 0
    rng = random.Random(0)
    results = search("reinforcement learning", rng, limit=4)
    assert 0 < len(results) <= 4
    assert all(hasattr(d, "doc_id") for d in results)


def test_sandbox_env_executes_code():
    from environment import SandboxEnv

    env = SandboxEnv(timeout=10)
    out = env.step("execute_code", {"code": "print('hi')"})
    assert "hi" in out["observation"]
    assert out["done"] is False
    done = env.step("finish", {})
    assert done["done"] is True
    unknown = env.step("nope", {})
    assert unknown["done"] is False


def test_policy_grpo_step_offline():
    from transformers import GPT2Config, GPT2LMHeadModel
    from policy import Policy

    class WordTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __init__(self):
            self.vocab = {"<pad>": 0, "<eos>": 1}

        def _tok(self, text):
            ids = []
            for w in text.split():
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
                ids.append(self.vocab[w])
            return ids

        def __call__(self, text, return_tensors="pt", add_special_tokens=True):
            ids = self._tok(text)
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.tensor([[1] * len(ids)]),
            }

        def decode(self, ids, skip_special_tokens=True):
            inv = {i: w for w, i in self.vocab.items()}
            return " ".join(inv.get(int(i), "?") for i in ids)

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=128, n_positions=64, n_ctx=64,
        n_embd=32, n_layer=2, n_head=2,
    )
    model = GPT2LMHeadModel(config)
    tok = WordTokenizer()
    policy = Policy(model, tok, lr=1e-3)
    ref = GPT2LMHeadModel(config)
    ref.load_state_dict(model.state_dict())
    policy.set_ref_model(ref)
    trajectories = [
        ([("solve it", "print 55")], 1.0),
        ([("solve it", "wrong answer here")], -1.0),
    ]
    loss = policy.train_step_with_advantage(trajectories)
    assert loss != 0.0


def test_synthetic_data_module_import_safe():
    # generate_synthetic_data must import without the openai package or a key.
    import generate_synthetic_data as g

    assert g.get_model_name() == os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")


def test_external_service_scripts_skipped_reason():
    # Needs API key: generate_synthetic_data.main. Needs model download + GPU:
    # mini_deep_research_grpo.main, agentic_system/run.py, trainer.fit e2e.
    assert True
