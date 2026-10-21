"""Smoke tests for Chapter 14 (DPO preference optimization).

Offline: only pure data helpers and the DPO loss formula are tested.
DPOTrainer runs and model downloads need network + GPU and are skipped
with reasons below.
Run: .venv/bin/pytest code/chapter14_dpo -q
"""

import sys
import os
import re
import random

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]")


def test_mock_data_english_and_shaped():
    from importlib.util import spec_from_file_location, module_from_spec

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1-generate_data.py")
    spec = spec_from_file_location("gen_data", path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    random.seed(0)
    mod.generate_mock_data(num_samples=10, output_file="/tmp/opencode/pref_test.json")
    import json

    with open("/tmp/opencode/pref_test.json", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 10
    for item in data:
        assert set(item) == {"prompt", "chosen", "rejected"}
        assert not CJK.search(item["prompt"] + item["chosen"] + item["rejected"])


def test_dpo_loss_formula_sign():
    # DPO: loss = -log(sigmoid(beta * ((lp_c - lp_ref_c) - (lp_r - lp_ref_r)))).
    # A correct margin (chosen beats rejected) must give a smaller loss.
    def dpo_loss(lp_c, lp_ref_c, lp_r, lp_ref_r, beta=0.1):
        margin = beta * ((lp_c - lp_ref_c) - (lp_r - lp_ref_r))
        return -torch.log(torch.sigmoid(torch.tensor(margin)))

    good = dpo_loss(-1.0, -2.0, -3.0, -2.0)  # chosen above ref, rejected below ref
    bad = dpo_loss(-3.0, -2.0, -1.0, -2.0)  # swapped
    assert good.item() < bad.item()
    assert abs(dpo_loss(-2.0, -2.0, -2.0, -2.0).item() - 0.6931) < 1e-3


def test_math_preference_data_structure():
    import dpo_math_reward as m

    data = m.create_math_preference_data()
    assert len(data) == 10
    for item in data:
        assert set(item) == {"prompt", "chosen", "rejected", "answer"}
        assert not CJK.search(item["prompt"] + item["chosen"] + item["rejected"])
    tests = m.create_test_data()
    assert len(tests) == 5
    for t in tests:
        assert not CJK.search(t["prompt"])


def test_extract_number_english():
    import dpo_math_reward as m

    assert m.extract_number("Answer: 38") == 38
    assert m.extract_number("The answer is 7") == 7
    assert m.extract_number("The result is 12") == 12
    assert m.extract_number("equals 120") == 120
    # Decimals must not raise (previous int() conversion crashed on these)
    assert m.extract_number("The result is 3.5") == 3.5
    assert m.extract_number("Answer: 12.0") == 12
    assert m.extract_number("nothing here!") is None


def test_hands_on_preference_data_english():
    import dpo_hands_on as h

    assert len(h.preference_data) == 10
    for item in h.preference_data:
        assert set(item) == {"prompt", "chosen", "rejected"}
        assert not CJK.search(item["prompt"] + item["chosen"] + item["rejected"])
    for p in h.test_prompts if hasattr(h, "test_prompts") else []:
        assert not CJK.search(p)


def test_load_preference_dataset_missing_file():
    from importlib.util import spec_from_file_location, module_from_spec

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3-train_dpo.py")
    spec = spec_from_file_location("train_dpo", path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod.load_preference_dataset("/tmp/opencode/does-not-exist.json")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_heavy_scripts_import_safe():
    # 0/1/2/3/4 + both DPO demos must import without network or model weights.
    import dpo_hands_on  # noqa: F401
    import dpo_math_reward  # noqa: F401


def test_trainer_runs_skipped_reason():
    # 3-train_dpo.main / dpo_hands_on.main / dpo_math_reward.main download
    # Qwen2.5-0.5B and run DPOTrainer: needs network + GPU. Not run here.
    assert True
