"""Smoke tests for Chapter 23 (VLM RL: geometry counting + multimodal reward).

Offline: synthetic images and pure reward logic only. No VLM weights or
downloads needed (vlm_grpo_train is a simulation; full VLM training needs a
40 GB+ GPU and is not run here).
Run: .venv/bin/pytest code/chapter23_vlm -q
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]")

from geometry_counting_dataset import SHAPE_DRAWERS, generate_single_image
from multi_modal_reward import (
    compute_total_reward,
    reward_correctness,
    reward_format,
    reward_reasoning_quality,
    reward_visual_grounding,
)
from vlm_grpo_train import (
    STANDARD_PROMPT,
    extract_numbers,
    generate_correct_response,
    generate_low_quality_response,
    simple_reward,
)


def test_shape_keys_english():
    assert set(SHAPE_DRAWERS) == {"triangle", "circle", "square"}
    assert not CJK.search(STANDARD_PROMPT)


def test_single_image_ground_truth_consistent():
    image, gt = generate_single_image(img_width=128, img_height=128, seed=0)
    assert set(gt) == {"triangle", "circle", "square"}
    assert all(0 <= v <= 5 for v in gt.values())
    assert image.size == (128, 128)


def test_reward_correctness_english():
    gt = {"triangle": 3, "circle": 1, "square": 2}
    good = "Triangles 3, circles 1, squares 2, total 6."
    assert reward_correctness(good, gt) == 1.0
    assert reward_correctness("Triangles 2, circles 1, squares 2.", gt) == 0.0
    assert reward_correctness("Triangles 3, circles 1.", gt) == 0.0  # missing shape


def test_reasoning_format_visual_ranges():
    rich = (
        "First, the triangles: I count 3 triangles on the left.\n"
        "Then, the circles: 1 blue circle above the center.\n"
        "Finally, the squares: 2 squares, one red and one green.\n"
        "Answer: triangles 3, circles 1, squares 2."
    )
    terse = "Triangles 3, circles 1, squares 2."
    assert reward_reasoning_quality(rich) > reward_reasoning_quality(terse)
    assert 0.0 <= reward_reasoning_quality(terse) <= 0.5
    assert reward_format(rich) >= reward_format("nothing to see here")
    assert 0.0 <= reward_format(rich) <= 0.2
    info = {"present_shapes": ["triangle", "circle", "square"], "absent_shapes": []}
    assert reward_visual_grounding(rich, info) > 0
    assert 0.0 <= reward_visual_grounding("Triangles 3.", info) <= 0.3


def test_absent_shape_grounding():
    info = {"present_shapes": ["circle", "square"], "absent_shapes": ["triangle"]}
    grounded = "Circles 2, squares 3, no triangles. Answer: circles 2, squares 3, triangles 0."
    assert reward_visual_grounding(grounded, info) > 0


def test_total_reward_ranks_perfect_first():
    from multi_modal_reward import test_cases

    totals = [
        compute_total_reward(tc["response"], tc["ground_truth"], tc["image_info"])["total"]
        for tc in test_cases
    ]
    assert totals[0] == max(totals)  # the perfect response must win
    for tc in test_cases:
        assert not CJK.search(tc["response"] + tc["name"])


def test_extract_numbers_both_directions():
    assert extract_numbers("triangles 3, circles 1", ["triangle", "circle"]) == {
        "triangle": 3, "circle": 1}
    assert extract_numbers("3 triangles and 1 circle", ["triangle", "circle"]) == {
        "triangle": 3, "circle": 1}


def test_simple_reward_prefers_correct():
    gt = {"triangle": 3, "circle": 1, "square": 2}
    good = generate_correct_response(gt)
    bad = generate_low_quality_response(gt)
    assert simple_reward(good, gt) >= 1.0
    assert simple_reward(good, gt) > simple_reward(bad, gt)


def test_dataset_generation_offline(tmp_path):
    from geometry_counting_dataset import generate_dataset

    out = str(tmp_path / "geo")
    meta = generate_dataset(output_dir=out, num_train=2, num_test=1,
                            img_width=64, img_height=64, base_seed=0)
    assert len(meta["train"]) == 2 and len(meta["test"]) == 1
    for split in ("train", "test"):
        for item in meta[split]:
            assert set(item["ground_truth"]) == {"triangle", "circle", "square"}
            assert not CJK.search(item["prompt"])
            assert os.path.exists(os.path.join(out, item["image_path"]))
