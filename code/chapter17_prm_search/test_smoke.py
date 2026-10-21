"""Smoke tests for Chapter 17 (Tree-of-Thought search).

Fully offline: pure Python + numpy. No model or network needed.
Run: .venv/bin/pytest code/chapter17_prm_search -q
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tree_of_thought import (
    ThoughtNode,
    TwentyFourGame,
    evaluate_node,
    generate_children,
    search_chain_of_thought,
    search_random,
    search_tree_of_thought,
)


def test_apply_op():
    assert TwentyFourGame.apply_op(3, 5, "+") == 8
    assert TwentyFourGame.apply_op(3, 5, "-") == -2
    assert TwentyFourGame.apply_op(3, 5, "*") == 15
    assert TwentyFourGame.apply_op(6, 3, "/") == 2
    assert TwentyFourGame.apply_op(1, 0, "/") is None  # division by zero guarded
    assert TwentyFourGame.is_close_to_target(24.0 + 1e-9)
    assert not TwentyFourGame.is_close_to_target(23.0)


def test_generate_children_counts():
    node = ThoughtNode([1, 2, 3, 4])
    children = generate_children(node)
    # Ordered pairs 4*3 = 12, times 4 ops = 48 (no zero divisor here).
    assert len(children) == 48
    for child in children:
        assert len(child.numbers) == 3
        assert len(child.history) == 1
    assert generate_children(ThoughtNode([24])) == []  # terminal: no children


def test_evaluate_node_terminal():
    solved = ThoughtNode([24.0], history=["done"])
    assert evaluate_node(solved) == 1.0
    far = ThoughtNode([0.0], history=["done"])
    assert 0.0 <= evaluate_node(far) < 0.6  # 1 - 24/50 = 0.52: low, but not zero
    mid = ThoughtNode([1, 2])
    assert 0.0 <= evaluate_node(mid) <= 1.0


def test_tot_finds_exact_solution():
    # [1, 2, 3, 4] is solvable: 1*2*3*4 = 24.
    result, tree_data = search_tree_of_thought(
        [1, 2, 3, 4], breadth=3, max_depth=4, target=24.0, verbose=False
    )
    assert result.is_terminal()
    assert TwentyFourGame.is_close_to_target(result.numbers[0], 24.0)
    assert len(tree_data["nodes_per_level"]) >= 1


def test_cot_returns_terminal_node():
    result = search_chain_of_thought([1, 2, 3, 4], target=24.0, verbose=False)
    assert result.is_terminal()
    assert len(result.numbers) == 1


def test_random_search_contract():
    np.random.seed(0)
    best_result, best_diff, successes = search_random(
        [1, 2, 3, 4], n_trials=10, target=24.0, verbose=False
    )
    assert best_result is None or isinstance(best_result, (int, float, np.floating))
    assert best_diff >= 0
    assert 0 <= successes <= 10
