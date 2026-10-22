"""Smoke + unit tests for the Chapter 21 deep-research lab.

Run from the repository root with::

    .venv/bin/pytest code/chapter21_deep_research -q

Covers: environment transitions and termination, the reward function
(correctness, step cost, shaping paid once), the discounted-return and
baseline computation checked against hand-computed values, determinism, the
scripted heuristic, and a subprocess run of the training script in ``--smoke``
mode (every runnable script in this directory).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from web_env import TinyResearchEnv, make_corpus
from policies import (
    ActorCriticMLP,
    HeuristicPolicy,
    RandomPolicy,
    build_features,
    discounted_returns,
    feature_dim,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus():
    return make_corpus(seed=0, scale="smoke")


@pytest.fixture(scope="module")
def easy_q(corpus):
    return next(q for q in corpus.questions if q.difficulty == "easy")


@pytest.fixture(scope="module")
def hard_q(corpus):
    return next(q for q in corpus.questions if q.difficulty == "hard")


def make_env(corpus, q, **kw):
    kw.setdefault("step_cost", 0.02)
    kw.setdefault("max_steps", 10)
    return TinyResearchEnv(corpus, [q], seed=0, **kw)


# --------------------------------------------------------------------------
# environment: search / transitions / termination
# --------------------------------------------------------------------------

def test_search_finds_paper_by_title(corpus, easy_q):
    env = make_env(corpus, easy_q)
    hits = env.search(easy_q.paper_title)
    assert hits, "paper-title search must return at least one document"
    assert hits[0] == easy_q.paper_id


def test_search_unknown_term_returns_empty(corpus, easy_q):
    env = make_env(corpus, easy_q)
    assert env.search("zzz-no-such-term") == []


def test_open_reveals_text_and_counts_once(corpus, easy_q):
    env = make_env(corpus, easy_q)
    env.reset(easy_q)
    env.results = [easy_q.paper_id]
    _, r1, done, info = env.step(4)  # open-0
    assert not done
    assert info["doc_id"] == easy_q.paper_id
    assert easy_q.paper_id in env.opened
    assert env.n_opens == 1
    _, r2, _, _ = env.step(4)  # re-open: no new information
    assert env.n_opens == 1
    assert r2 == pytest.approx(-env.step_cost)


def test_open_empty_slot_is_safe(corpus, easy_q):
    env = make_env(corpus, easy_q)
    env.reset(easy_q)
    env.results = []
    _, _, done, info = env.step(4)
    assert not done
    assert info.get("error") == "empty-slot"


def test_correct_answer_ends_episode_with_reward_one(corpus, easy_q):
    env = make_env(corpus, easy_q)
    env.reset(easy_q)
    idx = corpus.candidates.index(easy_q.gold_answer)
    action = env.N_SEARCH + env.n_slots + idx
    _, reward, done, info = env.step(action)
    assert done
    assert info["correct"] is True
    assert reward == pytest.approx(1.0 - env.step_cost)


def test_wrong_answer_gives_no_correctness_reward(corpus, easy_q):
    env = make_env(corpus, easy_q)
    env.reset(easy_q)
    wrong = next(c for c in corpus.candidates if c != easy_q.gold_answer)
    idx = corpus.candidates.index(wrong)
    action = env.N_SEARCH + env.n_slots + idx
    _, reward, done, info = env.step(action)
    assert done
    assert info["correct"] is False
    assert reward == pytest.approx(-env.step_cost)


def test_step_budget_truncates(corpus, easy_q):
    env = make_env(corpus, easy_q, max_steps=3)
    env.reset(easy_q)
    done = False
    for _ in range(3):
        _, _, done, info = env.step(0)  # repeated paper searches
    assert done
    assert info.get("truncated") is True


# --------------------------------------------------------------------------
# reward shaping: paid once, only for new documents
# --------------------------------------------------------------------------

def test_relevant_bonus_paid_once(corpus, easy_q):
    env = make_env(corpus, easy_q, rel_bonus=0.2)
    env.reset(easy_q)
    gold = easy_q.gold_doc_ids[0]
    env.results = [gold]
    _, r1, _, _ = env.step(4)
    assert r1 == pytest.approx(0.2 - env.step_cost)
    # Re-opening the same document pays nothing extra.
    _, r2, _, _ = env.step(4)
    assert r2 == pytest.approx(-env.step_cost)


def test_any_bonus_paid_for_any_new_doc(corpus, easy_q):
    # The any-document bonus pays for every newly opened document, including
    # gold documents (when rel_bonus is 0 the `elif self.any_bonus` branch
    # fires for them too) — which is exactly why the proxy is gameable.
    env = make_env(corpus, easy_q, any_bonus=0.4)
    env.reset(easy_q)
    other = next(d for d in corpus.docs if d not in easy_q.gold_doc_ids)
    env.results = [other]
    _, r, _, _ = env.step(4)
    assert r == pytest.approx(0.4 - env.step_cost)
    env2 = make_env(corpus, easy_q, any_bonus=0.4)
    env2.reset(easy_q)
    env2.results = [easy_q.gold_doc_ids[0]]
    _, r2, _, _ = env2.step(4)
    assert r2 == pytest.approx(0.4 - env.step_cost)


# --------------------------------------------------------------------------
# returns and baseline computation (hand-checked values)
# --------------------------------------------------------------------------

def test_discounted_returns_hand_computed():
    # G_2 = 1, G_1 = 0 + 0.9*1 = 0.9, G_0 = 0 + 0.9*0.9 = 0.81.
    assert discounted_returns([0.0, 0.0, 1.0], gamma=0.9) == \
        pytest.approx([0.81, 0.9, 1.0])
    # Immediate reward is not discounted away.
    assert discounted_returns([1.0, 0.0, 0.0], gamma=0.9) == \
        pytest.approx([1.0, 0.0, 0.0])
    # gamma = 1 accumulates the full episodic return.
    assert discounted_returns([-0.02, -0.02, 1.0], gamma=1.0) == \
        pytest.approx([0.96, 0.98, 1.0])
    assert discounted_returns([], gamma=0.99) == []


def test_value_loss_matches_hand_computed_mse():
    # One batch element: V(s) = 0.5, target G = 1.0 -> MSE = 0.25.
    values = torch.tensor([0.5, -0.5])
    targets = torch.tensor([1.0, 0.0])
    assert float(((values - targets) ** 2).mean()) == pytest.approx(0.25)


def test_advantage_normalization_direction():
    # Advantage = G - V must be positive for better-than-baseline steps.
    G = torch.tensor([1.0, 0.0])
    V = torch.tensor([0.2, 0.2])
    adv = G - V
    assert adv[0] > 0 > adv[1]


def test_actor_critic_shapes(corpus):
    cands = len(corpus.candidates)
    model = ActorCriticMLP(feature_dim(cands, 6), 4 + 6 + cands, hidden=16)
    x = torch.zeros(3, feature_dim(cands, 6))
    logits, values = model(x)
    assert logits.shape == (3, 4 + 6 + cands)
    assert values.shape == (3,)


def test_features_use_only_agent_visible_state(corpus, easy_q):
    env = make_env(corpus, easy_q)
    env.reset(easy_q)
    f = build_features(env)
    assert f.shape == (feature_dim(len(corpus.candidates), env.n_slots),)
    assert np.all(np.isfinite(f))
    # Nothing opened yet: no candidate may be marked as seen.
    assert f[7: 7 + len(corpus.candidates)].sum() == 0.0


# --------------------------------------------------------------------------
# determinism and heuristic quality
# --------------------------------------------------------------------------

def test_same_seed_same_corpus():
    a = make_corpus(seed=7, scale="smoke")
    b = make_corpus(seed=7, scale="smoke")
    assert [d.text for d in a.docs.values()] == [d.text for d in b.docs.values()]
    assert [q.gold_answer for q in a.questions] == [q.gold_answer for q in b.questions]


def test_heuristic_solves_all_smoke_questions(corpus):
    heur = HeuristicPolicy()
    env = TinyResearchEnv(corpus, corpus.questions, seed=0)
    for q in corpus.questions:
        env.reset(q)
        done = False
        info: dict = {}
        while not done:
            _, _, done, info = env.step(heur.select_action(env))
        assert info.get("correct") is True, f"heuristic failed on {q.qid}"


def test_heuristic_disambiguates_hard_question(corpus, hard_q):
    heur = HeuristicPolicy()
    env = make_env(corpus, hard_q)
    env.reset(hard_q)
    done = False
    while not done:
        _, _, done, info = env.step(heur.select_action(env))
    assert info.get("correct") is True
    # The corpus really is ambiguous: two profiles share the target name.
    twins = [d for d, doc in corpus.docs.items()
             if doc.kind == "profile" and hard_q.target_name in doc.text]
    assert len(twins) >= 2
    # The answer came from the profile whose affiliation matches the lab cue.
    used = [d for d in env.opened
            if d.startswith("profile-") and hard_q.lab in env.opened[d]]
    assert len(used) >= 1


def test_random_policy_actspace_coverage(corpus, easy_q):
    env = make_env(corpus, easy_q)
    pol = RandomPolicy(seed=1)
    actions = {pol.select_action(env) for _ in range(200)}
    assert len(actions) > 1
    assert all(0 <= a < env.n_actions for a in actions)


# --------------------------------------------------------------------------
# end-to-end: training script in smoke mode
# --------------------------------------------------------------------------

def test_train_script_smoke(tmp_path):
    cmd = [sys.executable, str(_HERE / "train_research_agent.py"),
           "--smoke", "--seed", "0", "--device", "cpu",
           "--outdir", str(tmp_path)]
    subprocess.run(cmd, check=True, timeout=110,
                   cwd=str(_HERE.parents[1]))
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert set(metrics["results"]) >= {
        "reinforce-outcome", "reinforce-relevant", "reinforce-any",
        "reinforce-scratch", "random", "heuristic"}
    for name, m in metrics["results"].items():
        for split in ("all", "easy", "hard"):
            if split in m:
                assert 0.0 <= m[split]["success_rate"] <= 1.0, name
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "learning_curves.png").exists()
    assert (tmp_path / "eval_success.png").exists()
