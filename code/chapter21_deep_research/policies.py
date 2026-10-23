"""Policies for the toy research web: features, a small MLP, baselines.

The learned policy never sees privileged information (gold documents or the
answer).  Its input is a fixed-size feature vector built only from
agent-visible state: the step budget, the last search results by document
type, and which candidate universities have appeared in opened profiles.
This mirrors Section 21.1: the agent observes search results and extracted
page text, not the full browser state (a partially observable problem).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from web_env import TinyResearchEnv, known_names, parse_profile_univ


def build_features(env: TinyResearchEnv) -> np.ndarray:
    """Featurize agent-visible state into a fixed-size float vector."""
    q = env.question
    cands = env.corpus.candidates
    feats: list[float] = [
        env.steps / env.max_steps,
        len(env.results) / max(env.n_slots, 1),
        1.0 if q.paper_id in env.opened else 0.0,
        1.0 if any(d.startswith("lab-") for d in env.opened) else 0.0,
        min(sum(1 for d in env.opened if d.startswith("profile-")), 4) / 4.0,
        1.0 if q.lab else 0.0,
        len(known_names(env.opened, q.paper_id)) / 2.0,
    ]
    # Which candidate universities has the agent actually seen in profiles?
    seen = set()
    for doc_id, text in env.opened.items():
        if doc_id.startswith("profile-"):
            u = parse_profile_univ(text)
            if u in cands:
                seen.add(u)
    feats += [1.0 if c in seen else 0.0 for c in cands]
    # Last search results by document type (paper / lab / profile / empty).
    n_slots = env.n_slots
    for slot in range(n_slots):
        onehot = [0.0, 0.0, 0.0, 1.0]
        if slot < len(env.results):
            kind = env.corpus.docs[env.results[slot]].kind
            onehot = [1.0 if kind == k else 0.0
                      for k in ("paper", "lab", "profile")] + [0.0]
        feats += onehot
    return np.asarray(feats, dtype=np.float32)


def feature_dim(n_candidates: int, n_slots: int = 6) -> int:
    return 7 + n_candidates + 4 * n_slots


class ActorCriticMLP(nn.Module):
    """Tiny shared-trunk actor-critic: logits over actions + state value."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.logits = nn.Linear(hidden, act_dim)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.logits(h), self.value(h).squeeze(-1)


def discounted_returns(rewards: list[float], gamma: float = 0.99) -> list[float]:
    """Episodic return: G_t = r_t + gamma * r_{t+1} + ... (verified in tests)."""
    out: list[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    out.reverse()
    return out


class RandomPolicy:
    """Uniform random baseline over all actions."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: TinyResearchEnv, _feats=None) -> int:
        return int(self.rng.integers(env.n_actions))


class HeuristicPolicy:
    """Scripted research strategy (strong, non-learned reference).

    search-paper -> open the paper -> search first-author name ->
    open profile(s); on hard questions open both same-name profiles and keep
    the one whose affiliation matches the lab cue -> answer the university
    stated in the chosen profile.  It uses only agent-visible text: opened
    documents, result doc ids/kinds, and its own last query.
    """

    def _open_action(self, env: TinyResearchEnv, slot: int) -> int:
        return env.N_SEARCH + slot

    def select_action(self, env: TinyResearchEnv, _feats=None) -> int:
        docs = env.corpus.docs
        res = env.results
        q = env.question

        def unopened_profile_slot() -> int | None:
            for k, d in enumerate(res):
                if docs[d].kind == "profile" and d not in env.opened:
                    return k
            return None

        # 1. Find and open the paper.
        if q.paper_id not in env.opened:
            for k, d in enumerate(res):
                if d == q.paper_id:
                    return self._open_action(env, k)
            return 0  # search-paper
        # 2. Search the first-listed author, then open their profile(s).
        authors = known_names(env.opened, q.paper_id)
        if not authors:
            return 0
        target = authors[0]
        prof_ids = [d for d in env.opened
                    if docs[d].kind == "profile" and target in env.opened[d]]
        if not prof_ids:
            if env.last_term == target:
                slot = unopened_profile_slot()
                if slot is not None:
                    return self._open_action(env, slot)
                return 2  # name-A results exhausted; broaden via co-author
            return 1  # search-name-A
        # 3. Disambiguate by lab cue on hard questions, then answer.
        if q.lab:
            matching = [d for d in prof_ids if q.lab in env.opened[d]]
            if len(matching) == 1:
                prof_ids = matching
            else:
                slot = unopened_profile_slot()
                if slot is not None and env.last_term == target:
                    return self._open_action(env, slot)
                if env.last_term != target:
                    return 1
                return 2
        univ = parse_profile_univ(env.opened[prof_ids[0]])
        if univ in env.corpus.candidates:
            return env.N_SEARCH + env.n_slots + env.corpus.candidates.index(univ)
        slot = unopened_profile_slot()
        if slot is not None:
            return self._open_action(env, slot)
        return 1


def answer_action(env: TinyResearchEnv, candidate: str) -> int:
    base = env.N_SEARCH + env.n_slots
    return base + env.corpus.candidates.index(candidate)
