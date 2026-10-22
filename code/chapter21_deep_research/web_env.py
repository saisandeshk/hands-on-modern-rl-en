"""Offline toy web for the Chapter 21 deep-research lab.

The environment is a small synthetic "web" of interlinked documents plus a set
of multi-hop questions.  Answering a question requires finding and combining
facts from 2-3 documents, mirroring the browser-RL harness in Section 21.1:

    search(term) -> open(doc) -> ... -> answer(candidate)

Only three tool types exist.  ``search`` matches a term against document text,
``open`` reveals a document's full text, and ``answer`` ends the episode with
a verifiable short answer (Section 21.2, Layer 1: exact match).

Question design (combining facts across documents)
-------------------------------------------------
* Scholars have profiles:  "<name> is a researcher at <lab> ... undergraduate
  degree at <university>."
* Papers list two authors but no affiliations:  "Paper <T>. Authors: <A> and
  <B>."
* Labs list members by name only.

Easy questions name a paper whose first author has a unique name, so the chain
is paper -> one profile (2 documents).  Hard questions target a first author
whose name is shared by two scholars and additionally name the author's lab,
so the agent must open the paper plus both same-name profiles and use the
affiliation to disambiguate (3 documents).  This reproduces the "Chen Li"
same-name disambiguation example from Section 21.1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


TOPICS = [
    "Optics", "Robotics", "Genomics", "Astronomy", "Cryptography", "Ecology",
    "Linguistics", "Catalysis", "Seismology", "Immunology", "Photonics",
    "Glaciology",
]

FIRST_NAMES = [
    "Ava", "Ben", "Chen", "Dara", "Elif", "Farah", "Gopal", "Hana",
    "Ivan", "June", "Kofi", "Lena", "Marco", "Nadia", "Omar", "Priya",
    "Ravi", "Sofia", "Tariq", "Uma", "Victor", "Wren", "Yuki", "Zara",
]

LAST_NAMES = [
    "Li", "Sharma", "Okafor", "Garcia", "Kim", "Novak", "Haddad", "Tanaka",
    "Muller", "Rossi", "Silva", "Khan", "Dubois", "Nguyen", "Ali", "Smith",
]

LAB_NAMES = [
    "Lab Atlas", "Lab Beacon", "Lab Cipher", "Lab Delta", "Lab Ember",
    "Lab Flux", "Lab Grove", "Lab Harbor", "Lab Ion", "Lab Juniper",
]

UNIVERSITIES = [
    "Meridian University", "Copperfield College", "Vostok Institute",
    "Alder State", "Beacon Hill University", "Cinderford College",
    "Dunmore Institute", "Eastvale University",
]

MAX_RESULTS = 6  # search returns at most this many documents

# Search ranking: entity profiles first, then papers, then lab pages, each by
# id.  This stands in for a real search engine's relevance ranking: a name
# query surfaces the people it names before labs that merely list them.


@dataclass
class Document:
    doc_id: str
    kind: str  # "paper" | "profile" | "lab"
    title: str
    text: str


@dataclass
class Question:
    qid: str
    paper_id: str
    paper_title: str
    target_scholar: int
    target_name: str
    lab: str | None  # set for hard questions (disambiguation cue)
    gold_answer: str
    difficulty: str  # "easy" | "hard"
    gold_doc_ids: list[str]
    text: str


@dataclass
class Corpus:
    docs: dict[str, Document]
    scholars: list[dict]  # {name, lab, univ}
    questions: list[Question]
    candidates: list[str]  # answer vocabulary (universities)
    labs: list[str]


def make_corpus(seed: int = 0, scale: str = "full") -> Corpus:
    """Build a deterministic synthetic web.  ``scale`` is "full" or "smoke"."""
    rng = random.Random(seed)
    if scale == "smoke":
        n_unique, n_pairs, n_papers, n_labs = 6, 2, 8, 4
        candidates = UNIVERSITIES[:4]
    elif scale == "full":
        n_unique, n_pairs, n_papers, n_labs = 40, 12, 60, 10
        candidates = list(UNIVERSITIES)
    else:
        raise ValueError(f"unknown scale {scale!r}")

    labs = LAB_NAMES[:n_labs]

    # --- scholars: unique names plus ambiguous pairs sharing one name ------
    first = FIRST_NAMES[:]
    last = LAST_NAMES[:]
    rng.shuffle(first)
    rng.shuffle(last)
    scholars: list[dict] = []
    used_names: set[str] = set()

    def fresh_name() -> str:
        while True:
            name = f"{rng.choice(first)} {rng.choice(last)}"
            if name not in used_names:
                used_names.add(name)
                return name

    for _ in range(n_unique):
        scholars.append({
            "name": fresh_name(),
            "lab": rng.choice(labs),
            "univ": rng.choice(candidates),
        })
    ambiguous_names: list[str] = []
    for _ in range(n_pairs):
        name = fresh_name()
        ambiguous_names.append(name)
        lab_a, lab_b = rng.sample(labs, 2)
        scholars.append({"name": name, "lab": lab_a, "univ": rng.choice(candidates)})
        univ_b = rng.choice([u for u in candidates if u != scholars[-1]["univ"]])
        scholars.append({"name": name, "lab": lab_b, "univ": univ_b})

    # --- documents -----------------------------------------------------------
    docs: dict[str, Document] = {}
    for i, s in enumerate(scholars):
        doc_id = f"profile-{i:02d}"
        text = (
            f"{s['name']} is a researcher at {s['lab']}. "
            f"{s['name']} completed their undergraduate degree at {s['univ']}. "
            f"Research focus: {rng.choice(TOPICS).lower()}."
        )
        docs[doc_id] = Document(doc_id, "profile", f"Profile of {s['name']}", text)
    scholar_profile = {i: f"profile-{i:02d}" for i in range(len(scholars))}

    unique_idx = list(range(n_unique))
    pair_first_idx = [n_unique + 2 * k for k in range(n_pairs)]
    rng.shuffle(unique_idx)
    rng.shuffle(pair_first_idx)

    questions: list[Question] = []
    for i in range(n_papers):
        hard = (i % 2 == 1) and pair_first_idx  # alternate easy / hard
        if hard:
            scholar_idx = pair_first_idx[(i // 2) % len(pair_first_idx)]
            used = [scholar_idx, scholar_idx + 1]
        else:
            scholar_idx = unique_idx[(i // 2) % len(unique_idx)]
            used = [scholar_idx]
        # Second author: anyone except the shared-name twin.
        others = [j for j in range(len(scholars)) if j not in used]
        coauthor = rng.choice(others)
        a1, a2 = scholars[scholar_idx], scholars[coauthor]
        title = f"Paper {rng.choice(TOPICS)}-{i:02d}"
        doc_id = f"paper-{i:02d}"
        docs[doc_id] = Document(
            doc_id, "paper", title,
            f"{title}. Authors: {a1['name']} and {a2['name']}. "
            f"Topic: {rng.choice(TOPICS).lower()}.",
        )
        target = scholars[scholar_idx]
        if hard:
            qtext = (
                f"In {title}, the first-listed author works at {target['lab']}. "
                f"Where did that person complete their undergraduate degree?"
            )
            gold_docs = [doc_id, scholar_profile[scholar_idx],
                         scholar_profile[scholar_idx + 1]]
            difficulty = "hard"
        else:
            qtext = (
                f"In {title}, where did the first-listed author "
                f"complete their undergraduate degree?"
            )
            gold_docs = [doc_id, scholar_profile[scholar_idx]]
            difficulty = "easy"
        questions.append(Question(
            qid=f"q-{i:02d}", paper_id=doc_id, paper_title=title,
            target_scholar=scholar_idx, target_name=target["name"],
            lab=target["lab"] if hard else None,
            gold_answer=target["univ"], difficulty=difficulty,
            gold_doc_ids=gold_docs, text=qtext,
        ))

    # --- lab documents (member lists only, no education facts) ---------------
    by_lab: dict[str, list[str]] = {lab: [] for lab in labs}
    for s in scholars:
        by_lab[s["lab"]].append(s["name"])
    for i, lab in enumerate(labs):
        members = sorted(set(by_lab[lab]))
        # Keep member lists short so search results stay readable.
        shown = members[:8]
        docs[f"lab-{i:02d}"] = Document(
            f"lab-{i:02d}", "lab", lab,
            f"{lab} is a research lab working on {rng.choice(TOPICS).lower()}. "
            f"Members include {', '.join(shown)}.",
        )

    return Corpus(docs, scholars, questions, candidates, labs)


class TinyResearchEnv:
    """Multi-turn research environment (Section 21.1 harness, minimal form).

    Action space (discrete indices, see :meth:`action_names`):

    * ``search-paper``: search the question's paper title.
    * ``search-name-A`` / ``search-name-B``: search author names seen in the
      opened paper document (a no-op empty search until the paper is open).
    * ``search-lab``: search the lab named in the question (hard questions).
    * ``open-k``: open the k-th document of the last search results.
    * ``answer-c``: submit ``candidates[c]``; terminates the episode.

    Reward: 1.0 for a correct final answer, 0.0 otherwise, minus a small
    per-step cost.  Optional shaping bonuses (Section 21.1 / Web-Shepherd
    discussion in 21.1) pay for opening documents; see :meth:`step`.
    """

    N_SEARCH = 4

    def __init__(self, corpus: Corpus, questions: list[Question],
                 max_steps: int = 10, step_cost: float = 0.02,
                 rel_bonus: float = 0.0, any_bonus: float = 0.0,
                 seed: int = 0):
        self.corpus = corpus
        self.questions = questions
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.rel_bonus = rel_bonus  # bonus for first-open of a gold doc
        self.any_bonus = any_bonus  # bonus for first-open of any new doc
        self.rng = random.Random(seed)
        self.n_candidates = len(corpus.candidates)
        self.n_slots = MAX_RESULTS
        self.n_actions = self.N_SEARCH + self.n_slots + self.n_candidates
        self.reset(self.rng.choice(self.questions) if questions else None)

    # -- action layout ------------------------------------------------------
    def action_names(self) -> list[str]:
        names = ["search-paper", "search-name-A", "search-name-B", "search-lab"]
        names += [f"open-{k}" for k in range(self.n_slots)]
        names += [f"answer:{c}" for c in self.corpus.candidates]
        return names

    # -- episode control ----------------------------------------------------
    def reset(self, question: Question | None = None):
        self.question = question or self.rng.choice(self.questions)
        self.results: list[str] = []
        self.opened: dict[str, str] = {}
        self.steps = 0
        self.done = False
        self.n_searches = 0
        self.n_opens = 0
        self.last_term = ""  # agent-visible: the agent knows what it searched
        self.shaped_paid: set[str] = set()
        return self._obs()

    def _obs(self) -> dict:
        return {
            "question": self.question.text,
            "results": list(self.results),
            "opened": dict(self.opened),
            "steps": self.steps,
            "done": self.done,
        }

    # -- tools ----------------------------------------------------------------
    def search(self, term: str) -> list[str]:
        """Substring search over document texts, deterministic ranking."""
        if not term:
            return []
        term = term.lower()
        rank = {"profile": 0, "paper": 1, "lab": 2}
        hits = [d for d in self.corpus.docs.values() if term in d.text.lower()]
        hits.sort(key=lambda d: (rank[d.kind], d.doc_id))
        return [d.doc_id for d in hits[:MAX_RESULTS]]

    def open_doc(self, doc_id: str) -> str:
        return self.corpus.docs[doc_id].text

    # -- step -----------------------------------------------------------------
    def _search_term(self, action: int) -> str:
        q = self.question
        if action == 0:
            return q.paper_title
        if action in (1, 2):
            names = known_names(self.opened, q.paper_id)
            idx = action - 1
            return names[idx] if idx < len(names) else ""
        if action == 3:
            return q.lab or ""
        return ""

    def step(self, action: int):
        """Take one action; returns (obs, reward, done, info)."""
        assert not self.done, "episode already terminated; call reset()"
        info: dict = {}
        reward = -self.step_cost  # per-step cost keeps trajectories short
        if action < self.N_SEARCH:
            self.last_term = self._search_term(action)
            self.results = self.search(self.last_term)
            self.n_searches += 1
        elif action < self.N_SEARCH + self.n_slots:
            slot = action - self.N_SEARCH
            if slot < len(self.results):
                doc_id = self.results[slot]
                if doc_id not in self.opened:
                    self.opened[doc_id] = self.open_doc(doc_id)
                    self.n_opens += 1
                    # Shaping: paid once per document (Section 21.1 ATR/PRM idea).
                    if doc_id in self.shaped_paid:
                        pass
                    elif doc_id in self.question.gold_doc_ids and self.rel_bonus:
                        reward += self.rel_bonus
                        self.shaped_paid.add(doc_id)
                    elif self.any_bonus:
                        reward += self.any_bonus
                        self.shaped_paid.add(doc_id)
                info["doc_id"] = doc_id
            else:
                info["error"] = "empty-slot"
        else:
            cand = self.corpus.candidates[action - self.N_SEARCH - self.n_slots]
            correct = (cand == self.question.gold_answer)
            reward += 1.0 if correct else 0.0
            self.done = True
            info["correct"] = correct
            info["answer"] = cand
        self.steps += 1
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            info["truncated"] = True
        return self._obs(), reward, self.done, info


def known_names(opened: dict[str, str], paper_id: str) -> list[str]:
    """Author names listed in the opened paper document (agent-visible)."""
    if paper_id not in opened:
        return []
    text = opened[paper_id]
    try:
        inner = text.split("Authors: ")[1].split(". Topic")[0]
        return [p.strip() for p in inner.split(" and ")]
    except IndexError:
        return []


def parse_profile_univ(text: str) -> str | None:
    """University named in a profile document (agent-visible)."""
    try:
        return text.split("undergraduate degree at ")[1].split(".")[0].strip()
    except IndexError:
        return None
