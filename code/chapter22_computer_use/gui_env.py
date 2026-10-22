"""MiniForm: a tiny text-rendered GUI environment for computer-use RL.

Chapter 22 formulates computer use as an MDP over screenshots with mouse and
keyboard actions and a sparse terminal reward ("did the task complete?"). Real
desktop benchmarks (OSWorld, AndroidWorld) are far too heavy for a laptop
lab, so this module distils the same structure into a miniature form-filling
task:

* The "screen" is a grid of UI elements: text fields, checkboxes, a submit
  button, and distractor buttons. Every episode shuffles element positions,
  Set-of-Mark style element ids, labels, and task targets from a seed, so
  memorising one layout never suffices.
* The observation is a structured screen description: per element its type,
  position, label, and current state, plus the task specification. This plays
  the role of the screenshot + accessibility tree in Chapter 22, Step 1.
* Actions are ``click(element_id)`` and ``type(element_id, token)`` over a
  small vocabulary -- the discrete, SoM-style action space of Chapter 22,
  Step 2 -- flattened into one discrete action index.
* Reward is sparse by default: +1 only for a correct submission, 0
  otherwise. Optional shaping pays for newly satisfied sub-goals; a
  deliberately hackable variant also pays +0.4 for every *failed* submit
  click (a correct submit still pays only the +1 terminal reward), which lets
  the lab reproduce the "submit early" reward-hacking failure mode.

Notation follows the book: states ``s``, actions ``a``, rewards ``r``,
discount ``gamma``, policy ``pi_theta``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Screen and task constants
# ---------------------------------------------------------------------------

COLS, ROWS = 8, 6            # screen grid size (cells)
N_FIELDS = 2                 # text fields per screen
N_BOXES = 2                  # checkboxes per screen
N_DISTRACT = 2               # distractor buttons (clicking them does nothing)
N_ELEMENTS = N_FIELDS + N_BOXES + 1 + N_DISTRACT  # +1 for the submit button

FIELD_LABELS = ("name", "city", "code")
BOX_LABELS = ("fast", "gift", "news", "sale")
SUBMIT_LABEL = "submit"
DISTRACT_LABELS = ("help", "info")
LABEL_POOL = FIELD_LABELS + BOX_LABELS + (SUBMIT_LABEL,) + DISTRACT_LABELS
VOCAB = ("a", "b", "c")      # tokens accepted by type()

#: Curriculum levels: which sub-goals are active. Level 0 is one field plus
#: submit; the full task (level 3) needs both fields and both boxes.
LEVELS = ("fill-one", "fill-two", "fill-and-check-one", "full")
N_LEVELS = len(LEVELS)
#: Active (field-target, box-target) counts per level.
LEVEL_TASKS = ((1, 0), (2, 0), (2, 1), (2, 2))

MAX_STEPS = 16               # episode horizon T
SUBMIT_REWARD = 1.0          # sparse terminal reward for a correct submission
PROGRESS_REWARD = 0.25       # shaping: paid once per newly satisfied sub-goal
SUBMIT_BONUS = 0.4           # hackable shaping: paid for ANY submit click
STEP_PENALTY = 0.01          # small per-step cost in shaped modes

N_TOKENS = len(VOCAB)
N_CLICK_ACTIONS = N_ELEMENTS
N_TYPE_ACTIONS = N_ELEMENTS * N_TOKENS
N_ACTIONS = N_CLICK_ACTIONS + N_TYPE_ACTIONS

#: Observation layout: per element (kind 4 + pos 2 + label 10 + state 5),
#: field targets 2 x (label 10 + token 3 + active 1), box targets
#: 2 x (label 10 + target 1 + active 1), level one-hot 4, steps-left 1.
OBS_DIM = N_ELEMENTS * 21 + 2 * 14 + 2 * 12 + N_LEVELS + 1

HELDOUT_SEED_BASE = 900_000  # eval layouts use seeds from a disjoint range


# ---------------------------------------------------------------------------
# Layout helpers (deterministic in the episode seed)
# ---------------------------------------------------------------------------

@dataclass
class Element:
    """One on-screen UI element (a Set-of-Mark style labelled box)."""

    eid: int          # SoM label 0..N_ELEMENTS-1, reshuffled every episode
    kind: str         # "field" | "box" | "submit" | "distract"
    label: str        # human-readable label, e.g. "city"
    x: float          # rect centre x in [0, COLS]
    y: float          # rect centre y in [0, ROWS]
    w: float = 0.9
    h: float = 0.8


@dataclass
class Task:
    """What the episode asks for: target tokens/states keyed by label."""

    level: int
    field_targets: list = field(default_factory=list)  # [(label, token)]
    box_targets: list = field(default_factory=list)    # [(label, checked)]


def make_layout(seed: int) -> list[Element]:
    """Place all elements on the grid. Deterministic in ``seed``.

    Positions, SoM ids, and which label each slot shows are all drawn from
    one ``random.Random(seed)`` stream, so the same seed always rebuilds the
    same screen while nearby seeds give different screens.
    """
    import random

    rng = random.Random(seed)
    cells = [(cx, cy) for cx in range(COLS) for cy in range(ROWS)]
    rng.shuffle(cells)
    kinds = (["field"] * N_FIELDS + ["box"] * N_BOXES + ["submit"]
             + ["distract"] * N_DISTRACT)
    labels = (rng.sample(list(FIELD_LABELS), N_FIELDS)
              + rng.sample(list(BOX_LABELS), N_BOXES)
              + [SUBMIT_LABEL] + list(DISTRACT_LABELS))
    slots = list(zip(kinds, labels))
    rng.shuffle(slots)  # which slot lands on which element id
    elements = []
    for eid, ((cx, cy), (kind, label)) in enumerate(zip(cells, slots)):
        elements.append(Element(eid=eid, kind=kind, label=label,
                                x=cx + 0.5, y=cy + 0.5))
    return elements


def element_at(elements: list[Element], x: float, y: float) -> Element | None:
    """Hit-test: return the element whose rect contains point (x, y).

    This is the miniature analogue of coordinate grounding (Chapter 22,
    Step 2): a continuous click position resolves to a discrete element.
    Rect edges are half-open [x-w/2, x+w/2) x [y-h/2, y+h/2) so adjacent
    rects never both claim a boundary point.
    """
    for el in elements:
        if (el.x - el.w / 2 <= x < el.x + el.w / 2
                and el.y - el.h / 2 <= y < el.y + el.h / 2):
            return el
    return None


def make_task(elements: list[Element], level: int, seed: int) -> Task:
    """Sample target values for the active sub-goals at ``level``."""
    import random

    rng = random.Random(seed + 1_000_003)  # separate stream from the layout
    n_fields, n_boxes = LEVEL_TASKS[level]
    field_labels = [e.label for e in elements if e.kind == "field"]
    box_labels = [e.label for e in elements if e.kind == "box"]
    field_targets = [(lab, rng.choice(VOCAB)) for lab in field_labels[:n_fields]]
    box_targets = [(lab, rng.choice([True, False])) for lab in box_labels[:n_boxes]]
    return Task(level=level, field_targets=field_targets, box_targets=box_targets)


# ---------------------------------------------------------------------------
# State, success checking, and observation encoding
# ---------------------------------------------------------------------------

def check_success(field_values: dict[str, str | None],
                  box_states: dict[str, bool], task: Task) -> bool:
    """Verifier: True iff every active sub-goal currently holds its target.

    ``field_values``/``box_states`` are keyed by element *label*. An empty
    field (None) never matches a target token.
    """
    for label, token in task.field_targets:
        if field_values.get(label) != token:
            return False
    for label, wanted in task.box_targets:
        if box_states.get(label, False) != wanted:
            return False
    return True


def count_satisfied(field_values: dict[str, str | None],
                    box_states: dict[str, bool], task: Task) -> int:
    """Number of active sub-goals currently satisfied (for shaping)."""
    n = sum(1 for label, token in task.field_targets
            if field_values.get(label) == token)
    n += sum(1 for label, wanted in task.box_targets
             if box_states.get(label, False) == wanted)
    return n


def encode_obs(elements: list[Element], field_values: dict[str, str | None],
               box_states: dict[str, bool], task: Task,
               steps_left: int) -> np.ndarray:
    """Flatten the structured screen description into a feature vector."""
    label_index = {lab: i for i, lab in enumerate(LABEL_POOL)}
    kinds = ("field", "box", "submit", "distract")
    obs: list[float] = []
    for el in sorted(elements, key=lambda e: e.eid):
        one_hot = [1.0 if el.kind == k else 0.0 for k in kinds]
        obs += one_hot + [el.x / COLS, el.y / ROWS]
        lab = [0.0] * len(LABEL_POOL)
        lab[label_index[el.label]] = 1.0
        obs += lab
        if el.kind == "field":
            val = field_values.get(el.label)
            state = [1.0 if val == t else 0.0 for t in (None,) + VOCAB]
            obs.append(0.0)
        else:
            state = [0.0] * (N_TOKENS + 1)
            obs.append(1.0 if (el.kind == "box"
                               and box_states.get(el.label, False)) else 0.0)
        obs += state
    for targets, is_field in ((task.field_targets, True),
                              (task.box_targets, False)):
        for k in range(2):
            lab = [0.0] * len(LABEL_POOL)
            if k < len(targets):
                lab[label_index[targets[k][0]]] = 1.0
                active = 1.0
            else:
                active = 0.0
            obs += lab
            if is_field:
                tok = [0.0] * N_TOKENS
                if k < len(targets):
                    tok[VOCAB.index(targets[k][1])] = 1.0
                obs += tok + [active]
            else:
                want = 1.0 if (k < len(targets) and targets[k][1]) else 0.0
                obs += [want, active]
    lvl = [0.0] * N_LEVELS
    lvl[task.level] = 1.0
    obs += lvl + [steps_left / MAX_STEPS]
    return np.asarray(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MiniFormEnv:
    """A GUI-like form-filling MDP with SoM-style discrete actions.

    Actions ``0..6`` are ``click(eid)``; actions ``7..27`` are
    ``type(eid, token)`` with ``eid = (a - 7) // 3`` and token ``(a-7) % 3``.
    ``click`` on a checkbox toggles it; ``type`` on a text field sets its
    value; anything else is a harmless no-op. A correct submit (or running
    out of steps) ends the episode; a wrong submit returns a form error and
    the episode continues, so the agent can retry.
    """

    def __init__(self, reward_mode: str = "sparse", level: int = 3) -> None:
        assert reward_mode in ("sparse", "shaped", "shaped_hackable")
        assert 0 <= level < N_LEVELS
        self.reward_mode = reward_mode
        self.level = level
        self.elements: list[Element] = []
        self.task = Task(level=level)
        self.field_values: dict[str, str | None] = {}
        self.box_states: dict[str, bool] = {}
        self.steps_left = MAX_STEPS
        self.best_satisfied = 0
        self.t = 0

    # -- episode control ----------------------------------------------------
    def reset(self, seed: int = 0) -> np.ndarray:
        self.elements = make_layout(seed)
        self.task = make_task(self.elements, self.level, seed)
        self.field_values = {e.label: None for e in self.elements
                             if e.kind == "field"}
        self.box_states = {e.label: False for e in self.elements
                           if e.kind == "box"}
        self.steps_left = MAX_STEPS
        self.best_satisfied = count_satisfied(self.field_values,
                                              self.box_states, self.task)
        self.t = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        return encode_obs(self.elements, self.field_values, self.box_states,
                          self.task, self.steps_left)

    # -- transitions ----------------------------------------------------------
    def step(self, action: int):
        """Apply one action. Returns (obs, reward, done, info)."""
        assert 0 <= action < N_ACTIONS
        self.t += 1
        self.steps_left = max(0, self.steps_left - 1)
        reward = 0.0 if self.reward_mode == "sparse" else -STEP_PENALTY
        done, success, submitted = False, False, False

        if action < N_CLICK_ACTIONS:
            target = next(e for e in self.elements if e.eid == action)
            if target.kind == "box":
                self.box_states[target.label] = not self.box_states[target.label]
            elif target.kind == "submit":
                submitted = True
                success = check_success(self.field_values, self.box_states,
                                        self.task)
                if success:
                    # The only terminal reward: a correct submission ends the
                    # episode with +1 (Chapter 22: sparse terminal verifier).
                    done = True
                    reward += SUBMIT_REWARD
                else:
                    # A wrong submit shows a form error and the episode goes
                    # on, so the agent can use the feedback to recover.
                    if self.reward_mode == "shaped_hackable":
                        # Deliberate reward flaw: every *failed* submit click
                        # pays +0.4 (a correct submit still pays only the +1
                        # terminal reward), so spamming submit without doing
                        # the work earns more than solving the task. See the
                        # docs page ("what to look for") for the fallout.
                        reward += SUBMIT_BONUS
            # clicks on fields/distractors are no-ops
        else:
            eid, tok = (action - N_CLICK_ACTIONS) // N_TOKENS, \
                (action - N_CLICK_ACTIONS) % N_TOKENS
            target = next(e for e in self.elements if e.eid == eid)
            if target.kind == "field":
                self.field_values[target.label] = VOCAB[tok]
            # typing anywhere else is a no-op

        if self.reward_mode in ("shaped", "shaped_hackable") and not done:
            sat = count_satisfied(self.field_values, self.box_states, self.task)
            if sat > self.best_satisfied:
                # Progress reward: paid only when the episode reaches a new
                # best sub-goal count, so toggling a box on/off cannot farm it.
                reward += PROGRESS_REWARD * (sat - self.best_satisfied)
                self.best_satisfied = sat

        if self.steps_left == 0 and not done:
            done = True  # timeout: reward stays as accumulated above

        info = {"success": success, "submitted": submitted,
                "early_submit": submitted and self.t <= 2}
        return self._obs(), float(reward), done, info

    # -- text rendering ---------------------------------------------------------
    def render_text(self) -> str:
        """ASCII rendering of the screen (for debugging and the docs page)."""
        grid = [["." for _ in range(COLS)] for _ in range(ROWS)]
        glyph = {"field": "F", "box": "B", "submit": "S", "distract": "d"}
        for el in self.elements:
            grid[int(el.y)][int(el.x)] = f"{glyph[el.kind]}{el.eid}"
        lines = [" ".join(f"{c:>3}" for c in row) for row in grid]
        lines.append("task: " + describe_task(self.task))
        return "\n".join(lines)


def describe_task(task: Task) -> str:
    """One-line English description of the task (mirrors the lab prompts)."""
    parts = [f"set {lab!r} to {tok!r}" for lab, tok in task.field_targets]
    parts += [f"{'check' if w else 'uncheck'} {lab!r}"
              for lab, w in task.box_targets]
    parts.append("then submit")
    return ("fill the form: " if parts else "submit: ") + ", ".join(parts)


def action_name(action: int) -> str:
    """Human-readable name of a flat action index (for logs and tests)."""
    if action < N_CLICK_ACTIONS:
        return f"click({action})"
    eid = (action - N_CLICK_ACTIONS) // N_TOKENS
    tok = VOCAB[(action - N_CLICK_ACTIONS) % N_TOKENS]
    return f"type({eid}, {tok!r})"
