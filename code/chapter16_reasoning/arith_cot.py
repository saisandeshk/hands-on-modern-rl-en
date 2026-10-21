"""Tiny character-level reasoner for the Chapter 16 test-time-compute lab.

The task is two-digit addition written as a short chain of thought that makes
every carry explicit. For example the problem ``23+58=`` must be completed as::

    3+8=11>1c1;2+5+1=8>8c0;=81

Reading the trace left to right: add the units (3+8=11), write down 1 and
carry 1; add the tens plus the carry (2+5+1=8), write down 8 and carry 0; the
final answer follows ``;=``. The ``>`` separates the raw column sum from the
digit that is written down, and ``c`` records the outgoing carry. This is the
lab's analogue of a chain of thought: a fixed, checkable scratchpad whose last
field is the answer the verifier reads.

Everything here is implemented from scratch with torch/numpy (no RL or LLM
libraries) so the lab runs offline on a laptop CPU in minutes.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Vocabulary and trace format
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
SPECIAL_CHARS = ["+", "=", ">", "c", ";"]
VOCAB = [PAD_TOKEN, EOS_TOKEN] + [str(d) for d in range(10)] + SPECIAL_CHARS
PAD_ID = 0
EOS_ID = 1
CHAR_TO_ID = {ch: i for i, ch in enumerate(VOCAB)}


def encode(text: str) -> list[int]:
    """Map each character to its vocabulary id."""
    return [CHAR_TO_ID[ch] for ch in text]


def decode(ids: list[int] | torch.Tensor) -> str:
    """Map ids back to characters, stopping at the first end-of-trace.

    Padding is skipped and decoding stops at ``<eos>``: the model must
    learn *when to stop*, and anything generated after the stop token is
    babble that the answer parser never sees.
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    chars = []
    for i in ids:
        if i == EOS_ID:
            break
        if i == PAD_ID:
            continue
        chars.append(VOCAB[i])
    return "".join(chars)


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------

LOW, HIGH = 10, 99  # two-digit operands


def render(a: int, b: int) -> tuple[str, str, int]:
    """Render one problem as (prompt, completion, answer).

    The completion spells out each column addition least-significant digit
    first, recording the digit written down and the outgoing carry, so the
    answer can be re-derived from the trace alone.
    """
    answer = a + b
    a0, a1 = a % 10, a // 10
    b0, b1 = b % 10, b // 10
    s0 = a0 + b0
    d0, c1 = s0 % 10, s0 // 10
    s1 = a1 + b1 + c1
    d1, c2 = s1 % 10, s1 // 10
    prompt = f"{a}+{b}="
    completion = f"{a0}+{b0}={s0}>{d0}c{c1};{a1}+{b1}+{c1}={s1}>{d1}c{c2};={answer}"
    return prompt, completion, answer


def make_splits(n_train: int, n_test: int, seed: int) -> tuple[list, list]:
    """Sample disjoint train/test operand pairs without replacement."""
    rng = random.Random(seed)
    pairs = [(a, b) for a in range(LOW, HIGH + 1) for b in range(LOW, HIGH + 1)]
    rng.shuffle(pairs)
    assert n_train + n_test <= len(pairs), "not enough distinct pairs"
    return pairs[:n_train], pairs[n_train : n_train + n_test]


def parse_answer(text: str) -> int | None:
    """Extract the integer after the final ``;=`` marker.

    Returns None when the trace has no parseable answer (missing marker,
    empty completion, non-numeric field). Parsing and verifying are kept
    separate on purpose: the parser reads *what* the model claimed, the
    verifier (§16.2's ``v(x, y)``) decides whether the claim is right.
    """
    import re

    match = re.search(r";=\s*(\d+)", text)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, numpy and torch RNGs for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Tiny character-level transformer
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Single multi-head causal self-attention layer."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, c // self.n_head)
        q, k, v = qkv.unbind(dim=2)  # each [b, t, h, dh]
        q, k, v = (m.transpose(1, 2) for m in (q, k, v))  # [b, h, t, dh]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class Block(nn.Module):
    """Pre-LN transformer block: attention sublayer + MLP sublayer."""

    def __init__(self, d_model: int, n_head: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyReasoner(nn.Module):
    """Small decoder-only transformer over the character vocabulary.

    The prompt is fed as a prefix and the model continues it with the
    carry trace; training uses teacher forcing with the prompt masked out
    of the loss.
    """

    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        d_model: int = 64,
        n_layer: int = 2,
        n_head: int = 4,
        d_ff: int = 128,
        max_len: int = 64,
    ):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_ff) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for every position of ``idx``."""
        _, t = idx.shape
        assert t <= self.max_len, f"sequence length {t} exceeds max_len"
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(
        self,
        prompts: torch.Tensor,
        prompt_lens: torch.Tensor,
        max_new: int,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Complete each prompt, stopping per row at ``<eos>`` (batched).

        ``temperature=0`` is greedy decoding (argmax); otherwise tokens are
        sampled from ``softmax(logits / temperature)``. Generation stops a
        row once it emits ``<eos>`` (later slots are left as padding) or
        when ``max_new`` tokens are reached, since every valid trace has a
        bounded length. The answer parser reads the ``;=`` field of the
        decoded prefix and ignores anything after the stop token.
        """
        self.eval()
        out = prompts.clone()
        finished = torch.zeros(
            prompts.shape[0], dtype=torch.bool, device=prompts.device
        )
        for _ in range(max_new):
            if bool(finished.all()):
                break
            logits = self.forward(out[:, -self.max_len :])[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(
                    probs, num_samples=1, generator=generator
                )
            nxt[finished] = PAD_ID
            out = torch.cat([out, nxt], dim=1)
            fresh = (~finished) & (nxt[:, 0] == EOS_ID)
            finished = finished | fresh
        # Never rewrite the prompt region (guard against any in-place slip).
        assert out.shape[1] <= prompts.shape[1] + max_new
        return out


@dataclass
class TrainExample:
    prompt_ids: list[int]
    completion_ids: list[int]
    answer: int


def build_examples(pairs: list[tuple[int, int]]) -> list[TrainExample]:
    """Render operand pairs into training examples (traces end with <eos>)."""
    examples = []
    for a, b in pairs:
        prompt, completion, answer = render(a, b)
        examples.append(
            TrainExample(
                prompt_ids=encode(prompt),
                completion_ids=encode(completion) + [EOS_ID],
                answer=answer,
            )
        )
    return examples


def train_reasoner(
    model: TinyReasoner,
    examples: list[TrainExample],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    log_every: int | None = None,
) -> list[float]:
    """Teacher-forced training; loss applies to completion tokens only.

    Returns the mean loss per epoch. Prompt positions and padding carry
    label -100 so they never contribute to the gradient.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    max_comp = max(len(e.completion_ids) for e in examples)
    losses: list[float] = []
    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        total, count = 0.0, 0
        for start in range(0, len(order), batch_size):
            chunk = [examples[i] for i in order[start : start + batch_size]]
            plen = max(len(e.prompt_ids) for e in chunk)
            seqs, labels = [], []
            for e in chunk:
                pad_p = [PAD_ID] * (plen - len(e.prompt_ids))
                pad_c = [PAD_ID] * (max_comp - len(e.completion_ids))
                seq = pad_p + e.prompt_ids + e.completion_ids + pad_c
                lab = (
                    [-100] * plen
                    + e.completion_ids
                    + [-100] * (max_comp - len(e.completion_ids))
                )
                seqs.append(seq)
                labels.append(lab)
            seq_t = torch.tensor(seqs, dtype=torch.long, device=device)
            lab_t = torch.tensor(labels, dtype=torch.long, device=device)
            logits = model(seq_t)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, len(VOCAB)),
                lab_t[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(chunk)
            count += len(chunk)
        losses.append(total / count)
        if log_every is not None and (epoch + 1) % log_every == 0:
            print(f"    epoch {epoch + 1}/{epochs} loss={losses[-1]:.4f}", flush=True)
    return losses


# ---------------------------------------------------------------------------
# Test-time-compute metrics (§16.3)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al.): 1 - C(n-c, k) / C(n, k).

    Given ``n`` samples of which ``c`` are correct, this estimates the
    probability that at least one of ``k`` draws is correct. Edge cases:
    ``c == 0`` gives 0, and ``k`` beyond ``n`` uses all ``n`` samples.
    """
    if c <= 0:
        return 0.0
    if k >= n:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def majority_answer(answers: list[int | None]) -> int | None:
    """Self-consistency vote: most common parsed answer.

    Unparseable samples (None) abstain. Ties are broken deterministically
    by the smallest numeric answer so repeated runs agree exactly.
    Returns None when every sample is unparseable.
    """
    votes = [a for a in answers if a is not None]
    if not votes:
        return None
    counts = Counter(votes)
    top = max(counts.values())
    return min(a for a, n in counts.items() if n == top)


def coverage_at_n(flags: list[bool], n: int) -> float:
    """Fraction indicator: is at least one of the first ``n`` samples correct?

    With a perfectly reliable verifier, best-of-N achieves exactly this
    coverage: the verifier accepts a correct trace whenever the pool
    contains one.
    """
    return 1.0 if any(flags[:n]) else 0.0


def best_of_n_noisy(flags: list[bool], accepted: list[bool], n: int) -> bool:
    """Best-of-N under a fallible verifier.

    ``accepted[i]`` says whether the verifier *claims* sample ``i`` is
    correct (true positives always accepted; wrong samples accepted with
    the verifier's false-positive rate). The rule picks the first accepted
    sample among the first ``n`` and falls back to sample 0 when the
    verifier rejects everything. Returns whether the pick is truly correct.
    """
    for i in range(min(n, len(flags))):
        if accepted[i]:
            return flags[i]
    return flags[0] if flags else False
