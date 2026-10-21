"""
Chapter 15: GRPO Math Reasoning — RLVR on GSM8K-Style Problems
===============================================================

This script trains a small language model (Qwen2.5-0.5B-Instruct) with the
GRPO (Group Relative Policy Optimization) objective from DeepSeekMath.

Full GRPO pipeline per prompt:
  1. Sample a group of G completions from the old (sampling) policy.
  2. Score each completion with a verifiable, rule-based reward (answer checker).
  3. Group-normalize rewards into advantages:
         A_hat_i = (r_i - mean(r)) / (std(r) + eps),
     broadcast to every completion token of response i.
  4. Per-token importance ratio against the old policy with PPO-style clipping:
         ratio = exp(log pi_theta - log pi_old),
         clipped = clamp(ratio, 1 - eps_clip, 1 + eps_clip).
  5. KL penalty to a frozen reference policy with the k3 estimator:
         KL_k3 = exp(log pi_ref - log pi_theta) - (log pi_ref - log pi_theta) - 1.
  6. Completion-only masks: prompt tokens and padding are excluded from the loss.
  7. Documented loss aggregation: sequence-mean (default) vs token-mean (see
     LOSS_AGGREGATION below).

How to run (full mode needs a GPU; ~2-4 GB VRAM in fp16):
  pip install -r requirements.txt
  python grpo_math_reasoning.py
  python grpo_math_reasoning.py --group-size 8 --epochs 1   # quick run
  python grpo_math_reasoning.py --smoke                     # no network: tiny random model

Model download: the default model "Qwen/Qwen2.5-0.5B-Instruct" (~1 GB) is
downloaded automatically from Hugging Face on first use and cached. To use less
VRAM, pass --lora (requires the `peft` package) or pick a smaller model with
--model. Smoke mode builds a tiny random GPT-2 locally and downloads nothing.
"""

import argparse
import copy
import re

import numpy as np
import torch
import torch.nn.functional as F

# ==========================================
# Loss aggregation choice (documented)
# ==========================================
# "sequence-mean" (default): average the per-token loss within each completion
#   first (so long and short completions count equally), then average over the
#   group. This avoids biasing the update toward long completions.
# "token-mean": pool all completion tokens across the group and take one mean.
#   This gives long completions proportionally more weight.
# Pass --aggregation token-mean to switch. Unit tests cover both.
LOSS_AGGREGATION = "sequence-mean"

EPS = 1e-8


# ==========================================
# Part 1: GSM8K-style math dataset (English)
# ==========================================
# 20 elementary-school arithmetic word problems covering +, -, *, /
# Each problem contains a question and a standard answer

math_dataset = [
    {
        "question": "Ming has 15 apples. Hong gives him 27 more. How many apples does Ming have now?",
        "answer": "42",
    },
    {
        "question": "A class has 36 students split evenly into 6 groups. How many students are in each group?",
        "answer": "6",
    },
    {
        "question": "Zhang runs 5 km every day. How many km does he run in a week (7 days)?",
        "answer": "35",
    },
    {
        "question": "A book has 240 pages. Li has read 85 pages. How many pages are left unread?",
        "answer": "155",
    },
    {
        "question": "A fruit shop has 48 apples, 36 oranges, and 25 bananas. How many fruits are there in total?",
        "answer": "109",
    },
    {
        "question": "A crate holds 24 bottles. How many bottles do 8 crates hold?",
        "answer": "192",
    },
    {
        "question": "Hong has 100 dollars. She spends 32 on a book and 8 on a pen. How much money is left?",
        "answer": "60",
    },
    {
        "question": "A rectangle is 12 cm long and 8 cm wide. What is its area in square cm?",
        "answer": "96",
    },
    {
        "question": "A school buys 5 boxes of pencils with 12 pencils per box. How many pencils is that?",
        "answer": "60",
    },
    {
        "question": "A worker makes 15 parts per hour and works 8 hours. How many parts does he make?",
        "answer": "120",
    },
    {
        "question": "A train travels 120 km per hour for 3 hours. How far does it travel?",
        "answer": "360",
    },
    {
        "question": "A farm has 45 chickens. There are 18 more ducks than chickens. How many ducks are there?",
        "answer": "63",
    },
    {
        "question": "A triangle has base 10 cm and height 6 cm. What is its area in square cm?",
        "answer": "30",
    },
    {
        "question": "A mother buys 3 kg of apples at $8 per kg and 2 kg of oranges at $5 per kg. How much does she spend?",
        "answer": "34",
    },
    {
        "question": "A library has 280 story books. There are 95 fewer science books than story books. How many science books are there?",
        "answer": "185",
    },
    {
        "question": "Ming walks 15 minutes to school at 60 meters per minute. How far is school from home?",
        "answer": "900",
    },
    {
        "question": "A swimming pool is 50 m long and 25 m wide. What is its perimeter in meters?",
        "answer": "150",
    },
    {
        "question": "Four children share 72 candies equally. How many does each child get?",
        "answer": "18",
    },
    {
        "question": "A factory makes 500 parts, of which 468 pass inspection. How many fail?",
        "answer": "32",
    },
    {
        "question": "A bucket holds 18 kg of oil. One third is used. How many kg remain?",
        "answer": "12",
    },
]


# ==========================================
# Part 2: Rule-based reward function
# ==========================================
def extract_answer_from_response(response):
    """
    Extract the final answer from the model's response.

    Priority order:
      1. \\boxed{...} format
      2. An English answer marker ("The answer is ..." / "Answer: ...")
      3. The last number in the response (fallback)

    Args:
        response: text generated by the model
    Returns:
        str or None: the extracted answer string
    """
    # Method 1: \boxed{} format
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Method 2: English answer marker
    en_match = re.search(
        r'(?:[Tt]he answer is|[Aa]nswer\s*:|[Ff]inal answer is)\s*([+-]?\d+\.?\d*)',
        response,
    )
    if en_match:
        return en_match.group(1).strip()

    # Method 3: last number in the response (fallback)
    numbers = re.findall(r'([+-]?\d+\.?\d*)', response)
    if numbers:
        return numbers[-1]

    return None


def compute_reward(response, ground_truth):
    """
    Compute the reward value for a response.

    A simplified verifiable reward focused on answer correctness:
      - Correct answer: reward = 1.0
      - Incorrect answer: reward = 0.0
      - Format bonus: +0.1 if step-by-step reasoning markers are present
        (only when the answer is correct, to avoid rewarding
        well-formatted wrong answers)

    The final reward is capped at 1.0.

    Args:
        response: text generated by the model
        ground_truth: the standard answer string
    Returns:
        float: reward value in [0.0, 1.0]
    """
    reward = 0.0

    # Extract the answer and check correctness
    extracted = extract_answer_from_response(response)
    if extracted is not None:
        try:
            if abs(float(extracted) - float(ground_truth)) < 1e-6:
                reward = 1.0
        except (ValueError, TypeError):
            if str(extracted).strip() == str(ground_truth).strip():
                reward = 1.0

    # Format bonus (only when the answer is correct)
    if reward > 0:
        has_steps = bool(
            re.search(
                r'Step\s*\d|step\s*\d|First|Then|Next|Finally|Therefore|So,',
                response,
            )
        )
        if has_steps:
            reward = min(1.0, reward + 0.1)

    return reward


# ==========================================
# Part 3: GRPO core math (pure functions, unit-tested)
# ==========================================
def compute_group_advantages(rewards, eps=EPS):
    """
    GRPO group-relative normalization.

    Formula:
        A_hat_i = (r_i - mean(r)) / (std(r) + eps)

    The population std (unbiased=False) matches the original GRPO definition.
    When all rewards in the group are equal, std is 0 and every advantage is 0
    (no learning signal — nothing to prefer within the group).

    Args:
        rewards: Tensor of shape (G,) with one scalar reward per completion
        eps: small constant guarding against division by zero
    Returns:
        Tensor of shape (G,) with the group-normalized advantages
    """
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False)
    return (rewards - mean_r) / (std_r + eps)


def compute_grpo_advantages(rewards):
    """
    Numpy-compat wrapper around compute_group_advantages (kept for the
    mechanism demo style: pass a list of floats, get a numpy array back).
    """
    adv = compute_group_advantages(torch.as_tensor(list(rewards), dtype=torch.float32))
    return adv.detach().cpu().numpy()


def k3_kl_estimate(new_logp, ref_logp):
    """
    The k3 KL estimator (Schulman 2020), always non-negative:

        KL_k3 = exp(d) - d - 1,  where d = log pi_ref - log pi_theta.

    It is zero exactly when the policies match (d = 0) and penalizes large
    deviations more steeply than the naive k1 estimator (d itself), while
    remaining unbiased in expectation under pi_theta.

    Args:
        new_logp: per-token log-probs under the current policy
        ref_logp: per-token log-probs under the frozen reference policy
    Returns:
        per-token KL estimate, same shape as the inputs
    """
    diff = ref_logp - new_logp
    return torch.exp(diff) - diff - 1.0


def build_completion_mask(prompt_lens, completion_ids, pad_id=None):
    """
    Build a completion-only mask: 1 for completion tokens, 0 for prompt and
    padding tokens.

    Args:
        prompt_lens: list/array of prompt lengths, one per sequence (batch G)
        completion_ids: LongTensor (G, L) of completion token ids
        pad_id: optional padding id; positions holding it are masked out
    Returns:
        FloatTensor (G, L) mask
    """
    g, seq_len = completion_ids.shape
    positions = torch.arange(seq_len).unsqueeze(0).expand(g, seq_len)
    prompt_lens_t = torch.as_tensor(list(prompt_lens)).unsqueeze(1)
    mask = (positions >= prompt_lens_t).float()
    if pad_id is not None:
        mask = mask * (completion_ids != pad_id).float()
    return mask


def grpo_loss(
    new_logp,
    old_logp,
    ref_logp,
    advantages,
    mask,
    clip_eps=0.2,
    kl_coef=0.01,
    aggregation=LOSS_AGGREGATION,
):
    """
    Full GRPO per-token loss with PPO-style clipping and a k3 KL penalty.

    Per-token terms (A_i is the group advantage of completion i, broadcast to
    all its completion tokens):
        ratio   = exp(new_logp - old_logp)
        policy  = -min(ratio * A_i, clamp(ratio, 1-eps, 1+eps) * A_i)
        kl      = k3(new_logp, ref_logp)
        token_loss = policy + kl_coef * kl

    For positive A_i the min() keeps the update from growing the ratio past
    1+eps; for negative A_i it keeps the update from shrinking the ratio below
    1-eps — the standard PPO trust region in both directions.

    Aggregation (documented choice):
      - "sequence-mean": mean within each completion first (long and short
        completions count equally), then mean over the group. Default.
      - "token-mean": one mean over all unmasked completion tokens (long
        completions weigh more).

    All inputs except advantages/mask are per-token log-probs of shape (G, L).
    advantages has shape (G,) and is broadcast across tokens. mask has shape
    (G, L); masked positions contribute nothing.

    Returns:
        scalar loss tensor
    """
    ratio = torch.exp(new_logp - old_logp)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    adv = advantages.unsqueeze(1)
    policy_loss = -torch.min(ratio * adv, clipped_ratio * adv)
    kl = k3_kl_estimate(new_logp, ref_logp)
    token_loss = (policy_loss + kl_coef * kl) * mask

    if aggregation == "token-mean":
        denom = mask.sum().clamp_min(1.0)
        return token_loss.sum() / denom
    if aggregation == "sequence-mean":
        per_seq = token_loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return per_seq.mean()
    raise ValueError(f"unknown aggregation: {aggregation!r}")


# ==========================================
# Part 4: Tokenizer/model helpers (incl. smoke mode)
# ==========================================
class CharTokenizer:
    """Minimal character-level tokenizer for --smoke mode (no downloads).

    Maps each distinct character to an id; reserves 0 for padding and 1 for EOS.
    Implements just enough of the HF tokenizer API for this script.
    """

    def __init__(self, texts=None):
        chars = set()
        for t in texts or []:
            chars.update(t)
        # Reserve 0 = pad, 1 = eos/bos.
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self._stoi = {"<pad>": 0, "<eos>": 1}
        for i, ch in enumerate(sorted(chars), start=2):
            self._stoi[ch] = i
        self._itos = {i: ch for ch, i in self._stoi.items()}

    @property
    def vocab_size(self):
        return len(self._stoi)

    def encode(self, text, return_tensors=None):
        ids = [self._stoi.get(ch, 1) for ch in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        out = []
        for i in ids:
            if skip_special_tokens and i in (self.pad_token_id, self.eos_token_id):
                continue
            out.append(self._itos.get(i, "?"))
        return "".join(out)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = ""
        for m in messages:
            text += f"{m['role']}: {m['content']}\n"
        if add_generation_prompt:
            text += "assistant: "
        return text

    def __call__(self, texts, return_tensors="pt", padding=False, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        ids = [self.encode(t) for t in texts]
        max_len = max(len(s) for s in ids)
        padded = [s + [self.pad_token_id] * (max_len - len(s)) for s in ids]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1] * len(s) + [0] * (max_len - len(s)) for s in ids],
                dtype=torch.long,
            ),
        }


def build_smoke_model_and_tokenizer(device="cpu"):
    """Build a tiny random GPT-2 plus a char tokenizer (no network needed)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    texts = [p["question"] for p in math_dataset[:4]]
    tokenizer = CharTokenizer(texts)
    config = GPT2Config(
        vocab_size=max(tokenizer.vocab_size, 32),
        n_positions=128,
        n_ctx=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config).to(device)
    return model, tokenizer


def _model_device(model):
    return getattr(model, "device", next(model.parameters()).device)


def _token_log_probs(model, input_ids):
    """Per-token log-probs of each position predicting the next token.

    Returns Tensor (batch, seq_len - 1): log pi(input_ids[t+1] | prefix .. t).
    """
    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    labels = input_ids[:, 1:]
    return log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)


# ==========================================
# Part 5: Generate model responses
# ==========================================
def generate_responses(model, tokenizer, prompt, num_responses=4, max_new_tokens=200):
    """
    Have the model generate several different responses to the same question.

    Uses do_sample=True + temperature > 0 to produce diversity.

    Args:
        model: the language model
        tokenizer: the tokenizer
        prompt: the question text
        num_responses: number of responses to generate (group_size)
        max_new_tokens: max number of tokens to generate
    Returns:
        list[str]: list of generated responses
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer([text], return_tensors="pt")
    device = _model_device(model)
    inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()
    }
    input_ids = inputs["input_ids"]

    responses = []
    for _ in range(num_responses):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=getattr(tokenizer, "pad_token_id", None)
                or getattr(tokenizer, "eos_token_id", None),
            )
        # Decode: keep only the generated part (skip the input prompt)
        response = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True,
        )
        responses.append(response)

    return responses


# ==========================================
# Part 6: Full GRPO policy update
# ==========================================
def grpo_policy_update(
    model,
    ref_model,
    tokenizer,
    optimizer,
    prompt,
    responses,
    advantages,
    clip_eps=0.2,
    kl_coef=0.01,
    aggregation=LOSS_AGGREGATION,
    update_epochs=1,
):
    """
    Update the model with one GRPO step over a group of completions.

    For each completion, the old (sampling) policy log-probs and the frozen
    reference log-probs are computed under no_grad; then the current policy is
    optimized with grpo_loss() (ratio + clipping + k3 KL, completion-masked).

    Args:
        model: the policy being trained
        ref_model: frozen reference policy (may be None to skip the KL term)
        tokenizer: the tokenizer
        optimizer: optimizer
        prompt: the original question
        responses: list of sampled completion strings (group of size G)
        advantages: group-normalized advantages, shape (G,)
        clip_eps: PPO clip range
        kl_coef: weight of the k3 KL penalty
        aggregation: "sequence-mean" (default) or "token-mean"
        update_epochs: PPO-style epochs over the same sampled group
    Returns:
        float: the final loss value
    """
    model.train()
    device = _model_device(model)

    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[-1]

    # Tokenize prompt + each completion; pad completions to a batch.
    comp_id_list = []
    for resp in responses:
        ids = tokenizer.encode(resp, return_tensors="pt").to(device)
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is not None and ids.shape[-1] > 0 and ids[0, 0] == bos:
            ids = ids[:, 1:]
        comp_id_list.append(ids[0])
    max_len = max(t.shape[0] for t in comp_id_list)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    full_ids = []
    for t in comp_id_list:
        pad_needed = max_len - t.shape[0]
        comp_padded = (
            torch.cat([t, torch.full((pad_needed,), pad_id, dtype=torch.long, device=device)])
            if pad_needed > 0
            else t
        )
        full_ids.append(torch.cat([prompt_ids[0], comp_padded]))
    batch_ids = torch.stack(full_ids)

    # Completion mask: exclude prompt tokens (and padding).
    batch_size, total_len = batch_ids.shape
    positions = torch.arange(total_len, device=device).unsqueeze(0).expand(batch_size, total_len)
    mask_full = (positions >= prompt_len).float()
    mask_full = mask_full * (batch_ids != pad_id).float()
    # Shift for next-token prediction: token t predicts t+1, so drop column 0.
    token_mask = mask_full[:, 1:]
    if token_mask.sum() == 0:
        return 0.0

    with torch.no_grad():
        old_logp_full = _token_log_probs(model, batch_ids)
        if ref_model is not None:
            ref_logp_full = _token_log_probs(ref_model, batch_ids)
        else:
            ref_logp_full = old_logp_full.detach()
    old_logp = old_logp_full * 0 + old_logp_full.detach()  # frozen sampling log-probs
    advantages_t = torch.as_tensor(list(advantages), dtype=torch.float32, device=device)

    final_loss = 0.0
    for _ in range(update_epochs):
        new_logp_full = _token_log_probs(model, batch_ids)
        loss = grpo_loss(
            new_logp_full,
            old_logp,
            ref_logp_full,
            advantages_t,
            token_mask,
            clip_eps=clip_eps,
            kl_coef=kl_coef,
            aggregation=aggregation,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        final_loss = loss.item()

    return final_loss


# ==========================================
# Part 7: Full training pipeline
# ==========================================
def train(args):
    """
    Full GRPO math-reasoning training pipeline.
    """
    model_name = args.model
    group_size = args.group_size
    num_epochs = args.epochs
    learning_rate = args.lr
    max_new_tokens = args.max_new_tokens

    print("=" * 70)
    print("  GRPO Math Reasoning Training (GSM8K style)")
    print("=" * 70)
    print(f"  Model: {model_name}")
    print(f"  Training data: {len(math_dataset)} math problems")
    print(f"  Sample count (group_size): {group_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max generation length: {max_new_tokens}")
    print(f"  Clip eps: {args.clip_eps} | KL coef: {args.kl_coef} | Aggregation: {args.aggregation}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if args.smoke:
        # Smoke mode always runs on CPU (shared-GPU friendly) with a tiny model.
        device = "cpu"
        print("Smoke mode: building a tiny local model (no download).")
        model, tokenizer = build_smoke_model_and_tokenizer(device=device)
        ref_model = None
        dataset = math_dataset[:2]
        num_epochs = 1
        group_size = min(group_size, 2)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        use_lora = args.lora
        if use_lora:
            from peft import LoraConfig, get_peft_model

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        if use_lora:
            lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                  task_type="CAUSAL_LM")
            model = get_peft_model(model, lora_cfg)
            print("  LoRA enabled (r=8): only adapter weights are trained.")
        # Frozen reference copy for the KL term.
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        dataset = math_dataset

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ---------- Pre-training test ----------
    if not args.smoke:
        print("\n" + "=" * 70)
        print("  Model performance before training (test on 3 sampled problems)")
        print("=" * 70)
        test_indices = [0, 7, 14]
        pre_train_results = evaluate_on_samples(model, tokenizer, math_dataset, test_indices)
    else:
        pre_train_results, test_indices = [], []

    # ---------- Training loop ----------
    print("\n" + "=" * 70)
    print("  Starting GRPO + RLVR training")
    print("=" * 70)

    epoch_metrics = []

    for epoch in range(num_epochs):
        print(f"\n{'#' * 70}")
        print(f"  Epoch {epoch + 1}/{num_epochs}")
        print(f"{'#' * 70}")

        epoch_total_rewards = []
        epoch_accuracies = []
        epoch_lengths = []
        epoch_losses = []

        for idx, problem in enumerate(dataset):
            question = problem["question"]
            ground_truth = problem["answer"]

            # Step 1: sample a group of completions from the current policy
            responses = generate_responses(
                model, tokenizer, question,
                num_responses=group_size,
                max_new_tokens=max_new_tokens,
            )

            # Step 2: verifiable reward for each completion
            rewards = [compute_reward(resp, ground_truth) for resp in responses]

            # Step 3: group-relative normalization -> advantages
            advantages = compute_group_advantages(
                torch.tensor(rewards, dtype=torch.float32)
            )

            # Step 4: GRPO update (ratio + clipping + reference KL)
            loss = grpo_policy_update(
                model, ref_model, tokenizer, optimizer,
                question, responses, advantages,
                clip_eps=args.clip_eps, kl_coef=args.kl_coef,
                aggregation=args.aggregation,
                update_epochs=args.update_epochs,
            )

            # Record metrics
            avg_reward = float(np.mean(rewards))
            accuracy = float(np.mean([1.0 if r >= 1.0 else 0.0 for r in rewards]))
            avg_length = float(np.mean([len(r) for r in responses]))

            epoch_total_rewards.append(avg_reward)
            epoch_accuracies.append(accuracy)
            epoch_lengths.append(avg_length)
            epoch_losses.append(loss)

            if (idx + 1) % 5 == 0:
                recent_acc = float(np.mean(epoch_accuracies[-5:]))
                recent_reward = float(np.mean(epoch_total_rewards[-5:]))
                print(
                    f"  Problem {idx + 1:2d}/{len(dataset)} | "
                    f"last-5 accuracy: {recent_acc:.1%} | "
                    f"avg reward: {recent_reward:.3f} | "
                    f"loss: {loss:.4f}"
                )

        epoch_metrics.append({
            "epoch": epoch + 1,
            "avg_reward": float(np.mean(epoch_total_rewards)),
            "accuracy": float(np.mean(epoch_accuracies)),
            "avg_length": float(np.mean(epoch_lengths)),
            "avg_loss": float(np.mean([l for l in epoch_losses if l != 0.0] or [0.0])),
        })

        print(f"\n  Epoch {epoch + 1} summary:")
        print(f"    Avg reward:   {epoch_metrics[-1]['avg_reward']:.4f}")
        print(f"    Avg accuracy: {epoch_metrics[-1]['accuracy']:.2%}")
        print(f"    Avg response length: {epoch_metrics[-1]['avg_length']:.0f} chars")
        print(f"    Avg loss:   {epoch_metrics[-1]['avg_loss']:.4f}")

    if args.smoke:
        print("\nSmoke run complete (no download, tiny local model).")
        return

    # ---------- Post-training test ----------
    print("\n" + "=" * 70)
    print("  Model performance after training (same 3 test problems)")
    print("=" * 70)
    post_train_results = evaluate_on_samples(model, tokenizer, math_dataset, test_indices)

    # ---------- Before/after comparison ----------
    print("\n" + "=" * 70)
    print("  Before/after training comparison")
    print("=" * 70)
    for i, test_idx in enumerate(test_indices):
        problem = math_dataset[test_idx]
        pre = pre_train_results[i]
        post = post_train_results[i]
        print(f"\n  Problem: {problem['question'][:40]}...")
        print(f"  Ground truth: {problem['answer']}")
        print(f"  Before training: {pre['response'][:60]}... -> reward: {pre['reward']:.2f}")
        print(f"  After training: {post['response'][:60]}... -> reward: {post['reward']:.2f}")

    # ---------- Final summary ----------
    print("\n" + "=" * 70)
    print("  GRPO + RLVR Training Summary")
    print("=" * 70)
    print(f"""
  This experiment ran the full GRPO objective (ratio + clipping + reference KL).

  1. Data preparation:
     - {len(math_dataset)} GSM8K-style arithmetic word problems
     - Each problem has a clear standard answer, enabling rule-based verification

  2. GRPO training:
     - {group_size} responses sampled per problem
     - A rule-based reward function verifies answer correctness
     - Group-relative normalization computes advantages, with no Critic network needed
     - PPO-style clipping keeps the new policy near the sampling policy
     - The k3 KL term keeps the policy near the frozen reference model

  3. Key observations:
     - Before training: the model's outputs may be near-random, with low accuracy
     - After training: the model gradually learns the reasoning format and correct computation
     - Average reward and accuracy should steadily improve over training

  4. Comparison with DPO:
     - DPO requires paired preference data (chosen/rejected)
     - GRPO only requires a verifiable reward signal
     - For tasks with a standard answer (math, coding), GRPO + RLVR is a more natural fit

  5. Directions for extension:
     - Increase the amount of training data (the full GSM8K dataset)
     - Use a larger model (Qwen2.5-1.5B / 7B)
     - Increase group_size (8-16) for a more stable advantage estimate
     - Use a DeepSeek-R1-style multi-stage training approach
    """)


# ==========================================
# Part 8: Evaluation function
# ==========================================
def evaluate_on_samples(model, tokenizer, dataset, indices):
    """
    Evaluate the model's performance on the given samples (greedy decoding).

    Args:
        model: the language model
        tokenizer: the tokenizer
        dataset: the dataset list
        indices: sample indices to evaluate
    Returns:
        list[dict]: evaluation results for each sample
    """
    model.eval()
    results = []
    device = _model_device(model)

    for idx in indices:
        problem = dataset[idx]
        messages = [{"role": "user", "content": problem["question"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer([text], return_tensors="pt")
        inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()
        }
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=getattr(tokenizer, "pad_token_id", None)
                or getattr(tokenizer, "eos_token_id", None),
            )

        response = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True,
        )

        reward = compute_reward(response, problem["answer"])
        extracted = extract_answer_from_response(response)

        result = {
            "question": problem["question"],
            "ground_truth": problem["answer"],
            "response": response,
            "extracted_answer": extracted,
            "reward": reward,
            "correct": str(extracted) == str(problem["answer"]) if extracted else False,
        }
        results.append(result)

        print(f"\n  Problem: {problem['question'][:50]}...")
        print(f"  Ground truth: {problem['answer']}")
        print(f"  Model response: {response[:80]}{'...' if len(response) > 80 else ''}")
        print(f"  Extracted answer: {extracted}")
        print(f"  Correct: {'yes' if result['correct'] else 'no'} | Reward: {reward:.2f}")

    return results


# ==========================================
# Entry point
# ==========================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GRPO math reasoning (DeepSeekMath-style)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HF model id (0.5B fits a 6 GB GPU in fp16)")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--aggregation", type=str, default="sequence-mean",
                        choices=["sequence-mean", "token-mean"])
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--lora", action="store_true",
                        help="Train LoRA adapters instead of full weights (needs peft)")
    parser.add_argument("--smoke", action="store_true",
                        help="Offline smoke run: tiny random model, 2 prompts, no download")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
