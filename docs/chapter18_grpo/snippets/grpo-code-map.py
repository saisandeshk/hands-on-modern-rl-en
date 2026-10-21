import re

import torch


def completion_mask_until_eos(completion_ids, eos_token_id):
    """Keep the first EOS and all tokens before it; mask the padding that follows."""
    is_eos = completion_ids.eq(eos_token_id)
    eos_seen_before = is_eos.cumsum(dim=-1) - is_eos.to(torch.long)
    return eos_seen_before.eq(0)


# [A] Group sampling: generate group_size responses per prompt and keep the original token boundaries
def sample_groups(model, tokenizer, prompts, group_size=8, max_new_tokens=256):
    expanded_prompts = [prompt for prompt in prompts for _ in range(group_size)]
    prompt_batch = tokenizer(expanded_prompts, padding=True, return_tensors="pt")
    prompt_batch = {
        key: value.to(model.device)
        for key, value in prompt_batch.items()
    }

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **prompt_batch,
            do_sample=True,
            temperature=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_width = prompt_batch["input_ids"].size(1)
    completion_ids = output_ids[:, prompt_width:]
    completion_mask = completion_mask_until_eos(
        completion_ids,
        tokenizer.eos_token_id,
    )
    attention_mask = torch.cat(
        [prompt_batch["attention_mask"], completion_mask.to(torch.long)],
        dim=1,
    )

    responses = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    group_ids = torch.arange(len(prompts), device=model.device).repeat_interleave(
        group_size
    )
    batch = {
        "input_ids": output_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
    }
    return responses, group_ids, batch


# [B] Rule-based reward: score points for a correct math answer and a well-formed format
def rule_reward(response, ground_truth):
    reward = 0.0
    boxed = re.search(r"\\boxed\{([^}]+)\}", response)

    if boxed:
        reward += 0.5
        if boxed.group(1).strip() == str(ground_truth).strip():
            reward += 1.0

    return reward


def score_responses(responses, ground_truths, group_size=8, device="cpu"):
    rewards = []
    for i, response in enumerate(responses):
        prompt_id = i // group_size
        rewards.append(rule_reward(response, ground_truths[prompt_id]))
    return torch.tensor(rewards, dtype=torch.float32, device=device)


# [C] Group-relative advantage: replace the critic baseline with the mean over responses to the same prompt
def group_advantages(rewards, group_size=8, eps=1e-8):
    grouped_rewards = rewards.view(-1, group_size)
    group_mean = grouped_rewards.mean(dim=1, keepdim=True)
    group_std = grouped_rewards.std(dim=1, keepdim=True, correction=1)

    advantages = (grouped_rewards - group_mean) / (group_std + eps)
    advantages = torch.where(
        group_std < eps,
        torch.zeros_like(advantages),
        advantages,
    )
    return advantages.reshape(-1)


# [D] Return only the per-token log probability of the response part, with shape [B, T]
def per_token_logprobs(model, input_ids, attention_mask, completion_length):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    all_token_logprobs = logits.log_softmax(dim=-1)
    picked_logprobs = all_token_logprobs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    return picked_logprobs[:, -completion_length:]


def masked_sequence_mean(values, mask):
    """Average over valid tokens within each response first, so long responses do not get more weight."""
    mask = mask.to(values.dtype)
    token_count = mask.sum(dim=-1).clamp_min(1.0)
    return (values * mask).sum(dim=-1) / token_count


# [E-F] Original GRPO: per-token ratio, clip, and KL, then normalize by response length
def grpo_objective_from_logprobs(
    new_logprobs,
    old_logprobs,
    ref_logprobs,
    completion_mask,
    advantages,
    clip_eps=0.2,
    kl_coef=0.04,
):
    token_ratio = torch.exp(new_logprobs - old_logprobs)
    token_advantages = advantages.unsqueeze(-1)
    unclipped = token_ratio * token_advantages
    clipped_ratio = torch.clamp(token_ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    clipped = clipped_ratio * token_advantages

    # DeepSeekMath Eq. (4): per-token unbiased, non-negative estimate of D_KL(policy || ref)
    log_ratio_ref = ref_logprobs - new_logprobs
    per_token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

    per_token_objective = torch.minimum(unclipped, clipped) - kl_coef * per_token_kl
    per_response_objective = masked_sequence_mean(
        per_token_objective,
        completion_mask,
    )
    loss = -per_response_objective.mean()

    policy_loss = -masked_sequence_mean(
        torch.minimum(unclipped, clipped),
        completion_mask,
    ).mean()
    approx_kl = masked_sequence_mean(per_token_kl, completion_mask).mean()
    metrics = {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "approx_kl": approx_kl.detach(),
        "mean_ratio": masked_sequence_mean(token_ratio, completion_mask).mean().detach(),
    }
    return loss, metrics


def grpo_loss(
    policy_model,
    ref_model,
    batch,
    advantages,
    old_logprobs=None,
    clip_eps=0.2,
    kl_coef=0.04,
):
    completion_length = batch["completion_mask"].size(1)
    new_logprobs = per_token_logprobs(
        policy_model,
        batch["input_ids"],
        batch["attention_mask"],
        completion_length,
    )

    # With a single update, the old policy is the sampling policy; detach keeps the gradient of the ratio.
    if old_logprobs is None:
        old_logprobs = new_logprobs.detach()

    with torch.no_grad():
        ref_logprobs = per_token_logprobs(
            ref_model,
            batch["input_ids"],
            batch["attention_mask"],
            completion_length,
        )

    return grpo_objective_from_logprobs(
        new_logprobs,
        old_logprobs,
        ref_logprobs,
        batch["completion_mask"],
        advantages,
        clip_eps,
        kl_coef,
    )


# [G] Training step: sample, score, normalize within the group, then backpropagate
def train_step(
    policy_model,
    ref_model,
    optimizer,
    tokenizer,
    prompts,
    ground_truths,
    group_size=8,
):
    responses, _, batch = sample_groups(
        policy_model,
        tokenizer,
        prompts,
        group_size,
    )
    rewards = score_responses(
        responses,
        ground_truths,
        group_size,
        policy_model.device,
    )
    advantages = group_advantages(rewards, group_size)

    loss, metrics = grpo_loss(policy_model, ref_model, batch, advantages)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return metrics


# [H] GRPO training loop: generate fresh responses online every iteration
def train_grpo(policy_model, ref_model, optimizer, tokenizer, dataloader):
    ref_model.eval()
    for prompts, ground_truths in dataloader:
        metrics = train_step(
            policy_model,
            ref_model,
            optimizer,
            tokenizer,
            prompts,
            ground_truths,
        )
        print(
            "loss=",
            float(metrics["loss"]),
            "kl=",
            float(metrics["approx_kl"]),
        )
