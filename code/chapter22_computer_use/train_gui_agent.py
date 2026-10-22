"""Train a form-filling GUI agent with REINFORCE plus a learned baseline.

The lab compares four training regimes on the MiniForm environment
(see ``gui_env.py``):

* ``sparse``      -- full task, reward only on correct submission (+1 / 0).
* ``curriculum``  -- same sparse reward, but tasks ramp from easy (one
  field) to full (two fields + two boxes) as success improves.
* ``shaped``      -- full task with a progress reward for each newly
  satisfied sub-goal (Chapter 22, Step 7: dense progress rewards).
* ``hackable``    -- shaped rewards plus a bonus for every *failed*
  submit click (a correct submit still pays only +1), which reproduces
  the "submit early" reward-hacking failure mode.

Algorithm: Monte-Carlo REINFORCE with a state-value baseline, implemented
from scratch in torch (no RL libraries):

    G_t = r_t + gamma * r_{t+1} + ...          (discounted return)
    A_t = (G_t - V(s_t) - mean) / std          (standardised advantage)
    policy loss = -mean(log pi(a_t|s_t) * A_t) - c_ent * entropy
    value loss  = mean((V(s_t) - G_t)^2)

Subtracting the baseline leaves the expected gradient unchanged but cuts
its variance (see Chapter 6); standardising the advantages keeps one lucky
episode from dominating an update. Because lucky successes are rare and
each uses a different shuffled layout, the loop also replays
verifier-filtered successful trajectories with a small behaviour-cloning
loss -- the demonstration-SFT-then-online-RL recipe of Chapter 22, Step 4.
Only true task completions enter the replay buffer, never high-reward
failures.

Usage (from the repository root)::

    .venv/bin/python code/chapter22_computer_use/train_gui_agent.py --smoke
    .venv/bin/python code/chapter22_computer_use/train_gui_agent.py --mode all

Metrics (CSV), a JSON summary, and PNG curves are written under
``runs/chapter22_computer_use/``; a short summary table goes to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from gui_env import (HELDOUT_SEED_BASE, N_ACTIONS, N_LEVELS, OBS_DIM,
                     MiniFormEnv, count_satisfied)

EVAL_SEED_BASE = 500_000
TRAIN_SEED_BASE = 1_000_000


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    """Seed Python, numpy, and torch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(choice: str) -> torch.device:
    """Map --device {auto,cpu,cuda} to a torch device."""
    choice = choice.lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if choice != "cpu":
        raise ValueError(f"--device must be auto|cpu|cuda, got {choice!r}")
    return torch.device(choice)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    """pi_theta(a|s): screen features -> distribution over 28 GUI actions."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
    """V(s): screen features -> expected return (the REINFORCE baseline)."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# REINFORCE core (pure functions -- easy to unit-test by hand)
# ---------------------------------------------------------------------------

def compute_returns(rewards: list[float], gamma: float) -> list[float]:
    """Discounted returns G_t = r_t + gamma*r_{t+1} + ... (backwards pass)."""
    out, running = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def reinforce_losses(log_probs: torch.Tensor, returns: torch.Tensor,
                     values: torch.Tensor,
                     entropy: torch.Tensor, entropy_coef: float):
    """Policy + value losses for one batch of steps.

    Advantage A = standardised(G - V). Standardisation (zero mean, unit
    variance over the batch) keeps the update scale independent of the
    reward scale, which matters here because shaped returns are ~2x the
    sparse ones.
    """
    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8)
    policy_loss = -(log_probs * advantages).mean() - entropy_coef * entropy.mean()
    value_loss = nn.functional.mse_loss(values, returns)
    return policy_loss, value_loss, advantages


# ---------------------------------------------------------------------------
# Rollouts and evaluation
# ---------------------------------------------------------------------------

def run_episode(env: MiniFormEnv, policy: PolicyNet, device: torch.device,
                seed: int, greedy: bool = False) -> dict:
    """Roll out one episode; return states/actions/rewards plus the outcome.

    Each step is also marked "good" when it verifiably helped: it raised the
    satisfied sub-goal count to a new episode best, or it was the winning
    submit. This process-level filter (cf. progress rewards, Chapter 22
    Step 7, and process supervision, Chapter 17) lets the success replay
    below imitate only the useful steps of a lucky trajectory instead of
    its dozen junk steps.
    """
    obs = env.reset(seed=seed)
    best = count_satisfied(env.field_values, env.box_states, env.task)
    submit_eid = next(e.eid for e in env.elements if e.kind == "submit")
    states, actions, rewards, good = [], [], [], []
    done, info = False, {}
    while not done:
        states.append(obs)
        t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = policy(t)
            if greedy:
                action = int(logits.argmax())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample())
        obs, reward, done, info = env.step(action)
        sat = count_satisfied(env.field_values, env.box_states, env.task)
        good.append(sat > best or (info["submitted"] and info["success"]))
        best = max(best, sat)
        actions.append(action)
        rewards.append(reward)
    acts = np.array(actions, dtype=np.int64)
    return {"states": np.array(states, dtype=np.float32),
            "actions": acts,
            "rewards": rewards,
            "good": np.array(good, dtype=bool),
            "success": info["success"],
            "early_submit": info["early_submit"],
            "submit_frac": float((acts == submit_eid).mean()),
            "length": len(actions)}


def evaluate(policy: PolicyNet, seeds: list[int], level: int,
             reward_mode: str, device: torch.device) -> dict:
    """Greedy success rate / mean length / early-submit rate on fixed seeds."""
    env = MiniFormEnv(reward_mode=reward_mode, level=level)
    succ, lens, early, sfr = [], [], [], []
    for s in seeds:
        ep = run_episode(env, policy, device, seed=s, greedy=True)
        succ.append(float(ep["success"]))
        lens.append(ep["length"])
        early.append(float(ep["early_submit"]))
        sfr.append(ep["submit_frac"])
    return {"success": float(np.mean(succ)), "mean_length": float(np.mean(lens)),
            "early_submit": float(np.mean(early)),
            "submit_frac": float(np.mean(sfr))}


def train_mode(mode: str, episodes: int, seed: int, device: torch.device,
               eval_every: int, eval_n: int, progress: bool = True) -> dict:
    """Train one regime; return per-episode log + periodic eval snapshots."""
    seed_all(seed)
    reward_mode = {"sparse": "sparse", "curriculum": "sparse",
                   "shaped": "shaped", "hackable": "shaped_hackable"}[mode]
    policy = PolicyNet().to(device)
    value = ValueNet().to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=3e-3)
    opt_v = torch.optim.Adam(value.parameters(), lr=3e-3)
    # Imitation uses its own gentler optimizer: sharing the RL step size
    # lets a handful of early successes get memorised before the buffer
    # holds enough layouts to generalise from.
    opt_bc = torch.optim.Adam(policy.parameters(), lr=1e-3)

    env = MiniFormEnv(reward_mode=reward_mode, level=0)
    level = 0 if mode == "curriculum" else N_LEVELS - 1
    gamma, entropy_coef, batch_eps = 0.99, 0.05, 8
    # Verifier-filtered success replay (Chapter 22, Steps 4 and 6): rare
    # lucky successes are stored and imitated with a supervised loss next to
    # the REINFORCE update. Only true task completions enter the buffer -- in
    # "hackable" mode, high-reward submit-spam does NOT qualify -- and only
    # the steps that verifiably helped (see run_episode) are imitated, since
    # a lucky trajectory is mostly junk steps around one or two good ones.
    succ_buffer: list[dict] = []
    bc_coef, bc_steps, bc_epochs, bc_min_succ = 1.0, 256, 2, 32
    eps_at_level = 0

    ep_log: list[dict] = []
    eval_log: list[dict] = []
    batch: list[dict] = []
    last_bc: float = float("nan")
    eval_train = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + eval_n))
    eval_held = list(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + eval_n))

    for ep in range(episodes):
        env.level = level  # curriculum: current difficulty; else full task
        traj = run_episode(env, policy, device,
                           seed=TRAIN_SEED_BASE + seed * 1_000_000 + ep)
        batch.append(traj)
        ep_log.append({"episode": ep, "level": level,
                       "success": float(traj["success"]),
                       "length": traj["length"],
                       "reward": float(sum(traj["rewards"])),
                       "early_submit": float(traj["early_submit"]),
                       "submit_frac": traj["submit_frac"]})

        if len(batch) >= batch_eps:
            # One batched REINFORCE update over all steps in the batch.
            all_logp, all_ret, all_val, all_ent = [], [], [], []
            for b in batch:
                rets = compute_returns(b["rewards"], gamma)
                st = torch.as_tensor(b["states"], dtype=torch.float32, device=device)
                at = torch.as_tensor(b["actions"], dtype=torch.int64, device=device)
                logits = policy(st)
                dist = torch.distributions.Categorical(logits=logits)
                all_logp.append(dist.log_prob(at))
                all_ent.append(dist.entropy())
                all_val.append(value(st))
                all_ret.append(torch.as_tensor(rets, dtype=torch.float32, device=device))
                if b["success"]:
                    succ_buffer.append(b)
            succ_buffer = succ_buffer[-256:]
            logp = torch.cat(all_logp)
            ent = torch.cat(all_ent)
            val = torch.cat(all_val)
            ret = torch.cat(all_ret)
            p_loss, v_loss, _ = reinforce_losses(logp, ret, val, ent, entropy_coef)
            opt_p.zero_grad()
            p_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt_p.step()
            if len(succ_buffer) >= bc_min_succ:
                # Imitate filtered successes: behaviour cloning on the GOOD
                # steps sampled from the success buffer. Gated on buffer
                # size and kept gentle: cloning a dozen early successes
                # memorises layouts instead of learning the shared
                # label-matching rule.
                for _ in range(bc_epochs):
                    eps = random.sample(succ_buffer, min(len(succ_buffer), 32))
                    ss = np.concatenate([e["states"][e["good"]] for e in eps
                                         if e["good"].any()])
                    aa = np.concatenate([e["actions"][e["good"]] for e in eps
                                         if e["good"].any()])
                    if len(ss) == 0:
                        continue
                    bs = torch.as_tensor(ss, dtype=torch.float32, device=device)
                    ba = torch.as_tensor(aa, dtype=torch.int64, device=device)
                    if len(bs) > bc_steps:
                        idx = torch.randperm(len(bs), device=device)[:bc_steps]
                        bs, ba = bs[idx], ba[idx]
                    blogits = policy(bs)
                    bc_loss = nn.functional.cross_entropy(blogits, ba)
                    last_bc = float(bc_loss.detach().cpu())
                    opt_bc.zero_grad()
                    (bc_coef * bc_loss).backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    opt_bc.step()
            opt_v.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(value.parameters(), 1.0)
            opt_v.step()
            batch = []

        if (ep + 1) % eval_every == 0 or ep == episodes - 1:
            # Eval snapshots always use the FULL task, so regimes stay
            # comparable even while the curriculum trains on easy levels.
            snap: dict = {"episode": ep + 1, "level": level}
            for split, seeds in (("train", eval_train), ("heldout", eval_held)):
                m = evaluate(policy, seeds, N_LEVELS - 1, reward_mode, device)
                snap[f"{split}_success"] = m["success"]
                snap[f"{split}_length"] = m["mean_length"]
                snap[f"{split}_early"] = m["early_submit"]
                snap[f"{split}_submit_frac"] = m["submit_frac"]
            eval_log.append(snap)
            if mode == "curriculum":
                # Promote once the current difficulty is mostly solved
                # (rolling training success over the last 200 episodes).
                window = ep_log[-200:]
                rate = sum(e["success"] for e in window) / len(window)
                eps_at_level += eval_every
                if rate >= 0.5 and eps_at_level >= 400 and level < N_LEVELS - 1:
                    level += 1
                    eps_at_level = 0
                    succ_buffer = []  # old-level successes would teach
                    # premature submit at the harder level; relearn from new
                    # successes instead.
            if progress:
                print(f"  [{mode}] ep={ep + 1}/{episodes} level={level} "
                      f"train_succ={snap['train_success']:.2f} "
                      f"heldout_succ={snap['heldout_success']:.2f} "
                      f"len={snap['train_length']:.1f} "
                      f"buf={len(succ_buffer)} bc={last_bc:.3f}", flush=True)
    return {"ep_log": ep_log, "eval_log": eval_log}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_curves(all_evals: dict[str, list[dict]], out: str) -> None:
    modes = list(all_evals.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [("train_success", "Train success rate (greedy, full task)"),
             ("heldout_success", "Held-out layout success rate (greedy)"),
             ("train_length", "Mean episode length (train eval)"),
             ("train_submit_frac", "Submit-click share (train eval)")]
    for ax, (key, title) in zip(axes.flat, specs):
        for m in modes:
            xs = [e["episode"] for e in all_evals[m]]
            ys = [e[key] for e in all_evals[m]]
            ax.plot(xs, ys, marker="o", ms=3, label=m)
        ax.set_title(title)
        ax.set_xlabel("training episode")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("MiniForm GUI agent: REINFORCE + baseline under four regimes")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def print_table(summary: dict[str, dict]) -> None:
    print("\nSummary (greedy policy, full task, last eval snapshot; "
          "mean_rew = mean undiscounted reward over last 200 training episodes)")
    print(f"  {'regime':>10}  {'train_succ':>10}  {'heldout':>8}  "
          f"{'mean_len':>8}  {'submit_fr':>9}  {'mean_rew':>8}")
    for mode, s in summary.items():
        print(f"  {mode:>10}  {s['train_success']:>10.3f}  "
              f"{s['heldout_success']:>8.3f}  {s['mean_length']:>8.2f}  "
              f"{s['submit_frac']:>9.3f}  {s['mean_reward']:>8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a MiniForm GUI agent.")
    ap.add_argument("--mode", default="all",
                    choices=["all", "sparse", "curriculum", "shaped", "hackable"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny config finishing in well under 60 s on CPU")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--outdir", default="runs/chapter22_computer_use")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = resolve_device(args.device)
    if device.type == "cpu":
        # Tiny MLPs are dominated by BLAS thread-pool overhead (measured
        # ~25 ms vs ~0.04 ms per forward on this machine); one thread is
        # ~600x faster here and keeps the full run inside the CPU budget.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    print(f"device: {device}", flush=True)

    if args.smoke:
        episodes = {"sparse": 96, "curriculum": 96, "shaped": 96, "hackable": 96}
        eval_every, eval_n = 48, 8
    else:
        episodes = {"sparse": 3000, "curriculum": 15000,
                    "shaped": 15000, "hackable": 3000}
        eval_every, eval_n = 500, 64
    if args.episodes is not None:
        episodes = {k: args.episodes for k in episodes}
    if args.eval_every is not None:
        eval_every = args.eval_every

    modes = [args.mode] if args.mode != "all" else list(episodes)
    t0 = time.time()
    all_evals, summary = {}, {}
    for i, mode in enumerate(modes):
        print(f"[{mode}] training {episodes[mode]} episodes "
              f"(seed={args.seed + i}) ...", flush=True)
        res = train_mode(mode, episodes[mode], args.seed + i, device,
                         eval_every, eval_n)
        write_csv(f"{args.outdir}/metrics_{mode}.csv", res["ep_log"])
        write_csv(f"{args.outdir}/eval_{mode}.csv", res["eval_log"])
        all_evals[mode] = res["eval_log"]
        last = res["eval_log"][-1]
        summary[mode] = {"train_success": last["train_success"],
                         "heldout_success": last["heldout_success"],
                         "mean_length": last["train_length"],
                         "early_submit": last["train_early"],
                         "submit_frac": last["train_submit_frac"],
                         "mean_reward": float(np.mean(
                             [e["reward"] for e in res["ep_log"][-200:]])),
                         "episodes": episodes[mode]}
    if len(modes) > 1:
        plot_curves(all_evals, f"{args.outdir}/training_curves.png")
    with open(f"{args.outdir}/summary.json", "w") as f:
        json.dump({"seed": args.seed, "episodes": episodes,
                   "eval_every": eval_every, "eval_n": eval_n,
                   "runtime_s": time.time() - t0, "summary": summary},
                  f, indent=2)
    print_table(summary)
    print(f"runtime: {time.time() - t0:.1f} s; outputs in {args.outdir}/")


if __name__ == "__main__":
    main()
