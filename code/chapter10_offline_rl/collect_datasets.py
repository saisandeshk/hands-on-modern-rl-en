"""Step 1 of the Chapter 10 lab: build fixed offline datasets from CartPole.

Pipeline
--------
1. Train a small online DQN from scratch (a few hundred episodes on CPU).
   This is the *only* place in the lab where environment interaction is used
   to improve a policy.
2. Freeze that DQN and derive behaviour policies of different quality:
   ``random`` (uniform), ``medium`` (trained Q with eps=0.3), ``expert``
   (trained Q, nearly greedy), and ``mixed`` (round-robin over the three).
3. Roll out each behaviour policy *without any learning* and store the
   transitions (s, a, r, s', done) plus the behaviour action probabilities
   needed later for importance sampling.

Outputs (under ``runs/chapter10_offline_rl/``)
----------------------------------------------
- ``datasets/<name>.npz`` + ``datasets/<name>_meta.json`` for each behaviour.
- ``expert_q.pt``: the trained DQN weights (the "expert" target policy).
- ``collect_summary.json``: behaviour returns observed during collection.

Example:
    python collect_datasets.py --seed 0 --device cpu
    python collect_datasets.py --smoke --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from offline_common import (
    QMLP,
    default_outdir,
    evaluate_greedy,
    resolve_device,
    save_dataset,
    seed_everything,
)

BEHAVIOURS = ("random", "medium", "expert", "mixed")
MEDIUM_EPS = 0.3
EXPERT_EPS = 0.02


# ---------------------------------------------------------------------------
# Online DQN (used only to create behaviour policies of varying quality)
# ---------------------------------------------------------------------------


def train_online_dqn(
    env_name: str,
    n_episodes: int,
    seed: int,
    device: torch.device,
    hidden: int = 64,
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    target_sync_every: int = 300,
    warmup: int = 1000,
) -> QMLP:
    """Train a small DQN online with epsilon-greedy exploration."""
    import gymnasium as gym
    from collections import deque

    train_env = gym.make(env_name)
    train_env.action_space.seed(seed)
    train_env.observation_space.seed(seed)
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    q_net = QMLP(state_dim, action_dim, hidden=hidden).to(device)
    target_net = QMLP(state_dim, action_dim, hidden=hidden).to(device)
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=lr)

    buf: deque = deque(maxlen=20000)
    eps = eps_start
    eps_decay = (eps_end / eps_start) ** (1.0 / max(1, n_episodes))
    steps = 0

    best_score = -1.0
    best_state = None

    def _greedy_return(episodes: int = 5) -> float:
        import gymnasium as gym

        total = 0.0
        q_net.eval()
        with torch.no_grad():
            for k in range(episodes):
                e = gym.make(env_name)
                o, _ = e.reset(seed=987654 + steps + k)
                d = False
                while not d:
                    s_ = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                    o, r_, term, trunc, _ = e.step(int(q_net(s_).argmax(dim=1).item()))
                    total += float(r_)
                    d = bool(term or trunc)
                e.close()
        q_net.train()
        return total / episodes

    obs, _ = train_env.reset(seed=seed)
    for ep in range(n_episodes):
        done = False
        while not done:
            if np.random.rand() < eps:
                action = int(train_env.action_space.sample())
            else:
                with torch.no_grad():
                    s = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(q_net(s).argmax(dim=1).item())
            nxt, reward, terminated, truncated, _ = train_env.step(action)
            done = bool(terminated or truncated)
            buf.append((obs, action, float(reward), nxt, float(terminated)))
            obs = nxt
            if done:
                obs, _ = train_env.reset()
            steps += 1

            if len(buf) >= batch_size and steps >= warmup:
                idx = np.random.choice(len(buf), batch_size, replace=False)
                b = [buf[i] for i in idx]
                s_b = torch.as_tensor(np.array([t[0] for t in b]), dtype=torch.float32, device=device)
                a_b = torch.as_tensor(np.array([t[1] for t in b]), dtype=torch.int64, device=device)
                r_b = torch.as_tensor(np.array([t[2] for t in b]), dtype=torch.float32, device=device)
                ns_b = torch.as_tensor(np.array([t[3] for t in b]), dtype=torch.float32, device=device)
                d_b = torch.as_tensor(np.array([t[4] for t in b]), dtype=torch.float32, device=device)
                q_sa = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    y = r_b + gamma * (1.0 - d_b) * target_net(ns_b).max(dim=1)[0]
                loss = nn.functional.mse_loss(q_sa, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()
                if steps % target_sync_every == 0:
                    target_net.load_state_dict(q_net.state_dict())
        eps = max(eps_end, eps * eps_decay)

        # Keep the best checkpoint: online evaluation during online
        # training is legitimate (the offline ban starts at collection).
        if (ep + 1) % 25 == 0:
            score = _greedy_return()
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in q_net.state_dict().items()}
    train_env.close()
    if best_state is not None:
        q_net.load_state_dict(best_state)
        print(f"[collect] best online checkpoint: {best_score:.1f} mean return")
    return q_net


# ---------------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------------


def make_behaviour_sampler(
    name: str, q_net: QMLP, action_dim: int, device: torch.device
):
    """Return ``obs -> (action, prob_vector)`` for a behaviour policy."""
    q_net = q_net.to(device).eval()

    def greedy_probs(obs: np.ndarray, eps: float) -> tuple[int, np.ndarray]:
        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            greedy = int(q_net(s).argmax(dim=1).item())
        probs = np.full(action_dim, eps / action_dim)
        probs[greedy] += 1.0 - eps
        action = greedy if np.random.rand() > eps else int(np.random.randint(action_dim))
        return action, probs

    def sampler(obs: np.ndarray, ep_index: int) -> tuple[int, np.ndarray]:
        if name == "random":
            probs = np.full(action_dim, 1.0 / action_dim)
            return int(np.random.randint(action_dim)), probs
        if name == "medium":
            return greedy_probs(obs, MEDIUM_EPS)
        if name == "expert":
            return greedy_probs(obs, EXPERT_EPS)
        if name == "mixed":
            # Round-robin episodes across random / medium / expert sources.
            source = ("random", "medium", "expert")[ep_index % 3]
            if source == "random":
                probs = np.full(action_dim, 1.0 / action_dim)
                return int(np.random.randint(action_dim)), probs
            return greedy_probs(obs, MEDIUM_EPS if source == "medium" else EXPERT_EPS)
        raise ValueError(f"unknown behaviour {name!r}")

    return sampler


def collect_one_dataset(
    name: str,
    q_net: QMLP,
    env_name: str,
    num_transitions: int,
    seed: int,
    device: torch.device,
) -> tuple[dict, dict]:
    """Roll out a behaviour policy and record fixed transitions (no learning)."""
    import gymnasium as gym

    action_dim = gym.make(env_name).action_space.n
    sampler = make_behaviour_sampler(name, q_net, action_dim, device)
    rng_seed = seed * 1000 + BEHAVIOURS.index(name)

    states, actions, rewards, next_states, dones, bprobs, ends = [], [], [], [], [], [], []
    ep_returns: list[float] = []
    ep_index, total, ep_ret = 0, 0, 0.0
    env = gym.make(env_name)
    env.action_space.seed(rng_seed)
    env.observation_space.seed(rng_seed)
    obs, _ = env.reset(seed=rng_seed)
    while total < num_transitions:
        action, probs = sampler(obs, ep_index)
        nxt, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        states.append(obs)
        actions.append(action)
        rewards.append(float(reward))
        next_states.append(nxt)
        dones.append(float(terminated))  # timeouts are not terminal for bootstrapping
        ends.append(float(done))  # episode boundary (terminated or timeout)
        bprobs.append(probs)
        ep_ret += float(reward)
        total += 1
        obs = nxt
        if done:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            ep_index += 1
            obs, _ = env.reset(seed=rng_seed + ep_index)
    env.close()
    arrays = {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
        "ends": np.array(ends, dtype=np.float32),
        "behaviour_probs": np.array(bprobs, dtype=np.float32),
    }
    meta = {
        "behaviour": name,
        "num_transitions": int(total),
        "num_episodes": int(len(ep_returns)),
        "mean_episode_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "seed": seed,
    }
    return arrays, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect fixed offline-RL datasets (CartPole-v1).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--smoke", action="store_true", help="tiny config finishing in <60 s on CPU")
    p.add_argument("--num-transitions", type=int, default=50000, help="transitions per dataset")
    p.add_argument("--online-episodes", type=int, default=500, help="online DQN training episodes")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.num_transitions = 1500
        args.online_episodes = 25
    seed_everything(args.seed)
    device = resolve_device(args.device)
    outdir = args.outdir or default_outdir()
    t0 = time.time()

    print(f"[collect] training online DQN ({args.online_episodes} episodes) on {device} ...")
    warmup = 128 if args.smoke else 1000
    q_net = train_online_dqn("CartPole-v1", args.online_episodes, args.seed, device,
                             warmup=warmup)
    expert = evaluate_greedy(q_net, n_episodes=10, base_seed=10_000 + args.seed, device=str(device))
    print(f"[collect] online DQN greedy return: {expert.mean:.1f} +/- {expert.std:.1f} (10 eps)")

    os.makedirs(os.path.join(outdir, "datasets"), exist_ok=True)
    torch.save(q_net.state_dict(), os.path.join(outdir, "expert_q.pt"))

    summary: dict = {
        "seed": args.seed,
        "online_episodes": args.online_episodes,
        "expert_eval": expert.as_dict(),
        "datasets": {},
    }
    for name in BEHAVIOURS:
        arrays, meta = collect_one_dataset(
            name, q_net, "CartPole-v1", args.num_transitions, args.seed, device
        )
        path = os.path.join(outdir, "datasets", f"{name}.npz")
        meta["expert_q"] = "expert_q.pt"
        save_dataset(path, arrays, meta)
        summary["datasets"][name] = meta
        print(
            f"[collect] {name:6s}: {meta['num_transitions']} transitions, "
            f"{meta['num_episodes']} episodes, mean return {meta['mean_episode_return']:.1f}"
        )

    with open(os.path.join(outdir, "collect_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[collect] done in {time.time() - t0:.1f}s. Files under {outdir}/datasets/")


if __name__ == "__main__":
    main()
