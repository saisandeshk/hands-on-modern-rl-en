"""Chapter 18 lab: rollout/learner systems and policy staleness.

A small actor-learner system on CartPole-v1 (optionally LunarLander-v3) that
mirrors, at textbook scale, the industrial rollout/training split described in
Chapter 18 (18.1 sync vs async, 18.4 distributed RL):

* K rollout workers hold a *copy* of the policy, collect trajectories, and
  send them through a queue (real ``multiprocessing`` in the throughput
  experiment, a faithful deterministic simulation of the same queue system in
  the staleness experiments).
* A learner updates the policy and broadcasts new weights every M updates.

Three experiments:

1. ``throughput`` -- env steps/second and learner utilization for K workers.
2. ``staleness``  -- learning curves for a synchronous on-policy baseline vs
   asynchronous updates with no correction vs truncated importance weights
   (PPO-style ratio clipping or V-trace-style truncation against the stored
   behaviour log-probs).
3. ``sweep``      -- final return as the broadcast delay (policy lag) grows.

Every batch records the behaviour-policy version that generated it, so the
learner can report exactly how many versions stale each update is -- the same
bookkeeping AReaL-style systems attach to each trajectory (18.1 on sync
vs async, 18.4 on asynchronous scheduling).

Run from the repository root::

    .venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment all
    .venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment all --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, numpy and torch (CPU ops are then deterministic)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(choice: str) -> torch.device:
    choice = choice.lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(choice)


# ---------------------------------------------------------------------------
# Policy: shared-trunk actor-critic (same family as Chapter 8's PPO net)
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """MLP actor-critic: obs -> 64 -> 64 (ReLU), then action logits + V(s)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs: np.ndarray):
        """Sample one action; return (action, logprob, value) as Python floats."""
        with torch.no_grad():
            logits, value = self.forward(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Batched log-probs, values and entropy for a learner update."""
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()


# ---------------------------------------------------------------------------
# Batch + staleness correction math (the core RL of this lab)
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    states: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    beh_logprobs: np.ndarray  # log pi_behaviour(a|s), stored at rollout time
    beh_version: int = 0      # policy version that generated this batch
    ep_returns: list = field(default_factory=list)


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Discounted return G_t for one episode, computed backwards."""
    out, g = [0.0] * len(rewards), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def collect_batch(policy: ActorCritic, env_name: str, num_steps: int,
                   seed: int, beh_version: int = 0) -> Batch:
    """Roll out `policy` until `num_steps` env steps are stored.

    Each transition keeps the log-probability under the policy that generated
    it -- the rollout-time record industrial systems must keep (18.4's
    training-inference mismatch discussion: never substitute a trainer-side
    recomputation for the true behaviour prob).
    """
    env = gym.make(env_name)
    states, actions, rewards, beh_lps, dones = [], [], [], [], []
    ep_returns, ep_rew = [], 0.0
    obs, _ = env.reset(seed=seed)
    for _ in range(num_steps):
        a, lp, _ = policy.act(obs)
        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        states.append(obs)
        actions.append(a)
        rewards.append(float(r))
        beh_lps.append(lp)
        dones.append(done)
        ep_rew += float(r)
        obs = obs2
        if done:
            ep_returns.append(ep_rew)
            ep_rew = 0.0
            obs, _ = env.reset()
    env.close()
    # Monte-Carlo returns with episode boundaries respected.
    returns = np.zeros(len(rewards), dtype=np.float64)
    g, gamma = 0.0, 0.99
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + (0.0 if dones[t] else gamma * g)
        returns[t] = g
    return Batch(states=np.asarray(states, dtype=np.float32),
                 actions=np.asarray(actions, dtype=np.int64),
                 returns=returns,
                 beh_logprobs=np.asarray(beh_lps, dtype=np.float32),
                 beh_version=beh_version,
                 ep_returns=ep_returns)


def importance_ratios(new_logprobs: torch.Tensor,
                      beh_logprobs: torch.Tensor) -> torch.Tensor:
    """rho = pi(a|s) / pi_behaviour(a|s); the staleness ratio of 18.1."""
    return torch.exp(new_logprobs - beh_logprobs)


def ppo_clipped_weights(rho: torch.Tensor, eps: float = 0.2) -> torch.Tensor:
    """PPO-style truncation of the staleness ratio to [1-eps, 1+eps]."""
    return torch.clamp(rho, 1.0 - eps, 1.0 + eps)


def vtrace_truncated_weights(new_logprobs: torch.Tensor,
                            beh_logprobs: torch.Tensor,
                            c_bar: float = 1.0,
                            rho_bar: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """V-trace-style two-level truncation (cf. AReaL's [0.8, 1.2] clip, 18.4).

    * ``c = min(c_bar, rho)`` scales the trace / policy term, cutting how far
      one stale step can pull the update.
    * ``rho_tr = min(rho_bar, rho)`` truncates the fixed-point weight used for
      the value target, so a stale batch cannot dominate the critic either.

    This is a simplified one-step Monte-Carlo analogue of V-trace (Espeholt et
    al., 2018): the real V-trace recursion applies the truncation to TD
    residuals ``r + gamma * V(s') - V(s)`` with bootstrapped values, while here
    the target is ``V(s) + rho_tr * (G - V(s))`` with Monte-Carlo returns G.
    """
    rho = torch.exp(new_logprobs - beh_logprobs)
    c = torch.minimum(torch.full_like(rho, c_bar), rho)
    rho_tr = torch.minimum(torch.full_like(rho, rho_bar), rho)
    return c, rho_tr


def behaviour_version_at(update: int, broadcast_every: int, delay: int) -> int:
    """Policy version the rollout workers hold at learner update `update`.

    Weights are broadcast every `broadcast_every` updates and arrive `delay`
    updates late, so workers hold the newest broadcast not newer than
    ``update - delay`` (version 0 = initial weights).
    """
    t = update - delay
    if t < 0:
        return 0
    return (t // broadcast_every) * broadcast_every


@dataclass
class LagTracker:
    """Bookkeeping: how many versions stale was each batch used for training."""
    lags: list[int] = field(default_factory=list)

    def record(self, learner_version: int, beh_version: int) -> int:
        lag = learner_version - beh_version
        self.lags.append(lag)
        return lag

    def summary(self) -> dict:
        if not self.lags:
            return {"mean_lag": 0.0, "max_lag": 0, "n": 0}
        return {"mean_lag": float(np.mean(self.lags)),
                "max_lag": int(np.max(self.lags)), "n": len(self.lags)}


def learner_update(policy: ActorCritic, opt: torch.optim.Optimizer, batch: Batch,
                   mode: str, device: torch.device, epochs: int = 4,
                   clip_eps: float = 0.2, c_bar: float = 1.0,
                   rho_bar: float = 1.0) -> dict:
    """One learner step on a batch. `mode` selects the staleness correction.

    * ``sync``/``clip``: PPO-clip loss against the stored behaviour log-probs
      (for sync data the ratio is ~1 and clipping rarely binds).
    * ``none``: plain policy gradient -- stale ratios are ignored entirely.
    * ``vtrace``: V-trace-style truncated weights on both actor and critic.
    """
    states = torch.as_tensor(batch.states, dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=device)
    returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
    beh_lp = torch.as_tensor(batch.beh_logprobs, dtype=torch.float32, device=device)

    for _ in range(epochs):
        new_lp, values, entropy = policy.evaluate(states, actions)
        adv = returns - values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        rho = importance_ratios(new_lp, beh_lp)
        if mode == "none":
            pg_loss = -(new_lp * adv.detach()).mean()
            v_loss = ((values - returns) ** 2).mean()
        elif mode == "vtrace":
            c, rho_tr = vtrace_truncated_weights(new_lp, beh_lp, c_bar, rho_bar)
            pg_loss = -(c.detach() * adv.detach() * new_lp).mean()
            v_target = (values.detach()
                        + rho_tr.detach() * (returns - values.detach()))
            v_loss = ((values - v_target.detach()) ** 2).mean()
        else:  # "sync" or "clip": PPO-style ratio clipping
            clipped = ppo_clipped_weights(rho, clip_eps)
            pg_loss = -torch.min(rho * adv.detach(),
                                 clipped * adv.detach()).mean()
            v_loss = ((values - returns) ** 2).mean()
        loss = pg_loss + 0.5 * v_loss - 0.01 * entropy.mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()

    with torch.no_grad():
        new_lp, _, _ = policy.evaluate(states, actions)
        rho = importance_ratios(new_lp, beh_lp)
    return {"loss": float(loss.item()),
            "mean_rho": float(rho.mean().item()),
            "clip_frac": float((rho.lt(1 - clip_eps)
                                | rho.gt(1 + clip_eps)).float().mean().item())}


@torch.no_grad()
def evaluate_greedy(policy: ActorCritic, env_name: str, episodes: int,
                    seed: int) -> float:
    """Mean greedy return -- the learning-curve metric."""
    env = gym.make(env_name)
    total = 0.0
    for e in range(episodes):
        obs, _ = env.reset(seed=seed + e)
        done, ep = False, 0.0
        while not done:
            logits, _ = policy.forward(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            obs, r, term, trunc, _ = env.step(int(logits.argmax(-1).item()))
            done = term or trunc
            ep += float(r)
        total += ep
    env.close()
    return total / max(episodes, 1)


# ---------------------------------------------------------------------------
# Experiment 2/3: deterministic simulation of the async actor-learner system
# ---------------------------------------------------------------------------

def simulate_training(env_name: str, num_updates: int, steps_per_update: int,
                      mode: str, broadcast_every: int, delay: int, seed: int,
                      device: torch.device, eval_episodes: int = 3,
                      eval_every: int = 1, hidden: int = 64, lr: float = 3e-3,
                      epochs: int = 4, clip_eps: float = 0.2,
                      c_bar: float = 1.0, rho_bar: float = 1.0) -> dict:
    """Simulate K-worker async training in one process, deterministically.

    The learner keeps every past weight version; at update `t` the (simulated)
    workers hold ``behaviour_version_at(t, broadcast_every, delay)``. A real
    queue would interleave the same (version, data) pairs -- the simulation
    only removes process scheduling noise so curves are reproducible.
    """
    set_seed(seed)
    env_probe = gym.make(env_name)
    obs_dim = env_probe.observation_space.shape[0]
    act_dim = env_probe.action_space.n
    env_probe.close()

    learner = ActorCritic(obs_dim, act_dim, hidden).to(device)
    behaviour = ActorCritic(obs_dim, act_dim, hidden).to(device)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)
    snapshots = [{k: v.cpu().clone() for k, v in learner.state_dict().items()}]
    tracker = LagTracker()
    curve, lags, rhos, clip_fracs = [], [], [], []

    for t in range(num_updates):
        bv = behaviour_version_at(t, broadcast_every, delay)
        behaviour.load_state_dict(
            {k: v.to(device) for k, v in snapshots[bv].items()})
        batch = collect_batch(behaviour, env_name, steps_per_update,
                              seed=seed * 100003 + t, beh_version=bv)
        # Read the version back off the batch -- the way a real queue would
        # deliver (version, data) pairs -- rather than reusing the local `bv`.
        lag = tracker.record(t, batch.beh_version)
        stats = learner_update(learner, opt, batch, mode, device, epochs,
                               clip_eps, c_bar, rho_bar)
        snapshots.append(
            {k: v.cpu().clone() for k, v in learner.state_dict().items()})
        lags.append(lag)
        rhos.append(stats["mean_rho"])
        clip_fracs.append(stats["clip_frac"])
        if (t + 1) % eval_every == 0 or t == num_updates - 1:
            curve.append(evaluate_greedy(learner, env_name, eval_episodes,
                                         seed=10_000_000 + t))
    return {"eval_curve": curve, "lags": lags, "mean_lag": float(np.mean(lags)),
            "max_lag": int(np.max(lags)), "final_return": curve[-1],
            "mean_rho": float(np.mean(rhos)),
            "mean_clip_frac": float(np.mean(clip_fracs)),
            "per_update_rho": rhos, "per_update_clip_frac": clip_fracs,
            "lag_summary": tracker.summary()}


# ---------------------------------------------------------------------------
# Experiment 1: real multiprocessing throughput benchmark
# ---------------------------------------------------------------------------

def _mp_worker(seed: int, state_dict: dict, steps_budget: int, env_name: str,
               worker_id: int, queue, hidden: int) -> None:
    """Rollout worker: load broadcast weights, collect, send shard via queue."""
    import gymnasium as _gym
    import torch as _torch
    _torch.manual_seed(seed)
    env = _gym.make(env_name)
    policy = ActorCritic(env.observation_space.shape[0],
                         env.action_space.n, hidden)
    policy.load_state_dict(state_dict)
    policy.eval()
    queue.put(("ready", worker_id, 0.0))
    obs, _ = env.reset(seed=seed)
    steps, ep_rew, ep_returns = 0, 0.0, []
    with _torch.no_grad():
        while steps < steps_budget:
            logits, _ = policy.forward(
                _torch.as_tensor(obs, dtype=_torch.float32).unsqueeze(0))
            dist = _torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            obs, r, term, trunc, _ = env.step(int(a.item()))
            steps += 1
            ep_rew += float(r)
            if term or trunc:
                ep_returns.append(ep_rew)
                ep_rew = 0.0
                obs, _ = env.reset()
    env.close()
    queue.put(("shard", worker_id,
               {"steps": steps, "mean_ep_return": float(np.mean(ep_returns))
                if ep_returns else 0.0, "n_episodes": len(ep_returns)}))


def run_throughput(env_name: str, workers_list: list[int], steps_per_worker: int,
                   seed: int, device: torch.device, hidden: int = 64,
                   lr: float = 3e-3) -> list[dict]:
    """Spawn K rollout processes per setting; the parent is the learner.

    Steady-state throughput excludes spawn/startup (timed from all-ready to
    all-received); end-to-end wall time includes it. Learner utilization is the
    fraction of the steady-state window spent doing real gradient updates on
    received shards (a fixed synthetic minibatch standing in for the update
    workload, so the number is comparable across K).
    """
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    probe = gym.make(env_name)
    obs_dim, act_dim = probe.observation_space.shape[0], probe.action_space.n
    probe.close()
    learner = ActorCritic(obs_dim, act_dim, hidden).to(device)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)
    state_dict = {k: v.cpu() for k, v in learner.state_dict().items()}
    # Fixed synthetic update workload for a fair utilization comparison.
    synth = (torch.randn(256, obs_dim), torch.randint(0, act_dim, (256,)))

    rows = []
    for k in workers_list:
        queue = ctx.Queue()
        t_spawn = time.perf_counter()
        procs = [ctx.Process(target=_mp_worker,
                             args=(seed * 1000 + w, state_dict,
                                   steps_per_worker, env_name, w, queue, hidden))
                 for w in range(k)]
        for p in procs:
            p.start()
        ready, shards, total_steps, ep_ret_sum, ep_n = 0, 0, 0, 0.0, 0
        t0 = None
        busy = 0.0
        while ready < k or shards < k:
            kind, _wid, payload = queue.get()
            if kind == "ready":
                ready += 1
                if ready == k:
                    t0 = time.perf_counter()
            else:
                shards += 1
                total_steps += payload["steps"]
                ep_ret_sum += payload["mean_ep_return"] * payload["n_episodes"]
                ep_n += payload["n_episodes"]
                # Learner update workload on the received shard.
                t1 = time.perf_counter()
                s, a = synth[0].to(device), synth[1].to(device)
                lp, vals, ent = learner.evaluate(s, a)
                loss = -lp.mean() + 0.5 * (vals ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                busy += time.perf_counter() - t1
        t1 = time.perf_counter()
        for p in procs:
            p.join()
        rows.append({"workers": k, "total_steps": total_steps,
                     "startup_s": (t0 - t_spawn) if t0 else math.nan,
                     "steady_s": t1 - t0 if t0 else math.nan,
                     "wall_s": t1 - t_spawn,
                     "steps_per_sec": total_steps / max(t1 - t0, 1e-9) if t0 else 0.0,
                     "learner_util": busy / max(t1 - t0, 1e-9) if t0 else 0.0,
                     "mean_ep_return": ep_ret_sum / max(ep_n, 1)})
    return rows


# ---------------------------------------------------------------------------
# Plotting + persistence
# ---------------------------------------------------------------------------

def save_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_curves(outdir: str, curves: dict[str, list[float]], title: str,
                fname: str, xlabel: str = "learner update") -> str:
    plt.figure()
    for label, ys in curves.items():
        plt.plot(range(1, len(ys) + 1), ys, marker="o", markersize=3, label=label)
    plt.xlabel(xlabel)
    plt.ylabel("mean greedy return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_sweep(outdir: str, sweep: list[dict]) -> str:
    plt.figure()
    for mode in sorted({r["mode"] for r in sweep}):
        xs = [r["delay"] for r in sweep if r["mode"] == mode]
        ys = [r["final_return"] for r in sweep if r["mode"] == mode]
        plt.plot(xs, ys, marker="o", label=f"async-{mode}")
    plt.xlabel("broadcast delay (versions stale)")
    plt.ylabel("final mean greedy return")
    plt.title("Staleness sweep: correction vs no correction")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "staleness_sweep.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def print_table(title: str, rows: list[dict], cols: list[str]) -> None:
    print(f"\n{title}")
    widths = [max(len(c), max(len(f"{r[c]:.4g}" if isinstance(r[c], float)
                                   else str(r[c])) for r in rows)) for c in cols]
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(
            (f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])).ljust(w)
            for c, w in zip(cols, widths)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiment", default="all",
                   choices=["throughput", "staleness", "sweep", "all"])
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--workers", default="1,2,4,8")
    p.add_argument("--steps-per-worker", type=int, default=8000)
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--steps-per-update", type=int, default=2048)
    p.add_argument("--modes", default="sync,none,clip,vtrace")
    p.add_argument("--delay", type=int, default=4)
    p.add_argument("--delays", default="0,1,2,4,8")
    p.add_argument("--broadcast-every", type=int, default=1)
    p.add_argument("--sweep-modes", default="none,clip")
    p.add_argument("--sweep-updates", type=int, default=25)
    p.add_argument("--sweep-steps", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--c-bar", type=float, default=1.0)
    p.add_argument("--rho-bar", type=float, default=1.0)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--outdir", default="runs/chapter18_llm_rl_industrial")
    p.add_argument("--smoke", action="store_true",
                   help="tiny config finishing in <60 s on CPU")
    return p


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    if gym is None:
        raise RuntimeError("gymnasium is required")
    if args.smoke:
        args.workers = "1,2"
        args.steps_per_worker = 800
        args.updates = 4
        args.steps_per_update = 256
        args.modes = "sync,clip"
        args.delays = "0,2"
        args.sweep_modes = "none,clip"
        args.sweep_updates = 3
        args.sweep_steps = 256
        args.eval_episodes = 2
    device = resolve_device(args.device)
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    summary: dict = {"seed": args.seed, "env": args.env,
                     "device": str(device)}

    if args.experiment in ("throughput", "all"):
        klist = [int(x) for x in args.workers.split(",")]
        rows = run_throughput(args.env, klist, args.steps_per_worker,
                              args.seed, device, args.hidden, args.lr)
        save_csv(os.path.join(args.outdir, "throughput.csv"), rows)
        print_table("Throughput (steady-state; startup excluded)",
                    rows, ["workers", "total_steps", "startup_s", "steady_s",
                           "steps_per_sec", "learner_util", "mean_ep_return"])
        summary["throughput"] = rows

    if args.experiment in ("staleness", "all"):
        curves, stale_detail = {}, {}
        for mode in args.modes.split(","):
            delay = 0 if mode == "sync" else args.delay
            res = simulate_training(args.env, args.updates,
                                    args.steps_per_update, mode,
                                    args.broadcast_every, delay, args.seed,
                                    device, args.eval_episodes,
                                    hidden=args.hidden, lr=args.lr,
                                    epochs=args.epochs,
                                    clip_eps=args.clip_eps, c_bar=args.c_bar,
                                    rho_bar=args.rho_bar)
            label = f"{mode}(delay={delay})"
            curves[label] = res["eval_curve"]
            stale_detail[label] = res
        save_csv(os.path.join(args.outdir, "staleness_curves.csv"),
                 [{"mode": label, "update": i + 1, "return": y,
                   "lag": stale_detail[label]["lags"][i],
                   "mean_rho": stale_detail[label]["per_update_rho"][i],
                   "clip_frac": stale_detail[label]["per_update_clip_frac"][i]}
                  for label, ys in curves.items() for i, y in enumerate(ys)])
        plot_curves(args.outdir, curves,
                    f"On-policy vs stale async ({args.env}, delay={args.delay})",
                    "staleness_curves.png")
        stale_table = []
        for mode in args.modes.split(","):
            delay = 0 if mode == "sync" else args.delay
            res = stale_detail[f"{mode}(delay={delay})"]
            stale_table.append({"mode": mode, "delay": delay,
                                "final_return": res["final_return"],
                                "mean_lag": res["mean_lag"],
                                "max_lag": res["max_lag"],
                                "mean_rho": res["mean_rho"],
                                "mean_clip_frac": res["mean_clip_frac"]})
        print_table("Staleness: sync baseline vs async corrections", stale_table,
                    ["mode", "delay", "final_return", "mean_lag", "max_lag",
                     "mean_rho", "mean_clip_frac"])
        summary["staleness"] = stale_table

    if args.experiment in ("sweep", "all"):
        sweep = []
        for d in [int(x) for x in args.delays.split(",")]:
            for mode in args.sweep_modes.split(","):
                res = simulate_training(args.env, args.sweep_updates,
                                        args.sweep_steps, mode,
                                        args.broadcast_every, d, args.seed,
                                        device, args.eval_episodes,
                                        hidden=args.hidden, lr=args.lr,
                                        epochs=args.epochs,
                                        clip_eps=args.clip_eps, c_bar=args.c_bar,
                                        rho_bar=args.rho_bar)
                sweep.append({"mode": mode, "delay": d,
                              "final_return": res["final_return"],
                              "mean_lag": res["mean_lag"],
                              "max_lag": res["max_lag"],
                              "mean_rho": res["mean_rho"],
                              "mean_clip_frac": res["mean_clip_frac"]})
        save_csv(os.path.join(args.outdir, "staleness_sweep.csv"), sweep)
        plot_sweep(args.outdir, sweep)
        print_table("Staleness sweep (final greedy return)", sweep,
                    ["mode", "delay", "final_return", "mean_lag",
                     "mean_clip_frac"])
        summary["sweep"] = sweep

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWrote CSV/JSON metrics and PNG plots to {args.outdir}/")
    return summary


if __name__ == "__main__":
    main()
