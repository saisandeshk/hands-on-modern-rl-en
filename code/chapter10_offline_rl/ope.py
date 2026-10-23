"""Step 3 of the Chapter 10 lab: off-policy evaluation (OPE) from fixed data.

Given a behaviour dataset collected by ``pi_b`` and a *target* policy
``pi_e`` (here: the frozen expert DQN, epsilon-smoothed so all ratios are
finite), estimate ``V^{pi_e}`` *without* running ``pi_e`` in the
environment, then compare against the true online value:

1. **Per-decision importance sampling (PDIS)** -- unbiased but high
   variance: each reward ``r_t`` is reweighted only by the ratios seen so
   far, ``prod_{k<=t} rho_k`` with ``rho = pi_e/pi_b``.
2. **Weighted (self-normalized) IS (WIS)** -- biased but far lower
   variance: ``sum_i w_i G_i / sum_i w_i`` with full-trajectory weights.
3. **Fitted Q evaluation (FQE)** -- fit ``Q^{pi_e}`` by regression on fixed
   batches (``y = r + gamma E_{a' ~ pi_e}[Q_target(s',a')]``, no ``max``
   over OOD actions, no importance ratios), then average over initial states.

Poor coverage (e.g. evaluating the expert from ``random`` data) makes IS
weights degenerate -- the script reports the effective sample size (ESS)
so you can see this directly.

Outputs (under ``runs/chapter10_offline_rl/``)
----------------------------------------------
- ``ope/ope_summary.json`` and ``ope/ope_summary.csv``.
- ``ope/ope_comparison.png``: estimate vs true value per behaviour dataset.

Example:
    python ope.py --seed 0 --device cpu
    python ope.py --smoke --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from offline_common import (
    QMLP,
    default_outdir,
    load_dataset,
    resolve_device,
    seed_everything,
)

TARGET_EPS = 0.05  # smoothing of the greedy target so ratios stay finite
GAMMA = 0.99


# ---------------------------------------------------------------------------
# Pure estimators (unit-tested in test_ch10_lab.py)
# ---------------------------------------------------------------------------


def wis_estimate(returns: np.ndarray, weights: np.ndarray) -> float:
    """Self-normalized importance sampling: ``sum(w*G)/sum(w)`."""
    returns = np.asarray(returns, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denom = weights.sum()
    if denom <= 0:
        return float("nan")
    return float((weights * returns).sum() / denom)


def pdis_estimate(
    episodes_rewards: list[np.ndarray],
    episodes_ratios: list[np.ndarray],
    gamma: float = GAMMA,
) -> float:
    """Per-decision IS: mean over episodes of ``sum_t gamma^t r_t prod_{k<=t} rho_k``."""
    vals = []
    for rew, rho in zip(episodes_rewards, episodes_ratios):
        cum, g = 1.0, 0.0
        for t, (r, rr) in enumerate(zip(rew, rho)):
            cum *= float(rr)
            g += (gamma ** t) * cum * float(r)
        vals.append(g)
    return float(np.mean(vals)) if vals else float("nan")


def full_weights(episodes_ratios: list[np.ndarray]) -> np.ndarray:
    """Full-trajectory importance weight per episode."""
    return np.array([float(np.prod(r)) for r in episodes_ratios])


def discounted_return(rewards: np.ndarray, gamma: float = GAMMA) -> float:
    disc = gamma ** np.arange(len(rewards))
    return float((np.asarray(rewards, dtype=np.float64) * disc).sum())


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def reconstruct_episodes(arrays: dict) -> list[np.ndarray]:
    """Split flat transitions into episodes using the ``ends``/``dones`` flags."""
    flags = arrays.get("ends", arrays["dones"])
    idx = np.where(np.asarray(flags) > 0.5)[0]
    eps, start = [], 0
    for e in idx:
        eps.append(np.arange(start, e + 1))
        start = int(e) + 1
    if start < len(arrays["rewards"]):
        eps.append(np.arange(start, len(arrays["rewards"])))
    return eps


def target_probs_for(states: np.ndarray, q_net: QMLP, device: torch.device) -> np.ndarray:
    """Epsilon-smoothed greedy target probs for a batch of states."""
    action_dim = q_net.net[-1].out_features
    with torch.no_grad():
        greedy = q_net(torch.as_tensor(states, dtype=torch.float32, device=device)).argmax(dim=1).cpu().numpy()
    probs = np.full((len(states), action_dim), TARGET_EPS / action_dim)
    probs[np.arange(len(states)), greedy] += 1.0 - TARGET_EPS
    return probs


# ---------------------------------------------------------------------------
# FQE
# ---------------------------------------------------------------------------


def train_fqe(
    arrays: dict,
    q_target_policy: QMLP,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    hidden: int = 64,
    lr: float = 1e-3,
    gamma: float = GAMMA,
) -> QMLP:
    """Fit Q^{pi_e} on fixed data: ``y = r + gamma (1-d) E_{a' ~ pi_e}[Q_bar(s',a')]``.

    Unlike Q-learning there is no ``max`` over actions -- the backup follows
    the smoothed target policy ``pi_e`` in expectation, so no OOD action is
    ever optimistically maximised. The target net is a periodically synced
    copy (FQE iteration).
    """
    rng = np.random.default_rng(seed + 331)
    torch.manual_seed(seed)
    q_target_policy = q_target_policy.to(device).eval()
    state_dim = arrays["states"].shape[1]
    action_dim = int(arrays["actions"].max()) + 1
    q = QMLP(state_dim, action_dim, hidden=hidden).to(device)
    q_bar = QMLP(state_dim, action_dim, hidden=hidden).to(device)
    q_bar.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    n = len(arrays["states"])
    for step in range(steps):
        idx = rng.integers(0, n, size=batch_size)
        s = torch.as_tensor(arrays["states"][idx], device=device)
        a = torch.as_tensor(arrays["actions"][idx], dtype=torch.int64, device=device)
        r = torch.as_tensor(arrays["rewards"][idx], device=device)
        ns = torch.as_tensor(arrays["next_states"][idx], device=device)
        d = torch.as_tensor(arrays["dones"][idx], device=device)
        with torch.no_grad():
            # Expected backup under the eps-smoothed target (matches
            # ``target_probs_for`` and ``true_target_value``): with prob
            # (1-eps) follow the greedy action, otherwise act uniformly.
            greedy_next = q_target_policy(ns).argmax(dim=1)
            q_bar_next = q_bar(ns)
            greedy_q = q_bar_next.gather(1, greedy_next.unsqueeze(1)).squeeze(1)
            mean_q = q_bar_next.mean(dim=1)
            exp_q = (1.0 - TARGET_EPS) * greedy_q + TARGET_EPS * mean_q
            y = r + gamma * (1.0 - d) * exp_q
        loss = nn.functional.mse_loss(q(s).gather(1, a.unsqueeze(1)).squeeze(1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            q_bar.load_state_dict(q.state_dict())
    return q


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OPE: PDIS / WIS / FQE on fixed datasets.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--behaviour", default="all", choices=["all", "random", "medium", "expert", "mixed"])
    p.add_argument("--fqe-steps", type=int, default=20000)
    p.add_argument("--true-episodes", type=int, default=50)
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def true_target_value(q_net: QMLP, device: torch.device, n_episodes: int, seed: int) -> dict:
    """Online Monte-Carlo value of the (smoothed) target policy: ground truth."""
    import gymnasium as gym

    q_net = q_net.to(device).eval()
    action_dim = q_net.net[-1].out_features
    rng = np.random.default_rng(seed + 999)
    vals = []
    for ep in range(n_episodes):
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=20_000 + seed * 100 + ep)
        total, t, done = 0.0, 0, False
        while not done:
            with torch.no_grad():
                s = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                greedy = int(q_net(s).argmax(dim=1).item())
            probs = np.full(action_dim, TARGET_EPS / action_dim)
            probs[greedy] += 1.0 - TARGET_EPS
            action = int(rng.choice(action_dim, p=probs))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += (GAMMA ** t) * float(reward)
            t += 1
            done = bool(terminated or truncated)
        vals.append(total)
        env.close()
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": n_episodes}


def run_ope_for(behaviour: str, outdir: str, args: argparse.Namespace,
                device: torch.device, q_target: QMLP) -> dict:
    arrays, _ = load_dataset(os.path.join(outdir, "datasets", f"{behaviour}.npz"))
    episodes = reconstruct_episodes(arrays)
    n_actions = q_target.net[-1].out_features

    ep_rews = [np.asarray(arrays["rewards"][e], dtype=np.float64) for e in episodes]
    b_probs = np.asarray(arrays["behaviour_probs"], dtype=np.float64)
    t_probs = target_probs_for(np.asarray(arrays["states"]), q_target, device)
    chosen_t = t_probs[np.arange(len(b_probs)), np.asarray(arrays["actions"])]
    chosen_b = b_probs[np.arange(len(b_probs)), np.asarray(arrays["actions"])]
    ratios = np.clip(chosen_t / np.clip(chosen_b, 1e-8, None), 0.0, 1e6)
    ep_ratios = [ratios[e] for e in episodes]

    disc_returns = np.array([discounted_return(r) for r in ep_rews])
    w = full_weights(ep_ratios)
    is_full = float(np.mean(w * disc_returns))  # ordinary full-trajectory IS
    pdis = pdis_estimate(ep_rews, ep_ratios)
    wis = wis_estimate(disc_returns, w)
    ess = float(w.sum() ** 2 / max(1e-12, (w ** 2).sum()))

    # FQE then average over the dataset episodes' initial states (every
    # CartPole episode starts from env.reset(), so this matches the reset
    # distribution). The value uses the smoothed-target expectation to match
    # the backup and the online truth.
    n_fqe = args.fqe_steps
    import hashlib as _hashlib

    _h = _hashlib.sha256(f"{args.seed}|{behaviour}|fqe".encode()).digest()
    fqe_seed = int.from_bytes(_h[:4], "little") % (2**31)
    q_fqe = train_fqe(arrays, q_target, n_fqe, 256, fqe_seed, device)
    s0 = np.array([arrays["states"][e[0]] for e in episodes if len(e) > 0])
    with torch.no_grad():
        s0_t = torch.as_tensor(s0, dtype=torch.float32, device=device)
        q0 = q_fqe(s0_t)
        greedy0 = q_target(s0_t).argmax(dim=1)
        greedy_q0 = q0.gather(1, greedy0.unsqueeze(1)).squeeze(1)
        mean_q0 = q0.mean(dim=1)
        exp_q0 = (1.0 - TARGET_EPS) * greedy_q0 + TARGET_EPS * mean_q0
        fqe_v = float(exp_q0.mean().item())

    _ = n_actions
    return {
        "behaviour": behaviour,
        "n_episodes": len(episodes),
        "is_full": is_full,
        "pdis": pdis,
        "wis": wis,
        "ess": ess,
        "ess_pct": 100.0 * ess / max(1, len(episodes)),
        "fqe": fqe_v,
        "dataset_mc": float(disc_returns.mean()),
    }


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.fqe_steps = 300
        args.true_episodes = 5
    seed_everything(args.seed)
    device = resolve_device(args.device)
    outdir = args.outdir or default_outdir()
    ope_dir = os.path.join(outdir, "ope")
    os.makedirs(ope_dir, exist_ok=True)
    t0 = time.time()

    # Target policy = frozen expert DQN from the collection step.
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    env.close()
    q_target = QMLP(state_dim, action_dim)
    q_target.load_state_dict(torch.load(os.path.join(outdir, "expert_q.pt"), map_location="cpu"))

    truth = true_target_value(q_target, device, args.true_episodes, args.seed)
    print(f"[ope] true target value (online, {args.true_episodes} eps): "
          f"{truth['mean']:.2f} +/- {truth['std']:.2f} (discounted, gamma={GAMMA})")

    behaviours = ["random", "medium", "expert", "mixed"] if args.behaviour == "all" else [args.behaviour]
    rows = []
    for b in behaviours:
        row = run_ope_for(b, outdir, args, device, q_target)
        rows.append(row)
        print(f"[ope] behaviour={b:6s} eps={row['n_episodes']:4d} "
              f"PDIS={row['pdis']:8.2f} WIS={row['wis']:8.2f} FQE={row['fqe']:8.2f} "
              f"ESS={row['ess']:.1f} ({row['ess_pct']:.1f}%) datasetMC={row['dataset_mc']:.2f}")

    with open(os.path.join(ope_dir, "ope_summary.json"), "w") as f:
        json.dump({"seed": args.seed, "gamma": GAMMA, "target_eps": TARGET_EPS,
                   "truth": truth, "rows": rows}, f, indent=2)
    with open(os.path.join(ope_dir, "ope_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["behaviour", "n_episodes", "truth", "pdis", "wis", "fqe",
                    "ess", "ess_pct", "dataset_mc"])
        for r in rows:
            w.writerow([r["behaviour"], r["n_episodes"], f"{truth['mean']:.3f}",
                        f"{r['pdis']:.3f}", f"{r['wis']:.3f}", f"{r['fqe']:.3f}",
                        f"{r['ess']:.2f}", f"{r['ess_pct']:.2f}", f"{r['dataset_mc']:.3f}"])

    labels = [r["behaviour"] for r in rows]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 1.5 * width, [truth["mean"]] * len(rows), width, label="true (online)")
    for i, key in enumerate(["pdis", "wis", "fqe", "dataset_mc"]):
        ax.bar(x + (i - 0.5) * width, [r[key] for r in rows], width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"discounted return (gamma={GAMMA})")
    ax.set_title("OPE of the expert target policy from each behaviour dataset")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(ope_dir, "ope_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"[ope] done in {time.time() - t0:.1f}s. Files under {ope_dir}/")


if __name__ == "__main__":
    main()
