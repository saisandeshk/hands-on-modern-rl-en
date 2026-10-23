"""Train a multi-turn research agent with REINFORCE + learned baseline.

Compares three reward regimes on the toy web (``web_env.py``):

* ``outcome``  - trajectory-level reward only: 1.0 for a correct final
  answer, 0.0 otherwise (minus a small per-step cost).  Sparse credit
  assignment, as in DeepResearcher / Search-R1 outcome supervision.
* ``shaped-relevant`` - outcome reward plus a bonus the first time the agent
  opens a question-relevant (gold-chain) document.  Dense process signal, as
  in Web-Shepherd / Atom-Searcher step rewards.  Note it uses privileged
  relevance labels, exactly the caveat Section 21.1 raises for PRMs.
* ``shaped-any`` - outcome reward plus a bonus for opening *any* new
  document.  A loose proxy that can be gamed: opening documents pays even
  when the final answer is wrong (Section 21.1 reward hacking).

All regimes start from the same behavior-cloned warm-start (heuristic demos;
the SFT cold-start of Section 21.1), so they differ only in the RL signal.
A from-scratch outcome run (``reinforce-scratch``) shows why the cold-start
matters: sparse terminal reward alone cannot discover a 5-step tool chain.

Each regime trains a small MLP policy from scratch (no RL libraries) and is
compared against a random policy and a scripted heuristic.  Metrics are split
by question difficulty (easy = 2-doc chain, hard = 3-doc disambiguation).

Outputs (under runs/chapter21_deep_research/): metrics.json, results.csv,
learning_curves.png, eval_success.png, plus a stdout summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_CODE_ROOT = _HERE.parents[0]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from device_utils import resolve_torch_device
from web_env import MAX_RESULTS, Corpus, Question, TinyResearchEnv, make_corpus
from policies import (
    ActorCriticMLP,
    HeuristicPolicy,
    RandomPolicy,
    build_features,
    discounted_returns,
    feature_dim,
)


def run_episode(env: TinyResearchEnv, policy_fn, gamma: float):
    """Roll out one episode; returns (features, actions, rewards, info)."""
    feats, actions, rewards = [], [], []
    done = False
    info: dict = {}
    while not done:
        f = build_features(env)
        a = policy_fn(env, f)
        feats.append(f)
        actions.append(a)
        _, r, done, info = env.step(a)
        rewards.append(r)
    returns = discounted_returns(rewards, gamma)
    return feats, actions, returns, info


def evaluate(env: TinyResearchEnv, questions: list[Question],
             policy_fn, seed: int = 0) -> dict:
    """Evaluate a policy; returns aggregate + per-difficulty metrics."""
    per_q = []
    for i, q in enumerate(questions):
        env.reset(q)
        done = False
        while not done:
            a = policy_fn(env, build_features(env))
            _, _, done, info = env.step(a)
        per_q.append({
            "qid": q.qid,
            "difficulty": q.difficulty,
            "correct": bool(info.get("correct", False)),
            "steps": env.steps,
            "searches": env.n_searches,
            "opens": env.n_opens,
        })
    out = {"n": len(per_q)}
    for diff in ("all", "easy", "hard"):
        rows = per_q if diff == "all" else [r for r in per_q if r["difficulty"] == diff]
        if not rows:
            continue
        out[diff] = {
            "success_rate": float(np.mean([r["correct"] for r in rows])),
            "avg_steps": float(np.mean([r["steps"] for r in rows])),
            "avg_searches": float(np.mean([r["searches"] for r in rows])),
            "avg_opens": float(np.mean([r["opens"] for r in rows])),
            "n": len(rows),
        }
    return out


def behavior_clone(corpus: Corpus, train_qs: list[Question],
                   args, device: torch.device) -> dict:
    """Supervised warm-start on heuristic demos (the SFT cold-start of Ch. 21).

    Returns the state dict used to initialize every REINFORCE variant, so the
    reward regimes differ only in their fine-tuning signal.
    """
    heur = HeuristicPolicy()
    F, A = [], []
    env = TinyResearchEnv(corpus, train_qs, max_steps=args.max_steps,
                          step_cost=args.step_cost, seed=args.seed)
    for q in train_qs:
        for _ in range(args.bc_repeats):
            env.reset(q)
            done = False
            while not done:
                f = build_features(env)
                a = heur.select_action(env)
                F.append(f)
                A.append(a)
                _, _, done, _ = env.step(a)
    obs_dim = feature_dim(len(corpus.candidates), MAX_RESULTS)
    act_dim = TinyResearchEnv.N_SEARCH + MAX_RESULTS + len(corpus.candidates)
    model = ActorCriticMLP(obs_dim, act_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    F = torch.as_tensor(np.stack(F), dtype=torch.float32, device=device)
    A = torch.as_tensor(A, dtype=torch.long, device=device)
    model.train()
    for _ in range(args.bc_epochs):
        logits, _ = model(F)
        loss = nn.functional.cross_entropy(logits, A)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(F)[0].argmax(dim=-1)
        acc = (pred == A).float().mean().item()
    print(f"[bc] {len(A)} demo steps, train accuracy={acc:.2f}")
    return model.state_dict()


def train_variant(corpus: Corpus, train_qs: list[Question], test_qs: list[Question],
                  shaping: str, args, device: torch.device,
                  init_state: dict | None = None) -> tuple:
    """REINFORCE with a learned state-value baseline for one reward regime."""
    rel_bonus = args.rel_bonus if shaping == "relevant" else 0.0
    any_bonus = args.any_bonus if shaping == "any" else 0.0

    def make_env(seed: int) -> TinyResearchEnv:
        return TinyResearchEnv(
            corpus, train_qs, max_steps=args.max_steps,
            step_cost=args.step_cost, rel_bonus=rel_bonus,
            any_bonus=any_bonus, seed=seed)

    obs_dim = feature_dim(len(corpus.candidates), MAX_RESULTS)
    act_dim = TinyResearchEnv.N_SEARCH + MAX_RESULTS + len(corpus.candidates)
    model = ActorCriticMLP(obs_dim, act_dim, hidden=args.hidden).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    eval_env = TinyResearchEnv(
        corpus, test_qs, max_steps=args.max_steps, step_cost=args.step_cost,
        rel_bonus=0.0, any_bonus=0.0, seed=999)

    shaping_ids = {"outcome": 11, "relevant": 22, "any": 33}
    rng = random.Random(args.seed * 1000 + shaping_ids[shaping])
    entropy_coef = 0.02
    value_coef = 0.5
    batch = args.batch
    curve_ep, curve_succ, curve_ret = [], [], []
    ep_count = 0
    buf_f, buf_a, buf_g, buf_v = [], [], [], []
    ret_window: list[float] = []

    model.train()
    while ep_count < args.episodes:
        env = make_env(rng.randint(0, 1_000_000))
        env.reset(rng.choice(train_qs))

        def sample_fn(e: TinyResearchEnv, f, _m=model, _r=rng):
            with torch.no_grad():
                t = torch.as_tensor(f, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = _m(t)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            return int(_r.choices(range(len(probs)), weights=probs)[0])

        feats, actions, returns, _ = run_episode(env, sample_fn, args.gamma)
        ret_window.append(float(returns[0]) if returns else 0.0)
        with torch.no_grad():
            t = torch.as_tensor(np.stack(feats), dtype=torch.float32, device=device)
            _, vals = model(t)
        buf_f += feats
        buf_a += actions
        buf_g += returns
        buf_v += vals.cpu().tolist()
        ep_count += 1

        if len(buf_g) >= batch * 2 or ep_count >= args.episodes:
            # Batch update: advantage = return - learned baseline.
            F = torch.as_tensor(np.stack(buf_f), dtype=torch.float32, device=device)
            A = torch.as_tensor(buf_a, dtype=torch.long, device=device)
            G = torch.as_tensor(buf_g, dtype=torch.float32, device=device)
            V = torch.as_tensor(buf_v, dtype=torch.float32, device=device)
            adv = G - V
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            logits, values = model(F)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(A)
            policy_loss = -(logp * adv.detach()).mean()
            value_loss = ((values - G.detach()) ** 2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            mean_ret = float(G.mean())
            buf_f, buf_a, buf_g, buf_v = [], [], [], []
        else:
            mean_ret = float(np.mean(returns))

        if ep_count % args.eval_every == 0 or ep_count == args.episodes:
            def greedy_fn(e, f, _m=model):
                with torch.no_grad():
                    t = torch.as_tensor(f, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                    logits, _ = _m(t)
                return int(torch.argmax(logits, dim=-1))

            m = evaluate(eval_env, test_qs, greedy_fn)
            curve_ep.append(ep_count)
            curve_succ.append(m["all"]["success_rate"])
            curve_ret.append(float(np.mean(ret_window[-args.eval_every:])))
            print(f"    [{shaping}] ep={ep_count} "
                  f"test_success={m['all']['success_rate']:.2f} "
                  f"mean_train_return={curve_ret[-1]:+.3f}")

    def greedy_fn(e, f, _m=model):
        with torch.no_grad():
            t = torch.as_tensor(f, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = _m(t)
        return int(torch.argmax(logits, dim=-1))

    final = evaluate(eval_env, test_qs, greedy_fn)
    return model, final, (curve_ep, curve_succ, curve_ret)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Multi-turn research agent lab (Ch. 21)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--smoke", action="store_true",
                   help="tiny configuration finishing in <60 s on CPU")
    p.add_argument("--episodes", type=int, default=4000,
                   help="training episodes per reward regime")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--step-cost", type=float, default=0.02)
    p.add_argument("--rel-bonus", type=float, default=0.2)
    p.add_argument("--any-bonus", type=float, default=0.4)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--bc-epochs", type=int, default=300,
                   help="supervised warm-start epochs on heuristic demos")
    p.add_argument("--bc-repeats", type=int, default=2,
                   help="heuristic demo rollouts per train question")
    p.add_argument("--outdir", type=str, default="",
                   help="output directory (default runs/chapter21_deep_research)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.smoke:
        args.episodes = 120
        args.eval_every = 60
        args.hidden = 32
    device = resolve_torch_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scale = "smoke" if args.smoke else "full"
    corpus = make_corpus(seed=args.seed, scale=scale)
    easy = [q for q in corpus.questions if q.difficulty == "easy"]
    hard = [q for q in corpus.questions if q.difficulty == "hard"]
    # Split by paper/question id (Section 21.2: disjoint train/test entities).
    train_qs = easy[: len(easy) * 3 // 4] + hard[: len(hard) * 3 // 4]
    test_qs = easy[len(easy) * 3 // 4:] + hard[len(hard) * 3 // 4:]
    random.Random(args.seed).shuffle(train_qs)

    print(f"Corpus: {len(corpus.docs)} docs "
          f"({sum(1 for d in corpus.docs.values() if d.kind == 'paper')} papers, "
          f"{sum(1 for d in corpus.docs.values() if d.kind == 'profile')} profiles, "
          f"{sum(1 for d in corpus.docs.values() if d.kind == 'lab')} labs), "
          f"{len(corpus.questions)} questions "
          f"({len(easy)} easy / {len(hard)} hard), "
          f"{len(corpus.candidates)} answer candidates.")
    print(f"Train: {len(train_qs)} questions, test: {len(test_qs)} questions; "
          f"episodes/regime: {args.episodes} (device: {device}).")

    t0 = time.time()
    results: dict = {}
    curves: dict = {}
    bc_state = behavior_clone(corpus, train_qs, args, device)
    for shaping in ("outcome", "relevant", "any"):
        _, final, (xs, ys, rs) = train_variant(corpus, train_qs, test_qs, shaping,
                                              args, device, init_state=bc_state)
        results[f"reinforce-{shaping}"] = final
        curves[shaping] = {"episodes": xs, "test_success": ys,
                           "mean_train_return": rs}
        print(f"[{shaping}] test success={final['all']['success_rate']:.2f} "
              f"avg_steps={final['all']['avg_steps']:.1f}")
    # From-scratch outcome REINFORCE: the sparse-reward control.
    _, final, _ = train_variant(corpus, train_qs, test_qs, "outcome",
                                args, device, init_state=None)
    results["reinforce-scratch"] = final
    print(f"[scratch] test success={final['all']['success_rate']:.2f} "
          f"avg_steps={final['all']['avg_steps']:.1f}")
    # From-scratch with relevant shaping: does dense process reward enable
    # discovery that sparse outcome reward cannot?
    _, final, (sxs, sys_, srs) = train_variant(corpus, train_qs, test_qs,
                                              "relevant", args, device,
                                              init_state=None)
    results["reinforce-scratch-relevant"] = final
    curves["scratch-relevant"] = {"episodes": sxs, "test_success": sys_,
                                  "mean_train_return": srs}
    print(f"[scratch-relevant] test success={final['all']['success_rate']:.2f} "
          f"avg_steps={final['all']['avg_steps']:.1f}")

    # Non-learned references.
    ref_env = TinyResearchEnv(corpus, test_qs, max_steps=args.max_steps,
                              step_cost=args.step_cost, seed=123)
    rand_policy = RandomPolicy(args.seed)
    results["random"] = evaluate(ref_env, test_qs, rand_policy.select_action)
    results["heuristic"] = evaluate(ref_env, test_qs,
                                    lambda e, f: HeuristicPolicy().select_action(e))
    for name in ("random", "heuristic"):
        m = results[name]
        print(f"[{name}] test success={m['all']['success_rate']:.2f} "
              f"avg_steps={m['all']['avg_steps']:.1f}")

    outdir = Path(args.outdir) if args.outdir else \
        _CODE_ROOT / "runs" / "chapter21_deep_research"
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "metrics.json", "w") as f:
        json.dump({"results": results, "curves": curves,
                   "config": vars(args)}, f, indent=2)
    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "split", "n", "success_rate", "avg_steps",
                    "avg_searches", "avg_opens"])
        for name, m in results.items():
            for split in ("all", "easy", "hard"):
                if split in m:
                    d = m[split]
                    w.writerow([name, split, d["n"],
                                f"{d['success_rate']:.4f}",
                                f"{d['avg_steps']:.3f}",
                                f"{d['avg_searches']:.3f}",
                                f"{d['avg_opens']:.3f}"])

    # -- plots ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for shaping, c in curves.items():
        ax.plot(c["episodes"], c["test_success"], marker="o", ms=3,
                label=f"REINFORCE ({shaping})")
    ax.set_xlabel("training episodes")
    ax.set_ylabel("held-out test success rate (greedy)")
    ax.set_title("Learning curves by reward regime")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "learning_curves.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    names = list(results.keys())
    short = [n.replace("reinforce-", "RL-") for n in names]
    for ai, split in enumerate(("easy", "hard")):
        succ = [results[n].get(split, {}).get("success_rate", 0.0) for n in names]
        axes[ai].bar(range(len(names)), succ)
        axes[ai].set_xticks(range(len(names)))
        axes[ai].set_xticklabels(short, rotation=25, ha="right", fontsize=8)
        axes[ai].set_ylim(0, 1.05)
        axes[ai].set_ylabel("success rate")
        axes[ai].set_title(f"{split} questions")
        axes[ai].grid(alpha=0.3, axis="y")
    fig.suptitle("Test success by policy and difficulty")
    fig.tight_layout()
    fig.savefig(outdir / "eval_success.png", dpi=130)
    plt.close(fig)

    # -- summary table --------------------------------------------------------
    print("\n=== Test results ===")
    print(f"{'policy':<20}{'split':<6}{'n':>3}  "
          f"{'success':>7}  {'steps':>5}  {'search':>6}  {'opens':>5}")
    for name, m in results.items():
        for split in ("all", "easy", "hard"):
            if split not in m:
                continue
            d = m[split]
            print(f"{name:<20}{split:<6}{d['n']:>3}  "
                  f"{d['success_rate']:>7.2f}  {d['avg_steps']:>5.1f}  "
                  f"{d['avg_searches']:>6.1f}  {d['avg_opens']:>5.1f}")
    print(f"\nFinished in {time.time() - t0:.1f} s. "
          f"Outputs in {outdir}")


if __name__ == "__main__":
    main()
