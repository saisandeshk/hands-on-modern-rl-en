"""Run the Chapter 16 lab: train a tiny reasoner, then scale test-time compute.

Stages:
  1. Train a character-level transformer on addition carry traces (§16.1-16.2:
     the base model must already solve the task some of the time, or repeated
     sampling has nothing to amplify).
  2. Single-sample accuracy at several temperatures.
  3. Test-time scaling at fixed temperature: pass@k (unbiased estimator),
     majority vote (self-consistency) and best-of-N with an exact verifier,
     for N = 1..64, plus the §16.3 coverage formula 1-(1-p)^N for comparison.
  4. Best-of-N with noisy verifiers (false positives at a set rate).
  5. One rejection-sampling fine-tuning round (train on self-generated,
     verifier-accepted traces) and its effect on single-sample accuracy.

Outputs (under ``runs/chapter16_reasoning/``): ``metrics.json``,
``scaling.csv``, ``temperature.csv``, ``scaling_curves.png`` and
``temperature.png``. A short summary table is printed to stdout.

Examples:
  python run_lab.py --smoke --device cpu          # <60 s sanity check
  python run_lab.py --device auto                 # full lab (~minutes on CPU)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arith_cot import (
    EOS_ID,
    TinyReasoner,
    TrainExample,
    best_of_n_noisy,
    build_examples,
    coverage_at_n,
    decode,
    encode,
    make_splits,
    majority_answer,
    parse_answer,
    pass_at_k,
    render,
    set_seed,
    train_reasoner,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO_ROOT / "runs" / "chapter16_reasoning"
N_GRID = [1, 2, 4, 8, 16, 32, 64]
T_SCALE = 0.7  # sampling temperature used for the scaling pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(choice: str) -> torch.device:
    """Resolve --device {auto,cpu,cuda} to a torch device."""
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(choice)


def max_completion_len() -> int:
    """Longest possible trace over all two-digit pairs (bounds generation)."""
    longest = 0
    for a in range(10, 100):
        for b in range(10, 100):
            _, comp, _ = render(a, b)
            longest = max(longest, len(comp))
    return longest


@torch.no_grad()
def sample_completions(
    model: TinyReasoner,
    prompt_ids_list: list[list[int]],
    n_samples: int,
    max_new: int,
    temperature: float,
    seed: int,
    batch_rows: int,
    device: torch.device,
) -> list[list[str]]:
    """Draw ``n_samples`` completions per prompt; return decoded strings."""
    model.eval()
    max_prompt = max(len(p) for p in prompt_ids_list)
    rows: list[list[int]] = []
    owner: list[int] = []
    for i, p in enumerate(prompt_ids_list):
        pad = [0] * (max_prompt - len(p))
        for _ in range(n_samples):
            rows.append(pad + p)
            owner.append(i)
    gen = torch.Generator().manual_seed(seed)
    texts: list[list[str]] = [[] for _ in prompt_ids_list]  # type: ignore
    for start in range(0, len(rows), batch_rows):
        chunk = torch.tensor(rows[start : start + batch_rows], dtype=torch.long)
        lens = torch.tensor(
            [len(prompt_ids_list[o]) for o in owner[start : start + batch_rows]]
        )
        full = model.generate(
            chunk.to(device),
            lens.to(device),
            max_new=max_new,
            temperature=temperature,
            generator=gen,
        ).cpu()
        for r, o in enumerate(owner[start : start + batch_rows]):
            comp_ids = full[r, max_prompt : max_prompt + max_new]
            texts[o].append(decode(comp_ids))
    return texts


def eval_single_sample(
    model: TinyReasoner,
    test_prompts: list[list[int]],
    test_truth: list[int],
    temperature: float,
    max_new: int,
    seed: int,
    batch_rows: int,
    device: torch.device,
) -> float:
    """One completion per problem; fraction whose parsed answer is exact."""
    texts = sample_completions(
        model, test_prompts, 1, max_new, temperature, seed, batch_rows, device
    )
    hits = sum(parse_answer(t[0]) == y for t, y in zip(texts, test_truth))
    return hits / len(test_truth)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    """Execute every stage; return the metrics dict (also written to disk)."""
    t_start = time.time()
    set_seed(args.seed)
    device = resolve_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    temps = [float(t) for t in args.temps.split(",")]
    fp_rates = [float(f) for f in args.fp_rates.split(",")]
    n_grid = [n for n in N_GRID if n <= args.n_samples]

    print(f"[lab] seed={args.seed} device={device} outdir={outdir}", flush=True)

    # -- Stage 0: data ------------------------------------------------------
    train_pairs, test_pairs = make_splits(args.n_train, args.n_test, args.seed)
    train_examples = build_examples(train_pairs)
    test_prompts = [encode(f"{a}+{b}=") for a, b in test_pairs]
    test_truth = [a + b for a, b in test_pairs]
    max_new = max_completion_len() + 1  # +1 for the <eos> stop token
    print(
        f"[lab] train={len(train_pairs)} test={len(test_pairs)} "
        f"gen_len={max_new} vocab=16",
        flush=True,
    )

    # -- Stage 1: train the base reasoner -----------------------------------
    model = TinyReasoner().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[lab] model params={n_params} epochs={args.epochs}", flush=True)
    t0 = time.time()
    losses = train_reasoner(
        model,
        train_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
        log_every=max(1, args.epochs // 4),
    )
    print(f"[lab] training done in {time.time() - t0:.1f}s", flush=True)

    # -- Stage 2: single-sample accuracy vs temperature ----------------------
    print("[lab] stage 2: temperature study", flush=True)
    temp_rows = []
    for ti, temp in enumerate(temps):
        acc = eval_single_sample(
            model, test_prompts, test_truth, temp, max_new,
            args.seed + 100 + ti, args.batch_rows, device,
        )
        temp_rows.append({"temperature": temp, "accuracy": acc})
        print(f"    T={temp:<5} single-sample accuracy={acc:.3f}", flush=True)

    # -- Stage 3: scaling pool (n samples per problem at T_SCALE) ------------
    print(f"[lab] stage 3: drawing pool of {args.n_samples} @T={T_SCALE}", flush=True)
    t0 = time.time()
    pool_texts = sample_completions(
        model, test_prompts, args.n_samples, max_new,
        T_SCALE, args.seed + 777, args.batch_rows, device,
    )
    print(f"[lab] pool drawn in {time.time() - t0:.1f}s", flush=True)
    pool_answers = [[parse_answer(t) for t in texts] for texts in pool_texts]
    pool_flags = [
        [(a == y) for a in ans] for ans, y in zip(pool_answers, test_truth)
    ]
    p_pool = float(np.mean([f[0] for f in pool_flags])) if pool_flags else 0.0

    # Noisy-verifier acceptance matrices (fixed per fp rate for all N).
    vrng = np.random.default_rng(args.seed + 999)
    accept = {}
    for fp in fp_rates:
        mat = []
        for flags in pool_flags:
            mat.append([f or (vrng.random() < fp) for f in flags])
        accept[fp] = mat

    scale_rows = []
    for n in n_grid:
        row: dict = {"N": n}
        row["pass_at_k"] = float(
            np.mean([pass_at_k(args.n_samples, sum(f), n) for f in pool_flags])
        )
        maj = [
            majority_answer(ans[:n]) == y
            for ans, y in zip(pool_answers, test_truth)
        ]
        row["majority"] = float(np.mean(maj))
        row["best_of_n_exact"] = float(
            np.mean([coverage_at_n(f, n) for f in pool_flags])
        )
        for fp in fp_rates:
            picks = [
                best_of_n_noisy(f, acc[:n] if len(acc) >= n else acc, n)
                for f, acc in zip(pool_flags, accept[fp])
            ]
            row[f"best_of_n_noisy_fp{fp:g}"] = float(np.mean(picks))
        row["coverage_theory"] = 1.0 - (1.0 - p_pool) ** n
        scale_rows.append(row)
        detail = " ".join(
            f"{k}={v:.3f}" for k, v in row.items() if k != "N"
        )
        print(f"    N={n:<3} {detail}", flush=True)

    # -- Stage 4: rejection-sampling fine-tuning (optional) --------------------
    rft_info: dict = {"done": False}
    greedy_before = eval_single_sample(
        model, test_prompts, test_truth, 0.0, max_new,
        args.seed + 555, args.batch_rows, device,
    )
    if not args.no_rft:
        print("[lab] stage 4: rejection-sampling fine-tuning", flush=True)
        rft_prompts = [encode(f"{a}+{b}=") for a, b in train_pairs]
        rft_texts = sample_completions(
            model, rft_prompts, args.rft_samples, max_new,
            0.8, args.seed + 31337, args.batch_rows, device,
        )
        kept: list[TrainExample] = []
        for (a, b), texts in zip(train_pairs, rft_texts):
            for t in texts:
                if parse_answer(t) == a + b and len(kept) < args.rft_cap:
                    kept.append(
                        TrainExample(
                            prompt_ids=encode(f"{a}+{b}="),
                            completion_ids=encode(t) + [EOS_ID],
                            answer=a + b,
                        )
                    )
        kept = [k for k in kept if len(k.completion_ids) > 0]
        keep_rate = len(kept) / max(1, len(train_pairs) * args.rft_samples)
        print(f"    kept {len(kept)} verified traces (rate={keep_rate:.3f})", flush=True)
        if len(kept) >= 16:
            # Mix self-generated traces 1:1 with original teacher traces.
            # Pure self-training skews toward problems the model already
            # solves and makes it forget the rest; the teacher mix anchors
            # the original distribution (standard RFT practice).
            rft_lr = args.rft_lr if args.rft_lr > 0 else 0.2 * args.lr
            rft_rng = random.Random(args.seed + 2)
            n_mix = min(len(kept), len(train_examples))
            mixed = kept + rft_rng.sample(train_examples, n_mix)
            rft_rng.shuffle(mixed)
            print(f"    finetune on {len(mixed)} traces "
                  f"({len(kept)} self + {n_mix} teacher), lr={rft_lr:g}",
                  flush=True)
            train_reasoner(
                model, mixed, epochs=args.rft_epochs,
                batch_size=args.batch_size, lr=rft_lr,
                device=device, seed=args.seed + 1,
            )
            greedy_after = eval_single_sample(
                model, test_prompts, test_truth, 0.0, max_new,
                args.seed + 555, args.batch_rows, device,
            )
            t07_after = eval_single_sample(
                model, test_prompts, test_truth, 0.7, max_new,
                args.seed + 556, args.batch_rows, device,
            )
            rft_info = {
                "done": True,
                "kept": len(kept),
                "keep_rate": keep_rate,
                "greedy_before": greedy_before,
                "greedy_after": greedy_after,
                "t07_after": t07_after,
            }
            print(
                f"    greedy {greedy_before:.3f} -> {greedy_after:.3f}; "
                f"T=0.7 after={t07_after:.3f}",
                flush=True,
            )
        else:
            rft_info = {"done": False, "kept": len(kept), "keep_rate": keep_rate,
                        "greedy_before": greedy_before, "reason": "too few traces"}
            print("    too few verified traces; skipping fine-tuning", flush=True)
    else:
        rft_info = {"done": False, "greedy_before": greedy_before}

    # -- Stage 5: write outputs ----------------------------------------------
    metrics = {
        "config": {
            "seed": args.seed, "device": str(device),
            "n_train": args.n_train, "n_test": args.n_test,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "n_samples": args.n_samples,
            "t_scale": T_SCALE, "temps": temps, "fp_rates": fp_rates,
            "n_params": n_params,
            "runtime_s": time.time() - t_start,
        },
        "train_loss": losses,
        "temperature": temp_rows,
        "pool_single_sample_rate": p_pool,
        "scaling": scale_rows,
        "rft": rft_info,
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(outdir / "temperature.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["temperature", "accuracy"])
        w.writeheader()
        w.writerows(temp_rows)
    scale_fields = list(scale_rows[0].keys())
    with open(outdir / "scaling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scale_fields)
        w.writeheader()
        w.writerows(scale_rows)
    plot_scaling(scale_rows, p_pool, outdir / "scaling_curves.png")
    plot_temperature(temp_rows, outdir / "temperature.png")
    print(f"[lab] wrote {outdir}/metrics.json scaling.csv temperature.csv *.png",
          flush=True)

    # -- Summary table ---------------------------------------------------------
    print("\n== single-sample accuracy vs temperature ==")
    for r in temp_rows:
        print(f"  T={r['temperature']:<5} acc={r['accuracy']:.3f}")
    print("\n== accuracy vs N (sampling temperature "
          f"{T_SCALE}, pool rate p={p_pool:.3f}) ==")
    header = ["N"] + [k for k in scale_rows[0] if k != "N"]
    print("  " + "  ".join(f"{h:>22}" for h in header))
    for r in scale_rows:
        print("  " + "  ".join(
            f"{r[h]:>22.3f}" if h != "N" else f"{r[h]:>22d}" for h in header
        ))
    if rft_info.get("done"):
        print("\n== rejection-sampling fine-tuning ==")
        print(f"  kept={rft_info['kept']} "
              f"greedy {rft_info['greedy_before']:.3f} -> "
              f"{rft_info['greedy_after']:.3f}")
    print(f"\n[lab] total runtime {metrics['config']['runtime_s']:.1f}s")
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_scaling(rows: list[dict], p_pool: float, path: Path) -> None:
    """Accuracy-vs-N curves for every aggregation rule."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = [r["N"] for r in rows]
    ax.plot(xs, [r["pass_at_k"] for r in rows], "o-",
            label="pass@k (unbiased est.)")
    ax.plot(xs, [r["majority"] for r in rows], "s-",
            label="majority vote")
    ax.plot(xs, [r["best_of_n_exact"] for r in rows], "^-",
            label="best-of-N (exact verifier)")
    for key in sorted(k for k in rows[0] if k.startswith("best_of_n_noisy")):
        ax.plot(xs, [r[key] for r in rows], "x--",
                label=f"best-of-N ({key.split('_fp')[1]} FP rate)")
    ax.plot(xs, [r["coverage_theory"] for r in rows], ":",
            color="gray", label=f"theory 1-(1-p)^N, p={p_pool:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("N (candidates per problem)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Test-time scaling on carry-trace addition")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_temperature(rows: list[dict], path: Path) -> None:
    """Single-sample accuracy at each sampling temperature."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot([r["temperature"] for r in rows],
            [r["accuracy"] for r in rows], "o-")
    ax.set_xlabel("sampling temperature")
    ax.set_ylabel("single-sample test accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Sampling temperature vs accuracy (one sample)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Chapter 16 test-time-compute lab")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny config finishing in under ~60 s on CPU")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--batch-rows", type=int, default=2048,
                    help="sampling batch size (rows = problems x samples)")
    ap.add_argument("--temps", type=str, default="0.0,0.3,0.7,1.0,1.3")
    ap.add_argument("--fp-rates", type=str, default="0.1,0.3")
    ap.add_argument("--rft-samples", type=int, default=8)
    ap.add_argument("--rft-epochs", type=int, default=2)
    ap.add_argument("--rft-cap", type=int, default=4000)
    ap.add_argument("--rft-lr", type=float, default=0.0,
                    help="RFT learning rate; 0 means 0.2 * --lr")
    ap.add_argument("--no-rft", action="store_true")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    return ap


def apply_smoke(args: argparse.Namespace) -> None:
    """Shrink every size knob so the full pipeline runs in <~60 s on CPU."""
    args.epochs = 2
    args.n_train = 256
    args.n_test = 24
    args.n_samples = 8
    args.batch_rows = 512
    args.temps = "0.0,0.7"
    args.fp_rates = "0.3"
    args.rft_samples = 4
    args.rft_epochs = 1
    args.rft_cap = 200


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        apply_smoke(args)
    run(args)


if __name__ == "__main__":
    main()
