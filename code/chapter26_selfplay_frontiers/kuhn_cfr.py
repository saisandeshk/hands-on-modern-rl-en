"""Chapter 26 lab, part 2: counterfactual regret minimization (CFR) in Kuhn poker.

Kuhn poker (3 cards J/Q/K, one ante each, one betting round) is the
smallest meaningful imperfect-information game: 12 information sets and
a known continuum of Nash equilibria. This script implements vanilla CFR
from scratch (regret matching + full-tree traversal, no sampling) and
tracks exploitability of the average strategy, which must approach 0.

The known equilibrium family (parameter alpha in [0, 1/3]) used here is:
  player 1: bet J with prob alpha, bet K with prob 3*alpha, never bet Q;
            after checking and facing a bet, call with J: 0, Q: alpha+1/3,
            K: 1.
  player 2: after a check, bet with J: 1/3, Q: 0, K: 1;
            facing a bet, call with J: 0, Q: 1/3, K: 1.
The Q-call probability alpha+1/3 (not 1/3) is what keeps player 2
indifferent between bluffing and checking with a Jack: conditional on
player 1 having checked, player 1 disproportionately holds a Queen
(because Kings often bet first), so the indifference point shifts up
from 1/3. Game value to player 1 is -1/18. These facts are re-derived in
the book's lab page; the unit test pins the alpha = 0.2 member to
exploitability ~0.

How to run:
    .venv/bin/python code/chapter26_selfplay_frontiers/kuhn_cfr.py --smoke
    .venv/bin/python code/chapter26_selfplay_frontiers/kuhn_cfr.py \
        --iterations 20000 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from device_utils import resolve_torch_device

CARDS = ("J", "Q", "K")
# History strings follow standard Kuhn notation: check "c", bet "b", fold
# "f", call "c" (the position in the string disambiguates check from call).
CHECK, BET, FOLD, CALL = "c", "b", "f", "c"


def acting_player(history: str) -> tuple[int, tuple[str, str]]:
    """Who acts at a non-terminal history and with which actions."""
    if history in ("",):
        return 0, (CHECK, BET)
    if history == "c":
        return 1, (CHECK, BET)
    if history == "b":
        return 1, (FOLD, CALL)
    if history == "cb":
        return 0, (FOLD, CALL)
    raise ValueError(f"terminal or unknown history {history!r}")


def is_terminal(history: str) -> bool:
    return history in ("cc", "bf", "bc", "cbf", "cbc")


def payoff(player1_card: int, player2_card: int, history: str) -> float:
    """Net winnings of player 1 (zero-sum). Antes are 1 chip each."""
    if history == "cc":  # both checked: showdown for the 2-chip pot
        return 1.0 if player1_card > player2_card else -1.0
    if history == "bf":  # player 2 folded to the bet: player 1 takes the pot
        return 1.0
    if history == "bc":  # player 2 called: showdown for the 4-chip pot
        return 2.0 if player1_card > player2_card else -2.0
    if history == "cbf":  # player 1 folded to the check-raise
        return -1.0
    if history == "cbc":  # player 1 called: showdown for the 4-chip pot
        return 2.0 if player1_card > player2_card else -2.0
    raise ValueError(f"non-terminal history {history!r}")


def all_infosets() -> list[tuple[int, int, str]]:
    infos: list[tuple[int, int, str]] = []
    for history in ("", "c", "b", "cb"):
        player, _ = acting_player(history)
        for card in range(3):
            infos.append((player, card, history))
    return infos


def equilibrium_profile(alpha: float) -> dict[tuple[int, int, str], np.ndarray]:
    """One member of Kuhn poker's Nash equilibrium family."""
    assert 0.0 <= alpha <= 1.0 / 3.0
    return {
        (0, 0, ""): np.array([1 - alpha, alpha]),  # P1, J: bluff with alpha
        (0, 1, ""): np.array([1.0, 0.0]),  # P1, Q: never open-bet
        (0, 2, ""): np.array([1 - 3 * alpha, 3 * alpha]),  # P1, K: value-bet 3*alpha
        (0, 0, "cb"): np.array([1.0, 0.0]),  # P1, J facing bet: always fold
        (0, 1, "cb"): np.array([2.0 / 3.0 - alpha, 1.0 / 3.0 + alpha]),  # P1, Q: call alpha+1/3
        (0, 2, "cb"): np.array([0.0, 1.0]),  # P1, K facing bet: always call
        (1, 0, "c"): np.array([2.0 / 3.0, 1.0 / 3.0]),  # P2, J after check: bluff 1/3
        (1, 1, "c"): np.array([1.0, 0.0]),  # P2, Q after check: always check
        (1, 2, "c"): np.array([0.0, 1.0]),  # P2, K after check: always bet
        (1, 0, "b"): np.array([1.0, 0.0]),  # P2, J facing bet: always fold
        (1, 1, "b"): np.array([2.0 / 3.0, 1.0 / 3.0]),  # P2, Q facing bet: call 1/3
        (1, 2, "b"): np.array([0.0, 1.0]),  # P2, K facing bet: always call
    }


class KuhnCFR:
    """Vanilla CFR with full-tree traversal and regret matching.

    Regret matching turns cumulative counterfactual regrets into the
    current strategy: actions with positive regret are played in
    proportion to that regret, otherwise the infoset plays uniformly.
    The average strategy (weighted by the player's own reach probability,
    the vanilla CFR convention) is what converges to equilibrium.
    """

    def __init__(self) -> None:
        self.regret: dict[tuple[int, int, str], np.ndarray] = {
            infoset: np.zeros(2) for infoset in all_infosets()
        }
        self.avg_sum: dict[tuple[int, int, str], np.ndarray] = {
            infoset: np.zeros(2) for infoset in all_infosets()
        }

    def current_strategy(self, infoset: tuple[int, int, str]) -> np.ndarray:
        positive = np.maximum(self.regret[infoset], 0.0)
        total = positive.sum()
        if total > 0:
            return positive / total
        _, actions = acting_player(infoset[2])
        return np.full(len(actions), 1.0 / len(actions))

    def average_strategy(self) -> dict[tuple[int, int, str], np.ndarray]:
        avg = {}
        for infoset, total in self.avg_sum.items():
            mass = total.sum()
            if mass > 0:
                avg[infoset] = total / mass
            else:
                _, actions = acting_player(infoset[2])
                avg[infoset] = np.full(len(actions), 1.0 / len(actions))
        return avg

    def _traverse(
        self, card1: int, card2: int, history: str, reach1: float, reach2: float
    ) -> float:
        """One CFR traversal; returns the player-1-perspective utility."""
        if is_terminal(history):
            return payoff(card1, card2, history)
        player, _ = acting_player(history)
        card = card1 if player == 0 else card2
        infoset = (player, card, history)
        strategy = self.current_strategy(infoset)
        own_reach, opp_reach = (reach1, reach2) if player == 0 else (reach2, reach1)
        self.avg_sum[infoset] += own_reach * strategy
        action_utils = np.zeros(2)
        for index, action in enumerate(acting_player(history)[1]):
            if player == 0:
                action_utils[index] = self._traverse(
                    card1, card2, history + action, reach1 * strategy[index], reach2
                )
            else:
                action_utils[index] = self._traverse(
                    card1, card2, history + action, reach1, reach2 * strategy[index]
                )
        node_util = float(strategy @ action_utils)
        # Regrets are measured in the acting player's own payoff. Utilities
        # above are player-1-centric, so flip the sign for player 2 (who
        # minimizes player 1's payoff).
        advantage = action_utils - node_util
        if player == 1:
            advantage = -advantage
        self.regret[infoset] += opp_reach * advantage
        return node_util

    def iterate(self) -> None:
        for card1, card2 in itertools.permutations(range(3), 2):
            self._traverse(card1, card2, "", 1.0, 1.0)


def expected_value(
    profile1: dict[tuple[int, int, str], np.ndarray],
    profile2: dict[tuple[int, int, str], np.ndarray],
) -> float:
    """Exact expected value to player 1 under two behavioral profiles."""

    def value(card1: int, card2: int, history: str) -> float:
        if is_terminal(history):
            return payoff(card1, card2, history)
        player, actions = acting_player(history)
        card = card1 if player == 0 else card2
        strategy = profile1[(player, card, history)] if player == 0 else profile2[(player, card, history)]
        return float(
            sum(
                strategy[index] * value(card1, card2, history + action)
                for index, action in enumerate(actions)
            )
        )

    total = sum(value(c1, c2, "") for c1, c2 in itertools.permutations(range(3), 2))
    return total / 6.0


def _pure_profiles(player: int) -> list[dict[tuple[int, int, str], np.ndarray]]:
    """All 64 deterministic strategies for one player (exact best response)."""
    own = [infoset for infoset in all_infosets() if infoset[0] == player]
    profiles = []
    for bits in itertools.product([0, 1], repeat=len(own)):
        profile = {}
        for infoset, bit in zip(own, bits):
            vector = np.zeros(2)
            vector[bit] = 1.0
            profile[infoset] = vector
        profiles.append(profile)
    return profiles


def exploitability(
    profile1: dict[tuple[int, int, str], np.ndarray],
    profile2: dict[tuple[int, int, str], np.ndarray],
) -> tuple[float, float, float]:
    """Exact (NashConv, exploitability, game value) from player 1's view.

    Best responses are computed by enumerating all 64 pure strategies per
    player. Exploitability (NashConv / 2) is 0 exactly at equilibrium.
    """
    value = expected_value(profile1, profile2)
    best1 = max(expected_value(pure, profile2) for pure in _pure_profiles(0))
    best2 = min(expected_value(profile1, pure) for pure in _pure_profiles(1))
    nash_conv = best1 - best2
    return nash_conv, nash_conv / 2.0, value


def torch_expected_value(
    profile1: dict[tuple[int, int, str], np.ndarray],
    profile2: dict[tuple[int, int, str], np.ndarray],
    device: str,
) -> float:
    """Recompute the game value with torch tensors (device cross-check)."""
    import torch

    dev = resolve_torch_device(device)
    t1 = {k: torch.as_tensor(v, dtype=torch.float64, device=dev) for k, v in profile1.items()}
    t2 = {k: torch.as_tensor(v, dtype=torch.float64, device=dev) for k, v in profile2.items()}

    def value(card1: int, card2: int, history: str) -> "torch.Tensor":
        import torch as _torch

        if is_terminal(history):
            return _torch.tensor(payoff(card1, card2, history), dtype=_torch.float64, device=dev)
        player, actions = acting_player(history)
        card = card1 if player == 0 else card2
        strategy = t1[(player, card, history)] if player == 0 else t2[(player, card, history)]
        total = _torch.zeros((), dtype=_torch.float64, device=dev)
        for index, action in enumerate(actions):
            total = total + strategy[index] * value(card1, card2, history + action)
        return total

    total = sum(value(c1, c2, "") for c1, c2 in itertools.permutations(range(3), 2))
    return float((total / 6.0).item())


def key_strategy_summary(
    avg: dict[tuple[int, int, str], np.ndarray],
) -> dict[str, float]:
    """Pull out the textbook frequencies: bluff J, value-bet K, calls with Q."""
    return {
        "p1_bet_J": float(avg[(0, 0, "")][1]),
        "p1_bet_Q": float(avg[(0, 1, "")][1]),
        "p1_bet_K": float(avg[(0, 2, "")][1]),
        "p1_call_Q": float(avg[(0, 1, "cb")][1]),
        "p2_bet_J": float(avg[(1, 0, "c")][1]),
        "p2_bet_K": float(avg[(1, 2, "c")][1]),
        "p2_call_Q": float(avg[(1, 1, "b")][1]),
    }


def plot_progress(
    iters: list[int], expls: list[float], snapshots: list[dict[str, float]], path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(iters, np.maximum(expls, 1e-12))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("CFR exploitability (average strategy)")
    axes[0].set_xlabel("iteration (log scale)")
    axes[0].set_ylabel("exploitability (log scale)")
    axes[0].grid(True, alpha=0.3)
    for key in ("p1_bet_J", "p1_bet_K", "p1_call_Q", "p2_bet_J", "p2_call_Q"):
        axes[1].plot(iters, [s[key] for s in snapshots], label=key)
    axes[1].axhline(1.0 / 3.0, color="black", linestyle="--", linewidth=1.0, label="1/3 ref")
    axes[1].set_xscale("log")
    axes[1].set_title("Learned key frequencies vs CFR iterations")
    axes[1].set_xlabel("iteration (log scale)")
    axes[1].set_ylabel("probability")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Kuhn poker: CFR from scratch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kuhn poker CFR lab")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args(argv)


def default_outdir() -> Path:
    return Path(__file__).resolve().parents[2] / "runs" / "chapter26_selfplay_frontiers" / "kuhn"


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.smoke:
        args.iterations = 200
        args.log_every = 50
    np.random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)
    device = resolve_torch_device(args.device)
    outdir = Path(args.outdir) if args.outdir else default_outdir()
    outdir.mkdir(parents=True, exist_ok=True)

    solver = KuhnCFR()
    iters: list[int] = []
    expls: list[float] = []
    snapshots: list[dict[str, float]] = []
    values: list[float] = []
    for step in range(args.iterations):
        solver.iterate()
        done = step + 1
        if done % args.log_every == 0 or done == args.iterations:
            avg = solver.average_strategy()
            _, expl, value = exploitability(avg, avg)
            iters.append(done)
            expls.append(expl)
            values.append(value)
            snapshots.append(key_strategy_summary(avg))

    avg = solver.average_strategy()
    nash_conv, expl, value = exploitability(avg, avg)
    torch_value = torch_expected_value(avg, avg, str(device))
    assert abs(torch_value - value) < 1e-9, "numpy/torch value mismatch"
    summary = key_strategy_summary(avg)
    alpha_hat = summary["p1_bet_J"]

    with open(outdir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "exploitability", "game_value", *sorted(summary)])
        for it, ex, va, snap in zip(iters, expls, values, snapshots):
            # recompute per-log snapshot already stored; write stored values row-wise
            writer.writerow([it, ex, va, *[snap[k] for k in sorted(summary)]])
    plot_progress(iters, expls, snapshots, outdir / "cfr_exploitability.png")
    with open(outdir / "average_strategy.json", "w") as f:
        json.dump(
            {f"p{c}_card{CARDS[c]}_hist_{h or 'root'}": v.tolist()
             for (p, c, h), v in avg.items()},
            f,
            indent=2,
        )

    print("Kuhn poker CFR (vanilla, full-tree traversal)")
    print(f"  iterations: {args.iterations}, device {device}, seed {args.seed}")
    print(f"  {'iter':>8}  {'exploitability':>15}  {'game value':>11}")
    for it, ex, va in zip(iters, expls, values):
        if it == iters[0] or it == iters[-1] or it in iters[len(iters) // 2 - 1 : len(iters) // 2 + 1]:
            print(f"  {it:>8}  {ex:>15.6f}  {va:>+11.6f}")
    print(f"  final exploitability {expl:.6f} (NashConv {nash_conv:.6f}), value {value:+.6f} (true -1/18 = {-1/18:+.6f})")
    print(f"  P1 bet J {summary['p1_bet_J']:.4f} (alpha), P1 bet K {summary['p1_bet_K']:.4f} "
          f"(ratio {summary['p1_bet_K'] / max(summary['p1_bet_J'], 1e-12):.3f}, equilibrium 3.0)")
    print(f"  P1 call Q {summary['p1_call_Q']:.4f} (equilibrium alpha+1/3 = {alpha_hat + 1/3:.4f}), "
          f"P2 bet J {summary['p2_bet_J']:.4f}, "
          f"P2 call Q {summary['p2_call_Q']:.4f} (equilibrium 1/3 each)")
    print(f"outputs written to {outdir}")
    return {
        "iterations": args.iterations,
        "seed": args.seed,
        "exploitability": expl,
        "nash_conv": nash_conv,
        "game_value": value,
        "alpha_hat": alpha_hat,
        "summary": summary,
        "history": {"iters": iters, "expls": expls, "values": values},
    }


if __name__ == "__main__":
    main()
