"""Chapter 26 lab, part 3: tabular self-play TD learning on tic-tac-toe.

A Q-learner plays X against a mixture of a random opponent and a
self-play opponent (O picks the move that minimizes X's own Q-values,
epsilon-greedy). Learning is plain tabular Q-learning from the X
perspective: +1 win, 0 draw, -1 loss, gamma = 1. Progress is measured
against two fixed yardsticks: a random player and an optimal minimax
player (a perfect tic-tac-toe player always draws against another
perfect player, so "draw vs minimax" is the ceiling for X).

The RL ideas: self-play supplies an adaptive curriculum (the opponent
improves as X improves), while periodic evaluation against fixed
opponents detects whether the policy really improved or just overfit to
its own quirks -- the same role the opponent pool and Elo tracking play
in Section 26.1.

How to run:
    .venv/bin/python code/chapter26_selfplay_frontiers/tictactoe_self_play.py --smoke
    .venv/bin/python code/chapter26_selfplay_frontiers/tictactoe_self_play.py \
        --episodes 30000 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from device_utils import resolve_torch_device

EMPTY, X, O = 0, 1, 2
LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)
OUTCOME_LABEL = {1.0: "X wins", 0.0: "draw", -1.0: "O wins"}


def winner(board: tuple[int, ...]) -> int | None:
    """Return X (1) / O (2) on a win, 0 on a full-board draw, else None."""
    for a, b, c in LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    if EMPTY not in board:
        return 0
    return None


def legal_moves(board: tuple[int, ...]) -> list[int]:
    return [i for i, v in enumerate(board) if v == EMPTY]


def encode(board: tuple[int, ...]) -> int:
    index, base = 0, 1
    for v in board:
        index += v * base
        base *= 3
    return index


@lru_cache(maxsize=None)
def minimax_value(board: tuple[int, ...], to_move: int) -> float:
    """Optimal game result from X's perspective (+1/0/-1) with perfect play."""
    result = winner(board)
    if result is not None:
        return 1.0 if result == X else (-1.0 if result == O else 0.0)
    values = []
    for move in legal_moves(board):
        cells = list(board)
        cells[move] = to_move
        values.append(minimax_value(tuple(cells), O if to_move == X else X))
    return max(values) if to_move == X else min(values)


def minimax_move(board: tuple[int, ...], to_move: int) -> int:
    """Optimal move for the player to move (ties broken by lowest index)."""
    best_move, best_value = -1, None
    for move in legal_moves(board):
        cells = list(board)
        cells[move] = to_move
        value = minimax_value(tuple(cells), O if to_move == X else X)
        if best_value is None or (
            value > best_value if to_move == X else value < best_value
        ):
            best_move, best_value = move, value
    return best_move


class TabularQLearner:
    """Epsilon-greedy tabular Q-learner for X with the table on a torch device."""

    def __init__(self, alpha: float, device: str) -> None:
        import torch

        self.alpha = alpha
        self.device = device
        self.table = torch.zeros((3**9, 9), dtype=torch.float32, device=device)

    def values(self, board: tuple[int, ...]) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.table[encode(board)].detach().cpu().numpy().copy()

    def greedy(self, board: tuple[int, ...], player: int) -> int:
        """Greedy move: X maximizes, O (self-play rival) minimizes X's values."""
        q = self.values(board)
        masked = np.where(
            np.array([cell == EMPTY for cell in board]), q,
            np.inf if player == O else -np.inf,
        )
        if player == X:
            return int(np.argmax(masked))
        return int(np.argmin(masked))

    def egreedy(
        self, board: tuple[int, ...], player: int, eps: float,
        rng: np.random.Generator,
    ) -> int:
        if rng.random() < eps:
            return int(rng.choice(legal_moves(board)))
        return self.greedy(board, player)

    def update(
        self, board: tuple[int, ...], action: int, reward: float,
        nxt: tuple[int, ...] | None,
    ) -> None:
        import torch

        s, target = encode(board), reward
        if nxt is not None:
            with torch.no_grad():
                q_next = self.table[encode(nxt)]
            mask = torch.tensor(
                [cell == EMPTY for cell in nxt],
                dtype=torch.bool, device=self.device,
            )
            best = torch.where(mask, q_next, torch.tensor(-torch.inf, device=self.device)).max()
            target = reward + float(best.item())
        self.table[s, action] += self.alpha * (target - self.table[s, action].item())


def play_game(
    learner: TabularQLearner,
    opponent: str,
    eps: float,
    rng: np.random.Generator,
    train: bool,
) -> float:
    """Play one game as X; return +1/0/-1. Updates X's table if training.

    Only X learns. The training target for X's move (s, a) uses the board
    after O's reply as the next state, so O's strength directly shapes
    what X learns to expect -- the self-play curriculum.
    """
    board: tuple[int, ...] = (EMPTY,) * 9
    while True:
        # X to move.
        state = board
        move = learner.egreedy(board, X, eps if train else 0.0, rng)
        cells = list(board)
        cells[move] = X
        board = tuple(cells)
        result = winner(board)
        if result is not None:
            reward = 1.0 if result == X else (0.0 if result == 0 else -1.0)
            if train:
                learner.update(state, move, reward, None)
            return reward
        # O to move.
        if opponent == "random":
            omove = int(rng.choice(legal_moves(board)))
        elif opponent == "self":
            omove = learner.egreedy(board, O, eps if train else 0.0, rng)
        elif opponent == "minimax":
            omove = minimax_move(board, O)
        else:
            raise ValueError(f"unknown opponent {opponent!r}")
        cells = list(board)
        cells[omove] = O
        board = tuple(cells)
        result = winner(board)
        if result is not None:
            reward = 1.0 if result == X else (0.0 if result == 0 else -1.0)
            if train:
                learner.update(state, move, reward, None)
            return reward
        if train:
            learner.update(state, move, 0.0, board)


def evaluate(
    learner: TabularQLearner, opponent: str, games: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    wins = draws = losses = 0
    for _ in range(games):
        outcome = play_game(learner, opponent, 0.0, rng, train=False)
        if outcome > 0:
            wins += 1
        elif outcome == 0:
            draws += 1
        else:
            losses += 1
    return {
        "win": wins / games, "draw": draws / games, "loss": losses / games,
        "score": (wins - losses) / games,
    }


def plot_progress(rows: list[dict], path: Path) -> None:
    episodes = [r["episode"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(episodes, [r["vs_random_win"] for r in rows], label="win vs random")
    axes[0].plot(episodes, [r["vs_random_draw"] for r in rows], label="draw vs random")
    axes[0].plot(episodes, [r["vs_random_loss"] for r in rows], label="loss vs random")
    axes[0].set_title("X (Q-learner) vs random O")
    axes[0].set_xlabel("training episodes")
    axes[0].set_ylabel("rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(episodes, [r["vs_minimax"] for r in rows], marker="o", markersize=3)
    axes[1].set_title("X (Q-learner) vs minimax O (+1 win / 0 draw / -1 loss)")
    axes[1].set_xlabel("training episodes")
    axes[1].set_ylabel("game outcome")
    axes[1].set_yticks([-1, 0, 1])
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Tic-tac-toe: tabular self-play TD")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tic-tac-toe self-play TD lab")
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--eps-start", type=float, default=0.3)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--opp-random", type=float, default=0.3,
                        help="fraction of training games vs random (rest self-play)")
    parser.add_argument("--eval-every", type=int, default=1500)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args(argv)


def default_outdir() -> Path:
    return Path(__file__).resolve().parents[2] / "runs" / "chapter26_selfplay_frontiers" / "tictactoe"


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.smoke:
        args.episodes = 400
        args.eval_every = 200
        args.eval_games = 20
    rng = np.random.default_rng(args.seed)
    import torch

    torch.manual_seed(args.seed)
    device = resolve_torch_device(args.device)
    outdir = Path(args.outdir) if args.outdir else default_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    learner = TabularQLearner(alpha=args.alpha, device=device)

    rows: list[dict] = []
    for episode in range(1, args.episodes + 1):
        frac = episode / args.episodes
        eps = args.eps_start + frac * (args.eps_end - args.eps_start)
        opponent = "random" if rng.random() < args.opp_random else "self"
        play_game(learner, opponent, eps, rng, train=True)
        if episode % args.eval_every == 0 or episode == args.episodes:
            rand = evaluate(learner, "random", args.eval_games, args.seed + episode)
            mini = evaluate(learner, "minimax", 1, args.seed)
            rows.append({
                "episode": episode, "eps": eps,
                "vs_random_win": rand["win"], "vs_random_draw": rand["draw"],
                "vs_random_loss": rand["loss"], "vs_minimax": mini["score"],
            })

    with open(outdir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_progress(rows, outdir / "tictactoe_curves.png")
    final = rows[-1]
    result = {
        "episodes": args.episodes, "seed": args.seed,
        "vs_random": {k: final[f"vs_random_{k}"] for k in ("win", "draw", "loss")},
        "vs_minimax": final["vs_minimax"],
        "history": rows,
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Tic-tac-toe tabular self-play TD (Q-learning X vs mix of random/self-play O)")
    print(f"  episodes: {args.episodes}, device {device}, seed {args.seed}")
    print(f"  {'episode':>8}  {'win(baseline rnd)':>17}  {'draw':>6}  {'loss':>6}  {'vs minimax':>10}")
    for r in rows:
        print(f"  {r['episode']:>8}  {r['vs_random_win']:>17.3f}  {r['vs_random_draw']:>6.3f}"
              f"  {r['vs_random_loss']:>6.3f}  {OUTCOME_LABEL[float(r['vs_minimax'])]:>10}")
    print(f"outputs written to {outdir}")
    return result


if __name__ == "__main__":
    main()
