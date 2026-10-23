"""Tests for the Chapter 26 self-play lab: exact game checks plus smoke runs.

Run with:
    .venv/bin/pytest code/chapter26_selfplay_frontiers -q
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))

import kuhn_cfr
import rps_self_play
import tictactoe_self_play as ttt

PYTHON = sys.executable


def run_script(name: str, outdir: Path, extra: list[str] | None = None) -> None:
    cmd = [PYTHON, str(DIR / name), "--smoke", "--seed", "0", "--device", "cpu",
           "--outdir", str(outdir)]
    if extra:
        cmd.extend(extra)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=110)
    assert proc.returncode == 0, f"{name} failed:\n{proc.stdout}\n{proc.stderr}"
    assert "outputs written to" in proc.stdout


def test_kuhn_terminal_payoffs() -> None:
    # (player-1 card, player-2 card, line) -> player-1 net winnings.
    cases = [
        ((2, 1, "cc"), 1.0),   # K vs Q, both check: K takes the 2-chip pot
        ((0, 2, "cc"), -1.0),  # J vs K, both check: J loses the ante
        ((2, 0, "bc"), 2.0),   # K vs J, call: K takes the 4-chip pot
        ((0, 2, "bc"), -2.0),  # J vs K, call: J loses bet + ante
        ((1, 0, "cbc"), 2.0),  # Q vs J after check-call: Q takes 4-chip pot
        ((0, 1, "cbc"), -2.0),  # J vs Q after check-call: J loses 2
        ((1, 2, "bf"), 1.0),   # anyone's bet folded to: bettor wins the ante
        ((0, 2, "cbf"), -1.0),  # check-then-fold: folder loses the ante
    ]
    for (c1, c2, history), expected in cases:
        assert kuhn_cfr.payoff(c1, c2, history) == expected


def test_known_equilibrium_has_zero_exploitability() -> None:
    profile = kuhn_cfr.equilibrium_profile(0.2)
    nash_conv, expl, value = kuhn_cfr.exploitability(profile, profile)
    assert expl < 1e-9, f"equilibrium must be unexploitable, got {expl}"
    assert abs(value - (-1 / 18)) < 1e-9, f"game value must be -1/18, got {value}"


def test_cfr_average_strategy_approaches_equilibrium() -> None:
    solver = kuhn_cfr.KuhnCFR()
    uniform = {
        infoset: np.full(2, 0.5) for infoset in kuhn_cfr.all_infosets()
    }
    _, initial_expl, _ = kuhn_cfr.exploitability(uniform, uniform)
    for _ in range(2000):
        solver.iterate()
    avg = solver.average_strategy()
    _, final_expl, value = kuhn_cfr.exploitability(avg, avg)
    assert final_expl < initial_expl
    assert final_expl < 0.02, f"CFR should nearly converge, got {final_expl}"
    assert abs(value - (-1 / 18)) < 0.01
    keys = kuhn_cfr.key_strategy_summary(avg)
    # Structural checks from the equilibrium family (alpha = P1's J bluff
    # frequency): K value bets 3x the J bluff, and the Q-call frequency is
    # alpha + 1/3 (the posterior after a check is Queen-heavy, shifting the
    # bluff-catching indifference point up from 1/3).
    assert abs(keys["p1_bet_K"] - 3 * keys["p1_bet_J"]) < 0.05
    assert abs(keys["p1_call_Q"] - (keys["p1_bet_J"] + 1 / 3)) < 0.05
    # Fast-converging coordinates sit at their equilibrium values.
    assert keys["p1_bet_Q"] < 0.05
    assert keys["p2_bet_K"] > 0.95
    assert abs(keys["p2_bet_J"] - 1 / 3) < 0.1
    assert abs(keys["p2_call_Q"] - 1 / 3) < 0.1


def test_rps_nash_profiles_have_zero_exploitability() -> None:
    for variant in ("standard", "biased"):
        utility = rps_self_play.payoff_matrix(variant)
        target = rps_self_play.nash_equilibrium(variant)
        assert rps_self_play.exploitability(target, target, utility) < 1e-12


def test_fictitious_play_averages_converge_to_nash() -> None:
    for variant in ("standard", "biased"):
        rng = np.random.default_rng(0)
        utility = rps_self_play.payoff_matrix(variant)
        traj = rps_self_play.run_fictitious(utility, 5000, rng)
        target = rps_self_play.nash_equilibrium(variant)
        for key in ("average1", "average2"):
            gap = np.abs(traj[key][-1] - target).sum()
            assert gap < 0.1, f"{variant} {key}: L1 gap {gap}"
        stats = rps_self_play.summarize(traj, utility, target)
        assert stats["expl_average"] < 0.05


def test_naive_best_response_cycles() -> None:
    # Pure best responses never settle: the last-iterate profile stays
    # fully exploitable on standard RPS.
    rng = np.random.default_rng(0)
    utility = rps_self_play.payoff_matrix("standard")
    traj = rps_self_play.run_best_response(utility, 300, rng)
    assert rps_self_play.exploitability(
        traj["current1"][-1], traj["current2"][-1], utility
    ) >= 1.0 - 1e-9


def test_tictactoe_minimax_is_optimal() -> None:
    assert ttt.minimax_value((ttt.EMPTY,) * 9, ttt.X) == 0.0
    assert ttt.minimax_value(
        (ttt.X, ttt.X, ttt.X, ttt.O, ttt.O, ttt.EMPTY, ttt.EMPTY, ttt.EMPTY, ttt.EMPTY),
        ttt.O,
    ) == 1.0
    # Perfect play from the empty board draws.
    board: tuple[int, ...] = (ttt.EMPTY,) * 9
    to_move = ttt.X
    while ttt.winner(board) is None:
        move = ttt.minimax_move(board, to_move)
        assert move in ttt.legal_moves(board)
        cells = list(board)
        cells[move] = to_move
        board = tuple(cells)
        to_move = ttt.O if to_move == ttt.X else ttt.X
    assert ttt.winner(board) == 0


def test_rps_script_smoke(tmp_path: Path) -> None:
    run_script("rps_self_play.py", tmp_path / "rps",
               ["--variant", "standard", "--dynamics", "all"])
    assert (tmp_path / "rps" / "summary_standard.json").exists()


def test_kuhn_script_smoke(tmp_path: Path) -> None:
    run_script("kuhn_cfr.py", tmp_path / "kuhn")
    assert (tmp_path / "kuhn" / "metrics.csv").exists()
    assert (tmp_path / "kuhn" / "average_strategy.json").exists()


def test_tictactoe_script_smoke(tmp_path: Path) -> None:
    run_script("tictactoe_self_play.py", tmp_path / "tictactoe")
    assert (tmp_path / "tictactoe" / "metrics.csv").exists()
