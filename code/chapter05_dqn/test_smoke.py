"""Smoke tests for Chapter 5 (DQN).

``dqn_atari_sb3.py --smoke`` validates Atari env construction and DQN
wiring only (no gradient training); if the Atari ROM/license is missing
it exits non-zero with a clear message and the test is skipped.
``dqn_pokemon_red_pyboy.py --smoke`` needs no ROM by design.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = [
    "dqn_cartpole.py",
    "double_dqn_cartpole.py",
    "dqn_gym_sb3.py",
    "dqn_atari_sb3.py",
    "dqn_pokemon_red_pyboy.py",
    "export_dqn_curves.py",
    "render_atari.py",
    "render_lunarlander.py",
    "render_lunarlander_split.py",
]

TIMEOUT = 300


@pytest.mark.parametrize("script", SCRIPTS)
def test_smoke(script):
    here = Path(__file__).resolve().parent
    proc = subprocess.run(
        [sys.executable, script, "--smoke"],
        cwd=here,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    combined = proc.stdout + proc.stderr
    if proc.returncode != 0 and (
        "ROM" in combined
        or "Atari" in combined
        or "ale-py" in combined
        or "ale_py" in combined
    ):
        pytest.skip(f"{script} --smoke skipped (Atari ROM/license unavailable)")
    assert proc.returncode == 0, (
        f"{script} --smoke exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr ---\n{proc.stderr[-3000:]}"
    )
