"""Smoke tests for Chapter 6 (policy gradients, REINFORCE, Actor-Critic)."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = [
    "reinforce_cartpole.py",
    "reinforce_with_baseline.py",
    "actor_critic_cartpole.py",
    "render_cartpole_baseline.py",
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
    assert proc.returncode == 0, (
        f"{script} --smoke exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr ---\n{proc.stderr[-3000:]}"
    )
    assert (here / "runs").exists(), f"{script} --smoke wrote nothing under runs/"
