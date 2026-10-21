"""Smoke tests for Chapter 8 (PPO, GAE)."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = [
    "ppo_from_scratch.py",
    "ppo_lunar_lander.py",
    "ppo_bipedal_walker.py",
    "gae_visualization.py",
    "render_bipedal_walker.py",
]

TIMEOUT = 600


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
