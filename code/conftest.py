"""Shared pytest configuration for the classic-RL code.

Smoke tests must run on CPU: the 6 GB GPU in this environment is shared
with other jobs, so every training script is exercised with ``--smoke``,
which forces CPU, tiny budgets and output under ``runs/``.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONHASHSEED", "0")
