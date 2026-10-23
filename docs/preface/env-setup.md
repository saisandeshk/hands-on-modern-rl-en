---
title: Environment Setup
---

# Environment Setup

::: tip Ask a coding assistant to install it for you
If you are using a coding assistant such as Codex, tell it your learning goal first and then ask it to do the installation:

> Hello, I am about to start this reinforcement learning course and want to set up the lab environment first. Please check my operating system, Python version, and GPU/CUDA status, create a Python 3.12 virtualenv named `.venv` at the repository root with uv, install the pinned dependencies from `code/requirements.lock.txt`, and then run the CartPole verification program plus the smoke tests with `.venv/bin/pytest code -q`. If you hit installation problems, keep troubleshooting and fixing them. If you need administrator privileges or system-level configuration changes, ask for my confirmation first.
> :::

> **Goal of this section**: build a complete development environment for this course from scratch, covering Python, PyTorch, the RL toolchain, and LLM training frameworks. Follow the steps end to end, and you will be able to run the core experiments in the book (niche simulators and some chapter extras need additional installs described below).

This repository keeps one pinned core environment for the whole book in `code/requirements.lock.txt` (Python 3.12, Linux + CUDA resolution), a merged dependency summary in `code/requirements.txt`, and a minimal dependency list for each chapter that ships code in `code/chapter*/requirements.txt` (Chapters 1-3, 5-9, 12-15, 17, 19, 23, plus Appendix A). Install the lock file for the core stack on Linux with CUDA; on other platforms, or when you only want one chapter's dependencies, install the merged file or a single chapter file on top, since the lock file does not include every chapter extra (for example `matplotlib`, `swanlab`, `ale-py`, `opencv-python`, `modelscope`, `openai`; `scipy` appears in no requirements file).

## Quickstart Setup (Core Stack)

If you cannot wait to start coding, the following steps install the pinned core environment (torch 2.14.1 plus the CUDA 13 wheel stack — allow time for a multi-GB download), which covers the core experiments in Chapters 1-9 and the LLM post-training chapters. Chapter extras such as plotting (`matplotlib`), Atari (`ale-py`), logging (`swanlab`), and agentic/LLM helpers (`openai`, `modelscope`) come from `code/requirements.txt` or the per-chapter files. Niche simulators (Atari ROMs, Isaac Lab, Unity) can be installed later as needed.

::: warning The lock file is Linux + CUDA only
`code/requirements.lock.txt` pins Linux/CUDA wheels (`nvidia-*`, `cuda-toolkit`, `triton`), so it cannot be installed as-is on macOS or Windows, and it downloads the full CUDA stack even for CPU-only use. Readers on macOS, Windows (WSL2 excepted), or CPU-only machines should install the per-chapter file (or `code/requirements.txt`) with a CPU PyTorch wheel instead — see [Installing PyTorch](#installing-pytorch).
:::

::: code-group

```bash [uv (recommended)]
# From the repository root. Creates the project virtualenv with Python 3.12.
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r code/requirements.lock.txt
```

```bash [pip]
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r code/requirements.lock.txt
```

```bash [per-chapter (lightweight)]
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r code/chapter01_cartpole/requirements.txt
```

:::

After installation, verify with the following code, then run the repository smoke tests. If a `CartPole` window pops up, everything is ready:

```python
import gymnasium as gym
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
print(f"MPS:     {torch.backends.mps.is_available()}")

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
print("Minimal environment verified!")
```

```bash
# Repository smoke tests (from the repository root)
.venv/bin/pytest code -q   # Windows: .venv\Scripts\pytest code -q
```

::: details No GPU? You can still run the course.
The first parts of the course (CartPole, classical methods, DQN, and other control experiments in Chapters 1-9) can be trained on CPU; a GPU is not required. LLM fine-tuning (Chapters 13-18) recommends at least 24 GB of VRAM, or you can use a cloud GPU environment in [ModelScope Notebook](https://modelscope.cn/my/mynotebook).
:::

---

**The following is the complete installation guide. Consult it as needed.**

## Python Environment

We recommend **Python 3.12**, matching the pinned `code/requirements.lock.txt`. We suggest using [uv](https://docs.astral.sh/uv/) to create the project virtualenv, since the lock file pins every dependency for you.

```bash
# Using uv (recommended, from the repository root)
uv venv --python 3.12
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install the full pinned environment
uv pip install -r code/requirements.lock.txt

# Or install only one chapter's minimal dependencies
uv pip install -r code/chapter01_cartpole/requirements.txt
```

::: tip Chapter extras are not in the lock file
Chapter scripts also import packages the lock file omits (`matplotlib`, `swanlab`, `ale-py`, and others). After the lock install, add the chapter extras you need, for example `uv pip install -r code/requirements.txt` (merged summary) or the chapter file above.
:::

If you prefer conda or plain venv, any Python 3.12 environment works on Linux + CUDA; install the same lock file with `pip install -r code/requirements.lock.txt`. On macOS or Windows, or for CPU-only use, prefer the per-chapter route with a matching PyTorch wheel instead of the Linux/CUDA lock file.

::: tip Why Python 3.12?
The lock file in `code/requirements.lock.txt` was resolved with Python 3.12 (torch 2.14.1, transformers 4.57.3, gymnasium 1.4.0). Using the same interpreter avoids binary-wheel surprises.
:::

## Installing PyTorch

PyTorch is the foundation for all deep learning experiments in this course. Choose the installation command that matches your hardware.

**Step 1: Check your GPU driver**

```bash
# Check NVIDIA driver and CUDA version
nvidia-smi
```

The upper-right corner of the output shows `CUDA Version: 12.x`, which is the highest CUDA version supported by your installed driver.

**Step 2: Install PyTorch**

```bash
# CUDA 12.4 (recommended for most modern GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8 (for older GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU-only (no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (M1/M2/M3/M4 Macs)
pip install torch torchvision
```

**Step 3: Verify PyTorch GPU**

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Core RL Packages

These packages are used throughout the classic and deep RL parts of the course (Chapters 1-9).

```bash
# Gymnasium (the successor to OpenAI Gym; the foundation for all environments)
pip install gymnasium

# Stable-Baselines3 (a well-tested RL algorithms library)
pip install stable-baselines3[extra]

# Scientific computing and visualization
pip install numpy scipy matplotlib seaborn pandas

# Progress bars and logging
pip install tqdm tensorboard wandb
```

## Simulator Environments

Different chapters use different simulators. Install them as needed.

```bash
# PyBullet robotics simulation
pip install pybullet

# Atari games (requires ale-py)
pip install "gymnasium[atari,accept-rom-license]"
pip install ale-py

# ViZDoom first-person 3D
pip install vizdoom

# stable-retro (classic games)
pip install stable-retro
```

**MuJoCo installation (optional, for high-fidelity robotics simulation)**

```bash
# MuJoCo is now free and open source; install directly via pip
pip install mujoco
# To use MuJoCo environments in gymnasium
pip install "gymnasium[mujoco]"
```

::: warning MuJoCo deep learning version
MuJoCo requires GPU rendering support. On headless servers, you need to set `export MUJOCO_GL=egl` or `export MUJOCO_GL=osmesa`.
:::

**Isaac Lab installation (optional, for GPU-parallel robotics simulation)**

Isaac Lab is the successor to NVIDIA Isaac Gym. It supports training thousands of robots in parallel on GPU and is suitable for large-scale robotics RL research. Requires an NVIDIA GPU + Linux.

```bash
# Isaac Lab depends on Isaac Sim; install Isaac Sim first
pip install isaacsim[all]

# Clone the Isaac Lab repository
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

# Verify installation
python -c "import isaacsim; print('Isaac Lab ready')"
```

::: warning Isaac Lab system requirements
Isaac Lab requires Linux (Ubuntu 22.04 recommended) and an NVIDIA RTX series GPU (at least 8 GB VRAM). macOS is not supported. If you use macOS, skip Isaac Lab and use PyBullet or MuJoCo as alternatives.
:::

**Unity ML-Agents installation (optional, for 3D game RL)**

Unity ML-Agents lets you train RL agents inside 3D environments built with the Unity engine, suitable for tasks like platform jumping, obstacle avoidance, and other tasks requiring spatial reasoning.

```bash
# Install the Python-side mlagents package
pip install mlagents-envs

# If you need training functionality (includes PyTorch dependencies)
pip install mlagents

# Verify installation
python -c "from mlagents_envs.environment import UnityEnvironment; print('ML-Agents ready')"
```

Using ML-Agents also requires downloading or building Unity environments (`.exe` / `.app` / Linux executables). Pre-built environments are available from [ML-Agents GitHub Releases](https://github.com/Unity-Technologies/ml-agents/releases). For detailed usage instructions, see [Learning Resources and Reproduction Projects](../appendix_c_resources/learning-resources).

::: tip Unity ML-Agents use cases
The unique value of ML-Agents is **3D spatial reasoning**: Atari uses 2D pixels, CartPole uses low-dimensional vectors, while ML-Agents provides a complete 3D physics environment (gravity, collisions, occlusion). If your research involves visual navigation, spatial reasoning, or multi-agent 3D coordination, ML-Agents is a strong complement to Gymnasium/PyBullet.
:::

## LLM Training Frameworks

The LLM post-training part of the course (Chapters 13-18) involves large model alignment training.

```bash
# Hugging Face ecosystem
pip install transformers datasets accelerate peft

# TRL (Transformer Reinforcement Learning, core of DPO/PPO training)
pip install trl

# Quantized inference (optional, saves VRAM)
pip install bitsandbytes

# Evaluation tools
pip install lm-eval
```

::: tip Version compatibility
If you encounter version conflicts, you can install a tested combination with the following command:

```bash
pip install "transformers==4.57.3" "trl==0.24.0" "datasets==4.4.1" "accelerate==1.10.1" "peft==0.21.2"
```

:::

## GPU Driver Checklist

Before starting training, confirm that all of the following are working properly.

| Check item    | Command                          | Expected result                   |
| ------------- | -------------------------------- | --------------------------------- |
| NVIDIA driver | `nvidia-smi`                     | Shows driver version and GPU info |
| CUDA version  | `nvcc --version`                 | Matches PyTorch compile version   |
| cuDNN         | `torch.backends.cudnn.version()` | Returns a version number          |
| PyTorch GPU   | `torch.cuda.is_available()`      | `True`                            |

```python
# One-click check script
import torch
print("=" * 50)
print("Environment Check Report")
print("=" * 50)
print(f"Python:       {__import__('sys').version}")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda or 'CPU'}")
if torch.cuda.is_available():
    print(f"cuDNN:        {torch.backends.cudnn.version()}")
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Apple MPS:    {torch.backends.mps.is_available()}")
print("=" * 50)
```

## Common Installation Problems and Fixes

### CUDA Version Mismatch

**Symptoms**: `RuntimeError: CUDA out of memory` or `Found no NVIDIA driver`

**Cause**: The CUDA version PyTorch was compiled with does not match the system driver.

```bash
# Check the CUDA version used by the current PyTorch
python -c "import torch; print(torch.version.cuda)"

# If mismatched, reinstall the corresponding version of PyTorch
# Uninstall first
pip uninstall torch torchvision -y
# Then install the matching version (see the PyTorch installation section)
```

### Package Version Conflicts

**Symptoms**: `ImportError` or `AttributeError` indicating an incorrect package version.

```bash
# Check versions of all installed packages
pip list | grep -E "torch|gymnasium|transformers|trl"

# Force reinstall to resolve conflicts
pip install --force-reinstall <package_name>
```

### Apple Silicon (M-Series Chip) Compatibility

```bash
# M-series chips use MPS acceleration
# Make sure the arm64 version of PyTorch is installed
python -c "import platform; print(platform.processor())"
# Should output arm64

# Some RL environments do not support rendering on Mac; use headless mode
# export PYOPENGL_PLATFORM=egl
```

### MuJoCo Rendering Failure

```bash
# On headless servers, set the environment variable
export MUJOCO_GL=egl
# Or use osmesa
export MUJOCO_GL=osmesa
```

## Global Verification Script

Run the following script. "OK" on every row except "CUDA GPU" means the setup is complete (that row is expected to read "MISSING" on CPU-only and macOS machines); a "MISSING" extra just means you have not installed that chapter's extras yet (see the tip under [Python Environment](#python-environment)).

```python
"""
Course environment one-click verification script
Run: python verify_env.py
"""
import sys

checks = []

# 1. Python version (the lock file targets Python 3.12)
ver = sys.version_info
checks.append(("Python >= 3.12", ver >= (3, 12)))

# 2. Core packages (pinned in the lock file) and chapter extras
for pkg in ["numpy", "tqdm"]:
    try:
        __import__(pkg)
        checks.append((pkg, True))
    except ImportError:
        checks.append((pkg, False))

# Chapter extras (installed via code/requirements.txt or a chapter file, not the lock file)
for pkg in ["matplotlib"]:
    try:
        __import__(pkg)
        checks.append((pkg + " (extra)", True))
    except ImportError:
        checks.append((pkg + " (extra)", False))

# 3. PyTorch
try:
    import torch
    checks.append(("PyTorch", True))
    checks.append(("CUDA GPU", torch.cuda.is_available()))
except ImportError:
    checks.append(("PyTorch", False))

# 4. RL environments
try:
    import gymnasium
    checks.append(("Gymnasium", True))
except ImportError:
    checks.append(("Gymnasium", False))

# 5. LLM frameworks
for pkg in ["transformers", "datasets", "peft", "accelerate", "trl"]:
    try:
        __import__(pkg)
        checks.append((pkg, True))
    except ImportError:
        checks.append((pkg, False))

# Output report
print("\n" + "=" * 45)
print("  Environment Verification Report")
print("=" * 45)
for name, ok in checks:
    status = "OK" if ok else "MISSING"
    print(f"  {name:<25} {status}")
print("=" * 45)
passed = sum(1 for _, ok in checks if ok)
print(f"  Passed: {passed}/{len(checks)}")
print("=" * 45 + "\n")
```

## Package Overview

| Package           | Recommended version | Purpose                                        | Relevant chapters   |
| ----------------- | ------------------- | ---------------------------------------------- | ------------------- |
| Python            | 3.12                | Runtime (matches `code/requirements.lock.txt`) | Entire book         |
| PyTorch           | 2.14+               | Deep learning framework                        | Entire book         |
| gymnasium         | 1.4+                | RL environment interface                       | Ch1-Ch9             |
| stable-baselines3 | 2.9+                | Pre-packaged RL algorithms                     | Ch1-Ch9             |
| numpy             | 2.5+                | Numerical computing                            | Entire book         |
| matplotlib        | 3.7+                | Visualization and plotting                     | Entire book         |
| pybullet          | 3.2+                | Robotics simulation                            | Appendix C          |
| mujoco            | 3.0+                | High-fidelity physics simulation               | Ch4, Ch7-Ch10, Ch24 |
| isaacsim          | 4.0+                | GPU-parallel robotics simulation (Isaac Lab)   | Ch7, Ch24           |
| mlagents          | 1.0+                | Unity 3D game RL environments                  | Ch8, Appendix C     |
| ale-py            | 0.10+               | Atari emulator                                 | Ch5                 |
| transformers      | 4.57+               | LLM model loading                              | Ch13-Ch18           |
| trl               | 0.24+               | LLM reinforcement learning training            | Ch13-Ch15           |
| peft              | 0.21+               | Parameter-efficient fine-tuning (LoRA)         | Ch19, Ch23          |
| accelerate        | 1.10+               | Distributed training                           | Ch13-Ch18           |
| datasets          | 4.4+                | Dataset loading                                | Ch13-Ch18           |
| wandb             | 0.16+               | Experiment tracking                            | Entire book         |
| tensorboard       | 2.15+               | Training visualization                         | Entire book         |

::: tip No need to install everything at once
We recommend installing packages as you progress through the course: start with the basics (Python environment, PyTorch, core RL packages), then install simulator environments and LLM frameworks when you reach the corresponding chapters. This avoids having to resolve too many dependency conflicts at once.
:::

## Running the Course Website

The book itself is a VitePress site. From the repository root:

```bash
npm install
npm run dev:fast   # dev server, no asset optimization (~5s startup)
```

`npm run dev:fast` is the day-to-day command. The full `npm run build` also optimizes assets and regenerates the sitemap.
