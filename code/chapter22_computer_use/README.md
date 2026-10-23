# Lab: GUI Form-Filling Agent (Chapter 22)

A miniature computer-use RL lab. A text-rendered "screen" holds shuffled UI
elements (text fields, checkboxes, a submit button, distractors); the agent
sees a structured screen description and acts with `click(element_id)` /
`type(element_id, token)` in the Set-of-Mark style of Chapter 22, Step 2.
Reward is sparse by default (+1 only for a correct submission).

**What the lab shows**

- Sparse terminal reward fails on the full multi-step task (0% success).
- A curriculum (easy tasks first, promoted at 50% rolling success) climbs
  all four difficulty levels and generalises to held-out layouts.
- Dense progress shaping helps (beats sparse) but learns slower than the
  curriculum on equal budgets.
- A hackable shaping bonus (pay per failed submit click -- a correct
  submit still pays only +1) teaches submit-spam:
  near-maximum reward with ~0% success -- the reward-hacking failure mode.

**Files**

- `gui_env.py` -- MiniForm environment (layout, hit-testing, verifier,
  observation encoding, reward modes).
- `train_gui_agent.py` -- REINFORCE with a learned value baseline, success
  replay, curriculum scheduler, evaluation, CSV/JSON/PNG outputs.
- `test_ch22_lab.py` -- unit tests plus a subprocess smoke run.

## How to run

From the repository root (uses the project virtualenv):

```bash
# Smoke test: all four regimes, tiny config (<60 s on CPU)
.venv/bin/python code/chapter22_computer_use/train_gui_agent.py --smoke

# Full comparison (~4 min on CPU, seed 0; reproducible on the same machine)
.venv/bin/python code/chapter22_computer_use/train_gui_agent.py --device cpu

# One regime, custom budget
.venv/bin/python code/chapter22_computer_use/train_gui_agent.py --mode shaped --episodes 2000
```

Useful flags: `--mode {all,sparse,curriculum,shaped,hackable}`,
`--seed`, `--device {auto,cpu,cuda}`, `--smoke`, `--episodes`,
`--eval-every`, `--outdir` (default `runs/chapter22_computer_use/`).

## Tests

```bash
.venv/bin/pytest code/chapter22_computer_use -q
```

Runs the unit tests (hit-testing, state updates, success checker,
deterministic layouts, REINFORCE math vs hand-computed values) and one
`--smoke` training run in a subprocess.

## Expected runtime and results

- Smoke: a few seconds on CPU.
- Full (`--device cpu`): ~3-6 min on a laptop CPU; writes
  `metrics_<mode>.csv`, `eval_<mode>.csv`, `summary.json`
  (plus `training_curves.png` for a multi-regime run) under
  `runs/chapter22_computer_use/`
  (gitignored), and prints a summary table.

Reference results (seed 0, greedy policy on the full task; `mean reward`
is the mean undiscounted reward over the last 200 stochastic training
episodes, not the greedy snapshot reward):

| regime     | train success | held-out success | mean length | submit share | mean train reward (last 200 eps) |
| ---------- | ------------- | ---------------- | ----------- | ------------ | -------------------------------- |
| sparse     | 0.000         | 0.000            | 16.00       | 0.000        | 0.005                            |
| curriculum | 0.328         | 0.391            | 12.09       | 0.565        | 0.435                            |
| shaped     | 0.141         | 0.172            | 14.77       | 0.459        | 0.522                            |
| hackable   | 0.000         | 0.000            | 16.00       | 1.000        | 6.173                            |

Exact numbers vary with `--seed`; the ordering (curriculum > shaped >
sparse on success; hackable highest reward at zero success) is the robust
finding. See `docs/chapter22_computer_use/lab.md` for the walkthrough.
