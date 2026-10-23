---
title: '22.3 Lab: Form-Filling GUI Agent'
---

# 22.3 Lab: Form-Filling GUI Agent

> **Section goal**: Train a policy with REINFORCE plus a baseline on a miniature GUI task, and compare sparse reward against a curriculum and reward shaping -- including a shaping bug that teaches submit-spam.
>
> **Learning path**: [19.x Agentic RL](../chapter19_agentic/overview) -> [22.1 GUI Agent Training](./training) -> **22.L Form-Filling GUI Agent** -> [22.2 Prompt Injection](./safety-swarm)
>
> **Code and resources**: [gui_env.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter22_computer_use/gui_env.py) · [train_gui_agent.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter22_computer_use/train_gui_agent.py) · [test_smoke.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter22_computer_use/test_smoke.py)

Real GUI benchmarks (OSWorld, AndroidWorld) need virtual machines and GPU-scale models, so this lab distils their structure into a task a laptop can run in minutes: fill a small on-screen form and submit it. The observation is a structured screen description, the actions are Set-of-Mark style `click(element_id)` / `type(element_id, token)`, and the reward is sparse by default. Everything the chapter warns about -- sparse terminal feedback, long-horizon credit assignment, curriculum sampling, progress rewards, and reward hacking -- shows up here in measurable form.

## What you will learn

- How to formulate a GUI task as an MDP with a terminal verifier, and why sparse reward makes a 5-step form unsolvable from scratch without help.
- How a curriculum (easy tasks first) and dense progress rewards each restore learning, and why they learn at different speeds.
- How a seemingly harmless shaping bonus (pay per submit click) produces submit-spam: maximum reward at zero success.
- How to evaluate layout generalisation with held-out episode seeds, and how to read success rate, episode length, and submit share together instead of trusting reward alone.

## Background

[22.1](./training) formulates computer use as an MDP over screenshots with mouse/keyboard actions and a binary terminal reward (Step 1), then reduces coordinate prediction to discrete element selection via Set-of-Mark prompting (Step 2). Our environment copies that reduction: the policy never outputs pixels, it selects element ids, exactly as a SoM agent selects box labels.

The training recipe follows the chapter's pipeline (Step 4): demonstrations first, then verifier-filtered successful trajectories, then online RL. Our "demonstrations" are the agent's own lucky successes, filtered by the task verifier and imitated with a supervised loss next to the REINFORCE update. The curriculum (Step 6) borrows the chapter's 30-70% "possible but not yet mastered" band as a promotion rule: difficulty ramps one level once rolling success reaches 50% (the chapter itself uses the interval to filter which tasks to sample, not as a level-promotion threshold). The shaped regime adds dense progress rewards (Step 7), and the hackable regime shows a mis-specified reward of the kind Step 9 asks you to diagnose: the metric goes up while the task does not (here the verifier still reports failure correctly -- it is the reward function, not the verifier, that pays for wrong submits). The algorithm itself is REINFORCE with a learned value baseline ([Chapter 6](../chapter06_policy_gradient/reinforce), [baseline experiment](../chapter06_policy_gradient/baseline-experiment)) plus batch-standardised advantages and an entropy bonus; the success filter uses a step-level progress signal in the spirit of [process supervision](../chapter17_prm_search/outcome-vs-process).

## Setup and how to run

From the repository root, using the project virtualenv:

```bash
pip install -r code/chapter22_computer_use/requirements.txt  # numpy, torch, matplotlib
# Smoke test: all four regimes, tiny config, a few seconds on CPU
.venv/bin/python code/chapter22_computer_use/train_gui_agent.py --smoke
# Full comparison (~4 min on CPU, reproducible under --seed on the same machine)
.venv/bin/python code/chapter22_computer_use/train_gui_agent.py --device cpu
# One regime with a custom budget
.venv/bin/python code/chapter22_computer_use/train_gui_agent.py --mode shaped --episodes 2000
```

Useful flags: `--mode {all,sparse,curriculum,shaped,hackable}`, `--seed`, `--device {auto,cpu,cuda}`, `--episodes`, `--eval-every`, `--outdir` (default `runs/chapter22_computer_use/`, gitignored). Tests:

```bash
.venv/bin/pytest code/chapter22_computer_use -q
```

Outputs per run: `metrics_<mode>.csv` (per-episode success, length, reward, submit share), `eval_<mode>.csv` (periodic greedy snapshots on train and held-out seeds), `summary.json`, plus a summary table on stdout. A multi-regime run (`--mode all`) additionally writes `training_curves.png`; a single-regime run does not produce the PNG.

## Walkthrough of the key code

**The screen** (`gui_env.py`). Each episode draws a fresh layout from its seed: 2 text fields, 2 checkboxes, 1 submit button, and 2 distractor buttons are placed on an 8x6 grid, SoM ids 0-6 and labels are reshuffled, and fresh target values are sampled. The same seed always rebuilds the same screen:

```python
elements = make_layout(seed)          # positions, ids, labels from one RNG stream
task = make_task(elements, level, seed)  # e.g. set 'city' to 'b', check 'gift', then submit
```

Four difficulty levels activate 1, 2, 3, then all 4 sub-goals. `element_at(elements, x, y)` hit-tests a continuous click point to an element (half-open rect edges, covered by unit tests) -- the miniature of coordinate grounding. `check_success` is the verifier: every active sub-goal must hold its target when submit is clicked. A wrong submit returns a form error and the episode continues, so the agent can recover; only a correct submit (+1) or the 16-step timeout ends the episode.

**Actions and observations.** Actions `0..6` are `click(eid)`; `7..27` are `type(eid, token)` over a 3-token vocabulary. Clicking a checkbox toggles it, typing into a field sets its value, everything else is a harmless no-op. The observation flattens the structured screen description (per element: kind, position, label, state) plus the task targets into a 204-dim feature vector -- the accessibility-tree analogue from Step 1.

**Reward modes.** `sparse` pays +1 only on correct submission. `shaped` adds +0.25 per newly satisfied sub-goal (paid only on a new episode best, so toggling cannot farm it) minus 0.01 per step. `shaped_hackable` adds +0.4 for every _failed_ submit click (a correct submit still pays only the +1 terminal reward) -- the deliberate bug. Note that a box target of the form "uncheck X" is already satisfied at reset, so it never pays a progress reward: depending on the sampled targets, an episode has 2, 3, or 4 payable sub-goals.

**The update** (`train_gui_agent.py`). Batched REINFORCE with a state-value baseline and standardised advantages -- the Chapter 6 update plus batch-standardised advantages (zero mean, unit variance over the batch) and an entropy bonus:

```python
advantages = returns - values.detach()
advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
policy_loss = -(log_probs * advantages).mean() - entropy_coef * entropy.mean()
value_loss = mse_loss(values, returns)
```

Next to it, a success-replay loss imitates verifier-filtered successes. Two details matter. First, only _true_ completions enter the buffer, never high-reward failures -- so replay cannot launder the hack. Second, only steps that verifiably helped are imitated (sub-goal count reached a new best, or the winning submit): a lucky trajectory is a dozen junk steps around one or two good ones, and cloning all of it teaches nothing. Cloning starts once 32 successes are banked, with a gentler optimizer, so early successes are not memorised before the buffer holds enough layouts to generalise from.

**Curriculum and evaluation.** The curriculum trains at the current level and promotes when rolling training success over the last 200 episodes passes 50% (minimum 400 episodes per level). Every snapshot evaluates the greedy policy on the _full_ task over 64 seeds from a train-eval block and 64 seeds from a disjoint held-out block. Both eval blocks use seed ranges disjoint from every training seed, so the "train" column is itself a generalisation check on unseen layouts from a familiar range rather than a training-set score; the held-out column tests a second, farther range.

## Results from a run you can reproduce

Full run, `--seed 0`, CPU, 213.5 s total (sparse 3000, curriculum 15000, shaped 15000, hackable 3000 episodes). Greedy policy on the full task, last snapshot; the `mean reward` column is the mean undiscounted reward over the last 200 stochastic _training_ episodes (not the greedy snapshot reward), shown so reward and success can be read side by side:

| regime     | train success | held-out success | mean length | submit share | mean train reward (last 200 eps) |
| ---------- | ------------- | ---------------- | ----------- | ------------ | -------------------------------- |
| sparse     | 0.000         | 0.000            | 16.00       | 0.000        | 0.005                            |
| curriculum | 0.328         | 0.391            | 12.09       | 0.565        | 0.435                            |
| shaped     | 0.141         | 0.172            | 14.77       | 0.459        | 0.522                            |
| hackable   | 0.000         | 0.000            | 16.00       | 1.000        | 6.173                            |

The curriculum promoted at episodes 1000, 4000, and 7500 (levels 1, 2, 3) and its stochastic success over the final 1000 episodes was 0.366. Shaping's replay gate (32 successes) opened around episode 7500; its success was still climbing at the end (0.05 at episode 12000 to 0.14 at 15000). The hackable policy's submit share rose 0.09 -> 0.98 across its 3000 episodes while its mean reward rose 0.67 -> 6.10, against a ceiling of 16 x (0.4 - 0.01) = 6.24 for submitting on every step. Sparse saw 16 lucky successes in 3000 episodes -- too few to move the policy.

## What to look for and why it happens

**Sparse stays at zero.** Five correct actions among 28 (two types, two box clicks when both box targets are "check", plus submit), with shuffled ids every episode, means a random policy almost never completes the form; the few lucky trajectories each use a different layout, so batch gradients cannot extract the shared rule. This is the chapter's central difficulty in miniature: high-dimensional varying observations, a multi-step horizon, and a terminal reward that does not identify the first correct branch.

**Curriculum climbs one rule at a time.** Level 0 (one field plus submit) succeeds by luck ~8% of the time -- dense enough for the success filter to collect clean `type`-then-`submit` pairs and teach label matching. Each promotion reuses that skill and adds one sub-goal. Held-out success (0.39) matches train success (0.33) within eval noise: the policy reads labels from the observation rather than memorising layouts, so unseen position/id/label combinations work.

**Shaping helps but starts later.** Progress rewards dense-ify credit, yet the full 5-step chain still succeeds by luck only ~0.5% of the time, so the replay gate opens late and the policy is still climbing when the budget ends. Equal budgets rank curriculum above shaping here because the curriculum manufactures its own dense successes.

**Hackable is the memorable row.** Reward is the highest of all four regimes while success is zero and every greedy action is a submit click. Two mechanisms combine: the per-click bonus makes submit-spam (6.17) out-earn solving the task (an optimal run averages ~1.7, up to ~1.95 when both box targets happen to be "check"), and wrong submits are free retries, so nothing punishes spamming. Note the curriculum's submit share of 0.56 for contrast: with free retries, submitting often is a _reasonable_ checking strategy once the policy also does the work -- the hackable policy never does the work. Always read reward next to success rate; either alone misleads. For the broader safety picture see [22.2](./safety-swarm) and [Chapter 25](../chapter25_alignment_failures/classical-failures).

## Extension exercises

1. **Make wrong submits terminal.** End the episode (reward 0) on any incorrect submit, rerun curriculum and shaped, and predict the outcome before you look. _Hint: which regime depends on retry-after-error to discover its first success? Check the promotion episodes against the baseline run._
2. **Find the hacking threshold.** Lower `SUBMIT_BONUS` until the hackable run starts succeeding (a shaped run is unaffected: the bonus is only read in the `shaped_hackable` reward mode). _Hint: compare the bonus against the return of a full solution (mean ~1.7) and of pure spam (16 x (bonus - 0.01) -- each step also pays the 0.01 step penalty); the flip happens where the two cross, around bonus ~0.12-0.13._
3. **Ablate the success replay.** Set the replay gate absurdly high (or `bc_epochs=0`) and rerun the curriculum. _Hint: pilot runs showed the net only fits the shared label-matching rule once the buffer holds hundreds of clean `type`/`submit` steps from dozens of layouts; watch the buffer fill while success flatlines._
4. **Harden the verifier.** Charge 0.1 per wrong submit (a "lockout timer") and rerun hackable. _Hint: spam now costs 16 x 0.1 per episode; does the policy solve the task or just submit less? Read submit share together with success._
5. **Stress generalisation.** Add a fourth field label to the pool and new held-out seeds that use it, then evaluate the trained curriculum policy without retraining. _Hint: the task encoder has fixed one-hot slots; decide what an unseen label should map to before measuring the gap._
6. **Profile the first submit.** Record, per episode, the step index of the first submit click for hackable vs curriculum policies. _Hint: `run_episode` already returns actions; the hackable agent should peak at step 1 while the curriculum agent's first submit tracks task completion._
