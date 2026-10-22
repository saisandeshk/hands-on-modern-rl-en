---
title: '11.L Lab: Behaviour Cloning versus DAgger'
---

# 11.L Lab: Behaviour Cloning versus DAgger

This lab turns Sections [11.1](./bc-dagger) and [11.2](./irl-gail) into
running code. You will train behaviour-cloning policies from a handful of
CartPole demonstrations, watch a small per-step error snowball into a
failed episode, and then repair the failure with DAgger's
roll-out-and-relabel loop. A short companion script demonstrates the
GAIL-style discriminator reward of Section 11.2.3.

Code:
[bc_dagger_cartpole.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter11_imitation_meta_rl/bc_dagger_cartpole.py) ·
[gail_discriminator_cartpole.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter11_imitation_meta_rl/gail_discriminator_cartpole.py) ·
[tests](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter11_imitation_meta_rl/test_smoke.py)

## What you will learn

- How behaviour cloning reduces imitation to supervised learning, and how
  to measure its failure mode: high per-step agreement on expert states
  alongside low episode return.
- Why that failure happens: the learner is trained on the expert's state
  distribution $d_{\text{expert}}$ but deployed on its own
  $d_{\pi_\theta}$ (distribution shift, Section 11.1.2).
- How DAgger fixes it by aggregating expert labels on the learner's own
  states, reaching expert return from a single demonstration trajectory.
- What a GAIL discriminator reward looks like numerically: high
  $-\log(1-D(s,a))$ on expert-visited pairs, near zero elsewhere.

## Background

Read [11.1 Behaviour Cloning and Interactive Imitation
Learning](./bc-dagger) first. The lab implements its three core objects
directly: the BC loss
$\mathcal{L}_{BC}(\theta) = -\mathbb{E}[\log \pi_\theta(a \mid s)]$
(11.1.1), the $O(T^2\epsilon)$ compounding-error argument (11.1.2), and
the DAgger loop of rollout, expert relabelling, aggregation and retraining
(11.1.3). The companion script implements the discriminator objective of
[11.2.3](./irl-gail) without the adversarial policy update.

## Setup and how to run

You need the project virtualenv (Python 3.12, torch, gymnasium, matplotlib;
no downloads required). Run from the repository root:

```bash
# Full lab (~8 min on a laptop CPU)
.venv/bin/python code/chapter11_imitation_meta_rl/bc_dagger_cartpole.py --device cpu
.venv/bin/python code/chapter11_imitation_meta_rl/gail_discriminator_cartpole.py --device cpu

# Quick check (<60 s per script)
.venv/bin/python code/chapter11_imitation_meta_rl/bc_dagger_cartpole.py --smoke
```

Both scripts accept `--seed`, `--device {auto,cpu,cuda}` and `--outdir`.
Outputs (CSV/JSON metrics and PNG plots) go to
`runs/chapter11_imitation_meta_rl/` (gitignored); a summary table is
printed to stdout. Tests:

```bash
.venv/bin/pytest code/chapter11_imitation_meta_rl -q
```

## Walkthrough of the key code

**The expert is a three-line controller, not a trained network.** Because
DAgger must query the expert on arbitrary learner-visited states, a
deterministic function of state is ideal — and it scores 500/500 on every
seed, so nothing is lost:

```python
def expert_act(obs):
    x, xd, th, thd = (float(v) for v in obs)
    u = 1.0 * th + 0.3 * thd + 0.05 * x + 0.05 * xd
    return 1 if u > 0 else 0
```

Push toward the lean ($\theta$ term), damp the angular velocity, drift
back to the centre. The full BC update is then ordinary cross-entropy
over the demonstration pairs — exactly the
`behavior_cloning_step` of Section 11.1.1:

```python
def bc_loss_from_logits(logits, actions):
    return nn.functional.cross_entropy(logits, actions)  # -mean log pi(a|s)
```

**DAgger is the aggregate loop of Section 11.1.3, verbatim:** roll out
the learner (`rollout_states`), relabel every visited state with
`label_with_expert`, concatenate onto the dataset, and retrain from
scratch. The script's `run_dagger` does one BC training per iteration on
the growing aggregate and records return against the expert-label count,
so BC and DAgger can be compared at matched label budgets.

**The GAIL companion trains only the discriminator.** Expert pairs get
label 1, a weak one-trajectory BC policy's pairs get label 0, and the
reported quantity is the implicit reward $-\log(1-D(s,a))$ from Section
11.2.3.1. The adversarial policy half is omitted on purpose: a
from-scratch REINFORCE learner on the discriminator reward stalls below
return 100 after 40 iterations on this CPU budget, which is itself the
lesson of Section 11.2.4.1 about GAIL's sample efficiency.

## Results from a run we actually performed

Reference run: `--seed 0`, CPU, 472 s for the main script. The expert
scores 500.0 over 10 episodes. BC from $N$ demonstrations (3 seeds):

| Expert trajectories (BC)          | 1     | 2     | 5     | 10    | 25    |
| --------------------------------- | ----- | ----- | ----- | ----- | ----- |
| Mean return                       | 377.7 | 500.0 | 500.0 | 500.0 | 500.0 |
| Agreement on expert states        | 0.823 | 0.935 | 0.988 | 0.991 | 0.996 |
| Agreement on learner's own states | 0.766 | 0.780 | 0.966 | 0.989 | 0.994 |

DAgger starting from a single expert trajectory (mean over 3 seeds):

| DAgger iteration | 0     | 1     | 2     | 3     | 4     |
| ---------------- | ----- | ----- | ----- | ----- | ----- |
| Expert labels    | 500   | 1365  | 2365  | 3365  | 4365  |
| Mean return      | 377.7 | 483.8 | 500.0 | 500.0 | 500.0 |

The discriminator script (6 s) reports accuracy 0.739 on held-out expert
pairs and 0.819 on learner pairs, with mean implicit reward 3.451 on
expert pairs versus 0.502 on learner pairs.

## What to look for and why it happens

Look at the $N=1$ column first: 82% per-step agreement but return 378 —
and on seed 1 the same setting scores just 133 with 77.5% agreement.
That gap _is_ Section 11.1.2: with per-step error $\epsilon \approx 0.2$
and horizon $T = 500$, the no-mistake probability
$(1-\epsilon)^T$ is essentially zero, so nearly every episode leaves the
demonstration manifold at some point, and off-manifold the policy has no
supervision. The second agreement row confirms the mechanism directly:
agreement measured on the learner's _own_ states is consistently below
agreement on expert states, exactly the $d_{\text{expert}}$ vs
$d_{\pi_\theta}$ shift.

Then look at seed 1 under DAgger: 133.1 → 451.4 → 500.0. A single round
of on-policy corrections (597 extra labels from the learner's own
failing rollouts) repairs most of the damage, and the second round
finishes the job — using one full expert trajectory where reliable BC
needs two. The aggregate dataset now covers the states the learner
actually visits, which is the whole point of DAgger's improved
$O(T\epsilon)$ bound.

Finally, the discriminator numbers reproduce the table in Section
11.2.3.2 numerically: pairs the expert visits get several units of
implicit reward, learner-only pairs get near zero, so optimising that
reward would steer a policy back onto the expert's occupancy — without
ever writing down a CartPole reward by hand.

## Extension exercises

1. **Widen the agreement gap.** Evaluate the $N=1$ BC policy's agreement
   as a function of episode time (first 50 steps vs last 50). Hint: bin
   visited states by step index during `rollout_states` and label each
   bin with the expert; late bins should show much lower agreement.
2. **Break the expert's determinism.** Add Gaussian noise to
   `expert_act`'s control signal and re-run the BC sweep. Hint: noisy
   labels raise the Bayes error $\epsilon$; check whether $N=2$ still
   reaches 500 and how the agreement columns change.
3. **DAgger with $\beta$-mixing.** The chapter's pseudocode mixes expert
   actions into rollouts with a decaying $\beta$. Hint: modify
   `rollout_states` to take expert actions with probability $\beta$ and
   compare the label-efficiency curve against $\beta = 0$.
4. **Close the GAIL loop (ambitious).** Replace the fixed weak learner
   with REINFORCE on the frozen discriminator reward you just trained.
   Hint: normalise the per-episode discounted rewards as advantages; if
   it stalls, warm-start the policy with 1-trajectory BC and describe
   what changed.
5. **Try LunarLander-v3.** Swap the environment and the expert (a
   scripted lander or a short trained policy). Hint: the observation and
   action spaces change, so `PolicyNet`, `expert_act` and the label
   budget need rework; expect BC to need far more than 2 trajectories.
6. **First-order check of the $T^2$ claim.** Fit return against
   per-step error across your BC runs and test whether the shortfall
   grows super-linearly in the error. Hint: use $1 - $ agreement on own
   states as $\epsilon$ and plot $500 - $ return versus $\epsilon$.
