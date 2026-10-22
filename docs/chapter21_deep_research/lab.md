---
title: '21.L Lab: Multi-Turn Research with Delayed Reward'
---

# 21.L Lab: Multi-Turn Research with Delayed Reward

Train a small tool-calling research agent on an offline toy web: a corpus of
134 synthetic documents with links between them, and questions whose answers
require finding and combining facts from 2-3 documents. The agent uses three
tools, `search(term)`, `open(doc)`, and `answer(candidate)`, under a step
budget. The reward is 1 for a correct final answer (with a small per-step
cost), so credit for a good search must propagate back across a 5-step
trajectory — the same delayed-reward problem that makes Deep Research
training hard.

## What you will learn

- How to cast multi-turn web research as a repeatable RL environment: fixed
  tools, a step budget, and a verifiable terminal reward.
- Why sparse trajectory-level reward struggles with discovery: REINFORCE from
  scratch collapses to answering immediately, barely above chance.
- What a dense process reward changes: a bonus for opening relevant
  documents keeps fine-tuning stable and raises training returns — but only
  when the relevance signal is trustworthy.
- How a loose shaping proxy gets gamed: a bonus for opening _any_ document
  produces the highest training return in the lab and the worst success rate,
  with trajectories that burn the whole step budget.
- How to read a Deep Research run the way
  [Section 21.2](./deep-research-eval) prescribes: answer correctness split by
  difficulty, plus steps and tool-call counts — never a single aggregate
  number.

## Background

The environment mirrors the [Browser RL Harness](./browser-rl-harness):
`search` returns ranked document ids, `open` reveals page text, and `answer`
submits a short answer that is scored by exact match against a reference
answer. Questions come in two difficulties. Easy questions name a paper whose
first author has a unique name, so the chain is paper plus one profile (2
documents). Hard questions target a first author whose name is shared by two
scholars and additionally name the author's lab, so the agent must open the
paper plus both same-name profiles and use the affiliation to disambiguate
(3 documents) — the "Chen Li" same-name case from Section 21.1.

The training regimes mirror the chapter's reward discussion. Outcome-only
supervision is the DeepResearcher / Search-R1 setup from Section 21.1: one
terminal signal, no process reward. The relevant-document bonus is the
Web-Shepherd / Atom-Searcher idea: dense credit for intermediate steps. The
any-document bonus is the failure mode the chapter warns about: once the
proxy (documents opened) diverges from the goal (correct, cited answers),
optimizing the proxy is reward hacking. All learned policies start from a
behavior-cloned warm-start on scripted-heuristic demos — the SFT cold-start
that pipelines such as Tongyi DeepResearch and O-Researcher put before RL.
A from-scratch outcome run is included as the control that shows why the
cold-start exists.

## Setup and how to run

The lab needs only numpy, torch, and matplotlib, and runs on CPU. From the
repository root:

```bash
# Smoke mode: tiny corpus, finishes in seconds
.venv/bin/python code/chapter21_deep_research/train_research_agent.py --smoke --seed 0 --device cpu

# Full run: 134 documents, 60 questions, ~25 s on CPU
.venv/bin/python code/chapter21_deep_research/train_research_agent.py --seed 0 --device cpu

# Unit tests plus a smoke run in a subprocess
.venv/bin/pytest code/chapter21_deep_research -q
```

Useful flags: `--episodes` (training episodes per reward regime),
`--rel-bonus` (relevant-document shaping), `--any-bonus` (loose shaping),
`--step-cost`, `--max-steps`, `--gamma`, `--hidden`, `--lr`,
`--device {auto,cpu,cuda,mps}`, `--outdir`. Outputs (JSON metrics, CSV,
learning-curve and bar-chart PNGs) go to `runs/chapter21_deep_research/`,
and the script prints a summary table to stdout.

## Walkthrough of the key code

The corpus builder
([web_env.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter21_deep_research/web_env.py))
generates scholars, papers, and labs from one seed so every run is
reproducible. Scholars with colliding display names create the hard
disambiguation questions; paper texts list authors but never affiliations,
so the affiliation cue can only come from opening profiles:

```python
text = (
    f"{title}. Authors: {a1} and {a2}. "
    f"Topic: {rng.choice(TOPICS).lower()}.",
)
```

The environment step
([web_env.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter21_deep_research/web_env.py))
implements the three tools over a discrete action space
(`search-paper`, `search-name-A/B`, `search-lab`, `open-k` slots,
`answer-c` candidates), charges the per-step cost on every action, and pays
shaping bonuses at most once per document:

```python
reward = -self.step_cost
...
elif doc_id in self.question.gold_doc_ids and self.rel_bonus:
    reward += self.rel_bonus
elif self.any_bonus:
    reward += self.any_bonus
```

The policy never sees privileged state
([policies.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter21_deep_research/policies.py)).
Its features are built only from agent-visible information — steps used,
last results by document type, which candidate universities have appeared in
opened profiles — and the actor-critic is a two-layer MLP trained with
REINFORCE against its own value baseline:

```python
adv = G - V
logits, values = model(F)
policy_loss = -(logp * adv.detach()).mean()
value_loss = ((values - G.detach()) ** 2).mean()
```

The trainer
([train_research_agent.py](https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/code/chapter21_deep_research/train_research_agent.py))
warm-starts every regime from the same behavior-cloned policy, then
fine-tunes under each reward, and evaluates greedily on 16 held-out
questions (8 easy, 8 hard) with disjoint papers from training, as
[Section 21.2](./deep-research-eval) recommends.

## Results from a full run

Seed 0, CPU, 4000 episodes per regime (24 s wall-clock). Test set: 16
held-out questions, 8 easy / 8 hard.

| Policy                                   | All  | Easy | Hard | Avg steps | Searches | Opens |
| ---------------------------------------- | ---- | ---- | ---- | --------- | -------- | ----- |
| Heuristic (scripted)                     | 1.00 | 1.00 | 1.00 | 5.0       | 2.0      | 2.0   |
| REINFORCE, outcome only                  | 1.00 | 1.00 | 1.00 | 5.0       | 2.0      | 2.0   |
| REINFORCE, relevant shaping              | 1.00 | 1.00 | 1.00 | 5.0       | 2.0      | 2.0   |
| REINFORCE, loose (any-doc) shaping       | 0.44 | 0.12 | 0.75 | 9.1       | 2.0      | 2.8   |
| REINFORCE from scratch, outcome          | 0.12 | 0.00 | 0.25 | 1.0       | 0.0      | 0.0   |
| REINFORCE from scratch, relevant shaping | 0.12 | 0.00 | 0.25 | 1.0       | 0.0      | 0.0   |
| Random                                   | 0.00 | 0.00 | 0.00 | 1.9       | 0.4      | 0.0   |

Mean training return over the last 400 episodes tells the other half of the
story: outcome +0.86, relevant shaping +1.25, loose shaping +1.82. The
loose-shaping agent earns the _highest_ training return and the _worst_
success rate among warm-started policies. Its test success over training
episodes was 1.00, 1.00, 0.94, 0.94, 0.00, 0.62, 0.88, 1.00, 0.44, 0.44
while its mean return climbed monotonically from +1.63 to +1.82.

## What to look for and why it happens

- **From-scratch REINFORCE answers on step one.** Both from-scratch regimes
  average 1.0 steps with ~12% success (one lucky guess in eight candidates).
  Guessing immediately has positive expected return (about +0.1), while a
  random 5-step search sequence almost never hits the full chain, so the
  gradient pushes the policy into the immediate-answer local optimum and the
  baseline cannot pull it out. This is the sparse-credit-assignment failure
  that motivates SFT cold-starts in real pipelines.
- **Shaping alone does not fix discovery here.** The from-scratch
  relevant-shaping run also stalls at 0.12: the +0.2 document bonuses are
  never sampled often enough to compete with the sure +0.1 from guessing.
  Dense rewards help most when the policy already explores near good
  trajectories — which is exactly what the warm-start provides.
- **Tight shaping densifies fine-tuning.** From the warm-start, both
  outcome-only and relevant shaping hold 1.00, but the shaped regime's
  training return (+1.25 vs +0.86) carries a per-step signal instead of one
  terminal number — the Web-Shepherd effect at toy scale.
- **Loose shaping is gamed, and the metric hides it.** With a +0.4 bonus
  per opened document, opening six documents and answering wrong
  (about +2.2) outscores answering correctly in five steps (about +1.5)
  under the _training_ objective. The policy learns to burn the full
  10-step budget (9.1 average steps, 2.8 opens) while test success
  collapses to 0.44 — even as its training return keeps rising. That
  rising-return / falling-success split is the signature of reward hacking,
  and the reason Section 21.2 insists on scoring answer, evidence, process,
  and cost separately instead of trusting one training curve.

## Extension exercises

1. **Close the gaming loophole with a budget.** Cap the total shaping bonus
   per episode (or decay it per open) and re-run. Hint: change the
   `shaped_paid` logic so the bonus pool is finite, then check whether
   success recovers while steps fall.
2. **Make shaping ungameable by construction.** Replace the any-document
   bonus with a citation check: pay only if the opened document's text
   contains the submitted answer's supporting entity. Hint: reuse
   `parse_profile_univ` at answer time in `TinyResearchEnv.step`.
3. **Ablate the warm-start.** Train the outcome regime from scratch with 5x
   episodes and higher entropy (`--episodes 20000`, edit `entropy_coef`).
   Hint: watch whether success ever lifts off 0.12, and explain what the
   flat curve implies about undirected exploration on long tool chains.
4. **Add a distractor trap.** Inject profile documents whose text mentions a
   wrong university alongside the true one, and extend the heuristic to
   prefer the sentence matching the lab cue. Hint: this tests Layer 2 of
   Section 21.2 (citation support) rather than Layer 1 (answer match).
5. **Price the tools.** Give `search` and `open` different step costs and
   re-derive the optimal trajectory length. Hint: with expensive search,
   opening extra result slots beats re-searching — check the policy's
   search/open counts shift.
6. **Rerun across seeds and report variance.** Run seeds 0-4 and report
   mean and range per regime. Hint: the loose-shaping collapse is
   seed-sensitive (seed 0 ends at 0.44 success with 9.1-step trajectories,
   seed 1 at 0.81 with 6.8 steps); discuss what that implies for trusting
   a single RL run.
