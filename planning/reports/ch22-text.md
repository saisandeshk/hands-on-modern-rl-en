# Report: ch22-text

## Summary

Reviewed both in-scope pages fully, fixed 5 technical/cross-reference errors, added two compact subsections to `training.md` (test-time compute + invalid traces; verifier isolation + rollout accounting) to close the main curriculum gaps, and created `review.md` (outcomes, key equations, pitfalls, 8 exercises with verified solutions). `npx vitepress build docs` succeeds; prettier clean.

## Technical fixes

1. `training.md` (Set-of-Mark example): the agent turn read `type("alice") → click(Box 1) → …` — typing before selecting the field. Corrected to `click(Box 1) → type("alice") → click(Box 2) → type("***") → click(Box 3)` with a sentence noting select-then-type order. Verified by reading the snippet; the old order contradicts how focus works.
2. `training.md` (joint objective): called $\mathcal{J}(\theta) = \mathbb{E}[\sum \gamma^t r_t] - \beta \mathcal{L}_{\text{ground}}$ a "single PPO objective". It is a return-minus-supervision teaching objective, not the PPO clipped surrogate. Relabeled to "joint RL-plus-supervision objective". Verified against the equation itself and standard PPO form.
3. `training.md` (reward decomposition): illegal-action weight was written $\gamma$, colliding with the discount factor $\gamma$ in the Step 1 MDP. Renamed to $\delta$ with an explicit note. Verified by grep: both uses on the same page.
4. `training.md` cross-refs: "Tool Use and Trajectory" → "Tool Use and Trajectory Generation" (matches sidebar 19.4); "Instruction-Level" → "Instruction Hierarchy" (matches sidebar 22.2 and the page's own subject). Verified against `config.mjs` sidebar.
5. `safety-swarm.md` cross-refs: heading and body said "Chapter 25: Reward Hacking and Alignment Failures" / "Echoing [Chapter 25: Alignment Failures]"; sidebar Ch. 25 is "Reward Hacking and RL Evaluation". Fixed both. Verified against `config.mjs`.

No other confident errors found: grounding MSE, DPO loss, curriculum-band math, difficulty weights ($0.4+0.2+0.2+0.2=1.0$), and the OSWorld/AndroidWorld/Set-of-Mark/Instruction-Hierarchy/DPO arXiv ids match well-known records; illustrative numbers (reverse-curriculum table, $\alpha/\beta/\delta$ defaults) were already correctly labeled as teaching records, not paper results.

## Other changes

- `training.md`, end of Step 8: new "Test-Time Compute and Invalid Traces" (~180 words) — $K$-sample click voting as GUI answer aggregation with Ch. 17 link, verify-then-act, step budget $B$, invalid-trace filtering/counting.
- `training.md`, end of Step 9: new "Isolate the Verifier and Account for Rollouts" (~330 words) — state-not-pixels verifiers, isolated grading copies, hidden held-out tests, (task, snapshot, seed) dedup, env-vs-policy error logging, $\mathcal{D}_{\text{RFT}}$ definition, tokens/s + verifier-throughput + utilization accounting, actor/worker/verifier/learner roles, sync/async with policy-version tags, snapshot+weight checkpointing, prespecified reproduction triple.
- New `docs/chapter22_computer_use/review.md`: 7 outcomes, 8 key equations with one-line readings, 5 pitfalls, 8 exercises (conceptual, numeric, derivation, $2\times$ coding) with complete `::: details Solution` blocks. All numerics executed and confirmed: MSE $0.0029$/RMSE $\approx 0.0539$/gradient $(0.10,-0.04)$; curriculum probs $[1/6,1/3,1/2]$; dedup $50/190 \approx 26.3\%$ vs naive $25.0\%$, RFT set $46$; shaped return $1.521$ vs unshaped $0.729$; vote mean $(0.455, 0.625)$.

## Needs external verification

- Paper-specific attributions I could not check offline and left as-is (all already softened in-text as "studies/emphasizes" or teaching-labeled): UI-TARS-2 arXiv:2509.02544 claims (multi-turn RL, sandboxes, data flywheel); AutoGLM arXiv:2411.00820 interfaces/curriculum; MobileRL arXiv:2509.18119 replay/filtering; ComputerRL arXiv:2508.14040 hybrid actions/Entropulse; CogAgent 18B-paper/9B-repo and the arXiv:2408.16500-is-CogVLM2 note; EVA arXiv:2505.14289; Agent Security Bench arXiv:2410.02644; InjecAgent "24% of GPT-4 tests" figure (already scoped to its ReAct setting).
- `training.md` device commands (`python -m open_autoglm.server/run`, model id `Open-AutoGLM`) are already flagged as revision-dependent; confirm against the repo before any stronger claim.

## Sidebar additions

- Title: `22.R Summary and Exercises`, link: `/chapter22_computer_use/review`. (No other new pages.)

## Commands run and results

- `.venv/bin/python -c …` (recomputed Ex. 2/3/5/8 numbers): all match the written solutions — $(0.455, 0.625)$, $[1,2,3]$ / $[1/6,1/3,1/2]$, $1.521$, $0.0539$.
- `npx prettier --write` on the three files: clean (unchanged).
- `npx vitepress build docs`: success, no dead links.

## Open issues

- Filename `safety-swarm.md` vs its content/sidebar title (prompt injection, no swarm content) is a legacy misnomer; renaming breaks the sidebar so I left it for the coordinator.
- `safety-swarm.md` H1 ("Blocking Malicious Instructions in Web Content") differs in wording from sidebar 22.2 ("Prompt Injection and Instruction Hierarchy"); content covers both, so I left the H1 untouched — coordinator may want alignment.
- Full FLOP-level sample/inference/training compute accounting and disaggregated-training depth live in Ch. 18; Ch. 22 now points at the GUI bottleneck (app latency) without duplicating that chapter.

## Coverage table (RL-12; scope: computer-use/GUI observation, action, environments, rewards)

| Item                                                         | Status          | Where                                                                                 |
| ------------------------------------------------------------ | --------------- | ------------------------------------------------------------------------------------- |
| RL with verifiable rewards                                   | covered         | 22.1 binary task verifier, GUIEnv                                                     |
| Math/code/structured-output task envs                        | partial         | GUI app-state verifiers are the in-scope instance; math/code envs belong to Ch. 15/20 |
| Executable verifier isolation + hidden tests                 | covered (added) | new verifier-isolation subsection                                                     |
| Outcome rewards, sparse terminal credit                      | covered         | 22.1 Step 1, Step 7                                                                   |
| Process rewards, step-level credit                           | covered         | progress evaluator + Ch. 17 link                                                      |
| Reasoning traces, hidden reasoning, observable answers       | covered         | thought/action/coord, reflection records                                              |
| Inference-time search, self-consistency, aggregation         | covered (added) | K-sample voting, verify-then-act                                                      |
| Train-time vs test-time compute                              | covered (added) | test-time subsection, budget $B$                                                      |
| Rejection fine-tuning, rollout distillation                  | covered (added) | $\mathcal{D}_{\text{RFT}}$ definition                                                 |
| On-policy teacher-student distillation vocabulary            | partial         | RFT/distillation in GUI terms; general treatment in Ch. 15                            |
| Reasoning length, budget control, invalid traces             | covered (added) | budget $B$, invalid-trace filtering                                                   |
| Multi-turn state, observations, tool-call actions            | covered         | history-conditioned policy, action spaces                                             |
| Env transitions, tool results, terminal conditions           | covered         | GUIEnv, snapshots, verifiers                                                          |
| Trajectory-level rewards, delayed credit                     | covered         | Steps 6–7 (curriculum, backward, shaping)                                             |
| Sandbox, reset, deterministic replay, side-effect boundaries | covered         | VM snapshots, sandbox, export checks                                                  |
| Tool-error, timeout, retry, duplicate-action semantics       | covered (added) | env-error logging, dedup key                                                          |
| Actor, rollout worker, verifier, learner roles               | covered (added) | role-mapping paragraph                                                                |
| Sync vs async rollout-update loops                           | covered (added) | collect-then-update vs version-tagged streaming                                       |
| Policy broadcast, rollout version, staleness                 | covered (added) | version tags per trajectory                                                           |
| Colocated vs disaggregated generation/training               | partial         | GPU-vs-desktop bottleneck framing; depth in Ch. 18                                    |
| Generation batching, backpressure, learner starvation        | partial         | utilization framing; depth in Ch. 18                                                  |
| Checkpoint, rollout, optimizer, sampler recovery             | partial         | snapshot+weight checkpoints; optimizer depth in Ch. 18                                |
| Retry dedup, failed-rollout accounting                       | covered (added) | dedup key, RFT exclusion                                                              |
| Prompt curriculum, filtering, evolving distributions         | covered         | Step 6 + flywheel                                                                     |
| Rollout tokens/s, verifier throughput, update utilization    | covered (added) | three-number accounting                                                               |
| Sample/inference/training/total compute accounting           | partial         | rollout/verifier/utilization tracked; full FLOP accounting in Ch. 18                  |
| Independent reasoning/capability/safety/agent evaluation     | covered         | Step 9 slices + 22.2 four-outcome eval                                                |
| Reward hacking, spec gaming, held-out shifts                 | covered         | hidden tests, state-not-pixels, Ch. 25 echo                                           |
| Clean reproduction, prespecified objective/reward/extension  | covered (added) | task list + verifier revision + seed                                                  |
| Multi-agent self-play, auto curriculum (Depth)               | out of scope    | Depth; swarm content lives in Ch. 19                                                  |
| Multimodal, embodied, production agent training (Later)      | out of scope    | Later; Step 10 keeps deployment boundaries only                                       |
