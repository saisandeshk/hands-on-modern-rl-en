# Report: ch22-lab (Chapter 22 computer-use lab)

## Summary

Built a hands-on lab for Chapter 22 (previously no runnable code): a
text-rendered GUI-like form-filling environment (`code/chapter22_computer_use/`)
plus a docs page (`docs/chapter22_computer_use/lab.md`). The agent trains with
REINFORCE + learned value baseline (from scratch, torch only) under four
regimes -- sparse, curriculum, shaped, hackable -- and is evaluated on
train vs held-out layout seeds. Final full run (seed 0, CPU, 213.5 s):
sparse 0.00/0.00, curriculum 0.33/0.39, shaped 0.14/0.17, hackable 0.00/0.00
success (train/held-out) with hackable earning the highest reward (6.17) via
submit-spam (submit share 1.00). All 22 pytest tests pass; smoke runs in
~3-7 s. `npx prettier --write` applied; `npx vitepress build docs` must still
be run (see Commands).

## Technical fixes

Design bugs found by actually running pilots (each: file, wrong behaviour,
correction, verification):

1. `train_gui_agent.py` -- CPU thread oversubscription. Symptom: ~0.3 s per
   episode, full run projected >1 h. Measured 25 ms vs 0.04 ms per tiny-MLP
   forward with 16 vs 1 torch threads. Fix: `torch.set_num_threads(1)` (+
   interop) on the CPU path. Verified: smoke 4x96 eps in 3.3 s; full
   36000 eps in 213.5 s.
2. `train_gui_agent.py` -- advantage standardisation used `torch.std()`
   (unbiased, N-1), so the hand-computed oracle test expected +/-1 and got
   +/-0.7071. Fix: population std (`unbiased=False`), matching the numpy
   convention. Verified by `ReinforceLossTest.test_exact_oracle` passing.
3. `gui_env.py` -- wrong submit was terminal. Pilot result: REINFORCE
   _suppressed_ submitting (early submits cut off progress collection) and
   the +1 was never discovered (shaped 0.00 after 800 eps). Fix: wrong
   submit returns a form error and the episode continues (real-form retry,
   Ch.22 "recover from feedback"); only correct submit or timeout ends it.
   Verified: shaped/curriculum subsequently learned.
4. `train_gui_agent.py` -- success replay imitated whole lucky trajectories
   (~14 junk steps + 2 good ones); BC loss fell to 0.01 = memorisation with
   zero transfer. Fix: process-level filter -- imitate only steps that
   raised the satisfied sub-goal count to a new best or were the winning
   submit (Ch.22 Step 7 / Ch.17 process-signal idea). Verified: curriculum
   began promoting (level 1 at ep ~1000-1300).
5. `train_gui_agent.py` -- BC with shared Adam lr=3e-3 memorised the tiny
   early buffer before it held enough layouts to generalise from. Fix: gate
   replay on >=32 successes, separate Adam lr=1e-3, 2 passes/update over
   <=256 sampled good steps; clear buffer on curriculum promotion.
   Verified: level-0 stochastic success climbed past the 50% gate;
   promotions at eps 1000/4000/7500 in the final run.
6. Hyperparameters set by pilot measurement: `MAX_STEPS` 12 -> 16, batch
   8 (was 16), entropy coef 0.05, hidden 64 (was 128), promotion gate
   0.6 -> 0.5 (chapter's 30-70% band). Final budgets: sparse 3000,
   curriculum 15000, shaped 15000, hackable 3000.

Other corrections while writing: `OBS_DIM` asserted by test (204);
hackable bonus documented as per-click farmable (ceiling 16x0.39=6.24,
run reached 6.17); curriculum submit share 0.565 interpreted as
submit-as-free-check (wrong submits cost nothing in sparse reward).

## Other changes

- New: `code/chapter22_computer_use/{gui_env,train_gui_agent,test_smoke}.py`,
  `requirements.txt`, `README.md` (with reference results table).
- New: `docs/chapter22_computer_use/lab.md` (front matter
  `22.L Lab: Form-Filling GUI Agent`; sections per spec; results are the
  real seed-0 numbers above; no image files added).
- No existing docs/config/code files edited (chapter pages owned by another
  worker; sidebar untouched).

## Needs external verification

None. No papers, benchmark numbers, quotes, or URLs were invented: all
linked pages exist in-tree, and all reported numbers are from the
author's own seed-0 run (reproducible via `--seed 0`). One observation,
not a claim: held-out success slightly exceeds train success (0.39 vs
0.33 curriculum) -- within 64-seed eval noise, reported as "matches
within noise".

## Sidebar additions

- Title: `22.L Lab: Form-Filling GUI Agent`, link: `/chapter22_computer_use/lab`
  (insert after `22.2 Prompt Injection and Instruction Hierarchy` in the
  `22. Computer Use and GUI Agents` group). Coordinator to add (config.mjs
  out of scope).

## Commands run and results

- `.venv/bin/pytest code/chapter22_computer_use -q` -- 22 passed,
  7 subtests passed, 12.81 s (includes `--smoke` subprocess run).
- `.venv/bin/python code/chapter22_computer_use/train_gui_agent.py --smoke --device cpu` -- exit 0, ~3.3 s.
- `.venv/bin/python code/chapter22_computer_use/train_gui_agent.py --device cpu` (full, seed 0) -- exit 0, 213.5 s; outputs in `runs/chapter22_computer_use/` (gitignored); summary table as in Summary.
- Pilots: `--mode shaped/curriculum/hackable` with reduced episodes (diagnosis above); inline `python -c` BC probes (oracle demos -> 99% level-0, proving capacity; motivated replay volume).
- `npx prettier --write` on the new/changed markdown files -- lab.md and
  ch22-lab.md reformatted, README unchanged (`.py` files have no prettier
  parser; repo convention covers them via nothing -- same as other chapters).
- `npx vitepress build docs` -- exit 0, build complete in ~60 s (unlisted
  lab.md builds fine without a sidebar entry).
- `node scripts/check-code-links.mjs` -- "Repository links OK (90 checked)".

## Open issues

1. One transient SIGSEGV observed once after a 3000-ep pilot completed
   (all output printed; exit path only). Never reproduced across ~10
   subsequent runs incl. the 213 s full run and pytest. Suspect
   torch/MKL shutdown flake; no action taken -- flagging for awareness.
2. Shaped was still climbing at episode 15000 (0.05 -> 0.14 over the last
   3000); a larger budget would raise it further. Kept 15000 to match the
   curriculum budget for a fair comparison.
3. Curriculum greedy submit share is 0.565 (submit-as-free-check under free
   retries). Documented as intended reading; exercise 4 explores charging
   for wrong submits.

## Review response

Reviewer report: `planning/reviews/ch22-lab.md` (6 errors, 8 doubts).
Each was re-checked by re-reading the code/page and recomputing where
applicable; all fixes below are in the original scope
(`code/chapter22_computer_use/`, `docs/chapter22_computer_use/lab.md`).

- Error 1 (hackable bonus described as "every submit click, right or wrong"): accepted. The `step` failure branch is the only place `SUBMIT_BONUS` is added, so a correct submit pays only +1. Fixed the `gui_env.py` module docstring, the inline comment, the `train_gui_agent.py` docstring, and the lab page reward-modes paragraph; added `test_correct_submit_hackable_pays_no_bonus` asserting a correct hackable submit pays exactly `SUBMIT_REWARD - STEP_PENALTY`. Verified: full `pytest` green (23 passed, 7 subtests).
- Error 2 (results-table "mean reward" mixes training mean under a greedy header): accepted. `mean_reward` is computed from the last 200 stochastic training episodes, not the greedy snapshot (the hackable greedy value would be exactly 6.24, not 6.173). Renamed the column to "mean train reward (last 200 eps)" in `lab.md` and `README.md`, and clarified the `print_table` caption to state the definition. The printed numbers are unchanged -- only the header/definition was wrong, matching the reviewer's diagnosis.
- Error 3 (level-0 luck "~10-15%"): accepted. Re-ran the measurement: uniform-random policy over 20000 episodes gives 8.2% level-0 success, confirming the reviewer's 8.2%. Page now says "~8%".
- Error 4 (exercise 2 asks to break a shaped run via `SUBMIT_BONUS`, which shaped never reads): accepted. Reworded to target only the hackable run and fixed the spam arithmetic to 16 x (bonus - 0.01), with the crossing at bonus ~0.12-0.13 (mean optimal solve 1.71 measured over 500 tasks, min 1.47, max 1.95).
- Error 5 (exercise 3 cites a "30-line BC probe in the report" that does not exist): accepted. The hint now describes the pilot observation directly (net fits the shared rule only once the buffer holds hundreds of clean steps from dozens of layouts) without pointing at a nonexistent artifact.
- Error 6 (page not in sidebar; "22.L" numbering): partially accepted but not changed -- `docs/.vitepress/config.mjs` is outside my scope, so the coordinator must still add the entry. Note the "22.L" front matter/H1 follows the task spec verbatim (`title: 22.L Lab: <short title>`); renumbering to "22.3" as suggested conflicts with that spec, so left for the coordinator to decide when adding the sidebar line (proposed: `22.3 Hands-on: Form-Filling GUI Agent`, link `/chapter22_computer_use/lab`).
- Doubt 1 (eval "train" seeds disjoint from training seeds): accepted as a wording fix. Confirmed: training uses 1M+ seeds, eval uses 500k/900k blocks. Column names and numbers kept (results must stay reproducible); added a sentence stating both eval blocks are disjoint from training seeds, so "train" is itself a generalisation check.
- Doubt 2 (optimal solve "~1.9"): accepted. Measured the optimal action sequence over 500 level-3 tasks: mean 1.71, min 1.47, max 1.95 ("uncheck" targets are satisfied at reset and never pay progress, hence the spread). Page now reads "averages ~1.7, up to ~1.95".
- Doubt 3 ("exactly the Chapter 6 update"): accepted. The lab adds batch advantage standardisation and an entropy bonus on top of the Chapter 6 baseline update; wording fixed.
- Doubt 4 (Step 9 "verifier failure" label): accepted. In the hackable regime the verifier is correct and the reward is mis-specified; rephrased to "a mis-specified reward of the kind Step 9 asks you to diagnose".
- Doubt 5 (Step 6 30-70% band): accepted. Clarified that the lab borrows the band as a level-promotion rule while the chapter uses it as a task-sampling filter, citing the chapter's own "experimental teaching setting" disclaimer.
- Doubt 6 (`training_curves.png` only for multi-mode runs): accepted. Setup paragraph now says the PNG is written only for multi-regime runs.
- Doubt 7 ("deterministic under --seed"): accepted. Softened to "reproducible under --seed on the same machine" in both `lab.md` and `README.md`.
- Doubt 8 ("uncheck" box targets need no action): accepted. Added an explicit note (2-4 payable sub-goals per episode depending on sampled targets) in the reward-modes paragraph and qualified the "five correct actions" sentence.

No reported numbers changed: the full-run table, promotion episodes, and CSV-confirmed narrative figures are untouched, consistent with the reviewer's own reproduction of the identical summary table.
