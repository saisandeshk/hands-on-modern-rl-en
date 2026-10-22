# Report: ch04-lab — Hands-on Lab for Chapter 4 (Classical Methods)

## Summary

Built a hands-on lab for Chapter 4 (DP, MC, TD) in `code/chapter04_classical_methods/` plus the docs page
`docs/chapter04_classical_methods/lab.md`. The lab uses slippery 4x4 `FrozenLake-v1` and `CliffWalking-v1`
(v0 fallback), reads the exact transition model from `env.unwrapped.P`, and runs four experiments: (1) DP
oracle — iterative policy evaluation, policy iteration, value iteration with a residual stopping rule;
(2) MC first-visit / SARSA / Expected SARSA / Q-learning control against the oracle Q\*; (3) CliffWalking
SARSA-vs-Q-learning path comparison; (4) off-policy evaluation with ordinary vs. weighted importance
sampling. Every script takes `--seed`, `--device`, `--smoke`, writes CSV/JSON/PNG under
`runs/chapter04_classical_methods/`, and prints a summary table. An independent review (11 errors, 8 doubts)
was then worked through: every quantitative claim was re-derived or re-run, accepted findings were fixed,
rejected ones are justified below.

## Technical fixes

- `code/chapter04_classical_methods/common.py` — `load_model` docstring said it returns 4 values but it
  returns 5 (`resolved_id` included). Fixed the docstring. (Review error 11.)
- `code/chapter04_classical_methods/mc_td_control.py`, `cliffwalking.py` — SARSA bootstrapped from a
  hardcoded action `0` on truncation steps instead of the sampled next behaviour action, breaking its
  on-policy property exactly on time-limit cuts. Now the next behaviour action is sampled whenever the
  update bootstraps (both files); verified by re-running both full experiments. (Review error 10.)
- `docs/chapter04_classical_methods/lab.md` Experiment 2 — replaced the false headline ("greedy return rises
  toward 0.82 while oracle error falls") and the `RESULT` placeholders with measured numbers (MC
  0.1400±0.118 / 0.8516 / 0.0734; SARSA 0.3800±0.384 / 0.7957 / 0.1932; Expected SARSA 0.3800±0.384 /
  0.8486 / 0.1960; Q-learning 0.2300±0.353 / 0.7071 / 0.1337). Verified by running
  `mc_td_control.py --seed 0` (92 s). Also verified the failure mechanism: on seed 2 the learned Q
  overestimates the start state's "left" action (0.89 vs true 0.82), so the greedy policy walks into the
  wall and its exact DP-evaluated success probability is 0.0. (Review errors 2, 3.)
- `docs/chapter04_classical_methods/lab.md` Experiments 2 + "What to look for" — the shared
  `||Q − Q*||` yardstick is now labelled a convergence measure for Q-learning only; the on-policy methods
  target `Q^{eps-greedy}`, whose distance to Q\* is 0.5057 (start value 0.4175, both recomputed from the DP
  code: `Veps(s0)=0.417536`, `floor=0.505676`). Online return is given as the fairer on-policy metric, and
  the MC-vs-TD truncation asymmetry is now stated. (Review error 4, doubt 5.)
- `docs/chapter04_classical_methods/lab.md` Experiment 4 + `ope_is.py` docstring + README — filled the OPE
  table with measured numbers (OIS 0.7722/0.2437/−0.0513/0.2491; WIS 0.7949/0.1458/−0.0286/0.1486; ESS
  fraction 0.0226; 100 trials × 2,000 episodes). The "unbiased" claim is corrected: OIS is unbiased for
  full episodes but shows a −0.05 bias here (~2 SE over 100 trials) because gymnasium's 100-step truncation
  is not bootstrapped. Verified by running `ope_is.py --seed 0` (31 s). (Review errors 2, 5.)
- `docs/chapter04_classical_methods/lab.md` Background — the 4.3 bullet no longer claims that
  `reward-design.md` discusses CliffWalking (verified: zero occurrences of "cliff" there); it now points at
  the Coast Runners / Goodhart content that page actually contains. (Review error 6.)
- `code/chapter04_classical_methods/README.md` — fixed the unparseable table header with unescaped pipes
  (`|Q-Q*|` → `Final max oracle error`); filled the MC/TD, CliffWalking and OPE tables with measured
  numbers; corrected runtime estimates (see below); fixed the "approximately unbiased" OIS sentence.
  (Review errors 2, 7; doubt 1.)
- `docs/chapter04_classical_methods/lab.md` Experiment 1 — policy-iteration cell corrected to 0.823512
  (ran `dp_oracle.py`: PI V\*(s0)=0.8235125, VI V\*(s0)=0.8235061). (Review error 8.)
- `docs/chapter04_classical_methods/lab.md` + README — Experiment 2 preamble now says α=0.1 applies to the
  three TD methods and MC uses the incremental 1/N mean. (Review error 9.)
- `docs/chapter04_classical_methods/lab.md` Experiment 3 — refreshed numbers after the SARSA fix (SARSA
  −38.56±71.44 / −35.50±54.84 / 0.108 falls; Q-learning −56.62±90.20 / −13.00±0.00 / 0.362 falls),
  replaced the seed-0 greedy maps with the actual output (verified byte-identical to stdout), fixed "bottom
  row of the Q-learning map" → "row directly above the cliff", and added the seed-spread caveat.
  (Doubts 2, 7.)
- `docs/chapter04_classical_methods/lab.md` exercises — Ex.1 hint softened ("roughly geometric" →
  steadily shrinking, ratio not constant; ~95–100 sweeps per decade of θ); Ex.4 hint corrected (all-zero
  batches are occasional, not guaranteed, at eps 0.3/0.5). (Doubts 3, 4.)
- `docs/chapter04_classical_methods/lab.md` front matter — quoted the title (`title: "4.L Lab: ..."`):
  the unquoted colon is invalid YAML and `npx vitepress build docs` failed on it. Build now succeeds.
  (Found while verifying; pre-existing.)
- Runtime claims — measured on this machine (`--seed 0`, CPU): dp < 1 s, mc_td ~90 s, cliff ~2 s,
  ope ~31 s (≈2 min total); pytest smoke 2.6 s. README/lab now say "~90 s", "~5 s", "~35 s" and "a few
  minutes". (Doubt 1.)

## Other changes

- `mc_td_control.py` module docstring now notes the `||Q − Q*||` curve is a convergence measure for
  Q-learning while on-policy methods bottom out at a floor.
- `ope_is.py` module docstring notes the truncation-induced downward bias.
- `cliffwalking.py` comment block expanded to document the bootstrap-on-truncation rule.

## Needs external verification

None. All paper/benchmark-type claims were either removed or reduced to statements verified by running the
lab code (DP values, floor 0.5057, seed-2 exact 0.0, OPE bias). The Sutton & Barto Example 6.6 citation for
the cliff result is standard textbook knowledge kept as-is.

## Sidebar additions

Cannot edit `docs/.vitepress/config.mjs` (out of scope for this task). Coordinator, please add under
"4. Classical RL Methods": `{ text: '4.L Lab: Tabular Methods Against Exact Oracles', link: '/chapter04_classical_methods/lab' }`.
Note: the task spec mandates the `4.L` number in the front matter and H1, so the page keeps that number
even though `scripts/fix-heading-numbers.mjs` only recognises `N.M` patterns — heading sync for this page
will need manual handling.

## Commands run and results

- `.venv/bin/python code/chapter04_classical_methods/dp_oracle.py --seed 0 --device cpu` — pass; PI 617
  eval sweeps/2 improvements/residual 6.56e-07/V\*(s0)=0.823512; VI 431 sweeps/residual 9.61e-07/V\*(s0)=0.823506.
- `.venv/bin/python code/chapter04_classical_methods/mc_td_control.py --seed 0 --device cpu` (before and
  after the SARSA fix) — pass; post-fix table reported above; 81.5 s.
- `.venv/bin/python code/chapter04_classical_methods/cliffwalking.py --seed 0 --device cpu` (before and
  after fix) — pass; post-fix table and maps reported above; 2.3 s.
- `.venv/bin/python code/chapter04_classical_methods/ope_is.py --seed 0 --device cpu` — pass; table
  reported above; 31.1 s.
- `.venv/bin/python -c` checks — eps-greedy start value 0.417536 and floor 0.505676 recomputed from DP
  code; Q-learning seed-2 greedy policy exact success 0.0 via DP policy evaluation; seed-2 greedy map and
  overestimated Q[0] inspected.
- `.venv/bin/pytest code/chapter04_classical_methods -q` — 12 passed, 4 subtests passed in 2.56 s.
- `npx prettier --write` on the two Markdown files (the .py files have no prettier parser) — applied,
  tables reflowed only.
- `npx vitepress build docs` — succeeds (15.25 s) after the front-matter title quoting fix.

## Open issues

- `code/requirements.txt` (global merged summary) lists `gymnasium[classic-control,box2d,mujoco,atari]` but
  not the `toy-text` extra this lab's `requirements.txt` declares. Merging it is out of this task's scope;
  coordinator decision needed. (Review doubt 6; the chapter-local file is kept as-is since FrozenLake and
  CliffWalking are never rendered, so no pygame use occurs.)
- `docs/.vitepress/config.mjs` sidebar entry still missing (see "Sidebar additions" above).

## Review response

- Error 1 (sidebar missing / "4.L" number): partially accepted — sidebar entry needed but `config.mjs` is
  out of scope, listed under "Sidebar additions"; rename to "4.4" rejected because the task spec mandates
  the `4.L` title/H1.
- Error 2 (RESULT placeholders / TODO markers): accepted — all tables filled with measured numbers,
  markers removed, in `lab.md` and `README.md`.
- Error 3 (false "rises toward 0.82" headline): accepted — headline rewritten; post-fix full run gives
  greedy 0.14–0.51 and oracle error plateau 0.55–0.85; seed-2 exact-0.0 failure verified and described.
- Error 4 (shared oracle-error yardstick): accepted — `||Q − Q*||` now framed as Q-learning-only
  convergence measure; on-policy floor 0.5057 / start value 0.4175 recomputed and stated.
- Error 5 (OIS "unbiased" vs measured −0.05 bias): accepted — tables filled, text and `ope_is.py`
  docstring now attribute the bias to unbootstrapped 100-step truncation; no estimator change.
- Error 6 (4.3 CliffWalking cross-reference): accepted — bullet repointed to the Coast Runners / Goodhart
  content `reward-design.md` actually contains.
- Error 7 (README pipe-header table): accepted — header rewritten without inner pipes.
- Error 8 (PI cell copied VI value): accepted — PI cell now 0.823512 (both values re-measured).
- Error 9 (α listed for MC): accepted — both docs now say MC uses the 1/N mean, α is TD-only.
- Error 10 (SARSA truncation bootstrap bug): accepted — fixed in both files (sample next behaviour action
  whenever the update bootstraps), both full experiments re-run.
- Error 11 (`load_model` docstring arity): accepted — docstring now documents all five return values.
- Doubt 1 (inflated runtimes): accepted — replaced with measured times (90 s / 2 s / 31 s, ~2 min total).
- Doubt 2 ("bottom row" map description): accepted — reworded to the row directly above the cliff.
- Doubt 3 (Ex.4 hint overstates all-zero batches): accepted — hint now says occasional, not guaranteed.
- Doubt 4 (Ex.1 "roughly geometric" hint): partially accepted — softened; logarithmic half of the hint kept.
- Doubt 5 (MC truncation asymmetry unmentioned): accepted — asymmetry now stated in Experiment 2.
- Doubt 6 (global requirements lack `toy-text`): no change (out of scope) — noted under Open issues.
- Doubt 7 (SARSA spread hidden): accepted — seed-to-seed spread sentence added.
- Doubt 8 (report file missing): addressed — this file is the report.
