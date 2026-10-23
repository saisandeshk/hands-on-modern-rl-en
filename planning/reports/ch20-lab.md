# Report: ch20-lab — Toy code-agent RL lab for Chapter 20

## Summary

Built the missing hands-on lab for Chapter 20 from scratch: a new
`code/chapter20_swe_agents/` package (14 toy Python function tasks with
visible + hidden tests, a subprocess sandbox verifier, a softmax policy with
a REINFORCE-with-baseline update, a `train.py` CLI comparing three reward
regimes, and a 12-test pytest suite), plus a new docs page
`docs/chapter20_swe_agents/lab.md`. The lab trains on visible-test pass
rates and evaluates on hidden tests; rewarding visible tests only produces
train 1.000 / hidden 0.491 with 0.537 policy mass on memorizing shortcuts,
while hidden-test or randomized-input rewards recover correct solutions
(hidden 0.980 / 0.998, hack mass ~0.002). Full run: 461 s on CPU. All 12
tests pass; `npx vitepress build docs` succeeds. No existing files were
edited.

## Technical fixes

No pre-existing code was fixed (the lab is new). The following design points
were verified by execution rather than assumed:

1. **File:** `code/chapter20_swe_agents/tasks.py` (candidate reward contract)
   **Risk:** a "correct" candidate failing a hidden test, or a hack passing
   one, would invalidate the whole reward-hacking demo.
   **Correction/verification:** `test_candidate_reward_contract` executes all
   14 tasks x 4 candidates x 2 suites in the real sandbox: every correct
   candidate scores 1.0/1.0, every hack/heuristic scores 1.0 visible and
   < 1.0 hidden, every wrong candidate scores < 1.0 visible. Boolean tasks
   (palindrome) were the hardest to keep strict; hidden inputs were chosen
   so both constant-True and first-last-character fallbacks fail at least one
   hidden test (verified: hack hidden rates 0.50-0.75 on that task, never 1.0).
2. **File:** `code/chapter20_swe_agents/policy.py` (REINFORCE update)
   **Risk:** sign/scale error in the policy-gradient loss.
   **Verification:** `test_reinforce_loss_matches_hand_computation` checks the
   loss against a hand-derived value (logits [1,0], rewards [1,0], baseline
   0.2 -> expected loss -0.00602 within 1e-6), and
   `test_reinforce_step_reinforces_positive_advantage` checks one SGD step
   raises the probability of the advantage-positive arm. Also fixed a real
   bug found during smoke testing: rewards/baselines on CPU vs logits on
   CUDA raised `RuntimeError`; `reinforce_loss`/`reinforce_step` now move
   tensors to the log-prob device.
3. **File:** `code/chapter20_swe_agents/sandbox.py` (verifier semantics)
   **Risk:** conflating infrastructure failure with test failure.
   **Verification:** unit tests cover pass/fail counting, exception-in-test
   (recorded per-test, not a crash), infinite loop (timeout flag at 2 s),
   and temp-cwd isolation (a file-writing probe lands in the sandbox temp
   dir, never the caller cwd).
4. **File:** `docs/chapter20_swe_agents/lab.md` (reported numbers)
   **Risk:** invented or stale result numbers.
   **Verification:** every number in the results tables is taken from the
   actual full run (`runs/chapter20_swe_agents/summary.json`,
   `metrics_*.csv`, seed 0, CPU, 461 s): visible 1.000/0.999/0.491/0.500
   resolved/0.537 P(hack); hidden 0.982/0.945/0.980/0.929/0.002;
   randomized 1.000/0.976/0.998/1.000/0.002. YAML front-matter title is
   quoted (unquoted `20.L Lab: ...` broke the VitePress build; fixed).

## Other changes

- Added `code/chapter20_swe_agents/README.md` (what/why/how, runtimes,
  expected-results table) and `requirements.txt` (torch/numpy/matplotlib,
  already in the venv).
- Notation in code and docs matches the chapter: $R_{\text{test}}$ pass
  fraction, $R_{\text{resolved}}$ resolved rate, $\pi_\theta$, baseline
  advantage. Internal links use only verified chapter-20 and chapter-19
  pages (no links to chapters whose pages are being edited by others).
- Docs page contains no image files per spec; results are markdown tables.

## Needs external verification

None. All paper/benchmark claims in the docs page are links to the chapter's
own pages (which attribute their sources); the lab's numbers are measured,
not cited. One known limitation is stated in both README and lab page: the
sandbox is a temp-dir + timeout isolation around fixed trusted snippets, not
a container boundary -- adequate for a CPU textbook lab, not a claim about
production agent sandboxes.

## Sidebar additions

- Title: `20.L Lab: Reward Hacking in a Toy Code Agent`, link: `/chapter20_swe_agents/lab` (new page; config untouched).

## Commands run and results

- `.venv/bin/pytest code/chapter20_swe_agents -q` -> **12 passed in 10.05 s**
  (includes sandbox unit tests, hand-computed REINFORCE check, candidate
  contract over all 56 sandbox executions, and a CPU smoke train in a
  subprocess).
- `.venv/bin/python code/chapter20_swe_agents/train.py --smoke --seed 0 --device cpu`
  -> 7.9 s, exit 0, visible-regime hacking already visible (E[hid] 0.352).
- `.venv/bin/python code/chapter20_swe_agents/train.py --seed 0 --device cpu`
  (full: 14 tasks x 300 epochs x 3 regimes) -> **461.0 s, exit 0**; results
  as in the table above; `runs/chapter20_swe_agents/` holds 3 CSVs,
  `summary.json`, 2 PNGs.
- `npx prettier --write` on the two new `.md` files -> clean (prettier has no
  parser for `.py`/`.txt` in this repo, so those were style-checked by hand
  against neighboring chapter code).
- `npx vitepress build docs` -> **build complete in 44.83 s** (only warning:
  pre-existing `jinja2` highlighting fallback and chunk-size notes).

## Open issues

- Seed-to-seed variation in the visible-regime row is expected (any mix of
  the three visible-optimal arms is a REINFORCE fixed point); the lab page
  documents the expected range (hidden ~0.4-0.5). A multi-seed table was
  omitted to keep the full-run cost to one ~8 min run.
- `runs/` output is gitignored per repo config, so the coordinator cannot
  inspect `summary.json`/CSVs from the worktree diff; the final numbers are
  duplicated in the README and lab page tables.
- Sidebar entry (`/chapter20_swe_agents/lab`) still needs the coordinator to
  add it to `docs/.vitepress/config.mjs`.
