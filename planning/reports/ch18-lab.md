# Report: ch18-lab — Rollout/Learner Systems and Policy Staleness Lab

## Summary

Continued the interrupted `ch18-lab` task without starting over. The prior run
had already created `code/chapter18_llm_rl_industrial/` (`rollout_system.py`,
`test_smoke.py`, `README.md`, `requirements.txt`) and
`docs/chapter18_llm_rl_industrial/lab.md`, all in good shape: 11/11 tests
passed on first check and the code was technically sound. What was missing was
the full run with real numbers in the lab page (only a
`<!-- RESULTS-FROM-RUN -->` placeholder existed). I finished the work:

- Ran the full configuration (`--experiment all --seed 0`) twice under
  `timeout`, confirmed the learning curves are exactly deterministic per seed,
  and filled the real tables into `lab.md` under a new "Results from a full
  run" section.
- Fixed content that the real results contradicted: the draft claimed
  truncation "recovers most of the baseline" / "fixes most of the damage",
  but at delay 4 all async variants collapse (sync 118.67 vs async 9–22) and
  clipping only visibly helps at delay 8 (clip 46.00 vs none 12.33). Lab page
  and README now report this honestly.
- Fixed bogus subsection citations (`18.4.1`, `18.4.4`, `18.1.3` — the chapter
  pages use titled sections, not these numbers) in `lab.md` and code
  docstrings, pointing at the real sections instead.
- Verified: `pytest` 11 passed, `npx vitepress build docs` succeeds,
  `npx prettier --write` clean.

## Technical fixes

1. **Overclaim about truncation corrected** (`docs/.../lab.md`,
   `code/.../README.md`). Wrong claim: "PPO-style clipping and V-trace-style
   truncation recover most of the sync baseline" / "fix most of the damage".
   Correction: at delay 4 (60×2048-step runs, seed 0) sync reaches 118.67
   while none/clip/vtrace reach 22.00/13.67/9.33 with `mean_rho ≈ 1.0`, so
   clipping rarely binds and single-seed noise dominates the ordering; the
   correction visibly helps only at sweep delay 8 (clip 46.00 vs none 12.33).
   How verified: two full runs under `timeout`; staleness/sweep tables
   reproduced exactly across repeats (deterministic per seed), so the ordering
   is the true output of this configuration, not a flaky run.
2. **Nonexistent subsection numbers removed** (`lab.md` ×2, `rollout_system.py`
   ×3: `18.4.1`, `18.4.4`, `18.1.3`). The chapter pages (`single-machine-to-
industrial.md`, `distributed-sync.md`) use titled `##` sections with no such
   numbering. Replaced with references to the actual sections (training-
   inference mismatch and asynchronous-scheduling comparison in 18.4, sync vs
   async in 18.1). Verified by grepping the chapter headings and confirming
   the cited facts exist there (staleness ratio formula in 18.1, AReaL
   `[0.8, 1.2]` token-level clip and ResourcePool/Worker/Driver in 18.4).
3. **Learner-utilization bullet qualified** (`lab.md`). The draft implied
   utilization rises strongly with K; the reference run shows 0.07–0.19
   (deliberately tiny synthetic update workload), so the bullet now says to
   read the trend, not the level. Verified from `summary.json`.
4. **Runtime note added** (`lab.md`): timed repeat took 231 s wall-clock;
   throughput timings vary run to run while learning curves are deterministic.

No code-logic bugs were found: `behaviour_version_at`, `LagTracker`,
`importance_ratios`/`ppo_clipped_weights`/`vtrace_truncated_weights` all check
against hand-computed values in the unit tests, and the 1-worker-equals-sync
test passes. (Minor: `discounted_returns` helper is defined but unused —
`collect_batch` inlines its own boundary-aware version. Left as is to keep the
diff minimal.)

## Other changes

- `lab.md`: added "Results from a full run" section with three real tables
  (throughput, staleness curves, sweep) plus honest interpretation paragraphs.
- `README.md`: "Expected results" rewritten to match the observed numbers.
- No new files created; no existing chapter pages touched.

## Needs external verification

None. All chapter-linked claims (staleness-ratio formula, AReaL clip range,
veRL abstractions, LlamaRL/AReaL/AgentRL comparison) were confirmed against
the chapter pages in this repo. The arXiv URLs cited in the lab's background
are the chapter's own links, not new ones.

## Sidebar additions

None — `lab.md` lives in the existing `chapter18_llm_rl_industrial` section;
coordinator to decide sidebar placement (no config edits allowed from here).

## Commands run and results

- `.venv/bin/pytest code/chapter18_llm_rl_industrial -q` → 11 passed (twice:
  before and after edits; ~22–31 s).
- `timeout 900 .venv/bin/python code/chapter18_llm_rl_industrial/rollout_system.py --experiment all --seed 0`
  → success; outputs in `runs/chapter18_llm_rl_industrial/` (CSVs, PNGs,
  `summary.json`).
- Timed repeat via `timeout 890 .venv/bin/python -c "..."` wrapper →
  `ELAPSED_S 231.0`, identical staleness/sweep tables, confirming determinism.
- `npx prettier --write <4 files>` → lab.md formatted; README unchanged;
  (prettier has no Python parser here, so the `.py` files errored with "No
  parser could be inferred" — expected, not a problem).
- `npx vitepress build docs` → success (twice, ~16–17 s).

## Open issues

- Single-seed (seed 0) results only; the clip-vs-none ordering at small delays
  is within CartPole noise. A multi-seed follow-up would strengthen the
  correction comparison but exceeds the ~15 min CPU budget per script.
- Throughput numbers are machine-specific (spawn-dominated at small K); the
  lab page notes they vary between runs.
