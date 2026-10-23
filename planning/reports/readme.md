# Report: `readme` — top-level README refresh

## Summary

Rewrote `README.md` and updated `code/README.md` so both describe the
finished book accurately: published site link near the top, a chapter
table matching the sidebar in `docs/.vitepress/config.mjs` (docs dir,
code dir, lab page, summary page per chapter), `uv`-based code setup
with the pinned lock file, and all stale bilingual/worker-process text
removed. `npx prettier --write` on both files, `npm run
verify:code-links` (129 links OK) and `npx vitepress build docs`
(success) all pass.

## Technical fixes

No math or code claims were changed; this task is documentation-only.
Factual corrections made while rewriting (each verified against the
working tree):

1. **File:** `README.md` — **wrong claim:** chapters 4, 10, 11, 16, 18,
   20, 21, 22, 24, 25, 26 have no runnable code (`–` in the Code column).
   **Correction:** every chapter 1–26 now has a `code/chapterNN_*`
   directory (verified with `ls code`), so all show a code link.
   Verified by listing each directory's scripts.
2. **File:** `README.md` — **wrong claim:** `code/online-experiments/`
   and `modelscope-space/` are both under `code/`. **Correction:**
   `modelscope-space/` lives at the repository root; only
   `code/online-experiments/` is under `code/`. Verified with `ls`.
3. **File:** `README.md` — **wrong claim:** code setup is
   `python -m venv` + per-chapter `pip install`. **Correction:** `uv
venv --python 3.12 .venv` + `uv pip install -r
code/requirements.lock.txt`, then `.venv/bin/pytest code -q`.
   Verified `code/requirements.lock.txt` exists and `.venv/bin/python`
   exists.
4. **File:** `README.md` — **wrong claim:** smoke suite is “232 tests”.
   **Correction:** `.venv/bin/pytest code --collect-only -q` collects
   **234 tests** in this snapshot. Verified by running the collection
   (3.29 s).
5. **File:** `code/README.md` — **stale claim:** “There is one
   `test_smoke.py` per chapter directory”. **Correction:** classic
   chapters use `test_smoke.py`; the 11 lab chapters use
   `test_chNN_lab.py` with additional hand-computation unit tests.
   Verified with a `code/**/test_*.py` listing (16 `test_smoke.py`
   including appendix A, 11 `test_chNN_lab.py`, plus
   `test_gridworld_q_learning.py`).
6. **File:** `code/README.md` — **stale scaffolding:** 14
   “(unreferenced — suggested: …)” markers and the section header
   “LLM chapters (maintained by another worker — described here, not
   edited)”. **Correction:** replaced the header with “LLM, agentic
   and frontier chapters”; resolved four markers to real, verified
   docs references found with `git grep` in `docs/` —
   `two_armed_bandit.py` → `/chapter02_rl_basics/review`,
   `dqn_cartpole.py` and `double_dqn_cartpole.py` →
   `/chapter05_dqn/review`, `ppo_td3_sac_comparison.py` →
   `/chapter09_continuous_control/review`, `multi_agent_marl.py` →
   `/chapter12_exploration_marl_hierarchical/review` — and reduced
   the remaining ten (incl. both appendix A scripts, confirmed
   unreferenced via `git grep`) to “—” with a legend explaining “—”.
7. **File:** `code/README.md` — **missing coverage:** lab chapters
   (4, 10, 11, 16, 18, 20, 21, 22, 24, 25, 26) absent from the index.
   **Correction:** added a “Lab chapters” table with each directory's
   scripts (verified with `ls code/chapterNN_*`) pointing at its
   `/chapterNN_*/lab` page.
8. **File:** `code/README.md` — **stale process text:** “You cannot
   install packages at review time” and `uv venv .venv` without a
   Python pin. **Correction:** user-facing wording, `uv venv --python
3.12 .venv`, lock-file documented in intro and Notes.
9. **File:** `README.md` — removed the stale “Compared with upstream”
   bullet about removed Chinese originals / `/en/` locale, per the
   task's bilingual-mention cleanup. (No Vercel or release
   instructions were present in either README.)

## Other changes

- `README.md`: published site link
  (`https://saisandeshk.github.io/hands-on-modern-rl-en/`, matching
  `package.json` repository `saisandeshk/hands-on-modern-rl-en`) as a
  bold line under the title; new “Getting started” heading; Lab/Review
  columns with ✓ links (labs: chapters 4, 10, 11, 16, 18, 20, 21, 22,
  24, 25, 26 — verified against `docs/**/lab.md`; reviews: all 26
  chapters — verified against `docs/**/review.md`); Appendix D row
  notes the notation page (`docs/appendix_d_math/notation.md`,
  verified present). WalkingLabs attribution and CC BY-NC-SA 4.0
  license section kept unchanged.
- `code/README.md`: intro now mentions per-lab-chapter `README.md`
  files (verified present in all 11 lab code dirs) and the lock file.

## Needs external verification

None. All cross-references were checked against files in this
snapshot (sidebar, `lab.md`/`review.md` listings, `ls` of code dirs,
`git grep` of docs references, pytest collection).

## Sidebar additions

None (no pages added; README files are not sidebar pages).

## Commands run and results

- `npx prettier --write README.md code/README.md` — pass (both files
  reformatted, table pipes realigned).
- `npm run verify:code-links` — pass (“Repository links OK (129
  checked)”).
- `npx vitepress build docs` — pass (build complete in 23.74 s; only
  pre-existing chunk-size / jinja2-fallback warnings).
- `.venv/bin/pytest code --collect-only -q` — 234 tests collected.
- `.venv/bin/pytest code -q` (full suite) — 231 passed, 2 skipped
  (Atari ROMs / `pyboy`, skipped with reasons), 1 failed:
  `code/appendix_a_engineering/test_smoke.py::test_smoke[debug_reward_hacking.py]`
  exits with code -11 (SIGSEGV) after writing its diagnostic report
  and plot; stderr shows only matplotlib `findfont` warnings. The crash
  is in appendix A code, outside this task's scope — left untouched
  for the appendix A worker / coordinator.

## Open issues

- `test_smoke[debug_reward_hacking.py]` (appendix A) segfaults (exit
  code -11) in this environment after completing its report and plot.
  Appendix A code is outside this task's scope, so it was left
  untouched; flagging for the appendix A worker / coordinator. All
  other 233 collected tests pass or skip with reasons.
- The 234-vs-232 test-count drift is likely from other workers adding
  lab unit tests concurrently; the README states the count collected
  in this snapshot (234).
- The per-chapter “Pages” links in the top-level table still point at
  each chapter's first sidebar page (unchanged targets); if other
  workers rename pages, `npm run verify:code-links` only covers
  `docs/`, so root README links would need a manual re-check.
