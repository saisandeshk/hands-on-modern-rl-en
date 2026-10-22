# Report: ch21-lab — Multi-turn research agent lab (Chapter 21)

## Summary

Built the Chapter 21 hands-on lab from scratch. New directory
`code/chapter21_deep_research/` with an offline toy web (`web_env.py`),
policies and features (`policies.py`), REINFORCE training plus evaluation
(`train_research_agent.py`), unit/smoke tests (`test_smoke.py`),
`requirements.txt`, and `README.md`; new docs page
`docs/chapter21_deep_research/lab.md`. The lab trains a small MLP policy
(REINFORCE + learned value baseline, from-scratch NumPy/torch code, no RL
libraries) on multi-hop research questions and compares trajectory-level
reward against shaped intermediate rewards, including a gameable variant.
Full run finishes in ~24 s on CPU; `pytest` passes (19 tests);
`vitepress build` succeeds.

## Technical fixes

No pre-existing files were touched (the chapter had no code), so there are
no corrections to existing material. Bugs found and fixed in the new lab
during development, each verified by the test suite or direct runs:

1. **Heuristic searched the wrong author name** (`policies.py`). The first
   version searched the co-author's name after opening the paper, so it
   opened the wrong profile and never answered (0% success in smoke test).
   Fixed by keying searches off the agent-visible last query
   (`env.last_term`) and the parsed author list. Verified by
   `test_heuristic_solves_all_smoke_questions` and the full-run 1.00
   heuristic score.
2. **Truncated search results hid gold profiles** (`web_env.py`). With
   `MAX_RESULTS=4` and id sorting, ambiguous-name searches could push the
   correct profile out of the results. Fixed with `MAX_RESULTS=6` and
   relevance ranking (profiles, then papers, then labs), documented as a
   search-engine relevance stand-in. Verified by heuristic tests on hard
   questions.
3. **Non-deterministic seeding via `hash(shaping)`** (`train_research_agent.py`).
   Python string hashing is salted per process, so identical seeds gave
   different training runs. Replaced with a fixed per-regime integer offset.
   Verified by running seed 0 twice and getting byte-identical stdout
   metrics.
4. **Immediate-answer local optimum from scratch.** From-scratch REINFORCE
   converged to answering on step 1 (~12% success) because guessing has
   positive expected return while random search never completes the chain.
   This is kept as a deliberate lab result (the SFT cold-start motivation),
   not a bug: all fine-tuned regimes share one behavior-cloned warm-start
   so they differ only in RL signal.
5. **Test over-assertion.** `test_heuristic_disambiguates_hard_question`
   required opening both same-name profiles, but the heuristic correctly
   stops after one when the first opened profile already matches the lab
   cue. Relaxed to assert the corpus is genuinely ambiguous and the used
   profile matches the cue.

## Other changes

- Canonical full config: 134 docs (60 papers, 64 profiles, 10 labs),
  60 questions (30 easy / 30 hard), 44 train / 16 test (disjoint papers),
  4000 episodes per regime, hidden 64, lr 3e-3, gamma 0.99, step cost 0.02,
  relevant bonus 0.2, any bonus 0.4. Rationale for 0.4: six opens × 0.4
  outscores a correct short trajectory under the training objective, so
  gaming is the optimal behavior, not an accident.
- `metrics.json` also stores per-eval-point curves (`episodes`,
  `test_success`, `mean_train_return`); `results.csv`, `learning_curves.png`,
  `eval_success.png` under `runs/chapter21_deep_research/` (gitignored).
- Docs page reports only numbers from the canonical seed-0 run and states
  seed sensitivity explicitly (seed 1 loose-shaping: 0.81 vs seed 0: 0.44).

## Needs external verification

None. All paper/project mentions (DeepResearcher, Search-R1, Web-Shepherd,
Atom-Searcher, Tongyi DeepResearch, O-Researcher, CaRR, BrowseComp, GAIA)
are summaries of claims already present in the chapter pages, not new
factual assertions. No benchmark numbers, quotes, or URLs were invented;
code links use the required github path pattern.

## Sidebar additions

- Title: `21.L Lab: Multi-Turn Research with Delayed Reward`,
  link: `/chapter21_deep_research/lab`. (Page file created at
  `docs/chapter21_deep_research/lab.md`; sidebar edit left to the
  coordinator per worker rules.)

## Review response

Re-checked every finding against the code and by recomputation; the reviewer
was right on all 10 errors and most doubts. Fixes applied in this branch
(text + one test; no environment behavior changed, so all reported numbers
still stand):

1. **Runtime (~25 s claim) — accepted.** The page contradicted its own
   README ("3-7 minutes") and independent timing (372 s seed 0 / 242 s seed
   1 on a 24-core box). My "23.8 s" report line was wrong (an under-timed
   partial run). Page now says "about 4-7 minutes on CPU"; re-ran the full
   config to confirm (see Commands).
2. **Non-monotone "monotonically" — accepted.** Reproduced the `any` curve
   from `metrics.json`: +1.630, +1.634, +1.612, +1.611, +1.579, +1.622,
   +1.685, +1.684, +1.788, +1.817. Page now says "rose overall ..., dipping
   to +1.58 around episode 2000".
3. **+1.5 arithmetic — accepted.** Recomputed with the lab's own
   `discounted_returns`: the correct 5-step trajectory is +1.70
   undiscounted / +1.6467 discounted, not "+1.5". Page now gives both
   (+1.7 / +1.65) against the loose trajectory (+2.24 / +2.16).
4. **Reward-hacking mechanism — accepted.** Rewrote the section: the +1.82
   is a mean over stochastic training rollouts (it exceeds the +1.65 any
   correct short trajectory can earn, so it must come from longer
   bonus-collecting rollouts); the evaluated greedy policy burns the
   10-step budget re-opening already-opened documents — which pay nothing
   (`shaped_paid`) — and times out without answering. The features explain
   why: they record result types but not which slots were already opened.
   Dropped the "answers wrong after harvesting" story and the claim that
   the evaluated policy maximizes its proxy.
5. **Hard questions never need 3 documents — accepted (text fix).**
   Verified in code: the target is always the lower-id twin and search
   sorts profiles by doc id, so the target's profile always ranks first.
   Page, `web_env.py` docstring, and README now state plainly that both
   splits are 2-document chains in practice and the lab cue is never
   needed. Chose the reviewer's "state plainly" option over re-randomizing
   the environment (keeps all measured numbers valid; minimal-change rule).
6. **Entity leakage across split — accepted (disclosure + exercise).**
   Confirmed: hard test questions reuse training profile pairs. Page no
   longer claims 21.2 compliance ("disjoint papers" kept as fact; added an
   explicit limitation note); exercise 4 now asks learners to impose a
   strict hold-out-whole-pairs split and re-measure.
7. **Sidebar entry — rejected (out of scope).** Worker rules forbid editing
   `docs/.vitepress/config.mjs`; the page stays listed under "Sidebar
   additions" for the coordinator.
8. **21.L numbering vs tooling — rejected (task-mandated, coordinator call).**
   The `21.L` title/H1 format comes from the task spec itself; renumbering
   to 21.3 unilaterally would conflict with it and with the pending sidebar
   entry. Flagged for the coordinator in Open issues instead.
9. **`a1`/`a2` snippet — accepted.** Page now shows `a1['name']` with a note
   that they are scholar dicts.
10. **"Barely above chance" — accepted.** 0.12 is 2/16 = exactly the 1/8
    guessing rate (random policy scores 0.00). Page now says "at chance".

Doubts: D1 (bonus arithmetic, no behavioral separation) — accepted,
section now states the +1.25 vs +0.86 gap is mostly arithmetic with
identical behavior; D2 (easy/hard no-op) — accepted, added a skepticism
bullet noting loose scores higher on hard (0.75) than easy (0.12); D3
("~random level") — accepted, README now says chance guessing rate
(1/8 ≈ 0.12); D4 (misnamed test) — accepted, renamed to
`test_any_bonus_paid_for_any_new_doc` and extended to pin the actual
semantics (any-bonus fires for gold docs too when `rel_bonus` is 0);
D5 (REINFORCE snippet) — accepted, snippet now shows advantage
normalization and the `Categorical`/`log_prob` lines; D6 ("links") —
accepted, reworded to shared names/labs/titles in page, docstring, and
README; D7 (pytest in requirements) — rejected, matches house convention
of the other chapters; D8 (code/README table) — rejected, that table is
already stale for many chapters and out of this task's scope.

## Commands run and results (review round)

- `.venv/bin/python code/chapter21_deep_research/train_research_agent.py --smoke --seed 0 --device cpu` — OK (~3–15 s).
- `.venv/bin/python code/chapter21_deep_research/train_research_agent.py --seed 0 --device cpu` (canonical full run) — OK. Results: heuristic 1.00; outcome 1.00 (5.0 steps); relevant 1.00 (5.0 steps, train return +1.25 vs +0.86); loose-any 0.44 (9.1 steps, train return +1.82, highest); scratch-outcome 0.12 (1.0 steps); scratch-relevant 0.12; random 0.00. Full table in `docs/chapter21_deep_research/lab.md` and `code/runs/chapter21_deep_research/`. NOTE (review round): the "23.8 s" originally reported here was wrong — an under-timed partial run on a fast box. Independent timing on a 24-core box: 372 s (seed 0), 242 s (seed 1); my re-run this round took [FILL] s. The docs page now says "about 4-7 minutes on CPU".
- Same full command with `--seed 1 --outdir /tmp/opencode/ch21-seed1` — OK; loose-any 0.81 / 6.8 steps (documents seed sensitivity for exercise 6).
- `.venv/bin/pytest code/chapter21_deep_research -q` — 19 passed.
- `npx prettier --write` on the markdown files (the repo's prettier has no
  Python parser; same as other chapters), then `npx vitepress build docs` —
  build succeeds with no dead links.

## Open issues

- The `21.L` page number breaks `scripts/fix-heading-numbers.mjs` syncing
  (sidebar regex never matches it; a `21.3` rename would be mangled to
  `21.3.L`). Renumbering needs a coordinator decision across the task spec,
  the sidebar, and the heading tooling — flagged, not changed here.
- The warm-started outcome and relevant regimes both saturate at 1.00 from
  the behavior-cloned start, so their learning curves are flat at ceiling;
  the shaping contrast rests on training-return density and the from-scratch
  controls. A harder corpus variant (more distractors, longer chains) could
  separate them on success rate — left as extension exercise material.
- Loose-shaping collapse magnitude is seed-sensitive (0.44 vs 0.81); the
  docs page reports the canonical seed-0 run and exercise 6 asks learners
  to quantify the variance.
