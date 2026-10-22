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

## Commands run and results

- `.venv/bin/python code/chapter21_deep_research/train_research_agent.py --smoke --seed 0 --device cpu` — OK (~3–15 s).
- `.venv/bin/python code/chapter21_deep_research/train_research_agent.py --seed 0 --device cpu` (canonical full run) — OK, 23.8 s. Results: heuristic 1.00; outcome 1.00 (5.0 steps); relevant 1.00 (5.0 steps, train return +1.25 vs +0.86); loose-any 0.44 (9.1 steps, train return +1.82, highest); scratch-outcome 0.12 (1.0 steps); scratch-relevant 0.12; random 0.00. Full table in `docs/chapter21_deep_research/lab.md` and `code/runs/chapter21_deep_research/`.
- Same full command with `--seed 1 --outdir /tmp/opencode/ch21-seed1` — OK; loose-any 0.81 / 6.8 steps (documents seed sensitivity for exercise 6).
- `.venv/bin/pytest code/chapter21_deep_research -q` — 19 passed.
- `npx prettier --write` on the markdown files (the repo's prettier has no
  Python parser; same as other chapters), then `npx vitepress build docs` —
  build succeeds with no dead links.

## Open issues

- The warm-started outcome and relevant regimes both saturate at 1.00 from
  the behavior-cloned start, so their learning curves are flat at ceiling;
  the shaping contrast rests on training-return density and the from-scratch
  controls. A harder corpus variant (more distractors, longer chains) could
  separate them on success rate — left as extension exercise material.
- Loose-shaping collapse magnitude is seed-sensitive (0.44 vs 0.81); the
  docs page reports the canonical seed-0 run and exercise 6 asks learners
  to quantify the variance.
