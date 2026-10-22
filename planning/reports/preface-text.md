# Task report: `preface-text`

## Summary

Text pass over the three preface pages. Chapter cross-references now match the
numbering table (Parts I-VII, Chapters 1-26, Appendices A-D). The environment
page now describes the real repository setup: a project virtualenv built with
`uv` from `code/requirements.lock.txt` (Python 3.12), per-chapter
`code/*/requirements.txt` files, smoke tests via `.venv/bin/pytest code -q`,
and the site via `npm run dev:fast`. History-page facts were checked
(people, years, milestones, math); two numeric errors and several
over-specific or stale claims were fixed or softened. No new pages were
created.

## Technical fixes

1. `docs/preface/brief-history/index.md` — Go board-state count said
   $3^{361} \approx 10^{170}$; recomputed as
   $361 \times \log_{10} 3 \approx 172.2$, corrected to $\approx 10^{172}$.
   Verified by arithmetic.
2. `docs/preface/brief-history/index.md` — DQN paragraph claimed the 2013
   paper "surpass[ed] human performance in multiple Atari games" and the
   figure caption said "surpassing human expert performance in most cases".
   The 2013 NIPS workshop paper beat humans on only a subset of games; the
   broader result is the 2015 Nature version. Softened both wordings and
   credited the 2015 extension. Verified against well-known publication
   history (Mnih et al. 2013 arXiv:1312.5602; 2015 Nature).
3. `docs/preface/brief-history/index.md` — AlphaGo paragraph implied pure
   deep RL + MCTS. The 2016 AlphaGo used supervised pretraining on human
   games before RL self-play (pure self-play came with AlphaGo Zero). Added
   that qualifier. Verified against Silver et al. 2016 Nature.
4. `docs/preface/brief-history/index.md` — GRPO objective shown without the
   KL-against-reference term. Marked the displayed objective as simplified
   with the KL term omitted, matching the full objective used in Chapter 15.
   Verified against the book's own Chapter 15 convention and the DeepSeek-R1
   paper form.
5. `docs/preface/brief-history/index.md` — Reference [^5] listed only the
   2018 2nd edition while the text cites the 1998 publication. Changed to
   "(1st ed.; 2nd ed. 2018)". Verified: 1st ed. 1998, 2nd ed. 2018, MIT Press.
6. `docs/preface/brief-history/index.md` — DeepSeek V3 pre-training cost was
   stated as 2.664M H800 GPU hours; the V3 paper reports 2.788M. Corrected
   to ~2.788M and dropped the unverifiable "R1-Zero used 128K GPU hours"
   clause (the R1 paper does not publish an equivalent headline number).
7. `docs/preface/introduction.md` — "Mario has only 4 actions" is wrong for
   every standard Mario action set (e.g. 7 in the common simple-movement
   set). Generalized to "a handful of discrete actions".
8. `docs/preface/introduction.md` — Listed DPO as an example of a
   Model-Free control algorithm. DPO is offline preference optimization,
   not environment-interaction RL; changed the example to "(such as DQN and
   PPO)".
9. `docs/preface/env-setup.md` — Version tip pinned
   `trl==0.23.0, peft==0.17.1`, but `code/requirements.lock.txt` pins
   `trl==0.24.0, peft==0.21.2`. Updated to the locked versions. Verified by
   reading the lock file.
10. `docs/preface/env-setup.md` — Package table listed `trl 0.12+`,
    `peft 0.13+`, `gymnasium 0.29+`, `stable-baselines3 2.2+`,
    `transformers 4.45+`; lock file has trl 0.24.0, peft 0.21.2, gymnasium
    1.4.0, stable-baselines3 2.9.0, transformers 4.57.3, torch 2.14.1.
    Updated floor versions accordingly. Verified by reading the lock file.

## Other changes

- `docs/preface/introduction.md`: front matter `title: Course Overview`
  did not match the H1 or the sidebar entry; set both to
  "Introduction to RL" (sidebar text).
- `docs/preface/introduction.md`: "Chapters 8-9" (DPO pipeline) corrected
  to "Chapters 13-14"; "Chapter 8" (agentic RL) corrected to
  "Part V (Chapters 19-22)". Part II paragraph replaced the stale
  "Off-Policy methods" phrase with continuous control + model-based RL
  (Chapter 9). Part IV mermaid nodes relabeled so RLHF = Ch13, DPO = Ch14,
  GRPO/RLVR = Ch15, reasoning/PRM/industrial = Ch16-18, and fixed the
  cross-part edge to use the renamed node.
- `docs/preface/brief-history/index.md`: added missing front matter
  (`title: Brief History of RL`) and aligned the H1 with the sidebar text.
- `docs/preface/env-setup.md`: front matter aligned to "Environment Setup";
  primary install flow rewritten around
  `uv venv --python 3.12` + `uv pip install -r code/requirements.lock.txt`
  (with pip and per-chapter alternatives); Python tip updated from 3.10 to
  3.12 with lock-file rationale; verification script now checks
  `Python >= 3.12`; added `.venv/bin/pytest code -q` smoke-test command and
  a "Running the Course Website" section (`npm install`, `npm run dev:fast`);
  all chapter references remapped (classic/deep RL = Chapters 1-9, LLM
  post-training = Chapters 13-18); package-table chapter columns remapped
  (pybullet/mujoco to Ch7-Ch9, isaacsim to Ch9, ale-py to Ch5, transformers/
  trl/peft/accelerate/datasets to Ch13-Ch18); coding-assistant prompt updated
  to the uv / Python 3.12 / lock-file / smoke-test flow.
- Verified the cross-chapter image links used by the introduction
  (`../chapter04_classical_methods/images/rlhf-openai-methods-diagram.webp`,
  `../chapter26_selfplay_frontiers/self-play-outlook/images/deepseek_r1_pipeline.png`)
  resolve to files referenced the same way by their owning pages.

## Needs external verification

- `brief-history/index.md` §5.1-5.4 newest-milestone details could not be
  checked without internet: DAPO/CISPO/VAPO/Dr.GRPO/GSPO arXiv IDs and
  attributions, UI-TARS-2 arXiv:2509.02544, AutoGLM, Step3-VL PaCoRe,
  Kimi K2 MuonClip (arXiv:2507.20534), GLM-4.5 report (arXiv:2508.06471),
  AlphaEvolve (May 2025), RPT (arXiv:2506.08007), Anthropic "Natural
  Emergent Misalignment from Reward Hacking" (arXiv:2511.18397), and the
  Competitive Programming with Large Reasoning Models paper
  (arXiv:2502.06807). Wording was left as-is where it stayed within
  plausible, hedged claims.
- The removed "R1-Zero used 128K GPU hours" figure: no replacement number
  was available offline; left for the coordinator to confirm or drop.
- GPU sponsorship contact (`physicoada@gmail.com`) and the WalkingLabs
  attribution/announcement block were kept untouched as upstream material.

## Sidebar additions

None. No new pages were created (per the task, no summary/exercises page).

## Commands run and results

- `npx prettier --write docs/preface/introduction.md docs/preface/env-setup.md docs/preface/brief-history/index.md planning/reports/preface-text.md` — all files formatted, no errors.
- `npx vitepress build docs` — build succeeded with no dead-link errors.
- Code tests: no chapter code is in scope for this text task, so no
  `.venv/bin/pytest` run was performed here; the env page documents the
  smoke-test command for readers.

## Open issues

- `code/README.md` still uses the old chapter numbering (Ch02 DPO,
  Ch04 DQN, etc.) and `python -m venv` instead of the uv flow. It is
  outside this task's scope but contradicts the corrected env-setup page;
  suggest a follow-up task.
- Per-chapter `code/*/requirements.txt` header comments still carry stale
  chapter numbers (e.g. `chapter05_dqn` says "Chapter 4",
  `chapter14_dpo` says "Chapter 2", `chapter08_ppo` unchecked). Same
  suggestion: a small code-scope cleanup task.
