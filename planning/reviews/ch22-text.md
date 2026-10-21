# Review of ch22-text

Scope: `git diff english-only...w/ch22-text` (4 files: new `docs/chapter22_computer_use/review.md`, edits to
`docs/chapter22_computer_use/training.md`, one heading/link edit in `docs/chapter22_computer_use/safety-swarm.md`,
and the writer's report `planning/reports/ch22-text.md`). All eight exercise solutions in the new `review.md` were
re-derived numerically; all of them reproduce the written numbers. Findings below are in priority order.

## Errors

1. `docs/.vitepress/config.mjs` (not modified in this branch) — the writer's report claims "Sidebar additions | Title:
   `22.R Summary and Exercises`, link: `/chapter22_computer_use/review`" — `git diff english-only...w/ch22-text --
   docs/.vitepress/config.mjs` is empty, and the Chapter 22 block of the sidebar still contains only
   `22.1 GUI Agent Training` and `22.2 Prompt Injection and Instruction Hierarchy`. The new page
   `docs/chapter22_computer_use/review.md` is therefore not reachable from the navigation, which violates the repo
   rule in `AGENTS.md` ("Every navigable page must be listed there"). `scripts/generate-sitemap.mjs` scans every
   `docs/**/*.md`, so the page still appears in `sitemap.xml` as an orphan, and `scripts/fix-heading-numbers.mjs`
   (which syncs section numbers from the sidebar) will never touch it. The report is factually wrong about the state
   of the branch. — Suggested fix: add
   `{ text: '22.3 Summary and Exercises', link: '/chapter22_computer_use/review' }` to the Chapter 22 sidebar block.

2. `docs/chapter22_computer_use/review.md` — "title: 22.R Summary and Exercises" / "# 22.R Summary and Exercises" —
   the section-number convention in this book is `N.M Summary and Exercises` (sidebar entries 1.4, 2.4, 3.4, 4.4, 5.6,
   7.8, 10.4 all use it, and `docs/chapter01_cartpole/review.md` … `docs/chapter10_offline_rl/review.md` all carry
   `N.M` front-matter titles). Chapter 22 currently has 22.1 and 22.2, so the review page is 22.3; `22.R` is a letter
   where the rest of the book uses a digit, and because the page is missing from the sidebar (error 1) the heading
   fixer cannot repair it. — Suggested fix: rename to `22.3 Summary and Exercises` in both the front matter and the
   H1, and use `22.3` in the sidebar entry.

3. `docs/chapter22_computer_use/training.md` (new subsection "Test-Time Compute and Invalid Traces") — "sample $K$
   candidate clicks, keep the ones that fall inside the target element's bounding box (or carry high grounding
   confidence), and execute their average" — the filter is circular: knowing "the target element's bounding box" is
   exactly the quantity the grounding head is supposed to predict (Step 2: "the model might output empty space"),
   so the policy cannot compute this filter on its own. It is executable only with an external detector, i.e. the
   Set-of-Mark pipeline that Step 2 explicitly frames as the agent's dependency ("the agent is helpless when the
   detector misses elements"). And in that case the scheme is dominated by the trivial alternative: if a detector
   supplies the box, clicking the box center is at least as good as averaging $K$ in-box samples, because the mean of
   in-box samples is just a noisier estimate of that center. If the "box" is a container region rather than the
   interactive widget, the averaged coordinate can also land on dead pixels, so averaging does not even guarantee
   validity. The same objection applies to `review.md` Exercise 8, which hands the function the ground-truth box.
   — Suggested fix: state the mechanism honestly — e.g. filter by the detector's confidence or by the modal cluster
   of the $K$ sampled coordinates (self-consistency), and note explicitly that with a SoM detector the box center is
   the cheaper baseline that the vote must beat.

4. `docs/chapter22_computer_use/training.md` (Step 7, missed by the writer) — "Answer with a score in [0, 1]:" with the
   rubric line "- -0.5: regression (e.g., closed important dialog)" — the stated range and the rubric contradict each
   other: a judge constrained to $[0,1]$ cannot emit $-0.5$, so the dense progress term $r_{\text{progress}}(s_t,
   s_{t+1})$ is structurally unable to express backtracking and the negative anchor is dead code. This also propagates:
   `review.md` Exercise 5 uses only non-negative progress scores $[0.0, 0.5, 0.5, 1.0]$ and its shaped return
   $G_0 = 0.9(0.25) + 0.81(0.25) + 0.729(1.5) = 1.521$ silently assumes the $[0,1]$ range. — Suggested fix: change
   the prompt to "Answer with a score in [-1, 1]" (keeping the four anchors), or delete the $-0.5$ anchor.

5. `docs/chapter22_computer_use/training.md` (Step 6) — "$\text{Curriculum}(\pi_\theta) = \arg\max_{\text{task } \tau}\;
   \text{Difficulty}(\tau)\quad\text{s.t.}\quad 0.3 \le P_\theta(\text{success}\mid\tau) \le 0.7$" — the equation and
   the algorithm it is supposed to define disagree. The `argmax` selects a single hardest in-band task, while the
   surrounding prose ("This concentrates sampling on tasks with an intermediate success rate"), the code below it
   (`weights = [t.difficulty for t in candidates]; return weighted_sample(candidates, weights, batch_size)`) and the new
   `review.md` Exercise 3 all implement proportional sampling over the whole band (weights $[10,20,30] \to
   $[1/6,1/3,1/2]$). A reader who implements the displayed `argmax` gets a curriculum that always trains one task.
   — Suggested fix: write $\text{Curriculum}(\pi_\theta)=\{\tau:0.3\le P_\theta(\text{success}\mid\tau)\le0.7\}$ with
   $\Pr(\tau)\propto\text{Difficulty}(\tau)$, matching the sampler.

6. `docs/chapter22_computer_use/training.md` (Step 7) — "This LLM-as-judge approach for intermediate rewards is similar
   to the idea in [Chapter 17: Process Reward Model](../chapter17_prm_search/inference-time-search)" — the link target is
   §17.5 "Inference-Time Search", not a process-reward-model section; Chapter 17's PRM material is 17.1 "Outcome and
   Process Rewards", 17.2 "Discriminative PRMs", 17.3 "Generative PRMs" and 17.4 "Formal Verifiers". The link text names
   a page that does not exist. The writer fixed three other link labels on this page (19.4 and 22.2) and missed this
   one. — Suggested fix: link to `../chapter17_prm_search/discriminative-prm` (or `formal-prm`) and use that section's
   name.

7. `docs/chapter22_computer_use/training.md` — the writer removed the $\gamma$/illegal-action-weight collision but left
   the second instance of the same defect on the same page: $\beta$ is the grounding-loss weight in the Step 2
   objective "$-\beta \cdot \mathcal{L}_{\text{ground}}(\theta)$" (never assigned a value) and the recovery-reward
   weight in the Step 4 decomposition "$+\beta r_{\text{recovery}}-\delta r_{\text{invalid}}$" (stated as
   $\beta = 0.3$). The same chapter also uses $\beta$ as the hierarchy/safety weight and as the DPO temperature in
   `safety-swarm.md`. A reader who takes the page's only stated $\beta$ value ($0.3$) as applying to the joint
   objective gets the wrong mixing weight for SFT+RL. — Suggested fix: rename the recovery weight (e.g. $\rho$) or
   state "$\beta$ (grounding weight, distinct from the recovery weight in Step 4)".

8. `planning/content-review-notes.md` (not modified) — the writer fixed five factual/cross-reference errors in this
   branch, but `AGENTS.md` requires "When fixing a factual error, record it (and the fix) in
   `planning/content-review-notes.md`"; the file is untouched in the diff and has no Chapter 22 section. — Suggested
   fix: add the five fixes (Set-of-Mark action order, PPO-objective mislabel, $\gamma\to\delta$, 19.4/22.2/25
   cross-references) to the notes file with the fix description.

## Doubts

1. `docs/chapter22_computer_use/review.md`, Exercise 2 — "RMSE $= \sqrt{0.0029} \approx 0.0539$, about $5.4\%$ of the
   screen width" — the loss arithmetic is right ($0.05^2 + 0.02^2 = 0.0029$, gradient $2(\hat p - p) = (0.10,-0.04)$,
   pixels $0.05\times1920 = 96$ and $0.02\times1080 = 21.6$), but the normalized 2-D norm mixes axes that are scaled
   differently on screen: the actual on-screen offset is 96 px in $x$ (5.0% of the width) and 21.6 px in $y$ (2.0% of
   the height), Euclidean 98.4 px = 5.1% of the width but 9.1% of the height. "5.4% of the screen width" is a
   convenient but inaccurate gloss. — Consider: "5% of the width and 2% of the height".

2. `docs/chapter22_computer_use/training.md` — "The primitive actions for Anthropic Computer Use are defined as follows
   (similar to OpenAI Operator and Google Mariner)" — the listed schema (`click`/`double`/`drag` with
   `{start,end}`/`type`/`key`/`scroll`/`wait`/`screenshot`/`done`) does not match Anthropic's actual computer-use
   tool, whose actions are `left_click`, `right_click`, `middle_click`, `double_click`, `left_click_drag`, `key`,
   `type`, `scroll`, `wait`, `screenshot`, `cursor_position`; `drag` with start/end and a `done` action are closer to
   OpenAI Operator's action space, and Anthropic has no `done` action at all. The page reads as a transcription of a
   published schema. — Consider: "a simplified action space modeled on Anthropic Computer Use, OpenAI Operator, and
   Google Mariner".

3. `docs/chapter22_computer_use/training.md` (new subsection) — "Because grounding is the dominant failure mode" — this
   is a quantitative claim about the distribution of GUI-agent failures, but the chapter cites no source for it and its
   own Step 9 lists four failure classes with no frequencies; "dominant" is an unsupported superlative in a book that
   otherwise labels its unsupported claims ("teaching record", "classroom defaults"). — Consider softening to "the
   most frequently reported failure mode" with a citation, or "the failure mode this chapter's pipeline targets".

4. `docs/chapter22_computer_use/training.md` (new subsection) — "This is rollout distillation in its simplest form — the
   student imitates only the teacher policy's successful, well-formed rollouts" — in the pipeline described in Step 4
   ("verifier-filtered successful trajectories" produced by the current policy and then used for SFT) the teacher and
   the student are the same model, which Chapter 15 classifies as on-policy *self*-distillation (OPSD) rather than
   teacher-student distillation; the term "teacher policy" needs a qualifier, or a pointer to Ch. 15's OPD/OPSD
   distinction. — Consider: "rejection-sampling fine-tuning; if the rollouts come from a separate stronger policy this
   is off-policy distillation (Chapter 15)".

5. `docs/chapter22_computer_use/safety-swarm.md` — "The InjecAgent paper reports that, under its ReAct setting, 24% of
   GPT-4 tests were attacked" — InjecAgent reports attack success rates separately for direct and indirect injection
   and per agent/model/tool-set combination; a single "24% of GPT-4 tests" figure without naming the injection type
   and the tool configuration should be checked against the paper's table before print (the writer also flags this as
   unverified). — Suggest: give the table reference and the setting.

6. `docs/chapter22_computer_use/safety-swarm.md` — "EVA ... studies indirect injections in GUI scenes such as pop-ups,
   chat, payments, and email composition, adapting attacks to the GUI agent's attention region" — the "attention
   region" adaptation is a strong, checkable claim about arXiv:2505.14289 that I could not confirm offline; if EVA
   actually uses a different mechanism (e.g. UI-region targeting versus attention visualization), the sentence
   overstates the paper. — Suggest verifying against the abstract before print.

7. `docs/chapter22_computer_use/training.md` (new subsection) — "Deduplicate trajectories by (task, snapshot id, seed)"
   — that key identifies an *attempt configuration*, not a trajectory: two legitimate attempts of the same task under
   the same seed and snapshot (e.g. from different checkpoints) would collapse into one and a real success could be
   dropped, which is exactly what Exercise 4's arithmetic assumes away. — Consider adding the rollout id / policy
   version to the key.

## Verdict

The revision is careful and mostly correct: all five fixes the writer claims are genuine improvements (the
select-then-type SoM order, the honest relabeling of the joint objective, the $\gamma\to\delta$ rename, and the
19.4 / 22.2 / Chapter 25 cross-references all check out), and every one of the eight exercise solutions in the new
`review.md` reproduces under independent recomputation (loss 0.0029, RMSE 0.0539, gradient (0.10, −0.04); band
probabilities [1/6, 1/3, 1/2]; dedup 50/190 = 26.3% vs naive 25.0%, RFT set 46; shaped return 1.521 vs unshaped
0.729; vote mean (0.455, 0.625); the Exercise 7 and 8 code runs and its assertions hold). The two new teaching
subsections are useful and mostly technically sound. The serious problems are structural and one conceptual: the new
page is not registered in the sidebar at all (and the report claims it is), its section number "22.R" breaks the
book's numbering convention; the new test-time-click-voting recipe is circular because it filters on the very
bounding box the grounding head must predict; and the writer missed the contradictory "[0,1] / −0.5" progress-rubric
range, the `argmax`-versus-weighted-sampling curriculum mismatch, the mislabeled "Chapter 17: Process Reward Model"
link, a second reuse of $\beta$ on the same page, and the required `content-review-notes.md` entry. Fix the sidebar
registration and the $K$-click recipe first; the remaining items are quick local edits.
