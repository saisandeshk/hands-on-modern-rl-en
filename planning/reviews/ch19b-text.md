# Review of ch19b-text

Scope reviewed: `docs/chapter19_agentic/overview.md`, `tool-use-and-trajectory.md`,
`tool-use-agents.md`, `trajectory-synthesis.md`, `projects.md`, `rllm-deepcoder-lab.md`, the new
`docs/chapter19_agentic/review.md`, and `planning/reports/ch19b-text.md`. Every changed equation was
re-derived, every numeric example recomputed (`.venv/bin/python`), every code snippet in the new
`review.md` executed, and every cross-reference checked against `docs/.vitepress/config.mjs` and the
target pages.

## Errors

1. `docs/chapter19_agentic/review.md` (lines 26, 34, 46, 50, 109, 113, 117) — seven display
   equations use `*` where a LaTeX subscript `_` is required, e.g.
   `$$ \langle S*{\text{agent}},\ A*{\text{agent}},\ P*{\text{agent}},\ R*{\text{agent}},\ \gamma,\ O \rangle $$`,
   `$$ G*t = r_t + \gamma G*{t+1} $$`,
   `$$ R*{\text{total}} = R*{\text{task}} - \lambda*{\text{efficiency}} \cdot T - \lambda*{\text{format}} \cdot \mathbb{1}(\text{format error}) $$`,
   `$$ \text{update utilization} = \frac{T*{\text{update}}}{T*{\text{rollout}} + T*{\text{verify}} + T*{\text{update}}} $$`,
   `$$ \pi*\theta(\tau) = \prod*{t=0}^{T} p*\theta(x_t \mid x*{<t}) $$`,
   `$$ \nabla*\theta J(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}\left[ R(\tau) \nabla*\theta \log \pi*\theta(\tau) \right] = \mathbb{E}*{\tau \sim \pi*\theta}\left[ R(\tau) \sum*{t=0}^{T} \nabla*\theta \log p*\theta(x*t \mid x*{<t}) \right]. $$`,
   `$$ \nabla*\theta J(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}\left[ R(\tau) \sum*{t \in a(\tau)} \nabla*\theta \log p*\theta(x*t \mid x*{<t}) \right], $$`.
   In KaTeX `*` is the binary operator `\ast`, not a subscript, so these render as
   "S∗agent, A∗agent, …", "G∗t = r_t + γG∗t+1", "R∗total = R∗task − λ∗efficiency·T − …",
   "T∗update / (T∗rollout + T∗verify + T∗update)", "π∗θ(τ) = ∏∗t=0T p∗θ(x_t | x∗<t)",
   "∇∗θJ(θ) = E∗τ∼π∗θ[…]". In the last three equations the corruption is not cosmetic: the
   `\prod` loses its limits entirely, the expectation's subscript is gone, and `\nabla*\theta`
   becomes a product of ∇ and θ. I checked the whole `docs/` tree: this file is the only page with
   this pattern (the only other `*{` hits are the legitimate `\operatorname*{arg\,max}` in
   `chapter03_value_functions/value-q.md`), and the repo's own linter
   `scripts/check-english-quality.mjs` has a check literally named "corrupted reward subscripts"
   with the pattern `/\bG\*t\b|\br\*\{t\+1\}|\\pi\*|\\phi\*|\bs\*\{?t\b/g` — the new file matches both
   `\bG\*t\b` and `\\pi\*` (the script currently scans a `docs/en` directory that does not exist in
   this repository, so the check never fires, but the pattern documents exactly this failure mode).
   The writer's report states the page was rebuilt with "no KaTeX errors",
   which is true (these do not raise KaTeX errors) but the rendered mathematics is wrong.
   — Suggested fix: replace every `*` used as a subscript with `_` in those seven equations
   (`S_{\text{agent}}`, `G_t = r_t + \gamma G_{t+1}`, `R_{\text{total}} = R_{\text{task}} - \lambda_{\text{efficiency}}\cdot T - \lambda_{\text{format}}\cdot\mathbb{1}(\text{format error})`,
   `T_{\text{update}} / (T_{\text{rollout}} + T_{\text{verify}} + T_{\text{update}})`,
   `\pi_\theta(\tau) = \prod_{t=0}^{T} p_\theta(x_t \mid x_{<t})`, `\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[\dots]`,
   `\sum_{t\in a(\tau)} \nabla_\theta \log p_\theta(x_t \mid x_{<t})`).

2. `docs/chapter19_agentic/tool-use-and-trajectory.md` (line 975, new subsection "Staleness, Retries,
   and Recovery: Making Async Rollouts Sound") — "Give each tool call an idempotency key (task id,
   turn index, attempt number) so the environment can deduplicate retries". Including the **attempt
   number** in the key defeats the stated purpose: the second attempt of the same logical action
   produces a *different* key, so the environment cannot recognise it as a duplicate and will
   execute the state-changing call a second time — exactly the failure ("a retried order placement
   or file write may execute twice") the sentence is trying to prevent. An idempotency key must be
   stable across retries of the same action (e.g. task id + turn index, or a hash of the action
   arguments); the attempt counter has to be stored *alongside* the key for backoff/logging, not
   inside it.
   — Suggested fix: "Give each tool call an idempotency key that is stable across retries (task id
   and turn index, or a hash of the action arguments) so the environment can deduplicate retries,
   and keep the attempt number as a separate counter used for backoff and logging."

3. `docs/chapter19_agentic/tool-use-and-trajectory.md` (lines 170, 172) and
   `docs/chapter19_agentic/trajectory-synthesis.md` (lines 171, 173) — "In Section 8.4, we saw HER
   in robotics" / "Combined with the discussion in Section 8.4". HER is in Chapter 8, but in
   **Section 8.6**, not 8.4: `docs/chapter08_ppo/rl-long-horizon-planning.md` opens with
   `# 8.6 Hands-On: Long-Horizon Tasks and Planning` and contains the section
   `## Hindsight Experience Replay`. Section 8.4 is "Hands-on: PPO Mathematical Derivation"
   (`docs/chapter08_ppo/ppo-math.md`), which never mentions HER. The writer's report explicitly
   left this reference unfixed as "unverifiable"; it is verifiable from the sidebar in
   `docs/.vitepress/config.mjs` (8.4 = ppo-math, 8.6 = rl-long-horizon-planning).
   — Suggested fix: "In Section 8.6, we saw HER in robotics." (both files).

4. `docs/chapter19_agentic/review.md` (line 119, Exercise 4 solution) — "This is the same masking
   used in the rollout code of Sections 19.4 and 19.8." There is no masking/rollout code in either
   of those sections: `docs/chapter19_agentic/rllm-deepcoder-lab.md` (19.8) contains no
   action-mask, `info_mask`, or `labels = -100` code at all (grep for `mask`/`-100` returns
   nothing), and 19.4 (`tool-use-and-trajectory.md`) only mentions masking in prose about
   SearchR1. The concrete masked-rollout implementation (`masked_ids`, `info_mask`,
   `labels[info_mask == 0] = -100`) lives in the legacy page `docs/chapter19_agentic/projects.md`
   (lines 379–496), and the formalism of the action mask is in 19.2 (`formulation.md`, "Action
   Mask: Model Generation and Environment Return Must Be Distinguished") and 19.3
   (`credit-assignment.md`).
   — Suggested fix: "This is the same masking used in the action-mask formulation of Section 19.2
   and in the rollout code of the chapter's project pages."

5. `docs/chapter19_agentic/review.md` (lines 158, 168, Exercise 6 part 3) — "The baseline evaluation
   solves 6 of 20 problems and the post-training evaluation solves 20 of 50 … Post-training Pass@1
   $= 20/50 = 40.0\%$. Gain $= 40.0 - 30.0 = 10$ percentage points (consistent with the lab's
   reported +8–10 point range)." The lab (19.8, `rllm-deepcoder-lab.md` lines 793–796) prints
   `Pass@1 (n=20): 30.0%  # 6 out of 20 problems correct` and
   `Pass@1 (n=50): 39.0%    # ~20 out of 50 problems correct, +9 percentage points`. 20/50 is
   exactly 40.0%, not the 39.0% the lab reports, so the exercise silently changes the referenced
   measurement from +9 pp to +10 pp and then declares the difference "consistent" with the
   lab's own "+8–10 point range" (the lab's summary table says "~38-40%", "+8-10 percentage
   points"). The exercise's own arithmetic is right; its premise about the lab's output is not
   what the lab prints.
   — Suggested fix: keep the lab's numbers — "the post-training evaluation solves about 20 of 50
   (39.0%) … Gain ≈ 9 percentage points" — or state explicitly that the exercise rounds the lab's
   39.0% to 20/50 = 40.0%.

6. `docs/chapter19_agentic/review.md` (lines 1–5) — the new page is not registered in
   `docs/.vitepress/config.mjs` (the chapter-19 sidebar ends at "19.10 Hands-on: Building an
   Agentic RL Training System from Scratch", lines 1225–1228) and its title/H1 use the placeholder
   number "19.R" instead of a sidebar-driven number. Consequences: (a) the page is unreachable from
   the site navigation, which `AGENTS.md` forbids ("Every navigable page must be listed there");
   (b) `scripts/fix-heading-numbers.mjs` only rewrites pages that appear in the sidebar, so
   "19.R" will never be reconciled with a real section number; (c) "19.R" departs from the
   convention of every other summary page in the book (2.4, 3.4, 4.4, 5.6, 7.8). The writer's report
   defers this to the coordinator, so it is a known open item rather than an oversight.
   — Suggested fix: add `{ text: '19.11 Summary and Exercises', link: '/chapter19_agentic/review' }`
   to the chapter-19 sidebar and change the front-matter `title` and the H1 to
   `19.11 Summary and Exercises` (then run `npm run fix:headings`).

7. `docs/chapter19_agentic/credit-assignment.md` (lines 618–624, page 19.3 — *not* touched by this
   branch, but the writer's report asserts that "the chapter's core technical content (… ORM/PRM
   credit assignment …) is sound" and that "the numeric examples I recomputed are correct") — the
   printed PRM column contradicts the code printed directly above it. `prm_credit` (lines 592–607)
   assigns immediate rewards `[0.3, 0.3, 0.3, -0.5]` to the bad episode and back-propagates with
   `γ = 0.95`, which yields
   `G_3 = -0.5`, `G_2 = 0.3 + 0.95·(-0.5) = -0.175`, `G_1 = 0.3 + 0.95·(-0.175) = 0.13375`,
   `G_0 = 0.3 + 0.95·0.13375 = 0.42706`, i.e. `[0.427, 0.134, -0.175, -0.500]`. The page instead
   prints `PRM Credit = 0.656, 0.376, 0.170, -0.500`, which no choice of the code's parameters
   reproduces (inverting the recursion gives immediate rewards `[0.299, 0.215, 0.645, -0.5]`, which
   correspond to no rule in the code); the values `0.656/0.590/0.729/0.810/0.900/1.000` are the
   *other* example on the same page (the 7-turn turn-discounting table at line 368). The ORM column
   (`0.000` everywhere) and the `-0.857` figure quoted for `orm_negative` do check out
   (`G_0 = 0.95^3·(-1) = -0.8574`). Because the printed table is also what a reader would use to
   read the ORM-vs-PRM contrast, the sign of the third turn's PRM credit (positive `0.170` in the
   table, negative `-0.175` in the code) misrepresents the very point the section makes.
   — Suggested fix: replace the PRM column with the values the code produces,
   `[0.427, 0.134, -0.175, -0.500]`, and re-check the surrounding prose (the claim that PRM
   "precisely locates the steps that need improvement" still holds, but the turn-3 credit is
   negative because the wrong final answer already dominates the discounted return).

## Doubts

1. `docs/chapter19_agentic/tool-use-agents.md` (line 189) — "510 RL steps can outperform a model 40
   times larger" versus `docs/chapter19_agentic/tool-use-and-trajectory.md` (line 578) — "a model
   roughly 48 times larger, 671B versus 14B" for the same rStar2-Agent result. 671/14 = 47.93, so
   19.4's figure is the accurate one and the two now-duplicated pages state different facts about
   the same experiment. (The writer only fixed the GRPO-RoC sentence in both files.)
   — Suggested fix: use "roughly 48 times larger (671B vs 14B)" in both places.

2. `docs/chapter19_agentic/overview.md` (line 206) — "Works such as AEM and RAGEN-2 have addressed
   this from the perspective of stability." "RAGEN-2" appears nowhere else in the book and is not in
   the writer's "needs external verification" list; the only RAGEN reference is the project
   `RAGEN-AI/RAGEN`. If there is no separate "RAGEN-2" work, the name reads as fabricated.
   — Suggested fix: verify the citation or replace with a work that exists (e.g. AEM alone).

3. `docs/chapter19_agentic/review.md` (line 115, Exercise 4 solution) — "Environment tokens are
   sampled from the world (retriever, sandbox), not from $\pi_\theta$, so $\nabla_\theta \log p(x_t
   \mid x_{<t}) = 0$ for them as far as the policy is concerned — yet if they are fed through the
   policy network as if self-generated, the sum includes nonzero pseudo-gradients". The literal
   statement is false and is contradicted by the clause that follows it: ∇_θ log p_θ(x_t | x_{<t})
   is *not* zero for an environment token (the network parameters do change that probability).
   The correct statement is the one given in 19.2: the trajectory factorization
   ρ_θ(x|s_0,c) = 𝕀(s_0,x)·∏_{t∈a(x)} p_θ(x_t|·) contains **no factor** for environment
   positions, so those positions contribute no term to ∇_θ log ρ_θ; the masking is what keeps the
   implemented loss (which sums log-probs over *all* positions) from adding spurious terms.
   — Suggested fix: "…so they carry no factor in the trajectory log-probability and contribute no
   term to the gradient; if they are nevertheless fed through the policy network as if
   self-generated, the implemented loss includes spurious pseudo-gradients…"

4. `docs/chapter19_agentic/tool-use-and-trajectory.md` (line 241) and
   `docs/chapter19_agentic/trajectory-synthesis.md` (line 242) — "takes test-time search techniques
   from Chapter 16, such as beam search and best-of-N". Best-of-N is indeed 16.3
   (`chapter16_reasoning/test-time-scaling.md`, "Parallel Sampling: Best-of-N"), but the same page
   states at line 148: "The specific algorithms (Beam Search, Tree of Thoughts, MCTS, and others)
   are developed in Section 17.5". Beam search therefore lives in 17.5 (Chapter 17), not 16.
   — Suggested fix: "…from Chapter 16 (best-of-N) and 17.5 (beam search), such as …".

5. `docs/chapter19_agentic/review.md` (line 67, Exercise 1 solution) — the solution lists
   `$S$: the true world state — airline database contents, live seat inventory, prices, plus the
   conversation history so far`. In 19.2 the state is decomposed as `z_t = [s_0, c, x_{1:t}]`, i.e.
   the *hidden* environment state `s_0` plus the task context `c` plus the token history; mixing
   "the true world state" and "the conversation history" into one slot is looser than the chapter's
   own decomposition (and the POMDP tuple in 19.1 defines `S_agent` as "the state space jointly
   determined by the environment and the context"). Not wrong, but it is the one place where the
   summary quietly redefines the chapter's notation.
   — Suggested fix: align the wording with 19.1/19.2 ("$S$: the environment state space —
   airline database, live seat inventory, prices — plus the agent's context and token history").

## Verdict

The revision is technically careful and the writer's fixes are sound: the prompt-masking bug in
`projects.md` (prompt tokens replaced by `PAD_ID` so only model decision tokens enter the loss),
the `ResearchEnvironment.step` arity fix, the `ground_truth` parameter, the `has_format_error`
parameter, and the softened GRPO-RoC description are all genuine corrections, and every numeric
example in the new `review.md` that I recomputed (discounted returns `[0.81, 0.9, 1.0]` /
`[0.25, 0.5, 1.0]`, group advantages `[-1, +1, -1, +1]`, the one-step REINFORCE probability
`0.3445`, and the copied `compute_discounted_returns` implementation) is correct. The most
important problem is the new page itself: seven display equations use `*` instead of `_`, so the
POMDP tuple, the discounted-return recursion, the agentic reward formula, the update-utilization
formula, and the whole Exercise 4 policy-gradient derivation render with multiplication signs and
without subscripts or product limits — the exact corruption the repository's own linter checks for,
and the page is also missing from the sidebar, so it is unreachable and its "19.R" number will
never be reconciled. The new "Staleness, Retries, and Recovery" section contains one genuine
mechanism error (putting the attempt number inside the idempotency key, which defeats retry
deduplication), the HER cross-reference still points at Section 8.4 instead of 8.6 in two files
the writer edited, and the review's Exercise 4 points at masking "code" in sections that contain
none. Finally, the ORM/PRM credit-assignment table in 19.3 — a page the report calls "sound" —
does not follow from the code printed directly above it, so the chapter's most quoted numeric
illustration of credit assignment is wrong.
