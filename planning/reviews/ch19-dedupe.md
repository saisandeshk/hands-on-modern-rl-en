# Review of ch19-dedupe

Scope reviewed: `git diff english-only...w/ch19-dedupe` (4 doc files + the writer's report):
`docs/chapter19_agentic/tool-use-agents.md` (rewritten 765 → 110 lines),
`docs/chapter19_agentic/tool-use-and-trajectory.md` (+1 pointer sentence),
`docs/chapter19_agentic/projects.md` (2 link fixes),
`docs/chapter19_agentic/trajectory-synthesis.md` (1 link fix),
`planning/reports/ch19-dedupe.md`.

Both changed pages were read in full in their final form on the branch. Every link target and
every `#anchor` used by the rewrite was checked against the headings in
`docs/chapter19_agentic/tool-use-and-trajectory.md` and against
`docs/.vitepress/config.mjs`; every numeric example in the chapter's exercise page
(`docs/chapter19_agentic/review.md`, unchanged by this branch but affected by it) was recomputed
with `.venv/bin/python`; the referenced code was checked in `code/chapter19_agentic/`.
All 10 anchors the new 19.5 uses (`#why-is-rl-essential-for-tool-use`, `#core-methods`,
`#reward-design-the-scenario-determines-the-reward`, `#web-agent-rl-teaching-a-model-to-use-the-web`,
`#code-agent-rl-write-code-debug-and-iterate`, `#training-flow-for-tool-use-strategy`,
`#connection-to-rlvr`, `#agentic-rl-engineering-practice`,
`#agent-reward-design-and-evaluation-systems`, `#chapter-summary`) resolve to exactly one heading
each in 19.4, and the two repaired links in `projects.md` /
`trajectory-synthesis.md` resolve as well; no `#anchor` link to 19.5 exists anywhere in `docs/`.
The 7 footnote definitions kept in the new 19.5 are all used and all used ones are defined.

## Errors

1. `docs/chapter19_agentic/tool-use-and-trajectory.md` (line 607, "### Search Tool RL: SearchR1 and
   Search-Augmented Reasoning", through line 672, the closing fence of the `search_reward` block) and
   `docs/chapter19_agentic/tool-use-agents.md`
   (new "## Search-Tool RL: SearchR1 and Search-Augmented Reasoning") — **the one block that was
   supposed to be deduped is still duplicated in full.** 19.4 lines 607–672 and the new 19.5 lines
   ~22–88 carry the same material: the same three prompting limits, the same MDP formulation
   (state/action/transition/reward + the `<search>` interaction transcript), the same three "Key
   Findings", the same "Technical Lineage After SearchR1" table *with the same numbers*
   (SearchR1 819, ToRL 131, ReTool 247, ZeroTIR "NeurIPS 2025"), and the same `search_reward`
   function. The branch even adds to 19.4 the sentence "The deeper treatment of search-augmented RL
   — prompting limits, the SearchR1 MDP, and the work that followed — continues in [Section
   19.5](./tool-use-agents)", which is false from the reader's point of view: the reader has just
   been given exactly that treatment, 60 lines above. The visible consequence is that the same
   citation counts now live in two files and will silently drift apart. The report itself concedes
   "Residual overlap is intentional", but the 19.4 subsection is not a condensed on-ramp — it is a
   full second copy.
   — Suggested fix: keep the treatment in one place only. Either delete 19.4's "Search Tool RL"
   subsection and leave the pointer (then 19.5 is the single source of truth), or keep 19.4's full
   version and reduce 19.5 to a 5-line summary plus pointer. Do not keep two copies of the table.

2. `docs/chapter19_agentic/review.md` (19.11 Summary and Exercises, Exercise 2 solution) — "This
   matches the finding discussed in **Section 19.5** that single-step reward can guide multi-turn
   code generation." After this rewrite that finding no longer exists in 19.5: it lives in 19.4
   under "### Code Agent RL: Write Code, Debug, and Iterate" ("One key finding in Code Agent RL
   comes from ICML 2025 research: **single-step reward can effectively guide multi-turn code
   generation**"). Before the dedupe the sentence was correct (the old 19.5 carried the whole Code
   Agent RL section); the branch removed the section from 19.5 but did not update the chapter's own
   exercise page, which is not in the diff (`git diff english-only...w/ch19-dedupe --stat` shows no
   `review.md`).
   — Suggested fix: change "Section 19.5" to "Section 19.4" in that solution.

3. `docs/chapter19_agentic/overview.md` (line 82, roadmap table) — "| [19.5 Search-Augmented
   RL](./tool-use-agents) | How do systems such as DeepSeek-Researcher and Kimi-Researcher train
   search behavior? |". The rewritten 19.5 never mentions either system: `grep -r "DeepSeek-Researcher\|Kimi-Researcher" docs/`
   returns this overview line, `docs/chapter19_agentic/industrial-practice.md` (19.6) and
   `docs/chapter18_llm_rl_industrial/industrial-post-training.md`. The chapter's roadmap therefore
   sends the reader to a page that does not answer the question it promises. Two further problems in
   the same cell: (a) "DeepSeek-Researcher" is not a system that exists in this book — the system
   the book actually describes is **DeepResearcher** (Zheng et al., arXiv:2504.03160, described in
   the legacy page `docs/chapter19_agentic/projects.md` and its footnote `[^deepresearcher]`), so the
   overview names a non-existent system; (b) Kimi-Researcher's search-behaviour training is covered
   in 19.6 ("Code Interpreter RL", Scenario 1/3/4), not in 19.5. The writer flagged the gap in the
   report ("Coordinator may want a follow-up task"), but it is a one-line fix and the page it breaks
   is the chapter's index.
   — Suggested fix: retarget the overview row (e.g. 19.5 → "How does RL teach a model when to
   search, what to query, and how to use the results?" and 19.6 → the DeepResearcher/Kimi-Researcher
   question, with the correct name "DeepResearcher"), or add the promised material to 19.5.

4. `docs/chapter19_agentic/tool-use-agents.md` ("### Technical Lineage After SearchR1") — the table
   header is "| Work | Core innovation | Citations |" but the third column contains a mixture of
   citation counts and a venue: `| **ZeroTIR **[^zeroscaling]** | ... | NeurIPS 2025 |`, while the
   ReSearch row shows "|" (unknown) and the other three rows show counts (819 / 131 / 247). A venue
   is not a citation count, so the column has no defined meaning for one of its five rows. The
   counts themselves are snapshot numbers with no "as of" date; they are now also duplicated in
   19.4 (same table, lines ~654–660), so the two copies will drift.
   — Suggested fix: use two columns, "Venue" and "Citations (as of <date>)", or drop the counts and
   keep only venues; and keep the table in one of the two pages (see Error 1).

5. `docs/chapter19_agentic/tool-use-and-trajectory.md` ("## Agentic RL Engineering Practice",
   `class ToolSandbox`) — the sandbox is declared `async def execute(self, code)` but calls the
   **blocking** docker-py API: `self.client.containers.run(...)`, `result = container.wait(timeout=...)`,
   `output = container.logs()`. These are synchronous calls inside a coroutine, so they block the
   event loop, which is exactly the failure mode the surrounding section is written against
   ("Whenever the model generates a 'tool-call' action, it must pause and wait for the tool result";
   "**GPU generation and tool execution are different kinds of computation, and they can be
   pipelined**"). With this code the `asyncio.gather` in the next snippet cannot overlap tool
   execution with generation — the event loop is held by the container call, so the "asynchronous
   concurrency" argument of the section is not demonstrated by its own code.
   — Suggested fix: wrap the blocking calls (`await asyncio.to_thread(container.wait, ...)`) or use an
   async Docker client (`aiodocker`, or `docker.from_client()` with `await client.containers.run(...)`).

6. `docs/chapter19_agentic/tool-use-and-trajectory.md` (same snippet) — the container is started with
   `detach=True, remove=True` (Docker auto-remove) and the code then reads the logs *after* the
   container has exited: `result = container.wait(timeout=self.timeout)` then
   `output = container.logs().decode("utf-8")`. With auto-remove, the container is deleted as soon as
   it exits, so the second call typically raises `docker.errors.NotFound: No such container`; at best
   it is a race that depends on how fast the daemon reaps the container.
   — Suggested fix: start with `remove=False`, read `container.logs()`, then call
   `container.remove()` explicitly (or use `docker run --rm` semantics only for the stdout stream
   captured at start time).

7. `docs/chapter19_agentic/tool-use-and-trajectory.md` ("### Asynchronous Concurrency") —
   `async def parallel_rollouts(model, tasks, sandbox, num_workers=16)` never uses `num_workers`:
   it builds one coroutine per task and calls `await asyncio.gather(*coroutines)`, so there is no
   concurrency limit at all (all tasks are scheduled at once). The parameter silently does nothing
   while the docstring promises "Launch rollouts for multiple trajectories in parallel" with a
   worker count. This is the reader's first impression of how to bound tool concurrency in a
   training system.
   — Suggested fix: bound it, e.g. `sem = asyncio.Semaphore(num_workers)` around the rollout body, or
   use a worker-pool pattern; otherwise delete the parameter.

8. `docs/chapter19_agentic/tool-use-agents.md` (and the identical copy in 19.4) — the reward
   function contradicts its own signature:
   ```python
   def search_reward(answer, ground_truth, num_searches, max_searches=5):
       correctness = 1.0 if verify_answer(answer, ground_truth) else 0.0
       efficiency_penalty = -0.05 * num_searches
   ```
   `max_searches` is never used. Two consequences: (a) the advertised cap does not exist — with
   `num_searches = 20` the total reward is `1.0 - 1.0 = 0`, and from 21 searches on the total
   reward is negative, so a *correct* answer can be worth less than a wrong one with fewer
   searches; (b) the page's argument is that "stopping too late wastes resources" and that the
   trade-off "is exactly where RL is useful" — an unbounded linear penalty that can dominate the
   task reward is not the bounded trade-off the text describes. (Note `verify_answer` is also
   undefined in the snippet; that is acceptable as pseudo-code, the unused cap is not.)
   — Suggested fix: either use the cap, e.g.
   `efficiency_penalty = -0.05 * (min(num_searches, max_searches) / max_searches)`, or drop
   `max_searches` from the signature and state in the text that the penalty is unbounded.

## Doubts

1. "The 32B model trained by ToRL[^torl] outperformed a 70B model without tools on mathematical
   reasoning" (19.5 and 19.4 line 664) — ToRL's headline comparison is a 32B tool-integrated model
   against a larger *no-tool* model on AIME (I believe Qwen2.5-72B-Instruct, hence "70B"), so the
   claim is probably right, but no benchmark, no base model and no baseline name are given, and the
   reader cannot check "32B > 70B" against anything. I cannot verify the exact figures offline.
   — Suggest naming the benchmark and the compared model explicitly (e.g. "on AIME 2024, ToRL-32B
   reaches X% pass@1, above the no-tool Qwen2.5-72B-Instruct"), rather than only the parameter
   counts.

2. "A complex task may require 5-10 searches" (both copies) — an unsourced number presented as a
   property of the task class. Fine as intuition, but it is stated with the same confidence as the
   other claims in that list.

3. "this kind of asynchronous concurrency often raises GPU utilization from 20-30% to 70-80%,
   improving training throughput by 2-3x" (19.4) — the stated range is not consistent with the stated
   factor: 20% → 80% is a 4x speed-up, 30% → 70% is 2.3x. "2-3x" only matches the middle of the
   range. Also "often raises" is a strong quantitative claim with no source; the writer's own
   references (AReaL 2.77x) are for a different system.

4. The new 19.5's closing pointer — "In the next section, we will look at practical agentic RL
   experience from industry: [Industrial Practice](./industrial-practice)" — the link text does not
   match the target page: `industrial-practice.md` opens with "# 19.6 Code Interpreter RL" and the
   sidebar says "19.6 Code Interpreter RL", while the overview describes 19.6 as "How do code agents
   handle instability, excessive trajectory length, and reward hacking?". The branch's new prose
   therefore labels the next section by a name that appears nowhere in it. Pre-existing, but the
   rewrite re-states the pointer, so it is a good moment to align the three texts.

5. The "Background in Brief" bullet "**Core methods.** ReTool ... VERL-TOOL ... ToolRL ..." omits
   MCP-RL, which in 19.4 is by far the longest tool-method subsection (with the MDP-element mapping
   table and the "MCP is an interface layer, not an algorithm" argument). A reader who trusts the
   bullet list will not know that 19.4 contains a whole protocol-level treatment.

6. The retained search claims that the writer listed as "needs external verification" (SearchR1
   citation count 819, ToRL 131, ReTool 247; Search-R1 "COLM 2025"; ZeroTIR "NeurIPS 2025";
   ToolRL "NeurIPS 2025" in 19.4) — I cannot check any of these offline. The venue claims are the
   ones that matter (a wrong venue is a factual error); the citation counts are self-invalidating
   within a year and should carry a date or be dropped.

7. The new 19.5 is now a 110-line page whose only original content is the "Background in Brief"
   pointer list plus a search section that also exists in 19.4; the chapter's own 19.11 exercise
   page still refers to 19.5 content that moved (Error 2), and the 19.5 page has no reflection
   question of its own (both old 19.5 reflection questions were moved to 19.4, which is fine, but
   the search section now has no `<details>` exercise in either page). Not an error, but worth a
   follow-up task if the page is meant to stand on its own.

## Verdict

The rewrite is mechanically careful: every anchor resolves, the footnote set is exactly the seven
still cited with no orphans, the front matter and heading match the sidebar, the three repaired
links point at the right targets, and the "Background in Brief" bullets are accurate summaries of
19.4 — I found no false statement introduced by the new prose. The real problem is that the job was
only half done and the half that was done created new inconsistencies. The largest duplicated block
(the whole SearchR1 treatment, with the same table and the same citation numbers) survives in both
19.4 and 19.5, and 19.4 now advertises 19.5 as "the deeper treatment" of material the reader has
already just read. The rewrite left three dangling references it should have closed: the chapter's
own exercise page still points at "Section 19.5" for a finding that moved to 19.4, the overview
still promises 19.5 will explain "DeepSeek-Researcher and Kimi-Researcher" (and names a system that
does not exist in this book), and the new page's forward pointer calls 19.6 "Industrial Practice"
while 19.6 is titled "Code Interpreter RL". On substance, the engineering snippets that the
dedupe consolidated into 19.4 have real defects that a reader will copy verbatim: an `async def`
that calls blocking Docker APIs, an auto-removed container whose logs are read after exit, a
`num_workers` parameter that does nothing, and a `search_reward` whose `max_searches` cap is never
applied (so a correct answer can score below zero). Fix the duplication and the three stale
cross-references first, then the four code defects.
