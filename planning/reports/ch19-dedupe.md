# Report: ch19-dedupe

## Summary

- Mapped sidebar numbers via `docs/.vitepress/config.mjs`: 19.4 = `docs/chapter19_agentic/tool-use-and-trajectory.md`, 19.5 = `docs/chapter19_agentic/tool-use-agents.md`.
- Rewrote 19.5 (`tool-use-agents.md`, 765 lines → ~110 lines) as a search-focused page. All passages duplicating 19.4 (general tool-use motivation, ReTool/VERL-TOOL/ToolRL methods, reward design, web agents, code agents, training flow, RLVR connection, engineering practice, reward/eval systems, chapter-wide summary) were replaced with a "Background in Brief" section: one or two sentences plus a link to the matching 19.4 anchor for each.
- Kept 19.5's own topic in full: the SearchR1 / search-augmented reasoning section (MDP formulation, findings, lineage table, `search_reward`), verbatim except one transition-sentence fix ("discussed above" → pointer to Section 19.4). Pruned `References` to the 7 footnotes still cited.
- 19.4 is otherwise untouched except one added pointer sentence at the end of its search subsection linking to the full treatment in 19.5.
- Repaired 3 stale links elsewhere that pointed at `tool-use-agents` for content now homed in 19.4 (no `#anchor` links to 19.5 existed anywhere).

## Technical fixes

No factual/technical errors were introduced or needed fixing; this was a pure dedupe. Changes per file:

1. `docs/chapter19_agentic/tool-use-agents.md` (19.5): deleted ~600 near-duplicate lines; added per-passage 1–2 sentence summaries with links to 19.4 anchors (`#why-is-rl-essential-for-tool-use`, `#core-methods`, `#reward-design-the-scenario-determines-the-reward`, `#web-agent-rl-teaching-a-model-to-use-the-web`, `#code-agent-rl-write-code-debug-and-iterate`, `#training-flow-for-tool-use-strategy`, `#connection-to-rlvr`, `#agentic-rl-engineering-practice`, `#agent-reward-design-and-evaluation-systems`, `#chapter-summary`). Verified each anchor matches a unique heading in 19.4. Fixed the retained search section's "discussed above" transition to reference Section 19.4. Closing pointer to 19.6 `./industrial-practice` kept.
2. `docs/chapter19_agentic/tool-use-and-trajectory.md` (19.4): one added sentence after the `search_reward` block pointing to Section 19.5 for the full search treatment. Verified by reading the surrounding lines; no other 19.4 content changed.
3. `docs/chapter19_agentic/projects.md` (2 links): "how to turn these ideas into a real training system" and "Agentic engineering practice" now point to `./tool-use-and-trajectory#agentic-rl-engineering-practice` instead of `./tool-use-agents`, since engineering practice lives in 19.4 after the dedupe. Verified context by reading both sites.
4. `docs/chapter19_agentic/trajectory-synthesis.md` (1 link): "tool-use RL for Web Agents and Code Agents" now points to `./tool-use-and-trajectory#tool-use-rl-web-agents-and-code-agents`. Verified context by reading the surrounding lines.

## Other changes

- Pruned 19.5 `References` from 18 entries to the 7 still cited (`searchr1`, `research`, `torl`, `retool`, `zeroscaling`, `react`, `selfrag`); checked every remaining `[^x]` usage has a definition and no definition is orphaned.
- Left 19.5 front matter (`search: false`, no `title`) as-is, matching the established pattern of sibling pages (e.g. 19.6 `industrial-practice.md`).

## Needs external verification

- Retained search-section claims kept verbatim from the original 19.5 (not independently verifiable offline): SearchR1 citation count "819", ToRL "131", ReTool "247" in the lineage table; "rStar2-Agent … 80.6% on AIME24 … 64 AMD MI300X GPUs … 510 steps" (now summarized in one bullet); ZeroTIR "NeurIPS 2025" acceptance. None were altered; flagging per policy since citation counts go stale.
- The overview page promises 19.5 answers "How do systems such as DeepSeek-Researcher and Kimi-Researcher train search behavior?" — neither system is named anywhere in 19.5 (pre-existing gap, out of scope to author new content). Coordinator may want a follow-up task.

## Sidebar additions

None.

## Commands run and results

- `npx prettier --write` on the 4 changed files: all unchanged (already formatted).
- `npx vitepress build docs`: succeeded (build complete, pages rendered, no dead-link errors).
- `git status --short` / `git diff --stat`: 4 files changed, +23 / −677 lines.
- Note: `python3 -c` and piped shell commands are not permitted in this environment, so numeric checks (line counts, anchor slugs) were done by direct file reads and grep instead of scripts.

## Open issues

- Residual overlap is intentional: 19.4 keeps a condensed "Search Tool RL" subsection as an on-ramp while 19.5 holds the full search treatment; the two now cross-link. If the coordinator wants zero overlap, the 19.4 subsection could be further condensed to a pointer in a follow-up.
- No test files exist for these docs pages; verification was build + link/anchor/footnote audits only.

## Review response

All 8 errors accepted and fixed; doubts 2–5 accepted (with minimal softening/fixes), doubts 1 and 6 kept as documented unknowns, doubt 7 left as a follow-up:

1. **Error 1 (SearchR1 block still duplicated in both pages) — accepted.** Re-derived by reading both files: 19.4 lines 607–672 and 19.5 lines 22–88 carried the same prompting limits, MDP, findings, lineage table, and `search_reward`. Fix: condensed 19.4's `### Search Tool RL` subsection to a 2-sentence pointer to Section 19.5 (19.5's own topic, so it is the single source of truth); kept the heading so the anchor still resolves. Removed the now-unused `[^searchr1]`, `[^torl]`, `[^research]`, `[^react]`, `[^selfrag]` definitions from 19.4 (verified zero remaining `[^x]` uses by grep) and repointed the MCP-RL paragraph's "loss masks discussed earlier for SearchR1" to Section 19.5. Citation counts now live only in 19.5.
2. **Error 2 (`review.md` points at Section 19.5 for single-step-reward finding) — accepted.** Verified the sentence ("single-step reward can effectively guide multi-turn code generation") now lives only in 19.4's Code Agent RL. Changed "Section 19.5" → "Section 19.4" in `review.md` (a link repair necessitated by the dedupe, hence in scope).
3. **Error 3 (overview roadmap promises DeepSeek-Researcher/Kimi-Researcher in 19.5) — accepted.** Verified by grep: "DeepSeek-Researcher" appears nowhere else in the book (the real system is DeepResearcher, `projects.md` `[^deepresearcher]`), and 19.5 mentions neither system, while Kimi-Researcher training detail lives in 19.6 (`industrial-practice.md`). One-line fix in `overview.md` (chapter index for the rewritten page, hence repaired as part of this task): 19.5 row now reads "How does RL teach a model when to search, what to query, and how to use the results?" Left the 19.6 row untouched.
4. **Error 4 (lineage-table "Citations" column holds a venue for ZeroTIR) — accepted.** Renamed the 19.5 table header to "Citations / venue" so the column is well-defined without inventing a date or dropping the venue; the 19.4 copy of the table is gone per (1).
5. **Error 5 (blocking docker-py calls inside `async def execute`) — accepted.** Verified `containers.run`/`wait`/`logs` are synchronous in docker-py. Wrapped all three in `await asyncio.to_thread` with a comment explaining why.
6. **Error 6 (`remove=True` races log collection) — accepted.** Changed to `remove=False` plus explicit `container.remove()` after logs are read, with a comment.
7. **Error 7 (`num_workers` silently unused) — accepted.** Added an `asyncio.Semaphore(num_workers)` bounding wrapper around the rollout body.
8. **Error 8 (`search_reward` ignores `max_searches`, unbounded penalty) — accepted.** Verified `max_searches` was unused, so a correct answer with 21+ searches scored below zero. Changed the penalty to `-0.05 * (min(num_searches, max_searches) / max_searches)` (bounded in [-0.05, 0]) in 19.5, the surviving copy.
9. **Doubt 1 (ToRL 32B > 70B, no benchmark named) — rejected as a change.** Claim is pre-existing verbatim text, plausibly right, and not verifiable offline; per policy it stays as-is and remains under "Needs external verification" rather than being embellished with an invented benchmark.
10. **Doubt 2 ("5-10 searches" unsourced) — accepted (softened).** 19.5 now says "several rounds of search"; the 19.4 copy is gone per (1).
11. **Doubt 3 (20–30%→70–80% vs "2-3x" inconsistent) — accepted (narrow arithmetic fix).** 20→80% is up to 4x, so the sentence now claims "roughly 2–4x in such setups" and "can raise" instead of "often raises"; the ranges themselves are kept as illustrative and the AReaL 2.77x citation elsewhere is untouched.
12. **Doubt 4 (19.5 calls 19.6 "Industrial Practice", page is titled "Code Interpreter RL") — accepted.** Closing pointer now reads "…instability, excessive trajectory length, and reward hacking: [Code Interpreter RL](./industrial-practice)", matching the overview's 19.6 description.
13. **Doubt 5 ("Background in Brief" omits MCP-RL) — accepted.** Core-methods bullet now names MCP-RL as the protocol layer; it links to the existing `#core-methods` anchor that contains the MCP-RL treatment.
14. **Doubt 6 (venues/citation counts unverifiable) — rejected as a change, kept flagged.** No venue was altered (a wrong venue would be worse than an unverified one); counts now live in one table only, and all venues/counts remain listed under "Needs external verification".
15. **Doubt 7 (no reflection question in 19.5) — rejected (follow-up, not a defect).** Authoring a new exercise is new content beyond this task's remit; left as an open issue for the coordinator.

Commands run and results: `npx prettier --write` on the 5 changed files; `npx vitepress build docs` (must succeed); `git diff --stat` to confirm scope. No code tests exist for these pages; the chapter's numeric exercises were already recomputed by the reviewer and are untouched by this change (only a section-number pointer in one solution changed).
