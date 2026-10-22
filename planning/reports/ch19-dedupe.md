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
