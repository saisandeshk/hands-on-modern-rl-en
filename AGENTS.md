# Repository Guidelines

## Project Structure

- `docs/`: VitePress course site, English only. Each chapter is `docs/chapterNN_<slug>/`; appendices are `docs/appendix_[a-d]_<slug>/`. The number `NN` is the displayed chapter number.
- Images for a page live in `images/` next to the page. Vue components and code-map snippets used by a chapter live in that chapter's `components/` and `snippets/`.
- `docs/.vitepress/config.mjs`: navigation (`nav`) and the hand-written `sidebar`. Every navigable page must be listed there.
- `docs/.vitepress/theme/`: layout and components. `docs/public/`: static assets copied to the site root.
- `code/`: runnable Python experiments, numbered like the chapters (`code/chapter05_dqn` belongs to Chapter 5). `code/online-experiments/` and `modelscope-space/` are upstream ModelScope material, kept as is.
- `scripts/`: checks, sitemap, asset optimization, PDF/EPUB builds.
- `planning/`: upstream outline notes; `planning/content-review-notes.md` tracks suspected content errors.

## Commands

```bash
npm install
npm run dev:fast        # dev server, no asset optimization (~5s startup)
npm run dev             # dev server with asset optimization
npm run build
npm run preview
npm run verify          # format, lint, English checks, code-link check, build
npm run fix:headings    # sync heading numbers with the sidebar
```

## Editing Rules

- Keep changes scoped. Do not rewrite unrelated docs or config.
- When editing existing course material, preserve the structure, narrative style, examples, and level of detail. By default make the smallest local change that corrects an error or improves wording. A request to review, polish, or fix a page does not authorize a full rewrite, a course reorganization, or edits to neighboring pages. If broader restructuring seems necessary, explain the scope and get explicit confirmation first.
- New pages go under the right `docs/chapterNN_*` directory and must be added to the sidebar in `docs/.vitepress/config.mjs`. Section numbers in page headings follow the sidebar (`npm run fix:headings`).
- New code goes under the matching `code/chapterNN_*` directory with a `requirements.txt` entry for any new dependency.
- Links into the repository from pages use `https://github.com/saisandeshk/hands-on-modern-rl-en/blob/main/<path>`; `npm run verify:code-links` checks that each target exists.
- Run `npm run verify` after touching config, theme, scripts, or many pages.
- When fixing a factual error, record it (and the fix) in `planning/content-review-notes.md`.

## TikZ → SVG Workflow

For academic-style diagrams with proper LaTeX math rendering:

1. Write the diagram as a `standalone` TikZ document (see `docs/preface/images/actor-critic.tex`).
2. Compile: `xelatex <file>.tex && pdf2svg <file>.pdf <file>.svg`.
3. Keep both `.tex` (source) and `.svg` (output) in the same `images/` directory.
4. Reference the `.svg` from markdown with `<img src="./images/<name>.svg">`.

Style: rounded-corner boxes (`draw=black!70, fill=black!3`), Stealth arrows, `$...$` for math. All labels in English.

## Writing Conventions

- Write docs as tutorial and lecture material, not terse reference notes: polished English textbook prose in a patient, guided lecture voice ("Let's start with the intuition", "Look at it from another angle", "What this step means is…", "The real issue is…"), serious and precise.
- Open a section by connecting to what the learner already knows, then name the new problem the section solves. Use concrete scenes before abstractions (CartPole, bandits, LLM generation, games).
- Follow the causal line: new capability or setting appears → it changes the training/problem object → this enables or forces a new formulation. Do not circle the same point with summarize-restate-contrast.
- Every conceptual step needs a transition. Do not jump from a definition straight to a formula or algorithm; say why the next idea is needed. Definitions come after motivation and are followed by interpretation.
- Before a formula or algorithm, state the problem it solves. Explain every important symbol, then restate the formula in plain English.
- Good rhythm: familiar example → tension → formal tool → worked example → takeaway. Intuition first, then formal definitions, then code or practice.
- Be direct: no filler ("it is more intuitive to explain with…"), no redundancy, no vague metaphors or hedging; state precise causal mechanisms.
- Avoid the "not A, but B" construction where a direct positive claim works. Avoid slogan-like summaries; state the mechanism.
- Avoid AI-style parallel error lists. Describe the process and show how an early wrong state propagates into later actions.
- Use bold sparingly, for conceptual pivots only. Use short sections with explicit headings.
- Terminology: advantage, return, reward model, behavior cloning, value function, action value, rollout, trajectory, verifier, process reward model (PRM), outcome reward model (ORM), KL penalty, importance sampling ratio, group-relative advantage, test-time compute, credit assignment. Keep standard acronyms (PPO, GRPO, DPO, RLHF, RLVR, SFT, MDP, TD, GAE, MCTS) and expand them on first use in a chapter.
- Use kebab-case file names under `docs/`; prefer directory routes with `index.md` for multi-page sections.

## Commit Style

- Use Conventional Commits such as `feat:`, `fix:`, `docs:`, `chore:`.
