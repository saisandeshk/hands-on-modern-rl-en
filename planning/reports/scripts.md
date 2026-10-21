# Worker report: `scripts` (English-only tooling port)

## Summary

Ported the build and verification tooling from the old bilingual layout
(Chinese docs root + `docs/en/`, `zhSidebar`/`enSidebar`, `PDF_LOCALE`) to the
English-only layout (English pages are the docs root, one `sidebar` in
`docs/.vitepress/config.mjs`). Deleted the bilingual check
(`scripts/check-bilingual-sync.mjs`) and the bilingual release workflow
(`.github/workflows/release-pdf.yml`) via `git rm`. All `scripts/` files were
checked for leftover CJK strings, `locales`/`PDF_LOCALE` branches, and Chinese
fonts/TeX packages; the remaining CJK hits in `scripts/` are intentionally out
of scope (`optimize-assets.mjs` font stack, `readme-visuals/` figure labels —
not listed in the task). `node --test scripts/*.test.mjs` passes 4/4.
`npm run verify` still stops at `format:check` because of 5 pre-existing
failures in files outside this task's scope (see Open issues); every later
verify step passes when run directly.

## Technical fixes

- `package.json`: removed the `verify:bilingual` script entry; description
  changed from Chinese to `Hands-On Modern RL: From Code to Principles
(English edition, adapted from the WalkingLabs original)`; `repository.url`
  now points at the fork `saisandeshk/hands-on-modern-rl-en` (this also fixes
  the sitemap fallback host). Verified by reading the file and by the sitemap
  output host below.
- `scripts/check-bilingual-sync.mjs`: deleted via
  `git rm` (the bilingual coverage check has no meaning with one locale).
- `scripts/verify.sh`: removed the `Checking bilingual coverage...` /
  `npm run verify:bilingual` step. Final output checks
  (`preface/introduction.html`, `sitemap.xml`, `robots.txt`) were kept and
  verified to exist in `docs/.vitepress/dist` after `npm run build`.
- `scripts/check-english-quality.mjs`: now scans every page under `docs/`
  (was `docs/en/`), skipping `.vitepress`, `public`, `node_modules` (was only
  skipping `_archive`). All 8 English-quality checks (CJK chars, full-width
  punctuation, `result reward`, corrupted subscripts, truncated links,
  corrupted section numbers, duplicated term definitions, `more dense`) are
  unchanged. Verified: `npm run verify:english` passes with 0 findings.
- `scripts/generate-sitemap.mjs`: `getPriority` no longer has `zh/en`
  branches (`index.md` → 1.0, `guide/` → 0.8, else 0.6); fallback owner is
  `saisandeshk`. Verified: `npm run sitemap` generates 183 pages with
  English-only root routes, e.g.
  `https://saisandeshk.github.io/hands-on-modern-rl-en/appendix_a_engineering/agentic-rl-infra/`;
  `node --test scripts/generate-sitemap.test.mjs` passes.
- `scripts/book-shared.mjs`: English metadata (`bookTitle`,
  `bookSubtitle`, `bookTagline`, `en-US`/`UTC` build date), author default
  `WalkingLabs; sanbuphy` (was `散步`), new `forkGithubUrl`,
  `originalGithubUrl` (WalkingLabs attribution kept), `bookAdaptedBy`;
  `loadSidebar` returns `config.themeConfig.sidebar['/']` (was
  `locales.zh/en`); `frontMatter` cover/intro/publication-note fully English.
  Verified by grep (no CJK, no `locales`) and by the EPUB/PDF builds consuming it.
- `scripts/build-pdf.mjs`: chapter labels `Chapter N` / `Appendix X` (were
  `第 N 章` / `附录 X`), `Untitled Section`, `Front Matter` / `About This Book`
  / `Edition Notes` / `Contents`, cover `subtitle` + `Adapted by` row, fixed
  curly-quote `id="pdf-book-intro"` (was `id=”...”`, broken HTML), footnote
  heading matcher `References|Further Reading` (was `参考文献|...`), English
  intro/publication-note/chapter-opener copy with attribution + original-project
  link. Verified by grep (no CJK/`titleZh`) and successful `npm run build`.
- `scripts/build-epub.mjs`: `epubLang = 'en'` (was `'zh-CN'`), Georgia font
  stacks (were `Noto Serif CJK SC` etc.), `cover.subtitle` (was `titleZh`) +
  `Adapted by` row, `Untitled`, container default `Output` (was `运行结果`),
  `[Diagram]` (was `[图表]`), `External image` / `[Online image:]` (were
  Chinese), English nav/cover/intro/publication-note with attribution.
  Verified by grep and by a full `node scripts/build-epub.mjs`-equivalent run
  inside the test (EPUB builds, 183-page sidebar).
- `scripts/build-latex-book.mjs`: removed the `PDF_LOCALE`/`isEnglishPdf`
  branches entirely (English-only, no fallback dead code). Fonts are now
  `Libertinus Serif`/`TeX Gyre Pagella`/`DejaVu Sans Mono`/`Noto Sans Mono`
  (were `Songti SC`/`PingFang SC`/`Noto Serif/Sans CJK SC`); removed
  `\XeTeXlinebreaklocale "zh"`; `\contentsname`/`figurename`/`tablename` are
  Contents/Figure/Table; fancy headers/footers English; `renderFrontMatter` is
  the single English version (Copyright and Version, Publication Information
  with Adapted by + Original project, APA/BibTeX citing the adaptation,
  Preface, Edition Notes); `loadSidebar` reads the single sidebar; all
  warnings/placeholders English (`Image omitted`, `Interactive component`,
  `Missing page`, `Skipping a display-math fence...`, Ghostscript note).
  Verified by grep (no CJK, no `isEnglishPdf`/`PDF_LOCALE`/`locales`, no
  `Songti`/`PingFang`/`Noto* CJK`). The `xelatex` compile step itself was not
  executed here (see Open issues).
- `scripts/render-book-asset.mjs`: Mermaid font stack is now
  `Helvetica Neue, Helvetica, Arial, DejaVu Sans` (was `Noto Sans CJK SC`...
  CJK stack). Verified by reading the file.
- `scripts/build-epub.test.mjs`: the old assertion searched the built EPUB
  for the Chinese line `给定状态直接输出动作` (absent from the English docs,
  so the test failed). Replaced with a verified English line that exercises
  the same regression (inline math at end of a list item):
  `docs/chapter07_actor_critic/advantage-function.md:180`
  (`...receives immediate reward $r = +2$`, unique phrase in `docs/`), asserting
  no raw `$r = +2$` remains, exactly 1 `<math`, and the line ends
  `</math></li>`. Note: the unfinished earlier port in `planning/port-partial`
  used the phrase `Current latent, timestep, and prompt`, which does not exist
  anywhere in `docs/` — I did not use it. Verified: test passes (~15 s).
- `scripts/fix-heading-numbers.mjs`: rewritten for the single English sidebar.
  Entry regex now accepts appendix numbers (`A.1`, `D.1.1`) as well as
  `N.M`; for each linked page it checks the front-matter `title:` number
  prefix AND the first `# H1` number against the sidebar number (previously
  only the body H1), remaps sub-heading numbers on mismatch, and supports
  `--check` as an alias for `--dry-run`. Ran in check mode only (no docs
  page was rewritten):
  `node scripts/fix-heading-numbers.mjs --dry-run` over 161 numbered sidebar
  entries reports exactly one finding (see Open issues).
- `.github/workflows/content-quality.yml`: renamed to `Verify content
quality`, watches `docs/**` + `check-english-quality.mjs`, and runs only
  `npm run verify:english` (bilingual job removed).
- `.github/workflows/release-pdf.yml`: deleted via `git rm` (bilingual
  Chinese/English PDF+EPUB release; deployment out of scope).
- `AGENTS.md` (commands section only): now lists the real npm scripts
  (`build`, `verify`, `verify:english`, `verify:code-links`, `sitemap`,
  `fix:headings`, `pdf:build`, `epub:build`, `assets:optimize`, `format`,
  `format:check`, `lint` with one-line notes).

## Other changes

- None. `docs/`, `code/`, and `docs/.vitepress/config.mjs` were not touched.
  `planning/port-partial/` was used read-only as a reference (its
  `build-epub.test.mjs` phrase and several small wording choices were
  independently verified or corrected rather than copied blindly).

## Needs external verification

- None. No papers, benchmarks, quotes, or URLs were invented. The sitemap host
  and footer display text use the fork URL already recorded in `AGENTS.md`
  (`saisandeshk/hands-on-modern-rl-en`); the LaTeX footer display label
  hardcodes that repo path while the hyperlink itself stays dynamic (same as
  the reference port; cosmetic only).

## Sidebar additions

None (no pages added; `docs/.vitepress/config.mjs` untouched).

## Commands run and results

- `npx prettier --write <changed files>` — all in-scope files pass
  (`npx prettier --check package.json scripts/ AGENTS.md .github/` → clean).
- `node --test scripts/generate-sitemap.test.mjs` — pass (1/1).
- `node --test scripts/epub-image-conversion.test.mjs` — pass (2/2).
- `node --test scripts/build-epub.test.mjs` — pass after the assertion port
  (full-book EPUB build, ~15 s).
- `node --test scripts/*.test.mjs` — pass 4/4.
- `node scripts/fix-heading-numbers.mjs --dry-run` — 161 numbered entries, 1
  finding: `[dry-run] front-matter title missing number, set to 8.5:
docs/chapter08_ppo/ppo-game-benchmark.md` (front-matter is `title: PPO Game
Project Practice Guide`, H1 `# 8.5 Hands-on: PPO Game Projects` is correct).
- `npm run lint` — pass. `npm run verify:english` — pass (0 findings).
  `npm run verify:code-links` — pass (84 checked). `npm run sitemap` —
  183 pages. `npm run build` — succeeds (~18 s vitepress render).
- `npm run verify` — FAILS at `format:check` (tail pasted below). All steps
  after it pass when run directly (see Open issues).

Tail of `npm run verify`:

```text
> hands-on-modern-rl@1.0.0 verify
> bash scripts/verify.sh

Checking formatting...

> hands-on-modern-rl@1.0.0 format:check
> prettier --check .

Checking formatting...
[warn] code/chapter15_grpo/verl_code_rlvr/README.md
[warn] code/README.md
[warn] opencode.json
[warn] planning/content-review-notes.md
[warn] README.md
[warn] Code style issues found in 5 files. Run Prettier with --write to fix.

Exited with code 1
```

It does not say `Verification passed` for the reason in Open issues (1).

## Open issues

1. **`npm run verify` cannot reach `Verification passed`.** `format:check`
   fails on 5 files outside this task's scope, all unmodified in this worktree
   (pre-existing at HEAD): `code/chapter15_grpo/verl_code_rlvr/README.md`,
   `code/README.md`, `opencode.json`, `planning/content-review-notes.md`,
   `README.md`. I must not edit them (scope) and must not redefine
   `format:check` to dodge them. Coordinator decision needed: run a repo-wide
   `prettier --write` or assign the files to their owners; everything my scope
   controls is format-clean, and every post-format verify step (lint, english,
   code-links, build, dist assertions) passes.
2. **Heading-number finding (check mode only):**
   `docs/chapter08_ppo/ppo-game-benchmark.md` front-matter title lacks the
   `8.5` number (sidebar `8.5 Hands-on: PPO Game Projects`, H1 already
   `8.5 ...`). `docs/` is out of scope and I ran fix mode never, so this is
   left for the chapter owner.
3. **Offline book compiles not executed here.** Per task instructions xelatex
   was not run; `build-latex-book.mjs` TeX generation is ported but
   uncompiled — needs a runner with texlive to validate `book.tex`. Tool
   fallbacks, verified by code reading (Chrome, rsvg-convert, ImageMagick,
   cwebp absent here): `build-pdf` requires Chrome and errors clearly without
   it; EPUB cover falls back to the logo PNG; Mermaid/SVG assets fall back to
   static placeholder notes; gif/webp conversion tries
   magick→convert→ffmpeg→sips and the full EPUB build succeeded in this
   environment. `zipinfo`/`unzip` are required for the EPUB test and exist here.
4. **Release automation gap.** Deleting `release-pdf.yml` removes all
   PDF/EPUB release jobs. If the fork wants book artifacts, someone needs to
   author a new English-only release workflow (deployment, out of scope).
