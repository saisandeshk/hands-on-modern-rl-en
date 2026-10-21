# Report: figures-b — Translate Chinese text in figures (Ch. 14–17, 20–22)

## Summary

Translated every piece of Chinese text into English in all 35 in-scope figure files: 30 `.svg` plus 5 TikZ `.tex` sources
(`docs/chapter14_dpo/`, `docs/chapter15_grpo/`, `docs/chapter16_reasoning/`, `docs/chapter17_prm_search/`,
`docs/chapter20_swe_agents/`, `docs/chapter21_deep_research/`, `docs/chapter22_computer_use/`).
Hand-authored SVGs were reworded concisely against the terminology of the pages that embed them (checked via the
`<img>` alt texts and surrounding sections, e.g. "direct answering / extended reasoning under a budget constraint").
For the four matplotlib-generated SVGs, Chinese glyphs exist only as vector paths with Chinese `<!-- -->` comments,
so each Chinese path group was replaced with a real `<text>` element (Latin labels such as "Reward Accuracy" were
already paths and were left untouched). All `.tex` sources had CJK packages/fonts replaced with `lmodern`.
No `.md` pages, code, config, or raster images were touched. Verified: CJK sweep clean, all 29 SVGs in the touched
directories parse as XML, `npx vitepress build docs` succeeds. Rendered SVG outputs from `.tex` were not regenerated
(no LaTeX toolchain here).

## Technical fixes

No factual/technical corrections were needed (translation-only task). Per-file translation notes:

- `docs/chapter14_dpo/images/dpo-from-ppo.svg` — title/desc/H1, three path boxes ("LM RL objective / stay close to the
  reference", "PPO / RLHF path / sample responses online / score with reward model", "DPO path / use offline
  preference pairs / recover implicit reward"), bottom KL-constraint line, three state/action/trajectory chips
  (13 px to fit); fonts → Inter stack.
- `docs/chapter14_dpo/images/dpo-reward-accuracy.svg` — x-label → "Training steps"; annotation → "0.5 (chance level)"
  (kept "0.5" vector digits, replaced only the parenthetical with `<text>`).
- `docs/chapter14_dpo/images/dpo-reward-margin.svg` — x-label → "Training steps"; annotation → "margin = 0 (no separation)".
- `docs/chapter14_dpo/images/dpo-training-loss.svg` — x-label → "Training steps"; annotation → "ln2 ≈ 0.69 (chance)".
- `docs/chapter14_dpo/images/sigmoid-dpo-loss.svg` — x-label → "Reward margin"; "x=0 时" → "at x = 0"; "梯度弱" →
  "weak gradient" (7 px); "梯度强" → "strong gradient" (7 px); widened the two small tag boxes (green to x≈412,
  pink to x≈164) to fit; Latin annotations ("loss ≈ 4.0", "loss → 0", "loss = ln2 ≈ 0.69") kept as paths.
- `docs/chapter15_grpo/images/grpo-from-ppo.svg` — title/H1, PPO box ("one response gets reward R / critic predicts
  V(s)"), GRPO box ("sample a group per prompt / group mean as baseline"), shared box ("sample from old policy /
  clip limits update size"), one-line contrast with critic-vs-groupmates question; fonts → Inter stack.
- `docs/chapter16_reasoning/images/adaptive-thinking-budget.svg` — "New request / Generate / call tools / gain new
  evidence / Enough evidence? / value of more compute / Stop and answer / hard caps on tokens, time, and tool calls
  always apply".
- `docs/chapter16_reasoning/images/cot-visibility-layers.svg` — "Private reasoning / drafts and revisions / System
  action trace / tools, time, states / Access and redaction / what may be shown / User explanation / key basis and
  conclusion / Internal audit / replay and failure analysis".
- `docs/chapter16_reasoning/images/hybrid-thinking-routing.svg` — "One model / shared weights / Answer directly /
  simple tasks, low latency / Think it through / hard tasks, long chains / Thinking budget / caps length and cost /
  Final output".
- `docs/chapter16_reasoning/images/r1-zero-training-loop.svg` — "Base model / no CoT warm start / Group sampling /
  G answers per problem / Rule-based check / correctness + format / Advantage + GRPO update / upweight
  high-reward traces / updated policy keeps sampling".
- `docs/chapter16_reasoning/images/reasoning-model-loop.svg` — "Problem / Direct generation / early errors propagate
  / Reasoning model / lists conditions, tries methods / Check and revise / switch path on error / Final answer".
- `docs/chapter16_reasoning/images/test-time-compute-options.svg` — "Fixed weights / more inference budget / Sample
  in parallel / Revise sequentially / Search over steps / Verify, select, aggregate / turn extra compute into accuracy".
- `docs/chapter17_prm_search/images/discriminative-prm-pipeline.svg` — "Problem + prefix / keep the context / Step
  splitting / Step labels / human or auto / Train classifier / cross-entropy loss / Class probabilities / rerank or
  RL step reward".
- `docs/chapter17_prm_search/images/formal-verifier-pipeline.svg` — "Problem in words / Formal statement / Lean-Coq-
  Isabelle kept / Draft proof / tactics and subgoals / Kernel check / fixed formal rules / Reward / accept 1 /
  reject 0 / formal claim must match the original problem".
- `docs/chapter17_prm_search/images/generative-prm-pipeline.svg` — "Problem + step / with prior context / Write a
  critique / recompute and cite rules / Locate the error / keep NL rationale / Final judgment / for rerank, search,
  debug".
- `docs/chapter17_prm_search/images/inference-search-tree.svg` — "Shared problem + prefix / computed once /
  Candidate step A / high score: expand / Candidate step B / low score: stop / Candidate step C / keep exploring /
  next layer / revisit later".
- `docs/chapter17_prm_search/images/outcome-process-feedback.svg` — "Step 1/2 correct / Step 3 wrong / error spreads
  from here / Final answer right / possibly by luck / Outcome reward 1 / Process feedback finds first error".
- `docs/chapter17_prm_search/images/parallel-reasoning-aggregation.svg` — "One problem / Path 1–4 / Normalize +
  aggregate / merge equivalents first / Final output / majority vote / verifier / or coordinator model" (also fixed
  a dropped `}` in the pasted `<style>` block).
- `docs/chapter20_swe_agents/images/cwm-vs-deepswe.svg` (+`.tex`) — "Executable software env / Python / Docker /
  tests / CWM: learn how software behaves / observation-action traces → mid-training / multi-task verifiable RL →
  open weights / focus: how actions change the env / DeepSWE: long-horizon code agents / R2E-Gym tasks → long
  rollouts / real test feedback → pure RL / focus: scale rollouts, test-time search / Extension: world / value
  models, MCTS / beam search (needs own experiments)".
- `docs/chapter20_swe_agents/images/ssr-self-play-loop.svg` (+`.tex`) — "Sandbox repo / tests pass initially /
  Bug-injection phase / inject a valid bug / plus a test patch / done by the current model / Task validity check /
  orig passes, buggy fails / Bug-fixing phase / read repo + failing tests / write patch, run for real / same model
  again / test rewards update the policy and select harder tasks".
- `docs/chapter20_swe_agents/images/swe-rl-data-loop.svg` — "From software history to one SWE-RL update / Real
  issues and PRs / SWE-bench: real eval / Software history / SWE-RL: recovered signal / Synthetic bugs / self-play
  / SWE-smith and SSR / Standardized tasks / repo snapshot + issue / isolated env + tests / unified state-action
  logs / Agent rollout / read · search · edit / run tests · revise / submit patch / Verifier / apply patch / run
  tests / reward + logs / Updated policy samples the next batch / Guardrail: read-only eval tests; log env errors
  apart (16 px to fit) / no test deletion, skipped runs, or env failure as progress".
- `docs/chapter20_swe_agents/images/swe-rl-verifier-loop.svg` (+`.tex`) — "Task input / repo snapshot + issue / Code
  agent / read files → edit code / run tests → fix errors / Candidate patch / one full rollout / Executable verifier
  / fail-to-pass tests / pass-to-pass tests / test results become rewards that raise the next fix rate".
- `docs/chapter21_deep_research/images/browser-rl-harness-loop.svg` — "Browser RL Harness: web interaction as
  trainable trajectories / Research question / task, budget, format / Research agent / propose queries / pick pages
  / save evidence / submit answer / actions: search / open / click / Web environment / search API / text extraction
  / Playwright browser / cache · timeout · retry / snapshots and error codes / obs: results, text, screenshots, tool
  errors / Verifier / answer correctness / citation support / steps and cost / reward + logs / Trainer upweights
  high-reward traces, then samples new questions".
- `docs/chapter21_deep_research/images/deep-research-eval-protocol.svg` — "A reproducible Deep Research eval pipeline
  / 1. Freeze tasks / questions + aliases / reference evidence, dates / train / test split / 2. Fix conditions /
  search + browser versions / tool permissions / steps · tokens · time / 3. Run trajectories / search · open · read
  / save page snapshots / submit answers + cites / 4. Score by layer / answer · cites · process / cost · reliability
  / keep subscores + reasons / Failure atlas: split the score into fixable issues / missing page · bad extraction ·
  reasoning error · citation drift · over budget · env noise / keep pass/fail traces per class; rerun only the
  affected slice / Update the item set, harness, or model from stable failures; run the next round".
- `docs/chapter21_deep_research/images/deep-research-evaluation.svg` (+`.tex`) — "Research question / Search and read
  / produces browsing traces / Final answer / with citations / Is the process real? / steps and cost / Does evidence
  support it? / key claims / Is the answer correct? / semantically equivalent".
- `docs/chapter22_computer_use/images/gui-agent-one-step.svg` — mock form fields → "Email address / Verification
  code / Submit"; "Observation o_t: screenshot"; "1. Find target / next object: "Submit" / 2. Locate action /
  click(214, 209) kept / 3. Env executes / page loads or errors / 4. Verify result / new screenshot as expected? /
  If the page did not change as expected, re-observe and correct".
- `docs/chapter22_computer_use/images/gui-agent-training-loop.svg` (+`.tex`) — "Task description / Screenshot /
  observation o_t / Ground + decide / policy πθ / Mouse / keyboard / action a_t / UI changes / env transition /
  Task verifier / reward r_t" (math subscripts kept as-is).
- `docs/chapter22_computer_use/images/indirect-injection-path.svg` — "User goal / "Summarize this file" / Untrusted
  PDF / text + "forward it" / Model action / send_email(...) kept / Email / files / payments / real-world side
  effects / Block 1: content boundary (18 px) / PDF supplies data only / never widens the user goal / Block 2:
  action gating / check action and permission / deny cross-app overreach / Block 3: user confirmation /
  before send / delete / pay / show concrete side effects".
- `docs/chapter22_computer_use/images/prompt-injection-defense.svg` (+`.tex`) — "User goal / authorized instruction /
  Web / mail / PDF / untrusted content / Instruction hierarchy (19 px) / resolve conflicts / Action + permission
  check (gate box widened 180→220 px with arrow endpoints adjusted) / Low risk, authorized / execute / Risky or out
  of scope (20 px) / block / ask to confirm".

How verified: re-read each edited file; CJK-range grep (U+4E00–U+9FFF, U+3000–U+303F, U+FF00–U+FFEF) over all 14
scope directories returns nothing in `.svg`/`.tex`; XML parse of all 29 SVGs in touched dirs passes; `npx vitepress
build docs` succeeds. Text-fit was checked by character-count × font-size arithmetic against box widths (see Open
issues for the tight ones). `.tex` files use only `lmodern` + standard TikZ libraries, so they should compile, but
could not be compiled here.

## Other changes

- All CJK font stacks (`Songti SC`, `Noto Sans SC/CJK`, `PingFang SC`, `Heiti SC`) in `font-family` / `<style>` /
  `.tex` preambles replaced with `Inter, Helvetica, Arial, sans-serif` (SVG) or `lmodern` (TikZ).
- Leftover Chinese glyph `<defs>` path data in the four matplotlib SVGs was left in place but is now unreferenced
  (only the Latin/digit glyph paths are still `<use>`d); the `PingFangHK-Regular-*` path IDs are Latin-only
  identifiers, so they do not violate the no-CJK rule. Renaming them was avoided as needless churn.
- `npx prettier --write` reports "No parser could be inferred" for `.svg`/`.tex` — nothing to format; files left as-is.

## Needs external verification

None — no papers, numbers, or URLs were added or altered (translation-only task; terminology follows the embedding pages).

## Sidebar additions

None (no pages added).

## Commands run and results

- CJK grep over each scope dir (`[一-鿿　-〿＀-￯]`): clean in all `.svg`/`.tex` files.
- CJK-font grep (`Songti|PingFang|Noto Sans CJK|YaHei|SimSun|xeCJK|ctex|Heiti`): clean in all touched image dirs
  (remaining `PingFangHK-Regular-*` hits are vector-path IDs, not font references).
- `timeout 60 .venv/bin/python -c` XML-parse check: "OK 29 files parsed".
- `npx prettier --write <35 files>`: no-op, no parser for svg/tex (exit 2, expected; no changes made).
- `npx vitepress build docs`: success in ~17.6 s (only pre-existing warnings about jinja2 highlighting and chunk size).

## Open issues

- Visual layout could not be eyeballed (no browser/screenshot tool here). Tightest fits for the coordinator to check:
  `sigmoid-dpo-loss.svg` ("strong gradient" tag overlaps the legend box's bottom-right corner region — the backing
  box was widened over part of the legend border; "weak gradient" box now touches the plot's right axis area),
  `test-time-compute-options.svg`, `r1-zero-training-loop.svg` ("Advantage + GRPO update" ≈ 216/230 px),
  `swe-rl-data-loop.svg` guardrail line (16 px), `prompt-injection-defense.svg` ("Risky or out of scope" exactly
  fills its box; gate box was widened with arrows adjusted), `indirect-injection-path.svg` block labels (18 px).
- Out-of-scope CJK noticed but not touched (`.md` editing forbidden): `docs/chapter19_agentic/rllm-deepcoder-lab.md:1033`
  contains a fullwidth colon `：` inside a Python regex (`Total cost[:：]`).
- `.tex` sources were translated and de-CJK'd but the SVG renders were not regenerated (no LaTeX here); the checked-in
  `.svg` files were translated directly instead, so tex/svg pairs are consistent by construction.
