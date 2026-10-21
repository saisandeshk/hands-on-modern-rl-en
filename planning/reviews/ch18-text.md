# Review of ch18-text

Scope reviewed: the five Chapter 18 pages on `w/ch18-text`
(`docs/chapter18_llm_rl_industrial/{single-machine-to-industrial,industrial-post-training,modern-industrial-practice,distributed-sync,data-engineering}.md`),
the new `docs/chapter18_llm_rl_industrial/review.md`, and `planning/reports/ch18-text.md`.
Every changed number was recomputed with `.venv/bin/python` (6ND GPU-hours, 13B/70B memory
accounting, ZeRO saving factors, the GRPO group statistics, the two-action KL/entropy example,
the stale-policy ratio, and both code exercises in `review.md`, which I re-ran), and the five
pages were read in full.

## Errors

1. `docs/chapter18_llm_rl_industrial/single-machine-to-industrial.md` §3.3 — "if 30% of generated
   groups carry no learning signal and each training token costs roughly three times a rollout
   token (forward plus backward versus forward only), cutting the zero-signal fraction from 30%
   to 10% saves more than a 10% faster trainer" — the two numbers given in the sentence produce a
   **tie**, not a win. Count 1 unit for one rollout token and 3 units for one training token.
   Baseline, 100 groups: rollout 100 + training $100\times3=300$ → 400 units. Filtering the
   zero-signal groups (generation is still paid, only the training on them disappears):
   $100 + 90\times3 = 370$ units, i.e. a 7.5% saving. A 10% faster trainer:
   $100 + 300\times0.9 = 370$ units, i.e. also exactly 7.5%. Under the alternative reading
   ("dynamic sampling keeps generating until the batch is full of useful groups", so the useful
   count is fixed at 70): the baseline is $100+70\times3 = 310$ units, the filtered run is
   $70/0.9 + 70\times3 = 287.8$ units, and the faster trainer is $100 + 210\times0.9 = 289$ units,
   i.e. 7.2% versus 6.8% — still a wash. The claim only holds if "saves" is
   reinterpreted as *useful learning signal per unit of compute* (0.243 vs 0.175 useful groups per
   unit, +39% vs +8% for the faster trainer), which is a different statement.
   — Suggested fix: "cutting the zero-signal fraction from 30% to 10% costs about the same as a
   10% faster trainer (both save ~7.5% of the step), but it buys roughly 39% more useful learning
   signal per unit of compute, because the saved compute is the wasted training pass."

2. `docs/chapter18_llm_rl_industrial/single-machine-to-industrial.md` §3.3 — "$$C_{\text{total}} =
   C_{\text{sample}} + C_{\text{inference}} + C_{\text{training}}.$$" with "$C_{\text{sample}}$
   covers discarded work: prompts filtered by difficulty, all-correct/all-wrong groups, and retries
   after environment failure. $C_{\text{inference}}$ covers rollout generation and judge scoring.
   $C_{\text{training}}$ covers forward-backward updates of the Actor (and Critic, if any)." — the
   three buckets are not a partition: the rollout generation of an all-correct/all-wrong group is
   simultaneously "discarded work" ($C_{\text{sample}}$) and "rollout generation"
   ($C_{\text{inference}}$), so the same tokens are counted twice, and the worked example in the
   next sentence depends on which bucket they land in. The reference-model forward pass is also
   missing from all three buckets (it is not sampling, not judge scoring, not an actor update),
   even though §3.2 of the same page counts it as a first-class cost
   ($C_{\text{ref-forward}}$). The new decomposition therefore contradicts the page's own
   decomposition of the same step.
   — Suggested fix: state the partition explicitly, e.g. "$C_{\text{sample}}$ = generation of
   rollouts that are thrown away; $C_{\text{inference}}$ = generation of the retained rollouts plus
   judge scoring; $C_{\text{training}}$ = actor forward-backward plus the reference-model forward",
   and note that $C_{\text{total}}$ here is the same quantity as $C_{\text{RL-step}} + C_{\text{sample}}$
   in §3.2.

3. `docs/chapter18_llm_rl_industrial/distributed-sync.md` — the rewritten table row "| ZeRO-3
   training-state sharding | ~35 GB + activations | 10%–15% slower | Yes, tight |" is not
   supported by the numbers on the same page. The memory table a few lines above lists
   "Activations | BF16/FP16 | Dynamic | ~100 GB" for the 70B model (Total ~1.22 TB), and
   activations are **not** sharded by ZeRO-3 (only parameters, gradients and optimizer state are).
   $35 + 100 \approx 135$ GB does not fit on an 80 GB H100, so plain ZeRO-3 is not feasible at that
   activation budget; only the next row (with gradient checkpointing, which cuts activations by
   roughly $\sqrt{L}$) is. The same table also drops the "+ activations" term from the LoRA row
   ("~8 GB"), although activations are still there.
   — Suggested fix: make the activation budget explicit, e.g. "| ZeRO-3 training-state sharding |
   ~35 GB + activations (≈100 GB at the batch size above) | … | No without gradient checkpointing |"
   and add "+ activations" to the LoRA row, or state the micro-batch the numbers assume.

4. `docs/chapter18_llm_rl_industrial/distributed-sync.md` (new "Colocated versus Disaggregated
   Deployment" paragraph) — "Nothing crosses the network and every GPU works in each phase" — both
   halves are wrong. (a) Even colocated, the updated weights must be copied from the training
   workers to the rollout engine, and the rollouts, tool returns and rewards must still travel
   between processes — across the network whenever the cluster has more than one node. What
   colocation removes is the *inter-pool weight broadcast*, not all transfers. (b) In colocation
   the same GPUs alternate between the rollout role and the training role, so a GPU cannot work in
   both phases at once: the step time is the **sum** of the two phases, and the overlap that the
   next paragraph advertises for disaggregation ("rollout GPUs produce batch $k+1$ while the trainer
   updates on batch $k$") is exactly what colocation cannot do. The text therefore sells colocation
   with the one property that belongs to disaggregation.
   — Suggested fix: "Colocation needs no separate GPU pool and no inter-pool weight broadcast, but
   the rollout and the update phases cannot overlap: a step costs rollout time plus update time,
   and each switch pays an unload/reload cost."

5. `docs/chapter18_llm_rl_industrial/distributed-sync.md` (new queue subsection) — "Every rollout
   therefore carries an ID of prompt, policy version, and attempt number: retries after a crash or
   timeout resume under the same ID so they are neither counted twice in advantages and pass rates
   nor mistaken for fresh samples." — self-contradictory: if the attempt number is part of the ID,
   a retry has a *different* ID and cannot be deduplicated by ID at all; if retries keep the same
   ID, the attempt number cannot be part of it. The new `review.md` Exercise 8 makes the
   consequence concrete: it asks to "deduplicate retries by `id` (keep first)" and its `drain()`
   skips only on an exact `id` match, so with the ID as defined in 18.4 (prompt + policy version +
   attempt number) the retry of prompt `a` would carry a different id and would be trained on
   twice — precisely the double counting the sentence tries to prevent. (In the exercise's own
   bookkeeping a crashed attempt is diverted to the `failed` list before the id is registered, so
   the successful retry still survives; the two texts simply describe different ID schemes.)
   — Suggested fix: make the dedup key `(prompt_id, policy_version)` and keep `attempt` as a
   separate field of the record ("keep the latest successful attempt, count attempts separately");
   then state that rule identically in 18.4 and in Exercise 8, and have `drain()` prefer the
   successful attempt over an earlier failed one.

6. `docs/chapter18_llm_rl_industrial/distributed-sync.md` (new queue subsection) — "Rollout workers
   should bucket prompts by expected length so one long sequence does not hold a whole batch
   hostage (the straggler row in the table below)." — there is no straggler row in the table that
   follows (it is the MoE "Challenge / Symptom / Mitigation" table); the straggler row is in the
   third table that follows, "Locating Performance Bottlenecks" ("Stragglers | Longest sequence
   determines the batch time | Length bucketing or predicted-length grouping"). The pointer sends
   the reader to the wrong table.
   — Suggested fix: delete the parenthetical, or point explicitly at the "Locating Performance
   Bottlenecks" table.

7. `docs/chapter18_llm_rl_industrial/review.md` — the new page is not registered in
   `docs/.vitepress/config.mjs` (the branch touches no config file), so it is unreachable from the
   site and breaks the repository rule that every navigable page must appear in the hand-written
   sidebar. The page number is also wrong for this book: every other chapter summary page is
   "N.M Summary and Exercises" matching its sidebar entry (1.4, 2.4, 3.4, 4.4, 5.6, 10.4), and
   Chapter 18 already uses 18.1–18.5, so this page is 18.6 — "18.R" has no home, and
   `scripts/fix-heading-numbers.mjs` only rewrites the H1 of pages that are listed in the sidebar,
   so the mismatch will not be reconciled automatically when the entry is added.
   — Suggested fix: add `{ text: '18.6 Summary and Exercises', link:
   '/chapter18_llm_rl_industrial/review' }` to the Chapter 18 sidebar group and change the front
   matter and H1 of `review.md` to "18.6 Summary and Exercises".

8. `docs/chapter18_llm_rl_industrial/modern-industrial-practice.md` (pre-existing, but inside the
   writer's scope) — "A 7B model runs GRPO on the MATH dataset for 200 steps." and, eleven lines
   later, "At step $312$ the loss suddenly became NaN" — a 200-step run cannot fail at step 312.
   The same paragraph is quoted again in the "Data and Rewards" section ("the first 200 steps"), so
   the contradiction is visible inside one page. The writer's report claims all five pages were
   worked through in full and that nothing needed to be flagged.
   — Suggested fix: make the run length consistent, e.g. "for 400 steps" and keep "At step 312", or
   move the NaN to "at step 112".

9. `docs/chapter18_llm_rl_industrial/review.md` Exercise 1 solution — "(4) New weights broadcast
   back to rollout (and reference) GPUs, typically over NCCL" — the reference model is a frozen
   copy of the SFT model; its weights never change, so no weight synchronization is performed for
   it. A reader who implements the described loop will look for a sync that does not exist.
   — Suggested fix: "(4) New weights are broadcast back to the rollout GPUs, typically over NCCL;
   the frozen reference model needs no sync."

10. `docs/chapter18_llm_rl_industrial/review.md` Exercise 7 — the exercise asks the reader to
    "verify by hand" case (b), but the solution stops at the second token's term
    ("$\min(-0.75, -0.60) = -0.75$") and never gives the resulting loss. The two terms are
    $[1.0, -0.75]$, their mean is $0.125$, and the loss is $-0.125$ (the reference implementation
    returns exactly that). Every other exercise in the file ends with the final number, so this one
    is incomplete as an answer key.
    — Suggested fix: add "so the terms are $[1.0, -0.75]$, the mean is $0.125$, and the loss is
    $-0.125$."

## Doubts

1. `docs/chapter18_llm_rl_industrial/modern-industrial-practice.md` — "Public results show that this
   combination lowered loss spikes from about once per 1T tokens to once per 10T tokens, and
   improved training speed by about 15% relative to Adam." I could not verify the 15% figure
   offline; my recollection of the Kimi K2 report is a larger wall-clock factor for MuonClip
   (something near 1.5×). Both numbers are exactly the kind of vendor-specific figure that should
   be checked against arXiv:2507.20534 before it stays in the text.

2. `docs/chapter18_llm_rl_industrial/modern-industrial-practice.md` — "GLM-4.5 (Zhipu AI, released
   in July 2025) and GLM-4.6 (released in October 2025)". My recollection is that GLM-4.6 was
   published at the end of September 2025, which would make "October 2025" off by a few days. Worth
   a check; the writer listed this file as unverified but did not change it.

3. `docs/chapter18_llm_rl_industrial/distributed-sync.md` — the asynchronous-framework table keeps
   "**10.4×** over synchronous Llama-3-70B GRPO" (LlamaRL), "**2.77×** for 671B MoE GRPO" and the
   "clipped to $[0.8,1.2]$" token weight (AReaL), and "1024 GPUs". These are the load-bearing
   numbers of that table and they are reproduced without a source. `review.md` Exercise 4 then
   builds an exercise on the AReaL clip range, so an incorrect range would propagate into the
   exercise answers.

4. `docs/chapter18_llm_rl_industrial/industrial-post-training.md` — the page carries very precise
   vendor parameters that no reader can check from the text: "about 13K SFT samples, batch size
   32, learning rate 1e-5, and trains for 4 epochs", "about 12K samples for GRPO, 8 rollouts per
   group, batch size 64, learning rate 1e-6 … 128K … 100 tool-call rounds",
   "R = 0.2 * R_format + R_correct", "5 rounds of evolution and ultimately synthesizes about 40K
   WebExplorer-QA examples", "more than 10,000 runnable PRs, and more than 140,000 variable
   tasks", Hunyuan-T1's "96.7% of post-training compute", Skywork-OR1's "57.8% to 72.8% for 32B …
   43.6% to 57.5% for 7B", MiMo's "130K verifiable problems", MiniCPM 5's "about 30,000 high-quality
   seed prompts", Qwen-AgentWorld's "5,000 seed tasks … 30,000 tasks over two weeks … 60 GPUs", plus
   arXiv IDs 2601.04962, 2602.02779, 2602.15763 and 2607.07508. The writer's own report lists these
   as "needs external verification" and then leaves every one of them in the text. A pass that
   rewrites a chapter should either verify these or mark them as unverified in the page.

5. `docs/chapter18_llm_rl_industrial/single-machine-to-industrial.md` §3.2 — "In a typical
   configuration with a 7B model and a batch of 512 prompts × 8 rollouts, rollout generation
   accounts for 50%–60% of total computation." With the chapter's own FLOP model (rollout = one
   forward pass = $2ND$, actor update = $6ND$, reference forward = $2ND$) the rollout share of
   FLOPs is $2/(2+6+2) = 20$ percent; 50–60% is reachable only as a *wall-clock* share, because
   autoregressive decoding runs at a far lower MFU than training. Since the new §3.3 speaks of
   "total compute" in a cost sense and repeats the same three-bucket split, the ambiguity is now
   load-bearing. It may be worth saying "50%–60% of wall-clock time" explicitly.

6. `docs/chapter18_llm_rl_industrial/data-engineering.md` — "Store the mask alongside the tokens so
   that rescoring old samples under a new reward version does not silently change what was
   trained." The loss mask is a function of the trajectory, not of the reward version, so
   re-scoring changes the *rewards*, not the mask; the stated causal link does not hold as written.
   The surrounding advice (keep the mask with the tokens, so that a re-scored sample is still
   masked the same way) is sound — only the reason given is muddled.

7. `docs/chapter18_llm_rl_industrial/distributed-sync.md` — the new "Generation batching" bullet
   ("one long sequence does not hold a whole batch hostage") sits in tension with the page's own
   vLLM table, which lists Continuous Batching as the cure for "Static batches wait for every
   sequence". In an RL rollout the batch still ends when its longest sequence ends, so length
   bucketing does help, but the page should say why the two statements do not conflict (inference
   throughput vs. batch completion).

8. `docs/chapter18_llm_rl_industrial/review.md` Exercise 8 — the `drain()` implementation silently
   drops entries beyond `max_size` (they are neither returned nor counted), so the exercise's
   "backpressure" behaviour is invisible in the returned tuple; the spec says "take at most
   `max_size` entries", so this is within the letter of the exercise, but a reader who wants to
   observe backpressure needs a counter for capacity-dropped entries.

9. Process: `AGENTS.md` requires that a factual fix be recorded in
   `planning/content-review-notes.md` ("When fixing a factual error, record it (and the fix) in
   `planning/content-review-notes.md`"). The branch makes three factual fixes (the z-score range
   claim in 18.1, the 940 GB → 1.12 TB memory baseline in 18.4, the ZeRO saving factors in 18.4)
   and the file is not touched; the report nevertheless states "Known suspected issues: (none
   recorded) — nothing to mark `[x]`". The 200-step/step-312 contradiction in 18.3 and the
   18.5 title mismatch (sidebar "Data Engineering for Large-Scale RL" vs. page
   "Large-Scale RL Data Engineering") are also in scope and were not recorded.

## Verdict

The revision is careful and mostly sound: I re-derived every changed number and it holds — the
z-score fix in 18.1 is right (a z-score does not bound the value), the 70B accounting is right
($70\times10^9\times16\,\mathrm{B} = 1.12$ TB, $1120/32 = 35$ GB per GPU, ZeRO-1/ZeRO-2 limits of
$4\times$/$8\times$ follow from $16/(4+12/N)$ and $16/(2+14/N)$), the loss-mask paragraph in 18.5
is correct and correctly cross-references the Qwen2.5-Math TIR recipe in 18.2, and all eight
`review.md` exercises are numerically correct ($8.4\times10^{19}$ FLOPs, $212{,}336$ s, $58.98$
GPU-hours, 208 GB / 26 GB, $e^{-0.19}=0.827$, $\mu=0.375$, $\sigma=0.4841$, $+1.291/-0.775$,
KL $0.1927\to0.4946$, entropy $0.5004\to0.1985$, loss $-0.25$ and $-0.125$, batch $[a,d,e]$) — the
two Python snippets also run as written. The important problems are concentrated in the new
prose: the headline cost argument in 18.1 §3.3 is a tie under the numbers it supplies and should
be restated as a throughput argument, the new three-bucket cost equation double-counts rollout
tokens and drops the reference-model forward that §3.2 counts, the rewritten 70B table in 18.4
keeps a "Feasible? Yes" verdict that its own ~100 GB activation figure contradicts, the new
colocation paragraph claims the properties that belong to disaggregation, the rollout-ID
definition is self-contradictory and inconsistent with the new Exercise 8, and the brand-new
`review.md` is unreachable because it is not in the sidebar and is numbered "18.R" instead of
18.6. The 200-step/step-312 contradiction in 18.3 shows that the claim of having read all five
pages in full was not true of the details. Fix the ten items above, register the new page, and
push the unverified vendor numbers in 18.2/18.3/18.4 to someone with source access.
