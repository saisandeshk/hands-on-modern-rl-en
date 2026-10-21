# Review of ch23-text

Scope reviewed: `docs/chapter23_vlm/vlm-challenges.md`, `docs/chapter23_vlm/qwen3-vl-reflection.md`,
`docs/chapter23_vlm/vlm-grpo-hands-on.md`, `docs/chapter23_vlm/easyr1-geoqa.md`,
`docs/chapter23_vlm/vlm-frameworks.md`, the new `docs/chapter23_vlm/review.md`, and
`planning/reports/ch23-text.md`. Every equation and number in the new material was re-derived and
recomputed with `.venv/bin/python`, every exercise solution in `review.md` was worked out
independently, the two `code/chapter23_vlm/` scripts referenced by the new exercises were executed,
and every figure used by the five in-scope pages was opened and compared with its caption.

The writer's own report is accurate about most of the mechanical fixes (`verl.tainer.main` →
`verl.trainer.main` is correct; the `11.1` → `23.3` table header and the chapter cross-references
15/19/13 are correct; the 448/14 → 1024 patch count, the 108 km/h → 30 m/s → 60 m latency figure,
the 8000/512 ≈ 16 steps/epoch, and the `0.5·format + 0.5·accuracy` log values all check out).
The problems below are elsewhere: two exercise solutions state things the referenced code does not
show, one chapter code snippet leaks the validation set into the training set, and three figure
captions do not describe the images they point at.

## Errors

1. `docs/chapter23_vlm/vlm-grpo-hands-on.md` — §23.3.1 dataset snippet: `img = generate_shape_image(n_tri, n_cir, n_sqr, seed=i)` followed by `train_dataset = generate_dataset(500)` / `val_dataset = generate_dataset(100)` — the "validation" set is almost entirely a copy of the training set.
   `generate_shape_image` calls `random.seed(seed)` on the **global** RNG. Because the seed is the
   sample index `i`, the i-th sample produced by *any* call to `generate_dataset` is a deterministic
   function of `i` (from the second sample on, the state left by the previous call is re-seeded to
   `i-1`). Reproducing the snippet exactly (PIL 13.0, same code) gives: of the 100 validation
   samples, **91 are byte-identical to a training sample** (same image, same counts, same question);
   only indices 0–8 differ, because those depend on the RNG state left over from the training run.
   The page then instructs the reader that "Accuracy must still be measured on a separate validation
   set" (§23.3.3) and that generalization should be tested "on never-before-seen shape combinations"
   (§23.3.4) — both are impossible with this generator. Note that the repository's own script gets
   this right: `code/chapter23_vlm/geometry_counting_dataset.py` uses `seed = base_seed + i` for
   training and `seed = base_seed + 1000 + i` for the test set, "to avoid overlap with the training
   set". The page's snippet contradicts the code it is supposed to introduce.
   — Suggested fix: use a local generator (`rng = random.Random(seed)` threaded through
   `generate_shape_image`) and derive the validation seeds from a different offset
   (`seed = 100_000 + i` for the validation split), or split the generated samples by index after
   one `generate_dataset(600)` call. Add one line noting that the val split must use disjoint seeds.

2. `docs/chapter23_vlm/review.md` — Exercise 8 solution: "(b) As quality rises, group members are mostly correct responses with similar rewards …, so the within-group spread — and hence the std — falls; the annotation in `plot_training_curves` says the same." — the referenced script does not show that trend for the seed the exercise prescribes.
   Running the exercise's own snippet
   (`create_training_samples(num_samples=20, seed=42)`, then `simulate_grpo_training(samples, group_size=g, num_epochs=5, seed=42)`)
   gives `avg_advantage_std`:
   G=2 → [0.297, 0.267, **0.331**, 0.181, 0.070];
   G=4 → [0.494, **0.586**, 0.448, 0.409, 0.108];
   G=16 → [0.542, **0.611**, 0.599, 0.441, 0.185].
   The metric **rises** from epoch 1 to epoch 2 in all three settings and only collapses in the last
   epochs. A student who follows the instruction ("plot or tabulate `avg_advantage_std` across
   epochs … explain (b)") sees a curve that contradicts the solution. The supporting citation is
   also weak: the quoted annotation is a hard-coded matplotlib string in
   `plot_training_curves`, not an empirical observation.
   The real mechanism is the opposite of what the solution states in the early phase: at low
   quality a group mixes correct responses (reward ≈ 1.1–1.44) with wrong ones (0–0.44), which is a
   *larger* spread than the late-phase homogeneous groups (all correct, spread ≈ 0.34 → std ≈ 0.17,
   and → 0 once every member is the same template). The std therefore first grows and only then
   collapses.
   — Suggested fix: "(b) The curve is not monotone: with the default seed the within-group reward
   std first *rises* (epoch 1 → 2, e.g. G=4: 0.494 → 0.586) because early groups mix correct
   (1.1–1.44) and incorrect (0–0.44) responses, and only collapses in the last epochs (0.108 at
   epoch 5) once the policy is good enough that every group member is the same correct template.
   A shrinking std therefore signals group homogeneity, not necessarily a shrinking advantage
   signal."

3. `docs/chapter23_vlm/review.md` — Exercise 7: "add a length guard that subtracts 0.1 when the response exceeds 300 characters without containing any step keyword", with the solution's expected behaviour "the perfect, short-correct, and reasoning-wrong cases are long-with-keywords or short, so the length guard does not fire" — the guard can never fire on the file's own test set, and the solution discusses only 3 of the 8 cases.
   The eight `test_cases` in `code/chapter23_vlm/multi_modal_reward.py` have lengths
   174, 17, 115, 27, 57, 41, 72 and 108 characters (measured with `len(response)`). The maximum is
   **174 < 300**, so the length guard contributes 0.0 for *all eight* cases, and the "report which
   cases change score and by how much" part of the exercise has a degenerate answer (no case changes
   at all). The solution's phrasing leaves the reader expecting the guard to fire on some of the
   other five cases; it never does.
   — Suggested fix: either state the measured result explicitly ("no shipped case exceeds 300
   characters — the longest is 174 — so the guard is inert on the current test set; to see it fire,
   lower the threshold to 50 characters, where case 5 (57 characters, no step keyword) loses 0.1 and
   drops from 0.55 to 0.45"), or change the threshold to 50 characters and give that delta in the
   solution.

4. `docs/chapter23_vlm/review.md` — Exercise 7 solution: "Clamping at 0 is justified because GRPO normalizes within the group, so negative outliers would distort sibling advantages; the penalty's job is ranking, not unbounded punishment." — the justification is mathematically backwards.
   GRPO's advantage is $A_i = (R_i - \bar R)/\sigma_R$, and both $\bar R$ and $\sigma_R$ are
   computed **within the group**. Adding a constant $c$ to every reward in a group leaves $A_i$
   unchanged, so a shared negative offset cannot "distort sibling advantages" at all. The real
   damage comes from the clamp: if the penalty pushes *all* members of a group to 0, the group
   becomes degenerate ($\sigma_R = 0$, advantages $0/0$), which destroys exactly the ranking
   information GRPO needs. Clamping at 0 can therefore hide the penalty the exercise is asking the
   student to add.
   — Suggested fix: justify the clamp (if kept) by scale, not by advantage distortion: "Clamping at
   0 keeps the reward inside the designed $[0, 2]$ range and keeps the penalty small relative to
   the correctness signal; note that a clamp can collapse a group whose members are all penalised
   (zero within-group std, hence zero advantage), so the penalty magnitude should stay well below
   the within-group reward spread."

5. `docs/chapter23_vlm/review.md` — "Group-relative advantage (the GRPO core used in Sections 23.3 and 23.4): $A_i = (R_i - \mathrm{mean}(R_1,\dots,R_G)) / \mathrm{std}(R_1,\dots,R_G)$" with Exercise 2 asking for "the population standard deviation" — the estimator contradicts the chapter's own code in §23.3.
   §23.3 prints `std_r = rewards_tensor.std(dim=-1, keepdim=True) + 1e-8`. `torch.Tensor.std` defaults
   to the **unbiased** estimator (correction = 1), i.e. the N−1 denominator, whereas the review
   page uses the population estimator (ddof = 0, as in DeepSeekMath and in
   `code/chapter23_vlm/vlm_grpo_train.py`, which uses `numpy.std()`). For the exercise's group
   $R=[1.5,1.0,0.3,0.1]$ the two estimators give
   population: $\sigma = \sqrt{1.2475/4} = 0.55846$, $A = [1.388, 0.492, -0.761, -1.119]$;
   unbiased (what §23.3's code computes): $\sigma = \sqrt{1.2475/3} = 0.64485$,
   $A = [1.202, 0.426, -0.659, -0.969]$.
   The review's worked numbers therefore do not match the code the same chapter prints, and a
   reader who implements §23.3 and checks the exercise gets different advantages.
   — Suggested fix: state the convention in the equation ("population standard deviation, ddof = 0,
   as in DeepSeekMath"), and make §23.3's snippet explicit — `rewards_tensor.std(dim=-1, keepdim=True, unbiased=False)` —
   or change Exercise 2 to the N−1 estimator and recompute.

6. `docs/chapter23_vlm/easyr1-geoqa.md` — §23.4.4 details box: "the original r1v.py's `format_reward` regex `</think\s*>` cannot match `</thinkutan>` (because the Chinese character is not whitespace), so the format reward is always effectively 0" contradicts §23.4.5 and §23.4.7.
   §23.4.5 presents the same file as "The official reward function for GeoQA-8K" with
   `pattern = re.compile(r"<thinkutan>.*?</thinkutan>\s*<answer>.*?</answer>", re.DOTALL)`, i.e. a
   regex that *does* match the template, and §23.4.7 then shows `reward/format=0.45 → 0.61 → 0.68`
   rising during training. Both cannot be true of the same upstream file. The explanation inside
   the parenthesis is also wrong on its own terms: the characters after `think` in `</thinkutan>`
   are the Latin letters "utan", not "the Chinese character"; the regex fails because `\s*` cannot
   consume "utan" and the next character is not `>`.
   — Suggested fix: pick one version of the upstream file and describe it consistently — e.g. keep
   §23.4.5's working regex, and in the details box say that the *upstream* `r1v.jinja`/`r1v.py`
   pair uses `<think>`/`</think>` while the GeoQA-8K R1-V prompts ask for `<thinkutan>`/`</thinkutan>`
   (or vice versa), so that a user running the official scripts should check which tags the reward
   function matches before trusting `reward/format`. Correct "the Chinese character" to "the letters
   `utan`".

7. `docs/chapter23_vlm/easyr1-geoqa.md` — §23.4.12: "The 3B model has a smaller vision encoder, so its geometry reasoning accuracy ceiling may be lower" — false for the Qwen2.5-VL family.
   Qwen2.5-VL uses the *same* vision tower (the window-attention ViT, ≈0.67 B parameters) in the 3B
   and 7B checkpoints; only the language model differs (3B vs 7B). The vision encoder is not
   smaller in the 3B model, so the stated reason for the lower ceiling is wrong — the difference,
   if any, comes from the language model that has to interpret the diagram.
   — Suggested fix: "…the 3B model has a much smaller language model, so its geometry reasoning
   ceiling may be lower; both checkpoints share the same vision encoder."

8. `docs/chapter23_vlm/easyr1-geoqa.md` — §23.4.8 Phase 2 example: "The shaded region contains two triangles, with a total area of 24 cm², so the shaded area is 12 cm²." — arithmetically impossible in the figure it illustrates (Figure 2, `images/geoqa-example.png`).
   In that sample the parallelogram has area $S_\square = AB \times h = 6 \times 4 = 24$ cm² and its
   diagonals split it into **four equal-area triangles of 6 cm² each** (a fact the example itself
   states). Two of those triangles therefore total 12 cm², not 24 cm²; 24 cm² is the area of the
   whole parallelogram. Moreover the shaded region drawn in the figure is a **single** triangle
   ($\triangle AOB$, the dataset's own answer for that item is 12 with the reasoning
   "S△AOB = S□/4 = 6 cm²×2 = 12 cm²", i.e. the dataset item itself is noisy). The narrative in the
   page is inconsistent both with the figure and with the geometry it claims to reason about.
   — Suggested fix: align the example with the figure — "the shaded region is one of the four
   triangles, each with area 24/4 = 6 cm²; the dataset's ground-truth answer for this item is 12"
   (and, if the point is to show a *correct* reasoning trace, use an item whose geometry is
   unambiguous).

9. `docs/chapter23_vlm/qwen3-vl-reflection.md` — Figure 2 caption: "Visualization of visual grounding in VLM-R1. During reasoning the model explicitly outputs bounding-box coordinates, turning visual evidence into a verifiable trajectory." — the image is a reward curve, not a grounding visualization.
   `images/ref-vlm-r1-iou.png` is a single W&B-style line plot with the title
   `train/rewards/iou_reward` against `train/global_step` (rising from ≈0.05 to ≈0.8 over ~540
   steps). It contains no bounding boxes and no model output. The same image is used in §23.3 with
   the correct caption "IoU reward curve from VLM-R1 training logs", which confirms the mismatch is
   in §23.2 only.
   — Suggested fix: replace the caption with "IoU reward curve from the VLM-R1 training logs: the
   rule-based grounding reward rises during RL training" (or swap in a real grounding screenshot).

10. `docs/chapter23_vlm/vlm-challenges.md` — Figure 4 caption: "Ranking of PickScore reward models. Different reward models differ significantly in their performance on visual quality assessment, showing that reward design has a large influence on the final result." — the image is a qualitative generation comparison, not a model ranking.
    `images/ref-pickscore-ranking.png` shows four pairs of generated images captioned "Vanilla" vs
    "PickScore" (flying island, futuristic cabin, bee, lion hybrid, corgi on a bicycle, bunny with
    a guitar, mushroom-inspired car, zebra-skin car) — a qualitative before/after of image
    generation with and without the PickScore preference reward. It does not compare reward models
    and does not show any "ranking". Two further problems with the same block: the figure is not
    referred to anywhere in the prose (it sits between the Visual-RFT paragraph and the sentence
    "How large is the influence of reward design on the final result?", which is answered by the
    VISTA-R1 ablation), and it is numbered **Figure 4** while the VISTA-R1 ablation that follows it
    is **Figure 3**, so the figures appear out of order (1, 2, 4, 3).
    — Suggested fix: caption it "Qualitative comparison of images generated with and without the
    PickScore preference reward" (or drop the figure from this page, which is about VQA reward
    design, not image generation), renumber the figures so that they run 1, 2, 3, 4, and add a
    sentence that refers to the figure.

11. `docs/chapter23_vlm/easyr1-geoqa.md` — §23.4.7 sample log "[Step 1]  train | reward/overall=0.28 | reward/format=0.45 | reward/accuracy=0.11 | kl=0.000 | loss=2.34" and §23.4.8 "Phase 1: Learning format (steps 1-30). `format` quickly rises to 0.8+, while `accuracy` barely moves." — neither matches the figure printed immediately below them.
    `images/easyr1-geoqa-curves.png` (Figure 5) contains four panels — `actor/entropy_loss`,
    `reward/accuracy`, `response_length/mean`, `val/accuracy_reward` — over ~60 steps.
    `reward/accuracy` starts at ≈0.34 at step 1 and rises to ≈0.51 at step 60; there is **no
    format-reward panel at all**, so the "format quickly rises to 0.8+" story cannot be read off
    this figure, and the illustrative log in §23.4.7 (accuracy 0.11 at step 1 → 0.38 at step 10)
    disagrees with the curve of the same run (accuracy ≈0.34 → ≈0.35 over the same steps). The
    three-phase narrative is therefore attached to a figure that does not show it.
    — Suggested fix: either plot the format-reward curve that the phase analysis discusses, or
    restate the phases in terms of the curves that are shown ("`reward/accuracy` climbs from
    ≈0.34 to ≈0.51 over 60 steps while `response_length/mean` stays around 330–350 tokens, so the
    gain comes from accuracy rather than from length inflation").

12. `docs/chapter23_vlm/review.md` — the new page is unreachable and its number breaks the repo's numbering tooling.
    `docs/.vitepress/config.mjs` was not touched, so the sidebar still lists only 23.1–23.4 and the
    new `review.md` cannot be reached by navigation (AGENTS.md: "Every navigable page must be
    listed there"). On top of that, the title `23.R Summary and Exercises` breaks the house
    convention — every other chapter review page is numbered (`1.4 Summary and Exercises`,
    `5.6 Summary and Exercises`, `7.8 Summary and Exercises`, `10.4 Summary and Exercises`) — and
    `scripts/fix-heading-numbers.mjs` cannot parse a sidebar entry whose number is not numeric: its
    regex is `text:\s*'((?:[A-Z]\.)?\d+(?:\.\d+)*)\s+([^']+)'`, which does not match
    `text: '23.R Summary and Exercises'` (verified by executing the regex). With such an entry the
    number-sync tool would silently skip the page forever, so its section number could never be
    validated or renumbered.
    — Suggested fix: title the page `23.5 Summary and Exercises` (H1 and front matter) and add
    `{ text: '23.5 Summary and Exercises', link: '/chapter23_vlm/review' }` to the Chapter 23
    sidebar block.

13. `docs/chapter23_vlm/qwen3-vl-reflection.md` — Figure 2 source link `https://github.com/River-Zhang/VLM-R1` — wrong repository URL for a project the book cites elsewhere as `https://github.com/om-ai-lab/VLM-R1`.
    Every other occurrence in the book points to the organisation account: §23.1 (twice), §23.3
    (twice) and Appendix C (four times) all use `om-ai-lab/VLM-R1`. `River-Zhang` is the first
    author of the VLM-R1 paper, not the organisation that hosts the repository, so this link points
    at a personal account that does not host the project.
    — Suggested fix: use `https://github.com/om-ai-lab/VLM-R1` in the Figure 2 caption, exactly as
    in §23.1 and §23.3.

## Doubts

1. `docs/chapter23_vlm/qwen3-vl-reflection.md` — footnote `perception_rft`: "Harikrishnan P M, et
   al., *Stop Thinking, Start Looking: Efficient Post-Training for Multimodal Document Question
   Answering via Reasoning-Free Alignment* (https://arxiv.org/abs/2607.14682), 2026", used for the
   claim that "training without explicit reasoning outperformed the reasoning variant for a 4B
   model". The arXiv identifier (2607 = July 2026) and the result are plausible for the book's
   present date, but I cannot verify the paper offline; the very specific "reasoning-free beats
   reasoning for a 4B model" claim should be checked against the paper before it stays in the text
   (and `review.md` repeats the framing as a fact).

2. `docs/chapter23_vlm/qwen3-vl-reflection.md` — "The official report also gives a native 256K
   interleaved multimodal context and provides dense models of 2B, 4B, 8B, and 32B and MoE models of
   30B-A3B and 235B-A22B", sourced to arXiv 2511.21631 and the Qwen3-VL repository. Sizes, MRoPE
   layout, DeepStack and text timestamps all match my knowledge of the Qwen3-VL release, but the
   report itself could not be opened; please re-check the model list and the context length.

3. `docs/chapter23_vlm/vlm-frameworks.md` — "On the reference Qwen-Image FlowGRPO setup, end-to-end
   throughput is roughly **~25% higher** than the diffusers-based `flow_grpo` implementation", plus
   the "integrates vLLM-Omni for fast rollout" claim. This is a vendor benchmark quoted at a
   precision nobody outside the project can check; if it stays, it should be attributed to the
   project's own benchmark page rather than stated as fact.

4. `docs/chapter23_vlm/easyr1-geoqa.md` — operational details that will drift: "EasyR1 (4900+
   stars)", the Docker tag `hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`, "Algorithms: … and 7 total",
   "math.py defaults to 0.1" for `format_weight`, and the claim that `r1v.jinja` uses
   `<thinkutan>` tags. I cannot check any of them offline; the star count in particular should be
   rounded ("about 5k stars") so it does not become wrong next month.

5. `docs/chapter23_vlm/vlm-challenges.md` — "**Unified Transformer** uses a shared Transformer for
   visual encoding, fuses through patch embedding, and updates fully in RL, suiting
   resource-constrained scenarios." A unified model that is updated fully during RL is not
   resource-saving relative to freezing a separate ViT; the sentence looks like it was meant to
   describe a different trade-off (fewer parameters, one optimizer, but no cheap freezing).

6. `docs/chapter23_vlm/review.md` — Exercise 7(a) asks for a term that "subtracts 0.3 when the same
   response text would score as correct on a swapped image's ground truth", and the solution
   suggests constructing the swapped labels "by swapping two shapes' counts to match the response's
   numbers". Taken literally, that makes the "swapped" ground truth identical to the response's own
   counts, so the penalty also fires on a perfectly grounded response (case 1's perfect response
   would be penalised for "matching" the swapped labels). The term as specified therefore cannot
   separate prior-guessing from grounding; §23.1's `image_sensitivity_bonus` (penalise when the
   answer is *unchanged* across the swap) is the operative check and would be the better basis for
   the exercise.

7. `docs/chapter23_vlm/review.md` — Exercise 6 asks for "the diagnosis in the order of the Section
   23.1 checklist", but the solution's diagnosis is phrased in terms of "Phase 1 (format learned) …
   Phase 2 (content)", which are §23.4.8's phases, and it does not follow §23.1's checklist order
   (task accuracy → no-image control → general visual probe → answer length → format reward →
   IoU/grounding → KL). Relatedly, learning outcome 3 asks the reader to design "a three-part rule
   reward (accuracy + format + grounding)", while §23.3's three-part reward is
   *correctness + reasoning quality + format* (grounding appears only in the repository script's
   four-component reward and in §23.1's Visual-RFT discussion).

8. `docs/chapter23_vlm/review.md` — Exercise 7's snippet embeds Chinese literals
   (`STEP_KEYWORDS = ['首先', '然后', …]`) in an English-only book. The automated check
   (`scripts/check-english-quality.mjs`) only scans prose outside code fences, so the page passes,
   but the chapter text is otherwise English; consider naming the keywords in English with the
   Chinese ones as a comment, or explain in the prose why the Chinese markers are needed (the
   reward file itself is Chinese-only).

9. `docs/chapter23_vlm/vlm-challenges.md` — the writer changed the front-matter title to the long
   H1 form ("23.1 Why a VLM Can Guess Right Without Looking: Visual Rewards and Hallucination")
   while the sidebar entry still reads "23.1 Visual Reward Design", and 23.3/23.4 were aligned *to*
   the sidebar. `scripts/fix-heading-numbers.mjs` only syncs the numeric prefix, so this mismatch
   will persist silently; either keep the front-matter title equal to the sidebar text (the
   convention used by, e.g., `docs/chapter24_multimodal_frontier/embodied-intelligence/index.md`) or
   update the sidebar entry.

10. `docs/chapter23_vlm/easyr1-geoqa.md` — Figure 5's caption attributes the curves to
    "Qwen2.5-VL-7B trained with EasyR1 GRPO on Geo3K/GeoQA", while the section's own experiment
    uses Qwen2.5-VL-3B on GeoQA-8K; the image carries no labels, so the attribution cannot be
    checked. If the figure really comes from the upstream Geo3K run, say so explicitly, otherwise
    a reader will assume it is the run described in §23.4.6–§23.4.7.

## Verdict

The mechanical pass was careful and mostly right: the `verl.trainer.main` path, the "23.3" table
header, the four chapter cross-references, the "Section 23.3" pointer, the stale outline table in
the supplement, and the "Chapter 9" continuous-control reference are all genuine fixes, and the
arithmetic the writer re-checked (patch counts, GRPO advantages, IoU, unit conversion, epoch count,
reward weights) is correct. The new `review.md` is well structured, but three of its eight exercise
solutions assert behaviour the referenced code does not have (the advantage std does not shrink
monotonically, the 300-character length guard is inert on all eight shipped test cases, and the
clamp justification inverts how GRPO's normalization behaves), and its GRPO equation silently
switches to the population standard deviation while §23.3's own snippet uses the unbiased one.
The most consequential single defect is in §23.3.1's dataset snippet, where the global
`random.seed(i)` makes 91 of the 100 validation samples identical to training samples, which
silently invalidates the page's own instruction to evaluate on a separate validation set. Finally,
three figure captions (PickScore, VLM-R1 IoU, GeoQA curves) do not describe the images they point
at, the upstream `r1v.py` format-regex story contradicts itself between §23.4.4 and §23.4.5, and
the new page is neither in the sidebar nor numbered in a way the repo's `fix:headings` tooling can
parse. Fix the exercise solutions and the dataset snippet first, then the captions and the
`23.5` numbering/sidebar entry.
