# Report: fc-5 — Fact-check of Chapters 19–22 against primary sources

## Summary

Checked the flagged "Needs external verification" items from `ch19a-text`, `ch19b-text`, `ch19-dedupe`, `ch20-text`, `ch20-lab`, `ch21-text`, `ch21-lab`, `ch22-text`, `ch22-lab` against primary sources (arXiv abstracts, official blogs, repos, model cards). Made 10 minimal edits: corrected the Gamma-decay formula to the Kimi-Researcher source, re-grounded the Kimi K2.5 swarm section in the actual tech report, restored three softened numbers now confirmed (Memento 87.88%, CWM 65.8%, Search-R1 relative gains), fixed Web-Shepherd precision in three places, disambiguated the two papers both named "AgentPRM", and fixed a typo. `npx vitepress build docs` passes; no code touched.

## Technical fixes

| Page | Claim (short quote) | Verdict | Source URL |
| --- | --- | --- | --- |
| `chapter19_agentic/industrial-practice.md`, `industrial-evaluation.md` | Gamma-decay as trajectory reward `R = r·γ^T`, e.g. `T=5 → 0.77` | **Corrected** to source: per-step `r_i = r·γ^{T−i}`; efficiency works through early steps (`0.95^4≈0.81` vs `0.95^19≈0.38`); kept do-not-double-discount note | https://moonshotai.github.io/Kimi-Researcher/ |
| `chapter19_agentic/multi-agent-swarm.md` | Kimi K2.5: 32 agents, 12M trajectories, 68.3%/72.1%, 280K vs 18K tokens | **Corrected/hedged**: none of these figures are in the cited report; rewrote to report-grounded PARL description + headline numbers (SWE-Verified 76.8 single-model, BrowseComp 78.4 swarm, 4.5× latency cut); kept old figures only as explicitly unverified secondary-report claims | https://arxiv.org/abs/2602.02276 |
| `chapter19_agentic/multi-agent-swarm.md` | Step 3.7 Flash Advisor 62.4% / 52K | **Hedged**: labeled as secondary-summary figures, re-check before citing | not findable in a primary source |
| `chapter21_deep_research/browser-rl-harness.md` | Search-R1 table `~35%→~76% (+41%)` | **Corrected**: paper reports *relative* gains; table now lists +41%/+20%/+10% relative only, absolutes removed | https://arxiv.org/abs/2503.09516 |
| `chapter21_deep_research/browser-rl-harness.md` | Memento "strong results on GAIA validation" | **Confirmed → restored precise**: 87.88% Pass@3 validation (79.40% test) | https://arxiv.org/abs/2508.16153 |
| `chapter20_swe_agents/world-model-and-deep-swe.md` | CWM row "check the paper's table" | **Confirmed → restored precise**: 65.8% Pass@1 with test-time scaling; 32B dense, Python-interpreter + Docker mid-training | https://ai.meta.com/research/publications/cwm-an-open-weights-llm-for-research-on-code-generation-with-world-models, https://arxiv.org/abs/2510.02387 |
| `chapter19_agentic/credit-assignment.md`, `chapter21_deep_research/browser-rl-harness.md`, `chapter19_agentic/tool-use-and-trajectory.md` | Web-Shepherd "+10.9%", "1/10 cost", "40,000 examples" | **Confirmed + refined**: 10.9pp on WebArena-lite (GPT-4o-mini policy), ~1/10 judge cost, 40K step-level preference pairs, NeurIPS 2025 Spotlight | https://arxiv.org/abs/2505.15277, https://github.com/kyle8581/Web-Shepherd |
| `chapter19_agentic/industrial-evaluation.md`, `credit-assignment.md` | "AgentPRM" cited as 2511.08325 in one page, 2502.10325 in another | **Corrected**: both papers exist and share the name; renamed footnote to `[^agent-prm-pp]` and added reciprocal disambiguation notes | https://arxiv.org/abs/2511.08325, https://arxiv.org/abs/2502.10325 |
| `chapter21_deep_research/deep-research-eval.md` | "xbench-Deep/Searh" | **Corrected**: typo → "xbench-DeepSearch" | — |

## Confirmed, no edit needed

- SWE-RL: Llama3-SWE-RL-70B 41.0% Verified, NeurIPS 2025, `facebookresearch/swe-rl` (https://arxiv.org/abs/2502.18449). Book's hedged binary-reward wording kept (reward-definition caveat retained).
- DeepSWE: Agentica × Together AI, Qwen3-32B, 4,500 R2E-Gym tasks, 6 days/64 H100s, 42.2% Pass@1 (avg 16 runs), 71.0% Pass@16, 59% hybrid TTS, pure 0/1 RL (https://www.together.ai/blog/deepswe). SWE-agent-LM-32B 40.2% confirmed via same blog's table.
- SSR: +10.4 Verified / +7.8 Pro, single model inject+repair, test-patch spec, ICML 2026 (https://arxiv.org/abs/2512.18552).
- SALT: Li et al., trajectory-graph step advantage, EACL 2026 Findings, arXiv 2510.20022 (https://aclanthology.org/2026.findings-eacl.247).
- GiGPO 2505.10978 (NeurIPS 2025, Feng et al.), SPA-RL 2505.20732, HGPO 2602.22817 (historical-context inconsistency), ARPO 2507.19849 (post-tool entropy spike) — all match book text.
- SFR-DeepResearch: Salesforce single-agent, 2509.06283 (https://arxiv.org/abs/2509.06283).
- Salesforce SFR-RL: 250× vs VERL/FSDP+CP, gpt-oss-120b at 1M ctx on 16 H200s, Least-Loaded EP, pipelined synchronous (https://www.salesforce.com/blog/efficient-rl-training-agentic-era/).
- Bespoke Labs: composite-reward instability, gibberish blowup, KL=0 collapse ~300 steps, KL 0.001 + ref-refresh-every-100, GRPO/Qwen2.5-7B/BFCL (https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning).
- Amazon: 72 AppWorld samples, Qwen2.5-32B 39.2%→72% vs Sonnet 3.7/4.0 ~69%, small-model distillation advice (https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning).
- Kimi-Researcher: REINFORCE, strict on-policy, dynamic envs, context manager >50 iterations (+30%), xbench-DeepSearch 69% (https://moonshotai.github.io/Kimi-Researcher/).
- BrowseComp: OpenAI 2025, 1,266 questions (https://openai.com/index/browsecomp); GAIA: 2023, 2311.12983, Meta/HF/AutoGPT, ICLR 2024 — citations already correct.
- rStar2-Agent: 14B, 510 steps, 64 MI300X, 80.6% AIME24, beats 671B R1; GRPO-RoC = Resample-on-Correct (https://arxiv.org/abs/2508.20722).
- SWE-smith: 50k+ instances / 128 repos, SWE-agent-LM-32B 40.2%, Yang et al. 2504.21798 (https://arxiv.org/abs/2504.21798).
- OpenResearcher: 97K+ trajectories, search/open/find, 100+ tool calls, offline (https://arxiv.org/abs/2603.20278).
- Tongyi: 30.5B total / 3.3B active, SOTA list incl. HLE/BrowseComp/WebWalkerQA/FRAMES/xbench-DeepSearch (https://arxiv.org/abs/2510.24701).
- UI-TARS-2: ByteDance Seed, multi-turn RL/RLVR, sandbox/data flywheel (https://arxiv.org/abs/2509.02544).
- InjecAgent: 1,054 cases, ReAct GPT-4 24% base (https://arxiv.org/abs/2403.02691).

## Needs external verification (could not resolve)

- Kimi K2.5 swarm specifics (68.3%/72.1%, 280K/18K, 12M trajectories, 32 agents) and Step 3.7 Flash figures: absent from arXiv:2602.02276; left hedged.
- LinkedIn attention-sink internals beyond the public HF-blog outline; Alibaba WebShaper/AgentFounder/rLLM specifics; SWE-smith API names and per-strategy yield table (book cites real paper 2504.21798; numbers plausible but unverified line-by-line); SWE-smith "NeurIPS 2025 D&B Spotlight" venue tag.
- Tongyi "outperforms o3 / DeepSeek-V3.1 (671B)" naming; xbench-DeepSearch authorship (Tsinghua+H​KU); O-Researcher 72B/2.92K/10K counts; PokeeResearch-7B; DeepResearcher emergent-behavior anecdotes; CaRR/Atom-Searcher/PAIR/IGPO/AEM/RefGRPO/StepPO/Turn-PPO/AT²PO/Group-Graph-PO/survey-2604.09459 result claims; ZeroTIR/Astra/HardGen/TSR+15%/LoopTool/Magnet-14B/Seer/Relax venues; AWM-ICML-2025 and ToolDecathlon-ICLR-2026 venue tags; FinQA/DeepCoder/DeepScaleR vendor-blog figures in rLLM labs; EVA "attention region" mechanism; CogAgent 18B/9B packaging; swesmith storage (~128 GB) figure.
- Amazon/Salesforce/Bespoke/LinkedIn/Kimi numbers were verified against the cited blogs as of this writing; blogs may have been updated since the book cited them — re-check URLs at publish time.

## Sidebar additions

None (no pages added).

## Commands run and results

- `npx prettier --write` on the 8 edited pages: clean (7 unchanged, 1 reformatted).
- `npx vitepress build docs`: success (19.5s), no dead links.
- No code or tests in scope (docs-only checking pass).

## Open issues

- `industrial-practice.md` (19.6) and `industrial-evaluation.md` (supplement) still duplicate scenario content including the now-fixed Gamma-decay block — coordinator should pick one canonical home (pre-existing issue).
- Kimi K2.5 paper is v2 (Aug 2026); if the chapter wants swarm-mode depth, the PARL reward and critical-steps formalism are now citable.
- SWE-RL "Meta" attribution was previously removed as unverifiable; primary evidence (`github.com/facebookresearch/swe-rl`, Meta-affiliated authors) now supports restoring it — left for the coordinator since the chapter deliberately hedges it.
