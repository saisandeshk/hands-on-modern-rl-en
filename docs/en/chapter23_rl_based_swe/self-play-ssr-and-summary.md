# 20.3 Self-Play SWE-RL

Suppose we have only an open-source repository whose dependencies can be installed and whose tests can run, with no human-written issue and no gold patch. The model first edits the code to make a newly added test fail. It then looks for a fix using only the test patch and the current repository. Once the fix works, this round of "injection and repair" becomes a new training experience.

This is the problem studied by the [Self-play SWE-RL (SSR) paper](https://arxiv.org/abs/2512.18552): how can a software agent create practice problems of gradually increasing difficulty in a real repository by itself, without relying on human-annotated issues and tests? The paper uses **a single model** to alternately inject and repair defects, and the defects are specified formally by a test patch.

<img src="../../chapter23_rl_based_swe/images/ssr-self-play-loop.svg" alt="The same model alternates between defect injection, task validation, and defect repair in a sandboxed repository">

## Self-Posed Problems and Self-Repair

The basic idea of self-play is to let the current policy produce both the challenge and the experience of meeting it. SSR places this idea inside an executable software repository, and the procedure can first be simplified into two roles. Player A and Player B are role names chosen to make the explanation easier; in the paper's implementation they can still share one model:

```text
┌──────────────────────────────────────────────────────────┐
│ Player A (Bug Generator):                                │
│   - Inject a bug somewhere in the repository             │
│   - Generate the corresponding test (verifying the bug)  │
│   - Generate the corresponding issue description         │
├──────────────────────────────────────────────────────────┤
│ Player B (Bug Fixer):                                    │
│   - See the issue description                            │
│   - Try to fix it                                        │
│   - Run the tests to verify                              │
├──────────────────────────────────────────────────────────┤
│ RL Update:                                               │
│   - Player A learns to "generate harder bugs"            │
│     (ones Player B cannot fix)                           │
│   - Player B learns to "fix more complex bugs"           │
│   - This forms adversarial improvement                   │
└──────────────────────────────────────────────────────────┘
```

### Limitations of Offline Synthesis

[Section 20.1 SWE-smith](./swe-bench-and-rlvr) is **offline synthetic data**: 50K samples are generated in one pass, and training follows.

SSR keeps producing new defects during training. Whether the difficulty rises with capability depends on whether the filtering and the reward work well. "Online generation" alone does not guarantee a quality improvement.

The two can be compared along four dimensions. In data generation, SWE-smith finishes in one pass, while SSR keeps producing data during training. In data difficulty, SWE-smith is fixed, while SSR adjusts with model capability. In data quality, SWE-smith is independent of the generator's capability, while SSR improves as the model's capability improves. In applicable stage, SWE-smith suits early training, while SSR covers the entire training process.

### From a Single Loop to a Data Flywheel

In the ideal case, improved repair ability forces the injection stage to propose harder valid defects, and the new experience in turn trains repair ability. This feedback loop is usually called a data flywheel.

```text
Strong model → generates hard bugs + excellent fixes → high-quality training data → stronger model → ...
```

The flywheel can also stall. The model may generate the same kind of simple defect over and over, or manufacture "hard problems" out of syntax errors. The later sections therefore also check the validity of defects, the difficulty balance between the two roles, and transfer to real issues.

## An Executable Algorithm

The paper was proposed by researchers at Meta FAIR and other institutions. The code below keeps the two-role presentation of the original draft to explain the inputs, outputs, and order of verification. It is a teaching simplification and is not the source code of the paper's repository.

### The Defect-Injection Role

The defect-injection role reads a runnable repository and outputs "code with the injected defect + a test patch." The test patch is the formal task specification: it should pass on the original code and fail once the defect is injected. A natural-language issue can be generated as a teaching extension, but it is not a required input for SSR training.

```python
def generate_bug(generator_model, repo, file_path):
    # 1. Select a file
    original_code = repo.read(file_path)

    # 2. Let the generator inject a bug
    prompt = f"""
    Here is the code in {file_path}:
    {original_code}

    Please:
    1. Choose a function to modify
    2. Inject a subtle bug (logic error, not syntax error)
    3. Generate a test that would fail with the bug
    4. Optionally summarize the failed behavior without revealing the fix
    """

    response = generator_model.generate(prompt)
    bug_code, test_patch, summary = parse_response(response)

    # 3. Verify the bug is valid (the test fails on the bug code and passes on the original code)
    if not validate_bug(original_code, bug_code, test_patch):
        return None  # Invalid bug, discard

    return {
        "original_code": original_code,
        "bug_code": bug_code,
        "test_patch": test_patch,
        "summary": summary,
    }
```

### The Defect-Repair Role

The defect-repair role sees the repository after injection together with the test patch, and outputs a repair patch. Only at evaluation time do we need to verify whether this ability transfers to natural-language issues.

```python
def fix_bug(fixer_model, task):
    # 1. Show the fixer the failing test and the injected code (not the original code)
    prompt = f"""
    Failing test patch: {task['test_patch']}

    Current code: {task['bug_code']}

    Please fix the bug.
    """

    # 2. The fixer repairs in an agentic way
    trajectory = []
    while not done:
        action = fixer_model.act(prompt)
        trajectory.append(action)

        if action.type == "edit":
            apply_edit(action)
        elif action.type == "test":
            result = run_tests()
            if result.all_passed:
                done = True

    # 3. Compute the reward
    reward = 1.0 if tests_passed else 0.0

    return trajectory, reward
```

### Teaching Pseudocode for Alternating Updates

```python
def ssr_training(generator_model, fixer_model, repo):
    for epoch in range(N_EPOCHS):
        # 1. The generator produces a bug
        task = generate_bug(generator_model, repo, random_file())

        # 2. The fixer tries to repair it
        trajectory, reward = fix_bug(fixer_model, task)

        # 3. A zero-sum reward for teaching purposes
        generator_reward = -reward  # Fixer fails to fix → Generator wins
        fixer_reward = reward       # Fixer succeeds → Fixer wins

        # 4. Update both models
        update_generator(generator_model, task, generator_reward)
        update_fixer(fixer_model, trajectory, fixer_reward)
```

### Gradual Increase in Difficulty

The SSR training objective pushes defect complexity up step by step. To keep this process stable, the system still has to filter out invalid defects and prevent the injection role from earning high scores merely by producing syntax errors.

```text
Epoch 0-100:    Generator produces simple typos / one-line bugs
Epoch 100-500:  Generator produces multi-file, cross-function bugs
Epoch 500-2000: Generator produces subtle logic errors, cross-module effects
```

The epoch ranges above are an illustration to aid understanding. They are not a fixed curriculum published in the paper. Real training still has to tune the sampling difficulty according to the defect validity rate and the repair success rate.

## Verifying Self-Improvement

The SSR paper reports relative gains: a self-improvement of **10.4 percentage points** on SWE-bench Verified and **7.8 percentage points** on SWE-Bench Pro, and it also stays ahead of baselines that use human data throughout training. The evaluation inputs are natural-language issues, while self-play training uses no such issues, so this experiment mainly checks transfer and not memorization of self-generated problems.

The original draft once placed Meta SWE-RL, DeepSWE, SSR, and "SSR + DeepSWE" in a single absolute-score leaderboard. The paper does not provide this comparison of 41.0%, 50.0%, 47.5%, and 53.2% under a unified setting, so these numbers should not be treated as an SSR experiment table. To keep this set of research questions, the measurements should be redone with the same base model, scaffold, budget, and Pass@k protocol.

### Reporting Data Efficiency

The original draft's "50K offline samples," "5K seeds + 50K self-play," and "5K seeds + 100K self-play" can be kept as an experimental design, but the corresponding 41%, 47%, and 53% are not conclusions published in the SSR paper. A real data-efficiency comparison must also report the fraction of valid defects, the number of environment executions per experience, total tokens, training GPU hours, and the final task gain.

## Flywheel Failure

### The Generator May Produce Invalid Bugs

If the Generator learns to produce "syntax-error" bugs (which the Fixer finds hard to fix), this is in fact ineffective training, because syntax errors are rare in real SWE tasks.

The mitigation should start with executable rules: the test must pass on the original code, fail after injection, and pass again after repair. An LLM judge can supplement the check on whether a defect resembles a real maintenance task, but it cannot replace execution-based verification.

### Imbalance Between Generator and Fixer

If the Generator is far stronger than the Fixer, the Fixer can never fix the problems and training has no signal. If the Fixer is far stronger than the Generator, the Generator cannot produce effective challenges and the curriculum stagnates.

The mitigation is to adjust the training frequency of the two dynamically to keep them balanced.

### Domain Drift

The bugs produced by self-play may differ from the real bug distribution. For example, the Generator may concentrate on one kind of bug (typos), while real-world bugs are diverse.

Mitigation: use real PRs as seeds and let the Generator mutate real bug patterns.

## Research Methods and Commercial Products

Cursor, Devin, Trae, Codex, and Claude Code all offer multi-step software engineering capabilities, but most of these products have not published a complete training recipe. A product that reads files, edits, and runs tests does not by itself prove that it uses the method of a particular SWE-RL paper. Below we keep the product observations of the original draft while separating speculation from public fact.

### Cursor

[Cursor](https://cursor.sh) is one of the most popular AI code editors. Its core capabilities:

- **Multi-file understanding**: uses RAG to let the model see the whole project.
- **Agentic repair**: the model can read, edit, and test autonomously.
- **Based on Claude Opus + tool calling.**

Cursor has not published a complete training method, so its training data cannot be inferred from product behavior.

### Cognition Devin

[Devin](https://devin.ai) is the "AI software engineer" released by Cognition. It can complete a full development task on its own (planning, writing code, testing, and deployment).

The full training details of Devin are not public. When citing its training method, link to Cognition's original technical material and note the version. A product demo cannot stand in as evidence about training.

### ByteDance Trae

[Trae](https://www.trae.ai) is an AI programming product. DeepSWE comes from Agentica and Together AI, so Trae cannot be described as "based on DeepSWE research results."

### OpenAI Codex (2025+)

OpenAI Codex provides agent capabilities for handling code tasks in isolated environments. The specific models and evaluations change with product versions, so the text does not tie it to any particular model name:

- Uses o3's reasoning ability for complex planning.
- Integrates with ChatGPT and can process several tasks in parallel.
- A public evaluation should record the model, the date, the tool permissions, and the sampling budget.

### Anthropic Claude Code

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) is Anthropic's code agent tool. The available underlying models and the context configuration are updated over time, so we keep a description of the stable product capabilities:

- Reasoning model + agentic tools.
- Supports reading context in a project, editing files, and calling development tools.

## Multi-Language and Multi-Repository Extensions

Currently SWE-RL focuses mainly on Python. Directions for future expansion:

### Multi-Language

- **JavaScript/TypeScript**: The Jest and Mocha test frameworks are mature and can be handled much as in Python.
- **Java**: JUnit testing is mature, but the code style is strict and needs a stronger KL constraint.
- **C/C++**: Compiled languages with slow test runs, which create a greater need for a world model.
- **Go/Rust**: Modern languages with generally high test coverage, well suited to SWE-RL.

### Multi-Repository

- **Enterprise internal code**: Each company has its own code style, dependencies, and testing conventions.
- **Microservice architecture**: Cross-repository changes and API compatibility.
- **Legacy systems**: Old code, missing tests, and incomplete documentation.

Multi-repository expansion requires:

- **Fast environment setup**: Dependency management for each repository.
- **Domain-specific rewards**: The standard of "good code" differs across repositories.
- **Cross-repository reasoning**: Understanding the dependencies between repositories.

## Division of Roles

Complex SWE tasks may need several agents to cooperate:

```text
Planner Agent: Analyze the issue and draft a repair plan
  ↓
Explorer Agent: Locate the relevant files in the repository
  ↓
Editor Agent: Carry out the changes
  ↓
Tester Agent: Run the tests and report back
  ↓
Reviewer Agent: Check code quality
```

This division of roles can be implemented at the workflow layer, and joint training of multiple policies is a further research direction. Seeing several subtasks in a product interface does not tell us whether there are several independent agents behind it or a single model orchestrating tasks. Training such a system requires:

- **Multi-agent RL**: Joint training of multiple policies.
- **Communication protocol**: How agents pass information to each other.
- **Shared value model**: Evaluating the quality of the overall trajectory.

This is a concrete application in the SWE domain of the [multi-agent part of Chapter 19 on Agentic RL](../chapter22_agentic/build-agentic-training-system).

## Summary

This chapter followed the line "task, reward, environment, data expansion" through RL-based SWE. Section 20.1 introduced SWE-bench and the RLVR paradigm and explained why SWE is an ideal battlefield for RLVR. The supplementary reading on SWE-RL showed learning from the developer problem-solving process in open software-evolution records. Section 20.2 had CWM learn software behavior from environment trajectories, and had DeepSWE scale pure RL in real environments. Section 20.3 had SSR alternately inject and repair defects with a single model, which reduces the dependence on human-written training tasks.

There are five core takeaways. Tests give software tasks automatic feedback, but test coverage determines whether the reward is faithful to the real requirement. SWE-RL, CWM, and DeepSWE use different training objects and recipes, and no single "simple reward" slogan can summarize them. Explicit world models, value models, and search are extensions worth verifying, and they cannot be written up as established components that the papers did not publish. Self-play can generate experience without human issues and tests, but it must filter out invalid defects and check transfer to real issues. Commercial code agents show multi-step engineering ability, and whether they use a particular RL recipe needs to be supported by public training material.

In the following chapters, [Chapter 17 PRM and Search](../chapter20_prm_search/outcome-vs-process) corresponds to the step-level reward in SWE-RL. [Chapter 25 Reward Hacking](../chapter30_alignment_failures/classical-failures) corresponds to hacking in SWE tasks (such as deleting tests to raise the reward). [Section 19.10 Agentic RL Training Systems](../chapter22_agentic/build-agentic-training-system) corresponds to the engineering implementation of SWE-RL.
