# 17.5 Inference-Time Search

The formal verifier in the previous section can give a deterministic right-or-wrong verdict for every derivation step, and a PRM can score intermediate steps as well. With a step-level evaluator, generation no longer has to run from beginning to end in one pass. At every step we can judge whether the current direction is worth continuing, and back up and start again if it has gone wrong.

Let's use the quadratic equation $x^2 + 5x + 6 = 0$ to illustrate. Asked to solve it, the model may take three different routes. The first uses factoring and smoothly writes $(x+2)(x+3)=0$. The second uses the quadratic formula, but while computing the discriminant it evaluates $25-24$ as $-1$ and has already walked into a dead end. The third tries completing the square and is still stuck at the completion stage after writing for a long time.

With Best-of-N, we sample $N$ complete paths independently and pick the best at the end, and two things are wasted. The second route is already wrong at the discriminant step, yet we still wait for it to finish the whole answer, spending thousands of tokens. And although the first and second routes use different methods, the understanding "this is a quadratic equation and we need its roots" is identical in both, yet it is generated twice.

This section deals with exactly that problem: organize generation as a tree, compute a correct prefix only once, turn back early when a path goes wrong, and concentrate compute on the most promising directions. We start from the simplest algorithm, **Beam Search**, move to **Tree of Thoughts**, which supports backtracking, and then to **MCTS**, which balances exploration and exploitation using visit counts. We follow the same problem through each algorithm, and finally discuss when the extra search compute pays off.

<img src="../../chapter20_prm_search/images/inference-search-tree.svg" alt="Inference-time search reuses shared prefixes and scores branches to expand, retain, or stop">

## Why We Need to Reuse Intermediate Reasoning

Let's expand the quadratic example a little. The model can generate several reasoning paths for this problem:

```text
Path A: use the quadratic formula
  → x = (-5 ± √(25-24)) / 2 = (-5 ± 1) / 2
  → x = -2 or x = -3

Path B: use factoring
  → x² + 5x + 6 = (x+2)(x+3) = 0
  → x = -2 or x = -3

Path C: try completing the square
  → x² + 5x = -6
  → x² + 5x + 25/4 = 25/4 - 6 = 1/4
  → (x + 5/2)² = 1/4
  → x + 5/2 = ±1/2
  → x = -2 or x = -3
```

All three paths reach the correct answer. But if the model errs on some path, say path A computes $25-24$ as $-1$ in the discriminant, then the result of that single sample is wrong.

Best-of-N handles this by generating $N$ independent paths and letting a PRM pick the best one. This approach wastes work in two clear ways:

- **It ignores similarity between paths**: when two paths share the same first half (for example, both begin by judging "this is a quadratic equation and can be factored"), Best-of-N generates that shared prefix twice.
- **It cannot back up midway**: when a path realizes halfway through that it has gone wrong (the discriminant was miscomputed), Best-of-N can only write it to the end and cannot go back to recompute.

In essence, $N$ independent samples are brute-force enumeration, and the allocation of compute is crude. Inference-time search turns this duplicated work into an explicit tree of states:

- **Shared prefixes**: an identical prefix of reasoning is computed once and not regenerated.
- **Intermediate evaluation**: after each step is written, the PRM scores it and decides whether to continue or back up.
- **Resource allocation**: search compute is concentrated on the most promising directions.

The three algorithms that follow differ mainly in two questions: how many nodes are kept in each round, and whether a path that has been set aside has any chance of coming back later.

::: details The search process when humans solve math problems
When a person solves a quadratic equation, they are already doing something similar:

- On seeing the problem they first ask "can this be factored?", and try the first route.
- They notice that $6=2\times3$ and $2+3=5$, so it factors, and they continue down this route.
- Reaching "set each factor equal to zero," they are close to the answer and invest more attention.
- If they tried completing the square first and found it too cumbersome after two steps, they back up and switch to factoring.
- If two routes both look feasible, they walk each a few steps to see which goes more smoothly.

A person does not wait for completing the square to be fully worked out before noticing it is cumbersome, and does not reread the problem three times. A search algorithm mechanizes exactly these three things: shared prefixes, intermediate evaluation, and resource allocation.
:::

## How Beam Search and ToT Expand the Reasoning Tree

**Beam Search** uses the most direct rule: at any moment, maintain only the $K$ highest-scoring partial reasonings. In each round, expand these nodes, rescore them with the PRM, and keep the new top $K$.

### The Beam Search Algorithm

```python
def beam_search_thoughts(prompt, model, prm, K=4, expansions=2, max_steps=10):
    # Initial beam: a single empty state
    beams = [{"thought": "", "score": 1.0}]

    for step in range(max_steps):
        # Expand each beam: let the model generate the next reasoning step
        candidates = []
        for beam in beams:
            for _ in range(expansions):
                next_thought = model.generate_next(prompt, beam["thought"])
                score = prm.score(prompt, beam["thought"] + next_thought)
                candidates.append({
                    "thought": beam["thought"] + next_thought,
                    "score": score
                })

        # Take the top K as the new beams
        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:K]

        # Stop if a complete answer has been found
        if any(is_complete(b["thought"]) for b in beams):
            break

    return beams[0]["thought"]  # return the best beam
```

Let's run two rounds on the quadratic equation with $K=2$ and watch how the state changes. Suppose the three candidates in the first round are "factoring 0.9, quadratic formula 0.85, completing the square 0.4," and that in the second round each is expanded by one step and rescored:

- **Round 1**
  - Candidates (score): factoring $0.9$, quadratic formula $0.85$, completing the square $0.4$
  - Kept ($K=2$): factoring, quadratic formula
- **Round 2**
  - Candidates (score): factoring → set factors to zero $0.92$, quadratic formula → discriminant error $0.3$, quadratic formula → computed correctly $0.88$
  - Kept ($K=2$): factoring → set factors to zero, quadratic formula → computed correctly

Completing the square is eliminated in the first round because its score is too low. The erroneous expansion on the quadratic formula path is eliminated in the second round, while the correct one stays. Several attempts from the same prefix share a single prefix computation. This is the direct payoff of "reusing intermediate results."

### Properties and Use Cases of Beam Search

The fixed width makes Beam Search easy to implement, and the $K$ paths can be expanded in parallel, so it is fast. The cost comes from the same fixed $K$: on an easy problem it may maintain redundant paths ($K=4$ when a single path would have worked), and on a hard problem it may delete too early a branch whose value only shows up later. A node that has been eliminated never re-enters the beam. If the early PRM scores are inaccurate and wrongly kill a path that would later have worked, that path never returns.

So Beam Search is a suitable starting point when step boundaries are clear, single-step scores are reliable, and only a few candidates need to be kept at each level. If errors do not surface until much later, so that current scores can hardly justify early pruning, the risk of a fixed beam grows.

::: details Beam Search has been used in machine translation for decades
Beam Search is not an invention of the large-model era. It was the standard algorithm in traditional statistical machine translation and early neural machine translation. The reason is that the choices at each expansion step in translation are relatively controllable: the vocabulary is large, but few next words are truly reasonable, and backtracking is usually unnecessary. Long-chain mathematical reasoning is different. Choosing the wrong method in the first step may not be discovered until the end, and a fixed beam is no longer enough.
:::

### How ToT Keeps Branches and Chances to Backtrack

[Tree of Thoughts](https://arxiv.org/abs/2305.10601) (Yao et al. 2023) is a natural extension of Beam Search: it supports **branching, backtracking, and mixed DFS/BFS**.

The core structure of ToT:

```text
                Root
              /      \
            A1        A2
           /  \      /  \
         B1   B2   B3   B4
        / \    |    |   / \
       C1  C2  C3   C4 C5  C6

       Search algorithm: BFS (breadth-first) or DFS (depth-first)
       Evaluation: score each step with the PRM
       Backtracking: low-scoring nodes are pruned, but we can return to an
                     ancestor node to try a new branch
```

Implementation:

```python
def tree_of_thoughts(prompt, model, prm, max_depth=10, breadth=4):
    # Run DFS from the root
    def dfs(thought, depth):
        if depth >= max_depth:
            return [{"thought": thought, "score": prm.score(prompt, thought)}]

        # Generate N candidate next steps
        candidates = []
        for _ in range(breadth):
            next_thought = model.generate_next(prompt, thought)
            full_thought = thought + next_thought
            score = prm.score(prompt, full_thought)
            candidates.append({"thought": full_thought, "score": score})

        # Sort by score and prune the low scorers
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:breadth // 2]  # prune half

        # Recurse on the retained candidates
        results = []
        for c in candidates:
            results.extend(dfs(c["thought"], depth + 1))

        return results

    return dfs("", 0)
```

ToT lets the system first cut the reasoning into relatively coarse thoughts and then expand them with BFS, DFS, or a bounded beam. It can return to an intermediate node that has not been deleted and try a new continuation, but it is not inherently cheaper than Best-of-N; the gain depends on whether shared prefixes and intermediate scoring are really useful. If every level keeps $B$ branches in full, the number of nodes grows exponentially with depth. With $B=4$ and depth $D=5$ there are $4^5=1024$ nodes. An implementation must set a width, depth, or total node budget, or the compute is quickly exhausted.

### ToT Experimental Results

On the [24 Game](https://arxiv.org/abs/2305.10601) task:

- **Greedy decoding:** 7.3%
- **CoT prompting:** 4.0%
- **Self-consistency (multiple samples + voting):** 9.0%
- **Tree of Thoughts: 74.0%**

In this task and prompt setting, GPT-4 with ToT raised the success rate from single digits to 74%. The improvement on the 24 Game is so large because its intermediate states are especially easy to judge: the numbers used so far and the remaining target can both be checked mechanically, which suits search very well. This margin cannot be extrapolated directly to open-ended tasks without clear states and verification rules (such as essay writing or open-ended question answering).

## How MCTS Uses Verification Feedback to Choose Paths

Beam Search looks only at the current score at every level, and a path that was underestimated early never comes back once eliminated. MCTS (Monte Carlo Tree Search) adds the dimension of visit counts: it makes full use of nodes that currently look high-scoring, while keeping a chance for nodes that have not been explored enough.

MCTS allocates its budget by visiting the tree repeatedly. In LLM reasoning, the model can propose the next step, and rollout results, a PRM, or an external checker are then used to update node values:

- evaluate nodes with an outcome reward, a PRM, or an external verifier
- use the model as the policy (recommending the next step)
- balance exploration and exploitation with the UCB formula

### The Four Steps of MCTS

Each iteration performs these four steps, and the loop repeats until the budget is used up:

1. **Selection**: starting from the root, use the UCB formula to choose the best child node, all the way down to a leaf node.
2. **Expansion**: generate $N$ child nodes at the leaf node.
3. **Simulation**: run a rollout from the child node (quickly generate the full reasoning to the end).
4. **Backpropagation**: send the reward obtained from the rollout back to all ancestor nodes on the path, updating their average scores and visit counts.

The four steps as pseudocode:

```python
def mcts(root, budget, c):
    for _ in range(budget):
        # 1. Selection: from the root, pick the child with the largest UCB at each level, down to a leaf
        node, path = root, [root]
        while node.children:
            node = max(node.children, key=lambda n: ucb(n, node, c))
            path.append(node)

        # 2. Expansion: use the model to generate several candidate next steps at the leaf
        for step in model.generate_steps(node.state, n=EXPAND_WIDTH):
            node.children.append(Node(state=step, parent=node))
        child = node.children[0]

        # 3. Simulation: roll out quickly from the new node to the end and get a reward from the verifier
        reward = verifier.score(rollout(child))

        # 4. Backpropagation: update each ancestor's visit count and average score along the path
        for n in path + [child]:
            n.visits += 1
            n.q += (reward - n.q) / n.visits   # incremental average

    return best_answer(root)   # pick the final answer by visit count or average score
```

Here `ucb(n, parent, c)` is the formula below. The incremental average in the backpropagation step corresponds to this update rule:

$$
Q(n) \leftarrow Q(n) + \frac{r - Q(n)}{N(n)},
$$

where $r$ is the reward of this rollout and $N(n)$ is the visit count after the update. It does not need to store historical rewards. It corrects the old estimate using only the new result, and it is the same form as the update rule in the $\epsilon$-greedy implementation of Section 2.1. One more implementation detail: when choosing the final answer, we usually look at visit counts and not at $Q$ values, because visit counts reflect a combined judgment that has already been corrected by the exploration term.

### The UCB Formula

When selecting a node, two things must be balanced: a node with a high current average deserves continued investment (exploitation), and a node with few visits should also get a chance to be tried (exploration). The UCB (Upper Confidence Bound) formula simply adds the two terms:

$$\text{UCB}(n) = Q(n) + c \cdot \sqrt{\frac{\ln N(p)}{N(n)}}$$

where:

- $Q(n)$: the average reward of node $n$, coming from a PRM, rollout results, or an external verifier.
- $N(n)$: the number of times node $n$ has been visited. The fewer the visits, the less we know about it and the more it needs to be tried.
- $N(p)$: the number of times the parent node has been visited. The more the parent has been visited, the more thorough the overall exploration, and the more meaningful the comparison among its children.
- $c$: the exploration constant, which controls the weight of the second term. The larger $c$ is, the more willing we are to try new paths; the smaller $c$ is, the more conservative.

The first term $Q(n)$ is the average value observed so far. When it is high, the path currently looks workable. The second term grows as the parent's visit count $N(p)$ grows and shrinks as the node's own visit count $N(n)$ grows, so it prioritizes exploring the children with fewer visits. The selection rule is to expand the child with the largest UCB at each step. In implementations, unexplored children are usually visited first (when $N(n)=0$ the UCB is treated as infinite), or a very small smoothing term is added to the denominator to avoid division by zero.

Let's feel this with a set of concrete numbers. Let the parent be visited $N(p)=100$ times ($\ln 100 \approx 4.61$) and take $c=\sqrt{2} \approx 1.414$ (the theoretically recommended value of classic UCB1):

- **A**
  - $Q(n)$: $0.8$
  - $N(n)$: $50$
  - Exploration term $c\sqrt{\ln N(p)/N(n)}$: $1.414 \times \sqrt{4.61/50} \approx 0.43$
  - UCB: $1.23$
- **B**
  - $Q(n)$: $0.7$
  - $N(n)$: $30$
  - Exploration term $c\sqrt{\ln N(p)/N(n)}$: $1.414 \times \sqrt{4.61/30} \approx 0.55$
  - UCB: $1.25$
- **C**
  - $Q(n)$: $0.6$
  - $N(n)$: $5$
  - Exploration term $c\sqrt{\ln N(p)/N(n)}$: $1.414 \times \sqrt{4.61/5} \approx 1.36$
  - UCB: $1.96$

Node A currently has the highest average (0.8), but node C, which has the fewest visits, gets an exploration term of 1.36 and so has the highest UCB, and the next round expands C first. After C has been visited a few dozen times, if its average has not improved (still around 0.6), its exploration term is used up and the budget naturally returns to the truly high-scoring nodes A and B. This is how UCB works: it gives up on no path that looks promising, but it also never keeps pouring compute into a path that yields no return.

Here $Q$ can come from the final outcome (whether the answer is right after a full rollout), from the PRM's step scores, or from a combination of the two, depending on what kind of verifier is available.

### Properties and Representative Work on MCTS

The visit-count mechanism lets MCTS give more budget to high-value branches while still probing branches with few visits, so it does not abandon a path for good because of an early misjudgment. Note, however, that the asymptotic convergence properties of classic MCTS depend on assumptions such as a finite action space, sufficient exploration, and reliable returns. In open-ended text generation, the action candidates are produced by the model with truncation (not by enumerating every possible next token), and the evaluator also makes mistakes, so the theoretical guarantees of classic MCTS cannot be treated as guarantees that the answer is correct. Its main costs are repeated rollouts, state caching, and value updates, and it is considerably more complex to engineer than Beam Search.

Representative work:

- **rStar** ([arXiv:2408.06195](https://arxiv.org/abs/2408.06195)): MCTS plus self-play, designed specifically for mathematical reasoning.
- **AlphaProof** ([DeepMind 2024](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)): AlphaZero-style reinforcement learning combined with proof search and a Lean verifier, which won an IMO silver medal.
- **RAP** ([Reasoning via Planning](https://arxiv.org/abs/2305.14992)): MCTS with an LLM as a world model, applied to more general reasoning tasks.

### AlphaCodium's Search for Code Generation

[AlphaCodium](https://arxiv.org/abs/2401.08500) (January 2024) is a special case. It organizes code generation as an iterative workflow of "understand the problem, generate tests, write a first draft, execute, repair." It is not quite tree search, but the idea comes from the same line:

- Code tasks naturally have **unit tests** that can be checked automatically. New tests generated by the model itself can only add coverage and cannot guarantee complete verification.
- It uses iterative search: generate → test → read the error message → repair → test again, instead of branching at every step as in tree search.

The workflow is:

```text
1. Problem understanding: have the LLM extract key information and generate test cases
2. Initial solution: generate one candidate solution
3. Iterative repair:
   a. Run the test cases
   b. If they fail, analyze the error message
   c. Have the LLM repair the code according to the error message
   d. Repeat until all tests pass
4. Output the final solution
```

Existing tests can serve directly as an outcome verifier. Of course, if test coverage is insufficient, a wrong implementation may still slip through. The iterative structure does not need to maintain a complex tree, so it is simple and efficient. The paper reports gains over direct generation on benchmarks such as CodeContests, and the exact margin varies with the model and the evaluation setting.

The division of labor with tree search can be seen this way: tree search allocates compute on "which way to go next," while AlphaCodium allocates compute on "where this version is wrong." The former needs step-level scoring, and the latter needs only executable test results.

::: details How AlphaGo's MCTS differs from MCTS for LLM reasoning
AlphaGo also used MCTS to play Go. The differences are:

- **Action space**: in Go each move has at most 361 choices (finite), while for an LLM each step is the entire vocabulary (tens of thousands to hundreds of thousands of tokens).
- **State transitions**: placing a stone in Go is deterministic, while LLM generation is probabilistic.
- **Final return**: in Go the win or loss is fully determined, while the correctness of an LLM answer may have gray areas.
- **Simulation cost**: a quick rollout in Go is fast, while generating a whole reasoning chain with an LLM is expensive.

So Go can run millions of MCTS iterations, while LLM reasoning can usually run only tens to hundreds. This directly leads to many differences in the algorithmic details.
:::

## When Search Is Worth the Compute

Search does not raise accuracy for free. Every node expansion calls the generative model, and many methods also call a PRM, execute tests, or run a proof checker. Whether to adopt search depends on how reliable the intermediate feedback is and how costly a failure is.

First, a rough estimate of the cost of the different methods, in terms of generation branches or expansions:

- **Greedy decoding:** 1 complete generation
- **Best-of-N:** $N$ complete generations and $N$ outcome scorings
- **Beam Search ($K,D$):** about $K\times D$ groups of node expansions with per-level scoring
- **Tree of Thoughts ($B,D$):** $O(B^D)$ if fully expanded, in practice controlled by pruning and node caps
- **MCTS:** number of iterations, expansions per iteration, rollout length, and scoring cost

The $O(B^D)$ of Tree of Thoughts is exponential growth: keeping $B=4$ branches per level at depth $D=6$ gives $4^6=4096$ nodes, and with a few hundred tokens generated per node the cost quickly reaches millions of tokens. A real system must prune or set a node budget. MCTS does not expand the entire tree, and its compute is determined mainly by the number of iterations and the number of expansions per iteration, so it is relatively controllable. Both carry more overhead than independent sampling, in the form of state maintenance and step-by-step scoring, so whether to use search depends on whether the intermediate feedback can offset these extra costs.

Scientific computing, formal proving, and competitive programming usually have executable checkers (numerical verification, Lean verification, unit tests), so every path the search expands gets fairly reliable feedback: right is right and wrong is wrong. In that case extra compute converts more easily into a higher success rate. Without a reliable verifier, the PRM itself often misjudges, and search may keep expanding along wrong scores, drifting further the more it searches.

### Training-Time Search vs. Inference-Time Search

There is another important engineering choice: does search happen during training or during inference?

**Using search results at training time** (for example, the reinforcement learning loop of AlphaProof):

- Use the correct trajectories found by search as training data.
- Let the model directly raise the generation probability of high-value steps, internalizing the experience gained from search.
- At deployment, search can still be continued according to task difficulty if needed.

**Search at inference time** (for example, the ToT and MCTS in this section):

- After the model is trained, use search at inference time to improve performance.
- The search budget can be changed without retraining: search less on easy problems and more on hard ones.
- The cost is that every inference pays the compute cost of search.

The two can also be combined: light search during training helps the model converge faster, and at inference time the task's difficulty and importance decide whether to search and how deep. This is consistent with the idea in [Chapter 16: Test-time Compute Scaling](../chapter19_reasoning/test-time-scaling), where the question of where to spend compute and how much is an engineering trade-off.

## Section Summary

A PRM supplies process rewards during training and scores partial paths at inference time. Search algorithms use these scores to decide which intermediate steps to keep, expand, or abandon:

- **Motivation for reuse**: independent sampling regenerates identical prefixes and cannot abandon a wrong path midway. An explicit state tree computes a prefix once and stops errors early.
- **How the three structures differ**: Beam Search keeps a fixed top $K$ and eliminates permanently, which is simple and efficient but vulnerable to early misjudgment. ToT supports pruning and backtracking, with the number of nodes growing exponentially as $B^D$, so a budget must be set. MCTS uses the UCB formula and visit counts to leave a chance for rarely visited nodes.
- **Meaning of the two UCB terms**: $Q(n)$ exploits known high scores, and $c\sqrt{\ln N(p)/N(n)}$ compensates for insufficient exploration. When average scores are equal, the node with fewer visits is chosen first thanks to the exploration term.
- **Judging the cost**: from 1 generation to $O(B^D)$ nodes, the budget differs by several orders of magnitude. Whether search pays off depends on how reliable the intermediate feedback is.

When each of the four methods fits:

- **Beam Search**: simple and parallel, suited to tasks of medium difficulty with reliable step scoring.
- **Tree of Thoughts**: supports backtracking and pruning, suited to tasks with clear states that can be checked mechanically (such as the 24 Game).
- **MCTS**: allocates budget dynamically between exploration and exploitation by visit counts, suited to complex tasks that need finer trade-offs.
- **AlphaCodium**: specific to code tasks, using unit tests as the verifier and repairing iteratively instead of searching a tree.

A real system must choose among Best-of-N, bounded search, and direct generation according to the value of the task, the reliability of the verifier, and the compute budget.

[17.6 Parallel Reasoning and Summary Aggregation](./parallel-reasoning-and-summary) discusses another way to allocate compute: instead of searching deep along a single tree, generate several complete reasoning chains in parallel, and then have the model or verifier exchange information and aggregate the results.
