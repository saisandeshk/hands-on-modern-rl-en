---
title: 2.1 Exploration and Exploitation
---

# 2.1 Exploration and Exploitation

## Reading Guide

**Key Ideas**

- Understand how an agent improves its actions from rewards when there are no action labels.
- Build the minimal reinforcement learning loop of "choose an action, observe the reward, update the estimate."
- Use expected return to compare three policies (uniform random, always choose the best, and explore-then-commit), and see the cost of exploration and the misjudgments that small samples cause.
- Understand why exploration and exploitation arise naturally from trial-and-error learning.

Reinforcement learning involves many concepts at once: states, actions, rewards, policies, and values. If these concepts are discussed together from the start, it is hard to see how an agent actually learns to choose actions from experience. This section therefore begins with a simple multi-armed bandit story that keeps only actions and rewards, and then adds the other concepts reinforcement learning needs step by step.

In Chapter 1, we used PPO to train an agent that controls CartPole. In that task the program never tells the agent the correct action at each step. The agent chooses to push left or push right, the environment returns a reward, and the agent adjusts its later choices according to the reward. This is the basic way reinforcement learning learns.

Now let us simplify this process. Suppose the agent faces two slot machines, A and B. Each time it picks one, pulls the lever, and receives a score. Here, which machine to pick is the **action**, and the score received after the pull is the **reward**.

On its first pull of A the agent gets 6 points, and on its first pull of B it gets 4 points. Based on these two results, choosing A next is natural. But the reward each machine gives is not fixed. Getting 6 points once does not mean 6 points every time afterward, and getting 4 points once does not show that the machine must be worse.

The agent therefore has to pull the machines many times and estimate their average rewards from the rewards already obtained. There can also be more than two machines. Let $K$ denote the number of machines available, so the agent chooses repeatedly among $K$ machines. Such a task is called the **Multi-Armed Bandit Problem**.

With estimates of the average rewards, the agent can keep choosing the machine that currently looks best. This is called **exploitation**. But if it always chooses the same machine, no data accumulates for the others, and early misjudgments can never be corrected. The agent therefore also needs to try machines that have been chosen few times and whose average reward is still uncertain. This is called **exploration**.

In this section we learn how to estimate the average reward of an action and how to choose between exploration and exploitation. The next section adds a state that changes with actions and extends this single-step problem into a sequential decision problem.

::: info Key Concept
Exploitation is choosing the action that currently looks best, and exploration is trying actions that are still uncertain. Exploration temporarily spends opportunities to act but produces new information. Exploitation uses existing information to earn reward but may lock the agent onto an action that was misjudged early on.
:::

**Key Formulas**

$$
\mathbb{E}[R_a]=\sum_{r} r\,P(R_a=r)
\quad \text{(Expected reward: the long-run average payoff per pull of action }a\text{)}
$$

> **Expected reward:**
>
> - $R_a$: the random reward received after choosing action $a$.
> - $P(R_a=r)$: the probability that action $a$ outputs reward $r$.
> - $\mathbb{E}[R_a]$: the probability-weighted average of all possible rewards.

$$
\hat{\mu}_a=\frac{1}{N_a}\sum_{i=1}^{N_a}r_i
\quad \text{(Sample mean: estimating the average reward of action }a\text{)}
$$

> **Estimating an action's reward:**
>
> - $N_a$: the number of times action $a$ has been chosen.
> - $r_i$: the reward received the $i$-th time action $a$ was chosen.
> - $\hat{\mu}_a$: the estimate of the average reward computed from the samples so far.

$$
A_t=
\begin{cases}
\arg\max_a\hat{\mu}_a, & \text{with probability }1-\epsilon\\
\text{choose uniformly among all actions}, & \text{with probability }\epsilon
\end{cases}
\quad \text{(epsilon-greedy: keeping exploration alongside exploitation)}
$$

> **The $\epsilon$-greedy policy:**
>
> - $A_t$: the action actually chosen at round $t$.
> - $1-\epsilon$: the probability of choosing the action with the best current estimate.
> - $\epsilon$: the probability of trying another action at random.

## Comparing Three Policies by Expectation

Now let us give the two slot machines concrete numbers: machine A pays out at a rate of 60%, and machine B at 40%. The reward rule is also changed to the easiest version to compute: +1 for a win and −1 for a loss. There are 100 rounds in total.

The two payout rates are parameters inside the environment, and the expectation calculations below use them. The agent cannot see these two numbers in the game and can only estimate them by trying.

### What Is an Expectation

Before comparing policies, we need a tool to answer "how many points does a random choice earn on average?" That tool is the **expectation**.

Start from a familiar example. Flip a fair coin: heads wins 1 dollar and tails loses 1 dollar. In 1000 flips there are about 500 heads and 500 tails, and the total profit or loss is

$$
500 \times (+1) + 500 \times (-1) = 0
$$

Let us look at it again with proportions in place of counts. 500 is exactly $0.5 \times 1000$, and after substituting, the 1000 in the numerator and denominator cancels:

$$
\frac{0.5 \times 1000 \times (+1) + 0.5 \times 1000 \times (-1)}{1000} = 0.5 \times (+1) + 0.5 \times (-1) = 0
$$

This is how an expectation is computed: **multiply the value of each outcome by the probability of its occurrence, and add everything up**. In general form:

$$
\mathbb{E}[X] = \sum_{i=1}^{n} p_i \cdot x_i
$$

Here $x_i$ is the $i$-th outcome and $p_i$ is the probability of that outcome. The meaning of an expectation is the long-run average: a single outcome fluctuates up and down, but with enough repetitions the average approaches the expected value.

Using this method, compute the expectation of each coin inserted into the two machines:

$$
\mathbb{E}[R_A] = 0.6 \times (+1) + 0.4 \times (-1) = +0.2
$$

$$
\mathbb{E}[R_B] = 0.4 \times (+1) + 0.6 \times (-1) = -0.2
$$

Machine A earns 0.2 per play on average, and machine B loses 0.2 per play on average. The policy comparisons below are all built on these two numbers.

### Policy 1: Uniform Random

In each round, choose A or B with probability 50% each. The outcome of a round is produced in two steps: first pick a machine, then see whether it pays out. A round therefore has four possible outcomes:

| Event             | Reward | Probability     |
| ----------------- | ------ | --------------- |
| Choose A and win  | +1     | 0.5 × 0.6 = 0.3 |
| Choose A and lose | −1     | 0.5 × 0.4 = 0.2 |
| Choose B and win  | +1     | 0.5 × 0.4 = 0.2 |
| Choose B and lose | −1     | 0.5 × 0.6 = 0.3 |

By the expectation method, multiply the value of each of the four outcomes by its probability and add them up:

$$
\mathbb{E}[R_{\text{per round}}] = 0.3 \times (+1) + 0.2 \times (-1) + 0.2 \times (+1) + 0.3 \times (-1) = 0
$$

We can also group by machine first: the rounds that picked A earn 0.2 on average, the rounds that picked B lose 0.2 on average, and the two cases each make up half:

$$
\mathbb{E}[R_{\text{per round}}] = 0.5 \times 0.2 + 0.5 \times (-0.2) = 0
$$

The two methods give the same number. The grouping method rests on the law of total expectation, whose derivation is in the supplementary box.

::: details Supplement: The Law of Total Expectation, the Basis for Grouping First and Then Averaging

The law of total expectation splits an overall average into a weighted sum of the averages within each group:

$$
\mathbb{E}[R] = P(\text{choose A})\,\mathbb{E}[R \mid \text{choose A}] + P(\text{choose B})\,\mathbb{E}[R \mid \text{choose B}]
$$

$\mathbb{E}[R \mid \text{choose A}]$ reads "the average reward given that A was chosen," which here is $\mathbb{E}[R_A] = 0.2$. The formula means: overall average = probability of entering a group × the average within that group, summed over the groups.

It is the same thing as term-by-term enumeration. The joint probability used in the enumeration can be split into the product of two step probabilities:

$$
P(\text{choose A and win}) = P(\text{choose A}) \times P(\text{win} \mid \text{choose A}) = 0.5 \times 0.6 = 0.3
$$

The grouped form simply regroups the four terms of the enumeration by machine:

$$
0.5 \times \big(0.6 \times 1 + 0.4 \times (-1)\big) + 0.5 \times \big(0.4 \times 1 + 0.6 \times (-1)\big)
$$

:::

The expectation per round is 0, so how do we compute the total expectation over 100 rounds? This uses a basic property of probability theory:

> **Linearity of expectation**: for any random variables $X_1, X_2, \ldots, X_n$ (independence is not required),
>
> $$
> \mathbb{E}\left[\sum_{i=1}^{n} X_i\right] = \sum_{i=1}^{n} \mathbb{E}[X_i]
> $$
>
> The expectation of a sum equals the sum of the expectations.

The total expectation over 100 rounds is therefore the sum of the per-round expectations:

$$
\mathbb{E}[R_{100}] = \underbrace{0 + 0 + \cdots + 0}_{100} = 0
$$

The return of uniform random play fluctuates around 0, with what is earned on A exactly cancelled by what is lost on B. This policy does not learn at all: no matter how many results it has seen, it always stays at fifty-fifty and never uses any evidence that "A seems better."

### Policy 2: Always Choose A

Suppose we know in advance that A is better and put all 100 rounds on A. The expectation per round is A's expectation, 0.2:

$$
\mathbb{E}[R_{100}] = 100 \times 0.2 = 20
$$

20 is the best result attainable in this environment and can serve as an upper bound for comparing other policies. Its premise cannot be met in real problems: which machine is better must be learned by trying.

### Policy 3: Explore Then Commit

An approach closer to reality has two phases: in the first 20 rounds, try A and B alternately (10 times each) and record each one's win fraction, and in the last 80 rounds put every round on the machine with the higher observed fraction.

The expectation of each of the first 20 rounds is the same as for uniform random, namely 0. Suppose the exploration phase judges correctly and the last 80 rounds commit to A. The total expectation is then

$$
\mathbb{E}[R_{100}] = 0 + 80 \times 0.2 = 16
$$

That is 4 points less than Policy 2. These 4 points are the **cost of exploration**: the chances spent on B in the first 20 rounds produce nothing, and what is bought in exchange is information about the two machines.

Explore-then-commit has so far shown the basic trade-off between exploration and exploitation: the first 20 rounds buy information, and the last 80 rounds use it. But this accounting hides one premise: that the exploration phase can really tell which machine is better. We now test this premise.

### Observed Fractions Deviate from True Fractions

The premise that "10 trials are enough to see the two machines clearly" is not solid. First distinguish two concepts. The **true payout rate** is an intrinsic parameter of the machine: A's is 60%, which means that if we keep playing for a long time, the fraction of wins approaches 60%. The **observed payout rate** is the fraction counted in this particular run of 10 plays, and it fluctuates with luck.

A's true payout rate of 60% does not guarantee exactly 6 wins in every 10 plays. One run of 10 plays may produce only 4 wins, or it may produce 7. Here is one possible exploration record (✅ win, ❌ loss):

| Machine | True payout rate | Results of 10 plays  | Wins | Observed payout rate |
| ------- | ---------------- | -------------------- | ---- | -------------------- |
| A       | 60%              | ❌✅❌❌✅❌✅❌✅❌ | 4    | 40%                  |
| B       | 40%              | ✅❌✅✅❌❌✅❌✅❌ | 5    | 50%                  |

The intrinsic parameters of the two machines have not changed: A still has a 60% chance of paying out each time, and B still has only 40%. It is only that in this sample, A had somewhat bad luck and B had somewhat good luck. If we decide by the observed fractions, the last 80 rounds all go to B, which is actually the worse machine: the truly better option looks worse in a short-term sample.

::: details Supplement: How Many Wins Out of 10 Is Itself Random

Denote the result of the $i$-th play of A by $X_i$ (1 for a win, 0 for a loss), so that $P(X_i = 1) = 0.6$. The total number of wins in 10 plays

$$
N_A = X_1 + X_2 + \cdots + X_{10}
$$

is a random variable. It may equal 6, or 4, or another value. The observed payout rate $\hat{p}_A = N_A / 10$ is only a temporary estimate from this sample, and the true parameter of the machine is still $p_A = 0.6$. The expectation describes the long-run average: $\mathbb{E}[N_A] = 10 \times 0.6 = 6$, that is, when many runs of 10 plays are repeated, the average number of wins per run approaches 6, while any single run can well deviate.

The probability of "exactly $k$ wins" is given by the binomial distribution:

$$
P(N = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

Take A winning exactly 4 times as an example:

$$
P(N_A = 4) = \binom{10}{4} \times 0.6^4 \times 0.4^6 \approx 11.1\%
$$

The meaning of each of the three factors: $\binom{10}{4} = 210$ is the binomial coefficient, the number of ways to choose 4 of the 10 positions as the winning positions, counting only which positions and not their order. $0.6^4$ is the probability of each of those 4 wins. $0.4^6$ is the probability of each of the remaining 6 losses. Every arrangement of positions has the same probability, so the total must be multiplied by the number of arrangements.

In other words, for a machine whose true payout rate is 60%, winning only 4 out of 10 is something encountered about once in every 9 runs, not at all rare.

What probability theory guarantees is that when we try enough times, the observed fraction approaches the true fraction (the law of large numbers). A short sample of 10 trials like this one is outside that guarantee.

:::

What really affects the return is the probability of misjudgment: the probability that, at the end of the exploration phase, B's number of wins exceeds A's. Split into cases by A's number of wins:

| A's wins | Number B must exceed | Misjudged? |
| -------- | -------------------- | ---------- |
| 0        | 1–10                 | Yes        |
| 1        | 2–10                 | Yes        |
| ⋮        | ⋮                    | ⋮          |
| 9        | 10                   | Yes        |
| 10       | Cannot exceed        | No         |

When A and B have the same number of wins, we commit to A, which counts as a correct judgment.

::: details Supplement: How the 12.8% Is Added Up

The misjudgment event is $n_A < n_B$. A's 10 plays and B's 10 plays are independent, so the probability of a combination such as "A wins 4 and B wins 5" is the product of the two probabilities. Different combinations do not overlap, so the total misjudgment probability is the sum of the probabilities of all combinations satisfying $n_A < n_B$:

$$
P(\text{misjudgment}) = \sum_{k=0}^{9} P(n_A = k) \cdot P(n_B \geq k+1) \approx 12.8\%
$$

:::

About 12.8% means that in roughly every 8 complete runs of the 100-round experiment, about 1 will judge the worse machine to be the better one. Once misjudged, all of the last 80 rounds go to B:

$$
\mathbb{E}[R_{100} \mid \text{misjudgment}] = 0 + 80 \times (-0.2) = -16
$$

Weighting the two outcomes by their probabilities gives the full expectation of Policy 3:

$$
\mathbb{E}[R_{100}] = 87.2\% \times 16 + 12.8\% \times (-16) \approx 11.9
$$

This is again about 4 points lower than the 16 that ignores misjudgment. The two layers of loss have different sources. The first layer (20 − 16 = 4) is the opportunities spent on exploration. The second layer (16 − 11.9) comes from exploration yielding too few samples, so the estimate goes wrong. The final performance of the policy is limited by both layers at once.

If the gap between the two machines narrows to 52% versus 48%, then with 10 trials each the probability of misjudgment is about 34% and the probability of a tie is about 17%. In more than half of the experiments we either commit to the wrong machine or cannot tell the two apart. The sample size of exploration directly determines the return: with too few samples the misjudgment probability is high, and with too many samples the cost of exploration eats up the gain. How to allocate opportunities for exploration is exactly the problem the following sections keep addressing.

The true expectation cannot be seen. All we can see are the samples from one trial after another. What policy design must solve is how to use these limited samples to estimate and use action values. Below we first generalize the two machines into the general multi-armed bandit problem.

## The Multi-Armed Bandit Problem

Two machines are the smallest version of this problem. Generalizing the number of machines to $K$ gives the general multi-armed bandit: each arm $a \in \{1,2,\ldots,K\}$ corresponds to an unknown reward distribution $R_a$, and pulling it once returns a random reward. The agent does not know which arm has the highest average reward and can only estimate it through repeated trials.

Formally, at round $t$ the agent chooses an action $A_t$ and then observes the reward:

$$
R_t \sim R_{A_t}
$$

The true expected reward of each arm is denoted:

$$
\mu_a = \mathbb{E}[R_a]
$$

If all the $\mu_a$ were known, the best action would be:

$$
a^* = \arg\max_a \mu_a
$$

The real problem is that the agent does not know these expectations at the start. It has to try first to obtain data, but every trial spends one opportunity to act. If it keeps exploring, the agent wastes a great many opportunities on poor actions. If it exploits too early, it may misjudge from early random outcomes and rule out the action that is truly good.

The multi-armed bandit removes state transitions and long-term returns and keeps only action selection and reward feedback. This simplification lets us first see the earliest difficulty in reinforcement learning: action values are not given in advance but are estimated during interaction.

## From Expected Reward to a Rule for Acting

The agent maintains an estimate $\hat{\mu}_a$ for each action. If action $a$ has been chosen $N_a$ times and has produced rewards $r_1,r_2,\ldots,r_{N_a}$, the most direct way to estimate is the sample mean:

$$
\hat{\mu}_a = \frac{1}{N_a}\sum_{i=1}^{N_a} r_i
$$

This formula answers "how many points does this action currently seem to earn on average?" Once we have the estimates, we also need a rule for acting that turns them into the next choice. The most direct rule is the greedy policy:

$$
A_t = \arg\max_a \hat{\mu}_a
$$

The greedy policy chooses the action with the highest current estimate at every step. It looks reasonable but has an obvious problem: early on the samples are few, and the estimates may be badly affected by chance outcomes. The earlier calculation gave a quantitative example: for two machines at 60% versus 40%, trying each 10 times gives about a 12.8% probability of judging the worse one to be the better one. An arm with a high true expectation that happens to return a low reward on its first pull may go for a long time without a chance to be tried again.

The policy in reinforcement learning therefore cannot only answer "which action has the highest estimate right now?" It must also answer "which actions, although their estimates are not high now, are still worth trying again?" This is the exploration and exploitation problem.

## The $\epsilon$-Greedy Policy

The $\epsilon$-greedy policy keeps a small portion of random exploration alongside the greedy action. At each step, flip a coin with probability $\epsilon$ first:

- With probability $1-\epsilon$, choose the action with the best current estimate.
- With probability $\epsilon$, choose an action at random.

Written as a formula:

$$
A_t =
\begin{cases}
\arg\max_a \hat{\mu}_a, & \text{with probability } 1-\epsilon \\
\text{choose uniformly among all actions}, & \text{with probability } \epsilon
\end{cases}
$$

The meaning of this rule is direct: most of the time trust current experience, and a small part of the time give unknown actions a chance. It keeps the agent from being completely locked in by a few early random outcomes.

Here is a minimal implementation:

```python
import numpy as np


class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q = np.zeros(n_arms)
        self.n = np.zeros(n_arms)

    def select(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q)

    def update(self, arm, reward):
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]
```

`self.q` stores the average-reward estimate of each action, and `self.n` records how many times each action has been tried. The update rule:

$$
\hat{\mu}_a \leftarrow \hat{\mu}_a + \frac{r - \hat{\mu}_a}{N_a}
$$

is the incremental form of the sample mean. It does not need to store the whole reward history. It only corrects the old estimate with the new reward.

The benefit of a fixed $\epsilon$ is stability. However certain the current estimates are, the agent keeps a chance to explore. Its cost is also clear: even when it basically knows which action is best, the agent still chooses at random at a fixed rate, which loses some reward in the long run.

## Letting Exploration Decrease over Time

Early in training the agent has almost no experience and needs more exploration. Late in training the estimates are fairly stable, and continuing a large amount of random trial lowers the payoff. So we can let $\epsilon$ decrease over time:

```python
class EpsilonDecaying:
    def __init__(self, n_arms, epsilon_start=1.0, epsilon_end=0.01, decay=0.995):
        self.n_arms = n_arms
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay = decay
        self.q = np.zeros(n_arms)
        self.n = np.zeros(n_arms)

    def select(self):
        if np.random.random() < self.epsilon:
            arm = np.random.randint(self.n_arms)
        else:
            arm = np.argmax(self.q)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.decay)
        return arm

    def update(self, arm, reward):
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]
```

This schedule expresses a common training rhythm: explore more early, and trust existing experience more later. We will meet this idea again when we study DQN. In early training DQN commonly uses a large ε to collect diverse experience, and then lowers ε step by step so that the policy uses the action values it has learned more.

## More Targeted Exploration

The exploration of $\epsilon$-greedy is crude. Once it enters the exploration branch, it chooses at random among all actions, including those that are already clearly poor. A better exploration strategy keeps asking: which actions are still uncertain and therefore worth trying again?

UCB (Upper Confidence Bound) adds an uncertainty bonus to each action's estimated reward:

$$
A_t = \arg\max_a \left[
\hat{\mu}_a + c\sqrt{\frac{\ln t}{N_a}}
\right]
$$

Here $\hat{\mu}_a$ is the current average-reward estimate and $N_a$ is the number of times action $a$ has been tried. The fewer the trials, the larger the uncertainty bonus. The more the trials, the more the choice depends on the actual estimate.

Thompson sampling takes another angle. For each action it maintains a distribution over "what this action's true average reward might be," draws one sample from each distribution in every round, and chooses the action with the highest sample. An action tried few times has a wider distribution and therefore still has a chance of being chosen. An action tried many times has a narrower distribution, and the choice gradually stabilizes.

::: details What is Thompson sampling?
Thompson sampling is a way to explore **according to uncertainty**. Instead of keeping only one estimated win rate for each machine, it keeps a range of plausible win rates.

Imagine that machine A has won 6 of 10 pulls and machine B has won 1 of 2:

- A's observed win rate is 60%, backed by ten observations.
- B's observed win rate is 50%, backed by only two observations.

A currently looks better, but B is much less certain. A greedy policy always chooses A and may stop learning about B. Uniform random exploration chooses B half the time, even after strong evidence shows that B is worse. Thompson sampling takes a middle path: **choose each machine in proportion to how likely it currently seems to be the best.**

### The three-step idea

For every round:

1. Keep a probability distribution for each machine's unknown true win rate.
2. Draw one possible win rate from each distribution.
3. Pull the machine with the largest draw, then update its distribution using the result.

The draw is not treated as the true win rate. It is one plausible answer to the question, "Given everything I have seen, what might this machine's win rate be?"

Suppose one round produces:

$$
\tilde p_A = 0.58,\qquad \tilde p_B = 0.72
$$

Then Thompson sampling chooses B. On another round it might draw:

$$
\tilde p_A = 0.63,\qquad \tilde p_B = 0.41
$$

and choose A. B is explored because its uncertainty sometimes produces a high draw, not because the algorithm follows a fixed random-exploration schedule.

### Representing uncertainty with a Beta distribution

For a machine whose outcome is win ($1$) or loss ($0$), a convenient model is:

$$
p_a \sim \operatorname{Beta}(\alpha_a,\beta_a)
$$

You can read the parameters as accumulated evidence:

- $\alpha_a = 1 +$ number of observed wins;
- $\beta_a = 1 +$ number of observed losses.

Starting with $\operatorname{Beta}(1,1)$ means every win rate between 0 and 1 is initially plausible. After observing a win, add 1 to $\alpha_a$. After a loss, add 1 to $\beta_a$:

$$
\text{win: }\alpha_a \leftarrow \alpha_a+1,
\qquad
\text{loss: }\beta_a \leftarrow \beta_a+1
$$

For example, six wins and four losses produce:

$$
p_A \sim \operatorname{Beta}(7,5)
$$

Its center is near the observed 60% win rate, but it still has some width because ten observations are limited. After 600 wins and 400 losses, the center remains near 60%, while the distribution becomes much narrower. The estimate has not changed much; our **confidence** has.

### Why exploration happens automatically

An arm tried only a few times has a wide distribution. Its samples sometimes land very high, so it continues to receive occasional trials. An arm tried many times has a narrow distribution. If its results are consistently poor, it rarely produces the largest sample and is selected less often.

This gives Thompson sampling its useful behavior:

- uncertain arms receive opportunities to prove themselves;
- strong arms are exploited frequently;
- clearly weak arms gradually stop consuming many trials.

This behavior is called **probability matching**. If the current evidence says A has roughly an 80% chance of being the best arm, Thompson sampling will choose A roughly 80% of the time. It does not need a separate exploration probability such as $\epsilon$.

### Minimal pseudocode

```python
alpha = [1, 1]
beta = [1, 1]

for _ in range(num_rounds):
    samples = [
        random.betavariate(alpha[a], beta[a])
        for a in range(2)
    ]
    action = max(range(2), key=lambda a: samples[a])
    reward = pull(action)  # 1 for win, 0 for loss

    alpha[action] += reward
    beta[action] += 1 - reward
```

The essential idea is simple: **sample one believable world, act optimally in that world, then revise what worlds remain believable.**
:::

The common goal of these methods is not to add randomness blindly, but to allocate exploration opportunities to the actions that still have informational value.

## From Bandits to MDPs

The multi-armed bandit can be seen as the smallest version of the reinforcement learning process:

- The set of actions already exists.
- Rewards appear after actions.
- The policy decides what to choose next.
- The agent updates its estimate of how good each action is from experience.

It lacks one key part: the state does not change because of actions. In CartPole, a push to the left changes the cart position and the pole angle, and the situation visible at the next step changes with them. After a language model generates one token, the following context changes, and the meaning of the available actions at the next step changes too.

With a state that changes, the problem expands from "choosing actions repeatedly in the same situation" to "making decisions in a sequence of continually changing situations." The next section introduces the Markov decision process and puts the states, actions, transitions, rewards, and discount into one definition of the environment.

## Extension: Multi-Armed Bandits, Regret, and Policy Comparison

> The text above used the multi-armed bandit to explain exploration and exploitation and gave the basic mechanisms of $\epsilon$-greedy, UCB, and Thompson sampling. Below we keep the same problem and go further by adding a finite number of rounds $T$ and regret, comparing how much reward different exploration methods pay to obtain information.

The following content is collapsed by default, and you may skip it on a first reading.

<details class="details custom-block">
<summary>Expand to read: Quantitative Comparison with a Finite Number of Rounds and Regret</summary>

### The Multi-Armed Bandit Problem

The text above was concerned with "how to choose in the next round." To compare algorithms further, we also have to specify how many rounds in total can be played. For example, if we are allowed to act only $T=1000$ times, the opportunities consumed by early exploration directly affect the final cumulative reward.

#### Problem Definition

Using the earlier notation, at round $t$ we choose $A_t$ and observe the reward $R_t$. With a limit on the number of rounds, the objective is written as

$$
\max\;\mathbb{E}\left[\sum_{t=1}^{T}R_t\right].
$$

This objective counts the results of learning and of using together. To judge whether an action is worth continuing to try, the agent must consider two parts of the payoff: the reward this choice may bring, and the information this observation provides for the following $T-t$ rounds of choices.

#### Regret

Since $\mu_a$ is unknown, the agent cannot choose $a^*$ in every round. We use **regret** to measure how good a policy is:

$$\text{Regret}(T) = T \cdot \mu^* - \mathbb{E}\left[\sum_{t=1}^T R_t\right] = \sum_{t=1}^T \mathbb{E}[\mu^* - \mu_{A_t}]$$

Here $\mu^* = \max_a \mu_a$. Regret is the expected gap of "how much more could have been earned if the optimal action had been chosen every round."

The **regret bound** is the fundamental metric for evaluating an algorithm. A good policy should have regret that grows **sublinearly** in $T$, that is, $\text{Regret}(T) = o(T)$. This means that as $T \to \infty$ the agent gets closer and closer to optimal, and the average loss per round tends to 0.

| Growth rate        | Meaning     | Assessment                                    |
| ------------------ | ----------- | --------------------------------------------- |
| $\Theta(T)$        | Linear      | The agent learned nothing, pure random        |
| $\Theta(\sqrt{T})$ | Sublinear   | A standard good policy (UCB, Thompson)        |
| $\Theta(\log T)$   | Logarithmic | The theoretical lower bound (Lai-Robins 1985) |

### $\epsilon$-Greedy and Decay Schedules

The text above already implemented fixed ε and decaying ε. Here we use regret to look at the long-run difference between the two schedules.

#### The $\epsilon$-Greedy Algorithm

Suppose there are only two arms, with true average rewards of $0.8$ and $0.5$, and the agent has already identified the first arm correctly. With a fixed $\epsilon=0.1$, there is a $10\%$ probability of exploring at random in each round, and when exploring there is a one-half probability of choosing the second arm. The expected loss per round caused by random exploration is therefore

$$
0.1\times\frac{1}{2}\times(0.8-0.5)=0.015.
$$

After $1000$ rounds, this part alone produces about $15$ points of regret. A fixed ε keeps guarding against early misjudgment, but even when the estimates have stabilized, the loss keeps accumulating.

#### Decay Schedules

A decay schedule lowers the exploration probability as data increases. For example, starting from $\epsilon_0=1.0$ and multiplying by $0.995$ each round, the value is about $0.61$ at round $100$, about $0.082$ at round $500$, and about $0.0067$ at round $1000$. If a floor of $0.01$ is set, $1\%$ random exploration is kept late in the run.

This schedule expresses a clear assumption: early on the estimation error is large and exploration is more valuable, and later the estimates tend to stabilize and the opportunities to act should be used more for earning reward. If the decay is too fast, the early lock-in of pure greedy reappears, and if it is too slow, it approaches the persistent loss of a fixed ε.

::: details Supplement: Theoretical Properties of $\epsilon$-Greedy

A fixed $\epsilon$-greedy policy keeps random exploration permanently, so even when it already knows which action is best, it still chooses other actions with probability ε. This guards against early misjudgment but produces a persistent loss in the long run, so the cumulative regret of a fixed ε usually grows at a linear rate.

If ε is made to decrease over time, for example decaying from a larger initial value to a very small one, the agent explores more early and exploits more later. A suitable decay strategy can bring the long-run regret down to sublinear, and under some stricter settings a regret bound close to logarithmic can be obtained.

PAC sample complexity poses a different question: it no longer asks how much is lost in total over the first $T$ rounds, but asks "how many trials are needed to find, with probability at least $1-\delta$, an action within $\epsilon$ of the optimum." This viewpoint is mainly used in theoretical analysis, and this course only needs you to know that it measures "the number of samples needed to learn a sufficiently good policy."

:::

### More Targeted Exploration

The text above already showed that UCB and Thompson sampling concentrate exploration on actions that may still be good. Below we expand this difference with concrete numbers and posterior updates.

#### UCB: Adding a Bonus for Uncertainty

The selection formula of UCB (Upper Confidence Bound) is still:

$$A_t = \arg\max_a \left[ \hat{\mu}_a + c \sqrt{\frac{\ln t}{N_a}} \right]$$

Suppose we are at round $100$ and take $c=1$. Arm 1 has an estimate of $0.65$ and has been tried $50$ times, while arm 2 has an estimate of $0.55$ and has been tried only $5$ times. Their UCB scores are approximately

$$
0.65+\sqrt{\frac{\ln100}{50}}\approx0.95,
\qquad
0.55+\sqrt{\frac{\ln100}{5}}\approx1.51.
$$

The current average reward of arm 2 is lower, but its samples are too few and its uncertainty bonus is larger, so it still gets a chance to be tried in this round.

::: details Supplement: The Regret Bound of UCB

It can be proved that the regret of UCB1 reaches the logarithmic level. A common form is:

$$\mathbb{E}[\text{Regret}(T)] \leq 8 \sum_{a: \mu_a < \mu^*} \frac{\ln T}{\Delta_a} + \left(1 + \frac{\pi^2}{3}\right) \sum_{a: \mu_a < \mu^*} \Delta_a$$

Here $\Delta_a = \mu^* - \mu_a$ is the gap between a suboptimal arm and the optimal arm. On a first pass you do not need to remember this expression, only the mechanism it expresses: UCB gradually reduces its trials of clearly suboptimal actions and concentrates exploration on actions that are truly hard to tell apart.

:::

#### Thompson Sampling: Choosing by the Likelihood of Being Optimal

Thompson sampling does not compute a confidence upper bound. It samples directly from the true payoffs each action might currently have. Below we look at the concrete update under 0/1 rewards.

In the 0/1 reward setting, a Beta distribution can represent the posterior of each arm:

$$\mu_a \sim \text{Beta}(\alpha_a, \beta_a)$$

Pull arm $a$ and receive reward 1, and set $\alpha_a \leftarrow \alpha_a + 1$. Receive reward 0, and set $\beta_a \leftarrow \beta_a + 1$. This update rule is very lightweight, so Thompson sampling is often used in recommendation, advertising, and A/B testing systems.

::: details Supplement: A Theoretical View of Thompson Sampling

The theoretical analysis of Thompson sampling often uses Bayesian regret: first assume the problem itself comes from a prior distribution, and then compute the policy's expected loss under this prior. In common stochastic bandit settings, it can achieve logarithmic regret of the same order as UCB.

This detail is not the main line of the section. The main line only needs you to understand that Thompson sampling turns "an action may be optimal" into a selection probability, and therefore wastes fewer exploration opportunities than $\epsilon$-greedy.

:::

#### Contextual Bandits and RLHF

An ordinary multi-armed bandit assumes that the reward distribution of each arm is fixed. In real systems, however, whether an action is good often depends on the current input.

In a recommender system, the same article attracts different users differently. In an advertising system, the same ad has different click rates for different audiences. For a large language model, the quality of the same answer must also be judged under a specific question.

This gives the **Contextual Bandit**:

- At each round, first observe the context $x_t$.
- Then choose the action $A_t$.
- Receive the reward $R_t$.
- The goal is to learn a policy $\pi(a \mid x)$ so that the action depends on the current context.

From this angle, RLHF can first be understood as a contextual bandit problem: the prompt is the context, the answer the model generates is the action, and the score from the reward model is the reward. This abstraction does not yet include the full token-level state transitions, but it already explains a key fact: the model cannot only learn "which answer is more common overall," and must learn "which kind of answer is more suitable under this prompt."

After we enter MDPs, we will develop this problem further: when an answer is generated token by token, the action is no longer a one-time choice, and the current token changes the later state and the available actions. At that point the bandit problem expands into a genuine sequential decision problem.

### Extension Summary

The multi-armed bandit (MAB) is the most simplified form of RL: stateless with immediate rewards, yet fully retaining the tension of "exploration versus exploitation." $\epsilon$-greedy keeps opportunities to try with a fixed probability and is the starting point for understanding exploration. UCB allocates opportunities to try to the actions with higher uncertainty. Thompson sampling uses posterior sampling to turn "may be optimal" into a selection probability.

The multi-armed bandit has no state transitions, and rewards appear immediately. The next section, [Markov Decision Processes](./mdp), first adds state transitions and enters sequential decision-making with ever-changing situations. Section 2.3 then adds policies, trajectories, and long-term returns.

</details>

### Further Reading

- Sutton & Barto, _Reinforcement Learning: An Introduction_, Chapter 2
- [Auer et al. 2002 "Finite-time Analysis of the Multiarmed Bandit Problem"](https://link.springer.com/article/10.1023/A:1013689704352)
- [Russo et al. 2018 "A Tutorial on Thompson Sampling"](https://arxiv.org/abs/1707.02038)
- [Lattimore & Szepesvári, _Bandit Algorithms_](https://banditalgs.com/)
