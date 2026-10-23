---
title: 10.5 Summary and Exercises
---

# 10.5 Summary and Exercises

## What You Should Now Be Able to Do

After working through Sections 10.1–10.3, you should be able to:

1. Explain why offline RL fails naively: name distribution shift and extrapolation error, and quantify how a per-step OOD error $\epsilon$ grows to roughly $\epsilon / (1 - \gamma)$ through the Bellman backup.
2. Compare BCQ, CQL, and IQL precisely: where each places its constraint (action space, value function, or implicit backup), and which of them ever queries Q-values of out-of-dataset actions.
3. Write down the CQL regularizer, the IQL expectile loss, and an advantage-weighted policy loss from memory, and explain what each term does.
4. Explain the TD3+BC normalization trick: why the adaptive weight multiplies the Q-term and how it keeps the improvement and cloning forces balanced across reward scales.
5. Train and prompt a Decision Transformer: compute return-to-go labels, describe the $(\hat{R}, s, a)$ token layout, and choose a deployment target return from the dataset's return range.
6. Evaluate candidate policies without deploying them: compute trajectory-wise and per-decision IS estimates, WIS, and effective sample size by hand on small logs, and describe when to prefer doubly robust or FQE.
7. Map LLM preference optimization onto offline RL: identify the fixed dataset, the behavior/reference policy, and the KL constraint in DPO, and state which way the temperature $\beta$ pushes the policy.

## Key Equations

- Bellman target with the dangerous max: $y = r + \gamma \max_{a'} Q(s', a')$ — offline, the max may select an action the data never tried, whose Q-value is pure extrapolation.
- Error amplification: $\|Q_T - Q^\pi\| \lesssim \epsilon_{\text{ood}} / (1 - \gamma)$ — with $\gamma = 0.99$, each unit of per-step OOD error compounds into about 100 units.
- Constrained offline objective: $\max_\theta\, \mathbb{E}_{s \sim \mathcal{D}}[Q(s, \pi_\theta(s))]$ subject to $D(\pi_\theta \| \pi_\beta) \le \epsilon$ — improve, but stay near the data support ($D$ is any divergence, e.g. KL or MMD; see §10.1).
- CQL regularizer: $\alpha\,(\mathbb{E}_s[\log\sum_a \exp Q(s,a)] - \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q(s,a)])$ — push down Q everywhere (including OOD actions), pull it back up on data actually seen.
- IQL expectile loss: $L_2^\tau(x) = |\tau - \mathbb{1}(x < 0)|\,x^2$ with $x = Q(s,a) - V(s)$ — an asymmetric square that makes $V$ lean toward the best in-data actions without ever maximizing over OOD actions.
- Advantage-weighted policy loss: $-\mathbb{E}_{(s,a) \sim \mathcal{D}}[\exp(\beta A(s,a)) \log \pi_\theta(a \mid s)]$ — clone the data selectively, imitating high-advantage actions exponentially more.
- TD3+BC actor: $\mathcal{L} = -\lambda\,\mathbb{E}[Q(s, \mu(s))] + \mathbb{E}[(\mu(s) - a)^2]$, $\lambda = \alpha / \text{mean}|Q(s, \mu(s))|$ over the batch — scale the Q-term (not the BC term) so both forces stay balanced at any reward scale.
- Return-to-go: $\hat{R}_t = \sum_{t'=t}^{T} r_{t'}$ — the remaining return, used by Decision Transformer as the prompt that steers generation toward good or bad behavior.
- Trajectory IS: $\hat{J}_{\text{IS}} = \frac{1}{N}\sum_i \rho_i G_i$, $\rho_i = \prod_t \pi(a_t\mid s_t)/\mu(a_t\mid s_t)$ — unbiased reweighting of logged returns, with variance that explodes in horizon.
- Per-decision IS: $\hat{J}_{\text{PDIS}} = \frac{1}{N}\sum_i \sum_t \gamma^t \rho_{1:t}^{(i)} r_{i,t}$ — scale each reward only by the action ratios that precede it.
- WIS and trust meter: $\hat{J}_{\text{WIS}} = \sum_i \rho_i G_i / \sum_i \rho_i$, $\text{ESS} = (\sum_i \rho_i)^2 / \sum_i \rho_i^2$ — self-normalize for stability, and distrust any ranking built on a single-digit ESS.
- FQE backup: $Q_{k+1} \leftarrow \arg\min_Q\, \mathbb{E}_{\mathcal{D}}[(Q(s,a) - (r + \gamma\,\mathbb{E}_{a' \sim \pi}[Q_k(s',a')]))^2]$ — evaluate $\pi$'s own actions by regression, with no max over OOD actions.

## Common Pitfalls

1. **Maximizing over OOD actions in an offline target.** Copying a DQN/TD3 target loop verbatim and taking $\max_{a'} Q(s', a')$ (or an unconstrained actor step toward it) replays the exact failure of Section 10.1. Every offline target must constrain, penalize, or avoid the max.
2. **Prompting Decision Transformer with an impossible return.** A target RTG above the best returns in the dataset conditions the model on an event it never saw; outputs turn incoherent. Always pick the target from inside the dataset's return range.
3. **Getting the DPO temperature backwards.** Larger $\beta$ means a _stronger_ KL anchor to the reference model (too conservative, preferences unseparated); smaller $\beta$ means _weaker_ regularization (larger drift, OOD gibberish). Tune $\beta$ while watching KL and held-out quality, not training preference accuracy alone.
4. **Trusting an importance-sampling estimate with a tiny ESS.** Two candidates ranked by IS on an ESS of 2–3 are ranked by noise. Clip, self-normalize, require a healthy ESS, and demand agreement between two estimators that fail differently (e.g. WIS and FQE) before shipping.
5. **Normalizing with the wrong statistics.** Compute state/reward normalization on the training dataset once and freeze it; recomputing statistics on deployment inputs silently shifts every Q-value and breaks the conservatism guarantees you trained for.

## Exercises

### Exercise 1 (easy): Return-to-Go by Hand

A logged episode earns rewards $[0, 0, 5, 1]$ over four steps. Compute $\hat{R}_1, \dots, \hat{R}_4$, and say what target return you would feed a Decision Transformer at deployment if this episode is the best in your dataset.

::: details Solution
$\hat{R}_t$ sums rewards from $t$ to the end: $\hat{R}_4 = 1$, $\hat{R}_3 = 5 + 1 = 6$, $\hat{R}_2 = 0 + 6 = 6$, $\hat{R}_1 = 0 + 6 = 6$. If this is the best episode in the dataset, a deployment target around $6$ (the top of the observed return range, never far above it) is the safe choice.
:::

### Exercise 2 (easy): Amplification Arithmetic

Suppose every Bellman update injects a fresh OOD error of size $\epsilon_{\text{ood}} = 0.2$ and $\gamma = 0.99$. Bound the total accumulated error. How does the bound change if $\gamma = 0.9$?

::: details Solution
The geometric series sums to $\epsilon_{\text{ood}} / (1 - \gamma) = 0.2 / 0.01 = 20$. With $\gamma = 0.9$ the bound is $0.2 / 0.1 = 2$: shorter effective horizons amplify extrapolation far less, which is one reason low-discount tasks are more forgiving offline.
:::

### Exercise 3 (conceptual): Why Online Gets Away With It

Online Q-learning also uses the $\max_{a'} Q(s', a')$ target. Explain in two or three sentences why the identical update is self-correcting online but collapses offline.

::: details Solution
Online, an overestimated OOD action gets selected, executed, and its inflated value is confronted with real rewards and next states, so the error is corrected by fresh data. Offline, the overestimated action is never executed, nothing ever contradicts the inflated value, and the Bellman backup recirculates it with $1/(1-\gamma)$ amplification.
:::

### Exercise 4 (derivation): Expectile at $\tau = 1/2$

Show that the IQL value loss with $\tau = 0.5$ reduces to (half of) ordinary mean squared error, and explain why $V(s)$ then estimates the average in-data $Q(s, a)$ rather than a soft maximum.

::: details Solution
$L_2^{0.5}(x) = |0.5 - \mathbb{1}(x < 0)|\,x^2$. For $x > 0$ the weight is $|0.5 - 0| = 0.5$; for $x < 0$ it is $|0.5 - 1| = 0.5$. So $L_2^{0.5}(x) = 0.5\,x^2$ in both cases — half the MSE, with identical gradients up to the constant $0.5$. Minimizing $\mathbb{E}_{a \sim \mathcal{D}}[(Q(s,a) - V(s))^2]$ over $V(s)$ gives the conditional mean of $Q$ under the data distribution, i.e. average behavior, with no tilt toward good actions. Only $\tau > 0.5$ (overweighting positive residuals $Q > V$) pushes $V$ toward the best in-data actions.
:::

### Exercise 5 (numeric): IS Estimate and ESS by Hand

A logger picks $a_1/a_2$ fifty-fifty; the candidate always picks $a_1$. Three length-2 logged trajectories: $(a_1,a_1)$ with return $10$, $(a_1,a_2)$ with return $2$, $(a_2,a_1)$ with return $6$. Compute each trajectory weight, the IS estimate, and the ESS. What does the ESS tell you?

::: details Solution
$\rho_1 = (1/0.5)(1/0.5) = 4$, $\rho_2 = (1/0.5)(0/0.5) = 0$, $\rho_3 = (0/0.5)(1/0.5) = 0$. $\hat{J}_{\text{IS}} = (4\cdot10 + 0\cdot2 + 0\cdot6)/3 = 40/3 \approx 13.3$. $\text{ESS} = (4+0+0)^2/(16+0+0) = 1$. Three episodes carry effectively one sample: the whole estimate rides on $\tau_1$, so its variance is enormous and no selection decision should rest on it.
:::

### Exercise 6 (numeric): Self-Normalization on the Same Logs

Compute the WIS estimate on the Exercise 5 logs. It happens to be exact here in a special sense — explain why, and state whether WIS is unbiased in general.

::: details Solution
$\hat{J}_{\text{WIS}} = (4\cdot10 + 0 + 0)/(4 + 0 + 0) = 40/4 = 10$. It equals $10$ because, under the same-start-state assumption of this example, after normalization only $\tau_1$ survives, and $\tau_1$'s return ($10$) is exactly the return of the candidate's deterministic action sequence $(a_1, a_1)$. This exactness is luck of the tiny example, not a property of WIS: self-normalized IS is biased in general (ratio of two expectations) but consistent, converging to the truth as $N \to \infty$ with much lower variance than plain IS.
:::

### Exercise 7 (derivation): TD3+BC Is Scale-Invariant

Suppose all rewards are multiplied by a constant $c > 0$. Show that the TD3+BC actor update direction is unchanged (so no retuning of $\alpha$ is needed), and explain why putting $\lambda$ on the BC term instead would break this property.

::: details Solution
Scaling rewards by $c$ scales all Q-values by $c$: $Q \to cQ$. The weight becomes $\lambda' = \alpha/\text{mean}|cQ| = \lambda/c$, so the improvement term is $\lambda' (cQ) = \lambda Q$ — identical to before, while the BC term never involved the reward scale. The balance is preserved exactly. Had $\lambda$ multiplied the BC term instead ($\mathcal{L} = -Q + \lambda\cdot\text{BC}$), scaling would give $-cQ + (\lambda/c)\cdot\text{BC}$: the Q-term grows by $c$ while the cloning term shrinks by $c$, so large reward scales would silently switch off the conservatism — the inverted behavior.
:::

### Exercise 8 (coding): Expectile Regression From Scratch

Implement the expectile loss $L_2^\tau$ in about five lines (NumPy or PyTorch). Generate 200 samples $Q \sim \mathcal{N}(0, 1)$, fit a scalar $V$ by gradient descent for $\tau \in \{0.5, 0.7, 0.9\}$, and check that (a) $\tau = 0.5$ recovers the sample mean, and (b) the fitted $V$ increases with $\tau$.

::: details Solution

```python
import numpy as np

def expectile_loss(q, v, tau):
    diff = q - v
    w = np.where(diff < 0, 1.0 - tau, tau)
    return (w * diff ** 2).mean()

rng = np.random.default_rng(0)
q = rng.normal(0, 1, size=200)
for tau in (0.5, 0.7, 0.9):
    v = 0.0
    for _ in range(2000):  # gradient descent on a scalar
        diff = q - v
        grad = (-2.0 * np.where(diff < 0, 1.0 - tau, tau) * diff).mean()
        v -= 0.1 * grad
    print(tau, round(float(v), 3))
```

Expected behavior: at $\tau = 0.5$ the loop minimizes half-MSE, so $V$ lands on the sample mean ($\approx 0$ up to sampling noise); at $\tau = 0.7$ and $0.9$ the asymmetric weight pulls $V$ upward toward the top of the sample distribution, so the three values come out ordered $V_{0.5} < V_{0.7} < V_{0.9}$. If your run shows this ordering plus $V_{0.5} \approx \bar{Q}$, the implementation is correct.
:::

### Exercise 9 (coding): An OPE Shootout on Synthetic Logs

Simulate a 2-armed bandit (horizon 1, so trajectory IS is exact): arm returns $r \sim \mathcal{N}(1, 1)$ and $r \sim \mathcal{N}(0, 1)$. Log $N = 50$ pulls from a uniform logger, then evaluate the deterministic candidate "always pull arm 1" with plain IS, WIS, and ESS. Repeat for 200 seeds and compare the estimators' variance; then shrink the logger to pull arm 1 only 5% of the time and explain what happens to the ESS.

::: details Solution

```python
import numpy as np

def ope_once(pull_arm1_prob=0.5, n=50, seed=0):
    rng = np.random.default_rng(seed)
    arms = (rng.random(n) < pull_arm1_prob).astype(int)  # 1 = arm 1
    rew = np.where(arms == 1, rng.normal(1, 1, n), rng.normal(0, 1, n))
    mu = np.where(arms == 1, pull_arm1_prob, 1 - pull_arm1_prob)
    rho = np.where(arms == 1, 1.0 / mu, 0.0)  # candidate always pulls arm 1
    is_est = (rho * rew).mean()
    den = rho.sum()
    if den == 0:  # logger never pulled arm 1: no information about the candidate
        return np.nan, np.nan, 0.0
    wis_est = (rho * rew).sum() / den
    ess = den ** 2 / (rho ** 2).sum()
    return is_est, wis_est, ess

est = np.array([ope_once(seed=s) for s in range(200)])
# Drop degenerate seeds with zero overlap before averaging.
est = est[~np.isnan(est[:, 1])]
print("kept seeds:", len(est), "of 200")
print("IS  mean/var:", est[:, 0].mean().round(3), est[:, 0].var().round(3))
print("WIS mean/var:", est[:, 1].mean().round(3), est[:, 1].var().round(3))
print("mean ESS:", est[:, 2].mean().round(1))
```

Expected behavior: both estimators center near the true value $1.0$, WIS shows clearly smaller variance across seeds (normalization damps the lucky/unlucky draws of high-weight pulls), and mean ESS is around $25$ ($N$ times the $0.5$ overlap). With the logger pulling arm 1 only 5% of the time, most weights are $0$ and the rare survivors carry weight $20$: ESS collapses to roughly $N \times 0.05 \approx 2$–$3$, correctly warning that the logs barely overlap the candidate and neither estimate should be trusted for selection. Note that with $N = 50$ and $p = 0.05$, about $0.95^{50} \approx 8\%$ of seeds never pull arm 1 at all ($\sum_i \rho_i = 0$); those runs carry zero information and are dropped before averaging.
:::
