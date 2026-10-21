---
title: 24.R Summary and Exercises
---

# 24.R Summary and Exercises

This chapter moved RL from text and game simulators into audio, bodies, and pixels. The settings differ, but the same questions return: what is the state and action, where does the reward come from, and what happens when the reward leaves something out.

## What you should now be able to do

- Separate an audio objective into content correctness, prosody naturalness, and real-time behavior, and explain why optimizing only correctness creates the verifiable-reward trap.
- Describe how a neural audio codec turns a waveform into discrete tokens (frame rate, codebook layers, residual quantization), and estimate the token cost of one second of speech.
- Trace one audio RL update end to end: sample a group of responses, score text and synthesized speech, normalize within the group, and apply a clipped policy update.
- Explain why embodied control is a partially observable, continuous-action, sim-to-real problem, and write down the Gaussian policy, PD control law, domain-randomization objective, and asymmetric actor-critic setup.
- Distinguish model-free from model-based RL by where the learning signal comes from, and describe the Dyna, MPC, and latent-imagination uses of a world model plus the role of uncertainty.
- Write the diffusion process as an RL trajectory (state, action, terminal reward), derive the DDPO/REINFORCE gradient, and explain the baseline and the PPO-style clipped loss.
- Compare group-relative advantages (DanceGRPO) with single-reward training, and explain why multiple rewards must be normalized separately before they are combined.
- Evaluate a generated video on four layers (single-frame quality, text alignment, temporal consistency, physical causality) and explain why fifty denoising passes with eighty frames each is fifty forward passes, not four thousand.

## Key equations

- Residual quantization: $c_k=\arg\min_c\|e^{(k-1)}-\mathrm{CB}_k[c]\|$, $e^{(k)}=e^{(k-1)}-\mathrm{CB}_k[c_k]$. Reading: each codebook encodes only the residual left by the previous one.
- Bradley–Terry preference loss: $\mathcal{L}_{\mathrm{RM}}=-\log\sigma(R_\phi(y_w)-R_\phi(y_l))$. Reading: push the preferred response above the rejected one.
- Audio total reward (teaching template): $R_{\mathrm{total}}=w_cR_{\mathrm{content}}+w_pR_{\mathrm{prosody}}+w_lR_{\mathrm{latency}}$. Reading: the weights state which layer the product cares about.
- Group-relative advantage: $\hat A_i=(r_i-\mathrm{mean}(r)) / (\mathrm{std}(r)+\varepsilon)$. Reading: score each sample against its own prompt group, not against other prompts.
- Continuous Gaussian policy: $\pi_\theta(a|s)=\mathcal{N}(\mu_\theta(s),\Sigma)$. Reading: the network outputs a mean action per joint; the covariance sets exploration.
- Low-level PD control: $\tau=K_p(q_{\mathrm{target}}-q_{\mathrm{current}})-K_d\dot q$. Reading: torque is proportional to position error minus damping on velocity.
- Domain randomization: $J(\theta)=\mathbb{E}_{\mu\sim p(\mu)}[\mathbb{E}_{\tau\sim\mathcal{P}_\mu}[R(\tau)]]$. Reading: maximize average return over randomized physics, not one simulator.
- World model: $\hat p_\phi(s_{t+1},r_t,d_t\mid s_t,a_t)$. Reading: given state and action, predict what the world does next.
- Probabilistic dynamics (PETS): $p_{\phi_i}(\Delta s_t,r_t\mid s_t,a_t)=\mathcal{N}(\mu_{\phi_i},\Sigma_{\phi_i})$. Reading: predict a mean change plus honest uncertainty.
- DDPO objective and gradient: $J(\theta)=\mathbb{E}_{\tau\sim p_\theta}[R(\tau,c)]$, $\nabla_\theta J=\mathbb{E}[R(\tau,c)\sum_t\nabla_\theta\log p_\theta(x_{t-1}\mid x_t,t,c)]$. Reading: one terminal image score updates every denoising step.
- PPO-style clip: $L_{\mathrm{clip}}=-\mathbb{E}[\min(\rho_t\hat A,\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A)]$. Reading: encourage better actions without letting one update move too far.
- Video reward split (teaching template): $R=\lambda_qR_{\mathrm{quality}}+\lambda_aR_{\mathrm{alignment}}+\lambda_tR_{\mathrm{temporal}}+\lambda_pR_{\mathrm{physics}}$. Reading: keep the four failure layers as separate signals.
- Causal order check: $t_{\mathrm{approach}}<t_{\mathrm{contact}}<t_{\mathrm{fall}}$. Reading: the fall must come after contact, however good each frame looks.

## Common pitfalls

- Treating fifty denoising steps over eighty frames as four thousand forward passes. There are roughly $50\times80$ frame-level latent updates, but the DiT processes all frames jointly, so the cost is about fifty forward passes.
- Optimizing a single easy reward. Correctness-only audio RL flattens prosody; HPS-only video training drifts toward static images; aesthetic-only image training breaks counts and positions.
- Adding raw rewards with different scales. The larger-numbered reward dominates unless each reward is normalized within its group first.
- Freezing nothing or everything. A common embodied pattern is to freeze the audio or vision encoder while RL changes the policy; updating perception without need wastes data, while never adapting the policy leaves actuator errors uncorrected.
- Trusting long model rollouts. One-step error $\epsilon$ compounds along the horizon; PETS replans every step, MBPO branches short rollouts from real states, and Dreamer imagines in compressed latent space for the same reason.
- Reporting only the training reward. A rising proxy with no held-out prompts, no per-item scores, and no blind human check is how reward hacking hides.

## Exercises

### Exercise 1: Name the three audio layers

A voice agent answers factual questions correctly, but users complain that every reply sounds flat and the agent often talks over them. Name the three reward layers from Section 24.1, state which layer is failing in each complaint, and propose one weight change to $R_{\mathrm{total}}=w_cR_{\mathrm{content}}+w_pR_{\mathrm{prosody}}+w_lR_{\mathrm{latency}}$ plus one architectural fix for the interruption problem.

::: details Solution
The three layers are content correctness, prosody naturalness, and real-time behavior (first-packet latency and interruptibility). Flat delivery is a prosody failure; talking over the user is a real-time behavior failure; factual accuracy (content) is fine. A minimal change is to lower $w_c$ slightly and raise $w_p$ and $w_l$, for example from $(0.8,0.1,0.1)$ toward $(0.5,0.3,0.2)$, then re-tune on interaction data. Weights alone do not add interruption handling, so add full-duplex turn-taking: stream input while generating, stop synthesis on new speech, and train the take-the-floor decision with trajectory or preference feedback rather than a single correctness label.
:::

### Exercise 2: Count audio tokens by hand

A codec runs at 75 Hz with eight RVQ layers. How many tokens does one second contain, and how many does ten seconds contain? A 5-second video at 16 fps has 80 frames; how many frames does 5 seconds at 24 fps have? Explain in one sentence why speech length grows with duration while text length grows with word count.

::: details Solution
Per second: $75\times8=600$ tokens. Ten seconds: $600\times10=6{,}000$ tokens. Video: $5\times16=80$ frames and $5\times24=120$ frames. Speech tokens scale with time and frame rate because every time slice is encoded, while text tokens scale with the number of words regardless of how slowly they are spoken.
:::

### Exercise 3: Compute group advantages by hand

For one prompt, a group of three samples gets rewards $r=(1,2,3)$. Compute the mean, the population standard deviation, and the DanceGRPO advantages $\hat A_i=(r_i-\mathrm{mean})/\mathrm{std}$. Which sample gets a positive update, and why does the standard deviation matter?

::: details Solution
Mean: $(1+2+3)/3=2$. Population variance: $((1-2)^2+(2-2)^2+(3-2)^2)/3=2/3$. Standard deviation: $\sqrt{2/3}\approx0.8165$. Advantages: $(1-2)/0.8165\approx-1.225$, $(2-2)/0.8165=0$, $(3-2)/0.8165\approx+1.225$. Only the third sample gets a positive update; the first is pushed down and the mean sample gets zero. Dividing by the within-group standard deviation puts easy and hard prompts on a comparable scale, so a hard prompt with small raw gaps still gives a usable learning signal.
:::

### Exercise 4: Why the baseline does not bias the gradient

Starting from $\nabla_\theta J=\mathbb{E}_{\tau\sim p_\theta}[R(\tau,c)\nabla_\theta\log p_\theta(\tau\mid c)]$ and $\hat A=R(\tau,c)-b(c)$ with $b(c)$ independent of the sampled trajectory, show that $\mathbb{E}_\tau[b(c)\nabla_\theta\log p_\theta(\tau\mid c)]=0$.

::: details Solution
Pull out the trajectory-independent baseline: $\mathbb{E}_\tau[b(c)\nabla_\theta\log p_\theta]=b(c)\,\mathbb{E}_\tau[\nabla_\theta\log p_\theta]$. Use the log-derivative identity $\nabla_\theta\log p_\theta=(\nabla_\theta p_\theta)/p_\theta$, so $\mathbb{E}_\tau[\nabla_\theta\log p_\theta]=\int p_\theta(\tau\mid c)\nabla_\theta\log p_\theta(\tau\mid c)\,d\tau=\int\nabla_\theta p_\theta(\tau\mid c)\,d\tau$. Swap gradient and integral: $\nabla_\theta\int p_\theta(\tau\mid c)\,d\tau=\nabla_\theta 1=0$. Hence the baseline term vanishes and $\mathbb{E}[\hat A\nabla\log p_\theta]=\mathbb{E}[R\nabla\log p_\theta]$.
:::

### Exercise 5: Read a video reward table

Two candidate videos have $(\lambda_q=\lambda_a=\lambda_t=\lambda_p=0.25)$ scores A $=(0.9,0.8,0.3,0.35)$ and B $=(0.6,0.55,0.6,0.65)$. Recompute both weighted totals. The terminal reward prefers B by $0.01$. Explain why the per-item signals recommend different fixes for A and B, and correct the "four thousand forward passes" claim for a 50-step, 80-frame run.

::: details Solution
A: $(0.9+0.8+0.3+0.35)/4=2.35/4=0.5875\approx0.59$. B: $(0.6+0.55+0.6+0.65)/4=2.40/4=0.60$. A fails on temporal consistency and physics while quality and alignment are strong, so the fix targets late-generation decisions (motion and contact order). B is mediocre everywhere, so it needs broad improvement. On compute: $50\times80=4{,}000$ is the count of frame-level latent updates, but each denoising step processes all frames in one forward pass, so the run costs about 50 forward passes. A $12.3\times$ speedup claim is consistent with $1429.5/116.5\approx12.27$.
:::

### Exercise 6: Compute a PD torque and an asymmetric update

A quadruped joint has $K_p=100$, $K_d=10$, $q_{\mathrm{target}}=0.5$, $q_{\mathrm{current}}=0.3$, $\dot q=0.4$. Compute $\tau=K_p(q_{\mathrm{target}}-q_{\mathrm{current}})-K_d\dot q$. Then state, in one sentence each, what the actor and the critic observe in asymmetric actor-critic, and which network is kept at deployment.

::: details Solution
Position error: $0.5-0.3=0.2$. Stiffness term: $100\times0.2=20$. Damping term: $10\times0.4=4$. Torque: $20-4=16$ (in the simulator torque units). In asymmetric actor-critic the actor sees only noisy onboard observations (IMU, joint encoders, depth), while the critic also sees privileged simulator state (true velocity, friction, disturbances). At deployment only the actor is kept; the critic is discarded.
:::

### Exercise 7: Code a minimal audio reward harness (coding)

Write a short Python function `total(content, prosody, latency, weights)` returning $w_cR_{\mathrm{content}}+w_pR_{\mathrm{prosody}}+w_lR_{\mathrm{latency}}$, plus a loop over a list of evaluation cases that records each sub-reward separately instead of only the total.

::: details Solution

```python
def total(content, prosody, latency, weights=(0.5, 0.3, 0.2)):
    wc, wp, wl = weights
    return wc * content + wp * prosody + wl * latency

report = []
for case in evaluation_cases:
    rc = content_model(case)
    rp = prosody_model(case)
    rl = latency_model(case)
    report.append({
        "id": case.id,
        "content": rc,
        "prosody": rp,
        "latency": rl,
        "total": total(rc, rp, rl),
    })
```

Each case stores the three sub-rewards and the weighted total. Plotting the three columns separately shows whether a rising total comes from better answers, more natural prosody, faster first packets, or a mix; a single total would hide a correctness-only regression like prosodic flattening.
:::

### Exercise 8: Code random-shooting MPC on the point-mass world (coding)

Using the 1-D point-mass teaching environment $v_{t+1}=0.95v_t+0.15\tanh(a_t)$, $x_{t+1}=x_t+v_{t+1}$, $r_t=-(x_{t+1}^2+0.1v_{t+1}^2+0.001a_t^2)$ with $a_t\in[-1,1]$, implement: (a) a learned one-step model interface `pred = model(state, action)` returning $[\widehat{\Delta x},\widehat{\Delta v},\hat r]$; (b) scoring of $M$ random action sequences over horizon $H$ inside the model; (c) execution of only the first action of the best sequence in the real environment, then replanning. Explain why the loop replans instead of executing the whole sequence.

::: details Solution

```python
# model(state, action) -> [dx, dv, r_hat]; imagined_state starts at real state
def score_sequences(model, state, sequences, gamma=0.97):
    n, h, _ = sequences.shape
    imagined = state.unsqueeze(0).expand(n, -1).clone()
    returns = torch.zeros(n)
    discount = 1.0
    for t in range(h):
        pred = model(imagined, sequences[:, t])
        dx_dv, r_hat = pred[:, :2], pred[:, 2]
        imagined = imagined + dx_dv
        returns = returns + discount * r_hat
        discount *= gamma
    return returns

def mpc_step(model, state, horizon=20, samples=512):
    seqs = torch.empty(samples, horizon, 1).uniform_(-1.0, 1.0)
    scores = score_sequences(model, state, seqs)
    return seqs[scores.argmax(), 0]  # first action of best sequence

# main loop: act, observe, replan
state = env_reset()
for _ in range(steps):
    action = mpc_step(model, state)
    state, _ = env_step(state, action)  # real environment, then replan
```

Only the first action is executed because model error compounds: $\epsilon_{t+k}\approx\mathcal{O}(k\epsilon_{\mathrm{model}})$, so a state predicted $H$ steps ahead is less trustworthy than the current real observation. Replanning every step closes the loop with fresh sensing, which is the same reason PETS, MBPO short rollouts, and receding-horizon MPC outperform open-loop execution.
:::
