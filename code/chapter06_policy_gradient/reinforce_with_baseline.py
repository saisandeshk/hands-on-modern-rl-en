"""
Chapter 6: REINFORCE with Baseline -- variance reduction comparison experiment
Compares vanilla REINFORCE against a version with a baseline (Value Network) added

Core problem: vanilla REINFORCE has high gradient variance, making training unstable
Solution: replace the raw return with an advantage function
    advantage = G_t - V(s_t)
    where V(s_t) is a value network's estimate of the state

Why does a baseline reduce variance?
    - The absolute value of G_t can be large (e.g. 200), but the differences across time steps are small
    - After subtracting V(s), the advantage fluctuates around 0 with much smaller magnitude
    - Mathematically, E[G_t - b] = E[G_t] (as long as b doesn't depend on the action), so the expectation is unchanged while variance drops

How to run:
    python reinforce_with_baseline.py
    python reinforce_with_baseline.py --smoke   # fast CPU smoke test writing under runs/
"""

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Create the output directory
os.makedirs("output", exist_ok=True)
SEED = 0

# Configure Chinese fonts
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: Network architecture definitions
# ==========================================
class PolicyNetwork(nn.Module):
    """
    Policy network (Actor): state -> action probabilities

    Architecture: 4 -> 128 -> 128 -> 2 (Softmax output)
    """

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        logits = self.network(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


class ValueNetwork(nn.Module):
    """
    Value network (Baseline/Critic): state -> value estimate

    Architecture: 4 -> 128 -> 128 -> 1 (scalar output)
    Used to estimate V(s), the expected cumulative return starting from state s

    This network is the "baseline": by subtracting V(s), we get the advantage function A(s)
    """

    def __init__(self, state_dim=4, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output a single scalar value
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)  # Drop the last dimension, yielding [batch_size]


# ==========================================
# Part 2: Compute discounted cumulative returns
# ==========================================
def compute_returns(rewards, gamma=0.99):
    """
    Compute the discounted cumulative return G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Args:
        rewards: list of immediate rewards
        gamma: discount factor
    Returns:
        returns: list of discounted cumulative returns
    """
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns


# ==========================================
# Part 3: Collect an episode trajectory
# ==========================================
def collect_episode(policy, env, device="cpu"):
    """
    Use the current policy to collect data for one full episode

    Args:
        policy: policy network
        env: environment
        device: torch device for tensors
    Returns:
        states, actions, rewards, episode_reward
    """
    state, _ = env.reset()
    states, actions, rewards = [], [], []
    done, truncated = False, False

    while not (done or truncated):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            probs = policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        next_state, reward, done, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    episode_reward = sum(rewards)
    return states, actions, rewards, episode_reward


# ==========================================
# Part 4: Vanilla REINFORCE training
# ==========================================
def train_vanilla_reinforce(num_episodes=500, gamma=0.99, lr=1e-3, seed=0, device="cpu"):
    """
    Vanilla REINFORCE (no baseline)

    Loss = -sum log pi(a_t|s_t) * G_t
    Uses the discounted cumulative return G_t directly as the weight
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    policy = PolicyNetwork(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []
    gradient_estimates = []  # Record gradient estimates for measuring variance

    for episode in range(num_episodes):
        # Collect a trajectory
        states, actions, rewards, episode_reward = collect_episode(policy, env, device)

        # Compute returns
        returns = compute_returns(rewards, gamma)

        # Convert to tensors
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        # Forward pass
        probs = policy(states_t)
        action_probs = probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(action_probs + 1e-8)

        # Policy gradient loss
        loss = -(log_probs * returns_t).mean()

        # Record the gradient estimate (used for subsequent variance calculation)
        with torch.no_grad():
            grad_estimate = (log_probs * returns_t).mean().item()
            gradient_estimates.append(grad_estimate)

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  [Vanilla] Episode {episode+1:4d} | Last-50 avg: {avg:6.1f}")

    env.close()
    return episode_rewards, gradient_estimates


# ==========================================
# Part 5: REINFORCE with Baseline training
# ==========================================
def train_reinforce_with_baseline(num_episodes=500, gamma=0.99, lr=1e-3, seed=0, device="cpu"):
    """
    REINFORCE + value baseline

    Advantage function: A(s,a) = G_t - V(s_t)
    Policy loss: -sum log pi(a_t|s_t) * A(s_t, a_t)
    Value loss: MSE(V(s_t), G_t)

    Two networks are trained simultaneously:
        - The policy network learns "which actions are better" (relative to the baseline)
        - The value network learns "how many points this state is worth on average" (the baseline)
    """
    baseline_seed = seed + 100
    random.seed(baseline_seed)
    np.random.seed(baseline_seed)
    torch.manual_seed(baseline_seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=baseline_seed)
    env.action_space.seed(baseline_seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize the policy network and value network
    policy = PolicyNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
    value_net = ValueNetwork(state_dim=state_dim).to(device)

    # Each network gets its own optimizer
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)

    episode_rewards = []
    gradient_estimates = []  # Record gradient estimates

    for episode in range(num_episodes):
        # Collect a trajectory
        states, actions, rewards, episode_reward = collect_episode(policy, env, device)

        # Compute returns
        returns = compute_returns(rewards, gamma)

        # Convert to tensors
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        # ========== Update the value network (Critic) ==========
        # The value network's goal: accurately predict V(s) ~= G_t
        values = value_net(states_t)
        value_loss = nn.MSELoss()(values, returns_t)

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # ========== Compute the advantage function ==========
        # Advantage = actual return - baseline prediction
        # A > 0 means "better than expected" -> increase that action's probability
        # A < 0 means "worse than expected" -> decrease that action's probability
        with torch.no_grad():
            values_pred = value_net(states_t)
        advantages = returns_t - values_pred

        # ========== Update the policy network (Actor) ==========
        probs = policy(states_t)
        action_probs = probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(action_probs + 1e-8)

        # Policy gradient loss: use the advantage function in place of the raw return
        policy_loss = -(log_probs * advantages).mean()

        # Record the gradient estimate (using the advantage in place of the return)
        with torch.no_grad():
            grad_estimate = (log_probs * advantages).mean().item()
            gradient_estimates.append(grad_estimate)

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  [Value Baseline] Episode {episode+1:4d} | Last-50 avg: {avg:6.1f}")

    env.close()
    return episode_rewards, gradient_estimates


# ==========================================
# Part 6: Comparison experiment main function
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="REINFORCE vs baseline comparison (Chapter 6)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--num-episodes", type=int, default=500, help="Training episodes per method")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test mode: tiny CPU run writing only under runs/",
    )
    return parser.parse_args()


def run_comparison(args=None):
    """
    Run the comparison experiment: Vanilla REINFORCE vs REINFORCE + Value Baseline

    Compares along two dimensions:
        1. Learning speed and final performance (reward curve)
        2. Variance of the gradient estimate (lower variance = more stable training)
    """
    if args is None:
        args = parse_args()
    num_episodes = 4 if args.smoke else args.num_episodes
    gamma = args.gamma
    lr = args.lr
    out_dir = os.path.join("runs", "smoke_reinforce_baseline") if args.smoke else "output"
    os.makedirs(out_dir, exist_ok=True)
    device = "cpu" if args.smoke or args.device == "cpu" else (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )

    print("=" * 60)
    print("  REINFORCE variance reduction comparison experiment")
    print("=" * 60)
    print(f"  Training episodes: {num_episodes}")
    print(f"  Discount factor gamma: {gamma}")
    print(f"  Learning rate: {lr}")
    print("=" * 60)

    # ---------- Experiment 1: Vanilla REINFORCE ----------
    print("\n[Experiment 1] Training Vanilla REINFORCE (no baseline)...")
    vanilla_rewards, vanilla_grads = train_vanilla_reinforce(
        num_episodes=num_episodes, gamma=gamma, lr=lr, seed=args.seed, device=device
    )

    # ---------- Experiment 2: REINFORCE + Value Baseline ----------
    print("\n[Experiment 2] Training REINFORCE + Value Baseline...")
    baseline_rewards, baseline_grads = train_reinforce_with_baseline(
        num_episodes=num_episodes, gamma=gamma, lr=lr, seed=args.seed, device=device
    )

    # ---------- Variance statistics comparison ----------
    print("\n" + "=" * 60)
    print("  Variance comparison statistics")
    print("=" * 60)

    vanilla_grad_var = np.var(vanilla_grads)
    baseline_grad_var = np.var(baseline_grads)

    print(f"  Vanilla REINFORCE gradient estimate variance: {vanilla_grad_var:.6f}")
    print(f"  REINFORCE+Value Baseline gradient estimate variance: {baseline_grad_var:.6f}")

    if vanilla_grad_var > 0:
        ratio = vanilla_grad_var / max(baseline_grad_var, 1e-10)
        print(f"  Variance ratio (Vanilla/Value Baseline): {ratio:.2f}x")
        print(f"  Value Baseline reduces variance to {1/ratio*100:.1f}% of the original")

    print(f"\n  Vanilla REINFORCE last-50-episode avg: {np.mean(vanilla_rewards[-50:]):.1f}")
    print(f"  REINFORCE+Value Baseline last-50-episode avg: {np.mean(baseline_rewards[-50:]):.1f}")
    print("=" * 60)

    # ---------- Comparison plot 1: reward curve ----------
    plot_reward_comparison(vanilla_rewards, baseline_rewards, out_dir, window=50)

    # ---------- Comparison plot 2: variance comparison ----------
    plot_variance_comparison(vanilla_grads, baseline_grads, out_dir, window=50)


# ==========================================
# Part 7: Plot the reward comparison curve
# ==========================================
def plot_reward_comparison(vanilla_rewards, baseline_rewards, out_dir="output", window=50):
    """
    Plot the reward curve comparison between the two experiments

    Includes both raw curves and moving-average curves, visually showing:
        - Whether the Value Baseline version converges faster
        - Whether the Value Baseline version is more stable (less fluctuation)
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Vanilla REINFORCE
    ax.plot(vanilla_rewards, alpha=0.2, color='steelblue')
    vanilla_avg = [np.mean(vanilla_rewards[max(0, i-window+1):i+1])
                   for i in range(len(vanilla_rewards))]
    ax.plot(vanilla_avg, color='steelblue', linewidth=2.0,
            label='Vanilla REINFORCE')

    # REINFORCE + Value Baseline
    ax.plot(baseline_rewards, alpha=0.2, color='crimson')
    baseline_avg = [np.mean(baseline_rewards[max(0, i-window+1):i+1])
                    for i in range(len(baseline_rewards))]
    ax.plot(baseline_avg, color='crimson', linewidth=2.0,
            label='REINFORCE + Value Baseline')

    ax.set_xlabel('Training episode', fontsize=12)
    ax.set_ylabel('Episode reward', fontsize=12)
    ax.set_title('REINFORCE reward curve comparison (Vanilla vs Value Baseline)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    reward_path = os.path.join(out_dir, 'reinforce_baseline_reward_comparison.png')
    plt.savefig(reward_path, dpi=150, bbox_inches='tight')
    print(f"  Reward comparison plot saved to {reward_path}")
    plt.close()


# ==========================================
# Part 8: Plot the variance comparison chart
# ==========================================
def plot_variance_comparison(vanilla_grads, baseline_grads, out_dir="output", window=50):
    """
    Plot the moving-window comparison of gradient estimate variance

    This chart is the core of the experiment: it shows how the Value Baseline
    reduces the variance of the policy gradient. Lower variance means more
    stable training and more reliable convergence.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Compute the moving-window variance
    def moving_variance(data, w):
        variances = []
        for i in range(len(data)):
            start = max(0, i - w + 1)
            variances.append(np.var(data[start:i + 1]))
        return variances

    vanilla_var = moving_variance(vanilla_grads, window)
    baseline_var = moving_variance(baseline_grads, window)

    ax.plot(vanilla_var, color='steelblue', linewidth=1.5, alpha=0.8,
            label='Vanilla REINFORCE')
    ax.plot(baseline_var, color='crimson', linewidth=1.5, alpha=0.8,
            label='REINFORCE + Value Baseline')

    ax.set_xlabel('Training episode', fontsize=12)
    ax.set_ylabel(f'Gradient estimate variance (window={window})', fontsize=12)
    ax.set_title('Policy gradient variance comparison -- variance reduction from Value Baseline', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add an annotation arrow highlighting the variance gap
    if len(vanilla_var) > 100:
        mid_point = len(vanilla_var) // 2
        ax.annotate(
            'Value Baseline reduces variance',
            xy=(mid_point, baseline_var[mid_point]),
            xytext=(mid_point + 50, max(vanilla_var) * 0.7),
            fontsize=11,
            arrowprops=dict(arrowstyle='->', color='gray'),
            color='gray',
        )

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    var_path = os.path.join(out_dir, 'reinforce_baseline_variance_comparison.png')
    plt.savefig(var_path, dpi=150, bbox_inches='tight')
    print(f"  Variance comparison plot saved to {var_path}")
    plt.close()


# ==========================================
# Program entry point
# ==========================================
if __name__ == "__main__":
    run_comparison()
