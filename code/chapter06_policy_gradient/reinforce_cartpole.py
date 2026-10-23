"""
Chapter 6: REINFORCE policy gradient algorithm -- CartPole-v1
Implements the most classic policy gradient method from scratch, illustrating
"do more of the good actions, do less of the bad ones"

Core algorithmic idea:
    An intuitive view of the policy gradient -- if an episode scores highly,
    then every action taken in that episode should be "encouraged" (its
    probability increased); otherwise it should be "suppressed" (its
    probability decreased).

REINFORCE formula:
    grad J(theta) ~= sum_t [grad log pi(a_t|s_t)] * G_t
    where G_t is the discounted cumulative return starting from time step t

How to run:
    python reinforce_cartpole.py
    python reinforce_cartpole.py --smoke   # fast CPU smoke test writing under runs/
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
from collections import deque

# Create the output directory
os.makedirs("output", exist_ok=True)

# Configure Chinese fonts to ensure chart titles and labels render correctly
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: Policy Network
# ==========================================
class PolicyNetwork(nn.Module):
    """
    Policy network: maps a state to an action probability distribution

    Architecture: 4 (state dim) -> 128 -> 128 -> 2 (action dim)
    The output is normalized with Softmax to give a valid probability distribution

    CartPole's state space: [cart position, cart velocity, pole angle, pole angular velocity]
    CartPole's action space: [push left, push right]
    """

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   # Input layer -> first hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   # First hidden layer -> second hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),   # Second hidden layer -> output layer (logits)
        )

    def forward(self, x):
        """
        Forward pass: state -> action probabilities

        Args:
            x: state tensor, shape [batch_size, state_dim]
        Returns:
            probs: action probabilities, shape [batch_size, action_dim], after Softmax
        """
        logits = self.network(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


# ==========================================
# Part 2: Compute discounted cumulative returns
# ==========================================
def compute_returns(rewards, gamma=0.99):
    """
    Compute the discounted cumulative return G_t for each step, working backward

    Formula: G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Example (gamma=0.99):
        rewards = [1, 1, 1, 1, 1]
        G_0 = 1 + 0.99*1 + 0.99^2*1 + ... ~= 4.90
        G_4 = 1

    Args:
        rewards: list of immediate rewards at each step
        gamma: discount factor; closer to 1 means future rewards matter more
    Returns:
        returns: list of discounted cumulative returns at each step
    """
    returns = []
    G = 0  # Cumulative return

    # Iterate from back to front, using the recurrence G_t = r_t + gamma * G_{t+1}
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)  # Insert at the head of the list to preserve time order

    return returns


# ==========================================
# Part 3: Collect a full episode trajectory
# ==========================================
def collect_episode(policy, env, device="cpu"):
    """
    Let the policy network run a full episode in the environment and collect
    the trajectory data

    REINFORCE is an on-policy algorithm, so it must collect data with the
    current policy; that data is discarded once used, and the next round
    must collect fresh data.

    Args:
        policy: policy network
        env: Gymnasium environment
    Returns:
        states: list of states
        actions: list of actions
        rewards: list of rewards
        episode_reward: total reward for the episode
    """
    state, _ = env.reset()
    states, actions, rewards = [], [], []

    done = False
    truncated = False

    while not (done or truncated):
        # Convert the state to a tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # Add the batch dimension

        # Get the action probability distribution
        with torch.no_grad():
            probs = policy(state_tensor)

        # Sample an action from the probability distribution (key to exploration! not argmax)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        # Execute the action and observe the result
        next_state, reward, done, truncated, _ = env.step(action)

        # Store the transition data
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

    episode_reward = sum(rewards)
    return states, actions, rewards, episode_reward


# ==========================================
# Part 4: Train on one episode (the core REINFORCE update)
# ==========================================
def train_one_episode(policy, optimizer, states, actions, returns, device="cpu"):
    """
    The core of REINFORCE: update the network parameters using the policy gradient formula

    Loss function = - sum_t [log pi(a_t|s_t) * G_t]

    The gradient of this loss function is exactly equal to the policy gradient:
    grad(loss) = - sum_t [grad log pi(a_t|s_t) * G_t] = -grad J(theta)

    So minimizing loss = maximizing J(theta) (the expected return)

    Args:
        policy: policy network
        optimizer: optimizer
        states: list of states
        actions: list of actions
        returns: list of discounted cumulative returns
        device: torch device for tensors
    Returns:
        loss_value: this round's loss value
    """
    # Convert the data to tensors
    states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
    actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=device)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

    # Forward pass: get the action probabilities for each state
    probs = policy(states_tensor)

    # Compute the log probability of the action taken, log pi(a_t|s_t)
    # gather(1, actions) selects the probability of the corresponding action for each state
    action_probs = probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    log_probs = torch.log(action_probs + 1e-8)  # Add a small constant to avoid log(0)

    # Policy gradient loss: -log pi(a_t|s_t) * G_t
    # Intuition:
    #   If G_t > 0 (a good outcome), -log_prob * G_t < 0, so gradient descent increases log_prob -> increases the probability
    #   If G_t < 0 (a bad outcome), -log_prob * G_t > 0, so gradient descent decreases log_prob -> decreases the probability
    loss = -(log_probs * returns_tensor).mean()

    # Backward pass + parameter update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ==========================================
# Part 5: Main training loop
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="REINFORCE on CartPole-v1 (Chapter 6)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--num-episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer width")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test mode: tiny CPU run writing only under runs/",
    )
    return parser.parse_args()


def resolve_device(choice):
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def train(args=None):
    """
    The full REINFORCE training pipeline

    Hyperparameter notes:
        - num_episodes = 500: train for 500 episodes
        - gamma = 0.99: discount factor, favoring long-term reward
        - learning_rate = 1e-3: learning rate
        - hidden_dim = 128: hidden layer width
    """
    if args is None:
        args = parse_args()
    # ---------- Hyperparameters ----------
    num_episodes = 5 if args.smoke else args.num_episodes
    gamma = args.gamma
    learning_rate = args.lr
    hidden_dim = args.hidden_dim

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.smoke:
        device = "cpu"
    out_dir = os.path.join("runs", "smoke_reinforce_cartpole") if args.smoke else "output"
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Initialization ----------
    env = gym.make("CartPole-v1")
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    policy = PolicyNetwork(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Record the training process
    episode_rewards = []  # Total reward for each episode
    episode_losses = []   # Loss for each episode

    print("=" * 60)
    print("  REINFORCE policy gradient -- CartPole-v1 training")
    print("=" * 60)
    print(f"  Hyperparameters:")
    print(f"    Episodes: {num_episodes}")
    print(f"    Discount factor gamma: {gamma}")
    print(f"    Learning rate: {learning_rate}")
    print(f"    Hidden layer dimension: {hidden_dim}")
    print("=" * 60)

    # ---------- Training loop ----------
    for episode in range(num_episodes):
        # Step 1: collect a full episode trajectory with the current policy
        states, actions, rewards, episode_reward = collect_episode(policy, env, device)

        # Step 2: compute the discounted cumulative returns
        returns = compute_returns(rewards, gamma=gamma)

        # Step 3: perform the policy gradient update
        loss_value = train_one_episode(policy, optimizer, states, actions, returns, device)

        # Record the data
        episode_rewards.append(episode_reward)
        episode_losses.append(loss_value)

        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            avg_reward = np.mean(recent_rewards)
            print(
                f"  Episode {episode + 1:4d}/{num_episodes} | "
                f"Reward: {episode_reward:6.1f} | "
                f"Last-50 avg: {avg_reward:6.1f} | "
                f"Loss: {loss_value:.4f}"
            )

    env.close()

    # ---------- Training results summary ----------
    print("=" * 60)
    print("  Training complete!")
    print(f"  Average reward over the last 50 episodes: {np.mean(episode_rewards[-50:]):.1f}")
    print(f"  Best episode reward: {np.max(episode_rewards):.1f}")
    print("=" * 60)

    # ---------- Plot the training curve ----------
    plot_training_curve(episode_rewards, out_dir)


# ==========================================
# Part 6: Plot the training curve
# ==========================================
def plot_training_curve(episode_rewards, out_dir="output"):
    """
    Plot the reward curve and the moving-average line

    The moving average (window=50) shows the learning trend more clearly,
    filtering out the random fluctuation of individual episodes.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Raw reward curve (light color, shows fluctuation)
    ax.plot(episode_rewards, alpha=0.3, color='steelblue', label='Episode reward (raw)')

    # Moving-average curve (dark color, shows the trend)
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = []
        for i in range(len(episode_rewards)):
            start = max(0, i - window + 1)
            moving_avg.append(np.mean(episode_rewards[start:i + 1]))
        ax.plot(moving_avg, color='crimson', linewidth=2.0,
                label=f'Moving average (window={window})')

    ax.set_xlabel('Training episode', fontsize=12)
    ax.set_ylabel('Episode reward', fontsize=12)
    ax.set_title('REINFORCE policy gradient -- CartPole-v1 training curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    curve_path = os.path.join(out_dir, 'reinforce_cartpole_rewards.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    print(f"  Training curve saved to {curve_path}")
    plt.close()


# ==========================================
# Program entry point
# ==========================================
if __name__ == "__main__":
    train()
