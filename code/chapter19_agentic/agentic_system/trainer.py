# trainer.py

from rollout_worker import RolloutWorker


class GRPOAgentTrainer:
    """
    Orchestrate the agentic RL training loop.
    rollout -> reward -> train -> repeat
    """

    def __init__(self, policy, env, reward_fn, group_size=4, max_turns=5):
        if group_size < 2:
            raise ValueError("GRPO group_size must be at least 2")
        self.policy = policy
        self.env = env
        self.reward_fn = reward_fn
        self.group_size = group_size
        self.worker = RolloutWorker(policy, env, max_turns=max_turns)
        self.history = []

    def fit(self, prompts: list, n_steps: int = 50):
        """
        Main training loop.
        prompts: list of training prompts
        n_steps: number of training steps
        """
        for step in range(n_steps):
            # ---- Rollout phase ----
            batch_trajectories = []
            for prompt in prompts:
                group = []
                for _ in range(self.group_size):
                    traj = self.worker.rollout(prompt, self.reward_fn)
                    group.append(traj)
                batch_trajectories.append(group)

            # ---- Reward normalization (GRPO's within-group comparison)----
            all_rewards = []
            for group in batch_trajectories:
                group_rewards = [t["reward"] for t in group]
                mean_r = sum(group_rewards) / len(group_rewards)
                std_r = (
                    sum((r - mean_r) ** 2 for r in group_rewards)
                    / (len(group_rewards) - 1)
                ) ** 0.5
                for t, r in zip(group, group_rewards):
                    t["advantage"] = (
                        0.0 if std_r < 1e-8 else (r - mean_r) / std_r
                    )
                all_rewards.extend(group_rewards)

            # ---- Train phase ----
            train_data = []
            for group in batch_trajectories:
                for traj in group:
                    generated_turns = [
                        (interaction["context"], interaction["response"])
                        for interaction in traj["interactions"]
                    ]
                    train_data.append(
                        (
                            generated_turns,
                            traj["advantage"],
                        )
                    )

            loss = self.policy.train_step_with_advantage(train_data)

            # ---- Log metrics ----
            mean_reward = sum(all_rewards) / len(all_rewards)
            self.history.append(
                {
                    "step": step,
                    "loss": loss,
                    "mean_reward": mean_reward,
                    "max_reward": max(all_rewards),
                }
            )
            if step % 5 == 0:
                print(
                    f"Step {step:3d} | loss={loss:.4f} | "
                    f"reward_mean={mean_reward:.3f} | "
                    f"reward_max={max(all_rewards):.3f}"
                )

        return self.history
