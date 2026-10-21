# rollout_worker.py

class RolloutWorker:
    """
    Drive the agent loop and collect multi-turn trajectories.
    Trajectory structure: [(prompt, response_1, obs_1, response_2, obs_2, ..., final_response), reward]
    """

    def __init__(self, policy, env, max_turns=5):
        self.policy = policy
        self.env = env
        self.max_turns = max_turns

    def rollout(self, prompt: str, reward_fn) -> dict:
        """
        Run one complete agent loop and return the trajectory and its reward.
        reward_fn: callable(trajectory) -> float
        """
        messages = [{"role": "user", "content": prompt}]
        trajectory = {"prompt": prompt, "interactions": []}

        for turn in range(self.max_turns):
            # The model generates the next action
            context = self._format_context(messages)
            model_output = self.policy.generate(context)

            # Parse the model output: code execution or final answer?
            action = self._parse_action(model_output)

            if action["type"] == "finish":
                trajectory["interactions"].append({
                    "turn": turn,
                    "context": context,
                    "response": model_output,
                    "action": action,
                    "observation": None,
                })
                trajectory["final_response"] = action.get("answer", model_output)
                break

            # The environment executes the action
            obs = self.env.step(action["type"], action["args"])

            trajectory["interactions"].append({
                "turn": turn,
                "context": context,
                "response": model_output,
                "action": action,
                "observation": obs["observation"],
            })

            messages.append({"role": "assistant", "content": model_output})
            messages.append({"role": "user", "content": f"Execution result:\n{obs['observation']}"})

            if obs.get("done"):
                break

        # Compute the reward
        trajectory["reward"] = reward_fn(trajectory)
        return trajectory

    def _format_context(self, messages):
        """Join the multi-turn message list into a prompt the model can understand."""
        parts = []
        for msg in messages:
            if msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
            else:
                parts.append(f"Assistant: {msg['content']}")
        return "\n".join(parts)

    def _parse_action(self, model_output: str) -> dict:
        """
        Parse the action type and arguments from the model output.
        Simplified implementation: parse by markers.
        """
        if "```python" in model_output:
            # Extract the code block
            code = model_output.split("```python")[1].split("```")[0]
            return {"type": "execute_code", "args": {"code": code}}
        elif "FINAL ANSWER:" in model_output:
            answer = model_output.split("FINAL ANSWER:")[1].strip()
            return {"type": "finish", "answer": answer}
        else:
            return {"type": "execute_code", "args": {"code": model_output}}
