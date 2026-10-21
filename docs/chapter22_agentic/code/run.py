# run.py
from transformers import AutoModelForCausalLM, AutoTokenizer

from environment import SandboxEnv
from policy import Policy
from trainer import GRPOAgentTrainer

# Load a small model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the components
env = SandboxEnv(timeout=10)
policy = Policy(model, tokenizer, lr=5e-5)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
policy.set_ref_model(ref_model)


TASK_EXPECTED_OUTPUTS = {
    "Write Python code to compute F(10), with F(0)=0 and F(1)=1. Print only the result.": "55",
    "Write Python code to test whether 'racecar' is a palindrome. Print only True or False.": "True",
    "Write Python code to sort [5, 1, 4, 2, 8] in ascending order. Print only the sorted list.": "[1, 2, 4, 5, 8]",
}


# Define a verifiable reward: the last line must equal the expected output for the task
def code_reward(trajectory):
    """If the last line of any code execution equals the expected output, reward = 1."""
    expected = TASK_EXPECTED_OUTPUTS[trajectory["prompt"]]
    for interaction in trajectory["interactions"]:
        obs = interaction.get("observation", "")
        output_lines = [line.strip() for line in obs.splitlines() if line.strip()]
        if output_lines and output_lines[-1] == expected:
            return 1.0
    return 0.0


# Training prompts
prompts = list(TASK_EXPECTED_OUTPUTS)

# Start training
trainer = GRPOAgentTrainer(
    policy=policy,
    env=env,
    reward_fn=code_reward,
    group_size=4,
    max_turns=3,
)
history = trainer.fit(prompts, n_steps=30)
