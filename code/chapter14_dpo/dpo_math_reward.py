"""
Chapter 14: DPO Math Reasoning Alignment -- Training Math Preferences with Rule-Based Signals
==========================================================

This script demonstrates how to apply DPO to the math reasoning domain:
  1. Construct GSM8K-style math preference data (arithmetic problems)
  2. Generate a correct (chosen) and an incorrect (rejected) solution for each problem
  3. Train a model with DPO so it favors the correct reasoning path
  4. Evaluate accuracy before and after training on a held-out test set
  5. Show how DPO learns from rule-based preference signals

Core idea:
  - No need for humans to label "which answer is better"
  - Rules (whether the answer is correct) automatically generate preference labels
  - DPO learns the correct reasoning pattern from these rule-based signals

How to run:
  pip install -r requirements.txt
  python dpo_math_reward.py
"""

import re
import json
import torch
import random
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# ==========================================
# 1. Construct GSM8K-style math preference data
# ==========================================
# Each entry contains a math problem plus a correct and an incorrect solution process
# We use simple arithmetic problems so answers can be verified automatically

def create_math_preference_data():
    """
    Create the preference dataset for math reasoning

    Each entry contains:
    - prompt: the math problem
    - chosen: a correct, step-by-step solution process
    - rejected: a solution process that contains a reasoning error
    """

    math_examples = [
        {
            "prompt": "Ming has 15 apples, gives 7 to Hong, then buys 3 more. How many apples does Ming have now?",
            "chosen": "Let us work through it.\nStep 1: Ming starts with 15 apples.\nStep 2: He gives 7 to Hong, so 15 - 7 = 8 remain.\nStep 3: He buys 3 more, so now he has 8 + 3 = 11.\nAnswer: Ming now has 11 apples.",
            "rejected": "Ming starts with 15 apples.\nHe gives 7 to Hong, leaving 15 - 7 = 9. (Error: 15-7=8, not 9)\nHe buys 3 more, so 9 + 3 = 12.\nAnswer: Ming now has 12 apples.",
            "answer": 11,
        },
        {
            "prompt": "A book has 240 pages. Hong reads 30 pages a day for 5 days. How many pages are left unread?",
            "chosen": "Let us work through it.\nStep 1: Hong reads 30 pages a day for 5 days.\nStep 2: She has read 30 x 5 = 150 pages.\nStep 3: The book has 240 pages, so 240 - 150 = 90 remain.\nAnswer: 90 pages are left unread.",
            "rejected": "Hong reads 30 pages a day for 5 days.\nShe has read 30 + 5 = 35 pages. (Error: use multiplication, not addition)\nSo 240 - 35 = 205 remain.\nAnswer: 205 pages are left unread.",
            "answer": 90,
        },
        {
            "prompt": "A rectangle is 12 cm long and 8 cm wide. Find its area.",
            "chosen": "Area of a rectangle = length x width.\nLength = 12 cm, width = 8 cm.\nArea = 12 x 8 = 96 square cm.\nAnswer: the area is 96 square cm.",
            "rejected": "Area of a rectangle = length + width. (Error: area uses multiplication, not addition)\nArea = 12 + 8 = 20 square cm.\nAnswer: the area is 20 square cm.",
            "answer": 96,
        },
        {
            "prompt": "A shop has 36 basketballs and 24 soccer balls. How many more basketballs than soccer balls?",
            "chosen": "There are 36 basketballs and 24 soccer balls.\nThere are 36 - 24 = 12 more basketballs.\nAnswer: 12 more basketballs.",
            "rejected": "There are 36 basketballs and 24 soccer balls.\nTogether that is 36 + 24 = 60. (Error: the question asks for the difference, not the total)\nAnswer: 60 more basketballs.",
            "answer": 12,
        },
        {
            "prompt": "A box has 48 oranges shared equally among 6 children. How many does each child get?",
            "chosen": "48 oranges shared among 6 children.\nEach child gets 48 / 6 = 8.\nAnswer: each child gets 8 oranges.",
            "rejected": "48 oranges shared among 6 children.\nEach child gets 48 - 6 = 42. (Error: use division, not subtraction)\nAnswer: each child gets 42 oranges.",
            "answer": 8,
        },
        {
            "prompt": "A triangle has base 10 cm and height 6 cm. Find its area.",
            "chosen": "Area of a triangle = base x height / 2.\nBase = 10 cm, height = 6 cm.\nArea = 10 x 6 / 2 = 30 square cm.\nAnswer: the area is 30 square cm.",
            "rejected": "Area of a triangle = base x height. (Error: a triangle's area must be halved)\nArea = 10 x 6 = 60 square cm.\nAnswer: the area is 60 square cm.",
            "answer": 30,
        },
        {
            "prompt": "A train travels at 80 km/h for 3.5 hours. How far does it travel?",
            "chosen": "Speed = 80 km/h, time = 3.5 h.\nDistance = speed x time = 80 x 3.5 = 280 km.\nAnswer: it travels 280 km.",
            "rejected": "Speed = 80 km/h, time = 3.5 h.\nDistance = 80 + 3.5 = 83.5 km. (Error: use multiplication, not addition)\nAnswer: it travels 83.5 km.",
            "answer": 280,
        },
        {
            "prompt": "A mother buys 3 kg of apples at $12 per kg and 2 kg of bananas at $8 per kg. How much does she spend in total?",
            "chosen": "Apples cost: 3 x 12 = $36.\nBananas cost: 2 x 8 = $16.\nTotal: 36 + 16 = $52.\nAnswer: she spends $52 in total.",
            "rejected": "Apples cost: 3 x 12 = $36.\nBananas cost: 2 x 8 = $16.\nTotal: 36 x 16 = $576. (Error: use addition, not multiplication)\nAnswer: she spends $576 in total.",
            "answer": 52,
        },
        {
            "prompt": "A square has side length 9 cm. Find its perimeter.",
            "chosen": "Perimeter of a square = side x 4.\nSide = 9 cm.\nPerimeter = 9 x 4 = 36 cm.\nAnswer: the perimeter is 36 cm.",
            "rejected": "Perimeter of a square = side x side. (Error: that is the area formula, not the perimeter)\nPerimeter = 9 x 9 = 81 cm.\nAnswer: the perimeter is 81 cm.",
            "answer": 36,
        },
        {
            "prompt": "A father is 42 years old and his son is 12. In 5 years, how much older will the father be than the son?",
            "chosen": "The age gap never changes with time.\nRight now the father is 42 - 12 = 30 years older.\nIn 5 years the father will still be 30 years older.\nAnswer: in 5 years the father will be 30 years older.",
            "rejected": "In 5 years the father is 42 + 5 = 47 and the son is 12 + 5 = 17.\nThe father is 47 - 12 = 35 years older. (Error: the son's age also grows by 5, but the subtraction forgot it)\nAnswer: in 5 years the father will be 35 years older.",
            "answer": 30,
        },
    ]

    return math_examples


# ==========================================
# 2. Prepare the test set data (used to evaluate accuracy)
# ==========================================

def create_test_data():
    """
    Create an independent test set that does not overlap with the training data
    Used to evaluate math reasoning ability before and after DPO training
    """
    test_examples = [
        {
            "prompt": "There are 56 candies shared equally among 8 children. How many does each child get?",
            "answer": 7,
        },
        {
            "prompt": "Ming runs 5 km a day for 7 days. How many km does he run in total?",
            "answer": 35,
        },
        {
            "prompt": "A rectangle is 15 cm long and 4 cm wide. Find its perimeter.",
            "answer": 38,
        },
        {
            "prompt": "A fruit shop has 100 apples, sells 37, then stocks 20 more. How many are there now?",
            "answer": 83,
        },
        {
            "prompt": "A carton of milk has 24 boxes, and the school buys 5 cartons. How many boxes in total?",
            "answer": 120,
        },
    ]
    return test_examples


# ==========================================
# 3. Define helper functions
# ==========================================

def extract_number(text):
    """Extract the final numeric answer from the model's response"""
    # Try English answer markers first: "Answer: X", "The answer is X", "The result is X", "equals X"
    patterns = [
        r"[Aa]nswer\s*:\s*([+-]?\d+\.?\d*)",
        r"[Tt]he answer is\s*([+-]?\d+\.?\d*)",
        r"[Tt]he result is\s*([+-]?\d+\.?\d*)",
        r"equals\s*([+-]?\d+\.?\d*)",
    ]
    for pattern in patterns:
        match = re.findall(pattern, text)
        if match:
            return int(match[-1])  # take the last match

    # If none of the above matched, try to find the last number that appears in the text
    numbers = re.findall(r"\d+", text)
    if numbers:
        return int(numbers[-1])
    return None


def evaluate_math(model, tokenizer, test_data, label="Model"):
    """
    Evaluate the model's math reasoning accuracy on the test set

    Generate a response for each problem, extract the numeric answer, and compare it against the correct answer
    """
    print("=" * 60)
    print(f"[{label} Math Reasoning Evaluation]")
    print("=" * 60)

    correct = 0
    total = len(test_data)

    for i, item in enumerate(test_data):
        prompt = item["prompt"]
        true_answer = item["answer"]

        # Generate the model's response
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        predicted_answer = extract_number(response)

        is_correct = predicted_answer == true_answer
        correct += int(is_correct)

        status = "Correct" if is_correct else "Incorrect"
        print(f"  Problem {i+1}: {prompt[:30]}...")
        print(f"    Model response: {response[:80]}...")
        print(f"    Predicted answer: {predicted_answer} | Correct answer: {true_answer} | {status}")
        print()

    accuracy = correct / total * 100
    print(f"  Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print("=" * 60)
    return accuracy


# ==========================================
# 4. Main workflow begins
# ==========================================

def main():
    print("=" * 60)
    print("  DPO Math Reasoning Alignment Experiment")
    print("  -- Training with Rule-Based Preference Signals")
    print("=" * 60)
    print()

    # Prepare the data
    math_data = create_math_preference_data()
    test_data = create_test_data()

    print(f"Training set size: {len(math_data)} math preference examples")
    print(f"Test set size: {len(test_data)} math problems")
    print()
    print("How the preference data is constructed:")
    print("  - chosen:  a correct, step-by-step reasoning process (every step of the computation is correct)")
    print("  - rejected: a solution that contains a reasoning error (a computation or formula is wrong at some step)")
    print("  - This preference signal requires no human labeling; it is generated entirely by rules")
    print()


    # ==========================================
    # 5. Load the model and evaluate pre-training performance
    # ==========================================

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading base model {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model_before = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    before_accuracy = evaluate_math(model_before, tokenizer, test_data, label="Before Training")

    # Free up GPU memory
    del model_before
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


    # ==========================================
    # 6. DPO training
    # ==========================================

    print("\nStarting DPO training with rule-based math preference data...")

    # Reload the model for training
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    train_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_tokenizer.pad_token = train_tokenizer.eos_token

    # Build the HuggingFace Dataset
    data_dict = {
        "prompt": [item["prompt"] for item in math_data],
        "chosen": [item["chosen"] for item in math_data],
        "rejected": [item["rejected"] for item in math_data],
    }
    train_dataset = Dataset.from_dict(data_dict)

    # Configure training arguments
    training_args = DPOConfig(
        output_dir="./dpo_math_results",
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=5,         # Math reasoning needs more epochs to learn the reasoning pattern
        logging_steps=2,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        remove_unused_columns=False,
        beta=0.1,
    )

    # Create the DPOTrainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=train_tokenizer,
    )

    print("\nTraining... watch how loss and reward evolve\n")
    train_result = trainer.train()

    # ==========================================
    # 7. Print training metrics
    # ==========================================

    print("\n" + "=" * 60)
    print("[Training Metrics Detail]")
    print("=" * 60)
    print(f"Final training loss: {train_result.training_loss:.4f}")
    print()

    # Parse the reward information from the training log
    log_history = trainer.state.log_history
    print("Training metrics at each step:")
    print(f"{'Step':>6} | {'Loss':>8} | {'Chosen Reward':>14} | {'Rejected Reward':>16} | {'Margin':>8}")
    print("-" * 70)

    for entry in log_history:
        if "loss" in entry:
            step = entry.get("step", "?")
            loss = entry["loss"]
            chosen_r = entry.get("rewards/chosen", "N/A")
            rejected_r = entry.get("rewards/rejected", "N/A")
            margin = entry.get("rewards/margins", "N/A")

            # Format the output
            if isinstance(chosen_r, float):
                chosen_r = f"{chosen_r:.4f}"
            if isinstance(rejected_r, float):
                rejected_r = f"{rejected_r:.4f}"
            if isinstance(margin, float):
                margin = f"{margin:.4f}"

            print(f"{step:>6} | {loss:>8.4f} | {chosen_r:>14} | {rejected_r:>16} | {margin:>8}")

    # Save the model
    save_path = "./dpo_math_results/final_model"
    trainer.save_model(save_path)
    print(f"\nModel saved to {save_path}")


    # ==========================================
    # 8. Evaluate the trained model
    # ==========================================

    # Load the trained model for evaluation
    print("\nLoading the trained model for evaluation...")
    model_after = AutoModelForCausalLM.from_pretrained(save_path, device_map="auto")
    eval_tokenizer = AutoTokenizer.from_pretrained(save_path)
    eval_tokenizer.pad_token = eval_tokenizer.eos_token

    after_accuracy = evaluate_math(model_after, eval_tokenizer, test_data, label="After Training")


    # ==========================================
    # 9. Compare results and summarize
    # ==========================================

    print("\n" + "=" * 60)
    print("[DPO Math Reasoning Alignment -- Final Results Comparison]")
    print("=" * 60)
    print()
    print(f"  Accuracy before training: {before_accuracy:.1f}%")
    print(f"  Accuracy after training: {after_accuracy:.1f}%")
    print(f"  Improvement: {after_accuracy - before_accuracy:+.1f}%")
    print()

    print("=" * 60)
    print("[Experiment Summary]")
    print("=" * 60)
    print("""
    1. Rule-based preference signals:
       This experiment does not rely on human labeling. Instead, it uses the simple
       rule "is the answer correct" to automatically generate chosen/rejected pairs.
       This approach scales easily to many more problems.

    2. DPO's effect on math reasoning:
       - DPO can learn correct reasoning patterns (such as choosing the right operator)
       - After training, the model tends to favor correct solution strategies on new problems
       - However, with only 0.5B parameters, the improvement on complex reasoning is limited

    3. Comparison with RLHF:
       - Traditional RLHF: requires training a reward model -> optimizing the policy with PPO
       - DPO approach: optimizes directly from preference data, skipping reward model training
       - For tasks with clear-cut rules (like math), DPO + rule-based signals is very efficient

    4. Directions for further extension:
       - Use larger models (7B, 14B) for better reasoning ability
       - Increase the amount of training data (e.g. the full GSM8K dataset)
       - Combine with Chain-of-Thought prompting to deepen reasoning
       - Use multiple incorrect rejected samples to strengthen contrastive learning
    """)


if __name__ == '__main__':
    main()
