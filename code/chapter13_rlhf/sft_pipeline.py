"""
Chapter 13: SFT (Supervised Fine-Tuning) Pipeline
==================================================

This script demonstrates the first stage of the three-stage RLHF pipeline —
supervised fine-tuning (SFT).
Contents:
  1. Generate Self-Instruct-style instruction-response training data
  2. Load the Qwen2.5-0.5B-Instruct base model
  3. Perform supervised fine-tuning with SFTTrainer
  4. Compare model output quality before and after fine-tuning
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Create the output directory
os.makedirs("output", exist_ok=True)

# ==========================================
# 1. Generate Self-Instruct-style training data
# ==========================================

def generate_sft_data():
    """
    Generate instruction-response training data for SFT.
    Covers four task categories: math, writing, coding, and knowledge Q&A.
    Each entry is an (instruction, response) pair.
    """
    sft_pairs = [
        # ---- Math ----
        {
            "instruction": "Compute (3 + 5) x 2.",
            "response": "Let us work through it step by step:\nFirst compute the addition inside the parentheses: 3 + 5 = 8\nThen multiply the result by 2: 8 x 2 = 16\nSo (3 + 5) x 2 = 16."
        },
        {
            "instruction": "A rectangle is 8 cm long and 5 cm wide. Find its area.",
            "response": "The area of a rectangle is: area = length x width\nPlug in the numbers: area = 8 x 5 = 40 square cm\nSo the area of this rectangle is 40 square cm."
        },
        {
            "instruction": "Add 3/4 and 2/5, giving the result as a fraction.",
            "response": "First find a common denominator: the least common multiple of 4 and 5 is 20.\nConvert both fractions:\n  3/4 = 15/20\n  2/5 = 8/20\nThen add: 15/20 + 8/20 = 23/20\nThe result is 23/20 (that is, 1 and 3/20)."
        },
        # ---- Writing ----
        {
            "instruction": "Describe a spring landscape in three sentences.",
            "response": "A light spring breeze moves across the fields as cherry petals drift down the path and a faint floral scent fills the air. Tender green buds sprout on the willows by the river, glinting in the sunlight. Children run and play on the grass, their laughter mixing with birdsong in a lively scene."
        },
        {
            "instruction": "Write a short encouraging note to a classmate preparing for exams.",
            "response": "Dear friend, you have already put so much effort into preparing, and none of that work is wasted. An exam is only one way of checking what you know; it does not define your full worth. Trust the steady progress you have built up, stay calm, and take each question seriously. Whatever the result, the knowledge and persistence you gained along the way are the most valuable reward. Good luck!"
        },
        {
            "instruction": "Write a short paragraph of about 100 words titled 'My Programming Journey'.",
            "response": "My programming journey began in my first C language class at university. I felt both curious and confused staring at the characters flickering on the screen. As line after line of code went from errors to running successfully, I gradually felt the joy of creating things. From a simple calculator to complex web apps, every debugging success filled me with accomplishment. Programming taught me logical thinking, and also patience and persistence when facing problems."
        },
        # ---- Coding ----
        {
            "instruction": "Write a Python function that checks whether a number is even.",
            "response": "```python\ndef is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n    return n % 2 == 0\n\n# Tests\nprint(is_even(4))   # True\nprint(is_even(7))   # False\n```\nThis function uses the modulo operator (%): if a number divided by 2 leaves remainder 0, it is even."
        },
        {
            "instruction": "Explain the difference between Python lists and tuples.",
            "response": "The main differences between Python lists and tuples are:\n\n1. **Mutability**: lists are mutable and can be changed after creation; tuples are immutable and cannot be modified.\n2. **Syntax**: lists use square brackets `[1, 2, 3]`, tuples use parentheses `(1, 2, 3)`.\n3. **Performance**: tuples are lighter and slightly faster to access.\n4. **Use**: lists suit collections that change often; tuples suit fixed data such as coordinates or settings.\n\nA simple memory aid: a list is like a notebook you can erase, a tuple is like a stone carving that stays fixed."
        },
        {
            "instruction": "Implement a simple bubble sort in Python.",
            "response": "```python\ndef bubble_sort(arr):\n    \"\"\"Bubble sort: repeatedly scan the list, bubbling the largest element to the end.\"\"\"\n    n = len(arr)\n    for i in range(n):\n        swapped = False\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n                swapped = True\n        # If no swap happened in a pass, the list is already sorted\n        if not swapped:\n            break\n    return arr\n\n# Test\nnumbers = [64, 34, 25, 12, 22, 11, 90]\nprint(\"Before:\", numbers)\nprint(\"After:\", bubble_sort(numbers))\n```\nBubble sort runs in O(n^2) time and is best for teaching sorting on small inputs."
        },
        # ---- Knowledge Q&A ----
        {
            "instruction": "Explain what machine learning is in plain language.",
            "response": "Imagine teaching a small child to recognize fruit. You show many photos of apples and bananas and tell the child which is which. After seeing hundreds of photos, the child can tell new fruit apart on their own - that is the core idea of machine learning.\n\nIn technical terms: machine learning lets computers learn patterns from data automatically instead of relying on people to hand-write every rule. Common types include:\n- **Supervised learning**: train on labeled data (such as captioned fruit photos) to make predictions.\n- **Unsupervised learning**: only data, no labels; the model finds patterns in the data itself.\n- **Reinforcement learning**: learn the best strategy by trial and error with reward feedback."
        },
        {
            "instruction": "List the eight planets of the solar system in order from the Sun outward.",
            "response": "The eight planets of the solar system, from nearest to farthest from the Sun:\n\n1. **Mercury** - the smallest planet, closest to the Sun\n2. **Venus** - the hottest planet at the surface\n3. **Earth** - our home, the only known planet with life\n4. **Mars** - known as the red planet\n5. **Jupiter** - the largest planet in the solar system\n6. **Saturn** - famous for its beautiful rings\n7. **Uranus** - the ice giant that spins on its side\n8. **Neptune** - the farthest planet, with the fastest winds\n\nMemory aid: My Very Educated Mother Just Served Us Nachos."
        },
        {
            "instruction": "What is the HTTP protocol? What is the difference between GET and POST requests?",
            "response": "HTTP (HyperText Transfer Protocol) is the rulebook browsers and servers use to talk to each other, like a shared language for a phone call.\n\nGET and POST are the two most common request methods:\n\n| Feature    | GET                          | POST                    |\n|-----------|------------------------------|-------------------------|\n| Purpose    | Fetch data                   | Submit data             |\n| Parameters | In the URL (visible)         | In the request body     |\n| Data size  | Limited (about 2KB)          | Not limited             |\n| Security   | Lower (params visible in URL)| Higher                  |\n| Idempotent | Yes (repeat gives same result)| No                      |\n| Caching    | Can be cached by browsers    | Not cached by default   |\n\nSimply put: GET is like looking something up, POST is like submitting a form."
        },
        {
            "instruction": "What is recursion? Give an everyday example.",
            "response": "Recursion is the programming technique where a function calls itself in its own definition.\n\n**Everyday example - nesting dolls:**\nYou pick up a big doll, open it, and find a smaller doll inside, then open that one to find an even smaller one... until the tiniest one cannot be opened - that is the base case of the recursion.\n\n**Programming example - factorial:**\n```python\ndef factorial(n):\n    if n == 1:        # base case (the tiniest doll)\n        return 1\n    return n * factorial(n - 1)  # open one layer, handle the smaller doll the same way\n\nprint(factorial(5))  # 120\n```\n\nThe two key parts of recursion:\n1. **Base case**: when to stop (otherwise it loops forever)\n2. **Recursive step**: break a big problem into a smaller one and handle it the same way"
        },
        {
            "instruction": "Explain what a database index is and why it speeds up queries.",
            "response": "A database index works like the table of contents of a book.\n\n**Without a contents page**: to find something you must flip through every page - this is a full table scan, and it is very slow.\n\n**With a contents page**: you look up the keyword, find its page number, and jump straight there - that is how an index works.\n\nHow an index works:\n1. The database builds an extra data structure (usually a B+ tree) for a column (such as user ID)\n2. The structure stores values in sorted order so matching rows can be located fast\n3. A query checks the index first, then follows a pointer to the actual data\n\n**Cost**: an index takes extra storage and must be maintained on every insert or update. So put indexes on frequently queried columns, not everywhere."
        },
    ]

    return sft_pairs


# ==========================================
# 2. Format data into conversational form
# ==========================================

def format_sft_dataset(sft_pairs, tokenizer):
    """
    Convert instruction-response pairs into a format the model can train on.
    Uses the chat template to wrap each entry as a multi-turn conversation.
    """
    formatted_texts = []
    for pair in sft_pairs:
        messages = [
            {"role": "user", "content": pair["instruction"]},
            {"role": "assistant", "content": pair["response"]},
        ]
        # Use the tokenizer's chat_template to format the conversation into training text
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append(text)

    return Dataset.from_dict({"text": formatted_texts})


# ==========================================
# 3. Main flow: SFT training
# ==========================================

def main():
    print("=" * 60)
    print("Chapter 13: SFT (Supervised Fine-Tuning) Pipeline")
    print("=" * 60)

    # ---- 3.1 Generate and inspect training data ----
    print("\n[Step 1] Generating Self-Instruct-style training data...")
    sft_pairs = generate_sft_data()
    print(f"  Generated {len(sft_pairs)} instruction-response training examples")
    print(f"  Categories: math ({3}), writing ({3}), coding ({3}), knowledge Q&A ({6})")

    # Print a sample
    sample = sft_pairs[0]
    print(f"\n  Sample example:")
    print(f"    Instruction: {sample['instruction']}")
    print(f"    Response: {sample['response'][:50]}...")

    # ---- 3.2 Load model and tokenizer ----
    print("\n[Step 2] Loading Qwen2.5-0.5B-Instruct model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
    )
    print(f"  Model loaded: {model_name}")

    # ---- 3.3 Test before fine-tuning ----
    print("\n[Step 3] Testing model output before fine-tuning...")
    test_instructions = [
        "Write a Python function that checks odd vs even numbers.",
        "Explain what recursion is.",
    ]

    print("  --- Output before fine-tuning ---")
    before_responses = []
    for inst in test_instructions:
        messages = [{"role": "user", "content": inst}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        before_responses.append(response)
        print(f"  Q: {inst}")
        print(f"  A: {response[:80]}...")
        print()

    # ---- 3.4 Prepare the SFT training dataset ----
    print("[Step 4] Formatting the training dataset...")
    train_dataset = format_sft_dataset(sft_pairs, tokenizer)
    print(f"  Dataset size: {len(train_dataset)} examples")
    print(f"  Data preview: {train_dataset[0]['text'][:100]}...")

    # ---- 3.5 Configure and run SFT training ----
    print("\n[Step 5] Starting SFT training...")
    print("  Hyperparameters:")
    print("    - per_device_train_batch_size = 2")
    print("    - learning_rate = 2e-5")
    print("    - num_train_epochs = 2")
    print("    - max_length = 512")

    sft_config = SFTConfig(
        output_dir="./output/sft_results",
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        num_train_epochs=2,
        max_length=512,
        logging_steps=1,          # Log every step (small dataset)
        save_strategy="epoch",    # Save once per epoch
        report_to="none",         # Don't upload to wandb, etc.
        fp16=False,               # Use float32 on CPU
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Start training
    train_result = trainer.train()

    # ---- 3.6 Print training loss curve data ----
    print("\n[Step 6] Training loss log:")
    print("  " + "-" * 40)
    print("  Step | Training Loss")
    print("  " + "-" * 40)

    # Extract loss data from the training log
    if hasattr(trainer, "state") and trainer.state.log_history:
        for log_entry in trainer.state.log_history:
            if "loss" in log_entry:
                step = log_entry.get("step", "?")
                loss = log_entry["loss"]
                print(f"  {str(step).rjust(4)} | {loss:.4f}")
    else:
        print(f"  Final training loss: {train_result.training_loss:.4f}")

    print("  " + "-" * 40)

    # ---- 3.7 Test after fine-tuning ----
    print("\n[Step 7] Testing model output after fine-tuning...")
    print("  --- Output after fine-tuning ---")

    # Load the fine-tuned model for inference
    model.eval()
    after_responses = []
    for inst in test_instructions:
        messages = [{"role": "user", "content": inst}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        after_responses.append(response)
        print(f"  Q: {inst}")
        print(f"  A: {response[:80]}...")
        print()

    # ---- 3.8 Before/after comparison summary ----
    print("=" * 60)
    print("Before/after fine-tuning comparison summary:")
    print("=" * 60)
    for i, inst in enumerate(test_instructions):
        print(f"\n  Instruction: {inst}")
        print(f"  Before: {before_responses[i][:60]}...")
        print(f"  After: {after_responses[i][:60]}...")

    # ---- 3.9 Save the SFT model ----
    print("\n[Step 8] Saving the SFT model...")
    save_path = "./output/sft_results/sft_model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  SFT model saved to: {save_path}")
    print("  This model will be used in the next stage (reward model training / PPO alignment).")

    # Also save the training data for use in later stages
    data_save_path = "./output/sft_results/sft_training_data.json"
    with open(data_save_path, "w", encoding="utf-8") as f:
        json.dump(sft_pairs, f, ensure_ascii=False, indent=2)
    print(f"  Training data saved to: {data_save_path}")

    print("\n" + "=" * 60)
    print("SFT stage complete!")
    print("Next, run reward_model_training.py to train the reward model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
