import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from arith_cot import (TinyReasoner, build_examples, decode, encode,
                       make_splits, parse_answer, render, set_seed,
                       train_reasoner)

set_seed(0)
device = torch.device("cpu")
train_pairs, test_pairs = make_splits(2000, 8, 0)
examples = build_examples(train_pairs)
model = TinyReasoner().to(device)
train_reasoner(model, examples, epochs=60, batch_size=256, lr=3e-3,
               device=device, seed=0)
model.eval()
for a, b in test_pairs:
    prompt, true_comp, ans = render(a, b)
    ids = torch.tensor([encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        full = model.generate(ids, torch.tensor([len(prompt)]), 28,
                              temperature=0.0)
    gen = decode(full[0, len(prompt):])
    print(f"prompt={prompt} truth={true_comp} (= {ans})")
    print(f"  gen  ={gen} parsed={parse_answer(gen)}")
print("--- train examples ---")
for a, b in train_pairs[:3]:
    prompt, true_comp, ans = render(a, b)
    ids = torch.tensor([encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        full = model.generate(ids, torch.tensor([len(prompt)]), 28,
                              temperature=0.0)
    gen = decode(full[0, len(prompt):])
    print(f"prompt={prompt} truth={true_comp}")
    print(f"  gen  ={gen} parsed={parse_answer(gen)}")
