# ========================================================
# 03_ngram_functional.py
# N-gram Language Model with F.cross_entropy (Functional API)
# ========================================================

"""
TUTORIAL 3: Using the Functional API (F.cross_entropy)

Learning Objectives:
- Understand the difference between nn.Module and functional APIs
- Use F.cross_entropy instead of nn.CrossEntropyLoss()
- Learn when to use each approach
- Compare results with previous tutorial

Estimated time: 10 minutes

Module vs Functional API:
-------------------------
PyTorch provides two ways to use most operations:

1. Module API (nn.CrossEntropyLoss):
   - Object-oriented
   - Can store state
   - Instantiate once, use many times

2. Functional API (F.cross_entropy):
   - Functional programming style
   - Stateless
   - Call directly when needed

Both compute the same thing - it's a matter of style!
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F  # Import functional API
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import word_embedding_ngram as ngr

print("=" * 70)
print("TUTORIAL 3: Using F.cross_entropy (Functional API)")
print("=" * 70)

# ========================================================
# PART 1: Module API vs Functional API
# ========================================================

print("\n" + "=" * 70)
print("PART 1: Comparing Module and Functional APIs")
print("=" * 70)

# Example to show they're equivalent
print("\n--- Simple Example ---")

# Create example data
logits = torch.tensor([[2.0, 5.0, 1.0],    # Predictions for sample 1
                       [1.0, 3.0, 2.0]])   # Predictions for sample 2
targets = torch.tensor([1, 2])              # Correct answers

print("Logits (model outputs):")
print(logits)
print("\nTargets (correct classes):")
print(targets)

# Method 1: Module API (what we used in Tutorial 2)
print("\n--- Method 1: Module API ---")
loss_module = nn.CrossEntropyLoss()  # Create loss object
loss_1 = loss_module(logits, targets)  # Use it
print(f"Loss using nn.CrossEntropyLoss: {loss_1:.6f}")

# Method 2: Functional API (what we'll use now)
print("\n--- Method 2: Functional API ---")
loss_2 = F.cross_entropy(logits, targets)  # Direct function call
print(f"Loss using F.cross_entropy: {loss_2:.6f}")

# Verify they're identical
print(f"\nAre they the same? {torch.allclose(loss_1, loss_2)}")
print("Yes! They compute exactly the same thing.")

print("\n" + "-" * 70)
print("When to use each:")
print("  • Module API (nn.CrossEntropyLoss):")
print("    - Training loops with consistent loss function")
print("    - When you might want to change loss parameters later")
print("    - More object-oriented style")
print("\n  • Functional API (F.cross_entropy):")
print("    - Quick one-off calculations")
print("    - When you want more concise code")
print("    - More functional programming style")
print("-" * 70)

# ========================================================
# PART 2: Training with Functional API
# ========================================================

print("\n" + "=" * 70)
print("PART 2: Training N-gram Model with F.cross_entropy")
print("=" * 70)

# Create model
model = ngr.NGramLanguageModeler()
optimizer = optim.SGD(model.parameters(), lr=ngr.ARGS.lr)

print("\nTraining configuration:")
print(f"  Loss function: F.cross_entropy (functional API)")
print(f"  Optimizer: SGD")
print(f"  Learning rate: {ngr.ARGS.lr}")
print(f"  Epochs: {ngr.ARGS.epochs}")

print("\nStarting training...")
print("(Notice we pass F.cross_entropy directly to the training function)\n")

# Train using functional API
# Note: We pass F.cross_entropy as a function, not an object!
losses = ngr.train(model, F.cross_entropy, optimizer, epochs=ngr.ARGS.epochs, verbose=True)

print("\nTraining complete!")

# ========================================================
# PART 3: Comparing Results
# ========================================================

print("\n" + "=" * 70)
print("PART 3: Comparing with Module API")
print("=" * 70)

print("\nLet's train another model with the Module API for comparison...")

# Train second model with Module API
model_2 = ngr.NGramLanguageModeler()
loss_function = nn.CrossEntropyLoss()  # Module API
optimizer_2 = optim.SGD(model_2.parameters(), lr=ngr.ARGS.lr)

print("Training with nn.CrossEntropyLoss...")
losses_2 = ngr.train(model_2, loss_function, optimizer_2, epochs=ngr.ARGS.epochs, verbose=False)

# Plot both
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Plot 1: Functional API
ax1.plot(losses, linewidth=2, color='blue')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('F.cross_entropy (Functional)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Module API
ax2.plot(losses_2, linewidth=2, color='green')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('nn.CrossEntropyLoss (Module)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "-" * 70)
print("Comparison of final losses:")
print(f"  F.cross_entropy:       {losses[-1]:.6f}")
print(f"  nn.CrossEntropyLoss:   {losses_2[-1]:.6f}")
print(f"  Difference:            {abs(losses[-1] - losses_2[-1]):.6f}")
print("-" * 70)

print("\nThe results are very similar (small differences due to random initialization).")
print("Both methods train the model equally well!")

# ========================================================
# PART 4: Code Comparison
# ========================================================

print("\n" + "=" * 70)
print("PART 4: Code Style Comparison")
print("=" * 70)

print("""
--- Module API Style ---
```python
# Create loss object once
loss_function = nn.CrossEntropyLoss()

# Use in training loop
for epoch in range(epochs):
    for batch in data:
        outputs = model(inputs)
        loss = loss_function(outputs, targets)  # Call object
        loss.backward()
        optimizer.step()
```

--- Functional API Style ---
```python
# No need to create object

# Use directly in training loop
for epoch in range(epochs):
    for batch in data:
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)  # Direct function call
        loss.backward()
        optimizer.step()
```

Key differences:
1. Functional: One less line (no instantiation)
2. Functional: Direct function call
3. Module: Can store parameters in the loss object if needed
""")

# ========================================================
# PART 5: When Each API Shines
# ========================================================

print("\n" + "=" * 70)
print("PART 5: When to Use Each API")
print("=" * 70)

print("""
Use Module API (nn.CrossEntropyLoss) when:
-------------------------------------------
✓ You need to configure the loss function
  Example: nn.CrossEntropyLoss(weight=class_weights)
  
✓ You're building a model class that includes the loss
  Example: class MyModel(nn.Module):
               def __init__(self):
                   self.loss = nn.CrossEntropyLoss()
                   
✓ You want to keep track of loss function state

Use Functional API (F.cross_entropy) when:
------------------------------------------
✓ You want cleaner, more concise code

✓ You don't need to configure loss parameters

✓ You prefer functional programming style

✓ You're computing loss once or in simple scenarios

Bottom line: Both are fine! Choose based on your preference and use case.
""")

# ========================================================
# PART 6: Testing the Model
# ========================================================

print("\n" + "=" * 70)
print("PART 6: Testing the Trained Model")
print("=" * 70)

# Make predictions
test_contexts = [["thy", "beauty's"], ["deep", "sunken"]]

print("\nPredictions from the functional API model:")
ix_to_word = {idx: word for word, idx in ngr.ARGS.word_to_ix.items()}

for context in test_contexts:
    if all(word in ngr.ARGS.word_to_ix for word in context):
        context_idxs = ngr.prepare_sequence(context, ngr.ARGS.word_to_ix)
        
        with torch.no_grad():
            logits = model(context_idxs.unsqueeze(0))
            predicted_idx = torch.argmax(logits, dim=1).item()
            predicted_word = ix_to_word[predicted_idx]
        
        print(f"\nContext: {context} → Predicted: {predicted_word}")

# ========================================================
# KEY TAKEAWAYS
# ========================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Module API vs Functional API:
   - Compute the same results
   - Different programming styles
   - Choose based on your needs and preferences

2. F.cross_entropy benefits:
   - More concise code
   - No object instantiation needed
   - Direct function call

3. nn.CrossEntropyLoss benefits:
   - Can store parameters
   - More object-oriented
   - Clearer when building complex models

4. In practice:
   - Both are widely used
   - PyTorch supports both styles
   - Pick what makes your code cleaner

5. Performance:
   - No performance difference
   - Both are equally efficient
   - Both are well-optimized in PyTorch

What's Next?
-----------
In the next tutorial, we'll explore NLLLoss and understand
the relationship between CrossEntropyLoss and NLLLoss!
""")

print("=" * 70)
print("END OF TUTORIAL 3")
print("=" * 70)

# ========================================================
# EXERCISE (Optional)
# ========================================================

print("\n" + "=" * 70)
print("OPTIONAL EXERCISES")
print("=" * 70)

print("""
1. Implement your own training loop:
   - Use F.cross_entropy directly
   - Print loss every 10 epochs
   - Compare with the provided train() function

2. Try other functional losses:
   - F.nll_loss (we'll cover this next!)
   - F.mse_loss (for regression)
   - Look up PyTorch docs for more

3. Benchmark both APIs:
   - Time the training with each
   - Confirm they take similar time

4. Mixed usage:
   - Use nn.CrossEntropyLoss for training
   - Use F.cross_entropy for validation
   - See that results are consistent
""")
