# ========================================================
# 04_ngram_nll_loss.py
# N-gram Language Model with NLLLoss
# ========================================================

"""
TUTORIAL 4: Understanding NLLLoss (Negative Log Likelihood Loss)

Learning Objectives:
- Understand what NLLLoss is
- Learn the relationship: CrossEntropyLoss = LogSoftmax + NLLLoss
- Modify model to output log probabilities
- Compare NLLLoss with CrossEntropyLoss

Estimated time: 15 minutes

What is NLLLoss?
----------------
NLLLoss (Negative Log Likelihood Loss) measures how well log probabilities
match the target. Unlike CrossEntropyLoss, it expects log probabilities
as input, not raw logits.

Key relationship:
  CrossEntropyLoss(logits, target) = NLLLoss(log_softmax(logits), target)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import word_embedding_ngram as ngr

print("=" * 70)
print("TUTORIAL 4: N-gram Model with NLLLoss")
print("=" * 70)

# ========================================================
# PART 1: Understanding the Relationship
# ========================================================

print("\n" + "=" * 70)
print("PART 1: CrossEntropyLoss vs NLLLoss")
print("=" * 70)

print("""
The Mathematical Relationship:
------------------------------

CrossEntropyLoss combines two operations:
1. LogSoftmax: Converts logits to log probabilities
2. NLLLoss: Computes negative log likelihood

Formula:
  CrossEntropyLoss(x, y) = NLLLoss(LogSoftmax(x), y)

Why this matters:
- CrossEntropyLoss: Input = logits (raw scores)
- NLLLoss: Input = log probabilities

Let's verify this mathematically!
""")

# Demonstration
print("\n--- Demonstration ---")

# Create example data
logits = torch.tensor([[2.0, 5.0, 1.0]])  # Raw scores
target = torch.tensor([1])  # Index 1 is correct

print(f"Logits (raw scores): {logits[0]}")
print(f"Target: {target.item()}")

# Method 1: CrossEntropyLoss directly
ce_loss = F.cross_entropy(logits, target)
print(f"\nMethod 1 - CrossEntropyLoss directly:")
print(f"  Loss: {ce_loss:.6f}")

# Method 2: LogSoftmax + NLLLoss
log_probs = F.log_softmax(logits, dim=1)
nll_loss = F.nll_loss(log_probs, target)
print(f"\nMethod 2 - LogSoftmax + NLLLoss:")
print(f"  Log probabilities: {log_probs[0]}")
print(f"  Loss: {nll_loss:.6f}")

# Verify they're the same
print(f"\nAre they equal? {torch.allclose(ce_loss, nll_loss)}")
print("Yes! CrossEntropyLoss = LogSoftmax + NLLLoss")

# Show the probabilities
probs = torch.softmax(logits, dim=1)
print(f"\nFor reference:")
print(f"  Softmax probabilities: {probs[0]}")
print(f"  Log softmax: {log_probs[0]}")
print(f"  Log(prob[target]): {log_probs[0, target.item()]:.6f}")
print(f"  Negative log likelihood: {-log_probs[0, target.item()]:.6f}")

# ========================================================
# PART 2: Modified Model for NLLLoss
# ========================================================

print("\n" + "=" * 70)
print("PART 2: Modifying the Model for NLLLoss")
print("=" * 70)

print("""
To use NLLLoss, we need to modify our model's forward pass
to output log probabilities instead of raw logits.

Changes needed:
1. Add LogSoftmax as the final layer
2. Use NLLLoss instead of CrossEntropyLoss

Let's create a modified model!
""")


class NGramLanguageModelerNLL(nn.Module):
    """
    N-gram model that outputs log probabilities for use with NLLLoss.
    
    The only difference from the original model is the addition of
    log_softmax in the forward pass.
    """
    
    def __init__(self, vocab_size=None, embedding_dim=None, context_size=None):
        super(NGramLanguageModelerNLL, self).__init__()
        
        if vocab_size is None:
            vocab_size = ngr.ARGS.vocab_size
        if embedding_dim is None:
            embedding_dim = ngr.ARGS.embedding_dim
        if context_size is None:
            context_size = ngr.ARGS.context_size
        
        # Same architecture as before
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        # Same forward pass as before...
        embeds = self.embeddings(inputs)
        embeds = embeds.view((embeds.shape[0], -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        
        # KEY DIFFERENCE: Apply log_softmax before returning
        # This converts logits to log probabilities
        log_probs = F.log_softmax(out, dim=1)
        return log_probs  # Return log probabilities, not logits


print("\nModified model created!")
print("Key change: forward() returns log probabilities using F.log_softmax()")

# ========================================================
# PART 3: Training with NLLLoss
# ========================================================

print("\n" + "=" * 70)
print("PART 3: Training with NLLLoss")
print("=" * 70)

# Create modified model
model_nll = NGramLanguageModelerNLL()
loss_function = nn.NLLLoss()  # Use NLLLoss this time
optimizer = optim.SGD(model_nll.parameters(), lr=ngr.ARGS.lr)

print("\nTraining configuration:")
print(f"  Model: NGramLanguageModelerNLL (outputs log probabilities)")
print(f"  Loss function: nn.NLLLoss")
print(f"  Optimizer: SGD")
print(f"  Learning rate: {ngr.ARGS.lr}")
print(f"  Epochs: {ngr.ARGS.epochs}")

print("\nStarting training...\n")

# Train the model
losses_nll = ngr.train(model_nll, loss_function, optimizer, epochs=ngr.ARGS.epochs, verbose=True)

# ========================================================
# PART 4: Comparing All Three Approaches
# ========================================================

print("\n" + "=" * 70)
print("PART 4: Comparing All Three Loss Functions")
print("=" * 70)

print("\nTraining comparison models...")

# Train with CrossEntropyLoss (Module API)
print("Training model 1: nn.CrossEntropyLoss...")
model_ce = ngr.NGramLanguageModeler()
loss_ce = nn.CrossEntropyLoss()
optimizer_ce = optim.SGD(model_ce.parameters(), lr=ngr.ARGS.lr)
losses_ce = ngr.train(model_ce, loss_ce, optimizer_ce, epochs=ngr.ARGS.epochs, verbose=False)

# Train with F.cross_entropy (Functional API)
print("Training model 2: F.cross_entropy...")
model_func = ngr.NGramLanguageModeler()
optimizer_func = optim.SGD(model_func.parameters(), lr=ngr.ARGS.lr)
losses_func = ngr.train(model_func, F.cross_entropy, optimizer_func, epochs=ngr.ARGS.epochs, verbose=False)

# Plot all three
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(losses_ce, label='nn.CrossEntropyLoss', linewidth=2, alpha=0.8)
ax.plot(losses_func, label='F.cross_entropy', linewidth=2, alpha=0.8)
ax.plot(losses_nll, label='nn.NLLLoss (with LogSoftmax)', linewidth=2, alpha=0.8)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Comparing Three Loss Function Approaches', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "-" * 70)
print("Final losses:")
print(f"  nn.CrossEntropyLoss:        {losses_ce[-1]:.6f}")
print(f"  F.cross_entropy:            {losses_func[-1]:.6f}")
print(f"  nn.NLLLoss (+ LogSoftmax):  {losses_nll[-1]:.6f}")
print("-" * 70)

print("\nAll three approaches give similar results!")
print("Small differences are due to random weight initialization.")

# ========================================================
# PART 5: When to Use Each
# ========================================================

print("\n" + "=" * 70)
print("PART 5: When to Use Each Loss Function")
print("=" * 70)

print("""
Decision Guide:
--------------

Use CrossEntropyLoss when:
✓ Your model outputs raw logits (no softmax)
✓ You want PyTorch to handle everything automatically
✓ Most common choice for classification
✓ Recommended for beginners

Use NLLLoss when:
✓ You need log probabilities for other purposes
✓ Your model already applies log_softmax
✓ You want more control over the probability computation
✓ Working with probability distributions directly

Code Comparison:
---------------

With CrossEntropyLoss:
```python
def forward(self, x):
    logits = self.layers(x)
    return logits  # No activation
    
loss = nn.CrossEntropyLoss()
output = model(input)
loss_value = loss(output, target)  # PyTorch applies softmax internally
```

With NLLLoss:
```python
def forward(self, x):
    logits = self.layers(x)
    return F.log_softmax(logits, dim=1)  # Apply log_softmax
    
loss = nn.NLLLoss()
output = model(input)  # Already log probabilities
loss_value = loss(output, target)  # Directly compute NLL
```

Key Insight:
-----------
Both approaches are equivalent:
  CrossEntropyLoss(logits, y) = NLLLoss(log_softmax(logits), y)

The choice is about where you apply the log_softmax:
- CrossEntropyLoss: Applied inside the loss function
- NLLLoss: Applied in your model's forward pass
""")

# ========================================================
# PART 6: Numerical Stability
# ========================================================

print("\n" + "=" * 70)
print("PART 6: Why LogSoftmax? (Numerical Stability)")
print("=" * 70)

print("""
Why use log_softmax instead of log(softmax)?
-------------------------------------------

log_softmax is numerically more stable!

Naive approach (unstable):
  probs = softmax(logits)
  log_probs = log(probs)  # Can have numerical issues!

Better approach (stable):
  log_probs = log_softmax(logits)  # Mathematically optimized

Example issue with naive approach:
  softmax([1000, 1001]) → [~0, ~1]  
  log(~0) → -inf (problem!)
  
log_softmax handles this correctly using mathematical tricks.

Lesson: Always use log_softmax, never log(softmax)!
""")

# Demonstrate the instability
print("\n--- Demonstration of Numerical Issues ---")

large_logits = torch.tensor([[100.0, 101.0, 99.0]])

# Naive approach (can have issues)
probs_naive = F.softmax(large_logits, dim=1)
print(f"Softmax of large logits: {probs_naive[0]}")
print(f"Notice the very small probability: {probs_naive[0, 2]:.2e}")

log_probs_naive = torch.log(probs_naive)
print(f"Log of softmax: {log_probs_naive[0]}")
print(f"See the very negative number? {log_probs_naive[0, 2]:.2f}")

# Proper approach (stable)
log_probs_stable = F.log_softmax(large_logits, dim=1)
print(f"\nUsing log_softmax directly: {log_probs_stable[0]}")
print("Much more stable and accurate!")

# ========================================================
# KEY TAKEAWAYS
# ========================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Mathematical relationship:
   CrossEntropyLoss = LogSoftmax + NLLLoss
   
2. Input requirements:
   - CrossEntropyLoss: Expects logits (raw scores)
   - NLLLoss: Expects log probabilities
   
3. When to use each:
   - CrossEntropyLoss: Default choice, easiest to use
   - NLLLoss: When you need log probabilities elsewhere
   
4. Numerical stability:
   - Always use log_softmax, never log(softmax)
   - PyTorch functions are optimized for stability
   
5. All approaches give same results:
   - Choose based on convenience and code structure
   - Performance is equivalent
   
6. Best practices:
   - For classification: Use CrossEntropyLoss
   - Keep model and loss consistent
   - Don't apply softmax in model if using CrossEntropyLoss

Congratulations!
---------------
You've completed all basic tutorials and understand:
✓ Word embeddings
✓ N-gram language models
✓ Three different loss function approaches
✓ The relationship between them

Ready for intermediate tutorials? Go to 02_intermediate/!
""")

print("=" * 70)
print("END OF TUTORIAL 4")
print("=" * 70)
