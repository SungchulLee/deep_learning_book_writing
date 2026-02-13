#!/usr/bin/env python3
"""
==============================================================================
EXERCISE 02: Implement Mini-Batch Training
==============================================================================

OBJECTIVE:
Implement mini-batch training instead of full-batch training.
Learn why mini-batches are crucial for deep learning.

DIFFICULTY: ⭐⭐ (Intermediate)

REQUIREMENTS:
- Complete beginner tutorials
- Understand DataLoader concept
- Basic understanding of batching

WHAT YOU'LL LEARN:
- Why mini-batches are used in practice
- How to split data into batches
- Memory vs. convergence trade-offs
- Batch size effect on training

TIME: 30-45 minutes

==============================================================================
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def generate_data(n_samples=1000):
    """Generate synthetic nonlinear data."""
    torch.manual_seed(42)
    x = torch.randn(n_samples, 1)
    y = 2 * x ** 2 + 3 * x + 1 + 0.5 * torch.randn(n_samples, 1)
    return x, y


def create_mini_batches(x, y, batch_size):
    """
    TODO: Create mini-batches from the data.
    
    Args:
        x: Input tensor of shape (n_samples, n_features)
        y: Target tensor of shape (n_samples, 1)
        batch_size: Number of samples per batch
    
    Returns:
        List of (x_batch, y_batch) tuples
    
    HINTS:
    - Shuffle the data first (get random indices)
    - Split into chunks of size batch_size
    - Use torch.randperm() for shuffling
    - The last batch might be smaller than batch_size
    
    EXAMPLE:
        x = torch.tensor([[1], [2], [3], [4], [5]])
        y = torch.tensor([[10], [20], [30], [40], [50]])
        batches = create_mini_batches(x, y, batch_size=2)
        # Returns: [
        #   (tensor([[3], [1]]), tensor([[30], [10]])),  # Random batch 1
        #   (tensor([[5], [2]]), tensor([[50], [20]])),  # Random batch 2
        #   (tensor([[4]]), tensor([[40]]))              # Last batch (size 1)
        # ]
    """
    # TODO: Your code here
    # 1. Get number of samples
    # 2. Generate random permutation of indices
    # 3. Loop through indices in steps of batch_size
    # 4. Create batches using slicing
    # 5. Return list of (x_batch, y_batch) tuples
    
    batches = []
    # Your implementation here
    
    return batches


def train_with_mini_batches(x, y, batch_size=32, epochs=50, lr=0.01):
    """
    TODO: Implement training loop with mini-batches.
    
    Model: y = w2*x^2 + w1*x + b  (quadratic regression)
    
    Your training loop should:
    1. Create mini-batches for each epoch
    2. For each batch:
        - Forward pass on batch
        - Compute loss on batch
        - Backward pass
        - Update parameters
    3. Track average loss per epoch
    """
    # Initialize parameters
    w2 = torch.randn(1, requires_grad=True)
    w1 = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    epoch_losses = []
    
    print(f"\nTraining with batch_size={batch_size}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # TODO: Create mini-batches for this epoch
        # batches = create_mini_batches(x, y, batch_size)
        
        batch_losses = []
        
        # TODO: Iterate over batches
        # for x_batch, y_batch in batches:
        #     # Forward pass
        #     # y_pred = ...
        #     # loss = ...
        #     
        #     # Backward pass
        #     # Zero gradients
        #     # loss.backward()
        #     
        #     # Update parameters
        #     # with torch.no_grad():
        #     #     w2 -= lr * w2.grad
        #     #     ...
        #     
        #     batch_losses.append(loss.item())
        
        # Average loss for this epoch
        epoch_loss = np.mean(batch_losses) if batch_losses else 0
        epoch_losses.append(epoch_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {epoch_loss:.6f}")
    
    print(f"\nFinal parameters:")
    print(f"  w2 = {w2.item():.4f} (true: 2.0)")
    print(f"  w1 = {w1.item():.4f} (true: 3.0)")
    print(f"  b  = {b.item():.4f} (true: 1.0)")
    
    return epoch_losses, (w2, w1, b)


def compare_batch_sizes():
    """
    TODO: Compare different batch sizes.
    
    Try batch sizes: [1, 16, 64, 256, 1000]
    Plot loss curves for comparison.
    
    Questions to answer:
    - Which converges fastest (fewest epochs)?
    - Which is most stable (smoothest curve)?
    - What are the trade-offs?
    """
    x, y = generate_data(n_samples=1000)
    
    batch_sizes = [1, 16, 64, 256, 1000]
    
    plt.figure(figsize=(12, 6))
    
    # TODO: Train with each batch size and plot results
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Effect of Batch Size on Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('/home/claude/exercise_02_batch_comparison.png', dpi=150)
    print("\nComparison plot saved!")


# ==============================================================================
# BONUS CHALLENGES
# ==============================================================================

def bonus_challenge_1():
    """
    BONUS 1: Implement a simple DataLoader
    
    Create a PyTorchDataset class and simple data loader.
    Compare with your manual batching.
    """
    pass


def bonus_challenge_2():
    """
    BONUS 2: Visualize batch variance
    
    For each batch size, plot:
    - Mean gradient magnitude
    - Gradient variance across batches
    - Update step sizes
    
    Understand the noise in gradient estimates!
    """
    pass


# ==============================================================================
# UNDERSTANDING MINI-BATCHES
# ==============================================================================

EXPLANATION = """
WHY MINI-BATCHES?

1. MEMORY EFFICIENCY
   - Full batch: Requires all data in GPU memory
   - Mini-batch: Only batch_size samples needed
   - Enables training on large datasets

2. FASTER CONVERGENCE
   - Full batch: 1 update per epoch
   - Mini-batch: n_samples/batch_size updates per epoch
   - More frequent updates → faster learning

3. BETTER GENERALIZATION
   - Noise in gradient estimates acts as regularization
   - Helps escape sharp minima
   - Leads to better test performance

4. COMPUTATIONAL EFFICIENCY
   - GPUs are optimized for batch operations
   - Batch sizes 32-256 often optimal
   - Parallelization benefits

BATCH SIZE TRADE-OFFS:

Small batches (1-32):
  ✓ More updates per epoch
  ✓ Better generalization
  ✗ Noisy gradients
  ✗ Slower per-batch computation

Large batches (256-1024):
  ✓ Stable gradients
  ✓ Faster per-batch computation
  ✗ Fewer updates per epoch
  ✗ May overfit to training data

PRACTICAL TIPS:
- Start with batch_size=32 or 64
- Increase if you have GPU memory
- Decrease if you see overfitting
- Try learning rate warmup for large batches
"""


# ==============================================================================
# SOLUTION HINTS
# ==============================================================================

"""
HINT 1: Creating batches
    n = x.shape[0]
    indices = torch.randperm(n)
    for i in range(0, n, batch_size):
        batch_idx = indices[i:i+batch_size]
        x_batch = x[batch_idx]
        y_batch = y[batch_idx]
        batches.append((x_batch, y_batch))

HINT 2: Model forward pass
    y_pred = w2 * x_batch**2 + w1 * x_batch + b

HINT 3: Don't forget to zero gradients before each backward pass!

COMPLETE SOLUTION: See solution_02_mini_batch.py
"""


if __name__ == "__main__":
    print(__doc__)
    print(EXPLANATION)
    
    print("\n" + "="*70)
    print("YOUR TASKS:")
    print("="*70)
    print("""
    1. Implement create_mini_batches() function
    2. Complete train_with_mini_batches() function
    3. Run and observe training with different batch sizes
    4. Implement compare_batch_sizes() to visualize differences
    5. (Optional) Attempt bonus challenges
    
    Start here:
    -----------
    x, y = generate_data(n_samples=1000)
    losses, params = train_with_mini_batches(x, y, batch_size=32)
    """)
    
    print("\n💡 TIP: Test create_mini_batches() first with small example")
    print("💡 TIP: Print batch shapes to debug")
    print("💡 TIP: Start with batch_size=32, then try 1 and 256")
