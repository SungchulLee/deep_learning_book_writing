"""
================================================================================
Level 2 - Example 1: Batch, Mini-batch, and Stochastic Gradient Descent
================================================================================

LEARNING OBJECTIVES:
- Understand different variants of gradient descent
- Compare batch GD, mini-batch GD, and SGD
- Learn about DataLoader and batching in PyTorch
- Understand trade-offs between variants

DIFFICULTY: ⭐⭐ Intermediate

TIME: 35-45 minutes

PREREQUISITES:
- Completed Level 1 examples
- Understanding of basic gradient descent

================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

print("="*80)
print("BATCH vs MINI-BATCH vs STOCHASTIC GRADIENT DESCENT")
print("="*80)

# ============================================================================
# PART 1: Understanding the Variants
# ============================================================================
print("\n" + "="*80)
print("PART 1: WHAT ARE THE VARIANTS?")
print("="*80)

print("""
GRADIENT DESCENT VARIANTS:
--------------------------

1. BATCH GRADIENT DESCENT (Batch GD)
   - Uses ALL training data in each iteration
   - Computes gradient over entire dataset
   - Most accurate gradient, but slow for large datasets
   
   Pseudocode:
   for epoch in epochs:
       gradient = compute_gradient(all_data)
       parameters -= lr * gradient

2. STOCHASTIC GRADIENT DESCENT (SGD)
   - Uses ONE random sample per iteration
   - Much faster per iteration
   - Noisy gradients, but can escape local minima
   
   Pseudocode:
   for epoch in epochs:
       for sample in shuffle(data):
           gradient = compute_gradient(sample)
           parameters -= lr * gradient

3. MINI-BATCH GRADIENT DESCENT (Mini-batch GD)
   - Uses SMALL BATCHES of data
   - Balance between Batch GD and SGD
   - Most common in practice (e.g., batch_size=32, 64, 128)
   
   Pseudocode:
   for epoch in epochs:
       for batch in create_batches(data, batch_size):
           gradient = compute_gradient(batch)
           parameters -= lr * gradient
""")

# ============================================================================
# PART 2: Create Dataset
# ============================================================================
print("\n" + "="*80)
print("PART 2: DATASET PREPARATION")
print("="*80)

torch.manual_seed(42)
np.random.seed(42)

# Larger dataset to see differences
n_samples = 1000
X_numpy = np.random.randn(n_samples, 1) * 2
y_numpy = 3 * X_numpy + 2 + np.random.randn(n_samples, 1) * 0.5

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"Dataset: {n_samples} samples")
print(f"True relationship: y = 3x + 2")

# ============================================================================
# PART 3: Define Model
# ============================================================================

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# ============================================================================
# PART 4: Training Functions for Each Variant
# ============================================================================

def train_batch_gd(X, y, n_epochs, lr):
    """Batch Gradient Descent - Use all data at once"""
    model = SimpleLinearModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Use ALL data in one forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
    
    training_time = time.time() - start_time
    return model, loss_history, training_time


def train_sgd(X, y, n_epochs, lr):
    """Stochastic Gradient Descent - One sample at a time"""
    model = SimpleLinearModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    loss_history = []
    start_time = time.time()
    n_samples = X.shape[0]
    
    for epoch in range(n_epochs):
        # Shuffle data each epoch
        indices = torch.randperm(n_samples)
        
        # Process one sample at a time
        epoch_losses = []
        for i in indices:
            X_sample = X[i:i+1]  # Keep dimension (1, 1)
            y_sample = y[i:i+1]
            
            y_pred = model(X_sample)
            loss = criterion(y_pred, y_sample)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Record average loss for the epoch
        loss_history.append(np.mean(epoch_losses))
    
    training_time = time.time() - start_time
    return model, loss_history, training_time


def train_minibatch_gd(X, y, n_epochs, lr, batch_size):
    """Mini-batch Gradient Descent - Use small batches"""
    model = SimpleLinearModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Create DataLoader for automatic batching
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Iterate over mini-batches
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Record average loss for the epoch
        loss_history.append(np.mean(epoch_losses))
    
    training_time = time.time() - start_time
    return model, loss_history, training_time

# ============================================================================
# PART 5: Train with All Three Variants
# ============================================================================
print("\n" + "="*80)
print("PART 5: COMPARING ALL VARIANTS")
print("="*80)

n_epochs = 50
learning_rate = 0.01
batch_size = 32

print("Training with different variants...")
print(f"Epochs: {n_epochs}, Learning rate: {learning_rate}\n")

# Train Batch GD
print("1. Training with Batch GD (all 1000 samples at once)...")
model_batch, loss_batch, time_batch = train_batch_gd(X, y, n_epochs, learning_rate)
print(f"   ✓ Completed in {time_batch:.3f} seconds")

# Train SGD
print("2. Training with SGD (1 sample at a time)...")
model_sgd, loss_sgd, time_sgd = train_sgd(X, y, n_epochs, learning_rate)
print(f"   ✓ Completed in {time_sgd:.3f} seconds")

# Train Mini-batch GD
print(f"3. Training with Mini-batch GD (batch_size={batch_size})...")
model_minibatch, loss_minibatch, time_minibatch = train_minibatch_gd(
    X, y, n_epochs, learning_rate, batch_size
)
print(f"   ✓ Completed in {time_minibatch:.3f} seconds")

# ============================================================================
# PART 6: Compare Results
# ============================================================================
print("\n" + "="*80)
print("PART 6: COMPARISON RESULTS")
print("="*80)

print("\nTraining Time Comparison:")
print(f"  Batch GD:      {time_batch:.3f}s")
print(f"  SGD:           {time_sgd:.3f}s")
print(f"  Mini-batch GD: {time_minibatch:.3f}s")

print("\nFinal Loss:")
print(f"  Batch GD:      {loss_batch[-1]:.6f}")
print(f"  SGD:           {loss_sgd[-1]:.6f}")
print(f"  Mini-batch GD: {loss_minibatch[-1]:.6f}")

print("\nLearned Parameters:")
for name, model in [("Batch", model_batch), ("SGD", model_sgd), ("Mini-batch", model_minibatch)]:
    w = model.linear.weight.item()
    b = model.linear.bias.item()
    print(f"  {name:11s}: w={w:.4f}, b={b:.4f}")
print(f"  {'True':11s}: w=3.0000, b=2.0000")

# ============================================================================
# PART 7: Visualization
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss curves (linear scale)
axes[0, 0].plot(loss_batch, label='Batch GD', linewidth=2)
axes[0, 0].plot(loss_sgd, label='SGD', linewidth=2, alpha=0.7)
axes[0, 0].plot(loss_minibatch, label='Mini-batch GD', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Convergence (Linear Scale)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Loss curves (log scale)
axes[0, 1].plot(loss_batch, label='Batch GD', linewidth=2)
axes[0, 1].plot(loss_sgd, label='SGD', linewidth=2, alpha=0.7)
axes[0, 1].plot(loss_minibatch, label='Mini-batch GD', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Loss Convergence (Log Scale)')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Training time comparison
methods = ['Batch GD', 'SGD', 'Mini-batch']
times = [time_batch, time_sgd, time_minibatch]
colors = ['blue', 'orange', 'green']
axes[1, 0].bar(methods, times, color=colors, alpha=0.7)
axes[1, 0].set_ylabel('Training Time (seconds)')
axes[1, 0].set_title('Training Time Comparison')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Loss variance (smoothness)
axes[1, 1].plot(loss_sgd[-20:], 'o-', label='SGD (last 20 epochs)', linewidth=2)
axes[1, 1].plot(loss_minibatch[-20:], 's-', label='Mini-batch (last 20 epochs)', linewidth=2)
axes[1, 1].plot(loss_batch[-20:], '^-', label='Batch (last 20 epochs)', linewidth=2)
axes[1, 1].set_xlabel('Last 20 Epochs')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Loss Stability (Zoomed In)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_2_intermediate/batch_comparison.png', dpi=150)
print("\n✓ Plot saved as 'batch_comparison.png'")
print("\nClose the plot window to continue...")
plt.show()

# ============================================================================
# PART 8: Key Takeaways
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. BATCH GD:
   ✓ Most stable, smooth convergence
   ✗ Slow for large datasets
   ✗ High memory usage
   ✗ Can get stuck in local minima
   
2. STOCHASTIC GD:
   ✓ Fast per-epoch training
   ✓ Can escape shallow local minima
   ✗ Noisy gradients, erratic convergence
   ✗ May not reach exact minimum
   
3. MINI-BATCH GD: (BEST OF BOTH WORLDS!)
   ✓ Good balance of speed and stability
   ✓ Can leverage GPU parallelization
   ✓ Generalizes well
   → Most commonly used in practice!

BATCH SIZE SELECTION:
- Too small (e.g., 1-8): Noisy, slow to converge
- Too large (e.g., >512): Memory issues, poor generalization
- Sweet spot: 16, 32, 64, 128 (power of 2 for GPU efficiency)

PRACTICAL RECOMMENDATIONS:
• Use mini-batch GD with batch_size=32 or 64 as default
• Increase batch size if you have memory and need speed
• Decrease batch size if overfitting or for better generalization
• Try different batch sizes - it's a hyperparameter!
""")

print("="*80)
print("EXPERIMENTS TO TRY")
print("="*80)
print("""
1. Different batch sizes: 8, 16, 64, 128, 256
   - How does it affect convergence?
   - What about training time?

2. Larger dataset (10,000 samples)
   - Which variant is fastest now?

3. Different learning rates for each variant
   - SGD might need smaller lr
   - Batch GD might work with larger lr

4. Add momentum (next example!)
   - optimizer = torch.optim.SGD(..., momentum=0.9)
""")
print("="*80)
