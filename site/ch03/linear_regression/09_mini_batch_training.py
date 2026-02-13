"""
==============================================================================
09_mini_batch_training.py
==============================================================================
DIFFICULTY: ⭐⭐⭐⭐ (Advanced)

DESCRIPTION:
    Mini-batch gradient descent using PyTorch DataLoader.
    Efficient training with batches instead of full dataset.

TOPICS COVERED:
    - Dataset and DataLoader classes
    - Mini-batch gradient descent
    - Batch size effects
    - Shuffling and sampling
    - Training efficiency

PREREQUISITES:
    - Tutorial 05 (nn.Module)

TIME: ~25 minutes
==============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time

print("=" * 70)
print("MINI-BATCH TRAINING WITH DATALOADER")
print("=" * 70)

# ============================================================================
# PART 1: GENERATE LARGE DATASET
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE LARGE DATASET")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)

# Generate large dataset
n_samples = 10000
n_features = 50

X = torch.randn(n_samples, n_features)
true_weights = torch.randn(n_features, 1)
y = X @ true_weights + torch.randn(n_samples, 1) * 0.5

print(f"Dataset created:")
print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# ============================================================================
# PART 2: CREATE CUSTOM DATASET CLASS
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: CUSTOM DATASET CLASS")
print("=" * 70)

class RegressionDataset(Dataset):
    """Custom Dataset class for regression data"""
    
    def __init__(self, X, y):
        """
        Args:
            X: Feature tensor
            y: Target tensor
        """
        self.X = X
        self.y = y
        
    def __len__(self):
        """Return the number of samples"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Return a single sample
        
        Args:
            idx: Sample index
        
        Returns:
            tuple: (features, target)
        """
        return self.X[idx], self.y[idx]

# Create dataset
dataset = RegressionDataset(X, y)
print(f"Dataset created with {len(dataset)} samples")

# Test dataset
sample_x, sample_y = dataset[0]
print(f"\nFirst sample:")
print(f"  X shape: {sample_x.shape}")
print(f"  y value: {sample_y.item():.4f}")

# Alternative: Use TensorDataset (simpler for basic cases)
tensor_dataset = TensorDataset(X, y)
print(f"\nTensorDataset created (alternative approach)")

# ============================================================================
# PART 3: CREATE DATALOADER
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: CREATE DATALOADER")
print("=" * 70)

batch_size = 128  # Number of samples per batch

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,      # Shuffle data each epoch
    num_workers=0,     # Number of subprocesses for data loading
    drop_last=False    # Keep last incomplete batch
)

print(f"DataLoader created:")
print(f"  Batch size: {batch_size}")
print(f"  Number of batches: {len(dataloader)}")
print(f"  Shuffle: True")

# Demonstrate batching
print(f"\nIterating through first batch:")
for batch_X, batch_y in dataloader:
    print(f"  Batch X shape: {batch_X.shape}")
    print(f"  Batch y shape: {batch_y.shape}")
    break

# ============================================================================
# PART 4: COMPARE TRAINING METHODS
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: COMPARE BATCH SIZES")
print("=" * 70)

class SimpleModel(nn.Module):
    def __init__(self, n_features):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return self.linear(x)

def train_with_batch_size(batch_size, n_epochs=20):
    """Train model with specified batch size"""
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model and optimizer
    model = SimpleModel(n_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    losses = []
    epoch_times = []
    
    for epoch in range(n_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_X, batch_y in loader:
            # Forward pass
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        epoch_times.append(time.time() - start_time)
    
    return losses, epoch_times, model

# Train with different batch sizes
batch_sizes = [32, 128, 512, len(dataset)]  # Last one is full batch
results = {}

print(f"Training with different batch sizes...")
for bs in batch_sizes:
    print(f"\n  Batch size: {bs}")
    losses, times, model = train_with_batch_size(bs)
    results[bs] = {
        'losses': losses,
        'times': times,
        'final_loss': losses[-1],
        'avg_time': np.mean(times)
    }
    print(f"    Final loss: {losses[-1]:.6f}")
    print(f"    Avg time/epoch: {np.mean(times):.4f}s")

# ============================================================================
# PART 5: VISUALIZE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: VISUALIZE BATCH SIZE EFFECTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss curves
ax = axes[0, 0]
for bs in batch_sizes:
    label = f'Batch size: {bs}' if bs != len(dataset) else 'Full batch (GD)'
    ax.plot(results[bs]['losses'], label=label, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss vs Batch Size')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 2: Training time
ax = axes[0, 1]
batch_labels = [f'{bs}' if bs != len(dataset) else 'Full' for bs in batch_sizes]
times = [results[bs]['avg_time'] for bs in batch_sizes]
ax.bar(batch_labels, times)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Avg Time per Epoch (s)')
ax.set_title('Training Speed vs Batch Size')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Final loss comparison
ax = axes[1, 0]
final_losses = [results[bs]['final_loss'] for bs in batch_sizes]
ax.bar(batch_labels, final_losses)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Final Loss')
ax.set_title('Final Loss vs Batch Size')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Summary text
ax = axes[1, 1]
summary_text = """
BATCH SIZE EFFECTS:

Small Batches (32, 128):
✓ Faster iteration
✓ More gradient updates
✓ Noisier gradients
✓ Better generalization
✗ Less computational efficiency

Medium Batches (512):
✓ Good balance
✓ Stable training
✓ Efficient

Full Batch (GD):
✓ Most stable gradients
✓ Deterministic
✗ Slow updates
✗ Memory intensive
✗ May overfit

Recommendation:
- Start with 32-256
- Increase if memory allows
- Monitor validation loss
"""
ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/09_batch_size_comparison.png', dpi=100)
print("Saved visualization")
plt.show()

# ============================================================================
# PART 6: EFFICIENT TRAINING LOOP TEMPLATE
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: EFFICIENT TRAINING LOOP TEMPLATE")
print("=" * 70)

print("""
STANDARD PYTORCH TRAINING LOOP WITH DATALOADER:

# Setup
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
model = MyModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(n_epochs):
    model.train()  # Set to training mode
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluation (optional)
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)

Key Points:
1. DataLoader handles batching and shuffling
2. Each epoch iterates through all batches
3. One epoch = one pass through entire dataset
4. Gradients computed per batch, not entire dataset
""")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Mini-Batch Training:

1. DATASET CLASS:
   - Holds data (X, y)
   - Implements __len__ and __getitem__
   - Can add data augmentation, transforms

2. DATALOADER:
   - Batches data automatically
   - Shuffles each epoch
   - Parallel data loading (num_workers)
   - Handles last incomplete batch

3. BATCH SIZE TRADE-OFFS:
   - Smaller: Noisy but fast
   - Larger: Stable but slow
   - Sweet spot: 32-512
   - Powers of 2 recommended

4. BENEFITS:
   ✓ Memory efficient (don't load all data)
   ✓ Faster convergence (more updates)
   ✓ Better generalization
   ✓ Enables GPU parallelization

Next: Tutorial 10 - Complete production pipeline!
""")
