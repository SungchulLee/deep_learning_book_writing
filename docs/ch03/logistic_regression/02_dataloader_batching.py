"""
================================================================================
02_dataloader_batching.py - Efficient Batch Processing with DataLoader
================================================================================

LEARNING OBJECTIVES:
- Understand mini-batch gradient descent
- Use PyTorch DataLoader for efficient batching
- Learn about Dataset and DataLoader classes
- Handle shuffling and batch processing
- Scale to larger datasets

PREREQUISITES:
- Completed 01_proper_training_loop.py
- Understanding of gradient descent

TIME TO COMPLETE: ~1.5 hours

DIFFICULTY: ⭐⭐⭐☆☆ (Intermediate)
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple

print("="*80)
print("EFFICIENT BATCH PROCESSING WITH DATALOADER")
print("="*80)

# =============================================================================
# PART 1: UNDERSTANDING BATCHING
# =============================================================================

print("\n" + "="*80)
print("PART 1: WHY USE BATCHES?")
print("="*80)

print("""
Mini-Batch Gradient Descent:

Instead of processing all data at once (full batch) or one sample at a time
(stochastic), we process small batches:

Batch Size = 1 (SGD):
  ✓ Fast updates
  ✗ Noisy gradients
  ✗ Slow computation (no vectorization)

Batch Size = ALL (Full Batch GD):
  ✓ Stable gradients
  ✗ Memory intensive
  ✗ Slow for large datasets

Batch Size = 32-256 (Mini-Batch):
  ✓ Balance between speed and stability
  ✓ Efficient GPU utilization
  ✓ Regularization effect from noise
  ✓ Can process datasets larger than memory

Common batch sizes: 16, 32, 64, 128, 256
""")

# =============================================================================
# PART 2: PREPARE DATA
# =============================================================================

print("\n" + "="*80)
print("PART 2: PREPARING DATA")
print("="*80)

# Generate larger dataset
torch.manual_seed(42)
np.random.seed(42)

X, y = make_classification(
    n_samples=5000,  # Larger dataset
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

print(f"Dataset size: {X.shape[0]} samples")

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

print(f"Training: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")

# =============================================================================
# PART 3: CREATING DATALOADERS
# =============================================================================

print("\n" + "="*80)
print("PART 3: CREATING DATALOADERS")
print("="*80)

# Create datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

print("\nDataset created using TensorDataset")
print(f"Train dataset length: {len(train_dataset)}")
print(f"Each item shape: features={train_dataset[0][0].shape}, label={train_dataset[0][1].shape}")

# Create dataloaders
batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,      # Shuffle training data each epoch
    num_workers=0,     # 0 for Windows, 2-4 for Linux/Mac
    drop_last=False    # Keep last incomplete batch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,     # Don't shuffle test data
    num_workers=0
)

print(f"\nDataLoader created:")
print(f"Batch size: {batch_size}")
print(f"Number of batches (train): {len(train_loader)}")
print(f"Number of batches (test): {len(test_loader)}")
print(f"Samples per batch: {batch_size}")
print(f"Last batch size (train): {len(X_train) % batch_size if len(X_train) % batch_size != 0 else batch_size}")

# =============================================================================
# PART 4: MODEL DEFINITION
# =============================================================================

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegression(X_train.shape[1])

# =============================================================================
# PART 5: TRAINING WITH BATCHES
# =============================================================================

print("\n" + "="*80)
print("PART 5: TRAINING WITH MINI-BATCHES")
print("="*80)

def train_epoch_with_batches(model, dataloader, criterion, optimizer):
    """Train for one epoch using batches"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Iterate over batches
    for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
        # batch_X shape: (batch_size, n_features)
        # batch_y shape: (batch_size, 1)
        
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * len(batch_X)  # Accumulate loss
        predicted_classes = (predictions >= 0.5).float()
        correct += (predicted_classes == batch_y).sum().item()
        total += len(batch_X)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate_with_batches(model, dataloader, criterion):
    """Validate using batches"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item() * len(batch_X)
            predicted_classes = (predictions >= 0.5).float()
            correct += (predicted_classes == batch_y).sum().item()
            total += len(batch_X)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


# Training setup
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []
}

print(f"Training for {num_epochs} epochs with batch_size={batch_size}")
print("-" * 60)

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch_with_batches(
        model, train_loader, criterion, optimizer
    )
    
    # Validate
    test_loss, test_acc = validate_with_batches(
        model, test_loader, criterion
    )
    
    # Store history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

print("\nTraining completed!")

# =============================================================================
# PART 6: COMPARING BATCH SIZES
# =============================================================================

print("\n" + "="*80)
print("PART 6: COMPARING DIFFERENT BATCH SIZES")
print("="*80)

def train_with_batch_size(batch_size, num_epochs=30):
    """Train model with specific batch size"""
    # Create fresh model
    model = LogisticRegression(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    losses = []
    
    for epoch in range(num_epochs):
        train_loss, _ = train_epoch_with_batches(
            model, train_loader, criterion, optimizer
        )
        losses.append(train_loss)
    
    final_test_loss, final_test_acc = validate_with_batches(
        model, test_loader, criterion
    )
    
    return losses, final_test_acc

# Compare different batch sizes
batch_sizes = [16, 32, 64, 128, 256]
results = {}

print("Training with different batch sizes...")
for bs in batch_sizes:
    print(f"  Batch size {bs:3d}... ", end="", flush=True)
    losses, acc = train_with_batch_size(bs, num_epochs=30)
    results[bs] = {'losses': losses, 'accuracy': acc}
    print(f"Final accuracy: {acc:.4f}")

# =============================================================================
# PART 7: VISUALIZATION
# =============================================================================

print("\n" + "="*80)
print("PART 7: VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(15, 10))

# Plot 1: Training curves
ax1 = plt.subplot(2, 2, 1)
ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Curves (Batch Size = 64)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy curves
ax2 = plt.subplot(2, 2, 2)
ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
ax2.plot(history['test_acc'], label='Test Acc', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Curves', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Comparing batch sizes (loss curves)
ax3 = plt.subplot(2, 2, 3)
for bs in batch_sizes:
    ax3.plot(results[bs]['losses'], label=f'BS={bs}', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
ax3.set_title('Effect of Batch Size on Training', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Final accuracies by batch size
ax4 = plt.subplot(2, 2, 4)
accuracies = [results[bs]['accuracy'] for bs in batch_sizes]
bars = ax4.bar(range(len(batch_sizes)), accuracies, color='steelblue', alpha=0.7)
ax4.set_xticks(range(len(batch_sizes)))
ax4.set_xticklabels(batch_sizes)
ax4.set_xlabel('Batch Size')
ax4.set_ylabel('Final Test Accuracy')
ax4.set_title('Final Accuracy vs Batch Size', fontweight='bold')
ax4.set_ylim([min(accuracies)-0.01, max(accuracies)+0.01])
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("Visualizations created!")

# =============================================================================
# KEY TAKEAWAYS
# =============================================================================

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. DATALOADER BENEFITS
   ✓ Automatic batching
   ✓ Efficient memory usage
   ✓ Built-in shuffling
   ✓ Parallel data loading (num_workers)
   ✓ Can handle datasets larger than RAM

2. BATCH SIZE SELECTION
   ✓ Smaller batches: More noise, better regularization
   ✓ Larger batches: More stable, faster training
   ✓ Common choices: 32, 64, 128
   ✓ Limited by GPU memory

3. BEST PRACTICES
   ✓ Always shuffle training data
   ✓ Don't shuffle test/validation data
   ✓ Use TensorDataset for simple cases
   ✓ Create custom Dataset for complex data
   ✓ Set num_workers=0 on Windows

4. WHEN TO USE BATCHES
   ✓ Datasets > 10,000 samples
   ✓ Limited memory
   ✓ GPU training
   ✓ Want regularization effect
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. EASY: Try batch_size=1. How does it compare to mini-batch?

2. MEDIUM: Implement custom Dataset class:
   class CustomDataset(Dataset):
       def __init__(self, X, y):
           # Your code
       
       def __len__(self):
           # Your code
       
       def __getitem__(self, idx):
           # Your code

3. MEDIUM: Add data augmentation during batch loading

4. HARD: Implement dynamic batch sizing:
   - Start with small batches
   - Gradually increase during training

5. HARD: Compare training time for different batch sizes
   Use time.time() to measure
""")

print("\n" + "="*80)
print("NEXT: 03_model_checkpointing.py - Save and resume training")
print("="*80)
