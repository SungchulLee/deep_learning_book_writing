"""
================================================================================
03_model_checkpointing.py - Saving and Loading Models
================================================================================

LEARNING OBJECTIVES:
- Save and load model weights
- Implement checkpoint system
- Resume training from checkpoints
- Save complete training state
- Best practices for model persistence

TIME TO COMPLETE: ~1 hour
DIFFICULTY: ⭐⭐⭐☆☆ (Intermediate)
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json

print("="*80)
print("MODEL CHECKPOINTING AND PERSISTENCE")
print("="*80)

# =============================================================================
# SETUP
# =============================================================================

# Create checkpoint directory
checkpoint_dir = Path("/home/claude/pytorch_logistic_regression_tutorial/02_intermediate/checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Prepare data
torch.manual_seed(42)
np.random.seed(42)

X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

# =============================================================================
# MODEL
# =============================================================================

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# =============================================================================
# CHECKPOINT FUNCTIONS
# =============================================================================

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save complete training state
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy
        filepath: Where to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load checkpoint and restore state
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        
    Returns:
        Dictionary with training info
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    print(f"  Accuracy: {checkpoint['accuracy']:.4f}")
    
    return checkpoint


def save_best_model(model, filepath):
    """Save only model weights (for deployment)"""
    torch.save(model.state_dict(), filepath)
    print(f"✓ Best model saved to {filepath}")


# =============================================================================
# TRAINING WITH CHECKPOINTING
# =============================================================================

print("\n" + "="*80)
print("TRAINING WITH AUTOMATIC CHECKPOINTING")
print("="*80)

model = LogisticRegression(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
save_every = 10  # Save checkpoint every N epochs
best_accuracy = 0.0

print(f"Training for {num_epochs} epochs")
print(f"Saving checkpoints every {save_every} epochs")
print("-" * 60)

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(batch_X)
        predicted_classes = (predictions >= 0.5).float()
        correct += (predicted_classes == batch_y).sum().item()
        total += len(batch_X)
    
    avg_loss = train_loss / total
    accuracy = correct / total
    
    # Validation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            predicted_classes = (predictions >= 0.5).float()
            test_correct += (predicted_classes == batch_y).sum().item()
            test_total += len(batch_X)
    
    test_accuracy = test_correct / test_total
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} "
              f"Train Acc: {accuracy:.4f} "
              f"Test Acc: {test_accuracy:.4f}")
    
    # Save checkpoint every N epochs
    if (epoch + 1) % save_every == 0:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        save_checkpoint(model, optimizer, epoch, avg_loss, test_accuracy, checkpoint_path)
    
    # Save best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_path = checkpoint_dir / "best_model.pt"
        save_best_model(model, best_model_path)
        print(f"  → New best accuracy: {best_accuracy:.4f}")

print("\nTraining completed!")
print(f"Best test accuracy: {best_accuracy:.4f}")

# =============================================================================
# RESUMING TRAINING
# =============================================================================

print("\n" + "="*80)
print("DEMONSTRATING RESUME FROM CHECKPOINT")
print("="*80)

# Create new model and optimizer
new_model = LogisticRegression(X_train.shape[1])
new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

# Load checkpoint
checkpoint_path = checkpoint_dir / "checkpoint_epoch_20.pt"
checkpoint_info = load_checkpoint(checkpoint_path, new_model, new_optimizer)

# Continue training from this point
print(f"\nResuming training from epoch {checkpoint_info['epoch']+1}...")
resume_epochs = 10

for epoch in range(checkpoint_info['epoch'], checkpoint_info['epoch'] + resume_epochs):
    # Training loop (same as before)
    new_model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        predictions = new_model(batch_X)
        loss = criterion(predictions, batch_y)
        
        new_optimizer.zero_grad()
        loss.backward()
        new_optimizer.step()
        
        train_loss += loss.item() * len(batch_X)
        predicted_classes = (predictions >= 0.5).float()
        correct += (predicted_classes == batch_y).sum().item()
        total += len(batch_X)
    
    avg_loss = train_loss / total
    accuracy = correct / total
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f}")

print("\n✓ Successfully resumed and continued training!")

# =============================================================================
# LOADING FOR INFERENCE
# =============================================================================

print("\n" + "="*80)
print("LOADING MODEL FOR INFERENCE (DEPLOYMENT)")
print("="*80)

# Create clean model for deployment
deployment_model = LogisticRegression(X_train.shape[1])

# Load only the weights (not optimizer state)
best_model_path = checkpoint_dir / "best_model.pt"
deployment_model.load_state_dict(torch.load(best_model_path))
deployment_model.eval()

print("✓ Model loaded for inference")

# Test inference
with torch.no_grad():
    sample_input = X_test[:5]  # 5 test samples
    predictions = deployment_model(sample_input)
    predicted_classes = (predictions >= 0.5).float()
    
    print("\nInference test on 5 samples:")
    for i in range(5):
        print(f"  Sample {i+1}: Predicted={int(predicted_classes[i].item())}, "
              f"Probability={predictions[i].item():.4f}, "
              f"Actual={int(y_test[i].item())}")

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

print("\n" + "="*80)
print("CHECKPOINT MANAGEMENT")
print("="*80)

# List all checkpoints
checkpoints = sorted(checkpoint_dir.glob("*.pt"))
print(f"Found {len(checkpoints)} checkpoint files:")
for cp in checkpoints:
    size_kb = cp.stat().st_size / 1024
    print(f"  {cp.name:30s} ({size_kb:.1f} KB)")

# Clean up old checkpoints (keep only last 3)
def cleanup_old_checkpoints(checkpoint_dir, keep_last=3):
    """Keep only the most recent N checkpoints"""
    checkpoints = sorted(
        [f for f in checkpoint_dir.glob("checkpoint_epoch_*.pt")],
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    if len(checkpoints) > keep_last:
        to_delete = checkpoints[:-keep_last]
        print(f"\nCleaning up {len(to_delete)} old checkpoints...")
        for cp in to_delete:
            cp.unlink()
            print(f"  Deleted: {cp.name}")
    
    print(f"✓ Kept {min(keep_last, len(checkpoints))} most recent checkpoints")

# Uncomment to actually clean up:
# cleanup_old_checkpoints(checkpoint_dir, keep_last=3)

# =============================================================================
# KEY TAKEAWAYS
# =============================================================================

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. TYPES OF SAVING
   ✓ Full checkpoint: model + optimizer + training state
   ✓ Best model: only model weights (for deployment)
   ✓ Periodic: save every N epochs

2. CHECKPOINT CONTENTS
   ✓ model_state_dict: Model weights
   ✓ optimizer_state_dict: Optimizer state (momentum, etc.)
   ✓ epoch: Current epoch number
   ✓ loss/accuracy: Performance metrics
   ✓ Additional: learning rate, random state, etc.

3. BEST PRACTICES
   ✓ Save checkpoints regularly
   ✓ Save best model separately
   ✓ Include training state for resuming
   ✓ Clean up old checkpoints
   ✓ Use meaningful filenames
   ✓ Save metadata (config, date, etc.)

4. WHEN TO CHECKPOINT
   ✓ Every N epochs (e.g., every 10)
   ✓ When validation improves
   ✓ Before long training runs
   ✓ Before hyperparameter changes

5. FILE ORGANIZATION
   checkpoints/
   ├── best_model.pt          # Best performing model
   ├── checkpoint_epoch_10.pt # Periodic checkpoints
   ├── checkpoint_epoch_20.pt
   └── last_checkpoint.pt     # Most recent state
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. EASY: Save training history (losses, accuracies) to JSON

2. MEDIUM: Implement early stopping with checkpoint loading:
   - Save when validation improves
   - If no improvement for N epochs, stop and load best

3. MEDIUM: Add metadata to checkpoints:
   - Timestamp
   - Hyperparameters
   - Model architecture details

4. HARD: Implement checkpoint versioning:
   - Keep different versions of model
   - Compare performance across versions
   - Rollback to previous version if needed

5. HARD: Create deployment package:
   - Save model + preprocessing (scaler)
   - Add inference function
   - Create simple API
""")
