# PyTorch Implementation of Logistic Regression

## Learning Objectives

By the end of this section, you will be able to:

- Implement logistic regression using PyTorch's `nn.Module`
- Build complete training pipelines with proper data handling
- Use DataLoaders for efficient batch processing
- Implement model checkpointing and evaluation
- Apply logistic regression to real-world datasets

---

## Implementation Overview

A complete logistic regression implementation in PyTorch involves:

1. **Data Preparation**: Loading, preprocessing, and creating DataLoaders
2. **Model Definition**: Creating a `nn.Module` subclass
3. **Training Loop**: Forward pass, loss computation, backward pass, optimization
4. **Evaluation**: Testing on held-out data with appropriate metrics
5. **Deployment**: Saving/loading models for inference

---

## Complete Implementation

```python
"""
Complete PyTorch Logistic Regression Implementation
===================================================

A production-ready implementation covering:
- Data handling with train/val/test splits
- Model definition with proper documentation
- Training with early stopping
- Comprehensive evaluation metrics
- Model checkpointing

Author: Deep Learning Foundations
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from pathlib import Path
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("PYTORCH LOGISTIC REGRESSION - COMPLETE IMPLEMENTATION")
print("="*70)

# ============================================================================
# PART 1: DATA PREPARATION
# ============================================================================

class BinaryClassificationDataset(Dataset):
    """
    Custom Dataset for binary classification.
    
    Handles data preprocessing including standardization.
    Can work with either raw numpy arrays or sklearn datasets.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,) with values in {0, 1}
        scaler: Optional pre-fitted StandardScaler
        fit_scaler: Whether to fit the scaler (True for training data)
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 scaler: Optional[StandardScaler] = None,
                 fit_scaler: bool = False):
        
        # Handle scaling
        if scaler is None and fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif scaler is not None:
            self.scaler = scaler
            X = self.scaler.transform(X)
        else:
            self.scaler = None
        
        # Convert to tensors
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def get_scaler(self) -> Optional[StandardScaler]:
        return self.scaler


def prepare_data(X: np.ndarray, y: np.ndarray,
                test_size: float = 0.2,
                val_size: float = 0.2,
                batch_size: int = 32,
                random_state: int = 42) -> Dict:
    """
    Prepare train, validation, and test DataLoaders.
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining)
        batch_size: Batch size for DataLoaders
        random_state: Random seed
        
    Returns:
        Dictionary with DataLoaders and metadata
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    # Create datasets
    train_dataset = BinaryClassificationDataset(X_train, y_train, fit_scaler=True)
    scaler = train_dataset.get_scaler()
    
    val_dataset = BinaryClassificationDataset(X_val, y_val, scaler=scaler)
    test_dataset = BinaryClassificationDataset(X_test, y_test, scaler=scaler)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nData prepared:")
    print(f"  Training samples:   {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print(f"  Test samples:       {len(test_dataset):,}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Features:           {X.shape[1]}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'n_features': X.shape[1]
    }

# ============================================================================
# PART 2: MODEL DEFINITION
# ============================================================================

class LogisticRegression(nn.Module):
    """
    Logistic Regression model for binary classification.
    
    Architecture: Linear -> Sigmoid
    
    For numerical stability during training, we recommend using
    BCEWithLogitsLoss and removing the sigmoid from forward().
    This implementation includes sigmoid for direct probability output.
    
    Args:
        n_features: Number of input features
        
    Attributes:
        linear: Linear transformation layer
    """
    
    def __init__(self, n_features: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        
        # Initialize weights (optional, PyTorch has good defaults)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Probabilities of shape (batch_size, 1) in range [0, 1]
        """
        return torch.sigmoid(self.linear(x))
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits before sigmoid."""
        return self.linear(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make binary predictions.
        
        Args:
            x: Input features
            threshold: Classification threshold
            
        Returns:
            Binary predictions {0, 1}
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
            return (probs >= threshold).float()

# ============================================================================
# PART 3: TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model: nn.Module, 
               train_loader: DataLoader,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * len(batch_X)
        predicted_classes = (predictions >= 0.5).float()
        correct += (predicted_classes == batch_y).sum().item()
        total += len(batch_X)
    
    return total_loss / total, correct / total


def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module) -> Tuple[float, float]:
    """
    Validate the model.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item() * len(batch_X)
            predicted_classes = (predictions >= 0.5).float()
            correct += (predicted_classes == batch_y).sum().item()
            total += len(batch_X)
    
    return total_loss / total, correct / total


def train_model(model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               num_epochs: int = 100,
               learning_rate: float = 0.01,
               patience: int = 10,
               verbose: bool = True) -> Dict:
    """
    Complete training loop with early stopping.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training history
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    if verbose:
        print(f"\nTraining for up to {num_epochs} epochs (patience={patience})...")
        print("-" * 60)
    
    for epoch in range(num_epochs):
        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val Acc={val_acc:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    history['best_epoch'] = epoch - patience_counter + 1
    history['total_epochs'] = epoch + 1
    
    if verbose:
        print(f"\nTraining complete. Best epoch: {history['best_epoch']}")
    
    return history

# ============================================================================
# PART 4: EVALUATION
# ============================================================================

def evaluate_model(model: nn.Module,
                  test_loader: DataLoader,
                  threshold: float = 0.5) -> Dict:
    """
    Comprehensive model evaluation.
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            probs = model(batch_X)
            preds = (probs >= threshold).float()
            
            all_probs.extend(probs.numpy().flatten())
            all_preds.extend(preds.numpy().flatten())
            all_targets.extend(batch_y.numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
        'auc': roc_auc_score(all_targets, all_probs),
        'confusion_matrix': confusion_matrix(all_targets, all_preds),
        'probabilities': all_probs,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return metrics


def print_evaluation_report(metrics: Dict):
    """Print formatted evaluation report."""
    print("\n" + "="*60)
    print("MODEL EVALUATION REPORT")
    print("="*60)
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"           Predicted")
    print(f"             0     1")
    print(f"Actual 0   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       1   {cm[1,0]:4d}  {cm[1,1]:4d}")

# ============================================================================
# PART 5: CHECKPOINTING
# ============================================================================

def save_checkpoint(model: nn.Module, 
                   scaler: StandardScaler,
                   history: Dict,
                   filepath: str):
    """Save model checkpoint with all necessary components."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'n_features': model.linear.in_features,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'history': history
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to: {filepath}")


def load_checkpoint(filepath: str) -> Tuple[nn.Module, StandardScaler, Dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(filepath)
    
    # Recreate model
    model = LogisticRegression(checkpoint['n_features'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Recreate scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint['scaler_mean'])
    scaler.scale_ = np.array(checkpoint['scaler_scale'])
    
    print(f"Checkpoint loaded from: {filepath}")
    return model, scaler, checkpoint['history']

# ============================================================================
# PART 6: COMPLETE EXAMPLE
# ============================================================================

def main():
    """Complete example with breast cancer dataset."""
    
    print("\n" + "="*70)
    print("EXAMPLE: BREAST CANCER CLASSIFICATION")
    print("="*70)
    
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print(f"\nDataset: Wisconsin Breast Cancer")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {data.target_names.tolist()}")
    
    # Prepare data
    data_dict = prepare_data(X, y, batch_size=32)
    
    # Create model
    model = LogisticRegression(data_dict['n_features'])
    print(f"\nModel architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    history = train_model(
        model,
        data_dict['train_loader'],
        data_dict['val_loader'],
        num_epochs=200,
        learning_rate=0.01,
        patience=15
    )
    
    # Evaluate
    metrics = evaluate_model(model, data_dict['test_loader'])
    print_evaluation_report(metrics)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training curves
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].axvline(x=history['best_epoch']-1, color='r', linestyle='--', 
                   label=f'Best (epoch {history["best_epoch"]})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].axvline(x=history['best_epoch']-1, color='r', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Confusion matrix
    import seaborn as sns
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', 
               cmap='Blues', ax=axes[2],
               xticklabels=['Malignant', 'Benign'],
               yticklabels=['Malignant', 'Benign'])
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    axes[2].set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('logistic_regression_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Results visualization saved!")
    
    # Save checkpoint
    save_checkpoint(model, data_dict['scaler'], history, 
                   'logistic_regression_checkpoint.pt')
    
    return model, data_dict, metrics


if __name__ == "__main__":
    model, data_dict, metrics = main()
```

---

## Best Practices Summary

### Data Handling

| Practice | Reason |
|----------|--------|
| Use `DataLoader` | Efficient batching, shuffling, multiprocessing |
| Standardize features | Faster convergence, prevents feature dominance |
| Stratified splits | Preserves class distribution |
| Three-way split | Prevents overfitting to validation set |

### Model Design

| Practice | Reason |
|----------|--------|
| `nn.Module` subclass | Clean, reusable, integrates with PyTorch ecosystem |
| `BCEWithLogitsLoss` | Numerical stability |
| Xavier initialization | Helps with training dynamics |

### Training

| Practice | Reason |
|----------|--------|
| Early stopping | Prevents overfitting |
| Learning rate tuning | Critical for convergence |
| Gradient clipping | Prevents exploding gradients |
| Checkpoint best model | Don't lose best performance |

### Evaluation

| Practice | Reason |
|----------|--------|
| Multiple metrics | Accuracy alone is insufficient |
| Confusion matrix | Understand error types |
| AUC-ROC | Threshold-independent performance |

---

## Exercises

1. **Extend** the implementation to support GPU training with `.to(device)`.

2. **Implement** learning rate scheduling (e.g., `ReduceLROnPlateau`).

3. **Add** L2 regularization using the `weight_decay` parameter in the optimizer.

4. **Create** an inference function that loads a checkpoint and predicts on new data.

5. **Implement** k-fold cross-validation for more robust performance estimates.

---

## Summary

This implementation provides a complete, production-ready logistic regression pipeline in PyTorch, demonstrating:

- Proper data handling with custom `Dataset` classes
- Clean model architecture following PyTorch conventions
- Robust training with early stopping
- Comprehensive evaluation with multiple metrics
- Model persistence for deployment

The same patterns extend directly to more complex neural networks, making this an essential foundation for deep learning practitioners.
