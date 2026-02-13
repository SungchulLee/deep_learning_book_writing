"""
================================================================================
01_proper_training_loop.py - Professional Training Structure
================================================================================

LEARNING OBJECTIVES:
- Learn proper code organization for training
- Understand train vs eval mode
- Implement validation during training
- Create reusable training functions
- Best practices for production code

PREREQUISITES:
- Completed all basics tutorials
- Understanding of train/val/test split

TIME TO COMPLETE: ~1 hour

DIFFICULTY: ⭐⭐⭐☆☆ (Intermediate)
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import matplotlib.pyplot as plt

print("="*80)
print("PROPER TRAINING LOOP STRUCTURE")
print("="*80)

# =============================================================================
# PART 1: MODEL DEFINITION (Clean and Modular)
# =============================================================================

class LogisticRegression(nn.Module):
    """
    Logistic Regression Model with proper docstring
    
    Args:
        input_dim (int): Number of input features
        
    Attributes:
        linear (nn.Linear): Linear transformation layer
    """
    
    def __init__(self, input_dim: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, 1)
        """
        return torch.sigmoid(self.linear(x))


# =============================================================================
# PART 2: DATA PREPARATION FUNCTION
# =============================================================================

def prepare_data(n_samples: int = 1000, 
                 test_size: float = 0.2,
                 val_size: float = 0.2,
                 random_state: int = 42) -> Tuple:
    """
    Prepare train, validation, and test sets
    
    Args:
        n_samples: Total number of samples
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining data)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n1. Preparing Data")
    print("-" * 40)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=random_state
    )
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    y_val = torch.FloatTensor(y_val).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# PART 3: TRAINING FUNCTION
# =============================================================================

def train_one_epoch(model: nn.Module,
                    train_data: torch.Tensor,
                    train_labels: torch.Tensor,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    """
    Train for one epoch
    
    Args:
        model: The model to train
        train_data: Training features
        train_labels: Training labels
        criterion: Loss function
        optimizer: Optimizer
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    # Set model to training mode
    # This is important for layers like Dropout and BatchNorm
    model.train()
    
    # Forward pass
    predictions = model(train_data)
    loss = criterion(predictions, train_labels)
    
    # Backward pass
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Compute new gradients
    optimizer.step()       # Update parameters
    
    # Calculate accuracy
    with torch.no_grad():
        predicted_classes = (predictions >= 0.5).float()
        accuracy = (predicted_classes == train_labels).float().mean().item()
    
    return loss.item(), accuracy


# =============================================================================
# PART 4: VALIDATION FUNCTION
# =============================================================================

def validate(model: nn.Module,
             val_data: torch.Tensor,
             val_labels: torch.Tensor,
             criterion: nn.Module) -> Tuple[float, float]:
    """
    Validate the model
    
    Args:
        model: The model to validate
        val_data: Validation features
        val_labels: Validation labels
        criterion: Loss function
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    # Set model to evaluation mode
    # This disables dropout and sets batchnorm to use running statistics
    model.eval()
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        predictions = model(val_data)
        loss = criterion(predictions, val_labels)
        
        predicted_classes = (predictions >= 0.5).float()
        accuracy = (predicted_classes == val_labels).float().mean().item()
    
    return loss.item(), accuracy


# =============================================================================
# PART 5: MAIN TRAINING LOOP
# =============================================================================

def train_model(model: nn.Module,
                train_data: torch.Tensor,
                train_labels: torch.Tensor,
                val_data: torch.Tensor,
                val_labels: torch.Tensor,
                num_epochs: int = 100,
                learning_rate: float = 0.01,
                print_every: int = 10) -> Dict:
    """
    Complete training loop with validation
    
    Args:
        model: Model to train
        train_data: Training features
        train_labels: Training labels
        val_data: Validation features
        val_labels: Validation labels
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        print_every: Print progress every N epochs
        
    Returns:
        Dictionary containing training history
    """
    print("\n2. Training Model")
    print("-" * 40)
    
    # Setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"Training for {num_epochs} epochs with lr={learning_rate}")
    print("-" * 40)
    
    # Training loop
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_data, train_labels, criterion, optimizer
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_data, val_labels, criterion
        )
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    
    print("\nTraining completed!")
    return history


# =============================================================================
# PART 6: TESTING FUNCTION
# =============================================================================

def test_model(model: nn.Module,
               test_data: torch.Tensor,
               test_labels: torch.Tensor) -> Dict:
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_data: Test features
        test_labels: Test labels
        
    Returns:
        Dictionary with test metrics
    """
    print("\n3. Testing Model")
    print("-" * 40)
    
    model.eval()
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        predictions = model(test_data)
        loss = criterion(predictions, test_labels)
        
        predicted_classes = (predictions >= 0.5).float()
        accuracy = (predicted_classes == test_labels).float().mean().item()
    
    results = {
        'test_loss': loss.item(),
        'test_accuracy': accuracy
    }
    
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    
    return results


# =============================================================================
# PART 7: VISUALIZATION
# =============================================================================

def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the figure
    """
    print("\n4. Creating Visualizations")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss During Training', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy During Training', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("RUNNING COMPLETE TRAINING PIPELINE")
    print("="*80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    
    # Create model
    input_dim = X_train.shape[1]
    model = LogisticRegression(input_dim)
    print(f"\nModel created with {input_dim} input features")
    
    # Train model
    history = train_model(
        model=model,
        train_data=X_train,
        train_labels=y_train,
        val_data=X_val,
        val_labels=y_val,
        num_epochs=200,
        learning_rate=0.01,
        print_every=20
    )
    
    # Test model
    test_results = test_model(model, X_test, y_test)
    
    # Visualize
    plot_training_history(history)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
    
    # Check for overfitting
    if history['train_acc'][-1] - history['val_acc'][-1] > 0.05:
        print("\n⚠️  Warning: Possible overfitting detected!")
        print("   Training accuracy is much higher than validation accuracy")
    else:
        print("\n✅ Model generalizes well!")


# Run main function
if __name__ == "__main__":
    main()

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. CODE ORGANIZATION
   ✓ Separate functions for each task
   ✓ Clear function signatures with type hints
   ✓ Proper docstrings
   ✓ Modular and reusable code

2. TRAIN VS EVAL MODE
   ✓ model.train() - Enables training behavior
   ✓ model.eval() - Disables dropout, uses fixed batchnorm stats
   ✓ torch.no_grad() - Disables gradient computation

3. THREE-WAY SPLIT
   ✓ Training set - For learning
   ✓ Validation set - For hyperparameter tuning
   ✓ Test set - For final evaluation (use only once!)

4. MONITORING
   ✓ Track both train and validation metrics
   ✓ Watch for overfitting
   ✓ Plot learning curves
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. Add gradient clipping to prevent exploding gradients
2. Implement early stopping when val_loss doesn't improve
3. Add learning rate warmup for first few epochs
4. Try different optimizers (SGD, RMSprop, AdamW)
5. Implement k-fold cross-validation
""")
