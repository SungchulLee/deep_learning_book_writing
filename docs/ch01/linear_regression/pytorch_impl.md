# PyTorch Implementation of Linear Regression

## Overview

This comprehensive guide demonstrates the complete PyTorch implementation of linear regression, progressing from basic tensor operations to production-ready code. Each implementation builds on previous concepts, culminating in a fully-featured training pipeline.

## Implementation Progression

| Level | Approach | Key Learning |
|-------|----------|--------------|
| 1 | Manual tensors | Understanding the math |
| 2 | Autograd | Automatic differentiation |
| 3 | nn.Module | PyTorch model structure |
| 4 | Full pipeline | Production practices |

## Level 1: Manual Implementation

### Pure Tensor Operations

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def linear_regression_manual(
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 0.01,
    n_epochs: int = 100
) -> dict:
    """
    Linear regression with manual gradient computation.
    
    No autograd - understand the mathematics directly.
    """
    n_samples, n_features = X.shape
    
    # Initialize parameters
    w = torch.zeros(n_features, 1, dtype=torch.float32)
    b = torch.tensor([0.0], dtype=torch.float32)
    
    history = {'loss': [], 'w_norm': []}
    
    for epoch in range(n_epochs):
        # ========== FORWARD PASS ==========
        # Linear model: y_pred = X @ w + b
        y_pred = X @ w + b
        
        # ========== COMPUTE LOSS ==========
        # MSE = (1/n) * sum((y - y_pred)^2)
        loss = torch.mean((y - y_pred) ** 2)
        
        # ========== COMPUTE GRADIENTS ==========
        # d(MSE)/dw = (2/n) * X.T @ (y_pred - y)
        # d(MSE)/db = (2/n) * sum(y_pred - y)
        error = y_pred - y
        grad_w = (2.0 / n_samples) * (X.T @ error)
        grad_b = (2.0 / n_samples) * torch.sum(error)
        
        # ========== UPDATE PARAMETERS ==========
        # Gradient descent: param = param - lr * gradient
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        
        # Track history
        history['loss'].append(loss.item())
        history['w_norm'].append(torch.norm(w).item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.6f}")
    
    return {'w': w, 'b': b.item(), 'history': history}

# Generate synthetic data
torch.manual_seed(42)
n_samples, n_features = 200, 3
X = torch.randn(n_samples, n_features)
true_w = torch.tensor([[2.0], [-1.5], [0.5]])
true_b = 1.0
noise = 0.3 * torch.randn(n_samples, 1)
y = X @ true_w + true_b + noise

print("=" * 60)
print("LEVEL 1: MANUAL IMPLEMENTATION")
print("=" * 60)
result_manual = linear_regression_manual(X, y, learning_rate=0.1, n_epochs=100)
print(f"\nTrue weights: {true_w.squeeze().tolist()}")
print(f"Learned weights: {result_manual['w'].squeeze().tolist()}")
print(f"True bias: {true_b}")
print(f"Learned bias: {result_manual['b']:.4f}")
```

## Level 2: Using Autograd

### Automatic Gradient Computation

```python
def linear_regression_autograd(
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 0.01,
    n_epochs: int = 100
) -> dict:
    """
    Linear regression with PyTorch autograd.
    
    Key: requires_grad=True enables automatic differentiation.
    """
    n_features = X.shape[1]
    
    # Parameters WITH gradient tracking
    w = torch.zeros(n_features, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    history = {'loss': []}
    
    for epoch in range(n_epochs):
        # ========== FORWARD PASS ==========
        # PyTorch builds computational graph automatically
        y_pred = X @ w + b
        
        # ========== COMPUTE LOSS ==========
        loss = torch.mean((y - y_pred) ** 2)
        
        # ========== BACKWARD PASS ==========
        # Autograd computes gradients!
        loss.backward()
        
        # ========== UPDATE PARAMETERS ==========
        # Must use no_grad() to prevent tracking updates
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        
        # ========== ZERO GRADIENTS ==========
        # Critical: gradients accumulate by default!
        w.grad.zero_()
        b.grad.zero_()
        
        history['loss'].append(loss.item())
    
    return {'w': w.detach(), 'b': b.detach().item(), 'history': history}

print("\n" + "=" * 60)
print("LEVEL 2: AUTOGRAD IMPLEMENTATION")
print("=" * 60)
result_autograd = linear_regression_autograd(X, y, learning_rate=0.1, n_epochs=100)
print(f"Final loss: {result_autograd['history']['loss'][-1]:.6f}")
```

## Level 3: Using nn.Module

### Standard PyTorch Model Structure

```python
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    """
    Linear Regression as a PyTorch Module.
    
    This is the standard way to define models in PyTorch.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1):
        """
        Initialize the model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of outputs (1 for regression)
        """
        super(LinearRegressionModel, self).__init__()
        
        # nn.Linear handles weight initialization and forward computation
        # Implements: y = x @ W.T + b
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        return self.linear(x)

def train_nn_module(
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 0.01,
    n_epochs: int = 100
) -> dict:
    """
    Train using nn.Module and optimizer.
    """
    n_features = X.shape[1]
    
    # Create model
    model = LinearRegressionModel(n_features)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer handles parameter updates
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    history = {'loss': []}
    
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
        
        history['loss'].append(loss.item())
    
    return {'model': model, 'history': history}

print("\n" + "=" * 60)
print("LEVEL 3: NN.MODULE IMPLEMENTATION")
print("=" * 60)
result_module = train_nn_module(X, y, learning_rate=0.1, n_epochs=100)
print(f"Final loss: {result_module['history']['loss'][-1]:.6f}")

# Access learned parameters
model = result_module['model']
print(f"Learned weights: {model.linear.weight.data.squeeze().tolist()}")
print(f"Learned bias: {model.linear.bias.data.item():.4f}")
```

## Level 4: Complete Training Pipeline

### Production-Ready Implementation

```python
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearRegressionPipeline:
    """
    Complete, production-ready linear regression pipeline.
    
    Features:
    - Train/validation/test splits
    - Feature scaling
    - Mini-batch training
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    """
    
    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        patience: int = 10
    ):
        self.model = LinearRegressionModel(input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.batch_size = batch_size
        self.patience = patience
        self.history = {'train_loss': [], 'val_loss': []}
        
        # For data scaling
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def prepare_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        val_size: float = 0.2,
        test_size: float = 0.1
    ):
        """Prepare train/val/test splits with scaling."""
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
        )
        
        # Scale features (fit only on training data)
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Scale targets
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1))
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1))
        
        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_scaled)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val_scaled)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.FloatTensor(y_test_scaled)
        )
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        print(f"Data prepared:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val:   {len(val_dataset)} samples")
        print(f"  Test:  {len(test_dataset)} samples")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in self.train_loader:
            # Forward
            y_pred = self.model(batch_X)
            loss = self.criterion(y_pred, batch_y)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                y_pred = self.model(batch_X)
                loss = self.criterion(y_pred, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def fit(self, n_epochs: int = 100, verbose: bool = True):
        """
        Full training loop with early stopping.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}: Train={train_loss:.6f}, "
                      f"Val={val_loss:.6f}, LR={lr:.2e}")
            
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        
        return self.history
    
    def evaluate(self):
        """Evaluate on test set."""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                y_pred = self.model(batch_X)
                predictions.append(y_pred)
                targets.append(batch_y)
        
        predictions = torch.cat(predictions).numpy()
        targets = torch.cat(targets).numpy()
        
        # Inverse transform to original scale
        predictions_orig = self.scaler_y.inverse_transform(predictions)
        targets_orig = self.scaler_y.inverse_transform(targets)
        
        # Metrics
        mse = np.mean((targets_orig - predictions_orig) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(targets_orig - predictions_orig))
        
        # R-squared
        ss_res = np.sum((targets_orig - predictions_orig) ** 2)
        ss_tot = np.sum((targets_orig - np.mean(targets_orig)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print("\nTest Set Evaluation:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        self.model.eval()
        X_scaled = self.scaler_X.transform(X_new)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).numpy()
        
        return self.scaler_y.inverse_transform(y_pred_scaled)
    
    def save(self, path: str):
        """Save model and scalers."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and scalers."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.history = checkpoint['history']
        print(f"Model loaded from {path}")


# Example usage
print("\n" + "=" * 60)
print("LEVEL 4: COMPLETE PIPELINE")
print("=" * 60)

# Generate larger dataset
torch.manual_seed(42)
np.random.seed(42)
n_samples, n_features = 1000, 5
X_np = np.random.randn(n_samples, n_features)
true_coef = np.array([2.0, -1.5, 0.5, 1.0, -0.8])
y_np = X_np @ true_coef + 0.5 + 0.3 * np.random.randn(n_samples)

# Create and train pipeline
pipeline = LinearRegressionPipeline(
    input_dim=n_features,
    learning_rate=0.01,
    batch_size=32,
    patience=15
)

pipeline.prepare_data(X_np, y_np, val_size=0.15, test_size=0.15)
history = pipeline.fit(n_epochs=200, verbose=True)
metrics = pipeline.evaluate()

# Make predictions on new data
X_new = np.random.randn(5, n_features)
predictions = pipeline.predict(X_new)
print(f"\nPredictions on new data: {predictions.flatten()}")
```

## Visualization

```python
def plot_training_history(history: dict, title: str = "Training History"):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='Train', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2 = axes[1]
    ax2.plot(history['train_loss'], label='Train', linewidth=2)
    if 'val_loss' in history:
        ax2.plot(history['val_loss'], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE, log scale)')
    ax2.set_title(f'{title} - Log Scale')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# plot_training_history(history, "Complete Pipeline")
```

## Summary

### Implementation Levels

| Level | Components Used | Use Case |
|-------|-----------------|----------|
| Manual | Tensors only | Learning fundamentals |
| Autograd | requires_grad, backward() | Understanding autodiff |
| nn.Module | nn.Linear, Optimizer | Standard PyTorch |
| Pipeline | DataLoader, schedulers, etc. | Production code |

### Key PyTorch Patterns

```python
# Standard training loop
model = nn.Linear(d_in, d_out)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(n_epochs):
    for X_batch, y_batch in dataloader:
        # Forward
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Best Practices

1. **Always scale features** before training
2. **Use DataLoader** for efficient batching
3. **Monitor validation loss** for overfitting
4. **Implement early stopping** to prevent overfitting
5. **Save checkpoints** during training
6. **Use appropriate optimizers** (Adam is a good default)

## References

- PyTorch Documentation: [torch.nn](https://pytorch.org/docs/stable/nn.html)
- PyTorch Tutorials: [Learning PyTorch](https://pytorch.org/tutorials/)
- PyTorch Examples: [Linear Regression](https://github.com/pytorch/examples)
