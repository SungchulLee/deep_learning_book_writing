# PyTorch Implementation of Softmax Regression

## Learning Objectives

By the end of this section, you will be able to:

- Build complete softmax classifiers using PyTorch's nn.Module
- Implement proper training and evaluation loops
- Use PyTorch's built-in loss functions correctly
- Apply best practices for model development
- Debug common implementation mistakes
- Train on real datasets like MNIST

---

## PyTorch Fundamentals for Classification

### The nn.Module Pattern

All PyTorch models inherit from `nn.Module`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxClassifier(nn.Module):
    """
    Basic softmax classifier using PyTorch.
    
    Architecture: Input → Linear → Softmax (implicit in loss)
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        """
        Initialize the classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
        """
        super().__init__()  # Always call parent constructor
        
        # Define layers
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - compute logits.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Logits of shape (batch_size, num_classes)
            
        Note: We return LOGITS, not probabilities!
              CrossEntropyLoss applies softmax internally.
        """
        return self.linear(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted probabilities (for inference)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
```

---

## Multi-Layer Softmax Classifier

### Architecture with Hidden Layers

```python
class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron for classification.
    
    Architecture:
        Input → Hidden1 → ReLU → Dropout → 
        Hidden2 → ReLU → Dropout → Output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        num_classes: int,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the MLP classifier.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes, e.g., [128, 64]
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation - logits)
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Combine into Sequential
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.network(x)


# Example instantiation
model = MLPClassifier(
    input_dim=784,      # MNIST: 28x28 = 784
    hidden_dims=[256, 128],
    num_classes=10,
    dropout_rate=0.3
)

print(model)
```

Output:
```
MLPClassifier(
  (network): Sequential(
    (0): Linear(in_features=784, out_features=256, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.3, inplace=False)
    (6): Linear(in_features=128, out_features=10, bias=True)
  )
)
```

---

## Loss Functions and Optimizers

### CrossEntropyLoss: The Standard Choice

```python
import torch.optim as optim

# Create model
model = MLPClassifier(input_dim=784, hidden_dims=[256, 128], num_classes=10)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example forward pass
batch_size = 32
x = torch.randn(batch_size, 784)
y = torch.randint(0, 10, (batch_size,))

# Compute loss
logits = model(x)
loss = criterion(logits, y)

print(f"Logits shape: {logits.shape}")  # (32, 10)
print(f"Loss: {loss.item():.4f}")
```

### Critical: CrossEntropyLoss Input Requirements

```python
# ==========================================
# CORRECT USAGE
# ==========================================

# Input: LOGITS (raw model output)
# Target: CLASS INDICES (not one-hot!)

logits = torch.tensor([[2.0, 1.0, 0.5],
                       [0.5, 2.5, 1.0]])  # Shape: (2, 3)
targets = torch.tensor([0, 1])             # Shape: (2,)

loss = criterion(logits, targets)
print(f"Correct loss: {loss.item():.4f}")

# ==========================================
# COMMON MISTAKES
# ==========================================

# MISTAKE 1: Applying softmax before CrossEntropyLoss
# probs = F.softmax(logits, dim=1)
# loss = criterion(probs, targets)  # WRONG! Double softmax!

# MISTAKE 2: Using one-hot encoded targets
# targets_onehot = F.one_hot(targets, num_classes=3)
# loss = criterion(logits, targets_onehot)  # ERROR!

# MISTAKE 3: Wrong tensor shapes
# targets_wrong = torch.tensor([[0], [1]])  # Shape (2, 1) instead of (2,)
# loss = criterion(logits, targets_wrong)  # ERROR!
```

### Alternative Loss Functions

```python
# For binary classification
bce_with_logits = nn.BCEWithLogitsLoss()

# For multi-label classification (multiple labels per sample)
multilabel_loss = nn.BCEWithLogitsLoss()

# With class weights (for imbalanced data)
class_weights = torch.tensor([1.0, 2.0, 3.0])  # Weight rare classes higher
weighted_ce = nn.CrossEntropyLoss(weight=class_weights)

# With label smoothing (regularization)
smooth_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

# Ignoring certain labels (e.g., padding)
ignore_ce = nn.CrossEntropyLoss(ignore_index=-100)
```

---

## Complete Training Loop

### Standard Training Pipeline

```python
def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> tuple:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (CPU/GPU)
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()  # Set to training mode (enables dropout, etc.)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data to device
        data, targets = data.to(device), targets.to(device)
        
        # Flatten images if needed (e.g., for MNIST)
        if data.dim() > 2:
            data = data.view(data.size(0), -1)
        
        # Forward pass
        logits = model(data)
        loss = criterion(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        # Track metrics
        total_loss += loss.item() * data.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    Evaluate model on a dataset.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            
            if data.dim() > 2:
                data = data.view(data.size(0), -1)
            
            logits = model(data)
            loss = criterion(logits, targets)
            
            total_loss += loss.item() * data.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy
```

---

## Complete MNIST Example

### Full Training Script

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'hidden_dims': [256, 128],
    'dropout_rate': 0.2,
    'val_split': 0.1,
    'random_seed': 42,
}

# Set random seed for reproducibility
torch.manual_seed(CONFIG['random_seed'])

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# Data Loading
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load datasets
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# Split training into train and validation
val_size = int(len(train_dataset) * CONFIG['val_split'])
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(CONFIG['random_seed'])
)

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ==========================================
# Model, Loss, Optimizer
# ==========================================
model = MLPClassifier(
    input_dim=28*28,
    hidden_dims=CONFIG['hidden_dims'],
    num_classes=10,
    dropout_rate=CONFIG['dropout_rate']
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# ==========================================
# Training Loop
# ==========================================
print("\nStarting training...")
print("=" * 60)

best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

start_time = time.time()

for epoch in range(CONFIG['num_epochs']):
    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # Validate
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    # Update scheduler
    scheduler.step(val_loss)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    # Print progress
    print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}]")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# ==========================================
# Final Evaluation
# ==========================================
# Load best model
model.load_state_dict(torch.load('best_model.pth'))

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\nFinal Test Results:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
```

---

## Making Predictions

### Inference with Trained Model

```python
def predict_with_confidence(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device
) -> tuple:
    """
    Make predictions with confidence scores.
    
    Args:
        model: Trained model
        x: Input tensor
        device: Device
    
    Returns:
        Tuple of (predicted_classes, confidence_scores, all_probabilities)
    """
    model.eval()
    x = x.to(device)
    
    with torch.no_grad():
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        logits = model(x)
        probabilities = F.softmax(logits, dim=1)
        
        confidence, predictions = torch.max(probabilities, dim=1)
    
    return predictions, confidence, probabilities


# Example: Predict on test samples
sample_data, sample_labels = next(iter(test_loader))
predictions, confidence, probs = predict_with_confidence(model, sample_data, device)

print("Sample Predictions:")
print("-" * 50)
for i in range(5):
    print(f"True: {sample_labels[i].item()}, "
          f"Pred: {predictions[i].item()}, "
          f"Confidence: {confidence[i].item():.4f}")
```

### Visualizing Predictions

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(model, test_loader, device, num_samples=10):
    """Visualize model predictions on test samples."""
    model.eval()
    
    # Get a batch
    data, labels = next(iter(test_loader))
    data, labels = data.to(device), labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        logits = model(data.view(data.size(0), -1))
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = data[i].cpu().squeeze().numpy()
        true_label = labels[i].item()
        pred_label = predictions[i].item()
        prob = probs[i, pred_label].item()
        
        axes[i].imshow(img, cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\n'
                         f'Conf: {prob:.2f}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

# Uncomment to visualize
# visualize_predictions(model, test_loader, device)
# plt.show()
```

---

## Model Saving and Loading

### Best Practices for Checkpointing

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': {
            'input_dim': 784,
            'hidden_dims': CONFIG['hidden_dims'],
            'num_classes': 10,
            'dropout_rate': CONFIG['dropout_rate'],
        }
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, device: torch.device) -> tuple:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    # Recreate model
    config = checkpoint['model_config']
    model = MLPClassifier(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Recreate optimizer
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    
    return model, optimizer, epoch, loss


# Usage
save_checkpoint(model, optimizer, epoch=5, loss=0.1, path='checkpoint.pth')
model, optimizer, epoch, loss = load_checkpoint('checkpoint.pth', device)
```

---

## Common Debugging Techniques

### Gradient Checking

```python
def check_gradients(model: nn.Module, sample_input: torch.Tensor, sample_target: torch.Tensor):
    """Check that gradients are flowing properly."""
    model.train()
    model.zero_grad()
    
    # Forward pass
    logits = model(sample_input)
    loss = F.cross_entropy(logits, sample_target)
    
    # Backward pass
    loss.backward()
    
    print("Gradient Check")
    print("=" * 50)
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            print(f"{name}:")
            print(f"  Shape: {param.shape}")
            print(f"  Grad norm: {grad_norm:.6f}")
            print(f"  Grad mean: {grad_mean:.6f}")
            
            # Warning if gradients are too small or too large
            if grad_norm < 1e-7:
                print("  ⚠️ WARNING: Gradients may be vanishing!")
            elif grad_norm > 1000:
                print("  ⚠️ WARNING: Gradients may be exploding!")
        else:
            print(f"{name}: No gradient!")


# Usage
sample_x = torch.randn(4, 784).to(device)
sample_y = torch.randint(0, 10, (4,)).to(device)
check_gradients(model, sample_x, sample_y)
```

### Overfitting Check

```python
def overfit_single_batch(model, train_loader, criterion, device, num_iterations=100):
    """
    Try to overfit on a single batch.
    If the model can't overfit, there might be a bug.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Get single batch
    data, targets = next(iter(train_loader))
    data, targets = data.to(device), targets.to(device)
    if data.dim() > 2:
        data = data.view(data.size(0), -1)
    
    print("Overfitting Test (should reach ~100% accuracy)")
    print("=" * 50)
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean().item()
        
        if (i + 1) % 20 == 0:
            print(f"Iteration {i+1}: Loss = {loss.item():.4f}, Acc = {accuracy:.4f}")
    
    if accuracy < 0.95:
        print("⚠️ WARNING: Model couldn't overfit! Check for bugs.")
    else:
        print("✅ Model can overfit successfully.")


# Usage (with fresh model)
fresh_model = MLPClassifier(784, [256, 128], 10).to(device)
overfit_single_batch(fresh_model, train_loader, criterion, device)
```

---

## Summary

### Key PyTorch Components

| Component | Purpose | Key Notes |
|-----------|---------|-----------|
| `nn.Module` | Base class for models | Override `__init__` and `forward` |
| `nn.Linear` | Fully connected layer | Returns logits |
| `nn.CrossEntropyLoss` | Loss function | Takes logits, not probabilities |
| `optim.Adam` | Optimizer | Good default choice |
| `model.train()` | Training mode | Enables dropout |
| `model.eval()` | Evaluation mode | Disables dropout |
| `torch.no_grad()` | Disable gradients | Use during inference |

### Training Loop Checklist

- [ ] Move data to correct device
- [ ] Set model to train/eval mode
- [ ] Zero gradients before backward pass
- [ ] Use `torch.no_grad()` during evaluation
- [ ] Save best model based on validation metrics
- [ ] Use learning rate scheduling

### Common Mistakes

!!! warning "Avoid These Errors"
    1. **Applying softmax before CrossEntropyLoss** — it's applied internally
    2. **Using one-hot targets** — use class indices instead
    3. **Forgetting `model.eval()`** — dropout will still be active
    4. **Forgetting `torch.no_grad()`** — wastes memory during inference
    5. **Wrong tensor shapes** — targets should be 1D, not 2D

---

## Next Steps

With the PyTorch implementation complete, explore:

1. **Advanced Architectures** — CNNs, ResNets, Transformers
2. **Regularization** — Batch normalization, weight decay
3. **Hyperparameter Tuning** — Learning rate schedules, grid search
4. **Deployment** — TorchScript, ONNX export

---

## References

1. PyTorch Documentation: [torch.nn](https://pytorch.org/docs/stable/nn.html)
2. PyTorch Tutorials: [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
3. PyTorch Documentation: [torch.optim](https://pytorch.org/docs/stable/optim.html)
