# Numerical Stability

## Learning Objectives

By the end of this section, you will be able to:

- Identify overflow and underflow problems in naive softmax and cross-entropy computation
- Derive and implement the log-sum-exp trick for numerically stable computation
- Build complete softmax classifiers using PyTorch's `nn.Module`
- Implement proper training and evaluation loops with best practices
- Train and evaluate models on MNIST and other datasets
- Debug common implementation mistakes in classification pipelines

---

## The Numerical Stability Problem

### Why Naive Computation Fails

The softmax function involves exponentials that can easily overflow or underflow in floating-point arithmetic:

$$\sigma(\mathbf{z})_k = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}$$

For large logits, $\exp(z_k)$ overflows to infinity. For very negative logits, $\exp(z_k)$ underflows to zero.

```python
import numpy as np

# Overflow example
z = np.array([1000, 1001, 1002])
print(np.exp(z))  # [inf, inf, inf] — OVERFLOW!

# Underflow example
z = np.array([-1000, -1001, -1002])
print(np.exp(z))  # [0., 0., 0.] — UNDERFLOW!
```

In IEEE 754 double precision: $\exp(710) \approx \infty$ (overflow) and $\exp(-746) \approx 0$ (underflow). Since neural network logits can easily reach these ranges, naive computation is unreliable.

---

## The Log-Sum-Exp Trick

### Core Idea

The key observation is that softmax is **translation invariant** (proved in the Softmax Function section):

$$\sigma(\mathbf{z})_k = \sigma(\mathbf{z} - c\mathbf{1})_k \quad \forall\, c \in \mathbb{R}$$

By choosing $c = z_{\max} = \max_j z_j$, we shift all logits so the largest is zero:

$$\sigma(\mathbf{z})_k = \frac{\exp(z_k - z_{\max})}{\sum_j \exp(z_j - z_{\max})}$$

This ensures:

- The largest exponent is $\exp(0) = 1$ (no overflow)
- Other exponents are in $(0, 1]$ (controlled magnitude)
- The denominator is at least 1 (no division by zero)

### Stable Softmax Implementation

```python
import torch
import torch.nn.functional as F

def softmax_naive(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Naive softmax — susceptible to overflow/underflow."""
    exp_z = torch.exp(z)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

def softmax_stable(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax using the max trick."""
    z_max = z.max(dim=dim, keepdim=True).values
    exp_z = torch.exp(z - z_max)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

# Test with extreme values
z_extreme = torch.tensor([1000.0, 1001.0, 1002.0])

# Naive: produces nan due to overflow
naive_result = softmax_naive(z_extreme)
print(f"Naive softmax: {naive_result}")  # [nan, nan, nan]

# Stable: works correctly
stable_result = softmax_stable(z_extreme)
print(f"Stable softmax: {stable_result}")  # [0.0900, 0.2447, 0.6652]

# PyTorch's built-in (already stable)
pytorch_result = F.softmax(z_extreme, dim=0)
print(f"PyTorch softmax: {pytorch_result}")  # Same as stable
```

### Stable Log-Softmax

When computing cross-entropy, we need $\log(\sigma(\mathbf{z}))$. Computing $\log(\text{softmax}(\mathbf{z}))$ in two steps loses precision because softmax outputs small numbers whose logarithms amplify floating-point errors. Instead, compute log-softmax directly:

$$\log \sigma(\mathbf{z})_k = z_k - \log\sum_j \exp(z_j)$$

The log-sum-exp term is stabilized as:

$$\log\sum_j \exp(z_j) = z_{\max} + \log\sum_j \exp(z_j - z_{\max})$$

So log-softmax becomes:

$$\log \sigma(\mathbf{z})_k = z_k - z_{\max} - \log\sum_j \exp(z_j - z_{\max})$$

```python
def log_softmax_stable(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable log-softmax.

    log(softmax(z)_k) = z_k - log(sum_j exp(z_j))
                      = z_k - z_max - log(sum_j exp(z_j - z_max))
    """
    z_max = z.max(dim=dim, keepdim=True).values
    shifted = z - z_max
    log_sum_exp = shifted.exp().sum(dim=dim, keepdim=True).log()
    return shifted - log_sum_exp

# PyTorch's built-in is equivalent and optimized
z_extreme = torch.tensor([1000.0, 1001.0, 1002.0])
log_probs_builtin = F.log_softmax(z_extreme, dim=0)
log_probs_manual = log_softmax_stable(z_extreme, dim=0)
print(f"Max difference: {(log_probs_builtin - log_probs_manual).abs().max():.2e}")
```

### Stable Cross-Entropy Computation

The cross-entropy loss for true class $c$ is:

$$\mathcal{L} = -\log \hat{\pi}_c = -z_c + \log\sum_j \exp(z_j)$$

With the log-sum-exp trick:

$$\mathcal{L} = -z_c + z_{\max} + \log\sum_j \exp(z_j - z_{\max})$$

This avoids both overflow (large exponentials) and underflow (log of small numbers).

```python
def ce_naive(logits, targets):
    """UNSTABLE: Computing softmax then log loses precision."""
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log(probs)  # log(small number) → large negative
    return F.nll_loss(log_probs, targets)

def ce_stable(logits, targets):
    """STABLE: Using log_softmax directly."""
    log_probs = F.log_softmax(logits, dim=1)
    return F.nll_loss(log_probs, targets)

# Test with extreme logits
extreme_logits = torch.tensor([[100.0, 0.0, 0.0]])  # Very confident
targets = torch.tensor([0])

print(f"Naive:   {ce_naive(extreme_logits, targets).item()}")    # May be unstable
print(f"Stable:  {ce_stable(extreme_logits, targets).item()}")   # Always works
print(f"PyTorch: {F.cross_entropy(extreme_logits, targets).item()}")  # Built-in
```

!!! warning "Always Use the Stable Form"
    Never compute `torch.log(F.softmax(logits))`. Always use `F.log_softmax(logits)` or `F.cross_entropy(logits, targets)` directly.

---

## PyTorch Classification Pipeline

### The nn.Module Pattern

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
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — compute logits.

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

### Multi-Layer Classifier

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

        # Output layer (no activation — returns logits)
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)
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
```

### CrossEntropyLoss: Correct Usage

```python
import torch.optim as optim

model = MLPClassifier(input_dim=784, hidden_dims=[256, 128], num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# CORRECT USAGE
# ==========================================
logits = torch.tensor([[2.0, 1.0, 0.5],
                       [0.5, 2.5, 1.0]])  # Shape: (2, 3)
targets = torch.tensor([0, 1])             # Shape: (2,)
loss = criterion(logits, targets)

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

---

## Complete Training Pipeline

### Training and Evaluation Functions

```python
def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> tuple:
    """Train for one epoch."""
    model.train()  # Enable dropout, batch norm, etc.

    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)

        # Flatten images if needed (e.g., for MNIST)
        if data.dim() > 2:
            data = data.view(data.size(0), -1)

        # Forward pass
        logits = model(data)
        loss = criterion(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * data.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Evaluate model on a dataset."""
    model.eval()  # Disable dropout, use running stats for batch norm

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient computation needed
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

    return total_loss / total, correct / total
```

### Full MNIST Training Script

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time

# Configuration
CONFIG = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'hidden_dims': [256, 128],
    'dropout_rate': 0.2,
    'val_split': 0.1,
    'random_seed': 42,
}

torch.manual_seed(CONFIG['random_seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

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

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# Model, Loss, Optimizer
model = MLPClassifier(
    input_dim=28*28,
    hidden_dims=CONFIG['hidden_dims'],
    num_classes=10,
    dropout_rate=CONFIG['dropout_rate']
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# Training Loop
best_val_acc = 0.0
for epoch in range(CONFIG['num_epochs']):
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}]")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

# Final Evaluation
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\nFinal Test Accuracy: {test_acc:.2%}")
```

---

## Inference with Confidence Scores

```python
def predict_with_confidence(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device
) -> tuple:
    """
    Make predictions with confidence scores.

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

# Example usage
sample_data, sample_labels = next(iter(test_loader))
predictions, confidence, probs = predict_with_confidence(model, sample_data, device)

for i in range(5):
    print(f"True: {sample_labels[i].item()}, "
          f"Pred: {predictions[i].item()}, "
          f"Confidence: {confidence[i].item():.4f}")
```

---

## Model Saving and Loading

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
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

def load_checkpoint(path, device):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)

    config = checkpoint['model_config']
    model = MLPClassifier(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint['epoch'], checkpoint['loss']
```

---

## Common Debugging Techniques

### Gradient Checking

```python
def check_gradients(model, sample_input, sample_target):
    """Check that gradients are flowing properly."""
    model.train()
    model.zero_grad()

    logits = model(sample_input)
    loss = F.cross_entropy(logits, sample_target)
    loss.backward()

    print("Gradient Check")
    print("=" * 50)

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: shape={param.shape}, grad_norm={grad_norm:.6f}")

            if grad_norm < 1e-7:
                print("  ⚠️ WARNING: Gradients may be vanishing!")
            elif grad_norm > 1000:
                print("  ⚠️ WARNING: Gradients may be exploding!")
```

### Overfitting a Single Batch

```python
def overfit_single_batch(model, train_loader, criterion, device, num_iterations=100):
    """
    Try to overfit on a single batch.
    If the model can't overfit, there might be a bug.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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
```

---

## Summary

### Numerical Stability Rules

| Problem | Solution | PyTorch Function |
|---------|----------|------------------|
| Softmax overflow | Subtract max before exp | `F.softmax()` |
| Log-softmax precision | Compute log-softmax directly | `F.log_softmax()` |
| Cross-entropy stability | Use log-softmax + NLL | `F.cross_entropy()` |
| Gradient computation | Use autograd | `loss.backward()` |

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

!!! tip "Before Training"
    - [ ] Move data to correct device (`data.to(device)`)
    - [ ] Set model to `train()` / `eval()` mode
    - [ ] Zero gradients before backward pass (`optimizer.zero_grad()`)
    - [ ] Use `torch.no_grad()` during evaluation
    - [ ] Save best model based on validation metrics
    - [ ] Use learning rate scheduling

### Common Mistakes

!!! warning "Avoid These Errors"
    1. **Applying softmax before `CrossEntropyLoss`** — it is applied internally
    2. **Using one-hot targets** — use class indices instead
    3. **Forgetting `model.eval()`** — dropout will still be active during inference
    4. **Forgetting `torch.no_grad()`** — wastes memory during inference
    5. **Computing `log(softmax(x))`** — use `log_softmax(x)` for numerical stability

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 6.2.2.
2. PyTorch Documentation: [torch.nn](https://pytorch.org/docs/stable/nn.html)
3. PyTorch Documentation: [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
4. PyTorch Tutorials: [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
