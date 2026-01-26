# Batch, Mini-Batch, and Stochastic Gradient Descent

## Introduction

When training on datasets with thousands or millions of samples, computing the gradient over the entire dataset at each iteration becomes computationally expensive. This chapter explores three fundamental variants of gradient descent that differ in **how much data they use to compute each gradient update**.

Understanding these variants is essential for practical machine learning, as mini-batch gradient descent has become the de facto standard for training neural networks.

## The Three Variants

### Overview

| Variant | Batch Size | Gradient Computation |
|---------|------------|---------------------|
| **Batch GD** | $N$ (entire dataset) | Exact gradient |
| **Stochastic GD (SGD)** | 1 (single sample) | Very noisy estimate |
| **Mini-Batch GD** | $B$ (typically 16-256) | Balanced estimate |

### Mathematical Formulation

For a loss function over $N$ samples:

$$L(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell(\theta; x_i, y_i)$$

**Batch GD** (Full Gradient):
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \ell(\theta_t; x_i, y_i)$$

**Stochastic GD** (Single Sample):
$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \ell(\theta_t; x_{i_t}, y_{i_t})$$
where $i_t$ is randomly sampled.

**Mini-Batch GD** (Batch of Size $B$):
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{B}\sum_{j \in \mathcal{B}_t} \nabla_\theta \ell(\theta_t; x_j, y_j)$$
where $\mathcal{B}_t$ is a randomly sampled mini-batch.

## Batch Gradient Descent

### Algorithm

```python
def batch_gradient_descent(X, y, model, criterion, 
                           learning_rate, n_epochs):
    """
    Batch Gradient Descent: Use ALL data in each iteration
    """
    for epoch in range(n_epochs):
        # Forward pass on entire dataset
        y_pred = model(X)
        
        # Compute loss over all samples
        loss = criterion(y_pred, y)
        
        # Compute gradient over all samples
        loss.backward()
        
        # Update parameters
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
        
        # Zero gradients
        model.zero_grad()
    
    return model
```

### Characteristics

**Pros**:

- Computes the **exact gradient** of the loss
- Smooth, stable convergence trajectory
- Guaranteed to converge to minimum (for convex problems)

**Cons**:

- Computationally expensive for large datasets
- Memory-intensive (must load all data)
- Cannot escape shallow local minima
- Slower updates (one update per epoch)

### When to Use

- Small datasets ($N < 1000$)
- Problems where exact gradients matter
- Convex optimization problems
- When memory is not a constraint

## Stochastic Gradient Descent (SGD)

### Algorithm

```python
def stochastic_gradient_descent(X, y, model, criterion,
                                 learning_rate, n_epochs):
    """
    Stochastic Gradient Descent: Use ONE sample per iteration
    """
    n_samples = X.shape[0]
    
    for epoch in range(n_epochs):
        # Shuffle data at start of each epoch
        indices = torch.randperm(n_samples)
        
        for i in indices:
            # Select single sample
            X_sample = X[i:i+1]
            y_sample = y[i:i+1]
            
            # Forward pass on single sample
            y_pred = model(X_sample)
            
            # Compute loss for single sample
            loss = criterion(y_pred, y_sample)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
            
            model.zero_grad()
    
    return model
```

### Characteristics

**Pros**:

- Very fast per-iteration
- Can escape local minima (noise helps exploration)
- Online learning capable (process data as it arrives)
- Low memory footprint

**Cons**:

- Very noisy gradient estimates
- Erratic convergence (high variance)
- May not converge to exact minimum
- Cannot leverage vectorization/GPU parallelism

### The Noise-Exploration Trade-off

The gradient noise in SGD is often **beneficial**:

$$\nabla_\theta \ell(\theta; x_i, y_i) = \nabla_\theta L(\theta) + \epsilon_i$$

where $\epsilon_i$ is the "noise" from using a single sample.

This noise helps:

- Escape shallow local minima
- Explore more of the loss landscape
- Lead to solutions that generalize better (flatter minima)

## Mini-Batch Gradient Descent

### Algorithm

```python
def minibatch_gradient_descent(X, y, model, criterion,
                                learning_rate, n_epochs, batch_size):
    """
    Mini-Batch Gradient Descent: Best of both worlds
    """
    # Create DataLoader for automatic batching
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(n_epochs):
        for X_batch, y_batch in dataloader:
            # Forward pass on mini-batch
            y_pred = model(X_batch)
            
            # Compute loss for mini-batch
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
            
            model.zero_grad()
    
    return model
```

### PyTorch DataLoader

```python
from torch.utils.data import TensorDataset, DataLoader

# Create dataset
dataset = TensorDataset(X, y)

# Create DataLoader with mini-batches
train_loader = DataLoader(
    dataset,
    batch_size=64,        # Mini-batch size
    shuffle=True,         # Shuffle each epoch
    num_workers=4,        # Parallel data loading
    pin_memory=True       # Faster GPU transfer
)

# Training loop
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        # Training step on mini-batch
        ...
```

### Characteristics

**Pros**:

- Balances gradient accuracy and computation
- Leverages GPU parallelism efficiently
- Moderate noise (enough for exploration, not too much)
- Most practical for real-world problems

**Cons**:

- Introduces batch_size hyperparameter
- Gradient is still an estimate (not exact)
- Batch statistics can vary (affects BatchNorm)

## Comparison

### Convergence Behavior

```
Loss
  │
  │╲  Batch GD (smooth)
  │ ╲__________
  │
  │╱╲
  │  ╲╱╲╱╲_____  Mini-batch GD (moderate noise)
  │
  │╱╲╱╲
  │   ╲╱╲╱╲╱╲_  SGD (noisy)
  │
  └──────────────────→ Iterations
```

### Computational Comparison

For dataset of size $N = 10,000$:

| Metric | Batch GD | SGD | Mini-batch ($B=64$) |
|--------|----------|-----|-------------------|
| Samples per update | 10,000 | 1 | 64 |
| Updates per epoch | 1 | 10,000 | 156 |
| Memory per step | High | Low | Medium |
| GPU utilization | Good | Poor | Excellent |
| Gradient variance | 0 | Very high | Moderate |

### Empirical Example

```python
import time

# Setup
n_samples = 10000
X = torch.randn(n_samples, 10)
y = torch.randn(n_samples, 1)

# Time comparison for 10 epochs
# Results will vary by hardware

# Batch GD
# Time: ~0.3s, Final loss: 1.0012

# SGD (1 sample)
# Time: ~45s, Final loss: 1.0089

# Mini-batch (batch_size=64)
# Time: ~0.8s, Final loss: 1.0015
```

## Batch Size Selection

### General Guidelines

| Batch Size | Use Case |
|------------|----------|
| 1 | True SGD (rarely used in practice) |
| 8-32 | Memory-constrained, good generalization |
| 64-256 | Common default range |
| 512-2048 | Large-scale training, requires LR scaling |
| 4096+ | Distributed training (with careful tuning) |

### Factors Affecting Choice

1. **GPU Memory**: Larger batches require more memory
2. **Generalization**: Smaller batches often generalize better
3. **Convergence Speed**: Larger batches can use larger LR
4. **Hardware Efficiency**: Batch size should be power of 2

### Powers of 2

```python
# Good: Powers of 2 for hardware efficiency
batch_sizes = [16, 32, 64, 128, 256, 512]

# Avoid: Arbitrary sizes
# batch_size = 100  # Less efficient
```

## Gradient Variance Analysis

### Variance Reduction

The variance of the mini-batch gradient estimate:

$$\text{Var}[\nabla_\theta L_B] = \frac{\sigma^2}{B}$$

where $\sigma^2$ is the variance of single-sample gradients.

**Implications**:

- Doubling batch size halves gradient variance
- Variance approaches zero as $B \to N$ (Batch GD)
- Diminishing returns: $B=64$ vs $B=128$ is less impactful than $B=1$ vs $B=64$

### Noise and Generalization

Research has shown that gradient noise from small batches can lead to:

- Flatter minima (better generalization)
- Implicit regularization
- Escape from sharp local minima

> "The noise in SGD acts as a regularizer." — Keskar et al., 2017

## Implementation Details

### Shuffling

**Why shuffle?**

- Prevents learning order-dependent patterns
- Ensures each mini-batch is representative
- Improves generalization

```python
# PyTorch handles shuffling automatically
DataLoader(dataset, shuffle=True)

# Manual shuffling
indices = torch.randperm(len(dataset))
shuffled_data = data[indices]
```

### Handling the Last Batch

When $N$ is not divisible by $B$:

```python
# Option 1: Drop incomplete batch
DataLoader(dataset, batch_size=64, drop_last=True)

# Option 2: Keep incomplete batch (default)
DataLoader(dataset, batch_size=64, drop_last=False)
```

### Gradient Accumulation

Simulate larger batch sizes with limited memory:

```python
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

for i, (X_batch, y_batch) in enumerate(train_loader):
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Practical Training Loop

### Complete Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Data
X_train = torch.randn(1000, 10)
y_train = torch.randn(1000, 1)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    epoch_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        # Forward
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Update
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
```

## Key Takeaways

1. **Batch GD**: Exact gradient, slow, stable convergence
2. **SGD**: Fast iterations, noisy, good exploration
3. **Mini-batch**: Best trade-off, standard in practice
4. **Batch size matters**: Affects speed, memory, and generalization
5. **Use DataLoader**: Handles batching, shuffling, parallelization
6. **Start with 32-64**: Reasonable default for most problems
7. **Gradient noise helps**: Some noise is beneficial for generalization

## Connections to Other Topics

- **Learning Rate**: Interacts with batch size, see [Learning Rate](learning_rate.md)
- **Momentum**: Reduces effective noise, see [Classical Momentum](../../ch02/optimizers/classical_momentum.md)
- **Batch Normalization**: Statistics depend on batch size, see [Batch Normalization](../../ch02/normalization/batch_norm.md)
- **Distributed Training**: Requires careful batch size scaling

## Exercises

1. **Compare variants**: Train the same model using Batch GD, SGD, and Mini-batch GD. Plot loss curves and compare:
   - Convergence speed (wall-clock time)
   - Final loss achieved
   - Loss curve smoothness

2. **Batch size sweep**: Train with batch sizes [8, 16, 32, 64, 128, 256, 512]. Plot:
   - Training time per epoch vs. batch size
   - Final validation accuracy vs. batch size

3. **Variance analysis**: For a simple linear regression problem, compute and plot the variance of gradient estimates for different batch sizes.

4. **Memory analysis**: Measure GPU memory usage for different batch sizes. Find the maximum batch size your GPU can handle.

5. **Gradient accumulation**: Implement gradient accumulation to simulate batch_size=256 using only batch_size=32. Verify the results match.

## References

- Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. COMPSTAT.
- Keskar, N. S., et al. (2017). On large-batch training for deep learning: Generalization gap and sharp minima. ICLR.
- Smith, S. L., et al. (2018). Don't decay the learning rate, increase the batch size. ICLR.
- Goyal, P., et al. (2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour.
