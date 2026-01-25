# Optimizer Fundamentals

## Overview

Optimization lies at the heart of deep learning. The training process seeks to minimize a loss function $\mathcal{L}(\theta)$ with respect to model parameters $\theta \in \mathbb{R}^d$. Optimizers are the algorithms that update model parameters to minimize the loss function. Without an optimizer, a neural network cannot learn—it's the engine that transforms gradient information into parameter improvements.

## The Optimization Problem

Given a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ and a parameterized model $f_\theta$, we seek:

$$\theta^* = \arg\min_\theta \mathcal{L}(\theta) = \arg\min_\theta \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)$$

where $\ell$ is a per-sample loss function (e.g., cross-entropy, MSE).

## The Optimization Loop

Training a neural network follows a cyclic pattern:

1. **Forward pass**: Model produces predictions from inputs
2. **Loss computation**: Measure error between predictions and targets
3. **Backward pass**: Compute gradients via automatic differentiation
4. **Parameter update**: Optimizer uses gradients to adjust parameters
5. **Repeat** until convergence

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Setup
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    predictions = model(inputs)           # Forward
    loss = criterion(predictions, targets) # Loss
    optimizer.zero_grad()                  # Clear gradients
    loss.backward()                        # Compute gradients
    optimizer.step()                       # Update parameters
```

## Gradient Descent: The Foundation

The simplest optimization approach follows the negative gradient direction:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

where $\eta > 0$ is the learning rate (step size).

### Mathematical Justification

The gradient $\nabla_\theta \mathcal{L}(\theta)$ points in the direction of steepest ascent. By Taylor expansion around $\theta_t$:

$$\mathcal{L}(\theta_t + \Delta\theta) \approx \mathcal{L}(\theta_t) + \nabla_\theta \mathcal{L}(\theta_t)^\top \Delta\theta + \frac{1}{2} \Delta\theta^\top H \Delta\theta$$

For small steps where $\Delta\theta = -\eta \nabla_\theta \mathcal{L}(\theta_t)$:

$$\mathcal{L}(\theta_{t+1}) \approx \mathcal{L}(\theta_t) - \eta \|\nabla_\theta \mathcal{L}(\theta_t)\|^2$$

This guarantees descent when $\eta$ is sufficiently small.

## Stochastic Gradient Descent (SGD)

Computing the full gradient requires processing all $N$ samples, which is computationally prohibitive. SGD approximates the gradient using a mini-batch $\mathcal{B} \subset \mathcal{D}$:

$$g_t = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_\theta \ell(f_\theta(x_i), y_i)$$

The update becomes:

$$\theta_{t+1} = \theta_t - \eta g_t$$

### Properties of Stochastic Gradients

The mini-batch gradient is an unbiased estimator:

$$\mathbb{E}_\mathcal{B}[g_t] = \nabla_\theta \mathcal{L}(\theta_t)$$

However, it introduces variance:

$$\text{Var}(g_t) = \frac{\sigma^2}{|\mathcal{B}|}$$

where $\sigma^2$ is the variance of individual sample gradients.

### Basic SGD Implementation

```python
# Create model and optimizer
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("Initial parameters:")
print(f"  Weight: {model.weight.item():.4f}")
print(f"  Bias: {model.bias.item():.4f}")
```

### Step-by-Step Training

```python
# Generate simple linear data: y = 2x + 1
torch.manual_seed(42)
X = torch.linspace(0, 10, 50).reshape(-1, 1)
y = 2 * X + 1 + torch.randn(50, 1) * 0.5
criterion = nn.MSELoss()

# Single training step - detailed
y_pred = model(X)                    # Step 1: Forward pass
loss = criterion(y_pred, y)          # Step 2: Compute loss
optimizer.zero_grad()                 # Step 3: Zero gradients
loss.backward()                       # Step 4: Backward pass

print(f"Gradients computed:")
print(f"  Weight gradient: {model.weight.grad.item():.4f}")
print(f"  Bias gradient: {model.bias.grad.item():.4f}")

optimizer.step()                      # Step 5: Update parameters

print(f"Parameters after update:")
print(f"  Weight: {model.weight.item():.4f}")
print(f"  Bias: {model.bias.item():.4f}")
```

The update follows: `new_weight = old_weight - learning_rate × gradient`

## Understanding Learning Rate

The learning rate $\eta$ controls the step size of parameter updates:

| Learning Rate | Behavior |
|--------------|----------|
| Too high (>0.1) | Overshoots optimum, may diverge |
| Too low (<0.0001) | Very slow convergence |
| Appropriate | Steady loss decrease |

```python
# Compare different learning rates
learning_rates = [0.001, 0.01, 0.1]

for lr in learning_rates:
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    for _ in range(100):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    print(f"LR {lr}: Final loss = {losses[-1]:.4f}")
```

## Complete Training Example

```python
# Reset model
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Results
print(f"\nLearned: y = {model.weight.item():.4f}x + {model.bias.item():.4f}")
print(f"Target:  y = 2.0x + 1.0")
```

## Challenges in Neural Network Optimization

### Loss Landscape Complexity

Neural network loss surfaces exhibit:

1. **Non-convexity**: Multiple local minima and saddle points
2. **Ill-conditioning**: Vastly different curvatures along different directions
3. **Ravines**: Long, narrow valleys that slow convergence
4. **Plateaus**: Regions with near-zero gradients

### The Condition Number Problem

For a quadratic loss $\mathcal{L}(\theta) = \frac{1}{2}\theta^\top H \theta$, the condition number is:

$$\kappa = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}$$

High condition numbers (common in deep learning) cause:
- Oscillation along high-curvature directions
- Slow progress along low-curvature directions
- Learning rate selection becomes critical

### Per-Parameter Learning Rate Needs

Different parameters may require different learning rates:
- Sparse features need larger updates when they appear
- Frequently updated parameters may need smaller rates
- Deep vs. shallow layers have different gradient scales

## The Evolution of Optimizers

The limitations of vanilla SGD motivated the development of advanced optimizers:

| Era | Optimizer | Key Innovation |
|-----|-----------|----------------|
| 1986 | Momentum | Accumulates velocity to accelerate convergence |
| 1983/2013 | NAG | Looks ahead before computing gradient |
| 2011 | AdaGrad | Per-parameter adaptive learning rates |
| 2012 | RMSprop | Exponential moving average of squared gradients |
| 2014 | Adam | Combines momentum with adaptive rates |
| 2017 | AdamW | Decouples weight decay from adaptive updates |
| 2018 | AMSGrad | Fixes Adam's non-convergence issues |
| 2019 | RAdam | Rectifies variance in early training |

## General Optimizer Framework

Most modern optimizers follow a general pattern:

```
Algorithm: General Optimizer
Input: Initial parameters θ₀, learning rate η, decay rates
Initialize: State variables (moments, caches)

for t = 1, 2, ... do
    g_t ← ∇_θ L(θ_{t-1})           # Compute gradient
    s_t ← UpdateState(s_{t-1}, g_t)  # Update optimizer state
    Δθ_t ← ComputeUpdate(s_t, g_t)   # Compute parameter update
    θ_t ← θ_{t-1} - η · Δθ_t         # Apply update
end for
```

## PyTorch Optimizer Interface

All PyTorch optimizers share a common interface:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model and loss
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
criterion = nn.CrossEntropyLoss()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
```

### Optimizer API Methods

```python
# Construction: register parameters and set hyperparameters
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Clear gradients (before backward)
optimizer.zero_grad()

# Update parameters (after backward)
optimizer.step()

# Access current learning rate
current_lr = optimizer.param_groups[0]['lr']

# Modify learning rate
optimizer.param_groups[0]['lr'] = 0.001
```

### Parameter Groups

Optimizers can apply different settings to different parameter groups:

```python
# Different learning rates for different layers
optimizer = optim.SGD([
    {'params': model.encoder.parameters(), 'lr': 0.001},
    {'params': model.decoder.parameters(), 'lr': 0.01}
], lr=0.005)  # Default LR for any ungrouped params
```

## The Importance of zero_grad()

PyTorch accumulates gradients by default. Without `zero_grad()`, gradients from previous iterations add up:

```python
# Demonstration of gradient accumulation
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for step in range(3):
    pred = model(X)
    loss = criterion(pred, y)
    # Forgetting zero_grad!
    loss.backward()
    print(f"Step {step+1}: Gradient = {model.weight.grad.item():.4f}")

# Gradients accumulate: 3x the correct value after 3 steps
```

**Warning:** Always call `optimizer.zero_grad()` before `loss.backward()`. Forgetting this causes gradients to accumulate across iterations, leading to incorrect updates and unstable training.

## Custom Optimizer Implementation

The base class structure for implementing custom optimizers:

```python
import torch
from torch.optim import Optimizer

class CustomOptimizer(Optimizer):
    """Template for custom optimizer implementation."""
    
    def __init__(self, params, lr=1e-3, **kwargs):
        defaults = dict(lr=lr, **kwargs)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize state on first call
                if len(state) == 0:
                    self._init_state(state, p)
                
                # Compute and apply update
                self._update_param(p, grad, state, group)
        
        return loss
    
    def _init_state(self, state, param):
        """Initialize optimizer state for a parameter."""
        raise NotImplementedError
    
    def _update_param(self, param, grad, state, group):
        """Compute and apply parameter update."""
        raise NotImplementedError
```

## Training Loop Template

A production-ready training loop structure:

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Enable training mode (dropout, batchnorm)
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Usage
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}")
```

## Convergence Analysis

### Convex Case

For a convex function with $L$-Lipschitz continuous gradients, gradient descent with $\eta \leq 1/L$ satisfies:

$$\mathcal{L}(\theta_T) - \mathcal{L}(\theta^*) \leq \frac{\|\theta_0 - \theta^*\|^2}{2\eta T}$$

This gives $O(1/T)$ convergence rate.

### Strongly Convex Case

If additionally $\mathcal{L}$ is $\mu$-strongly convex:

$$\|\theta_T - \theta^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^T \|\theta_0 - \theta^*\|^2$$

This gives linear (exponential) convergence rate.

### Non-Convex Case

For non-convex smooth functions, we can only guarantee convergence to a stationary point:

$$\min_{t \leq T} \|\nabla \mathcal{L}(\theta_t)\|^2 \leq \frac{2(\mathcal{L}(\theta_0) - \mathcal{L}^*)}{\eta T}$$

## Common Pitfalls

| Mistake | Symptom | Solution |
|---------|---------|----------|
| Forgetting `zero_grad()` | Gradients explode | Always call before `backward()` |
| Wrong order of operations | NaN loss or no learning | Follow: forward → loss → zero_grad → backward → step |
| Using `torch.no_grad()` during training | Model doesn't learn | Reserve for evaluation only |
| Learning rate too high | Loss increases or NaN | Start with 0.001-0.01 |
| Not calling `model.train()` | Dropout/BatchNorm disabled | Call before training loop |

## Key Concepts Summary

| Concept | Definition | Importance |
|---------|------------|------------|
| Learning Rate | Step size $\eta$ | Controls convergence speed vs. stability |
| Gradient | $\nabla_\theta \mathcal{L}$ | Direction of steepest ascent |
| Mini-batch | Subset of training data | Enables efficient computation |
| Momentum | Accumulated gradient direction | Accelerates through ravines |
| Adaptive Rate | Per-parameter scaling | Handles ill-conditioning |

## Key Takeaways

Optimizers transform gradients into parameter updates, with the learning rate controlling update magnitude. SGD is the simplest algorithm: $\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$. The training loop follows a strict order: forward → loss → zero_grad → backward → step. Always call `zero_grad()` to prevent gradient accumulation. Learning rate is the most critical hyperparameter—start with conservative values (0.001-0.01) and adjust based on loss curves.

## References

1. Ruder, S. (2016). "An Overview of Gradient Descent Optimization Algorithms"
2. Bottou, L., Curtis, F., & Nocedal, J. (2018). "Optimization Methods for Large-Scale Machine Learning"
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" - Chapter 8
