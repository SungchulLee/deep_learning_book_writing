# Exponential Decay

Exponential decay reduces the learning rate by a constant multiplicative factor each epoch, providing smooth, continuous decay without the discontinuities of step-based schedules.

## Mathematical Definition

$$\eta_t = \eta_0 \cdot \gamma^t$$

where:
- $\eta_0$ is the initial learning rate
- $\gamma$ is the decay factor (typically 0.9-0.99)
- $t$ is the epoch number

After $t$ epochs, the learning rate is multiplied by $\gamma^t$.

## Decay Behavior

| Gamma | LR after 10 epochs | LR after 50 epochs | LR after 100 epochs |
|-------|-------------------|-------------------|---------------------|
| 0.99 | 90.4% | 60.5% | 36.6% |
| 0.95 | 59.9% | 7.7% | 0.6% |
| 0.90 | 34.9% | 0.5% | ~0% |

Lower $\gamma$ values produce more aggressive decay.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Multiply LR by gamma every epoch
scheduler = ExponentialLR(
    optimizer,
    gamma=0.95  # LR = LR * 0.95 each epoch
)

# Training loop
for epoch in range(100):
    # ... training code ...
    
    scheduler.step()  # Update learning rate
    
    if epoch % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: LR = {current_lr:.6f}")
```

## Visualizing the Schedule

```python
import matplotlib.pyplot as plt

def plot_exponential_decay(gamma_values, epochs=100):
    plt.figure(figsize=(10, 5))
    
    for gamma in gamma_values:
        optimizer = optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        
        lrs = []
        for _ in range(epochs):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        plt.plot(lrs, label=f'γ = {gamma}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Exponential Decay Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()

plot_exponential_decay([0.99, 0.95, 0.90])
```

## Comparison with Step Decay

| Aspect | Exponential | Step |
|--------|-------------|------|
| Curve | Smooth, continuous | Staircase |
| Configuration | Single gamma | Step size + gamma |
| Early decay | Faster | Slower (waits for milestone) |
| Late decay | Slower (asymptotic) | Discrete drops |
| Predictability | Every epoch changes | Fixed milestones |

```python
# Comparison
initial_lr = 0.1
epochs = 100

# Exponential: gamma=0.95
exp_scheduler = ExponentialLR(optimizer, gamma=0.95)

# Step: equivalent total decay
step_scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

# Exponential decays smoothly
# Step holds constant, then drops
```

## Choosing Gamma

### By Training Length

| Training Length | Suggested Gamma | Final LR (% of initial) |
|-----------------|-----------------|-------------------------|
| 50 epochs | 0.95 | ~7.7% |
| 100 epochs | 0.97 | ~4.8% |
| 200 epochs | 0.98 | ~1.8% |
| 500 epochs | 0.99 | ~0.7% |

### Target-Based Calculation

To decay from $\eta_0$ to $\eta_{final}$ over $T$ epochs:

$$\gamma = \left(\frac{\eta_{final}}{\eta_0}\right)^{1/T}$$

```python
def calculate_gamma(initial_lr, final_lr, epochs):
    """Calculate gamma to achieve target final LR."""
    return (final_lr / initial_lr) ** (1 / epochs)

# Example: decay from 0.1 to 0.001 over 100 epochs
gamma = calculate_gamma(0.1, 0.001, 100)
print(f"Gamma needed: {gamma:.4f}")  # ~0.9550
```

## Practical Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

# Model and data
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
X = torch.randn(200, 10)
y = X.sum(dim=1, keepdim=True) + torch.randn(200, 1) * 0.1

# Setup
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.97)
criterion = nn.MSELoss()

# Training
for epoch in range(100):
    pred = model(X)
    loss = criterion(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 20 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, LR = {lr:.6f}")
```

## When to Use Exponential Decay

**Good choices:**
- Long training runs (100+ epochs)
- When smooth decay is preferred over discrete drops
- Simple, single-parameter configuration
- Continuous fine-tuning throughout training

**Consider alternatives when:**
- Short training (step or cosine may be better)
- You want most decay to happen late (cosine)
- You need adaptive scheduling (ReduceLROnPlateau)
- Standard benchmarks use step decay

## Combining with Warmup

```python
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ExponentialLR

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Warmup for 10 epochs
warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)

# Then exponential decay
exp_decay = ExponentialLR(optimizer, gamma=0.98)

# Combine
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, exp_decay],
    milestones=[10]
)
```

## Limitations

1. **Can decay too fast**: With aggressive gamma, LR can become negligible
2. **No adaptation**: Decay happens regardless of training progress
3. **Asymptotic behavior**: Never reaches exactly zero
4. **Less common**: Step and cosine are more standard in benchmarks

## Key Takeaways

Exponential decay provides smooth, continuous learning rate reduction controlled by a single parameter $\gamma$. Choose $\gamma$ based on training length and desired final learning rate. It offers simpler configuration than step decay but is less commonly used in standard benchmarks. For most modern applications, cosine annealing has become preferred due to its natural "spend more time at intermediate LRs" behavior, but exponential decay remains a solid, intuitive choice.
