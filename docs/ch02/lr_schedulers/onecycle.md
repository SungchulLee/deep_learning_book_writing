# OneCycleLR

OneCycleLR implements the 1cycle learning rate policy, which uses a single cycle of learning rate that first increases then decreases, combined with momentum that moves inversely. This approach often achieves faster convergence and better final performance than traditional schedules.

## The 1Cycle Policy

The 1cycle policy, introduced by Leslie Smith, consists of:

1. **Warmup phase**: LR increases from `initial_lr` to `max_lr`
2. **Annealing phase**: LR decreases from `max_lr` to `final_lr` (much lower than initial)
3. **Momentum**: Moves inversely to LR (decreases during warmup, increases during annealing)

```
Learning Rate:
     max_lr
       /\
      /  \
     /    \
    /      \____
initial     final (very low)

Momentum:
base_momentum
    \      /‾‾‾‾
     \    /
      \  /
       \/
   max_momentum
```

## Why It Works

The intuition behind 1cycle:

1. **Start low**: Allow model to find a good region
2. **Increase LR**: Explore aggressively, escape local minima
3. **Decrease LR**: Fine-tune in the discovered good region
4. **Very low final LR**: Polish the solution

The inverse momentum schedule:
- High momentum when LR is low → stable updates
- Low momentum when LR is high → allow exploration

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# OneCycleLR scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,              # Peak learning rate
    total_steps=1000,        # Total training steps
    # OR use epochs and steps_per_epoch:
    # epochs=10,
    # steps_per_epoch=100,
    pct_start=0.3,           # Fraction of cycle spent increasing LR
    anneal_strategy='cos',   # 'cos' or 'linear'
    div_factor=25,           # initial_lr = max_lr / div_factor
    final_div_factor=1e4,    # final_lr = initial_lr / final_div_factor
)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_lr` | Required | Peak learning rate |
| `total_steps` | Required* | Total number of training steps |
| `epochs` | None | Alternative to total_steps |
| `steps_per_epoch` | None | Required if using epochs |
| `pct_start` | 0.3 | Fraction for LR increase phase |
| `anneal_strategy` | 'cos' | 'cos' or 'linear' for decrease phase |
| `div_factor` | 25 | initial_lr = max_lr / 25 |
| `final_div_factor` | 1e4 | final_lr = initial_lr / 10000 |

## Learning Rate Calculation

```python
# With default parameters:
max_lr = 0.1
div_factor = 25
final_div_factor = 1e4

initial_lr = max_lr / div_factor           # 0.1 / 25 = 0.004
final_lr = initial_lr / final_div_factor   # 0.004 / 10000 = 4e-7

print(f"Initial LR: {initial_lr}")
print(f"Max LR: {max_lr}")
print(f"Final LR: {final_lr}")
```

## Step-Level vs Epoch-Level

**Important**: OneCycleLR is typically called every **step** (batch), not every epoch:

```python
# Correct usage: step after each batch
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step after EACH batch
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

# Data
torch.manual_seed(42)
X = torch.randn(1000, 10)
y = X.sum(dim=1, keepdim=True) + torch.randn(1000, 1) * 0.1

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Setup
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.MSELoss()

epochs = 10
steps_per_epoch = len(train_loader)

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch
)

# Training
lr_history = []
for epoch in range(epochs):
    epoch_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        lr_history.append(optimizer.param_groups[0]['lr'])
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss = {epoch_loss/len(train_loader):.4f}")

# Visualize LR schedule
import matplotlib.pyplot as plt
plt.plot(lr_history)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('OneCycleLR Schedule')
plt.show()
```

## Finding max_lr

The 1cycle policy works best when `max_lr` is set appropriately. Use the learning rate finder:

```python
def find_lr(model, train_loader, optimizer, criterion, 
            start_lr=1e-7, end_lr=10, num_iter=100):
    """Simple learning rate finder."""
    lrs, losses = [], []
    
    # Exponential LR increase
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    lr = start_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    for i, (data, target) in enumerate(train_loader):
        if i >= num_iter:
            break
            
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        lrs.append(lr)
        losses.append(loss.item())
        
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return lrs, losses

# Plot and find the LR where loss decreases fastest
# max_lr should be slightly before the minimum loss point
```

## Comparison with Other Schedules

| Schedule | Configuration | Warmup | Best For |
|----------|---------------|--------|----------|
| OneCycleLR | max_lr, total_steps | Built-in | Fast training |
| Step | milestones, gamma | Manual | Standard benchmarks |
| Cosine | T_max, eta_min | Manual | Modern architectures |
| ReduceLROnPlateau | patience, factor | Manual | Unknown dynamics |

## Benefits of OneCycleLR

1. **Super-convergence**: Can train in fewer epochs
2. **Built-in warmup**: No separate warmup scheduler needed
3. **Regularization effect**: Large LR acts as regularizer
4. **Momentum scheduling**: Automatic inverse momentum

## When to Use OneCycleLR

**Excellent for:**
- Fast training experiments
- When you want to minimize epochs
- CNNs (where it was originally developed)
- Transfer learning / fine-tuning

**Consider alternatives when:**
- Following established benchmark recipes
- Very long training (standard schedules may be more stable)
- You need precise control over schedule shape

## Key Takeaways

OneCycleLR implements the 1cycle policy with learning rate that increases then decreases, combined with inverse momentum scheduling. It enables "super-convergence"—achieving good results in fewer epochs. Call `scheduler.step()` after each batch, not each epoch. Use learning rate finder to determine optimal `max_lr`. It provides built-in warmup and often acts as a regularizer through the high LR phase.
