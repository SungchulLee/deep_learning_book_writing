# Cyclic Learning Rates

Cyclic learning rates (CLR) oscillate the learning rate between bounds rather than monotonically decreasing. This approach can escape saddle points, find wider minima, and sometimes achieve faster convergence than traditional schedules.

## Motivation

Traditional schedules assume learning rate should only decrease. However:

1. **Saddle points**: Increasing LR can help escape flat regions
2. **Local minima**: LR spikes can jump out of sharp minima
3. **Exploration vs exploitation**: Cycles balance both
4. **Wider minima**: Oscillation tends to find flatter, more generalizable solutions

## Cyclic LR Concept

```
Learning Rate
     ↑
max  │    /\      /\      /\
     │   /  \    /  \    /  \
     │  /    \  /    \  /    \
min  │ /      \/      \/      \
     └──────────────────────────→ Steps
       ←cycle→
```

## CyclicLR in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Cyclic LR with triangular policy
scheduler = CyclicLR(
    optimizer,
    base_lr=0.001,        # Minimum LR
    max_lr=0.1,           # Maximum LR
    step_size_up=2000,    # Steps to increase from base to max
    step_size_down=2000,  # Steps to decrease from max to base (optional)
    mode='triangular',    # Cycle shape
    cycle_momentum=True,  # Also cycle momentum
)
```

## Cycling Modes

### Triangular (Default)

Linear increase and decrease between bounds:

```python
scheduler = CyclicLR(
    optimizer,
    base_lr=0.001,
    max_lr=0.1,
    mode='triangular'
)
```

```
LR
↑
  /\    /\    /\
 /  \  /  \  /  \
/    \/    \/    \
```

### Triangular2

Same as triangular, but max_lr is halved each cycle:

```python
scheduler = CyclicLR(
    optimizer,
    base_lr=0.001,
    max_lr=0.1,
    mode='triangular2'
)
```

```
LR
↑
  /\
 /  \  /\
/    \/  \/\
          \/\
```

### Exp_range

Exponential decay of max_lr each cycle:

```python
scheduler = CyclicLR(
    optimizer,
    base_lr=0.001,
    max_lr=0.1,
    mode='exp_range',
    gamma=0.99994  # max_lr *= gamma each step
)
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `base_lr` | Minimum learning rate |
| `max_lr` | Maximum learning rate |
| `step_size_up` | Steps to go from base to max |
| `step_size_down` | Steps to go from max to base (default: step_size_up) |
| `mode` | 'triangular', 'triangular2', or 'exp_range' |
| `gamma` | Decay factor for exp_range mode |
| `cycle_momentum` | Whether to cycle momentum inversely |

## Step Size Guidelines

The original paper recommends:

$$\text{step\_size} = 2 \times \text{epochs\_per\_cycle} \times \text{steps\_per\_epoch}$$

For typical training:
- **2-10 epochs per cycle** is common
- Fewer cycles (2-4 total) often works well

```python
# Example: 4 epochs per cycle, 100 batches per epoch
steps_per_epoch = 100
epochs_per_cycle = 4
step_size = 2 * epochs_per_cycle * steps_per_epoch  # 800

scheduler = CyclicLR(
    optimizer,
    base_lr=0.001,
    max_lr=0.1,
    step_size_up=step_size // 2  # 400 steps up
)
```

## Complete Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset

# Data
torch.manual_seed(42)
X = torch.randn(1000, 10)
y = (X.sum(dim=1, keepdim=True) > 0).float()

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# Setup
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.BCELoss()

steps_per_epoch = len(train_loader)
scheduler = CyclicLR(
    optimizer,
    base_lr=0.0001,
    max_lr=0.01,
    step_size_up=steps_per_epoch * 2,  # 2 epochs up
    mode='triangular2',
    cycle_momentum=True,
    base_momentum=0.8,
    max_momentum=0.9
)

# Training
lr_history = []
for epoch in range(20):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step every batch!
        
        lr_history.append(optimizer.param_groups[0]['lr'])
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, "
              f"LR = {optimizer.param_groups[0]['lr']:.6f}")
```

## Finding LR Range

Use the LR range test to find appropriate bounds:

```python
def lr_range_test(model, train_loader, optimizer, criterion,
                  start_lr=1e-7, end_lr=1, num_iter=100):
    """Find good base_lr and max_lr for cyclic scheduling."""
    lrs, losses = [], []
    lr = start_lr
    mult = (end_lr / start_lr) ** (1 / num_iter)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    for i, (data, target) in enumerate(train_loader):
        if i >= num_iter:
            break
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        if torch.isnan(loss) or loss.item() > 4 * losses[0] if losses else False:
            break
            
        loss.backward()
        optimizer.step()
        
        lrs.append(lr)
        losses.append(loss.item())
        
        lr *= mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Plot to find:
    # base_lr: where loss starts decreasing
    # max_lr: where loss is still decreasing (before explosion)
    return lrs, losses
```

**Guidelines from the plot:**
- `base_lr`: LR where loss starts decreasing significantly
- `max_lr`: ~10× base_lr, or where loss is still decreasing

## Cyclic vs OneCycle

| Aspect | CyclicLR | OneCycleLR |
|--------|----------|------------|
| Cycles | Multiple | Single |
| Use case | Longer training | Fast convergence |
| LR pattern | Oscillating | Rise then fall |
| Final LR | Continues cycling | Very low |
| Popularity | Less common now | More popular |

OneCycleLR is often preferred for its simplicity and super-convergence properties.

## When to Use Cyclic LR

**Good choices:**
- Exploratory training
- When stuck in local minima
- Ensemble methods (snapshot ensembles)
- Long training where multiple restarts help

**Consider alternatives when:**
- Short training runs (use OneCycleLR)
- Well-established benchmark protocols
- Transformers (use cosine with warmup)

## Snapshot Ensembles

Cyclic LR enables "free" ensembles by saving models at cycle minima:

```python
# Save model at end of each cycle
snapshots = []
for epoch in range(num_epochs):
    train_epoch(...)
    scheduler.step()
    
    # Check if at cycle minimum (base_lr)
    current_lr = optimizer.param_groups[0]['lr']
    if abs(current_lr - base_lr) < 1e-8:
        snapshots.append(copy.deepcopy(model.state_dict()))

# Ensemble predictions from all snapshots
```

## Key Takeaways

Cyclic learning rates oscillate LR between bounds, enabling exploration of the loss landscape and escape from local minima. Use triangular2 or exp_range modes for decaying cycles. Step the scheduler every batch, not every epoch. The LR range test helps find appropriate base_lr and max_lr values. While CyclicLR is useful for specific scenarios (snapshot ensembles, escaping local minima), OneCycleLR has become more popular for general fast training.
