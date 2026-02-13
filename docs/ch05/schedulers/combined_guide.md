# Combined Usage Guide

This guide shows you how to leverage both PyTorch's built-in schedulers and custom advanced implementations together.

## Table of Contents

1. [Understanding the Two Approaches](#understanding-the-two-approaches)
2. [When to Use Which](#when-to-use-which)
3. [Integration Patterns](#integration-patterns)
4. [Real-World Examples](#real-world-examples)
5. [Performance Tips](#performance-tips)

## Understanding the Two Approaches

### PyTorch Built-in Schedulers (via `scheduler.py`)

**Characteristics:**
- Part of torch.optim.lr_scheduler
- Well-tested and optimized
- Integrated with PyTorch ecosystem
- Step-based scheduling (typically epoch-level)

**Pros:**
- Easy to use with existing code
- Official support and documentation
- Compatible with all PyTorch optimizers
- Battle-tested in production

**Cons:**
- Less flexibility for custom behavior
- Some schedulers update per epoch, not per step
- Limited control over mathematical details

### Custom Schedulers (via `scheduler/custom/`)

**Characteristics:**
- Mathematical implementations from papers
- Step-level control (batch-level updates)
- Full customization possible
- Educational and transparent

**Pros:**
- Complete control over behavior
- Implement paper methods exactly
- Easy to modify and extend
- Great for research and experiments

**Cons:**
- Need manual LR updates in training loop
- Requires understanding of implementation
- No automatic PyTorch integration

## When to Use Which

### Use PyTorch Built-in When:

1. **Standard Training Patterns**
   ```bash
   # Simple classification task
   python scheduler.py --scheduler step --epochs 100
   ```

2. **Production Code**
   - Stability is critical
   - Need official support
   - Working with PyTorch Lightning, Ignite, etc.

3. **Epoch-Level Scheduling is Sufficient**
   - Small datasets with few batches per epoch
   - Not doing aggressive batch-level scheduling

### Use Custom Schedulers When:

1. **Implementing Papers**
   ```python
   # Exact implementation of warmup + cosine decay from paper
   from scheduler.custom import WarmupWithDecay
   scheduler = WarmupWithDecay(5e-4, 10000, 100000, min_lr=1e-6)
   ```

2. **Transformer/BERT-Style Training**
   - Need precise warmup control
   - Batch-level learning rate updates
   - Following paper specifications exactly

3. **Research and Experimentation**
   - Testing new scheduling strategies
   - Need to modify scheduler behavior
   - Want to understand internals

4. **Batch-Level Updates Required**
   - Large datasets with many batches
   - Cyclical learning rates
   - OneCycle policy

## Integration Patterns

### Pattern 1: Custom Warmup → PyTorch Scheduler

Warmup with custom implementation, then use PyTorch for decay:

```python
from scheduler.custom import LinearWarmup
from torch.optim.lr_scheduler import CosineAnnealingLR

# Setup
warmup_steps = 1000
base_lr = 1e-3
total_epochs = 100

warmup_scheduler = LinearWarmup(base_lr, warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

# Training loop
global_step = 0
for epoch in range(total_epochs):
    for batch in dataloader:
        # Apply warmup during initial steps
        if global_step < warmup_steps:
            lr = warmup_scheduler.get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Training step
        loss = train_step(batch)
        global_step += 1
    
    # After warmup, step the cosine scheduler once per epoch
    if global_step >= warmup_steps:
        cosine_scheduler.step()
```

**Best for:** Combining precise warmup control with PyTorch's stable decay

### Pattern 2: All-in-One Custom Scheduler

Use a single custom scheduler that handles everything:

```python
from scheduler.custom import WarmupWithDecay

# Single scheduler for entire training
total_steps = len(dataloader) * epochs
scheduler = WarmupWithDecay(
    base_lr=5e-4,
    warmup_steps=10000,
    total_steps=total_steps,
    min_lr=1e-6
)

# Training loop - very simple
for step in range(total_steps):
    lr = scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Training step
    loss = train_step(batch)
```

**Best for:** Transformer training, papers with specific schedules, maximum control

### Pattern 3: Compare Implementations

Test PyTorch vs Custom side-by-side:

```python
from torch.optim.lr_scheduler import OneCycleLR as PyTorchOneCycle
from scheduler.custom import OneCycleLR as CustomOneCycle

# PyTorch version
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1)
torch_sched = PyTorchOneCycle(
    optimizer1,
    max_lr=0.1,
    steps_per_epoch=len(dataloader),
    epochs=10
)

# Custom version
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
custom_sched = CustomOneCycle(
    max_lr=0.1,
    total_steps=len(dataloader) * 10
)

# Train both and compare
for step, batch in enumerate(dataloader * epochs):
    # PyTorch model
    train_step_1(model1, batch, optimizer1)
    torch_sched.step()
    
    # Custom model
    lr = custom_sched.get_lr(step)
    for pg in optimizer2.param_groups:
        pg['lr'] = lr
    train_step_2(model2, batch, optimizer2)
```

**Best for:** Understanding differences, validation, research

### Pattern 4: Hybrid Approach

Mix PyTorch and custom schedulers strategically:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scheduler.custom import ExponentialWarmup

# Warmup with custom implementation
warmup = ExponentialWarmup(base_lr=1e-3, warmup_steps=500)

# After warmup, use PyTorch's adaptive scheduler
plateau = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)

# Training loop
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        global_step = epoch * len(dataloader) + step
        
        # Warmup phase
        if global_step < 500:
            lr = warmup.get_lr(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        
        train_step(batch)
    
    # After warmup, use adaptive scheduling
    if epoch >= (500 // len(dataloader)):
        val_loss = validate()
        plateau.step(val_loss)
```

**Best for:** Combining precise warmup with adaptive decay

## Real-World Examples

### Example 1: Image Classification (ResNet-50)

**Scenario:** Training ResNet-50 on ImageNet

```python
# Use PyTorch's MultiStepLR - standard approach
from torch.optim.lr_scheduler import MultiStepLR

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],  # Reduce at these epochs
    gamma=0.1
)

# Training loop
for epoch in range(100):
    train_epoch(model, train_loader, optimizer)
    validate(model, val_loader)
    scheduler.step()
```

**Why PyTorch built-in?** Standard training recipe, epoch-level updates sufficient

### Example 2: BERT Pretraining

**Scenario:** Pretraining BERT-base model

```python
# Use custom WarmupWithDecay - matches BERT paper
from scheduler.custom import WarmupWithDecay

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# BERT paper specifies exact warmup + decay schedule
total_steps = 1_000_000
warmup_steps = 10_000

scheduler = WarmupWithDecay(
    base_lr=1e-4,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    min_lr=0
)

# Training loop - step-level updates
for step in range(total_steps):
    lr = scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    batch = next(dataloader)
    loss = train_step(model, batch, optimizer)
```

**Why custom?** Paper specifies exact schedule, need step-level control

### Example 3: Small Dataset Training

**Scenario:** Training on CIFAR-10 with limited data

```python
# Use custom CyclicLR - helps with small datasets
from scheduler.custom import CyclicLR

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9
)

# Cycle every 4 epochs, decreasing amplitude
step_size = len(train_loader) * 4
scheduler = CyclicLR(
    base_lr=1e-4,
    max_lr=1e-1,
    step_size=step_size,
    mode='triangular2'
)

# Training loop - batch-level updates
global_step = 0
for epoch in range(200):
    for batch in train_loader:
        lr = scheduler.get_lr(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        train_step(model, batch, optimizer)
        global_step += 1
```

**Why custom?** Cyclical LR helps escape local minima, batch-level updates

### Example 4: Fast Training with 1cycle

**Scenario:** Train quickly with limited time

**Option A: PyTorch (easier)**
```python
from torch.optim.lr_scheduler import OneCycleLR

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,
    steps_per_epoch=len(train_loader),
    epochs=10,
    pct_start=0.3
)

for epoch in range(10):
    for batch in train_loader:
        train_step(model, batch, optimizer)
        scheduler.step()  # Step every batch
```

**Option B: Custom (more control)**
```python
from scheduler.custom import OneCycleLR

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

total_steps = len(train_loader) * 10
scheduler = OneCycleLR(
    max_lr=0.1,
    total_steps=total_steps,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=1e4
)

for step in range(total_steps):
    lr = scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    batch = get_batch()
    train_step(model, batch, optimizer)
```

**Both work well!** Choose based on preference and integration needs

### Example 5: Long Training with Restarts

**Scenario:** Training for 500 epochs, want periodic restarts

```python
# Use custom SGDR - not in PyTorch (they have CosineAnnealingWarmRestarts though)
from scheduler.custom import CosineAnnealingWarmRestarts

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# First restart after 50 epochs, then double each time
t_0 = len(train_loader) * 50
scheduler = CosineAnnealingWarmRestarts(
    max_lr=0.1,
    min_lr=1e-5,
    t_0=t_0,
    t_mult=2
)

# Training loop
global_step = 0
for epoch in range(500):
    for batch in train_loader:
        lr = scheduler.get_lr(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        train_step(model, batch, optimizer)
        global_step += 1
```

**Why custom?** More control over restart timing and behavior

## Performance Tips

### Tip 1: Minimize LR Updates

```python
# BAD: Update LR every step even during constant period
for step in range(100000):
    lr = scheduler.get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

# GOOD: Only update when needed
warmup_steps = 1000
for step in range(100000):
    if step < warmup_steps:
        lr = scheduler.get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
    elif step == warmup_steps:
        # Set final LR once
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr
```

### Tip 2: Cache Expensive Computations

```python
# For schedulers with expensive get_lr()
class CachedScheduler:
    def __init__(self, scheduler, total_steps):
        self.lrs = [scheduler.get_lr(i) for i in range(total_steps)]
    
    def get_lr(self, step):
        return self.lrs[step]

# Compute all LRs once
cached = CachedScheduler(scheduler, total_steps)

# Fast lookups during training
for step in range(total_steps):
    lr = cached.get_lr(step)  # O(1) lookup
```

### Tip 3: Batch LR Updates

```python
# Update all parameter groups at once
def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Use it
set_lr(optimizer, scheduler.get_lr(step))
```

### Tip 4: Visualize Before Training

```python
# Check your schedule before committing to long training
import matplotlib.pyplot as plt

steps = range(total_steps)
lrs = [scheduler.get_lr(step) for step in steps]

plt.figure(figsize=(12, 4))
plt.plot(steps, lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('LR Schedule Preview')
plt.grid(True)
plt.show()

# Make sure it looks right before training!
```

## Common Pitfalls

### Pitfall 1: Forgetting to Update LR

```python
# WRONG: Scheduler created but never used
scheduler = LinearWarmup(1e-3, 1000)
for step in range(10000):
    train_step()  # LR never changes!

# RIGHT: Actually update the LR
scheduler = LinearWarmup(1e-3, 1000)
for step in range(10000):
    lr = scheduler.get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    train_step()
```

### Pitfall 2: Wrong Step Count

```python
# WRONG: Step counts don't match
total_steps = 1000
scheduler = OneCycleLR(1e-2, total_steps=total_steps)
for step in range(5000):  # More steps than scheduler expects!
    lr = scheduler.get_lr(step)  # Will behave unexpectedly

# RIGHT: Match step counts
total_steps = len(dataloader) * epochs
scheduler = OneCycleLR(1e-2, total_steps=total_steps)
for step in range(total_steps):
    lr = scheduler.get_lr(step)
```

### Pitfall 3: Mixing Epoch and Step Schedulers

```python
# CONFUSING: Mixing epoch and step-level schedulers
warmup = LinearWarmup(1e-3, 1000)  # Step-level
step_lr = StepLR(optimizer, 30)     # Epoch-level

# Be clear about when each applies
if global_step < 1000:
    lr = warmup.get_lr(global_step)
    set_lr(optimizer, lr)
elif epoch_num % 30 == 0:
    step_lr.step()
```

### Pitfall 4: Not Considering Warmup Duration

```python
# BAD: Warmup too short for large batch size
warmup_steps = 100  # Only 100 steps!
batch_size = 8192   # Very large batch
scheduler = LinearWarmup(1e-3, warmup_steps)

# GOOD: Scale warmup with batch size
base_warmup = 1000
warmup_steps = base_warmup * (batch_size // 32)
scheduler = LinearWarmup(1e-3, warmup_steps)
```

## Debugging Tips

### Check LR Values

```python
# Print LR at key points
print(f"Initial LR: {scheduler.get_lr(0)}")
print(f"After warmup: {scheduler.get_lr(warmup_steps)}")
print(f"Mid-training: {scheduler.get_lr(total_steps // 2)}")
print(f"Final LR: {scheduler.get_lr(total_steps - 1)}")
```

### Verify Optimizer LR

```python
# Check optimizer's actual LR
current_lr = optimizer.param_groups[0]['lr']
expected_lr = scheduler.get_lr(step)
assert abs(current_lr - expected_lr) < 1e-10, "LR mismatch!"
```

### Plot Actual vs Expected

```python
# Track actual LRs during training
actual_lrs = []
for step in range(total_steps):
    lr = scheduler.get_lr(step)
    set_lr(optimizer, lr)
    actual_lrs.append(optimizer.param_groups[0]['lr'])
    train_step()

# Plot to verify
plt.plot(actual_lrs)
plt.title("Actual LR During Training")
plt.show()
```

## Summary

**Quick Decision Guide:**

- **Standard CV/NLP task?** → PyTorch built-in (StepLR, CosineAnnealing)
- **Following a paper's schedule?** → Custom implementation
- **Need batch-level updates?** → Custom schedulers
- **Transformer training?** → Custom WarmupWithDecay
- **Fast experimentation?** → PyTorch OneCycleLR
- **Deep understanding needed?** → Custom implementations
- **Production deployment?** → PyTorch built-in

**Remember:**
- Both approaches are valid and useful
- PyTorch schedulers are great for standard use cases
- Custom schedulers give you full control
- You can mix and match as needed
- Always visualize your schedule before training

Happy training! 🚀
