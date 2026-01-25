# Cosine Annealing

Cosine annealing smoothly decays the learning rate following a cosine curve, providing gradual reduction with natural acceleration near the end. It has become one of the most popular scheduling strategies for modern deep learning.

## Motivation

The learning rate dilemma during training:

- **Early training**: Need high LR to explore and escape poor regions
- **Mid training**: Need moderate LR for steady progress  
- **Late training**: Need low LR for fine-grained optimization

Cosine annealing addresses this with a smooth, non-linear decay that spends more iterations at intermediate learning rates.

## Mathematical Formulation

For a cycle of $T_{\max}$ epochs:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t \cdot \pi}{T_{\max}}\right)\right)$$

Properties of this schedule:
- Starts at $\eta_{\max}$
- Ends at $\eta_{\min}$
- Decreases slowly initially, faster in the middle, slowly again near the end
- Smooth, differentiable curve

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Cosine annealing over 100 epochs
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,       # Number of iterations for one cycle
    eta_min=0.001    # Minimum learning rate
)

# Training loop
for epoch in range(100):
    # ... training code ...
    
    scheduler.step()  # Update learning rate
    
    current_lr = optimizer.param_groups[0]['lr']
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: LR = {current_lr:.6f}")
```

## Visualizing the Schedule

```python
import matplotlib.pyplot as plt

# Collect learning rates over epochs
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

lrs = []
for epoch in range(100):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.figure(figsize=(10, 4))
plt.plot(lrs)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing Schedule')
plt.grid(True, alpha=0.3)
plt.show()
```

The curve shows characteristic cosine behavior: slow decrease at start, faster in middle, slow at end.

## Comparison with Other Schedules

| Schedule | Shape | Best For |
|----------|-------|----------|
| Step | Discontinuous drops | Traditional CNNs |
| Exponential | Smooth exponential | Long training |
| **Cosine** | Smooth S-curve | Modern networks |
| Linear | Straight line | Simple baseline |

Cosine annealing often outperforms step decay because:
- No sudden LR drops that can destabilize training
- More time at intermediate learning rates
- Natural "cooling" toward the end

## Cosine Annealing with Warm Restarts

`CosineAnnealingWarmRestarts` periodically resets the learning rate, allowing the model to escape local minima:

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Restart every T_0 epochs, with period multiplied by T_mult after each restart
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # Initial restart period
    T_mult=2,    # Multiply period by 2 after each restart
    eta_min=0.001
)

# Results in restarts at: 0, 10, 30, 70, 150, ...
```

Warm restarts can help:
- Escape sharp local minima
- Find flatter, more generalizable solutions
- Ensemble predictions from different restart points

## Combining with Warmup

Best practice for large models combines warmup with cosine decay:

```python
from torch.optim.lr_scheduler import LinearLR, SequentialLR

optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Warmup for 10 epochs
warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)

# Then cosine decay for remaining 90 epochs
cosine = CosineAnnealingLR(optimizer, T_max=90, eta_min=1e-6)

# Combine them
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[10]
)

for epoch in range(100):
    # ... training ...
    scheduler.step()
```

## Practical Guidelines

**Choosing `T_max`:**
- Set to total training epochs for single decay
- Use smaller values for warm restarts

**Choosing `eta_min`:**
- Typically 0 or 1e-6 for final fine-tuning
- Higher values (1e-4) if continuing training

**Integration with training:**
```python
# Standard training loop with cosine annealing
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # Forward, backward, step
        ...
    
    # Validation
    val_loss = validate(model, val_loader)
    
    # Update learning rate (once per epoch)
    scheduler.step()
    
    # Log
    print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, "
          f"LR = {optimizer.param_groups[0]['lr']:.6f}")
```

## When to Use Cosine Annealing

**Strongly recommended:**
- Training transformers
- Modern CNN architectures
- Any training > 50 epochs
- When smooth convergence is desired

**Consider alternatives when:**
- Training is very short (<20 epochs)
- You need predictable LR drops at specific points
- Using learning rate finding techniques

## Key Takeaways

Cosine annealing provides smooth learning rate decay following a half-cosine curve, spending more time at intermediate learning rates than linear or exponential decay. It has become the standard choice for transformer training and modern deep learning. Warm restarts can help escape local minima by periodically resetting the learning rate. For best results, combine cosine annealing with initial warmup when training large models.
