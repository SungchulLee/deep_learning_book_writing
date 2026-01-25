# OneCycleLR: Super-Convergence

## Overview

OneCycleLR implements the 1cycle learning rate policy developed by Leslie Smith, which enables **super-convergence**—training neural networks significantly faster while achieving equal or better accuracy. The policy consists of a single cycle where the learning rate first increases from a low value to a maximum, then decreases to a very small value.

## The Super-Convergence Phenomenon

Super-convergence refers to the observation that with appropriate hyperparameters, neural networks can be trained in dramatically fewer iterations while achieving comparable or superior performance. The key insight is that using **larger learning rates** for a significant portion of training acts as a regularizer and accelerates convergence.

**Key findings from Smith's research:**

1. Large learning rates regularize training (similar to dropout)
2. The optimal maximum LR is much higher than typically used
3. Shorter training with higher LR often outperforms longer training with lower LR
4. The 1cycle policy provides a principled approach to achieving this

## Mathematical Formulation

### Two-Phase Schedule

The 1cycle policy divides training into two main phases:

**Phase 1: Increasing (Warmup)**
- Duration: `pct_start` fraction of total steps
- Learning rate increases from `initial_lr` to `max_lr`

**Phase 2: Decreasing (Annealing)**
- Duration: `1 - pct_start` fraction of total steps
- Learning rate decreases from `max_lr` to `final_lr`

### Learning Rate Formulas

**Phase 1 (Linear Increase):**

$$\eta_t = \eta_{initial} + \frac{t}{T_1}(\eta_{max} - \eta_{initial})$$

where $T_1 = \text{pct\_start} \times T_{total}$

**Phase 2 (Cosine Annealing):**

$$\eta_t = \eta_{final} + \frac{1}{2}(\eta_{max} - \eta_{final})\left(1 + \cos\left(\frac{t - T_1}{T_2} \pi\right)\right)$$

where $T_2 = T_{total} - T_1$

### Default Parameters

$$\eta_{initial} = \frac{\eta_{max}}{\text{div\_factor}}$$

$$\eta_{final} = \frac{\eta_{max}}{\text{final\_div\_factor}}$$

Default values:
- `div_factor = 25` → initial_lr = max_lr / 25
- `final_div_factor = 1e4` → final_lr = max_lr / 10000
- `pct_start = 0.3` → 30% warmup, 70% annealing

## Learning Rate Curve

```
Learning Rate
    │
    │        ╱╲
max │       ╱  ╲
    │      ╱    ╲___
    │     ╱         ╲___
init├────╱              ╲___
    │                       ╲___
final├───────────────────────────╲
    │
    └─────────────────────────────→ Step
          ↑              ↑
      pct_start      1.0
     (warmup)    (annealing)
```

## PyTorch Implementation

### Using Built-in OneCycleLR

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Calculate total steps
epochs = 10
steps_per_epoch = len(train_loader)
total_steps = epochs * steps_per_epoch

# Create OneCycleLR scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,               # Peak learning rate
    total_steps=total_steps,  # Total training steps
    pct_start=0.3,            # Fraction for warmup
    anneal_strategy='cos',    # 'cos' or 'linear'
    div_factor=25.0,          # initial_lr = max_lr / div_factor
    final_div_factor=1e4      # final_lr = max_lr / final_div_factor
)

# Training loop - step after each batch
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        
        # Step scheduler after each batch (not epoch!)
        scheduler.step()
```

### Custom Implementation

```python
import math
from typing import Literal, Optional

class OneCycleLR:
    """
    1cycle learning rate policy implementation.
    
    Reference: "Super-Convergence: Very Fast Training of Neural Networks"
    https://arxiv.org/abs/1708.07120
    """
    
    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        anneal_strategy: Literal['cos', 'linear'] = 'cos'
    ):
        """
        Args:
            max_lr: Peak learning rate
            total_steps: Total number of training steps
            pct_start: Fraction of training for warmup phase
            div_factor: Determines initial LR = max_lr / div_factor
            final_div_factor: Determines final LR = max_lr / final_div_factor
            anneal_strategy: Annealing method ('cos' or 'linear')
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        
        # Computed learning rates
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        # Phase boundaries
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up
        
        self.current_step = 0
    
    def get_lr(self, step: Optional[int] = None) -> float:
        """Calculate learning rate for given step."""
        if step is None:
            step = self.current_step
        
        if step < self.step_up:
            # Phase 1: Increasing (linear warmup)
            progress = step / self.step_up
            return self.initial_lr + progress * (self.max_lr - self.initial_lr)
        else:
            # Phase 2: Decreasing (annealing)
            progress = (step - self.step_up) / self.step_down
            progress = min(progress, 1.0)
            
            if self.anneal_strategy == 'cos':
                cos_value = math.cos(math.pi * progress)
                return self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (1 + cos_value)
            else:
                return self.max_lr - progress * (self.max_lr - self.final_lr)
    
    def step(self):
        """Advance one training step."""
        self.current_step += 1
        return self.get_lr()
```

## Finding the Optimal max_lr

### LR Range Test

```python
def lr_range_test(
    model,
    train_loader,
    start_lr: float = 1e-7,
    end_lr: float = 10,
    num_steps: int = 100
):
    """
    Learning rate range test to find optimal max_lr.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / num_steps)
    
    results = {'lr': [], 'loss': []}
    criterion = torch.nn.CrossEntropyLoss()
    batch_iter = iter(train_loader)
    
    for step in range(num_steps):
        try:
            inputs, targets = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            inputs, targets = next(batch_iter)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        results['lr'].append(optimizer.param_groups[0]['lr'])
        results['loss'].append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
    
    return results
```

**Interpretation:**
- Find where loss decreases steepest → Good LR range
- Suggested max_lr = LR at minimum loss / 10

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

def train_with_onecycle(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 10,
    max_lr: float = 0.1,
    pct_start: float = 0.3,
    device: str = 'cuda'
):
    """Train model with OneCycleLR for super-convergence."""
    model = model.to(device)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=max_lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    total_steps = epochs * len(train_loader)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos'
    )
    
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_acc': [], 'lr': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Step after each batch!
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_acc'].append(100 * correct / total)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Val Acc: {history['val_acc'][-1]:.2f}%")
    
    return history
```

## Advantages and Disadvantages

### Advantages

✅ **Dramatically faster training** - Often 10× fewer epochs needed

✅ **Better or equal accuracy** - High LR acts as regularizer

✅ **Less hyperparameter tuning** - Just need to find max_lr

✅ **Built-in warmup** - No separate warmup scheduler needed

✅ **Works out-of-the-box** - Default parameters are often good

### Disadvantages

❌ **Requires knowing total steps** - Can't use for indefinite training

❌ **Fixed schedule** - Not adaptive to training dynamics

❌ **max_lr is critical** - Must be tuned correctly

❌ **Not for transformers** - They need different schedules

## Practical Guidelines

### Choosing max_lr

| Method | Approach |
|--------|----------|
| LR range test | Run test, use 10× below minimum |
| Heuristic | Start with 0.1, adjust based on results |
| Literature | Check papers for similar architectures |

### Choosing pct_start

| Training Length | Recommended pct_start |
|-----------------|----------------------|
| Very short (5-10 epochs) | 0.2-0.3 |
| Short (10-30 epochs) | 0.3 |
| Medium (30-100 epochs) | 0.3-0.4 |
| Long (100+ epochs) | 0.3-0.5 |

## When to Use OneCycleLR

**Ideal use cases:**

- Time-constrained training (competitions, quick experiments)
- Image classification (ResNet, EfficientNet, etc.)
- When you know total training steps in advance
- SGD with momentum optimization

**Avoid when:**

- Training transformers/LLMs
- Indefinite training duration
- Very small datasets
- Transfer learning with frozen layers

## Summary

OneCycleLR enables super-convergence through a principled warmup-annealing cycle. It achieves fast training without sacrificing accuracy.

**Key takeaways:**

1. Use LR range test to find optimal max_lr
2. Default parameters often work well
3. Step scheduler after each batch, not epoch
4. Expect 3-10× faster training
5. Best with SGD + momentum
