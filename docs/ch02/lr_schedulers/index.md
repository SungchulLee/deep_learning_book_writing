# Learning Rate Schedulers

## Overview

Learning rate scheduling is one of the most impactful techniques for training deep neural networks effectively. The learning rate $\eta$ controls the magnitude of parameter updates during gradient descent, directly influencing convergence speed, training stability, and final model performance.

## The Learning Rate Dilemma

Training neural networks with a fixed learning rate presents a fundamental optimization challenge:

**High Learning Rate Problems:**

- Loss oscillations or divergence
- Overshooting optimal parameter values
- Training instability, especially early in training

**Low Learning Rate Problems:**

- Extremely slow convergence
- Risk of getting trapped in suboptimal local minima
- Insufficient exploration of the loss landscape

Learning rate schedulers resolve this dilemma by **dynamically adjusting the learning rate during training**, enabling both rapid initial progress and precise final convergence.

## Mathematical Foundation

The standard gradient descent update rule is:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

With a learning rate scheduler, this becomes:

$$\theta_{t+1} = \theta_t - \eta(t) \nabla_\theta \mathcal{L}(\theta_t)$$

where $\eta(t)$ is a function of the training step or epoch that defines how the learning rate evolves over time.

## Categories of Schedulers

Learning rate schedulers can be organized into several categories based on their behavior:

### Monotonic Decay Schedulers

These schedulers decrease the learning rate monotonically throughout training:

| Scheduler | Behavior | Update Frequency |
|-----------|----------|------------------|
| **StepLR** | Discrete drops at fixed intervals | Per epoch |
| **MultiStepLR** | Discrete drops at specific milestones | Per epoch |
| **ExponentialLR** | Continuous exponential decay | Per epoch |
| **CosineAnnealingLR** | Smooth cosine curve decay | Per epoch |

### Cyclic Schedulers

These schedulers oscillate the learning rate, potentially helping escape local minima:

| Scheduler | Behavior | Update Frequency |
|-----------|----------|------------------|
| **CyclicLR** | Oscillates between bounds | Per batch |
| **OneCycleLR** | Single warmup-decay cycle | Per batch |
| **SGDR** | Cosine with warm restarts | Per batch |

### Adaptive Schedulers

These schedulers respond to training dynamics:

| Scheduler | Behavior | Update Frequency |
|-----------|----------|------------------|
| **ReduceLROnPlateau** | Reduces when metric stagnates | After validation |

### Warmup Strategies

Warmup gradually increases the learning rate before the main schedule begins:

| Strategy | Behavior |
|----------|----------|
| **Linear Warmup** | Linear ramp from 0 to target |
| **Exponential Warmup** | Exponential ramp |
| **Cosine Warmup** | Smooth cosine ramp |

## The Training Phases Perspective

A well-designed learning rate schedule typically supports three training phases:

```
Phase 1: Exploration (High LR)
├── Rapid initial progress
├── Exploration of parameter space
└── Escape poor initialization

Phase 2: Transition (Decreasing LR)
├── Focus on promising regions
├── Refinement of parameters
└── Gradual convergence

Phase 3: Fine-tuning (Low LR)
├── Precise parameter adjustment
├── Settling into good minima
└── Final performance optimization
```

## PyTorch Scheduler Interface

All PyTorch schedulers follow a consistent interface:

```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Create scheduler
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Training loop
for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
    
    # Update scheduler (typically after each epoch)
    scheduler.step()
    
    # Access current learning rate
    current_lr = scheduler.get_last_lr()[0]
```

## Key Concepts

### Step vs Epoch Updates

Schedulers differ in their update frequency:

- **Epoch-level schedulers** (StepLR, CosineAnnealing): Call `scheduler.step()` after each epoch
- **Batch-level schedulers** (OneCycleLR, CyclicLR): Call `scheduler.step()` after each batch
- **Metric-based schedulers** (ReduceLROnPlateau): Call `scheduler.step(metric)` after validation

### Learning Rate vs Iteration Relationship

For batch-level schedulers, the total number of steps is:

$$\text{total\_steps} = \text{epochs} \times \left\lfloor \frac{\text{dataset\_size}}{\text{batch\_size}} \right\rfloor$$

### Combining Schedulers

Multiple strategies can be combined:

```python
# Example: Warmup followed by cosine decay
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)  # Linear warmup
else:
    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
```

## Chapter Contents

This chapter covers the following topics:

1. **Step Decay** - Discrete learning rate drops at regular intervals
2. **Exponential Decay** - Continuous exponential reduction
3. **Cosine Annealing** - Smooth cosine curve scheduling
4. **OneCycleLR** - Super-convergence through single cycle policy
5. **Warmup Strategies** - Gradual learning rate ramp-up techniques
6. **ReduceLROnPlateau** - Adaptive scheduling based on metrics
7. **Cyclic Learning Rates** - Oscillating schedules for exploration
8. **SGDR and Warm Restarts** - Periodic schedule restarts
9. **Scheduler Comparison** - Selection guidelines and benchmarks
10. **Custom Schedulers** - Implementing custom scheduling strategies
11. **PyTorch Implementation** - Complete implementation guide

## Quick Reference: Scheduler Selection

```
Need to choose a scheduler?
│
├─ Unsure what will work?
│  └─ ReduceLROnPlateau (safe default)
│
├─ Want fastest training (5-20 epochs)?
│  └─ OneCycleLR (super-convergence)
│
├─ Training modern architectures?
│  └─ CosineAnnealingLR or Warmup + Cosine
│
├─ Training transformers/LLMs?
│  └─ Warmup + Cosine Decay
│
├─ Know optimal decay epochs?
│  ├─ Single decay point → StepLR
│  └─ Multiple decay points → MultiStepLR
│
├─ Want to explore LR range?
│  └─ CyclicLR
│
└─ Default recommendation
   └─ OneCycleLR or CosineAnnealingLR
```

## References

1. Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks." arXiv:1506.01186
2. Smith, L. N. (2018). "A disciplined approach to neural network hyper-parameters." arXiv:1803.09820
3. Loshchilov, I., & Hutter, F. (2016). "SGDR: Stochastic Gradient Descent with Warm Restarts." arXiv:1608.03983
4. Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv:1706.02677
5. Vaswani, A., et al. (2017). "Attention Is All You Need." arXiv:1706.03762
