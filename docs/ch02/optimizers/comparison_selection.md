# Optimizer Comparison and Selection

## Overview

Choosing the right optimizer is crucial for successful deep learning training. This document provides comprehensive comparisons, benchmarks, and practical guidelines for selecting optimizers based on your specific use case. No single optimizer is universally best—the choice depends on your task, constraints, and priorities.

## Optimizer Family Tree

```
Gradient Descent
├── Momentum (1964)
│   └── Nesterov Accelerated Gradient (1983)
│
├── Adaptive Learning Rate
│   ├── AdaGrad (2011)
│   │   └── RMSprop (2012)
│   │       └── Adam (2014)
│   │           ├── AdamW (2017)
│   │           ├── AMSGrad (2018)
│   │           ├── RAdam (2019)
│   │           └── NAdam (2016)
│   └── Adadelta (2012)
│
└── Second-Order Methods
    ├── Newton's Method
    ├── BFGS / L-BFGS
    └── K-FAC
```

Alternative view by innovation:

```
SGD (baseline)
 ├── + Momentum → SGD with Momentum
 │    └── + Nesterov look-ahead → NAG
 │
 ├── + Adaptive LR → AdaGrad
 │    └── + Decaying average → RMSprop
 │
 └── + Momentum + Adaptive LR → Adam
      ├── + Decoupled weight decay → AdamW
      ├── + Max v tracking → AMSGrad
      └── + Variance adaptation → RAdam
```

## Quick Reference Table

| Optimizer | Momentum | Adaptive LR | Bias Correction | Key Feature | Best For |
|-----------|----------|-------------|-----------------|-------------|----------|
| SGD | No | No | N/A | Simplest baseline | Baseline |
| Momentum | Yes | No | N/A | Accelerates convergence | Deep CNNs |
| NAG | Yes (lookahead) | No | N/A | Reduced overshooting | When momentum overshoots |
| AdaGrad | No | Yes (sum) | No | Sparse data specialist | Short training, NLP |
| RMSprop | No | Yes (EMA) | No | RNN training | RNNs, non-stationary |
| Adam | Yes | Yes (EMA) | Yes | Default choice | Quick prototyping |
| AdamW | Yes | Yes (EMA) | Yes | Decoupled weight decay | Transformers |
| AMSGrad | Yes | Yes (max) | Yes | Convergence guarantee | Theoretical guarantees |
| RAdam | Yes | Yes (EMA) | Yes | Automatic warmup | No-warmup training |

## Mathematical Comparison

### Update Rule Comparison

**SGD:**
$$\theta_t = \theta_{t-1} - \eta g_t$$

**Momentum:**
$$v_t = \gamma v_{t-1} + \eta g_t; \quad \theta_t = \theta_{t-1} - v_t$$

**AdaGrad:**
$$G_t = G_{t-1} + g_t^2; \quad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t$$

**RMSprop:**
$$v_t = \rho v_{t-1} + (1-\rho)g_t^2; \quad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t$$

**Adam:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t; \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### Effective Learning Rate

| Optimizer | Effective LR | Behavior |
|-----------|--------------|----------|
| SGD | $\eta$ | Constant |
| Momentum | $\approx \frac{\eta}{1-\gamma}$ | Amplified in consistent directions |
| AdaGrad | $\frac{\eta}{\sqrt{\sum g^2}}$ | Monotonically decreasing |
| RMSprop | $\frac{\eta}{\sqrt{EMA(g^2)}}$ | Adaptive, can increase |
| Adam | $\frac{\eta}{\sqrt{\hat{v}_t}}$ | Adaptive with bias correction |

## Empirical Comparison

### Test Results Summary

**Simple Quadratic:** $f(x, y) = x^2 + y^2$, starting at $(10, 10)$

| Optimizer | Steps to f < 0.01 | Final Loss |
|-----------|-------------------|------------|
| SGD | ~100 | 0.0001 |
| Momentum | ~30 | < 0.0001 |
| Adam | ~25 | < 0.0001 |
| RMSprop | ~30 | < 0.0001 |
| AdaGrad | ~40 | < 0.0001 |

**Ill-Conditioned Problem:** $f(x, y) = 100x^2 + y^2$ (condition number = 100)

| Optimizer | Steps to f < 1.0 | Notes |
|-----------|------------------|-------|
| SGD | ~500 | Very slow along x |
| Momentum | ~200 | Oscillates across ravine |
| Adam | ~50 | Handles scale difference well |
| RMSprop | ~60 | Good adaptation |
| AdaGrad | ~80 | Learning rate eventually too small |

### PyTorch Benchmark Code

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Non-convex optimization problem
torch.manual_seed(42)
n_samples = 200
X = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
y = torch.sin(X) + 0.5 * X + torch.randn(n_samples, 1) * 0.1

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_optimizer(opt_name, opt_class, lr, epochs=200, **kwargs):
    torch.manual_seed(42)
    model = MLP()
    optimizer = opt_class(model.parameters(), lr=lr, **kwargs)
    criterion = nn.MSELoss()
    
    losses = []
    for _ in range(epochs):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses

# Compare optimizers
configs = [
    ('SGD', optim.SGD, 0.01, {}),
    ('SGD+Momentum', optim.SGD, 0.01, {'momentum': 0.9}),
    ('RMSprop', optim.RMSprop, 0.01, {}),
    ('Adam', optim.Adam, 0.001, {}),
    ('AdamW', optim.AdamW, 0.001, {'weight_decay': 0.01}),
]

print("Final losses after 200 epochs:")
print("-" * 50)
for name, opt_class, lr, kwargs in configs:
    losses = train_optimizer(name, opt_class, lr, **kwargs)
    print(f"{name:15s}: {losses[-1]:.6f} (LR={lr})")
```

### NumPy Comprehensive Comparison

```python
import numpy as np

def compare_all_optimizers():
    """
    Compare all optimizers on multiple test problems.
    """
    
    # Test Problem 1: Simple Quadratic
    print("=" * 80)
    print("TEST 1: Simple Quadratic Function")
    print("=" * 80)
    print("Minimizing f(x, y) = x² + y²")
    
    results = {}
    
    # Adam
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    adam = Adam(learning_rate=0.1)
    losses = []
    for i in range(50):
        f = params['x']**2 + params['y']**2
        losses.append(f[0])
        grads = {'x': 2*params['x'], 'y': 2*params['y']}
        params = adam.update(params, grads)
    results['Adam'] = losses
    
    # RMSprop
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    rmsprop = RMSprop(learning_rate=0.1)
    losses = []
    for i in range(50):
        f = params['x']**2 + params['y']**2
        losses.append(f[0])
        grads = {'x': 2*params['x'], 'y': 2*params['y']}
        params = rmsprop.update(params, grads)
    results['RMSprop'] = losses
    
    # AdaGrad
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    adagrad = AdaGrad(learning_rate=1.0)
    losses = []
    for i in range(50):
        f = params['x']**2 + params['y']**2
        losses.append(f[0])
        grads = {'x': 2*params['x'], 'y': 2*params['y']}
        params = adagrad.update(params, grads)
    results['AdaGrad'] = losses
    
    # Print comparison
    print(f"\n{'Iter':<8} {'Adam':<15} {'RMSprop':<15} {'AdaGrad':<15}")
    print("-" * 60)
    for i in [0, 10, 20, 30, 40, 49]:
        print(f"{i:<8} {results['Adam'][i]:<15.6f} "
              f"{results['RMSprop'][i]:<15.6f} {results['AdaGrad'][i]:<15.6f}")
    
    return results


def test_rosenbrock():
    """
    Test on challenging Rosenbrock function.
    f(x, y) = (1-x)² + 100(y-x²)²
    """
    print("\n" + "=" * 80)
    print("TEST: Rosenbrock Function")
    print("=" * 80)
    print("Minimizing f(x, y) = (1-x)² + 100(y-x²)²")
    print("Global minimum at (1, 1)")
    
    def rosenbrock_grad(x, y):
        dx = -2*(1-x) - 400*x*(y - x**2)
        dy = 200*(y - x**2)
        return dx, dy
    
    # Initialize
    params_adam = {'x': np.array([-1.0]), 'y': np.array([-1.0])}
    params_rmsprop = {'x': np.array([-1.0]), 'y': np.array([-1.0])}
    params_adagrad = {'x': np.array([-1.0]), 'y': np.array([-1.0])}
    
    adam = Adam(learning_rate=0.01)
    rmsprop = RMSprop(learning_rate=0.01)
    adagrad = AdaGrad(learning_rate=0.1)
    
    print(f"\n{'Iter':<8} {'Adam f':<15} {'RMSprop f':<15} {'AdaGrad f':<15}")
    print("-" * 60)
    
    for i in range(1000):
        # Compute gradients and update
        dx, dy = rosenbrock_grad(params_adam['x'][0], params_adam['y'][0])
        params_adam = adam.update(params_adam, {'x': np.array([dx]), 'y': np.array([dy])})
        
        dx, dy = rosenbrock_grad(params_rmsprop['x'][0], params_rmsprop['y'][0])
        params_rmsprop = rmsprop.update(params_rmsprop, {'x': np.array([dx]), 'y': np.array([dy])})
        
        dx, dy = rosenbrock_grad(params_adagrad['x'][0], params_adagrad['y'][0])
        params_adagrad = adagrad.update(params_adagrad, {'x': np.array([dx]), 'y': np.array([dy])})
        
        if i % 200 == 0:
            f_adam = (1-params_adam['x'][0])**2 + 100*(params_adam['y'][0]-params_adam['x'][0]**2)**2
            f_rms = (1-params_rmsprop['x'][0])**2 + 100*(params_rmsprop['y'][0]-params_rmsprop['x'][0]**2)**2
            f_ada = (1-params_adagrad['x'][0])**2 + 100*(params_adagrad['y'][0]-params_adagrad['x'][0]**2)**2
            print(f"{i:<8} {f_adam:<15.6f} {f_rms:<15.6f} {f_ada:<15.6f}")
    
    print("\nFinal positions:")
    print(f"  Adam:    ({params_adam['x'][0]:.4f}, {params_adam['y'][0]:.4f})")
    print(f"  RMSprop: ({params_rmsprop['x'][0]:.4f}, {params_rmsprop['y'][0]:.4f})")
    print(f"  AdaGrad: ({params_adagrad['x'][0]:.4f}, {params_adagrad['y'][0]:.4f})")
    print(f"  Target:  (1.0, 1.0)")


if __name__ == "__main__":
    compare_all_optimizers()
    test_rosenbrock()
```

## Selection Guidelines

### Decision Tree

```
START
│
├─ Is your data sparse (NLP, recommendations)?
│   └─ YES → Consider AdaGrad (short training) or Adam/AdamW
│
├─ Are you training RNNs/LSTMs?
│   └─ YES → RMSprop or Adam with gradient clipping
│
├─ Are you training Transformers (BERT, GPT, ViT)?
│   └─ YES → AdamW with warmup + cosine decay
│
├─ Do you need best generalization?
│   └─ YES → SGD with momentum + careful tuning
│           OR AdamW with proper weight decay
│
├─ Do you want good defaults with minimal tuning?
│   └─ YES → Adam or AdamW
│
├─ Are you doing online/continual learning?
│   └─ YES → RMSprop or Adam (adapt to distribution shift)
│
└─ Are you training a very large model?
    └─ YES → AdamW with gradient checkpointing
```

### By Application Domain

| Domain | Recommended | Alternative | Notes |
|--------|-------------|-------------|-------|
| **NLP / Transformers** | AdamW | Adam | With warmup and decay |
| **Image Classification** | SGD+Momentum | AdamW | SGD often generalizes better |
| **Object Detection** | SGD+Momentum | AdamW | Standard in YOLO, Faster R-CNN |
| **GANs** | Adam | RMSprop | β₁=0.5, β₂=0.999 common |
| **RNNs / LSTMs** | Adam | RMSprop | With gradient clipping |
| **Reinforcement Learning** | Adam | RMSprop | Stable under non-stationarity |
| **Fine-tuning** | AdamW | - | Lower LR, lower weight decay |
| **Quick Prototyping** | Adam | AdamW | Works out-of-box |
| **Production (vision)** | SGD+Momentum | AdamW | Tuned for best accuracy |
| **Meta-learning** | Adam | - | Handles second-order gradients |

### By Priority

**Prioritize fast convergence:**
```python
# Adam family converges quickly with minimal tuning
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Prioritize generalization:**
```python
# SGD+Momentum often generalizes best for vision
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# Pair with learning rate schedule
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)
```

**Prioritize robustness:**
```python
# AdamW with proper weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### By Training Characteristics

| Characteristic | Best Optimizer | Avoid |
|----------------|----------------|-------|
| Sparse gradients | AdaGrad, Adam | SGD (no adaptation) |
| Noisy gradients | Adam (momentum smooths) | AdaGrad |
| Many parameters | Adam, AdamW | Higher-order methods |
| Small dataset | SGD+Momentum | Over-parameterized Adam |
| Large batch | LAMB, AdamW | Vanilla SGD |
| Non-stationary | RMSprop, Adam | AdaGrad (doesn't forget) |

### By Constraints

| Constraint | Recommendation |
|------------|----------------|
| Limited tuning time | Adam (works out-of-box) |
| Memory limited | SGD (lowest memory) |
| Best final accuracy | SGD+Momentum (tuned) |
| Sparse features | Adam family |
| Very deep networks | AdamW with warmup |

## Hyperparameter Recommendations

### Default Configurations

```python
import torch.optim as optim

# SGD with Momentum
optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=1e-4)

# Nesterov SGD
optimizer = optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)

# Adam (general)
optimizer = optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)

# AdamW (Transformers)
optimizer = optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)

# AdamW (Vision Transformers)
optimizer = optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.05)

# RMSprop (RNNs)
optimizer = optim.RMSprop(params, lr=0.001, alpha=0.99)

# Adam for GANs
optimizer = optim.Adam(params, lr=0.0002, betas=(0.5, 0.999))
```

### Learning Rate Ranges

| Optimizer | Typical Range | Starting Point |
|-----------|---------------|----------------|
| SGD | 0.01 - 0.1 | 0.1 |
| SGD+Momentum | 0.01 - 0.1 | 0.1 |
| Adam | 0.0001 - 0.001 | 0.001 |
| AdamW | 0.0001 - 0.001 | 0.001 |
| RMSprop | 0.0001 - 0.01 | 0.001 |
| AdaGrad | 0.001 - 1.0 | 0.01 |

**Learning Rate Rule of Thumb:** Adam-family optimizers typically need 10-100× lower learning rates than SGD due to their adaptive scaling.

## Memory and Computation Comparison

| Optimizer | Memory per Parameter | For 100M params | Computation per Step |
|-----------|---------------------|-----------------|---------------------|
| SGD | 0 (gradients only) | ~400 MB | $O(d)$ |
| Momentum | $d$ (velocity) | ~800 MB | $O(d)$ |
| NAG | $d$ (velocity) | ~800 MB | $O(d)$ |
| AdaGrad | $d$ (cache) | ~800 MB | $O(d)$ |
| RMSprop | $d$ (cache) | ~800 MB | $O(d)$ |
| Adam/AdamW | $2d$ (m, v) | ~1.2 GB | $O(d)$ |
| AMSGrad | $3d$ (m, v, v_max) | ~1.6 GB | $O(d)$ |

Where $d$ is the number of parameters. For memory-constrained settings, SGD is most efficient.

## Common Patterns and Anti-Patterns

### Good Practices

1. **Always use a learning rate scheduler**
   ```python
   scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
   ```

2. **Use warmup for Transformers**
   ```python
   # Linear warmup for first 10% of training
   warmup_steps = int(0.1 * total_steps)
   ```

3. **Match optimizer to architecture**
   - CNNs: SGD+Momentum often best
   - Transformers: AdamW standard

4. **Use gradient clipping with RNNs**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   optimizer.step()
   ```

### Anti-Patterns to Avoid

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Using Adam with L2 regularization | Poor generalization | Use AdamW instead |
| Same LR for Adam and SGD | Adam diverges or SGD too slow | Use 0.001 for Adam, 0.01-0.1 for SGD |
| Very high LR with adaptive optimizers | Divergence | They already adapt; use lower LR |
| AdaGrad for long training | Learning rate vanishes | Use Adam or RMSprop |
| No learning rate decay | Plateau in training | Add cosine or step decay |
| Same hyperparameters for all layers | Suboptimal | Consider layer-wise LR |
| Momentum too high | Oscillating loss | Reduce to 0.9 |
| Fixed LR too long | Overfitting | Decay LR in later epochs |

## Practical Recommendations

### For New Projects

Start with this default configuration:

```python
# Safe default for most tasks
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs,
    eta_min=1e-6
)
```

### For Best Vision Results

```python
# Proven for image classification
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[60, 120, 160],
    gamma=0.1
)
```

### For Transformers

```python
# Standard for BERT, GPT, ViT
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# With warmup
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=warmup_steps
)
main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[warmup_steps]
)
```

## Summary Recommendations

### Quick Selection Guide

| If you want... | Use... |
|----------------|--------|
| Minimal tuning | Adam |
| Best generalization | SGD + Momentum + careful tuning |
| Transformers | AdamW + warmup |
| Sparse data | Adam or AdaGrad |
| Non-stationary problems | RMSprop or Adam |

### For Beginners
Start with **Adam** (lr=0.001). It works well out of the box for most problems.

### For Production
- **Computer Vision:** SGD+Momentum with careful tuning, or AdamW
- **NLP/Transformers:** AdamW with warmup + cosine decay
- **Generative Models:** Adam with task-specific betas

### For Research
- Benchmark against **SGD+Momentum** as a strong baseline
- Use **AdamW** as the adaptive baseline
- Always include learning rate schedules

## Summary Table

| Optimizer | Year | Best For | Key Strength | Key Weakness |
|-----------|------|----------|--------------|--------------|
| SGD | - | Final training, generalization | Simple, robust | Slow, sensitive to LR |
| Momentum | 1964 | General acceleration | Fast in ravines | Can overshoot |
| NAG | 1983 | When momentum overshoots | Look-ahead correction | Slight complexity |
| AdaGrad | 2011 | Sparse data, NLP | Per-param adaptation | LR vanishes |
| RMSprop | 2012 | RNNs, non-stationary | Fixes AdaGrad | No momentum |
| Adam | 2014 | General default | Robust, fast | May generalize worse |
| AdamW | 2017 | Transformers | Proper regularization | Slight complexity |
| AMSGrad | 2018 | Convergence guarantees | Theoretical fix | Often slower |
| RAdam | 2019 | No-warmup training | Automatic warmup | Slight overhead |

## References

1. Ruder, S. (2016). "An Overview of Gradient Descent Optimization Algorithms"
2. Schmidt, R., Schneider, F., & Hennig, P. (2021). "Descending through a Crowded Valley"
3. Choi, D., et al. (2019). "On Empirical Comparisons of Optimizers for Deep Learning"
4. Zhang, M., et al. (2019). "Which Algorithmic Choices Matter at Which Batch Sizes?"
