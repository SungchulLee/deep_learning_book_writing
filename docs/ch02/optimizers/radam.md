# RAdam (Rectified Adam)

## Overview

RAdam (Rectified Adam) was introduced by Liu et al. in 2019 ("On the Variance of the Adaptive Learning Rate and Beyond"). It addresses the problem of high variance in the adaptive learning rate during early training stages by dynamically adjusting the adaptivity based on the variance of the second moment estimate. This provides warmup-like behavior automatically, eliminating the need for manual warmup scheduling.

## Motivation

### The Warmup Problem

Practitioners discovered that Adam often benefits from learning rate warmup—starting with a small learning rate and gradually increasing it. This suggests something is problematic with Adam's behavior in early training.

### Root Cause: Variance in Second Moment

Adam's adaptive learning rate relies on the second moment estimate $v_t$. Early in training, this estimate has high variance due to limited samples:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

For small $t$:
- $v_t$ is based on very few gradient samples
- The estimate has high variance
- This leads to unreliable adaptive learning rates

This causes:
1. Unreliable learning rate adaptation
2. Need for manual learning rate warmup
3. Sensitivity to initial learning rate choice

### Mathematical Analysis

The bias-corrected second moment estimate:

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The variance of this estimate depends on the effective sample size, which is small initially.

Liu et al. showed that the simple maximum length of the approximated SMA (Simple Moving Average) is:

$$\rho_\infty = \frac{2}{1 - \beta_2} - 1$$

At step $t$, the SMA approximation length is:

$$\rho_t = \rho_\infty - \frac{2t\beta_2^t}{1 - \beta_2^t}$$

When $\rho_t \leq 4$ (early training), the variance is too high for reliable adaptation.

## RAdam Solution

### Variance Rectification

RAdam computes a variance rectification term $r_t$ that:
- Disables adaptivity when variance is high (early training)
- Gradually enables adaptivity as variance decreases
- Provides automatic warmup behavior

RAdam decides between:
- **Adaptive mode**: When variance is low (reliable estimate), $\rho_t > 4$
- **Unadapted mode**: When variance is high (unreliable estimate), $\rho_t \leq 4$

### Rectification Term

$$r_t = \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\rho_\infty}{(\rho_\infty - 4)(\rho_\infty - 2)\rho_t}}$$

This term:
- Approaches 1 as $t \to \infty$
- Is undefined (triggers SGD mode) when $\rho_t \leq 4$

### Update Rules

**First moment:**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**Second moment:**
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

**Bias-corrected first moment:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

**Compute SMA length:**
$$\rho_t = \rho_\infty - \frac{2t\beta_2^t}{1 - \beta_2^t}$$

**Parameter update:**
$$\theta_t = \begin{cases}
\theta_{t-1} - \eta \cdot r_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} & \text{if } \rho_t > 4 \\
\theta_{t-1} - \eta \cdot \hat{m}_t & \text{if } \rho_t \leq 4
\end{cases}$$

## Algorithm

```
Algorithm: RAdam
Input: Initial θ₀, learning rate η, β₁=0.9, β₂=0.999, ε=10⁻⁸
Compute: ρ_∞ = 2/(1-β₂) - 1
Initialize: m₀ = 0, v₀ = 0

for t = 1, 2, ... do
    g_t ← ∇_θ L(θ_{t-1})                           # Compute gradient
    m_t ← β₁ · m_{t-1} + (1 - β₁) · g_t            # Update first moment
    v_t ← β₂ · v_{t-1} + (1 - β₂) · g_t²           # Update second moment
    m̂_t ← m_t / (1 - β₁ᵗ)                          # Bias-corrected first moment
    
    ρ_t ← ρ_∞ - 2t·β₂ᵗ/(1 - β₂ᵗ)                  # SMA length
    
    if ρ_t > 4 then                                 # Variance is tractable
        v̂_t ← v_t / (1 - β₂ᵗ)                      # Bias-corrected second moment
        r_t ← √[(ρ_t-4)(ρ_t-2)ρ_∞ / ((ρ_∞-4)(ρ_∞-2)ρ_t)]  # Rectification
        θ_t ← θ_{t-1} - η · r_t · m̂_t / (√v̂_t + ε)
    else                                            # Use SGD (no adaptation)
        θ_t ← θ_{t-1} - η · m̂_t
    end if
end for
```

## Automatic Warmup Effect

RAdam provides implicit warmup without explicit scheduling:

```
Step 1-5:    ρ_t < 5  →  SGD-like updates (conservative)
Step 5+:     ρ_t > 5  →  Adaptive updates (full Adam)
```

This transition happens automatically based on variance reliability.

```python
# Comparison: Adam with warmup vs RAdam (no warmup needed)

# Adam typically needs warmup
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer_adam, start_factor=0.1, total_iters=1000
)

# RAdam handles this automatically
optimizer_radam = optim.RAdam(model.parameters(), lr=0.001)
# No warmup scheduler needed!
```

## PyTorch Implementation

### Using Built-in RAdam

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# RAdam optimizer
optimizer = optim.RAdam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0  # Can add L2 regularization
)
```

### From-Scratch PyTorch Implementation

```python
import torch
from torch.optim import Optimizer
import math

class RAdam(Optimizer):
    """
    RAdam (Rectified Adam) optimizer implementation.
    
    Automatically adjusts adaptivity based on variance of the
    second moment estimate, providing implicit warmup.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients for moment estimates (default: (0.9, 0.999))
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0)
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            # Compute rho_infinity
            rho_inf = 2.0 / (1.0 - beta2) - 1.0
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                m = state['exp_avg']
                v = state['exp_avg_sq']
                
                state['step'] += 1
                t = state['step']
                
                # Update moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction for first moment
                bias_correction1 = 1 - beta1 ** t
                m_hat = m / bias_correction1
                
                # Compute SMA length
                beta2_t = beta2 ** t
                rho_t = rho_inf - 2.0 * t * beta2_t / (1.0 - beta2_t)
                
                # Check if variance is tractable
                if rho_t > 4:
                    # Use adaptive learning rate
                    bias_correction2 = 1 - beta2_t
                    v_hat = v / bias_correction2
                    
                    # Compute rectification term
                    rect = math.sqrt(
                        (rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                        ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                    )
                    
                    # Adaptive update
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr * rect)
                else:
                    # Use unadapted update (like SGD with momentum)
                    p.add_(m_hat, alpha=-lr)
        
        return loss
```

### NumPy Implementation

```python
import numpy as np

class RAdam:
    """
    RAdam optimizer implementation (NumPy).
    
    Parameters:
    -----------
    learning_rate : float, default=0.001
        Step size for parameter updates
    beta1 : float, default=0.9
        Exponential decay rate for first moment
    beta2 : float, default=0.999
        Exponential decay rate for second moment
    epsilon : float, default=1e-8
        Small constant for numerical stability
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Maximum SMA length
        self.rho_inf = 2.0 / (1.0 - beta2) - 1.0
        
        self.m = {}
        self.v = {}
        self.t = 0
    
    def update(self, params, grads):
        """
        Update parameters using RAdam algorithm.
        """
        self.t += 1
        updated_params = {}
        
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update first moment
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update second moment
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Bias-corrected first moment
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute SMA length
            beta2_t = self.beta2 ** self.t
            rho_t = self.rho_inf - 2.0 * self.t * beta2_t / (1.0 - beta2_t)
            
            if rho_t > 4:
                # Variance is tractable - use adaptive learning rate
                v_hat = self.v[key] / (1 - beta2_t)
                
                # Rectification term
                rect = np.sqrt(
                    (rho_t - 4.0) * (rho_t - 2.0) * self.rho_inf /
                    ((self.rho_inf - 4.0) * (self.rho_inf - 2.0) * rho_t)
                )
                
                updated_params[key] = (
                    params[key] - 
                    self.lr * rect * m_hat / (np.sqrt(v_hat) + self.epsilon)
                )
            else:
                # Variance too high - use SGD with momentum
                updated_params[key] = params[key] - self.lr * m_hat
        
        return updated_params
    
    def get_rho_t(self):
        """Return current SMA length for debugging."""
        beta2_t = self.beta2 ** self.t
        return self.rho_inf - 2.0 * self.t * beta2_t / (1.0 - beta2_t)
```

## Demonstrations

### Demo 1: Automatic Warmup Behavior

```python
import numpy as np

def demo_radam_warmup():
    """
    Demonstrate RAdam's automatic warmup behavior.
    """
    print("=" * 70)
    print("RAdam Automatic Warmup Demonstration")
    print("=" * 70)
    print()
    
    beta2 = 0.999
    rho_inf = 2.0 / (1.0 - beta2) - 1.0
    
    print(f"ρ_∞ = {rho_inf:.2f}")
    print(f"Threshold for adaptive mode: ρ_t > 4")
    print()
    
    print(f"{'Step t':<10} {'ρ_t':<15} {'Mode':<20} {'Rect. Term':<15}")
    print("-" * 60)
    
    for t in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 500, 1000]:
        beta2_t = beta2 ** t
        rho_t = rho_inf - 2.0 * t * beta2_t / (1.0 - beta2_t)
        
        if rho_t > 4:
            mode = "Adaptive (Adam)"
            rect = np.sqrt(
                (rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
            )
            rect_str = f"{rect:.6f}"
        else:
            mode = "Non-adaptive (SGD)"
            rect_str = "N/A"
        
        print(f"{t:<10} {rho_t:<15.4f} {mode:<20} {rect_str:<15}")
    
    print()
    print("Observation: RAdam automatically transitions from SGD to Adam!")
    print("This provides implicit warmup without manual scheduling.")
```

### Demo 2: RAdam vs Adam Early Training

```python
def demo_radam_vs_adam():
    """
    Compare RAdam vs Adam behavior in early training.
    """
    print("=" * 70)
    print("RAdam vs Adam: Early Training Comparison")
    print("=" * 70)
    print()
    
    # Initialize
    params_adam = {'x': np.array([10.0])}
    params_radam = {'x': np.array([10.0])}
    
    adam = Adam(learning_rate=0.1)
    radam = RAdam(learning_rate=0.1)
    
    # Track effective learning rates
    print(f"{'Step':<8} {'Adam x':<12} {'RAdam x':<12} {'RAdam mode':<15} {'RAdam rect':<12}")
    print("-" * 60)
    
    for t in range(1, 21):
        # Gradient for f(x) = x²
        grad_adam = {'x': 2 * params_adam['x']}
        grad_radam = {'x': 2 * params_radam['x']}
        
        params_adam = adam.update(params_adam, grad_adam)
        params_radam = radam.update(params_radam, grad_radam)
        
        rho_t = radam.get_rho_t()
        if rho_t > 4:
            mode = "Adaptive"
            rect = np.sqrt(
                (rho_t - 4.0) * (rho_t - 2.0) * radam.rho_inf /
                ((radam.rho_inf - 4.0) * (radam.rho_inf - 2.0) * rho_t)
            )
            rect_str = f"{rect:.4f}"
        else:
            mode = "SGD"
            rect_str = "N/A"
        
        if t <= 10 or t % 5 == 0:
            print(f"{t:<8} {params_adam['x'][0]:<12.6f} {params_radam['x'][0]:<12.6f} "
                  f"{mode:<15} {rect_str:<12}")
```

### Demo 3: PyTorch Comparison

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(500, 10)
y = X @ torch.randn(10, 1) + torch.randn(500, 1) * 0.1

def train_optimizer(opt_class, name, epochs=200, **kwargs):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    optimizer = opt_class(model.parameters(), lr=0.001, **kwargs)
    criterion = nn.MSELoss()
    
    losses = []
    for epoch in range(epochs):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    print(f"{name}: Final loss = {losses[-1]:.6f}")
    return losses

# Compare
adam_losses = train_optimizer(optim.Adam, "Adam")
radam_losses = train_optimizer(optim.RAdam, "RAdam")
```

### Demo 4: Variance Comparison

```python
def demo_variance_comparison():
    """
    Show the variance of second moment estimates over time.
    """
    print("=" * 70)
    print("Second Moment Variance Analysis")
    print("=" * 70)
    print()
    
    np.random.seed(42)
    beta2 = 0.999
    
    # Simulate many gradient sequences
    n_simulations = 1000
    n_steps = 100
    true_variance = 1.0  # Assume gradients have variance 1
    
    v_estimates = np.zeros((n_simulations, n_steps))
    
    for sim in range(n_simulations):
        v = 0.0
        for t in range(n_steps):
            g = np.random.randn()  # Random gradient
            v = beta2 * v + (1 - beta2) * g**2
            v_hat = v / (1 - beta2 ** (t + 1))
            v_estimates[sim, t] = v_hat
    
    # Compute variance of estimates at each step
    estimate_variance = np.var(v_estimates, axis=0)
    
    print(f"{'Step':<10} {'Mean v̂':<15} {'Var(v̂)':<15} {'CV':<15}")
    print("-" * 55)
    
    for t in [0, 1, 2, 4, 9, 19, 49, 99]:
        mean_v = np.mean(v_estimates[:, t])
        var_v = estimate_variance[t]
        cv = np.sqrt(var_v) / mean_v if mean_v > 0 else 0
        print(f"{t+1:<10} {mean_v:<15.4f} {var_v:<15.4f} {cv:<15.4f}")
    
    print()
    print("Observation: Variance is very high initially, decreasing over time.")
    print("RAdam avoids using v̂ when its variance (CV) is high.")
```

## Theoretical Properties

### SMA Length Approximation

The effective sample size for the second moment estimate can be approximated by the SMA length:

$$\rho_t \approx \frac{1 - \beta_2^t}{1 - \beta_2}$$

This grows from ~1 to $\rho_\infty$ as training progresses.

### Variance Rectification Derivation

The rectification term $r_t$ is derived to approximate the ratio of:
- The true second moment
- The estimated second moment with its variance

This ensures the update magnitude is appropriately scaled regardless of estimation uncertainty.

### Convergence Properties

RAdam inherits Adam's convergence properties while:
- Avoiding unstable updates in early training
- Providing smoother optimization trajectory
- Eliminating the need for manual warmup

## Comparison Tables

### Adam vs RAdam vs Adam+Warmup

| Aspect | Adam | Adam+Warmup | RAdam |
|--------|------|-------------|-------|
| Early training | Potentially unstable | Stable (manual) | Stable (automatic) |
| Hyperparameters | Standard | +warmup steps | Standard |
| Implementation | Simple | Extra scheduling | Moderate |
| Performance | Good after warmup | Good | Good |

### RAdam vs AdamW

| Aspect | RAdam | AdamW |
|--------|-------|-------|
| Warmup needed | No (automatic) | Yes (typically) |
| Weight decay | L2 style | Decoupled (better) |
| Transformer training | Good | Best with warmup |
| Simplicity | Simpler setup | Needs scheduler |

For transformers, **AdamW + warmup** remains the gold standard, but **RAdam** is a solid choice when simplicity is prioritized.

## Hyperparameter Guidelines

### Default Values

RAdam uses the same defaults as Adam:

| Parameter | Default | Notes |
|-----------|---------|-------|
| lr | 0.001 | Can often use higher than Adam |
| betas | (0.9, 0.999) | Same as Adam |
| eps | 1e-8 | Same as Adam |
| weight_decay | 0 | L2 regularization |

### When to Adjust

- **Higher $\beta_2$ (0.9999)**: Extends warmup period
- **Lower $\beta_2$ (0.99)**: Shortens warmup period
- Generally, defaults work well without tuning

## When to Use RAdam

### Recommended Scenarios

1. **When warmup is currently used**: RAdam can replace manual warmup
2. **Training from scratch**: Benefits most from automatic warmup
3. **Large learning rates**: More robust to aggressive LR choices
4. **Quick experiments**: Skip warmup tuning
5. **Transformers and large models**: Where warmup is typically critical
6. **Variable learning rate schedules**: Works well without careful warmup tuning
7. **Large batch training**: More stable than Adam

### When Adam May Be Fine

1. **Transfer learning/fine-tuning**: Warmup less critical
2. **When warmup is already optimized**: Similar performance
3. **Very long training**: Warmup period negligible
4. **Using AdamW with established hyperparameters**

## Combining with Weight Decay

For proper weight decay with RAdam, you may want to implement decoupled weight decay manually:

```python
# Standard RAdam with L2 regularization
optimizer = optim.RAdam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 style, not decoupled
)

# For decoupled weight decay, apply manually
optimizer = optim.RAdam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # ... forward, backward ...
    optimizer.step()
    
    # Manual weight decay
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(1 - lr * weight_decay)
```

## Variants

### Ranger

Combines RAdam with Lookahead:

```python
# Ranger = RAdam + Lookahead
base_optimizer = RAdam(params, lr=0.001)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

### AdamP

Combines Adam with projection to prevent excessive weight changes.

### LAMB

Layer-wise Adaptive Moments for Batch training (related ideas for large batch training).

## Summary

| Aspect | Detail |
|--------|--------|
| Key Innovation | Automatic variance-based warmup through rectification |
| Problem Solved | High variance in early training adaptive rates |
| Update Rule | Uses rectification term when $\rho_t > 4$, else SGD |
| Benefits | No manual warmup needed, more stable early training |
| Drawbacks | Slightly more computation |
| When to Use | Training from scratch, when warmup would help |
| Hyperparameters | Same as Adam |
| PyTorch | `torch.optim.RAdam` |

## References

1. Liu, L., et al. (2019). "On the Variance of the Adaptive Learning Rate and Beyond"
2. Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization"
3. Zhang, M., et al. (2019). "Lookahead Optimizer: k steps forward, 1 step back"
