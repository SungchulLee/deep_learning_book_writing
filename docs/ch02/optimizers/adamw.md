# AdamW (Decoupled Weight Decay)

## Overview

AdamW was introduced by Loshchilov and Hutter in 2017 ("Fixing Weight Decay Regularization in Adam"). It addresses a subtle but important issue with how weight decay (L2 regularization) interacts with adaptive learning rate methods like Adam. AdamW decouples weight decay from the gradient-based update, leading to better generalization. It has become the de facto standard optimizer for transformer models and modern deep learning architectures.

## The Problem with L2 Regularization in Adam

### Standard L2 Regularization

The regularized loss function:

$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|^2$$

The gradient becomes:

$$\nabla \mathcal{L}_{\text{reg}}(\theta) = \nabla \mathcal{L}(\theta) + \lambda \theta$$

In standard SGD, L2 regularization and weight decay are equivalent:

$$\theta_{t+1} = \theta_t - \eta (\nabla_\theta \mathcal{L} + \lambda \theta_t) = (1 - \eta\lambda)\theta_t - \eta \nabla_\theta \mathcal{L}$$

### L2 in Adam (Problematic)

When Adam uses this regularized gradient:

$$g_t = \nabla \mathcal{L}(\theta_{t-1}) + \lambda \theta_{t-1}$$

Both $g_t$ and $g_t^2$ include the weight decay term. This means:

1. The weight decay contribution is scaled by the adaptive learning rate
2. Parameters with large gradients get less regularization
3. Parameters with small gradients get more regularization

**Adam with L2 (incorrect):**
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}(\hat{m}_t + \lambda\theta_t)$$

The weight decay term is scaled by $1/\sqrt{\hat{v}_t}$, reducing its effect for parameters with large gradient variance.

### Mathematical Illustration

Consider parameter $\theta_i$ with different gradient magnitudes:

| Scenario | Gradient $g_i$ | Adam's $v_i$ | Effective Weight Decay |
|----------|----------------|--------------|------------------------|
| Large gradient | 10 | Large | $\frac{\lambda \theta_i}{\sqrt{v_i}}$ (small) |
| Small gradient | 0.01 | Small | $\frac{\lambda \theta_i}{\sqrt{v_i}}$ (large) |

This inconsistency defeats the purpose of uniform regularization.

## AdamW Solution: Decoupled Weight Decay

AdamW separates weight decay from the gradient update:

**AdamW (correct decoupling):**
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t - \eta\lambda\theta_t$$

The weight decay is applied directly to parameters, independent of the adaptive learning rate.

### Update Rules

**First moment (unchanged):**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**Second moment (unchanged):**
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

**Bias correction (unchanged):**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Parameter update (modified):**
$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

The key difference: weight decay ($\lambda \theta_{t-1}$) is added **after** the adaptive scaling, not included in the gradient.

### Equivalent Formulation

The update can be written as two separate steps:

1. **Adam update:**
   $$\theta_t' = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

2. **Weight decay:**
   $$\theta_t = \theta_t' - \eta \lambda \theta_{t-1} = (1 - \eta\lambda)\theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

## Why Decoupling Matters

The difference is significant:

| Aspect | Adam + L2 | AdamW |
|--------|-----------|-------|
| Weight decay effect | Varies per parameter | Uniform across parameters |
| Regularization strength | Depends on gradient history | Consistent |
| Generalization | Often worse | Often better |
| Recommended for transformers | No | Yes |

## Algorithm

```
Algorithm: AdamW
Input: Initial θ₀, learning rate η, β₁=0.9, β₂=0.999, ε=10⁻⁸, weight decay λ
Initialize: m₀ = 0, v₀ = 0, t = 0

for t = 1, 2, ... do
    t ← t + 1
    g_t ← ∇_θ L(θ_{t-1})                           # Gradient (NO weight decay)
    m_t ← β₁ · m_{t-1} + (1 - β₁) · g_t            # Update first moment
    v_t ← β₂ · v_{t-1} + (1 - β₂) · g_t²           # Update second moment
    m̂_t ← m_t / (1 - β₁ᵗ)                          # Bias correction
    v̂_t ← v_t / (1 - β₂ᵗ)                          # Bias correction
    θ_t ← θ_{t-1} - η · (m̂_t / (√v̂_t + ε) + λ · θ_{t-1})  # Decoupled update
end for
```

## PyTorch Implementation

### Using Built-in AdamW

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

# AdamW with weight decay
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,           # Learning rate
    betas=(0.9, 0.999), # β₁ and β₂
    eps=1e-8,           # Epsilon for stability
    weight_decay=0.01   # Decoupled weight decay λ
)

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)  # No L2 in loss
        loss.backward()
        optimizer.step()
```

### From-Scratch PyTorch Implementation

```python
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.
    
    Unlike Adam with L2 regularization, AdamW applies weight decay
    directly to parameters, independent of the adaptive learning rate.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients for moment estimates (default: (0.9, 0.999))
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: Decoupled weight decay coefficient (default: 0.01)
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
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
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                m, v = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                
                # Update moment estimates (NO weight decay in gradient)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                
                # Adam update
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
                
                # Decoupled weight decay (applied separately)
                if weight_decay > 0:
                    p.add_(p, alpha=-lr * weight_decay)
        
        return loss
```

### NumPy Implementation

```python
import numpy as np

class AdamW:
    """
    AdamW optimizer with decoupled weight decay (NumPy).
    
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
    weight_decay : float, default=0.01
        Decoupled weight decay coefficient
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, weight_decay=0.01):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m = {}
        self.v = {}
        self.t = 0
    
    def update(self, params, grads):
        """
        Update parameters using AdamW algorithm.
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameters to update
        grads : dict
            Dictionary of gradients (should NOT include weight decay)
        
        Returns:
        --------
        dict : Updated parameters
        """
        self.t += 1
        updated_params = {}
        
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update moments (gradient only, no weight decay)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Adam update + decoupled weight decay
            updated_params[key] = (
                params[key] - 
                self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + 
                          self.weight_decay * params[key])
            )
        
        return updated_params
```

## Comparing Adam vs AdamW

### PyTorch Comparison

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# Data with some noise
X = torch.randn(200, 10)
y = X[:, 0:1] + 0.5 * X[:, 1:2] + torch.randn(200, 1) * 0.5

def train_and_evaluate(optimizer_class, weight_decay, name):
    torch.manual_seed(42)
    
    model = nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    optimizer = optimizer_class(
        model.parameters(),
        lr=0.001,
        weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    
    # Train
    for epoch in range(500):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Check weight magnitudes
    total_weight_norm = sum(p.norm().item() for p in model.parameters())
    
    print(f"{name}: Final loss = {loss.item():.4f}, "
          f"Weight norm = {total_weight_norm:.4f}")

# Compare
train_and_evaluate(optim.Adam, 0.01, "Adam (L2)")
train_and_evaluate(optim.AdamW, 0.01, "AdamW")
train_and_evaluate(optim.Adam, 0.0, "Adam (no reg)")
```

With the same weight decay coefficient, AdamW typically produces smaller weight norms, indicating more effective regularization.

### Detailed NumPy Demonstration

```python
import numpy as np

def compare_adam_adamw():
    """
    Compare Adam (with L2) vs AdamW on a regularized problem.
    """
    print("=" * 70)
    print("Adam (L2 reg) vs AdamW (decoupled weight decay)")
    print("=" * 70)
    print()
    
    # Simple quadratic with regularization target
    # Minimize f(x) = x² with regularization pushing toward 0
    
    # Adam with L2 regularization
    x_adam = np.array([10.0])
    m_adam = np.array([0.0])
    v_adam = np.array([0.0])
    
    # AdamW with decoupled weight decay
    x_adamw = np.array([10.0])
    m_adamw = np.array([0.0])
    v_adamw = np.array([0.0])
    
    lr = 0.1
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 0.1
    
    print(f"{'Step':<8} {'Adam x':<15} {'AdamW x':<15} {'Adam eff_wd':<15} {'AdamW eff_wd':<15}")
    print("-" * 70)
    
    for t in range(1, 51):
        # Gradient of f(x) = x²
        grad_base = 2 * x_adam
        
        # Adam: L2 regularization in gradient
        grad_adam = grad_base + weight_decay * x_adam
        m_adam = beta1 * m_adam + (1 - beta1) * grad_adam
        v_adam = beta2 * v_adam + (1 - beta2) * (grad_adam ** 2)
        m_hat = m_adam / (1 - beta1 ** t)
        v_hat = v_adam / (1 - beta2 ** t)
        adam_update = lr * m_hat / (np.sqrt(v_hat) + eps)
        x_adam = x_adam - adam_update
        
        # Effective weight decay in Adam (scaled by adaptive rate)
        eff_wd_adam = lr * weight_decay * x_adam / (np.sqrt(v_hat) + eps)
        
        # AdamW: Decoupled weight decay
        grad_adamw = 2 * x_adamw  # No weight decay in gradient
        m_adamw = beta1 * m_adamw + (1 - beta1) * grad_adamw
        v_adamw = beta2 * v_adamw + (1 - beta2) * (grad_adamw ** 2)
        m_hat_w = m_adamw / (1 - beta1 ** t)
        v_hat_w = v_adamw / (1 - beta2 ** t)
        adamw_update = lr * (m_hat_w / (np.sqrt(v_hat_w) + eps) + weight_decay * x_adamw)
        x_adamw = x_adamw - adamw_update
        
        # Effective weight decay in AdamW (constant rate)
        eff_wd_adamw = lr * weight_decay * x_adamw
        
        if t % 10 == 0 or t == 1:
            print(f"{t:<8} {x_adam[0]:<15.6f} {x_adamw[0]:<15.6f} "
                  f"{eff_wd_adam[0]:<15.8f} {eff_wd_adamw[0]:<15.8f}")
    
    print()
    print("Observation: AdamW applies consistent weight decay regardless of gradient magnitude.")


if __name__ == "__main__":
    compare_adam_adamw()
```

## Hyperparameter Guidelines

### Recommended Values

| Parameter | AdamW Default | Notes |
|-----------|---------------|-------|
| Learning rate | 0.001 | Same as Adam |
| $\beta_1$ | 0.9 | Same as Adam |
| $\beta_2$ | 0.999 | Same as Adam |
| $\epsilon$ | $10^{-8}$ | Same as Adam |
| Weight decay | 0.01 - 0.1 | Higher than Adam's typical L2 |

### Task-Specific Recommendations

**For transformer models:**

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Learning rate | 1e-4 to 3e-4 | Often with warmup |
| Weight decay | 0.01 to 0.1 | Higher for larger models |
| $\beta_1$ | 0.9 | First moment decay |
| $\beta_2$ | 0.999 | Second moment decay |
| $\epsilon$ | 1e-8 | Numerical stability |

**For CNNs and general use:**

| Parameter | Typical Value |
|-----------|---------------|
| Learning rate | 1e-3 to 1e-4 |
| Weight decay | 0.01 |

### Weight Decay Selection by Task

| Task | Typical Weight Decay |
|------|---------------------|
| Image Classification | 0.01 - 0.05 |
| Language Models | 0.01 - 0.1 |
| Fine-tuning | 0.0001 - 0.01 |
| Small datasets | 0.1 - 0.3 |

### Learning Rate and Weight Decay Coupling

With AdamW, the effective regularization is $\eta \cdot \lambda$. When changing learning rate:

```python
# If increasing learning rate, consider decreasing weight decay proportionally
# Option 1: Fixed effective regularization
lr = 0.01
wd = 0.001  # effective: 0.01 * 0.001 = 0.00001

# Option 2: Let them scale together (often works fine)
lr = 0.001
wd = 0.01
```

When using learning rate schedules, weight decay's effective strength changes:
- Higher LR → stronger effective regularization
- Lower LR → weaker effective regularization

Some implementations scale weight decay inversely with learning rate to maintain consistent regularization strength.

## Weight Decay Best Practices

### What to Apply Weight Decay To

```python
# Separate parameters
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if 'bias' in name or 'norm' in name or 'embedding' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

# Parameter groups with different weight decay
optimizer = optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=0.001)
```

**Apply weight decay to:**
- Dense layer weights
- Convolutional kernels
- Attention weights

**Don't apply weight decay to:**
- Biases
- LayerNorm/BatchNorm parameters
- Embeddings (debated)

## Integration with Learning Rate Schedules

AdamW is commonly paired with warmup and cosine decay:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Warmup for 10 epochs, then cosine decay
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=90, eta_min=1e-6)

scheduler = SequentialLR(
    optimizer, 
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[10]
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

## When to Use AdamW

**Strongly recommended:**
- Transformer models (BERT, GPT, ViT)
- Any model where regularization is important
- Large-scale training
- Transfer learning / fine-tuning

**Also good for:**
- General deep learning (can replace Adam)
- CNNs where good generalization is needed
- Models prone to overfitting

### Use AdamW When:
- Training Transformers (BERT, GPT, ViT)
- You want regularization and adaptive rates
- Working on tasks where generalization matters
- Following modern best practices

### Use Adam When:
- Not using weight decay
- Legacy codebases requiring L2 regularization semantics
- Specific cases where L2 behavior is desired

## AdamW vs SGD for Final Performance

While AdamW is excellent for most applications, well-tuned SGD with momentum can still achieve better final performance on some vision tasks:

| Criterion | AdamW | SGD + Momentum |
|-----------|-------|----------------|
| Convergence speed | Faster | Slower |
| Tuning required | Minimal | More careful |
| Memory usage | 3× params | 2× params |
| Final accuracy (vision) | Very good | Often best |
| Final accuracy (NLP) | Best | Good |

A common strategy: train with AdamW initially, then optionally switch to SGD for final fine-tuning on vision tasks.

## Migration Guide

```python
# From Adam with L2:
optimizer = optim.Adam(params, lr=0.001, weight_decay=0.0001)

# To AdamW (increase weight_decay since it's not scaled):
optimizer = optim.AdamW(params, lr=0.001, weight_decay=0.01)
```

Note: The weight decay values are different because:
- Adam's L2 is scaled by adaptive rate (effectively smaller)
- AdamW's weight decay is applied directly

## Summary

| Aspect | Detail |
|--------|--------|
| Key Innovation | Decouples weight decay from adaptive gradient scaling |
| Problem Solved | L2 regularization is inconsistently applied in Adam |
| Update Rule | $\theta_t = \theta_{t-1} - \eta(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta_{t-1})$ |
| Benefits | Better generalization, proper regularization behavior |
| When to Use | Modern default for most deep learning tasks, especially transformers |
| Weight Decay | Typically 0.01-0.1 (higher than Adam's L2) |

## References

1. Loshchilov, I., & Hutter, F. (2017). "Decoupled Weight Decay Regularization"
2. Loshchilov, I., & Hutter, F. (2016). "SGDR: Stochastic Gradient Descent with Warm Restarts"
3. Zhang, M., et al. (2019). "Which Algorithmic Choices Matter at Which Batch Sizes?"
