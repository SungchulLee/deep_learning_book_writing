# AMSGrad

## Overview

AMSGrad was proposed by Reddi, Kale, and Kumar in their 2018 paper "On the Convergence of Adam and Beyond." It addresses a theoretical convergence issue with Adam where the algorithm can fail to converge to the optimal solution in certain convex settings. AMSGrad fixes this by maintaining the maximum of all past squared gradients rather than the exponential moving average, ensuring the effective learning rate is monotonically non-increasing.

## The Convergence Problem with Adam

### Theoretical Issue

Reddi et al. demonstrated that Adam can fail to converge even on simple convex problems. The issue arises from the non-monotonic decay of the effective learning rate.

In Adam, the second moment estimate can decrease:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

If recent gradients are small, $v_t$ decreases, causing the effective learning rate to increase. In pathological cases, this can prevent convergence.

### Illustrative Example

Consider a simple online convex optimization problem where:

$$f_t(x) = \begin{cases} Cx & \text{for } t \mod 3 = 1 \\ -x & \text{otherwise} \end{cases}$$

with $C > 2$. The optimal solution is $x^* = -1$ (boundary), but Adam can converge to $x^* = +1$ instead.

### Root Cause

In Adam, the second moment estimate $v_t$ can decrease when recent gradients are smaller than past gradients. This can cause the effective learning rate to increase at inopportune times, leading to divergence from the optimum.

**Adam's problematic behavior:**
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

If $g_t^2 < v_{t-1}$, then $v_t < v_{t-1}$, causing the learning rate to increase.

**Example scenario:**
1. Large gradients early → large $v_t$ → small effective LR
2. Small gradients later → $v_t$ decreases → large effective LR
3. Sudden large gradient → overshoots due to large effective LR

## AMSGrad Solution

### Key Modification

AMSGrad maintains the maximum of all past second moment estimates:

$$\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$$

This ensures the effective learning rate can only decrease (or stay the same), never increase, providing the monotonic decay needed for convergence guarantees.

### Full Update Rules

**First moment (unchanged from Adam):**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**Second moment (unchanged from Adam):**
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

**Maximum second moment (new in AMSGrad):**
$$\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$$

**Parameter update:**
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} m_t$$

Note: The original AMSGrad paper doesn't apply bias correction to $m_t$ in the update, but implementations often include it.

## Algorithm

```
Algorithm: AMSGrad
Input: Initial θ₀, learning rate η, β₁=0.9, β₂=0.999, ε=10⁻⁸
Initialize: m₀ = 0, v₀ = 0, v̂₀ = 0

for t = 1, 2, ... do
    g_t ← ∇_θ L(θ_{t-1})                           # Compute gradient
    m_t ← β₁ · m_{t-1} + (1 - β₁) · g_t            # Update first moment
    v_t ← β₂ · v_{t-1} + (1 - β₂) · g_t²           # Update second moment
    v̂_t ← max(v̂_{t-1}, v_t)                        # Maximum second moment
    θ_t ← θ_{t-1} - η · m_t / (√v̂_t + ε)          # Update parameters
end for
```

## Theoretical Properties

### Convergence Guarantee

For convex functions with bounded gradients $\|g_t\|_\infty \leq G$, AMSGrad achieves:

$$\text{Regret}(T) = O\left(\sqrt{T}\right)$$

This matches the optimal rate for online convex optimization, unlike Adam which may not converge.

### Key Theorem (Informal)

AMSGrad converges to the optimal solution for convex problems because the effective learning rate:

$$\alpha_t = \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$$

is non-increasing: $\alpha_{t+1} \leq \alpha_t$.

This monotonicity is crucial for convergence guarantees.

## PyTorch Implementation

### Using Built-in Adam with AMSGrad

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Enable AMSGrad variant of Adam
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    amsgrad=True  # Enable AMSGrad
)

# Also available with AdamW
optimizer_w = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
    amsgrad=True
)

# Training loop (unchanged)
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
```

### From-Scratch PyTorch Implementation

```python
import torch
from torch.optim import Optimizer

class AMSGrad(Optimizer):
    """
    AMSGrad optimizer implementation.
    
    Maintains maximum of all past squared gradient averages to ensure
    non-increasing effective learning rates, fixing Adam's convergence issues.
    
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
                    state['exp_avg'] = torch.zeros_like(p)       # m
                    state['exp_avg_sq'] = torch.zeros_like(p)    # v
                    state['max_exp_avg_sq'] = torch.zeros_like(p) # v̂ (max)
                
                m = state['exp_avg']
                v = state['exp_avg_sq']
                v_max = state['max_exp_avg_sq']
                
                state['step'] += 1
                t = state['step']
                
                # Update first moment
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update second moment
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Update maximum second moment (key AMSGrad modification)
                torch.max(v_max, v, out=v_max)
                
                # Bias correction for first moment
                bias_correction1 = 1 - beta1 ** t
                m_hat = m / bias_correction1
                
                # Note: Some implementations don't bias-correct v_max
                # Here we use v_max directly as in the original paper
                
                # Update parameters using maximum v
                denom = v_max.sqrt().add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)
        
        return loss
```

### NumPy Implementation

```python
import numpy as np

class AMSGrad:
    """
    AMSGrad optimizer implementation (NumPy).
    
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
        
        self.m = {}       # First moment
        self.v = {}       # Second moment
        self.v_max = {}   # Maximum second moment
        self.t = 0
    
    def update(self, params, grads):
        """
        Update parameters using AMSGrad algorithm.
        """
        self.t += 1
        updated_params = {}
        
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
                self.v_max[key] = np.zeros_like(params[key])
            
            # Update first moment
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update second moment
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Update maximum second moment (AMSGrad's key modification)
            self.v_max[key] = np.maximum(self.v_max[key], self.v[key])
            
            # Bias correction for first moment
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Update parameters using maximum v
            updated_params[key] = (
                params[key] - 
                self.lr * m_hat / (np.sqrt(self.v_max[key]) + self.epsilon)
            )
        
        return updated_params
```

## Demonstrations

### Demo 1: Monotonic Learning Rate

```python
import numpy as np

def demo_amsgrad_vs_adam():
    """
    Demonstrate AMSGrad's monotonic effective learning rate vs Adam.
    """
    print("=" * 70)
    print("AMSGrad vs Adam: Effective Learning Rate Behavior")
    print("=" * 70)
    print()
    
    # Simulate varying gradient magnitudes
    np.random.seed(42)
    
    # Initialize
    x_adam = np.array([5.0])
    x_amsgrad = np.array([5.0])
    
    m_adam = np.array([0.0])
    v_adam = np.array([0.0])
    
    m_ams = np.array([0.0])
    v_ams = np.array([0.0])
    v_max = np.array([0.0])
    
    lr = 0.1
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    print(f"{'Step':<8} {'Adam v':<15} {'AMS v_max':<15} {'Adam eff_lr':<15} {'AMS eff_lr':<15}")
    print("-" * 70)
    
    # Simulate gradients that vary in magnitude
    for t in range(1, 51):
        # Large gradient every 10 steps, small otherwise
        if t % 10 == 1:
            grad = np.array([10.0])  # Large gradient
        else:
            grad = np.array([0.1])   # Small gradient
        
        # Adam update
        m_adam = beta1 * m_adam + (1 - beta1) * grad
        v_adam = beta2 * v_adam + (1 - beta2) * (grad ** 2)
        m_hat_adam = m_adam / (1 - beta1 ** t)
        v_hat_adam = v_adam / (1 - beta2 ** t)
        eff_lr_adam = lr / (np.sqrt(v_hat_adam) + eps)
        x_adam = x_adam - lr * m_hat_adam / (np.sqrt(v_hat_adam) + eps)
        
        # AMSGrad update
        m_ams = beta1 * m_ams + (1 - beta1) * grad
        v_ams = beta2 * v_ams + (1 - beta2) * (grad ** 2)
        v_max = np.maximum(v_max, v_ams)
        m_hat_ams = m_ams / (1 - beta1 ** t)
        eff_lr_ams = lr / (np.sqrt(v_max) + eps)
        x_amsgrad = x_amsgrad - lr * m_hat_ams / (np.sqrt(v_max) + eps)
        
        if t % 5 == 0 or t == 1:
            print(f"{t:<8} {v_adam[0]:<15.6f} {v_max[0]:<15.6f} "
                  f"{eff_lr_adam[0]:<15.6f} {eff_lr_ams[0]:<15.6f}")
    
    print()
    print("Observation: Adam's v (and effective LR) can oscillate.")
    print("AMSGrad's v_max only increases, ensuring monotonic LR decay.")
```

### Demo 2: PyTorch Comparison

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(200, 10)
y = torch.sin(X.sum(dim=1, keepdim=True)) + torch.randn(200, 1) * 0.1

def train_and_compare(amsgrad, epochs=300):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=amsgrad)
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

adam_losses = train_and_compare(amsgrad=False)
amsgrad_losses = train_and_compare(amsgrad=True)

print(f"Adam final loss:    {adam_losses[-1]:.6f}")
print(f"AMSGrad final loss: {amsgrad_losses[-1]:.6f}")
```

### Demo 3: Convergence Comparison with Noisy Gradients

```python
def demo_convergence():
    """
    Compare convergence of Adam vs AMSGrad on a constructed example.
    """
    print("=" * 70)
    print("Adam vs AMSGrad: Convergence Behavior")
    print("=" * 70)
    print()
    
    # Initialize
    params_adam = {'x': np.array([5.0])}
    params_amsgrad = {'x': np.array([5.0])}
    
    adam = Adam(learning_rate=0.1)
    amsgrad = AMSGrad(learning_rate=0.1)
    
    print(f"{'Step':<8} {'Adam x':<15} {'AMSGrad x':<15} {'Adam f(x)':<15} {'AMS f(x)':<15}")
    print("-" * 70)
    
    # Optimization with varying gradient noise
    np.random.seed(42)
    for t in range(1, 101):
        # Gradient with occasional spikes
        base_grad = 2 * params_adam['x']  # df/dx for f(x) = x²
        
        # Add gradient noise
        if t % 7 == 0:
            noise = np.array([np.random.randn() * 5])
        else:
            noise = np.array([np.random.randn() * 0.1])
        
        grads_adam = {'x': base_grad + noise}
        grads_amsgrad = {'x': 2 * params_amsgrad['x'] + noise}
        
        params_adam = adam.update(params_adam, grads_adam)
        params_amsgrad = amsgrad.update(params_amsgrad, grads_amsgrad)
        
        f_adam = params_adam['x'] ** 2
        f_amsgrad = params_amsgrad['x'] ** 2
        
        if t % 20 == 0 or t == 1:
            print(f"{t:<8} {params_adam['x'][0]:<15.6f} {params_amsgrad['x'][0]:<15.6f} "
                  f"{f_adam[0]:<15.6f} {f_amsgrad[0]:<15.6f}")
    
    print()
    print("Note: AMSGrad can be more conservative due to max operation.")
```

## Comparison: Adam vs AMSGrad

| Aspect | Adam | AMSGrad |
|--------|------|---------|
| Second moment | $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ | Same, plus $\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$ |
| Effective LR | Can increase | Only decreases |
| Memory | $O(2d)$ | $O(3d)$ |
| Convergence | May fail on convex | Guaranteed on convex |
| Practice | Often faster | Often slower |

## Memory Overhead

AMSGrad requires storing one additional buffer per parameter:

| Optimizer | Memory per Parameter |
|-----------|---------------------|
| SGD | 1× (gradients) |
| Adam | 3× (gradients + m + v) |
| AMSGrad | 4× (gradients + m + v + v_max) |

For large models, this 33% increase over Adam can be significant.

## Practical Considerations

### When AMSGrad Helps

The theoretical issues with Adam are rarely encountered in practice. AMSGrad may help when:

1. **Convex optimization problems**
2. **Training on adversarial examples**
3. **Non-stationary optimization landscapes**
4. **Highly non-stationary gradients**
5. **Tasks where Adam shows instability**
6. **When theoretical guarantees matter**

### When AMSGrad May Not Help

1. **Most practical deep learning tasks**
2. **When Adam already converges well**
3. **When faster convergence is preferred**
4. **Memory is constrained**

### Empirical Reality

Despite its theoretical advantages, AMSGrad often doesn't outperform Adam in practice:

- Deep learning loss landscapes differ from adversarial convex examples
- The additional conservatism can slow convergence
- Adam's "failures" are rare in typical neural network training

**Original paper claims:**
- Fixes convergence issues in synthetic examples
- Provides theoretical guarantees

**Follow-up studies found:**
- Minimal improvement on standard benchmarks
- Sometimes slightly worse than Adam
- Benefits mainly in constructed adversarial cases

## Hyperparameters

AMSGrad uses the same hyperparameters as Adam:

| Parameter | Default | Notes |
|-----------|---------|-------|
| lr | 0.001 | Same as Adam |
| betas | (0.9, 0.999) | Same as Adam |
| eps | 1e-8 | Same as Adam |
| amsgrad | True | Enable the fix |

## Variants and Extensions

### AdaBound

Combines AMSGrad with dynamic bounds on learning rate:

```python
# AdaBound clips effective learning rate to [lower, upper]
lower = final_lr * (1 - 1/(gamma*t + 1))
upper = final_lr * (1 + 1/(gamma*t))
```

### AdamNC (No Correction)

Some implementations skip bias correction with AMSGrad since the max operation provides some stabilization.

## When to Use AMSGrad

**Consider AMSGrad when:**
- Adam shows unexpected instability
- Working with adversarial training
- Theoretical convergence guarantees are important
- You observe non-convergence with Adam
- Convex problems where guarantees matter

**Stick with Adam/AdamW when:**
- Standard training scenarios
- Memory is constrained
- Empirical performance is the priority
- Adam is working well

## Summary

| Aspect | Detail |
|--------|--------|
| Key Innovation | Maintains maximum of past $v_t$ values |
| Problem Solved | Adam's non-convergence on some convex problems |
| Update Rule | $\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$; uses $\hat{v}_t$ instead of $v_t$ |
| Benefits | Guaranteed convergence on convex problems |
| Drawbacks | More memory (4× vs 3×), often slower in practice |
| When to Use | Convex problems, theoretical guarantees needed, Adam instability |
| PyTorch | `optim.Adam(..., amsgrad=True)` |

## References

1. Reddi, S. J., Kale, S., & Kumar, S. (2018). "On the Convergence of Adam and Beyond"
2. Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization"
3. Chen, X., et al. (2018). "Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks"
