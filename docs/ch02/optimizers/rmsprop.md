# RMSprop (Root Mean Square Propagation)

## Overview

RMSprop was proposed by Geoffrey Hinton in his Coursera course (Lecture 6.5, 2012). It addresses AdaGrad's main limitation—the monotonically decreasing learning rate—by using an exponential moving average of squared gradients instead of a cumulative sum.

## Motivation

### AdaGrad's Problem

In AdaGrad, the accumulated squared gradients $G_t = \sum_{s=1}^t g_s^2$ grows unboundedly, causing:

$$\lim_{t \to \infty} \frac{\eta}{\sqrt{G_t}} = 0$$

This means the learning rate effectively becomes zero, preventing further learning.

### RMSprop's Solution

Replace the sum with an exponential moving average (EMA):

$$v_t = \rho \cdot v_{t-1} + (1 - \rho) \cdot g_t^2$$

The EMA "forgets" old gradients, preventing the denominator from growing without bound.

## Mathematical Formulation

### Core Update Rules

**Moving average of squared gradients:**
$$v_t = \rho \cdot v_{t-1} + (1 - \rho) \cdot g_t \odot g_t$$

**Parameter update:**
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} \odot g_t$$

where:
- $\rho \in (0, 1)$ is the decay rate (typically 0.9 or 0.99)
- $\eta$ is the learning rate
- $\epsilon$ is a small constant for numerical stability
- All operations are element-wise

### Effective Learning Rate

The effective learning rate for parameter $i$ at step $t$:

$$\eta_{t,i}^{\text{eff}} = \frac{\eta}{\sqrt{v_{t,i}} + \epsilon}$$

Unlike AdaGrad, this can increase or decrease based on recent gradient magnitudes.

### Exponential Moving Average Properties

The EMA gives exponentially decaying weights to past squared gradients:

$$v_t = (1 - \rho) \sum_{s=1}^t \rho^{t-s} g_s^2$$

This means:
- Recent gradients have more influence
- Effective window size $\approx \frac{1}{1-\rho}$ (e.g., 10 steps for $\rho=0.9$)
- Old gradients are "forgotten"

## Comparison with AdaGrad

| Aspect | AdaGrad | RMSprop |
|--------|---------|---------|
| Squared gradient accumulation | Sum: $\sum_s g_s^2$ | EMA: $\rho v + (1-\rho)g^2$ |
| Learning rate trend | Only decreases | Can adapt both ways |
| Long training | May stall | Remains effective |
| Memory of old gradients | Complete | Exponentially decaying |
| Non-stationary problems | Poor | Good |

## Algorithm

```
Algorithm: RMSprop
Input: Initial θ₀, learning rate η, decay rate ρ, small constant ε
Initialize: v₀ = 0

for t = 1, 2, ... do
    g_t ← ∇_θ L(θ_{t-1})                          # Compute gradient
    v_t ← ρ · v_{t-1} + (1 - ρ) · g_t ⊙ g_t       # Update moving average
    θ_t ← θ_{t-1} - η · g_t / (√v_t + ε)          # Update parameters
end for
```

## PyTorch Implementation

### Using Built-in RMSprop

```python
import torch
import torch.optim as optim

model = YourModel()
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.01,           # Learning rate
    alpha=0.99,        # Decay rate (ρ)
    eps=1e-8,          # Epsilon for stability
    momentum=0,        # Optional momentum
    centered=False     # Whether to use centered RMSprop
)

# Training loop
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

class RMSprop(Optimizer):
    """
    RMSprop optimizer implementation.
    
    Uses exponential moving average of squared gradients:
    v_t = ρ * v_{t-1} + (1 - ρ) * g_t²
    θ_t = θ_{t-1} - η * g_t / (√v_t + ε)
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        alpha: Decay rate ρ (default: 0.99)
        eps: Small constant for numerical stability (default: 1e-8)
        momentum: Optional momentum factor (default: 0)
        centered: If True, use centered RMSprop (default: False)
    """
    
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, 
                 momentum=0, centered=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha < 0.0 or alpha >= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        
        defaults = dict(
            lr=lr, alpha=alpha, eps=eps,
            momentum=momentum, centered=centered
        )
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
            alpha = group['alpha']
            eps = group['eps']
            momentum = group['momentum']
            centered = group['centered']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p)
                    if momentum > 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    if centered:
                        state['grad_avg'] = torch.zeros_like(p)
                
                square_avg = state['square_avg']
                
                # Update moving average of squared gradients
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                
                if centered:
                    # Centered RMSprop: subtract mean squared gradient
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(eps)
                else:
                    avg = square_avg.sqrt().add_(eps)
                
                if momentum > 0:
                    # Apply momentum
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-lr)
                else:
                    p.addcdiv_(grad, avg, value=-lr)
        
        return loss
```

### NumPy Implementation

```python
import numpy as np

class RMSprop:
    """
    RMSprop optimizer implementation (NumPy).
    
    Parameters:
    -----------
    learning_rate : float, default=0.001
        Step size for parameter updates
    rho : float, default=0.9
        Decay rate for moving average of squared gradients
    epsilon : float, default=1e-8
        Small constant for numerical stability
    """
    
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}  # Moving average of squared gradients
    
    def update(self, params, grads):
        """
        Update parameters using RMSprop algorithm.
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameters to update
        grads : dict
            Dictionary of gradients for each parameter
        
        Returns:
        --------
        dict : Updated parameters
        """
        updated_params = {}
        
        for key in params.keys():
            # Initialize cache if not exists
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            
            # Update moving average of squared gradients
            self.cache[key] = (
                self.rho * self.cache[key] + 
                (1 - self.rho) * (grads[key] ** 2)
            )
            
            # Update parameters
            updated_params[key] = (
                params[key] - 
                self.learning_rate * grads[key] / 
                (np.sqrt(self.cache[key]) + self.epsilon)
            )
        
        return updated_params
```

## Demonstrations

### Demo 1: Basic Optimization

```python
def demo_rmsprop():
    """
    Demonstrate RMSprop on a simple quadratic function.
    Minimize f(x, y) = x² + y²
    """
    print("=" * 60)
    print("RMSprop Optimizer Demo")
    print("=" * 60)
    print("Minimizing f(x, y) = x² + y²")
    print()
    
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    optimizer = RMSprop(learning_rate=0.1, rho=0.9)
    
    print(f"{'Iteration':<12} {'x':<12} {'y':<12} {'f(x,y)':<12}")
    print("-" * 60)
    
    for i in range(50):
        grads = {
            'x': 2 * params['x'],
            'y': 2 * params['y']
        }
        params = optimizer.update(params, grads)
        f_val = params['x']**2 + params['y']**2
        
        if i % 10 == 0:
            print(f"{i:<12} {params['x'][0]:<12.6f} {params['y'][0]:<12.6f} {f_val[0]:<12.6f}")
    
    print(f"\nFinal: x = {params['x'][0]:.8f}, y = {params['y'][0]:.8f}")
```

### Demo 2: Comparison with SGD on Ill-Conditioned Problem

```python
def demo_rmsprop_vs_sgd():
    """
    Compare RMSprop vs SGD on an ill-conditioned problem.
    f(x, y) = 100x² + y² (condition number = 100)
    """
    print("=" * 60)
    print("RMSprop vs SGD on Ill-Conditioned Problem")
    print("=" * 60)
    print("Minimizing f(x, y) = 100x² + y²")
    print()
    
    # Initialize
    params_rmsprop = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_sgd = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    optimizer_rmsprop = RMSprop(learning_rate=0.1)
    lr_sgd = 0.001  # Must be small for SGD stability
    
    print(f"{'Iteration':<12} {'RMSprop f(x,y)':<20} {'SGD f(x,y)':<20}")
    print("-" * 60)
    
    for i in range(100):
        # Compute gradients
        grads_rmsprop = {
            'x': 200 * params_rmsprop['x'],
            'y': 2 * params_rmsprop['y']
        }
        grads_sgd = {
            'x': 200 * params_sgd['x'],
            'y': 2 * params_sgd['y']
        }
        
        # Updates
        params_rmsprop = optimizer_rmsprop.update(params_rmsprop, grads_rmsprop)
        params_sgd['x'] = params_sgd['x'] - lr_sgd * grads_sgd['x']
        params_sgd['y'] = params_sgd['y'] - lr_sgd * grads_sgd['y']
        
        # Compute function values
        f_rmsprop = 100 * params_rmsprop['x']**2 + params_rmsprop['y']**2
        f_sgd = 100 * params_sgd['x']**2 + params_sgd['y']**2
        
        if i % 20 == 0:
            print(f"{i:<12} {f_rmsprop[0]:<20.6f} {f_sgd[0]:<20.6f}")
    
    print("\nRMSprop handles different gradient scales automatically!")
```

### Demo 3: RMSprop vs AdaGrad on Long Training

```python
def demo_rmsprop_vs_adagrad():
    """
    Compare RMSprop vs AdaGrad over many iterations.
    Shows RMSprop's advantage in maintaining learning ability.
    """
    print("=" * 60)
    print("RMSprop vs AdaGrad: Long Training Comparison")
    print("=" * 60)
    
    # Initialize
    params_rmsprop = {'x': np.array([10.0])}
    params_adagrad = {'x': np.array([10.0])}
    
    optimizer_rmsprop = RMSprop(learning_rate=0.5, rho=0.9)
    optimizer_adagrad = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iter':<8} {'RMSprop x':<15} {'AdaGrad x':<15} {'RMS eff_lr':<15} {'Ada eff_lr':<15}")
    print("-" * 70)
    
    for i in range(200):
        # Same gradient for both
        grad_rms = 2 * params_rmsprop['x']
        grad_ada = 2 * params_adagrad['x']
        
        # Effective learning rates
        if 'x' in optimizer_rmsprop.cache and optimizer_rmsprop.cache['x'][0] > 0:
            eff_lr_rms = 0.5 / (np.sqrt(optimizer_rmsprop.cache['x'][0]) + 1e-8)
        else:
            eff_lr_rms = 0.5
        
        if 'x' in optimizer_adagrad.cache and optimizer_adagrad.cache['x'][0] > 0:
            eff_lr_ada = 1.0 / (np.sqrt(optimizer_adagrad.cache['x'][0]) + 1e-8)
        else:
            eff_lr_ada = 1.0
        
        # Updates
        params_rmsprop = optimizer_rmsprop.update(params_rmsprop, {'x': grad_rms})
        params_adagrad = optimizer_adagrad.update(params_adagrad, {'x': grad_ada})
        
        if i % 40 == 0:
            print(f"{i:<8} {params_rmsprop['x'][0]:<15.8f} {params_adagrad['x'][0]:<15.8f} "
                  f"{eff_lr_rms:<15.10f} {eff_lr_ada:<15.10f}")
    
    print("\nObservation: RMSprop maintains reasonable effective learning rate!")
    print("AdaGrad's effective LR approaches zero over time.")
```

## Centered RMSprop

### Motivation

Standard RMSprop normalizes by the root mean square of gradients. Centered RMSprop normalizes by the variance instead, which can provide better conditioning.

### Formulation

**Mean gradient (first moment):**
$$\mu_t = \rho \cdot \mu_{t-1} + (1 - \rho) \cdot g_t$$

**Mean squared gradient (second raw moment):**
$$v_t = \rho \cdot v_{t-1} + (1 - \rho) \cdot g_t^2$$

**Variance (centered second moment):**
$$\tilde{v}_t = v_t - \mu_t^2$$

**Parameter update:**
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\tilde{v}_t} + \epsilon} \odot g_t$$

### When to Use Centered RMSprop

- When gradient means are far from zero
- Can improve stability in some cases
- Slightly more computation and memory

```python
# Enable centered RMSprop in PyTorch
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
    centered=True  # Use centered version
)
```

## RMSprop with Momentum

RMSprop can be combined with momentum:

$$v_t = \rho \cdot v_{t-1} + (1 - \rho) \cdot g_t^2$$
$$m_t = \gamma \cdot m_{t-1} + \frac{g_t}{\sqrt{v_t} + \epsilon}$$
$$\theta_t = \theta_{t-1} - \eta \cdot m_t$$

This combines:
- Adaptive learning rates (RMSprop)
- Momentum for acceleration

```python
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
    momentum=0.9  # Add momentum
)
```

## Hyperparameter Guidelines

### Decay Rate ($\rho$)

| Value | Effective Window | Use Case |
|-------|------------------|----------|
| 0.9 | ~10 steps | Rapidly changing gradients |
| 0.99 | ~100 steps | Standard choice |
| 0.999 | ~1000 steps | Very stable gradients |

### Learning Rate

| Scenario | Recommended $\eta$ |
|----------|-------------------|
| General | 0.001 - 0.01 |
| RNNs | 0.0001 - 0.001 |
| Fine-tuning | 0.0001 |

### Common Configurations

```python
# Standard RMSprop
optim.RMSprop(params, lr=0.001, alpha=0.99)

# RMSprop for RNNs (Hinton's recommendation)
optim.RMSprop(params, lr=0.0001, alpha=0.9)

# RMSprop with momentum
optim.RMSprop(params, lr=0.001, alpha=0.99, momentum=0.9)
```

## When to Use RMSprop

### Recommended Scenarios

1. **Recurrent Neural Networks (RNNs):** RMSprop was specifically designed for RNN training
2. **Non-stationary objectives:** The forgetting mechanism helps adapt to changing distributions
3. **Online learning:** Suitable for continuous data streams
4. **When AdaGrad stalls:** Natural replacement

### Comparison with Adam

| Aspect | RMSprop | Adam |
|--------|---------|------|
| First moment | No (unless with momentum) | Yes |
| Bias correction | No | Yes |
| Parameters | Simpler | More hyperparameters |
| Performance | Good | Often slightly better |
| Memory | Less | More |

## Connection to Other Methods

### Relationship to AdaGrad

RMSprop is essentially "leaky" AdaGrad:

$$\text{AdaGrad: } G_t = G_{t-1} + g_t^2$$
$$\text{RMSprop: } v_t = \rho \cdot v_{t-1} + (1-\rho) \cdot g_t^2$$

When $\rho = 1 - \frac{1}{t}$, RMSprop approximates AdaGrad's average.

### Foundation for Adam

Adam combines RMSprop (second moment) with momentum (first moment) and adds bias correction:

$$\text{Adam} = \text{RMSprop} + \text{Momentum} + \text{Bias Correction}$$

## Summary

| Aspect | Detail |
|--------|--------|
| Key Innovation | Exponential moving average of squared gradients |
| Update Rule | $v_t = \rho v_{t-1} + (1-\rho)g_t^2$; $\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t}+\epsilon}g_t$ |
| Hyperparameters | $\eta$ (LR, ~0.001), $\rho$ (decay, 0.9-0.99), $\epsilon$ ($10^{-8}$) |
| Benefits | Fixes AdaGrad's diminishing LR, adapts to non-stationary problems |
| Drawbacks | No momentum by default, no bias correction |
| Best For | RNNs, non-stationary objectives, online learning |

## References

1. Tieleman, T., & Hinton, G. (2012). "Lecture 6.5 - RMSprop" COURSERA: Neural Networks for Machine Learning
2. Graves, A. (2013). "Generating Sequences with Recurrent Neural Networks"
3. Hinton, G. (2012). Neural Networks for Machine Learning (Coursera course materials)
