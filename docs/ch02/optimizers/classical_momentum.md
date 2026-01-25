# Classical Momentum

## Overview

Momentum is one of the most important modifications to gradient descent, drawing inspiration from physics to accelerate optimization. Introduced by Polyak (1964), momentum helps navigate ravines in the loss landscape and reduces oscillation. It enhances SGD by accumulating gradient history, enabling faster convergence and improved stability in challenging loss landscapes.

## The Problem with Vanilla SGD

Standard SGD updates parameters using only the current gradient:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_t$$

This approach suffers from several issues:

1. **Oscillation in narrow valleys**: When gradients point in different directions across dimensions, SGD oscillates perpendicular to the optimal path
2. **Slow progress along consistent gradients**: Each step is independent, missing opportunities to accelerate
3. **Sensitivity to noisy gradients**: Random fluctuations cause erratic updates

## Physical Intuition

Consider a ball rolling down a hilly terrain:
- Without momentum: The ball moves only based on local slope, stopping instantly whenever the slope changes
- With momentum: The ball accumulates velocity and can roll through small bumps

This physical analogy translates directly to optimization: we accumulate a velocity vector that smooths the optimization trajectory, allowing the optimizer to:
- Roll through small bumps (local minima)
- Accelerate on consistent slopes
- Smooth out direction changes

## Mathematical Formulation

### Velocity Update

Momentum maintains a velocity vector $v_t$ that accumulates exponentially decaying gradients:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta \mathcal{L}(\theta_{t-1})$$

where:
- $\gamma \in [0, 1)$ is the momentum coefficient (typically 0.9)
- $\eta$ is the learning rate
- $v_0 = 0$ (initialized to zero)

### Parameter Update

The parameter update uses the accumulated velocity:

$$\theta_t = \theta_{t-1} - v_t$$

### Expanded Form

Unrolling the recursion shows how momentum combines past gradients:

$$v_t = \eta \sum_{i=0}^{t-1} \gamma^{t-1-i} \nabla_\theta \mathcal{L}(\theta_i)$$

This is an exponentially weighted average with more recent gradients having larger weights. Recent gradients contribute more than older ones, with contribution decaying as $\gamma^k$ for $k$ steps ago.

## Why Momentum Works

### Acceleration in Consistent Directions

When gradients consistently point in the same direction:

$$v_t \approx \frac{\eta}{1 - \gamma} g$$

where $g$ is the consistent gradient direction. This amplifies the effective learning rate by factor $\frac{1}{1-\gamma}$ (e.g., 10× for $\gamma = 0.9$).

### Damping Oscillations

In directions where gradients alternate signs (oscillation), the accumulated momentum cancels out:

$$v_t^{\text{oscillating}} \approx \frac{\eta(g - g)}{1 + \gamma} \approx 0$$

### Ravine Navigation

Consider a ravine-shaped loss surface:
- High curvature (steep walls) perpendicular to the ravine
- Low curvature (gentle slope) along the ravine

Without momentum: Oscillates across the ravine, slow progress along it

With momentum: Oscillations cancel, accumulates speed along the ravine

## Convergence Analysis

### Quadratic Functions

For minimizing $f(\theta) = \frac{1}{2}\theta^\top H \theta$ where $H$ has eigenvalues $0 < \mu \leq \lambda_i \leq L$:

**Optimal Parameters:**
$$\gamma^* = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2, \quad \eta^* = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}$$

where $\kappa = L/\mu$ is the condition number.

**Convergence Rate:**
$$\|\theta_t - \theta^*\| \leq \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^t \|\theta_0 - \theta^*\|$$

This improves upon gradient descent's rate from $O\left(\frac{\kappa-1}{\kappa+1}\right)^t$ to $O\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^t$.

### Improvement Factor

For ill-conditioned problems ($\kappa \gg 1$):
- GD: requires $O(\kappa \log(1/\epsilon))$ iterations
- Momentum: requires $O(\sqrt{\kappa} \log(1/\epsilon))$ iterations

This is a significant improvement for neural networks where $\kappa$ can be very large.

## Algorithm

```
Algorithm: SGD with Momentum
Input: Initial θ₀, learning rate η, momentum γ
Initialize: v₀ = 0

for t = 1, 2, ... do
    g_t ← ∇_θ L(θ_{t-1})           # Compute gradient
    v_t ← γ · v_{t-1} + η · g_t     # Update velocity
    θ_t ← θ_{t-1} - v_t             # Update parameters
end for
```

## PyTorch Implementation

### Using Built-in SGD with Momentum

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model setup
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# SGD without momentum
optimizer_vanilla = optim.SGD(model.parameters(), lr=0.01)

# SGD with momentum
optimizer_momentum = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9  # Momentum coefficient γ
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

class SGDMomentum(Optimizer):
    """
    SGD with classical (Polyak) momentum.
    
    v_t = γ * v_{t-1} + η * g_t
    θ_t = θ_{t-1} - v_t
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        momentum: Momentum coefficient γ (default: 0.9)
    """
    
    def __init__(self, params, lr=0.01, momentum=0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        
        defaults = dict(lr=lr, momentum=momentum)
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
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize velocity buffer
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(p)
                
                v = state['velocity']
                
                # Update velocity: v = γ * v + η * g
                v.mul_(momentum).add_(grad, alpha=lr)
                
                # Update parameters: θ = θ - v
                p.sub_(v)
        
        return loss
```

### NumPy Implementation

```python
import numpy as np

class MomentumSGD:
    """
    Classical momentum optimizer (NumPy implementation).
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.gamma = momentum
        self.velocity = {}
    
    def update(self, params, grads):
        """
        Update parameters using momentum.
        
        Args:
            params: dict of parameter arrays
            grads: dict of gradient arrays
        
        Returns:
            Updated parameters
        """
        updated = {}
        
        for key in params:
            # Initialize velocity if needed
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            # Update velocity
            self.velocity[key] = (
                self.gamma * self.velocity[key] + 
                self.lr * grads[key]
            )
            
            # Update parameter
            updated[key] = params[key] - self.velocity[key]
        
        return updated
```

## Demonstrations

### Demo 1: Comparing SGD vs Momentum

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train_and_record(model_class, optimizer_fn, X, y, epochs=500):
    model = model_class()
    optimizer = optimizer_fn(model.parameters())
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

# Generate data
torch.manual_seed(42)
X = torch.randn(100, 10)
y = X.sum(dim=1, keepdim=True) + torch.randn(100, 1) * 0.1

# Compare
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

losses_vanilla = train_and_record(
    SimpleNet, 
    lambda p: optim.SGD(p, lr=0.01), 
    X, y
)

losses_momentum = train_and_record(
    SimpleNet,
    lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
    X, y
)

print(f"Vanilla SGD final loss: {losses_vanilla[-1]:.4f}")
print(f"Momentum SGD final loss: {losses_momentum[-1]:.4f}")
```

### Demo 2: Ravine-Shaped Function

```python
import numpy as np

def demo_momentum():
    """
    Compare SGD with and without momentum on a ravine-shaped function.
    f(x, y) = 0.1*x² + y²  (elongated in x direction)
    """
    # Without momentum
    params_sgd = {'x': np.array([10.0]), 'y': np.array([1.0])}
    lr = 0.1
    
    # With momentum
    params_mom = {'x': np.array([10.0]), 'y': np.array([1.0])}
    optimizer = MomentumSGD(learning_rate=0.1, momentum=0.9)
    
    print("Iteration | SGD Loss     | Momentum Loss")
    print("-" * 45)
    
    for i in range(50):
        # Gradients: df/dx = 0.2*x, df/dy = 2*y
        grad_sgd = {
            'x': 0.2 * params_sgd['x'],
            'y': 2.0 * params_sgd['y']
        }
        grad_mom = {
            'x': 0.2 * params_mom['x'],
            'y': 2.0 * params_mom['y']
        }
        
        # SGD update
        params_sgd['x'] = params_sgd['x'] - lr * grad_sgd['x']
        params_sgd['y'] = params_sgd['y'] - lr * grad_sgd['y']
        
        # Momentum update
        params_mom = optimizer.update(params_mom, grad_mom)
        
        # Compute losses
        loss_sgd = 0.1 * params_sgd['x']**2 + params_sgd['y']**2
        loss_mom = 0.1 * params_mom['x']**2 + params_mom['y']**2
        
        if i % 10 == 0:
            print(f"{i:9d} | {loss_sgd[0]:12.6f} | {loss_mom[0]:12.6f}")
    
    print("\nMomentum converges faster along the ravine (x direction)!")


if __name__ == "__main__":
    demo_momentum()
```

### Demo 3: Visualization on Rosenbrock Function

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_momentum():
    """Visualize optimization trajectories with and without momentum."""
    
    # Rosenbrock function (challenging landscape)
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def grad_rosenbrock(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return dx, dy
    
    # Track trajectories
    traj_sgd = [(-1.0, -1.0)]
    traj_mom = [(-1.0, -1.0)]
    
    # SGD
    x, y = -1.0, -1.0
    lr = 0.001
    for _ in range(1000):
        dx, dy = grad_rosenbrock(x, y)
        x -= lr * dx
        y -= lr * dy
        traj_sgd.append((x, y))
    
    # Momentum
    x, y = -1.0, -1.0
    vx, vy = 0.0, 0.0
    gamma = 0.9
    for _ in range(1000):
        dx, dy = grad_rosenbrock(x, y)
        vx = gamma * vx + lr * dx
        vy = gamma * vy + lr * dy
        x -= vx
        y -= vy
        traj_mom.append((x, y))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create contour plot
    X = np.linspace(-1.5, 1.5, 100)
    Y = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = rosenbrock(X, Y)
    
    for ax, traj, title in [
        (axes[0], traj_sgd, 'SGD (no momentum)'),
        (axes[1], traj_mom, 'SGD with Momentum (γ=0.9)')
    ]:
        ax.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='viridis')
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], 'r.-', alpha=0.5, markersize=2)
        ax.plot(1, 1, 'g*', markersize=15)  # Optimum
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig('momentum_comparison.png', dpi=150)
    plt.show()
```

## Effect of Momentum Coefficient

The momentum coefficient $\gamma$ (or $\beta$) controls how much history to retain:

| Value | Effect |
|-------|--------|
| 0.0 | No momentum (vanilla SGD) |
| 0.5 | Moderate smoothing |
| 0.9 | Standard choice, good balance |
| 0.99 | Very heavy momentum, may overshoot |

```python
# Comparing different momentum values
for beta in [0.0, 0.5, 0.9, 0.99]:
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=beta)
    criterion = nn.MSELoss()
    
    for _ in range(200):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"β={beta:.2f}: Final loss = {loss.item():.4f}")
```

## Alternative Formulations

### Heavy Ball Method (Original Polyak)

The original formulation uses slightly different notation:

$$\theta_t = \theta_{t-1} - \eta \nabla_\theta \mathcal{L}(\theta_{t-1}) + \gamma(\theta_{t-1} - \theta_{t-2})$$

This is mathematically equivalent to our velocity formulation.

### PyTorch's Dampening Parameter

PyTorch's SGD includes an optional dampening parameter $\tau$:

$$v_t = \gamma v_{t-1} + (1 - \tau) g_t$$
$$\theta_t = \theta_{t-1} - \eta v_t$$

With $\tau = 0$ (default), this matches our formulation up to the placement of $\eta$.

```python
# Heavy ball method via dampening
optimizer = optim.SGD(
    model.parameters(), 
    lr=0.01, 
    momentum=0.9, 
    dampening=0.9  # Same as momentum
)
```

## Effective Learning Rate

With momentum, the effective step size increases. For consistent gradients, the accumulated update approaches:

$$\text{effective step} = \frac{\eta}{1 - \gamma}$$

With $\gamma = 0.9$, the effective learning rate is approximately 10× the nominal value. This means you may need to reduce the nominal learning rate when adding momentum.

## When Momentum Helps Most

Momentum provides the greatest benefit in:

1. **Ill-conditioned problems**: Where Hessian eigenvalues span multiple orders of magnitude
2. **Deep networks**: Where gradients must propagate through many layers
3. **Noisy gradients**: Small batch sizes or stochastic data
4. **Saddle points**: Momentum can escape more easily than vanilla SGD
5. **Ravine-shaped loss surfaces**: Oscillations cancel, speed accumulates

## Hyperparameter Selection

### Learning Rate Interaction

Momentum effectively increases the learning rate. When switching from SGD to momentum:
- Consider reducing learning rate
- Effective rate is approximately $\frac{\eta}{1-\gamma}$ in steady state

### Warm-up Strategies

For stability, consider:
1. Start with $\gamma = 0$ and increase to target
2. Use learning rate warm-up with momentum
3. Reduce learning rate when adding momentum

## Practical Guidelines

**Typical hyperparameters:**

- Learning rate: 0.01-0.1 (may need reduction from vanilla SGD)
- Momentum: 0.9 (standard starting point)
- Dampening: 0 (default)

```python
# Recommended starting configuration
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

**When to use momentum:**

- Deep neural networks (always beneficial)
- Convolutional networks (standard practice)
- When vanilla SGD oscillates
- When training is slow but stable

**When to be cautious:**

- Very high momentum (>0.95) can overshoot
- May need to reduce learning rate from vanilla SGD
- Combined with learning rate warmup for large models

## Common Issues and Solutions

### Overshooting

**Problem:** Momentum carries past the minimum and oscillates.

**Solutions:**
- Reduce momentum coefficient $\gamma$
- Use learning rate decay
- Try Nesterov momentum (see separate documentation)

### Poor Conditioning Interactions

**Problem:** Different parameters have vastly different scales.

**Solutions:**
- Use adaptive methods (Adam, RMSprop)
- Normalize gradients
- Per-layer learning rates

### Stale Momentum

**Problem:** In non-stationary settings, old gradients may mislead.

**Solutions:**
- Reduce $\gamma$ for non-stationary problems
- Reset momentum periodically
- Use restart strategies

## Summary

| Aspect | Detail |
|--------|--------|
| Key Innovation | Accumulate gradient direction over time |
| Update Rule | $v_t = \gamma v_{t-1} + \eta g_t$; $\theta_t = \theta_{t-1} - v_t$ |
| Hyperparameters | $\eta$ (learning rate), $\gamma$ (momentum, typically 0.9) |
| Benefits | Accelerates convergence, damps oscillations, escapes saddle points |
| Drawbacks | Can overshoot, adds one hyperparameter |
| Effective LR | Approximately $\frac{\eta}{1-\gamma}$ in steady state |
| Best For | Most optimization problems, especially with ravines |

## References

1. Polyak, B. T. (1964). "Some methods of speeding up the convergence of iteration methods"
2. Sutskever, I., et al. (2013). "On the importance of initialization and momentum in deep learning"
3. Goh, G. (2017). "Why Momentum Really Works" - Distill
