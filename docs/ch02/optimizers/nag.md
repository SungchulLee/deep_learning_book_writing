# Nesterov Accelerated Gradient (NAG)

## Overview

Nesterov Accelerated Gradient (NAG), also known as Nesterov momentum, is a modification of classical momentum that provides faster convergence by computing gradients at a "lookahead" position. Proposed by Yurii Nesterov (1983), it achieves optimal convergence rates for convex optimization.

## Motivation

The key insight of NAG is that classical momentum has a limitation: it computes the gradient at the current position $\theta_{t-1}$, but we know we're going to move in the direction of the accumulated momentum $\gamma v_{t-1}$. Why not compute the gradient at where we're going to be?

### The "Look-Ahead" Intuition

Consider rolling a ball down a hill:
- **Classical Momentum:** Compute slope at current position, then move
- **Nesterov:** First take a "test step" in the momentum direction, compute slope there, then adjust

This allows NAG to "correct" for overshooting before it happens.

## Mathematical Formulation

### Standard Form

The NAG update consists of two steps:

**Look-ahead position:**
$$\tilde{\theta}_t = \theta_{t-1} - \gamma v_{t-1}$$

**Velocity and parameter update:**
$$v_t = \gamma v_{t-1} + \eta \nabla_\theta \mathcal{L}(\tilde{\theta}_t)$$
$$\theta_t = \theta_{t-1} - v_t$$

### Equivalent Reformulation

For implementation efficiency, NAG can be rewritten in terms of a different variable $\phi_t = \theta_t - \gamma v_t$:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta \mathcal{L}(\theta_{t-1} - \gamma v_{t-1})$$
$$\theta_t = \theta_{t-1} - v_t$$

This avoids computing the look-ahead explicitly while maintaining mathematical equivalence.

### Comparison with Classical Momentum

| Aspect | Classical Momentum | Nesterov Momentum |
|--------|-------------------|-------------------|
| Gradient computed at | $\theta_{t-1}$ | $\theta_{t-1} - \gamma v_{t-1}$ |
| Responsiveness | Reacts after overshooting | Anticipates and corrects |
| Convergence rate | $O(1/t)$ for convex | $O(1/t^2)$ for convex |

## Convergence Analysis

### Convex Functions

For a convex function with $L$-Lipschitz continuous gradients:

**NAG achieves:**
$$f(\theta_T) - f(\theta^*) \leq \frac{2L\|\theta_0 - \theta^*\|^2}{(T+1)^2}$$

This is the optimal rate for first-order methods, improving upon gradient descent's $O(1/T)$ rate.

### Strongly Convex Functions

For $\mu$-strongly convex functions with condition number $\kappa = L/\mu$:

**NAG achieves:**
$$\|\theta_T - \theta^*\|^2 \leq \left(1 - \frac{1}{\sqrt{\kappa}}\right)^T \|\theta_0 - \theta^*\|^2$$

Compared to gradient descent's $(1 - 1/\kappa)^T$ rate, this is a significant improvement when $\kappa$ is large.

### Optimal Parameter Settings

For optimal convergence on quadratic functions:

$$\gamma = \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}}, \quad \eta = \frac{1}{L}$$

## Algorithm

```
Algorithm: Nesterov Accelerated Gradient
Input: Initial θ₀, learning rate η, momentum γ
Initialize: v₀ = 0

for t = 1, 2, ... do
    θ̃_t ← θ_{t-1} - γ · v_{t-1}      # Look-ahead position
    g_t ← ∇_θ L(θ̃_t)                  # Gradient at look-ahead
    v_t ← γ · v_{t-1} + η · g_t        # Update velocity
    θ_t ← θ_{t-1} - v_t                # Update parameters
end for
```

## PyTorch Implementation

### Using Built-in SGD with Nesterov

```python
import torch
import torch.optim as optim

model = YourModel()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True  # Enable Nesterov momentum
)

# Standard training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
```

### From-Scratch Implementation

```python
import torch
from torch.optim import Optimizer

class NesterovSGD(Optimizer):
    """
    SGD with Nesterov Accelerated Gradient.
    
    Computes gradient at look-ahead position:
    θ̃ = θ - γ * v
    v = γ * v + η * ∇L(θ̃)
    θ = θ - v
    
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
        """
        Performs a single optimization step.
        
        Note: PyTorch's implementation uses an equivalent reformulation
        that computes gradients at the current position but adjusts
        the update formula. Here we implement the conceptually clearer
        version that explicitly computes at the look-ahead position.
        """
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
                
                state = self.state[p]
                
                # Initialize velocity buffer
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(p)
                
                v = state['velocity']
                
                # PyTorch-style Nesterov update (equivalent reformulation)
                # This uses the gradient at current position but adjusts the formula
                grad = p.grad
                
                # v_new = γ * v + g
                # θ_new = θ - η * (γ * v_new + g)
                #       = θ - η * (γ² * v + γ * g + g)
                #       = θ - η * γ² * v - η * (1 + γ) * g
                
                v.mul_(momentum).add_(grad)
                p.add_(v, alpha=-lr * momentum)
                p.add_(grad, alpha=-lr)
        
        return loss


class NesterovSGDExplicit(Optimizer):
    """
    Nesterov SGD with explicit look-ahead computation.
    
    This version explicitly computes gradients at the look-ahead position,
    which requires a closure that can be called multiple times.
    
    More conceptually clear but less efficient than the reformulation.
    """
    
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self._params_list = list(self.param_groups[0]['params'])
    
    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
                    Required for this explicit implementation.
        """
        lr = self.param_groups[0]['lr']
        momentum = self.param_groups[0]['momentum']
        
        # Step 1: Move to look-ahead position
        for p in self._params_list:
            state = self.state[p]
            if 'velocity' not in state:
                state['velocity'] = torch.zeros_like(p)
            v = state['velocity']
            p.sub_(v, alpha=momentum)  # θ̃ = θ - γ * v
        
        # Step 2: Compute gradient at look-ahead position
        with torch.enable_grad():
            loss = closure()
        
        # Step 3: Update velocity and parameters
        for p in self._params_list:
            if p.grad is None:
                continue
            state = self.state[p]
            v = state['velocity']
            
            # Move back from look-ahead
            p.add_(v, alpha=momentum)  # θ = θ̃ + γ * v
            
            # Update velocity
            v.mul_(momentum).add_(p.grad, alpha=lr)
            
            # Update parameter
            p.sub_(v)
        
        return loss
```

### NumPy Implementation

```python
import numpy as np

class NesterovMomentum:
    """
    Nesterov Accelerated Gradient optimizer (NumPy implementation).
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.gamma = momentum
        self.velocity = {}
    
    def update(self, params, grad_fn):
        """
        Update parameters using Nesterov momentum.
        
        Args:
            params: dict of parameter arrays
            grad_fn: function that takes params and returns gradients
        
        Returns:
            Updated parameters
        """
        updated = {}
        
        # Compute look-ahead position
        lookahead = {}
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            lookahead[key] = params[key] - self.gamma * self.velocity[key]
        
        # Compute gradient at look-ahead
        grads = grad_fn(lookahead)
        
        # Update velocity and parameters
        for key in params:
            self.velocity[key] = (
                self.gamma * self.velocity[key] + 
                self.lr * grads[key]
            )
            updated[key] = params[key] - self.velocity[key]
        
        return updated


def demo_nesterov_vs_momentum():
    """
    Compare Nesterov vs classical momentum on overshooting scenario.
    """
    print("=" * 60)
    print("Nesterov vs Classical Momentum")
    print("=" * 60)
    print("Minimizing f(x) = x² (watching overshooting behavior)")
    print()
    
    # Classical momentum
    x_classical = np.array([10.0])
    v_classical = np.array([0.0])
    
    # Nesterov momentum
    x_nesterov = np.array([10.0])
    v_nesterov = np.array([0.0])
    
    lr = 0.1
    gamma = 0.9
    
    print(f"{'Iter':<6} {'Classical x':<15} {'Nesterov x':<15} {'Classical f':<15} {'Nesterov f':<15}")
    print("-" * 66)
    
    for i in range(30):
        # Classical: gradient at current position
        grad_classical = 2 * x_classical
        v_classical = gamma * v_classical + lr * grad_classical
        x_classical = x_classical - v_classical
        
        # Nesterov: gradient at look-ahead position
        lookahead = x_nesterov - gamma * v_nesterov
        grad_nesterov = 2 * lookahead
        v_nesterov = gamma * v_nesterov + lr * grad_nesterov
        x_nesterov = x_nesterov - v_nesterov
        
        f_classical = x_classical ** 2
        f_nesterov = x_nesterov ** 2
        
        if i % 5 == 0:
            print(f"{i:<6} {x_classical[0]:<15.6f} {x_nesterov[0]:<15.6f} "
                  f"{f_classical[0]:<15.8f} {f_nesterov[0]:<15.8f}")
    
    print()
    print("Notice: Nesterov typically overshoots less and converges faster!")


if __name__ == "__main__":
    demo_nesterov_vs_momentum()
```

## Geometric Interpretation

### The "Correction" Effect

Consider the update geometrically:

1. **Classical Momentum:** Adds gradient to momentum → May overshoot
2. **Nesterov:** Evaluates gradient after tentative momentum step → Can brake before overshooting

```
Position Timeline:

Classical Momentum:
θ₀ ----[g₀]----> θ₁ ----[g₁ + γv₀]----> θ₂ (may overshoot)

Nesterov:
θ₀ --[γv₀]--> θ̃₀ --[g(θ̃₀)]--> compute --> θ₁ (corrected)
```

### Phase Space View

In the (θ, v) phase space:
- Classical momentum: Spirals toward equilibrium
- Nesterov: Tighter spiral with faster convergence

## Practical Considerations

### When to Use Nesterov

**Advantages:**
- Better theoretical convergence guarantees
- Reduced overshooting
- Often slightly better empirical performance

**Recommended scenarios:**
- Convex or nearly convex problems
- When momentum tends to overshoot
- Fine-tuning after initial training

### Hyperparameter Settings

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Learning rate $\eta$ | 0.01 - 0.1 | May need tuning |
| Momentum $\gamma$ | 0.9 - 0.99 | 0.9 is standard |

### Interaction with Learning Rate Schedules

Nesterov momentum works well with:
- Step decay
- Cosine annealing
- Warm restarts (SGDR)

## Nesterov in Modern Optimizers

### NAdam: Nesterov + Adam

The NAdam optimizer incorporates Nesterov momentum into Adam:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$\hat{m}_t = \frac{\beta_1 m_t + (1 - \beta_1) g_t}{1 - \beta_1^t}$$

This look-ahead is applied to the first moment estimate.

### Lookahead Optimizer

The Lookahead optimizer generalizes the look-ahead concept:
- Maintain slow and fast weights
- Periodically update slow weights toward fast weights
- Can wrap any optimizer (including NAG)

## Comparison with Other Methods

```python
def compare_optimizers():
    """Compare SGD, Momentum, and Nesterov on ill-conditioned problem."""
    
    def f(x, y):
        return 0.1 * x**2 + 10 * y**2  # Condition number = 100
    
    def grad_f(x, y):
        return 0.2 * x, 20 * y
    
    results = {'SGD': [], 'Momentum': [], 'Nesterov': []}
    
    # Initialize
    x = {'SGD': 10.0, 'Momentum': 10.0, 'Nesterov': 10.0}
    y = {'SGD': 10.0, 'Momentum': 10.0, 'Nesterov': 10.0}
    v_x = {'Momentum': 0.0, 'Nesterov': 0.0}
    v_y = {'Momentum': 0.0, 'Nesterov': 0.0}
    
    lr, gamma = 0.01, 0.9
    
    for _ in range(100):
        # SGD
        gx, gy = grad_f(x['SGD'], y['SGD'])
        x['SGD'] -= lr * gx
        y['SGD'] -= lr * gy
        results['SGD'].append(f(x['SGD'], y['SGD']))
        
        # Momentum
        gx, gy = grad_f(x['Momentum'], y['Momentum'])
        v_x['Momentum'] = gamma * v_x['Momentum'] + lr * gx
        v_y['Momentum'] = gamma * v_y['Momentum'] + lr * gy
        x['Momentum'] -= v_x['Momentum']
        y['Momentum'] -= v_y['Momentum']
        results['Momentum'].append(f(x['Momentum'], y['Momentum']))
        
        # Nesterov
        lx = x['Nesterov'] - gamma * v_x['Nesterov']
        ly = y['Nesterov'] - gamma * v_y['Nesterov']
        gx, gy = grad_f(lx, ly)
        v_x['Nesterov'] = gamma * v_x['Nesterov'] + lr * gx
        v_y['Nesterov'] = gamma * v_y['Nesterov'] + lr * gy
        x['Nesterov'] -= v_x['Nesterov']
        y['Nesterov'] -= v_y['Nesterov']
        results['Nesterov'].append(f(x['Nesterov'], y['Nesterov']))
    
    return results
```

## Summary

| Aspect | Detail |
|--------|--------|
| Key Innovation | Compute gradient at look-ahead position |
| Update Rule | $\tilde{\theta} = \theta - \gamma v$; $v = \gamma v + \eta \nabla\mathcal{L}(\tilde{\theta})$; $\theta = \theta - v$ |
| Convergence | Optimal $O(1/t^2)$ for convex functions |
| Benefits | Reduced overshooting, faster convergence |
| When to Use | Most problems where momentum is appropriate |
| Implementation | `nesterov=True` in PyTorch's SGD |

## References

1. Nesterov, Y. (1983). "A method for unconstrained convex minimization problem with the rate of convergence O(1/k²)"
2. Sutskever, I., et al. (2013). "On the importance of initialization and momentum in deep learning"
3. Dozat, T. (2016). "Incorporating Nesterov Momentum into Adam"
