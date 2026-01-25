# AdaGrad (Adaptive Gradient)

## Overview

AdaGrad (Adaptive Gradient Algorithm) was introduced by Duchi, Hazan, and Singer in 2011. It adapts the learning rate for each parameter individually based on the historical sum of squared gradients, making it particularly effective for sparse data and features that appear with different frequencies.

## Motivation

In many machine learning problems, different features appear with vastly different frequencies:

- **Sparse features** (e.g., rare words): Need larger updates when they appear
- **Frequent features** (e.g., common words): Need smaller, fine-tuned updates

A single global learning rate cannot optimally handle both cases. AdaGrad addresses this by automatically scaling learning rates per-parameter.

## Mathematical Formulation

### Core Update Rule

AdaGrad accumulates the squared gradients for each parameter:

$$G_t = G_{t-1} + g_t \odot g_t$$

where:
- $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$ is the gradient at step $t$
- $\odot$ denotes element-wise multiplication
- $G_t \in \mathbb{R}^d$ accumulates squared gradients element-wise

The parameter update is:

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t$$

where:
- $\eta$ is the initial learning rate
- $\epsilon \approx 10^{-8}$ prevents division by zero
- The division and square root are element-wise

### Effective Learning Rate

The effective learning rate for parameter $\theta_i$ at step $t$ is:

$$\eta_{t,i}^{\text{eff}} = \frac{\eta}{\sqrt{G_{t,i}} + \epsilon} = \frac{\eta}{\sqrt{\sum_{s=1}^t g_{s,i}^2} + \epsilon}$$

This rate:
- **Decreases** for parameters with large accumulated gradients
- **Remains larger** for parameters with small accumulated gradients

### Matrix Formulation

The full outer product form (as in the original paper) uses:

$$G_t = \sum_{s=1}^t g_s g_s^\top$$

with update:

$$\theta_t = \theta_{t-1} - \eta G_t^{-1/2} g_t$$

The diagonal approximation (used in practice) keeps only the diagonal of $G_t$.

## Why AdaGrad Works for Sparse Data

Consider a word embedding scenario:

| Word | Frequency | Accumulated $G$ | Effective LR |
|------|-----------|-----------------|--------------|
| "the" | Very high | Large | Small (fine-tuning) |
| "serendipity" | Rare | Small | Large (significant updates) |

This automatic adaptation is especially valuable for:
- Natural Language Processing
- Recommender systems  
- Click-through rate prediction
- Any sparse feature setting

## Algorithm

```
Algorithm: AdaGrad
Input: Initial θ₀, learning rate η, small constant ε
Initialize: G₀ = 0

for t = 1, 2, ... do
    g_t ← ∇_θ L(θ_{t-1})                    # Compute gradient
    G_t ← G_{t-1} + g_t ⊙ g_t               # Accumulate squared gradients
    θ_t ← θ_{t-1} - η · g_t / (√G_t + ε)    # Update parameters
end for
```

## PyTorch Implementation

### Using Built-in Adagrad

```python
import torch
import torch.optim as optim

model = YourModel()
optimizer = optim.Adagrad(
    model.parameters(),
    lr=0.01,           # Initial learning rate
    lr_decay=0,        # Learning rate decay
    eps=1e-10,         # Epsilon for numerical stability
    initial_accumulator_value=0  # Initial value for G
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

class AdaGrad(Optimizer):
    """
    AdaGrad optimizer implementation.
    
    Adapts learning rate based on accumulated squared gradients:
    G_t = G_{t-1} + g_t²
    θ_t = θ_{t-1} - η * g_t / (√G_t + ε)
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        eps: Small constant for numerical stability (default: 1e-10)
        initial_accumulator_value: Starting value for G (default: 0)
    """
    
    def __init__(self, params, lr=0.01, eps=1e-10, initial_accumulator_value=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if initial_accumulator_value < 0.0:
            raise ValueError(f"Invalid initial_accumulator_value: {initial_accumulator_value}")
        
        defaults = dict(
            lr=lr,
            eps=eps,
            initial_accumulator_value=initial_accumulator_value
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
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize accumulated squared gradients
                if 'sum' not in state:
                    state['sum'] = torch.full_like(
                        p, 
                        group['initial_accumulator_value']
                    )
                
                sum_sq = state['sum']
                
                # Accumulate squared gradients
                sum_sq.addcmul_(grad, grad, value=1)
                
                # Compute adaptive learning rate and update
                std = sum_sq.sqrt().add_(eps)
                p.addcdiv_(grad, std, value=-lr)
        
        return loss
```

### NumPy Implementation

```python
import numpy as np

class AdaGrad:
    """
    AdaGrad optimizer implementation (NumPy).
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Initial learning rate (global step size)
    epsilon : float, default=1e-8
        Small constant for numerical stability
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.cache = {}  # Accumulated sum of squared gradients
    
    def update(self, params, grads):
        """
        Update parameters using AdaGrad algorithm.
        
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
            
            # Accumulate squared gradients
            self.cache[key] += grads[key] ** 2
            
            # Update parameters with adaptive learning rate
            updated_params[key] = (
                params[key] - 
                self.learning_rate * grads[key] / 
                (np.sqrt(self.cache[key]) + self.epsilon)
            )
        
        return updated_params
    
    def get_effective_lr(self, key):
        """Return the effective learning rate for a parameter."""
        if key not in self.cache:
            return self.learning_rate
        return self.learning_rate / (np.sqrt(self.cache[key]) + self.epsilon)
```

## Demonstrations

### Demo 1: Basic Optimization

```python
def demo_basic():
    """
    Demonstrate AdaGrad on a simple quadratic function.
    Minimize f(x, y) = x² + y²
    """
    print("=" * 60)
    print("AdaGrad Optimizer Demo")
    print("=" * 60)
    print("Minimizing f(x, y) = x² + y²")
    print()
    
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    optimizer = AdaGrad(learning_rate=1.0)  # AdaGrad can use higher initial LR
    
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

### Demo 2: Sparse Gradients

```python
def demo_sparse_gradients():
    """
    Demonstrate AdaGrad's advantage with sparse gradients.
    Simulates scenario where some parameters are updated infrequently.
    """
    print("=" * 60)
    print("AdaGrad with Sparse Gradients")
    print("=" * 60)
    print("Parameters x, y, z where z is rarely updated (sparse)")
    print()
    
    params = {
        'x': np.array([5.0]),  # Updated every iteration
        'y': np.array([5.0]),  # Updated every iteration
        'z': np.array([5.0])   # Updated only every 10th iteration
    }
    
    optimizer = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'x':<12} {'y':<12} {'z':<12} {'z eff_lr':<12}")
    print("-" * 66)
    
    for i in range(50):
        # Sparse gradient: z only updated every 10 iterations
        grads = {
            'x': 2 * params['x'],
            'y': 2 * params['y'],
            'z': 2 * params['z'] if i % 10 == 0 else np.array([0.0])
        }
        
        params = optimizer.update(params, grads)
        
        if i % 10 == 0:
            z_eff_lr = optimizer.get_effective_lr('z')
            print(f"{i:<12} {params['x'][0]:<12.6f} {params['y'][0]:<12.6f} "
                  f"{params['z'][0]:<12.6f} {z_eff_lr[0]:<12.6f}")
    
    print("\nObservation: z retains higher effective learning rate due to sparse updates!")
```

### Demo 3: Learning Rate Decay Visualization

```python
def demo_learning_rate_decay():
    """
    Visualize how AdaGrad's effective learning rate decreases over time.
    """
    print("=" * 60)
    print("AdaGrad Learning Rate Decay")
    print("=" * 60)
    print("Effective LR = η / √(sum of squared gradients)")
    print()
    
    param = np.array([10.0])
    optimizer = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'Param':<15} {'Gradient':<15} {'Effective LR':<15}")
    print("-" * 60)
    
    for i in range(50):
        grad = 2 * param  # Constant relative gradient
        
        # Calculate effective LR before update
        if 'param' in optimizer.cache:
            eff_lr = optimizer.learning_rate / (
                np.sqrt(optimizer.cache['param']) + optimizer.epsilon
            )
        else:
            eff_lr = np.array([optimizer.learning_rate])
        
        # Update
        result = optimizer.update({'param': param}, {'param': grad})
        param = result['param']
        
        if i % 10 == 0:
            print(f"{i:<12} {param[0]:<15.6f} {grad[0]:<15.6f} {eff_lr[0]:<15.8f}")
    
    print("\nObservation: Effective learning rate monotonically decreases!")
    print("This can cause AdaGrad to stop learning prematurely in some cases.")
```

## Theoretical Analysis

### Regret Bound

AdaGrad achieves the following regret bound for online convex optimization:

$$R(T) = \sum_{t=1}^T f_t(\theta_t) - \min_{\theta} \sum_{t=1}^T f_t(\theta) \leq O\left(\sqrt{T}\right)$$

More specifically:

$$R(T) \leq \frac{D^2}{2\eta} \sum_{i=1}^d \sqrt{\sum_{t=1}^T g_{t,i}^2}$$

where $D$ is the diameter of the constraint set.

### Data-Dependent Bound

The key insight is that the bound depends on $\sum_t g_{t,i}^2$ rather than $T$ directly. For sparse gradients where many $g_{t,i} = 0$, this can be much smaller than $O(\sqrt{dT})$.

### Comparison with SGD

| Property | SGD | AdaGrad |
|----------|-----|---------|
| Regret bound | $O(\sqrt{dT})$ | $O(\sqrt{\sum_i \sum_t g_{t,i}^2})$ |
| Sparse data | Poor | Excellent |
| Long training | Stable | May stall |

## Limitations and Solutions

### The Diminishing Learning Rate Problem

AdaGrad's main limitation: the accumulated squared gradients $G_t$ only increases, causing the effective learning rate to monotonically decrease.

**Consequence:** Learning can effectively stop in later stages of training.

**Mathematical view:**
$$\lim_{t \to \infty} \frac{\eta}{\sqrt{G_t}} = 0$$

### Solutions

1. **RMSprop:** Uses exponential moving average instead of sum
2. **Adadelta:** Removes need for initial learning rate
3. **Adam:** Combines momentum with adaptive rates

### When AdaGrad is Still Best

Despite limitations, AdaGrad excels when:
- Training data is sparse
- Training is not extremely long
- Features have highly varying frequencies
- Online learning scenarios

## Hyperparameter Guidelines

### Learning Rate

| Scenario | Recommended $\eta$ |
|----------|-------------------|
| Dense features | 0.01 - 0.1 |
| Sparse features | 0.1 - 1.0 |
| NLP embeddings | 0.5 - 1.0 |

Unlike other optimizers, AdaGrad often benefits from higher initial learning rates since the adaptive mechanism will reduce them.

### Epsilon

The default $\epsilon = 10^{-8}$ works for most cases. Increase to $10^{-7}$ or $10^{-6}$ if experiencing numerical instability.

### Initial Accumulator Value

PyTorch allows initializing $G_0 \neq 0$:
- Small positive value can stabilize early training
- Typically leave at 0

## Comparison: AdaGrad vs SGD

```python
def compare_adagrad_sgd():
    """Compare AdaGrad vs SGD on ill-conditioned problem."""
    print("=" * 70)
    print("AdaGrad vs SGD on Ill-Conditioned Problem")
    print("=" * 70)
    print("f(x, y) = 100x² + y² (condition number = 100)")
    print()
    
    # Initialize
    params_sgd = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_ada = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    lr_sgd = 0.001  # Must be small for SGD stability
    optimizer_ada = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iter':<8} {'SGD f(x,y)':<18} {'AdaGrad f(x,y)':<18}")
    print("-" * 50)
    
    for i in range(100):
        # Gradients
        grad_sgd = {'x': 200 * params_sgd['x'], 'y': 2 * params_sgd['y']}
        grad_ada = {'x': 200 * params_ada['x'], 'y': 2 * params_ada['y']}
        
        # SGD update
        params_sgd['x'] = params_sgd['x'] - lr_sgd * grad_sgd['x']
        params_sgd['y'] = params_sgd['y'] - lr_sgd * grad_sgd['y']
        
        # AdaGrad update
        params_ada = optimizer_ada.update(params_ada, grad_ada)
        
        # Compute losses
        f_sgd = 100 * params_sgd['x']**2 + params_sgd['y']**2
        f_ada = 100 * params_ada['x']**2 + params_ada['y']**2
        
        if i % 20 == 0:
            print(f"{i:<8} {f_sgd[0]:<18.6f} {f_ada[0]:<18.6f}")
    
    print("\nAdaGrad handles different gradient scales automatically!")
```

## Summary

| Aspect | Detail |
|--------|--------|
| Key Innovation | Per-parameter adaptive learning rate based on gradient history |
| Update Rule | $G_t = G_{t-1} + g_t^2$; $\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t}+\epsilon} g_t$ |
| Hyperparameters | $\eta$ (initial LR, typically 0.01-1.0), $\epsilon$ (stability, $10^{-8}$) |
| Benefits | Excellent for sparse data, automatic adaptation |
| Drawbacks | Learning rate can decrease to zero |
| Best For | NLP, recommender systems, sparse features, online learning |

## References

1. Duchi, J., Hazan, E., & Singer, Y. (2011). "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
2. McMahan, H. B., & Streeter, M. (2010). "Adaptive Bound Optimization for Online Convex Optimization"
3. Dean, J., et al. (2012). "Large Scale Distributed Deep Networks"
