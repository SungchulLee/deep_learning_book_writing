# Adam (Adaptive Moment Estimation)

## Overview

Adam (Adaptive Moment Estimation) was introduced by Kingma and Ba in 2014. It combines the benefits of two other extensions of SGD: AdaGrad's ability to handle sparse gradients and RMSprop's ability to deal with non-stationary objectives. Adam computes adaptive learning rates for each parameter using estimates of first and second moments of the gradients.

## Motivation

Adam addresses limitations of previous optimizers:

| Optimizer | Limitation | Adam's Solution |
|-----------|------------|-----------------|
| SGD | Uniform learning rate | Per-parameter adaptation |
| Momentum | No adaptation to gradient scale | Second moment scaling |
| AdaGrad | Diminishing learning rate | Exponential moving average |
| RMSprop | No momentum | First moment (momentum) |

Adam combines momentum (first moment) with RMSprop-style adaptation (second moment), plus bias correction for early training stability.

## Mathematical Formulation

### First Moment (Mean) Estimate

The exponential moving average of gradients:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

This serves as momentum, smoothing the gradient direction.

### Second Moment (Uncentered Variance) Estimate

The exponential moving average of squared gradients:

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

This adapts the learning rate per-parameter (like RMSprop).

### Bias Correction

The moment estimates are biased toward zero, especially in early iterations when $m_0 = v_0 = 0$:

$$\mathbb{E}[m_t] = \mathbb{E}[g_t] \cdot (1 - \beta_1^t) + \xi$$

Bias-corrected estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### Parameter Update

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

### Understanding Bias Correction

At step $t$, with initialization $m_0 = 0$:

$$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$

Taking expectation (assuming stationary $g$):

$$\mathbb{E}[m_t] = \mathbb{E}[g] \cdot (1 - \beta_1^t)$$

The factor $(1 - \beta_1^t)$ causes underestimation, corrected by dividing by it. The bias correction terms compensate for the zero-initialization of $m$ and $v$, which would otherwise cause severe underestimation in early iterations.

## Algorithm

```
Algorithm: Adam
Input: Initial θ₀, learning rate η=0.001, β₁=0.9, β₂=0.999, ε=10⁻⁸
Initialize: m₀ = 0, v₀ = 0, t = 0

for t = 1, 2, ... do
    t ← t + 1
    g_t ← ∇_θ L(θ_{t-1})                           # Compute gradient
    m_t ← β₁ · m_{t-1} + (1 - β₁) · g_t            # Update first moment
    v_t ← β₂ · v_{t-1} + (1 - β₂) · g_t²           # Update second moment
    m̂_t ← m_t / (1 - β₁ᵗ)                          # Bias correction
    v̂_t ← v_t / (1 - β₂ᵗ)                          # Bias correction
    θ_t ← θ_{t-1} - η · m̂_t / (√v̂_t + ε)          # Update parameters
end for
```

## Default Hyperparameters

The original paper recommends:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| $\eta$ (lr) | 0.001 | 0.0001 - 0.01 | Learning rate |
| $\beta_1$ | 0.9 | 0.8 - 0.99 | First moment decay |
| $\beta_2$ | 0.999 | 0.99 - 0.9999 | Second moment decay |
| $\epsilon$ | 1e-8 | 1e-8 - 1e-6 | Numerical stability |

These defaults work well across most problems and rarely need adjustment.

## PyTorch Implementation

### Using Built-in Adam

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

# Adam optimizer with defaults
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Explicit parameters
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,    # L2 regularization (not recommended, use AdamW)
    amsgrad=False      # Whether to use AMSGrad variant
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

class Adam(Optimizer):
    """
    Adam optimizer implementation.
    
    Combines momentum (first moment) with adaptive learning rates (second moment)
    and includes bias correction for both moments.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Tuple of (β₁, β₂) coefficients (default: (0.9, 0.999))
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
                
                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)      # m (first moment)
                    state['exp_avg_sq'] = torch.zeros_like(p)   # v (second moment)
                
                m, v = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                
                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                
                # Compute bias-corrected estimates
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                
                # Update parameters
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        
        return loss
```

### NumPy Implementation

```python
import numpy as np

class Adam:
    """
    Adam optimizer implementation (NumPy).
    
    Parameters:
    -----------
    learning_rate : float, default=0.001
        Step size for parameter updates
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates
    epsilon : float, default=1e-8
        Small constant for numerical stability
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # State variables
        self.m = {}  # First moment (mean of gradients)
        self.v = {}  # Second moment (variance of gradients)
        self.t = 0   # Time step
    
    def update(self, params, grads):
        """
        Update parameters using Adam algorithm.
        
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
        self.t += 1
        updated_params = {}
        
        for key in params.keys():
            # Initialize moment vectors if not exists
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_params[key] = (
                params[key] - 
                self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            )
        
        return updated_params
```

## Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# Data
X = torch.randn(200, 10)
y = torch.sin(X.sum(dim=1, keepdim=True)) + torch.randn(200, 1) * 0.1

# Model
model = nn.Sequential(
    nn.Linear(10, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1)
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(500):
    pred = model(X)
    loss = criterion(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

## Adaptive Learning Rate Intuition

Adam adapts the learning rate per parameter:

- Parameters with large gradients get smaller effective learning rates
- Parameters with small gradients get larger effective learning rates

This automatic scaling helps:
- Sparse features learn appropriately
- Different network layers train at suitable rates
- Noisy gradients are dampened

## Demonstrations

### Demo 1: Basic Optimization

```python
import numpy as np

def demo_adam():
    """
    Demonstrate Adam on a simple quadratic function.
    Minimize f(x, y) = x² + y²
    """
    print("=" * 60)
    print("Adam Optimizer Demo")
    print("=" * 60)
    print("Minimizing f(x, y) = x² + y²")
    print()
    
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    optimizer = Adam(learning_rate=0.1)
    
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

### Demo 2: Bias Correction Visualization

```python
def demo_bias_correction():
    """
    Visualize the effect of bias correction in Adam.
    """
    print("=" * 60)
    print("Adam Bias Correction Effect")
    print("=" * 60)
    
    beta1, beta2 = 0.9, 0.999
    
    print(f"{'Step t':<10} {'1-β₁ᵗ':<15} {'1-β₂ᵗ':<15} {'m correction':<15} {'v correction':<15}")
    print("-" * 70)
    
    for t in [1, 2, 5, 10, 20, 50, 100, 500, 1000]:
        correction1 = 1 - beta1 ** t
        correction2 = 1 - beta2 ** t
        m_factor = 1 / correction1
        v_factor = 1 / correction2
        
        print(f"{t:<10} {correction1:<15.6f} {correction2:<15.6f} {m_factor:<15.4f} {v_factor:<15.4f}")
    
    print()
    print("Observation: Correction factors are large early (t small) and approach 1 later.")
    print("This compensates for the bias from initializing m₀ = v₀ = 0.")
```

### Demo 3: Optimizer Comparison on Rosenbrock Function

```python
def demo_comprehensive_comparison():
    """
    Compare Adam with SGD, Momentum, and RMSprop on Rosenbrock function.
    """
    print("=" * 80)
    print("Optimizer Comparison on Rosenbrock Function")
    print("=" * 80)
    print("f(x, y) = (1-x)² + 100(y-x²)²")
    print("Minimum at (1, 1)")
    print()
    
    def rosenbrock_grad(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return dx, dy
    
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    # Initialize
    start = (-1.0, -1.0)
    results = {
        'SGD': {'x': start[0], 'y': start[1], 'losses': []},
        'Momentum': {'x': start[0], 'y': start[1], 'vx': 0, 'vy': 0, 'losses': []},
        'Adam': {'x': start[0], 'y': start[1], 'losses': []}
    }
    
    adam = Adam(learning_rate=0.01)
    params_adam = {'x': np.array([start[0]]), 'y': np.array([start[1]])}
    
    lr_sgd = 0.0001
    lr_mom = 0.0001
    gamma = 0.9
    
    print(f"{'Iter':<8} {'SGD':<18} {'Momentum':<18} {'Adam':<18}")
    print("-" * 65)
    
    for i in range(1000):
        # SGD
        dx, dy = rosenbrock_grad(results['SGD']['x'], results['SGD']['y'])
        results['SGD']['x'] -= lr_sgd * dx
        results['SGD']['y'] -= lr_sgd * dy
        results['SGD']['losses'].append(rosenbrock(results['SGD']['x'], results['SGD']['y']))
        
        # Momentum
        dx, dy = rosenbrock_grad(results['Momentum']['x'], results['Momentum']['y'])
        results['Momentum']['vx'] = gamma * results['Momentum']['vx'] + lr_mom * dx
        results['Momentum']['vy'] = gamma * results['Momentum']['vy'] + lr_mom * dy
        results['Momentum']['x'] -= results['Momentum']['vx']
        results['Momentum']['y'] -= results['Momentum']['vy']
        results['Momentum']['losses'].append(
            rosenbrock(results['Momentum']['x'], results['Momentum']['y'])
        )
        
        # Adam
        grads = {
            'x': rosenbrock_grad(params_adam['x'][0], params_adam['y'][0])[0] * np.ones(1),
            'y': rosenbrock_grad(params_adam['x'][0], params_adam['y'][0])[1] * np.ones(1)
        }
        params_adam = adam.update(params_adam, grads)
        results['Adam']['losses'].append(
            rosenbrock(params_adam['x'][0], params_adam['y'][0])
        )
        
        if i % 200 == 0:
            print(f"{i:<8} {results['SGD']['losses'][-1]:<18.6f} "
                  f"{results['Momentum']['losses'][-1]:<18.6f} "
                  f"{results['Adam']['losses'][-1]:<18.6f}")
    
    print()
    print("Final positions:")
    print(f"  SGD:      ({results['SGD']['x']:.4f}, {results['SGD']['y']:.4f})")
    print(f"  Momentum: ({results['Momentum']['x']:.4f}, {results['Momentum']['y']:.4f})")
    print(f"  Adam:     ({params_adam['x'][0]:.4f}, {params_adam['y'][0]:.4f})")
    print(f"  Target:   (1.0, 1.0)")
```

## Comparing Adam vs SGD

```python
def compare_optimizers(X, y, epochs=500):
    results = {}
    
    for name, opt_fn in [
        ('SGD', lambda p: optim.SGD(p, lr=0.01)),
        ('SGD+Momentum', lambda p: optim.SGD(p, lr=0.01, momentum=0.9)),
        ('Adam', lambda p: optim.Adam(p, lr=0.001)),
    ]:
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        optimizer = opt_fn(model.parameters())
        criterion = nn.MSELoss()
        
        losses = []
        for _ in range(epochs):
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        results[name] = losses
        print(f"{name}: Final loss = {losses[-1]:.4f}")
    
    return results

results = compare_optimizers(X, y)
```

Adam typically converges faster in the early stages, especially without careful learning rate tuning.

## Theoretical Properties

### Convergence Analysis

For convex functions with bounded gradients $\|g_t\|_\infty \leq G$:

$$\text{Regret}(T) = \sum_{t=1}^T [f(\theta_t) - f(\theta^*)] = O\left(\sqrt{T}\right)$$

More precisely, Adam achieves:

$$\text{Regret}(T) \leq \frac{D^2}{2\eta(1-\beta_1)} \sum_{i=1}^d \sqrt{T \cdot \hat{v}_{T,i}} + O(1)$$

### Step Size Bounds

The effective step size for parameter $i$ at step $t$:

$$\Delta \theta_{t,i} = \frac{\eta \cdot \hat{m}_{t,i}}{\sqrt{\hat{v}_{t,i}} + \epsilon}$$

This is bounded:

$$|\Delta \theta_{t,i}| \leq \frac{\eta}{\sqrt{1-\beta_2}}$$

### Connection to Natural Gradient

Adam approximately preconditions the gradient with the inverse square root of the diagonal Fisher information matrix, connecting it to natural gradient methods. This provides a principled geometric interpretation of the adaptive scaling.

## When Adam Excels

**Best use cases:**
- Rapid prototyping (works out-of-box)
- Natural language processing (transformers, RNNs)
- Generative models (GANs, VAEs)
- When limited time for hyperparameter tuning
- Sparse gradients (embeddings, attention)

**Characteristics:**
- Fast initial convergence
- Robust to hyperparameter choices
- Higher memory usage (stores m and v per parameter)
- May generalize slightly worse than well-tuned SGD

## Limitations and Known Issues

### Generalization Gap

Research has shown Adam can converge to sharper minima than SGD, potentially leading to worse generalization on some tasks (especially image classification). Reddi et al. (2018) demonstrated cases where Adam may not converge to optimal solutions.

**Solutions:**
- Switch to SGD for final epochs
- Use AdamW with proper weight decay
- Learning rate warmup and decay

### Weight Decay Problem

Adam's original weight decay implementation is incorrect. Standard L2 regularization in Adam is suboptimal:

```python
# Problematic: L2 regularization coupled with adaptive rates
grad = grad + weight_decay * param
```

The adaptive learning rate scales down the regularization effect inconsistently.

**Solution:** Use **AdamW** (decoupled weight decay) instead.

### Memory Overhead

Adam stores two additional buffers (m and v) per parameter:
- Memory = 3× parameter count (vs 1× for vanilla SGD)
- For a 100M parameter model: ~1.2GB additional memory

## Hyperparameter Tuning Guidelines

**Learning Rate ($\eta$):**
- Start with 0.001 (rarely needs changing)
- Decrease for fine-tuning (0.0001)
- Can use learning rate schedules

**$\beta_1$ (momentum):**
- 0.9 works well for most cases
- Increase to 0.95 for noisier gradients
- Decrease to 0.8 for faster adaptation

**$\beta_2$ (RMSprop-like):**
- 0.999 is robust default
- Decrease to 0.99 for sparse gradients
- Rarely needs changing

**$\epsilon$:**
- Increase if seeing numerical instability
- 1e-7 or 1e-6 for very small gradients

### Task-Specific Learning Rates

| Task | Suggested LR |
|------|--------------|
| General | 0.001 |
| Fine-tuning | 0.0001 |
| Transformers | 0.0001 with warmup |
| Very deep networks | 0.0001 - 0.001 |

## Practical Recommendations

**Starting configuration:**
```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,  # Good default
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**For better generalization, consider:**
- Switching to AdamW with weight decay
- Using learning rate warmup
- Decaying to a lower learning rate later in training
- Switching to SGD+Momentum for final fine-tuning

## Variants

### AMSGrad

Maintains maximum of past $v_t$ values to fix convergence issues:

```python
v_hat_max = max(v_hat_max, v_hat)
param.addcdiv_(m_hat, v_hat_max.sqrt().add_(eps), value=-lr)
```

### AdamW

Decouples weight decay from gradient updates (see separate documentation).

### NAdam

Incorporates Nesterov momentum into Adam.

### RAdam

Rectified Adam with adaptive learning rate warmup (see separate documentation).

## Summary

| Aspect | Detail |
|--------|--------|
| Key Innovation | Combines momentum (first moment) with adaptive rates (second moment) plus bias correction |
| Update Rule | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$; $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$; $\theta_t = \theta_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ |
| Hyperparameters | $\eta=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ |
| Benefits | Works well out-of-the-box, handles sparse gradients, fast convergence |
| Drawbacks | May generalize worse than SGD, weight decay issues, 3× memory |
| Best For | Default choice for most deep learning tasks, NLP, generative models |

## References

1. Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization"
2. Reddi, S. J., Kale, S., & Kumar, S. (2018). "On the Convergence of Adam and Beyond"
3. Loshchilov, I., & Hutter, F. (2017). "Decoupled Weight Decay Regularization"
