# Adjoint Sensitivity Method

## Learning Objectives

By the end of this section, you will:

- Understand why standard backpropagation is memory-inefficient for Neural ODEs
- Derive the adjoint sensitivity equations from first principles
- Implement the adjoint method for memory-efficient gradient computation
- Analyze the trade-offs between standard backprop and adjoint methods
- Use `odeint_adjoint` in practice

## Prerequisites

- ODE fundamentals and numerical integration
- Neural ODE architecture and forward pass
- Chain rule and backpropagation
- Basic calculus of variations (helpful but not required)

---

## 1. The Memory Problem

### 1.1 Standard Backpropagation Through ODE Solvers

Consider a Neural ODE that computes:

$$h(T) = h(0) + \int_0^T f_\theta(h(t), t) \, dt$$

With a loss function $L(h(T))$, we need gradients $\frac{\partial L}{\partial \theta}$ to train.

**Standard approach:** Treat the ODE solver as a sequence of differentiable operations:

$$h_0 \xrightarrow{\text{step 1}} h_1 \xrightarrow{\text{step 2}} h_2 \xrightarrow{\text{step 3}} \cdots \xrightarrow{\text{step N}} h_N$$

Backpropagation requires storing all intermediate states $\{h_1, h_2, \ldots, h_N\}$ for the backward pass.

**Memory cost:** $O(N)$ where $N$ is the number of solver steps. For adaptive solvers, $N$ can be very large (hundreds to thousands of steps).

### 1.2 Why This Matters

| Scenario | Approximate NFE | Memory (hidden dim 256) |
|----------|-----------------|-------------------------|
| Short integration | 20-50 | ~50 MB |
| Long integration | 100-500 | ~250 MB |
| High accuracy | 500-2000 | ~1 GB |
| Stiff dynamics | 1000-10000 | ~5 GB |

For complex dynamics or high-dimensional states, memory becomes the bottleneck.

> **Deep Insight:** This is fundamentally different from standard neural networks where depth is fixed. With Neural ODEs, the "effective depth" (number of solver steps) is determined at runtime by the dynamics complexity.

---

## 2. The Adjoint Sensitivity Method

### 2.1 Core Idea

Instead of storing the forward trajectory, we:

1. **Forward pass:** Solve ODE, store only the final state $h(T)$
2. **Backward pass:** Solve another ODE *backwards in time* to compute gradients
3. **Reconstruct states on-the-fly** during the backward ODE solve

This achieves **$O(1)$ memory** regardless of the number of solver steps!

### 2.2 Mathematical Setup

We want to compute:

$$\frac{\partial L}{\partial \theta} \quad \text{where} \quad L = L(h(T))$$

and $h(t)$ satisfies:

$$\frac{dh}{dt} = f_\theta(h(t), t), \quad h(0) = h_0$$

### 2.3 The Adjoint State

Define the **adjoint state** $a(t)$ as:

$$a(t) = \frac{\partial L}{\partial h(t)}$$

This represents how the loss changes with respect to the hidden state at time $t$.

**Boundary condition:** At the final time,

$$a(T) = \frac{\partial L}{\partial h(T)}$$

This is just the gradient of the loss with respect to the Neural ODE output.

### 2.4 Deriving the Adjoint ODE

**Theorem:** The adjoint state satisfies the ODE:

$$\frac{da}{dt} = -a(t)^\top \frac{\partial f}{\partial h}$$

**Proof sketch:**

Consider how the loss depends on $h(t)$ for $t < T$. A small change $\delta h(t)$ propagates to $h(T)$ via the ODE. Using the chain rule and Leibniz's rule:

$$\frac{da}{dt} = \frac{d}{dt}\frac{\partial L}{\partial h(t)} = -a(t)^\top \frac{\partial f}{\partial h}(h(t), t)$$

The negative sign arises because we're going backwards in time.

### 2.5 Parameter Gradients

The parameter gradients are:

$$\frac{\partial L}{\partial \theta} = \int_0^T a(t)^\top \frac{\partial f}{\partial \theta} \, dt$$

We accumulate parameter gradients *during* the backward ODE solve.

---

## 3. The Complete Algorithm

### 3.1 Augmented State Formulation

For efficient implementation, we solve a single augmented ODE backwards. Define:

$$s(t) = \begin{bmatrix} h(t) \\ a(t) \\ \frac{\partial L}{\partial \theta}(t) \end{bmatrix}$$

The augmented dynamics are:

$$\frac{ds}{dt} = \begin{bmatrix} f_\theta(h, t) \\ -a^\top \frac{\partial f}{\partial h} \\ -a^\top \frac{\partial f}{\partial \theta} \end{bmatrix}$$

### 3.2 Memory Analysis

| Component | Memory |
|-----------|--------|
| Final state $h(T)$ | $O(d)$ |
| Adjoint state $a(t)$ | $O(d)$ |
| Gradient accumulator | $O(\|\theta\|)$ |
| **Total** | **$O(d + \|\theta\|)$** |

This is independent of the number of solver steps—**constant memory**!

---

## 4. PyTorch Implementation

### 4.1 Using `odeint_adjoint`

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

class ODEFunc(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        self.nfe = 0
    
    def forward(self, t, h):
        self.nfe += 1
        return self.net(h)


def compare_backprop_methods():
    """Compare standard backprop vs adjoint method."""
    
    dim = 64
    batch_size = 32
    
    func_standard = ODEFunc(dim)
    func_adjoint = ODEFunc(dim)
    func_adjoint.load_state_dict(func_standard.state_dict())
    
    h0 = torch.randn(batch_size, dim, requires_grad=True)
    t = torch.linspace(0, 1, 50)
    
    # Standard backprop
    func_standard.nfe = 0
    h_standard = odeint(func_standard, h0, t, method='dopri5')
    loss_standard = h_standard[-1].sum()
    loss_standard.backward()
    grad_standard = h0.grad.clone()
    nfe_forward = func_standard.nfe
    
    print(f"Standard backprop: {nfe_forward} forward NFE")
    
    # Adjoint method
    h0_adj = h0.detach().clone().requires_grad_()
    func_adjoint.nfe = 0
    h_adjoint = odeint_adjoint(func_adjoint, h0_adj, t, method='dopri5')
    loss_adjoint = h_adjoint[-1].sum()
    
    func_adjoint.nfe = 0
    loss_adjoint.backward()
    grad_adjoint = h0_adj.grad.clone()
    nfe_backward = func_adjoint.nfe
    
    print(f"Adjoint method: {nfe_backward} backward NFE")
    
    # Verify gradients match
    grad_diff = (grad_standard - grad_adjoint).abs().max().item()
    print(f"Max gradient difference: {grad_diff:.2e}")

compare_backprop_methods()
```

### 4.2 Using Adjoint in Models

```python
class NeuralODEWithAdjoint(nn.Module):
    """Neural ODE with memory-efficient adjoint training."""
    
    def __init__(self, dim, hidden_dim=64, use_adjoint=True):
        super().__init__()
        self.func = ODEFunc(dim, hidden_dim)
        self.use_adjoint = use_adjoint
        self.register_buffer('t', torch.tensor([0., 1.]))
    
    def forward(self, h0):
        if self.use_adjoint:
            h = odeint_adjoint(self.func, h0, self.t, method='dopri5')
        else:
            h = odeint(self.func, h0, self.t, method='dopri5')
        
        return h[-1]
```

---

## 5. Trade-offs and Considerations

### 5.1 Adjoint Method Drawbacks

| Aspect | Standard Backprop | Adjoint Method |
|--------|-------------------|----------------|
| Memory | $O(N)$ | $O(1)$ |
| Forward computation | $N$ evals | $N$ evals |
| Backward computation | (stored) | $\sim N$ evals |
| Total computation | $N$ | $\sim 2N$ |
| Gradient accuracy | Exact | Numerical |

### 5.2 When to Use Each Method

| Situation | Recommendation |
|-----------|----------------|
| Memory sufficient, speed critical | Standard backprop |
| Memory limited | Adjoint |
| Long trajectories | Adjoint |
| Research/debugging | Standard |
| Production/deployment | Adjoint |

### 5.3 Improving Adjoint Stability

```python
# Tighter tolerances for adjoint solve
h = odeint_adjoint(
    func, h0, t,
    method='dopri5',
    rtol=1e-6,
    atol=1e-8,
    adjoint_rtol=1e-6,
    adjoint_atol=1e-8
)
```

---

## 6. Key Takeaways

1. **Standard backprop through ODE solvers requires O(N) memory** to store all intermediate states.

2. **The adjoint method achieves O(1) memory** by solving another ODE backwards to compute gradients.

3. **The adjoint state** $a(t) = \frac{\partial L}{\partial h(t)}$ satisfies $\frac{da}{dt} = -a^\top \frac{\partial f}{\partial h}$.

4. **Trade-offs exist**: Adjoint doubles computation but eliminates memory scaling with depth.

5. **Use `odeint_adjoint`** for memory-constrained settings or long integrations.

---

## 7. Exercises

### Exercise 1: Derive Adjoint for Initial Time

Extend the adjoint derivation to compute $\frac{\partial L}{\partial t_0}$.

### Exercise 2: Memory Profiling

Create detailed memory profiles comparing standard and adjoint methods across different batch sizes, hidden dimensions, and integration tolerances.

### Exercise 3: Gradient Verification

Implement finite difference gradient checking to verify adjoint gradient accuracy for various dynamics functions.

---

## References

1. Chen, R. T. Q., et al. (2018). Neural Ordinary Differential Equations. *NeurIPS*. (Appendix B)

2. Pontryagin, L. S. (1962). The Mathematical Theory of Optimal Processes.

3. Griewank, A., & Walther, A. (2008). Evaluating Derivatives. SIAM.
