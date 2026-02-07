# Universal Approximation Theorem

## Learning Objectives

!!! abstract "What You Will Learn"
    - State the Universal Approximation Theorem precisely in its classical, generalized, and ReLU forms
    - Explain geometrically why sigmoid and ReLU networks can approximate arbitrary continuous functions
    - Distinguish between existence (the theorem) and findability (training) of good solutions
    - Analyze the width complexity of shallow vs. deep approximation and the curse of dimensionality
    - Demonstrate universal approximation empirically with PyTorch experiments

## Prerequisites

| Topic | Why It Matters |
|-------|---------------|
| MLP Architecture (§4.2.1) | The theorem applies to single-hidden-layer networks |
| Continuity and compactness | Theorem hypotheses involve continuous functions on compact domains |
| Fourier analysis (optional) | Barron's theorem uses Fourier moments |

---

## Overview

The **Universal Approximation Theorem** is one of the most important theoretical results in neural network theory. It establishes that feedforward neural networks with a single hidden layer can approximate any continuous function to arbitrary accuracy, given sufficient width. This provides the theoretical foundation for using neural networks as flexible function approximators, though it says nothing about how to find the right parameters.

---

## Theorem Statements

### Classical Form (Cybenko, 1989)

!!! abstract "Theorem (Universal Approximation — Width Version)"
    Let $\sigma: \mathbb{R} \to \mathbb{R}$ be a continuous **sigmoidal** function, i.e.,
    
    $$
    \sigma(t) \to \begin{cases} 1 & t \to +\infty \\ 0 & t \to -\infty \end{cases}
    $$
    
    Let $I_n = [0,1]^n$ denote the $n$-dimensional unit hypercube and $C(I_n)$ the space of continuous functions on $I_n$ with the supremum norm.
    
    Then for any $f \in C(I_n)$ and any $\varepsilon > 0$, there exist $N \in \mathbb{N}$, real constants $v_i, b_i \in \mathbb{R}$, and vectors $\mathbf{w}_i \in \mathbb{R}^n$ for $i = 1, \ldots, N$ such that the function
    
    $$
    F(\mathbf{x}) = \sum_{i=1}^{N} v_i \, \sigma\!\left(\mathbf{w}_i^\top \mathbf{x} + b_i\right)
    $$
    
    satisfies $\|F - f\|_\infty = \sup_{\mathbf{x} \in I_n} |F(\mathbf{x}) - f(\mathbf{x})| < \varepsilon$.

In neural network language: a single-hidden-layer network with $N$ sigmoid neurons, using output weights $v_i$ (no output activation), can uniformly approximate any continuous function on a compact domain.

### Generalized Form (Hornik, 1991)

Hornik extended Cybenko's result significantly:

1. **Any non-polynomial activation** works (not just sigmoid) — including tanh, softplus, etc.
2. The result holds for **any compact subset** $K \subset \mathbb{R}^n$, not just $[0,1]^n$
3. Both the function and its **derivatives** can be approximated
4. Approximation holds in $L^p$ norms, not just the supremum norm

!!! abstract "Theorem (Hornik, 1991)"
    Let $\sigma$ be any continuous, non-polynomial function. Then the class of single-hidden-layer feedforward networks is **dense** in $C(K)$ for any compact $K \subset \mathbb{R}^n$.

The non-polynomial condition is necessary: a polynomial activation of degree $d$ can only represent polynomials up to degree $d \cdot L$ with $L$ layers, which is not dense in $C(K)$.

### ReLU Form (Modern)

!!! abstract "Theorem (Universal Approximation with ReLU)"
    Let $\sigma(x) = \max(0, x)$ be the ReLU activation. For any continuous function $f: K \to \mathbb{R}$ on a compact domain $K \subset \mathbb{R}^n$ and any $\varepsilon > 0$, there exists a single-hidden-layer ReLU network
    
    $$
    F(\mathbf{x}) = \sum_{i=1}^{N} v_i \max\!\left(0,\; \mathbf{w}_i^\top \mathbf{x} + b_i\right)
    $$
    
    such that $\sup_{\mathbf{x} \in K} |F(\mathbf{x}) - f(\mathbf{x})| < \varepsilon$.

Despite ReLU being neither bounded nor smooth, the result holds because ReLU networks produce **piecewise linear** functions, and any continuous function on a compact domain can be uniformly approximated by piecewise linear functions.

---

## Geometric Intuition

### Sigmoid Networks: Sums of Soft Steps

Each hidden neuron $\sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)$ produces a **soft step function** — a smooth transition from 0 to 1 along the direction $\mathbf{w}_i$:

- The vector $\mathbf{w}_i$ determines the **orientation** of the step
- The bias $b_i$ determines the **position** (offset along $\mathbf{w}_i$)
- The magnitude $\|\mathbf{w}_i\|$ controls the **steepness** (larger $\|\mathbf{w}_i\| \Rightarrow$ sharper step)

A **bump function** can be constructed from two opposing steps:

$$
\text{bump}(x) \approx v \cdot \sigma(w x + b_1) - v \cdot \sigma(w x + b_2), \quad b_1 > b_2
$$

By combining many such bumps with different positions, heights ($v_i$), and widths, we can reconstruct any continuous function — analogous to how step functions approximate integrals.

### ReLU Networks: Piecewise Linear Approximation

Each ReLU neuron creates a "hinge" — a piecewise linear function with a single breakpoint (knot):

$$
\max(0, w x + b) = \begin{cases} 0 & \text{if } wx + b \leq 0 \\ wx + b & \text{if } wx + b > 0 \end{cases}
$$

Key observations:

- $N$ ReLU neurons in 1D create a function with up to $N+1$ linear regions
- In $n$ dimensions, $N$ neurons create up to $O(N^n)$ linear regions (hyperplane arrangement)
- Any continuous function on a compact domain can be uniformly approximated by piecewise linear functions (a standard result in real analysis)

The ReLU network is therefore a **trainable piecewise linear approximator** whose breakpoints, slopes, and offsets are all learned from data.

---

## Proof Sketch

### Stone-Weierstrass Approach (Cybenko's Proof)

The key idea is to show that the set of single-hidden-layer networks is **dense** in $C(I_n)$ using a functional analysis argument.

**Step 1.** Define the function class:

$$
\mathcal{S} = \left\{ \sum_{i=1}^{N} v_i \, \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i) \;\Big|\; N \in \mathbb{N},\, v_i, b_i \in \mathbb{R},\, \mathbf{w}_i \in \mathbb{R}^n \right\}
$$

**Step 2.** Assume for contradiction that $\overline{\mathcal{S}} \neq C(I_n)$, so there exists $f \in C(I_n) \setminus \overline{\mathcal{S}}$.

**Step 3.** By the Hahn-Banach theorem, there exists a bounded linear functional $\mu \in C(I_n)^*$ such that:

$$
\int_{I_n} g(\mathbf{x}) \, d\mu(\mathbf{x}) = 0 \quad \forall\, g \in \mathcal{S}, \qquad \text{but} \qquad \int_{I_n} f(\mathbf{x}) \, d\mu(\mathbf{x}) \neq 0
$$

By the Riesz representation theorem, $\mu$ corresponds to a signed measure.

**Step 4.** From $\int \sigma(\mathbf{w}^\top \mathbf{x} + b) \, d\mu = 0$ for all $\mathbf{w}, b$, exploit the sigmoidal property to show $\mu = 0$, yielding a contradiction.

The critical step uses the fact that $\sigma(t) \to \mathbf{1}_{t > 0}$ as the weights scale, so the integrals against $\mu$ must vanish on all half-spaces, forcing $\mu = 0$.

### Constructive Approach for ReLU

For ReLU networks in 1D, the proof is constructive:

**Step 1.** For any continuous $f: [a,b] \to \mathbb{R}$ and $\varepsilon > 0$, choose $N$ large enough and partition $[a,b]$ into $N$ equal subintervals.

**Step 2.** On each subinterval $[x_i, x_{i+1}]$, the linear interpolant of $f$ satisfies $|f(x) - L_i(x)| < \varepsilon$ by uniform continuity.

**Step 3.** The piecewise linear interpolant can be written exactly using ReLU:

$$
F(x) = f(x_0) + \sum_{i=0}^{N-1} (s_{i+1} - s_i) \max(0, x - x_i)
$$

where $s_i = \frac{f(x_{i+1}) - f(x_i)}{x_{i+1} - x_i}$ are the slopes. This is a single-hidden-layer ReLU network with $N$ neurons.

---

## Width Complexity and the Curse of Dimensionality

### Shallow Networks

To approximate a function $f$ with Lipschitz constant $L$ on $[0,1]^n$ to accuracy $\varepsilon$ using piecewise linear (or piecewise constant) regions:

$$
N_{\text{regions}} = O\!\left(\left(\frac{L}{\varepsilon}\right)^n\right)
$$

Each region requires $O(1)$ neurons, so the required width is **exponential** in input dimension $n$. This is the **curse of dimensionality** for shallow networks.

### Deep Networks (Exponential Separation)

Deep networks can be **exponentially more efficient** than shallow ones for certain function classes.

!!! abstract "Depth Efficiency (Telgarsky, 2016)"
    There exist functions computable by depth-$O(k)$ ReLU networks with $O(1)$ width that require width $\Omega(2^{k/3})$ to approximate with depth-2 networks.

**Canonical example:** $f(x_1, \ldots, x_n) = x_1 \cdot x_2 \cdots x_n$

| Network | Width | Depth | Parameters |
|---------|-------|-------|------------|
| Shallow (1 hidden layer) | $\Omega(2^n)$ | 2 | Exponential |
| Deep (tree structure) | $O(n)$ | $O(\log n)$ | $O(n \log n)$ |

The deep network computes the product via a binary tree of pairwise multiplications, each of which can be implemented with a constant number of ReLU neurons (using the identity $xy = \frac{1}{4}[(x+y)^2 - (x-y)^2]$).

### Barron's Theorem (1993)

Barron provided dimension-independent error bounds for a specific function class:

!!! abstract "Barron's Theorem"
    Define the **Barron norm** (first Fourier moment) of $f$:
    
    $$
    C_f = \int_{\mathbb{R}^n} \|\boldsymbol{\omega}\| \, |\hat{f}(\boldsymbol{\omega})| \, d\boldsymbol{\omega}
    $$
    
    where $\hat{f}$ is the Fourier transform. If $C_f < \infty$, then a single-hidden-layer network with $N$ neurons achieves:
    
    $$
    \inf_{F_N} \|F_N - f\|_{L^2}^2 \leq \frac{C_f^2}{N}
    $$
    
    **Key insight:** The approximation rate $O(1/N)$ is **independent of input dimension** $n$.

This means that for "smooth enough" functions (finite Barron norm), shallow networks avoid the curse of dimensionality. However, many practically relevant functions do not have finite Barron norm, and the Barron norm itself may grow with dimension.

---

## Practical Implications

### What the Theorem Guarantees

The Universal Approximation Theorem tells us:

- Neural networks are expressive enough to model **any** continuous input-output relationship
- There are **no intrinsic representational limitations** — if the true function is continuous, a neural network can represent it
- A single hidden layer is sufficient **in principle** (though not necessarily in practice)

### What the Theorem Does NOT Guarantee

!!! warning "Existence ≠ Findability"
    The theorem is a **pure existence result**. It guarantees that good weights exist but is silent on:
    
    - **How many neurons are needed** for a specific problem and accuracy
    - **Whether gradient-based training will find** those weights (optimization landscape)
    - **Whether training will converge** in polynomial time
    - **Generalization** to unseen data (the theorem addresses approximation, not statistical learning)
    - **Computational efficiency** — the required width may be astronomically large

In practice, deep networks with moderate width consistently outperform wide-shallow networks, even though the theorem only requires one hidden layer. This gap between theory (existence) and practice (efficiency) motivates the study of depth (§4.2.3).

---

## PyTorch Demonstration

### Approximating a Complex 1D Function

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def target_function(x):
    """Complex non-linear target: product of oscillation and Gaussian envelope."""
    return np.sin(3 * x) * np.exp(-x**2) + 0.5 * np.cos(5 * x)


class ShallowApproximator(nn.Module):
    """Single hidden layer — the architecture the theorem guarantees."""
    def __init__(self, width: int, activation: nn.Module = nn.Tanh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width),
            activation(),
            nn.Linear(width, 1),
        )
    
    def forward(self, x):
        return self.net(x)


# ── Generate data ──
np.random.seed(42)
x_train = np.random.uniform(-3, 3, 1000).reshape(-1, 1).astype(np.float32)
y_train = target_function(x_train) + np.random.normal(0, 0.05, x_train.shape).astype(np.float32)

x_train_t = torch.from_numpy(x_train)
y_train_t = torch.from_numpy(y_train)

# ── Compare different widths ──
widths = [5, 20, 100, 500]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, width in enumerate(widths):
    ax = axes[idx // 2, idx % 2]
    
    torch.manual_seed(0)
    model = ShallowApproximator(width)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Train
    for epoch in range(3000):
        optimizer.zero_grad()
        loss = criterion(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    x_plot = torch.linspace(-3, 3, 500).reshape(-1, 1)
    with torch.no_grad():
        y_pred = model(x_plot).numpy()
    
    ax.scatter(x_train[:200], y_train[:200], alpha=0.2, s=5, color='gray', label='Train data')
    ax.plot(x_plot.numpy(), target_function(x_plot.numpy()), 'g-', lw=2, label='True $f(x)$')
    ax.plot(x_plot.numpy(), y_pred, 'r--', lw=2, label='NN approx')
    ax.set_title(f'Width $N = {width}$  |  Final MSE = {loss.item():.4f}')
    ax.legend(fontsize=8)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.grid(True, alpha=0.3)

plt.suptitle('Universal Approximation: Effect of Width (1 hidden layer, Tanh)', fontsize=14)
plt.tight_layout()
plt.savefig('universal_approximation_width.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Depth vs. Width Comparison

```python
class DeepNarrowNet(nn.Module):
    """Multiple hidden layers with moderate width."""
    def __init__(self, width: int = 32, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# Nested composition target — naturally favors depth
def nested_target(x):
    return torch.sin(torch.sin(torch.sin(x * 3) * 2) * 4)


x = torch.linspace(-2, 2, 1000).reshape(-1, 1)
y = nested_target(x)


def train_model(model, x, y, epochs=5000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


torch.manual_seed(42)
wide_model = ShallowApproximator(500, activation=nn.ReLU)   # 1 hidden, 500 wide
deep_model = DeepNarrowNet(width=32, depth=5)                # 5 hidden, 32 wide

print(f"Wide-shallow params: {sum(p.numel() for p in wide_model.parameters()):,}")
print(f"Deep-narrow  params: {sum(p.numel() for p in deep_model.parameters()):,}")

wide_losses = train_model(wide_model, x, y)
deep_losses = train_model(deep_model, x, y)

print(f"Wide-shallow final MSE: {wide_losses[-1]:.6f}")
print(f"Deep-narrow  final MSE: {deep_losses[-1]:.6f}")
```

---

## Key Takeaways

!!! success "Summary"
    1. **Universal approximation** guarantees that a single-hidden-layer network can approximate any continuous function on a compact domain to arbitrary accuracy
    2. The theorem applies to sigmoid (Cybenko), any non-polynomial activation (Hornik), and ReLU networks
    3. It is a **pure existence result** — it says nothing about how many neurons are needed or whether training will find the right weights
    4. **Width complexity** for shallow networks is $O((L/\varepsilon)^n)$, suffering from the curse of dimensionality
    5. **Deep networks** can achieve the same approximation with polynomially many parameters for compositional functions (exponential separation)
    6. **Barron's theorem** gives dimension-independent $O(1/N)$ rates for smooth (finite Fourier moment) functions
    7. In practice, **moderate depth with moderate width** consistently outperforms extreme width or extreme depth

---

## References

- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
- Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251–257.
- Barron, A. R. (1993). Universal approximation bounds for superpositions of a sigmoidal function. *IEEE Transactions on Information Theory*, 39(3), 930–945.
- Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
- Lu, Z., Pu, H., Wang, F., Hu, Z., & Wang, L. (2017). The expressive power of neural networks: A view from the width. *NeurIPS*.
