# Gradient Flow

## Learning Objectives

!!! abstract "What You Will Learn"
    - Analyze gradient magnitudes through deep networks using the Jacobian product perspective
    - Derive why sigmoid causes exponential gradient decay and quantify the vanishing gradient problem
    - Explain how ReLU, proper initialization, batch normalization, and skip connections each address gradient pathology
    - Derive the Xavier and He initialization schemes from variance-preservation arguments
    - Implement gradient monitoring tools and diagnose training issues from gradient statistics
    - Build residual networks that train reliably at 20+ layers

## Prerequisites

| Topic | Why It Matters |
|-------|---------------|
| Backpropagation (§4.2.5) | Gradient flow is the behavior of backprop through many layers |
| Chain rule and Jacobians | Gradient magnitudes depend on products of Jacobians |
| Activation functions (Ch 4.1) | Different activations have different gradient properties |

---

## Overview

The chain rule tells us that gradients in a deep network are **products of many factors** — one per layer. If these factors are consistently less than 1, gradients shrink exponentially toward zero (**vanishing gradients**). If they are consistently greater than 1, gradients grow exponentially (**exploding gradients**). Understanding and controlling this multiplicative dynamic is essential for training deep networks.

---

## The Jacobian Product Perspective

### Gradient as a Product of Jacobians

From the backpropagation recurrence (§4.2.5), the error signal at layer $l$ satisfies:

$$
\boldsymbol{\delta}^{[l]} = \left[\prod_{k=l+1}^{L} \text{diag}\!\left((\sigma^{[k]})'(\mathbf{z}^{[k]})\right) \cdot \mathbf{W}^{[k]}\right]^\top \boldsymbol{\delta}^{[L]}
$$

Writing $\mathbf{D}^{[k]} = \text{diag}((\sigma^{[k]})'(\mathbf{z}^{[k]}))$ for the diagonal matrix of activation derivatives, the **Jacobian** of the map $\mathbf{a}^{[k-1]} \mapsto \mathbf{z}^{[k]}$ (after activation) is $\mathbf{J}^{[k]} = \mathbf{D}^{[k]} \mathbf{W}^{[k]}$, and the gradient involves the product:

$$
\boldsymbol{\delta}^{[l]} = \left(\mathbf{J}^{[L]} \cdots \mathbf{J}^{[l+1]}\right)^\top \boldsymbol{\delta}^{[L]}
$$

### Gradient Norm Bound

Taking operator norms:

$$
\|\boldsymbol{\delta}^{[l]}\| \leq \left(\prod_{k=l+1}^{L} \|\mathbf{D}^{[k]}\| \cdot \|\mathbf{W}^{[k]}\|\right) \|\boldsymbol{\delta}^{[L]}\|
$$

The key quantity is $\gamma^{[k]} = \|\mathbf{D}^{[k]}\| \cdot \|\mathbf{W}^{[k]}\|$, the **per-layer gradient scaling factor**:

- If $\gamma^{[k]} < 1$ for most layers: $\|\boldsymbol{\delta}^{[l]}\| \to 0$ exponentially → **vanishing gradients**
- If $\gamma^{[k]} > 1$ for most layers: $\|\boldsymbol{\delta}^{[l]}\| \to \infty$ exponentially → **exploding gradients**
- If $\gamma^{[k]} \approx 1$ for all layers: gradients are **preserved** → healthy training

---

## Vanishing Gradient Problem

### Sigmoid: Quantitative Analysis

For the sigmoid activation $\sigma(z) = 1/(1 + e^{-z})$:

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z)) \leq \frac{1}{4}
$$

The maximum derivative is $\frac{1}{4}$, achieved at $z = 0$. Therefore $\|\mathbf{D}^{[k]}\| \leq \frac{1}{4}$, and:

$$
\|\boldsymbol{\delta}^{[l]}\| \leq \left(\frac{1}{4}\right)^{L-l} \prod_{k=l+1}^{L} \|\mathbf{W}^{[k]}\| \cdot \|\boldsymbol{\delta}^{[L]}\|
$$

Even if $\|\mathbf{W}^{[k]}\| = 1$, a 20-layer network attenuates gradients by:

$$
\left(\frac{1}{4}\right)^{19} \approx 3.6 \times 10^{-12}
$$

Gradients reaching early layers are effectively **zero** in floating-point precision.

### Tanh: Somewhat Better

For $\tanh(z)$, the maximum derivative is $\tanh'(0) = 1$, but for inputs away from zero, $\tanh'(z) = 1 - \tanh^2(z) < 1$. In practice, tanh still suffers from vanishing gradients, though less severely than sigmoid.

### Symptoms of Vanishing Gradients

- Early layers learn extremely slowly (weights barely change)
- Training loss plateaus despite non-converged performance
- Gradient norms decrease geometrically with layer index

---

## Exploding Gradient Problem

### When Gradients Grow

If $\|\mathbf{W}^{[k]}\|$ is large enough that $\gamma^{[k]} = \|\mathbf{D}^{[k]}\| \cdot \|\mathbf{W}^{[k]}\| > 1$, gradients grow exponentially. With ReLU ($\|\mathbf{D}^{[k]}\| = 1$ for active units), this happens when $\|\mathbf{W}^{[k]}\| > 1$.

### Symptoms

- Loss suddenly becomes `NaN` or `Inf`
- Weights grow to very large values
- Training becomes unstable (oscillating or diverging loss)

### Solution: Gradient Clipping

Gradient clipping constrains the gradient norm before the parameter update:

**Clip by norm** (rescale if too large):

$$
\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \leq \tau \\ \tau \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > \tau \end{cases}
$$

```python
# Clip global gradient norm to max_norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Clip by value** (clamp each component):

```python
# Clamp each gradient element to [-0.5, 0.5]
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

Norm clipping is generally preferred because it preserves the gradient direction.

---

## Solutions to Vanishing Gradients

### 1. ReLU Activation

$$
\text{ReLU}(z) = \max(0, z), \qquad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}
$$

For active neurons ($z > 0$), the gradient is exactly 1 — no attenuation. This is the primary reason ReLU became the default activation for deep networks.

**Caveat — Dead neurons:** If a neuron's pre-activation $z$ is always negative (for all training inputs), its gradient is always 0 and it can never recover. Variants like Leaky ReLU ($\alpha z$ for $z < 0$, with $\alpha \approx 0.01$) address this.

### 2. Proper Weight Initialization

The goal of initialization is to keep $\gamma^{[k]} \approx 1$ at the start of training.

#### Xavier (Glorot) Initialization

**Setting:** Layer with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs, using tanh or sigmoid activation.

**Derivation:** We want $\text{Var}(\mathbf{a}^{[l]}) = \text{Var}(\mathbf{a}^{[l-1]})$ (variance preservation through the forward pass).

For $z_j^{[l]} = \sum_{i=1}^{n_\text{in}} W_{ji} a_i^{[l-1]}$, assuming independence and zero mean:

$$
\text{Var}(z_j^{[l]}) = n_{\text{in}} \cdot \text{Var}(W_{ji}) \cdot \text{Var}(a_i^{[l-1]})
$$

Setting $\text{Var}(z_j^{[l]}) = \text{Var}(a_i^{[l-1]})$ requires:

$$
\text{Var}(W_{ji}) = \frac{1}{n_{\text{in}}}
$$

A symmetric argument for the backward pass gives $\text{Var}(W_{ji}) = 1/n_{\text{out}}$. The Xavier compromise averages both:

$$
\boxed{W_{ji} \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\; \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)}
$$

or equivalently $W_{ji} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$.

```python
nn.init.xavier_uniform_(layer.weight)   # uniform version
nn.init.xavier_normal_(layer.weight)    # normal version
```

#### He (Kaiming) Initialization

**Setting:** ReLU activation.

Since ReLU zeros out roughly half the activations, the effective fan-in is halved. This gives:

$$
\text{Var}(W_{ji}) = \frac{2}{n_{\text{in}}}
$$

$$
\boxed{W_{ji} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_{\text{in}}}\right)}
$$

```python
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

!!! tip "Rule of Thumb"
    Use **Xavier** for tanh/sigmoid, **He/Kaiming** for ReLU/Leaky ReLU. PyTorch's `nn.Linear` uses Kaiming uniform by default.

### 3. Batch Normalization

Batch normalization (Ioffe & Szegedy, 2015) normalizes the pre-activations to have zero mean and unit variance within each mini-batch, then applies a learned affine transform:

$$
\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \qquad \tilde{z}_i = \gamma \hat{z}_i + \beta
$$

where $\mu_B, \sigma_B^2$ are the batch statistics and $\gamma, \beta$ are learnable parameters.

**Why it helps gradient flow:** By keeping activations centered and normalized, batch normalization prevents them from entering the saturation regions of sigmoid/tanh, maintaining $|\sigma'(z)|$ near its maximum.

### 4. Residual (Skip) Connections

Residual connections (He et al., 2016) provide a **direct additive path** for gradients:

$$
\mathbf{a}^{[l]} = \underbrace{f(\mathbf{a}^{[l-1]})}_{\text{learned residual}} + \underbrace{\mathbf{a}^{[l-1]}}_{\text{identity shortcut}}
$$

The gradient through this connection:

$$
\frac{\partial \mathbf{a}^{[l]}}{\partial \mathbf{a}^{[l-1]}} = \underbrace{\frac{\partial f}{\partial \mathbf{a}^{[l-1]}}}_{\text{can vanish}} + \underbrace{\mathbf{I}}_{\text{always = 1}}
$$

The identity term $\mathbf{I}$ guarantees that gradients can flow **directly** from output to any layer, regardless of what happens in the learned residual branch. This is why ResNets can be trained with 100+ layers.

For a network with $L$ residual blocks, the gradient at any layer $l$ receives a **sum of paths** rather than a product:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \left(\mathbf{I} + \sum_{\text{paths}} \prod_{\text{residual Jacobians}}\right)
$$

Even if all residual paths vanish, the identity term preserves the gradient.

---

## PyTorch Implementation

### Gradient Monitoring

```python
import torch
import torch.nn as nn


def monitor_gradients(model: nn.Module) -> dict[str, dict]:
    """Collect gradient statistics after a backward pass."""
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            stats[name] = {
                'norm': g.norm().item(),
                'mean': g.mean().item(),
                'std':  g.std().item(),
                'max':  g.abs().max().item(),
            }
    return stats


def print_gradient_report(stats: dict):
    """Print gradient health report with warnings."""
    print(f"{'Parameter':<35s} {'Norm':>10s} {'Max':>10s} {'Status':>8s}")
    print("-" * 68)
    for name, s in stats.items():
        if s['norm'] < 1e-7:
            status = "⚠️ VANISH"
        elif s['norm'] > 1e3:
            status = "⚠️ EXPLOD"
        else:
            status = "✓"
        print(f"{name:<35s} {s['norm']:>10.2e} {s['max']:>10.2e} {status:>8s}")
```

### Comparing Activation Functions

```python
import matplotlib.pyplot as plt


def gradient_flow_experiment():
    """Compare gradient flow across different activations in a 10-layer network."""
    activations = {
        'ReLU':    nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh':    nn.Tanh,
    }
    
    results = {}
    
    for act_name, act_cls in activations.items():
        # Build 10-layer network
        layers = []
        for _ in range(10):
            layers.extend([nn.Linear(64, 64), act_cls()])
        layers.append(nn.Linear(64, 1))
        model = nn.Sequential(*layers)
        
        # Forward + backward
        torch.manual_seed(42)
        x = torch.randn(32, 64)
        y = torch.randn(32, 1)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        
        # Collect weight gradient norms per layer
        norms = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                norms.append(layer.weight.grad.norm().item())
        results[act_name] = norms
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {'ReLU': 'o-', 'Sigmoid': 's-', 'Tanh': '^-'}
    for act_name, norms in results.items():
        ax.semilogy(range(1, len(norms) + 1), norms, markers[act_name],
                     label=act_name, lw=2, ms=8)
    
    ax.set_xlabel('Layer (1 = closest to output)', fontsize=12)
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=12)
    ax.set_title('Gradient Flow: Effect of Activation Function (10-layer MLP)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gradient_flow_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


gradient_flow_experiment()
```

### Initialization Comparison

```python
def initialization_experiment():
    """Show effect of initialization on gradient flow."""
    inits = {
        'Default (Kaiming Uniform)': lambda m: None,  # PyTorch default
        'Xavier Normal':             lambda m: nn.init.xavier_normal_(m.weight),
        'He/Kaiming Normal':         lambda m: nn.init.kaiming_normal_(m.weight, nonlinearity='relu'),
        'Too Small (σ=0.01)':        lambda m: nn.init.normal_(m.weight, std=0.01),
        'Too Large (σ=1.0)':         lambda m: nn.init.normal_(m.weight, std=1.0),
    }
    
    results = {}
    for init_name, init_fn in inits.items():
        torch.manual_seed(42)
        layers = []
        for _ in range(10):
            lin = nn.Linear(64, 64)
            init_fn(lin)
            layers.extend([lin, nn.ReLU()])
        layers.append(nn.Linear(64, 1))
        model = nn.Sequential(*layers)
        
        x = torch.randn(32, 64)
        y = torch.randn(32, 1)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        
        norms = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                norms.append(layer.weight.grad.norm().item())
        results[init_name] = norms
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, norms in results.items():
        ax.semilogy(range(1, len(norms) + 1), norms, 'o-', label=name, lw=2, ms=6)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=12)
    ax.set_title('Effect of Weight Initialization on Gradient Flow', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('initialization_gradient_flow.png', dpi=150, bbox_inches='tight')
    plt.show()


initialization_experiment()
```

### Residual MLP

```python
class ResidualBlock(nn.Module):
    """Two-layer residual block with batch norm."""
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
    
    def forward(self, x):
        return torch.relu(x + self.block(x))   # skip connection


class ResidualMLP(nn.Module):
    """Deep MLP with residual connections."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.head(self.blocks(self.proj(x)))


# ── Verify gradient health at 20 layers ──
torch.manual_seed(42)
model = ResidualMLP(784, 128, 10, num_blocks=10)   # 20 effective layers

x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))
loss = nn.CrossEntropyLoss()(model(x), y)
loss.backward()

print("Residual MLP Gradient Report (20 layers):")
stats = monitor_gradients(model)
print_gradient_report(stats)
```

---

## Diagnostic Summary

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| Early layer gradients $\approx 0$ | Vanishing gradients | Use ReLU, He init, batch norm, or skip connections |
| Loss becomes `NaN`/`Inf` | Exploding gradients | Gradient clipping, lower learning rate, better init |
| Many neurons output 0 always | Dead ReLU neurons | Use Leaky ReLU or lower learning rate |
| Gradient norms vary wildly across layers | Poor initialization | Apply Xavier/He init; add batch normalization |
| Training stalls after initial progress | Saturated activations | Check activation distributions; add batch norm |

---

## Key Takeaways

!!! success "Summary"
    1. Gradients in deep networks are **products of per-layer Jacobians**, making them susceptible to exponential decay or growth
    2. **Sigmoid** causes vanishing gradients because $|\sigma'(z)| \leq 0.25$; a 20-layer sigmoid network attenuates gradients by $\sim 10^{-12}$
    3. **ReLU** preserves gradient magnitude ($\sigma'(z) = 1$ for active units) but introduces dead neuron risk
    4. **Xavier initialization** preserves variance for tanh/sigmoid; **He initialization** accounts for ReLU's halving effect
    5. **Batch normalization** keeps activations in the non-saturated regime
    6. **Residual connections** provide identity gradient paths: $\partial \mathbf{a}^{[l]} / \partial \mathbf{a}^{[l-1]} = \mathbf{I} + \partial f / \partial \mathbf{a}^{[l-1]}$, enabling training at 100+ layers
    7. **Gradient clipping** prevents explosion; norm clipping ($\tau$-rescaling) is preferred over value clipping
    8. **Monitor gradient norms** during training — they are the most direct diagnostic of training health

---

## References

- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *ICCV*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML*.
