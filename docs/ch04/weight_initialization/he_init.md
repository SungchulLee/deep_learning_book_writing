# He (Kaiming) Initialization

He initialization (He et al., 2015) sets weight variance to account for the fact that ReLU and its variants zero out a fraction of their inputs, reducing the effective variance passed to the next layer. By compensating with a factor of 2 (for standard ReLU), He initialization maintains stable activation and gradient magnitudes in deep networks with ReLU-family nonlinearities. It is the standard initialization for modern convolutional and feedforward architectures.

## Derivation

### The ReLU Variance Factor

For a pre-activation $z \sim \mathcal{N}(0, \sigma_z^2)$, the ReLU output is:

$$h = \text{ReLU}(z) = \max(0, z)$$

Since ReLU zeros the negative half of a symmetric distribution:

$$\mathbb{E}[h] = \frac{1}{2}\,\mathbb{E}[|z|] = \frac{\sigma_z}{\sqrt{2\pi}}$$

$$\mathbb{E}[h^2] = \frac{1}{2}\,\mathbb{E}[z^2] = \frac{\sigma_z^2}{2}$$

Therefore:

$$\text{Var}(h) = \mathbb{E}[h^2] - (\mathbb{E}[h])^2 = \frac{\sigma_z^2}{2} - \frac{\sigma_z^2}{2\pi} = \frac{\sigma_z^2}{2}\left(1 - \frac{1}{\pi}\right)$$

The dominant term is $\sigma_z^2/2$. He et al. use the approximation:

$$\text{Var}(\text{ReLU}(z)) \approx \frac{1}{2}\,\text{Var}(z)$$

which accounts for the zeroing of negative values while absorbing the smaller $1/\pi$ correction. This approximation is also exact for the second moment: $\mathbb{E}[\text{ReLU}(z)^2] = \frac{1}{2}\text{Var}(z)$.

### Forward Pass Condition

Combining the pre-activation variance from the previous section:

$$\text{Var}(z^{(l)}) = n_{\text{in}}\,\sigma^2\,\mathbb{E}\bigl[(h^{(l-1)})^2\bigr]$$

With $\mathbb{E}[h^2] = \frac{1}{2}\text{Var}(z^{(l-1)})$ for ReLU:

$$\text{Var}(z^{(l)}) = n_{\text{in}}\,\sigma^2 \cdot \frac{1}{2}\,\text{Var}(z^{(l-1)})$$

For layer-to-layer stability, $\text{Var}(z^{(l)}) = \text{Var}(z^{(l-1)})$:

$$n_{\text{in}}\,\sigma^2 \cdot \frac{1}{2} = 1$$

$$\boxed{\sigma^2 = \frac{2}{n_{\text{in}}}}$$

This is the **He initialization** variance formula (fan-in mode).

### Backward Pass Condition

Analogous analysis on the gradient signal gives:

$$\sigma^2 = \frac{2}{n_{\text{out}}}$$

The fan-out mode is sometimes used when backward gradient stability is more critical (e.g., in very deep networks without skip connections).

### Comparison with Xavier

| | Xavier | He |
|---|--------|-----|
| Forward condition | $n_{\text{in}}\,\sigma^2 = 1$ | $n_{\text{in}}\,\sigma^2 = 2$ |
| Activation assumption | $\text{Var}(f(z)) \approx \text{Var}(z)$ | $\text{Var}(\text{ReLU}(z)) = \frac{1}{2}\text{Var}(z)$ |
| Variance | $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ | $\frac{2}{n_{\text{in}}}$ |
| Effect | Correct for tanh/sigmoid | Correct for ReLU |

He initialization is exactly 2× the Xavier fan-in variance, compensating for the factor of $1/2$ introduced by ReLU.

## Distribution Variants

### He Normal (Kaiming Normal)

$$W \sim \mathcal{N}\!\left(0,\;\frac{2}{n_{\text{in}}}\right)$$

```python
import torch.nn as nn
import torch.nn.init as init

linear = nn.Linear(256, 128)
init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
```

### He Uniform (Kaiming Uniform)

Setting $a^2/3 = 2/n_{\text{in}}$:

$$W \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}}}},\;\sqrt{\frac{6}{n_{\text{in}}}}\right)$$

```python
init.kaiming_uniform_(linear.weight, mode='fan_in', nonlinearity='relu')
```

### Fan-In vs Fan-Out Mode

```python
# Fan-in: preserves forward activation magnitudes (default)
init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')

# Fan-out: preserves backward gradient magnitudes
init.kaiming_normal_(linear.weight, mode='fan_out', nonlinearity='relu')
```

Fan-in is the default and is appropriate for most architectures. Fan-out may be preferred when the network has a wide-to-narrow shape (e.g., a compression bottleneck).

## Extension to Leaky ReLU and PReLU

For Leaky ReLU with negative slope $a$:

$$\text{LeakyReLU}(z) = \begin{cases} z & z \geq 0 \\ az & z < 0 \end{cases}$$

The second moment becomes:

$$\mathbb{E}[\text{LeakyReLU}(z)^2] = \frac{1}{2}(1 + a^2)\,\text{Var}(z)$$

Setting the stability condition:

$$n_{\text{in}}\,\sigma^2 \cdot \frac{1 + a^2}{2} = 1$$

$$\boxed{\sigma^2 = \frac{2}{(1 + a^2)\,n_{\text{in}}}}$$

For standard ReLU ($a = 0$), this recovers $2/n_{\text{in}}$. For Leaky ReLU with $a = 0.01$, the correction is negligible ($1 + 0.0001 \approx 1$). For PReLU, $a$ is learned but initialised to 0.25, giving $\sigma^2 \approx 2/(1.0625 \cdot n_{\text{in}})$.

```python
# Leaky ReLU with slope 0.2
init.kaiming_normal_(linear.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
```

## Empirical Validation

```python
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt


def compare_init_relu(n_layers=50, hidden=512, n_samples=1024):
    """Compare Xavier vs He initialization with ReLU across many layers."""
    x = torch.randn(n_samples, hidden)

    results = {}
    for name, init_fn in [
        ('Xavier Normal', init.xavier_normal_),
        ('He Normal (fan_in)', lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')),
        ('He Normal (fan_out)', lambda w: init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')),
    ]:
        h = x.clone()
        stds = []
        for _ in range(n_layers):
            linear = nn.Linear(hidden, hidden, bias=False)
            init_fn(linear.weight)
            h = torch.relu(linear(h))
            stds.append(h.std().item())
        results[name] = stds

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, stds in results.items():
        ax.plot(stds, linewidth=2, label=name)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Activation Std')
    ax.set_title('Activation Standard Deviation Through 50 ReLU Layers')
    ax.legend()
    ax.set_ylim(0, 3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, label='Target')
    plt.tight_layout()
    plt.savefig('he_vs_xavier_relu.png', dpi=150, bbox_inches='tight')
    plt.show()


compare_init_relu()
```

### Gradient Magnitude Comparison

```python
def compare_gradient_flow(n_layers=30, hidden=256, n_samples=256):
    """Compare gradient magnitudes at each layer."""
    results = {}

    for name, init_fn in [
        ('Xavier', init.xavier_normal_),
        ('He', lambda w: init.kaiming_normal_(w, nonlinearity='relu')),
    ]:
        # Build network
        layers = []
        for _ in range(n_layers):
            linear = nn.Linear(hidden, hidden, bias=False)
            init_fn(linear.weight)
            layers.append(linear)

        # Forward pass with ReLU
        x = torch.randn(n_samples, hidden, requires_grad=True)
        h = x
        activations = []
        for linear in layers:
            h = torch.relu(linear(h))
            h.retain_grad()
            activations.append(h)

        # Backward pass
        loss = h.sum()
        loss.backward()

        grad_norms = [a.grad.norm().item() for a in activations]
        results[name] = grad_norms

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, norms in results.items():
        ax.plot(range(n_layers, 0, -1), norms, linewidth=2, label=name)
    ax.set_xlabel('Distance from Output (layers)')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Magnitude vs Distance from Output')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig('gradient_flow_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


compare_gradient_flow()
```

## Special Cases and Extensions

### Residual Networks

In ResNets, the output of each block is $h + F(h)$ where $F$ is the residual function. If $F$ is initialised with standard He weights, the variance doubles at every residual connection. Two common mitigations:

**Zero initialization of the last layer.** Set the final BatchNorm's $\gamma = 0$ or the final convolution's weights to zero so that $F(h) = 0$ initially:

```python
# ResNet block with zero-init last BN
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # He init for conv layers
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

        # Zero-init last BN so residual block starts as identity
        init.zeros_(self.bn2.weight)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(x + out)
```

**Variance scaling by depth.** GPT-2 and similar Transformer models scale the residual path by $1/\sqrt{2L}$ where $L$ is the number of layers:

```python
# Transformer-style residual scaling
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        # Scale output projection by 1/√(2L)
        init.normal_(self.ffn[2].weight, std=0.02 / (2 * n_layers) ** 0.5)

    def forward(self, x):
        return x + self.ffn(x)
```

### Orthogonal Initialization

An alternative to He: initialise $W$ as a random orthogonal matrix (or a sub-block thereof). Orthogonal matrices preserve norms exactly: $\|Wx\|_2 = \|x\|_2$. This provides perfect forward signal preservation regardless of activation function, though ReLU breaks the orthogonality after the first layer.

```python
linear = nn.Linear(256, 256, bias=False)
init.orthogonal_(linear.weight, gain=2 ** 0.5)  # gain=√2 for ReLU
```

The `gain` parameter accounts for the activation function's effect, analogous to the factor of 2 in He initialization.

### LSUV (Layer-Sequential Unit-Variance)

Data-driven initialization that iteratively adjusts each layer's weights until the output variance equals 1, measured on a batch of real data. This avoids reliance on distributional assumptions:

```python
def lsuv_init(model, data_batch, target_std=1.0, max_iter=10, tol=0.05):
    """Layer-Sequential Unit-Variance initialization."""
    model.eval()
    hooks = []
    activation_stds = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # He init as starting point
            init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                init.zeros_(module.bias)

            # Register hook to capture output std
            def hook_fn(mod, inp, out, name=name):
                activation_stds[name] = out.detach().std().item()
            hooks.append(module.register_forward_hook(hook_fn))

    # Iteratively adjust scales
    for iteration in range(max_iter):
        with torch.no_grad():
            model(data_batch)

        all_close = True
        for name, module in model.named_modules():
            if name in activation_stds:
                std = activation_stds[name]
                if abs(std - target_std) > tol and std > 1e-8:
                    module.weight.data *= target_std / std
                    all_close = False

        if all_close:
            break

    for h in hooks:
        h.remove()

    model.train()
```

## Quantitative Finance Application

He initialization is the default for most modern architectures used in finance:

**Deep hedging networks.** Recurrent or feedforward networks that learn dynamic hedging strategies use ReLU or Leaky ReLU activations throughout the hidden layers. He initialization ensures that the network can learn across long time horizons (many layers corresponding to rebalancing steps) without gradient degradation.

**Factor models with neural feature extractors.** When a neural network extracts nonlinear factors from a cross-section of asset returns, the initial forward pass must produce meaningful activations for all assets simultaneously. He initialization prevents the dead neuron problem where some assets receive zero activation and therefore zero gradient.

```python
import torch
import torch.nn as nn
import torch.nn.init as init


class DeepHedgingNetwork(nn.Module):
    """Network for learning hedging strategies with proper initialization."""

    def __init__(self, n_features, n_instruments, hidden_dim=128, n_layers=6):
        super().__init__()

        layers = []
        for i in range(n_layers):
            in_dim = n_features if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(hidden_dim, n_instruments))
        self.net = nn.Sequential(*layers)

        # He init for Leaky ReLU layers
        for m in self.net:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

        # Output layer: small init for conservative initial hedges
        init.xavier_normal_(self.net[-1].weight)
        init.zeros_(self.net[-1].bias)

    def forward(self, market_state):
        """Return hedge ratios for each instrument."""
        return self.net(market_state)
```

## Summary

| Aspect | Detail |
|--------|--------|
| **Variance formula** | $\sigma^2 = \frac{2}{n_{\text{in}}}$ (fan-in) or $\frac{2}{n_{\text{out}}}$ (fan-out) |
| **Key insight** | ReLU halves the variance → compensate with factor of 2 |
| **Best for** | ReLU, Leaky ReLU, PReLU, ELU |
| **Leaky ReLU extension** | $\sigma^2 = \frac{2}{(1+a^2)\,n_{\text{in}}}$ |
| **PyTorch** | `init.kaiming_normal_`, `init.kaiming_uniform_` |
| **Residual networks** | Combine with zero-init or $1/\sqrt{2L}$ scaling |
| **Also known as** | Kaiming initialization, MSRA initialization |

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." *ICCV*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Identity Mappings in Deep Residual Networks." *ECCV*.
3. Mishkin, D., & Matas, J. (2016). "All You Need is a Good Init." *ICLR*.
4. Zhang, H., Dauphin, Y. N., & Ma, T. (2019). "Fixup Initialization: Residual Learning Without Normalization." *ICLR*.
