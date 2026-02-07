# Xavier (Glorot) Initialization

Xavier initialization (Glorot & Bengio, 2010) sets the weight variance to preserve signal magnitude through layers when the activation function is approximately linear around zero — the regime where sigmoid and tanh operate for properly scaled inputs. It is the correct default for networks using sigmoid, tanh, or other symmetric, zero-centred activations.

## Derivation

### Setup

Consider layer $l$ with fan-in $n_{\text{in}} = n_{l-1}$ and fan-out $n_{\text{out}} = n_l$. Weights are drawn i.i.d. with $\mathbb{E}[W] = 0$ and $\text{Var}(W) = \sigma^2$. Biases are zero. Inputs have zero mean.

### Forward Pass Condition

The pre-activation at layer $l$ is:

$$z_j^{(l)} = \sum_{i=1}^{n_{\text{in}}} W_{ij}^{(l)}\,h_i^{(l-1)}$$

Taking the variance (using $\mathbb{E}[W] = 0$ and independence):

$$\text{Var}(z_j^{(l)}) = n_{\text{in}}\,\sigma^2\,\text{Var}(h_i^{(l-1)})$$

For **symmetric activations near the origin** — sigmoid centred at 0.5 or tanh — the activation behaves approximately as $f(z) \approx \alpha z$ for small $z$ (where $\alpha = 1$ for tanh at $z = 0$). Under this linear approximation:

$$\text{Var}(h^{(l)}) \approx \text{Var}(z^{(l)})$$

To preserve variance across layers, we need:

$$n_{\text{in}}\,\sigma^2 = 1 \quad \Longrightarrow \quad \sigma^2 = \frac{1}{n_{\text{in}}}$$

### Backward Pass Condition

For the gradient flowing backward through layer $l$:

$$\frac{\partial \mathcal{L}}{\partial h_i^{(l-1)}} = \sum_{j=1}^{n_{\text{out}}} W_{ij}^{(l)}\,f'(z_j^{(l)})\,\frac{\partial \mathcal{L}}{\partial z_j^{(l)}}$$

Under the same linear activation approximation ($f'(z) \approx 1$):

$$\text{Var}\!\left(\frac{\partial \mathcal{L}}{\partial h^{(l-1)}}\right) = n_{\text{out}}\,\sigma^2\,\text{Var}\!\left(\frac{\partial \mathcal{L}}{\partial z^{(l)}}\right)$$

Backward stability requires:

$$n_{\text{out}}\,\sigma^2 = 1 \quad \Longrightarrow \quad \sigma^2 = \frac{1}{n_{\text{out}}}$$

### The Xavier Compromise

The forward condition gives $\sigma^2 = 1/n_{\text{in}}$; the backward condition gives $\sigma^2 = 1/n_{\text{out}}$. These are simultaneously satisfiable only when $n_{\text{in}} = n_{\text{out}}$. Glorot and Bengio proposed the harmonic mean:

$$\boxed{\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}}$$

This balances forward and backward signal preservation. For square layers ($n_{\text{in}} = n_{\text{out}} = n$), this reduces to $\sigma^2 = 1/n$, satisfying both conditions exactly.

## Distribution Variants

### Xavier Normal

$$W \sim \mathcal{N}\!\left(0,\;\frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

```python
import torch.nn.init as init

linear = nn.Linear(256, 128)
init.xavier_normal_(linear.weight)  # σ² = 2 / (256 + 128) ≈ 0.0052
```

### Xavier Uniform

For a uniform distribution $W \sim \mathcal{U}(-a, a)$, the variance is $a^2/3$. Setting $a^2/3 = 2/(n_{\text{in}} + n_{\text{out}})$:

$$W \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\;\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

```python
init.xavier_uniform_(linear.weight)  # a = √(6 / (256 + 128)) ≈ 0.125
```

The uniform variant is **PyTorch's default** for `nn.Linear`, which initialises weights from $\mathcal{U}(-1/\sqrt{n_{\text{in}}},\, 1/\sqrt{n_{\text{in}}})$ — close to but not exactly Xavier uniform.

## Empirical Validation

```python
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt


def track_activation_stats(init_fn, activation, n_layers=30, hidden=256, n_samples=1024):
    """Track mean and std of activations through a deep network."""
    x = torch.randn(n_samples, hidden)
    stds = []
    h = x.clone()

    for l in range(n_layers):
        linear = nn.Linear(hidden, hidden, bias=False)
        init_fn(linear.weight)
        h = activation(linear(h))
        stds.append(h.std().item())

    return stds


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Tanh with Xavier — stable
stds = track_activation_stats(init.xavier_normal_, torch.tanh)
axes[0].plot(stds, 'g-', linewidth=2)
axes[0].set_title('Tanh + Xavier (Correct)')
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Activation Std')
axes[0].set_ylim(0, 2)
axes[0].axhline(y=stds[0], color='gray', linestyle='--', alpha=0.5)

# Tanh with He — too large
stds = track_activation_stats(
    lambda w: init.kaiming_normal_(w, nonlinearity='relu'), torch.tanh
)
axes[1].plot(stds, 'r-', linewidth=2)
axes[1].set_title('Tanh + He (Too Large)')
axes[1].set_xlabel('Layer')
axes[1].set_ylim(0, 2)

# ReLU with Xavier — decays
stds = track_activation_stats(init.xavier_normal_, torch.relu)
axes[2].plot(stds, 'orange', linewidth=2, label='Xavier')
stds_he = track_activation_stats(
    lambda w: init.kaiming_normal_(w, nonlinearity='relu'), torch.relu
)
axes[2].plot(stds_he, 'g-', linewidth=2, label='He')
axes[2].set_title('ReLU: Xavier vs He')
axes[2].set_xlabel('Layer')
axes[2].set_ylim(0, 2)
axes[2].legend()

plt.tight_layout()
plt.savefig('xavier_validation.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Why Xavier Fails for ReLU

The derivation assumes the activation function preserves variance: $\text{Var}(f(z)) \approx \text{Var}(z)$. For tanh near zero, $f'(0) = 1$ so this holds. For ReLU:

$$\text{ReLU}(z) = \max(0, z)$$

Half the activations are zeroed out, so:

$$\text{Var}(\text{ReLU}(z)) = \frac{1}{2}\,\text{Var}(z)$$

Each layer halves the variance, causing exponential decay. After $L$ layers:

$$\text{Var}(h^{(L)}) \approx \left(\frac{1}{2}\right)^L \text{Var}(x)$$

For $L = 20$ layers, activations are attenuated by a factor of $2^{-20} \approx 10^{-6}$. This motivates He initialization, which compensates with a factor of 2 (covered in the next section).

## For Convolutional Layers

For a convolutional layer with kernel size $k \times k$, $C_{\text{in}}$ input channels, and $C_{\text{out}}$ output channels:

$$n_{\text{in}} = C_{\text{in}} \cdot k \cdot k, \qquad n_{\text{out}} = C_{\text{out}} \cdot k \cdot k$$

```python
conv = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
init.xavier_normal_(conv.weight)  # σ² = 2 / (64·9 + 128·9) = 2 / 1728 ≈ 0.00116
```

## Quantitative Finance Application

Xavier initialization is well-suited for networks that use tanh or sigmoid activations, which appear frequently in finance:

**Bounded output networks.** Models that predict probabilities (default probability, probability of fill) or bounded quantities (correlation between –1 and 1) often use tanh or sigmoid output activations. Xavier ensures these output layers receive inputs in the linear regime where gradients are informative.

**Recurrent networks for time series.** Classical RNNs and LSTMs use tanh and sigmoid gates. Xavier initialization of the input-to-hidden weights, combined with orthogonal initialization of the hidden-to-hidden weights, is the standard recipe for stable training.

```python
class CorrelationPredictor(nn.Module):
    """Predict bounded correlation ρ ∈ [-1, 1] using tanh output."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # output in [-1, 1]
        )
        # Xavier init for all layers (tanh activation)
        for m in self.net:
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
```

## Summary

| Aspect | Detail |
|--------|--------|
| **Variance formula** | $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ |
| **Assumption** | Activation is approximately linear near origin ($f'(0) \approx 1$) |
| **Best for** | Sigmoid, tanh, and other symmetric saturating activations |
| **Fails for** | ReLU (variance halves per layer → exponential decay) |
| **PyTorch** | `init.xavier_normal_`, `init.xavier_uniform_` |
| **Also known as** | Glorot initialization |

## References

1. Glorot, X., & Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*.
2. LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). "Efficient BackProp." *Neural Networks: Tricks of the Trade*, Springer.
