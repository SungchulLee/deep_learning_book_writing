# RealNVP and Coupling Layers

Coupling layers are the backbone of modern normalizing flows.  The idea is disarmingly simple: split the input into two parts, pass one part through unchanged, and transform the other using parameters computed from the first.  This guarantees invertibility by construction and yields triangular Jacobians with $O(d)$ determinant computation.  **RealNVP** (Real-valued Non-Volume Preserving flows; Dinh et al., 2017) demonstrated that stacking affine coupling layers with alternating masks produces a practical, scalable architecture for high-dimensional density estimation.

## The Coupling Principle

### Split-Transform-Merge

Given input $z \in \mathbb{R}^D$, partition it as $z = (z_A, z_B)$:

$$x_A = z_A \qquad\text{(identity)}$$
$$x_B = g(z_B;\;\theta(z_A)) \qquad\text{(parameterised transform)}$$

The function $\theta(\cdot)$ (the *conditioner*) can be an **arbitrary** neural network—it need not be invertible, and it does not contribute to the Jacobian determinant.  The only requirement is that $g$ be invertible in $z_B$ for any fixed $\theta$.

### Why This Works

**Invertibility.**  Since $x_A = z_A$, we recover the conditioning information from the output.  Then $z_B = g^{-1}(x_B;\;\theta(x_A))$.

**Efficient Jacobian.**  The Jacobian is block-triangular:

$$J = \begin{pmatrix} I & 0 \\ \partial x_B / \partial z_A & \partial g / \partial z_B \end{pmatrix}$$

so $\det J = \det(\partial g / \partial z_B)$, which depends only on how $g$ transforms $z_B$—not on the (potentially very complex) conditioner $\theta$.

## Affine Coupling

The most widely used coupling transform is the *affine* coupling:

$$x_B = z_B \odot \exp\!\bigl(s(z_A)\bigr) + t(z_A)$$

where $s(\cdot)$ and $t(\cdot)$ are scale and translation networks.

### Inverse

$$z_B = \bigl(x_B - t(x_A)\bigr) \odot \exp\!\bigl(-s(x_A)\bigr)$$

The inverse is as cheap as the forward pass—no iterative procedures.

### Log-Determinant

Since the Jacobian with respect to $z_B$ is diagonal with entries $\exp(s_i)$:

$$\log|\det J| = \sum_i s_i(z_A)$$

Simply the sum of the scale network outputs.

### Implementation

```python
import torch
import torch.nn as nn
import numpy as np


class AffineCouplingLayer(nn.Module):
    """Single affine coupling layer."""

    def __init__(self, dim, mask, hidden_dims=(256, 256)):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask.float())
        d_cond = int(mask.sum().item())
        d_trans = dim - d_cond

        layers = []
        prev = d_cond
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, d_trans * 2))
        self.net = nn.Sequential(*layers)
        # near-identity initialisation
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x_cond):
        params = self.net(x_cond)
        s, t = params.chunk(2, dim=-1)
        s = torch.tanh(s) * 2          # bound log-scale for stability
        return s, t

    def forward(self, x):
        x_cond = x[:, self.mask.bool()]
        s, t = self._st(x_cond)
        y = x.clone()
        y[:, ~self.mask.bool()] = x[:, ~self.mask.bool()] * torch.exp(s) + t
        return y, s.sum(dim=-1)

    def inverse(self, y):
        y_cond = y[:, self.mask.bool()]   # same as x_cond
        s, t = self._st(y_cond)
        x = y.clone()
        x[:, ~self.mask.bool()] = (y[:, ~self.mask.bool()] - t) * torch.exp(-s)
        return x, -s.sum(dim=-1)
```

### Scale Parameterisation

Using $\exp(s)$ ensures a positive scale factor.  In practice the raw scale output is clamped or squashed through $\tanh$ to prevent extreme values:

```python
# Bounded log-scale
s = torch.tanh(s_raw) * scale_bound      # e.g. scale_bound = 2

# Alternative: softplus for smooth positivity
scale = torch.nn.functional.softplus(s_raw)
```

## Splitting and Masking Strategies

A single coupling layer leaves half the dimensions unchanged.  **Alternating masks** across layers ensure every dimension is eventually transformed.

### Common Patterns

**Half-split (vectors):**  First half unchanged in even layers, second half in odd layers.

**Checkerboard (images):**  Spatial checkerboard pattern alternating between layers.

**Channel-wise (images):**  Split along the channel dimension.

```python
def alternating_mask(dim, layer_idx):
    mask = torch.zeros(dim)
    if layer_idx % 2 == 0:
        mask[:dim // 2] = 1
    else:
        mask[dim // 2:] = 1
    return mask
```

## RealNVP Architecture

RealNVP stacks affine coupling layers with alternating masks and adds batch normalisation for training stability.  For image data it employs a multi-scale architecture.

### Core Design

```
For each scale:
    For L coupling blocks:
        Affine coupling (checkerboard mask)
        Batch normalisation
    Affine coupling (channel mask)
    Factor out half channels → base distribution
```

### Complete 1-D RealNVP

```python
class RealNVP(nn.Module):
    """RealNVP for vector data."""

    def __init__(self, dim, n_layers=8, hidden_dims=(256, 256)):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = alternating_mask(dim, i)
            self.layers.append(AffineCouplingLayer(dim, mask, hidden_dims))

    def forward(self, x):
        ld = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det = layer.forward(x)
            ld += log_det
        return x, ld

    def inverse(self, z):
        ld = torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            ld += log_det
        return z, ld

    def log_prob(self, x):
        z, ld = self.forward(x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + ld

    def sample(self, n, device="cpu"):
        z = torch.randn(n, self.dim, device=device)
        x, _ = self.inverse(z)
        return x
```

### Multi-Scale Architecture (Images)

For image data, RealNVP uses a **squeeze** operation that trades spatial resolution for channel depth, followed by **factor-out** steps that send half the channels directly to the base distribution.  This reduces computation at later scales and allows early layers to model fine-grained spatial structure while deeper layers focus on semantics.

```
Input (3, 32, 32)
  → Squeeze → (12, 16, 16) → Coupling blocks → Factor out (6, 16, 16)
  → Squeeze → (24, 8, 8)   → Coupling blocks → Factor out (12, 8, 8)
  → Squeeze → (48, 4, 4)   → Coupling blocks → Base distribution
```

## The Conditioner Network

The conditioner $\theta(z_A)$ can be any neural network.  Its architecture controls per-layer expressiveness:

**MLP** for vector data — two or three hidden layers with ReLU, zero-initialised output.

**CNN** for image data — several $3 \times 3$ convolutional layers; ResNet blocks for deeper conditioners.

**Key design principle:** initialise the conditioner output to zero so that the flow starts as (or close to) the identity transformation, ensuring stable early training.

## Additive vs. Affine Coupling

Additive coupling ($x_B = z_B + t(z_A)$) is a special case with $s = 0$: the Jacobian determinant is always 1, making it volume-preserving.  This was used in NICE (Dinh et al., 2015).  Affine coupling with a learnable scale is strictly more expressive and has become the standard.

## Training

```python
def train_realnvp(model, data, epochs=100, batch_size=256, lr=1e-3):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=batch_size, shuffle=True,
    )
    for epoch in range(epochs):
        for (batch,) in loader:
            loss = -model.log_prob(batch).mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
```

## Summary

Coupling layers provide guaranteed invertibility, $O(d)$ Jacobian computation, and arbitrary conditioner flexibility.  RealNVP demonstrated that this design scales to complex, high-dimensional distributions by combining affine coupling with alternating masks and multi-scale processing.  The architecture remains a standard baseline and the foundation for Glow and Neural Spline Flows.

## References

1. Dinh, L., Krueger, D. & Bengio, Y. (2015). NICE: Non-linear Independent Components Estimation. *ICLR Workshop*.
2. Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2017). Density Estimation Using Real-NVP. *ICLR*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
