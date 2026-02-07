# Glow

Glow (Kingma & Dhariwal, 2018) refines RealNVP with three targeted improvements: **Activation Normalisation** (ActNorm) replaces batch normalisation, **invertible 1×1 convolutions** replace fixed permutations for channel mixing, and a more systematic multi-scale architecture ties the components together.  These changes yield higher-quality image generation with exact likelihood computation, producing results competitive with GANs while retaining the principled probabilistic framework of normalizing flows.

## From RealNVP to Glow

| Component | RealNVP | Glow |
|---|---|---|
| Normalisation | Batch norm | ActNorm |
| Channel mixing | Fixed permutation | Invertible 1×1 conv |
| Architecture | Ad-hoc stacking | Systematic *flow step* |

## The Glow Block

Each *step of flow* in Glow comprises three sub-layers applied in sequence:

```
Input → ActNorm → Invertible 1×1 Conv → Affine Coupling → Output
```

Multiple steps are stacked per scale, and squeeze/factor-out operations separate the scales.

## Component 1: ActNorm

Batch normalisation is problematic in flows because it depends on batch statistics, behaves differently at train and test time, and complicates invertibility.  ActNorm is a data-dependent initialisation of a learnable diagonal affine layer: on the first batch it sets bias and scale so that activations have zero mean and unit variance, then both parameters are trained normally.

```python
class ActNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.initialized = False

    def initialize(self, x):
        with torch.no_grad():
            # x: (batch, channels, ...) — flatten spatial dims
            if x.dim() == 4:
                x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
            else:
                x_flat = x
            self.bias.data = -x_flat.mean(0)
            self.log_scale.data = -torch.log(x_flat.std(0) + 1e-6)
        self.initialized = True

    def forward(self, z):
        if not self.initialized:
            self.initialize(z)
        if z.dim() == 4:
            s = self.log_scale.exp().view(1, -1, 1, 1)
            b = self.bias.view(1, -1, 1, 1)
            x = (z + b) * s
            log_det = z.shape[2] * z.shape[3] * self.log_scale.sum()
        else:
            x = (z + self.bias) * self.log_scale.exp()
            log_det = self.log_scale.sum()
        return x, log_det.expand(z.shape[0])

    def inverse(self, x):
        if x.dim() == 4:
            s = self.log_scale.exp().view(1, -1, 1, 1)
            b = self.bias.view(1, -1, 1, 1)
            z = x / s - b
            log_det = -x.shape[2] * x.shape[3] * self.log_scale.sum()
        else:
            z = x / self.log_scale.exp() - self.bias
            log_det = -self.log_scale.sum()
        return z, log_det.expand(x.shape[0])
```

## Component 2: Invertible 1×1 Convolution

A $1 \times 1$ convolution with $c$ input and $c$ output channels is equivalent to applying the same $c \times c$ invertible matrix $W$ at every spatial location.  It generalises the fixed permutation used in RealNVP to a *learned* channel mixing.

### Log-Determinant

For an $h \times w$ feature map:

$$\log|\det J| = h \cdot w \cdot \log|\det W|$$

A naïve $O(c^3)$ determinant per forward pass would be costly.  Glow uses an **LU parameterisation**: decompose $W = PLU$ once at initialisation, then learn $L$ and $U$ while keeping $P$ fixed.  The log-determinant reduces to:

$$\log|\det W| = \sum_{i=1}^{c}\log|U_{ii}|$$

which is $O(c)$.

```python
class Invertible1x1Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c = channels
        W_init = torch.linalg.qr(torch.randn(c, c))[0]
        P, L, U = torch.linalg.lu(W_init)
        self.register_buffer("P", P)
        self.L_entries = nn.Parameter(torch.tril(L, -1))
        self.U_diag_log = nn.Parameter(torch.diagonal(U).abs().log())
        self.U_upper = nn.Parameter(torch.triu(U, 1))
        self.c = c

    def _W(self):
        L = self.L_entries + torch.eye(self.c, device=self.L_entries.device)
        U = self.U_upper + torch.diag(self.U_diag_log.exp())
        return self.P @ L @ U

    def forward(self, z):
        b, c, h, w = z.shape
        W = self._W()
        x = torch.nn.functional.conv2d(z, W.view(c, c, 1, 1))
        log_det = h * w * self.U_diag_log.sum()
        return x, log_det.expand(b)

    def inverse(self, x):
        b, c, h, w = x.shape
        W_inv = torch.inverse(self._W())
        z = torch.nn.functional.conv2d(x, W_inv.view(c, c, 1, 1))
        log_det = -h * w * self.U_diag_log.sum()
        return z, log_det.expand(b)
```

## Component 3: Affine Coupling

Glow uses the same affine coupling layer as RealNVP, splitting along the channel dimension.  The conditioner is typically a small ResNet or shallow CNN with zero-initialised output.

## Full Architecture

```
For each scale (L scales total):
    For K steps of flow:
        1. ActNorm
        2. Invertible 1×1 conv
        3. Affine coupling (channel split)

    If not final scale:
        Squeeze: (C, H, W) → (4C, H/2, W/2)
        Factor out half channels → Gaussian prior
```

### Squeeze Operation

The squeeze trades spatial resolution for channels by reshaping $2 \times 2$ spatial blocks into 4 channel slices:

```python
def squeeze(x):
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * 4, h // 2, w // 2)

def unsqueeze(x):
    b, c, h, w = x.shape
    x = x.view(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(b, c // 4, h * 2, w * 2)
```

### Glow Block Assembly

```python
def build_glow_block(channels, hidden_channels=512):
    return nn.ModuleList([
        ActNorm(channels),
        Invertible1x1Conv(channels),
        AffineCouplingConv(channels, hidden_channels),
    ])

def build_glow(channels, K=32, L=3, hidden=512):
    """Build Glow with L scales and K steps per scale."""
    blocks = nn.ModuleList()
    c = channels
    for scale in range(L):
        c *= 4                 # after squeeze
        for _ in range(K):
            blocks.append(build_glow_block(c, hidden))
        if scale < L - 1:
            c //= 2           # factor out half
    return blocks
```

## Typical Hyperparameters

| Dataset | $K$ (steps/scale) | $L$ (scales) | Hidden channels |
|---|---|---|---|
| MNIST | 16 | 2 | 256 |
| CIFAR-10 | 32 | 3 | 512 |
| CelebA 256 | 32 | 6 | 512 |

## Glow for Non-Image Data

The three-component design (normalisation → linear mixing → coupling) applies equally well to tabular data.  Replace $1 \times 1$ convolutions with dense linear layers (LU-parameterised), omit squeeze/factor-out, and use MLP conditioners.  This "1-D Glow" is a strong default for financial time-series modelling.

## Summary

Glow's contribution is architectural refinement: ActNorm solves the batch-norm problem, invertible $1 \times 1$ convolutions generalise fixed permutations, and the systematic flow-step design scales cleanly.  These ideas have become standard components in virtually all subsequent coupling-based flow architectures.

## References

1. Kingma, D. P. & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. *NeurIPS*.
2. Dinh, L., et al. (2017). Density Estimation Using Real-NVP. *ICLR*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
