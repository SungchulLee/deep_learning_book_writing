# Invertibility and Flow Composition

Invertibility is the defining constraint of normalizing flows.  Without a bijective transformation, neither density evaluation (which requires $f^{-1}$) nor sampling (which requires $f$) is possible.  This section defines what invertibility means for flows, catalogues the building blocks available for constructing invertible networks, and then develops the composition principle that lets simple invertible layers be stacked into expressive deep models.

## Mathematical Requirements

### Bijection

A function $f: \mathbb{R}^d \to \mathbb{R}^d$ is a bijection if it is both injective (one-to-one) and surjective (onto), so that a unique inverse $f^{-1}$ exists satisfying $f^{-1}(f(z)) = z$ and $f(f^{-1}(x)) = x$ for all $z, x$.

### Diffeomorphism

Normalizing flows require **diffeomorphisms**: bijections where both $f$ and $f^{-1}$ are continuously differentiable.  Equivalently, the Jacobian $\partial f / \partial z$ must be non-singular everywhere.  This is what permits the change-of-variables formula to be applied.

### Why Invertibility Matters

| Capability | Requires |
|---|---|
| Density evaluation $\log p(x)$ | Inverse $f^{-1}(x)$ |
| Sampling $x \sim p_X$ | Forward $f(z)$ |
| Exact reconstruction | Bijection: $f(f^{-1}(x)) = x$ |
| Deterministic latent code | No stochastic encoder |

## Invertible Building Blocks

Standard neural-network layers are typically not invertible: ReLU destroys information for negative inputs, pooling discards spatial detail, and fully connected layers with fewer outputs than inputs reduce dimensionality.  Flow layers must be designed from scratch.

### Element-Wise Invertible Functions

Apply a scalar invertible function to each dimension independently:

| Function | $f(z)$ | $f^{-1}(x)$ |
|---|---|---|
| Affine | $\alpha z + \beta$ | $(x-\beta)/\alpha$ |
| Exp / Log | $e^z$ | $\ln x$ |
| Sigmoid / Logit | $\sigma(z)$ | $\ln(x/(1-x))$ |
| Leaky ReLU | $\max(\alpha z, z)$ | $\max(x/\alpha, x)$ |
| Softplus | $\ln(1+e^z)$ | $\ln(e^x - 1)$ |

```python
class LeakyReLUFlow(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, z):
        x = torch.where(z >= 0, z, self.alpha * z)
        log_det = torch.where(
            z >= 0, torch.zeros_like(z),
            torch.full_like(z, np.log(self.alpha))
        ).sum(dim=-1)
        return x, log_det

    def inverse(self, x):
        z = torch.where(x >= 0, x, x / self.alpha)
        log_det = torch.where(
            x >= 0, torch.zeros_like(x),
            torch.full_like(x, -np.log(self.alpha))
        ).sum(dim=-1)
        return z, log_det
```

### Permutations

Reorder dimensions via a permutation matrix $P$.  The inverse is $P^T$ and $|\det P| = 1$, so permutations contribute zero to the log-determinant.

```python
class Permutation(nn.Module):
    def __init__(self, dim, mode="reverse"):
        super().__init__()
        perm = torch.arange(dim - 1, -1, -1) if mode == "reverse" \
               else torch.randperm(dim)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))

    def forward(self, z):
        return z[:, self.perm], torch.zeros(z.shape[0], device=z.device)

    def inverse(self, x):
        return x[:, self.inv_perm], torch.zeros(x.shape[0], device=x.device)
```

### Affine (Linear) Transformations

$f(z) = Az + b$ with invertible $A$.  The log-determinant is $\log|\det A|$.  For full $d \times d$ matrices the determinant costs $O(d^3)$; efficient parameterisations include LU decomposition (Glow) and Householder reflections (orthogonal flows).

### Coupling Layers

Split $z = (z_A, z_B)$, pass $z_A$ through unchanged, and transform $z_B$ using parameters computed from $z_A$:

$$x_A = z_A, \qquad x_B = g(z_B;\;\theta(z_A))$$

Invertibility only requires that $g$ is invertible in $z_B$ for any fixed $\theta$.  The function $\theta(\cdot)$ can be an arbitrary neural network—it need not be invertible and does not affect the Jacobian determinant.

```python
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.split = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.split, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (dim - self.split)),
        )

    def forward(self, z):
        z_a, z_b = z[:, :self.split], z[:, self.split:]
        params = self.net(z_a)
        log_s, t = params.chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * 2
        x_b = z_b * torch.exp(log_s) + t
        return torch.cat([z_a, x_b], dim=-1), log_s.sum(dim=-1)

    def inverse(self, x):
        x_a, x_b = x[:, :self.split], x[:, self.split:]
        params = self.net(x_a)            # x_a = z_a
        log_s, t = params.chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * 2
        z_b = (x_b - t) * torch.exp(-log_s)
        return torch.cat([x_a, z_b], dim=-1), -log_s.sum(dim=-1)
```

## Operations to Avoid

| Operation | Why Not Invertible | Flow Alternative |
|---|---|---|
| ReLU | $\text{ReLU}(-1) = \text{ReLU}(-2) = 0$ | Leaky ReLU |
| Max pooling | Many inputs map to same output | Invertible downsampling |
| Dropout | Random zeroing | Deterministic regularisation |
| Batch norm (naïve) | Running statistics | ActNorm |
| Dimension reduction | Information loss | Multi-scale factoring |

## The Composition Principle

A single invertible layer is rarely expressive enough to capture a complex target distribution.  The power of normalizing flows comes from **composing** many simple layers.

### Chain of Flows

Given invertible layers $f_1, f_2, \ldots, f_K$:

$$f = f_K \circ f_{K-1} \circ \cdots \circ f_1$$

$$z_0 \;\xrightarrow{f_1}\; z_1 \;\xrightarrow{f_2}\; z_2 \;\xrightarrow{\;\cdots\;}\; z_K = x$$

### Properties Preserved Under Composition

If each $f_k$ is invertible with a tractable Jacobian determinant, the composition is also invertible with a tractable Jacobian:

$$f^{-1} = f_1^{-1} \circ f_2^{-1} \circ \cdots \circ f_K^{-1}$$

$$\log|\det J_f| = \sum_{k=1}^{K}\log|\det J_{f_k}|$$

### Design Patterns

**Homogeneous stacking** — repeat the same layer type $K$ times.  Simple but may require many layers.

**Alternating coupling** — alternate which dimensions are left unchanged, so that all dimensions are transformed after two consecutive layers.

**Coupling + normalisation** — interleave coupling layers with normalisation (ActNorm) and channel mixing (invertible 1×1 convolutions), as in Glow.

**Multi-scale** — process at different spatial resolutions, factoring out half the channels at each scale.  This reduces computation for image flows.

### Depth Guidelines

| Application | Typical Layers | Architecture |
|---|---|---|
| 2-D toy data | 4–8 | Planar or coupling |
| Tabular / financial | 8–16 | Coupling or MAF |
| MNIST | 8–16 | RealNVP |
| CIFAR-10 | 32 | Glow (multi-scale) |

### Gradient Flow in Deep Compositions

Deep stacks can suffer from vanishing or exploding gradients.  Practical mitigations include ActNorm (data-dependent initialisation of affine normalisation), near-identity initialisation (all flow layers start close to the identity map), and gradient checkpointing to trade compute for memory.

```python
class ActNorm(nn.Module):
    """Activation Normalization — data-dependent init, then learned."""

    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.initialized = False

    def initialize(self, x):
        with torch.no_grad():
            self.bias.data = -x.mean(dim=0)
            self.log_scale.data = -torch.log(x.std(dim=0) + 1e-6)
        self.initialized = True

    def forward(self, z):
        if not self.initialized:
            self.initialize(z)
        x = (z + self.bias) * torch.exp(self.log_scale)
        return x, self.log_scale.sum().expand(z.shape[0])

    def inverse(self, x):
        z = x * torch.exp(-self.log_scale) - self.bias
        return z, -self.log_scale.sum().expand(x.shape[0])
```

## Verifying Invertibility

```python
def verify_invertibility(flow, z, tol=1e-5):
    """Check f^{-1}(f(z)) ≈ z and f(f^{-1}(x)) ≈ x."""
    x, _ = flow.forward(z)
    z_rec, _ = flow.inverse(x)
    err_fwd = (z - z_rec).abs().max().item()

    z_from_x, _ = flow.inverse(x)
    x_rec, _ = flow.forward(z_from_x)
    err_inv = (x - x_rec).abs().max().item()

    assert max(err_fwd, err_inv) < tol, "Invertibility check failed"
```

## Numerical Precision

Even mathematically invertible functions can accumulate floating-point error:

```python
# Potential overflow / underflow
x = torch.exp(z)          # overflow for large z
z_back = torch.log(x)     # -inf for x ≈ 0

# Safer version
x = torch.exp(torch.clamp(z, -20, 20))
z_back = torch.log(torch.clamp(x, 1e-8))
```

Well-conditioned transformations have bounded Jacobian eigenvalues.  Monitoring the condition number during training is a useful diagnostic.

## Key Takeaways

Invertibility is the non-negotiable constraint of normalizing flows.  The practical toolkit—element-wise functions, permutations, coupling layers, autoregressive structures—provides a rich set of building blocks.  Composing these blocks in well-chosen patterns (alternating masks, interleaved normalisation, multi-scale) yields deep, expressive models with tractable exact likelihoods.

## References

1. Dinh, L., et al. (2015). NICE: Non-linear Independent Components Estimation. *ICLR Workshop*.
2. Dinh, L., et al. (2017). Density Estimation Using Real-NVP. *ICLR*.
3. Kingma, D. P. & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. *NeurIPS*.
4. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
