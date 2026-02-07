# Jacobian Determinant

The Jacobian determinant is the central computational bottleneck in normalizing flows.  A naïve $d \times d$ determinant costs $O(d^3)$, which is infeasible for high-dimensional data (e.g., a $64 \times 64 \times 3$ image has $d = 12\,288$).  All practical flow architectures are designed around structured Jacobians that admit $O(d)$ determinant computation.

## Geometric Interpretation

The absolute value of the Jacobian determinant measures local volume scaling:

$$\text{Volume}(f(R)) = |\det J_f(z)| \cdot \text{Volume}(R)$$

for an infinitesimal region $R$ about $z$.  If $|\det J| > 1$, the map expands volume; if $|\det J| < 1$, it contracts.  A zero determinant means the map is singular and the transformation is not invertible at that point—precisely what flows must avoid.

## Computational Complexity of General Determinants

| Representation | Storage | Determinant Cost |
|---|---|---|
| General $d \times d$ | $O(d^2)$ | $O(d^3)$ (LU decomposition) |
| Diagonal | $O(d)$ | $O(d)$ |
| Triangular | $O(d^2)$ | $O(d)$ |
| Block diagonal | varies | $O(\sum d_i^3)$ |
| Identity + rank-$r$ update | $O(dr)$ | $O(dr^2)$ |

For a $64 \times 64 \times 3$ image, a general determinant would require $\sim 10^{12}$ operations per sample—completely infeasible.  The design of normalizing flow architectures is fundamentally the search for structured Jacobians in the right-hand column of this table.

## Triangular Jacobians

The most widely exploited structure is the triangular Jacobian.  For a lower- (or upper-) triangular matrix, the determinant is simply the product of diagonal entries:

$$\det(J) = \prod_{i=1}^{d} J_{ii}$$

In log-space:

$$\log|\det J| = \sum_{i=1}^{d} \log|J_{ii}|$$

This reduces the cost from $O(d^3)$ to $O(d)$.

### Autoregressive Transformations

If output $x_i$ depends only on inputs $z_1, \ldots, z_i$:

$$x_i = f_i(z_1, \ldots, z_i)$$

then the Jacobian is lower triangular:

$$J = \begin{pmatrix}
\partial x_1/\partial z_1 & 0 & \cdots & 0 \\
\partial x_2/\partial z_1 & \partial x_2/\partial z_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
\partial x_d/\partial z_1 & \cdots & \cdots & \partial x_d/\partial z_d
\end{pmatrix}$$

and $\det J = \prod_i \partial x_i / \partial z_i$.  This is the structure exploited by MAF and IAF.

### Coupling Layers

Coupling layers split input into two parts $z = (z_A, z_B)$ and set $x_A = z_A$, $x_B = g(z_B; \theta(z_A))$.  The Jacobian has block-triangular form:

$$J = \begin{pmatrix} I & 0 \\ \partial x_B / \partial z_A & \partial g / \partial z_B \end{pmatrix}$$

so $\det J = \det(\partial g / \partial z_B)$.  For affine coupling where $g$ is element-wise, this inner determinant is diagonal and costs $O(d)$.

## Diagonal Jacobians

Element-wise transformations $f(z)_i = h(z_i)$ yield diagonal Jacobians:

$$\det J = \prod_{i=1}^{d} h'(z_i), \qquad \log|\det J| = \sum_{i=1}^{d}\log|h'(z_i)|$$

Common element-wise flows and their log-determinant contributions:

| Transform | $f(z)$ | $\log\lvert\det J\rvert$ |
|---|---|---|
| Affine | $\alpha \odot z + \beta$ | $\sum_i \log|\alpha_i|$ |
| Leaky ReLU | $\max(\alpha z_i, z_i)$ | $\sum_{i:\,z_i < 0} \log|\alpha|$ |
| Sigmoid | $\sigma(z)$ | $\sum_i \log[\sigma(z_i)(1-\sigma(z_i))]$ |

## Matrix Determinant Lemma

For invertible $A$ and rank-$r$ update $UV^T$ ($U, V \in \mathbb{R}^{d \times r}$):

$$\det(A + UV^T) = \det(A)\;\det(I_r + V^T A^{-1} U)$$

This reduces a $d \times d$ determinant to an $r \times r$ determinant—a massive saving when $r \ll d$.

### Application to Planar Flows

A planar flow $f(z) = z + u\,h(w^Tz + b)$ is a rank-1 update to the identity.  Applying the lemma with $A = I$:

$$\det J_f = 1 + h'(w^Tz + b)\;u^Tw$$

This costs only $O(d)$.  The same idea extends to Sylvester flows with rank-$M$ updates, where the determinant costs $O(M^3 + dM)$.

## Implementation Patterns

### Diagonal Flow

```python
class DiagonalFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        x = z * torch.exp(self.log_scale) + self.shift
        log_det = self.log_scale.sum().expand(z.shape[0])
        return x, log_det

    def inverse(self, x):
        z = (x - self.shift) * torch.exp(-self.log_scale)
        log_det = -self.log_scale.sum().expand(x.shape[0])
        return z, log_det
```

### Coupling Flow (Block-Triangular)

```python
class CouplingFlow(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.split = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.split, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (dim - self.split)),
        )

    def forward(self, z):
        z1, z2 = z[:, :self.split], z[:, self.split:]
        params = self.net(z1)
        log_s, t = params.chunk(2, dim=-1)
        log_s = torch.tanh(log_s)          # bound for stability
        x2 = z2 * torch.exp(log_s) + t
        x = torch.cat([z1, x2], dim=-1)
        return x, log_s.sum(dim=-1)

    def inverse(self, x):
        x1, x2 = x[:, :self.split], x[:, self.split:]
        params = self.net(x1)              # x1 = z1
        log_s, t = params.chunk(2, dim=-1)
        log_s = torch.tanh(log_s)
        z2 = (x2 - t) * torch.exp(-log_s)
        z = torch.cat([x1, z2], dim=-1)
        return z, -log_s.sum(dim=-1)
```

## Numerical Stability

```python
# Use slogdet instead of log(det(.)) to avoid NaN for negative determinants
sign, log_abs_det = torch.slogdet(jacobian)

# Clamp diagonal entries away from zero before taking log
log_det = torch.log(torch.abs(diagonal) + 1e-8).sum(dim=-1)
```

## Verification

```python
def verify_jacobian(flow, z, tol=1e-3):
    """Compare analytical log-det against finite-difference estimate."""
    _, log_det_flow = flow.forward(z)
    log_det_num = numerical_log_det(flow, z)   # see Change of Variables page
    error = (log_det_flow - log_det_num).abs().max().item()
    assert error < tol, f"Jacobian error {error:.2e} exceeds tolerance"
```

## Summary

| Jacobian Structure | $\det$ Formula | Cost | Example Architectures |
|---|---|---|---|
| General | LU decomposition | $O(d^3)$ | Infeasible for large $d$ |
| Diagonal | $\prod_i J_{ii}$ | $O(d)$ | Element-wise transforms |
| Triangular | $\prod_i J_{ii}$ | $O(d)$ | Autoregressive (MAF, IAF) |
| Block triangular | $\prod_k \det J_k$ | $O(d)$ | Coupling (RealNVP, Glow) |
| Identity + rank-$r$ | Matrix det lemma | $O(dr^2)$ | Planar, Sylvester |

**Core insight**: flow architecture design is fundamentally about achieving efficient Jacobian determinant computation while preserving expressiveness.

## References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
2. Dinh, L., et al. (2015). NICE: Non-linear Independent Components Estimation. *ICLR Workshop*.
3. Kingma, D. P., et al. (2016). Improved Variational Inference with Inverse Autoregressive Flow. *NeurIPS*.
