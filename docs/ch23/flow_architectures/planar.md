# Planar, Radial, and Simple Flows

The earliest normalizing flow architectures apply simple, analytically tractable transformations.  While largely superseded by coupling and spline methods for state-of-the-art density estimation, these simple flows remain valuable for building intuition, for variational inference posteriors, and as lightweight building blocks inside larger systems.

## Linear Flows

Linear (affine) transformations $f(z) = Az + b$ with invertible $A$ are the simplest possible flows.  They are limited in expressiveness—they can only rotate, scale, and shift—but they appear as components inside almost every modern architecture.

### Key Variants

| Variant | Parameters | Log-det Cost | Role in Practice |
|---|---|---|---|
| General affine | $d^2 + d$ | $O(d^3)$ | Small $d$ only |
| Diagonal affine | $2d$ | $O(d)$ | Scaling / shifting |
| Triangular | $d(d+1)/2 + d$ | $O(d)$ | Moderate expressiveness |
| LU-parameterised | $d^2 + d$ | $O(d)$ | Invertible 1×1 conv (Glow) |
| Orthogonal (Householder) | $kd$ | $O(1)$ | Volume-preserving rotations |
| ActNorm | $2d$ | $O(d)$ | Data-dependent normalisation |

### Diagonal Affine Flow

```python
class DiagonalAffineFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        x = z * torch.exp(self.log_scale) + self.shift
        return x, self.log_scale.sum().expand(z.shape[0])

    def inverse(self, x):
        z = (x - self.shift) * torch.exp(-self.log_scale)
        return z, -self.log_scale.sum().expand(x.shape[0])
```

### LU-Parameterised Affine Flow

Any invertible matrix decomposes as $A = PLU$.  Storing $P$ (fixed permutation), $L$ (lower triangular, unit diagonal), and $U$ (upper triangular) yields $\log|\det A| = \sum_i \log|U_{ii}|$ in $O(d)$.  This is the parameterisation used by Glow's invertible $1 \times 1$ convolution.

```python
class LUFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        W_init = torch.linalg.qr(torch.randn(dim, dim))[0]
        P, L, U = torch.linalg.lu(W_init)
        self.register_buffer("P", P)
        self.L_entries = nn.Parameter(torch.tril(L, -1))
        self.U_diag_log = nn.Parameter(torch.diagonal(U).abs().log())
        self.U_upper = nn.Parameter(torch.triu(U, 1))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.dim = dim

    def _get_W(self):
        L = self.L_entries + torch.eye(self.dim, device=self.L_entries.device)
        U = self.U_upper + torch.diag(torch.exp(self.U_diag_log))
        return self.P @ L @ U

    def forward(self, z):
        W = self._get_W()
        x = z @ W.T + self.bias
        return x, self.U_diag_log.sum().expand(z.shape[0])

    def inverse(self, x):
        W = self._get_W()
        z = (x - self.bias) @ torch.inverse(W).T
        return z, -self.U_diag_log.sum().expand(x.shape[0])
```

### Orthogonal Flows (Householder)

Orthogonal matrices satisfy $A^T A = I$, so $|\det A| = 1$ and the log-determinant is zero.  They are useful for mixing dimensions without changing the density.  A product of Householder reflections $H_v = I - 2 v v^T / \|v\|^2$ can represent any orthogonal matrix.

---

## Planar Flows

Introduced by Rezende & Mohamed (2015), planar flows apply a rank-1 perturbation along a hyperplane:

$$f(z) = z + u \cdot h(w^T z + b)$$

where $u, w \in \mathbb{R}^d$, $b \in \mathbb{R}$, and $h$ is a smooth activation (typically $\tanh$).

### Geometric Picture

The scalar $w^T z + b$ measures signed distance from the hyperplane normal to $w$.  The nonlinearity $h$ modulates this distance, and $u$ specifies the direction in which points are displaced.  The net effect is to "bend" space around the hyperplane.

### Log-Determinant via the Matrix Determinant Lemma

Because the Jacobian $J = I + u\,h'(w^Tz + b)\,w^T$ is a rank-1 update to the identity:

$$\log|\det J| = \log|1 + h'(w^Tz + b)\;u^Tw|$$

Cost: $O(d)$.

### Invertibility Constraint

For $h = \tanh$ (where $h' \in (0, 1]$), invertibility requires $u^Tw \ge -1$.  This is enforced by replacing $u$ with a constrained $\hat{u}$:

$$\hat{u} = u + \bigl(m(w^Tu) - w^Tu\bigr)\frac{w}{\|w\|^2}, \qquad m(x) = -1 + \text{softplus}(x)$$

### Implementation

```python
class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.u = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def _get_u_hat(self):
        wTu = torch.dot(self.w, self.u)
        m = -1 + F.softplus(wTu)
        return self.u + (m - wTu) * self.w / (self.w @ self.w + 1e-8)

    def forward(self, z):
        u_hat = self._get_u_hat()
        act = torch.tanh(z @ self.w + self.b)
        x = z + u_hat.unsqueeze(0) * act.unsqueeze(1)
        h_prime = 1 - act ** 2
        log_det = torch.log(torch.abs(1 + h_prime * (u_hat @ self.w)) + 1e-8)
        return x, log_det

    def inverse(self, x, n_iter=100, tol=1e-6):
        u_hat = self._get_u_hat()
        z = x.clone()
        for _ in range(n_iter):
            act = torch.tanh(z @ self.w + self.b)
            z_new = x - u_hat.unsqueeze(0) * act.unsqueeze(1)
            if (z_new - z).abs().max() < tol:
                break
            z = z_new
        _, ld = self.forward(z)
        return z, -ld
```

### Limitations

Each planar layer adds only a rank-1 perturbation, so many layers are needed for complex distributions.  There is no analytical inverse—fixed-point iteration must be used.

---

## Radial Flows

Radial flows (Rezende & Mohamed, 2015) expand or contract space around a reference point $z_0$:

$$f(z) = z + \frac{\beta}{\alpha + r}(z - z_0), \qquad r = \|z - z_0\|$$

Points are pushed away from $z_0$ when $\beta > 0$ and pulled toward it when $\beta < 0$.  The effect is strongest near the centre and vanishes at infinity.

### Log-Determinant

$$\log|\det J| = (d-1)\log\!\left|1 + \frac{\beta}{\alpha + r}\right| + \log\!\left|1 + \frac{\beta}{\alpha + r} - \frac{\beta r}{(\alpha + r)^2}\right|$$

Cost: $O(d)$.

### Invertibility

Requires $\beta \ge -\alpha$, enforced via $\beta = -\alpha + \text{softplus}(\beta_{\text{raw}})$.

### Implementation

```python
class RadialFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.z0 = nn.Parameter(torch.zeros(dim))
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.beta_raw = nn.Parameter(torch.zeros(1))

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @property
    def beta(self):
        return -self.alpha + F.softplus(self.beta_raw)

    def forward(self, z):
        diff = z - self.z0
        r = diff.norm(dim=1, keepdim=True)
        h = 1.0 / (self.alpha + r)
        x = z + self.beta * h * diff
        h_prime = -1.0 / (self.alpha + r) ** 2
        t1 = 1 + self.beta * h
        t2 = 1 + self.beta * h + self.beta * h_prime * r
        log_det = ((self.dim - 1) * torch.log(t1.abs())
                   + torch.log(t2.abs())).squeeze(1)
        return x, log_det

    def inverse(self, x, n_iter=100, tol=1e-6):
        z = x.clone()
        for _ in range(n_iter):
            diff = z - self.z0
            r = diff.norm(dim=1, keepdim=True)
            h = 1.0 / (self.alpha + r)
            z_new = x - self.beta * h * diff
            if (z_new - z).abs().max() < tol:
                break
            z = z_new
        _, ld = self.forward(z)
        return z, -ld
```

### Planar vs. Radial

Planar flows bend space along hyperplanes; radial flows deform space around points.  They are complementary—planar excels at capturing linear structures, radial at blob-like concentrations.  Stacking both can model richer distributions.

---

## Sylvester Flows

Sylvester flows (van den Berg et al., 2018) generalise planar flows from rank-1 to rank-$M$ updates:

$$f(z) = z + A\,h(B^Tz + b)$$

where $A, B \in \mathbb{R}^{d \times M}$ and $M \ll d$.

### Sylvester Determinant Identity

The identity $\det(I_d + AB^T) = \det(I_M + B^TA)$ reduces the $d \times d$ determinant to an $M \times M$ one, at cost $O(M^3 + dM)$ instead of $O(d^3)$.

For the flow's Jacobian:

$$\det J = \det(I_M + \text{diag}(h')\,B^TA)$$

### Orthogonal Parameterisation

Setting $A = QR$ and $B = QS$ with orthonormal $Q$ and positive diagonal $R, S$ gives $B^TA = SR$ (diagonal, positive definite), simplifying the determinant to:

$$\log|\det J| = \sum_{m=1}^{M}\log|1 + h'_m\,S_m\,R_m|$$

```python
class OrthogonalSylvesterFlow(nn.Module):
    def __init__(self, dim, M):
        super().__init__()
        self.dim, self.M = dim, M
        self.v = nn.Parameter(torch.randn(M, dim) * 0.01)
        self.log_R = nn.Parameter(torch.zeros(M))
        self.log_S = nn.Parameter(torch.zeros(M))
        self.b = nn.Parameter(torch.zeros(M))

    def _get_Q(self):
        """Gram-Schmidt orthonormalisation."""
        Q = torch.zeros(self.dim, self.M, device=self.v.device)
        for i in range(self.M):
            vi = self.v[i].clone()
            for j in range(i):
                vi -= torch.dot(vi, Q[:, j]) * Q[:, j]
            Q[:, i] = vi / (vi.norm() + 1e-8)
        return Q

    def forward(self, z):
        Q = self._get_Q()
        R, S = torch.exp(self.log_R), torch.exp(self.log_S)
        pre = S * (z @ Q) + self.b
        h = torch.tanh(pre)
        x = z + (h * R) @ Q.T
        h_prime = 1 - h ** 2
        log_det = torch.log(torch.abs(1 + h_prime * S * R) + 1e-8).sum(1)
        return x, log_det
```

### Trade-Off

| $M$ | Equivalent To | Log-det Cost |
|---|---|---|
| $1$ | Planar flow | $O(d)$ |
| $d/4$ | Good balance | $O(d)$ |
| $d$ | Full expressiveness | $O(d^3)$ |

Sylvester flows occupy the middle ground between the cheap-but-weak planar flow and the expensive-but-expressive full affine transformation.

---

## Comparison of Simple Flows

| Architecture | Update Rank | Analytical Inverse | Parameters | Best For |
|---|---|---|---|---|
| Planar | 1 | No | $2d+1$ | Hyperplane bending |
| Radial | 1 | No | $d+2$ | Point-centred deformation |
| Sylvester | $M$ | No | $O(dM)$ | Higher expressiveness |
| Linear (LU) | $d$ | Yes | $d^2$ | Channel mixing |
| Diagonal | — | Yes | $2d$ | Element-wise scaling |

Modern architectures (RealNVP, Glow, Neural Spline Flows) have largely superseded these simple flows for density estimation.  However, planar and radial flows remain popular in variational inference where the flow augments a simple approximate posterior, and linear/diagonal flows appear as sub-components of every modern architecture.

## References

1. Rezende, D. J. & Mohamed, S. (2015). Variational Inference with Normalizing Flows. *ICML*.
2. van den Berg, R., et al. (2018). Sylvester Normalizing Flows for Variational Inference. *UAI*.
3. Tomczak, J. M. & Welling, M. (2016). Improving Variational Auto-Encoders using Householder Flow. *arXiv*.
4. Kingma, D. P. & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. *NeurIPS*.
