# Change of Variables

The **change-of-variables formula** is the mathematical foundation of every normalizing flow.  It describes how probability density transforms under an invertible mapping, and getting it right—conceptually, algebraically, and numerically—is essential.

## Intuition: Conservation of Probability

When an invertible function $f$ maps random variable $Z$ to $X = f(Z)$, the total probability mass is conserved but the *density* is re-distributed.  Regions where $f$ expands space have lower density; regions where $f$ compresses space have higher density.  The Jacobian determinant quantifies this local volume change.

Consider a simple 1-D example: $Z \sim \mathrm{Uniform}(0,1)$ and $X = Z^2$.  The map compresses probability near the origin and stretches it near 1.  The density of $X$ must adjust so that total probability remains one.

## One-Dimensional Derivation

Let $Z$ have density $p_Z(z)$ and $X = f(Z)$ where $f$ is monotone and differentiable.

**Step 1 — CDF.**  $\;P(X \le x) = P(Z \le f^{-1}(x))$

**Step 2 — Differentiate.**  $\;p_X(x) = p_Z(f^{-1}(x))\;\left|\frac{d\,f^{-1}(x)}{dx}\right|$

The absolute value is necessary because $f$ may be decreasing, but density must be non-negative.

### Example: Log-Normal from Gaussian

Let $Z \sim \mathcal{N}(0,1)$ and $X = e^Z$.  Then $f^{-1}(x) = \ln x$ and $|df^{-1}/dx| = 1/x$:

$$p_X(x) = \frac{1}{\sqrt{2\pi}}\,\exp\!\Bigl(-\tfrac{(\ln x)^2}{2}\Bigr)\;\frac{1}{x}$$

which is exactly the log-normal density.

## Multivariate Formula

For $Z \in \mathbb{R}^d$ with density $p_Z$ and diffeomorphism $f: \mathbb{R}^d \to \mathbb{R}^d$, the density of $X = f(Z)$ is:

$$\boxed{p_X(x) = p_Z\!\bigl(f^{-1}(x)\bigr)\;\left|\det \frac{\partial f^{-1}}{\partial x}\right|}$$

Equivalently, writing $z = f^{-1}(x)$:

$$p_X(x) = p_Z(z)\;\left|\det \frac{\partial f}{\partial z}\right|^{-1}$$

### The Jacobian Matrix

The Jacobian of $f: \mathbb{R}^d \to \mathbb{R}^d$ is the $d \times d$ matrix:

$$J_f(z) = \frac{\partial f}{\partial z} = \begin{pmatrix}
\frac{\partial f_1}{\partial z_1} & \cdots & \frac{\partial f_1}{\partial z_d} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_d}{\partial z_1} & \cdots & \frac{\partial f_d}{\partial z_d}
\end{pmatrix}$$

The absolute value of its determinant, $|\det J_f(z)|$, measures the factor by which $f$ scales infinitesimal volume at $z$.  If $|\det J| > 1$ the transformation expands volume locally; if $|\det J| < 1$ it contracts.

## Log-Space Formulation

For numerical stability and gradient-based training, the log-density is the working form:

$$\boxed{\log p_X(x) = \log p_Z(z) - \log\left|\det \frac{\partial f}{\partial z}\right|}$$

where $z = f^{-1}(x)$.  Equivalently, using the inverse Jacobian:

$$\log p_X(x) = \log p_Z(f^{-1}(x)) + \log\left|\det \frac{\partial f^{-1}}{\partial x}\right|$$

## Composing Transformations

For a chain of $K$ invertible functions $f = f_K \circ f_{K-1} \circ \cdots \circ f_1$, the chain rule for determinants gives:

$$\log|\det J_f| = \sum_{k=1}^{K}\log|\det J_{f_k}|$$

The full log-likelihood becomes:

$$\log p_X(x) = \log p_Z(z_0) + \sum_{k=1}^{K}\log\left|\det \frac{\partial f_k}{\partial z_{k-1}}\right|$$

where $z_0 = f_1^{-1}(f_2^{-1}(\cdots f_K^{-1}(x)))$.  Each layer contributes an additive log-determinant term, making training via back-propagation straightforward.

## Implementation

### Base Flow and Composition Classes

```python
import torch
import torch.nn as nn
import numpy as np


class Flow(nn.Module):
    """Abstract base class for a single flow layer."""

    def forward(self, z):
        """z -> x, returns (x, log_det)."""
        raise NotImplementedError

    def inverse(self, x):
        """x -> z, returns (z, log_det)."""
        raise NotImplementedError


class FlowSequence(nn.Module):
    """Compose multiple flow layers with a base distribution."""

    def __init__(self, flows, base_distribution):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.base_dist = base_distribution

    def forward(self, z):
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        for flow in self.flows:
            z, log_det = flow.forward(z)
            log_det_total += log_det
        return z, log_det_total

    def inverse(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            log_det_total += log_det
        return x, log_det_total

    def log_prob(self, x):
        z, log_det = self.inverse(x)
        log_pz = self.base_dist.log_prob(z)
        return log_pz + log_det

    def sample(self, n_samples, device="cpu"):
        z = self.base_dist.sample(n_samples, device)
        x, _ = self.forward(z)
        return x
```

### Gaussian Base Distribution

```python
class GaussianBase:
    """Standard multivariate Gaussian N(0, I)."""

    def __init__(self, dim):
        self.dim = dim

    def sample(self, n_samples, device="cpu"):
        return torch.randn(n_samples, self.dim, device=device)

    def log_prob(self, z):
        return -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
```

## Verification

Always verify implementations numerically during development.

### Numerical Jacobian Check

```python
def numerical_log_det(flow, z, eps=1e-5):
    """Compute log|det J| by finite differences."""
    d = z.shape[-1]
    jac = torch.zeros(z.shape[0], d, d)
    for i in range(d):
        z_plus = z.clone();  z_plus[:, i] += eps
        z_minus = z.clone(); z_minus[:, i] -= eps
        x_plus, _ = flow.forward(z_plus)
        x_minus, _ = flow.forward(z_minus)
        jac[:, :, i] = (x_plus - x_minus) / (2 * eps)
    _, log_det = torch.slogdet(jac)
    return log_det
```

### Invertibility Check

```python
def verify_invertibility(flow, z, tol=1e-5):
    x, ld_fwd = flow.forward(z)
    z_rec, ld_inv = flow.inverse(x)
    assert (z - z_rec).abs().max() < tol, "Invertibility failed"
    assert (ld_fwd + ld_inv).abs().max() < tol, "Log-det inconsistency"
```

## Common Pitfalls

**Sign errors in log-det.**  The Jacobian determinant can be negative (for orientation-reversing maps), but the density formula uses the absolute value.  Always use `torch.slogdet` or `torch.log(torch.abs(det))`.

**Forgetting the log-det term.**  Writing `log_prob = base_dist.log_prob(z)` without adding the Jacobian correction produces an incorrect density.

**Inconsistent sign convention.**  If `log_det` stores $\log|\det(\partial f/\partial z)|$ (forward Jacobian), then $\log p(x) = \log p_Z(z) - \text{log\_det}$.  If it stores the inverse Jacobian, the sign is $+$.  Be explicit in your code about which convention is used.

## Key Takeaways

The change-of-variables formula $p_X(x) = p_Z(f^{-1}(x))\;|\det J_{f^{-1}}(x)|$ is the single equation that underpins all normalizing flows.  In practice we work in log-space, composition multiplies determinants (adds in log-space), and the central architectural challenge is designing transformations whose Jacobian determinants can be computed in $O(d)$ rather than $O(d^3)$.

## References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*, Section 2.
2. Rudin, W. (1976). *Principles of Mathematical Analysis*. McGraw-Hill. (Change of variables theorem.)
