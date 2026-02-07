# Neural Spline Flows

Neural Spline Flows (NSF; Durkan et al., 2019) replace the affine transformation inside coupling layers with **monotonic rational quadratic splines**.  This single change dramatically increases per-layer expressiveness: where an affine coupling can only apply a linear map to each dimension (for fixed conditioning), a spline coupling can apply an arbitrary smooth monotonic function.  The result is better density estimation with fewer layers and comparable parameter counts.

## Motivation: Beyond Affine Transforms

An affine coupling transforms each dimension of $z_B$ as:

$$y = z \cdot \exp(s) + t$$

For fixed $s$ and $t$ (which depend only on $z_A$), this is a *linear* function of $z$.  Capturing complex, nonlinear marginal structures requires stacking many affine layers.

A spline coupling replaces the affine map with:

$$y = \text{Spline}(z;\;\theta(z_A))$$

where $\theta$ are spline parameters predicted by the conditioner network.  Each layer can now express rich nonlinear behaviour, reducing the number of layers needed.

## Rational Quadratic Splines

### Why Rational Quadratic?

Among monotonic spline families, rational quadratic splines offer a compelling combination: they are smooth (infinitely differentiable), analytically invertible in closed form, have tractable derivatives for the log-determinant, and provide flexible shape control with a modest number of parameters.

### Parameterisation

A rational quadratic spline is defined over a bounded interval $[x_{\min}, x_{\max}]$ by $K$ bins.  The parameters per dimension are:

- **$K$ widths** $w_k > 0$ summing to $x_{\max} - x_{\min}$
- **$K$ heights** $h_k > 0$ summing to $y_{\max} - y_{\min}$
- **$K+1$ derivatives** $d_k > 0$ at knot points

The widths and heights are obtained by applying softmax to raw network outputs and scaling; the derivatives are obtained via softplus to ensure positivity.

### Within-Bin Formula

Within bin $k$, let $\xi = (x - x_k)/(x_{k+1} - x_k) \in [0, 1]$ and $s_k = (y_{k+1} - y_k)/(x_{k+1} - x_k)$.  The spline evaluates as:

$$y = y_k + \frac{(y_{k+1} - y_k)\bigl[s_k\,\xi^2 + d_k\,\xi(1-\xi)\bigr]}{s_k + (d_{k+1} + d_k - 2s_k)\,\xi(1-\xi)}$$

This is a ratio of two quadratics in $\xi$—hence "rational quadratic."

### Derivative (for Log-Determinant)

$$\frac{dy}{dx} = \frac{s_k^2\bigl[d_{k+1}\,\xi^2 + 2s_k\,\xi(1-\xi) + d_k(1-\xi)^2\bigr]}{\bigl[s_k + (d_{k+1} + d_k - 2s_k)\,\xi(1-\xi)\bigr]^2}$$

Because all $s_k, d_k > 0$, the numerator and denominator are positive, guaranteeing $dy/dx > 0$ (monotonicity) everywhere within each bin.

### Analytical Inverse

Because the forward map is a ratio of quadratics in $\xi$, inverting it amounts to solving a quadratic equation, which has a closed-form solution.  This is a major practical advantage over other flexible coupling transforms.

### Outside the Domain

Outside $[x_{\min}, x_{\max}]$ the spline typically defaults to the identity or a linear tail, ensuring the transformation is well-defined over all of $\mathbb{R}$.

## Implementation

### Spline Transform

```python
import torch
import torch.nn.functional as F


def rational_quadratic_spline_forward(
    x, widths, heights, derivatives, tail_bound=3.0
):
    """
    Apply rational quadratic spline element-wise.

    Args:
        x: input (batch, dim) — values in [-tail_bound, tail_bound]
        widths: (batch, dim, K)  — unnormalised bin widths
        heights: (batch, dim, K) — unnormalised bin heights
        derivatives: (batch, dim, K+1) — unnormalised knot derivatives

    Returns:
        y: transformed values
        log_det: log |dy/dx| per sample (batch,)
    """
    # Normalise parameters
    W = 2 * tail_bound * F.softmax(widths, dim=-1)       # positive, sum to 2*B
    H = 2 * tail_bound * F.softmax(heights, dim=-1)
    D = F.softplus(derivatives)                            # positive

    # Cumulative sums give knot positions
    x_k = torch.cumsum(W, dim=-1) - tail_bound
    y_k = torch.cumsum(H, dim=-1) - tail_bound
    x_k = F.pad(x_k, (1, 0), value=-tail_bound)
    y_k = F.pad(y_k, (1, 0), value=-tail_bound)

    # Find which bin each x falls in
    bin_idx = torch.searchsorted(x_k[..., 1:], x.unsqueeze(-1)).squeeze(-1)
    bin_idx = bin_idx.clamp(0, W.shape[-1] - 1)

    # Gather bin parameters
    x_lo = x_k.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    x_hi = x_k.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
    y_lo = y_k.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    y_hi = y_k.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
    d_lo = D.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    d_hi = D.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

    w = x_hi - x_lo
    h = y_hi - y_lo
    s = h / w

    xi = (x - x_lo) / w

    # Forward
    num = h * (s * xi ** 2 + d_lo * xi * (1 - xi))
    den = s + (d_hi + d_lo - 2 * s) * xi * (1 - xi)
    y = y_lo + num / den

    # Log derivative
    deriv_num = s ** 2 * (d_hi * xi ** 2 + 2 * s * xi * (1 - xi)
                          + d_lo * (1 - xi) ** 2)
    log_deriv = torch.log(deriv_num + 1e-8) - 2 * torch.log(den.abs() + 1e-8)

    return y, log_deriv.sum(dim=-1)
```

### Neural Spline Coupling Layer

```python
class NeuralSplineCoupling(nn.Module):
    """Coupling layer with rational quadratic spline transform."""

    def __init__(self, dim, hidden_dim=128, K=8, tail_bound=3.0):
        super().__init__()
        self.split = dim // 2
        self.K = K
        self.tail_bound = tail_bound
        d_trans = dim - self.split

        # Conditioner: outputs 3K+1 parameters per transformed dimension
        self.net = nn.Sequential(
            nn.Linear(self.split, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, d_trans * (3 * K + 1)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _params(self, x_cond):
        raw = self.net(x_cond)
        d_trans = raw.shape[-1] // (3 * self.K + 1)
        raw = raw.view(-1, d_trans, 3 * self.K + 1)
        W = raw[..., :self.K]
        H = raw[..., self.K:2*self.K]
        D = raw[..., 2*self.K:]
        return W, H, D

    def forward(self, z):
        z_a, z_b = z[:, :self.split], z[:, self.split:]
        W, H, D = self._params(z_a)
        x_b, log_det = rational_quadratic_spline_forward(
            z_b, W, H, D, self.tail_bound
        )
        return torch.cat([z_a, x_b], dim=-1), log_det

    def inverse(self, x):
        x_a, x_b = x[:, :self.split], x[:, self.split:]
        W, H, D = self._params(x_a)
        z_b, log_det = rational_quadratic_spline_inverse(
            x_b, W, H, D, self.tail_bound
        )
        return torch.cat([x_a, z_b], dim=-1), log_det
```

## Advantages Over Affine Coupling

| Aspect | Affine | Spline |
|---|---|---|
| Per-dimension transform | Linear | Arbitrary monotonic |
| Layers for same quality | Many (16–32) | Fewer (4–8) |
| Parameters per layer | Fewer | More ($3K+1$ per dim) |
| Inverse | Closed-form | Closed-form (quadratic) |
| Typical $K$ | — | 4–16 bins |

Increasing the number of bins $K$ increases flexibility at moderate cost.  In practice $K = 8$ is a good default; beyond $K = 16$ the returns diminish.

## Autoregressive Spline Flows

The spline transform can also be used within autoregressive structures (replacing the affine transform in MAF or IAF).  This combines the flexibility of splines with the full triangular Jacobian of autoregressive models.

## Applications to Finance

Neural spline flows are particularly well-suited to financial distribution modelling because asset return distributions exhibit complex shapes—skewness, heavy tails, multi-modality—that affine flows struggle to capture efficiently.  The spline's ability to model arbitrary marginal shapes means fewer layers are needed, reducing both computational cost and overfitting risk on the moderate-sized datasets typical of financial applications.

## Summary

Neural Spline Flows achieve a step change in per-layer expressiveness by replacing affine transforms with rational quadratic splines.  The key properties—monotonicity (guaranteed invertibility), closed-form inverse, tractable derivative—make the spline a drop-in replacement for the affine transform in any coupling or autoregressive architecture.

## References

1. Durkan, C., Bekasov, A., Murray, I. & Papamakarios, G. (2019). Neural Spline Flows. *NeurIPS*.
2. Gregory, J. A. & Delbourgo, R. (1982). Piecewise Rational Quadratic Interpolation to Monotonic Data. *IMA Journal of Numerical Analysis*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
