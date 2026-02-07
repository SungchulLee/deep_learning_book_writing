# KL Divergence and Fisher Information

The connection between KL divergence and the Fisher information matrix reveals that KL divergence, despite not being a global metric, behaves locally like a quadratic distance. This local geometry provides the foundation for natural gradient descent, trust region methods (TRPO, PPO), and the information-geometric perspective on statistical inference.

## Fisher Information Matrix

### Definition

For a parametric family $\{p(x|\theta) : \theta \in \Theta\}$, the **Fisher information matrix** is:

$$\mathbf{I}(\theta) = \mathbb{E}_{p(x|\theta)}\!\left[\nabla_\theta \log p(x|\theta) \;\nabla_\theta \log p(x|\theta)^T\right]$$

The vector $s(\theta; x) = \nabla_\theta \log p(x|\theta)$ is called the **score function**. Under regularity conditions (exchangeability of integration and differentiation), the Fisher information admits an equivalent form:

$$\mathbf{I}(\theta) = -\mathbb{E}_{p(x|\theta)}\!\left[\nabla_\theta^2 \log p(x|\theta)\right]$$

This second form shows that the Fisher information measures the **expected curvature** of the log-likelihood surface—a high Fisher information in a parameter direction means the log-likelihood is sharply peaked, making the parameter precisely estimable from data.

### Score Function Has Zero Mean

A fundamental property: $\mathbb{E}_{p(x|\theta)}[s(\theta; x)] = 0$. This follows from:

$$\int \nabla_\theta p(x|\theta)\,dx = \nabla_\theta \int p(x|\theta)\,dx = \nabla_\theta 1 = 0$$

Since $\nabla_\theta \log p = \nabla_\theta p / p$, we get $\mathbb{E}_p[\nabla_\theta \log p] = \int \nabla_\theta p \,dx = 0$. This means the Fisher information is exactly the **covariance matrix of the score function**.

## Taylor Expansion of KL Divergence

### Setup

Consider two nearby distributions in the same parametric family: $p_0 = p(x|\theta_0)$ and $p = p(x|\theta)$ with $\theta = \theta_0 + \delta\theta$. We expand $D_{\text{KL}}(p_0 \| p)$ in powers of $\delta\theta$.

### Expansion

$$D_{\text{KL}}(p_0 \| p) = \int p_0(x)\log\frac{p_0(x)}{p(x)}\,dx = -\int p_0(x)\log\frac{p(x)}{p_0(x)}\,dx$$

Define $g(\theta) = \int p_0(x)\log p(x|\theta)\,dx$. Then $D_{\text{KL}}(p_0 \| p) = g(\theta_0) - g(\theta)$ (since $\int p_0 \log p_0$ is constant).

Taylor-expand $g(\theta)$ around $\theta_0$:

$$g(\theta) = g(\theta_0) + \nabla_\theta g(\theta_0)^T \delta\theta + \frac{1}{2}\delta\theta^T \nabla_\theta^2 g(\theta_0)\,\delta\theta + O(\|\delta\theta\|^3)$$

**First derivative:**

$$\frac{\partial g}{\partial \theta_k}\bigg|_{\theta_0} = \int p_0(x)\frac{\partial}{\partial \theta_k}\log p(x|\theta)\bigg|_{\theta_0}dx = \mathbb{E}_{p_0}\!\left[\frac{\partial \log p}{\partial \theta_k}\bigg|_{\theta_0}\right] = 0$$

The last equality uses the zero-mean property of the score when evaluated at $\theta_0$, since $p_0 = p(\cdot|\theta_0)$.

**Second derivative:**

$$\frac{\partial^2 g}{\partial \theta_j \partial \theta_k}\bigg|_{\theta_0} = \mathbb{E}_{p_0}\!\left[\frac{\partial^2 \log p}{\partial \theta_j \partial \theta_k}\bigg|_{\theta_0}\right] = -I_{jk}(\theta_0)$$

This is the negative Fisher information matrix entry (using the second-form identity).

### Result

$$\boxed{D_{\text{KL}}(p_{\theta_0} \| p_\theta) \approx \frac{1}{2}(\theta - \theta_0)^T\, \mathbf{I}(\theta_0)\, (\theta - \theta_0)}$$

The **first-order term vanishes** (KL divergence has a minimum at $\theta = \theta_0$), and the **second-order term is the Fisher information quadratic form**.

### Interpretation

Locally, KL divergence behaves like a **Mahalanobis distance** with the Fisher information matrix as the metric tensor:

$$D_{\text{KL}}(p_{\theta_0} \| p_\theta) \approx \frac{1}{2}\|\theta - \theta_0\|_{\mathbf{I}(\theta_0)}^2$$

This means that moving 1 unit in a direction where the Fisher information is large (the log-likelihood is sharply curved) incurs a large KL cost—the distribution changes substantially. Moving 1 unit in a direction where the Fisher information is small (the log-likelihood is flat) incurs a small KL cost—the distribution barely changes.

## The Fisher–Rao Metric

The Fisher information matrix defines a **Riemannian metric** on the space of probability distributions, called the **Fisher–Rao metric** (or information metric). The infinitesimal distance element is:

$$ds^2 = d\theta^T \mathbf{I}(\theta)\, d\theta$$

This gives the space of distributions a geometric structure where:

- **Geodesics** are the shortest paths between distributions
- **Distance** accounts for the intrinsic curvature of the distribution family
- **Volume elements** reflect the local distinguishability of nearby distributions

The Fisher–Rao metric is unique (up to scaling) among Riemannian metrics on statistical manifolds that are invariant under sufficient statistics—a deep result known as **Čencov's theorem**.

## Example: Gaussian Family

For $p(x|\mu, \sigma^2) = \mathcal{N}(\mu, \sigma^2)$ with $\theta = (\mu, \sigma^2)$:

$$\mathbf{I}(\mu, \sigma^2) = \begin{pmatrix} \frac{1}{\sigma^2} & 0 \\ 0 & \frac{1}{2\sigma^4} \end{pmatrix}$$

**Derivation.** The log-density is $\log p = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$.

The score components are:

$$s_\mu = \frac{\partial \log p}{\partial \mu} = \frac{x - \mu}{\sigma^2}, \qquad s_{\sigma^2} = \frac{\partial \log p}{\partial \sigma^2} = -\frac{1}{2\sigma^2} + \frac{(x-\mu)^2}{2\sigma^4}$$

Computing $\mathbb{E}[s_\mu^2] = 1/\sigma^2$, $\mathbb{E}[s_{\sigma^2}^2] = 1/(2\sigma^4)$, and $\mathbb{E}[s_\mu s_{\sigma^2}] = 0$ gives the diagonal matrix above.

**Interpretation.** The Fisher information is higher for smaller $\sigma^2$: a narrow Gaussian changes more (in KL sense) per unit parameter shift than a wide one. The mean direction has Fisher information $1/\sigma^2$ and the variance direction has $1/(2\sigma^4)$.

### Numerical Verification

```python
import torch
import numpy as np

def kl_gaussian_1d(mu1, s1, mu2, s2):
    """Analytical KL(N(mu1,s1^2) || N(mu2,s2^2))."""
    return np.log(s2/s1) + (s1**2 + (mu1-mu2)**2)/(2*s2**2) - 0.5

# Verify quadratic approximation near theta_0
mu0, sigma0 = 2.0, 1.5

# Fisher information at theta_0
I_mu = 1.0 / sigma0**2
I_sigma2 = 1.0 / (2 * sigma0**4)

deltas = np.linspace(-0.5, 0.5, 100)

# KL for perturbing mu
kl_exact_mu = [kl_gaussian_1d(mu0, sigma0, mu0 + d, sigma0) for d in deltas]
kl_approx_mu = [0.5 * I_mu * d**2 for d in deltas]

# KL for perturbing sigma^2
sigma0_sq = sigma0**2
kl_exact_s = [kl_gaussian_1d(mu0, sigma0, mu0, np.sqrt(sigma0_sq + d)) for d in deltas]
kl_approx_s = [0.5 * I_sigma2 * d**2 for d in deltas]

print("Mean perturbation (delta=0.1):")
print(f"  Exact KL:  {kl_gaussian_1d(mu0, sigma0, mu0+0.1, sigma0):.6f}")
print(f"  Approx KL: {0.5 * I_mu * 0.1**2:.6f}")

print("\nVariance perturbation (delta=0.1):")
print(f"  Exact KL:  {kl_gaussian_1d(mu0, sigma0, mu0, np.sqrt(sigma0_sq+0.1)):.6f}")
print(f"  Approx KL: {0.5 * I_sigma2 * 0.1**2:.6f}")
```

## Applications in Deep Learning

### Natural Gradient Descent

Standard gradient descent updates $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$, which uses the Euclidean metric on parameter space. But parameter space geometry is arbitrary—a small Euclidean step might cause a large change in the distribution (or vice versa).

**Natural gradient descent** replaces the Euclidean metric with the Fisher–Rao metric:

$$\theta \leftarrow \theta - \eta\, \mathbf{I}(\theta)^{-1} \nabla_\theta \mathcal{L}$$

The natural gradient $\tilde{\nabla} = \mathbf{I}^{-1}\nabla$ is the steepest descent direction in distribution space: it finds the direction that decreases the loss the most per unit KL change in the distribution.

```python
def natural_gradient_step(params, grad, fisher_inv, lr=0.01):
    """Natural gradient descent step.

    Args:
        params: Current parameters.
        grad: Euclidean gradient of the loss.
        fisher_inv: Inverse Fisher information matrix.
        lr: Learning rate.

    Returns:
        Updated parameters.
    """
    natural_grad = fisher_inv @ grad
    return params - lr * natural_grad
```

### Trust Region Policy Optimization (TRPO)

TRPO constrains policy updates to a KL divergence trust region:

$$\max_\theta \; \mathbb{E}\!\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s, a)\right] \quad\text{s.t.}\quad \mathbb{E}\!\left[D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))\right] \leq \delta$$

The Fisher information approximation converts this constrained optimization into a natural gradient step with an adaptive step size. The constraint ensures that each policy update changes the behavior by at most $\delta$ nats, preventing catastrophic policy collapse.

### Proximal Policy Optimization (PPO)

PPO approximates TRPO's KL constraint with a clipped surrogate objective, avoiding the expensive Fisher matrix computation while achieving similar trust region behavior.

### Variational Inference

In variational inference, we minimize $D_{\text{KL}}(q_\phi \| p)$ over variational parameters $\phi$. The Fisher–Rao geometry of $q_\phi$ determines the natural gradient for this optimization. Natural gradient variational inference converges faster than standard gradient methods because it accounts for the curvature of the variational family.

## Connection to the Cramér–Rao Bound

The Fisher information also bounds the variance of any unbiased estimator $\hat{\theta}$:

$$\text{Var}(\hat{\theta}) \geq \mathbf{I}(\theta)^{-1}$$

This is the **Cramér–Rao lower bound**. The connection to KL divergence is intuitive: if nearby distributions are easily distinguishable (high Fisher information, meaning a small parameter change produces a large KL divergence), then the parameter can be estimated precisely (low variance).

## Key Takeaways

The Taylor expansion of KL divergence reveals the Fisher information matrix as the local metric tensor on the space of distributions. The first-order term vanishes (KL has a minimum at identity), and the second-order term gives $D_{\text{KL}} \approx \frac{1}{2}\delta\theta^T \mathbf{I} \,\delta\theta$. This local quadratic structure defines the Fisher–Rao Riemannian metric, which is the unique invariant metric on statistical manifolds (Čencov's theorem). Practical applications include natural gradient descent (which uses $\mathbf{I}^{-1}\nabla$ to account for distribution geometry), TRPO (which constrains policy updates via KL trust regions), and variational inference (where the Fisher geometry accelerates optimization over the variational family).
