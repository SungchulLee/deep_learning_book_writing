# Evidence Lower Bound (ELBO)

## Learning Objectives

By the end of this section, you will be able to:

1. Derive the ELBO from first principles using multiple approaches
2. Understand the decomposition of ELBO into reconstruction and regularization terms
3. Compute ELBO for various probabilistic models
4. Implement ELBO-based optimization in PyTorch
5. Interpret ELBO as a tool for model comparison

## The ELBO: Heart of Variational Inference

The Evidence Lower Bound (ELBO) is the central quantity in variational inference. It provides:

1. A tractable optimization objective
2. A lower bound on the log marginal likelihood
3. A measure of approximation quality

## Complete Mathematical Derivation

### Derivation 1: From KL Divergence

**Starting point**: We want to minimize $\text{KL}(q(\theta) \| p(\theta | \mathcal{D}))$ but cannot compute the posterior directly.

**Step 1**: Expand the KL divergence

$$
\text{KL}(q(\theta) \| p(\theta | \mathcal{D})) = \int q(\theta) \log \frac{q(\theta)}{p(\theta | \mathcal{D})} \, d\theta
$$

**Step 2**: Apply Bayes' rule to the posterior

$$
p(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})}
$$

Therefore:

$$
\log p(\theta | \mathcal{D}) = \log p(\mathcal{D} | \theta) + \log p(\theta) - \log p(\mathcal{D})
$$

**Step 3**: Substitute into KL divergence

$$
\begin{aligned}
\text{KL}(q \| p) &= \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(\theta | \mathcal{D})] \\
&= \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(\mathcal{D} | \theta)] - \mathbb{E}_q[\log p(\theta)] + \log p(\mathcal{D})
\end{aligned}
$$

Note that $\log p(\mathcal{D})$ does not depend on $\theta$, so $\mathbb{E}_q[\log p(\mathcal{D})] = \log p(\mathcal{D})$.

**Step 4**: Rearrange to isolate $\log p(\mathcal{D})$

$$
\log p(\mathcal{D}) = \text{KL}(q \| p) + \underbrace{\mathbb{E}_q[\log p(\mathcal{D} | \theta)] + \mathbb{E}_q[\log p(\theta)] - \mathbb{E}_q[\log q(\theta)]}_{\text{ELBO}(q)}
$$

**Step 5**: Define the ELBO

$$
\boxed{\text{ELBO}(q) = \mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \mathbb{E}_q[\log q(\theta)]}
$$

where $p(\mathcal{D}, \theta) = p(\mathcal{D} | \theta) p(\theta)$ is the joint distribution.

### Derivation 2: Jensen's Inequality

An alternative derivation uses Jensen's inequality directly:

$$
\begin{aligned}
\log p(\mathcal{D}) &= \log \int p(\mathcal{D}, \theta) \, d\theta \\
&= \log \int q(\theta) \frac{p(\mathcal{D}, \theta)}{q(\theta)} \, d\theta \\
&= \log \mathbb{E}_q\left[\frac{p(\mathcal{D}, \theta)}{q(\theta)}\right] \\
&\geq \mathbb{E}_q\left[\log \frac{p(\mathcal{D}, \theta)}{q(\theta)}\right] \quad \text{(Jensen's inequality)} \\
&= \text{ELBO}(q)
\end{aligned}
$$

The inequality follows because $\log$ is concave.

### Derivation 3: Importance Sampling Perspective

From an importance sampling viewpoint:

$$
p(\mathcal{D}) = \int p(\mathcal{D}, \theta) \, d\theta = \int q(\theta) \frac{p(\mathcal{D}, \theta)}{q(\theta)} \, d\theta = \mathbb{E}_q\left[\frac{p(\mathcal{D}, \theta)}{q(\theta)}\right]
$$

The ELBO is the log of a lower bound on this expectation:

$$
\log p(\mathcal{D}) = \log \mathbb{E}_q\left[\frac{p(\mathcal{D}, \theta)}{q(\theta)}\right] \geq \mathbb{E}_q\left[\log \frac{p(\mathcal{D}, \theta)}{q(\theta)}\right] = \text{ELBO}(q)
$$

## Alternative Formulations of ELBO

The ELBO can be written in several equivalent forms, each providing different insights.

### Formulation 1: Joint-Based

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \mathbb{E}_q[\log q(\theta)]
$$

**Interpretation**: Expected log joint minus entropy of $q$.

### Formulation 2: Reconstruction + Regularization

$$
\text{ELBO}(q) = \underbrace{\mathbb{E}_q[\log p(\mathcal{D} | \theta)]}_{\text{Reconstruction}} - \underbrace{\text{KL}(q(\theta) \| p(\theta))}_{\text{Regularization}}
$$

**Interpretation**:

- **Reconstruction term**: How well does the model explain the data?
- **Regularization term**: How close is $q$ to the prior?

This formulation shows that VI balances data fit against staying close to prior beliefs.

### Formulation 3: Entropy-Based

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(\mathcal{D} | \theta)] + \mathbb{E}_q[\log p(\theta)] + H[q]
$$

where $H[q] = -\mathbb{E}_q[\log q(\theta)]$ is the entropy of $q$.

**Interpretation**: The entropy term encourages uncertainty in $q$, preventing collapse to a point mass.

### Formulation 4: Negative Free Energy

In physics literature, the ELBO is called the **negative variational free energy**:

$$
\mathcal{F}(q) = -\text{ELBO}(q) = \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(\mathcal{D}, \theta)]
$$

Minimizing free energy is equivalent to maximizing ELBO.

## The Fundamental Identity

The relationship between log evidence, ELBO, and KL divergence is:

$$
\boxed{\log p(\mathcal{D}) = \text{ELBO}(q) + \text{KL}(q(\theta) \| p(\theta | \mathcal{D}))}
$$

This identity reveals several important properties:

1. **Lower bound**: Since $\text{KL} \geq 0$, we have $\log p(\mathcal{D}) \geq \text{ELBO}(q)$
2. **Gap interpretation**: The gap $\log p(\mathcal{D}) - \text{ELBO}(q)$ equals the KL divergence
3. **Tightness**: When $q = p(\theta | \mathcal{D})$, the bound is tight: $\text{ELBO}(q) = \log p(\mathcal{D})$

## ELBO for Specific Models

### Gaussian Model with Known Variance

**Model**:
- Prior: $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$
- Likelihood: $x_i | \theta \sim \mathcal{N}(\theta, \sigma^2)$ for $i = 1, \ldots, n$
- Variational family: $q(\theta) = \mathcal{N}(m, s^2)$

**ELBO Derivation**:

$$
\text{ELBO}(m, s) = \mathbb{E}_q[\log p(\mathcal{D} | \theta)] + \mathbb{E}_q[\log p(\theta)] - \mathbb{E}_q[\log q(\theta)]
$$

**Term 1**: Expected log-likelihood

$$
\begin{aligned}
\mathbb{E}_q[\log p(\mathcal{D} | \theta)] &= \mathbb{E}_q\left[-\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \theta)^2\right] \\
&= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n \mathbb{E}_q[(x_i - \theta)^2] \\
&= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\left[\sum_{i=1}^n (x_i - m)^2 + ns^2\right]
\end{aligned}
$$

where we used $\mathbb{E}_q[(x_i - \theta)^2] = (x_i - m)^2 + s^2$.

**Term 2**: Expected log-prior

$$
\mathbb{E}_q[\log p(\theta)] = -\frac{1}{2}\log(2\pi\sigma_0^2) - \frac{1}{2\sigma_0^2}\left[(m - \mu_0)^2 + s^2\right]
$$

**Term 3**: Negative entropy

$$
-\mathbb{E}_q[\log q(\theta)] = \frac{1}{2}\log(2\pi e s^2)
$$

**Complete ELBO**:

$$
\text{ELBO}(m, s) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\left[\sum_{i=1}^n (x_i - m)^2 + ns^2\right] - \frac{1}{2}\log(2\pi\sigma_0^2) - \frac{(m-\mu_0)^2 + s^2}{2\sigma_0^2} + \frac{1}{2}\log(2\pi e s^2)
$$

### Latent Variable Models

For models with latent variables $z$ and parameters $\theta$:

$$
p(\mathcal{D}) = \int \int p(\mathcal{D}, z, \theta) \, dz \, d\theta
$$

The ELBO becomes:

$$
\text{ELBO}(q) = \mathbb{E}_{q(z, \theta)}[\log p(\mathcal{D}, z, \theta)] - \mathbb{E}_{q(z, \theta)}[\log q(z, \theta)]
$$

## PyTorch Implementation

### ELBO Computation and Optimization

```python
import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Tuple, Dict
import matplotlib.pyplot as plt

class GaussianELBO:
    """
    ELBO computation for Gaussian mean estimation.
    
    Demonstrates all formulations of ELBO.
    """
    
    def __init__(self, prior_mean: float, prior_std: float, 
                 likelihood_std: float):
        self.mu_0 = prior_mean
        self.sigma_0 = prior_std
        self.sigma = likelihood_std
    
    def elbo_joint_form(self, data: torch.Tensor, m: torch.Tensor, 
                        s: torch.Tensor) -> torch.Tensor:
        """
        ELBO = E_q[log p(D,θ)] - E_q[log q(θ)]
        """
        n = len(data)
        
        # E_q[log p(D,θ)] = E_q[log p(D|θ)] + E_q[log p(θ)]
        # = E_q[log p(D|θ) + log p(θ)]
        
        # Expected log joint
        # For each data point: log p(x_i|θ) = -0.5*log(2πσ²) - (x_i-θ)²/(2σ²)
        # E_q[(x_i-θ)²] = (x_i-m)² + s²
        
        expected_log_likelihood = (
            -0.5 * n * torch.log(torch.tensor(2 * torch.pi * self.sigma**2))
            - 0.5 / self.sigma**2 * (
                torch.sum((data - m)**2) + n * s**2
            )
        )
        
        expected_log_prior = (
            -0.5 * torch.log(torch.tensor(2 * torch.pi * self.sigma_0**2))
            - 0.5 / self.sigma_0**2 * ((m - self.mu_0)**2 + s**2)
        )
        
        expected_log_joint = expected_log_likelihood + expected_log_prior
        
        # E_q[log q(θ)] = -H[q] = -0.5*log(2πes²)
        negative_entropy = -0.5 * torch.log(2 * torch.pi * torch.e * s**2)
        
        return expected_log_joint - negative_entropy
    
    def elbo_reconstruction_regularization(self, data: torch.Tensor, 
                                           m: torch.Tensor, 
                                           s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ELBO = E_q[log p(D|θ)] - KL(q(θ) || p(θ))
        
        Returns: (elbo, reconstruction_term, kl_term)
        """
        n = len(data)
        
        # Reconstruction term: E_q[log p(D|θ)]
        reconstruction = (
            -0.5 * n * torch.log(torch.tensor(2 * torch.pi * self.sigma**2))
            - 0.5 / self.sigma**2 * (
                torch.sum((data - m)**2) + n * s**2
            )
        )
        
        # KL divergence: KL(N(m,s²) || N(μ₀,σ₀²))
        # = log(σ₀/s) + (s² + (m-μ₀)²)/(2σ₀²) - 0.5
        kl = (
            torch.log(torch.tensor(self.sigma_0) / s)
            + (s**2 + (m - self.mu_0)**2) / (2 * self.sigma_0**2)
            - 0.5
        )
        
        elbo = reconstruction - kl
        
        return elbo, reconstruction, kl
    
    def gradient_elbo(self, data: torch.Tensor, m: torch.Tensor, 
                      s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients of ELBO analytically.
        
        ∂ELBO/∂m = Σᵢ(xᵢ - m)/σ² - (m - μ₀)/σ₀²
        ∂ELBO/∂s = -ns/σ² - s/σ₀² + 1/s
        """
        n = len(data)
        
        grad_m = (
            torch.sum(data - m) / self.sigma**2 
            - (m - self.mu_0) / self.sigma_0**2
        )
        
        grad_s = -n * s / self.sigma**2 - s / self.sigma_0**2 + 1 / s
        
        return grad_m, grad_s


def optimize_elbo_gradient_ascent(data: torch.Tensor, 
                                  elbo_computer: GaussianELBO,
                                  n_iterations: int = 500,
                                  learning_rate: float = 0.05) -> Dict:
    """
    Optimize ELBO using gradient ascent with analytical gradients.
    """
    # Initialize variational parameters
    m = torch.tensor(0.0)
    s = torch.tensor(1.0)
    
    history = {
        'elbo': [], 'm': [], 's': [],
        'reconstruction': [], 'kl': []
    }
    
    for i in range(n_iterations):
        # Compute ELBO and components
        elbo, recon, kl = elbo_computer.elbo_reconstruction_regularization(
            data, m, s
        )
        
        # Compute gradients
        grad_m, grad_s = elbo_computer.gradient_elbo(data, m, s)
        
        # Gradient ascent update
        m = m + learning_rate * grad_m
        s = s + learning_rate * grad_s
        s = torch.clamp(s, min=0.01)  # Ensure positive
        
        # Record history
        history['elbo'].append(elbo.item())
        history['m'].append(m.item())
        history['s'].append(s.item())
        history['reconstruction'].append(recon.item())
        history['kl'].append(kl.item())
    
    return history, m, s


def optimize_elbo_autograd(data: torch.Tensor,
                          elbo_computer: GaussianELBO,
                          n_iterations: int = 500,
                          learning_rate: float = 0.05) -> Dict:
    """
    Optimize ELBO using PyTorch autograd.
    """
    # Initialize variational parameters with gradients
    m = torch.tensor([0.0], requires_grad=True)
    log_s = torch.tensor([0.0], requires_grad=True)  # log-scale for positivity
    
    optimizer = torch.optim.Adam([m, log_s], lr=learning_rate)
    
    history = {'elbo': [], 'm': [], 's': []}
    
    for i in range(n_iterations):
        optimizer.zero_grad()
        
        s = torch.exp(log_s)
        elbo = elbo_computer.elbo_joint_form(data, m, s)
        
        loss = -elbo  # Minimize negative ELBO
        loss.backward()
        optimizer.step()
        
        history['elbo'].append(elbo.item())
        history['m'].append(m.item())
        history['s'].append(s.item())
    
    return history, m.detach(), torch.exp(log_s).detach()


# Visualization
def visualize_elbo_optimization(history: Dict, exact_mean: float, 
                                exact_std: float, data: torch.Tensor):
    """Comprehensive visualization of ELBO optimization."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: ELBO convergence
    ax = axes[0, 0]
    ax.plot(history['elbo'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('(a) ELBO Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: ELBO decomposition (if available)
    ax = axes[0, 1]
    if 'reconstruction' in history:
        ax.plot(history['reconstruction'], 'g-', linewidth=2, 
                label='Reconstruction')
        ax.plot([-x for x in history['kl']], 'r-', linewidth=2, 
                label='-KL Divergence')
        ax.plot(history['elbo'], 'b--', linewidth=2, label='ELBO')
        ax.legend()
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('(b) ELBO Decomposition', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Parameter trajectory
    ax = axes[0, 2]
    ax.plot(history['m'], history['s'], 'b-', alpha=0.5, linewidth=1)
    ax.plot(history['m'][0], history['s'][0], 'go', markersize=12, 
            label='Start', zorder=5)
    ax.plot(history['m'][-1], history['s'][-1], 'ro', markersize=12, 
            label='End', zorder=5)
    ax.plot(exact_mean, exact_std, 'k*', markersize=15, label='Exact', zorder=5)
    ax.set_xlabel('Mean (m)', fontsize=11)
    ax.set_ylabel('Std Dev (s)', fontsize=11)
    ax.set_title('(c) Parameter Trajectory', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Mean convergence
    ax = axes[1, 0]
    ax.plot(history['m'], 'b-', linewidth=2, label='VI mean')
    ax.axhline(exact_mean, color='r', linestyle='--', linewidth=2, 
               label='Exact mean')
    ax.axhline(data.mean().item(), color='g', linestyle=':', linewidth=2,
               label='Sample mean')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Mean', fontsize=11)
    ax.set_title('(d) Mean Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Std convergence
    ax = axes[1, 1]
    ax.plot(history['s'], 'b-', linewidth=2, label='VI std')
    ax.axhline(exact_std, color='r', linestyle='--', linewidth=2, 
               label='Exact std')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Std Dev', fontsize=11)
    ax.set_title('(e) Std Dev Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Final posterior comparison
    ax = axes[1, 2]
    theta_range = torch.linspace(
        min(history['m'][-1] - 3*history['s'][-1], exact_mean - 3*exact_std),
        max(history['m'][-1] + 3*history['s'][-1], exact_mean + 3*exact_std),
        200
    )
    
    vi_pdf = dist.Normal(history['m'][-1], history['s'][-1]).log_prob(theta_range).exp()
    exact_pdf = dist.Normal(exact_mean, exact_std).log_prob(theta_range).exp()
    
    ax.plot(theta_range.numpy(), exact_pdf.numpy(), 'b-', linewidth=2.5, 
            label='Exact Posterior')
    ax.plot(theta_range.numpy(), vi_pdf.numpy(), 'r--', linewidth=2.5, 
            label='VI Approximation')
    ax.fill_between(theta_range.numpy(), exact_pdf.numpy(), alpha=0.2, color='blue')
    ax.set_xlabel('θ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(f) Posterior Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('elbo_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Generate data
    true_mean = 3.0
    n_samples = 100
    sigma = 2.0
    data = torch.randn(n_samples) * sigma + true_mean
    
    print("=" * 60)
    print("ELBO Computation and Optimization")
    print("=" * 60)
    
    # Setup
    elbo_computer = GaussianELBO(
        prior_mean=0.0, prior_std=5.0, likelihood_std=sigma
    )
    
    # Compute exact posterior
    precision_0 = 1 / 5.0**2
    precision_data = n_samples / sigma**2
    precision_n = precision_0 + precision_data
    exact_mean = (0.0 * precision_0 + data.mean().item() * precision_data) / precision_n
    exact_std = 1 / (precision_n ** 0.5)
    
    print(f"\nTrue mean: {true_mean}")
    print(f"Sample mean: {data.mean().item():.4f}")
    print(f"Exact posterior: N({exact_mean:.4f}, {exact_std:.4f}²)")
    
    # Optimize with analytical gradients
    print("\n--- Gradient Ascent (Analytical) ---")
    history, m_final, s_final = optimize_elbo_gradient_ascent(
        data, elbo_computer, n_iterations=200
    )
    print(f"Final VI posterior: N({m_final.item():.4f}, {s_final.item():.4f}²)")
    
    # Visualize
    visualize_elbo_optimization(history, exact_mean, exact_std, data)
```

## ELBO as Model Selection Criterion

The ELBO can be used for model comparison, similar to AIC or BIC:

$$
\log p(\mathcal{D} | \mathcal{M}) \approx \text{ELBO}(q^*)
$$

where $q^*$ is the optimized variational distribution for model $\mathcal{M}$.

### Comparison with Other Criteria

| Criterion | Formula | Properties |
|-----------|---------|------------|
| ELBO | $\mathbb{E}_q[\log p(\mathcal{D},\theta)] - \mathbb{E}_q[\log q]$ | Lower bound on evidence |
| BIC | $\log p(\mathcal{D}\|\hat{\theta}) - \frac{k}{2}\log n$ | Penalizes complexity |
| AIC | $\log p(\mathcal{D}\|\hat{\theta}) - k$ | Less penalty than BIC |
| WAIC | Bayesian generalization of AIC | Uses posterior |

## Summary

The ELBO is defined as:

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \mathbb{E}_q[\log q(\theta)]
$$

**Key properties:**

1. $\log p(\mathcal{D}) = \text{ELBO}(q) + \text{KL}(q \| p)$
2. $\log p(\mathcal{D}) \geq \text{ELBO}(q)$ (lower bound)
3. $\max_q \text{ELBO}(q) \Leftrightarrow \min_q \text{KL}(q \| p)$

**Alternative formulations:**

- Reconstruction - KL: $\mathbb{E}_q[\log p(\mathcal{D}|\theta)] - \text{KL}(q \| p(\theta))$
- Entropy form: $\mathbb{E}_q[\log p(\mathcal{D}|\theta)] + \mathbb{E}_q[\log p(\theta)] + H[q]$

## Exercises

### Exercise 1: ELBO for Beta-Binomial

Derive the ELBO for:
- Prior: $\theta \sim \text{Beta}(\alpha_0, \beta_0)$
- Likelihood: $x | \theta \sim \text{Binomial}(n, \theta)$
- Variational: $q(\theta) = \text{Beta}(\alpha, \beta)$

### Exercise 2: Tightness of the Bound

For the Gaussian model, verify numerically that the ELBO equals log evidence when $q = p(\theta | \mathcal{D})$.

### Exercise 3: ELBO Surface Visualization

Create a 3D surface plot of the ELBO as a function of variational parameters $(m, s)$, showing the optimization landscape.

## References

1. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). "Variational Inference: A Review for Statisticians."

2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 10.

3. Hoffman, M. D., & Johnson, M. J. (2016). "ELBO Surgery: Yet Another Way to Carve up the Variational Evidence Lower Bound."

4. Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes."

5. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). "Stochastic Backpropagation and Approximate Inference in Deep Generative Models."
