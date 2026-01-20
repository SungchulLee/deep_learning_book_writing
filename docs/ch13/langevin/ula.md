# Unadjusted Langevin Algorithm (ULA)

## Learning Objectives

By the end of this section, you will be able to:

- Derive ULA from Euler-Maruyama discretization of the Langevin SDE
- Understand the sources and magnitude of discretization bias
- Analyze the trade-off between step size and accuracy
- Implement ULA efficiently in PyTorch
- Diagnose when ULA's bias is acceptable for practical applications

## From Continuous to Discrete Time

### Euler-Maruyama Discretization

The Langevin SDE is:

$$
dx_t = s(x_t) \, dt + \sqrt{2} \, dW_t
$$

where $s(x) = \nabla_x \log \pi(x)$ is the score function.

The simplest discretization scheme is **Euler-Maruyama**: approximate the continuous dynamics by taking discrete steps of size $\epsilon$:

$$
x_{t+\epsilon} \approx x_t + s(x_t) \cdot \epsilon + \sqrt{2\epsilon} \cdot \xi_t
$$

where $\xi_t \sim \mathcal{N}(0, I)$ is standard Gaussian noise.

!!! note "Scaling of the Noise Term"
    The noise term scales as $\sqrt{\epsilon}$ (not $\epsilon$) because Brownian motion increments have variance proportional to the time step: $\text{Var}(W_{t+\epsilon} - W_t) = \epsilon$.

### The ULA Update Rule

**Unadjusted Langevin Algorithm (ULA)**, also called the Langevin Monte Carlo (LMC) algorithm:

$$
\boxed{x_{k+1} = x_k + \epsilon \cdot s(x_k) + \sqrt{2\epsilon} \cdot \xi_k, \quad \xi_k \sim \mathcal{N}(0, I)}
$$

where $k$ indexes discrete time steps.

**Algorithm**:

```
Input: Score function s(x), step size ε, number of steps K, initial point x₀
Output: Samples x₁, x₂, ..., xₖ

for k = 0 to K-1:
    ξ ~ N(0, I)                           # Sample noise
    x_{k+1} = x_k + ε·s(x_k) + √(2ε)·ξ   # ULA update
    
return {x_k}
```

### Comparison with Gradient Descent

ULA resembles **stochastic gradient ascent** on $\log \pi(x)$:

| Method | Update Rule | Goal |
|--------|-------------|------|
| Gradient Ascent | $x_{k+1} = x_k + \epsilon \cdot s(x_k)$ | Find mode of $\pi$ |
| ULA | $x_{k+1} = x_k + \epsilon \cdot s(x_k) + \sqrt{2\epsilon} \cdot \xi_k$ | Sample from $\pi$ |

The added noise transforms optimization into sampling. The magnitude $\sqrt{2\epsilon}$ is precisely calibrated—too little noise and you converge to a mode; too much and you don't converge at all.

## Understanding Discretization Bias

### ULA Does Not Sample from π Exactly

**Critical insight**: ULA does not have $\pi(x)$ as its stationary distribution. The discrete Markov chain converges to a different distribution $\tilde{\pi}_\epsilon(x)$ that depends on the step size $\epsilon$.

The difference $\|\tilde{\pi}_\epsilon - \pi\|$ is the **discretization bias**.

### Sources of Bias

The bias arises from two sources:

**1. Drift approximation error**: We use $s(x_k)$ instead of the "average" score along the trajectory:

$$
\int_0^\epsilon s(x_{t+s}) ds \neq \epsilon \cdot s(x_k)
$$

**2. Curved probability landscape**: In curved regions of $\log \pi$, the linear approximation overshoots or undershoots.

### Quantifying the Bias

For well-behaved targets (strongly log-concave with Lipschitz score), the bias scales as:

$$
D_{KL}(\tilde{\pi}_\epsilon \| \pi) = \mathcal{O}(\epsilon)
$$

or in total variation distance:

$$
\|\tilde{\pi}_\epsilon - \pi\|_{TV} = \mathcal{O}(\sqrt{\epsilon})
$$

**Key implication**: To reduce bias by half, you need to reduce $\epsilon$ by a factor of 4.

### Visualizing the Bias

For a 1D Gaussian $\mathcal{N}(0, 1)$, the ULA stationary distribution has:

- Slightly different variance (biased)
- Correct mean (unbiased due to symmetry)

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def demonstrate_ula_bias():
    """
    Demonstrate that ULA has discretization bias.
    """
    # Target: standard Gaussian
    def score(x):
        return -x
    
    # Run ULA with different step sizes
    epsilons = [0.01, 0.1, 0.5, 1.0]
    n_steps = 50000
    burn_in = 5000
    
    results = {}
    
    for eps in epsilons:
        x = torch.tensor(0.0)
        samples = []
        
        for k in range(n_steps):
            xi = torch.randn(1).item()
            x = x + eps * score(x) + np.sqrt(2 * eps) * xi
            
            if k >= burn_in:
                samples.append(x)
        
        samples = np.array(samples)
        results[eps] = {
            'samples': samples,
            'mean': samples.mean(),
            'std': samples.std(),
            'bias_var': samples.var() - 1.0  # True variance is 1
        }
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x_grid = np.linspace(-4, 4, 200)
    true_density = stats.norm.pdf(x_grid)
    
    for idx, eps in enumerate(epsilons):
        ax = axes[idx // 2, idx % 2]
        
        samples = results[eps]['samples']
        
        # Histogram of ULA samples
        ax.hist(samples, bins=50, density=True, alpha=0.7, 
                color='steelblue', label='ULA samples')
        
        # True density
        ax.plot(x_grid, true_density, 'r-', linewidth=2, label='True N(0,1)')
        
        # Statistics
        stats_text = f"Sample mean: {results[eps]['mean']:.4f}\n"
        stats_text += f"Sample std: {results[eps]['std']:.4f}\n"
        stats_text += f"Variance bias: {results[eps]['bias_var']:.4f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'ULA with ε = {eps}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-4, 4)
    
    plt.suptitle('ULA Discretization Bias: Larger ε → Larger Bias', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('ula_bias_demonstration.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: ula_bias_demonstration.png")
    
    # Summary
    print("\nULA Bias Analysis:")
    print("-" * 50)
    print(f"{'ε':>10} {'Mean':>12} {'Std':>12} {'Var Bias':>12}")
    print("-" * 50)
    for eps in epsilons:
        r = results[eps]
        print(f"{eps:>10.2f} {r['mean']:>12.4f} {r['std']:>12.4f} {r['bias_var']:>12.4f}")


if __name__ == "__main__":
    demonstrate_ula_bias()
```

## Convergence Analysis

### Mixing Time

The **mixing time** of ULA is the number of steps needed to get close to the stationary distribution. For a $d$-dimensional target:

**Strongly log-concave targets** (Hessian bounded below):

$$
T_{mix}(\delta) = \mathcal{O}\left(\frac{d}{\epsilon} \log \frac{1}{\delta}\right)
$$

But the bias is $\mathcal{O}(\epsilon)$, so there's a fundamental trade-off.

### The Bias-Variance Trade-off

| Small $\epsilon$ | Large $\epsilon$ |
|------------------|------------------|
| ✓ Low bias | ✗ High bias |
| ✗ Slow mixing | ✓ Fast mixing |
| ✗ High autocorrelation | ✓ Low autocorrelation |
| ✗ High variance estimators | May diverge |

The **optimal** $\epsilon$ depends on the specific target and the desired accuracy.

### Theoretical Optimal Step Size

For isotropic targets in $d$ dimensions:

$$
\epsilon_{opt} \sim d^{-1/3}
$$

This gives a mixing time of:

$$
T_{mix} \sim d^{4/3}
$$

which is worse than the $\mathcal{O}(d)$ of ideal continuous Langevin but still polynomial.

## PyTorch Implementation

### Basic ULA Sampler

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

class ULASampler:
    """
    Unadjusted Langevin Algorithm sampler.
    """
    
    def __init__(self, score_fn, dim, epsilon=0.1):
        """
        Args:
            score_fn: Function that computes s(x) = ∇log π(x)
            dim: Dimension of the state space
            epsilon: Step size
        """
        self.score_fn = score_fn
        self.dim = dim
        self.epsilon = epsilon
    
    def step(self, x):
        """
        One ULA update step.
        
        Args:
            x: Current position (batch_size, dim) or (dim,)
            
        Returns:
            x_new: New position
        """
        score = self.score_fn(x)
        noise = torch.randn_like(x)
        
        x_new = x + self.epsilon * score + np.sqrt(2 * self.epsilon) * noise
        
        return x_new
    
    def sample(self, x0, n_samples, burn_in=1000, thin=1):
        """
        Generate samples using ULA.
        
        Args:
            x0: Initial position
            n_samples: Number of samples to collect
            burn_in: Number of initial samples to discard
            thin: Keep every thin-th sample
            
        Returns:
            samples: Array of shape (n_samples, dim)
        """
        x = x0.clone()
        samples = []
        
        total_steps = burn_in + n_samples * thin
        
        for k in range(total_steps):
            x = self.step(x)
            
            if k >= burn_in and (k - burn_in) % thin == 0:
                samples.append(x.clone())
        
        return torch.stack(samples)
    
    def sample_chain(self, x0, n_steps):
        """
        Return the full chain (for diagnostics).
        
        Args:
            x0: Initial position
            n_steps: Number of steps
            
        Returns:
            chain: Array of shape (n_steps, dim)
        """
        x = x0.clone()
        chain = [x.clone()]
        
        for _ in range(n_steps - 1):
            x = self.step(x)
            chain.append(x.clone())
        
        return torch.stack(chain)


class ULAWithDiagnostics(ULASampler):
    """
    ULA sampler with diagnostic capabilities.
    """
    
    def __init__(self, score_fn, log_prob_fn, dim, epsilon=0.1):
        """
        Args:
            score_fn: Score function
            log_prob_fn: Log probability function (for diagnostics)
            dim: Dimension
            epsilon: Step size
        """
        super().__init__(score_fn, dim, epsilon)
        self.log_prob_fn = log_prob_fn
        
        # Diagnostics storage
        self.log_probs = []
        self.squared_jumps = []
    
    def step_with_diagnostics(self, x):
        """Step with diagnostic recording."""
        x_new = self.step(x)
        
        # Record log probability
        self.log_probs.append(self.log_prob_fn(x_new).item())
        
        # Record squared jump distance
        jump = x_new - x
        self.squared_jumps.append((jump ** 2).sum().item())
        
        return x_new
    
    def sample_with_diagnostics(self, x0, n_steps):
        """Sample and record diagnostics."""
        self.log_probs = []
        self.squared_jumps = []
        
        x = x0.clone()
        chain = [x.clone()]
        
        for _ in range(n_steps - 1):
            x = self.step_with_diagnostics(x)
            chain.append(x.clone())
        
        return torch.stack(chain)
    
    def get_diagnostics(self):
        """Return diagnostic summary."""
        log_probs = np.array(self.log_probs)
        jumps = np.array(self.squared_jumps)
        
        return {
            'mean_log_prob': log_probs.mean(),
            'std_log_prob': log_probs.std(),
            'mean_squared_jump': jumps.mean(),
            'effective_step': np.sqrt(jumps.mean())
        }
```

### Example: 2D Gaussian with Different Step Sizes

```python
def ula_step_size_comparison():
    """
    Compare ULA performance with different step sizes.
    """
    # Target: 2D Gaussian with correlation
    mu = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
    precision = torch.linalg.inv(cov)
    
    def score_fn(x):
        return -torch.matmul(x - mu, precision)
    
    def log_prob_fn(x):
        mvn = dist.MultivariateNormal(mu, cov)
        return mvn.log_prob(x)
    
    # Test different step sizes
    epsilons = [0.01, 0.1, 0.5, 1.5]
    n_steps = 10000
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Create reference contours
    x_grid = torch.linspace(-4, 4, 100)
    y_grid = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    
    mvn = dist.MultivariateNormal(mu, cov)
    Z = torch.exp(mvn.log_prob(points)).reshape(100, 100)
    
    for idx, eps in enumerate(epsilons):
        ax = axes[idx // 2, idx % 2]
        
        # Run ULA
        sampler = ULAWithDiagnostics(score_fn, log_prob_fn, dim=2, epsilon=eps)
        x0 = torch.tensor([3.0, 3.0])
        
        chain = sampler.sample_with_diagnostics(x0, n_steps)
        diagnostics = sampler.get_diagnostics()
        
        # Burn-in
        burn_in = n_steps // 5
        samples = chain[burn_in:].numpy()
        
        # Plot
        ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=10, 
                   alpha=0.5, colors='red')
        ax.scatter(samples[::10, 0], samples[::10, 1], alpha=0.3, s=5, c='blue')
        
        # First 100 steps of trajectory
        trajectory = chain[:100].numpy()
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'g-', alpha=0.5, linewidth=1)
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
        
        # Statistics
        sample_mean = samples.mean(axis=0)
        sample_cov = np.cov(samples.T)
        
        stats_text = f"ε = {eps}\n"
        stats_text += f"Mean: [{sample_mean[0]:.2f}, {sample_mean[1]:.2f}]\n"
        stats_text += f"True: [0.00, 0.00]\n"
        stats_text += f"Var(x₁): {sample_cov[0,0]:.3f} (true: 1.0)\n"
        stats_text += f"Cov: {sample_cov[0,1]:.3f} (true: 0.8)"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.set_title(f'ULA with ε = {eps}', fontsize=13, fontweight='bold')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ula_step_size_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: ula_step_size_comparison.png")


if __name__ == "__main__":
    ula_step_size_comparison()
```

### Example: ULA for Bayesian Inference

```python
def ula_bayesian_linear_regression():
    """
    ULA for Bayesian linear regression posterior inference.
    """
    torch.manual_seed(42)
    
    # Generate data
    n = 50
    d = 3  # Number of features (including intercept)
    
    X = torch.randn(n, d - 1)
    X = torch.cat([torch.ones(n, 1), X], dim=1)  # Add intercept
    
    true_beta = torch.tensor([1.0, -2.0, 0.5])
    sigma = 0.5
    
    y = X @ true_beta + sigma * torch.randn(n)
    
    # Prior: β ~ N(0, τ²I) with τ = 2
    tau = 2.0
    prior_precision = torch.eye(d) / (tau ** 2)
    
    # Posterior score function
    # log π(β|y) ∝ -1/(2σ²)||y - Xβ||² - 1/(2τ²)||β||²
    # Score: s(β) = X'(y - Xβ)/σ² - β/τ²
    
    def posterior_score(beta):
        residual = y - X @ beta
        likelihood_grad = X.T @ residual / (sigma ** 2)
        prior_grad = -beta / (tau ** 2)
        return likelihood_grad + prior_grad
    
    def posterior_log_prob(beta):
        residual = y - X @ beta
        log_likelihood = -0.5 * (residual ** 2).sum() / (sigma ** 2)
        log_prior = -0.5 * (beta ** 2).sum() / (tau ** 2)
        return log_likelihood + log_prior
    
    # Compute true posterior (conjugate case)
    # Posterior precision: Λ = X'X/σ² + I/τ²
    # Posterior mean: μ = Λ⁻¹(X'y/σ²)
    posterior_precision = X.T @ X / (sigma ** 2) + prior_precision
    posterior_cov = torch.linalg.inv(posterior_precision)
    posterior_mean = posterior_cov @ (X.T @ y / (sigma ** 2))
    
    print("True posterior mean:", posterior_mean.numpy())
    print("True posterior std:", torch.sqrt(torch.diag(posterior_cov)).numpy())
    
    # Run ULA
    sampler = ULASampler(posterior_score, dim=d, epsilon=0.01)
    
    x0 = torch.zeros(d)
    n_samples = 10000
    burn_in = 2000
    
    print("\nRunning ULA...")
    samples = sampler.sample(x0, n_samples, burn_in=burn_in)
    
    # Analyze results
    sample_mean = samples.mean(dim=0)
    sample_std = samples.std(dim=0)
    
    print("\nULA Results:")
    print("-" * 50)
    print(f"{'Parameter':>10} {'True':>10} {'ULA Mean':>10} {'ULA Std':>10}")
    print("-" * 50)
    for i in range(d):
        print(f"β_{i:>8} {posterior_mean[i]:>10.4f} {sample_mean[i]:>10.4f} {sample_std[i]:>10.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    param_names = ['$\\beta_0$ (intercept)', '$\\beta_1$', '$\\beta_2$']
    
    for i in range(3):
        ax = axes[i]
        
        # Histogram of samples
        ax.hist(samples[:, i].numpy(), bins=50, density=True, 
                alpha=0.7, color='steelblue', label='ULA samples')
        
        # True posterior
        x_range = torch.linspace(
            posterior_mean[i] - 4 * torch.sqrt(posterior_cov[i, i]),
            posterior_mean[i] + 4 * torch.sqrt(posterior_cov[i, i]),
            100
        )
        true_density = torch.exp(dist.Normal(
            posterior_mean[i], 
            torch.sqrt(posterior_cov[i, i])
        ).log_prob(x_range))
        
        ax.plot(x_range.numpy(), true_density.numpy(), 'r-', 
                linewidth=2, label='True posterior')
        
        ax.axvline(x=true_beta[i].item(), color='green', linestyle='--',
                   linewidth=2, label=f'True value = {true_beta[i]:.1f}')
        
        ax.set_xlabel(param_names[i], fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Posterior of {param_names[i]}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ula_bayesian_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: ula_bayesian_regression.png")


if __name__ == "__main__":
    ula_bayesian_linear_regression()
```

## When to Use ULA

### Advantages

- **Simplicity**: No acceptance/rejection step, just gradient + noise
- **Speed**: Each iteration is fast (one gradient evaluation)
- **Scalability**: Works well in high dimensions with appropriate $\epsilon$
- **Gradient-based**: Exploits geometry of the target

### Disadvantages

- **Bias**: Does not sample from $\pi$ exactly
- **Tuning**: Step size $\epsilon$ is critical
- **Stability**: Large $\epsilon$ can cause divergence
- **No correction**: Errors accumulate (unlike MALA)

### Practical Guidelines

| Scenario | Recommendation |
|----------|----------------|
| Quick exploration | Use ULA with moderate $\epsilon$ |
| High accuracy needed | Use MALA or HMC instead |
| Very high dimensions | ULA can be competitive |
| Gradients expensive | Consider ULA with larger $\epsilon$ + bias correction |
| Debugging | Start with ULA, then add MH correction |

### Bias Correction Techniques

If ULA's bias is problematic, consider:

1. **Metropolis correction** → MALA (next section)
2. **Higher-order integrators** → Reduced bias
3. **Extrapolation** → Combine samples from different $\epsilon$ values
4. **Tempering** → Use as a proposal in a tempered scheme

## Summary

| Aspect | ULA |
|--------|-----|
| **Update rule** | $x_{k+1} = x_k + \epsilon s(x_k) + \sqrt{2\epsilon} \xi_k$ |
| **Stationary distribution** | $\tilde{\pi}_\epsilon \neq \pi$ (biased) |
| **Bias** | $\mathcal{O}(\epsilon)$ |
| **Mixing time** | $\mathcal{O}(d/\epsilon)$ |
| **Optimal $\epsilon$** | $\mathcal{O}(d^{-1/3})$ |
| **Computational cost** | One score evaluation per step |

**Key insight**: ULA trades exactness for simplicity. For many applications, the bias is acceptable; when it isn't, the Metropolis correction (MALA) restores exactness.

## Exercises

### Exercise 1: Bias Analysis

For a 1D Gaussian $\mathcal{N}(0, \sigma^2)$, derive the stationary variance of ULA as a function of $\epsilon$ and $\sigma^2$. Verify your derivation numerically.

### Exercise 2: Step Size Selection

Implement an adaptive step size scheme for ULA based on monitoring the acceptance rate of a hypothetical MH correction (without actually accepting/rejecting).

### Exercise 3: High-Dimensional Scaling

Run ULA on $d$-dimensional isotropic Gaussians for $d = 2, 10, 50, 100$. Measure the mixing time and verify the theoretical scaling.

### Exercise 4: Stochastic Gradients

Implement ULA with stochastic gradients (minibatch estimation of the score). How does the additional noise affect the bias?

## References

1. Parisi, G. (1981). Correlation functions and computer simulations. *Nuclear Physics B*, 180(3), 378-384.

2. Dalalyan, A. S. (2017). Theoretical guarantees for approximate sampling from smooth and log-concave densities. *Journal of the Royal Statistical Society: Series B*, 79(3), 651-676.

3. Durmus, A., & Moulines, E. (2017). Nonasymptotic convergence analysis for the unadjusted Langevin algorithm. *The Annals of Applied Probability*, 27(3), 1551-1587.

4. Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
