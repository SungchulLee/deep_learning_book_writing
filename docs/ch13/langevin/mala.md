# Metropolis-Adjusted Langevin Algorithm (MALA)

## Learning Objectives

By the end of this section, you will be able to:

- Understand why ULA needs a Metropolis-Hastings correction
- Derive the MALA acceptance probability with asymmetric proposals
- Implement MALA correctly, accounting for the gradient in the proposal
- Compare MALA's efficiency with random-walk Metropolis and ULA
- Tune MALA for optimal performance (57% acceptance rate)

## From ULA to MALA: Correcting the Bias

### The Problem with ULA

Recall that ULA generates samples from a biased distribution $\tilde{\pi}_\epsilon$ rather than the true target $\pi$. The bias is $\mathcal{O}(\epsilon)$, which can be significant for practical step sizes.

**Solution**: Add a Metropolis-Hastings (MH) acceptance step to correct for the discretization error.

### The Key Insight

The ULA proposal can be viewed as a proposal distribution:

$$
q(x' | x) = \mathcal{N}\left(x' \,\Big|\, x + \epsilon \, s(x), 2\epsilon I\right)
$$

This is an **asymmetric** proposal because the drift $\epsilon \, s(x)$ depends on the current position $x$.

By applying the Metropolis-Hastings correction with this proposal, we obtain **MALA** (Metropolis-Adjusted Langevin Algorithm), which has $\pi$ as its exact stationary distribution.

## The MALA Algorithm

### Proposal Step

Given current state $x$, propose:

$$
x' = x + \epsilon \, s(x) + \sqrt{2\epsilon} \, \xi, \quad \xi \sim \mathcal{N}(0, I)
$$

This is exactly the ULA update.

### Acceptance Step

Accept $x'$ with probability:

$$
\alpha(x \to x') = \min\left(1, \frac{\pi(x') \, q(x | x')}{\pi(x) \, q(x' | x)}\right)
$$

Since $q$ is Gaussian:

$$
q(x' | x) = \frac{1}{(4\pi\epsilon)^{d/2}} \exp\left(-\frac{\|x' - x - \epsilon \, s(x)\|^2}{4\epsilon}\right)
$$

$$
q(x | x') = \frac{1}{(4\pi\epsilon)^{d/2}} \exp\left(-\frac{\|x - x' - \epsilon \, s(x')\|^2}{4\epsilon}\right)
$$

### The Log Acceptance Ratio

In practice, we compute the **log** acceptance ratio:

$$
\log \alpha = \underbrace{\log \pi(x') - \log \pi(x)}_{\text{target ratio}} + \underbrace{\log q(x | x') - \log q(x' | x)}_{\text{Hastings correction}}
$$

The Hastings correction simplifies to:

$$
\log q(x | x') - \log q(x' | x) = \frac{1}{4\epsilon}\left[\|x' - x - \epsilon s(x)\|^2 - \|x - x' - \epsilon s(x')\|^2\right]
$$

!!! warning "Don't Forget the Hastings Correction!"
    Because the proposal is **asymmetric** (the drift depends on the current position), omitting the Hastings correction gives incorrect results. This is a common implementation bug.

### Complete MALA Algorithm

```
Input: Log density log π(x), score s(x), step size ε, num iterations K, initial x₀
Output: Samples x₁, x₂, ..., xₖ

for k = 0 to K-1:
    # Propose
    ξ ~ N(0, I)
    x' = x_k + ε·s(x_k) + √(2ε)·ξ
    
    # Compute log acceptance ratio
    log_target_ratio = log π(x') - log π(x_k)
    
    # Forward proposal: q(x'|x)
    mean_forward = x_k + ε·s(x_k)
    log_q_forward = -||x' - mean_forward||² / (4ε)
    
    # Backward proposal: q(x|x')
    mean_backward = x' + ε·s(x')
    log_q_backward = -||x_k - mean_backward||² / (4ε)
    
    log_α = log_target_ratio + log_q_backward - log_q_forward
    
    # Accept/reject
    u ~ Uniform(0, 1)
    if log(u) < log_α:
        x_{k+1} = x'     # Accept
    else:
        x_{k+1} = x_k    # Reject
        
return {x_k}
```

## PyTorch Implementation

### Complete MALA Sampler

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Tuple, Optional

class MALASampler:
    """
    Metropolis-Adjusted Langevin Algorithm sampler.
    """
    
    def __init__(
        self, 
        log_prob_fn: Callable, 
        score_fn: Callable, 
        dim: int, 
        epsilon: float = 0.1
    ):
        """
        Args:
            log_prob_fn: Function that computes log π(x)
            score_fn: Function that computes s(x) = ∇log π(x)
            dim: Dimension of the state space
            epsilon: Step size
        """
        self.log_prob_fn = log_prob_fn
        self.score_fn = score_fn
        self.dim = dim
        self.epsilon = epsilon
        
        # Statistics
        self.n_accepted = 0
        self.n_proposed = 0
    
    def _log_proposal(self, x_to: torch.Tensor, x_from: torch.Tensor) -> torch.Tensor:
        """
        Compute log q(x_to | x_from).
        
        The proposal is N(x_from + ε·s(x_from), 2ε·I).
        """
        mean = x_from + self.epsilon * self.score_fn(x_from)
        diff = x_to - mean
        # log q = -||x_to - mean||² / (4ε) - (d/2)log(4πε)
        # We can ignore the constant since it cancels in the ratio
        return -torch.sum(diff ** 2) / (4 * self.epsilon)
    
    def step(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        One MALA update step.
        
        Args:
            x: Current position
            
        Returns:
            x_new: New position (may be same as x if rejected)
            accepted: Whether the proposal was accepted
        """
        # Current score and log prob
        score_x = self.score_fn(x)
        log_prob_x = self.log_prob_fn(x)
        
        # Propose
        noise = torch.randn_like(x)
        x_prop = x + self.epsilon * score_x + torch.sqrt(2 * torch.tensor(self.epsilon)) * noise
        
        # Log probability at proposed point
        log_prob_prop = self.log_prob_fn(x_prop)
        
        # Log proposal densities (Hastings correction)
        log_q_forward = self._log_proposal(x_prop, x)  # q(x'|x)
        log_q_backward = self._log_proposal(x, x_prop)  # q(x|x')
        
        # Log acceptance ratio
        log_alpha = (log_prob_prop - log_prob_x) + (log_q_backward - log_q_forward)
        
        # Accept/reject
        self.n_proposed += 1
        
        if torch.log(torch.rand(1)) < log_alpha:
            self.n_accepted += 1
            return x_prop, True
        else:
            return x.clone(), False
    
    def sample(
        self, 
        x0: torch.Tensor, 
        n_samples: int, 
        burn_in: int = 1000, 
        thin: int = 1,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Generate samples using MALA.
        
        Args:
            x0: Initial position
            n_samples: Number of samples to collect
            burn_in: Number of initial samples to discard
            thin: Keep every thin-th sample
            verbose: Print progress
            
        Returns:
            samples: Array of shape (n_samples, dim)
        """
        # Reset statistics
        self.n_accepted = 0
        self.n_proposed = 0
        
        x = x0.clone()
        samples = []
        
        total_steps = burn_in + n_samples * thin
        
        for k in range(total_steps):
            x, _ = self.step(x)
            
            if k >= burn_in and (k - burn_in) % thin == 0:
                samples.append(x.clone())
            
            if verbose and (k + 1) % 1000 == 0:
                print(f"Step {k+1}/{total_steps}, "
                      f"Acceptance rate: {self.acceptance_rate:.1%}")
        
        return torch.stack(samples)
    
    @property
    def acceptance_rate(self) -> float:
        """Current acceptance rate."""
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed
    
    def reset_statistics(self):
        """Reset acceptance statistics."""
        self.n_accepted = 0
        self.n_proposed = 0


def mala_example_2d_gaussian():
    """
    Example: MALA for 2D correlated Gaussian.
    """
    # Target
    mu = torch.tensor([1.0, -1.0])
    cov = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
    precision = torch.linalg.inv(cov)
    
    mvn = dist.MultivariateNormal(mu, cov)
    
    def log_prob_fn(x):
        return mvn.log_prob(x)
    
    def score_fn(x):
        return -torch.matmul(x - mu, precision)
    
    # Compare different step sizes
    epsilons = [0.1, 0.5, 1.0, 2.0]
    n_samples = 5000
    burn_in = 1000
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Reference contours
    x_grid = torch.linspace(-3, 5, 100)
    y_grid = torch.linspace(-5, 3, 100)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    Z = torch.exp(mvn.log_prob(points)).reshape(100, 100)
    
    for idx, eps in enumerate(epsilons):
        ax = axes[idx // 2, idx % 2]
        
        # Run MALA
        sampler = MALASampler(log_prob_fn, score_fn, dim=2, epsilon=eps)
        x0 = torch.tensor([4.0, 2.0])
        
        samples = sampler.sample(x0, n_samples, burn_in=burn_in)
        
        # Plot
        ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=10, 
                   alpha=0.5, colors='red')
        ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), 
                   alpha=0.3, s=5, c='blue')
        ax.plot(mu[0], mu[1], 'r*', markersize=15)
        
        # Statistics
        sample_mean = samples.mean(dim=0)
        sample_cov = torch.cov(samples.T)
        
        stats_text = f"ε = {eps}\n"
        stats_text += f"Accept rate: {sampler.acceptance_rate:.1%}\n"
        stats_text += f"Mean: [{sample_mean[0]:.2f}, {sample_mean[1]:.2f}]\n"
        stats_text += f"True: [{mu[0]:.2f}, {mu[1]:.2f}]"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.set_title(f'MALA with ε = {eps}', fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mala_step_size_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: mala_step_size_comparison.png")


if __name__ == "__main__":
    mala_example_2d_gaussian()
```

## Optimal Tuning: The 57% Rule

### Theoretical Foundation

Roberts & Tweedie (1996) and subsequent work established that for MALA:

**Optimal acceptance rate**: Approximately **57.4%** (for $d \to \infty$)

This is **higher** than the 23.4% optimal rate for random-walk Metropolis, reflecting that gradient-guided proposals are more informed.

### Practical Tuning

```python
def mala_tuning_analysis():
    """
    Analyze MALA performance as a function of step size.
    """
    # Target: 10-D isotropic Gaussian
    d = 10
    mu = torch.zeros(d)
    
    def log_prob_fn(x):
        return -0.5 * torch.sum(x ** 2)
    
    def score_fn(x):
        return -x
    
    # Test range of step sizes
    epsilons = np.logspace(-2, 0.5, 20)
    n_samples = 5000
    burn_in = 1000
    
    results = []
    
    print("Testing step sizes...")
    for eps in epsilons:
        sampler = MALASampler(log_prob_fn, score_fn, dim=d, epsilon=eps)
        x0 = torch.randn(d)
        
        samples = sampler.sample(x0, n_samples, burn_in=burn_in)
        
        # Compute effective sample size (simple autocorrelation-based)
        samples_np = samples[:, 0].numpy()
        centered = samples_np - samples_np.mean()
        acf_1 = np.correlate(centered[:-1], centered[1:], mode='valid')[0]
        acf_0 = np.correlate(centered, centered, mode='valid')[0]
        rho_1 = acf_1 / acf_0 if acf_0 > 0 else 0
        
        ess_factor = 1 / (1 + 2 * max(0, rho_1))  # Simplified ESS
        ess = n_samples * ess_factor
        
        results.append({
            'epsilon': eps,
            'acceptance_rate': sampler.acceptance_rate,
            'ess': ess,
            'ess_per_step': ess / (n_samples + burn_in)
        })
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epsilons_arr = np.array([r['epsilon'] for r in results])
    accept_arr = np.array([r['acceptance_rate'] for r in results])
    ess_arr = np.array([r['ess'] for r in results])
    efficiency_arr = np.array([r['ess_per_step'] for r in results])
    
    # Plot 1: Acceptance rate
    ax = axes[0]
    ax.semilogx(epsilons_arr, accept_arr, 'o-', linewidth=2, markersize=6)
    ax.axhline(y=0.574, color='red', linestyle='--', linewidth=2, 
               label='Optimal (57.4%)')
    ax.set_xlabel('Step size ε', fontsize=12)
    ax.set_ylabel('Acceptance rate', fontsize=12)
    ax.set_title('Acceptance Rate vs Step Size', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 2: ESS
    ax = axes[1]
    ax.semilogx(epsilons_arr, ess_arr, 'o-', linewidth=2, markersize=6, color='green')
    ax.set_xlabel('Step size ε', fontsize=12)
    ax.set_ylabel('Effective Sample Size', fontsize=12)
    ax.set_title('ESS vs Step Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency (ESS per step)
    ax = axes[2]
    ax.semilogx(epsilons_arr, efficiency_arr, 'o-', linewidth=2, markersize=6, color='orange')
    ax.set_xlabel('Step size ε', fontsize=12)
    ax.set_ylabel('ESS / Total Steps', fontsize=12)
    ax.set_title('Sampling Efficiency', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Find optimal
    opt_idx = np.argmax(efficiency_arr)
    ax.axvline(x=epsilons_arr[opt_idx], color='red', linestyle='--', alpha=0.7)
    ax.text(epsilons_arr[opt_idx]*1.2, efficiency_arr[opt_idx]*0.9,
            f'ε* ≈ {epsilons_arr[opt_idx]:.3f}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('mala_tuning_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: mala_tuning_analysis.png")
    
    # Print summary
    print(f"\nOptimal step size: ε = {epsilons_arr[opt_idx]:.4f}")
    print(f"Acceptance rate at optimal: {accept_arr[opt_idx]:.1%}")
    print(f"ESS at optimal: {ess_arr[opt_idx]:.0f}")


if __name__ == "__main__":
    mala_tuning_analysis()
```

### Step Size Scaling with Dimension

For a $d$-dimensional target, the optimal step size scales as:

$$
\epsilon_{opt} \sim d^{-1/6}
$$

This is **better** than random-walk Metropolis ($\epsilon \sim d^{-1}$) but worse than HMC ($\epsilon \sim d^{-1/4}$).

| Method | Optimal $\epsilon$ scaling | Mixing time |
|--------|---------------------------|-------------|
| Random Walk MH | $d^{-1}$ | $\mathcal{O}(d^2)$ |
| MALA | $d^{-1/6}$ | $\mathcal{O}(d^{5/3})$ |
| HMC | $d^{-1/4}$ | $\mathcal{O}(d^{5/4})$ |

## Comparing MALA, ULA, and Random Walk MH

### Implementation Comparison

```python
class RandomWalkMH:
    """Random walk Metropolis-Hastings for comparison."""
    
    def __init__(self, log_prob_fn, dim, sigma=1.0):
        self.log_prob_fn = log_prob_fn
        self.dim = dim
        self.sigma = sigma
        self.n_accepted = 0
        self.n_proposed = 0
    
    def step(self, x):
        x_prop = x + self.sigma * torch.randn_like(x)
        log_alpha = self.log_prob_fn(x_prop) - self.log_prob_fn(x)
        
        self.n_proposed += 1
        if torch.log(torch.rand(1)) < log_alpha:
            self.n_accepted += 1
            return x_prop, True
        return x.clone(), False
    
    @property
    def acceptance_rate(self):
        return self.n_accepted / max(1, self.n_proposed)


def compare_methods():
    """
    Compare MALA, ULA, and Random Walk MH.
    """
    # Target: 2D correlated Gaussian
    mu = torch.zeros(2)
    cov = torch.tensor([[1.0, 0.9], [0.9, 1.0]])  # Highly correlated
    precision = torch.linalg.inv(cov)
    
    mvn = dist.MultivariateNormal(mu, cov)
    
    def log_prob_fn(x):
        return mvn.log_prob(x)
    
    def score_fn(x):
        return -torch.matmul(x - mu, precision)
    
    n_steps = 5000
    x0 = torch.tensor([3.0, 3.0])
    
    # Tuned step sizes for fair comparison
    samplers = {
        'Random Walk MH': RandomWalkMH(log_prob_fn, dim=2, sigma=0.6),
        'ULA': None,  # Will use custom implementation
        'MALA': MALASampler(log_prob_fn, score_fn, dim=2, epsilon=0.5)
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Reference contours
    x_grid = torch.linspace(-4, 5, 100)
    y_grid = torch.linspace(-4, 5, 100)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    Z = torch.exp(mvn.log_prob(points)).reshape(100, 100)
    
    methods = ['Random Walk MH', 'ULA', 'MALA']
    colors = ['blue', 'green', 'red']
    
    for col, (method, color) in enumerate(zip(methods, colors)):
        # Generate samples
        x = x0.clone()
        chain = [x.clone()]
        
        if method == 'Random Walk MH':
            sampler = samplers[method]
            for _ in range(n_steps - 1):
                x, _ = sampler.step(x)
                chain.append(x.clone())
            accept_rate = sampler.acceptance_rate
            
        elif method == 'ULA':
            eps = 0.3
            for _ in range(n_steps - 1):
                x = x + eps * score_fn(x) + np.sqrt(2 * eps) * torch.randn_like(x)
                chain.append(x.clone())
            accept_rate = 1.0  # No rejection
            
        else:  # MALA
            sampler = samplers[method]
            for _ in range(n_steps - 1):
                x, _ = sampler.step(x)
                chain.append(x.clone())
            accept_rate = sampler.acceptance_rate
        
        chain = torch.stack(chain)
        
        # Top row: Trajectories
        ax = axes[0, col]
        ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=10, alpha=0.5)
        
        # Plot first 200 steps
        traj = chain[:200].numpy()
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=0.5, color=color)
        ax.scatter(traj[::5, 0], traj[::5, 1], s=10, alpha=0.5, color=color)
        ax.plot(traj[0, 0], traj[0, 1], 'ko', markersize=10)
        
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.set_title(f'{method}\nFirst 200 steps', fontsize=13, fontweight='bold')
        ax.set_xlim(-4, 5)
        ax.set_ylim(-4, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Bottom row: Trace plots
        ax = axes[1, col]
        ax.plot(chain[:, 0].numpy(), alpha=0.7, linewidth=0.5, label='$x_1$')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('$x_1$', fontsize=12)
        
        # Statistics
        burn_in = 1000
        samples = chain[burn_in:]
        mean = samples.mean(dim=0)
        
        stats_text = f"Accept: {accept_rate:.1%}\n"
        stats_text += f"Mean: [{mean[0]:.3f}, {mean[1]:.3f}]"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.set_title(f'{method}\nTrace plot', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mala_method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: mala_method_comparison.png")


if __name__ == "__main__":
    compare_methods()
```

### Efficiency Comparison

| Aspect | Random Walk MH | ULA | MALA |
|--------|----------------|-----|------|
| Uses gradients | No | Yes | Yes |
| Exact samples | Yes | No (biased) | Yes |
| Acceptance step | Yes | No | Yes |
| Optimal acceptance | ~23% | N/A | ~57% |
| Cost per step | 1 log π eval | 1 score eval | 2 score evals |
| Best for | Simple targets | Quick exploration | Moderate dims |

## Advanced MALA Variants

### Preconditioned MALA

For targets with different scales in different directions, use a preconditioning matrix $M$:

$$
x' = x + \epsilon M^{-1} s(x) + \sqrt{2\epsilon} M^{-1/2} \xi
$$

Common choices for $M$:

- **Identity**: Standard MALA
- **Diagonal approximation**: $M_{ii} \approx -\partial^2 \log \pi / \partial x_i^2$
- **Full Hessian**: $M \approx -\nabla^2 \log \pi(x^*)$ (computed at mode)

### Adaptive MALA

Adapt the step size during burn-in to achieve target acceptance rate:

```python
class AdaptiveMALA(MALASampler):
    """
    MALA with adaptive step size tuning.
    """
    
    def __init__(self, log_prob_fn, score_fn, dim, 
                 epsilon_init=1.0, target_accept=0.574):
        super().__init__(log_prob_fn, score_fn, dim, epsilon_init)
        self.target_accept = target_accept
        self.adapt_rate = 0.05
    
    def adapt_step_size(self, accepted: bool, iteration: int):
        """Adapt step size based on acceptance."""
        # Robbins-Monro adaptation
        gamma = 1.0 / (iteration + 1) ** self.adapt_rate
        
        if accepted:
            log_eps_update = gamma * (1 - self.target_accept)
        else:
            log_eps_update = -gamma * self.target_accept
        
        self.epsilon *= np.exp(log_eps_update)
        
        # Bounds to prevent extreme values
        self.epsilon = np.clip(self.epsilon, 1e-6, 10.0)
```

## Practical Considerations

### When to Use MALA

**Good scenarios for MALA:**

- Moderate-dimensional problems ($d \lesssim 100$)
- Smooth, unimodal targets
- Gradients are cheap to compute
- Need exact samples (bias not acceptable)

**Consider alternatives when:**

- Very high dimensions → HMC often better
- Multimodal targets → May need tempering
- Gradients unavailable → Random walk MH
- Quick exploration needed → ULA might suffice

### Common Implementation Issues

1. **Forgetting Hastings correction**: Leads to biased samples
2. **Numerical instability**: Clip gradients or use log-sum-exp
3. **Poor initialization**: Start from approximate mode
4. **Wrong step size**: Always tune for ~57% acceptance

## Summary

| Aspect | MALA |
|--------|------|
| **Proposal** | $x' = x + \epsilon s(x) + \sqrt{2\epsilon} \xi$ |
| **Acceptance** | MH correction with asymmetric $q$ |
| **Stationary distribution** | Exact $\pi$ |
| **Optimal acceptance** | ~57.4% |
| **Optimal $\epsilon$** | $\mathcal{O}(d^{-1/6})$ |
| **Mixing time** | $\mathcal{O}(d^{5/3})$ |

**Key insight**: MALA combines the gradient-guided proposals of Langevin dynamics with the exactness guarantee of Metropolis-Hastings. It's a sweet spot between simple random walk MH and the more complex HMC.

## Exercises

### Exercise 1: Hastings Correction

Implement MALA both with and without the Hastings correction. Show empirically that omitting the correction leads to biased samples.

### Exercise 2: High-Dimensional Scaling

Run MALA on isotropic Gaussians in dimensions $d = 2, 10, 50, 100, 500$. Verify the $d^{-1/6}$ scaling of the optimal step size.

### Exercise 3: Preconditioned MALA

Implement MALA with diagonal preconditioning for a target with varying scales. Compare efficiency with standard MALA.

### Exercise 4: Banana Target

Apply MALA to the banana-shaped distribution $p(x, y) \propto \exp(-\frac{1}{2}[x^2 + (y - x^2)^2])$. Compare with random walk MH.

## References

1. Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4), 341-363.

2. Roberts, G. O., & Rosenthal, J. S. (1998). Optimal scaling of discrete approximations to Langevin diffusions. *Journal of the Royal Statistical Society: Series B*, 60(1), 255-268.

3. Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. *Journal of the Royal Statistical Society: Series B*, 73(2), 123-214.

4. Xifara, T., Sherlock, C., Livingstone, S., Byrne, S., & Girolami, M. (2014). Langevin diffusions and the Metropolis-adjusted Langevin algorithm. *Statistics & Probability Letters*, 91, 14-19.
