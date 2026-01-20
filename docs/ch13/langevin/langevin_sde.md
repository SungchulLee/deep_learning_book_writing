# Langevin Stochastic Differential Equation

## Learning Objectives

By the end of this section, you will be able to:

- Derive the Langevin SDE from physical principles
- Explain the role of drift and diffusion terms in sampling
- Understand the Fokker-Planck equation and its stationary solution
- Prove that Langevin dynamics converges to the target distribution
- Implement continuous-time Langevin simulation in PyTorch

## Physical Origins: Brownian Motion

### Historical Context

In 1908, Paul Langevin proposed an equation to describe **Brownian motion**—the random motion of a particle suspended in a fluid, buffeted by collisions with surrounding molecules. This equation unified deterministic forces (from potential energy) with random fluctuations (from thermal noise).

The remarkable insight: the same mathematics that describes physical particles can be used to **sample from probability distributions**.

### The Classical Langevin Equation

Consider a particle with position $x$ and mass $m$ moving in a potential $U(x)$:

$$
m\frac{d^2 x}{dt^2} = -\gamma \frac{dx}{dt} - \nabla U(x) + \sqrt{2\gamma k_B T} \, \xi(t)
$$

where:

- $\gamma$ is the **friction coefficient** (viscous damping)
- $k_B T$ is the **thermal energy** (Boltzmann constant times temperature)
- $\xi(t)$ is **white noise** satisfying $\langle \xi(t) \xi(t') \rangle = \delta(t - t')$

The three forces are:

| Force | Expression | Physical Meaning |
|-------|------------|------------------|
| Friction | $-\gamma \dot{x}$ | Opposes motion, dissipates energy |
| Potential | $-\nabla U(x)$ | Pushes toward energy minima |
| Thermal | $\sqrt{2\gamma k_B T} \, \xi(t)$ | Random kicks from molecular collisions |

### The Overdamped Limit

In the **high-friction regime** where inertia is negligible ($m \to 0$), the acceleration term vanishes. This gives the **overdamped Langevin equation**:

$$
\gamma \frac{dx}{dt} = -\nabla U(x) + \sqrt{2\gamma k_B T} \, \xi(t)
$$

Rescaling time by $\gamma$ and setting $k_B T = 1$ (choosing units where thermal energy is unity):

$$
\frac{dx}{dt} = -\nabla U(x) + \sqrt{2} \, \xi(t)
$$

In rigorous SDE notation:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2} \, dW_t
$$

where $W_t$ is standard **Brownian motion** (Wiener process).

!!! info "From Physics to Sampling"
    The Boltzmann distribution at temperature $T=1$ is $\pi(x) \propto \exp(-U(x))$. The Langevin equation converges to this distribution—particles spend more time where energy is low!

## The Langevin SDE for Sampling

### Formulation

To sample from a target distribution $\pi(x) \propto \exp(-U(x))$, we simulate the SDE:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2} \, dW_t
$$

Equivalently, using the score function $s(x) = \nabla_x \log \pi(x) = -\nabla U(x)$:

$$
\boxed{dx_t = s(x_t) \, dt + \sqrt{2} \, dW_t}
$$

This is the **Langevin SDE** for sampling.

### Interpreting the Two Terms

**Drift term** $s(x_t) \, dt$:

- Deterministic flow toward high-probability regions
- Points "uphill" in the probability landscape
- Drives exploitation of modes

**Diffusion term** $\sqrt{2} \, dW_t$:

- Random exploration via Brownian motion
- Prevents getting stuck at modes
- Enables exploration of the full distribution

The constant $\sqrt{2}$ in front of $dW_t$ is not arbitrary—it's precisely calibrated so that the stationary distribution is $\pi(x)$.

### General Form with Temperature

At temperature $T$, the target distribution is:

$$
\pi_T(x) \propto \exp\left(-\frac{U(x)}{T}\right)
$$

The corresponding Langevin SDE is:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2T} \, dW_t
$$

| Temperature | Noise Level | Behavior |
|-------------|-------------|----------|
| High $T$ | Large $\sqrt{2T}$ | Wide exploration, modes blurred |
| $T = 1$ | $\sqrt{2}$ | Standard sampling from $\pi$ |
| Low $T$ | Small $\sqrt{2T}$ | Concentrates near modes |
| $T \to 0$ | No noise | Gradient descent to minimum |

## The Fokker-Planck Equation

### Density Evolution

The Langevin SDE describes the evolution of individual samples. But what about the probability density $\rho(x, t)$ of $x_t$?

The **Fokker-Planck equation** (also called the forward Kolmogorov equation) describes how $\rho$ evolves:

$$
\frac{\partial \rho}{\partial t} = -\nabla \cdot (\rho \, f) + \nabla \cdot (D \nabla \rho)
$$

For a general SDE $dx_t = f(x_t) \, dt + \sqrt{2D} \, dW_t$.

For the Langevin SDE with $f(x) = s(x) = \nabla \log \pi(x)$ and $D = 1$:

$$
\frac{\partial \rho}{\partial t} = -\nabla \cdot (\rho \, \nabla \log \pi) + \Delta \rho
$$

where $\Delta = \nabla \cdot \nabla$ is the Laplacian.

### Stationary Solution

At stationarity, $\frac{\partial \rho}{\partial t} = 0$. We claim $\rho = \pi$ is the stationary solution.

**Proof**: Substitute $\rho = \pi$:

$$
-\nabla \cdot (\pi \nabla \log \pi) + \Delta \pi = -\nabla \cdot \left(\pi \cdot \frac{\nabla \pi}{\pi}\right) + \Delta \pi = -\nabla \cdot (\nabla \pi) + \Delta \pi = -\Delta \pi + \Delta \pi = 0 \quad \checkmark
$$

Therefore, if we run Langevin dynamics long enough, the distribution of $x_t$ converges to $\pi(x)$.

### An Elegant Reformulation

The Fokker-Planck equation can be rewritten as:

$$
\frac{\partial \rho}{\partial t} = \nabla \cdot \left( \pi \nabla \left( \frac{\rho}{\pi} \right) \right)
$$

This form makes it obvious that $\rho = \pi$ is stationary (the gradient of a constant vanishes).

Moreover, this shows that Langevin dynamics performs **gradient flow** in the space of probability distributions, minimizing the KL divergence from $\rho$ to $\pi$.

## Convergence Guarantees

### Detailed Balance in Continuous Time

The Langevin SDE satisfies **detailed balance** (time-reversibility) with respect to $\pi$. This means if $(x_t, x_{t+dt})$ is a trajectory from the Langevin dynamics started at $\pi$, then the reversed trajectory $(x_{t+dt}, x_t)$ has the same distribution.

**Mathematical statement**: The adjoint dynamics (time-reversed SDE) is the same as the original dynamics when started from $\pi$.

### Convergence Theorem

**Theorem** (Langevin Convergence): Under mild conditions on $U(x)$:

1. **Strong convexity**: If $U$ is $m$-strongly convex ($\nabla^2 U \succeq m I$), the KL divergence contracts exponentially:
   
   $$
   D_{KL}(\rho_t \| \pi) \leq e^{-2mt} D_{KL}(\rho_0 \| \pi)
   $$

2. **General case**: With Lipschitz gradients and dissipativity conditions, the chain is geometrically ergodic.

**Intuition**: Strong convexity means the potential "curves upward" everywhere—like a bowl. The particle slides down into the bowl and explores around the bottom, converging to the Boltzmann distribution.

### Mixing Time

The **mixing time** is how long until $\rho_t$ is close to $\pi$. For a $d$-dimensional Gaussian target, the mixing time scales as $\mathcal{O}(d \cdot \kappa)$, where $\kappa$ is the condition number (ratio of largest to smallest eigenvalue of the covariance).

## PyTorch Implementation

### Simulating the Langevin SDE

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class LangevinSDE:
    """
    Continuous-time Langevin dynamics simulation via Euler-Maruyama discretization.
    """
    
    def __init__(self, score_fn, dim, temperature=1.0):
        """
        Args:
            score_fn: Function that computes s(x) = ∇log π(x)
            dim: Dimension of the state space
            temperature: Temperature parameter (default 1.0)
        """
        self.score_fn = score_fn
        self.dim = dim
        self.temperature = temperature
    
    def step(self, x, dt):
        """
        One step of Langevin dynamics via Euler-Maruyama.
        
        Args:
            x: Current position (batch_size, dim) or (dim,)
            dt: Time step size
            
        Returns:
            x_new: New position after one step
        """
        # Compute score at current position
        score = self.score_fn(x)
        
        # Drift term: s(x) * dt
        drift = score * dt
        
        # Diffusion term: sqrt(2T * dt) * N(0, I)
        diffusion = torch.sqrt(2 * self.temperature * torch.tensor(dt)) * torch.randn_like(x)
        
        # Euler-Maruyama update
        x_new = x + drift + diffusion
        
        return x_new
    
    def sample(self, x0, n_steps, dt, return_trajectory=False):
        """
        Generate samples via Langevin dynamics.
        
        Args:
            x0: Initial position
            n_steps: Number of steps
            dt: Time step size
            return_trajectory: Whether to return full trajectory
            
        Returns:
            samples or trajectory
        """
        x = x0.clone()
        
        if return_trajectory:
            trajectory = [x.clone()]
        
        for _ in range(n_steps):
            x = self.step(x, dt)
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return torch.stack(trajectory)
        else:
            return x


def langevin_gaussian_example():
    """
    Example: Langevin dynamics for sampling from a 2D Gaussian.
    """
    # Target: 2D Gaussian
    mu = torch.tensor([2.0, -1.0])
    cov = torch.tensor([[1.0, 0.6], [0.6, 1.0]])
    precision = torch.linalg.inv(cov)
    
    def score_fn(x):
        """Score function for Gaussian: s(x) = -Σ⁻¹(x - μ)"""
        return -torch.matmul(x - mu, precision)
    
    # Create sampler
    sampler = LangevinSDE(score_fn, dim=2)
    
    # Generate multiple chains
    n_chains = 500
    n_steps = 1000
    dt = 0.1
    
    # Start from dispersed initial positions
    x0 = torch.randn(n_chains, 2) * 3 + torch.tensor([5.0, 5.0])
    
    # Run dynamics
    print("Running Langevin dynamics...")
    trajectories = []
    x = x0.clone()
    
    for step in tqdm(range(n_steps)):
        trajectories.append(x.clone())
        x = sampler.step(x, dt)
    
    trajectories = torch.stack(trajectories)  # (n_steps, n_chains, 2)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Trajectories
    ax = axes[0]
    for i in range(min(20, n_chains)):
        traj = trajectories[:, i, :].numpy()
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.3, linewidth=0.5)
    
    # Add target contours
    x_grid = torch.linspace(-2, 7, 100)
    y_grid = torch.linspace(-5, 6, 100)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    
    mvn = dist.MultivariateNormal(mu, cov)
    Z = torch.exp(mvn.log_prob(points)).reshape(100, 100)
    
    ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=10, alpha=0.5, colors='red')
    ax.plot(mu[0], mu[1], 'r*', markersize=15, label='Target mean')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Sample Trajectories', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Evolution of sample distribution
    ax = axes[1]
    time_points = [0, 100, 500, 999]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))
    
    for t, color in zip(time_points, colors):
        samples = trajectories[t, :, :].numpy()
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, 
                   color=color, label=f't = {t*dt:.1f}')
    
    ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=10, alpha=0.3, colors='red')
    ax.plot(mu[0], mu[1], 'r*', markersize=15)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Distribution Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Convergence of statistics
    ax = axes[2]
    
    # Compute running mean over time
    mean_x1 = trajectories[:, :, 0].mean(dim=1).numpy()
    mean_x2 = trajectories[:, :, 1].mean(dim=1).numpy()
    
    times = np.arange(n_steps) * dt
    ax.plot(times, mean_x1, label='Mean $x_1$', linewidth=2)
    ax.plot(times, mean_x2, label='Mean $x_2$', linewidth=2)
    ax.axhline(y=mu[0].item(), color='C0', linestyle='--', alpha=0.7, label=f'True μ₁ = {mu[0]:.1f}')
    ax.axhline(y=mu[1].item(), color='C1', linestyle='--', alpha=0.7, label=f'True μ₂ = {mu[1]:.1f}')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Sample Mean', fontsize=12)
    ax.set_title('Convergence of Sample Statistics', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('langevin_sde_gaussian.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: langevin_sde_gaussian.png")
    
    # Print statistics
    final_samples = trajectories[-1, :, :]
    print(f"\nFinal sample mean: [{final_samples[:, 0].mean():.3f}, {final_samples[:, 1].mean():.3f}]")
    print(f"True mean: [{mu[0]:.3f}, {mu[1]:.3f}]")


def langevin_multimodal_example():
    """
    Example: Langevin dynamics for a multimodal distribution.
    """
    # Target: Mixture of Gaussians
    weights = torch.tensor([0.4, 0.6])
    means = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
    covs = torch.stack([
        torch.tensor([[0.5, 0.2], [0.2, 0.5]]),
        torch.tensor([[0.5, -0.2], [-0.2, 0.5]])
    ])
    
    def mixture_score(x):
        """Score function for Gaussian mixture."""
        batch_size = x.shape[0] if x.dim() > 1 else 1
        x = x.view(batch_size, 2)
        
        # Compute responsibilities
        log_probs = []
        for k in range(2):
            mvn = dist.MultivariateNormal(means[k], covs[k])
            log_probs.append(torch.log(weights[k]) + mvn.log_prob(x))
        
        log_probs = torch.stack(log_probs, dim=-1)  # (batch, 2)
        responsibilities = torch.softmax(log_probs, dim=-1)  # (batch, 2)
        
        # Compute component scores
        scores = []
        for k in range(2):
            precision = torch.linalg.inv(covs[k])
            score_k = -torch.matmul(x - means[k], precision)
            scores.append(score_k)
        
        scores = torch.stack(scores, dim=1)  # (batch, 2, 2)
        
        # Weighted average
        total_score = (responsibilities.unsqueeze(-1) * scores).sum(dim=1)
        
        return total_score.squeeze(0) if batch_size == 1 else total_score
    
    # Create sampler
    sampler = LangevinSDE(mixture_score, dim=2)
    
    # Generate samples
    n_chains = 1000
    n_steps = 2000
    dt = 0.05
    
    # Start from origin
    x0 = torch.randn(n_chains, 2) * 0.5
    
    print("\nRunning Langevin dynamics for mixture...")
    x = x0.clone()
    for _ in tqdm(range(n_steps)):
        x = sampler.step(x, dt)
    
    final_samples = x
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create density grid
    x_grid = torch.linspace(-5, 5, 100)
    y_grid = torch.linspace(-5, 5, 100)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    
    # Compute true mixture density
    log_p = []
    for k in range(2):
        mvn = dist.MultivariateNormal(means[k], covs[k])
        log_p.append(torch.log(weights[k]) + mvn.log_prob(points))
    log_p = torch.stack(log_p, dim=-1)
    Z = torch.logsumexp(log_p, dim=-1).exp().reshape(100, 100)
    
    # Plot 1: True density
    ax = axes[0]
    ax.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=20, cmap='Blues', alpha=0.7)
    ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=10, colors='darkblue', alpha=0.5)
    ax.scatter(means[:, 0].numpy(), means[:, 1].numpy(), c='red', s=100, marker='*', zorder=5)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('True Mixture Density', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Langevin samples
    ax = axes[1]
    ax.scatter(final_samples[:, 0].numpy(), final_samples[:, 1].numpy(), 
               alpha=0.3, s=10, c='green', label='Langevin samples')
    ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=10, colors='blue', alpha=0.3)
    ax.scatter(means[:, 0].numpy(), means[:, 1].numpy(), c='red', s=100, marker='*', 
               zorder=5, label='Modes')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Langevin Samples (t = 100)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('langevin_sde_multimodal.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: langevin_sde_multimodal.png")


if __name__ == "__main__":
    langevin_gaussian_example()
    langevin_multimodal_example()
```

## The Role of Step Size

### Discretization Error

The Euler-Maruyama scheme introduces discretization error of order $\mathcal{O}(\sqrt{dt})$ per step. Over $T$ time units ($T/dt$ steps), the total error accumulates.

For small $dt$:

- **Pros**: Better approximation of continuous dynamics, smaller bias
- **Cons**: More steps needed to reach equilibrium, higher computational cost

For large $dt$:

- **Pros**: Faster exploration, fewer steps
- **Cons**: Larger discretization bias, may diverge

### The Stability Condition

For a quadratic potential $U(x) = \frac{1}{2}x^\top H x$ with Hessian $H$, stability requires:

$$
dt < \frac{2}{\lambda_{\max}(H)}
$$

where $\lambda_{\max}(H)$ is the largest eigenvalue. In the probability interpretation, this corresponds to $dt$ being smaller than twice the inverse of the precision in the narrowest direction.

## Connection to Optimization

### Gradient Descent as Zero-Temperature Limit

As $T \to 0$, Langevin dynamics becomes deterministic gradient descent:

$$
\frac{dx}{dt} = -\nabla U(x) = s(x)
$$

This is gradient ascent on $\log \pi(x)$, converging to a mode.

### Stochastic Gradient Descent Connection

In machine learning, **Stochastic Gradient Descent (SGD)** can be viewed as noisy gradient descent:

$$
\theta_{t+1} = \theta_t - \eta \nabla \hat{L}(\theta_t)
$$

where $\nabla \hat{L}$ is a noisy estimate of the true gradient $\nabla L$.

Under certain conditions, SGD approximates Langevin dynamics with temperature proportional to the learning rate and gradient noise variance. This connection explains:

- Why SGD finds "flat" minima (Langevin prefers low energy with high entropy)
- Why learning rate schedules matter (they're like temperature annealing)
- Why batch size affects generalization (it changes the noise level)

## Summary

The Langevin SDE provides a principled framework for sampling:

| Component | Expression | Role |
|-----------|------------|------|
| **SDE** | $dx_t = s(x_t) dt + \sqrt{2} dW_t$ | Dynamics definition |
| **Drift** | $s(x) = \nabla \log \pi(x)$ | Moves toward high probability |
| **Diffusion** | $\sqrt{2} dW_t$ | Enables exploration |
| **Stationary distribution** | $\pi(x)$ | Samples converge to this |
| **Fokker-Planck** | $\partial_t \rho = -\nabla \cdot (\rho s) + \Delta \rho$ | Density evolution |

**Key insight**: The careful balance between drift (exploitation) and diffusion (exploration) ensures convergence to the target distribution.

## Exercises

### Exercise 1: Verify Stationarity

Show analytically that $\rho = \pi \propto \exp(-U(x))$ satisfies the Fokker-Planck equation $\partial_t \rho = 0$ for the Langevin SDE.

### Exercise 2: Temperature Effects

Modify the implementation to explore different temperatures. For a bimodal distribution, show how:

- Low temperature concentrates samples at the global mode
- High temperature allows exploration of both modes
- Find the critical temperature where mode-hopping becomes frequent

### Exercise 3: Convergence Rate

For a 1D Gaussian with variance $\sigma^2$, derive the convergence rate of the sample variance to the true variance as a function of time and step size.

### Exercise 4: Anisotropic Targets

Implement Langevin dynamics for a 2D Gaussian with condition number $\kappa = 100$. Study how convergence depends on step size and compare with the theoretical stability limit.

## References

1. Langevin, P. (1908). Sur la théorie du mouvement brownien. *Comptes Rendus de l'Académie des Sciences*, 146, 530-533.

2. Gardiner, C. W. (2009). *Stochastic Methods: A Handbook for the Natural and Social Sciences*. Springer.

3. Pavliotis, G. A. (2014). *Stochastic Processes and Applications*. Springer.

4. Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4), 341-363.
