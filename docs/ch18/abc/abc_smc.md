# ABC-SMC (Sequential Monte Carlo ABC)

ABC-SMC combines Approximate Bayesian Computation with Sequential Monte Carlo methods, enabling adaptive tolerance selection and efficient sampling from complex posteriors. This section presents the algorithm, its variants, and practical implementation guidance.

---

## Motivation

### Limitations of Previous Methods

**ABC Rejection**:
- Samples from prior (inefficient)
- Fixed tolerance (hard to choose)
- No adaptation

**ABC-MCMC**:
- Better than rejection for concentrated posteriors
- But can get stuck in local modes
- Still requires choosing $\epsilon$ upfront

### The SMC Solution

ABC-SMC addresses these issues by:
- Iteratively moving particles from prior to posterior
- Adapting tolerance from large (easy) to small (accurate)
- Maintaining diversity through resampling and mutation
- Providing natural parallelization

---

## The ABC-SMC Algorithm

### Overview

ABC-SMC generates a sequence of particle populations targeting:

$$
\pi_1(\theta) \to \pi_2(\theta) \to \cdots \to \pi_T(\theta) \approx p(\theta | \mathbf{y})
$$

where each $\pi_t$ is an ABC posterior with decreasing tolerance $\epsilon_1 > \epsilon_2 > \cdots > \epsilon_T$.

### Algorithm Statement

**Input**: Prior $p(\theta)$, simulator $p(\mathbf{x}|\theta)$, observed data $\mathbf{y}$, summary statistics $S(\cdot)$, distance $\rho$, tolerance schedule $\{\epsilon_1, \ldots, \epsilon_T\}$, number of particles $N$

**Output**: Weighted particles $\{(\theta_i^{(T)}, w_i^{(T)})\}_{i=1}^N$ approximating posterior

```
# Generation t = 1: Sample from prior
for i = 1 to N:
    repeat:
        Sample θᵢ⁽¹⁾ ~ p(θ)
        Simulate x ~ p(x | θᵢ⁽¹⁾)
    until ρ(S(x), S(y)) < ε₁
    
    Set wᵢ⁽¹⁾ = 1/N

# Generations t = 2, ..., T
for t = 2 to T:
    # Compute variance of previous generation for perturbation
    Σₜ = 2 × Cov({θᵢ⁽ᵗ⁻¹⁾})
    
    for i = 1 to N:
        repeat:
            # Sample from previous generation
            Sample θ* from {θⱼ⁽ᵗ⁻¹⁾} with probabilities {wⱼ⁽ᵗ⁻¹⁾}
            
            # Perturb
            Sample θᵢ⁽ᵗ⁾ ~ K(θ | θ*) = N(θ*, Σₜ)
            
            # Simulate
            Simulate x ~ p(x | θᵢ⁽ᵗ⁾)
        until ρ(S(x), S(y)) < εₜ
        
        # Compute weight
        wᵢ⁽ᵗ⁾ ∝ p(θᵢ⁽ᵗ⁾) / Σⱼ wⱼ⁽ᵗ⁻¹⁾ K(θᵢ⁽ᵗ⁾ | θⱼ⁽ᵗ⁻¹⁾)
    
    # Normalize weights
    Normalize {wᵢ⁽ᵗ⁾} to sum to 1

return {(θᵢ⁽ᵀ⁾, wᵢ⁽ᵀ⁾)}
```

---

## Implementation

### Complete Python Implementation

```python
import numpy as np
from scipy.stats import multivariate_normal

class ABCSMC:
    def __init__(self, prior_sampler, prior_logpdf, simulator, 
                 summary_fn, distance_fn, y_obs, n_particles=1000):
        """
        ABC-SMC sampler.
        
        Args:
            prior_sampler: Function returning samples from prior
            prior_logpdf: Function theta -> log p(theta)
            simulator: Function theta -> simulated data
            summary_fn: Function x -> summary statistics
            distance_fn: Function (s1, s2) -> distance
            y_obs: Observed data
            n_particles: Number of particles
        """
        self.prior_sampler = prior_sampler
        self.prior_logpdf = prior_logpdf
        self.simulator = simulator
        self.summary_fn = summary_fn
        self.distance_fn = distance_fn
        self.s_obs = summary_fn(y_obs)
        self.n_particles = n_particles
        
    def sample_initial_population(self, epsilon):
        """Sample initial population from prior with ABC."""
        particles = []
        distances = []
        
        while len(particles) < self.n_particles:
            theta = self.prior_sampler()
            x = self.simulator(theta)
            s = self.summary_fn(x)
            d = self.distance_fn(s, self.s_obs)
            
            if d < epsilon:
                particles.append(theta)
                distances.append(d)
        
        particles = np.array(particles)
        weights = np.ones(self.n_particles) / self.n_particles
        
        return particles, weights, distances
    
    def sample_next_population(self, prev_particles, prev_weights, epsilon):
        """Sample next population via importance sampling."""
        dim = prev_particles.shape[1]
        
        # Compute perturbation covariance
        cov = 2 * np.cov(prev_particles.T, aweights=prev_weights)
        if dim == 1:
            cov = np.array([[cov]])
        
        particles = []
        distances = []
        weights = []
        
        for i in range(self.n_particles):
            # Resample-move
            accepted = False
            while not accepted:
                # Sample from previous population
                idx = np.random.choice(
                    self.n_particles, p=prev_weights
                )
                theta_star = prev_particles[idx]
                
                # Perturb
                theta = np.random.multivariate_normal(theta_star, cov)
                
                # Check prior support
                if self.prior_logpdf(theta) == -np.inf:
                    continue
                
                # Simulate and check ABC
                x = self.simulator(theta)
                s = self.summary_fn(x)
                d = self.distance_fn(s, self.s_obs)
                
                if d < epsilon:
                    accepted = True
            
            particles.append(theta)
            distances.append(d)
            
            # Compute weight
            kernel_sum = sum(
                prev_weights[j] * multivariate_normal.pdf(
                    theta, prev_particles[j], cov
                )
                for j in range(self.n_particles)
            )
            w = np.exp(self.prior_logpdf(theta)) / kernel_sum
            weights.append(w)
        
        particles = np.array(particles)
        weights = np.array(weights)
        weights /= weights.sum()  # Normalize
        
        return particles, weights, distances
    
    def run(self, epsilon_schedule, verbose=True):
        """
        Run ABC-SMC with given tolerance schedule.
        
        Args:
            epsilon_schedule: List of decreasing tolerances
            verbose: Print progress
        
        Returns:
            particles: Final particle positions
            weights: Final particle weights
            history: List of (particles, weights) for each generation
        """
        history = []
        
        # Initial population
        if verbose:
            print(f"Generation 1, ε={epsilon_schedule[0]:.4f}")
        
        particles, weights, distances = self.sample_initial_population(
            epsilon_schedule[0]
        )
        history.append((particles.copy(), weights.copy()))
        
        if verbose:
            print(f"  Accepted {self.n_particles} particles")
        
        # Subsequent populations
        for t, epsilon in enumerate(epsilon_schedule[1:], start=2):
            if verbose:
                print(f"Generation {t}, ε={epsilon:.4f}")
            
            particles, weights, distances = self.sample_next_population(
                particles, weights, epsilon
            )
            history.append((particles.copy(), weights.copy()))
            
            # Check effective sample size
            ess = 1 / np.sum(weights**2)
            if verbose:
                print(f"  ESS: {ess:.1f}")
        
        return particles, weights, history
```

### Adaptive Tolerance Selection

Instead of prespecifying tolerances, adapt based on particle distances:

```python
def run_adaptive(self, epsilon_init, epsilon_final, alpha=0.5, 
                 max_generations=20, verbose=True):
    """
    Run ABC-SMC with adaptive tolerance.
    
    Args:
        epsilon_init: Initial tolerance (or None for automatic)
        epsilon_final: Target final tolerance
        alpha: Quantile for tolerance selection (e.g., 0.5 = median)
        max_generations: Maximum number of generations
    """
    history = []
    
    # Initial population
    if epsilon_init is None:
        # Set initial epsilon to accept ~50% from prior
        epsilon_init = self.calibrate_initial_epsilon()
    
    particles, weights, distances = self.sample_initial_population(epsilon_init)
    history.append((particles.copy(), weights.copy()))
    
    epsilon = epsilon_init
    
    for t in range(2, max_generations + 1):
        # Adaptive epsilon: quantile of current distances
        epsilon_new = np.percentile(distances, alpha * 100)
        epsilon_new = max(epsilon_new, epsilon_final)
        
        if verbose:
            print(f"Generation {t}, ε: {epsilon:.4f} -> {epsilon_new:.4f}")
        
        if epsilon_new >= epsilon * 0.99:  # No progress
            if verbose:
                print("  Tolerance not decreasing, stopping")
            break
        
        epsilon = epsilon_new
        particles, weights, distances = self.sample_next_population(
            particles, weights, epsilon
        )
        history.append((particles.copy(), weights.copy()))
        
        ess = 1 / np.sum(weights**2)
        if verbose:
            print(f"  ESS: {ess:.1f}")
        
        if epsilon <= epsilon_final:
            if verbose:
                print("  Reached target tolerance")
            break
    
    return particles, weights, history
```

---

## Theoretical Properties

### Target Distribution

At generation $t$, ABC-SMC targets:

$$
\pi_t(\theta) \propto p(\theta) \cdot P(\rho(S(\mathbf{X}), S(\mathbf{y})) < \epsilon_t | \theta)
$$

As $\epsilon_t \to 0$:

$$
\pi_t(\theta) \to p(\theta | \mathbf{y})
$$

### Weight Derivation

The importance weight at generation $t$ is:

$$
w_i^{(t)} \propto \frac{\pi_t(\theta_i^{(t)})}{\sum_{j=1}^N w_j^{(t-1)} K_t(\theta_i^{(t)} | \theta_j^{(t-1)})}
$$

Since $\pi_t(\theta) \propto p(\theta) \cdot \alpha_t(\theta)$ where $\alpha_t$ is the ABC acceptance probability:

$$
w_i^{(t)} \propto \frac{p(\theta_i^{(t)})}{\sum_{j=1}^N w_j^{(t-1)} K_t(\theta_i^{(t)} | \theta_j^{(t-1)})}
$$

(The ABC acceptance is implicit in the sampling.)

### Effective Sample Size

Monitor the ESS to detect weight degeneracy:

$$
\text{ESS} = \frac{1}{\sum_{i=1}^N (w_i^{(t)})^2}
$$

If ESS drops too low, resample to restore uniform weights.

---

## Perturbation Kernels

### Gaussian Kernel (Standard)

$$
K(\theta' | \theta) = \mathcal{N}(\theta' | \theta, \Sigma_t)
$$

**Covariance choices**:
- $\Sigma_t = 2 \times \text{Cov}(\{\theta_i^{(t-1)}\})$ (twice weighted sample covariance)
- Optimal linear shrinkage estimator
- Component-wise: $\Sigma_t = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$

### Optimal Perturbation

The optimal perturbation kernel minimizes the expected number of simulations. For Gaussian targets:

$$
K^*(\theta' | \theta) = \mathcal{N}\left(\theta' \big| \theta, \frac{4}{(d+2)} \text{Cov}(\pi_t)\right)
$$

### Local Linear Regression Kernel

Adapt kernel to local geometry:

```python
def local_kernel(theta, particles, weights, k=50):
    """Local covariance based on nearest neighbors."""
    distances = np.linalg.norm(particles - theta, axis=1)
    nearest = np.argsort(distances)[:k]
    
    local_particles = particles[nearest]
    local_weights = weights[nearest]
    local_weights /= local_weights.sum()
    
    return 2 * np.cov(local_particles.T, aweights=local_weights)
```

---

## Resampling Strategies

### When to Resample

Resample when ESS drops below threshold (e.g., $N/2$):

```python
def maybe_resample(particles, weights, threshold_ratio=0.5):
    """Resample if ESS is below threshold."""
    ess = 1 / np.sum(weights**2)
    
    if ess < threshold_ratio * len(weights):
        # Systematic resampling
        indices = systematic_resample(weights)
        particles = particles[indices]
        weights = np.ones(len(weights)) / len(weights)
    
    return particles, weights

def systematic_resample(weights):
    """Systematic resampling."""
    n = len(weights)
    positions = (np.arange(n) + np.random.rand()) / n
    
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    
    return indices
```

### Resampling Schemes

| Scheme | Variance | Complexity |
|--------|----------|------------|
| Multinomial | Higher | O(N) |
| Systematic | Lower | O(N) |
| Residual | Lower | O(N) |
| Stratified | Lower | O(N) |

Systematic resampling is typically preferred for its low variance.

---

## Practical Considerations

### Choosing the Number of Particles

**Rule of thumb**: $N = 1000$ to $10000$

**Trade-offs**:
- More particles → better approximation, more computation
- Fewer particles → faster, but higher variance

### Tolerance Schedule

**Fixed schedule**:
```python
epsilon_schedule = [2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
```

**Adaptive** (recommended):
- Set each $\epsilon_t$ as a quantile (e.g., median) of current distances
- Automatically adjusts to problem difficulty

### Stopping Criteria

1. **Target tolerance reached**: $\epsilon_t \leq \epsilon_{target}$
2. **Tolerance stagnation**: $\epsilon_t \approx \epsilon_{t-1}$
3. **Maximum generations**: Safety limit
4. **Computation budget**: Maximum simulations

### Parallelization

ABC-SMC is naturally parallel within each generation:

```python
from multiprocessing import Pool

def sample_particle_parallel(args):
    """Sample one particle (for parallel execution)."""
    prev_particles, prev_weights, cov, epsilon, simulator, summary_fn, \
        distance_fn, s_obs, prior_logpdf = args
    
    while True:
        # Sample and perturb
        idx = np.random.choice(len(prev_weights), p=prev_weights)
        theta = np.random.multivariate_normal(prev_particles[idx], cov)
        
        if prior_logpdf(theta) == -np.inf:
            continue
        
        x = simulator(theta)
        s = summary_fn(x)
        d = distance_fn(s, s_obs)
        
        if d < epsilon:
            return theta, d

def sample_population_parallel(prev_particles, prev_weights, epsilon, 
                               n_particles, n_workers=4, **kwargs):
    """Sample population in parallel."""
    cov = 2 * np.cov(prev_particles.T, aweights=prev_weights)
    
    args = [(prev_particles, prev_weights, cov, epsilon, 
             kwargs['simulator'], kwargs['summary_fn'],
             kwargs['distance_fn'], kwargs['s_obs'], 
             kwargs['prior_logpdf'])] * n_particles
    
    with Pool(n_workers) as pool:
        results = pool.map(sample_particle_parallel, args)
    
    particles = np.array([r[0] for r in results])
    distances = [r[1] for r in results]
    
    return particles, distances
```

---

## Variants

### ABC-SMC with Regression Adjustment

Post-process particles with regression adjustment:

```python
def regression_adjustment(particles, summaries, s_obs):
    """Adjust particles using local linear regression."""
    from sklearn.linear_model import LinearRegression
    
    adjusted = []
    for i, (theta, s) in enumerate(zip(particles, summaries)):
        # Local regression
        reg = LinearRegression()
        reg.fit(summaries, particles)
        
        # Adjust
        theta_adj = theta - reg.predict([s])[0] + reg.predict([s_obs])[0]
        adjusted.append(theta_adj)
    
    return np.array(adjusted)
```

### ABC-SMC with Multiple Distances

Use different distances for different summaries:

```python
def multi_distance(s1, s2, weights):
    """Weighted combination of component-wise distances."""
    return np.sum(weights * np.abs(s1 - s2))
```

### Population Monte Carlo ABC

A variant that doesn't require a schedule—adapts everything online.

---

## Example: Ecological Model

```python
# Lotka-Volterra predator-prey model
def lotka_volterra(theta, T=100, dt=0.1):
    """Simulate Lotka-Volterra model."""
    alpha, beta, gamma, delta = theta
    
    prey = [50.0]
    predator = [20.0]
    
    for _ in range(int(T / dt)):
        x, y = prey[-1], predator[-1]
        
        dx = (alpha * x - beta * x * y) * dt + np.sqrt(max(x, 0)) * np.random.randn() * 0.1
        dy = (delta * x * y - gamma * y) * dt + np.sqrt(max(y, 0)) * np.random.randn() * 0.1
        
        prey.append(max(0, x + dx))
        predator.append(max(0, y + dy))
    
    return np.array([prey[::10], predator[::10]])  # Subsample

def lv_summaries(x):
    """Summary statistics for LV model."""
    prey, predator = x
    return np.array([
        np.mean(prey), np.std(prey),
        np.mean(predator), np.std(predator),
        np.corrcoef(prey, predator)[0, 1],
    ])

# Prior
def lv_prior_sample():
    return np.array([
        np.random.uniform(0.5, 1.5),   # alpha
        np.random.uniform(0.01, 0.1),  # beta
        np.random.uniform(0.5, 1.5),   # gamma
        np.random.uniform(0.01, 0.1),  # delta
    ])

# Run ABC-SMC
sampler = ABCSMC(
    prior_sampler=lv_prior_sample,
    prior_logpdf=lambda t: 0 if all(t > 0) else -np.inf,
    simulator=lotka_volterra,
    summary_fn=lv_summaries,
    distance_fn=lambda s1, s2: np.linalg.norm(s1 - s2),
    y_obs=lotka_volterra([1.0, 0.05, 1.0, 0.05]),  # "True" data
    n_particles=1000
)

particles, weights, history = sampler.run_adaptive(
    epsilon_init=10.0, 
    epsilon_final=1.0
)
```

---

## Comparison

| Method | Efficiency | Adaptivity | Parallelism | Complexity |
|--------|------------|------------|-------------|------------|
| ABC Rejection | Low | None | High | Simple |
| ABC-MCMC | Medium | None | Low | Medium |
| ABC-SMC | High | High | High | Complex |

ABC-SMC is generally the most efficient for complex problems, but requires more implementation effort.

---

## Summary

| Aspect | Description |
|--------|-------------|
| **Algorithm** | Sequential importance sampling with ABC |
| **Adaptation** | Tolerance decreases across generations |
| **Efficiency** | Higher than rejection/MCMC |
| **Parallelization** | Natural within generations |
| **Output** | Weighted particles approximating posterior |
| **Key tuning** | Number of particles, tolerance schedule/adaptation |

ABC-SMC represents the state-of-the-art in classical ABC methods, providing efficient, adaptive inference for simulator-based models.

---

## Exercises

1. **Implementation**. Implement ABC-SMC for the normal model. Compare efficiency to rejection and MCMC.

2. **Adaptive vs fixed schedule**. Compare adaptive tolerance selection to a fixed geometric schedule. Which is more efficient?

3. **Particle number sensitivity**. Study how results change with $N = 100, 500, 1000, 5000$ particles.

4. **Kernel comparison**. Compare Gaussian kernels with different covariance estimators.

5. **Ecological inference**. Apply ABC-SMC to a real ecological dataset with a Lotka-Volterra model.

---

## References

1. Sisson, S. A., Fan, Y., & Tanaka, M. M. (2007). "Sequential Monte Carlo Without Likelihoods." *PNAS*.
2. Beaumont, M. A., Cornuet, J.-M., Marin, J.-M., & Robert, C. P. (2009). "Adaptive Approximate Bayesian Computation." *Biometrika*.
3. Del Moral, P., Doucet, A., & Jasra, A. (2012). "An Adaptive Sequential Monte Carlo Method for Approximate Bayesian Computation." *Statistics and Computing*.
4. Toni, T., Welch, D., Strelkowa, N., Ipsen, A., & Stumpf, M. P. (2009). "Approximate Bayesian Computation Scheme for Parameter Inference and Model Selection in Dynamical Systems." *JRSS-B*.
