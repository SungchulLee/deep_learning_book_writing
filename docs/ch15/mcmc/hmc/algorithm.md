# The HMC Algorithm

This section presents the complete Hamiltonian Monte Carlo algorithm: the full procedure, correctness proof, implementation details, and practical considerations for deployment.

---

## Algorithm Overview

### The HMC Recipe

HMC combines three ingredients to sample from $\pi(\mathbf{x}) \propto \exp(-U(\mathbf{x}))$:

1. **Momentum augmentation**: Extend state space with auxiliary momentum $\mathbf{v}$
2. **Hamiltonian dynamics**: Propose distant states via deterministic trajectories
3. **Metropolis-Hastings correction**: Accept/reject to ensure exactness

### The Complete Algorithm

**Input**: Initial position $\mathbf{x}^{(0)}$, number of samples $N$, step size $\epsilon$, trajectory length $L$, mass matrix $\mathbf{M}$

**Output**: Samples $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(N)}\}$ from $\pi(\mathbf{x})$

```
for t = 1 to N:
    # Step 1: Sample momentum
    v ~ Normal(0, M)
    
    # Step 2: Compute initial Hamiltonian
    H_current = U(x⁽ᵗ⁻¹⁾) + ½ vᵀ M⁻¹ v
    
    # Step 3: Leapfrog integration
    x_prop, v_prop = Leapfrog(x⁽ᵗ⁻¹⁾, v, ε, L)
    
    # Step 4: Negate momentum (for reversibility)
    v_prop = -v_prop
    
    # Step 5: Compute proposed Hamiltonian
    H_prop = U(x_prop) + ½ v_propᵀ M⁻¹ v_prop
    
    # Step 6: Metropolis-Hastings acceptance
    α = min(1, exp(H_current - H_prop))
    u ~ Uniform(0, 1)
    
    if u < α:
        x⁽ᵗ⁾ = x_prop    # Accept
    else:
        x⁽ᵗ⁾ = x⁽ᵗ⁻¹⁾    # Reject
```

---

## Detailed Step-by-Step

### Step 1: Momentum Resampling

Draw fresh momentum from the marginal:

$$
\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})
$$

**Why resample every iteration?**

- Injects stochasticity (otherwise HMC would be deterministic)
- Ensures ergodicity (allows exploration of different energy surfaces)
- Maintains the joint distribution $\pi(\mathbf{x}, \mathbf{v}) = \pi(\mathbf{x}) \cdot \mathcal{N}(\mathbf{v}; \mathbf{0}, \mathbf{M})$

**Implementation**:
```python
v = np.random.multivariate_normal(np.zeros(d), M)
# Or equivalently:
v = L_M @ np.random.randn(d)  # where M = L_M @ L_M.T
```

### Step 2: Initial Hamiltonian

Compute the total energy at the starting point:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + K(\mathbf{v}) = -\log \tilde{\pi}(\mathbf{x}) + \frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}
$$

This serves as the reference for the acceptance decision.

### Step 3: Leapfrog Integration

Simulate Hamiltonian dynamics for $L$ steps with step size $\epsilon$:

$$
(\mathbf{x}', \mathbf{v}') = \text{Leapfrog}(\mathbf{x}, \mathbf{v}, \epsilon, L)
$$

The trajectory explores phase space while approximately preserving energy.

**Computational cost**: $L$ gradient evaluations (the dominant cost).

### Step 4: Momentum Negation

Negate the final momentum:

$$
\mathbf{v}' \leftarrow -\mathbf{v}'
$$

**Why negate?**

This ensures the proposal is its own inverse: starting from $(\mathbf{x}', -\mathbf{v}')$ and running leapfrog returns to $(\mathbf{x}, -\mathbf{v})$.

**Note**: For the MH ratio, negation doesn't affect the kinetic energy $K(\mathbf{v}) = K(-\mathbf{v})$, so it's often omitted in implementations. But conceptually, it's part of ensuring detailed balance.

### Step 5: Proposed Hamiltonian

Compute energy at the proposal:

$$
H(\mathbf{x}', \mathbf{v}') = U(\mathbf{x}') + K(\mathbf{v}')
$$

The energy change $\Delta H = H(\mathbf{x}', \mathbf{v}') - H(\mathbf{x}, \mathbf{v})$ determines acceptance.

### Step 6: Metropolis-Hastings Acceptance

Accept with probability:

$$
\alpha = \min\left(1, \exp(-\Delta H)\right) = \min\left(1, \frac{\pi(\mathbf{x}', \mathbf{v}')}{\pi(\mathbf{x}, \mathbf{v})}\right)
$$

- If $\Delta H \leq 0$ (energy decreased): Accept with probability 1
- If $\Delta H > 0$ (energy increased): Accept with probability $\exp(-\Delta H)$

**Implementation**:
```python
delta_H = H_prop - H_current
if np.random.rand() < np.exp(-delta_H):
    x = x_prop  # Accept
# else: keep current x (reject)
```

---

## Correctness Proof

### What We Need to Show

HMC is a valid MCMC algorithm if the transition kernel $K(\mathbf{x}' | \mathbf{x})$ satisfies **detailed balance**:

$$
\pi(\mathbf{x}) K(\mathbf{x}' | \mathbf{x}) = \pi(\mathbf{x}') K(\mathbf{x} | \mathbf{x}')
$$

### The Transition Kernel

The HMC transition from $\mathbf{x}$ to $\mathbf{x}'$ involves:

1. Sample $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})$
2. Deterministic map $(\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{x}', \mathbf{v}')$ via leapfrog + negation
3. Accept with probability $\alpha(\mathbf{x}, \mathbf{v}, \mathbf{x}', \mathbf{v}')$

### Key Properties

**Property 1: Involution**

The map $T: (\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{x}', \mathbf{v}')$ (leapfrog + negation) is an **involution**: $T \circ T = \text{identity}$.

*Proof*: Starting from $(\mathbf{x}', \mathbf{v}')$, applying leapfrog gives $(\mathbf{x}, -\mathbf{v})$ (by time-reversibility), then negation gives $(\mathbf{x}, \mathbf{v})$.

**Property 2: Volume Preservation**

The leapfrog map has Jacobian determinant 1 (symplectic). Momentum negation also has determinant 1 (actually $(-1)^d$, but absolute value is 1).

**Property 3: Energy Symmetry**

$H(\mathbf{x}, \mathbf{v}) = H(\mathbf{x}, -\mathbf{v})$ because kinetic energy is even in $\mathbf{v}$.

### Detailed Balance Proof

Consider the probability flow from $(\mathbf{x}, \mathbf{v})$ to $(\mathbf{x}', \mathbf{v}')$:

$$
\text{Flow}[(\mathbf{x}, \mathbf{v}) \to (\mathbf{x}', \mathbf{v}')] = \pi(\mathbf{x}, \mathbf{v}) \cdot \delta(T(\mathbf{x}, \mathbf{v}) - (\mathbf{x}', \mathbf{v}')) \cdot \alpha(\mathbf{x}, \mathbf{v}, \mathbf{x}', \mathbf{v}')
$$

The reverse flow from $(\mathbf{x}', \mathbf{v}')$ to $(\mathbf{x}, \mathbf{v})$:

$$
\text{Flow}[(\mathbf{x}', \mathbf{v}') \to (\mathbf{x}, \mathbf{v})] = \pi(\mathbf{x}', \mathbf{v}') \cdot \delta(T(\mathbf{x}', \mathbf{v}') - (\mathbf{x}, \mathbf{v})) \cdot \alpha(\mathbf{x}', \mathbf{v}', \mathbf{x}, \mathbf{v})
$$

By the involution property, $T(\mathbf{x}', \mathbf{v}') = (\mathbf{x}, \mathbf{v})$, so the delta functions match.

The MH acceptance ratio ensures:

$$
\frac{\alpha(\mathbf{x}, \mathbf{v}, \mathbf{x}', \mathbf{v}')}{\alpha(\mathbf{x}', \mathbf{v}', \mathbf{x}, \mathbf{v})} = \frac{\pi(\mathbf{x}', \mathbf{v}')}{\pi(\mathbf{x}, \mathbf{v})}
$$

Therefore the flows balance:

$$
\pi(\mathbf{x}, \mathbf{v}) \cdot \alpha(\mathbf{x}, \mathbf{v}, \mathbf{x}', \mathbf{v}') = \pi(\mathbf{x}', \mathbf{v}') \cdot \alpha(\mathbf{x}', \mathbf{v}', \mathbf{x}, \mathbf{v})
$$

Integrating out momentum (which is resampled each iteration) gives detailed balance for the marginal on $\mathbf{x}$.

---

## Implementation

### Complete Python Implementation

```python
import numpy as np

class HMC:
    def __init__(self, log_prob, grad_log_prob, dim, 
                 epsilon=0.1, L=10, M=None):
        """
        Hamiltonian Monte Carlo sampler.
        
        Args:
            log_prob: Function returning log π(x) (up to constant)
            grad_log_prob: Function returning ∇ log π(x)
            dim: Dimension of the target distribution
            epsilon: Leapfrog step size
            L: Number of leapfrog steps
            M: Mass matrix (default: identity)
        """
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob
        self.dim = dim
        self.epsilon = epsilon
        self.L = L
        
        if M is None:
            self.M = np.eye(dim)
            self.M_inv = np.eye(dim)
            self.L_M = np.eye(dim)
        else:
            self.M = M
            self.M_inv = np.linalg.inv(M)
            self.L_M = np.linalg.cholesky(M)
    
    def kinetic_energy(self, v):
        """Compute kinetic energy K(v) = ½ vᵀ M⁻¹ v"""
        return 0.5 * v @ self.M_inv @ v
    
    def hamiltonian(self, x, v):
        """Compute total energy H(x,v) = U(x) + K(v)"""
        U = -self.log_prob(x)
        K = self.kinetic_energy(v)
        return U + K
    
    def leapfrog(self, x, v):
        """Perform L leapfrog steps."""
        x = x.copy()
        v = v.copy()
        
        # Half step for momentum
        v = v + (self.epsilon / 2) * self.grad_log_prob(x)
        
        # Full steps
        for _ in range(self.L - 1):
            x = x + self.epsilon * self.M_inv @ v
            v = v + self.epsilon * self.grad_log_prob(x)
        
        # Final position step
        x = x + self.epsilon * self.M_inv @ v
        
        # Final half step for momentum
        v = v + (self.epsilon / 2) * self.grad_log_prob(x)
        
        return x, v
    
    def sample_momentum(self):
        """Sample momentum from N(0, M)."""
        return self.L_M @ np.random.randn(self.dim)
    
    def step(self, x):
        """Perform one HMC iteration."""
        # Sample momentum
        v = self.sample_momentum()
        
        # Current Hamiltonian
        H_current = self.hamiltonian(x, v)
        
        # Leapfrog integration
        x_prop, v_prop = self.leapfrog(x, v)
        
        # Proposed Hamiltonian (momentum negation doesn't affect H)
        H_prop = self.hamiltonian(x_prop, v_prop)
        
        # Metropolis-Hastings acceptance
        delta_H = H_prop - H_current
        
        if np.isnan(delta_H):
            # Numerical issue - reject
            return x, False
        
        if np.random.rand() < np.exp(-delta_H):
            return x_prop, True
        else:
            return x, False
    
    def sample(self, x0, n_samples, n_warmup=1000):
        """
        Generate samples from the target distribution.
        
        Args:
            x0: Initial position
            n_samples: Number of samples to generate
            n_warmup: Number of warmup iterations (discarded)
        
        Returns:
            samples: Array of shape (n_samples, dim)
            accept_rate: Acceptance rate
        """
        x = x0.copy()
        samples = np.zeros((n_samples, self.dim))
        n_accept = 0
        
        # Warmup
        for _ in range(n_warmup):
            x, _ = self.step(x)
        
        # Sampling
        for i in range(n_samples):
            x, accepted = self.step(x)
            samples[i] = x
            n_accept += accepted
        
        accept_rate = n_accept / n_samples
        return samples, accept_rate
```

### Usage Example

```python
# Target: 2D Gaussian with correlation
mu = np.array([0, 0])
Sigma = np.array([[1, 0.8], [0.8, 1]])
Sigma_inv = np.linalg.inv(Sigma)

def log_prob(x):
    return -0.5 * (x - mu) @ Sigma_inv @ (x - mu)

def grad_log_prob(x):
    return -Sigma_inv @ (x - mu)

# Create sampler
hmc = HMC(log_prob, grad_log_prob, dim=2, epsilon=0.1, L=20)

# Generate samples
x0 = np.zeros(2)
samples, accept_rate = hmc.sample(x0, n_samples=10000, n_warmup=1000)

print(f"Acceptance rate: {accept_rate:.2%}")
print(f"Sample mean: {samples.mean(axis=0)}")
print(f"Sample cov:\n{np.cov(samples.T)}")
```

---

## Practical Considerations

### Tuning Parameters

| Parameter | Effect | Tuning Strategy |
|-----------|--------|-----------------|
| $\epsilon$ (step size) | Acceptance rate, stability | Adapt to target 65-80% acceptance |
| $L$ (trajectory length) | Exploration distance | Long enough to decorrelate; NUTS automates |
| $\mathbf{M}$ (mass matrix) | Adaptation to geometry | Estimate from warmup samples |

### Warmup Phase

The warmup phase serves multiple purposes:

1. **Burn-in**: Move from initial position to typical set
2. **Adaptation**: Learn good values for $\epsilon$, $L$, $\mathbf{M}$
3. **Diagnostics**: Detect problems early

**Typical warmup strategy**:
- Phase 1: Fast adaptation of $\epsilon$
- Phase 2: Estimate $\mathbf{M}$ from sample covariance
- Phase 3: Final tuning of $\epsilon$

### Handling Numerical Issues

**NaN/Inf detection**:
```python
if np.isnan(H_prop) or np.isinf(H_prop):
    return x, False  # Reject and continue
```

**Divergent transitions**: When $\Delta H$ is extremely large (e.g., > 1000), the trajectory has likely entered a problematic region. Log these for diagnostics.

**Gradient clipping** (use cautiously):
```python
grad = grad_log_prob(x)
grad_norm = np.linalg.norm(grad)
if grad_norm > max_grad:
    grad = grad * max_grad / grad_norm
```

### Multiple Chains

Running multiple chains in parallel:
- Enables convergence diagnostics (R-hat)
- Provides independent samples
- Utilizes multi-core hardware

```python
def sample_parallel(hmc, x0_list, n_samples, n_chains=4):
    from multiprocessing import Pool
    
    def sample_chain(args):
        x0, seed = args
        np.random.seed(seed)
        return hmc.sample(x0, n_samples)
    
    with Pool(n_chains) as pool:
        results = pool.map(sample_chain, 
                          [(x0_list[i], i) for i in range(n_chains)])
    
    return results
```

---

## Diagnostics

### Acceptance Rate

**Target**: 65-80% for standard HMC, ~80% for NUTS.

**Too low** (< 50%): Step size too large, energy errors too big.

**Too high** (> 95%): Step size too small, inefficient exploration.

### Energy Change Distribution

Plot the distribution of $\Delta H$ across iterations:
- Should be centered near 0
- Typical range: [-1, 1]
- Large outliers indicate problems

```python
delta_H_values = []
for i in range(n_samples):
    x, accepted, delta_H = hmc.step_with_diagnostics(x)
    delta_H_values.append(delta_H)

plt.hist(delta_H_values, bins=50)
plt.xlabel('ΔH')
plt.title('Energy Change Distribution')
```

### Divergent Transitions

A **divergent transition** occurs when the leapfrog trajectory encounters numerical instability, typically indicated by very large energy changes.

**Detection**:
```python
if delta_H > 1000:
    n_divergent += 1
    divergent_points.append(x_prop)
```

**Diagnosis**: Divergences often occur near:
- Boundaries of constrained parameters
- Regions of high curvature
- Multimodal structure

### Effective Sample Size

ESS measures the equivalent number of independent samples:

$$
\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^{\infty} \rho_k}
$$

where $\rho_k$ is the autocorrelation at lag $k$.

**Target**: ESS / second should be maximized.

---

## Variants and Extensions

### Partial Momentum Refreshment

Instead of fully resampling momentum, partially refresh:

$$
\mathbf{v}_{\text{new}} = \alpha \mathbf{v}_{\text{old}} + \sqrt{1 - \alpha^2} \boldsymbol{\eta}, \quad \boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})
$$

This maintains some of the previous direction, which can help for some targets.

### Randomized Trajectory Length

Instead of fixed $L$, sample $L \sim \text{Uniform}(1, L_{\max})$ or similar.

**Benefits**:
- Avoids periodic behavior that can cause poor mixing
- More robust to poor choice of $L$

### Windowed HMC

Accept not just the endpoint but any point along the trajectory, using appropriate acceptance probabilities.

### Riemannian HMC

Use position-dependent mass matrix $\mathbf{M}(\mathbf{x})$ that adapts to local geometry. More complex but can dramatically improve sampling for ill-conditioned targets.

---

## Comparison with Other MCMC

| Method | Gradient | Proposals | Acceptance | Mixing |
|--------|----------|-----------|------------|--------|
| Random Walk MH | No | Local | ~23% optimal | Slow (diffusive) |
| MALA | Yes | Local | ~57% optimal | Moderate |
| HMC | Yes | Distant | ~65-80% | Fast (ballistic) |
| Gibbs | No | Full conditional | 100% | Variable |

**HMC advantages**:
- Efficient exploration in high dimensions
- Utilizes gradient information
- Can take large steps while maintaining high acceptance

**HMC limitations**:
- Requires differentiable target
- Tuning can be challenging
- Struggles with multimodality

---

## Summary

| Component | Purpose |
|-----------|---------|
| Momentum resampling | Inject stochasticity, ensure ergodicity |
| Leapfrog integration | Propose distant states efficiently |
| Momentum negation | Ensure involution property |
| MH acceptance | Correct for discretization error |

The HMC algorithm achieves efficient sampling by:
1. Using gradient information for informed proposals
2. Taking long trajectories for distant proposals
3. Maintaining high acceptance through energy conservation
4. Preserving exactness through MH correction

---

## Exercises

1. **Implementation verification**. Implement HMC and verify it correctly samples from a known distribution (e.g., 2D Gaussian). Compare sample statistics to ground truth.

2. **Tuning experiment**. For a 10D Gaussian, plot acceptance rate and ESS/second as functions of $\epsilon$ and $L$. Find the optimal settings.

3. **Comparison with MALA**. Implement MALA and compare to HMC on the same target. How does ESS/gradient-evaluation compare?

4. **Divergence investigation**. Create a target with a funnel geometry (e.g., Neal's funnel). Run HMC and identify where divergences occur.

5. **Partial refreshment**. Implement partial momentum refreshment and compare to full refreshment on a target of your choice.

---

## References

1. Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D. (1987). "Hybrid Monte Carlo." *Physics Letters B*.
2. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
3. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
4. Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler." *JMLR*.
