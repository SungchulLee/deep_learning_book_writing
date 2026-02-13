# Mass Matrix

The mass matrix $\mathbf{M}$ is a critical tuning parameter in HMC that controls how momentum translates to velocity. This section covers the role of the mass matrix, its geometric interpretation, estimation strategies, and the connection to preconditioning.

---

## Role of the Mass Matrix

### Definition and Basic Mechanics

The mass matrix $\mathbf{M}$ appears in the kinetic energy:

$$
K(\mathbf{v}) = \frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}
$$

and in Hamilton's equations:

$$
\frac{d\mathbf{x}}{dt} = \mathbf{M}^{-1}\mathbf{v}, \quad \frac{d\mathbf{v}}{dt} = -\nabla U(\mathbf{x})
$$

The mass matrix determines:
1. **Momentum distribution**: $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})$
2. **Velocity from momentum**: $\dot{\mathbf{x}} = \mathbf{M}^{-1}\mathbf{v}$
3. **Effective step sizes** in different directions

### Physical Interpretation

In classical mechanics, mass determines inertia—how much an object resists acceleration. Similarly:

- **Large mass** in direction $i$: Slow response to forces, small velocity for given momentum
- **Small mass** in direction $i$: Fast response, large velocity for given momentum

For sampling, we want the "mass" to match the "stiffness" of the potential in each direction.

### The Identity Mass Matrix

The simplest choice $\mathbf{M} = \mathbf{I}$ gives:

$$
K(\mathbf{v}) = \frac{1}{2}|\mathbf{v}|^2, \quad \mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

This works well when the target distribution has similar scales in all directions. It fails when:
- Variances differ greatly across dimensions
- Strong correlations exist between variables
- The posterior is "narrow" in some directions

---

## Geometric Interpretation

### Metric on Position Space

The inverse mass matrix $\mathbf{M}^{-1}$ defines a **Riemannian metric** on position space. The kinetic energy becomes:

$$
K(\mathbf{v}) = \frac{1}{2}\|\mathbf{v}\|_{\mathbf{M}^{-1}}^2
$$

where $\|\mathbf{v}\|_{\mathbf{M}^{-1}}^2 = \mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}$ is the squared norm under this metric.

### Momentum Space Geometry

The mass matrix $\mathbf{M}$ defines the metric on momentum space:

$$
\|\mathbf{p}\|_{\mathbf{M}}^2 = \mathbf{p}^T \mathbf{M} \mathbf{p}
$$

Position and momentum spaces are **dual** under these metrics.

### Energy Surfaces

For a quadratic potential $U(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T \mathbf{A} \mathbf{x}$, energy surfaces are ellipsoids:

$$
H = \frac{1}{2}\mathbf{x}^T \mathbf{A} \mathbf{x} + \frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v} = E
$$

**Optimal choice**: When $\mathbf{M} = \mathbf{A}^{-1}$, the energy surfaces become spherical in appropriately scaled coordinates, and the dynamics become isotropic.

---

## The Matching Principle

### Why Matching Matters

Consider a 2D target where $x_1$ has variance 1 and $x_2$ has variance 0.01:

**With $\mathbf{M} = \mathbf{I}$**:
- Momentum in both directions has variance 1
- Velocity $\dot{x}_2 = v_2$ is "too fast" for the narrow $x_2$ direction
- Large step size causes instability in $x_2$
- Small step size causes slow exploration in $x_1$

**With $\mathbf{M} = \text{diag}(1, 100)$**:
- Momentum variance in $x_2$ is 100, but velocity $\dot{x}_2 = v_2/100$ is small
- Effective step sizes match the scales
- Exploration is balanced

### The Optimal Mass Matrix

**Theorem**: For a Gaussian target $\pi(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, the optimal mass matrix is:

$$
\mathbf{M}^* = \boldsymbol{\Sigma}^{-1}
$$

**Why?** With this choice:
1. The Hamiltonian becomes $H = \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) + \frac{1}{2}\mathbf{v}^T\boldsymbol{\Sigma}\mathbf{v}$
2. In standardized coordinates $\mathbf{z} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{x} - \boldsymbol{\mu})$, both terms are isotropic
3. Trajectories explore all directions equally efficiently

### For Non-Gaussian Targets

For general targets, set:

$$
\mathbf{M} \approx \text{Cov}[\mathbf{x}]^{-1} = \mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T]^{-1}
$$

This is the **inverse posterior covariance**, estimated from samples.

---

## Types of Mass Matrices

### Scalar (Isotropic)

$$
\mathbf{M} = m \mathbf{I}
$$

- **1 parameter** to tune
- Assumes all directions have equal scale
- Rarely appropriate in practice

### Diagonal

$$
\mathbf{M} = \text{diag}(m_1, \ldots, m_d)
$$

- **$d$ parameters** to tune
- Adapts to different marginal variances
- Ignores correlations
- **Default choice** in most software (Stan, PyMC)

**Estimation**: $m_i = 1/\text{Var}[x_i]$, estimated from warmup samples.

### Full (Dense)

$$
\mathbf{M} = \text{any positive definite matrix}
$$

- **$d(d+1)/2$ parameters**
- Captures correlations
- More expensive: $O(d^2)$ storage, $O(d^3)$ for Cholesky
- Can dramatically improve sampling for correlated targets

**Estimation**: $\mathbf{M} = \text{Cov}[\mathbf{x}]^{-1}$, but requires many samples to estimate well.

### Comparison

| Type | Parameters | Captures Correlations | Cost per Step | Best For |
|------|------------|----------------------|---------------|----------|
| Scalar | 1 | No | $O(d)$ | Isotropic targets |
| Diagonal | $d$ | No | $O(d)$ | Different scales |
| Full | $d(d+1)/2$ | Yes | $O(d^2)$ | Correlated targets |

---

## Adaptation During Warmup

### The Adaptation Problem

We need $\mathbf{M} \approx \text{Cov}[\mathbf{x}]^{-1}$, but we don't know $\text{Cov}[\mathbf{x}]$ until we've sampled. This chicken-and-egg problem is solved by **adaptive warmup**.

### Warmup Strategy

**Phase 1: Initial exploration** (iterations 1–75)
- Use $\mathbf{M} = \mathbf{I}$
- Adapt step size $\epsilon$ aggressively
- Goal: Find the typical set

**Phase 2: Covariance estimation** (iterations 76–900)
- Estimate $\hat{\boldsymbol{\Sigma}}$ from samples
- Update $\mathbf{M} = \hat{\boldsymbol{\Sigma}}^{-1}$ periodically
- Continue adapting $\epsilon$

**Phase 3: Final tuning** (iterations 901–1000)
- Fix $\mathbf{M}$
- Final adaptation of $\epsilon$
- Verify stability

**Sampling phase**: (iterations 1001+)
- No more adaptation
- Collect samples for inference

### Welford's Online Algorithm

For efficient online covariance estimation:

```python
class OnlineCovariance:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros((dim, dim))
    
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)
    
    def covariance(self):
        if self.n < 2:
            return np.eye(len(self.mean))
        return self.M2 / (self.n - 1)
```

### Regularization

With few samples, $\hat{\boldsymbol{\Sigma}}$ may be poorly conditioned or singular. Regularize:

$$
\hat{\boldsymbol{\Sigma}}_{\text{reg}} = (1 - \alpha)\hat{\boldsymbol{\Sigma}} + \alpha \cdot \text{diag}(\hat{\boldsymbol{\Sigma}})
$$

or add a ridge:

$$
\hat{\boldsymbol{\Sigma}}_{\text{reg}} = \hat{\boldsymbol{\Sigma}} + \lambda \mathbf{I}
$$

---

## Implementation Details

### Cholesky Factorization

For efficient computation, store the Cholesky factor $\mathbf{L}$ where $\mathbf{M} = \mathbf{L}\mathbf{L}^T$:

**Sampling momentum**:
```python
v = L @ np.random.randn(d)  # v ~ N(0, M)
```

**Computing kinetic energy**:
```python
z = solve_triangular(L, v, lower=True)  # L z = v
K = 0.5 * np.dot(z, z)  # K = ½ vᵀ M⁻¹ v = ½ |z|²
```

**Velocity from momentum**:
```python
# M⁻¹ v = (L Lᵀ)⁻¹ v = L⁻ᵀ L⁻¹ v
z = solve_triangular(L, v, lower=True)
velocity = solve_triangular(L.T, z, lower=False)
```

### For Diagonal Mass Matrix

When $\mathbf{M} = \text{diag}(m_1, \ldots, m_d)$:

```python
# Store diagonal and its square root
m_diag = np.array([m1, m2, ..., md])
sqrt_m = np.sqrt(m_diag)
inv_m = 1.0 / m_diag

# Sample momentum
v = sqrt_m * np.random.randn(d)

# Kinetic energy
K = 0.5 * np.sum(v**2 * inv_m)

# Velocity
velocity = v * inv_m
```

### Numerical Stability

When inverting the estimated covariance:

```python
def safe_inverse(Sigma, min_var=1e-6):
    """Safely invert covariance matrix."""
    # Ensure minimum variance
    Sigma = Sigma.copy()
    np.fill_diagonal(Sigma, np.maximum(np.diag(Sigma), min_var))
    
    # Use pseudo-inverse for near-singular matrices
    try:
        M = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        M = np.linalg.pinv(Sigma)
    
    # Ensure symmetry
    M = 0.5 * (M + M.T)
    
    return M
```

---

## Connection to Preconditioning

### HMC as Preconditioned Gradient Descent

The leapfrog momentum update is:

$$
\mathbf{v}_{n+1/2} = \mathbf{v}_n + \frac{\epsilon}{2}\nabla \log \pi(\mathbf{x}_n)
$$

The position update is:

$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \epsilon \mathbf{M}^{-1}\mathbf{v}_{n+1/2}
$$

Combining (approximately, for one step):

$$
\mathbf{x}_{n+1} \approx \mathbf{x}_n + \frac{\epsilon^2}{2}\mathbf{M}^{-1}\nabla \log \pi(\mathbf{x}_n)
$$

This is **preconditioned gradient ascent** with preconditioner $\mathbf{M}^{-1}$.

### Optimal Preconditioning

For optimization, the optimal preconditioner is the inverse Hessian (Newton's method):

$$
\mathbf{P}^* = (-\nabla^2 \log \pi)^{-1} = \boldsymbol{\Sigma}
$$

For HMC sampling, this suggests $\mathbf{M}^{-1} = \boldsymbol{\Sigma}$, i.e., $\mathbf{M} = \boldsymbol{\Sigma}^{-1}$.

### Condition Number

The **condition number** of the preconditioned system determines convergence:

$$
\kappa = \frac{\lambda_{\max}(\mathbf{M}^{-1}\mathbf{A})}{\lambda_{\min}(\mathbf{M}^{-1}\mathbf{A})}
$$

where $\mathbf{A} = -\nabla^2 \log \pi$ is the Hessian.

- **Without preconditioning** ($\mathbf{M} = \mathbf{I}$): $\kappa = \lambda_{\max}(\mathbf{A})/\lambda_{\min}(\mathbf{A})$
- **With optimal preconditioning** ($\mathbf{M} = \mathbf{A}$): $\kappa = 1$

Lower condition number → faster mixing.

---

## Riemannian HMC (Preview)

### Position-Dependent Mass

Standard HMC uses constant $\mathbf{M}$. **Riemannian HMC** allows $\mathbf{M}(\mathbf{x})$ to vary with position:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + \frac{1}{2}\mathbf{v}^T\mathbf{M}(\mathbf{x})^{-1}\mathbf{v} + \frac{1}{2}\log|\mathbf{M}(\mathbf{x})|
$$

The extra log-determinant term is required for the correct marginal.

### Benefits

- Adapts to local geometry
- Can handle varying curvature
- Particularly useful for funnel-shaped distributions

### Challenges

- Hamiltonian is no longer separable
- Leapfrog doesn't directly apply
- Requires implicit or generalized integrators
- More expensive per step

---

## Practical Recommendations

### Default Strategy

1. **Start with diagonal**: Use $\mathbf{M} = \text{diag}(1/\hat{\sigma}_1^2, \ldots, 1/\hat{\sigma}_d^2)$
2. **Adapt during warmup**: Estimate marginal variances from samples
3. **Consider full matrix** if:
   - Strong correlations expected
   - Diagonal adaptation gives poor results
   - Dimension is moderate ($d < 100$)

### When to Use Full Mass Matrix

✓ **Use full matrix when**:
- Parameters are highly correlated (|ρ| > 0.8)
- Dimension is not too large (d < 100-200)
- You have enough warmup samples (> 100d)

✗ **Stick with diagonal when**:
- High dimension (d > 200)
- Parameters are approximately independent
- Limited computational budget
- Warmup samples are few

### Diagnostics

**Check adaptation worked**:
```python
# After warmup
estimated_cov = np.cov(warmup_samples.T)
M_inv = np.linalg.inv(M)

# Should be approximately identity
whitened = estimated_cov @ M  # ≈ I if well-adapted
print("Whitening check:", np.diag(whitened))  # Should be ≈ 1
```

**Check for poor conditioning**:
```python
cond_number = np.linalg.cond(M)
if cond_number > 1e6:
    print("Warning: Mass matrix poorly conditioned")
```

---

## Summary

| Aspect | Recommendation |
|--------|----------------|
| **Role** | Preconditions dynamics to match target geometry |
| **Optimal choice** | $\mathbf{M} = \text{Cov}[\mathbf{x}]^{-1}$ |
| **Default** | Diagonal, adapted during warmup |
| **Full matrix** | When correlations are strong and $d$ is moderate |
| **Adaptation** | Estimate from warmup samples using online algorithm |
| **Regularization** | Add ridge or blend with diagonal for stability |

The mass matrix transforms HMC from a method that struggles with ill-conditioned targets to one that can efficiently explore complex posterior geometries. Proper adaptation is essential for practical performance.

---

## Exercises

1. **Effect of mass matrix**. Sample from a 2D Gaussian with covariance $\begin{pmatrix} 1 & 0.9 \\ 0.9 & 1 \end{pmatrix}$ using (a) $\mathbf{M} = \mathbf{I}$, (b) diagonal $\mathbf{M}$, (c) full $\mathbf{M} = \boldsymbol{\Sigma}^{-1}$. Compare acceptance rates and ESS.

2. **Online adaptation**. Implement warmup with online covariance estimation. Plot how the estimated covariance converges to the true covariance over iterations.

3. **Condition number experiment**. For targets with varying condition numbers (1, 10, 100, 1000), compare HMC performance with and without mass matrix adaptation.

4. **High-dimensional diagonal**. For a 100-dimensional target with different marginal variances, implement diagonal mass matrix adaptation and verify it improves sampling efficiency.

5. **Regularization comparison**. Compare different regularization strategies (ridge, shrinkage toward diagonal) for mass matrix estimation with limited samples.

---

## References

1. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
2. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
3. Girolami, M., & Calderhead, B. (2011). "Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods." *JRSS-B*.
4. Stan Development Team. "Stan Reference Manual: HMC Algorithm Parameters."
