# EM Algorithm Foundations

The Expectation-Maximization (EM) algorithm is one of the most elegant and widely-used methods for maximum likelihood estimation in the presence of latent variables. This section develops the foundational concepts that motivate the algorithm.

---

## Latent Variable Models

### The Role of Latent Variables

Many probabilistic models posit the existence of **latent (hidden) variables** $\mathbf{Z}$ that are not directly observed but help explain the structure of the observed data $\mathbf{X}$. These latent variables serve several purposes:

1. **Dimensionality reduction**: Capturing low-dimensional structure in high-dimensional data
2. **Clustering**: Representing discrete group memberships
3. **Missing data**: Modeling unobserved portions of the data
4. **Hierarchical structure**: Encoding dependencies at multiple levels

### Formal Definition

A **latent variable model** specifies a joint distribution over observed variables $\mathbf{X}$ and latent variables $\mathbf{Z}$:

$$
p(\mathbf{X}, \mathbf{Z} | \theta)
$$

where $\theta$ represents the model parameters. The joint distribution factorizes as:

$$
p(\mathbf{X}, \mathbf{Z} | \theta) = p(\mathbf{X} | \mathbf{Z}, \theta) \, p(\mathbf{Z} | \theta)
$$

Here:

- $p(\mathbf{Z} | \theta)$ is the **prior** over latent variables
- $p(\mathbf{X} | \mathbf{Z}, \theta)$ is the **likelihood** of observations given latents

### Canonical Examples

**Gaussian Mixture Models (GMM)**: The latent variable $z_i \in \{1, \ldots, K\}$ indicates which of $K$ Gaussian components generated observation $\mathbf{x}_i$:

$$
p(z_i = k) = \pi_k, \quad p(\mathbf{x}_i | z_i = k) = \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

**Hidden Markov Models (HMM)**: The latent variables $\{z_1, \ldots, z_T\}$ form a Markov chain representing hidden states that generate the observed sequence $\{x_1, \ldots, x_T\}$.

**Factor Analysis**: Latent factors $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ generate observations through a linear transformation plus noise:

$$
\mathbf{x} = \mathbf{W}\mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi})
$$

### The Inference Problem

Given observed data, we face two fundamental tasks:

1. **Parameter estimation**: Find $\theta$ that maximizes the likelihood of observed data
2. **Posterior inference**: Compute $p(\mathbf{Z} | \mathbf{X}, \theta)$ for interpreting latent structure

These two tasks are intimately connected, and the EM algorithm elegantly addresses both.

---

## Marginal Log-Likelihood Intractability

### The Marginal Likelihood

Since latent variables are unobserved, we must work with the **marginal likelihood** obtained by integrating (or summing) over all possible values of $\mathbf{Z}$:

$$
p(\mathbf{X} | \theta) = \int p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z}
$$

For discrete latent variables, the integral becomes a sum:

$$
p(\mathbf{X} | \theta) = \sum_{\mathbf{Z}} p(\mathbf{X}, \mathbf{Z} | \theta)
$$

The **marginal log-likelihood** (also called the **incomplete data log-likelihood**) is:

$$
\ell(\theta) = \log p(\mathbf{X} | \theta) = \log \int p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z}
$$

### Why Direct Maximization Fails

Direct optimization of $\ell(\theta)$ is generally intractable for several reasons:

**1. The Log-Sum Problem**

The logarithm of an integral (or sum) does not simplify:

$$
\log \int p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z} \neq \int \log p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z}
$$

This prevents us from pushing the log inside and working with the simpler **complete data log-likelihood** $\log p(\mathbf{X}, \mathbf{Z} | \theta)$.

**2. High-Dimensional Integration**

The latent space may be enormous:

- In a GMM with $N$ data points and $K$ components, there are $K^N$ possible assignments
- In continuous latent variable models, the integral may be over millions of dimensions
- Even Monte Carlo approximations become prohibitively expensive

**3. No Closed-Form Solution**

For most models, the marginalization integral has no analytical solution. Even when it does (e.g., linear Gaussian models), the resulting expression may be complex.

**4. Coupled Parameters**

The parameters $\theta$ appear inside the integral in a complex way, making gradient computation difficult:

$$
\nabla_\theta \ell(\theta) = \nabla_\theta \log \int p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z}
$$

This requires computing expectations under $p(\mathbf{Z} | \mathbf{X}, \theta)$, which itself depends on $\theta$.

### Contrast with Complete Data

If we observed both $\mathbf{X}$ and $\mathbf{Z}$, we could work with the **complete data log-likelihood**:

$$
\ell_c(\theta) = \log p(\mathbf{X}, \mathbf{Z} | \theta)
$$

This is typically much easier to optimize because:

- The log applies directly to the joint probability
- For exponential family distributions, it often yields closed-form updates
- There is no integration over latent variables

The EM algorithm cleverly uses the tractability of complete data likelihood while acknowledging that $\mathbf{Z}$ is unobserved.

---

## The Optimization Problem

### Problem Statement

We observe data $\mathbf{X}$ and posit latent variables $\mathbf{Z}$. Our goal is to find the maximum likelihood estimate:

$$
\theta^* = \arg\max_\theta \ell(\theta) = \arg\max_\theta \log p(\mathbf{X} | \theta)
$$

Given the intractability of direct optimization, we need an alternative approach.

### Why Direct Optimization Fails

The integral (or sum) over the latent space makes direct optimization intractable for several interconnected reasons:

1. **High-dimensional integration**: The latent space may have exponentially many configurations or continuous dimensions
2. **No closed form**: The integral rarely has an analytical solution
3. **Coupling**: Parameters $\theta$ are entangled inside the integral, making gradients expensive

### The EM Strategy: Lower Bound Optimization

Instead of optimizing $\ell(\theta)$ directly, the EM algorithm constructs a **sequence of lower bounds** that are easier to optimize.

**Key Insight**: For any distribution $q(\mathbf{Z})$ over the latent variables, we can establish:

$$
\ell(\theta) \geq \mathcal{L}(q, \theta)
$$

where $\mathcal{L}(q, \theta)$ is the **Evidence Lower Bound (ELBO)**:

$$
\mathcal{L}(q, \theta) = \mathbb{E}_{q(\mathbf{Z})}[\log p(\mathbf{X}, \mathbf{Z} | \theta)] - \mathbb{E}_{q(\mathbf{Z})}[\log q(\mathbf{Z})]
$$

This can be rewritten as:

$$
\mathcal{L}(q, \theta) = \mathbb{E}_{q(\mathbf{Z})}[\log p(\mathbf{X}, \mathbf{Z} | \theta)] + H[q]
$$

where $H[q] = -\mathbb{E}_q[\log q(\mathbf{Z})]$ is the **entropy** of $q$.

### Iterative Optimization

The EM algorithm alternates between two steps. Given current parameters $\theta^{(t)}$:

**E-step (Expectation)**: Find the distribution $q$ that makes the bound tight at $\theta^{(t)}$

$$
q^{(t+1)}(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})
$$

**M-step (Maximization)**: Optimize the bound with respect to $\theta$

$$
\theta^{(t+1)} = \arg\max_\theta \mathcal{L}(q^{(t+1)}, \theta)
$$

**Repeat** until convergence.

### Monotonic Improvement Guarantee

This iterative approach guarantees that the log-likelihood never decreases:

$$
\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})
$$

**Proof sketch**:

1. After the E-step, $\mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})$ (the bound is tight)
2. The M-step ensures $\mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)})$
3. Since $\mathcal{L}$ is always a lower bound: $\ell(\theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)})$
4. Combining: $\ell(\theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})$

### Connection to Coordinate Ascent

The EM algorithm can be viewed as **coordinate ascent** on the functional $\mathcal{L}(q, \theta)$:

- **E-step**: Maximize over $q$ (holding $\theta$ fixed) — this is a functional optimization
- **M-step**: Maximize over $\theta$ (holding $q$ fixed) — this is a parameter optimization

Each step increases (or maintains) the ELBO, which in turn provides a non-decreasing sequence of log-likelihood values. This coordinate ascent perspective:

1. Explains why EM converges to a local optimum
2. Motivates generalized EM variants that only partially optimize each step
3. Connects EM to variational inference methods

### Geometric Interpretation

Geometrically, the EM algorithm can be understood as:

1. **E-step**: At current $\theta^{(t)}$, find the tangent plane to $\ell(\theta)$ that touches at $\theta^{(t)}$
2. **M-step**: Move to the maximum of this tangent plane

Since the ELBO is a lower bound that touches $\ell(\theta)$ at $\theta^{(t)}$, optimizing the ELBO guarantees improvement in the true objective.

---

## Summary

The EM algorithm addresses a fundamental challenge in statistical learning: maximizing likelihood when some variables are unobserved. The key insights are:

1. **Latent variable models** provide powerful representations but create optimization challenges
2. **Marginal likelihood** is intractable due to integration over latent space
3. **Lower bound optimization** via ELBO provides a tractable alternative
4. **Coordinate ascent** on $(q, \theta)$ guarantees monotonic improvement

These foundations set the stage for deriving the ELBO, understanding the E-step and M-step in detail, and analyzing convergence properties.

---

# Appendix: Convergence Theory

# Convergence Theory

Understanding when and how fast the EM algorithm converges is essential for practitioners. This section presents Wu's classical convergence theorem, conditions for global optimality, the role of exponential family structure, and a detailed analysis of convergence rates via the missing information principle.

---

## Wu's Convergence Theorem

### The Classical Result

Wu (1983) provided rigorous sufficient conditions for EM convergence. These conditions form the theoretical foundation for understanding when EM works and remain the standard reference for EM convergence analysis.

### Sufficient Conditions

**Condition A (Continuity)**: The mappings $\theta \mapsto Q(\theta | \theta')$ and $\theta' \mapsto Q(\theta | \theta')$ are continuous, where:

$$
Q(\theta | \theta') = \mathbb{E}_{p(\mathbf{Z}|\mathbf{X}, \theta')}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]
$$

This ensures the Q-function behaves smoothly as we vary either argument.

**Condition B (M-step Well-Defined)**: For each $\theta^{(t)}$, the M-step yields a unique solution:

$$
\theta^{(t+1)} = M(\theta^{(t)}) = \arg\max_{\theta} Q(\theta | \theta^{(t)})
$$

and the mapping $M: \Theta \to \Theta$ is continuous. This requires the M-step optimization problem to have a unique, continuously-varying solution.

**Condition C (Compactness)**: The sequence $\{\theta^{(t)}\}$ remains in a compact subset of the parameter space $\Theta$. This prevents parameters from diverging to infinity.

### Main Theorem

**Theorem (Wu, 1983)**: Under Conditions A, B, and C:

1. The sequence $\{\ell(\theta^{(t)})\}$ converges monotonically
2. **Every limit point** of $\{\theta^{(t)}\}$ is a **stationary point** of $\ell(\theta)$

### What Wu's Theorem Guarantees

- The algorithm terminates (in the limit) at a stationary point
- No oscillation or divergence occurs
- The likelihood improves at every step

### What Wu's Theorem Does NOT Guarantee

- Convergence to a **global** maximum
- Convergence to even a **local** maximum (could be a saddle point)
- A unique limit point (the sequence might have multiple accumulation points)

### Practical Verification

| Condition | Typically Verified By |
|-----------|----------------------|
| Continuity | Smooth likelihood (most exponential families) |
| Compactness | Bounded parameter space or regularization |
| Unique M-step | Strictly concave $Q$-function in $\theta$ |

For most well-specified models (Gaussian mixtures, HMMs with proper priors), Wu's conditions hold, ensuring convergence to *some* stationary point—though not necessarily the global optimum.

---

## Sufficient Conditions for Global Optimum

### The Challenge

Wu's theorem guarantees convergence to a **stationary point**, but practitioners need the **global** maximum. Under what conditions does EM find it?

### Condition 1: Strictly Concave Log-Likelihood

If $\ell(\theta) = \log p(\mathbf{X} | \theta)$ is **strictly concave** in $\theta$:

- There exists a unique global maximum $\theta^*$
- Any stationary point is the global maximum
- EM's monotonic improvement guarantees convergence to $\theta^*$

**Reality check**: Strict concavity of the marginal log-likelihood is rare in latent variable models. The marginalization over $\mathbf{Z}$ typically destroys concavity even when the complete-data log-likelihood is well-behaved.

### Condition 2: Unimodality

If $\ell(\theta)$ has a **unique stationary point** (which must then be the global maximum):

$$
\nabla_\theta \ell(\theta) = 0 \implies \theta = \theta^*
$$

Combined with Wu's conditions, EM converges to the global optimum.

### Condition 3: Contraction Mapping

Define the EM mapping $M: \theta^{(t)} \mapsto \theta^{(t+1)}$. If:

1. $\theta^*$ is the unique fixed point: $M(\theta^*) = \theta^*$
2. $M$ is a **contraction**: $\|M(\theta) - M(\theta')\| \leq \rho \|\theta - \theta'\|$ for some $\rho < 1$

Then by **Banach's fixed-point theorem**:

- EM converges globally to $\theta^*$ from **any** initialization
- Convergence is geometric with rate $\rho$

**Verifying Contraction**: Near a fixed point $\theta^*$, linearize:

$$
M(\theta) \approx M(\theta^*) + J_M(\theta^*)(\theta - \theta^*)
$$

where $J_M$ is the Jacobian of $M$. The algorithm is a local contraction if:

$$
\rho(J_M(\theta^*)) < 1
$$

where $\rho(\cdot)$ denotes spectral radius.

### Condition 4: Q-Function Concavity + Likelihood Unimodality

**Theorem (Dempster, Laird, Rubin, 1977)**: If for all $\theta'$:

1. The function $Q(\theta | \theta')$ is **strictly concave** in $\theta$
2. The marginal log-likelihood $\ell(\theta)$ is **unimodal**

Then EM converges to the global maximum.

This is the most practically useful condition for models like single-component Gaussian with missing data and some simple exponential family models.

### Summary of Global Convergence Conditions

| Condition | Guarantees |
|-----------|------------|
| Wu's conditions (continuity, compactness) | Convergence to stationary point |
| $\ell(\theta)$ strictly concave | Global optimum |
| $\ell(\theta)$ unimodal | Global optimum |
| $M$ is a contraction | Global optimum + geometric rate |
| $Q$ strictly concave + $\ell$ unimodal | Global optimum |

---

## Exponential Family Structure

### Complete-Data Exponential Family

When the complete-data likelihood belongs to an **exponential family**:

$$
p(\mathbf{X}, \mathbf{Z} | \theta) = h(\mathbf{X}, \mathbf{Z}) \exp\bigl(\eta(\theta)^\top T(\mathbf{X}, \mathbf{Z}) - A(\theta)\bigr)
$$

where:

- $\eta(\theta)$ is the natural parameter
- $T(\mathbf{X}, \mathbf{Z})$ is the sufficient statistic
- $A(\theta)$ is the log-partition function
- $h(\mathbf{X}, \mathbf{Z})$ is the base measure

### Q-Function Simplification

The Q-function becomes:

$$
Q(\theta | \theta^{(t)}) = \eta(\theta)^\top \mathbb{E}_{p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})}[T(\mathbf{X}, \mathbf{Z})] - A(\theta) + \text{const}
$$

**Key insight**: The E-step only needs to compute $\mathbb{E}[T(\mathbf{X}, \mathbf{Z})]$—the expected sufficient statistics. This is often tractable even when the full posterior is complex.

### Concavity of Q-Function

Since $A(\theta)$ is **convex** (a fundamental property of exponential families), the Q-function is **concave** in $\theta$. This ensures:

- The M-step has a unique solution
- The M-step often has a closed-form solution
- But does **not** guarantee global optimality of EM overall

### Why Exponential Family Structure Helps

1. **E-step simplification**: Only expected sufficient statistics needed
2. **M-step tractability**: Concave optimization with often closed-form solutions
3. **Natural parameterization**: Well-understood optimization landscape
4. **Conjugate priors**: Bayesian extensions are straightforward

### Canonical Examples

| Model | Sufficient Statistics | M-step Updates |
|-------|----------------------|----------------|
| GMM | $\sum_i \gamma_{ik}$, $\sum_i \gamma_{ik} x_i$, $\sum_i \gamma_{ik} x_i x_i^\top$ | Weighted MLEs |
| HMM | Expected transition counts, expected emission counts | Normalized counts |
| Factor Analysis | $\mathbb{E}[\mathbf{z}]$, $\mathbb{E}[\mathbf{z}\mathbf{z}^\top]$ | Linear algebra |

---

## Local vs Global Convergence

### The Multimodality Problem

In mixture models (the canonical EM application), $\ell(\theta)$ is typically **multimodal** due to:

**1. Label Switching Symmetry**: Permuting component labels gives identical likelihood. A $K$-component mixture has at least $K!$ equivalent global maxima.

For a GMM with $K=3$ components, the solutions $(\mu_1, \mu_2, \mu_3)$, $(\mu_2, \mu_1, \mu_3)$, $(\mu_3, \mu_2, \mu_1)$, etc., all achieve the same likelihood.

**2. Spurious Local Maxima**: Components can collapse or spread pathologically, creating local maxima that are not permutations of the global solution:

- **Component collapse**: A component shrinks to fit a single point ($\Sigma_k \to 0$)
- **Component absorption**: One component absorbs another
- **Suboptimal splits**: Data is partitioned suboptimally

### Landscape Characterization

Near the global maximum, the likelihood surface typically has:

- A **basin of attraction** around each equivalent global maximum
- **Saddle points** separating different basins
- **Local maxima** corresponding to suboptimal solutions

### Strategies for Finding Global Optimum

**1. Multiple Random Initializations**

Run EM from many starting points and select the solution with highest likelihood:

```
best_likelihood = -∞
for i = 1 to num_restarts:
    θ⁽⁰⁾ ← random_initialization()
    θ* ← run_EM(θ⁽⁰⁾)
    if ℓ(θ*) > best_likelihood:
        best_likelihood = ℓ(θ*)
        best_θ = θ*
return best_θ
```

**2. Smart Initialization**

- **K-means++**: Initialize GMM means using k-means++ algorithm
- **Hierarchical**: Start with fewer components and split
- **Spectral methods**: Use eigendecomposition for initialization

**3. Deterministic Annealing**

Introduce a temperature parameter and anneal from high to low temperature:

$$
Q_T(\theta | \theta^{(t)}) = \frac{1}{T} \mathbb{E}_{p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]
$$

At high $T$, the landscape is smoothed; as $T \to 1$, we recover standard EM.

**4. Hybrid Methods**

Combine EM with global optimization:

- Start with simulated annealing or genetic algorithms
- Refine with EM once near a good solution

### Practical Recommendations

| Scenario | Recommendation |
|----------|----------------|
| Few components ($K \leq 5$) | 10-50 random restarts |
| Many components ($K > 10$) | Smart initialization + few restarts |
| High dimensions | Dimensionality reduction first |
| Time-constrained | Single smart initialization |

---

## Rate of Convergence

### The Missing Information Principle

The rate of EM convergence is governed by how much information the latent variables carry relative to the complete data. This elegant principle, due to Dempster, Laird, and Rubin (1977) and formalized by Louis (1982), provides deep insight into when EM is fast or slow.

### Information Matrices

Define three key quantities at a stationary point $\theta^*$:

**Observed Information**:

$$
I_{\text{obs}}(\theta) = -\nabla^2_\theta \ell(\theta) = -\frac{\partial^2 \log p(\mathbf{X}|\theta)}{\partial \theta \partial \theta^\top}
$$

This is the curvature of the marginal log-likelihood—what we can learn from observed data alone.

**Complete Information**:

$$
I_{\text{comp}}(\theta) = -\mathbb{E}_{p(\mathbf{Z}|\mathbf{X}, \theta)}[\nabla^2_\theta \log p(\mathbf{X}, \mathbf{Z} | \theta)]
$$

This is the expected curvature of the complete-data log-likelihood—what we could learn if we observed both $\mathbf{X}$ and $\mathbf{Z}$.

**Missing Information**:

$$
I_{\text{miss}}(\theta) = I_{\text{comp}}(\theta) - I_{\text{obs}}(\theta)
$$

This is the information "lost" by not observing the latent variables.

### The Fundamental Identity

Louis (1982) established:

$$
\boxed{I_{\text{obs}}(\theta) = I_{\text{comp}}(\theta) - I_{\text{miss}}(\theta)}
$$

This is not merely a definition but a deep identity. It can be derived by differentiating the log-likelihood and using properties of the posterior distribution.

### Convergence Rate Matrix

Near a local maximum $\theta^*$, the EM iteration satisfies:

$$
\theta^{(t+1)} - \theta^* \approx J(\theta^*)(\theta^{(t)} - \theta^*)
$$

where the **rate matrix** is:

$$
J = I_{\text{comp}}(\theta^*)^{-1} I_{\text{miss}}(\theta^*)
$$

This matrix governs how quickly the error $\theta^{(t)} - \theta^*$ shrinks.

### Spectral Analysis

The **convergence rate** is determined by the spectral radius:

$$
\rho = \rho(J) = \max_i |\lambda_i(J)|
$$

where $\lambda_i$ are eigenvalues of $J$.

**Geometric Convergence**: Near the optimum:

$$
\|\theta^{(t)} - \theta^*\| \sim \rho^t \|\theta^{(0)} - \theta^*\|
$$

### Fast vs Slow Convergence

**Fast Convergence ($\rho \ll 1$)**:

When $I_{\text{miss}}$ is small relative to $I_{\text{comp}}$:

- The latent variables carry little additional information
- EM behaves almost like direct optimization
- Few iterations needed

**Slow Convergence ($\rho \approx 1$)**:

When $I_{\text{miss}} \approx I_{\text{comp}}$:

- The latent variables dominate the information content
- EM can be painfully slow
- Many iterations needed for small improvements

### Intuitive Interpretation

Consider the **fraction of observed information**:

$$
r = \frac{I_{\text{obs}}}{I_{\text{comp}}} = I - J
$$

| Observed Fraction | Rate Matrix Eigenvalues | Convergence |
|-------------------|------------------------|-------------|
| $r = 0.9$ (90% observed) | $\rho \approx 0.1$ | Fast |
| $r = 0.5$ (50% observed) | $\rho \approx 0.5$ | Moderate |
| $r = 0.1$ (10% observed) | $\rho \approx 0.9$ | Slow |

### Practical Implications

**High Missing Information (Slow EM)**:

- Many mixture components with overlapping distributions
- High noise levels
- Many latent dimensions relative to observed
- Weak signal-to-noise ratio

**Low Missing Information (Fast EM)**:

- Well-separated clusters
- Low noise
- Informative observations
- Strong signal

### Example: Gaussian Mixture

For a two-component Gaussian mixture with separation $\delta$ (distance between means in units of standard deviation):

- **Large $\delta$** (well-separated): Responsibilities are nearly 0 or 1, $I_{\text{miss}}$ is small, fast convergence
- **Small $\delta$** (overlapping): Responsibilities are uncertain, $I_{\text{miss}}$ is large, slow convergence

Quantitatively, for equal-variance components:

$$
\rho \approx 1 - \Phi(\delta/2)
$$

where $\Phi$ is the standard normal CDF. At $\delta = 4$, $\rho \approx 0.02$ (fast); at $\delta = 1$, $\rho \approx 0.69$ (slow).

### Acceleration Strategies

When EM is slow ($\rho$ near 1), consider:

**1. Aitken Acceleration**: Extrapolate the limit from observed convergence rate:

$$
\theta^* \approx \theta^{(t)} + \frac{\theta^{(t+1)} - \theta^{(t)}}{1 - \rho}
$$

**2. SQUAREM**: Squared iterative methods that effectively square the EM mapping, reducing $\rho$ to $\rho^2$.

**3. Quasi-Newton Methods**: Use approximate Hessian information to accelerate:

$$
\theta^{(t+1)} = \theta^{(t)} - H^{-1} \nabla \ell(\theta^{(t)})
$$

**4. Parameter Expansion (PX-EM)**: Artificially expand parameter space to improve conditioning, then project back.

**5. ECME (Expectation/Conditional Maximization Either)**: Replace some M-steps with direct likelihood maximization.

---

## Summary

| Aspect | Key Result |
|--------|------------|
| **Wu's Theorem** | Continuity + compactness → convergence to stationary point |
| **Global Convergence** | Requires concavity, unimodality, or contraction |
| **Exponential Family** | Q-function is concave; M-step often closed-form |
| **Local vs Global** | Multimodality requires multiple restarts |
| **Convergence Rate** | $\rho = \rho(I_{\text{comp}}^{-1} I_{\text{miss}})$ |
| **Missing Information** | High → slow; Low → fast |

### Key References

- Wu, C. F. J. (1983). On the convergence properties of the EM algorithm. *The Annals of Statistics*, 11(1), 95-103.
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *JRSS-B*, 44(2), 226-233.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS-B*, 39(1), 1-38.
- Meng, X. L., & Rubin, D. B. (1993). Maximum likelihood estimation via the ECM algorithm. *Biometrika*, 80(2), 267-278.
