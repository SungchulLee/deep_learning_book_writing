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
