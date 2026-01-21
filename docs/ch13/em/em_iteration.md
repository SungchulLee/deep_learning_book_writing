# The EM Iteration

The EM algorithm alternates between two steps: the E-step (Expectation) and the M-step (Maximization). This section provides complete derivations of both steps, proves the monotonic improvement guarantee, and interprets EM as coordinate ascent on the ELBO.

---

## E-Step (Expectation)

### Computing the Posterior over Latents

The E-step computes the **posterior distribution** over latent variables given the current parameter estimate $\theta^{(t)}$:

$$
q^{(t+1)}(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})
$$

Using Bayes' theorem:

$$
p(\mathbf{Z} | \mathbf{X}, \theta^{(t)}) = \frac{p(\mathbf{X}, \mathbf{Z} | \theta^{(t)})}{p(\mathbf{X} | \theta^{(t)})} = \frac{p(\mathbf{X} | \mathbf{Z}, \theta^{(t)}) \, p(\mathbf{Z} | \theta^{(t)})}{\int p(\mathbf{X}, \mathbf{Z}' | \theta^{(t)}) \, d\mathbf{Z}'}
$$

**Key Point**: The E-step requires computing the posterior, which involves the same integral that made direct optimization intractable. However, for many models (especially those in the exponential family), this posterior has a tractable closed form.

### Example: Gaussian Mixture Model

For a GMM with $K$ components, the latent variable $z_i \in \{1, \ldots, K\}$ indicates cluster membership for observation $\mathbf{x}_i$. The E-step computes the **responsibilities**:

$$
\gamma_{ik} = p(z_i = k | \mathbf{x}_i, \theta^{(t)}) = \frac{\pi_k^{(t)} \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{j=1}^{K} \pi_j^{(t)} \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}
$$

These responsibilities represent the soft assignment of each data point to each cluster.

### Making the Bound Tight

The E-step serves a crucial purpose: it makes the ELBO **tight** at the current parameter value $\theta^{(t)}$.

Recall the fundamental decomposition:

$$
\ell(\theta) = \mathcal{L}(q, \theta) + D_{\mathrm{KL}}\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)
$$

When we set $q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$:

$$
D_{\mathrm{KL}}\bigl(p(\mathbf{Z} | \mathbf{X}, \theta^{(t)}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})\bigr) = 0
$$

Therefore:

$$
\mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})
$$

**The ELBO equals the log-likelihood at $\theta^{(t)}$**—the bound is tight.

### Why Tightness Matters

Making the bound tight at the current point ensures that:

1. Any improvement in the ELBO translates to improvement in the log-likelihood
2. The algorithm doesn't get stuck at suboptimal points due to a loose bound
3. We have a precise starting point for the M-step optimization

---

## M-Step (Maximization)

### The Q-Function

The M-step maximizes the ELBO with respect to $\theta$, holding $q$ fixed at $q^{(t+1)}$. Since $q^{(t+1)} = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$, the ELBO becomes:

$$
\mathcal{L}(q^{(t+1)}, \theta) = \mathbb{E}_{p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})}[\log p(\mathbf{X}, \mathbf{Z} | \theta)] + H[q^{(t+1)}]
$$

The entropy term $H[q^{(t+1)}]$ does not depend on $\theta$, so maximizing $\mathcal{L}$ is equivalent to maximizing the **Q-function**:

$$
Q(\theta | \theta^{(t)}) = \mathbb{E}_{p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]
$$

This is the **expected complete-data log-likelihood**, where the expectation is taken over the posterior distribution of latent variables computed in the E-step.

### Q-Function Optimization

The M-step finds:

$$
\theta^{(t+1)} = \arg\max_\theta Q(\theta | \theta^{(t)})
$$

**Why Q is Easier to Optimize**:

1. **No integral over $\theta$**: The log is inside the expectation, not outside an integral
2. **Exponential family structure**: For exponential family models, $Q$ often has closed-form maximizers
3. **Decoupling**: Parameters often decouple, allowing separate optimization

### Expected Complete-Data Log-Likelihood

Expanding the Q-function:

$$
Q(\theta | \theta^{(t)}) = \int p(\mathbf{Z} | \mathbf{X}, \theta^{(t)}) \log p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z}
$$

For discrete latent variables:

$$
Q(\theta | \theta^{(t)}) = \sum_{\mathbf{Z}} p(\mathbf{Z} | \mathbf{X}, \theta^{(t)}) \log p(\mathbf{X}, \mathbf{Z} | \theta)
$$

### Example: GMM M-Step

For a Gaussian Mixture Model, the M-step updates are:

**Mixing proportions**:

$$
\pi_k^{(t+1)} = \frac{1}{N} \sum_{i=1}^{N} \gamma_{ik}
$$

**Means**:

$$
\boldsymbol{\mu}_k^{(t+1)} = \frac{\sum_{i=1}^{N} \gamma_{ik} \, \mathbf{x}_i}{\sum_{i=1}^{N} \gamma_{ik}}
$$

**Covariances**:

$$
\boldsymbol{\Sigma}_k^{(t+1)} = \frac{\sum_{i=1}^{N} \gamma_{ik} \, (\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})^\top}{\sum_{i=1}^{N} \gamma_{ik}}
$$

These are weighted versions of the standard maximum likelihood estimators, where the weights are the responsibilities from the E-step.

### Partial M-Step (Generalized EM)

In some cases, finding the global maximum of $Q(\theta | \theta^{(t)})$ is difficult. **Generalized EM (GEM)** only requires:

$$
Q(\theta^{(t+1)} | \theta^{(t)}) \geq Q(\theta^{(t)} | \theta^{(t)})
$$

Any improvement in $Q$ suffices—we don't need the global maximum. This is useful when:

- The M-step involves constrained optimization
- Closed-form solutions don't exist
- Gradient-based methods are used

---

## Monotonic Improvement Guarantee

### The Central Theorem

**Theorem (Monotonic Improvement)**: For any EM iteration, the log-likelihood never decreases:

$$
\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})
$$

with equality if and only if $\theta^{(t+1)} = \theta^{(t)}$ (i.e., we are at a fixed point).

### Proof

We establish this through a chain of inequalities:

$$
\ell(\theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})
$$

**Step 1 — ELBO is a Lower Bound**:

$$
\ell(\theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)})
$$

This holds because the ELBO is **always** a lower bound on the log-likelihood for any $q$ and any $\theta$:

$$
\ell(\theta) = \mathcal{L}(q, \theta) + D_{\mathrm{KL}}(q \| p(\mathbf{Z}|\mathbf{X}, \theta)) \geq \mathcal{L}(q, \theta)
$$

since $D_{\mathrm{KL}} \geq 0$.

**Step 2 — M-Step Improves ELBO**:

$$
\mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)})
$$

This holds by definition of the M-step: $\theta^{(t+1)}$ is chosen to **maximize** $\mathcal{L}(q^{(t+1)}, \theta)$ over $\theta$.

**Step 3 — E-Step Makes Bound Tight**:

$$
\mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})
$$

This holds because the E-step sets $q^{(t+1)} = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$, making the KL divergence zero:

$$
D_{\mathrm{KL}}(q^{(t+1)} \| p(\mathbf{Z}|\mathbf{X}, \theta^{(t)})) = 0
$$

**Combining all steps**:

$$
\ell(\theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})
$$

### Geometric Picture

```
Log-likelihood ℓ(θ)
         │
         │     ╱───────────────  ℓ(θ)
         │    ╱        ●──────────○  ℓ(θ^(t+1))
         │   ╱        ╱
         │  ╱    ●───●  ELBO at iteration t
         │ ╱    ╱   ↑
         │╱    ╱    M-step maximizes bound
         │    ● ────┘
         │   ↑ θ^(t)
         │   E-step makes bound tight here
         └────────────────────────────────────── θ
```

1. **E-step**: Construct a lower bound that touches $\ell(\theta)$ at $\theta^{(t)}$
2. **M-step**: Move to $\theta^{(t+1)}$ that maximizes the bound
3. At $\theta^{(t+1)}$, the bound may be loose, but $\ell(\theta^{(t+1)})$ is even higher

### Strict Improvement

If $\theta^{(t+1)} \neq \theta^{(t)}$ and the M-step achieves a strict improvement in the bound, then:

$$
\ell(\theta^{(t+1)}) > \ell(\theta^{(t)})
$$

This rules out cycles in the algorithm—EM either converges to a fixed point or strictly improves at each iteration.

### When Does Improvement Stop?

The sequence $\{\ell(\theta^{(t)})\}$ is monotonically increasing and bounded above (assuming the likelihood is proper). By the **monotone convergence theorem**, the sequence converges.

**Stationarity condition**: At convergence, $\theta^* = \theta^{(t)} = \theta^{(t+1)}$, meaning:

$$
\nabla_\theta Q(\theta | \theta^*)\big|_{\theta = \theta^*} = \nabla_\theta \mathcal{L}(q^*, \theta)\big|_{\theta = \theta^*} = 0
$$

This is a **necessary condition** for local optimality of $\ell(\theta)$, but not sufficient for global optimality—EM can converge to local maxima or saddle points.

---

## Coordinate Ascent Interpretation

### ELBO as Joint Objective

The EM algorithm can be understood as **coordinate ascent** on the functional $\mathcal{L}(q, \theta)$, which depends on both a distribution $q(\mathbf{Z})$ and parameters $\theta$.

### Two-Block Optimization

**E-step**: Maximize $\mathcal{L}(q, \theta)$ over $q$, holding $\theta = \theta^{(t)}$ fixed:

$$
q^{(t+1)} = \arg\max_q \mathcal{L}(q, \theta^{(t)})
$$

The solution is $q^{(t+1)} = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$, which sets $D_{\mathrm{KL}} = 0$.

**M-step**: Maximize $\mathcal{L}(q, \theta)$ over $\theta$, holding $q = q^{(t+1)}$ fixed:

$$
\theta^{(t+1)} = \arg\max_\theta \mathcal{L}(q^{(t+1)}, \theta)
$$

### Why Coordinate Ascent Works

Each step increases (or maintains) the ELBO:

$$
\mathcal{L}(q^{(t)}, \theta^{(t)}) \leq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) \leq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)})
$$

Since the ELBO lower-bounds the log-likelihood, and the E-step makes the bound tight, improvements in the ELBO translate to improvements in $\ell(\theta)$.

### Functional Optimization in E-Step

The E-step is a **functional optimization** problem: we optimize over the space of all distributions $q(\mathbf{Z})$. This is an infinite-dimensional optimization!

**Remarkably**, the solution has a closed form. Using calculus of variations or Lagrange multipliers:

$$
\frac{\delta}{\delta q(\mathbf{Z})} \left[ \mathcal{L}(q, \theta) - \lambda \left( \int q(\mathbf{Z}) d\mathbf{Z} - 1 \right) \right] = 0
$$

This yields:

$$
\log q^*(\mathbf{Z}) = \log p(\mathbf{X}, \mathbf{Z} | \theta) - \log p(\mathbf{X} | \theta)
$$

Thus $q^*(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta)$—the posterior distribution.

### Connection to Block Coordinate Descent

The coordinate ascent view connects EM to a broader class of optimization algorithms:

| Algorithm | Variables | Update Rule |
|-----------|-----------|-------------|
| EM | $(q, \theta)$ | Alternating maximization |
| Gibbs Sampling | $(z_1, \ldots, z_d)$ | Cycle through conditionals |
| ADMM | $(x, z, u)$ | Alternating with dual update |

All share the property of optimizing one block while holding others fixed.

### Implications of the Coordinate Ascent View

1. **Convergence**: Standard results for coordinate ascent apply—EM converges to a stationary point under mild regularity conditions

2. **Rate of Convergence**: The convergence rate depends on the curvature of $\mathcal{L}$ and the coupling between $q$ and $\theta$

3. **Generalizations**: This view motivates variants like:
   - **Partial E-step**: Don't fully optimize over $q$
   - **Partial M-step**: Don't fully optimize over $\theta$ (Generalized EM)
   - **Variational EM**: Restrict $q$ to a tractable family

4. **Connection to Variational Inference**: When the exact posterior is intractable, we can restrict $q$ to a variational family and still perform coordinate ascent—this is **variational inference**.

---

## Summary: The Complete EM Iteration

Given current parameters $\theta^{(t)}$:

### E-Step

1. Compute the posterior over latent variables:
   $$q^{(t+1)}(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$$

2. This makes the ELBO tight at $\theta^{(t)}$:
   $$\mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})$$

### M-Step

1. Define the Q-function:
   $$Q(\theta | \theta^{(t)}) = \mathbb{E}_{q^{(t+1)}}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]$$

2. Maximize to get new parameters:
   $$\theta^{(t+1)} = \arg\max_\theta Q(\theta | \theta^{(t)})$$

### Guarantees

- **Monotonic improvement**: $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$
- **Convergence**: The sequence $\{\theta^{(t)}\}$ converges to a stationary point
- **No cycles**: Either strict improvement or convergence at each step

### Algorithm Summary

```
Initialize θ⁽⁰⁾
repeat until convergence:
    # E-step: compute posterior
    q(Z) ← p(Z | X, θ⁽ᵗ⁾)
    
    # M-step: maximize expected complete-data log-likelihood
    θ⁽ᵗ⁺¹⁾ ← argmax_θ E_q[log p(X, Z | θ)]
    
    t ← t + 1
return θ⁽ᵗ⁾
```
