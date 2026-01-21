# Evidence Lower Bound (ELBO) for EM

The Evidence Lower Bound (ELBO) is the mathematical foundation of the EM algorithm. This section presents two complementary derivations—via Jensen's inequality and via KL divergence decomposition—and analyzes the gap between the bound and the true likelihood.

---

## Jensen's Inequality Approach

### The Core Setup

For any distribution $q(\mathbf{Z})$ over the latent variables, we can rewrite the marginal log-likelihood by introducing $q$ through multiplication by $1 = q(\mathbf{Z})/q(\mathbf{Z})$:

$$
\ell(\theta) = \log p(\mathbf{X} | \theta) = \log \int p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z} = \log \int q(\mathbf{Z}) \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})} \, d\mathbf{Z}
$$

This expression has the form $\log \mathbb{E}_q[f(\mathbf{Z})]$ where $f(\mathbf{Z}) = \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})}$.

### Jensen's Inequality

**Theorem (Jensen's Inequality)**: For a concave function $\phi$ and a random variable $Y$:

$$
\phi(\mathbb{E}[Y]) \geq \mathbb{E}[\phi(Y)]
$$

For convex functions, the inequality reverses. Since $\log$ is **strictly concave**, we have:

$$
\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]
$$

with equality if and only if $Y$ is constant (i.e., $\text{Var}(Y) = 0$).

### Applying Jensen's Inequality

Applying Jensen's inequality to our expression:

$$
\ell(\theta) = \log \mathbb{E}_q\left[\frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})}\right] \geq \mathbb{E}_q\left[\log \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})}\right]
$$

The right-hand side is the **Evidence Lower Bound (ELBO)**:

$$
\mathcal{L}(q, \theta) = \mathbb{E}_q\left[\log \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})}\right] = \int q(\mathbf{Z}) \log \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})} \, d\mathbf{Z}
$$

### ELBO Decomposition

The ELBO can be decomposed into interpretable components:

$$
\mathcal{L}(q, \theta) = \int q(\mathbf{Z}) \log p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z} - \int q(\mathbf{Z}) \log q(\mathbf{Z}) \, d\mathbf{Z}
$$

$$
= \underbrace{\mathbb{E}_{q}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]}_{\text{Expected complete-data log-likelihood}} + \underbrace{H[q]}_{\text{Entropy of } q}
$$

where $H[q] = -\mathbb{E}_q[\log q(\mathbf{Z})] = -\int q(\mathbf{Z}) \log q(\mathbf{Z}) \, d\mathbf{Z}$ is the **entropy** of the distribution $q$.

**Interpretation**:

- The first term encourages $q$ to place mass where the joint probability $p(\mathbf{X}, \mathbf{Z} | \theta)$ is high
- The entropy term encourages $q$ to be spread out, preventing collapse to a point mass
- Together, they balance fit and uncertainty

### When Equality Holds

Jensen's inequality becomes an equality when the random variable inside the expectation is constant. For our setting, this requires:

$$
\frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})} = c \quad \text{for all } \mathbf{Z} \text{ where } q(\mathbf{Z}) > 0
$$

for some constant $c$. This means:

$$
q(\mathbf{Z}) \propto p(\mathbf{X}, \mathbf{Z} | \theta)
$$

To find the normalizing constant, integrate both sides:

$$
\int q(\mathbf{Z}) \, d\mathbf{Z} = 1 = \frac{1}{c} \int p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z} = \frac{p(\mathbf{X} | \theta)}{c}
$$

Thus $c = p(\mathbf{X} | \theta)$, and:

$$
q(\mathbf{Z}) = \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{p(\mathbf{X} | \theta)} = p(\mathbf{Z} | \mathbf{X}, \theta)
$$

**Key Result**: The bound is tight ($\mathcal{L}(q, \theta) = \ell(\theta)$) if and only if $q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta)$.

---

## KL Divergence Decomposition

### Alternative Derivation

The gap between the log-likelihood $\ell(\theta)$ and the ELBO $\mathcal{L}(q, \theta)$ has a precise characterization in terms of **Kullback-Leibler (KL) divergence**.

### The Fundamental Decomposition

$$
\boxed{\ell(\theta) = \mathcal{L}(q, \theta) + D_{\mathrm{KL}}\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)}
$$

This elegant identity shows that the log-likelihood decomposes into the ELBO plus the KL divergence from $q$ to the true posterior.

### Complete Derivation

**Step 1**: Write out the KL divergence from $q(\mathbf{Z})$ to the posterior $p(\mathbf{Z} | \mathbf{X}, \theta)$:

$$
D_{\mathrm{KL}}\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr) = \int q(\mathbf{Z}) \log \frac{q(\mathbf{Z})}{p(\mathbf{Z} | \mathbf{X}, \theta)} \, d\mathbf{Z}
$$

**Step 2**: Expand the posterior using Bayes' theorem:

$$
p(\mathbf{Z} | \mathbf{X}, \theta) = \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{p(\mathbf{X} | \theta)}
$$

**Step 3**: Substitute into the KL divergence:

$$
D_{\mathrm{KL}} = \int q(\mathbf{Z}) \log \frac{q(\mathbf{Z}) \cdot p(\mathbf{X} | \theta)}{p(\mathbf{X}, \mathbf{Z} | \theta)} \, d\mathbf{Z}
$$

$$
= \int q(\mathbf{Z}) \left[\log q(\mathbf{Z}) - \log p(\mathbf{X}, \mathbf{Z} | \theta) + \log p(\mathbf{X} | \theta)\right] d\mathbf{Z}
$$

**Step 4**: Distribute the expectation:

$$
D_{\mathrm{KL}} = \mathbb{E}_q[\log q(\mathbf{Z})] - \mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} | \theta)] + \log p(\mathbf{X} | \theta) \cdot \underbrace{\int q(\mathbf{Z}) \, d\mathbf{Z}}_{=1}
$$

$$
= -H[q] - \mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} | \theta)] + \ell(\theta)
$$

**Step 5**: Recognize the ELBO:

$$
D_{\mathrm{KL}} = -\mathcal{L}(q, \theta) + \ell(\theta)
$$

**Step 6**: Rearrange to obtain:

$$
\ell(\theta) = \mathcal{L}(q, \theta) + D_{\mathrm{KL}}\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)
$$

### Why ELBO is a Lower Bound

The decomposition immediately implies that ELBO is a lower bound:

**Property (Non-negativity of KL)**: For any two distributions $p$ and $q$:

$$
D_{\mathrm{KL}}(q \| p) \geq 0
$$

with equality if and only if $q = p$ almost everywhere.

**Consequence**: Since $D_{\mathrm{KL}} \geq 0$:

$$
\ell(\theta) = \mathcal{L}(q, \theta) + D_{\mathrm{KL}} \geq \mathcal{L}(q, \theta)
$$

The ELBO is always a **lower bound** on the log-likelihood, for any choice of $q$.

---

## Gap Analysis

### The Gap is the KL Divergence

The difference between the true log-likelihood and the ELBO is exactly the KL divergence:

$$
\text{Gap} = \ell(\theta) - \mathcal{L}(q, \theta) = D_{\mathrm{KL}}\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)
$$

This gap measures how well $q(\mathbf{Z})$ approximates the true posterior $p(\mathbf{Z} | \mathbf{X}, \theta)$.

### Tightness Condition

The bound becomes tight (gap equals zero) under a precise condition:

$$
\mathcal{L}(q, \theta) = \ell(\theta) \quad \Longleftrightarrow \quad D_{\mathrm{KL}} = 0 \quad \Longleftrightarrow \quad q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta)
$$

**Interpretation**: The ELBO equals the log-likelihood if and only if $q$ is exactly the true posterior.

### Quantifying the Gap

For a given $q$, the gap can be understood through properties of KL divergence:

**1. Information-Theoretic Interpretation**:

$$
D_{\mathrm{KL}}(q \| p) = \mathbb{E}_q\left[\log \frac{q(\mathbf{Z})}{p(\mathbf{Z} | \mathbf{X}, \theta)}\right]
$$

This measures the expected log-ratio, or the "surprise" when using $p$ to encode samples from $q$.

**2. Asymmetry**: Note that $D_{\mathrm{KL}}(q \| p) \neq D_{\mathrm{KL}}(p \| q)$ in general. The EM algorithm uses $D_{\mathrm{KL}}(q \| p)$, which:

- Tends to be **zero-forcing**: $q$ avoids regions where $p$ is small
- Leads to **mode-seeking** behavior when $p$ is multimodal

**3. Relationship to Other Divergences**: The KL divergence is a special case of the $f$-divergence family and is related to the Fisher information metric in the limit of small perturbations.

### Gap Dynamics During EM

During EM iterations, the gap evolves systematically:

**After E-step** (at iteration $t$):

- We set $q^{(t+1)}(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$
- The gap becomes zero: $D_{\mathrm{KL}}(q^{(t+1)} \| p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})) = 0$
- The bound is tight: $\mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})$

**After M-step**:

- We update $\theta^{(t)} \to \theta^{(t+1)}$
- The gap may become positive again: $D_{\mathrm{KL}}(q^{(t+1)} \| p(\mathbf{Z} | \mathbf{X}, \theta^{(t+1)})) \geq 0$
- But the likelihood has improved: $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$

---

## Geometric Interpretation

### ELBO as Tangent Lower Bound

The ELBO provides a **tangent lower bound** to the log-likelihood function:

**At the current parameter** $\theta^{(t)}$:

- When $q = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$, the bound touches $\ell(\theta)$ at $\theta^{(t)}$
- $\mathcal{L}(q, \theta^{(t)}) = \ell(\theta^{(t)})$

**Away from the current parameter**:

- For $\theta \neq \theta^{(t)}$, the bound lies below $\ell(\theta)$
- $\mathcal{L}(q, \theta) \leq \ell(\theta)$

### Majorization-Minimization Perspective

The EM algorithm is an instance of **Majorization-Minimization (MM)** (or Minorization-Maximization for maximization):

1. **Minorize**: Construct a lower bound $\mathcal{L}(q, \theta)$ that touches $\ell(\theta)$ at the current point
2. **Maximize**: Find the maximum of the lower bound
3. **Iterate**: The new point becomes the current point

This guarantees that each step improves (or maintains) the objective:

$$
\ell(\theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})
$$

### Visualization

Consider plotting $\ell(\theta)$ as a function of $\theta$:

```
    ℓ(θ)
      │     ╭─────╮
      │    ╱       ╲        ← True log-likelihood
      │   ╱         ╲
      │  ╱           ╲
      │ ╱   ╭───╮     ╲     ← ELBO (tangent at θ⁽ᵗ⁾)
      │╱   ╱     ╲     ╲
      ├───●───────●─────→ θ
         θ⁽ᵗ⁾    θ⁽ᵗ⁺¹⁾
```

The ELBO curve touches $\ell(\theta)$ at $\theta^{(t)}$ and lies below elsewhere. The M-step moves to the peak of the ELBO.

---

## Connection to EM Algorithm

### Why This Matters for EM

The ELBO decomposition reveals the mechanism behind EM:

**E-step** sets $q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$:

- This makes $D_{\mathrm{KL}} = 0$
- The bound becomes tight at $\theta^{(t)}$
- We can now optimize a tractable lower bound

**M-step** maximizes $\mathcal{L}(q, \theta)$ over $\theta$:

- Since the bound is tight at $\theta^{(t)}$, any improvement in $\mathcal{L}$ translates to improvement in $\ell$
- The new $\theta^{(t+1)}$ satisfies $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$

### Connection to Variational Inference

The KL decomposition provides a direct bridge to **variational inference**:

- In EM, we set $q$ to the exact posterior (when tractable)
- In variational inference, we restrict $q$ to a tractable family and optimize within that family
- Both methods maximize the ELBO, but with different constraints on $q$

This connection is explored further in the Variational Inference chapter.

---

## Summary

The ELBO is the cornerstone of the EM algorithm, and we have established:

| Derivation | Key Insight |
|------------|-------------|
| Jensen's Inequality | $\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]$; equality when $Y$ is constant |
| KL Decomposition | $\ell(\theta) = \mathcal{L}(q, \theta) + D_{\mathrm{KL}}(q \| p)$ |
| Gap Analysis | Gap $= D_{\mathrm{KL}}$; zero iff $q$ equals true posterior |
| Geometric View | ELBO is tangent lower bound; EM is MM algorithm |

These perspectives are complementary:

- Jensen's approach is constructive (shows how to build the bound)
- KL approach is analytical (shows why the bound is tight)
- Gap analysis is quantitative (measures approximation quality)
- Geometric view is intuitive (visualizes the optimization)
