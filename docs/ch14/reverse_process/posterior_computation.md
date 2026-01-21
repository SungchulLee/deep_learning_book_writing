# Posterior Computation: $q(x_{t-1} | x_t, x_0)$

This document derives the closed-form posterior distribution used in DDPM, explaining why it comes from multiplying two Gaussians.

## The Key Result

The posterior $q(x_{t-1} | x_t, x_0)$ is Gaussian with:

**Mean:**
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

**Variance:**
$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$$

## Why is it a Product of Two Gaussians?

### The Markov Property

The forward process is a Markov chain:
$$q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)I)$$

Because it's Markov:
$$x_t \perp x_0 \mid x_{t-1}$$

This means "$x_t$ is independent of $x_0$ given $x_{t-1}$."

### Joint Distribution Factorization

The joint distribution factorizes as:
$$q(x_t, x_{t-1} | x_0) = q(x_t | x_{t-1}) \cdot q(x_{t-1} | x_0)$$

### Bayes' Rule

Apply Bayes' rule:
$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t, x_{t-1} | x_0)}{q(x_t | x_0)} \propto q(x_t | x_{t-1}) \cdot q(x_{t-1} | x_0)$$

So **up to a normalization constant**, the posterior over $x_{t-1}$ is the **product** of:
- **Likelihood**: $q(x_t | x_{t-1})$
- **Prior**: $q(x_{t-1} | x_0)$

Both are Gaussians in $x_{t-1}$, so their product is also Gaussian!

## The Two Gaussians

Using the DDPM forward formulas:

### 1. Forward Step (Likelihood)
$$q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)I)$$

### 2. Collapsed Forward (Prior)
$$q(x_{t-1} | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}} x_0, (1-\bar{\alpha}_{t-1})I)$$

where:
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
- $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$
- $\beta_t = 1 - \alpha_t$

## Product of Gaussians Derivation

### Canonical Form

For a scalar (vector case is identical with $I$), write the likelihood as a function of $x_{t-1}$:

$$q(x_t | x_{t-1}) \propto \exp\left(-\frac{1}{2(1-\alpha_t)}(x_t - \sqrt{\alpha_t} x_{t-1})^2\right)$$

Expand the square:
$$(x_t - \sqrt{\alpha_t} x_{t-1})^2 = \alpha_t x_{t-1}^2 - 2\sqrt{\alpha_t} x_t x_{t-1} + \text{const}$$

In canonical quadratic form:
$$\propto \exp\left(-\frac{1}{2}\left[A_1 x_{t-1}^2 - 2 B_1 x_{t-1}\right]\right)$$

where:
- $A_1 = \frac{\alpha_t}{1-\alpha_t}$ (precision from likelihood)
- $B_1 = \frac{\sqrt{\alpha_t} x_t}{1-\alpha_t}$

Similarly for the prior:
$$q(x_{t-1} | x_0) \propto \exp\left(-\frac{1}{2}\left[A_2 x_{t-1}^2 - 2 B_2 x_{t-1}\right]\right)$$

where:
- $A_2 = \frac{1}{1-\bar{\alpha}_{t-1}}$ (precision from prior)
- $B_2 = \frac{\sqrt{\bar{\alpha}_{t-1}} x_0}{1-\bar{\alpha}_{t-1}}$

### Product Rule for Gaussians

Multiply the two Gaussians:
$$q(x_t | x_{t-1}) \cdot q(x_{t-1} | x_0) \propto \exp\left(-\frac{1}{2}\left[(A_1 + A_2) x_{t-1}^2 - 2(B_1 + B_2) x_{t-1}\right]\right)$$

A Gaussian $\mathcal{N}(\mu, \sigma^2)$ in canonical form has:
- Precision: $A = 1/\sigma^2$
- $B = \mu/\sigma^2$

So the posterior has:
- **Precision**: $A = A_1 + A_2 = \frac{\alpha_t}{1-\alpha_t} + \frac{1}{1-\bar{\alpha}_{t-1}}$
- **Canonical mean**: $B = B_1 + B_2 = \frac{\sqrt{\alpha_t} x_t}{1-\alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} x_0}{1-\bar{\alpha}_{t-1}}$

Then:
- Posterior variance: $\tilde{\sigma}_t^2 = 1/A$
- Posterior mean: $\tilde{\mu}_t(x_t, x_0) = B/A$

## Simplifying the Algebra

Using $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$ and $\beta_t = 1 - \alpha_t$:

### Posterior Variance

$$\tilde{\beta}_t = \frac{1}{A_1 + A_2} = \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{\alpha_t(1-\bar{\alpha}_{t-1}) + (1-\alpha_t)}$$

Simplify the denominator:
$$\alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t = 1 - \bar{\alpha}_t$$

So:
$$\tilde{\beta}_t = \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$$

### Posterior Mean

After similar algebra:
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

## Summary in One Sentence

The **Markov property** gives $q(x_{t-1}|x_t,x_0) \propto q(x_t|x_{t-1}) \cdot q(x_{t-1}|x_0)$. Both factors are **Gaussians in $x_{t-1}$**. The **product of Gaussians is another Gaussian**, whose mean you get by summing precisions and precision-weighted means → this simplifies to the DDPM formula.

## Visual Intuition

```
         Prior                    Likelihood
    q(x_{t-1} | x_0)    ×    q(x_t | x_{t-1})
          ↓                       ↓
    Gaussian around          Gaussian around
    √ᾱ_{t-1} x_0              x_t/√α_t
          \                     /
           \                   /
            ╲                 ╱
             ╲               ╱
              ↘             ↙
                Posterior
           q(x_{t-1} | x_t, x_0)
                   ↓
            Gaussian with mean
            between the two
```

The posterior mean is a **weighted average** between:
- Where $x_0$ says $x_{t-1}$ should be (from the prior)
- Where $x_t$ says $x_{t-1}$ should be (from the likelihood)

The weights depend on the relative precisions (inverse variances).

## PyTorch Implementation

```python
def compute_posterior_params(x_0, x_t, t, alphas, alphas_cumprod, betas):
    """
    Compute parameters of q(x_{t-1} | x_t, x_0).
    """
    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]
    alpha_bar_t_prev = alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
    beta_t = betas[t]
    
    # Posterior variance
    posterior_var = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
    
    # Posterior mean coefficients
    coef_x0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
    coef_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
    
    # Reshape for broadcasting
    coef_x0 = coef_x0.view(-1, 1, 1, 1)
    coef_xt = coef_xt.view(-1, 1, 1, 1)
    
    posterior_mean = coef_x0 * x_0 + coef_xt * x_t
    
    return posterior_mean, posterior_var
```

## Navigation

- **Previous**: [Reverse Mean Derivation](reverse_mean.md)
- **Next**: [Score Learning](score_learning.md)
