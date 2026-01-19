# Conjugate Priors

## Overview

Conjugate priors enable analytical solutions to Bayesian inference problems, avoiding numerical integration or sampling methods. This module develops the theory of conjugacy and presents three fundamental conjugate families: Beta-Binomial, Gamma-Poisson, and Normal-Normal.

---

## 1. Theory of Conjugate Priors

### 1.1 Definition

A prior distribution $p(\theta)$ is **conjugate** to a likelihood function $p(D|\theta)$ if the posterior distribution $p(\theta|D)$ belongs to the same parametric family as the prior.

Formally, if $\mathcal{F}$ is a family of distributions and:

$$
p(\theta) \in \mathcal{F} \implies p(\theta|D) \in \mathcal{F} \quad \text{for all data } D
$$

then $\mathcal{F}$ is a **conjugate family** for the likelihood.

### 1.2 Advantages of Conjugate Priors

| Advantage | Description |
|-----------|-------------|
| **Analytical posteriors** | Closed-form solutions without numerical computation |
| **Computational efficiency** | No numerical integration or MCMC sampling required |
| **Interpretable updates** | Simple parameter transformations with clear meaning |
| **Sequential updating** | Straightforward online learning |
| **Mathematical elegance** | Deep insight into the structure of inference |

### 1.3 Common Conjugate Families

| Prior | Likelihood | Posterior | Use Case |
|-------|------------|-----------|----------|
| Beta | Binomial/Bernoulli | Beta | Binary outcomes, proportions |
| Gamma | Poisson | Gamma | Count data, rates |
| Gamma | Exponential | Gamma | Waiting times, lifetimes |
| Normal | Normal (known $\sigma^2$) | Normal | Continuous measurements |
| Normal-Inverse-Gamma | Normal | Normal-Inverse-Gamma | Unknown mean and variance |
| Dirichlet | Multinomial | Dirichlet | Categorical outcomes |

---

## 2. Beta-Binomial Model

### 2.1 Model Specification

**Prior:** $\theta \sim \text{Beta}(\alpha, \beta)$

$$
p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}
$$

**Likelihood:** $k | n, \theta \sim \text{Binomial}(n, \theta)$

$$
p(k|n, \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}
$$

**Posterior:** $\theta | k, n \sim \text{Beta}(\alpha + k, \beta + n - k)$

### 2.2 Conjugate Update

The update rule has an elegant interpretation:

$$
\boxed{\alpha_{\text{post}} = \alpha_{\text{prior}} + k, \quad \beta_{\text{post}} = \beta_{\text{prior}} + (n-k)}
$$

where:
- $k$ = number of successes observed
- $n - k$ = number of failures observed
- $\alpha - 1$ = prior pseudo-successes
- $\beta - 1$ = prior pseudo-failures

### 2.3 Posterior Predictive Distribution

The posterior predictive for $y$ successes in $m$ future trials is the **Beta-Binomial distribution**:

$$
P(y | D) = \int_0^1 \text{Binomial}(y | m, \theta) \cdot \text{Beta}(\theta | \alpha', \beta') \, d\theta
$$

$$
= \binom{m}{y} \frac{B(y + \alpha', m - y + \beta')}{B(\alpha', \beta')}
$$

### 2.4 Implementation

```python
from scipy import stats
import numpy as np

class BetaBinomialModel:
    """Beta-Binomial conjugate model for binary data."""
    
    def __init__(self, alpha=1, beta=1):
        """Initialize with Beta(alpha, beta) prior."""
        self.prior_alpha = alpha
        self.prior_beta = beta
        self.posterior_alpha = alpha
        self.posterior_beta = beta
        self.data_history = []
    
    def update(self, successes, trials):
        """Update posterior with observed data."""
        failures = trials - successes
        self.posterior_alpha += successes
        self.posterior_beta += failures
        self.data_history.append((successes, trials))
    
    def posterior_predictive(self, n_trials=1):
        """Compute posterior predictive probabilities."""
        y_values = np.arange(n_trials + 1)
        probs = []
        
        for y in y_values:
            prob = (stats.binom.comb(n_trials, y) * 
                   stats.beta.beta_func(y + self.posterior_alpha, 
                                       n_trials - y + self.posterior_beta) / 
                   stats.beta.beta_func(self.posterior_alpha, 
                                       self.posterior_beta))
            probs.append(prob)
        
        return np.array(probs)
    
    def summary(self):
        """Print summary statistics."""
        post_dist = stats.beta(self.posterior_alpha, self.posterior_beta)
        
        print(f"Posterior: Beta({self.posterior_alpha}, {self.posterior_beta})")
        print(f"  Mean: {post_dist.mean():.4f}")
        print(f"  95% CI: [{post_dist.ppf(0.025):.4f}, {post_dist.ppf(0.975):.4f}]")
```

### 2.5 Example: Coin Flipping

**Prior:** Beta$(2, 2)$ — weak belief in fairness

**Data:** 17 successes in 20 trials

**Posterior:** Beta$(2 + 17, 2 + 3) = $ Beta$(19, 5)$

| Statistic | Value |
|-----------|-------|
| Posterior Mean | $19/24 = 0.792$ |
| Posterior Mode | $18/22 = 0.818$ |
| 95% Credible Interval | $[0.60, 0.93]$ |

---

## 3. Gamma-Poisson Model

### 3.1 Model Specification

**Prior:** $\lambda \sim \text{Gamma}(\alpha, \beta)$

$$
p(\lambda) = \frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha-1} e^{-\beta\lambda}
$$

**Likelihood:** $x_i | \lambda \sim \text{Poisson}(\lambda)$ independently

$$
p(x_1, \ldots, x_n | \lambda) = \prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!}
$$

**Posterior:** $\lambda | x_1, \ldots, x_n \sim \text{Gamma}(\alpha + \sum x_i, \beta + n)$

### 3.2 Conjugate Update

$$
\boxed{\alpha_{\text{post}} = \alpha_{\text{prior}} + \sum_{i=1}^n x_i, \quad \beta_{\text{post}} = \beta_{\text{prior}} + n}
$$

**Interpretation:**
- $\alpha$ = prior pseudo-count (total events)
- $\beta$ = prior pseudo-observations (number of periods)
- Update: add actual total count to $\alpha$, number of observations to $\beta$

### 3.3 Prior and Posterior Statistics

| Statistic | Prior | Posterior |
|-----------|-------|-----------|
| Mean | $\alpha/\beta$ | $(\alpha + \sum x_i)/(\beta + n)$ |
| Variance | $\alpha/\beta^2$ | $(\alpha + \sum x_i)/(\beta + n)^2$ |
| Mode | $(\alpha-1)/\beta$ | $(\alpha + \sum x_i - 1)/(\beta + n)$ |

### 3.4 Posterior Predictive Distribution

The posterior predictive for a future count is **Negative Binomial**:

$$
P(x_{\text{new}} | D) = \text{NegBinom}\left(x_{\text{new}} \,\Big|\, \alpha', \frac{\beta'}{\beta' + 1}\right)
$$

### 3.5 Implementation

```python
class GammaPoissonModel:
    """Gamma-Poisson conjugate model for count data."""
    
    def __init__(self, alpha=1, beta=1):
        """Initialize with Gamma(alpha, beta) prior."""
        self.prior_alpha = alpha
        self.prior_beta = beta
        self.posterior_alpha = alpha
        self.posterior_beta = beta
        self.data = []
    
    def update(self, counts):
        """Update posterior with observed counts."""
        counts = np.asarray(counts)
        self.posterior_alpha += np.sum(counts)
        self.posterior_beta += len(counts)
        self.data.extend(counts)
    
    def posterior_predictive(self):
        """Return posterior predictive (Negative Binomial)."""
        n = self.posterior_alpha
        p = self.posterior_beta / (self.posterior_beta + 1)
        return stats.nbinom(n, p)
    
    def summary(self):
        """Print summary statistics."""
        post_dist = stats.gamma(self.posterior_alpha, 
                                scale=1/self.posterior_beta)
        
        print(f"Posterior: Gamma({self.posterior_alpha}, {self.posterior_beta})")
        print(f"  Mean (rate): {post_dist.mean():.4f}")
        print(f"  95% CI: [{post_dist.ppf(0.025):.4f}, {post_dist.ppf(0.975):.4f}]")
```

### 3.6 Example: Website Visits

**Prior:** Gamma$(2, 1)$ — expect approximately 2 events per period

**Data:** Daily visits $[5, 3, 7, 4, 6, 5, 8, 3, 4, 6]$ (total = 51, n = 10)

**Posterior:** Gamma$(2 + 51, 1 + 10) = $ Gamma$(53, 11)$

| Statistic | Value |
|-----------|-------|
| Posterior Mean | $53/11 = 4.82$ |
| Sample Mean | $51/10 = 5.10$ |
| 95% Credible Interval | $[3.61, 6.22]$ |

---

## 4. Normal-Normal Model (Known Variance)

### 4.1 Model Specification

**Prior:** $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

**Likelihood:** $x_i | \mu \sim \mathcal{N}(\mu, \sigma^2)$ independently, with $\sigma^2$ known

**Posterior:** $\mu | x_1, \ldots, x_n \sim \mathcal{N}(\mu_n, \sigma_n^2)$

### 4.2 Conjugate Update (Precision Form)

Define **precision** as inverse variance: $\tau = 1/\sigma^2$

$$
\tau_n = \tau_0 + n\tau_{\text{data}}
$$

$$
\boxed{\mu_n = \frac{\tau_0 \mu_0 + n\tau_{\text{data}} \bar{x}}{\tau_n}, \quad \sigma_n^2 = \frac{1}{\tau_n}}
$$

where $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ is the sample mean.

### 4.3 Interpretation

The posterior mean is a **precision-weighted average**:

$$
\mu_n = w_{\text{prior}} \cdot \mu_0 + w_{\text{data}} \cdot \bar{x}
$$

where:
- $w_{\text{prior}} = \tau_0 / \tau_n$ — weight on prior mean
- $w_{\text{data}} = n\tau_{\text{data}} / \tau_n$ — weight on sample mean

As $n \to \infty$: $w_{\text{data}} \to 1$ and $\mu_n \to \bar{x}$.

### 4.4 Implementation

```python
class NormalNormalModel:
    """Normal-Normal conjugate model (known variance)."""
    
    def __init__(self, prior_mean=0, prior_std=1, known_std=1):
        """Initialize with N(prior_mean, prior_std^2) prior."""
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.known_std = known_std
        
        self.posterior_mean = prior_mean
        self.posterior_std = prior_std
        self.data = []
    
    def update(self, observations):
        """Update posterior with new observations."""
        observations = np.asarray(observations)
        n = len(observations)
        x_bar = np.mean(observations)
        
        # Precision calculations
        prior_precision = 1 / (self.prior_std ** 2)
        data_precision = n / (self.known_std ** 2)
        posterior_precision = prior_precision + data_precision
        
        # Update parameters
        self.posterior_mean = ((prior_precision * self.prior_mean + 
                               data_precision * x_bar) / posterior_precision)
        self.posterior_std = np.sqrt(1 / posterior_precision)
        
        # For sequential updates
        self.prior_mean = self.posterior_mean
        self.prior_std = self.posterior_std
        
        self.data.extend(observations)
    
    def summary(self):
        """Print summary statistics."""
        post_dist = stats.norm(self.posterior_mean, self.posterior_std)
        
        print(f"Posterior: N({self.posterior_mean:.4f}, {self.posterior_std:.4f})")
        print(f"  95% CI: [{post_dist.ppf(0.025):.4f}, {post_dist.ppf(0.975):.4f}]")
```

### 4.5 Example: Measurements

**Prior:** $\mathcal{N}(100, 10^2)$ — prior belief about true value

**Known data std:** $\sigma = 5$

**Data:** Measurements $[102, 98, 105, 101, 99, 103, 97, 104, 100, 102]$

Sample statistics: $n = 10$, $\bar{x} = 101.1$

**Posterior calculation:**
- Prior precision: $\tau_0 = 1/100 = 0.01$
- Data precision: $n\tau = 10/25 = 0.4$
- Total precision: $\tau_n = 0.41$

$$
\mu_n = \frac{0.01 \times 100 + 0.4 \times 101.1}{0.41} \approx 101.07
$$

$$
\sigma_n = \sqrt{1/0.41} \approx 1.56
$$

| Weight | Value |
|--------|-------|
| Prior | $0.01/0.41 = 2.4\%$ |
| Data | $0.40/0.41 = 97.6\%$ |

---

## 5. Summary of Conjugate Update Rules

### 5.1 Quick Reference Table

| Model | Prior Parameters | Update Rule |
|-------|------------------|-------------|
| **Beta-Binomial** | $\alpha, \beta$ | $\alpha' = \alpha + k$, $\beta' = \beta + (n-k)$ |
| **Gamma-Poisson** | $\alpha, \beta$ | $\alpha' = \alpha + \sum x_i$, $\beta' = \beta + n$ |
| **Normal-Normal** | $\mu_0, \sigma_0$ | $\mu_n = \frac{\tau_0\mu_0 + n\tau\bar{x}}{\tau_0 + n\tau}$, $\sigma_n^2 = \frac{1}{\tau_0 + n\tau}$ |

### 5.2 Hyperparameter Interpretation

| Model | Hyperparameters | Interpretation |
|-------|-----------------|----------------|
| **Beta** | $\alpha, \beta$ | $\alpha - 1$ pseudo-successes, $\beta - 1$ pseudo-failures |
| **Gamma** | $\alpha, \beta$ | $\alpha$ pseudo-counts, $\beta$ pseudo-observations |
| **Normal** | $\mu_0, \tau_0$ | Prior mean, prior precision (confidence) |

---

## 6. Key Takeaways

1. **Conjugate priors** yield analytical posterior distributions in the same family as the prior, enabling closed-form Bayesian inference.

2. **Beta-Binomial** is the workhorse for binary and proportion data. The Beta hyperparameters act as pseudo-counts.

3. **Gamma-Poisson** handles count data naturally. The posterior predictive is Negative Binomial.

4. **Normal-Normal** (known variance) produces precision-weighted averages. The posterior mean interpolates between prior mean and sample mean.

5. **Sequential updating** is computationally efficient with conjugate priors — the posterior from one batch becomes the prior for the next.

---

## 7. Exercises

### Exercise 1: Conjugate Verification
Mathematically verify that Beta is conjugate to Binomial by working through Bayes' theorem. Show explicitly that $p(\theta|k, n)$ is Beta distributed.

### Exercise 2: Prior Elicitation
You believe a coin is fair but aren't certain. Express this as Beta$(\alpha, \beta)$. What values capture "weak belief" vs "strong belief" in fairness?

### Exercise 3: Gamma-Exponential
Research the Gamma-Exponential conjugate pair (for waiting times/lifetimes). Implement a class similar to `GammaPoissonModel` for this pair.

### Exercise 4: Dirichlet-Multinomial
Extend Beta-Binomial to multiple categories using Dirichlet-Multinomial. Implement inference for a 3-outcome dice rolling problem.

### Exercise 5: Non-Conjugate Prior
What happens when the prior is not conjugate? Compare analytical Beta-Binomial with grid-based numerical integration using a non-conjugate prior (e.g., Uniform on $\log(\theta)$).

---

## References

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), Chapter 2
- Murphy, K. *Machine Learning: A Probabilistic Perspective*, Chapter 3
- Hoff, P. *A First Course in Bayesian Statistical Methods*, Chapters 3–5
