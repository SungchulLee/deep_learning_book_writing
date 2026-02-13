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

---

# Appendix: Detailed Conjugate Model Derivations

The following sections provide complete derivations, implementations, and analysis for each conjugate family introduced above.


---

# Bernoulli-Beta Conjugate Model

The Bernoulli-Beta model is the simplest and most instructive example of Bayesian conjugate analysis. It provides a complete, analytically tractable framework for inference about probabilities from binary data. This canonical model illustrates fundamental Bayesian concepts that generalize to more complex settings.

---

## Problem Setup

### The Inference Problem

We observe binary outcomes $x_1, x_2, \ldots, x_n \in \{0, 1\}$ (e.g., coin flips, success/failure, click/no-click) and wish to infer the underlying success probability $\theta \in [0, 1]$.

**Frequentist approach**: Point estimate $\hat{\theta} = \bar{x} = k/n$ where $k = \sum_i x_i$.

**Bayesian approach**: Full posterior distribution $p(\theta \mid x_1, \ldots, x_n)$ quantifying uncertainty about $\theta$.

### The Bernoulli Likelihood

Each observation follows a Bernoulli distribution:

$$
x_i \mid \theta \sim \text{Bernoulli}(\theta)
$$

$$
p(x_i \mid \theta) = \theta^{x_i}(1-\theta)^{1-x_i}
$$

For $n$ independent observations with $k$ successes:

$$
p(x_1, \ldots, x_n \mid \theta) = \prod_{i=1}^{n} \theta^{x_i}(1-\theta)^{1-x_i} = \theta^k(1-\theta)^{n-k}
$$

**Key observation**: The likelihood depends on the data only through $k$ and $n$. The sufficient statistic is $(k, n)$.

---

## The Beta Prior

### Why Beta?

We need a prior $p(\theta)$ on $[0, 1]$. The **Beta distribution** is the natural choice because:

1. **Support**: Defined on $[0, 1]$, matching the parameter space
2. **Flexibility**: Can represent diverse prior beliefs (uniform, U-shaped, skewed)
3. **Conjugacy**: Posterior is also Beta, enabling closed-form updates

### Beta Distribution Definition

$$
\theta \sim \text{Beta}(\alpha, \beta)
$$

$$
p(\theta \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha-1}(1-\theta)^{\beta-1} = \frac{1}{B(\alpha, \beta)} \theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}$ is the Beta function.

### Beta Distribution Properties

**Moments**:

$$
\mathbb{E}[\theta] = \frac{\alpha}{\alpha + \beta}
$$

$$
\text{Var}[\theta] = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}
$$

$$
\text{Mode}[\theta] = \frac{\alpha - 1}{\alpha + \beta - 2} \quad \text{(for } \alpha, \beta > 1\text{)}
$$

**Interpretation of parameters**:

- $\alpha - 1$: "pseudo-count" of prior successes
- $\beta - 1$: "pseudo-count" of prior failures
- $\alpha + \beta$: "prior sample size" or concentration

### Common Beta Priors

| Prior | $\alpha$ | $\beta$ | Mean | Shape | Use Case |
|-------|----------|---------|------|-------|----------|
| Uniform | 1 | 1 | 0.5 | Flat | Maximum ignorance |
| Jeffreys | 0.5 | 0.5 | 0.5 | U-shaped | Reference prior |
| Haldane | 0 | 0 | — | Improper | Limiting non-informative |
| Symmetric | $a$ | $a$ | 0.5 | Peaked/U | No directional preference |
| Informative | 10 | 2 | 0.83 | Right-skewed | Prior belief $\theta$ is high |

### Visualizing Beta Priors

```
α=1, β=1 (Uniform)        α=0.5, β=0.5 (Jeffreys)    α=2, β=5 (Informative)
    ___________               ∪                           /\
   |           |             / \                         /  \____
   |___________|            /   \                       /        \
   0           1           0     1                     0          1
```

---

## Conjugate Posterior Derivation

### The Conjugacy Property

**Definition**: A prior $p(\theta)$ is **conjugate** to a likelihood $p(\mathcal{D} \mid \theta)$ if the posterior $p(\theta \mid \mathcal{D})$ belongs to the same distributional family as the prior.

### Derivation

**Prior**: $\theta \sim \text{Beta}(\alpha, \beta)$

$$
p(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

**Likelihood**: $k$ successes in $n$ trials

$$
p(\mathcal{D} \mid \theta) = \theta^k(1-\theta)^{n-k}
$$

**Posterior** (via Bayes' theorem):

$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \cdot p(\theta)
$$

$$
p(\theta \mid \mathcal{D}) \propto \theta^k(1-\theta)^{n-k} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

$$
p(\theta \mid \mathcal{D}) \propto \theta^{k + \alpha - 1}(1-\theta)^{n-k+\beta-1}
$$

This is the kernel of a Beta distribution! Therefore:

$$
\boxed{\theta \mid \mathcal{D} \sim \text{Beta}(\alpha + k, \beta + n - k)}
$$

### The Update Rule

| Quantity | Prior | Posterior |
|----------|-------|-----------|
| Distribution | $\text{Beta}(\alpha, \beta)$ | $\text{Beta}(\alpha + k, \beta + n - k)$ |
| "Successes" | $\alpha - 1$ | $\alpha + k - 1$ |
| "Failures" | $\beta - 1$ | $\beta + n - k - 1$ |
| "Sample size" | $\alpha + \beta$ | $\alpha + \beta + n$ |

**Intuition**: We simply add the observed successes to $\alpha$ and observed failures to $\beta$. The prior acts like "pseudo-data" from a previous experiment.

---

## Posterior Analysis

### Posterior Mean

$$
\mathbb{E}[\theta \mid \mathcal{D}] = \frac{\alpha + k}{\alpha + \beta + n}
$$

This can be rewritten as a **weighted average**:

$$
\mathbb{E}[\theta \mid \mathcal{D}] = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \underbrace{\frac{\alpha}{\alpha + \beta}}_{\text{prior mean}} + \frac{n}{\alpha + \beta + n} \cdot \underbrace{\frac{k}{n}}_{\text{MLE}}
$$

Let $w = \frac{n}{\alpha + \beta + n}$ be the data weight. Then:

$$
\mathbb{E}[\theta \mid \mathcal{D}] = (1-w) \cdot \text{prior mean} + w \cdot \text{MLE}
$$

**Key insight**: The posterior mean interpolates between prior mean and MLE, with weight determined by relative "sample sizes."

### Posterior Mode (MAP Estimate)

$$
\hat{\theta}_{\text{MAP}} = \frac{\alpha + k - 1}{\alpha + \beta + n - 2} \quad \text{(for } \alpha + k > 1, \beta + n - k > 1\text{)}
$$

### Posterior Variance

$$
\text{Var}[\theta \mid \mathcal{D}] = \frac{(\alpha + k)(\beta + n - k)}{(\alpha + \beta + n)^2(\alpha + \beta + n + 1)}
$$

As $n \to \infty$:

$$
\text{Var}[\theta \mid \mathcal{D}] \approx \frac{\hat{\theta}(1-\hat{\theta})}{n} \to 0
$$

The posterior concentrates around the true value.

### Credible Intervals

The $(1-\alpha)$ **equal-tailed credible interval** is:

$$
\left[F^{-1}_{\text{Beta}}\left(\frac{\alpha}{2}\right), F^{-1}_{\text{Beta}}\left(1 - \frac{\alpha}{2}\right)\right]
$$

where $F^{-1}_{\text{Beta}}$ is the quantile function of $\text{Beta}(\alpha + k, \beta + n - k)$.

---

## Sequential Updating

### Online Learning

A powerful feature of conjugate models is efficient sequential updating. Given a stream of observations, we update the posterior incrementally:

$$
\text{Beta}(\alpha_0, \beta_0) \xrightarrow{x_1} \text{Beta}(\alpha_1, \beta_1) \xrightarrow{x_2} \text{Beta}(\alpha_2, \beta_2) \xrightarrow{x_3} \cdots
$$

where:

$$
\alpha_{t+1} = \alpha_t + x_{t+1}, \quad \beta_{t+1} = \beta_t + (1 - x_{t+1})
$$

### Order Independence

The final posterior is **independent of observation order**:

$$
p(\theta \mid x_1, x_2, \ldots, x_n) = p(\theta \mid x_{\pi(1)}, x_{\pi(2)}, \ldots, x_{\pi(n)})
$$

for any permutation $\pi$. This follows from exchangeability of the Bernoulli likelihood.

---

## Posterior Predictive Distribution

### Predicting the Next Observation

Given observed data, what is the probability the next observation is a success?

$$
p(x_{n+1} = 1 \mid \mathcal{D}) = \int_0^1 p(x_{n+1} = 1 \mid \theta) \, p(\theta \mid \mathcal{D}) \, d\theta
$$

$$
= \int_0^1 \theta \cdot p(\theta \mid \mathcal{D}) \, d\theta = \mathbb{E}[\theta \mid \mathcal{D}]
$$

$$
\boxed{p(x_{n+1} = 1 \mid \mathcal{D}) = \frac{\alpha + k}{\alpha + \beta + n}}
$$

This is **Laplace's rule of succession**: The predictive probability equals the posterior mean.

### Predicting Multiple Future Observations

For $m$ future trials, the number of successes $k'$ follows a **Beta-Binomial** distribution:

$$
p(k' \mid m, \mathcal{D}) = \binom{m}{k'} \frac{B(\alpha + k + k', \beta + n - k + m - k')}{B(\alpha + k, \beta + n - k)}
$$

This accounts for both sampling variability and parameter uncertainty.

---

## Special Cases and Connections

### Uniform Prior ($\alpha = \beta = 1$)

$$
p(\theta \mid \mathcal{D}) = \text{Beta}(1 + k, 1 + n - k)
$$

$$
\mathbb{E}[\theta \mid \mathcal{D}] = \frac{k + 1}{n + 2}
$$

This is **Laplace's rule**: Add one success and one failure to the observed counts.

### Jeffreys Prior ($\alpha = \beta = 1/2$)

$$
p(\theta \mid \mathcal{D}) = \text{Beta}(k + 1/2, n - k + 1/2)
$$

$$
\mathbb{E}[\theta \mid \mathcal{D}] = \frac{k + 1/2}{n + 1}
$$

The Jeffreys prior is the unique prior invariant under reparameterization $\theta \leftrightarrow 1 - \theta$ and under monotonic transformations.

### Haldane Prior ($\alpha = \beta = 0$)

$$
p(\theta \mid \mathcal{D}) = \text{Beta}(k, n - k)
$$

$$
\mathbb{E}[\theta \mid \mathcal{D}] = \frac{k}{n} = \text{MLE}
$$

**Warning**: The Haldane prior is improper and yields an improper posterior if $k = 0$ or $k = n$.

### Connection to Maximum Likelihood

As prior strength $\to 0$ (or $n \to \infty$):

$$
\hat{\theta}_{\text{Bayes}} \to \hat{\theta}_{\text{MLE}} = \frac{k}{n}
$$

The Bayesian and frequentist estimates converge asymptotically.

---

## Practical Considerations

### Choosing Prior Parameters

**Method 1: Prior Mean and "Equivalent Sample Size"**

If you believe the prior mean is $\mu_0$ with prior equivalent to $n_0$ observations:

$$
\alpha = \mu_0 \cdot n_0, \quad \beta = (1 - \mu_0) \cdot n_0
$$

**Method 2: Prior Mean and Variance**

Given prior mean $\mu$ and variance $\sigma^2$:

$$
\alpha = \mu \left(\frac{\mu(1-\mu)}{\sigma^2} - 1\right), \quad \beta = (1-\mu) \left(\frac{\mu(1-\mu)}{\sigma^2} - 1\right)
$$

**Method 3: Prior Credible Interval**

If you believe $\theta \in [a, b]$ with 95% probability, solve numerically for $(\alpha, \beta)$ such that:

$$
F_{\text{Beta}}(b; \alpha, \beta) - F_{\text{Beta}}(a; \alpha, \beta) = 0.95
$$

### Prior Sensitivity Analysis

Always check how conclusions change with different priors:

| Prior | Parameters | Posterior Mean (k=7, n=10) |
|-------|------------|---------------------------|
| Uniform | (1, 1) | 0.667 |
| Jeffreys | (0.5, 0.5) | 0.682 |
| Skeptical | (1, 9) | 0.400 |
| Optimistic | (9, 1) | 0.800 |

If conclusions are robust across reasonable priors, they are more credible.

---

## Python Implementation

```python
"""
Bernoulli-Beta Conjugate Model: Complete Implementation

This module provides a comprehensive implementation of Bayesian inference
for binary data using the Beta-Bernoulli conjugate pair.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import beta as beta_func
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class BetaPosterior:
    """
    Represents a Beta posterior distribution.
    
    Attributes
    ----------
    alpha : float
        First shape parameter (pseudo-successes + 1)
    beta : float
        Second shape parameter (pseudo-failures + 1)
    n_successes : int
        Observed number of successes
    n_trials : int
        Observed number of trials
    """
    alpha: float
    beta: float
    n_successes: int = 0
    n_trials: int = 0
    
    @property
    def mean(self) -> float:
        """Posterior mean E[θ|D]."""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def mode(self) -> Optional[float]:
        """Posterior mode (MAP estimate)."""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        elif self.alpha <= 1 and self.beta > 1:
            return 0.0
        elif self.alpha > 1 and self.beta <= 1:
            return 1.0
        else:
            return None  # Bimodal or undefined
    
    @property
    def variance(self) -> float:
        """Posterior variance Var[θ|D]."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b)**2 * (a + b + 1))
    
    @property
    def std(self) -> float:
        """Posterior standard deviation."""
        return np.sqrt(self.variance)
    
    def pdf(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate posterior density."""
        return stats.beta.pdf(theta, self.alpha, self.beta)
    
    def cdf(self, theta: float) -> float:
        """Evaluate posterior CDF."""
        return stats.beta.cdf(theta, self.alpha, self.beta)
    
    def quantile(self, p: float) -> float:
        """Compute posterior quantile."""
        return stats.beta.ppf(p, self.alpha, self.beta)
    
    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """
        Compute equal-tailed credible interval.
        
        Parameters
        ----------
        level : float
            Credibility level (default 0.95 for 95% CI)
        
        Returns
        -------
        tuple
            (lower, upper) bounds
        """
        alpha_level = 1 - level
        lower = self.quantile(alpha_level / 2)
        upper = self.quantile(1 - alpha_level / 2)
        return (lower, upper)
    
    def hpd_interval(self, level: float = 0.95, n_points: int = 1000) -> Tuple[float, float]:
        """
        Compute Highest Posterior Density interval.
        
        The shortest interval containing the specified probability mass.
        """
        # Grid search for HPD
        theta_grid = np.linspace(0.001, 0.999, n_points)
        pdf_vals = self.pdf(theta_grid)
        
        # Sort by density (descending)
        sorted_idx = np.argsort(pdf_vals)[::-1]
        sorted_theta = theta_grid[sorted_idx]
        sorted_pdf = pdf_vals[sorted_idx]
        
        # Accumulate probability mass
        cumsum = np.cumsum(sorted_pdf) * (theta_grid[1] - theta_grid[0])
        cutoff_idx = np.searchsorted(cumsum, level)
        
        # HPD region bounds
        hpd_theta = sorted_theta[:cutoff_idx + 1]
        return (hpd_theta.min(), hpd_theta.max())
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Draw samples from posterior."""
        return stats.beta.rvs(self.alpha, self.beta, size=n_samples)
    
    def predictive_prob(self) -> float:
        """Probability next observation is success (Laplace's rule)."""
        return self.mean
    
    def __repr__(self) -> str:
        return f"Beta({self.alpha:.2f}, {self.beta:.2f})"


class BetaBernoulliModel:
    """
    Complete Beta-Bernoulli conjugate model.
    
    Parameters
    ----------
    prior_alpha : float
        Prior α parameter
    prior_beta : float
        Prior β parameter
    """
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self._reset()
    
    def _reset(self):
        """Reset to prior state."""
        self.current_alpha = self.prior_alpha
        self.current_beta = self.prior_beta
        self.n_successes = 0
        self.n_trials = 0
    
    @property
    def prior(self) -> BetaPosterior:
        """Return prior distribution."""
        return BetaPosterior(self.prior_alpha, self.prior_beta)
    
    @property
    def posterior(self) -> BetaPosterior:
        """Return current posterior distribution."""
        return BetaPosterior(
            self.current_alpha, 
            self.current_beta,
            self.n_successes,
            self.n_trials
        )
    
    def update(self, successes: int, trials: int) -> BetaPosterior:
        """
        Update posterior with new observations.
        
        Parameters
        ----------
        successes : int
            Number of successes observed
        trials : int
            Number of trials observed
        
        Returns
        -------
        BetaPosterior
            Updated posterior distribution
        """
        self.current_alpha += successes
        self.current_beta += (trials - successes)
        self.n_successes += successes
        self.n_trials += trials
        return self.posterior
    
    def update_single(self, outcome: int) -> BetaPosterior:
        """
        Update with a single observation.
        
        Parameters
        ----------
        outcome : int
            0 or 1
        
        Returns
        -------
        BetaPosterior
            Updated posterior
        """
        return self.update(outcome, 1)
    
    def update_sequence(self, outcomes: List[int]) -> List[BetaPosterior]:
        """
        Update sequentially, returning posterior history.
        
        Parameters
        ----------
        outcomes : list
            Sequence of 0/1 observations
        
        Returns
        -------
        list
            List of posterior distributions after each update
        """
        history = [self.posterior]
        for outcome in outcomes:
            self.update_single(outcome)
            history.append(self.posterior)
        return history
    
    def log_marginal_likelihood(self) -> float:
        """
        Compute log marginal likelihood (log evidence).
        
        log p(D) = log B(α + k, β + n - k) - log B(α, β)
        
        Returns
        -------
        float
            Log marginal likelihood
        """
        from scipy.special import betaln
        
        prior_term = betaln(self.prior_alpha, self.prior_beta)
        posterior_term = betaln(self.current_alpha, self.current_beta)
        
        return posterior_term - prior_term
    
    def predictive_distribution(self, m: int) -> np.ndarray:
        """
        Compute Beta-Binomial predictive distribution for m future trials.
        
        Parameters
        ----------
        m : int
            Number of future trials
        
        Returns
        -------
        array
            Probabilities for k' = 0, 1, ..., m successes
        """
        from scipy.special import comb, betaln
        
        a, b = self.current_alpha, self.current_beta
        k_vals = np.arange(m + 1)
        
        log_probs = (
            np.log(comb(m, k_vals, exact=False)) +
            betaln(a + k_vals, b + m - k_vals) -
            betaln(a, b)
        )
        
        return np.exp(log_probs)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_beta_distribution(
    alpha: float, 
    beta: float, 
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: str = 'steelblue',
    fill: bool = True
) -> plt.Axes:
    """Plot a Beta distribution."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    
    theta = np.linspace(0.001, 0.999, 500)
    pdf = stats.beta.pdf(theta, alpha, beta)
    
    if fill:
        ax.fill_between(theta, pdf, alpha=0.3, color=color)
    ax.plot(theta, pdf, color=color, linewidth=2, label=label)
    
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_prior_posterior(
    model: BetaBernoulliModel,
    true_theta: Optional[float] = None,
    title: str = "Bayesian Update"
) -> plt.Figure:
    """Visualize prior, likelihood, and posterior."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    theta = np.linspace(0.001, 0.999, 500)
    
    # Prior
    prior_pdf = stats.beta.pdf(theta, model.prior_alpha, model.prior_beta)
    ax.plot(theta, prior_pdf / prior_pdf.max(), 'b--', 
            linewidth=2, label=f'Prior: Beta({model.prior_alpha}, {model.prior_beta})')
    
    # Likelihood (normalized for visualization)
    if model.n_trials > 0:
        k, n = model.n_successes, model.n_trials
        likelihood = theta**k * (1 - theta)**(n - k)
        ax.plot(theta, likelihood / likelihood.max(), 'g:', 
                linewidth=2, label=f'Likelihood ({k}/{n} successes)')
    
    # Posterior
    post = model.posterior
    posterior_pdf = post.pdf(theta)
    ax.fill_between(theta, posterior_pdf / posterior_pdf.max(), 
                    alpha=0.3, color='red')
    ax.plot(theta, posterior_pdf / posterior_pdf.max(), 'r-', 
            linewidth=2, label=f'Posterior: {post}')
    
    # True value
    if true_theta is not None:
        ax.axvline(true_theta, color='black', linestyle='--', 
                   linewidth=2, label=f'True θ = {true_theta}')
    
    # Posterior mean
    ax.axvline(post.mean, color='red', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Density (normalized)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_sequential_update(
    outcomes: List[int],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    true_theta: Optional[float] = None
) -> plt.Figure:
    """Visualize sequential Bayesian updating."""
    
    model = BetaBernoulliModel(prior_alpha, prior_beta)
    history = model.update_sequence(outcomes)
    
    n_steps = len(history)
    n_cols = min(4, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    theta = np.linspace(0.001, 0.999, 200)
    
    for i, (ax, post) in enumerate(zip(axes[:n_steps], history)):
        pdf = post.pdf(theta)
        ax.fill_between(theta, pdf, alpha=0.4, color='steelblue')
        ax.plot(theta, pdf, 'b-', linewidth=2)
        
        if true_theta is not None:
            ax.axvline(true_theta, color='red', linestyle='--', linewidth=1.5)
        
        ax.axvline(post.mean, color='green', linestyle=':', linewidth=1.5)
        
        if i == 0:
            ax.set_title(f'Prior\nE[θ]={post.mean:.3f}')
        else:
            cumsum = sum(outcomes[:i])
            ax.set_title(f'After {i} obs ({cumsum}/{i})\nE[θ]={post.mean:.3f}')
        
        ax.set_xlim(0, 1)
        ax.set_xlabel('θ')
    
    # Hide unused axes
    for ax in axes[n_steps:]:
        ax.set_visible(False)
    
    plt.suptitle('Sequential Bayesian Updating', fontsize=14)
    plt.tight_layout()
    
    return fig


# =============================================================================
# Demonstrations
# =============================================================================

def demo_basic_inference():
    """Demonstrate basic Beta-Bernoulli inference."""
    
    print("=" * 60)
    print("BASIC BETA-BERNOULLI INFERENCE")
    print("=" * 60)
    
    # Setup
    true_theta = 0.7
    n_trials = 20
    np.random.seed(42)
    data = np.random.binomial(1, true_theta, n_trials)
    k = data.sum()
    
    print(f"\nTrue θ: {true_theta}")
    print(f"Data: {k} successes in {n_trials} trials")
    print(f"MLE: {k/n_trials:.4f}")
    
    # Different priors
    priors = [
        ("Uniform", 1, 1),
        ("Jeffreys", 0.5, 0.5),
        ("Informative (pessimistic)", 2, 8),
        ("Informative (optimistic)", 8, 2),
    ]
    
    print("\nPosterior summaries under different priors:")
    print("-" * 60)
    
    for name, alpha, beta in priors:
        model = BetaBernoulliModel(alpha, beta)
        model.update(k, n_trials)
        post = model.posterior
        ci = post.credible_interval(0.95)
        
        print(f"\n{name} prior: Beta({alpha}, {beta})")
        print(f"  Posterior: Beta({post.alpha:.1f}, {post.beta:.1f})")
        print(f"  Mean: {post.mean:.4f}")
        print(f"  Mode: {post.mode:.4f}" if post.mode else "  Mode: undefined")
        print(f"  Std:  {post.std:.4f}")
        print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  P(next success): {post.predictive_prob():.4f}")


def demo_sequential_learning():
    """Demonstrate sequential updating."""
    
    print("\n" + "=" * 60)
    print("SEQUENTIAL BAYESIAN LEARNING")
    print("=" * 60)
    
    true_theta = 0.6
    np.random.seed(123)
    outcomes = list(np.random.binomial(1, true_theta, 15))
    
    print(f"\nTrue θ: {true_theta}")
    print(f"Outcomes: {outcomes}")
    
    model = BetaBernoulliModel(1, 1)  # Uniform prior
    
    print("\nEvolution of posterior mean:")
    print("-" * 40)
    
    for i, outcome in enumerate(outcomes):
        model.update_single(outcome)
        post = model.posterior
        cumsum = sum(outcomes[:i+1])
        print(f"After obs {i+1:2d} (x={outcome}): "
              f"E[θ|D] = {post.mean:.4f}, "
              f"σ = {post.std:.4f}, "
              f"Data: {cumsum}/{i+1}")
    
    # Create visualization
    fig = plot_sequential_update(outcomes, true_theta=true_theta)
    fig.savefig('sequential_beta_update.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSee: sequential_beta_update.png")


def demo_predictive_distribution():
    """Demonstrate posterior predictive distribution."""
    
    print("\n" + "=" * 60)
    print("POSTERIOR PREDICTIVE DISTRIBUTION")
    print("=" * 60)
    
    # Observed data
    k, n = 7, 10
    
    model = BetaBernoulliModel(1, 1)
    model.update(k, n)
    
    print(f"\nObserved: {k} successes in {n} trials")
    print(f"Posterior: Beta({model.current_alpha}, {model.current_beta})")
    
    # Predict next m trials
    m = 10
    predictive = model.predictive_distribution(m)
    
    print(f"\nPredictive distribution for next {m} trials:")
    print("-" * 40)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    k_vals = np.arange(m + 1)
    ax.bar(k_vals, predictive, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Expected value
    expected = np.sum(k_vals * predictive)
    ax.axvline(expected, color='red', linestyle='--', linewidth=2,
               label=f'E[k\'] = {expected:.2f}')
    
    ax.set_xlabel('Number of successes in next 10 trials', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Beta-Binomial Posterior Predictive Distribution', fontsize=14)
    ax.legend()
    ax.set_xticks(k_vals)
    
    plt.tight_layout()
    plt.savefig('predictive_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Expected successes in next {m}: {expected:.2f}")
    print(f"Most likely outcome: {k_vals[np.argmax(predictive)]} successes")
    print("\nSee: predictive_distribution.png")


if __name__ == "__main__":
    demo_basic_inference()
    demo_sequential_learning()
    demo_predictive_distribution()
```

---

## Summary

| Aspect | Formula |
|--------|---------|
| **Prior** | $\theta \sim \text{Beta}(\alpha, \beta)$ |
| **Likelihood** | $p(\mathcal{D} \mid \theta) = \theta^k(1-\theta)^{n-k}$ |
| **Posterior** | $\theta \mid \mathcal{D} \sim \text{Beta}(\alpha + k, \beta + n - k)$ |
| **Posterior Mean** | $\frac{\alpha + k}{\alpha + \beta + n}$ |
| **Posterior Mode** | $\frac{\alpha + k - 1}{\alpha + \beta + n - 2}$ |
| **Predictive** | $p(x_{n+1}=1 \mid \mathcal{D}) = \frac{\alpha + k}{\alpha + \beta + n}$ |

### Key Insights

1. **Conjugacy**: Beta prior + Bernoulli likelihood → Beta posterior
2. **Pseudo-counts**: Prior parameters act like additional observations
3. **Weighted average**: Posterior mean interpolates prior mean and MLE
4. **Sequential updating**: Add successes to $\alpha$, failures to $\beta$
5. **Laplace's rule**: Predictive probability equals posterior mean
6. **Asymptotic MLE**: As $n \to \infty$, Bayesian estimate → frequentist MLE

### Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Prior-likelihood-posterior | Ch13: Foundations | General framework |
| Gaussian conjugate | Ch13: Gaussian Models | Continuous analog |
| Conjugate priors | Ch13: Conjugate Priors | General theory |
| Model comparison | Ch13: Bayes Factor | Evidence computation |
| BNN classification | Ch13: BNN | Multi-layer extension |

### Key References

- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapter 2.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. Chapter 3.
- Hoff, P. D. (2009). *A First Course in Bayesian Statistical Methods*. Springer.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Section 2.1.

---

# Gaussian Inference with Known Variance

Bayesian inference for the mean of a Gaussian distribution with known variance is the foundational continuous conjugate model. It provides the clearest illustration of how Bayesian updating combines prior information with data through **precision-weighted averaging**. This elegant result generalizes to multivariate settings and forms the basis for understanding more complex Bayesian models.

---

## Problem Setup

### The Inference Problem

We observe continuous measurements $x_1, x_2, \ldots, x_n \in \mathbb{R}$ assumed to be drawn from a Gaussian distribution with **unknown mean** $\mu$ and **known variance** $\sigma^2$:

$$
x_i \mid \mu \sim \mathcal{N}(\mu, \sigma^2)
$$

The goal is to infer the posterior distribution $p(\mu \mid x_1, \ldots, x_n)$.

**When is variance known?**

- Measurement devices with calibrated precision
- Long-run historical estimates of variability
- Theoretical constraints (e.g., quantum noise limits)
- Simplifying assumption for pedagogical purposes

### The Gaussian Likelihood

For a single observation:

$$
p(x_i \mid \mu) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

For $n$ independent observations:

$$
p(x_1, \ldots, x_n \mid \mu) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

$$
= (2\pi\sigma^2)^{-n/2} \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2\right)
$$

### Sufficient Statistics

The likelihood can be rewritten using the sufficient statistic $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$:

$$
\sum_{i=1}^{n}(x_i - \mu)^2 = \sum_{i=1}^{n}(x_i - \bar{x})^2 + n(\bar{x} - \mu)^2
$$

The first term is constant with respect to $\mu$, so:

$$
p(\mathcal{D} \mid \mu) \propto \exp\left(-\frac{n(\bar{x} - \mu)^2}{2\sigma^2}\right)
$$

**Key insight**: The likelihood depends on the data only through $(\bar{x}, n)$. The sample mean $\bar{x}$ is sufficient for $\mu$.

---

## Precision: The Natural Parameterization

### Definition

The **precision** is the inverse variance:

$$
\tau = \frac{1}{\sigma^2}
$$

Precision measures **information content**: higher precision means more informative (less uncertain) measurements.

### Why Precision?

Precision is the natural parameter for combining Gaussian information:

- **Precisions add**: When combining independent information sources, precisions sum
- **Variances don't add simply**: $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$ only for independent $X, Y$
- **Linear updates**: Bayesian updates become linear in precision

| Quantity | Variance Form | Precision Form |
|----------|---------------|----------------|
| Single observation | $\sigma^2$ | $\tau = 1/\sigma^2$ |
| Sample mean of $n$ obs | $\sigma^2/n$ | $n\tau$ |
| Prior | $\sigma_0^2$ | $\tau_0 = 1/\sigma_0^2$ |
| Posterior | $\sigma_n^2$ | $\tau_n = \tau_0 + n\tau$ |

---

## The Gaussian Prior

### Conjugate Prior

For conjugacy, we use a Gaussian prior on $\mu$:

$$
\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)
$$

or equivalently, with precision $\tau_0 = 1/\sigma_0^2$:

$$
p(\mu) = \sqrt{\frac{\tau_0}{2\pi}} \exp\left(-\frac{\tau_0}{2}(\mu - \mu_0)^2\right)
$$

### Prior Parameters Interpretation

| Parameter | Symbol | Interpretation |
|-----------|--------|----------------|
| Prior mean | $\mu_0$ | Best guess before seeing data |
| Prior variance | $\sigma_0^2$ | Uncertainty in prior belief |
| Prior precision | $\tau_0$ | Confidence in prior belief |
| Prior "effective sample size" | $n_0 = \tau_0/\tau = \sigma^2/\sigma_0^2$ | Prior equivalent to $n_0$ observations |

### Common Prior Choices

**Informative prior**: $\mu_0$ and $\sigma_0^2$ reflect genuine prior knowledge

$$
\mu \sim \mathcal{N}(100, 5^2) \quad \text{("Mean is around 100, ± 10")}
$$

**Weakly informative prior**: Broad but proper

$$
\mu \sim \mathcal{N}(0, 100^2) \quad \text{("Probably not astronomically large")}
$$

**Improper flat prior**: Limiting case $\sigma_0^2 \to \infty$

$$
p(\mu) \propto 1 \quad \text{(improper, but yields proper posterior)}
$$

---

## Conjugate Posterior Derivation

### The Derivation

**Prior**:

$$
p(\mu) \propto \exp\left(-\frac{\tau_0}{2}(\mu - \mu_0)^2\right)
$$

**Likelihood**:

$$
p(\mathcal{D} \mid \mu) \propto \exp\left(-\frac{n\tau}{2}(\mu - \bar{x})^2\right)
$$

where $\tau = 1/\sigma^2$ is the known data precision.

**Posterior** (via Bayes' theorem):

$$
p(\mu \mid \mathcal{D}) \propto p(\mathcal{D} \mid \mu) \cdot p(\mu)
$$

$$
\propto \exp\left(-\frac{n\tau}{2}(\mu - \bar{x})^2 - \frac{\tau_0}{2}(\mu - \mu_0)^2\right)
$$

### Completing the Square

Expand the exponent:

$$
-\frac{1}{2}\left[n\tau(\mu^2 - 2\mu\bar{x} + \bar{x}^2) + \tau_0(\mu^2 - 2\mu\mu_0 + \mu_0^2)\right]
$$

$$
= -\frac{1}{2}\left[(n\tau + \tau_0)\mu^2 - 2\mu(n\tau\bar{x} + \tau_0\mu_0) + \text{const}\right]
$$

$$
= -\frac{n\tau + \tau_0}{2}\left[\mu^2 - 2\mu\frac{n\tau\bar{x} + \tau_0\mu_0}{n\tau + \tau_0}\right] + \text{const}
$$

$$
= -\frac{\tau_n}{2}\left(\mu - \mu_n\right)^2 + \text{const}
$$

This is the kernel of a Gaussian! Therefore:

$$
\boxed{\mu \mid \mathcal{D} \sim \mathcal{N}(\mu_n, \sigma_n^2)}
$$

### The Update Formulas

**Posterior precision** (precisions add):

$$
\boxed{\tau_n = \tau_0 + n\tau}
$$

$$
\sigma_n^2 = \frac{1}{\tau_n} = \frac{1}{\tau_0 + n\tau} = \frac{\sigma^2\sigma_0^2}{n\sigma_0^2 + \sigma^2}
$$

**Posterior mean** (precision-weighted average):

$$
\boxed{\mu_n = \frac{\tau_0\mu_0 + n\tau\bar{x}}{\tau_0 + n\tau} = \frac{\tau_0\mu_0 + n\tau\bar{x}}{\tau_n}}
$$

Or in variance form:

$$
\mu_n = \frac{\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}
$$

---

## Precision-Weighted Averaging

### The Fundamental Insight

The posterior mean is a **weighted average** of prior mean and data mean:

$$
\mu_n = w_0 \cdot \mu_0 + w_{\text{data}} \cdot \bar{x}
$$

where the weights are **proportional to precisions**:

$$
w_0 = \frac{\tau_0}{\tau_0 + n\tau}, \quad w_{\text{data}} = \frac{n\tau}{\tau_0 + n\tau}
$$

Note that $w_0 + w_{\text{data}} = 1$.

### Interpretation

- **More precise prior** ($\tau_0$ large) → More weight on prior mean
- **More data** ($n$ large) → More weight on sample mean
- **More precise measurements** ($\tau$ large) → Each observation counts more

### Equivalent Sample Size

Define the prior's **equivalent sample size**:

$$
n_0 = \frac{\tau_0}{\tau} = \frac{\sigma^2}{\sigma_0^2}
$$

Then:

$$
\mu_n = \frac{n_0 \cdot \mu_0 + n \cdot \bar{x}}{n_0 + n}
$$

The prior is worth $n_0$ observations at the data precision.

---

## Posterior Analysis

### Point Estimates

All three common point estimates coincide for the Gaussian posterior:

$$
\mathbb{E}[\mu \mid \mathcal{D}] = \text{Mode}[\mu \mid \mathcal{D}] = \text{Median}[\mu \mid \mathcal{D}] = \mu_n
$$

This is a unique property of symmetric unimodal distributions.

### Posterior Variance and Standard Error

$$
\text{Var}[\mu \mid \mathcal{D}] = \sigma_n^2 = \frac{1}{\tau_0 + n\tau}
$$

The posterior standard deviation (Bayesian "standard error"):

$$
\sigma_n = \frac{1}{\sqrt{\tau_0 + n\tau}}
$$

### Credible Intervals

For a Gaussian posterior, the $(1-\alpha)$ credible interval is:

$$
\mu_n \pm z_{\alpha/2} \cdot \sigma_n
$$

where $z_{\alpha/2}$ is the standard normal quantile.

**95% credible interval**:

$$
\left[\mu_n - 1.96\sigma_n, \mu_n + 1.96\sigma_n\right]
$$

For Gaussian posteriors, equal-tailed and HPD intervals coincide due to symmetry.

### Shrinkage

The posterior mean "shrinks" the MLE toward the prior mean:

$$
\mu_n - \mu_0 = \frac{n\tau}{\tau_0 + n\tau}(\bar{x} - \mu_0)
$$

The shrinkage factor $\frac{n\tau}{\tau_0 + n\tau} < 1$ pulls the estimate toward the prior.

---

## Asymptotic Behavior

### Large Sample Limit

As $n \to \infty$:

**Posterior mean**:

$$
\mu_n = \frac{\tau_0\mu_0 + n\tau\bar{x}}{\tau_0 + n\tau} \to \bar{x} \to \mu_{\text{true}}
$$

**Posterior variance**:

$$
\sigma_n^2 = \frac{1}{\tau_0 + n\tau} \sim \frac{1}{n\tau} = \frac{\sigma^2}{n} \to 0
$$

**Rate of convergence**:

$$
\sigma_n = O(n^{-1/2})
$$

The posterior concentrates around the true mean at the standard $\sqrt{n}$ rate.

### Prior Washout

The influence of the prior vanishes:

$$
\text{Prior weight} = \frac{\tau_0}{\tau_0 + n\tau} \to 0 \quad \text{as } n \to \infty
$$

With enough data, reasonable priors yield identical posteriors.

### Bernstein-von Mises

This model exactly satisfies the Bernstein-von Mises theorem: the posterior is asymptotically normal centered at the MLE with variance equal to the inverse Fisher information:

$$
p(\mu \mid \mathcal{D}_n) \xrightarrow{d} \mathcal{N}\left(\bar{x}, \frac{\sigma^2}{n}\right)
$$

---

## Sequential Updating

### Online Bayesian Learning

With conjugate priors, we can update sequentially without storing all data:

$$
\mathcal{N}(\mu_0, \sigma_0^2) \xrightarrow{x_1} \mathcal{N}(\mu_1, \sigma_1^2) \xrightarrow{x_2} \mathcal{N}(\mu_2, \sigma_2^2) \xrightarrow{x_3} \cdots
$$

**Update equations** (single observation $x$):

$$
\tau_{t+1} = \tau_t + \tau
$$

$$
\mu_{t+1} = \frac{\tau_t \mu_t + \tau x}{\tau_{t+1}}
$$

### Recursive Form

Equivalently:

$$
\mu_{t+1} = \mu_t + \frac{\tau}{\tau_{t+1}}(x - \mu_t) = \mu_t + K_t(x - \mu_t)
$$

where $K_t = \frac{\tau}{\tau_t + \tau} = \frac{\sigma_t^2}{\sigma_t^2 + \sigma^2}$ is the **Kalman gain**.

This is the simplest Kalman filter: a scalar state with known observation noise.

---

## Posterior Predictive Distribution

### Predicting New Observations

Given observed data, what is the distribution of a new observation $x_{n+1}$?

$$
p(x_{n+1} \mid \mathcal{D}) = \int p(x_{n+1} \mid \mu) \, p(\mu \mid \mathcal{D}) \, d\mu
$$

Both distributions are Gaussian, so the predictive is also Gaussian:

$$
x_{n+1} \mid \mathcal{D} \sim \mathcal{N}(\mu_n, \sigma^2 + \sigma_n^2)
$$

### Predictive Variance Decomposition

$$
\text{Var}[x_{n+1} \mid \mathcal{D}] = \underbrace{\sigma^2}_{\text{aleatoric}} + \underbrace{\sigma_n^2}_{\text{epistemic}}
$$

- **Aleatoric uncertainty** ($\sigma^2$): Inherent randomness in observations (irreducible)
- **Epistemic uncertainty** ($\sigma_n^2$): Uncertainty about $\mu$ (reducible with more data)

As $n \to \infty$, epistemic uncertainty vanishes, and predictive variance approaches $\sigma^2$.

---

## Connection to Frequentist Inference

### Comparison Table

| Aspect | Bayesian | Frequentist |
|--------|----------|-------------|
| Point estimate | $\mu_n$ (posterior mean) | $\bar{x}$ (MLE) |
| Interval | $\mu_n \pm z_{\alpha/2}\sigma_n$ (credible) | $\bar{x} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}$ (confidence) |
| Interpretation | "95% probability $\mu$ is in interval" | "95% of such intervals contain $\mu$" |
| Prior required | Yes | No |
| Incorporates prior info | Explicitly | Not directly |

### When They Agree

With flat prior ($\tau_0 \to 0$):

$$
\mu_n \to \bar{x}, \quad \sigma_n^2 \to \frac{\sigma^2}{n}
$$

The Bayesian credible interval equals the frequentist confidence interval.

### When They Differ

With informative prior, the Bayesian estimate is "regularized" toward the prior mean. This provides:

- **Better small-sample behavior** when prior is reasonable
- **Worse estimates** if prior is badly misspecified

---

## Multivariate Extension

### Setup

For $\boldsymbol{x}_i \in \mathbb{R}^d$ with known covariance $\boldsymbol{\Sigma}$:

$$
\boldsymbol{x}_i \mid \boldsymbol{\mu} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

Prior:

$$
\boldsymbol{\mu} \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)
$$

### Posterior

$$
\boldsymbol{\mu} \mid \mathcal{D} \sim \mathcal{N}(\boldsymbol{\mu}_n, \boldsymbol{\Sigma}_n)
$$

**Posterior precision** (precision matrices add):

$$
\boldsymbol{\Lambda}_n = \boldsymbol{\Lambda}_0 + n\boldsymbol{\Lambda}
$$

where $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ and $\boldsymbol{\Lambda}_0 = \boldsymbol{\Sigma}_0^{-1}$.

**Posterior mean**:

$$
\boldsymbol{\mu}_n = \boldsymbol{\Sigma}_n(\boldsymbol{\Lambda}_0\boldsymbol{\mu}_0 + n\boldsymbol{\Lambda}\bar{\boldsymbol{x}})
$$

The same precision-weighted averaging, now in matrix form.

---

## Python Implementation

```python
"""
Gaussian Inference with Known Variance: Complete Implementation

This module provides Bayesian inference for the mean of a Gaussian
distribution when the variance is known, demonstrating precision-weighted
averaging and sequential updating.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class GaussianPosterior:
    """
    Represents a Gaussian posterior distribution for μ.
    
    Attributes
    ----------
    mean : float
        Posterior mean μₙ
    variance : float
        Posterior variance σₙ²
    n_observations : int
        Number of observations incorporated
    """
    mean: float
    variance: float
    n_observations: int = 0
    
    @property
    def precision(self) -> float:
        """Posterior precision τₙ = 1/σₙ²."""
        return 1.0 / self.variance
    
    @property
    def std(self) -> float:
        """Posterior standard deviation σₙ."""
        return np.sqrt(self.variance)
    
    def pdf(self, mu: np.ndarray) -> np.ndarray:
        """Evaluate posterior density."""
        return stats.norm.pdf(mu, self.mean, self.std)
    
    def cdf(self, mu: float) -> float:
        """Evaluate posterior CDF."""
        return stats.norm.cdf(mu, self.mean, self.std)
    
    def quantile(self, p: float) -> float:
        """Compute posterior quantile."""
        return stats.norm.ppf(p, self.mean, self.std)
    
    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """
        Compute credible interval.
        
        For Gaussian, equal-tailed and HPD intervals coincide.
        """
        alpha = 1 - level
        z = stats.norm.ppf(1 - alpha/2)
        return (self.mean - z * self.std, self.mean + z * self.std)
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Draw samples from posterior."""
        return np.random.normal(self.mean, self.std, n_samples)
    
    def __repr__(self) -> str:
        return f"N({self.mean:.4f}, {self.variance:.4f})"


class GaussianKnownVarianceModel:
    """
    Bayesian inference for Gaussian mean with known variance.
    
    Parameters
    ----------
    prior_mean : float
        Prior mean μ₀
    prior_variance : float
        Prior variance σ₀²
    known_variance : float
        Known data variance σ²
    """
    
    def __init__(
        self, 
        prior_mean: float, 
        prior_variance: float, 
        known_variance: float
    ):
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.known_variance = known_variance
        
        # Precisions
        self.prior_precision = 1.0 / prior_variance
        self.data_precision = 1.0 / known_variance
        
        # Current state
        self._reset()
    
    def _reset(self):
        """Reset to prior state."""
        self.current_precision = self.prior_precision
        self.current_mean = self.prior_mean
        self.n_observations = 0
        self.sum_x = 0.0
    
    @property
    def prior(self) -> GaussianPosterior:
        """Return prior distribution."""
        return GaussianPosterior(self.prior_mean, self.prior_variance, 0)
    
    @property
    def posterior(self) -> GaussianPosterior:
        """Return current posterior distribution."""
        return GaussianPosterior(
            self.current_mean,
            1.0 / self.current_precision,
            self.n_observations
        )
    
    def update(self, data: np.ndarray) -> GaussianPosterior:
        """
        Update posterior with new observations.
        
        Parameters
        ----------
        data : array
            New observations
        
        Returns
        -------
        GaussianPosterior
            Updated posterior
        """
        data = np.atleast_1d(data)
        n = len(data)
        
        # Update sufficient statistics
        self.n_observations += n
        self.sum_x += data.sum()
        
        # Update precision (precisions add)
        self.current_precision = self.prior_precision + self.n_observations * self.data_precision
        
        # Update mean (precision-weighted average)
        self.current_mean = (
            self.prior_precision * self.prior_mean + 
            self.data_precision * self.sum_x
        ) / self.current_precision
        
        return self.posterior
    
    def update_single(self, x: float) -> GaussianPosterior:
        """Update with a single observation."""
        return self.update(np.array([x]))
    
    def update_sequential(self, data: np.ndarray) -> List[GaussianPosterior]:
        """
        Update sequentially, returning posterior history.
        
        Parameters
        ----------
        data : array
            Sequence of observations
        
        Returns
        -------
        list
            Posterior after each observation
        """
        self._reset()
        history = [self.posterior]
        
        for x in data:
            self.update_single(x)
            history.append(self.posterior)
        
        return history
    
    def predictive_distribution(self) -> Tuple[float, float]:
        """
        Compute posterior predictive distribution for next observation.
        
        Returns
        -------
        tuple
            (predictive_mean, predictive_variance)
        """
        pred_mean = self.current_mean
        pred_var = self.known_variance + 1.0 / self.current_precision
        return pred_mean, pred_var
    
    def log_marginal_likelihood(self, data: np.ndarray) -> float:
        """
        Compute log marginal likelihood (log evidence).
        
        log p(D) = log ∫ p(D|μ) p(μ) dμ
        
        For Gaussian-Gaussian, this is available in closed form.
        """
        n = len(data)
        x_bar = data.mean()
        
        # Marginal is Gaussian with inflated variance
        marginal_var = self.prior_variance + self.known_variance / n
        
        # Sum of squared deviations from prior mean
        ss_from_prior = np.sum((data - self.prior_mean)**2)
        
        # Log marginal likelihood
        log_ml = (
            -0.5 * n * np.log(2 * np.pi * self.known_variance)
            - 0.5 * ss_from_prior / self.known_variance
            + 0.5 * np.log(self.prior_variance / (self.prior_variance + self.known_variance / n))
            + 0.5 * n**2 * (x_bar - self.prior_mean)**2 / 
              (self.known_variance * (n * self.prior_variance / self.known_variance + 1))
        )
        
        return log_ml
    
    def prior_weight(self) -> float:
        """Compute weight given to prior mean."""
        return self.prior_precision / self.current_precision
    
    def data_weight(self) -> float:
        """Compute weight given to data mean."""
        return (self.n_observations * self.data_precision) / self.current_precision
    
    def equivalent_prior_samples(self) -> float:
        """Prior expressed as equivalent number of observations."""
        return self.prior_precision / self.data_precision


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_precision_weighted_averaging(
    model: GaussianKnownVarianceModel,
    data: np.ndarray,
    true_mu: Optional[float] = None
) -> plt.Figure:
    """Visualize precision-weighted averaging."""
    
    model._reset()
    model.update(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Prior, likelihood, posterior
    ax = axes[0]
    
    x_bar = data.mean()
    mu_range = np.linspace(
        min(model.prior_mean, x_bar) - 3 * max(np.sqrt(model.prior_variance), np.sqrt(model.known_variance)),
        max(model.prior_mean, x_bar) + 3 * max(np.sqrt(model.prior_variance), np.sqrt(model.known_variance)),
        500
    )
    
    # Prior
    prior_pdf = stats.norm.pdf(mu_range, model.prior_mean, np.sqrt(model.prior_variance))
    ax.plot(mu_range, prior_pdf, 'b--', linewidth=2, 
            label=f'Prior: N({model.prior_mean}, {model.prior_variance})')
    
    # Likelihood (normalized for visualization)
    likelihood_var = model.known_variance / len(data)
    likelihood_pdf = stats.norm.pdf(mu_range, x_bar, np.sqrt(likelihood_var))
    ax.plot(mu_range, likelihood_pdf, 'g:', linewidth=2,
            label=f'Likelihood: centered at x̄={x_bar:.2f}')
    
    # Posterior
    post = model.posterior
    posterior_pdf = post.pdf(mu_range)
    ax.fill_between(mu_range, posterior_pdf, alpha=0.3, color='red')
    ax.plot(mu_range, posterior_pdf, 'r-', linewidth=2,
            label=f'Posterior: {post}')
    
    if true_mu is not None:
        ax.axvline(true_mu, color='black', linestyle='--', linewidth=2,
                   label=f'True μ = {true_mu}')
    
    ax.axvline(post.mean, color='red', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Bayesian Update: Precision-Weighted Averaging', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Right: Weight diagram
    ax = axes[1]
    
    weights = [model.prior_weight(), model.data_weight()]
    labels = [f'Prior\nμ₀ = {model.prior_mean}', f'Data\nx̄ = {x_bar:.2f}']
    colors = ['steelblue', 'forestgreen']
    
    bars = ax.bar(labels, weights, color=colors, edgecolor='black', linewidth=2)
    
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{w:.1%}', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Weight in Posterior Mean', fontsize=12)
    ax.set_title(f'Weights (n={len(data)}, prior ≈ {model.equivalent_prior_samples():.1f} samples)', 
                 fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_sequential_updating(
    data: np.ndarray,
    prior_mean: float,
    prior_variance: float,
    known_variance: float,
    true_mu: Optional[float] = None
) -> plt.Figure:
    """Visualize sequential Bayesian updating."""
    
    model = GaussianKnownVarianceModel(prior_mean, prior_variance, known_variance)
    history = model.update_sequential(data)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: Evolution of posterior mean and CI
    ax = axes[0]
    
    n_vals = np.arange(len(history))
    means = [h.mean for h in history]
    cis = [h.credible_interval(0.95) for h in history]
    lowers = [ci[0] for ci in cis]
    uppers = [ci[1] for ci in cis]
    
    ax.fill_between(n_vals, lowers, uppers, alpha=0.3, color='steelblue',
                    label='95% Credible Interval')
    ax.plot(n_vals, means, 'b-', linewidth=2, marker='o', markersize=4,
            label='Posterior Mean')
    
    if true_mu is not None:
        ax.axhline(true_mu, color='red', linestyle='--', linewidth=2,
                   label=f'True μ = {true_mu}')
    
    ax.axhline(prior_mean, color='gray', linestyle=':', linewidth=1.5,
               label=f'Prior Mean = {prior_mean}')
    
    ax.set_xlabel('Number of Observations', fontsize=12)
    ax.set_ylabel('μ', fontsize=12)
    ax.set_title('Sequential Bayesian Updating', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Bottom: Evolution of posterior std
    ax = axes[1]
    
    stds = [h.std for h in history]
    ax.plot(n_vals, stds, 'g-', linewidth=2, marker='s', markersize=4)
    
    # Theoretical asymptotic
    asymptotic_std = np.sqrt(known_variance) / np.sqrt(np.maximum(n_vals, 1))
    asymptotic_std[0] = np.sqrt(prior_variance)
    ax.plot(n_vals, asymptotic_std, 'r--', linewidth=1.5, 
            label=r'Asymptotic: $\sigma/\sqrt{n}$')
    
    ax.set_xlabel('Number of Observations', fontsize=12)
    ax.set_ylabel('Posterior Std Dev', fontsize=12)
    ax.set_title('Uncertainty Reduction', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_predictive_distribution(
    model: GaussianKnownVarianceModel,
    true_mu: Optional[float] = None
) -> plt.Figure:
    """Visualize posterior predictive distribution."""
    
    pred_mean, pred_var = model.predictive_distribution()
    post = model.posterior
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_range = np.linspace(pred_mean - 4*np.sqrt(pred_var), 
                          pred_mean + 4*np.sqrt(pred_var), 500)
    
    # Posterior for μ
    posterior_pdf = post.pdf(x_range)
    ax.plot(x_range, posterior_pdf, 'b-', linewidth=2,
            label=f'Posterior for μ: N({post.mean:.2f}, {post.variance:.3f})')
    
    # Predictive for x_{n+1}
    predictive_pdf = stats.norm.pdf(x_range, pred_mean, np.sqrt(pred_var))
    ax.fill_between(x_range, predictive_pdf, alpha=0.3, color='orange')
    ax.plot(x_range, predictive_pdf, 'orange', linewidth=2,
            label=f'Predictive for x: N({pred_mean:.2f}, {pred_var:.3f})')
    
    if true_mu is not None:
        ax.axvline(true_mu, color='red', linestyle='--', linewidth=2,
                   label=f'True μ = {true_mu}')
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Posterior vs Predictive Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Annotate variance decomposition
    textstr = (f'Predictive Var = {pred_var:.3f}\n'
               f'  = Aleatoric ({model.known_variance:.3f})\n'
               f'  + Epistemic ({post.variance:.3f})')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


# =============================================================================
# Demonstrations
# =============================================================================

def demo_basic_inference():
    """Demonstrate basic Gaussian inference."""
    
    print("=" * 60)
    print("GAUSSIAN INFERENCE WITH KNOWN VARIANCE")
    print("=" * 60)
    
    # Setup
    true_mu = 5.0
    known_var = 4.0  # σ² = 4, so σ = 2
    
    np.random.seed(42)
    n = 10
    data = np.random.normal(true_mu, np.sqrt(known_var), n)
    
    print(f"\nTrue μ: {true_mu}")
    print(f"Known σ²: {known_var}")
    print(f"Sample: n = {n}, x̄ = {data.mean():.4f}")
    print(f"MLE: {data.mean():.4f}")
    
    # Different priors
    priors = [
        ("Weak prior (σ₀² = 100)", 0.0, 100.0),
        ("Moderate prior", 3.0, 4.0),
        ("Strong prior (wrong)", 10.0, 1.0),
        ("Strong prior (right)", 5.0, 1.0),
    ]
    
    print("\nPosterior summaries under different priors:")
    print("-" * 60)
    
    for name, mu0, var0 in priors:
        model = GaussianKnownVarianceModel(mu0, var0, known_var)
        model.update(data)
        post = model.posterior
        ci = post.credible_interval(0.95)
        
        print(f"\n{name}")
        print(f"  Prior: N({mu0}, {var0})")
        print(f"  Posterior: {post}")
        print(f"  Prior weight: {model.prior_weight():.1%}")
        print(f"  Data weight: {model.data_weight():.1%}")
        print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")


def demo_sequential_updating():
    """Demonstrate sequential updating."""
    
    print("\n" + "=" * 60)
    print("SEQUENTIAL BAYESIAN UPDATING")
    print("=" * 60)
    
    true_mu = 100.0
    known_var = 25.0  # σ = 5
    
    np.random.seed(123)
    data = np.random.normal(true_mu, np.sqrt(known_var), 20)
    
    # Start with wrong prior
    prior_mean = 80.0
    prior_var = 100.0
    
    print(f"\nTrue μ: {true_mu}")
    print(f"Prior: N({prior_mean}, {prior_var}) [wrong!]")
    print(f"Known σ²: {known_var}")
    
    model = GaussianKnownVarianceModel(prior_mean, prior_var, known_var)
    
    print("\nPosterior evolution:")
    print("-" * 50)
    print(f"{'n':>4} {'x':>8} {'E[μ|D]':>10} {'σ_post':>10} {'Data Wt':>10}")
    print("-" * 50)
    
    for i, x in enumerate(data[:10]):
        model.update_single(x)
        print(f"{i+1:4d} {x:8.2f} {model.current_mean:10.3f} "
              f"{model.posterior.std:10.3f} {model.data_weight():10.1%}")
    
    # Create visualization
    model._reset()
    fig = plot_sequential_updating(data, prior_mean, prior_var, known_var, true_mu)
    fig.savefig('gaussian_sequential_update.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSee: gaussian_sequential_update.png")


def demo_predictive():
    """Demonstrate predictive distribution."""
    
    print("\n" + "=" * 60)
    print("POSTERIOR PREDICTIVE DISTRIBUTION")
    print("=" * 60)
    
    true_mu = 50.0
    known_var = 16.0  # σ = 4
    
    np.random.seed(456)
    data = np.random.normal(true_mu, np.sqrt(known_var), 15)
    
    model = GaussianKnownVarianceModel(
        prior_mean=45.0,
        prior_variance=25.0,
        known_variance=known_var
    )
    model.update(data)
    
    pred_mean, pred_var = model.predictive_distribution()
    post = model.posterior
    
    print(f"\nObserved: {len(data)} observations")
    print(f"Posterior for μ: N({post.mean:.2f}, {post.variance:.4f})")
    print(f"\nPredictive for x_{len(data)+1}:")
    print(f"  Mean: {pred_mean:.2f}")
    print(f"  Variance: {pred_var:.4f}")
    print(f"    = Aleatoric ({known_var:.4f}) + Epistemic ({post.variance:.4f})")
    
    # 95% prediction interval
    z = 1.96
    pi_lower = pred_mean - z * np.sqrt(pred_var)
    pi_upper = pred_mean + z * np.sqrt(pred_var)
    print(f"  95% Prediction Interval: [{pi_lower:.2f}, {pi_upper:.2f}]")
    
    fig = plot_predictive_distribution(model, true_mu)
    fig.savefig('gaussian_predictive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSee: gaussian_predictive.png")


if __name__ == "__main__":
    demo_basic_inference()
    demo_sequential_updating()
    demo_predictive()
```

---

## Summary

| Aspect | Formula |
|--------|---------|
| **Prior** | $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$ |
| **Likelihood** | $p(\mathcal{D} \mid \mu) \propto \exp\left(-\frac{n(\bar{x}-\mu)^2}{2\sigma^2}\right)$ |
| **Posterior** | $\mu \mid \mathcal{D} \sim \mathcal{N}(\mu_n, \sigma_n^2)$ |
| **Posterior precision** | $\tau_n = \tau_0 + n\tau$ |
| **Posterior mean** | $\mu_n = \frac{\tau_0\mu_0 + n\tau\bar{x}}{\tau_n}$ |
| **Predictive** | $x_{n+1} \mid \mathcal{D} \sim \mathcal{N}(\mu_n, \sigma^2 + \sigma_n^2)$ |

### Key Insights

1. **Precisions add**: Posterior precision = prior precision + data precision
2. **Precision-weighted averaging**: Posterior mean weights sources by precision
3. **Equivalent sample size**: Prior worth $n_0 = \sigma^2/\sigma_0^2$ observations
4. **Shrinkage**: Posterior mean shrinks MLE toward prior mean
5. **Predictive variance**: Decomposes into aleatoric + epistemic
6. **Kalman filter**: Sequential updating is simplest Kalman filter

### Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Beta-Bernoulli | Ch13: Bernoulli-Beta | Discrete analog |
| Unknown variance | Ch13: Gaussian Unknown Variance | More realistic setting |
| Multivariate | Ch13: Bayesian Linear Regression | Extension to regression |
| Kalman filter | Ch16: State Space Models | Sequential inference |
| BNN priors | Ch13: BNN Priors | Weight distribution design |

### Key References

- DeGroot, M. H. (1970). *Optimal Statistical Decisions*. McGraw-Hill.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapter 2.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. Chapter 4.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Section 2.3.

---

# Gaussian Inference with Unknown Variance

When both the mean $\mu$ and variance $\sigma^2$ of a Gaussian distribution are unknown, Bayesian inference requires a joint prior over both parameters. The conjugate prior is the **Normal-Inverse-Gamma** (NIG) distribution, which leads to elegant closed-form posteriors. This setting is far more realistic than known variance and introduces the important concept of **marginalizing over nuisance parameters**.

---

## Problem Setup

### The Inference Problem

We observe continuous measurements $x_1, x_2, \ldots, x_n \in \mathbb{R}$ assumed to be drawn from a Gaussian distribution with **unknown mean** $\mu$ and **unknown variance** $\sigma^2$:

$$
x_i \mid \mu, \sigma^2 \sim \mathcal{N}(\mu, \sigma^2)
$$

The goal is to infer the joint posterior distribution $p(\mu, \sigma^2 \mid \mathcal{D})$ and, importantly, the **marginal posterior** for $\mu$ alone:

$$
p(\mu \mid \mathcal{D}) = \int_0^\infty p(\mu, \sigma^2 \mid \mathcal{D}) \, d\sigma^2
$$

### Why Unknown Variance Matters

In most real applications, variance is unknown:

- **Scientific experiments**: Measurement precision varies between instruments
- **Financial data**: Volatility changes over time
- **A/B testing**: Effect size variability is typically unknown
- **Machine learning**: Model uncertainty estimation

The known-variance assumption, while pedagogically useful, is rarely realistic.

### The Gaussian Likelihood

For $n$ independent observations:

$$
p(\mathcal{D} \mid \mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

$$
= (2\pi\sigma^2)^{-n/2} \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2\right)
$$

### Sufficient Statistics

The likelihood depends on the data only through two sufficient statistics:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i, \quad s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

Using the decomposition:

$$
\sum_{i=1}^{n}(x_i - \mu)^2 = (n-1)s^2 + n(\bar{x} - \mu)^2
$$

The likelihood becomes:

$$
p(\mathcal{D} \mid \mu, \sigma^2) \propto (\sigma^2)^{-n/2} \exp\left(-\frac{(n-1)s^2 + n(\bar{x} - \mu)^2}{2\sigma^2}\right)
$$

---

## The Inverse-Gamma Distribution

### Definition

The **Inverse-Gamma** distribution is the conjugate prior for the variance parameter. If $\sigma^2 \sim \text{Inv-Gamma}(\alpha, \beta)$:

$$
p(\sigma^2) = \frac{\beta^\alpha}{\Gamma(\alpha)} (\sigma^2)^{-\alpha-1} \exp\left(-\frac{\beta}{\sigma^2}\right), \quad \sigma^2 > 0
$$

### Parameters and Moments

| Parameter | Symbol | Interpretation |
|-----------|--------|----------------|
| Shape | $\alpha$ | Controls concentration (larger = more peaked) |
| Scale | $\beta$ | Controls location (larger = larger variances) |

**Moments** (for $\alpha > 1$ and $\alpha > 2$ respectively):

$$
\mathbb{E}[\sigma^2] = \frac{\beta}{\alpha - 1}, \quad \text{Var}[\sigma^2] = \frac{\beta^2}{(\alpha-1)^2(\alpha-2)}
$$

**Mode**:

$$
\text{Mode}[\sigma^2] = \frac{\beta}{\alpha + 1}
$$

### Connection to Chi-Square

If $X \sim \chi^2_\nu$, then:

$$
\frac{\nu s_0^2}{X} \sim \text{Inv-Gamma}\left(\frac{\nu}{2}, \frac{\nu s_0^2}{2}\right)
$$

This connects to the sampling distribution of the sample variance.

### Why Inverse-Gamma?

The Inverse-Gamma is natural for variance because:

1. **Support**: Defined on $(0, \infty)$, matching variance's domain
2. **Conjugacy**: Leads to closed-form posteriors
3. **Interpretability**: Parameters relate to "prior observations"

---

## The Normal-Inverse-Gamma Prior

### Joint Prior Specification

The conjugate prior for $(\mu, \sigma^2)$ is the **Normal-Inverse-Gamma** (NIG) distribution:

$$
\sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)
$$

$$
\mu \mid \sigma^2 \sim \mathcal{N}\left(\mu_0, \frac{\sigma^2}{\kappa_0}\right)
$$

This is written as:

$$
(\mu, \sigma^2) \sim \text{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)
$$

### Joint Density

$$
p(\mu, \sigma^2) = p(\mu \mid \sigma^2) \cdot p(\sigma^2)
$$

$$
= \frac{1}{\sqrt{2\pi\sigma^2/\kappa_0}} \exp\left(-\frac{\kappa_0(\mu - \mu_0)^2}{2\sigma^2}\right) \cdot \frac{\beta_0^{\alpha_0}}{\Gamma(\alpha_0)} (\sigma^2)^{-\alpha_0-1} \exp\left(-\frac{\beta_0}{\sigma^2}\right)
$$

$$
\propto (\sigma^2)^{-\alpha_0-3/2} \exp\left(-\frac{1}{\sigma^2}\left[\beta_0 + \frac{\kappa_0}{2}(\mu - \mu_0)^2\right]\right)
$$

### Prior Parameters Interpretation

| Parameter | Symbol | Interpretation |
|-----------|--------|----------------|
| Prior mean location | $\mu_0$ | Best guess for $\mu$ before seeing data |
| Prior precision scaling | $\kappa_0$ | "Equivalent observations" for mean |
| Variance shape | $\alpha_0$ | Half the "prior degrees of freedom" |
| Variance scale | $\beta_0$ | Scales the prior variance estimate |

**Pseudo-observation interpretation**:

- $\kappa_0$: Prior worth $\kappa_0$ observations for estimating $\mu$
- $2\alpha_0$: Prior worth $2\alpha_0$ observations for estimating $\sigma^2$
- $\beta_0 / \alpha_0$: Prior estimate of $\sigma^2$ (at the mode)

### Common Prior Choices

**Weakly informative prior**:

$$
\mu_0 = 0, \quad \kappa_0 = 0.01, \quad \alpha_0 = 0.01, \quad \beta_0 = 0.01
$$

**Jeffrey's prior** (improper but reference):

$$
p(\mu, \sigma^2) \propto \frac{1}{\sigma^2}
$$

This corresponds to $\kappa_0 \to 0$, $\alpha_0 \to 0$, $\beta_0 \to 0$.

**Data-dependent prior** (empirical Bayes style):

$$
\mu_0 = \bar{x}_{\text{pilot}}, \quad \alpha_0 = 1, \quad \beta_0 = s^2_{\text{pilot}}
$$

---

## Conjugate Posterior Derivation

### The Derivation

**Prior**:

$$
p(\mu, \sigma^2) \propto (\sigma^2)^{-\alpha_0-3/2} \exp\left(-\frac{1}{\sigma^2}\left[\beta_0 + \frac{\kappa_0}{2}(\mu - \mu_0)^2\right]\right)
$$

**Likelihood**:

$$
p(\mathcal{D} \mid \mu, \sigma^2) \propto (\sigma^2)^{-n/2} \exp\left(-\frac{1}{2\sigma^2}\left[(n-1)s^2 + n(\bar{x} - \mu)^2\right]\right)
$$

**Posterior** (via Bayes' theorem):

$$
p(\mu, \sigma^2 \mid \mathcal{D}) \propto p(\mathcal{D} \mid \mu, \sigma^2) \cdot p(\mu, \sigma^2)
$$

### Combining Exponents

The exponent in $\sigma^2$ becomes:

$$
-\alpha_0 - \frac{3}{2} - \frac{n}{2} = -\left(\alpha_0 + \frac{n}{2}\right) - \frac{3}{2} = -\alpha_n - \frac{3}{2}
$$

where $\alpha_n = \alpha_0 + n/2$.

The terms inside the exponential:

$$
\beta_0 + \frac{\kappa_0}{2}(\mu - \mu_0)^2 + \frac{(n-1)s^2}{2} + \frac{n}{2}(\bar{x} - \mu)^2
$$

### Completing the Square for $\mu$

The $\mu$-dependent terms:

$$
\frac{\kappa_0}{2}(\mu - \mu_0)^2 + \frac{n}{2}(\bar{x} - \mu)^2
$$

$$
= \frac{\kappa_0}{2}\left[\mu^2 - 2\mu\mu_0 + \mu_0^2\right] + \frac{n}{2}\left[\mu^2 - 2\mu\bar{x} + \bar{x}^2\right]
$$

$$
= \frac{\kappa_0 + n}{2}\mu^2 - \mu(\kappa_0\mu_0 + n\bar{x}) + \frac{\kappa_0\mu_0^2 + n\bar{x}^2}{2}
$$

Completing the square:

$$
= \frac{\kappa_n}{2}\left(\mu - \mu_n\right)^2 + \text{const}
$$

where:

$$
\kappa_n = \kappa_0 + n, \quad \mu_n = \frac{\kappa_0\mu_0 + n\bar{x}}{\kappa_0 + n}
$$

### The $\beta_n$ Update

The constant term (independent of $\mu$) contributes to $\beta_n$:

$$
\beta_n = \beta_0 + \frac{(n-1)s^2}{2} + \frac{\kappa_0 n(\bar{x} - \mu_0)^2}{2(\kappa_0 + n)}
$$

The last term arises from completing the square and represents the "prior-data conflict" for the mean.

### The Posterior Distribution

$$
\boxed{(\mu, \sigma^2) \mid \mathcal{D} \sim \text{NIG}(\mu_n, \kappa_n, \alpha_n, \beta_n)}
$$

with update formulas:

$$
\boxed{
\begin{aligned}
\kappa_n &= \kappa_0 + n \\
\mu_n &= \frac{\kappa_0\mu_0 + n\bar{x}}{\kappa_n} \\
\alpha_n &= \alpha_0 + \frac{n}{2} \\
\beta_n &= \beta_0 + \frac{(n-1)s^2}{2} + \frac{\kappa_0 n(\bar{x} - \mu_0)^2}{2\kappa_n}
\end{aligned}
}
$$

---

## Marginal Posterior Distributions

### Marginal for $\sigma^2$

Integrating out $\mu$:

$$
p(\sigma^2 \mid \mathcal{D}) = \int_{-\infty}^{\infty} p(\mu, \sigma^2 \mid \mathcal{D}) \, d\mu
$$

$$
\sigma^2 \mid \mathcal{D} \sim \text{Inv-Gamma}(\alpha_n, \beta_n)
$$

**Point estimates**:

$$
\mathbb{E}[\sigma^2 \mid \mathcal{D}] = \frac{\beta_n}{\alpha_n - 1} \quad (\text{if } \alpha_n > 1)
$$

$$
\text{Mode}[\sigma^2 \mid \mathcal{D}] = \frac{\beta_n}{\alpha_n + 1}
$$

### Marginal for $\mu$: The Student-t Distribution

Integrating out $\sigma^2$:

$$
p(\mu \mid \mathcal{D}) = \int_0^\infty p(\mu, \sigma^2 \mid \mathcal{D}) \, d\sigma^2
$$

This integral yields a **Student-t distribution**:

$$
\boxed{\mu \mid \mathcal{D} \sim t_{2\alpha_n}\left(\mu_n, \frac{\beta_n}{\alpha_n \kappa_n}\right)}
$$

where $t_\nu(\mu, \sigma^2)$ denotes a Student-t with $\nu$ degrees of freedom, location $\mu$, and scale $\sigma$.

### The Student-t Distribution

The Student-t pdf with $\nu$ degrees of freedom, location $\mu$, and scale $\sigma$:

$$
p(x) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\sigma} \left(1 + \frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right)^{-\frac{\nu+1}{2}}
$$

**Properties**:

| Property | Formula |
|----------|---------|
| Mean | $\mu$ (if $\nu > 1$) |
| Variance | $\frac{\nu}{\nu-2}\sigma^2$ (if $\nu > 2$) |
| Degrees of freedom | $\nu = 2\alpha_n$ |

### Why Student-t?

The Student-t arises because:

1. **Uncertainty about $\sigma^2$** makes tails heavier than Gaussian
2. **With more data**, $\nu \to \infty$ and $t_\nu \to \mathcal{N}$
3. **Robustness**: Heavier tails accommodate outliers

---

## Posterior Analysis

### Point Estimates for $\mu$

**Posterior mean** (equals posterior mode for Student-t):

$$
\mathbb{E}[\mu \mid \mathcal{D}] = \mu_n = \frac{\kappa_0\mu_0 + n\bar{x}}{\kappa_0 + n}
$$

This is the same precision-weighted average as in the known-variance case!

**Posterior variance** (if $\nu = 2\alpha_n > 2$):

$$
\text{Var}[\mu \mid \mathcal{D}] = \frac{\nu}{\nu - 2} \cdot \frac{\beta_n}{\alpha_n \kappa_n} = \frac{\beta_n}{(\alpha_n - 1)\kappa_n}
$$

### Point Estimates for $\sigma^2$

**Posterior mean**:

$$
\mathbb{E}[\sigma^2 \mid \mathcal{D}] = \frac{\beta_n}{\alpha_n - 1}
$$

**Posterior mode**:

$$
\text{Mode}[\sigma^2 \mid \mathcal{D}] = \frac{\beta_n}{\alpha_n + 1}
$$

### Credible Intervals

**For $\mu$**: Use Student-t quantiles

$$
\left[\mu_n - t_{\nu, \alpha/2} \cdot \sqrt{\frac{\beta_n}{\alpha_n\kappa_n}}, \; \mu_n + t_{\nu, \alpha/2} \cdot \sqrt{\frac{\beta_n}{\alpha_n\kappa_n}}\right]
$$

**For $\sigma^2$**: Use Inverse-Gamma quantiles (asymmetric interval)

---

## Connection to Frequentist Inference

### The $t$-Test Connection

With Jeffrey's prior ($\kappa_0 \to 0$, $\alpha_0 \to 0$, $\beta_0 \to 0$):

$$
\mu_n \to \bar{x}, \quad \kappa_n \to n, \quad \alpha_n \to \frac{n}{2}, \quad \beta_n \to \frac{(n-1)s^2}{2}
$$

The marginal posterior for $\mu$ becomes:

$$
\mu \mid \mathcal{D} \sim t_{n-1}\left(\bar{x}, \frac{s^2}{n}\right)
$$

This matches the frequentist sampling distribution used in the one-sample $t$-test!

### Comparison Table

| Aspect | Bayesian (Jeffrey's) | Frequentist |
|--------|---------------------|-------------|
| Point estimate | $\bar{x}$ | $\bar{x}$ |
| Interval for $\mu$ | $\bar{x} \pm t_{n-1,\alpha/2} \cdot \frac{s}{\sqrt{n}}$ | $\bar{x} \pm t_{n-1,\alpha/2} \cdot \frac{s}{\sqrt{n}}$ |
| Distribution | Posterior (probability for $\mu$) | Sampling distribution |
| Interpretation | "95% probability $\mu$ in interval" | "95% of intervals contain $\mu$" |

### The Remarkable Agreement

With Jeffrey's prior, Bayesian credible intervals exactly match frequentist confidence intervals. This is not a coincidence—it reflects the deep connection between:

- **Jeffrey's prior**: Designed for "objective" Bayesian inference
- **Maximum likelihood**: Asymptotically efficient under regularity conditions

---

## Sequential Updating

### Online Learning

The NIG conjugate family enables sequential updating:

$$
\text{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0) \xrightarrow{x_1} \text{NIG}(\mu_1, \kappa_1, \alpha_1, \beta_1) \xrightarrow{x_2} \cdots
$$

**Single observation update** (given observation $x$):

$$
\begin{aligned}
\kappa_{t+1} &= \kappa_t + 1 \\
\mu_{t+1} &= \frac{\kappa_t \mu_t + x}{\kappa_{t+1}} \\
\alpha_{t+1} &= \alpha_t + \frac{1}{2} \\
\beta_{t+1} &= \beta_t + \frac{\kappa_t(x - \mu_t)^2}{2\kappa_{t+1}}
\end{aligned}
$$

### Interpretation

- $\kappa_t$ and $\alpha_t$ grow linearly with observations
- $\mu_t$ converges to the sample mean
- $\beta_t$ accumulates squared deviations, scaled appropriately

---

## Posterior Predictive Distribution

### Predicting New Observations

The posterior predictive integrates over both unknown parameters:

$$
p(x_{n+1} \mid \mathcal{D}) = \int_0^\infty \int_{-\infty}^\infty p(x_{n+1} \mid \mu, \sigma^2) \, p(\mu, \sigma^2 \mid \mathcal{D}) \, d\mu \, d\sigma^2
$$

**Result**: The predictive distribution is also Student-t:

$$
\boxed{x_{n+1} \mid \mathcal{D} \sim t_{2\alpha_n}\left(\mu_n, \frac{\beta_n(\kappa_n + 1)}{\alpha_n \kappa_n}\right)}
$$

### Predictive Variance Decomposition

For $\nu = 2\alpha_n > 2$:

$$
\text{Var}[x_{n+1} \mid \mathcal{D}] = \frac{\nu}{\nu - 2} \cdot \frac{\beta_n(\kappa_n + 1)}{\alpha_n \kappa_n}
$$

This variance has three components:

1. **Aleatoric**: Inherent randomness in observations
2. **Epistemic (mean)**: Uncertainty about $\mu$
3. **Epistemic (variance)**: Uncertainty about $\sigma^2$

As $n \to \infty$, only aleatoric uncertainty remains.

---

## Asymptotic Behavior

### Large Sample Limits

As $n \to \infty$:

**Posterior for $\mu$**:

$$
\mu \mid \mathcal{D} \xrightarrow{d} \mathcal{N}\left(\bar{x}, \frac{s^2}{n}\right)
$$

The Student-t converges to Gaussian as degrees of freedom increase.

**Posterior for $\sigma^2$**:

$$
\sigma^2 \mid \mathcal{D} \xrightarrow{p} s^2
$$

The posterior concentrates around the sample variance.

### Prior Washout

With enough data, reasonable priors are "washed out":

$$
\frac{\kappa_0}{\kappa_n} \to 0, \quad \frac{\alpha_0}{\alpha_n} \to 0
$$

The posterior is dominated by the likelihood.

---

## Numerical Stability Considerations

### Computing $\beta_n$

The formula for $\beta_n$ can suffer from numerical issues. A more stable form:

$$
\beta_n = \beta_0 + \frac{1}{2}\left[\sum_{i=1}^n (x_i - \bar{x})^2 + \frac{\kappa_0 n}{\kappa_n}(\bar{x} - \mu_0)^2\right]
$$

Using Welford's algorithm for the sum of squares ensures numerical stability.

### Log-Space Computations

For evaluating densities, work in log-space:

$$
\log p(\sigma^2 \mid \mathcal{D}) = \alpha_n \log \beta_n - \log\Gamma(\alpha_n) - (\alpha_n + 1)\log\sigma^2 - \frac{\beta_n}{\sigma^2}
$$

---

## Python Implementation

```python
"""
Gaussian Inference with Unknown Variance: Complete Implementation

This module provides Bayesian inference for the mean and variance of a 
Gaussian distribution using the Normal-Inverse-Gamma conjugate prior,
demonstrating the Student-t marginal posterior for the mean.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class NIGParameters:
    """
    Normal-Inverse-Gamma distribution parameters.
    
    The NIG distribution is parameterized as:
        σ² ~ Inv-Gamma(α, β)
        μ | σ² ~ N(μ₀, σ²/κ)
    
    Attributes
    ----------
    mu : float
        Location parameter μ₀
    kappa : float
        Precision scaling κ (effective sample size for mean)
    alpha : float
        Shape parameter α for variance
    beta : float
        Scale parameter β for variance
    """
    mu: float
    kappa: float
    alpha: float
    beta: float
    
    def __post_init__(self):
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
    
    @property
    def variance_mean(self) -> Optional[float]:
        """E[σ²] = β/(α-1) if α > 1."""
        if self.alpha > 1:
            return self.beta / (self.alpha - 1)
        return None
    
    @property
    def variance_mode(self) -> float:
        """Mode[σ²] = β/(α+1)."""
        return self.beta / (self.alpha + 1)
    
    @property
    def degrees_of_freedom(self) -> float:
        """Degrees of freedom for marginal t-distribution on μ."""
        return 2 * self.alpha
    
    @property
    def mu_scale(self) -> float:
        """Scale parameter for marginal t-distribution on μ."""
        return np.sqrt(self.beta / (self.alpha * self.kappa))
    
    def __repr__(self) -> str:
        return f"NIG(μ={self.mu:.4f}, κ={self.kappa:.4f}, α={self.alpha:.4f}, β={self.beta:.4f})"


class StudentTPosterior:
    """
    Represents the marginal Student-t posterior for μ.
    
    Parameters
    ----------
    loc : float
        Location parameter (posterior mean)
    scale : float
        Scale parameter
    df : float
        Degrees of freedom
    """
    
    def __init__(self, loc: float, scale: float, df: float):
        self.loc = loc
        self.scale = scale
        self.df = df
        self._dist = stats.t(df=df, loc=loc, scale=scale)
    
    @property
    def mean(self) -> Optional[float]:
        """Mean exists if df > 1."""
        return self.loc if self.df > 1 else None
    
    @property
    def variance(self) -> Optional[float]:
        """Variance exists if df > 2."""
        if self.df > 2:
            return (self.df / (self.df - 2)) * self.scale**2
        return None
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate posterior density."""
        return self._dist.pdf(x)
    
    def cdf(self, x: float) -> float:
        """Evaluate posterior CDF."""
        return self._dist.cdf(x)
    
    def quantile(self, p: float) -> float:
        """Compute posterior quantile."""
        return self._dist.ppf(p)
    
    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Compute equal-tailed credible interval."""
        alpha = 1 - level
        return (self.quantile(alpha/2), self.quantile(1 - alpha/2))
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Draw samples from posterior."""
        return self._dist.rvs(n_samples)
    
    def __repr__(self) -> str:
        return f"t_{self.df:.1f}({self.loc:.4f}, {self.scale:.4f})"


class InverseGammaPosterior:
    """
    Represents the marginal Inverse-Gamma posterior for σ².
    
    Parameters
    ----------
    alpha : float
        Shape parameter
    beta : float
        Scale parameter
    """
    
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self._dist = stats.invgamma(a=alpha, scale=beta)
    
    @property
    def mean(self) -> Optional[float]:
        """Mean exists if α > 1."""
        return self.beta / (self.alpha - 1) if self.alpha > 1 else None
    
    @property
    def mode(self) -> float:
        """Mode = β/(α+1)."""
        return self.beta / (self.alpha + 1)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate posterior density."""
        return self._dist.pdf(x)
    
    def cdf(self, x: float) -> float:
        """Evaluate posterior CDF."""
        return self._dist.cdf(x)
    
    def quantile(self, p: float) -> float:
        """Compute posterior quantile."""
        return self._dist.ppf(p)
    
    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Compute equal-tailed credible interval."""
        alpha = 1 - level
        return (self.quantile(alpha/2), self.quantile(1 - alpha/2))
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Draw samples from posterior."""
        return self._dist.rvs(n_samples)
    
    def __repr__(self) -> str:
        return f"Inv-Gamma({self.alpha:.4f}, {self.beta:.4f})"


class GaussianUnknownVarianceModel:
    """
    Bayesian inference for Gaussian with unknown mean and variance.
    
    Uses Normal-Inverse-Gamma conjugate prior.
    
    Parameters
    ----------
    prior_mu : float
        Prior mean location μ₀
    prior_kappa : float
        Prior precision scaling κ₀
    prior_alpha : float
        Prior shape α₀
    prior_beta : float
        Prior scale β₀
    """
    
    def __init__(
        self,
        prior_mu: float = 0.0,
        prior_kappa: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ):
        self.prior = NIGParameters(prior_mu, prior_kappa, prior_alpha, prior_beta)
        self._reset()
    
    def _reset(self):
        """Reset to prior state."""
        self.current = NIGParameters(
            self.prior.mu, self.prior.kappa, 
            self.prior.alpha, self.prior.beta
        )
        self.n_observations = 0
        self._data_sum = 0.0
        self._data_sum_sq = 0.0
    
    @property
    def posterior_nig(self) -> NIGParameters:
        """Return current NIG posterior parameters."""
        return self.current
    
    @property
    def posterior_mu(self) -> StudentTPosterior:
        """Return marginal posterior for μ (Student-t)."""
        return StudentTPosterior(
            loc=self.current.mu,
            scale=self.current.mu_scale,
            df=self.current.degrees_of_freedom
        )
    
    @property
    def posterior_variance(self) -> InverseGammaPosterior:
        """Return marginal posterior for σ² (Inverse-Gamma)."""
        return InverseGammaPosterior(
            alpha=self.current.alpha,
            beta=self.current.beta
        )
    
    def update(self, data: np.ndarray) -> NIGParameters:
        """
        Update posterior with new observations.
        
        Parameters
        ----------
        data : array
            New observations
        
        Returns
        -------
        NIGParameters
            Updated posterior parameters
        """
        data = np.atleast_1d(data).astype(float)
        n = len(data)
        
        if n == 0:
            return self.current
        
        # Update sufficient statistics
        self.n_observations += n
        self._data_sum += data.sum()
        self._data_sum_sq += (data**2).sum()
        
        # Overall sample mean
        overall_mean = self._data_sum / self.n_observations
        
        # Compute sample variance (using all data)
        if self.n_observations > 1:
            ss = self._data_sum_sq - self.n_observations * overall_mean**2
        else:
            ss = 0.0
        
        # NIG update formulas
        kappa_n = self.prior.kappa + self.n_observations
        mu_n = (self.prior.kappa * self.prior.mu + self._data_sum) / kappa_n
        alpha_n = self.prior.alpha + self.n_observations / 2
        
        # Beta update
        prior_data_sq = (self.prior.kappa * self.n_observations / kappa_n) * \
                        (overall_mean - self.prior.mu)**2
        beta_n = self.prior.beta + 0.5 * ss + 0.5 * prior_data_sq
        
        self.current = NIGParameters(mu_n, kappa_n, alpha_n, beta_n)
        return self.current
    
    def update_single(self, x: float) -> NIGParameters:
        """Update with a single observation using online formulas."""
        kappa_old = self.current.kappa
        mu_old = self.current.mu
        
        # Update parameters
        kappa_new = kappa_old + 1
        mu_new = (kappa_old * mu_old + x) / kappa_new
        alpha_new = self.current.alpha + 0.5
        beta_new = self.current.beta + (kappa_old * (x - mu_old)**2) / (2 * kappa_new)
        
        self.current = NIGParameters(mu_new, kappa_new, alpha_new, beta_new)
        self.n_observations += 1
        self._data_sum += x
        self._data_sum_sq += x**2
        
        return self.current
    
    def update_sequential(self, data: np.ndarray) -> List[NIGParameters]:
        """Update sequentially, returning posterior history."""
        self._reset()
        history = [self.current]
        
        for x in data:
            self.update_single(x)
            history.append(self.current)
        
        return history
    
    def predictive_distribution(self) -> StudentTPosterior:
        """Compute posterior predictive distribution for next observation."""
        pred_scale = np.sqrt(
            self.current.beta * (self.current.kappa + 1) / 
            (self.current.alpha * self.current.kappa)
        )
        return StudentTPosterior(
            loc=self.current.mu,
            scale=pred_scale,
            df=self.current.degrees_of_freedom
        )
    
    def sample_posterior(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Draw joint samples from posterior (mu, sigma2)."""
        # First sample σ² from Inverse-Gamma
        sigma2_samples = self.posterior_variance.sample(n_samples)
        
        # Then sample μ | σ² from Normal
        mu_std = np.sqrt(sigma2_samples / self.current.kappa)
        mu_samples = np.random.normal(self.current.mu, mu_std)
        
        return mu_samples, sigma2_samples
    
    def log_marginal_likelihood(self, data: np.ndarray) -> float:
        """Compute log marginal likelihood (model evidence)."""
        data = np.atleast_1d(data)
        n = len(data)
        
        if n == 0:
            return 0.0
        
        # Compute posterior parameters
        x_bar = data.mean()
        ss = ((data - x_bar)**2).sum() if n > 1 else 0.0
        
        kappa_n = self.prior.kappa + n
        alpha_n = self.prior.alpha + n / 2
        prior_data_sq = (self.prior.kappa * n / kappa_n) * (x_bar - self.prior.mu)**2
        beta_n = self.prior.beta + 0.5 * ss + 0.5 * prior_data_sq
        
        # Log marginal likelihood
        log_ml = (
            gammaln(alpha_n) - gammaln(self.prior.alpha)
            + self.prior.alpha * np.log(self.prior.beta) - alpha_n * np.log(beta_n)
            + 0.5 * np.log(self.prior.kappa / kappa_n)
            - (n / 2) * np.log(2 * np.pi)
        )
        
        return log_ml


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_joint_posterior(
    model: GaussianUnknownVarianceModel,
    true_mu: Optional[float] = None,
    true_sigma2: Optional[float] = None,
    n_grid: int = 100
) -> plt.Figure:
    """Visualize joint and marginal posteriors."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    post_mu = model.posterior_mu
    post_var = model.posterior_variance
    
    # Determine plotting ranges
    mu_std = post_mu.scale * np.sqrt(post_mu.df / (post_mu.df - 2)) if post_mu.df > 2 else post_mu.scale * 3
    mu_range = (post_mu.loc - 4*mu_std, post_mu.loc + 4*mu_std)
    
    var_mean = post_var.mean if post_var.mean is not None else post_var.mode
    var_range = (max(0.01, var_mean * 0.1), var_mean * 3)
    
    mu_vals = np.linspace(mu_range[0], mu_range[1], n_grid)
    var_vals = np.linspace(var_range[0], var_range[1], n_grid)
    
    # Top-left: Joint posterior contour
    ax = axes[0, 0]
    MU, VAR = np.meshgrid(mu_vals, var_vals)
    
    joint_log_pdf = np.zeros_like(MU)
    for i, v in enumerate(var_vals):
        mu_given_var = stats.norm(loc=model.current.mu, scale=np.sqrt(v / model.current.kappa))
        joint_log_pdf[i, :] = mu_given_var.logpdf(mu_vals) + post_var._dist.logpdf(v)
    
    joint_pdf = np.exp(joint_log_pdf - joint_log_pdf.max())
    
    contour = ax.contourf(MU, VAR, joint_pdf, levels=20, cmap='Blues')
    if true_mu is not None:
        ax.axvline(true_mu, color='red', linestyle='--', linewidth=2)
    if true_sigma2 is not None:
        ax.axhline(true_sigma2, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('σ²', fontsize=12)
    ax.set_title('Joint Posterior p(μ, σ² | D)', fontsize=14)
    plt.colorbar(contour, ax=ax)
    
    # Top-right: Marginal for μ
    ax = axes[0, 1]
    ax.plot(mu_vals, post_mu.pdf(mu_vals), 'b-', linewidth=2, label=f'{post_mu}')
    ax.fill_between(mu_vals, post_mu.pdf(mu_vals), alpha=0.3)
    if true_mu is not None:
        ax.axvline(true_mu, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Marginal Posterior for μ (df = {post_mu.df:.0f})', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Marginal for σ²
    ax = axes[1, 0]
    ax.plot(var_vals, post_var.pdf(var_vals), 'b-', linewidth=2, label=f'{post_var}')
    ax.fill_between(var_vals, post_var.pdf(var_vals), alpha=0.3)
    if true_sigma2 is not None:
        ax.axvline(true_sigma2, color='red', linestyle='--', linewidth=2)
    ax.axvline(post_var.mode, color='green', linestyle=':', label=f'Mode = {post_var.mode:.3f}')
    ax.set_xlabel('σ²', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Marginal Posterior for σ²', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Posterior samples
    ax = axes[1, 1]
    mu_samples, var_samples = model.sample_posterior(1000)
    ax.scatter(mu_samples, var_samples, alpha=0.3, s=10, c='steelblue')
    if true_mu is not None and true_sigma2 is not None:
        ax.scatter([true_mu], [true_sigma2], color='red', s=100, marker='*', zorder=5)
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('σ²', fontsize=12)
    ax.set_title('Posterior Samples', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sequential_updating(
    data: np.ndarray,
    prior_mu: float,
    prior_kappa: float,
    prior_alpha: float,
    prior_beta: float,
    true_mu: Optional[float] = None,
    true_sigma2: Optional[float] = None
) -> plt.Figure:
    """Visualize sequential Bayesian updating."""
    
    model = GaussianUnknownVarianceModel(prior_mu, prior_kappa, prior_alpha, prior_beta)
    history = model.update_sequential(data)
    
    n_vals = np.arange(len(history))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Posterior mean for μ
    ax = axes[0, 0]
    mu_means = [h.mu for h in history]
    
    ci_lower, ci_upper = [], []
    for h in history:
        post = StudentTPosterior(h.mu, h.mu_scale, h.degrees_of_freedom)
        ci = post.credible_interval(0.95)
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])
    
    ax.fill_between(n_vals, ci_lower, ci_upper, alpha=0.3, color='steelblue', label='95% CI')
    ax.plot(n_vals, mu_means, 'b-', linewidth=2, marker='o', markersize=4, label='E[μ|D]')
    if true_mu is not None:
        ax.axhline(true_mu, color='red', linestyle='--', linewidth=2, label=f'True μ = {true_mu}')
    ax.set_xlabel('Observations', fontsize=12)
    ax.set_ylabel('μ', fontsize=12)
    ax.set_title('Posterior for Mean', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Top-right: Posterior mode for σ²
    ax = axes[0, 1]
    var_modes = [h.variance_mode for h in history]
    ax.plot(n_vals, var_modes, 'g-', linewidth=2, marker='s', markersize=4, label='Mode[σ²|D]')
    if true_sigma2 is not None:
        ax.axhline(true_sigma2, color='red', linestyle='--', linewidth=2, label=f'True σ² = {true_sigma2}')
    ax.set_xlabel('Observations', fontsize=12)
    ax.set_ylabel('σ²', fontsize=12)
    ax.set_title('Posterior for Variance', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Degrees of freedom
    ax = axes[1, 0]
    dfs = [h.degrees_of_freedom for h in history]
    ax.plot(n_vals, dfs, 'm-', linewidth=2, marker='d', markersize=4)
    ax.axhline(30, color='gray', linestyle=':', alpha=0.7, label='df=30 (≈Normal)')
    ax.set_xlabel('Observations', fontsize=12)
    ax.set_ylabel('Degrees of Freedom', fontsize=12)
    ax.set_title('Student-t df (2αₙ)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: κ and α growth
    ax = axes[1, 1]
    kappas = [h.kappa for h in history]
    alphas = [h.alpha for h in history]
    ax.plot(n_vals, kappas, 'b-', linewidth=2, label='κₙ')
    ax.plot(n_vals, alphas, 'g-', linewidth=2, label='αₙ')
    ax.set_xlabel('Observations', fontsize=12)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('NIG Parameter Evolution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Demonstrations
# =============================================================================

def demo_basic_inference():
    """Demonstrate basic inference with unknown variance."""
    
    print("=" * 70)
    print("GAUSSIAN INFERENCE WITH UNKNOWN VARIANCE")
    print("=" * 70)
    
    true_mu, true_sigma2 = 5.0, 4.0
    np.random.seed(42)
    data = np.random.normal(true_mu, np.sqrt(true_sigma2), 20)
    
    print(f"\nTrue: μ = {true_mu}, σ² = {true_sigma2}")
    print(f"Data: n = {len(data)}, x̄ = {data.mean():.4f}, s² = {data.var(ddof=1):.4f}")
    
    model = GaussianUnknownVarianceModel(0.0, 0.1, 0.1, 0.1)
    model.update(data)
    
    print(f"\nPosterior NIG: {model.posterior_nig}")
    print(f"Marginal for μ: {model.posterior_mu}")
    print(f"Marginal for σ²: {model.posterior_variance}")


def demo_t_test_connection():
    """Demonstrate connection to frequentist t-test."""
    
    print("\n" + "=" * 70)
    print("CONNECTION TO t-TEST")
    print("=" * 70)
    
    np.random.seed(456)
    data = np.random.normal(50, 10, 25)
    
    # Frequentist
    x_bar, s = data.mean(), data.std(ddof=1)
    t_crit = stats.t.ppf(0.975, df=len(data)-1)
    freq_ci = (x_bar - t_crit * s/np.sqrt(len(data)), x_bar + t_crit * s/np.sqrt(len(data)))
    
    # Bayesian with vague prior
    model = GaussianUnknownVarianceModel(0.0, 0.001, 0.001, 0.001)
    model.update(data)
    bayes_ci = model.posterior_mu.credible_interval(0.95)
    
    print(f"\nFrequentist 95% CI: [{freq_ci[0]:.4f}, {freq_ci[1]:.4f}]")
    print(f"Bayesian 95% CI:    [{bayes_ci[0]:.4f}, {bayes_ci[1]:.4f}]")
    print(f"Difference: {abs(freq_ci[1] - bayes_ci[1]):.6f}")


if __name__ == "__main__":
    demo_basic_inference()
    demo_t_test_connection()
```

---

## Summary

| Aspect | Formula |
|--------|---------|
| **Prior** | $(\mu, \sigma^2) \sim \text{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)$ |
| **Likelihood** | $p(\mathcal{D} \mid \mu, \sigma^2) \propto (\sigma^2)^{-n/2} \exp\left(-\frac{(n-1)s^2 + n(\bar{x}-\mu)^2}{2\sigma^2}\right)$ |
| **Posterior** | $(\mu, \sigma^2) \mid \mathcal{D} \sim \text{NIG}(\mu_n, \kappa_n, \alpha_n, \beta_n)$ |
| **Marginal for $\mu$** | $\mu \mid \mathcal{D} \sim t_{2\alpha_n}\left(\mu_n, \sqrt{\beta_n/(\alpha_n\kappa_n)}\right)$ |
| **Marginal for $\sigma^2$** | $\sigma^2 \mid \mathcal{D} \sim \text{Inv-Gamma}(\alpha_n, \beta_n)$ |
| **Predictive** | $x_{n+1} \mid \mathcal{D} \sim t_{2\alpha_n}\left(\mu_n, \sqrt{\beta_n(\kappa_n+1)/(\alpha_n\kappa_n)}\right)$ |

### Update Formulas

$$
\kappa_n = \kappa_0 + n, \quad \mu_n = \frac{\kappa_0\mu_0 + n\bar{x}}{\kappa_n}
$$

$$
\alpha_n = \alpha_0 + \frac{n}{2}, \quad \beta_n = \beta_0 + \frac{(n-1)s^2}{2} + \frac{\kappa_0 n(\bar{x} - \mu_0)^2}{2\kappa_n}
$$

### Key Insights

1. **Joint inference**: Must infer $\mu$ and $\sigma^2$ together
2. **Normal-Inverse-Gamma**: Conjugate prior for both parameters
3. **Student-t marginal**: Uncertainty about variance gives heavier tails
4. **t-test connection**: Jeffrey's prior yields frequentist intervals
5. **Sequential updating**: NIG family enables online learning
6. **Asymptotic normality**: Student-t approaches Gaussian as $n \to \infty$

### Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Known variance | Ch13: Gaussian Known Variance | Simpler special case |
| Bayesian regression | Ch13: Bayesian Linear Regression | Extension to multiple parameters |
| Model comparison | Ch13: Model Evidence | Marginal likelihood computation |
| BNN uncertainty | Ch13: BNN Uncertainty | Epistemic vs aleatoric |
| Robust inference | Ch8: Robust Methods | Student-t as robust likelihood |

### Key References

- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapter 3.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. Chapter 4.
- DeGroot, M. H. (1970). *Optimal Statistical Decisions*. McGraw-Hill.
- Box, G. E. P., & Tiao, G. C. (1973). *Bayesian Inference in Statistical Analysis*.
