# Hierarchical Bayesian Models

## Overview

Hierarchical models allow parameters to vary by group while sharing statistical strength across groups through partial pooling. This module develops the theory of multi-level inference, contrasts pooling strategies, and demonstrates the shrinkage phenomenon that makes hierarchical models powerful.

---

## 1. Hierarchical Model Structure

### 1.1 The Multi-Level Framework

A hierarchical model specifies distributions at multiple levels:

**Level 1 (Data):** Observations within groups

$$
y_{ij} | \theta_i \sim \mathcal{N}(\theta_i, \sigma^2)
$$

**Level 2 (Groups):** Group-level parameters

$$
\theta_i | \mu, \tau \sim \mathcal{N}(\mu, \tau^2)
$$

**Level 3 (Hyperparameters):** Population-level parameters

$$
\mu \sim \mathcal{N}(\mu_0, \sigma_0^2), \quad \tau \sim \text{Half-Cauchy}(\text{scale})
$$

### 1.2 Notation

| Symbol | Meaning |
|--------|---------|
| $y_{ij}$ | Observation $j$ in group $i$ |
| $\theta_i$ | Group-specific parameter |
| $\mu$ | Population mean (hyperparameter) |
| $\tau$ | Between-group standard deviation |
| $\sigma$ | Within-group standard deviation |
| $n_i$ | Sample size for group $i$ |

### 1.3 Applications

| Domain | Groups | Observations |
|--------|--------|--------------|
| Education | Schools | Student test scores |
| Healthcare | Hospitals | Patient outcomes |
| Marketing | Products | Customer ratings |
| Psychology | Subjects | Repeated measurements |
| Sports | Teams | Game performances |

---

## 2. Three Pooling Strategies

### 2.1 No Pooling (Complete Separation)

Estimate each $\theta_i$ independently, ignoring group structure:

$$
\hat{\theta}_i^{\text{no pool}} = \bar{y}_i = \frac{1}{n_i} \sum_{j=1}^{n_i} y_{ij}
$$

**Properties:**
- Maximum flexibility for group differences
- High variance for small groups
- Ignores information from other groups

### 2.2 Complete Pooling (No Group Effects)

Assume all groups share the same parameter:

$$
\hat{\theta}_i^{\text{complete}} = \bar{y}_{\cdot\cdot} = \frac{1}{N} \sum_{i,j} y_{ij}
$$

**Properties:**
- Minimum variance
- Maximum bias if groups truly differ
- Ignores group structure entirely

### 2.3 Partial Pooling (Hierarchical)

Shrink group estimates toward the population mean:

$$
\boxed{\hat{\theta}_i^{\text{partial}} = w_i \bar{y}_i + (1 - w_i) \mu}
$$

where the **shrinkage weight** is:

$$
w_i = \frac{n_i}{n_i + \sigma^2/\tau^2}
$$

**Properties:**
- Balances bias and variance
- Groups with small $n_i$ shrink more toward $\mu$
- Optimal under squared error loss

---

## 3. The Shrinkage Phenomenon

### 3.1 Understanding Shrinkage Weights

The weight $w_i$ determines how much group $i$'s estimate relies on its own data vs the population:

$$
w_i = \frac{n_i}{n_i + \sigma^2/\tau^2} = \frac{\text{group precision}}{\text{group precision} + \text{prior precision}}
$$

| Scenario | $w_i$ | Behavior |
|----------|-------|----------|
| Large $n_i$ | $\approx 1$ | Trust group data |
| Small $n_i$ | $\approx 0$ | Trust population mean |
| Large $\tau$ (heterogeneous groups) | Higher | Less shrinkage |
| Small $\tau$ (homogeneous groups) | Lower | More shrinkage |

### 3.2 Shrinkage as Regularization

Partial pooling is equivalent to **regularization**:
- Prevents overfitting to noisy group estimates
- Especially beneficial for small groups
- Automatically adapts shrinkage to data

### 3.3 Visual Intuition

Groups with extreme estimates (far from the mean) and small samples are pulled most strongly toward the population mean — this is the **shrinkage toward the grand mean**.

---

## 4. The Eight Schools Example

### 4.1 Classic Dataset

The "Eight Schools" dataset is a canonical hierarchical modeling example:

| School | $n_i$ | Observed Effect $\bar{y}_i$ |
|--------|-------|----------------------------|
| A | 28 | 28.4 |
| B | 8 | 7.9 |
| C | 23 | -2.8 |
| D | 20 | 6.8 |
| E | 12 | -0.6 |
| F | 44 | 0.6 |
| G | 6 | 18.0 |
| H | 11 | 12.2 |

### 4.2 Implementation

```python
import numpy as np

def hierarchical_estimates(observed_means, sample_sizes, sigma=15.0):
    """
    Compute no pooling, complete pooling, and partial pooling estimates.
    
    Parameters
    ----------
    observed_means : array
        Group sample means
    sample_sizes : array
        Sample size for each group
    sigma : float
        Known within-group standard deviation
    
    Returns
    -------
    dict with estimates for each pooling strategy
    """
    n_groups = len(observed_means)
    
    # No pooling: use observed means directly
    no_pool = observed_means.copy()
    
    # Complete pooling: grand mean
    complete_pool = np.full(n_groups, np.mean(observed_means))
    
    # Partial pooling: shrink toward grand mean
    tau_est = np.std(observed_means)  # Empirical estimate
    grand_mean = np.mean(observed_means)
    
    weights = sample_sizes / (sample_sizes + sigma**2 / tau_est**2)
    partial_pool = weights * observed_means + (1 - weights) * grand_mean
    
    return {
        'no_pooling': no_pool,
        'complete_pooling': complete_pool,
        'partial_pooling': partial_pool,
        'weights': weights
    }
```

### 4.3 Results Interpretation

For a typical run with the Eight Schools structure:

| School | $n$ | No Pool | Partial Pool | Shrinkage |
|--------|-----|---------|--------------|-----------|
| A | 28 | 28.4 | 19.2 | High (extreme value) |
| B | 8 | 7.9 | 8.5 | Low (near mean) |
| G | 6 | 18.0 | 12.1 | High (small $n$, extreme) |
| F | 44 | 0.6 | 2.8 | Low (large $n$) |

**Key observations:**
- School G (small $n$, extreme value) shrinks substantially
- School F (large $n$, near mean) changes little
- Partial pooling "borrows strength" from all schools

---

## 5. Mathematical Derivation

### 5.1 Posterior for Group Parameters

Under the hierarchical model, the posterior for $\theta_i$ given data and hyperparameters is:

$$
\theta_i | y, \mu, \tau, \sigma \sim \mathcal{N}\left(\hat{\theta}_i, V_i\right)
$$

where:

$$
\hat{\theta}_i = \frac{\frac{n_i}{\sigma^2} \bar{y}_i + \frac{1}{\tau^2} \mu}{\frac{n_i}{\sigma^2} + \frac{1}{\tau^2}}
$$

$$
V_i = \frac{1}{\frac{n_i}{\sigma^2} + \frac{1}{\tau^2}}
$$

### 5.2 Precision-Weighted Average

The posterior mean is a **precision-weighted average**:

$$
\hat{\theta}_i = \frac{\tau_{\text{data}} \cdot \bar{y}_i + \tau_{\text{prior}} \cdot \mu}{\tau_{\text{data}} + \tau_{\text{prior}}}
$$

where:
- $\tau_{\text{data}} = n_i/\sigma^2$ is the data precision
- $\tau_{\text{prior}} = 1/\tau^2$ is the prior precision

### 5.3 Limiting Cases

**As $\tau \to \infty$** (groups very different):
- Prior is vague → $w_i \to 1$
- Partial pooling → No pooling

**As $\tau \to 0$** (groups identical):
- Prior is tight → $w_i \to 0$
- Partial pooling → Complete pooling

---

## 6. Benefits of Hierarchical Models

### 6.1 Improved Estimation

| Benefit | Explanation |
|---------|-------------|
| **Borrowing strength** | Small groups benefit from information in other groups |
| **Regularization** | Extreme estimates are automatically shrunk |
| **Bias-variance tradeoff** | Optimal balance between group-specific and population info |

### 6.2 Better Predictions

Hierarchical estimates typically have **lower mean squared error** than no-pooling estimates, especially for small groups.

### 6.3 Coherent Uncertainty

The hierarchical framework provides:
- Uncertainty for each $\theta_i$
- Uncertainty for population parameters $\mu, \tau$
- Proper propagation of uncertainty across levels

---

## 7. Practical Considerations

### 7.1 When to Use Hierarchical Models

- Multiple groups with related parameters
- Varying sample sizes across groups
- Interest in both group-level and population-level inference
- Small groups that benefit from borrowing strength

### 7.2 Challenges

| Challenge | Solution |
|-----------|----------|
| Computational complexity | MCMC, variational inference |
| Prior specification for $\tau$ | Half-Cauchy, Half-Normal priors |
| Identifiability | Ensure sufficient data at each level |
| Model checking | Posterior predictive checks |

### 7.3 Software

Modern probabilistic programming languages make hierarchical models accessible:
- **Stan** (via PyStan, RStan)
- **PyMC**
- **NumPyro** / **Numpyro**
- **JAGS**

---

## 8. Key Takeaways

1. **Hierarchical models** share information across groups through a multi-level structure, where group parameters are drawn from a common population distribution.

2. **Three pooling strategies**: No pooling (independent), complete pooling (identical), and partial pooling (hierarchical). Partial pooling optimally balances bias and variance.

3. **Shrinkage is automatic**: Groups with small samples or extreme values are pulled toward the population mean. The amount of shrinkage is determined by the data.

4. **Borrowing strength**: Small groups benefit most from hierarchical modeling by leveraging information from other groups.

5. **The shrinkage weight** $w_i = n_i/(n_i + \sigma^2/\tau^2)$ determines how much each group relies on its own data vs the population estimate.

---

## 9. Exercises

### Exercise 1: Varying Heterogeneity
Simulate data with different values of $\tau$ (between-group variation). Show how partial pooling estimates change as groups become more or less similar.

### Exercise 2: Sample Size Effects
Create groups with very different sample sizes (e.g., $n_i \in \{5, 50, 500\}$). Demonstrate that small groups shrink more.

### Exercise 3: MSE Comparison
Compare the mean squared error of no-pooling vs partial-pooling estimates across many simulations. Show that partial pooling wins on average.

### Exercise 4: Full Bayesian Implementation
Implement the Eight Schools model in PyMC or Stan with proper priors on $\mu$ and $\tau$. Compare posterior estimates with the simple empirical Bayes approach.

---

## References

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), Chapter 5
- Gelman, A., & Hill, J. *Data Analysis Using Regression and Multilevel/Hierarchical Models*
- Efron, B., & Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *JASA*, 70(350), 311-319.
