# Multilevel Models

Multilevel (mixed-effects) models extend hierarchical Bayesian models by introducing structured random effects that capture group-level variation. These models are essential for analyzing nested data — students within schools, trades within portfolios, or repeated measurements within subjects — where observations share group-level structure.

---

## Model Specification

### Varying Intercepts Model

The simplest multilevel model allows group-specific intercepts while sharing a common slope:

$$
y_{ij} = \alpha_{j} + \beta x_{ij} + \epsilon_{ij}, \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)
$$

where $i$ indexes observations within group $j$, and:

$$
\alpha_j \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^2)
$$

The group intercepts $\alpha_j$ are drawn from a common population distribution, enabling **partial pooling** — each group's estimate borrows strength from other groups.

### Varying Intercepts and Slopes

A richer model allows both intercepts and slopes to vary by group:

$$
y_{ij} = \alpha_j + \beta_j x_{ij} + \epsilon_{ij}
$$

$$
\begin{pmatrix} \alpha_j \\ \beta_j \end{pmatrix} \sim \mathcal{N}\left( \begin{pmatrix} \mu_\alpha \\ \mu_\beta \end{pmatrix}, \boldsymbol{\Sigma} \right)
$$

The covariance matrix $\boldsymbol{\Sigma}$ captures the correlation between group-level intercepts and slopes — for example, groups with higher baseline returns might also show stronger sensitivity to a factor.

---

## The Three Pooling Strategies

### Complete Pooling (Single Model)

Ignore group structure entirely: $y_{ij} = \alpha + \beta x_{ij} + \epsilon_{ij}$

**Problem**: Underestimates group-level variation, biases estimates for groups that differ from the population.

### No Pooling (Separate Models)

Fit independent models per group: $y_{ij} = \alpha_j + \beta_j x_{ij} + \epsilon_{ij}$

**Problem**: Noisy estimates for groups with few observations, no information sharing.

### Partial Pooling (Multilevel)

Group estimates are shrunk toward the population mean, with the degree of shrinkage determined by the relative precision of group-level and population-level information:

$$
\hat{\alpha}_j^{\text{partial}} = \lambda_j \hat{\alpha}_j^{\text{no pool}} + (1 - \lambda_j) \hat{\alpha}^{\text{complete pool}}
$$

where $\lambda_j$ depends on the number of observations in group $j$ and the between-group variance.

**Key property**: Groups with fewer observations experience more shrinkage toward the population mean.

---

## PyTorch Implementation

```python
import torch
import torch.distributions as dist


class MultilevelModel:
    """
    Bayesian multilevel model with varying intercepts.
    
    Implements Gibbs sampling for:
        y_{ij} = alpha_j + beta * x_{ij} + epsilon_{ij}
        alpha_j ~ N(mu_alpha, sigma_alpha^2)
    """
    
    def __init__(self, n_groups: int):
        self.n_groups = n_groups
    
    def fit_gibbs(self, x, y, group_ids, n_samples=2000, warmup=500):
        """
        Gibbs sampler for the varying intercepts model.
        
        Parameters
        ----------
        x : Tensor of shape (N,)
        y : Tensor of shape (N,)
        group_ids : Tensor of shape (N,), integer group labels
        """
        N = len(y)
        
        # Initialize parameters
        alpha = torch.zeros(self.n_groups)
        beta = torch.tensor(0.0)
        mu_alpha = torch.tensor(0.0)
        sigma2 = torch.tensor(1.0)
        sigma2_alpha = torch.tensor(1.0)
        
        samples = {
            'alpha': [], 'beta': [], 'mu_alpha': [],
            'sigma2': [], 'sigma2_alpha': []
        }
        
        for t in range(n_samples + warmup):
            # --- Sample alpha_j (group intercepts) ---
            for j in range(self.n_groups):
                mask = (group_ids == j)
                n_j = mask.sum().float()
                if n_j == 0:
                    alpha[j] = dist.Normal(mu_alpha, sigma2_alpha.sqrt()).sample()
                    continue
                
                resid_j = y[mask] - beta * x[mask]
                
                # Posterior precision and mean
                precision_j = n_j / sigma2 + 1.0 / sigma2_alpha
                mean_j = (resid_j.sum() / sigma2 + mu_alpha / sigma2_alpha) / precision_j
                alpha[j] = dist.Normal(mean_j, (1.0 / precision_j).sqrt()).sample()
            
            # --- Sample beta (common slope) ---
            resid = y - alpha[group_ids]
            precision_beta = (x ** 2).sum() / sigma2 + 0.01  # weak prior
            mean_beta = (x * resid).sum() / sigma2 / precision_beta
            beta = dist.Normal(mean_beta, (1.0 / precision_beta).sqrt()).sample()
            
            # --- Sample mu_alpha (population mean) ---
            precision_mu = self.n_groups / sigma2_alpha + 0.001
            mean_mu = alpha.sum() / sigma2_alpha / precision_mu
            mu_alpha = dist.Normal(mean_mu, (1.0 / precision_mu).sqrt()).sample()
            
            # --- Sample sigma2 (observation noise) ---
            resid_all = y - alpha[group_ids] - beta * x
            ss = (resid_all ** 2).sum()
            sigma2 = dist.InverseGamma(
                torch.tensor(N / 2.0 + 1.0),
                ss / 2.0 + 0.1
            ).sample()
            
            # --- Sample sigma2_alpha (between-group variance) ---
            ss_alpha = ((alpha - mu_alpha) ** 2).sum()
            sigma2_alpha = dist.InverseGamma(
                torch.tensor(self.n_groups / 2.0 + 1.0),
                ss_alpha / 2.0 + 0.1
            ).sample()
            
            if t >= warmup:
                samples['alpha'].append(alpha.clone())
                samples['beta'].append(beta.item())
                samples['mu_alpha'].append(mu_alpha.item())
                samples['sigma2'].append(sigma2.item())
                samples['sigma2_alpha'].append(sigma2_alpha.item())
        
        return {k: torch.tensor(v) if not isinstance(v[0], torch.Tensor) 
                else torch.stack(v) for k, v in samples.items()}
```

---

## Applications in Quantitative Finance

### Cross-Sectional Asset Pricing

Multilevel models naturally handle panel data in asset pricing:

$$
r_{it} = \alpha_i + \beta_i f_t + \epsilon_{it}
$$

where asset-level parameters $(\alpha_i, \beta_i)$ are drawn from a population distribution. This provides:

- Shrinkage of noisy alpha estimates toward zero (addressing multiple testing)
- Improved beta estimation for assets with short histories
- Principled uncertainty quantification for each asset

### Portfolio Risk Decomposition

Varying-effects models decompose portfolio risk into:

- **Within-asset variance** ($\sigma^2$): Idiosyncratic risk
- **Between-asset variance** ($\sigma_\alpha^2$): Systematic dispersion in expected returns
- **Population parameters** ($\mu_\alpha, \mu_\beta$): Market-wide risk factors

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Partial pooling** | Optimal compromise between ignoring and overfitting group structure |
| **Shrinkage** | Groups with less data are pulled more toward the population mean |
| **Varying effects** | Intercepts, slopes, or both can vary by group |
| **Covariance structure** | Multivariate random effects capture correlations between group parameters |

---

## References

- Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press. Chapters 13-14.
- Raudenbush, S. W., & Bryk, A. S. (2002). *Hierarchical Linear Models* (2nd ed.). Sage.
