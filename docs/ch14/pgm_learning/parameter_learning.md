# Parameter Learning

## Overview

**Parameter learning** addresses the problem: given a fixed graph structure $G$ and data $\mathcal{D} = \{x^{(1)}, \ldots, x^{(N)}\}$, learn the parameters (CPTs for Bayesian networks, potential function parameters for MRFs) that best explain the observed data.

This is fundamentally simpler than structure learning because the graph is given — we only need to fill in the numerical values of the conditional distributions or potentials.

## Maximum Likelihood Estimation (MLE)

### Complete Data Case

When all variables are observed in every data point, MLE for Bayesian networks decomposes into independent problems for each variable.

**Objective**: Maximize the log-likelihood:

$$\ell(\theta; \mathcal{D}) = \log P(\mathcal{D} \mid G, \theta) = \sum_{n=1}^{N} \sum_{i=1}^{|V|} \log P(X_i^{(n)} \mid \text{Pa}(X_i)^{(n)}; \theta_i)$$

Because the log-likelihood decomposes as a sum over variables, each CPT can be estimated independently.

### MLE for Discrete CPTs

For a discrete variable $X_i$ with parents $\text{Pa}(X_i)$, the MLE parameters are simply the **empirical conditional frequencies**:

$$\hat{P}(X_i = k \mid \text{Pa}(X_i) = j) = \frac{N_{ijk}}{\sum_{k'} N_{ijk'}} = \frac{N_{ijk}}{N_{ij}}$$

where $N_{ijk}$ counts the number of data points where $X_i = k$ and $\text{Pa}(X_i) = j$.

```python
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class MLEParameterLearner:
    """
    Maximum Likelihood parameter estimation for Bayesian Networks
    with complete data.
    """
    
    def __init__(self, data: pd.DataFrame, structure: Dict[str, List[str]]):
        """
        Args:
            data: DataFrame with one column per variable
            structure: Dict mapping each variable to its parent list
        """
        self.data = data
        self.structure = structure
        self.variables = list(structure.keys())
        self.n_samples = len(data)
    
    def estimate_cpt(self, variable: str,
                     laplace_smoothing: float = 0.0) -> torch.Tensor:
        """
        Estimate CPT for a single variable via MLE.
        
        P(Xi=k | Pa(Xi)=j) = (N_ijk + alpha) / (N_ij + alpha * K)
        
        Args:
            variable: Variable name
            laplace_smoothing: Smoothing parameter (0 = pure MLE)
            
        Returns:
            CPT tensor with shape (card_pa1, card_pa2, ..., card_variable)
        """
        parents = self.structure[variable]
        var_card = self.data[variable].nunique()
        
        if not parents:
            # No parents: just count frequencies
            counts = self.data[variable].value_counts().sort_index().values
            counts = counts.astype(float) + laplace_smoothing
            probs = counts / counts.sum()
            return torch.tensor(probs, dtype=torch.float32)
        
        # Group by parent configurations
        parent_cards = [self.data[p].nunique() for p in parents]
        shape = tuple(parent_cards) + (var_card,)
        cpt = torch.zeros(shape)
        
        for idx, group in self.data.groupby(parents):
            if not isinstance(idx, tuple):
                idx = (idx,)
            
            counts = np.zeros(var_card)
            value_counts = group[variable].value_counts()
            for val, count in value_counts.items():
                counts[int(val)] = count
            
            counts += laplace_smoothing
            probs = counts / counts.sum()
            cpt[idx] = torch.tensor(probs, dtype=torch.float32)
        
        return cpt
    
    def estimate_all(self, laplace_smoothing: float = 1.0) -> Dict[str, torch.Tensor]:
        """Estimate CPTs for all variables."""
        cpts = {}
        for var in self.variables:
            cpts[var] = self.estimate_cpt(var, laplace_smoothing)
        return cpts
    
    def log_likelihood(self, cpts: Dict[str, torch.Tensor] = None) -> float:
        """Compute log-likelihood of data given parameters."""
        if cpts is None:
            cpts = self.estimate_all(laplace_smoothing=0.0)
        
        ll = 0.0
        for _, row in self.data.iterrows():
            for var in self.variables:
                parents = self.structure[var]
                if parents:
                    idx = tuple(int(row[p]) for p in parents) + (int(row[var]),)
                else:
                    idx = (int(row[var]),)
                
                p = cpts[var][idx].item()
                if p > 0:
                    ll += np.log(p)
                else:
                    ll += np.log(1e-10)
        
        return ll
```

### Demonstration

```python
def demonstrate_mle():
    """Demonstrate MLE parameter learning."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate data from known distribution
    # Structure: A -> B -> C
    A = np.random.binomial(1, 0.6, n_samples)
    B = np.random.binomial(1, 0.3 + 0.5 * A)
    C = np.random.binomial(1, 0.2 + 0.6 * B)
    
    data = pd.DataFrame({'A': A, 'B': B, 'C': C})
    structure = {'A': [], 'B': ['A'], 'C': ['B']}
    
    learner = MLEParameterLearner(data, structure)
    cpts = learner.estimate_all(laplace_smoothing=1.0)
    
    print("Learned CPTs (with Laplace smoothing):")
    print(f"\nP(A): {cpts['A']}")
    print(f"  True: [0.4, 0.6]")
    
    print(f"\nP(B|A):\n{cpts['B']}")
    print(f"  True: [[0.7, 0.3], [0.2, 0.8]]")
    
    print(f"\nP(C|B):\n{cpts['C']}")
    print(f"  True: [[0.8, 0.2], [0.2, 0.8]] (approx)")
    
    ll = learner.log_likelihood(cpts)
    print(f"\nLog-likelihood: {ll:.2f}")


demonstrate_mle()
```

## Bayesian Parameter Estimation

Instead of point estimates, **Bayesian estimation** maintains a posterior distribution over parameters, providing uncertainty quantification and natural regularization.

### Dirichlet Prior

The conjugate prior for categorical distributions is the **Dirichlet distribution**:

$$P(\theta \mid \alpha) = \text{Dir}(\theta; \alpha_1, \ldots, \alpha_K) = \frac{1}{B(\alpha)} \prod_{k=1}^{K} \theta_k^{\alpha_k - 1}$$

where $\alpha_k > 0$ are concentration parameters. The posterior is also Dirichlet:

$$P(\theta \mid \mathcal{D}, \alpha) = \text{Dir}(\theta; \alpha_1 + N_1, \ldots, \alpha_K + N_K)$$

The posterior mean (Bayesian point estimate) is:

$$\hat{\theta}_k^{\text{Bayes}} = \frac{N_k + \alpha_k}{\sum_{k'} (N_{k'} + \alpha_{k'})}$$

Note that setting $\alpha_k = 1$ for all $k$ recovers Laplace smoothing.

### Equivalent Sample Size

The Dirichlet parameters $\alpha_k$ can be interpreted as "pseudo-counts" from a prior equivalent sample of size $\sum_k \alpha_k$. A uniform prior with $\alpha_k = 1$ corresponds to a prior equivalent sample of $K$ observations uniformly distributed across states.

```python
class BayesianParameterLearner:
    """
    Bayesian parameter estimation with Dirichlet priors.
    """
    
    def __init__(self, data: pd.DataFrame, structure: Dict[str, List[str]],
                 prior_strength: float = 1.0):
        """
        Args:
            prior_strength: Equivalent sample size for symmetric Dirichlet prior
        """
        self.data = data
        self.structure = structure
        self.prior_strength = prior_strength
    
    def posterior_cpt(self, variable: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance for a CPT.
        
        Returns:
            (posterior_mean, posterior_variance) tensors
        """
        parents = self.structure[variable]
        var_card = self.data[variable].nunique()
        alpha = self.prior_strength / var_card  # Symmetric Dirichlet
        
        if not parents:
            counts = np.zeros(var_card)
            for val, cnt in self.data[variable].value_counts().items():
                counts[int(val)] = cnt
            
            posterior_alpha = counts + alpha
            total = posterior_alpha.sum()
            
            mean = torch.tensor(posterior_alpha / total, dtype=torch.float32)
            
            # Dirichlet variance
            var = torch.tensor(
                posterior_alpha * (total - posterior_alpha)
                / (total ** 2 * (total + 1)),
                dtype=torch.float32
            )
            return mean, var
        
        parent_cards = [self.data[p].nunique() for p in parents]
        shape = tuple(parent_cards) + (var_card,)
        mean_cpt = torch.zeros(shape)
        var_cpt = torch.zeros(shape)
        
        for idx, group in self.data.groupby(parents):
            if not isinstance(idx, tuple):
                idx = (idx,)
            
            counts = np.zeros(var_card)
            for val, cnt in group[variable].value_counts().items():
                counts[int(val)] = cnt
            
            posterior_alpha = counts + alpha
            total = posterior_alpha.sum()
            
            mean_cpt[idx] = torch.tensor(posterior_alpha / total)
            var_cpt[idx] = torch.tensor(
                posterior_alpha * (total - posterior_alpha)
                / (total ** 2 * (total + 1))
            )
        
        return mean_cpt, var_cpt
```

## The EM Algorithm for Incomplete Data

When some variables are unobserved (latent or missing), the log-likelihood no longer decomposes and MLE has no closed-form solution. The **Expectation-Maximization (EM)** algorithm iterates between:

**E-step**: Compute expected sufficient statistics using current parameters and inference:

$$\mathbb{E}[N_{ijk} \mid \mathcal{D}, \theta^{(t)}] = \sum_{n=1}^{N} P(X_i = k, \text{Pa}(X_i) = j \mid x^{(n)}_{\text{obs}}, \theta^{(t)})$$

**M-step**: Update parameters using the expected counts as if they were observed:

$$\theta^{(t+1)}_{ijk} = \frac{\mathbb{E}[N_{ijk}]}{\sum_{k'} \mathbb{E}[N_{ijk'}]}$$

### Properties

- Each iteration is guaranteed to increase (or maintain) the log-likelihood
- Converges to a **local maximum** (not necessarily global)
- Multiple random restarts help find better optima
- The E-step requires running inference, making it computationally expensive for large models

```python
class EMParameterLearner:
    """
    EM algorithm for parameter learning with missing data.
    """
    
    def __init__(self, bn, observed_vars: List[str]):
        """
        Args:
            bn: Bayesian Network with structure (CPTs will be learned)
            observed_vars: Variables that are observed in data
        """
        self.bn = bn
        self.observed_vars = observed_vars
        self.latent_vars = [v for v in bn.variables if v not in observed_vars]
    
    def fit(self, data: List[Dict[str, int]],
            max_iterations: int = 50,
            tolerance: float = 1e-4,
            verbose: bool = False) -> List[float]:
        """
        Run EM algorithm.
        
        Args:
            data: List of partial observations (dicts with observed values)
            
        Returns:
            List of log-likelihoods per iteration
        """
        from ch17.pgm_inference.variable_elimination import VariableElimination
        
        log_likelihoods = []
        
        for iteration in range(max_iterations):
            # E-step: compute expected sufficient statistics
            expected_counts = self._e_step(data)
            
            # M-step: update CPTs from expected counts
            self._m_step(expected_counts)
            
            # Compute log-likelihood
            ll = self._compute_log_likelihood(data)
            log_likelihoods.append(ll)
            
            if verbose:
                print(f"  EM iteration {iteration + 1}: LL = {ll:.4f}")
            
            if len(log_likelihoods) > 1:
                if abs(log_likelihoods[-1] - log_likelihoods[-2]) < tolerance:
                    if verbose:
                        print(f"  Converged at iteration {iteration + 1}")
                    break
        
        return log_likelihoods
    
    def _e_step(self, data):
        """Compute expected sufficient statistics."""
        ve = VariableElimination(self.bn)
        
        expected_counts = {}
        for var in self.bn.variables:
            parents = self.bn.parents[var]
            all_vars = parents + [var]
            cards = [self.bn.cardinalities[v] for v in all_vars]
            expected_counts[var] = torch.zeros(cards)
        
        for obs in data:
            for var in self.bn.variables:
                parents = self.bn.parents[var]
                query_vars = [v for v in parents + [var] if v not in obs]
                
                if not query_vars:
                    # All relevant variables observed
                    idx = tuple(obs[v] for v in parents + [var])
                    expected_counts[var][idx] += 1.0
                else:
                    # Need inference for unobserved variables
                    all_vars = parents + [var]
                    result = ve.query(query_vars, obs)
                    
                    for assignment in cartesian_product(
                        *[range(self.bn.cardinalities[v]) for v in query_vars]
                    ):
                        full_assign = dict(obs)
                        full_assign.update(dict(zip(query_vars, assignment)))
                        
                        idx = tuple(full_assign[v] for v in all_vars)
                        prob = result[assignment].item()
                        expected_counts[var][idx] += prob
        
        return expected_counts
    
    def _m_step(self, expected_counts):
        """Update CPTs from expected sufficient statistics."""
        for var in self.bn.variables:
            counts = expected_counts[var]
            # Normalize along the last axis (variable dimension)
            sums = counts.sum(dim=-1, keepdim=True)
            sums = torch.clamp(sums, min=1e-10)
            cpt_values = counts / sums
            self.bn.set_cpt(var, cpt_values)
    
    def _compute_log_likelihood(self, data):
        """Compute marginal log-likelihood of observed data."""
        ve = VariableElimination(self.bn)
        ll = 0.0
        
        for obs in data:
            # P(observed) = sum over hidden of P(all)
            hidden = [v for v in self.bn.variables if v not in obs]
            if not hidden:
                ll += np.log(max(self.bn.joint_probability(obs), 1e-10))
            else:
                # Use VE to compute marginal probability of evidence
                result = ve.query(list(obs.keys()), {})
                idx = tuple(obs[v] for v in obs.keys())
                ll += np.log(max(result[idx].item(), 1e-10))
        
        return ll
```

## Parameter Learning for MRFs

Parameter learning for MRFs is more challenging because the partition function $Z(\theta)$ depends on the parameters, making the gradient of the log-likelihood:

$$\frac{\partial \ell}{\partial \theta_k} = \sum_{n=1}^{N} f_k(x^{(n)}) - N \cdot \mathbb{E}_{P(X; \theta)}[f_k(X)]$$

The first term (data-dependent) is easy to compute. The second term (model expectation) requires inference, which is generally intractable. Common approaches include:

- **Contrastive divergence**: Approximate the model expectation with a few steps of MCMC
- **Pseudo-likelihood**: Replace the joint likelihood with a product of conditional likelihoods
- **Score matching**: Avoid computing $Z$ entirely by matching score functions

## Summary

| Method | Data | Result | Complexity |
|--------|------|--------|------------|
| **MLE** | Complete | Point estimate (frequency counts) | $O(N)$ per CPT |
| **Bayesian (Dirichlet)** | Complete | Posterior distribution over parameters | $O(N)$ per CPT |
| **EM** | Incomplete | Point estimate (local optimum) | $O(N \cdot \text{inference})$ per iteration |
| **MRF MLE** | Complete | Requires inference for gradient | Depends on inference method |

## Key Takeaways

1. **Complete data is easy**: MLE decomposes into independent counting problems per variable.
2. **Priors prevent overfitting**: Bayesian estimation with Dirichlet priors is equivalent to Laplace smoothing and is essential for sparse data.
3. **Missing data requires EM**: The E-step uses inference to compute expected counts; the M-step updates parameters from those counts.
4. **MRFs are harder**: The partition function makes gradient computation require inference, unlike Bayesian networks where MLE is closed-form.
5. **Multiple restarts**: EM converges to local optima; run from multiple initializations.

## Quantitative Finance Application

Parameter learning is essential for calibrating PGM-based risk models to market data. In credit risk, the CPTs $P(\text{Default}_i \mid \text{SectorHealth}, \text{FirmRating})$ must be estimated from historical default data. With limited default observations (especially for high-rated entities), Bayesian estimation with informative priors from expert knowledge or regulatory guidelines prevents extreme probability estimates. The EM algorithm is particularly relevant when some risk factors are latent — for example, an unobserved "systemic stress" variable that drives correlated defaults but is never directly measured.
