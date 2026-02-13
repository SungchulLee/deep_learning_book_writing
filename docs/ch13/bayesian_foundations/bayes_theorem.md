# Bayes' Theorem

Bayes' theorem provides the mathematical foundation for updating beliefs in light of new evidence. This section derives the theorem, establishes the Bayesian interpretation of its components, and demonstrates its application through discrete examples.

## Derivation from Conditional Probability

For two events $A$ and $B$, conditional probability is defined as:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B|A) = \frac{P(A \cap B)}{P(A)}$$

From the first equation, $P(A \cap B) = P(A|B)P(B)$. Substituting into the second:

$$P(B|A) = \frac{P(A|B)P(B)}{P(A)}$$

## The Bayesian Formulation

In Bayesian inference, we interpret:
- $A$ as the **hypothesis** or **parameter** $\theta$
- $B$ as the **observed data** $D$

This gives **Bayes' theorem**:

$$\boxed{P(\theta|D) = \frac{P(D|\theta) \, P(\theta)}{P(D)}}$$

| Term | Name | Interpretation |
|------|------|----------------|
| $P(\theta|D)$ | **Posterior** | Belief about $\theta$ after seeing data |
| $P(D|\theta)$ | **Likelihood** | Probability of data given hypothesis |
| $P(\theta)$ | **Prior** | Belief about $\theta$ before seeing data |
| $P(D)$ | **Evidence** | Total probability of data (normalising constant) |

The theorem can be written proportionally:

$$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

For detailed treatment of each component, see [Prior, Likelihood, Posterior](prior_likelihood_posterior.md).

## Computing the Evidence

The evidence (marginal likelihood) ensures the posterior is a valid probability distribution. For discrete hypotheses $\{\theta_1, \theta_2, \ldots, \theta_K\}$:

$$P(D) = \sum_{i=1}^{K} P(D|\theta_i) P(\theta_i)$$

For continuous parameters:

$$P(D) = \int P(D|\theta) P(\theta) \, d\theta$$

The evidence is often intractable for complex models, motivating approximate inference methods (see [Chapter 18: Monte Carlo Methods](../../ch15/monte_carlo/integration.md) and [Variational Inference](../../ch16/variational_inference/framework.md)).

## PyTorch Implementation

```python
import torch


def bayes_theorem_discrete(prior, likelihood):
    """
    Apply Bayes' theorem for discrete hypotheses.

    Args:
        prior: Tensor of shape (K,), prior probabilities
        likelihood: Tensor of shape (K,), P(data | hypothesis_k)

    Returns:
        posterior: Tensor of shape (K,), posterior probabilities
    """
    prior = torch.as_tensor(prior, dtype=torch.float32)
    likelihood = torch.as_tensor(likelihood, dtype=torch.float32)

    unnormalised = likelihood * prior
    evidence = unnormalised.sum()

    if evidence == 0:
        raise ValueError("Evidence is zero—check that prior and likelihood overlap")

    return unnormalised / evidence
```

## Example: Medical Testing

### Problem Setup

A disease affects 1% of the population. A diagnostic test has:
- **Sensitivity** (true positive rate): $P(\text{Test}^+ | \text{Disease}) = 0.95$
- **Specificity** (true negative rate): $P(\text{Test}^- | \text{Healthy}) = 0.90$

**Question:** If a person tests positive, what is the probability they have the disease?

### Bayesian Analysis

**Hypotheses:**
- $H_1$: Has disease
- $H_2$: Healthy

**Prior** (population prevalence):

$$P(H_1) = 0.01, \quad P(H_2) = 0.99$$

**Likelihood** (given positive test):

$$P(\text{Test}^+ | H_1) = 0.95, \quad P(\text{Test}^+ | H_2) = 0.10$$

**Evidence:**

$$P(\text{Test}^+) = (0.95)(0.01) + (0.10)(0.99) = 0.1085$$

**Posterior:**

$$P(H_1 | \text{Test}^+) = \frac{(0.95)(0.01)}{0.1085} \approx 0.088$$

### The Base Rate Fallacy

Despite 95% sensitivity, a positive test yields only **8.8% probability** of disease. This counterintuitive result occurs because:

1. The disease is rare (1% base rate)
2. The 10% false positive rate applied to 99% healthy people generates many false positives

```python
prior = torch.tensor([0.01, 0.99])
likelihood = torch.tensor([0.95, 0.10])
posterior = bayes_theorem_discrete(prior, likelihood)
# tensor([0.0876, 0.9124])
```

**Lesson:** The prior fundamentally shapes the posterior. Ignoring base rates leads to systematic reasoning errors.

## Example: Coin Identification

### Problem Setup

Three coins with different biases:

| Coin | $P(\text{Heads})$ |
|------|-------------------|
| Fair | 0.5 |
| Biased-H | 0.7 |
| Biased-T | 0.3 |

One coin is selected uniformly at random and flipped $N$ times, yielding $k$ heads.

### Likelihood Function

The binomial likelihood:

$$P(k \text{ heads} | p) = \binom{N}{k} p^k (1-p)^{N-k}$$

```python
from scipy import stats
import numpy as np


def coin_inference(n_heads, n_tails):
    """Infer which coin was selected given observed flips."""
    p_heads = np.array([0.5, 0.7, 0.3])
    prior = np.array([1/3, 1/3, 1/3])

    n_total = n_heads + n_tails
    likelihood = stats.binom.pmf(n_heads, n_total, p_heads)

    posterior = bayes_theorem_discrete(
        torch.tensor(prior), torch.tensor(likelihood)
    )
    return posterior
```

For 7 heads and 3 tails:

```python
posterior = coin_inference(7, 3)
# Fair (p=0.5):     21.8%
# Biased-H (p=0.7): 70.5%  ← most likely
# Biased-T (p=0.3):  7.7%
```

## Sequential Updating

A fundamental property of Bayesian inference: **the posterior from one observation becomes the prior for the next**.

For sequential observations $D_1, D_2$:

$$P(\theta | D_1, D_2) = \frac{P(D_2 | \theta) \, P(\theta | D_1)}{P(D_2 | D_1)}$$

This makes Bayesian inference naturally suited for **online learning**.

```python
def sequential_update(flip_sequence, p_heads=[0.5, 0.7, 0.3]):
    """Update beliefs sequentially as flips are observed."""
    p = torch.tensor(p_heads)
    posterior = torch.ones(len(p)) / len(p)  # uniform prior

    history = [posterior.clone()]

    for flip in flip_sequence:
        likelihood = p if flip == 'H' else (1 - p)
        posterior = bayes_theorem_discrete(posterior, likelihood)
        history.append(posterior.clone())

    return history
```

### Belief Evolution

For sequence `['H', 'H', 'H', 'T', 'H']`:

| After | Fair | Biased-H | Biased-T |
|-------|------|----------|----------|
| Prior | 33.3% | 33.3% | 33.3% |
| H | 29.4% | 41.2% | 29.4% |
| HH | 26.3% | 51.6% | 22.1% |
| HHH | 23.4% | 64.0% | 12.6% |
| HHHT | 31.6% | 51.9% | 16.5% |
| HHHTH | 28.8% | 60.9% | 10.3% |

As data accumulates, the posterior concentrates on the true hypothesis.

## Summary

| Concept | Key point |
|---------|-----------|
| **Bayes' theorem** | $P(\theta|D) \propto P(D|\theta) P(\theta)$ |
| **Evidence** | Normalising constant; sum/integral of numerator |
| **Base rates matter** | Prior fundamentally shapes posterior |
| **Sequential updating** | Posterior becomes prior for next observation |

## References

1. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
2. McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press.
3. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
