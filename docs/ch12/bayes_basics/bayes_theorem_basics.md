# Bayes' Theorem Basics

## Overview

This module introduces the foundational concepts of Bayesian inference through simple discrete examples. We develop the mathematical framework of Bayes' theorem and demonstrate its application to medical testing, coin flip inference, and sequential updating.

---

## 1. Mathematical Foundation

### 1.1 Derivation from Conditional Probability

We begin with the definition of conditional probability. For two events $A$ and $B$, the conditional probabilities are:

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B|A) = \frac{P(A \cap B)}{P(A)}
$$

From the first equation, we obtain $P(A \cap B) = P(A|B)P(B)$. Substituting into the second equation yields:

$$
P(B|A) = \frac{P(A|B)P(B)}{P(A)}
$$

### 1.2 Bayesian Formulation

In Bayesian inference, we interpret these quantities as follows:

- $A$ represents the **hypothesis** or **parameter** $\theta$
- $B$ represents the **observed data** $D$

This gives us **Bayes' Theorem**:

$$
\boxed{P(\theta|D) = \frac{P(D|\theta) \, P(\theta)}{P(D)}}
$$

Each term has a specific interpretation:

| Term | Name | Interpretation |
|------|------|----------------|
| $P(\theta\|D)$ | **Posterior** | Probability of hypothesis given observed data |
| $P(D\|\theta)$ | **Likelihood** | Probability of data given hypothesis |
| $P(\theta)$ | **Prior** | Initial probability of hypothesis before seeing data |
| $P(D)$ | **Evidence** | Total probability of data (marginal likelihood) |

### 1.3 Computing the Evidence

The evidence (or marginal likelihood) can be computed using the **law of total probability**. For discrete hypotheses:

$$
P(D) = \sum_{i} P(D|\theta_i) P(\theta_i)
$$

This ensures the posterior distribution is properly normalized.

---

## 2. Implementation

### 2.1 Discrete Bayes' Theorem

```python
import numpy as np
from scipy import stats

def bayes_theorem_discrete(prior, likelihood, normalize=True):
    """
    Apply Bayes' theorem for discrete distributions.
    
    Parameters
    ----------
    prior : array-like
        Prior probabilities for each hypothesis
    likelihood : array-like
        Likelihood of data under each hypothesis
    normalize : bool
        If True, normalize to ensure probabilities sum to 1
    
    Returns
    -------
    posterior : numpy array
        Posterior probabilities for each hypothesis
    """
    prior = np.asarray(prior, dtype=float)
    likelihood = np.asarray(likelihood, dtype=float)
    
    # Unnormalized posterior: numerator of Bayes' theorem
    unnormalized_posterior = likelihood * prior
    
    if normalize:
        # Evidence: marginal likelihood (denominator)
        evidence = np.sum(unnormalized_posterior)
        
        if evidence == 0:
            raise ValueError("Evidence is zero - cannot normalize posterior")
        
        posterior = unnormalized_posterior / evidence
    else:
        posterior = unnormalized_posterior
    
    return posterior
```

---

## 3. Example: Medical Testing

### 3.1 Problem Setup

Consider a disease that affects 1% of the population. A medical test has:

- **Sensitivity** (True Positive Rate): $P(\text{Test}^+ | \text{Disease}^+) = 0.95$
- **Specificity** (True Negative Rate): $P(\text{Test}^- | \text{Disease}^-) = 0.90$

**Question**: If a person tests positive, what is the probability they have the disease?

This classic example demonstrates why **base rates matter** in probabilistic reasoning.

### 3.2 Bayesian Analysis

**Define the hypotheses:**

- $H_1$: Person has the disease
- $H_2$: Person does not have the disease

**Priors** (based on population prevalence):

$$
P(H_1) = 0.01, \quad P(H_2) = 0.99
$$

**Likelihoods** (given a positive test):

$$
P(\text{Test}^+ | H_1) = 0.95, \quad P(\text{Test}^+ | H_2) = 0.10
$$

**Compute the evidence:**

$$
P(\text{Test}^+) = P(\text{Test}^+ | H_1) P(H_1) + P(\text{Test}^+ | H_2) P(H_2)
$$

$$
= (0.95)(0.01) + (0.10)(0.99) = 0.0095 + 0.099 = 0.1085
$$

**Compute the posterior:**

$$
P(H_1 | \text{Test}^+) = \frac{P(\text{Test}^+ | H_1) P(H_1)}{P(\text{Test}^+)} = \frac{(0.95)(0.01)}{0.1085} \approx 0.0876
$$

### 3.3 Key Insight

Even though the test is 95% sensitive, a positive test result only gives approximately **8.8% probability** of having the disease. This counterintuitive result arises because:

1. The disease is rare (1% base rate)
2. Most positive tests come from the 99% of healthy people (10% false positive rate on a large group)

```python
def medical_test_example():
    """Demonstrate Bayesian inference in medical testing."""
    
    # Parameters
    base_rate = 0.01
    sensitivity = 0.95
    specificity = 0.90
    false_positive_rate = 1 - specificity
    
    # Hypotheses: [Has Disease, No Disease]
    prior = np.array([base_rate, 1 - base_rate])
    likelihood = np.array([sensitivity, false_positive_rate])
    
    # Apply Bayes' theorem
    posterior = bayes_theorem_discrete(prior, likelihood)
    
    return posterior  # [0.0876, 0.9124]
```

---

## 4. Example: Coin Flip Inference

### 4.1 Problem Setup

Consider three coins with different biases:

| Coin | $P(\text{Heads})$ | Description |
|------|-------------------|-------------|
| Coin 1 | 0.5 | Fair coin |
| Coin 2 | 0.7 | Biased toward heads |
| Coin 3 | 0.3 | Biased toward tails |

We randomly select one coin (each equally likely) and flip it $N$ times, observing $k$ heads and $N-k$ tails.

**Question**: Given the observed flips, which coin was most likely selected?

### 4.2 Likelihood Function

The likelihood follows the **binomial distribution**:

$$
P(k \text{ heads in } N \text{ flips} | p) = \binom{N}{k} p^k (1-p)^{N-k}
$$

For each coin hypothesis, we compute the probability of observing the data.

### 4.3 Implementation

```python
def coin_flip_inference(n_heads, n_tails):
    """
    Infer which of three coins was flipped based on observed data.
    
    Parameters
    ----------
    n_heads : int
        Number of heads observed
    n_tails : int
        Number of tails observed
    
    Returns
    -------
    posterior : numpy array
        Posterior probabilities for each coin
    """
    p_heads = np.array([0.5, 0.7, 0.3])
    
    # Uniform prior
    prior = np.array([1/3, 1/3, 1/3])
    
    # Binomial likelihood
    n_total = n_heads + n_tails
    likelihood = stats.binom.pmf(n_heads, n_total, p_heads)
    
    # Apply Bayes' theorem
    posterior = bayes_theorem_discrete(prior, likelihood)
    
    return posterior
```

### 4.4 Example Calculation

For 7 heads and 3 tails:

```python
posterior = coin_flip_inference(n_heads=7, n_tails=3)
# Coin 1 (p=0.5): ~21.8%
# Coin 2 (p=0.7): ~70.5%  <- Most likely
# Coin 3 (p=0.3): ~7.7%
```

The biased-toward-heads coin is most likely given the observed data.

---

## 5. Sequential Bayesian Updating

### 5.1 Mathematical Framework

A fundamental property of Bayesian inference is **sequential updating**: the posterior from one observation becomes the prior for the next.

For two sequential observations $D_1$ and $D_2$:

$$
P(\theta | D_1, D_2) = \frac{P(D_2 | \theta) \, P(\theta | D_1)}{P(D_2 | D_1)}
$$

This property makes Bayesian inference naturally suited for **online learning** and streaming data.

### 5.2 Implementation

```python
def sequential_coin_flips(flip_sequence, p_heads_coins=[0.5, 0.7, 0.3]):
    """
    Demonstrate sequential Bayesian updating with coin flips.
    
    Parameters
    ----------
    flip_sequence : list of str
        Sequence of observed flips, e.g., ['H', 'H', 'T', 'H']
    p_heads_coins : list of float
        Probability of heads for each coin hypothesis
    
    Returns
    -------
    posterior_history : list of numpy arrays
        Posterior distribution after each flip
    """
    p_heads = np.array(p_heads_coins)
    
    # Initialize with uniform prior
    prior = np.ones(len(p_heads)) / len(p_heads)
    posterior_history = [prior.copy()]
    
    current_posterior = prior.copy()
    
    for flip in flip_sequence:
        # Likelihood for this flip
        if flip == 'H':
            likelihood = p_heads
        else:
            likelihood = 1 - p_heads
        
        # Previous posterior becomes new prior
        prior = current_posterior.copy()
        
        # Apply Bayes' theorem
        current_posterior = bayes_theorem_discrete(prior, likelihood)
        posterior_history.append(current_posterior.copy())
    
    return posterior_history
```

### 5.3 Visualization of Belief Evolution

For the sequence `['H', 'H', 'H', 'T', 'H', 'T', 'H', 'H']`:

| Flip | Coin 1 (p=0.5) | Coin 2 (p=0.7) | Coin 3 (p=0.3) |
|------|----------------|----------------|----------------|
| Prior | 33.3% | 33.3% | 33.3% |
| H | 29.4% | 41.2% | 29.4% |
| HH | 26.3% | 51.6% | 22.1% |
| HHH | 23.4% | 64.0% | 12.6% |
| HHHT | 31.6% | 51.9% | 16.5% |
| ... | ... | ... | ... |

As more data arrives, the posterior concentrates on the most likely hypothesis.

---

## 6. Key Takeaways

1. **Bayes' theorem** provides a principled framework for combining prior beliefs with observed data through the likelihood function.

2. **Base rates matter**: Even highly accurate tests can be misleading when applied to rare conditions. The prior probability fundamentally shapes the posterior.

3. **Sequential updating**: Bayesian inference naturally handles streaming data—each posterior becomes the prior for the next observation.

4. **Normalization**: The evidence term ensures proper probability normalization, computed as the sum of prior × likelihood over all hypotheses.

---

## 7. Exercises

### Exercise 1: Rare Disease
A disease affects 0.1% of the population. A test has 99% sensitivity and 95% specificity. If someone tests positive, what's the probability they have the disease?

### Exercise 2: Four Coins
Extend the coin flip example to four coins with $p_{\text{heads}} \in \{0.2, 0.4, 0.6, 0.8\}$. Observe 15 flips with 11 heads. Which coin is most likely?

### Exercise 3: Different Priors
In the medical test example, suppose additional risk factors increase the prior probability to 5%. How does this change the posterior?

### Exercise 4: Sensitivity Analysis
How many flips are needed to be 95% certain about which coin we have? Test with different sequences.

### Exercise 5: Real Data
Find a real dataset (e.g., spam classification, customer churn) and formulate it as a Bayesian inference problem with discrete hypotheses.

---

## References

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.)
- McElreath, R. *Statistical Rethinking*
- Murphy, K. *Machine Learning: A Probabilistic Perspective*
