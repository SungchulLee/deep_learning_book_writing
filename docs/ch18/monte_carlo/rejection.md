# Rejection Sampling

Rejection sampling is a fundamental Monte Carlo method for generating samples from a target distribution when direct sampling is difficult. It forms the conceptual foundation for more advanced methods including MCMC and approximate Bayesian computation.

## The Problem

We want to sample from a target distribution $p(x)$ but can only:
1. Evaluate $p(x)$ up to a normalising constant: $p(x) = \tilde{p}(x) / Z$ where $Z$ is unknown
2. Sample from a simpler proposal distribution $q(x)$

## Core Intuition

### The Dartboard Analogy

Imagine the target density $p(x)$ as a 2D region. To sample uniformly from this region:

1. Draw a bounding box (the proposal)
2. Throw darts uniformly in the box
3. Keep darts that land inside the target region; reject those outside

The kept darts are uniform samples from the target region. Rejection sampling generalises this to arbitrary distributions.

### Geometric View

For 1D sampling from $p(x)$:

```
    M·q(x)
    ┌─────────────────────┐
    │    ╱╲               │
    │   ╱  ╲    ← Accept  │
    │  ╱    ╲   (under p) │
    │ ╱ p(x) ╲            │
    │╱        ╲           │
    └─────────────────────┘
         x
```

We sample $(x, u)$ uniformly under the envelope $M \cdot q(x)$, then accept if $u \leq p(x)$.

## Algorithm

**Input:** Target $p(x)$ (up to normalisation), proposal $q(x)$, bound $M$ such that $p(x) \leq M \cdot q(x)$ for all $x$

**Output:** Sample from $p(x)$

```
1. Sample x ~ q(x)
2. Sample u ~ Uniform(0, 1)
3. If u ≤ p(x) / (M · q(x)):
      Accept: return x
   Else:
      Reject: go to step 1
```

### Why It Works

The joint density of accepted $(x, u)$ pairs is uniform under the curve $p(x)$. Marginalising over $u$ gives exactly $p(x)$.

**Proof sketch:** The acceptance region is $\{(x, u) : 0 \leq u \leq p(x)/(M \cdot q(x))\}$. The probability of accepting $x$ is proportional to the height $p(x)/(M \cdot q(x))$, and we proposed $x$ with density $q(x)$. The product $q(x) \cdot p(x)/(M \cdot q(x)) \propto p(x)$.

## Acceptance Probability

The overall acceptance probability is:

$$P(\text{accept}) = \int q(x) \cdot \frac{p(x)}{M \cdot q(x)} dx = \frac{1}{M} \int p(x) dx = \frac{1}{M}$$

(assuming $p(x)$ is normalised; otherwise $P(\text{accept}) = Z/M$)

**Key insight:** Efficiency is determined by how tight the envelope is. Smaller $M$ means higher acceptance rate.

### Expected Number of Iterations

The number of proposals until acceptance follows a geometric distribution:

$$\mathbb{E}[\text{iterations}] = M$$

If $M = 100$, we reject 99% of samples on average—very inefficient.

## The Envelope Condition

The bound $M$ must satisfy:

$$p(x) \leq M \cdot q(x) \quad \text{for all } x$$

Equivalently:

$$M \geq \sup_x \frac{p(x)}{q(x)}$$

### Finding M

**Analytically:** If $p(x)/q(x)$ is tractable, find its maximum via calculus.

**Numerically:** Grid search or optimisation over the support.

**Adaptively:** Start with an estimate, increase if violations are detected (but this biases the sample).

## PyTorch Implementation

### Basic Rejection Sampler

```python
import torch
import torch.distributions as dist


def rejection_sample(target_log_prob, proposal, M, n_samples):
    """
    Rejection sampling from target using proposal.

    Args:
        target_log_prob: Callable, returns log p(x) (unnormalised ok)
        proposal: torch.distributions object with sample() and log_prob()
        M: float, envelope constant (p(x) ≤ M * q(x))
        n_samples: int, number of samples to generate

    Returns:
        samples: Tensor of shape (n_samples, dim)
        acceptance_rate: float
    """
    samples = []
    n_proposed = 0

    while len(samples) < n_samples:
        # Propose
        x = proposal.sample()
        n_proposed += 1

        # Acceptance ratio
        log_p = target_log_prob(x)
        log_q = proposal.log_prob(x)
        log_ratio = log_p - log_q - torch.log(torch.tensor(M))

        # Accept/reject
        u = torch.rand(1)
        if torch.log(u) < log_ratio:
            samples.append(x)

    samples = torch.stack(samples)
    acceptance_rate = n_samples / n_proposed

    return samples, acceptance_rate
```

### Example: Sampling from a Mixture

```python
def mixture_log_prob(x):
    """Mixture of two Gaussians."""
    comp1 = dist.Normal(-2, 0.5)
    comp2 = dist.Normal(2, 1.0)
    # log(0.3 * p1 + 0.7 * p2)
    return torch.logsumexp(
        torch.stack([
            torch.log(torch.tensor(0.3)) + comp1.log_prob(x),
            torch.log(torch.tensor(0.7)) + comp2.log_prob(x)
        ]), dim=0
    )


# Proposal: wide Gaussian covering both modes
proposal = dist.Normal(0, 3)

# Find M numerically
x_grid = torch.linspace(-6, 6, 1000)
log_ratios = mixture_log_prob(x_grid) - proposal.log_prob(x_grid)
M = torch.exp(log_ratios.max()).item() * 1.1  # 10% safety margin

# Sample
samples, acc_rate = rejection_sample(mixture_log_prob, proposal, M, n_samples=1000)
print(f"Acceptance rate: {acc_rate:.2%}")
```

### Vectorised Implementation

```python
def rejection_sample_vectorised(target_log_prob, proposal, M, n_samples, 
                                  batch_size=10000, max_iterations=100):
    """
    Vectorised rejection sampling for efficiency.
    """
    samples = []
    total_proposed = 0

    for _ in range(max_iterations):
        if len(samples) >= n_samples:
            break

        # Propose batch
        x = proposal.sample((batch_size,))
        total_proposed += batch_size

        # Compute acceptance probabilities
        log_p = target_log_prob(x)
        log_q = proposal.log_prob(x)
        log_accept_prob = log_p - log_q - torch.log(torch.tensor(M))

        # Accept/reject
        u = torch.rand(batch_size)
        accepted = torch.log(u) < log_accept_prob

        samples.append(x[accepted])

    samples = torch.cat(samples)[:n_samples]
    acceptance_rate = n_samples / total_proposed

    return samples, acceptance_rate
```

## Choosing the Proposal

Good proposals have:

1. **Coverage:** $q(x) > 0$ wherever $p(x) > 0$
2. **Similar shape:** $q(x) \approx p(x)$ up to a constant
3. **Heavier tails:** $q(x)$ should not decay faster than $p(x)$

### Common Choices

| Target | Good proposal |
|--------|---------------|
| Unimodal, bounded | Uniform over support |
| Unimodal, unbounded | Gaussian or t-distribution |
| Log-concave | Gaussian at mode |
| Heavy-tailed | Student-t with low df |
| Multimodal | Mixture matching modes |

### The Tail Problem

If $p(x)$ has heavier tails than $q(x)$:

$$\lim_{|x| \to \infty} \frac{p(x)}{q(x)} = \infty$$

No finite $M$ exists. **Solution:** Use a heavier-tailed proposal (e.g., Student-t instead of Gaussian).

## Efficiency Analysis

### Acceptance Rate vs Dimension

In $d$ dimensions, if $p$ and $q$ are both Gaussians with slight mismatch:

$$P(\text{accept}) \approx e^{-O(d)}$$

Rejection sampling suffers from the **curse of dimensionality**—acceptance rate decays exponentially with dimension.

| Dimension | Typical acceptance rate |
|-----------|------------------------|
| 1–2 | 50–90% |
| 3–5 | 10–50% |
| 10+ | < 1% |

**Rule of thumb:** Rejection sampling is practical only for $d \lesssim 5$.

### Optimal Proposal

For a given target $p(x)$, the optimal proposal minimising $M$ is:

$$q^*(x) \propto p(x)$$

But if we could sample from this, we wouldn't need rejection sampling. The art is finding $q$ that's both easy to sample and close to $p$.

## Connection to Other Methods

### Importance Sampling

Both use proposals, but differently:

| Aspect | Rejection sampling | Importance sampling |
|--------|-------------------|---------------------|
| Output | Exact samples | Weighted samples |
| Requirement | Envelope $M$ | None |
| Efficiency | Acceptance rate | Effective sample size |
| Bias | Unbiased | Unbiased (with weights) |

Importance sampling reweights all proposals; rejection sampling discards some but keeps others unweighted.

### MCMC

Rejection sampling generates **independent** samples but scales poorly with dimension. MCMC generates **correlated** samples but scales better:

| Method | Independence | Dimension scaling | Burn-in |
|--------|--------------|-------------------|---------|
| Rejection | Independent | $O(e^d)$ | None |
| MCMC | Correlated | $O(d)$ to $O(d^2)$ | Required |

### Approximate Bayesian Computation

ABC rejection sampling replaces the exact acceptance condition with:

$$\text{Accept if } \rho(S(x_{\text{sim}}), S(x_{\text{obs}})) < \varepsilon$$

where $S$ is a summary statistic and $\varepsilon$ is a tolerance. See [ABC Rejection Sampling](../abc/rejection_sampling.md).

## Practical Guidelines

1. **Check envelope:** Verify $p(x) \leq M \cdot q(x)$ on a grid before sampling
2. **Monitor acceptance rate:** If < 1%, consider alternative methods
3. **Use log-space:** Compute $\log p - \log q - \log M$ to avoid overflow
4. **Vectorise:** Generate proposals in batches for efficiency
5. **Dimension limit:** Use rejection sampling only for $d \lesssim 5$

## Summary

| Aspect | Description |
|--------|-------------|
| **Idea** | Propose from $q$, accept with probability $p/(M \cdot q)$ |
| **Requirement** | Envelope $M$ such that $p \leq M \cdot q$ everywhere |
| **Acceptance rate** | $1/M$ (higher is better) |
| **Dimension scaling** | Exponentially bad—practical only for low dimensions |
| **Output** | Exact, independent samples |

## References

1. von Neumann, J. (1951). "Various Techniques Used in Connection with Random Digits." *NBS Applied Mathematics Series*.
2. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer. Chapter 2.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 11.1.
