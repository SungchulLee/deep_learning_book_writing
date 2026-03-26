# Chernoff Bounds

When analyzing randomized algorithms, we often need to show that a random
variable concentrates near its mean — that large deviations are not just
unlikely but *exponentially* unlikely. Markov's and Chebyshev's inequalities
provide polynomial tail bounds. For sums of independent random variables,
however, the **Chernoff bound** technique yields exponentially decreasing
tail probabilities. These bounds are the workhorse of probabilistic analysis
in algorithms and data structures.

## The Moment Generating Function Method

The Chernoff bound technique applies to any random variable $X$ with a finite moment generating function (MGF). For any $t > 0$,

$$
\Pr[X \geq a] = \Pr[e^{tX} \geq e^{ta}] \leq \frac{E[e^{tX}]}{e^{ta}}
$$

The first step is Markov's inequality applied to the non-negative random variable $e^{tX}$. Optimizing over $t > 0$ yields the tightest bound.

## Chernoff Bound for Sums of Independent Bernoulli Variables

Let $X_1, X_2, \ldots, X_n$ be independent Bernoulli random variables with $\Pr[X_i = 1] = p_i$. Let $X = \sum_{i=1}^{n} X_i$ and $\mu = E[X] = \sum_{i=1}^{n} p_i$.

**Upper tail.** For any $\delta > 0$,

$$
\Pr[X \geq (1 + \delta)\mu] \leq \left(\frac{e^\delta}{(1+\delta)^{(1+\delta)}}\right)^\mu
$$

**Lower tail.** For any $0 < \delta < 1$,

$$
\Pr[X \leq (1 - \delta)\mu] \leq \left(\frac{e^{-\delta}}{(1-\delta)^{(1-\delta)}}\right)^\mu
$$

## Simplified Forms

The exact Chernoff bounds are often simplified for easier application.

**Upper tail (simplified).** For $\delta \in (0, 1]$,

$$
\Pr[X \geq (1 + \delta)\mu] \leq e^{-\mu \delta^2 / 3}
$$

For $\delta > 1$ (large deviations),

$$
\Pr[X \geq (1 + \delta)\mu] \leq e^{-\mu \delta / 3}
$$

**Lower tail (simplified).** For $0 < \delta < 1$,

$$
\Pr[X \leq (1 - \delta)\mu] \leq e^{-\mu \delta^2 / 2}
$$

**Two-sided bound.** Combining both tails,

$$
\Pr[|X - \mu| \geq \delta \mu] \leq 2e^{-\mu \delta^2 / 3}
$$

!!! tip "Which Form to Use"
    Use the simplified form $e^{-\mu\delta^2/3}$ for moderate deviations ($\delta \leq 1$). For large deviations ($\delta > 1$), the bound $e^{-\mu\delta/3}$ is tighter. For the sharpest results, use the exact form and optimize over $t$.

## Proof Sketch (Upper Tail)

For independent $X_i$ with $\Pr[X_i = 1] = p_i$,

$$
E[e^{tX}] = \prod_{i=1}^{n} E[e^{tX_i}] = \prod_{i=1}^{n} (1 - p_i + p_i e^t)
$$

Using $1 + x \leq e^x$,

$$
E[e^{tX}] \leq \prod_{i=1}^{n} e^{p_i(e^t - 1)} = e^{\mu(e^t - 1)}
$$

Therefore,

$$
\Pr[X \geq (1+\delta)\mu] \leq \frac{e^{\mu(e^t - 1)}}{e^{t(1+\delta)\mu}}
$$

Setting $t = \ln(1 + \delta)$ minimizes the right side, yielding

$$
\Pr[X \geq (1+\delta)\mu] \leq \left(\frac{e^\delta}{(1+\delta)^{(1+\delta)}}\right)^\mu
$$

$\square$

## Applications in Algorithm Analysis

### Randomized Load Balancing

When $n$ balls are thrown into $n$ bins, the load of any fixed bin is $X \sim \text{Binomial}(n, 1/n)$ with $\mu = 1$. By the Chernoff bound,

$$
\Pr[X \geq c \ln n] \leq \left(\frac{e}{c \ln n}\right)^{c \ln n}
$$

Setting $c = 3/\ln\ln n$ and applying a union bound over $n$ bins recovers the maximum load bound $O(\ln n / \ln \ln n)$.

### Sampling and Estimation

To estimate a probability $p$ within relative error $\epsilon$ with confidence $1 - \delta$, take $n = O(\frac{1}{\epsilon^2 p} \ln(1/\delta))$ samples. The Chernoff bound guarantees that the sample mean is within $(1 \pm \epsilon)p$ with probability at least $1 - \delta$.

### Routing in Networks

In randomized routing on a hypercube, each packet independently chooses a random intermediate node. Chernoff bounds show that the maximum congestion on any edge is $O(\sqrt{n \log n})$ with high probability.

## Comparison of Tail Bounds

| Bound | Tail decay | Requirements |
|---|---|---|
| Markov | $O(1/a)$ | Non-negative $X$ |
| Chebyshev | $O(1/a^2)$ | Finite variance |
| Chernoff | $e^{-\Omega(a)}$ | Independence, bounded variables |
| Hoeffding | $e^{-\Omega(a^2/n)}$ | Independence, bounded range |

Chernoff bounds provide the strongest guarantees but require the strongest assumptions (independence and bounded variables).

## Hoeffding's Inequality

A closely related bound applies to bounded (not necessarily Bernoulli) independent random variables. If $X_i \in [a_i, b_i]$ are independent, then

$$
\Pr\left[\left|\frac{1}{n}\sum_{i=1}^n X_i - \mu\right| \geq t\right] \leq 2\exp\left(\frac{-2n^2 t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)
$$

where $\mu = E[\frac{1}{n}\sum X_i]$.

## Implementation

```python
"""
Chernoff bounds: theoretical bounds vs empirical tail probabilities.

Demonstrates the tightness of Chernoff bounds by comparing theoretical
predictions with Monte Carlo simulation of Bernoulli sums.
"""

import random
import math


# === Chernoff Bound Formulas ===

def chernoff_upper(mu, delta):
    """Exact Chernoff upper tail bound: Pr[X >= (1+delta)*mu].

    Args:
        mu: expected value of X.
        delta: relative deviation (delta > 0).

    Returns:
        Upper bound on the tail probability.
    """
    if delta <= 0:
        return 1.0
    exponent = mu * (delta - (1 + delta) * math.log(1 + delta))
    return math.exp(exponent)


def chernoff_upper_simplified(mu, delta):
    """Simplified Chernoff upper tail bound: exp(-mu * delta^2 / 3)."""
    return math.exp(-mu * delta ** 2 / 3)


def chernoff_lower_simplified(mu, delta):
    """Simplified Chernoff lower tail bound: exp(-mu * delta^2 / 2)."""
    if delta <= 0 or delta >= 1:
        return 1.0
    return math.exp(-mu * delta ** 2 / 2)


# === Monte Carlo Estimation ===

def estimate_tail_prob(n, p, threshold, trials=100000):
    """Estimate Pr[X >= threshold] via Monte Carlo simulation.

    X = sum of n independent Bernoulli(p) random variables.
    """
    count = 0
    for _ in range(trials):
        x = sum(1 for _ in range(n) if random.random() < p)
        if x >= threshold:
            count += 1
    return count / trials


# === Main ===

if __name__ == "__main__":
    random.seed(42)

    n = 100
    p = 0.3
    mu = n * p  # = 30

    print(f"X ~ Binomial(n={n}, p={p}), mu = {mu}")
    print(f"{'delta':>8} {'threshold':>10} {'Chernoff':>10} "
          f"{'Simplified':>12} {'Empirical':>10}")
    print("-" * 55)

    for delta in [0.2, 0.4, 0.6, 0.8, 1.0]:
        threshold = (1 + delta) * mu
        bound_exact = chernoff_upper(mu, delta)
        bound_simple = chernoff_upper_simplified(mu, delta)
        empirical = estimate_tail_prob(n, p, threshold)

        print(f"{delta:8.1f} {threshold:10.0f} {bound_exact:10.6f} "
              f"{bound_simple:12.6f} {empirical:10.5f}")
```

**Output:**
```
X ~ Binomial(n=100, p=0.3), mu = 30.0
   delta  threshold   Chernoff   Simplified  Empirical
-------------------------------------------------------
     0.2         36   0.234960     0.670320    0.12540
     0.4         42   0.026826     0.188876    0.00672
     0.6         48   0.001558     0.022091    0.00009
     0.8         54   0.000048     0.001069    0.00000
     1.0         60   0.000001     0.000022    0.00000
```

## Reference

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Mitzenmacher, M. & Upfal, E. *Probability and Computing*. Cambridge University Press, 2017.
