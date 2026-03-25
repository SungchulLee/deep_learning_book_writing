# Chernoff Bounds

When analyzing randomized algorithms, we often need to show that a random variable concentrates near its mean — that large deviations are not just unlikely but exponentially unlikely. Markov's and Chebyshev's inequalities provide polynomial tail bounds, but for sums of independent random variables, the **Chernoff bound** technique yields exponentially decreasing tail probabilities. These bounds are the workhorse of probabilistic analysis in algorithms and data structures.

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

## Reference

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Mitzenmacher, M. & Upfal, E. *Probability and Computing*. Cambridge University Press, 2017.
