# Bisection Method

Binary search finds a target in a discrete sorted array. The **bisection method** extends this idea to continuous functions: given a continuous function $f$ that changes sign on an interval $[a, b]$, repeatedly halve the interval to locate a root. The bisection method is the continuous analogue of binary search and is one of the most reliable root-finding algorithms in numerical analysis.

## Problem Statement

Given a continuous function $f: [a, b] \to \mathbb{R}$ with $f(a) \cdot f(b) < 0$ (i.e., $f$ has opposite signs at the endpoints), find a value $c \in [a, b]$ such that $f(c) = 0$.

The existence of such a root is guaranteed by the **Intermediate Value Theorem**: if $f$ is continuous on $[a, b]$ and $f(a) \cdot f(b) < 0$, then there exists at least one $c \in (a, b)$ with $f(c) = 0$.

## The Algorithm

At each step, the bisection method computes the midpoint $m = (a + b) / 2$ and evaluates $f(m)$:

- If $f(m) = 0$, then $m$ is a root.
- If $f(a) \cdot f(m) < 0$, the root lies in $[a, m]$, so set $b = m$.
- If $f(m) \cdot f(b) < 0$, the root lies in $[m, b]$, so set $a = m$.

The process repeats until the interval width $b - a$ falls below a specified tolerance $\epsilon$.

### Python Implementation

```python
def bisection(f, a, b, tol=1e-10, max_iter=100):
    """
    Find a root of f in [a, b] using the bisection method.

    Parameters
    ----------
    f : callable
        A continuous function with f(a) * f(b) < 0.
    a : float
        Left endpoint of the interval.
    b : float
        Right endpoint of the interval.
    tol : float
        Convergence tolerance on the interval width.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    float
        An approximate root of f.

    Raises
    ------
    ValueError
        If f(a) and f(b) have the same sign.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    for _ in range(max_iter):
        mid = (a + b) / 2.0
        fm = f(mid)

        if fm == 0.0 or (b - a) / 2.0 < tol:
            return mid

        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm

    return (a + b) / 2.0
```

## Convergence Analysis

### Error Bound

After $k$ iterations, the interval has width

$$
b_k - a_k = \frac{b_0 - a_0}{2^k}
$$

where $[a_0, b_0]$ is the initial interval. The midpoint $m_k = (a_k + b_k) / 2$ satisfies

$$
|m_k - c^*| \le \frac{b_0 - a_0}{2^{k+1}}
$$

where $c^*$ is the true root.

### Iterations to Achieve Tolerance

To achieve $|m_k - c^*| < \epsilon$, we need

$$
\frac{b_0 - a_0}{2^{k+1}} < \epsilon
$$

Solving for $k$:

$$
k > \log_2\!\left(\frac{b_0 - a_0}{\epsilon}\right) - 1
$$

Therefore, the number of iterations required is

$$
k = \left\lceil \log_2\!\left(\frac{b_0 - a_0}{\epsilon}\right) \right\rceil
$$

### Convergence Rate

The bisection method converges **linearly** with rate $1/2$. Each iteration adds approximately one bit of accuracy to the result. This is slower than Newton's method (quadratic convergence) but the bisection method is guaranteed to converge, while Newton's method may diverge for poorly chosen starting points.

!!! note "Guaranteed Convergence"
    Unlike Newton's method or the secant method, bisection never fails: it converges for any continuous function satisfying the sign-change condition. This robustness makes it a reliable fallback when faster methods are unstable.

## Worked Example

Find a root of $f(x) = x^3 - x - 2$ on $[1, 2]$.

We have $f(1) = 1 - 1 - 2 = -2 < 0$ and $f(2) = 8 - 2 - 2 = 4 > 0$, so a root exists in $[1, 2]$.

| Iteration | $a$ | $b$ | $m$ | $f(m)$ | Action |
|---|---|---|---|---|---|
| 1 | 1.000 | 2.000 | 1.500 | $-0.125$ | $f(a) \cdot f(m) > 0$, set $a = 1.5$ |
| 2 | 1.500 | 2.000 | 1.750 | $1.609$ | $f(a) \cdot f(m) < 0$, set $b = 1.75$ |
| 3 | 1.500 | 1.750 | 1.625 | $0.666$ | $f(a) \cdot f(m) < 0$, set $b = 1.625$ |
| 4 | 1.500 | 1.625 | 1.5625 | $0.252$ | $f(a) \cdot f(m) < 0$, set $b = 1.5625$ |
| 5 | 1.500 | 1.5625 | 1.5313 | $0.059$ | $f(a) \cdot f(m) < 0$, set $b = 1.5313$ |

After 5 iterations, the root is bracketed in $[1.500, 1.531]$, an interval of width $0.031$, consistent with $1.0 / 2^5 = 0.03125$.

The exact root is $c^* \approx 1.5214$, and the approximation after 5 iterations is $m_5 \approx 1.5156$, with error $\approx 0.006$.

## Connection to Binary Search

The bisection method is structurally identical to binary search:

| Aspect | Binary Search | Bisection |
|---|---|---|
| Domain | Discrete sorted array | Continuous interval |
| Condition | Comparison with target | Sign of function value |
| Splitting | Midpoint index | Midpoint of interval |
| Convergence | Exact in $O(\log n)$ steps | Approximate: error halves each step |
| Guarantee | Correct if target exists | Converges if $f$ is continuous and sign-changing |

Both algorithms exploit monotonicity to halve the search space at each step. The binary search template from the [template page](binary_search_template.md) can be seen as a discrete version of bisection.

## Limitations

- **Only finds one root**: if $f$ has multiple roots in $[a, b]$, bisection finds one but not all.
- **Requires sign change**: if $f$ touches zero without crossing (e.g., $f(x) = x^2$ at $x = 0$), bisection cannot detect the root.
- **Linear convergence**: for high-precision results, bisection requires many iterations. In practice, it is often used to produce a good initial bracket, which is then refined by a faster method like Newton's.

## Summary

The bisection method applies the divide-and-conquer strategy to continuous root finding. By repeatedly halving an interval where a sign change occurs, it converges to a root at a rate of one bit of accuracy per iteration. The method requires $\lceil \log_2((b-a)/\epsilon) \rceil$ iterations to achieve tolerance $\epsilon$ and is guaranteed to converge for any continuous function satisfying the sign-change condition.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 4. MIT Press.
- Burden, R. L., & Faires, J. D. (2011). *Numerical Analysis* (9th ed.), Chapter 2. Cengage Learning.
