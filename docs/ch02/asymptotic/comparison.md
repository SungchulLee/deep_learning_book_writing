# Growth Rate Comparison

After cataloguing the standard growth-rate classes (see [Common Growth Rates](growth_rates.md)), the next step is to develop techniques for *comparing* two functions and deciding which grows faster.  This page presents the standard hierarchy, the limit-based comparison method, and worked examples that illustrate how to rank functions asymptotically.

## The Standard Hierarchy

The following chain orders common growth rates from slowest to fastest.  Each function on the left is $o$ (little-o) of the function on its right, meaning it grows strictly slower.

$$
1 \;\ll\; \log \log n \;\ll\; \log n \;\ll\; \sqrt{n} \;\ll\; n \;\ll\; n \log n \;\ll\; n^2 \;\ll\; n^3 \;\ll\; 2^n \;\ll\; n! \;\ll\; n^n
$$

Here $f \ll g$ is shorthand for $f(n) = o(g(n))$, that is, $\lim_{n \to \infty} f(n)/g(n) = 0$.

!!! tip "Rule of thumb"

    Logarithmic beats polynomial, polynomial beats exponential, and exponential beats factorial.  Within the polynomial class, the exponent determines the ranking.

## Limit-Based Comparison

The most powerful technique for comparing two functions is to evaluate the limit of their ratio.

!!! info "Theorem -- Limit comparison"

    Let $f(n) > 0$ and $g(n) > 0$ for sufficiently large $n$.  Define

    $$
    L = \lim_{n \to \infty} \frac{f(n)}{g(n)}
    $$

    Then:

    - If $L = 0$, then $f(n) = o(g(n))$, so $f$ grows strictly slower than $g$.
    - If $0 < L < \infty$, then $f(n) = \Theta(g(n))$, so $f$ and $g$ grow at the same rate.
    - If $L = \infty$, then $f(n) = \omega(g(n))$, so $f$ grows strictly faster than $g$.

When the limit does not exist (e.g., the ratio oscillates), this technique does not apply directly.  In such cases, fall back to the definition with explicit constants.

## Useful Limit Tools

Two standard results handle the most common comparisons.

### L'Hopital's Rule

When both $f(n)$ and $g(n)$ tend to infinity, L'Hopital's rule (applied to the continuous extension) gives:

$$
\lim_{n \to \infty} \frac{f(n)}{g(n)} = \lim_{n \to \infty} \frac{f'(n)}{g'(n)}
$$

provided the right-hand limit exists.  This is especially useful for comparing logarithmic and polynomial functions.

### Stirling's Approximation

For factorials, Stirling's formula provides:

$$
n! \approx \sqrt{2\pi n} \left(\frac{n}{e}\right)^n
$$

This shows that $n!$ grows faster than any exponential $c^n$ but slower than $n^n$.

## Examples

### Example 1 -- Logarithm vs Polynomial

Compare $f(n) = \log n$ and $g(n) = n^{0.1}$.

$$
\lim_{n \to \infty} \frac{\log n}{n^{0.1}}
$$

Both numerator and denominator tend to infinity, so apply L'Hopital's rule (using natural log):

$$
\lim_{n \to \infty} \frac{1/n}{0.1 \, n^{-0.9}} = \lim_{n \to \infty} \frac{1}{0.1 \, n^{0.1}} = 0
$$

Since $L = 0$, we conclude $\log n = o(n^{0.1})$.  Logarithms grow slower than *any* positive power of $n$.

### Example 2 -- Polynomial vs Exponential

Compare $f(n) = n^{10}$ and $g(n) = 2^n$.

$$
\lim_{n \to \infty} \frac{n^{10}}{2^n} = 0
$$

This follows by applying L'Hopital's rule 10 times (each application reduces the polynomial degree by one while the denominator remains exponential).  Therefore $n^{10} = o(2^n)$.

### Example 3 -- Same Growth Rate

Compare $f(n) = 5n^2 + 3n$ and $g(n) = n^2$.

$$
\lim_{n \to \infty} \frac{5n^2 + 3n}{n^2} = \lim_{n \to \infty} \left(5 + \frac{3}{n}\right) = 5
$$

Since $0 < 5 < \infty$, we conclude $5n^2 + 3n = \Theta(n^2)$.

## Analogy with Real-Number Comparisons

Asymptotic notation has a natural analogy with comparing real numbers.

| Asymptotic | Analogy | Meaning |
|---|---|---|
| $f = O(g)$ | $a \leq b$ | $f$ grows no faster than $g$ |
| $f = \Omega(g)$ | $a \geq b$ | $f$ grows no slower than $g$ |
| $f = \Theta(g)$ | $a = b$ | $f$ and $g$ grow at the same rate |
| $f = o(g)$ | $a < b$ | $f$ grows strictly slower than $g$ |
| $f = \omega(g)$ | $a > b$ | $f$ grows strictly faster than $g$ |

!!! warning "Limits of the analogy"

    Unlike real numbers, not all functions are asymptotically comparable.  For example, $f(n) = n^{1 + \sin n}$ oscillates between $1$ and $n^2$, so it is neither $O(n)$ nor $\Omega(n^2)$.

## Practical Guidelines

When comparing growth rates in algorithm analysis:

1. **Identify the dominant term** in each expression by dropping lower-order terms and constant factors.
2. **Use the hierarchy** to order the dominant terms.
3. **Apply the limit test** when the hierarchy is not immediately clear (e.g., $n^{1.5}$ vs $n \log^3 n$).
4. **Check for edge cases** such as oscillating functions where the limit may not exist.

For a catalogue of individual growth-rate classes and their properties, see [Common Growth Rates](growth_rates.md).  For the formal definitions of all five asymptotic notations used in comparisons, see [Formal Definitions](formal.md).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 3. MIT Press.
