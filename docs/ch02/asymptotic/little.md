# Little-o and Little-omega

Big-O and Big-Omega provide upper and lower bounds that may or may not be tight: $n = O(n)$ and $n = O(n^2)$ are both true.  Sometimes we need to express that one function grows **strictly slower** (or **strictly faster**) than another, ruling out the possibility that they grow at the same rate.  Little-o and little-omega fill this role, providing the asymptotic analogues of strict inequality.

## Little-o -- Strictly Slower Growth

### Definition

!!! info "Definition -- Little-o"

    $f(n) = o(g(n))$ if for **every** constant $c > 0$, there exists $n_0 > 0$ such that

    $$
    0 \leq f(n) < c \cdot g(n) \quad \text{for all } n \geq n_0
    $$

The key difference from Big-O is the quantifier: Big-O requires the inequality to hold for *some* $c > 0$, whereas little-o requires it for *every* $c > 0$.  No matter how small a positive constant you choose, $f$ is eventually dominated by $c \cdot g$.

### Limit Characterization

The limit form is often easier to work with.

!!! info "Theorem -- Limit characterization of little-o"

    $$
    f(n) = o(g(n)) \iff \lim_{n \to \infty} \frac{f(n)}{g(n)} = 0
    $$

    provided $g(n) > 0$ for sufficiently large $n$.

This says that $f$ becomes negligible compared to $g$ as $n$ grows.

### Examples

**Example 1:** $n = o(n^2)$ because $\lim_{n \to \infty} n / n^2 = \lim_{n \to \infty} 1/n = 0$.

**Example 2:** $5n^2 + 3n = o(n^3)$ because $\lim_{n \to \infty} (5n^2 + 3n)/n^3 = \lim_{n \to \infty} (5/n + 3/n^2) = 0$.

**Example 3:** $n^2 \neq o(n^2)$ because $\lim_{n \to \infty} n^2/n^2 = 1 \neq 0$.  This illustrates the "strict" nature of little-o: a function cannot be little-o of itself.

## Little-omega -- Strictly Faster Growth

### Definition

!!! info "Definition -- Little-omega"

    $f(n) = \omega(g(n))$ if for **every** constant $c > 0$, there exists $n_0 > 0$ such that

    $$
    0 \leq c \cdot g(n) < f(n) \quad \text{for all } n \geq n_0
    $$

Little-omega is the mirror image of little-o, just as Big-Omega mirrors Big-O.

### Limit Characterization

!!! info "Theorem -- Limit characterization of little-omega"

    $$
    f(n) = \omega(g(n)) \iff \lim_{n \to \infty} \frac{f(n)}{g(n)} = \infty
    $$

    provided $g(n) > 0$ for sufficiently large $n$.

### Examples

**Example 1:** $n^2 = \omega(n)$ because $\lim_{n \to \infty} n^2 / n = \lim_{n \to \infty} n = \infty$.

**Example 2:** $2^n = \omega(n^k)$ for any fixed $k$, because exponentials grow strictly faster than any polynomial.

**Example 3:** $n \neq \omega(n)$ because the limit is $1$, not $\infty$.

## Duality

Little-o and little-omega are dual to each other, just like Big-O and Big-Omega.

$$
f(n) = o(g(n)) \iff g(n) = \omega(f(n))
$$

This follows directly from the limit characterizations: if $f/g \to 0$, then $g/f \to \infty$.

## Relationship to Big-O and Big-Omega

Little-o is a **stronger** statement than Big-O, and little-omega is stronger than Big-Omega.

| Statement | Implies | Does not imply |
|---|---|---|
| $f = o(g)$ | $f = O(g)$ | $f = \Theta(g)$ |
| $f = \omega(g)$ | $f = \Omega(g)$ | $f = \Theta(g)$ |
| $f = \Theta(g)$ | $f = O(g)$ and $f = \Omega(g)$ | $f = o(g)$ or $f = \omega(g)$ |

In other words, $o(g) \subset O(g)$ and $\omega(g) \subset \Omega(g)$, with strict containment.  If $f = \Theta(g)$, then $f$ is neither $o(g)$ nor $\omega(g)$, because the functions grow at the same asymptotic rate.

## When to Use Little-o and Little-omega

These notations appear most often in three contexts:

1. **Error terms.** When approximating a function, little-o describes the remainder: for example, $\ln(1+x) = x + o(x)$ as $x \to 0$.
2. **Strict separations.** To state that one algorithm is strictly faster than another: "Algorithm A runs in $o(n^2)$" means it is *better than quadratic*, not merely *at most quadratic*.
3. **Lower-bound arguments.** To prove that a problem *requires* strictly more than a certain number of operations: "every comparison-based sorting algorithm uses $\omega(n)$ comparisons."

For the complete set of all five asymptotic definitions and their interrelationships, see [Formal Definitions](formal.md).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 3. MIT Press.
