# Big-Theta Notation

Big-O gives an upper bound and Big-Omega gives a lower bound, but neither one alone pins down the exact growth rate of a function.  When both bounds match -- when $f$ grows at the same rate as $g$ up to constant factors -- we combine them into a single **tight bound** using Big-Theta notation ($\Theta$).  This is the notation we use most often in algorithm analysis because it tells the complete story.

## Formal Definition

Let $f$ and $g$ be functions from the positive integers to the non-negative reals.  We say $f(n)$ is $\Theta(g(n))$ if $f$ is sandwiched between two constant multiples of $g$ for all sufficiently large $n$.

!!! info "Definition -- Big-Theta"

    $f(n) = \Theta(g(n))$ if there exist constants $c_1 > 0$, $c_2 > 0$, and $n_0 > 0$ such that

    $$
    c_1 \cdot g(n) \leq f(n) \leq c_2 \cdot g(n) \quad \text{for all } n \geq n_0
    $$

Three constants are involved:

- **$c_1$ (lower multiplier)** establishes the floor: $f$ grows at least as fast as $c_1 \cdot g$.
- **$c_2$ (upper multiplier)** establishes the ceiling: $f$ grows at most as fast as $c_2 \cdot g$.
- **$n_0$ (threshold)** lets us ignore finitely many small values.

## Equivalence to Big-O and Big-Omega

The sandwich definition is equivalent to requiring both an upper and a lower bound simultaneously.

!!! info "Theorem -- Theta as O plus Omega"

    $$
    f(n) = \Theta(g(n)) \iff f(n) = O(g(n)) \text{ and } f(n) = \Omega(g(n))
    $$

The forward direction is immediate: the upper half of the sandwich gives $O(g(n))$ and the lower half gives $\Omega(g(n))$.  For the converse, take the larger of the two thresholds and the corresponding constants.

This equivalence is useful in practice.  Often it is easier to prove the $O$ and $\Omega$ bounds separately and then combine them into a $\Theta$ statement.

## Intuition

Imagine the function $f(n)$ as a curve on a graph.  Saying $f(n) = \Theta(g(n))$ means that, for large $n$, the curve $f$ is trapped inside a **constant-width band** around $g$.  It cannot escape above $c_2 \cdot g(n)$ or drop below $c_1 \cdot g(n)$.  The band may be wide (if $c_2 / c_1$ is large) or narrow, but it exists.

In algorithm analysis, $\Theta(n^2)$ means "the running time is *exactly* quadratic, up to constant factors."  This is a much stronger statement than $O(n^2)$, which only says "at most quadratic."

## Examples

### Example 1 -- Polynomial

Show that $f(n) = 3n^2 + 7n - 4$ is $\Theta(n^2)$.

**Upper bound ($O$):** For $n \geq 1$, we have $7n \leq 7n^2$ and $-4 < 0 \leq n^2$, so $f(n) \leq 3n^2 + 7n^2 + n^2 = 11n^2$.  Choose $c_2 = 11$.

**Lower bound ($\Omega$):** For $n \geq 4$, we have $7n - 4 \geq 0$, so $f(n) \geq 3n^2$.  Choose $c_1 = 3$.

Taking $n_0 = 4$, we have $3n^2 \leq f(n) \leq 11n^2$ for all $n \geq 4$, confirming $f(n) = \Theta(n^2)$.

### Example 2 -- General Polynomial Rule

Any polynomial $p(n) = a_k n^k + a_{k-1} n^{k-1} + \cdots + a_0$ with leading coefficient $a_k > 0$ satisfies $p(n) = \Theta(n^k)$.  The lower-order terms become negligible relative to $a_k n^k$ as $n$ grows.

### Example 3 -- When Theta Does Not Apply

$f(n) = n$ is $O(n^2)$ but not $\Omega(n^2)$, so $f(n) \neq \Theta(n^2)$.  The Theta notation only applies when the growth rates genuinely match.

## Proof Strategy

To prove $f(n) = \Theta(g(n))$, the cleanest approach is usually:

1. **Prove $f(n) = O(g(n))$** by finding $c_2$ and a threshold.
2. **Prove $f(n) = \Omega(g(n))$** by finding $c_1$ and a threshold.
3. **Combine** using the equivalence theorem, taking $n_0 = \max(n_{0,O}, n_{0,\Omega})$.

Alternatively, find $c_1$, $c_2$, and $n_0$ directly from the sandwich definition.

## Why Theta Matters in Practice

When we analyze an algorithm and establish a $\Theta$ bound, we know the bound is **tight** -- no better asymptotic bound exists.  Consider the following:

| Statement | Meaning |
|---|---|
| Merge sort runs in $O(n \log n)$ | At most linearithmic (could be faster) |
| Merge sort runs in $\Omega(n \log n)$ | At least linearithmic (could be slower) |
| Merge sort runs in $\Theta(n \log n)$ | Exactly linearithmic (tight) |

The $\Theta$ bound settles the question: merge sort's worst-case running time is fully characterized as $n \log n$ up to constant factors.  For a detailed comparison of all asymptotic notations, see [Growth Rate Comparison](comparison.md).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 3. MIT Press.
