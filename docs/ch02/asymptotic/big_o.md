# Big-O Notation

When analyzing algorithms, we care about how running time or space usage grows as the input size increases.  Constant factors and lower-order terms depend on hardware and implementation details, so they obscure the fundamental growth behavior.  Big-O notation gives us a precise way to state that one function grows *at most as fast as* another, capturing the idea of an **asymptotic upper bound** while discarding those irrelevant details.

## Formal Definition

Let $f$ and $g$ be functions from the positive integers (or positive reals) to the non-negative reals.  We say $f(n)$ is $O(g(n))$ if, beyond some starting point, $f(n)$ never exceeds a fixed constant multiple of $g(n)$.

!!! info "Definition -- Big-O"

    $f(n) = O(g(n))$ if there exist constants $c > 0$ and $n_0 > 0$ such that

    $$
    f(n) \leq c \cdot g(n) \quad \text{for all } n \geq n_0
    $$

The two constants play distinct roles:

- **$c$ (the constant multiplier)** absorbs all constant factors.  It lets us ignore the exact coefficients in $f$ and focus on its growth shape.
- **$n_0$ (the threshold)** lets us ignore finitely many small values of $n$ where the relationship might not hold.  Asymptotic analysis only cares about behavior as $n \to \infty$.

## Intuition

Think of $O(g(n))$ as an **upper envelope**: once you move past the threshold $n_0$, the curve $c \cdot g(n)$ always sits above (or equals) the curve $f(n)$.  The exact multiplier $c$ and threshold $n_0$ do not matter -- what matters is that *some* pair $(c, n_0)$ exists.

A useful analogy: saying "this algorithm runs in $O(n^2)$ time" is like saying "the running time is *at most quadratic* for large inputs, up to a constant factor."

## Examples

### Example 1 -- Linear Function

Show that $f(n) = 3n + 5$ is $O(n)$.

Choose $c = 4$ and $n_0 = 5$.  For all $n \geq 5$:

$$
3n + 5 \leq 3n + n = 4n = c \cdot n
$$

Since $5 \leq n$ when $n \geq 5$, the inequality holds, confirming $3n + 5 = O(n)$.

### Example 2 -- Quadratic Function

Show that $f(n) = 2n^2 + 3n + 1$ is $O(n^2)$.

For $n \geq 1$, observe that $3n \leq 3n^2$ and $1 \leq n^2$, so:

$$
2n^2 + 3n + 1 \leq 2n^2 + 3n^2 + n^2 = 6n^2
$$

Choosing $c = 6$ and $n_0 = 1$ satisfies the definition.

### Example 3 -- Big-O is an Upper Bound, Not a Tight Bound

Since $n = O(n^2)$ (choose $c = 1$, $n_0 = 1$), Big-O does not claim that $f$ grows at the *same* rate as $g$ -- only that $f$ grows no faster.  For a tight bound, use $\Theta$-notation instead.

## Proof Strategy

To prove that $f(n) = O(g(n))$:

1. **Find suitable constants.** Pick a candidate $c$ and $n_0$.
2. **Verify the inequality.** Show that $f(n) \leq c \cdot g(n)$ for every $n \geq n_0$.

To prove that $f(n) \neq O(g(n))$:

1. **Assume the contrary.** Suppose constants $c > 0$ and $n_0 > 0$ exist such that $f(n) \leq c \cdot g(n)$ for all $n \geq n_0$.
2. **Derive a contradiction.** Exhibit a sequence of $n$ values that violate the inequality.

??? example "Proof that $n^2 \neq O(n)$"

    Assume for contradiction that there exist $c > 0$ and $n_0 > 0$ with $n^2 \leq c \cdot n$ for all $n \geq n_0$.  Dividing both sides by $n$ (valid for $n > 0$) gives $n \leq c$ for all $n \geq n_0$.  But choosing $n = \lceil c \rceil + 1$ contradicts this, so $n^2 \neq O(n)$.

## Common Big-O Classes

The table below lists growth rates from slowest to fastest.  Each class appears frequently in algorithm analysis.

| Big-O | Name | Example Algorithm |
|---|---|---|
| $O(1)$ | Constant | Hash table lookup |
| $O(\log n)$ | Logarithmic | Binary search |
| $O(n)$ | Linear | Linear scan |
| $O(n \log n)$ | Linearithmic | Merge sort |
| $O(n^2)$ | Quadratic | Insertion sort (worst case) |
| $O(n^3)$ | Cubic | Naive matrix multiplication |
| $O(2^n)$ | Exponential | Brute-force subset enumeration |

For a detailed discussion of these and other growth rates, see [Common Growth Rates](growth_rates.md).

## Connection to Algorithm Analysis

Big-O notation appears throughout algorithm design in two main ways:

- **Running time analysis.** We express worst-case (or average-case) running time as $T(n) = O(g(n))$, meaning the algorithm takes at most a constant multiple of $g(n)$ steps for inputs of size $n$.
- **Space analysis.** We express memory usage as $S(n) = O(g(n))$, bounding auxiliary or total space.

Because Big-O provides only an upper bound, it gives a **guarantee**: the algorithm will never perform worse than the stated bound (up to constants).  When a matching lower bound is also known, we use $\Theta$-notation (see [Big-Theta Notation](big_theta.md)).  When we need only a lower bound, we use $\Omega$-notation (see [Big-Omega Notation](big_omega.md)).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapters 3-4. MIT Press.
