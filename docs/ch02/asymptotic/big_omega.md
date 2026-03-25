# Big-Omega Notation

Big-O notation tells us that a function grows *at most* as fast as another, providing an upper bound.  In many situations we need the opposite guarantee: a function grows *at least* as fast as another.  Big-Omega notation ($\Omega$) formalizes this idea of an **asymptotic lower bound**, and it is the mirror image of Big-O.

## Formal Definition

Let $f$ and $g$ be functions from the positive integers (or positive reals) to the non-negative reals.  We say $f(n)$ is $\Omega(g(n))$ if, beyond some threshold, $f(n)$ is always at least a fixed constant multiple of $g(n)$.

!!! info "Definition -- Big-Omega"

    $f(n) = \Omega(g(n))$ if there exist constants $c > 0$ and $n_0 > 0$ such that

    $$
    f(n) \geq c \cdot g(n) \quad \text{for all } n \geq n_0
    $$

The constants mirror those in the Big-O definition:

- **$c$ (the constant multiplier)** sets the scale.  We only need $f$ to exceed *some* positive fraction of $g$, not $g$ itself.
- **$n_0$ (the threshold)** lets us ignore finitely many small input sizes.

## Relationship to Big-O

Big-Omega is the symmetric counterpart of Big-O.  The two notations are linked by a simple duality:

$$
f(n) = \Omega(g(n)) \iff g(n) = O(f(n))
$$

This means every Big-Omega statement can be restated as a Big-O statement with the roles of $f$ and $g$ swapped.  The duality is useful for converting between upper-bound and lower-bound perspectives.

When both an upper and a lower bound hold simultaneously -- that is, $f(n) = O(g(n))$ *and* $f(n) = \Omega(g(n))$ -- we write $f(n) = \Theta(g(n))$, indicating a **tight bound** (see [Big-Theta Notation](big_theta.md)).

## Intuition

Think of $\Omega(g(n))$ as a **lower envelope**: past the threshold $n_0$, the curve $c \cdot g(n)$ always sits below (or equals) the curve $f(n)$.  Saying "this algorithm runs in $\Omega(n \log n)$ time" means that no matter how clever the implementation, it must perform at least a constant multiple of $n \log n$ operations for sufficiently large inputs.

## Examples

### Example 1 -- Linear Lower Bound

Show that $f(n) = 3n + 5$ is $\Omega(n)$.

Choose $c = 3$ and $n_0 = 1$.  For all $n \geq 1$:

$$
3n + 5 \geq 3n = c \cdot n
$$

Since $5 \geq 0$, the inequality holds immediately, confirming $3n + 5 = \Omega(n)$.

### Example 2 -- Quadratic Lower Bound

Show that $f(n) = 2n^2 + 3n + 1$ is $\Omega(n^2)$.

Choose $c = 2$ and $n_0 = 1$.  For all $n \geq 1$:

$$
2n^2 + 3n + 1 \geq 2n^2 = c \cdot n^2
$$

The additional non-negative terms $3n + 1$ only increase the left side, so the inequality holds.

### Example 3 -- Proving a Non-Lower-Bound

Show that $n \neq \Omega(n^2)$.

Assume for contradiction that there exist $c > 0$ and $n_0 > 0$ with $n \geq c \cdot n^2$ for all $n \geq n_0$.  Dividing by $n$ (valid for $n > 0$) yields $1 \geq c \cdot n$, which fails for $n > 1/c$.  This contradiction shows $n \neq \Omega(n^2)$.

## Proof Strategy

To prove that $f(n) = \Omega(g(n))$:

1. **Find suitable constants.** Pick a candidate $c$ and $n_0$.
2. **Verify the inequality.** Show that $f(n) \geq c \cdot g(n)$ for every $n \geq n_0$.

To prove that $f(n) \neq \Omega(g(n))$:

1. **Assume the contrary.** Suppose constants $c > 0$ and $n_0 > 0$ exist.
2. **Derive a contradiction.** Show that the inequality $f(n) \geq c \cdot g(n)$ must eventually fail.

## Use in Algorithm Analysis

Big-Omega appears in two main contexts:

- **Lower bounds on problems.** A statement like "comparison-based sorting requires $\Omega(n \log n)$ comparisons" means *every* algorithm in that model needs at least that many operations.  This establishes a fundamental limit, not just a property of one particular algorithm.
- **Best-case analysis.** Saying an algorithm runs in $\Omega(n)$ time means that even on the most favorable input, it still takes at least linear time.

Combining Big-Omega lower bounds with Big-O upper bounds is how we establish that an algorithm is **asymptotically optimal**: if the lower bound on the problem matches the upper bound of our algorithm, no other algorithm can do asymptotically better.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 3. MIT Press.
